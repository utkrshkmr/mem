"""Run a jobs/<phase>.jsonl file on the GPUs, one job per GPU (PLAN.md §8.14).

Each line: {job_id, cmd, out_dir, est_hours, gpu_need: 1} plus optional `resume_cmd` and `env`.
  * Longest-processing-time-first order; starts staggered by --stagger seconds.
  * A job whose <out_dir>/DONE exists is skipped.
  * A job with partial output (out_dir non-empty, no DONE) starts in resume mode.
  * On failure (non-zero exit, or exit 0 without DONE) it is retried once in resume mode,
    then marked FAILED; the other jobs continue.
  * stdout/stderr -> <jobs dir>/logs/<job_id>.log; status -> <jobs dir>/<phase>.status.json.
Resume mode runs `resume_cmd` when the job has one, otherwise `cmd` again.

Launch inside tmux:  tmux new -d -s kq "python scripts/run_queue.py jobs/sweep.jsonl --gpus 0,1,2,3"
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX_ATTEMPTS = 2   # first run + one retry


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def load_jobs(path: Path) -> list[dict]:
    jobs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ids = [j["job_id"] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate job_id in " + str(path))
    for j in jobs:
        if j.get("gpu_need", 1) != 1:
            raise ValueError(f"{j['job_id']}: only gpu_need == 1 is supported")
    # LPT: longest first; ties keep file order (sorted is stable)
    return sorted(jobs, key=lambda j: -float(j.get("est_hours", 0.0)))


def has_partial(out_dir: Path) -> bool:
    return out_dir.is_dir() and any(p.name != "DONE" for p in out_dir.iterdir())


def job_env(gpu: str, n_gpus: int, extra: dict | None, distinct_ports: bool) -> dict:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 4) // max(4, n_gpus))),
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HF_HUB_OFFLINE": "1",
    })
    if distinct_ports:   # PLAN.md §13 R7
        g = int(gpu)
        env.update({"VLLM_PORT": str(29500 + 10 * g), "MASTER_PORT": str(29500 + 10 * g + 1),
                    "VLLM_RPC_BASE_PATH": f"/tmp/vllm_rpc_gpu{g}"})
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


class Queue:
    def __init__(self, jobs_file: Path, gpus: list[str], stagger: float, status_every: float,
                 poll: float, distinct_ports: bool):
        self.jobs_file = jobs_file
        self.jobs = load_jobs(jobs_file)
        self.gpus = gpus
        self.stagger = stagger
        self.status_every = status_every
        self.poll = poll
        self.distinct_ports = distinct_ports
        self.log_dir = jobs_file.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = jobs_file.parent / f"{jobs_file.stem}.status.json"
        self.state = {j["job_id"]: {"state": "PENDING", "gpu": None, "attempts": 0, "launches": [],
                                    "exit_codes": [], "est_hours": j.get("est_hours")}
                      for j in self.jobs}
        self.running: dict[str, tuple[dict, subprocess.Popen, object]] = {}   # gpu -> (job, proc, log)
        self.pending: list[tuple[dict, bool]] = []                           # (job, resume)
        self.launch_order: list[str] = []
        for j in self.jobs:
            if (Path(j["out_dir"]) / "DONE").exists():
                self.state[j["job_id"]]["state"] = "SKIPPED_DONE"
            else:
                self.pending.append((j, False))
        self._last_launch = -1e18
        self._last_status = -1e18

    # ------------------------------------------------------------------ status
    def write_status(self):
        counts: dict[str, int] = {}
        for s in self.state.values():
            counts[s["state"]] = counts.get(s["state"], 0) + 1
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"updated_at": now_iso(), "jobs_file": str(self.jobs_file),
                                   "gpus": self.gpus, "counts": counts,
                                   "launch_order": self.launch_order, "jobs": self.state},
                                  indent=2))
        tmp.replace(self.status_path)
        self._last_status = time.monotonic()

    # ------------------------------------------------------------------ launch/reap
    def launch(self, gpu: str, job: dict, resume: bool):
        jid = job["job_id"]
        out_dir = Path(job["out_dir"])
        resume = resume or has_partial(out_dir)
        cmd = job.get("resume_cmd", job["cmd"]) if resume else job["cmd"]
        log = open(self.log_dir / f"{jid}.log", "a")
        log.write(f"\n===== {now_iso()} launch on GPU {gpu} (resume={resume}) =====\n{cmd}\n")
        log.flush()
        proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                env=job_env(gpu, len(self.gpus), job.get("env"), self.distinct_ports),
                                start_new_session=True)
        st = self.state[jid]
        st.update(state="RUNNING", gpu=gpu, attempts=st["attempts"] + 1)
        st["launches"].append({"at": now_iso(), "gpu": gpu, "resume": resume, "pid": proc.pid})
        self.launch_order.append(jid)
        self.running[gpu] = (job, proc, log)
        self._last_launch = time.monotonic()

    def reap(self):
        for gpu in list(self.running):
            job, proc, log = self.running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del self.running[gpu]
            st = self.state[job["job_id"]]
            st["exit_codes"].append(rc)
            done = (Path(job["out_dir"]) / "DONE").exists()
            if rc == 0 and done:
                st["state"] = "DONE"
            elif st["attempts"] < MAX_ATTEMPTS:
                st["state"] = "RETRY_PENDING"
                self.pending.insert(0, (job, True))
            else:
                st["state"] = "FAILED"

    def free_gpus(self):
        return [g for g in self.gpus if g not in self.running]

    def run(self):
        self.write_status()
        while self.pending or self.running:
            self.reap()
            free = self.free_gpus()
            if self.pending and free and time.monotonic() - self._last_launch >= self.stagger:
                job, resume = self.pending.pop(0)
                self.launch(free[0], job, resume)
                continue          # re-check immediately; the stagger gates the next launch
            if time.monotonic() - self._last_status >= self.status_every:
                self.write_status()
            time.sleep(self.poll)
        self.write_status()
        return sum(s["state"] == "FAILED" for s in self.state.values())

    def terminate_all(self):
        for job, proc, log in self.running.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for job, proc, log in self.running.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            self.state[job["job_id"]]["state"] = "INTERRUPTED"
            log.close()
        self.running.clear()
        self.write_status()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_file", type=Path)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--stagger", type=float, default=45.0, help="seconds between launches")
    ap.add_argument("--status-every", type=float, default=60.0)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--distinct-ports", action="store_true", help="PLAN.md §13 R7")
    a = ap.parse_args(argv)
    q = Queue(a.jobs_file, [g.strip() for g in a.gpus.split(",") if g.strip()], a.stagger,
              a.status_every, a.poll, a.distinct_ports)

    def on_signal(signum, frame):
        q.terminate_all()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    n_failed = q.run()
    print(json.dumps({s: sum(v["state"] == s for v in q.state.values())
                      for s in sorted({v["state"] for v in q.state.values()})}))
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
