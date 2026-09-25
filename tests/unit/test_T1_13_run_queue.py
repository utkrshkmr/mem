"""T1.13: run_queue dry run with 7 fake jobs on fake GPUs 0-3."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FAKE_JOB = r'''
import json, os, sys, time
job, trace, out_dir, sleep_s, rc, mode = sys.argv[1:7]
def log(ev):
    with open(trace, "a") as f:
        f.write(json.dumps({"job": job, "ev": ev, "t": time.time(), "mode": mode,
                            "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                            "tok_par": os.environ["TOKENIZERS_PARALLELISM"],
                            "spawn": os.environ["VLLM_WORKER_MULTIPROC_METHOD"]}) + "\n")
log("start"); time.sleep(float(sleep_s)); log("end")
if int(rc) == 0:
    os.makedirs(out_dir, exist_ok=True); open(os.path.join(out_dir, "DONE"), "w").close()
sys.exit(int(rc))
'''


def test_T1_13_run_queue_dry_run(tmp_path, record):
    fake = tmp_path / "fake_job.py"
    fake.write_text(FAKE_JOB)
    trace = tmp_path / "trace.jsonl"
    specs = [  # job_id, est_hours, exit code, note
        ("j_a", 1.0, 0, "partial"), ("j_b", 5.0, 0, ""), ("j_c", 3.0, 0, ""), ("j_d", 0.5, 0, ""),
        ("j_e", 4.0, 1, "always fails"), ("j_f", 2.0, 0, ""), ("j_g", 6.0, 0, "already DONE"),
    ]
    jobs = []
    for jid, est, rc, note in specs:
        out = tmp_path / "out" / jid
        base = f"{sys.executable} {fake} {jid} {trace} {out} 0.6 {rc}"
        jobs.append({"job_id": jid, "cmd": base + " fresh", "resume_cmd": base + " resume",
                     "out_dir": str(out), "est_hours": est, "gpu_need": 1})
        if note == "already DONE":
            out.mkdir(parents=True)
            (out / "DONE").touch()
        if note == "partial":
            out.mkdir(parents=True)
            (out / "ckpt_it0010").touch()
    jobs_file = tmp_path / "jobs" / "dry.jsonl"
    jobs_file.parent.mkdir()
    jobs_file.write_text("".join(json.dumps(j) + "\n" for j in jobs))

    p = subprocess.run([sys.executable, str(REPO / "scripts" / "run_queue.py"), str(jobs_file),
                        "--gpus", "0,1,2,3", "--stagger", "0.05", "--poll", "0.02", "--status-every", "0.2"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 1, p.stdout + p.stderr          # one job FAILED

    status = json.loads((tmp_path / "jobs" / "dry.status.json").read_text())
    st = status["jobs"]
    ev = [json.loads(l) for l in trace.read_text().splitlines()]

    # never more than one job per GPU at a time
    by_gpu = {}
    for e in ev:
        by_gpu.setdefault(e["gpu"], []).append(e)
    for gpu, es in by_gpu.items():
        es.sort(key=lambda e: e["t"])
        depth = 0
        for e in es:
            depth += 1 if e["ev"] == "start" else -1
            assert 0 <= depth <= 1, f"GPU {gpu} ran two jobs at once"
    assert set(by_gpu) <= {"0", "1", "2", "3"}

    # LPT order of first launches (DONE job excluded); the failed job's retry comes later
    first = list(dict.fromkeys(status["launch_order"]))
    assert first == ["j_b", "j_e", "j_c", "j_f", "j_a", "j_d"], first
    assert status["launch_order"].count("j_e") == 2

    # DONE job skipped and never started
    assert st["j_g"]["state"] == "SKIPPED_DONE" and not any(e["job"] == "j_g" for e in ev)
    # failing job: retried once in resume mode, then FAILED
    assert st["j_e"]["state"] == "FAILED" and st["j_e"]["attempts"] == 2
    assert st["j_e"]["exit_codes"] == [1, 1]
    assert [l["resume"] for l in st["j_e"]["launches"]] == [False, True]
    # partial output -> resume mode on the first launch
    assert [e["mode"] for e in ev if e["job"] == "j_a" and e["ev"] == "start"] == ["resume"]
    for jid in ("j_a", "j_b", "j_c", "j_d", "j_f"):
        assert st[jid]["state"] == "DONE" and st[jid]["attempts"] == 1
        assert (tmp_path / "jobs" / "logs" / f"{jid}.log").exists()
    assert all(e["tok_par"] == "false" and e["spawn"] == "spawn" for e in ev)
    record("T1.13", launch_order=status["launch_order"], counts=status["counts"])
