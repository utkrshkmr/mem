"""T2 GPU tests that need their own processes (Gate G1): T2.7, T2.12, T2.13, T2.14, T2.16.
Runs after test_T2a_integration.py has released GPU 0."""
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu
REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
HELPERS = Path(__file__).resolve().parent / "helpers"
TESTS_RUNS = REPO / "runs" / "_tests"
TINY = ["env.L=4", "train.K=2", "train.F=16", "eval.small.n_prefixes=8", "eval.small.futures_per_prefix=2",
        "eval.large.n_prefixes=8", "eval.large.futures_per_prefix=2"]


def _env(gpu, **extra):
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
    env.update(extra)
    return env


def _wait_gpu_free(gpu, timeout=120):
    """Wait until no process other than this pytest process (which keeps a CUDA context after
    the in-process module) is on the GPU."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.split()
        others = [p for p in out if p.strip() and int(p) != os.getpid()]
        if not others:
            return
        time.sleep(2)
    raise TimeoutError(f"GPU {gpu} still busy: {others}")


# --------------------------------------------------------------------------- T2.7
def test_T2_7_executor_equivalence(record):
    out_dir = TESTS_RUNS / "T2_7"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    _wait_gpu_free(0)
    pa = subprocess.run([PY, str(HELPERS / "equivalence.py"), "a", str(out_dir / "a.json"), str(out_dir / "ad_a")],
                        env=_env(0), capture_output=True, text=True, timeout=1800, cwd=REPO)
    reason = None
    if pa.returncode == 0:
        a = json.loads((out_dir / "a.json").read_text())
        if a["identical"]:
            record("T2.7", variant="a", **a)
            return
        reason = f"batch-invariant mode ran but only {a['n_identical']}/{a['n_calls']} completions were identical"
    else:
        tail = (pa.stderr or "")[-1500:]
        reason = f"batch-invariant mode failed to run with LoRA (exit {pa.returncode}): {tail}"
    (out_dir / "a_failure.txt").write_text(reason + "\n\n" + (pa.stderr or "")[-20000:])
    _wait_gpu_free(0)
    pb = subprocess.run([PY, str(HELPERS / "equivalence.py"), "b", str(out_dir / "b.json"), str(out_dir / "ad_b")],
                        env=_env(0), capture_output=True, text=True, timeout=3600, cwd=REPO)
    assert pb.returncode == 0, pb.stderr[-3000:]
    b = json.loads((out_dir / "b.json").read_text())
    record("T2.7", variant="b", reason_a_unavailable=reason[:500], **b)
    assert b["len_rel_diff"] < 0.03 and b["reward_diff_over_pooled_se"] < 2.0, b


# --------------------------------------------------------------------------- T2.8 / T2.9 (fp32)
@pytest.fixture(scope="module")
def fp32_checks():
    out_dir = TESTS_RUNS / "T2_8_9_fp32"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    _wait_gpu_free(0)
    p = subprocess.run([PY, str(HELPERS / "fp32_grad_checks.py"), str(out_dir / "res.json"), str(out_dir / "ad")],
                       env=_env(0), capture_output=True, text=True, timeout=3600, cwd=REPO)
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads((out_dir / "res.json").read_text())


def test_T2_8_microbatch_invariance_fp32(fp32_checks, record):
    r = fp32_checks["T2.8"]
    record("T2.8", dtype="float32", **r)
    assert r["cosine"] >= 0.9999 and r["rel_l2"] <= 1e-2, r


def test_T2_9_variance_matches_training_fp32(fp32_checks, record):
    r = fp32_checks["T2.9"]
    record("T2.9", dtype="float32", **r)
    assert r["grad_norm"] > 0
    assert r["cosine"] >= 0.9999 and r["rel_l2"] <= 1e-2, r


# --------------------------------------------------------------------------- T2.12
def _metrics(run_dir):
    p = run_dir / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _rollout_keys(run_dir, it):
    with gzip.open(run_dir / "rollouts" / f"it{it:04d}.jsonl.gz", "rt") as f:
        return [(tuple(l["key"]), [(tuple(fu["key"]), fu["question"], fu["gold"]) for fu in l["futures"]])
                for l in map(json.loads, f)]


@pytest.mark.timeout(3600)
def test_T2_12_resume(record):
    root = TESTS_RUNS / "T2_12"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    over = TINY + ["train.K=2", "train.F=16", "train.ckpt_every=5", "train.iterations=14", "train.eval_every=100",
                   "train.seed=12"]
    cmd = [PY, "scripts/train.py", "--config", "configs/base.yaml", *sum((["--set", o] for o in over), []),
           "--tag", "t212", "--runs-root", str(root)]
    _wait_gpu_free(0)
    log1 = open(root / "attempt1.log", "w")
    p = subprocess.Popen(cmd, cwd=REPO, env=_env(0), stdout=log1, stderr=subprocess.STDOUT, start_new_session=True)
    run_dir = None
    t0 = time.time()
    while True:
        if p.poll() is not None:
            pytest.fail(f"first attempt exited early ({p.returncode}); see {root / 'attempt1.log'}")
        dirs = [d for d in root.iterdir() if d.is_dir()]
        run_dir = dirs[0] if dirs else None
        if run_dir is not None and len(_metrics(run_dir)) >= 11:     # iterations 0..10 done: 11 is running
            break
        if time.time() - t0 > 1800:
            os.killpg(p.pid, signal.SIGKILL)
            pytest.fail("first attempt too slow")
        time.sleep(0.2)
    time.sleep(1.0)
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    log1.close()
    n_at_kill = len(_metrics(run_dir))
    assert n_at_kill == 11, f"kill did not land during iteration index 11 ({n_at_kill} metrics lines)"
    keys10_first = _rollout_keys(run_dir, 10)
    assert (run_dir / "ckpt" / "it0010" / "COMPLETE").exists()
    _wait_gpu_free(0)

    with open(root / "attempt2.log", "w") as log2:
        p2 = subprocess.run(cmd + ["--resume"], cwd=REPO, env=_env(0), stdout=log2, stderr=subprocess.STDOUT,
                            timeout=1800)
    assert p2.returncode == 0, f"resume failed; see {root / 'attempt2.log'}"
    resumes = [json.loads(l) for l in (run_dir / "resumes.jsonl").read_text().splitlines()]
    assert len(resumes) == 1 and resumes[0]["from_ckpt"] == 10 and resumes[0]["opt_step"] == 10, resumes
    m = _metrics(run_dir)
    assert [r["it"] for r in m] == list(range(14)), [r["it"] for r in m]
    assert m[-1]["opt_step"] == 14 and m[10]["opt_step"] == 11
    assert _rollout_keys(run_dir, 10) == keys10_first            # data of iteration 10 identical
    # iteration 11's data after the restart equals the data keys (seed, 11, i, k) define
    from kmatters.env.ledger import make_future, make_prefix
    k11 = _rollout_keys(run_dir, 11)
    expect = []
    for i in range(8):
        pr = make_prefix("train", (12, 11, i), 4, 3)
        expect.append((tuple(pr.key), [(tuple(f.key), f.question, f.gold)
                                       for f in (make_future("train", pr, k, 2, 3) for k in range(2))]))
    assert k11 == expect
    assert (run_dir / "DONE").exists()
    record("T2.12", run_dir=str(run_dir.relative_to(REPO)), n_lines_at_kill=n_at_kill, resumes=resumes,
           final_opt_step=m[-1]["opt_step"])


# --------------------------------------------------------------------------- T2.13
@pytest.mark.timeout(3600)
def test_T2_13_four_gpu_concurrency(record):
    root = TESTS_RUNS / "T2_13"
    jobs_dir = REPO / "jobs" / "_tests"
    shutil.rmtree(root, ignore_errors=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    for f in jobs_dir.glob("t2_13*"):
        f.unlink()
    shutil.rmtree(jobs_dir / "logs", ignore_errors=True)
    from kmatters import config
    jobs = []
    for s in range(4):
        over = TINY + ["train.iterations=10", "train.ckpt_every=5", "train.eval_every=100", f"train.seed={20 + s}"]
        cfg = config.load([REPO / "configs/base.yaml"], over)
        run_id = cfg.run_id("t213")
        cmd = " ".join([PY, "scripts/train.py", "--config", "configs/base.yaml",
                        *sum((["--set", o] for o in over), []), "--tag", "t213", "--runs-root", str(root)])
        jobs.append({"job_id": run_id, "cmd": cmd, "resume_cmd": cmd + " --resume", "cwd": str(REPO),
                     "out_dir": str(root / run_id), "est_hours": 0.1, "gpu_need": 1})
    jf = jobs_dir / "t2_13.jsonl"
    jf.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    for g in range(4):
        _wait_gpu_free(g)
    p = subprocess.run([PY, "scripts/run_queue.py", str(jf), "--gpus", "0,1,2,3", "--stagger", "45",
                        "--poll", "1", "--status-every", "10"], cwd=REPO, capture_output=True, text=True, timeout=3000)
    status = json.loads((jobs_dir / "t2_13.status.json").read_text())
    bad_words = ("Address already in use", "EADDRINUSE", "zmq.error", "ZMQError", "port is already")
    log_hits = {}
    for j in jobs:
        txt = (jobs_dir / "logs" / f"{j['job_id']}.log").read_text(errors="replace")
        hits = [w for w in bad_words if w in txt]
        if hits:
            log_hits[j["job_id"]] = hits
    gpus = sorted({st["launches"][0]["gpu"] for st in status["jobs"].values()})
    record("T2.13", counts=status["counts"], gpus=gpus, port_ipc_errors=log_hits,
           exit_codes={k: v["exit_codes"] for k, v in status["jobs"].items()})
    assert p.returncode == 0, p.stdout + p.stderr[-2000:]
    assert all(v["state"] == "DONE" and v["exit_codes"] == [0] for v in status["jobs"].values()), status["jobs"]
    assert gpus == ["0", "1", "2", "3"] and not log_hits


# --------------------------------------------------------------------------- T2.14
def test_T2_14_nvml_mapping(record):
    out_dir = TESTS_RUNS / "T2_14"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    _wait_gpu_free(2)
    p = subprocess.run([PY, str(HELPERS / "nvml_map.py"), str(out_dir / "nvml.json"), str(out_dir / "ad")],
                       env=_env(2), capture_output=True, text=True, timeout=1800, cwd=REPO)
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads((out_dir / "nvml.json").read_text())
    phys = subprocess.run(["nvidia-smi", "-i", "2", "--query-gpu=uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    record("T2.14", sampler_uuid=res["uuid"], physical_gpu2_uuid=phys, idle_power=res["idle"]["power"],
           rollout_power=res["rollout"]["power"], rollout_s=res["rollout_s"])
    assert res["uuid"] == phys
    assert res["rollout"]["power"] > res["idle"]["power"] + 50, res


# --------------------------------------------------------------------------- T2.16
def test_T2_16_no_token_leakage(record):
    from kmatters.hf_auth import load_hf_token
    tok = load_hf_token()
    targets = [d for d in ("runs", "profiles", "reports", "jobs", "variance") if (REPO / d).exists()]
    with tempfile.TemporaryDirectory() as d:
        pat = Path(d) / "pat"
        pat.touch(mode=0o600)
        pat.write_text(tok + "\n")
        p = subprocess.run(["grep", "-rlF", "-f", str(pat), *targets], cwd=REPO, capture_output=True, text=True)
    record("T2.16", searched=targets, found_in=p.stdout.split() if p.returncode == 0 else [])
    assert p.returncode == 1, f"token found in: {p.stdout.split()}" if p.returncode == 0 else p.stderr
