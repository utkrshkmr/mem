"""Write jobs/<phase>.jsonl for run_queue.py (PLAN.md §8.14).

python scripts/make_jobs.py --phase {pilot,sweep,grid_main,grid_budget,variance0,variance_trained,trained_grid} [--ckpt X]
Each line: {job_id, cmd, out_dir, est_hours, gpu_need: 1} (+ resume_cmd for training jobs).
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kmatters import config  # noqa: E402

PY = sys.executable
KS = (1, 2, 4, 8, 16)
PILOT_LRS = (1e-5, 3e-5, 1e-4)
SWEEP_SEEDS = (0, 1, 2)


def base_configs(extra=None):
    cfgs = ["configs/base.yaml"]
    if (REPO / "configs" / "locked.yaml").exists():
        cfgs.append("configs/locked.yaml")
    return cfgs + ([extra] if extra else [])


def est_iter_minutes(N: int) -> float:
    """Planning estimate (PLAN.md §2): ~3 min at N=64 (K=1), ~1.2 min at N=4 (K=16)."""
    return 1.2 + (3.0 - 1.2) * (N - 4) / 60


def train_job(tag, cfgs, overrides):
    cfg = config.load([REPO / c for c in cfgs], overrides)
    run_id = cfg.run_id(tag)
    cmd = " ".join([PY, "scripts/train.py", "--config", *cfgs, *sum((["--set", o] for o in overrides), []),
                    "--tag", tag])
    return {"job_id": run_id, "cmd": cmd, "resume_cmd": cmd + " --resume",
            "out_dir": f"{cfg.paths.runs}/{run_id}", "gpu_need": 1,
            "est_hours": round(cfg.train.iterations * est_iter_minutes(cfg.train.N) / 60 * 1.15, 2)}


def grid_jobs(tag, grid, ckpt, shards=4, est_total_hours=5.0):
    return [{"job_id": f"{tag}_shard{i}of{shards}",
             "cmd": " ".join([PY, "scripts/profile_costs.py", "--grid", grid, "--ckpt", shlex.quote(ckpt),
                              "--shard", f"{i}/{shards}", "--out", f"profiles/{tag}", "--config", *base_configs()]),
             "out_dir": f"profiles/{tag}/shard{i}of{shards}", "gpu_need": 1,
             "est_hours": round(est_total_hours / shards, 2)} for i in range(shards)]


def variance_job(tag, ckpt, ckpt_it, L, B, vseed, est_hours=0.6):
    out = f"variance/{tag}"
    return {"job_id": tag, "cmd": " ".join([PY, "scripts/measure_variance.py", "--ckpt", shlex.quote(ckpt),
                                            "--ckpt-it", str(ckpt_it), "--L", str(L), "--B", str(B),
                                            "--vseed", str(vseed), "--out", out, "--config", *base_configs()]),
            "out_dir": out, "gpu_need": 1, "est_hours": est_hours}


def sweep_ckpt(K, seed, it):
    N = 64 // K
    hits = sorted(REPO.glob(f"runs/sweep_K{K}_N{N}_s{seed}_*/ckpt/it{it:04d}/adapter"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one checkpoint for K={K} seed={seed} it={it}, found {hits}")
    return str(hits[0].relative_to(REPO))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["pilot", "sweep", "grid_main", "grid_budget", "variance0",
                                                       "variance_trained", "trained_grid"])
    ap.add_argument("--ckpt", default="base")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    jobs = []
    if a.phase == "pilot":
        for K in (1, 16):
            for lr in PILOT_LRS:
                jobs.append(train_job("pilot", base_configs("configs/pilot.yaml"), [f"train.K={K}", f"train.lr={lr}"]))
    elif a.phase == "sweep":
        for K in KS:
            for s in SWEEP_SEEDS:
                jobs.append(train_job("sweep", base_configs("configs/sweep_main.yaml"),
                                      [f"train.K={K}", f"train.seed={s}"]))
    elif a.phase == "grid_main":
        jobs = grid_jobs(f"grid_main_{Path(a.ckpt).name if a.ckpt != 'base' else 'base'}", "configs/grid_main.yaml",
                         a.ckpt, est_total_hours=10.0)
    elif a.phase == "grid_budget":
        jobs = grid_jobs("grid_budget_base", "configs/grid_budget.yaml", a.ckpt, est_total_hours=2.5)
    elif a.phase == "trained_grid":
        jobs = grid_jobs("grid_trained", "configs/grid_trained.yaml", a.ckpt, est_total_hours=4.0)
    elif a.phase == "variance0":
        v = yaml.safe_load((REPO / "configs/variance_conditions.yaml").read_text())["variance0"]
        for c in v["conditions"]:
            for s in v["vseeds"]:
                jobs.append(variance_job(f"vm0_L{c['L']}_B{c['B']}_v{s}", "base", 0, c["L"], c["B"], s))
    elif a.phase == "variance_trained":
        vc = yaml.safe_load((REPO / "configs/variance_conditions.yaml").read_text())
        I_max = config.load([REPO / c for c in base_configs("configs/sweep_main.yaml")]).train.iterations
        ref = sweep_ckpt(vc["reference"]["K"], vc["reference"]["seed"], I_max)
        va = vc["variance_trained"]["a"]
        for K in va["Ks"]:
            for it in va["its"]:
                it = I_max if it == "final" else int(it)
                jobs.append(variance_job(f"vt_K{K}_s{va['seed']}_it{it:04d}_L{va['L']}_B{va['B']}_v{va['vseed']}",
                                         sweep_ckpt(K, va["seed"], it), it, va["L"], va["B"], va["vseed"]))
        vb = vc["variance_trained"]["b"]
        for c in vb["conditions"]:
            jobs.append(variance_job(f"vt_ref_L{c['L']}_B{c['B']}_v{vb['vseed']}", ref, I_max, c["L"], c["B"],
                                     vb["vseed"]))
        c = vc["variance_trained"]["c"]
        jobs.append(variance_job(f"vt_ref_L{c['L']}_B{c['B']}_v{c['vseed']}", ref, I_max, c["L"], c["B"], c["vseed"]))
    for j in jobs:
        j["cwd"] = str(REPO)
    out = Path(a.out or REPO / "jobs" / f"{a.phase}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"wrote {len(jobs)} jobs to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
