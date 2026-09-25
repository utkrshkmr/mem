"""Cost grids (Phases 3, 6): one shard of a grid, every cell through run_iteration(lr=0).

python scripts/profile_costs.py --grid configs/grid_main.yaml --ckpt base --shard 0/4 --out profiles/grid_main_base
python scripts/profile_costs.py --tune-engine --out profiles/tune_engine      (Phase 3 engine tuning)
"""
from kmatters.hf_auth import load_hf_token
load_hf_token()

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

TUNE_CELLS = [(8, 8), (32, 4), (128, 4)]
TUNE_SEQS = [256, 512]
TUNE_TOKENS = [8192, 16384]


def _cell_list(spec) -> list[tuple[int, int]]:
    if "list" in spec:
        return [tuple(map(int, c)) for c in spec["list"]]
    return [(N, K) for N in spec["N"] for K in spec["K"] if N * K <= spec.get("max_NK", 10**9)]


def expand_grid(path) -> list[dict]:
    g = yaml.safe_load(Path(path).read_text())
    cells = []
    for name, cond in g["conditions"].items():
        for N, K in _cell_list(cond.get("cells", g["cells"])):
            cells.append({"condition": name, "N": N, "K": K, "L": int(cond["L"]), "B": int(cond["B"]),
                          "executor": cond["executor"]})
    return cells


def est_cost(c) -> float:
    """Relative cost proxy for LPT sharding: writer calls weighted by memory budget."""
    return c["N"] * (c["L"] + c["K"] * 3) * (c["B"] / 512) ** 0.5 + 20


def shard(cells, i, n) -> list[dict]:
    loads, out = [0.0] * n, [[] for _ in range(n)]
    for c in sorted(cells, key=lambda c: (-est_cost(c), c["condition"], c["N"], c["K"])):
        j = int(np.argmin(loads))
        loads[j] += est_cost(c)
        out[j].append(c)
    mine = out[i]
    rng = np.random.default_rng([1234, i, n])          # seeded random order within the shard
    return [mine[k] for k in rng.permutation(len(mine))]


def _done_keys(path: Path, reps: int, warmup: int) -> set:
    if not path.exists():
        return set()
    counts = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        key = (r["executor"], r["N"], r["K"], r["L"], r["B"])
        counts[key] = counts.get(key, 0) + 1
    return {k for k, v in counts.items() if v >= reps + warmup}


def run_shard(a):
    from kmatters import config
    from kmatters.logging_utils import setup_logging
    from kmatters.loop import build_context, close_context, init_v0, load_adapter
    from kmatters.profiling import engine_label, run_cell
    i, n = map(int, a.shard.split("/"))
    out = Path(a.out) / f"shard{i}of{n}"
    if (out / "DONE").exists():
        return 0
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "profile.log")
    cfg = config.load(a.config, a.overrides)
    config.save_resolved(cfg, out)
    reps = a.reps if a.reps is not None else cfg.profile.reps
    warmup = cfg.profile.warmup
    cells = shard(expand_grid(a.grid), i, n)
    if a.only:
        wanted = {tuple(map(int, x.split(","))) for x in a.only.split(";")}
        cells = [c for c in cells if (c["N"], c["K"]) in wanted]
    n_sat = None
    bench = Path(cfg.paths.profiles) / f"bench_{engine_label(cfg)}.json"
    if bench.exists():
        n_sat = int(json.loads(bench.read_text())["n_sat"])
    done = _done_keys(out / "cells.jsonl", reps, warmup)
    ctx = build_context(cfg, f"profile_{Path(a.out).name}_{i}", out / "adapters", split="profile", key0=0,
                        phase="profile")
    try:
        if a.ckpt == "base":
            init_v0(ctx)
        else:
            load_adapter(ctx, a.ckpt, version=1)
        for c in cells:
            key = (c["executor"], c["N"], c["K"], c["L"], c["B"])
            if key in done:
                continue
            rows = run_cell(ctx, c, reps, warmup, a.ckpt, n_sat=n_sat, intervals_dir=out / "intervals")
            with open(out / "cells.jsonl", "a") as f:        # a cell's rows land together or not at all
                for r in rows:
                    f.write(json.dumps({"condition": c["condition"], **r}) + "\n")
    finally:
        close_context(ctx)
    (out / "DONE").touch()
    return 0


def tune_engine(a):
    """Run the tuning cells once per engine configuration (fresh process each), pick the lowest
    summed median T_iter, and write the winner into configs/locked.yaml."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    grid = out / "tune_grid.yaml"
    grid.write_text(yaml.safe_dump({"cells": {"list": [list(c) for c in TUNE_CELLS]},
                                    "conditions": {"A": {"executor": "async_ready", "L": 16, "B": 512}}}))
    results = {}
    for seqs, toks in itertools.product(TUNE_SEQS, TUNE_TOKENS):
        tag = f"s{seqs}_b{toks}"
        cmd = [sys.executable, __file__, "--grid", str(grid), "--ckpt", a.ckpt, "--shard", "0/1",
               "--out", str(out / tag), "--config", *a.config,
               "--set", f"vllm.max_num_seqs={seqs}", "--set", f"vllm.max_num_batched_tokens={toks}",
               *sum((["--set", o] for o in a.overrides), [])]
        subprocess.run(cmd, check=True)
        rows = [json.loads(l) for l in (out / tag / "shard0of1" / "cells.jsonl").read_text().splitlines()]
        med = {}
        for (N, K) in TUNE_CELLS:
            med[f"{N},{K}"] = float(np.median([r["T_iter"] for r in rows if (r["N"], r["K"]) == (N, K)
                                               and not r["warmup"]]))
        results[tag] = {"max_num_seqs": seqs, "max_num_batched_tokens": toks, "median_T_iter": med,
                        "sum": sum(med.values())}
    best = min(results.values(), key=lambda r: r["sum"])
    (out / "tune_results.json").write_text(json.dumps({"results": results, "best": best}, indent=2))
    locked_path = Path("configs/locked.yaml")
    locked = yaml.safe_load(locked_path.read_text()) if locked_path.exists() else {}
    locked = locked or {}
    locked.setdefault("vllm", {}).update(max_num_seqs=best["max_num_seqs"],
                                         max_num_batched_tokens=best["max_num_batched_tokens"])
    locked_path.write_text(yaml.safe_dump(locked, sort_keys=True))
    print(json.dumps(best))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid")
    ap.add_argument("--ckpt", default="base")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--only", default=None, help="restrict to cells 'N,K;N,K'")
    ap.add_argument("--tune-engine", action="store_true")
    ap.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    a = ap.parse_args(argv)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if a.tune_engine:
        return tune_engine(a)
    if not a.grid:
        ap.error("--grid is required unless --tune-engine")
    return run_shard(a)


if __name__ == "__main__":
    sys.exit(main())
