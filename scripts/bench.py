"""Microbenchmarks (Phase 3): writes profiles/bench_<engcfg>.json (n_sat, decode_rate_sat, prefill_rate,
train tokens/s). python scripts/bench.py [--config ...] [--set ...]"""
from kmatters.hf_auth import load_hf_token
load_hf_token()

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from kmatters import config
    from kmatters.bench import decode_curve, prefill_rate, train_rate
    from kmatters.loop import build_context, close_context, init_v0
    from kmatters.profiling import engine_label
    cfg = config.load(a.config, a.overrides)
    out = Path(a.out or Path(cfg.paths.profiles) / f"bench_{engine_label(cfg)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx = build_context(cfg, "bench", out.parent / "bench_adapters", phase="bench")
    try:
        init_v0(ctx)
        lora = ctx.engine.current
        dec = decode_curve(ctx.engine, ctx.loop, ctx.tok, lora)
        pre = prefill_rate(ctx.engine, ctx.loop, ctx.tok, lora)
        trn = train_rate(ctx.trainer, ctx.tok, cfg.train.max_tokens_per_microbatch, cfg.estimator.T_norm)
    finally:
        close_context(ctx)
    res = {"engine_cfg": engine_label(cfg), **dec, "prefill_rate": pre, "train_tokens_per_s": trn,
           "config_hash": cfg.config_hash}
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
