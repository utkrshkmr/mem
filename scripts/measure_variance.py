"""Variance components at a checkpoint (Phases 3, 6).

python scripts/measure_variance.py --ckpt <adapter_dir|base> --ckpt-it 0 --L 16 --B 512 --vseed 0 --out variance/<tag>
"""
from kmatters.hf_auth import load_hf_token
load_hf_token()

import argparse
import os
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="adapter directory, or 'base' for adapter v0")
    ap.add_argument("--ckpt-it", type=int, default=0, help="iterations completed at the checkpoint (0 = base)")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--vseed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--N0", type=int, default=None)
    ap.add_argument("--K0", type=int, default=None)
    ap.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    a = ap.parse_args(argv)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from kmatters import config
    from kmatters.logging_utils import setup_logging
    from kmatters.loop import build_context, close_context, init_v0, load_adapter
    from kmatters.variance import measure
    out = Path(a.out)
    if (out / "DONE").exists():
        return 0
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "variance.log")
    cfg = config.load(a.config, a.overrides + [f"env.L={a.L}", f"env.B={a.B}"])
    config.save_resolved(cfg, out)
    ctx = build_context(cfg, f"var_{out.name}", out / "adapters", split="variance", key0=a.vseed,
                        executor="async_ready", phase="variance")
    try:
        if a.ckpt == "base":
            init_v0(ctx)
        else:
            load_adapter(ctx, a.ckpt, version=1)
        measure(ctx, a.ckpt, a.ckpt_it, out, a.N0 or cfg.variance.N0, a.K0 or cfg.variance.K0)
    finally:
        close_context(ctx)
    (out / "DONE").touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
