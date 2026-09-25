"""One training run (Phases 4, 5): python scripts/train.py --config A.yaml [B.yaml ...] --set k=v --tag T [--resume]"""
from kmatters.hf_auth import load_hf_token
load_hf_token()

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/base.yaml"])
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--runs-root", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from kmatters import config
    from kmatters.loop import train
    cfg = config.load(a.config, a.overrides)
    run_dir = train(cfg, a.tag, resume=a.resume, runs_root=a.runs_root)
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
