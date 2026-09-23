"""Launch entry point. Uncertified configs exit non-zero and start no workers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from memrl.config import load_resolved, validate_for_launch
from memrl.io import load_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memrl.runtime.launch")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--gpu-uuids-file", type=Path, required=True)
    parser.add_argument("--gpu-hour-cap", type=float, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode != "split_2train_2rollout":
        print("alternative scheduling needs a measured protocol", file=sys.stderr)
        return 2
    try:
        config = load_resolved(args.config)
        lock = load_json(args.lock)
        gates = load_json(args.gates)
        manifest = load_json(args.manifest)
        if args.gpu_hour_cap > config["resources"]["project_gpu_hour_cap"]:
            raise ValueError("GPU-hour cap exceeds the project envelope")
        validate_for_launch(config, lock, manifest, gates, kind="train")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"launch refused: {exc}", file=sys.stderr)
        return 2
    print("launch certified but the worker supervisor is not started by this command yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
