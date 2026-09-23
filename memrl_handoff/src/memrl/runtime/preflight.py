"""Preflight. Does not certify a launch by itself."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from memrl.config import ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memrl.runtime.preflight")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    blockers = []
    if args.config is None or not args.config.is_file():
        blockers.append("resolved config path is missing")
    hardware = ROOT / "reports" / "hardware.json"
    if not hardware.is_file():
        blockers.append("reports/hardware.json is missing; run scripts/probe_hardware.py")
    launch = ROOT / "src" / "memrl" / "runtime" / "launch.py"
    if not launch.is_file():
        blockers.append("production launcher is missing")
    if blockers:
        print("\n".join(blockers), file=sys.stderr)
        return 2
    print("preflight structural checks passed; launch certification is separate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
