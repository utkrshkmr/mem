"""Evaluation entry point. It does not score with a stand-in model."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memrl.eval.run")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.parse_args(argv)
    print("evaluation refused: no frozen reader job has been certified for this manifest", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
