"""Profile entry point. It never invents timings."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memrl.runtime.profile")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--warmup-iterations", type=int, required=True)
    parser.add_argument("--measure-iterations", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.parse_args(argv)
    print(
        "profile refused: no measured iteration trace is available and timings will not be synthesized",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
