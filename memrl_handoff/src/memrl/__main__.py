"""CLI router. Missing production inputs fail closed."""

from __future__ import annotations

import argparse
import sys

from memrl import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memrl")
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("config")
    sub.add_parser("data")
    sub.add_parser("hardware")
    sub.add_parser("train")
    sub.add_parser("eval")
    sub.add_parser("analyze")
    args, rest = parser.parse_known_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "train":
        print("training launch is python -m memrl.runtime.launch and refuses an uncertified config", file=sys.stderr)
        return 2
    if args.command == "hardware":
        from memrl.runtime.preflight import main as preflight_main
        return preflight_main(rest)
    print(f"unknown or incomplete command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
