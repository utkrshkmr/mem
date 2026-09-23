"""API probe entry point. GPU execution is opt-in and writes only measured fields."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memrl.runtime.probe_api")
    parser.add_argument("--config", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print("probe_api refused: pass --execute after hardware inventory; no mock parity is reported", file=sys.stderr)
        return 2
    from memrl.runtime.gpu_probe import execute
    return execute(args.output, args.devices)


if __name__ == "__main__":
    raise SystemExit(main())
