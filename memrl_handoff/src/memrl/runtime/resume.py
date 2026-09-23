"""Resume from the last complete checkpoint boundary."""

from __future__ import annotations

import json
from pathlib import Path

from memrl.io import atomic_json


def write_checkpoint(directory: str | Path, iteration: int, state: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"iteration": iteration, "complete": True, "state": state}
    path = directory / f"checkpoint-{iteration:06d}.json"
    atomic_json(path, payload)
    atomic_json(directory / "LATEST", {"iteration": iteration, "path": path.name})
    return path


def latest_complete(directory: str | Path) -> dict:
    pointer = Path(directory) / "LATEST"
    if not pointer.exists():
        raise FileNotFoundError("no complete checkpoint")
    meta = json.loads(pointer.read_text(encoding="utf-8"))
    path = Path(directory) / meta["path"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise ValueError("latest checkpoint is incomplete")
    return payload


def recover_iteration(directory: str | Path) -> int:
    """The next iteration is one past the last complete checkpoint."""
    try:
        payload = latest_complete(directory)
    except FileNotFoundError:
        return 0
    return int(payload["iteration"]) + 1


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    parser = argparse.ArgumentParser(prog="memrl.runtime.resume")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--replay-incomplete-iteration", action="store_true")
    args = parser.parse_args(argv)
    if args.checkpoint != "latest-complete":
        print("only latest-complete resume is implemented", file=sys.stderr)
        return 2
    try:
        nxt = recover_iteration(args.run)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"resume refused: {exc}", file=sys.stderr)
        return 2
    print(f"next_iteration {nxt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
