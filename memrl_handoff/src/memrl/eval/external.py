"""External benchmarks stay locked until a protocol freeze is recorded."""

from __future__ import annotations

import json
from pathlib import Path


def assert_frozen(freeze: dict | None) -> None:
    if not isinstance(freeze, dict) or freeze.get("status") != "frozen":
        raise ValueError("final external access locked")
    if freeze.get("allow_selection") is not False:
        raise ValueError("final-benchmark path in the checkpoint selector")


def load_official_fixture(path: str | Path, freeze: dict | None) -> list[dict]:
    """Load a local official-format fixture. This does not score or download."""
    assert_frozen(freeze)
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("question_id", "question", "answer"):
            if key not in row:
                raise ValueError(f"official fixture missing {key}")
        rows.append(row)
    if not rows:
        raise ValueError("official fixture is empty")
    return rows
