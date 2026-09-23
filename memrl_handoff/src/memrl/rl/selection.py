"""Checkpoint selection stays on development data."""

from __future__ import annotations

from memrl.config import assert_not_final_selector


def select_checkpoint(candidates: list[dict], *, split: str) -> dict:
    assert_not_final_selector(" ".join(str(row.get("source", "")) for row in candidates), split)
    if not candidates:
        raise ValueError("no development checkpoints")
    return min(
        candidates,
        key=lambda row: (-float(row["dev_score"]), int(row["iteration"])),
    )
