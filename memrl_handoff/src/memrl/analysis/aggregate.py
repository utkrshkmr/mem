"""Completeness checks and source-world cluster intervals."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def validate_completeness(rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> None:
    expected = {(item["world_id"], item["question_id"], item.get("branch_index")) for item in manifest}
    seen = []
    for row in rows:
        key = (row["world_id"], row["question_id"], row.get("branch_index"))
        if key in seen:
            raise ValueError("duplicate result row")
        seen.append(key)
        if "failure_class" not in row:
            raise ValueError("missing failure class")
    missing = expected - set(seen)
    extra = set(seen) - expected
    if missing or extra:
        raise ValueError(f"incomplete rows missing={len(missing)} extra={len(extra)}")


def cluster_means(rows: list[dict[str, Any]], *, arm: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("arm") != arm:
            continue
        if row.get("infrastructure"):
            continue
        ancestor = row["ancestor_id"]
        grouped[ancestor].append(1.0 if row["success"] else 0.0)
    if not grouped:
        raise ValueError("no scored rows for arm")
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def paired_difference(left: dict[str, float], right: dict[str, float]) -> list[tuple[str, float]]:
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("paired comparison has no shared source worlds")
    if set(left) != set(right):
        raise ValueError("unmatched comparison")
    return [(key, left[key] - right[key]) for key in keys]


def bootstrap_mean_interval(pairs: list[tuple[str, float]], *, replicates: int, seed: int,
                            alpha: float = 0.05) -> dict[str, float | int]:
    if replicates < 100:
        raise ValueError("bootstrap replicates must be at least 100")
    values = [value for _key, value in pairs]
    point = sum(values) / len(values)
    rng = random.Random(seed)
    stats = []
    n = len(values)
    for _ in range(replicates):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(draw) / n)
    stats.sort()
    lo = stats[int((alpha / 2) * (replicates - 1))]
    hi = stats[int((1 - alpha / 2) * (replicates - 1))]
    return {
        "estimate": point,
        "low": lo,
        "high": hi,
        "n_worlds": n,
        "replicates": replicates,
    }


def table_cell(*, estimate: float | None, low: float | None, high: float | None,
               n_worlds: int, n_seeds: int, metric: str, contrast_id: str,
               source_sha256: str, status: str = "measured") -> dict[str, Any]:
    if status == "not_run":
        return {
            "estimate": None, "low": None, "high": None, "n_worlds": 0, "n_seeds": 0,
            "metric": metric, "contrast_id": contrast_id, "source_sha256": source_sha256,
            "status": "not_run",
        }
    return {
        "estimate": estimate, "low": low, "high": high, "n_worlds": n_worlds,
        "n_seeds": n_seeds, "metric": metric, "contrast_id": contrast_id,
        "source_sha256": source_sha256, "status": status,
    }
