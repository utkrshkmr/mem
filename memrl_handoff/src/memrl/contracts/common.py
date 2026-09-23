"""Strict scalars shared by production contracts."""

from __future__ import annotations

import math
import re

SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def strict_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; bool is invalid")
    return value


def valid_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8 text") from exc
    return value


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def sha256_hex(value: object, name: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} requires a 64-character SHA-256")
    if value == "0" * 64:
        raise ValueError(f"{name} rejects an unresolved all-zero hash")
    return value


def commit_hex(value: object, name: str) -> str:
    if type(value) is not str or COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} requires a resolved 40-character commit")
    return value
