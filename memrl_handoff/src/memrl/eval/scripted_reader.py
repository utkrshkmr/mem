"""Deterministic memory reader for CPU fixtures. Not a neural reader and not a result."""

from __future__ import annotations

import json
import re

from memrl.io import canonical_json


def _entries(snapshot_text: str) -> dict[str, str]:
    found = {}
    for line in snapshot_text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        found[str(payload["key"])] = str(payload["value"])
    return found


def answer_json(snapshot_text: str, prompt: str, answer_type: str) -> str:
    entries = _entries(snapshot_text)
    if answer_type == "integer" and prompt.startswith("What is the total"):
        total = sum(int(value) for value in entries.values())
        return canonical_json({"answer": total})
    match = re.search(r"value of (\S+)\?", prompt)
    if answer_type in {"integer", "unknown"} and match:
        ref = match.group(1)
        if ref not in entries:
            return canonical_json({"answer": None})
        return canonical_json({"answer": int(entries[ref])})
    match = re.search(r"Is (\S+) active\?", prompt)
    if answer_type == "member" and match:
        ref = match.group(1)
        return canonical_json({"answer": "active" if ref in entries else "inactive"})
    return canonical_json({"answer": None})
