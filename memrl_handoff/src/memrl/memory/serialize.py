"""Whole-store canonical serialization. Visible fields are id, key, and value."""

from __future__ import annotations

import json

from memrl.contracts.common import strict_integer, valid_text


class Entry:
    __slots__ = ("slot_id", "key", "value")

    def __init__(self, slot_id: int, key: str, value: str) -> None:
        strict_integer(slot_id, "slot_id", minimum=1)
        valid_text(key, "key")
        valid_text(value, "value")
        self.slot_id = slot_id
        self.key = key
        self.value = value

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is Entry
            and other.slot_id == self.slot_id
            and other.key == self.key
            and other.value == self.value
        )

    def __hash__(self) -> int:
        return hash((self.slot_id, self.key, self.value))


def canonical_entry(entry: Entry) -> str:
    if type(entry) is not Entry:
        raise ValueError("expected Entry")
    return json.dumps(
        {"id": entry.slot_id, "key": entry.key, "value": entry.value},
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ) + "\n"


def serialize_entries(entries: tuple[Entry, ...]) -> str:
    if len({entry.slot_id for entry in entries}) != len(entries):
        raise ValueError("duplicate memory slot ID")
    ordered = tuple(sorted(entries, key=lambda entry: entry.slot_id))
    return "".join(canonical_entry(entry) for entry in ordered)
