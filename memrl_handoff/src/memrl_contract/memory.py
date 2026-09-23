"""Transactional bounded memory with exact canonical serialization.

The v3 budget charges the COMPLETE concatenated canonical store, including all
newlines and visible IDs/keys/values. Per-record additive counts are diagnostic
only: tokenizer boundary effects can make their sum smaller OR larger than the
whole-string count. Production must inject the PINNED reference tokenizer.
ByteCounter is only a deliberately different, dependency-free toy counter; its
units must never be reported as model tokens or paper capacities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol

from .contracts import sha256_text, strict_integer, valid_text


class TextCounter(Protocol):
    """A stateless, deterministic text counter safe to share across clones."""

    identity: str
    unit: str

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class ByteCounter:
    identity: str = "toy-utf8-bytes-v1"
    unit: str = "utf8_bytes_TOY_ONLY"

    def count(self, text: str) -> int:
        return len(text.encode("utf-8", errors="strict"))


@dataclass(frozen=True, slots=True)
class MemoryLimits:
    capacity: int
    max_entries: int
    key_cap: int
    value_cap: int

    def __post_init__(self) -> None:
        for name in ("capacity", "max_entries", "key_cap", "value_cap"):
            strict_integer(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class Entry:
    slot_id: int
    key: str
    value: str

    def __post_init__(self) -> None:
        strict_integer(self.slot_id, "slot_id", minimum=1)
        valid_text(self.key, "key")
        valid_text(self.value, "value")


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
    return "".join(canonical_entry(e) for e in sorted(entries, key=lambda e: e.slot_id))


@dataclass(frozen=True, slots=True)
class Snapshot:
    entries: tuple[Entry, ...]
    serialized: str
    sha256: str
    accounted_charge: int
    concatenated_charge: int
    additive_charge: int
    utf8_bytes: int
    counter_identity: str
    counter_unit: str


@dataclass(frozen=True, slots=True)
class OpResult:
    ok: bool
    code: str
    affected_ids: tuple[int, ...] = ()
    missing_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AuditEvent:
    operation: str
    before_sha256: str
    after_sha256: str
    before_charge: int
    after_charge: int
    affected_ids: tuple[int, ...]


_MISSING = object()


class MemoryStore:
    """Mutable controller, immutable public snapshots, no automatic eviction.

    Failed mutations leave entries, next ID, and mutation audit unchanged.
    Delete atomically removes all existing requested IDs and explicitly reports
    missing IDs; an entirely missing request is a no-op/NOT_FOUND. Malformed
    delete arguments cause complete rejection. Audit data is evaluator-private
    and must not be supplied to a writer or ordinary heuristic.
    """

    def __init__(self, limits: MemoryLimits, counter: TextCounter) -> None:
        if type(limits) is not MemoryLimits:
            raise ValueError("limits must be MemoryLimits")
        self.limits = limits
        self.counter = counter
        valid_text(counter.identity, "counter identity")
        valid_text(counter.unit, "counter unit")
        self._entries: dict[int, Entry] = {}
        self._next_id = 1
        self._audit: list[AuditEvent] = []

    def _count(self, text: str) -> int:
        value = self.counter.count(text)
        if type(value) is not int or value < 0:
            # Tokenizer implementation errors must not look like policy errors.
            raise RuntimeError("counter returned a nonnegative-integer violation")
        return value

    def snapshot(self) -> Snapshot:
        entries = tuple(self._entries[i] for i in sorted(self._entries))
        serialized = serialize_entries(entries)
        whole_charge = self._count(serialized)
        return Snapshot(
            entries=entries,
            serialized=serialized,
            sha256=sha256_text(serialized),
            accounted_charge=whole_charge,
            concatenated_charge=whole_charge,
            additive_charge=sum(self._count(canonical_entry(e)) for e in entries),
            utf8_bytes=len(serialized.encode("utf-8")),
            counter_identity=self.counter.identity,
            counter_unit=self.counter.unit,
        )

    def clone(self) -> MemoryStore:
        """Fork mutable controller state while retaining immutable record values.

        Each branch receives separate entry/audit containers and its own next-ID
        counter. Immutable limits/entries/audit events and the stateless tokenizer
        may be shared. Production also clones environment cursors and RNG state;
        this method covers only the bounded memory controller.
        """
        result = MemoryStore(self.limits, self.counter)
        result._entries = self._entries.copy()
        result._audit = self._audit.copy()
        result._next_id = self._next_id
        return result

    def audit_events(self) -> tuple[AuditEvent, ...]:
        """Evaluator-only trace; not an input to retention or retrieval."""
        return tuple(self._audit)

    def _check(self, proposed: dict[int, Entry]) -> str | None:
        if len(proposed) > self.limits.max_entries:
            return "ENTRY_LIMIT"
        for entry in proposed.values():
            if self._count(entry.key) > self.limits.key_cap:
                return "KEY_LIMIT"
            if self._count(entry.value) > self.limits.value_cap:
                return "VALUE_LIMIT"
        serialized = serialize_entries(tuple(proposed.values()))
        if self._count(serialized) > self.limits.capacity:
            return "MEMORY_FULL"
        return None

    def _commit(
        self, operation: str, proposed: dict[int, Entry], affected: tuple[int, ...],
    ) -> None:
        before = self.snapshot()
        # Validate and serialize before mutating, so counter failures are atomic.
        ordered = tuple(proposed[i] for i in sorted(proposed))
        text = serialize_entries(ordered)
        charge = self._count(text)
        event = AuditEvent(
            operation, before.sha256, sha256_text(text), before.accounted_charge,
            charge, affected,
        )
        self._entries = proposed
        self._audit.append(event)

    def write(self, key: str, value: str) -> OpResult:
        try:
            candidate = Entry(self._next_id, key, value)
        except ValueError:
            return OpResult(False, "BAD_ARGUMENT")
        proposed = {**self._entries, candidate.slot_id: candidate}
        failure = self._check(proposed)
        if failure:
            return OpResult(False, failure)
        self._commit("write", proposed, (candidate.slot_id,))
        self._next_id += 1
        return OpResult(True, "OK", (candidate.slot_id,))

    def update(
        self, slot_id: int, *, key: object = _MISSING, value: object = _MISSING,
    ) -> OpResult:
        try:
            strict_integer(slot_id, "slot_id", minimum=1)
        except ValueError:
            return OpResult(False, "BAD_ARGUMENT")
        if slot_id not in self._entries:
            return OpResult(False, "NOT_FOUND", missing_ids=(slot_id,))
        if key is _MISSING and value is _MISSING:
            return OpResult(False, "BAD_ARGUMENT")
        old = self._entries[slot_id]
        try:
            candidate = Entry(
                slot_id, old.key if key is _MISSING else key,
                old.value if value is _MISSING else value,
            )
        except ValueError:
            return OpResult(False, "BAD_ARGUMENT")
        proposed = {**self._entries, slot_id: candidate}
        failure = self._check(proposed)
        if failure:
            return OpResult(False, failure)
        self._commit("update", proposed, (slot_id,))
        return OpResult(True, "OK", (slot_id,))

    def delete(self, slot_ids: tuple[int, ...]) -> OpResult:
        if type(slot_ids) is not tuple or not slot_ids:
            return OpResult(False, "BAD_ARGUMENT")
        try:
            for slot_id in slot_ids:
                strict_integer(slot_id, "slot_id", minimum=1)
            if len(set(slot_ids)) != len(slot_ids):
                raise ValueError("duplicate slot ID")
        except ValueError:
            return OpResult(False, "BAD_ARGUMENT")
        existing = tuple(i for i in slot_ids if i in self._entries)
        missing = tuple(i for i in slot_ids if i not in self._entries)
        if not existing:
            return OpResult(False, "NOT_FOUND", missing_ids=missing)
        proposed = {i: e for i, e in self._entries.items() if i not in set(existing)}
        # Whole-string token counts are not assumed monotone under deletion.
        failure = self._check(proposed)
        if failure:
            return OpResult(False, failure)
        self._commit("delete", proposed, existing)
        return OpResult(True, "OK_WITH_MISSING" if missing else "OK", existing, missing)
