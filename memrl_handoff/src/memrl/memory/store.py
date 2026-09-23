"""Transactional bounded memory. Failed mutations do not change live state."""

from __future__ import annotations

from dataclasses import dataclass
import random

from memrl.contracts.common import strict_integer, valid_text
from memrl.io import sha256_text
from memrl.memory.counter import TextCounter
from memrl.memory.serialize import Entry, canonical_entry, serialize_entries

_MISSING = object()


@dataclass(frozen=True, slots=True)
class MemoryLimits:
    capacity: int
    max_entries: int
    key_cap: int
    value_cap: int
    index_cap: int | None = None

    def __post_init__(self) -> None:
        for name in ("capacity", "max_entries", "key_cap", "value_cap"):
            strict_integer(getattr(self, name), name, minimum=1)
        if self.index_cap is not None:
            strict_integer(self.index_cap, "index_cap", minimum=1)


@dataclass(frozen=True, slots=True)
class Snapshot:
    entries: tuple[Entry, ...]
    serialized: str
    sha256: str
    accounted_charge: int
    additive_charge: int
    counter_identity: str
    counter_unit: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    code: str
    affected_ids: tuple[int, ...] = ()
    missing_ids: tuple[int, ...] = ()

    @property
    def public_message(self) -> str:
        affected = ",".join(str(item) for item in self.affected_ids)
        missing = ",".join(str(item) for item in self.missing_ids)
        return f"{self.code} affected={affected} missing={missing}"


@dataclass(frozen=True, slots=True)
class TransactionPreview:
    source_sha256: str
    ok: bool
    code: str
    entries: tuple[Entry, ...]
    serialized: str
    sha256: str
    charge: int
    next_id: int
    affected_ids: tuple[int, ...]
    missing_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AuditEvent:
    operation: str
    before_sha256: str
    after_sha256: str
    before_charge: int
    after_charge: int
    affected_ids: tuple[int, ...]


class MemoryStore:
    def __init__(self, limits: MemoryLimits, counter: TextCounter, *,
                 eviction: str = "reject", rng: random.Random | None = None) -> None:
        if type(limits) is not MemoryLimits:
            raise ValueError("limits must be MemoryLimits")
        if eviction not in {"reject", "fifo", "random"}:
            raise ValueError("unsupported eviction policy")
        self.limits = limits
        self.counter = counter
        valid_text(counter.identity, "counter identity")
        valid_text(counter.unit, "counter unit")
        if "TOY" in counter.unit.upper() or counter.unit.endswith("TEST_ONLY"):
            self._paper_tokens = False
        else:
            self._paper_tokens = counter.unit == "model_tokens"
        self.eviction = eviction
        self._entries: dict[int, Entry] = {}
        self._next_id = 1
        self._audit: list[AuditEvent] = []
        self._rng = rng or random.Random(0)

    def public_snapshot(self) -> bytes:
        return self.snapshot().serialized.encode("utf-8")

    def snapshot_sha256(self) -> str:
        return self.snapshot().sha256

    def charged_tokens(self) -> int:
        return self.snapshot().accounted_charge

    def snapshot(self) -> Snapshot:
        entries = tuple(self._entries[i] for i in sorted(self._entries))
        serialized = serialize_entries(entries)
        whole = self._count(serialized)
        return Snapshot(
            entries=entries,
            serialized=serialized,
            sha256=sha256_text(serialized),
            accounted_charge=whole,
            additive_charge=sum(self._count(canonical_entry(entry)) for entry in entries),
            counter_identity=self.counter.identity,
            counter_unit=self.counter.unit,
        )

    def clone(self) -> "MemoryStore":
        result = MemoryStore(self.limits, self.counter, eviction=self.eviction)
        result._entries = dict(self._entries)
        result._audit = list(self._audit)
        result._next_id = self._next_id
        result._rng.setstate(self._rng.getstate())
        return result

    def audit_events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._audit)

    def _count(self, text: str) -> int:
        value = self.counter.count(text)
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise RuntimeError("counter returned a nonnegative-integer violation")
        return value

    def _check(self, proposed: dict[int, Entry]) -> str | None:
        if len(proposed) > self.limits.max_entries:
            return "ENTRY_LIMIT"
        for entry in proposed.values():
            if self._count(entry.key) > self.limits.key_cap:
                return "KEY_LIMIT"
            if self._count(entry.value) > self.limits.value_cap:
                return "VALUE_LIMIT"
        serialized = serialize_entries(tuple(proposed[i] for i in sorted(proposed)))
        charge = self._count(serialized)
        if charge > self.limits.capacity:
            return "MEMORY_FULL"
        if self.limits.index_cap is not None and charge > self.limits.index_cap:
            return "INDEX_LIMIT"
        return None

    def _apply_eviction(self, proposed: dict[int, Entry], protected: int | None) -> dict[int, Entry] | None:
        if self.eviction == "reject":
            return None
        working = dict(proposed)
        while self._check(working) is not None and any(i != protected for i in working):
            victims = [i for i in working if i != protected]
            if not victims:
                break
            if self.eviction == "fifo":
                victim = min(victims)
            else:
                victim = self._rng.choice(sorted(victims))
            del working[victim]
        if self._check(working) is not None:
            return None
        return working

    def preview_write(self, key: str, value: str) -> TransactionPreview:
        before = self.snapshot()
        try:
            candidate = Entry(self._next_id, key, value)
        except ValueError:
            return self._reject(before, "BAD_ARGUMENT")
        proposed = {**self._entries, candidate.slot_id: candidate}
        failure = self._check(proposed)
        removed: tuple[int, ...] = ()
        if failure:
            evicted = self._apply_eviction(proposed, candidate.slot_id)
            if evicted is None:
                return self._reject(before, failure)
            removed = tuple(sorted(set(proposed) - set(evicted)))
            proposed = evicted
        affected = (candidate.slot_id,) + removed
        return self._ok(before, proposed, self._next_id + 1, affected, ())

    def preview_update(self, slot_id: object, *, key: object = _MISSING,
                       value: object = _MISSING) -> TransactionPreview:
        before = self.snapshot()
        try:
            strict_integer(slot_id, "slot_id", minimum=1)
        except ValueError:
            return self._reject(before, "BAD_ARGUMENT")
        if slot_id not in self._entries:
            return self._reject(before, "NOT_FOUND", missing=(slot_id,))
        if key is _MISSING and value is _MISSING:
            return self._reject(before, "BAD_ARGUMENT")
        old = self._entries[slot_id]
        try:
            candidate = Entry(
                slot_id,
                old.key if key is _MISSING else key,  # type: ignore[arg-type]
                old.value if value is _MISSING else value,  # type: ignore[arg-type]
            )
        except ValueError:
            return self._reject(before, "BAD_ARGUMENT")
        proposed = {**self._entries, slot_id: candidate}
        failure = self._check(proposed)
        if failure:
            return self._reject(before, failure)
        return self._ok(before, proposed, self._next_id, (slot_id,), ())

    def preview_delete(self, slot_ids: object) -> TransactionPreview:
        before = self.snapshot()
        if type(slot_ids) is not tuple or not slot_ids:
            return self._reject(before, "BAD_ARGUMENT")
        try:
            for slot_id in slot_ids:
                strict_integer(slot_id, "slot_id", minimum=1)
            if len(set(slot_ids)) != len(slot_ids):
                raise ValueError("duplicate slot ID")
        except ValueError:
            return self._reject(before, "BAD_ARGUMENT")
        existing = tuple(i for i in slot_ids if i in self._entries)
        missing = tuple(i for i in slot_ids if i not in self._entries)
        if not existing:
            return self._reject(before, "NOT_FOUND", missing=missing)
        proposed = {i: entry for i, entry in self._entries.items() if i not in set(existing)}
        failure = self._check(proposed)
        if failure:
            return self._reject(before, failure)
        code_ok = True
        preview = self._ok(before, proposed, self._next_id, existing, missing)
        if missing:
            return TransactionPreview(
                before.sha256, code_ok, "OK_WITH_MISSING", preview.entries, preview.serialized,
                preview.sha256, preview.charge, preview.next_id, existing, missing,
            )
        return preview

    def commit(self, preview: TransactionPreview) -> ToolResult:
        if type(preview) is not TransactionPreview:
            raise ValueError("preview must be TransactionPreview")
        current = self.snapshot()
        if preview.source_sha256 != current.sha256:
            return ToolResult(False, "STALE_SNAPSHOT")
        if not preview.ok:
            return ToolResult(False, preview.code, preview.affected_ids, preview.missing_ids)
        event = AuditEvent(
            "commit", current.sha256, preview.sha256, current.accounted_charge,
            preview.charge, preview.affected_ids,
        )
        self._entries = {entry.slot_id: entry for entry in preview.entries}
        self._next_id = preview.next_id
        self._audit.append(event)
        return ToolResult(True, preview.code, preview.affected_ids, preview.missing_ids)

    def write(self, key: str, value: str) -> ToolResult:
        return self.commit(self.preview_write(key, value))

    def update(self, slot_id: int, *, key: object = _MISSING, value: object = _MISSING) -> ToolResult:
        return self.commit(self.preview_update(slot_id, key=key, value=value))

    def delete(self, slot_ids: tuple[int, ...]) -> ToolResult:
        return self.commit(self.preview_delete(slot_ids))

    def _reject(self, before: Snapshot, code: str,
                missing: tuple[int, ...] = ()) -> TransactionPreview:
        return TransactionPreview(
            before.sha256, False, code, before.entries, before.serialized, before.sha256,
            before.accounted_charge, self._next_id, (), missing,
        )

    def _ok(self, before: Snapshot, proposed: dict[int, Entry], next_id: int,
            affected: tuple[int, ...], missing: tuple[int, ...]) -> TransactionPreview:
        entries = tuple(proposed[i] for i in sorted(proposed))
        serialized = serialize_entries(entries)
        charge = self._count(serialized)
        return TransactionPreview(
            before.sha256, True, "OK", entries, serialized, sha256_text(serialized),
            charge, next_id, affected, missing,
        )
