"""Scripted constructors. Model summaries are not fabricated."""

from __future__ import annotations

import json

from memrl.data.private.events import DomainEvent
from memrl.memory.counter import TextCounter
from memrl.memory.store import MemoryLimits, MemoryStore


class ModelRequired(RuntimeError):
    pass


def empty_store(limits: MemoryLimits, counter: TextCounter) -> MemoryStore:
    return MemoryStore(limits, counter, eviction="reject")


def event_ledger(chunks: tuple[str, ...], limits: MemoryLimits, counter: TextCounter) -> MemoryStore:
    """Parse rendered public events only. Private generator objects are not inputs."""
    store = MemoryStore(limits, counter, eviction="reject")
    slots: dict[str, int] = {}
    for chunk in chunks:
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = payload.get("event")
            if kind == "create":
                result = store.write(str(payload["id"]), str(payload["value"]))
                if result.ok:
                    slots[str(payload["id"])] = result.affected_ids[0]
            elif kind == "correction":
                slot = slots.get(str(payload["ref"]))
                if slot is None:
                    continue
                store.update(slot, value=str(payload["new_value"]))
            elif kind == "cancel":
                slot = slots.get(str(payload["ref"]))
                if slot is None:
                    continue
                result = store.delete((slot,))
                if result.ok:
                    slots.pop(str(payload["ref"]), None)
    return store


def recent_window(chunks: tuple[str, ...], limits: MemoryLimits, counter: TextCounter) -> MemoryStore:
    store = MemoryStore(limits, counter, eviction="fifo")
    for chunk in chunks:
        for line in chunk.splitlines():
            if line:
                store.write("raw", line)
    return store


def fifo_fragments(chunks: tuple[str, ...], limits: MemoryLimits, counter: TextCounter) -> MemoryStore:
    return recent_window(chunks, limits, counter)


def random_fragments(chunks: tuple[str, ...], limits: MemoryLimits, counter: TextCounter,
                     seed: int) -> MemoryStore:
    import random
    store = MemoryStore(limits, counter, eviction="random", rng=random.Random(seed))
    for chunk in chunks:
        for line in chunk.splitlines():
            if line:
                store.write("raw", line)
    return store


def rolling_summary(*_args, **_kwargs):
    raise ModelRequired("rolling summary requires a pinned model; no text was invented")


def require_public_events(events: tuple[DomainEvent, ...]) -> None:
    if not events:
        raise ValueError("ledger baseline received no events")
