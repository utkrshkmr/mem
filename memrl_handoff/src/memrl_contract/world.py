"""Exact toy ledger demonstrating why present sufficiency can be misleading.

Values are nonnegative integers. A create ID is never reusable, including after
cancellation. Cancellation removes that entry's CURRENT amount. A correction
replaces the current amount of an active entry. Missing/cancelled corrections,
duplicate creates, and repeated cancellation are defined no-ops with explicit
codes. These are domain semantics, not researcher-side data repairs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import canonical_json, strict_integer, valid_text


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    ref: str
    value: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"create", "cancel", "correction"}:
            raise ValueError("unknown event kind")
        valid_text(self.ref, "ref")
        if not self.ref:
            raise ValueError("ref must be nonempty")
        if self.kind == "cancel":
            if self.value is not None:
                raise ValueError("cancel has no value")
        else:
            strict_integer(self.value, "value")

    def render_observation(self) -> str:
        # This is source text shown to the writer; it does not expose a ledger.
        if self.kind == "create":
            return canonical_json({"event": "create", "id": self.ref, "value": self.value})
        if self.kind == "cancel":
            return canonical_json({"event": "cancel", "ref": self.ref})
        return canonical_json({"event": "correction", "ref": self.ref, "new_value": self.value})


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    ref: str
    value: int
    active: bool

    def __post_init__(self) -> None:
        valid_text(self.ref, "ref")
        if not self.ref:
            raise ValueError("ref must be nonempty")
        strict_integer(self.value, "value")
        if type(self.active) is not bool:
            raise ValueError("active must be bool")


@dataclass(frozen=True, slots=True)
class LedgerWorld:
    entries: tuple[LedgerEntry, ...] = ()

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or not all(
            type(e) is LedgerEntry for e in self.entries
        ):
            raise ValueError("entries must be a tuple of LedgerEntry")
        if tuple(e.ref for e in self.entries) != tuple(sorted(e.ref for e in self.entries)):
            raise ValueError("entries must be sorted by ref")
        if len({e.ref for e in self.entries}) != len(self.entries):
            raise ValueError("duplicate ledger ID")

    @property
    def total(self) -> int:
        return sum(entry.value for entry in self.entries if entry.active)


@dataclass(frozen=True, slots=True)
class EventOutcome:
    world: LedgerWorld
    changed: bool
    code: str


def apply_event(world: LedgerWorld, event: Event) -> EventOutcome:
    if type(world) is not LedgerWorld or type(event) is not Event:
        raise ValueError("expected LedgerWorld and Event")
    entries = {entry.ref: entry for entry in world.entries}
    old = entries.get(event.ref)
    if event.kind == "create":
        if old is not None:
            return EventOutcome(world, False, "ID_ALREADY_USED")
        entries[event.ref] = LedgerEntry(event.ref, event.value, True)
    elif old is None:
        return EventOutcome(world, False, "UNKNOWN_REF")
    elif not old.active:
        code = "ALREADY_CANCELLED" if event.kind == "cancel" else "INACTIVE_REF"
        return EventOutcome(world, False, code)
    elif event.kind == "cancel":
        entries[event.ref] = LedgerEntry(event.ref, old.value, False)
    else:
        if event.value == old.value:
            return EventOutcome(world, False, "UNCHANGED_VALUE")
        entries[event.ref] = LedgerEntry(event.ref, event.value, True)
    result = LedgerWorld(tuple(entries[ref] for ref in sorted(entries)))
    return EventOutcome(result, True, "OK")


def replay(events: tuple[Event, ...], initial: LedgerWorld | None = None) -> LedgerWorld:
    world = LedgerWorld() if initial is None else initial
    for event in events:
        world = apply_event(world, event).world
    return world


def continuation_counterexample() -> tuple[LedgerWorld, LedgerWorld, Event]:
    """Both current totals are 10; the SAME cancel(p) leaves 8 versus 3."""
    first = replay((Event("create", "p", 2), Event("create", "q", 8)))
    second = replay((Event("create", "p", 7), Event("create", "q", 3)))
    return first, second, Event("cancel", "p")

