"""Runnable CPU reference contracts, not a GPU training implementation.

Run tests without installing dependencies:
    PYTHONPATH=src python -m unittest discover -s tests -v
"""

from .memory import ByteCounter, MemoryLimits, MemoryStore
from .world import Event, LedgerWorld, apply_event

__all__ = [
    "ByteCounter", "MemoryLimits", "MemoryStore", "Event", "LedgerWorld",
    "apply_event",
]

