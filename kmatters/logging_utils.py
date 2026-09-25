"""Logging setup with token redaction on every handler, and a JSONL writer.

Filters attached to a logger do not apply to records from child loggers, so the
RedactFilter goes on each handler (PLAN.md §5.3).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from .hf_auth import RedactFilter

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger with a console handler and an optional file handler.
    Calling it again replaces the handlers it installed before."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_kmatters", False):
            root.removeHandler(h)
            h.close()
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h._kmatters = True
        h.setFormatter(logging.Formatter(_FORMAT))
        h.addFilter(RedactFilter())
        root.addHandler(h)
    root.setLevel(level)
    return root


class JsonlWriter:
    """Append-only JSON-lines file; one flushed line per record."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record, sort_keys=True, default=_default) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
