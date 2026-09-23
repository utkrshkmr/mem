"""GPU leases. One owner at a time; stale locks cannot be stolen silently."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from memrl.io import atomic_json


class LeaseError(RuntimeError):
    pass


class LeaseRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, gpu_uuid: str) -> Path:
        safe = gpu_uuid.replace("/", "_")
        return self.root / f"{safe}.lease.json"

    def acquire(self, gpu_uuid: str, owner: str, ttl_seconds: float) -> dict:
        path = self._path(gpu_uuid)
        now = time.time()
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current["expires_at"] > now and current["owner"] != owner:
                raise LeaseError(f"GPU lease held by {current['owner']}")
        payload = {
            "gpu_uuid": gpu_uuid,
            "owner": owner,
            "acquired_at": now,
            "expires_at": now + ttl_seconds,
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(path, flags, 0o644)
        except FileExistsError:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current["expires_at"] > now and current["owner"] != owner:
                raise LeaseError(f"GPU lease held by {current['owner']}")
            if current["owner"] != owner:
                raise LeaseError("refusing to steal a lease file")
            atomic_json(path, payload)
            return payload
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        return payload

    def release(self, gpu_uuid: str, owner: str) -> None:
        path = self._path(gpu_uuid)
        if not path.exists():
            return
        current = json.loads(path.read_text(encoding="utf-8"))
        if current["owner"] != owner:
            raise LeaseError("owner mismatch on release")
        path.unlink()

    def heartbeat(self, gpu_uuid: str, owner: str, ttl_seconds: float) -> dict:
        path = self._path(gpu_uuid)
        current = json.loads(path.read_text(encoding="utf-8"))
        if current["owner"] != owner:
            raise LeaseError("owner mismatch on heartbeat")
        current["expires_at"] = time.time() + ttl_seconds
        atomic_json(path, current)
        return current
