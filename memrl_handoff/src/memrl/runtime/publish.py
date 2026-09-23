"""Atomic job and adapter publication."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from memrl.io import atomic_json, canonical_json, fingerprint, sha256_bytes


def publish_json(directory: str | Path, name: str, payload: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.exists():
        raise FileExistsError(f"refusing to overwrite published artifact {name}")
    atomic_json(target, payload)
    return target


def publish_adapter(directory: str | Path, files: dict[str, bytes], *, version: int,
                    base_revision: str) -> dict:
    directory = Path(directory)
    temporary = directory / f".adapter-{version}.partial"
    final = directory / f"adapter-{version}"
    if final.exists():
        raise FileExistsError("adapter version already published")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    hashes = {}
    for name, payload in sorted(files.items()):
        path = temporary / name
        with path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        hashes[name] = sha256_bytes(payload)
    manifest = {
        "schema_version": 3,
        "version": version,
        "base_revision": base_revision,
        "files": hashes,
    }
    manifest_bytes = canonical_json(manifest).encode("utf-8")
    (temporary / "manifest.json").write_bytes(manifest_bytes)
    digest = sha256_bytes(manifest_bytes + canonical_json(hashes).encode("utf-8"))
    manifest["adapter_sha256"] = digest
    (temporary / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary, final)
    return manifest


def acknowledge(directory: str | Path, worker_id: str, adapter_sha256: str) -> Path:
    return publish_json(
        directory, f"ack-{worker_id}.json",
        {"worker_id": worker_id, "adapter_sha256": adapter_sha256},
    )


def load_complete(path: str | Path) -> dict:
    path = Path(path)
    if path.name.startswith(".") or path.suffix == ".partial":
        raise ValueError("refusing to read a partial artifact")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("empty artifact")
    return __import__("json").loads(text)


def job_id(payload: dict) -> str:
    return fingerprint(payload)
