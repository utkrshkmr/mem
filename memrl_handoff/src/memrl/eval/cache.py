"""Frozen-reader prompt cache. Keys include the snapshot and forbid writer adapters."""

from __future__ import annotations

from memrl.io import canonical_json, sha256_text


def reader_cache_key(*, prompt_ids: tuple[int, ...], model_revision: str,
                     adapter_sha256: str | None, snapshot_sha256: str,
                     template_sha256: str) -> str:
    if adapter_sha256 is not None:
        raise ValueError("writer adapter on reader")
    return sha256_text(canonical_json({
        "prompt_ids": list(prompt_ids),
        "model_revision": model_revision,
        "adapter_sha256": None,
        "snapshot_sha256": snapshot_sha256,
        "template_sha256": template_sha256,
        "role": "reader",
    }))


class ReaderCache:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._values.get(key)

    def put(self, key: str, value: str) -> None:
        self._values[key] = value
