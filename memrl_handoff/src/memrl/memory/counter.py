"""Injected text counters. Toy byte counts are not model tokens."""

from __future__ import annotations

from dataclasses import dataclass

from memrl.contracts.common import valid_text


class TextCounter:
    identity: str
    unit: str

    def count(self, text: str) -> int:  # pragma: no cover - protocol
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class CodepointCounter:
    """Deterministic test counter. Not a model tokenizer and not a paper budget."""

    identity: str = "test-codepoint-v1"
    unit: str = "unicode_code_points_TEST_ONLY"

    def count(self, text: str) -> int:
        valid_text(text, "text")
        return len(text)


@dataclass(frozen=True, slots=True)
class NonMonotoneCounter:
    """Deletion can increase the charge when a sentinel leaves the store."""

    identity: str = "test-nonmonotone-v1"
    unit: str = "nonmonotone_TEST_ONLY"
    sentinel: str = "DROPME"

    def count(self, text: str) -> int:
        valid_text(text, "text")
        if self.sentinel in text:
            return 1
        return len(text)


class HuggingFaceCounter:
    """Pinned tokenizer with special tokens disabled. Identity is recorded."""

    def __init__(self, tokenizer, *, repository: str, revision: str) -> None:
        self._tokenizer = tokenizer
        self.repository = repository
        self.revision = revision
        self.identity = f"hf:{repository}@{revision}"
        self.unit = "model_tokens"

    def count(self, text: str) -> int:
        valid_text(text, "text")
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(encoded, list) or not all(type(token) is int for token in encoded):
            raise RuntimeError("tokenizer returned a non-integer encoding")
        return len(encoded)
