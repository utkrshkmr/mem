"""Immutable boundaries between writer-visible data and evaluator secrets.

Immutability prevents accidental in-place edits. It is not an OS security
boundary: production workers must receive only the public serialization. Never
pass PrivateHistory to a writer callback and rely on that callback to ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any


def strict_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; bool is invalid")
    return value


def valid_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    # Reject lone UTF-16 surrogates rather than allowing platform-dependent bytes.
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8 text") from exc
    return value


def canonical_json(value: Any) -> str:
    """For manifests only; memory has its separately specified key ordering."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicChunk:
    chunk_id: str
    ordinal: int
    text: str

    def __post_init__(self) -> None:
        valid_text(self.chunk_id, "chunk_id")
        strict_integer(self.ordinal, "ordinal")
        valid_text(self.text, "text")


@dataclass(frozen=True, slots=True)
class PublicHistory:
    history_id: str
    chunks: tuple[PublicChunk, ...]

    def __post_init__(self) -> None:
        valid_text(self.history_id, "history_id")
        if type(self.chunks) is not tuple:
            raise ValueError("chunks must be an immutable tuple")
        if not all(type(chunk) is PublicChunk for chunk in self.chunks):
            raise ValueError("chunks must contain PublicChunk objects only")
        if tuple(c.ordinal for c in self.chunks) != tuple(range(len(self.chunks))):
            raise ValueError("chunk ordinals must be consecutive and start at zero")
        if len({c.chunk_id for c in self.chunks}) != len(self.chunks):
            raise ValueError("chunk IDs must be unique")


@dataclass(frozen=True, slots=True)
class PrivateQuestion:
    question_id: str
    question: str
    expected_answer: str
    cohort: str

    def __post_init__(self) -> None:
        for field in ("question_id", "question", "expected_answer", "cohort"):
            valid_text(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class PrivateHistory:
    public: PublicHistory
    hidden_questions: tuple[PrivateQuestion, ...]
    hidden_world_digest: str

    def __post_init__(self) -> None:
        if type(self.public) is not PublicHistory:
            raise ValueError("public must be PublicHistory")
        if type(self.hidden_questions) is not tuple or not all(
            type(q) is PrivateQuestion for q in self.hidden_questions
        ):
            raise ValueError("hidden_questions must be a tuple of PrivateQuestion")
        valid_text(self.hidden_world_digest, "hidden_world_digest")


def public_projection(history: PrivateHistory) -> PublicHistory:
    """Construct a whitelist projection; no __dict__/asdict of private objects."""
    if type(history) is not PrivateHistory:
        raise ValueError("expected PrivateHistory")
    return PublicHistory(
        history_id=history.public.history_id,
        chunks=tuple(
            PublicChunk(c.chunk_id, c.ordinal, c.text) for c in history.public.chunks
        ),
    )


def serialize_public(history: PublicHistory) -> str:
    if type(history) is not PublicHistory:
        raise ValueError("only PublicHistory may cross this boundary")
    return canonical_json({
        "history_id": history.history_id,
        "chunks": [
            {"chunk_id": c.chunk_id, "ordinal": c.ordinal, "text": c.text}
            for c in history.chunks
        ],
    })


@dataclass(frozen=True, slots=True)
class ModelLock:
    """Resolved identity for one model role, not a claim that it was downloaded."""

    repository: str
    revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    chat_template_sha256: str
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for name in ("repository", "tokenizer_repository"):
            value = valid_text(getattr(self, name), name)
            if not value or value.strip() != value:
                raise ValueError(f"{name} must be a nonempty trimmed identity")
        for name in ("revision", "tokenizer_revision"):
            value = getattr(self, name)
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ValueError(f"{name} requires a resolved 40-character commit")
        if (
            type(self.chat_template_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.chat_template_sha256) is None
        ):
            raise ValueError("chat template requires its 64-character SHA-256")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported dtype")


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

