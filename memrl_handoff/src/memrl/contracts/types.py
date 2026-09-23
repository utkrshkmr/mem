"""Immutable production records. Unknown fields are rejected by constructors."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from memrl.io import canonical_json, sha256_text
from memrl.contracts.common import (
    commit_hex,
    finite_number,
    sha256_hex,
    strict_integer,
    valid_text,
)

SCHEMA_VERSION = 3
StopReason = Literal["eos", "stop", "length", "error"]
Phase = Literal["prefix", "suffix"]
Role = Literal["writer", "reader"]

FORBIDDEN_OBSERVATION_FIELDS = frozenset({
    "world_id", "family", "gold", "future", "answer", "support", "split",
    "path", "filepath", "filename", "query_ticket", "expected_answer",
    "hidden", "suffix",
})


def _reject_forbidden(payload: dict[str, Any], where: str) -> None:
    found = FORBIDDEN_OBSERVATION_FIELDS.intersection(payload)
    if found:
        raise ValueError(f"{where} contains forbidden fields: {sorted(found)}")


@dataclass(frozen=True, slots=True)
class ObservationChunk:
    text: str
    sequence_index: int
    is_last_in_visible_phase: bool

    def __post_init__(self) -> None:
        valid_text(self.text, "text")
        strict_integer(self.sequence_index, "sequence_index")
        if type(self.is_last_in_visible_phase) is not bool:
            raise ValueError("is_last_in_visible_phase must be bool")


@dataclass(frozen=True, slots=True)
class WriterObservation:
    task_instruction: str
    chunk: ObservationChunk
    memory_index: str
    loaded_text: str
    last_tool_result: str

    def __post_init__(self) -> None:
        valid_text(self.task_instruction, "task_instruction")
        if type(self.chunk) is not ObservationChunk:
            raise ValueError("chunk must be ObservationChunk")
        valid_text(self.memory_index, "memory_index")
        valid_text(self.loaded_text, "loaded_text")
        valid_text(self.last_tool_result, "last_tool_result")

    def payload(self) -> dict[str, Any]:
        return observation_payload(self)


def observation_payload(observation: WriterObservation) -> dict[str, Any]:
    payload = {
        "task_instruction": observation.task_instruction,
        "chunk": {
            "text": observation.chunk.text,
            "sequence_index": observation.chunk.sequence_index,
            "is_last_in_visible_phase": observation.chunk.is_last_in_visible_phase,
        },
        "memory_index": observation.memory_index,
        "loaded_text": observation.loaded_text,
        "last_tool_result": observation.last_tool_result,
    }
    _reject_forbidden(payload, "writer observation")
    _reject_forbidden(payload["chunk"], "writer chunk")
    return payload


def serialize_observation(observation: WriterObservation) -> str:
    return canonical_json(observation_payload(observation))


@dataclass(frozen=True, slots=True)
class ManifestChunk:
    chunk_id: str
    ordinal: int
    text: str

    def __post_init__(self) -> None:
        valid_text(self.chunk_id, "chunk_id")
        if not self.chunk_id:
            raise ValueError("chunk_id must be nonempty")
        strict_integer(self.ordinal, "ordinal")
        valid_text(self.text, "text")
        blob = canonical_json({"chunk_id": self.chunk_id, "text": self.text})
        lowered = blob.lower()
        for token in ("gold", "expected_answer", "query_ticket"):
            if token in lowered:
                raise ValueError("manifest chunk text carries a private marker")


@dataclass(frozen=True, slots=True)
class PublicManifest:
    """Writer-visible history. Split, gold, and futures are absent."""

    history_id: str
    chunks: tuple[ManifestChunk, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        valid_text(self.history_id, "history_id")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", self.history_id):
            raise ValueError("history_id must be an opaque routing token")
        if any(part in self.history_id.lower() for part in ("train", "test", "dev", "split", "gold")):
            raise ValueError("history_id must not encode split or gold")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if type(self.chunks) is not tuple or not self.chunks:
            raise ValueError("chunks must be a nonempty tuple")
        if not all(type(chunk) is ManifestChunk for chunk in self.chunks):
            raise ValueError("chunks must contain ManifestChunk objects")
        if tuple(chunk.ordinal for chunk in self.chunks) != tuple(range(len(self.chunks))):
            raise ValueError("chunk ordinals must be consecutive and start at zero")
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("chunk IDs must be unique")

    def public_json(self) -> str:
        return canonical_json({
            "schema_version": self.schema_version,
            "history_id": self.history_id,
            "chunks": [
                {"chunk_id": chunk.chunk_id, "ordinal": chunk.ordinal, "text": chunk.text}
                for chunk in self.chunks
            ],
        })

    @property
    def sha256(self) -> str:
        return sha256_text(self.public_json())


@dataclass(frozen=True, slots=True)
class ModelLock:
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
        commit_hex(self.revision, "revision")
        commit_hex(self.tokenizer_revision, "tokenizer_revision")
        sha256_hex(self.chat_template_sha256, "chat_template_sha256")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported dtype")


@dataclass(frozen=True, slots=True)
class TokenizedCall:
    prompt_ids: tuple[int, ...]
    generation_cap: int
    adapter_sha256: str | None
    action_seed: int
    role: Role
    request_id: str

    def __post_init__(self) -> None:
        if type(self.prompt_ids) is not tuple or not self.prompt_ids:
            raise ValueError("prompt_ids must be a nonempty tuple of ints")
        if not all(type(token) is int and token >= 0 for token in self.prompt_ids):
            raise ValueError("prompt_ids must be nonnegative integers")
        strict_integer(self.generation_cap, "generation_cap", minimum=1)
        if self.adapter_sha256 is not None:
            sha256_hex(self.adapter_sha256, "adapter_sha256")
        if self.role == "reader" and self.adapter_sha256 is not None:
            raise ValueError("writer adapter on reader")
        strict_integer(self.action_seed, "action_seed")
        if self.role not in ("writer", "reader"):
            raise ValueError("invalid role")
        valid_text(self.request_id, "request_id")


@dataclass(frozen=True, slots=True)
class Generation:
    completion_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...]
    text: str
    stop_reason: StopReason
    adapter_sha256: str | None
    request_id: str
    role: Role

    def __post_init__(self) -> None:
        if type(self.completion_ids) is not tuple or type(self.completion_logprobs) is not tuple:
            raise ValueError("completion ids and logprobs must be tuples")
        if len(self.completion_ids) != len(self.completion_logprobs):
            raise ValueError("corrupt behavior-logprob length")
        if not all(type(token) is int and token >= 0 for token in self.completion_ids):
            raise ValueError("completion ids must be nonnegative integers")
        for value in self.completion_logprobs:
            finite_number(value, "completion_logprob")
        valid_text(self.text, "text")
        if self.stop_reason not in ("eos", "stop", "length", "error"):
            raise ValueError("invalid stop_reason")
        if self.adapter_sha256 is not None:
            sha256_hex(self.adapter_sha256, "adapter_sha256")
        if self.role == "reader" and self.adapter_sha256 is not None:
            raise ValueError("writer adapter on reader")
        valid_text(self.request_id, "request_id")
        if self.role not in ("writer", "reader"):
            raise ValueError("invalid role")


@dataclass(frozen=True, slots=True)
class AnswerReceipt:
    snapshot_sha256: str
    question_id: str
    answer_text: str
    reader_prompt_sha256: str
    failure_class: str
    adapter_sha256: None = None

    def __post_init__(self) -> None:
        sha256_hex(self.snapshot_sha256, "snapshot_sha256")
        sha256_hex(self.reader_prompt_sha256, "reader_prompt_sha256")
        valid_text(self.question_id, "question_id")
        valid_text(self.answer_text, "answer_text")
        valid_text(self.failure_class, "failure_class")
        if self.adapter_sha256 is not None:
            raise ValueError("writer adapter on reader")


@dataclass(frozen=True, slots=True)
class ScoreRow:
    schema_version: int
    run_id: str
    world_id: str
    ancestor_id: str
    branch_index: int | None
    question_id: str
    success: bool
    failure_class: str
    answer_type: str
    scorer_version: str
    snapshot_sha256: str
    gold_sha256: str
    infrastructure: bool

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        for name in ("run_id", "world_id", "ancestor_id", "question_id", "failure_class",
                     "answer_type", "scorer_version"):
            valid_text(getattr(self, name), name)
        if type(self.success) is not bool or type(self.infrastructure) is not bool:
            raise ValueError("success and infrastructure must be bool")
        if self.branch_index is not None:
            strict_integer(self.branch_index, "branch_index")
        sha256_hex(self.snapshot_sha256, "snapshot_sha256")
        sha256_hex(self.gold_sha256, "gold_sha256")
        if self.infrastructure and self.success:
            raise ValueError("infrastructure failure cannot be a success")


@dataclass(frozen=True, slots=True)
class WeightedCall:
    schema_version: int
    call_id: str
    phase: Phase
    history_index: int
    prefix_sample: int
    branch_index: int | None
    adapter_sha256: str
    raw_advantage: float
    branch_factor: float
    inclusion_probability: float
    global_groups: int
    G: int
    K: int
    fixed_scale: float
    group_denominator: int
    call_logprob_sum: float

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        valid_text(self.call_id, "call_id")
        if not self.call_id:
            raise ValueError("call_id must be nonempty")
        if self.phase not in ("prefix", "suffix"):
            raise ValueError("invalid phase")
        strict_integer(self.history_index, "history_index")
        strict_integer(self.prefix_sample, "prefix_sample")
        sha256_hex(self.adapter_sha256, "adapter_sha256")
        finite_number(self.raw_advantage, "raw_advantage")
        factor = finite_number(self.branch_factor, "branch_factor")
        if not 0 < factor <= 1:
            raise ValueError("branch_factor must lie in (0, 1]")
        probability = finite_number(self.inclusion_probability, "inclusion_probability")
        if not 0 < probability <= 1:
            raise ValueError("inclusion_probability must lie in (0, 1]")
        strict_integer(self.global_groups, "global_groups", minimum=1)
        strict_integer(self.G, "G", minimum=2)
        strict_integer(self.K, "K", minimum=1)
        scale = finite_number(self.fixed_scale, "fixed_scale")
        if scale <= 0:
            raise ValueError("fixed_scale must be > 0")
        strict_integer(self.group_denominator, "group_denominator", minimum=1)
        finite_number(self.call_logprob_sum, "call_logprob_sum")
        if self.phase == "prefix":
            if self.branch_index is not None or self.branch_factor != 1:
                raise ValueError("prefix calls have a null branch and branch_factor 1")
        else:
            strict_integer(self.branch_index, "branch_index")
            if self.branch_index >= self.K:
                raise ValueError("branch_index out of range")
        expected = self.global_groups * self.G
        if self.group_denominator != expected:
            raise ValueError("group_denominator must equal global_groups * G")

    @property
    def coefficient(self) -> float:
        return (
            -self.raw_advantage
            * self.branch_factor
            / (self.group_denominator * self.fixed_scale * self.inclusion_probability)
        )

    def wire(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "call_id": self.call_id,
            "phase": self.phase,
            "history_index": self.history_index,
            "prefix_sample": self.prefix_sample,
            "branch_index": self.branch_index,
            "adapter_sha256": self.adapter_sha256,
            "raw_advantage": self.raw_advantage,
            "branch_factor": self.branch_factor,
            "inclusion_probability": self.inclusion_probability,
            "global_groups": self.global_groups,
            "G": self.G,
            "K": self.K,
            "fixed_scale": self.fixed_scale,
            "group_denominator": self.group_denominator,
            "call_logprob_sum": self.call_logprob_sum,
        }


@dataclass(frozen=True, slots=True)
class LaunchReceipt:
    protocol_id: str
    config_sha256: str
    code_commit: str
    code_dirty_diff_sha256: str
    environment_lock_sha256: str
    data_manifest_sha256: str
    gate_record_sha256: str
    resource_lease_id: str
    created_at_utc: str

    def __post_init__(self) -> None:
        valid_text(self.protocol_id, "protocol_id")
        for name in (
            "config_sha256", "code_dirty_diff_sha256", "environment_lock_sha256",
            "data_manifest_sha256", "gate_record_sha256",
        ):
            sha256_hex(getattr(self, name), name)
        commit_hex(self.code_commit, "code_commit")
        valid_text(self.resource_lease_id, "resource_lease_id")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", self.created_at_utc):
            raise ValueError("created_at_utc must be UTC Zulu time")
