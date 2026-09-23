"""Typed scoring. Substring containment is not a match."""

from __future__ import annotations

import json
from typing import Any

from memrl.contracts.types import SCHEMA_VERSION, AnswerReceipt, ScoreRow
from memrl.io import sha256_text

SCORER_VERSION = "typed-exact-v1"


def parse_answer_object(text: str) -> tuple[object, str]:
    if type(text) is not str:
        return None, "malformed_answer"
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "malformed_answer"
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None, "malformed_answer"
    if not isinstance(obj, dict) or set(obj) != {"answer"}:
        return None, "malformed_answer"
    return obj["answer"], "parsed"


def score_typed(predicted: object, gold: object, answer_type: str) -> tuple[bool, str]:
    if answer_type == "unknown":
        return predicted is None, "ok" if predicted is None else "wrong"
    if answer_type == "integer":
        if isinstance(predicted, bool) or type(predicted) is not int:
            return False, "type_mismatch"
        return predicted == gold, "ok" if predicted == gold else "wrong"
    if answer_type == "member":
        if type(predicted) is not str:
            return False, "type_mismatch"
        return predicted == gold, "ok" if predicted == gold else "wrong"
    if answer_type == "set":
        if type(predicted) is not list or any(type(item) is not str for item in predicted):
            return False, "type_mismatch"
        return sorted(predicted) == sorted(gold), "ok" if sorted(predicted) == sorted(gold) else "wrong"
    raise ValueError(f"unknown answer type: {answer_type}")


def score_text(answer_text: str, gold: object, answer_type: str, *, failure_class: str,
               receipt: AnswerReceipt, world_id: str, ancestor_id: str, run_id: str,
               branch_index: int | None, infrastructure: bool = False) -> ScoreRow:
    if failure_class == "infrastructure_failure" or infrastructure:
        return ScoreRow(
            SCHEMA_VERSION, run_id, world_id, ancestor_id, branch_index, receipt.question_id,
            False, "infrastructure_failure", answer_type, SCORER_VERSION,
            receipt.snapshot_sha256, sha256_text(json.dumps(gold, sort_keys=True)), True,
        )
    if failure_class in {"model_generation_limit", "model_invalid_action"}:
        success = False
        klass = failure_class
    else:
        predicted, parsed = parse_answer_object(answer_text)
        if parsed != "parsed":
            success, klass = False, "malformed_answer"
        else:
            success, klass = score_typed(predicted, gold, answer_type)
    return ScoreRow(
        SCHEMA_VERSION, run_id, world_id, ancestor_id, branch_index, receipt.question_id,
        success, klass, answer_type, SCORER_VERSION, receipt.snapshot_sha256,
        sha256_text(json.dumps({"answer": gold}, sort_keys=True, default=str)), False,
    )


def support_status(snapshot_text: str, ref: str, value: int | None, *, active: bool) -> str:
    """Symbolic audit. A missing surface string is not automatic absence."""
    if not active or value is None:
        return "certified_absent" if ref not in snapshot_text else "uncertain"
    needle = json.dumps(str(value))
    if f'"key":"{ref}"' in snapshot_text.replace(" ", "") and needle in snapshot_text:
        return "certified_present"
    if ref in snapshot_text:
        return "uncertain"
    return "certified_absent"
