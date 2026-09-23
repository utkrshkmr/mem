"""Exact writer and reader prompt assembly. No silent truncation."""

from __future__ import annotations

from memrl.contracts.types import WriterObservation, observation_payload
from memrl.io import canonical_json, sha256_text
from memrl.memory.counter import TextCounter

WRITER_SYSTEM = """You will read a history once, one part at a time. Later a separate assistant
will answer questions using only the memory you leave. You will not see the
actual questions while reading.

Memory limits: {C} accounted tokens, {K} entries, at most {key_cap} tokens
per key and {value_cap} tokens per value. The displayed entry format and IDs
also count toward memory capacity.

Preserve accurate facts, relationships, and dates. Questions may ask current
or earlier states, so preserve history when it is relevant to the stated task.
Use the task prior below; do not assume all questions ask only the newest value.

You can read/search only the current memory. You cannot revisit past history.
Use one tool call per response:
<tool_call>{{"name":"...","arguments":{{...}}}}</tool_call>

{tool_schemas}
Task prior: {public_task_prior}
"""

READER_SYSTEM = """Answer using the provided memory. If the required information is absent or
insufficient, return {"answer": null}. Do not infer a personal/random fact from
general knowledge. Follow the required answer type exactly.
"""


class PromptBudgetError(RuntimeError):
    pass


def ensure_prompt_fits(prompt_units: int, max_new_units: int, total: int) -> None:
    required = prompt_units + max_new_units
    if required > total:
        raise PromptBudgetError(f"prompt budget bug: {required} > {total}")


def section_fits(counter: TextCounter, text: str, cap: int, name: str) -> None:
    used = counter.count(text)
    if used > cap:
        raise PromptBudgetError(f"{name} exceeds its region: {used} > {cap}")


def writer_user_text(observation: WriterObservation, *, used: int, capacity: int,
                     n_entries: int, max_entries: int, page: int) -> str:
    return (
        f"Memory status: {used}/{capacity} tokens; {n_entries}/{max_entries} entries\n"
        f"Current index page: {page}\n"
        f"{observation.memory_index}\n"
        "Loaded current values:\n"
        f"{observation.loaded_text}\n"
        "Previous tool result:\n"
        f"{observation.last_tool_result}\n"
        "Current history part:\n"
        f"{observation.chunk.text}"
    )


def render_writer_prompt(observation: WriterObservation, *, system: str, counter: TextCounter,
                         budgets: dict[str, int], used: int, capacity: int,
                         n_entries: int, max_entries: int) -> str:
    """The last-chunk flag is controller state and is excluded from prompt text."""
    user = writer_user_text(
        observation, used=used, capacity=capacity, n_entries=n_entries,
        max_entries=max_entries, page=0,
    )
    section_fits(counter, observation.chunk.text, budgets["chunk"], "chunk")
    section_fits(counter, observation.memory_index, budgets["index"], "index")
    section_fits(counter, observation.loaded_text, budgets["loaded"], "loaded")
    section_fits(counter, observation.last_tool_result, budgets["last_result"], "last_result")
    section_fits(counter, system, budgets["system"], "system")
    prompt = system + "\n" + user
    ensure_prompt_fits(counter.count(prompt), budgets["generation"], budgets["total"])
    return prompt


def reader_prompt(snapshot_text: str, question: str, answer_type: str,
                  question_time: str | None) -> str:
    when = "none" if question_time is None else question_time
    user = (
        f"Question: {question}\n"
        f"Required answer type: {answer_type}\n"
        f"Question date: {when}\n"
        "Memory:\n"
        f"{snapshot_text}\n"
        'Return only one JSON object: {"answer": ...}'
    )
    return READER_SYSTEM + "\n" + user


def reader_prompt_sha256(snapshot_text: str, question: str, answer_type: str,
                         question_time: str | None, *, template_sha256: str,
                         order: str) -> str:
    text = reader_prompt(snapshot_text, question, answer_type, question_time)
    return sha256_text(canonical_json({
        "template_sha256": template_sha256,
        "order": order,
        "text": text,
    }))


def observation_canary_bytes(observation: WriterObservation) -> str:
    payload = observation_payload(observation)
    return canonical_json(payload)
