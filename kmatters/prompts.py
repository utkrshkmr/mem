"""Exact prompt text (PLAN.md §8.4) and token-id builders.

Leakage rules (T1.4): writer prompts contain only the current memory and one chunk;
reader prompts contain only the memory and the question.
"""
from __future__ import annotations

WRITER_SYSTEM = """You maintain the memory of a bookkeeping assistant. You read a ledger log one chunk at a time and never see earlier chunks again: after each chunk, only your memory is kept.

Later, someone who sees ONLY your memory must answer questions such as:
- a person's net balance (total received minus total paid),
- the total amount a person has paid,
- the total amount in a spending category,
- the current amount of a specific transaction (for example T07).
More log chunks may arrive before a question, including cancellations and amount corrections of earlier transactions, so keep whatever details are needed to apply them correctly.

Rules: output only the updated memory, nothing else. Text beyond {B} tokens is cut off, so stay well under that. Ignore log lines that are not transactions, cancellations or corrections."""

WRITER_USER = """CURRENT MEMORY:
{memory}

NEW LOG CHUNK:
{chunk}

Write the updated memory."""

READER_SYSTEM = """You answer questions about a ledger using only the notes provided. Reply with a single integer (use a minus sign for negative numbers) and nothing else. If the notes do not contain the information, reply with your best guess as a single integer."""

READER_USER = """NOTES:
{memory}

QUESTION: {question}"""

EMPTY_MEMORY = "(empty)"


def writer_messages(memory: str, chunk: str, B: int) -> list[dict]:
    return [{"role": "system", "content": WRITER_SYSTEM.format(B=B)},
            {"role": "user", "content": WRITER_USER.format(memory=memory, chunk=chunk)}]


def reader_messages(memory: str, question: str) -> list[dict]:
    return [{"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": READER_USER.format(memory=memory, question=question)}]


def _to_ids(tok, msgs) -> list[int]:
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    if not (isinstance(ids, list) and all(isinstance(t, int) for t in ids)):
        raise TypeError(f"apply_chat_template returned {type(ids).__name__}, expected list[int]")
    return ids


def writer_ids(tok, memory: str, chunk: str, B: int) -> list[int]:
    return _to_ids(tok, writer_messages(memory, chunk, B))


def reader_ids(tok, memory: str, question: str) -> list[int]:
    return _to_ids(tok, reader_messages(memory, question))


def memory_from_output(text: str) -> str:
    """A truncated output (finish_reason == "length") is kept as is."""
    s = text.strip()
    return s if s else EMPTY_MEMORY
