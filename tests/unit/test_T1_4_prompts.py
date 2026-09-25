"""T1.4: prompt builders return list[int]; no leakage of futures/questions into writer prompts,
and no log lines in reader prompts outside the memory string.

Worst-case memories are used: the "memory" is every log line the writer has seen so far, so
anything a real writer could leak would show up here."""
import re
from pathlib import Path

import pytest

from kmatters.env.ledger import make_future, make_prefix
from kmatters.prompts import EMPTY_MEMORY, reader_ids, writer_ids

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Qwen3-4B-Instruct-2507"
LOG_LINE = re.compile(r"^T\d+:|was cancelled|Correction:", re.M)
N_PREFIXES, N_FUT, L, U, B, LEVEL = 50, 3, 16, 2, 512, 3


@pytest.fixture(scope="module")
def tok():
    if not (MODEL_DIR / "tokenizer.json").exists():
        pytest.fail(f"tokenizer not found at {MODEL_DIR}; run scripts/02_download_model.py")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def _decode(tok, ids):
    assert isinstance(ids, list) and ids and all(type(t) is int for t in ids)
    return tok.decode(ids, skip_special_tokens=False)


def test_T1_4_no_leakage(tok, record):
    n_writer = n_reader = 0
    for j in range(N_PREFIXES):
        p = make_prefix("train", (0, 0, j), L=L, level=LEVEL)
        futs = [make_future("train", p, k, U=U, level=LEVEL) for k in range(N_FUT)]
        prefix_lines = set("\n".join(p.chunks).splitlines())
        # future lines that do not also occur in the prefix (e.g. a repeated noise sentence)
        fut_only = {ln for f in futs for ln in "\n".join(f.chunks).splitlines()} - prefix_lines
        questions = [f.question for f in futs]

        memory = EMPTY_MEMORY
        for chunk in p.chunks:
            s = _decode(tok, writer_ids(tok, memory, chunk, B))
            n_writer += 1
            assert not any(ln in s for ln in fut_only), "future chunk line in a prefix-call prompt"
            assert not any(q in s for q in questions), "question text in a writer prompt"
            memory = chunk if memory == EMPTY_MEMORY else memory + "\n" + chunk
        prefix_memory = memory

        for f in futs:
            memory = prefix_memory
            for chunk in f.chunks:
                s = _decode(tok, writer_ids(tok, memory, chunk, B))
                n_writer += 1
                assert not any(q in s for q in questions), "question text in a future writer prompt"
                memory = memory + "\n" + chunk
            s = _decode(tok, reader_ids(tok, memory, f.question))
            n_reader += 1
            assert memory in s and f.question in s
            outside = s.replace(memory, "", 1)
            assert not LOG_LINE.search(outside), "log line in a reader prompt outside the memory"
    record("T1.4", n_prefixes=N_PREFIXES, futures_per_prefix=N_FUT, n_writer_prompts=n_writer,
           n_reader_prompts=n_reader)
