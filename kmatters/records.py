"""Rollout records (PLAN.md §8.7)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class CallRecord:
    role: str               # "prefix" | "future" | "reader"
    i: int
    k: int                  # -1 for prefix calls
    step: int
    prompt_ids: list[int]
    completion_ids: list[int]
    text: str
    finish_reason: str
    vllm_lp: list[float] | None
    n_prompt: int
    n_cached: int
    n_completion: int
    t_submit: float
    t_end: float

    def summary(self) -> dict:
        """Everything except token ids and per-token logprobs (for rollout logs)."""
        d = asdict(self)
        for k in ("prompt_ids", "completion_ids", "vllm_lp"):
            d.pop(k)
        return d


@dataclass
class FutureRecord:
    i: int
    k: int
    key: tuple
    calls: list[CallRecord]          # U writer calls
    reader: CallRecord
    question: str
    q: tuple
    gold: int
    answer_text: str
    reward: float
    final_memory: str


@dataclass
class PrefixRecord:
    i: int
    key: tuple
    calls: list[CallRecord]          # L writer calls
    memory: str
    futures: list[FutureRecord]      # exactly K


@dataclass
class RolloutBatch:
    prefixes: list[PrefixRecord]
    t_start: float
    t_end: float
    phase_times: dict = field(default_factory=dict)   # barrier executor: t_phase_prefix/future/reader

    @property
    def N(self) -> int:
        return len(self.prefixes)

    @property
    def K(self) -> int:
        return len(self.prefixes[0].futures)

    def rewards(self) -> np.ndarray:
        return np.array([[f.reward for f in p.futures] for p in self.prefixes], dtype=np.float64)

    def calls(self, role: str | None = None) -> list[CallRecord]:
        out = []
        for p in self.prefixes:
            out += p.calls
            for f in p.futures:
                out += f.calls + [f.reader]
        return out if role is None else [c for c in out if c.role == role]

    def to_log_lines(self) -> list[dict]:
        """One dict per prefix, token ids omitted (PLAN.md §12 rollouts/*.jsonl.gz)."""
        lines = []
        for p in self.prefixes:
            lines.append({
                "i": p.i, "key": list(p.key), "calls": [c.summary() for c in p.calls], "memory": p.memory,
                "futures": [{"k": f.k, "key": list(f.key), "calls": [c.summary() for c in f.calls],
                             "reader": f.reader.summary(), "question": f.question, "q": list(f.q),
                             "gold": f.gold, "answer_text": f.answer_text, "reward": f.reward,
                             "final_memory": f.final_memory} for f in p.futures],
            })
        return lines
