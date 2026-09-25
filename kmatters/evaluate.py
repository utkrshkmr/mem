"""Evaluation and controls (PLAN.md §8.11).

evaluate(): deterministic writer (temperature 0) with the current adapter and deterministic
reader on the fixed eval set, through the AsyncReadyExecutor. Eval time is logged as t_eval
and excluded from all cost pricing.

Memory-length statistics (here and in training metrics) are over all writer calls:
mem_len = completion tokens of a writer call; trunc_rate = fraction with finish_reason "length".
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from .engine import eval_writer_sp, reader_sp, request_id
from .env.ledger import make_future, make_prefix
from .env.oracle import oracle_memory
from .executor import AsyncReadyExecutor, gather_all
from .prompts import EMPTY_MEMORY, reader_ids
from .reward import reward

QTYPES = ("net", "paid", "cat", "amount")


def eval_specs(eval_cfg, L, U, level):
    pspec = [make_prefix("eval", (eval_cfg.eval_seed, j), L, level) for j in range(eval_cfg.n_prefixes)]
    fspec = [[make_future("eval", p, k, U, level) for k in range(eval_cfg.futures_per_prefix)] for p in pspec]
    return pspec, fspec


def writer_stats(calls) -> dict:
    lens = np.array([c.n_completion for c in calls], float)
    return {"mem_len_mean": float(lens.mean()) if lens.size else None,
            "mem_len_p90": float(np.percentile(lens, 90)) if lens.size else None,
            "trunc_rate": float(np.mean([c.finish_reason == "length" for c in calls])) if calls else None}


def _summary(rewards, qtypes, per_prefix):
    r = np.asarray(rewards, float)
    by_type = {t: (float(r[[q == t for q in qtypes]].mean()) if any(q == t for q in qtypes) else None)
               for t in QTYPES}
    return {"acc": float(r.mean()), "acc_by_type": by_type, "n_questions": int(r.size),
            "per_prefix_correct": per_prefix}


def evaluate(ctx, which: str = "small") -> dict:
    ecfg = getattr(ctx.cfg.eval, which)
    t0 = perf_counter()
    pspec, fspec = eval_specs(ecfg, ctx.L, ctx.U, ctx.level)
    ex = AsyncReadyExecutor(ctx.engine, ctx.tok, ctx.B, ctx.run_id, f"eval_{which}")
    batch = ctx.loop.run_until_complete(ex.collect(pspec, fspec, ctx.engine.current, eval_writer_sp(ctx.B),
                                                   reader_sp(ctx.cfg.sampling.reader_max_tokens)))
    futs = [f for p in batch.prefixes for f in p.futures]
    out = _summary([f.reward for f in futs], [f.q[0] for f in futs],
                   [int(sum(f.reward for f in p.futures)) for p in batch.prefixes])
    writers = [c for c in batch.calls() if c.role != "reader"]
    out.update(writer_stats(writers))
    out["t_eval"] = perf_counter() - t0
    return out


def reader_only_accuracy(ctx, which: str, mode: str) -> dict:
    """Controls: the reader answers from a given memory (oracle | empty | shuffled)."""
    ecfg = getattr(ctx.cfg.eval, which)
    pspec, fspec = eval_specs(ecfg, ctx.L, ctx.U, ctx.level)
    n = len(pspec)
    items = []
    for j in range(n):
        for k, f in enumerate(fspec[j]):
            if mode == "oracle":
                mem = oracle_memory(pspec[j], f)
            elif mode == "empty":
                mem = EMPTY_MEMORY
            elif mode == "shuffled":        # oracle memory of a different eval prefix
                jj = (j + 1) % n
                mem = oracle_memory(pspec[jj], fspec[jj][k % len(fspec[jj])])
            else:
                raise ValueError(mode)
            items.append((j, k, f, mem))
    sp = reader_sp(ctx.cfg.sampling.reader_max_tokens)

    async def run():
        return await gather_all([ctx.engine.generate(reader_ids(ctx.tok, mem, f.question), sp,
                                                     request_id(ctx.run_id, f"control_{mode}", 0, "reader", j, k, 0),
                                                     None) for j, k, f, mem in items])

    res = ctx.loop.run_until_complete(run())
    rewards = [reward(r.text, int(f.gold)) for r, (_, _, f, _) in zip(res, items)]
    per_prefix = [0] * n
    for rw, (j, _, _, _) in zip(rewards, items):
        per_prefix[j] += int(rw)
    out = _summary(rewards, [f.q[0] for _, _, f, _ in items], per_prefix)
    if mode == "oracle":
        lens = [len(ctx.tok(mem, add_special_tokens=False)["input_ids"]) for _, _, _, mem in items]
        out["oracle_tokens"] = {"mean": float(np.mean(lens)), "p95": float(np.percentile(lens, 95)),
                                "max": int(max(lens)), "frac_le_B": float(np.mean([l <= ctx.B for l in lens]))}
    return out
