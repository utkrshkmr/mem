"""Microbenchmarks (PLAN.md §8.13): decode throughput vs concurrency, prefill rate, training tokens/s.

Prompts are writer-shaped (real writer prompts padded to the target length) and made unique by a
nonce at the very start of the system message, so prefix caching cannot hit across requests.
"""
from __future__ import annotations

import time
import uuid

import numpy as np
import torch
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from .env.ledger import make_prefix
from .env.oracle import final_state, render_oracle
from .executor import gather_all
from .prompts import WRITER_SYSTEM, WRITER_USER
from .rl.logprobs import Call

DECODE_NS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def writer_shaped_ids(tok, n_tokens: int, idx: int, B: int = 512, level: int = 5) -> list[int]:
    """A writer prompt with a realistic memory, padded to about n_tokens, unique per idx."""
    p = make_prefix("profile", (99, 0, idx), 16, level)
    memory = render_oracle(final_state(p))
    chunk = p.chunks[-1]
    nonce = f"[req {idx}-{uuid.uuid4().hex[:6]}] "

    def ids(mem):
        msgs = [{"role": "system", "content": nonce + WRITER_SYSTEM.format(B=B)},
                {"role": "user", "content": WRITER_USER.format(memory=mem, chunk=chunk)}]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)

    mem = memory
    out = ids(mem)
    extra = memory.splitlines()
    j = 0
    while len(out) < n_tokens:
        mem = mem + "\n" + extra[j % len(extra)]
        j += 1
        out = ids(mem)
    return out


def _timed_batch(engine, loop, prompts, sp, lora):
    async def run():
        return await gather_all([engine.generate(ids, sp, f"bench|{uuid.uuid4().hex}", lora) for ids in prompts])
    t0 = time.perf_counter()
    res = loop.run_until_complete(run())
    return time.perf_counter() - t0, res


def decode_curve(engine, loop, tok, lora, ns=DECODE_NS, prompt_tokens=900, max_tokens=256) -> dict:
    sp = SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=max_tokens, ignore_eos=True,
                        output_kind=RequestOutputKind.FINAL_ONLY)
    _timed_batch(engine, loop, [writer_shaped_ids(tok, prompt_tokens, 10_000 + j) for j in range(8)], sp, lora)
    thr = {}
    base = 0
    for n in ns:
        prompts = [writer_shaped_ids(tok, prompt_tokens, base + j) for j in range(n)]
        base += n
        dt, res = _timed_batch(engine, loop, prompts, sp, lora)
        thr[n] = sum(len(r.completion_ids) for r in res) / dt
    mx = max(thr.values())
    n_sat = min(n for n, v in thr.items() if v >= 0.8 * mx)
    return {"throughput": {str(k): v for k, v in thr.items()}, "n_sat": n_sat, "decode_rate_sat": mx}


def prefill_rate(engine, loop, tok, lora, n=64, prompt_tokens=1500) -> float:
    sp = SamplingParams(temperature=1.0, max_tokens=1, output_kind=RequestOutputKind.FINAL_ONLY)
    prompts = [writer_shaped_ids(tok, prompt_tokens, 50_000 + j) for j in range(n)]
    dt, res = _timed_batch(engine, loop, prompts, sp, lora)
    uncached = sum(r.num_prompt_tokens - r.num_cached_tokens for r in res)
    return uncached / dt


def train_rate(trainer, tok, max_tokens: int, T_norm: float, n: int = 64, prompt_tokens=900,
               completion_tokens=300) -> float:
    rng = np.random.default_rng(0)
    calls = [Call(writer_shaped_ids(tok, prompt_tokens, 70_000 + j),
                  rng.integers(100, 20_000, completion_tokens).tolist(), float(rng.normal()))
             for j in range(n)]
    trainer.opt.zero_grad(set_to_none=True)
    trainer._accumulate(calls[:4], T_norm, max_tokens)      # warmup
    trainer.opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer._accumulate(calls, T_norm, max_tokens)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    trainer.opt.zero_grad(set_to_none=True)
    return sum(len(c.prompt_ids) + len(c.completion_ids) for c in calls) / dt
