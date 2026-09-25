"""Rollout executors (PLAN.md §8.7).

AsyncReadyExecutor: the "competent" executor. All prefix chains run concurrently and each
prefix's K futures start the moment its memory exists. Used everywhere by default.
BarrierExecutor: step-synchronous diagnostic ablation, used only in profiling.

Both return identically structured RolloutBatch objects. A failed request cancels the rest
and propagates (TaskGroup): there is never a partial batch.
"""
from __future__ import annotations

import asyncio
from time import perf_counter

from .engine import request_id
from .prompts import EMPTY_MEMORY, memory_from_output, reader_ids, writer_ids
from .records import CallRecord, FutureRecord, PrefixRecord, RolloutBatch
from .reward import reward


async def gather_all(coros):
    """Like asyncio.gather, but the first failure cancels the remaining tasks."""
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(c) for c in coros]
    return [t.result() for t in tasks]


class _Executor:
    def __init__(self, engine, tok, B: int, run_id: str = "run", phase: str = "train"):
        self.engine, self.tok, self.B = engine, tok, B
        self.run_id, self.phase = run_id, phase

    @staticmethod
    def _sp(sp, role, i, k, step):
        return sp(role, i, k, step) if callable(sp) else sp

    async def _call(self, it, role, i, k, step, ids, sp, lora) -> CallRecord:
        r = await self.engine.generate(ids, self._sp(sp, role, i, k, step),
                                       request_id(self.run_id, self.phase, it, role, i, k, step), lora)
        return CallRecord(role=role, i=i, k=k, step=step, prompt_ids=r.prompt_ids,
                          completion_ids=r.completion_ids, text=r.text, finish_reason=r.finish_reason,
                          vllm_lp=r.vllm_token_logprobs, n_prompt=r.num_prompt_tokens,
                          n_cached=r.num_cached_tokens, n_completion=len(r.completion_ids),
                          t_submit=r.t_submit, t_end=r.t_end)

    async def _writer(self, it, role, i, k, step, mem, chunk, sp, lora) -> CallRecord:
        return await self._call(it, role, i, k, step, writer_ids(self.tok, mem, chunk, self.B), sp, lora)

    async def _reader(self, it, i, k, step, mem, question, sp) -> CallRecord:
        return await self._call(it, "reader", i, k, step, reader_ids(self.tok, mem, question), sp, None)

    @staticmethod
    def _future_record(i, k, fs, calls, rd, mem) -> FutureRecord:
        return FutureRecord(i=i, k=k, key=tuple(fs.key), calls=calls, reader=rd, question=fs.question,
                            q=tuple(fs.q), gold=int(fs.gold), answer_text=rd.text,
                            reward=reward(rd.text, int(fs.gold)), final_memory=mem)


class AsyncReadyExecutor(_Executor):
    async def collect(self, prefix_specs, future_specs, lora, writer_sp, reader_sp, it: int = 0) -> RolloutBatch:
        t0 = perf_counter()

        async def run_future(i, k, mem):
            fs, calls = future_specs[i][k], []
            for u, chunk in enumerate(fs.chunks):
                r = await self._writer(it, "future", i, k, u, mem, chunk, writer_sp, lora)
                calls.append(r)
                mem = memory_from_output(r.text)
            rd = await self._reader(it, i, k, len(fs.chunks), mem, fs.question, reader_sp)
            return self._future_record(i, k, fs, calls, rd, mem)

        async def run_prefix(i):
            ps, mem, calls = prefix_specs[i], EMPTY_MEMORY, []
            for t, chunk in enumerate(ps.chunks):
                r = await self._writer(it, "prefix", i, -1, t, mem, chunk, writer_sp, lora)
                calls.append(r)
                mem = memory_from_output(r.text)
            futs = await gather_all([run_future(i, k, mem) for k in range(len(future_specs[i]))])
            return PrefixRecord(i=i, key=tuple(ps.key), calls=calls, memory=mem, futures=futs)

        prefixes = await gather_all([run_prefix(i) for i in range(len(prefix_specs))])
        return RolloutBatch(prefixes, t0, perf_counter())


class BarrierExecutor(_Executor):
    async def collect(self, prefix_specs, future_specs, lora, writer_sp, reader_sp, it: int = 0) -> RolloutBatch:
        t0 = perf_counter()
        N = len(prefix_specs)
        L = len(prefix_specs[0].chunks)
        mems = [EMPTY_MEMORY] * N
        pcalls = [[] for _ in range(N)]
        for t in range(L):
            recs = await gather_all([self._writer(it, "prefix", i, -1, t, mems[i], prefix_specs[i].chunks[t],
                                                  writer_sp, lora) for i in range(N)])
            for i, r in enumerate(recs):
                pcalls[i].append(r)
                mems[i] = memory_from_output(r.text)
        t1 = perf_counter()

        pairs = [(i, k) for i in range(N) for k in range(len(future_specs[i]))]
        fmem = {(i, k): mems[i] for i, k in pairs}
        fcalls = {p: [] for p in pairs}
        U = len(future_specs[0][0].chunks)
        for u in range(U):
            recs = await gather_all([self._writer(it, "future", i, k, u, fmem[(i, k)],
                                                  future_specs[i][k].chunks[u], writer_sp, lora)
                                     for i, k in pairs])
            for p, r in zip(pairs, recs):
                fcalls[p].append(r)
                fmem[p] = memory_from_output(r.text)
        t2 = perf_counter()

        readers = await gather_all([self._reader(it, i, k, U, fmem[(i, k)], future_specs[i][k].question,
                                                 reader_sp) for i, k in pairs])
        t3 = perf_counter()
        rd = dict(zip(pairs, readers))
        prefixes = [PrefixRecord(i=i, key=tuple(prefix_specs[i].key), calls=pcalls[i], memory=mems[i],
                                 futures=[self._future_record(i, k, future_specs[i][k], fcalls[(i, k)],
                                                              rd[(i, k)], fmem[(i, k)])
                                          for k in range(len(future_specs[i]))])
                    for i in range(N)]
        return RolloutBatch(prefixes, t0, t3, phase_times={"t_phase_prefix": t1 - t0, "t_phase_future": t2 - t1,
                                                           "t_phase_reader": t3 - t2})


def make_executor(name: str, engine, tok, B: int, run_id: str = "run", phase: str = "train"):
    if name == "async_ready":
        return AsyncReadyExecutor(engine, tok, B, run_id, phase)
    if name == "barrier":
        return BarrierExecutor(engine, tok, B, run_id, phase)
    raise ValueError(f"unknown executor {name!r}")
