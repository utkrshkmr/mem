"""vLLM wrapper (PLAN.md §8.6): one AsyncLLM per process/GPU serves the writer (LoRA) and the
reader (base model, no LoRA).

Process and event-loop pattern (mandatory):
  load_hf_token(); loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
  engine = Engine(cfg, loop); loop.run_until_complete(engine.start())   # before the HF model
  ... every later phase: loop.run_until_complete(coro); never asyncio.run() twice.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


@dataclass
class GenResult:
    prompt_ids: list[int]
    completion_ids: list[int]          # includes EOS when the call stopped on EOS
    text: str
    finish_reason: str
    vllm_token_logprobs: list[float] | None
    num_prompt_tokens: int
    num_cached_tokens: int
    t_submit: float
    t_end: float


def train_writer_sp(B: int, seed: int | None = None) -> SamplingParams:
    """Exact on-policy sampling: temperature 1, no truncation, sampled-token logprobs."""
    return SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=B, logprobs=0, seed=seed,
                          output_kind=RequestOutputKind.FINAL_ONLY)


def eval_writer_sp(B: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=B, output_kind=RequestOutputKind.FINAL_ONLY)


def reader_sp(max_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, output_kind=RequestOutputKind.FINAL_ONLY)


def request_id(run_id, phase, it, role, i, k, step) -> str:
    return f"{run_id}|{phase}|{it}|{role}|{i}|{k}|{step}|{uuid.uuid4().hex[:8]}"


class Engine:
    """One AsyncLLM per process/GPU."""

    def __init__(self, cfg, loop: asyncio.AbstractEventLoop, **arg_overrides):
        self.cfg = cfg
        self.loop = loop
        self.arg_overrides = arg_overrides
        self.llm: AsyncLLM | None = None
        self.current: LoRARequest | None = None      # writer adapter used for writer calls
        self.version: int | None = None

    def engine_args(self) -> AsyncEngineArgs:
        c, v = self.cfg, self.cfg.vllm
        kw = dict(
            model=c.model.local_dir, tokenizer=c.model.local_dir, dtype="bfloat16",
            seed=c.train.seed, max_model_len=v.max_model_len,
            enable_lora=True, max_loras=v.max_loras, max_lora_rank=v.max_lora_rank,
            max_cpu_loras=v.max_cpu_loras, enable_prefix_caching=v.enable_prefix_caching,
            max_num_seqs=v.max_num_seqs, max_num_batched_tokens=v.max_num_batched_tokens,
            kv_cache_memory_bytes=v.kv_cache_memory_bytes, enforce_eager=v.enforce_eager,
            disable_log_stats=False)
        kw.update(self.arg_overrides)
        return AsyncEngineArgs(**kw)

    async def start(self):
        self.llm = AsyncLLM.from_engine_args(self.engine_args())
        return self

    async def generate(self, ids, sp: SamplingParams, rid: str, lora: LoRARequest | None) -> GenResult:
        assert sp.output_kind == RequestOutputKind.FINAL_ONLY
        ids = list(ids)
        t_submit = time.perf_counter()
        final = None
        async for out in self.llm.generate(TokensPrompt(prompt_token_ids=ids), sp, request_id=rid,
                                           lora_request=lora):
            final = out
        t_end = time.perf_counter()
        if final is None or not final.finished:
            raise RuntimeError(f"request {rid} returned no final output")
        c = final.outputs[0]
        comp = list(c.token_ids)
        lps = None
        if sp.logprobs is not None:
            if c.logprobs is None or len(c.logprobs) != len(comp):
                raise RuntimeError(f"request {rid}: missing sampled-token logprobs")
            lps = [float(c.logprobs[t][tid].logprob) for t, tid in enumerate(comp)]
        if list(final.prompt_token_ids) != ids:
            raise RuntimeError(f"request {rid}: vLLM prompt ids differ from the submitted ids")
        return GenResult(prompt_ids=ids, completion_ids=comp, text=c.text, finish_reason=str(c.finish_reason),
                         vllm_token_logprobs=lps, num_prompt_tokens=len(ids),
                         num_cached_tokens=int(final.num_cached_tokens or 0), t_submit=t_submit, t_end=t_end)

    async def set_writer_adapter(self, path: str, version: int):
        """Load adapter `version` (lora_int_id = version + 1) and drop the previous one."""
        new = LoRARequest(lora_name=f"v{version}", lora_int_id=version + 1, lora_path=str(path))
        await self.llm.add_lora(new)
        prev = self.current
        self.current, self.version = new, version
        if prev is not None and prev.lora_int_id != new.lora_int_id:
            await self.llm.remove_lora(prev.lora_int_id)

    async def score_tokens(self, prompt_ids, completion_ids, lora: LoRARequest | None) -> list[float]:
        """Per-token logprobs of `completion_ids` given `prompt_ids` (tests only)."""
        sp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0,
                            output_kind=RequestOutputKind.FINAL_ONLY)
        full = list(prompt_ids) + list(completion_ids)
        final = None
        async for out in self.llm.generate(TokensPrompt(prompt_token_ids=full), sp,
                                           request_id=f"score|{uuid.uuid4().hex}", lora_request=lora):
            final = out
        pl = final.prompt_logprobs
        P = len(prompt_ids)
        return [float(pl[P + t][tid].logprob) for t, tid in enumerate(completion_ids)]

    async def shutdown(self):
        if self.llm is not None:
            self.llm.shutdown()
            self.llm = None
            self.current = None
