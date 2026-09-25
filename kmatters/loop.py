"""One iteration (the single code path for training AND profiling) and the training run
(PLAN.md §8.10).

Counting convention: iteration index `it` runs 0..I_max-1, samples with policy version `it`
and produces version `it + 1`. "Checkpoint n" is the state after n completed iterations.
Evaluations are attached to versions in evals.jsonl; a metrics line also carries the eval of
the version its iteration produced (eval_version = it + 1) when one ran.
"""
from __future__ import annotations

import asyncio
import datetime
import gzip
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .config import save_resolved
from .engine import Engine, reader_sp, train_writer_sp
from .env.ledger import make_future, make_prefix
from .executor import make_executor
from .rl.estimator import advantages, call_weights
from .rl.logprobs import Call
from .seeding import seed_from

log = logging.getLogger(__name__)
MAX_ROLLOUT_ATTEMPTS = 3        # first attempt + at most two retries with the same data keys


@dataclass
class Ctx:
    cfg: object
    loop: asyncio.AbstractEventLoop
    engine: Engine
    tok: object
    trainer: object
    executor: object
    run_id: str
    split: str
    key0: int
    L: int
    U: int
    B: int
    level: int
    adapter_root: Path
    version: int = 0
    nvml: object = None
    last_batch: object = None
    last_windows: dict = field(default_factory=dict)

    def adir(self, v: int) -> Path:
        return self.adapter_root / f"v{v}"

    def set_executor(self, name: str, phase: str = "train"):
        self.executor = make_executor(name, self.engine, self.tok, self.B, self.run_id, phase)


def build_context(cfg, run_id: str, adapter_root, split: str = "train", key0: int | None = None,
                  executor: str | None = None, phase: str = "train", nvml: bool = True,
                  engine_overrides: dict | None = None, with_trainer: bool = True) -> Ctx:
    """Persistent event loop, then the vLLM engine (clean GPU), then the HF trainer."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    engine = Engine(cfg, loop, **(engine_overrides or {}))
    loop.run_until_complete(engine.start())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.local_dir)
    trainer = None
    if with_trainer:
        from .trainer import Trainer
        trainer = Trainer(cfg, pad_id=tok.pad_token_id)
    sampler = None
    if nvml:
        from .profiling import NvmlSampler
        sampler = NvmlSampler().start()
    ctx = Ctx(cfg=cfg, loop=loop, engine=engine, tok=tok, trainer=trainer, executor=None, run_id=run_id,
              split=split, key0=cfg.train.seed if key0 is None else key0, L=cfg.env.L, U=cfg.env.U,
              B=cfg.env.B, level=cfg.env.level, adapter_root=Path(adapter_root), nvml=sampler)
    ctx.set_executor(executor or cfg.train.executor, phase)
    return ctx


def close_context(ctx: Ctx):
    if ctx.nvml is not None:
        ctx.nvml.stop()
    if ctx.engine.llm is not None:
        ctx.loop.run_until_complete(ctx.engine.shutdown())


def load_adapter(ctx: Ctx, path, version: int):
    """Put adapter `path` into the trainer and vLLM as `version`."""
    if ctx.trainer is not None:
        ctx.trainer.load_adapter_weights(path)
    ctx.loop.run_until_complete(ctx.engine.set_writer_adapter(path, version))
    ctx.version = version


def init_v0(ctx: Ctx):
    """Adapter v0 (LoRA B = 0: writer == base model) is saved and loaded like any other version."""
    ctx.adir(0).parent.mkdir(parents=True, exist_ok=True)
    ctx.trainer.save_adapter(ctx.adir(0))
    ctx.loop.run_until_complete(ctx.engine.set_writer_adapter(ctx.adir(0), 0))
    ctx.version = 0


def _writer_sp(ctx: Ctx, it: int):
    if not ctx.cfg.vllm.per_request_seeds:
        return train_writer_sp(ctx.B)

    def per_request(role, i, k, step):
        return train_writer_sp(ctx.B, seed=seed_from(ctx.split, ctx.key0, it, role, i, k, step) % 2**31)
    return per_request


def make_specs(ctx: Ctx, it: int, N: int, K: int):
    pspec = [make_prefix(ctx.split, (ctx.key0, it, i), ctx.L, ctx.level) for i in range(N)]
    fspec = [[make_future(ctx.split, pspec[i], k, ctx.U, ctx.level) for k in range(K)] for i in range(N)]
    return pspec, fspec


def run_iteration(ctx: Ctx, it: int, N: int, K: int, lr_override=None) -> dict:
    """The single code path for training AND profiling.
    ctx.split/ctx.key0 = ("train", seed) in training, ("profile", cell_id) in profiling (it = rep)."""
    cfg = ctx.cfg
    torch.cuda.reset_peak_memory_stats()
    for attempt in range(MAX_ROLLOUT_ATTEMPTS):
        t0 = perf_counter()
        pspec, fspec = make_specs(ctx, it, N, K)
        tr0 = perf_counter()
        try:
            batch = ctx.loop.run_until_complete(ctx.executor.collect(
                pspec, fspec, ctx.engine.current, _writer_sp(ctx, it),
                reader_sp(cfg.sampling.reader_max_tokens), it=it))
            break
        except Exception:
            log.exception("rollout failed (iteration %d, attempt %d/%d)", it, attempt + 1, MAX_ROLLOUT_ATTEMPTS)
            if attempt == MAX_ROLLOUT_ATTEMPTS - 1:
                raise
    t1 = perf_counter()
    r = batch.rewards()                                          # [N, K]
    w_pre, w_fut = call_weights(r, cfg.estimator.baseline)
    calls, roles, recs = [], [], []
    for p in batch.prefixes:
        for c in p.calls:
            calls.append(Call(c.prompt_ids, c.completion_ids, float(w_pre[p.i])))
            roles.append("prefix")
            recs.append(c)
    for p in batch.prefixes:                                     # reader calls are never trained
        for f in p.futures:
            for c in f.calls:
                calls.append(Call(c.prompt_ids, c.completion_ids, float(w_fut[f.i, f.k])))
                roles.append("future")
                recs.append(c)
    tw = perf_counter()
    stats = ctx.trainer.step(calls, cfg.estimator.T_norm, cfg.train.max_tokens_per_microbatch,
                             lr=lr_override, roles=roles)
    t2 = perf_counter()
    v_new = ctx.version + 1
    ctx.trainer.save_adapter(ctx.adir(v_new))
    ctx.loop.run_until_complete(ctx.engine.set_writer_adapter(ctx.adir(v_new), v_new))
    ctx.version = v_new
    t3 = perf_counter()

    ctx.last_batch = batch
    ctx.last_windows = {"iter": (t0, t3), "rollout": (tr0, t1), "train": (tw, t2), "sync": (t2, t3)}
    times = {"t_rollout": t1 - tr0, "t_train": t2 - tw, "t_sync": t3 - t2, "t_iter": t3 - t0}
    times["t_other"] = times["t_iter"] - times["t_rollout"] - times["t_train"] - times["t_sync"]
    return iteration_metrics(ctx, it, N, K, batch, r, stats, recs, times, rollout_attempts=attempt + 1)


def token_counts(batch) -> dict:
    out = {}
    for role in ("prefix", "future", "reader"):
        cs = batch.calls(role)
        out[role] = {"prompt": int(sum(c.n_prompt for c in cs)), "cached": int(sum(c.n_cached for c in cs)),
                     "completion": int(sum(c.n_completion for c in cs))}
    return out


def iteration_metrics(ctx, it, N, K, batch, r, stats, recs, times, rollout_attempts=1) -> dict:
    cfg = ctx.cfg
    A_pre, A_fut = advantages(r, cfg.estimator.baseline)
    rbar = r.mean(axis=1)
    lp_hf = np.concatenate([t.numpy() for t in stats["tok_lps"]])
    lp_v = np.concatenate([np.asarray(c.vllm_lp, dtype=np.float64) for c in recs])
    d = lp_hf - lp_v
    lens = np.array([c.n_completion for c in recs], float)
    m = {
        "it": it, "policy_version": ctx.version - 1, "K": K, "N": N, "F": N * K, "L": ctx.L, "U": ctx.U,
        "B": ctx.B, "level": ctx.level,
        "reward_mean": float(r.mean()), "reward_std": float(r.std()), "rbar_std": float(rbar.std()),
        "frac_prefix_all0": float(np.mean(rbar == 0)), "frac_prefix_all1": float(np.mean(rbar == 1)),
        "adv_pre_abs_mean": float(np.abs(A_pre).mean()), "adv_fut_abs_mean": float(np.abs(A_fut).mean()),
        "n_calls": {role: len(batch.calls(role)) for role in ("prefix", "future", "reader")},
        "tokens": token_counts(batch),
        "mem_len_mean": float(lens.mean()), "mem_len_p90": float(np.percentile(lens, 90)),
        "trunc_rate": float(np.mean([c.finish_reason == "length" for c in recs])),
        "loss": stats["loss"], "grad_norm": stats["grad_norm"], "clipped": stats["clipped"],
        "opt_step": ctx.trainer.optimizer_step_count(),
        "lp_absdiff_mean": float(np.abs(d).mean()), "lp_diff_mean": float(d.mean()),
        "kl_k3": float(np.mean(np.exp(d) - 1 - d)), "neg_lp_mean": float(-lp_v.mean()),
        **times, "t_train_prefix": stats["t_train_prefix"], "t_train_future": stats["t_train_future"],
        "t_opt": stats["t_opt"], "rollout_attempts": rollout_attempts,
        "torch_mem_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        "wall_time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    m.update(batch.phase_times)
    if ctx.nvml is not None:
        w = ctx.last_windows
        m["gpu_mem_peak_gb"] = ctx.nvml.summary(*w["iter"])["mem_used_peak_gb"]
        m["nvml"] = {k: ctx.nvml.summary(*w[k]) for k in ("rollout", "train")}
        ctx.nvml.prune(w["iter"][0])
    else:
        m["gpu_mem_peak_gb"] = None
    return m


# =========================================================================== training run
def _latest_complete_ckpt(run_dir: Path):
    cks = sorted((run_dir / "ckpt").glob("it*/COMPLETE"))
    return cks[-1].parent if cks else None


def _truncate_lines(path: Path, n: int):
    if path.exists():
        lines = path.read_text().splitlines(keepends=True)[:n]
        path.write_text("".join(lines))


def _count_lines(path: Path) -> int:
    return len(path.read_text().splitlines()) if path.exists() else 0


def _append(path: Path, rec: dict):
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True, default=float) + "\n")


def checkpoint(ctx: Ctx, run_dir: Path, n: int):
    cfg = ctx.cfg
    d = run_dir / "ckpt" / f"it{n:04d}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copytree(ctx.adir(n), d / "adapter")
    ctx.trainer.save_state(d / "optim.pt")
    (d / "state.json").write_text(json.dumps({
        "iteration": n, "policy_version": n, "config_hash": cfg.config_hash,
        "metrics_lines": _count_lines(run_dir / "metrics.jsonl"),
        "evals_lines": _count_lines(run_dir / "evals.jsonl"),
        "opt_step": ctx.trainer.optimizer_step_count(), "rng_states_in": "optim.pt",
    }, indent=2))
    (d / "COMPLETE").touch()
    keep = cfg.train.keep_optimizer_ckpts
    olds = sorted(p for p in (run_dir / "ckpt").glob("it*/optim.pt"))
    for p in olds[:-keep] if keep > 0 else olds:
        p.unlink()


def _adapter_hygiene(ctx: Ctx, n: int):
    """Keep adapters v{m} with m % ckpt_every == 0 and the latest two; the others go once v{n} is loaded."""
    every = ctx.cfg.train.ckpt_every
    for p in ctx.adapter_root.glob("v*"):
        m = int(p.name[1:])
        if m % every != 0 and m < n - 1:
            shutil.rmtree(p, ignore_errors=True)


def _run_evals(ctx: Ctx, run_dir: Path, n: int, I_max: int) -> dict:
    out = {"eval_small": None, "eval_large": None, "t_eval": None}
    t_eval = 0.0
    todo = []
    if n == 0 or n % ctx.cfg.train.eval_every == 0 or n == I_max:
        todo.append("small")
    if n in (0, I_max // 2, I_max):
        todo.append("large")
    from .evaluate import evaluate
    for which in todo:
        res = evaluate(ctx, which)
        t_eval += res["t_eval"]
        _append(run_dir / "evals.jsonl", {"version": n, "which": which, **res})
        out[f"eval_{which}"] = res
    if todo:
        out["t_eval"] = t_eval
    return out


def train(cfg, tag: str, resume: bool = False, runs_root: str | None = None) -> Path:
    from .logging_utils import setup_logging
    run_id = cfg.run_id(tag)
    run_dir = Path(runs_root or cfg.paths.runs) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    if (run_dir / "DONE").exists():
        log.info("run %s already DONE", run_id)
        return run_dir
    save_resolved(cfg, run_dir)
    metrics_path, evals_path = run_dir / "metrics.jsonl", run_dir / "evals.jsonl"
    ck = _latest_complete_ckpt(run_dir)
    if metrics_path.exists() and not resume:
        raise RuntimeError(f"{run_dir} already has metrics; pass --resume to continue it")

    ctx = build_context(cfg, run_id, run_dir / "adapters", split="train", key0=cfg.train.seed)
    I_max, N, K = cfg.train.iterations, cfg.train.N, cfg.train.K
    if resume and ck is not None:
        st = json.loads((ck / "state.json").read_text())
        n = int(st["iteration"])
        if st["config_hash"] != cfg.config_hash:
            raise RuntimeError("checkpoint config hash differs from the current config")
        ctx.trainer.load_state(ck / "adapter", ck / "optim.pt")
        if not ctx.adir(n).exists():
            shutil.copytree(ck / "adapter", ctx.adir(n))
        ctx.loop.run_until_complete(ctx.engine.set_writer_adapter(ctx.adir(n), n))
        ctx.version = n
        _truncate_lines(metrics_path, st["metrics_lines"])
        _truncate_lines(evals_path, st["evals_lines"])
        start = n
        _append(run_dir / "resumes.jsonl", {"from_ckpt": n, "opt_step": ctx.trainer.optimizer_step_count(),
                                            "at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        log.info("resumed %s from checkpoint %d (opt_step=%d)", run_id, n, ctx.trainer.optimizer_step_count())
    else:
        for p in (metrics_path, evals_path):
            p.unlink(missing_ok=True)
        init_v0(ctx)
        _run_evals(ctx, run_dir, 0, I_max)
        start = 0

    (run_dir / "rollouts").mkdir(exist_ok=True)
    try:
        for it in range(start, I_max):
            m = run_iteration(ctx, it, N, K)
            n = it + 1
            m.update(run_id=run_id, seed=cfg.train.seed, lr=cfg.train.lr, config_hash=cfg.config_hash,
                     timing_contaminated=True)
            with gzip.open(run_dir / "rollouts" / f"it{it:04d}.jsonl.gz", "wt") as f:
                for line in ctx.last_batch.to_log_lines():
                    f.write(json.dumps(line) + "\n")
            _adapter_hygiene(ctx, n)
            ev = _run_evals(ctx, run_dir, n, I_max)
            m.update(ev, eval_version=n if ev["t_eval"] is not None else None)
            _append(metrics_path, m)
            log.info("it=%d reward=%.3f loss=%.4g gn=%.3g t_iter=%.1fs lp_absdiff=%.4f", it, m["reward_mean"],
                     m["loss"], m["grad_norm"], m["t_iter"], m["lp_absdiff_mean"])
            if n % cfg.train.ckpt_every == 0 or n == I_max:
                checkpoint(ctx, run_dir, n)
        (run_dir / "DONE").touch()
    finally:
        close_context(ctx)
    return run_dir
