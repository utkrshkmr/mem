"""V_prefix and V_future at a checkpoint (PLAN.md §8.12).

Z_ik = (r_ik - b_i) * (g_pre_i + g_fut_ik), with b_i the leave-one-prefix-out mean (E1) and
g_* = grad of (1/T_norm) sum log pi over the calls of that prefix / future. Then
mean_ik Z_ik equals minus the gradient of the training loss for the same batch (T2.9).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .analysis.variance_math import TorchVarianceAccumulator, VarianceAccumulator
from .engine import reader_sp, train_writer_sp
from .loop import make_specs


def loo_baseline(r: np.ndarray) -> np.ndarray:
    rbar = r.mean(axis=1)
    return (rbar.sum() - rbar) / (len(rbar) - 1)


def iter_Z(trainer, batch, T_norm, max_tokens):
    """Yield (i, [Z_i0 .. Z_iK-1]) as flat fp32 GPU tensors, one prefix at a time."""
    r = batch.rewards()
    b = loo_baseline(r)
    for p in batch.prefixes:
        g_pre = trainer.score_grad([(c.prompt_ids, c.completion_ids) for c in p.calls], T_norm, max_tokens)
        Z = []
        for f in p.futures:
            g_fut = trainer.score_grad([(c.prompt_ids, c.completion_ids) for c in f.calls], T_norm, max_tokens)
            Z.append(float(r[p.i, f.k] - b[p.i]) * (g_pre + g_fut))
            del g_fut
        del g_pre
        yield p.i, Z


def _pcts(x):
    return {"p2_5": float(np.percentile(x, 2.5)), "p50": float(np.percentile(x, 50)),
            "p97_5": float(np.percentile(x, 97.5))}


def collect_pool(ctx, ckpt_it: int, N0: int, K0: int):
    """N0 prefixes x K0 futures with the training writer temperature (split "variance",
    keys (vseed, ckpt_it, i) with ctx.key0 = vseed)."""
    pspec, fspec = make_specs(ctx, ckpt_it, N0, K0)
    return ctx.loop.run_until_complete(ctx.executor.collect(
        pspec, fspec, ctx.engine.current, train_writer_sp(ctx.B),
        reader_sp(ctx.cfg.sampling.reader_max_tokens), it=ckpt_it))


def measure(ctx, ckpt_label: str, ckpt_it: int, out_dir, N0: int, K0: int, n_boot: int = 2000,
            store_device: str | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batch = collect_pool(ctx, ckpt_it, N0, K0)
    r = batch.rewards()
    np.save(out / "rewards.npy", r)
    # no more generation: free the engine's memory before the backward passes
    ctx.loop.run_until_complete(ctx.engine.shutdown())
    torch.cuda.empty_cache()

    cfg = ctx.cfg
    acc = TorchVarianceAccumulator(store_device=store_device)
    for _, Z in iter_Z(ctx.trainer, batch, cfg.estimator.T_norm, cfg.train.max_tokens_per_microbatch):
        acc.add_prefix(Z)
        del Z
    Vp, Vf = acc.estimates()
    boot = acc.bootstrap(B=n_boot, seed=0)
    np.save(out / "gram.npy", acc.gram())
    np.save(out / "within.npy", np.array(acc.within))
    np.save(out / "boot.npy", boot)

    racc = VarianceAccumulator()
    for i in range(r.shape[0]):
        racc.add_prefix([np.array([v]) for v in r[i]])
    rVp, rVf = racc.estimates()

    res = {"ckpt": ckpt_label, "it": ckpt_it, "L": ctx.L, "B": ctx.B, "N0": N0, "K0": K0, "vseed": ctx.key0,
           "reward_mean": float(r.mean()),
           "V_prefix": {"est": Vp, **_pcts(boot[:, 0])}, "V_future": {"est": Vf, **_pcts(boot[:, 1])},
           "reward_level": {"V_prefix": rVp, "V_future": rVf},
           "frac_Vp_nonpositive_boot": float(np.mean(boot[:, 0] <= 0)),
           "n_trainable": ctx.trainer.n_trainable, "config_hash": cfg.config_hash}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    return res
