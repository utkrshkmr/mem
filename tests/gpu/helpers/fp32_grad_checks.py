"""T2.8 / T2.9 correctness checks in fp32 on the real model (see prereg/DEVIATIONS.md).

Real calls come from a vLLM rollout; the engine is then shut down and the HF model is loaded in
fp32, where results do not depend on microbatch shapes.
usage: python fp32_grad_checks.py OUT.json ADAPTER_ROOT
"""
import json
import sys

from kmatters.hf_auth import load_hf_token

load_hf_token()


def main():
    OUT, AROOT = sys.argv[1], sys.argv[2]
    import numpy as np
    import torch
    from kmatters import config
    from kmatters.engine import reader_sp, train_writer_sp
    from kmatters.loop import build_context, make_specs
    from kmatters.rl.estimator import call_weights
    from kmatters.rl.logprobs import Call, flat_grad
    from kmatters.trainer import Trainer
    from kmatters.variance import collect_pool, iter_Z

    cfg = config.load(["configs/base.yaml"], ["env.L=4"])
    ctx = build_context(cfg, "fp32chk", AROOT, split="profile", key0=8080, with_trainer=False, nvml=False)
    pspec, fspec = make_specs(ctx, 8, 8, 2)
    b8 = ctx.loop.run_until_complete(ctx.executor.collect(pspec, fspec, None, train_writer_sp(ctx.B),
                                                          reader_sp(cfg.sampling.reader_max_tokens), it=8))
    writers = [c for c in b8.calls() if c.role != "reader"][:48]
    ctx.split, ctx.key0 = "variance", 0
    pool = collect_pool(ctx, 0, 4, 4)
    ctx.loop.run_until_complete(ctx.engine.shutdown())

    tr = Trainer(cfg, pad_id=ctx.tok.pad_token_id, dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():                        # nonzero LoRA B: exercise the A and B gradient paths
        for n, p in tr.model.named_parameters():
            if "lora_B" in n:
                p.copy_((torch.randn(p.shape, generator=gen) * 1e-3).to(p.device))
    cos = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm()))
    T_norm, mt = cfg.estimator.T_norm, cfg.train.max_tokens_per_microbatch

    rng = np.random.default_rng(1)
    calls = [Call(c.prompt_ids, c.completion_ids, float(rng.normal())) for c in writers]
    grads = {}
    for budget in (32768, 4096):
        tr.opt.zero_grad(set_to_none=True)
        tr._accumulate(calls, T_norm, budget)
        grads[budget] = flat_grad(tr.model).clone()
    tr.opt.zero_grad(set_to_none=True)
    g1, g2 = grads[32768], grads[4096]
    out = {"T2.8": {"cosine": cos(g1, g2), "rel_l2": float((g1 - g2).norm() / g1.norm()), "n_calls": len(calls)}}

    Zsum, n = None, 0
    for _, Z in iter_Z(tr, pool, T_norm, mt):
        for z in Z:
            Zsum = z.clone() if Zsum is None else Zsum + z
            n += 1
    meanZ = Zsum / n
    r = pool.rewards()
    w_pre, w_fut = call_weights(r)
    tcalls, roles = [], []
    for p in pool.prefixes:
        for c in p.calls:
            tcalls.append(Call(c.prompt_ids, c.completion_ids, float(w_pre[p.i])))
            roles.append("prefix")
        for f in p.futures:
            for c in f.calls:
                tcalls.append(Call(c.prompt_ids, c.completion_ids, float(w_fut[f.i, f.k])))
                roles.append("future")
    tr.step(tcalls, T_norm, mt, lr=0.0, roles=roles)
    g = flat_grad(tr.model).clone()
    out["T2.9"] = {"cosine": cos(meanZ, -g), "rel_l2": float((meanZ + g).norm() / g.norm()),
                   "reward_mean": float(r.mean()), "grad_norm": float(g.norm())}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
