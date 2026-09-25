"""T2.7 helper: compare AsyncReadyExecutor and BarrierExecutor on identical data.

mode a: VLLM_BATCH_INVARIANT=1 + per-request seeds, N=8 K=4 L=4 -> token-identical completions?
mode b: 3 reps of N=32 K=4 L=4 per executor -> completion length and reward statistics.
usage: python equivalence.py {a|b} OUT.json ADAPTER_ROOT
"""
import json
import os
import sys

MODE, OUT, AROOT = sys.argv[1], sys.argv[2], sys.argv[3]
if MODE == "a":
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

from kmatters.hf_auth import load_hf_token  # noqa: E402

load_hf_token()


def calls_by_key(batch):
    return {(c.role, c.i, c.k, c.step): c for c in batch.calls()}


def main():
    import numpy as np
    from kmatters import config
    from kmatters.engine import reader_sp
    from kmatters.loop import _writer_sp, build_context, close_context, init_v0, make_specs
    over = ["env.L=4"] + (["vllm.per_request_seeds=true"] if MODE == "a" else [])
    cfg = config.load(["configs/base.yaml"], over)
    ctx = build_context(cfg, f"equiv_{MODE}", AROOT, split="profile", key0=7070, phase="equiv")
    try:
        init_v0(ctx)
        N, K, reps = (8, 4, 1) if MODE == "a" else (32, 4, 3)
        res = {}
        for ex in ("async_ready", "barrier"):
            ctx.set_executor(ex, phase="equiv")
            res[ex] = []
            for rep in range(reps):
                pspec, fspec = make_specs(ctx, rep, N, K)
                res[ex].append(ctx.loop.run_until_complete(ctx.executor.collect(
                    pspec, fspec, ctx.engine.current, _writer_sp(ctx, rep),
                    reader_sp(cfg.sampling.reader_max_tokens), it=rep)))
        if MODE == "a":
            a, b = calls_by_key(res["async_ready"][0]), calls_by_key(res["barrier"][0])
            same = sum(a[k].completion_ids == b[k].completion_ids for k in a)
            out = {"mode": "a", "n_calls": len(a), "n_identical": same, "identical": same == len(a) == len(b)}
        else:
            def stats(batches):
                lens = [c.n_completion for bt in batches for c in bt.calls() if c.role != "reader"]
                rw = np.concatenate([bt.rewards().ravel() for bt in batches])
                return float(np.mean(lens)), rw
            la, ra = stats(res["async_ready"])
            lb, rb = stats(res["barrier"])
            p = float(np.concatenate([ra, rb]).mean())
            se = float(np.sqrt(max(p * (1 - p), 1e-12) * (1 / len(ra) + 1 / len(rb))))
            out = {"mode": "b", "mean_len_async": la, "mean_len_barrier": lb,
                   "len_rel_diff": abs(la - lb) / ((la + lb) / 2),
                   "reward_async": float(ra.mean()), "reward_barrier": float(rb.mean()),
                   "reward_diff_over_pooled_se": abs(float(ra.mean() - rb.mean())) / se}
    finally:
        close_context(ctx)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
