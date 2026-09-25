"""T2 GPU integration tests that share one engine + trainer on one GPU (Gate G1).

T2.1-T2.6, T2.8-T2.11, T2.15. The process-level tests (T2.7, T2.12-T2.14, T2.16) are in
test_T2b_processes.py and run after this module has released the GPU.
"""
import shutil
import statistics
from pathlib import Path

import numpy as np
import pytest
import torch

from kmatters import config
from kmatters.engine import reader_sp, request_id, train_writer_sp
from kmatters.executor import AsyncReadyExecutor, BarrierExecutor
from kmatters.loop import build_context, close_context, init_v0, make_specs, run_iteration
from kmatters.prompts import EMPTY_MEMORY, reader_ids, writer_ids
from kmatters.rl.estimator import call_weights
from kmatters.rl.logprobs import Call, completion_logprob_sums, flat_grad

pytestmark = pytest.mark.gpu
REPO = Path(__file__).resolve().parents[2]
T2_DIR = REPO / "runs" / "_tests" / "T2_inprocess"


@pytest.fixture(scope="module")
def ctx():
    shutil.rmtree(T2_DIR, ignore_errors=True)
    cfg = config.load([REPO / "configs" / "base.yaml"], ["env.L=4"])
    c = build_context(cfg, "t2", T2_DIR / "adapters", split="profile", key0=4242)
    init_v0(c)
    yield c
    close_context(c)
    # release the trainer's GPU memory for the process-level tests that follow in this session
    import gc
    c.trainer = None
    gc.collect()
    torch.cuda.empty_cache()


def configure(c, L=4, B=512, split="profile", key0=4242, executor="async_ready"):
    c.L, c.B, c.split, c.key0 = L, B, split, key0
    c.set_executor(executor, phase="test")


def rollout(c, N, K, it, executor=None):
    pspec, fspec = make_specs(c, it, N, K)
    ex = executor or c.executor
    batch = c.loop.run_until_complete(ex.collect(pspec, fspec, c.engine.current, train_writer_sp(c.B),
                                                 reader_sp(c.cfg.sampling.reader_max_tokens), it=it))
    return batch, pspec, fspec


def hf_token_lps(trainer, pairs, chunk=8):
    """Per-token HF log-probs of completions (no grad)."""
    out = []
    with torch.no_grad():
        for s in range(0, len(pairs), chunk):
            sub = [Call(p, c, 0.0) for p, c in pairs[s:s + chunk]]
            _, tok_lp, _ = completion_logprob_sums(trainer.model, sub, trainer.pad_id, "cuda")
            tl = tok_lp.float().cpu().numpy()
            off = 0
            for cl in sub:
                out.append(tl[off:off + len(cl.completion_ids)])
                off += len(cl.completion_ids)
    return out


def lp_agreement(hf, vl):
    d = np.concatenate(hf) - np.concatenate(vl)
    return {"mean_abs": float(np.abs(d).mean()), "p99_abs": float(np.percentile(np.abs(d), 99)),
            "mean": float(d.mean()), "n_tokens": int(d.size)}


def meets_T2_3(a):
    return a["mean_abs"] <= 0.03 and a["p99_abs"] <= 0.30 and abs(a["mean"]) <= 0.02


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


# --------------------------------------------------------------------------- T2.1
def test_T2_1_engine_smoke(ctx, record):
    configure(ctx)
    pspec, fspec = make_specs(ctx, 0, 8, 1)

    async def run():
        from kmatters.executor import gather_all
        w = await gather_all([ctx.engine.generate(writer_ids(ctx.tok, EMPTY_MEMORY, p.chunks[0], ctx.B),
                                                  train_writer_sp(ctx.B), request_id("t2", "smoke", 0, "prefix", i, -1, 0),
                                                  ctx.engine.current) for i, p in enumerate(pspec)])
        r = await gather_all([ctx.engine.generate(reader_ids(ctx.tok, "(empty)", fspec[i][0].question),
                                                  reader_sp(12), request_id("t2", "smoke", 0, "reader", i, 0, 0), None)
                              for i in range(8)])
        return w, r

    w, r = ctx.loop.run_until_complete(run())
    for g in w + r:
        assert g.completion_ids and g.text.strip() and g.finish_reason in ("stop", "length")
    assert all(g.vllm_token_logprobs is not None and len(g.vllm_token_logprobs) == len(g.completion_ids) for g in w)
    assert ctx.engine.current is not None and ctx.engine.current.lora_name == "v0"
    record("T2.1", writer_finish=[g.finish_reason for g in w], reader_texts=[g.text for g in r])


# --------------------------------------------------------------------------- T2.2
def test_T2_2_token_id_round_trip(ctx, record):
    from vllm.inputs import TokensPrompt
    configure(ctx)
    pspec, _ = make_specs(ctx, 1, 4, 1)
    ids = writer_ids(ctx.tok, EMPTY_MEMORY, pspec[0].chunks[0], ctx.B)

    async def raw():
        final = None
        async for o in ctx.engine.llm.generate(TokensPrompt(prompt_token_ids=ids), train_writer_sp(ctx.B),
                                               request_id="t2_2_raw", lora_request=ctx.engine.current):
            final = o
        return final

    o = ctx.loop.run_until_complete(raw())
    assert list(o.prompt_token_ids) == ids
    batch, _, _ = rollout(ctx, 4, 2, it=2)
    n = 0
    for c in batch.calls():
        assert ctx.tok.decode(c.completion_ids, skip_special_tokens=True).strip() == c.text.strip()
        n += 1
    record("T2.2", n_calls_checked=n)


# --------------------------------------------------------------------------- T2.3
def test_T2_3_hf_vs_vllm_logprobs_v0(ctx, record):
    configure(ctx)
    batch, _, _ = rollout(ctx, 8, 2, it=3)                     # 32 prefix + 32 future writer calls
    writers = [c for c in batch.calls() if c.role != "reader"]
    assert len(writers) == 64
    hf = hf_token_lps(ctx.trainer, [(c.prompt_ids, c.completion_ids) for c in writers])
    a = lp_agreement(hf, [np.asarray(c.vllm_lp) for c in writers])
    record("T2.3", **a)
    assert a["mean_abs"] <= 0.03 and a["p99_abs"] <= 0.30 and abs(a["mean"]) <= 0.02, a


# --------------------------------------------------------------------------- T2.4 / T2.5
@pytest.fixture(scope="module")
def hot_swap(ctx):
    configure(ctx)
    batch, _, _ = rollout(ctx, 8, 2, it=4)
    writers = [c for c in batch.calls() if c.role != "reader"][:16]
    readers = batch.calls("reader")[:16]
    seqs = [(c.prompt_ids, c.completion_ids) for c in writers]
    rseqs = [(c.prompt_ids, c.completion_ids) for c in readers]
    assert len(seqs) == 16 and len(rseqs) == 16
    score = lambda pairs, lora: [np.array(ctx.loop.run_until_complete(ctx.engine.score_tokens(p, c, lora)))
                                 for p, c in pairs]
    v0 = score(seqs, ctx.engine.current)
    rd0 = score(rseqs, None)
    rng = np.random.default_rng(0)
    calls = [Call(p, c, float(rng.choice([-1.0, 1.0]))) for p, c in seqs]
    ctx.trainer.step(calls, ctx.cfg.estimator.T_norm, ctx.cfg.train.max_tokens_per_microbatch, lr=1e-3)
    v = ctx.version + 1
    ctx.trainer.save_adapter(ctx.adir(v))
    ctx.loop.run_until_complete(ctx.engine.set_writer_adapter(ctx.adir(v), v))
    ctx.version = v
    v1 = score(seqs, ctx.engine.current)
    rd1 = score(rseqs, None)
    hf1 = hf_token_lps(ctx.trainer, seqs)
    # fresh on-policy samples from v1 (see DEVIATIONS.md, T2.4)
    fresh_batch, _, _ = rollout(ctx, 8, 2, it=41)
    fresh = [c for c in fresh_batch.calls() if c.role != "reader"]
    hf_fresh = hf_token_lps(ctx.trainer, [(c.prompt_ids, c.completion_ids) for c in fresh])
    out = {"v0": v0, "v1": v1, "hf1": hf1, "rd0": rd0, "rd1": rd1, "version": v,
           "fresh_hf": hf_fresh, "fresh_vllm": [np.asarray(c.vllm_lp) for c in fresh]}
    # all v1 measurements are done: restore the v0 policy (weights and a fresh optimizer) now,
    # so the tests after T2.4/T2.5 do not inherit the far-moved v1 policy
    ctx.trainer.load_adapter_weights(ctx.adir(0))
    ctx.trainer.reset_optimizer()
    v_restore = ctx.version + 1
    ctx.loop.run_until_complete(ctx.engine.set_writer_adapter(ctx.adir(0), v_restore))
    ctx.version = v_restore
    return out


def test_T2_4_adapter_hot_swap(hot_swap, record):
    change = float(np.abs(np.concatenate(hot_swap["v1"]) - np.concatenate(hot_swap["v0"])).mean())
    a_old = lp_agreement(hot_swap["hf1"], hot_swap["v1"])
    a_fresh = lp_agreement(hot_swap["fresh_hf"], hot_swap["fresh_vllm"])
    record("T2.4", mean_abs_change_v1_vs_v0=change, hf_vs_vllm_v1_fresh_samples=a_fresh,
           hf_vs_vllm_v1_old_sequences_report_only=a_old, version=hot_swap["version"])
    assert change >= 0.02, change
    assert meets_T2_3(a_fresh), a_fresh


def test_T2_5_reader_frozen(hot_swap, record):
    d = float(np.abs(np.concatenate(hot_swap["rd1"]) - np.concatenate(hot_swap["rd0"])).max())
    record("T2.5", max_abs_delta=d)
    assert d <= 0.01, d


# --------------------------------------------------------------------------- T2.6
@pytest.mark.parametrize("executor", ["async_ready", "barrier"])
def test_T2_6_executor_completeness(ctx, executor, record):
    configure(ctx, executor=executor)
    N, K, L, U = 8, 4, 4, 2
    batch, pspec, fspec = rollout(ctx, N, K, it=6)
    assert len(batch.calls("prefix")) == N * L == 32
    assert len(batch.calls("future")) == N * K * U == 64
    assert len(batch.calls("reader")) == N * K == 32
    assert [p.i for p in batch.prefixes] == list(range(N))
    for p in batch.prefixes:
        assert len(p.calls) == L and len(p.futures) == K and p.key == tuple(pspec[p.i].key)
        for f in p.futures:
            assert f.i == p.i and f.key == tuple(fspec[p.i][f.k].key) and len(f.calls) == U
            assert f.reward in (0.0, 1.0)
            # the memory handed to future k is exactly the prefix's final memory
            assert f.calls[0].prompt_ids == writer_ids(ctx.tok, p.memory, fspec[p.i][f.k].chunks[0], ctx.B)
            assert f.reader.prompt_ids == reader_ids(ctx.tok, f.final_memory, f.question)
    r = batch.rewards()
    assert r.shape == (N, K) and set(np.unique(r)) <= {0.0, 1.0}
    if executor == "barrier":
        assert set(batch.phase_times) == {"t_phase_prefix", "t_phase_future", "t_phase_reader"}
    record(f"T2.6_{executor}", reward_mean=float(r.mean()), phase_times=batch.phase_times)


# --------------------------------------------------------------------------- T2.8
def test_T2_8_microbatch_invariance_real_model(ctx, record):
    configure(ctx)
    batch, _, _ = rollout(ctx, 8, 2, it=8)
    writers = [c for c in batch.calls() if c.role != "reader"][:48]
    rng = np.random.default_rng(1)
    calls = [Call(c.prompt_ids, c.completion_ids, float(rng.normal())) for c in writers]
    grads = {}
    for budget in (32768, 4096):
        ctx.trainer.opt.zero_grad(set_to_none=True)
        ctx.trainer._accumulate(calls, ctx.cfg.estimator.T_norm, budget)
        grads[budget] = flat_grad(ctx.trainer.model).clone()
    ctx.trainer.opt.zero_grad(set_to_none=True)
    g1, g2 = grads[32768], grads[4096]
    cos = cosine(g1, g2)
    rel = float((g1 - g2).norm() / g1.norm())
    # bf16 measurement of shape-dependent compute noise; the plan threshold is checked in fp32
    # (test_T2b_processes.py::test_T2_8_microbatch_invariance_fp32, see DEVIATIONS.md)
    record("T2.8_bf16", cosine=cos, rel_l2=rel, n_calls=len(calls), note="bf16 compute-noise report")
    assert cos >= 0.8, (cos, rel)


# --------------------------------------------------------------------------- T2.9
def test_T2_9_variance_tool_matches_training_gradient(ctx, record):
    from kmatters.variance import collect_pool, iter_Z
    # this runs after T2.4's large step; find a pool whose rewards are not all equal
    for vseed in range(10):
        configure(ctx, split="variance", key0=vseed)
        batch = collect_pool(ctx, ckpt_it=0, N0=4, K0=4)
        if np.ptp(batch.rewards()) > 0:
            break
    else:
        pytest.fail("no pool with non-constant rewards in 10 vseeds")
    T_norm, mt = ctx.cfg.estimator.T_norm, ctx.cfg.train.max_tokens_per_microbatch
    Zsum, n = None, 0
    for _, Z in iter_Z(ctx.trainer, batch, T_norm, mt):
        for z in Z:
            Zsum = z.clone() if Zsum is None else Zsum + z
            n += 1
    meanZ = Zsum / n
    r = batch.rewards()
    w_pre, w_fut = call_weights(r)
    calls, roles = [], []
    for p in batch.prefixes:
        for c in p.calls:
            calls.append(Call(c.prompt_ids, c.completion_ids, float(w_pre[p.i])))
            roles.append("prefix")
        for f in p.futures:
            for c in f.calls:
                calls.append(Call(c.prompt_ids, c.completion_ids, float(w_fut[f.i, f.k])))
                roles.append("future")
    sha0 = ctx.trainer.adapter_sha256()
    ctx.trainer.step(calls, T_norm, mt, lr=0.0, roles=roles)
    g = flat_grad(ctx.trainer.model).clone()
    ctx.trainer.opt.zero_grad(set_to_none=True)
    assert ctx.trainer.adapter_sha256() == sha0            # lr = 0 leaves the policy unchanged
    if float(g.norm()) == 0.0:
        pytest.fail("all advantages are zero in this pool; pick another vseed")
    cos = cosine(meanZ, -g)
    rel = float((meanZ + g).norm() / g.norm())
    # bf16 report; the plan threshold is checked in fp32 (test_T2_9_variance_matches_training_fp32)
    record("T2.9_bf16", cosine=cos, rel_l2=rel, reward_mean=float(r.mean()), note="bf16 compute-noise report")
    assert cos >= 0.8, (cos, rel)


# --------------------------------------------------------------------------- T2.10
@pytest.mark.slow
@pytest.mark.parametrize("N,K,L,B", [(128, 4, 16, 512), (32, 16, 16, 512), (32, 16, 16, 1024)])
def test_T2_10_memory_headroom(ctx, N, K, L, B, record):
    configure(ctx, L=L, B=B, key0=1010)
    m = run_iteration(ctx, 0, N, K, lr_override=0.0)
    record(f"T2.10_N{N}_K{K}_L{L}_B{B}", gpu_mem_peak_gib=m["gpu_mem_peak_gb"],
           torch_mem_peak_gib=m["torch_mem_peak_gb"], t_iter=m["t_iter"], t_rollout=m["t_rollout"],
           t_train=m["t_train"], trunc_rate=m["trunc_rate"])
    assert m["gpu_mem_peak_gb"] is not None and m["gpu_mem_peak_gb"] <= 78.0, m["gpu_mem_peak_gb"]


# --------------------------------------------------------------------------- T2.11
@pytest.mark.timeout(3 * 900)
def test_T2_11_loop_robustness(ctx, record):
    configure(ctx, key0=1111)
    t = []
    for cycle in range(3):
        # same data keys every cycle, so the timing comparison measures the loop, not the data
        m = run_iteration(ctx, 0, 16, 4)
        assert m["t_iter"] < 900
        t.append(m["t_iter"])
    rel = abs(t[2] - t[1]) / t[1]
    record("T2.11", t_iter=t, rel_diff_cycles_2_3=rel)
    assert rel <= 0.20, t


# --------------------------------------------------------------------------- T2.15
def test_T2_15_timing_accounting(ctx, record):
    configure(ctx, key0=1515)
    rows = [run_iteration(ctx, it, 16, 4, lr_override=0.0) for it in range(5)]
    other = [m["t_other"] / m["t_iter"] for m in rows]
    split_err = [abs(m["t_train"] - m["t_train_prefix"] - m["t_train_future"]) / m["t_train"] for m in rows]
    record("T2.15", t_other_frac=other, train_split_rel_err=split_err,
           t_iter=[m["t_iter"] for m in rows])
    assert statistics.median(other) <= 0.05, other
    assert max(split_err) <= 0.03, split_err
