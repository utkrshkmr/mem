from collections import Counter

import numpy as np
import pytest

from kmatters.env.ledger import LedgerState, brute_force_answer, make_future, make_prefix
from kmatters.rl.estimator import advantages, call_weights
from kmatters.analysis.variance_math import VarianceAccumulator, mse_model
from kmatters.analysis.kstar import compare_pricing, fit_linear_costs
from kmatters.analysis.curves import normalized_auc, smooth, time_to_target


# ---------------------------------------------------------------- ledger ----
@pytest.mark.parametrize("level", [1, 3, 5])
def test_answers_match_text_reparse(level):
    for j in range(300):
        p = make_prefix("train", (7, 0, j), L=8, level=level)
        for k in range(3):
            f = make_future("train", p, k, U=2, level=level)
            assert brute_force_answer(p, f, level) == f.gold


def test_determinism_and_independence():
    p1 = make_prefix("train", (1, 2, 3), L=6, level=3)
    p2 = make_prefix("train", (1, 2, 3), L=6, level=3)
    assert p1.chunks == p2.chunks and p1.state_json == p2.state_json
    fa = make_future("train", p1, 0, U=2, level=3)
    fb = make_future("train", p1, 0, U=2, level=3)
    fc = make_future("train", p1, 1, U=2, level=3)
    assert fa.chunks == fb.chunks and fa.q == fb.q
    assert (fa.chunks, fa.q) != (fc.chunks, fc.q)
    # different split -> different stream
    pe = make_prefix("eval", (1, 2, 3), L=6, level=3)
    assert pe.chunks != p1.chunks


def test_prefix_does_not_depend_on_future_generation():
    a = make_prefix("train", (9, 9, 9), L=5, level=3)
    for k in range(5):
        make_future("train", a, k, U=2, level=3)
    b = make_prefix("train", (9, 9, 9), L=5, level=3)
    assert a.chunks == b.chunks and a.state_json == b.state_json


def test_state_roundtrip():
    p = make_prefix("train", (3, 1, 4), L=10, level=4)
    st = LedgerState.from_json(p.state_json)
    assert st.to_json() == p.state_json


def test_answers_not_guessable():
    golds = []
    for j in range(2000):
        p = make_prefix("calib", (0, j), L=16, level=3)
        golds.append(make_future("calib", p, 0, U=2, level=3).gold)
    top = Counter(golds).most_common(1)[0][1] / len(golds)
    assert top < 0.10, top


def test_futures_touch_prefix_transactions():
    hits = 0
    for j in range(500):
        p = make_prefix("calib", (1, j), L=16, level=3)
        f = make_future("calib", p, 0, U=2, level=3)
        pre = set(LedgerState.from_json(p.state_json).prefix_ids)
        text = "\n".join(f.chunks)
        hits += any(f"T{t:02d} was cancelled" in text or f"amount of T{t:02d}" in text
                    for t in pre)
    assert hits / 500 > 0.8


# ------------------------------------------------------------- estimator ----
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _toy_truth(th, ph, P):
    pa, pb = _softmax(th), _softmax(ph)
    J = lambda pa, pb: sum(pa[a] * 0.5 * sum(pb[b] * P[a, b, y] for b in range(2))
                           for a in range(2) for y in range(2))
    eps = 1e-6
    gth = np.array([(J(_softmax(th + eps * np.eye(2)[i]), pb) -
                     J(_softmax(th - eps * np.eye(2)[i]), pb)) / (2 * eps) for i in range(2)])
    gph = np.array([(J(pa, _softmax(ph + eps * np.eye(2)[i])) -
                     J(pa, _softmax(ph - eps * np.eye(2)[i]))) / (2 * eps) for i in range(2)])
    return np.concatenate([gth, gph])


def _toy_estimate(rng, th, ph, P, N, K, baseline, reps):
    pa, pb = _softmax(th), _softmax(ph)
    a = rng.choice(2, size=(reps, N), p=pa)
    y = rng.integers(0, 2, size=(reps, N, K))
    b = rng.choice(2, size=(reps, N, K), p=pb)
    r = (rng.random((reps, N, K)) < P[a[..., None], b, y]).astype(float)
    out = np.zeros((reps, 4))
    for t in range(reps):
        wp, wf = call_weights(r[t], baseline)
        sa = np.eye(2)[a[t]] - pa            # grad log softmax
        sb = np.eye(2)[b[t]] - pb
        out[t, :2] = (wp[:, None] * sa).sum(0)
        out[t, 2:] = (wf[..., None] * sb).sum((0, 1))
    return out


@pytest.mark.parametrize("N,K", [(2, 1), (4, 2), (3, 5), (8, 8)])
@pytest.mark.parametrize("baseline", ["loo_prefix", "tree"])
def test_estimator_unbiased(N, K, baseline):
    rng = np.random.default_rng(0)
    th, ph = np.array([0.3, -0.2]), np.array([-0.4, 0.5])
    P = rng.uniform(0.05, 0.95, size=(2, 2, 2))
    truth = _toy_truth(th, ph, P)
    est = _toy_estimate(rng, th, ph, P, N, K, baseline, reps=40000)
    se = est.std(0) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(0) - truth) < 4 * se + 1e-4), (est.mean(0), truth, se)


def test_weights_sum_rules():
    rng = np.random.default_rng(1)
    for N, K in [(4, 16), (64, 1), (16, 4)]:
        r = rng.integers(0, 2, size=(N, K)).astype(float)
        wp, wf = call_weights(r)
        A_pre, A_fut = advantages(r)
        # futures of a prefix carry total weight A-average / N, like one prefix
        assert np.allclose(wf.sum(1), A_fut.mean(1) / N)
        assert np.allclose(wp, A_pre / N)


def test_fixed_baseline_variance_formula():
    """With a constant baseline, tr Cov(g_hat) == Vp/N + Vf/(NK) exactly."""
    rng = np.random.default_rng(2)
    th, ph = np.array([0.1, 0.0]), np.array([0.0, 0.2])
    pa, pb = _softmax(th), _softmax(ph)
    P = rng.uniform(0.05, 0.95, size=(2, 2, 2))
    bconst = 0.4
    # population Z = (r - b) (s_a, s_b); compute Vp, Vf by enumeration
    Zs, probs, groups = [], [], []
    for a in range(2):
        for y in range(2):
            for b in range(2):
                for rr in (0, 1):
                    pr = pa[a] * 0.5 * pb[b] * (P[a, b, y] if rr else 1 - P[a, b, y])
                    Zs.append((rr - bconst) * np.concatenate([np.eye(2)[a] - pa, np.eye(2)[b] - pb]))
                    probs.append(pr)
                    groups.append(a)
    Zs, probs, groups = np.array(Zs), np.array(probs), np.array(groups)
    mu = probs @ Zs
    cond_means = {a: (probs[groups == a] @ Zs[groups == a]) / probs[groups == a].sum() for a in range(2)}
    Vp = sum(pa[a] * np.sum((cond_means[a] - mu) ** 2) for a in range(2))
    Vf = sum(probs[i] * np.sum((Zs[i] - cond_means[groups[i]]) ** 2) for i in range(len(Zs)))
    for N, K in [(2, 1), (4, 4), (8, 2)]:
        reps = 60000
        a = rng.choice(2, size=(reps, N), p=pa)
        y = rng.integers(0, 2, size=(reps, N, K))
        b = rng.choice(2, size=(reps, N, K), p=pb)
        r = (rng.random((reps, N, K)) < P[a[..., None], b, y]).astype(float)
        sa = np.eye(2)[a] - pa
        sb = np.eye(2)[b] - pb
        g = np.concatenate([((r.mean(2) - bconst)[..., None] * sa).mean(1),
                            (((r - bconst)[..., None] * sb).mean(2)).mean(1)], axis=1)
        emp = np.sum(g.var(0))
        pred = mse_model(Vp, Vf, N, K)
        assert abs(emp / pred - 1) < 0.03, (N, K, emp, pred)


# -------------------------------------------------------------- variance ----
def test_variance_accumulator_recovers_components():
    rng = np.random.default_rng(3)
    d, N0, K0 = 50, 64, 8
    Au = rng.normal(size=(d, d)) * 0.1
    Ae = rng.normal(size=(d, d)) * 0.2
    Vp_true = np.trace(Au @ Au.T)
    Vf_true = np.trace(Ae @ Ae.T)
    est = []
    for trial in range(200):
        acc = VarianceAccumulator()
        mu = np.ones(d)
        for i in range(N0):
            u = Au @ rng.normal(size=d)
            acc.add_prefix([mu + u + Ae @ rng.normal(size=d) for _ in range(K0)])
        est.append(acc.estimates())
    est = np.array(est)
    assert abs(est[:, 0].mean() / Vp_true - 1) < 0.05
    assert abs(est[:, 1].mean() / Vf_true - 1) < 0.02
    bs = acc.bootstrap(B=500)
    assert bs.shape == (500, 2) and np.all(np.isfinite(bs))


# ----------------------------------------------------------------- kstar ----
def _grid():
    return [(N, K) for N in (4, 8, 16, 32, 64, 128) for K in (1, 2, 4, 8, 16) if N * K <= 512]


def test_linear_surface_choices_agree():
    cells = [dict(N=N, K=K, T=5 + 2.0 * N + 0.25 * N * K) for N, K in _grid()]
    fit = fit_linear_costs(cells)
    assert fit["r2"] > 0.999
    res = compare_pricing(cells, Vp=1.0, Vf=4.0, budget=100)
    assert res["k_differs"] is False and abs(res["regret"]) < 1e-9


def test_saturating_surface_moves_choice():
    # rollout futures nearly free until N*K reaches 256 (batch saturation)
    def T(N, K):
        return 5 + 2.0 * N + 0.02 * N * K + 0.5 * max(0, N * K - 256)
    cells = [dict(N=N, K=K, T=T(N, K)) for N, K in _grid()]
    res = compare_pricing(cells, Vp=1.0, Vf=8.0, budget=70)
    assert res["emp"] is not None and res["lin"] is not None
    assert res["regret"] >= 0
    assert res["k_differs"] is True        # (16, 8) vs (16, 16) for this surface
    assert "T_lin" not in cells[0]          # inputs not mutated


def test_profile_once_model_a():
    # a profile-once additive model with no intercept misprices the fixed 30 s overhead
    def T(N, K):
        return 30 + 1.0 * N + 0.25 * N * K
    cells = [dict(N=N, K=K, T=T(N, K)) for N, K in _grid()]
    ref = dict(N=32, K=4)
    Cp = (1.0 * ref["N"] + 30 * 0.5) / ref["N"]          # half of the overhead lands on prefixes
    Cf = (0.25 * ref["N"] * ref["K"] + 30 * 0.5) / (ref["N"] * ref["K"])
    res = compare_pricing(cells, Vp=1.0, Vf=6.0, budget=T(32, 1), model={"Cp": Cp, "Cf": Cf})
    assert res["regret"] >= 0
    res_b = compare_pricing(cells, Vp=1.0, Vf=6.0, budget=T(32, 1))
    assert res_b["fit"]["r2"] > 0.999 and res_b["regret"] == 0


# ---------------------------------------------------------------- curves ----
from kmatters.analysis.curves import normalized_auc, smooth, time_to_target


def test_curves():
    x = [0, 10, 20, 30]
    y = [0.2, 0.3, 0.5, 0.5]
    assert abs(normalized_auc(x, y, 30) - (0.25 * 10 + 0.4 * 10 + 0.5 * 10) / 30) < 1e-12
    assert abs(normalized_auc(x, y, 25) - (0.25 * 10 + 0.4 * 10 + 0.5 * 5) / 25) < 1e-12
    assert time_to_target(x, y, 0.4) == (15.0, False)
    assert time_to_target(x, y, 0.6) == (30.0, True)
    assert np.allclose(smooth([1, 2, 3], 3), [4 / 3, 2, 8 / 3])


def test_torch_accumulator_matches_numpy():
    torch = pytest.importorskip("torch")
    from kmatters.analysis.variance_math import TorchVarianceAccumulator
    rng = np.random.default_rng(4)
    a_np, a_t = VarianceAccumulator(), TorchVarianceAccumulator(chunk=37)
    for i in range(12):
        Z = [rng.normal(size=301) + (i % 3) for _ in range(5)]
        a_np.add_prefix(Z)
        a_t.add_prefix([torch.tensor(z, dtype=torch.float32) for z in Z])
    e_np, e_t = np.array(a_np.estimates()), np.array(a_t.estimates())
    assert np.allclose(e_np, e_t, rtol=1e-6, atol=1e-9), (e_np, e_t)
    assert np.allclose(a_np.bootstrap(B=50, seed=1), a_t.bootstrap(B=50, seed=1), rtol=1e-6, atol=1e-9)
