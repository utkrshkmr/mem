"""T1.14: q1, q2 and q3 decision code on planted synthetic inputs."""
import numpy as np
import pytest

from kmatters.analysis import q1, q2, q3

KS = (1, 2, 4, 8, 16)
I_MAX = 100
ITS = list(range(0, I_MAX + 1, 10))


def _runs(gain_of_k, noise_sd=0.0, seed=0, seeds=(0, 1, 2), a0=0.30, tau=30.0):
    rng = np.random.default_rng(seed)
    runs = []
    for K in KS:
        for s in seeds:
            g = gain_of_k(K) + rng.normal(0, noise_sd)
            acc = [a0] + [a0 + g * (1 - np.exp(-it / tau)) for it in ITS[1:]]
            runs.append({"run_id": f"K{K}_s{s}", "K": K, "N": 64 // K, "seed": s, "it": ITS, "acc": acc})
    return runs


# ------------------------------------------------------------------- Q1a
def test_T1_14_q1a_planted_slope_yes(record):
    res = q1.q1a(_runs(lambda K: 0.05 + 0.03 * np.log2(K), noise_sd=0.005), I_MAX, n_perm=2000)
    assert res["decision"] == "YES", res
    assert res["D"] > 0.02 and res["D_ci95"][0] > 0 and res["p_perm"] < 0.01
    record("T1.14", q1a_planted_D=res["D"], q1a_planted_ci=res["D_ci95"])


def test_T1_14_q1a_flat_no():
    res = q1.q1a(_runs(lambda K: 0.10, noise_sd=0.002), I_MAX, n_perm=2000)
    assert res["decision"] == "NO", res
    exact = q1.q1a(_runs(lambda K: 0.10), I_MAX, n_perm=200)
    assert exact["decision"] == "NO" and abs(exact["D"]) < 1e-12


def test_T1_14_q1a_boundary_inconclusive():
    res = q1.q1a(_runs(lambda K: 0.10 + 0.007 * np.log2(K), noise_sd=0.03, seed=3), I_MAX, n_perm=2000)
    lo, hi = res["D_ci95"]
    assert lo < 0 < hi and hi > 0.02, res["D_ci95"]
    assert res["decision"] == "INCONCLUSIVE"
    assert q1.decide_q1a(0.015, 0.001, 0.029) == "INCONCLUSIVE"   # CI excludes 0 but |D| < 0.02
    assert q1.decide_q1a(0.030, 0.001, 0.059) == "YES"
    assert q1.decide_q1a(0.000, -0.019, 0.019) == "NO"


# ------------------------------------------------------------------- Q1b
def _cells(T_of_k, jitter=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return {(64 // K, K): list(T_of_k(K) * (1 + rng.uniform(-jitter, jitter, 3))) for K in KS}


def test_T1_14_q1b_cost_ratio_yes():
    runs = _runs(lambda K: 0.20)                     # identical learning per iteration
    T = {1: 180.0, 2: 150.0, 4: 120.0, 8: 100.0, 16: 72.0}
    res = q1.q1b(runs, I_MAX, _cells(T.get), _cells(T.get, seed=1), n_boot=500)
    assert res["decision"] == "YES", res
    assert res["fastest_arm"] == 16 and not res["interior"]
    assert res["R"] == pytest.approx(2.5, rel=0.03)
    assert res["target"]["ref_arm"] == 1


def test_T1_14_q1b_interior_optimum():
    runs = _runs(lambda K: 0.20)
    T = {1: 180.0, 2: 130.0, 4: 100.0, 8: 130.0, 16: 180.0}
    res = q1.q1b(runs, I_MAX, _cells(T.get), _cells(T.get, seed=1), n_boot=500)
    assert res["decision"] == "YES" and res["fastest_arm"] == 4 and res["interior"]


def test_T1_14_q1b_flat_no():
    runs = _runs(lambda K: 0.20)
    res = q1.q1b(runs, I_MAX, _cells(lambda K: 120.0), _cells(lambda K: 120.0, seed=1), n_boot=500)
    assert res["decision"] == "NO", res
    assert res["R"] == pytest.approx(1.0, abs=0.03)


def test_T1_14_q1b_censored_arm():
    # K=16 barely learns: never reaches the target -> its median TTT is +inf, R = +inf
    runs = _runs(lambda K: 0.20 if K < 16 else 0.02)
    res = q1.q1b(runs, I_MAX, _cells(lambda K: 120.0), _cells(lambda K: 120.0, seed=1), n_boot=300)
    assert np.isinf(res["arm_median_ttt_hours"][16]) and np.isinf(res["R"])
    assert res["decision"] == "YES"


def test_T1_14_q1b_priced_hours():
    G = q1.priced_hours([0, 1, 2, 10], 10, 100.0, 200.0)
    brute = [sum(100 + j / 10 * 100 for j in range(it)) / 3600 for it in (0, 1, 2, 10)]
    assert np.allclose(G, brute)


# ------------------------------------------------------------------- Q2
BENCH = {"prefill_rate": 50_000.0, "decode_rate_sat": 5_000.0, "n_sat": 64}
GRID = [(N, K) for N in (4, 8, 16, 32, 64, 128) for K in KS if N * K <= 512]


def _q2_cells(t_roll, frac=0.5):
    cells = {}
    for N, K in GRID:
        tok = {"prefix": {"prompt": 16 * 900.0 * N, "cached": 0.0, "completion": 16 * 300.0 * N},
               "future": {"prompt": 2 * 900.0 * N * K, "cached": 0.0, "completion": 2 * 300.0 * N * K},
               "reader": {"prompt": 400.0 * N * K, "cached": 0.0, "completion": 5.0 * N * K}}
        tr = t_roll(N, K)
        cells[(N, K)] = {"t_rollout": tr, "T_iter": tr / frac, "tokens": tok,
                         "occupancy": {"mean_inflight": 32.0, "idle_slot_frac": 0.5}}
    return cells


C_F_SAT = (2 * 900 + 400) / 50_000 + (2 * 300 + 5) / 5_000
C_P_SAT = 16 * 900 / 50_000 + 16 * 300 / 5_000


def test_T1_14_q2_free_futures_yes_material(record):
    res = q2.q2(_q2_cells(lambda N, K: 30 + 0.2 * N), BENCH)
    assert res["decision"] == "YES-material" and res["median_rho_f"] == pytest.approx(0.0)
    assert not res["capacity_specific_to_futures"]          # extra prefixes are cheap too
    record("T1.14", q2_free_decision=res["decision"])


def test_T1_14_q2_saturated_no():
    res = q2.q2(_q2_cells(lambda N, K: N * C_P_SAT + N * K * C_F_SAT), BENCH)
    assert res["decision"] == "NO" and res["median_rho_f"] == pytest.approx(1.0)
    assert res["median_rho_p"] == pytest.approx(1.0)


def test_T1_14_q2_immaterial_and_partial():
    res = q2.q2(_q2_cells(lambda N, K: 30 + 0.2 * N, frac=0.1), BENCH)
    assert res["decision"] == "YES-immaterial"
    res = q2.q2(_q2_cells(lambda N, K: N * C_P_SAT + 0.65 * N * K * C_F_SAT), BENCH)
    assert res["decision"] == "PARTIAL" and res["median_rho_f"] == pytest.approx(0.65)


def test_T1_14_q2_pairs():
    pairs = q2.future_pairs({c: None for c in GRID})
    assert all(a[0] * a[1] >= 32 and b[0] * b[1] <= 256 and b == (a[0], 2 * a[1]) for a, b in pairs)
    assert ((32, 1), (32, 2)) in pairs and ((16, 8), (16, 16)) in pairs and ((64, 4), (64, 8)) not in pairs


# ------------------------------------------------------------------- Q3
Q3_GRID = [(N, K) for N in (4, 8, 16, 32, 64, 128) for K in KS if N * K <= 512]


def _vdraws(Vp, Vf, n=400, rel=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([Vp * (1 + rng.normal(0, rel, n)), Vf * (1 + rng.normal(0, rel, n))])


def _brute_choice(T, Vp, Vf, budget):
    feas = [(Vp / N + Vf / (N * K), N, K) for (N, K), t in T.items() if t <= budget]
    return min(feas)[1:]


def test_T1_14_q3_additive_surface_no(record):
    Cp, Cf, Vp, Vf = 2.0, 0.25, 1.0, 6.0
    T = {(N, K): N * Cp + N * K * Cf for N, K in Q3_GRID}
    reps = {c: [t, t, t] for c, t in T.items()}
    barrier = [{"t_phase_prefix": 32 * Cp * 0.8, "t_train_prefix": 32 * Cp * 0.2,
                "t_phase_future": 128 * Cf * 0.6, "t_phase_reader": 128 * Cf * 0.1,
                "t_train_future": 128 * Cf * 0.3}] * 3
    res = q3.q3(_vdraws(Vp, Vf, rel=0.0), (Vp, Vf), reps, barrier)
    assert res["decision"] == "NO", res["per_beta"]
    assert res["model_a"]["Cp"] == pytest.approx(Cp) and res["model_a"]["Cf"] == pytest.approx(Cf)
    assert res["kstar_a_cont"] == pytest.approx(np.sqrt(Vf * Cp / (Vp * Cf)))
    for b in (1, 2, 4):
        N, K = _brute_choice(T, Vp, Vf, b * T[(32, 1)])
        pc = res["point_choices"][str(b)]
        assert (pc["N_e"], pc["K_e"]) == (N, K) and (pc["N_a"], pc["K_a"]) == (N, K)
    record("T1.14", q3_additive_choices={b: res["point_choices"][b] for b in res["point_choices"]})


def test_T1_14_q3_saturating_surface_yes():
    # futures nearly free until N*K = 256, then expensive; profiled at (32, 4) inside the free zone
    def T(N, K):
        return 10 + 2.0 * N + 0.01 * N * K + 1.0 * max(0, N * K - 256)
    reps = {(N, K): [T(N, K)] * 3 for N, K in Q3_GRID}
    t = T(32, 4)
    barrier = [{"t_phase_prefix": 5 + 64.0, "t_train_prefix": 0.0,
                "t_phase_future": 5 + 1.28, "t_phase_reader": 0.0, "t_train_future": 0.0}] * 3
    assert sum(barrier[0].values()) == pytest.approx(t)
    res = q3.q3(_vdraws(1.0, 20.0), (1.0, 20.0), reps, barrier)
    assert res["decision"] == "YES", res["per_beta"]
    b4 = res["per_beta"]["a"]["4"]
    assert b4["p_k_differs"] >= 0.8 and b4["median_regret"] >= 0.10
    pc = res["point_choices"]["4"]
    assert (pc["N_e"], pc["K_e"]) == _brute_choice({c: v[0] for c, v in reps.items()}, 1.0, 20.0, 4 * T(32, 1))
    assert pc["K_a"] != pc["K_e"]


def test_T1_14_q3_unresolved_vprefix_inconclusive():
    Cp, Cf = 2.0, 0.25
    T = {(N, K): N * Cp + N * K * Cf for N, K in Q3_GRID}
    barrier = [{"t_phase_prefix": 64.0, "t_train_prefix": 0.0, "t_phase_future": 32.0,
                "t_phase_reader": 0.0, "t_train_future": 0.0}] * 3
    rng = np.random.default_rng(5)                            # V_future stays positive (a variance)
    draws = np.column_stack([0.2 * (1 + rng.normal(0, 2.0, 400)), 6.0 * (1 + rng.normal(0, 0.05, 400))])
    assert (draws[:, 0] <= 0).mean() > 0.3
    res = q3.q3(draws, (0.2, 6.0), {c: [t] * 3 for c, t in T.items()}, barrier)
    assert res["frac_Vp_floored"] > 0.2 and not res["Vp_resolved"]
    assert res["decision"] == "INCONCLUSIVE"


def test_T1_14_q3_decide_rule():
    row = lambda p, r: {"p_k_differs": p, "median_regret": r}
    assert q3.decide_rule({"1": row(0.9, 0.2), "2": row(0.1, 0.0)}) == "YES"
    assert q3.decide_rule({"1": row(0.4, 0.5), "2": row(0.9, 0.01)}) == "NO"
    assert q3.decide_rule({"1": row(0.6, 0.05), "2": row(0.1, 0.0)}) == "INCONCLUSIVE"
