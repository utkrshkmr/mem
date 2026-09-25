"""Q1: does K matter for learning? (PLAN.md §11.1)

Inputs are plain structures so the rule can be tested on synthetic data:
  run  = {"run_id", "K", "N", "seed", "it": [0, 10, ..., I_max], "acc": [...]}   (small-eval accuracy)
  cells_base / cells_trained = {(N, K): [T_iter of each measured rep, seconds]}   (condition A grids)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats

from .curves import normalized_auc, smooth, time_to_target

D_MARGIN = 0.02          # Q1a practical margin on D = 4*beta
R_YES = 1.25             # Q1b ratio threshold
TARGET_MIN_GAIN = 0.03   # below this K=1 gain, the target uses the best arm instead
INTERIOR_K = (2, 4, 8)
INTERIOR_FRAC = 0.8


def _q(x, p):
    """Quantile that never interpolates, so +inf draws stay +inf instead of NaN."""
    return float(np.quantile(np.asarray(x, float), p, method="inverted_cdf"))


def final_acc(run) -> float:
    return float(np.mean(run["acc"][-2:]))


def gain_auc(run, I_max) -> float:
    return normalized_auc(run["it"], run["acc"], I_max) - float(run["acc"][0])


def _by_arm(runs):
    d = defaultdict(list)
    for r in runs:
        d[int(r["K"])].append(r)
    return dict(sorted(d.items()))


def decide_q1a(D, lo, hi, margin=D_MARGIN) -> str:
    if (lo > 0 or hi < 0) and abs(D) >= margin:
        return "YES"
    if lo >= -margin and hi <= margin:
        return "NO"
    return "INCONCLUSIVE"


def q1a(runs, I_max, n_perm=10_000, seed=0) -> dict:
    """OLS of gain-AUC on log2 K across runs; D = 4*beta (K=16 vs K=1)."""
    K = np.array([int(r["K"]) for r in runs], float)
    g = np.array([gain_auc(r, I_max) for r in runs])
    x = np.log2(K)
    n = len(g)
    if n < 3 or np.ptp(x) == 0:
        raise ValueError("need >= 3 runs with at least two K values")
    xc = x - x.mean()
    Sxx = float(xc @ xc)
    beta = float(xc @ (g - g.mean()) / Sxx)
    alpha = float(g.mean() - beta * x.mean())
    resid = g - alpha - beta * x
    s2 = float(resid @ resid) / (n - 2)
    se_beta = float(np.sqrt(s2 / Sxx))
    t = float(stats.t.ppf(0.975, n - 2))
    D, se_D = 4 * beta, 4 * se_beta
    lo, hi = D - t * se_D, D + t * se_D

    rng = np.random.default_rng(seed)
    perm_x = np.stack([rng.permutation(xc) for _ in range(n_perm)])
    perm_beta = perm_x @ (g - g.mean()) / Sxx
    p_perm = float((1 + np.sum(np.abs(perm_beta) >= abs(beta) - 1e-12)) / (n_perm + 1))

    arms = {}
    groups = []
    for k, rs in _by_arm(runs).items():
        v = np.array([gain_auc(r, I_max) for r in rs])
        groups.append(v)
        half = (float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
                if len(v) > 1 else float("nan"))
        arms[k] = {"n": len(v), "mean": float(v.mean()), "ci95": [float(v.mean() - half), float(v.mean() + half)],
                   "values": v.tolist()}
    if len(groups) >= 2 and all(len(v) >= 2 for v in groups) and any(np.ptp(v) > 0 for v in groups):
        f, p_anova = stats.f_oneway(*groups)
        anova = {"F": float(f), "p": float(p_anova)}
    else:
        anova = {"F": None, "p": None}

    return {"alpha": alpha, "beta": beta, "D": D, "D_se": se_D, "D_ci95": [lo, hi], "df": n - 2,
            "p_perm": p_perm, "n_perm": n_perm, "anova": anova, "arms": arms,
            "decision": decide_q1a(D, lo, hi),
            "rule": f"YES if CI(D) excludes 0 and |D| >= {D_MARGIN}; NO if CI(D) inside "
                    f"[-{D_MARGIN}, {D_MARGIN}]; else INCONCLUSIVE"}


# ----------------------------------------------------------------------- Q1b
def priced_hours(its, I_max, T_base, T_trained) -> np.ndarray:
    """G(it) = sum_{j<it} T_price(j) / 3600 with T_price(j) = T_base + (j/I_max)(T_trained - T_base)."""
    its = np.asarray(its, float)
    return (its * T_base + (T_trained - T_base) / I_max * its * (its - 1) / 2) / 3600.0


def target_accuracy(runs) -> dict:
    a0 = float(np.mean([r["acc"][0] for r in runs]))
    arms = _by_arm(runs)
    ref = 1
    a_ref = float(np.mean([final_acc(r) for r in arms[1]])) if 1 in arms else float("nan")
    if not (a_ref - a0 >= TARGET_MIN_GAIN):
        gains = {k: float(np.mean([final_acc(r) for r in rs])) - a0 for k, rs in arms.items()}
        ref = max(gains, key=gains.get)
        a_ref = float(np.mean([final_acc(r) for r in arms[ref]]))
    return {"a_star": a0 + 0.5 * (a_ref - a0), "a0": a0, "a_ref": a_ref, "ref_arm": ref}


def _run_ttt(run, I_max, Tb, Tt, a_star) -> float:
    G = priced_hours(run["it"], I_max, Tb, Tt)
    v, censored = time_to_target(G, smooth(run["acc"], 3), a_star)
    return float("inf") if censored else v


def _arm_medians(runs, I_max, Tb, Tt, a_star) -> dict:
    out = {}
    for k, rs in _by_arm(runs).items():
        key = (int(rs[0]["N"]), k)
        out[k] = float(np.median([_run_ttt(r, I_max, Tb[key], Tt[key], a_star) for r in rs]))
    return out


def _ratio(med: dict):
    vals = np.array(list(med.values()))
    fast = min(med, key=med.get)
    if not np.isfinite(vals.min()):
        return float("nan"), fast
    return float(vals.max() / vals.min()), fast


def decide_q1b(R, lo, hi) -> str:
    # comparisons with NaN are False, and +inf compares as expected
    if R >= R_YES and lo > 1.0:
        return "YES"
    if hi < R_YES:
        return "NO"
    return "INCONCLUSIVE"


def q1b(runs, I_max, cells_base, cells_trained, n_boot=5000, seed=0) -> dict:
    Tb = {c: float(np.median(v)) for c, v in cells_base.items()}
    Tt = {c: float(np.median(v)) for c, v in cells_trained.items()}
    tgt = target_accuracy(runs)
    med = _arm_medians(runs, I_max, Tb, Tt, tgt["a_star"])
    R, fast = _ratio(med)

    arms = _by_arm(runs)
    rng = np.random.default_rng(seed)
    R_b, interior_hits = [], 0
    for _ in range(n_boot):
        Tb_b = {c: float(np.median(rng.choice(v, len(v)))) for c, v in cells_base.items()}
        Tt_b = {c: float(np.median(rng.choice(v, len(v)))) for c, v in cells_trained.items()}
        runs_b = [rs[i] for rs in arms.values() for i in rng.integers(0, len(rs), len(rs))]
        tgt_b = target_accuracy(runs_b)
        med_b = _arm_medians(runs_b, I_max, Tb_b, Tt_b, tgt_b["a_star"])
        R_b.append(_ratio(med_b)[0])
        if 1 in med_b and 16 in med_b and med_b[fast] < med_b[1] and med_b[fast] < med_b[16]:
            interior_hits += 1
    R_b = np.array(R_b)
    finite_or_inf = R_b[~np.isnan(R_b)]
    lo = _q(finite_or_inf, 0.025) if finite_or_inf.size else float("nan")
    hi = _q(finite_or_inf, 0.975) if finite_or_inf.size else float("nan")
    decision = decide_q1b(R, lo, hi)

    budget = med[fast]
    priced_acc = {}
    for k, rs in arms.items():
        key = (int(rs[0]["N"]), k)
        accs, beyond = [], False
        for r in rs:
            G = priced_hours(r["it"], I_max, Tb[key], Tt[key])
            beyond |= bool(budget > G[-1])
            accs.append(float(np.interp(budget, G, r["acc"])))
        priced_acc[k] = {"mean": float(np.mean(accs)), "beyond_horizon": beyond}

    interior_frac = interior_hits / n_boot
    return {"target": tgt, "arm_median_ttt_hours": med, "R": R, "R_ci95": [lo, hi],
            "frac_boot_nan": float(np.mean(np.isnan(R_b))), "n_boot": n_boot,
            "fastest_arm": fast,
            "interior": bool(fast in INTERIOR_K and interior_frac >= INTERIOR_FRAC),
            "interior_boot_frac": interior_frac,
            "priced_acc_at_budget": {"budget_hours": budget, "arms": priced_acc},
            "decision": decision,
            "rule": f"YES if R >= {R_YES} and CI lower > 1.0; NO if CI upper < {R_YES}; else INCONCLUSIVE"}


def mechanism(arm_gain_auc: dict, Vp: float, Vf: float, F: int = 64, ratios: dict | None = None) -> dict:
    """Report only: predicted per-iteration MSE_K = (K*Vp + Vf)/F vs arm-mean gain-AUC."""
    ks = sorted(arm_gain_auc)
    mse = {k: (k * Vp + Vf) / F for k in ks}
    rho = stats.spearmanr([-mse[k] for k in ks], [arm_gain_auc[k] for k in ks]).statistic
    return {"mse_pred": mse, "spearman_negmse_vs_gain_auc": float(rho),
            "Vf_over_Vp": ratios or {}}
