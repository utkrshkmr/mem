"""Q3: does marginal-cost pricing move K*? (PLAN.md §11.3)

Inputs:
  v_draws      [n, 2] bootstrap draws of (V_prefix, V_future) at the reference checkpoint
  v_point      (V_prefix, V_future) point estimate
  cells_reps   {(N, K): [T_iter of each rep]}  trained grid, condition A (27 cells)
  barrier_reps [ {t_phase_prefix, t_phase_future, t_phase_reader, t_train_prefix, t_train_future} ]
               reps of the barrier cell (32, 4) at the reference checkpoint (model a)
"""
from __future__ import annotations

import numpy as np

from .kstar import compare_pricing, fit_linear_costs, kstar_continuous

BETAS = (1, 2, 4)
BUDGET_CELL = (32, 1)
BARRIER_CELL = (32, 4)
P_YES, REGRET_YES = 0.8, 0.10
P_NO, REGRET_NO = 0.5, 0.03
FLOOR_REL = 1e-6
MAX_FLOORED = 0.20
_PHASES = ("t_phase_prefix", "t_phase_future", "t_phase_reader", "t_train_prefix", "t_train_future")


def model_a_costs(barrier: dict, N=BARRIER_CELL[0], K=BARRIER_CELL[1]) -> dict:
    """Profile-once additive model from the barrier cell; no intercept."""
    return {"Cp": (barrier["t_phase_prefix"] + barrier["t_train_prefix"]) / N,
            "Cf": (barrier["t_phase_future"] + barrier["t_phase_reader"] + barrier["t_train_future"]) / (N * K),
            "c0": 0.0}


def floor_vp(Vp, Vf):
    return Vp if Vp > 0 else FLOOR_REL * Vf


def decide_rule(per_beta: dict) -> str:
    rows = list(per_beta.values())
    if any(r["p_k_differs"] >= P_YES and r["median_regret"] >= REGRET_YES for r in rows):
        return "YES"
    if all(r["p_k_differs"] < P_NO or r["median_regret"] < REGRET_NO for r in rows):
        return "NO"
    return "INCONCLUSIVE"


def _cells(Tmed):
    return [{"N": N, "K": K, "T": T} for (N, K), T in sorted(Tmed.items())]


def _choice(res):
    """(K_model, K_emp, k_differs, regret). A model that picks nothing under a budget where a
    measured-feasible cell exists counts as infeasible: K differs and regret = +inf."""
    if res["emp"] is None:
        return None
    if res["lin"] is None:
        return None, res["emp"]["K"], True, float("inf")
    return res["lin"]["K"], res["emp"]["K"], bool(res["k_differs"]), float(res["regret"])


def q3(v_draws, v_point, cells_reps, barrier_reps, betas=BETAS, seed=0) -> dict:
    v_draws = np.asarray(v_draws, float)
    floored = v_draws[:, 0] <= 0
    frac_floored = float(floored.mean())
    Tmed = {c: float(np.median(v)) for c, v in cells_reps.items()}
    bar_med = {f: float(np.median([r[f] for r in barrier_reps])) for f in _PHASES}
    ma = model_a_costs(bar_med)
    fit_b = fit_linear_costs(_cells(Tmed))
    Vp0, Vf0 = floor_vp(*v_point), v_point[1]

    rng = np.random.default_rng(seed)
    acc = {m: {b: [] for b in betas} for m in ("a", "b")}
    for Vp, Vf in v_draws:
        Vp = floor_vp(Vp, Vf)
        Tb = {c: float(np.median(rng.choice(v, len(v)))) for c, v in cells_reps.items()}
        idx = rng.integers(0, len(barrier_reps), len(barrier_reps))
        ma_b = model_a_costs({f: float(np.median([barrier_reps[i][f] for i in idx])) for f in _PHASES})
        cells = _cells(Tb)
        for b in betas:
            budget = b * Tb[BUDGET_CELL]
            acc["a"][b].append(_choice(compare_pricing(cells, Vp, Vf, budget, model=ma_b)))
            acc["b"][b].append(_choice(compare_pricing(cells, Vp, Vf, budget)))

    def summarize(rows):
        rows = [r for r in rows if r is not None]
        Km = [r[0] for r in rows]
        Ke = [r[1] for r in rows]
        return {"p_k_differs": float(np.mean([r[2] for r in rows])),
                "median_regret": float(np.median([r[3] for r in rows])),
                "frac_regret_inf": float(np.mean([np.isinf(r[3]) for r in rows])),
                "frac_model_no_choice": float(np.mean([k is None for k in Km])),
                "K_model_mode": _mode([k for k in Km if k is not None]),
                "K_emp_mode": _mode(Ke), "n": len(rows)}

    per = {m: {str(b): summarize(acc[m][b]) for b in betas} for m in ("a", "b")}
    unresolved = frac_floored > MAX_FLOORED
    dec_a = "INCONCLUSIVE" if unresolved else decide_rule(per["a"])
    dec_b = "INCONCLUSIVE" if unresolved else decide_rule(per["b"])
    if dec_a == "YES" and dec_b == "NO":
        qual = ("marginal pricing moves K*, but a profiled linear model with an intercept is enough; "
                "no state-dependent scheduler is needed")
    elif dec_a == "YES" and dec_b == "YES":
        qual = "the cost surface is genuinely non-linear (motivates a systems contribution)"
    else:
        qual = None

    point = {}
    for b in betas:
        budget = b * Tmed[BUDGET_CELL]
        ra = compare_pricing(_cells(Tmed), Vp0, Vf0, budget, model=ma)
        rb = compare_pricing(_cells(Tmed), Vp0, Vf0, budget)
        point[str(b)] = {"budget": budget, "K_a": ra["lin"] and ra["lin"]["K"],
                         "N_a": ra["lin"] and ra["lin"]["N"], "K_b": rb["lin"] and rb["lin"]["K"],
                         "N_b": rb["lin"] and rb["lin"]["N"], "K_e": rb["emp"]["K"], "N_e": rb["emp"]["N"]}

    return {"model_a": ma, "model_b_fit": {k: (float(v) if v is not None else None) for k, v in fit_b.items()},
            "kstar_a_cont": kstar_continuous(Vp0, Vf0, ma["Cp"], ma["Cf"]),
            "kstar_b_cont": kstar_continuous(Vp0, Vf0, fit_b["Cp"], fit_b["Cf"]),
            "frac_Vp_floored": frac_floored, "Vp_resolved": not unresolved,
            "per_beta": per, "point_choices": point,
            "decision": dec_a, "qualifier_model_b": dec_b, "qualifier_text": qual,
            "rule": f"YES if some beta has P(K_a != K_e) >= {P_YES} and median regret_a >= {REGRET_YES}; "
                    f"NO if every beta has P < {P_NO} or median regret < {REGRET_NO}; else INCONCLUSIVE; "
                    f"INCONCLUSIVE if > {MAX_FLOORED:.0%} of V_prefix draws are floored"}


def empirical_choices(Tmed: dict, Vp, Vf, betas=BETAS) -> dict:
    """Secondary conditions: measured-surface K_e (and model b) at the point estimate, per budget."""
    Vp = floor_vp(Vp, Vf)
    out = {}
    for b in betas:
        res = compare_pricing(_cells(Tmed), Vp, Vf, b * Tmed[BUDGET_CELL])
        out[str(b)] = {"K_e": res["emp"]["K"], "N_e": res["emp"]["N"],
                       "K_b": res["lin"] and res["lin"]["K"], "kstar_b_cont": res["kstar_lin_cont"]}
    return out


def _mode(xs):
    if not xs:
        return None
    vals, counts = np.unique(xs, return_counts=True)
    return int(vals[np.argmax(counts)])
