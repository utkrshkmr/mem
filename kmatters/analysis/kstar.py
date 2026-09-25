"""K* under an additive (average-cost) model vs the measured cost surface."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .variance_math import mse_model


def fit_linear_costs(cells):
    """cells: list of dicts with N, K, T (median seconds). Fits
    T ~= c0 + N*Cp + N*K*Cf with non-negative coefficients."""
    X = np.array([[1.0, c["N"], c["N"] * c["K"]] for c in cells])
    y = np.array([c["T"] for c in cells])
    coef, _ = nnls(X, y)
    pred = X @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"c0": coef[0], "Cp": coef[1], "Cf": coef[2],
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "max_rel_resid": float(np.max(np.abs(pred - y) / y))}


def kstar_continuous(Vp, Vf, Cp, Cf):
    if Vp <= 0 or Cf <= 0:
        return float("inf")
    return float(np.sqrt(Vf * Cp / (Vp * Cf)))


def choose_under_budget(cells, Vp, Vf, budget, cost_key="T"):
    """Among cells with cost <= budget, pick min model-MSE. Returns the cell or None."""
    feas = [c for c in cells if c[cost_key] <= budget]
    if not feas:
        return None
    return min(feas, key=lambda c: (mse_model(Vp, Vf, c["N"], c["K"]), c[cost_key]))


def compare_pricing(cells, Vp, Vf, budget, model=None):
    """Model-based choice vs measured-surface choice, both judged on measured T.

    cells: dicts with N, K, T (measured median seconds). Not mutated.
    model: None -> best linear fit with intercept (model b);
           dict {"Cp":..., "Cf":..., "c0": 0.0} -> profile-once additive model (model a).
    Returns the two choices, whether K differs, and regret =
    MSE(model choice) / MSE(measured choice) - 1. If the model's choice is
    infeasible under measured T, regret = +inf and lin_infeasible = True."""
    cells = [dict(c) for c in cells]
    lin = fit_linear_costs(cells) if model is None else {
        "c0": model.get("c0", 0.0), "Cp": model["Cp"], "Cf": model["Cf"],
        "r2": None, "max_rel_resid": None}
    for c in cells:
        c["T_lin"] = lin["c0"] + lin["Cp"] * c["N"] + lin["Cf"] * c["N"] * c["K"]
    pick_lin = choose_under_budget(cells, Vp, Vf, budget, "T_lin")
    pick_emp = choose_under_budget(cells, Vp, Vf, budget, "T")
    out = {"lin": pick_lin, "emp": pick_emp, "fit": lin,
           "kstar_lin_cont": kstar_continuous(Vp, Vf, lin["Cp"], lin["Cf"])}
    if pick_lin is None or pick_emp is None:
        out.update(k_differs=None, regret=None, lin_infeasible=None)
        return out
    infeasible = pick_lin["T"] > budget
    mse_l = mse_model(Vp, Vf, pick_lin["N"], pick_lin["K"])
    mse_e = mse_model(Vp, Vf, pick_emp["N"], pick_emp["K"])
    out.update(k_differs=pick_lin["K"] != pick_emp["K"],
               regret=float("inf") if infeasible else mse_l / mse_e - 1.0,
               lin_infeasible=infeasible)
    return out
