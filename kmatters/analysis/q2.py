"""Q2: does exploitable capacity survive competent scheduling? (PLAN.md §11.2)

cells = {(N, K): cell} with per-cell medians over reps:
  cell = {"t_rollout", "T_iter",
          "tokens": {role: {"prompt", "cached", "completion"}} for role in prefix/future/reader,
          "occupancy": {"mean_inflight", "idle_slot_frac"}}          (occupancy optional)
bench = {"prefill_rate", "decode_rate_sat", "n_sat"}                  (from bench.py)
"""
from __future__ import annotations

import numpy as np

RHO_YES = 0.5
RHO_NO = 0.8
ROLLOUT_FRAC_MATERIAL = 0.25
PAIR_MIN_F, PAIR_MAX_F = 32, 256     # practical pairs: N*K >= 32 and the larger cell's N*K <= 256


def cell_medians(rows) -> dict:
    """Collapse cells.jsonl rows (warmup excluded) into per-(N, K) medians."""
    by = {}
    for r in rows:
        if r.get("warmup"):
            continue
        by.setdefault((int(r["N"]), int(r["K"])), []).append(r)
    out = {}
    for key, rs in by.items():
        med = lambda f: float(np.median([f(r) for r in rs]))
        roles = rs[0]["tokens"].keys()
        out[key] = {
            "t_rollout": med(lambda r: r["t_rollout"]), "T_iter": med(lambda r: r["T_iter"]),
            "tokens": {role: {f: med(lambda r, role=role, f=f: r["tokens"][role][f])
                              for f in ("prompt", "cached", "completion")} for role in roles},
            "n_reps": len(rs),
        }
        if "occupancy" in rs[0]:
            out[key]["occupancy"] = {f: med(lambda r, f=f: r["occupancy"][f])
                                     for f in ("mean_inflight", "idle_slot_frac")}
    return out


def _sat_seconds(tok: dict, bench: dict) -> float:
    return (tok["prompt"] - tok["cached"]) / bench["prefill_rate"] + tok["completion"] / bench["decode_rate_sat"]


def sat_cost_per_future(cell, N, K, bench) -> float:
    """Saturated rollout seconds of one future: its U writer calls plus the reader call."""
    t = cell["tokens"]
    return (_sat_seconds(t["future"], bench) + _sat_seconds(t["reader"], bench)) / (N * K)


def sat_cost_per_prefix(cell, N, bench) -> float:
    """Saturated rollout seconds of one prefix together with its K futures."""
    return sum(_sat_seconds(t, bench) for t in cell["tokens"].values()) / N


def future_pairs(cells):
    return [((N, K), (N, 2 * K)) for (N, K) in sorted(cells)
            if (N, 2 * K) in cells and N * K >= PAIR_MIN_F and N * 2 * K <= PAIR_MAX_F]


def prefix_pairs(cells):
    return [((N, K), (2 * N, K)) for (N, K) in sorted(cells)
            if (2 * N, K) in cells and N * K >= PAIR_MIN_F and 2 * N * K <= PAIR_MAX_F]


def decide_q2(med_rho_f, med_rollout_frac) -> str:
    if med_rho_f <= RHO_YES:
        return "YES-material" if med_rollout_frac >= ROLLOUT_FRAC_MATERIAL else "YES-immaterial"
    if med_rho_f >= RHO_NO:
        return "NO"
    return "PARTIAL"


def q2(cells, bench, barrier_cells=None) -> dict:
    fp = future_pairs(cells)
    if not fp:
        raise ValueError("no practical (N,K)->(N,2K) pairs in the grid")
    pairs = []
    for (a, b) in fp:
        N, K = a
        dT = (cells[b]["t_rollout"] - cells[a]["t_rollout"]) / (N * K)
        c_sat = sat_cost_per_future(cells[b], *b, bench)
        dT_total = (cells[b]["T_iter"] - cells[a]["T_iter"]) / (N * K)
        avg_total = cells[b]["T_iter"] / (b[0] * b[1])
        pairs.append({"from": list(a), "to": list(b), "dT_roll_per_future": dT, "c_f_sat": c_sat,
                      "rho_f": dT / c_sat, "total_marginal_over_avg": dT_total / avg_total})

    ppairs = []
    for (a, b) in prefix_pairs(cells):
        N, K = a
        dT = (cells[b]["t_rollout"] - cells[a]["t_rollout"]) / N
        c_sat = sat_cost_per_prefix(cells[b], b[0], bench)
        ppairs.append({"from": list(a), "to": list(b), "dT_roll_per_prefix": dT, "c_p_sat": c_sat,
                       "rho_p": dT / c_sat})

    pair_cells = sorted({c for pr in fp for c in pr})
    rollout_frac = {c: cells[c]["t_rollout"] / cells[c]["T_iter"] for c in cells}
    med_rho_f = float(np.median([p["rho_f"] for p in pairs]))
    med_rho_p = float(np.median([p["rho_p"] for p in ppairs])) if ppairs else float("nan")
    med_frac = float(np.median([rollout_frac[c] for c in pair_cells]))

    per_cell = {}
    for c, cell in sorted(cells.items()):
        occ = cell.get("occupancy", {})
        per_cell[f"{c[0]},{c[1]}"] = {
            "rollout_frac": rollout_frac[c],
            "idle_slot_frac": occ.get("idle_slot_frac"),
            "inflight_over_nsat": (occ["mean_inflight"] / bench["n_sat"]) if "mean_inflight" in occ else None,
            "barrier_over_async": (barrier_cells[c]["t_rollout"] / cell["t_rollout"])
            if barrier_cells and c in barrier_cells else None,
        }

    return {"future_pairs": pairs, "prefix_pairs": ppairs, "per_cell": per_cell,
            "median_rho_f": med_rho_f, "median_rho_p": med_rho_p,
            "median_rollout_frac_pair_cells": med_frac,
            "capacity_specific_to_futures": not (med_rho_p <= RHO_YES),
            "decision": decide_q2(med_rho_f, med_frac),
            "rule": f"YES-material if median rho_f <= {RHO_YES} and median rollout_frac >= "
                    f"{ROLLOUT_FRAC_MATERIAL}; YES-immaterial if rho_f <= {RHO_YES} and frac < "
                    f"{ROLLOUT_FRAC_MATERIAL}; NO if rho_f >= {RHO_NO}; else PARTIAL"}
