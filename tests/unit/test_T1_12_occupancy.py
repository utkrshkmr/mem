"""T1.12: occupancy statistics against an exact event-sweep integral."""
import numpy as np

from kmatters.profiling import occupancy


def _exact(intervals, t0, t1, n_sat):
    """Exact time-averages of n(t) and max(0, n_sat - n(t)) over [t0, t1)."""
    ev = sorted([(max(s, t0), +1) for s, e in intervals if e > t0 and s < t1] +
                [(min(e, t1), -1) for s, e in intervals if e > t0 and s < t1])
    t_prev, n, area, idle = t0, 0, 0.0, 0.0
    for t, d in ev:
        area += n * (t - t_prev)
        idle += max(0, n_sat - n) * (t - t_prev)
        t_prev, n = t, n + d
    area += n * (t1 - t_prev)
    idle += max(0, n_sat - n) * (t1 - t_prev)
    T = t1 - t0
    return area / T, idle / T / n_sat


def test_T1_12_occupancy_matches_exact(record):
    rng = np.random.default_rng(0)
    t0, t1, n_sat = 100.0, 220.0, 16
    recs = []
    for role, n, dur in (("prefix", 40, 8.0), ("future", 120, 3.0), ("reader", 60, 0.4)):
        s = rng.uniform(t0 - 2, t1 - 1, n)
        e = s + rng.exponential(dur, n) + 0.05
        recs += [{"role": role, "t_submit": a, "t_end": b} for a, b in zip(s, e)]
    occ = occupancy(recs, t0, t1, n_sat)
    ex_mean, ex_idle = _exact([(r["t_submit"], r["t_end"]) for r in recs], t0, t1, n_sat)
    assert abs(occ["mean_inflight"] - ex_mean) < 1e-3 * max(1.0, ex_mean) + 1e-3
    assert abs(occ["idle_slot_frac"] - ex_idle) < 1e-3
    for role in ("prefix", "future", "reader"):
        sub = [(r["t_submit"], r["t_end"]) for r in recs if r["role"] == role]
        m, i = _exact(sub, t0, t1, n_sat)
        assert abs(occ["by_role"][role]["mean_inflight"] - m) < 1e-3 * max(1.0, m) + 1e-3
        assert abs(occ["by_role"][role]["idle_slot_frac"] - i) < 1e-3
    record("T1.12", mean_inflight=occ["mean_inflight"], exact_mean=ex_mean,
           idle_slot_frac=occ["idle_slot_frac"], exact_idle=ex_idle)


def test_T1_12_simple_known_case():
    # two requests over a 10 s window: one covers [0, 5), one covers [2, 10) -> integral 13
    recs = [{"role": "a", "t_submit": 0.0, "t_end": 5.0}, {"role": "a", "t_submit": 2.0, "t_end": 10.0}]
    occ = occupancy(recs, 0.0, 10.0, n_sat=4)
    assert abs(occ["mean_inflight"] - 1.3) < 1e-3
    # idle = (4-1)*2 + (4-2)*3 + (4-1)*5 = 27 slot-seconds over 40
    assert abs(occ["idle_slot_frac"] - 27 / 40) < 1e-3
