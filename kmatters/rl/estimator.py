"""Advantages and per-call loss weights for the nested (N prefixes x K futures) estimator.

g_hat = (1/N) sum_i [ A_pre_i * s_pre_i + (1/K) sum_k A_fut_ik * s_fut_ik ]
where s_* are score vectors (sum of grad log-probs of the writer tokens in those calls).

Baselines:
  "loo_prefix" (E1, primary): b_i = mean_{j != i} rbar_j, used for both prefix and future terms.
  "tree"       (E2, ablation): prefix term as E1; future term uses b_ik = mean_{l != k} r_il
                               (falls back to E1 when K == 1).
Both are unbiased: each baseline is independent of the actions it multiplies.
"""
from __future__ import annotations

import numpy as np


def advantages(r: np.ndarray, baseline: str = "loo_prefix"):
    """r: array [N, K] of rewards. Returns (A_pre [N], A_fut [N, K])."""
    r = np.asarray(r, dtype=np.float64)
    N, K = r.shape
    if N < 2:
        raise ValueError("need N >= 2 for a leave-one-out baseline")
    rbar = r.mean(axis=1)
    b = (rbar.sum() - rbar) / (N - 1)
    A_pre = rbar - b
    if baseline == "loo_prefix" or K == 1:
        A_fut = r - b[:, None]
    elif baseline == "tree":
        b_in = (r.sum(axis=1, keepdims=True) - r) / (K - 1)
        A_fut = r - b_in
    else:
        raise ValueError(baseline)
    return A_pre, A_fut


def call_weights(r: np.ndarray, baseline: str = "loo_prefix"):
    """Coefficient multiplying each call's score vector in g_hat.

    Returns (w_pre [N], w_fut [N, K]). Every writer call inside prefix i gets
    w_pre[i]; every writer call inside future (i, k) gets w_fut[i, k].
    Training loss = -(1 / T_norm) * sum_calls w_call * sum_tokens log pi.
    """
    N, K = np.asarray(r).shape
    A_pre, A_fut = advantages(r, baseline)
    return A_pre / N, A_fut / (N * K)
