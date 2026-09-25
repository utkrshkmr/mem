"""Learning-curve summaries: normalized AUC and time-to-target with censoring."""
from __future__ import annotations

import numpy as np


def normalized_auc(x, y, x_max):
    """Trapezoid AUC of y over [0, x_max] divided by x_max. Requires x[0] == 0
    and x[-1] >= x_max (curve truncated/interpolated at x_max)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x[0] != 0 or x[-1] < x_max:
        raise ValueError("curve must span [0, x_max]")
    yi = np.interp(x_max, x, y)
    keep = x < x_max
    xs = np.append(x[keep], x_max)
    ys = np.append(y[keep], yi)
    return float(np.trapezoid(ys, xs) / x_max)


def time_to_target(x, y, target):
    """First x where the linearly interpolated curve reaches target.
    Returns (value, censored). Censored runs return (x[-1], True)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if y[0] >= target:
        return float(x[0]), False
    for i in range(1, len(x)):
        if y[i] >= target:
            f = (target - y[i - 1]) / (y[i] - y[i - 1])
            return float(x[i - 1] + f * (x[i] - x[i - 1])), False
    return float(x[-1]), True


def smooth(y, window=3):
    """Centered moving average used only for time-to-target (eval noise)."""
    y = np.asarray(y, float)
    if window <= 1:
        return y
    pad = window // 2
    yp = np.pad(y, pad, mode="edge")
    return np.convolve(yp, np.ones(window) / window, mode="valid")
