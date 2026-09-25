"""Profiling helpers (PLAN.md §8.13): request occupancy and the NVML sampler.

The cost-grid cell runner is added in Phase 1; it calls loop.run_iteration with lr=0.
"""
from __future__ import annotations

import threading
import time

import numpy as np

OCC_DT = 0.010   # 10 ms occupancy grid


def inflight_series(t_submit, t_end, t0, t1, dt=OCC_DT) -> np.ndarray:
    """n(t) = #requests with t_submit <= t < t_end, on the grid t0, t0+dt, ... < t1."""
    grid = t0 + dt * np.arange(max(0, int(np.ceil((t1 - t0) / dt))))
    s = np.sort(np.asarray(t_submit, dtype=float))
    e = np.sort(np.asarray(t_end, dtype=float))
    return np.searchsorted(s, grid, side="right") - np.searchsorted(e, grid, side="right")


def occupancy(records, t0, t1, n_sat, dt=OCC_DT) -> dict:
    """records: iterable of objects/dicts with role, t_submit, t_end (CallRecord-like).
    Returns mean_inflight, idle_slot_frac and the same split by role."""
    def get(r, k):
        return r[k] if isinstance(r, dict) else getattr(r, k)

    recs = list(records)

    def summarize(sub):
        n = inflight_series([get(r, "t_submit") for r in sub], [get(r, "t_end") for r in sub],
                            t0, t1, dt)
        if n.size == 0:
            return {"mean_inflight": 0.0, "idle_slot_frac": 1.0}
        return {"mean_inflight": float(n.mean()),
                "idle_slot_frac": float(np.maximum(0, n_sat - n).mean() / n_sat)}

    out = summarize(recs)
    out["by_role"] = {role: summarize([r for r in recs if get(r, "role") == role])
                      for role in sorted({get(r, "role") for r in recs})}
    return out


class NvmlSampler:
    """Background thread sampling SM util, memory util, power (W) and memory used every 50 ms.

    NVML ignores CUDA_VISIBLE_DEVICES, so the device is mapped by UUID from torch."""

    def __init__(self, period=0.050, device_index=0):
        import pynvml
        import torch
        self._nvml = pynvml
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        if not uuid.startswith("GPU-"):
            uuid = "GPU-" + uuid
        self.uuid = uuid
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        self.period = period
        self.samples = []            # (t, sm, mem_util, power_w, mem_used_gb)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        nv = self._nvml
        while not self._stop.is_set():
            u = nv.nvmlDeviceGetUtilizationRates(self.handle)
            p = nv.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            m = nv.nvmlDeviceGetMemoryInfo(self.handle).used / 2**30
            self.samples.append((time.perf_counter(), u.gpu, u.memory, p, m))
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def summary(self, t0, t1) -> dict:
        """Means over [t0, t1) and the peak memory used in that window."""
        rows = [s for s in self.samples if t0 <= s[0] < t1]
        if not rows:
            return {"sm": None, "mem": None, "power": None, "mem_used_peak_gb": None, "n": 0}
        a = np.array(rows)
        return {"sm": float(a[:, 1].mean()), "mem": float(a[:, 2].mean()),
                "power": float(a[:, 3].mean()), "mem_used_peak_gb": float(a[:, 4].max()),
                "n": len(rows)}
