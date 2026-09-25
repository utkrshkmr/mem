"""Profiling (PLAN.md §8.13): request occupancy, the NVML sampler, and the cost-grid cell
runner, which calls the training code path loop.run_iteration with lr=0.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .seeding import seed_from

OCC_DT = 0.010   # 10 ms occupancy grid
ROLE_CODE = {"prefix": 0, "future": 1, "reader": 2}


def cell_id(N: int, K: int, L: int, B: int) -> int:
    """Derived from (N, K, L, B) only, not the executor, so executors see identical data."""
    return seed_from(N, K, L, B) % 2**31


def engine_label(cfg) -> str:
    return f"s{cfg.vllm.max_num_seqs}_b{cfg.vllm.max_num_batched_tokens}"


def run_cell(ctx, cell: dict, reps: int, warmup: int, ckpt_label: str, n_sat: int | None = None,
             intervals_dir: Path | None = None) -> list[dict]:
    """warmup + reps iterations of run_iteration(lr=0) for one (N, K, L, B, executor) cell.
    Keys (cell_id, rep, i); rep < warmup is the discarded warmup. Raises if the policy moved."""
    from .loop import run_iteration, token_counts
    N, K, L, B, ex = cell["N"], cell["K"], cell["L"], cell["B"], cell["executor"]
    cid = cell_id(N, K, L, B)
    ctx.L, ctx.B, ctx.split, ctx.key0 = L, B, "profile", cid
    ctx.set_executor(ex, phase="profile")
    sha0 = ctx.trainer.adapter_sha256()
    rows = []
    for rep in range(warmup + reps):
        m = run_iteration(ctx, rep, N, K, lr_override=0.0)
        batch, w = ctx.last_batch, ctx.last_windows
        calls = batch.calls()
        if intervals_dir is not None:
            intervals_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(intervals_dir / f"{cid}_{ex}_rep{rep}.npz",
                                t_submit=np.array([c.t_submit for c in calls]) - w["rollout"][0],
                                t_end=np.array([c.t_end for c in calls]) - w["rollout"][0],
                                role=np.array([ROLE_CODE[c.role] for c in calls], np.int8),
                                window=np.array([0.0, w["rollout"][1] - w["rollout"][0]]))
        row = {"cell_id": cid, "executor": ex, "N": N, "K": K, "L": L, "B": B, "ckpt": ckpt_label,
               "rep": rep, "warmup": rep < warmup,
               "T_iter": m["t_iter"], "t_rollout": m["t_rollout"], "t_train": m["t_train"],
               "t_train_prefix": m["t_train_prefix"], "t_train_future": m["t_train_future"], "t_opt": m["t_opt"],
               "t_sync": m["t_sync"], "t_other": m["t_other"], "tokens": token_counts(batch),
               "occupancy": occupancy(calls, *w["rollout"], n_sat) if n_sat else None,
               "nvml": m.get("nvml"), "gpu_mem_peak_gb": m["gpu_mem_peak_gb"],
               "reward_mean": m["reward_mean"], "mem_len_mean": m["mem_len_mean"], "trunc_rate": m["trunc_rate"],
               "adapter_sha256": ctx.trainer.adapter_sha256(), "engine_cfg": engine_label(ctx.cfg)}
        for k in ("t_phase_prefix", "t_phase_future", "t_phase_reader"):
            if k in m:
                row[k] = m[k]
        rows.append(row)
        # scratch adapters: keep only the loaded one and its predecessor
        for p in ctx.adapter_root.glob("v*"):
            if int(p.name[1:]) < ctx.version - 1:
                shutil.rmtree(p, ignore_errors=True)
    if ctx.trainer.adapter_sha256() != sha0:
        raise RuntimeError(f"policy moved during profiling of cell {cell}")
    return rows


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
        self.samples = []            # (t, sm, mem_util, power_w, mem_used_gib)
        self._lock = threading.Lock()
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
            with self._lock:
                self.samples.append((time.perf_counter(), u.gpu, u.memory, p, m))
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def prune(self, before: float):
        """Drop samples older than `before` so long runs do not grow memory."""
        with self._lock:
            self.samples = [s for s in self.samples if s[0] >= before]

    def summary(self, t0, t1) -> dict:
        """Means over [t0, t1) and the peak memory used (GiB) in that window."""
        with self._lock:
            rows = [s for s in self.samples if t0 <= s[0] < t1]
        if not rows:
            return {"sm": None, "mem": None, "power": None, "mem_used_peak_gb": None, "n": 0}
        a = np.array(rows)
        return {"sm": float(a[:, 1].mean()), "mem": float(a[:, 2].mean()),
                "power": float(a[:, 3].mean()), "mem_used_peak_gb": float(a[:, 4].max()),
                "n": len(rows)}
