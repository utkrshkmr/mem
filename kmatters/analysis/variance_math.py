"""Streaming two-level variance decomposition of per-future gradient vectors.

Z_ik = gradient contribution of future k of prefix i (a vector).
V_future = E[ tr Cov(Z | X) ]          (within-prefix)
V_prefix = tr Cov( E[Z | X] )          (between-prefix)
MSE(N, K) = V_prefix / N + V_future / (N K)
"""
from __future__ import annotations

import numpy as np


class VarianceAccumulator:
    """Feed one prefix at a time: add_prefix(list_of_K0_vectors).

    Keeps per-prefix mean vectors (rows of M) to form a Gram matrix for exact
    point estimates and a prefix-level bootstrap. Works with numpy or torch
    arrays via the `xp` shim (only dot/sum are needed).
    """

    def __init__(self):
        self.means = []      # m_i vectors
        self.within = []     # per-prefix unbiased within variance (trace)
        self.K0 = None

    def add_prefix(self, Z_list):
        K0 = len(Z_list)
        if K0 < 2:
            raise ValueError("need K0 >= 2 futures per prefix")
        if self.K0 is None:
            self.K0 = K0
        elif K0 != self.K0:
            raise ValueError("all prefixes must have the same K0")
        S = None
        q = 0.0
        for z in Z_list:
            z = np.asarray(z, dtype=np.float64)
            S = z.copy() if S is None else S + z
            q += float(z @ z)
        self.within.append((q - float(S @ S) / K0) / (K0 - 1))
        self.means.append(S / K0)

    def gram(self):
        M = np.stack(self.means)
        return M @ M.T

    @staticmethod
    def _estimates(G, within, K0, idx=None):
        n = G.shape[0]
        if idx is None:
            idx = np.arange(n)
        Gs = G[np.ix_(idx, idx)]
        W = (np.trace(Gs) - Gs.sum() / len(idx)) / (len(idx) - 1)
        Vf = float(np.mean(np.asarray(within)[idx]))
        Vp = float(W - Vf / K0)
        return Vp, Vf

    def estimates(self):
        return self._estimates(self.gram(), self.within, self.K0)

    def bootstrap(self, B=2000, seed=0):
        G = self.gram()
        n = G.shape[0]
        rng = np.random.default_rng(seed)
        out = np.empty((B, 2))
        for b in range(B):
            out[b] = self._estimates(G, self.within, self.K0, rng.integers(0, n, n))
        return out   # columns: Vp, Vf


def mse_model(Vp, Vf, N, K):
    return Vp / N + Vf / (N * K)


class TorchVarianceAccumulator:
    """Same estimator as VarianceAccumulator for large torch vectors (e.g. 33M LoRA grads).

    Per-prefix sums are accumulated in float64; per-prefix mean vectors are stored
    in float32 on `store_device`; the Gram matrix is built in float64 in column
    chunks so no N0 x P float64 matrix is ever materialized.
    """

    def __init__(self, store_device=None, chunk=1 << 22):
        self.means = []
        self.within = []
        self.K0 = None
        self.store_device = store_device
        self.chunk = chunk

    def add_prefix(self, Z_list):
        import torch
        K0 = len(Z_list)
        if K0 < 2:
            raise ValueError("need K0 >= 2 futures per prefix")
        if self.K0 is None:
            self.K0 = K0
        elif K0 != self.K0:
            raise ValueError("all prefixes must have the same K0")
        S = torch.zeros_like(Z_list[0], dtype=torch.float64)
        q = 0.0
        for z in Z_list:
            z64 = z.to(torch.float64)
            S += z64
            q += float(torch.dot(z64, z64))
        self.within.append((q - float(torch.dot(S, S)) / K0) / (K0 - 1))
        m = (S / K0).to(torch.float32)
        self.means.append(m if self.store_device is None else m.to(self.store_device))

    def gram(self):
        import torch
        n = len(self.means)
        G = torch.zeros((n, n), dtype=torch.float64, device=self.means[0].device)
        P = self.means[0].numel()
        for s in range(0, P, self.chunk):
            blk = torch.stack([m[s:s + self.chunk] for m in self.means]).to(torch.float64)
            G += blk @ blk.T
        return G.cpu().numpy()

    def estimates(self):
        return VarianceAccumulator._estimates(self.gram(), self.within, self.K0)

    def bootstrap(self, B=2000, seed=0):
        G = self.gram()
        n = G.shape[0]
        rng = np.random.default_rng(seed)
        out = np.empty((B, 2))
        for b in range(B):
            out[b] = VarianceAccumulator._estimates(G, self.within, self.K0, rng.integers(0, n, n))
        return out
