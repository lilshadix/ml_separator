"""Zero-shot curve arms behind one interface, so every adapter and every covariance builder
treats them alike.

    arm.fit(ctx)                      fit on ctx.train (features of all cells are visible, labels of
                                      training cells only)
    arm.curves(idx) -> (len(idx), 14) centred zero-shot curves for any cells

``Gen15Arm`` wraps a gen15 ``arm(ctx) -> coef`` function (G14, FLAT, MEAN_CURVE, G13_FULL).
``NNTanimoto`` is the nearest training extractant's own ridge curve (the lookup baseline; the
chemotype hold-out is single-linkage Tanimoto 0.7 on the same bits, so the nearest neighbour is
never a chemotype sibling).  The neural contrast arm lives in ``nnc.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from . import bootstrap  # noqa: F401
from .data import N_LN
from gen15 import arms as A15  # noqa: E402
from gen15.valuebench import Ctx  # noqa: E402


class Gen15Arm:
    def __init__(self, name: str, fn: Callable[[Ctx], np.ndarray]):
        self.name, self.fn, self.ctx = name, fn, None

    def fit(self, ctx: Ctx):
        self.ctx = ctx
        return self

    def curves(self, idx) -> np.ndarray:
        idx = np.asarray(idx)
        ctx = replace(self.ctx, test=idx)
        coef = np.asarray(self.fn(ctx), dtype=float).reshape(len(idx), self.ctx.bench.basis.shape[0])
        return coef @ self.ctx.bench.basis

    def prior_coef(self) -> np.ndarray:
        """Training-fold mean coefficients (for the ridge line prior)."""
        return np.array([self.ctx.train_mean_amplitude(), self.ctx.train_mean_curvature()])


class NNTanimoto:
    """Nearest training extractant by ECFP Tanimoto; its chemotype-weighted mean ridge curve."""
    name = "NN_TANIMOTO"

    def __init__(self):
        self.ctx = None

    def fit(self, ctx: Ctx):
        self.ctx = ctx
        fp = ctx.bench.frames["ECFP"].to_numpy(dtype=float) > 0.5
        ext = ctx.bench.frame.extractant.to_numpy()
        tr = ctx.train
        names, first = np.unique(ext[tr], return_index=True)
        self.train_ext = names
        self.train_fp = fp[tr][first]
        coef = ctx.bench.coef
        self.train_curve = np.vstack([np.average(coef[tr][ext[tr] == e], axis=0, weights=ctx.w[ext[tr] == e]) @ ctx.bench.basis
                                      for e in names])
        self.fp = fp
        return self

    def curves(self, idx) -> np.ndarray:
        te = self.fp[np.asarray(idx)].astype(np.int32)
        tr = self.train_fp.astype(np.int32)
        inter = te @ tr.T
        union = te.sum(1)[:, None] + tr.sum(1)[None, :] - inter
        sim = inter / np.maximum(union, 1)
        out = np.zeros((len(te), N_LN))
        for i in range(len(te)):
            best = np.flatnonzero(sim[i] == sim[i].max())
            out[i] = self.train_curve[best].mean(axis=0)
        return out


def as_fit_curves(make_arm: Callable[[], object]) -> Callable[[Ctx], np.ndarray]:
    """Turn an arm factory into ``fit_curves(ctx)`` for the covariance builder."""
    def f(ctx: Ctx) -> np.ndarray:
        return make_arm().fit(ctx).curves(ctx.test)
    return f


def standard_arms() -> dict[str, Callable[[], object]]:
    return {
        "FLAT": lambda: Gen15Arm("FLAT", A15.flat),
        "G14": lambda: Gen15Arm("G14", A15.g14),
        "MEAN_CURVE": lambda: Gen15Arm("MEAN_CURVE", A15.mean_curve),
        "NN_TANIMOTO": lambda: NNTanimoto(),
    }
