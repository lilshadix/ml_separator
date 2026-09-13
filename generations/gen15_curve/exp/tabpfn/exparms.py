"""Arms: a direction, a magnitude and a curvature composed exactly the way gen14 composes them.

Gen14's deployed arm is ``sign(direction) * mean|a| of the training fold``, with ``b`` held at the
training fold's mean.  Every arm here changes **one** of those three slots and leaves the other two
at gen14's setting, so a difference in the MAE table is attributable to the learner and not to the
composition rule.  ``ONE_SHOT`` composes all three at once, which is the only arm allowed to move
more than one part.
"""
from __future__ import annotations

import numpy as np

import common as K
from common import V                       # noqa: F401  (keeps sys.path set up)
from gen15 import arms as GA


_G14 = {"key": None, "sign": None}


def g14_sign(ctx) -> np.ndarray:
    """Gen14's logistic direction call, memoised so several arms share one fit per fold."""
    k = (ctx.design, ctx.seed, ctx.fold)
    if _G14["key"] != k:
        _G14["key"], _G14["sign"] = k, GA._logistic_sign(ctx)
    return _G14["sign"]


def compose(name: str, rec: K.Recorder, *, dir_fn=None, mag_fn=None, curv_fn=None):
    """Build a valuebench arm out of at most one replacement part."""
    def arm(ctx) -> np.ndarray:
        n = len(ctx.test)
        if dir_fn is None:
            s = g14_sign(ctx)
        else:
            p = np.asarray(dir_fn(ctx), dtype=float).ravel()
            s = np.where(p >= 0.5, -1.0, 1.0)
        mag = (np.full(n, ctx.train_mean_magnitude()) if mag_fn is None
               else np.asarray(mag_fn(ctx), dtype=float).ravel())
        b = (np.full(n, ctx.train_mean_curvature()) if curv_fn is None
             else np.asarray(curv_fn(ctx), dtype=float).ravel())
        a = s * mag
        rec.log(ctx, name, a)
        return np.c_[a, b]
    return arm


def reference(name: str, fn, rec: K.Recorder):
    """Wrap an existing gen15 arm so its direction accuracy is recorded on the same folds."""
    def arm(ctx) -> np.ndarray:
        coef = np.asarray(fn(ctx), dtype=float).reshape(len(ctx.test), 2)
        rec.log(ctx, name, coef[:, 0])
        return coef
    return arm


# --------------------------------------------------------------------------------------
# TabPFN, read from the precomputed out-of-fold cache
# --------------------------------------------------------------------------------------
class TPCache:
    def __init__(self, path):
        import pickle
        with open(path, "rb") as fh:
            self.store = pickle.load(fh)

    def has(self, ctx) -> bool:
        r = self.store.get((ctx.design, ctx.seed, ctx.fold))
        return bool(r) and not r.get("skip", False)

    def get(self, ctx, key: str) -> np.ndarray | None:
        r = self.store.get((ctx.design, ctx.seed, ctx.fold))
        if not r or r.get("skip", False):
            return None
        if not np.array_equal(np.asarray(r["test"]), np.asarray(ctx.test)):
            raise RuntimeError("cached fold does not match the fold plan")
        return np.asarray(r[key], dtype=float)

    def dir_fn(self, key: str):
        def f(ctx):
            v = self.get(ctx, key)
            if v is not None:
                return v
            return np.where(g14_sign(ctx) < 0, 0.9, 0.1)      # fall back to the deployed call
        return f

    def mag_fn(self):
        def f(ctx):
            v = self.get(ctx, "mag")
            return np.full(len(ctx.test), ctx.train_mean_magnitude()) if v is None else v
        return f

    def curv_fn(self):
        def f(ctx):
            v = self.get(ctx, "curv")
            return np.full(len(ctx.test), ctx.train_mean_curvature()) if v is None else v
        return f
