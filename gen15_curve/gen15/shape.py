"""Gen15 shape arms: the second coefficient, which the programme has always held constant.

Every generation from gen13 on writes a cell's centred curve as ``a*r + b*r^2`` and then models
``a`` only; ``b`` is set to the training fold's mean for every cell.  The gen15 locate run measured
what that costs: giving each cell its own ``b`` and keeping gen14's amplitude takes the
extractant-macro MAE from 0.500 to 0.427 under BP (+0.073, p = 0.0006, passes P1).  That is half
the size of the amplitude oracle and none of it has ever been claimed.

Three facts make ``b`` a different problem from ``a``, not a repeat of it:

* ``b`` is uncorrelated with ``a`` (Pearson 0.03), so it is not a redundant re-parameterisation;
* its mean depends strongly on the *direction* of ``a`` -- heavy-selective cells sit at
  b = -0.009 and light-selective ones at b = -0.136, a gap of two thirds of a standard deviation --
  so a direction-conditional constant is free;
* it carries the only thing a monotone ramp cannot express, an interior extremum, which 27 % of
  well-determined cells actually have.

The arms below go up that ladder: a better constant, a direction-conditional constant, a sign
classifier for ``b`` built exactly like gen14's for ``a``, and a small alphabet of whole curve
shapes with the prototype chosen by a multiclass logistic.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .arms import _curve, _logistic_sign
from .valuebench import Ctx


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, 0.5)])


def _logit(X_tr: np.ndarray, y: np.ndarray, w: np.ndarray, X_te: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Gen14's estimator: L2 logistic on standardised, median-imputed columns."""
    if len(set(y.tolist())) < 2:
        return np.full(len(X_te), float(y.mean()))
    m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                      LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
    m.fit(X_tr, y, logisticregression__sample_weight=w)
    return m.predict_proba(X_te)[:, 1]


# --------------------------------------------------------------------------------------
# better constants for b
# --------------------------------------------------------------------------------------
def curv_constant(kind: str = "median", rich_only: bool = True):
    """A single curvature for every cell, but chosen for the loss it is judged by.

    Gen14 uses the weighted *mean* over all training cells.  Under an absolute-error metric the
    optimal constant is the weighted median, and restricting to well-determined cells removes the
    two-metal cells whose ``b`` is almost pure ridge shrinkage.
    """
    def f(ctx: Ctx) -> np.ndarray:
        tr = ctx.rich_train() if rich_only else ctx.train
        w = ctx.rich_weights() if rich_only else ctx.w
        b = ctx.cur[tr]
        val = _wmedian(b, w) if kind == "median" else float(np.average(b, weights=w))
        return _curve(_logistic_sign(ctx) * ctx.train_mean_magnitude(), val, len(ctx.test))
    return f


def curv_direction_conditional(kind: str = "median"):
    """One curvature for heavy-selective ligands and one for light-selective ones.

    The class is the *predicted* direction of the test cell and the *observed* direction of the
    training cells, so nothing about the held-out cell is used.  This is free: it adds no
    parameters beyond the two the training fold already estimates.
    """
    def f(ctx: Ctx) -> np.ndarray:
        s = _logistic_sign(ctx)
        tr, w = ctx.rich_train(), ctx.rich_weights()
        a_tr, b_tr = ctx.amp[tr], ctx.cur[tr]
        out = np.empty(len(ctx.test))
        for sign in (-1.0, 1.0):
            m = (a_tr < 0) if sign < 0 else (a_tr > 0)
            if m.sum() < 10:
                val = _wmedian(b_tr, w)
            else:
                val = _wmedian(b_tr[m], w[m]) if kind == "median" \
                    else float(np.average(b_tr[m], weights=w[m]))
            out[s == sign] = val
        return _curve(s * ctx.train_mean_magnitude(), out, len(ctx.test))
    return f


# --------------------------------------------------------------------------------------
# a classifier for the sign of b, built exactly like gen14's for a
# --------------------------------------------------------------------------------------
def curv_sign_model(features: str = "TOPO39", C: float = 1.0, direction_conditional: bool = True,
                    oracle: bool = False):
    """Call the sign of the curvature from donor topology; magnitude from the training fold.

    ``direction_conditional`` takes the magnitude of ``b`` from the training cells that share the
    test cell's *predicted* a-direction, because the two coefficients' magnitudes are not
    independent of the direction even though the coefficients themselves are uncorrelated.
    """
    def f(ctx: Ctx) -> np.ndarray:
        s = _logistic_sign(ctx)
        tr, w = ctx.rich_train(), ctx.rich_weights()
        X = ctx.feat(features)
        a_tr, b_tr = ctx.amp[tr], ctx.cur[tr]
        if oracle:
            sb = np.where(ctx.cur[ctx.test] < 0, -1.0, 1.0)
        else:
            p = _logit(X[tr], (b_tr < 0).astype(int), w, X[ctx.test], C)
            sb = np.where(p >= 0.5, -1.0, 1.0)
        mag = np.empty(len(ctx.test))
        for sign in (-1.0, 1.0):
            m = (a_tr < 0) if sign < 0 else (a_tr > 0)
            use = m if (direction_conditional and m.sum() >= 10) else np.ones(len(tr), dtype=bool)
            mag[s == sign] = float(np.average(np.abs(b_tr[use]), weights=w[use]))
        return _curve(s * ctx.train_mean_magnitude(), sb * mag, len(ctx.test))
    return f


# --------------------------------------------------------------------------------------
# a small alphabet of whole curve shapes
# --------------------------------------------------------------------------------------
def _prototypes(a: np.ndarray, b: np.ndarray, w: np.ndarray, k: int, seed: int,
                iters: int = 60) -> np.ndarray:
    """Weighted k-medians on (a, b): centres that minimise absolute, not squared, deviation."""
    P = np.c_[a, b]
    scale = np.array([np.abs(a).mean() + 1e-9, np.abs(b).mean() + 1e-9])
    Q = P / scale
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Q), size=min(k, len(Q)), replace=False, p=w / w.sum())
    C = Q[idx].copy()
    for _ in range(iters):
        d = np.abs(Q[:, None, :] - C[None, :, :]).sum(axis=2)
        lab = d.argmin(axis=1)
        new = C.copy()
        for j in range(len(C)):
            m = lab == j
            if m.sum():
                new[j] = [_wmedian(Q[m, 0], w[m]), _wmedian(Q[m, 1], w[m])]
        if np.allclose(new, C):
            break
        C = new
    return C * scale


def shape_alphabet(k: int = 4, features: str = "TOPO39", C: float = 1.0, oracle: bool = False):
    """Predict which of ``k`` canonical lanthanide curves a ligand produces.

    Stage 2 measured about a dozen effective training units for a chemotype-level quantity, so the
    honest capacity of this corpus is a handful of discrete answers, not a continuous surface.
    Gen13's one bit is this with ``k = 2`` and the alphabet fixed by hand; here the alphabet is
    learned from the training fold by weighted k-medians on (a, b) -- k-medians, not k-means,
    because the endpoint is an absolute error -- and the assignment is a multiclass logistic on the
    same donor-topology columns that call the direction.
    """
    def f(ctx: Ctx) -> np.ndarray:
        tr, w = ctx.rich_train(), ctx.rich_weights()
        X = ctx.feat(features)
        P = _prototypes(ctx.amp[tr], ctx.cur[tr], w, k, ctx.model_seed)
        scale = np.array([np.abs(ctx.amp[tr]).mean() + 1e-9, np.abs(ctx.cur[tr]).mean() + 1e-9])
        d_tr = np.abs(np.c_[ctx.amp[tr], ctx.cur[tr]][:, None, :] / scale - P[None] / scale).sum(2)
        y = d_tr.argmin(axis=1)
        if oracle:
            te = np.c_[ctx.amp[ctx.test], ctx.cur[ctx.test]]
            lab = np.abs(te[:, None, :] / scale - P[None] / scale).sum(2).argmin(axis=1)
        else:
            if len(set(y.tolist())) < 2:
                lab = np.full(len(ctx.test), int(y[0]))
            else:
                m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler(),
                                  LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
                m.fit(X[tr], y, logisticregression__sample_weight=w)
                lab = m.predict(X[ctx.test])
        return P[np.asarray(lab, dtype=int)]
    return f
