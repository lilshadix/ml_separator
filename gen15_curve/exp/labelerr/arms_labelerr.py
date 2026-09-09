"""Arms that change only the *label* the gen14 model is fitted to, or the weight it is fitted with.

Every arm here is gen14's deployed predictor -- an L2 logistic on the 39 donor-topology columns
calls the direction, the training fold's mean magnitude sets the size, the training fold's mean
curvature sets ``b`` -- with exactly one thing swapped:

``dir_label``    the direction target: ``sign(a_raw)`` (gen14) or ``sign(a_shrunk)``;
``mag_source``   the magnitude constant: ``mean|a_raw|`` (gen14), ``mean|a_shrunk|``, or the
                 moment-de-attenuated ``mean sqrt(max(a^2 - var_a, 0))``;
``weights``      chemotype-balanced (gen14) or additionally precision-weighted by
                 ``1 / (tau^2 + var(a_i))`` with the sampling variance computed from the ridge hat
                 matrix, renormalised inside each chemotype so chemotype shares never move.

The hierarchical shrinkage is fitted on the fold's TRAINING cells only and cached per fold, so no
test cell can enter any prior, group mean or covariance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from noise import coef_cov, shrink, sigma_table  # noqa: E402
from gen15.arms import _curve  # noqa: E402
from gen15.valuebench import Ctx  # noqa: E402

_SE: dict[str, np.ndarray] = {}
_VAR: dict[str, np.ndarray] = {}
_SHRUNK: dict[tuple, np.ndarray] = {}


def se_of(ctx: Ctx, mode: str) -> np.ndarray:
    if mode not in _SE:
        _SE[mode] = sigma_table(ctx.bench, mode)
    return _SE[mode]


def var_of(ctx: Ctx, mode: str) -> np.ndarray:
    """Sampling covariance of every cell's (a, b) -- depends only on the frozen cohort."""
    if mode not in _VAR:
        _VAR[mode] = coef_cov(ctx.bench, se_of(ctx, mode))
    return _VAR[mode]


def shrunk_of(ctx: Ctx, mode: str, group: str) -> np.ndarray:
    """EB-de-noised coefficients, fitted on this fold's training cells only."""
    key = (ctx.design, ctx.seed, ctx.fold, mode, group, len(ctx.train))
    if key not in _SHRUNK:
        out, _ = shrink(ctx.bench, ctx.train, se_of(ctx, mode), group=group)
        _SHRUNK[key] = out
    return _SHRUNK[key]


def _precision_weights(w: np.ndarray, var_a: np.ndarray, groups: np.ndarray,
                       tau2: float) -> np.ndarray:
    raw = w / (tau2 + np.maximum(var_a, 0.0))
    out = raw.copy()
    for g in np.unique(groups):
        m = groups == g
        s = raw[m].sum()
        out[m] = raw[m] * (w[m].sum() / s) if s > 1e-12 else w[m]
    return out


def _logistic_sign(ctx: Ctx, amp_label: np.ndarray, weights: np.ndarray,
                   features: str = "TOPO39", C: float = 1.0) -> np.ndarray:
    rtr = ctx.rich_train()
    X = ctx.feat(features)
    y = (amp_label[rtr] < 0).astype(int)
    if len(set(y.tolist())) < 2:
        p = np.full(len(ctx.test), float(y.mean()))
    else:
        m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(),
                          LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
        m.fit(X[rtr], y, logisticregression__sample_weight=weights)
        p = m.predict_proba(X[ctx.test])[:, 1]
    return np.where(np.asarray(p) >= 0.5, -1.0, 1.0)


def labelerr_arm(dir_label: str = "raw", mag_source: str = "raw", weights: str = "balanced",
                 sigma_mode: str = "resid_const", group: str = "extractant",
                 curv: str = "raw", tau2: float = 0.02):
    """gen14 with the label, the magnitude constant and/or the fitting weight swapped."""
    def f(ctx: Ctx) -> np.ndarray:
        rtr, w = ctx.rich_train(), ctx.rich_weights()
        a_raw = ctx.amp
        need_shrunk = "shrunk" in (dir_label, mag_source, curv)
        a_sh = shrunk_of(ctx, sigma_mode, group) if need_shrunk else None
        lab = a_sh[:, 0] if dir_label == "shrunk" else a_raw
        ww = w
        if weights == "precision":
            va = var_of(ctx, sigma_mode)[rtr, 0, 0]
            ww = _precision_weights(w, va, ctx.bench.groups[rtr], tau2)
        s = _logistic_sign(ctx, lab, ww)
        if mag_source == "raw":
            mag = float(np.average(np.abs(a_raw[rtr]), weights=w))
        elif mag_source == "shrunk":
            mag = float(np.average(np.abs(a_sh[rtr, 0]), weights=w))
        elif mag_source == "deattenuated":
            va = var_of(ctx, sigma_mode)[rtr, 0, 0]
            mag = float(np.average(np.sqrt(np.maximum(a_raw[rtr] ** 2 - va, 0.0)), weights=w))
        else:
            raise ValueError(mag_source)
        if curv == "raw":
            b = float(np.average(ctx.cur[ctx.train], weights=ctx.w))
        elif curv == "shrunk":
            b = float(np.average(a_sh[ctx.train, 1], weights=ctx.w))
        else:
            raise ValueError(curv)
        return _curve(s * mag, b, len(ctx.test))
    return f


def rich_threshold_arm(min_metals: int = 8, weights: str = "balanced",
                       sigma_mode: str = "resid_const", tau2: float = 0.02,
                       max_se_a: float | None = None):
    """gen14 with a different definition of "well determined" for the training cells.

    The sharpest form of the precision idea: instead of weighting a noisy label down, drop it.
    ``min_metals`` raises gen14's >= 5 cut; ``max_se_a`` cuts on the *computed* standard error of
    the cell's own ``a`` instead, which is the precision-native version of the same filter.
    Test cells are untouched -- only the training set the direction and the magnitude are fitted on.
    """
    from gen13sep.amplitude_bench import cell_weights

    def f(ctx: Ctx) -> np.ndarray:
        m = ctx.bench.frame.n_metals.to_numpy()
        keep = m >= min_metals
        if max_se_a is not None:
            keep &= np.sqrt(var_of(ctx, sigma_mode)[:, 0, 0]) <= max_se_a
        tr = ctx.train[keep[ctx.train]]
        if len(tr) < 30:
            tr = ctx.rich_train()
        w = cell_weights(ctx.bench.groups[tr], ctx.bench.n_obs[tr])
        ww = w
        if weights == "precision":
            ww = _precision_weights(w, var_of(ctx, sigma_mode)[tr, 0, 0],
                                    ctx.bench.groups[tr], tau2)
        X = ctx.feat("TOPO39")
        y = (ctx.amp[tr] < 0).astype(int)
        if len(set(y.tolist())) < 2:
            p = np.full(len(ctx.test), float(y.mean()))
        else:
            mdl = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"))
            mdl.fit(X[tr], y, logisticregression__sample_weight=ww)
            p = mdl.predict_proba(X[ctx.test])[:, 1]
        s = np.where(np.asarray(p) >= 0.5, -1.0, 1.0)
        mag = float(np.average(np.abs(ctx.amp[tr]), weights=w))
        b = float(np.average(ctx.cur[ctx.train], weights=ctx.w))
        return _curve(s * mag, b, len(ctx.test))
    return f
