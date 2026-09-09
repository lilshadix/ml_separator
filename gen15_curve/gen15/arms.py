"""Gen15 arms: references, oracles that locate information, and the first feasible candidates.

An *oracle* here is an arm that is allowed to see a quantity a deployed model would not have, in
order to answer "is there anything in it at all".  Every oracle is named ``O_*`` and can never be a
deployed arm.  The three that matter:

* ``O_AMP``  the cell's own radius coefficient -- the ceiling gen14 measured (0.32 under BP).
* ``O_LEVEL`` the magnitude as a fitted function of the cell's own *level* of log D, which the
  centred target is algebraically independent of.  A gain here says the mass-action coupling
  between how strongly a system extracts and how steeply it discriminates is worth chasing with a
  log-D model; no gain kills that whole route.
* ``O_PUBAMP`` the mean magnitude of the cell's own publication -- the ceiling on anything the 64
  condition columns could recover, since they identify the publication at 94 %.

``QUERY_SPAN`` and ``N_METALS`` are *not* oracles: which metals a query asks about is known before
any experiment is done.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .valuebench import Ctx          # inserts gen13_separation/ and gen14_direction/ on sys.path
from gen14.models import dir_logistic  # noqa: E402

EPS = 0.05


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _logistic_sign(ctx: Ctx, features: str = "TOPO39") -> np.ndarray:
    """Gen14's deployed direction call: -1 heavy-selective, +1 light-selective."""
    rtr = ctx.rich_train()
    X = ctx.feat(features)
    p = dir_logistic()(X[rtr], ctx.amp[rtr], ctx.rich_weights(), ctx.bench.groups[rtr],
                       X[ctx.test], ctx.model_seed, ctx.bench.frame.extractant.to_numpy()[rtr])
    return np.where(np.asarray(p) >= 0.5, -1.0, 1.0)


def _fit_magnitude_1d(v_tr: np.ndarray, mag_tr: np.ndarray, w_tr: np.ndarray, v_te: np.ndarray,
                      how: str = "isotonic") -> np.ndarray:
    """Monotone or log-linear map from one covariate to the magnitude, fitted on the training fold."""
    ok = np.isfinite(v_tr) & np.isfinite(mag_tr)
    if ok.sum() < 20:
        return np.full(len(v_te), float(np.average(mag_tr, weights=w_tr)))
    fill = float(np.median(v_tr[ok]))
    vt = np.where(np.isfinite(v_te), v_te, fill)
    if how == "isotonic":
        best, best_loss = None, np.inf
        for inc in (True, False):
            iso = IsotonicRegression(increasing=inc, out_of_bounds="clip")
            iso.fit(v_tr[ok], mag_tr[ok], sample_weight=w_tr[ok])
            loss = float(np.average(np.abs(iso.predict(v_tr[ok]) - mag_tr[ok]), weights=w_tr[ok]))
            if loss < best_loss:
                best, best_loss = iso, loss
        return np.clip(best.predict(vt), 0.0, None)
    y = np.log(mag_tr[ok] + EPS)
    A = np.c_[np.ones(int(ok.sum())), v_tr[ok]]
    sw = np.sqrt(w_tr[ok])
    beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return np.clip(np.exp(beta[0] + beta[1] * vt) - EPS, 0.0, None)


def _cell_level(ctx: Ctx, stat: str) -> np.ndarray:
    Y = ctx.bench.Y
    with np.errstate(invalid="ignore"):
        if stat == "mean":
            return np.nanmean(Y, axis=1)
        if stat == "max":
            return np.nanmax(Y, axis=1)
        if stat == "range":
            return np.nanmax(Y, axis=1) - np.nanmin(Y, axis=1)
    raise ValueError(stat)


def _radius_span(ctx: Ctx) -> np.ndarray:
    """Spread of the standardised Shannon radius over the metals a cell actually measured."""
    r = ctx.bench.basis[0]
    obs = ~np.isnan(ctx.bench.Y)
    return np.array([float(r[o].max() - r[o].min()) if o.sum() > 1 else 0.0 for o in obs])


def _curve(a, b, n: int) -> np.ndarray:
    b = np.full(n, float(b)) if np.isscalar(b) else np.asarray(b, dtype=float)
    return np.c_[np.asarray(a, dtype=float), b]


# --------------------------------------------------------------------------------------
# reference arms
# --------------------------------------------------------------------------------------
def flat(ctx: Ctx) -> np.ndarray:
    """Predict no separation at all.  The floor every gain in this programme should be quoted over."""
    return np.zeros((len(ctx.test), 2))


def mean_curve(ctx: Ctx) -> np.ndarray:
    return _curve(np.full(len(ctx.test), ctx.train_mean_amplitude()),
                  ctx.train_mean_curvature(), len(ctx.test))


def g14(ctx: Ctx) -> np.ndarray:
    """The deployed gen14 arm: logistic direction x the training fold's mean magnitude."""
    return _curve(_logistic_sign(ctx) * ctx.train_mean_magnitude(),
                  ctx.train_mean_curvature(), len(ctx.test))


def g13_full(ctx: Ctx) -> np.ndarray:
    """Gen13's 209-column extra-trees regression on both coefficients."""
    from gen13sep.models import tree_pipeline
    m = tree_pipeline(ctx.model_seed, n_estimators=400, max_features=0.5, min_samples_leaf=2)
    m.fit(ctx.X[ctx.train], ctx.bench.coef[ctx.train], extratreesregressor__sample_weight=ctx.w)
    return np.asarray(m.predict(ctx.X[ctx.test]), dtype=float).reshape(len(ctx.test), 2)


# --------------------------------------------------------------------------------------
# oracles
# --------------------------------------------------------------------------------------
def o_sign(ctx: Ctx) -> np.ndarray:
    return _curve(np.sign(ctx.amp[ctx.test]) * ctx.train_mean_magnitude(),
                  ctx.train_mean_curvature(), len(ctx.test))


def o_amp(ctx: Ctx) -> np.ndarray:
    return _curve(ctx.amp[ctx.test], ctx.train_mean_curvature(), len(ctx.test))


def o_both(ctx: Ctx) -> np.ndarray:
    return ctx.bench.coef[ctx.test]


def o_curvature(ctx: Ctx) -> np.ndarray:
    """Gen14's amplitude with the cell's own curvature: what the second coefficient is worth."""
    return _curve(_logistic_sign(ctx) * ctx.train_mean_magnitude(), ctx.cur[ctx.test], len(ctx.test))


def _group_oracle_magnitude(ctx: Ctx, column: str) -> np.ndarray:
    """Magnitude = mean |amplitude| of the test cell's own group, other cells only."""
    g = ctx.bench.frame[column].astype(str).to_numpy()
    mag = np.abs(ctx.amp)
    out = np.full(len(ctx.test), ctx.train_mean_magnitude())
    for j, ci in enumerate(ctx.test):
        peers = np.flatnonzero((g == g[ci]) & ctx.rich)
        peers = peers[peers != ci]
        if len(peers):
            out[j] = float(mag[peers].mean())
    return out


def o_extractant_magnitude(ctx: Ctx) -> np.ndarray:
    return _curve(_logistic_sign(ctx) * _group_oracle_magnitude(ctx, "extractant"),
                  ctx.train_mean_curvature(), len(ctx.test))


def o_publication_magnitude(ctx: Ctx) -> np.ndarray:
    return _curve(_logistic_sign(ctx) * _group_oracle_magnitude(ctx, "publication_id"),
                  ctx.train_mean_curvature(), len(ctx.test))


def o_level(stat: str = "mean", how: str = "isotonic"):
    """Magnitude as a fitted monotone function of the cell's own level of log D.

    The centred target is invariant to the level, so this is not label leakage in the pairwise
    sense; it is a question about whether a log-D model would buy anything for the *shape*.
    """
    def f(ctx: Ctx) -> np.ndarray:
        v = _cell_level(ctx, stat)
        rtr = ctx.rich_train()
        mag = _fit_magnitude_1d(v[rtr], np.abs(ctx.amp[rtr]), ctx.rich_weights(), v[ctx.test], how)
        return _curve(_logistic_sign(ctx) * mag, ctx.train_mean_curvature(), len(ctx.test))
    return f


# --------------------------------------------------------------------------------------
# feasible candidates
# --------------------------------------------------------------------------------------
def query_span_magnitude(how: str = "isotonic"):
    """Magnitude from the *query*: how wide a radius span the metals asked about cover.

    Not an oracle -- a deployment knows which separation it is being asked to predict before any
    experiment exists.  The ridge that defines the label shrinks a narrow-coverage cell's
    coefficient, and narrow-coverage cells are also the ones a laboratory measures when the trend
    is weak, so the training fold can calibrate the magnitude against the query itself.
    """
    def f(ctx: Ctx) -> np.ndarray:
        v = _radius_span(ctx)
        rtr = ctx.rich_train()
        mag = _fit_magnitude_1d(v[rtr], np.abs(ctx.amp[rtr]), ctx.rich_weights(), v[ctx.test], how)
        return _curve(_logistic_sign(ctx) * mag, ctx.train_mean_curvature(), len(ctx.test))
    return f


def n_metals_magnitude(how: str = "isotonic"):
    def f(ctx: Ctx) -> np.ndarray:
        v = ctx.bench.n_obs.astype(float)
        rtr = ctx.rich_train()
        mag = _fit_magnitude_1d(v[rtr], np.abs(ctx.amp[rtr]), ctx.rich_weights(), v[ctx.test], how)
        return _curve(_logistic_sign(ctx) * mag, ctx.train_mean_curvature(), len(ctx.test))
    return f
