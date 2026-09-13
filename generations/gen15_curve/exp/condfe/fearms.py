"""Within-publication (fixed-effects) condition arms for the gen15 bench.

The programme's finding is that the 64 ``cond__`` columns are a laboratory fingerprint: they name a
cell's publication with 94 % 1-NN accuracy, so a conditions-only model collapses under design BP.
The econometric fix is the within-transformation (Frisch-Waugh-Lovell): estimate the effect of a
condition on the magnitude using only variation *inside* a publication, where the laboratory is held
fixed, then apply that estimate to a held-out cell with the publication effect set to zero.

Every arm here is gen14 with its constant magnitude replaced by a *modulated* magnitude,

    |a| = mean_magnitude(train) * exp(beta . (x - xbar))          (target "logmag")
    |a| = mean_magnitude(train) + beta . (x - xbar)               (target "mag")

so beta = 0 reproduces ``gen15.arms.g14`` exactly and any change in the score is attributable to
beta alone.  The direction bit and the curvature are gen14's untouched, except in ``fe_curv``.

Four estimators of beta, as the task asks:

``fe_arm``      the within (fixed-effects) estimator: demean every column and the target by their
                weighted publication mean inside the training fold, then a ridge through the origin.
``resid_arm``   the same quantity by the literal Frisch-Waugh-Lovell two-step -- regress the columns
                on publication dummies, regress the target on publication dummies, fit
                residual-on-residual.  Algebraically identical to ``fe_arm``; kept as an
                independent implementation so the identity is *verified*, not assumed.
``ri_arm``      a publication random intercept.  statsmodels is not installed, so the mixed model is
                its exact Gaussian equivalent: one ridge on [1, Z, publication dummies] whose
                penalty on the dummies is lambda = sigma_within^2 / sigma_between^2 from a weighted
                one-way ANOVA of the target inside the training fold, i.e. the BLUP shrinkage.
                Prediction sets every dummy to zero.
``pooled_arm``  no publication handling at all -- the confounded twin the programme already knows
                collapses under BP.  Here so the within-transformation is scored against the naive
                model it is supposed to repair, not only against gen14.

Oracles (``O_*``) bound the route and are never deployable:
  ``o_pubmag``   magnitude = the cell's own publication's mean |a| -- what conditions could at best
                 recover if they were a perfect publication fingerprint;
  ``o_within``   magnitude = the training mean plus the cell's own *deviation from its publication
                 mean* -- the exact ceiling of the within-transformation, because that deviation is
                 the only thing a within estimate can ever predict for an unseen laboratory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for _p in (str(ROOT / "generations" / "gen15_curve"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gen15.arms import _curve, _logistic_sign  # noqa: E402
from gen15.valuebench import Ctx  # noqa: E402
import conds as CD  # noqa: E402

EPS = 0.05
_CACHE: dict = {}
_SIGN: dict = {}


# ---------------------------------------------------------------------------------------------
# condition matrices, built once per bench;  gen14's direction bit, cached once per fold
# ---------------------------------------------------------------------------------------------
def cond_frame(ctx: Ctx):
    key = id(ctx.bench)
    if key not in _CACHE:
        _CACHE[key] = CD.build(ctx.bench)
    return _CACHE[key]


def design_matrix(ctx: Ctx, columns) -> np.ndarray:
    return cond_frame(ctx)[list(columns)].to_numpy(dtype=float)


def cond64(ctx: Ctx) -> np.ndarray:
    return ctx.bench.frames["COND"].to_numpy(dtype=float)


def pubs(ctx: Ctx) -> np.ndarray:
    return ctx.bench.frame["publication_id"].astype(str).to_numpy()


def sign(ctx: Ctx) -> np.ndarray:
    """Gen14's deployed direction call, memoised per fold (it is identical for every arm here)."""
    key = (id(ctx.bench), ctx.design, ctx.seed, ctx.fold)
    if key not in _SIGN:
        _SIGN[key] = _logistic_sign(ctx)
    return _SIGN[key]


def reset_caches() -> None:
    _CACHE.clear()
    _SIGN.clear()


# ---------------------------------------------------------------------------------------------
# machinery
# ---------------------------------------------------------------------------------------------
def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(v, weights=w))


def _impute(X_all: np.ndarray, rtr: np.ndarray) -> np.ndarray:
    """Fill NaN with the *training fold's* column median; all-missing columns become zero."""
    X = np.asarray(X_all, dtype=float).copy()
    med = np.nanmedian(X[rtr], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(med, np.nonzero(bad)[1])
    return X


def _demean_rows(X: np.ndarray, g: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Subtract the weighted publication mean from every row: the within-transformation."""
    out = X.astype(float).copy()
    for u in np.unique(g):
        m = g == u
        out[m] -= np.average(X[m], axis=0, weights=w[m])
    return out


def _demean_vec(y: np.ndarray, g: np.ndarray, w: np.ndarray) -> np.ndarray:
    out = y.astype(float).copy()
    for u in np.unique(g):
        m = g == u
        out[m] -= _wmean(y[m], w[m])
    return out


def _wridge0(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Weighted ridge through the origin (the within-transformation removes the intercept)."""
    sw = np.sqrt(w)
    A = X * sw[:, None]
    G = A.T @ A + lam * np.eye(X.shape[1])
    return np.linalg.solve(G, A.T @ (y * sw))


def _standardise(Xtr: np.ndarray, w: np.ndarray):
    mu = np.average(Xtr, axis=0, weights=w)
    sd = np.sqrt(np.average((Xtr - mu) ** 2, axis=0, weights=w))
    return mu, np.where(sd < 1e-9, 1.0, sd)


def _within_beta(Z: np.ndarray, y: np.ndarray, w: np.ndarray, g: np.ndarray, lam: float,
                 min_cells: int = 0):
    """beta of the within (fixed-effects) regression, and which columns were identified at all.

    A column that never moves inside any publication is demeaned to exactly zero, gets beta = 0 by
    construction, and is reported in ``keep`` -- that is the honest answer for a condition the
    corpus only ever varies between laboratories.

    ``min_cells`` drops publications with fewer than that many cells from the *identification*.  It
    exists because the pooled within estimate of the acid effect is 70 % carried by one two-cell
    publication whose two cells differ in acid identity and extractant concentration as well as in
    acid concentration -- an experiment change dressed as a condition contrast.
    """
    if min_cells > 1:
        cnt = pd_value_counts(g)
        use = np.array([cnt[v] >= min_cells for v in g], dtype=bool)
        if use.sum() < 20:
            use = np.ones(len(g), dtype=bool)
        Z, y, w, g = Z[use], y[use], w[use], g[use]
    Zt = _demean_rows(Z, g, w)
    yt = _demean_vec(y, g, w)
    keep = np.std(Zt, axis=0) > 1e-9
    beta = np.zeros(Z.shape[1])
    if keep.sum():
        beta[keep] = _wridge0(Zt[:, keep], yt, w, lam)
    return beta, keep


def pd_value_counts(g: np.ndarray) -> dict:
    u, c = np.unique(g, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))


def _meta_beta(Z: np.ndarray, y: np.ndarray, w: np.ndarray, g: np.ndarray, lam: float,
               min_cells: int = 4):
    """A within slope fitted *separately inside every publication*, then combined by the median.

    The pooled within regression weights a publication by the variance of its own conditions, which
    in this corpus hands most of the estimate to one or two series with a very wide nominal range.
    A per-publication fit followed by a column-wise median gives every laboratory one vote, which is
    the estimator that matches the diagnostic (a pooled within Spearman of +0.36 for acid) and is
    what "the effect the typical publication agrees on" actually means.
    """
    cnt = pd_value_counts(g)
    betas, ident = [], np.zeros(Z.shape[1], dtype=int)
    for u in np.unique(g):
        m = g == u
        if int(m.sum()) < max(min_cells, 2):
            continue
        Zp = Z[m] - np.average(Z[m], axis=0, weights=w[m])
        yp = y[m] - _wmean(y[m], w[m])
        k = np.std(Zp, axis=0) > 1e-9
        if not k.any():
            continue
        bp = np.zeros(Z.shape[1])
        bp[k] = _wridge0(Zp[:, k], yp, w[m], lam)
        bp[~k] = np.nan
        ident += k.astype(int)
        betas.append(bp)
    beta = np.zeros(Z.shape[1])
    keep = np.zeros(Z.shape[1], dtype=bool)
    if betas:
        Bm = np.vstack(betas)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(Bm, axis=0)
        keep = ident >= 2
        beta = np.where(keep & np.isfinite(med), med, 0.0)
    _ = cnt
    return beta, keep


def _resid_beta(Z: np.ndarray, y: np.ndarray, w: np.ndarray, g: np.ndarray, lam: float):
    """Literal Frisch-Waugh-Lovell: residualise on publication dummies, then residual-on-residual.

    Independent of ``_within_beta`` on purpose; ``check_fwl_identity`` asserts they agree.
    """
    u = np.unique(g)
    D = (g[:, None] == u[None, :]).astype(float)
    sw = np.sqrt(w)
    Dw = D * sw[:, None]
    G = Dw.T @ Dw + 1e-10 * np.eye(D.shape[1])

    def resid(M: np.ndarray) -> np.ndarray:
        M = np.atleast_2d(M.T).T
        c = np.linalg.solve(G, Dw.T @ (M * sw[:, None]))
        return M - D @ c

    Zt, yt = resid(Z), resid(y.reshape(-1, 1)).ravel()
    keep = np.std(Zt, axis=0) > 1e-9
    beta = np.zeros(Z.shape[1])
    if keep.sum():
        beta[keep] = _wridge0(Zt[:, keep], yt, w, lam)
    return beta, keep


SHRINK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _estimator(name: str, min_cells: int):
    if name == "fe":
        return lambda Z, y, w, g, lam: _within_beta(Z, y, w, g, lam, min_cells=min_cells)
    if name == "resid":
        return _resid_beta
    if name == "meta":
        return lambda Z, y, w, g, lam: _meta_beta(Z, y, w, g, lam, min_cells=max(min_cells, 4))
    raise ValueError(name)


def fit_within(ctx: Ctx, X_all: np.ndarray, *, target: str = "logmag", lam: float = 1.0,
               shrink: float | None = None, estimator: str = "fe", min_cells: int = 0,
               shrink_grid=SHRINK_GRID):
    """Fit the within estimate on the training fold.  Returns (beta, mu, sd, info).

    ``shrink=None`` chooses how much of a noisy within estimate to keep by leave-one-publication-out
    *inside the training fold*: each held-out training publication is predicted with its own effect
    set to zero, which is exactly the situation a design-BP test cell is in.  Nothing about the test
    fold enters.  The leave-one-out betas do not depend on the shrinkage, so they are fitted once
    and reused across the grid.
    """
    est = _estimator(estimator, min_cells)
    rtr = ctx.rich_train()
    w = ctx.rich_weights()
    g = pubs(ctx)[rtr]
    mag = np.abs(ctx.amp[rtr])
    y = np.log(mag + EPS) if target == "logmag" else mag
    mu, sd = _standardise(X_all[rtr], w)
    Z = (X_all[rtr] - mu) / sd
    beta, keep = est(Z, y, w, g, lam)

    if shrink is None:
        loo = []
        for u in np.unique(g):
            m = g == u
            if m.sum() < 2 or (~m).sum() < 20:
                continue
            bo, _ = est(Z[~m], y[~m], w[~m], g[~m], lam)
            xb = np.average(Z[~m], axis=0, weights=w[~m])
            anchor = _wmean(mag[~m], w[~m])
            loo.append((anchor, (Z[m] - xb) @ bo, mag[m], w[m]))
        best_s, best_loss = 0.0, np.inf
        for s in shrink_grid:
            err, wts = [], []
            for anchor, d, mtrue, ww in loo:
                pm = anchor * np.exp(s * d) if target == "logmag" else anchor + s * d
                err.append(np.abs(np.clip(pm, 0.0, None) - mtrue))
                wts.append(ww)
            if err:
                loss = float(np.average(np.concatenate(err), weights=np.concatenate(wts)))
                if loss < best_loss - 1e-12:
                    best_s, best_loss = s, loss
        shrink = best_s
    info = {"n_identified": int(keep.sum()), "shrink": float(shrink), "keep": keep,
            "beta_raw": beta.copy()}
    return beta * float(shrink), mu, sd, info


def _magnitude(ctx: Ctx, X_all, beta, mu, sd, target: str) -> np.ndarray:
    """Apply the estimate to the held-out cell's raw conditions, publication effect set to zero."""
    Zte = (X_all[ctx.test] - mu) / sd          # weighted training mean of Z is 0 by construction
    delta = Zte @ beta
    anchor = ctx.train_mean_magnitude()
    return np.clip(anchor * np.exp(delta) if target == "logmag" else anchor + delta, 0.0, None)


# ---------------------------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------------------------
def fe_arm(columns=None, *, target: str = "logmag", lam: float = 1.0,
           shrink: float | None = None, use64: bool = False, estimator: str = "fe",
           min_cells: int = 0, log: dict | None = None):
    """Gen14 with the magnitude modulated by a within-publication estimate of the condition effect."""
    def f(ctx: Ctx) -> np.ndarray:
        rtr = ctx.rich_train()
        X_all = _impute(cond64(ctx) if use64 else design_matrix(ctx, columns), rtr)
        beta, mu, sd, info = fit_within(ctx, X_all, target=target, lam=lam, shrink=shrink,
                                        estimator=estimator, min_cells=min_cells)
        if log is not None:
            log.setdefault("shrink", []).append(info["shrink"])
            log.setdefault("beta", []).append(beta.copy())
            log.setdefault("beta_raw", []).append(info["beta_raw"])
            log.setdefault("keep", []).append(info["keep"].copy())
        mag = _magnitude(ctx, X_all, beta, mu, sd, target)
        return _curve(sign(ctx) * mag, ctx.train_mean_curvature(), len(ctx.test))
    return f


def resid_arm(columns=None, **kw):
    """``fe_arm`` computed by the literal residual-on-residual two-step."""
    return fe_arm(columns, estimator="resid", **kw)


def meta_arm(columns=None, *, min_cells: int = 4, **kw):
    """``fe_arm`` with one within slope per publication, combined by the column-wise median."""
    return fe_arm(columns, estimator="meta", min_cells=min_cells, **kw)


def pooled_arm(columns=None, *, target: str = "logmag", lam: float = 1.0, use64: bool = False):
    """The same model WITHOUT the within-transformation -- publication identity left in."""
    def f(ctx: Ctx) -> np.ndarray:
        rtr = ctx.rich_train()
        X_all = _impute(cond64(ctx) if use64 else design_matrix(ctx, columns), rtr)
        w = ctx.rich_weights()
        mag_tr = np.abs(ctx.amp[rtr])
        y = np.log(mag_tr + EPS) if target == "logmag" else mag_tr
        mu, sd = _standardise(X_all[rtr], w)
        Z = (X_all[rtr] - mu) / sd
        A = np.c_[np.ones(len(Z)), Z]
        pen = lam * np.ones(A.shape[1]); pen[0] = 0.0
        sw = np.sqrt(w)
        Aw = A * sw[:, None]
        coef = np.linalg.solve(Aw.T @ Aw + np.diag(pen), Aw.T @ (y * sw))
        pred = np.c_[np.ones(len(ctx.test)), (X_all[ctx.test] - mu) / sd] @ coef
        mag = np.exp(pred) - EPS if target == "logmag" else pred
        return _curve(sign(ctx) * np.clip(mag, 0.0, None),
                      ctx.train_mean_curvature(), len(ctx.test))
    return f


def ri_arm(columns=None, *, lam_beta: float = 1.0, use64: bool = False):
    """Publication random intercept fitted directly as its ridge-on-dummies equivalent."""
    def f(ctx: Ctx) -> np.ndarray:
        rtr = ctx.rich_train()
        X_all = _impute(cond64(ctx) if use64 else design_matrix(ctx, columns), rtr)
        w = ctx.rich_weights()
        g = pubs(ctx)[rtr]
        y = np.log(np.abs(ctx.amp[rtr]) + EPS)
        mu, sd = _standardise(X_all[rtr], w)
        Z = (X_all[rtr] - mu) / sd
        u = np.unique(g)
        D = (g[:, None] == u[None, :]).astype(float)
        gm = _wmean(y, w)
        sw_num = sw_den = sb_num = 0.0
        for v in u:
            m = g == v
            mv = _wmean(y[m], w[m])
            sw_num += float(np.sum(w[m] * (y[m] - mv) ** 2))
            sw_den += float(w[m].sum()) * (1.0 - 1.0 / max(int(m.sum()), 1))
            sb_num += float(w[m].sum()) * (mv - gm) ** 2
        s2w = sw_num / max(sw_den, 1e-9)
        s2b = max(sb_num / max(float(w.sum()) - float(w.sum()) / len(u), 1e-9) - s2w / max(len(u), 1),
                  1e-4)
        lam_u = s2w / s2b
        A = np.c_[np.ones(len(Z)), Z, D]
        pen = np.concatenate([[0.0], np.full(Z.shape[1], lam_beta), np.full(D.shape[1], lam_u)])
        sq = np.sqrt(w)
        Aw = A * sq[:, None]
        coef = np.linalg.solve(Aw.T @ Aw + np.diag(pen), Aw.T @ (y * sq))
        beta = coef[1:1 + Z.shape[1]]
        delta = ((X_all[ctx.test] - mu) / sd) @ beta
        mag = np.clip(ctx.train_mean_magnitude() * np.exp(delta), 0.0, None)
        return _curve(sign(ctx) * mag, ctx.train_mean_curvature(), len(ctx.test))
    return f


def fe_curv(columns=None, *, lam: float = 1.0, shrink: float | None = None):
    """As ``fe_arm``, and the curvature b additionally gets a within-publication condition offset."""
    def f(ctx: Ctx) -> np.ndarray:
        rtr = ctx.rich_train()
        X_all = _impute(design_matrix(ctx, columns), rtr)
        beta, mu, sd, _ = fit_within(ctx, X_all, target="logmag", lam=lam, shrink=shrink)
        mag = _magnitude(ctx, X_all, beta, mu, sd, "logmag")
        w = ctx.rich_weights(); g = pubs(ctx)[rtr]
        Z = (X_all[rtr] - mu) / sd
        bb, _ = _within_beta(Z, ctx.cur[rtr], w, g, lam)
        db = ((X_all[ctx.test] - mu) / sd) @ bb
        return _curve(sign(ctx) * mag, ctx.train_mean_curvature() + db, len(ctx.test))
    return f


# ---------------------------------------------------------------------------------------------
# oracles that bound the route
# ---------------------------------------------------------------------------------------------
def _peer_mean(ctx: Ctx, ci: int, g: np.ndarray, mag: np.ndarray):
    peers = np.flatnonzero((g == g[ci]) & ctx.rich)
    peers = peers[peers != ci]
    return float(mag[peers].mean()) if len(peers) else None


def o_pubmag(ctx: Ctx) -> np.ndarray:
    """Magnitude = mean |a| of the test cell's own publication (its other cells only)."""
    g, mag = pubs(ctx), np.abs(ctx.amp)
    out = np.full(len(ctx.test), ctx.train_mean_magnitude())
    for j, ci in enumerate(ctx.test):
        pm = _peer_mean(ctx, ci, g, mag)
        if pm is not None:
            out[j] = pm
    return _curve(sign(ctx) * out, ctx.train_mean_curvature(), len(ctx.test))


def o_within(ctx: Ctx) -> np.ndarray:
    """THE CEILING OF THE WITHIN-TRANSFORMATION.

    The training mean magnitude plus the cell's own deviation from its publication's mean, i.e. a
    within model that recovered that deviation perfectly, with the publication effect still set to
    zero because a held-out laboratory is unknown.  If this does not beat gen14, then no
    within-publication condition model can, however good its features.
    """
    g, mag = pubs(ctx), np.abs(ctx.amp)
    anchor = ctx.train_mean_magnitude()
    out = np.full(len(ctx.test), anchor)
    for j, ci in enumerate(ctx.test):
        pm = _peer_mean(ctx, ci, g, mag)
        if pm is not None:
            out[j] = float(np.clip(anchor + (mag[ci] - pm), 0.0, None))
    return _curve(sign(ctx) * out, ctx.train_mean_curvature(), len(ctx.test))


def o_within_mul(ctx: Ctx) -> np.ndarray:
    """Multiplicative version of the ceiling: the training mean times |a| / publication mean |a|."""
    g, mag = pubs(ctx), np.abs(ctx.amp)
    anchor = ctx.train_mean_magnitude()
    out = np.full(len(ctx.test), anchor)
    for j, ci in enumerate(ctx.test):
        pm = _peer_mean(ctx, ci, g, mag)
        if pm is not None and pm > 1e-6:
            out[j] = float(np.clip(anchor * mag[ci] / pm, 0.0, 3.0))
    return _curve(sign(ctx) * out, ctx.train_mean_curvature(), len(ctx.test))


# ---------------------------------------------------------------------------------------------
# identity check: the within-transformation and the FWL two-step must give the same beta
# ---------------------------------------------------------------------------------------------
def check_fwl_identity(ctx: Ctx, columns, lam: float = 1.0) -> float:
    rtr = ctx.rich_train()
    X_all = _impute(design_matrix(ctx, columns), rtr)
    w, g = ctx.rich_weights(), pubs(ctx)[rtr]
    y = np.log(np.abs(ctx.amp[rtr]) + EPS)
    mu, sd = _standardise(X_all[rtr], w)
    Z = (X_all[rtr] - mu) / sd
    b1, _ = _within_beta(Z, y, w, g, lam)
    b2, _ = _resid_beta(Z, y, w, g, lam)
    return float(np.max(np.abs(b1 - b2)))
