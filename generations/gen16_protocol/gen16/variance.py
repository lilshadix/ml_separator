"""Variance components for the cell amplitude, with intervals, and an order-free R2 split.

Two things the programme currently reports without uncertainty:

* ``d6_step3_nested_anova.csv`` -- a *sequential* sum-of-squares decomposition whose publication
  share is 0.08 % if conditions are entered first and 5.4 % if provenance is entered first.  That
  order dependence is not a nuisance, it *is* the condition/publication confounding, and the
  honest way to report it is the Shapley value of the R2 game (Gromping's LMG), which averages the
  sequential contribution over all orderings and is the unique allocation satisfying efficiency,
  symmetry and the null-player axiom.  ``lmg_shapley`` computes it exactly by enumerating subsets.

* the amplitude ICC (0.72 at chemotype level) -- a point estimate with no interval.  ``reml_fit``
  fits a crossed Gaussian variance-components model by REML on the marginal covariance
  ``V = sigma_e^2 I + sum_k sigma_k^2 Z_k Z_k'`` (n = 289, so V is 289x289 and a Cholesky per
  likelihood evaluation is free), and ``profile_ci`` / ``parametric_bootstrap_ci`` give it two
  independent intervals.  Because a variance is bounded below by zero, the profile-likelihood
  critical value for a component's lower limit is the 50:50 chi2_0/chi2_1 mixture value 2.706, not
  3.841 (Self & Liang 1987; Stram & Lee 1994); both are reported.

``nakagawa_r2`` returns the marginal R2 (fixed effects only -- the part that transfers to a new
extractant in a new laboratory) and the conditional R2 (fixed + random -- what a design that lets
the extractant appear on both sides of the split is really measuring).  Feeding it the *deployed
model's own out-of-fold prediction* as the single fixed effect turns it into the honest headline
for "this model predicts lanthanide separation": marginal R2 is the fraction of amplitude variance
the transferable model actually carries.

numpy + scipy only.  References
-------------------------------
Nakagawa & Schielzeth, "A general and simple method for obtaining R2 from generalized linear
    mixed-effects models", Methods Ecol. Evol. 4(2):133-142, 2013, doi:10.1111/j.2041-210x.2012.00261.x
Nakagawa, Johnson & Schielzeth, "The coefficient of determination R2 and the intra-class
    correlation coefficient from generalized linear mixed-effects models revisited and expanded",
    J. R. Soc. Interface 14(134):20170213, 2017, doi:10.1098/rsif.2017.0213
Stoffel, Nakagawa & Schielzeth, "rptR: repeatability estimation and variance decomposition by
    generalized linear mixed-effects models", Methods Ecol. Evol. 8(11):1639-1644, 2017,
    doi:10.1111/2041-210X.12797   (parametric-bootstrap CIs for variance components)
Gromping, "Estimators of Relative Importance in Linear Regression Based on Variance
    Decomposition", The American Statistician 61(2):139-147, 2007, doi:10.1198/000313007X188252
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize

LOG2PI = float(np.log(2 * np.pi))
FLOOR = -18.0          # log-variance floor: exp(-18) ~ 1.5e-8, effectively zero


# --------------------------------------------------------------------------------------
# design matrices
# --------------------------------------------------------------------------------------
def indicator(labels) -> np.ndarray:
    """n x L 0/1 matrix of a categorical factor, columns in sorted label order."""
    lab = np.asarray(labels).astype(str)
    names = np.unique(lab)
    Z = np.zeros((len(lab), len(names)))
    Z[np.arange(len(lab)), np.searchsorted(names, lab)] = 1.0
    return Z


# --------------------------------------------------------------------------------------
# REML for a crossed variance-components model
# --------------------------------------------------------------------------------------
@dataclass
class REMLFit:
    names: tuple[str, ...]
    var: np.ndarray            # sigma_k^2 in the order of ``names``, residual last
    beta: np.ndarray
    loglik: float
    n: int
    p: int
    converged: bool

    @property
    def total(self) -> float:
        return float(self.var.sum())

    def share(self) -> dict[str, float]:
        return {n: float(v / self.total) for n, v in zip(self.names, self.var)}


def _V(theta: np.ndarray, ZZ: list[np.ndarray], n: int) -> np.ndarray:
    s = np.exp(theta)
    V = np.eye(n) * s[-1]
    for k, M in enumerate(ZZ):
        V += s[k] * M
    return V


def _neg_reml(theta: np.ndarray, y: np.ndarray, X: np.ndarray, ZZ: list[np.ndarray]) -> float:
    n, p = X.shape
    V = _V(theta, ZZ, n)
    try:
        L = cholesky(V, lower=True)
    except np.linalg.LinAlgError:
        return 1e12
    logdetV = 2.0 * np.log(np.diag(L)).sum()
    Vi_X = cho_solve((L, True), X)
    XtViX = X.T @ Vi_X
    try:
        Lx = cholesky(XtViX, lower=True)
    except np.linalg.LinAlgError:
        return 1e12
    logdetXtViX = 2.0 * np.log(np.diag(Lx)).sum()
    Vi_y = cho_solve((L, True), y)
    beta = cho_solve((Lx, True), X.T @ Vi_y)
    r = y - X @ beta
    q = float(r @ cho_solve((L, True), r))
    return 0.5 * (logdetV + logdetXtViX + q + (n - p) * LOG2PI)


def reml_fit(y: np.ndarray, X: np.ndarray, Z: dict[str, np.ndarray], *,
             start: np.ndarray | None = None, fixed: dict[str, float] | None = None) -> REMLFit:
    """REML fit of ``y = X beta + sum_k Z_k u_k + e``, all ``u_k`` and ``e`` spherical Gaussian.

    ``fixed`` pins named components at a given variance (used by ``profile_ci``).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(y):
        X = X.T
    n = len(y)
    names = tuple(Z) + ("residual",)
    ZZ = [np.asarray(Z[k], dtype=float) @ np.asarray(Z[k], dtype=float).T for k in Z]
    fixed = fixed or {}
    free = [i for i, nm in enumerate(names) if nm not in fixed]
    v0 = float(np.var(y)) / max(len(names), 1)
    th_full = np.full(len(names), np.log(max(v0, 1e-6)))
    for nm, val in fixed.items():
        th_full[names.index(nm)] = np.log(max(val, np.exp(FLOOR)))
    if start is not None:
        th_full[free] = np.asarray(start)[free]

    def obj(t):
        th = th_full.copy()
        th[free] = np.clip(t, FLOOR, 6.0)
        return _neg_reml(th, y, X, ZZ)

    # L-BFGS-B on the log-variances first (the REML objective is smooth in theta, so a
    # quasi-Newton pass costs ~300 evaluations where Nelder-Mead costs ~5000), then a short
    # simplex polish because the objective is flat near a boundary component.
    res = minimize(obj, th_full[free], method="L-BFGS-B",
                   bounds=[(FLOOR, 6.0)] * len(free),
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    res = minimize(obj, res.x, method="Nelder-Mead",
                   options={"maxiter": 800, "xatol": 1e-7, "fatol": 1e-9})
    th = th_full.copy()
    th[free] = np.clip(res.x, FLOOR, 6.0)
    V = _V(th, ZZ, n)
    L = cho_factor(V, lower=True)
    XtViX = X.T @ cho_solve(L, X)
    beta = np.linalg.solve(XtViX, X.T @ cho_solve(L, y))
    return REMLFit(names=names, var=np.exp(th), beta=beta, loglik=float(-res.fun),
                   n=n, p=X.shape[1], converged=bool(res.success))


def profile_ci(y, X, Z: dict[str, np.ndarray], fit: REMLFit, component: str, *,
               n_grid: int = 24, boundary_mixture: bool = True) -> dict:
    """Profile-likelihood interval for one variance component.

    Upper limit uses the chi2_1 critical value 3.841; the lower limit uses 2.706, the 50:50
    chi2_0/chi2_1 mixture that is correct when the null puts the variance on the boundary
    (Self & Liang 1987; Stram & Lee 1994).  Both are reported so the difference is visible.
    """
    k = fit.names.index(component)
    hat = float(fit.var[k])
    crit_hi, crit_lo = 3.841, (2.706 if boundary_mixture else 3.841)
    # The upper grid must reach the *total* variance, not a fixed factor above the estimate:
    # a component that REML puts on the boundary has hat ~ 1e-8, and a span of exp(3) above it
    # is still ~1e-7, so the walk never crosses the critical value and the returned upper limit
    # is the estimate itself -- a spurious zero-width interval on exactly the components whose
    # interval matters most (here: extractant identity, hat = 0, true upper limit 0.023).
    total = float(fit.var.sum())
    lo_start = max(hat, 1e-9)
    hi_top = max(total, 4.0 * lo_start)
    grid_lo = np.exp(np.linspace(np.log(lo_start), FLOOR, n_grid))
    grid_hi = np.exp(np.linspace(np.log(max(hat, total * 1e-4)), np.log(hi_top), n_grid))

    def prof(v):
        f = reml_fit(y, X, Z, start=np.log(fit.var), fixed={component: float(v)})
        return 2.0 * (fit.loglik - f.loglik)

    def walk(grid, crit):
        prev_v, prev_d = grid[0], 0.0
        for v in grid[1:]:
            d = prof(v)
            if d >= crit:
                # linear interpolation in log v
                t = (crit - prev_d) / max(d - prev_d, 1e-12)
                return float(np.exp(np.log(prev_v) + t * (np.log(v) - np.log(prev_v))))
            prev_v, prev_d = v, d
        return float(grid[-1])

    lo = walk(grid_lo, crit_lo)
    hi = walk(grid_hi, crit_hi)
    return {"component": component, "estimate": hat, "profile_low": lo, "profile_high": hi,
            "crit_low": crit_lo, "crit_high": crit_hi,
            "at_boundary": bool(lo <= np.exp(FLOOR) * 10)}


def parametric_bootstrap_ci(y, X, Z: dict[str, np.ndarray], fit: REMLFit, *,
                            reps: int = 400, seed: int = 8675309) -> dict:
    """Percentile intervals for every component and for each component's *share* of the total.

    Data are regenerated from the fitted model and refitted (rptR's procedure).  Shares are the
    quantity the report actually quotes, and their intervals are much wider than a delta-method
    interval on the variance would suggest.
    """
    rng = np.random.default_rng(seed)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(y):
        X = X.T
    mu = X @ fit.beta
    Zs = {k: np.asarray(v, dtype=float) for k, v in Z.items()}
    draws, shares = [], []
    for _ in range(reps):
        ys = mu.copy()
        for j, (k, Zk) in enumerate(Zs.items()):
            ys = ys + Zk @ rng.normal(0.0, np.sqrt(fit.var[j]), size=Zk.shape[1])
        ys = ys + rng.normal(0.0, np.sqrt(fit.var[-1]), size=len(ys))
        f = reml_fit(ys, X, Zs, start=np.log(fit.var))
        draws.append(f.var)
        shares.append(f.var / f.var.sum())
    D = np.array(draws)
    S = np.array(shares)
    return {"names": fit.names, "reps": reps,
            "var_low": np.quantile(D, 0.025, axis=0), "var_high": np.quantile(D, 0.975, axis=0),
            "share_hat": fit.var / fit.total,
            "share_low": np.quantile(S, 0.025, axis=0), "share_high": np.quantile(S, 0.975, axis=0)}


# --------------------------------------------------------------------------------------
# Nakagawa-Schielzeth R2
# --------------------------------------------------------------------------------------
def nakagawa_r2(y, X, Z: dict[str, np.ndarray], fit: REMLFit, *,
                conditional_on: tuple[str, ...] | None = None) -> dict:
    """Marginal and conditional R2.

    ``sigma_f^2`` is the variance of the fitted fixed part over the sample.  ``conditional_on``
    names the random effects a *leaky* design would let the model borrow (default: all of them);
    the conditional R2 is what such a design reports, the marginal R2 is what transfers.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(y):
        X = X.T
    sig_f = float(np.var(X @ fit.beta))
    rand = dict(zip(fit.names[:-1], fit.var[:-1]))
    sig_e = float(fit.var[-1])
    tot = sig_f + sum(rand.values()) + sig_e
    cond = conditional_on if conditional_on is not None else tuple(rand)
    return {"sigma2_fixed": sig_f, "sigma2_residual": sig_e,
            "sigma2_random": {k: float(v) for k, v in rand.items()},
            "R2_marginal": sig_f / tot,
            "R2_conditional": (sig_f + sum(rand[k] for k in cond if k in rand)) / tot,
            "conditional_on": cond}


# --------------------------------------------------------------------------------------
# LMG / Shapley decomposition of R2 (order-free replacement for the sequential ANOVA)
# --------------------------------------------------------------------------------------
def _r2(y: np.ndarray, cols: np.ndarray | None) -> float:
    n = len(y)
    yc = y - y.mean()
    sst = float(yc @ yc)
    if cols is None or cols.size == 0:
        return 0.0
    A = np.c_[np.ones(n), cols]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return float(1.0 - (r @ r) / max(sst, 1e-12))


def lmg_shapley(y: np.ndarray, blocks: dict[str, np.ndarray], *,
                max_exact: int = 12) -> dict:
    """Exact LMG (= Shapley value of the R2 game) for a set of *blocks* of columns.

    Each key of ``blocks`` is one term of the current nested ANOVA (extractant identity, diluent,
    acid identity, an acid-concentration slope, publication, ...) and its value the columns that
    encode it.  Returns each term's Shapley share of the model R2; unlike the sequential SS the
    shares do not depend on entry order and sum exactly to R2 of the full model.

    Cost is ``2^K`` regressions (K = number of blocks).  With K = 8 that is 256 least-squares fits
    on n ~ 150 rows -- under a second.  Refuses to run past ``max_exact`` blocks.
    """
    keys = list(blocks)
    K = len(keys)
    if K > max_exact:
        raise ValueError(f"{K} blocks -> 2^{K} subsets; group them first")
    y = np.asarray(y, dtype=float).ravel()
    cache: dict[frozenset, float] = {}

    def val(S: frozenset) -> float:
        if S not in cache:
            cols = np.concatenate([blocks[k] for k in keys if k in S], axis=1) if S else None
            cache[S] = _r2(y, cols)
        return cache[S]

    from math import factorial
    phi = {k: 0.0 for k in keys}
    for k in keys:
        others = [o for o in keys if o != k]
        for size in range(K):
            w = factorial(size) * factorial(K - size - 1) / factorial(K)
            for S in combinations(others, size):
                fs = frozenset(S)
                phi[k] += w * (val(fs | {k}) - val(fs))
    full = val(frozenset(keys))
    return {"r2_full": full, "shapley": phi,
            "share_of_r2": {k: (v / full if full > 0 else np.nan) for k, v in phi.items()},
            "check_sums_to_r2": float(sum(phi.values()))}


def lmg_bootstrap(y: np.ndarray, blocks: dict[str, np.ndarray], groups: np.ndarray, *,
                  reps: int = 400, seed: int = 8675309) -> dict:
    """Cluster (chemotype) bootstrap intervals for the Shapley shares."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups).astype(str)
    names = np.unique(g)
    members = [np.flatnonzero(g == c) for c in names]
    keys = list(blocks)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(names), size=len(names))
        take = np.concatenate([members[j] for j in pick])
        try:
            r = lmg_shapley(y[take], {k: v[take] for k, v in blocks.items()})
        except Exception:
            continue
        out.append([r["shapley"][k] for k in keys])
    A = np.array(out)
    return {"keys": keys, "low": np.quantile(A, 0.025, axis=0),
            "high": np.quantile(A, 0.975, axis=0), "reps": len(A)}
