"""Cluster-robust inference for a paired mean when the clusters are few and wildly unequal.

Why this module exists
----------------------
Every headline in gen13/gen14 is a *paired mean difference* over extractants, resampled in
chemotype blocks (``gen13sep.inference.paired_contrasts``, ``gen14.dirbench.Blocked``).  That is
the **pairs cluster bootstrap** with a **percentile** p-value: the null is never imposed and the
statistic is never studentised.  With G = 45 chemotypes of which one holds 23 of the 90
extractants (Kish effective G* = 11.7) this is the exact regime where cluster-robust inference is
known to under-state the standard error and over-reject.  See

* Cameron, Gelbach & Miller, "Bootstrap-Based Improvements for Inference with Clustered Errors",
  Review of Economics and Statistics 90(3):414-427, 2008, doi:10.1162/rest.90.3.414
* MacKinnon & Webb, "Wild Bootstrap Inference for Wildly Different Cluster Sizes",
  Journal of Applied Econometrics 32(2):233-254, 2017, doi:10.1002/jae.2508
* MacKinnon, Nielsen & Webb, "Cluster-robust inference: A guide to empirical practice",
  Journal of Econometrics 232(2):272-299, 2023, doi:10.1016/j.jeconom.2022.04.001 (arXiv:2205.03285)
* Imbens & Kolesar, "Robust Standard Errors in Small Samples: Some Practical Advice",
  Review of Economics and Statistics 98(4):701-712, 2016, doi:10.1162/REST_a_00552

The model here is the simplest possible regression -- ``d_i = mu + u_i`` with X = a column of
ones, ``d_i`` the per-extractant paired difference (reference minus candidate, positive favours
the candidate) and clusters = chemotypes -- so every quantity below has a closed form and no
linear algebra beyond a G x G Gram matrix is needed.

What it provides
----------------
``cluster_robust``   CR0/CR1/CR2/CR3 standard errors and the Bell-McCaffrey/Satterthwaite
                     degrees of freedom.  For balanced clusters the dof reduces exactly to G-1;
                     for this corpus it is far smaller, and that number is worth reporting.
``wild_cluster_t``   the WCR bootstrap-t of Cameron-Gelbach-Miller: impose H0, multiply each
                     *cluster's* residual vector by an i.i.d. Rademacher (or Webb 6-point) draw,
                     recompute the studentised statistic, and read the p off that distribution.
                     A confidence interval is obtained by inverting the test on a grid, reusing
                     the same weight matrix, which is what ``boottest`` does
                     (Roodman, MacKinnon, Nielsen & Webb, Stata Journal 19(1):4-60, 2019).
``bayes_bootstrap``  Rubin's Bayesian bootstrap (Annals of Statistics 9(1):130-134, 1981) over
                     *chemotypes*, giving a posterior for the macro difference and hence
                     P(worse | rope | better) against a region of practical equivalence.
``random_effects``   a random-effects (hierarchical) posterior over chemotype means by 2-D grid
                     quadrature -- the Corani/Benavoli hierarchical comparison
                     (Machine Learning 106(11):1817-1837, 2017) without a Stan dependency.
``detectable``       the minimum detectable effect at a stated power using the cluster-robust SE
                     and t critical values, replacing ``2.80 * bootstrap_se``.

Everything is numpy-only and runs in well under a second for N = 90, G = 45, B = 9999.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

WEIGHTS = ("rademacher", "webb", "mammen")
#: Webb's 6-point distribution, for G < 12 where Rademacher has too few distinct draws.
_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


# --------------------------------------------------------------------------------------
# cluster bookkeeping
# --------------------------------------------------------------------------------------
def _prepare(d: Sequence[float], cluster: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop non-finite units; return (d, cluster code 0..G-1, cluster sizes)."""
    d = np.asarray(d, dtype=float)
    c = np.asarray(cluster)
    ok = np.isfinite(d)
    d, c = d[ok], c[ok]
    names, code = np.unique(c, return_inverse=True)
    sizes = np.bincount(code, minlength=len(names)).astype(float)
    return d, code, sizes


def kish(sizes: Sequence[float]) -> float:
    """Kish's effective number of clusters, ``(sum n)^2 / sum n^2``.

    This is the cluster-size heterogeneity measure that MacKinnon & Webb (2017) use to explain
    why cluster-robust tests fail: it is 45 when the 45 chemotypes are balanced and 11.7 for the
    corpus as it stands.
    """
    n = np.asarray(sizes, dtype=float)
    return float(n.sum() ** 2 / (n ** 2).sum())


# --------------------------------------------------------------------------------------
# cluster-robust variance for the mean
# --------------------------------------------------------------------------------------
@dataclass
class ClusterFit:
    point: float
    n: int
    n_clusters: int
    kish_clusters: float
    max_cluster_share: float
    se_cr0: float
    se_cr1: float
    se_cr2: float
    se_cr3: float
    se_iid: float
    dof_bm: float
    t_cr2: float

    def as_dict(self) -> dict:
        return asdict(self)


def cluster_robust(d: Sequence[float], cluster: Sequence) -> ClusterFit:
    """Cluster-robust SEs and Bell-McCaffrey dof for the mean of ``d`` clustered by ``cluster``.

    With X = 1 the sandwich collapses to sums of *cluster residual sums* ``s_g``:

        CR0  = sum_g s_g^2 / N^2
        CR1  = CR0 * G/(G-1) * (N-1)/(N-1)                      (K = 1)
        CR2  = sum_g s_g^2 / (1 - n_g/N) / N^2                  (Bell-McCaffrey)
        CR3  = (G-1)/G * sum_g s_g^2 / (1 - n_g/N)^2 / N^2      (cluster jackknife)

    because ``X_g' M_gg^{-1/2} u_g = (1 - n_g/N)^{-1/2} s_g`` for the intercept-only design.
    The Satterthwaite dof for CR2 is ``(tr A)^2 / tr(A^2)`` with ``A = N^-2 sum_g c_g^2 a_g a_g'``,
    ``a_g = e_g - (n_g/N) 1``; ``tr A = 1/N`` exactly, and for balanced clusters the dof is G-1.
    """
    d, code, n_g = _prepare(d, cluster)
    N = float(len(d))
    G = len(n_g)
    if N < 3 or G < 2:
        raise ValueError(f"need >= 3 units and >= 2 clusters, got N={N:.0f} G={G}")
    mu = float(d.mean())
    u = d - mu
    s = np.bincount(code, weights=u, minlength=G)          # cluster residual sums
    h = n_g / N                                            # leverage of cluster g
    cr0 = float((s ** 2).sum()) / N ** 2
    cr1 = cr0 * (G / (G - 1.0)) * ((N - 1.0) / (N - 1.0))
    cr2 = float((s ** 2 / (1.0 - h)).sum()) / N ** 2
    cr3 = (G - 1.0) / G * float((s ** 2 / (1.0 - h) ** 2).sum()) / N ** 2
    # Bell-McCaffrey / Satterthwaite dof: eigen-free closed form via the cluster Gram matrix.
    #   a_g'a_h = delta_gh n_g - n_g n_h / N ,  c_g^2 = 1/(1 - n_g/N)
    c2 = 1.0 / (1.0 - h)
    gram = np.diag(n_g) - np.outer(n_g, n_g) / N           # a_g' a_h
    W = (c2[:, None] * c2[None, :]) * gram ** 2            # c_g^2 c_h^2 (a_g'a_h)^2
    tr_A = 1.0 / N                                         # exact
    tr_A2 = float(W.sum()) / N ** 4
    dof = tr_A ** 2 / tr_A2 if tr_A2 > 0 else np.nan
    se2 = float(np.sqrt(cr2))
    return ClusterFit(
        point=mu, n=int(N), n_clusters=G, kish_clusters=kish(n_g),
        max_cluster_share=float(n_g.max() / N),
        se_cr0=float(np.sqrt(cr0)), se_cr1=float(np.sqrt(cr1)),
        se_cr2=se2, se_cr3=float(np.sqrt(cr3)),
        se_iid=float(d.std(ddof=1) / np.sqrt(N)),
        dof_bm=float(dof), t_cr2=float(mu / se2) if se2 > 0 else np.nan,
    )


def _t_stat(d: np.ndarray, code: np.ndarray, n_g: np.ndarray, mu0: float, se_kind: str) -> float:
    N = float(len(d))
    G = len(n_g)
    mu = float(d.mean())
    u = d - mu
    s = np.bincount(code, weights=u, minlength=G)
    h = n_g / N
    if se_kind == "CR1":
        v = (s ** 2).sum() * (G / (G - 1.0)) / N ** 2
    elif se_kind == "CR2":
        v = (s ** 2 / (1.0 - h)).sum() / N ** 2
    elif se_kind == "CR3":
        v = (G - 1.0) / G * (s ** 2 / (1.0 - h) ** 2).sum() / N ** 2
    else:
        raise ValueError(se_kind)
    return (mu - mu0) / np.sqrt(v) if v > 0 else np.nan


# --------------------------------------------------------------------------------------
# wild cluster bootstrap-t
# --------------------------------------------------------------------------------------
@dataclass
class WildResult:
    point: float
    t_obs: float
    p_wcr: float
    p_wcu: float
    ci_low: float
    ci_high: float
    se_kind: str
    weights: str
    reps: int
    n_clusters: int
    kish_clusters: float
    dof_bm: float
    p_t_dof: float
    p_normal: float

    def as_dict(self) -> dict:
        return asdict(self)


def _draw_weights(rng: np.random.Generator, reps: int, G: int, kind: str) -> np.ndarray:
    if kind == "rademacher":
        return rng.integers(0, 2, size=(reps, G)).astype(float) * 2.0 - 1.0
    if kind == "webb":
        return _WEBB[rng.integers(0, 6, size=(reps, G))]
    if kind == "mammen":
        p = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))
        a, b = -(np.sqrt(5.0) - 1.0) / 2.0, (np.sqrt(5.0) + 1.0) / 2.0
        return np.where(rng.random((reps, G)) < p, a, b)
    raise ValueError(f"weights must be one of {WEIGHTS}")


def wild_cluster_t(d: Sequence[float], cluster: Sequence, *, reps: int = 9999,
                   seed: int = 8675309, weights: str = "rademacher", se_kind: str = "CR2",
                   ci: bool = True, grid: int = 161) -> WildResult:
    """WCR (null-imposed) wild cluster bootstrap-t for H0: mean(d) = 0.

    For the intercept-only model the restricted residuals under ``mu = mu0`` are ``d_i - mu0``,
    so a bootstrap sample is ``d*_i = mu0 + v_{g(i)} (d_i - mu0)`` and every bootstrap quantity
    depends on the data only through the G cluster sums of ``d - mu0``.  That makes the whole
    grid inversion a few matrix products: the same (reps x G) weight matrix is reused for every
    ``mu0``, exactly as ``boottest`` does.

    ``p_wcu`` is the same bootstrap *without* imposing the null (residuals centred at the
    estimate).  CGM 2008 recommend the restricted version; the pair is reported because a large
    gap between them is itself a warning that the design is fragile.
    ``p_t_dof`` and ``p_normal`` are the analytic CR2 t-tests against t_(Bell-McCaffrey dof) and
    against the normal, for reference.
    """
    d, code, n_g = _prepare(d, cluster)
    N, G = float(len(d)), len(n_g)
    rng = np.random.default_rng(seed)
    V = _draw_weights(rng, reps, G, weights)               # (reps, G)
    h = n_g / N
    if se_kind == "CR1":
        scale, corr = np.ones(G), G / (G - 1.0)
    elif se_kind == "CR2":
        scale, corr = 1.0 / np.sqrt(1.0 - h), 1.0
    elif se_kind == "CR3":
        scale, corr = 1.0 / (1.0 - h), (G - 1.0) / G
    else:
        raise ValueError(se_kind)

    def boot_t(mu0: float) -> np.ndarray:
        """|t*| for every bootstrap replicate, with the null mu = mu0 imposed."""
        e = d - mu0                                        # restricted residuals
        S = np.bincount(code, weights=e, minlength=G)      # cluster sums of restricted residuals
        # d*_i - mu0 = v_g e_i ; mean of that is (1/N) sum_g v_g S_g
        num = V @ S / N                                    # (reps,)
        # residuals of the bootstrap sample about its own mean: v_g e_i - num
        # cluster sum: v_g S_g - n_g * num
        Sb = V * S[None, :] - num[:, None] * n_g[None, :]  # (reps, G)
        v = corr * ((Sb * scale[None, :]) ** 2).sum(axis=1) / N ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(num) / np.sqrt(v)

    fit = cluster_robust(d, cluster)
    t_obs = abs(_t_stat(d, code, n_g, 0.0, se_kind))
    tb = boot_t(0.0)
    ok = np.isfinite(tb)
    p_wcr = float((1.0 + np.sum(tb[ok] >= t_obs)) / (ok.sum() + 1.0))
    # unrestricted variant: centre the residuals at mu_hat, keep the same weights
    e_u = d - fit.point
    S_u = np.bincount(code, weights=e_u, minlength=G)
    num_u = V @ S_u / N
    Sb_u = V * S_u[None, :] - num_u[:, None] * n_g[None, :]
    v_u = corr * ((Sb_u * scale[None, :]) ** 2).sum(axis=1) / N ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        tb_u = np.abs(num_u) / np.sqrt(v_u)
    ok_u = np.isfinite(tb_u)
    p_wcu = float((1.0 + np.sum(tb_u[ok_u] >= t_obs)) / (ok_u.sum() + 1.0))

    lo = hi = np.nan
    if ci:
        span = max(6.0 * fit.se_cr2, 1e-9)
        mus = np.linspace(fit.point - span, fit.point + span, grid)
        keep = []
        for m0 in mus:
            t0 = abs(_t_stat(d, code, n_g, m0, se_kind))
            tbm = boot_t(m0)
            okm = np.isfinite(tbm)
            p = (1.0 + np.sum(tbm[okm] >= t0)) / (okm.sum() + 1.0)
            keep.append(p >= 0.05)
        keep = np.asarray(keep)
        if keep.any():
            lo, hi = float(mus[keep][0]), float(mus[keep][-1])

    from scipy import stats
    p_t = float(2 * stats.t.sf(abs(fit.t_cr2), max(fit.dof_bm, 1.0)))
    p_n = float(2 * stats.norm.sf(abs(fit.t_cr2)))
    return WildResult(point=fit.point, t_obs=float(t_obs), p_wcr=p_wcr, p_wcu=p_wcu,
                      ci_low=lo, ci_high=hi, se_kind=se_kind, weights=weights, reps=reps,
                      n_clusters=G, kish_clusters=fit.kish_clusters, dof_bm=fit.dof_bm,
                      p_t_dof=p_t, p_normal=p_n)


# --------------------------------------------------------------------------------------
# region of practical equivalence
# --------------------------------------------------------------------------------------
@dataclass
class RopeResult:
    estimand: str
    rope: float
    p_worse: float
    p_rope: float
    p_better: float
    post_mean: float
    post_low: float
    post_high: float
    method: str

    def as_dict(self) -> dict:
        return asdict(self)


def bayes_bootstrap(d: Sequence[float], cluster: Sequence, *, rope: float = 0.02,
                    reps: int = 20000, seed: int = 8675309,
                    estimand: str = "unit_macro") -> RopeResult:
    """Rubin's Bayesian bootstrap over *chemotypes*, then the ROPE probabilities.

    ``estimand='unit_macro'``    weights a chemotype by how many extractants it holds, which is
                                 the estimand the current pairs bootstrap targets.
    ``estimand='cluster_macro'`` gives every chemotype one vote -- the estimand that matches the
                                 Kish effective count of 11.7 and the one to report when a single
                                 chemotype holds a quarter of the units.

    Unlike a p-value this returns three probabilities that sum to one, so a tie is reportable as
    a positive statement (Benavoli, Corani, Demsar & Zaffalon, JMLR 18(77):1-36, 2017).
    """
    d, code, n_g = _prepare(d, cluster)
    G = len(n_g)
    mean_g = np.bincount(code, weights=d, minlength=G) / n_g
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(G), size=reps)               # (reps, G)
    if estimand == "cluster_macro":
        theta = W @ mean_g
    elif estimand == "unit_macro":
        Wn = W * n_g[None, :]
        theta = (Wn @ mean_g) / Wn.sum(axis=1)
    else:
        raise ValueError(estimand)
    return RopeResult(estimand=estimand, rope=rope,
                      p_worse=float((theta < -rope).mean()),
                      p_rope=float((np.abs(theta) <= rope).mean()),
                      p_better=float((theta > rope).mean()),
                      post_mean=float(theta.mean()),
                      post_low=float(np.quantile(theta, 0.025)),
                      post_high=float(np.quantile(theta, 0.975)),
                      method="bayesian bootstrap over chemotypes (Rubin 1981)")


def random_effects(d: Sequence[float], cluster: Sequence, *, rope: float = 0.02,
                   n_mu: int = 601, n_tau: int = 241) -> RopeResult:
    """Random-effects posterior for the grand mean over chemotypes, by 2-D grid quadrature.

    Each chemotype contributes its mean ``y_g`` with sampling variance ``sigma_g^2``; a chemotype
    with one extractant borrows the pooled within-chemotype variance.  Priors: flat on ``mu``,
    uniform on ``tau`` (between-chemotype sd) over the grid -- the standard weakly-informative
    choice for a meta-analysis with few groups.  This is the analytic twin of the Bayesian
    hierarchical classifier comparison of Corani, Benavoli, Demsar, Mangili & Zaffalon
    (Machine Learning 106(11):1817-1837, 2017), with chemotypes in place of datasets, and needs
    no MCMC.
    """
    d, code, n_g = _prepare(d, cluster)
    G = len(n_g)
    y = np.bincount(code, weights=d, minlength=G) / n_g
    ss = np.bincount(code, weights=(d - y[code]) ** 2, minlength=G)
    dfw = max(float((n_g - 1).sum()), 1.0)
    pooled = float(ss.sum() / dfw) if dfw > 0 else float(np.var(d, ddof=1))
    within = np.where(n_g > 1, ss / np.maximum(n_g - 1.0, 1.0), pooled)
    within = np.maximum(within, 1e-12)
    s2 = within / n_g                                       # sampling variance of each y_g

    spread = float(np.std(y, ddof=1)) if G > 1 else 1.0
    mus = np.linspace(y.mean() - 8 * spread / np.sqrt(G) - 4 * spread,
                      y.mean() + 8 * spread / np.sqrt(G) + 4 * spread, n_mu)
    taus = np.linspace(0.0, max(4.0 * spread, 1e-6), n_tau)
    V = s2[None, :] + (taus ** 2)[:, None]                  # (n_tau, G)
    logl = np.empty((n_tau, n_mu))
    for i in range(n_tau):
        r = y[None, :] - mus[:, None]
        logl[i] = -0.5 * (np.log(2 * np.pi * V[i])[None, :] + r ** 2 / V[i][None, :]).sum(axis=1)
    logl -= logl.max()
    post = np.exp(logl)
    post /= post.sum()
    pmu = post.sum(axis=0)
    pmu /= pmu.sum()
    cdf = np.cumsum(pmu)
    return RopeResult(estimand="cluster_macro", rope=rope,
                      p_worse=float(pmu[mus < -rope].sum()),
                      p_rope=float(pmu[np.abs(mus) <= rope].sum()),
                      p_better=float(pmu[mus > rope].sum()),
                      post_mean=float((pmu * mus).sum()),
                      post_low=float(np.interp(0.025, cdf, mus)),
                      post_high=float(np.interp(0.975, cdf, mus)),
                      method="random-effects grid posterior over chemotype means")


# --------------------------------------------------------------------------------------
# detectability
# --------------------------------------------------------------------------------------
def detectable(se: float, dof: float, *, power: float = 0.80, alpha: float = 0.05) -> float:
    """Minimum detectable effect: ``(t_{1-a/2,dof} + t_{power,dof}) * se``.

    The repository currently reports ``2.80 * se`` -- the *normal* constant -- on top of a
    standard error that is itself too small.  Both corrections push in the same direction.
    """
    from scipy import stats
    return float((stats.t.ppf(1 - alpha / 2, dof) + stats.t.ppf(power, dof)) * se)


def detectable_r2(n_effective: float, *, power: float = 0.80, alpha: float = 0.05,
                  n_predictors: int = 1) -> float:
    """Smallest between-cluster R^2 a covariate block could have and still be found.

    Solves for R^2 in an F-test with ``df1 = n_predictors``, ``df2 = n_effective - n_predictors - 1``
    and non-centrality ``lambda = n_effective * R^2 / (1 - R^2)`` at the stated power.  Report it
    with both the nominal cluster count (45) and the Kish effective count (11.7): the truth is
    bracketed by the two, and the pair is the honest form of the statement.
    """
    from scipy import stats
    df1 = float(n_predictors)
    df2 = float(n_effective) - df1 - 1.0
    if df2 <= 1:
        return float("nan")
    crit = stats.f.ppf(1 - alpha, df1, df2)
    lo, hi = 1e-6, 0.999999
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lam = n_effective * mid / (1.0 - mid)
        if stats.ncf.sf(crit, df1, df2, lam) < power:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))
