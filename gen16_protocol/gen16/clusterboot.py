"""Few-cluster inference for the paired arm contrasts.

Every published p-value in gen13/gen14 comes from ``gen13sep.inference.paired_contrasts``: a
*percentile* bootstrap that resamples the 40 chemotypes with replacement and takes the 2.5/97.5
quantiles of the resampled mean of a per-extractant delta.  That estimator is a cluster bootstrap
with G = 40 clusters holding N = 82 units, of which one cluster (``sc009``) holds 23 units and 375
of the corpus's 521 cells.  MacKinnon & Webb (JAE 2017) show the "rule of 42" fails exactly in that
regime -- with cluster sizes as unequal as US state populations, 50 clusters is not enough and
cluster-robust t-tests over-reject; MacKinnon, Nielsen & Webb (J. Econometrics 2023) recommend the
*restricted wild cluster bootstrap-t* (WCR) as the default and the cluster jackknife (CV3) variance
as the non-bootstrap fallback.

The programme's statistic, extractant-macro MAE, is not a regression coefficient, so the clean way
to bring that machinery to bear is the one the lead names: write each arm comparison as the
intercept-only regression

    d_i = mu + u_i,      d_i = MAE_reference(i) - MAE_candidate(i)   for extractant i,

clustered by the extractant's chemotype, and test H0: mu = 0.  For this design the restricted wild
cluster bootstrap with Rademacher weights is *exactly* a chemotype-level sign-flip randomisation
test (the restricted residuals are the data themselves, because mu is restricted to 0), so the
resulting p-value is exact under the sharp null of sign-symmetry within a chemotype rather than an
asymptotic approximation.  That is a strictly stronger guarantee than the percentile interval it
replaces, at the same cost.

Everything here is numpy/scipy only.  Nothing writes into a locked run directory.

References
----------
Cameron & Miller, "A Practitioner's Guide to Cluster-Robust Inference", J. Human Resources
    50(2):317-372, 2015, doi:10.3368/jhr.50.2.317
Cameron, Gelbach & Miller, "Bootstrap-Based Improvements for Inference with Clustered Errors",
    Rev. Econ. Stat. 90(3):414-427, 2008, doi:10.1162/rest.90.3.414
MacKinnon & Webb, "Wild bootstrap inference for wildly different cluster sizes", J. Applied
    Econometrics 32(2):233-254, 2017, doi:10.1002/jae.2508
MacKinnon, Nielsen & Webb, "Cluster-robust inference: A guide to empirical practice",
    J. Econometrics 232(2):272-299, 2023, doi:10.1016/j.jeconom.2022.04.001
MacKinnon, Nielsen & Webb, "Fast and reliable jackknife and bootstrap methods for cluster-robust
    inference", J. Applied Econometrics 38(5):671-694, 2023, doi:10.1002/jae.2969
Webb, "Reworking wild bootstrap-based inference for clustered errors", Canadian J. Economics
    56(3):839-858, 2023, doi:10.1111/caje.12661
Carter, Schnepel & Steigerwald, "Asymptotic Behavior of a t-Test Robust to Cluster Heterogeneity",
    Rev. Econ. Stat. 99(4):698-709, 2017, doi:10.1162/rest_a_00639
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

WEBB6 = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


# --------------------------------------------------------------------------------------
# cluster bookkeeping
# --------------------------------------------------------------------------------------
def _blocks(cluster: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    names = np.unique(np.asarray(cluster, dtype=object).astype(str))
    idx = [np.flatnonzero(np.asarray(cluster).astype(str) == c) for c in names]
    return idx, names


def kish_neff(sizes) -> float:
    """Kish effective number of clusters, ``(sum n)^2 / sum n^2``.  Counts units, not leverage."""
    n = np.asarray(list(sizes), dtype=float)
    return float(n.sum() ** 2 / (n ** 2).sum())


def css_effective_clusters(cluster: np.ndarray, rho: float | None = None,
                           values: np.ndarray | None = None) -> dict:
    """Carter-Schnepel-Steigerwald effective number of clusters for a cluster mean.

    CSS measure heterogeneity by ``Gamma = (1/G) sum_g (gamma_g/gammabar - 1)^2`` where
    ``gamma_g`` is cluster ``g``'s contribution to the asymptotic variance of the estimator,
    and report ``G* = G / (1 + Gamma)``.  For an intercept-only regression with within-cluster
    equicorrelated errors the contribution is ``gamma_g = n_g + n_g(n_g - 1) * rho``.  ``rho`` is
    estimated from the data when ``values`` is given (a one-way ANOVA moment estimator), else it
    must be supplied.

    The closed form for ``gamma_g`` is this module's specialisation of the CSS construction to the
    cluster mean; the general definition in the paper is in terms of the cluster's score.  Treat
    ``G_star`` as an order-of-magnitude diagnostic, not a quantity to quote to three digits.
    """
    idx, names = _blocks(cluster)
    n = np.array([len(i) for i in idx], dtype=float)
    G = len(n)
    if rho is None:
        if values is None:
            raise ValueError("give rho or values")
        v = np.asarray(values, dtype=float)
        gm = v.mean()
        between = sum(len(i) * (v[i].mean() - gm) ** 2 for i in idx if len(i))
        within = sum(((v[i] - v[i].mean()) ** 2).sum() for i in idx if len(i) > 1)
        df_w = max(int(n.sum() - G), 1)
        ms_w = within / df_w
        ms_b = between / max(G - 1, 1)
        n0 = (n.sum() - (n ** 2).sum() / n.sum()) / max(G - 1, 1)
        var_b = max((ms_b - ms_w) / max(n0, 1e-9), 0.0)
        rho = float(var_b / max(var_b + ms_w, 1e-12))
    gamma = n + n * (n - 1.0) * rho
    Gamma = float(np.mean((gamma / gamma.mean() - 1.0) ** 2))
    return {"G": int(G), "n_units": int(n.sum()), "rho_hat": float(rho),
            "Gamma": Gamma, "G_star": float(G / (1.0 + Gamma)),
            "kish_neff_units": kish_neff(n), "largest_cluster": int(n.max()),
            "largest_cluster_name": str(names[int(np.argmax(n))])}


# --------------------------------------------------------------------------------------
# point estimate and cluster-robust variances for the intercept-only model
# --------------------------------------------------------------------------------------
def _mean_and_cr(d: np.ndarray, idx: list[np.ndarray], cv: str = "CV1") -> tuple[float, float]:
    """OLS mean of ``d`` and its cluster-robust variance (CV1 sandwich or CV3 cluster jackknife)."""
    N = len(d)
    G = len(idx)
    mu = float(d.mean())
    if cv == "CV1":
        e = d - mu
        meat = sum((e[i].sum()) ** 2 for i in idx)
        c = (G / max(G - 1, 1)) * ((N - 1) / max(N - 1, 1))   # K = 1 -> (N-1)/(N-K) = 1
        return mu, float(c * meat / N ** 2)
    if cv == "CV3":                       # cluster jackknife, MNW (JAE 2023) default fallback
        loo = np.array([float(np.delete(d, i).mean()) for i in idx])
        return mu, float((G - 1) / G * ((loo - loo.mean()) ** 2).sum())
    raise ValueError(cv)


@dataclass
class WCRResult:
    comparison: str
    n_units: int
    G: int
    point: float
    se_cv1: float
    se_cv3: float
    t_cv1: float
    p_wcr: float
    p_cv1_t: float          # naive: t against t(G-1)
    ci_low: float
    ci_high: float
    ci_method: str
    weights: str
    reps: int
    clusters_positive: int
    clusters_negative: int
    G_star: float
    kish_neff_units: float


# --------------------------------------------------------------------------------------
# the wild cluster bootstrap-t
# --------------------------------------------------------------------------------------
def _wcr_pvalue(d: np.ndarray, idx: list[np.ndarray], mu0: float, V: np.ndarray,
                cv: str) -> float:
    """Two-sided symmetric WCR p for H0: mu = mu0.  ``V`` is (reps, G) of bootstrap weights."""
    mu, var = _mean_and_cr(d, idx, cv)
    t_hat = abs(mu - mu0) / np.sqrt(max(var, 1e-300))
    u = d - mu0                                     # restricted residuals
    N = len(d)
    G = len(idx)
    # expand cluster weights to units once
    unit_block = np.empty(N, dtype=int)
    for j, i in enumerate(idx):
        unit_block[i] = j
    W = V[:, unit_block]                            # (reps, N)
    star = u[None, :] * W                           # bootstrap samples, null imposed
    mu_star = star.mean(axis=1)
    e_star = star - mu_star[:, None]
    if cv == "CV1":
        sums = np.zeros((V.shape[0], G))
        for j, i in enumerate(idx):
            sums[:, j] = e_star[:, i].sum(axis=1)
        c = G / max(G - 1, 1)
        var_star = c * (sums ** 2).sum(axis=1) / N ** 2
    else:
        loo = np.empty((V.shape[0], G))
        tot = star.sum(axis=1)
        for j, i in enumerate(idx):
            loo[:, j] = (tot - star[:, i].sum(axis=1)) / (N - len(i))
        var_star = (G - 1) / G * ((loo - loo.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    t_star = np.abs(mu_star) / np.sqrt(np.maximum(var_star, 1e-300))
    # +1 correction: the observed sample is one of the possible sign patterns
    return float((1.0 + (t_star >= t_hat - 1e-12).sum()) / (1.0 + len(t_star)))


def wcr_test(delta: np.ndarray, cluster: np.ndarray, *, comparison: str = "",
             reps: int = 9_999, seed: int = 8675309, weights: str = "rademacher",
             cv: str = "CV1", ci: bool = True, ci_grid: int = 41) -> WCRResult:
    """Restricted wild cluster bootstrap-t for H0: mean(delta) = 0, clustered by ``cluster``.

    ``delta`` is one number per scoring unit (per extractant: reference MAE - candidate MAE,
    positive = candidate better), ``cluster`` its chemotype.  With Rademacher weights this is a
    chemotype-level sign-flip test.  ``weights='webb'`` uses Webb's six-point distribution, which
    MNW recommend when G is small enough that 2^G sign patterns is a binding coarseness
    (G < ~12); at G = 40 it makes no difference and is kept only as a robustness switch.

    The interval is obtained by *inverting* the same test on a grid of null values and bisecting
    at p = 0.05 -- the method MNW recommend -- not by a percentile of the resampled mean.
    """
    d = np.asarray(delta, dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    cl = np.asarray(cluster).astype(str)[ok]
    idx, _ = _blocks(cl)
    G = len(idx)
    rng = np.random.default_rng(seed)
    if weights == "rademacher":
        V = rng.choice(np.array([-1.0, 1.0]), size=(reps, G))
    elif weights == "webb":
        V = rng.choice(WEBB6, size=(reps, G))
    else:
        raise ValueError(weights)

    mu, var1 = _mean_and_cr(d, idx, "CV1")
    _, var3 = _mean_and_cr(d, idx, "CV3")
    p0 = _wcr_pvalue(d, idx, 0.0, V, cv)
    from scipy import stats
    t1 = mu / np.sqrt(max(var1, 1e-300))
    p_naive = float(2 * stats.t.sf(abs(t1), df=max(G - 1, 1)))

    lo = hi = np.nan
    method = "none"
    if ci:
        se = np.sqrt(max(var1, 1e-300))
        def p_at(m):
            return _wcr_pvalue(d, idx, float(m), V, cv)
        # bracket then bisect on each side of the point estimate
        def solve(direction: int) -> float:
            step, edge = se, mu
            for _ in range(ci_grid):
                cand = mu + direction * step
                if p_at(cand) < 0.05:
                    edge = cand
                    break
                step *= 1.6
            else:
                return np.nan
            a, b = mu, edge
            for _ in range(40):
                m = 0.5 * (a + b)
                if p_at(m) >= 0.05:
                    a = m
                else:
                    b = m
            return float(0.5 * (a + b))
        lo, hi = solve(-1), solve(+1)
        method = "WCR test inversion (bisection at p=0.05)"

    eff = css_effective_clusters(cl, values=d)
    per_cluster = np.array([d[i].mean() for i in idx])
    return WCRResult(comparison=comparison, n_units=int(len(d)), G=int(G), point=float(mu),
                     se_cv1=float(np.sqrt(var1)), se_cv3=float(np.sqrt(var3)),
                     t_cv1=float(t1), p_wcr=float(p0), p_cv1_t=p_naive,
                     ci_low=float(lo), ci_high=float(hi), ci_method=method,
                     weights=weights, reps=int(reps),
                     clusters_positive=int((per_cluster > 0).sum()),
                     clusters_negative=int((per_cluster < 0).sum()),
                     G_star=eff["G_star"], kish_neff_units=eff["kish_neff_units"])


def percentile_block_p(delta: np.ndarray, cluster: np.ndarray, *, reps: int = 10_000,
                       seed: int = 8675309) -> dict:
    """The *existing* gen13 estimator, reimplemented, so the two can be printed side by side."""
    d = np.asarray(delta, dtype=float)
    ok = np.isfinite(d)
    d, cl = d[ok], np.asarray(cluster).astype(str)[ok]
    idx, names = _blocks(cl)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(reps, len(names)))
    draws = np.array([d[np.concatenate([idx[j] for j in row])].mean() for row in picks])
    p = float(2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return {"point": float(d.mean()), "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)), "p_percentile": min(1.0, p),
            "boot_se": float(draws.std(ddof=1))}


def compare_inference(delta: np.ndarray, cluster: np.ndarray, *, comparison: str = "",
                      reps: int = 9_999, seed: int = 8675309) -> dict:
    """One row: point, the old percentile interval/p, the WCR interval/p, CV1 and CV3 SEs."""
    old = percentile_block_p(delta, cluster, reps=reps + 1, seed=seed)
    new = wcr_test(delta, cluster, comparison=comparison, reps=reps, seed=seed)
    row = {"comparison": comparison, "point": new.point, "n_units": new.n_units, "G": new.G,
           "G_star": round(new.G_star, 2), "kish_neff_units": round(new.kish_neff_units, 2),
           "pct_ci_low": old["ci_low"], "pct_ci_high": old["ci_high"],
           "p_percentile": old["p_percentile"], "pct_se": old["boot_se"],
           "se_cv1": new.se_cv1, "se_cv3": new.se_cv3, "t_cv1": new.t_cv1,
           "p_cv1_t": new.p_cv1_t, "p_wcr": new.p_wcr,
           "wcr_ci_low": new.ci_low, "wcr_ci_high": new.ci_high,
           "clusters_positive": new.clusters_positive, "clusters_negative": new.clusters_negative}
    row["se_ratio_cv3_over_pct"] = new.se_cv3 / max(old["boot_se"], 1e-12)
    row["verdict_changes_at_05"] = bool((old["p_percentile"] < 0.05) != (new.p_wcr < 0.05))
    return row


# --------------------------------------------------------------------------------------
# calibration: does either method hold its size on THIS cluster structure?
# --------------------------------------------------------------------------------------
def size_study(cluster: np.ndarray, *, icc: float = 0.72, sims: int = 400, reps: int = 999,
               seed: int = 12345, sd: float = 1.0) -> dict:
    """Rejection rate of both tests under the null on the corpus's own chemotype structure.

    Data are generated with a chemotype random effect of intraclass correlation ``icc`` (0.72 is
    the amplitude ICC measured in gen13 stage 2) and *zero* mean, so every rejection is a false
    positive.  A nominal-5% test that rejects 12% of the time is the thing this whole exercise is
    about; run it before rewriting any p-value.
    """
    cl = np.asarray(cluster).astype(str)
    idx, names = _blocks(cl)
    G, N = len(idx), len(cl)
    rng = np.random.default_rng(seed)
    s_b = sd * np.sqrt(icc)
    s_w = sd * np.sqrt(1.0 - icc)
    Vw = rng.choice(np.array([-1.0, 1.0]), size=(reps, G))
    rej_pct = rej_wcr = rej_t = 0
    for _ in range(sims):
        d = np.empty(N)
        for j, i in enumerate(idx):
            d[i] = rng.normal(0.0, s_b) + rng.normal(0.0, s_w, size=len(i))
        picks = rng.integers(0, G, size=(reps, G))
        draws = np.array([d[np.concatenate([idx[j] for j in row])].mean() for row in picks])
        p_pct = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
        rej_pct += p_pct < 0.05
        rej_wcr += _wcr_pvalue(d, idx, 0.0, Vw, "CV1") < 0.05
        mu, var = _mean_and_cr(d, idx, "CV1")
        from scipy import stats
        rej_t += 2 * stats.t.sf(abs(mu) / np.sqrt(var), df=G - 1) < 0.05
    return {"icc": icc, "sims": sims, "G": G, "n_units": N,
            "size_percentile_bootstrap": rej_pct / sims,
            "size_wcr_bootstrap_t": rej_wcr / sims,
            "size_cv1_t": rej_t / sims, "nominal": 0.05}
