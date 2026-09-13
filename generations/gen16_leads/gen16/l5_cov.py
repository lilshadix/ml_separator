"""L5 covariance estimators for the measured mode (PRE_REGISTRATION §3 L5).

Every estimator has the hook signature

    estimator(rows: ndarray (n_curves x 14, NaN where a cell did not measure a metal),
              groups: ndarray of chemotype per row) -> 14 x 14 covariance

and is used by ``l5_fewshot.evaluate`` in place of ``gen15.fewshot.residual_covariance(rows)``.
Everything else in the measured-mode route (leave-chemotype-out, publication mask, greedy
D-optimal support, BLUP, noise) is untouched.

* ``POOLED``    the deployed estimator, literally ``residual_covariance`` (shrink 0.25 to the
                diagonal, ridge 1e-3 x median variance).
* ``LW``        Ledoit-Wolf (2004) shrinkage of the pairwise-complete second-moment matrix toward
                a scaled identity.  ``sklearn.covariance.LedoitWolf`` cannot take the NaN-padded
                rows (a cell measures 2-14 of the 14 metals; zero-filling would scale every entry by
                its measurement fraction), so the LW intensity is computed by the same formula with
                each entry's sampling variance taken over its own pair count.  On complete rows it
                is *identical* to sklearn's (``check_lw_matches_sklearn``).
* ``LOWRANK2`` / ``LOWRANK3``  probabilistic PCA of the pairwise-complete matrix: the top-r
                eigencomponents plus an isotropic residual variance equal to the mean of the
                remaining eigenvalues, plus the same ridge.
* ``HIER``      chemotype-balanced: each residual row weighted by 1 / (rows in its chemotype) so
                every chemotype carries equal total mass, weighted pairwise-complete second moment,
                then the same shrink 0.25 toward its own diagonal and the same ridge as
                ``residual_covariance``.

All of them share ``residual_covariance``'s conventions for sparsely measured metal pairs
(entries with fewer than 20 co-measured curves are zero, a non-positive variance is replaced by
the median positive variance) so the only thing that differs between arms is the shrinkage
structure, which is the registered question.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from gen16 import bootstrap  # noqa: F401  (sys.path)
from gen15.fewshot import N_LN, residual_covariance  # noqa: E402

SHRINK = 0.25
RIDGE = 1e-3
MIN_PAIRS = 20

Estimator = Callable[[np.ndarray, np.ndarray], np.ndarray]


def pairwise_moment(rows: np.ndarray, weights: np.ndarray | None = None, *,
                    min_pairs: int = MIN_PAIRS) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Pairwise-complete (optionally weighted) second-moment matrix of centred residual rows.

    Mirrors ``residual_covariance`` up to (not including) its shrink and ridge: entries with fewer
    than ``min_pairs`` co-measured curves are 0, a non-positive variance is replaced by the median
    positive variance (``fallback``), and the result is symmetrised.  Returns
    ``(S, fallback, counts, filled)`` with ``filled`` the NaN->0 rows.
    """
    rows = np.asarray(rows, dtype=float)
    mask = ~np.isnan(rows)
    filled = np.where(mask, rows, 0.0)
    m = mask.astype(float)
    counts = m.T @ m
    if weights is None:
        cross, denom = filled.T @ filled, counts
    else:
        w = np.asarray(weights, dtype=float)
        cross = (filled * w[:, None]).T @ filled
        denom = (m * w[:, None]).T @ m
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(counts >= min_pairs, cross / np.maximum(denom, 1e-300), 0.0)
    var = np.diag(S).copy()
    fallback = float(np.nanmedian(var[var > 0])) if np.any(var > 0) else 1.0
    var[var <= 0] = fallback
    np.fill_diagonal(S, var)
    S = (S + S.T) / 2.0
    return S, fallback, counts, filled


def pooled_cov(rows: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    """The deployed estimator, unchanged."""
    return residual_covariance(np.asarray(rows, dtype=float))


def ledoit_wolf_shrinkage_pairwise(rows: np.ndarray, *, min_pairs: int = MIN_PAIRS) -> tuple[float, np.ndarray, float]:
    """LW intensity for NaN-padded rows: sum of entry sampling variances / squared distance to mu*I.

    With complete rows this is sklearn's ``ledoit_wolf_shrinkage(X, assume_centered=True)``
    exactly: beta_ij = (1/n) [ (1/n) sum_k x_ki^2 x_kj^2 - S_ij^2 ], delta = ||S - mu I||_F^2.
    """
    S, fallback, counts, filled = pairwise_moment(rows, min_pairs=min_pairs)
    p = S.shape[0]
    mu = float(np.trace(S)) / p
    sq = filled ** 2
    m2 = sq.T @ sq
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_sq = np.where(counts >= min_pairs, m2 / np.maximum(counts, 1.0), 0.0)
        beta_ij = np.where(counts >= min_pairs, (mean_sq - S ** 2) / np.maximum(counts, 1.0), 0.0)
    beta_sum = float(beta_ij.sum())
    delta_sum = float(((S - mu * np.eye(p)) ** 2).sum())
    if delta_sum <= 0.0 or beta_sum <= 0.0:
        s = 0.0
    else:
        s = min(1.0, beta_sum / delta_sum)
    return s, S, fallback


def ledoit_wolf_cov(rows: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
    s, S, fallback = ledoit_wolf_shrinkage_pairwise(rows)
    p = S.shape[0]
    mu = float(np.trace(S)) / p
    cov = (1.0 - s) * S + s * mu * np.eye(p)
    return cov + RIDGE * np.eye(p) * fallback


def lowrank_cov(rows: np.ndarray, groups: np.ndarray | None = None, *, rank: int = 2) -> np.ndarray:
    """Probabilistic PCA: W W' + sigma^2 I with W from the top-``rank`` eigenpairs."""
    S, fallback, _, _ = pairwise_moment(rows)
    lam, V = np.linalg.eigh(S)
    lam, V = lam[::-1], V[:, ::-1]
    p = len(lam)
    r = int(min(rank, p - 1))
    sigma2 = float(np.mean(np.clip(lam[r:], 0.0, None)))
    sigma2 = max(sigma2, 1e-6 * fallback)
    core = np.clip(lam[:r] - sigma2, 0.0, None)
    cov = (V[:, :r] * core) @ V[:, :r].T + sigma2 * np.eye(p)
    cov = (cov + cov.T) / 2.0
    return cov + RIDGE * np.eye(p) * fallback


def lowrank2_cov(rows, groups=None):
    return lowrank_cov(rows, groups, rank=2)


def lowrank3_cov(rows, groups=None):
    return lowrank_cov(rows, groups, rank=3)


def hier_cov(rows: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Chemotype-balanced second moment, then residual_covariance's shrink and ridge."""
    g = np.asarray(groups).astype(str)
    if len(g) != len(rows):
        raise ValueError("groups must align with rows")
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    w = w / w.sum()
    S, fallback, _, _ = pairwise_moment(rows, weights=w)
    var = np.diag(S).copy()
    cov = (1.0 - SHRINK) * S + SHRINK * np.diag(var)
    return cov + RIDGE * np.eye(S.shape[0]) * fallback


ESTIMATORS: dict[str, Estimator] = {
    "POOLED": pooled_cov,
    "LW": ledoit_wolf_cov,
    "LOWRANK2": lowrank2_cov,
    "LOWRANK3": lowrank3_cov,
    "HIER": hier_cov,
}


# --------------------------------------------------------------------------------------
# self-checks (run by scripts/l5_verify.py before any arm is scored)
# --------------------------------------------------------------------------------------
def check_pooled_matches_residual_covariance(rows: np.ndarray) -> float:
    """``pairwise_moment`` + shrink + ridge must equal ``residual_covariance`` to the last bit."""
    S, fallback, _, _ = pairwise_moment(rows)
    var = np.diag(S).copy()
    mine = (1.0 - SHRINK) * S + SHRINK * np.diag(var) + RIDGE * np.eye(S.shape[0]) * fallback
    return float(np.abs(mine - residual_covariance(rows)).max())


def check_lw_matches_sklearn(n: int = 200, seed: int = 0) -> tuple[float, float]:
    """On complete rows the pairwise LW intensity and covariance equal sklearn's LedoitWolf."""
    from sklearn.covariance import LedoitWolf
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N_LN, 3))
    X = rng.normal(size=(n, 3)) @ A.T + 0.3 * rng.normal(size=(n, N_LN))
    X = X - X.mean(axis=1, keepdims=True)            # centred rows, as the residual curves are
    lw = LedoitWolf(assume_centered=True).fit(X)
    s, S, fallback = ledoit_wolf_shrinkage_pairwise(X)
    mu = float(np.trace(S)) / N_LN
    mine = (1.0 - s) * S + s * mu * np.eye(N_LN)
    return abs(float(lw.shrinkage_) - s), float(np.abs(lw.covariance_ - mine).max())
