"""Measurement adapters: how a zero-shot curve is corrected by k measured separation factors.

Every adapter takes the zero-shot centred curve ``c0`` (14,), the training-derived residual
covariance ``cov`` (14 x 14, PSD), the support (pairs + values) and the noise matrix ``R`` for
those pairs, and returns an adapted curve.  Nothing here reads anything but the support values.

* ``blup``: the conditional mean of the curve given the supports (gen15's estimator with a general
  ``R``); ``blup_posterior`` its posterior covariance.
* ``naive_line``: least-squares radius ramp through the supports, no corpus at all (gen15).
* ``ridge_line``: radius + radius² coefficients shrunk toward a training-fold prior
  (``prior_coef``), weighted by ``R^-1``; the two-parameter statistical baseline with the
  shrinkage lesson of gen8 ("four ridge-shrunk coefficients beat one free one").
* ``rescale``: the zero-shot curve times one scalar fitted to the supports (for the nearest-
  neighbour lookup baseline).
"""
from __future__ import annotations

import numpy as np

from .data import N_LN
from .design import Support


def pair_matrix(pairs) -> np.ndarray:
    D = np.zeros((len(pairs), N_LN))
    for i, (a, b) in enumerate(pairs):
        D[i, a], D[i, b] = 1.0, -1.0
    return D


def blup(c0: np.ndarray, cov: np.ndarray, support: Support, R: np.ndarray) -> np.ndarray:
    if support.k == 0:
        return np.asarray(c0, dtype=float).copy()
    D = pair_matrix(support.pairs)
    r = np.asarray(support.values, dtype=float) - D @ c0
    S = D @ cov @ D.T + R
    w = np.linalg.lstsq(S, r, rcond=None)[0]        # minimum-norm; S is singular for redundant supports
    return c0 + cov @ D.T @ w


def blup_posterior(cov: np.ndarray, pairs, R: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return cov.copy()
    D = pair_matrix(pairs)
    S = D @ cov @ D.T + R
    G = np.linalg.lstsq(S, D @ cov, rcond=None)[0]
    return cov - cov @ D.T @ G


def pair_sd(post: np.ndarray, a: int, b: int, noise_var: float) -> float:
    d = np.zeros(N_LN)
    d[a], d[b] = 1.0, -1.0
    return float(np.sqrt(max(d @ post @ d + noise_var, 1e-12)))


def naive_line(basis: np.ndarray, support: Support) -> np.ndarray:
    if support.k == 0:
        return np.zeros(N_LN)
    r = basis[0]
    x = np.array([r[a] - r[b] for a, b in support.pairs])
    y = np.asarray(support.values, dtype=float)
    den = float(x @ x)
    return (float(x @ y) / den if den > 1e-12 else 0.0) * r


def ridge_line(basis: np.ndarray, support: Support, prior_coef: np.ndarray, lam: float,
               R: np.ndarray | None = None) -> np.ndarray:
    """``argmin (y - D B' coef)' R^-1 (y - D B' coef) + lam ||coef - prior||^2``; k = 0 returns the
    prior curve.  ``basis`` is k x 14 (gen13's physics basis, row norm sqrt(14))."""
    prior = np.asarray(prior_coef, dtype=float)
    if support.k == 0:
        return prior @ basis
    D = pair_matrix(support.pairs)
    A = D @ basis.T                                  # k_supports x n_coef
    y = np.asarray(support.values, dtype=float)
    Rinv = np.linalg.pinv(R) if R is not None else np.eye(len(y))
    H = A.T @ Rinv @ A + lam * np.eye(basis.shape[0])
    g = A.T @ Rinv @ y + lam * prior
    coef = np.linalg.solve(H, g)
    return coef @ basis


def rescale(c0: np.ndarray, support: Support, shrink: float = 1.0) -> np.ndarray:
    """``s * c0`` with ``s`` the least-squares scale of the supports on the zero-shot contrasts,
    shrunk toward 1 by ``shrink`` pseudo-observations."""
    if support.k == 0:
        return np.asarray(c0, dtype=float).copy()
    D = pair_matrix(support.pairs)
    x = D @ c0
    y = np.asarray(support.values, dtype=float)
    den = float(x @ x) + shrink
    s = (float(x @ y) + shrink) / den if den > 1e-12 else 1.0
    return s * np.asarray(c0, dtype=float)


def pairs_from_curve(curve: np.ndarray, pairs) -> np.ndarray:
    return np.array([curve[a] - curve[b] for a, b in pairs], dtype=float)


def project(support: Support, ridge: float = 1e-9) -> np.ndarray:
    """Arithmetic only: the minimum-norm centred curve that reproduces the supports exactly
    (``D^T (D D^T)^+ y``).  No corpus, no chemistry, no radius; what the measurements alone fix."""
    if support.k == 0:
        return np.zeros(N_LN)
    D = pair_matrix(support.pairs)
    y = np.asarray(support.values, dtype=float)
    c = D.T @ np.linalg.lstsq(D @ D.T + ridge * np.eye(len(y)), y, rcond=None)[0]
    return c - c.mean()
