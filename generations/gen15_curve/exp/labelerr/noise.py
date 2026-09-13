"""Label-error machinery for gen15: how noisy is the two-stage (a, b) label, and what does that cost?

The gen13/14/15 target is built in two stages.  Stage one turns a cell's centred 14-metal log-D
curve into ``(a, b)`` by a per-cell ridge (penalty 0.5, basis rows at norm sqrt(14)); stage two
fits a model to those coefficients.  A noisy first stage attenuates everything downstream, so
before anything else we need the sampling variance of ``a`` and ``b``.

Three things this module provides.

``sigma_table``     per-(cell, metal) standard error of the recorded log D, under several
                    estimators of the per-measurement noise sigma, because the corpus gives two
                    mutually inconsistent handles on sigma and the choice must be exposed, not
                    hidden.
``coef_cov``        the exact sampling covariance of ``(a, b)`` for every cell, from the ridge hat
                    matrix and the centring operator -- no simulation needed, it is a linear map.
``shrink``          the one-stage / hierarchical alternative to the two-stage label: an empirical
                    Bayes (EM) fit of ``logD_ij = alpha_i + a_i r_j + b_i r_j^2`` with
                    ``(a_i, b_i) ~ N(m_group(i), T)``, fitted on a supplied subset of cells only,
                    so a fold can de-noise its training labels without ever touching a test cell.

The two handles on sigma
------------------------
* **replicates**  216 of 3359 (cell, metal) means rest on more than one row, in 41 of 521 cells.
  Their within-(cell, metal) sd has median 0.302 and a dof-pooled RMS of 0.760.
* **residuals**   every cell with m >= 5 metals has a residual about its own fitted quadratic, with
  m - 3 degrees of freedom (two basis rows plus the centring).  Its dof-pooled RMS is ~0.17.

The second is an *upper* bound on the per-metal noise, because it also contains any real
non-quadratic chemistry (tetrad, Gd break).  It is four times smaller than the first.  The two
cannot both be measuring the same quantity, and the residual bound is the binding one, so the
replicate spread must be mostly a hidden between-series axis rather than measurement noise on a
metal within a cell.  Everything downstream is reported at both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
for p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen15_curve"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gen13sep.metals import LANTHANIDES  # noqa: E402

RIDGE = 0.5
N_BASIS_DOF = 3          # two basis rows + the row-centring


# ------------------------------------------------------------------------------------------
# sigma
# ------------------------------------------------------------------------------------------
def replicate_arrays(bench) -> tuple[np.ndarray, np.ndarray]:
    """(nrep, repsd) as cells x 14 arrays, NaN where the metal is unmeasured."""
    f = bench.frame
    nr = f[[f"nrep__{m}" for m in LANTHANIDES]].to_numpy(dtype=float)
    sd = f[[f"repsd__{m}" for m in LANTHANIDES]].to_numpy(dtype=float)
    obs = ~np.isnan(bench.Y)
    nr = np.where(obs, np.where(np.isfinite(nr), nr, 1.0), np.nan)
    sd = np.where(obs, sd, np.nan)
    return nr, sd


def cell_residual_sigma(bench, ridge: float = RIDGE, min_metals: int = 5) -> np.ndarray:
    """Per-cell RMS residual about its own fitted quadratic, dof = m - 3.  NaN if m < min_metals."""
    Y, basis = bench.Y, bench.basis
    obs = ~np.isnan(Y)
    C = Y - np.nanmean(Y, axis=1, keepdims=True)
    out = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        o = obs[i]
        m = int(o.sum())
        if m < max(min_metals, N_BASIS_DOF + 1):
            continue
        B = basis[:, o].T
        y = C[i, o]
        c = np.linalg.solve(B.T @ B + ridge * np.eye(basis.shape[0]), B.T @ y)
        r = y - B @ c
        out[i] = float(np.sqrt((r ** 2).sum() / (m - N_BASIS_DOF)))
    return out


def pooled_residual_sigma(bench, ridge: float = RIDGE, min_metals: int = 5) -> float:
    s = cell_residual_sigma(bench, ridge, min_metals)
    m = bench.frame.n_metals.to_numpy(dtype=float)
    ok = np.isfinite(s)
    dof = m[ok] - N_BASIS_DOF
    return float(np.sqrt(np.sum(dof * s[ok] ** 2) / dof.sum()))


def pooled_replicate_sigma(bench) -> tuple[float, float]:
    """(dof-pooled RMS, median) of the within-(cell, metal) replicate sd."""
    nr, sd = replicate_arrays(bench)
    m = np.isfinite(sd) & (nr >= 2)
    dof = nr[m] - 1.0
    return float(np.sqrt(np.sum(dof * sd[m] ** 2) / dof.sum())), float(np.median(sd[m]))


def sigma_table(bench, mode: str = "resid_const", *, sigma: float | None = None,
                shrink_k: float = 6.0) -> np.ndarray:
    """Per-(cell, metal) *standard error of the recorded mean* log D, cells x 14, NaN off-support.

    ``resid_const``  one global sigma from the pooled residual bound, divided by sqrt(nrep).
    ``resid_cell``   the cell's own residual sigma, shrunk toward the global with weight
                     ``dof / (dof + shrink_k)``; cells with m < 5 take the global value.
    ``rep_const``    one global sigma from the dof-pooled replicate sd.
    ``rep_cell``     the (cell, metal) replicate sd where it exists, else the cell's median
                     replicate sd, else the global replicate sigma.
    ``const``        an explicit ``sigma`` supplied by the caller.
    """
    nr, rsd = replicate_arrays(bench)
    obs = ~np.isnan(bench.Y)
    if mode == "const":
        if sigma is None:
            raise ValueError("mode='const' needs sigma")
        s = np.full(bench.Y.shape, float(sigma))
    elif mode == "resid_const":
        s = np.full(bench.Y.shape, pooled_residual_sigma(bench))
    elif mode == "rep_const":
        s = np.full(bench.Y.shape, pooled_replicate_sigma(bench)[0])
    elif mode == "resid_cell":
        g = pooled_residual_sigma(bench)
        cs = cell_residual_sigma(bench)
        m = bench.frame.n_metals.to_numpy(dtype=float)
        dof = np.maximum(m - N_BASIS_DOF, 0.0)
        lam = dof / (dof + shrink_k)
        var = np.where(np.isfinite(cs), lam * np.nan_to_num(cs) ** 2 + (1 - lam) * g ** 2, g ** 2)
        s = np.sqrt(var)[:, None] * np.ones((1, bench.Y.shape[1]))
    elif mode == "rep_cell":
        g, _ = pooled_replicate_sigma(bench)
        cellmed = np.nanmedian(np.where(nr >= 2, rsd, np.nan), axis=1)
        s = np.where(np.isfinite(rsd) & (nr >= 2), rsd,
                     np.where(np.isfinite(cellmed)[:, None], cellmed[:, None], g))
        s = np.maximum(s, 0.02)
    else:
        raise ValueError(f"unknown sigma mode {mode!r}")
    se = s / np.sqrt(np.where(np.isfinite(nr), np.maximum(nr, 1.0), 1.0))
    return np.where(obs, se, np.nan)


# ------------------------------------------------------------------------------------------
# exact sampling covariance of the two-stage label
# ------------------------------------------------------------------------------------------
def coef_operator(bench, i: int, ridge: float = RIDGE) -> tuple[np.ndarray, np.ndarray]:
    """(G, obs) with ``coef_i = G @ y_i[obs]``: the ridge hat matrix folded with the row-centring."""
    o = ~np.isnan(bench.Y[i])
    m = int(o.sum())
    B = bench.basis[:, o].T
    k = bench.basis.shape[0]
    H = np.linalg.solve(B.T @ B + ridge * np.eye(k), B.T)      # k x m
    M = np.eye(m) - np.ones((m, m)) / m
    return H @ M, o


def coef_cov(bench, se: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Sampling covariance of every cell's ``(a, b)``: cells x k x k."""
    n, k = len(bench.Y), bench.basis.shape[0]
    out = np.zeros((n, k, k))
    for i in range(n):
        G, o = coef_operator(bench, i, ridge)
        v = se[i, o] ** 2
        out[i] = (G * v) @ G.T
    return out


def reliability(values: np.ndarray, var: np.ndarray, w: np.ndarray | None = None) -> dict:
    """ICC-style reliability: 1 - E[sampling variance] / Var(observed)."""
    v = np.asarray(values, dtype=float)
    s = np.asarray(var, dtype=float)
    if w is None:
        w = np.ones(len(v))
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    mu = float(np.sum(w * v))
    obs = float(np.sum(w * (v - mu) ** 2))
    noise = float(np.sum(w * s))
    return {"var_observed": obs, "var_sampling": noise,
            "var_true": max(obs - noise, 0.0),
            "reliability": float(np.clip(1.0 - noise / obs, 0.0, 1.0)) if obs > 0 else np.nan,
            "n": int(len(v))}


# ------------------------------------------------------------------------------------------
# the one-stage / hierarchical alternative to the two-stage label
# ------------------------------------------------------------------------------------------
def shrink(bench, fit_index: np.ndarray, se: np.ndarray, *, group: str = "extractant",
           iters: int = 120, ridge: float = RIDGE, tol: float = 1e-6,
           floor: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Empirical-Bayes de-noising of the per-cell coefficients (the one-stage alternative).

    Model, over all (cell, metal) observations of the cells in ``fit_index``::

        logD_ij = alpha_i + a_i r_j + b_i r_j^2 + eps_ij ,   eps_ij ~ N(0, se_ij^2)
        theta_i = (a_i, b_i) ~ N(mu_g(i), T)      g = the cell's ligand (or chemotype)
        mu_g                 ~ N(m0, T0)          group means shrunk toward the corpus mean

    ``alpha_i`` is profiled out by the row-centring -- exactly what stage one does -- so the
    likelihood for ``theta_i`` is Gaussian with precision ``P_i = Bc' S_i^-1 Bc`` and score
    ``r_i = Bc' S_i^-1 yc`` on the centred rows.  ``P_i`` is singular for a two-metal cell; nothing
    below ever inverts it.

    EM, three blocks per sweep:

      * group means from the *marginal* of the data, ``theta_hat_i ~ N(mu_g, T + V_i)``, written
        without ``V_i = P_i^-1`` using ``(T + P^-1)^-1 = P - P (P + T^-1)^-1 P``.  Using the
        marginal rather than the posterior means is what keeps the group mean from chasing its own
        shrinkage (the naive version has a positive feedback and diverges);
      * ``m0`` and ``T0`` from the spread of the group means plus their posterior covariances;
      * ``T`` from the posterior deviations plus the posterior covariances -- the standard EM
        update for a normal prior's covariance.

    Only cells in ``fit_index`` enter any estimate.  The returned array carries the posterior mean
    for those cells and the untouched stage-one coefficient everywhere else, so an arm can use it
    without a test cell ever having entered a prior.

    ``group='none'`` collapses every cell into one group (pure global shrinkage).
    """
    k = bench.basis.shape[0]
    Y = bench.Y
    fit_index = np.asarray(fit_index)
    gcol = (np.full(len(Y), "ALL") if group == "none"
            else bench.frame[group].astype(str).to_numpy())
    coef = bench.coef.copy()

    P, r = {}, {}
    for i in fit_index:
        o = ~np.isnan(Y[i])
        m = int(o.sum())
        B = bench.basis[:, o].T
        Mc = np.eye(m) - np.ones((m, m)) / m
        Bc, yc = Mc @ B, Mc @ Y[i, o]
        w = 1.0 / np.maximum(se[i, o] ** 2, 1e-8)
        P[i] = Bc.T @ (Bc * w[:, None])
        r[i] = Bc.T @ (yc * w)

    groups: dict[str, list[int]] = {}
    for i in fit_index:
        groups.setdefault(gcol[i], []).append(int(i))
    gk = list(groups)

    theta0 = coef[fit_index]
    T = np.cov(theta0.T) + floor * np.eye(k)
    m0 = theta0.mean(axis=0)
    T0 = T.copy()
    mu = {g: m0.copy() for g in gk}
    info = {"iters": 0, "converged": False}

    def _psd(M):
        M = 0.5 * (M + M.T)
        val, vec = np.linalg.eigh(M)
        return (vec * np.maximum(val, floor)) @ vec.T

    for it in range(iters):
        Ti = np.linalg.inv(T)
        W, u, Ainv = {}, {}, {}
        for i in fit_index:
            Ai = np.linalg.inv(P[i] + Ti)
            Ainv[i] = Ai
            W[i] = P[i] - P[i] @ Ai @ P[i]           # (T + V_i)^-1, valid for singular P_i
            u[i] = r[i] - P[i] @ Ai @ r[i]           # (T + V_i)^-1 theta_hat_i
        T0i = np.linalg.inv(T0)
        mu_new, Cg = {}, {}
        for g, idx in groups.items():
            Wg = sum(W[i] for i in idx)
            ug = sum(u[i] for i in idx)
            Cg[g] = np.linalg.inv(Wg + T0i)
            mu_new[g] = Cg[g] @ (ug + T0i @ m0)
        Pm = sum(np.linalg.inv(T0 + Cg[g]) for g in gk)
        bm = sum(np.linalg.inv(T0 + Cg[g]) @ mu_new[g] for g in gk)
        m0_new = np.linalg.solve(Pm, bm)
        S0 = np.zeros((k, k))
        for g in gk:
            d = (mu_new[g] - m0_new).reshape(-1, 1)
            S0 += d @ d.T + Cg[g]
        T0_new = _psd(S0 / len(gk))
        S = np.zeros((k, k))
        for i in fit_index:
            pm = Ainv[i] @ (r[i] + Ti @ mu_new[gcol[i]])
            d = (pm - mu_new[gcol[i]]).reshape(-1, 1)
            S += d @ d.T + Ainv[i]
        T_new = _psd(S / len(fit_index))
        delta = float(max(np.abs(T_new - T).max(), np.abs(T0_new - T0).max(),
                          max(np.abs(mu_new[g] - mu[g]).max() for g in gk)))
        T, T0, mu, m0 = T_new, T0_new, mu_new, m0_new
        info["iters"] = it + 1
        if delta < tol:
            info["converged"] = True
            break

    Ti = np.linalg.inv(T)
    out = coef.copy()
    for i in fit_index:
        out[i] = np.linalg.solve(P[i] + Ti, r[i] + Ti @ mu[gcol[i]])
    info.update({"T": T, "T0": T0, "m0": m0, "mu": mu, "n_groups": len(gk),
                 "group_sizes": {g: len(v) for g, v in groups.items()}})
    return out, info
