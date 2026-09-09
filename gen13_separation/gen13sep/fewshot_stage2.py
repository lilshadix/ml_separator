"""Stage-2 one-pair calibration: the best linear use of a single measured separation factor.

The locked study (``gen13sep.fewshot``) adapts a predicted curve with ``basis_shift``: it moves the
curve along the basis direction that changes the measured contrast, with a fixed shrinkage
``lam = 0.7``.  That is a reasonable heuristic but it is not the best linear estimator, and it
ignores two facts the diagnostics established:

1. **A cell's held-out pair residuals are exactly additive in one per-cell metal curve.**  A cell
   with ``m`` measured metals has ``m(m-1)/2`` pairs but only ``m-1`` independent residual numbers,
   and that residual curve is smooth along the series (lag-1 correlation +0.92).  So one measured
   pair genuinely constrains the rest, and how much it constrains them is exactly the residual
   covariance between metals.
2. **The error is dominated by one scalar.**  Sixty per cent of an arm's extractant-macro MAE on
   cells with at least eight metals is removed by a single correct per-cell amplitude.

The best linear correction from one measurement is therefore the conditional mean of the residual
curve given that measurement, which is the standard BLUP:

    u_hat = Sigma d (d' Sigma d + sigma^2)^-1 r,     d = e_a - e_b,  r = y_ab - (c_a - c_b)

with ``Sigma`` the residual covariance between metals and ``sigma^2`` the measurement noise on one
``log SF``.  ``basis_shift`` is the special case where ``Sigma`` is the basis Gram matrix and the
shrinkage is set by hand.

``Sigma`` is estimated **leave-chemotype-out** from the held-out residuals of the other chemotypes
of the same split seed, so no cell, and no member of its chemotype, contributes to the covariance
used to correct it.  The support pair is either drawn at random (the locked protocol, so the
numbers stay comparable) or chosen as the widest available ``dZ`` -- a design choice a laboratory
genuinely has, since it picks which pair to measure first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fewshot import stable_hash
from .metals import LANTHANIDES

N_LN = len(LANTHANIDES)


def centred_residual(y_row: np.ndarray, c_pred: np.ndarray) -> np.ndarray:
    """Residual curve on the metals the cell measured, centred over those metals.

    Both the observation and the prediction are only defined up to a per-cell constant, so the
    residual is centred over the observed metals; every pairwise difference is unaffected.
    """
    obs = ~np.isnan(y_row)
    out = np.full(N_LN, np.nan)
    diff = y_row[obs] - c_pred[obs]
    out[obs] = diff - diff.mean()
    return out


def residual_covariance(residuals: np.ndarray, *, shrink: float = 0.25, ridge: float = 1e-3,
                        min_pairs: int = 20) -> np.ndarray:
    """Pairwise-complete covariance of the residual curves, shrunk toward its diagonal.

    ``residuals`` is ``cells x 14`` with NaN where the metal was not measured.  Entries with fewer
    than ``min_pairs`` jointly observed cells are set to zero covariance rather than estimated from
    a handful of cells.  The shrinkage toward the diagonal keeps the matrix positive definite and
    stops a lightly sampled off-diagonal from dominating the correction.
    """
    mask = ~np.isnan(residuals)
    filled = np.where(mask, residuals, 0.0)
    counts = mask.astype(float).T @ mask.astype(float)
    cross = filled.T @ filled
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(counts >= min_pairs, cross / np.maximum(counts, 1.0), 0.0)
    var = np.diag(cov).copy()
    fallback = float(np.nanmedian(var[var > 0])) if np.any(var > 0) else 1.0
    var[var <= 0] = fallback
    np.fill_diagonal(cov, var)
    cov = (cov + cov.T) / 2.0
    cov = (1.0 - shrink) * cov + shrink * np.diag(var)
    return cov + ridge * np.eye(N_LN) * fallback


def adapt_conditional(c_pred: np.ndarray, cov: np.ndarray, a: int, b: int, y_ab: float,
                      *, noise_var: float = 0.09) -> np.ndarray:
    """Best linear correction of ``c_pred`` from one measured contrast (the BLUP).

    ``noise_var`` is the variance of a single measured ``log SF``.  The default 0.09 is
    ``2 x 0.212^2``, i.e. two independent ``log D`` readings at the cohort's median within-replicate
    sd; it is what stops the correction from chasing a noisy support measurement.
    """
    d = np.zeros(N_LN)
    d[a], d[b] = 1.0, -1.0
    r = y_ab - (c_pred[a] - c_pred[b])
    sd = cov @ d
    denom = float(d @ sd) + noise_var
    if denom <= 1e-12:
        return c_pred
    return c_pred + sd * (r / denom)


def _support_random(cell_id: str, pairs: list[tuple[int, int]], repeat: int, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(stable_hash(f"{cell_id}|{repeat}|{seed}"))
    return pairs[int(rng.integers(0, len(pairs)))]


def _support_widest(pairs: list[tuple[int, int]]) -> tuple[int, int]:
    return max(pairs, key=lambda p: (p[1] - p[0], -p[0]))


def evaluate_one_pair_conditional(curve_frames: dict[str, pd.DataFrame], cohort_frame: pd.DataFrame,
                                  *, arms: list[str], repeats: int = 5, min_metals: int = 3,
                                  shrink: float = 0.25, noise_var: float = 0.09,
                                  basis_lookup: dict | None = None) -> pd.DataFrame:
    """Score ``zero_shot``, ``conditional`` and ``conditional_widest`` on every held-out cell.

    ``curve_frames[arm]`` is the saved ``predictions/<label>/curves/<arm>.parquet`` (columns
    ``split_seed``, ``fold``, ``cell_id``, ``c__<Ln>``).  For every split seed the residual
    covariance is estimated once per chemotype from the held-out residuals of all *other*
    chemotypes of that seed, so the correction applied to a cell never saw its chemotype.
    """
    from .fewshot import adapt_basis_shift, adapt_no_model_linear, adapt_rescale

    ccols = [f"c__{m}" for m in LANTHANIDES]
    ycols = [f"logD__{m}" for m in LANTHANIDES]
    meta = cohort_frame.set_index("cell_id")
    rows: list[dict] = []

    for arm in arms:
        frame = curve_frames[arm]
        for seed, block in frame.groupby("split_seed", sort=True):
            cell_ids = block["cell_id"].tolist()
            folds = block["fold"].to_numpy()
            C = block[ccols].to_numpy(dtype=float)
            Y = meta.loc[cell_ids, ycols].to_numpy(dtype=float)
            chemo = meta.loc[cell_ids, "chemotype"].to_numpy()
            ext = meta.loc[cell_ids, "extractant"].to_numpy()
            resid = np.vstack([centred_residual(Y[i], C[i]) for i in range(len(Y))])
            cov_by_chemo = {g: residual_covariance(resid[chemo != g], shrink=shrink)
                            for g in pd.unique(chemo)}
            for i in range(len(Y)):
                obs = np.flatnonzero(~np.isnan(Y[i]))
                if len(obs) < min_metals:
                    continue
                pairs = [(a, b) for a in obs for b in obs if a < b]
                cov = cov_by_chemo[chemo[i]]
                basis = None if basis_lookup is None else basis_lookup.get((arm, int(seed), int(folds[i])))
                widest = _support_widest(pairs)
                for rep in range(repeats):
                    draws = {"random": _support_random(cell_ids[i], pairs, rep, int(seed))}
                    if rep == 0:
                        draws["widest"] = widest
                    for design, (a, b) in draws.items():
                        y_ab = float(Y[i, a] - Y[i, b])
                        adapted = adapt_conditional(C[i], cov, a, b, y_ab, noise_var=noise_var)
                        rescaled = adapt_rescale(C[i], a, b, y_ab)
                        linear = adapt_no_model_linear(a, b, y_ab)
                        shifted = (adapt_basis_shift(C[i], basis, a, b, y_ab) if basis is not None
                                   else np.full(N_LN, np.nan))
                        for p, q in pairs:
                            if (p, q) == (a, b):
                                continue
                            rows.append({"arm": arm, "split_seed": int(seed), "repeat": rep,
                                         "support_design": design, "cell_id": cell_ids[i],
                                         "extractant": ext[i], "chemotype": chemo[i],
                                         "A": LANTHANIDES[p], "B": LANTHANIDES[q],
                                         "dZ": int(q - p), "support_dz": int(b - a),
                                         "y": float(Y[i, p] - Y[i, q]),
                                         "zero_shot": float(C[i, p] - C[i, q]),
                                         "conditional": float(adapted[p] - adapted[q]),
                                         "basis_shift": float(shifted[p] - shifted[q]),
                                         "rescale": float(rescaled[p] - rescaled[q]),
                                         "no_model_linear": float(linear[p] - linear[q])})
    return pd.DataFrame(rows)


def macro_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Extractant-macro MAE of the zero-shot and adapted predictions, per arm and support design."""
    out = []
    for (arm, design), block in table.groupby(["arm", "support_design"], sort=True):
        rec = {"arm": arm, "support_design": design, "n_pairs": int(len(block)),
               "n_extractants": int(block["extractant"].nunique()),
               "n_cells": int(block["cell_id"].nunique())}
        for col in ("zero_shot", "conditional", "basis_shift", "rescale", "no_model_linear"):
            if col not in block.columns or block[col].isna().all():
                continue
            per = (block.assign(err=(block["y"] - block[col]).abs())
                   .groupby(["split_seed", "extractant"])["err"].mean()
                   .groupby(level=0).mean())
            rec[f"macro_mae_{col}"] = float(per.mean())
            rec[f"macro_mae_{col}_seed_sd"] = float(per.std(ddof=1)) if len(per) > 1 else np.nan
        rec["gain"] = rec["macro_mae_zero_shot"] - rec["macro_mae_conditional"]
        out.append(rec)
    return pd.DataFrame(out).sort_values("macro_mae_conditional").reset_index(drop=True)


def adapt_conditional_multi(c_pred: np.ndarray, cov: np.ndarray, supports: list[tuple[int, int]],
                            y_values: list[float], *, noise_var: float = 0.09) -> np.ndarray:
    """Conditional mean of the residual curve given ``k`` measured contrasts.

    Generalises :func:`adapt_conditional` to a design matrix ``D`` whose rows are ``e_a - e_b``:
    ``u_hat = Sigma D' (D Sigma D' + sigma^2 I)^-1 r``.  With ``k = 1`` it reproduces the single-pair
    formula exactly.
    """
    if not supports:
        return c_pred
    D = np.zeros((len(supports), N_LN))
    r = np.empty(len(supports))
    for i, ((a, b), y) in enumerate(zip(supports, y_values)):
        D[i, a], D[i, b] = 1.0, -1.0
        r[i] = y - (c_pred[a] - c_pred[b])
    SD = cov @ D.T                                   # 14 x k
    middle = D @ SD + noise_var * np.eye(len(supports))
    try:
        weights = np.linalg.solve(middle, r)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(middle, r, rcond=None)[0]
    return c_pred + SD @ weights


def greedy_support_design(cov: np.ndarray, pairs: list[tuple[int, int]], k: int,
                          *, noise_var: float = 0.09) -> list[tuple[int, int]]:
    """Choose ``k`` pairs to measure, before any of them is measured.

    At each step the pair chosen is the one that removes most of the posterior variance summed over
    all the cell's pairs, which for a linear-Gaussian update is the largest trace reduction
    ``||Sigma d||^2 / (d' Sigma d + sigma^2)``.  The choice uses only the covariance and which
    metals the cell has, never a measured value, so it is a design a laboratory can follow.
    """
    chosen: list[tuple[int, int]] = []
    current = cov.copy()
    remaining = list(pairs)
    for _ in range(min(k, len(pairs))):
        best, best_gain = None, -np.inf
        for (a, b) in remaining:
            d = np.zeros(N_LN)
            d[a], d[b] = 1.0, -1.0
            sd = current @ d
            denom = float(d @ sd) + noise_var
            gain = float(sd @ sd) / denom if denom > 1e-12 else 0.0
            if gain > best_gain:
                best, best_gain = (a, b), gain
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)
        d = np.zeros(N_LN)
        d[best[0]], d[best[1]] = 1.0, -1.0
        sd = current @ d
        current = current - np.outer(sd, sd) / (float(d @ sd) + noise_var)
    return chosen


def evaluate_support_budget(curve_frames: dict[str, pd.DataFrame], cohort_frame: pd.DataFrame,
                            *, arms: list[str], k_max: int = 3, min_metals: int = 4,
                            shrink: float = 0.25, noise_var: float = 0.09) -> pd.DataFrame:
    """Extractant-macro MAE as a function of how many pairs of a new cell have been measured.

    For each held-out cell the support pairs are picked by :func:`greedy_support_design` from the
    leave-chemotype-out residual covariance, then revealed one at a time; the score at budget ``k``
    is over the pairs that are still unmeasured, so budgets are compared on the pairs a laboratory
    would still be predicting.
    """
    ccols = [f"c__{m}" for m in LANTHANIDES]
    ycols = [f"logD__{m}" for m in LANTHANIDES]
    meta = cohort_frame.set_index("cell_id")
    rows: list[dict] = []
    for arm in arms:
        frame = curve_frames[arm]
        for seed, block in frame.groupby("split_seed", sort=True):
            cell_ids = block["cell_id"].tolist()
            C = block[ccols].to_numpy(dtype=float)
            Y = meta.loc[cell_ids, ycols].to_numpy(dtype=float)
            chemo = meta.loc[cell_ids, "chemotype"].to_numpy()
            ext = meta.loc[cell_ids, "extractant"].to_numpy()
            resid = np.vstack([centred_residual(Y[i], C[i]) for i in range(len(Y))])
            cov_by_chemo = {g: residual_covariance(resid[chemo != g], shrink=shrink)
                            for g in pd.unique(chemo)}
            for i in range(len(Y)):
                obs = np.flatnonzero(~np.isnan(Y[i]))
                if len(obs) < min_metals:
                    continue
                pairs = [(a, b) for a in obs for b in obs if a < b]
                cov = cov_by_chemo[chemo[i]]
                design = greedy_support_design(cov, pairs, k_max, noise_var=noise_var)
                held = [p for p in pairs if p not in set(design)]
                if not held:
                    continue
                for k in range(0, len(design) + 1):
                    sup = design[:k]
                    yv = [float(Y[i, a] - Y[i, b]) for a, b in sup]
                    curve = adapt_conditional_multi(C[i], cov, sup, yv, noise_var=noise_var)
                    err = float(np.mean([abs((Y[i, p] - Y[i, q]) - (curve[p] - curve[q]))
                                         for p, q in held]))
                    rows.append({"arm": arm, "split_seed": int(seed), "k_measured": k,
                                 "cell_id": cell_ids[i], "extractant": ext[i], "chemotype": chemo[i],
                                 "n_metals": int(len(obs)), "n_held_pairs": len(held), "mae": err})
    table = pd.DataFrame(rows)
    summary = (table.groupby(["arm", "k_measured", "split_seed", "extractant"])["mae"].mean()
               .groupby(level=[0, 1, 2]).mean().groupby(level=[0, 1]).agg(["mean", "std"])
               .rename(columns={"mean": "macro_mae", "std": "seed_sd"}).reset_index())
    summary["n_cells"] = table.groupby(["arm", "k_measured"])["cell_id"].nunique().to_numpy()
    return summary
