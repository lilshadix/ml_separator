"""Gen15 measurement-assisted mode: what one, two or three measured separation factors are worth.

Stage 2 established the largest lever in the programme: a cell's pair residuals are exactly additive
in one per-cell metal curve, so one measured ``log SF`` constrains all the others, and the best linear
use of it is the BLUP with a leave-chemotype-out residual covariance.  It also established the
uncomfortable part -- a straight line drawn through the measured pair with no model at all scores
about as well, so after one measurement the chemistry was worth roughly 0.02.

This module re-runs that comparison inside the gen15 bench so it can be scored under all five designs
with the full metric panel, and adds the three things the locked study did not try:

* **a fixed scoring set.**  The support pairs are removed from scoring for *every* k, so the k = 0,
  1, 2, 3 numbers are computed on byte-identical pairs and the ladder is a like-for-like comparison
  rather than an easier question at each rung.
* **a direction-conditional curvature after the measurement.**  A measured pair identifies the
  direction of selectivity far more reliably than the classifier does, and the corpus mean curvature
  differs by direction (-0.009 for heavy-selective cells, -0.136 for light-selective ones).  The
  no-model line cannot use that; a model can.
* **which pair to measure.**  Widest available ``dZ`` is the locked heuristic; the D-optimal choice
  under the residual covariance is the estimator-aware one, and a laboratory genuinely gets to choose.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights
from gen13sep.metals import ATOMIC_NUMBER, LANTHANIDES
from gen13sep.splits import all_folds
from gen14.dirbench import feature_sets

from .valuebench import Ctx, MIN_METALS

N_LN = len(LANTHANIDES)
Z_VEC = np.array([ATOMIC_NUMBER[m] for m in LANTHANIDES])
#: variance of one measured log SF: two independent log D readings at the cohort's median
#: within-replicate sd (0.212), the locked stage-2 value
NOISE_VAR = 0.09


# --------------------------------------------------------------------------------------
# residual covariance
# --------------------------------------------------------------------------------------
def centred_residual(y_row: np.ndarray, c_pred: np.ndarray) -> np.ndarray:
    """Observed minus predicted curve on the metals a cell measured, centred over those metals."""
    obs = ~np.isnan(y_row)
    out = np.full(N_LN, np.nan)
    d = y_row[obs] - c_pred[obs]
    out[obs] = d - d.mean()
    return out


def residual_covariance(residuals: np.ndarray, *, shrink: float = 0.25, ridge: float = 1e-3,
                        min_pairs: int = 20) -> np.ndarray:
    """Pairwise-complete covariance of residual curves, shrunk toward its own diagonal."""
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


def smooth_covariance(scale: float = 1.0, length: float = 1.0, nugget: float = 0.05,
                      basis_row: np.ndarray | None = None) -> np.ndarray:
    """A parametric alternative: a squared-exponential kernel in the standardised ionic radius.

    The empirical covariance is estimated from a few dozen residual curves over 14 metals, i.e. 105
    free parameters from little data.  The residual curve is known to be smooth along the series
    (lag-1 correlation +0.92), so a two-parameter kernel is the better-conditioned prior, and it is
    also defined for metal pairs the corpus never measured together.
    """
    from gen13sep.metals import physics_basis
    r = physics_basis()["radius"] if basis_row is None else np.asarray(basis_row, dtype=float)
    d2 = (r[:, None] - r[None, :]) ** 2
    K = scale * np.exp(-0.5 * d2 / max(length, 1e-6) ** 2)
    return K + nugget * scale * np.eye(N_LN)


# --------------------------------------------------------------------------------------
# the correction
# --------------------------------------------------------------------------------------
def blup(c_pred: np.ndarray, cov: np.ndarray, supports: Sequence[tuple[int, int, float]],
         *, noise_var: float = NOISE_VAR) -> np.ndarray:
    """Conditional mean of the curve given ``supports`` = [(a, b, observed log SF), ...]."""
    if not supports:
        return c_pred
    D = np.zeros((len(supports), N_LN))
    r = np.empty(len(supports))
    for i, (a, b, y) in enumerate(supports):
        D[i, a], D[i, b] = 1.0, -1.0
        r[i] = y - (c_pred[a] - c_pred[b])
    S = D @ cov @ D.T + noise_var * np.eye(len(supports))
    try:
        w = np.linalg.solve(S, r)
    except np.linalg.LinAlgError:
        return c_pred
    return c_pred + cov @ D.T @ w


def blup_posterior(cov: np.ndarray, supports: Sequence[tuple[int, int, float]],
                   *, noise_var: float = NOISE_VAR) -> np.ndarray:
    """Posterior covariance of the curve after conditioning on ``supports``.

    The BLUP carries its own uncertainty and nothing in the programme has ever used it.  A predicted
    ``log SF`` for the pair (a, b) has predictive variance ``d' Sigma_post d + noise_var`` with
    ``d = e_a - e_b``, which is the interval a laboratory would need before acting on the number.
    """
    if not supports:
        return cov.copy()
    D = np.zeros((len(supports), N_LN))
    for i, (a, b, _) in enumerate(supports):
        D[i, a], D[i, b] = 1.0, -1.0
    S = D @ cov @ D.T + noise_var * np.eye(len(supports))
    try:
        G = np.linalg.solve(S, D @ cov)
    except np.linalg.LinAlgError:
        return cov.copy()
    return cov - cov @ D.T @ G


def pair_sd(post: np.ndarray, a: int, b: int, *, noise_var: float = NOISE_VAR) -> float:
    """Predictive standard deviation of one pair's log SF under the posterior."""
    d = np.zeros(N_LN)
    d[a], d[b] = 1.0, -1.0
    return float(np.sqrt(max(d @ post @ d + noise_var, 1e-12)))


def naive_line(basis: np.ndarray, supports: Sequence[tuple[int, int, float]]) -> np.ndarray:
    """The honest reference: least squares on the radius ramp alone, no ligand information at all."""
    if not supports:
        return np.zeros(N_LN)
    r = basis[0]
    x = np.array([r[a] - r[b] for a, b, _ in supports])
    y = np.array([v for _, _, v in supports])
    denom = float(x @ x)
    a_hat = float(x @ y / denom) if denom > 1e-12 else 0.0
    return a_hat * r


def line_plus_curvature(basis: np.ndarray, supports: Sequence[tuple[int, int, float]],
                        b_heavy: float, b_light: float) -> np.ndarray:
    """The naive line, plus the training fold's curvature for the direction the measurement reveals.

    This is the cheapest thing a model can add on top of a measurement, and it is the one thing the
    no-model line structurally cannot do: it needs a training corpus to know that heavy-selective and
    light-selective systems curve differently.
    """
    if not supports:
        return np.zeros(N_LN)
    r = basis[0]
    x = np.array([r[a] - r[b] for a, b, _ in supports])
    y = np.array([v for _, _, v in supports])
    denom = float(x @ x)
    a_hat = float(x @ y / denom) if denom > 1e-12 else 0.0
    return a_hat * r + (b_heavy if a_hat < 0 else b_light) * basis[1]


# --------------------------------------------------------------------------------------
# choosing which pair to measure
# --------------------------------------------------------------------------------------
def pick_support(pairs: list[tuple[int, int]], k: int, how: str, cov: np.ndarray,
                 *, rng: np.random.Generator | None = None,
                 noise_var: float = NOISE_VAR) -> list[tuple[int, int]]:
    """Choose ``k`` pairs to measure.  ``widest`` is the locked heuristic; ``dopt`` is greedy
    D-optimal under the residual covariance -- it maximises the information the measurements carry
    about the whole curve rather than the span of one contrast."""
    if k <= 0 or not pairs:
        return []
    if how == "random":
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=min(k, len(pairs)), replace=False)
        return [pairs[i] for i in idx]
    if how == "widest":
        return sorted(pairs, key=lambda p: (-(Z_VEC[p[1]] - Z_VEC[p[0]]), p[0]))[:k]
    if how != "dopt":
        raise ValueError(how)
    chosen: list[tuple[int, int]] = []
    remaining = list(pairs)
    post = cov.copy()
    for _ in range(min(k, len(pairs))):
        best, best_gain = None, -np.inf
        for p in remaining:
            d = np.zeros(N_LN)
            d[p[0]], d[p[1]] = 1.0, -1.0
            v = post @ d
            den = float(d @ v) + noise_var
            gain = float(v @ v) / den if den > 1e-12 else -np.inf
            if gain > best_gain:
                best, best_gain = p, gain
        if best is None:
            break
        d = np.zeros(N_LN)
        d[best[0]], d[best[1]] = 1.0, -1.0
        v = post @ d
        post = post - np.outer(v, v) / (float(d @ v) + noise_var)
        chosen.append(best)
        remaining.remove(best)
    return chosen


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------
@dataclass
class FewShotResult:
    pairs: pd.DataFrame       # long pair table with one column per (arm, k) mode
    modes: list[str]
    seconds: float


def evaluate(bench, arms: dict[str, Callable[[Ctx], np.ndarray]], design: str, *,
             ks: Sequence[int] = (0, 1, 2, 3), how: str = "widest",
             hows: Sequence[str] | None = None,
             cov_kind: str = "empirical", noise_var: float = NOISE_VAR,
             mask_publication: bool = False, per_cell_noise: bool = False,
             min_metals: int = 3, seeds: Sequence[int] | None = None,
             verbose: bool = True) -> FewShotResult:
    """Score every arm at every measurement budget on a FIXED pair set.

    The support pairs of the largest budget are excluded from scoring at *every* budget, so k = 0 is
    asked exactly the question k = 3 is asked.  The residual covariance is estimated leave-chemotype-
    out from the held-out residuals of the other chemotypes of the same split seed, so no cell and no
    member of its chemotype contributes to the correction applied to it.

    ``hows`` compares several support-selection strategies in one pass.  It must be used whenever
    strategies are compared, because each one removes *different* pairs from scoring: measuring the
    widest ``dZ`` pair takes the hardest pair out of the test set and a random pair does not, which
    on its own moves the k = 0 score by 0.02.  With ``hows`` the union of every strategy's support
    pairs is excluded for every strategy, so the comparison is on byte-identical pairs.  Modes are
    then named ``arm@<how>k<k>``.

    ``mask_publication`` closes a hole the locked stage-2 protocol leaves open.  Design BP removes a
    held-out cell's publication from *training*, but the covariance is built from other chemotypes'
    held-out residuals, and those may come from the very publication BP masked.  With the flag set,
    a cell's covariance is estimated only from residual curves whose publication differs from its
    own as well, which is the construction BP's logic actually requires.

    ``per_cell_noise`` replaces the fixed measurement variance with the cell's own replicate spread
    (``repsd__<metal>``), so a cell measured once is trusted less than one measured in triplicate.
    """
    t0 = time.time()
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    kmax = max(ks)
    folds = all_folds(bench.frame, design=design) if seeds is None \
        else all_folds(bench.frame, design=design, seeds=seeds)

    # ---- pass 1: predicted curves for every held-out cell, per arm ----
    curves: dict[str, dict[int, np.ndarray]] = {a: {} for a in arms}
    meta: dict[int, dict] = {}
    bstats: dict[tuple[int, int], tuple[float, float]] = {}
    for f in folds:
        ctx = Ctx(bench=bench, design=design, seed=f.seed, fold=f.fold, train=f.train_index,
                  test=f.test_index, w=cell_weights(bench.groups[f.train_index],
                                                    bench.n_obs[f.train_index]),
                  model_seed=f.model_seed, fs=fs, X=X, rich=rich)
        rtr, rw = ctx.rich_train(), ctx.rich_weights()
        a_tr, b_tr = ctx.amp[rtr], ctx.cur[rtr]
        hv, lt = a_tr < 0, a_tr > 0
        bstats[(f.seed, f.fold)] = (
            float(np.average(b_tr[hv], weights=rw[hv])) if hv.sum() >= 10 else ctx.train_mean_curvature(),
            float(np.average(b_tr[lt], weights=rw[lt])) if lt.sum() >= 10 else ctx.train_mean_curvature())
        for name, arm in arms.items():
            coef = np.asarray(arm(ctx), dtype=float).reshape(len(f.test_index), bench.basis.shape[0])
            cv = coef @ bench.basis
            for j, ci in enumerate(f.test_index):
                curves[name].setdefault((f.seed, f.fold), {})[ci] = cv[j]
        for ci in f.test_index:
            meta[(f.seed, f.fold, ci)] = {"seed": f.seed, "fold": f.fold, "cell": ci}

    # ---- pass 2: leave-chemotype-out residual covariance, per (seed, reference arm) ----
    ref = next(iter(arms))
    pub = bench.frame.publication_id.astype(str).to_numpy()
    resid_by_seed: dict[int, dict[int, np.ndarray]] = {}
    for (seed, fold), d in curves[ref].items():
        for ci, cv in d.items():
            resid_by_seed.setdefault(seed, {})[ci] = centred_residual(bench.Y[ci], cv)
    fallback_cov = smooth_covariance(scale=0.25, length=1.0, nugget=0.05, basis_row=bench.basis[0])

    def cov_for(seed: int, chemotype: str, publication: str) -> np.ndarray:
        if cov_kind == "smooth":
            return fallback_cov
        rows = [r for ci, r in resid_by_seed.get(seed, {}).items()
                if bench.groups[ci] != chemotype
                and not (mask_publication and pub[ci] == publication)]
        if len(rows) < 10:
            return fallback_cov
        emp = residual_covariance(np.array(rows))
        if cov_kind == "blend":
            s = float(np.trace(emp) / np.trace(fallback_cov))
            return 0.5 * emp + 0.5 * s * fallback_cov
        return emp

    cov_cache: dict[tuple[int, str, str], np.ndarray] = {}
    # per-cell measurement variance on one log SF: two log D readings at the cell's replicate spread
    repsd = bench.frame[[f"repsd__{m}" for m in LANTHANIDES]].to_numpy(dtype=float)
    med_repsd = float(np.nanmedian(repsd)) if np.isfinite(repsd).any() else 0.212

    def noise_for(ci: int) -> float:
        if not per_cell_noise:
            return noise_var
        row = repsd[ci]
        s = float(np.nanmedian(row)) if np.isfinite(row).any() else med_repsd
        return float(np.clip(2.0 * s ** 2, 0.02, 0.5))

    # ---- pass 3: correct and score ----
    modes: list[str] = []
    rows = []
    for (seed, fold), d in curves[ref].items():
        for ci in d:
            y = bench.Y[ci]
            obs = np.flatnonzero(~np.isnan(y))
            if len(obs) < min_metals + 1:
                continue
            pairs = [(int(a), int(b)) for i, a in enumerate(obs) for b in obs[i + 1:]]
            if len(pairs) <= kmax:
                continue
            key = (seed, str(bench.groups[ci]), pub[ci] if mask_publication else "")
            if key not in cov_cache:
                cov_cache[key] = cov_for(seed, str(bench.groups[ci]), pub[ci])
            cov = cov_cache[key]
            nv = noise_for(ci)
            rng = np.random.default_rng(abs(hash((int(ci), int(seed)))) % (2 ** 32))
            strategies = list(hows) if hows else [how]
            orders = {h: pick_support(pairs, kmax, h, cov, rng=rng, noise_var=nv) for h in strategies}
            excluded = {p for order in orders.values() for p in order}
            held = [p for p in pairs if p not in excluded]
            if not held:
                continue
            bh, bl = bstats[(seed, fold)]
            preds: dict[str, np.ndarray] = {}
            posts: dict[tuple[str, int], np.ndarray] = {}
            for h in strategies:
                tag = "" if not hows else h
                for k in ks:
                    sup = [(a, b, float(y[a] - y[b])) for a, b in orders[h][:k]]
                    posts[(h, k)] = blup_posterior(cov, sup, noise_var=nv)
                    for name in arms:
                        preds[f"{name}@{tag}k{k}"] = blup(curves[name][(seed, fold)][ci], cov, sup,
                                                          noise_var=nv)
                    if k > 0:
                        preds[f"NAIVE_LINE@{tag}k{k}"] = naive_line(bench.basis, sup)
                        if not hows:
                            preds[f"LINE_PLUS_CURV@k{k}"] = line_plus_curvature(
                                bench.basis, sup, bh, bl)
            if not modes:
                modes = list(preds)
            for a, b in held:
                rec = {"split_seed": seed, "fold": fold, "cell_id": bench.frame.cell_id.iat[ci],
                       "extractant": bench.frame.extractant.iat[ci],
                       "chemotype": bench.frame.chemotype.iat[ci],
                       "n_metals": int(bench.frame.n_metals.iat[ci]),
                       "A": LANTHANIDES[a], "B": LANTHANIDES[b],
                       "dZ": int(Z_VEC[b] - Z_VEC[a]), "y": float(y[a] - y[b])}
                for m, cv in preds.items():
                    rec[m] = float(cv[a] - cv[b])
                for (h, k), po in posts.items():
                    tag = "" if not hows else h
                    rec[f"sd@{tag}k{k}"] = pair_sd(po, a, b, noise_var=nv)
                rows.append(rec)
    table = pd.DataFrame(rows)
    if verbose:
        print(f"   [{design}] few-shot {len(table)} scored pairs, {len(modes)} modes, "
              f"{time.time() - t0:.0f}s", flush=True)
    return FewShotResult(pairs=table, modes=modes, seconds=time.time() - t0)
