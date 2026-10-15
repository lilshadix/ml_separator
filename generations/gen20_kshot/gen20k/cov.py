"""Residual covariance of the lanthanide curve from inner cross-fits of the masked training fold.

Gen15 estimated the covariance from the out-of-fold residuals of *other test cells* of the same
split seed (second-order contamination by models that had seen the corrected cell, and no
publication mask unless opted in), and the estimate was indefinite in every one of 430 cases
(gen16 L5).  Here the covariance is estimated **inside the outer training fold only**: the
training cells are dealt into four inner chemotype folds with gen13's inner seed, the arm is fitted
on each inner-training part (with the design's publication/copy-group mask re-applied inside), the
inner-validation cells get out-of-fold predicted curves, their centred residual curves are pooled,
shrunk toward the diagonal (gen15's estimator) and **projected to positive semidefinite**.  The
minimum eigenvalue before projection is recorded per fold.

The measurement-noise model ``R`` for a set of support pairs accounts for pairs that share a log D
reading: with per-metal replicate variance ``s2`` (from training-fold replicates only),
``Var y(a,b) = 2 s2`` and ``Cov(y(a,b), y(a,c)) = s2``, i.e. ``R = s2 * D D^T``.  Gen15's
``NOISE_VAR * I`` is kept as a named alternative for replication.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bootstrap  # noqa: F401
from .data import N_LN
from .splits import N_INNER, copy_groups_of, inner_seed
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen14.dirbench import feature_sets  # noqa: E402
from gen15.fewshot import centred_residual, residual_covariance, smooth_covariance  # noqa: E402
from gen15.valuebench import Ctx, MIN_METALS  # noqa: E402
from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402

GEN15_NOISE_VAR = 0.09
MIN_RESIDUAL_ROWS = 10


@dataclass
class CovRecord:
    cov: np.ndarray
    n_rows: int
    min_eig_before: float
    shrink: float
    noise_s2: float
    source: str
    cells: tuple = ()          # cell indices whose residual curves entered the estimate


def assert_cov_provenance(rec: "CovRecord", allowed) -> None:
    """Every residual curve must come from a cell in ``allowed`` (the training cells)."""
    bad = sorted(set(int(c) for c in rec.cells) - set(int(c) for c in allowed))
    if bad:
        raise AssertionError(f"covariance built from cells outside the allowed set: {bad[:5]}")


def make_ctx(bench, fold, train, test, *, fs=None, X=None, rich=None) -> Ctx:
    fs = fs if fs is not None else feature_sets(bench)
    X = X if X is not None else bench.matrix(LEAN_BLOCKS)
    rich = rich if rich is not None else bench.frame.n_metals.to_numpy() >= MIN_METALS
    return Ctx(bench=bench, design=fold.design, seed=fold.seed, fold=fold.fold, train=np.asarray(train),
               test=np.asarray(test), w=cell_weights(bench.groups[train], bench.n_obs[train]),
               model_seed=fold.model_seed, fs=fs, X=X, rich=rich)


def inner_mask(bench, design: str, inner_train: np.ndarray, inner_val: np.ndarray) -> np.ndarray:
    """Re-apply the design's mask inside the training fold: drop inner-training cells whose
    publication (BP) or copy group (BPG) appears among the inner-validation cells."""
    if design not in ("BP", "BPG"):
        return inner_train
    key = bench.frame["publication_id"].astype(str).to_numpy() if design == "BP" else copy_groups_of(bench.frame)
    held = set(key[inner_val])
    keep = np.array([key[i] not in held for i in inner_train], dtype=bool)
    return inner_train[keep]


def inner_oof_curves(bench, fold, fit_curves, *, fs=None, X=None, rich=None, min_train: int = 20) -> dict[int, np.ndarray]:
    """Out-of-fold predicted curves for the training cells of ``fold`` from ``N_INNER`` inner
    chemotype folds; ``fit_curves(ctx) -> curves (n_test, 14)`` fits on ``ctx.train`` and predicts
    ``ctx.test`` (see ``arms.as_fit_curves``)."""
    tr = np.asarray(fold.train_index)
    groups = bench.groups[tr].astype(str)
    out: dict[int, np.ndarray] = {}
    for tr_rel, va_rel in seeded_group_kfold(groups, N_INNER, inner_seed(fold.seed, fold.fold)):
        itr, iva = inner_mask(bench, fold.design, tr[tr_rel], tr[va_rel]), tr[va_rel]
        if len(itr) < min_train or len(iva) == 0:
            continue
        ctx = make_ctx(bench, fold, itr, iva, fs=fs, X=X, rich=rich)
        curves = np.asarray(fit_curves(ctx), dtype=float).reshape(len(iva), N_LN)
        for j, ci in enumerate(iva):
            out[int(ci)] = curves[j]
    return out


def psd_project(cov: np.ndarray, floor_frac: float = 1e-6) -> tuple[np.ndarray, float]:
    w, V = np.linalg.eigh((cov + cov.T) / 2.0)
    min_eig = float(w.min())
    floor = floor_frac * max(float(w.max()), 1e-12)
    w = np.maximum(w, floor)
    return (V * w) @ V.T, min_eig


REPLICATE_SD_CAP = 0.5   # replicate disagreements above this are decade-level double entries, not scatter


def training_noise_s2(bench, train_index, *, cap: float = REPLICATE_SD_CAP) -> float:
    """Per-metal replicate variance from the training cells' ``repsd__`` columns: the median of the
    replicate sd over readings whose disagreement is below ``cap`` (the review found 12 % of the 216
    replicated readings differ by an integer number of decades, the exponent-corruption signature,
    which would otherwise put the implied SF noise above the target's own spread), squared."""
    cols = [c for c in bench.frame.columns if c.startswith("repsd__")]
    v = bench.frame[cols].to_numpy(dtype=float)[np.asarray(train_index)]
    v = v[np.isfinite(v)]
    v = v[v <= cap]
    s = float(np.median(v)) if v.size else np.sqrt(GEN15_NOISE_VAR / 2.0)
    return float(s * s)


def covariance_from_residuals(rows: np.ndarray, *, shrink: float = 0.25, basis_row=None,
                              source: str = "inner_oof", noise_s2: float = 0.045, cells=None) -> CovRecord:
    if len(rows) < MIN_RESIDUAL_ROWS:
        cov = smooth_covariance(scale=0.25, length=1.0, nugget=0.05, basis_row=basis_row)
        return CovRecord(cov=cov, n_rows=int(len(rows)), min_eig_before=float(np.linalg.eigvalsh(cov).min()),
                         shrink=shrink, noise_s2=noise_s2, source="smooth_fallback", cells=tuple(cells or ()))
    emp = residual_covariance(np.asarray(rows, dtype=float), shrink=shrink)
    cov, min_eig = psd_project(emp)
    return CovRecord(cov=cov, n_rows=int(len(rows)), min_eig_before=min_eig, shrink=shrink,
                     noise_s2=noise_s2, source=source, cells=tuple(cells or ()))


def fold_covariance(bench, fold, fit_curves, *, shrink: float = 0.25, min_metals: int = 3, **kw) -> CovRecord:
    """The covariance one arm uses for every held-out cell of ``fold``: inner OOF residuals of the
    masked training fold, PSD-projected; the noise variance from training replicates."""
    curves = inner_oof_curves(bench, fold, fit_curves, **kw)
    rows, cells = [], []
    for ci, cv in curves.items():
        if bench.n_obs[ci] >= min_metals:
            rows.append(centred_residual(bench.Y[ci], cv))
            cells.append(int(ci))
    rec = covariance_from_residuals(np.array(rows) if rows else np.zeros((0, N_LN)), shrink=shrink,
                                    basis_row=bench.basis[0], noise_s2=training_noise_s2(bench, fold.train_index), cells=cells)
    assert_cov_provenance(rec, fold.train_index)
    return rec


def noise_matrix(pairs, s2: float, kind: str = "shared") -> np.ndarray:
    """``R`` for the support pairs: ``shared`` = s2 * D D^T (pairs sharing a metal are correlated),
    ``gen15`` = 2*s2 on the diagonal only."""
    D = np.zeros((len(pairs), N_LN))
    for i, (a, b) in enumerate(pairs):
        D[i, a], D[i, b] = 1.0, -1.0
    if kind == "gen15":
        return 2.0 * s2 * np.eye(len(pairs))
    if kind != "shared":
        raise ValueError(kind)
    return s2 * (D @ D.T)
