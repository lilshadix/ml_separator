"""Shared helpers for the models-side tests (not collected: no ``test_`` prefix).

Unit runs use tiny forests (``learners.N_ESTIMATORS`` lowered for the call only) on design BP, split seed 104729,
fold 0 (the giant fold: 76 training cells), so a full ``run_unit`` takes seconds.  Results are cached per process.
Nothing here computes a metric against an outer-test y; tests that need y-based arithmetic replace y by synthetic
values first.
"""
from __future__ import annotations

import contextlib
import copy

import numpy as np

from gen201 import data as D
from gen201 import fitting as FI
from gen201 import folds as FO
from gen201 import learners as L

SEED = 104729
TINY_TREES = 8
TINY_PAIR_TREES = 4
_CACHE: dict = {}


@contextlib.contextmanager
def tiny_trees(n: int = TINY_TREES, n_pair: int = TINY_PAIR_TREES):
    old = (L.N_ESTIMATORS, L.C_PAIR_N_ESTIMATORS)
    L.N_ESTIMATORS, L.C_PAIR_N_ESTIMATORS = int(n), int(n_pair)
    try:
        yield
    finally:
        L.N_ESTIMATORS, L.C_PAIR_N_ESTIMATORS = old


def context(bench) -> FI.UnitContext:
    if "ctx" not in _CACHE:
        _CACHE["ctx"] = FI.make_context(bench)
    return _CACHE["ctx"]


def bp_folds(bench, seed: int = SEED):
    key = ("bp", seed)
    if key not in _CACHE:
        _CACHE[key] = FO.build(bench, seeds=(seed,), designs=("BP",))["BP"]
    return _CACHE[key]


def giant_fold(bench):
    return next(f for f in bp_folds(bench) if D.SC009 in f.held_out_groups)


def small_unit(bench, variant: str = "primary") -> FI.UnitResult:
    key = ("unit", variant)
    if key not in _CACHE:
        with tiny_trees():
            _CACHE[key] = FI.run_unit(bench, "BP", giant_fold(bench), variant=variant, folds_ctx=context(bench))
    return _CACHE[key]


def masked_bench(bench, cells):
    """A copy of ``bench`` whose outer-test labels are masked (LT-11): ``bench.Y`` and ``bench.coef`` rows and the
    frame columns ``logD__*``, ``nrep__*``, ``repsd__*``, ``n_replicated_metals``, ``replicate_sd_median`` are NaN for
    ``cells``; ``n_metals`` and ``bench.n_obs`` keep their unmasked values."""
    cells = np.asarray(cells, dtype=int)
    b = copy.copy(bench)
    b.Y = np.array(bench.Y, dtype=float, copy=True)
    b.Y[cells] = np.nan
    b.coef = np.array(bench.coef, dtype=float, copy=True)
    b.coef[cells] = np.nan
    f = bench.frame.copy()
    cols = [c for c in f.columns if c.startswith(("logD__", "nrep__", "repsd__"))]
    cols += [c for c in ("n_replicated_metals", "replicate_sd_median") if c in f.columns]
    for c in cols:
        v = f[c].to_numpy(dtype=float, copy=True)
        v[cells] = np.nan
        f[c] = v
    b.frame = f
    b.n_obs = np.array(bench.n_obs, copy=True)
    return b


def synthetic_bench(bench, seed: int = 7):
    """A copy of ``bench`` whose every label is synthetic: ``Y`` (and the frame's ``logD__*``) drawn N(0, 1) on the
    observed entries (NaN pattern kept), ``coef`` drawn N(0, 1), ``repsd__*`` drawn U(0, 1) where ``nrep__ >= 2``.
    Features, identity columns, NaN masks and replicate counts are the real, label-free ones.  Used to exercise
    the sealed scripts' wiring without computing any statistic of the real targets."""
    rng = np.random.default_rng(seed)
    b = copy.copy(bench)
    obs = ~np.isnan(np.asarray(bench.Y, float))
    Y = np.where(obs, rng.normal(size=obs.shape), np.nan)
    b.Y = Y
    b.coef = rng.normal(size=np.asarray(bench.coef).shape)
    f = bench.frame.copy()
    for i, m in enumerate(D.LANTHANIDES):
        f[f"logD__{m}"] = Y[:, i]
        nrep = f[f"nrep__{m}"].to_numpy(dtype=float)
        f[f"repsd__{m}"] = np.where(obs[:, i] & (nrep >= 2), rng.uniform(0.0, 1.0, size=len(f)), np.nan)
    for c in ("replicate_sd_median",):
        if c in f.columns:
            f[c] = np.nan
    b.frame = f
    return b


def strip_volatile(sel: dict) -> dict:
    """A selection record without its timing block (the only field that may differ between identical runs)."""
    out = {k: v for k, v in sel.items() if k != "timing"}
    return out


def r2_within_cell(cells, y, p) -> float:
    """MET-08 on synthetic y (tests only)."""
    cells, y, p = np.asarray(cells), np.asarray(y, float), np.asarray(p, float)
    _, code = np.unique(cells, return_inverse=True)
    n = np.bincount(code).astype(float)
    yt = y - (np.bincount(code, weights=y) / n)[code]
    pt = p - (np.bincount(code, weights=p) / n)[code]
    return float(1.0 - np.sum((yt - pt) ** 2) / np.sum(yt ** 2))
