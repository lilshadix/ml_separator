"""gen20.1 row learners (IMPLEMENTATION_PLAN.md §2 "Features, leak guard, weights", §3; PRE_REGISTRATION.md §4.1).

* ``RowForest`` re-implements gen13's row model without the centring of its output (gen13's ``DirectRowArm``
  centres, gen13sep/models.py:269): gen13 ``tree_pipeline`` (SimpleImputer(median, keep_empty_features=True) ->
  ExtraTreesRegressor(400 trees, random_state = fold.model_seed); the imputer is fitted on the fit's training rows)
  on the cell feature block (LEAN209 by default) plus METAL_BASIC2 = (Z, CN8 Shannon radius), one training row per
  observed (cell, metal) of the fit's training cells, built exactly like ``DirectRowArm._rows`` (cell-major,
  metal-minor), row weight = chemotype_balanced_weights over the fit's training cells / n_obs(c), renormalised to
  mean 1 (models.py:257-261).  ``target='raw'`` fits uncentred log D (A, BD_COND, BD_LEVEL, A_PRIME);
  ``target='centred'`` fits y - mean over the cell's observed metals (C).  ``predict`` returns the 14-metal matrix:
  raw uncentred, centred as fitted (never re-centred).  ``metal_cols=False`` drops the metal columns (BD_LEVEL), so
  the prediction is constant within a cell by construction.
* Grid K1..K4 = (min_samples_leaf, max_features) (8, 0.3), (8, 0.5), (2, 0.3), (2, 0.5), simplicity order; ties
  |delta| < 1e-12 go to the lowest K (``select_config``).
* ``CPair`` wraps gen13 ``AntisymmetricPairArm`` (200 trees, max_features 0.3, min_samples_leaf 2, every training
  within-cell pair plus its reversal, no subsampling; prediction (f(a,b) - f(b,a)) / 2) on LEAN209.

Every fit takes the full-cohort arrays plus the explicit training cell indices (``cells``) so that leakage tests can
spy on exactly which cells a fit saw (LT-11, LT-12, LT-18).  ``N_ESTIMATORS`` / ``C_PAIR_N_ESTIMATORS`` are the
registered tree counts; tests may lower them, and ``fitting.run_design`` refuses to write under
generations/gen20_1/predictions unless they equal 400 / 200.

Warning filters: scikit-learn 1.9's ``_FuncWrapper`` runs ``warnings.catch_warnings()`` + ``resetwarnings()`` inside
every joblib worker thread; on CPython 3.14 without context-aware warnings that is not thread-safe, and two
overlapping forest tasks can leave the process-wide ``warnings.filters`` empty, after which every later task of every
fit and predict prints "`sklearn.utils.parallel.delayed` should be used ..." (numbers unaffected, logs flooded).
Every forest fit / predict here runs inside ``guarded_filters()``: an emptied filter list is restored before the
call, and any change a call leaves behind is undone after it (``FILTER_RESTORES`` counts the restores; fitting
records the per-unit count in the selection record's timing block).
"""
from __future__ import annotations

import contextlib
import hashlib
import warnings

import numpy as np

from . import bootstrap  # noqa: F401
import gen13sep.models as M13  # noqa: E402
from gen13sep.basis import centre_rows  # noqa: E402

from . import data as D  # noqa: E402

N_LN = D.N_LN
CONFIG_ORDER: tuple[str, ...] = ("K1", "K2", "K3", "K4")
CONFIGS: dict[str, dict] = {
    "K1": {"min_samples_leaf": 8, "max_features": 0.3},
    "K2": {"min_samples_leaf": 8, "max_features": 0.5},
    "K3": {"min_samples_leaf": 2, "max_features": 0.3},
    "K4": {"min_samples_leaf": 2, "max_features": 0.5},
}
#: registered tree counts (tests may lower them; production writes refuse anything else)
N_ESTIMATORS = 400
REGISTERED_N_ESTIMATORS = 400
C_PAIR_N_ESTIMATORS = 200
REGISTERED_C_PAIR_N_ESTIMATORS = 200
C_PAIR_PARAMS = {"max_features": 0.3, "min_samples_leaf": 2}
TIE_TOL = 1e-12
TARGETS = ("raw", "centred")


class LeakageError(AssertionError):
    """A fit or a prediction would see a cell it must not see (LT-10, LT-30)."""


# --------------------------------------------------------------------------------------
# warning filters around the threaded forest calls (see the module docstring)
# --------------------------------------------------------------------------------------
SKLEARN_DELAYED_MESSAGE = r"`sklearn\.utils\.parallel\.delayed` should be used"
warnings.filterwarnings("ignore", message=SKLEARN_DELAYED_MESSAGE, category=UserWarning)
_FILTERS_AT_IMPORT: list = list(warnings.filters)
#: number of times ``guarded_filters`` had to put the process-wide warning filters back
FILTER_RESTORES = 0


def _set_filters(target) -> None:
    """Put ``target`` back into the process-wide filter list, in place and exactly (``warnings.filters`` is the
    module global every non-context-aware ``warn`` reads)."""
    global FILTER_RESTORES
    warnings.filters[:] = list(target)
    FILTER_RESTORES += 1


@contextlib.contextmanager
def guarded_filters():
    """Run one threaded scikit-learn call with intact warning filters: an empty filter list (the trace of the
    worker-thread race) is replaced by the import-time filters before the call, and whatever the call leaves
    changed is put back to the pre-call state afterwards (balanced ``catch_warnings`` blocks never change the
    filters, so any change across the call is the race)."""
    if not warnings.filters and _FILTERS_AT_IMPORT:
        _set_filters(_FILTERS_AT_IMPORT)
    before = list(warnings.filters)
    try:
        yield
    finally:
        if warnings.filters != before:
            _set_filters(before)


def registered_tree_counts() -> bool:
    return N_ESTIMATORS == REGISTERED_N_ESTIMATORS and C_PAIR_N_ESTIMATORS == REGISTERED_C_PAIR_N_ESTIMATORS


def config_params(config: str) -> dict:
    if config not in CONFIGS:
        raise KeyError(f"unknown configuration {config!r}; the sealed grid is {CONFIG_ORDER}")
    return dict(CONFIGS[config])


def select_config(scores: dict, order=CONFIG_ORDER, tol: float = TIE_TOL) -> tuple[str, str]:
    """Argmin over ``order`` with ties |delta| < tol resolved to the earliest (simplest) entry; non-finite scores
    never win; if no score is finite the first entry is returned.  Returns (choice, reason)."""
    fin = {k: float(scores[k]) for k in order if k in scores and scores[k] is not None and np.isfinite(scores[k])}
    if not fin:
        return order[0], "no finite inner score (all inner folds skipped or empty): first in order"
    m = min(fin.values())
    for k in order:
        if k in fin and fin[k] - m < tol:
            return k, "argmin" if fin[k] == m else f"tie within {tol:g} of the minimum: earliest in order"
    raise AssertionError("unreachable")  # pragma: no cover


def _digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def row_layout(Yc: np.ndarray):
    """(local cell, metal, keep) of ``DirectRowArm._rows``: cell-major, metal-minor over all 14 metals, then the
    observed entries kept."""
    n = len(Yc)
    lci = np.repeat(np.arange(n), N_LN)
    mi = np.tile(np.arange(N_LN), n)
    keep = ~np.isnan(Yc[lci, mi])
    return lci[keep], mi[keep]


def row_weights(groups_cells, Yc: np.ndarray, lci: np.ndarray) -> np.ndarray:
    """w_c / n_obs(c), w_c = chemotype_balanced_weights of the fit's training cells, renormalised to mean 1 (the
    exact operation order of gen13sep/models.py:257-261, so LT-13 holds bit for bit)."""
    n_obs = (~np.isnan(Yc)).sum(axis=1).astype(float)
    w = M13.chemotype_balanced_weights(groups_cells)[lci] / n_obs[lci]
    return w * len(w) / w.sum()


def _single_thread_predict(pipeline) -> None:
    """Predict with n_jobs = 1: the fit is seed-deterministic for any n_jobs, but a threaded predict sums the trees
    in scheduling order, so the last bits of a prediction could differ between runs; sequential accumulation makes
    every saved prediction reproducible bit for bit (LT-05)."""
    pipeline.set_params(extratreesregressor__n_jobs=1)


def _as_cells(cells) -> np.ndarray:
    c = np.asarray(cells, dtype=int).ravel()
    if c.size == 0:
        raise ValueError("empty cell set")
    return c


class RowForest:
    """ExtraTrees row learner on (cell features [+ METAL_BASIC2]) -> raw or centred log D."""

    def __init__(self, target: str = "raw", config: str = "K4", seed: int = 0, *, metal_cols: bool = True,
                 n_estimators: int | None = None, name: str | None = None):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        self.target = target
        self.config = config
        self.params = config_params(config)
        self.seed = int(seed)
        self.metal_cols = bool(metal_cols)
        self.n_estimators = int(N_ESTIMATORS if n_estimators is None else n_estimators)
        self.name = name or f"RowForest[{target},{config}{'' if metal_cols else ',nometal'}]"

    # -- features ---------------------------------------------------------------------------
    def _feats(self, Xc: np.ndarray, lci: np.ndarray, mi: np.ndarray) -> np.ndarray:
        if not self.metal_cols:
            return Xc[lci]
        return np.hstack([Xc[lci], D.Z[mi][:, None], D.R[mi][:, None]])

    # -- fit / predict ----------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, groups, cells) -> "RowForest":
        """Fit on every observed (cell, metal) of ``cells`` (indices into the full-cohort arrays ``X``, ``Y``,
        ``groups``); nothing outside ``cells`` is read."""
        cells = _as_cells(cells)
        Xc = np.asarray(X, dtype=np.float64)[cells]
        Yc = np.asarray(Y, dtype=np.float64)[cells]
        T = centre_rows(Yc) if self.target == "centred" else Yc
        lci, mi = row_layout(Yc)
        y = T[lci, mi]
        if not np.isfinite(y).all():
            raise ValueError("non-finite training target")
        w = row_weights(np.asarray(groups)[cells], Yc, lci)
        feats = self._feats(Xc, lci, mi)
        self.model_ = M13.tree_pipeline(self.seed, n_estimators=self.n_estimators, **self.params)
        with guarded_filters():
            self.model_.fit(feats, y, extratreesregressor__sample_weight=w)
        self.train_cells_ = cells.copy()
        self.train_ci_ = cells[lci]
        self.train_mi_ = mi
        self.n_rows_ = int(len(y))
        self.fit_record_ = {"name": self.name, "target": self.target, "config": self.config, "seed": self.seed,
                            "metal_cols": self.metal_cols, "n_estimators": self.n_estimators,
                            "n_cells": int(len(cells)), "n_rows": int(len(y)), "n_features": int(feats.shape[1]),
                            "X_sha256": _digest(feats), "w_sha256": _digest(w), "y_sha256": _digest(y)}
        return self

    def predict(self, X: np.ndarray, cells) -> np.ndarray:
        """(len(cells), 14) float64: raw log D (raw) or the centred value as fitted (centred)."""
        _single_thread_predict(self.model_)
        cells = _as_cells(cells)
        Xc = np.asarray(X, dtype=np.float64)[cells]
        n = len(cells)
        if not self.metal_cols:
            with guarded_filters():
                p = np.asarray(self.model_.predict(Xc), dtype=np.float64)
            return np.repeat(p[:, None], N_LN, axis=1)
        lci = np.repeat(np.arange(n), N_LN)
        mi = np.tile(np.arange(N_LN), n)
        feats = self._feats(Xc, lci, mi)
        with guarded_filters():
            p = np.asarray(self.model_.predict(feats), dtype=np.float64)
        return p.reshape(n, N_LN)

    def discard(self) -> None:
        """Free the fitted forest (one process at a time on 8 GB, fits discarded after use)."""
        self.model_ = None


def bd_level(config: str, seed: int, **kw) -> RowForest:
    """BD_LEVEL: A's per-fold configuration refit without metal columns (constant within a cell)."""
    return RowForest("raw", config, seed, metal_cols=False, name=f"BD_LEVEL[{config}]", **kw)


def a_prime(config: str, seed: int, **kw) -> RowForest:
    """A_PRIME (NC1): A's configuration with random_state = fold.model_seed + 1."""
    return RowForest("raw", config, int(seed) + 1, name=f"A_PRIME[{config}]", **kw)


class CPair:
    """C_PAIR: gen13 ``AntisymmetricPairArm`` on LEAN209, every training within-cell pair + reversal."""

    def __init__(self, seed: int, *, n_estimators: int | None = None):
        self.seed = int(seed)
        self.n_estimators = int(C_PAIR_N_ESTIMATORS if n_estimators is None else n_estimators)

    def fit(self, X: np.ndarray, Y: np.ndarray, groups, cells, fingerprints=None) -> "CPair":
        cells = _as_cells(cells)
        fp = np.zeros((len(cells), 0)) if fingerprints is None else np.asarray(fingerprints)[cells]
        ctx = M13.FitContext(X_train=np.asarray(X, dtype=np.float64)[cells], Y_train=np.asarray(Y, dtype=np.float64)[cells],
                             groups_train=np.asarray(groups)[cells], fingerprints_train=fp, seed=self.seed,
                             extra={"cells": cells.copy(), "who": "C_PAIR"})
        with guarded_filters():
            self.arm_ = M13.AntisymmetricPairArm(n_estimators=self.n_estimators, max_pairs_per_cell=None,
                                                 **C_PAIR_PARAMS).fit(ctx)
        self.train_cells_ = cells.copy()
        Yc = ctx.Y_train
        self.n_pairs_ = int(sum(int(k) * (int(k) - 1) // 2 for k in (~np.isnan(Yc)).sum(1)))
        return self

    def predict_pairs(self, X: np.ndarray, ci, ia, ib) -> np.ndarray:
        """(f(a,b) - f(b,a)) / 2 for cells ``ci`` (global indices) and metals ``ia``, ``ib``."""
        _single_thread_predict(self.arm_.model_)
        with guarded_filters():
            out = self.arm_.predict_pairs(np.asarray(X, dtype=np.float64), np.asarray(ia, dtype=int),
                                          np.asarray(ib, dtype=int), np.asarray(ci, dtype=int))
        return np.asarray(out, dtype=np.float64)

    def discard(self) -> None:
        self.arm_ = None
