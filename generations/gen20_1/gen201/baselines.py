"""gen20.1 baselines and gen20 references (IMPLEMENTATION_PLAN.md §3 "Baselines"; PRE_REGISTRATION.md §4.4).

Every baseline is fitted on training cells only, given as explicit indices into the full-cohort arrays
(``cells``), and predicts for any cells.  Weighted means use A's row weights (``learners.row_weights``: w_c /
n_obs(c), w_c chemotype-balanced over the fit's own training cells, renormalised to mean 1).

log D baselines (14-metal matrices): BD_MEAN, BD_METAL, BD_NN, BD_LEVEL_CURVE's mean curve (BD_COND and BD_LEVEL
are ``learners.RowForest`` fits, run by ``fitting``).
log SF baselines (per pair): FLAT (0), PAIRMEAN (gen13 ``PairMeanArm`` at strict-adjacent types, absent type ->
chemotype-balanced mean of every strict-adjacent training pair), HEAVIER_ALWAYS (gen13 ``HeavierAlwaysArm``),
G14_ZERO / MEAN_CURVE_REF / NN_TANIMOTO_REF (gen20 arms refit in fold through ``gen20k.cov.make_ctx`` and
``gen20k.arms.standard_arms()``; k = 0 curves, pair = c[a] - c[b]).  BD_*_SF = in-cell differences of the log D
baselines (``pair_diff``).
"""
from __future__ import annotations

import warnings

import numpy as np

from . import bootstrap  # noqa: F401
import gen13sep.models as M13  # noqa: E402
import gen20k.arms as AR  # noqa: E402
import gen20k.cov as CV  # noqa: E402

from . import data as D  # noqa: E402
from .learners import _as_cells, row_layout, row_weights  # noqa: E402

N_LN = D.N_LN
REFS: tuple[str, ...] = ("G14", "MEAN_CURVE", "NN_TANIMOTO")
REF_COLUMNS = {"G14": "G14_ZERO", "MEAN_CURVE": "MEAN_CURVE_REF", "NN_TANIMOTO": "NN_TANIMOTO_REF"}
STRICT = np.array([[abs(int(D.Z[a]) - int(D.Z[b])) == 1 for b in range(N_LN)] for a in range(N_LN)])


def pair_diff(M: np.ndarray, rows, ia, ib) -> np.ndarray:
    """In-cell difference ``M[r, a] - M[r, b]`` of a 14-metal prediction matrix (lighter first)."""
    rows, ia, ib = (np.asarray(v, dtype=int) for v in (rows, ia, ib))
    return M[rows, ia] - M[rows, ib]


def fit_context(X, Y, groups, cells, *, fingerprints=None, seed: int = 0, who: str = ""):
    """A gen13 ``FitContext`` on ``cells`` only (constructed through the module attribute so tests can spy on it);
    ``extra['cells']`` records the global training indices."""
    cells = _as_cells(cells)
    fp = np.zeros((len(cells), 0)) if fingerprints is None else np.asarray(fingerprints)[cells]
    Xt = np.zeros((len(cells), 0)) if X is None else np.asarray(X, dtype=np.float64)[cells]
    return M13.FitContext(X_train=Xt, Y_train=np.asarray(Y, dtype=np.float64)[cells], groups_train=np.asarray(groups)[cells],
                          fingerprints_train=fp, seed=int(seed), extra={"cells": cells.copy(), "who": who})


# --------------------------------------------------------------------------------------
# log D baselines
# --------------------------------------------------------------------------------------
class MeanBaselines:
    """BD_MEAN (weighted mean of every training log D entry) and BD_METAL (per-metal weighted mean; a metal absent
    from training falls back to BD_MEAN)."""

    def fit(self, Y, groups, cells) -> "MeanBaselines":
        cells = _as_cells(cells)
        Yc = np.asarray(Y, dtype=np.float64)[cells]
        lci, mi = row_layout(Yc)
        y = Yc[lci, mi]
        w = row_weights(np.asarray(groups)[cells], Yc, lci)
        self.mean_ = float(np.sum(w * y) / np.sum(w))
        num = np.bincount(mi, weights=w * y, minlength=N_LN)
        den = np.bincount(mi, weights=w, minlength=N_LN)
        self.seen_ = den > 0
        self.metal_ = np.where(self.seen_, num / np.where(self.seen_, den, 1.0), self.mean_)
        self.train_cells_ = cells.copy()
        return self

    def predict_mean(self, n: int) -> np.ndarray:
        return np.full((int(n), N_LN), self.mean_, dtype=np.float64)

    def predict_metal(self, n: int) -> np.ndarray:
        return np.tile(self.metal_, (int(n), 1)).astype(np.float64)

    @property
    def metal_fallback(self) -> np.ndarray:
        """14 booleans: the metal was absent from training (BD_METAL = BD_MEAN there)."""
        return ~self.seen_


class NearestNeighbour:
    """BD_NN: nearest training extractant(s) by ECFP Tanimoto (``bench.frames['ECFP'] > 0.5``; gen20 NNTanimoto:
    unique training extractants, the fingerprint of their first training cell, exact ties averaged).  For each tied
    extractant e: its weighted mean training log D at m if e has an observation at m, else its weighted mean offset
    from BD_METAL plus BD_METAL(m); the prediction is the mean over the tied extractants.  Conditions are ignored."""

    def fit(self, Y, groups, cells, *, fp: np.ndarray, extractants, means: MeanBaselines) -> "NearestNeighbour":
        cells = _as_cells(cells)
        Yc = np.asarray(Y, dtype=np.float64)[cells]
        lci, mi = row_layout(Yc)
        y = Yc[lci, mi]
        w = row_weights(np.asarray(groups)[cells], Yc, lci)
        ext = np.asarray(extractants).astype(str)
        names, first = np.unique(ext[cells], return_index=True)
        code = np.searchsorted(names, ext[cells])[lci]
        E = len(names)
        num = np.zeros((E, N_LN))
        den = np.zeros((E, N_LN))
        np.add.at(num, (code, mi), w * y)
        np.add.at(den, (code, mi), w)
        off_num = np.bincount(code, weights=w * (y - means.metal_[mi]), minlength=E)
        off_den = np.bincount(code, weights=w, minlength=E)
        self.has_ = den > 0
        self.level_ = np.where(self.has_, num / np.where(self.has_, den, 1.0), np.nan)
        self.offset_ = off_num / off_den
        self.metal_ = means.metal_.copy()
        self.train_ext_ = names
        self.train_fp_ = np.asarray(fp, dtype=bool)[cells][first]
        self.fp_all_ = np.asarray(fp, dtype=bool)
        self.train_cells_ = cells.copy()
        return self

    def neighbours(self, cells) -> list[np.ndarray]:
        te = self.fp_all_[_as_cells(cells)].astype(np.int32)
        tr = self.train_fp_.astype(np.int32)
        inter = te @ tr.T
        union = te.sum(1)[:, None] + tr.sum(1)[None, :] - inter
        sim = inter / np.maximum(union, 1)
        return [np.flatnonzero(sim[i] == sim[i].max()) for i in range(len(te))]

    def predict(self, cells) -> tuple[np.ndarray, np.ndarray]:
        """(prediction matrix, fallback matrix): fallback[i, m] counts the tied extractants that used the offset
        path at m (0 = none)."""
        nbrs = self.neighbours(cells)
        P = np.zeros((len(nbrs), N_LN))
        FB = np.zeros((len(nbrs), N_LN), dtype=int)
        for i, best in enumerate(nbrs):
            has = self.has_[best]                                         # (n_best, 14)
            via_offset = self.offset_[best][:, None] + self.metal_[None, :]
            vals = np.where(has, self.level_[best], via_offset)
            P[i] = vals.mean(axis=0)
            FB[i] = (~has).sum(axis=0)
        return P, FB


def mean_curve(Y, groups, cells) -> np.ndarray:
    """MC(m): gen13 ``MeanCurveArm`` (chemotype-balanced mean centred curve of the training cells)."""
    ctx = fit_context(None, Y, groups, cells, who="MEAN_CURVE(BD_LEVEL_CURVE)")
    return np.asarray(M13.MeanCurveArm().fit(ctx).curve_, dtype=np.float64)


# --------------------------------------------------------------------------------------
# log SF baselines
# --------------------------------------------------------------------------------------
def _pair_weights(Yc: np.ndarray, groups_cells) -> tuple[np.ndarray, np.ndarray]:
    """gen13 PairMeanArm's accumulators: weighted pair sums and weights (all a < b observed pairs)."""
    w = M13.chemotype_balanced_weights(groups_cells)
    num = np.zeros((N_LN, N_LN))
    den = np.zeros((N_LN, N_LN))
    for i in range(len(Yc)):
        obs = np.flatnonzero(~np.isnan(Yc[i]))
        for a in obs:
            for b in obs:
                if a < b:
                    num[a, b] += w[i] * (Yc[i, a] - Yc[i, b])
                    den[a, b] += w[i]
    return num, den


class PairMean:
    """PAIRMEAN: gen13 ``PairMeanArm`` (chemotype-balanced) at strict-adjacent types; a type absent from training
    -> the chemotype-balanced mean of every strict-adjacent training pair (also used for any other absent type)."""

    def fit(self, Y, groups, cells) -> "PairMean":
        ctx = fit_context(None, Y, groups, cells, who="PAIRMEAN")
        self.table_ = np.asarray(M13.PairMeanArm().fit(ctx).table_, dtype=np.float64)
        num, den = _pair_weights(ctx.Y_train, ctx.groups_train)
        self.present_ = den > 0
        sn, sd = float(num[STRICT].sum()), float(den[STRICT].sum())
        self.strict_mean_ = sn / sd if sd > 0 else 0.0
        self.train_cells_ = ctx.extra["cells"]
        return self

    def predict(self, ia, ib) -> tuple[np.ndarray, np.ndarray]:
        ia, ib = np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)
        pres = self.present_[ia, ib]
        return np.where(pres, self.table_[ia, ib], self.strict_mean_).astype(np.float64), ~pres


class HeavierAlways:
    """HEAVIER_ALWAYS: gen13 ``HeavierAlwaysArm`` (-(chemotype-balanced training mean |log SF|) of the type, -0.3
    for an absent type): the MET-14 sign yardstick."""

    def fit(self, Y, groups, cells) -> "HeavierAlways":
        ctx = fit_context(None, Y, groups, cells, who="HEAVIER_ALWAYS")
        self.table_ = np.asarray(M13.HeavierAlwaysArm().fit(ctx).table_, dtype=np.float64)
        self.train_cells_ = ctx.extra["cells"]
        return self

    def predict(self, ia, ib) -> np.ndarray:
        return self.table_[np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)].astype(np.float64)


# --------------------------------------------------------------------------------------
# gen20 references (k = 0 curves)
# --------------------------------------------------------------------------------------
def gen20_ref_curves(bench, fold, train, cells, names=REFS) -> dict[str, np.ndarray]:
    """``{name: (len(cells), 14) centred k = 0 curve}`` from ``gen20k.cov.make_ctx(bench, fold, train, cells)`` and
    ``gen20k.arms.standard_arms()[name]().fit(ctx).curves(cells)`` (gen20k/runner.py:186-188, 223); FLAT is 0."""
    train = np.asarray(train, dtype=int)
    cells = _as_cells(cells)
    ctx = CV.make_ctx(bench, fold, train, cells)
    facts = AR.standard_arms()
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)        # sklearn's 'penalty' deprecation inside gen14's logistic
        for n in names:
            if n == "FLAT":
                out[n] = np.zeros((len(cells), N_LN))
                continue
            out[n] = np.asarray(facts[n]().fit(ctx).curves(cells), dtype=np.float64).reshape(len(cells), N_LN)
    return out
