"""``evaluation/calibration.py`` -- conformal intervals and the uncertainty evaluation of section 12.

Hand-written (no mapie).  Nothing here fits a point predictor: every function takes predictions made
elsewhere.

Conformal methods (section 12 "split-conformal and CV+ conformal with publication-grouped calibration
folds; Mondrian conformal conditional on domain-status category")
-------------------------------------------------------------------------------------------------------
* :func:`fit_split_conformal` -- scores on an **inner calibration set only**:
  ``absolute``   s_i = |y_i - yhat_i|,            interval yhat +- q;
  ``normalized`` s_i = |y_i - yhat_i| / sd_i,     interval yhat +- q * sd.
  ``q`` is the ``k``-th smallest score with ``k = ceil((n + 1)(1 - alpha))`` (the finite-sample quantile
  ``ceil((n+1)(1-alpha))/n``); when ``k > n`` the interval is unbounded (``q = inf``), never clipped.
  The fit refuses a calibration index that overlaps the outer test index or scores a V6_TARGET_ROWS row
  (section 2 lists calibration sets among the scoring sets), and :func:`apply_conformal` refuses a
  prediction index that overlaps the calibration index.
* :func:`fit_mondrian_conformal` -- the same quantile per category.  A category with too few calibration
  rows, or one never seen in calibration, gets an unbounded interval (no pooled fallback: none is registered).
* :func:`cv_plus_intervals` -- CV+ (Barber, Candes, Ramdas and Tibshirani 2021, absolute residuals): with
  out-of-fold residuals R_i and the fold models' predictions at the test point,
  lower = the ``floor(alpha (n + 1))``-th smallest of ``mu_{-k(i)}(x) - R_i``,
  upper = the ``ceil((1 - alpha)(n + 1))``-th smallest of ``mu_{-k(i)}(x) + R_i`` (unbounded when the rank is
  outside 1..n).  With ``groups`` it checks that every publication group lies in exactly one fold.

Evaluation tables
-----------------
:func:`coverage_by_category` (coverage and width per domain-status category with its cell count),
:func:`width_vs_distance` (Spearman and 10 equal-count bins), :func:`sd_reliability_table` (the binned
reliability curve of |error| against predicted SD, 10 equal-count bins), :func:`gaussian_coverage_curve`,
and the registered pass bands :func:`s1d_check` (section 9 S1(d)), :func:`f3_check` (section 10 F3) and
:func:`knows_when_it_does_not_know` (section 12).

Unregistered details fixed here: equal-count bins are formed on the stable sort of the binned quantity
(ties keep frame order) with ``numpy.array_split``; the section 12 coverage-gap test is applied at every
interval level.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gen19ct.evaluation import metrics as EM

LEVELS: tuple[float, ...] = EM.INTERVAL_LEVELS
METHODS: tuple[str, ...] = ("absolute", "normalized")

#: section 13 domain-status labels, in first-match order
DOMAIN_STATUSES: tuple[str, ...] = ("UNSUPPORTED", "FAMILY_EXTRAPOLATION", "CROSS_METAL_LIGAND_TRANSFER",
                                    "CROSS_LIGAND_TRANSFER", "CROSS_METAL_TRANSFER", "CONDITION_EXTRAPOLATION",
                                    "IN_DOMAIN", "INTERPOLATION")
#: section 9 S1(d) bands on macro coverage over V5-primary cells
S1D_BANDS: dict[float, tuple[float, float]] = {0.50: (0.40, 0.60), 0.80: (0.70, 0.90), 0.95: (0.88, 0.99)}
S1D_CATEGORY_BAND_80: tuple[float, float] = (0.65, 0.92)
S1D_CATEGORY_MIN_CELLS = 20
#: section 10 F3
F3_BAND_80: tuple[float, float] = (0.60, 0.95)
F3_MIN_95 = 0.85
#: section 12 "knows when it does not know"
KNOWS_CATEGORIES: tuple[str, ...] = ("UNSUPPORTED", "FAMILY_EXTRAPOLATION")
KNOWS_REFERENCE_CATEGORY = "IN_DOMAIN"
KNOWS_MAX_COVERAGE_GAP = 0.10
RELIABILITY_BINS = 10
_RANK_TOL = 1e-9


# --------------------------------------------------------------------------------------------- #
# Finite-sample conformal quantile
# --------------------------------------------------------------------------------------------- #

def conformal_rank(n: int, alpha: float) -> int:
    """``k = ceil((n + 1)(1 - alpha))`` computed robustly (``(99+1)*0.8`` is 80, not 81)."""
    if n < 0 or not 0 < alpha < 1:
        raise ValueError("n >= 0 and 0 < alpha < 1 required")
    return int(math.ceil((n + 1) * (1.0 - alpha) - _RANK_TOL))


def conformal_quantile(scores: Sequence[float] | np.ndarray, alpha: float) -> float:
    """The ``conformal_rank``-th smallest score, or ``inf`` when that rank exceeds n."""
    s = np.asarray(scores, dtype=float)
    if not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("conformity scores must be finite and >= 0")
    k = conformal_rank(s.size, alpha)
    if k > s.size or s.size == 0:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])


# --------------------------------------------------------------------------------------------- #
# Split and Mondrian conformal
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConformalCalibration:
    """A fitted conformal calibration: ``quantiles[category][level] = q`` (category ``"__all__"`` for the
    plain split method)."""

    method: str
    levels: tuple[float, ...]
    quantiles: Mapping[str, Mapping[float, float]]
    n_calibration: Mapping[str, int]
    calibration_index: pd.Index = field(repr=False)
    mondrian: bool = False

    def table(self) -> pd.DataFrame:
        rows = [{"method": self.method, "mondrian": self.mondrian, "category": c, "level": lvl, "q": q,
                 "n_calibration": int(self.n_calibration[c]), "rank_k": conformal_rank(int(self.n_calibration[c]), 1 - lvl)}
                for c, per in sorted(self.quantiles.items()) for lvl, q in sorted(per.items())]
        return pd.DataFrame(rows)


def _scores(y: np.ndarray, pred: np.ndarray, sd: np.ndarray | None, method: str) -> np.ndarray:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    r = np.abs(y - pred)
    if method == "absolute":
        return r
    if sd is None:
        raise ValueError("the normalized method needs sd")
    if not np.isfinite(sd).all() or (sd <= 0).any():
        raise ValueError("normalized conformal needs finite sd > 0 on every row")
    return r / sd


def _series(values: Any, index: pd.Index, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        if not values.index.equals(index):
            raise ValueError(f"{name} must be indexed like the calibration / prediction index")
        values = values.to_numpy()
    arr = np.asarray(values, dtype=float)
    if arr.shape != (len(index),):
        raise ValueError(f"{name} must have one value per row")
    return arr


def _check_inner(calibration_index: pd.Index, outer_test_index: Iterable[Any], v6_mask: pd.Series | None,
                 design: str) -> None:
    if calibration_index.has_duplicates:
        raise ValueError("calibration index has duplicate labels")
    overlap = calibration_index.intersection(pd.Index(list(outer_test_index)))
    if len(overlap):
        raise AssertionError(f"conformal calibration must use inner rows only: {len(overlap)} calibration row(s) "
                             f"are in the outer test set; first {list(overlap[:5])}")
    EM.guard_scoring_index(calibration_index, v6_mask, design, what="conformal calibration set")


def fit_split_conformal(y: Any, pred: Any, *, calibration_index: Iterable[Any], outer_test_index: Iterable[Any],
                        v6_mask: pd.Series | None, design: str, sd: Any = None, method: str = "absolute",
                        levels: Sequence[float] = LEVELS) -> ConformalCalibration:
    """Split conformal on an inner calibration set (see module docstring)."""
    cal = pd.Index(list(calibration_index))
    _check_inner(cal, outer_test_index, v6_mask, design)
    yv, pv = _series(y, cal, "y"), _series(pred, cal, "pred")
    sv = None if sd is None else _series(sd, cal, "sd")
    if not (np.isfinite(yv).all() and np.isfinite(pv).all()):
        raise ValueError("calibration y and pred must be finite")
    s = _scores(yv, pv, sv, method)
    q = {float(lvl): conformal_quantile(s, 1.0 - float(lvl)) for lvl in levels}
    return ConformalCalibration(method=method, levels=tuple(float(x) for x in levels), quantiles={"__all__": q},
                                n_calibration={"__all__": int(s.size)}, calibration_index=cal, mondrian=False)


def fit_mondrian_conformal(y: Any, pred: Any, categories: Any, *, calibration_index: Iterable[Any],
                           outer_test_index: Iterable[Any], v6_mask: pd.Series | None, design: str, sd: Any = None,
                           method: str = "absolute", levels: Sequence[float] = LEVELS) -> ConformalCalibration:
    """Mondrian split conformal: one quantile per category (e.g. domain status), no pooled fallback."""
    cal = pd.Index(list(calibration_index))
    _check_inner(cal, outer_test_index, v6_mask, design)
    yv, pv = _series(y, cal, "y"), _series(pred, cal, "pred")
    sv = None if sd is None else _series(sd, cal, "sd")
    cats = np.asarray(categories.to_numpy() if isinstance(categories, pd.Series) else categories, dtype=object)
    if cats.shape != (len(cal),) or pd.isna(cats).any():
        raise ValueError("categories must be one non-missing label per calibration row")
    s = _scores(yv, pv, sv, method)
    q: dict[str, dict[float, float]] = {}
    n: dict[str, int] = {}
    for c in sorted({str(x) for x in cats}):
        m = np.array([str(x) == c for x in cats])
        q[c] = {float(lvl): conformal_quantile(s[m], 1.0 - float(lvl)) for lvl in levels}
        n[c] = int(m.sum())
    return ConformalCalibration(method=method, levels=tuple(float(x) for x in levels), quantiles=q, n_calibration=n,
                                calibration_index=cal, mondrian=True)


def apply_conformal(calibration: ConformalCalibration, pred: Any, *, index: Iterable[Any], sd: Any = None,
                    categories: Any = None) -> pd.DataFrame:
    """``lower_<pct>`` / ``upper_<pct>`` for every level on the prediction rows (index must not overlap the
    calibration rows).  Mondrian: an unseen category gets an unbounded interval."""
    idx = pd.Index(list(index))
    overlap = idx.intersection(calibration.calibration_index)
    if len(overlap):
        raise AssertionError(f"{len(overlap)} prediction row(s) were conformal calibration rows")
    pv = _series(pred, idx, "pred")
    scale = np.ones(len(idx))
    if calibration.method == "normalized":
        if sd is None:
            raise ValueError("the normalized method needs sd at prediction time")
        scale = _series(sd, idx, "sd")
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("normalized conformal needs finite sd > 0 on every prediction row")
    if calibration.mondrian:
        if categories is None:
            raise ValueError("Mondrian calibration needs categories at prediction time")
        cats = [str(c) for c in (categories.to_numpy() if isinstance(categories, pd.Series) else categories)]
        if len(cats) != len(idx):
            raise ValueError("one category per prediction row")
    else:
        cats = ["__all__"] * len(idx)
    out = pd.DataFrame(index=idx)
    for lvl in calibration.levels:
        q = np.array([calibration.quantiles.get(c, {}).get(lvl, np.inf) for c in cats], dtype=float)
        half = np.where(np.isinf(q), np.inf, q * scale)
        lo_c, hi_c = EM.interval_columns(lvl)
        out[lo_c] = pv - half
        out[hi_c] = pv + half
    return out


def gaussian_intervals(pred: Any, sd: Any, *, index: Iterable[Any], levels: Sequence[float] = LEVELS) -> pd.DataFrame:
    """Central Gaussian intervals ``pred +- z_{(1+level)/2} sd`` (for heads that output a mean and an SD)."""
    from scipy.special import ndtri

    idx = pd.Index(list(index))
    pv, sv = _series(pred, idx, "pred"), _series(sd, idx, "sd")
    if not np.isfinite(sv).all() or (sv < 0).any():
        raise ValueError("sd must be finite and >= 0")
    out = pd.DataFrame(index=idx)
    for lvl in levels:
        z = float(ndtri(0.5 + float(lvl) / 2.0))
        lo_c, hi_c = EM.interval_columns(lvl)
        out[lo_c] = pv - z * sv
        out[hi_c] = pv + z * sv
    return out


# --------------------------------------------------------------------------------------------- #
# CV+
# --------------------------------------------------------------------------------------------- #

def _kth_smallest(mat: np.ndarray, k: int) -> np.ndarray:
    n = mat.shape[1]
    if k < 1:
        return np.full(mat.shape[0], -np.inf)
    if k > n:
        return np.full(mat.shape[0], np.inf)
    return np.partition(mat, k - 1, axis=1)[:, k - 1]


def cv_plus_intervals(y_cal: Any, oof_pred: Any, cal_fold: Any, test_pred_by_fold: pd.DataFrame, *,
                      calibration_index: Iterable[Any], v6_mask: pd.Series | None, design: str,
                      levels: Sequence[float] = LEVELS, groups: Any = None, chunk: int = 256) -> pd.DataFrame:
    """CV+ intervals (absolute residuals).

    ``y_cal``, ``oof_pred`` (each training row predicted by the model not trained on its fold) and ``cal_fold``
    are indexed by ``calibration_index`` (inner training rows).  ``test_pred_by_fold`` has one row per test
    point (its index is the test index) and one column per fold label: that fold-model's prediction.  The test
    index must not overlap the calibration rows.  With ``groups`` (publication group per calibration row)
    each group must lie in a single fold."""
    cal = pd.Index(list(calibration_index))
    _check_inner(cal, test_pred_by_fold.index, v6_mask, design)
    yv, pv = _series(y_cal, cal, "y_cal"), _series(oof_pred, cal, "oof_pred")
    folds = np.asarray(cal_fold.to_numpy() if isinstance(cal_fold, pd.Series) else cal_fold, dtype=object)
    if folds.shape != (len(cal),):
        raise ValueError("cal_fold must have one label per calibration row")
    if groups is not None:
        g = pd.Series(np.asarray(groups.to_numpy() if isinstance(groups, pd.Series) else groups, dtype=object))
        per = pd.DataFrame({"g": g.to_numpy(), "f": folds}).groupby("g")["f"].nunique()
        if (per > 1).any():
            raise AssertionError(f"CV+ calibration folds split {int((per > 1).sum())} publication group(s)")
    labels = list(test_pred_by_fold.columns)
    missing = sorted({str(f) for f in folds} - {str(c) for c in labels})
    if missing:
        raise ValueError(f"test_pred_by_fold lacks fold column(s) {missing}")
    col_of = {str(c): i for i, c in enumerate(labels)}
    fold_col = np.array([col_of[str(f)] for f in folds], dtype=int)
    resid = np.abs(yv - pv)
    tp = test_pred_by_fold.to_numpy(dtype=float)
    if not (np.isfinite(resid).all() and np.isfinite(tp).all()):
        raise ValueError("CV+ inputs must be finite")
    n = len(cal)
    out = pd.DataFrame(index=test_pred_by_fold.index)
    for lvl in levels:
        alpha = 1.0 - float(lvl)
        k_lo = int(math.floor(alpha * (n + 1) + _RANK_TOL))
        k_hi = conformal_rank(n, alpha)
        lo = np.empty(len(tp))
        hi = np.empty(len(tp))
        for start in range(0, len(tp), max(1, int(chunk))):
            mu = tp[start:start + chunk][:, fold_col]  # (m, n): fold-model prediction for each calibration row
            lo[start:start + chunk] = _kth_smallest(mu - resid[None, :], k_lo)
            hi[start:start + chunk] = _kth_smallest(mu + resid[None, :], k_hi)
        lo_c, hi_c = EM.interval_columns(lvl)
        out[lo_c] = lo
        out[hi_c] = hi
    return out


# --------------------------------------------------------------------------------------------- #
# Evaluation tables
# --------------------------------------------------------------------------------------------- #

def coverage_by_category(frame: pd.DataFrame, regime: EM.Regime, *, category_col: str, unit_cols: Sequence[str],
                         v6_mask: pd.Series | None, levels: Sequence[float] = LEVELS, y_col: str = EM.Y_COL,
                         pred_col: str = EM.PRED_COL, categories: Sequence[str] = DOMAIN_STATUSES) -> pd.DataFrame:
    """Coverage and mean width per category (section 12), with each category's unit (cell) count.

    Rows are split by ``category_col`` first; within a category, per unit then equal-weight macro
    (``unit_macro``) and pooled (``row_pooled``).  Every registered category is printed, with n_units = 0 and
    NaN values when absent; unexpected labels are printed too."""
    EM.require_columns(frame, [category_col, y_col, pred_col] + list(unit_cols))
    EM.guard_scoring_index(frame.index, v6_mask, regime.design, what="coverage by category")
    if frame[category_col].isna().any():
        raise ValueError(f"{category_col!r} has missing values")
    present = [str(c) for c in frame[category_col].astype(str).unique()]
    order = list(categories) + sorted(set(present) - set(categories))
    recs = []
    for cat in order:
        sub = frame[frame[category_col].astype(str) == cat]
        for agg, ucols in (("unit_macro", tuple(unit_cols)), ("row_pooled", ())):
            if len(sub):
                pu = EM.per_unit_table(sub, unit_cols=ucols, y_col=y_col, pred_col=pred_col, levels=levels)
                n_units = int(len(pu)) if ucols else int(len(sub))
                cells = int(len(EM.unit_index(sub, unit_cols)[1]))
            else:
                pu, n_units, cells = None, 0, 0
            for lvl in levels:
                pct = int(round(lvl * 100))
                for metric in (f"coverage_{pct}", f"width_{pct}"):
                    val = EM.macro_mean(pu[metric]) if pu is not None else float("nan")
                    r = EM.summary_record(regime, stratum=f"{category_col}={cat}", metric=metric, aggregation=agg,
                                          role="primary" if agg == "unit_macro" else "side",
                                          averaging_unit=EM.unit_label(ucols), value=val, n_units=n_units,
                                          n_rows=len(sub), note=f"category_units={cells}")
                    r["category"] = cat
                    r["category_units"] = cells
                    recs.append(r)
    return pd.DataFrame(recs, columns=list(EM.SUMMARY_COLUMNS) + ["category", "category_units"])


def _equal_count_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    bins = np.empty(len(x), dtype=int)
    for b, chunk in enumerate(np.array_split(order, min(n_bins, max(1, len(x))))):
        bins[chunk] = b
    return bins


def _with_regime(rows: list[dict[str, Any]], regime: EM.Regime, columns: Sequence[str]) -> pd.DataFrame:
    """Row-level tables: the regime columns first, averaging unit ``row``, n_units = n_rows of the table row."""
    out = []
    for r in rows:
        rec = regime.record()
        rec.update({"averaging_unit": "row", "n_units": int(r["n_rows"]), "n_rows": int(r["n_rows"])})
        rec.update(r)
        out.append(rec)
    return pd.DataFrame(out, columns=list(EM.REGIME_COLUMNS) + [c for c in columns if c not in EM.REGIME_COLUMNS])


def width_vs_distance(frame: pd.DataFrame, regime: EM.Regime, *, distance_cols: Sequence[str],
                      v6_mask: pd.Series | None, levels: Sequence[float] = LEVELS, n_bins: int = RELIABILITY_BINS,
                      y_col: str = EM.Y_COL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interval width against transfer distance (section 12): ``(spearman, binned)``.

    ``spearman``: one row per (distance column, level) with Spearman(width, distance) over rows with a finite
    distance.  ``binned``: equal-count bins of each distance with mean distance, mean width and coverage.
    Pass ``support_score`` and each raw support quantity as ``distance_cols``."""
    need = list(distance_cols) + [y_col]
    for lvl in levels:
        need += list(EM.interval_columns(lvl))
    EM.require_columns(frame, need)
    EM.guard_scoring_index(frame.index, v6_mask, regime.design, what="width vs distance")
    y = EM.finite_values(frame, y_col)
    sp, bn = [], []
    for dcol in distance_cols:
        d = pd.to_numeric(frame[dcol], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(d)
        bins = _equal_count_bins(d[ok], n_bins) if ok.any() else np.array([], dtype=int)
        for lvl in levels:
            lo_c, hi_c = EM.interval_columns(lvl)
            lo = frame[lo_c].to_numpy(dtype=float)[ok]
            hi = frame[hi_c].to_numpy(dtype=float)[ok]
            width = hi - lo
            cov = ((y[ok] >= lo) & (y[ok] <= hi)).astype(float)
            sp.append({"distance": dcol, "level": lvl, "n_rows": int(ok.sum()), "n_missing_distance": int((~ok).sum()),
                       "spearman_width_distance": EM.spearman_rho(width, d[ok]) if ok.sum() > 1 else float("nan")})
            for b in range(int(bins.max()) + 1 if bins.size else 0):
                m = bins == b
                bn.append({"distance": dcol, "level": lvl, "bin": b, "n_rows": int(m.sum()),
                           "distance_min": float(d[ok][m].min()), "distance_max": float(d[ok][m].max()),
                           "distance_mean": float(d[ok][m].mean()), "mean_width": float(width[m].mean()),
                           "coverage": float(cov[m].mean())})
    sp_cols = ["distance", "level", "n_missing_distance", "spearman_width_distance"]
    bn_cols = ["distance", "level", "bin", "distance_min", "distance_max", "distance_mean", "mean_width", "coverage"]
    return _with_regime(sp, regime, sp_cols), _with_regime(bn, regime, bn_cols)


def sd_reliability_table(frame: pd.DataFrame, regime: EM.Regime, *, v6_mask: pd.Series | None, y_col: str = EM.Y_COL,
                         pred_col: str = EM.PRED_COL, sd_col: str = EM.SD_COL,
                         n_bins: int = RELIABILITY_BINS) -> pd.DataFrame:
    """Binned reliability curve (section 12): 10 equal-count bins of predicted SD; per bin the mean SD, the
    observed RMSE and mean |error|, and the Gaussian expectation ``sqrt(2/pi) * mean SD`` of mean |error|."""
    EM.require_columns(frame, [y_col, pred_col, sd_col])
    EM.guard_scoring_index(frame.index, v6_mask, regime.design, what="SD reliability table")
    y, p, sd = EM.finite_values(frame, y_col), EM.finite_values(frame, pred_col), EM.finite_values(frame, sd_col)
    err = p - y
    bins = _equal_count_bins(sd, n_bins)
    rows = []
    for b in range(int(bins.max()) + 1 if len(bins) else 0):
        m = bins == b
        rows.append({"bin": b, "n_rows": int(m.sum()), "sd_min": float(sd[m].min()), "sd_max": float(sd[m].max()),
                     "mean_sd": float(sd[m].mean()), "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                     "mean_abs_error": float(np.mean(np.abs(err[m]))),
                     "expected_mean_abs_error": float(math.sqrt(2.0 / math.pi) * sd[m].mean())})
    return _with_regime(rows, regime, ["bin", "sd_min", "sd_max", "mean_sd", "rmse", "mean_abs_error",
                                       "expected_mean_abs_error"])


def gaussian_coverage_curve(frame: pd.DataFrame, regime: EM.Regime, *, v6_mask: pd.Series | None,
                            y_col: str = EM.Y_COL, pred_col: str = EM.PRED_COL, sd_col: str = EM.SD_COL,
                            nominal: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
                            ) -> pd.DataFrame:
    """Reliability-diagram table for a Gaussian predictive: nominal vs empirical central-interval coverage."""
    from scipy.special import ndtri

    EM.require_columns(frame, [y_col, pred_col, sd_col])
    EM.guard_scoring_index(frame.index, v6_mask, regime.design, what="Gaussian coverage curve")
    y, p, sd = EM.finite_values(frame, y_col), EM.finite_values(frame, pred_col), EM.finite_values(frame, sd_col)
    z = np.abs(y - p)
    rows = []
    for lvl in nominal:
        half = float(ndtri(0.5 + float(lvl) / 2.0)) * sd
        rows.append({"nominal": float(lvl), "empirical": float(np.mean(z <= half)), "n_rows": int(len(y))})
    return _with_regime(rows, regime, ["nominal", "empirical"])


# --------------------------------------------------------------------------------------------- #
# Multi-seed conformal intervals of the deterministic arms (section 15, resolved 2026-09-15)
# --------------------------------------------------------------------------------------------- #

SEED_MEAN_LABEL = "mean_of_seeds"


def seed_mean_interval_metrics(y: pd.Series, intervals_by_seed: Mapping[Any, pd.DataFrame], *,
                               units: pd.Series | None = None, levels: Sequence[float] = LEVELS) -> pd.DataFrame:
    """Coverage and mean width of the conformal intervals drawn with each seed, and their mean over the seeds.

    Section 15 resolution: the deterministic arms' conformal inner folds are drawn with each discovery seed (each
    withheld seed at confirmation) and the coverage and width metrics are the mean over those seeds; point predictions
    are unaffected.  ``y`` is indexed by scored row; every frame of ``intervals_by_seed`` carries ``lower_<pct>`` /
    ``upper_<pct>`` on the same rows.  ``units`` (same index) gives the section 4 averaging unit: ``unit_macro``
    (per unit, then an equal-weight mean) is printed with ``row_pooled`` beside; without ``units`` only
    ``row_pooled``.  An unbounded interval covers and has infinite width.  Tidy rows: ``seed`` (``mean_of_seeds`` for
    the mean), ``metric`` (``coverage_<pct>`` / ``width_<pct>``), ``aggregation``, ``value``, ``n_units``,
    ``n_rows``, ``n_seeds``."""
    if not isinstance(y, pd.Series) or y.index.has_duplicates:
        raise ValueError("y must be a Series with a unique row index")
    if not intervals_by_seed:
        raise ValueError("at least one seed is needed")
    yv = y.to_numpy(dtype=float)
    if not np.isfinite(yv).all():
        raise ValueError("y must be finite on every scored row")
    if units is not None:
        if not isinstance(units, pd.Series) or not units.index.equals(y.index) or units.isna().any():
            raise ValueError("units must be a complete Series on the y index")
        codes, uniq = pd.factorize(units.astype(str), sort=True)
    recs = []
    for seed, iv in intervals_by_seed.items():
        if not iv.index.equals(y.index):
            iv = iv.reindex(y.index)
        for lvl in levels:
            lo_c, hi_c = EM.interval_columns(lvl)
            lo = pd.to_numeric(iv[lo_c], errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(iv[hi_c], errors="coerce").to_numpy(dtype=float)
            if np.isnan(lo).any() or np.isnan(hi).any() or (lo > hi).any():
                raise ValueError(f"seed {seed}: interval {lo_c}/{hi_c} missing or inverted on a scored row")
            covered = ((yv >= lo) & (yv <= hi)).astype(float)
            width = hi - lo
            pct = int(round(float(lvl) * 100))
            for metric, vals in ((f"coverage_{pct}", covered), (f"width_{pct}", width)):
                recs.append({"seed": seed, "metric": metric, "aggregation": "row_pooled", "value": float(np.mean(vals)),
                             "n_units": int(len(yv)), "n_rows": int(len(yv)), "n_seeds": 1})
                if units is not None:
                    per = pd.Series(vals).groupby(codes).mean()
                    recs.append({"seed": seed, "metric": metric, "aggregation": "unit_macro", "value": float(per.mean()),
                                 "n_units": int(len(uniq)), "n_rows": int(len(yv)), "n_seeds": 1})
    per_seed = pd.DataFrame(recs)
    per_seed["seed"] = per_seed["seed"].astype(object)
    mean = per_seed.groupby(["metric", "aggregation"], sort=False).agg(
        value=("value", "mean"), n_units=("n_units", "first"), n_rows=("n_rows", "first"),
        n_seeds=("seed", "nunique")).reset_index()
    mean.insert(0, "seed", SEED_MEAN_LABEL)
    return pd.concat([per_seed, mean[per_seed.columns]], ignore_index=True)


# --------------------------------------------------------------------------------------------- #
# Registered pass bands
# --------------------------------------------------------------------------------------------- #

def _in(v: float, band: tuple[float, float]) -> bool:
    return bool(np.isfinite(v) and band[0] <= v <= band[1])


def s1d_check(macro_coverage: Mapping[float, float],
              category_coverage_80: Mapping[str, tuple[float, int]]) -> dict[str, Any]:
    """Section 9 S1(d): macro 50 / 80 / 95 % coverage inside their bands, and in every domain-status category
    with >= 20 scored cells the 80 % coverage inside [0.65, 0.92].  ``category_coverage_80`` maps category ->
    (macro 80 % coverage, scored cells).  A missing or NaN value fails."""
    items = {}
    for lvl, band in S1D_BANDS.items():
        v = float(macro_coverage.get(lvl, float("nan")))
        items[f"coverage_{int(round(lvl * 100))}"] = {"value": v, "band": band, "pass": _in(v, band)}
    for cat, (v, cells) in sorted(category_coverage_80.items()):
        if int(cells) >= S1D_CATEGORY_MIN_CELLS:
            items[f"category_{cat}_coverage_80"] = {"value": float(v), "cells": int(cells), "band": S1D_CATEGORY_BAND_80,
                                                    "pass": _in(float(v), S1D_CATEGORY_BAND_80)}
    return {"pass": all(i["pass"] for i in items.values()), "items": items}


def f3_check(coverage_80: float, coverage_95: float) -> dict[str, Any]:
    """Section 10 F3: failure when 80 % coverage is outside [0.60, 0.95] or 95 % coverage < 0.85 (NaN fails)."""
    ok80 = _in(float(coverage_80), F3_BAND_80)
    ok95 = bool(np.isfinite(coverage_95) and coverage_95 >= F3_MIN_95)
    return {"failure": not (ok80 and ok95), "coverage_80_in_band": ok80, "coverage_95_at_least_0.85": ok95}


def knows_when_it_does_not_know(spearman_point: float, spearman_interval_low: float,
                                coverage: Mapping[str, Mapping[float, float]],
                                levels: Sequence[float] = LEVELS) -> dict[str, Any]:
    """Section 12: established only if Spearman(|error|, SD) > 0 with the system-cluster interval excluding 0
    (``spearman_interval_low > 0``) **and** coverage in UNSUPPORTED and FAMILY_EXTRAPOLATION is not lower than
    in IN_DOMAIN by more than 0.10 (checked at every level; a category without a value cannot establish it)."""
    sp_ok = bool(np.isfinite(spearman_point) and spearman_point > 0 and np.isfinite(spearman_interval_low)
                 and spearman_interval_low > 0)
    gaps = {}
    ok = True
    ref = coverage.get(KNOWS_REFERENCE_CATEGORY, {})
    for cat in KNOWS_CATEGORIES:
        for lvl in levels:
            r = float(ref.get(lvl, float("nan")))
            v = float(coverage.get(cat, {}).get(lvl, float("nan")))
            gap = r - v
            good = bool(np.isfinite(gap) and gap <= KNOWS_MAX_COVERAGE_GAP)
            gaps[f"{cat}@{lvl:g}"] = {"in_domain": r, "category": v, "gap": gap, "pass": good}
            ok &= good
    return {"established": sp_ok and ok, "spearman_pass": sp_ok, "coverage_gap_pass": ok, "gaps": gaps}
