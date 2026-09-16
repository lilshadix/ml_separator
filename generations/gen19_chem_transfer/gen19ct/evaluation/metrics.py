"""``evaluation/metrics.py`` -- log D and interval metrics of ``preregistration_draft.md`` section 4.

Pure functions on pandas / numpy.  Nothing here fits a model, reads the archive or knows a corpus
constant; the only numbers are the ones section 4 fixes (rank-accuracy |dy| >= 0.1, within-cell Spearman
on cells with >= 5 rows spanning >= 3 condition keys, the 50 / 80 / 95 % intervals).

Pipeline of every summary
-------------------------
per-row errors  ->  per-unit summaries (:func:`per_unit_table`)  ->  equal-weight mean over units.

* The **averaging unit** is an explicit argument (``unit_cols``): the hidden cell for V5 / V5-P, the
  outer fold for V1 (section 3.2 resolution, below), the metal state for V2, the system for V3 / V6, the publication
  group for V0 ... (section 4 table).  ``equal_weight_within`` gives the V6 rule "each metal weight 1/2" inside a
  system.  :func:`registered_unit_cols` names each design's registered unit and :func:`design_logd_summary` /
  :func:`design_interval_summary` / :func:`design_per_unit_table` apply it.
* Every summary also prints two **side versions** that never decide (section 4): ``row_pooled`` (every
  scored row weight 1) and ``system_macro`` (rows pooled within ``extractant_system_key``, then an
  equal-weight mean over systems).
* Every summary is a tidy long frame: one row per (stratum, metric, aggregation) carrying the regime
  columns of section 0 -- design, variant, half, arm, seed, seed_set, status -- plus averaging_unit,
  n_units, n_rows (:data:`SUMMARY_COLUMNS`).

Readings fixed here (the pre-registration is silent; the conservative option is taken)
---------------------------------------------------------------------------------------
* **RMSE / R^2 under the averaging unit.**  RMSE: per-unit RMSE, then the equal-weight mean.  R^2:
  weighted R^2 with each unit's rows carrying total weight 1 (row-pooled R^2 printed beside).
* **Constant predictions and Spearman.**  Cell eligibility for the within-cell Spearman depends only on the
  observed rows (>= 5 rows, >= 3 distinct condition keys, non-constant log D), so every arm is scored on
  the same cells.  An arm whose prediction is constant inside an eligible cell has no within-cell ranking
  and scores rho = 0 (``constant_prediction_rho``); the count is printed in ``note``.  The same rule applies
  to the Spearman across rows.
* **Float tolerances.**  ``|dy| >= 0.1`` is tested as ``|dy| >= 0.1 - 1e-9`` (so 0.3 - 0.2 qualifies), and a
  prediction tie is ``|d yhat| <= 1e-12``.
* **Missing values.**  A scored row without a finite observation or prediction raises: every scored row
  must carry a prediction (baselines record their fallback level instead of skipping a row).

V1 scoring unit (section 3.2, resolved by the orchestrator 2026-09-15)
---------------------------------------------------------------------
The V1 scoring unit is the **outer fold** of the exact design: each single-group fold is one unit and the pooled
remainder fold (``REMAINDER``, every group below 20 MODEL rows) is **one** unit -- the unit the halves were balanced
on.  :func:`v1_scoring_units` maps fold ids to units (exact folds: the fold id; the ten grouped folds of the heavy
arms: the row's exact-design unit, so every arm is scored on the same units).  The per-publication-group reading is
kept as an **exploratory** side output (``unit_reading`` column of the design summaries).  Cluster unit (section 8
table: publication group): a single-group fold's cluster is its group; the remainder fold, which cannot be split
without splitting the unit, is one cluster (:func:`design_unit_clusters`; a consequence of nesting units in
clusters, INFERRED from section 8 "scored units nested in them").

Scoring guard (section 2)
-------------------------
Every summary takes a keyword-only ``v6_mask``.  The scored index is passed through
``gen19ct.folds.registered.assert_not_scored`` before any number is computed.  ``v6_mask=None`` is
accepted only for design ``V6`` (the single confirmation run, where those rows are the targets), and a
``V6`` regime with ``seed_set="discovery"`` is refused.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------- #
# Column names and registered constants
# --------------------------------------------------------------------------------------------- #

METAL_STATE_COL = "g19_metal_state"
ELEMENT_COL = "g19_metal"
SYSTEM_COL = "extractant_system_key"
#: ``group_cross_publication_copy`` of ``leakage_publication_components.csv`` mapped onto rows (section 2)
PUB_GROUP_COL = "publication_group"
CONDITION_KEY_COL = "condition_key"
Y_COL = "log_D"
PRED_COL = "pred"
SD_COL = "pred_sd"
CELL_COLS: tuple[str, str] = (METAL_STATE_COL, SYSTEM_COL)

#: central predictive interval levels (section 4)
INTERVAL_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.95)
#: rank accuracy: row pairs with |dy| >= this (section 4)
RANK_MIN_ABS_DY = 0.1
#: within-cell Spearman eligibility (section 4)
WITHIN_CELL_MIN_ROWS = 5
WITHIN_CELL_MIN_CONDITION_KEYS = 3
#: tolerance for ">= threshold" on observed differences
FLOAT_TOL = 1e-9
#: a prediction difference at or below this is a tie (counts 1/2)
TIE_TOL = 1e-12

DESIGNS: tuple[str, ...] = ("V0", "V1", "V2", "V3", "V4", "V5", "V5-P", "V5-PAIR", "V6", "V7")
HALVES: tuple[str, ...] = ("selection", "confirmation", "none")
SEED_SETS: tuple[str, ...] = ("discovery", "confirmation", "deterministic")
STATUSES: tuple[str, ...] = ("registered", "exploratory")
_HALF_ALIASES = {"S": "selection", "C": "confirmation"}

REGIME_COLUMNS: tuple[str, ...] = ("design", "variant", "half", "arm", "seed", "seed_set", "status",
                                   "averaging_unit", "n_units", "n_rows")
SUMMARY_COLUMNS: tuple[str, ...] = REGIME_COLUMNS + ("stratum", "metric", "aggregation", "role", "value", "note")


def interval_columns(level: float) -> tuple[str, str]:
    """``0.8 -> ("lower_80", "upper_80")`` -- the per-row interval column names."""
    pct = int(round(float(level) * 100))
    if not 0 < pct < 100 or not math.isclose(pct / 100, float(level), abs_tol=1e-9):
        raise ValueError(f"interval level must be a whole percentage in (0, 1), got {level!r}")
    return f"lower_{pct}", f"upper_{pct}"


# --------------------------------------------------------------------------------------------- #
# Regime and scoring guard
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Regime:
    """The regime fields every table carries (section 0 "regime of every number").

    ``variant`` names the threshold setting or sensitivity (``primary``, ``loose``, ``HNO3-only`` ...),
    ``half`` the scored half (``selection`` / ``confirmation``; ``none`` for V0 and V6), ``seed`` the run seed
    (``None`` for deterministic arms, with ``seed_set="deterministic"``), ``status`` registered or exploratory.
    """

    design: str
    arm: str
    variant: str = "primary"
    half: str = "selection"
    seed: int | None = None
    seed_set: str = "deterministic"
    status: str = "registered"

    def __post_init__(self) -> None:
        half = _HALF_ALIASES.get(self.half, self.half)
        object.__setattr__(self, "half", half)
        if self.design not in DESIGNS:
            raise ValueError(f"design must be one of {DESIGNS}, got {self.design!r}")
        if half not in HALVES:
            raise ValueError(f"half must be one of {HALVES} (or S / C), got {self.half!r}")
        if self.seed_set not in SEED_SETS:
            raise ValueError(f"seed_set must be one of {SEED_SETS}, got {self.seed_set!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if (self.seed is None) != (self.seed_set == "deterministic"):
            raise ValueError("seed is None exactly when seed_set == 'deterministic'")
        if not isinstance(self.arm, str) or not self.arm:
            raise ValueError("arm must be a non-empty string")
        if self.design == "V6" and self.seed_set == "discovery":
            raise ValueError("V6 is run once, at confirmation (section 3.4); a discovery V6 regime is refused")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def guard_scoring_index(index: Iterable[Any], v6_mask: pd.Series | None, design: str,
                        what: str = "scoring set") -> None:
    """Section 2: every scoring index passes ``registered.assert_not_scored`` before a metric is computed.

    ``v6_mask=None`` is accepted only for design ``V6`` (the confirmation run scores those rows)."""
    if v6_mask is None:
        if design != "V6":
            raise ValueError(f"{what}: v6_mask (V6_TARGET_ROWS) is required for design {design!r}; "
                             "only the single V6 confirmation run may score without it")
        return
    from gen19ct.folds.registered import assert_not_scored  # lazy: pulls the chemistry stack

    assert_not_scored(index, v6_mask, what=what)


# --------------------------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------------------------- #

def require_columns(frame: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c is not None and c not in frame.columns]
    if missing:
        raise KeyError(f"missing column(s) {missing}")
    if frame.index.has_duplicates:
        raise ValueError("the row index must be unique (one label per scored row)")


def finite_values(frame: pd.DataFrame, col: str) -> np.ndarray:
    v = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    bad = ~np.isfinite(v)
    if bad.any():
        raise ValueError(f"column {col!r} has {int(bad.sum())} non-finite value(s) among scored rows")
    return v


def unit_index(frame: pd.DataFrame, unit_cols: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame]:
    """``(codes, keys)``: a code per row (sorted group order) and the unit key frame (one row per code).

    ``unit_cols=()`` makes every row one unit (the row-pooled version)."""
    cols = list(unit_cols)
    if not cols:
        return np.zeros(len(frame), dtype=int), pd.DataFrame({"unit": ["all_rows"]})
    require_columns(frame, cols)
    for c in cols:
        if frame[c].isna().any():
            raise ValueError(f"unit column {c!r} has missing values")
    g = frame.groupby(cols, sort=True)
    codes = g.ngroup().to_numpy(dtype=int)
    keys = g.size().index.to_frame(index=False)
    keys.columns = cols
    return codes, keys.reset_index(drop=True)


def unit_label(unit_cols: Sequence[str]) -> str:
    return "+".join(unit_cols) if unit_cols else "row"


def row_weights(frame: pd.DataFrame, codes: np.ndarray, equal_weight_within: str | None = None) -> np.ndarray:
    """Row weights that sum to 1 inside each unit: ``1/n_u``, or ``1/(n_strata_u * n_us)`` with
    ``equal_weight_within`` (each stratum of the unit -- e.g. each metal of a V6 system -- weight equal)."""
    n_units = int(codes.max()) + 1 if len(codes) else 0
    if equal_weight_within is None:
        n = np.bincount(codes, minlength=n_units).astype(float)
        return 1.0 / n[codes]
    if frame[equal_weight_within].isna().any():
        raise ValueError(f"equal_weight_within column {equal_weight_within!r} has missing values")
    sub = pd.DataFrame({"u": codes, "s": frame[equal_weight_within].to_numpy(dtype=object)})
    n_us = sub.groupby(["u", "s"])["u"].transform("size").to_numpy(dtype=float)
    n_strata = sub.groupby("u")["s"].transform("nunique").to_numpy(dtype=float)
    return 1.0 / (n_us * n_strata)


def spearman_rho(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Spearman rho with average ranks for ties; NaN when either side is constant or n < 2."""
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < 2:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def r2_score(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    """(Weighted) coefficient of determination ``1 - SS_res / SS_tot``; NaN when SS_tot = 0."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if y.size == 0:
        return float("nan")
    ybar = float((w * y).sum() / w.sum())
    ss_tot = float((w * (y - ybar) ** 2).sum())
    ss_res = float((w * (y - pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def crps_gaussian(y: Sequence[float] | np.ndarray, mu: Sequence[float] | np.ndarray,
                  sd: Sequence[float] | np.ndarray) -> np.ndarray:
    """Closed-form CRPS of a Gaussian predictive N(mu, sd^2) at observation y (hand-written).

    ``CRPS = sd * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]`` with ``z = (y - mu) / sd``; at sd = 0 it is
    ``|y - mu|``.  A negative or non-finite sd raises."""
    from scipy.special import ndtr

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    y, mu, sd = np.broadcast_arrays(y, mu, sd)
    if (~np.isfinite(sd)).any() or (sd < 0).any():
        raise ValueError("sd must be finite and >= 0")
    diff = y - mu
    pos = sd > 0
    z = np.divide(diff, sd, out=np.zeros_like(diff, dtype=float), where=pos)
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    closed = sd * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    out = np.where(pos, closed, np.abs(diff))
    return out


def majority_label(frame: pd.DataFrame, unit_cols: Sequence[str], label_col: str) -> pd.DataFrame:
    """Per unit, the ``label_col`` value holding most of its rows (ties: smallest label as text).

    This is the V5 secondary cluster: "publication group of each cell's majority rows" (section 8)."""
    require_columns(frame, list(unit_cols) + [label_col])
    if frame[label_col].isna().any():
        raise ValueError(f"{label_col!r} has missing values")
    cnt = frame.groupby(list(unit_cols) + [label_col], sort=True).size().rename("n").reset_index()
    cnt["_lab"] = cnt[label_col].astype(str)
    cnt = cnt.sort_values(list(unit_cols) + ["n", "_lab"], ascending=[True] * len(unit_cols) + [False, True],
                          kind="stable")
    out = cnt.drop_duplicates(list(unit_cols), keep="first")
    return out[list(unit_cols) + [label_col, "n"]].rename(columns={"n": f"n_rows_majority_{label_col}"}) \
        .reset_index(drop=True)


# --------------------------------------------------------------------------------------------- #
# Per-unit tables
# --------------------------------------------------------------------------------------------- #

def per_unit_table(frame: pd.DataFrame, *, unit_cols: Sequence[str], y_col: str = Y_COL, pred_col: str = PRED_COL,
                   equal_weight_within: str | None = None, sd_col: str | None = None,
                   levels: Sequence[float] | None = None, carry_cols: Sequence[str] = ()) -> pd.DataFrame:
    """One row per averaging unit: ``n_rows``, ``mae``, ``mse``, ``rmse``, ``mean_error``; with ``levels``
    ``coverage_<pct>`` and ``width_<pct>`` from ``lower_<pct>`` / ``upper_<pct>``; with ``sd_col``
    ``crps`` and ``mean_sd``.  Within a unit rows are weighted by :func:`row_weights`.  ``carry_cols`` must be
    constant inside a unit (e.g. the system of a cell) and are copied."""
    need = [y_col, pred_col] + list(carry_cols) + ([sd_col] if sd_col else []) \
        + ([equal_weight_within] if equal_weight_within else [])
    for lvl in levels or ():
        need += list(interval_columns(lvl))
    require_columns(frame, need)
    y, p = finite_values(frame, y_col), finite_values(frame, pred_col)
    codes, keys = unit_index(frame, unit_cols)
    n_units = len(keys)
    w = row_weights(frame, codes, equal_weight_within)
    err = p - y
    out = keys.copy()
    out["n_rows"] = np.bincount(codes, minlength=n_units).astype(int)
    out["mae"] = np.bincount(codes, weights=w * np.abs(err), minlength=n_units)
    out["mse"] = np.bincount(codes, weights=w * err * err, minlength=n_units)
    out["rmse"] = np.sqrt(out["mse"].to_numpy(dtype=float))
    out["mean_error"] = np.bincount(codes, weights=w * err, minlength=n_units)
    for lvl in levels or ():
        lo_c, hi_c = interval_columns(lvl)
        lo = pd.to_numeric(frame[lo_c], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(frame[hi_c], errors="coerce").to_numpy(dtype=float)
        if np.isnan(lo).any() or np.isnan(hi).any():
            raise ValueError(f"interval {lo_c}/{hi_c} has missing bounds among scored rows")
        if (lo > hi).any():
            raise ValueError(f"interval {lo_c} > {hi_c} for {int((lo > hi).sum())} row(s)")
        pct = int(round(lvl * 100))
        covered = ((y >= lo) & (y <= hi)).astype(float)
        out[f"coverage_{pct}"] = np.bincount(codes, weights=w * covered, minlength=n_units)
        width = hi - lo
        out[f"width_{pct}"] = _weighted_sums_allow_inf(codes, w, width, n_units)
    if sd_col:
        sd = finite_values(frame, sd_col)
        out["crps"] = np.bincount(codes, weights=w * crps_gaussian(y, p, sd), minlength=n_units)
        out["mean_sd"] = np.bincount(codes, weights=w * sd, minlength=n_units)
    for c in carry_cols:
        per = pd.DataFrame({"u": codes, "v": frame[c].to_numpy(dtype=object)}).groupby("u")["v"]
        if (per.nunique(dropna=False) > 1).any():
            raise ValueError(f"carry column {c!r} is not constant within a unit")
        out[c] = per.first().reindex(range(n_units)).to_numpy(dtype=object)
    return out


def _weighted_sums_allow_inf(codes: np.ndarray, w: np.ndarray, x: np.ndarray, n_units: int) -> np.ndarray:
    """Per-unit weighted sum where ``x`` may hold +inf (an unbounded conformal interval stays unbounded)."""
    inf = np.isinf(x)
    finite_sum = np.bincount(codes, weights=np.where(inf, 0.0, w * x), minlength=n_units)
    any_inf = np.bincount(codes, weights=inf.astype(float), minlength=n_units) > 0
    return np.where(any_inf, np.inf, finite_sum)


def within_cell_spearman_table(frame: pd.DataFrame, *, cell_cols: Sequence[str] = CELL_COLS, y_col: str = Y_COL,
                               pred_col: str = PRED_COL, condition_key_col: str = CONDITION_KEY_COL,
                               min_rows: int = WITHIN_CELL_MIN_ROWS,
                               min_condition_keys: int = WITHIN_CELL_MIN_CONDITION_KEYS,
                               constant_prediction_rho: float = 0.0) -> pd.DataFrame:
    """Per cell: ``n_rows``, ``n_condition_keys``, ``eligible`` (>= ``min_rows`` rows spanning >=
    ``min_condition_keys`` distinct condition keys, observed log D not constant), ``rho`` (NaN when
    ineligible), ``prediction_constant`` and ``ineligible_reason``.  Eligibility never looks at the prediction."""
    require_columns(frame, list(cell_cols) + [y_col, pred_col, condition_key_col])
    y, p = finite_values(frame, y_col), finite_values(frame, pred_col)
    codes, keys = unit_index(frame, cell_cols)
    ck = frame[condition_key_col].to_numpy(dtype=object)
    rows = []
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for block in np.split(order, bounds):
        if block.size == 0:
            continue
        n = int(block.size)
        n_ck = int(len({str(v) for v in ck[block]}))
        yb, pb = y[block], p[block]
        reason = ""
        if n < min_rows:
            reason = f"rows<{min_rows}"
        elif n_ck < min_condition_keys:
            reason = f"condition_keys<{min_condition_keys}"
        elif np.ptp(yb) == 0:
            reason = "observed_constant"
        const = bool(np.ptp(pb) <= TIE_TOL)
        if reason:
            rho = float("nan")
        elif const:
            rho = float(constant_prediction_rho)
        else:
            rho = spearman_rho(pb, yb)
        rows.append({"_code": int(codes[block[0]]), "n_rows": n, "n_condition_keys": n_ck,
                     "eligible": not reason, "rho": rho, "prediction_constant": const, "ineligible_reason": reason})
    tab = pd.DataFrame(rows).sort_values("_code").reset_index(drop=True)
    return pd.concat([keys, tab.drop(columns="_code")], axis=1)


def rank_accuracy_table(frame: pd.DataFrame, *, unit_cols: Sequence[str], pub_group_col: str = PUB_GROUP_COL,
                        y_col: str = Y_COL, pred_col: str = PRED_COL,
                        min_abs_dy: float = RANK_MIN_ABS_DY) -> pd.DataFrame:
    """Per unit: row pairs inside the unit and inside one publication group with ``|dy| >= min_abs_dy``;
    ``n_pairs``, ``score`` (correct 1, prediction tie 1/2) and ``rank_accuracy = score / n_pairs`` (NaN when
    the unit has no such pair).  Pair eligibility never looks at the prediction."""
    require_columns(frame, list(unit_cols) + [pub_group_col, y_col, pred_col])
    y, p = finite_values(frame, y_col), finite_values(frame, pred_col)
    codes, keys = unit_index(frame, unit_cols)
    if frame[pub_group_col].isna().any():
        raise ValueError(f"{pub_group_col!r} has missing values")
    n_units = len(keys)
    n_pairs = np.zeros(n_units, dtype=np.int64)
    score = np.zeros(n_units, dtype=float)
    blocks = pd.DataFrame({"u": codes, "g": frame[pub_group_col].astype(str).to_numpy()}).groupby(["u", "g"]).indices
    for (u, _g), pos in blocks.items():
        if len(pos) < 2:
            continue
        i, j = np.triu_indices(len(pos), k=1)
        a, b = pos[i], pos[j]
        dy = y[a] - y[b]
        keep = np.abs(dy) >= min_abs_dy - FLOAT_TOL
        if not keep.any():
            continue
        dy, dp = dy[keep], p[a[keep]] - p[b[keep]]
        s = np.where(np.abs(dp) <= TIE_TOL, 0.5, (np.sign(dp) == np.sign(dy)).astype(float))
        n_pairs[u] += int(keep.sum())
        score[u] += float(s.sum())
    out = keys.copy()
    out["n_pairs"] = n_pairs
    out["score"] = score
    out["rank_accuracy"] = np.divide(score, n_pairs, out=np.full(n_units, np.nan), where=n_pairs > 0)
    return out


def spearman_across_units(per_unit: pd.DataFrame, x_col: str, y_col: str) -> float:
    """Spearman rho across units of two per-unit quantities (e.g. cell MAE vs ``support_score``, S1(e))."""
    require_columns(per_unit, [x_col, y_col])
    return spearman_rho(finite_values(per_unit, x_col), finite_values(per_unit, y_col))


# --------------------------------------------------------------------------------------------- #
# Tidy summaries
# --------------------------------------------------------------------------------------------- #

def summary_record(regime: Regime, *, stratum: str, metric: str, aggregation: str, role: str, averaging_unit: str,
                   value: float, n_units: int, n_rows: int, note: str = "") -> dict[str, Any]:
    """One row of a tidy summary (:data:`SUMMARY_COLUMNS`)."""
    r = regime.record()
    r.update({"averaging_unit": averaging_unit, "n_units": int(n_units), "n_rows": int(n_rows), "stratum": stratum,
              "metric": metric, "aggregation": aggregation, "role": role,
              "value": float(value) if value is not None else float("nan"), "note": note})
    return r


def _strata(frame: pd.DataFrame, strata_col: str | None) -> list[tuple[str, pd.DataFrame]]:
    out = [("all", frame)]
    if strata_col is None:
        return out
    require_columns(frame, [strata_col])
    if frame[strata_col].isna().any():
        raise ValueError(f"strata column {strata_col!r} has missing values")
    for val in sorted(frame[strata_col].astype(str).unique()):
        out.append((f"{strata_col}={val}", frame[frame[strata_col].astype(str) == val]))
    return out


def _aggregations(unit_cols: Sequence[str], system_col: str,
                  equal_weight_within: str | None) -> list[tuple[str, str, tuple[str, ...], str | None]]:
    """(aggregation, role, unit columns, equal_weight_within) of the primary and the two side versions."""
    return [("unit_macro", "primary", tuple(unit_cols), equal_weight_within),
            ("row_pooled", "side", (), None),
            ("system_macro", "side", (system_col,), None)]


def macro_mean(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def logd_summary(frame: pd.DataFrame, regime: Regime, *, unit_cols: Sequence[str], v6_mask: pd.Series | None,
                 y_col: str = Y_COL, pred_col: str = PRED_COL, system_col: str = SYSTEM_COL,
                 pub_group_col: str = PUB_GROUP_COL, cell_cols: Sequence[str] = CELL_COLS,
                 condition_key_col: str = CONDITION_KEY_COL, equal_weight_within: str | None = None,
                 strata_col: str | None = None, constant_prediction_rho: float = 0.0) -> pd.DataFrame:
    """Section 4 log D metrics of one arm on one scoring set, as a tidy frame (:data:`SUMMARY_COLUMNS`).

    Metrics: ``mae`` (the primary metric), ``rmse``, ``r2`` (each ``unit_macro`` primary plus ``row_pooled`` and
    ``system_macro`` side versions); ``spearman_rows`` (across the scored rows); ``within_cell_spearman``
    (macro over eligible cells); ``rank_accuracy`` (``unit_macro`` over units with an eligible pair, plus
    ``pair_pooled``).  ``strata_col`` repeats everything per stratum value at row level (e.g. acid medium)."""
    require_columns(frame, list(unit_cols) + [y_col, pred_col, system_col, pub_group_col, condition_key_col]
                    + list(cell_cols))
    guard_scoring_index(frame.index, v6_mask, regime.design, what=f"{regime.design}/{regime.arm} log D scoring")
    recs: list[dict[str, Any]] = []
    for stratum, sub in _strata(frame, strata_col):
        if not len(sub):
            continue
        y, p = finite_values(sub, y_col), finite_values(sub, pred_col)
        for agg, role, ucols, eww in _aggregations(unit_cols, system_col, equal_weight_within):
            pu = per_unit_table(sub, unit_cols=ucols, y_col=y_col, pred_col=pred_col, equal_weight_within=eww)
            codes, _ = unit_index(sub, ucols)
            w = row_weights(sub, codes, eww)
            label = unit_label(ucols)
            common = dict(stratum=stratum, aggregation=agg, role=role, averaging_unit=label,
                          n_units=len(pu) if ucols else len(sub), n_rows=len(sub))
            recs.append(summary_record(regime, metric="mae", value=macro_mean(pu["mae"]), **common))
            recs.append(summary_record(regime, metric="rmse", value=macro_mean(pu["rmse"]), **common))
            recs.append(summary_record(regime, metric="r2", value=r2_score(y, p, w), **common))
        rho = spearman_rho(p, y)
        note = ""
        if np.ptp(p) <= TIE_TOL and np.ptp(y) > 0:
            rho, note = float(constant_prediction_rho), "prediction constant across rows: rho set to " \
                                                        f"{constant_prediction_rho:g}"
        recs.append(summary_record(regime, stratum=stratum, metric="spearman_rows", aggregation="rows",
                                   role="primary", averaging_unit="row", value=rho, n_units=len(sub), n_rows=len(sub),
                                   note=note))
        wc = within_cell_spearman_table(sub, cell_cols=cell_cols, y_col=y_col, pred_col=pred_col,
                                        condition_key_col=condition_key_col,
                                        constant_prediction_rho=constant_prediction_rho)
        el = wc[wc["eligible"]]
        n_const = int(el["prediction_constant"].sum())
        recs.append(summary_record(regime, stratum=stratum, metric="within_cell_spearman", aggregation="cell_macro",
                                   role="primary", averaging_unit=unit_label(cell_cols), value=macro_mean(el["rho"]),
                                   n_units=len(el), n_rows=int(el["n_rows"].sum()),
                                   note=f"cells with constant prediction scored rho={constant_prediction_rho:g}: "
                                        f"{n_const}"))
        ra = rank_accuracy_table(sub, unit_cols=unit_cols, pub_group_col=pub_group_col, y_col=y_col, pred_col=pred_col)
        ra_el = ra[ra["n_pairs"] > 0]
        tot_pairs = int(ra["n_pairs"].sum())
        recs.append(summary_record(regime, stratum=stratum, metric="rank_accuracy", aggregation="unit_macro",
                                   role="primary", averaging_unit=unit_label(unit_cols),
                                   value=macro_mean(ra_el["rank_accuracy"]), n_units=len(ra_el), n_rows=len(sub),
                                   note=f"row pairs: {tot_pairs}"))
        pooled = float(ra["score"].sum() / tot_pairs) if tot_pairs else float("nan")
        recs.append(summary_record(regime, stratum=stratum, metric="rank_accuracy", aggregation="pair_pooled",
                                   role="side", averaging_unit="row_pair", value=pooled, n_units=tot_pairs,
                                   n_rows=len(sub), note=f"row pairs: {tot_pairs}"))
    return pd.DataFrame(recs, columns=list(SUMMARY_COLUMNS))


def interval_summary(frame: pd.DataFrame, regime: Regime, *, unit_cols: Sequence[str], v6_mask: pd.Series | None,
                     levels: Sequence[float] = INTERVAL_LEVELS, y_col: str = Y_COL, pred_col: str = PRED_COL,
                     sd_col: str | None = None, system_col: str = SYSTEM_COL,
                     equal_weight_within: str | None = None, strata_col: str | None = None) -> pd.DataFrame:
    """Section 4 / 12 interval metrics as a tidy frame: ``coverage_<pct>`` and ``width_<pct>`` per level
    (per unit, then ``unit_macro`` primary, ``row_pooled`` and ``system_macro`` beside); with ``sd_col`` also
    ``crps`` (Gaussian closed form) and ``spearman_abs_error_sd`` (across rows)."""
    require_columns(frame, list(unit_cols) + [y_col, pred_col, system_col] + ([sd_col] if sd_col else []))
    guard_scoring_index(frame.index, v6_mask, regime.design, what=f"{regime.design}/{regime.arm} interval scoring")
    recs: list[dict[str, Any]] = []
    for stratum, sub in _strata(frame, strata_col):
        if not len(sub):
            continue
        for agg, role, ucols, eww in _aggregations(unit_cols, system_col, equal_weight_within):
            pu = per_unit_table(sub, unit_cols=ucols, y_col=y_col, pred_col=pred_col, equal_weight_within=eww,
                                sd_col=sd_col, levels=levels)
            common = dict(stratum=stratum, aggregation=agg, role=role, averaging_unit=unit_label(ucols),
                          n_units=len(pu) if ucols else len(sub), n_rows=len(sub))
            for lvl in levels:
                pct = int(round(lvl * 100))
                for metric in (f"coverage_{pct}", f"width_{pct}"):
                    recs.append(summary_record(regime, metric=metric, value=macro_mean(pu[metric]), **common))
            if sd_col:
                recs.append(summary_record(regime, metric="crps", value=macro_mean(pu["crps"]), **common))
        if sd_col:
            err = np.abs(finite_values(sub, pred_col) - finite_values(sub, y_col))
            sd = finite_values(sub, sd_col)
            recs.append(summary_record(regime, stratum=stratum, metric="spearman_abs_error_sd", aggregation="rows",
                                       role="primary", averaging_unit="row", value=spearman_rho(err, sd),
                                       n_units=len(sub), n_rows=len(sub), note="NaN when the predicted SD is constant"))
    return pd.DataFrame(recs, columns=list(SUMMARY_COLUMNS))


def metric_value(summary: pd.DataFrame, metric: str, aggregation: str = "unit_macro", stratum: str = "all") -> float:
    """The single value of ``metric`` / ``aggregation`` / ``stratum`` in a tidy summary (raises if not unique)."""
    sel = summary[(summary["metric"] == metric) & (summary["aggregation"] == aggregation)
                  & (summary["stratum"] == stratum)]
    if len(sel) != 1:
        raise KeyError(f"{len(sel)} rows for metric={metric!r} aggregation={aggregation!r} stratum={stratum!r}")
    return float(sel["value"].iloc[0])


def as_mapping(summary: pd.DataFrame, aggregation: str = "unit_macro", stratum: str = "all") -> Mapping[str, float]:
    """``{metric: value}`` of one aggregation and stratum (convenience for decision code and tests)."""
    sel = summary[(summary["aggregation"] == aggregation) & (summary["stratum"] == stratum)]
    return dict(zip(sel["metric"], sel["value"].astype(float)))


# --------------------------------------------------------------------------------------------- #
# Registered averaging units per design (section 4; section 3.2 resolution for V1)
# --------------------------------------------------------------------------------------------- #

#: the outer-fold id column of a prediction frame
FOLD_COL = "fold_id"
#: the pooled remainder fold of the exact V1 design (equal to ``folds.source_holdout.REMAINDER``; tested)
V1_REMAINDER_UNIT = "REMAINDER"
#: the column :func:`with_registered_units` adds for V1: the registered scoring unit (the outer fold)
V1_UNIT_COL = "v1_outer_fold_unit"
V1_SCHEMES: tuple[str, ...] = ("exact", "grouped")
#: registered averaging unit of every design with a row-level unit (V4 / V7 are named by the caller; V5-PAIR
#: averages over cell pairs, ``pairs.CELL_PAIR_COLS``)
REGISTERED_UNIT_COLS: dict[str, tuple[str, ...]] = {
    "V0": (PUB_GROUP_COL,), "V1": (V1_UNIT_COL,), "V2": (METAL_STATE_COL,), "V3": (SYSTEM_COL,),
    "V5": CELL_COLS, "V5-P": CELL_COLS, "V6": (SYSTEM_COL,),
}
REGISTERED_UNIT_READING: dict[str, str] = {
    "V0": "publication group (group_cross_publication_copy) within each seed (section 3.6 resolution)",
    "V1": "outer fold: each single-group fold one unit, the pooled remainder fold one unit (section 3.2 resolution)",
    "V2": "metal state", "V3": "system", "V5": "hidden cell", "V5-P": "hidden cell", "V6": "system",
}
#: exploratory unit readings printed beside the registered one (never deciding)
EXPLORATORY_UNIT_READINGS: dict[str, tuple[str, ...]] = {"V1": ("publication_group",)}
UNIT_KEY_SEP = " x "


def v1_scoring_units(fold_ids: pd.Series, *, scheme: str = "exact", groups: pd.Series | None = None,
                     remainder_groups: Iterable[str] | None = None) -> pd.Series:
    """The registered V1 scoring unit of every scored row (section 3.2 resolution: the outer fold).

    ``scheme="exact"`` (the 104-fold design of B0-B4, B6, B7): the unit is the fold id -- the group label of a
    single-group fold, ``REMAINDER`` for the pooled remainder fold.  With ``groups`` (the row's group under the fold
    design's grouping) every non-remainder fold must hold exactly one group, equal to its fold id, and with
    ``remainder_groups`` the remainder fold may hold only those groups and no other fold any of them.

    ``scheme="grouped"`` (the ten grouped folds of B5, B8, M0-M7, fold ids ``s<seed>_f<k>``): the fold id is not a
    unit; each row takes its exact-design unit -- its group, or ``REMAINDER`` when the group is one of
    ``remainder_groups`` (the groups below 20 MODEL rows, ``folds.source_holdout.unit_of_rows``) -- so every arm is
    scored on the same units.  ``groups`` and ``remainder_groups`` are required."""
    if scheme not in V1_SCHEMES:
        raise ValueError(f"scheme must be one of {V1_SCHEMES}")
    if not isinstance(fold_ids, pd.Series):
        raise TypeError("fold_ids must be a pandas Series indexed by scored row")
    if fold_ids.isna().any():
        raise ValueError("every scored row needs a fold id")
    rem = None if remainder_groups is None else {str(g) for g in remainder_groups}
    if groups is not None:
        if not isinstance(groups, pd.Series) or not groups.index.equals(fold_ids.index):
            raise ValueError("groups must be a Series on the fold_ids index")
        if groups.isna().any():
            raise ValueError("every scored row needs a publication group")
    fid = fold_ids.astype(str)
    if scheme == "grouped":
        if groups is None or rem is None:
            raise ValueError("the grouped scheme needs groups and remainder_groups (the exact-design units)")
        g = groups.astype(str)
        return g.where(~g.isin(rem), V1_REMAINDER_UNIT).rename(V1_UNIT_COL)
    if groups is not None:
        g = groups.astype(str)
        single = fid != V1_REMAINDER_UNIT
        bad = single & (g != fid)
        if bad.any():
            raise ValueError(f"exact V1 folds: {int(bad.sum())} scored row(s) of a single-group fold carry another "
                             f"group than the fold id (first fold {fid[bad].iloc[0]!r}); pass the fold design's grouping")
        if rem is not None:
            if (single & g.isin(rem)).any():
                raise ValueError("a remainder group is scored in a single-group fold")
            if (~single & ~g.isin(rem)).any():
                raise ValueError("the remainder fold holds a group that is not a remainder group")
    return fid.rename(V1_UNIT_COL)


def registered_unit_cols(design: str) -> tuple[str, ...]:
    """The registered averaging-unit columns of ``design`` (section 4 table, section 3.2 resolution for V1)."""
    if design not in REGISTERED_UNIT_COLS:
        raise ValueError(f"design {design!r}: the averaging unit is named by the caller (V4 family, V7 unit x "
                         "direction) or is the cell pair (V5-PAIR, pairs.CELL_PAIR_COLS)")
    return REGISTERED_UNIT_COLS[design]


def with_registered_units(frame: pd.DataFrame, design: str, *, fold_col: str = FOLD_COL, v1_scheme: str = "exact",
                          v1_group_col: str | None = PUB_GROUP_COL,
                          remainder_groups: Iterable[str] | None = None) -> pd.DataFrame:
    """``frame`` with the registered unit columns present: for V1 a copy with :data:`V1_UNIT_COL` from
    :func:`v1_scoring_units` (checked against ``v1_group_col`` when that column is present); other designs are
    returned unchanged (their unit columns are row attributes)."""
    if design != "V1":
        require_columns(frame, registered_unit_cols(design))
        return frame
    require_columns(frame, [fold_col])
    groups = frame[v1_group_col] if (v1_group_col is not None and v1_group_col in frame.columns) else None
    out = frame.copy()
    out[V1_UNIT_COL] = v1_scoring_units(frame[fold_col], scheme=v1_scheme, groups=groups,
                                        remainder_groups=remainder_groups)
    return out


def unit_keys(frame: pd.DataFrame, unit_cols: Sequence[str]) -> pd.Series:
    """One text key per row of a per-unit table (``"Nd(III) x TODGA"``), used to pair arms and map clusters."""
    cols = list(unit_cols)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"missing column(s) {missing}")
    return frame[cols].astype(str).agg(UNIT_KEY_SEP.join, axis=1).rename("unit_key")


def _unit_readings(design: str, v1_group_col: str | None) -> list[tuple[str, str, tuple[str, ...]]]:
    """``(status, unit_reading label, unit columns)``: the registered reading first, then the exploratory ones."""
    out = [("registered", f"registered: {REGISTERED_UNIT_READING[design]}", registered_unit_cols(design))]
    if design == "V1" and "publication_group" in EXPLORATORY_UNIT_READINGS["V1"] and v1_group_col is not None:
        out.append(("exploratory", "exploratory: publication group (the as-run pre-seal V1 reading)", (v1_group_col,)))
    return out


def _design_summary(kind: str, frame: pd.DataFrame, regime: Regime, *, v6_mask: pd.Series | None, fold_col: str,
                    v1_scheme: str, v1_group_col: str | None, remainder_groups: Iterable[str] | None,
                    exploratory_unit_readings: bool, **kwargs: Any) -> pd.DataFrame:
    fr = with_registered_units(frame, regime.design, fold_col=fold_col, v1_scheme=v1_scheme, v1_group_col=v1_group_col,
                               remainder_groups=remainder_groups)
    fn = logd_summary if kind == "logd" else interval_summary
    parts = []
    for status, label, ucols in _unit_readings(regime.design, v1_group_col):
        if status == "exploratory" and not exploratory_unit_readings:
            continue
        reg = regime if status == "registered" else replace(regime, status="exploratory")
        s = fn(fr, reg, unit_cols=ucols, v6_mask=v6_mask, **kwargs)
        s["unit_reading"] = label
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def design_logd_summary(frame: pd.DataFrame, regime: Regime, *, v6_mask: pd.Series | None, fold_col: str = FOLD_COL,
                        v1_scheme: str = "exact", v1_group_col: str | None = PUB_GROUP_COL,
                        remainder_groups: Iterable[str] | None = None, exploratory_unit_readings: bool = True,
                        **kwargs: Any) -> pd.DataFrame:
    """:func:`logd_summary` under the design's registered averaging unit (:func:`registered_unit_cols`; for V1 the
    outer fold of :func:`v1_scoring_units`), plus the exploratory unit readings (V1: per publication group, status
    ``exploratory``).  Columns :data:`SUMMARY_COLUMNS` + ``unit_reading``."""
    return _design_summary("logd", frame, regime, v6_mask=v6_mask, fold_col=fold_col, v1_scheme=v1_scheme,
                           v1_group_col=v1_group_col, remainder_groups=remainder_groups,
                           exploratory_unit_readings=exploratory_unit_readings, **kwargs)


def design_interval_summary(frame: pd.DataFrame, regime: Regime, *, v6_mask: pd.Series | None, fold_col: str = FOLD_COL,
                            v1_scheme: str = "exact", v1_group_col: str | None = PUB_GROUP_COL,
                            remainder_groups: Iterable[str] | None = None, exploratory_unit_readings: bool = True,
                            **kwargs: Any) -> pd.DataFrame:
    """:func:`interval_summary` under the registered averaging unit (and the exploratory readings), as
    :func:`design_logd_summary`."""
    return _design_summary("interval", frame, regime, v6_mask=v6_mask, fold_col=fold_col, v1_scheme=v1_scheme,
                           v1_group_col=v1_group_col, remainder_groups=remainder_groups,
                           exploratory_unit_readings=exploratory_unit_readings, **kwargs)


def design_per_unit_table(frame: pd.DataFrame, design: str, *, v6_mask: pd.Series | None, fold_col: str = FOLD_COL,
                          v1_scheme: str = "exact", v1_group_col: str | None = PUB_GROUP_COL,
                          remainder_groups: Iterable[str] | None = None, reading: str = "registered",
                          **kwargs: Any) -> pd.DataFrame:
    """:func:`per_unit_table` under the registered unit (``reading="registered"``) or the exploratory V1
    publication-group reading (``reading="publication_group"``), indexed by :func:`unit_keys` -- the per-unit
    values a paired contrast takes.  The scored index passes the section 2 guard first."""
    guard_scoring_index(frame.index, v6_mask, design, what=f"{design} per-unit table")
    fr = with_registered_units(frame, design, fold_col=fold_col, v1_scheme=v1_scheme, v1_group_col=v1_group_col,
                               remainder_groups=remainder_groups)
    if reading == "registered":
        ucols = registered_unit_cols(design)
    elif reading == "publication_group" and design == "V1" and v1_group_col is not None:
        ucols = (v1_group_col,)
    else:
        raise ValueError(f"unknown unit reading {reading!r} for {design}")
    pu = per_unit_table(fr, unit_cols=ucols, **kwargs)
    pu.index = pd.Index(unit_keys(pu, ucols).to_numpy(), name="unit_key")
    return pu


def design_unit_clusters(frame: pd.DataFrame, design: str, *, fold_col: str = FOLD_COL, v1_scheme: str = "exact",
                         v1_group_col: str | None = PUB_GROUP_COL, remainder_groups: Iterable[str] | None = None,
                         pub_group_col: str = PUB_GROUP_COL, reading: str = "registered") -> dict[str, pd.Series]:
    """Every registered cluster unit of the design (section 8 table) as ``{name: Series(cluster label) indexed by
    unit key}`` -- the names of ``transfer.REGISTERED_CLUSTER_UNITS``.

    V5 / V5-P: ``system`` (the cell's system) and ``publication_group`` (the group holding most of the cell's scored
    rows, :func:`majority_label`); V1: ``publication_group`` -- under the registered outer-fold unit a single-group
    fold's cluster is its group and the remainder fold is one cluster (a unit is never split across clusters); under
    the exploratory reading the group itself; V2: ``metal_state``; V0: ``publication_group``; V3 / V6: ``system``."""
    fr = with_registered_units(frame, design, fold_col=fold_col, v1_scheme=v1_scheme, v1_group_col=v1_group_col,
                               remainder_groups=remainder_groups)
    if design in ("V5", "V5-P"):
        cols = list(CELL_COLS)
        require_columns(fr, cols + [pub_group_col])
        keys = fr.groupby(cols, sort=True).size().index.to_frame(index=False)
        idx = pd.Index(unit_keys(keys, cols).to_numpy(), name="unit_key")
        maj = majority_label(fr, cols, pub_group_col)
        maj.index = pd.Index(unit_keys(maj, cols).to_numpy())
        return {"system": pd.Series(keys[SYSTEM_COL].astype(str).to_numpy(), index=idx, name="system"),
                "publication_group": maj[pub_group_col].astype(str).reindex(idx).rename("publication_group")}
    if design == "V1":
        col = V1_UNIT_COL if reading == "registered" else v1_group_col
        if col is None:
            raise ValueError("the V1 publication-group reading needs v1_group_col")
        u = pd.Index(sorted(fr[col].astype(str).unique()), name="unit_key")
        return {"publication_group": pd.Series(u.to_numpy(), index=u, name="publication_group")}
    name = {"V2": "metal_state", "V0": "publication_group", "V3": "system", "V6": "system"}.get(design)
    if name is None:
        raise ValueError(f"design {design!r}: clusters are named by the caller")
    ucols = list(registered_unit_cols(design))
    u = pd.Index(sorted(unit_keys(fr.drop_duplicates(ucols), ucols).unique()), name="unit_key")
    return {name: pd.Series(u.to_numpy(), index=u, name=name)}
