"""``evaluation/pairs.py`` -- comparable pairs and the logSF / direction endpoints (sections 2, 4, 5).

Pair rule (section 2, brief section 8)
--------------------------------------
Two rows form a pair only if they are in the **same fold**, the same publication group, the same
``extractant_system_key`` and an identical ``gen19ct.data.normalize.condition_key`` (which includes metal
concentration), and they carry **different known metal states**.  Rows with an unknown state (``X(?)``)
never pair: a different state cannot be established.  Pairs are generated after fold assignment
(:func:`comparable_pairs` needs the fold column and groups by it), and ``leakage.pair_isolation_check``
runs on every generated set and again before any pair is scored (:func:`pair_summary`).

Orientation.  ``state_a`` is the heavier state: larger atomic number, then higher oxidation state (e.g.
Nd(III) before Pr(III), Pu(VI) before Pu(IV)).  ``logSF = log D_a - log D_b`` is therefore logSF_Nd/Pr for a
Pr/Nd pair, the section 3.4 convention.  MAE and direction accuracy do not depend on the orientation.

Endpoints (section 4)
---------------------
* predicted logSF = yhat_a - yhat_b of one fitted arm (:func:`derived_logsf`; any arm, lookups included);
* ``logsf_mae``; ``direction_accuracy_abs_ge_0.3`` on pairs with |observed logSF| >= 0.3 and, for design V6
  only, also ``direction_accuracy_abs_ge_0.1``.  A predicted logSF of zero (``|pred| <= 1e-12``) counts 1/2;
  every direction number carries its n;
* **FLAT** floor: predicted logSF = 0, so its MAE is mean |observed| and its direction accuracy 1/2;
* **HEAVIER** yardstick: for Ln(III)-Ln(III) pairs the heavier lanthanide is predicted more extracted;
  undefined (NaN, excluded and counted) for every other pair.  It has a direction but no magnitude, so its
  logSF MAE is not reported;
* aggregation: per averaging unit (default the unordered cell pair ``(system, state_a, state_b)``; the
  system for V6), then an equal-weight mean over units; ``pair_pooled`` and ``system_macro`` beside.

Unregistered details fixed here: the ``>= threshold`` test uses a 1e-9 tolerance; a unit with no qualifying
pair is excluded from that direction macro (its count is printed).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from gen19ct.chemistry import metals as MET
from gen19ct.data import leakage as LK
from gen19ct.evaluation import metrics as EM

FOLD_COL = "fold"
PAIR_KEY_COLS: tuple[str, ...] = (EM.PUB_GROUP_COL, EM.SYSTEM_COL, EM.CONDITION_KEY_COL)
CELL_PAIR_COLS: tuple[str, str, str] = (EM.SYSTEM_COL, "state_a", "state_b")
#: direction accuracy threshold on |observed logSF| (every design)
DIRECTION_THRESHOLD = 0.3
#: the additional threshold used for V6 only
V6_DIRECTION_THRESHOLD = 0.1
CATEGORY_CLASSES: tuple[str, ...] = ("Ln-Ln", "An-Ln", "An-An", "other")


def direction_thresholds(design: str) -> tuple[float, ...]:
    """(0.3,) for every design; (0.3, 0.1) for V6 (section 4)."""
    return (DIRECTION_THRESHOLD, V6_DIRECTION_THRESHOLD) if design == "V6" else (DIRECTION_THRESHOLD,)


def _metric_name(thr: float) -> str:
    return f"direction_accuracy_abs_ge_{thr:g}"


def parse_state(label: Any) -> tuple[str, int]:
    """``"Nd(III)" -> ("Nd", 3)``; raises on an unknown-state or malformed label."""
    if not isinstance(label, str) or "(" not in label or not label.endswith(")"):
        raise ValueError(f"not a known metal-state label: {label!r}")
    el, ox = label[:-1].split("(", 1)
    if el not in MET.ATOMIC_NUMBER:
        raise ValueError(f"unknown element in {label!r}")
    return el, MET.parse_oxidation_state(ox)


def state_order_key(label: str) -> tuple[int, int]:
    el, ox = parse_state(label)
    return MET.ATOMIC_NUMBER[el], ox


def metal_category(label: str) -> str:
    el, _ = parse_state(label)
    return "lanthanide" if el in MET.LANTHANIDES else "actinide" if el in MET.ACTINIDES else "other"


def category_class(state_a: str, state_b: str) -> str:
    """``Ln-Ln`` / ``An-Ln`` / ``An-An`` / ``other`` (any member outside the f-block series)."""
    cats = sorted([metal_category(state_a), metal_category(state_b)])
    if "other" in cats:
        return "other"
    return {("lanthanide", "lanthanide"): "Ln-Ln", ("actinide", "lanthanide"): "An-Ln",
            ("actinide", "actinide"): "An-An"}[tuple(cats)]


def comparable_pairs(df_rows: pd.DataFrame, key_cols: Sequence[str] = PAIR_KEY_COLS, *, fold_col: str = FOLD_COL,
                     state_col: str = EM.METAL_STATE_COL, y_col: str = EM.Y_COL,
                     carry_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Every comparable row pair inside one fold (section 2).

    ``df_rows`` carries a unique row index, ``fold_col`` (the fold assignment, e.g. ``train`` / ``test``),
    ``key_cols`` (publication group, system, condition key by default), ``state_col`` and ``y_col``.  Rows with
    a missing key, fold or state are never paired.  Returns one row per pair: ``idx_a`` / ``idx_b`` (index
    labels, ``a`` = heavier state), ``fold``, the key columns, ``state_a`` / ``state_b``, ``category_class``,
    ``y_a`` / ``y_b``, ``logsf_obs = y_a - y_b`` and ``<col>_a`` / ``<col>_b`` for ``carry_cols``.
    ``leakage.pair_isolation_check`` is run on the result against ``df_rows[fold_col]``."""
    if fold_col in key_cols:
        raise ValueError("fold_col is added to the grouping automatically; do not list it in key_cols")
    need = [fold_col, state_col, y_col] + list(key_cols) + list(carry_cols)
    missing = [c for c in need if c not in df_rows.columns]
    if missing:
        raise KeyError(f"comparable_pairs: missing column(s) {missing} (pairs are generated after fold assignment)")
    if df_rows.index.has_duplicates:
        raise ValueError("comparable_pairs: the row index must be unique")
    group_cols = [fold_col] + list(key_cols)
    ok = df_rows[state_col].notna().to_numpy(dtype=bool).copy()
    for c in group_cols:
        ok = ok & df_rows[c].notna().to_numpy()
    rows = df_rows[ok]
    y = pd.to_numeric(rows[y_col], errors="coerce").to_numpy(dtype=float)
    states = rows[state_col].to_numpy(dtype=object)
    order_keys = {s: state_order_key(s) for s in set(states)}
    idx = rows.index.to_numpy(dtype=object)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for _key, pos in sorted(rows.groupby(group_cols, sort=True).indices.items(), key=lambda kv: kv[1][0]):
        if len(pos) < 2:
            continue
        i, j = np.triu_indices(len(pos), k=1)
        a, b = pos[i], pos[j]
        diff = states[a] != states[b]
        if not diff.any():
            continue
        a, b = a[diff], b[diff]
        swap = np.array([order_keys[states[x]] < order_keys[states[z]] for x, z in zip(a, b)], dtype=bool)
        left.append(np.where(swap, b, a))
        right.append(np.where(swap, a, b))
    cols = ["idx_a", "idx_b", "fold", *key_cols, "state_a", "state_b", "category_class", "y_a", "y_b", "logsf_obs"]
    for c in carry_cols:
        cols += [f"{c}_a", f"{c}_b"]
    if not left:
        return pd.DataFrame(columns=cols)
    a = np.concatenate(left)
    b = np.concatenate(right)
    out = pd.DataFrame({"idx_a": idx[a], "idx_b": idx[b], "fold": rows[fold_col].to_numpy(dtype=object)[a]})
    for c in key_cols:
        out[c] = rows[c].to_numpy(dtype=object)[a]
    out["state_a"] = states[a]
    out["state_b"] = states[b]
    out["category_class"] = [category_class(s, t) for s, t in zip(states[a], states[b])]
    out["y_a"] = y[a]
    out["y_b"] = y[b]
    out["logsf_obs"] = y[a] - y[b]
    for c in carry_cols:
        vals = rows[c].to_numpy(dtype=object)
        out[f"{c}_a"] = vals[a]
        out[f"{c}_b"] = vals[b]
    LK.pair_isolation_check(out, df_rows[fold_col], member_cols=("idx_a", "idx_b"))
    return out[cols].reset_index(drop=True)


def derived_logsf(pairs: pd.DataFrame, predictions: pd.Series) -> pd.Series:
    """Predicted logSF ``yhat_a - yhat_b`` of one arm (row-indexed ``predictions``); a missing prediction raises."""
    pa = pairs["idx_a"].map(predictions)
    pb = pairs["idx_b"].map(predictions)
    bad = pa.isna() | pb.isna()
    if bad.any():
        raise ValueError(f"derived_logsf: {int(bad.sum())} pair(s) lack a prediction for a member")
    return pd.Series(pa.to_numpy(dtype=float) - pb.to_numpy(dtype=float), index=pairs.index, name="logsf_pred")


def flat_logsf(pairs: pd.DataFrame) -> pd.Series:
    """The FLAT floor: no separation, predicted logSF = 0."""
    return pd.Series(0.0, index=pairs.index, name="logsf_pred")


def heavier_direction(pairs: pd.DataFrame) -> pd.Series:
    """HEAVIER yardstick: +1 / -1 = sign(Z_a - Z_b) for Ln(III)-Ln(III) pairs, NaN (undefined) otherwise."""
    out = np.full(len(pairs), np.nan)
    for k, (sa, sb) in enumerate(zip(pairs["state_a"], pairs["state_b"])):
        ea, oa = parse_state(sa)
        eb, ob = parse_state(sb)
        if ea in MET.LANTHANIDES and eb in MET.LANTHANIDES and oa == 3 and ob == 3 and ea != eb:
            out[k] = float(np.sign(MET.ATOMIC_NUMBER[ea] - MET.ATOMIC_NUMBER[eb]))
    return pd.Series(out, index=pairs.index, name="logsf_pred")


def score_pairs(pairs: pd.DataFrame, pred_logsf: pd.Series, *, design: str, direction_only: bool = False) -> pd.DataFrame:
    """Per pair: ``logsf_pred``, ``abs_error`` (NaN when ``direction_only``), and per registered threshold
    ``dir_<thr>`` = 1 (sign agrees), 1/2 (predicted zero), 0 (sign disagrees), NaN when |observed| is below the
    threshold or the prediction is undefined."""
    if not pred_logsf.index.equals(pairs.index):
        raise ValueError("pred_logsf must be indexed like pairs")
    obs = pairs["logsf_obs"].to_numpy(dtype=float)
    pred = pred_logsf.to_numpy(dtype=float)
    if not np.isfinite(obs).all():
        raise ValueError("observed logSF must be finite")
    if not direction_only and not np.isfinite(pred).all():
        raise ValueError("a magnitude arm must predict every pair (NaN only allowed with direction_only)")
    out = pairs.copy()
    out["logsf_pred"] = pred
    out["abs_error"] = np.nan if direction_only else np.abs(pred - obs)
    for thr in direction_thresholds(design):
        qualifies = np.abs(obs) >= thr - EM.FLOAT_TOL
        s = np.where(np.abs(pred) <= EM.TIE_TOL, 0.5, (np.sign(pred) == np.sign(obs)).astype(float))
        s = np.where(np.isfinite(pred) & qualifies, s, np.nan)
        out[f"dir_{thr:g}"] = s
        out[f"dir_{thr:g}_qualifies"] = qualifies
    return out


def pair_unit_table(scored: pd.DataFrame, *, design: str, unit_cols: Sequence[str] = CELL_PAIR_COLS) -> pd.DataFrame:
    """Per averaging unit: ``n_pairs``, ``logsf_mae``, and per threshold ``n_dir_<thr>`` (qualifying pairs with a
    defined prediction), ``n_undefined_<thr>`` and ``direction_<thr>`` (NaN when ``n_dir`` = 0)."""
    codes, keys = EM.unit_index(scored, unit_cols)
    n = len(keys)
    out = keys.copy()
    out["n_pairs"] = np.bincount(codes, minlength=n).astype(int)
    ae = scored["abs_error"].to_numpy(dtype=float)
    if np.isnan(ae).all():
        out["logsf_mae"] = np.nan
    else:
        out["logsf_mae"] = np.bincount(codes, weights=ae, minlength=n) / out["n_pairs"].to_numpy(dtype=float)
    for thr in direction_thresholds(design):
        s = scored[f"dir_{thr:g}"].to_numpy(dtype=float)
        q = scored[f"dir_{thr:g}_qualifies"].to_numpy(dtype=bool)
        defined = np.isfinite(s)
        nd = np.bincount(codes, weights=defined.astype(float), minlength=n)
        tot = np.bincount(codes, weights=np.where(defined, s, 0.0), minlength=n)
        out[f"n_dir_{thr:g}"] = nd.astype(int)
        out[f"n_undefined_{thr:g}"] = np.bincount(codes, weights=(q & ~defined).astype(float), minlength=n).astype(int)
        out[f"direction_{thr:g}"] = np.divide(tot, nd, out=np.full(n, np.nan), where=nd > 0)
    return out


def pair_summary(pairs: pd.DataFrame, pred_logsf: pd.Series, regime: EM.Regime, *, v6_mask: pd.Series | None,
                 folds: pd.Series, test_label: Any = "test", unit_cols: Sequence[str] = CELL_PAIR_COLS,
                 system_col: str = EM.SYSTEM_COL, direction_only: bool = False,
                 strata_col: str | None = None) -> pd.DataFrame:
    """Tidy logSF / direction summary of one arm (or FLAT / HEAVIER) on test-test pairs.

    ``folds`` maps every row label to its fold; ``leakage.pair_isolation_check`` must pass and every pair must
    lie in ``test_label`` before anything is scored.  The union of both members passes the V6_TARGET_ROWS
    guard.  Metrics: ``logsf_mae`` (unless ``direction_only``) and the registered direction accuracies, each as
    ``unit_macro`` (primary), ``pair_pooled`` and ``system_macro`` (side).  ``n_units`` counts units entering
    the value, ``n_rows`` distinct scored rows, and ``note`` gives the pair count n."""
    LK.pair_isolation_check(pairs, folds, member_cols=("idx_a", "idx_b"))
    in_test = pairs["idx_a"].map(folds).astype(object) == test_label
    if not bool(in_test.all()):
        raise AssertionError(f"pair_summary scores only {test_label!r}-{test_label!r} pairs; "
                             f"{int((~in_test).sum())} pair(s) lie in another fold")
    scored_rows = pd.Index(pd.unique(np.concatenate([pairs["idx_a"].to_numpy(dtype=object),
                                                     pairs["idx_b"].to_numpy(dtype=object)])))
    if v6_mask is not None and len(scored_rows.difference(v6_mask.index)):
        raise ValueError(f"{regime.design}/{regime.arm} pair scoring: v6_mask does not cover "
                         f"{len(scored_rows.difference(v6_mask.index))} member label(s); a mask keyed otherwise would "
                         "pass the V6 guard vacuously")
    EM.guard_scoring_index(scored_rows, v6_mask, regime.design, what=f"{regime.design}/{regime.arm} pair scoring")
    scored_all = score_pairs(pairs, pred_logsf, design=regime.design, direction_only=direction_only)
    recs: list[dict[str, Any]] = []
    strata = [("all", scored_all)]
    if strata_col is not None:
        for val in sorted(scored_all[strata_col].astype(str).unique()):
            strata.append((f"{strata_col}={val}", scored_all[scored_all[strata_col].astype(str) == val]))
    for stratum, sc in strata:
        if not len(sc):
            continue
        members = np.concatenate([sc["idx_a"].to_numpy(dtype=object), sc["idx_b"].to_numpy(dtype=object)])
        n_rows = len(pd.unique(members))
        for agg, role, ucols in (("unit_macro", "primary", tuple(unit_cols)), ("pair_pooled", "side", ()),
                                 ("system_macro", "side", (system_col,))):
            pu = pair_unit_table(sc, design=regime.design, unit_cols=ucols)
            label = EM.unit_label(ucols) if ucols else "row_pair"
            if not direction_only:
                recs.append(EM.summary_record(regime, stratum=stratum, metric="logsf_mae", aggregation=agg,
                                              role=role, averaging_unit=label, value=EM.macro_mean(pu["logsf_mae"]),
                                              n_units=len(pu) if ucols else len(sc), n_rows=n_rows,
                                              note=f"n_pairs={len(sc)}"))
            for thr in direction_thresholds(regime.design):
                ok = pu[pu[f"n_dir_{thr:g}"] > 0]
                n_dir = int(pu[f"n_dir_{thr:g}"].sum())
                n_undef = int(pu[f"n_undefined_{thr:g}"].sum())
                recs.append(EM.summary_record(regime, stratum=stratum, metric=_metric_name(thr), aggregation=agg,
                                              role=role, averaging_unit=label,
                                              value=EM.macro_mean(ok[f"direction_{thr:g}"]),
                                              n_units=len(ok) if ucols else n_dir, n_rows=n_rows,
                                              note=f"n_pairs={n_dir}; n_undefined={n_undef}"))
    return pd.DataFrame(recs, columns=list(EM.SUMMARY_COLUMNS))
