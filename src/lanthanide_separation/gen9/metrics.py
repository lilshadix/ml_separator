"""Shape metrics — what "the response surface became more realistic" means numerically.

Macro MAE cannot answer gen9's question.  A model that shifts every unseen ligand's
level a little closer while still drawing all of its titrations flat improves macro
MAE and has fixed nothing; a model that recovers the true slopes but misplaces the
level looks *worse* on macro MAE while being the more useful object, because one
measurement fixes a level and no number of measurements fixes a shape the model
cannot express.  So gen9 pre-registers eight shape endpoints and requires them to
move **coherently** (brief §19) before any flattening claim is made.

Every metric here is computed per *curve*, on held-out chemotypes, from the same
``curve_membership`` geometry gen8 built — so a gen9 number and a gen8 number are
the same arithmetic on the same rows.

The one place a definition had to be chosen rather than inherited:

``span_recovery = predicted_span / true_span``
    is unstable when the true span is small — a curve spanning 0.02 log units is
    noise, and dividing by it manufactures ratios of 30 as easily as 0.03.  The
    headline is therefore the **median over curves with a true span of at least
    :data:`MIN_TRUE_SPAN`** (0.5 log units, declared before any arm was scored),
    with the unguarded value reported beside it so the guard's effect is visible.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

#: A curve whose measured range is below this carries no dynamic range to recover.
MIN_TRUE_SPAN = 0.5

#: A curve needs this many points before a slope is worth quoting.  gen8's
#: ``gen8_slopes.py`` uses 4; gen9 keeps 4 for the headline so the tables line up,
#: and reports 3 as a coverage sensitivity.
MIN_SLOPE_POINTS = 4


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _linear_r2(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    """How close the curve is to a straight line on the supervised axis.

    The measured titrations are very nearly linear in log concentration (gen8 median
    R2 0.994 on the extractant axis), so a *predicted* curve's linearity is a direct
    readout of whether the model expresses the response as a slope or as a step.
    """
    total = float(((y - y.mean()) ** 2).sum())
    if total <= 0:
        return float("nan")
    residual = y - (slope * x + intercept)
    return float(1.0 - (residual ** 2).sum() / total)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def _sign_accuracy(x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of within-curve row pairs whose predicted ordering matches truth.

    Ties in truth are dropped; ties in prediction score 0.5, so a flat predictor
    scores exactly 0.5 rather than being rewarded for refusing to commit.
    """
    n = len(x)
    if n < 2:
        return float("nan")
    i, j = np.triu_indices(n, k=1)
    dt = y_true[j] - y_true[i]
    dp = y_pred[j] - y_pred[i]
    keep = dt != 0
    if not keep.any():
        return float("nan")
    dt, dp = dt[keep], dp[keep]
    score = np.where(dp == 0, 0.5, (np.sign(dp) == np.sign(dt)).astype(float))
    return float(score.mean())


def curve_shape_table(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    truth_column: str = "log_D",
    prediction_column: str = "prediction",
    min_points: int = MIN_SLOPE_POINTS,
    extra_columns: Sequence[str] = ("extractant", "tanimoto_cluster", "split_seed"),
) -> pd.DataFrame:
    """One row per curve: true and predicted slope, span, shape error, ordering.

    ``frame`` must carry ``row_id``, the truth column and the prediction column and
    must be unique on ``row_id`` (call it once per model × split seed).
    """
    if frame["row_id"].duplicated().any():
        raise ValueError("frame must be unique on row_id — group by model and split_seed first")
    indexed = frame.set_index("row_id")
    truth_all = indexed[truth_column]
    pred_all = indexed[prediction_column]
    meta = {c: indexed[c] for c in extra_columns if c in indexed.columns}

    present = membership[membership["row_id"].isin(indexed.index)]
    records: list[dict] = []
    for curve_id, block in present.groupby("curve_id", sort=True):
        row_ids = block["row_id"].to_numpy()
        x = block["axis_value"].to_numpy(dtype=float)
        yt = truth_all.reindex(row_ids).to_numpy(dtype=float)
        yp = pred_all.reindex(row_ids).to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(yt) & np.isfinite(yp)
        x, yt, yp, row_ids = x[ok], yt[ok], yp[ok], row_ids[ok]
        if len(x) < min_points or np.ptp(x) <= 0:
            continue
        order = np.argsort(x, kind="stable")
        x, yt, yp = x[order], yt[order], yp[order]
        slope_true, intercept_true = _fit_line(x, yt)
        slope_pred, intercept_pred = _fit_line(x, yp)
        span_true = float(np.ptp(yt))
        span_pred = float(np.ptp(yp))
        # How *smooth* is the predicted curve?  An axis-aligned tree can only draw a
        # titration as a staircase, and a staircase can reach the right range with
        # one tall step in the wrong place.  Distinguishing "the model got steeper
        # everywhere" from "the model grew one big step" needs the step count and
        # the linearity of the predicted curve, not just its slope and range.
        n_distinct_pred = int(np.unique(np.round(yp, 9)).size)
        n_distinct_true = int(np.unique(np.round(yt, 9)).size)
        linear_r2_pred = _linear_r2(x, yp, slope_pred, intercept_pred)
        linear_r2_true = _linear_r2(x, yt, slope_true, intercept_true)
        # Shape error is the residual after removing each curve's own mean, which
        # is the level a single measurement would supply.  It is therefore "what
        # is left after the experiment you are going to do anyway".
        shape_mae = float(np.abs((yp - yp.mean()) - (yt - yt.mean())).mean())
        record = {
            "curve_id": str(curve_id),
            "axis_label": str(block["axis_label"].iloc[0]),
            "series_id": str(block["series_id"].iloc[0]),
            "n_points": int(len(x)),
            "x_span": float(np.ptp(x)),
            "slope_true": slope_true, "slope_pred": slope_pred,
            "slope_error": slope_pred - slope_true,
            "slope_abs_error": abs(slope_pred - slope_true),
            "span_true": span_true, "span_pred": span_pred,
            "span_recovery": span_pred / span_true if span_true > 0 else float("nan"),
            "slope_recovery": slope_pred / slope_true if abs(slope_true) > 1e-9 else float("nan"),
            "shape_mae": shape_mae,
            "row_mae": float(np.abs(yp - yt).mean()),
            "spearman": _spearman(yt, yp),
            "sign_accuracy": _sign_accuracy(x, yt, yp),
            "sd_true": float(np.std(yt)), "sd_pred": float(np.std(yp)),
            "n_distinct_pred": n_distinct_pred, "n_distinct_true": n_distinct_true,
            "step_share_pred": n_distinct_pred / len(x),
            "linear_r2_pred": linear_r2_pred, "linear_r2_true": linear_r2_true,
        }
        for name, series in meta.items():
            record[name] = series.reindex([row_ids[0]]).iloc[0]
        records.append(record)
    if not records:
        return pd.DataFrame(columns=["curve_id", "axis_label", "slope_true", "slope_pred"])
    return pd.DataFrame.from_records(records)


def summarise_shape(table: pd.DataFrame, *, keys: Sequence[str] = ("axis_label",),
                    min_true_span: float = MIN_TRUE_SPAN) -> pd.DataFrame:
    """The eight endpoints of brief §8, one row per group.

    ``*_guarded`` columns restrict to curves with a real dynamic range; the
    unguarded twin is kept beside every one of them so nobody has to take the
    guard on trust.
    """
    if table.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for key, block in table.groupby(list(keys), sort=True):
        values = key if isinstance(key, tuple) else (key,)
        guarded = block[block["span_true"] >= min_true_span]
        slope_ok = block[np.abs(block["slope_true"]) >= 1e-6]
        record = dict(zip(keys, values))
        record.update({
            "n_curves": int(len(block)),
            "n_curves_guarded": int(len(guarded)),
            "n_ligands": int(block["extractant"].nunique()) if "extractant" in block else np.nan,
            "slope_true_median": float(block["slope_true"].median()),
            "slope_pred_median": float(block["slope_pred"].median()),
            "slope_mae": float(block["slope_abs_error"].mean()),
            "slope_mae_median": float(block["slope_abs_error"].median()),
            "slope_ratio_median": float(slope_ok["slope_recovery"].median())
            if len(slope_ok) else float("nan"),
            "slope_pearson": float(block[["slope_true", "slope_pred"]].corr().iloc[0, 1])
            if len(block) > 2 else float("nan"),
            "slope_spearman": _spearman(block["slope_true"].to_numpy(),
                                        block["slope_pred"].to_numpy()),
            "slope_sign_agree": float((np.sign(block["slope_true"])
                                       == np.sign(block["slope_pred"])).mean()),
            "span_true_median": float(block["span_true"].median()),
            "span_pred_median": float(block["span_pred"].median()),
            "span_recovery_median": float(guarded["span_recovery"].median())
            if len(guarded) else float("nan"),
            "span_recovery_median_unguarded": float(block["span_recovery"].median()),
            "span_recovery_q25": float(guarded["span_recovery"].quantile(0.25))
            if len(guarded) else float("nan"),
            "span_recovery_q75": float(guarded["span_recovery"].quantile(0.75))
            if len(guarded) else float("nan"),
            "compression_median": float((guarded["sd_pred"] / guarded["sd_true"].replace(0, np.nan)).median())
            if len(guarded) else float("nan"),
            "shape_mae": float(block["shape_mae"].mean()),
            "shape_mae_median": float(block["shape_mae"].median()),
            "within_curve_spearman": float(block["spearman"].mean()),
            "within_curve_sign_accuracy": float(block["sign_accuracy"].mean()),
            "row_mae": float(block["row_mae"].mean()),
            "n_distinct_pred_median": float(block["n_distinct_pred"].median()),
            "n_distinct_true_median": float(block["n_distinct_true"].median()),
            "step_share_pred_median": float(block["step_share_pred"].median()),
            "linear_r2_pred_median": float(block["linear_r2_pred"].median()),
            "linear_r2_true_median": float(block["linear_r2_true"].median()),
        })
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def slope_distribution(table: pd.DataFrame, *, keys: Sequence[str] = ("axis_label",)) -> pd.DataFrame:
    """Brief §21: is flattening being replaced by over-steepening?

    Reports the true and predicted slope quantiles side by side, plus the share of
    curves whose predicted slope magnitude exceeds the corpus's own 95th percentile
    of *measured* slopes on that axis — an over-steepening rate defined from the
    data rather than from a clipping constant chosen after seeing results.
    """
    if table.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for key, block in table.groupby(list(keys), sort=True):
        values = key if isinstance(key, tuple) else (key,)
        true_hi = float(np.nanquantile(np.abs(block["slope_true"]), 0.95))
        record = dict(zip(keys, values))
        for tag, column in (("true", "slope_true"), ("pred", "slope_pred")):
            series = block[column].to_numpy(dtype=float)
            record.update({
                f"{tag}_median": float(np.nanmedian(series)),
                f"{tag}_q05": float(np.nanquantile(series, 0.05)),
                f"{tag}_q25": float(np.nanquantile(series, 0.25)),
                f"{tag}_q75": float(np.nanquantile(series, 0.75)),
                f"{tag}_q95": float(np.nanquantile(series, 0.95)),
                f"{tag}_absmax": float(np.nanmax(np.abs(series))),
            })
        record["n_curves"] = int(len(block))
        record["true_abs_q95"] = true_hi
        record["frac_pred_steeper_than_true_q95"] = float(
            (np.abs(block["slope_pred"]) > true_hi).mean())
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def dynamic_range(frame: pd.DataFrame, *, truth_column: str = "log_D",
                  prediction_column: str = "prediction",
                  group: str = "extractant") -> pd.DataFrame:
    """Per-ligand compression: predicted spread against measured spread."""
    out = frame.groupby(group).agg(
        n_rows=(truth_column, "size"),
        sd_true=(truth_column, "std"), sd_pred=(prediction_column, "std"),
        span_true=(truth_column, lambda s: float(np.ptp(s)) if len(s) > 1 else np.nan),
        span_pred=(prediction_column, lambda s: float(np.ptp(s)) if len(s) > 1 else np.nan),
    ).reset_index()
    out["sd_ratio"] = out["sd_pred"] / out["sd_true"].replace(0, np.nan)
    out["span_ratio"] = out["span_pred"] / out["span_true"].replace(0, np.nan)
    return out


def macro_mae(frame: pd.DataFrame, *, truth_column: str = "log_D",
              prediction_column: str = "prediction",
              unit: str = "ecfp_cluster") -> float:
    """One unit, one vote — the gen5..gen8 headline metric, unchanged."""
    error = np.abs(frame[prediction_column].to_numpy(dtype=float)
                   - frame[truth_column].to_numpy(dtype=float))
    return float(pd.Series(error).groupby(frame[unit].to_numpy()).mean().mean())
