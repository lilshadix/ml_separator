"""Metrics.  Every number carries its evaluation unit in its name.

Three rules this module enforces rather than documents:

* **A row mean is never called ``mae``.**  It is ``pooled_mae``.  The headline is
  ``macro_mae_extractant``: the mean over held-out extractants of the MAE within
  each extractant, one vote per extractant regardless of how many times it was
  measured.  On this cohort a single extractant is 23 % of the rows, so the two
  differ systematically.
* **Centring is per (split_seed, extractant).**  gen11 measured that centring a
  ligand's residuals on a mean pooled across split seeds scores between-seed
  level wobble as *shape*, and that this reached a published headline number
  (a shape effect of +0.0119 collapsed to +0.005 once corrected).  Every
  decomposition here therefore groups by seed first.
* **A shape statistic needs at least two rows.**  A one-row extractant would
  contribute a free zero.  Such extractants are excluded from shape statistics,
  counted, and still contribute their level error.

The exact identity ``SSE_total = SSE_centred + sum_l n_l b_l^2`` holds per
(seed, extractant) and is unit-tested.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

TARGET = "log_D"
PREDICTION = "prediction"
MIN_ROWS_FOR_SHAPE = 2
MIN_ROWS_FOR_RANK = 3


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < MIN_ROWS_FOR_RANK or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def per_extractant(frame: pd.DataFrame, *, prediction: str = PREDICTION,
                   target: str = TARGET) -> pd.DataFrame:
    """One row per (split_seed, extractant): MAE, bias, shape, counts, band.

    This is the atom of every headline table and of the bootstrap.  Grouping by
    ``split_seed`` first is not optional — see the module docstring.
    """
    required = {"split_seed", "extractant", target, prediction}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"per_extractant needs {sorted(missing)}")
    carry = [c for c in ("chemotype", "ecfp_cluster", "band", "max_train_tanimoto", "fold")
             if c in frame.columns]
    rows: list[dict] = []
    for (seed, extractant), block in frame.groupby(["split_seed", "extractant"], sort=True):
        y = block[target].to_numpy(dtype=float)
        p = block[prediction].to_numpy(dtype=float)
        if not np.isfinite(p).all():
            raise ValueError(f"non-finite prediction for {extractant!r} at seed {seed}")
        residual = p - y
        bias = float(residual.mean())
        n = len(y)
        shaped = n >= MIN_ROWS_FOR_SHAPE
        yc, pc = y - y.mean(), p - p.mean()
        record = {
            "split_seed": int(seed), "extractant": str(extractant), "n_rows": n,
            "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt((residual ** 2).mean())),
            "median_abs_error": float(np.median(np.abs(residual))),
            "bias": bias, "offset_abs": abs(bias),
            "shape_mae": float(np.abs(pc - yc).mean()) if shaped else np.nan,
            "sse": float((residual ** 2).sum()),
            "sse_centred": float(((pc - yc) ** 2).sum()),
            "sse_offset": float(n * bias ** 2),
            "sst_centred": float((yc ** 2).sum()),
            "spearman": _spearman(y, p),
            "y_sd": float(y.std(ddof=0)),
        }
        for column in carry:
            values = block[column].dropna().unique()
            record[column] = values[0] if len(values) == 1 else (
                float(np.mean(block[column])) if pd.api.types.is_numeric_dtype(block[column])
                else ";".join(sorted(map(str, values))))
        rows.append(record)
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, *, prediction: str = PREDICTION, target: str = TARGET,
              label: str = "") -> dict:
    """Every headline metric for one arm on one set of rows.

    Macro metrics are computed per seed and then averaged over seeds, so a seed
    that happens to hold more rows cannot outvote another.
    """
    units = per_extractant(frame, prediction=prediction, target=target)
    per_seed_ext = units.groupby("split_seed")["mae"].mean()
    chem = frame.copy()
    chem["_ae"] = (chem[prediction] - chem[target]).abs()
    per_seed_chem = (chem.groupby(["split_seed", "chemotype"])["_ae"].mean()
                     .groupby("split_seed").mean()) if "chemotype" in chem.columns else pd.Series(dtype=float)
    residual = (frame[prediction] - frame[target]).to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    p = frame[prediction].to_numpy(dtype=float)
    sst = float(((y - y.mean()) ** 2).sum())
    shaped = units[units["n_rows"] >= MIN_ROWS_FOR_SHAPE]
    slope, intercept = (np.polyfit(p, y, 1) if np.ptp(p) > 0 else (np.nan, np.nan))
    out = {
        "arm": label,
        "macro_mae_extractant": float(per_seed_ext.mean()),
        "macro_mae_extractant_sd_over_seeds": float(per_seed_ext.std(ddof=0)),
        "macro_mae_chemotype": float(per_seed_chem.mean()) if len(per_seed_chem) else np.nan,
        "pooled_mae": float(np.abs(residual).mean()),
        "pooled_rmse": float(np.sqrt((residual ** 2).mean())),
        "median_abs_error": float(np.median(np.abs(residual))),
        "pooled_r2": float(1.0 - (residual ** 2).sum() / sst) if sst > 0 else np.nan,
        "pooled_spearman": _spearman(y, p),
        "calibration_slope": float(slope), "calibration_intercept": float(intercept),
        "offset_mae": float(units["offset_abs"].mean()),
        "shape_mae": float(shaped["shape_mae"].mean()) if len(shaped) else np.nan,
        "mean_bias": float(units["bias"].mean()),
        "frac_within_0_5_log": float(np.mean(np.abs(residual) <= 0.5)),
        "frac_within_1_log": float(np.mean(np.abs(residual) <= 1.0)),
        "n_rows": int(len(frame)),
        "n_extractant_units": int(units["extractant"].nunique()),
        "n_chemotypes": int(frame["chemotype"].nunique()) if "chemotype" in frame.columns else 0,
        "n_seeds": int(frame["split_seed"].nunique()),
        "n_extractants_with_shape": int(shaped["extractant"].nunique()),
    }
    return out


def by_band(frame: pd.DataFrame, *, prediction: str = PREDICTION, target: str = TARGET,
            label: str = "", bands: Sequence[str] = ("far", "mid", "near")) -> pd.DataFrame:
    """The same summary restricted to each similarity band, plus ``overall``.

    A band with no rows yields a row of NaN with ``n_rows = 0`` rather than being
    silently omitted, so a missing band is visible in the table.
    """
    if "band" not in frame.columns:
        raise KeyError("frame has no band column; attach it from the similarity table")
    rows = [{**summarise(frame, prediction=prediction, target=target, label=label),
             "band": "overall"}]
    for band in bands:
        block = frame[frame["band"] == band]
        if block.empty:
            rows.append({"arm": label, "band": band, "n_rows": 0,
                         "macro_mae_extractant": np.nan, "n_extractant_units": 0})
            continue
        rows.append({**summarise(block, prediction=prediction, target=target, label=label),
                     "band": band})
    return pd.DataFrame(rows)


def effective_sample_size(counts) -> float:
    """Kish n_eff — how many independent units a weighted mean really rests on."""
    w = np.asarray(list(counts), dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.sum() > 0 else np.nan
