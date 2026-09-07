"""Metrics for the level task.  The evaluation unit is the extractant, always.

The level task has one number per (split seed, held-out extractant), so there is no row
mean to confuse with a macro mean.  What *can* be confused is the cohort: the full 183
and the level-reliable 71 are different populations and every function here carries the
cohort through rather than defaulting to one.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

STATISTICS: tuple[str, ...] = ("level_abs_error", "level_signed_error")


def per_extractant(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per (split_seed, extractant), in the shape Gen12's bootstrap consumes."""
    out = predictions.rename(columns={"level_abs_error": "mae",
                                      "level_signed_error": "bias"}).copy()
    out["n_rows"] = 1
    return out[["split_seed", "extractant", "chemotype", "band", "max_train_tanimoto",
                "alpha_true", "alpha_pred", "mae", "bias", "n_rows"]]


def summarise(predictions: pd.DataFrame, *, label: str = "", cohort: str = "full") -> dict:
    """Level metrics for one arm on one cohort, averaged per seed then over seeds."""
    units = per_extractant(predictions)
    per_seed = units.groupby("split_seed")["mae"].mean()
    # Average the truth over seeds exactly as the prediction is averaged.  Under LVL_MEAN
    # the truth is seed-invariant and this is a no-op; under LVL_COND_RESIDUAL the
    # evaluation target is fold-local and taking the first seed's value while averaging the
    # prediction over five would compare two different quantities.
    truth = units.groupby("extractant")["alpha_true"].mean()
    predicted = units.groupby("extractant")["alpha_pred"].mean()
    residual = (units["alpha_pred"] - units["alpha_true"]).to_numpy(dtype=float)
    sst = float(((truth - truth.mean()) ** 2).sum())
    between = truth.reindex(predicted.index)
    sse = float((((predicted - between) ** 2)).sum())
    return {
        "arm": label, "cohort": cohort,
        "level_mae": float(per_seed.mean()),
        "level_mae_sd_over_seeds": float(per_seed.std(ddof=0)),
        "level_mae_chemotype_macro": float(
            units.groupby(["split_seed", "chemotype"])["mae"].mean()
            .groupby("split_seed").mean().mean()),
        "level_rmse": float(np.sqrt((residual ** 2).mean())),
        "level_median_abs_error": float(np.median(np.abs(residual))),
        "level_signed_bias": float(units["bias"].mean()),
        "between_extractant_r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "spearman": float(pd.Series(predicted).corr(between, method="spearman")),
        "pearson": float(pd.Series(predicted).corr(between, method="pearson")),
        "dispersion_ratio": float(predicted.std() / between.std()) if between.std() > 0 else np.nan,
        "frac_within_0_5": float(np.mean(np.abs(residual) <= 0.5)),
        "frac_within_1_0": float(np.mean(np.abs(residual) <= 1.0)),
        "n_extractants": int(units["extractant"].nunique()),
        "n_chemotypes": int(units["chemotype"].nunique()),
        "n_seeds": int(units["split_seed"].nunique()),
    }


def by_band(predictions: pd.DataFrame, *, label: str = "", cohort: str = "full",
            bands: Sequence[str] = ("far", "mid", "near")) -> pd.DataFrame:
    rows = [{**summarise(predictions, label=label, cohort=cohort), "band": "overall"}]
    for band in bands:
        block = predictions[predictions["band"] == band]
        if block.empty:
            rows.append({"arm": label, "cohort": cohort, "band": band, "n_extractants": 0,
                         "level_mae": np.nan})
            continue
        rows.append({**summarise(block, label=label, cohort=cohort), "band": band})
    return pd.DataFrame(rows)


def variance_captured(predictions: pd.DataFrame) -> dict:
    """How much of the between-extractant variance of the level a representation captures.

    The decision report needs this beside the MAE: with 63 % of the target variance
    sitting between extractants, "how much of that component is recovered zero-shot" says
    more than a pooled R2 does.
    """
    units = per_extractant(predictions)
    truth = units.groupby("extractant")["alpha_true"].mean()
    predicted = units.groupby("extractant")["alpha_pred"].mean().reindex(truth.index)
    total = float(((truth - truth.mean()) ** 2).sum())
    unexplained = float(((truth - predicted) ** 2).sum())
    return {"between_extractant_variance": total / max(len(truth) - 1, 1),
            "unexplained": unexplained / max(len(truth) - 1, 1),
            "share_captured": float(1.0 - unexplained / total) if total > 0 else np.nan,
            "n_extractants": int(len(truth))}
