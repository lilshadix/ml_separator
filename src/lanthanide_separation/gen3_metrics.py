"""Predeclared generation-3 metrics and paired extractant inference."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .pairs import PAIR_TARGET_COLUMN


# Per-extractant R2 is undefined when the group's target barely varies: with
# SS_tot near zero the statistic explodes to arbitrarily large negative values
# (observed: -269 on a 91-row extractant with target sd 0.05) and one such
# group dominates any average. Groups below these floors report NaN and are
# excluded from macro/median R2, with the surviving count published alongside.
R2_MIN_GROUP_ROWS = 8
R2_MIN_GROUP_TARGET_STD = 0.1


def _guarded_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < R2_MIN_GROUP_ROWS or float(np.std(truth)) < R2_MIN_GROUP_TARGET_STD:
        return float("nan")
    return float(r2_score(truth, prediction))


def equal_group_macro_mae(
    truth: Iterable[float], prediction: Iterable[float], groups: Iterable[Any]
) -> float:
    table = pd.DataFrame(
        {
            "group": np.asarray(list(groups), dtype=object).astype(str),
            "absolute_error": np.abs(
                np.asarray(list(truth), dtype=float)
                - np.asarray(list(prediction), dtype=float)
            ),
        }
    )
    if table.empty:
        raise ValueError("Cannot compute macro MAE on zero rows.")
    return float(table.groupby("group", sort=False)["absolute_error"].mean().mean())


def per_extractant_metric_table(
    predictions: pd.DataFrame,
    arms: Iterable[str],
    *,
    baseline_arm: str = "A2_current_champion",
) -> pd.DataFrame:
    arms_tuple = tuple(arms)
    rows: list[dict[str, Any]] = []
    for extractant, group in predictions.groupby("extractant", sort=False):
        truth = group[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        baseline_mae = float(
            mean_absolute_error(
                truth, group[f"prediction_{baseline_arm}"].to_numpy(dtype=float)
            )
        )
        for arm in arms_tuple:
            prediction = group[f"prediction_{arm}"].to_numpy(dtype=float)
            mae = float(mean_absolute_error(truth, prediction))
            rows.append(
                {
                    "extractant": str(extractant),
                    "extractant_family": str(group["extractant_family"].iloc[0]),
                    "arm": arm,
                    "n_rows": int(len(group)),
                    "mae": mae,
                    "r2": _guarded_r2(truth, prediction),
                    "bias_mean_residual": float(np.mean(truth - prediction)),
                    "sign_accuracy": float(
                        np.mean(np.sign(truth) == np.sign(prediction))
                    ),
                    "baseline_mae": baseline_mae,
                    "delta_mae_baseline_minus_candidate": baseline_mae - mae,
                    "improved_vs_baseline": bool(mae < baseline_mae),
                }
            )
    return pd.DataFrame(rows)


def arm_metric_table(
    predictions: pd.DataFrame,
    arms: Iterable[str],
    *,
    baseline_arm: str = "A2_current_champion",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return overall, per-extractant, and extractant-family summaries."""

    arms_tuple = tuple(arms)
    per_extractant = per_extractant_metric_table(
        predictions, arms_tuple, baseline_arm=baseline_arm
    )
    truth = predictions[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
    adjacent = (
        predictions["pair__Z_B"].to_numpy(dtype=float)
        - predictions["pair__Z_A"].to_numpy(dtype=float)
    ) == 1.0
    rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for arm in arms_tuple:
        prediction = predictions[f"prediction_{arm}"].to_numpy(dtype=float)
        group_frame = per_extractant[per_extractant["arm"].eq(arm)].copy()
        sorted_group_mae = np.sort(group_frame["mae"].to_numpy(dtype=float))
        worst_count = max(1, int(np.ceil(0.25 * len(sorted_group_mae))))
        group_r2 = group_frame["r2"].to_numpy(dtype=float)
        eligible_r2 = group_r2[np.isfinite(group_r2)]
        pearson = float(np.corrcoef(truth, prediction)[0, 1])
        truth_std = float(np.std(truth))
        prediction_std = float(np.std(prediction))
        rows.append(
            {
                "arm": arm,
                "n_rows": int(len(predictions)),
                "n_extractants": int(len(group_frame)),
                "equal_extractant_macro_mae": float(group_frame["mae"].mean()),
                "pooled_micro_mae": float(mean_absolute_error(truth, prediction)),
                "median_extractant_mae": float(group_frame["mae"].median()),
                "worst_quartile_extractant_mae": float(
                    sorted_group_mae[-worst_count:].mean()
                ),
                "fraction_extractants_improved_vs_A2": float(
                    group_frame["improved_vs_baseline"].mean()
                ),
                "extractants_improved_vs_A2": int(
                    group_frame["improved_vs_baseline"].sum()
                ),
                "pooled_r2": float(r2_score(truth, prediction)),
                # Equal-extractant R2 view: mean/median of per-extractant R2
                # over groups above the variance/row floors (degenerate groups
                # are NaN in the per-extractant table and excluded here).
                "equal_extractant_macro_r2": float(eligible_r2.mean())
                if len(eligible_r2)
                else float("nan"),
                "median_extractant_r2": float(np.median(eligible_r2))
                if len(eligible_r2)
                else float("nan"),
                "extractants_in_r2_metrics": int(len(eligible_r2)),
                # Scale-free calibration diagnostics: pooled_r2 can move on
                # dispersion alone, so any pooled_r2 gain must be read next to
                # these (a gain that vanishes under pearson_squared is
                # recalibration, not information).
                "pearson_squared_scale_free": float(pearson**2),
                "prediction_dispersion_ratio": (
                    prediction_std / truth_std if truth_std > 0.0 else float("nan")
                ),
                "adjacent_ln_mae": float(
                    mean_absolute_error(truth[adjacent], prediction[adjacent])
                )
                if bool(adjacent.any())
                else float("nan"),
                "nonadjacent_ln_mae": float(
                    mean_absolute_error(truth[~adjacent], prediction[~adjacent])
                )
                if bool((~adjacent).any())
                else float("nan"),
                "sign_accuracy": float(
                    np.mean(np.sign(truth) == np.sign(prediction))
                ),
                "pooled_statistics_are_row_weighted": True,
            }
        )
        for family, family_frame in group_frame.groupby(
            "extractant_family", sort=True
        ):
            family_rows.append(
                {
                    "arm": arm,
                    "extractant_family": str(family),
                    "n_extractants": int(len(family_frame)),
                    "macro_mae": float(family_frame["mae"].mean()),
                    "fraction_extractants_improved_vs_A2": float(
                        family_frame["improved_vs_baseline"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows), per_extractant, pd.DataFrame(family_rows)


def paired_extractant_bootstrap(
    predictions: pd.DataFrame,
    comparisons: Mapping[str, tuple[str, str]],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Paired cluster bootstrap preserving repeated-extractant multiplicity."""

    if int(replicates) < 1:
        raise ValueError("Bootstrap replicates must be positive.")
    extractants = predictions["extractant"].astype(str).drop_duplicates().to_numpy()
    if not len(extractants):
        raise ValueError("No extractants are available for bootstrap.")
    group_delta: dict[str, np.ndarray] = {}
    improved_counts: dict[str, int] = {}
    for name, (reference, candidate) in comparisons.items():
        deltas = []
        for extractant in extractants:
            group = predictions[predictions["extractant"].astype(str).eq(extractant)]
            truth = group[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
            reference_mae = mean_absolute_error(
                truth, group[f"prediction_{reference}"].to_numpy(dtype=float)
            )
            candidate_mae = mean_absolute_error(
                truth, group[f"prediction_{candidate}"].to_numpy(dtype=float)
            )
            deltas.append(float(reference_mae - candidate_mae))
        group_delta[name] = np.asarray(deltas, dtype=float)
        improved_counts[name] = int(np.sum(group_delta[name] > 0.0))

    rng = np.random.default_rng(int(seed))
    draws = {
        name: np.empty(int(replicates), dtype=float) for name in comparisons
    }
    for replicate in range(int(replicates)):
        # Index draws preserve multiplicity automatically: an extractant drawn
        # twice contributes two equal-weight votes to this bootstrap replicate.
        sampled = rng.integers(0, len(extractants), size=len(extractants))
        for name in comparisons:
            draws[name][replicate] = float(group_delta[name][sampled].mean())

    rows: list[dict[str, Any]] = []
    for name, (reference, candidate) in comparisons.items():
        values = draws[name]
        rows.append(
            {
                "comparison": name,
                "reference": reference,
                "candidate": candidate,
                "point_delta_mae": float(group_delta[name].mean()),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "p_candidate_better": float(np.mean(values > 0.0)),
                "extractants_improved": improved_counts[name],
                "extractants_total": int(len(extractants)),
                "bootstrap_replicates": int(replicates),
                "bootstrap_seed": int(seed),
                "bootstrap_unit": "extractant",
                "positive_delta_means_improvement": True,
                "multiplicity_preserved": True,
            }
        )
    return pd.DataFrame(rows)

