#!/usr/bin/env python3
"""R2 supplement for an aggregated cross-seed OOF prediction table.

Reads ``cross_seed_oof_predictions.csv`` (as written by
``aggregate_ablation_runs.py``) and produces, per real (non-shuffled) arm:

- pooled R2, scale-free Pearson^2 and the prediction-dispersion ratio;
- guarded per-extractant R2 (groups below the row/variance floors are NaN)
  with its macro mean, median and surviving-group count;
- per-outer-fold pooled R2, which shows where a pooled number is made;
- calibration bounds for the unseen-extractant regime: oracle per-extractant
  offset and affine corrections (upper bounds, computed with test labels and
  labelled as such) plus k-shot offset calibration scored only on
  non-calibration rows.

R2 on unseen extractants is dominated by per-extractant level offsets, not
ranking failures, so these views are the honest way to read it. Nothing here
feeds selection; the primary metric remains equal-extractant macro MAE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

R2_MIN_GROUP_ROWS = 8
R2_MIN_GROUP_TARGET_STD = 0.1


def pooled_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.sum((truth - prediction) ** 2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - residual / total if total > 0.0 else float("nan")


def guarded_group_r2(
    truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray
) -> pd.Series:
    values: dict[str, float] = {}
    frame = pd.DataFrame({"g": groups, "y": truth, "p": prediction})
    for group, block in frame.groupby("g", sort=False):
        y = block["y"].to_numpy()
        if len(block) < R2_MIN_GROUP_ROWS or float(y.std()) < R2_MIN_GROUP_TARGET_STD:
            values[str(group)] = float("nan")
            continue
        values[str(group)] = pooled_r2(y, block["p"].to_numpy())
    return pd.Series(values, dtype=float)


def kshot_offset_calibration(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    shots: int,
    repeats: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean/sd of pooled R2 after k-shot per-group offset, scored on the rest."""

    scores = []
    unique_groups = np.unique(groups)
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        calibration = np.zeros(len(truth), dtype=bool)
        for group in unique_groups:
            index = np.flatnonzero(groups == group)
            if len(index) > shots:
                calibration[rng.choice(index, size=shots, replace=False)] = True
        offsets = {}
        for group in unique_groups:
            mask = calibration & (groups == group)
            offsets[group] = float((truth[mask] - prediction[mask]).mean()) if mask.any() else 0.0
        adjusted = prediction + np.array([offsets[g] for g in groups])
        holdout = ~calibration
        scores.append(pooled_r2(truth[holdout], adjusted[holdout]))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--calibration-arm", default="A2")
    args = parser.parse_args()

    table = pd.read_csv(args.predictions_csv)
    truth = table["y_true"].to_numpy(dtype=float)
    extractant = table["extractant"].astype(str).to_numpy()
    fold = table["outer_fold"].to_numpy()

    arms = sorted(
        column[len("prediction_") : -len("_mean")]
        for column in table.columns
        if column.endswith("_mean") and "SHUFFLED" not in column
    )

    rows = []
    fold_rows = []
    for arm in arms:
        prediction = table[f"prediction_{arm}_mean"].to_numpy(dtype=float)
        group_r2 = guarded_group_r2(truth, prediction, extractant)
        eligible = group_r2[np.isfinite(group_r2)]
        pearson = float(np.corrcoef(truth, prediction)[0, 1])
        rows.append(
            {
                "arm": arm,
                "pooled_r2": pooled_r2(truth, prediction),
                "pearson_squared_scale_free": pearson**2,
                "prediction_dispersion_ratio": float(prediction.std() / truth.std()),
                "equal_extractant_macro_r2": float(eligible.mean()) if len(eligible) else float("nan"),
                "median_extractant_r2": float(eligible.median()) if len(eligible) else float("nan"),
                "extractants_in_r2_metrics": int(len(eligible)),
                "extractants_total": int(table["extractant"].nunique()),
            }
        )
        for fold_id in np.unique(fold):
            mask = fold == fold_id
            fold_rows.append(
                {
                    "arm": arm,
                    "outer_fold": int(fold_id),
                    "n_rows": int(mask.sum()),
                    "n_extractants": int(len(set(extractant[mask]))),
                    "pooled_r2": pooled_r2(truth[mask], prediction[mask]),
                }
            )

    overall = pd.DataFrame(rows).sort_values("pooled_r2", ascending=False)
    per_fold = pd.DataFrame(fold_rows)

    calibration_rows = []
    calibration_arm = args.calibration_arm
    prediction = table[f"prediction_{calibration_arm}_mean"].to_numpy(dtype=float)
    residual = pd.DataFrame({"e": extractant, "r": truth - prediction})
    oracle_offset = prediction + residual.groupby("e")["r"].transform("mean").to_numpy()
    calibration_rows.append(
        {
            "scheme": "uncorrected",
            "pooled_r2": pooled_r2(truth, prediction),
            "upper_bound_only": False,
        }
    )
    calibration_rows.append(
        {
            "scheme": "oracle_per_extractant_offset",
            "pooled_r2": pooled_r2(truth, oracle_offset),
            "upper_bound_only": True,
        }
    )
    affine = prediction.copy()
    for group, block in pd.DataFrame(
        {"e": extractant, "y": truth, "p": prediction}
    ).groupby("e"):
        p = block["p"].to_numpy()
        if len(block) >= 3 and p.std() > 1e-9:
            slope, intercept = np.polyfit(p, block["y"].to_numpy(), 1)
            affine[extractant == group] = slope * prediction[extractant == group] + intercept
    calibration_rows.append(
        {
            "scheme": "oracle_per_extractant_affine",
            "pooled_r2": pooled_r2(truth, affine),
            "upper_bound_only": True,
        }
    )
    for shots in (2, 5, 10):
        mean, sd = kshot_offset_calibration(
            truth, prediction, extractant, shots=shots
        )
        calibration_rows.append(
            {
                "scheme": f"{shots}_shot_offset_scored_on_rest",
                "pooled_r2": mean,
                "pooled_r2_sd": sd,
                "upper_bound_only": False,
            }
        )
    calibration = pd.DataFrame(calibration_rows)

    output_dir = args.output_dir or args.predictions_csv.parent / "r2_supplement"
    output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output_dir / "arm_r2.csv", index=False)
    per_fold.to_csv(output_dir / "per_fold_r2.csv", index=False)
    calibration.to_csv(output_dir / "calibration_bounds.csv", index=False)

    print(overall.head(10).to_string(index=False))
    print()
    print(calibration.to_string(index=False))
    print(f"\nwritten to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
