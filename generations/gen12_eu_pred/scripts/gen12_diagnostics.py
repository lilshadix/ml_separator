#!/usr/bin/env python
"""Diagnostics the decision report cites: calibration, shrinkage, influence, strata.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import inference, paths  # noqa: E402
from gen12eu.metrics import per_extractant  # noqa: E402

TARGET = "log_D"
ARMS = {
    "ABL_A_CONDITIONS": paths.PREDICTION_DIR / "B_ablation" / "ABL_A_CONDITIONS.parquet",
    "ABL_C_MOL_COND": paths.PREDICTION_DIR / "B_ablation" / "ABL_C_MOL_COND.parquet",
    "ABL_D_PLUS_LIG2D": paths.PREDICTION_DIR / "B_ablation" / "ABL_D_PLUS_LIG2D.parquet",
    "T3_DMPNN_COND": paths.PREDICTION_DIR / "B" / "T3_DMPNN_COND.parquet",
}


def main() -> int:
    frames = {a: pd.read_parquet(p) for a, p in ARMS.items() if p.exists()}

    rows = []
    for arm, frame in frames.items():
        y, p = frame[TARGET].to_numpy(float), frame["prediction"].to_numpy(float)
        slope, intercept = np.polyfit(p, y, 1)
        rows.append({"arm": arm, "sd_truth": y.std(), "sd_prediction": p.std(),
                     "dispersion_ratio": p.std() / y.std(),
                     "calibration_slope": slope, "calibration_intercept": intercept,
                     "pooled_spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman"))})
    calibration = pd.DataFrame(rows)
    calibration.to_csv(paths.METRIC_DIR / "B" / "calibration.csv", index=False)

    units = {a: per_extractant(f) for a, f in frames.items()}
    per_unit = inference.unit_table(units, statistic="mae")
    influence = []
    for candidate in [a for a in frames if a != "ABL_A_CONDITIONS"]:
        table = inference.leave_one_chemotype_out(per_unit, "ABL_A_CONDITIONS", candidate)
        table["candidate"] = candidate
        table["sign_flips"] = np.sign(table["delta_without"]) != np.sign(table["delta_full"])
        influence.append(table)
    influence = pd.concat(influence, ignore_index=True)
    influence.to_csv(paths.BOOTSTRAP_DIR / "B_influence_hypothesis.csv", index=False)

    # data-quality strata: does the review queue mark the rows the model gets wrong?
    strata = []
    for arm, frame in frames.items():
        block = frame.assign(flagged=frame["review_flags"].astype(str) != "",
                             ae=(frame["prediction"] - frame[TARGET]).abs())
        for flagged, piece in block.groupby("flagged"):
            per = piece.groupby(["split_seed", "extractant"])["ae"].mean()
            strata.append({"arm": arm, "review_flagged": bool(flagged),
                           "n_rows": int(len(piece) / frame["split_seed"].nunique()),
                           "n_extractants": int(piece["extractant"].nunique()),
                           "macro_mae": float(per.groupby(level=0).mean().mean()),
                           "pooled_mae": float(piece["ae"].mean())})
    strata = pd.DataFrame(strata)
    strata.to_csv(paths.METRIC_DIR / "B" / "review_flag_strata.csv", index=False)

    # worst extractants for the champion — named, so a reader can check them
    champion = "ABL_D_PLUS_LIG2D" if "ABL_D_PLUS_LIG2D" in units else next(iter(units))
    worst = (units[champion].groupby("extractant")
             .agg(mae=("mae", "mean"), bias=("bias", "mean"), n_rows=("n_rows", "first"),
                  similarity=("max_train_tanimoto", "mean"), band=("band", "first"),
                  chemotype=("chemotype", "first"))
             .sort_values("mae", ascending=False).head(15).reset_index())
    worst.to_csv(paths.METRIC_DIR / "B" / "worst_extractants.csv", index=False)

    print(calibration.round(4).to_string(index=False))
    print("\ninfluence — does any single chemotype flip a sign?")
    print(influence.groupby("candidate")["sign_flips"].sum().to_string())
    print(influence.sort_values("influence", key=np.abs, ascending=False)
          .head(6)[["candidate", "chemotype_removed", "n_units_removed", "delta_full",
                    "delta_without", "influence"]].round(4).to_string(index=False))
    print("\nreview-queue strata:")
    print(strata.round(4).to_string(index=False))
    print("\nworst extractants for", champion)
    print(worst.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
