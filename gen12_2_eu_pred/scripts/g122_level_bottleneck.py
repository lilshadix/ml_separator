#!/usr/bin/env python
"""H1: is unseen-extractant *level* really where Gen12's zero-shot error lives?

Gen12 said so.  This quantifies it rather than restating it, from Gen12's frozen
predictions, and adds the two counterfactuals that make the claim falsifiable: what a
perfect level with the model's own shape would score, and what a perfect shape with the
model's own level would score.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python \
        gen12_2_eu_pred/scripts/g122_level_bottleneck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import paths  # noqa: E402
from gen12eu.metrics import per_extractant  # noqa: E402

TARGET = "log_D"
ARMS = {
    "GEN12_ABL_D (best observed test)": paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_D_PLUS_LIG2D.parquet",
    "GEN12_ABL_C (pre-registered selection)": paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_C_MOL_COND.parquet",
    "GEN12_ABL_A (no chemistry)": paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_A_CONDITIONS.parquet",
    "GEN12_ABL_E (fingerprint + conditions)": paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_E_ECFP_COND.parquet",
    "GEN12_B1_COND_ONLY": paths.GEN12_PREDICTIONS / "B" / "B1_COND_ONLY.parquet",
    "GEN12_B0_GLOBAL_MEAN": paths.GEN12_PREDICTIONS / "B" / "B0_GLOBAL_MEAN.parquet",
}


def macro(frame: pd.DataFrame, prediction: np.ndarray) -> float:
    ae = np.abs(prediction - frame[TARGET].to_numpy(dtype=float))
    return float(pd.Series(ae).groupby([frame["split_seed"].to_numpy(),
                                        frame["extractant"].to_numpy()]).mean()
                 .groupby(level=0).mean().mean())


def main() -> int:
    cohort = pd.read_parquet(paths.GEN12_COHORT_PARQUET)
    y = cohort[TARGET].to_numpy(dtype=float)
    means = cohort.groupby("extractant")[TARGET].transform("mean").to_numpy()
    total = float(((y - y.mean()) ** 2).sum())
    variance = {
        "between_extractant_share": float(((means - y.mean()) ** 2).sum() / total),
        "within_extractant_share": float(((y - means) ** 2).sum() / total),
        "target_sd": float(y.std()),
        "between_extractant_sd_of_levels": float(cohort.groupby("extractant")[TARGET].mean().std()),
    }

    rows = []
    for label, path in ARMS.items():
        frame = pd.read_parquet(path)
        units = per_extractant(frame)
        prediction = frame["prediction"].to_numpy(dtype=float)
        truth = frame[TARGET].to_numpy(dtype=float)
        group = [frame["split_seed"].to_numpy(), frame["extractant"].to_numpy()]
        true_level = pd.Series(truth).groupby(group).transform("mean").to_numpy()
        predicted_level = pd.Series(prediction).groupby(group).transform("mean").to_numpy()
        # counterfactual A: the model's shape, the true level
        level_oracle = prediction - predicted_level + true_level
        # counterfactual B: the model's level, the true shape
        shape_oracle = truth - true_level + predicted_level
        # counterfactual C: the true level and no shape model at all
        level_only = true_level
        observed = macro(frame, prediction)
        rows.append({
            "arm": label,
            "macro_mae": observed,
            "level_component": float(units["offset_abs"].mean()),
            "shape_component": float(units["shape_mae"].mean()),
            "level_share_of_components": float(
                units["offset_abs"].mean() / (units["offset_abs"].mean()
                                              + units["shape_mae"].mean())),
            "with_true_level_own_shape": macro(frame, level_oracle),
            "with_true_shape_own_level": macro(frame, shape_oracle),
            "true_level_no_shape_model": macro(frame, level_only),
            "recoverable_by_fixing_level": observed - macro(frame, level_oracle),
            "recoverable_by_fixing_shape": observed - macro(frame, shape_oracle),
            "spearman_level_error_vs_mae": float(
                units["offset_abs"].corr(units["mae"], method="spearman")),
            "share_of_sse_in_level": float(units["sse_offset"].sum() / units["sse"].sum()),
        })
    table = pd.DataFrame(rows)
    out = paths.ANALYSIS_DIR
    table.to_csv(out / "level_bottleneck.csv", index=False)
    (out / "level_bottleneck.json").write_text(json.dumps(
        {"variance": variance, "arms": table.to_dict(orient="records")}, indent=1))

    report = ["# H1 — is the level the bottleneck?\n",
              "*Every number from Gen12's frozen design-B predictions, zero-shot, "
              "extractant-macro MAE, level and shape centred per (split_seed, extractant).*\n",
              "## The target itself\n",
              pd.Series(variance).to_frame("value").round(4).to_markdown() + "\n",
              "## What each Gen12 arm's error is made of, and what fixing each half buys\n",
              table.round(4).to_markdown(index=False) + "\n",
              "`with_true_level_own_shape` replaces each held-out extractant's predicted level "
              "with its true level and keeps the model's own within-extractant shape. "
              "`with_true_shape_own_level` does the reverse. `true_level_no_shape_model` "
              "predicts the true level for every row and models no condition response at all.\n"]
    (out / "level_bottleneck.md").write_text("\n".join(report))
    print(json.dumps(variance, indent=1))
    print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
