#!/usr/bin/env python
"""Reproduce Gen12's failure family exactly, from its frozen predictions.

Gen12 reported that ten of its fifteen worst extractants are bridged, tripodal or
multi-armed diglycolamides, under-predicted by 2.6 to 4.2 decades.  This script
reproduces that from ``gen12_eu_pred/predictions/`` and describes each one with the
coordination descriptors, so that the multi-arm hypothesis is tested against a
prospectively defined structural rule rather than against a hand-picked list.

**These errors are not used to design anything.**  The descriptor specification and the
MULTI_ARM rule were frozen and digested before this ran.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python \
        gen12_2_eu_pred/scripts/g122_failure_family.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import paths  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402

TARGET = "log_D"
CHAMPION = paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_D_PLUS_LIG2D.parquet"
SELECTED = paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_C_MOL_COND.parquet"


def level_table(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    table = (frame.groupby("extractant")
             .agg(true_level=(TARGET, "mean"), predicted_level=("prediction", "mean"),
                  n_cells=("row_id", "nunique"),
                  max_train_tanimoto=("max_train_tanimoto", "mean"),
                  band=("band", "first"), chemotype=("chemotype", "first"),
                  row_mae=("prediction", "size")).reset_index())
    per_row = frame.assign(ae=(frame["prediction"] - frame[TARGET]).abs())
    table["row_mae"] = (per_row.groupby(["split_seed", "extractant"])["ae"].mean()
                        .groupby("extractant").mean().reindex(table["extractant"]).to_numpy())
    table["level_bias"] = table["predicted_level"] - table["true_level"]
    table["arm"] = label
    return table


def main() -> int:
    cohort = build_cohort()
    identity = (cohort.frame.drop_duplicates("extractant")
                .set_index("extractant")[["extractant_name", "chem_family"]])
    coordination_table = pd.read_parquet(paths.FEATURE_DIR / "coordination_descriptors.parquet")
    subgroup = pd.read_csv(paths.MANIFEST_DIR / "multi_arm_subgroup.csv", index_col=0)

    champion = level_table(CHAMPION, "ABL_D_PLUS_LIG2D").set_index("extractant")
    selected = level_table(SELECTED, "ABL_C_MOL_COND").set_index("extractant")
    table = champion.join(identity).join(subgroup[["MULTI_ARM"]])
    for column in ("coord__arm__local_donor_cluster_count", "coord__arm__repeated_arm_count",
                   "coord__motif__n_dga_unit", "coord__donor__potential_donor_count",
                   "coord__arm__max_connected_donor_motif_size",
                   "coord__dist__min_intercluster_distance",
                   "coord__arch__heavy_atoms_per_donor"):
        table[column] = coordination_table[column].reindex(table.index).to_numpy()
    nearest = (pd.read_parquet(paths.GEN12_PREDICTIONS / "B" / "similarity.parquet")
               .groupby("extractant")["nearest_train_extractant"]
               .agg(lambda s: s.value_counts().index[0]))
    table["nearest_train_extractant"] = nearest.reindex(table.index).to_numpy()
    table["nearest_train_name"] = identity["extractant_name"].reindex(
        table["nearest_train_extractant"]).to_numpy()

    worst = table.sort_values("row_mae", ascending=False).head(15)
    worst_by_level = table.sort_values(table["level_bias"].abs().name, key=lambda s: np.abs(
        table["level_bias"]), ascending=False).head(15)

    # ---- the subgroup contrast, on the frozen Gen12 predictions ------------- #
    grouped = table.groupby("MULTI_ARM").agg(
        n_extractants=("row_mae", "size"), n_chemotypes=("chemotype", "nunique"),
        mean_row_mae=("row_mae", "mean"), mean_level_bias=("level_bias", "mean"),
        median_level_bias=("level_bias", "median"),
        mean_true_level=("true_level", "mean"),
        mean_predicted_level=("predicted_level", "mean"),
        frac_underpredicted=("level_bias", lambda s: float((s < 0).mean())))
    by_topicity = table.groupby(
        table["coord__arm__local_donor_cluster_count"].clip(upper=4)).agg(
        n=("row_mae", "size"), mean_level_bias=("level_bias", "mean"),
        mean_true_level=("true_level", "mean"), mean_row_mae=("row_mae", "mean"))

    summary = {
        "champion_arm": "ABL_D_PLUS_LIG2D (Gen12 best observed test arm)",
        "selected_arm": "ABL_C_MOL_COND (Gen12 pre-registered selection)",
        "n_worst_15_that_are_multi_arm": int(worst["MULTI_ARM"].sum()),
        "n_worst_15_that_are_diglycolamide": int(
            (worst["chem_family"] == "diglycolamide").sum()),
        "worst_15_level_bias_range": [float(worst["level_bias"].min()),
                                      float(worst["level_bias"].max())],
        "n_worst_15_underpredicted": int((worst["level_bias"] < 0).sum()),
        "multi_arm_mean_level_bias": float(grouped.loc[True, "mean_level_bias"]),
        "single_arm_mean_level_bias": float(grouped.loc[False, "mean_level_bias"]),
        "gap": float(grouped.loc[True, "mean_level_bias"]
                     - grouped.loc[False, "mean_level_bias"]),
        "selected_arm_multi_arm_mean_level_bias": float(
            selected.join(subgroup[["MULTI_ARM"]]).groupby("MULTI_ARM")["level_bias"].mean()[True]),
    }
    (paths.ANALYSIS_DIR / "gen12_failure_family.json").write_text(json.dumps(summary, indent=1))

    show = ["extractant_name", "chem_family", "n_cells", "true_level", "predicted_level",
            "level_bias", "row_mae", "max_train_tanimoto", "band", "MULTI_ARM",
            "coord__arm__local_donor_cluster_count", "coord__arm__repeated_arm_count",
            "coord__motif__n_dga_unit", "coord__donor__potential_donor_count",
            "nearest_train_name"]
    report = [
        "# Gen12's failure family, reproduced\n",
        "*Every number here comes from `gen12_eu_pred/predictions/B_ablation/`, unchanged. "
        "The coordination descriptors and the MULTI_ARM rule were frozen and digested before "
        "this ran, and none of these errors was used to design either. This file exists so "
        "that the multi-arm hypothesis is tested against a structural rule applied to all 183 "
        "extractants, not against a hand-picked list.*\n",
        "Regime: zero-shot, design B, arm `ABL_D_PLUS_LIG2D`, level = per-extractant mean of "
        "`log_D`, bias = predicted level minus true level, so a negative bias is "
        "under-prediction.\n",
        "## The fifteen worst extractants by row MAE\n",
        worst[show].round(3).to_markdown(index=False) + "\n",
        "## The fifteen worst by absolute level bias\n",
        worst_by_level[show].round(3).to_markdown(index=False) + "\n",
        "## The pre-registered subgroup, on Gen12's own predictions\n",
        grouped.round(4).to_markdown() + "\n",
        "## Level bias against pocket count\n",
        by_topicity.round(3).to_markdown() + "\n",
        "## Canonical SMILES of the fifteen worst\n",
        "\n".join(f"- **{r.extractant_name}** ({r.chem_family}, {int(r.n_cells)} cells, "
                  f"level bias {r.level_bias:+.2f}): `{i}`"
                  for i, r in worst.iterrows()) + "\n",
    ]
    (paths.ANALYSIS_DIR / "gen12_failure_family.md").write_text("\n".join(report))
    table.to_csv(paths.ANALYSIS_DIR / "gen12_level_bias_per_extractant.csv")

    print(worst[show[:11]].round(3).to_string(index=False))
    print("\nsubgroup means on Gen12's frozen predictions:")
    print(grouped.round(4).to_string())
    print("\nlevel bias by pocket count:")
    print(by_topicity.round(3).to_string())
    print("\n", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
