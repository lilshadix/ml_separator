#!/usr/bin/env python
"""Phase 3: does coordination topology track the level at all?  Training rows only.

Run *after* the descriptor specification is frozen and its digest recorded, so nothing
here can feed back into descriptor design, and *before* any level model is fitted.  The
development fold of `config/level_definition_rules.json` supplies the rows; no held-out
extractant of that fold contributes to any number below.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python \
        gen12_2_eu_pred/scripts/g122_exploratory_level.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import coordination, levels, paths  # noqa: E402
from gen12eu import splits  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402

HEADLINE = [
    "coord__donor__potential_donor_count",
    "coord__arm__local_donor_cluster_count",
    "coord__arm__repeated_arm_count",
    "coord__arm__max_connected_donor_motif_size",
    "coord__motif__n_dga_unit",
    "coord__motif__n_named_chelating_units",
    "coord__dist__frac_donor_pairs_within_3",
    "coord__dist__donor_network_diameter",
    "coord__dist__min_intercluster_distance",
    "coord__arch__heavy_atoms_per_donor",
    "coord__arch__mol_weight_per_donor",
    "coord__arch__clogp_per_donor",
    "coord__arch__n_degree3_atoms",
    "coord__arch__frac_atoms_in_largest_orbit",
]


def main() -> int:
    rules = json.loads((paths.CONFIG_DIR / "level_definition_rules.json").read_text())
    development = rules["development_fold"]
    cohort = build_cohort()
    frame = cohort.frame
    fold = next(f for f in splits.all_folds(frame, design=development["design"])
                if f.seed == development["split_seed"] and f.fold == development["fold"])
    train = frame.iloc[fold.train_index]
    table = pd.read_parquet(paths.FEATURE_DIR / "coordination_descriptors.parquet")
    digest = json.loads((paths.FEATURE_DIR / "coordination_audit.json").read_text())["spec_sha256"]
    if digest != coordination.spec_digest():
        raise SystemExit("the coordination specification changed after the matrix was built")

    alpha = levels.build_level_targets(train, train, cohort.blocks,
                                       definition="LVL_MEAN", seed=fold.model_seed).train
    counts = train["extractant"].astype(str).value_counts().reindex(alpha.index)
    identity = (train.drop_duplicates("extractant").set_index("extractant")
                [["chem_family"]].reindex(alpha.index))
    block = table.reindex(alpha.index)
    informative = [c for c in block.columns if block[c].nunique(dropna=True) > 1]

    # ---- univariate Spearman, all informative columns ----------------------- #
    rows = []
    for column in informative:
        series = block[column].astype(float)
        rho = float(series.corr(alpha, method="spearman"))
        # partial correlation given molecular size: the honest control for
        # "bigger molecules have more donors and were measured differently"
        size = block["coord__arch__n_heavy_atoms"].astype(float)
        if column == "coord__arch__n_heavy_atoms":
            partial = np.nan
        else:
            def residualise(v):
                v, s = np.asarray(v, float), np.asarray(size, float)
                beta = np.polyfit(s, v, 1)
                return v - np.polyval(beta, s)
            partial = float(pd.Series(residualise(series)).corr(
                pd.Series(residualise(alpha)), method="spearman"))
        rows.append({"descriptor": column, "family": coordination.family_of(column),
                     "spearman": rho, "abs_spearman": abs(rho),
                     "partial_spearman_given_size": partial,
                     "n_extractants": int(alpha.notna().sum())})
    correlations = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False,
                                                  ignore_index=True)
    correlations.to_csv(paths.ANALYSIS_DIR / "exploratory_level_correlations.csv", index=False)

    # ---- family-stratified view of the headline descriptors ----------------- #
    stratified = []
    for column in HEADLINE:
        if column not in block.columns:
            continue
        for family, index in identity.groupby("chem_family").groups.items():
            if len(index) < 6:
                continue
            stratified.append({
                "descriptor": column, "chem_family": str(family), "n": len(index),
                "spearman": float(block.loc[index, column].astype(float)
                                  .corr(alpha.loc[index], method="spearman"))})
    stratified = pd.DataFrame(stratified)
    stratified.to_csv(paths.ANALYSIS_DIR / "exploratory_level_by_family.csv", index=False)

    # ---- level against topicity, the central construct ---------------------- #
    topicity = block["coord__arm__local_donor_cluster_count"].astype(float).clip(upper=4)
    by_topicity = pd.DataFrame({
        "n_extractants": topicity.groupby(topicity).size(),
        "mean_level": alpha.groupby(topicity).mean(),
        "median_level": alpha.groupby(topicity).median(),
        "sd_level": alpha.groupby(topicity).std(),
        "mean_cells": counts.groupby(topicity).mean(),
    })
    within_dga = identity["chem_family"] == "diglycolamide"
    by_topicity_dga = pd.DataFrame({
        "n_extractants": topicity[within_dga].groupby(topicity[within_dga]).size(),
        "mean_level": alpha[within_dga].groupby(topicity[within_dga]).mean(),
    })

    report = ["# Exploratory analysis — coordination topology against the level target\n",
              f"Development fold: design {development['design']}, seed "
              f"{development['split_seed']}, fold {development['fold']}. "
              f"{len(alpha)} training extractants, {train['chemotype'].nunique()} training "
              "chemotypes. No held-out extractant and no model output contributed. The "
              "descriptor specification was frozen and digested before this ran, so nothing "
              "here can feed back into descriptor design.\n",
              "**This section describes a representation. It does not claim chemistry: a "
              "Spearman correlation between a graph descriptor and an average log D over "
              "heterogeneous conditions is not a mechanism.**\n",
              "## Strongest univariate associations (top 20 of "
              f"{len(informative)} informative descriptors)\n",
              correlations.head(20).round(3).to_markdown(index=False) + "\n",
              "## The pre-registered headline descriptors\n",
              correlations[correlations["descriptor"].isin(HEADLINE)].round(3)
              .to_markdown(index=False) + "\n",
              "## Level against pocket count\n",
              by_topicity.round(3).to_markdown() + "\n",
              "### within the diglycolamide family only\n",
              by_topicity_dga.round(3).to_markdown() + "\n",
              "## Family-stratified correlations\n",
              stratified.pivot_table(index="descriptor", columns="chem_family",
                                     values="spearman").round(3).to_markdown() + "\n"]
    (paths.ANALYSIS_DIR / "exploratory_level_analysis.md").write_text("\n".join(report))

    print(correlations.head(18).round(3).to_string(index=False))
    print("\nlevel by pocket count:")
    print(by_topicity.round(3).to_string())
    print("\nwithin diglycolamides only:")
    print(by_topicity_dga.round(3).to_string())
    print("\nfamily-stratified (headline descriptors):")
    print(stratified.pivot_table(index="descriptor", columns="chem_family",
                                 values="spearman").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
