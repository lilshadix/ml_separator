#!/usr/bin/env python
"""Recompute every number ``DECISION_REPORT.md`` quotes and fail on any mismatch.

A report whose numbers cannot be recomputed from the artefacts is a claim, not a result.
Exit code is non-zero if any check fails.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_verify_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import paths  # noqa: E402


def main() -> int:
    checks: list[dict] = []

    def check(claim: str, source: str, got, expected, tol: float = 6e-4):
        try:
            ok = abs(float(got) - float(expected)) <= tol
        except (TypeError, ValueError):
            ok = got == expected
        checks.append({"claim": claim, "source": source, "recomputed": got,
                       "reported": expected, "pass": bool(ok)})

    level = pd.read_csv(paths.METRIC_DIR / "level" / "LVL_MEAN" / "leaderboard.csv")
    full_cohort = level[level["cohort"] == "full"].set_index("arm")
    reliable = level[level["cohort"] == "level_reliable"].set_index("arm")
    pairwise = pd.read_csv(paths.BOOTSTRAP_DIR / "level_LVL_MEAN_pairwise.csv")
    primary = pairwise[(pairwise["cohort"] == "full")
                       & (pairwise["statistic"] == "mae")].set_index("contrast")
    caveats = pd.read_csv(paths.BOOTSTRAP_DIR / "statistical_caveats.csv")
    caveats = caveats[caveats["cohort"] == "full"].set_index("contrast")
    bottleneck = pd.read_csv(paths.ANALYSIS_DIR / "level_bottleneck.csv").set_index("arm")
    full = pd.read_csv(paths.METRIC_DIR / "full" / "leaderboard.csv").set_index("arm")
    full_pairs = pd.read_csv(paths.BOOTSTRAP_DIR / "full_pairwise.csv")
    full_pairs = full_pairs[full_pairs["statistic"] == "mae"].set_index("comparison")
    subgroup = pd.read_csv(paths.BOOTSTRAP_DIR / "level_LVL_MEAN_subgroup.csv")
    subgroup = subgroup.drop_duplicates("subgroup").set_index("subgroup")
    importance = pd.read_csv(paths.METRIC_DIR / "level" / "grouped_permutation_importance.csv")
    importance = importance.set_index("group")
    one_shot = pd.read_csv(paths.METRIC_DIR / "one_shot" / "one_shot_gain.csv").set_index("arm")
    posthoc = json.loads((paths.ANALYSIS_DIR / "descriptor_defect_sensitivity.json").read_text())
    bands = pd.read_csv(paths.BOOTSTRAP_DIR / "level_LVL_MEAN_by_band.csv")
    overlap = json.loads(
        (paths.METRIC_DIR / "level" / "LVL_MEAN" / "band_overlap.json").read_text())

    # --- §2 the bottleneck --------------------------------------------------- #
    row = bottleneck.loc["GEN12_ABL_D (best observed test)"]
    check("fixing the level takes ABL_D to", "analysis/level_bottleneck.csv",
          row["with_true_level_own_shape"], 0.310)
    check("fixing the shape takes ABL_D to", "analysis/level_bottleneck.csv",
          row["with_true_shape_own_level"], 0.917)
    check("recoverable by fixing the level", "analysis/level_bottleneck.csv",
          row["recoverable_by_fixing_level"], 0.738)
    check("recoverable by fixing the shape", "analysis/level_bottleneck.csv",
          row["recoverable_by_fixing_shape"], 0.131)
    check("Spearman level error vs overall error", "analysis/level_bottleneck.csv",
          row["spearman_level_error_vs_mae"], 0.883)
    check("share of squared error in the level", "analysis/level_bottleneck.csv",
          row["share_of_sse_in_level"], 0.560, 1e-3)

    # --- §3 the level ladder ------------------------------------------------- #
    for arm, value in (("L4_COORD", 0.962), ("L5_ECFP_COORD", 1.021), ("L7_ALL", 1.028),
                       ("L6_GENERIC_COORD", 1.032), ("L3_ECFP_GENERIC", 1.062),
                       ("L2_GENERIC", 1.074), ("L1_ECFP", 1.080),
                       ("L0_NN_TANIMOTO", 1.100), ("L0_GLOBAL_MEAN", 1.596)):
        check(f"level MAE {arm}", "metrics/level/LVL_MEAN/leaderboard.csv",
              full_cohort.loc[arm, "level_mae"], value)
    for arm, value in (("L4_COORD", 0.486), ("L7_ALL", 0.373), ("L3_ECFP_GENERIC", 0.332),
                       ("L1_ECFP", 0.309), ("L0_NN_TANIMOTO", 0.293)):
        check(f"between-extractant R2 {arm}", "metrics/level/LVL_MEAN/leaderboard.csv",
              full_cohort.loc[arm, "between_extractant_r2"], value)

    # --- §4 the primary and the other ablation contrasts --------------------- #
    for contrast, delta, low, high, p in (
            ("PRIMARY_G_vs_D", 0.034, -0.003, 0.063, 0.092),
            ("E_vs_A", 0.060, 0.016, 0.092, 0.007),
            ("F_vs_B", 0.042, -0.015, 0.086, 0.218),
            ("C_vs_NN", 0.138, -0.003, 0.265, 0.053)):
        check(f"{contrast} delta", "bootstrap/level_LVL_MEAN_pairwise.csv",
              primary.loc[contrast, "point_delta"], delta, 1e-3)
        check(f"{contrast} percentile low", "bootstrap/level_LVL_MEAN_pairwise.csv",
              primary.loc[contrast, "ci95_low"], low, 1e-3)
        check(f"{contrast} percentile high", "bootstrap/level_LVL_MEAN_pairwise.csv",
              primary.loc[contrast, "ci95_high"], high, 1e-3)
        check(f"{contrast} p", "bootstrap/level_LVL_MEAN_pairwise.csv",
              primary.loc[contrast, "p_two_sided"], p, 1e-3)
    check("primary BCa low", "bootstrap/level_LVL_MEAN_pairwise.csv",
          primary.loc["PRIMARY_G_vs_D", "bca_low"], 0.004, 1e-3)
    check("primary BCa high", "bootstrap/level_LVL_MEAN_pairwise.csv",
          primary.loc["PRIMARY_G_vs_D", "bca_high"], 0.075, 1e-3)
    check("primary realised power", "bootstrap/statistical_caveats.csv",
          caveats.loc["PRIMARY_G_vs_D", "power"], 0.46, 5e-3)
    check("primary Type-M", "bootstrap/statistical_caveats.csv",
          caveats.loc["PRIMARY_G_vs_D", "type_m"], 1.47, 5e-3)
    check("E_vs_A realised power", "bootstrap/statistical_caveats.csv",
          caveats.loc["E_vs_A", "power"], 0.86, 5e-3)
    check("primary MDE", "bootstrap/statistical_caveats.csv",
          caveats.loc["PRIMARY_G_vs_D", "mde_80pct"], 0.052, 1e-3)
    reliable_primary = pairwise[(pairwise["cohort"] == "level_reliable")
                                & (pairwise["statistic"] == "mae")].set_index("contrast")
    check("primary on the level-reliable cohort", "bootstrap/level_LVL_MEAN_pairwise.csv",
          reliable_primary.loc["PRIMARY_G_vs_D", "point_delta"], 0.036, 1e-3)
    check("L3 on the level-reliable cohort", "metrics/level/LVL_MEAN/leaderboard.csv",
          reliable.loc["L3_ECFP_GENERIC", "level_mae"], 0.995)
    check("L7 on the level-reliable cohort", "metrics/level/LVL_MEAN/leaderboard.csv",
          reliable.loc["L7_ALL", "level_mae"], 0.958)

    # --- §4 grouped importance ------------------------------------------------ #
    for group, value in (("ECFP", 0.256), ("generic_rdkit_lig2d", 0.239),
                         ("coord_donor", 0.037), ("coord_motif", 0.021),
                         ("coord_dist", 0.019), ("coord_arch", 0.017),
                         ("gen12_donor_census", 0.012), ("coord_arm", 0.010),
                         ("generic_physchem", 0.007)):
        check(f"grouped importance {group}",
              "metrics/level/grouped_permutation_importance.csv",
              importance.loc[group, "importance"], value, 1e-3)

    # --- §5 bands ------------------------------------------------------------- #
    for band, delta in (("far", 0.033), ("mid", 0.037), ("near", 0.023)):
        row = bands[(bands["band"] == band) & (bands["comparison"] == "PRIMARY_G_vs_D")]
        check(f"primary delta, {band} band", "bootstrap/level_LVL_MEAN_by_band.csv",
              row["point_delta"].iloc[0], delta, 1e-3)
    check("extractants in more than one band",
          "metrics/level/LVL_MEAN/band_overlap.json",
          overlap["extractants_in_more_than_one_band"], 52, 0)
    check("per-band counts sum to", "metrics/level/LVL_MEAN/band_overlap.json",
          overlap["sum_over_bands"], 235, 0)

    # --- §6, §7 full prediction ----------------------------------------------- #
    for arm, value in (("GEN12_ABL_D", 1.048), ("DIRECT_ALL", 1.049), ("DEC_L7_S0", 1.060),
                       ("DEC_L4_S1", 1.067), ("GEN12_ABL_C", 1.092), ("DEC_L7_S2", 1.101),
                       ("DEC_L7_S3", 1.108), ("DEC_L7_S1", 1.109), ("GEN12_ABL_A", 1.134),
                       ("DEC_L3_S1", 1.137)):
        check(f"full macro MAE {arm}", "metrics/full/leaderboard.csv",
              full.loc[arm, "macro_mae_extractant"], value)
    check("secondary endpoint delta", "bootstrap/full_pairwise.csv",
          full_pairs.loc["SECONDARY_decomposed_L7_vs_L3", "point_delta"], 0.028, 1e-3)
    check("secondary endpoint p", "bootstrap/full_pairwise.csv",
          full_pairs.loc["SECONDARY_decomposed_L7_vs_L3", "p_two_sided"], 0.319, 1e-3)
    check("DIRECT_ALL vs Gen12 delta", "bootstrap/full_pairwise.csv",
          full_pairs.loc["direct_ALL_vs_GEN12", "point_delta"], -0.001, 1e-3)
    check("DIRECT_ALL vs Gen12 p", "bootstrap/full_pairwise.csv",
          full_pairs.loc["direct_ALL_vs_GEN12", "p_two_sided"], 0.857, 1e-3)
    check("decomposed L3 vs direct delta", "bootstrap/full_pairwise.csv",
          full_pairs.loc["decomposed_L3_vs_direct_GEN12", "point_delta"], -0.089, 1e-3)
    check("decomposed L3 vs direct p", "bootstrap/full_pairwise.csv",
          full_pairs.loc["decomposed_L3_vs_direct_GEN12", "p_two_sided"], 0.002, 1e-3)

    # --- §8 the multi-arm subgroup -------------------------------------------- #
    check("multi-arm signed error, L3", "bootstrap/level_LVL_MEAN_subgroup.csv",
          subgroup.loc["MULTI_ARM", "signed_L3_ECFP_GENERIC"], -1.164, 1e-3)
    check("multi-arm signed error, L7", "bootstrap/level_LVL_MEAN_subgroup.csv",
          subgroup.loc["MULTI_ARM", "signed_L7_ALL"], -1.125, 1e-3)
    check("multi-arm signed error, L4", "bootstrap/level_LVL_MEAN_subgroup.csv",
          subgroup.loc["MULTI_ARM", "signed_L4_COORD"], -0.859, 1e-3)
    check("single-pocket signed error, L3", "bootstrap/level_LVL_MEAN_subgroup.csv",
          subgroup.loc["single_arm", "signed_L3_ECFP_GENERIC"], -0.242, 1e-3)
    check("multi-arm effective sample size", "bootstrap/level_LVL_MEAN_subgroup.csv",
          subgroup.loc["MULTI_ARM", "n_eff_chemotype"], 3.96, 5e-3)

    # --- §9 one-shot ----------------------------------------------------------- #
    for arm, zero, one in (("GEN12_ABL_D", 1.232, 1.018), ("DEC_L7_S1", 1.219, 0.965),
                           ("DEC_L3_S1", 1.243, 0.966), ("DIRECT_ALL", 1.220, 1.014)):
        check(f"one-shot zero-shot {arm}", "metrics/one_shot/one_shot_gain.csv",
              one_shot.loc[arm, "zero_shot_mae"], zero, 1e-3)
        check(f"one-shot k=1 {arm}", "metrics/one_shot/one_shot_gain.csv",
              one_shot.loc[arm, "one_shot_mae"], one, 1e-3)
    check("one-shot gain, DEC_L7_S1", "metrics/one_shot/one_shot_gain.csv",
          one_shot.loc["DEC_L7_S1", "one_shot_gain"], 0.254, 1e-3)
    check("one-shot gain, DEC_L3_S1", "metrics/one_shot/one_shot_gain.csv",
          one_shot.loc["DEC_L3_S1", "one_shot_gain"], 0.278, 1e-3)

    # --- post-hoc sensitivity --------------------------------------------------- #
    by_contrast = {r["contrast"]: r for r in posthoc["contrasts"]}
    check("post-hoc primary delta", "analysis/descriptor_defect_sensitivity.json",
          by_contrast["PRIMARY_G_vs_D"]["posthoc_delta"], 0.046, 1e-3)
    check("post-hoc primary p", "analysis/descriptor_defect_sensitivity.json",
          by_contrast["PRIMARY_G_vs_D"]["posthoc_p"], 0.020, 1e-3)
    check("post-hoc columns changed", "analysis/descriptor_defect_sensitivity.json",
          posthoc["n_columns_changed"], 52, 0)

    # --- family-stratified contrasts ---------------------------------------------- #
    families = pd.read_csv(paths.METRIC_DIR / "level" / "family_stratified_contrasts.csv")
    families = families.set_index(["fam", "comparison"])
    for key, delta, p_value in ((("diglycolamide", "C_minus_A"), 0.368, 0.012),
                                (("diglycolamide", "G_minus_D"), 0.076, 0.080),
                                (("diglycolamide", "C_minus_D"), 0.349, 0.070),
                                (("n_heterocyclic_polydentate", "C_minus_D"), -0.228, 0.001),
                                (("n_heterocyclic_polydentate", "C_minus_A"), -0.211, 0.014),
                                (("amide_other", "G_minus_D"), -0.013, 0.422)):
        check(f"{key[0]} {key[1]} delta", "metrics/level/family_stratified_contrasts.csv",
              families.loc[key, "point_delta"], delta, 1e-3)
        check(f"{key[0]} {key[1]} p", "metrics/level/family_stratified_contrasts.csv",
              families.loc[key, "p_two_sided"], p_value, 1e-3)

    # --- multiplicity ------------------------------------------------------------ #
    summary = json.loads((paths.BOOTSTRAP_DIR / "statistical_caveats.json").read_text())
    check("bootstrap rows emitted", "bootstrap/statistical_caveats.json",
          summary["n_bootstrap_rows_emitted"], 348, 0)
    check("rows flagged as excluding zero", "bootstrap/statistical_caveats.json",
          summary["n_flagged_bca_excludes_zero"], 76, 0)
    check("of which the percentile interval includes zero",
          "bootstrap/statistical_caveats.json",
          summary["n_contrasts_where_bca_and_percentile_disagree"], 18, 0)

    table = pd.DataFrame(checks)
    table.to_csv(paths.MANIFEST_DIR / "report_verification.csv", index=False)
    failed = table[~table["pass"]]
    print(table[["claim", "recomputed", "reported", "pass"]].to_string(index=False))
    print(f"\n{len(table) - len(failed)}/{len(table)} checks pass")
    if len(failed):
        print("\nFAILURES:")
        print(failed.to_string(index=False))
    return 1 if len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
