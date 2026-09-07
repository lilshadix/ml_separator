#!/usr/bin/env python
"""Recompute every number ``DECISION_REPORT.md`` quotes, and fail on any mismatch.

The repository's own practice: a report whose numbers cannot be recomputed from
the artefacts is a claim, not a result.  Each entry names the claim, the artefact
it comes from, and the tolerance.  Exit code is non-zero if any check fails.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_verify_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import paths  # noqa: E402


def main() -> int:
    checks: list[dict] = []

    def check(claim: str, source: str, got, expected, tol: float = 5e-4):
        try:
            ok = abs(float(got) - float(expected)) <= tol
        except (TypeError, ValueError):
            ok = got == expected
        checks.append({"claim": claim, "source": source, "recomputed": got,
                       "reported": expected, "tolerance": tol, "pass": bool(ok)})

    audit = json.loads((paths.MANIFEST_DIR / "data_audit.json").read_text())
    ladder = pd.read_csv(paths.METRIC_DIR / "B" / "leaderboard.csv").set_index("arm")
    ablation = pd.read_csv(paths.METRIC_DIR / "B_ablation" / "leaderboard.csv").set_index("arm")
    hypotheses = pd.read_csv(paths.BOOTSTRAP_DIR / "B_hypotheses.csv")

    # --- cohort ------------------------------------------------------------ #
    check("cohort cells", "data_audit.json", audit["rows"], 1329, 0)
    check("extractants", "data_audit.json", audit["extractants"], 183, 0)
    check("ECFP clusters", "data_audit.json", audit["ecfp_clusters"], 162, 0)
    check("chemotypes", "data_audit.json", audit["chemotypes"], 97, 0)
    check("cohort fingerprint", "data_audit.json", audit["cohort_fingerprint"], "2a364bb5264e9935")
    check("target sd", "data_audit.json", audit["target"]["sd"], 1.899)
    check("target mean", "data_audit.json", audit["target"]["mean"], -0.060, 1e-3)
    check("quarantined rows", "data_audit.json", audit["quarantine_rows"], 125, 0)
    check("single-row extractants", "data_audit.json",
          audit["distribution"]["n_extractants_single_row"], 85, 0)
    check("n_eff extractants by chemotype", "data_audit.json",
          audit["distribution"]["n_eff_extractants_by_chemotype"], 13.7, 5e-2)
    check("largest chemotype extractants", "data_audit.json",
          audit["distribution"]["largest_chemotype_extractants"], 46, 0)
    check("design B max train Tanimoto", "data_audit.json",
          audit["splits"]["B"]["max_train_tanimoto"]["max"], 0.698, 1e-3)
    check("design A ECFP-cluster leaks", "data_audit.json",
          audit["splits"]["A"]["train_test_overlap_counts"]["ecfp_cluster"], 139, 0)
    check("design A chemotype leaks", "data_audit.json",
          audit["splits"]["A"]["train_test_overlap_counts"]["chemotype"], 234, 0)
    check("non-Eu rows on Eu extractants", "data_audit.json",
          audit["cross_metal"]["non_eu_rows_on_eu_extractants"], 4148, 0)

    # --- zero-shot leaderboard --------------------------------------------- #
    for arm, value in (("ABL_D_PLUS_LIG2D", 1.048), ("ABL_C_MOL_COND", 1.092),
                       ("ABL_A_CONDITIONS", 1.134), ("ABL_B_MOLECULE", 1.215),
                       ("ABL_E_ECFP_COND", 1.107), ("ABL_F_GRAPH_COND", 1.275)):
        check(f"{arm} macro MAE", "metrics/B_ablation/leaderboard.csv",
              ablation.loc[arm, "macro_mae_extractant"], value)
    for arm, value in (("B0_GLOBAL_MEAN", 1.731), ("B1_COND_ONLY", 1.167),
                       ("B2_NN_CHEMICAL", 1.170), ("B3_NN_LEVEL_ONLY", 1.222),
                       ("T3_DMPNN_COND_DESC", 1.197), ("T3_DMPNN_COND", 1.275),
                       ("T3_DMPNN_GRAPH_ONLY", 1.350),
                       ("T2_MLP", 1.241), ("T1_EXTRATREES", 1.092)):
        check(f"{arm} macro MAE", "metrics/B/leaderboard.csv",
              ladder.loc[arm, "macro_mae_extractant"], value)
    check("champion dispersion ratio", "metrics/B/calibration.csv",
          pd.read_csv(paths.METRIC_DIR / "B" / "calibration.csv")
          .set_index("arm").loc["ABL_D_PLUS_LIG2D", "dispersion_ratio"], 0.595, 1e-3)

    # --- H1 / H2 ------------------------------------------------------------ #
    h1 = hypotheses[(hypotheses["control"] == "ABL_A_CONDITIONS")
                    & (hypotheses["candidate"] == "ABL_D_PLUS_LIG2D")].set_index("band")
    check("H1 delta", "bootstrap/B_hypotheses.csv", h1.loc["overall", "point_delta"], 0.086, 1e-3)
    check("H1 BCa low", "bootstrap/B_hypotheses.csv", h1.loc["overall", "bca_low"], -0.025, 1e-3)
    check("H1 BCa high", "bootstrap/B_hypotheses.csv", h1.loc["overall", "bca_high"], 0.189, 1e-3)
    check("H1 MDE", "bootstrap/B_hypotheses.csv", h1.loc["overall", "mde_80pct_power"], 0.163, 1e-3)
    for band, delta, mde in (("far", 0.065, 0.302), ("mid", 0.109, 0.290), ("near", 0.066, 0.119)):
        check(f"H2 {band} delta", "bootstrap/B_hypotheses.csv", h1.loc[band, "point_delta"], delta, 1e-3)
        check(f"H2 {band} MDE", "bootstrap/B_hypotheses.csv", h1.loc[band, "mde_80pct_power"], mde, 1e-3)
    hc = hypotheses[(hypotheses["control"] == "ABL_A_CONDITIONS")
                    & (hypotheses["candidate"] == "T1_EXTRATREES")].set_index("band")
    check("ABL_C far delta is negative", "bootstrap/B_hypotheses.csv",
          hc.loc["far", "point_delta"], -0.031, 1e-3)

    # --- few-shot ----------------------------------------------------------- #
    curve = pd.read_csv(paths.METRIC_DIR / "B" / "fewshot_curve.csv")
    common = curve[(curve["cohort"] == "common") & (curve["arm"] == "T1_EXTRATREES")
                   & (curve["adapter"] == "A_OFFSET_SHRUNK")].set_index("k")
    for k, value in ((0, 1.271), (1, 1.006), (2, 0.926), (3, 0.893), (5, 0.867)):
        check(f"k-shot k={k}", "metrics/B/fewshot_curve.csv", common.loc[k, "mae"], value, 1e-3)
    check("k-shot level at k=0", "metrics/B/fewshot_curve.csv",
          common.loc[0, "offset_abs"], 0.990, 1e-3)
    check("k-shot level at k=5", "metrics/B/fewshot_curve.csv",
          common.loc[5, "offset_abs"], 0.437, 1e-3)
    check("k-shot shape at k=0", "metrics/B/fewshot_curve.csv",
          common.loc[0, "shape_mae"], 0.767, 1e-3)
    check("k-shot shape at k=5", "metrics/B/fewshot_curve.csv",
          common.loc[5, "shape_mae"], 0.728, 1e-3)
    one = pd.read_csv(paths.BOOTSTRAP_DIR / "B_one_measurement.csv")
    row = one[one["cohort"] == "all_scorable"].iloc[0]
    check("zero-shot on shared query rows", "bootstrap/B_one_measurement.csv",
          row["macro_ZERO_SHOT_MODEL"], 1.154, 1e-3)
    check("one measurement, no model", "bootstrap/B_one_measurement.csv",
          row["macro_ONE_MEASUREMENT_NO_MODEL"], 1.030, 1e-3)
    check("model plus one measurement", "bootstrap/B_one_measurement.csv",
          row["macro_MODEL_PLUS_ONE_MEASUREMENT"], 0.922, 1e-3)

    # --- multi-lanthanide ---------------------------------------------------- #
    multiln = pd.read_csv(paths.METRIC_DIR / "B_multiln" / "leaderboard.csv").set_index("arm")
    for arm, value in (("T4_STRICT", 1.071), ("T4_MATCHED_EU_ONLY", 1.068),
                       ("T4_LEAKY_NO_FILTER", 0.741), ("T4_PERMUTED_METAL", 1.101),
                       ("T4_SHUFFLED_TARGET", 1.114)):
        check(f"{arm} macro MAE", "metrics/B_multiln/leaderboard.csv",
              multiln.loc[arm, "macro_mae_extractant"], value, 1e-3)
    pair = pd.read_csv(paths.BOOTSTRAP_DIR / "B_multiln_pairwise.csv")
    pair = pair[pair["statistic"] == "mae"].set_index("candidate")
    ablation_pairs = pd.read_csv(paths.BOOTSTRAP_DIR / "B_ablation_pairwise.csv").set_index("candidate")
    for arm, delta in (("ABL_D_PLUS_LIG2D", 0.044), ("ABL_F_GRAPH_COND", -0.184),
                       ("ABL_B_MOLECULE", -0.123), ("ABL_E_ECFP_COND", -0.015)):
        check(f"ablation {arm} vs C", "bootstrap/B_ablation_pairwise.csv",
              ablation_pairs.loc[arm, "point_delta"], delta, 1e-3)

    check("strict transfer delta", "bootstrap/B_multiln_pairwise.csv",
          pair.loc["T4_STRICT", "point_delta"], -0.003, 1e-3)
    check("leaky transfer delta", "bootstrap/B_multiln_pairwise.csv",
          pair.loc["T4_LEAKY_NO_FILTER", "point_delta"], 0.327, 1e-3)

    # --- design A cost -------------------------------------------------------- #
    design_a = pd.read_csv(paths.METRIC_DIR / "A" / "leaderboard.csv").set_index("arm")
    gap_model = ladder.loc["T1_EXTRATREES", "macro_mae_extractant"] - design_a.loc[
        "T1_EXTRATREES", "macro_mae_extractant"]
    gap_nochem = ladder.loc["B1_COND_ONLY", "macro_mae_extractant"] - design_a.loc[
        "B1_COND_ONLY", "macro_mae_extractant"]
    check("design A inflates the champion by", "metrics/A/leaderboard.csv", gap_model, 0.320, 1e-3)
    check("share of design-A gain available without chemistry",
          "metrics/{A,B}/leaderboard.csv", gap_nochem / gap_model, 0.887, 1e-3)

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
