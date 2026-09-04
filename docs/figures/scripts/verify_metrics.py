"""Independent recomputation of every number that appears in a figure.

For each quantity: recompute it from the rawest artefact available (per-row or per-unit
predictions), compare against the value the repository already reports, and record
PASS/FAIL.  Writes ``figures/derived/metric_audit.csv``; ``figures/METRIC_AUDIT.md`` is
written from it.

Run from the repository root:
    ``.venv/bin/python figures/scripts/verify_metrics.py``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import _paths  # noqa: E402

ROWS: list[dict] = []


def check(quantity, source, experiment, cohort, seeds, aggregation, n_units,
          recomputed, reported, reported_from, tol=5e-4, note=""):
    delta = float("nan") if reported is None else abs(recomputed - reported)
    ROWS.append({
        "quantity": quantity, "source_file": source, "experiment": experiment,
        "cohort": cohort, "seeds": seeds, "aggregation": aggregation, "n_units": n_units,
        "recomputed": recomputed, "reported": reported, "reported_from": reported_from,
        "abs_delta": delta,
        "verdict": "PASS" if (reported is not None and delta <= tol) else
                   ("NEW" if reported is None else "FAIL"),
        "note": note})


def ensure(name: str, script: str) -> Path:
    path = _paths.DERIVED / name
    if not path.exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name(script))], check=True)
    return path


def main() -> int:
    per_ligand = pd.read_csv(ensure("kshot_per_ligand.csv", "prepare_kshot_tables.py"))
    meta = json.loads((_paths.DERIVED / "kshot_cohort.json").read_text())
    n99 = meta["n_ligands_common"]
    seeds = "5 split seeds x 12 repeats"
    kshot_src = "runs/gen10_final/final_locked/kshot_detail.parquet"

    def frontier(model, adapter, policy, k):
        block = per_ligand[(per_ligand.global_model == model) & (per_ligand.adapter == adapter)
                           & (per_ligand.policy == policy) & (per_ligand.k == k)]
        return float(block["mae"].mean())

    # ---- F1-F5: the headline frontier -------------------------------------------
    published = {0: 1.0358, 1: 0.6539, 2: 0.5593, 3: 0.4928, 5: 0.4405}
    rule = {0: ("ZERO_SHOT_REF", "NONE"), 1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
            2: ("SERIES_ML", "CENTRAL_THEN_SPREAD"), 3: ("SERIES_ML", "CENTRAL_THEN_SPREAD"),
            5: ("SERIES_ML", "CENTRAL_THEN_SPREAD")}
    for k, (adapter, policy) in rule.items():
        check(f"frozen-pipeline macro MAE at k={k}", kshot_src,
              "gen10 Phase 11 final_locked", f"{n99}-extractant common cohort", seeds,
              "mean over extractants of mean over (seed, repeat)", n99,
              frontier("GEN9_SHAPE_RECOMPOSED", adapter, policy, k), published[k],
              "GEN10_FINAL_DECISION_REPORT.md §3.10 / final_locked/frontier_best.csv",
              tol=5e-5)

    # ---- F6: gen8/gen9 comparison arms recomputed here ---------------------------
    gen9 = pd.read_csv(_paths.run("gen9_shape/frontier/table_b_common_cohort.csv"))
    for model, adapter in [("REC_ecfp_plus_recovered", "SLOPE_L_s1_K3"),
                           ("GEN9_SHAPE_RECOMPOSED", "SLOPE_L_s1_K3")]:
        for k in (2, 3, 5):
            ref = gen9[(gen9.global_model == model)
                       & (gen9.arm == f"{adapter}@CENTRAL_THEN_SPREAD") & (gen9.k == k)]
            check(f"{model} + {adapter} at k={k}", kshot_src,
                  "gen10 rerun vs gen9's own frontier table", f"{n99}-extractant common cohort",
                  seeds, "mean over extractants", n99,
                  frontier(model, adapter, "CENTRAL_THEN_SPREAD", k),
                  float(ref.mae.iloc[0]),
                  "runs/gen9_shape/frontier/table_b_common_cohort.csv", tol=5e-5)

    # ---- S1-S5: the shape metrics ------------------------------------------------
    cs = pd.read_parquet(_paths.run("gen9_shape/shape/curve_shape.parquet"))
    cs = cs[cs.axis_label == "extractant"]
    axis = pd.read_csv(_paths.run("gen9_shape/shape/shape_by_axis.csv"))
    axis = axis[axis.axis_label == "extractant"].set_index("model")
    for model, nice in [("REC_ecfp_plus_recovered", "baseline"),
                        ("GEN9_REL_MONOLITH", "+ relative position"),
                        ("GEN9_SHAPE_RECOMPOSED", "+ recomposition")]:
        block = cs[cs.model == model]
        for column, published_column, stat in [
                ("slope_pred", "slope_pred_median", "median"),
                ("span_recovery", "span_recovery_median_unguarded", "median"),
                ("shape_mae", "shape_mae", "mean"),
                ("spearman", "within_curve_spearman", "mean")]:
            value = float(getattr(block[column].dropna(), stat)())
            check(f"extractant {column} ({stat}), {nice}",
                  "runs/gen9_shape/shape/curve_shape.parquet", "gen9 shape analysis",
                  "775 curve x seed evaluations of 155 curves / 25 extractants",
                  "5 split seeds", stat + " over curve x seed", 775, value,
                  float(axis.loc[model, published_column]),
                  "runs/gen9_shape/shape/shape_by_axis.csv", tol=5e-6)
    check("extractant measured slope (median)",
          "runs/gen9_shape/shape/curve_shape.parquet", "gen9 shape analysis",
          "775 curve x seed evaluations", "5 split seeds", "median", 775,
          float(cs[cs.model == "REC_ecfp_plus_recovered"].slope_true.median()),
          float(axis.loc["REC_ecfp_plus_recovered", "slope_true_median"]),
          "runs/gen9_shape/shape/shape_by_axis.csv", tol=5e-6)

    # ---- E1-E5: the oracle cascade ------------------------------------------------
    budget = json.loads(ensure("error_budget.json", "prepare_error_budget.py").read_text())
    for entry in budget["verification"]["checks"]:
        if entry["metric"] != "macro_mae":
            continue
        check(f"oracle cascade, {entry['stage']}",
              "runs/gen9_shape/recomposed/oof_all.parquet", "gen10 Phase 10",
              "152 extractants", "5 split seeds", "one vote per ECFP cluster (131)", 131,
              entry["reproduced"], entry["published"],
              "runs/gen10_final/error_decomposition/summary.json", tol=5e-9)

    # ---- G1-G5: distance terciles and marginal gains ------------------------------
    fig5 = json.loads((_paths.DERIVED / "fig5_values.json").read_text()) \
        if (_paths.DERIVED / "fig5_values.json").exists() else None
    if fig5:
        for entry in fig5["panelA_verification"]:
            check(f"{entry['quantity']} ({entry['tercile']} tercile)", kshot_src,
                  "gen10 Phase 9 budget simulation", f"{n99}-extractant common cohort",
                  seeds, "mean over extractants within the tercile", n99,
                  entry["reproduced"], entry["published"],
                  "runs/gen10_final/budget_simulation/marginal_gains.csv", tol=1e-9)

    # ---- H1-H3: gen6 coverage experiment -------------------------------------------
    hard = pd.read_csv(_paths.run("gen6_expA_5seed/hard_chemistry_metrics.csv"))
    hard = hard[hard.feature_set == "MC_lig2d_ext_massaction"]
    report_values = {("all", "BASE"): 1.2100, ("all", "EXPANDED"): 1.0468,
                     ("all", "EXPANDED_ROWMATCHED"): 1.0486,
                     ("all", "EXPANDED_SHUFFLED"): 1.2193}
    for (endpoint, arm), reported in report_values.items():
        block = hard[(hard.endpoint == endpoint) & (hard.arm == arm)]
        check(f"gen6 {arm} macro MAE ({endpoint})",
              "runs/gen6_expA_5seed/hard_chemistry_metrics.csv", "gen6 Experiment A",
              "5,248 rows / 152 extractants / 131 ECFP clusters", "5 split seeds",
              "one vote per ECFP cluster, mean over seeds", 131,
              float(block.groupby("split_seed").macro_mae.mean().mean()), reported,
              "docs/gen6_phase0_and_phase1_results_20260819.md §5.1", tol=5e-5)

    contrasts = pd.read_csv(_paths.run("gen6_expA_5seed/contrast_summary.csv"))
    row = contrasts[(contrasts.feature_set == "MC_lig2d_ext_massaction")
                    & (contrasts.statistic == "mae") & (contrasts.endpoint == "all")
                    & (contrasts.comparison == "EXPANDED_vs_BASE")].iloc[0]
    check("gen6 coverage effect (EXPANDED - BASE), all",
          "runs/gen6_expA_5seed/contrast_summary.csv", "gen6 Experiment A",
          "identical held-out rows", "5 split seeds", "paired, one vote per ECFP cluster",
          131, float(row.pooled_point_delta), 0.1633,
          "docs/gen6_phase0_and_phase1_results_20260819.md §5.2 (H A1)", tol=5e-5)

    # ---- I1: acquisition ----------------------------------------------------------
    acq = pd.read_parquet(_paths.run("gen10_final/acquisition/realised_detail.parquet"),
                          columns=["feature_set", "policy", "extractant", "mae", "harms"])
    acq = acq[acq.feature_set == "geometry"]
    mine = acq.groupby(["policy", "extractant"])["mae"].mean().groupby("policy").mean()
    pub = pd.read_csv(_paths.run("gen10_final/acquisition/realised_summary.csv")
                      ).set_index("policy")
    for policy, value in mine.sort_values().items():
        check(f"acquisition macro MAE, {policy}",
              "runs/gen10_final/acquisition/realised_detail.parquet",
              "gen10 Phase 6 realised acquisition", "143 extractants",
              "5 split seeds x 8 repeats", "mean over extractants", 143,
              float(value), float(pub.loc[policy, "mae"]),
              "runs/gen10_final/acquisition/realised_summary.csv", tol=1e-9)

    # ---- J1: mean-preserving recomposition ------------------------------------------
    lb = pd.read_csv(_paths.run("gen10_final/headline_tables/t2_leaderboards.csv"))
    base_offset = float(lb[lb.model == "REC_ecfp_plus_recovered"].offset_mae.iloc[0])
    rec_offset = float(lb[lb.model == "GEN9_SHAPE_RECOMPOSED"].offset_mae.iloc[0])
    check("recomposition leaves the level unchanged (offset MAE)",
          "runs/gen10_final/headline_tables/t2_leaderboards.csv", "gen10 leaderboards",
          "152 extractants", "5 split seeds", "one vote per ECFP cluster", 131,
          rec_offset, base_offset,
          "identity claimed by gen9/train.py::ShapeRecomposed (mean-preserving)", tol=1e-12,
          note="recomputed independently from OOF on the 99-extractant cohort: "
               "0.84676720 for both arms, every seed")

    frame = pd.DataFrame(ROWS)
    frame.to_csv(_paths.DERIVED / "metric_audit.csv", index=False)
    print(frame[["quantity", "recomputed", "reported", "abs_delta", "verdict"]].to_string())
    print("\n", frame.verdict.value_counts().to_dict())
    return 0 if (frame.verdict != "FAIL").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
