"""GEN11 control, stage 3: assemble the locked control artefacts.

Stages 1 and 2 recomputed every headline number from the raw stored predictions;
this stage adds the strata cross-check (the one remaining ``final_locked`` table
whose inputs are row-level) and writes the four control files GEN11 compares against.
Claimed values are carried verbatim from the GEN10 brief so the delta column is
auditable without re-reading the model card.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/lilshadix/PycharmProjects/ml_separator/.claude/worktrees/"
            "lanthanide-separation-finalize-81154b")
sys.path.insert(0, str(ROOT / "src"))
from lanthanide_separation.gen9.metrics import macro_mae  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / "_intermediate"
SCRATCH.mkdir(parents=True, exist_ok=True)
LOCKED = ROOT / "runs" / "gen10_final" / "final_locked"
GEN8 = ROOT / "runs" / "gen8_architecture"
GEN10 = ROOT / "runs" / "gen10_final"
OUT = ROOT / "runs" / "gen11_transfer" / "control"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "GEN9_SHAPE_RECOMPOSED"
SEEDS = (104729, 130363, 155921, 196613, 262147)

stage1 = json.loads((SCRATCH / "stage1.json").read_text())
stage2 = json.loads((SCRATCH / "stage2.json").read_text())
pooled = pd.read_csv(SCRATCH / "shape_pooled_seeds.csv").set_index("axis_label")
by_seed = pd.read_csv(SCRATCH / "shape_mean_of_seeds.csv").set_index("axis_label")

# --- strata cross-check, FROZEN cohort, from the raw predictions ---------------
oof = pd.read_parquet(LOCKED / "oof_finalists.parquet")
oof = oof[oof["model"] == MODEL]
classes = pd.read_csv(GEN8 / "case_studies" / "ligand_failure_classes.csv")[
    ["extractant", "failure_class"]].set_index("extractant")["failure_class"]
row_cohorts = pd.read_csv(GEN10 / "data_ceiling" / "row_cohorts.csv").set_index("row_id")["cohort"]
stored_strata = pd.read_csv(LOCKED / "strata.csv")
stored_strata = stored_strata[(stored_strata["data_cohort"] == "FROZEN")
                              & (stored_strata["model"] == MODEL)].set_index("stratum")

strata_rows = []
for name in ("ALL", "RESIDUAL_SHAPE", "PURE_LEVEL", "ALREADY_GOOD", "PARTIAL_LEVEL",
             "MISMATCH_ROWS", "CLEAN_ROWS"):
    per_seed = []
    for seed in SEEDS:
        block = oof[oof["split_seed"] == seed]
        if name == "ALL":
            mask = np.ones(len(block), dtype=bool)
        elif name == "MISMATCH_ROWS":
            mask = (block["row_id"].astype(str).map(row_cohorts) == "B_MISMATCH").to_numpy()
        elif name == "CLEAN_ROWS":
            mask = (block["row_id"].astype(str).map(row_cohorts) == "A_CONSISTENT").to_numpy()
        else:
            mask = (block["extractant"].map(classes) == name).to_numpy()
        sub = block[mask]
        per_seed.append(macro_mae(sub))
    recomputed = float(np.mean(per_seed))
    stored = float(stored_strata.loc[name, "macro_mae"])
    strata_rows.append({"stratum": name, "recomputed_macro_mae": recomputed,
                        "stored_macro_mae": stored,
                        "abs_delta": abs(recomputed - stored),
                        "n_rows": int(stored_strata.loc[name, "n_rows"]),
                        "n_ligands": int(stored_strata.loc[name, "n_ligands"])})

# --- the claimed-vs-recomputed table ------------------------------------------
zs = stage1["zero_shot_level"]["mean_over_seeds"]
ex, ac, ms = pooled.loc["extractant"], pooled.loc["acid"], pooled.loc["metal_series"]
exs = by_seed.loc["extractant"]

zero_shot = [
    {"metric": "macro MAE (ECFP cluster, FROZEN cohort)", "claimed": 0.9695,
     "recomputed": zs["macro_mae_ecfp"], "aggregation": "mean of 5 per-seed values",
     "source": "oof_finalists.parquet row-level predictions"},
    {"metric": "offset MAE (per-ligand |mean residual|)", "claimed": 0.8212,
     "recomputed": zs["offset_mae"], "aggregation": "mean of 5 per-seed values",
     "source": "oof_finalists.parquet row-level predictions"},
    {"metric": "extractant-axis shape MAE", "claimed": 0.469,
     "recomputed": float(ex["shape_mae"]), "aggregation": "mean over all 775 curves (5 seeds)",
     "source": "curve_shape_table recomputed from oof_finalists + gen9 curve_membership"},
    {"metric": "extractant-axis predicted slope, median", "claimed": 1.024,
     "recomputed": float(ex["slope_pred_median"]),
     "aggregation": "median over all 775 curves pooled across seeds",
     "source": "curve_shape_table recomputed from oof_finalists + gen9 curve_membership",
     "alternative_aggregation": {
         "mean of 5 per-seed medians": float(exs["slope_pred_median"]),
         "note": "final_locked/shape_by_axis.csv averages per-seed summaries and so "
                 "reports 1.139; the claimed 1.024 is the seed-pooled median in "
                 "axis_representation/shape/shape_by_axis.csv"}},
    {"metric": "extractant-axis span recovery (median, guarded span>=0.5)", "claimed": 0.423,
     "recomputed": float(ex["span_recovery_median"]),
     "aggregation": "median over all 775 curves pooled across seeds",
     "source": "curve_shape_table recomputed from oof_finalists + gen9 curve_membership",
     "alternative_aggregation": {
         "mean of 5 per-seed medians": float(exs["span_recovery_median"]),
         "note": "final_locked/shape_by_axis.csv reports 0.417 by the per-seed-mean "
                 "convention"}},
    {"metric": "within-curve Spearman (extractant axis)", "claimed": 0.886,
     "recomputed": float(ex["within_curve_spearman"]),
     "aggregation": "mean over curves (identical under both conventions)",
     "source": "curve_shape_table recomputed from oof_finalists + gen9 curve_membership"},
    {"metric": "within-curve sign accuracy (extractant axis)", "claimed": 0.926,
     "recomputed": float(ex["within_curve_sign_accuracy"]),
     "aggregation": "mean over curves (identical under both conventions)",
     "source": "curve_shape_table recomputed from oof_finalists + gen9 curve_membership"},
    {"metric": "acid-axis shape MAE", "claimed": 0.545, "recomputed": float(ac["shape_mae"]),
     "aggregation": "mean over curves (identical under both conventions)",
     "source": "curve_shape_table recomputed from oof_finalists + gen9 curve_membership"},
    {"metric": "lanthanide (metal_series) axis shape MAE", "claimed": 0.291,
     "recomputed": float(ms["shape_mae"]),
     "aggregation": "mean over curves (identical under both conventions)",
     "source": "curve_shape_table recomputed from oof_finalists + gen9 curve_membership"},
]
for row in zero_shot:
    row["abs_delta"] = abs(row["recomputed"] - row["claimed"])
    row["reproduces_at_claimed_precision"] = bool(
        row["abs_delta"] <= 0.5 * 10 ** -(len(str(row["claimed"]).split(".")[1])))

frontier_rows = stage2["frontier"]
for row in frontier_rows:
    row["reproduces_at_claimed_precision"] = bool(row["abs_delta_vs_claimed"] <= 0.00005)

control = {
    "purpose": "GEN11 control: every GEN10 headline number re-derived here from the raw "
               "stored predictions in runs/gen10_final/final_locked, never copied from a "
               "summary table.",
    "generated_utc": pd.Timestamp.now("UTC").isoformat(),
    "frozen_model": MODEL,
    "identity": {
        "cohort": stage1["cohort"],
        "bundle": stage1["bundle"],
        "fold_reproduction": stage1["fold_reproduction"],
        "note_on_frame_width": "the GEN11 brief states the cohort frame is 5,248 x 2,492; "
                               "it is 5,248 x 5,642. 2,146 is the FROZEN arm's column count "
                               "(METAL 3 + COND 64 + ECFP 2048 + MASSACTION 8 + RECOVERED 23), "
                               "which is confirmed.",
    },
    "zero_shot_frozen_cohort": {
        "metrics": zero_shot,
        "per_seed_level": stage1["zero_shot_level"]["per_seed"],
        "pooled_mae_all_seeds": stage1["zero_shot_level"]["pooled_all_seeds_pooled_mae"],
        "curve_table_reproduction": stage1["curve_table_reproduction"],
    },
    "frontier_common_cohort": {
        "common_cohort_size_recomputed": stage2["common_cohort_size"],
        "common_cohort_size_claimed": stage2["common_cohort_claimed"],
        "common_cohort_rule": "every ligand whose (n_pool >= 5) and (n_eval >= 2) hold in "
                              "every (model, adapter, policy, k, seed, fold, repeat) record",
        "kshot_detail_rows": stage2["kshot_detail_rows"],
        "rows": frontier_rows,
        "full_table_agreement": stage2["frontier_common_full_agreement"],
        "zero_shot_arm_on_common_cohort": stage2["zero_shot_common_cohort"],
    },
    "strata_frozen_cohort": strata_rows,
    "verdict": {
        "cohort_fingerprint_reproduces": stage1["cohort"]["fingerprint_match"],
        "bundle_sha256_reproduces": stage1["bundle"]["match_claimed"],
        "folds_reproduce": stage1["fold_reproduction"]["all_folds_reproduce"],
        "curve_table_bit_identical": max(
            stage1["curve_table_reproduction"]["max_abs_delta"].values()) == 0.0,
        "frontier_reproduces": all(r["reproduces_at_claimed_precision"] for r in frontier_rows),
        "zero_shot_metrics_reproduce": all(
            r["reproduces_at_claimed_precision"] for r in zero_shot),
        "zero_shot_metrics_failing": [r["metric"] for r in zero_shot
                                      if not r["reproduces_at_claimed_precision"]],
        "strata_max_abs_delta": max(r["abs_delta"] for r in strata_rows),
    },
}
(OUT / "gen10_control.json").write_text(json.dumps(control, indent=2, default=float))

# --- the locked anchor table --------------------------------------------------
anchor = pd.DataFrame([{
    "k": r["k"], "pipeline_arm": r["arm"],
    "adapter": "ZERO_SHOT" if r["k"] == 0 else r["arm"].split("@")[0],
    "policy": "NONE" if r["k"] == 0 else r["arm"].split("@")[1],
    "gen10_macro_mae": r["recomputed_from_kshot_detail"],
    "claimed_in_brief": r["claimed"],
    "abs_delta": r["abs_delta_vs_claimed"],
    "n_ligands": r["n_ligands"],
    "best_deployable_arm": r["best_deployable_arm"],
    "best_deployable_mae": r["best_deployable_mae"],
} for r in frontier_rows])
gen8 = {0: 1.0605, 1: 0.6674, 2: 0.5879, 3: 0.5230, 5: 0.4743}
gen9 = {0: 1.0358, 1: 0.6539, 2: 0.5746, 3: 0.5109, 5: 0.4675}
anchor["gen8_frontier"] = anchor["k"].map(gen8)
anchor["gen9_frontier"] = anchor["k"].map(gen9)
anchor["global_model"] = MODEL
anchor["common_cohort_size"] = stage2["common_cohort_size"]
anchor.to_csv(OUT / "gen10_frontier.csv", index=False)

# --- the protocol -------------------------------------------------------------
protocol = {
    "purpose": "The evaluation protocol GEN11 must reuse unchanged for any number that "
               "is compared against the GEN10 control.",
    "cohort": {
        "path": "runs/gen7_architecture/cache/cohort.parquet",
        "loader": "lanthanide_separation.gen10.runner.prepared_cohort()",
        "fingerprint": stage1["cohort"]["fingerprint_recomputed"],
        "n_rows": stage1["cohort"]["n_rows"],
        "n_columns": stage1["cohort"]["n_columns"],
        "n_extractants": stage1["cohort"]["n_extractants"],
        "n_tanimoto_clusters": stage1["cohort"]["n_tanimoto_clusters"],
        "n_ecfp_clusters": stage1["cohort"]["n_ecfp_clusters"],
    },
    "frozen_bundle": {
        "path": "dataset with 3D structures/dataset.parquet",
        "sha256": stage1["bundle"]["sha256_recomputed"],
        "size_bytes": stage1["bundle"]["size_bytes"],
        "pinned_in": "gen3_protocol.json :: dataset_sha256",
        "matches_pin": stage1["bundle"]["match_gen3_protocol"],
    },
    "splits": {
        "seeds": list(SEEDS),
        "rule": "lanthanide_separation.gen6.cohorts.seeded_group_kfold(groups, n_splits, seed)",
        "group_column": "tanimoto_cluster",
        "n_splits": 5,
        "hash": "BLAKE2b (inside seeded_group_kfold); Python hash() is never used",
        "verified": "fold assignment recomputed for all 5 seeds and compared row-by-row "
                    "against oof_finalists.parquet: 0 mismatches in 5 x 5,248 rows",
    },
    "model_seed": {
        "formula": "42 + fold*1009 + 9999991",
        "base": 42, "fold_stride": 1009, "offset": 9999991,
        "independent_of_split_seed": True,
        "values_by_fold": {str(f): 42 + f * 1009 + 9_999_991 for f in range(5)},
        "definition": "lanthanide_separation.gen7.harness.Fold.model_seed",
    },
    "frozen_model_arm": {
        "name": MODEL,
        "blocks": ["METAL", "COND", "ECFP", "MASSACTION", "RECOVERED"],
        "n_columns": 2146,
        "level_target_column": "log_D",
    },
    "metrics": {
        "macro_mae": "lanthanide_separation.gen9.metrics.macro_mae, unit='ecfp_cluster'; "
                     "computed per split seed and averaged over the 5 seeds",
        "offset_mae": "lanthanide_separation.gen6.metrics.decompose_level_shape -> "
                      "summary['offset_mae'] = mean over ligands of |mean residual|",
        "shape_metrics": "lanthanide_separation.gen9.metrics.curve_shape_table + "
                         "summarise_shape over runs/gen9_shape/curves/curve_membership.parquet; "
                         "span_recovery guarded at true span >= 0.5",
        "shape_aggregation_warning": "final_locked/shape_by_axis.csv averages the per-seed "
                                     "summaries; the model-card slope median (1.024) and span "
                                     "recovery (0.423) are the seed-pooled medians. Pick one "
                                     "convention and state it.",
        "kshot": "lanthanide_separation.gen8.evaluate.evaluate_fewshot, repeats=12, "
                 "k in (0,1,2,3,5), policies (CENTRAL_THEN_SPREAD, MEDOID, D_OPTIMAL, RANDOM); "
                 "frontier = mean over ligands of the mean MAE over (seed, fold, repeat), "
                 "restricted to the 99-ligand common cohort",
    },
    "frozen_pipeline_rule": {
        "0": {"adapter": "ZERO_SHOT", "policy": "NONE"},
        "1": {"adapter": "SLOPE_L_s1_K1", "policy": "CENTRAL_THEN_SPREAD"},
        "2": {"adapter": "SERIES_ML", "policy": "CENTRAL_THEN_SPREAD"},
        "3": {"adapter": "SERIES_ML", "policy": "CENTRAL_THEN_SPREAD"},
        "5": {"adapter": "SERIES_ML", "policy": "CENTRAL_THEN_SPREAD"},
    },
    "environment": {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
}
try:
    import sklearn
    protocol["environment"]["scikit_learn"] = sklearn.__version__
except Exception:  # pragma: no cover
    pass
(OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, default=float))

print(json.dumps(control["verdict"], indent=2, default=float))
print()
print(pd.DataFrame(zero_shot)[["metric", "claimed", "recomputed", "abs_delta"]].to_string(index=False))
print()
print(anchor[["k", "pipeline_arm", "gen10_macro_mae", "claimed_in_brief", "abs_delta"]]
      .to_string(index=False))
print()
print(pd.DataFrame(strata_rows).to_string(index=False))
