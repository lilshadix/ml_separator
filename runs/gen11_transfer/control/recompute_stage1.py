"""GEN11 control, stage 1: re-derive the GEN10 zero-shot headline from raw predictions.

Nothing here reads a summary table.  The level metrics come from the row-level
out-of-fold predictions in ``final_locked/oof_finalists.parquet`` and the shape
metrics are recomputed by running gen9's ``curve_shape_table`` over those same rows
and gen9's curve membership, so a mismatch against the stored CSVs is a real
discrepancy and not a copy of one.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/lilshadix/PycharmProjects/ml_separator/.claude/worktrees/"
            "lanthanide-separation-finalize-81154b")
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402
from lanthanide_separation.gen6.metrics import decompose_level_shape as level_shape_decomposition  # noqa: E402
from lanthanide_separation.gen9.metrics import (  # noqa: E402
    curve_shape_table, macro_mae, summarise_shape,
)
from lanthanide_separation.gen10.runner import prepared_cohort  # noqa: E402

OUT = ROOT / "runs" / "gen11_transfer" / "control"
SCRATCH = Path(__file__).resolve().parent / "_intermediate"
SCRATCH.mkdir(parents=True, exist_ok=True)
LOCKED = ROOT / "runs" / "gen10_final" / "final_locked"
SEEDS = (104729, 130363, 155921, 196613, 262147)
MODEL = "GEN9_SHAPE_RECOMPOSED"

report: dict = {}

# --- 1. cohort identity -------------------------------------------------------
cohort = prepared_cohort()
frame = cohort.frame
report["cohort"] = {
    "fingerprint_recomputed": cohort.fingerprint,
    "fingerprint_claimed": "bed178ec1a7a82b0",
    "fingerprint_match": cohort.fingerprint == "bed178ec1a7a82b0",
    "n_rows": int(len(frame)),
    "n_columns": int(frame.shape[1]),
    "n_extractants": int(frame["extractant"].nunique()),
    "n_tanimoto_clusters": int(frame["tanimoto_cluster"].nunique()),
    "n_ecfp_clusters": int(frame["ecfp_cluster"].nunique()),
}

# --- 2. frozen bundle digest ---------------------------------------------------
bundle = ROOT / "dataset with 3D structures" / "dataset.parquet"
h = hashlib.sha256()
with bundle.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 22), b""):
        h.update(chunk)
digest = h.hexdigest()
protocol = json.loads((ROOT / "gen3_protocol.json").read_text())


def find_digest(obj, hits):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and len(value) == 64 and all(
                    c in "0123456789abcdef" for c in value):
                hits.append((key, value))
            else:
                find_digest(value, hits)
    elif isinstance(obj, list):
        for value in obj:
            find_digest(value, hits)


hits: list = []
find_digest(protocol, hits)
pinned = [v for _, v in hits]
report["bundle"] = {
    "path": str(bundle),
    "sha256_recomputed": digest,
    "sha256_claimed": "fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd",
    "match_claimed": digest == ("fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084"
                                "cc1df4e024faf5dd"),
    "gen3_protocol_digests": hits,
    "match_gen3_protocol": digest in pinned,
    "size_bytes": bundle.stat().st_size,
}

# --- 3. fold reproduction ------------------------------------------------------
oof = pd.read_parquet(LOCKED / "oof_finalists.parquet")
groups = frame["tanimoto_cluster"].astype(str).to_numpy()
row_ids = frame["row_id"].astype(str).to_numpy()
fold_checks = []
for seed in SEEDS:
    assign = np.full(len(frame), -1, dtype=int)
    for k, (_, test_index) in enumerate(seeded_group_kfold(groups, 5, seed)):
        assign[test_index] = k
    recomputed = pd.Series(assign, index=row_ids)
    block = oof[(oof["model"] == MODEL) & (oof["split_seed"] == seed)]
    stored = block.set_index(block["row_id"].astype(str))["fold"].astype(int)
    aligned = recomputed.reindex(stored.index)
    fold_checks.append({"split_seed": int(seed), "n_rows": int(len(stored)),
                        "n_fold_mismatches": int((aligned != stored).sum()),
                        "unassigned": int((assign < 0).sum())})
report["fold_reproduction"] = {
    "rule": "seeded_group_kfold(tanimoto_cluster, n_splits=5, seed)",
    "per_seed": fold_checks,
    "all_folds_reproduce": all(c["n_fold_mismatches"] == 0 for c in fold_checks),
}

# --- 4. zero-shot level metrics from the raw predictions -----------------------
per_seed = []
for seed in SEEDS:
    block = oof[(oof["model"] == MODEL) & (oof["split_seed"] == seed)]
    dec = level_shape_decomposition(block["log_D"], block["prediction"], block["extractant"])
    per_seed.append({
        "split_seed": int(seed),
        "n_rows": int(len(block)),
        "macro_mae_ecfp": macro_mae(block),
        "offset_mae": float(dec.summary["offset_mae"]),
        "shape_mae_ligand": float(dec.summary["shape_mae"]),
        "pooled_mae": float(np.abs(block["prediction"] - block["log_D"]).mean()),
        "macro_mae_ligand": float(dec.summary["macro_mae_ligand"]),
    })
level = pd.DataFrame(per_seed)
report["zero_shot_level"] = {
    "model": MODEL,
    "per_seed": per_seed,
    "mean_over_seeds": {c: float(level[c].mean()) for c in
                        ("macro_mae_ecfp", "offset_mae", "shape_mae_ligand", "pooled_mae",
                         "macro_mae_ligand")},
    "pooled_all_seeds_macro_mae": macro_mae(oof[oof["model"] == MODEL]),
    "pooled_all_seeds_pooled_mae": float(np.abs(
        oof.loc[oof["model"] == MODEL, "prediction"]
        - oof.loc[oof["model"] == MODEL, "log_D"]).mean()),
}

# --- 5. shape metrics recomputed from the raw predictions ----------------------
membership = pd.read_parquet(ROOT / "runs" / "gen9_shape" / "curves" / "curve_membership.parquet")
tables = []
summaries = []
for seed in SEEDS:
    block = oof[(oof["model"] == MODEL) & (oof["split_seed"] == seed)]
    table = curve_shape_table(block, membership)
    table["split_seed"] = seed
    tables.append(table)
    summary = summarise_shape(table)
    summary.insert(0, "split_seed", seed)
    summaries.append(summary)
curves = pd.concat(tables, ignore_index=True)
curves.to_parquet(SCRATCH / "recomputed_curve_shape.parquet", index=False)
per_seed_shape = pd.concat(summaries, ignore_index=True)
mean_of_seeds = per_seed_shape.groupby("axis_label").mean(numeric_only=True).reset_index()
pooled = summarise_shape(curves)
mean_of_seeds.to_csv(SCRATCH / "shape_mean_of_seeds.csv", index=False)
pooled.to_csv(SCRATCH / "shape_pooled_seeds.csv", index=False)

# agreement with the stored per-curve table
stored_curves = pd.read_parquet(LOCKED / "curve_shape.parquet")
stored_curves = stored_curves[(stored_curves["data_cohort"] == "FROZEN")
                              & (stored_curves["model"] == MODEL)]
key = ["curve_id", "split_seed"]
joined = curves.merge(stored_curves, on=key, suffixes=("_new", "_old"), how="outer",
                      indicator=True)
numeric = ["shape_mae", "slope_pred", "slope_true", "span_recovery", "spearman",
           "sign_accuracy", "row_mae"]
deltas = {c: float(np.nanmax(np.abs(joined[f"{c}_new"] - joined[f"{c}_old"]))) for c in numeric}
report["curve_table_reproduction"] = {
    "n_curves_recomputed": int(len(curves)),
    "n_curves_stored": int(len(stored_curves)),
    "join_both": int((joined["_merge"] == "both").sum()),
    "join_left_only": int((joined["_merge"] == "left_only").sum()),
    "join_right_only": int((joined["_merge"] == "right_only").sum()),
    "max_abs_delta": deltas,
}

report["zero_shot_shape"] = {
    "pooled_over_seeds": pooled.set_index("axis_label").to_dict(orient="index"),
    "mean_of_per_seed": mean_of_seeds.set_index("axis_label").to_dict(orient="index"),
}

(SCRATCH / "stage1.json").write_text(json.dumps(report, indent=2, default=float))
print(json.dumps({k: v for k, v in report.items() if k != "zero_shot_shape"},
                 indent=2, default=float))
print("\n--- pooled-over-seeds shape, extractant/acid/metal_series ---")
cols = ["axis_label", "n_curves", "n_curves_guarded", "n_ligands", "shape_mae",
        "slope_pred_median", "slope_true_median", "span_recovery_median",
        "span_recovery_median_unguarded", "within_curve_spearman",
        "within_curve_sign_accuracy", "slope_mae"]
print(pooled[pooled["axis_label"].isin(["extractant", "acid", "metal_series"])][cols]
      .to_string(index=False))
print("\n--- mean-of-per-seed shape ---")
print(mean_of_seeds[mean_of_seeds["axis_label"].isin(["extractant", "acid", "metal_series"])]
      [cols].to_string(index=False))
