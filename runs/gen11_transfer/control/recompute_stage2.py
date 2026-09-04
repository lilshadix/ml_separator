"""GEN11 control, stage 2: re-derive the GEN10 k-shot frontier from the raw k-shot records.

``kshot_detail.parquet`` is the rawest artefact the frozen run kept for the few-shot
protocol: one row per (global model, adapter, policy, k, ligand, split seed, fold,
repeat) carrying that evaluation's own MAE.  The published frontier is two group-means
away from it, and both are reproduced here rather than read from ``frontier_common.csv``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("/Users/lilshadix/PycharmProjects/ml_separator/.claude/worktrees/"
            "lanthanide-separation-finalize-81154b")
sys.path.insert(0, str(ROOT / "src"))
SCRATCH = Path(__file__).resolve().parent / "_intermediate"
SCRATCH.mkdir(parents=True, exist_ok=True)
LOCKED = ROOT / "runs" / "gen10_final" / "final_locked"
K_VALUES = (0, 1, 2, 3, 5)
SELECTED = "GEN9_SHAPE_RECOMPOSED"
CLAIMED = {0: 1.0358, 1: 0.6539, 2: 0.5593, 3: 0.4928, 5: 0.4405}
# the frozen rule, from pipeline_frontier.json (checked against gen10_final_locked.py)
RULE = {0: ("ZERO_SHOT", "NONE"),
        1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
        2: ("SERIES_ML", "CENTRAL_THEN_SPREAD"),
        3: ("SERIES_ML", "CENTRAL_THEN_SPREAD"),
        5: ("SERIES_ML", "CENTRAL_THEN_SPREAD")}

columns = ["global_model", "extractant", "adapter", "policy", "k", "mae", "offset",
           "shape_mae", "spearman", "sign_accuracy", "n_pool", "n_eval", "split_seed",
           "repeat", "deployable"]
table = pq.read_table(LOCKED / "kshot_detail.parquet", columns=columns)
detail = table.to_pandas()
del table
for c in ("global_model", "extractant", "adapter", "policy"):
    detail[c] = detail[c].astype("category")

report: dict = {"kshot_detail_rows": int(len(detail))}

# --- the common cohort: every ligand with >=5 pool rows and >=2 eval rows everywhere ---
mins = detail.groupby("extractant", observed=True)[["n_pool", "n_eval"]].min()
cohort_ligands = set(mins.index[(mins["n_pool"] >= 5) & (mins["n_eval"] >= 2)])
report["common_cohort_size"] = len(cohort_ligands)
report["common_cohort_claimed"] = 99

common = detail[detail["extractant"].isin(cohort_ligands)].copy()
common["arm"] = np.where(common["adapter"] == "ZERO_SHOT_REF", "ZERO_SHOT",
                         common["adapter"].astype(str) + "@" + common["policy"].astype(str))
per_ligand = common.groupby(["global_model", "arm", "adapter", "policy", "k", "extractant"],
                            observed=True)["mae"].mean().reset_index()
table_b = per_ligand.groupby(["global_model", "arm", "adapter", "policy", "k"],
                             observed=True).agg(
    mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()
table_b.to_csv(SCRATCH / "recomputed_frontier_common.csv", index=False)

# --- the pipeline frontier ----------------------------------------------------
stored_common = pd.read_csv(LOCKED / "frontier_common.csv")
stored_best = pd.read_csv(LOCKED / "frontier_best.csv")
rows = []
for k in K_VALUES:
    adapter, policy = RULE[k]
    arm = "ZERO_SHOT" if k == 0 else f"{adapter}@{policy}"
    block = table_b[(table_b["global_model"] == SELECTED) & (table_b["k"] == k)]
    hit = block[block["arm"] == arm]
    recomputed = float(hit["mae"].iloc[0])
    deployable = block[~block["policy"].astype(str).str.startswith("ORACLE")]
    best = deployable.sort_values("mae").iloc[0]
    stored_hit = stored_common[(stored_common["global_model"] == SELECTED)
                               & (stored_common["k"] == k)
                               & (stored_common["arm"] == arm)]
    stored_pipeline = stored_best[(stored_best["global_model"] == SELECTED)
                                  & (stored_best["k"] == k)]["pipeline_mae"].iloc[0]
    rows.append({
        "k": k, "arm": arm,
        "claimed": CLAIMED[k],
        "recomputed_from_kshot_detail": recomputed,
        "stored_frontier_common_csv": float(stored_hit["mae"].iloc[0]) if len(stored_hit)
        else float("nan"),
        "stored_frontier_best_csv": float(stored_pipeline),
        "abs_delta_vs_claimed": abs(recomputed - CLAIMED[k]),
        "abs_delta_vs_stored": abs(recomputed - float(stored_hit["mae"].iloc[0]))
        if len(stored_hit) else float("nan"),
        "n_ligands": int(hit["n_ligands"].iloc[0]),
        "best_deployable_arm": str(best["arm"]),
        "best_deployable_mae": float(best["mae"]),
    })
frontier = pd.DataFrame(rows)
report["frontier"] = frontier.to_dict(orient="records")

# --- k = 0 sanity: the zero-shot arm's per-ligand offset / shape on the common cohort ---
zero = common[(common["global_model"] == SELECTED) & (common["k"] == 0)
              & (common["arm"] == "ZERO_SHOT")]
report["zero_shot_common_cohort"] = {
    "n_records": int(len(zero)),
    "macro_mae": float(zero.groupby("extractant", observed=True)["mae"].mean().mean()),
    "offset_mean": float(zero.groupby("extractant", observed=True)["offset"].mean().mean()),
    "shape_mae_mean": float(zero.groupby("extractant", observed=True)["shape_mae"].mean().mean()),
    "spearman_mean": float(zero.groupby("extractant", observed=True)["spearman"].mean().mean()),
    "sign_accuracy_mean": float(
        zero.groupby("extractant", observed=True)["sign_accuracy"].mean().mean()),
}

# --- full stored-vs-recomputed agreement over every arm ------------------------
merged = stored_common.merge(table_b, on=["global_model", "arm", "adapter", "policy", "k"],
                             suffixes=("_stored", "_new"), how="outer", indicator=True)
report["frontier_common_full_agreement"] = {
    "n_stored": int(len(stored_common)), "n_recomputed": int(len(table_b)),
    "both": int((merged["_merge"] == "both").sum()),
    "stored_only": int((merged["_merge"] == "left_only").sum()),
    "recomputed_only": int((merged["_merge"] == "right_only").sum()),
    "max_abs_delta_mae": float(np.nanmax(np.abs(merged["mae_stored"] - merged["mae_new"]))),
    "n_ligands_mismatch": int((merged["n_ligands_stored"] != merged["n_ligands_new"]).sum()),
}

(SCRATCH / "stage2.json").write_text(json.dumps(report, indent=2, default=float))
print(json.dumps(report, indent=2, default=float))
