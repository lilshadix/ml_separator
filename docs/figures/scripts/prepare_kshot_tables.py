"""Derive every k-shot quantity the figures need from ONE frozen source table.

Source: ``runs/gen10_final/final_locked/kshot_detail.parquet`` — one row per
(global model, adapter, acquisition policy, k, extractant, split seed, repeat), the raw
output of the frozen k-shot harness.  12,834,720 rows.

Everything downstream (Figure 2, Figure 4C, Figure 5A, Figure 6, several supplements)
is computed here so that every figure uses the same cohort, the same aggregation and the
same support/query draws.  Nothing is transcribed from a report.

Writes, under ``figures/derived/``:

``kshot_per_ligand.csv``  mean MAE per (global model, adapter, policy, k, extractant),
                          restricted to the 99-ligand common cohort, with the ligand's
                          chemotype and nearest-training-neighbour Tanimoto.
``kshot_macro.csv``       macro MAE per (global model, adapter, policy, k) = mean over
                          the 99 ligands, with a chemotype-block bootstrap 95 % CI.
``kshot_cohort.json``     cohort definition and counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402
import _stats  # noqa: E402

SEEDS = (104729, 130363, 155921, 196613, 262147)
MIN_POOL, MIN_EVAL = 5, 2          # gen10's common-cohort rule (COMMON_MIN_POOL = 5)
COLUMNS = ["global_model", "adapter", "policy", "k", "extractant", "tanimoto_cluster",
           "nn_train_tanimoto", "split_seed", "repeat", "n_pool", "n_eval", "mae",
           "shape_mae", "offset", "spearman", "deployable"]


def main() -> int:
    src = _paths.run("gen10_final/final_locked/kshot_detail.parquet")
    detail = pd.read_parquet(src, columns=COLUMNS)
    detail = detail[detail["split_seed"].isin(SEEDS)]

    eligible = detail.groupby("extractant").apply(
        lambda b: bool((b["n_pool"] >= MIN_POOL).all() and (b["n_eval"] >= MIN_EVAL).all()),
        include_groups=False)
    cohort = sorted(eligible.index[eligible])
    common = detail[detail["extractant"].isin(cohort)]

    keys = ["global_model", "adapter", "policy", "k", "extractant"]
    per_ligand = common.groupby(keys, observed=True).agg(
        mae=("mae", "mean"), shape_mae=("shape_mae", "mean"), offset=("offset", "mean"),
        spearman=("spearman", "mean"), n_obs=("mae", "size"),
        tanimoto_cluster=("tanimoto_cluster", "first"),
        nn_train_tanimoto=("nn_train_tanimoto", "first"),
        deployable=("deployable", "first")).reset_index()
    per_ligand.to_csv(_paths.DERIVED / "kshot_per_ligand.csv", index=False)

    rows = []
    for key, block in per_ligand.groupby(["global_model", "adapter", "policy", "k"],
                                         observed=True):
        point, low, high = _stats.block_bootstrap_mean(block["mae"], block["tanimoto_cluster"])
        rows.append({"global_model": key[0], "adapter": key[1], "policy": key[2], "k": key[3],
                     "macro_mae": point, "ci95_low": low, "ci95_high": high,
                     "n_ligands": int(block["extractant"].nunique()),
                     "n_chemotypes": int(block["tanimoto_cluster"].nunique()),
                     "deployable": bool(block["deployable"].iloc[0])})
    macro = pd.DataFrame(rows).sort_values(["global_model", "adapter", "policy", "k"])
    macro.to_csv(_paths.DERIVED / "kshot_macro.csv", index=False)

    per_seed = common.groupby(["global_model", "adapter", "policy", "k", "split_seed"],
                              observed=True)["mae"].mean().reset_index()
    per_seed.to_csv(_paths.DERIVED / "kshot_per_seed.csv", index=False)

    meta = {
        "source": str(src.relative_to(_paths.ROOT)),
        "source_rows": int(len(detail)),
        "seeds": list(SEEDS),
        "repeats_per_seed": int(common["repeat"].nunique()),
        "common_cohort_rule": f"n_pool >= {MIN_POOL} and n_eval >= {MIN_EVAL} in every arm",
        "n_ligands_all": int(detail["extractant"].nunique()),
        "n_ligands_common": len(cohort),
        "n_chemotypes_common": int(common["tanimoto_cluster"].nunique()),
        "k_values": sorted(int(k) for k in common["k"].unique()),
        "adapters": sorted(common["adapter"].unique()),
        "policies": sorted(common["policy"].unique()),
        "global_models": sorted(common["global_model"].unique()),
    }
    (_paths.DERIVED / "kshot_cohort.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
