#!/usr/bin/env python
"""Phase 8: the similarity frontier, in the form the central claim needs.

Two things the band table alone cannot say: whether error rises *monotonically*
with chemical distance, and whether the molecular arms beat the no-chemistry
arm *inside* each band with a paired interval.  Both are computed here on
matched evaluation units.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_frontier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import inference, paths  # noqa: E402
from gen12eu.metrics import per_extractant  # noqa: E402

ARMS = {
    "T1_EXTRATREES": paths.PREDICTION_DIR / "B" / "T1_EXTRATREES.parquet",
    "B1_COND_ONLY": paths.PREDICTION_DIR / "B" / "B1_COND_ONLY.parquet",
    "B2_NN_CHEMICAL": paths.PREDICTION_DIR / "B" / "B2_NN_CHEMICAL.parquet",
    "ABL_A_CONDITIONS": paths.PREDICTION_DIR / "B_ablation" / "ABL_A_CONDITIONS.parquet",
    "ABL_D_PLUS_LIG2D": paths.PREDICTION_DIR / "B_ablation" / "ABL_D_PLUS_LIG2D.parquet",
}
for name in ("T3_DMPNN_COND", "T3_DMPNN_COND_DESC", "T3_DMPNN_GRAPH_ONLY"):
    ARMS[name] = paths.PREDICTION_DIR / "B" / f"{name}.parquet"


def main() -> int:
    frames = {a: pd.read_parquet(p) for a, p in ARMS.items() if p.exists()}
    units = {a: per_extractant(f) for a, f in frames.items()}

    # --- error against distance, in fixed similarity deciles ---------------- #
    rows = []
    for arm, table in units.items():
        block = table.dropna(subset=["max_train_tanimoto"]).copy()
        block["decile"] = pd.qcut(block["max_train_tanimoto"], 6, duplicates="drop")
        for decile, piece in block.groupby("decile", observed=True):
            rows.append({"arm": arm, "bin": str(decile),
                         "similarity_mid": float(piece["max_train_tanimoto"].mean()),
                         "macro_mae": float(piece.groupby("split_seed")["mae"].mean().mean()),
                         "n_extractant_seeds": int(len(piece)),
                         "n_extractants": int(piece["extractant"].nunique())})
        rows.append({"arm": arm, "bin": "spearman(similarity, error)",
                     "similarity_mid": np.nan,
                     "macro_mae": float(block["max_train_tanimoto"].corr(
                         block["mae"], method="spearman")),
                     "n_extractant_seeds": int(len(block)),
                     "n_extractants": int(block["extractant"].nunique())})
    frontier = pd.DataFrame(rows)
    frontier.to_csv(paths.METRIC_DIR / "B" / "frontier_deciles.csv", index=False)

    # --- inside each band, every arm against the no-chemistry arm ----------- #
    reference = "ABL_A_CONDITIONS" if "ABL_A_CONDITIONS" in units else "B1_COND_ONLY"
    tables = []
    for band in ("far", "mid", "near"):
        subset = {a: t[t["band"] == band] for a, t in units.items()}
        shared = set.intersection(*[set(t["extractant"]) for t in subset.values()])
        subset = {a: t[t["extractant"].isin(shared)] for a, t in subset.items()}
        if min(len(t) for t in subset.values()) < 5:
            continue
        block = inference.unit_table(subset, statistic="mae")
        comparisons = {f"{reference}_vs_{a}": (reference, a) for a in subset if a != reference}
        table = inference.paired_bootstrap(block, comparisons, statistic="mae")
        table["band"] = band
        table["n_blocks"] = block["chemotype"].nunique()
        for arm, piece in subset.items():
            table[f"macro_{arm}"] = piece.groupby("split_seed")["mae"].mean().mean()
        tables.append(table)
    if tables:
        out = pd.concat(tables, ignore_index=True)
        out.to_csv(paths.BOOTSTRAP_DIR / "B_frontier_by_band.csv", index=False)
        print(out[["band", "candidate", "point_delta", "bca_low", "bca_high",
                   "bca_excludes_zero", "units_total", "n_blocks"]].round(4).to_string(index=False))

    print("\nmacro MAE by similarity bin:")
    print(frontier[frontier["bin"] != "spearman(similarity, error)"]
          .pivot_table(index="bin", columns="arm", values="macro_mae").round(3).to_string())
    print("\nSpearman(similarity, per-extractant error):")
    print(frontier[frontier["bin"] == "spearman(similarity, error)"]
          [["arm", "macro_mae"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
