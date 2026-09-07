#!/usr/bin/env python
"""Phase 10: the chemical frontier, in continuous similarity as well as in bands.

The band table alone cannot say whether level error rises *monotonically* with chemical
distance, and a band average can hide a gradient.  This reports the level error of every
arm in fixed similarity sextiles and the Spearman correlation between an extractant's
train-similarity and its level error, for the level task and for the full-prediction arms.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_frontier.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import levelmetrics, paths  # noqa: E402
from gen12eu.metrics import per_extractant  # noqa: E402


def deciles(units: dict[str, pd.DataFrame], statistic: str, label: str) -> pd.DataFrame:
    rows = []
    for arm, table in units.items():
        block = table.dropna(subset=["max_train_tanimoto"]).copy()
        block["bin"] = pd.qcut(block["max_train_tanimoto"], 6, duplicates="drop")
        for edge, piece in block.groupby("bin", observed=True):
            rows.append({"task": label, "arm": arm, "bin": str(edge),
                         "similarity_mid": float(piece["max_train_tanimoto"].mean()),
                         "value": float(piece.groupby("split_seed")[statistic].mean().mean()),
                         "n_extractants": int(piece["extractant"].nunique())})
        rows.append({"task": label, "arm": arm, "bin": "spearman(similarity, error)",
                     "similarity_mid": np.nan,
                     "value": float(block["max_train_tanimoto"].corr(block[statistic],
                                                                     method="spearman")),
                     "n_extractants": int(block["extractant"].nunique())})
    return pd.DataFrame(rows)


def main() -> int:
    tables = []
    directory = paths.PREDICTION_DIR / "level" / "LVL_MEAN"
    level_units = {}
    for path in sorted(directory.glob("*.parquet")):
        frame = pd.read_parquet(path)
        level_units[frame["arm"].iloc[0]] = levelmetrics.per_extractant(frame)
    tables.append(deciles(level_units, "mae", "level"))

    full = paths.PREDICTION_DIR / "full"
    full_units = {}
    for path in sorted(full.glob("*.parquet")):
        frame = pd.read_parquet(path)
        full_units[frame["arm"].iloc[0]] = per_extractant(frame)
    for arm, relative in (("GEN12_ABL_D", "B_ablation/ABL_D_PLUS_LIG2D"),
                          ("GEN12_ABL_A", "B_ablation/ABL_A_CONDITIONS")):
        frame = pd.read_parquet(paths.GEN12_PREDICTIONS / f"{relative}.parquet")
        full_units[arm] = per_extractant(frame)
    if full_units:
        tables.append(deciles(full_units, "mae", "full_logD"))

    frontier = pd.concat(tables, ignore_index=True)
    out = paths.METRIC_DIR / "frontier_sextiles.csv"
    frontier.to_csv(out, index=False)

    for task in frontier["task"].unique():
        block = frontier[frontier["task"] == task]
        print(f"\n=== {task}: MAE by train-similarity sextile ===")
        print(block[block["bin"] != "spearman(similarity, error)"]
              .pivot_table(index="bin", columns="arm", values="value").round(3).to_string())
        print(f"\n=== {task}: Spearman(similarity, per-extractant error) ===")
        print(block[block["bin"] == "spearman(similarity, error)"][["arm", "value"]]
              .round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
