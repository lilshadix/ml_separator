#!/usr/bin/env python3
"""Seen-extractant / unseen-conditions evaluation of the A2 champion.

The frozen benchmarks answer one question: generalization to *unseen
extractants* (grouped on extractant, primary metric equal-extractant macro
MAE). This script answers the complementary deployment question — "known
ligand, new process conditions" — by rerunning the same cohort, the same A2
column set and the same antisymmetric ExtraTrees estimator with folds grouped
on ``condition_id`` instead. Every extractant appears in training; every test
condition cell is unseen.

The two regimes are different questions and must never be mixed in one
leaderboard: unseen-extractant pooled R2 sits near +0.31 (oracle-calibration
upper bound ≈ +0.58), while this regime reaches ≈ +0.74 with the identical
model. Interpolation over conditions is where R2 in the 0.6–0.7 range
honestly lives on this dataset.

Writes ``oof_predictions.csv``, ``fold_metrics.csv`` and ``summary.json`` to
the output directory (default ``runs/condition_regime_<UTC>``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sklearn.model_selection import GroupKFold  # noqa: E402

from lanthanide_separation.evaluation import (  # noqa: E402
    AntisymmetricExtraTreesRegressor,
)
from lanthanide_separation.feature_registry import build_feature_registry  # noqa: E402
from lanthanide_separation.pairs import (  # noqa: E402
    PAIR_TARGET_COLUMN,
    build_lanthanide_pair_dataset,
)

DATASET_PATH = REPO_ROOT / "dataset with 3D structures/dataset.parquet"


def pooled_r2(truth: np.ndarray, prediction: np.ndarray) -> float:
    total = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - float(np.sum((truth - prediction) ** 2)) / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--model-seed", type=int, default=42)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=Warning, module="pandas")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / "runs" / f"condition_regime_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_parquet(DATASET_PATH)
    pair_data = build_lanthanide_pair_dataset(
        source,
        pair_scope="all",
        require_geometry=True,
        replicate_policy="unique",
        quarantine_known_bad=True,
        require_complete_conditions=True,
        delta3d_feature_set="compact-invariant",
    )
    frame = pair_data.frame.reset_index(drop=True)
    registry = build_feature_registry(pair_data)
    a2_columns = list(registry.ablation_columns("A2"))

    truth = frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
    conditions = frame["condition_id"].astype(str).to_numpy()
    extractant = frame["extractant"].astype(str).to_numpy()
    print(
        f"cohort: {len(frame)} pairs, {frame['extractant'].nunique()} extractants, "
        f"{len(set(conditions))} condition cells, {len(a2_columns)} A2 columns"
    )

    oof = np.full(len(truth), np.nan)
    fold_rows = []
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (train, test) in enumerate(
        splitter.split(frame, truth, groups=conditions)
    ):
        model = AntisymmetricExtraTreesRegressor(
            a2_columns,
            n_estimators=args.n_estimators,
            random_state=args.model_seed,
        )
        # Group-balanced weights stay on the extractant, matching the frozen
        # benchmark, so only the fold grouping differs between the regimes.
        model.fit(frame.iloc[train], truth[train], extractant[train])
        oof[test] = model.predict(frame.iloc[test])
        fold_rows.append(
            {
                "fold": fold,
                "n_test": int(len(test)),
                "n_test_condition_cells": int(len(set(conditions[test]))),
                "pooled_r2": pooled_r2(truth[test], oof[test]),
                "pooled_mae": float(np.abs(truth[test] - oof[test]).mean()),
            }
        )
        print(f"fold {fold}: {fold_rows[-1]}")

    macro_mae = (
        pd.DataFrame({"e": extractant, "a": np.abs(truth - oof)})
        .groupby("e")["a"]
        .mean()
        .mean()
    )
    per_extractant_r2 = []
    for _, block in pd.DataFrame(
        {"e": extractant, "y": truth, "p": oof}
    ).groupby("e"):
        if len(block) >= 20 and float(block["y"].std()) >= 0.2:
            per_extractant_r2.append(
                pooled_r2(block["y"].to_numpy(), block["p"].to_numpy())
            )

    summary = {
        "regime": "seen_extractant_unseen_conditions",
        "grouping": "condition_id",
        "not_comparable_with": "unseen-extractant leaderboards (different question)",
        "n_pairs": int(len(frame)),
        "n_extractants": int(frame["extractant"].nunique()),
        "n_condition_cells": int(len(set(conditions))),
        "n_feature_columns": len(a2_columns),
        "folds": int(args.folds),
        "model_seed": int(args.model_seed),
        "pooled_r2": pooled_r2(truth, oof),
        "pooled_mae": float(np.abs(truth - oof).mean()),
        "equal_extractant_macro_mae": float(macro_mae),
        "pearson_squared": float(np.corrcoef(truth, oof)[0, 1] ** 2),
        "median_extractant_r2_guarded": float(np.median(per_extractant_r2)),
        "extractants_in_r2": len(per_extractant_r2),
    }
    frame.assign(prediction=oof)[
        ["pair_id", "extractant", "condition_id", PAIR_TARGET_COLUMN, "prediction"]
    ].to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
