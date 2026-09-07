#!/usr/bin/env python
"""Phase 11: feature ablations on the winning architecture.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_ablation.py [--design B]

Six pre-registered arms, one learner, identical folds and identical test rows, so
the difference between two arms is the information and nothing else.  Arm F
(graph + conditions) is not run here — it is the Tier-3 D-MPNN and is compared
from its own predictions, because swapping the learner as well as the
representation would confound the two.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import inference, paths, splits  # noqa: E402
from gen12eu.chemistry import band_of  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402
from gen12eu.metrics import by_band, per_extractant, summarise  # noqa: E402
from gen12eu.models import TreeContender  # noqa: E402
from gen12eu.runner import run_contender  # noqa: E402

ARMS = {
    "ABL_A_CONDITIONS": ("COND", "MASSACT"),
    "ABL_B_MOLECULE": ("ECFP", "PHYSCHEM", "DONORS"),
    "ABL_C_MOL_COND": ("COND", "MASSACT", "ECFP", "PHYSCHEM", "DONORS"),
    "ABL_D_PLUS_LIG2D": ("COND", "MASSACT", "ECFP", "PHYSCHEM", "DONORS", "LIG2D"),
    "ABL_E_ECFP_COND": ("COND", "MASSACT", "ECFP"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default="B")
    parser.add_argument("--family", default="extratrees")
    args = parser.parse_args()

    cohort = build_cohort()
    frame = cohort.frame
    folds = splits.all_folds(frame, design=args.design)
    similarity = splits.similarity_table(frame, folds)
    similarity["band"] = band_of(similarity["max_train_tanimoto"]).to_numpy()
    out = paths.PREDICTION_DIR / f"{args.design}_ablation"
    out.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for name, blocks in ARMS.items():
        contender = TreeContender(family=args.family, name=name, blocks=blocks)
        print(f"[{args.design}] {name} blocks={blocks}", flush=True)
        predictions, selection = run_contender(contender, frame, folds, cohort.blocks,
                                               similarity, verbose=False)
        predictions.to_parquet(out / f"{name}.parquet", index=False)
        selection.to_csv(out / f"{name}__selection.csv", index=False)
        frames[name] = predictions
        print(f"  -> {summarise(predictions, label=name)['macro_mae_extractant']:.4f}", flush=True)

    graph = paths.PREDICTION_DIR / args.design / "T3_DMPNN_COND.parquet"
    if graph.exists():
        frames["ABL_F_GRAPH_COND"] = pd.read_parquet(graph)

    metric_dir = paths.METRIC_DIR / f"{args.design}_ablation"
    metric_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame([summarise(f, label=a) for a, f in frames.items()]).sort_values(
        "macro_mae_extractant", ignore_index=True)
    leaderboard.to_csv(metric_dir / "leaderboard.csv", index=False)
    pd.concat([by_band(f, label=a) for a, f in frames.items()], ignore_index=True).to_csv(
        metric_dir / "bands.csv", index=False)

    units = {a: per_extractant(f) for a, f in frames.items()}
    per_unit = inference.unit_table(units, statistic="mae")
    reference = "ABL_C_MOL_COND"
    comparisons = {f"{reference}_vs_{a}": (reference, a) for a in frames if a != reference}
    inference.paired_bootstrap(per_unit, comparisons, statistic="mae").to_csv(
        paths.BOOTSTRAP_DIR / f"{args.design}_ablation_pairwise.csv", index=False)
    print(leaderboard[["arm", "macro_mae_extractant", "macro_mae_chemotype",
                       "offset_mae", "shape_mae"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
