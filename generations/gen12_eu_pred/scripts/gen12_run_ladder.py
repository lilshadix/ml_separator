#!/usr/bin/env python
"""Phases 2-5: run the model ladder and write per-row out-of-fold predictions.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_run_ladder.py \
        [--design B] [--arms T1_CATBOOST,...] [--seeds 104729,...]

Predictions land in ``predictions/<design>/<arm>.parquet``; the per-fold
hyperparameter choices land beside them.  Re-running an arm overwrites only that
arm, so the ladder can be extended without recomputing what already ran.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import paths, splits  # noqa: E402
from gen12eu.chemistry import band_of  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402
from gen12eu.dmpnn import dmpnn_variants  # noqa: E402
from gen12eu.models import default_ladder  # noqa: E402
from gen12eu.runner import run_contender  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default="B", choices=["B", "A"])
    parser.add_argument("--arms", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--with-graph", action="store_true")
    args = parser.parse_args()

    cohort = build_cohort()
    frame = cohort.frame
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else splits.SPLIT_SEEDS
    folds = splits.all_folds(frame, design=args.design, seeds=seeds)
    integrity = splits.assert_fold_integrity(frame, folds)
    if not integrity["ok"]:
        raise SystemExit(f"fold integrity failed for design {args.design}")
    similarity = splits.similarity_table(frame, folds)
    similarity["band"] = band_of(similarity["max_train_tanimoto"]).to_numpy()

    out_dir = paths.PREDICTION_DIR / args.design
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fold_plan.json").write_text(json.dumps({
        "design": args.design, "seeds": list(seeds), "n_splits": splits.N_SPLITS,
        "cohort_fingerprint": cohort.fingerprint,
        "folds": [{"seed": f.seed, "fold": f.fold, "model_seed": f.model_seed,
                   "n_train": int(f.train_index.size), "n_test": int(f.test_index.size),
                   "held_out_groups": list(f.held_out_groups),
                   "test_row_ids": frame.iloc[f.test_index]["row_id"].tolist()}
                  for f in folds]}, indent=1))
    similarity.to_parquet(out_dir / "similarity.parquet", index=False)

    wanted = set(args.arms.split(",")) if args.arms else None
    ladder = default_ladder() + (dmpnn_variants() if args.with_graph else [])
    for contender in ladder:
        if wanted and contender.name not in wanted:
            continue
        print(f"[{args.design}] {contender.name}", flush=True)
        predictions, selection = run_contender(contender, frame, folds, cohort.blocks, similarity)
        predictions.to_parquet(out_dir / f"{contender.name}.parquet", index=False)
        selection.to_csv(out_dir / f"{contender.name}__selection.csv", index=False)
        macro = (predictions.assign(ae=(predictions["prediction"] - predictions["log_D"]).abs())
                 .groupby(["split_seed", "extractant"])["ae"].mean()
                 .groupby("split_seed").mean().mean())
        print(f"  -> extractant-macro MAE {macro:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
