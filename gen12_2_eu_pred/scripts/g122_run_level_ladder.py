#!/usr/bin/env python
"""Phases 4-7: the level ladder, on Gen12's frozen design-B folds.

Every arm sees identical folds, identical held-out extractants and an identical level
target; the only thing that differs is the feature block.  Predictions land in
``predictions/level/<definition>/<arm>.parquet`` with the per-fold selection log beside
them, so re-running one arm overwrites only that arm.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python \
        gen12_2_eu_pred/scripts/g122_run_level_ladder.py \
        [--definition LVL_MEAN] [--family extratrees] [--arms L7_ALL,...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import coordination, levelmodels, levelrunner, paths  # noqa: E402
from gen12eu import splits  # noqa: E402
from gen12eu.chemistry import band_of  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402


def frozen_fold_plan_matches(frame: pd.DataFrame, folds) -> dict:
    """Invariant 1: the design-B fold plan must be byte-identical to Gen12's."""
    plan = json.loads(paths.GEN12_FOLD_PLAN_B.read_text())
    if len(plan["folds"]) != len(folds):
        raise AssertionError("fold count differs from Gen12's plan")
    mismatches = 0
    for fold, recorded in zip(folds, plan["folds"]):
        if (fold.seed, fold.fold) != (recorded["seed"], recorded["fold"]):
            raise AssertionError("fold ordering differs from Gen12's plan")
        if frame.iloc[fold.test_index]["row_id"].tolist() != recorded["test_row_ids"]:
            mismatches += 1
        if list(fold.held_out_groups) != recorded["held_out_groups"]:
            mismatches += 1
    if mismatches:
        raise AssertionError(f"{mismatches} design-B folds differ from Gen12's frozen plan")
    return {"folds_checked": len(folds),
            "gen12_cohort_fingerprint": plan["cohort_fingerprint"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition", default="LVL_MEAN")
    parser.add_argument("--family", default="extratrees")
    parser.add_argument("--arms", default="")
    args = parser.parse_args()

    cohort = build_cohort()
    frame = cohort.frame
    folds = splits.all_folds(frame, design="B")
    plan_check = frozen_fold_plan_matches(frame, folds)
    if plan_check["gen12_cohort_fingerprint"] != cohort.fingerprint:
        raise SystemExit("cohort fingerprint differs from Gen12's")

    similarity = pd.read_parquet(paths.GEN12_PREDICTIONS / "B" / "similarity.parquet")
    similarity["band"] = band_of(similarity["max_train_tanimoto"]).to_numpy()

    table = pd.read_parquet(paths.FEATURE_DIR / "coordination_descriptors.parquet")
    audit = json.loads((paths.FEATURE_DIR / "coordination_audit.json").read_text())
    if audit["spec_sha256"] != coordination.spec_digest():
        raise SystemExit("the coordination specification changed after the matrix was built")
    if audit["feature_matrix_blake2b"] != paths.blake2b_of_frame(table):
        raise SystemExit("the coordination feature matrix on disk does not match its audit hash")

    structure, columns = levelrunner.structure_frame(frame, table, cohort.blocks)
    targets = levelrunner.level_targets_by_fold(frame, folds, cohort.blocks,
                                                definition=args.definition)

    out = paths.PREDICTION_DIR / "level" / args.definition
    out.mkdir(parents=True, exist_ok=True)
    if args.family != "extratrees":
        out = out / args.family
        out.mkdir(parents=True, exist_ok=True)
    (out / "run_manifest.json").write_text(json.dumps({
        "definition": args.definition, "family": args.family,
        "cohort_fingerprint": cohort.fingerprint,
        "fold_plan_matches_gen12": plan_check,
        "coordination_spec_sha256": audit["spec_sha256"],
        "coordination_matrix_blake2b": audit["feature_matrix_blake2b"],
        "block_sizes": {k: len(v) for k, v in columns.items()},
        "n_extractants": int(len(structure)),
    }, indent=1))

    wanted = set(args.arms.split(",")) if args.arms else None
    for contender in levelmodels.default_ladder(family=args.family):
        if wanted and contender.name not in wanted:
            continue
        predictions, selection = levelrunner.run_level_contender(
            contender, frame, structure, columns, folds, similarity,
            definition=args.definition, blocks=cohort.blocks, targets_by_fold=targets)
        predictions.to_parquet(out / f"{contender.name}.parquet", index=False)
        selection.to_csv(out / f"{contender.name}__selection.csv", index=False)
        mae = (predictions.groupby(["split_seed", "extractant"])["level_abs_error"].mean()
               .groupby("split_seed").mean().mean())
        inner = selection["inner_level_mae"].mean() if "inner_level_mae" in selection else np.nan
        print(f"{contender.name:20s} level MAE {mae:.4f}   inner-validation {inner:.4f}   "
              f"clipped {int(selection['n_clipped'].sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
