#!/usr/bin/env python
"""Phases 8-9: the decomposed predictor, and direct versus decomposed on identical queries.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_decomposed.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import decomposed, levelrunner, paths  # noqa: E402
from gen12eu import splits  # noqa: E402
from gen12eu.chemistry import band_of  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402
from gen12eu.models import TreeContender  # noqa: E402
from gen12eu.runner import run_contender  # noqa: E402

#: Level representation x shape representation.  ``DEC_L3_S1`` and ``DEC_L7_S1`` are the
#: secondary endpoint: the *only* difference between them is whether the level model sees
#: the coordination block.
ARMS: dict[str, tuple[tuple[str, ...], str]] = {
    "DEC_L3_S1": (("ECFP", "PHYSCHEM", "DONORS", "LIG2D"), "S1"),
    "DEC_L7_S1": (("ECFP", "PHYSCHEM", "DONORS", "LIG2D", "COORD"), "S1"),
    "DEC_L7_S0": (("ECFP", "PHYSCHEM", "DONORS", "LIG2D", "COORD"), "S0"),
    "DEC_L7_S2": (("ECFP", "PHYSCHEM", "DONORS", "LIG2D", "COORD"), "S2"),
    "DEC_L7_S3": (("ECFP", "PHYSCHEM", "DONORS", "LIG2D", "COORD"), "S3"),
    "DEC_L4_S1": (("COORD",), "S1"),
    "DEC_L6_S1": (("PHYSCHEM", "DONORS", "LIG2D", "COORD"), "S1"),
}
#: Gen12's molecular contract plus the coordination block, in Gen12's own direct learner.
DIRECT_BLOCKS = ("COND", "MASSACT", "ECFP", "PHYSCHEM", "DONORS", "LIG2D")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="")
    parser.add_argument("--definition", default="LVL_MEAN")
    args = parser.parse_args()

    cohort = build_cohort()
    frame = cohort.frame
    folds = splits.all_folds(frame, design="B")
    similarity = pd.read_parquet(paths.GEN12_PREDICTIONS / "B" / "similarity.parquet")
    similarity["band"] = band_of(similarity["max_train_tanimoto"]).to_numpy()
    table = pd.read_parquet(paths.FEATURE_DIR / "coordination_descriptors.parquet")
    structure, columns = levelrunner.structure_frame(frame, table, cohort.blocks)
    # A shape contract may use the coordination block, so the row-level frame needs it
    # too.  ``merge(how="left")`` on a many-to-one key preserves row order and length, so
    # the fold indices, which are positional into ``frame``, stay valid — asserted below.
    merged = frame.merge(table, left_on="extractant", right_index=True, how="left",
                         validate="many_to_one")
    if len(merged) != len(frame) or not (merged["row_id"].to_numpy()
                                         == frame["row_id"].to_numpy()).all():
        raise SystemExit("merging the coordination block reordered the cohort rows")
    blocks = dict(cohort.blocks)
    blocks["COORD"] = tuple(table.columns)
    targets = levelrunner.level_targets_by_fold(merged, folds, blocks,
                                                definition=args.definition)
    out = paths.PREDICTION_DIR / "full"
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.arms.split(",")) if args.arms else None

    for name, (level_blocks, shape) in ARMS.items():
        if wanted and name not in wanted:
            continue
        contender = decomposed.DecomposedContender(name=name, level_blocks=level_blocks,
                                                   shape=shape)
        predictions, selection = decomposed.run_decomposed(
            contender, merged, structure, columns, blocks, folds, similarity, targets)
        predictions.to_parquet(out / f"{name}.parquet", index=False)
        selection.to_csv(out / f"{name}__selection.csv", index=False)
        macro = (predictions.assign(ae=(predictions["prediction"] - predictions["log_D"]).abs())
                 .groupby(["split_seed", "extractant"])["ae"].mean()
                 .groupby("split_seed").mean().mean())
        print(f"{name:12s} extractant-macro MAE {macro:.4f}", flush=True)

    # ---- the direct comparator, in Gen12's own learner and runner ----------- #
    if not wanted or "DIRECT_ALL" in wanted:
        contender = TreeContender(family="extratrees", name="DIRECT_ALL",
                                  blocks=DIRECT_BLOCKS + ("COORD",))
        predictions, selection = run_contender(contender, merged, folds, blocks, similarity,
                                               verbose=False)
        predictions.to_parquet(out / "DIRECT_ALL.parquet", index=False)
        selection.to_csv(out / "DIRECT_ALL__selection.csv", index=False)
        macro = (predictions.assign(ae=(predictions["prediction"] - predictions["log_D"]).abs())
                 .groupby(["split_seed", "extractant"])["ae"].mean()
                 .groupby("split_seed").mean().mean())
        print(f"{'DIRECT_ALL':12s} extractant-macro MAE {macro:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
