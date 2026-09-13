#!/usr/bin/env python
"""Phase 9: what 1, 2, 3 and 5 Eu measurements of a new extractant buy.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_fewshot.py \
        [--design B] [--arms T1_EXTRATREES,B1_COND_ONLY]

Runs only on frozen zero-shot predictions — no model is refitted — so the k = 0
column of the learning curve is exactly the zero-shot number in the leaderboard,
restricted to the same cohort.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import fewshot, inference, paths  # noqa: E402

TARGET = "log_D"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default="B")
    parser.add_argument("--arms", default="T1_EXTRATREES,B1_COND_ONLY,B3_NN_LEVEL_ONLY")
    parser.add_argument("--repeats", type=int, default=fewshot.DEFAULT_REPEATS)
    args = parser.parse_args()

    directory = paths.PREDICTION_DIR / args.design
    frames = []
    for arm in args.arms.split(","):
        path = directory / f"{arm}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        frames.append(pd.read_parquet(path))
    oof = pd.concat(frames, ignore_index=True)

    # The shrinkage constant is estimated from the *reference* arm's residuals and
    # is a single number reused at every k, never refitted per extractant.
    reference = args.arms.split(",")[0]
    shrinkage = fewshot.shrinkage_from_training(oof[oof["arm"] == reference])
    global_mean = float(oof[oof["arm"] == reference][TARGET].mean())
    adapters = [
        fewshot.OffsetAdapter(ratio=shrinkage["ratio"]),
        fewshot.PlainOffsetAdapter(),
        fewshot.NoModelAdapter(),
    ]

    detail = fewshot.evaluate(oof, adapters, repeats=args.repeats)
    out = paths.METRIC_DIR / args.design
    out.mkdir(parents=True, exist_ok=True)
    detail.drop(columns=["support_rows"]).to_parquet(out / "fewshot_detail.parquet", index=False)
    detail[["arm", "adapter", "k", "split_seed", "extractant", "repeat", "support_rows"]].to_parquet(
        out / "fewshot_support_membership.parquet", index=False)

    common = fewshot.common_cohort(oof[oof["arm"] == reference])
    curves = pd.concat([
        fewshot.learning_curve(detail).assign(cohort="per_k"),
        fewshot.learning_curve(detail, cohort=common).assign(cohort="common"),
    ], ignore_index=True)
    curves.to_csv(out / "fewshot_curve.csv", index=False)

    # Paired bootstrap at each k, on the common cohort, blocked on chemotype.
    rows = []
    block = detail[detail["extractant"].isin(set(common))]
    for k in (1, 2, 3, 5):
        at_k = block[block["k"] == k]
        per_unit_frames = {}
        for (arm, adapter), piece in at_k.groupby(["arm", "adapter"]):
            table = (piece.groupby(["split_seed", "extractant"])
                     .agg(mae=("mae", "mean"), chemotype=("chemotype", "first"),
                          n_rows=("n_rows", "first")).reset_index())
            per_unit_frames[f"{arm}|{adapter}"] = table
        if f"{reference}|A_OFFSET_SHRUNK" not in per_unit_frames:
            continue
        shared = set.intersection(*[set(t["extractant"]) for t in per_unit_frames.values()])
        per_unit_frames = {k2: v[v["extractant"].isin(shared)] for k2, v in per_unit_frames.items()}
        units = inference.unit_table(per_unit_frames, statistic="mae")
        anchor = f"{reference}|A_OFFSET_SHRUNK"
        comparisons = {f"{anchor}_vs_{name}": (anchor, name)
                       for name in per_unit_frames if name != anchor}
        table = inference.paired_bootstrap(units, comparisons, statistic="mae")
        table["k"] = k
        rows.append(table)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            paths.BOOTSTRAP_DIR / f"{args.design}_fewshot_pairwise.csv", index=False)

    (out / "fewshot_meta.json").write_text(json.dumps({
        "design": args.design, "arms": args.arms.split(","), "reference": reference,
        "repeats": args.repeats, "shrinkage": shrinkage, "global_mean": global_mean,
        "common_cohort_extractants": len(common),
        "common_cohort_chemotypes": int(
            oof[oof["extractant"].isin(common)]["chemotype"].nunique()),
    }, indent=1))

    view = curves[(curves["cohort"] == "common") & (curves["arm"] == reference)]
    print(view.pivot_table(index="adapter", columns="k",
                           values="mae").round(4).to_string())
    print("\ncommon cohort:", len(common), "extractants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
