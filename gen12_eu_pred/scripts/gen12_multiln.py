#!/usr/bin/env python
"""Phases 10-11: does other-lanthanide data improve Eu on strictly held-out chemistry?

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_multiln.py [--design B]

Runs the strict transfer arm, its three controls and the deliberately leaky arm
on byte-identical folds and Eu test rows, then the paired chemotype-blocked
bootstrap between them.  The Eu-only anchor is the frozen champion re-fitted with
the metal columns present, so the only difference between the anchor and the
strict arm is which rows entered training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import inference, paths, splits  # noqa: E402
from gen12eu.chemistry import band_of  # noqa: E402
from gen12eu.cohort import build_cohort  # noqa: E402
from gen12eu.metrics import by_band, per_extractant, summarise  # noqa: E402
from gen12eu.multiln import MultiLanthanideContender, build_auxiliary  # noqa: E402
from gen12eu.runner import run_contender  # noqa: E402


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

    auxiliary = build_auxiliary(frame.columns)
    inventory = {
        "auxiliary_rows": int(len(auxiliary)),
        "auxiliary_extractants": int(auxiliary["extractant"].nunique()),
        "auxiliary_metals": sorted(auxiliary["metal_symbol"].astype(str).unique()),
        "auxiliary_extractants_shared_with_eu": int(
            auxiliary["extractant"].isin(set(frame["extractant"])).groupby(
                auxiliary["extractant"]).first().sum()),
        "auxiliary_chemotypes": int(auxiliary["chemotype"].nunique()),
        "auxiliary_chemotypes_absent_from_eu": int(
            len(set(auxiliary["chemotype"]) - set(frame["chemotype"]))),
    }

    out = paths.PREDICTION_DIR / f"{args.design}_multiln"
    out.mkdir(parents=True, exist_ok=True)
    shared_mass: dict = {}
    # STRICT runs first so that MATCHED_EU_ONLY can be matched to its training mass.
    order = ("STRICT", "MATCHED_EU_ONLY", "PERMUTED_METAL", "SHUFFLED_TARGET", "LEAKY_NO_FILTER")
    frames: dict[str, pd.DataFrame] = {}
    audits: list[dict] = []
    for policy in order:
        contender = MultiLanthanideContender(auxiliary=auxiliary, policy=policy,
                                             family=args.family, reference_mass=shared_mass)
        print(f"[{args.design}] {contender.name}", flush=True)
        predictions, selection = run_contender(contender, frame, folds, cohort.blocks,
                                               similarity, verbose=False)
        predictions.to_parquet(out / f"{contender.name}.parquet", index=False)
        selection.to_csv(out / f"{contender.name}__selection.csv", index=False)
        frames[contender.name] = predictions
        audits.append({"policy": policy, **selection.mean(numeric_only=True).to_dict()})
        macro = summarise(predictions, label=contender.name)["macro_mae_extractant"]
        print(f"  -> extractant-macro MAE {macro:.4f}", flush=True)

    anchor_path = paths.PREDICTION_DIR / args.design / "T1_EXTRATREES.parquet"
    if anchor_path.exists():
        anchor = pd.read_parquet(anchor_path)
        frames["EU_ONLY_ANCHOR"] = anchor

    leaderboard = pd.DataFrame([summarise(f, label=a) for a, f in frames.items()]).sort_values(
        "macro_mae_extractant", ignore_index=True)
    metric_dir = paths.METRIC_DIR / f"{args.design}_multiln"
    metric_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(metric_dir / "leaderboard.csv", index=False)
    pd.concat([by_band(f, label=a) for a, f in frames.items()], ignore_index=True).to_csv(
        metric_dir / "bands.csv", index=False)

    units = {a: per_extractant(f) for a, f in frames.items()}
    per_unit = inference.unit_table(units, statistic="mae")
    reference = "T4_MATCHED_EU_ONLY"
    comparisons = {f"{reference}_vs_{a}": (reference, a) for a in frames if a != reference}
    tables = []
    for statistic in ("mae", "offset_abs", "shape_mae"):
        block = (per_unit if statistic == "mae"
                 else inference.unit_table(units, statistic=statistic).rename(
                     columns={statistic: "mae"}))
        table = inference.paired_bootstrap(block, comparisons, statistic="mae")
        table["statistic"] = statistic
        tables.append(table)
    pd.concat(tables, ignore_index=True).to_csv(
        paths.BOOTSTRAP_DIR / f"{args.design}_multiln_pairwise.csv", index=False)
    pd.DataFrame([inference.minimum_detectable_effect(per_unit, reference, a)
                  for a in frames if a != reference]).to_csv(
        paths.BOOTSTRAP_DIR / f"{args.design}_multiln_power.csv", index=False)

    (metric_dir / "inventory.json").write_text(json.dumps(
        {"inventory": inventory, "per_policy": audits,
         "reference": reference,
         "reference_rationale": "the design-matched control, not the frozen anchor: "
                                "gen11 showed that comparing an auxiliary arm against an "
                                "anchor it does not share a design with attributes the "
                                "design change to the auxiliary data"}, indent=1, default=str))
    print(leaderboard[["arm", "macro_mae_extractant", "macro_mae_chemotype", "offset_mae",
                       "shape_mae"]].round(4).to_string(index=False))
    print(json.dumps(inventory, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
