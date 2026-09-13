#!/usr/bin/env python
"""The contrast the whole programme keeps arriving at, tested here for Eu alone.

Does **one measurement of a new extractant, used with no model at all**, beat the
best **zero-shot model**?  Both are scored on the identical query rows: the
zero-shot arm is evaluated only on the rows the k = 1 protocol leaves as queries,
so the comparison is not a model scored on more rows than its rival.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_one_measurement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import fewshot, inference, paths  # noqa: E402

TARGET = "log_D"
ARM = "T1_EXTRATREES"


def main() -> int:
    oof = pd.read_parquet(paths.PREDICTION_DIR / "B" / f"{ARM}.parquet")
    common = fewshot.common_cohort(oof)
    records: list[dict] = []
    for (seed, extractant), block in oof.groupby(["split_seed", "extractant"], sort=True):
        block = block.sort_values("row_id")
        truth = block[TARGET].to_numpy(dtype=float)
        prediction = block["prediction"].to_numpy(dtype=float)
        n = len(block)
        if n < 3:
            continue
        for repeat in range(fewshot.DEFAULT_REPEATS):
            support = fewshot.support_draw(str(extractant), n, 1, repeat, seed)
            query = np.setdiff1d(np.arange(n), support)
            if query.size < 2:
                continue
            records.append({
                "split_seed": int(seed), "extractant": str(extractant), "repeat": repeat,
                "chemotype": str(block["chemotype"].iloc[0]),
                "band": str(block["band"].iloc[0]), "n_rows": n,
                # scored on exactly the same query rows
                "ZERO_SHOT_MODEL": float(np.abs(prediction[query] - truth[query]).mean()),
                "ONE_MEASUREMENT_NO_MODEL": float(
                    np.abs(np.full(query.size, truth[support][0]) - truth[query]).mean()),
                "MODEL_PLUS_ONE_MEASUREMENT": float(np.abs(
                    prediction[query] + (truth[support] - prediction[support]).mean()
                    - truth[query]).mean()),
            })
    detail = pd.DataFrame(records)
    detail.to_parquet(paths.METRIC_DIR / "B" / "one_measurement_detail.parquet", index=False)

    arms = ["ZERO_SHOT_MODEL", "ONE_MEASUREMENT_NO_MODEL", "MODEL_PLUS_ONE_MEASUREMENT"]
    frames = {}
    for arm in arms:
        frames[arm] = (detail.groupby(["split_seed", "extractant"])
                       .agg(mae=(arm, "mean"), chemotype=("chemotype", "first"),
                            n_rows=("n_rows", "first"), band=("band", "first")).reset_index())
    summary = []
    for cohort_name, keep in (("all_scorable", set(detail["extractant"])),
                              ("common_cohort", set(common))):
        subset = {a: f[f["extractant"].isin(keep)] for a, f in frames.items()}
        units = inference.unit_table(subset, statistic="mae")
        table = inference.paired_bootstrap(
            units, {"zero_shot_vs_one_measurement": ("ZERO_SHOT_MODEL", "ONE_MEASUREMENT_NO_MODEL"),
                    "zero_shot_vs_model_plus_one": ("ZERO_SHOT_MODEL", "MODEL_PLUS_ONE_MEASUREMENT"),
                    "one_measurement_vs_model_plus_one": ("ONE_MEASUREMENT_NO_MODEL",
                                                          "MODEL_PLUS_ONE_MEASUREMENT")},
            statistic="mae")
        table["cohort"] = cohort_name
        table["n_extractants"] = units["unit"].nunique()
        for arm, frame in subset.items():
            table[f"macro_{arm}"] = frame.groupby("split_seed")["mae"].mean().mean()
        summary.append(table)
    out = pd.concat(summary, ignore_index=True)
    out.to_csv(paths.BOOTSTRAP_DIR / "B_one_measurement.csv", index=False)

    by_band = []
    for band in ("far", "mid", "near"):
        subset = {a: f[(f["band"] == band) & f["extractant"].isin(set(common))]
                  for a, f in frames.items()}
        if min(len(f) for f in subset.values()) < 4:
            continue
        row = {"band": band, "n_extractants": subset[arms[0]]["extractant"].nunique()}
        for arm, frame in subset.items():
            row[arm] = frame.groupby("split_seed")["mae"].mean().mean()
        by_band.append(row)
    pd.DataFrame(by_band).to_csv(paths.METRIC_DIR / "B" / "one_measurement_by_band.csv", index=False)

    view = out[["cohort", "comparison", "point_delta", "bca_low", "bca_high",
                "bca_excludes_zero", "n_extractants", "macro_ZERO_SHOT_MODEL",
                "macro_ONE_MEASUREMENT_NO_MODEL", "macro_MODEL_PLUS_ONE_MEASUREMENT"]]
    print(view.round(4).to_string(index=False))
    print("\nby band (common cohort):")
    print(pd.DataFrame(by_band).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
