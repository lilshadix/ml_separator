#!/usr/bin/env python
"""Phase 12: does a better structural level leave less for one measurement to correct?

H6.  If a coordination-aware level model explains more of ``alpha_i``, the gap from
zero-shot to one-shot should shrink.  Every arm is evaluated with Gen12's own few-shot
machinery on **byte-identical support and query draws**, because the draw is a pure
function of ``(seed, repeat, extractant, n_rows)`` and never of the model.

H6 is not required for the generation to succeed and is reported either way.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_one_shot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import paths  # noqa: E402
from gen12eu import fewshot, inference  # noqa: E402

TARGET = "log_D"
ARMS = {
    "GEN12_ABL_D": paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_D_PLUS_LIG2D.parquet",
    "GEN12_ABL_C": paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_C_MOL_COND.parquet",
    "DEC_L3_S1": paths.PREDICTION_DIR / "full" / "DEC_L3_S1.parquet",
    "DEC_L7_S1": paths.PREDICTION_DIR / "full" / "DEC_L7_S1.parquet",
    "DIRECT_ALL": paths.PREDICTION_DIR / "full" / "DIRECT_ALL.parquet",
}


def main() -> int:
    frames = []
    for arm, path in ARMS.items():
        if not path.exists():
            print(f"skipping {arm}: {path} not present")
            continue
        frame = pd.read_parquet(path)
        frame["arm"] = arm
        frames.append(frame[["arm", "split_seed", "fold", "row_id", "extractant", "chemotype",
                             "band", TARGET, "prediction"]])
    oof = pd.concat(frames, ignore_index=True)

    # Invariant: every arm must carry identical rows, or the draws are not comparable.
    counts = oof.groupby("arm")["row_id"].count()
    if counts.nunique() != 1:
        raise SystemExit(f"arms hold different row counts: {counts.to_dict()}")

    reference = "GEN12_ABL_D"
    shrinkage = fewshot.shrinkage_from_training(oof[oof["arm"] == reference])
    adapters = [fewshot.OffsetAdapter(ratio=shrinkage["ratio"]),
                fewshot.PlainOffsetAdapter(), fewshot.NoModelAdapter()]
    detail = fewshot.evaluate(oof, adapters, repeats=fewshot.DEFAULT_REPEATS)
    common = fewshot.common_cohort(oof[oof["arm"] == reference])
    out = paths.METRIC_DIR / "one_shot"
    out.mkdir(parents=True, exist_ok=True)
    detail.drop(columns=["support_rows"]).to_parquet(out / "fewshot_detail.parquet", index=False)
    detail[["arm", "adapter", "k", "split_seed", "extractant", "repeat", "support_rows"]] \
        .to_parquet(out / "fewshot_support_membership.parquet", index=False)

    # A hard check that the draws really are shared, on the artefacts rather than in theory.
    membership = detail[detail["k"] > 0].groupby(
        ["k", "split_seed", "extractant", "repeat"])["support_rows"].nunique()
    if int((membership > 1).sum()):
        raise SystemExit("support draws differ between arms")

    curve = pd.concat([
        fewshot.learning_curve(detail).assign(cohort="per_k"),
        fewshot.learning_curve(detail, cohort=common).assign(cohort="common"),
    ], ignore_index=True)
    curve.to_csv(out / "fewshot_curve.csv", index=False)

    shrunk = curve[(curve["cohort"] == "common") & (curve["adapter"] == "A_OFFSET_SHRUNK")]
    gains = []
    for arm, block in shrunk.groupby("arm"):
        block = block.set_index("k")
        if 0 not in block.index or 1 not in block.index:
            continue
        gains.append({
            "arm": arm,
            "zero_shot_mae": float(block.loc[0, "mae"]),
            "one_shot_mae": float(block.loc[1, "mae"]),
            "one_shot_gain": float(block.loc[0, "mae"] - block.loc[1, "mae"]),
            "five_shot_mae": float(block.loc[5, "mae"]) if 5 in block.index else np.nan,
            "level_k0": float(block.loc[0, "offset_abs"]),
            "level_k1": float(block.loc[1, "offset_abs"]),
            "shape_k0": float(block.loc[0, "shape_mae"]),
            "shape_k1": float(block.loc[1, "shape_mae"]),
            "n_extractants": int(block.loc[0, "n_extractants"]),
        })
    gains = pd.DataFrame(gains).sort_values("one_shot_gain", ignore_index=True)
    gains.to_csv(out / "one_shot_gain.csv", index=False)

    # ---- paired bootstrap of the one-shot gain, blocked on chemotype -------- #
    block = detail[(detail["extractant"].isin(set(common)))
                   & (detail["adapter"] == "A_OFFSET_SHRUNK")]
    per_unit_frames = {}
    for (arm, k), piece in block.groupby(["arm", "k"]):
        if k not in (0, 1):
            continue
        table = (piece.groupby(["split_seed", "extractant"])
                 .agg(mae=("mae", "mean"), chemotype=("chemotype", "first"),
                      n_rows=("n_rows", "first")).reset_index())
        per_unit_frames[f"{arm}|k{k}"] = table
    shared = set.intersection(*[set(t["extractant"]) for t in per_unit_frames.values()])
    per_unit_frames = {k: v[v["extractant"].isin(shared)] for k, v in per_unit_frames.items()}
    units = inference.unit_table(per_unit_frames, statistic="mae")
    comparisons = {}
    for arm in ARMS:
        if f"{arm}|k0" in per_unit_frames:
            comparisons[f"{arm}_gain_k0_to_k1"] = (f"{arm}|k0", f"{arm}|k1")
    comparisons["zero_shot_DEC_L7_vs_DEC_L3"] = ("DEC_L3_S1|k0", "DEC_L7_S1|k0")
    comparisons["one_shot_DEC_L7_vs_DEC_L3"] = ("DEC_L3_S1|k1", "DEC_L7_S1|k1")
    comparisons["zero_shot_DEC_L7_vs_GEN12"] = ("GEN12_ABL_D|k0", "DEC_L7_S1|k0")
    comparisons["one_shot_DEC_L7_vs_GEN12"] = ("GEN12_ABL_D|k1", "DEC_L7_S1|k1")
    comparisons = {k: v for k, v in comparisons.items()
                   if v[0] in per_unit_frames and v[1] in per_unit_frames}
    table = inference.paired_bootstrap(units, comparisons, statistic="mae")
    table.to_csv(paths.BOOTSTRAP_DIR / "one_shot_pairwise.csv", index=False)

    (out / "meta.json").write_text(json.dumps({
        "reference": reference, "shrinkage": shrinkage,
        "common_cohort_extractants": len(common),
        "common_cohort_chemotypes": int(
            oof[oof["extractant"].isin(common)]["chemotype"].nunique()),
        "repeats": fewshot.DEFAULT_REPEATS,
        "draws_shared_across_arms": True,
    }, indent=1))

    print(gains.round(4).to_string(index=False))
    print("\npaired, chemotype-blocked:")
    print(table[["comparison", "point_delta", "bca_low", "bca_high", "bca_excludes_zero",
                 "units_total", "bootstrap_blocks"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
