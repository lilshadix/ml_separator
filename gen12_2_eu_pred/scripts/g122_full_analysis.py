#!/usr/bin/env python
"""Phase 9 analysis: direct versus decomposed full log_D prediction, on identical queries.

Reads the Gen12.2 decomposed arms, the new direct arm, and Gen12's frozen predictions,
and produces the leaderboard, the level/shape decomposition, the band tables and the
paired chemotype-blocked bootstrap.  Level/shape centring is per ``(split_seed,
extractant)`` throughout, using Gen12's own ``per_extractant``.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_full_analysis.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import paths  # noqa: E402
from gen12eu import inference  # noqa: E402
from gen12eu.metrics import by_band, per_extractant, summarise  # noqa: E402

TARGET = "log_D"
GEN12_ARMS = {
    "GEN12_ABL_D": paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_D_PLUS_LIG2D.parquet",
    "GEN12_ABL_C": paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_C_MOL_COND.parquet",
    "GEN12_ABL_A": paths.GEN12_PREDICTIONS / "B_ablation" / "ABL_A_CONDITIONS.parquet",
}
CONTRASTS = {
    "SECONDARY_decomposed_L7_vs_L3": ("DEC_L3_S1", "DEC_L7_S1"),
    "direct_ALL_vs_GEN12": ("GEN12_ABL_D", "DIRECT_ALL"),
    "decomposed_L7_vs_direct_GEN12": ("GEN12_ABL_D", "DEC_L7_S1"),
    "decomposed_L7_vs_direct_ALL": ("DIRECT_ALL", "DEC_L7_S1"),
    "decomposed_L3_vs_direct_GEN12": ("GEN12_ABL_D", "DEC_L3_S1"),
    "shape_S3_vs_S1": ("DEC_L7_S1", "DEC_L7_S3"),
    "shape_S0_vs_S1": ("DEC_L7_S1", "DEC_L7_S0"),
    "shape_S2_vs_S1": ("DEC_L7_S1", "DEC_L7_S2"),
    "level_L4_vs_L3": ("DEC_L3_S1", "DEC_L4_S1"),
    "level_L6_vs_L3": ("DEC_L3_S1", "DEC_L6_S1"),
    "vs_no_chemistry_control": ("GEN12_ABL_A", "DEC_L7_S1"),
}


def main() -> int:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted((paths.PREDICTION_DIR / "full").glob("*.parquet")):
        frame = pd.read_parquet(path)
        frames[frame["arm"].iloc[0]] = frame
    for arm, path in GEN12_ARMS.items():
        frame = pd.read_parquet(path)
        frame["arm"] = arm
        frames[arm] = frame
    if not frames:
        raise SystemExit("no full-prediction arms found")

    # Invariant 9: identical query rows and identical truth across every arm.
    reference = None
    for arm, frame in frames.items():
        ordered = frame.sort_values(["split_seed", "fold", "row_id"])
        key = hashlib.blake2b("|".join(
            f"{s}:{f}:{r}" for s, f, r in zip(ordered["split_seed"], ordered["fold"],
                                              ordered["row_id"])).encode(),
            digest_size=16).hexdigest()
        if reference is None:
            reference = (arm, key, ordered[TARGET].to_numpy())
        else:
            if key != reference[1]:
                raise AssertionError(f"{arm!r} is scored on different rows from {reference[0]!r}")
            if not np.allclose(ordered[TARGET].to_numpy(), reference[2]):
                raise AssertionError(f"{arm!r} sees a different target vector")

    out = paths.METRIC_DIR / "full"
    out.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame([summarise(f, label=a) for a, f in frames.items()])
    inner = []
    for arm in frames:
        path = paths.PREDICTION_DIR / "full" / f"{arm}__selection.csv"
        if path.exists():
            table = pd.read_csv(path)
            column = next((c for c in ("inner_macro_mae", "inner_val_mae")
                           if c in table.columns), None)
            inner.append({"arm": arm,
                          "inner_validation": float(table[column].mean()) if column else np.nan})
    if inner:
        leaderboard = leaderboard.merge(pd.DataFrame(inner), on="arm", how="left")
    leaderboard = leaderboard.sort_values("macro_mae_extractant", ignore_index=True)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)
    pd.concat([by_band(f, label=a) for a, f in frames.items()],
              ignore_index=True).to_csv(out / "bands.csv", index=False)

    units = {arm: per_extractant(frame) for arm, frame in frames.items()}
    comparisons = {k: v for k, v in CONTRASTS.items() if v[0] in units and v[1] in units}
    tables = []
    for statistic in ("mae", "offset_abs", "shape_mae"):
        block = inference.unit_table(units, statistic=statistic)
        if statistic != "mae":
            block = block.rename(columns={statistic: "mae"})
        table = inference.paired_bootstrap(block, comparisons, statistic="mae")
        table["statistic"] = statistic
        per_seed = {a: t.groupby("split_seed")[statistic].mean() for a, t in units.items()}
        table["seeds_favouring_candidate"] = [
            int(((per_seed[r] - per_seed[c]) > 0).sum())
            for r, c in zip(table["reference"], table["candidate"])]
        tables.append(table)
    pairwise = pd.concat(tables, ignore_index=True)
    pairwise.to_csv(paths.BOOTSTRAP_DIR / "full_pairwise.csv", index=False)

    power = [dict(inference.minimum_detectable_effect(
        inference.unit_table(units, statistic="mae"), r, c), contrast=k)
        for k, (r, c) in comparisons.items()]
    pd.DataFrame(power).to_csv(paths.BOOTSTRAP_DIR / "full_power.csv", index=False)

    band_tables = []
    for band in ("far", "mid", "near"):
        band_units = {a: t[t["band"] == band] for a, t in units.items()}
        shared = set.intersection(*[set(t["extractant"]) for t in band_units.values()])
        band_units = {a: t[t["extractant"].isin(shared)] for a, t in band_units.items()}
        if min(len(t) for t in band_units.values()) < 5:
            continue
        block = inference.unit_table(band_units, statistic="mae")
        table = inference.paired_bootstrap(block, comparisons, statistic="mae")
        table["band"] = band
        table["n_blocks"] = block["chemotype"].nunique()
        for arm, piece in band_units.items():
            table[f"macro_{arm}"] = piece.groupby("split_seed")["mae"].mean().mean()
        band_tables.append(table)
    pd.concat(band_tables, ignore_index=True).to_csv(
        paths.BOOTSTRAP_DIR / "full_by_band.csv", index=False)

    influence = []
    for name in ("SECONDARY_decomposed_L7_vs_L3", "decomposed_L7_vs_direct_GEN12"):
        if name not in comparisons:
            continue
        reference, candidate = comparisons[name]
        table = inference.leave_one_chemotype_out(
            inference.unit_table(units, statistic="mae"), reference, candidate)
        table["contrast"] = name
        influence.append(table)
    if influence:
        pd.concat(influence, ignore_index=True).to_csv(
            paths.BOOTSTRAP_DIR / "full_influence.csv", index=False)

    show = ["arm", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae", "pooled_r2",
            "offset_mae", "shape_mae", "inner_validation"]
    print(leaderboard[[c for c in show if c in leaderboard.columns]].round(4).to_string(index=False))
    print("\n=== pre-registered contrasts, extractant-macro MAE ===")
    view = pairwise[pairwise["statistic"] == "mae"]
    print(view[["comparison", "point_delta", "bca_low", "bca_high", "bca_excludes_zero",
                "seeds_favouring_candidate", "units_improved", "units_total"]]
          .round(4).to_string(index=False))
    print("\n=== the same contrasts on the level and shape components ===")
    view = pairwise[pairwise["statistic"] != "mae"]
    print(view[["comparison", "statistic", "point_delta", "bca_low", "bca_high",
                "bca_excludes_zero"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
