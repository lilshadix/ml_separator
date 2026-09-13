#!/usr/bin/env python
"""Phases 6-8 and 12: leaderboard, bands, bootstrap, power, influence.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_analysis.py [--design B]

Reads every ``predictions/<design>/*.parquet`` written by the ladder and emits
the tables the decision report quotes.  Selection is reported separately from
the locked test evaluation: the ``inner validation`` column comes from each arm's
own per-fold selection log and is the only quantity a model choice may use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import inference, paths  # noqa: E402
from gen12eu.metrics import by_band, effective_sample_size, per_extractant, summarise  # noqa: E402

TARGET = "log_D"


def load(design: str) -> dict[str, pd.DataFrame]:
    directory = paths.PREDICTION_DIR / design
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*.parquet")):
        if path.stem in ("similarity",):
            continue
        frame = pd.read_parquet(path)
        if "prediction" not in frame.columns:
            continue
        frames[frame["arm"].iloc[0]] = frame
    if not frames:
        raise SystemExit(f"no predictions under {directory}")
    return frames


def assert_matched(frames: dict[str, pd.DataFrame]) -> None:
    """Invariant 7: every arm must be scored on byte-identical test rows."""
    import hashlib

    def row_key(frame: pd.DataFrame) -> str:
        # A mixed-dtype frame's ``to_numpy().tobytes()`` hashes object *pointers*,
        # not values, so the key is built from text.  This bit the check itself
        # first: it reported every arm as mismatched.
        ordered = frame.sort_values(["split_seed", "fold", "row_id"])
        joined = "|".join(f"{s}:{f}:{r}" for s, f, r in zip(
            ordered["split_seed"], ordered["fold"], ordered["row_id"]))
        return hashlib.blake2b(joined.encode(), digest_size=16).hexdigest()

    reference = None
    for arm, frame in frames.items():
        key = row_key(frame)
        if reference is None:
            reference = (arm, key)
        elif key != reference[1]:
            raise AssertionError(f"arm {arm!r} is scored on different rows from {reference[0]!r}")
    truths = {arm: frame.sort_values(["split_seed", "fold", "row_id"])[TARGET].to_numpy()
              for arm, frame in frames.items()}
    base = next(iter(truths.values()))
    for arm, values in truths.items():
        if not np.allclose(values, base):
            raise AssertionError(f"arm {arm!r} sees a different target vector")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", default="B")
    parser.add_argument("--reference", default="")
    args = parser.parse_args()

    frames = load(args.design)
    assert_matched(frames)
    out = paths.METRIC_DIR / args.design
    out.mkdir(parents=True, exist_ok=True)

    # ---- leaderboard and bands ------------------------------------------- #
    leaderboard = pd.DataFrame([summarise(f, label=arm) for arm, f in frames.items()])
    selection = []
    for arm in frames:
        path = paths.PREDICTION_DIR / args.design / f"{arm}__selection.csv"
        if path.exists():
            table = pd.read_csv(path)
            column = next((c for c in ("inner_macro_mae", "inner_val_mae") if c in table.columns), None)
            selection.append({"arm": arm,
                              "inner_validation": float(table[column].mean()) if column else np.nan,
                              "seconds_total": float(table["seconds"].sum())})
    if selection:
        leaderboard = leaderboard.merge(pd.DataFrame(selection), on="arm", how="left")
    leaderboard = leaderboard.sort_values("macro_mae_extractant", ignore_index=True)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)

    bands = pd.concat([by_band(f, label=arm) for arm, f in frames.items()], ignore_index=True)
    bands.to_csv(out / "bands.csv", index=False)

    units = {arm: per_extractant(f) for arm, f in frames.items()}
    pd.concat(units.values(), keys=units.keys(), names=["arm"]).reset_index(level=0).to_csv(
        out / "per_extractant.csv", index=False)

    chemotype_rows = []
    for arm, frame in frames.items():
        block = frame.assign(ae=(frame["prediction"] - frame[TARGET]).abs())
        table = (block.groupby(["chemotype", "split_seed"])["ae"].mean()
                 .groupby("chemotype").mean().rename("mae").reset_index())
        table["n_rows"] = block.groupby("chemotype").size().reindex(table["chemotype"]).to_numpy() / frame["split_seed"].nunique()
        table["n_extractants"] = block.groupby("chemotype")["extractant"].nunique().reindex(
            table["chemotype"]).to_numpy()
        table["arm"] = arm
        chemotype_rows.append(table)
    pd.concat(chemotype_rows, ignore_index=True).to_csv(out / "per_chemotype.csv", index=False)

    # ---- reference arm: the best Tier-1 on INNER VALIDATION --------------- #
    tier1 = [a for a in frames if a.startswith("T1_")]
    if args.reference:
        reference = args.reference
    elif tier1 and selection:
        inner = {r["arm"]: r["inner_validation"] for r in selection}
        reference = min(tier1, key=lambda a: inner.get(a, np.inf))
    else:
        reference = leaderboard["arm"].iloc[0]

    # ---- paired bootstrap, power, influence ------------------------------- #
    per_unit = inference.unit_table(units, statistic="mae")
    per_unit.to_csv(out / "bootstrap_input_mae.csv", index=False)
    candidates = [a for a in frames if a != reference]
    comparisons = {f"{reference}_vs_{a}": (reference, a) for a in candidates}
    tables = []
    for statistic in ("mae", "offset_abs", "shape_mae"):
        block = inference.unit_table(units, statistic=statistic).rename(
            columns={statistic: "mae"}) if statistic != "mae" else per_unit
        table = inference.paired_bootstrap(block, comparisons, statistic="mae")
        table["statistic"] = statistic
        tables.append(table)
    pairwise = pd.concat(tables, ignore_index=True)
    pairwise.to_csv(paths.BOOTSTRAP_DIR / f"{args.design}_pairwise.csv", index=False)

    power = pd.DataFrame([inference.minimum_detectable_effect(per_unit, reference, a)
                          for a in candidates])
    power.to_csv(paths.BOOTSTRAP_DIR / f"{args.design}_power.csv", index=False)

    # The influence diagnostic needs two *distinct* arms.  The reference is chosen
    # on validation and the leader on test; when they coincide, the contrast that
    # matters is the reference against the best arm that is not itself.
    best = leaderboard["arm"].iloc[0]
    challenger = next((a for a in leaderboard["arm"] if a != reference), None)
    influence = pd.concat([
        inference.leave_one_chemotype_out(per_unit, reference, arm).assign(candidate=arm)
        for arm in dict.fromkeys(a for a in (best, challenger) if a and a != reference)
    ], ignore_index=True) if challenger else pd.DataFrame()
    influence.to_csv(paths.BOOTSTRAP_DIR / f"{args.design}_influence.csv", index=False)

    # DOI-blocked robustness variant
    doi_of = (next(iter(frames.values())).dropna(subset=["doi"])
              .drop_duplicates("extractant").set_index("extractant")["doi"].astype(str))
    doi_unit = per_unit.assign(chemotype=per_unit["unit"].map(doi_of).fillna("no_doi"))
    doi_table = inference.paired_bootstrap(doi_unit, comparisons, statistic="mae")
    doi_table["blocking"] = "doi"
    doi_table.to_csv(paths.BOOTSTRAP_DIR / f"{args.design}_pairwise_doi_blocked.csv", index=False)

    # ---- band-restricted bootstrap for the far-band claim ----------------- #
    band_tables = []
    for band in ("far", "mid", "near"):
        band_units = {arm: table[table["band"] == band] for arm, table in units.items()}
        if min(len(t) for t in band_units.values()) < 5:
            continue
        shared = set.intersection(*[set(t["extractant"]) for t in band_units.values()])
        band_units = {a: t[t["extractant"].isin(shared)] for a, t in band_units.items()}
        block = inference.unit_table(band_units, statistic="mae")
        table = inference.paired_bootstrap(block, comparisons, statistic="mae")
        table["band"] = band
        table["n_blocks_in_band"] = block["chemotype"].nunique()
        band_tables.append(table)
    if band_tables:
        pd.concat(band_tables, ignore_index=True).to_csv(
            paths.BOOTSTRAP_DIR / f"{args.design}_pairwise_by_band.csv", index=False)

    summary = {
        "design": args.design, "reference_arm": reference, "best_arm_on_test": best,
        "reference_chosen_on": "inner validation macro MAE among Tier-1 arms",
        "n_arms": len(frames),
        "n_extractant_units": int(per_unit["unit"].nunique()),
        "n_bootstrap_blocks": int(per_unit["chemotype"].nunique()),
        "n_eff_units_by_chemotype": effective_sample_size(
            per_unit.drop_duplicates("unit").groupby("chemotype").size()),
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(leaderboard[["arm", "macro_mae_extractant", "macro_mae_chemotype", "pooled_mae",
                           "pooled_r2", "offset_mae", "shape_mae", "inner_validation"]]
              .to_string(index=False))
    print(f"\nreference (chosen on inner validation): {reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
