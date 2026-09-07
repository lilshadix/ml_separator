#!/usr/bin/env python
"""Phases 7, 10, 11, 13, 14 for the level task: leaderboard, bands, bootstrap, power, subgroup.

Selection and the locked test are kept apart throughout: the ``inner_validation`` column
comes from each arm's own per-fold selection log and is the only quantity a model choice
may use, and the table names the *selected* arm and the *best observed test* arm
separately and never merges them.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python \
        gen12_2_eu_pred/scripts/g122_level_analysis.py [--definition LVL_MEAN] [--family extratrees]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import levelmetrics, levelmodels, paths  # noqa: E402
from gen12eu import inference  # noqa: E402
from gen12eu.metrics import effective_sample_size  # noqa: E402

#: The pre-registered feature-family ablation contrasts.  Letters are §19 of the brief.
CONTRASTS: dict[str, tuple[str, str]] = {
    "PRIMARY_G_vs_D": ("L3_ECFP_GENERIC", "L7_ALL"),
    "E_vs_A": ("L1_ECFP", "L5_ECFP_COORD"),
    "F_vs_B": ("L2_GENERIC", "L6_GENERIC_COORD"),
    "C_vs_A": ("L1_ECFP", "L4_COORD"),
    "C_vs_D": ("L3_ECFP_GENERIC", "L4_COORD"),
    "D_vs_A": ("L1_ECFP", "L3_ECFP_GENERIC"),
    "G_vs_NN": ("L0_NN_TANIMOTO", "L7_ALL"),
    "D_vs_NN": ("L0_NN_TANIMOTO", "L3_ECFP_GENERIC"),
    "C_vs_NN": ("L0_NN_TANIMOTO", "L4_COORD"),
    "G_vs_GLOBAL": ("L0_GLOBAL_MEAN", "L7_ALL"),
}


def load(directory: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(directory.glob("*.parquet")):
        frame = pd.read_parquet(path)
        frames[frame["arm"].iloc[0]] = frame
    if not frames:
        raise SystemExit(f"no level predictions under {directory}")
    return frames


def assert_matched(frames: dict[str, pd.DataFrame]) -> None:
    """Invariant 9: every arm is scored on identical evaluation units and identical truth."""
    import hashlib
    reference = None
    for arm, frame in frames.items():
        ordered = frame.sort_values(["split_seed", "fold", "extractant"])
        key = hashlib.blake2b("|".join(
            f"{s}:{f}:{e}" for s, f, e in zip(ordered["split_seed"], ordered["fold"],
                                              ordered["extractant"])).encode(),
            digest_size=16).hexdigest()
        truth = ordered["alpha_true"].to_numpy()
        if reference is None:
            reference = (arm, key, truth)
        else:
            if key != reference[1]:
                raise AssertionError(f"{arm!r} is scored on different units from {reference[0]!r}")
            if not np.allclose(truth, reference[2]):
                raise AssertionError(f"{arm!r} sees a different level target")


def cohorts(frames: dict[str, pd.DataFrame], threshold: int) -> dict[str, set]:
    cells = pd.read_parquet(paths.GEN12_COHORT_PARQUET)["extractant"].value_counts()
    everything = set(next(iter(frames.values()))["extractant"])
    return {"full": everything,
            "level_reliable": {e for e in everything if cells.get(e, 0) >= threshold}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition", default="LVL_MEAN")
    parser.add_argument("--family", default="extratrees")
    args = parser.parse_args()

    directory = paths.PREDICTION_DIR / "level" / args.definition
    if args.family != "extratrees":
        directory = directory / args.family
    frames = load(directory)
    assert_matched(frames)
    choice = json.loads((paths.MANIFEST_DIR / "level_definition_choice.json").read_text())
    threshold = int(choice["level_reliable_cohort"]["threshold_cells"])
    membership = cohorts(frames, threshold)
    subgroup = pd.read_csv(paths.MANIFEST_DIR / "multi_arm_subgroup.csv", index_col=0)

    tag = args.definition if args.family == "extratrees" else f"{args.definition}_{args.family}"
    out = paths.METRIC_DIR / "level" / tag
    out.mkdir(parents=True, exist_ok=True)

    # ---- leaderboard on both cohorts, with inner validation beside it -------- #
    rows, band_rows = [], []
    for cohort_name, keep in membership.items():
        for arm, frame in frames.items():
            block = frame[frame["extractant"].isin(keep)]
            record = levelmetrics.summarise(block, label=arm, cohort=cohort_name)
            record["ablation_letter"] = levelmodels.ABLATION_LETTER.get(arm, "")
            selection_path = directory / f"{arm}__selection.csv"
            if selection_path.exists():
                selection = pd.read_csv(selection_path)
                record["inner_validation"] = float(selection["inner_level_mae"].mean()) \
                    if "inner_level_mae" in selection else np.nan
                record["n_clipped_total"] = int(selection["n_clipped"].sum())
            record.update({f"variance_{k}": v for k, v in
                           levelmetrics.variance_captured(block).items()})
            rows.append(record)
            band_rows.append(levelmetrics.by_band(block, label=arm, cohort=cohort_name))
    leaderboard = pd.DataFrame(rows).sort_values(["cohort", "level_mae"], ignore_index=True)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)
    pd.concat(band_rows, ignore_index=True).to_csv(out / "bands.csv", index=False)

    # ---- selection versus best observed test -------------------------------- #
    tree_arms = [a for a in frames if a.startswith(("L1", "L2", "L3", "L4", "L5", "L6", "L7"))]
    full = leaderboard[leaderboard["cohort"] == "full"].set_index("arm")
    selected = min(tree_arms, key=lambda a: full.loc[a, "inner_validation"])
    best_test = full.loc[tree_arms, "level_mae"].idxmin()

    # ---- paired bootstrap on both cohorts ----------------------------------- #
    pairwise, power, influence = [], [], []
    for cohort_name, keep in membership.items():
        units = {arm: levelmetrics.per_extractant(frame[frame["extractant"].isin(keep)])
                 for arm, frame in frames.items()}
        for statistic in ("mae", "bias"):
            block = inference.unit_table(units, statistic=statistic)
            if statistic != "mae":
                block = block.rename(columns={statistic: "mae"})
            comparisons = {k: v for k, v in CONTRASTS.items()
                           if v[0] in frames and v[1] in frames}
            table = inference.paired_bootstrap(block, comparisons, statistic="mae")
            table["statistic"] = statistic
            table["cohort"] = cohort_name
            table["contrast"] = table["comparison"]
            per_seed = {a: t.groupby("split_seed")["mae" if statistic == "mae" else "bias"]
                        .mean() for a, t in units.items()}
            # ``paired_bootstrap`` fixes delta = reference - candidate and labels a positive
            # delta "candidate is better".  That is right for an error magnitude and
            # BACKWARDS for a signed error on a negatively-biased population: a positive
            # delta there means the candidate is *more* negative, i.e. under-predicts more.
            # The frozen Gen12 function is not modified; the orientation is corrected here.
            better_is_negative = statistic == "bias"
            table["improvement_direction"] = (
                "candidate less negative (delta < 0)" if better_is_negative
                else "candidate lower error (delta > 0)")
            table["seeds_favouring_candidate"] = [
                int((((per_seed[r] - per_seed[c]) < 0) if better_is_negative
                     else ((per_seed[r] - per_seed[c]) > 0)).sum())
                for r, c in zip(table["reference"], table["candidate"])]
            if better_is_negative:
                table["units_improved"] = table["units_total"] - table["units_improved"]
            for column, arm_key in (("reference_value", "reference"),
                                    ("candidate_value", "candidate")):
                table[column] = [
                    float(units[a].groupby("split_seed")[
                        "mae" if statistic == "mae" else "bias"].mean().mean())
                    for a in table[arm_key]]
            pairwise.append(table)
            if statistic == "mae":
                for name, (reference, candidate) in comparisons.items():
                    record = inference.minimum_detectable_effect(block, reference, candidate)
                    record.update({"contrast": name, "cohort": cohort_name})
                    power.append(record)
                for name in ("PRIMARY_G_vs_D", "E_vs_A"):
                    reference, candidate = CONTRASTS[name]
                    influence_table = inference.leave_one_chemotype_out(
                        block, reference, candidate)
                    influence_table["contrast"] = name
                    influence_table["cohort"] = cohort_name
                    influence.append(influence_table)
    pairwise = pd.concat(pairwise, ignore_index=True)
    pairwise.to_csv(paths.BOOTSTRAP_DIR / f"level_{tag}_pairwise.csv", index=False)
    pd.DataFrame(power).to_csv(paths.BOOTSTRAP_DIR / f"level_{tag}_power.csv", index=False)
    pd.concat(influence, ignore_index=True).to_csv(
        paths.BOOTSTRAP_DIR / f"level_{tag}_influence.csv", index=False)

    # ---- bands, for the H5 frontier claim ----------------------------------- #
    band_tables = []
    for band in ("far", "mid", "near"):
        units = {arm: levelmetrics.per_extractant(frame[frame["band"] == band])
                 for arm, frame in frames.items()}
        shared = set.intersection(*[set(t["extractant"]) for t in units.values()])
        units = {a: t[t["extractant"].isin(shared)] for a, t in units.items()}
        if min(len(t) for t in units.values()) < 5:
            continue
        block = inference.unit_table(units, statistic="mae")
        table = inference.paired_bootstrap(block, {k: v for k, v in CONTRASTS.items()
                                                   if v[0] in units and v[1] in units},
                                           statistic="mae")
        table["band"] = band
        table["n_blocks"] = block["chemotype"].nunique()
        # The arm value printed beside a delta must be the SAME quantity the delta is
        # computed from: the per-extractant mean over the seeds in which that extractant
        # fell in this band, then the plain mean over extractants.  A seed-macro would be a
        # different weighting and would not reproduce the delta by subtraction.
        wide = block.pivot_table(index="unit", columns="arm", values="mae", aggfunc="first")
        for arm in units:
            table[f"macro_{arm}"] = float(wide[arm].mean())
        table["delta_reproduces_from_macros"] = [
            bool(abs((float(wide[r].mean()) - float(wide[c].mean())) - d) < 1e-9)
            for r, c, d in zip(table["reference"], table["candidate"], table["point_delta"])]
        band_tables.append(table)
    pd.concat(band_tables, ignore_index=True).to_csv(
        paths.BOOTSTRAP_DIR / f"level_{tag}_by_band.csv", index=False)

    # The bands are NOT a partition of the extractants.  ``max_train_tanimoto`` is a
    # property of a fold's training set, so an extractant can be far under one split seed
    # and near under another.  The band contrasts therefore share units, and the per-band
    # extractant counts sum to more than the cohort.  Measured and published rather than
    # left for a reader to discover.
    any_frame = next(iter(frames.values()))
    per_band = any_frame.groupby("band")["extractant"].nunique()
    multi = any_frame.groupby("extractant")["band"].nunique()
    modal = any_frame.groupby("extractant")["band"].agg(lambda s: s.value_counts().index[0])
    overlap = {
        "n_extractants": int(any_frame["extractant"].nunique()),
        "unique_extractants_per_band": per_band.to_dict(),
        "sum_over_bands": int(per_band.sum()),
        "extractants_in_more_than_one_band": int((multi > 1).sum()),
        "modal_band_partition": modal.value_counts().to_dict(),
        "note": ("bands are assigned per (split_seed, fold, extractant); they are not a "
                 "partition and the band contrasts share units"),
    }
    (out / "band_overlap.json").write_text(json.dumps(overlap, indent=1, default=int))

    # ---- the multi-arm subgroup, signed error ------------------------------- #
    multi_arm = set(subgroup.index[subgroup["MULTI_ARM"]])
    subgroup_rows = []
    for name, keep in (("MULTI_ARM", multi_arm),
                       ("single_arm", set(subgroup.index) - multi_arm)):
        units = {arm: levelmetrics.per_extractant(frame[frame["extractant"].isin(keep)])
                 for arm, frame in frames.items()}
        block = inference.unit_table(units, statistic="bias").rename(columns={"bias": "mae"})
        table = inference.paired_bootstrap(
            block, {k: v for k, v in CONTRASTS.items() if v[0] in units and v[1] in units},
            statistic="mae")
        table["subgroup"] = name
        table["statistic"] = "signed_level_error"
        # signed error: a NEGATIVE delta means the candidate under-predicts less
        table["improvement_direction"] = "candidate less negative (delta < 0)"
        table["units_improved"] = table["units_total"] - table["units_improved"]
        for arm, piece in units.items():
            table[f"signed_{arm}"] = piece.groupby("split_seed")["bias"].mean().mean()
            table[f"absolute_{arm}"] = piece.groupby("split_seed")["mae"].mean().mean()
        table["n_extractants"] = block["unit"].nunique()
        table["n_chemotypes"] = block["chemotype"].nunique()
        table["n_eff_chemotype"] = effective_sample_size(
            block.drop_duplicates("unit").groupby("chemotype").size())
        subgroup_rows.append(table)
        for name2, (reference, candidate) in CONTRASTS.items():
            if reference in units and candidate in units:
                record = inference.minimum_detectable_effect(
                    inference.unit_table(units, statistic="bias").rename(
                        columns={"bias": "mae"}), reference, candidate)
                record.update({"contrast": name2, "cohort": f"subgroup_{name}",
                               "statistic": "signed_level_error"})
                power.append(record)
    pd.concat(subgroup_rows, ignore_index=True).to_csv(
        paths.BOOTSTRAP_DIR / f"level_{tag}_subgroup.csv", index=False)
    pd.DataFrame(power).to_csv(paths.BOOTSTRAP_DIR / f"level_{tag}_power.csv", index=False)

    summary = {"definition": args.definition, "family": args.family,
               "level_reliable_threshold_cells": threshold,
               "selected_arm_on_inner_validation": selected,
               "best_observed_test_arm": best_test,
               "selection_and_best_agree": bool(selected == best_test),
               "n_units": {k: len(v) for k, v in membership.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    show = ["arm", "ablation_letter", "level_mae", "level_mae_chemotype_macro",
            "level_signed_bias", "between_extractant_r2", "spearman", "dispersion_ratio",
            "inner_validation", "n_extractants"]
    for cohort_name in membership:
        print(f"\n=== level leaderboard — {cohort_name} cohort, {args.definition}, "
              f"{args.family} ===")
        print(leaderboard[leaderboard["cohort"] == cohort_name][show].round(4)
              .to_string(index=False))
    print(f"\nselected on inner validation: {selected}   best observed on test: {best_test}")
    print("\n=== pre-registered contrasts, full cohort, level MAE ===")
    view = pairwise[(pairwise["cohort"] == "full") & (pairwise["statistic"] == "mae")]
    print(view[["contrast", "reference_value", "candidate_value", "point_delta",
                "ci95_low", "ci95_high", "bca_low", "bca_high", "bca_excludes_zero",
                "p_two_sided", "seeds_favouring_candidate", "units_improved",
                "units_total"]].round(4).to_string(index=False))
    disagree = view[(view["bca_excludes_zero"])
                    & ((view["ci95_low"] <= 0) & (view["ci95_high"] >= 0))]
    if len(disagree):
        print("\nBCa and percentile DISAGREE on these contrasts — the BCa endpoint shift is "
              "doing the work, and the two-sided bootstrap p is printed above:")
        print(disagree[["contrast", "point_delta", "ci95_low", "ci95_high", "bca_low",
                        "bca_high", "p_two_sided"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
