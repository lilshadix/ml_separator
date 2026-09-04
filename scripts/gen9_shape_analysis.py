#!/usr/bin/env python
"""GEN9-A analysis — did within-curve training actually change the response surface?

Macro MAE cannot answer that, and the brief (§8, §19) is explicit about why: a model
that nudges every unseen ligand's level a little closer while still drawing all its
titrations flat improves macro MAE and has fixed nothing.  So this script scores
eight shape endpoints per arm and requires them to move **coherently** before any
flattening claim is made.

Everything is computed on the same reconstructed curves gen8 used, from
out-of-fold predictions on held-out chemotypes, so a gen9 row and a gen8 row in the
same table are the same arithmetic on the same rows.

Three sections:

**By curve type** — slope, span and shape error on the extractant, acid and
lanthanide axes separately, because gen8 established that the physics differs
between them and a single pooled number hides it.

**Slope distributions** — because gen9 explicitly rewards steeper predictions, and
the failure mode it invites is replacing flattening with over-steepening.  The
over-steepening rate is defined against the corpus's own 95th percentile of
*measured* slopes, so it is a property of the data rather than a clipping constant
chosen after seeing results.

**Paired intervals** — chemotype-blocked paired bootstrap through gen8's own
``paired_chemotype_bootstrap``, one ligand one vote, resampling the Tanimoto
chemotype the folds actually held out.  Metrics where "bigger is better" are
converted to a distance from their ideal first, so the sign convention (positive =
candidate better) never flips inside a table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen9.metrics import (  # noqa: E402
    MIN_SLOPE_POINTS, MIN_TRUE_SPAN, curve_shape_table, dynamic_range, macro_mae,
    slope_distribution, summarise_shape,
)

MEMBERSHIP_PATH = REPO_ROOT / "runs" / "gen9_shape" / "curves" / "curve_membership.parquet"
FAILURE_CLASSES = (REPO_ROOT / "runs" / "gen8_architecture" / "case_studies"
                   / "ligand_failure_classes.csv")
PRIMARY_AXES = ("extractant", "acid", "metal_series")

#: Metrics where a *larger* value is better, and the ideal they are measured against.
#: Bootstrapping ``|metric - ideal|`` keeps one sign convention across every table.
DISTANCE_METRICS: dict[str, float] = {"span_recovery": 1.0, "slope_recovery": 1.0}
#: Metrics that are already losses.
LOSS_METRICS: tuple[str, ...] = ("slope_abs_error", "shape_mae", "row_mae")


def build_shape_tables(oof: pd.DataFrame, membership: pd.DataFrame,
                       *, min_points: int) -> pd.DataFrame:
    frames = []
    for (model, seed), block in oof.groupby(["model", "split_seed"], sort=True):
        table = curve_shape_table(block.reset_index(drop=True), membership,
                                  min_points=min_points)
        if table.empty:
            continue
        table["model"] = model
        table["split_seed"] = int(seed)
        frames.append(table)
    if not frames:
        raise SystemExit("no curves scored — check the membership table and min_points")
    return pd.concat(frames, ignore_index=True)


def per_ligand_shape(curves: pd.DataFrame, *, axes=PRIMARY_AXES) -> pd.DataFrame:
    """Collapse curves to ligands, one row per (model, ligand, axis, seed).

    The bootstrap's unit is the ligand and its block is the chemotype, so the
    per-curve table has to be collapsed first — otherwise a ligand carrying 40
    curves would count forty times and the interval would be a pseudo-replication.
    """
    work = curves[curves["axis_label"].isin(axes)].copy()
    work["span_recovery_distance"] = (work["span_recovery"] - 1.0).abs()
    work["slope_recovery_distance"] = (work["slope_recovery"] - 1.0).abs()
    guard = work["span_true"] >= MIN_TRUE_SPAN
    work.loc[~guard, "span_recovery_distance"] = np.nan
    columns = ["slope_abs_error", "shape_mae", "row_mae", "spearman", "sign_accuracy",
               "span_recovery", "span_recovery_distance", "slope_recovery_distance"]
    grouped = work.groupby(["model", "axis_label", "extractant", "tanimoto_cluster",
                            "split_seed"], dropna=False)[columns].mean().reset_index()
    grouped["n_curves"] = work.groupby(
        ["model", "axis_label", "extractant", "tanimoto_cluster", "split_seed"],
        dropna=False).size().to_numpy()
    return grouped


def bootstrap_shape(per_ligand: pd.DataFrame, comparisons: dict, *, axis: str,
                    metrics=("slope_abs_error", "shape_mae", "span_recovery_distance",
                             "row_mae"), replicates: int = 5000) -> pd.DataFrame:
    rows = []
    block = per_ligand[per_ligand["axis_label"] == axis]
    for metric in metrics:
        sub = block.dropna(subset=[metric])
        if sub["model"].nunique() < 2:
            continue
        try:
            table = paired_chemotype_bootstrap(
                sub.rename(columns={"model": "arm"}), comparisons,
                value_column=metric, replicates=replicates)
        except ValueError:
            continue
        table.insert(0, "axis", axis)
        table.insert(1, "metric", metric)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def failure_class_table(oof: pd.DataFrame, per_ligand: pd.DataFrame) -> pd.DataFrame:
    """gen8's locked ligand strata, used as a *stratification* and never as a target.

    The classes were read off gen8's own held-out errors, so they are descriptive.
    Two consequences are handled here rather than in prose: the labels are joined,
    never recomputed from gen9 predictions (which would be circular), and the
    ``A0_ROW_ONLY`` control travels into every stratum so that regression to the
    mean — which selects RESIDUAL_SHAPE on high gen8 error and would flatter *any*
    different model on it — is measurable rather than assumed away.
    """
    if not FAILURE_CLASSES.exists():
        return pd.DataFrame()
    classes = pd.read_csv(FAILURE_CLASSES)[["extractant", "failure_class"]]
    rows = []
    macro = (oof.assign(abs_error=(oof["prediction"] - oof["log_D"]).abs())
             .groupby(["model", "extractant", "ecfp_cluster"])["abs_error"].mean()
             .reset_index())
    macro = macro.merge(classes, on="extractant", how="left")
    for (model, failure_class), block in macro.groupby(["model", "failure_class"], sort=True):
        rows.append({"model": model, "failure_class": failure_class,
                     "n_ligands": int(block["extractant"].nunique()),
                     "macro_mae": float(block.groupby("ecfp_cluster")["abs_error"].mean().mean()),
                     "mean_mae": float(block["abs_error"].mean())})
    table = pd.DataFrame.from_records(rows)
    shape = per_ligand.merge(classes, on="extractant", how="left")
    shape = shape[shape["axis_label"] == "extractant"]
    if len(shape):
        extra = shape.groupby(["model", "failure_class"]).agg(
            span_recovery=("span_recovery", "median"),
            slope_abs_error=("slope_abs_error", "mean"),
            shape_mae=("shape_mae", "mean")).reset_index()
        table = table.merge(extra, on=["model", "failure_class"], how="left")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", default="GEN9_A0_ROW_ONLY",
                        help="the paired control every shape arm is quoted against")
    parser.add_argument("--anchor", default="REC_ecfp_plus_recovered",
                        help="the longitudinal anchor (gen8's frozen model)")
    parser.add_argument("--min-points", type=int, default=MIN_SLOPE_POINTS)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "gen9_shape" / "shape")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    membership = pd.read_parquet(MEMBERSHIP_PATH)
    oof = pd.concat([pd.read_parquet(p) for p in args.oof], ignore_index=True)
    oof = oof.drop_duplicates(["model", "split_seed", "row_id"])
    models = sorted(oof["model"].unique())
    print(f"{len(oof):,} OOF rows over {len(models)} models: {models}")

    # --- row-level anchor -------------------------------------------------
    macro = []
    for (model, seed), block in oof.groupby(["model", "split_seed"], sort=True):
        macro.append({"model": model, "split_seed": int(seed),
                      "macro_mae": macro_mae(block),
                      "pooled_mae": float((block["prediction"] - block["log_D"]).abs().mean())})
    macro_table = pd.DataFrame.from_records(macro)
    macro_summary = macro_table.groupby("model").agg(
        macro_mae=("macro_mae", "mean"), macro_sd=("macro_mae", "std"),
        pooled_mae=("pooled_mae", "mean"), n_seeds=("split_seed", "nunique")).reset_index()
    macro_summary.to_csv(args.out / "macro_mae.csv", index=False)
    print("\n--- row-level ---")
    print(macro_summary.sort_values("macro_mae").to_string(index=False))

    # --- per-curve shape --------------------------------------------------
    curves = build_shape_tables(oof, membership, min_points=args.min_points)
    curves.to_parquet(args.out / "curve_shape.parquet", index=False)
    summary = summarise_shape(curves, keys=("model", "axis_label"))
    summary.to_csv(args.out / "shape_by_axis.csv", index=False)
    print(f"\n--- shape by axis ({len(curves):,} scored curves) ---")
    show = ["model", "axis_label", "n_curves", "slope_true_median", "slope_pred_median",
            "slope_mae", "span_recovery_median", "shape_mae", "within_curve_spearman",
            "within_curve_sign_accuracy"]
    print(summary[summary["axis_label"].isin(PRIMARY_AXES)][show].to_string(index=False))

    distributions = slope_distribution(curves, keys=("model", "axis_label"))
    distributions.to_csv(args.out / "slope_distribution.csv", index=False)
    print("\n--- slope distribution (over-steepening check) ---")
    print(distributions[distributions["axis_label"].isin(PRIMARY_AXES)][
        ["model", "axis_label", "true_median", "pred_median", "pred_q95", "pred_absmax",
         "true_abs_q95", "frac_pred_steeper_than_true_q95"]].to_string(index=False))

    ranges = []
    for (model, seed), block in oof.groupby(["model", "split_seed"], sort=True):
        table = dynamic_range(block)
        table["model"], table["split_seed"] = model, int(seed)
        ranges.append(table)
    dynamic = pd.concat(ranges, ignore_index=True)
    dynamic.to_csv(args.out / "dynamic_range.csv", index=False)
    compression = dynamic.groupby("model")[["sd_ratio", "span_ratio"]].median().reset_index()
    print("\n--- per-ligand dynamic range (median ratio, 1.0 = faithful) ---")
    print(compression.to_string(index=False))

    # --- paired intervals -------------------------------------------------
    per_ligand = per_ligand_shape(curves)
    per_ligand.to_csv(args.out / "per_ligand_shape.csv", index=False)
    comparisons = {}
    for model in models:
        if model != args.reference:
            comparisons[f"{model} vs {args.reference}"] = (args.reference, model)
        if args.anchor in models and model not in (args.anchor,):
            comparisons[f"{model} vs {args.anchor}"] = (args.anchor, model)
    intervals = []
    for axis in PRIMARY_AXES:
        table = bootstrap_shape(per_ligand, comparisons, axis=axis, replicates=args.replicates)
        if len(table):
            intervals.append(table)
    if intervals:
        bootstrap = pd.concat(intervals, ignore_index=True)
        bootstrap.to_csv(args.out / "shape_bootstrap.csv", index=False)
        print("\n--- paired chemotype-blocked intervals (positive = candidate better) ---")
        core = bootstrap[bootstrap["metric"].isin(["slope_abs_error", "shape_mae"])]
        print(core[["axis", "metric", "comparison", "point", "ci95_low", "ci95_high",
                    "n_units", "units_improved", "seeds_positive"]].to_string(index=False))

    # --- failure-class strata ---------------------------------------------
    classes = failure_class_table(oof, per_ligand)
    if len(classes):
        classes.to_csv(args.out / "failure_class.csv", index=False)
        print("\n--- gen8 failure-class strata (locked labels, joined not recomputed) ---")
        print(classes.to_string(index=False))

    (args.out / "analysis.json").write_text(json.dumps({
        "models": models, "n_oof_rows": int(len(oof)), "n_curves": int(len(curves)),
        "min_points": int(args.min_points), "min_true_span": MIN_TRUE_SPAN,
        "reference": args.reference, "anchor": args.anchor,
        "replicates": int(args.replicates),
        "sources": [str(p) for p in args.oof],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
