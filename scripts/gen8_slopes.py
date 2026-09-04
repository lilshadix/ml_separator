"""Does the model get the *slope* right? (brief §19, §11)

A distribution coefficient is used to design a separation, and what a separation
designer reads off a titration is not the absolute value at one point but the
**slope** — how fast extraction rises with nitrate, how many extractant molecules
enter the complex, and whether the curve across the 4f series is steep enough to
separate two neighbouring lanthanides.  Those are derivatives, and a model can be
badly wrong about the level while being right about all of them.

This script recomputes every reconstructed curve twice — once from the measured
``log D`` and once from the frozen model's out-of-fold prediction on the *same*
rows — and compares slope, curvature and turning point directly.  It also asks the
level-free question that matters for §11: is the lanthanide response of a held-out
ligand reproduced as a smooth curve, or only as an unordered set of metal offsets?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen8.report import md_table  # noqa: E402
from lanthanide_separation.gen8.series import curve_statistics  # noqa: E402

OOF = REPO_ROOT / "runs" / "gen7_architecture" / "finalists" / "oof_predictions.parquet"
MEMBERSHIP = REPO_ROOT / "runs" / "gen8_architecture" / "series" / "curve_membership.parquet"
OUT = REPO_ROOT / "runs" / "gen8_architecture" / "functional"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=OOF)
    parser.add_argument("--membership", type=Path, default=MEMBERSHIP)
    parser.add_argument("--models", nargs="*",
                        default=["REC_ecfp_plus_recovered", "NULL_metal_cond",
                                 "TREE_MC_lig2d_ext_massaction"])
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    membership = pd.read_parquet(args.membership)
    oof = pd.read_parquet(args.oof)
    oof = oof[oof["model"].isin(args.models)]

    frames: list[pd.DataFrame] = []
    for (model, seed), block in oof.groupby(["model", "split_seed"], sort=True):
        sub = membership[membership["row_id"].isin(set(block["row_id"]))]
        keep = sub.groupby("curve_id")["row_id"].transform("size") >= args.min_points
        sub = sub[keep]
        if sub.empty:
            continue
        truth = curve_statistics(block, sub, value_column="log_D")
        prediction = curve_statistics(block, sub, value_column="prediction")
        merged = truth.merge(prediction, on="curve_id", suffixes=("_true", "_pred"))
        merged["model"] = model
        merged["split_seed"] = int(seed)
        ligand = block.drop_duplicates("row_id").set_index("row_id")["extractant"]
        merged["extractant"] = merged["curve_id"].map(
            sub.drop_duplicates("curve_id").set_index("curve_id")["row_id"].map(ligand))
        merged["tanimoto_cluster"] = merged["curve_id"].map(
            sub.drop_duplicates("curve_id").set_index("curve_id")["row_id"].map(
                block.drop_duplicates("row_id").set_index("row_id")["tanimoto_cluster"]))
        frames.append(merged)
    curves = pd.concat(frames, ignore_index=True)
    curves.to_parquet(args.out / "curve_slope_comparison.parquet", index=False)

    from scipy.stats import pearsonr, spearmanr
    rows: list[dict] = []
    for (model, axis), block in curves.groupby(["model", "axis_label_true"], sort=True):
        per_ligand = block.groupby("extractant")
        finite = block[np.isfinite(block["slope_true"]) & np.isfinite(block["slope_pred"])]
        if len(finite) < 5:
            continue
        record = {
            "model": model, "curve_type": axis, "n_curves": int(len(finite)),
            "n_ligands": int(finite["extractant"].nunique()),
            "slope_true_median": float(finite["slope_true"].median()),
            "slope_pred_median": float(finite["slope_pred"].median()),
            "slope_mae": float(np.abs(finite["slope_true"] - finite["slope_pred"]).mean()),
            "slope_pearson": float(pearsonr(finite["slope_true"], finite["slope_pred"]).statistic),
            "slope_spearman": float(spearmanr(finite["slope_true"], finite["slope_pred"]).statistic),
            "slope_sign_agree": float((np.sign(finite["slope_true"])
                                       == np.sign(finite["slope_pred"])).mean()),
            "y_span_true_median": float(finite["y_span_true"].median()),
            "y_span_pred_median": float(finite["y_span_pred"].median()),
            "span_ratio_median": float((finite["y_span_pred"]
                                        / finite["y_span_true"].replace(0, np.nan)).median()),
            "linear_r2_true_median": float(finite["linear_r2_true"].median()),
            "linear_r2_pred_median": float(finite["linear_r2_pred"].median()),
        }
        curved = finite[np.isfinite(finite["curvature_true"]) & np.isfinite(finite["curvature_pred"])]
        if len(curved) >= 5:
            record["curvature_pearson"] = float(
                pearsonr(curved["curvature_true"], curved["curvature_pred"]).statistic)
            record["curvature_sign_agree"] = float(
                (np.sign(curved["curvature_true"]) == np.sign(curved["curvature_pred"])).mean())
            record["n_curvature"] = int(len(curved))
        # A null a slope model must beat: predicting every curve's slope as the
        # *training* median slope of its own curve type.  If the model cannot beat
        # that, it has learned no ligand-specific slope at all.
        record["slope_mae_median_null"] = float(
            np.abs(finite["slope_true"] - finite["slope_true"].median()).mean())
        rows.append(record)
    board = pd.DataFrame(rows)
    board.to_csv(args.out / "slope_accuracy.csv", index=False)

    # --- lanthanide response: is the metal curve reproduced as a *curve*? ---- #
    metal = curves[curves["axis_label_true"] == "metal_series"]
    ln_rows = []
    for model, block in metal.groupby("model", sort=True):
        finite = block[np.isfinite(block["slope_true"]) & np.isfinite(block["slope_pred"])]
        if len(finite) < 5:
            continue
        ln_rows.append({
            "model": model, "n_curves": int(len(finite)),
            "slope_spearman": float(spearmanr(finite["slope_true"], finite["slope_pred"]).statistic),
            "slope_sign_agree": float((np.sign(finite["slope_true"])
                                       == np.sign(finite["slope_pred"])).mean()),
            "flat_prediction_share": float((finite["y_span_pred"] < 0.1).mean()),
            "flat_truth_share": float((finite["y_span_true"] < 0.1).mean()),
            "turning_point_true_share": float(finite["turning_point_true"].notna().mean()),
            "turning_point_pred_share": float(finite["turning_point_pred"].notna().mean()),
        })
    lanthanide = pd.DataFrame(ln_rows)
    lanthanide.to_csv(args.out / "lanthanide_response.csv", index=False)

    lines = ["# Slopes, curvature and the lanthanide response", "",
             "*Every reconstructed curve scored twice: once from the measured `log D`, once "
             "from the frozen model's out-of-fold prediction on the same rows. "
             "`slope_mae_median_null` is what a model scores if it predicts every curve's slope "
             "as the median slope of that curve type — the null a slope model has to beat.*", "",
             "## Slope accuracy by curve type", "",
             md_table(board.sort_values(["model", "curve_type"])), "",
             "## Is the lanthanide response reproduced as a curve?", "",
             md_table(lanthanide), ""]
    (args.out / "slopes.md").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 260)
    print(board.to_string(index=False))
    print()
    print(lanthanide.to_string(index=False))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
