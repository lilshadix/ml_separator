"""The twenty worst held-out ligands, and why each one failed (brief §18).

A leaderboard says how much error there is; this says what kind.  For every serious
model it takes the worst ligands by held-out MAE and attaches everything that could
explain them — nearest training neighbour and its Tanimoto, donor pattern, scaffold
family, publication and batch structure, the conditions actually measured, whether a
3D structure exists, whether the row carries an unmodelled second species, and the
split of the error into level and shape.

It then assigns each failure to a **mechanism**, using rules that are stated rather
than eyeballed:

``TRUE_EXTRAPOLATION``   nearest training Tanimoto < 0.4 and the error is almost all
                         offset.  Nothing in training resembles this molecule.
``UNMODELLED_SPECIES``   the ligand's rows carry a name that is not the modal name of
                         its structure — a second, unrecorded chemical was present.
``SINGLE_BATCH``         every row of the ligand comes from one (DOI, table) batch, so
                         its level and that batch's calibration are inseparable.
``MECHANISM_SWITCH``     the shape error is large while the offset is small: the model
                         has the level and gets the *response* wrong, which is what a
                         different extraction mechanism looks like.
``SPARSE``               fewer than five rows: the ligand's own mean is barely defined.
``LEVEL_ONLY``           large offset, small shape, close neighbours available — the
                         representation had the information and did not use it.
``UNCLASSIFIED``         none of the above; listed explicitly rather than folded into
                         a bucket it does not belong in.

The point of the rules is that the next architecture change should answer to an
observed failure mode, and a mode nobody can name is not observed.
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

from lanthanide_separation.gen6.metrics import decompose_level_shape  # noqa: E402
from lanthanide_separation.gen7.harness import load_cohort  # noqa: E402


def classify(row: pd.Series) -> str:
    offset, shape = row.get("offset_abs", np.nan), row.get("shape_mae", np.nan)
    dominated_by_offset = np.isfinite(shape) and offset > 2.0 * shape
    dominated_by_shape = np.isfinite(shape) and shape > offset
    if row.get("n_rows", 0) < 5:
        return "SPARSE"
    if row.get("frac_name_mismatch", 0.0) > 0.25:
        return "UNMODELLED_SPECIES"
    if row.get("nn_train_tanimoto", 1.0) < 0.4 and dominated_by_offset:
        return "TRUE_EXTRAPOLATION"
    if row.get("n_batches", 2) <= 1 and dominated_by_offset:
        return "SINGLE_BATCH"
    if dominated_by_shape:
        return "MECHANISM_SWITCH"
    if dominated_by_offset:
        return "LEVEL_ONLY"
    return "UNCLASSIFIED"


def build(oof: pd.DataFrame, cohort, *, model: str, top: int) -> pd.DataFrame:
    block = oof[oof["model"] == model]
    if block.empty:
        raise SystemExit(f"no rows for model {model!r}")
    frame = cohort.frame
    extras = ["nuisance__batch", "nuisance__doi", "rec__name_mismatch",
              "rec__aqueous_complexant", "donor__n_total", "DENTATE", "coreCN"]
    extras = [c for c in extras if c in frame.columns]
    context = frame[["row_id", *extras]]
    block = block.merge(context, on="row_id", how="left", validate="many_to_one")

    rows: list[dict] = []
    for seed, seed_block in block.groupby("split_seed"):
        decomposition = decompose_level_shape(
            seed_block["log_D"], seed_block["prediction"], seed_block["extractant"])
        per_ligand = decomposition.per_ligand.set_index("extractant")
        for ligand, sub in seed_block.groupby("extractant"):
            stats = per_ligand.loc[ligand]
            record = {
                "model": model, "split_seed": int(seed), "extractant": ligand,
                "n_rows": int(stats["n_rows"]), "mae": float(stats["mae"]),
                "offset_abs": float(stats["offset_abs"]), "bias": float(stats["bias"]),
                "shape_mae": float(stats["shape_mae"]), "y_sd": float(stats["y_sd"]),
                "rank_spearman": float(stats["rank_spearman"]),
                "nn_train_tanimoto": float(sub["nn_train_tanimoto"].max()),
                "nn_base_tanimoto": float(sub["nn_base_tanimoto"].max())
                if "nn_base_tanimoto" in sub.columns else np.nan,
                "n_metals": int(sub["metal_symbol"].nunique()),
                "n_conditions": int(sub["condition_id"].nunique())
                if "condition_id" in sub.columns else np.nan,
                "n_series": int(sub["series_id"].nunique()) if "series_id" in sub.columns else np.nan,
                "ecfp_cluster": str(sub["ecfp_cluster"].iat[0]),
                "tanimoto_cluster": str(sub["tanimoto_cluster"].iat[0]),
                "y_mean": float(sub["log_D"].mean()),
                "prediction_mean": float(sub["prediction"].mean()),
            }
            if "nuisance__batch" in sub.columns:
                record["n_batches"] = int(sub["nuisance__batch"].nunique())
                record["n_publications"] = int(sub["nuisance__doi"].nunique())
            if "rec__name_mismatch" in sub.columns:
                record["frac_name_mismatch"] = float(sub["rec__name_mismatch"].mean())
                record["frac_aqueous_complexant"] = float(sub["rec__aqueous_complexant"].mean())
            for column in ("donor__n_total", "DENTATE", "coreCN"):
                if column in sub.columns:
                    record[column] = float(pd.to_numeric(sub[column], errors="coerce").mean())
            rows.append(record)
    table = pd.DataFrame(rows)
    table["mechanism"] = table.apply(classify, axis=1)
    # Rank by the mean over seeds, so a ligand is not "worst" because one split was unlucky
    ranking = table.groupby("extractant")["mae"].mean().sort_values(ascending=False)
    table["mean_mae_over_seeds"] = table["extractant"].map(ranking)
    table["is_worst"] = table["extractant"].isin(ranking.head(top).index)
    return table.sort_values(["mean_mae_over_seeds", "split_seed"], ascending=[False, True])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="*", required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "runs" / "gen7_architecture" / "error_analysis")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    oof = pd.concat([pd.read_parquet(p) for p in args.oof], ignore_index=True)
    oof = oof.drop_duplicates(["model", "split_seed", "row_id"])

    tables, summaries = [], {}
    for model in args.models:
        table = build(oof, cohort, model=model, top=args.top)
        table.to_csv(args.out / f"worst_ligands_{model}.csv", index=False)
        tables.append(table)
        worst = table[table["is_worst"]]
        counts = worst.groupby("mechanism").size().sort_values(ascending=False)
        share = (worst.groupby("mechanism")["mae"].sum() / worst["mae"].sum()).sort_values(
            ascending=False)
        summaries[model] = {
            "n_ligands": int(table["extractant"].nunique()),
            "worst_mechanism_counts": {k: int(v) for k, v in counts.items()},
            "worst_mechanism_error_share": {k: float(v) for k, v in share.items()},
            "all_mechanism_counts": {k: int(v) for k, v in
                                     table.drop_duplicates("extractant")
                                     .groupby("mechanism").size().items()},
            "worst_20_share_of_total_error": float(
                worst["mae"].sum() / table["mae"].sum()),
            "median_nn_tanimoto_worst": float(worst["nn_train_tanimoto"].median()),
            "median_nn_tanimoto_all": float(table["nn_train_tanimoto"].median()),
        }
        print(f"\n=== {model} ===")
        print(counts.to_string())
        columns = [c for c in ["extractant", "n_rows", "mae", "offset_abs", "shape_mae",
                               "nn_train_tanimoto", "n_batches", "frac_name_mismatch",
                               "mechanism"] if c in worst.columns]
        display = worst.drop_duplicates("extractant")[columns].head(args.top)
        display = display.assign(extractant=display["extractant"].str.slice(0, 46))
        pd.set_option("display.width", 220)
        print(display.to_string(index=False))

    pd.concat(tables, ignore_index=True).to_csv(args.out / "per_ligand_all_models.csv", index=False)
    (args.out / "mechanism_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
