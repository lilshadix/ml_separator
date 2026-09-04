#!/usr/bin/env python
"""PHASE 7 — how much error lives in rows whose system identity is incomplete?

Diagnostic stratification of *already fitted* models — nothing is refitted, no row
is corrected or deleted.  Five mutually exclusive cohorts, assigned in priority
order so a row lands in exactly one:

``D_TWE24``        the one ligand gen8 and gen9 both flag as a suspected
                   ~3-decade transcription error (UNRESOLVED, retained);
``C_DUPLICATE``    the 70 DMDPhPDA rows gen8 showed are one publication entered
                   twice, plus every other row in an exact-decade duplicate cell;
``B_MISMATCH``     the 353 rows whose recorded extractant name does not match the
                   structure they are modelled as (gen7's unmodelled second
                   species, ``rec__name_mismatch == 1``);
``E_UNCERTAIN``    every remaining row of a ligand gen9's scan flagged as a
                   family-local level outlier;
``A_CONSISTENT``   everything else — representation-consistent single-component rows.

For each cohort and each model: row MAE, ligand macro MAE, the offset / shape
decomposition, and the k = 0 / 1 / 2 / 5 frontier from the stored k-shot detail.
Then the two tests the brief asks for: do mismatch rows carry systematically
larger *irreducible* residuals (error that survives a per-ligand offset), and do
the nearest-neighbour and model residual distributions differ on them?  Finally an
audit table per mismatch extractant: recorded name, represented structure, the
names the corpus attaches to that structure, publications, rows, current error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen9.metrics import curve_shape_table  # noqa: E402

GEN9 = REPO_ROOT / "runs" / "gen9_shape"
GEN8 = REPO_ROOT / "runs" / "gen8_architecture"
GEN7 = REPO_ROOT / "runs" / "gen7_architecture"
OUT = REPO_ROOT / "runs" / "gen10_final" / "data_ceiling"
COHORT_PATH = GEN7 / "cache" / "cohort.parquet"
RECOVERED_PATH = GEN7 / "cache" / "recovered_cells.parquet"
MEMBERSHIP = GEN9 / "curves" / "curve_membership.parquet"
TWE24 = "CCCCOP(=S)(CP(=S)(OCCCC)OCCCC)OCCCC"
COHORT_ORDER = ("A_CONSISTENT", "B_MISMATCH", "C_DUPLICATE", "D_TWE24", "E_UNCERTAIN")
DEFAULT_MODELS = ("REC_ecfp_plus_recovered", "GEN9_REL_MONOLITH", "GEN9_SHAPE_RECOMPOSED")


def assign_cohorts(cohort: pd.DataFrame, recovered: pd.DataFrame) -> pd.Series:
    flags = json.loads((GEN8 / "case_studies" / "dmdphpda_cohorts.json").read_text())
    duplicate_ids = set(str(r) for r in flags["quarantined"])
    duplicates = pd.read_csv(GEN9 / "data_audit" / "duplicate_cells.csv") \
        if (GEN9 / "data_audit" / "duplicate_cells.csv").exists() else pd.DataFrame()
    decade_ligands = set()
    if not duplicates.empty and "is_exact_decade" in duplicates:
        decade_ligands = set(duplicates[duplicates["is_exact_decade"]]["extractant"].astype(str))
    outliers = pd.read_csv(GEN9 / "data_audit" / "level_outliers.csv") \
        if (GEN9 / "data_audit" / "level_outliers.csv").exists() else pd.DataFrame()
    outlier_ligands = set(outliers["extractant"].astype(str)) if "extractant" in outliers else set()
    mismatch_ids = set(recovered.loc[recovered["rec__name_mismatch"].fillna(0) > 0,
                                     "row_id"].astype(str))

    row_ids = cohort["row_id"].astype(str)
    ligands = cohort["extractant"].astype(str)
    out = pd.Series("A_CONSISTENT", index=cohort.index)
    out[ligands.isin(outlier_ligands)] = "E_UNCERTAIN"
    out[row_ids.isin(mismatch_ids)] = "B_MISMATCH"
    # a DMDPhPDA row is a duplicate whether or not it is also in a decade ligand
    out[row_ids.isin(duplicate_ids) | ligands.isin(decade_ligands - {TWE24})] = "C_DUPLICATE"
    out[ligands == TWE24] = "D_TWE24"
    return out


def offset_shape(block: pd.DataFrame) -> tuple[float, float]:
    """gen6's per-ligand decomposition: |mean residual| and MAE after removing it."""
    residual = block["log_D"] - block["prediction"]
    offsets, shapes = [], []
    for _, sub in residual.groupby(block["extractant"]):
        offsets.append(abs(float(sub.mean())))
        shapes.append(float(np.abs(sub - sub.mean()).mean()))
    return float(np.mean(offsets)), float(np.mean(shapes))


def stratify(oof: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cohort_name), block in oof.groupby(["model", "cohort"], sort=True):
        per_ligand = (block["log_D"] - block["prediction"]).abs().groupby(block["extractant"]).mean()
        per_cluster = (block["log_D"] - block["prediction"]).abs().groupby(block["ecfp_cluster"]).mean()
        offset, shape = offset_shape(block)
        shapes = []
        for seed, sub in block.groupby("split_seed"):
            table = curve_shape_table(sub, membership)
            if not table.empty:
                shapes.append(table["shape_mae"].mean())
        rows.append({"model": model, "cohort": cohort_name,
                     "n_rows": int(len(block)), "n_ligands": int(block["extractant"].nunique()),
                     "row_mae": float((block["log_D"] - block["prediction"]).abs().mean()),
                     "ligand_macro_mae": float(per_ligand.mean()),
                     "cluster_macro_mae": float(per_cluster.mean()),
                     "offset_mae": offset, "ligand_shape_mae": shape,
                     "curve_shape_mae": float(np.mean(shapes)) if shapes else np.nan})
    return pd.DataFrame(rows)


def kshot_by_cohort(detail: pd.DataFrame, cohort_of_ligand: pd.Series) -> pd.DataFrame:
    """Best deployable arm per (model, cohort, k), one ligand one vote."""
    work = detail[detail["deployable"]].copy()
    work["cohort"] = work["extractant"].map(cohort_of_ligand)
    work["arm"] = np.where(work["adapter"].isin(["ZERO_SHOT_REF", "ZERO_SHOT"]), "ZERO_SHOT",
                           work["adapter"] + "@" + work["policy"])
    per = work.groupby(["global_model", "cohort", "arm", "k", "extractant"])["mae"].mean().reset_index()
    summary = per.groupby(["global_model", "cohort", "arm", "k"]).agg(
        mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()
    rows = []
    for (model, cohort_name, k), block in summary.groupby(["global_model", "cohort", "k"]):
        reference = block[block["arm"].isin(["ZERO_SHOT", "OFFSET_K1@CENTRAL_THEN_SPREAD",
                                             "SLOPE_L_s1_K3@CENTRAL_THEN_SPREAD",
                                             "OFFSET_K3@CENTRAL_THEN_SPREAD"])]
        best = block.sort_values("mae").iloc[0]
        rows.append({"global_model": model, "cohort": cohort_name, "k": int(k),
                     "best_arm": best["arm"], "best_mae": float(best["mae"]),
                     "n_ligands": int(best["n_ligands"]),
                     **{f"mae_{r['arm']}": float(r["mae"]) for _, r in reference.iterrows()}})
    return pd.DataFrame(rows)


def irreducible_test(oof: pd.DataFrame, model: str) -> dict:
    """Do mismatch rows carry larger residuals *after* a per-ligand offset?"""
    block = oof[oof["model"] == model].copy()
    residual = block["log_D"] - block["prediction"]
    centred = residual - residual.groupby([block["extractant"], block["split_seed"]]).transform("mean")
    block["irreducible"] = centred.abs()
    block["raw"] = residual.abs()
    a = block[block["cohort"] == "B_MISMATCH"]
    b = block[block["cohort"] == "A_CONSISTENT"]
    out = {"model": model, "n_mismatch_rows": int(len(a)), "n_consistent_rows": int(len(b))}
    for column in ("raw", "irreducible"):
        u = stats.mannwhitneyu(a[column], b[column], alternative="two-sided")
        out[f"{column}_median_mismatch"] = float(a[column].median())
        out[f"{column}_median_consistent"] = float(b[column].median())
        out[f"{column}_mean_mismatch"] = float(a[column].mean())
        out[f"{column}_mean_consistent"] = float(b[column].mean())
        out[f"{column}_mannwhitney_p"] = float(u.pvalue)
        # a ligand-level version, because rows within a ligand are not independent
        la = a.groupby("extractant")[column].mean()
        lb = b.groupby("extractant")[column].mean()
        out[f"{column}_ligand_median_mismatch"] = float(la.median())
        out[f"{column}_ligand_median_consistent"] = float(lb.median())
        out[f"{column}_ligand_mannwhitney_p"] = float(
            stats.mannwhitneyu(la, lb, alternative="two-sided").pvalue) if len(la) > 2 else np.nan
    return out


def neighbour_vs_model(oof: pd.DataFrame, cohort: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per cohort: the model's residual against a 1-NN (condition-cell) lookup residual.

    The lookup predicts a row by the mean ``log_D`` of the same condition cell
    among *other* ligands of the training fold — the no-chemistry baseline gen7's
    ceiling analysis used.  Where the two residual distributions differ, the
    model is adding or subtracting something the lookup cannot see.
    """
    block = oof[oof["model"] == model]
    if "condition_id" not in block.columns:
        block = block.merge(cohort[["row_id", "condition_id"]], on="row_id", how="left")
    rows = []
    for seed, sub in block.groupby("split_seed"):
        # The lookup may only use rows of *other* folds — the fold's training
        # chemistry.  Pooling same-fold ligands would hand it same-chemotype
        # siblings the model was never shown, and it would win for that reason.
        sub = sub.copy()
        cell_fold = sub.groupby(["condition_id", "fold"])["log_D"].agg(["sum", "size"]).reset_index()
        cell_all = sub.groupby("condition_id")["log_D"].agg(["sum", "size"]).reset_index()
        merged = sub[["condition_id", "fold"]].merge(cell_all, on="condition_id", how="left") \
            .merge(cell_fold, on=["condition_id", "fold"], how="left", suffixes=("", "_fold"))
        other_n = (merged["size"] - merged["size_fold"]).to_numpy(dtype=float)
        other_sum = (merged["sum"] - merged["sum_fold"]).to_numpy(dtype=float)
        other = np.divide(other_sum, other_n, out=np.full(len(sub), np.nan), where=other_n > 0)
        sub = sub.assign(nn_residual=(sub["log_D"].to_numpy() - other).__abs__(),
                         model_residual=(sub["log_D"] - sub["prediction"]).abs())
        for cohort_name, part in sub.groupby("cohort"):
            part = part.dropna(subset=["nn_residual"])
            if len(part) < 5:
                continue
            w = stats.wilcoxon(part["model_residual"], part["nn_residual"])
            rows.append({"model": model, "split_seed": int(seed), "cohort": cohort_name,
                         "n_rows": int(len(part)),
                         "model_residual_median": float(part["model_residual"].median()),
                         "nn_residual_median": float(part["nn_residual"].median()),
                         "model_minus_nn_mean": float(
                             (part["model_residual"] - part["nn_residual"]).mean()),
                         "wilcoxon_p": float(w.pvalue)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.groupby(["model", "cohort"]).agg(
        n_rows=("n_rows", "mean"), model_residual_median=("model_residual_median", "mean"),
        nn_residual_median=("nn_residual_median", "mean"),
        model_minus_nn_mean=("model_minus_nn_mean", "mean"),
        seeds_model_better=("model_minus_nn_mean", lambda s: int((s < 0).sum())),
        median_wilcoxon_p=("wilcoxon_p", "median")).reset_index()


def mismatch_audit(cohort: pd.DataFrame, recovered: pd.DataFrame, oof: pd.DataFrame,
                   model: str) -> pd.DataFrame:
    flagged = recovered[recovered["rec__name_mismatch"].fillna(0) > 0]
    merged = cohort[["row_id", "extractant", "series_id"]].merge(
        flagged[["row_id", "nuisance__extractant_name", "nuisance__doi",
                 "rec__n_names_for_structure", "rec__aqueous_complexant"]], on="row_id")
    errors = oof[oof["model"] == model].assign(
        abs_error=lambda d: (d["log_D"] - d["prediction"]).abs())
    per_ligand_error = errors.groupby("extractant")["abs_error"].mean()
    all_names = recovered.merge(cohort[["row_id", "extractant"]], on="row_id").groupby(
        "extractant")["nuisance__extractant_name"].agg(lambda s: sorted(set(s.dropna().astype(str))))
    rows = []
    for ligand, block in merged.groupby("extractant"):
        rows.append({
            "represented_structure": ligand,
            "recorded_names_on_mismatch_rows": "; ".join(
                sorted(set(block["nuisance__extractant_name"].dropna().astype(str)))),
            "all_names_for_structure": "; ".join(all_names.get(ligand, [])),
            "n_names_for_structure": int(block["rec__n_names_for_structure"].max()),
            "possible_second_species": "; ".join(sorted(set(
                block["rec__aqueous_complexant"].dropna().astype(str)))) or "unrecorded",
            "publications": "; ".join(sorted(set(block["nuisance__doi"].dropna().astype(str)))),
            "n_publications": int(block["nuisance__doi"].nunique()),
            "n_mismatch_rows": int(len(block)),
            "n_rows_total": int((cohort["extractant"] == ligand).sum()),
            "current_mae": float(per_ligand_error.get(ligand, np.nan)),
        })
    return pd.DataFrame(rows).sort_values("n_mismatch_rows", ascending=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", nargs="+", type=Path, default=[
        GEN9 / "finalists" / "oof_all.parquet", GEN9 / "relmono" / "oof_all.parquet",
        GEN9 / "recomposed" / "oof_all.parquet"])
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--detail", nargs="*", type=Path, default=[
        GEN9 / "kshot" / "frozen_detail.parquet", GEN9 / "kshot" / "new_detail.parquet",
        GEN9 / "kshot" / "recomposed_detail.parquet"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(COHORT_PATH)
    recovered = pd.read_parquet(RECOVERED_PATH)
    membership = pd.read_parquet(MEMBERSHIP)
    cohort["cohort"] = assign_cohorts(cohort, recovered).to_numpy()
    assignment = cohort[["row_id", "extractant", "tanimoto_cluster", "cohort"]]
    assignment.to_csv(args.out / "row_cohorts.csv", index=False)
    counts = assignment.groupby("cohort").agg(n_rows=("row_id", "size"),
                                              n_ligands=("extractant", "nunique")).reindex(COHORT_ORDER)
    counts.to_csv(args.out / "cohort_counts.csv")
    print(counts.to_string())
    ligand_cohort = (assignment.groupby("extractant")["cohort"]
                     .agg(lambda s: s.value_counts().index[0]))

    oof = pd.concat([pd.read_parquet(p) for p in args.oof], ignore_index=True)
    oof = oof[oof["model"].isin(args.models)].drop_duplicates(["model", "split_seed", "row_id"])
    oof = oof.merge(assignment[["row_id", "cohort"]], on="row_id", how="left")
    if "ecfp_cluster" not in oof.columns:
        oof = oof.merge(cohort[["row_id", "ecfp_cluster"]], on="row_id", how="left")

    strat = stratify(oof, membership)
    strat.to_csv(args.out / "stratified_metrics.csv", index=False)
    print("\n" + strat.pivot(index="cohort", columns="model", values="ligand_macro_mae")
          .reindex(COHORT_ORDER).round(3).to_string())

    detail = pd.concat([pd.read_parquet(p, columns=[
        "global_model", "split_seed", "extractant", "policy", "adapter", "k", "deployable", "mae"])
        for p in args.detail], ignore_index=True)
    detail = detail[detail["global_model"].isin(args.models)]
    kshot = kshot_by_cohort(detail, ligand_cohort)
    kshot.to_csv(args.out / "kshot_by_cohort.csv", index=False)

    tests = pd.DataFrame([irreducible_test(oof, m) for m in args.models])
    tests.to_csv(args.out / "irreducible_residual_test.csv", index=False)
    nn = pd.concat([neighbour_vs_model(oof, cohort, m) for m in args.models], ignore_index=True)
    nn.to_csv(args.out / "neighbour_vs_model_residuals.csv", index=False)

    audit = mismatch_audit(cohort, recovered, oof, "GEN9_SHAPE_RECOMPOSED"
                           if "GEN9_SHAPE_RECOMPOSED" in args.models else args.models[0])
    audit.to_csv(args.out / "mismatch_audit.csv", index=False)

    # --- the headline: error associated with incomplete representation -------
    best = "GEN9_SHAPE_RECOMPOSED" if "GEN9_SHAPE_RECOMPOSED" in args.models else args.models[0]
    block = oof[oof["model"] == best]
    err = (block["log_D"] - block["prediction"]).abs()
    total = float(err.sum())
    share = {c: float(err[block["cohort"] == c].sum() / total) for c in COHORT_ORDER}
    rows_share = {c: float((block["cohort"] == c).mean()) for c in COHORT_ORDER}
    consistent_mae = float(err[block["cohort"] == "A_CONSISTENT"].mean())
    excess = {c: float(max(err[block["cohort"] == c].mean() - consistent_mae, 0.0)
                       * (block["cohort"] == c).sum() / len(block))
              for c in COHORT_ORDER if c != "A_CONSISTENT"}
    summary = {
        "model": best, "pooled_mae": float(err.mean()),
        "share_of_total_absolute_error": share, "share_of_rows": rows_share,
        "mae_by_cohort": {c: float(err[block["cohort"] == c].mean()) for c in COHORT_ORDER},
        "pooled_mae_excess_attributable": excess,
        "pooled_mae_excess_total": float(sum(excess.values())),
        "pooled_mae_if_all_rows_were_consistent_grade": consistent_mae,
        "note": ("excess = (cohort MAE - consistent MAE) x cohort row share; it is the "
                 "pooled-MAE reduction if those rows errored like consistent rows, an "
                 "upper bound on what fixing their representation could recover"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nirreducible residual, mismatch vs consistent:")
    print(tests[["model", "irreducible_median_mismatch", "irreducible_median_consistent",
                 "irreducible_ligand_mannwhitney_p"]].round(4).to_string(index=False))
    print(f"\nexcess pooled MAE attributable to non-consistent cohorts: "
          f"{summary['pooled_mae_excess_total']:.4f} of {summary['pooled_mae']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
