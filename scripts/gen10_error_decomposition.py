#!/usr/bin/env python
"""PHASE 10 — where the remaining error lives, and how much of it is reachable.

For one global model, partition held-out error into the brief's seven components
using counterfactual controls that are each *legitimate as a bound* and labelled
as oracle where they read a target:

1. ``predictable_level``     error a perfect per-ligand constant offset removes
                              (``ORACLE_level``: the median residual, known) — the
                              part one measurement is supposed to supply;
2. ``curve_shape``           error a perfect per-curve shape removes on top of the
                              level: rows on a curve, residual after the curve's
                              own mean and after a per-curve oracle slope repair;
3. ``series_local``          level error that is series-specific: the gap between
                              a ligand-wide oracle offset and a per-series oracle
                              offset;
4. ``system_identity``       Phase 7's excess on name/structure-mismatch rows;
5. ``sparse_support``        excess error of ligands in the far tercile of
                              Tanimoto distance to the training chemistry;
6. ``data_quality``          Phase 7's excess on duplicate, TWE-24 and flagged
                              rows;
7. ``residual``              what is left once everything above is removed.

Each component is reported three ways: its current contribution to pooled and
macro MAE, the oracle-removable amount (the bound), and the realistically
removable amount — which is the part a *deployable* intervention in this study
actually recovered (e.g. k = 1 central for the level, the recomposition for
shape, ``OFFSET_K3`` at k = 3 for series-local), read off the stored k-shot
detail so the number is a measurement, not a guess.
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

from lanthanide_separation.gen9.metrics import curve_shape_table  # noqa: E402

GEN9 = REPO_ROOT / "runs" / "gen9_shape"
GEN7 = REPO_ROOT / "runs" / "gen7_architecture"
OUT = REPO_ROOT / "runs" / "gen10_final" / "error_decomposition"
CEILING = REPO_ROOT / "runs" / "gen10_final" / "data_ceiling"


def macro(err: pd.Series, unit: pd.Series) -> float:
    return float(err.groupby(unit.to_numpy()).mean().mean())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, default=GEN9 / "recomposed" / "oof_all.parquet")
    parser.add_argument("--model", default="GEN9_SHAPE_RECOMPOSED")
    parser.add_argument("--detail", type=Path, default=GEN9 / "kshot" / "recomposed_detail.parquet")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(args.oof)
    oof = oof[oof["model"] == args.model].copy()
    cohort = pd.read_parquet(GEN7 / "cache" / "cohort.parquet",
                             columns=["row_id", "series_id", "condition_id", "ecfp_cluster",
                                      "tanimoto_cluster"])
    for column in cohort.columns:
        if column != "row_id" and column in oof.columns:
            cohort = cohort.drop(columns=[column])
    oof = oof.merge(cohort, on="row_id", how="left")
    membership = pd.read_parquet(GEN9 / "curves" / "curve_membership.parquet")
    row_cohorts = pd.read_csv(CEILING / "row_cohorts.csv")[["row_id", "cohort"]]
    oof = oof.merge(row_cohorts, on="row_id", how="left")

    residual = oof["log_D"] - oof["prediction"]
    unit = oof["ecfp_cluster"]
    seed = oof["split_seed"]
    ligand = oof["extractant"]

    # --- 1. level: per-ligand oracle constant ---------------------------------
    ligand_median = residual.groupby([ligand, seed]).transform("median")
    after_level = residual - ligand_median
    # --- 3. series-local: per-series oracle constant ---------------------------
    series_median = residual.groupby([oof["series_id"], seed]).transform("median")
    after_series = residual - series_median
    # --- 2. shape: per-curve oracle mean (level) + oracle slope ---------------
    primary = membership.sort_values(["row_id", "n_points", "curve_id"],
                                     ascending=[True, False, True]).drop_duplicates("row_id")
    curve_of = primary.set_index("row_id")["curve_id"]
    axis_value = primary.set_index("row_id")["axis_value"]
    oof["curve_id"] = oof["row_id"].map(curve_of)
    oof["axis_value"] = oof["row_id"].map(axis_value)
    on_curve = oof["curve_id"].notna()
    curve_key = [oof["curve_id"].fillna("none"), seed]
    curve_mean = residual.groupby(curve_key).transform("mean")
    after_curve_mean = residual - curve_mean
    # oracle per-curve slope repair: fit residual ~ a + b*x per curve, remove the fit
    repaired = after_curve_mean.copy()
    for (cid, s), block in oof[on_curve].groupby(["curve_id", "split_seed"]):
        x = block["axis_value"].to_numpy(dtype=float)
        r = after_curve_mean.loc[block.index].to_numpy(dtype=float)
        if len(x) >= 3 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, r, 1)
            repaired.loc[block.index] = r - (slope * x + intercept)

    def both(err: pd.Series) -> dict:
        return {"pooled_mae": float(err.abs().mean()), "macro_mae": macro(err.abs(), unit)}

    stages = {
        "current": both(residual),
        "after_ligand_level_oracle": both(after_level),
        "after_series_level_oracle": both(after_series),
        "after_curve_level_oracle": both(pd.Series(np.where(on_curve, after_curve_mean, after_level),
                                                   index=oof.index)),
        "after_curve_level_and_slope_oracle": both(pd.Series(np.where(on_curve, repaired, after_level),
                                                             index=oof.index)),
    }

    # --- realistically removable: what deployable arms recovered --------------
    detail = pd.read_parquet(args.detail, columns=["global_model", "extractant", "policy",
                                                   "adapter", "k", "mae", "deployable"])
    detail = detail[detail["global_model"] == args.model]

    def frontier(adapter: str, policy: str, k: int) -> float:
        block = detail[(detail["adapter"] == adapter) & (detail["policy"] == policy)
                       & (detail["k"] == k)]
        return float(block.groupby("extractant")["mae"].mean().mean()) if len(block) else np.nan

    zero_shot = float(detail[detail["k"] == 0].groupby("extractant")["mae"].mean().mean())
    realised = {
        "zero_shot_common_protocol": zero_shot,
        "k1_central_offset": frontier("OFFSET_K1", "CENTRAL_THEN_SPREAD", 1),
        "k1_central_slope_repair": frontier("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD", 1),
        "k3_offset_k3": frontier("OFFSET_K3", "CENTRAL_THEN_SPREAD", 3),
        "k3_slope_k3": frontier("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD", 3),
        "k5_slope_k3": frontier("SLOPE_L_s1_K3", "CENTRAL_THEN_SPREAD", 5),
        "k1_oracle_point": frontier("OFFSET_K1", "ORACLE[OFFSET_K1]", 1),
    }

    # --- 4-6. data cohorts and sparse support, as ligand-level macro excess -------
    # Component = (mean per-ligand MAE of the affected ligands - that of consistent
    # ligands) x (affected share of ligands).  The same arithmetic as Phase 7, on
    # the unit the headline is scored on; a negative raw value means the affected
    # ligands are *easier* than average and is reported as such, clipped at zero
    # in the removable column.
    err = residual.abs()
    per_ligand = err.groupby([ligand, seed]).mean().groupby(level=0).mean()
    ligand_cohort = oof.groupby("extractant")["cohort"].agg(lambda s: s.value_counts().index[0])
    consistent_ligands = per_ligand[ligand_cohort.reindex(per_ligand.index) == "A_CONSISTENT"]
    cohort_excess = {}
    for name in ("B_MISMATCH", "C_DUPLICATE", "D_TWE24", "E_UNCERTAIN"):
        affected = per_ligand[ligand_cohort.reindex(per_ligand.index) == name]
        if len(affected):
            raw = (float(affected.mean()) - float(consistent_ligands.mean())) \
                * len(affected) / len(per_ligand)
            cohort_excess[name] = {"n_ligands": int(len(affected)),
                                   "n_rows": int((oof["cohort"] == name).sum()),
                                   "ligand_mae": float(affected.mean()),
                                   "consistent_ligand_mae": float(consistent_ligands.mean()),
                                   "macro_excess_raw": raw,
                                   "macro_excess": max(raw, 0.0)}
    ligand_distance = (1.0 - oof["nn_train_tanimoto"]).groupby(ligand).first()
    far_ligands = ligand_distance[ligand_distance >= ligand_distance.quantile(2 / 3)].index
    far = per_ligand.reindex(far_ligands).dropna()
    rest = per_ligand.drop(index=far.index)
    sparse_raw = (float(far.mean()) - float(rest.mean())) * len(far) / len(per_ligand)
    sparse = {"n_ligands_far": int(len(far)), "ligand_mae_far": float(far.mean()),
              "ligand_mae_rest": float(rest.mean()), "macro_excess_raw": sparse_raw,
              "macro_excess": max(sparse_raw, 0.0)}

    # --- the table --------------------------------------------------------------
    cur = stages["current"]["macro_mae"]
    table = [
        {"component": "1 predictable level (per-ligand constant)",
         "current_contribution": cur - stages["after_ligand_level_oracle"]["macro_mae"],
         "oracle_removable": cur - stages["after_ligand_level_oracle"]["macro_mae"],
         "realistically_removable": zero_shot - realised["k1_central_offset"],
         "required_intervention": "one measurement per ligand (central), OFFSET_K1",
         "basis": "ORACLE_level vs OFFSET_K1@CENTRAL_THEN_SPREAD, k=1"},
        {"component": "3 series-local level (beyond the ligand constant)",
         "current_contribution": stages["after_ligand_level_oracle"]["macro_mae"]
         - stages["after_series_level_oracle"]["macro_mae"],
         "oracle_removable": stages["after_ligand_level_oracle"]["macro_mae"]
         - stages["after_series_level_oracle"]["macro_mae"],
         "realistically_removable": realised["k1_central_offset"] - realised["k3_offset_k3"],
         "required_intervention": "measure inside each series; OFFSET_K3 / corrected SERIES adapter at k>=3",
         "basis": "per-series oracle offset vs per-ligand; OFFSET_K1@k1 vs OFFSET_K3@k3"},
        {"component": "2 curve shape (within-curve, after the curve's level)",
         "current_contribution": stages["after_curve_level_oracle"]["macro_mae"]
         - stages["after_curve_level_and_slope_oracle"]["macro_mae"],
         "oracle_removable": stages["after_curve_level_oracle"]["macro_mae"]
         - stages["after_curve_level_and_slope_oracle"]["macro_mae"],
         "realistically_removable": realised["k1_central_offset"] - realised["k1_central_slope_repair"],
         "required_intervention": "gen8 slope repair; gen9 recomposition is already in the model",
         "basis": "oracle per-curve slope vs SLOPE_L_s1_K1@k1"},
        {"component": "4 system identity (name/structure mismatch rows)",
         "current_contribution": cohort_excess.get("B_MISMATCH", {}).get("macro_excess_raw", 0.0),
         "oracle_removable": cohort_excess.get("B_MISMATCH", {}).get("macro_excess", 0.0),
         "realistically_removable": 0.0,
         "required_intervention": "record the second species / represent the mixture (data work, not modelling)",
         "basis": "Phase 7 cohort B vs A"},
        {"component": "5 sparse support / distant chemotype",
         "current_contribution": sparse["macro_excess_raw"],
         "oracle_removable": sparse["macro_excess"],
         "realistically_removable": 0.0,
         "required_intervention": "measure new chemotypes (Phase 9: one point on a far ligand is worth 0.49)",
         "basis": "far tercile of Tanimoto distance errored like the rest"},
        {"component": "6 suspected data quality (duplicates, TWE-24, flagged)",
         "current_contribution": sum(v["macro_excess_raw"]
                                     for k, v in cohort_excess.items() if k != "B_MISMATCH"),
         "oracle_removable": sum(v["macro_excess"]
                                 for k, v in cohort_excess.items() if k != "B_MISMATCH"),
         "realistically_removable": 0.0,
         "required_intervention": "primary-source check of TWE-24 and the decade duplicates",
         "basis": "Phase 7 cohorts C, D, E vs A"},
    ]
    explained = sum(r["oracle_removable"] for r in table)
    table.append({"component": "7 residual unexplained",
                  "current_contribution": max(cur - explained, 0.0),
                  "oracle_removable": 0.0, "realistically_removable": 0.0,
                  "required_intervention": "none identified",
                  "basis": "current macro MAE minus the sum of oracle-removable components "
                           "(components overlap; this is a floor, not an exact remainder)"})
    frame = pd.DataFrame(table)
    frame.to_csv(args.out / "decomposition.csv", index=False)
    payload = {"model": args.model, "stages": stages, "realised": realised,
               "cohort_excess": cohort_excess, "sparse_support": sparse,
               "macro_current": cur, "sum_oracle_removable": explained}
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(stages, indent=2))
    print(frame.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
