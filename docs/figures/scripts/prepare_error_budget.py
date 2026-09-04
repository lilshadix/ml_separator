"""Rebuild the oracle error cascade from raw out-of-fold predictions.

Two purposes:

1. **Verification** — reproduce ``runs/gen10_final/error_decomposition/summary.json``
   exactly, on that study's own cohort (all 152 extractants) and its own aggregation
   (one vote per ECFP cluster).  If this fails the figure is not drawn.
2. **A single comparable axis for Figure 4A** — repeat the cascade restricted to the
   99-extractant k-shot common cohort with one vote per *extractant*, which is the cohort
   and the aggregation the realised k-shot numbers use.  Oracle bounds and achieved
   values can then be drawn on one axis without mixing cohorts.

The oracle definitions are copied from ``scripts/gen10_error_decomposition.py`` so that
the reproduction is a check on the arithmetic, not on a re-specification.

Writes ``figures/derived/error_budget.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402

MODEL = "GEN9_SHAPE_RECOMPOSED"
TOL = 5e-9


def cascade(oof: pd.DataFrame, membership: pd.DataFrame, unit_column: str) -> dict:
    residual = oof["log_D"] - oof["prediction"]
    seed = oof["split_seed"]
    unit = oof[unit_column]

    after_level = residual - residual.groupby([oof["extractant"], seed]).transform("median")
    after_series = residual - residual.groupby([oof["series_id"], seed]).transform("median")

    primary = membership.sort_values(["row_id", "n_points", "curve_id"],
                                     ascending=[True, False, True]).drop_duplicates("row_id")
    oof = oof.assign(curve_id=oof["row_id"].map(primary.set_index("row_id")["curve_id"]),
                     axis_value=oof["row_id"].map(primary.set_index("row_id")["axis_value"]))
    on_curve = oof["curve_id"].notna()
    after_curve_mean = residual - residual.groupby(
        [oof["curve_id"].fillna("none"), seed]).transform("mean")

    repaired = after_curve_mean.copy()
    for _key, block in oof[on_curve].groupby(["curve_id", "split_seed"]):
        x = block["axis_value"].to_numpy(dtype=float)
        r = after_curve_mean.loc[block.index].to_numpy(dtype=float)
        if len(x) >= 3 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, r, 1)
            repaired.loc[block.index] = r - (slope * x + intercept)

    def both(err: pd.Series) -> dict:
        return {"pooled_mae": float(err.abs().mean()),
                "macro_mae": float(err.abs().groupby(unit.to_numpy()).mean().mean())}

    return {
        "current": both(residual),
        "after_ligand_level_oracle": both(after_level),
        "after_series_level_oracle": both(after_series),
        "after_curve_level_oracle": both(
            pd.Series(np.where(on_curve, after_curve_mean, after_level), index=oof.index)),
        "after_curve_level_and_slope_oracle": both(
            pd.Series(np.where(on_curve, repaired, after_level), index=oof.index)),
    }


def main() -> int:
    oof = pd.read_parquet(_paths.run("gen9_shape/recomposed/oof_all.parquet"))
    oof = oof[oof["model"] == MODEL].copy()
    cohort = pd.read_parquet(_paths.run("gen7_architecture/cache/cohort.parquet"),
                             columns=["row_id", "series_id", "condition_id", "ecfp_cluster",
                                      "tanimoto_cluster"])
    cohort = cohort.drop(columns=[c for c in cohort.columns
                                  if c != "row_id" and c in oof.columns])
    oof = oof.merge(cohort, on="row_id", how="left")
    membership = pd.read_parquet(_paths.run("gen9_shape/curves/curve_membership.parquet"))

    published = json.loads(
        _paths.run("gen10_final/error_decomposition/summary.json").read_text())
    reproduced = cascade(oof, membership, "ecfp_cluster")

    checks = []
    for stage, values in published["stages"].items():
        for metric in ("pooled_mae", "macro_mae"):
            delta = abs(reproduced[stage][metric] - values[metric])
            checks.append({"stage": stage, "metric": metric,
                           "published": values[metric], "reproduced": reproduced[stage][metric],
                           "abs_delta": delta, "pass": bool(delta < TOL)})
    if not all(c["pass"] for c in checks):
        print(pd.DataFrame(checks).to_string())
        raise SystemExit("oracle cascade did NOT reproduce — figure not drawn")

    common = json.loads((_paths.DERIVED / "kshot_cohort.json").read_text())
    per_ligand = pd.read_csv(_paths.DERIVED / "kshot_per_ligand.csv")
    ligands = sorted(per_ligand["extractant"].unique())
    restricted = cascade(oof[oof["extractant"].isin(ligands)].reset_index(drop=True),
                         membership, "extractant")

    def realised(adapter: str, policy: str, k: int) -> float:
        block = per_ligand[(per_ligand.global_model == MODEL) & (per_ligand.adapter == adapter)
                           & (per_ligand.policy == policy) & (per_ligand.k == k)]
        return float(block["mae"].mean())

    achieved = {
        "zero_shot": realised("ZERO_SHOT_REF", "NONE", 0),
        "k1_offset_central": realised("OFFSET_K1", "CENTRAL_THEN_SPREAD", 1),
        "k1_slope_repair_central": realised("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD", 1),
        "k2_series_ml_central": realised("SERIES_ML", "CENTRAL_THEN_SPREAD", 2),
        "k3_series_ml_central": realised("SERIES_ML", "CENTRAL_THEN_SPREAD", 3),
        "k5_series_ml_central": realised("SERIES_ML", "CENTRAL_THEN_SPREAD", 5),
        "k1_oracle_point": realised("OFFSET_K1", "ORACLE[OFFSET_K1]", 1),
    }

    payload = {
        "model": MODEL,
        "verification": {"tolerance": TOL, "checks": checks, "all_pass": True},
        "published_study_cohort": {
            "description": "152 extractants, macro = one vote per ECFP cluster (131), 5 seeds",
            "stages": reproduced},
        "common_cohort": {
            "description": f"{len(ligands)} extractants of the k-shot common cohort, "
                           "macro = one vote per extractant, 5 seeds",
            "n_extractants": len(ligands),
            "n_chemotypes": common["n_chemotypes_common"],
            "stages": restricted,
            "achieved": achieved},
        "components": pd.read_csv(
            _paths.run("gen10_final/error_decomposition/decomposition.csv")).to_dict("records"),
    }
    (_paths.DERIVED / "error_budget.json").write_text(json.dumps(payload, indent=1))
    print("verification: all", len(checks), "stage/metric values reproduce within", TOL)
    print("\npublished-cohort macro:",
          {k: round(v["macro_mae"], 4) for k, v in reproduced.items()})
    print("common-cohort  macro:",
          {k: round(v["macro_mae"], 4) for k, v in restricted.items()})
    print("achieved            :", {k: round(v, 4) for k, v in achieved.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
