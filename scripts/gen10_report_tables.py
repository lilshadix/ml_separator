#!/usr/bin/env python
"""Assemble every headline table of the gen10 reports, machine-readable, in one place.

Reads only artefacts the phase scripts wrote and writes
``runs/gen10_final/headline_tables/*.csv`` plus ``headline.json``.  The Markdown
reports quote these files and nothing else, so a number in a report can always be
traced to a CSV here and from there to the raw predictions that produced it.
Missing phases are recorded as missing, never silently skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN10 = REPO_ROOT / "runs" / "gen10_final"
OUT = GEN10 / "headline_tables"


def read(path: Path, **kwargs):
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, **kwargs)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    headline: dict = {"missing": []}

    # --- Phase 1: query-set consistency ------------------------------------------
    qc = read(GEN10 / "query_consistency" / "summary_by_arm.csv")
    if qc is not None:
        keep = ["arm", "family", "n_comparisons", "n_ligands", "shift_median__median",
                "shift_median__p90", "shift_p95__median", "shift_p95__p95", "shift_max__max",
                "shape_shift_median__median", "level_shift__median"]
        table = qc[[c for c in keep if c in qc.columns]]
        table.to_csv(OUT / "t1_query_consistency.csv", index=False)
        headline["query_consistency"] = {
            f"{r['arm']}|{r['family']}": {"median_shift": r.get("shift_median__median"),
                                          "p95_shift": r.get("shift_p95__p95")}
            for _, r in table.iterrows()}
        control = json.loads((GEN10 / "query_consistency" / "control.json").read_text())
        headline["query_consistency_control"] = control
    else:
        headline["missing"].append("query_consistency")
    for extra in ("summary_by_axis.csv", "by_width.csv", "worst_ligands.csv", "shifted_window.csv"):
        table = read(GEN10 / "query_consistency" / extra)
        if table is not None:
            table.to_csv(OUT / f"t1_{extra}", index=False)

    # --- Phases 2 and 4: leaderboards ----------------------------------------------
    boards = []
    for stage in ("feature_access", "level_shape", "axis_representation", "set_context"):
        board = read(GEN10 / stage / "leaderboard.csv")
        if board is None:
            headline["missing"].append(stage)
            continue
        board.insert(0, "stage", stage)
        boards.append(board)
        det = read(GEN10 / stage / "determinism.csv")
        if det is not None:
            det.to_csv(OUT / f"t2_determinism_{stage}.csv", index=False)
    if boards:
        board = pd.concat(boards, ignore_index=True)
        board.to_csv(OUT / "t2_leaderboards.csv", index=False)
        headline["leaderboards"] = board.set_index("model")["macro_mae"].round(4).to_dict()

    # --- Phase 5: adaptation -------------------------------------------------------
    summary = read(GEN10 / "adaptation" / "summary.csv")
    if summary is not None:
        pivot = summary[summary["policy"] == "CENTRAL_THEN_SPREAD"].pivot_table(
            index=["global_model", "adapter"], columns="k", values="mae").reset_index()
        pivot.to_csv(OUT / "t5_adaptation_by_k.csv", index=False)
        for name in ("bootstrap_vs_offset_k3.csv", "verdict.csv", "priors.csv"):
            table = read(GEN10 / "adaptation" / name)
            if table is not None:
                table.to_csv(OUT / f"t5_{name}", index=False)
        verdict = read(GEN10 / "adaptation" / "verdict.csv")
        headline["adaptation_passes"] = verdict[verdict["passes"]][
            ["global_model", "policy", "comparison"]].to_dict("records") if verdict is not None else None
    else:
        headline["missing"].append("adaptation")

    # --- Phase 6: acquisition ----------------------------------------------------
    acq = read(GEN10 / "acquisition" / "realised_summary.csv")
    if acq is None:
        candidates = sorted((GEN10 / "acquisition").glob("*summary*.csv"))
        acq = read(candidates[0]) if candidates else None
    if acq is not None:
        acq.to_csv(OUT / "t6_acquisition.csv", index=False)
        if "policy" in acq.columns and "mae" in acq.columns:
            headline["acquisition"] = acq.set_index("policy")["mae"].round(4).to_dict()
    else:
        headline["missing"].append("acquisition")
    for name in sorted((GEN10 / "acquisition").glob("*bootstrap*.csv")):
        read(name).to_csv(OUT / f"t6_{name.name}", index=False)

    # --- Phase 7: data ceiling -----------------------------------------------------
    strat = read(GEN10 / "data_ceiling" / "stratified_metrics.csv")
    if strat is not None:
        strat.to_csv(OUT / "t7_stratified_metrics.csv", index=False)
        for name in ("irreducible_residual_test.csv", "neighbour_vs_model_residuals.csv",
                     "mismatch_audit.csv", "kshot_by_cohort.csv", "cohort_counts.csv"):
            table = read(GEN10 / "data_ceiling" / name)
            if table is not None:
                table.to_csv(OUT / f"t7_{name}", index=False)
        headline["data_ceiling"] = json.loads((GEN10 / "data_ceiling" / "summary.json").read_text())
    else:
        headline["missing"].append("data_ceiling")

    # --- Phase 8: publication sensitivity --------------------------------------------
    intervals = read(GEN10 / "publication_sensitivity" / "intervals_by_scheme.csv")
    if intervals is not None:
        intervals.to_csv(OUT / "t8_intervals_by_scheme.csv", index=False)
        read(GEN10 / "publication_sensitivity" / "leave_one_publication_out.csv").to_csv(
            OUT / "t8_leave_one_publication_out.csv", index=False)
        headline["publication_sensitivity"] = json.loads(
            (GEN10 / "publication_sensitivity" / "summary.json").read_text())
    else:
        headline["missing"].append("publication_sensitivity")

    # --- Phase 9: budget ---------------------------------------------------------------
    budget = read(GEN10 / "budget_simulation" / "budget_strategies.csv")
    if budget is not None:
        budget.to_csv(OUT / "t9_budget_strategies.csv", index=False)
        read(GEN10 / "budget_simulation" / "marginal_gains.csv").to_csv(
            OUT / "t9_marginal_gains.csv", index=False)
        headline["budget"] = json.loads((GEN10 / "budget_simulation" / "summary.json").read_text())
    else:
        headline["missing"].append("budget_simulation")

    # --- Phase 10: error decomposition ----------------------------------------------
    decomposition = read(GEN10 / "error_decomposition" / "decomposition.csv")
    if decomposition is not None:
        decomposition.to_csv(OUT / "t10_error_decomposition.csv", index=False)
        headline["error_decomposition"] = json.loads(
            (GEN10 / "error_decomposition" / "summary.json").read_text())["stages"]
    else:
        headline["missing"].append("error_decomposition")

    # --- Phase 11: final locked ---------------------------------------------------------
    frontier = read(GEN10 / "final_locked" / "frontier_best.csv")
    if frontier is not None:
        frontier.to_csv(OUT / "t11_frontier.csv", index=False)
        for name in ("shape_by_axis.csv", "shape_bootstrap.csv", "strata.csv", "macro_bootstrap.csv",
                     "kshot_bootstrap.csv", "frontier_common.csv"):
            table = read(GEN10 / "final_locked" / name)
            if table is not None:
                table.to_csv(OUT / f"t11_{name}", index=False)
        headline["final_pipeline"] = json.loads(
            (GEN10 / "final_locked" / "pipeline_frontier.json").read_text())
    else:
        headline["missing"].append("final_locked")

    # --- Phase 0 / 12 ------------------------------------------------------------------
    for name, path in (("phase0", GEN10 / "reproduction" / "phase0.json"),
                       ("self_audit", GEN10 / "self_audit" / "self_audit.json")):
        if path.exists():
            payload = json.loads(path.read_text())
            headline[name] = {"all_pass": payload.get("all_pass"),
                              "n_checks": len(payload.get("checks", []))}
        else:
            headline["missing"].append(name)

    (OUT / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    print(json.dumps({k: v for k, v in headline.items() if k in ("missing", "leaderboards",
                                                                 "final_pipeline")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
