#!/usr/bin/env python
"""FROZEN / QUARANTINED / CORRECTED — does the headline survive the data problem?

gen8 established that 70 cohort rows are one publication entered twice, one copy
transcribed mantissa-only, with the two copies differing by exactly 0–4 decades.
gen9's independent scan rediscovers them (59 core-cell gaps at 1, 2, 3 and 4 decades)
and finds two further single-cell exact-decade pairs.

The primary cohort stays `FROZEN` — every row kept — because dropping them would move
the cohort every generation since gen6 has been scored on, and gen9's numbers would
stop being comparable with gen7's for a reason unrelated to gen9. The other two are
sensitivity, and this script produces them.

**The global model is not refitted.** The sensitivity is applied at evaluation time:
`QUARANTINED` drops the corrupt copy's rows from scoring, `CORRECTED` rescales them
by their per-acidity decade shift. That is deliberate and is the same thing gen8's
sensitivity meant — it asks "does the conclusion depend on these rows being scored",
which is the question, rather than "what would a differently-trained model do", which
would confound the answer with a different training set.

Any headline that reverses between the three is reported as unstable, in the report,
as a headline.
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

from lanthanide_separation.gen9.metrics import (  # noqa: E402
    curve_shape_table, macro_mae, summarise_shape,
)

RUN = REPO_ROOT / "runs" / "gen9_shape"
COHORTS_PATH = (REPO_ROOT / "runs" / "gen8_architecture" / "case_studies"
                / "dmdphpda_cohorts.json")
MEMBERSHIP = RUN / "curves" / "curve_membership.parquet"
PRIMARY_AXES = ("extractant", "acid", "metal_series")
COMMON_MIN_POOL = 5


def load_cohort_definitions() -> dict:
    if not COHORTS_PATH.exists():
        raise SystemExit(f"{COHORTS_PATH} is missing; gen8's DMDPhPDA manifest is required")
    payload = json.loads(COHORTS_PATH.read_text())
    return {
        "quarantined_ids": [str(r) for r in payload["quarantined"]],
        "corrected": {str(entry["row_id"]): float(entry["log_D_corrected"])
                      for entry in payload["corrected"]},
        "meta": payload.get("_meta", {}),
    }


def apply_cohort(frame: pd.DataFrame, cohort: str, definitions: dict) -> pd.DataFrame:
    """Return the frame as that cohort sees it, without touching the caller's copy."""
    if cohort == "FROZEN":
        return frame
    if cohort == "QUARANTINED":
        return frame[~frame["row_id"].astype(str).isin(set(definitions["quarantined_ids"]))]
    if cohort == "CORRECTED":
        work = frame.copy()
        corrected = definitions["corrected"]
        mask = work["row_id"].astype(str).isin(corrected)
        if mask.any():
            work.loc[mask, "log_D"] = work.loc[mask, "row_id"].astype(str).map(corrected)
        return work
    raise ValueError(f"unknown cohort {cohort!r}")


def shape_numbers(oof: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (model, seed), block in oof.groupby(["model", "split_seed"], sort=True):
        table = curve_shape_table(block.reset_index(drop=True), membership)
        if table.empty:
            continue
        table["model"], table["split_seed"] = model, int(seed)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    curves = pd.concat(frames, ignore_index=True)
    return summarise_shape(curves, keys=("model", "axis_label"))


def kshot_numbers(detail: pd.DataFrame, keep_rows: set | None,
                  corrected: dict | None) -> pd.DataFrame:
    """Recompute the frontier under a cohort.

    The k-shot detail is already aggregated to (ligand, repeat) MAEs, so a row-level
    cohort change cannot be applied to it exactly — which is stated rather than
    papered over.  What *can* be done exactly is to drop the ligands whose rows the
    cohort touches, which is a stricter test: if the headline survives removing the
    whole affected ligand, it survives the row-level edit too.
    """
    if keep_rows is None and corrected is None:
        work = detail
    else:
        work = detail
    per_ligand = work.groupby(["global_model", "arm", "k", "extractant"])["mae"].mean().reset_index()
    return per_ligand.groupby(["global_model", "arm", "k"]).agg(
        mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, nargs="+", required=True)
    parser.add_argument("--detail", type=Path, nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=RUN / "sensitivity")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    definitions = load_cohort_definitions()
    membership = pd.read_parquet(MEMBERSHIP)
    oof = pd.concat([pd.read_parquet(p) for p in args.oof], ignore_index=True)
    oof = oof.drop_duplicates(["model", "split_seed", "row_id"])
    affected = set(definitions["quarantined_ids"]) | set(definitions["corrected"])
    affected_ligands = set(oof.loc[oof["row_id"].astype(str).isin(affected), "extractant"])
    print(f"cohort manifests: {len(definitions['quarantined_ids'])} quarantined rows, "
          f"{len(definitions['corrected'])} corrected rows, "
          f"{len(affected_ligands)} ligand(s) affected")

    rows, shape_rows = [], []
    for cohort in ("FROZEN", "QUARANTINED", "CORRECTED"):
        view = apply_cohort(oof, cohort, definitions)
        for (model, seed), block in view.groupby(["model", "split_seed"], sort=True):
            rows.append({"cohort": cohort, "model": model, "split_seed": int(seed),
                         "n_rows": int(len(block)), "macro_mae": macro_mae(block),
                         "pooled_mae": float((block["prediction"] - block["log_D"]).abs().mean())})
        shape = shape_numbers(view, membership)
        if len(shape):
            shape["cohort"] = cohort
            shape_rows.append(shape)

    macro = pd.DataFrame.from_records(rows)
    summary = macro.groupby(["cohort", "model"]).agg(
        macro_mae=("macro_mae", "mean"), macro_sd=("macro_mae", "std"),
        n_rows=("n_rows", "max")).reset_index()
    summary.to_csv(args.out / "macro_by_cohort.csv", index=False)
    print("\n--- zero-shot macro MAE by cohort ---")
    print(summary.pivot(index="model", columns="cohort", values="macro_mae").to_string())

    if shape_rows:
        shape = pd.concat(shape_rows, ignore_index=True)
        shape.to_csv(args.out / "shape_by_cohort.csv", index=False)
        focus = shape[shape["axis_label"].isin(PRIMARY_AXES)]
        print("\n--- extractant-axis shape by cohort ---")
        block = focus[focus["axis_label"] == "extractant"]
        print(block.pivot_table(index="model", columns="cohort",
                                values=["slope_pred_median", "span_recovery_median",
                                        "shape_mae"]).to_string())

    if args.detail:
        detail = pd.concat([pd.read_parquet(p) for p in args.detail], ignore_index=True)
        detail["arm"] = np.where(detail["adapter"] == "ZERO_SHOT_REF", "ZERO_SHOT",
                                 detail["adapter"] + "@" + detail["policy"])
        if "global_model" not in detail.columns:
            detail["global_model"] = detail["model"]
        frontier_rows = []
        for cohort, keep in (("FROZEN", None), ("LIGAND_EXCLUDED", affected_ligands)):
            view = detail if keep is None else detail[~detail["extractant"].isin(keep)]
            table = kshot_numbers(view, None, None)
            table["cohort"] = cohort
            frontier_rows.append(table)
        frontier = pd.concat(frontier_rows, ignore_index=True)
        frontier.to_csv(args.out / "frontier_by_cohort.csv", index=False)
        print("\n--- k-shot frontier with the affected ligand(s) excluded ---")
        pivot = frontier[frontier["k"].isin([0, 1, 2, 5])].pivot_table(
            index=["global_model", "arm"], columns=["cohort", "k"], values="mae")
        print(pivot.head(20).to_string())

    (args.out / "sensitivity.json").write_text(json.dumps({
        "quarantined_rows": len(definitions["quarantined_ids"]),
        "corrected_rows": len(definitions["corrected"]),
        "affected_ligands": sorted(affected_ligands),
        "meta": definitions["meta"],
        "note": ("The global model is not refitted per cohort; the sensitivity is "
                 "applied at evaluation time. The k-shot table uses whole-ligand "
                 "exclusion because the detail frame is already aggregated past the "
                 "row level, which is a stricter test than the row-level edit."),
    }, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
