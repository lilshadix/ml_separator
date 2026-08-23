#!/usr/bin/env python
"""Recompute the curve geometry gen9 trains on, and audit it against gen8's.

The follow-up brief (§5) asks for an *independent* re-run of curve reconstruction
rather than a reuse of gen8's parquet, and specifically for the gen8 bug class to
be searched for again: a row with a missing abscissa emitted into a curve, where it
then poisons every slope fit downstream.

This script therefore rebuilds the whole table from the cohort, checks it against
gen8's stored copy row for row, and refuses to write anything if the two disagree —
because a silent disagreement would mean gen9's shape objective and gen8's shape
*metrics* are defined on different curves, and every comparison between them would
be meaningless.

Writes ``runs/gen9_shape/curves/``:

* ``curve_membership.parquet`` — (curve_id, series_id, axis, axis_label, row_id,
  axis_value, n_points), the object the training pairs are drawn from;
* ``curve_table.parquet`` — per-curve measured statistics;
* ``curve_audit.json`` — the checks, including the gen8 comparison.
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

from lanthanide_separation.gen7.harness import load_cohort  # noqa: E402
from lanthanide_separation.gen8.series import (  # noqa: E402
    CURVE_AXES, MIN_CURVE_POINTS, reconstruct,
)
from lanthanide_separation.gen9.curves import AXIS_SETS, DEFAULT_AXES  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen9_shape" / "curves"
GEN8_MEMBERSHIP = REPO_ROOT / "runs" / "gen8_architecture" / "series" / "curve_membership.parquet"


def audit_membership(membership: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Every check the follow-up brief §5 names, as data rather than as prose."""
    report: dict = {}
    report["n_curves"] = int(membership["curve_id"].nunique())
    report["n_memberships"] = int(len(membership))
    report["n_rows_on_a_curve"] = int(membership["row_id"].nunique())
    report["n_cohort_rows"] = int(len(frame))

    # 1. no missing abscissa (the gen8 bug)
    report["n_nonfinite_axis_value"] = int((~np.isfinite(membership["axis_value"])).sum())

    # 2. (row_id, axis) unique — a row may sit on several axes but never twice on one
    duplicated = membership.duplicated(["row_id", "axis"]).sum()
    report["n_duplicate_row_axis"] = int(duplicated)

    # 3. curve membership never leaves its series
    per_curve = membership.groupby("curve_id")["series_id"].nunique()
    report["n_curves_spanning_series"] = int((per_curve > 1).sum())

    # 4. every curve has at least MIN_CURVE_POINTS distinct abscissae
    distinct = membership.groupby("curve_id")["axis_value"].apply(
        lambda s: int(np.unique(np.round(s.to_numpy(dtype=float), 9)).size))
    report["min_distinct_abscissae"] = int(distinct.min())
    report["n_curves_below_min_points"] = int((distinct < MIN_CURVE_POINTS).sum())

    # 5. duplicate coordinates inside a curve (legal, but they must be counted —
    #    they are the pairs gen9 drops)
    sizes = membership.groupby("curve_id").size()
    report["n_curves_with_repeated_abscissa"] = int((sizes > distinct).sum())
    report["n_repeated_abscissa_rows"] = int((sizes - distinct).clip(lower=0).sum())

    # 6. rows belonging to several one-axis curves (legal and expected)
    axes_per_row = membership.groupby("row_id")["axis"].nunique()
    report["rows_by_n_axes"] = {int(k): int(v) for k, v in axes_per_row.value_counts().sort_index().items()}

    # 7. axis inventory
    report["curves_by_axis"] = {
        str(k): int(v) for k, v in
        membership.drop_duplicates("curve_id")["axis_label"].value_counts().sort_index().items()}
    report["memberships_by_axis"] = {
        str(k): int(v) for k, v in membership["axis_label"].value_counts().sort_index().items()}

    # 8. the axes gen9 will actually supervise
    supervised = membership[membership["axis"].isin(DEFAULT_AXES)]
    report["supervised_axes"] = list(DEFAULT_AXES)
    report["n_supervised_curves"] = int(supervised["curve_id"].nunique())
    report["n_supervised_memberships"] = int(len(supervised))
    report["axis_sets"] = {k: list(v) for k, v in AXIS_SETS.items()}

    # 9. every row_id in the membership exists in the cohort
    unknown = set(membership["row_id"]) - set(frame["row_id"].astype(str))
    report["n_membership_rows_not_in_cohort"] = int(len(unknown))
    return report


def compare_to_gen8(membership: pd.DataFrame) -> dict:
    if not GEN8_MEMBERSHIP.exists():
        return {"available": False}
    other = pd.read_parquet(GEN8_MEMBERSHIP)
    key = ["curve_id", "row_id", "axis"]
    a = membership.sort_values(key).reset_index(drop=True)
    b = other.sort_values(key).reset_index(drop=True)
    same_shape = a.shape == b.shape
    identical_keys = same_shape and a[key].equals(b[key])
    max_delta = float("nan")
    if identical_keys:
        max_delta = float(np.nanmax(np.abs(
            a["axis_value"].to_numpy(dtype=float) - b["axis_value"].to_numpy(dtype=float))))
    return {
        "available": True, "gen8_rows": int(len(b)), "gen9_rows": int(len(a)),
        "same_shape": bool(same_shape), "identical_keys": bool(identical_keys),
        "max_abs_axis_value_delta": max_delta,
        "gen8_curves": int(b["curve_id"].nunique()), "gen9_curves": int(a["curve_id"].nunique()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-points", type=int, default=MIN_CURVE_POINTS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--allow-gen8-mismatch", action="store_true",
                        help="write the tables even if they differ from gen8's (records the diff)")
    args = parser.parse_args(argv)

    cohort = load_cohort()
    frame = cohort.frame
    print(f"cohort {frame.shape}  fingerprint {cohort.fingerprint}")

    rec = reconstruct(frame, min_points=args.min_points)
    membership, curves = rec.membership, rec.curves
    print(f"reconstructed {membership['curve_id'].nunique()} curves over "
          f"{membership['row_id'].nunique()} rows")

    report = audit_membership(membership, frame)
    report["cohort_fingerprint"] = cohort.fingerprint
    report["min_points"] = int(args.min_points)
    report["curve_axes"] = list(CURVE_AXES)
    report["vs_gen8"] = compare_to_gen8(membership)

    failures = []
    if report["n_nonfinite_axis_value"]:
        failures.append("membership carries a non-finite abscissa (the gen8 bug class)")
    if report["n_duplicate_row_axis"]:
        failures.append("a row appears twice on one axis")
    if report["n_curves_spanning_series"]:
        failures.append("a curve spans more than one series")
    if report["n_membership_rows_not_in_cohort"]:
        failures.append("membership references a row that is not in the cohort")
    if report["n_curves_below_min_points"]:
        failures.append("a curve has fewer distinct abscissae than min_points")
    gen8 = report["vs_gen8"]
    if gen8.get("available") and not gen8.get("identical_keys"):
        failures.append("gen9's reconstruction differs from gen8's stored membership")
    report["failures"] = failures

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "curve_audit.json").write_text(json.dumps(report, indent=2, default=str))
    for name, value in report.items():
        if name not in ("rows_by_n_axes", "curves_by_axis", "memberships_by_axis", "axis_sets"):
            print(f"  {name}: {value}")

    if failures and not args.allow_gen8_mismatch:
        print("\nFAILED:", *failures, sep="\n  - ")
        return 1

    membership.to_parquet(args.out / "curve_membership.parquet", index=False)
    curves.to_parquet(args.out / "curve_table.parquet", index=False)
    print(f"\nwrote {args.out}/curve_membership.parquet and curve_table.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
