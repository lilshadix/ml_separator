"""Reconstruct the experimental series and write the gen8 series audit (brief §1).

Answers, for every experimental series in the frozen cohort: which ligand, which
publication, which metal, **what variable is being swept**, what is held fixed,
what the sweep values are, what ``log D`` did, and how many observations there
are.  Then classifies the series and reports how many rows and ligands are usable
for each curve type — which is the number that decides whether a series-aware
architecture has anything to work with.

Writes ``runs/gen8_architecture/series_audit.md`` and ``series_table.parquet``
(plus the per-curve and per-row membership tables the rest of gen8 reads).
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
from lanthanide_separation.gen8.series import (  # noqa: E402
    AXIS_LABEL, CURVE_AXES, MIN_CURVE_POINTS, reconstruct,
)

COHORT = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
RECOVERED = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "recovered_cells.parquet"
OUT = REPO_ROOT / "runs" / "gen8_architecture"


def attach_publication(frame: pd.DataFrame) -> pd.DataFrame:
    """Join the upstream DOI per row.  Absent file is not an error, only a gap."""
    if not RECOVERED.exists():
        return frame.assign(doi="", data_location="")
    table = pd.read_parquet(RECOVERED)
    keep = [c for c in ("row_id", "nuisance__doi", "nuisance__data_location") if c in table.columns]
    table = table[keep].drop_duplicates("row_id")
    before = len(frame)
    out = frame.merge(table, on="row_id", how="left", validate="one_to_one")
    assert len(out) == before, "publication join changed the row count"
    return out.rename(columns={"nuisance__doi": "doi", "nuisance__data_location": "data_location"})


def sweep_summary(block: pd.DataFrame, axis: str) -> str:
    if axis == "metal":
        return ",".join(sorted(set(block["metal_symbol"].astype(str))))
    values = sorted(set(pd.to_numeric(block[axis], errors="coerce").dropna().round(4)))
    if len(values) > 8:
        return f"{values[0]:g}..{values[-1]:g} ({len(values)} levels)"
    return ",".join(f"{v:g}" for v in values)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--min-points", type=int, default=MIN_CURVE_POINTS)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "series").mkdir(exist_ok=True)

    frame = attach_publication(pd.read_parquet(args.cohort))
    rec = reconstruct(frame, min_points=args.min_points)
    membership, curves, series = rec.membership, rec.curves, rec.series

    # --- enrich the series table with publication + sweep description ------- #
    extra: list[dict] = []
    for series_id, block in frame.groupby("series_id", sort=True):
        dois = sorted({d for d in block.get("doi", pd.Series(dtype=str)).astype(str) if d and d != "nan"})
        locs = sorted({d for d in block.get("data_location", pd.Series(dtype=str)).astype(str)
                       if d and d != "nan"})
        row = {"series_id": series_id, "n_doi": len(dois),
               "doi": dois[0] if len(dois) == 1 else ("|".join(dois[:3]) if dois else ""),
               "n_data_location": len(locs)}
        for axis in CURVE_AXES:
            row[f"sweep__{AXIS_LABEL[axis]}"] = sweep_summary(block, axis)
        extra.append(row)
    series = series.merge(pd.DataFrame(extra), on="series_id", how="left", validate="one_to_one")

    curve_ligand = membership.drop_duplicates("curve_id").set_index("curve_id")["series_id"].map(
        series.set_index("series_id")["extractant"])
    curves = curves.drop(columns=[c for c in ("extractant",) if c in curves.columns])
    curves = curves.assign(extractant=curves["curve_id"].map(curve_ligand))
    curves = curves.assign(tanimoto_cluster=curves["curve_id"].map(
        membership.drop_duplicates("curve_id").set_index("curve_id")["series_id"].map(
            series.set_index("series_id")["tanimoto_cluster"])))

    series.to_parquet(args.out / "series_table.parquet", index=False)
    curves.to_parquet(args.out / "series" / "curve_table.parquet", index=False)
    membership.to_parquet(args.out / "series" / "curve_membership.parquet", index=False)

    # --- usability by curve type -------------------------------------------- #
    per_axis = []
    for axis in CURVE_AXES:
        sub = membership[membership["axis"] == axis]
        cur = curves[curves["axis"] == axis] if not curves.empty else curves
        if sub.empty:
            continue
        per_axis.append({
            "curve_type": AXIS_LABEL[axis],
            "n_curves": int(sub["curve_id"].nunique()),
            "n_rows": int(sub["row_id"].nunique()),
            "n_ligands": int(series.set_index("series_id").loc[sub["series_id"].unique(), "extractant"].nunique()),
            "n_chemotypes": int(series.set_index("series_id").loc[sub["series_id"].unique(), "tanimoto_cluster"].nunique()),
            "median_points": float(sub.drop_duplicates("curve_id")["n_points"].median()),
            "median_linear_r2": float(cur["linear_r2"].median()) if len(cur) else float("nan"),
            "median_slope": float(cur["slope"].median()) if len(cur) else float("nan"),
            "iqr_slope": f"[{cur['slope'].quantile(0.25):.2f}, {cur['slope'].quantile(0.75):.2f}]" if len(cur) else "",
            "median_y_span": float(cur["y_span"].median()) if len(cur) else float("nan"),
            "pct_monotone": float(100 * (cur["monotone_increasing"] | cur["monotone_decreasing"]).mean())
            if len(cur) else float("nan"),
        })
    usability = pd.DataFrame(per_axis).sort_values("n_rows", ascending=False)
    usability.to_csv(args.out / "series" / "usability_by_curve_type.csv", index=False)

    type_counts = series.groupby("series_type").agg(
        n_series=("series_id", "size"), n_rows=("n_rows", "sum"),
        n_ligands=("extractant", "nunique"), n_chemotypes=("tanimoto_cluster", "nunique")).reset_index()
    type_counts = type_counts.sort_values("n_rows", ascending=False)
    type_counts.to_csv(args.out / "series" / "series_type_counts.csv", index=False)

    # --- per-ligand curve inventory ----------------------------------------- #
    lig = membership.merge(series[["series_id", "extractant", "tanimoto_cluster"]],
                           on="series_id", how="left")
    per_ligand = lig.groupby("extractant").agg(
        n_curves=("curve_id", "nunique"), n_curve_rows=("row_id", "nunique"),
        n_axes=("axis", "nunique")).reset_index()
    total_rows = frame.groupby("extractant").size().rename("n_rows")
    per_ligand = per_ligand.merge(total_rows, on="extractant", how="right").fillna(
        {"n_curves": 0, "n_curve_rows": 0, "n_axes": 0})
    per_ligand.to_csv(args.out / "series" / "per_ligand_curves.csv", index=False)

    # --- the markdown audit -------------------------------------------------- #
    n_rows, n_lig = len(frame), frame["extractant"].nunique()
    covered = membership["row_id"].nunique()
    doi_known = int((series["n_doi"] > 0).sum())
    multi_doi = int((series["n_doi"] > 1).sum())
    lines: list[str] = []
    A = lines.append
    A("# gen8 series audit — what the model is actually supposed to represent")
    A("")
    A(f"*Frozen gen6/gen7 cohort: **{n_rows:,} rows / {n_lig} extractants / "
      f"{frame['tanimoto_cluster'].nunique()} Tanimoto chemotypes / {frame['series_id'].nunique()} series**. "
      f"Built by `scripts/gen8_series_audit.py`; tables in `runs/gen8_architecture/series/`.*")
    A("")
    A("## The headline")
    A("")
    A(f"**{covered:,} of {n_rows:,} rows ({100*covered/n_rows:.1f} %) lie on at least one "
      f"reconstructable curve** — a maximal run in which exactly one experimental axis varies "
      f"and every other axis is held fixed, with at least {args.min_points} points on the axis. "
      f"There are **{membership['curve_id'].nunique():,} such curves** across "
      f"{series['extractant'].nunique()} ligands. The response surface gen8 proposes to model "
      "is therefore actually present in the data; it is not an idealisation.")
    A("")
    A("Two facts decide how it must be modelled:")
    A("")
    A(f"* **{int((series['series_type'] == 'grid').sum())} series are grids, not curves.** A paper "
      "reporting a metal series at four acidities produces one `series_id` covering a "
      "2-D surface. Fitting one smooth curve through it — the brief's explicit warning — "
      "would be fitting a curve through a surface. Every grid is therefore *decomposed* "
      "into its constituent one-axis curves rather than discarded: a row at the "
      "intersection of an acid titration and a metal series belongs to both curves, "
      "which is what makes cross-series transfer (§18) a measurable question at all.")
    A(f"* **The mass-action law is visible in the raw slopes.** Extractant-concentration "
      f"curves have median slope "
      f"{usability.set_index('curve_type').loc['extractant','median_slope']:.2f} "
      f"(IQR {usability.set_index('curve_type').loc['extractant','iqr_slope']}) and median "
      f"linear R² {usability.set_index('curve_type').loc['extractant','median_linear_r2']:.3f} "
      f"on the log–log axis; acid curves have median slope "
      f"{usability.set_index('curve_type').loc['acid','median_slope']:.2f} and median R² "
      f"{usability.set_index('curve_type').loc['acid','median_linear_r2']:.3f}. Those are "
      "solvation numbers and nitrate stoichiometries, recovered without being told to look "
      "for them.")
    A("")
    A("## Usable rows and ligands per curve type")
    A("")
    A(md_table(usability))
    A("")
    A("`median_slope` is d(log D)/d(axis), the axis being **log10 concentration** for the "
      "three concentration sweeps (the scale the mass-action law is linear in), degrees "
      "Celsius for temperature, minutes for contact time, and the lanthanide index for the "
      "metal series. `pct_monotone` is the share of curves that never reverse direction.")
    A("")
    A("## Series classification")
    A("")
    A(md_table(type_counts))
    A("")
    A(f"`single_point` and `unusable` together are "
      f"{int(type_counts.set_index('series_type').reindex(['single_point','unusable'])['n_rows'].fillna(0).sum())} "
      "rows — the residue that carries no curve at all and against which no series-aware "
      "method can be scored.")
    A("")
    A("## Provenance")
    A("")
    A(f"* **{doi_known} of {len(series)} series ({100*doi_known/len(series):.0f} %) carry a DOI** "
      f"recovered from the upstream SAFE exports.")
    A(f"* **{multi_doi} series mix more than one publication** "
      f"({100*multi_doi/max(1,len(series)):.1f} %). A series is defined by its categorical "
      "conditions, so two papers that ran the same setup collapse into one series id; this is "
      "recorded rather than corrected, because splitting on DOI would change the gen5/gen6/gen7 "
      "series definition and break comparability.")
    A("")
    A("## Per-ligand inventory")
    A("")
    q = per_ligand["n_curves"].describe()
    A(f"Per ligand: median {q['50%']:.0f} curves, IQR [{per_ligand['n_curves'].quantile(0.25):.0f}, "
      f"{per_ligand['n_curves'].quantile(0.75):.0f}], max {q['max']:.0f}. "
      f"**{int((per_ligand['n_curves'] == 0).sum())} ligands carry no curve at all** and "
      f"{int((per_ligand['n_axes'] >= 2).sum())} carry curves along two or more different axes "
      "— the latter is the population on which cross-series transfer can be measured.")
    A("")
    A("### The twenty largest series")
    A("")
    show = series.sort_values("n_rows", ascending=False).head(20)[
        ["series_id", "n_rows", "series_type", "primary_axis", "varying_axes", "n_metals",
         "n_curves", "log_d_span", "n_doi"]]
    A(md_table(show, float_format=".2f"))
    A("")
    A("## What this licenses, and what it does not")
    A("")
    A("* A **functional** model (one that predicts a curve rather than a row) has "
      f"{membership['curve_id'].nunique():,} training curves — enough to learn a response "
      "surface, not enough to learn one per chemotype.")
    A("* A **smoothness or monotonicity** penalty is defensible on the extractant and acid "
      f"axes (median linear R² {usability.set_index('curve_type').loc['extractant','median_linear_r2']:.3f} "
      f"and {usability.set_index('curve_type').loc['acid','median_linear_r2']:.3f}) and is **not** "
      "defensible globally: contact-time curves have median linear R² "
      f"{usability.set_index('curve_type').loc['contact_time','median_linear_r2']:.3f}, i.e. they are "
      "flat noise once equilibrium is reached, and metal series are non-monotone by "
      "construction wherever the tetrad effect bites.")
    A("* **k-shot calibration has a well-defined candidate pool**: for a held-out ligand the "
      "candidate experiments are its curve points, and the stratification the brief asks for "
      "(§17 — low/middle/high end of a sweep) is computable from `axis_value` alone, without "
      "looking at any target.")
    (args.out / "series_audit.md").write_text("\n".join(lines) + "\n")

    summary = {
        "n_rows": int(n_rows), "n_ligands": int(n_lig),
        "n_series": int(frame["series_id"].nunique()),
        "n_curves": int(membership["curve_id"].nunique()),
        "rows_on_a_curve": int(covered),
        "min_points": int(args.min_points),
        "series_types": type_counts.set_index("series_type")["n_series"].to_dict(),
        "usability": usability.to_dict(orient="records"),
    }
    (args.out / "series" / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 220)
    print(usability.to_string(index=False))
    print()
    print(type_counts.to_string(index=False))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
