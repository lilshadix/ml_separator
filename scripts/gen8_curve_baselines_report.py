"""Merge the per-split-seed curve-baseline runs into the tables the report quotes.

    .venv/bin/python scripts/gen8_curve_baselines_report.py --pattern 'detail_f*.parquet'

Every table here is macro over ligands (one ligand, one vote) and every interval is a
bootstrap over ligands, because a ligand's rows are not independent evidence and a
per-row interval would be an order of magnitude too tight.  The comparison against
``OFFSET_K1`` is paired on ``(split seed, fold, ligand, repeat, policy, k)`` -- the
identical measured rows and the identical evaluation rows -- so a difference between
two rows of the table is a difference in method and nothing else.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

RUN = REPO_ROOT / "runs" / "gen8_architecture" / "curve_baselines"
PAIR_KEYS = ["split_seed", "fold", "extractant", "repeat", "policy", "k"]
#: Adapters that never read ``observed``; their value at any k *is* their k = 0 value.
SHAPE_ONLY = ("CURVE_SPLINE_SHAPEONLY", "MASSACTION_SHAPEONLY")
#: Where each adapter's k = 0 number comes from.  The driver's loop starts at k = 1,
#: so a zero-measurement value has to be read off an adapter that ignores
#: ``observed``: every ``CURVE_SPLINE_*`` variant returns the spline-corrected
#: surface when handed no measurements, and every ``MASSACTION_*`` variant returns
#: the mass-action one.  ``NO_MODEL`` and ``NEAREST_OBSERVED`` have no k = 0 at all —
#: with nothing measured they have nothing to say — and are left blank rather than
#: silently credited with the frozen model's score.
ZERO_SOURCE = {
    "ZERO_SHOT": "ZERO_SHOT_REF", "OFFSET_K1": "ZERO_SHOT_REF", "OFFSET_K2": "ZERO_SHOT_REF",
    "OFFSET_K3": "ZERO_SHOT_REF", "CURVE_GP": "ZERO_SHOT_REF",
    "CURVE_SPLINE_SHAPEONLY": "CURVE_SPLINE_SHAPEONLY",
    "CURVE_SPLINE_K1": "CURVE_SPLINE_SHAPEONLY", "CURVE_SPLINE_AMP": "CURVE_SPLINE_SHAPEONLY",
    "CURVE_SPLINE_GP": "CURVE_SPLINE_SHAPEONLY",
    "MASSACTION_SHAPEONLY": "MASSACTION_SHAPEONLY", "MASSACTION_K1": "MASSACTION_SHAPEONLY",
}


def macro(detail: pd.DataFrame, metric: str, keys=("adapter", "k")) -> pd.DataFrame:
    per_ligand = detail.groupby(list(keys) + ["extractant"])[metric].mean().reset_index()
    return per_ligand.groupby(list(keys)).agg(**{
        metric: (metric, "mean"), "n_ligands": ("extractant", "nunique")}).reset_index()


def bootstrap_delta(detail: pd.DataFrame, reference: str, metric: str,
                    *, seed: int = 20260820, draws: int = 4000) -> pd.DataFrame:
    base = detail[detail["adapter"] == reference].set_index(PAIR_KEYS)[metric]
    records = []
    for name, sub in detail.groupby("adapter"):
        if name in (reference, "ZERO_SHOT_REF"):
            continue
        joined = (sub.set_index(PAIR_KEYS)[metric].to_frame("mine")
                  .join(base.rename("reference"), how="inner").dropna().reset_index())
        if joined.empty:
            continue
        joined["delta"] = joined["mine"] - joined["reference"]
        grouped = joined.groupby(["k", "extractant"])["delta"]
        per_ligand = grouped.mean().reset_index()
        # A ligand whose every cell is bit-identical to the reference is a *tie*, not a
        # loss.  Half the held-out ligands carry no titration curve at all, so the
        # curve baselines are literally the reference for them; counting those as
        # losses drove a plain win rate to 0.29 for a method that in fact wins on 57%
        # of the ligands it can move.  Both numbers are reported.
        per_ligand["tied"] = grouped.apply(
            lambda v: bool(np.all(np.abs(v.to_numpy(dtype=float)) < 1e-12))).to_numpy()
        rng = np.random.default_rng(seed)
        for k, group in per_ligand.groupby("k"):
            values = group["delta"].to_numpy(dtype=float)
            untied = values[~group["tied"].to_numpy(dtype=bool)]
            samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
            records.append({"adapter": name, "k": int(k), "n_ligands": len(values),
                            f"delta_{metric}": float(values.mean()),
                            "ci_lo": float(np.percentile(samples, 2.5)),
                            "ci_hi": float(np.percentile(samples, 97.5)),
                            "win_rate": float((values < 0).mean()),
                            "n_tied": int(len(values) - len(untied)),
                            f"delta_{metric}_untied": float(untied.mean()) if len(untied) else np.nan,
                            "win_rate_untied": float((untied < 0).mean()) if len(untied) else np.nan})
    return pd.DataFrame(records)


def headline(detail: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per adapter over k = 0/1/2/3/5, with the k = 0 column filled in.

    The driver's loop starts at k = 1, so k = 0 reaches the table two ways: the frozen
    model's own value from ``ZERO_SHOT_REF``, and -- for a baseline that never reads
    ``observed`` -- the value it was recorded at under any k, which is by construction
    its zero-measurement number.
    """
    table = macro(detail, metric).pivot_table(index="adapter", columns="k", values=metric)
    zero = float(table.loc["ZERO_SHOT_REF", 0]) if "ZERO_SHOT_REF" in table.index else np.nan
    if 0 not in table.columns:
        table[0] = np.nan
    for name in table.index:
        source = ZERO_SOURCE.get(name)
        if source == "ZERO_SHOT_REF":
            table.loc[name, 0] = zero
        elif source in table.index:
            table.loc[name, 0] = table.loc[source, 1]
        else:
            table.loc[name, 0] = np.nan
    table.loc["ZERO_SHOT_REF", 0] = zero
    return table[[0, 1, 2, 3, 5]].sort_values(5)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--pattern", default="detail_f*.parquet")
    parser.add_argument("--policy", default="RANDOM")
    parser.add_argument("--axis-pattern", default="axis_shape_f*.parquet")
    args = parser.parse_args(argv)

    files = sorted(args.run.glob(args.pattern))
    detail = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"{len(files)} split seeds, {len(detail):,} records, "
          f"{detail['extractant'].nunique()} ligands, "
          f"{detail['tanimoto_cluster'].nunique()} chemotypes")
    work = detail[detail["policy"].isin([args.policy, "NONE"])]

    for metric in ("mae", "shape_mae"):
        print(f"\n=== {metric.upper()} -- macro over ligands, policy {args.policy} ===")
        print(headline(work, metric).round(4).to_string())
        table = bootstrap_delta(work, "OFFSET_K1", metric)
        table.to_csv(args.run / f"final_delta_{metric}.csv", index=False)
        print(f"\n--- paired delta vs OFFSET_K1 ({metric}); negative = better ---")
        print(table.sort_values(["k", f"delta_{metric}"]).round(4).to_string(index=False))

    # OFFSET_K1 is the historical reference, but it is *not* the strongest deployable
    # baseline: OFFSET_K3 fits a level plus three condition slopes from the same k
    # measurements and beats OFFSET_K1 by 0.11 at k = 5.  A method that only clears
    # OFFSET_K1 has cleared the weakest member of the offset family, so the K3
    # comparison is reported alongside it rather than left to the reader.
    strong = bootstrap_delta(work, "OFFSET_K3", "mae")
    strong.to_csv(args.run / "final_delta_mae_vs_offset_k3.csv", index=False)
    print("\n--- paired delta vs OFFSET_K3 (mae), the strongest deployable baseline ---")
    print(strong.sort_values(["k", "delta_mae"]).round(4).to_string(index=False))

    counts = work[work["adapter"] == "OFFSET_K1"].groupby("k")["extractant"].nunique()
    print("\nligands scored per k:", counts.to_dict())
    print("\npolicy CENTRAL, MAE:")
    print(headline(detail[detail["policy"].isin(["CENTRAL", "NONE"])], "mae").round(4).to_string())

    axis_files = sorted(args.run.glob(args.axis_pattern))
    if axis_files:
        diagnostic = pd.concat([pd.read_parquet(f) for f in axis_files], ignore_index=True)
        per_ligand = diagnostic.groupby(["axis", "surface", "extractant"]).agg(
            shape_mae=("shape_mae", "mean"), flat=("flat_shape_mae", "mean"),
            rows=("n_rows", "mean")).reset_index()
        summary = per_ligand.groupby(["axis", "surface"]).agg(
            shape_mae=("shape_mae", "mean"), flat=("flat", "mean"),
            n_ligands=("extractant", "nunique"), mean_rows=("rows", "mean")).reset_index()
        print("\n=== within-curve SHAPE MAE by swept axis, k = 0 surfaces (macro) ===")
        print(summary.round(4).to_string(index=False))
        summary.to_csv(args.run / "final_axis_shape.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
