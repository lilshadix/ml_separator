"""Step 1b: the two handles on sigma disagree by a factor of four.  Which one is the label noise?

The within-(cell, metal) replicate sd is 0.30-0.76.  The residual of a 14-metal cell about its own
fitted quadratic is 0.13.  Both cannot be per-metal measurement noise: if a single log D were
uncertain by 0.76, a 14-metal cell could not sit within 0.13 of a two-parameter curve.

The resolution has to be that the replicate spread is mostly a *level* shift shared by every metal
of a repeated series -- a different unrecorded phase ratio, a different batch -- and the target is
the *centred* curve, which is invariant to the level.  This script measures that directly: it goes
back to the row-level bundle, splits every cell into its replicate series, and decomposes the
between-replicate variance of log D into a series-level component (removed by centring) and a
metal-within-series component (the part that actually perturbs a and b).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from gen15 import valuebench as V  # noqa: E402
from gen13sep import paths  # noqa: E402
from gen13sep.cohort import apply_quarantine, load_bundle  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def main() -> None:
    bench = V.load()
    f = bench.frame

    bundle, _ = apply_quarantine(load_bundle())
    prov = pd.read_parquet(paths.GEN6_PROVENANCE_PARQUET)[
        ["safe_exp_id", "publication_id", "experiment_series_id"]]
    bundle = bundle.merge(prov, on="safe_exp_id", how="left", validate="one_to_one")
    bundle["safe_exp_id"] = bundle["safe_exp_id"].astype(str)

    # map safe_exp_id -> cell row index, from the frozen cohort itself
    m = {}
    for i, ids in enumerate(f["safe_exp_ids"].astype(str)):
        for s in ids.split(";"):
            m[s] = i
    bundle["cell"] = bundle["safe_exp_id"].map(m)
    b = bundle.dropna(subset=["cell"]).copy()
    b["cell"] = b["cell"].astype(int)
    print(f"rows mapped into the 521 cells: {len(b)} of {len(bundle)}")

    # cells whose rows form more than one replicate series
    rows = []
    for ci, blk in b.groupby("cell"):
        if blk["metal"].duplicated().sum() == 0:
            continue                                   # no repeated (cell, metal)
        # a "series" is the reconstructed experiment series; fall back to the row order
        key = blk["experiment_series_id"].astype(str)
        if key.nunique() < 2:
            # replicates inside one series: index them by their order within the metal
            key = blk.groupby("metal").cumcount().astype(str)
        piv = blk.pivot_table(index=key.rename("series"), columns="metal", values="log_D",
                              aggfunc="mean")
        if piv.shape[0] < 2:
            continue
        full = piv.dropna(axis=1, how="any")            # metals every series measured
        n_ser, n_met = full.shape
        if n_ser < 2 or n_met < 2:
            # can still measure the raw within-(cell,metal) spread, but not the decomposition
            continue
        A = full.to_numpy(dtype=float)
        grand = A.mean()
        ser = A.mean(axis=1) - grand                    # level per series
        met = A.mean(axis=0) - grand                    # the cell's curve
        resid = A - grand - ser[:, None] - met[None, :]  # series x metal interaction
        dof_r = (n_ser - 1) * (n_met - 1)
        rows.append({
            "cell": ci, "n_series": n_ser, "n_metals_common": n_met,
            "sd_total": float(A.std(ddof=0)),
            "sd_series_level": float(np.sqrt((ser ** 2).sum() * n_met / max(n_ser - 1, 1) / n_met))
            if n_ser > 1 else np.nan,
            "sd_curve": float(np.sqrt((met ** 2).sum() / max(n_met - 1, 1))),
            "ss_resid": float((resid ** 2).sum()), "dof_resid": int(dof_r),
            "sd_resid": float(np.sqrt((resid ** 2).sum() / dof_r)) if dof_r > 0 else np.nan,
            "n_metals_cell": int(f.n_metals.iat[ci]),
        })
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "s1b_replicate_decomposition.csv", index=False)
    print(f"\ncells with >=2 replicate series and >=2 shared metals: {len(d)}")
    if len(d):
        print(d[["n_series", "n_metals_common", "sd_total", "sd_series_level", "sd_curve",
                 "sd_resid"]].describe().round(4).to_string())
        pooled = float(np.sqrt(d.ss_resid.sum() / d.dof_resid.sum()))
        # series-level component, dof-pooled
        print("\n--- the number that matters ---")
        print(f"dof-pooled series x metal interaction sd (the part centring does NOT remove): "
              f"{pooled:.4f}   (dof {int(d.dof_resid.sum())})")
        lvl = d.sd_series_level.dropna()
        print(f"median series-level (whole-curve) shift sd: {lvl.median():.4f}   "
              f"-- removed by the row-centring, so it never enters (a, b)")
        print(f"median raw within-(cell,metal) sd over the same cells: "
              f"{d.sd_total.median():.4f}")
        pd.DataFrame([{"pooled_interaction_sd": pooled,
                       "median_series_level_sd": float(lvl.median()),
                       "n_cells": len(d), "dof": int(d.dof_resid.sum())}]).to_csv(
            OUT / "s1b_summary.csv", index=False)


if __name__ == "__main__":
    main()
