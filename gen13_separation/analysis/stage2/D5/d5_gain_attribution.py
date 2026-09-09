"""D5 part 3 - where the ensemble's gain over the incumbent comes from, and what marks the worst
extractants. Run from repo root."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
OUT = os.path.join(ROOT, "gen13_separation", "analysis", "stage2", "D5")
PRED = os.path.join(ROOT, "gen13_separation", "predictions", "B_primary")


def macro(df, col):
    return float(df.groupby(["split_seed", "extractant"])[col].mean().groupby("split_seed").mean().mean())


def main():
    arms = {"best": "X_ENS_DIRECT+LOWRANK_K2", "inc": "C_DIRECT_ROW", "mc": "B1_MEAN_CURVE",
            "heavier": "B4_HEAVIER_ALWAYS", "sel": "M_SELECTED"}
    base = None
    for tag, arm in arms.items():
        p = pd.read_parquet(os.path.join(PRED, f"{arm}.parquet"))
        p[f"ae_{tag}"] = (p["y"] - p["prediction"]).abs()
        cols = ["split_seed", "fold", "cell_id", "extractant", "chemotype", "n_metals", "A", "B",
                "dZ", "y"] if base is None else ["split_seed", "cell_id", "A", "B"]
        base = p[cols + [f"ae_{tag}"]] if base is None else base.merge(
            p[cols + [f"ae_{tag}"]], on=cols, how="left")
    d = base
    d["absy"] = d["y"].abs()
    coh = pd.read_parquet(os.path.join(ROOT, "gen13_separation", "manifests", "cohort_exact.parquet"))
    ncell = coh.groupby("extractant").agg(n_cells=("cell_id", "size"),
                                          n_pubs=("publication_id", "nunique"))
    d = d.merge(ncell, on="extractant", how="left")
    n_seeds = d.split_seed.nunique()

    print("macro:", {t: round(macro(d, f"ae_{t}"), 4) for t in arms})

    # 1. gain over incumbent, by adjacency / dZ
    d["adj"] = np.where(d.dZ == 1, "dZ=1", np.where(d.dZ <= 3, "dZ=2-3",
                        np.where(d.dZ <= 7, "dZ=4-7", "dZ>=8")))
    rows = []
    for lvl, g in d.groupby("adj"):
        rows.append(dict(level=lvl, n_pairs_per_seed=len(g) / n_seeds,
                         mae_best=macro(g, "ae_best"), mae_inc=macro(g, "ae_inc"),
                         mae_mc=macro(g, "ae_mc"), mae_heavier=macro(g, "ae_heavier"),
                         mean_abs_y=g.absy.mean()))
    t = pd.DataFrame(rows)
    t["gain_best_over_inc"] = t.mae_inc - t.mae_best
    t["gain_best_over_mc"] = t.mae_mc - t.mae_best
    t = t.sort_values("level")
    t.to_csv(os.path.join(OUT, "d5_gain_by_dZ_band.csv"), index=False)
    print("\n--- extractant-macro MAE by dZ band ---")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 2. which extractants supply the ensemble's gain over the incumbent
    pe = (d.groupby(["split_seed", "extractant"])[["ae_best", "ae_inc", "ae_mc", "absy"]].mean()
          .groupby("extractant").mean())
    pe["gain"] = pe.ae_inc - pe.ae_best
    pe = pe.join(ncell)
    pe = pe.sort_values("gain", ascending=False)
    tot_gain = pe["gain"].sum()
    print(f"\ntotal macro gain best-vs-incumbent = {tot_gain/len(pe):.4f} over {len(pe)} extractants")
    print(f"  extractants where the ensemble is better: {(pe.gain>0).sum()}, worse: {(pe.gain<0).sum()}")
    print(f"  top 10 gainers supply {pe.gain.head(10).sum()/tot_gain*100:.0f}% of the summed positive+negative gain")
    print(pe.head(5)[["ae_best", "ae_inc", "gain", "n_cells"]].to_string(float_format=lambda v: f"{v:.3f}"))
    pe.to_csv(os.path.join(OUT, "d5_gain_by_extractant.csv"))

    # 3. what marks the hard extractants
    print("\n--- correlates of per-extractant MAE (Spearman over 90 extractants) ---")
    stat = (d.drop_duplicates(["extractant", "cell_id"]).groupby("extractant")
            .agg(max_n_metals=("n_metals", "max"), mean_n_metals=("n_metals", "mean")))
    pe2 = pe.join(stat)
    pe2["n_pairs_per_seed"] = d[d.split_seed == d.split_seed.iloc[0]].groupby("extractant").size()
    rows = []
    for c in ["absy", "n_cells", "n_pubs", "max_n_metals", "n_pairs_per_seed"]:
        rows.append(dict(covariate=c, spearman_vs_mae=spearmanr(pe2[c], pe2.ae_best).statistic))
    cc = pd.DataFrame(rows)
    cc.to_csv(os.path.join(OUT, "d5_extractant_mae_correlates.csv"), index=False)
    print(cc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  MAE by number of cells the extractant has:")
    pe2["cellbin"] = pd.cut(pe2.n_cells, [0, 1, 2, 5, 1000], labels=["1 cell", "2 cells", "3-5", ">5"])
    b = pe2.groupby("cellbin", observed=True).agg(n_extractants=("ae_best", "size"),
                                                  mae_best=("ae_best", "mean"),
                                                  mean_abs_y=("absy", "mean"))
    b["ratio"] = b.mae_best / b.mean_abs_y
    print(b.to_string(float_format=lambda v: f"{v:.3f}"))
    b.to_csv(os.path.join(OUT, "d5_mae_by_n_cells.csv"))
    w10 = pe2.sort_values("ae_best", ascending=False).head(10)
    print(f"\n  of the 10 worst extractants: {(w10.n_cells==1).sum()} have exactly 1 cell, "
          f"median n_cells={w10.n_cells.median():.0f}, median mean|y|={w10.absy.median():.3f} "
          f"(corpus median mean|y| = {pe2.absy.median():.3f})")


if __name__ == "__main__":
    main()
