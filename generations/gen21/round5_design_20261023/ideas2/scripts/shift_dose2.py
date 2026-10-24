"""shift theme, step 12 (EXPLORATORY): per-publication view of the acid-distance rule and a within-publication
permutation test of the distance gradient (cell-level template MSE at 0 reads vs |dx_mean|), BPG, no C5."""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sa", HERE / "shift_analyze.py"); sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
T = pd.read_parquet(sa.OUT / "shift_rows.parquet")
for c5 in (False, True):
    t = T[(T.design == "BPG") & (T.n_AS >= 1) & (T.c5 == False if not c5 else True)].copy()
    t["near"] = t.dx_mean.abs() <= 0.30 + 1e-9
    rows = []
    for p, g in t.groupby("publication"):
        for nr, q in g.groupby("near"):
            rows.append({"pub": p, "near": nr, "cells": q.cell_id.nunique(), "rows/seed": len(q) / 5,
                         "MSE ACOPY_0": ((q.y - q.ACOPY_0) ** 2).mean(), "MSE B1_3": ((q.y - q.B1_3) ** 2).mean(),
                         "MSE SH_3": ((q.y - q.SH_3) ** 2).mean()})
    print(f"\n[per publication, with C5={c5}]")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    e = t.assign(se=(t.y - t.ACOPY_0) ** 2, se1=(t.y - t.B1_3) ** 2).groupby("cell_id").agg(
        adx=("dx_mean", lambda s: abs(s.iloc[0])), se=("se", "mean"), se1=("se1", "mean"), pub=("publication", "first"))
    e = e.groupby("pub").filter(lambda g: len(g) >= 3 and g.adx.nunique() >= 2)
    def stat(df, col):
        ra = df.groupby("pub").adx.transform(lambda s: rankdata(s) - (len(s) + 1) / 2)
        rs = df.groupby("pub")[col].transform(lambda s: rankdata(s) - (len(s) + 1) / 2)
        return float((ra * rs).sum() / np.sqrt((ra ** 2).sum() * (rs ** 2).sum()))
    for col in ("se", "se1"):
        obs = stat(e, col)
        rng = np.random.default_rng(7)
        null = []
        for _ in range(5000):
            e2 = e.copy()
            e2["adx"] = e2.groupby("pub").adx.transform(lambda s: rng.permutation(s.to_numpy()))
            null.append(stat(e2, col))
        null = np.array(null)
        print(f"within-pub rank corr(|dx|, cell MSE {'ACOPY_0' if col == 'se' else 'B1@4'}) = {obs:.3f}; one-sided perm p {(null >= obs).mean():.4f} "
              f"({e.pub.nunique()} pubs, {len(e)} cells)")
