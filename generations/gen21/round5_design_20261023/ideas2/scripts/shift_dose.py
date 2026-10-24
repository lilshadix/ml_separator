"""shift theme, step 8 (EXPLORATORY): does the acid-only sibling template's error grow with the acid distance?
Bins of |dx_near| (log10 decades to the nearest acid-only sibling) on A1 (no C5) and with C5; BPG."""
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sa", HERE / "shift_analyze.py"); sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
T = pd.read_parquet(sa.OUT / "shift_rows.parquet")
t = T[(T.design == "BPG") & (T.n_AS >= 1)].copy()
t["adx"] = t.dx_near.abs()
t["adxm"] = t.dx_mean.abs()
for nm, d in (("A1 no C5", t[~t.c5]), ("A1 with C5", t)):
    for col in ("adx", "adxm"):
        d = d.assign(bin=pd.cut(d[col], [-0.01, 0.15, 0.3, 0.6, 3.0]))
        rows = []
        for b, g in d.groupby("bin", observed=True):
            rows.append({"bin": str(b), "cells": g.cell_id.nunique(), "pubs": g.publication.nunique(), "rows/seed": len(g) / 5,
                         "var_y": g.y.var(), "MSE ACOPY_0": ((g.y - g.ACOPY_0) ** 2).mean(), "MSE AINT_0": ((g.y - g.AINT_0) ** 2).mean(),
                         "MSE ACOPY_3": ((g.y - g.ACOPY_3) ** 2).mean(), "MSE SH_3": ((g.y - g.SH_3) ** 2).mean(),
                         "MSE B1_3": ((g.y - g.B1_3) ** 2).mean(), "R2 ACOPY_0": sa.r2(g, "ACOPY_0"), "R2 B1_3": sa.r2(g, "B1_3")})
        print(f"\n[{nm}] by {col} (decades of acid)")
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    # within-cell-type: Spearman of per-cell ACOPY_0 MSE with adx, cells
    c = d.groupby("cell_id").agg(adx=("adx", "first"), mse0=("ACOPY_0", lambda s: 0), pub=("publication", "first"))
    e = d.assign(se0=(d.y - d.ACOPY_0) ** 2, se3=(d.y - d.SH_3) ** 2).groupby("cell_id").agg(adx=("adx", "first"), se0=("se0", "mean"), se3=("se3", "mean"), pub=("publication", "first"))
    from scipy.stats import spearmanr
    print(f"{nm}: Spearman(cell |dx_near|, cell MSE ACOPY_0) {spearmanr(e.adx, e.se0)[0]:.3f} (cells {len(e)}); with SH_3 {spearmanr(e.adx, e.se3)[0]:.3f}")
    # within publication (demeaned ranks)
    e["r_adx"] = e.groupby("pub").adx.rank(); e["r_se0"] = e.groupby("pub").se0.rank()
    w = e.groupby("pub").filter(lambda g: len(g) >= 3)
    print(f"   within-publication Spearman (pubs with >= 3 cells, pooled ranks centred): "
          f"{np.corrcoef(w.r_adx - w.groupby('pub').r_adx.transform('mean'), w.r_se0 - w.groupby('pub').r_se0.transform('mean'))[0, 1]:.3f} ({w.pub.nunique()} pubs, {len(w)} cells)")
