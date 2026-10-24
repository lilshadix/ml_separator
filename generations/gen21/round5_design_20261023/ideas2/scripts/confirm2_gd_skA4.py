#!/usr/bin/env python
"""SKEPTIC A for confirm-2: curvature-adjusted chord by stratum and cluster-level one-sample t (EXPLORATORY)."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
OUT = HERE.parent / "confirm2_skA_out"
sys.path.insert(0, str(HERE.parents[3] / "round4_20261022"))
import numpy as np, pandas as pd
from scipy import stats as SS
from r4 import paths
from r3 import interp
from gen20k import data as D
LN = list(D.LANTHANIDES); XR = np.asarray(interp.X_R, float)
iSm, iEu, iGd, iTb, iDy = (LN.index(m) for m in ("Sm", "Eu", "Gd", "Tb", "Dy"))
BETA = -0.0923
b = D.load_bench(); f = b.frame.reset_index(drop=True); Y = np.asarray(b.Y, float)
idx = [iSm, iEu, iTb, iDy]
X = np.vstack([np.ones(4), XR[idx] - XR[iGd], (XR[idx] - XR[iGd]) ** 2]).T
H = np.linalg.pinv(X)
cq = np.full(len(Y), np.nan); ok = np.isfinite(Y[:, [iSm, iEu, iGd, iTb, iDy]]).all(1)
cq[ok] = Y[ok, iGd] - Y[ok][:, idx] @ H[0]
u = pd.read_csv(OUT / "all_units.csv", index_col=0)
d = pd.DataFrame({"unit": f.publication_id.astype(str) + "|" + f.chemotype.astype(str), "cq": cq,
                  "c3": f.cell_id.isin(set(pd.read_parquet(paths.PRED / "REF_BPG.parquet").cell_id))})
d = d[d.c3 & np.isfinite(d.cq) & d.unit.isin(u.index)]
uq = d.groupby("unit").cq.mean().to_frame().join(u[["cls", "cw", "cl"]])
uq["cqw"] = uq.cq.clip(-0.40, 0.20)
print("curvature-adjusted chord (Sm,Eu,Tb,Dy quadratic), winsorised unit means; raw cw on the same units")
print(uq.groupby("cls").agg(units=("cqw", "size"), cq=("cqw", "mean"), cq_sd=("cqw", "std"), raw=("cw", "mean")).round(3).to_string())
hs = u[u.cls == "HS"]
for lev in ("cl",):
    v = hs.groupby(lev).cw.mean().to_numpy()
    t = (v.mean() - BETA) / (v.std(ddof=1) / np.sqrt(len(v)))
    print(f"HS one-sample t vs beta at {lev} level: n={len(v)} mean {v.mean():+.3f} t {t:.2f} p_one_sided {SS.t.cdf(t, len(v)-1):.4f}")
v = hs.cw.to_numpy(); t = (v.mean() - BETA) / (v.std(ddof=1) / np.sqrt(len(v)))
print(f"HS one-sample t vs beta at pubchem level: n={len(v)} t {t:.2f} p {SS.t.cdf(t, len(v)-1):.2e}")
