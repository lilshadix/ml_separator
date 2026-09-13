from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "D:/ml_separator_gh/generations/gen13_separation/analysis/stage3/verify_direction")
from v_core import *

P = load(); frame, coef, n_obs, frames = P["frame"], P["coef"], P["n_obs"], P["frames"]
mask = n_obs >= 5
f = frame[mask].reset_index(drop=True)
c0 = coef[mask, 0]
y = (c0 < 0).astype(int)            # 1 = heavy-selective
groups = f["chemotype"].to_numpy()

print(f"cells={len(f)} ext={f['extractant'].nunique()} chem={f['chemotype'].nunique()}")
print(f"heavy fraction (cells) = {y.mean():.4f}")
allcols = pd.concat([frames[b] for b in ("COND","MASSACT","PHYSCHEM","DONORS","ECFP","LIG2D","COORD")], axis=1)
topo = topo_columns(frames)
chem137 = block_columns(frames, CHEM)
lean209 = block_columns(frames, LEAN)
print("n topo", len(topo), "n chem", len(chem137), "n lean", len(lean209))

Xt = allcols.loc[mask, topo].to_numpy(float)
Xc = allcols.loc[mask, chem137].to_numpy(float)
Xl = allcols.loc[mask, lean209].to_numpy(float)

res = []
tabs = {}
for kind in ("logit", "rf"):
    for name, X in (("topo39", Xt), ("chem137", Xc), ("lean209", Xl)):
        t = run(f, y, X, groups, kind=kind, design="BP")
        tabs[(kind, name)] = t
        pe = per_extractant_acc(t)
        pb = per_extractant_acc(baseline_table(t, 1))
        bs = blocked_bootstrap(pe, pb)
        res.append({"model": kind, "features": name, "n_cols": X.shape[1],
                    "macro_acc": pe["acc"].mean(), "baseline_heavy": pb["acc"].mean(),
                    **bs, "pooled_acc": float((t["pred"] == t["y"]).mean())})
        print(res[-1])
R = pd.DataFrame(res)
R.to_csv(OUT / "v_headline.csv", index=False)
import pickle
pickle.dump(tabs, open(OUT / "v_headline_tabs.pkl", "wb"))
print(R.to_string())
