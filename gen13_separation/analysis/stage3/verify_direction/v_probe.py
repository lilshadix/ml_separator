from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("D:/ml_separator_gh/gen13_separation/analysis/stage3/verify_direction")
P = pickle.load(open(OUT / "bench_cache.pkl", "rb"))
frame, coef, n_obs, frames = P["frame"], P["coef"], P["n_obs"], P["frames"]

LEAN = ("COND", "MASSACT", "PHYSCHEM", "DONORS", "COORD")
lean_cols = [c for b in LEAN for c in frames[b].columns]
print("lean raw cols:", len(lean_cols))
coord_cols = list(frames["COORD"].columns)
topo = [c for c in coord_cols if c.startswith("coord__dist__") or c.startswith("coord__arm__")]
print("coord cols:", len(coord_cols), "topo cols:", len(topo))
print("topo names:", topo)

for thr in (2,4,5,6,8,14):
    m = n_obs >= thr
    sub = frame[m]
    X = pd.concat([frames[b][m] for b in LEAN], axis=1)
    nconst = int((X.nunique(dropna=True) <= 1).sum())
    allnan = int(X.isna().all().sum())
    print(f"thr>={thr}: cells={m.sum()} ext={sub['extractant'].nunique()} chemo={sub['chemotype'].nunique()} "
          f"pubs={sub['publication_id'].nunique()} lean_nonconst={len(lean_cols)-nconst} allnan={allnan}")
