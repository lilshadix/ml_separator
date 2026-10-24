"""Skeptic A for lab-2: inventory of TODGA cells and the non-TODGA targets in TODGA publications. EXPLORATORY; prints only."""
import sys
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np, pandas as pd
from r4 import paths
from r3 import datav2 as V2
from gen20k import data as D
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
b = D.load_bench(); f = b.frame.reset_index(drop=True); Y = np.asarray(b.Y, float)
print(type(b), f.shape, Y.shape)
cond = [c for c in f.columns if c.startswith("cond__")]
print("cond cols:", cond)
nm = f.extractant_name.astype(str)
tp = sorted(set(f.publication_id[nm == "TODGA"]))
print("TODGA pubs:", tp)
v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
c3 = set(par.cell_id)
def condstr(r):
    a = [c.replace("cond__acid__", "") for c in cond if c.startswith("cond__acid__") and r[c] > 0]
    dl = [c.replace("cond__diluent__", "") for c in cond if c.startswith("cond__diluent__") and r[c] > 0]
    return f"{'/'.join(a)} {r.get('cond__acid_concentration_M', np.nan):.3g}M {'/'.join(dl)}"
for p in tp:
    g = f[f.publication_id == p]
    print(f"\n== {p}: {len(g)} cells")
    for i, r in g.iterrows():
        nmet = np.isfinite(Y[i]).sum()
        print(f"  {r.cell_id[:22]:22s} {str(r.extractant_name)[:14]:14s} {str(r.chemotype)[:10]:10s} n={nmet:2d} C3={r.cell_id in c3!s:5s} v2={r.cell_id in v2!s:5s} {condstr(r)}  Y={np.round(Y[i],2).tolist()}")
