"""shift theme, step 1 (EXPLORATORY): publication x extractant groups and their condition grids (metadata only)."""
import sys
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np, pandas as pd
from r4 import paths
from r3 import datav2 as V2
from gen20k import data as D
from gen20k.data import LANTHANIDES
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 40)
b = D.load_bench()
f = b.frame.reset_index(drop=True)
Y = np.asarray(b.Y, float)
v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
c3 = set(pd.read_parquet(paths.PRED / "REF_BPG.parquet", columns=["cell_id"]).cell_id)
acid_cols = [c for c in f.columns if c.startswith("cond__acid__")]
dil_cols = [c for c in f.columns if c.startswith("cond__diluent__")]
add_cols = [c for c in f.columns if c.startswith("cond__additive__")]
def onehot(row, cols, pre):
    s = [c[len(pre):] for c in cols if row[c] == 1]
    return "+".join(s) if s else "-"
f["acid"] = [onehot(r, acid_cols, "cond__acid__") for _, r in f.iterrows()]
f["dil"] = [onehot(r, dil_cols, "cond__diluent__") for _, r in f.iterrows()]
f["add"] = [onehot(r, add_cols, "cond__additive__") for _, r in f.iterrows()]
f["c3"] = f.cell_id.isin(c3)
f["v2drop"] = f.cell_id.isin(v2drop)
f["nmet"] = np.isfinite(Y).sum(1)
f["meanlogD"] = np.nanmean(Y, 1)
f["ext_s"] = f.extractant_name.astype(str).str[:28]
g = f.groupby(["publication_id", "extractant"])
big = [k for k, v in g if len(v) >= 2 and v.c3.any()]
print("pub x ext groups with >=2 cells and >=1 C3 cell:", len(big))
cols = ["cell_id", "ext_s", "chemotype", "acid", "cond__acid_concentration_M", "dil", "add", "cond__extractant_concentration_M",
        "cond__temperature_C", "cond__contact_time_min", "cond__metal_concentration_mM", "nmet", "meanlogD", "c3", "v2drop"]
for k in big:
    v = g.get_group(k).sort_values(["acid", "dil", "cond__extractant_concentration_M", "cond__acid_concentration_M"])
    print(f"\n### {k[0]}  n={len(v)} c3={v.c3.sum()}")
    print(v[cols].round(3).to_string(index=False))
