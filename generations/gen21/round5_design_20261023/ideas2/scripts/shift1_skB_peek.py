"""Skeptic B check for idea shift-1 (EXPLORATORY: 229 already-seen C3 cells). Read-only peek at shift_rows.parquet."""
from pathlib import Path
import pandas as pd
OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out")
T = pd.read_parquet(OUT / "shift_rows.parquet")
print(T.shape); print(list(T.columns))
t = T[(T.design == "BPG") & (T.n_AS >= 1) & (T.split_seed == T.split_seed.min())]
c = t.groupby("cell_id").agg(pub=("publication", "first"), chem=("chemotype", "first"), acid=("acid", "first"), ext=("extractant", "first"),
    x=("x_t", "first"), dxm=("dx_mean", "first"), dxn=("dx_near", "first"), nAS=("n_AS", "first"), ex=("extrap", "first"),
    c5=("c5", "first"), rows=("y", "size"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 40)
print(c.groupby(["pub", "chem", "acid"]).agg(cells=("x", "size"), nAS_med=("nAS", "median"), xmin=("x", "min"), xmax=("x", "max"),
      n_extrap=("ex", "sum"), dxm_max=("dxm", lambda s: s.abs().max()), rows=("rows", "sum")).round(2).to_string())
print(c.chem.value_counts()); print(c.acid.value_counts())
print(t.extractant.nunique())
for p, g in c.groupby("pub"):
    print(p, g.ext.str[:40].unique()[:3])
