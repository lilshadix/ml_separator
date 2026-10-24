"""Skeptic B check for idea shift-1 (EXPLORATORY). Metadata only: what 'acid concentration' means in the A1 publications."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
from r4 import paths  # noqa: puts gen20k on sys.path
from gen20k import data as D
OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out")
T = pd.read_parquet(OUT / "shift_rows.parquet")
cells = T[(T.design == "BPG") & (T.n_AS >= 1)].cell_id.unique()
b = D.load_bench()
print(type(b))
f = b.frame.reset_index(drop=True)
print(f.shape)
idcol = "cell_id" if "cell_id" in f.columns else f.columns[0]
cc = [c for c in f.columns if c.startswith("cond__")]
s = f[f[idcol].isin(cells)]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
for p in ("pub_b0b3296ec5", "pub_c8195b4ca7", "pub_19783c039b", "pub_4cba27f5f3"):
    g = s[s.publication_id == p]
    nz = [c for c in cc if g[c].nunique(dropna=False) > 1 or (g[c].fillna(0) != 0).any()]
    print("\n", p, len(g)); print(g[nz].to_string())
