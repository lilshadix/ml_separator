"""shift theme, step 0: peek at the bench condition metadata (exploratory, no labels used beyond coverage)."""
import sys
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np, pandas as pd
from r4 import paths
from gen20k import data as D
from gen20k.data import LANTHANIDES
b = D.load_bench()
f = b.frame.reset_index(drop=True)
print(type(b), b.Y.shape, LANTHANIDES)
print(f.columns.tolist())
cc = [c for c in f.columns if c.startswith("cond__")]
print(f[cc].describe(include="all").T.to_string())
ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
print(ref.columns.tolist(), ref.shape)
print(ref.head(3).T)
