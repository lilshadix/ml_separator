"""Skeptic A for shift-1 (EXPLORATORY): which acid-only sibling does each pub_19783c039b target use, and which
cond__ fields separate its two acid series.  Prints only; writes nothing."""
import json
import sys
from pathlib import Path
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np, pandas as pd
from r4 import paths  # noqa
from gen20k import data as D
b = D.load_bench(); f = b.frame.reset_index(drop=True); Y = np.asarray(b.Y, float)
ids = f.cell_id.astype(str).tolist(); pos = {c: i for i, c in enumerate(ids)}
L = json.loads(Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out\shift_build_log.json").read_text())
S = pd.read_parquet(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out\shift_rows.parquet")
cells = sorted(set(S[(S.publication == "pub_19783c039b") & (S.n_AS >= 1)].cell_id), key=lambda c: f.loc[pos[c], "cond__acid_concentration_M"])
cond = [c for c in f.columns if c.startswith("cond__")]
var = [c for c in cond if f.loc[[pos[c_] for c_ in cells], c].astype(str).nunique() > 1]
print("varying cond cols:", var)
print(f.loc[[pos[c] for c in cells], ["cell_id"] + var].to_string(index=False))
print("finite metals per cell:", {c: int(np.isfinite(Y[pos[c]]).sum()) for c in cells})
