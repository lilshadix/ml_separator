#!/usr/bin/env python
"""Attempt 0 (diagnosis only): where RIDGE2's k = 3 error sits on dev BP, C3."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd  # noqa: E401,E402
from g21 import evaluate as E, paths  # noqa: E402
from gen20k import data as D  # noqa: E402

bench = D.load_bench()
base = E.gen20_baselines(bench)
def anchor_term(block, cell, obs, k):
    """level residual of RIDGE2 at the read metals (offset = mean over reads), 0 elsewhere"""
    v = base["RIDGE2"](block, cell, obs, k)
    r = list(obs.read)
    out = np.zeros(14)
    if r:
        res = obs.y_masked[r] - v[r]
        out[r] = res - res.mean()
    return out
m = {"RIDGE2": base["RIDGE2"], "ANCH_TERM": anchor_term}
t = E.outer_table(bench, "BP", paths.DEV_SEEDS, m, ks=(1, 2, 3))
t = t[(t.k_max >= 3)]
t["e"] = t["RIDGE2"] - t["y"]
t["cls"] = np.where(t.a_read | t.b_read, "one_end_read", "none_read")
for k in (1, 2, 3):
    g = t[t.k == k]
    print(f"k={k}")
    print(g.groupby("cls").agg(n=("e", "size"), mae=("e", lambda v: np.abs(v).mean()), rmse=("e", lambda v: np.sqrt((v**2).mean())), ysd=("y", "std")).to_string())
    one = g[g.cls == "one_end_read"]
    # regress residual target (y - pred) on the anchor term: slope = optimal beta
    x, r = one["ANCH_TERM"].to_numpy(), (one["y"] - one["RIDGE2"]).to_numpy()
    b = (x @ r) / (x @ x)
    print(f"  one-end rows: corr(anchor term, y - RIDGE2) = {np.corrcoef(x, r)[0,1]:.3f}, LS beta = {b:.3f}, "
          f"MAE with beta=1: {np.abs(r - x).mean():.4f} vs {np.abs(r).mean():.4f}")
g = t[t.k == 3]
print(g.assign(ae=g.e.abs()).groupby(["A", "B"], sort=False).agg(n=("e", "size"), share_read=("cls", lambda c: (c == "one_end_read").mean()), mae=("ae", "mean"), ysd=("y", "std")).to_string())
