#!/usr/bin/env python
"""PROBE P3 -- per-block SSE decomposition of the 5th-read policies (B1), in pooled-R2 units.
EXPLORATORY (already-seen C3 cells, dev seeds).  Reads P3_out/rows.parquet and cells.parquet only.

For each design and seed: SST = sum (y - mean y)^2 over the 1,802 C3 rows; block SSE of each policy (k = 4 population
row; k_max = 3 cells keep k = 3).  Reported: (SSE_DOPTQ_block - SSE_policy_block) / SST, seed mean, and the number of
seeds where it is > 0; so the block entries of one policy add up to its pooled R2 difference from DOPT_Q-nested.
Blocks: light = La/Ce, Ce/Pr, Pr/Nd; mid = Sm/Eu .. Dy/Ho; heavy = Ho/Er .. Yb/Lu.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "P3_out"
KEY = ["design", "split_seed", "fold", "cell_id", "A", "B"]
CELLKEY = ["design", "split_seed", "cell_id"]
C5 = "pub_6a0ef8528c"
BLOCK = {"La/Ce": "light", "Ce/Pr": "light", "Pr/Nd": "light", "Sm/Eu": "mid", "Eu/Gd": "mid", "Gd/Tb": "mid",
         "Tb/Dy": "mid", "Dy/Ho": "mid", "Ho/Er": "heavy", "Er/Tm": "heavy", "Tm/Yb": "heavy", "Yb/Lu": "heavy"}

rows = pd.read_parquet(OUT / "rows.parquet")
cells = pd.read_parquet(OUT / "cells.parquet")
rows["se"] = (rows.B1 - rows.y) ** 2
rows["block"] = (rows.A + "/" + rows.B).map(BLOCK)
k3 = rows[rows.k == 3]
k4 = rows[(rows.k == 4) & rows.strict_legal.astype(bool)]


def frame(pol):
    b = k3.set_index(KEY)[["se", "y", "block", "publication"]].copy()
    if pol == "K3":
        return b.reset_index()
    if pol == "RANDOM":
        s = k4.groupby(KEY).se.mean()
    else:
        pk = cells[cells["pick_" + pol] != ""][CELLKEY + ["pick_" + pol]].rename(columns={"pick_" + pol: "cand"})
        s = k4.merge(pk, on=CELLKEY + ["cand"]).set_index(KEY).se
    b.loc[s.index, "se"] = s
    return b.reset_index()


POLS = ["K3", "DOPTQ", "MAXMIN_R", "BISECT_R", "BISECT_RMID", "HULL_BISECT", "RANDOM"]
F = {p: frame(p) for p in POLS}
out = []
for d in ("BPG", "BP"):
    for excl in (False, True):
        for p in POLS:
            for ref in ("DOPTQ", "K3"):
                if p == ref:
                    continue
                vals = {blk: [] for blk in ("light", "mid", "heavy", "all")}
                for s in sorted(k3.split_seed.unique()):
                    a = F[ref][(F[ref].design == d) & (F[ref].split_seed == s)]
                    c = F[p][(F[p].design == d) & (F[p].split_seed == s)]
                    if excl:
                        a, c = a[a.publication != C5], c[c.publication != C5]
                    sst = ((a.y - a.y.mean()) ** 2).sum()
                    for blk in ("light", "mid", "heavy"):
                        vals[blk].append((a[a.block == blk].se.sum() - c[c.block == blk].se.sum()) / sst)
                    vals["all"].append((a.se.sum() - c.se.sum()) / sst)
                rec = {"design": d, "noC5": excl, "policy": p, "vs": ref}
                for blk, v in vals.items():
                    rec[blk] = float(np.mean(v))
                    rec[blk + "_seeds_pos"] = int(sum(x > 0 for x in v))
                out.append(rec)
T = pd.DataFrame(out)
T.to_csv(OUT / "blocks.csv", index=False)
pd.set_option("display.width", 220)
print(T.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
