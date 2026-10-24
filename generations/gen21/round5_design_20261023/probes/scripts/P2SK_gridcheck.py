"""P2SK: skeptic's read-only checks on P2's grid and an exact dev sign-flip at cluster level (EXPLORATORY).
    .venv/Scripts/python.exe -B generations/gen21/round5_design_20261023/probes/scripts/P2SK_gridcheck.py
Writes nothing."""
import itertools
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "round4_20261022"))
import numpy as np
import pandas as pd
from r4 import metrics as M4

OUT = HERE.parents[1] / "P2_out"
g = pd.concat([pd.read_csv(OUT / f"P2_grid_{d}.csv") for d in ("BPG", "BP")], ignore_index=True)
nul = g[g.world.isin(["SWAP", "SYM"]) & ~g.pool.str.contains("[|]")].copy()
nul["synthetic"] = nul.pool.isin(["B1s2-B0", "B1s50-B0"])
nul["flag"] = nul.cp_lo_025 > 0.035
for t in ("JK_G1", "JK_KISH", "WCR_LOG", "WCR_POOL", "SEALED"):
    s = nul[nul.test == t]
    print(t, s.groupby("synthetic").agg(cells=("rej_025", "size"), mean=("rej_025", "mean"), mx=("rej_025", "max"),
                                         flags=("flag", "sum")).round(4).to_dict("index"))
s = nul[(nul.test == "JK_G1") & ~nul.synthetic]
print("JK_G1 real pools, mean by G", s.groupby("G").rej_025.mean().round(4).to_dict())
print("JK_G1 real pools top", s.sort_values("rej_025").tail(3)[["design", "unit_type", "pool", "world", "tau", "G", "rej_025"]].to_string())
s = nul[(nul.test == "JK_KISH") & ~nul.synthetic]
print("JK_KISH real pools mean by G", s.groupby("G").rej_025.mean().round(4).to_dict(), "max", s.rej_025.max())
print(s.sort_values("rej_025").tail(3)[["design", "unit_type", "pool", "world", "tau", "G", "rej_025"]].to_string())
# kill: tests surviving the M34 size gate (no flag anywhere)
flagged = set(nul[nul.flag].test.unique())
print("tests with >= 1 gating flag:", sorted(flagged))
k = g[(g.pool == "B1-B0") & (g.world == "EMP") & (g.unit_type == "publication") & (g.G == 8)]
k = k.pivot_table(index="test", columns=["design", "tau"], values="rej_025")
print(k.round(3).to_string())
print("max over gate-surviving tests:", k[~k.index.isin(flagged)].max().round(3).to_dict())
# exact dev sign-flip, cluster level (2^13)
for d in ("BPG", "BP"):
    t = pd.read_parquet(HERE.parents[3] / "round4_20261022" / "predictions" / f"REF_{d}.parquet")
    t = M4.add_clusters(t[(t.k_max >= 3) & (t.k == 3)], None)
    a = t.assign(er=(t.B0 - t.y) ** 2, ec=(t.B1 - t.y) ** 2).groupby(["cluster_cpg", "split_seed"])[["er", "ec"]].sum()
    a = a.groupby(level=0).mean()
    L = np.log(a.er / a.ec).to_numpy(); dd = (a.er - a.ec).to_numpy()
    F = np.array(list(itertools.product((1.0, -1.0), repeat=13)))
    for nm, x in (("SF_LOG", L), ("SF_POOL", dd)):
        print(d, "cpg dev exact", nm, ((F @ x) >= x.sum() - 1e-10 * np.abs(x).sum()).mean())
    big = a.er.idxmax()
    print(d, "giant cluster share of rows", round(t.groupby("cluster_cpg").size().max() / len(t), 3),
          "giant L", round(float(np.log(a.er[big] / a.ec[big])), 4))
