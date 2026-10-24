#!/usr/bin/env python
"""SKEPTIC CHECK of probe P3 -- metrics on P3SK_out/rows.parquet (written by P3SK_recompute.py).
EXPLORATORY (already-seen C3 cells, dev seeds).  Own pooled-R2 / bootstrap / LOCO / leave-one-publication-out code."""
from __future__ import annotations

import sys
from pathlib import Path

R4 = Path(r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(R4))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r4 import metrics as M4  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import inference as INF  # noqa: E402

OUT = Path(__file__).resolve().parent / "P3SK_out"
C5 = "pub_6a0ef8528c"
T = pd.read_parquet(OUT / "rows.parquet")
bench = D.load_bench()
cl = M4.add_clusters(T[["chemotype", "publication"]].drop_duplicates(), bench.frame)
T["cluster"] = T.chemotype.map(dict(zip(cl.chemotype, cl.cluster_cpg)))
CL = sorted(T.cluster.unique())
W = INF._draw_counts(len(CL), 2000, INF.SEED)
POL = ["se_k3", "se_k3_B0", "se_DOPTQ", "se_DOPTQ_b3", "se_DOPTQ_B0", "se_MAXMIN_R", "se_BISECT_R", "se_RANDOM"]
print("rows per design/seed:", T.groupby(["design", "split_seed"]).size().unique(), "clusters", len(CL))
print("C5 rows/seed:", T[T.publication == C5].groupby(["design", "split_seed"]).size().unique(),
      "cells:", T[T.publication == C5].cell_id.nunique(), "C4 rows/seed:", T[T.k_max >= 4].groupby(["design", "split_seed"]).size().unique())


def r2(g, col):
    return {s: 1 - h[col].sum() / ((h.y - h.y.mean()) ** 2).sum() for s, h in g.groupby("split_seed")}


def boot(g, col):
    d = []
    for s, h in g.groupby("split_seed"):
        a = h.groupby("cluster").agg(n=("y", "size"), sy=("y", "sum"), syy=("y", lambda v: (v ** 2).sum()),
                                     sse=(col, "sum")).reindex(CL).fillna(0.0)
        n, sy, syy, sse = (a[c].to_numpy(float) for c in ("n", "sy", "syy", "sse"))
        d.append(1 - (W @ sse) / (W @ syy - (W @ sy) ** 2 / np.maximum(W @ n, 1e-12)))
    return np.mean(np.vstack(d), 0)


rows = []
for d in ("BPG", "BP"):
    for coh in ("C3", "C4", "C3-noC5", "C4-noC5"):
        g = T[T.design == d]
        if "noC5" in coh:
            g = g[g.publication != C5]
        if coh.startswith("C4"):
            g = g[g.k_max >= 4]
        for col in POL:
            r = pd.Series(r2(g, col))
            bd = boot(g, col) if col in ("se_k3", "se_DOPTQ", "se_RANDOM", "se_BISECT_R") else None
            rows.append({"design": d, "cohort": coh, "arm": col[3:], "R2": r.mean(), "n_ge": int((r >= 0.5).sum()),
                         "per_seed": " ".join(f"{v:.4f}" for v in r), "P_boot": np.nan if bd is None else float((bd >= 0.5).mean())})
S = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(S.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
S.to_csv(OUT / "ladder.csv", index=False)

# leave-one-publication-out and leave-one-cluster-out, k4 DOPTQ B1, C3 population and C4
print("\n=== leave-one-publication-out, k4 DOPTQ B1 (seed-mean pooled R2); sorted ===")
for d in ("BPG", "BP"):
    for coh in ("C3", "C4"):
        g = T[T.design == d]
        if coh == "C4":
            g = g[g.k_max >= 4]
        base = pd.Series(r2(g, "se_DOPTQ")).mean()
        lp = []
        for p, h in g.groupby("publication"):
            rest = g[g.publication != p]
            rr = pd.Series(r2(rest, "se_DOPTQ"))
            sh = h.se_DOPTQ.sum() / g.se_DOPTQ.sum()
            lp.append({"pub": p, "rows_seed": len(h) / 5, "sse_share": sh, "R2_wo": rr.mean(), "n_ge": int((rr >= 0.5).sum())})
        lp = pd.DataFrame(lp).sort_values("R2_wo", ascending=False)
        print(f"-- {d} {coh}: base {base:.4f}; pubs whose removal gives seed-mean >= 0.50: {int((lp.R2_wo >= 0.5).sum())} of {len(lp)}")
        print(lp.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        lc = []
        for c in CL:
            rest = g[g.cluster != c]
            lc.append({"cluster": c, "rows_seed": (g.cluster == c).sum() / 5, "R2_wo": pd.Series(r2(rest, "se_DOPTQ")).mean(),
                       "has_C5": bool((g[g.cluster == c].publication == C5).any())})
        print(pd.DataFrame(lc).sort_values("R2_wo", ascending=False).head(4).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# the C5 cluster minus C5: which cells, and why removing the whole cluster gives less than removing C5 only
g = T[T.design == "BPG"]
c5cl = g[g.publication == C5].cluster.iloc[0]
oth = g[(g.cluster == c5cl) & (g.publication != C5)]
print("\nC5 cluster", c5cl, "other pubs:", oth.publication.unique(), "cells", oth.cell_id.nunique(), "rows/seed", len(oth) / 5)
print("their y mean/sd", round(oth.y.mean(), 3), round(oth.y.std(), 3), "| all-rows y mean/sd", round(g.y.mean(), 3), round(g.y.std(), 3),
      "| their k4 DOPTQ MSE", round(oth.se_DOPTQ.mean(), 4), "all", round(g.se_DOPTQ.mean(), 4))
for nm, rest in (("w/o C5 pub", g[g.publication != C5]), ("w/o C5 cluster", g[g.cluster != c5cl])):
    print(nm, "k3", round(pd.Series(r2(rest, "se_k3")).mean(), 4), "k4 DOPTQ", round(pd.Series(r2(rest, "se_DOPTQ")).mean(), 4))
# C5 pub alone
h = g[g.publication == C5]
print("inside C5 pub: k3 R2", round(pd.Series(r2(h, "se_k3")).mean(), 4), "k4 DOPTQ R2", round(pd.Series(r2(h, "se_DOPTQ")).mean(), 4),
      "SSE share k4", round(h.se_DOPTQ.sum() / g.se_DOPTQ.sum(), 4), "row share", round(len(h) / len(g), 4))

# what the 286 k_max = 3 rows do to the population row
for d in ("BPG", "BP"):
    g = T[T.design == d]
    print(d, "k_max=3 rows R2 (k3, B1):", round(pd.Series(r2(g[g.k_max == 3], "se_k3")).mean(), 4),
          "| C3 pop k4 DOPTQ:", round(pd.Series(r2(g, "se_DOPTQ")).mean(), 4),
          "| C4 k4 DOPTQ:", round(pd.Series(r2(g[g.k_max >= 4], "se_DOPTQ")).mean(), 4))
