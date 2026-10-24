#!/usr/bin/env python
"""SKEPTIC CHECK of probe P3 -- extras on P3SK_out/rows.parquet (EXPLORATORY, already-seen data):
k_max strata at k = 4, full LOCO range, and the paired-seed / cluster spread of the policy contrasts."""
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
T = pd.read_parquet(OUT / "rows.parquet")
cl = M4.add_clusters(T[["chemotype", "publication"]].drop_duplicates(), D.load_bench().frame)
T["cluster"] = T.chemotype.map(dict(zip(cl.chemotype, cl.cluster_cpg)))
CL = sorted(T.cluster.unique())
W = INF._draw_counts(len(CL), 2000, INF.SEED)


def r2(g, col):
    return pd.Series({s: 1 - h[col].sum() / ((h.y - h.y.mean()) ** 2).sum() for s, h in g.groupby("split_seed")})


def boot(g, col):
    d = []
    for s, h in g.groupby("split_seed"):
        a = h.groupby("cluster").agg(n=("y", "size"), sy=("y", "sum"), syy=("y", lambda v: (v ** 2).sum()),
                                     sse=(col, "sum")).reindex(CL).fillna(0.0)
        n, sy, syy, sse = (a[c].to_numpy(float) for c in ("n", "sy", "syy", "sse"))
        d.append(1 - (W @ sse) / (W @ syy - (W @ sy) ** 2 / np.maximum(W @ n, 1e-12)))
    return np.mean(np.vstack(d), 0)


for d in ("BPG", "BP"):
    g = T[T.design == d]
    for km in (4, 5, 6):
        h = g[g.k_max == km]
        print(d, f"k_max={km} @k4 DOPTQ R2 {r2(h, 'se_DOPTQ').mean():.4f} rows/seed {len(h)/5:.0f} clusters {h.cluster.nunique()} "
              f"largest share {h.cluster.value_counts(normalize=True).iloc[0]:.2f}")
    lo = [r2(g[g.cluster != c], "se_DOPTQ").mean() for c in CL]
    print(d, "LOCO k4 DOPTQ C3 range", round(min(lo), 4), round(max(lo), 4))
    for a, b in (("se_DOPTQ", "se_BISECT_R"), ("se_DOPTQ", "se_RANDOM"), ("se_RANDOM", "se_BISECT_R")):
        dl = r2(g, b) - r2(g, a)
        bd = boot(g, b) - boot(g, a)
        se = bd.std()
        print(d, f"{b[3:]} - {a[3:]}: mean {dl.mean():+.4f}, seed sd {dl.std():.4f}, cluster-boot sd {se:.4f}, "
              f"implied 80% MDE (2.8 sd) {2.8 * se:.4f}")
