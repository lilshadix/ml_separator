"""Decile form of the applicability-domain curve, straight off the per-cell probe.

``ad.py`` bands the Tanimoto-to-nearest-training-ligand at the split's own thresholds (0.7 is the
chemotype definition, so the bands are the split's landmarks).  This renders the same quantity in
equal-count deciles, which is the shape a deployment would actually consult, plus the distribution
of the similarity itself.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DESIGNS = ["B", "BR", "BQ", "A", "BP"]

t = pd.read_csv(OUT / "ad_cells.csv")
t = t[t.n_metals >= 5]

qrows = []
for d in DESIGNS:
    s = t.loc[t.design == d, "s_max"]
    qrows.append(dict(design=d, n=len(s), **{f"q{int(q*100)}": float(s.quantile(q))
                                             for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                      frac_below_0p4=float((s < 0.4).mean()), frac_at_1=float((s > 0.999).mean())))
Q = pd.DataFrame(qrows)
Q.to_csv(OUT / "ad_smax_quantiles.csv", index=False)
print("=== Tanimoto to the nearest training ligand: distribution over held-out cells ===")
print(Q.round(4).to_string(index=False))

rows = []
for d in DESIGNS:
    sub = t[t.design == d].copy()
    sub["dec"] = pd.qcut(sub["s_max"].rank(method="first"), 10, labels=False)
    g = sub.groupby("dec").agg(n=("err_amp", "size"), mean_s=("s_max", "mean"),
                               mae_gp=("err_amp", "mean"), mae_g14=("err_g14", "mean"))
    g["delta"] = g.mae_gp - g.mae_g14
    g.insert(0, "design", d)
    rows.append(g.reset_index())
D = pd.concat(rows, ignore_index=True)
D.to_csv(OUT / "ad_deciles.csv", index=False)
print("\n=== amplitude MAE by decile of that similarity ===")
for d in DESIGNS:
    print(f"\n-- {d} --")
    print(D[D.design == d].drop(columns="design").round(4).to_string(index=False))

print("\n=== Spearman(s_max, |a| error), well-determined cells, by design ===")
from scipy.stats import spearmanr
for d in DESIGNS:
    sub = t[t.design == d]
    print(f"  {d}: GP arm {spearmanr(sub.s_max, sub.err_amp).statistic:+.4f}   "
          f"G14 arm {spearmanr(sub.s_max, sub.err_g14).statistic:+.4f}   "
          f"true |a| {spearmanr(sub.s_max, sub.true_mag).statistic:+.4f}")
