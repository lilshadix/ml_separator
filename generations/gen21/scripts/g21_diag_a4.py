#!/usr/bin/env python
"""Diagnosis after attempt 4 (HBK@ML vs HB2+GP, DOPT_Q, k = 1..3, C3, BP): error by Z-distance of
the query to the nearest read metal, and how the posterior query variance under HBK@ML relates to
the realised squared error (is the model's uncertainty a usable guide for choosing reads?).
Output: results/diag/diag_a4.txt"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from g21 import attempts as AT, cache as C, evaluate as E, paths  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import Z_VEC  # noqa: E402

bench = D.load_bench()
spec = AT.get("4", bench)
t = pd.read_parquet(paths.PREDICTIONS / "attempt4_BP.parquet")
t = t[(t.k_max >= 3) & (t.k.isin([1, 2, 3]))]
rows = []
for (seed, fold), g in t.groupby(["split_seed", "fold"]):
    rec = C.load("BP", int(seed), int(fold))
    f = spec.fit(bench, rec)
    mdl = f[f"HBK[{f['ml_kernel']}]"]
    Cf = mdl.cov_full(mdl.S, mdl.tau2, mdl.rho, mdl.sig2)
    for cell in rec["outer"]["cells"]:
        if cell["k_max"] < 3:
            continue
        for k in (1, 2, 3):
            o = E.observe(bench.Y[cell["ci"]], cell, cell["seqs"]["DOPT_Q"], k)
            read = list(o.read)
            Dm = np.zeros((len(read) - 1, 14))
            for i, x in enumerate(read[1:]):
                Dm[i, read[0]], Dm[i, x] = 1, -1
            S = Dm @ Cf @ Dm.T
            for a, b in cell["query"]:
                d = np.zeros(14); d[a], d[b] = 1, -1
                pv = d @ Cf @ d - d @ Cf @ Dm.T @ np.linalg.solve(S, Dm @ Cf @ d)
                dmin = int(min(min(abs(Z_VEC[a] - Z_VEC[r]), abs(Z_VEC[b] - Z_VEC[r])) for r in read))
                rows.append({"split_seed": seed, "cell_id": cell["cell_id"], "k": k, "A": D.LANTHANIDES[a], "B": D.LANTHANIDES[b],
                             "post_var": float(pv), "dmin": dmin})
t = t.merge(pd.DataFrame(rows), on=["split_seed", "cell_id", "k", "A", "B"])
out = io.StringIO()
with redirect_stdout(out):
    for arm in ("HB2+GP", "HBK@ML"):
        t["se_" + arm] = (t[arm] - t.y) ** 2
    print("k=3, MSE by Z-distance to the nearest read metal")
    print(t[t.k == 3].groupby("dmin")[["se_HB2+GP", "se_HBK@ML"]].agg(["size", "mean"]).round(4).to_string())
    for k in (1, 2, 3):
        s = t[t.k == k]
        q = pd.qcut(s.post_var, 5, duplicates="drop")
        print(f"\nk={k}: realised MSE of HBK@ML by quintile of its posterior query variance (calibration)")
        print(s.groupby(q, observed=True).agg(n=("post_var", "size"), mean_post_var=("post_var", "mean"),
                                              mse=("se_HBK@ML", "mean")).round(4).to_string())
        print(f"   total posterior variance {s.post_var.sum():.1f} vs realised SSE {s['se_HBK@ML'].sum():.1f}; "
              f"Spearman(post_var, se) {s[['post_var', 'se_HBK@ML']].corr('spearman').iloc[0, 1]:.3f}")
(paths.RESULTS / "diag" / "diag_a4.txt").write_text(out.getvalue(), encoding="utf-8")
print(out.getvalue())
