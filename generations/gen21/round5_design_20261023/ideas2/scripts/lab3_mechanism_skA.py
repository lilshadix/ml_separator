#!/usr/bin/env python
"""Skeptic A for lab-3: independent recompute of the mechanism numbers (cross-publication sharing of per-metal
deviations, overall and level-matched).  EXPLORATORY, descriptive, all labels of the bench are used.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab3_mechanism_skA.py

d[i, m] = log D minus the cell's own LS quadratic in the R coordinate (cells with >= 6 metals, outside V2 drop lists),
corpus-mean per metal removed.  r = sum(d_i d_j) / sum((d_i^2 + d_j^2) / 2) over shared metals and pairs of a class.
Also a publication-cluster bootstrap of the XE / XC / XX r (resampling publications) so that "about zero" has an interval.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402

XR = np.asarray(interp._coords("R"), float)


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    ok = np.array([c not in v2 for c in ids]) & (np.isfinite(Y).sum(1) >= 6)
    idx = np.where(ok)[0]
    dv = np.full((len(idx), Y.shape[1]), np.nan)
    for k, i in enumerate(idx):
        m = np.isfinite(Y[i])
        A = np.vstack([XR[m] ** 2, XR[m], np.ones(m.sum())]).T
        co, *_ = np.linalg.lstsq(A, Y[i, m], rcond=None)
        dv[k, m] = Y[i, m] - A @ co
    dc = dv - np.nanmean(dv, 0)
    Yl = Y[idx]
    pub = f.publication_id.astype(str).to_numpy()[idx]
    ext = f.extractant.astype(str).to_numpy()[idx]
    chem = f.chemotype.astype(str).to_numpy()[idx]
    n = len(idx)
    ii, jj = np.triu_indices(n, 1)
    sp = pub[ii] == pub[jj]
    se = ext[ii] == ext[jj]
    sc = chem[ii] == chem[jj]
    cls = np.where(sp & se, "SS", np.where(sp & sc, "SH", np.where(sp, "SO", np.where(se, "XE", np.where(sc, "XC", "XX")))))
    a, b = dc[ii], dc[jj]
    fin = np.isfinite(a) & np.isfinite(b)
    prod = np.where(fin, a * b, 0.0)
    var = np.where(fin, 0.5 * (a ** 2 + b ** 2), 0.0)
    lm = fin & (np.abs(Yl[ii] - Yl[jj]) <= 1.0)
    prodl = np.where(lm, a * b, 0.0)
    varl = np.where(lm, 0.5 * (a ** 2 + b ** 2), 0.0)
    P = prod.sum(1)
    V = var.sum(1)
    PL = prodl.sum(1)
    VL = varl.sum(1)
    rows = []
    for c in ("SS", "SH", "SO", "XE", "XC", "XX"):
        s = cls == c
        rows.append({"class": c, "pairs": int(s.sum()), "r": P[s].sum() / V[s].sum(), "r_levelmatched": PL[s].sum() / VL[s].sum()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # publication-cluster bootstrap for cross-publication classes (resample publications, keep pairs among the drawn set)
    up = np.unique(pub)
    pi = {p: k for k, p in enumerate(up)}
    pa = np.array([pi[p] for p in pub[ii]])
    pb = np.array([pi[p] for p in pub[jj]])
    rng = np.random.default_rng(7)
    B = 1000
    out = {c: [] for c in ("XE", "XC", "XX")}
    outl = {c: [] for c in ("XE", "XC", "XX")}
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)).astype(float)
        wp = w[pa] * w[pb]
        for c in out:
            s = cls == c
            out[c].append((wp[s] * P[s]).sum() / max((wp[s] * V[s]).sum(), 1e-12))
            outl[c].append((wp[s] * PL[s]).sum() / max((wp[s] * VL[s]).sum(), 1e-12))
    for c in out:
        q = np.nanpercentile(out[c], [2.5, 97.5])
        ql = np.nanpercentile(outl[c], [2.5, 97.5])
        print(f"{c}: publication-bootstrap 95% r [{q[0]:+.3f}, {q[1]:+.3f}]; level-matched [{ql[0]:+.3f}, {ql[1]:+.3f}]")
    # corpus-mean pattern share of deviation variance
    mu = np.nanmean(dv, 0)
    tot = np.nansum(dv ** 2)
    print(f"corpus-mean per-metal pattern share of the summed squared deviation: {np.nansum(np.where(np.isfinite(dv), mu, np.nan) ** 2) / tot:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
