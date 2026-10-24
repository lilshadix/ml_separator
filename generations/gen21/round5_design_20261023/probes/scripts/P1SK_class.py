#!/usr/bin/env python
"""Skeptic of probe P1, third pass: where the suspected cross-acid copy pair sits in P1's condition-change classes.

EXPLORATORY (229 already-seen C3 cells, 5 dev seeds).  Prints only; writes nothing.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/probes/scripts/P1SK_class.py

For PRIMARY (CLEAN n_sib >= 2, no C5, BPG) cells: P1's cell-level class (union of the CLEAN siblings' differing cond__
families, contact time and metal concentration ignored), and SH - B1 per class with and without the pair
44012938119033a5 / 5169c377777d282b as mutual siblings.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES  # noqa: E402

K = 1.78
C5 = "pub_6a0ef8528c"
XR = interp._coords("R")
XAC = {"44012938119033a5", "5169c377777d282b"}


def fam(c):
    if c.startswith("cond__acid__"):
        return "acid_type"
    if c == "cond__acid_concentration_M":
        return "acid_conc"
    if c.startswith("cond__additive__"):
        return "additive"
    if c.startswith("cond__diluent__"):
        return "diluent"
    return {"cond__contact_time_min": "contact_time", "cond__extractant_concentration_M": "extr_conc",
            "cond__metal_concentration_mM": "metal_conc", "cond__temperature_C": "temperature"}[c]


def cls(U):
    U = set(U) - {"contact_time", "metal_conc"}
    if U == {"acid_conc"}:
        return "acid_conc_only"
    if U and U <= {"acid_conc", "acid_type"}:
        return "acid_type(+conc)"
    if U == {"diluent"}:
        return "diluent_only"
    if U == {"extr_conc"}:
        return "extr_conc_only"
    if U == {"temperature"}:
        return "temperature_only"
    return "mixed"


def r2(d, col):
    out = []
    for _, g in d.groupby("split_seed"):
        y = g.y.to_numpy(float)
        out.append(1.0 - ((y - g[col].to_numpy(float)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).to_numpy()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    cond = [c for c in f.columns if c.startswith("cond__")]
    C = f[cond].to_numpy(float)
    drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    pools, klass = {}, {}
    for t in sorted(set(par.cell_id)):
        i = pos[t]
        mets = [m for m in range(len(LANTHANIDES)) if np.isfinite(Y[i, m])]
        cov = np.isfinite(Y[:, mets]).all(axis=1)
        cov[i] = False
        pl, U = [], set()
        for j in np.flatnonzero((pub == pub[i]) & (ext == ext[i]) & cov):
            fs = set()
            for q, c in enumerate(cond):
                a, b = C[i, q], C[j, q]
                if np.isnan(a) and np.isnan(b):
                    continue
                if np.isnan(a) or np.isnan(b) or not np.isclose(a, b):
                    fs.add(fam(c))
            if t in drop or ids[j] in drop or not (fs - {"contact_time", "metal_conc"}):
                continue
            pl.append(j)
            U |= fs
        pools[t] = pl
        klass[t] = cls(U) if pl else ""
    rows = []
    for r in par.itertuples(index=False):
        t = r.cell_id
        i = pos[t]
        ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
        S_ = [LANTHANIDES.index(m) for m in str(r.read_set).split(",")]
        xs = XR[S_]
        o = np.argsort(xs)
        out = {"split_seed": r.split_seed, "cell_id": t, "publication": r.publication, "y": r.y, "B1": r.B1,
               "n": len(pools[t]), "cls": klass[t]}
        for nm, pool in (("SH", pools[t]), ("SHx", [j for j in pools[t] if not ({t, ids[j]} <= XAC)])):
            n = len(pool)
            v = []
            for j in pool:
                rr = (Y[i, S_] - Y[j, S_])[o]
                v.append((Y[j, ia] + np.interp(XR[ia], xs[o], rr)) - (Y[j, ib] + np.interp(XR[ib], xs[o], rr)))
            out[nm] = (n / (n + K)) * np.mean(v) + (K / (n + K)) * r.B1 if n else r.B1
        rows.append(out)
    T = pd.DataFrame(rows)
    P = T[(T.n >= 2) & (T.publication != C5)]
    print("classes of the XAC cells:", {t: klass[t] for t in XAC})
    tot = (((P.y - P.B1) ** 2) - ((P.y - P.SH) ** 2)).sum() / 5
    for c, d in P.groupby("cls"):
        g = (((d.y - d.B1) ** 2) - ((d.y - d.SH) ** 2)).sum() / 5
        gx = (((d.y - d.B1) ** 2) - ((d.y - d.SHx) ** 2)).sum() / 5
        print(f"  {c}: cells {d.cell_id.nunique()} pubs {d.publication.nunique()} rows/seed {len(d) / 5:.0f}; B1 {r2(d, 'B1'):.4f} "
              f"SH {r2(d, 'SH'):.4f} SHx {r2(d, 'SHx'):.4f}; SSE gain/seed {g:.4f} (share {g / tot:.3f}) -> without XAC link {gx:.4f}")
    a = P[P.cls == "acid_conc_only"]
    print("  acid_conc_only cells per publication:", a.drop_duplicates("cell_id").groupby("publication").size().to_dict())
    for p, d in a.groupby("publication"):
        print(f"    {p}: SSE gain/seed {(((d.y - d.B1) ** 2) - ((d.y - d.SH) ** 2)).sum() / 5:.4f}; without XAC "
              f"{(((d.y - d.B1) ** 2) - ((d.y - d.SHx) ** 2)).sum() / 5:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
