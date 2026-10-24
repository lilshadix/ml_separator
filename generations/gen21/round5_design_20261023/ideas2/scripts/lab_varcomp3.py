#!/usr/bin/env python
"""E-LAB check C2: is the within-publication shared per-metal pattern carried by a shared AQUEOUS MEDIUM (chemistry of
the aqueous phase, which would also be shared across laboratories using the same medium) or by the publication itself
(laboratory / campaign / analytical procedure)?  EXPLORATORY, descriptive.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_varcomp3.py

Same deviations as lab_varcomp.py (own quadratic in -r, corpus-mean removed, >= 6 metals, V2-1/2/3 out).
Medium match of two cells: same acid one-hot, |dlog10 acid M| <= 0.15 (both recorded), same diluent one-hot.
Pooled r = sum(d_i d_j) / sum((d_i^2 + d_j^2) / 2) over metals and pairs; also publication-weighted (each publication's
within-publication pairs averaged first) with leave-one-publication-out range.
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

pd.set_option("display.width", 250)
X = np.asarray(interp._coords("R"), float)


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    ok = np.array([c not in v2drop for c in ids]) & (np.isfinite(Y).sum(1) >= 6)
    idx = np.where(ok)[0]
    Dv = np.full((len(idx), 14), np.nan)
    for k, i in enumerate(idx):
        m = np.isfinite(Y[i])
        co = np.polyfit(X[m], Y[i, m], 2)
        Dv[k, m] = Y[i, m] - np.polyval(co, X[m])
    Dc = Dv - np.nanmean(Dv, 0)
    Yl = Y[idx]
    g = f.iloc[idx].reset_index(drop=True)
    pub, ext, chem = g.publication_id.astype(str).to_numpy(), g.extractant.astype(str).to_numpy(), g.chemotype.astype(str).to_numpy()
    acid_cols = [c for c in f.columns if c.startswith("cond__acid__")]
    dil_cols = [c for c in f.columns if c.startswith("cond__diluent__")]
    acid = g[acid_cols].fillna(0).to_numpy() > 0
    dil = g[dil_cols].fillna(0).to_numpy() > 0
    ac = g["cond__acid_concentration_M"].to_numpy(float)
    lac = np.log10(np.clip(ac, 1e-4, None))
    n = len(idx)
    I, J = np.triu_indices(n, 1)
    same_acid = (acid[I] == acid[J]).all(1) & acid[I].any(1)
    same_dil = (dil[I] == dil[J]).all(1)
    conc_ok = np.isfinite(ac[I]) & np.isfinite(ac[J]) & (np.abs(lac[I] - lac[J]) <= 0.15)
    med = same_acid & same_dil & conc_ok
    sp, se, sc = pub[I] == pub[J], ext[I] == ext[J], chem[I] == chem[J]
    cls = np.full(len(I), "XX", dtype=object)
    cls[sp & se] = "SS"
    cls[sp & ~se & sc] = "SH"
    cls[sp & ~sc] = "SO"
    cls[~sp & se] = "XE"
    cls[~sp & ~se & sc] = "XC"
    A, B = Dc[I], Dc[J]
    samecov = (np.isfinite(Dc[I]) == np.isfinite(Dc[J])).all(1)
    print(f"pairs with identical metal coverage: {samecov.sum()} of {len(I)}")
    fin = np.isfinite(A) & np.isfinite(B) & samecov[:, None]
    P = np.where(fin, A * B, 0.0)
    V = np.where(fin, 0.5 * (A ** 2 + B ** 2), 0.0)
    inw = (np.abs(Yl[I]) <= 2) & (np.abs(Yl[J]) <= 2)
    near = np.abs(Yl[I] - Yl[J]) <= 1
    pp, vv = P.sum(1), V.sum(1)
    ppn, vvn = np.where(inw & near, P, 0).sum(1), np.where(inw & near, V, 0).sum(1)
    out = []
    for c in ("SS", "SH", "SO", "XE", "XC", "XX"):
        for mm, msk in (("medium-matched", med), ("not matched", ~med)):
            s = (cls == c) & msk & (vv > 0)
            if s.sum() == 0:
                continue
            out.append({"cls": c, "medium": mm, "pairs": int(s.sum()), "pubs": len(set(pub[I][s]) | set(pub[J][s])),
                        "r": pp[s].sum() / vv[s].sum(),
                        "r_in_near": ppn[s].sum() / vvn[s].sum() if vvn[s].sum() > 0 else np.nan})
    print("pooled r by class and aqueous-medium match (r_in_near: only metals where both cells are in [-2, 2] and within"
          " 1 log of each other):")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    # publication-weighted within-publication r with leave-one-publication-out
    print("\npublication-weighted within-publication r (each publication's pairs pooled first):")
    for c in ("SS", "SH", "SO"):
        s = (cls == c) & (vv > 0)
        d = pd.DataFrame({"pub": pub[I][s], "p": pp[s], "v": vv[s]}).groupby("pub").sum()
        d["r"] = d.p / d.v
        lo = [d.r.drop(q).mean() for q in d.index]
        print(f"   {c}: pubs {len(d)}, mean r {d.r.mean():+.3f} (median {d.r.median():+.3f}; positive in {(d.r > 0).sum()} of {len(d)}); "
              f"leave-one-pub-out range [{min(lo):+.3f}, {max(lo):+.3f}]")
        if c == "SO":
            print("      SO per publication:", d.r.round(3).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
