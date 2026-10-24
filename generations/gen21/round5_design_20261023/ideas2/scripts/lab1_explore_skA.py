#!/usr/bin/env python
"""Skeptic A for idea lab-1 (E-LAB, level-matched campaign calibration MX_LM10).  Step 1: inventory + near-copy scan.

EXPLORATORY: every number comes from the 229 already-seen C3 cells.  Read-only on all inputs; prints only.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab1_explore_skA.py

Checks
  * POPX targets (CLEAN C3 cells with >= 1 other-extractant same-publication donor having >= 6 metals);
  * for every target-donor pair: overlap, offset-removed residual sd of raw log D on shared metals, correlation of
    quadratic-detrended deviations, identical-vector flags (near-copies would inflate any 'shape transfer');
  * names of donors vs target (mixtures containing the target ligand, aliases);
  * whether pub_6a0ef8528c (C5-BTBP) is anywhere in POPX (as target or donor).
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
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 60)
X = np.asarray(interp._coords("R"), float)


def quad_dev(y):
    m = np.isfinite(y)
    out = np.full(N_LN, np.nan)
    if m.sum() < 6:
        return out
    co = np.polyfit(X[m], y[m], 2)
    out[m] = y[m] - np.polyval(co, X[m])
    return out


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    print("frame columns:", [c for c in f.columns if not c.startswith("cond__")][:60])
    print("cond columns:", [c for c in f.columns if c.startswith("cond__")])
    ids = f.cell_id.astype(str).to_numpy()
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    name = f.extractant_name.astype(str).to_numpy()
    chem = f.chemotype.astype(str).to_numpy()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    nmet = np.isfinite(Y).sum(1)
    Dv = np.array([quad_dev(Y[i]) for i in range(len(Y))])
    has = (nmet >= 6) & okd
    ref = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = sorted(set(ref.cell_id))
    pos = {c: i for i, c in enumerate(ids)}
    print(f"bench cells {len(f)}, with >=6 metals and not V2-dropped {has.sum()}, C3 {len(c3)}")
    # C5-BTBP
    for p in ("pub_6a0ef8528c",):
        m = pub == p
        print(f"{p}: bench cells {m.sum()}, extractants {sorted(set(name[m]))[:10]}, C3 cells {sum(pub[pos[c]] == p for c in c3)}")

    rec = []
    for t in c3:
        i = pos[t]
        if t in v2drop:
            continue
        dn = np.where(has & (pub == pub[i]) & (ext != ext[i]) & (np.arange(len(f)) != i))[0]
        for j in dn:
            sh = np.isfinite(Y[i]) & np.isfinite(Y[j])
            dy = Y[i, sh] - Y[j, sh]
            r_sd = float(np.std(dy - dy.mean())) if sh.sum() > 1 else np.nan
            a, b = Dv[i, sh], Dv[j, sh]
            ok = np.isfinite(a) & np.isfinite(b)
            rc = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 3 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else np.nan
            rec.append({"target": t, "donor": ids[j], "pub": pub[i], "t_name": name[i], "d_name": name[j], "t_chem": chem[i],
                        "d_chem": chem[j], "n_sh": int(sh.sum()), "maxabs_dy": float(np.abs(dy).max()) if sh.sum() else np.nan,
                        "offset_sd": r_sd, "r_dev": rc, "same_ext_name": name[i] == name[j]})
    P = pd.DataFrame(rec)
    print(f"\nPOPX target-donor pairs: {len(P)}; targets {P.target.nunique()}; pubs {P.pub.nunique()}")
    print("pubs in POPX:", sorted(P.pub.unique()))
    print("pub_6a0ef8528c in POPX:", "pub_6a0ef8528c" in set(P.pub))
    print("\nnear-copy flags:")
    print("  identical vectors on shared metals (maxabs_dy < 1e-9):", int((P.maxabs_dy < 1e-9).sum()))
    print("  offset-identical (offset_sd < 0.02):", int((P.offset_sd < 0.02).sum()))
    print("  offset_sd < 0.05:", int((P.offset_sd < 0.05).sum()))
    print("  r_dev > 0.95:", int((P.r_dev > 0.95).sum()), " r_dev > 0.9:", int((P.r_dev > 0.9).sum()))
    print("  same extractant_name but different extractant id:", int(P.same_ext_name.sum()))
    print(P[(P.offset_sd < 0.05) | (P.r_dev > 0.9) | P.same_ext_name].sort_values("offset_sd").head(60).to_string(index=False))
    print("\nr_dev distribution by pub (median, n pairs):")
    print(P.groupby("pub").agg(n=("r_dev", "size"), med_r=("r_dev", "median"), max_r=("r_dev", "max"),
                               min_offsd=("offset_sd", "min"), targets=("target", "nunique")).round(3).to_string())
    print("\nnames per POPX publication (targets):")
    for p, g in P.groupby("pub"):
        print(f"  {p}: targets {sorted(set(g.t_name))}  donors-only {sorted(set(g.d_name) - set(g.t_name))}")
    # bit-identical series anywhere among C3 POPX targets (any pair, any extractant)
    tg = sorted(P.target.unique())
    dup = []
    for a in range(len(tg)):
        for b in range(a + 1, len(tg)):
            ya, yb = Y[pos[tg[a]]], Y[pos[tg[b]]]
            sh = np.isfinite(ya) & np.isfinite(yb)
            if sh.sum() >= 4 and np.abs(ya[sh] - yb[sh]).max() < 1e-6:
                dup.append((tg[a], tg[b], name[pos[tg[a]]], name[pos[tg[b]]], int(sh.sum())))
    print(f"\nbit-identical target pairs among POPX targets: {len(dup)}", dup[:20])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
