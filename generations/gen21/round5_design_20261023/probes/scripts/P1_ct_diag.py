#!/usr/bin/env python
"""Probe P1 (idea M01), diagnostic step: why SH and OFF coincide, and two further offset comparators.

EXPLORATORY (229 already-seen C3 cells, 5 dev seeds).  Reads <out>/P1_rows.parquet and <out>/P1_sibs.csv
(P1_ct_build.py) and the round-4 REF parquets; prints only.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/probes/scripts/P1_ct_diag.py --out <dir>

1. Identity check.  SH = B1 + sum_j (SIB_j - B1) / (n + K) and OFF = B1 + sum_j (y_j - B1_j) / (n_p + K).  When a
   sibling j is a C3 cell read at the same metals as the target in the same fold (same beta), SIB_j - B1 at pair p
   equals sibling j's own B1 residual at p, so SH and OFF coincide row by row unless some sibling is not a C3
   cell, was read at other metals, or sits in another fold.  This step counts, per row, whether every CLEAN sibling
   is an OFF donor with the target's read set and fold, and compares |SH - OFF| on those rows and the others.
2. OFF_PUB: publication x pair-type offsets of B1's OOF residual over every other non-V2-dropped C3 cell of the
   publication (any extractant; leave-target-out; K = 1.78).  OFF_XEXT: the same over the publication's OTHER
   extractants only (a lab / campaign fingerprint with no same-extractant information).  OFF_PUB_V1 on the V1
   n_sib >= 2 no-C5 stratum (every C3 cell of the publication, no V2 rule) reconciles Skeptic A's 0.665-0.673.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402

K_SH = 1.78


def r2(d, arm):
    out = []
    for _, g in d.groupby("split_seed"):
        y = g.y.to_numpy(float)
        out.append(1.0 - ((y - g[arm].to_numpy(float)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    T = pd.read_parquet(out / "P1_rows.parquet")
    sibs = pd.read_csv(out / "P1_sibs.csv")
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    clean = sibs[sibs["clean"]].groupby("target")["sibling"].apply(list).to_dict()
    for design in ("BPG", "BP"):
        par = pd.read_parquet(paths.PRED / f"REF_{design}.parquet")
        meta = {(r.split_seed, r.cell_id): (r.read_set, r.fold)
                for r in par.drop_duplicates(["split_seed", "cell_id"]).itertuples(index=False)}
        ci = par.drop_duplicates("cell_id").set_index("cell_id")
        PUB, EXT = ci.publication.to_dict(), ci.extractant.to_dict()
        res = {(r.split_seed, r.cell_id, r.A, r.B): r.y - r.B1 for r in par[["split_seed", "cell_id", "A", "B", "y", "B1"]].itertuples(index=False)}
        c3 = set(par.cell_id)
        t = T[T.design == design].copy()
        # ---- 1. identity ----
        same = []
        for r in t.itertuples(index=False):
            sl = clean.get(r.cell_id, [])
            ok = bool(sl)
            for s in sl:
                if s not in c3 or (r.split_seed, s) not in meta:
                    ok = False
                    break
                rs, fo = meta[(r.split_seed, s)]
                if rs != r.read_set or fo != r.fold or (r.split_seed, s, r.A, r.B) not in res:
                    ok = False
                    break
            same.append(ok)
        t["geom_same"] = same
        t["absdiff"] = (t.CL_SH_3 - t.CL_OFF).abs()
        P = t[(t.n_clean >= 2) & ~t.c5]
        print(f"\n=== {design}: PRIMARY rows {len(P) / 5:.0f}/seed; every CLEAN sibling a C3 OFF donor with the target's read set "
              f"and fold: {P.geom_same.mean():.3f} of rows; max |SH-OFF| there {P.loc[P.geom_same, 'absdiff'].max():.2e}; "
              f"mean |SH-OFF| elsewhere {P.loc[~P.geom_same, 'absdiff'].mean():.4f}")
        g = P.groupby("publication").agg(rows=("y", "size"), geom_same=("geom_same", "mean"), n_clean=("n_clean", "mean"),
                                         off_np=("CL_OFF_np", "mean"), mean_absdiff=("absdiff", "mean"))
        g["rows"] = g.rows / 5
        print(g.round(4).to_string())
        if (~P.geom_same).any():
            Q = P[~P.geom_same]
            print(f"   rows where SH and OFF can differ: {len(Q) / 5:.0f}/seed, cells {Q.cell_id.nunique()}, pubs "
                  f"{sorted(Q.publication.unique())}; on them SH {r2(Q, 'CL_SH_3'):.4f} OFF {r2(Q, 'CL_OFF'):.4f} B1 {r2(Q, 'B1_3'):.4f}")
            Qc = Q.drop_duplicates("cell_id")
            why = []
            for r in Qc.itertuples(index=False):
                sl = clean.get(r.cell_id, [])
                nonc3 = sum(s not in c3 for s in sl)
                diffrs = sum((s in c3) and meta[(r.split_seed, s)][0] != r.read_set for s in sl)
                why.append((nonc3, diffrs))
            why = np.array(why)
            print(f"   per cell (first seed seen): siblings that are not C3 cells: mean {why[:, 0].mean():.2f}; C3 siblings read at "
                  f"other metals: mean {why[:, 1].mean():.2f}")
        # ---- 2. publication-level offsets ----
        dn = {}
        for x in c3:
            dn[("all", x)] = [c for c in c3 if c != x and c not in v2drop and PUB[c] == PUB[x]]
            dn[("xext", x)] = [c for c in dn[("all", x)] if EXT[c] != EXT[x]]
            dn[("v1", x)] = [c for c in c3 if c != x and PUB[c] == PUB[x]]
        for nm, rule in (("OFF_PUB", "all"), ("OFF_XEXT", "xext")):
            vals = []
            for r in t.itertuples(index=False):
                donors = dn[(rule, r.cell_id)]
                rr = [res[(r.split_seed, c, r.A, r.B)] for c in donors if (r.split_seed, c, r.A, r.B) in res]
                vals.append(r.B1_par + (sum(rr) / (len(rr) + K_SH) if rr else np.nan))
            t[nm] = vals
        P = t[(t.n_clean >= 2) & ~t.c5]
        for nm in ("OFF_PUB", "OFF_XEXT"):
            d = P[P[nm].notna()]
            print(f"   PRIMARY rows with a {nm} donor: {len(d) / len(P):.3f} of rows ({d.cell_id.nunique()} cells, "
                  f"{d.publication.nunique()} pubs): {nm} {r2(d, nm):.4f}; SH {r2(d, 'CL_SH_3'):.4f}; OFF {r2(d, 'CL_OFF'):.4f}; "
                  f"B1 {r2(d, 'B1_3'):.4f}")
        # V1 reconciliation (every C3 cell of the publication, no V2 rule)
        vals = []
        for r in t.itertuples(index=False):
            donors = dn[("v1", r.cell_id)]
            rr = [res[(r.split_seed, c, r.A, r.B)] for c in donors if (r.split_seed, c, r.A, r.B) in res]
            vals.append(r.B1_par + (sum(rr) / (len(rr) + K_SH) if rr else 0.0))
        t["OFF_PUB_V1"] = vals
        d = t[(t.n_v1 >= 2) & ~t.c5]
        print(f"   V1 n_sib>=2 no C5 ({d.cell_id.nunique()} cells, {len(d) / 5:.0f} rows/seed): OFF_PUB_V1 {r2(d, 'OFF_PUB_V1'):.4f}; "
              f"V1 SH {r2(d, 'V1_SH_3'):.4f}; V1 SIB {r2(d, 'V1_SIB_3'):.4f}; B1 {r2(d, 'B1_3'):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
