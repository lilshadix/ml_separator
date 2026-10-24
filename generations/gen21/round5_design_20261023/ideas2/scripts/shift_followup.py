#!/usr/bin/env python
"""shift theme, step 5 (EXPLORATORY and partly HINDSIGHT: the rules below were written after reading
shift_analyze.py's tables on the same 229 cells).  Reads shift_out/shift_rows.parquet; prints only.

1. AGATE: per-metal log-acid regression only when the target acid lies inside the acid-only siblings' range,
   otherwise ACOPY (variant AGATE2: nearest-acid level template outside = AINT).  Label-free gate (metadata only).
2. Direction of extrapolation for AREG (target above / below the siblings' acid range).
3. Light-end-only cross-publication tilt at 0 reads: TILT applied to La/Ce, Ce/Pr, Pr/Nd only (TILT_L3) and to Ce/Pr
   only (TILT_CE).  At 0 reads TILT - ACOPY on an adjacent pair equals s_p * dx, so these are exact.
4. Two-factor (acid x [L]) slope analysis per publication (C2 with C5) and per C5-BTBP grid position.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_followup.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sa", HERE / "shift_analyze.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)
OUT = sa.OUT
K = 1.78
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def main():
    T = pd.read_parquet(OUT / "shift_rows.parquet")
    for design in ("BPG", "BP"):
        t = T[T.design == design].copy()
        print("\n" + "=" * 90 + f"\nDESIGN {design}\n" + "=" * 90)
        hasA = t.n_AS >= 1
        w = t.n_AS / (t.n_AS + K)
        for k in (0, 1, 2, 3):
            inside = (t.reg_ok == True) & (t.extrap == False)            # noqa: E712
            t.loc[hasA, f"AGATE_{k}"] = np.where(inside[hasA], t.loc[hasA, f"AREG_{k}"], t.loc[hasA, f"ACOPY_{k}"])
            t.loc[hasA, f"AGATE2_{k}"] = np.where(inside[hasA], t.loc[hasA, f"AREG_{k}"], t.loc[hasA, f"AINT_{k}"])
            if k:
                for nm in ("AGATE", "AGATE2"):
                    t.loc[hasA, f"{nm}SH_{k}"] = w[hasA] * t.loc[hasA, f"{nm}_{k}"] + (1 - w[hasA]) * t.loc[hasA, f"B1_{k}"]
        light3 = t.pair.isin(["La/Ce", "Ce/Pr", "Pr/Nd"])
        t.loc[hasA, "TILT_L3_0"] = np.where(light3[hasA], t.loc[hasA, "TILT_0"], t.loc[hasA, "ACOPY_0"])
        t.loc[hasA, "TILT_CE_0"] = np.where(t.pair[hasA] == "Ce/Pr", t.loc[hasA, "TILT_0"], t.loc[hasA, "ACOPY_0"])
        S = {"A2": t[hasA & (t.reg_ok == True) & ~t.c5],                    # noqa: E712
             "A2_withC5": t[hasA & (t.reg_ok == True)],                     # noqa: E712
             "A1": t[hasA & ~t.c5], "A1_withC5": t[hasA]}
        for nm in ("A2", "A2_withC5"):
            d = S[nm]
            sa.ladder(d, ["B1", "SH", "ACOPY", "AREG", "AREGH", "AGATE", "AGATE2"], nm + " (gate rows)")
            for cand, ref in (("AGATE_0", "ACOPY_0"), ("AGATE_3", "ACOPY_3"), ("AGATESH_3", "SH_3"), ("AGATESH_3", "OFF"),
                              ("AGATE2_0", "ACOPY_0"), ("AGATE_0", "B1_3"), ("AGATESH_1", "B1_3"), ("AGATESH_1", "SH_1")):
                sa.infer(d, cand, ref, nm)
        d = S["A2"]
        di = d[d.extrap == False]                                            # noqa: E712
        print(f"\ninterpolation rows only (A2, target acid inside the sibling range): {sa.describe(di)}")
        for cand, ref in (("AREG_0", "ACOPY_0"), ("AREG_0", "B1_3"), ("AREG_3", "ACOPY_3"), ("AREGSH_3", "SH_3")):
            sa.infer(di, cand, ref, "A2 inside")
        # 2. extrapolation direction
        de = d[d.extrap == True].copy()                                      # noqa: E712
        de["dir"] = np.where(de.dx_mean > 0, "above", "below")
        print("\n[AREG extrapolation by direction, A2]")
        for dr, g in de.groupby("dir"):
            print(f"  {dr}: {sa.describe(g)}; ACOPY_0 {sa.r2(g, 'ACOPY_0'):.3f} AREG_0 {sa.r2(g, 'AREG_0'):.3f} "
                  f"AINT_0 {sa.r2(g, 'AINT_0'):.3f} AREGH_0 {sa.r2(g, 'AREGH_0'):.3f}; 4r ACOPY {sa.r2(g, 'ACOPY_3'):.3f} "
                  f"AREG {sa.r2(g, 'AREG_3'):.3f} B1 {sa.r2(g, 'B1_3'):.3f}; pubs {sorted(g.publication.unique())}")
        # 3. light-end tilt
        for nm in ("A1", "A1_withC5"):
            d = S[nm]
            print(f"\n[light-end tilt, 0 reads, {nm}] {sa.describe(d)}: ACOPY_0 {sa.r2(d, 'ACOPY_0'):.4f} TILT_0 "
                  f"{sa.r2(d, 'TILT_0'):.4f} TILT_L3_0 {sa.r2(d, 'TILT_L3_0'):.4f} TILT_CE_0 {sa.r2(d, 'TILT_CE_0'):.4f}")
            for cand in ("TILT_L3_0", "TILT_CE_0"):
                sa.infer(d, cand, "ACOPY_0", nm)
            dl = d[light3.loc[d.index]]
            print(f"   light-pair rows only ({sa.describe(dl)}): SSE/seed ACOPY {((dl.y - dl.ACOPY_0) ** 2).sum() / 5:.3f} "
                  f"TILT {((dl.y - dl.TILT_0) ** 2).sum() / 5:.3f}")
            for p in ("La/Ce", "Ce/Pr", "Pr/Nd"):
                q = d[d.pair == p]
                gain = ((q.y - q.ACOPY_0) ** 2 - (q.y - q.TILT_0) ** 2)
                bypub = gain.groupby(q.publication).sum() / 5
                print(f"   {p}: SSE gain/seed {gain.sum() / 5:+.3f}; pubs + {(bypub > 0).sum()}/{len(bypub)} "
                      f"{bypub.round(3).to_dict()}")
        # 4. two-factor slope analysis
        c = t[(t.n_CS >= 1) & (t.creg_ok == True)]                           # noqa: E712
        rows = []
        for p, g in c.groupby("publication"):
            rows.append({"pub": p, "coords": ",".join(sorted(g.cs_cols.unique())), "cells": g.cell_id.nunique(),
                         "rows/seed": len(g) / 5, "B1@4": sa.r2(g, "B1_3"), "SH@4": sa.r2(g, "SH_3"),
                         "CCOPY@0": sa.r2(g, "CCOPY_0"), "CREG@0": sa.r2(g, "CREG_0"), "CREGH@0": sa.r2(g, "CREGH_0"),
                         "CREG@4": sa.r2(g, "CREG_3"), "CREGSH@4": sa.r2(g, "CREGSH_3"), "CREGHSH@4": sa.r2(g, "CREGHSH_3")})
        print("\n[two-factor / one-factor continuous condition model, per publication]")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        g = c[c.publication == "pub_6a0ef8528c"]
        if len(g):
            # grid position: interior of the 5 x 3 (acid x L) grid vs edge
            xt = g.drop_duplicates("cell_id").set_index("cell_id").x_t
            print("\n[C5-BTBP acid x [L] grid] cells by acid level:", np.round(10 ** xt, 2).value_counts().to_dict())
            g = g.assign(edge=np.isclose(10 ** g.x_t, 0.1) | np.isclose(10 ** g.x_t, 1.0))
            for e, q in g.groupby("edge"):
                print(f"  acid edge={e}: {sa.describe(q)}; CCOPY_0 {sa.r2(q, 'CCOPY_0'):.3f} CREG_0 {sa.r2(q, 'CREG_0'):.3f} "
                      f"B1@4 {sa.r2(q, 'B1_3'):.3f} SH@4 {sa.r2(q, 'SH_3'):.3f} CREGSH@4 {sa.r2(q, 'CREGSH_3'):.3f}")
            q = g[g.cell_id != "c950dc2982e96b83"]
            print(f"  without c950dc: CCOPY_0 {sa.r2(q, 'CCOPY_0'):.3f} CREG_0 {sa.r2(q, 'CREG_0'):.3f} "
                  f"B1@4 {sa.r2(q, 'B1_3'):.3f} SH@4 {sa.r2(q, 'SH_3'):.3f} CREGSH@4 {sa.r2(q, 'CREGSH_3'):.3f}")


if __name__ == "__main__":
    main()
