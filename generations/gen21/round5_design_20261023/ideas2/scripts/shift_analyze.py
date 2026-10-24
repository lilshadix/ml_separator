#!/usr/bin/env python
"""shift theme, step 4 (EXPLORATORY: 229 already-seen C3 cells, 5 dev seeds): tables for the condition-shift arms
built by shift_build.py.  R2 = round-4 convention (pooled within seed, seed mean), identical rows in every
comparison.  Inference unit = publication x chemotype (exact sign-flip, one-sided), LOPO jackknife over publications.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_analyze.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as SS

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 300)
SUMMARY = {}


def r2s(d, arm):
    out = []
    for _, g in d.groupby("split_seed"):
        y = g.y.to_numpy(float)
        out.append(1 - ((y - g[arm].to_numpy(float)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.array(out)


def r2(d, arm):
    return float(r2s(d, arm).mean())


def relsse(d, cand, ref):
    out = []
    for _, g in d.groupby("split_seed"):
        out.append(1 - ((g.y - g[cand]) ** 2).sum() / ((g.y - g[ref]) ** 2).sum())
    return float(np.mean(out))


def describe(d):
    return (f"cells {d.cell_id.nunique()}, pubs {d.publication.nunique()}, units "
            f"{d[['publication', 'chemotype']].drop_duplicates().shape[0]}, rows/seed {len(d) / d.split_seed.nunique():.0f}")


def unit_c(d, cand, ref, unit=("publication", "chemotype")):
    sst = {s: float(((g.y - g.y.mean()) ** 2).sum()) for s, g in d.groupby("split_seed")}
    ns = d.split_seed.nunique()
    e = d[list(unit) + ["split_seed"]].copy()
    e["ec"] = (d.y - d[cand]) ** 2
    e["er"] = (d.y - d[ref]) ** 2
    e["c"] = (e.er - e.ec) / e.split_seed.map(sst) / ns
    g = e.groupby(list(unit)).agg(c=("c", "sum"), sc=("ec", "sum"), sr=("er", "sum"))
    g["rho"] = 1 - g.sc / g.sr
    return g.reset_index()


def signflip(c):
    c = np.asarray(c, float)
    G = len(c)
    S = np.array(list(itertools.product((1.0, -1.0), repeat=G))) if G <= 20 else \
        np.random.default_rng(1).choice((1.0, -1.0), size=(200000, G))
    return float(((S @ c) >= c.sum() - 1e-15).mean())


def lopo(d, cand, ref):
    pt = r2(d, cand) - r2(d, ref)
    pubs = sorted(d.publication.unique())
    if len(pubs) < 3:
        return pt, np.nan, np.nan, np.nan, ""
    jk = np.array([r2(d[d.publication != p], cand) - r2(d[d.publication != p], ref) for p in pubs])
    P = len(jk)
    se = float(np.sqrt((P - 1) / P * ((jk - jk.mean()) ** 2).sum()))
    q = float(SS.t.ppf(0.975, P - 1))
    return pt, pt - q * se, pt + q * se, float(jk.min()), pubs[int(np.argmin(jk))]


def infer(d, cand, ref, label):
    u = unit_c(d, cand, ref)
    pt, lo, hi, mn, mnp = lopo(d, cand, ref)
    pu = unit_c(d, cand, ref, unit=("publication",))
    res = {"label": label, "cand": cand, "ref": ref, "dR2": pt, "relSSE": relsse(d, cand, ref),
           "seeds_pos": int((r2s(d, cand) - r2s(d, ref) > 0).sum()),
           "units_pos": int((u.c > 1e-6).sum()), "units_neg": int((u.c < -1e-6).sum()), "G": len(u),
           "p_unit": signflip(u.c.to_numpy()), "lopo_lo": lo, "lopo_hi": hi, "lopo_min": mn, "lopo_min_pub": mnp,
           "eqw_pub_relSSE": float(pu.rho.mean()), "pubs_pos": int((pu.c > 0).sum()), "P": len(pu)}
    print(f"  [{label}] {cand} - {ref}: dR2 {pt:+.4f} relSSE {res['relSSE']:+.3f} seeds+ {res['seeds_pos']}/5; units "
          f"{res['units_pos']}+/{res['units_neg']}- of {res['G']}, sign-flip p {res['p_unit']:.3f}; LOPO 95% "
          f"[{lo:+.4f}, {hi:+.4f}] min {mn:+.4f} (w/o {mnp}); eq-w pub relSSE {res['eqw_pub_relSSE']:+.3f} "
          f"({res['pubs_pos']}/{res['P']} pubs +)")
    return res


def ladder(d, fams, label):
    rows = []
    for nm in fams:
        rr = {"arm": nm}
        for k, lab in ((0, "0r"), (1, "2r"), (2, "3r"), (3, "4r")):
            col = f"{nm}_{k}"
            if nm == "COPY" and k == 0:
                col = "COPY"
            if col in d and d[col].notna().all():
                rr[lab] = r2(d, col)
            sh = f"{nm}SH_{k}"
            if sh in d and d[sh].notna().all():
                rr[lab + "_SH"] = r2(d, sh)
        rows.append(rr)
    T = pd.DataFrame(rows).set_index("arm")
    print(f"\n[ladder] {label}: {describe(d)}")
    print(T.round(4).to_string())
    return T


def main():
    T = pd.read_parquet(OUT / "shift_rows.parquet")
    for design in ("BPG", "BP"):
        t = T[T.design == design]
        print("\n" + "=" * 100 + f"\nDESIGN {design}\n" + "=" * 100)
        S = {}
        S["PRIMARY(P1)"] = t[(t.n_clean >= 2) & ~t.c5]
        S["A2"] = t[(t.n_AS >= 1) & (t.reg_ok == True) & ~t.c5]          # noqa: E712
        S["A2_withC5"] = t[(t.n_AS >= 1) & (t.reg_ok == True)]           # noqa: E712
        S["A1"] = t[(t.n_AS >= 1) & ~t.c5]
        S["A1_only1x"] = t[(t.n_AS >= 1) & (t.reg_ok != True) & ~t.c5]  # noqa: E712
        S["ACLS"] = t[(t.cls == "acid_conc_only") & (t.n_clean >= 2) & ~t.c5]
        S["C2"] = t[(t.n_CS >= 1) & (t.creg_ok == True) & ~t.c5]         # noqa: E712
        S["C2_withC5"] = t[(t.n_CS >= 1) & (t.creg_ok == True)]          # noqa: E712
        S["C2_nonacid"] = S["C2"][S["C2"].cs_cols != "acid"]
        S["C2_withC5_nonacid"] = S["C2_withC5"][S["C2_withC5"].cs_cols != "acid"]
        for k, d in S.items():
            print(f"{k}: {describe(d)}")
        P = S["PRIMARY(P1)"]
        print(f"\nP1 reproduction (PRIMARY): B1 {r2(P, 'B1_3'):.4f} SH {r2(P, 'SH_3'):.4f} OFF {r2(P, 'OFF'):.4f} "
              f"SIB {r2(P, 'SIB_3'):.4f} COPY {r2(P, 'COPY'):.4f}")
        A = S["A2"]
        print("\nA2 cells by publication / class / extrapolation:")
        print(A.drop_duplicates("cell_id").groupby(["publication", "chemotype", "cls"]).agg(
            cells=("cell_id", "size"), n_AS=("n_AS", "mean"), extrap=("extrap", "mean"), acid=("acid", "first")).to_string())

        base = ["B1", "B0", "SIB", "SH", "COPY", "ACOPY", "AREG", "AREGH", "AINT", "AINTX", "TILT"]
        for nm in ("A2", "A2_withC5", "ACLS", "A1", "A1_only1x"):
            d = S[nm]
            if len(d):
                ladder(d, base, nm)
                print(f"   OFF {r2(d, 'OFF'):.4f}; B0@4 {r2(d, 'B0_3'):.4f}")
        cb = ["B1", "SIB", "SH", "COPY", "CCOPY", "CREG", "CREGH"]
        for nm in ("C2", "C2_withC5", "C2_nonacid", "C2_withC5_nonacid"):
            d = S[nm]
            if len(d):
                ladder(d, cb, nm)
                print("   cs_cols:", d.drop_duplicates("cell_id").groupby(["publication", "cs_cols"]).size().to_dict())
        # interpolation vs extrapolation inside A2
        for ex in (False, True):
            d = A[A.extrap == ex]
            if len(d):
                ladder(d, ["B1", "SH", "ACOPY", "AREG", "AREGH", "AINT", "AINTX", "TILT"], f"A2 extrap={ex}")
        # per publication inside A2 (4 reads raw + 0 reads)
        rows = []
        for p, d in A.groupby("publication"):
            rows.append({"pub": p, "cells": d.cell_id.nunique(), "rows/seed": len(d) / 5, "B1@4": r2(d, "B1_3"),
                         "SH@4": r2(d, "SH_3"), "ACOPY@4": r2(d, "ACOPY_3"), "AREG@4": r2(d, "AREG_3"),
                         "AINT@4": r2(d, "AINT_3"), "TILT@4": r2(d, "TILT_3"), "ACOPY@0": r2(d, "ACOPY_0"),
                         "AREG@0": r2(d, "AREG_0"), "AINT@0": r2(d, "AINT_0"), "TILT@0": r2(d, "TILT_0"),
                         "relSSE AREG vs ACOPY @0": relsse(d, "AREG_0", "ACOPY_0"),
                         "relSSE AREG vs ACOPY @4": relsse(d, "AREG_3", "ACOPY_3")})
        print("\n[per publication, A2]")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        # pair-type decomposition of AREG vs ACOPY at 0 reads and 4 reads
        e = A[["pair"]].copy()
        for c in ("ACOPY_0", "AREG_0", "ACOPY_3", "AREG_3", "B1_3", "SH_3", "AREGSH_3", "TILT_0"):
            e[c] = (A.y - A[c]) ** 2
        g = e.groupby("pair").sum() / 5
        g["gain0_AREG_vs_ACOPY"] = g.ACOPY_0 - g.AREG_0
        g["gain0_TILT_vs_ACOPY"] = g.ACOPY_0 - g.TILT_0
        g["gain4_AREG_vs_ACOPY"] = g.ACOPY_3 - g.AREG_3
        order = ["La/Ce", "Ce/Pr", "Pr/Nd", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu"]
        print("\n[pair decomposition, A2, SSE per seed]")
        print(g.reindex([o for o in order if o in g.index]).round(3).to_string())

        print("\n[inference]")
        res = []
        for nm in ("A2", "A2_withC5", "ACLS", "A1"):
            d = S[nm]
            if d.publication.nunique() < 2:
                continue
            for cand, ref in (("AREG_0", "ACOPY_0"), ("AINT_0", "ACOPY_0"), ("TILT_0", "ACOPY_0"), ("AREGH_0", "ACOPY_0"),
                              ("AREG_3", "ACOPY_3"), ("AINT_3", "ACOPY_3"), ("TILT_3", "ACOPY_3"),
                              ("AREGSH_3", "ACOPYSH_3"), ("AREGSH_3", "SH_3"), ("AINTSH_3", "SH_3"), ("TILTSH_3", "SH_3"),
                              ("AREGSH_3", "OFF"), ("AREGSH_3", "B1_3"), ("AREG_0", "B1_3"), ("AINT_0", "B1_3"),
                              ("AREGSH_1", "B1_3"), ("AREGSH_1", "SH_1"), ("AREG_1", "SIB_1"), ("COPY", "B1_3")):
                if cand in d and ref in d and d[cand].notna().all() and d[ref].notna().all():
                    res.append(infer(d, cand, ref, nm))
        for nm in ("C2", "C2_withC5", "C2_nonacid", "C2_withC5_nonacid"):
            d = S[nm]
            if d.publication.nunique() < 1 or not len(d):
                continue
            for cand, ref in (("CREG_0", "CCOPY_0"), ("CREG_3", "CCOPY_3"), ("CREGSH_3", "SH_3"), ("CREGH_0", "CCOPY_0"),
                              ("CREG_0", "B1_3")):
                res.append(infer(d, cand, ref, nm))
        SUMMARY[design] = res
    (OUT / "shift_summary.json").write_text(json.dumps(SUMMARY, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
