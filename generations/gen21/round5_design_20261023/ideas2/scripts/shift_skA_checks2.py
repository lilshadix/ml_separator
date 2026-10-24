#!/usr/bin/env python
"""SKEPTIC A for shift-2, follow-up (EXPLORATORY; prints only).
(a) CLEAN siblings of the carrier pub_19783c039b's A2 targets: near-copy status (sd of logD difference over metals).
(b) The FROZEN multi-coordinate AEBG as the test design words it, applied to the corpus' continuous-condition series:
    per-metal OLS on the varying coordinates; James-Stein lambda from the siblings' residuals only (lambda = 0 when
    dof = n - p - 1 < 1); lambda = 0 when the target is outside the siblings' per-coordinate range.  The idea does not
    say how lambda generalises to p > 1, so two label-free versions: TR (trace of slope covariance vs mean squared
    slope-vector deviation) and DIR (along the target's displacement dz).  Also a convex-hull gate variant (HULL) for
    factorial grids, where a removed corner is inside the per-coordinate box but outside the hull.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_skA_checks2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "shift_skA_out"
pd.set_option("display.width", 250)
TRIV = {"ctime", "mconc"}


def family(c):
    if c == "cond__acid_concentration_M":
        return "aconc"
    for pre, nm in (("cond__acid__", "atype"), ("cond__additive__", "add"), ("cond__diluent__", "dil")):
        if c.startswith(pre):
            return nm
    return {"cond__contact_time_min": "ctime", "cond__extractant_concentration_M": "lconc",
            "cond__metal_concentration_mM": "mconc", "cond__temperature_C": "temp"}[c]


def r2(d, c):
    return float(np.mean([1 - ((g.y - g[c]) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum() for _, g in d.groupby("split_seed")]))


def in_hull(P, x):
    n = len(P)
    res = linprog(np.zeros(n), A_eq=np.r_[P.T, np.ones((1, n))], b_eq=np.r_[x, 1.0], bounds=[(0, None)] * n,
                  method="highs")
    return bool(res.status == 0)


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    ids = f.cell_id.astype(str).to_numpy()
    pos = {c: i for i, c in enumerate(ids)}
    drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP.keys()) | set(V2.V2_3_CELLS)
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    ccols = [c for c in f.columns if c.startswith("cond__")]
    fams = np.array([family(c) for c in ccols])
    Mc = f[ccols].astype(float).to_numpy()
    xac = np.log10(f.cond__acid_concentration_M.astype(float).to_numpy())
    zL = np.log10(f.cond__extractant_concentration_M.astype(float).clip(lower=1e-4).to_numpy())
    zT = 1000.0 / (f.cond__temperature_C.astype(float).to_numpy() + 273.15)
    Z = np.c_[xac, zL, zT]

    def core_diff(i, j):
        a, c = Mc[i], Mc[j]
        same = (np.isnan(a) & np.isnan(c)) | np.isclose(a, c)
        return set(fams[~same]) - TRIV

    TI = pd.read_csv(OUT / "skA_targets.csv")
    R = pd.read_parquet(OUT / "skA_rows.parquet")

    # (a) carrier's CLEAN siblings
    print("(a) pub_19783c039b A2 targets: CLEAN siblings (sd of logD diff over common metals, offset, core diff)")
    for t in TI[(TI.pub == "pub_19783c039b") & (TI.reg_ok == True)].cell:                 # noqa: E712
        i = pos[t]
        met = np.isfinite(Y[i])
        cov = np.all(np.isfinite(Y[:, met]), axis=1)
        cov[i] = False
        out = []
        for j in np.where((pub == pub[i]) & (ext == ext[i]) & cov)[0]:
            cd = core_diff(i, j)
            if not cd or ids[j] in drop:
                continue
            dd = Y[i, met] - Y[j, met]
            out.append(f"{ids[j][:6]}@{10 ** xac[j]:.3g}M[{','.join(sorted(cd))}] sd {dd.std():.3f} off {dd.mean():+.2f}")
        print(f"  {t[:6]} @{10 ** xac[i]:.3g}M: " + "; ".join(out))

    # (b) frozen multi-coordinate AEBG on continuous-only series
    print("\n(b) frozen multi-coordinate AEBG (0 reads) on continuous-condition series")
    tm = {}
    for t in TI.cell:
        i = pos[t]
        if t in drop:
            continue
        met = np.where(np.isfinite(Y[i]))[0]
        cov = np.all(np.isfinite(Y[:, met]), axis=1)
        cov[i] = False
        CS = [j for j in np.where((pub == pub[i]) & (ext == ext[i]) & cov)[0]
              if ids[j] not in drop and core_diff(i, j) and core_diff(i, j) <= {"aconc", "lconc", "temp"}]
        if not CS:
            continue
        Zs = Z[CS + [i]]
        cols = [c for c in range(3) if np.nanmax(Zs[:, c]) - np.nanmin(Zs[:, c]) > 1e-6]
        if not cols or not np.all(np.isfinite(Z[i, cols])):
            continue
        CS = [j for j in CS if np.all(np.isfinite(Z[j, cols]))]
        p = len(cols)
        if len(CS) < p + 1:
            continue
        Zc = Z[CS][:, cols] - Z[CS][:, cols].mean(0)
        if np.linalg.matrix_rank(Zc, tol=1e-6) < p:
            continue
        Ys = Y[CS][:, met]
        Yb = Ys.mean(0)
        Bm = np.linalg.lstsq(Zc, Ys - Yb, rcond=None)[0]                      # p x M
        dz = Z[i, cols] - Z[CS][:, cols].mean(0)
        dof = len(CS) - p - 1
        lam_tr = lam_dir = 0.0
        if dof >= 1:
            res = Ys - Yb - Zc @ Bm
            s2 = (res ** 2).sum(0) / dof
            G = np.linalg.inv(Zc.T @ Zc)
            dev = Bm - Bm.mean(1, keepdims=True)
            M = len(met)
            v_tr = (dev ** 2).sum(0).mean()
            n_tr = s2.mean() * np.trace(G) * (1 - 1 / M)
            lam_tr = float(np.clip(1 - n_tr / v_tr, 0, 1)) if v_tr > 0 else 0.0
            v_d = ((dz @ dev) ** 2).mean()
            n_d = s2.mean() * float(dz @ G @ dz) * (1 - 1 / M)
            lam_dir = float(np.clip(1 - n_d / v_d, 0, 1)) if v_d > 0 else 0.0
        box = all(Z[CS][:, c].min() - 1e-9 <= Z[i, c] <= Z[CS][:, c].max() + 1e-9 for c in cols)
        hull = in_hull(Z[CS][:, cols], Z[i, cols]) if p > 1 else box
        reg = Yb + dz @ Bm
        T = {"CCOPY": Yb}
        for nm, lam, gate in (("CREG", 1.0, True), ("AEBG_TR", lam_tr, box), ("AEBG_DIR", lam_dir, box),
                              ("AEBG_TR_HULL", lam_tr, hull), ("AEBG_DIR_HULL", lam_dir, hull)):
            T[nm] = Yb + (lam if gate else 0.0) * (reg - Yb)
        full = {}
        for k, v in T.items():
            a = np.full(N_LN, np.nan)
            a[met] = v
            full[k] = a
        tm[t] = {"T": full, "cols": ",".join(("acid", "L", "T")[c] for c in cols), "nCS": len(CS), "dof": dof,
                 "lam_tr": lam_tr, "lam_dir": lam_dir, "box": box, "hull": hull}
    rows = []
    for r in R[(R.design == "BPG") & R.cell_id.isin(tm)].itertuples(index=False):
        a_, b_ = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
        d = tm[r.cell_id]
        rr = {"split_seed": r.split_seed, "cell_id": r.cell_id, "publication": r.publication, "y": r.y, "SH_3": r.SH_3,
              "B1_3": r.B1_3}
        for k, v in d["T"].items():
            rr[k] = v[a_] - v[b_]
        rows.append(rr)
    Q = pd.DataFrame(rows)
    out = []
    for p_, g in Q.groupby("publication"):
        cells = [tm[c] for c in g.cell_id.unique()]
        out.append({"pub": p_, "cols": cells[0]["cols"], "cells": len(cells), "dof_med": np.median([c["dof"] for c in cells]),
                    "lam_tr_med": np.median([c["lam_tr"] for c in cells]), "lam_dir_med": np.median([c["lam_dir"] for c in cells]),
                    "box_in": sum(c["box"] for c in cells), "hull_in": sum(c["hull"] for c in cells),
                    "CCOPY@0": r2(g, "CCOPY"), "CREG@0": r2(g, "CREG"), "AEBG_TR@0": r2(g, "AEBG_TR"),
                    "AEBG_DIR@0": r2(g, "AEBG_DIR"), "TR_HULL@0": r2(g, "AEBG_TR_HULL"),
                    "DIR_HULL@0": r2(g, "AEBG_DIR_HULL"), "SH@4": r2(g, "SH_3"), "B1@4": r2(g, "B1_3")})
    print(pd.DataFrame(out).round(3).to_string(index=False))
    g = Q[Q.publication == "pub_6a0ef8528c"]
    h = g[g.cell_id != "c950dc2982e96b83"]
    print("  C5 without c950dc: " + ", ".join(f"{c} {r2(h, c):.3f}" for c in
                                              ("CCOPY", "CREG", "AEBG_TR", "AEBG_DIR", "AEBG_TR_HULL", "SH_3")))
    print("  C5 per cell lam_tr / lam_dir / hull:",
          {c[:6]: (round(tm[c]["lam_tr"], 2), round(tm[c]["lam_dir"], 2), tm[c]["hull"]) for c in sorted(g.cell_id.unique())})


if __name__ == "__main__":
    main()
