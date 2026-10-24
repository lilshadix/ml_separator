#!/usr/bin/env python
"""SKEPTIC A for shift-2: duplicate / near-copy checks, the pub_19783c039b carrier, and an independent recompute of the
two-factor CREG claims (C5-BTBP grid, DMDO-pyran-DGA [L] series).  EXPLORATORY.  Prints only.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_skA_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "shift_skA_out"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
TRIV = {"ctime", "mconc"}
C5 = "pub_6a0ef8528c"


def family(c):
    if c == "cond__acid_concentration_M":
        return "aconc"
    for pre, nm in (("cond__acid__", "atype"), ("cond__additive__", "add"), ("cond__diluent__", "dil")):
        if c.startswith(pre):
            return nm
    return {"cond__contact_time_min": "ctime", "cond__extractant_concentration_M": "lconc",
            "cond__metal_concentration_mM": "mconc", "cond__temperature_C": "temp"}[c]


def r2s(d, c):
    return np.array([1 - ((g.y - g[c]) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum() for _, g in d.groupby("split_seed")])


def r2(d, c):
    return float(r2s(d, c).mean())


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
    acid_cols = [c for c in ccols if c.startswith("cond__acid__")]
    alab = np.array(["+".join(c[12:] for c in acid_cols if f.at[i, c] == 1) or "-" for i in range(len(f))])
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
    A2 = TI[(TI.n_AS >= 1) & (TI.reg_ok == True)]                                   # noqa: E712

    # ---------------- 1. near-copies: target vs its acid-only siblings, and vs every other bench cell
    print("[1] near-copy scan: sd over metals of (logD_t - logD_j)  (sd ~ 0 => identical log SF profile)")
    hits, mins = [], []
    for t in A2.cell:
        i = pos[t]
        mi = np.isfinite(Y[i])
        best = (9.0, None)
        for j in range(len(f)):
            if j == i:
                continue
            mm = mi & np.isfinite(Y[j])
            if mm.sum() < 5:
                continue
            dd = Y[i, mm] - Y[j, mm]
            s = float(dd.std())
            if s < best[0]:
                best = (s, j)
            if s < 0.03:
                hits.append({"target": t, "tpub": pub[i], "other": ids[j], "opub": pub[j], "same_pub": pub[i] == pub[j],
                             "same_ext": ext[i] == ext[j], "n_common": int(mm.sum()), "sd": round(s, 4),
                             "offset": round(float(dd.mean()), 3), "other_v2drop": ids[j] in drop,
                             "core_diff": ",".join(sorted(core_diff(i, j))) or "NONE"})
        mins.append(best[0])
    print(f"  A2 targets {len(A2)}; min sd per target: quantiles {np.round(np.quantile(mins, [0, .1, .25, .5]), 3)}")
    H = pd.DataFrame(hits)
    print(H.to_string(index=False) if len(H) else "  no pair with sd < 0.03")

    # ---------------- 2. the carrier: pub_19783c039b
    P = "pub_19783c039b"
    print(f"\n[2] {P}: every bench cell")
    ii = np.where(pub == P)[0]
    tab = pd.DataFrame({"cell": ids[ii], "ext": [e[:40] for e in ext[ii]], "acid": alab[ii],
                        "acid_M": f.cond__acid_concentration_M.to_numpy()[ii],
                        "L_M": f.cond__extractant_concentration_M.to_numpy()[ii],
                        "T": f.cond__temperature_C.to_numpy()[ii], "nmet": np.isfinite(Y[ii]).sum(1),
                        "v2drop": [c in drop for c in ids[ii]], "C3": [c in set(TI.cell) for c in ids[ii]],
                        "A2": [c in set(A2.cell) for c in ids[ii]]})
    print(tab.sort_values(["ext", "acid_M"]).to_string(index=False))
    # decimals of the reported log D (digitisation / fitted-line signature)
    vals = Y[ii][np.isfinite(Y[ii])]
    print(f"  log D values: n {vals.size}; fraction on a 0.01 grid {np.mean(np.isclose(vals * 100, np.round(vals * 100), atol=1e-6)):.2f}; "
          f"range [{vals.min():.2f}, {vals.max():.2f}]")
    for e in sorted(set(ext[ii])):
        jj = [j for j in ii if ext[j] == e and ids[j] not in drop]
        if len(jj) < 3:
            continue
        xs = xac[jj]
        met = np.where(np.all(np.isfinite(Y[jj]), axis=0))[0]
        if len(met) < 3 or np.ptp(xs) < 0.1:
            continue
        res_sd, slopes = [], []
        for m in met:
            p = np.polyfit(xs, Y[jj, m], 1)
            res_sd.append(np.sqrt(((Y[jj, m] - np.polyval(p, xs)) ** 2).sum() / max(len(xs) - 2, 1)))
            slopes.append(p[0])
        print(f"  ext {e[:40]}: {len(jj)} cells, acid M {np.round(10 ** np.sort(xs), 3).tolist()}, metals {len(met)}; "
              f"per-metal slope {np.round(slopes, 2).tolist()}; residual sd median {np.median(res_sd):.3f} max {np.max(res_sd):.3f}")
    # per-cell gain inside the carrier
    q = R[(R.design == "BPG") & (R.publication == P) & R.ACOPY_0.notna()]
    q = q.merge(TI[["cell", "lam", "inside", "n_AS", "x_t", "x_min", "x_max"]], left_on="cell_id", right_on="cell")
    g = q.groupby("cell_id").apply(lambda d: pd.Series({
        "lam": d.lam.iloc[0], "inside": d.inside.iloc[0], "n_AS": d.n_AS.iloc[0], "acid": round(10 ** d.x_t.iloc[0], 3),
        "range": f"{10 ** d.x_min.iloc[0]:.3g}-{10 ** d.x_max.iloc[0]:.3g}",
        "SSE_ACOPY0": ((d.y - d.ACOPY_0) ** 2).sum() / 5, "SSE_AREG0": ((d.y - d.AREG_0) ** 2).sum() / 5,
        "SSE_AEBG0": ((d.y - d.AEBG_0) ** 2).sum() / 5, "SSE_AINT0": ((d.y - d.AINT_0) ** 2).sum() / 5,
        "SSE_SH4": ((d.y - d.SH_3) ** 2).sum() / 5}), include_groups=False)
    print(g.round(3).to_string())

    # ---------------- 3. two-factor CREG recompute (C5-BTBP and every publication with a continuous-only grid)
    print("\n[3] CREG recompute (per-metal OLS on varying z among continuous-only siblings)")
    tm = {}
    for t in TI.cell:
        i = pos[t]
        met = np.where(np.isfinite(Y[i]))[0]
        cov = np.all(np.isfinite(Y[:, met]), axis=1)
        cov[i] = False
        if t in drop:
            continue
        CS = [j for j in np.where((pub == pub[i]) & (ext == ext[i]) & cov)[0]
              if ids[j] not in drop and core_diff(i, j) and core_diff(i, j) <= {"aconc", "lconc", "temp"}]
        if not CS:
            continue
        Zs = Z[CS + [i]]
        cols = [c for c in range(3) if np.nanmax(Zs[:, c]) - np.nanmin(Zs[:, c]) > 1e-6]
        CS = [j for j in CS if np.all(np.isfinite(Z[j, cols]))] if np.all(np.isfinite(Z[i, cols])) else []
        if not CS or not cols or len(CS) < len(cols) + 1:
            continue
        Zc = Z[CS][:, cols] - Z[CS][:, cols].mean(0)
        if np.linalg.matrix_rank(Zc, tol=1e-6) < len(cols):
            continue
        X = np.c_[np.ones(len(CS)), Z[CS][:, cols]]
        creg = np.full(N_LN, np.nan)
        for m in met:
            beta = np.linalg.lstsq(X, Y[CS, m], rcond=None)[0]
            creg[m] = np.r_[1.0, Z[i, cols]] @ beta
        inside = all(Z[CS][:, c].min() - 1e-9 <= Z[i, c] <= Z[CS][:, c].max() + 1e-9 for c in cols)
        tm[t] = {"CREG": creg, "CCOPY": Y[CS].mean(0), "cols": ",".join(("acid", "L", "T")[c] for c in cols),
                 "nCS": len(CS), "inside": inside}
    rows = []
    for r in R[R.cell_id.isin(tm)].itertuples(index=False):
        a_, b_ = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
        d = tm[r.cell_id]
        rows.append({"design": r.design, "split_seed": r.split_seed, "cell_id": r.cell_id, "publication": r.publication,
                     "pair": r.pair, "y": r.y, "B1_3": r.B1_3, "SH_3": r.SH_3, "cols": d["cols"], "inside": d["inside"],
                     "CREG_0": d["CREG"][a_] - d["CREG"][b_], "CCOPY_0": d["CCOPY"][a_] - d["CCOPY"][b_]})
    Q = pd.DataFrame(rows)
    Q = Q[Q.design == "BPG"]
    out = []
    for p_, g in Q.groupby("publication"):
        out.append({"pub": p_, "cols": ",".join(sorted(g.cols.unique())), "cells": g.cell_id.nunique(),
                    "inside_cells": g[g.inside].cell_id.nunique(), "rows/seed": len(g) / 5,
                    "CCOPY@0": r2(g, "CCOPY_0"), "CREG@0": r2(g, "CREG_0"), "B1@4": r2(g, "B1_3"), "SH@4": r2(g, "SH_3")})
    print(pd.DataFrame(out).round(3).to_string(index=False))
    g = Q[Q.publication == C5]
    print(f"  C5 grid: CREG@0 {r2(g, 'CREG_0'):.3f} CCOPY@0 {r2(g, 'CCOPY_0'):.3f} B1@4 {r2(g, 'B1_3'):.3f} SH@4 {r2(g, 'SH_3'):.3f}")
    # leave-one-cell-out sensitivity inside the C5 grid: CREG@0 - SH@4 without each cell
    loo = []
    for c in sorted(g.cell_id.unique()):
        h = g[g.cell_id != c]
        loo.append((c[:6], round(r2(h, "CREG_0") - r2(h, "SH_3"), 3)))
    print("  C5 CREG@0 - SH@4 dropping each cell:", loo)
    # per-cell SSE in C5 (which cells carry the CREG gain)
    pc = g.groupby("cell_id").apply(lambda d: pd.Series({
        "acid": None, "SSE_CREG0": ((d.y - d.CREG_0) ** 2).sum() / 5, "SSE_SH4": ((d.y - d.SH_3) ** 2).sum() / 5,
        "SSE_CCOPY0": ((d.y - d.CCOPY_0) ** 2).sum() / 5, "inside": d.inside.iloc[0]}), include_groups=False)
    for c in pc.index:
        i = pos[c]
        pc.loc[c, "acid"] = f"{10 ** xac[i]:.2g}M / L {f.cond__extractant_concentration_M[i]:.3g}"
    print(pc.round(3).to_string())
    c5c = np.where(pub == C5)[0]
    print(f"  C5 La/Ce log D range: La {np.nanmin(Y[c5c, 0]):.2f}..{np.nanmax(Y[c5c, 0]):.2f}, "
          f"Ce {np.nanmin(Y[c5c, 1]):.2f}..{np.nanmax(Y[c5c, 1]):.2f}")
    # number of independent factorial grids outside C5 in the corpus
    print("  publications with a >=2-coordinate continuous grid:",
          sorted({(d['cols'], pub[pos[c]]) for c, d in tm.items() if "," in d["cols"]}))


if __name__ == "__main__":
    main()
