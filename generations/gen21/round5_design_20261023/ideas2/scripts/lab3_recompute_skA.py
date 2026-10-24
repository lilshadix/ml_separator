#!/usr/bin/env python
"""Skeptic A for idea lab-3 (closure line: per-metal priors from other publications do not beat B1).

EXPLORATORY.  Every number comes from the 229 already-seen C3 cells and the 5 dev seeds (rounds 1-4 and the round-5
probes have seen them).  Nothing is sealed; this is an independent recomputation, not a test.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab3_recompute_skA.py

Independent of lab_calib.py / lab_calib2.py (no import from them): own PWL-with-potentials, own detrend, own donor
pools, own inference.  Shares only the read-only infrastructure (bench, fold cache, round-4 REF parquets and fitted
Gd amplitudes, round-3 V2 drop lists, cluster union-find).

Arms (all leave-target-out; K = 6.1 unless stated; v = (1 - w) u + w c, w = n / (n + K)):
  B1          recomputed (u = fold beta at Gd), checked against REF B1
  B2          round-4 REF column (fold-fitted universal 14-metal potentials + geometry; the fitted universal prior)
  MG          every other publication's cells (replicates lab_calib; NOT fold-legal)
  MG_tr       training-fold donors only (BPG-legal)
  MG_tr_kGd   MG_tr but B1's fitted beta kept at Gd (shape replaced only at the 13 other metals)
  MG_tr_h     half-strength: v = u + 0.5 (c - u)  (MG_tr is w ~ 0.98 everywhere, i.e. it replaces beta)
  MCO / MCO_tr      same chemotype, other publications, all / training-fold donors
  MG_LM{05,10,20}_tr, MCO_LM{05,10,20}_tr   level-matched (|log D_donor,m - L_t,m| <= tol), training donors
  MG_LM10_all, MCO_LM10                     level-matched, all other-publication donors (replicates)
  MG_LQ10_tr  level match on the DONOR'S QUADRATIC FIT at m instead of its raw log D (removes selection on the
              donor's own deviation; control for a selection-on-outcome artefact of raw matching)
  E_MRP       expected squared error with one random other publication's shape (replicates)
Checks: train/test isolation of donors (copy group under BPG), bit-identical / near-copy donor series in the
other-publication pools, with/without C5-BTBP (pub_6a0ef8528c), with/without V2-drop targets, publication LOPO and
cluster (cluster_cpg) jackknife intervals, sign-flip p, dependence on the top/bottom 3 publications.
Writes only ideas2/lab3_skA_out/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
from gen20k.splits import copy_group_map  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab3_skA_out")
XR = np.asarray(interp._coords("R"), float)
GD = CO.GD
K = 6.1
C5PUB = "pub_6a0ef8528c"
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)


def quad_dev(Y):
    """Deviation of each series from its own LS quadratic in the R coordinate (>= 6 metals) and the fit itself."""
    dev = np.full_like(Y, np.nan)
    fit = np.full_like(Y, np.nan)
    for i in range(len(Y)):
        m = np.isfinite(Y[i])
        if m.sum() < 6:
            continue
        A = np.vstack([XR[m] ** 2, XR[m], np.ones(m.sum())]).T
        co, *_ = np.linalg.lstsq(A, Y[i, m], rcond=None)
        fv = np.vstack([XR ** 2, XR, np.ones(N_LN)]).T @ co
        fit[i, m] = fv[m]
        dev[i, m] = Y[i, m] - fv[m]
    return dev, fit


def lev_curve(y, reads, metals, v):
    rd = sorted(reads, key=lambda m: XR[m])
    xr = XR[rd]
    zr = y[rd] - v[rd]
    out = np.full(N_LN, np.nan)
    mm = np.asarray(metals, int)
    out[mm] = np.interp(XR[mm], xr, zr) + v[mm]
    out[list(reads)] = y[list(reads)]
    return out


def mean_shape(dev, donors, mask=None):
    if len(donors) == 0:
        return np.zeros(N_LN), np.zeros(N_LN)
    A = dev[donors]
    ok = np.isfinite(A) if mask is None else (np.isfinite(A) & mask)
    n = ok.sum(0).astype(float)
    s = np.where(ok, A, 0.0).sum(0)
    c = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    return c, n


def sqe(d, a):
    return d[a].to_numpy(float) if a.startswith("E_") else (d[a].to_numpy(float) - d.y.to_numpy(float)) ** 2


def r2(d, a):
    out = []
    for _, g in d.groupby("split_seed"):
        y = g.y.to_numpy(float)
        out.append(1 - sqe(g, a).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out))


def rel(d, a, b="B1"):
    return float(1 - sqe(d, a).sum() / sqe(d, b).sum())


def jk(d, a, unit, b="B1"):
    if d[unit].nunique() < 3:
        return {k: np.nan for k in ("rel", "lo", "hi", "G", "up", "down", "p_sf", "jk_max", "jk_max_unit",
                                    "rel_drop_worst3", "rel_drop_best3")}
    g = d.assign(_a=sqe(d, a), _b=sqe(d, b)).groupby(unit)[["_a", "_b"]].sum()
    G = len(g)
    pt = 1 - g._a.sum() / g._b.sum()
    j = np.array([1 - g._a.drop(u).sum() / g._b.drop(u).sum() for u in g.index])
    se = np.sqrt((G - 1) / G * ((j - j.mean()) ** 2).sum())
    q = SS.t.ppf(0.975, G - 1)
    v = (g._b - g._a).to_numpy()
    rng = np.random.default_rng(1)
    sums = (rng.choice((-1, 1), size=(100000, G)) * np.abs(v)).sum(1)
    p = float((sums >= v.sum() - 1e-12).mean())
    order = np.argsort(v)
    worst3 = list(g.index[order[:3]])
    best3 = list(g.index[order[::-1][:3]])
    rw = 1 - g._a.drop(worst3).sum() / g._b.drop(worst3).sum()
    rb = 1 - g._a.drop(best3).sum() / g._b.drop(best3).sum()
    return {"rel": pt, "lo": pt - q * se, "hi": pt + q * se, "G": G, "up": int((v > 1e-12).sum()),
            "down": int((v < -1e-12).sum()), "p_sf": p, "jk_max": float(j.max()), "jk_max_unit": g.index[int(np.argmax(j))],
            "rel_drop_worst3": rw, "rel_drop_best3": rb}


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f["publication_id"].astype(str).to_numpy()
    chem = f["chemotype"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    nm = f["extractant_name"].astype(str).to_numpy()
    cgm = copy_group_map()
    cg = np.array([cgm.get(p, "?" + p) for p in pub])
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    dev, fit = quad_dev(Y)
    has = np.isfinite(dev).any(1) & okd
    ar = np.arange(len(f))
    print(f"bench {len(f)} cells; donor-eligible {has.sum()}; v2drop {len(v2drop)}", flush=True)

    # ---- near-copy / bit-identical donor check (other publications only) ----
    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in ("BPG", "BP")}
    c3 = sorted(set(refs["BPG"].cell_id))
    dup = []
    for t in c3:
        i = pos[t]
        mi = np.isfinite(Y[i])
        for j in np.where(has & (pub != pub[i]))[0]:
            mj = np.isfinite(Y[j])
            sh = mi & mj
            if sh.sum() < 4:
                continue
            dmax = np.abs(Y[i, sh] - Y[j, sh]).max()
            dd = Y[i, sh] - Y[j, sh]
            if dmax < 1e-6 or (dd - dd.mean()).std() < 0.02:
                dup.append({"target": t, "t_pub": pub[i], "donor": ids[j], "d_pub": pub[j], "shared": int(sh.sum()),
                            "max_abs": float(dmax), "sd_shift": float((dd - dd.mean()).std()),
                            "same_chem": chem[i] == chem[j], "same_cg": cg[i] == cg[j], "t_name": nm[i], "d_name": nm[j]})
    DUP = pd.DataFrame(dup)
    DUP.to_csv(OUT / "near_copy_donors.csv", index=False)
    print(f"near-copy (shift-sd < 0.02 or identical) other-publication donor links: {len(DUP)}; "
          f"targets {DUP.target.nunique() if len(DUP) else 0}; bit-identical {int((DUP.max_abs < 1e-6).sum()) if len(DUP) else 0}")
    if len(DUP):
        print(DUP.groupby(["t_pub", "d_pub"]).agg(n=("target", "size"), same_chem=("same_chem", "first"),
                                                     same_cg=("same_cg", "first"), sd=("sd_shift", "median")).to_string())
    dupset = {(r.target, r.donor) for r in DUP.itertuples()} if len(DUP) else set()

    rows = []
    iso_viol = 0
    for design in ("BPG", "BP"):
        par = refs[design]
        fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = C.load(design, seed, fold)
                tr = np.zeros(len(f), bool)
                tr[np.asarray(rec["outer"]["train_index"], int)] = True
                te = np.zeros(len(f), bool)
                te[np.asarray(rec["outer"]["test_index"], int)] = True
                assert not (tr & te).any()
                beta = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
                u = np.zeros(N_LN)
                u[GD] = beta
                prow = par[(par.split_seed == seed) & (par.fold == fold)]
                pcg = {c: g for c, g in prow.groupby("cell_id")}
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if t not in pcg:
                        continue
                    i = pos[t]
                    assert te[i] and not tr[i]
                    y = Y[i]
                    metals = list(cell["metals"])
                    reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                    base = has & (ar != i)
                    oth = base & (pub != pub[i])
                    dG = np.where(oth)[0]
                    dGt = np.where(oth & tr)[0]
                    if design == "BPG":
                        iso_viol += int((cg[dGt] == cg[i]).sum())
                    dC = np.where(oth & (chem == chem[i]))[0]
                    dCt = np.where(oth & tr & (chem == chem[i]))[0]
                    dGnc = np.array([j for j in dG if (t, ids[j]) not in dupset], int)
                    L = {"B1": lev_curve(y, reads, metals, u)}
                    lv = L["B1"]

                    def arm(c, n, keep_gd=False, lam=None):
                        w = n / (n + K) if lam is None else np.where(n > 0, lam, 0.0)
                        v = (1 - w) * u + w * c
                        if keep_gd:
                            v[GD] = u[GD]
                        return lev_curve(y, reads, metals, v)
                    cG = mean_shape(dev, dG)
                    cGt = mean_shape(dev, dGt)
                    L["MG"] = arm(*cG)
                    L["MG_nc"] = arm(*mean_shape(dev, dGnc))
                    L["MG_tr"] = arm(*cGt)
                    L["MG_tr_kGd"] = arm(*cGt, keep_gd=True)
                    L["MG_tr_h"] = arm(*cGt, lam=0.5)
                    L["MCO"] = arm(*mean_shape(dev, dC))
                    L["MCO_tr"] = arm(*mean_shape(dev, dCt))
                    for tol, tg in ((0.5, "05"), (1.0, "10"), (2.0, "20")):
                        mG = np.abs(Y[dGt] - lv[None, :]) <= tol
                        mC = np.abs(Y[dCt] - lv[None, :]) <= tol
                        L[f"MG_LM{tg}_tr"] = arm(*mean_shape(dev, dGt, mG))
                        L[f"MCO_LM{tg}_tr"] = arm(*mean_shape(dev, dCt, mC))
                    L["MG_LM10_all"] = arm(*mean_shape(dev, dG, np.abs(Y[dG] - lv[None, :]) <= 1.0))
                    L["MCO_LM10"] = arm(*mean_shape(dev, dC, np.abs(Y[dC] - lv[None, :]) <= 1.0))
                    L["MG_LQ10_tr"] = arm(*mean_shape(dev, dGt, np.abs(fit[dGt] - lv[None, :]) <= 1.0))
                    L["MG_LM10_tr_kGd"] = arm(*mean_shape(dev, dGt, np.abs(Y[dGt] - lv[None, :]) <= 1.0), keep_gd=True)
                    RP = []
                    for p in np.unique(pub[dG]):
                        dd = np.where(oth & (pub == p))[0]
                        RP.append(arm(*mean_shape(dev, dd)))
                    nG = cGt[1]
                    for r in pcg[t].itertuples(index=False):
                        ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                        o = {"design": design, "split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i],
                             "chemotype": chem[i], "name": nm[i], "pair": f"{r.A}/{r.B}", "y": r.y, "B1_ref": r.B1,
                             "B2": r.B2, "B0": r.B0, "t_v2drop": t in v2drop, "n_MCO_tr": len(dCt), "n_MCO": len(dC),
                             "wGd_MG_tr": float(nG[GD] / (nG[GD] + K)), "has_dup": any((t, ids[j]) in dupset for j in dG)}
                        for k2, l2 in L.items():
                            o[k2] = l2[ia] - l2[ib]
                        o["E_MRP"] = float(np.mean([(l2[ia] - l2[ib] - r.y) ** 2 for l2 in RP]))
                        rows.append(o)
            print(f"{design} done {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)
    print(f"\nmax |B1(recomputed) - B1(REF)| = {float((T.B1 - T.B1_ref).abs().max()):.2e} over {len(T)} rows")
    print(f"BPG training donors sharing the target's copy group: {iso_viol}")
    print("rows with any fold-legal same-chemotype donor (n_MCO_tr > 0):", int((T.n_MCO_tr > 0).sum()),
          "| max |MCO_tr - B1|:", float((T.MCO_tr - T.B1).abs().max()),
          "| max |MCO_LM10_tr - B1|:", float((T.MCO_LM10_tr - T.B1).abs().max()))
    T = M4.add_clusters(T, f)
    T.to_parquet(OUT / "lab3_skA_rows.parquet", index=False)

    ARMS = ["B2", "MG", "MG_nc", "MG_tr", "MG_tr_kGd", "MG_tr_h", "MCO", "MCO_tr", "MG_LM05_tr", "MG_LM10_tr",
            "MG_LM20_tr", "MCO_LM05_tr", "MCO_LM10_tr", "MCO_LM20_tr", "MG_LM10_all", "MCO_LM10", "MG_LQ10_tr",
            "MG_LM10_tr_kGd", "E_MRP"]
    summ = []
    for design in ("BPG", "BP"):
        d0 = T[T.design == design]
        pops = {"all": d0, "noC5": d0[d0.publication != C5PUB], "clean": d0[~d0.t_v2drop],
                "clean_noC5": d0[~d0.t_v2drop & (d0.publication != C5PUB)],
                "MCOrows": d0[d0.n_MCO > 0], "MCOrows_noC5": d0[(d0.n_MCO > 0) & (d0.publication != C5PUB)]}
        for pn, d in pops.items():
            if pn.startswith("MCO") and design == "BP":
                pass
            for a in ARMS:
                rp = jk(d, a, "publication")
                rc = jk(d, a, "cluster_cpg")
                summ.append({"design": design, "pop": pn, "arm": a, "cells": d.cell_id.nunique(), "R2": r2(d, a),
                             "R2_B1": r2(d, "B1"), "rel": rp["rel"], "pub_lo": rp["lo"], "pub_hi": rp["hi"],
                             "pub_up": rp["up"], "pub_down": rp["down"], "G": rp["G"], "p_sf": rp["p_sf"],
                             "cl_lo": rc["lo"], "cl_hi": rc["hi"], "Gc": rc["G"], "cl_up": rc["up"],
                             "cl_jk_max": rc["jk_max"], "cl_jk_max_unit": rc["jk_max_unit"],
                             "rel_drop_worst3pub": rp["rel_drop_worst3"], "rel_drop_best3pub": rp["rel_drop_best3"]})
    S = pd.DataFrame(summ)
    S.to_csv(OUT / "lab3_skA_summary.csv", index=False)
    for design in ("BPG", "BP"):
        for pn in ("all", "noC5", "clean_noC5", "MCOrows", "MCOrows_noC5"):
            s = S[(S.design == design) & (S["pop"] == pn)]
            print(f"\n==== {design} / {pn}: cells {s.cells.iloc[0]}, B1 R2 {s.R2_B1.iloc[0]:.4f} ====")
            print(s[["arm", "R2", "rel", "pub_lo", "pub_hi", "pub_up", "pub_down", "p_sf", "cl_lo", "cl_hi", "Gc", "cl_up",
                     "cl_jk_max", "cl_jk_max_unit", "rel_drop_worst3pub", "rel_drop_best3pub"]].round(4).to_string(index=False))
    d0 = T[T.design == "BPG"]
    print("\nMG_tr weight at Gd (w = n/(n+K)): min %.3f median %.3f" % (d0.wGd_MG_tr.min(), d0.wGd_MG_tr.median()))
    print("targets with a near-copy other-publication donor:", d0[d0.has_dup].cell_id.nunique())
    print("\nper-cluster rel SSE, BPG, MG_LM10_tr / MCO / MCO_tr / MG_tr_kGd / B2:")
    for c, g in d0.groupby("cluster_cpg"):
        print(f"  {c:28s} cells {g.cell_id.nunique():3d}: " + " ".join(f"{rel(g, a):+.3f}" for a in
              ("MG_LM10_tr", "MCO", "MCO_tr", "MG_tr_kGd", "B2")))
    print("\nMCO per publication (BPG), top 6 by |dSSE|:")
    g = d0.assign(_a=sqe(d0, "MCO"), _b=sqe(d0, "B1")).groupby("publication")[["_a", "_b"]].sum()
    g["d"] = (g._b - g._a) / 5
    print(g.reindex(g.d.abs().sort_values(ascending=False).index).head(6).round(4).to_string())
    print(f"\ndone {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
