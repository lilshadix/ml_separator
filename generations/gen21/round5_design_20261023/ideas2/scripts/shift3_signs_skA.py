#!/usr/bin/env python
"""Skeptic A for idea shift-3 (light-end acid steepening as a transferable sign).  EXPLORATORY: every cell of the
521-cell V1 bench has been seen; nothing here is confirmatory.

Independent recomputation (own grouping code, not importing shift_*.py):
  S0  rebuild acid-series groups; per-pair OLS slope d(logSF)/d(log10[acid]); per-publication mean; sign counts,
      exact sign-flip p (two-sided), Holm over the 13 pair types (multiplicity of picking Ce/Pr post hoc).
  S1  grouping variants: (a) proposer-like key (ignores contact time and metal conc.), raw V1 Y minus V2-1/2/3;
      (b) V2 matrix (also removes V2-5 floor readings); (c) strict key (contact time + metal conc. in key).
  S2  robustness of Ce/Pr: drop C5-BTBP; drop the round-3 cross-acid bit-identical copy pairs; drop any group that
      contains a within-group bit-identical (>= 7 of 14) pair; >= 3 acid levels only; HNO3 only; DGA (sc009/sc027)
      vs non-DGA; leave-k-publications-out; per-extractant units (TODGA pooled across publications).
  S3  floor / ceiling: Ce/Pr slope using only cells with both Ce and Pr log D inside [lo, hi] windows.
  S4  level (stretch / compression) control: per group, regress metal acid slopes on metal mean log D using metals
      other than La-Nd; predict Ce and Pr; residual Ce/Pr slope sign.  Plus a position-matched control: Ce/Pr vs
      other pairs whose level sits in the same within-group level quantile.
  S5  direction vs magnitude: sign of the Ce/Pr log SF itself in each group (is any group light-selective at Ce/Pr?).
  S6  Ce-specific vs light-end: La/Pr slope (skips Ce) and Ce deviation slope.
  S7  placebo: acid levels permuted within each group (null rate of 'all negative').
  S8  predictive leg on bench groups: leave-one-cell-out ACOPY (mean of the other cells' Ce/Pr log SF) vs TILT_CE
      (ACOPY + s_prior * dx, s_prior = mean per-publication Ce/Pr slope of OTHER publications), relative SSE on Ce/Pr.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift3_signs_skA.py
"""
from __future__ import annotations

import itertools
import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES as LN, N_LN  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 40)
C5 = "pub_6a0ef8528c"
XAC = {"44012938119033a5", "5169c377777d282b", "d4c1e23a091d0960", "e4ee526bda3920c2"}
PAIRS = [(i, i + 1) for i in range(N_LN - 1)]
PN = [f"{LN[a]}/{LN[b]}" for a, b in PAIRS]
iLa, iCe, iPr, iNd = 0, 1, 2, 3
DGA = {"sc009", "sc027", "sc018", "sc030", "sc031"}   # DGA-class chemotypes seen in the groups (name-based)


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) < 0.1 or len(np.unique(np.round(x, 3))) < 2:
        return np.nan
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def signflip(v, alt="two"):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0 or n > 18:
        return np.nan
    S = np.array(list(itertools.product((1, -1), repeat=n)))
    null = S @ v
    if alt == "two":
        return float((np.abs(null) >= abs(v.sum()) - 1e-12).mean())
    return float((null <= v.sum() + 1e-12).mean())    # one-sided: negative


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def label(f, pre):
    cols = [c for c in f.columns if c.startswith(pre)]
    M = f[cols].to_numpy()
    return ["+".join(c[len(pre):] for c, v in zip(cols, row) if v == 1) or "-" for row in M]


def build(f, Y, strict=False, drop=frozenset()):
    key = ["publication_id", "extractant", "acid", "dil", "add", "Lc", "T"] + (["ct", "mc"] if strict else [])
    keep = f.x.notna() & ~f.cell_id.isin(drop)
    G = []
    for k, g in f[keep].groupby(key, dropna=False):
        if len(g) < 2 or np.ptp(g.x) < 0.1:
            continue
        idx = g.index.to_numpy()
        G.append({"key": k, "pub": k[0], "ext": str(g.extractant_name.iloc[0])[:16], "chem": g.chemotype.iloc[0],
                  "acid": k[2], "idx": idx, "x": g.x.to_numpy(), "cells": tuple(g.cell_id)})
    return G


def pair_slopes(G, Y, win=None, pairs=None):
    rows = []
    pairs = pairs or list(zip(PAIRS, PN))
    for gi, g in enumerate(G):
        Yg = Y[g["idx"]].copy()
        for (a, b), pn in pairs:
            ya, yb = Yg[:, a].copy(), Yg[:, b].copy()
            if win is not None:
                bad = (ya < win[0]) | (ya > win[1]) | (yb < win[0]) | (yb > win[1])
                ya[bad] = np.nan
            yy = ya - yb
            s = slope(g["x"], yy)
            if np.isfinite(s):
                ok = np.isfinite(yy)
                rows.append({"g": gi, "pub": g["pub"], "ext": g["ext"], "chem": g["chem"], "acid": g["acid"], "pair": pn,
                             "slope": s, "ybar": float(np.nanmean(yy)), "n": int(ok.sum()),
                             "nx": len(np.unique(np.round(g["x"][ok], 3))),
                             "lev": float(np.nanmean((Yg[:, a] + Yg[:, b]) / 2))})
    return pd.DataFrame(rows)


def summarise(P, pair="Ce/Pr", unit="pub"):
    q = P[P.pair == pair]
    if q.empty:
        return {"units": 0}
    v = q.groupby(unit).slope.mean()
    return {"units": len(v), "neg": int((v < 0).sum()), "pos": int((v > 0).sum()), "mean": round(float(v.mean()), 3),
            "median": round(float(v.median()), 3), "p2": signflip(v.to_numpy()), "p1": signflip(v.to_numpy(), "one")}


def table(P, unit="pub", label_=""):
    rows = []
    for pn in PN:
        s = summarise(P, pn, unit)
        if s["units"]:
            rows.append({"pair": pn, **s})
    T = pd.DataFrame(rows)
    T["holm_p2"] = holm(T.p2.fillna(1).to_numpy())
    print(f"\n--- {label_} (unit = {unit}) ---")
    print(T.round(4).to_string(index=False))
    return T


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y1 = np.asarray(b.Y, float).copy()
    v123 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    Y1[f.cell_id.isin(v123).to_numpy()] = np.nan
    Y2 = np.asarray(V2.matrix(b), float)
    f["acid"] = label(f, "cond__acid__")
    f["dil"] = label(f, "cond__diluent__")
    f["add"] = label(f, "cond__additive__")
    f["Lc"] = f.cond__extractant_concentration_M.astype(float).round(4)
    f["T"] = f.cond__temperature_C.astype(float).round(0)
    f["ct"] = f.cond__contact_time_min.astype(float).round(1).fillna(-1)
    f["mc"] = f.cond__metal_concentration_mM.astype(float).round(4).fillna(-1)
    f["x"] = np.log10(f.cond__acid_concentration_M.astype(float))

    # ---------------- S0 / S1 ----------------
    print("=" * 100 + "\nS0/S1: recomputation of pooled pair slopes\n" + "=" * 100)
    G = build(f, Y1, drop=v123)
    P = pair_slopes(G, Y1)
    print(f"groups with x span >= 0.1: {len(G)}; with >= 1 pair slope: {P.g.nunique()}; publications {P.pub.nunique()}")
    T_main = table(P, label_="(a) proposer-like key, V1 minus V2-1/2/3")
    G2 = build(f, Y2, drop=v123)
    P2 = pair_slopes(G2, Y2)
    table(P2, label_="(b) V2 matrix (V2-5 floor readings removed)")
    Gs = build(f, Y1, strict=True, drop=v123)
    Ps = pair_slopes(Gs, Y1)
    print(f"strict key: groups {len(Gs)}, with slopes {Ps.g.nunique()}")
    table(Ps, label_="(c) strict key (contact time + metal concentration in key)")

    print("\nCe/Pr per group (variant a):")
    q = P[P.pair == "Ce/Pr"].copy()
    q["t"] = np.nan
    print(q[["g", "pub", "ext", "chem", "acid", "slope", "ybar", "n", "nx", "lev"]].round(3).to_string(index=False))

    # ---------------- S2 ----------------
    print("\n" + "=" * 100 + "\nS2: robustness of the Ce/Pr sign\n" + "=" * 100)
    res = {}
    res["main"] = summarise(P)
    res["no C5"] = summarise(P[P.pub != C5])
    Gx = build(f, Y1, drop=v123 | XAC)
    Px = pair_slopes(Gx, Y1)
    res["drop cross-acid copy cells"] = summarise(Px)
    # scan within-group bit-identical pairs (>= 7 of 14 metals equal at different x)
    dup_groups = set()
    dup_rows = []
    for gi, g in enumerate(G):
        Yg = Y1[g["idx"]]
        for i, j in itertools.combinations(range(len(g["idx"])), 2):
            both = np.isfinite(Yg[i]) & np.isfinite(Yg[j])
            eq = int((both & (np.abs(Yg[i] - Yg[j]) < 1e-12)).sum())
            if eq >= 7 and abs(g["x"][i] - g["x"][j]) > 1e-6:
                dup_groups.add(gi)
                dup_rows.append({"g": gi, "pub": g["pub"], "cell_i": g["cells"][i], "cell_j": g["cells"][j],
                                 "x_i": round(g["x"][i], 3), "x_j": round(g["x"][j], 3), "n_equal": eq,
                                 "n_both": int(both.sum())})
    print("within-group bit-identical (>= 7 metals) cell pairs at different acid:")
    print(pd.DataFrame(dup_rows).to_string(index=False) if dup_rows else "  none")
    # cross-group: identical Ce and Pr values (same pair value) across different publications
    cp = pd.DataFrame({"pub": f.publication_id, "cell": f.cell_id, "Ce": Y1[:, iCe], "Pr": Y1[:, iPr]}).dropna()
    cp["k"] = cp.Ce.round(6).astype(str) + "|" + cp.Pr.round(6).astype(str)
    dd = cp[cp.duplicated("k", keep=False)].sort_values("k")
    multi = dd.groupby("k").pub.nunique()
    print(f"cells sharing an identical (Ce, Pr) log D pair: {len(dd)}; of which across >1 publication: "
          f"{int((multi > 1).sum())} value-pairs")
    if (multi > 1).any():
        print(dd[dd.k.isin(multi[multi > 1].index)].to_string(index=False))
    res["drop groups with within-group copies"] = summarise(P[~P.g.isin(dup_groups)])
    gx = {gi: len(np.unique(np.round(g["x"], 3))) for gi, g in enumerate(G)}
    res[">= 3 acid levels"] = summarise(P[P.g.map(gx) >= 3])
    res[">= 3 acid levels, no C5"] = summarise(P[(P.g.map(gx) >= 3) & (P.pub != C5)])
    res["HNO3 only"] = summarise(P[P.acid == "hno3"])
    res["HNO3 only, no C5"] = summarise(P[(P.acid == "hno3") & (P.pub != C5)])
    res["DGA-class chemotypes"] = summarise(P[P.chem.isin(DGA)])
    res["non-DGA chemotypes"] = summarise(P[~P.chem.isin(DGA)])
    res["non-DGA, no C5"] = summarise(P[~P.chem.isin(DGA) & (P.pub != C5)])
    res["unit = extractant name"] = summarise(P, unit="ext")
    res["unit = chemotype"] = summarise(P, unit="chem")
    print(pd.DataFrame(res).T.to_string())
    q = P[P.pair == "Ce/Pr"]
    pv = q.groupby("pub").agg(slope=("slope", "mean"), groups=("g", "nunique"), chem=("chem", "first"),
                              ext=("ext", "first"), nx_max=("nx", "max"), ybar=("ybar", "mean"))
    print("\nCe/Pr per publication:")
    print(pv.round(3).to_string())
    print("\nextractant names among Ce/Pr publications:", q.groupby("ext").pub.nunique().to_dict())
    # leave-k-out on the mean and on the sign-flip p
    pubs = pv.index.tolist()
    v = pv.slope
    worst = {}
    for kk in (1, 2, 3):
        ps, ms = [], []
        for drop in itertools.combinations(pubs, kk):
            vv = v.drop(list(drop)).to_numpy()
            ps.append(signflip(vv, "one"))
            ms.append(vv.mean())
        worst[kk] = {"max one-sided p": round(max(ps), 4), "max mean": round(max(ms), 3), "min mean": round(min(ms), 3)}
    print("leave-k-publications-out (Ce/Pr):", worst)
    # top contributors to the mean
    print("share of the summed negative slope carried by the most negative publications:",
          (v.sort_values().cumsum() / v.sum()).round(2).head(4).to_dict())

    # ---------------- S3 ----------------
    print("\n" + "=" * 100 + "\nS3: floor/ceiling windows on Ce and Pr (cells kept only if both inside)\n" + "=" * 100)
    wres = {}
    for win in [(-2, 2.5), (-1.5, 2.0), (-1.0, 1.5), (-0.5, 1.0)]:
        Pw = pair_slopes(G, Y1, win=win, pairs=[((iCe, iPr), "Ce/Pr"), ((iLa, iCe), "La/Ce"), ((iPr, iNd), "Pr/Nd"),
                                                ((12, 13), "Yb/Lu")])
        for pn in ("Ce/Pr", "La/Ce", "Pr/Nd", "Yb/Lu"):
            wres[(str(win), pn)] = summarise(Pw, pn)
            wres[(str(win), pn + " noC5")] = summarise(Pw[Pw.pub != C5], pn)
    print(pd.DataFrame(wres).T.to_string())

    # ---------------- S4 ----------------
    print("\n" + "=" * 100 + "\nS4: level (stretch / compression) control\n" + "=" * 100)
    rows = []
    for gi, g in enumerate(G):
        Yg = Y1[g["idx"]]
        sm = np.array([slope(g["x"], Yg[:, m]) for m in range(N_LN)])
        lv = np.nanmean(Yg, 0)
        ref = [m for m in range(4, N_LN) if np.isfinite(sm[m]) and np.isfinite(lv[m])]
        if not (np.isfinite(sm[iCe]) and np.isfinite(sm[iPr])) or len(ref) < 4 or np.ptp(lv[ref]) < 0.2:
            continue
        A = np.c_[np.ones(len(ref)), lv[ref]]
        beta = np.linalg.lstsq(A, sm[ref], rcond=None)[0]
        pred = beta[0] + beta[1] * lv
        obs = sm[iCe] - sm[iPr]
        rows.append({"g": gi, "pub": g["pub"], "ext": g["ext"], "chem": g["chem"], "acid": g["acid"],
                     "nref": len(ref), "kappa_ref": beta[1], "lvCe": lv[iCe], "lvPr": lv[iPr],
                     "obs_CePr": obs, "pred_CePr_from_level": pred[iCe] - pred[iPr], "resid": obs - (pred[iCe] - pred[iPr]),
                     "obs_LaCe": sm[iLa] - sm[iCe] if np.isfinite(sm[iLa]) else np.nan,
                     "resid_LaCe": (sm[iLa] - sm[iCe]) - (pred[iLa] - pred[iCe]) if np.isfinite(sm[iLa]) else np.nan})
    L = pd.DataFrame(rows)
    print(L.round(3).to_string(index=False))
    pl = L.groupby("pub")[["obs_CePr", "pred_CePr_from_level", "resid", "kappa_ref"]].mean()
    print(f"\nper publication ({len(pl)}): obs Ce/Pr negative {int((pl.obs_CePr < 0).sum())}; level-predicted negative "
          f"{int((pl.pred_CePr_from_level < 0).sum())}; residual negative {int((pl.resid < 0).sum())} "
          f"(one-sided sign-flip p {signflip(pl.resid.to_numpy(), 'one'):.4f}); kappa_ref > 0 in {int((pl.kappa_ref > 0).sum())}")
    print(f"mean obs {pl.obs_CePr.mean():.3f}, mean level-predicted {pl.pred_CePr_from_level.mean():.3f}, mean residual {pl.resid.mean():.3f}")
    plnc = pl.drop(C5, errors="ignore")
    print(f"no C5 ({len(plnc)}): residual negative {int((plnc.resid < 0).sum())}, one-sided p {signflip(plnc.resid.to_numpy(), 'one'):.4f}, "
          f"mean residual {plnc.resid.mean():.3f}")
    # position-matched control: rank the pair's level within its group; compare Ce/Pr against other pairs in same tercile
    Pq = P.copy()
    Pq["rank"] = Pq.groupby("g").lev.rank(pct=True)
    Pq["terc"] = pd.cut(Pq["rank"], [0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"], include_lowest=True)
    print("\nposition-matched: share of negative slopes (per pair-in-group) by level tercile within group")
    Pq["is"] = np.where(Pq.pair == "Ce/Pr", "Ce/Pr", np.where(Pq.pair.isin(["La/Ce", "Pr/Nd", "Nd/Sm"]), "other light",
                                                              "middle/heavy"))
    print(Pq.groupby(["terc", "is"], observed=True).slope.agg(["count", lambda s: (s < 0).mean(), "mean", "median"]).round(3).to_string())
    # per-publication: in the low tercile, middle/heavy pairs sign
    lowmh = Pq[(Pq.terc == "low") & (Pq["is"] == "middle/heavy")].groupby("pub").slope.mean()
    print(f"low-tercile middle/heavy pairs, per publication: negative {int((lowmh < 0).sum())} of {len(lowmh)}; "
          f"one-sided p {signflip(lowmh.to_numpy(), 'one'):.4f}")
    # direction of DGA log D vs Z: where Ce/Pr sits in the group's ordering
    # ---------------- S5 ----------------
    print("\n" + "=" * 100 + "\nS5: direction vs magnitude (sign of Ce/Pr log SF itself)\n" + "=" * 100)
    q = P[P.pair == "Ce/Pr"]
    print(pd.crosstab(np.sign(q.ybar).rename("sign logSF(Ce/Pr)"), np.sign(q.slope).rename("sign slope")))
    print("groups light-selective at Ce/Pr (ybar > 0):")
    print(q[q.ybar > 0][["pub", "ext", "acid", "ybar", "slope", "nx"]].round(3).to_string(index=False))
    # all pairs: sign(slope) vs sign(ybar) -> 'stretch' consistency
    Pa = P.copy()
    Pa["stretch"] = np.sign(Pa.slope) == np.sign(Pa.ybar)
    print("share of pair slopes with the same sign as the pair's log SF (= |SF| grows with acid), by pair:")
    print(Pa.groupby("pair").stretch.mean().reindex(PN).round(2).to_dict())

    # ---------------- S6 ----------------
    print("\n" + "=" * 100 + "\nS6: Ce-specific or light-end?\n" + "=" * 100)
    Pc = pair_slopes(G, Y1, pairs=[((iLa, iPr), "La/Pr"), ((iPr, iNd), "Pr/Nd"), ((iLa, iNd), "La/Nd")])
    for pn in ("La/Pr", "La/Nd"):
        print(pn, summarise(Pc, pn), "noC5", summarise(Pc[Pc.pub != C5], pn))
    dev = []
    for gi, g in enumerate(G):
        Yg = Y1[g["idx"]]
        d = Yg[:, iCe] - (Yg[:, iLa] + Yg[:, iPr]) / 2
        s = slope(g["x"], d)
        if np.isfinite(s):
            dev.append({"pub": g["pub"], "slope": s})
    dv = pd.DataFrame(dev).groupby("pub").slope.mean()
    print(f"Ce deviation (Ce - mean(La,Pr)) acid slope per publication: negative {int((dv < 0).sum())} of {len(dv)}, "
          f"mean {dv.mean():.3f}, two-sided p {signflip(dv.to_numpy()):.4f}")

    # ---------------- S7 ----------------
    print("\n" + "=" * 100 + "\nS7: placebo, acid levels permuted within group\n" + "=" * 100)
    rng = np.random.default_rng(20261023)
    allneg, negs = 0, []
    R = 500
    gl = [g for g in G]
    for r in range(R):
        Gp = []
        for g in gl:
            gg = dict(g)
            gg["x"] = rng.permutation(g["x"])
            Gp.append(gg)
        Pp = pair_slopes(Gp, Y1, pairs=[((iCe, iPr), "Ce/Pr")])
        s = summarise(Pp)
        negs.append(s["neg"] / max(s["units"], 1))
        allneg += int(s["neg"] == s["units"])
    print(f"placebo: mean share negative {np.mean(negs):.3f}; share of permutations with all publications negative "
          f"{allneg / R:.4f}")

    # ---------------- S8 ----------------
    print("\n" + "=" * 100 + "\nS8: predictive leg on bench groups (leave-one-cell-out, prior from other publications)\n" + "=" * 100)
    pubslope = P[P.pair == "Ce/Pr"].groupby("pub").slope.mean()
    rows = []
    for gi, g in enumerate(G):
        Yg = Y1[g["idx"]]
        yy = Yg[:, iCe] - Yg[:, iPr]
        for i in range(len(yy)):
            if not np.isfinite(yy[i]):
                continue
            oth = np.isfinite(yy) & (np.arange(len(yy)) != i)
            if oth.sum() < 1:
                continue
            acopy = yy[oth].mean()
            dx = g["x"][i] - g["x"][oth].mean()
            sp = pubslope.drop(g["pub"], errors="ignore").mean()
            rows.append({"g": gi, "pub": g["pub"], "y": yy[i], "acopy": acopy, "tilt": acopy + sp * dx,
                         "tilt_half": acopy + 0.5 * sp * dx, "dx": dx})
    E = pd.DataFrame(rows)
    for nm, sub in (("all", E), ("no C5", E[E.pub != C5]), ("|dx| > 0.3", E[E.dx.abs() > 0.3]),
                    ("|dx| > 0.3 no C5", E[(E.dx.abs() > 0.3) & (E.pub != C5)])):
        sa = ((sub.y - sub.acopy) ** 2)
        st = ((sub.y - sub.tilt) ** 2)
        sh = ((sub.y - sub.tilt_half) ** 2)
        per = sub.assign(sa=sa, st=st).groupby("pub")[["sa", "st"]].sum()
        rel = (per.sa - per.st) / per.sa
        print(f"{nm:18s}: rows {len(sub)}, pubs {sub.pub.nunique()}; rel SSE gain TILT_CE vs ACOPY pooled "
              f"{(sa.sum() - st.sum()) / sa.sum():+.3f}; half-tilt {(sa.sum() - sh.sum()) / sa.sum():+.3f}; "
              f"pubs improved {int((rel > 0).sum())}/{len(rel)}; equal-weight mean rel {rel.mean():+.3f}")
        if nm == "no C5":
            print("   per publication rel SSE gain:", rel.round(3).to_dict())


if __name__ == "__main__":
    main()
