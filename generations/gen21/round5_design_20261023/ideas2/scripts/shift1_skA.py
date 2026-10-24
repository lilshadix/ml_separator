#!/usr/bin/env python
"""Skeptic A for idea shift-1 (acid-distance rule for zero-read condition transfer).  EXPLORATORY: every number is
on the 229 already-seen C3 cells and the 5 dev seeds.

Independent recomputation (own code; does not import shift_build / shift_analyze):
  * acid-only CLEAN siblings rebuilt from the V1 bench + round-3 V2 lists (own condition-diff code);
  * ACOPY_0 template, dx_mean, dx_near rebuilt; B1@4 and y taken straight from round-4 REF_{BPG,BP}.parquet;
  * SH@2 / SH@4 taken from shift_rows.parquet only after checking ACOPY_0 / y / B1 agree row by row.
Checks: dose table, within-publication permutation statistic (+ LOPO, metric fork dx_near, ratio control,
interpolation/extrapolation split, confounders), near/far per publication with and without C5, near-side against
SH@2, near-copies / bit-identical siblings, V2-5 floor cells, and a power simulation of the proposed external test.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift1_skA.py
Writes only ideas2/scripts/shift1_skA_out/.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift1_skA_out")
SHIFT_ROWS = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts\shift_out\shift_rows.parquet")
C5 = "pub_6a0ef8528c"
THR = 0.30
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def family(col: str) -> str:
    if col == "cond__acid_concentration_M":
        return "acid_conc"
    if col.startswith("cond__acid__"):
        return "acid_type"
    if col.startswith("cond__diluent__"):
        return "diluent"
    if col.startswith("cond__additive__"):
        return "additive"
    return col.replace("cond__", "")


def r2_seedmean(d, col):
    v = []
    for _, g in d.groupby("split_seed"):
        y = g.y.to_numpy(float)
        v.append(1 - ((y - g[col].to_numpy(float)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(v))


def _centred_ranks(df, col, unit):
    out = np.zeros(len(df))
    u = df[unit].to_numpy()
    v = df[col].to_numpy(float)
    for g in pd.unique(u):
        ix = np.where(u == g)[0]
        out[ix] = rankdata(v[ix]) - (len(ix) + 1) / 2
    return out


def within_unit_stat(df, xcol, ecol, unit="pub"):
    """Pooled correlation of within-unit centred ranks (the idea's statistic)."""
    ra = _centred_ranks(df, xcol, unit)
    rs = _centred_ranks(df, ecol, unit)
    den = np.sqrt((ra ** 2).sum() * (rs ** 2).sum())
    return float((ra * rs).sum() / den) if den > 0 else np.nan


def perm_p(df, xcol, ecol, unit="pub", n=4000, seed=11):
    """Within-unit permutation of the distance; one-sided (stat >= observed), (count + 1) / (n + 1)."""
    ra = _centred_ranks(df, xcol, unit)
    rs = _centred_ranks(df, ecol, unit)
    den = np.sqrt((ra ** 2).sum() * (rs ** 2).sum())
    if den == 0:
        return np.nan, np.nan
    obs = float((ra * rs).sum() / den)
    rng = np.random.default_rng(seed)
    u = df[unit].to_numpy()
    tot = np.zeros(n)
    for g in pd.unique(u):
        ix = np.where(u == g)[0]
        perm = np.argsort(rng.random((n, len(ix))), axis=1)
        tot += ra[ix][perm] @ rs[ix]
    null = tot / den
    return obs, float(((null >= obs - 1e-12).sum() + 1) / (n + 1))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).to_numpy()
    pos = {c: i for i, c in enumerate(ids)}
    cond = [c for c in f.columns if c.startswith("cond__")]
    C = f[cond].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    fams = np.array([family(c) for c in cond])
    pub = f["publication_id"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    x = np.log10(f["cond__acid_concentration_M"].astype(float).to_numpy())
    drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    floor = set(V2.V2_5_FLOOR)

    def diff(i, j):
        a, b = C[i], C[j]
        na, nb = np.isnan(a), np.isnan(b)
        dif = (na != nb) | (~na & ~nb & ~np.isclose(a, b))
        return set(fams[dif])

    ref = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in ("BPG", "BP")}
    c3 = sorted(set(ref["BPG"].cell_id))

    # ---------- acid-only CLEAN siblings, rebuilt -------------------------------------------------------------
    info, dup_rows = {}, []
    for t in c3:
        i = pos[t]
        if t in drop or not np.isfinite(x[i]):
            continue
        met = np.isfinite(Y[i])
        AS = []
        for j in np.where((pub == pub[i]) & (ext == ext[i]))[0]:
            if j == i or ids[j] in drop or not np.all(np.isfinite(Y[j, met])) or not np.isfinite(x[j]):
                continue
            core = diff(i, j) - {"contact_time", "metal_conc"}
            if core == {"acid_conc"}:
                AS.append(j)
                dY = Y[i, met] - Y[j, met]
                dup_rows.append({"target": t, "sib": ids[j], "pub": pub[i], "dx": x[i] - x[j],
                                 "n_met": int(met.sum()), "n_identical": int((np.abs(dY) < 1e-9).sum()),
                                 "rms_dlogD": float(np.sqrt(np.mean(dY ** 2))),
                                 "rms_dSF": float(np.sqrt(np.nanmean(np.diff(dY[np.isfinite(dY)]) ** 2)))})
        if not AS:
            continue
        xa = x[AS]
        info[t] = {"T": Y[AS].mean(0), "n_AS": len(AS), "dx_mean": float(x[i] - xa.mean()),
                   "dx_near": float(np.min(np.abs(xa - x[i]))),
                   "extrap": bool(x[i] < xa.min() - 1e-9 or x[i] > xa.max() + 1e-9),
                   "below": bool(x[i] < xa.min() - 1e-9), "x_t": float(x[i]), "pub": pub[i], "ext": ext[i],
                   "floor_t": t in floor, "floor_sib": any(ids[j] in floor for j in AS),
                   "sibs": [ids[j] for j in AS]}
    say(f"[rebuild] targets with >= 1 acid-only CLEAN sibling: {len(info)} "
        f"(without C5 {sum(v['pub'] != C5 for v in info.values())})")

    # ---------- rows ----------------------------------------------------------------------------------------
    S = pd.read_parquet(SHIFT_ROWS)
    allrows = []
    for d, r in ref.items():
        r = r[r.cell_id.isin(info)].copy()
        r["design"] = d
        r["ACOPY0"] = [info[c]["T"][LANTHANIDES.index(a)] - info[c]["T"][LANTHANIDES.index(b)]
                       for c, a, b in zip(r.cell_id, r.A, r.B)]
        for k in ("dx_mean", "dx_near", "extrap", "below", "n_AS", "x_t", "floor_t", "floor_sib"):
            r[k] = r.cell_id.map({c: v[k] for c, v in info.items()})
        r["pub"] = r.publication.astype(str)
        r["c5"] = r.pub == C5
        allrows.append(r[["design", "split_seed", "fold", "cell_id", "pub", "extractant", "chemotype", "A", "B", "y",
                          "B1", "ACOPY0", "dx_mean", "dx_near", "extrap", "below", "n_AS", "x_t", "c5", "floor_t",
                          "floor_sib"]])
    R = pd.concat(allrows, ignore_index=True)
    Sm = S[S.n_AS >= 1][["design", "split_seed", "cell_id", "A", "B", "y", "B1_3", "ACOPY_0", "SH_1", "SH_2", "SH_3",
                         "dx_mean", "COPY"]]
    M = R.merge(Sm, on=["design", "split_seed", "cell_id", "A", "B"], how="outer", indicator=True, suffixes=("", "_s"))
    say(f"[match] merge: {M._merge.value_counts().to_dict()}")
    M = M[M._merge == "both"].copy()
    say(f"[match] max|ACOPY0 - shift ACOPY_0| {np.abs(M.ACOPY0 - M.ACOPY_0).max():.2e}; max|y-y_s| "
        f"{np.abs(M.y - M.y_s).max():.2e}; max|B1-B1_3| {np.abs(M.B1 - M.B1_3).max():.2e}; "
        f"max|dx_mean diff| {np.abs(M.dx_mean - M.dx_mean_s).max():.2e}")
    M["adx"] = M.dx_mean.abs()
    M["adn"] = M.dx_near.abs()

    # ---------- A. dose table ------------------------------------------------------------------------------
    for design in ("BPG", "BP"):
        for nm, d in (("no C5", M[(M.design == design) & ~M.c5]), ("with C5", M[M.design == design])):
            for col in ("adx", "adn"):
                b = pd.cut(d[col], [-0.01, 0.15, 0.3, 0.6, 5.0])
                tab = d.assign(bin=b).groupby("bin", observed=True).apply(lambda g: pd.Series({
                    "cells": g.cell_id.nunique(), "pubs": g.pub.nunique(),
                    "MSE_ACOPY0": ((g.y - g.ACOPY0) ** 2).mean(), "MSE_B1@4": ((g.y - g.B1) ** 2).mean(),
                    "MSE_SH@2": ((g.y - g.SH_1) ** 2).mean(), "MSE_SH@4": ((g.y - g.SH_3) ** 2).mean(),
                    "var_y": g.y.var()}), include_groups=False)
                if design == "BPG" or col == "adx":
                    say(f"\n[A dose] {design} {nm} by {col}\n{tab.round(4).to_string()}")

    # ---------- B. within-publication permutation statistic ---------------------------------------------------
    def cell_table(d):
        e = d.assign(se0=(d.y - d.ACOPY0) ** 2, seB=(d.y - d.B1) ** 2, seS=(d.y - d.SH_3) ** 2, y2=d.y ** 2)
        c = e.groupby("cell_id").agg(pub=("pub", "first"), ext=("extractant", "first"), adx=("adx", "first"),
                                     adn=("adn", "first"), dx=("dx_mean", "first"), extrap=("extrap", "first"),
                                     below=("below", "first"), n_AS=("n_AS", "first"), x_t=("x_t", "first"),
                                     se0=("se0", "mean"), seB=("seB", "mean"), seS=("seS", "mean"),
                                     vy=("y", "var"), my2=("y2", "mean"), rows=("y", "size"), c5=("c5", "first"),
                                     floor_t=("floor_t", "first"), floor_sib=("floor_sib", "first"))
        c["ratio"] = np.log(c.se0 / c.seB)
        return c

    def eligible(c, unit="pub", xcol="adx"):
        return c.groupby(unit).filter(lambda g: len(g) >= 3 and g[xcol].nunique() >= 2)

    res = {}
    for design in ("BPG", "BP"):
        for c5 in (False, True):
            d = M[(M.design == design) & ((~M.c5) | c5)]
            c = cell_table(d)
            say(f"\n[B within-pub] {design} with C5={c5}: cells {len(c)}")
            for xcol in ("adx", "adn"):
                e = eligible(c, xcol=xcol)
                for ecol in ("se0", "seB", "seS", "ratio", "vy", "my2", "n_AS"):
                    if design == "BP" and ecol not in ("se0", "seB"):
                        continue
                    obs, p = perm_p(e, xcol, ecol)
                    res[(design, c5, xcol, ecol)] = (obs, p, e.pub.nunique(), len(e))
                    say(f"   dist={xcol} err={ecol:5s}: stat {obs:+.3f}, one-sided perm p {p:.4f} "
                        f"({e.pub.nunique()} pubs, {len(e)} cells)")

    # LOPO / per-pub Spearman (BPG, no C5, adx)
    d = M[(M.design == "BPG") & ~M.c5]
    c = cell_table(d)
    e = eligible(c)
    say("\n[B2 per-publication Spearman(|dx_mean|, cell MSE ACOPY0) and LOPO of the pooled statistic; BPG no C5]")
    rows = []
    for p_, g in e.groupby("pub"):
        rs = spearmanr(g.adx, g.se0)[0]
        rb = spearmanr(g.adx, g.seB)[0]
        rest = e[e.pub != p_]
        o, pp = perm_p(rest, "adx", "se0", n=2000)
        rows.append({"pub": p_, "cells": len(g), "n_ext": g.ext.nunique(), "rho_tmpl": rs, "rho_B1": rb,
                     "LOPO_stat": o, "LOPO_p": pp})
    T = pd.DataFrame(rows)
    say(T.round(3).to_string(index=False))
    say(f"   equal-weight mean per-pub Spearman: template {T.rho_tmpl.mean():+.3f} (positive {int((T.rho_tmpl > 0).sum())}"
        f"/{len(T)}), B1 {T.rho_B1.mean():+.3f}")
    # leave-two-out worst case
    worst = []
    for a, b in itertools.combinations(sorted(e.pub.unique()), 2):
        worst.append((within_unit_stat(e[~e.pub.isin([a, b])], "adx", "se0"), a, b))
    worst.sort()
    say(f"   leave-two-pubs-out minimum stat {worst[0][0]:+.3f} (without {worst[0][1]}, {worst[0][2]}); "
        f"share of L2O stats <= 0.10: {np.mean([w[0] <= 0.10 for w in worst]):.2f}")

    # within pub x extractant units
    e2 = c.copy()
    e2["unit"] = e2.pub + "|" + e2.ext
    e2 = eligible(e2, unit="unit")
    o, p = perm_p(e2, "adx", "se0", unit="unit")
    o2, p2 = perm_p(e2, "adx", "seB", unit="unit")
    say(f"   within pub x extractant units: template {o:+.3f} (p {p:.4f}), B1 {o2:+.3f} (p {p2:.4f}); "
        f"{e2.unit.nunique()} units, {len(e2)} cells")

    # ---------- C. interpolation vs extrapolation ---------------------------------------------------------------
    say("\n[C] inside vs outside the siblings' acid range (BPG no C5)")
    for nm, sub in (("inside", c[~c.extrap]), ("outside", c[c.extrap]), ("outside-above", c[c.extrap & ~c.below]),
                    ("outside-below", c[c.below])):
        say(f"   {nm:14s}: cells {len(sub)}, pubs {sub.pub.nunique()}, mean |dx| {sub.adx.mean():.3f}, "
            f"mean cell MSE ACOPY0 {sub.se0.mean():.4f}, B1@4 {sub.seB.mean():.4f}, SH@4 {sub.seS.mean():.4f}")
    for nm, sub in (("inside only", c[~c.extrap]), ("outside only", c[c.extrap])):
        ee = eligible(sub)
        if len(ee):
            o, p = perm_p(ee, "adx", "se0")
            say(f"   within-pub stat restricted to {nm}: {o:+.3f} (p {p:.3f}; {ee.pub.nunique()} pubs, {len(ee)} cells)")
    # Does |dx| add beyond the extrapolation flag? residualise rank of se0 within pub on extrap
    ee = eligible(c).copy()
    ee["extr_i"] = ee.extrap.astype(float)
    obs_ex, p_ex = perm_p(ee, "extr_i", "se0")
    say(f"   within-pub stat with the binary extrapolation flag as the 'distance': {obs_ex:+.3f} (p {p_ex:.3f})")
    # below-range share among far cells
    far = c[c.adx > THR]
    say(f"   far cells (> {THR}): {len(far)}, below-range {int(far.below.sum())}, above-range "
        f"{int((far.extrap & ~far.below).sum())}, inside {int((~far.extrap).sum())}")
    near = c[c.adx <= THR]
    say(f"   near cells (<= {THR}): {len(near)}, below-range {int(near.below.sum())}, above "
        f"{int((near.extrap & ~near.below).sum())}, inside {int((~near.extrap).sum())}")

    # ---------- D. near/far per publication ----------------------------------------------------------------------
    for c5 in (False, True):
        d = M[(M.design == "BPG") & ((~M.c5) | c5)]
        d = d.assign(near=d.adx <= THR + 1e-9)
        g = d.groupby(["pub", "near"]).apply(lambda q: pd.Series({"cells": q.cell_id.nunique(),
                                                                     "mse0": ((q.y - q.ACOPY0) ** 2).mean()}),
                                             include_groups=False).unstack("near")
        both = g.dropna()
        k = int((both[("mse0", True)] < both[("mse0", False)]).sum())
        say(f"\n[D] with C5={c5}: publications with both near and far cells: {len(both)}; near MSE lower in {k}")
        say(both.round(4).to_string())

    # ---------- E. near side against the cheapest sensible alternative ----------------------------------------
    say("\n[E] factor-of-2 split, BPG no C5: copy@0 against B1@4, SH@2, SH@4 (R2 pooled within seed, seed mean)")
    for design in ("BPG", "BP"):
        d = M[(M.design == design) & ~M.c5]
        for nm, sub in (("near", d[d.adx <= THR + 1e-9]), ("far", d[d.adx > THR + 1e-9])):
            pubrel = {}
            for p_, q in sub.groupby("pub"):
                pubrel[p_] = 1 - ((q.y - q.ACOPY0) ** 2).sum() / ((q.y - q.SH_1) ** 2).sum()
            say(f"   {design} {nm}: cells {sub.cell_id.nunique()}, pubs {sub.pub.nunique()}; ACOPY0 "
                f"{r2_seedmean(sub, 'ACOPY0'):.3f}, B1@4 {r2_seedmean(sub, 'B1'):.3f}, SH@2 {r2_seedmean(sub, 'SH_1'):.3f}, "
                f"SH@3 {r2_seedmean(sub, 'SH_2'):.3f}, SH@4 {r2_seedmean(sub, 'SH_3'):.3f}; eq-w pub relSSE ACOPY0 vs SH@2 "
                f"{np.mean(list(pubrel.values())):+.3f} ({sum(v > 0 for v in pubrel.values())}/{len(pubrel)} +)")
        # dx_near version of the split
        for nm, sub in (("near(dx_near)", d[d.adn <= THR + 1e-9]), ("far(dx_near)", d[d.adn > THR + 1e-9])):
            if design == "BPG":
                say(f"   {design} {nm}: cells {sub.cell_id.nunique()}, pubs {sub.pub.nunique()}; ACOPY0 "
                    f"{r2_seedmean(sub, 'ACOPY0'):.3f}, B1@4 {r2_seedmean(sub, 'B1'):.3f}, SH@4 {r2_seedmean(sub, 'SH_3'):.3f}")
    # near side leave-one-pub-out for copy@0 - B1@4
    d = M[(M.design == "BPG") & ~M.c5]
    nr = d[d.adx <= THR + 1e-9]
    lo = []
    for p_ in sorted(nr.pub.unique()):
        q = nr[nr.pub != p_]
        lo.append((r2_seedmean(q, "ACOPY0") - r2_seedmean(q, "B1"), p_))
    say(f"   near side copy@0 - B1@4 LOPO values: " + ", ".join(f"{p_[-6:]} {v:+.3f}" for v, p_ in sorted(lo)))
    shares = nr.assign(dse=(nr.y - nr.B1) ** 2 - (nr.y - nr.ACOPY0) ** 2).groupby("pub").dse.sum()
    say(f"   near side per-pub SSE gain of copy@0 over B1@4 (sum over seeds): "
        + ", ".join(f"{k[-6:]} {v:+.3f}" for k, v in shares.sort_values().items()))

    # ---------- F. near-copies, bit-identical siblings, floor cells ----------------------------------------------
    DU = pd.DataFrame(dup_rows)
    DU = DU[DU.target.isin(info)]
    say(f"\n[F] target-sibling pairs (acid-only): {len(DU)}; with any bit-identical metal: "
        f"{int((DU.n_identical > 0).sum())}; with >= half metals identical: {int((DU.n_identical >= DU.n_met / 2).sum())}; "
        f"|dx| < 0.02 decade: {int((DU.dx.abs() < 0.02).sum())}; rms dlogD < 0.02: {int((DU.rms_dlogD < 0.02).sum())}")
    say(DU.sort_values("rms_dlogD").head(12).round(4).to_string(index=False))
    sus = DU[(DU.n_identical >= 2) | (DU.rms_dlogD < 0.02) | (DU.dx.abs() < 0.02)]
    say(f"   suspicious pairs: {len(sus)}; targets involved: {sus.target.nunique()}")
    if len(sus):
        say(sus.round(4).to_string(index=False))
    c = cell_table(M[(M.design == "BPG") & ~M.c5])
    say(f"   V2-5 floor: targets {int(c.floor_t.sum())}, targets with a floor sibling {int(c.floor_sib.sum())}")
    cf = eligible(c[~c.floor_t & ~c.floor_sib])
    if len(cf):
        o, p = perm_p(cf, "adx", "se0")
        say(f"   within-pub stat without floor targets / floor-sibling targets: {o:+.3f} (p {p:.3f}; {cf.pub.nunique()} pubs, {len(cf)} cells)")
    # exclude suspicious targets
    if len(sus):
        cs = eligible(c[~c.index.isin(set(sus.target))])
        o, p = perm_p(cs, "adx", "se0")
        say(f"   within-pub stat without suspicious targets: {o:+.3f} (p {p:.3f})")

    # ---------- G. power of the proposed external test (in-corpus effect as truth) ---------------------------------
    # families = connected components of the acid-only sibling relation (pub x extractant x fixed other conditions),
    # >= 3 distinct acid levels spanning >= 0.5 decade, V2-clean, no C5.  A simulated external family = 3 or 4
    # levels drawn from one corpus family (span >= 0.5); leave-one-level-out mean template; cell MSE over all strict
    # adjacent pairs finite in every member.  Statistic and permutation test as proposed; kill (i): stat <= 0.10 or p > 0.10.
    comp, seen = [], set()
    cand = [i for i in range(len(f)) if ids[i] not in drop and pub[i] != C5 and np.isfinite(x[i])
            and np.isfinite(Y[i]).sum() >= 5]
    adj = {i: set() for i in cand}
    for i, j in itertools.combinations(cand, 2):
        if pub[i] == pub[j] and ext[i] == ext[j] and (diff(i, j) - {"contact_time", "metal_conc"}) == {"acid_conc"}:
            adj[i].add(j)
            adj[j].add(i)
    for i in cand:
        if i in seen or not adj[i]:
            continue
        st, cc = [i], set()
        while st:
            u = st.pop()
            if u in cc:
                continue
            cc.add(u)
            st.extend(adj[u] - cc)
        seen |= cc
        cc = sorted(cc)
        met = np.all(np.isfinite(Y[cc]), axis=0)
        lv = np.unique(np.round(x[cc], 3))
        if met.sum() >= 5 and len(lv) >= 3 and np.ptp(lv) >= 0.5:
            comp.append({"pub": pub[i], "ext": ext[i], "cells": cc, "met": np.where(met)[0]})
    say(f"\n[G] eligible corpus acid-series families (no C5, >= 3 levels spanning >= 0.5 dec): {len(comp)} from "
        f"{len({c_['pub'] for c_ in comp})} pubs; sizes {[len(c_['cells']) for c_ in comp]}")

    def fam_cells(cf, m, rng):
        cc = np.array(cf["cells"])
        for _ in range(50):
            pick = rng.choice(len(cc), size=min(m, len(cc)), replace=False)
            sel = cc[pick]
            if np.ptp(x[sel]) >= 0.5 and len(np.unique(np.round(x[sel], 3))) == len(sel):
                break
        else:
            return None
        met = cf["met"]
        out = []
        for a in sel:
            others = [b for b in sel if b != a]
            T_ = Y[others][:, met].mean(0)
            yy = Y[a, met]
            pairs = [(q, q + 1) for q in range(len(met) - 1) if met[q + 1] == met[q] + 1]
            if not pairs:
                return None
            se = np.mean([((yy[p] - yy[q]) - (T_[p] - T_[q])) ** 2 for p, q in pairs])
            out.append((abs(x[a] - x[others].mean()), se))
        return out

    rng = np.random.default_rng(2026)
    for nfam, m in ((8, 3), (8, 4), (12, 3), (16, 3)):
        kills, stats = 0, []
        for it in range(300):
            rows_ = []
            draw = rng.choice(len(comp), size=nfam, replace=True)
            for u, ci in enumerate(draw):
                r_ = fam_cells(comp[ci], m, rng)
                if r_ is None:
                    continue
                rows_ += [{"pub": f"u{u}", "adx": a, "se0": s} for a, s in r_]
            df = pd.DataFrame(rows_)
            df = df.groupby("pub").filter(lambda g: len(g) >= 3 and g.adx.nunique() >= 2)
            if df.pub.nunique() < 2:
                kills += 1
                continue
            o, p = perm_p(df, "adx", "se0", n=300, seed=it)
            stats.append(o)
            kills += (o <= 0.10) or (p > 0.10)
        say(f"   {nfam} families x {m} levels: P(kill (i)) = {kills / 300:.2f}; median stat {np.median(stats):+.3f}")
    # same simulation with the stated families but whole corpus families (all levels) for reference
    rows_ = []
    for u, cf in enumerate(comp):
        r_ = fam_cells(cf, len(cf["cells"]), np.random.default_rng(u))
        if r_:
            rows_ += [{"pub": f"u{u}", "adx": a, "se0": s} for a, s in r_]
    df = pd.DataFrame(rows_).groupby("pub").filter(lambda g: len(g) >= 3 and g.adx.nunique() >= 2)
    o, p = perm_p(df, "adx", "se0", n=2000)
    say(f"   all eligible corpus families, all levels, leave-one-level-out (label-free re-derivation of the gradient): "
        f"stat {o:+.3f}, p {p:.4f} ({df.pub.nunique()} families, {len(df)} cells)")

    (OUT / "shift1_skA.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
