#!/usr/bin/env python
"""Skeptic A for idea lab-1: independent recomputation of MX_LM10 vs B1 and the controls the idea did not run.

EXPLORATORY: all 229 C3 cells and the 5 dev seeds have been seen many times.  Nothing here is a test.  Read-only on
every input; writes nothing except stdout.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab1_recompute_skA.py

Own implementation (no import from lab_*.py): quadratic detrend in the PWL_R radius coordinate, shape-vector
interpolation levels = interp(x, x_reads, y_reads - v_reads) + v, v = (1 - w) u + w c, w = n / (n + 6.1),
level match |Y_donor,m - L_B1,m| <= 1.0.  B1 is rebuilt from the fold beta and DOPT_Q[:3] reads and checked against
REF_{design}.parquet; MX_LM10 is checked against the idea's lab_rows2.parquet.

Controls added by Skeptic A (pre-stated here, before running):
  NC05 / NC03   MX_LM10 with near-copy donors removed (offset-removed residual sd on shared metals < 0.05 / 0.03 with
                >= 6 shared metals; this also removes bit-identical series)
  RM            random-mask shrinkage control: per metal, the same NUMBER of donors as level matching keeps, drawn at
                random from the target's own-publication other-extractant donors measured there (30 draws, expected SSE)
                -> does the level SELECTION matter, or only the reduced weight?
  PBC           count- and chemotype-matched other-campaign placebo WITH level matching: every own donor is replaced by a
                random cell of the same chemotype from another publication (any cell if none), then level-matched (30
                draws, expected SSE) -> the proper 'not campaign-specific' control the test design names
  PBA           as PBC but replacement from any other-publication cell
Inference units: publication, laboratory (author-overlap union-find, >= 2 shared surnames, the idea's own rule), and
source cluster (cluster_cpg).  Leave-two-publications-out minimum.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys

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

G21 = r"D:\ml_separator_gh\generations\gen21"
LAB_OUT = G21 + r"\round5_design_20261023\ideas2\lab_out"
XR = np.asarray(interp._coords("R"), float)
K = 6.1
TOL = 1.0
NDRAW = 30
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)


def qdev(y):
    out = np.full(N_LN, np.nan)
    m = np.isfinite(y)
    if m.sum() >= 6:
        co = np.polyfit(XR[m], y[m], 2)
        out[m] = y[m] - np.polyval(co, XR[m])
    return out


def interp_levels(yi, reads, metals, v):
    xr = XR[reads]
    o = np.argsort(xr)
    out = np.full(N_LN, np.nan)
    mm = np.asarray(metals, int)
    out[mm] = np.interp(XR[mm], xr[o], (yi[reads] - v[reads])[o]) + v[mm]
    out[reads] = yi[reads]
    return out


def lm_cw(Dv, Y, donors, lv, tol=TOL, mask_override=None):
    if len(donors) == 0:
        return np.zeros(N_LN), np.zeros(N_LN)
    A = Dv[donors]
    msk = np.isfinite(A) & (np.abs(Y[donors] - lv[None, :]) <= tol) if mask_override is None else mask_override
    n = msk.sum(0).astype(float)
    c = np.where(n > 0, np.where(msk, A, 0.0).sum(0) / np.maximum(n, 1), 0.0)
    return c, n / (n + K)


def surnames(s):
    out = set()
    for a in re.split(r"[;,&]", str(s)):
        toks = [t for t in re.findall(r"[A-Za-zÀ-ÿ\-']+", a) if t.lower().strip(".") not in {"dr", "prof", "mr", "ms", "mrs", "phd"}]
        if toks:
            out.add(toks[-1].lower())
    return out


def lab_map(pubs):
    src = pd.read_csv(fr"{G21}\round4_20261022\results\provenance\sources_by_doi.csv")
    reg = pd.read_csv(fr"{G21}\round5_design_20261023\probes\P4_corpus_registry.csv")
    au = dict(zip(reg.key.astype(str), reg.authors.astype(str)))
    p2s = {}
    for r in src.itertuples(index=False):
        for p in str(r.publication_ids).split(";"):
            p = p.strip()
            if p and str(r.doi) in au:
                p2s[p] = p2s.get(p, set()) | surnames(au[str(r.doi)])
    par = {p: p for p in pubs}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    allp = sorted(p2s)
    for p, q in itertools.combinations(allp, 2):
        if len(p2s[p] & p2s[q]) >= 2:
            par.setdefault(p, p)
            par.setdefault(q, q)
            par[find(p)] = find(q)
    return {p: find(p) for p in pubs}


def sqe(d, a):
    return d[a].to_numpy(float) if a.startswith("E_") else (d[a].to_numpy(float) - d.y.to_numpy(float)) ** 2


def r2(d, a):
    return float(np.mean([1 - sqe(g, a).sum() / ((g.y - g.y.mean()) ** 2).sum() for _, g in d.groupby("split_seed")]))


def rel(d, a, b="B1"):
    return float(1 - sqe(d, a).sum() / sqe(d, b).sum())


def infer(d, a, b="B1", unit="publication"):
    d = d.assign(_a=sqe(d, a), _b=sqe(d, b))
    g = d.groupby(unit)[["_a", "_b"]].sum()
    G = len(g)
    point = 1 - g._a.sum() / g._b.sum()
    jk = np.array([1 - g._a.drop(p).sum() / g._b.drop(p).sum() for p in g.index]) if G > 1 else np.array([point])
    se = float(np.sqrt((G - 1) / G * ((jk - jk.mean()) ** 2).sum())) if G > 2 else np.nan
    q = SS.t.ppf(0.975, G - 1) if G > 2 else np.nan
    q95 = SS.t.ppf(0.95, G - 1) if G > 2 else np.nan
    v = ((g._b - g._a) / 5).to_numpy()
    obs = v.sum()
    if G <= 17:
        sums = np.array([np.dot(s, np.abs(v)) for s in itertools.product((-1, 1), repeat=G)])
    else:
        sums = (np.random.default_rng(0).choice((-1, 1), size=(200000, G)) * np.abs(v)).sum(1)
    p = float((sums >= obs - 1e-12).mean())
    relu = (g._b - g._a) / g._b                       # per-unit relative SSE reduction (equal-weight view)
    return {"rel": point, "lo95": point - q * se, "hi95": point + q * se, "lo90_1s": point - q95 * se, "G": G,
            "up": int((v > 1e-12).sum()), "down": int((v < -1e-12).sum()), "p_sf": p,
            "top3": float(np.sort(v)[::-1][:3].sum() / obs) if obs > 0 else np.nan,
            "jk_min": float(jk.min()), "jk_min_unit": g.index[int(np.argmin(jk))],
            "median_unit_rel": float(relu.median()), "mean_unit_rel": float(relu.mean())}


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    chem = f.chemotype.astype(str).to_numpy()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    Dv = np.array([qdev(Y[i]) for i in range(len(Y))])
    has = np.isfinite(Dv).any(1) & okd
    ar = np.arange(len(f))
    GD = CO.GD
    chem_pool = {c: np.where(has & (chem == c))[0] for c in set(chem)}

    # near-copy matrix (offset-removed residual sd on shared metals, >= 6 shared)
    def offsd(i, j):
        sh = np.isfinite(Y[i]) & np.isfinite(Y[j])
        if sh.sum() < 6:
            return np.inf
        dy = Y[i, sh] - Y[j, sh]
        return float(np.std(dy - dy.mean()))

    rows = []
    for design in ("BPG", "BP"):
        par = pd.read_parquet(paths.PRED / f"REF_{design}.parquet")
        fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = C.load(design, seed, fold)
                u = np.zeros(N_LN)
                u[GD] = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
                prow = par[(par.split_seed == seed) & (par.fold == fold)]
                pcg = {c: g for c, g in prow.groupby("cell_id")}
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if t not in pcg:
                        continue
                    i = pos[t]
                    yi = Y[i]
                    metals = list(cell["metals"])
                    reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                    dX = np.where(has & (ar != i) & (pub == pub[i]) & (ext != ext[i]))[0]
                    L = {"B1": interp_levels(yi, reads, metals, u)}
                    lv = L["B1"]

                    def lev(cw):
                        c, w = cw
                        return interp_levels(yi, reads, metals, (1 - w) * u + w * c)
                    E = {}
                    if len(dX) and t not in v2drop:
                        L["LM"] = lev(lm_cw(Dv, Y, dX, lv))
                        osd = np.array([offsd(i, j) for j in dX])
                        L["NC05"] = lev(lm_cw(Dv, Y, dX[osd >= 0.05], lv))
                        L["NC03"] = lev(lm_cw(Dv, Y, dX[osd >= 0.03], lv))
                        ncflag05, ncflag03 = bool((osd < 0.05).any()), bool((osd < 0.03).any())
                        if design == "BPG":
                            rng = np.random.default_rng(int(hashlib.sha256(f"{t}|{seed}".encode()).hexdigest()[:8], 16))
                            A = Dv[dX]
                            fin = np.isfinite(A)
                            close = fin & (np.abs(Y[dX] - lv[None, :]) <= TOL)
                            nsel = close.sum(0)
                            RM, PBC, PBA = [], [], []
                            pool_any = np.where(has & (pub != pub[i]))[0]
                            for _ in range(NDRAW):
                                msk = np.zeros_like(fin)
                                for m in range(N_LN):
                                    cand = np.where(fin[:, m])[0]
                                    if nsel[m] > 0:
                                        msk[rng.choice(cand, nsel[m], replace=False), m] = True
                                RM.append(lev(lm_cw(Dv, Y, dX, lv, mask_override=msk)))
                                rep = []
                                for j in dX:
                                    pc = np.setdiff1d(chem_pool[chem[j]][pub[chem_pool[chem[j]]] != pub[i]], rep)
                                    pa = np.setdiff1d(pool_any, rep)
                                    rep.append(int(rng.choice(pc if len(pc) else pa)))
                                PBC.append(lev(lm_cw(Dv, Y, np.asarray(rep, int), lv)))
                                PBA.append(lev(lm_cw(Dv, Y, rng.choice(pool_any, len(dX), replace=False), lv)))
                            E = {"E_RM": RM, "E_PBC": PBC, "E_PBA": PBA}
                    else:
                        ncflag05 = ncflag03 = False
                    for r in pcg[t].itertuples(index=False):
                        ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                        o = {"design": design, "split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i],
                             "extractant": ext[i], "chemotype": chem[i], "pair": f"{r.A}/{r.B}", "y": r.y, "B1_par": r.B1,
                             "popx": bool(len(dX) > 0 and t not in v2drop), "nc05": ncflag05, "nc03": ncflag03}
                        for k2, l2 in L.items():
                            o[k2] = l2[ia] - l2[ib]
                        for k2, lst in E.items():
                            o[k2] = float(np.mean([(l2[ia] - l2[ib] - r.y) ** 2 for l2 in lst]))
                        rows.append(o)
        print(f"{design} built", flush=True)
    T = pd.DataFrame(rows)
    print("max |B1(own) - B1(REF parquet)|:", float((T.B1 - T.B1_par).abs().max()))
    # cross-check against the idea's lab_rows2.parquet
    L2 = pd.read_parquet(LAB_OUT + r"\lab_rows2.parquet")
    L1 = pd.read_parquet(LAB_OUT + r"\lab_rows.parquet")[["design", "split_seed", "cell_id", "pair", "n_MS"]]
    key = ["design", "split_seed", "cell_id", "pair"]
    mm = T.merge(L2[key + ["MX_LM10"]], on=key, how="left", validate="one_to_one").merge(L1, on=key, how="left",
                                                                                         validate="one_to_one")
    P = mm[mm.popx]
    print("max |LM(own) - MX_LM10(idea)| on POPX rows:", float((P.LM - P.MX_LM10).abs().max()), "rows", len(P))
    T = mm
    T = M4.add_clusters(T, f)
    lm = lab_map(sorted(set(T.publication)))
    T["lab"] = T.publication.map(lm)
    for design in ("BPG", "BP"):
        d = T[(T.design == design) & T.popx]
        d0 = d[d.n_MS == 0]
        print(f"\n================ {design} ================")
        for nm, s in (("POPX", d), ("POPX0", d0)):
            print(f"{nm}: cells {s.cell_id.nunique()}, pubs {s.publication.nunique()}, labs {s.lab.nunique()}, "
                  f"clusters {s.cluster_cpg.nunique()}, rows/seed {len(s) / 5:.0f}; B1 {r2(s, 'B1'):.4f} LM {r2(s, 'LM'):.4f} "
                  f"(rel {rel(s, 'LM'):+.4f}); NC05 {rel(s, 'NC05'):+.4f}; NC03 {rel(s, 'NC03'):+.4f}")
        if design == "BP":
            continue
        print("labs among POPX pubs:", {k: sorted(v) for k, v in d.groupby("lab").publication.unique().to_dict().items()
                                        if len(v) > 1})
        for nm, s in (("POPX", d), ("POPX0", d0)):
            print(f"\n--- {nm} arms (BPG) ---")
            print(pd.DataFrame([{"arm": a, "R2": r2(s, a), "rel_vs_B1": rel(s, a)} for a in
                                ("B1", "LM", "NC05", "NC03", "E_RM", "E_PBC", "E_PBA")]).round(4).to_string(index=False))
            out = []
            for a, b in (("LM", "B1"), ("NC05", "B1"), ("NC03", "B1"), ("LM", "E_RM"), ("LM", "E_PBC"), ("LM", "E_PBA"),
                         ("E_PBC", "B1")):
                for unit in ("publication", "lab", "cluster_cpg"):
                    r = infer(s, a, b, unit)
                    out.append({"contrast": f"{a}-{b}", "unit": unit, **r})
            print(pd.DataFrame(out).round(4).to_string(index=False))
            # without near-copy targets
            s2 = s[~s.nc05]
            print(f"   {nm} dropping targets that have a near-copy donor (offset sd < 0.05): cells {s2.cell_id.nunique()}, "
                  f"pubs {s2.publication.nunique()}: LM rel {rel(s2, 'LM'):+.4f}; NC05 rel {rel(s2, 'NC05'):+.4f}")
            r = infer(s2, "LM", "B1")
            print(f"      LM-B1 pub-unit: [{r['lo95']:+.3f}, {r['hi95']:+.3f}] up/down {r['up']}/{r['down']} p {r['p_sf']:.3f} "
                  f"median unit {r['median_unit_rel']:+.3f}")
            # leave-two-publications-out
            g = s.assign(_a=sqe(s, "LM"), _b=sqe(s, "B1")).groupby("publication")[["_a", "_b"]].sum()
            l2o = [(p, q, 1 - g._a.drop([p, q]).sum() / g._b.drop([p, q]).sum()) for p, q in itertools.combinations(g.index, 2)]
            l2o.sort(key=lambda z: z[2])
            print(f"   leave-two-pubs-out min: {l2o[0][2]:+.4f} without {l2o[0][0]}, {l2o[0][1]}; next {l2o[1][2]:+.4f}")
            for drop in (["pub_1bd76c48d9"], ["pub_1bd76c48d9", "pub_5a68dc5665"], ["pub_1bd76c48d9", "pub_5a68dc5665", "pub_126bbe7976"],
                         ["pub_5a68dc5665", "pub_4cba27f5f3", "pub_b9dfc468cd"]):
                s3 = s[~s.publication.isin(drop)]
                r = infer(s3, "LM", "B1")
                print(f"   without {drop}: LM rel {r['rel']:+.4f} [{r['lo95']:+.3f}, {r['hi95']:+.3f}] up/down {r['up']}/{r['down']}"
                      f" p {r['p_sf']:.3f}; vs E_PBC {rel(s3, 'LM', 'E_PBC'):+.4f}")
            # per-publication and per-lab view
            if nm == "POPX":
                rec = []
                for p, gg in s.groupby("publication"):
                    rec.append({"pub": p, "lab": gg.lab.iloc[0][-10:], "cells": gg.cell_id.nunique(), "nc05_cells":
                                gg[gg.nc05].cell_id.nunique(), "dSSE_LM/seed": float((sqe(gg, "B1") - sqe(gg, "LM")).sum() / 5),
                                "rel_LM": rel(gg, "LM"), "rel_NC05": rel(gg, "NC05"), "rel_RM": rel(gg, "E_RM"),
                                "rel_PBC": rel(gg, "E_PBC"), "LM_vs_PBC": rel(gg, "LM", "E_PBC")})
                print(pd.DataFrame(rec).sort_values("dSSE_LM/seed", ascending=False).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
