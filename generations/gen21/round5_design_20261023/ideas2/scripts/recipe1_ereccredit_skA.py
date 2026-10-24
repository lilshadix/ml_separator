#!/usr/bin/env python
"""Skeptic A for idea recipe-1 (E-REC: reads-equivalent of an own prior series).

EXPLORATORY: every number is computed on the 229 already-seen C3 cells of the V1 bench.  Nothing here is a test.

Independent re-implementation (does not import or read any recipe_* script or output):
  - own / xext / xpub / chem pools from metadata + metal coverage (CLEAN rule: V2-1/2/3 out, sibling must differ
    from the target in a recorded cond__ field other than contact time / metal concentration);
  - training-free arms D7 = u + PWL_R[y_S - u_S] (u = HIST_D7 beta at Gd), SIB = T + PWL_R[y_S - T_S],
    SH = w SIB + (1 - w) D7, w = n/(n + K); COPY = T at 0 reads;
  - read policies HULL (lightest + heaviest, then max-min distance in -CN8 radius, ties -> lower Z), DOPT_Q-nested
    (gen21 fold cache, BPG, 5 dev seeds, seed mean), fixed ladder (La,Ho / La,Nd,Ho / La,Nd,Er,Lu / La,Nd,Gd,Er,Lu,
    HULL fallback);
  - whole-series pooled R2 over all strict-adjacent pairs of the population's cells (measured pairs score 0 error).
Extra checks: HIST_D7 reproduction on round-4 DOPT_Q 4-read sets, bit-identical / SF-identical / shape-sd near-copy
links, C5-BTBP, LOPO / top-k publication removal, per-publication and per-family contributions, family sign-flip,
K sensitivity, comparator policy (HULL vs DOPT_Q vs ladder), xpub weighting sensitivity, per-family power sketch.

Writes only ideas2/recipe1_skA_out/.
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe1_ereccredit_skA.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4.metrics import add_clusters  # noqa: E402
from r4.refs import HIST_D7_BETA  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe1_skA_out")
C5 = "pub_6a0ef8528c"
GD = LANTHANIDES.index("Gd")
X = np.asarray(interp.X_R, float)
TRIVIAL = {"cond__contact_time_min", "cond__metal_concentration_mM"}
LADDER = {2: ("La", "Ho"), 3: ("La", "Nd", "Ho"), 4: ("La", "Nd", "Er", "Lu"), 5: ("La", "Nd", "Gd", "Er", "Lu")}
KNOWN_COPIES = [("44012938119033a5", "5169c377777d282b"), ("d4c1e23a091d0960", "e4ee526bda3920c2")]
B = 4000
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def pwl(x, S, r):
    """Piecewise-linear interpolation of residual r through read positions S (constant continuation)."""
    if len(S) == 0:
        return np.zeros_like(x)
    S = sorted(S)
    return np.interp(x, x[S], r[S])


def hull_order(x):
    m = len(x)
    order = [0, m - 1] if m > 1 else [0]
    while len(order) < m:
        rest = [q for q in range(m) if q not in order]
        d = [min(abs(x[q] - x[o]) for o in order) for q in rest]
        best = max(d)
        order.append(min(q for q, dd in zip(rest, d) if dd == best))   # ties -> lower Z (lower index)
    return order


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    assert np.all(np.diff(X) > 0), "X_R must increase along the series"
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    chem = f.chemotype.astype(str).to_numpy()
    condc = [c for c in f.columns if c.startswith("cond__")]
    F = f[condc].to_numpy(float)
    v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okc = np.array([c not in v2 for c in ids])
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = sorted(par.cell_id.unique())
    cl = add_clusters(par[["cell_id", "chemotype", "publication"]].drop_duplicates(), bench.frame)
    clus = dict(zip(cl.cell_id, cl.cluster_cpg))

    def nontrivial_diff(i, j):
        out = set()
        for q, c in enumerate(condc):
            a, b = F[i, q], F[j, q]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b) or abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
                out.add(c)
        return out - TRIVIAL

    # ---------------- 0. HIST_D7 reproduction on round-4 DOPT_Q 4-read sets ----------------
    rep = []
    for seed, g in par.groupby("split_seed"):
        sse_m = 0.0
        for cid, gg in g.groupby("cell_id"):
            i = pos[cid]
            Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
            S = [Mi.index(LANTHANIDES.index(s)) for s in gg.read_set.iloc[0].split(",")]
            x, yv = X[Mi], Y[i, Mi]
            u = np.array([HIST_D7_BETA if m == GD else 0.0 for m in Mi])
            lev = u + pwl(x, S, yv - u)
            for r in gg.itertuples():
                a, b = Mi.index(int(r.ia)), Mi.index(int(r.ib))
                sse_m += (lev[a] - lev[b] - r.y) ** 2
        y = g.y.to_numpy()
        sst = ((y - y.mean()) ** 2).sum()
        rep.append({"seed": int(seed), "R2_skA": 1 - sse_m / sst, "R2_parquet": 1 - ((g.HIST_D7 - g.y) ** 2).sum() / sst})
    rep = pd.DataFrame(rep)
    log("REPRO HIST_D7 (DOPT_Q 4 reads, BPG):", rep.round(6).to_dict("records"), "mean", round(rep.R2_skA.mean(), 4))

    # ---------------- DOPT_Q nested sets (gen21 fold cache, BPG) ----------------
    dq = {}
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            for cell in rec["outer"]["cells"]:
                seq = cell["seqs"]["DOPT_Q"]
                d = {}
                for k in range(1, len(seq) + 1):
                    rd = tuple(sorted({int(m) for p in seq[:k] for m in (p if np.ndim(p) else [p])}))
                    d.setdefault(len(rd), rd)
                dq[(seed, cell["cell_id"])] = d

    # ---------------- per-cell build ----------------
    Ks = (0.0, 0.5, 1.0, 1.78, 3.0, 5.0, 10.0)
    NS = range(0, 7)
    cells, links = [], []
    S_ = {}                        # (arm, policy, n) -> per-cell SSE list
    PP = []                        # per-pair predictions (HULL) for a few arms

    def put(key, ci, v):
        S_.setdefault(key, {})[ci] = v

    for ci, t in enumerate(c3):
        i = pos[t]
        Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        m = len(Mi)
        x, yv = X[Mi], Y[i, Mi]
        pa = np.array([q for q in range(m - 1) if Z_VEC[Mi[q + 1]] - Z_VEC[Mi[q]] == 1])
        pb = pa + 1
        ysf = yv[pa] - yv[pb]
        cover = np.all(np.isfinite(Y[:, Mi]), axis=1)
        cover[i] = False
        tclean = t not in v2
        pools = {
            "own": [j for j in np.where((pub == pub[i]) & (ext == ext[i]) & cover & okc)[0]
                    if tclean and nontrivial_diff(i, j)],
            "xext": list(np.where((pub == pub[i]) & (ext != ext[i]) & cover & okc)[0]) if tclean else [],
            "xpub": list(np.where((ext == ext[i]) & (pub != pub[i]) & cover & okc)[0]) if tclean else [],
            "chem": list(np.where((chem == chem[i]) & (pub != pub[i]) & cover & okc)[0]) if tclean else [],
        }
        # near-copy diagnostics of own links
        flag_bit, flag_sd = [], []
        sf_t = ysf
        for j in pools["own"]:
            yj = Y[j, Mi]
            dlt = yv - yj
            nbit = int((np.abs(dlt) < 1e-9).sum())
            sfj = yj[pa] - yj[pb]
            nsf = int((np.abs(sfj - sf_t) < 1e-6).sum()) if len(pa) else 0
            sd = float(np.std(dlt - dlt.mean(), ddof=0))
            # slope-removed shape sd
            A = np.vstack([np.ones(m), x]).T
            res = dlt - A @ np.linalg.lstsq(A, dlt, rcond=None)[0]
            sd2 = float(np.std(res))
            known = any({t, ids[j]} == set(kc) for kc in KNOWN_COPIES)
            links.append({"target": t, "sib": ids[j], "pub": pub[i], "n_metals": m, "n_bitid": nbit,
                          "frac_bitid": nbit / m, "n_sf_id": nsf, "frac_sf_id": nsf / max(len(pa), 1),
                          "shape_sd": sd, "shape_sd_slope": sd2, "known_copy": known,
                          "diff": ",".join(sorted(c.replace("cond__", "") for c in nontrivial_diff(i, j)))})
            if nbit / m >= 0.5 or (len(pa) and nsf / len(pa) >= 0.5) or known:
                flag_bit.append(j)
            if sd < 0.02:
                flag_sd.append(j)
        pools["own_nobit"] = [j for j in pools["own"] if j not in flag_bit]
        pools["own_nobit_nosd"] = [j for j in pools["own_nobit"] if j not in flag_sd]
        u = np.array([HIST_D7_BETA if q == GD else 0.0 for q in Mi])
        hull = hull_order(x)
        T = {p: Y[np.ix_(v, Mi)].mean(0) for p, v in pools.items() if len(v)}
        cells.append({"ci": ci, "cell_id": t, "pub": pub[i], "ext": ext[i], "chem": chem[i], "cluster": clus[t],
                      "c5": pub[i] == C5, "n_metals": m, "N": len(pa), "sy": float(ysf.sum()),
                      "syy": float((ysf ** 2).sum()), **{f"n_{p}": len(v) for p, v in pools.items()},
                      "own_diff_classes": json.dumps(sorted({",".join(sorted(nontrivial_diff(i, j))) for j in pools["own"]}))})

        def sse_of(lev):
            return float(((lev[pa] - lev[pb] - ysf) ** 2).sum())

        def arms_at(Sset, n_label, pol):
            d7 = u + pwl(x, Sset, yv - u)
            put(("D7", pol, n_label), ci, sse_of(d7))
            for p, Tp in T.items():
                sib = Tp + pwl(x, Sset, yv - Tp)
                put((f"SIB_{p}", pol, n_label), ci, sse_of(sib))
                n_p = len(pools[p])
                for K in Ks:
                    w = n_p / (n_p + K) if (n_p + K) > 0 else 1.0
                    put((f"SH_{p}@K{K}", pol, n_label), ci, sse_of(w * sib + (1 - w) * d7))
                # fixed-w variant (w of a single series) for placebo pools
                w1 = 1 / (1 + 1.78)
                put((f"SHw1_{p}", pol, n_label), ci, sse_of(w1 * sib + (1 - w1) * d7))
            return d7

        for n in NS:
            Sset = hull[:min(n, m)]
            arms_at(Sset, n, "HULL")
            if n >= 1:
                put(("meas_pairs", "HULL", n), ci, float(sum(1 for a, b in zip(pa, pb) if a in Sset and b in Sset)))
            # ladder
            if n in LADDER:
                lad = [LANTHANIDES.index(s) for s in LADDER[n]]
                if all(q in Mi for q in lad):
                    arms_at([Mi.index(q) for q in lad], n, "LADDER")
                    put(("ladder_cov", "LADDER", n), ci, 1.0)
                else:
                    arms_at(Sset, n, "LADDER")
                    put(("ladder_cov", "LADDER", n), ci, 0.0)
        # DOPT_Q nested (per seed), D7 and SH_own only
        for seed in paths.DEV_SEEDS:
            d = dq.get((seed, t), {})
            for n in (2, 3, 4, 5):
                if n in d and all(q in Mi for q in d[n]):
                    Sset = [Mi.index(q) for q in d[n]]
                    covd = 1.0
                else:
                    Sset = hull[:min(n, m)]
                    covd = 0.0
                d7 = u + pwl(x, Sset, yv - u)
                put(("D7", f"DQ{seed}", n), ci, sse_of(d7))
                put(("dq_cov", f"DQ{seed}", n), ci, covd)
                if "own" in T:
                    sib = T["own"] + pwl(x, Sset, yv - T["own"])
                    w = len(pools["own"]) / (len(pools["own"]) + 1.78)
                    put(("SH_own@K1.78", f"DQ{seed}", n), ci, sse_of(w * sib + (1 - w) * d7))
        # per-pair HULL predictions for D7@4 / SH_own@2 / COPY
        if "own" in T:
            w = len(pools["own"]) / (len(pools["own"]) + 1.78)
            d7_4 = u + pwl(x, hull[:min(4, m)], yv - u)
            S2 = hull[:min(2, m)]
            sh2 = w * (T["own"] + pwl(x, S2, yv - T["own"])) + (1 - w) * (u + pwl(x, S2, yv - u))
            for q, (a, b) in enumerate(zip(pa, pb)):
                PP.append({"ci": ci, "pair": f"{LANTHANIDES[Mi[a]]}/{LANTHANIDES[Mi[b]]}", "y": ysf[q],
                           "D7_4": d7_4[a] - d7_4[b], "SH_2": sh2[a] - sh2[b], "COPY": T["own"][a] - T["own"][b]})

    cdf = pd.DataFrame(cells)
    ldf = pd.DataFrame(links)
    nc = len(cdf)

    def vec(key):
        d = S_.get(key, {})
        return np.array([d.get(c, np.nan) for c in range(nc)])

    # DOPT_Q seed-mean SSE per cell (R2 is linear in SSE given fixed SST, so seed-mean R2 == R2 of mean SSE)
    for arm in ("D7", "SH_own@K1.78"):
        for n in (2, 3, 4, 5):
            vs = np.vstack([vec((arm, f"DQ{s}", n)) for s in paths.DEV_SEEDS])
            S_[(arm, "DQ", n)] = {c: float(vs[:, c].mean()) for c in range(nc) if np.isfinite(vs[:, c]).all()}
    cdf.to_parquet(OUT / "skA_cells.parquet", index=False)
    ldf.to_csv(OUT / "skA_own_links.csv", index=False)
    pd.DataFrame(PP).to_parquet(OUT / "skA_pairs_own.parquet", index=False)

    # ---------------- populations ----------------
    pops = {
        "ALL229": np.ones(nc, bool),
        "noC5": ~cdf.c5.to_numpy(),
        "OWN1_noC5": (cdf.n_own >= 1).to_numpy() & ~cdf.c5.to_numpy(),
        "OWN1_C5": (cdf.n_own >= 1).to_numpy(),
        "OWN2_noC5": (cdf.n_own >= 2).to_numpy() & ~cdf.c5.to_numpy(),
        "OWNeq1_noC5": (cdf.n_own == 1).to_numpy() & ~cdf.c5.to_numpy(),
        "XPUB_noC5": (cdf.n_xpub >= 1).to_numpy() & ~cdf.c5.to_numpy(),
        "XEXT_noC5": (cdf.n_xext >= 1).to_numpy() & ~cdf.c5.to_numpy(),
        "CHEM_noC5": (cdf.n_chem >= 1).to_numpy() & ~cdf.c5.to_numpy(),
        "OWN&XPUB_noC5": (cdf.n_own >= 1).to_numpy() & (cdf.n_xpub >= 1).to_numpy() & ~cdf.c5.to_numpy(),
        "OWNnobit_noC5": (cdf.n_own_nobit >= 1).to_numpy() & ~cdf.c5.to_numpy(),
        "OWNnobitnosd_noC5": (cdf.n_own_nobit_nosd >= 1).to_numpy() & ~cdf.c5.to_numpy(),
    }
    rng = np.random.default_rng(777)

    class Pop:
        def __init__(self, mask, unit="pub"):
            self.idx = np.where(mask)[0]
            c = cdf.iloc[self.idx]
            self.c = c
            self.units = sorted(c[unit].unique())
            ui = c[unit].map({p: k for k, p in enumerate(self.units)}).to_numpy()
            self.G = np.zeros((len(c), len(self.units)))
            self.G[np.arange(len(c)), ui] = 1
            self.N = c.N.to_numpy(float) @ self.G
            self.sy = c.sy.to_numpy(float) @ self.G
            self.syy = c.syy.to_numpy(float) @ self.G
            self.W = rng.multinomial(len(self.units), np.full(len(self.units), 1 / len(self.units)), size=B).astype(float)
            self.sst = self.syy.sum() - self.sy.sum() ** 2 / self.N.sum()

        def s(self, key):
            v = vec(key)[self.idx]
            if np.isnan(v).any():
                raise ValueError(f"missing {key}: {np.isnan(v).sum()}")
            return v @ self.G

        def r2(self, key, w=None):
            s = self.s(key)
            if w is None:
                return 1 - s.sum() / self.sst
            return 1 - (w @ s) / (w @ self.syy - (w @ self.sy) ** 2 / (w @ self.N))

        def diff(self, ka, kb):
            sa, sb = self.s(ka), self.s(kb)
            d = (sb.sum() - sa.sum()) / self.sst
            sstw = self.W @ self.syy - (self.W @ self.sy) ** 2 / (self.W @ self.N)
            bd = (self.W @ sb - self.W @ sa) / sstw
            contrib = (sb - sa) / self.sst
            lopo = []
            for q in range(len(self.units)):
                w = np.ones(len(self.units))
                w[q] = 0
                sstq = w @ self.syy - (w @ self.sy) ** 2 / (w @ self.N)
                lopo.append((w @ sb - w @ sa) / sstq)
            order = np.argsort(-contrib)
            drop = {}
            for k in (1, 2, 3):
                w = np.ones(len(self.units))
                w[order[:k]] = 0
                sstq = w @ self.syy - (w @ self.sy) ** 2 / (w @ self.N)
                drop[f"drop_top{k}"] = float((w @ sb - w @ sa) / sstq)
            pos_ = contrib[contrib > 0]
            relw = np.where(sb > 0, 1 - sa / np.where(sb > 0, sb, 1), np.nan)
            return {"R2_a": float(1 - sa.sum() / self.sst), "R2_b": float(1 - sb.sum() / self.sst), "d": float(d),
                    "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5)),
                    "P_gt0": float((bd > 0).mean()), "units": len(self.units),
                    "units_pos": int((contrib > 1e-12).sum()), "units_neg": int((contrib < -1e-12).sum()),
                    "top3_share": float(np.sort(contrib)[::-1][:3].sum() / d) if d != 0 else np.nan,
                    "lopo_min": float(min(lopo)), "lopo_min_unit": self.units[int(np.argmin(lopo))],
                    "lopo_max": float(max(lopo)), **drop,
                    "eqw_rel_sse": float(np.nanmean(relw)),
                    "top_units": ", ".join(f"{self.units[q]}:{contrib[q]:+.3f}" for q in order[:3])}

    # ---------------- 1. reproduce the headline ladder ----------------
    res = {"repro": rep.to_dict("records")}
    log("\n=== populations")
    for k, msk in pops.items():
        c = cdf[msk]
        sh = c.groupby("cluster").N.sum()
        log(f"{k:20s} cells {len(c):4d} pubs {c.pub.nunique():3d} clusters {c.cluster.nunique():3d} "
            f"largest-cluster share {sh.max() / sh.sum():.2f} pairs {int(c.N.sum())}")
    log("\n=== ladders (R2, whole series)")
    lad_rows = []
    for pk, arms in (("OWN1_noC5", ["D7", "SH_own@K1.78", "SIB_own", "SH_own@K0.0"]), ("noC5", ["D7"]),
                     ("OWN1_C5", ["D7", "SH_own@K1.78"]), ("OWN2_noC5", ["D7", "SH_own@K1.78"]),
                     ("OWNeq1_noC5", ["D7", "SH_own@K1.78", "SIB_own"]),
                     ("XPUB_noC5", ["D7", "SH_xpub@K1.78", "SIB_xpub", "SHw1_xpub"]),
                     ("XEXT_noC5", ["D7", "SH_xext@K1.78", "SIB_xext", "SHw1_xext"]),
                     ("CHEM_noC5", ["D7", "SH_chem@K1.78", "SIB_chem", "SHw1_chem"]),
                     ("OWN&XPUB_noC5", ["D7", "SH_own@K1.78", "SH_xpub@K1.78", "SHw1_xpub"])):
        P = Pop(pops[pk])
        for arm in arms:
            for pol in ("HULL", "LADDER", "DQ"):
                if pol != "HULL" and arm not in ("D7", "SH_own@K1.78"):
                    continue
                row = {"pop": pk, "arm": arm, "pol": pol}
                for n in NS:
                    key = (arm, pol, n) if not (pol == "DQ" and n not in (2, 3, 4, 5)) and not (
                        pol == "LADDER" and n not in LADDER) else None
                    if key is None:
                        continue
                    try:
                        row[n] = round(float(P.r2(key)), 3)
                    except ValueError:
                        pass
                lad_rows.append(row)
                log(f"  {pk:15s} {arm:14s} {pol:6s} " + " ".join(f"n{n}:{row[n]:.3f}" for n in NS if n in row))
    pd.DataFrame(lad_rows).to_csv(OUT / "skA_ladders.csv", index=False)

    # measured pairs under HULL (share of pairs with both ends read), OWN1_noC5
    P = Pop(pops["OWN1_noC5"])
    for n in range(1, 7):
        mp = vec(("meas_pairs", "HULL", n))[P.idx].sum() / P.c.N.sum()
        log(f"measured-pair share HULL n={n}: {mp:.4f}")
    log("cells with <= 4 / <= 5 metals in OWN1_noC5:", int((P.c.n_metals <= 4).sum()), int((P.c.n_metals <= 5).sum()))
    for n in (2, 3, 4, 5):
        log(f"ladder coverage n={n}: {vec(('ladder_cov', 'LADDER', n))[P.idx].mean():.2f}; DOPT_Q coverage "
            f"{np.mean([vec(('dq_cov', f'DQ{s}', n))[P.idx].mean() for s in paths.DEV_SEEDS]):.2f}")

    # ---------------- 2. reads-equivalent contrasts with unit diagnostics ----------------
    log("\n=== contrasts (a minus b; d = R2_a - R2_b; publication bootstrap 95%; unit = publication)")
    con = []

    def contrast(pk, ka, kb, label, unit="pub"):
        if not pops[pk].any():
            return
        P = Pop(pops[pk], unit)
        try:
            d = P.diff(ka, kb)
        except ValueError as e:
            log("skip", label, e)
            return
        d.update({"pop": pk, "label": label, "unit": unit, "cells": len(P.idx)})
        con.append(d)
        log(f"{pk:18s} {unit:7s} {label:34s} {d['R2_a']:.3f} vs {d['R2_b']:.3f}  d {d['d']:+.3f} [{d['lo']:+.3f},"
            f"{d['hi']:+.3f}] units +{d['units_pos']}/-{d['units_neg']} of {d['units']} top3 {d['top3_share']:.2f} "
            f"LOPOmin {d['lopo_min']:+.3f} ({d['lopo_min_unit']}) drop1/2/3 {d['drop_top1']:+.3f}/"
            f"{d['drop_top2']:+.3f}/{d['drop_top3']:+.3f} eqw relSSE {d['eqw_rel_sse']:+.3f} | {d['top_units']}")

    SH = "SH_own@K1.78"
    for pk in ("OWN1_noC5", "OWN1_C5", "OWN2_noC5", "OWNeq1_noC5", "OWNnobit_noC5", "OWNnobitnosd_noC5"):
        own_key = {"OWNnobit_noC5": "SH_own_nobit@K1.78", "OWNnobitnosd_noC5": "SH_own_nobit_nosd@K1.78"}.get(pk, SH)
        copy_key = {"OWNnobit_noC5": "SIB_own_nobit", "OWNnobitnosd_noC5": "SIB_own_nobit_nosd"}.get(pk, "SIB_own")
        contrast(pk, (copy_key, "HULL", 0), ("D7", "HULL", 3), "COPY@0 - D7_HULL@3")
        contrast(pk, (own_key, "HULL", 2), ("D7", "HULL", 4), "SH@2 - D7_HULL@4")
        contrast(pk, (own_key, "HULL", 2), ("D7", "HULL", 5), "SH@2 - D7_HULL@5")
        contrast(pk, (own_key, "HULL", 4), ("D7", "HULL", 5), "SH@4 - D7_HULL@5")
        if pk in ("OWN1_noC5", "OWN1_C5", "OWN2_noC5"):
            contrast(pk, (own_key, "HULL", 2), ("D7", "DQ", 4), "SH@2 - D7_DOPTQ@4")
            contrast(pk, (own_key, "HULL", 2), ("D7", "LADDER", 4), "SH@2 - D7_LADDER@4")
            contrast(pk, (own_key, "LADDER", 2), ("D7", "LADDER", 4), "SH_LAD@2 - D7_LADDER@4")
            contrast(pk, (copy_key, "HULL", 0), ("D7", "LADDER", 3), "COPY@0 - D7_LADDER@3")
            contrast(pk, (copy_key, "HULL", 0), ("D7", "DQ", 3), "COPY@0 - D7_DOPTQ@3")
            contrast(pk, (own_key, "HULL", 2), ("D7", "HULL", 3), "SH@2 - D7_HULL@3")
    # family (publication x extractant) and cluster units for the headline contrast
    cdf["fam"] = cdf.pub + "|" + cdf.ext
    contrast("OWN1_noC5", (SH, "HULL", 2), ("D7", "HULL", 4), "SH@2 - D7_HULL@4", unit="fam")
    contrast("OWN1_noC5", (SH, "HULL", 2), ("D7", "HULL", 4), "SH@2 - D7_HULL@4", unit="cluster")
    contrast("OWN1_noC5", ("SIB_own", "HULL", 0), ("D7", "HULL", 3), "COPY@0 - D7_HULL@3", unit="fam")
    contrast("OWN1_noC5", ("SIB_own", "HULL", 0), ("D7", "HULL", 3), "COPY@0 - D7_HULL@3", unit="cluster")
    # literature / placebo clause (c)
    for pk, p in (("XPUB_noC5", "xpub"), ("XEXT_noC5", "xext"), ("CHEM_noC5", "chem"), ("OWN&XPUB_noC5", "xpub")):
        for kk in ("@K1.78", "@K0.0", "@K5.0", "@K10.0"):
            contrast(pk, (f"SH_{p}{kk}", "HULL", 2), ("D7", "HULL", 2), f"SH_{p}{kk}@2 - D7_HULL@2")
        contrast(pk, (f"SHw1_{p}", "HULL", 2), ("D7", "HULL", 2), f"SHw1_{p}@2 - D7_HULL@2")
    contrast("OWN&XPUB_noC5", (SH, "HULL", 2), ("SH_xpub@K1.78", "HULL", 2), "SH_own@2 - SH_xpub@2")
    contrast("OWN&XPUB_noC5", ("SIB_own", "HULL", 0), ("SIB_xpub", "HULL", 0), "COPY_own - COPY_xpub")
    # K sensitivity of headline
    for K in Ks:
        contrast("OWN1_noC5", (f"SH_own@K{K}", "HULL", 2), ("D7", "HULL", 4), f"SH(K={K})@2 - D7_HULL@4")
    for K in Ks:
        contrast("OWN1_noC5", (f"SH_own@K{K}", "HULL", 3), ("D7", "HULL", 5), f"SH(K={K})@3 - D7_HULL@5")
    cdf_con = pd.DataFrame(con)
    cdf_con.to_csv(OUT / "skA_contrasts.csv", index=False)

    # ---------------- 3. near-copy links ----------------
    log("\n=== own links: near-copy diagnostics (all C3 targets incl. C5)")
    log(f"links {len(ldf)}, targets {ldf.target.nunique()}")
    for nm, q in (("frac_bitid>=0.5", ldf.frac_bitid >= 0.5), ("frac_sf_id>=0.5", ldf.frac_sf_id >= 0.5),
                  ("shape_sd<0.02", ldf.shape_sd < 0.02), ("shape_sd_slope<0.01", ldf.shape_sd_slope < 0.01),
                  ("known_copy", ldf.known_copy)):
        g = ldf[q]
        log(f"  {nm:22s} links {len(g):4d} targets {g.target.nunique():3d} pubs {g.pub.value_counts().to_dict()}")
    log(ldf[(ldf.frac_bitid >= 0.3) | ldf.known_copy | (ldf.frac_sf_id >= 0.5)]
        .sort_values("frac_bitid", ascending=False).head(30).round(3).to_string(index=False))

    # ---------------- 4. per-family distribution of the headline contrast and a power sketch ----------------
    P = Pop(pops["OWN1_noC5"], "fam")
    sa, sb = P.s((SH, "HULL", 2)), P.s(("D7", "HULL", 4))
    sstf = P.syy - P.sy ** 2 / P.N
    fr = pd.DataFrame({"fam": P.units, "N": P.N, "sst": sstf, "sse_SH2": sa, "sse_D7_4": sb,
                       "dR2_fam": (sb - sa) / sstf, "rel_sse": 1 - sa / sb,
                       "cells": P.G.sum(0)})
    fr.to_csv(OUT / "skA_family_headline.csv", index=False)
    log("\n=== per family (pub|ext) SH@2 vs D7_HULL@4, OWN1_noC5")
    log(fr.sort_values("N", ascending=False).round(3).to_string(index=False))
    dv = fr.dR2_fam.to_numpy()
    G = len(dv)
    # exact sign-flip on family contributions to pooled dR2 (contrib = (sb - sa)/SST_pop)
    cf = (sb - sa) / P.sst
    obs = cf.sum()
    if G <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=G)))
        null = signs @ np.abs(cf)
        p_sf = float((null >= obs - 1e-15).mean())
    else:
        signs = rng.choice([-1, 1], size=(200000, G))
        p_sf = float(((signs @ np.abs(cf)) >= obs - 1e-15).mean())
    log(f"families {G}; pooled d {obs:+.4f}; one-sided sign-flip p (pooled contributions) {p_sf:.4f}; "
        f"per-family dR2 mean {dv.mean():+.3f} median {np.median(dv):+.3f} sd {dv.std(ddof=1):.3f}; "
        f"positive {int((dv > 0).sum())}/{G}; rel-SSE mean {fr.rel_sse.mean():+.3f} median {fr.rel_sse.median():+.3f}")
    # power sketch: kill fires if mean < 0 or one-sided 95% LB < -0.05 on 11 families; resample in-corpus families
    for G_new in (6, 11, 20):
        fires = 0
        R = 20000
        draws = rng.choice(dv, size=(R, G_new), replace=True)
        mu = draws.mean(1)
        sd = draws.std(1, ddof=1)
        from scipy import stats
        lb = mu - stats.t.ppf(0.95, G_new - 1) * sd / math.sqrt(G_new)
        fires = ((mu < 0) | (lb < -0.05)).mean()
        log(f"  bootstrap-of-families, G={G_new}: P(kill fires | in-corpus family distribution) = {fires:.3f}; "
            f"P(mean<0) {(mu < 0).mean():.3f}; P(LB<-0.05) {(lb < -0.05).mean():.3f}")
    # per-family xpub clause
    res["family_headline"] = fr.to_dict("records")
    res["signflip_p_family"] = p_sf
    (OUT / "skA_summary.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (OUT / "skA_run.log").write_text("\n".join(LOG), encoding="utf-8")
    log(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
