#!/usr/bin/env python
"""Skeptic B checks for idea recipe-4 (ICP-MS condition skipping: MEAN + half-shrunk, clamped log-acid line).

EXPLORATORY: every row here belongs to the 229 already-seen C3 cells (or to the whole bench for G); nothing is a test.
Read-only on existing files; writes only to ideas2/recipe4_skepB_out/.

  A  Re-build recipe_acidfit's rows (same population: C3, no C5, >= 2 CLEAN acid-only own siblings) and reproduce
     MEAN / LIN / LINCLIP / LINHALF interior / exterior R2.
  B  Cheapest-alternative competitors inside the range: PAIRMEAN (mean of the two bracketing levels),
     BRK (linear interpolation between the two bracketing levels), NEAR, shrinkage sweep lambda in 0..1,
     lambda chosen leave-one-publication-out (nested).
  C  Per-publication contributions of LINHALF - MEAN (interior); leave-top-publication-out; share of the gain.
  D  Acid-permutation control: permute the siblings' acid labels within each target's sibling set (target acid
     fixed), recompute LINHALF - MEAN on the fixed interior rows; 1,000 permutations.
  E  Pooled pair-type slope prior (leave-one-publication-out): MEAN + b_pair,-pub * (lt - mean(la)).
  F  Power: publication-level paired deltas -> resampled N-campaign sign-flip / t lower bound; P(pass kill) by N.
  G  The practical recipe itself: acid-only groups (whole bench, no C5 / V2 drops) with >= 5 distinct acid levels;
     keep 3 spanning levels (min, max, the one closest to the log midpoint); predict every other level
     inside the range; MEAN3 vs LIN3 vs LINHALF3 vs bracket arms vs the leave-one-out full-grid MEAN.
  H  Chemotype / extractant make-up of the 8 publications; within-campaign R2 (around campaign x pair mean).

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe4_skepB_check.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES, Z_VEC, N_LN  # noqa: E402
import recipe_analyze as RA  # noqa: E402

OUT_IN = RA.OUT
OUT = Path(__file__).resolve().parent.parent / "recipe4_skepB_out"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261023)


def r2(g: pd.DataFrame, col: str) -> float:
    y = g.y.to_numpy()
    return 1 - ((g[col] - g.y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def pop_summ(g: pd.DataFrame, col: str, ref: str | None = None) -> dict:
    cc = g.groupby("cell_id").agg(publication=("publication", "first"), N=("y", "size"), sy=("y", "sum"),
                                  syy=("y", lambda s: float((s ** 2).sum()))).reset_index()
    P = RA.Pop(cc, "x")

    def cs(c):
        return g.assign(e=(g[c] - g.y) ** 2).groupby("cell_id").e.sum().reindex(cc.cell_id).to_numpy()
    return P.summ(cs(col), cs(ref) if ref else None)


def build_rows(cells, bench):
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    pos = {c: i for i, c in enumerate(f.cell_id.astype(str))}
    acid = f["cond__acid_concentration_M"].to_numpy(float)
    rows = []
    for r in cells.itertuples():
        if r.n_own < 2 or r.c5:
            continue
        own = json.loads(r.own_ids)
        fams = json.loads(r.own_fams)
        js = [pos[j] for j, fm in zip(own, fams) if fm == "acid_conc" and np.isfinite(acid[pos[j]]) and acid[pos[j]] > 0]
        i = pos[r.cell_id]
        if len(js) < 2 or not np.isfinite(acid[i]) or acid[i] <= 0:
            continue
        la = np.log10(acid[js])
        if np.ptp(la) < 1e-9:
            continue
        lt = math.log10(acid[i])
        inside = la.min() - 1e-9 <= lt <= la.max() + 1e-9
        Mi = [LANTHANIDES.index(s) for s in r.metals.split(",")]
        for p in range(len(Mi) - 1):
            a, b = Mi[p], Mi[p + 1]
            if Z_VEC[b] - Z_VEC[a] != 1:
                continue
            v = Y[js, a] - Y[js, b]
            y = Y[i, a] - Y[i, b]
            rows.append({"cell_id": r.cell_id, "publication": r.publication, "extractant": r.extractant,
                         "chemotype": r.chemotype, "pair": f"{LANTHANIDES[a]}/{LANTHANIDES[b]}", "inside": inside,
                         "n_sib": len(js), "n_levels": len(np.unique(np.round(la, 6))), "y": y, "lt": lt,
                         "la": la.tolist(), "v": v.tolist(), "target_acid_M": float(acid[i])})
    return pd.DataFrame(rows)


def line_pred(la, v, lt, lam=1.0, clamp=False):
    la = np.asarray(la)
    v = np.asarray(v)
    m = v.mean()
    if np.ptp(la) < 1e-9:
        return m
    b = np.polyfit(la, v, 1)
    x = min(max(lt, la.min()), la.max()) if clamp else lt
    return m + lam * (float(np.polyval(b, x)) - m)


def add_arms(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    out = {k: [] for k in ("MEAN", "LIN", "LINCLIP", "LINHALF", "NEAR", "PAIRMEAN", "BRK")}
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    for lam in lams:
        out[f"LAM{lam:.2f}"] = []
    for r in t.itertuples():
        la, v, lt = np.asarray(r.la), np.asarray(r.v), r.lt
        out["MEAN"].append(v.mean())
        out["LIN"].append(line_pred(la, v, lt))
        out["LINCLIP"].append(line_pred(la, v, lt, clamp=True))
        out["LINHALF"].append(line_pred(la, v, lt, 0.5))
        for lam in lams:
            out[f"LAM{lam:.2f}"].append(line_pred(la, v, lt, lam, clamp=True))
        d = np.abs(la - lt)
        near = np.isclose(d, d.min())
        out["NEAR"].append(v[near].mean())
        lo = la < lt - 1e-9
        hi = la > lt + 1e-9
        if lo.any() and hi.any():
            xl = la[lo].max()
            xh = la[hi].min()
            vl = v[np.isclose(la, xl)].mean()
            vh = v[np.isclose(la, xh)].mean()
            out["PAIRMEAN"].append(0.5 * (vl + vh))
            out["BRK"].append(vl + (vh - vl) * (lt - xl) / (xh - xl))
        else:
            out["PAIRMEAN"].append(np.nan)
            out["BRK"].append(np.nan)
    for k, vv in out.items():
        t[k] = vv
    return t


def main() -> int:
    log = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.append(s)

    cells = pd.read_parquet(OUT_IN / "recipe_cells.parquet")
    bench = D.load_bench()
    t = add_arms(build_rows(cells, bench))
    ti = t[t.inside].copy()
    te = t[~t.inside].copy()
    P("EXPLORATORY: 229 already-seen C3 cells; no test.")
    P(f"rows {len(t)} (interior {len(ti)} rows / {ti.cell_id.nunique()} cells; exterior {len(te)} / "
      f"{te.cell_id.nunique()}), pubs {t.publication.nunique()}")

    # ---------- A reproduction ----------
    P("\n=== A reproduction (pooled R2, publication bootstrap)")
    for lab, g in (("interior", ti), ("exterior", te)):
        for arm in ("MEAN", "LIN", "LINCLIP", "LINHALF", "NEAR"):
            d = pop_summ(g, arm, "MEAN")
            P(f"{lab:9s} {arm:8s} R2 {d['R2']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]  d_vs_MEAN {d['d']:+.3f} "
              f"[{d['d_lo']:+.3f},{d['d_hi']:+.3f}]  lopo {d['lopo_min']:.3f}..{d['lopo_max']:.3f}")

    # ---------- B cheapest competitors inside ----------
    P("\n=== B interior competitors (all interior rows have a bracket)")
    P("bracket available on", int(ti.PAIRMEAN.notna().sum()), "of", len(ti), "interior rows")
    for arm in ("PAIRMEAN", "BRK", "LINHALF") + tuple(f"LAM{x:.2f}" for x in (0.0, 0.25, 0.5, 0.75, 1.0)):
        d = pop_summ(ti, arm, "MEAN")
        d2 = pop_summ(ti, arm, "LINHALF")
        P(f"{arm:9s} R2 {d['R2']:.3f}  d_vs_MEAN {d['d']:+.3f} [{d['d_lo']:+.3f},{d['d_hi']:+.3f}]  "
          f"d_vs_LINHALF {d2['d']:+.3f} [{d2['d_lo']:+.3f},{d2['d_hi']:+.3f}]")
    # nested LOPO lambda
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    pubs = sorted(ti.publication.unique())
    pred = pd.Series(np.nan, index=ti.index)
    chosen = {}
    for p in pubs:
        tr = ti[ti.publication != p]
        best = max(lams, key=lambda L: r2(tr, f"LAM{L:.2f}"))
        chosen[p] = best
        pred[ti.publication == p] = ti.loc[ti.publication == p, f"LAM{best:.2f}"]
    ti["LAMNEST"] = pred
    d = pop_summ(ti, "LAMNEST", "MEAN")
    P(f"nested-LOPO lambda: chosen {chosen}; R2 {d['R2']:.3f} d_vs_MEAN {d['d']:+.3f} [{d['d_lo']:+.3f},{d['d_hi']:+.3f}]")

    # ---------- C per-publication ----------
    P("\n=== C per-publication (interior): SSE MEAN, SSE LINHALF, rel. change, SSE PAIRMEAN, SSE NEAR, SST_pub")
    rows = []
    for p, g in ti.groupby("publication"):
        sse = {a: float(((g[a] - g.y) ** 2).sum()) for a in ("MEAN", "LINHALF", "PAIRMEAN", "NEAR", "LIN")}
        sst = float(((g.y - g.y.mean()) ** 2).sum())
        rows.append({"publication": p, "cells": g.cell_id.nunique(), "rows": len(g),
                     "chemotype": ",".join(sorted(g.chemotype.unique())), "n_levels_med": float(g.n_levels.median()),
                     **{f"sse_{a}": s for a, s in sse.items()}, "sst_pub": sst,
                     "rel_LINHALF": sse["LINHALF"] / sse["MEAN"] - 1, "rel_NEAR": sse["NEAR"] / sse["MEAN"] - 1,
                     "rel_PAIRMEAN": sse["PAIRMEAN"] / sse["MEAN"] - 1})
    pp = pd.DataFrame(rows).sort_values("sse_MEAN", ascending=False)
    pp.to_csv(OUT / "C_per_publication_interior.csv", index=False)
    with pd.option_context("display.width", 250):
        P(pp.round(4).to_string(index=False))
    gain = pp.sse_MEAN - pp.sse_LINHALF
    P(f"total SSE gain LINHALF vs MEAN {gain.sum():.4f}; positive pubs {int((gain > 0).sum())}/{len(gain)}; "
      f"top-1 share {gain.max() / gain.sum():.2f}; top-2 share {gain.nlargest(2).sum() / gain.sum():.2f}")
    sst_all = float(((ti.y - ti.y.mean()) ** 2).sum())
    for p in pp.publication:
        g = ti[ti.publication != p]
        P(f"  drop {p}: R2 MEAN {r2(g, 'MEAN'):.3f} LINHALF {r2(g, 'LINHALF'):.3f} "
          f"d {r2(g, 'LINHALF') - r2(g, 'MEAN'):+.3f}  NEAR-MEAN {r2(g, 'NEAR') - r2(g, 'MEAN'):+.3f}")
    # NEAR failure: is it one publication?
    gn = pp.sse_NEAR - pp.sse_MEAN
    P(f"NEAR excess SSE over MEAN {gn.sum():.4f}; top-1 pub share {gn.max() / gn.sum():.2f} "
      f"({pp.publication.iloc[int(np.argmax(gn.to_numpy()))]})")

    # ---------- D acid permutation ----------
    P("\n=== D acid-permutation control (sibling acids permuted within each target's sibling set; interior rows fixed)")
    obs = r2(ti, "LINHALF") - r2(ti, "MEAN")
    # permute per cell (same permutation for all pairs of the cell)
    cell_groups = {c: g.index.to_numpy() for c, g in ti.groupby("cell_id")}
    nulls = []
    yv = ti.y.to_numpy()
    sst = ((yv - yv.mean()) ** 2).sum()
    sse_mean = ((ti.MEAN - ti.y) ** 2).sum()
    for _ in range(1000):
        predp = np.empty(len(ti))
        k = 0
        idx_map = {ix: q for q, ix in enumerate(ti.index)}
        for c, ixs in cell_groups.items():
            la0 = np.asarray(ti.at[ixs[0], "la"])
            perm = RNG.permutation(len(la0))
            for ix in ixs:
                la = la0[perm]
                predp[idx_map[ix]] = line_pred(la, ti.at[ix, "v"], ti.at[ix, "lt"], 0.5)
        nulls.append((sse_mean - ((predp - yv) ** 2).sum()) / sst)
    nulls = np.array(nulls)
    P(f"observed LINHALF - MEAN {obs:+.4f}; permutation null mean {nulls.mean():+.4f}, 95th pct "
      f"{np.percentile(nulls, 95):+.4f}, max {nulls.max():+.4f}; P(null >= obs) {(nulls >= obs).mean():.3f}")

    # ---------- E pooled pair-type slope prior ----------
    P("\n=== E pooled pair-type slope (leave-one-publication-out), applied to MEAN (clamped)")
    # within-sibling-set slope data, demeaned per (cell, pair) -> per publication pooled slope per pair type
    recs = []
    for r in t.itertuples():
        la = np.asarray(r.la)
        v = np.asarray(r.v)
        recs.append({"publication": r.publication, "pair": r.pair, "sxy": float(((la - la.mean()) * (v - v.mean())).sum()),
                     "sxx": float(((la - la.mean()) ** 2).sum())})
    sl = pd.DataFrame(recs)
    preds = {"POOLPAIR": [], "POOLALL": [], "POOLPAIR_HALF": []}
    for r in ti.itertuples():
        o = sl[(sl.publication != r.publication)]
        op = o[o.pair == r.pair]
        bp = op.sxy.sum() / op.sxx.sum() if op.sxx.sum() > 0 else 0.0
        la = np.asarray(r.la)
        v = np.asarray(r.v)
        x = min(max(r.lt, la.min()), la.max()) - la.mean()
        preds["POOLPAIR"].append(v.mean() + bp * x)
        preds["POOLPAIR_HALF"].append(v.mean() + 0.5 * bp * x)
        bo = o.sxy.sum() / o.sxx.sum()
        preds["POOLALL"].append(v.mean() + bo * x)
    for k, vv in preds.items():
        ti[k] = vv
        d = pop_summ(ti, k, "MEAN")
        P(f"{k:14s} R2 {d['R2']:.3f} d_vs_MEAN {d['d']:+.3f} [{d['d_lo']:+.3f},{d['d_hi']:+.3f}]")
    # cross-publication sign agreement of pair slopes (from recipe_V2_acid.csv, whole bench)
    v2 = pd.read_csv(OUT_IN / "recipe_V2_acid.csv")
    agg = v2.groupby(["pub", "pair"]).slope.mean().reset_index()
    sg = agg.groupby("pair").slope.agg(lambda s: f"{(s > 0).sum()}+/{(s < 0).sum()}-  med {s.median():+.3f}")
    P("per-pair slope signs across publications (whole bench V2 groups):", sg.to_dict())

    # ---------- F power ----------
    P("\n=== F power for the prospective kill (unit = campaign; per-publication deltas as the campaign distribution)")
    # per-pub delta on the R2 scale: (SSE_MEAN - SSE_LINHALF)/SST_pub, and relative SSE
    dR = ((pp.sse_MEAN - pp.sse_LINHALF) / pp.sst_pub).to_numpy()
    rel = (-pp.rel_LINHALF).to_numpy()
    P(f"per-pub delta R2 (within-pub SST): mean {dR.mean():+.4f} sd {dR.std(ddof=1):.4f} median {np.median(dR):+.4f}; "
      f"per-pub rel SSE gain mean {rel.mean():+.4f} sd {rel.std(ddof=1):.4f}")
    res = []
    for N in (8, 12, 20, 30, 50):
        for lab, arr in (("dR2_pub", dR), ("relSSE", rel)):
            ok_t = ok_both = 0
            B = 4000
            for _ in range(B):
                s = RNG.choice(arr, N, replace=True)
                m = s.mean()
                se = s.std(ddof=1) / math.sqrt(N)
                lb = m - 1.645 * se
                ok_t += lb > 0
                ok_both += (lb > 0) and (m >= 0.02)
            res.append({"N": N, "metric": lab, "P_lb_gt0": ok_t / B, "P_pass_kill": ok_both / B})
    rp = pd.DataFrame(res)
    rp.to_csv(OUT / "F_power.csv", index=False)
    P(rp.to_string(index=False))

    # ---------- G the practical recipe: keep 3 spanning levels ----------
    P("\n=== G keep-3-spanning-levels recipe on acid-only groups with >= 5 distinct acid levels (whole bench)")
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    keep = np.array([(c not in drop) and (p != "pub_6a0ef8528c") for c, p in zip(ids, f.publication_id.astype(str))])
    cond = [c for c in f.columns if c.startswith("cond__") and c not in
            ("cond__acid_concentration_M", "cond__contact_time_min", "cond__metal_concentration_mM")]
    key = f[cond].astype(float).round(6).fillna(-999.0).map(repr).agg("|".join, axis=1)
    f = f.assign(gkey=f.publication_id.astype(str) + "#" + f.extractant.astype(str) + "#" + key)
    acid = f.cond__acid_concentration_M.to_numpy(float)
    grec = []
    ngroups = 0
    for gk, rr in f[keep].groupby("gkey"):
        ii = rr.index.to_numpy()
        ii = ii[np.isfinite(acid[ii]) & (acid[ii] > 0)]
        if len(ii) == 0:
            continue
        la_all = np.round(np.log10(acid[ii]), 6)
        lv = np.unique(la_all)
        if len(lv) < 5:
            continue
        ngroups += 1
        mid = 0.5 * (lv.min() + lv.max())
        kept = [lv.min(), lv.max(), lv[1:-1][np.argmin(np.abs(lv[1:-1] - mid))]]
        for a in range(N_LN - 1):
            if Z_VEC[a + 1] - Z_VEC[a] != 1:
                continue
            sf = Y[ii, a] - Y[ii, a + 1]
            ok = np.isfinite(sf)
            if len(np.unique(la_all[ok])) < 5:
                continue
            # level means
            lvl = {L: float(sf[ok & (la_all == L)].mean()) for L in np.unique(la_all[ok])}
            if not all(k in lvl for k in kept):
                continue
            kx = np.array(sorted(kept))
            kv = np.array([lvl[k] for k in kx])
            for L, yv_ in lvl.items():
                if L in kept:
                    continue
                others = np.array([q for q in lvl if q != L])
                ov = np.array([lvl[q] for q in others])
                lo_k = kx[kx < L].max()
                hi_k = kx[kx > L].min()
                vl, vh = lvl[lo_k], lvl[hi_k]
                grec.append({"group": gk, "publication": gk.split("#")[0], "pair": f"{LANTHANIDES[a]}/{LANTHANIDES[a + 1]}",
                             "y": yv_, "MEAN3": kv.mean(), "LIN3": line_pred(kx, kv, L, 1.0),
                             "LINHALF3": line_pred(kx, kv, L, 0.5), "BRK3": vl + (vh - vl) * (L - lo_k) / (hi_k - lo_k),
                             "PAIRMEAN3": 0.5 * (vl + vh),
                             "MEAN_LOO_ALL": ov.mean(), "LINHALF_LOO_ALL": line_pred(others, ov, L, 0.5, clamp=True),
                             "n_levels": len(lvl)})
    G = pd.DataFrame(grec)
    if len(G):
        G["cell_id"] = G.group + "|" + G.y.round(9).astype(str)  # unit for Pop is publication anyway
        P(f"groups with >= 5 levels: {ngroups}; scored (group,pair,level) rows {len(G)}; groups {G.group.nunique()}; "
          f"pubs {G.publication.nunique()}; median levels {G.n_levels.median()}")
        G.to_csv(OUT / "G_keep3_rows.csv", index=False)
        for arm in ("MEAN3", "LIN3", "LINHALF3", "BRK3", "PAIRMEAN3", "MEAN_LOO_ALL", "LINHALF_LOO_ALL"):
            yv2 = G.y.to_numpy()
            sstG = ((yv2 - yv2.mean()) ** 2).sum()
            # within-group-pair SS (acid-driven variation only)
            gm = G.groupby(["group", "pair"]).y.transform("mean")
            ssw = ((G.y - gm) ** 2).sum()
            sse = ((G[arm] - G.y) ** 2).sum()
            rmse = math.sqrt(sse / len(G))
            pubd = G.assign(e=(G[arm] - G.y) ** 2, e0=(G.MEAN3 - G.y) ** 2).groupby("publication")[["e", "e0"]].sum()
            P(f"{arm:16s} pooled R2 {1 - sse / sstG:.3f}  within-(group,pair) R2 {1 - sse / ssw:+.3f}  RMSE {rmse:.3f}  "
              f"pubs better than MEAN3: {int((pubd.e < pubd.e0).sum())}/{len(pubd)}")
        P("per-publication rows:", G.groupby("publication").size().to_dict())

    # ---------- H make-up and within-campaign R2 ----------
    P("\n=== H make-up of the interior population")
    P(ti.groupby(["publication", "chemotype"]).agg(cells=("cell_id", "nunique"), rows=("y", "size"),
                                                    acid_min=("target_acid_M", "min"),
                                                    acid_max=("target_acid_M", "max")).to_string())
    P("chemotype share of interior rows:", (ti.chemotype.value_counts(normalize=True).round(3)).to_dict())
    # within-campaign R2: SS of y around the (publication, extractant, pair) mean of the targets
    gm = ti.groupby(["publication", "extractant", "pair"]).y.transform("mean")
    ssw = ((ti.y - gm) ** 2).sum()
    P(f"interior: pooled SST {sst_all:.4f}; within (pub,extractant,pair) SS {ssw:.4f} ({ssw / sst_all:.2f} of SST)")
    for arm in ("MEAN", "LINHALF", "PAIRMEAN", "NEAR", "LIN"):
        sse = ((ti[arm] - ti.y) ** 2).sum()
        P(f"  {arm:8s} SSE {sse:.4f}  pooled R2 {1 - sse / sst_all:.3f}  SSE/within-SS {sse / ssw:.2f}  "
          f"RMSE {math.sqrt(sse / len(ti)):.3f}")
    # pair-type-only baseline: corpus mean per pair type from other publications in this population
    pt = []
    for r in ti.itertuples():
        o = t[(t.publication != r.publication) & (t.pair == r.pair)]
        pt.append(o.y.mean() if len(o) else 0.0)
    ti["PAIRTYPE_LOPO"] = pt
    P(f"  pair-type-mean (LOPO, no campaign info) pooled R2 {r2(ti, 'PAIRTYPE_LOPO'):.3f}")
    (OUT / "recipe4_skepB.log").write_text("\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
