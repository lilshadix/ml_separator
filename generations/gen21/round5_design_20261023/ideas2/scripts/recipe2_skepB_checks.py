#!/usr/bin/env python
"""Skeptic B checks on idea recipe-2 (fixed training-free metal ladder).  EXPLORATORY: all 229 C3 cells were already
seen; every number here is a lead, not a test.  Reads only ideas2/recipe_out (built by recipe_build.py), the V1 bench
and the gen21 fold cache.  Writes only ideas2/recipe2_skepB_out/.

  K1  Ladder / HULL / DOPT_Q-nested / random by stratum: DGA vs non-DGA donor, giant cluster vs rest, covered vs
      HULL-fallback cells; publication- and cluster-bootstrap intervals on the paired differences.
  K2  Leave-one-CLUSTER-out static legal-set selection (13 clusters; the giant cluster holds ~71 % of rows), picks
      per fold and score vs HULL.
  K3  Nesting: the proposed ladder is not nested (Ho dropped at 4 reads).  Score a nested ladder and the
      'sequential' cost (La,Ho,Nd,Er,Lu = 5 experiments to reach the 4-read set).
  K4  Where La,Ho beats La,Lu at 2 reads: per-pair-type SSE on cells that measure La, Ho and Lu, by donor class.
  K5  Unit-level heterogeneity of ladder - HULL (publication x chemotype units) and a normal-approximation G80.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe2_skepB_checks.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import donor as DN  # noqa: E402
from r4.refs import HIST_D7_BETA  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402

RO = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe_out")
OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe2_skepB_out")
LADDER = {2: "La,Ho", 3: "La,Nd,Ho", 4: "La,Nd,Er,Lu", 5: "La,Nd,Gd,Er,Lu"}
NESTED = {2: "La,Ho", 3: "La,Nd,Ho", 4: "La,Nd,Ho,Lu", 5: "La,Nd,Gd,Ho,Lu"}
B = 2000
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def gid(n: int):
    return {c: q for q, c in enumerate(itertools.combinations(range(N_LN), n))}


def setq(n: int, s: str) -> int:
    return gid(n)[tuple(sorted(LANTHANIDES.index(m) for m in s.split(",")))]


class Agg:
    """Pooled R2 with bootstrap over a unit column (publication or cluster)."""

    def __init__(self, sub: pd.DataFrame, unit: str, seed: int = 20261023):
        self.sub = sub.reset_index(drop=True)
        u = sorted(self.sub[unit].unique())
        self.ui = self.sub[unit].map({x: i for i, x in enumerate(u)}).to_numpy()
        self.G = np.zeros((len(self.sub), len(u)))
        self.G[np.arange(len(self.sub)), self.ui] = 1
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(float)
        self.N, self.sy, self.syy = (self.sub[k].to_numpy(float) @ self.G for k in ("N", "sy", "syy"))
        self.nu = len(u)

    def r2(self, sse, w=None):
        s = sse @ self.G
        if w is None:
            return 1 - s.sum() / (self.syy.sum() - self.sy.sum() ** 2 / self.N.sum())
        return 1 - (w @ s) / (w @ self.syy - (w @ self.sy) ** 2 / (w @ self.N))

    def diff(self, a, b):
        d = self.r2(a, self.W) - self.r2(b, self.W)
        return float(self.r2(a) - self.r2(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_parquet(RO / "recipe_cells.parquet")
    pairs = pd.read_parquet(RO / "recipe_pairs.parquet")
    npz = {n: np.load(RO / f"recipe_k{n}.npz")["sse_D7"] for n in range(2, 7)}
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    ext_of = dict(zip(f.cell_id.astype(str), f.extractant.astype(str)))
    cells["dga"] = cells.cell_id.map(lambda c: DN.is_dga(ext_of[c]))
    giant = cells.cluster.value_counts().idxmax()
    cells["giant"] = cells.cluster == giant
    nc = len(cells)

    # ---- per-cell SSE arrays
    def hull(n):
        return pairs[pairs.n_read == n].groupby("ci")["hull_D7"].sum().reindex(range(nc)).fillna(0).to_numpy()

    def rnd(n):
        return pairs[pairs.n_read == n].groupby("ci")["rnd_D7"].sum().reindex(range(nc)).fillna(0).to_numpy()

    def fixed(n, s, fb=None):
        v = npz[n][:, setq(n, s)].astype(float)
        has = ~np.isnan(v)
        return np.where(has, v, hull(n) if fb is None else fb), has

    # DOPT_Q nested sets from the fold cache (BPG, 5 seeds), seed-mean per-cell SSE; HULL fallback
    g = {n: gid(n) for n in range(2, 7)}
    ci_of = dict(zip(cells.cell_id, cells.ci))
    dq = {n: np.zeros((len(paths.DEV_SEEDS), nc)) for n in range(2, 6)}
    first2 = []
    for si, seed in enumerate(paths.DEV_SEEDS):
        got = {n: np.zeros(nc, bool) for n in range(2, 6)}
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            for cell in rec["outer"]["cells"]:
                c = ci_of.get(cell["cell_id"])
                if c is None:
                    continue
                seq = cell["seqs"]["DOPT_Q"]
                for k in range(1, 5):
                    if k <= len(seq):
                        rd = tuple(sorted({m for p in seq[:k] for m in p}))
                        n = len(rd)
                        if n in dq and not got[n][c]:
                            dq[n][si, c] = float(npz[n][c, g[n][rd]])
                            got[n][c] = True
                            if n == 2:
                                first2.append(",".join(LANTHANIDES[q] for q in rd))
        for n in dq:
            dq[n][si, ~got[n]] = hull(n)[~got[n]]
    dqm = {n: dq[n].mean(0) for n in dq}
    say("DOPT_Q 2-read sets (all seeds x cells):", pd.Series(first2).value_counts().head(8).to_dict())

    # ================= K1 strata =================
    say("\n=== K1  ladder vs HULL / DOPT_Q / random, D7 arm, by stratum (pooled R2; diffs with pub- and cluster-bootstrap)")
    base = cells[~cells.c5]
    strata = {"noC5_all": base, "noC5_DGA": base[base.dga], "noC5_nonDGA": base[~base.dga],
              "noC5_giant": base[base.giant], "noC5_notgiant": base[~base.giant],
              "noC5_14metal": base[base.n_metals == 14], "noC5_le8metal": base[base.n_metals <= 8],
              "ALL229_nonDGA": cells[~cells.dga]}
    rows = []
    for sk, sub in strata.items():
        idx = sub.ci.to_numpy()
        Ap = Agg(sub, "publication")
        Ac = Agg(sub, "cluster")
        for n in (2, 3, 4, 5):
            lad, has = fixed(n, LADDER[n])
            h, r, d = hull(n)[idx], rnd(n)[idx], dqm[n][idx]
            l = lad[idx]
            rec = {"stratum": sk, "n": n, "cells": len(idx), "pubs": Ap.nu, "clusters": Ac.nu,
                   "coverage": float(has[idx].mean()), "LADDER": Ap.r2(l), "HULL": Ap.r2(h), "DOPTQ": Ap.r2(d),
                   "RANDOM": Ap.r2(r)}
            for nm, (a, b) in {"L-H": (l, h), "L-D": (l, d), "H-R": (h, r)}.items():
                p = Ap.diff(a, b)
                c = Ac.diff(a, b)
                rec[nm] = p[0]
                rec[f"{nm}_pub"] = f"[{p[1]:+.3f},{p[2]:+.3f}]"
                rec[f"{nm}_clu"] = f"[{c[1]:+.3f},{c[2]:+.3f}]"
            # covered-only ladder-HULL
            cov = has[idx]
            if cov.sum() > 0:
                Acv = Agg(sub[cov], "publication")
                rec["L-H_covered"] = Acv.r2(l[cov]) - Acv.r2(h[cov])
                rec["cov_cells"] = int(cov.sum())
            rows.append(rec)
    K1 = pd.DataFrame(rows)
    K1.to_csv(OUT / "K1_strata.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        say(K1.round(3).to_string(index=False))

    # ================= K2 leave-one-cluster-out static legal selection =================
    say("\n=== K2  leave-one-CLUSTER-out static legal-set selection (noC5), each held-out cell takes best-ranked set it has")
    sub = base.reset_index(drop=True)
    idx = sub.ci.to_numpy()
    Ap = Agg(sub, "publication")
    rows = []
    for n in (2, 3, 4, 5):
        combos = list(itertools.combinations(range(N_LN), n))
        adj = np.array([any(Z_VEC[b] - Z_VEC[a] == 1 for a, b in itertools.combinations(c, 2)) for c in combos])
        a = np.where(adj[None, :], np.nan, npz[n][idx].astype(float))
        avail = ~np.isnan(a)
        a0 = np.nan_to_num(a)
        N, sy, syy = (sub[k].to_numpy(float) for k in ("N", "sy", "syy"))
        h = hull(n)[idx]
        out = np.zeros(len(sub))
        tops = {}
        for cl in sub.cluster.unique():
            tr = (sub.cluster != cl).to_numpy()
            ybar = sy[tr].sum() / N[tr].sum()
            sstc = syy - 2 * ybar * sy + N * ybar ** 2
            num = a0[tr].sum(0)
            den = (avail[tr] * sstc[tr, None]).sum(0)
            covg = avail[tr].mean(0)
            score = np.where((covg >= 0.5) & (den > 0), num / np.where(den > 0, den, 1), np.inf)
            rank = np.argsort(score)
            rank = rank[np.isfinite(score[rank])]
            tops[cl] = ",".join(LANTHANIDES[q] for q in combos[rank[0]]) if len(rank) else ""
            for rr in np.where(~tr)[0]:
                ok = rank[avail[rr, rank]]
                out[rr] = a0[rr, ok[0]] if len(ok) else h[rr]
        lad, _ = fixed(n, LADDER[n])
        p = Ap.diff(out, h)
        rows.append({"n": n, "LOCO_static": Ap.r2(out), "HULL": Ap.r2(h), "LADDER_insample": Ap.r2(lad[idx]),
                     "LOCO-HULL": p[0], "pub95": f"[{p[1]:+.3f},{p[2]:+.3f}]",
                     "pick_when_giant_out": tops[giant], "picks": pd.Series(tops).value_counts().to_dict()})
    K2 = pd.DataFrame(rows)
    K2.to_csv(OUT / "K2_loco_static.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_colwidth", 120):
        say(K2.round(3).to_string(index=False))

    # ================= K3 nesting =================
    say("\n=== K3  nesting (noC5, D7).  Proposed ladder drops Ho at 4 reads; nested alternative keeps it")
    rows = []
    for n in (2, 3, 4, 5):
        lad, hl = fixed(n, LADDER[n])
        nes, hn = fixed(n, NESTED[n])
        h = hull(n)[idx]
        rows.append({"n": n, "LADDER": Ap.r2(lad[idx]), "NESTED": Ap.r2(nes[idx]), "HULL": Ap.r2(h),
                     "DOPTQ": Ap.r2(dqm[n][idx]), "cov_ladder": hl[idx].mean(), "cov_nested": hn[idx].mean()})
    # sequential: a lab that follows the ladder step by step holds La,Nd,Ho,Er,Lu (5 experiments) at the 4-read stop
    seq5, hs = fixed(5, "La,Nd,Ho,Er,Lu", fb=hull(5))
    rows.append({"n": "5 exp (La,Nd,Ho,Er,Lu: sequential ladder at its 4-read stop)", "LADDER": Ap.r2(seq5[idx]),
                 "HULL": Ap.r2(hull(5)[idx]), "DOPTQ": Ap.r2(dqm[5][idx]), "cov_ladder": hs[idx].mean()})
    K3 = pd.DataFrame(rows)
    K3.to_csv(OUT / "K3_nesting.csv", index=False)
    with pd.option_context("display.width", 300):
        say(K3.round(3).to_string(index=False))

    # ================= K4 per-pair La,Ho vs La,Lu at 2 reads =================
    say("\n=== K4  2 reads: La,Ho vs La,Lu per pair type, cells measuring La, Ho, Lu (D7, all 229 incl. C5 flagged)")
    Y = np.asarray(bench.Y, float)
    pos = {c: i for i, c in enumerate(f.cell_id.astype(str))}
    X = interp.X_R
    GD = CO.GD
    recs = []
    iLa, iHo, iLu = (LANTHANIDES.index(m) for m in ("La", "Ho", "Lu"))
    for _, c in cells.iterrows():
        i = pos[c.cell_id]
        Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        if not {iLa, iHo, iLu} <= set(Mi):
            continue
        u = np.zeros(N_LN)
        u[GD] = HIST_D7_BETA if GD in Mi else 0.0
        preds = {}
        for nm, S in (("LaHo", (iLa, iHo)), ("LaLu", (iLa, iLu))):
            ym = np.full(N_LN, np.nan)
            ym[list(S)] = Y[i, list(S)] - u[list(S)]
            preds[nm] = interp.pwl_levels(ym, Mi, "R") + u
        for p in range(len(Mi) - 1):
            a_, b_ = Mi[p], Mi[p + 1]
            if Z_VEC[b_] - Z_VEC[a_] != 1:
                continue
            y = Y[i, a_] - Y[i, b_]
            rec = {"cell_id": c.cell_id, "c5": c.c5, "dga": c.dga, "pair": f"{LANTHANIDES[a_]}/{LANTHANIDES[b_]}",
                   "y": y}
            for nm, lev in preds.items():
                rec[f"se_{nm}"] = (lev[a_] - lev[b_] - y) ** 2
                rec[f"pred_{nm}"] = lev[a_] - lev[b_]
            recs.append(rec)
    K4 = pd.DataFrame(recs)
    K4.to_parquet(OUT / "K4_pairs.parquet", index=False)
    order = [f"{LANTHANIDES[q]}/{LANTHANIDES[q + 1]}" for q in range(N_LN - 1) if Z_VEC[q + 1] - Z_VEC[q] == 1]
    for nm, s in (("noC5 DGA", K4[~K4.c5 & K4.dga]), ("noC5 nonDGA", K4[~K4.c5 & ~K4.dga])):
        t = s.groupby("pair").agg(rows=("y", "size"), mean_y=("y", "mean"), sse_LaHo=("se_LaHo", "sum"),
                                  sse_LaLu=("se_LaLu", "sum")).reindex(order).dropna()
        t["LaHo-LaLu"] = t.sse_LaHo - t.sse_LaLu
        sst = ((s.y - s.y.mean()) ** 2).sum()
        say(f"-- {nm}: cells {s.cell_id.nunique()}, rows {len(s)}; R2 LaHo {1 - s.se_LaHo.sum() / sst:.3f}, "
            f"LaLu {1 - s.se_LaLu.sum() / sst:.3f}")
        say(t.round(3).to_string())
    # heavy-tail chemistry: cells whose mean heavy (>= Ho) adjacent log SF is clearly nonzero
    hv = K4[~K4.c5 & K4.pair.isin(["Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu"])]
    hm = hv.groupby("cell_id").agg(mean_heavy_y=("y", "mean"), dga=("dga", "first"),
                                   d_sse=("se_LaHo", "sum"), d2=("se_LaLu", "sum"))
    hm["LaHo_minus_LaLu_heavySSE"] = hm.d_sse - hm.d2
    say("-- heavy-block (Ho/Er..Yb/Lu) mean y and SSE(LaHo)-SSE(LaLu) by |mean heavy y| tercile (noC5):")
    hm["band"] = pd.cut(hm.mean_heavy_y.abs(), [0, 0.05, 0.15, 10], include_lowest=True)
    say(hm.groupby(["band", "dga"], observed=True).agg(cells=("d_sse", "size"),
                                                         dSSE=("LaHo_minus_LaLu_heavySSE", "sum")).round(3).to_string())

    # ================= K5 unit heterogeneity and G80 =================
    say("\n=== K5  unit-level ladder - HULL (publication x chemotype units, noC5, covered cells only), relative SSE")
    from scipy import stats as SS
    for n in (2, 4):
        lad, has = fixed(n, LADDER[n])
        h = hull(n)
        t = cells[~cells.c5 & has].copy()
        t = t.assign(L=lad[t.ci.to_numpy()], H=h[t.ci.to_numpy()])
        t["unit"] = t.publication + "|" + t.chemotype
        u = t.groupby("unit").agg(L=("L", "sum"), H=("H", "sum"), dga=("dga", "first"))
        u = u[(u.H > 0) & (u.L > 0)]
        lr = np.log(u.L / u.H)
        m, sd = float(lr.mean()), float(lr.std(ddof=1))
        g80 = ((SS.norm.ppf(0.975) + SS.norm.ppf(0.8)) * sd / abs(m)) ** 2 if m != 0 else np.inf
        say(f"n={n}: units {len(u)}, ladder better in {int((lr < 0).sum())}/{len(u)}, mean log(SSE_L/SSE_H) {m:+.3f}, "
            f"sd {sd:.3f}, normal-approx G80 (two-sided .05, units like these) {g80:.0f};  "
            f"nonDGA units {int((~u.dga).sum())}: mean {float(lr[~u.dga].mean()) if (~u.dga).any() else float('nan'):+.3f}")
    (OUT / "recipe2_skepB_checks.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
