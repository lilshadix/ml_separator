#!/usr/bin/env python
"""Theme "minimal-measurement recipes", extra quick checks (EXPLORATORY; 229 already-seen C3 cells).

  E1  head-to-head read-choice contrasts with publication-bootstrap intervals:
        no prior (D7): LOPO-static recipe vs HULL (lightest + heaviest first, then max-min radius) vs random;
        own prior (SH_own): HULL vs LOPO-static vs random;
        "reads saved": SH_own at n reads vs D7 at m reads (HULL both).
  E2  direction accuracy on |y| >= 0.1 unmeasured pairs for D7 / SH_own under HULL at n = 2..4, and COPY at n = 0.
  E3  ICP-MS condition skipping: acid-only own siblings, linear interpolation of the sibling log D curves in
      log10[acid] (bracketing), against mean-copy and nearest-acid copy, at 0 reads and at 2 HULL reads.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe_extra.py
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r4.refs import HIST_D7_BETA  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402
import recipe_analyze as RA  # noqa: E402
from recipe_build import interp_batch  # noqa: E402

OUT = RA.OUT
NMAX = 6
X = interp.X_R
K_SH = 1.78


def hull_order(Mi):
    x = X[Mi]
    m = len(Mi)
    order = [0, m - 1]
    while len(order) < m:
        rest = [q for q in range(m) if q not in order]
        dmin = [min(abs(x[q] - x[o]) for o in order) for q in rest]
        order.append(rest[int(np.argmax(dmin))])
    return order


def main() -> int:
    res = {}
    cells = pd.read_parquet(OUT / "recipe_cells.parquet")
    pairs = pd.read_parquet(OUT / "recipe_pairs.parquet")
    npz = {n: dict(np.load(OUT / f"recipe_k{n}.npz")) for n in range(1, NMAX + 1)}
    gidx = {n: {cmb: q for q, cmb in enumerate(itertools.combinations(range(N_LN), n))} for n in range(1, NMAX + 1)}
    nc = len(cells)
    metals = [[LANTHANIDES.index(s) for s in m.split(",")] for m in cells.metals]
    hull_g = {n: np.array([gidx[n][tuple(sorted(Mi[q] for q in hull_order(Mi)[:n]))] if len(Mi) >= n else -1
                           for Mi in metals]) for n in range(1, NMAX + 1)}

    def sse(arm, n, pol, sub):
        idx = sub.ci.to_numpy()
        if n == 0:
            s = pairs[pairs.n_read == 0].groupby("ci")[f"rnd_{arm}"].sum()
            return s.reindex(range(nc)).to_numpy()[idx]
        if pol == "RND":
            s = pairs[pairs.n_read == n].groupby("ci")[f"rnd_{arm}"].sum()
            return s.reindex(range(nc)).fillna(0).to_numpy()[idx]
        if pol == "HULL":
            s = pairs[pairs.n_read == n].groupby("ci")[f"hull_{arm}"].sum()
            return s.reindex(range(nc)).fillna(0).to_numpy()[idx]
        raise ValueError(pol)

    pops = {"ALL229": cells, "noC5_214": cells[~cells.c5], "OWN1_noC5": cells[(cells.n_own >= 1) & ~cells.c5],
            "OWN1_C5": cells[cells.n_own >= 1]}

    # ---------------- E1 contrasts ----------------
    E1 = []
    for pk in ("ALL229", "noC5_214", "OWN1_noC5"):
        sub = pops[pk]
        P = RA.Pop(sub, pk)
        for arm in (("D7",) if pk != "OWN1_noC5" else ("D7", "SH_own")):
            for n in (2, 3, 4, 5):
                lopo, _ = RA_lopo(npz, arm, n, sub, sse(arm, n, "HULL", sub))
                hull = sse(arm, n, "HULL", sub)
                rnd = sse(arm, n, "RND", sub)
                for nm, a, b in (("HULL-RND", hull, rnd), ("LOPO-HULL", lopo, hull)):
                    d = P.summ(a, b)
                    E1.append({"pop": pk, "arm": arm, "n": n, "contrast": nm, "R2": d["R2"], "d": d["d"],
                               "d_lo": d["d_lo"], "d_hi": d["d_hi"], "P>0": d["P_d_gt0"]})
    sub = pops["OWN1_noC5"]
    P = RA.Pop(sub, "OWN1_noC5")
    for n in (0, 2, 3, 4):
        a = sse("SH_own" if n else "SIB_own", n, "HULL", sub)
        for m in range(2, NMAX + 1):
            d = P.summ(a, sse("D7", m, "HULL", sub))
            E1.append({"pop": "OWN1_noC5", "arm": f"SH_own@{n} vs D7@{m}", "n": n, "contrast": "equiv",
                       "R2": d["R2"], "d": d["d"], "d_lo": d["d_lo"], "d_hi": d["d_hi"], "P>0": d["P_d_gt0"]})
    sub = pops["OWN1_C5"]
    P = RA.Pop(sub, "OWN1_C5")
    for n in (0, 2, 3, 4):
        a = sse("SH_own" if n else "SIB_own", n, "HULL", sub)
        for m in range(2, NMAX + 1):
            d = P.summ(a, sse("D7", m, "HULL", sub))
            E1.append({"pop": "OWN1_C5", "arm": f"SH_own@{n} vs D7@{m}", "n": n, "contrast": "equiv",
                       "R2": d["R2"], "d": d["d"], "d_lo": d["d_lo"], "d_hi": d["d_hi"], "P>0": d["P_d_gt0"]})
    E1 = pd.DataFrame(E1)
    E1.to_csv(OUT / "recipe_extra_E1.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print("=== E1 contrasts (publication bootstrap 95%)")
        print(E1.round(3).to_string(index=False))

    # ---------------- E2 direction accuracy under HULL ----------------
    E2 = []
    for pk in ("OWN1_noC5", "noC5_214"):
        sub = pops[pk]
        idx = sub.ci.to_numpy()
        q0 = pairs[(pairs.n_read == 0) & pairs.ci.isin(idx)]
        big = q0.y.abs() >= 0.1
        for arm in (["SIB_own", "D7"] if pk == "OWN1_noC5" else ["D7"]):
            if arm == "D7":
                acc0 = float((np.sign(q0[f"pred_{arm}"][big]) == np.sign(q0.y[big])).mean())
            else:
                acc0 = float((np.sign(q0[f"pred_{arm}"][big]) == np.sign(q0.y[big])).mean())
            E2.append({"pop": pk, "arm": arm, "n": 0, "dir_acc": acc0, "n_rows": int(big.sum())})
        for arm in (["SH_own", "D7"] if pk == "OWN1_noC5" else ["D7"]):
            for n in (2, 3, 4, 5):
                g = hull_g[n][idx]
                ok = np.array([npz[n][f"dok_{arm}"][c, q] if q >= 0 else np.nan for c, q in zip(idx, g)])
                nn = np.array([npz[n]["dirn"][c, q] if q >= 0 else np.nan for c, q in zip(idx, g)])
                E2.append({"pop": pk, "arm": arm, "n": n, "dir_acc": float(np.nansum(ok) / np.nansum(nn)),
                           "n_rows": int(np.nansum(nn))})
    E2 = pd.DataFrame(E2)
    E2.to_csv(OUT / "recipe_extra_E2.csv", index=False)
    print("\n=== E2 direction accuracy (|y| >= 0.1, unmeasured pairs; HULL reads)")
    print(E2.round(3).to_string(index=False))

    # ---------------- E3 acid interpolation (ICP-MS condition skipping) ----------------
    bench = D.load_bench()
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
        aonly = [j for j, fm in zip(own, fams) if fm == "acid_conc"]
        i = pos[r.cell_id]
        Mi = metals[r.ci]
        if len(aonly) < 2 or not np.isfinite(acid[i]):
            continue
        la = math.log10(acid[i])
        js = [pos[j] for j in aonly if np.isfinite(acid[pos[j]]) and acid[pos[j]] > 0]
        lo = [j for j in js if math.log10(acid[j]) < la - 1e-9]
        hi = [j for j in js if math.log10(acid[j]) > la + 1e-9]
        if lo and hi:
            jl = max(lo, key=lambda j: acid[j])
            jh = min(hi, key=lambda j: acid[j])
            kind = "bracket"
        else:
            side = sorted(js, key=lambda j: abs(math.log10(acid[j]) - la))
            if len(side) < 2:
                continue
            jl, jh = side[0], side[1]
            kind = "extrap"
        x0, x1 = math.log10(acid[jl]), math.log10(acid[jh])
        if abs(x1 - x0) < 1e-9:
            continue
        tt = (la - x0) / (x1 - x0)
        yi = Y[i, Mi]
        T_int = Y[jl, Mi] + (Y[jh, Mi] - Y[jl, Mi]) * tt
        T_near = Y[min(js, key=lambda j: abs(math.log10(acid[j]) - la)), Mi]
        T_mean_all = Y[np.ix_([pos[j] for j in own], Mi)].mean(0)
        T_mean_a = Y[np.ix_(js, Mi)].mean(0)
        T_pair = 0.5 * (Y[jl, Mi] + Y[jh, Mi])
        m = len(Mi)
        pa = np.array([p for p in range(m - 1) if Z_VEC[Mi[p + 1]] - Z_VEC[Mi[p]] == 1])
        pb = pa + 1
        u = np.zeros(m)
        if CO.GD in Mi:
            u[Mi.index(CO.GD)] = HIST_D7_BETA
        hord = hull_order(Mi)
        R2m = np.zeros((1, m), bool)
        R2m[0, hord[:2]] = True
        d7_2 = (u + interp_batch(R2m, yi - u, X[Mi])[0])
        n_own = len(own)
        w = n_own / (n_own + K_SH)
        for q in range(len(pa)):
            a, b = pa[q], pb[q]
            rec = {"cell_id": r.cell_id, "publication": r.publication, "kind": kind, "t": tt,
                   "dlogacid": abs(x1 - x0), "y": yi[a] - yi[b]}
            for nm, T in (("INTERP", T_int), ("NEAR", T_near), ("MEAN_OWN", T_mean_all), ("MEAN_ACID", T_mean_a),
                          ("PAIRMEAN", T_pair)):
                rec[f"{nm}_0"] = T[a] - T[b]
                sib2 = T + interp_batch(R2m, yi - T, X[Mi])[0]
                sh2 = w * sib2 + (1 - w) * d7_2
                rec[f"{nm}_2"] = sh2[a] - sh2[b]
            rec["D7_2"] = d7_2[a] - d7_2[b]
            rows.append(rec)
    E3 = pd.DataFrame(rows)
    E3.to_parquet(OUT / "recipe_extra_E3_rows.parquet", index=False)
    out3 = []
    for kind in ("bracket", "extrap", "all"):
        g = E3 if kind == "all" else E3[E3.kind == kind]
        if not len(g):
            continue
        cc = g.groupby("cell_id").agg(publication=("publication", "first")).reset_index()
        cc["N"] = g.groupby("cell_id").size().reindex(cc.cell_id).to_numpy()
        cc["sy"] = g.groupby("cell_id").y.sum().reindex(cc.cell_id).to_numpy()
        cc["syy"] = (g.assign(y2=g.y ** 2).groupby("cell_id").y2.sum()).reindex(cc.cell_id).to_numpy()
        P = RA.Pop(cc, kind)

        def cs(col):
            return (g.assign(e=(g[col] - g.y) ** 2).groupby("cell_id").e.sum()).reindex(cc.cell_id).to_numpy()
        for k in (0, 2):
            for nm in ("INTERP", "NEAR", "MEAN_ACID", "MEAN_OWN", "PAIRMEAN"):
                d = P.summ(cs(f"{nm}_{k}"), cs(f"MEAN_OWN_{k}"))
                out3.append({"kind": kind, "cells": len(cc), "pubs": cc.publication.nunique(), "rows": len(g),
                             "reads": k, "arm": nm, "R2": d["R2"], "lo": d["lo"], "hi": d["hi"],
                             "d_vs_MEAN_OWN": d["d"], "d_lo": d["d_lo"], "d_hi": d["d_hi"]})
        d = P.summ(cs("D7_2"))
        out3.append({"kind": kind, "cells": len(cc), "pubs": cc.publication.nunique(), "rows": len(g), "reads": 2,
                     "arm": "D7 (no prior)", "R2": d["R2"], "lo": d["lo"], "hi": d["hi"]})
    out3 = pd.DataFrame(out3)
    out3.to_csv(OUT / "recipe_extra_E3.csv", index=False)
    print("\n=== E3 acid-only condition interpolation (own siblings differing only in acid concentration), noC5")
    print(out3.round(3).to_string(index=False))
    print("E3 dlogacid quantiles (bracket):", E3[E3.kind == "bracket"].groupby("cell_id").dlogacid.first()
          .describe().round(3).to_dict())
    print("E3 pubs:", E3.groupby("kind").publication.unique().to_dict())
    return 0


def RA_lopo(npz, arm, n, sub, hull_sse, cov=0.5):
    """Same LOPO-static rule as recipe_analyze.lopo_static (copied to avoid recomputing the full analysis)."""
    combos = list(itertools.combinations(range(N_LN), n))
    a = npz[n][f"sse_{arm}"][sub.ci.to_numpy()]
    avail = ~np.isnan(a)
    a0 = np.nan_to_num(a)
    N, sy, syy = (sub[k].to_numpy(float) for k in ("N", "sy", "syy"))
    out = np.zeros(len(sub))
    tops = {}
    for p in sub.publication.unique():
        tr = (sub.publication != p).to_numpy()
        ybar = sy[tr].sum() / N[tr].sum()
        sstc = syy - 2 * ybar * sy + N * ybar ** 2
        num = a0[tr].sum(0)
        den = (avail[tr] * sstc[tr, None]).sum(0)
        covg = avail[tr].mean(0)
        score = np.where((covg >= cov) & (den > 0), num / np.where(den > 0, den, 1), np.inf)
        rank = np.argsort(score)
        rank = rank[np.isfinite(score[rank])]
        tops[p] = ",".join(LANTHANIDES[q] for q in combos[rank[0]]) if len(rank) else ""
        for r in np.where(~tr)[0]:
            ok = rank[avail[r, rank]]
            out[r] = a0[r, ok[0]] if len(ok) else hull_sse[r]
    return out, tops


if __name__ == "__main__":
    raise SystemExit(main())
