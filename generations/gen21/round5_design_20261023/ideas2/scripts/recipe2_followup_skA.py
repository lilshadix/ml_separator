#!/usr/bin/env python
"""SKEPTIC A for idea recipe-2, follow-ups (EXPLORATORY; already-seen cells).
  F1  do DOPT_Q-nested sets read adjacent metals (measured pairs = error-0 credit), by series length?
  F2  ladder - HULL / ladder - DOPT_Q with the top-2 gain publications dropped, and outside the giant cluster.
  F3  where the 2-read La,Ho vs La,Lu difference comes from (per pair type, 14-metal cells, D7).
  F4  legal vs all-sets static choice under the fallback objective: paired interval for the La,Ce-containing sets.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe2_followup_skA.py
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

from r4 import paths  # noqa: E402,F401
from gen20k import data as D  # noqa: E402
import recipe2_analyze_skA as A  # noqa: E402
import recipe2_build_skA as Bd  # noqa: E402

OUT = A.OUT


def main() -> int:
    cells = pd.read_parquet(OUT / "build_cells.parquet")
    npz = {n: dict(np.load(OUT / f"build_k{n}.npz")) for n in A.NS}
    seeds = json.loads((OUT / "build_log.json").read_text())["seeds"]
    nc = len(cells)
    ar = np.arange(nc)
    base = (~cells.c5).to_numpy()

    def hull(n):
        return npz[n]["D7"][ar, cells[f"hull{n}"].to_numpy()]

    def lad(n):
        q = cells[f"lad{n}"].to_numpy()
        return np.where(q >= 0, npz[n]["D7"][ar, np.maximum(q, 0)], hull(n))

    def dq(n):
        out = []
        for s in seeds:
            q = cells[f"dq{n}_{s}"].to_numpy()
            out.append(np.where(q >= 0, npz[n]["D7"][ar, np.maximum(q, 0)], hull(n)))
        return np.mean(out, 0)

    # F1
    rows = []
    for n in A.NS:
        for s in seeds:
            q = cells[f"dq{n}_{s}"].to_numpy()
            for lab, m in (("len<=8", cells.n_metals <= 8), ("len>=13", cells.n_metals >= 13), ("all", cells.n_metals > 0)):
                m = (m & ~cells.c5).to_numpy()
                ok = m & (q >= 0)
                illegal = np.array([not A.LEGAL[n][x] for x in q[ok]])
                rows.append({"n": n, "seed": s, "stratum": lab, "cells": int(m.sum()), "with_set": int(ok.sum()),
                             "share_sets_with_adjacent_metals": float(illegal.mean()) if len(illegal) else np.nan})
    F1 = pd.DataFrame(rows).groupby(["n", "stratum"]).agg(cells=("cells", "first"), with_set=("with_set", "mean"),
                                                          adj_share=("share_sets_with_adjacent_metals", "mean")).reset_index()
    print("=== F1 DOPT_Q sets containing adjacent metals (measured pairs)")
    print(F1.round(3).to_string(index=False))
    # HULL legality too
    for n in A.NS:
        q = cells.loc[base, f"hull{n}"].to_numpy()
        print(f"HULL n={n}: share of cells whose HULL set has adjacent metals:",
              round(float(np.mean([not A.LEGAL[n][x] for x in q])), 3),
              "| len<=8:", round(float(np.mean([not A.LEGAL[n][x] for x in cells.loc[base & (cells.n_metals <= 8).to_numpy(), f'hull{n}']])), 3))
    # short-series: legal-only DQ comparison (replace illegal DQ sets by HULL) to see if DQ's short-series edge is measured pairs
    rows = []
    for n in A.NS:
        out = []
        for s in seeds:
            q = cells[f"dq{n}_{s}"].to_numpy()
            leg = np.array([(x >= 0) and A.LEGAL[n][x] for x in q])
            out.append(np.where(leg, npz[n]["D7"][ar, np.maximum(q, 0)], np.nan))
        dql = np.nanmean(np.where(np.isnan(np.array(out)), hull(n)[None, :], np.array(out)), 0)
        for lab, m in (("len<=8", cells.n_metals <= 8), ("len>=13", cells.n_metals >= 13), ("noC5", cells.n_metals > 0)):
            m = (m & ~cells.c5).to_numpy()
            P = A.Pop(cells[m], "pub")
            d = P.diff(lad(n)[m], dq(n)[m])
            d2 = P.diff(lad(n)[m], dql[m])
            rows.append({"n": n, "stratum": lab, "LAD": P.r2(lad(n)[m]), "DQ": P.r2(dq(n)[m]), "DQ_legal_else_HULL": P.r2(dql[m]),
                         "HULL": P.r2(hull(n)[m]), "LAD-DQ": f"{d['d']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]",
                         "LAD-DQlegal": f"{d2['d']:+.3f} [{d2['lo']:+.3f},{d2['hi']:+.3f}]"})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # F2
    rows = []
    for lab, m in (("noC5", base), ("noC5 minus b0b3296ec5,af7a1cd919", base & ~cells.pub.isin(["pub_b0b3296ec5", "pub_af7a1cd919"]).to_numpy()),
                   ("noC5 minus top-3 LAD-DQ pubs", base & ~cells.pub.isin(["pub_19783c039b", "pub_126bbe7976", "pub_b0b3296ec5"]).to_numpy()),
                   ("noC5 not giant", base & (cells.cluster != "g:pub_74428aeab2").to_numpy())):
        P = A.Pop(cells[m], "pub")
        Pc = A.Pop(cells[m], "cluster")
        for n in A.NS:
            d1, d2 = P.diff(lad(n)[m], hull(n)[m]), P.diff(lad(n)[m], dq(n)[m])
            c1 = Pc.diff(lad(n)[m], hull(n)[m])
            rows.append({"pop": lab, "n": n, "cells": int(m.sum()),
                         "LAD-HULL": f"{d1['d']:+.3f} pub[{d1['lo']:+.3f},{d1['hi']:+.3f}] clu[{c1['lo']:+.3f},{c1['hi']:+.3f}]",
                         "LAD-DQ": f"{d2['d']:+.3f} [{d2['lo']:+.3f},{d2['hi']:+.3f}]"})
    print("\n=== F2 dependence on a few publications")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(pd.DataFrame(rows).to_string(index=False))

    # F3: per pair type, 14-metal noC5 cells: La,Ho vs La,Lu (D7 and linear continuation)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    pos = {c: i for i, c in enumerate(f.cell_id.astype(str))}
    recs = []
    for r in cells[base & (cells.n_metals == 14).to_numpy()].itertuples():
        y = Y[pos[r.cell_id]]
        u = np.zeros(14)
        u[Bd.GD] = Bd.BETA
        for nm, S in (("La,Ho", (0, 9)), ("La,Lu", (0, 13)), ("La,Er", (0, 10))):
            for linx in (False, True):
                L = Bd.lev(Bd.XR, y, u, S, linx=linx)
                for a in range(13):
                    if Bd.ZZ[a + 1] - Bd.ZZ[a] != 1:
                        continue
                    recs.append({"cell": r.cell_id, "pub": r.pub, "dga": r.family == "diglycolamide", "set": nm, "linx": linx,
                                 "pair": f"{Bd.LN[a]}/{Bd.LN[a + 1]}", "y": y[a] - y[a + 1],
                                 "se": ((L[a] - L[a + 1]) - (y[a] - y[a + 1])) ** 2})
    t = pd.DataFrame(recs)
    pv = t.pivot_table(index=["pair"], columns=["linx", "set"], values="se", aggfunc="sum")
    pv["meanY"] = t[(t.set == "La,Ho") & ~t.linx].groupby("pair").y.mean()
    order = [f"{Bd.LN[a]}/{Bd.LN[a + 1]}" for a in range(13) if Bd.ZZ[a + 1] - Bd.ZZ[a] == 1]
    print("\n=== F3 per-pair SSE, 14-metal noC5 cells (n =", t.cell.nunique(), "), D7 constant (False) vs linear (True)")
    print(pv.reindex(order).round(3).to_string())
    heavy = ["Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu"]
    for linx in (False, True):
        g = t[t.linx == linx].groupby(["set", t.pair.isin(heavy)]).se.sum().unstack()
        print("linx", linx, "SSE light+mid (False) vs heavy beyond Ho (True):", g.round(3).to_dict())
    print("mean y of heavy pairs (14-metal, noC5) by DGA:", t[(t.set == "La,Ho") & ~t.linx & t.pair.isin(heavy)]
          .groupby("dga").y.agg(["mean", "std", "count"]).round(3).to_dict())

    # F4: all-sets vs legal-only static choice (fallback objective, LOPO) paired interval
    def oos(n, legal):
        sse = np.zeros(nc)
        h = hull(n)
        for p in cells.loc[base, "pub"].unique():
            te = base & (cells.pub == p).to_numpy()
            tr = base & ~te
            a = npz[n]["D7"][tr]
            tot = np.where(np.isnan(a), h[tr][:, None], a).sum(0)
            if legal:
                tot = np.where(A.LEGAL[n], tot, np.inf)
            q = int(np.argmin(tot))
            col = npz[n]["D7"][:, q]
            sse[te] = np.where(np.isnan(col[te]), h[te], col[te])
        return sse
    P = A.Pop(cells[base], "pub")
    Pc = A.Pop(cells[base], "cluster")
    for n in (3, 4, 5):
        s_all, s_leg = oos(n, False)[base], oos(n, True)[base]
        d, dc = P.diff(s_all, s_leg), Pc.diff(s_all, s_leg)
        d2 = P.diff(s_all, lad(n)[base])
        print(f"F4 n={n}: LOPO all-sets {P.r2(s_all):.3f} vs LOPO legal {P.r2(s_leg):.3f}: {d['d']:+.3f} pub[{d['lo']:+.3f},{d['hi']:+.3f}]"
              f" clu[{dc['lo']:+.3f},{dc['hi']:+.3f}] LOCO[{dc['loo_min']:+.3f},{dc['loo_max']:+.3f}]; all-sets OOS - frozen ladder {d2['d']:+.3f} [{d2['lo']:+.3f},{d2['hi']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
