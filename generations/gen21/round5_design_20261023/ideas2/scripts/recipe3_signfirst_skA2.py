#!/usr/bin/env python
"""Skeptic A for recipe-3, part 2 (EXPLORATORY, already-seen 229 C3 cells).  Reads only skA outputs of part 1
(recipe3_skA_out/skA_rows.parquet, skA_sibrows.parquet).  Questions:
  (a) where does COPY's margin over always-heavier come from (pair, publication) on the covered rows?
  (b) campaign-artefact control: does an OTHER extractant of the same campaign beat always-heavier on its own
      covered rows?  (if yes, part of 'own' sign transfer may be campaign fingerprint, not chemistry)
  (c) family-level table on all covered rows (the kill criterion's unit)
  (d) the 'cuts confirmatory experiments roughly in half' claim: metals that still must be read per cell
  (e) literature leg per pair vs always-heavier
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe3_skA_out")
log = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


def main() -> int:
    R = pd.read_parquet(OUT / "skA_rows.parquet")
    R = R[~R.c5].copy()
    R["big"] = R.y.abs() >= 0.1

    # (a) margin decomposition
    own = R[R.n_own >= 1].copy()
    own["sel"] = own.copy_own.abs() >= 0.1
    own["okC"] = np.sign(own.copy_own) == np.sign(own.y)
    own["okH"] = own.y < 0
    d = own[own.big & own.sel]
    d = d.assign(win=(d.okC & ~d.okH).astype(int), loss=(~d.okC & d.okH).astype(int))
    P("=== (a) COPY vs always-heavier on covered |y|>=.1 rows (n=%d): wins %d, losses %d, net %+.3f" %
      (len(d), d.win.sum(), d.loss.sum(), (d.win.sum() - d.loss.sum()) / len(d)))
    P("by pair:\n" + d.groupby("pair")[["win", "loss"]].sum().assign(n=d.groupby("pair").size()).to_string())
    gp = d.groupby("pub")[["win", "loss"]].sum().assign(n=d.groupby("pub").size(), net=lambda g: g.win - g.loss)
    gp = gp.sort_values("net", ascending=False)
    P("by publication:\n" + gp.to_string())
    tot = gp.net.sum()
    P(f"net wins {tot}; top-1 pub share {gp.net.iloc[0] / tot:.2f}; top-3 share {gp.net.iloc[:3].sum() / tot:.2f}")
    P("same on ALL covered rows (incl |y|<0.1):")
    d2 = own[own.sel].assign(win=lambda g: (g.okC & ~g.okH).astype(int), loss=lambda g: (~g.okC & g.okH).astype(int))
    g2 = d2.groupby("pub")[["win", "loss"]].sum().assign(net=lambda g: g.win - g.loss).sort_values("net", ascending=False)
    P(g2.to_string())
    P(f"  net {g2.net.sum()} of {len(d2)} rows; top-3 share {g2.net.iloc[:3].sum() / max(g2.net.sum(), 1):.2f}")
    # without Eu/Gd (the Gd break, known) and without Gd pairs
    for excl in (["Eu/Gd"], ["Eu/Gd", "Gd/Tb"]):
        e = d[~d.pair.isin(excl)]
        P(f"  excl {excl}: COPY {e.okC.mean():.3f} always-heavier {e.okH.mean():.3f} n {len(e)}")

    # (b) campaign-artefact control: xext, xpub, chem against always-heavier on their OWN covered rows
    P("\n=== (b) each pool vs always-heavier and D7@2 on that pool's own covered |y|>=.1 rows ===")
    for k in ("own", "xext", "xpub", "chem"):
        q = R[R[f"n_{k}"] >= 1]
        m = q.big & (q[f"copy_{k}"].abs() >= 0.1)
        okc = (np.sign(q[f"copy_{k}"][m]) == np.sign(q.y[m])).mean()
        P(f"  {k}: n {int(m.sum())} pubs {q[m].pub.nunique()} COPY {okc:.3f} always-heavier {(q.y[m] < 0).mean():.3f} "
          f"D7@2 {(np.sign(q.d7_2[m]) == np.sign(q.y[m])).mean():.3f} D7@4 {(np.sign(q.d7_4[m]) == np.sign(q.y[m])).mean():.3f}")
        ma = q[f"copy_{k}"].abs() >= 0.1
        okca = (np.sign(q[f"copy_{k}"][ma]) == np.sign(q.y[ma])).mean()
        P(f"      all covered rows: n {int(ma.sum())} COPY {okca:.3f} always-heavier {(q.y[ma] < 0).mean():.3f}")
    # xext restricted to the own population's cells (same targets)
    q = R[(R.n_own >= 1) & (R.n_xext >= 1)]
    m = q.big & (q.copy_xext.abs() >= 0.1) & (q.copy_own.abs() >= 0.1)
    P(f"  cells with both own & xext, rows covered by both (|y|>=.1): n {int(m.sum())} pubs {q[m].pub.nunique()}: "
      f"own {(np.sign(q.copy_own[m]) == np.sign(q.y[m])).mean():.3f} xext {(np.sign(q.copy_xext[m]) == np.sign(q.y[m])).mean():.3f}"
      f" always-heavier {(q.y[m] < 0).mean():.3f}")

    # (c) family-level table, all covered rows
    P("\n=== (c) family table (pub|extractant), own, all rows with |COPY|>=0.1 ===")
    o = own[own.sel].assign(family=lambda g: g.pub + "|" + g.ext.str[:25])
    gf = o.groupby("family").agg(n=("okC", "size"), acc=("okC", "mean"), ah=("okH", "mean"),
                                 cluster=("cluster", "first")).sort_values("n", ascending=False)
    P(gf.round(3).to_string())
    P(f"  families {len(gf)}; acc<0.90: {(gf.acc < 0.9).sum()}; acc < always-heavier+0.05: {(gf.acc < gf.ah + 0.05).sum()};"
      f" families with n>=10: {(gf.n >= 10).sum()}, of which acc<0.90: {((gf.n >= 10) & (gf.acc < 0.9)).sum()}")

    # (d) metals still to read per cell under the recipe (radiotracer cost model)
    P("\n=== (d) metals that must still be read per cell (radiotracer), own cells ===")
    rem = {"Er/Tm", "Ho/Er", "La/Ce"}
    res = []
    for cid, g in own.groupby("cell_id"):
        metals = set()
        for pr in g.pair:
            a, b = pr.split("/")
            metals |= {a, b}
        need1 = set()   # pairs with |COPY|<0.1
        need2 = set()   # + always re-measure list
        for pr, s in zip(g.pair, g.sel):
            a, b = pr.split("/")
            if not s:
                need1 |= {a, b}
                need2 |= {a, b}
            elif pr in rem:
                need2 |= {a, b}
        res.append({"cell": cid, "pub": g.pub.iloc[0], "n_metals_in_pairs": len(metals), "n_pairs": len(g),
                    "pairs_uncovered": int((~g.sel).sum()), "metals_need_thr": len(need1), "metals_need_thr_plus_list": len(need2)})
    res = pd.DataFrame(res)
    P(res[["n_metals_in_pairs", "n_pairs", "pairs_uncovered", "metals_need_thr", "metals_need_thr_plus_list"]].describe().round(2).to_string())
    P(f"  pooled fraction of pair-metals still to read: thr only {res.metals_need_thr.sum() / res.n_metals_in_pairs.sum():.3f}; "
      f"thr + always list {res.metals_need_thr_plus_list.sum() / res.n_metals_in_pairs.sum():.3f}; "
      f"fraction of pairs uncovered {res.pairs_uncovered.sum() / res.n_pairs.sum():.3f}")
    P(f"  cells needing ALL their pair-metals read (thr + list): {(res.metals_need_thr_plus_list == res.n_metals_in_pairs).mean():.3f}")

    # (e) literature per pair
    P("\n=== (e) literature (xpub) per pair on covered |y|>=.1 rows ===")
    q = R[R.n_xpub >= 1]
    m = q.big & (q.copy_xpub.abs() >= 0.1)
    g = q[m].assign(okC=lambda z: np.sign(z.copy_xpub) == np.sign(z.y), okH=lambda z: z.y < 0)
    P(g.groupby("pair").agg(n=("okC", "size"), copy=("okC", "mean"), ah=("okH", "mean")).round(3).to_string())
    # literature excluding DGA? report chemotype composition
    P("  chemotypes on covered xpub rows:", g.chem.value_counts().to_dict())

    # (f) single-sibling version of the headline vs always-heavier on the same (target,pair,sib) rows
    S = pd.read_parquet(OUT / "skA_sibrows.parquet")
    S = S[~S.c5].copy()
    S["w"] = 1.0 / S.groupby(["cell_id", "pair"]).sib.transform("size")
    m = (S.y.abs() >= 0.1) & (S.copy1.abs() >= 0.1)
    okc = np.average(np.sign(S.copy1[m]) == np.sign(S.y[m]), weights=S.w[m])
    okh = np.average(S.y[m] < 0, weights=S.w[m])
    ma = S.copy1.abs() >= 0.1
    P(f"\n=== (f) single sibling, covered |y|>=.1: COPY1 {okc:.3f} vs always-heavier {okh:.3f}; all covered: COPY1 "
      f"{np.average(np.sign(S.copy1[ma]) == np.sign(S.y[ma]), weights=S.w[ma]):.3f} vs always-heavier {np.average(S.y[ma] < 0, weights=S.w[ma]):.3f}")
    # outside the dominant cluster
    dom = "g:pub_74428aeab2"
    for lab, mm in (("covered |y|>=.1", m), ("all covered", ma)):
        z = S[mm & (S.cluster != dom)]
        P(f"  outside dominant cluster [{lab}]: links {len(z)} targets {z.cell_id.nunique()} pubs {z.pub.nunique()} COPY1 "
          f"{np.average(np.sign(z.copy1) == np.sign(z.y), weights=z.w):.3f} always-heavier {np.average(z.y < 0, weights=z.w):.3f}")
    (OUT / "skA2_log.txt").write_text("\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
