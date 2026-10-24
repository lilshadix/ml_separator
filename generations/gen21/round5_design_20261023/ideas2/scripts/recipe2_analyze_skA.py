#!/usr/bin/env python
"""SKEPTIC A for idea recipe-2.  Independent ANALYSIS step (reads only ideas2/recipe2_skA_out/build_* written by
recipe2_build_skA.py).  EXPLORATORY: every number is on the 229 already-seen C3 cells.

Checks:
  T1  headline re-computation (noC5): frozen ladder / DOPT_Q-nested (5 seeds) / HULL / random (exact expectation,
      all subsets and legal-only subsets) at 2..5 reads, with paired publication- and cluster-bootstrap intervals,
      LOPO / LOCO ranges, giant cluster vs rest.
  T2  strata: DGA vs non-DGA, series length, V2-flag removal, with C5.
  T3  honest out-of-sample ladder: set chosen leave-one-publication-out and leave-one-cluster-out (legal sets,
      objective = total SSE with HULL fallback), applied with HULL fallback.
  T4  nestedness: the frozen ladder is not nested (Ho dropped at 4 reads); nested alternatives; sequential cost.
  T5  constant- vs linear-extrapolation (is "spread first" / La,Ho a continuation artefact?).
  T6  own-prior arm (SH_own, re-coded): do spread rules tie within 0.01?
  T7  duplicates / near-copies, DOPT_Q first pair, concentration of the gain.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe2_analyze_skA.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401  (sets up gen20k on sys.path)

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe2_skA_out")
LN = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
ZZ = np.array([57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71])
NS = (2, 3, 4, 5)
B = 2000
COMBOS = {n: list(itertools.combinations(range(14), n)) for n in NS}
GID = {n: {c: q for q, c in enumerate(COMBOS[n])} for n in NS}
LEGAL = {n: np.array([not any(ZZ[b] - ZZ[a] == 1 for a, b in itertools.combinations(c, 2)) for c in COMBOS[n]])
         for n in NS}
RES: dict = {}


def sid(n, names):
    return GID[n][tuple(sorted(LN.index(z) for z in names))]


def sname(n, q):
    return ",".join(LN[i] for i in COMBOS[n][q])


class Pop:
    def __init__(self, c: pd.DataFrame, unit: str = "pub", seed: int = 7):
        self.c = c.reset_index(drop=True)
        self.units = sorted(self.c[unit].unique())
        ui = self.c[unit].map({u: k for k, u in enumerate(self.units)}).to_numpy()
        self.G = np.zeros((len(self.c), len(self.units)))
        self.G[np.arange(len(self.c)), ui] = 1
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(len(self.units), np.full(len(self.units), 1 / len(self.units)), size=B).astype(float)
        self.N, self.sy, self.syy = (self.c[k].to_numpy(float) @ self.G for k in ("N", "sy", "syy"))

    def r2(self, sse, w=None):
        s = sse @ self.G
        if w is None:
            return 1 - s.sum() / (self.syy.sum() - self.sy.sum() ** 2 / self.N.sum())
        return 1 - (w @ s) / (w @ self.syy - (w @ self.sy) ** 2 / (w @ self.N))

    def diff(self, a, b):
        d = self.r2(a) - self.r2(b)
        bs = self.r2(a, self.W) - self.r2(b, self.W)
        loo = []
        for k in range(len(self.units)):
            w = np.ones((1, len(self.units)))
            w[0, k] = 0
            loo.append(float(self.r2(a, w)[0] - self.r2(b, w)[0]))
        return {"d": float(d), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                "P_gt0": float((bs > 0).mean()), "loo_min": min(loo), "loo_max": max(loo), "units": len(self.units)}


def main() -> int:
    cells = pd.read_parquet(OUT / "build_cells.parquet")
    npz = {n: dict(np.load(OUT / f"build_k{n}.npz")) for n in NS}
    seeds = json.loads((OUT / "build_log.json").read_text())["seeds"]
    nc = len(cells)
    ar = np.arange(nc)
    cells["dga"] = cells.family == "diglycolamide"
    cells["giant"] = cells.cluster == cells.cluster.value_counts().index[0]

    def pol(arm, n, p, cdf=cells):
        a = npz[n][arm]
        hull = a[ar, cdf[f"hull{n}"].to_numpy()]
        if p == "HULL":
            return hull
        if p == "LAD":
            q = cdf[f"lad{n}"].to_numpy()
            return np.where(q >= 0, a[ar, np.maximum(q, 0)], hull)
        if p == "DQ":
            out = []
            for s in seeds:
                q = cdf[f"dq{n}_{s}"].to_numpy()
                out.append(np.where(q >= 0, a[ar, np.maximum(q, 0)], hull))
            return np.mean(out, 0)
        if p.startswith("DQs"):
            s = seeds[int(p[3:])]
            q = cdf[f"dq{n}_{s}"].to_numpy()
            return np.where(q >= 0, a[ar, np.maximum(q, 0)], hull)
        if p == "RND":
            return np.nanmean(a, 1)
        if p == "RNDL":
            return np.nanmean(np.where(LEGAL[n][None, :], a, np.nan), 1)
        if p.startswith("SET:"):
            q = sid(n, p[4:].split(","))
            return np.where(np.isnan(a[:, q]), hull, a[:, q])
        raise ValueError(p)

    pops = {"noC5": ~cells.c5, "ALL229": np.ones(nc, bool), "noC5_noV2": ~cells.c5 & ~cells.v2,
            "noC5_DGA": ~cells.c5 & cells.dga, "noC5_nonDGA": ~cells.c5 & ~cells.dga,
            "noC5_len<=8": ~cells.c5 & (cells.n_metals <= 8), "noC5_len9-12": ~cells.c5 & cells.n_metals.between(9, 12),
            "noC5_len>=13": ~cells.c5 & (cells.n_metals >= 13), "noC5_giant": ~cells.c5 & cells.giant,
            "noC5_notgiant": ~cells.c5 & ~cells.giant}
    # ---------------- T1/T2: levels and paired contrasts ----------------
    rows = []
    for pk, msk in pops.items():
        msk = np.asarray(msk)
        sub = cells[msk]
        Pp, Pc = Pop(sub, "pub"), Pop(sub, "cluster")
        for n in NS:
            v = {p: pol("D7", n, p)[msk] for p in ("LAD", "DQ", "HULL", "RND", "RNDL")}
            dqs = [Pp.r2(pol("D7", n, f"DQs{k}")[msk]) for k in range(len(seeds))]
            lad_cov = float((sub[f"lad{n}"] >= 0).mean())
            r = {"pop": pk, "n": n, "cells": len(sub), "pubs": len(Pp.units), "clusters": len(Pc.units),
                 "lad_coverage": lad_cov, **{f"R2_{p}": float(Pp.r2(x)) for p, x in v.items()},
                 "DQ_seed_min": float(min(dqs)), "DQ_seed_max": float(max(dqs))}
            for nm, a, b in (("LAD-DQ", "LAD", "DQ"), ("LAD-HULL", "LAD", "HULL"), ("HULL-RND", "HULL", "RND"),
                             ("HULL-RNDL", "HULL", "RNDL"), ("DQ-HULL", "DQ", "HULL")):
                dp, dc = Pp.diff(v[a], v[b]), Pc.diff(v[a], v[b])
                r.update({f"{nm}": dp["d"], f"{nm}_pubCI": f"[{dp['lo']:+.3f},{dp['hi']:+.3f}]",
                          f"{nm}_cluCI": f"[{dc['lo']:+.3f},{dc['hi']:+.3f}]",
                          f"{nm}_LOPO": f"[{dp['loo_min']:+.3f},{dp['loo_max']:+.3f}]",
                          f"{nm}_LOCO": f"[{dc['loo_min']:+.3f},{dc['loo_max']:+.3f}]"})
            rows.append(r)
    T1 = pd.DataFrame(rows)
    T1.to_csv(OUT / "T1_levels_contrasts.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 80, "display.max_rows", 200):
        print("=== T1 levels (pub-pooled R2)")
        print(T1[["pop", "n", "cells", "pubs", "clusters", "lad_coverage", "R2_LAD", "R2_DQ", "DQ_seed_min",
                  "DQ_seed_max", "R2_HULL", "R2_RND", "R2_RNDL"]].round(3).to_string(index=False))
        for nm in ("LAD-DQ", "LAD-HULL", "HULL-RND", "HULL-RNDL"):
            print(f"\n=== contrast {nm}")
            print(T1[["pop", "n", nm, f"{nm}_pubCI", f"{nm}_cluCI", f"{nm}_LOPO", f"{nm}_LOCO"]]
                  .round(3).to_string(index=False))

    # ---------------- coverage-restricted (cells that HAVE the ladder metals) ----------------
    rows = []
    for n in NS:
        msk = (~cells.c5 & (cells[f"lad{n}"] >= 0)).to_numpy()
        sub = cells[msk]
        Pp, Pc = Pop(sub, "pub"), Pop(sub, "cluster")
        v = {p: pol("D7", n, p)[msk] for p in ("LAD", "DQ", "HULL")}
        dlh, dld = Pp.diff(v["LAD"], v["HULL"]), Pp.diff(v["LAD"], v["DQ"])
        dlhc = Pc.diff(v["LAD"], v["HULL"])
        rows.append({"n": n, "cells": len(sub), "pubs": len(Pp.units), "clusters": len(Pc.units),
                     "giant_share_cells": float(sub.giant.mean()), "dga_share": float(sub.dga.mean()),
                     "R2_LAD": float(Pp.r2(v["LAD"])), "R2_DQ": float(Pp.r2(v["DQ"])), "R2_HULL": float(Pp.r2(v["HULL"])),
                     "LAD-HULL": dlh["d"], "pubCI": f"[{dlh['lo']:+.3f},{dlh['hi']:+.3f}]",
                     "cluCI": f"[{dlhc['lo']:+.3f},{dlhc['hi']:+.3f}]", "LOCO": f"[{dlhc['loo_min']:+.3f},{dlhc['loo_max']:+.3f}]",
                     "LAD-DQ": dld["d"], "LAD-DQ_pubCI": f"[{dld['lo']:+.3f},{dld['hi']:+.3f}]"})
    TC = pd.DataFrame(rows)
    print("\n=== covered cells only (noC5)")
    print(TC.round(3).to_string(index=False))
    TC.to_csv(OUT / "T1b_covered.csv", index=False)

    # ---------------- concentration: per-publication share of the LAD-HULL / LAD-DQ SSE gain ----------------
    msk = (~cells.c5).to_numpy()
    sub = cells[msk]
    conc = []
    for n in NS:
        for nm, a, b in (("LAD-HULL", "LAD", "HULL"), ("LAD-DQ", "LAD", "DQ")):
            gain = (pol("D7", n, b) - pol("D7", n, a))[msk]          # positive = ladder better
            gp = pd.Series(gain).groupby(sub.pub.to_numpy()).sum().sort_values(ascending=False)
            gc = pd.Series(gain).groupby(sub.cluster.to_numpy()).sum().sort_values(ascending=False)
            tot = gain.sum()
            conc.append({"n": n, "contrast": nm, "total_sse_gain": float(tot),
                         "top1_pub_share": float(gp.iloc[0] / tot), "top3_pub_share": float(gp.iloc[:3].sum() / tot),
                         "top_pubs": gp.head(3).round(3).to_dict(), "n_pubs_pos": int((gp > 1e-12).sum()),
                         "n_pubs_neg": int((gp < -1e-12).sum()), "giant_cluster_share": float(gc.get(sub.cluster.value_counts().index[0], 0) / tot),
                         "cluster_gains": gc.round(3).to_dict()})
    TCON = pd.DataFrame(conc)
    print("\n=== concentration of ladder gains (noC5; SSE gain, positive = ladder better)")
    with pd.option_context("display.width", 300, "display.max_colwidth", 200):
        print(TCON.to_string(index=False))
    TCON.to_csv(OUT / "T1c_concentration.csv", index=False)

    # ---------------- T3: honest out-of-sample static ladder ----------------
    def choose(n, tr_mask, legal=True, rule="fallback", cov=0.5):
        a = npz[n]["D7"][tr_mask]
        hull = pol("D7", n, "HULL")[tr_mask]
        avail = ~np.isnan(a)
        if rule == "fallback":
            tot = np.where(avail, a, hull[:, None]).sum(0)
        else:                                            # covered-cell pooled ratio, coverage >= cov
            c = cells[tr_mask]
            N, sy, syy = (c[k].to_numpy(float) for k in ("N", "sy", "syy"))
            ybar = sy.sum() / N.sum()
            sst = syy - 2 * ybar * sy + N * ybar ** 2
            num = np.nan_to_num(a).sum(0)
            den = (avail * sst[:, None]).sum(0)
            tot = np.where((avail.mean(0) >= cov) & (den > 0), num / np.where(den > 0, den, 1), np.inf)
        if legal:
            tot = np.where(LEGAL[n], tot, np.inf)
        return int(np.argmin(tot))

    rows, picks = [], {}
    base = (~cells.c5).to_numpy()
    for unit in ("pub", "cluster"):
        for rule in ("fallback", "covratio"):
            for legal in (True, False):
                for n in NS:
                    sse = np.zeros(nc)
                    chosen = []
                    for uval in cells.loc[base, unit].unique():
                        te = base & (cells[unit] == uval).to_numpy()
                        tr = base & ~te
                        q = choose(n, tr, legal, rule)
                        chosen.append(sname(n, q))
                        a = npz[n]["D7"][:, q]
                        h = pol("D7", n, "HULL")
                        sse[te] = np.where(np.isnan(a[te]), h[te], a[te])
                    sub = cells[base]
                    Pp = Pop(sub, "pub")
                    Pc = Pop(sub, "cluster")
                    s = sse[base]
                    lad, dq, hull = (pol("D7", n, p)[base] for p in ("LAD", "DQ", "HULL"))
                    d1, d2, d3 = Pp.diff(s, hull), Pp.diff(s, dq), Pp.diff(lad, s)
                    d1c = Pc.diff(s, hull)
                    vc = pd.Series(chosen).value_counts()
                    rows.append({"holdout": unit, "rule": rule, "legal": legal, "n": n, "R2_OOS": float(Pp.r2(s)),
                                 "R2_frozen": float(Pp.r2(lad)), "R2_DQ": float(Pp.r2(dq)), "R2_HULL": float(Pp.r2(hull)),
                                 "OOS-HULL": d1["d"], "OOS-HULL_pubCI": f"[{d1['lo']:+.3f},{d1['hi']:+.3f}]",
                                 "OOS-HULL_cluCI": f"[{d1c['lo']:+.3f},{d1c['hi']:+.3f}]",
                                 "OOS-DQ": d2["d"], "OOS-DQ_pubCI": f"[{d2['lo']:+.3f},{d2['hi']:+.3f}]",
                                 "frozen-OOS": d3["d"], "picks": vc.head(4).to_dict(), "n_folds": len(chosen)})
    T3 = pd.DataFrame(rows)
    T3.to_csv(OUT / "T3_oos_ladder.csv", index=False)
    with pd.option_context("display.width", 320, "display.max_colwidth", 120, "display.max_rows", 100):
        print("\n=== T3 out-of-sample static set (noC5), applied with HULL fallback")
        print(T3.round(3).to_string(index=False))

    # in-sample best legal sets with fallback objective (what an in-sample ladder would be)
    ins = {}
    for n in NS:
        a = npz[n]["D7"][base]
        h = pol("D7", n, "HULL")[base]
        tot = np.where(np.isnan(a), h[:, None], a).sum(0)
        tot = np.where(LEGAL[n], tot, np.inf)
        r = np.argsort(tot)[:5]
        P = Pop(cells[base], "pub")
        ins[n] = [(sname(n, q), round(float(P.r2(np.where(np.isnan(a[:, q]), h, a[:, q]))), 4),
                   round(float((~np.isnan(a[:, q])).mean()), 2)) for q in r]
        print("in-sample top legal sets (fallback objective)", n, ins[n])
    RES["insample_top_legal"] = ins

    # ---------------- T4: nestedness ----------------
    chains = {"FROZEN(non-nested)": [("La", "Ho"), ("La", "Nd", "Ho"), ("La", "Nd", "Er", "Lu"), ("La", "Nd", "Gd", "Er", "Lu")],
              "N1 La,Ho>+Nd>+Lu>+Gd": [("La", "Ho"), ("La", "Nd", "Ho"), ("La", "Nd", "Ho", "Lu"), ("La", "Nd", "Gd", "Ho", "Lu")],
              "N2 La,Ho>+Nd>+Gd>+Lu": [("La", "Ho"), ("La", "Nd", "Ho"), ("La", "Nd", "Gd", "Ho"), ("La", "Nd", "Gd", "Ho", "Lu")],
              "N3 La,Er>+Nd>+Lu>+Gd": [("La", "Er"), ("La", "Nd", "Er"), ("La", "Nd", "Er", "Lu"), ("La", "Nd", "Gd", "Er", "Lu")],
              "N4 La,Lu>+Nd>+Er>+Gd": [("La", "Lu"), ("La", "Nd", "Lu"), ("La", "Nd", "Er", "Lu"), ("La", "Nd", "Gd", "Er", "Lu")]}
    rows = []
    for pk in ("noC5", "ALL229", "noC5_DGA", "noC5_nonDGA"):
        msk = np.asarray(pops[pk])
        P = Pop(cells[msk], "pub")
        for nm, ch in chains.items():
            r = {"pop": pk, "chain": nm}
            for n, s in zip(NS, ch):
                r[f"n{n}"] = float(P.r2(pol("D7", n, "SET:" + ",".join(s))[msk]))
            for n in NS:
                r[f"HULL{n}"] = float(P.r2(pol("D7", n, "HULL")[msk]))
                r[f"DQ{n}"] = float(P.r2(pol("D7", n, "DQ")[msk]))
            rows.append(r)
    T4 = pd.DataFrame(rows)
    T4.to_csv(OUT / "T4_nested.csv", index=False)
    with pd.option_context("display.width", 300):
        print("\n=== T4 nested chains (SET with HULL fallback)")
        print(T4[["pop", "chain", "n2", "n3", "n4", "n5"]].round(3).to_string(index=False))
    # sequential cost of the frozen ladder: after La,Ho -> La,Nd,Ho a chemist who wants the 4-set holds La,Nd,Ho,Er,Lu
    seq = {}
    msk = np.asarray(pops["noC5"])
    P = Pop(cells[msk], "pub")
    for nm, s in (("seq_after_4set(La,Nd,Ho,Er,Lu)=5 metals", ("La", "Nd", "Ho", "Er", "Lu")),
                  ("seq_after_5set(La,Nd,Gd,Ho,Er,Lu)=6 metals", None)):
        if s is not None:
            seq[nm] = float(P.r2(pol("D7", 5, "SET:" + ",".join(s))[msk]))
    seq["LAD5"] = float(P.r2(pol("D7", 5, "LAD")[msk]))
    RES["sequential"] = seq
    print("sequential cost:", seq)

    # ---------------- T5: linear extrapolation variant ----------------
    rows = []
    for pk in ("noC5", "noC5_DGA", "noC5_nonDGA"):
        msk = np.asarray(pops[pk])
        P = Pop(cells[msk], "pub")
        for n in NS:
            r = {"pop": pk, "n": n}
            for arm in ("D7", "D7LINX"):
                for p in ("LAD", "DQ", "HULL", "RND", "RNDL", "SET:La,Lu" if n == 2 else "HULL"):
                    r[f"{arm}_{p}"] = float(P.r2(pol(arm, n, p)[msk]))
            d = P.diff(pol("D7LINX", n, "HULL")[msk], pol("D7LINX", n, "RND")[msk])
            r["LINX HULL-RND"] = f"{d['d']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]"
            d = P.diff(pol("D7LINX", n, "LAD")[msk], pol("D7LINX", n, "HULL")[msk])
            r["LINX LAD-HULL"] = f"{d['d']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]"
            rows.append(r)
    T5 = pd.DataFrame(rows)
    T5.to_csv(OUT / "T5_extrapolation.csv", index=False)
    with pd.option_context("display.width", 320, "display.max_columns", 40):
        print("\n=== T5 constant (D7) vs linear (D7LINX) continuation")
        print(T5.round(3).to_string(index=False))

    # ---------------- T6: own-prior arm ----------------
    rows = []
    for pk, msk in (("OWN1_noC5", (cells.n_own >= 1) & ~cells.c5), ("OWN1_withC5", cells.n_own >= 1),
                    ("OWN2_noC5", (cells.n_own >= 2) & ~cells.c5)):
        msk = np.asarray(msk)
        sub = cells[msk]
        Pp, Pc = Pop(sub, "pub"), Pop(sub, "cluster")
        cp = sub.sse_COPY.to_numpy()
        for n in NS:
            v = {p: pol("SHown", n, p)[msk] for p in ("LAD", "DQ", "HULL", "RND", "RNDL")}
            r = {"pop": pk, "n": n, "cells": len(sub), "pubs": len(Pp.units), "clusters": len(Pc.units),
                 "COPY0": float(Pp.r2(cp)), **{f"R2_{p}": float(Pp.r2(x)) for p, x in v.items()},
                 "lad_cov": float((sub[f"lad{n}"] >= 0).mean())}
            spread = [r["R2_LAD"], r["R2_DQ"], r["R2_HULL"]]
            r["max_spread_rule_gap"] = max(spread) - min(spread)
            for nm, a, b in (("DQ-LAD", "DQ", "LAD"), ("HULL-LAD", "HULL", "LAD"), ("DQ-HULL", "DQ", "HULL"),
                             ("HULL-RND", "HULL", "RND")):
                d = Pp.diff(v[a], v[b])
                dc = Pc.diff(v[a], v[b])
                r[nm] = f"{d['d']:+.3f} pub[{d['lo']:+.3f},{d['hi']:+.3f}] clu[{dc['lo']:+.3f},{dc['hi']:+.3f}]"
            rows.append(r)
    T6 = pd.DataFrame(rows)
    T6.to_csv(OUT / "T6_own_prior.csv", index=False)
    with pd.option_context("display.width", 330, "display.max_columns", 40, "display.max_colwidth", 60):
        print("\n=== T6 own prior (SH_own re-coded)")
        print(T6.round(3).to_string(index=False))

    # ---------------- T7: duplicates, DOPT_Q first pair ----------------
    from gen20k import data as D  # noqa: E402
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    pos = {c: i for i, c in enumerate(f.cell_id.astype(str))}
    ii = [pos[c] for c in cells.cell_id]
    dup_exact, dup_near = [], []
    for a, b in itertools.combinations(range(nc), 2):
        ya, yb = Y[ii[a]], Y[ii[b]]
        com = np.isfinite(ya) & np.isfinite(yb)
        if com.sum() < 5:
            continue
        diff = np.abs(ya[com] - yb[com])
        sfd = np.abs(np.diff(ya[com]) - np.diff(yb[com]))
        if diff.max() < 1e-9:
            dup_exact.append((cells.cell_id[a], cells.cell_id[b], cells.pub[a], cells.pub[b]))
        elif sfd.max() < 0.02:
            dup_near.append((cells.cell_id[a], cells.cell_id[b], cells.pub[a], cells.pub[b], float(sfd.max())))
    RES["dup_exact_levels"] = dup_exact
    RES["dup_near_shape(maxdiff<0.02)"] = dup_near
    print("\n=== T7 exact-duplicate level vectors among C3:", len(dup_exact),
          "cross-pub:", sum(1 for d in dup_exact if d[2] != d[3]))
    print("near-identical shapes (all successive diffs within 0.02):", len(dup_near),
          "cross-pub:", sum(1 for d in dup_near if d[2] != d[3]))
    for d in dup_exact[:10] + dup_near[:10]:
        print("  ", d)
    # sensitivity: drop the second member of every exact/near duplicate pair
    drop = {d[1] for d in dup_exact + dup_near}
    msk = (~cells.c5 & ~cells.cell_id.isin(drop)).to_numpy()
    P = Pop(cells[msk], "pub")
    dd = {}
    for n in NS:
        v = {p: pol("D7", n, p)[msk] for p in ("LAD", "DQ", "HULL", "RND")}
        dd[n] = {"cells": int(msk.sum()), **{p: round(float(P.r2(x)), 3) for p, x in v.items()},
                 "LAD-DQ": round(P.diff(v["LAD"], v["DQ"])["d"], 3), "LAD-HULL": round(P.diff(v["LAD"], v["HULL"])["d"], 3)}
    RES["dedup_noC5"] = dd
    print("dedup noC5:", dd)
    # DOPT_Q second metal at 2 reads, per seed
    fp = {}
    for s in seeds:
        q = cells[f"dq2_{s}"]
        fp[s] = pd.Series([sname(2, int(x)) if x >= 0 else "none" for x in q[~cells.c5]]).value_counts().head(5).to_dict()
    RES["DQ_first_pair_noC5"] = {str(k): v for k, v in fp.items()}
    print("DOPT_Q 2-read set (noC5) by seed:", fp)
    hp = pd.Series([sname(2, int(x)) for x in cells.loc[~cells.c5, "hull2"]]).value_counts().head(5).to_dict()
    RES["HULL_first_pair_noC5"] = hp
    print("HULL 2-read set (noC5):", hp)
    (OUT / "analyze_summary.json").write_text(json.dumps(RES, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
