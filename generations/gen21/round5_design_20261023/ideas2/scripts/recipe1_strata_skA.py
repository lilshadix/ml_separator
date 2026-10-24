#!/usr/bin/env python
"""Skeptic A for recipe-1, part 2 (EXPLORATORY, 229 seen C3 cells): reads-equivalent by own-sibling count, by
condition-change class, without pub_b0b3296ec5, and per-family m* = min{m : D7@m >= SH_own@2}.
Rebuilds the own pool and arms independently (helpers from recipe1_ereccredit_skA.py).  Writes ideas2/recipe1_skA_out/.
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe1_strata_skA.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4.refs import HIST_D7_BETA  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402
from recipe1_ereccredit_skA import pwl, hull_order, X, GD, C5, TRIVIAL, LADDER, OUT  # noqa: E402

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def main() -> int:
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    condc = [c for c in f.columns if c.startswith("cond__")]
    F = f[condc].to_numpy(float)
    v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okc = np.array([c not in v2 for c in ids])
    c3 = sorted(pd.read_parquet(paths.PRED / "REF_BPG.parquet").cell_id.unique())

    def ntd(i, j):
        out = set()
        for q, c in enumerate(condc):
            a, b = F[i, q], F[j, q]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b) or abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
                out.add(c)
        return out - TRIVIAL

    def famname(c):
        if c.startswith("cond__acid__"):
            return "acid_type"
        if c.startswith("cond__diluent__"):
            return "diluent"
        if c.startswith("cond__additive__"):
            return "additive"
        return c.replace("cond__", "")

    rows = []
    for t in c3:
        i = pos[t]
        if pub[i] == C5 or t in v2:
            continue
        Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        m = len(Mi)
        x, yv = X[Mi], Y[i, Mi]
        pa = np.array([q for q in range(m - 1) if Z_VEC[Mi[q + 1]] - Z_VEC[Mi[q]] == 1])
        pb = pa + 1
        ysf = yv[pa] - yv[pb]
        cover = np.all(np.isfinite(Y[:, Mi]), axis=1)
        cover[i] = False
        own = [j for j in np.where((pub == pub[i]) & (ext == ext[i]) & cover & okc)[0] if ntd(i, j)]
        if not own:
            continue
        fams = set()
        for j in own:
            fams |= {famname(c) for c in ntd(i, j)}
        cls = "+".join(sorted(fams))
        u = np.array([HIST_D7_BETA if q == GD else 0.0 for q in Mi])
        T = Y[np.ix_(own, Mi)].mean(0)
        w = len(own) / (len(own) + 1.78)
        h = hull_order(x)
        rec = {"cell_id": t, "pub": pub[i], "fam": pub[i] + "|" + ext[i], "n_own": len(own), "cls": cls,
               "N": len(pa), "sy": ysf.sum(), "syy": (ysf ** 2).sum()}

        def sse(lev):
            return float(((lev[pa] - lev[pb] - ysf) ** 2).sum())
        for n in range(0, 7):
            S = h[:min(n, m)]
            d7 = u + pwl(x, S, yv - u)
            sib = T + pwl(x, S, yv - T)
            rec[f"D7_{n}"] = sse(d7)
            rec[f"SH_{n}"] = sse(w * sib + (1 - w) * d7)
            if n in LADDER:
                lad = [LANTHANIDES.index(s) for s in LADDER[n]]
                SL = [Mi.index(q) for q in lad] if all(q in Mi for q in lad) else S
                rec[f"D7L_{n}"] = sse(u + pwl(x, SL, yv - u))
        rows.append(rec)
    df = pd.DataFrame(rows)
    log("cells", len(df), "pubs", df.pub.nunique(), "families", df.fam.nunique())

    def r2(g, col):
        sst = g.syy.sum() - g.sy.sum() ** 2 / g.N.sum()
        return 1 - g[col].sum() / sst

    def ladder_line(g, name):
        d7 = {n: r2(g, f"D7_{n}") for n in range(0, 7)}
        d7l = {n: r2(g, f"D7L_{n}") for n in LADDER}
        sh = {n: r2(g, f"SH_{n}") for n in range(0, 7)}
        best = {n: max(d7[n], d7l.get(n, -9)) for n in range(0, 7)}
        mstar = {n: next((mm for mm in range(0, 7) if best[mm] >= sh[n]), ">6") for n in (0, 2, 3, 4)}
        log(f"{name:38s} cells {len(g):3d} pubs {g.pub.nunique():2d} | SH " +
            " ".join(f"{sh[n]:.3f}" for n in (0, 2, 3, 4)) + " | D7 HULL " +
            " ".join(f"{d7[n]:.3f}" for n in (2, 3, 4, 5)) + " | D7 LAD " +
            " ".join(f"{d7l[n]:.3f}" for n in (2, 3, 4, 5)) +
            f" | m*(best no-prior >= SH@0/2/3/4) {mstar}")

    log("\n=== by stratum (HULL for SH; m* uses the better of HULL / ladder for D7)")
    ladder_line(df, "OWN1_noC5 all")
    ladder_line(df[df.pub != "pub_b0b3296ec5"], "without pub_b0b3296ec5")
    ladder_line(df[~df.pub.isin(["pub_b0b3296ec5", "pub_19783c039b"])], "without b0b3296ec5 and 19783c039b")
    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 7), (8, 99)):
        ladder_line(df[(df.n_own >= lo) & (df.n_own <= hi)], f"n_own {lo}-{hi}")
    log("\n=== by condition-change class (union over own siblings)")
    for cls, g in df.groupby("cls"):
        if g.N.sum() >= 30:
            ladder_line(g, cls[:38])
    # per-family m*
    log("\n=== per family m* = min m with D7_HULL@m >= SH@2 (family SST); pairs = N")
    fr = []
    for fam, g in df.groupby("fam"):
        sh2 = r2(g, "SH_2")
        d7 = [r2(g, f"D7_{n}") for n in range(0, 7)]
        ms = next((mm for mm in range(0, 7) if d7[mm] >= sh2), 7)
        fr.append({"fam": fam[:60], "cells": len(g), "N": int(g.N.sum()), "n_own_med": float(g.n_own.median()),
                   "SH2": round(sh2, 3), "D7_3": round(d7[3], 3), "D7_4": round(d7[4], 3), "D7_5": round(d7[5], 3),
                   "m_star": ms})
    fr = pd.DataFrame(fr).sort_values("N", ascending=False)
    log(fr.to_string(index=False))
    log("m* distribution (families):", fr.m_star.value_counts().sort_index().to_dict(),
        "| median", fr.m_star.median(), "| pair-weighted median",
        float(np.repeat(fr.m_star.to_numpy(), fr.N.to_numpy()).mean()))
    df.to_parquet(OUT / "skA_strata_cells.parquet", index=False)
    fr.to_csv(OUT / "skA_family_mstar.csv", index=False)
    (OUT / "skA_strata_run.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
