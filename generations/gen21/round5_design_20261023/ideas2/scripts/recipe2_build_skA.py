#!/usr/bin/env python
"""SKEPTIC A for idea recipe-2 (fixed training-free metal ladder).  Independent BUILD step.

EXPLORATORY: all 229 C3 cells were seen in rounds 1-4 and by the recipe theme.  Nothing here is a test.

Independent re-implementation (does not import or read any recipe_* script or recipe_out file):
  * levels(S) = u + PWL_R[y_S - u_S] with u = HIST_D7 beta (-0.072) at Gd only; PWL along minus Shannon CN8 radius,
    constant continuation beyond the extreme reads (np.interp).  Variant LINX: linear extrapolation with the
    slope of the outermost read segment (to see how much "spread first" is an artefact of constant continuation).
  * for every C3 cell and EVERY subset S (|S| = 2..5) of its measured metals: SSE over all strict-adjacent pairs.
  * HULL order re-coded here; DOPT_Q-nested sets from the gen21 fold cache (BPG, 5 dev seeds).
  * own-prior SH arm re-coded (CLEAN rule: same pub + extractant, covering, outside round-3 V2 lists, condition differs
    in a family other than contact time / metal concentration; K = 1.78).
  * reproduction gate: D7 on round-4 DOPT_Q 4-read sets must equal the stored HIST_D7 column.

Writes only ideas2/recipe2_skA_out/build_*.  Run from D:\\ml_separator_gh:
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe2_build_skA.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4.metrics import add_clusters  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp as R3I  # noqa: E402  (only to verify the coordinate)
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe2_skA_out")
LN = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
ZZ = np.array([57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71])
R8 = {"La": 1.160, "Ce": 1.143, "Pr": 1.126, "Nd": 1.109, "Sm": 1.079, "Eu": 1.066, "Gd": 1.053,
      "Tb": 1.040, "Dy": 1.027, "Ho": 1.015, "Er": 1.004, "Tm": 0.994, "Yb": 0.985, "Lu": 0.977}
XR = -np.array([R8[m] for m in LN])
GD = 6
BETA = -0.072
KSH = 1.78
C5 = "pub_6a0ef8528c"
NS = (2, 3, 4, 5)
LADDER = {2: ("La", "Ho"), 3: ("La", "Nd", "Ho"), 4: ("La", "Nd", "Er", "Lu"), 5: ("La", "Nd", "Gd", "Er", "Lu")}


def lev(x, yv, base, S, linx=False):
    """levels at all metals of the cell from reads S (indices into x); base = additive structural term."""
    xs, rs = x[list(S)], (yv - base)[list(S)]
    v = np.interp(x, xs, rs)
    if linx and len(S) >= 2:
        lo, hi = x < xs[0], x > xs[-1]
        v[lo] = rs[0] + (x[lo] - xs[0]) * (rs[1] - rs[0]) / (xs[1] - xs[0])
        v[hi] = rs[-1] + (x[hi] - xs[-1]) * (rs[-1] - rs[-2]) / (xs[-1] - xs[-2])
    out = v + base
    out[list(S)] = yv[list(S)]
    return out


def fam(c: str) -> str:
    if c.startswith("cond__acid__"):
        return "acid_type"
    if c == "cond__acid_concentration_M":
        return "acid_conc"
    if c.startswith("cond__additive__"):
        return "additive"
    if c.startswith("cond__diluent__"):
        return "diluent"
    return {"cond__contact_time_min": "contact_time", "cond__extractant_concentration_M": "extr_conc",
            "cond__metal_concentration_mM": "metal_conc", "cond__temperature_C": "temperature"}.get(c, c)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    assert np.allclose(XR, R3I.X_R), "coordinate mismatch"
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    assert Y.shape[1] == 14
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = sorted(par.cell_id.unique())
    clus = add_clusters(par[["cell_id", "chemotype", "publication"]].drop_duplicates(), f)
    clof = dict(zip(clus.cell_id, clus.cluster_cpg))
    v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    condc = [c for c in f.columns if c.startswith("cond__")]
    F = f[condc].to_numpy(float)
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()

    def dfams(i, j):
        s = set()
        for q, c in enumerate(condc):
            a, b = F[i, q], F[j, q]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b) or not np.isclose(a, b):
                s.add(fam(c))
        return s - {"contact_time", "metal_conc"}

    # ---- DOPT_Q nested sets from the fold cache
    dq = {}
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            for cell in rec["outer"]["cells"]:
                seq = cell["seqs"]["DOPT_Q"]
                d = {}
                for k in range(1, len(seq) + 1):
                    rd = tuple(sorted({m for p in seq[:k] for m in p}))
                    d.setdefault(len(rd), rd)
                dq[(int(seed), cell["cell_id"])] = d
    seeds = [int(s) for s in paths.DEV_SEEDS]

    gidx = {n: {cmb: q for q, cmb in enumerate(itertools.combinations(range(14), n))} for n in NS}
    store = {n: {k: np.full((len(c3), len(gidx[n])), np.nan) for k in ("D7", "D7LINX", "SHown")} for n in NS}
    cells, extra = [], []
    for ci, t in enumerate(c3):
        i = pos[t]
        Mi = [m for m in range(14) if np.isfinite(Y[i, m])]
        m = len(Mi)
        x, yv = XR[Mi], Y[i, Mi]
        assert np.all(np.diff(x) > 0)
        pa = [p for p in range(m - 1) if ZZ[Mi[p + 1]] - ZZ[Mi[p]] == 1]
        pa = np.array(pa)
        pb = pa + 1
        ysf = yv[pa] - yv[pb]
        u = np.zeros(m)
        if GD in Mi:
            u[Mi.index(GD)] = BETA
        # own pool (CLEAN)
        cover = np.all(np.isfinite(Y[:, Mi]), axis=1)
        cover[i] = False
        own = []
        if t not in v2:
            for j in np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]:
                if ids[j] not in v2 and dfams(i, j):
                    own.append(int(j))
        T = Y[np.ix_(own, Mi)].mean(0) if own else None
        w = len(own) / (len(own) + KSH)
        # HULL order
        order = [0, m - 1]
        while len(order) < m:
            rest = [q for q in range(m) if q not in order]
            dm = [min(abs(x[q] - x[o]) for o in order) for q in rest]
            order.append(rest[int(np.argmax(dm))])
        rec = {"ci": ci, "cell_id": t, "pub": pub[i], "cluster": clof[t], "chemotype": f.chemotype.iloc[i],
               "family": f.chem_family.iloc[i], "c5": pub[i] == C5, "v2": t in v2, "n_metals": m,
               "metals": ",".join(LN[q] for q in Mi), "N": len(pa), "sy": float(ysf.sum()),
               "syy": float((ysf ** 2).sum()), "n_own": len(own)}
        if T is not None:
            sf0 = T[pa] - T[pb]
            rec["sse_COPY"] = float(((sf0 - ysf) ** 2).sum())
        for n in NS:
            if m < n:
                continue
            for S in itertools.combinations(range(m), n):
                g = gidx[n][tuple(Mi[q] for q in S)]
                L = lev(x, yv, u, S)
                e = (L[pa] - L[pb]) - ysf
                store[n]["D7"][ci, g] = float((e ** 2).sum())
                L2 = lev(x, yv, u, S, linx=True)
                e2 = (L2[pa] - L2[pb]) - ysf
                store[n]["D7LINX"][ci, g] = float((e2 ** 2).sum())
                if T is not None:
                    Ls = lev(x, yv, T, S)
                    sh = w * (Ls[pa] - Ls[pb]) + (1 - w) * (L[pa] - L[pb])
                    store[n]["SHown"][ci, g] = float(((sh - ysf) ** 2).sum())
            hs = tuple(sorted(order[:n]))
            rec[f"hull{n}"] = gidx[n][tuple(Mi[q] for q in hs)]
            lad = LADDER[n]
            rec[f"lad{n}"] = gidx[n][tuple(sorted(LN.index(z) for z in lad))] if all(LN.index(z) in Mi for z in lad) else -1
            for s in seeds:
                d = dq.get((s, t), {})
                rec[f"dq{n}_{s}"] = gidx[n][d[n]] if n in d else -1
        cells.append(rec)
        if ci % 40 == 0:
            print(ci, f"{time.time() - t0:.0f}s", flush=True)
    cdf = pd.DataFrame(cells)
    # ---- reproduction gate: D7 at round-4 DOPT_Q 4-read sets vs stored HIST_D7
    rep = []
    ciof = dict(zip(cdf.cell_id, cdf.ci))
    for seed, g in par.groupby("split_seed"):
        rs = g.groupby("cell_id").read_set.first()
        sse = sum(store[4]["D7"][ciof[c], gidx[4][tuple(sorted(LN.index(z) for z in r.split(",")))]]
                  for c, r in rs.items())
        y = g.y.to_numpy()
        sst = ((y - y.mean()) ** 2).sum()
        rep.append({"seed": int(seed), "R2_skA": 1 - sse / sst, "R2_HIST_D7": 1 - ((g.HIST_D7 - g.y) ** 2).sum() / sst,
                    "rows": len(g), "sse_diff": float(sse - ((g.HIST_D7 - g.y) ** 2).sum())})
        # DOPT_Q 4-read set in cache == read_set in parquet?
        mism = sum(1 for c, r in rs.items()
                   if cdf.loc[ciof[c], f"dq4_{int(seed)}"] != gidx[4][tuple(sorted(LN.index(z) for z in r.split(",")))])
        rep[-1]["dq4_set_mismatch_cells"] = int(mism)
    print("REPRO", json.dumps(rep, indent=0))
    cdf.to_parquet(OUT / "build_cells.parquet", index=False)
    for n in NS:
        np.savez_compressed(OUT / f"build_k{n}.npz", **store[n])
    (OUT / "build_log.json").write_text(json.dumps({"repro": rep, "seconds": time.time() - t0, "cells": len(c3),
                                                    "seeds": seeds}, indent=1))
    print("done", f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
