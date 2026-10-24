#!/usr/bin/env python
"""Theme "minimal-measurement recipes", build step.  EXPLORATORY: every number is computed on the 229 already-seen
C3 cells (V1 bench).  Training-free: no fold, no fitted parameter (the Gd amplitude is the fixed historical
HIST_D7 beta = -0.072, chosen in gen13 before gen21; round 4 showed R2 is flat between -0.072 and -0.094).

Question: given what a chemist already has, how well is the WHOLE strict-adjacent log SF series of a new
(extractant, condition) cell known after measuring a set S of metals?  For every C3 cell and EVERY subset S of its
measured metals with |S| = 1..6, every arm predicts all strict-adjacent pairs of the cell; a pair with both ends in S
is "measured" (prediction = reading, error 0: noise-free credit).

Prior-information scenarios (pools built from metadata + metal coverage only; no label is looked at):
  P0  nothing:              D7 = PWL_R[y_S - u_S] + u, u = beta at Gd (B0 = same with u = 0)
  P1  own series, same extractant, same publication, other condition (CLEAN rule of probe P1: target and sibling not
      in the round-3 V2-1/2/3 lists, sibling's cond__ differs in a field other than contact time / metal conc)
  P2  other extractant(s) of the same publication (same campaign / lab proxy)
  P3  same extractant, other publications (literature series of the same molecule)
  P4  same chemotype, other publications (literature series of related molecules)
  Template T = mean of the pool's covering series.  SIB_S = T + PWL_R[y_S - T_S];  SH_S = w SIB_S + (1-w) D7_S with
  w = n/(n + 1.78) (K from probe P1, the own-lab roughness correlation; applied to every pool, which is optimistic
  for P2-P4).  At |S| = 0 the arms are COPY = T (SIB) and w T + (1-w) u (SH); D7 at |S| <= 1 predicts only the Gd term.

Writes only under ideas2/recipe_out/: recipe_cells.parquet (per-cell metadata + all-pair y stats),
recipe_k{n}.npz (per cell x global subset: SSE per arm, |err|>0.1 counts, direction counts),
recipe_pairs.parquet (per cell x pair x n_read: random-expectation and HULL-policy squared errors per arm, COPY values).

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe_build.py
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
from r4.refs import HIST_D7_BETA  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\recipe_out")
K_SH = 1.78
GD = CO.GD
C5 = "pub_6a0ef8528c"
NMAX = 6
FAMS_TRIVIAL = {"contact_time", "metal_conc"}
POOLS = ("own", "xext", "xpub", "chem")
ARMS = ("B0", "D7") + tuple(f"{a}_{p}" for p in POOLS for a in ("SIB", "SH"))
X = interp.X_R


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
            "cond__metal_concentration_mM": "metal_conc", "cond__temperature_C": "temperature"}[c]


def interp_batch(R: np.ndarray, r: np.ndarray, x: np.ndarray) -> np.ndarray:
    """PWL along x with constant continuation, for many read masks at once.  R (n_sub, m) bool, r (m,) values
    (only read entries used), x (m,) increasing.  Returns (n_sub, m)."""
    n_sub, m = R.shape
    ar = np.arange(m)
    prev = np.maximum.accumulate(np.where(R, ar, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(R, ar, m)[:, ::-1], axis=1)[:, ::-1]
    hp, hn = prev >= 0, nxt < m
    pc, nc = prev.clip(0, m - 1), nxt.clip(0, m - 1)
    rp, rn, xp, xn = r[pc], r[nc], x[pc], x[nc]
    both = hp & hn & (nc != pc)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(both, (x[None, :] - xp) / np.where(both, xn - xp, 1.0), 0.0)
    return np.where(both, rp + (rn - rp) * t, np.where(hp, rp, rn))


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    cond_cols = [c for c in f.columns if c.startswith("cond__")]
    F = f[cond_cols].to_numpy(dtype=float)
    pub = f["publication_id"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    chem = f["chemotype"].astype(str).to_numpy()
    okp = np.array([c not in v2drop for c in ids])

    par = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = sorted(par.cell_id.unique())
    clus = add_clusters(par[["cell_id", "chemotype", "publication"]].drop_duplicates(), bench.frame)
    cl_of = dict(zip(clus.cell_id, clus.cluster_cpg))

    def diff_fams(i, j) -> frozenset:
        s = set()
        for q, c in enumerate(cond_cols):
            a, b = F[i, q], F[j, q]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b) or not np.isclose(a, b):
                s.add(fam(c))
        return frozenset(s)

    # global subset indices over the 14 metals
    gidx = {n: {cmb: q for q, cmb in enumerate(itertools.combinations(range(N_LN), n))} for n in range(1, NMAX + 1)}
    store = {n: {k: np.full((len(c3), len(gidx[n])), np.nan, np.float32) for k in
                 [f"sse_{a}" for a in ARMS] + [f"bad_{a}" for a in ARMS] + [f"dok_{a}" for a in ARMS]
                 + ["dirn", "nmeas"]} for n in range(1, NMAX + 1)}
    cells, pairrows = [], []
    chk = {"pwl_maxabs": 0.0, "n_checked": 0}
    rng = np.random.default_rng(20261023)
    for ci, t in enumerate(c3):
        i = pos[t]
        Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        m = len(Mi)
        x = X[Mi]
        yv = Y[i, Mi]
        pr = [(p, p + 1) for p in range(m - 1) if Z_VEC[Mi[p + 1]] - Z_VEC[Mi[p]] == 1]
        pa = np.array([p for p, _ in pr])
        pb = np.array([q for _, q in pr])
        ysf = yv[pa] - yv[pb]
        cover = np.all(np.isfinite(Y[:, Mi]), axis=1)
        cover[i] = False
        tv2 = t in v2drop
        pools = {}
        same = np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]
        own = [int(j) for j in same if (not tv2) and okp[j] and bool(diff_fams(i, j) - FAMS_TRIVIAL)]
        pools["own"] = own
        pools["xext"] = [] if tv2 else np.where((pub == pub[i]) & (ext != ext[i]) & cover & okp)[0].tolist()
        pools["xpub"] = [] if tv2 else np.where((ext == ext[i]) & (pub != pub[i]) & cover & okp)[0].tolist()
        pools["chem"] = [] if tv2 else np.where((chem == chem[i]) & (pub != pub[i]) & cover & okp)[0].tolist()
        u = np.zeros(m)
        if GD in Mi:
            u[Mi.index(GD)] = HIST_D7_BETA
        bases = {"B0": np.zeros(m), "D7": u}
        T = {}
        for p in POOLS:
            if pools[p]:
                T[p] = Y[np.ix_(pools[p], Mi)].mean(axis=0)
                bases[f"SIB_{p}"] = T[p]
        wv = {p: len(pools[p]) / (len(pools[p]) + K_SH) for p in POOLS if pools[p]}
        # own-sibling condition-change classes (for later strata)
        own_f = [sorted(diff_fams(i, j) - FAMS_TRIVIAL) for j in own]
        cells.append({"ci": ci, "cell_id": t, "publication": pub[i], "extractant": ext[i], "chemotype": chem[i],
                      "cluster": cl_of[t], "c5": pub[i] == C5, "v2drop": tv2, "n_metals": m,
                      "metals": ",".join(LANTHANIDES[q] for q in Mi), "n_pairs": len(pr),
                      "N": len(pr), "sy": float(ysf.sum()), "syy": float((ysf ** 2).sum()),
                      **{f"n_{p}": len(pools[p]) for p in POOLS},
                      "own_fams": json.dumps([",".join(s) for s in own_f]),
                      "own_ids": json.dumps([ids[j] for j in own])})
        # ---- 0 reads: COPY / Gd-only
        sf0 = {"B0": np.zeros(len(pr)), "D7": u[pa] - u[pb]}
        for p in T:
            sf0[f"SIB_{p}"] = T[p][pa] - T[p][pb]
            sf0[f"SH_{p}"] = wv[p] * sf0[f"SIB_{p}"] + (1 - wv[p]) * sf0["D7"]
        for q in range(len(pr)):
            r0 = {"cell_id": t, "ci": ci, "pair": f"{LANTHANIDES[Mi[pa[q]]]}/{LANTHANIDES[Mi[pb[q]]]}",
                  "n_read": 0, "y": ysf[q]}
            for a, v in sf0.items():
                r0[f"pred_{a}"] = v[q]
                r0[f"rnd_{a}"] = (v[q] - ysf[q]) ** 2
                r0[f"hull_{a}"] = (v[q] - ysf[q]) ** 2
            pairrows.append(r0)
        # ---- HULL_MAXMIN order (label-free): lightest, heaviest, then max-min radius distance, ties -> lower Z
        order = [0, m - 1]
        while len(order) < m:
            rest = [q for q in range(m) if q not in order]
            dmin = [min(abs(x[q] - x[o]) for o in order) for q in rest]
            order.append(rest[int(np.argmax(dmin))])
        for n in range(1, min(NMAX, m) + 1):
            combos = list(itertools.combinations(range(m), n))
            R = np.zeros((len(combos), m), bool)
            for s, cmb in enumerate(combos):
                R[s, list(cmb)] = True
            gi = np.array([gidx[n][tuple(Mi[q] for q in cmb)] for cmb in combos])
            hull_s = combos.index(tuple(sorted(order[:n])))
            meas = R[:, pa] & R[:, pb]
            st = store[n]
            st["nmeas"][ci, gi] = meas.sum(1)
            elig = (~meas) & (np.abs(ysf) >= 0.1)[None, :]
            st["dirn"][ci, gi] = elig.sum(1)
            sfp = {}
            for a, b in bases.items():
                lev = b[None, :] + interp_batch(R, yv - b, x)
                sfp[a] = lev[:, pa] - lev[:, pb]
                if a == "D7" and rng.random() < 0.05:        # spot-check against r3.interp.pwl_levels
                    s = int(rng.integers(len(combos)))
                    ym = np.full(N_LN, np.nan)
                    ym[[Mi[q] for q in combos[s]]] = (yv - b)[list(combos[s])]
                    ref = interp.pwl_levels(ym, Mi, "R")[Mi] + b
                    chk["pwl_maxabs"] = max(chk["pwl_maxabs"], float(np.abs(ref - lev[s]).max()))
                    chk["n_checked"] += 1
            for p in T:
                sfp[f"SH_{p}"] = wv[p] * sfp[f"SIB_{p}"] + (1 - wv[p]) * sfp["D7"]
            for a, v in sfp.items():
                e = v - ysf[None, :]
                st[f"sse_{a}"][ci, gi] = (e ** 2).sum(1)
                st[f"bad_{a}"][ci, gi] = (np.abs(e) > 0.1 + 1e-12).sum(1)
                st[f"dok_{a}"][ci, gi] = (elig & (np.sign(v) == np.sign(ysf)[None, :])).sum(1)
            for q in range(len(pr)):
                rr = {"cell_id": t, "ci": ci, "pair": f"{LANTHANIDES[Mi[pa[q]]]}/{LANTHANIDES[Mi[pb[q]]]}",
                      "n_read": n, "y": ysf[q], "p_meas": float(meas[:, q].mean()),
                      "hull_meas": bool(meas[hull_s, q])}
                for a, v in sfp.items():
                    se = (v[:, q] - ysf[q]) ** 2
                    rr[f"rnd_{a}"] = float(se.mean())
                    rr[f"hull_{a}"] = float(se[hull_s])
                pairrows.append(rr)
        if ci % 40 == 0:
            print(f"{ci}/{len(c3)} {time.time() - t0:.0f}s", flush=True)
    cdf = pd.DataFrame(cells)
    cdf.to_parquet(OUT / "recipe_cells.parquet", index=False)
    pd.DataFrame(pairrows).to_parquet(OUT / "recipe_pairs.parquet", index=False)
    for n in range(1, NMAX + 1):
        np.savez_compressed(OUT / f"recipe_k{n}.npz", **store[n])
    (OUT / "recipe_build_log.json").write_text(json.dumps({"checks": chk, "cells": len(c3), "arms": ARMS,
                                                           "K_SH": K_SH, "beta": HIST_D7_BETA,
                                                           "seconds": time.time() - t0}, indent=1))
    print(json.dumps(chk), f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
