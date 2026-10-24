#!/usr/bin/env python
"""Probe P1 (idea M01, estimand E-CT), build step: one row per (design, split seed, C3 target cell, scored
strict-adjacent pair) with every E-CT arm, on the round-4 REF rows (k = 3 DOPT_Q, 4 reads).

EXPLORATORY: every number this produces is computed on the 229 already-seen C3 cells and the 5 dev seeds.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/probes/scripts/P1_ct_build.py --out <dir>

Inputs (read-only): round-4 REF_{BPG,BP}.parquet (rows, y, B1, G14@ZERO), round-4 results/REF/fitted_<design>.json
(B1's Gd amplitude per fold), the gen21 fold cache (nested DOPT_Q read sequences), the V1 bench (gen20k), round-3
datav2 (V2-1 / V2-2 / V2-3 duplicate lists).  Writes only <out>/P1_rows.parquet, <out>/P1_sibs.csv and
<out>/P1_build_log.json.

Definitions (frozen before any score was looked at in this probe; the population and K come from IDEAS_RANKED M01):
  sibling of target t: another bench cell (all 521, not only C3) with the same publication_id and the same extractant
      SMILES, with a finite V1 log D at every metal t has.  CLEAN population: t and the sibling outside the round-3
      V2-1 / V2-2 / V2-3 lists, and the sibling's recorded cond__ metadata differs from t's in at least one field
      family other than contact time / metal concentration (phase ratio is not recorded in the bench, so a
      phase-ratio-only sibling shows as "no recorded difference" and is dropped by the same rule).
      V1 population (E6 reproduction): every covering sibling.
  single-sibling SIB_j at k reads: levels Y_j + PWL_R[y_S - Y_j,S]  (exact at the reads).  SIB = mean_j SIB_j
      (= T + PWL_R[y_S - T_S] with T the mean sibling curve, by linearity).  COPY = SIB at 0 reads = T.
  B1_k: PWL_R[y_S - u_S] + u with u = beta at Gd (the fold's fitted round-4 B1 amplitude), on the nested DOPT_Q set
      seq[:k] (k = 1, 2, 3 pairs = 2, 3, 4 reads).  B1_3 is checked against the parquet's B1.
  SH_k = w SIB_k + (1 - w) B1_k, w = n / (n + K), K = 1.78 (and K = 2 for the E6 reproduction).  At 0 reads
      SH0 = w COPY + (1 - w) G14@ZERO (the round-4 zero-shot column) as the 0-read shrinkage target.
  NEAR: the nearest-condition sibling, distance |dlog10 acid| + |dlog10 L| + 1[diluent one-hot differs] (concentrations
      floored at 1e-4 M; one side missing = 1, both missing = 0); exact ties averaged, never broken by table order.
  RAND1: exact expectation over one uniformly drawn sibling (squared error averaged over siblings), raw and shrunk
      (w = 1 / (1 + K)).
  Subsampling (cells with n >= 5): exact enumeration of all m-subsets (random 5,000 if more), m = 1, 2, 3, 5;
      expected squared error of the raw mean-template SIB_m and of SH_m (w = m / (m + K)).
  OFF: B1 + publication x extractant pair-type offset of B1's OOF residual: sum_j r_j,p / (n_p + K) over the target's
      CLEAN siblings that are themselves C3 cells with a scored row of pair p in the same seed and design (leave-the-
      target-out by construction; K pinned at 1.78).  OFF_ALL: same with every other CLEAN C3 cell of the same
      publication x extractant (no coverage / condition rule).
  Placebo blends at the target's own w: PB_CHEM = template from every same-chemotype covering cell of other
      publications; PB_XEXT = same extractant, other publications only; PB_PUB = other extractants of the same
      publication.  NaN when the pool is empty.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402  (puts r3, g21, gen20k on sys.path)
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

K_SH = 1.78
K_E6 = 2.0
GD = CO.GD
SUB_M = (1, 2, 3, 5)
SUB_MAX = 5000
C5 = "pub_6a0ef8528c"
DESIGNS = ("BPG", "BP")
FAMS_TRIVIAL = {"contact_time", "metal_conc"}


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


def change_class(fams_list) -> str:
    """Cell-level condition-change type from the union of its siblings' differing field families."""
    U = set().union(*fams_list) if fams_list else set()
    U = U - {"contact_time", "metal_conc"}
    if U == {"acid_conc"}:
        return "acid_conc_only"
    if U and U <= {"acid_conc", "acid_type"}:
        return "acid_type(+conc)"
    if U == {"diluent"}:
        return "diluent_only"
    if U == {"extr_conc"}:
        return "extr_conc_only"
    if U == {"temperature"}:
        return "temperature_only"
    return "mixed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)                       # V1
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    cond_cols = [c for c in f.columns if c.startswith("cond__")]
    F = f[cond_cols]
    dil_cols = [c for c in cond_cols if c.startswith("cond__diluent__")]
    pub = f["publication_id"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    chem = f["chemotype"].astype(str).to_numpy()

    def diff_fams(i, j) -> frozenset:
        s = set()
        for c in cond_cols:
            x, y = F.iat[i, F.columns.get_loc(c)], F.iat[j, F.columns.get_loc(c)]
            if pd.isna(x) and pd.isna(y):
                continue
            if pd.isna(x) or pd.isna(y) or not np.isclose(float(x), float(y)):
                s.add(fam(c))
        return frozenset(s)

    def lg(v):
        return math.log10(max(float(v), 1e-4))

    def cond_dist(i, j) -> float:
        d = 0.0
        for c in ("cond__acid_concentration_M", "cond__extractant_concentration_M"):
            x, y = f.at[i, c], f.at[j, c]
            if pd.isna(x) and pd.isna(y):
                continue
            if pd.isna(x) or pd.isna(y):
                d += 1.0
                continue
            d += abs(lg(x) - lg(y))
        dx = f.loc[i, dil_cols].fillna(0).to_numpy(float)
        dy = f.loc[j, dil_cols].fillna(0).to_numpy(float)
        d += 0.0 if np.allclose(dx, dy) else 1.0
        return d

    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in DESIGNS}
    c3 = sorted(set(refs["BPG"].cell_id) | set(refs["BP"].cell_id))
    assert set(refs["BPG"].cell_id) == set(refs["BP"].cell_id)

    # ---------------- sibling / placebo pools (metadata + coverage only; no label is looked at) -------------
    sib_rec, pools = [], {}
    for t in c3:
        i = pos[t]
        met = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        cover = np.all(np.isfinite(Y[:, met]), axis=1)
        cover[i] = False
        same = np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]
        v1 = [int(j) for j in same]
        clean, fams = [], []
        for j in v1:
            dfm = diff_fams(i, j)
            ok = (t not in v2drop) and (ids[j] not in v2drop) and bool(dfm - FAMS_TRIVIAL)
            sib_rec.append({"target": t, "sibling": ids[j], "publication": pub[i], "diff": ",".join(sorted(dfm)) or "NONE",
                            "target_v2drop": t in v2drop, "sib_v2drop": ids[j] in v2drop, "clean": ok,
                            "dist": cond_dist(i, j), "sib_is_c3": ids[j] in set(c3)})
            if ok:
                clean.append(j)
                fams.append(dfm)
        okp = np.array([c not in v2drop for c in ids])
        plc_chem = np.where((chem == chem[i]) & (pub != pub[i]) & cover & okp)[0].tolist()
        plc_xext = np.where((ext == ext[i]) & (pub != pub[i]) & cover & okp)[0].tolist()
        plc_pub = np.where((pub == pub[i]) & (ext != ext[i]) & cover & okp)[0].tolist()
        # nearest-condition sibling(s) inside CLEAN (ties averaged)
        if clean:
            dd = np.array([cond_dist(i, j) for j in clean])
            near = [clean[q] for q in np.where(np.isclose(dd, dd.min(), atol=1e-9))[0]]
        else:
            near = []
        off_all = [pos[c] for c in c3 if c != t and pub[pos[c]] == pub[i] and ext[pos[c]] == ext[i]
                   and c not in v2drop] if t not in v2drop else []
        pools[t] = {"v1": v1, "clean": clean, "near": near, "plc_chem": plc_chem, "plc_xext": plc_xext,
                    "plc_pub": plc_pub, "off_sibs": [ids[j] for j in clean if ids[j] in set(c3)],
                    "off_all": [ids[j] for j in off_all], "cls": change_class(fams) if clean else "",
                    "sib_fams": [",".join(sorted(x)) for x in fams]}
    pd.DataFrame(sib_rec).to_csv(out / "P1_sibs.csv", index=False)

    # subset lists per cell (fixed per cell; reused across seeds and designs)
    subsets = {}
    for t, P in pools.items():
        n = len(P["clean"])
        if n < 5:
            continue
        for m in SUB_M:
            combos = list(itertools.combinations(range(n), m)) if math.comb(n, m) <= SUB_MAX else None
            if combos is None:
                rng = np.random.default_rng(int(hashlib.sha256(f"{t}:{m}".encode()).hexdigest()[:8], 16))
                combos = [tuple(sorted(rng.choice(n, m, replace=False))) for _ in range(SUB_MAX)]
            subsets[(t, m)] = np.array(combos)

    # ---------------- rows -------------------------------------------------------------------------------
    rows, checks = [], {"b1_maxabs": 0.0, "y_maxabs": 0.0, "readset_mismatch": 0, "rows": 0}
    for design in DESIGNS:
        par = refs[design]
        fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        offres = {}                                       # (seed, cell, A, B) -> y - B1 (OOF)
        for r in par[["split_seed", "cell_id", "A", "B", "y", "B1"]].itertuples(index=False):
            offres[(r.split_seed, r.cell_id, r.A, r.B)] = r.y - r.B1
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = C.load(design, seed, fold)
                beta = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
                u = np.zeros(N_LN)
                u[GD] = beta
                prow = par[(par.split_seed == seed) & (par.fold == fold)]
                pc = {c: g for c, g in prow.groupby("cell_id")}
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if t not in pc:
                        continue
                    g = pc[t].set_index(["A", "B"])
                    i = pos[t]
                    yi = Y[i]
                    metals = cell["metals"]
                    seq = cell["seqs"]["DOPT_Q"]
                    reads, b1, b0, sib_single = {}, {}, {}, {}
                    P0 = pools[t]
                    allpool = sorted(set(pools[t]["v1"]) | set(pools[t]["plc_chem"]) | set(pools[t]["plc_xext"])
                                     | set(pools[t]["plc_pub"]))
                    for k in (0, 1, 2, 3):
                        rd = sorted({m for p in seq[:k] for m in p}) if k else []
                        reads[k] = rd
                        ym = np.full(N_LN, np.nan)
                        ym[rd] = yi[rd]
                        if k:
                            ymu = ym - u
                            b1[k] = interp.pwl_levels(ymu, metals, "R") + u
                            b0[k] = interp.pwl_levels(ym, metals, "R")
                        sl = {}
                        for j in (allpool if k == 3 else P0["v1"]):
                            if k == 0:
                                sl[j] = Y[j].copy()
                            else:
                                res = np.full(N_LN, np.nan)
                                res[rd] = yi[rd] - Y[j, rd]
                                sl[j] = Y[j] + interp.pwl_levels(res, metals, "R")
                        sib_single[k] = sl
                    rs = ",".join(LANTHANIDES[m] for m in reads[3])
                    if rs != str(pc[t].read_set.iloc[0]):
                        checks["readset_mismatch"] += 1
                    P = pools[t]
                    for (A, B), pr in g.iterrows():
                        ia, ib = LANTHANIDES.index(A), LANTHANIDES.index(B)
                        y = yi[ia] - yi[ib]
                        checks["y_maxabs"] = max(checks["y_maxabs"], abs(y - pr.y))
                        sf = {k: (b1[k][ia] - b1[k][ib]) for k in (1, 2, 3)}
                        checks["b1_maxabs"] = max(checks["b1_maxabs"], abs(sf[3] - pr.B1))
                        r = {"design": design, "split_seed": seed, "fold": fold, "cell_id": t,
                             "publication": pub[i], "extractant": ext[i], "chemotype": chem[i], "A": A, "B": B,
                             "pair": f"{A}/{B}", "read_set": rs, "y": y, "B1_par": pr.B1, "G14Z": pr["G14@ZERO"],
                             "B1_1": sf[1], "B1_2": sf[2], "B1_3": sf[3],
                             "B0_1": b0[1][ia] - b0[1][ib], "B0_2": b0[2][ia] - b0[2][ib], "B0_3": b0[3][ia] - b0[3][ib],
                             "n_v1": len(P["v1"]), "n_clean": len(P["clean"]), "cls": P["cls"],
                             "c5": pub[i] == C5, "t_v2drop": t in v2drop,
                             "n_plc_chem": len(P["plc_chem"]), "n_plc_xext": len(P["plc_xext"]),
                             "n_plc_pub": len(P["plc_pub"]), "n_near": len(P["near"])}

                        def sfs(pool, k):
                            return np.array([sib_single[k][j][ia] - sib_single[k][j][ib] for j in pool], float)
                        for popn, K in (("V1", None), ("CL", None)):
                            pool = P["v1"] if popn == "V1" else P["clean"]
                            n = len(pool)
                            if n == 0:
                                continue
                            w = n / (n + K_SH)
                            w2 = n / (n + K_E6)
                            s = {k: sfs(pool, k) for k in (0, 1, 2, 3)}
                            r[f"{popn}_COPY"] = s[0].mean()
                            r[f"{popn}_SH0"] = w * s[0].mean() + (1 - w) * pr["G14@ZERO"]
                            for k in (1, 2, 3):
                                r[f"{popn}_SIB_{k}"] = s[k].mean()
                                r[f"{popn}_SH_{k}"] = w * s[k].mean() + (1 - w) * sf[k]
                            r[f"{popn}_SHK2_3"] = w2 * s[3].mean() + (1 - w2) * sf[3]
                            w1 = 1.0 / (1.0 + K_SH)
                            r[f"{popn}_E_RAND1_raw"] = float(((s[3] - y) ** 2).mean())
                            r[f"{popn}_E_RAND1_sh"] = float(((w1 * s[3] + (1 - w1) * sf[3] - y) ** 2).mean())
                            r[f"{popn}_E_RAND1_raw_0"] = float(((s[0] - y) ** 2).mean())
                        if P["clean"]:
                            n = len(P["clean"])
                            w = n / (n + K_SH)
                            s3 = sfs(P["clean"], 3)
                            nr = sfs(P["near"], 3).mean()
                            r["CL_NEAR_3"] = nr
                            wn = len(P["near"]) / (len(P["near"]) + K_SH)
                            r["CL_NEARSH_3"] = wn * nr + (1 - wn) * sf[3]
                            for nm, key in (("PB_CHEM", "plc_chem"), ("PB_XEXT", "plc_xext"), ("PB_PUB", "plc_pub")):
                                if P[key]:
                                    ps = sfs(P[key], 3).mean()
                                    r[f"CL_{nm}_raw"] = ps
                                    r[f"CL_{nm}"] = w * ps + (1 - w) * sf[3]
                            # OFF (leave-target-out pair-type offsets of B1's OOF residual)
                            for nm, lst in (("OFF", P["off_sibs"]), ("OFF_ALL", P["off_all"])):
                                rr = [offres[(seed, c, A, B)] for c in lst if (seed, c, A, B) in offres]
                                r[f"CL_{nm}"] = pr.B1 + (sum(rr) / (len(rr) + K_SH) if rr else 0.0)
                                r[f"CL_{nm}_np"] = len(rr)
                            if n >= 5:
                                for m in SUB_M:
                                    S = subsets[(t, m)]
                                    means = s3[S].mean(axis=1)
                                    wm = m / (m + K_SH)
                                    r[f"CL_E_SUB{m}_raw"] = float(((means - y) ** 2).mean())
                                    r[f"CL_E_SUB{m}_sh"] = float(((wm * means + (1 - wm) * sf[3] - y) ** 2).mean())
                        rows.append(r)
                        checks["rows"] += 1
                print(f"{design} {seed} done: {checks}", flush=True)
    t = pd.DataFrame(rows)
    t.to_parquet(out / "P1_rows.parquet", index=False)
    meta = {"checks": checks, "K_SH": K_SH, "K_E6": K_E6, "n_rows": len(t),
            "cells_v1_ge1": int(sum(1 for P in pools.values() if P["v1"])),
            "cells_clean_ge1": int(sum(1 for P in pools.values() if P["clean"])),
            "pools": {k: {kk: (len(vv) if isinstance(vv, list) else vv) for kk, vv in v.items() if kk != "sib_fams"}
                      for k, v in pools.items()},
            "sib_fams": {k: v["sib_fams"] for k, v in pools.items() if v["clean"]}}
    (out / "P1_build_log.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(json.dumps(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
