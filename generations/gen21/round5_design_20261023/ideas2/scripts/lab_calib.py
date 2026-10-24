#!/usr/bin/env python
"""E-LAB (laboratory fingerprint calibration): exploratory checks on the round-4 REF rows.

EXPLORATORY.  Every number this produces comes from the 229 already-seen C3 cells and the 5 dev seeds.  Nothing is
sealed; K values are pinned from earlier analyses of the same data (K = 1.78 from r = 0.36, K = 6.1 from r = 0.14,
E6 section 5) and were not tuned here.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_calib.py

Inputs (read-only): round-4 REF_{BPG,BP}.parquet, results/REF/fitted_<design>.json, the gen21 fold cache (DOPT_Q read
sets), the V1 bench (gen20k), round-3 datav2 V2-1/2/3 lists.  Writes only ideas2/lab_out/lab_rows.parquet.

Model.  Every arm is a "shape-vector" interpolation: levels = PWL_R[y_S - v_S] + v (exact at the 4 reads), with
B1 = shape u (beta at Gd, the fold's fitted round-4 amplitude, 0 elsewhere).  A per-METAL laboratory fingerprint
replaces u by v_m = (1 - w_m) u_m + w_m c_m, where c_m is the mean, over donor cells, of the donor's deviation from
its own least-squares quadratic in -r (CN8 radius coordinate, the PWL_R axis), and w_m = n_m / (n_m + K) with n_m
the number of donors measured at metal m.  Donors are bench cells (all 521) outside the round-3 V2-1/2/3 lists with
>= 6 measured metals; the target is never its own donor.  Because the shape passes through the target's reads,
whatever the reads already capture of the fingerprint is not added twice (unlike a raw pair offset).

Arms (donor pools; "same pub" = same publication_id):
  MX   other extractants of the same publication (the E-LAB channel), K = 6.1 (MX1: K = 1.78; MXc: cubic detrend)
  MP   every other cell of the publication (any extractant)
  MS   same extractant, same publication only (detrended condition-transfer analogue of P1's SH)
  MXH / MXO  other extractants of the same publication with the same / another chemotype (homologue vs not)
  MX_light / MX_ce / MX_gd / MX_heavy   MX applied only at La-Nd / Ce / Gd / Sm-Lu (B1's u elsewhere)
  MX_LM / MX_LX  per metal, only donors whose log D at m is within / outside 1 log unit of the target's B1 level
  MG   all cells of every other publication (a universal per-metal prior; placebo)
  MCO  same chemotype, other publications (placebo)
  E_MRP  expected squared error over ONE random other publication's cells (each with its own n_m; placebo "another
       lab's fingerprint")
  OFFX / OFFP  P1-style pair-type offsets of B1's OOF residual over the other-extractant / all other C3 cells of the
       publication (leave-target-out), K = 6.1 and 1.78
  Reference-series protocol (TODGA as the reference; targets = non-TODGA C3 cells of publications that also report
  TODGA): MREF_own (own publication's TODGA cells), E_MREF_oth (expected error with one other publication's TODGA
  cells), MREF_pool (all other publications' TODGA cells pooled), OFFREF (pair offsets from own TODGA C3 cells).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab_out")
K1, K6 = 1.78, 6.1
GD = CO.GD
DESIGNS = ("BPG", "BP")
X = np.asarray(interp._coords("R"), float)
LIGHT = [0, 1, 2, 3]
HEAVY = list(range(4, 14))
REF_NAME = "TODGA"


def detrend(Y: np.ndarray, deg: int, min_n: int) -> np.ndarray:
    Dv = np.full_like(Y, np.nan)
    for i in range(len(Y)):
        m = np.isfinite(Y[i])
        if m.sum() < min_n:
            continue
        co = np.polyfit(X[m], Y[i, m], deg)
        Dv[i, m] = Y[i, m] - np.polyval(co, X[m])
    return Dv


def shape(Dv: np.ndarray, donors, K: float):
    if len(donors) == 0:
        return np.zeros(N_LN), np.zeros(N_LN)
    A = Dv[np.asarray(donors, int)]
    n = np.isfinite(A).sum(0).astype(float)
    with np.errstate(invalid="ignore"):
        c = np.where(n > 0, np.nansum(A, 0) / np.maximum(n, 1), 0.0)
    return c, n / (n + K)


def levels(yi, reads, metals, v):
    xr = X[reads]
    yr = yi[reads] - v[reads]
    o = np.argsort(xr)
    out = np.full(N_LN, np.nan)
    mm = np.asarray(metals, int)
    out[mm] = np.interp(X[mm], xr[o], yr[o]) + v[mm]
    out[reads] = yi[reads]
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f["publication_id"].astype(str).to_numpy()
    ext = f["extractant"].astype(str).to_numpy()
    chem = f["chemotype"].astype(str).to_numpy()
    name = f["extractant_name"].astype(str).to_numpy()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    D2 = detrend(Y, 2, 6)
    D3 = detrend(Y, 3, 8)
    has = np.isfinite(D2).any(1) & okd
    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in DESIGNS}
    c3 = sorted(set(refs["BPG"].cell_id))
    c3set = set(c3)
    allp = sorted(set(pub))
    ref_pubs = sorted({pub[i] for i in range(len(f)) if name[i] == REF_NAME})

    # ---- static donor pools and shapes per target ----
    S = {}
    for t in c3:
        i = pos[t]
        base = has & (np.arange(len(f)) != i)
        sp = base & (pub == pub[i])
        pools = {
            "MX": np.where(sp & (ext != ext[i]))[0], "MP": np.where(sp)[0], "MS": np.where(sp & (ext == ext[i]))[0],
            "MXH": np.where(sp & (ext != ext[i]) & (chem == chem[i]))[0],
            "MXO": np.where(sp & (ext != ext[i]) & (chem != chem[i]))[0],
            "MG": np.where(base & (pub != pub[i]))[0], "MCO": np.where(base & (pub != pub[i]) & (chem == chem[i]))[0],
        }
        sh = {}
        sh["MX"] = shape(D2, pools["MX"], K6)
        sh["MX1"] = shape(D2, pools["MX"], K1)
        sh["MXc"] = shape(D3, pools["MX"], K6)
        sh["MP"] = shape(D2, pools["MP"], K6)
        sh["MS"] = shape(D2, pools["MS"], K1)
        sh["MXH"] = shape(D2, pools["MXH"], K6)
        sh["MXO"] = shape(D2, pools["MXO"], K6)
        sh["MG"] = shape(D2, pools["MG"], K6)
        sh["MCO"] = shape(D2, pools["MCO"], K6)
        # one random other publication (expectation over publications that have donors)
        rp = []
        for p in allp:
            if p == pub[i]:
                continue
            dd = np.where(base & (pub == p))[0]
            if len(dd):
                rp.append(shape(D2, dd, K6))
        sh["RP"] = rp
        # reference-series protocol
        refinfo = None
        if pub[i] in ref_pubs and name[i] != REF_NAME:
            own = np.where(sp & (name == REF_NAME))[0]
            if len(own):
                oth = []
                for p in ref_pubs:
                    if p == pub[i]:
                        continue
                    dd = np.where(base & (pub == p) & (name == REF_NAME))[0]
                    if len(dd):
                        oth.append(shape(D2, dd, K6))
                pool = np.where(base & (pub != pub[i]) & (name == REF_NAME))[0]
                refinfo = {"own": shape(D2, own, K6), "oth": oth, "pool": shape(D2, pool, K6), "n_own": len(own),
                           "own_c3": [ids[j] for j in own if ids[j] in c3set]}
        S[t] = {"pools": {k: len(v) for k, v in pools.items()}, "sh": sh, "ref": refinfo, "MXd": pools["MX"]}

    rows = []
    for design in DESIGNS:
        par = refs[design]
        fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        res = {(r.split_seed, r.cell_id, r.A, r.B): r.y - r.B1
               for r in par[["split_seed", "cell_id", "A", "B", "y", "B1"]].itertuples(index=False)}
        pc3 = par.drop_duplicates("cell_id").set_index("cell_id")
        PUB, EXT, NAME = pc3.publication.to_dict(), pc3.extractant.to_dict(), {c: name[pos[c]] for c in c3}
        offd = {}
        for x in c3:
            same = [c for c in c3 if c != x and c not in v2drop and PUB[c] == PUB[x]]
            offd[("P", x)] = same
            offd[("X", x)] = [c for c in same if EXT[c] != EXT[x]]
            offd[("R", x)] = [c for c in same if NAME[c] == REF_NAME and NAME[x] != REF_NAME]
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = C.load(design, seed, fold)
                beta = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
                u = np.zeros(N_LN)
                u[GD] = beta
                prow = par[(par.split_seed == seed) & (par.fold == fold)]
                pcg = {c: g for c, g in prow.groupby("cell_id")}
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if t not in pcg:
                        continue
                    i = pos[t]
                    yi = Y[i]
                    metals = list(cell["metals"])
                    reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                    St = S[t]
                    L = {"B1": levels(yi, reads, metals, u)}

                    def blend(c, w, sub=None):
                        v = u.copy()
                        idx = np.arange(N_LN) if sub is None else np.asarray(sub)
                        v[idx] = (1 - w[idx]) * u[idx] + w[idx] * c[idx]
                        return v
                    for k in ("MX", "MX1", "MXc", "MP", "MS", "MXH", "MXO", "MG", "MCO"):
                        c, w = St["sh"][k]
                        L[k] = levels(yi, reads, metals, blend(c, w))
                    c, w = St["sh"]["MX"]
                    L["MX_light"] = levels(yi, reads, metals, blend(c, w, LIGHT))
                    L["MX_ce"] = levels(yi, reads, metals, blend(c, w, [1]))
                    L["MX_gd"] = levels(yi, reads, metals, blend(c, w, [GD]))
                    L["MX_heavy"] = levels(yi, reads, metals, blend(c, w, HEAVY))
                    # level-matched vs mismatched donors (per metal, relative to the target's B1 level: label-free)
                    dn = St["MXd"]
                    if len(dn):
                        lv = L["B1"]
                        A = D2[dn]
                        close = np.abs(Y[dn] - lv[None, :]) <= 1.0
                        for nm, msk in (("MX_LM", close), ("MX_LX", ~close)):
                            AA = np.where(msk & np.isfinite(A), A, np.nan)
                            n = np.isfinite(AA).sum(0).astype(float)
                            cc = np.where(n > 0, np.nansum(AA, 0) / np.maximum(n, 1), 0.0)
                            L[nm] = levels(yi, reads, metals, blend(cc, n / (n + K6)))
                    RP = [levels(yi, reads, metals, blend(c, w)) for c, w in St["sh"]["RP"]]
                    ref = St["ref"]
                    if ref is not None:
                        L["MREF_own"] = levels(yi, reads, metals, blend(*ref["own"]))
                        L["MREF_pool"] = levels(yi, reads, metals, blend(*ref["pool"]))
                        RO = [levels(yi, reads, metals, blend(c, w)) for c, w in ref["oth"]]
                    g = pcg[t]
                    for r in g.itertuples(index=False):
                        ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                        o = {"design": design, "split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i],
                             "extractant": ext[i], "name": name[i], "chemotype": chem[i], "pair": f"{r.A}/{r.B}",
                             "y": r.y, "B1_par": r.B1, "t_v2drop": t in v2drop,
                             "lvA": L["B1"][ia], "lvB": L["B1"][ib]}
                        for k, lv in L.items():
                            o[k] = lv[ia] - lv[ib]
                        if RP:
                            o["E_MRP"] = float(np.mean([(lv[ia] - lv[ib] - r.y) ** 2 for lv in RP]))
                        if ref is not None and RO:
                            o["E_MREF_oth"] = float(np.mean([(lv[ia] - lv[ib] - r.y) ** 2 for lv in RO]))
                        for nm, key in (("OFFX", "X"), ("OFFP", "P"), ("OFFREF", "R")):
                            rr = [res[(seed, c, r.A, r.B)] for c in offd[(key, t)] if (seed, c, r.A, r.B) in res]
                            o[f"{nm}_np"] = len(rr)
                            o[nm] = r.B1 + (sum(rr) / (len(rr) + K6) if rr else 0.0)
                            o[nm + "1"] = r.B1 + (sum(rr) / (len(rr) + K1) if rr else 0.0)
                        for k, v in St["pools"].items():
                            o[f"n_{k}"] = v
                        o["ref_target"] = ref is not None
                        o["n_ref_own"] = ref["n_own"] if ref is not None else 0
                        rows.append(o)
            print(f"{design} done {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)
    print("max |B1 - B1_par|:", float((T.B1 - T.B1_par).abs().max()), "rows", len(T))
    T.to_parquet(OUT / "lab_rows.parquet", index=False)
    print(f"wrote {OUT / 'lab_rows.parquet'} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
