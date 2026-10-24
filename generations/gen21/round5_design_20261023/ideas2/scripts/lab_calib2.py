#!/usr/bin/env python
"""E-LAB, second pass: controls for the level-matched lab fingerprint and the reference-series fingerprint.

EXPLORATORY (229 already-seen C3 cells, 5 dev seeds).  Same shape-vector machinery as lab_calib.py (levels =
PWL_R[y_S - v_S] + v, v = (1 - w) u + w c, w = n / (n + K), K = 6.1, quadratic-detrended donor deviations).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab_calib2.py

Level matching (label-free for the target): at metal m a donor counts only if |log D_donor,m - L_t,m| <= tol, where
L_t is the target's B1 level (the true value at its 4 reads, B1's interpolation elsewhere).  tol = 1.0 was fixed
before any score; 0.5 and 2.0 are sensitivities.

Arms: MX (unmatched, as lab_calib), MX_LM{05,10,20}; MXH_LM10 / MXO_LM10 (same / other chemotype, same publication);
MG_LM10_all (every cell of every other publication, level-matched: a universal level-conditional shape, placebo for
"lab"); MG_LM10_tr (the same, donors restricted to the fold's training cells: the E-A-legal version); MCO_LM10 (same
chemotype, other publications); E_MXO_PB (expected squared error of a count-matched placebo for MXO: the same number of
donors drawn from other-publication, other-chemotype cells, 30 seeded draws); reference series: MREF_cons (consensus
TODGA shape = equal-weight mean of the other publications' TODGA shapes), MREF_fp (B1 + w_own * (own-TODGA shape -
consensus): the lab-specific part only), MREF_own (as lab_calib).
Writes ideas2/lab_out/lab_rows2.parquet and prints the tables.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
from lab_calib import detrend, shape, levels, OUT, K6  # noqa: E402
from lab_analyze import r2, rel, infer  # noqa: E402

GD = CO.GD
REF_NAME = "TODGA"
NPB = 30
pd.set_option("display.width", 250)


def lm_shape(D2, Y, donors, lv, tol, K):
    if len(donors) == 0:
        return np.zeros(N_LN), np.zeros(N_LN)
    A = D2[donors]
    msk = np.isfinite(A) & (np.abs(Y[donors] - lv[None, :]) <= tol)
    AA = np.where(msk, A, np.nan)
    n = msk.sum(0).astype(float)
    c = np.where(n > 0, np.nansum(AA, 0) / np.maximum(n, 1), 0.0)
    return c, n / (n + K)


def main() -> int:
    t0 = time.time()
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
    has = np.isfinite(D2).any(1) & okd
    ar = np.arange(len(f))
    ref_pubs = sorted({pub[i] for i in range(len(f)) if name[i] == REF_NAME})
    tod_shape = {p: shape(D2, np.where(has & (pub == p) & (name == REF_NAME))[0], K6) for p in ref_pubs}
    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in ("BPG", "BP")}
    rows = []
    for design in ("BPG", "BP"):
        par = refs[design]
        fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = C.load(design, seed, fold)
                train = np.zeros(len(f), bool)
                train[np.asarray(rec["outer"]["train_index"], int)] = True
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
                    base = has & (ar != i)
                    sp = base & (pub == pub[i])
                    dX = np.where(sp & (ext != ext[i]))[0]
                    dH = np.where(sp & (ext != ext[i]) & (chem == chem[i]))[0]
                    dO = np.where(sp & (ext != ext[i]) & (chem != chem[i]))[0]
                    dG = np.where(base & (pub != pub[i]))[0]
                    dGt = np.where(base & (pub != pub[i]) & train)[0]
                    dC = np.where(base & (pub != pub[i]) & (chem == chem[i]))[0]
                    L = {"B1": levels(yi, reads, metals, u)}
                    lv = L["B1"]

                    def lev(cw):
                        c, w = cw
                        return levels(yi, reads, metals, (1 - w) * u + w * c)
                    L["MX"] = lev(shape(D2, dX, K6))
                    for tol, nm in ((0.5, "05"), (1.0, "10"), (2.0, "20")):
                        L[f"MX_LM{nm}"] = lev(lm_shape(D2, Y, dX, lv, tol, K6))
                    L["MXH_LM10"] = lev(lm_shape(D2, Y, dH, lv, 1.0, K6))
                    L["MXO_LM10"] = lev(lm_shape(D2, Y, dO, lv, 1.0, K6))
                    L["MXO"] = lev(shape(D2, dO, K6))
                    L["MG_LM10_all"] = lev(lm_shape(D2, Y, dG, lv, 1.0, K6))
                    L["MG_LM10_tr"] = lev(lm_shape(D2, Y, dGt, lv, 1.0, K6))
                    L["MCO_LM10"] = lev(lm_shape(D2, Y, dC, lv, 1.0, K6))
                    PB = []
                    if len(dO):
                        pool = np.where(base & (pub != pub[i]) & (chem != chem[i]))[0]
                        rng = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
                        for _ in range(NPB):
                            PB.append(lev(shape(D2, rng.choice(pool, len(dO), replace=False), K6)))
                    reft = pub[i] in ref_pubs and name[i] != REF_NAME
                    if reft:
                        own = np.where(sp & (name == REF_NAME))[0]
                        reft = len(own) > 0
                    if reft:
                        co, wo = shape(D2, own, K6)
                        oth = [tod_shape[p][0] for p in ref_pubs if p != pub[i] and tod_shape[p][1].max() > 0]
                        cons = np.mean(oth, 0)
                        L["MREF_own"] = levels(yi, reads, metals, (1 - wo) * u + wo * co)
                        L["MREF_cons"] = levels(yi, reads, metals, (1 - wo) * u + wo * cons)
                        L["MREF_fp"] = levels(yi, reads, metals, u + wo * (co - cons))
                    for r in pcg[t].itertuples(index=False):
                        ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                        o = {"design": design, "split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i],
                             "extractant": ext[i], "chemotype": chem[i], "pair": f"{r.A}/{r.B}", "y": r.y, "B1_par": r.B1,
                             "t_v2drop": t in v2drop, "n_MX": len(dX), "n_MXH": len(dH), "n_MXO": len(dO), "ref_target": reft,
                             "lvA": lv[ia], "lvB": lv[ib]}
                        for k, l2 in L.items():
                            o[k] = l2[ia] - l2[ib]
                        if PB:
                            o["E_MXO_PB"] = float(np.mean([(l2[ia] - l2[ib] - r.y) ** 2 for l2 in PB]))
                        rows.append(o)
            print(f"{design} done {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)
    print("max |B1 - B1_par|:", float((T.B1 - T.B1_par).abs().max()))
    T.to_parquet(OUT / "lab_rows2.parquet", index=False)
    T = M4.add_clusters(T, f)

    def table(title, d, arms):
        print(f"\n--- {title}: cells {d.cell_id.nunique()}, pubs {d.publication.nunique()}, rows/seed {len(d) / 5:.0f} ---")
        print(pd.DataFrame([{"arm": a, "R2": r2(d, a), "rel_vs_B1": rel(d, a)} for a in arms
                            if a in d and not d[a].isna().any()]).round(4).to_string(index=False))

    def inf(title, d, pairs):
        out = []
        for a, b in pairs:
            if d[a].isna().any() or d[b].isna().any():
                continue
            r = infer(d, a, b)
            out.append({"contrast": f"{a} - {b}", "dR2": r2(d, a) - r2(d, b), **{k: r[k] for k in
                        ("rel", "lo", "hi", "G", "up", "down", "p_sf", "top3", "jk_min", "jk_min_pub")}})
        print(f"[{title}]")
        print(pd.DataFrame(out).round(4).to_string(index=False))

    for design in ("BPG", "BP"):
        d0 = T[T.design == design]
        print(f"\n==================== {design} ====================")
        A = ["B1", "MX", "MX_LM05", "MX_LM10", "MX_LM20", "MXH_LM10", "MXO_LM10", "MXO", "MG_LM10_all", "MG_LM10_tr",
             "MCO_LM10"]
        table("ALL C3 rows", d0, A)
        X = d0[(d0.n_MX > 0) & ~d0.t_v2drop]
        table("POPX", X, A)
        inf("POPX", X, [("MX_LM10", "B1"), ("MX_LM10", "MG_LM10_all"), ("MX_LM10", "MG_LM10_tr"), ("MX_LM10", "MCO_LM10"),
                        ("MX_LM10", "MX"), ("MG_LM10_tr", "B1")])
        inf("ALL rows", d0, [("MG_LM10_tr", "B1"), ("MG_LM10_all", "B1")])
        XL = X[~X.publication.isin(["pub_126bbe7976", "pub_1bd76c48d9"])]
        table("POPX without pub_126bbe7976, pub_1bd76c48d9", XL, A)
        XG = X[X.cluster_cpg != "g:pub_74428aeab2"]
        table("POPX outside the giant cluster", XG, A)
        O = X[X.n_MXO > 0]
        table("rows with an other-chemotype same-publication donor", O, ["B1", "MXO", "E_MXO_PB", "MXO_LM10", "MG_LM10_all"])
        inf("other-chemotype donors", O, [("MXO", "E_MXO_PB"), ("MXO_LM10", "MG_LM10_all"), ("MXO_LM10", "B1")])
        inw = X.lvA.between(-2, 2) & X.lvB.between(-2, 2)
        table("POPX inside the level window", X[inw], ["B1", "MX", "MX_LM10", "MG_LM10_all", "MG_LM10_tr"])
        R = d0[d0.ref_target & ~d0.t_v2drop]
        table("REFERENCE SERIES (TODGA in the publication; non-TODGA targets)", R,
              ["B1", "MREF_own", "MREF_cons", "MREF_fp", "MX", "MX_LM10"])
        inf("reference series", R, [("MREF_fp", "B1"), ("MREF_own", "MREF_cons"), ("MREF_fp", "MREF_cons")])
        if design == "BPG":
            for p, g in R.groupby("publication"):
                print(f"   {p}: own {rel(g, 'MREF_own'):+.3f}  consensus-TODGA {rel(g, 'MREF_cons'):+.3f}  lab part only "
                      f"{rel(g, 'MREF_fp'):+.3f}  ({g.cell_id.nunique()} cells)")
            print("\nPOPX per publication, BPG: rel SSE of MX_LM10 / MG_LM10_all / MX")
            for p, g in X.groupby("publication"):
                print(f"   {p} ({g.cell_id.nunique():2d} cells): {rel(g, 'MX_LM10'):+.3f} / {rel(g, 'MG_LM10_all'):+.3f} / "
                      f"{rel(g, 'MX'):+.3f}")
    print(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
