#!/usr/bin/env python
"""Skeptic B check of idea lab-3, second pass.  EXPLORATORY (already-seen 229 C3 cells, 5 dev seeds).

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab3skepB_extra.py

Questions:
  a. Under BP / BPG, how many same-chemotype other-publication donors lie in the target fold's TRAINING cells (all
     seeds, all folds)?  If zero, the proposed MCO_*_tr arms are B1 by construction.
  b. How many of the "all other publications" donors of MG / MG_LM10_all are test-fold cells of the target's own
     cluster (cluster_cpg)?  That is the leak that makes MG_LM10_all beat MG_LM10_tr.
  c. Single-metal versions of the NON-fold-legal MCO (all other-publication same-chemotype donors: the E-KX / LOPO
     analogue, M07's estimand): does the quadratic-residual estimator recover M07's sc009 Nd lead?
  d. Positive control for the estimator: the Gd potential from the quadratic-residual mean over ALL other-publication
     donors vs B1's fold-fitted beta (both universal Gd terms).
Prints only.
"""
from __future__ import annotations

import json
import sys
import time

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
from lab_calib import detrend, shape, levels, K6  # noqa: E402
from lab_analyze import r2, rel, infer  # noqa: E402

GD = CO.GD
ND = LANTHANIDES.index("Nd")
C5 = "pub_6a0ef8528c"
pd.set_option("display.width", 250)


def main() -> int:
    t0 = time.time()
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f["publication_id"].astype(str).to_numpy()
    chem = f["chemotype"].astype(str).to_numpy()
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    okd = np.array([c not in v2drop for c in ids])
    D2 = detrend(Y, 2, 6)
    has = np.isfinite(D2).any(1) & okd
    ar = np.arange(len(f))
    fc = M4.add_clusters(f.rename(columns={"publication_id": "publication"})[["publication", "chemotype"]].astype(str), f)
    clu = fc["cluster_cpg"].to_numpy()
    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in ("BPG", "BP")}
    cnt = []
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
                    base = has & (ar != i) & (pub != pub[i])
                    dG = np.where(base)[0]
                    dC = np.where(base & (chem == chem[i]))[0]
                    cnt.append({"design": design, "seed": seed, "cell": t,
                                "nC_train": int((base & train & (chem == chem[i])).sum()),
                                "nC_all": len(dC), "nG_all": len(dG), "nG_train": int((base & train).sum()),
                                "nG_test_samecluster": int((base & ~train & (clu == clu[i])).sum()),
                                "nG_test_othercluster": int((base & ~train & (clu != clu[i])).sum())})
                    yi = Y[i]
                    metals = list(cell["metals"])
                    reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                    L = {"B1": levels(yi, reads, metals, u)}
                    cC, wC = shape(D2, dC, K6)
                    cG, wG = shape(D2, dG, K6)
                    for nm, (c, w) in (("MCO", (cC, wC)), ("MG", (cG, wG))):
                        for m in (ND, GD, LANTHANIDES.index("Ce")):
                            v = u.copy()
                            v[m] = (1 - w[m]) * u[m] + w[m] * c[m]
                            L[f"{nm}_only{LANTHANIDES[m]}"] = levels(yi, reads, metals, v)
                        L[nm] = levels(yi, reads, metals, (1 - w) * u + w * c)
                    # MCO with B1's Gd beta kept (prior applied everywhere except Gd)
                    v = (1 - wC) * u + wC * cC
                    v[GD] = u[GD]
                    L["MCO_keepGd"] = levels(yi, reads, metals, v)
                    for r in pcg[t].itertuples(index=False):
                        ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                        o = {"design": design, "split_seed": seed, "cell_id": t, "publication": pub[i],
                             "chemotype": chem[i], "pair": f"{r.A}/{r.B}", "y": r.y, "cG": cG[GD], "cN": cC[ND]}
                        for k, l2 in L.items():
                            o[k] = l2[ia] - l2[ib]
                        rows.append(o)
    K = pd.DataFrame(cnt)
    print("a/b. donor counts per target (median over cells x seeds x folds):")
    print(K.groupby("design")[["nC_train", "nC_all", "nG_all", "nG_train", "nG_test_samecluster",
                               "nG_test_othercluster"]].median().to_string())
    print("   targets with >= 1 same-chemotype TRAINING donor:", K.groupby("design").apply(
        lambda g: int((g.nC_train > 0).sum())).to_dict(), "of", K.groupby("design").size().to_dict())
    T = pd.DataFrame(rows)
    for design in ("BPG", "BP"):
        d0 = T[T.design == design]
        dn = d0[d0.publication != C5]
        s9 = dn[dn.chemotype == "sc009"]
        pn = s9[s9.pair == "Pr/Nd"]
        print(f"\n===== {design} (non-fold-legal donors: every other publication) =====")
        print(f"  mean c_Gd over all-other-pub donors {d0.cG.mean():+.4f}; mean sc009 c_Nd {s9.cN.mean():+.4f}")
        for a in ("MCO", "MCO_keepGd", "MCO_onlyNd", "MCO_onlyGd", "MCO_onlyCe", "MG", "MG_onlyNd", "MG_onlyGd",
                  "MG_onlyCe"):
            ia = infer(dn, a)
            print(f"  {a:12s} all rows no C5 rel {ia['rel']:+.4f} [{ia['lo']:+.4f}, {ia['hi']:+.4f}] up/down "
                  f"{ia['up']}/{ia['down']} | with C5 {rel(d0, a):+.4f} | sc009 no C5 {rel(s9, a):+.4f} | sc009 "
                  f"Pr/Nd R2 {r2(pn, 'B1'):.3f} -> {r2(pn, a):.3f}")
    print(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
