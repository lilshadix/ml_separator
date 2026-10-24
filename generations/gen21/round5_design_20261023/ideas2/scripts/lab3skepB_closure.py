#!/usr/bin/env python
"""Skeptic B check of idea lab-3 (closure line: per-metal priors from other publications do not beat B1).

EXPLORATORY: the 229 already-seen C3 cells and 5 dev seeds; nothing here is a test.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab3skepB_closure.py

Re-uses lab_calib / lab_calib2 machinery (levels = PWL_R[y_S - v_S] + v; v = (1-w) u + w c; quadratic-detrended
donor deviations; K = 6.1).  Adds:
  1. the proposed replay itself: MG_tr / MCO_tr (training-fold donors only, unmatched) and their level-matched
     versions at tol 0.5 / 1 / 2, with and without C5-BTBP (pub_6a0ef8528c), publication-unit and cluster-jackknife
     intervals;
  2. shrinkage sweep v = u + lam * w * (c - u), lam in {0.1, 0.25, 0.5}: is the null an artefact of w ~ 1 for
     hundred-donor pools?
  3. single-metal versions (only metal m takes the prior, B1's u elsewhere) for MCO_tr and MG_tr: does an
     all-metal average hide a single-metal effect (M07's sc009 Nd term)?
Writes ideas2/lab3skepB_out/lab3_rows.parquet and prints tables.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as SS  # noqa: E402

from r4 import paths  # noqa: E402
from r4 import metrics as M4  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
from lab_calib import detrend, shape, levels, K6  # noqa: E402
from lab_calib2 import lm_shape  # noqa: E402
from lab_analyze import r2, rel, infer  # noqa: E402

OUT = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\lab3skepB_out")
GD = CO.GD
C5 = "pub_6a0ef8528c"
LAMS = (0.1, 0.25, 0.5)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)


def cjk(d, arm, base="B1"):
    """Leave-one-cluster-out jackknife t interval on rel SSE (cluster_cpg units)."""
    a = (d[arm] - d.y) ** 2
    b = (d[base] - d.y) ** 2
    g = pd.DataFrame({"a": a, "b": b, "c": d.cluster_cpg}).groupby("c")[["a", "b"]].sum()
    G = len(g)
    point = 1 - g.a.sum() / g.b.sum()
    jk = np.array([1 - g.a.drop(k).sum() / g.b.drop(k).sum() for k in g.index])
    se = float(np.sqrt((G - 1) / G * ((jk - jk.mean()) ** 2).sum()))
    q = SS.t.ppf(0.975, G - 1)
    return point, point - q * se, point + q * se, G


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
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
    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in ("BPG", "BP")}
    rows = []
    wlog = []
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
                    base = has & (ar != i) & (pub != pub[i])
                    dGt = np.where(base & train)[0]
                    dCt = np.where(base & train & (chem == chem[i]))[0]
                    L = {"B1": levels(yi, reads, metals, u)}
                    lv = L["B1"]

                    def lev(v):
                        return levels(yi, reads, metals, v)
                    shp = {"MG_tr": shape(D2, dGt, K6), "MCO_tr": shape(D2, dCt, K6)}
                    for tol, nm in ((0.5, "05"), (1.0, "10"), (2.0, "20")):
                        shp[f"MG_LM{nm}_tr"] = lm_shape(D2, Y, dGt, lv, tol, K6)
                        shp[f"MCO_LM{nm}_tr"] = lm_shape(D2, Y, dCt, lv, tol, K6)
                    for k, (c, w) in shp.items():
                        L[k] = lev((1 - w) * u + w * c)
                    if design == "BPG" and seed == paths.DEV_SEEDS[0]:
                        wlog.append({"cell": t, "nG": len(dGt), "nC": len(dCt),
                                     "wG_med": float(np.median(shp["MG_tr"][1])),
                                     "wC_med": float(np.median(shp["MCO_tr"][1]))})
                    for k in ("MG_tr", "MCO_tr"):
                        c, w = shp[k]
                        for lam in LAMS:
                            L[f"{k}_lam{lam}"] = lev(u + lam * w * (c - u))
                        for m in range(N_LN):
                            v = u.copy()
                            v[m] = (1 - w[m]) * u[m] + w[m] * c[m]
                            L[f"{k}_only{LANTHANIDES[m]}"] = lev(v)
                    for r in pcg[t].itertuples(index=False):
                        ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                        o = {"design": design, "split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i],
                             "chemotype": chem[i], "pair": f"{r.A}/{r.B}", "y": r.y, "B1_par": r.B1,
                             "nCt": len(dCt)}
                        for k, l2 in L.items():
                            o[k] = l2[ia] - l2[ib]
                        rows.append(o)
            print(f"{design} done {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)
    print("max |B1 - B1_par|:", float((T.B1 - T.B1_par).abs().max()), "rows", len(T))
    T = M4.add_clusters(T, f)
    T.to_parquet(OUT / "lab3_rows.parquet", index=False)
    W = pd.DataFrame(wlog)
    print("\nshrinkage weights (BPG, seed 0): median n donors MG_tr", W.nG.median(), " MCO_tr", W.nC.median(),
          "| median per-cell median w: MG", round(W.wG_med.median(), 3), " MCO", round(W.wC_med.median(), 3),
          "| cells with no same-chemotype training donor:", int((W.nC == 0).sum()), "of", len(W))

    main_arms = ["MG_tr", "MCO_tr"] + [f"{a}_LM{n}_tr" for a in ("MG", "MCO") for n in ("05", "10", "20")]
    for design in ("BPG", "BP"):
        d0 = T[T.design == design]
        print(f"\n==================== {design} ====================")
        for lab, d in (("all rows", d0), ("without C5-BTBP", d0[d0.publication != C5])):
            print(f"\n--- {lab}: cells {d.cell_id.nunique()}, pubs {d.publication.nunique()}, clusters "
                  f"{d.cluster_cpg.nunique()}, rows/seed {len(d) / 5:.0f}; B1 R2 {r2(d, 'B1'):.4f} ---")
            out = []
            for a in main_arms + [f"{k}_lam{lam}" for k in ("MG_tr", "MCO_tr") for lam in LAMS]:
                ip = infer(d, a)
                cp, clo, chi, G = cjk(d, a)
                out.append({"arm": a, "R2": r2(d, a), "rel": ip["rel"], "pub_lo": ip["lo"], "pub_hi": ip["hi"],
                            "up": ip["up"], "down": ip["down"], "p_sf": ip["p_sf"], "cl_lo": clo, "cl_hi": chi,
                            "Gcl": G})
            print(pd.DataFrame(out).round(4).to_string(index=False))
        print(f"\n--- {design}: single-metal priors (only metal m takes the other-publication shape) ---")
        out = []
        s9a = d0[d0.chemotype == "sc009"]
        for m in LANTHANIDES:
            o = {"metal": m}
            for k in ("MG_tr", "MCO_tr"):
                a = f"{k}_only{m}"
                o[f"{k}_all"] = rel(d0, a)
                o[f"{k}_noC5"] = rel(d0[d0.publication != C5], a)
                o[f"{k}_sc009"] = rel(s9a, a)
            out.append(o)
        print(pd.DataFrame(out).round(4).to_string(index=False))
        s9 = d0[(d0.chemotype == "sc009") & (d0.publication != C5)]
        pn = s9[s9.pair == "Pr/Nd"]
        print(f"\n--- {design}: M07 overlap, sc009 rows (no C5): cells {s9.cell_id.nunique()}, pubs "
              f"{s9.publication.nunique()}; Pr/Nd rows/seed {len(pn) / 5:.0f} ---")
        for a in ("MCO_tr_onlyNd", "MG_tr_onlyNd", "MCO_tr", "MCO_LM10_tr"):
            ip = infer(s9, a)
            print(f"  {a:16s} sc009 all pairs rel {ip['rel']:+.4f} [{ip['lo']:+.4f}, {ip['hi']:+.4f}] up/down "
                  f"{ip['up']}/{ip['down']} p_sf {ip['p_sf']:.3f} | Pr/Nd R2 {r2(pn, 'B1'):.3f} -> {r2(pn, a):.3f}"
                  f" rel {rel(pn, a):+.4f}")
        dn = d0[d0.publication != C5]
        for a in ("MCO_tr_onlyNd", "MG_tr_onlyNd"):
            ip = infer(dn, a)
            cp, clo, chi, G = cjk(dn, a)
            print(f"  {a:16s} ALL rows (no C5) rel {rel(dn, a):+.4f}; dR2 {r2(dn, a) - r2(dn, 'B1'):+.4f}; pub "
                  f"[{ip['lo']:+.4f}, {ip['hi']:+.4f}] up/down {ip['up']}/{ip['down']}; cluster-jk "
                  f"[{clo:+.4f}, {chi:+.4f}] G {G}")
    print(f"done {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
