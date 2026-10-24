#!/usr/bin/env python
"""Skeptic A for idea confirm-4 (C-ECT), step 3: independent recompute of the E-CT quick check.

EXPLORATORY: every number is on the 229 already-seen C3 cells and the 5 dev seeds (rounds 1-4, P1, P2 and the confirm
theme have all seen them).  Nothing here is confirmatory.

Independent of P1_ct_build.py / confirm_ect_replay.py:
  - siblings re-derived from the V1 bench (same publication_id, same extractant SMILES, finite V1 log D at every metal of
    the target, neither cell in V2-1/V2-2/V2-3, recorded cond__ fields differ in something other than contact time or
    metal concentration);
  - PWL_R re-implemented with np.interp on -r(CN8) (end values held); B1_k = PWL[y_S - u_S] + u, u = beta_fold at Gd;
  - SIB_k = mean_j (Y_j + PWL[y_S - Y_j,S]); SH_k = w SIB_k + (1 - w) B1_k, w = n / (n + 1.78); COPY = mean_j Y_j.
  Only the read sequences (gen21 fold cache, DOPT_Q) and the fold betas (round-4 fitted_<design>.json) are shared inputs.
Checks: B1 at 4 reads vs the REF parquet's B1; SH/COPY vs the confirm theme's P1 rows.
Analyses (BPG primary, BP sensitivity):
  A  unit tables: the quick-check statistic (log of seed-mean summed SSE per unit) and the test design's statistic
     (mean over targets of per-target log SSE ratio), pubchem and pubext units, several strata;
  B  leave-one-publication-out and drop-top-k;
  C  cheapest sensible alternatives on identical rows: raw SIB, nearest-condition sibling (P1 column), B0;
  D  RMSE of COPY vs B1 in the "trivial-looking" (diluent / extractant conc / temperature) families;
  E  near-copy-sibling sensitivity: SH rebuilt without siblings whose dlogD is constant across metals (sd < 0.03).
Writes ideas2/scripts/confirm4_skA_out/{rows.parquet, units.csv, lopo.csv} and stdout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
R4DIR = HERE.parents[3] / "round4_20261022"
sys.path.insert(0, str(R4DIR))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import interp  # noqa: E402  (only the constant X_R axis is used)
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = HERE.parent / "confirm4_skA_out"
P1 = HERE.parent / "confirm_out" / "p1" / "P1_rows.parquet"
K = 1.78
C5 = "pub_6a0ef8528c"
TRIV = ("cond__contact_time_min", "cond__metal_concentration_mM")
LN = list(D.LANTHANIDES)
GD = LN.index("Gd")
X = np.asarray(interp.X_R, float)
NEAR_SD = 0.03


def pwl(vals: np.ndarray, reads: list[int]) -> np.ndarray:
    r = sorted(reads, key=lambda m: X[m])
    out = np.interp(X, X[r], vals[r])
    out[r] = vals[r]
    return out


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    ext = f.extractant.astype(str).to_numpy()
    drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    cond = [c for c in f.columns if c.startswith("cond__") and c not in TRIV]
    Cm = f[cond].astype(float).to_numpy()

    def differs(i, j) -> bool:
        x, y = Cm[i], Cm[j]
        both = np.isnan(x) & np.isnan(y)
        one = np.isnan(x) ^ np.isnan(y)
        neq = ~both & ~one & ~np.isclose(np.nan_to_num(x), np.nan_to_num(y))
        return bool(one.any() or neq.any())

    refs = {d: pd.read_parquet(paths.PRED / f"REF_{d}.parquet") for d in ("BPG", "BP")}
    c3 = sorted(set(refs["BPG"].cell_id))
    sibs, nearcopy = {}, {}
    for t in c3:
        i = pos[t]
        met = np.where(np.isfinite(Y[i]))[0]
        cover = np.all(np.isfinite(Y[:, met]), axis=1)
        cover[i] = False
        cand = np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]
        cl = [int(j) for j in cand if t not in drop and ids[j] not in drop and differs(i, j)]
        sibs[t] = cl
        nearcopy[t] = [j for j in cl if np.std(Y[j, met] - Y[i, met]) < NEAR_SD]

    rows = []
    chk = {"b1_maxabs": 0.0, "y_maxabs": 0.0, "readset_mismatch": 0}
    for design, par in refs.items():
        fitted = json.loads((paths.RES / "REF" / f"fitted_{design}.json").read_text())
        for seed in paths.DEV_SEEDS:
            for fold in range(5):
                rec = C.load(design, seed, fold)
                beta = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
                u = np.zeros(len(LN))
                u[GD] = beta
                prow = par[(par.split_seed == seed) & (par.fold == fold)]
                pc = dict(tuple(prow.groupby("cell_id")))
                for cell in rec["outer"]["cells"]:
                    t = cell["cell_id"]
                    if t not in pc or not sibs[t]:
                        continue
                    i = pos[t]
                    yi = Y[i]
                    seq = cell["seqs"]["DOPT_Q"]
                    S = sibs[t]
                    S_nc = [j for j in S if j not in nearcopy[t]]
                    lev = {}
                    for k in (2, 3):
                        rd = sorted({m for p in seq[:k] for m in p})
                        if k == 3:
                            rs = ",".join(LN[m] for m in rd)
                            if rs != str(pc[t].read_set.iloc[0]):
                                chk["readset_mismatch"] += 1
                        lev[("B1", k)] = pwl(yi - u, rd) + u
                        lev[("B0", k)] = pwl(yi, rd)
                        sl = np.array([Y[j] + pwl(yi - Y[j], rd) for j in S])
                        lev[("SIB", k)] = sl.mean(0)
                        if S_nc:
                            lev[("SIBNC", k)] = np.array([Y[j] + pwl(yi - Y[j], rd) for j in S_nc]).mean(0)
                    lev[("COPY", 0)] = Y[S].mean(0)
                    n = len(S)
                    w = n / (n + K)
                    nn = len(S_nc)
                    wnc = nn / (nn + K) if nn else 0.0
                    for r in pc[t].itertuples(index=False):
                        ia, ib = LN.index(r.A), LN.index(r.B)
                        sf = {key: v[ia] - v[ib] for key, v in lev.items()}
                        y = yi[ia] - yi[ib]
                        chk["y_maxabs"] = max(chk["y_maxabs"], abs(y - r.y))
                        chk["b1_maxabs"] = max(chk["b1_maxabs"], abs(sf[("B1", 3)] - r.B1))
                        row = {"design": design, "split_seed": seed, "fold": fold, "cell_id": t, "publication": pub[i],
                               "extractant": ext[i], "chemotype": str(f.chemotype.iat[i]), "A": r.A, "B": r.B,
                               "y": y, "B1_4": sf[("B1", 3)], "B1_3": sf[("B1", 2)], "B0_4": sf[("B0", 3)],
                               "SIB_4": sf[("SIB", 3)], "SIB_3": sf[("SIB", 2)], "COPY": sf[("COPY", 0)],
                               "SH_4": w * sf[("SIB", 3)] + (1 - w) * sf[("B1", 3)],
                               "SH_3": w * sf[("SIB", 2)] + (1 - w) * sf[("B1", 2)],
                               "n_sib": n, "n_nearcopy": len(nearcopy[t]), "c5": pub[i] == C5,
                               "SHNC_4": (wnc * sf[("SIBNC", 3)] + (1 - wnc) * sf[("B1", 3)]) if nn else sf[("B1", 3)]}
                        rows.append(row)
        print(f"{time.time() - t0:6.1f}s {design} built; {chk}", flush=True)
    t = pd.DataFrame(rows)
    t.to_parquet(OUT / "rows.parquet", index=False)

    # ---- check against the confirm theme's P1 rows
    p1 = pd.read_parquet(P1)
    key = ["design", "split_seed", "cell_id", "A", "B"]
    m = t.merge(p1[key + ["CL_SH_3", "CL_SH_2", "CL_COPY", "CL_NEAR_3", "n_clean", "cls", "t_v2drop"]], on=key,
                how="outer", indicator=True)
    print("merge:", m._merge.value_counts().to_dict())
    both = m[m._merge == "both"]
    bb = both[both.n_clean >= 1]
    print("P1 rows with n_clean>=1 missing here:", int(((m._merge == "right_only") & (m.n_clean >= 1)).sum()),
          "| here rows absent in P1 clean:", int(((m._merge == "left_only")).sum()))
    print("max |SH_4 - P1 CL_SH_3|:", float(np.nanmax(np.abs(bb.SH_4 - bb.CL_SH_3))),
          "max |SH_3 - P1 CL_SH_2|:", float(np.nanmax(np.abs(bb.SH_3 - bb.CL_SH_2))),
          "max |COPY - P1 CL_COPY|:", float(np.nanmax(np.abs(bb.COPY - bb.CL_COPY))),
          "n_sib == n_clean:", bool((bb.n_sib == bb.n_clean).all()))
    t = t.merge(p1[key + ["CL_NEAR_3", "cls"]], on=key, how="left")
    print("checks", chk)

    # ---- unit analyses
    def unit_table(tt, unit, cand, ref="B1_4"):
        g = tt.assign(er=(tt[ref] - tt.y) ** 2, ec=(tt[cand] - tt.y) ** 2)
        # quick-check statistic: per unit, seed-mean summed SSE, then log ratio
        a = g.groupby([unit, "split_seed"])[["er", "ec"]].sum().groupby(level=0).mean()
        Lq = np.log(a.er / a.ec)
        # design statistic: per target, seed-mean SSE, log ratio; then mean over the unit's targets
        b_ = g.groupby([unit, "cell_id", "split_seed"])[["er", "ec"]].sum().groupby(level=[0, 1]).mean()
        Lt = np.log(b_.er / b_.ec).replace([np.inf, -np.inf], np.nan)
        Ld = Lt.groupby(level=0).mean()
        return Lq, Ld, a

    def signflip_p(L):
        L = np.asarray(L, float)
        G = len(L)
        if G > 16:
            E = np.random.default_rng(1).choice((-1.0, 1.0), size=(65536, G))
        else:
            E = np.array(np.meshgrid(*[[1.0, -1.0]] * G)).reshape(G, -1).T
        return float(((E @ L) >= L.sum() - 1e-12).mean())

    out = []
    for design in ("BPG", "BP"):
        td = t[t.design == design].copy()
        td["pubchem"] = td.publication + "|" + td.chemotype
        td["pubext"] = td.publication + "|" + td.extractant
        strata = {"PRIMARY": (td.n_sib >= 2) & ~td.c5, "withC5": td.n_sib >= 2, "NSIB1": (td.n_sib == 1) & ~td.c5,
                  "PRIMARY_no_b0b3": (td.n_sib >= 2) & ~td.c5 & (td.publication != "pub_b0b3296ec5"),
                  "PRIMARY_no_b0b3_DMD": (td.n_sib >= 2) & ~td.c5 & ~td.publication.isin(["pub_b0b3296ec5",
                                                                                          "pub_8415538291"]),
                  "PRIMARY_no_5a68": (td.n_sib >= 2) & ~td.c5 & (td.publication != "pub_5a68dc5665"),
                  "PRIMARY_trivialish": (td.n_sib >= 2) & ~td.c5 & td.cls.isin(["diluent_only", "extr_conc_only",
                                                                                 "temperature_only"]),
                  "PRIMARY_acidconc": (td.n_sib >= 2) & ~td.c5 & (td.cls == "acid_conc_only")}
        for sname, mk in strata.items():
            tt = td[mk]
            for unit in ("pubchem", "pubext"):
                for cname, cand, ref in (("SH4-B1", "SH_4", "B1_4"), ("SH3-B1", "SH_3", "B1_4"),
                                         ("COPY-B1", "COPY", "B1_4"), ("SIB4-B1", "SIB_4", "B1_4"),
                                         ("SH4-SIB4", "SH_4", "SIB_4"), ("SH4-NEAR", "SH_4", "CL_NEAR_3"),
                                         ("SHNC4-B1", "SHNC_4", "B1_4"), ("B1-B0", "B1_4", "B0_4")):
                    ts = tt.dropna(subset=[cand, ref])
                    if ts.empty:
                        continue
                    Lq, Ld, a = unit_table(ts, unit, cand, ref)
                    out.append({"design": design, "stratum": sname, "unit": unit, "contrast": cname, "G": len(Lq),
                                "pubs": ts.publication.nunique(), "cells": ts.cell_id.nunique(),
                                "Wq": int((Lq > 1e-12).sum()), "Lq_": int((Lq < -1e-12).sum()),
                                "meanLq": float(Lq.mean()), "p_q": signflip_p(Lq.to_numpy()),
                                "Wd": int((Ld > 1e-12).sum()), "Ld_": int((Ld < -1e-12).sum()),
                                "meanLd": float(Ld.mean()), "sdLd": float(Ld.std(ddof=1)) if len(Ld) > 1 else np.nan,
                                "p_d": signflip_p(Ld.dropna().to_numpy()),
                                "pooled_rel": float(1 - a.ec.sum() / a.er.sum())})
    u = pd.DataFrame(out)
    u.to_csv(OUT / "units.csv", index=False)
    print(u.round(3).to_string())

    # ---- LOPO on PRIMARY (BPG), both statistics
    td = t[(t.design == "BPG") & (t.n_sib >= 2) & ~t.c5].copy()
    td["pubchem"] = td.publication + "|" + td.chemotype
    lo = []
    for p in ["(none)"] + sorted(td.publication.unique()):
        tt = td[td.publication != p]
        for cname, cand in (("SH4-B1", "SH_4"), ("SH3-B1", "SH_3")):
            Lq, Ld, a = unit_table(tt, "pubchem", cand)
            lo.append({"drop": p, "contrast": cname, "G": len(Lq), "Wq": int((Lq > 0).sum()), "meanLq": float(Lq.mean()),
                       "Wd": int((Ld > 0).sum()), "meanLd": float(Ld.mean()), "pooled_rel": float(1 - a.ec.sum() / a.er.sum())})
    lo = pd.DataFrame(lo)
    lo.to_csv(OUT / "lopo.csv", index=False)
    print(lo.round(3).to_string())

    # ---- per-unit table (BPG PRIMARY): both statistics, RMSE of B1 / SH / COPY
    g = td.assign(e_b1=(td.B1_4 - td.y) ** 2, e_sh=(td.SH_4 - td.y) ** 2, e_sh3=(td.SH_3 - td.y) ** 2,
                  e_cp=(td.COPY - td.y) ** 2, e_shnc=(td.SHNC_4 - td.y) ** 2)
    per = g.groupby("pubchem").agg(cells=("cell_id", "nunique"), rows=("y", "size"), rmse_b1=("e_b1", "mean"),
                                   rmse_sh=("e_sh", "mean"), rmse_sh3=("e_sh3", "mean"), rmse_copy=("e_cp", "mean"),
                                   rmse_shnc=("e_shnc", "mean"), ncopy=("n_nearcopy", "mean"), nsib=("n_sib", "mean"))
    for c in [c for c in per.columns if c.startswith("rmse")]:
        per[c] = np.sqrt(per[c])
    per["rows"] = per["rows"] / 5
    Lq, Ld, _ = unit_table(td, "pubchem", "SH_4")
    per["Lq_sh4"], per["Ld_sh4"] = Lq, Ld
    print(per.round(4).to_string())
    print("near-copy siblings (sd dlogD < %.2f) among PRIMARY targets: targets with >=1: %d of %d; share of sibling slots %.3f"
          % (NEAR_SD, td.groupby("cell_id").n_nearcopy.first().gt(0).sum(), td.cell_id.nunique(),
             td.groupby("cell_id").n_nearcopy.first().sum() / td.groupby("cell_id").n_sib.first().sum()))
    print("seconds", round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
