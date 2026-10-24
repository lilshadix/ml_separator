#!/usr/bin/env python
"""Skeptic A for idea lab-2, second pass.  EXPLORATORY (already-seen cells, 5 dev seeds); prints only.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/lab2_refseries2_skA.py

Reuses the independent machinery of lab2_refseries_skA.py (PWL on the radius axis, quadratic-detrended donor shapes,
w = n / (n + 6.1)).  BPG only.
  1. Raw near-copies (|dY| <= 0.03 on >= 4 shared metals) found by pass 1:
       37b0cdc532627e1d DMDPDGA/DCE (5a68dc5665)  == own TODGA/DCE fadca46ff2b140c2 (donor of the OWN arm)
       4fcbabe856178b53 'TDdDGA' 0.5 M (d68ed8c8cf) == TODGA 0.5 M of af0c74798c / bef16e560c (same JAEA group)
       259e68bb986bb484 TDDGA == 26cbed1ed70886fe TDdDGA (b9dfc468cd), both C3 targets, bit-identical, unflagged
     -> OWN / FP / CONS with these targets dropped, with the near-copy donor dropped, and with the duplicate dropped.
  2. Share of the pooled gain per publication (who carries the +5.3 % / +4.4 %).
  3. Single own-TODGA donor arms: which donor (condition) carries the gain in 4cba27f5f3 and 5a68dc5665.
  4. The hypothesis's estimator FP on its own terms: equal-weight publication mean, median, and the 3 % bar.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2\scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402
from r3 import datav2 as V2  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES  # noqa: E402
from lab2_refseries_skA import devs, shp, pwl, sq, r2, rel, unit, REF  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
NC_T_OWN = "37b0cdc532627e1d"
NC_DONOR = "fadca46ff2b140c2"
TODGA_COPY_T = "4fcbabe856178b53"
DUP_T = "26cbed1ed70886fe"   # bit-identical to 259e68bb986bb484


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    pub = f.publication_id.astype(str).to_numpy()
    nm = f.extractant_name.astype(str).to_numpy()
    acid = f["cond__acid_concentration_M"].to_numpy(float)
    dil = f[[c for c in f.columns if c.startswith("cond__diluent__")]].fillna(0)
    dname = dil.idxmax(axis=1).str.replace("cond__diluent__", "").to_numpy()
    v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    ok = np.array([c not in v2 for c in ids])
    Dv = devs(Y)
    has = np.isfinite(Dv).any(1) & ok
    ar = np.arange(len(f))
    tod = has & (nm == REF)
    tod_pubs = sorted(set(pub[tod]))
    P = pd.read_parquet(paths.PRED / "REF_BPG.parquet")
    c3 = sorted(set(P.cell_id))
    targets = [t for t in c3 if t not in v2 and nm[pos[t]] != REF and pub[pos[t]] in tod_pubs
               and (tod & (pub == pub[pos[t]]) & (ar != pos[t])).any()]
    fitted = json.loads((paths.RES / "REF" / "fitted_BPG.json").read_text())
    rows = []
    for seed in paths.DEV_SEEDS:
        for fold in range(5):
            rec = C.load("BPG", seed, fold)
            u = np.zeros(N_LN)
            u[CO.GD] = float(fitted[f"{seed}_{fold}"]["ref:B1"]["coef"][0])
            prow = P[(P.split_seed == seed) & (P.fold == fold)]
            pcg = {c: g for c, g in prow.groupby("cell_id")}
            for cell in rec["outer"]["cells"]:
                t = cell["cell_id"]
                if t not in pcg or t not in targets:
                    continue
                i = pos[t]
                yi = Y[i]
                metals = list(cell["metals"])
                reads = sorted({m for p in cell["seqs"]["DOPT_Q"][:3] for m in p})
                L = {"B1": pwl(yi, reads, metals, u)}
                own = np.where(tod & (pub == pub[i]) & (ar != i))[0]
                co, wo = shp(Dv, own)
                L["OWN"] = pwl(yi, reads, metals, (1 - wo) * u + wo * co)
                own2 = np.array([j for j in own if ids[j] != NC_DONOR], int)
                c2, w2 = shp(Dv, own2)
                L["OWN_dropNCdonor"] = pwl(yi, reads, metals, (1 - w2) * u + w2 * c2)
                shapes = [shp(Dv, np.where(tod & (pub == p))[0])[0] for p in tod_pubs if p != pub[i]]
                cons = np.mean(shapes, 0)
                L["CONS"] = pwl(yi, reads, metals, (1 - wo) * u + wo * cons)
                L["FP"] = pwl(yi, reads, metals, u + wo * (co - cons))
                singles = {}
                for j in own:
                    cj, wj = shp(Dv, [j])
                    lab = f"{dname[j][:10]}|{acid[j]:.3g}M|{'LaEu' if np.isfinite(Y[j, 0]) else 'GdLu'}"
                    singles[lab] = pwl(yi, reads, metals, (1 - wj) * u + wj * cj)
                for r in pcg[t].itertuples(index=False):
                    ia, ib = LANTHANIDES.index(r.A), LANTHANIDES.index(r.B)
                    o = {"split_seed": seed, "cell_id": t, "publication": pub[i], "name": nm[i], "pair": f"{r.A}/{r.B}",
                         "y": r.y, "tcond": f"{dname[i][:10]}|{acid[i]:.3g}M"}
                    for k, lv in L.items():
                        o[k] = lv[ia] - lv[ib]
                    for k, lv in singles.items():
                        o["S:" + k] = lv[ia] - lv[ib]
                    rows.append(o)
    T = pd.DataFrame(rows)
    print(f"targets {T.cell_id.nunique()} cells / {T.publication.nunique()} pubs; B1 R2 {r2(T, 'B1'):.4f}")

    # 2. share of pooled gain per publication
    print("\n=== share of the pooled gain per publication (BPG; per-seed SSE units) ===")
    g = T.assign(B=sq(T, "B1") / 5, O=sq(T, "OWN") / 5, F=sq(T, "FP") / 5).groupby("publication")[["B", "O", "F"]].sum()
    g["gain_OWN"] = g.B - g.O
    g["gain_FP"] = g.B - g.F
    g["share_OWN"] = g.gain_OWN / g.gain_OWN.sum()
    g["share_FP"] = g.gain_FP / g.gain_FP.sum()
    g["share_B1_SSE"] = g.B / g.B.sum()
    print(g.round(4).to_string())

    # 1. near-copy / duplicate sensitivity
    print("\n=== near-copy / duplicate sensitivity (BPG) ===")
    out = []
    for title, d, arm_own in (
            ("as reported (46 cells)", T, "OWN"),
            ("drop the near-copy DONOR (TODGA/DCE) from the own pool", T, "OWN_dropNCdonor"),
            ("drop target DMDPDGA/DCE (copy of own TODGA)", T[T.cell_id != NC_T_OWN], "OWN"),
            ("drop target 'TDdDGA' 0.5 M (copy of a TODGA series)", T[T.cell_id != TODGA_COPY_T], "OWN"),
            ("drop duplicate target TDdDGA (== TDDGA)", T[T.cell_id != DUP_T], "OWN"),
            ("drop all three", T[~T.cell_id.isin([NC_T_OWN, TODGA_COPY_T, DUP_T])], "OWN")):
        for a in (arm_own, "FP", "CONS"):
            u_ = unit(d, a)
            out.append({"population": title, "arm": a, "cells": d.cell_id.nunique(), "pooled": round(u_["pooled"], 4),
                        "eqw_mean": round(u_["eqw_mean"], 4), "median": round(u_["median"], 4),
                        "up/down": f"{u_['up']}/{u_['down']}", "per": u_["per"]})
    print(pd.DataFrame(out).to_string(index=False))
    s = T[T.cell_id == NC_T_OWN]
    print(f"\nDMDPDGA/DCE target alone: B1 SSE/seed {sq(s, 'B1').sum() / 5:.4f}, OWN gain/seed "
          f"{(sq(s, 'B1') - sq(s, 'OWN')).sum() / 5:.4f}, OWN without its near-copy donor gain/seed "
          f"{(sq(s, 'B1') - sq(s, 'OWN_dropNCdonor')).sum() / 5:.4f}")
    p5 = T[T.publication == "pub_5a68dc5665"]
    print(f"pub_5a68dc5665 total: B1 SSE/seed {sq(p5, 'B1').sum() / 5:.4f}; OWN gain {(sq(p5, 'B1') - sq(p5, 'OWN')).sum() / 5:.4f}; "
          f"OWN without the near-copy donor {(sq(p5, 'B1') - sq(p5, 'OWN_dropNCdonor')).sum() / 5:.4f}; rel "
          f"{rel(p5, 'OWN'):+.3f} -> {rel(p5, 'OWN_dropNCdonor'):+.3f}")

    # 3. single-donor arms
    print("\n=== single own-TODGA donor arms (rel SSE vs B1 on the targets each donor can serve) ===")
    for p in ("pub_4cba27f5f3", "pub_5a68dc5665", "pub_d68ed8c8cf"):
        d = T[T.publication == p]
        cols = [c for c in d.columns if c.startswith("S:") and d[c].notna().any()]
        out = []
        for c in cols:
            s = d[d[c].notna()]
            out.append({"donor": c[2:], "cells": s.cell_id.nunique(), "rel": round(rel(s, c), 4),
                        "OWN_on_same_rows": round(rel(s, "OWN"), 4)})
        print(f"-- {p}: target conditions {sorted(set(d.tcond))}")
        print(pd.DataFrame(out).to_string(index=False))
        if p == "pub_4cba27f5f3":
            print(d.groupby(["cell_id", "name"]).apply(lambda s: pd.Series({c[2:]: round(rel(s, c), 3) for c in cols}
                                                                         | {"OWN": round(rel(s, "OWN"), 3),
                                                                            "B1_SSE": round(sq(s, "B1").sum() / 5, 4)})).to_string())

    # 4. FP on its own terms
    print("\n=== hypothesis estimator FP = B1 + w (own TODGA - consensus) ===")
    for a in ("FP", "OWN"):
        u_ = unit(T, a)
        print(f"  {a}: pooled {u_['pooled']:+.4f}  equal-weight mean {u_['eqw_mean']:+.4f}  median {u_['median']:+.4f}  "
              f"up/down {u_['up']}/{u_['down']}  per {u_['per']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
