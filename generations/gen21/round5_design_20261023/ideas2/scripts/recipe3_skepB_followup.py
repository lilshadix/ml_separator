#!/usr/bin/env python
"""Skeptic B follow-up for recipe-3.  EXPLORATORY (229 already-seen C3 cells).  Reads the V1 bench, the recipe_out
parquets and this skeptic's own recipe3_skepB_out outputs; writes only under ideas2/recipe3_skepB_out/.

 F1  near-replicate sign agreement, relaxed: same publication + extractant, recorded cond__ differs only in contact
     time / metal concentration / nothing, V2 lists NOT excluded, but byte-identical series dropped.
 F2  the worst publication (pub_a49146c8fd): what extractant and which condition changes flip the sign.
 F3  the triage built on recipe-1's option instead of COPY: 2 HULL reads + SH_own, call where |SH@2| >= 0.1,
     re-measure the rest: metals needed and accuracy, against COPY triage.

    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/recipe3_skepB_followup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r4.refs import HIST_D7_BETA  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import corrector as CO  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN, LANTHANIDES, Z_VEC  # noqa: E402

BASE = Path(r"D:\ml_separator_gh\generations\gen21\round5_design_20261023\ideas2")
RIN = BASE / "recipe_out"
OUT = BASE / "recipe3_skepB_out"
TRIVIAL = {"contact_time", "metal_conc"}
FLAG_PAIRS = {"Er/Tm", "Ho/Er", "La/Ce"}
K_SH = 1.78
X = interp.X_R
GD = CO.GD


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


def main() -> int:
    R = {}
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f["cell_id"].astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    cond_cols = [c for c in f.columns if c.startswith("cond__")]
    F = f[cond_cols].to_numpy(dtype=float)
    pubA = f["publication_id"].astype(str).to_numpy()
    extA = f["extractant"].astype(str).to_numpy()
    cells = pd.read_parquet(RIN / "recipe_cells.parquet")
    c3 = set(cells.cell_id)

    # ---- F1 relaxed near-replicates over the whole bench (any cell, not only C3) ----
    rep = []
    seen = set()
    for i in range(len(ids)):
        for j in np.where((pubA == pubA[i]) & (extA == extA[i]))[0]:
            if j <= i:
                continue
            dfm = set()
            for q, c in enumerate(cond_cols):
                a, b = F[i, q], F[j, q]
                if np.isnan(a) and np.isnan(b):
                    continue
                if np.isnan(a) or np.isnan(b) or not np.isclose(a, b):
                    dfm.add(fam(c))
            if dfm - TRIVIAL:
                continue
            both = np.isfinite(Y[i]) & np.isfinite(Y[j])
            if both.sum() < 2 or np.allclose(Y[i, both], Y[j, both], atol=1e-6):
                continue
            for p in range(N_LN - 1):
                if Z_VEC[p + 1] - Z_VEC[p] != 1:
                    continue
                yi, yj = Y[i, p] - Y[i, p + 1], Y[j, p] - Y[j, p + 1]
                if np.isfinite(yi) and np.isfinite(yj):
                    rep.append({"i": ids[i], "j": ids[j], "pub": pubA[i], "c3": ids[i] in c3 or ids[j] in c3,
                                "pair": f"{LANTHANIDES[p]}/{LANTHANIDES[p + 1]}", "yi": yi, "yj": yj,
                                "dif": ",".join(sorted(dfm)) or "none"})
    rep = pd.DataFrame(rep)
    if len(rep):
        d = rep.yi - rep.yj
        out = {"n_rows": int(len(rep)), "n_pubs": int(rep.pub.nunique()), "n_series_pairs": int(rep[["i", "j"]].drop_duplicates().shape[0]),
               "sd_dSF": float(d.std()), "mad_dSF_x1.4826": float(1.4826 * np.median(np.abs(d - d.median()))),
               "dif": rep.dif.value_counts().to_dict(), "pubs": rep.pub.value_counts().to_dict()}
        for lo in (0.1, 0.2):
            m = (rep.yi.abs() >= lo) | (rep.yj.abs() >= lo)
            # symmetric: agreement of sign when at least one of the two exceeds lo
            out[f"sign_agree_either>={lo}"] = float((np.sign(rep.yi[m]) == np.sign(rep.yj[m])).mean())
            out[f"n_either>={lo}"] = int(m.sum())
            mb = (rep.yi.abs() >= lo) & (rep.yj.abs() >= lo)
            out[f"sign_agree_both>={lo}"] = float((np.sign(rep.yi[mb]) == np.sign(rep.yj[mb])).mean()) if mb.sum() else None
            out[f"n_both>={lo}"] = int(mb.sum())
        R["F1_near_replicates"] = out
        rep.to_csv(OUT / "F1_near_replicates.csv", index=False)
    else:
        R["F1_near_replicates"] = {"n_rows": 0}
    print("F1", json.dumps(R["F1_near_replicates"], indent=1, default=str), flush=True)

    # ---- F2 worst publication ----
    wpub = "pub_a49146c8fd"
    wc = cells[cells.publication == wpub]
    sub = f[f.publication_id.astype(str) == wpub]
    keep = [c for c in cond_cols if sub[c].notna().any()]
    info = {"cells_C3": wc[["cell_id", "extractant", "chemotype", "n_own", "own_fams", "metals"]].to_dict("records"),
            "conds": sub[["cell_id"] + keep].to_dict("records")}
    pr = pd.read_parquet(RIN / "recipe_pairs.parquet")
    pr = pr[(pr.n_read == 0) & pr.cell_id.isin(wc.cell_id)][["cell_id", "pair", "y", "pred_SIB_own"]]
    info["pairs"] = pr.round(3).to_dict("records")
    R["F2_worst_pub"] = info
    print("F2 worst pub", wpub, "extractant(s):", wc.extractant.unique().tolist(), "chemotype:",
          wc.chemotype.unique().tolist(), flush=True)
    print(sub[["cell_id"] + keep].to_string(), flush=True)
    print(pr.round(3).to_string(index=False), flush=True)

    # ---- F3 triage from SH_own@2 (recipe-1 option) vs COPY triage ----
    P = pd.read_parquet(RIN / "recipe_pairs.parquet")
    P = P[P.n_read == 0].merge(cells[["ci", "c5", "n_own", "own_ids"]], on="ci")
    P = P[(P.n_own >= 1) & ~P.c5].reset_index(drop=True)
    res = []
    for t, g in P.groupby("cell_id"):
        i = pos[t]
        Mi = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        m = len(Mi)
        x = X[Mi]
        yv = Y[i, Mi]
        u = np.zeros(m)
        if GD in Mi:
            u[Mi.index(GD)] = HIST_D7_BETA
        own = json.loads(g.own_ids.iloc[0])
        T = Y[np.ix_([pos[j] for j in own], Mi)].mean(axis=0)
        w = len(own) / (len(own) + K_SH)
        rd = [0, m - 1]
        lev = u + np.interp(x, x[rd], (yv - u)[rd])
        levS = T + np.interp(x, x[rd], (yv - T)[rd])
        name = {LANTHANIDES[Mi[q]]: q for q in range(m)}
        reads = {LANTHANIDES[Mi[0]], LANTHANIDES[Mi[-1]]}
        callsC, callsS = [], []
        metC, metS = set(), set(reads)
        for _, r in g.iterrows():
            a, b = r.pair.split("/")
            sh = w * (levS[name[a]] - levS[name[b]]) + (1 - w) * (lev[name[a]] - lev[name[b]])
            copy = r.pred_SIB_own
            fc = abs(copy) < 0.1 or r.pair in FLAG_PAIRS
            fs = abs(sh) < 0.1 or r.pair in FLAG_PAIRS
            if fc:
                metC |= {a, b}
            if fs:
                metS |= {a, b}
            res.append({"cell": t, "pair": r.pair, "y": r.y, "cp": copy, "sh2": sh, "flagC": fc, "flagS": fs})
        res[-1]["_metC"] = len(metC)
        res[-1]["_metS"] = len(metS)
        res[-1]["_m"] = m
    df = pd.DataFrame(res)
    big = df.y.abs() >= 0.1
    agg = df.dropna(subset=["_m"])
    out = {"COPY_triage_metals": float(agg._metC.mean()), "SH2_triage_metals": float(agg._metS.mean()),
           "metals_per_cell": float(agg._m.mean()),
           "COPY_triage_called_share": float((~df.flagC).mean()), "SH2_triage_called_share": float((~df.flagS).mean()),
           "COPY_triage_acc_called_|y|>=0.1": float((np.sign(df.cp) == np.sign(df.y))[big & ~df.flagC].mean()),
           "SH2_triage_acc_called_|y|>=0.1": float((np.sign(df.sh2) == np.sign(df.y))[big & ~df.flagS].mean()),
           "COPY_triage_acc_called_all": float((np.sign(df.cp) == np.sign(df.y))[~df.flagC].mean()),
           "SH2_triage_acc_called_all": float((np.sign(df.sh2) == np.sign(df.y))[~df.flagS].mean()),
           "share_separable_left_uncalled_COPY": float(df.flagC[big].mean()),
           "share_separable_left_uncalled_SH2": float(df.flagS[big].mean()),
           "share_hard_0.1-0.2_uncalled_COPY": float(df.flagC[(df.y.abs() >= 0.1) & (df.y.abs() < 0.2)].mean()),
           "n_hard_0.1-0.2": int(((df.y.abs() >= 0.1) & (df.y.abs() < 0.2)).sum())}
    R["F3_triage_compare"] = out
    print("F3", json.dumps(out, indent=1), flush=True)
    (OUT / "recipe3_skepB_followup.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
