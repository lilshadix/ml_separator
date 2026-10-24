#!/usr/bin/env python
"""shift-2 Skeptic B quick check (EXPLORATORY: the 229 already-seen C3 cells, 5 dev seeds; reads only
ideas2/scripts/shift_out/shift_rows.parquet and the V1 bench; writes nothing except stdout).

Questions:
 1. How many A2 targets actually get lambda_EB > 0 (and >= 0.2), inside the siblings' acid range, by publication?
    (lambda_EB = 0 when n_AS < 3, so AEBG == ACOPY there and the primary contrast is an exact tie.)
 2. Inside the range: is AEBG better than the cheapest local same-information comparator, AINT (per-metal
    piecewise-linear interpolation between the bracketing sibling levels), not only ACOPY?
 3. Gate test as an interaction: is the UNSHRUNK AREG - ACOPY contrast better in the lambda >= 0.2 stratum than in
    the lambda < 0.2 stratum? (The idea's stratum test on AEBG scales mechanically with lambda.)
 4. Does label-free lambda_EB beat a constant lambda (0.25) inside the range?

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift2_skepticB_check.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import N_LN  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sa", HERE / "shift_analyze.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)
OUT = sa.OUT
K = 1.78
FAMS_TRIVIAL = {"contact_time", "metal_conc"}
BIG = "pub_19783c039b"


def lambdas():
    bench = D.load_bench()
    f = bench.frame.reset_index(drop=True)
    Y = np.asarray(bench.Y, float)
    ids = f.cell_id.astype(str).tolist()
    pos = {c: i for i, c in enumerate(ids)}
    v2drop = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)
    cond_cols = [c for c in f.columns if c.startswith("cond__")]
    x = np.log10(f.cond__acid_concentration_M.astype(float).to_numpy())
    pub, ext = f.publication_id.astype(str).to_numpy(), f.extractant.astype(str).to_numpy()

    def fam(c):
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

    def diff(i, j):
        s = set()
        for c in cond_cols:
            a, b = f.at[i, c], f.at[j, c]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) or pd.isna(b) or not np.isclose(float(a), float(b)):
                s.add(fam(c))
        return s - FAMS_TRIVIAL

    return f, Y, ids, pos, v2drop, x, pub, ext, diff


def main():
    T = pd.read_parquet(OUT / "shift_rows.parquet")
    f, Y, ids, pos, v2drop, x, pub, ext, diff = lambdas()
    info = {}
    for t in sorted(T[T.n_AS >= 1].cell_id.unique()):
        i = pos[t]
        met = [m for m in range(N_LN) if np.isfinite(Y[i, m])]
        cover = np.all(np.isfinite(Y[:, met]), axis=1)
        cover[i] = False
        AS = [j for j in np.where((pub == pub[i]) & (ext == ext[i]) & cover)[0]
              if t not in v2drop and ids[j] not in v2drop and diff(i, j) == {"acid_conc"}]
        xa = x[AS]
        n = len(AS)
        dof = n - 2
        nlev = len(np.unique(np.round(xa, 3)))
        if dof < 1 or np.ptp(xa) < 0.1:
            info[t] = (0.0, n, nlev, np.nan, np.nan)
            continue
        Sxx = ((xa - xa.mean()) ** 2).sum()
        b = np.array([((xa - xa.mean()) * (Y[AS, m] - Y[AS, m].mean())).sum() / Sxx for m in met])
        s2 = np.array([(((Y[AS, m] - Y[AS, m].mean()) - bb * (xa - xa.mean())) ** 2).sum() / dof
                       for m, bb in zip(met, b)])
        dev = b - b.mean()
        noise = (s2 / Sxx).mean() * (1 - 1 / len(met))
        v = (dev ** 2).mean()
        lam = float(np.clip(1 - noise / v, 0, 1)) if v > 0 else 0.0
        info[t] = (lam, n, nlev, float(np.sqrt(v)), float(np.sqrt(noise)))

    for design in ("BPG",):
        t = T[(T.design == design) & (T.n_AS >= 1)].copy()
        t["lam"] = t.cell_id.map(lambda c: info[c][0])
        t["nlev"] = t.cell_id.map(lambda c: info[c][2])
        t["lamG"] = np.where(t.extrap == True, 0.0, t.lam)                 # noqa: E712
        w = t.n_AS / (t.n_AS + K)
        for k in (0, 1, 2, 3):
            t[f"AEBG_{k}"] = t[f"ACOPY_{k}"] + t.lamG * (t[f"AREG_{k}"] - t[f"ACOPY_{k}"])
            t[f"AL25_{k}"] = t[f"ACOPY_{k}"] + 0.25 * (t[f"AREG_{k}"] - t[f"ACOPY_{k}"])
            if k:
                for nm in ("AEBG", "AINT", "ACOPY", "AL25"):
                    t[f"{nm}SHx_{k}"] = w * t[f"{nm}_{k}"] + (1 - w) * t[f"B1_{k}"]
        A2 = t[(t.reg_ok == True) & ~t.c5]                                  # noqa: E712
        A2c = t[(t.reg_ok == True)]                                         # noqa: E712
        print(f"A2: {sa.describe(A2)}")
        cells = A2.drop_duplicates("cell_id")[["cell_id", "publication", "chemotype", "n_AS", "nlev", "lam",
                                              "extrap"]]
        print("\n[1] lambda_EB by publication (A2 cells): cells / inside-range / inside & n_AS>=3 / inside & lam>0 /"
              " inside & lam>=0.2")
        g = cells.assign(inside=~cells.extrap.astype(bool))
        tab = g.groupby("publication").apply(lambda z: pd.Series({
            "cells": len(z), "inside": int(z.inside.sum()), "in_nAS>=3": int((z.inside & (z.n_AS >= 3)).sum()),
            "in_lam>0": int((z.inside & (z.lam > 0)).sum()), "in_lam>=0.2": int((z.inside & (z.lam >= 0.2)).sum()),
            "lam_mean": z.lam.mean()}))
        print(tab.to_string())
        print("totals:", tab.sum(numeric_only=True).to_dict())
        print("chemotypes of inside & lam>=0.2:", g[g.inside & (g.lam >= 0.2)].groupby(["publication", "chemotype"]).size().to_dict())

        IN = A2[A2.extrap == False]                                         # noqa: E712
        print(f"\n[2] inside the range: {sa.describe(IN)}")
        for k, lab in ((0, "0 reads"), (1, "2 reads"), (3, "4 reads")):
            arms = ["ACOPY", "AINT", "AREG", "AEBG", "AL25"]
            s = "; ".join(f"{a} {sa.r2(IN, f'{a}_{k}'):.4f}" for a in arms)
            if k:
                s += "; SH " + f"{sa.r2(IN, f'SH_{k}'):.4f}" + "; B1 " + f"{sa.r2(IN, f'B1_{k}'):.4f}"
                s += "; " + "; ".join(f"{a}SH {sa.r2(IN, f'{a}SHx_{k}'):.4f}" for a in ("AEBG", "AINT", "ACOPY"))
            print(f"   {lab}: {s}")
        for cand, ref in (("AEBG_0", "ACOPY_0"), ("AEBG_0", "AINT_0"), ("AINT_0", "ACOPY_0"), ("AREG_0", "AINT_0"),
                          ("AEBG_0", "AL25_0"), ("AEBG_1", "B1_3"), ("AINT_1", "B1_3"), ("AEBGSHx_3", "SH_3"),
                          ("AINTSHx_3", "SH_3")):
            sa.infer(IN, cand, ref, "inside")
        INnb = IN[IN.publication != BIG]
        print(f"\n   inside, without {BIG}: {sa.describe(INnb)}")
        for cand, ref in (("AEBG_0", "ACOPY_0"), ("AEBG_0", "AINT_0"), ("AINT_0", "ACOPY_0")):
            sa.infer(INnb, cand, ref, "inside-noBIG")

        print("\n[3] gate as interaction (inside range): unshrunk AREG_0 - ACOPY_0 by lambda stratum")
        for nm, d in (("lam>=0.2", IN[IN.lam >= 0.2]), ("0<lam<0.2", IN[(IN.lam > 0) & (IN.lam < 0.2)]),
                      ("lam==0 (n_AS>=3)", IN[(IN.lam == 0) & (IN.n_AS >= 3)]), ("n_AS==2", IN[IN.n_AS == 2]),
                      ("lam<0.2 all", IN[IN.lam < 0.2])):
            if len(d) == 0:
                print(f"   {nm}: empty")
                continue
            print(f"   {nm}: {sa.describe(d)}; ACOPY {sa.r2(d, 'ACOPY_0'):.4f} AINT {sa.r2(d, 'AINT_0'):.4f} "
                  f"AREG {sa.r2(d, 'AREG_0'):.4f} AEBG {sa.r2(d, 'AEBG_0'):.4f}; relSSE AREG vs ACOPY "
                  f"{sa.relsse(d, 'AREG_0', 'ACOPY_0'):+.3f}; AEBG vs ACOPY {sa.relsse(d, 'AEBG_0', 'ACOPY_0'):+.3f}; "
                  f"AINT vs ACOPY {sa.relsse(d, 'AINT_0', 'ACOPY_0'):+.3f}")
            if d.publication.nunique() >= 2:
                sa.infer(d, "AREG_0", "ACOPY_0", nm)

        print("\n[4] with C5 (A2 inside, incl. C5):")
        INc = A2c[A2c.extrap == False]                                      # noqa: E712
        print(f"   {sa.describe(INc)}")
        for cand, ref in (("AEBG_0", "ACOPY_0"), ("AEBG_0", "AINT_0")):
            sa.infer(INc, cand, ref, "inside+C5")

        print("\n[5] per-cell lambda list inside range with lam>0:")
        print(g[g.inside & (g.lam > 0)].sort_values("lam", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
