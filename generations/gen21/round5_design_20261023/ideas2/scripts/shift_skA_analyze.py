#!/usr/bin/env python
"""SKEPTIC A for shift-2, analysis of shift_skA_recompute.py's rows (EXPLORATORY, already-seen 229 C3 cells).
Prints only.

    .venv/Scripts/python.exe -W ignore generations/gen21/round5_design_20261023/ideas2/scripts/shift_skA_analyze.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as SS

HERE = Path(__file__).resolve().parent
OUT = HERE / "shift_skA_out"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def r2s(d, c):
    return np.array([1 - ((g.y - g[c]) ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum() for _, g in d.groupby("split_seed")])


def r2(d, c):
    return float(r2s(d, c).mean())


def relsse(d, c, r):
    return float(np.mean([1 - ((g.y - g[c]) ** 2).sum() / ((g.y - g[r]) ** 2).sum() for _, g in d.groupby("split_seed")]))


def unitc(d, c, r):
    sst = {s: ((g.y - g.y.mean()) ** 2).sum() for s, g in d.groupby("split_seed")}
    ns = d.split_seed.nunique()
    e = ((d.y - d[r]) ** 2 - (d.y - d[c]) ** 2) / d.split_seed.map(sst) / ns
    sr = ((d.y - d[r]) ** 2).groupby(d.publication).sum()
    sc = ((d.y - d[c]) ** 2).groupby(d.publication).sum()
    return pd.DataFrame({"c": e.groupby(d.publication).sum(), "rho": 1 - sc / sr})


def signflip(c):
    c = np.asarray(c, float)
    S = np.array(list(itertools.product((1.0, -1.0), repeat=len(c))))
    return float(((S @ c) >= c.sum() - 1e-15).mean())


def contrast(d, c, r, lab):
    u = unitc(d, c, r)
    pt = r2(d, c) - r2(d, r)
    pubs = sorted(d.publication.unique())
    jk = np.array([r2(d[d.publication != p], c) - r2(d[d.publication != p], r) for p in pubs])
    P = len(jk)
    se = np.sqrt((P - 1) / P * ((jk - jk.mean()) ** 2).sum()) if P > 2 else np.nan
    q = SS.t.ppf(0.975, P - 1) if P > 2 else np.nan
    seeds = int(((r2s(d, c) - r2s(d, r)) > 0).sum())
    nz = u[np.abs(u.c) > 1e-9]
    print(f"  [{lab}] {c} - {r}: dR2 {pt:+.4f} relSSE {relsse(d, c, r):+.3f} seeds+ {seeds}/5 | pubs + {int((u.c > 1e-9).sum())}"
          f" / - {int((u.c < -1e-9).sum())} / 0 {int((np.abs(u.c) <= 1e-9).sum())} of {len(u)}; sign-flip p "
          f"{signflip(nz.c) if len(nz) else np.nan:.3f}; LOPO [{pt - q * se:+.3f}, {pt + q * se:+.3f}] min {jk.min():+.4f} "
          f"(w/o {pubs[int(np.argmin(jk))]}); eq-w pub relSSE {u.rho.mean():+.3f}")
    return u


def main():
    R = pd.read_parquet(OUT / "skA_rows.parquet")
    TI = pd.read_csv(OUT / "skA_targets.csv")
    R = R.merge(TI[["cell", "reg_ok", "inside", "lam", "lam_a", "lam_c", "n_AS", "nmet", "x_t", "x_min", "x_max"]],
                left_on="cell_id", right_on="cell", how="left")
    S0 = pd.read_parquet(HERE / "shift_out" / "shift_rows.parquet")

    # ------------------------------------------------ reproduction against the idea's parquet
    key = ["design", "split_seed", "cell_id", "A", "B"]
    m = R.merge(S0, on=key, suffixes=("", "_o"))
    print(f"merged rows {len(m)} of mine {len(R)} / theirs {len(S0)}")
    for c in ("ACOPY_0", "AREG_0", "AREG_3", "AINT_0", "ACOPYSH_3", "AREGSH_3", "SH_3", "B1_3", "B1_1"):
        mm = m[m[c].notna() | m[c + "_o"].notna()]
        print(f"  {c}: max|mine - theirs| {np.nanmax(np.abs(mm[c] - mm[c + '_o'])):.2e}; NaN mismatch "
              f"{int((mm[c].isna() != mm[c + '_o'].isna()).sum())}")

    for design in ("BPG", "BP"):
        t = R[R.design == design]
        A2 = t[t.ACOPY_0.notna() & (t.reg_ok == True) & ~t.c5]                       # noqa: E712
        A2c = t[t.ACOPY_0.notna() & (t.reg_ok == True)]                              # noqa: E712
        print("\n" + "=" * 100 + f"\n{design}: A2 cells {A2.cell_id.nunique()} pubs {A2.publication.nunique()} "
              f"rows/seed {len(A2) / 5:.0f}; A2 with C5 cells {A2c.cell_id.nunique()}")
        lad = []
        for arm in ("B1", "SH", "ACOPY", "AREG", "AEB", "AEBG", "AEBGa", "AEBGc", "AINT"):
            rr = {"arm": arm}
            for k in (0, 1, 2, 3):
                if f"{arm}_{k}" in A2:
                    rr[f"{k + 1 if k else 0}r"] = r2(A2, f"{arm}_{k}")
                if f"{arm}SH_{k}" in A2:
                    rr[f"{k + 1}r_SH"] = r2(A2, f"{arm}SH_{k}")
            lad.append(rr)
        print(pd.DataFrame(lad).set_index("arm").round(4).to_string())
        print(" frozen-arm contrasts on A2 (no C5):")
        for c, r in (("AEBG_0", "ACOPY_0"), ("AEB_0", "ACOPY_0"), ("AEBGa_0", "ACOPY_0"), ("AEBGc_0", "ACOPY_0"),
                     ("AEBG_1", "ACOPY_1"), ("AEBG_1", "B1_3"), ("ACOPY_1", "B1_3"), ("AEBG_1", "SH_1"),
                     ("AEBGSH_3", "SH_3"), ("ACOPYSH_3", "SH_3"), ("AEBGSH_3", "ACOPYSH_3"), ("AEBSH_3", "SH_3")):
            contrast(A2, c, r, "A2")
        print(" with C5:")
        for c, r in (("AEBG_0", "ACOPY_0"), ("AEBGSH_3", "SH_3")):
            contrast(A2c, c, r, "A2+C5")
        # inside-range rows: cheapest sensible physics-free alternative is AINT (bracketing interpolation)
        ins = A2[A2.inside == True]                                                    # noqa: E712
        print(f"\n inside-range rows: cells {ins.cell_id.nunique()} pubs {ins.publication.nunique()}; "
              f"ACOPY_0 {r2(ins, 'ACOPY_0'):.3f} AREG_0 {r2(ins, 'AREG_0'):.3f} AEBG_0 {r2(ins, 'AEBG_0'):.3f} "
              f"AINT_0 {r2(ins, 'AINT_0'):.3f} B1_3 {r2(ins, 'B1_3'):.3f} SH_3 {r2(ins, 'SH_3'):.3f}")
        for c, r in (("AREG_0", "ACOPY_0"), ("AEBG_0", "ACOPY_0"), ("AEBG_0", "AINT_0"), ("AREG_0", "AINT_0"),
                     ("AINT_0", "ACOPY_0"), ("AREG_0", "B1_3"), ("ACOPY_0", "B1_3")):
            contrast(ins, c, r, "inside")
        # pre-declared strata (exploratory here)
        for nm, q in (("lam>=0.2", A2[A2.lam >= 0.2]), ("lam<0.2", A2[A2.lam < 0.2]),
                      ("lam>=0.2 & inside", ins[ins.lam >= 0.2]), ("lam>0 & inside", ins[ins.lam > 0])):
            print(f"  stratum {nm}: cells {q.cell_id.nunique()} pubs {q.publication.nunique()} "
                  f"{sorted(q.drop_duplicates('cell_id').publication.value_counts().to_dict().items())}")
            if q.publication.nunique() >= 3:
                contrast(q, "AEBG_0", "ACOPY_0", nm)
            elif len(q):
                print(f"    AEBG_0 {r2(q, 'AEBG_0'):.3f} ACOPY_0 {r2(q, 'ACOPY_0'):.3f}")
        # per-publication table for the primary contrast
        u = unitc(A2, "AEBG_0", "ACOPY_0")
        cells = A2.drop_duplicates("cell_id")
        u["cells"] = cells.groupby("publication").size()
        u["cells_lam>0_inside"] = cells[(cells.lam > 0) & (cells.inside == True)].groupby("publication").size()  # noqa
        u["mean_lam_inside"] = cells[cells.inside == True].groupby("publication").lam.mean()               # noqa
        print("\n per-publication contribution, AEBG_0 - ACOPY_0 (R2 units):")
        print(u.round(4).fillna(0).to_string())
        # in-corpus kill-criterion check
        nzu = u[np.abs(u.c) > 1e-9]
        print(f"  kill check: pooled relSSE {relsse(A2, 'AEBG_0', 'ACOPY_0'):+.4f}; families improved "
              f"{int((u.c > 1e-9).sum())}/{len(u)} all = {(u.c > 1e-9).mean():.2f}; among moved {int((nzu.c > 0).sum())}/"
              f"{len(nzu)}; w/o pub_19783c039b pooled relSSE "
              f"{relsse(A2[A2.publication != 'pub_19783c039b'], 'AEBG_0', 'ACOPY_0'):+.4f}")
        ni = relsse(A2, "AEBGSH_3", "SH_3")
        print(f"  non-inferiority AEBGSH@4 vs SH@4 relSSE {ni:+.4f} (margin -0.01) -> {'FAIL' if ni < -0.01 else 'pass'}")
        # power: units needed for a one-sided sign-flip / t at 80 % for the observed per-unit effect
        cu = u.c.to_numpy()
        if cu.std(ddof=1) > 0:
            dz = cu.mean() / cu.std(ddof=1)
            print(f"  per-unit effect dz {dz:.3f} -> units for 80 % power (one-sided 0.05) ~ {((1.645 + 0.84) / dz) ** 2:.0f}")


if __name__ == "__main__":
    main()
