"""Degrees-of-freedom check on the amplitude-oracle bound of d2_completion_and_amplitude.py.

A cell with m observed metals has only m-1 free numbers in its residual curve, so
fitting 1 or 2 per-cell scalars is close to saturating a small cell (m=3: 2 dof, 2
scalars -> exactly zero residual).  Re-run the oracle stratified by m, and add an
honest held-out version: fit the per-cell amplitude on all pairs EXCEPT the one being
scored (leave-one-pair-out inside the cell), which cannot saturate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/ml_separator_gh")
sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
from gen13sep.metals import LANTHANIDES, ATOMIC_NUMBER

OUT = ROOT / "generations/gen13_separation/analysis/stage2/d2_metal_dependency"
M = list(LANTHANIDES); NM = len(M); IDX = {m: i for i, m in enumerate(M)}
Z = np.array([ATOMIC_NUMBER[m] for m in M], float)
log = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


df = pd.read_parquet(ROOT / "generations/gen13_separation/manifests/cohort_exact.parquet")
Yraw = df[["logD__" + m for m in M]].to_numpy(float)
full = (~np.isnan(Yraw)).sum(axis=1) == NM
Yf = Yraw[full]
Cf = Yf - Yf.mean(axis=1, keepdims=True)
S = np.cov(Cf, rowvar=False, ddof=1)
w, Vv = np.linalg.eigh(S)
o = np.argsort(w)[::-1]
V1, V2 = Vv[:, o[0]], Vv[:, o[1]]
if np.dot(V1, Z - Z.mean()) < 0:
    V1 = -V1
cellrow = {c: i for i, c in enumerate(df["cell_id"])}

allrows = []
for arm in ["C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2", "B1_MEAN_CURVE"]:
    p = pd.read_parquet(ROOT / f"generations/gen13_separation/predictions/B_primary/{arm}.parquet")
    seed = int(p.split_seed.min())
    p = p[p.split_seed == seed].copy()
    p["resid"] = p.y - p.prediction
    recs = []
    for cid, g in p.groupby("cell_id", sort=False):
        mets = sorted(set(g.A) | set(g.B), key=lambda m: IDX[m])
        m_ = len(mets)
        if m_ < 3:
            continue
        loc = {mm: t for t, mm in enumerate(mets)}
        D = np.zeros((len(g), m_))
        D[np.arange(len(g)), [loc[a] for a in g.A]] = 1
        D[np.arange(len(g)), [loc[b] for b in g.B]] = -1
        u, *_ = np.linalg.lstsq(D, g.resid.to_numpy(), rcond=None)
        u -= u.mean()
        ii = [IDX[mm] for mm in mets]
        B1 = np.column_stack([V1[ii] - V1[ii].mean()])
        B2 = np.column_stack([V1[ii] - V1[ii].mean(), V2[ii] - V2[ii].mean()])
        r_full = D @ u
        e1 = D @ (u - B1 @ np.linalg.lstsq(B1, u, rcond=None)[0])
        e2 = D @ (u - B2 @ np.linalg.lstsq(B2, u, rcond=None)[0])
        # honest: leave the scored pair out of the amplitude fit
        h1 = np.empty(len(g)); h2 = np.empty(len(g))
        DB1, DB2 = D @ B1, D @ B2
        for t in range(len(g)):
            k = np.ones(len(g), bool); k[t] = False
            a1, *_ = np.linalg.lstsq(DB1[k], r_full[k], rcond=None)
            h1[t] = r_full[t] - DB1[t] @ a1
            if np.linalg.matrix_rank(DB2[k]) == 2:
                a2, *_ = np.linalg.lstsq(DB2[k], r_full[k], rcond=None)
                h2[t] = r_full[t] - DB2[t] @ a2
            else:
                h2[t] = np.nan
        recs.append((df.extractant.iat[cellrow[cid]], m_, np.abs(r_full).mean(),
                     np.abs(e1).mean(), np.abs(e2).mean(),
                     np.abs(h1).mean(), float(np.nanmean(np.abs(h2)))))
    r = pd.DataFrame(recs, columns=["extractant", "n_metals", "mae", "oracle1", "oracle2",
                                    "loo1", "loo2"])
    for lo, hi, lab in [(3, 5, "3-5"), (6, 9, "6-9"), (10, 14, "10-14"), (3, 14, "all>=3"),
                        (8, 14, ">=8")]:
        s = r[(r.n_metals >= lo) & (r.n_metals <= hi)]
        if not len(s):
            continue
        mac = s.groupby("extractant")[["mae", "oracle1", "oracle2", "loo1", "loo2"]].mean().mean()
        allrows.append(dict(arm=arm, n_metals_band=lab, n_cells=len(s),
                            n_extractants=s.extractant.nunique(), macro_mae=mac.mae,
                            oracle_1scalar=mac.oracle1, oracle_2scalar=mac.oracle2,
                            heldout_1scalar=mac.loo1, heldout_2scalar=mac.loo2,
                            frac_removed_oracle1=1 - mac.oracle1 / mac.mae,
                            frac_removed_heldout1=1 - mac.loo1 / mac.mae))
        say(f"[8] {arm} m={lab} ({len(s)} cells, {s.extractant.nunique()} extractants): "
            f"macro MAE {mac.mae:.3f} | oracle 1-scalar {mac.oracle1:.3f} "
            f"2-scalar {mac.oracle2:.3f} | leave-the-scored-pair-out 1-scalar {mac.loo1:.3f} "
            f"2-scalar {mac.loo2:.3f}")
    del p, r
pd.DataFrame(allrows).to_csv(OUT / "d2_amplitude_dof_check.csv", index=False,
                             float_format="%.5g")
(OUT / "d2_stdout_part8.txt").write_text("\n".join(log), encoding="utf-8")
print("\nDONE")
