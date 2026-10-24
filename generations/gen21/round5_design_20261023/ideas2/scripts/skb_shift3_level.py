#!/usr/bin/env python
"""Skeptic B on idea shift-3 (EXPLORATORY; all bench cells already seen).

Question: is the light-end acid steepening (Ce/Pr, La/Ce, Pr/Nd slopes d(logSF)/d(log10 acid) < 0) a light-end
(metal-identity) property, or the light-end projection of a level effect in which a metal's log D acid slope tracks
its own log D level within the group (series fans out with acid; compresses near the top)?

Grouping copied from shift_slopes.py (publication x extractant x acid type x diluent x additive x [L] x T; V2-1/2/3
cells dropped; >= 2 cells spanning >= 0.1 decade of acid).  Per group with >= 10 metal slopes and La, Ce, Pr, Nd present:
  b_m = OLS slope of log D_m on log10[acid]; L_m = mean log D_m over the group's cells.
  Level model fitted on the NON-light metals (Nd..Lu, i.e. excluding La, Ce, Pr):  b = a + c*L   (linear)
  and b = a + c*L + d*L^2 (quadratic, allows saturation turn-over).  Predict La, Ce, Pr from their own levels.
  Observed Ce/Pr pair slope = b_Ce - b_Pr;  level-predicted = bhat_Ce - bhat_Pr;  residual = observed - predicted.
Aggregation: equal weight per publication; exact two-sided sign-flip over publications.
Strata: all; 'clean' = drops C5-BTBP (floor), pub_1605e436d7 (DTPA/complexant, V2-3 hidden merge), organic-acid media.
Nothing is written except stdout.
"""
from __future__ import annotations

import itertools
import sys

sys.path.insert(0, r"D:\ml_separator_gh\generations\gen21\round4_20261022")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from r4 import paths  # noqa: E402,F401
from r3 import datav2 as V2  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.data import LANTHANIDES as LN, N_LN  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)


def slope(x, y):
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) < 0.1 or len(np.unique(np.round(x, 3))) < 2:
        return np.nan
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def signflip(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0 or len(v) > 18:
        return np.nan
    S = np.array(list(itertools.product((1, -1), repeat=len(v))))
    return float((np.abs(S @ v) >= abs(v.sum()) - 1e-12).mean())


def main():
    b = D.load_bench()
    f = b.frame.reset_index(drop=True)
    Y = np.asarray(b.Y, float)
    v2 = set(V2.V2_1_CELLS) | set(V2.V2_2_DROP) | set(V2.V2_3_CELLS)

    def lab(r, pre):
        return "+".join(c[len(pre):] for c in f.columns if c.startswith(pre) and r[c] == 1)
    f["k"] = [(r.publication_id, r.extractant, lab(r, "cond__acid__"), lab(r, "cond__diluent__"),
               lab(r, "cond__additive__"), round(float(r.cond__extractant_concentration_M), 3),
               str(r.cond__temperature_C)) for _, r in f.iterrows()]
    f["x"] = np.log10(f.cond__acid_concentration_M.astype(float))
    keep = ~f.cell_id.isin(v2) & f.x.notna()
    rows = []
    for k, g in f[keep].groupby("k"):
        if len(g) < 2 or np.ptp(g.x) < 0.1:
            continue
        idx = g.index.to_numpy()
        x = g.x.to_numpy()
        for win in (False, True):
            Yg = Y[idx].copy()
            if win:
                Yg[(Yg < -2) | (Yg > 2.5)] = np.nan
            bm = np.array([slope(x, Yg[:, m]) for m in range(N_LN)])
            L = np.nanmean(Yg, 0)
            ok = np.isfinite(bm) & np.isfinite(L)
            if ok.sum() < 10 or not ok[[1, 2]].all():
                continue
            heavy = ok.copy()
            heavy[[0, 1, 2]] = False
            if heavy.sum() < 6:
                continue
            c1 = np.polyfit(L[heavy], bm[heavy], 1)
            c2 = np.polyfit(L[heavy], bm[heavy], 2) if heavy.sum() >= 8 else c1
            p1 = np.polyval(c1, L)
            p2 = np.polyval(c2, L)
            # rank correlation level vs slope over heavy metals only (no light metals involved)
            from scipy.stats import spearmanr
            rho_h = spearmanr(L[heavy], bm[heavy])[0]
            rows.append({"pub": k[0], "ext": str(g.extractant_name.iloc[0])[:12], "chem": g.chemotype.iloc[0],
                         "acid": k[2], "window": win, "nmet": int(ok.sum()), "nx": len(np.unique(np.round(x, 3))),
                         "span": float(np.ptp(x)), "c_level": c1[0], "rho_heavy": rho_h,
                         "L_La": L[0], "L_Ce": L[1], "L_Pr": L[2], "L_Lu": L[13],
                         "obs_CePr": bm[1] - bm[2], "pred1_CePr": p1[1] - p1[2], "pred2_CePr": p2[1] - p2[2],
                         "obs_LaCe": bm[0] - bm[1] if ok[0] else np.nan,
                         "pred1_LaCe": p1[0] - p1[1] if ok[0] else np.nan,
                         "res1_light": float(np.nanmean((bm - p1)[[0, 1, 2]])),
                         "res2_light": float(np.nanmean((bm - p2)[[0, 1, 2]]))})
    R = pd.DataFrame(rows)
    R["res1_CePr"] = R.obs_CePr - R.pred1_CePr
    R["res2_CePr"] = R.obs_CePr - R.pred2_CePr
    bad = (R.chem == "sc061") | (R.pub == "pub_1605e436d7") | ~R.acid.isin(["hno3", "hcl", "h2so4"])
    R["clean"] = ~bad
    print("=== per group (level model fitted on Nd..Lu only) ===")
    print(R.round(3).to_string(index=False))
    for win in (False, True):
        for lab_, Q in (("all", R), ("clean (no C5-BTBP, no DTPA pub, mineral acid only)", R[R.clean])):
            Q = Q[Q.window == win]
            P = Q.groupby("pub")[["obs_CePr", "pred1_CePr", "res1_CePr", "res2_CePr", "res1_light", "res2_light",
                                   "c_level", "rho_heavy"]].mean()
            if P.empty:
                continue
            print(f"\n--- window={win} | {lab_}: {len(Q)} groups, {len(P)} publications ---")
            for col in P.columns:
                v = P[col].to_numpy()
                print(f"{col:12s} mean {np.nanmean(v):+.3f}  median {np.nanmedian(v):+.3f}  neg {int((v < 0).sum())}/"
                      f"{int(np.isfinite(v).sum())}  pos {int((v > 0).sum())}  signflip p2 {signflip(v):.3f}")


if __name__ == "__main__":
    main()
