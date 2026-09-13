"""Third pass: the remaining numbers that carry a modelling implication.

* pair RMSE at baseline and with the amplitude made exact
* sign-induced share of total absolute error, and the |y| band split
* oracle gain fitted per extractant and per cell
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = "D:/ml_separator_gh"
PRED = os.path.join(ROOT, "generations/gen13_separation/predictions/B_primary")
OUT = os.path.join(ROOT, "generations/gen13_separation/analysis/stage2/d4_amplitude_vs_shape/verify")

sys.path.insert(0, os.path.join(ROOT, "generations", "gen13_separation"))
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

LN_IDX = {m: i for i, m in enumerate(LANTHANIDES)}
RADIUS = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], float)
ARMS = ["C_DIRECT_ROW", "M_SELECTED", "M_PHYSICS_radius+radius_sq",
        "M_LOWRANK_K2", "X_ENS_DIRECT+LOWRANK_K2"]
GRID = np.round(np.arange(0.80, 2.0001, 0.05), 4)


def macro(seed, extr, absval):
    t = pd.DataFrame({"s": seed, "e": extr, "a": absval})
    return float(t.groupby(["s", "e"])["a"].mean().groupby("s").mean().mean())


rows = []
for arm in ARMS:
    df = pd.read_parquet(os.path.join(PRED, arm + ".parquet"))
    amp_exact = np.empty(len(df))
    df = df.sort_values(["split_seed", "cell_id"]).reset_index(drop=True)
    k = df["split_seed"].astype(str) + "|" + df["cell_id"]
    st = np.flatnonzero(np.r_[True, k.values[1:] != k.values[:-1]])
    en = np.r_[st[1:], len(k)]
    for s, e in zip(st, en):
        sub = df.iloc[s:e]
        metals = sorted(set(sub["A"]).union(sub["B"]), key=lambda x: LN_IDX[x])
        m = len(metals)
        pos = {mm: i for i, mm in enumerate(metals)}
        n = len(sub)
        P = np.zeros((n, m))
        P[np.arange(n), [pos[a] for a in sub["A"]]] = 1.0
        P[np.arange(n), [pos[b] for b in sub["B"]]] = -1.0
        r = RADIUS[[LN_IDX[mm] for mm in metals]]
        u = r - r.mean()
        u /= np.linalg.norm(u)
        yv = sub["y"].values.astype(float)
        pv = sub["prediction"].values.astype(float)
        a_err = float(((P.T @ pv / m) - (P.T @ yv / m)) @ u)
        amp_exact[s:e] = pv - a_err * (P @ u)

    y = df["y"].values
    p = df["prediction"].values
    seed, extr = df["split_seed"].values, df["extractant"].values

    rmse_base = float(np.sqrt(np.mean((p - y) ** 2)))
    rmse_amp = float(np.sqrt(np.mean((amp_exact - y) ** 2)))

    # sign-induced excess: 2*min(|p|,|y|) when the signs disagree
    disagree = np.sign(p) != np.sign(y)
    excess = np.where(disagree, 2 * np.minimum(np.abs(p), np.abs(y)), 0.0)
    sign_share = float(excess.sum() / np.abs(p - y).sum())

    small = np.abs(y) < 0.25
    big = np.abs(y) > 1.0
    ae = np.abs(p - y)
    small_flip = float(disagree[small].mean())
    small_err = float(ae[small].sum() / ae.sum())
    big_flip = float(disagree[big].mean())
    big_err = float(ae[big].sum() / ae.sum())

    # oracle gain per extractant and per cell
    base = macro(seed, extr, np.abs(p - y))
    t = pd.DataFrame({"s": seed, "e": extr, "c": df["cell_id"].values, "y": y, "p": p})
    for level, keys in [("extractant", ["s", "e"]), ("cell", ["s", "c"])]:
        errs = np.stack([np.abs(g * p - y) for g in GRID], axis=1)
        E = pd.DataFrame(errs, columns=[f"g{i}" for i in range(len(GRID))])
        E[keys] = t[keys].values
        gsum = E.groupby(keys, sort=False)[[f"g{i}" for i in range(len(GRID))]].sum()
        bestcol = gsum.values.argmin(axis=1)
        gmap = dict(zip(gsum.index, GRID[bestcol]))
        gsel = np.array([gmap[tuple(v)] for v in t[keys].values])
        orac = macro(seed, extr, np.abs(gsel * p - y))
        rows.append(dict(arm=arm, quantity=f"oracle_gain_per_{level}",
                         macro_mae_base=base, macro_mae_oracle=orac,
                         gain=base - orac, n_groups=len(gsum)))
        print(f"{arm} per-{level} oracle: {base:.4f} -> {orac:.4f} "
              f"(-{base-orac:.4f}) over {len(gsum)} groups")

    rows.append(dict(arm=arm, quantity="pair_rmse", macro_mae_base=rmse_base,
                     macro_mae_oracle=rmse_amp, gain=rmse_base - rmse_amp,
                     n_groups=len(df)))
    rows.append(dict(arm=arm, quantity="sign_share_of_abs_error",
                     macro_mae_base=sign_share, macro_mae_oracle=np.nan,
                     gain=np.nan, n_groups=len(df)))
    print(f"{arm}: pair RMSE {rmse_base:.4f} -> {rmse_amp:.4f} | sign share "
          f"{100*sign_share:.1f}% | |y|<0.25 flip {100*small_flip:.1f}% "
          f"err {100*small_err:.1f}% (n={small.sum()}) | |y|>1 flip "
          f"{100*big_flip:.1f}% err {100*big_err:.1f}% (n={big.sum()})")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "v_d4_extra.csv"), index=False)
print("\nwrote v_d4_extra.csv")
