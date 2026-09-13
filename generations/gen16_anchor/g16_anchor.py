"""Anchor regression with publication identity as the anchor.

Rothenhausler, Meinshausen, Buhlmann & Peters, "Anchor regression: heterogeneous data meet
causality", JRSS-B 83(2):215-246, 2021, doi:10.1111/rssb.12398 (arXiv:1801.06229):

    b(gamma) = argmin_b || (I - Pi_A)(y - Xb) ||^2 / n  +  gamma * || Pi_A (y - Xb) ||^2 / n

    gamma = 0    adjusting for A  (partial out A: here the WITHIN-publication estimator)
    gamma = 1    ordinary least squares (the existing 209-column arm)
    gamma = inf  two-stage least squares (the BETWEEN-publication / IV estimator)

A = disjoint publication dummies, so Pi_A is the (weighted) per-publication mean operator and the
whole estimator is OLS/ridge on   Xt = (I - Pi_A)X + sqrt(gamma) Pi_A X,  yt likewise.

Note the sweep runs on BOTH sides of gamma=1.  Large gamma moves toward 2SLS, which explains y
using only the between-publication variation of X -- exactly where the 94%-accurate publication
fingerprint lives.  Small gamma moves toward the within-publication estimator, which is the one
that prices the laboratory effect out.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"D:\ml_separator_gh")
sys.path.insert(0, str(ROOT / "generations" / "gen14_direction")); sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
from gen14 import dirbench as db
from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame, cell_weights
from gen13sep.metrics import per_extractant, summarise
from gen13sep.splits import all_folds

GAMMAS = [0.0, 0.05, 0.15, 0.35, 0.6, 1.0, 2.0, 4.0, 16.0, 1e6]
ALPHA = 30.0            # fixed across every gamma so gamma is the only thing moving
DESIGNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BP"]
FEATS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["LEAN209"]


def anchor_fit(Xtr, ytr, w, pub, gamma, alpha):
    """Weighted anchor ridge.  Returns (coef, x_mean, y_mean) for prediction on a NEW publication."""
    w = w / w.sum()
    # weighted per-publication means = Pi_A in the w-inner product
    dfp = pd.DataFrame(Xtr); dfp["_w"] = w; dfp["_p"] = pub
    sw = dfp.groupby("_p")["_w"].transform("sum").to_numpy()[:, None]
    Xbar = (dfp[dfp.columns[:-2]].mul(w, axis=0).groupby(pub).transform("sum").to_numpy()) / sw
    ybar = (pd.Series(ytr * w).groupby(pub).transform("sum").to_numpy())[:, None] / sw
    g = np.sqrt(gamma)
    Xt = (Xtr - Xbar) + g * Xbar
    yt = (ytr[:, None] - ybar) + g * ybar
    yt = yt.ravel()
    # weighted grand means of the TRANSFORMED data (the ridge is fit on centred transformed data)
    mx = (Xt * w[:, None]).sum(0); my = float((yt * w).sum())
    Xc = Xt - mx; yc = yt - my
    sd = np.sqrt((Xc**2 * w[:, None]).sum(0)); sd[sd < 1e-9] = 1.0
    Xs = Xc / sd
    A = (Xs * w[:, None]).T @ Xs + alpha * np.eye(Xs.shape[1]) / len(Xs)
    b = np.linalg.solve(A, (Xs * w[:, None]).T @ yc) / sd
    # prediction level: anchor at the UNtransformed weighted grand means (a new publication has
    # no fitted effect, so its offset is the corpus average offset)
    mx0 = (Xtr * w[:, None]).sum(0); my0 = float((ytr * w).sum())
    return b, mx0, my0


def run(design, featname):
    bench = db.load(); FS = db.feature_sets(bench)
    cols = FS[featname]
    X_all = bench.matrix(LEAN_BLOCKS)[:, cols]
    amp = bench.coef[:, 0]
    pub_all = bench.frame["publication_id"].astype(str).to_numpy()
    rich = bench.frame.n_metals.to_numpy() >= db.MIN_METALS
    parts, diag = [], []
    for f in all_folds(bench.frame, design=design):
        tr, te = f.train_index, f.test_index
        pairs = _pair_frame(bench.frame, bench.Y, te, f.seed, f.fold)
        if pairs.empty: continue
        rtr = tr[rich[tr]]
        w = cell_weights(bench.groups[rtr], bench.n_obs[rtr])
        wa = cell_weights(bench.groups[tr], bench.n_obs[tr])
        bmean = float(np.average(bench.coef[tr, 1], weights=wa))
        amean = float(np.average(bench.coef[tr, 0], weights=wa))
        mag_const = float(np.average(np.abs(amp[rtr]), weights=w))
        Xtr = X_all[rtr]; ytr = amp[rtr]; pub = pub_all[rtr]
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xte = np.where(np.isnan(X_all[te]), med, X_all[te])
        sz = pd.Series(pub).map(pd.Series(pub).value_counts())
        diag.append({"seed": f.seed, "fold": f.fold, "n_train_rich": len(rtr),
                     "n_pub": pd.Series(pub).nunique(), "within_df": int(len(rtr) - pd.Series(pub).nunique()),
                     "cells_in_multi_pub": int((sz >= 2).sum())})
        curves = {}
        for gm in GAMMAS:
            b, mx, my = anchor_fit(Xtr, ytr, w, pub, gm, ALPHA)
            a_hat = my + (Xte - mx) @ b
            tag = f"ANCH_g{gm:g}"
            curves[tag] = np.c_[a_hat, np.full(len(te), bmean)] @ bench.basis
            # sign-only variant: the magnitude is known dead, so use the constant
            curves[tag + "_SIGN"] = np.c_[np.sign(a_hat) * mag_const,
                                          np.full(len(te), bmean)] @ bench.basis
        curves["MEAN_CURVE"] = np.tile(np.array([amean, bmean]) @ bench.basis, (len(te), 1))
        curves["DIR_ORACLE"] = np.c_[np.where(amp[te] < 0, -mag_const, mag_const),
                                     np.full(len(te), bmean)] @ bench.basis
        ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
        t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
        for nm, cv in curves.items(): t[nm] = cv[loc, ia] - cv[loc, ib]
        parts.append(t)
    table = pd.concat(parts, ignore_index=True)
    names = [c for c in table.columns if c.startswith(("ANCH_", "MEAN_", "DIR_"))]
    board = summarise(per_extractant(table, names), table, names)
    board.insert(0, "design", design); board.insert(1, "features", featname)
    print(f"\n### design {design}  features {featname}  alpha={ALPHA}")
    print(pd.DataFrame(diag).describe().loc[["mean", "min", "max"]].round(1).to_string())
    print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd",
                 "macro_mae_chemotype", "macro_sign_acc_strong"]].round(4).to_string(index=False))
    return board


if __name__ == "__main__":
    t0 = time.time(); out = []
    for d in DESIGNS:
        for fn in FEATS: out.append(run(d, fn))
    pd.concat(out).to_csv(ROOT / "generations" / "gen16_anchor" / f"g16_anchor_{'_'.join(DESIGNS)}.csv", index=False)
    print(f"\n[{time.time()-t0:.0f}s]")
