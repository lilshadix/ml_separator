"""Anchor sweep v2: cache the fold pair-tables once, then sweep (alpha, gamma) cheaply.

Guards against the obvious criticism of v1 -- that the ridge penalty was untuned, so a flat
gamma curve might only mean the base estimator was weak.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"D:\ml_separator_gh")
sys.path.insert(0, str(ROOT / "gen14_direction")); sys.path.insert(0, str(ROOT / "gen13_separation"))
from gen14 import dirbench as db
from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame, cell_weights
from gen13sep.metrics import per_extractant, summarise
from gen13sep.splits import all_folds

DESIGN = sys.argv[1] if len(sys.argv) > 1 else "BP"
FEAT = sys.argv[2] if len(sys.argv) > 2 else "LEAN209"
GAMMAS = [0.02, 0.1, 0.3, 0.6, 1.0, 2.0, 8.0, 1e6]
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
CACHE = ROOT / "gen16_anchor" / f"pairs_{DESIGN}.pkl"

bench = db.load(); FS = db.feature_sets(bench)
amp = bench.coef[:, 0]
pub_all = bench.frame["publication_id"].astype(str).to_numpy()
rich = bench.frame.n_metals.to_numpy() >= db.MIN_METALS

if CACHE.exists():
    folds_cached = pickle.loads(CACHE.read_bytes())
else:
    folds_cached = []
    for f in all_folds(bench.frame, design=DESIGN):
        pairs = _pair_frame(bench.frame, bench.Y, f.test_index, f.seed, f.fold)
        if pairs.empty: continue
        folds_cached.append({"tr": f.train_index, "te": f.test_index, "pairs": pairs})
    CACHE.write_bytes(pickle.dumps(folds_cached, protocol=5))
print(f"{len(folds_cached)} folds cached for design {DESIGN}", flush=True)


def anchor_fit(Xtr, ytr, w, pub, gamma, alpha):
    w = w / w.sum()
    idx = pd.Series(pub)
    sw = idx.map(pd.Series(w).groupby(pub).sum()).to_numpy()[:, None]
    Xbar = pd.DataFrame(Xtr * w[:, None]).groupby(pub).transform("sum").to_numpy() / sw
    ybar = (pd.Series(ytr * w).groupby(pub).transform("sum").to_numpy())[:, None] / sw
    g = np.sqrt(gamma)
    Xt = (Xtr - Xbar) + g * Xbar
    yt = ((ytr[:, None] - ybar) + g * ybar).ravel()
    mx = (Xt * w[:, None]).sum(0); my = float((yt * w).sum())
    Xc, yc = Xt - mx, yt - my
    sd = np.sqrt((Xc**2 * w[:, None]).sum(0)); sd[sd < 1e-9] = 1.0
    Xs = Xc / sd
    b = np.linalg.solve((Xs * w[:, None]).T @ Xs + alpha * np.eye(Xs.shape[1]) / len(Xs),
                        (Xs * w[:, None]).T @ yc) / sd
    return b, (Xtr * w[:, None]).sum(0), float((ytr * w).sum())


X_all = bench.matrix(LEAN_BLOCKS)[:, FS[FEAT]]
parts = []
t0 = time.time()
for fc in folds_cached:
    tr, te, pairs = fc["tr"], fc["te"], fc["pairs"]
    rtr = tr[rich[tr]]
    w = cell_weights(bench.groups[rtr], bench.n_obs[rtr])
    wa = cell_weights(bench.groups[tr], bench.n_obs[tr])
    bmean = float(np.average(bench.coef[tr, 1], weights=wa))
    amean = float(np.average(bench.coef[tr, 0], weights=wa))
    mag = float(np.average(np.abs(amp[rtr]), weights=w))
    Xtr = X_all[rtr]; med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(X_all[te]), med, X_all[te])
    curves = {"MEAN_CURVE": np.tile(np.array([amean, bmean]) @ bench.basis, (len(te), 1)),
              "DIR_ORACLE": np.c_[np.where(amp[te] < 0, -mag, mag), np.full(len(te), bmean)] @ bench.basis}
    for al in ALPHAS:
        for gm in GAMMAS:
            b, mx, my = anchor_fit(Xtr, amp[rtr], w, pub_all[rtr], gm, al)
            a_hat = my + (Xte - mx) @ b
            tag = f"a{al:g}_g{gm:g}"
            curves[tag] = np.c_[a_hat, np.full(len(te), bmean)] @ bench.basis
            curves[tag + "_S"] = np.c_[np.sign(a_hat) * mag, np.full(len(te), bmean)] @ bench.basis
    ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
    t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
    for nm, cv in curves.items(): t[nm] = cv[loc, ia] - cv[loc, ib]
    parts.append(t)

table = pd.concat(parts, ignore_index=True)
names = [c for c in table.columns if "_g" in c or c in ("MEAN_CURVE", "DIR_ORACLE")]
board = summarise(per_extractant(table, names), table, names)
board.insert(0, "design", DESIGN)
board.to_csv(ROOT / "gen16_anchor" / f"g16_anchor2_{DESIGN}_{FEAT}.csv", index=False)
sel = board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd", "macro_sign_acc_strong"]]
print(f"\n=== {DESIGN} / {FEAT}  ({time.time()-t0:.0f}s) === full arms (predicted magnitude)")
print(sel[~sel.arm.str.endswith("_S")].round(4).to_string(index=False))
print(f"\n=== {DESIGN} / {FEAT} === sign-only arms (constant magnitude)")
print(sel[sel.arm.str.endswith("_S") | sel.arm.isin(["MEAN_CURVE","DIR_ORACLE"])].round(4).to_string(index=False))
