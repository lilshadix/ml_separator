"""The one anchor arm that could actually move the metric: anchor the MAGNITUDE.

Gen14 established (a) the direction is exhausted -- a PERFECT direction call scores 0.486 under BP
against 0.500 for the deployed one -- and (b) seven priors for |a| all land within 0.007 of a
constant.  But every one of those seven priors was a function of the LIGAND.  The headroom that is
left lives between 0.486 (oracle direction, constant magnitude) and 0.322 (oracle both).

The within/between decomposition says the mass-action block is the only block in the corpus whose
amplitude R^2 is LARGER within publication (0.084) than between (0.042) -- the signature of an
effect that pooling destroys rather than manufactures.  So: keep gen14's direction, and take the
magnitude from an anchor ridge at gamma < 1, i.e. identified off within-publication contrasts.

Arms: MAGANCH_<block>_g<gamma>, direction = gen14 L2 logistic on TOPO39, hard 0.5 rule.
Reference: G14_DIR_HARD (direction x constant magnitude, 0.500 BP) and DIR_ORACLE (0.486).
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"D:\ml_separator_gh")
sys.path.insert(0, str(ROOT / "generations" / "gen14_direction")); sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
from gen14 import dirbench as db
from gen14 import models as M
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights
from gen13sep.metrics import per_extractant, summarise

DESIGN = "BP"
GAMMAS = [0.0, 0.05, 0.2, 0.5, 1.0]
BLOCKS = ["MASSACT8", "COND64", "TOPO_MASSACT", "LEAN209"]
ALPHA = 10.0

bench = db.load(); FS = db.feature_sets(bench)
amp = bench.coef[:, 0]; pub_all = bench.frame["publication_id"].astype(str).to_numpy()
rich = bench.frame.n_metals.to_numpy() >= db.MIN_METALS
folds_cached = pickle.loads((ROOT / "generations" / "gen16_anchor" / f"pairs_{DESIGN}.pkl").read_bytes())
print(f"{len(folds_cached)} folds", flush=True)

# gen14's direction, out of fold
o = db.run(bench, "G14", M.candidate(M.dir_logistic(), M.amp_constant), features=FS["TOPO39"],
           design=DESIGN)
P = o.cells.set_index(["split_seed", "fold", "cell_index"])["p"]
print(f"direction OOF in {o.seconds:.1f}s", flush=True)


def anchor_fit(X, y, w, pub, gamma, alpha):
    w = w / w.sum()
    sw = pd.Series(pub).map(pd.Series(w).groupby(pub).sum()).to_numpy()[:, None]
    Xbar = pd.DataFrame(X * w[:, None]).groupby(pub).transform("sum").to_numpy() / sw
    ybar = (pd.Series(y * w).groupby(pub).transform("sum").to_numpy())[:, None] / sw
    g = np.sqrt(gamma)
    Xt = (X - Xbar) + g * Xbar; yt = ((y[:, None] - ybar) + g * ybar).ravel()
    mx = (Xt * w[:, None]).sum(0); my = float((yt * w).sum())
    Xc, yc = Xt - mx, yt - my
    sd = np.sqrt((Xc**2 * w[:, None]).sum(0)); sd[sd < 1e-9] = 1.0
    Xs = Xc / sd
    b = np.linalg.solve((Xs * w[:, None]).T @ Xs + alpha * np.eye(Xs.shape[1]) / len(Xs),
                        (Xs * w[:, None]).T @ yc)
    return b / sd, (X * w[:, None]).sum(0), float((y * w).sum())


X_lean = bench.matrix(LEAN_BLOCKS)
parts = []
t0 = time.time()
for fc in folds_cached:
    tr, te, pairs = fc["tr"], fc["te"], fc["pairs"]
    seed, fold = int(pairs.split_seed.iat[0]), int(pairs.fold.iat[0])
    rtr = tr[rich[tr]]
    w = cell_weights(bench.groups[rtr], bench.n_obs[rtr])
    wa = cell_weights(bench.groups[tr], bench.n_obs[tr])
    bmean = float(np.average(bench.coef[tr, 1], weights=wa))
    amean = float(np.average(bench.coef[tr, 0], weights=wa))
    mag = float(np.average(np.abs(amp[rtr]), weights=w))
    p = P.loc[(seed, fold)].reindex(te).to_numpy()
    sgn = np.where(p >= 0.5, -1.0, 1.0)
    curves = {"G14_DIR_HARD": np.c_[sgn * mag, np.full(len(te), bmean)] @ bench.basis,
              "MEAN_CURVE": np.tile(np.array([amean, bmean]) @ bench.basis, (len(te), 1)),
              "DIR_ORACLE": np.c_[np.where(amp[te] < 0, -mag, mag), np.full(len(te), bmean)] @ bench.basis}
    y_mag = np.abs(amp[rtr])
    for bl in BLOCKS:
        Xb = X_lean[:, FS[bl]]
        Xtr = Xb[rtr]; med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0.0, med)
        Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xb[te]), med, Xb[te])
        for gm in GAMMAS:
            bb, mx, my = anchor_fit(Xtr, y_mag, w, pub_all[rtr], gm, ALPHA)
            m_hat = np.clip(my + (Xte - mx) @ bb, 0.05, 3.0)
            curves[f"MAGANCH_{bl}_g{gm:g}"] = np.c_[sgn * m_hat, np.full(len(te), bmean)] @ bench.basis
    ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
    t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
    for nm, cv in curves.items(): t[nm] = cv[loc, ia] - cv[loc, ib]
    parts.append(t)

table = pd.concat(parts, ignore_index=True)
names = [c for c in table.columns if c.startswith(("MAGANCH_", "G14_", "MEAN_", "DIR_"))]
board = summarise(per_extractant(table, names), table, names)
board.to_csv(ROOT / "generations" / "gen16_anchor" / "g16_maganchor_BP.csv", index=False)
print(f"\n=== MAGNITUDE anchor, design BP ({time.time()-t0:.0f}s), alpha={ALPHA} ===")
print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd",
             "macro_mae_chemotype", "macro_sign_acc_strong"]].round(4).to_string(index=False))
