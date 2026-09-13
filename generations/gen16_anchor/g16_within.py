"""Is there ANY within-publication signal?  Decomposition + restricted wild cluster bootstrap.

The falsifier for the anchor lead.  Anchor regression at gamma < 1 can only help if the
within-publication regression of the amplitude on the features carries signal that the pooled
regression is distorting.  This script measures that directly, block by block, and tests the
condition block's within-publication contribution with the wild cluster bootstrap of

  Cameron, Gelbach & Miller, "Bootstrap-Based Improvements for Inference with Clustered Errors",
  Rev. Econ. Stat. 90(3):414-427, 2008, doi:10.1162/rest.90.3.414
  MacKinnon & Webb, "Wild Bootstrap Inference for Wildly Different Cluster Sizes",
  J. Appl. Econometrics 32(2):233-254, 2017, doi:10.1002/jae.2508

clustered on publication (G = 41), imposing the null (WCR) and using Webb 6-point weights.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"D:\ml_separator_gh")
sys.path.insert(0, str(ROOT / "generations" / "gen14_direction")); sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
from gen14 import dirbench as db
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
ALPHA = 30.0


def demean(M, g):
    return M - pd.DataFrame(M).groupby(g).transform("mean").to_numpy()


def ridge_fit(X, y, alpha):
    n = len(X)
    mx, my = X.mean(0), y.mean()
    Xc, yc = X - mx, y - my
    sd = Xc.std(0); sd[sd < 1e-9] = 1.0
    Xs = Xc / sd
    b = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ yc) / sd
    return b, my - mx @ b


def r2(y, yhat):
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((y - yhat) ** 2).sum() / ss if ss > 0 else 0.0


bench = db.load(); FS = db.feature_sets(bench)
rich = bench.frame.n_metals.to_numpy() >= 5
pub = bench.frame["publication_id"].astype(str).to_numpy()[rich]
amp = bench.coef[:, 0][rich]
X_all = bench.matrix(LEAN_BLOCKS)[rich]
med = np.nanmedian(X_all, 0); med = np.where(np.isnan(med), 0.0, med)
X_all = np.where(np.isnan(X_all), med, X_all)

print(f"n={len(amp)} rich cells, G={pd.Series(pub).nunique()} publications, "
      f"within df={len(amp)-pd.Series(pub).nunique()}\n")

print("=== in-sample R^2 of the amplitude, pooled vs within-publication vs between-publication ===")
print(f"{'block':28s} {'ncol':>5s} {'pooled':>8s} {'within':>8s} {'between':>9s}")
rows = []
for name in ["COND64", "MASSACT8", "TOPO39", "DONORS13", "PHYSCHEM10", "COORD114", "CHEM137", "LEAN209"]:
    Xb = X_all[:, FS[name]]
    b, c = ridge_fit(Xb, amp, ALPHA); pooled = r2(amp, Xb @ b + c)
    Xw, yw = demean(Xb, pub), amp - pd.Series(amp).groupby(pub).transform("mean").to_numpy()
    bw, cw = ridge_fit(Xw, yw, ALPHA); within = r2(yw, Xw @ bw + cw)
    d = pd.DataFrame(Xb); d["_y"] = amp; gm = d.groupby(pub).mean()
    Xm, ym = gm.drop(columns="_y").to_numpy(), gm["_y"].to_numpy()
    bb, cb = ridge_fit(Xm, ym, ALPHA); between = r2(ym, Xm @ bb + cb)
    print(f"{name:28s} {len(FS[name]):5d} {pooled:8.3f} {within:8.3f} {between:9.3f}")
    rows.append({"block": name, "ncol": len(FS[name]), "pooled_r2": pooled,
                 "within_r2": within, "between_r2": between})
pd.DataFrame(rows).to_csv(ROOT / "generations" / "gen16_anchor" / "g16_within_between_r2.csv", index=False)

# ---- restricted wild cluster bootstrap: does COND64 add anything WITHIN publication? ----
print("\n=== WCR bootstrap, H0: the condition block has no within-publication effect ===")
rng = np.random.default_rng(20260909)
B = 9999
yw = amp - pd.Series(amp).groupby(pub).transform("mean").to_numpy()
pubs, inv = np.unique(pub, return_inverse=True)
for restricted_name, add_name in [("TOPO39", "COND64"), ("CHEM137", "COND64"),
                                  ("TOPO39", "MASSACT8"), ("ZERO", "COND64")]:
    Xr = np.zeros((len(amp), 1)) if restricted_name == "ZERO" else demean(X_all[:, FS[restricted_name]], pub)
    Xf = np.hstack([Xr, demean(X_all[:, FS[add_name]], pub)])
    br, cr = ridge_fit(Xr, yw, ALPHA); fit_r = Xr @ br + cr; u = yw - fit_r
    bf, cf = ridge_fit(Xf, yw, ALPHA)
    stat = r2(yw, Xf @ bf + cf) - r2(yw, fit_r)
    boot = np.empty(B)
    for i in range(B):
        v = WEBB[rng.integers(0, 6, size=len(pubs))][inv]
        ys = fit_r + v * u
        ys = ys - pd.Series(ys).groupby(pub).transform("mean").to_numpy()
        b1, c1 = ridge_fit(Xr, ys, ALPHA); b2, c2 = ridge_fit(Xf, ys, ALPHA)
        boot[i] = r2(ys, Xf @ b2 + c2) - r2(ys, Xr @ b1 + c1)
    p = (np.sum(boot >= stat) + 1) / (B + 1)
    print(f"  add {add_name:9s} to {restricted_name:9s}  delta_R2_within = {stat:.4f}   "
          f"WCR p = {p:.4f}   (boot 95th pct {np.quantile(boot,0.95):.4f})")
