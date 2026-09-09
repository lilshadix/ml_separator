"""Second verification pass for D7: the oracle-smooth-baseline prize (table 5a/5f),
the smooth vs non-smooth split of the arms' mean held-out error curve (table 5c),
and a leakage stress test of the chemotype-held-out correction."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "gen13_separation")
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

ROOT = "gen13_separation"
OUT = os.path.join(ROOT, "analysis", "stage2", "d7_tetrad_residual", "verify")
LN = list(LANTHANIDES)
NEL = len(LN)
IDX = {m: i for i, m in enumerate(LN)}
r_raw = np.array([SHANNON_RADIUS_CN8[m] for m in LN], float)
Z = (r_raw - r_raw.mean()) / r_raw.std(ddof=0)

coh = pd.read_parquet(os.path.join(ROOT, "manifests", "cohort_exact.parquet"))
logD = coh[[f"logD__{m}" for m in LN]].to_numpy(float)
obs = np.isfinite(logD)
n_obs = obs.sum(1)
mean_obs = np.nansum(np.where(obs, logD, 0.0), 1) / np.maximum(n_obs, 1)
Y = np.where(obs, logD - mean_obs[:, None], np.nan)
sel = n_obs >= 6
cells = np.where(sel)[0]

FIT = np.full((len(coh), NEL), np.nan)   # oracle per-cell quadratic fitted values
R = np.full((len(coh), NEL), np.nan)
for i in cells:
    m = obs[i]
    X = np.vander(Z[m], 3, increasing=True)
    beta, *_ = np.linalg.lstsq(X, Y[i, m], rcond=None)
    FIT[i, m] = X @ beta
    R[i, m] = Y[i, m] - FIT[i, m]

extr = coh["extractant"].to_numpy()
cid = coh["cell_id"].to_numpy()

# ---- all within-cell pairs on the >=6-metal cells, A lighter than B (Z_A < Z_B)
recs = []
for i in cells:
    ms = np.where(obs[i])[0]
    for a in range(len(ms)):
        for b in range(a + 1, len(ms)):
            recs.append((i, extr[i], ms[a], ms[b]))
pairs = pd.DataFrame(recs, columns=["row", "extractant", "ia", "ib"])
print("oracle pair rows:", len(pairs), "cells:", pairs.row.nunique(),
      "extractants:", pairs.extractant.nunique())

row = pairs["row"].to_numpy()
ia = pairs["ia"].to_numpy()
ib = pairs["ib"].to_numpy()
y_pair = Y[row, ia] - Y[row, ib]
fit_pair = FIT[row, ia] - FIT[row, ib]
err = y_pair - fit_pair


def macro(errvec):
    d = pd.DataFrame({"extractant": pairs["extractant"].to_numpy(), "ae": np.abs(errvec)})
    return float(d.groupby("extractant")["ae"].mean().mean())


base = macro(err)
print(f"oracle per-cell smooth baseline extractant-macro pairwise MAE = {base:.4f}")

# leave-one-extractant-out mean residual curve
uniq = np.unique(extr[sel])
loeo = {}
for e in uniq:
    keep = (extr[sel] != e)
    with np.errstate(invalid="ignore"):
        loeo[e] = np.nan_to_num(np.nanmean(R[sel][keep], axis=0))
with np.errstate(invalid="ignore"):
    insample_curve = np.nan_to_num(np.nanmean(R[sel], axis=0))

C = np.array([loeo[e] for e in pairs["extractant"].to_numpy()])
corr_loeo = C[np.arange(len(pairs)), ia] - C[np.arange(len(pairs)), ib]
corr_ins = insample_curve[ia] - insample_curve[ib]

rows = []
for lam in np.round(np.arange(0.0, 2.01, 0.05), 2):
    rows.append(dict(lam=float(lam), loeo=macro(err - lam * corr_loeo),
                     insample=macro(err - lam * corr_ins)))
g = pd.DataFrame(rows)
g.to_csv(os.path.join(OUT, "V_c6_oracle_lambda_grid.csv"), index=False)
b = g.loc[g["loeo"].idxmin()]
print(f"  LOEO lambda=1.0 -> {float(g.loc[g.lam==1.0,'loeo'].iloc[0]):.4f} "
      f"({float(g.loc[g.lam==1.0,'loeo'].iloc[0])-base:+.4f})")
print(f"  LOEO best lambda={b['lam']:.2f} -> {b['loeo']:.4f} ({b['loeo']-base:+.4f})")
print(f"  in-sample curve lambda=1.0 -> {float(g.loc[g.lam==1.0,'insample'].iloc[0]):.4f} "
      f"({float(g.loc[g.lam==1.0,'insample'].iloc[0])-base:+.4f})")

# ---- smooth vs non-smooth of the arms' mean held-out error curve (per-cell solve)
P = np.vander(Z, 3, increasing=True)
Pproj = P @ np.linalg.pinv(P)
out = []
for arm in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW"]:
    p = pd.read_parquet(os.path.join(ROOT, "predictions", "B_primary", f"{arm}.parquet"))
    p = p[p.split_seed == p.split_seed.min()]
    p = p[p.n_metals >= 6]
    e = (p["y"] - p["prediction"]).to_numpy()
    iA = p["A"].map(IDX).to_numpy()
    iB = p["B"].map(IDX).to_numpy()
    curves = []
    for cellid, gg in pd.DataFrame({"c": p["cell_id"].to_numpy(), "iA": iA, "iB": iB,
                                    "e": e}).groupby("c"):
        ms = sorted(set(gg["iA"]) | set(gg["iB"]))
        loc = {m: k for k, m in enumerate(ms)}
        D = np.zeros((len(gg), len(ms)))
        D[np.arange(len(gg)), [loc[m] for m in gg["iA"]]] += 1
        D[np.arange(len(gg)), [loc[m] for m in gg["iB"]]] -= 1
        D = np.hstack([D, np.ones((len(gg), 0))])
        o, *_ = np.linalg.lstsq(D, gg["e"].to_numpy(), rcond=None)
        o = o - o.mean()
        full = np.full(NEL, np.nan)
        full[ms] = o
        curves.append(full)
    Cm = np.array(curves)
    with np.errstate(invalid="ignore"):
        mean_curve = np.nanmean(Cm, axis=0)
    sm = Pproj @ mean_curve
    ns = mean_curve - sm
    print(f"  {arm}: mean held-out error curve over {len(curves)} cells "
          f"-> rms smooth {np.sqrt((sm**2).mean()):.3f}, rms non-smooth {np.sqrt((ns**2).mean()):.3f}")
    print("     curve La..Lu:", np.round(mean_curve, 3).tolist())
    out.append(dict(arm=arm, n_cells=len(curves), rms_smooth=float(np.sqrt((sm ** 2).mean())),
                    rms_nonsmooth=float(np.sqrt((ns ** 2).mean()))))
pd.DataFrame(out).to_csv(os.path.join(OUT, "V_c7_arm_error_curve_split.csv"), index=False)

# ---- leakage stress test: is the chemotype-held-out curve really out of sample?
p = pd.read_parquet(os.path.join(ROOT, "predictions", "B_primary",
                                 "X_ENS_DIRECT+LOWRANK_K2.parquet"))
n_ct_per_fold = p.groupby(["split_seed", "fold"])["chemotype"].nunique()
print("\nchemotypes held out per (seed,fold): min %d max %d median %.0f  [45 total]" %
      (n_ct_per_fold.min(), n_ct_per_fold.max(), n_ct_per_fold.median()))
print("-> a leave-ONE-chemotype-out residual curve still sees the other chemotypes")
print("   that were held out in the same fold: mildly optimistic, i.e. the negative")
print("   result (correction hurts) is conservative.")
