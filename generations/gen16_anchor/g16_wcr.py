"""Vectorised restricted wild cluster bootstrap: does any block move the amplitude WITHIN a lab?

Same test as g16_within.py but the ridge hat matrices and the within-publication annihilator are
precomputed once, so each of the 9999 bootstrap draws is two mat-vecs instead of two refits.

  H0: block B has no within-publication effect on the amplitude, given the retained block R.
  statistic: increase in within-publication R^2 from adding B to R.
  bootstrap: impose H0, multiply each PUBLICATION's residual vector by one Webb 6-point draw.

  Cameron, Gelbach & Miller, Rev. Econ. Stat. 90(3):414-427, 2008, doi:10.1162/rest.90.3.414
  MacKinnon & Webb, J. Appl. Econometrics 32(2):233-254, 2017, doi:10.1002/jae.2508
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"D:\ml_separator_gh")
sys.path.insert(0, str(ROOT / "generations" / "gen14_direction")); sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
from gen14 import dirbench as db
from gen13sep.amplitude_bench import LEAN_BLOCKS

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
ALPHA, B = 30.0, 9999

bench = db.load(); FS = db.feature_sets(bench)
rich = bench.frame.n_metals.to_numpy() >= 5
pub = bench.frame["publication_id"].astype(str).to_numpy()[rich]
X_all = bench.matrix(LEAN_BLOCKS)[rich]
med = np.nanmedian(X_all, 0); med = np.where(np.isnan(med), 0.0, med)
X_all = np.where(np.isnan(X_all), med, X_all)
pubs, inv = np.unique(pub, return_inverse=True)
n, G = len(inv), len(pubs)

# M = I - P_pub, the within-publication annihilator
D = np.zeros((n, G)); D[np.arange(n), inv] = 1.0
M = np.eye(n) - D @ np.diag(1.0 / D.sum(0)) @ D.T


def hat(Xw, alpha):
    """Ridge hat matrix for already-demeaned, standardised columns."""
    sd = Xw.std(0); sd[sd < 1e-9] = 1.0
    Z = (Xw - Xw.mean(0)) / sd
    return Z @ np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T)


rng = np.random.default_rng(20260909)
for target_name, y_raw in [("signed a", bench.coef[:, 0][rich]),
                           ("|a|", np.abs(bench.coef[:, 0][rich]))]:
    yw = M @ y_raw
    print(f"\n=== target {target_name}: within-publication R^2 gain, WCR p (G={G}, within df={n-G}) ===")
    for restricted, added in [("ZERO", "COND64"), ("TOPO39", "COND64"), ("CHEM137", "COND64"),
                              ("ZERO", "MASSACT8"), ("TOPO39", "MASSACT8"),
                              ("ZERO", "TOPO39"), ("ZERO", "CHEM137")]:
        Hr = np.zeros((n, n)) if restricted == "ZERO" else hat(M @ X_all[:, FS[restricted]], ALPHA)
        Xf = M @ X_all[:, FS[added]] if restricted == "ZERO" else \
            np.hstack([M @ X_all[:, FS[restricted]], M @ X_all[:, FS[added]]])
        Hf = hat(Xf, ALPHA)
        def gain(y):
            fr, ff = Hr @ y, Hf @ y
            ss = ((y - y.mean()) ** 2).sum()
            return (((y - fr) ** 2).sum() - ((y - ff) ** 2).sum()) / ss if ss > 0 else 0.0
        stat = gain(yw)
        fit_r = Hr @ yw; u = yw - fit_r
        V = WEBB[rng.integers(0, 6, size=(B, G))][:, inv]      # B x n cluster-constant signs
        Ys = (M @ (fit_r + V * u).T)                            # n x B, re-demeaned
        Fr, Ff = Hr @ Ys, Hf @ Ys
        ss = ((Ys - Ys.mean(0)) ** 2).sum(0)
        boot = (((Ys - Fr) ** 2).sum(0) - ((Ys - Ff) ** 2).sum(0)) / np.maximum(ss, 1e-12)
        p = (np.sum(boot >= stat) + 1) / (B + 1)
        flag = "  <-- SIGNIFICANT" if p < 0.05 else ""
        print(f"  add {added:9s} to {restricted:9s}  dR2_within={stat:7.4f}  "
              f"null 95th pct={np.quantile(boot,0.95):7.4f}  WCR p={p:.4f}{flag}")
