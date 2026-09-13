"""Fourth diagnostic: how much of the WITHIN-publication variance in the magnitude do conditions
explain, and does it generalise to a publication the fit has never seen?

This is the decisive number for the whole route.  The within-transformation can only ever predict a
cell's deviation from its own laboratory's mean; ``o_within`` prices perfect recovery of that
deviation on the bench.  Here we ask what fraction of it the conditions actually recover:

  R2_in    weighted within-publication R^2 of the fixed-effects fit, in sample;
  R2_loo   the same, but every publication is predicted by a fit that excluded it -- the honest
           number, and the one that corresponds to design BP.

Reported for the univariate acid model, the four hand-picked physical columns, the fourteen-column
recoding and the raw 64.  Also prints the univariate within slope for each column on its own, so the
report can say whether the multivariate sign disagrees with the marginal one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import cell_weights  # noqa: E402
import conds as CD  # noqa: E402
import fearms as F  # noqa: E402

HERE = Path(__file__).resolve().parent
EPS = 0.05


def within_r2(X: np.ndarray, y: np.ndarray, w: np.ndarray, g: np.ndarray, lam: float = 1.0):
    """In-sample and leave-one-publication-out within R^2 of the fixed-effects fit."""
    mu, sd = F._standardise(X, w)
    Z = (X - mu) / sd
    yt = F._demean_vec(y, g, w)
    sst = float(np.sum(w * yt ** 2))
    beta, keep = F._within_beta(Z, y, w, g, lam)
    Zt = F._demean_rows(Z, g, w)
    r2_in = 1.0 - float(np.sum(w * (yt - Zt @ beta) ** 2)) / sst

    num = 0.0
    den = 0.0
    for u in np.unique(g):
        m = g == u
        if m.sum() < 2 or (~m).sum() < 20:
            continue
        bo, _ = F._within_beta(Z[~m], y[~m], w[~m], g[~m], lam)
        # inside the held-out publication, compare the *deviations*: both sides demeaned
        zt = Z[m] - np.average(Z[m], axis=0, weights=w[m])
        yy = y[m] - np.average(y[m], weights=w[m])
        pred = zt @ bo
        num += float(np.sum(w[m] * (yy - pred) ** 2))
        den += float(np.sum(w[m] * yy ** 2))
    r2_loo = 1.0 - num / den if den > 0 else np.nan
    return r2_in, r2_loo, int(keep.sum())


def main() -> None:
    bench = V.load()
    a = bench.coef[:, 0]
    rich = bench.frame.n_metals.to_numpy() >= 5
    idx = np.flatnonzero(rich)
    C = CD.build(bench)
    w = cell_weights(bench.groups[idx], bench.n_obs[idx])
    g = C["publication_id"].to_numpy()[idx]
    y_log = np.log(np.abs(a[idx]) + EPS)
    y_mag = np.abs(a[idx])
    y_a = a[idx]
    y_b = bench.coef[idx, 1]

    blocks = {
        "ACID1": (["log_acid_M"], C[["log_acid_M"]].to_numpy(dtype=float)),
        "PHYS4": (list(CD.PHYS4), C[list(CD.PHYS4)].to_numpy(dtype=float)),
        "NUM14": (list(CD.NUMERIC), C[list(CD.NUMERIC)].to_numpy(dtype=float)),
        "COND64": (list(bench.frames["COND"].columns), bench.frames["COND"].to_numpy(dtype=float)),
    }
    print("=== within-publication R^2 of the fixed-effects fit (rich cells, %d of %d) ===" %
          (len(idx), len(C)))
    print(f"{'block':8s} {'target':8s} {'p':>4s} {'ident':>6s} {'R2_in':>8s} {'R2_loo':>8s}")
    rows = []
    for name, (cols, X) in blocks.items():
        Xf = F._impute(X, np.arange(len(X)))[idx]
        for tname, yv in (("log|a|", y_log), ("|a|", y_mag), ("a", y_a), ("b", y_b)):
            r2i, r2l, nid = within_r2(Xf, yv, w, g)
            print(f"{name:8s} {tname:8s} {X.shape[1]:4d} {nid:6d} {r2i:8.4f} {r2l:8.4f}")
            rows.append({"block": name, "target": tname, "p": X.shape[1], "identified": nid,
                         "r2_within_in": r2i, "r2_within_loo": r2l})
    pd.DataFrame(rows).to_csv(HERE / "diag_within_r2.csv", index=False)

    print("\n=== univariate within slope of log|a| on each column, standardised ===")
    print(f"{'column':18s} {'ident':>6s} {'beta':>9s} {'R2_in':>8s} {'R2_loo':>8s}")
    uni = []
    for c in list(CD.NUMERIC):
        X = F._impute(C[[c]].to_numpy(dtype=float), np.arange(len(C)))[idx]
        mu, sd = F._standardise(X, w)
        Z = (X - mu) / sd
        beta, keep = F._within_beta(Z, y_log, w, g, 1.0)
        r2i, r2l, nid = within_r2(X, y_log, w, g)
        print(f"{c:18s} {nid:6d} {beta[0]:+9.4f} {r2i:8.4f} {r2l:8.4f}")
        uni.append({"column": c, "identified": nid, "beta_within": float(beta[0]),
                    "r2_within_in": r2i, "r2_within_loo": r2l})
    pd.DataFrame(uni).to_csv(HERE / "diag_within_univariate.csv", index=False)

    # how big is the within part at all?
    yt = F._demean_vec(y_mag, g, w)
    print(f"\n|a| weighted sd total {np.sqrt(np.average((y_mag-np.average(y_mag,weights=w))**2, weights=w)):.4f}"
          f"   within-publication sd {np.sqrt(np.average(yt**2, weights=w)):.4f}")


if __name__ == "__main__":
    main()
