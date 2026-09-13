"""Fifth diagnostic: leave-one-chemotype-out and leave-one-publication-out stability of the
within-publication condition correlations, and the identifying-variance census.

The protocol requires every correlation in this corpus to be guarded against three confounds --
publication identity (handled by construction here, the correlation *is* the within one), the number
of metals a cell measured (Spearman +0.49 with |a|), and chemotype.  This script reports:

  * the pooled within-publication Spearman of each condition with |a|, and its full leave-one-
    chemotype-out and leave-one-publication-out range;
  * the census of *identifying variance*: which publications actually carry the pooled within
    estimate, under the chemotype-balanced weights the programme's models use.
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


def _rk(v):
    return pd.Series(v).rank().to_numpy()


def pooled_within_rho(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[float, int, int]:
    px, py, nb, nc = [], [], 0, 0
    for _, g in df.groupby("publication_id", sort=False):
        if len(g) < 2 or g[xcol].nunique() < 2:
            continue
        x = _rk(g[xcol].to_numpy()); y = _rk(g[ycol].to_numpy())
        px.append(x - x.mean()); py.append(y - y.mean()); nb += 1; nc += len(g)
    if not px:
        return np.nan, 0, 0
    X, Y = np.concatenate(px), np.concatenate(py)
    if np.std(X) < 1e-12 or np.std(Y) < 1e-12:
        return np.nan, nb, nc
    return float(np.corrcoef(X, Y)[0, 1]), nb, nc


def main() -> None:
    bench = V.load()
    a = bench.coef[:, 0]
    rich = bench.frame.n_metals.to_numpy() >= 5
    C = CD.build(bench).assign(a=a, abs_a=np.abs(a), b=bench.coef[:, 1])
    R = C[rich].reset_index(drop=True)

    cols = ["log_acid_M", "log_extr_M", "temp_C", "is_hno3", "modifier", "max_logD"]
    print("=== pooled within-publication Spearman with |a|, and its stability ===")
    print(f"{'column':12s} {'rho':>7s} {'pubs':>5s} {'cells':>6s} "
          f"{'LOCO min':>9s} {'LOCO max':>9s} {'LOCO sign stable':>17s} "
          f"{'LOPO min':>9s} {'LOPO max':>9s}")
    rows = []
    for c in cols:
        base, nb, nc = pooled_within_rho(R, c, "abs_a")
        loco = []
        for ch in sorted(R.chemotype.unique()):
            r, _, _ = pooled_within_rho(R[R.chemotype != ch], c, "abs_a")
            if np.isfinite(r):
                loco.append(r)
        lopo = []
        for p in sorted(R.publication_id.unique()):
            r, _, _ = pooled_within_rho(R[R.publication_id != p], c, "abs_a")
            if np.isfinite(r):
                lopo.append(r)
        stable = bool(np.all(np.sign(loco) == np.sign(base))) if loco else False
        print(f"{c:12s} {base:+7.3f} {nb:5d} {nc:6d} {min(loco):+9.3f} {max(loco):+9.3f} "
              f"{str(stable):>17s} {min(lopo):+9.3f} {max(lopo):+9.3f}")
        rows.append({"column": c, "rho_within": base, "n_pub": nb, "n_cells": nc,
                     "loco_min": min(loco), "loco_max": max(loco), "loco_sign_stable": stable,
                     "lopo_min": min(lopo), "lopo_max": max(lopo)})
    pd.DataFrame(rows).to_csv(HERE / "diag_loco_stability.csv", index=False)

    # --- identifying-variance census under the programme's own weights -------------------------
    idx = np.flatnonzero(rich)
    w = cell_weights(bench.groups[idx], bench.n_obs[idx])
    g = R.publication_id.to_numpy()
    print("\n=== identifying variance of the pooled within estimate, chemotype-balanced weights ===")
    cen = []
    for c in ("log_acid_M", "log_extr_M", "temp_C"):
        x = F._impute(R[[c]].to_numpy(float), np.arange(len(R)))[:, 0]
        xt = F._demean_vec(x, g, w)
        wxx = np.array([float(np.sum(w[g == u] * xt[g == u] ** 2)) for u in np.unique(g)])
        n = np.array([int((g == u).sum()) for u in np.unique(g)])
        o = np.argsort(-wxx)
        tot = wxx.sum()
        top = [(str(np.unique(g)[i]), int(n[i]), float(wxx[i] / tot)) for i in o[:3]]
        share2 = float(wxx[o[:2]].sum() / tot)
        print(f"  {c:12s} top-3 (pub, cells, share): " +
              "  ".join(f"{p}/{k}c/{s:.0%}" for p, k, s in top) +
              f"   top-2 share {share2:.0%}")
        cen.append({"column": c, "top2_share": share2,
                    "top1_pub": top[0][0], "top1_cells": top[0][1], "top1_share": top[0][2]})
    pd.DataFrame(cen).to_csv(HERE / "diag_identifying_variance.csv", index=False)


if __name__ == "__main__":
    main()
