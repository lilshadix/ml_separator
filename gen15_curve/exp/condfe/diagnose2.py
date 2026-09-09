"""Second diagnostic: is the within-publication acid-concentration signal physics or bookkeeping?

Three tightenings of the within-transformation:
  1. within (publication, extractant) -- same laboratory AND same ligand, so only the condition moves;
  2. partial out ``n_metals`` (Spearman +0.49 with |a| between publications) inside the same block;
  3. leave-one-publication-out on the pooled within correlation, to see whether one series carries it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen15 import valuebench as V  # noqa: E402
import conds as CD  # noqa: E402

HERE = Path(__file__).resolve().parent


def _rk(v):
    return pd.Series(v).rank().to_numpy()


def pooled_within(df, xcol, ycol, keys, min_n=2, resid_on=None):
    """Pool publication(-and-more)-demeaned ranks and correlate; optionally partial out a covariate."""
    px, py, blocks, ncell = [], [], 0, 0
    for _, g in df.groupby(keys, sort=False):
        if len(g) < min_n or g[xcol].nunique() < 2:
            continue
        x = _rk(g[xcol].to_numpy()); y = _rk(g[ycol].to_numpy())
        x = x - x.mean(); y = y - y.mean()
        if resid_on is not None and g[resid_on].nunique() > 1:
            z = _rk(g[resid_on].to_numpy()); z = z - z.mean()
            zz = float(z @ z)
            if zz > 1e-9:
                x = x - z * float(z @ x) / zz
                y = y - z * float(z @ y) / zz
        px.append(x); py.append(y); blocks += 1; ncell += len(g)
    if not px:
        return np.nan, 0, 0
    X = np.concatenate(px); Y = np.concatenate(py)
    if np.std(X) < 1e-12 or np.std(Y) < 1e-12:
        return np.nan, blocks, ncell
    return float(np.corrcoef(X, Y)[0, 1]), blocks, ncell


def main():
    bench = V.load()
    C = CD.build(bench).assign(a=bench.coef[:, 0], abs_a=np.abs(bench.coef[:, 0]), b=bench.coef[:, 1])
    R = C[bench.frame.n_metals.to_numpy() >= 5].reset_index(drop=True)

    print("== pooled within-block correlation, |a| ==")
    for xc in ("log_acid_M", "log_extr_M", "temp_C", "log_metal_mM", "max_logD"):
        for keys, lab in ((["publication_id"], "pub"),
                          (["publication_id", "extractant"], "pub+extractant"),
                          (["publication_id", "extractant", "diluent_name"], "pub+extr+diluent")):
            r, nb, nc = pooled_within(R, xc, "abs_a", keys)
            rp, _, _ = pooled_within(R, xc, "abs_a", keys, resid_on="n_metals")
            print(f"  {xc:12s} within {lab:18s} rho={r:+.3f}  rho|n_metals={rp:+.3f}  "
                  f"blocks={nb:3d} cells={nc:3d}")
        print()

    print("== within (pub+extractant), other targets ==")
    for tgt in ("a", "b"):
        for xc in ("log_acid_M", "log_extr_M", "temp_C"):
            r, nb, nc = pooled_within(R, xc, tgt, ["publication_id", "extractant"])
            print(f"  {tgt:5s} vs {xc:12s} rho={r:+.3f} blocks={nb} cells={nc}")

    print("\n== leave-one-publication-out on within(pub) rho, log_acid_M vs |a| ==")
    base, nb, nc = pooled_within(R, "log_acid_M", "abs_a", ["publication_id"])
    print(f"  all: {base:+.3f}  ({nb} varying publications, {nc} cells)")
    rows = []
    for p in sorted(R.publication_id.unique()):
        sub = R[R.publication_id != p]
        r, nb2, _ = pooled_within(sub, "log_acid_M", "abs_a", ["publication_id"])
        n_here = int(((R.publication_id == p)).sum())
        rows.append((p, n_here, r))
    rows.sort(key=lambda t: t[2])
    for p, n, r in rows[:5]:
        print(f"  drop {p} (n={n:2d}) -> {r:+.3f}")
    print("  ...")
    for p, n, r in rows[-3:]:
        print(f"  drop {p} (n={n:2d}) -> {r:+.3f}")

    print("\n== per-publication acid series (rich cells, >=3 cells, >=3 acid levels) ==")
    for (p,), g in R.groupby(["publication_id"], sort=False):
        if g.log_acid_M.nunique() < 3 or len(g) < 3:
            continue
        r = spearmanr(g.log_acid_M, g.abs_a).statistic
        print(f"  {p}  n={len(g):3d} levels={g.log_acid_M.nunique():2d} "
              f"extractants={g.extractant.nunique():2d} rho={r:+.3f} "
              f"acid={10**g.log_acid_M.min():.2f}-{10**g.log_acid_M.max():.2f}M "
              f"|a|={g.abs_a.min():.2f}-{g.abs_a.max():.2f}")


if __name__ == "__main__":
    main()
