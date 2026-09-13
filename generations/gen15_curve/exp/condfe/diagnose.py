"""Diagnostic: how much of each condition's association with the curve is within a publication?

For every condition variable and every target (a, |a|, b) we report

  rho_between   Spearman over publications, of the publication mean of the condition against the
                publication mean of the target -- the part a laboratory fingerprint can explain;
  rho_within    Spearman of the publication-demeaned condition against the publication-demeaned
                target, pooled over every publication that actually varies the condition -- the
                part the within-transformation can use;
  n_pub_var     publications with >= 2 distinct values of the condition (>= 2 cells);
  n_cells_var   cells inside those publications.

Restricted throughout to well-determined cells (>= 5 measured metals), which is the cohort the
programme's magnitude work uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "generations" / "gen15_curve"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen15 import valuebench as V  # noqa: E402
import conds as CD  # noqa: E402

HERE = Path(__file__).resolve().parent
MIN_METALS = 5


def _rank(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank().to_numpy()


def within_between(x: np.ndarray, y: np.ndarray, pub: np.ndarray) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, pub = x[ok], y[ok], pub[ok]
    # between
    d = pd.DataFrame({"p": pub, "x": x, "y": y}).groupby("p").mean()
    rb = np.nan
    if len(d) >= 4 and d["x"].nunique() > 1:
        rb = float(spearmanr(d["x"].to_numpy(), d["y"].to_numpy()).statistic)
    # within: rank inside each varying publication, then demean the ranks
    parts_x, parts_y, npv, ncv, per_pub = [], [], 0, 0, []
    for p, g in pd.DataFrame({"p": pub, "x": x, "y": y}).groupby("p"):
        if len(g) < 2 or g["x"].nunique() < 2:
            continue
        npv += 1
        ncv += len(g)
        rx = _rank(g["x"].to_numpy())
        ry = _rank(g["y"].to_numpy())
        parts_x.append(rx - rx.mean())
        parts_y.append(ry - ry.mean())
        if len(g) >= 4 and g["y"].nunique() > 1:
            per_pub.append(float(spearmanr(g["x"].to_numpy(), g["y"].to_numpy()).statistic))
    rw = np.nan
    if parts_x:
        X = np.concatenate(parts_x); Yv = np.concatenate(parts_y)
        if np.std(X) > 0 and np.std(Yv) > 0:
            rw = float(np.corrcoef(X, Yv)[0, 1])
    per_pub = [v for v in per_pub if np.isfinite(v)]
    return {"rho_between": rb, "rho_within": rw, "n_pub_var": npv, "n_cells_var": ncv,
            "n_pub_scored": len(per_pub),
            "med_pub_rho": float(np.median(per_pub)) if per_pub else np.nan,
            "frac_pub_pos": float(np.mean(np.array(per_pub) > 0)) if per_pub else np.nan}


def main() -> None:
    bench = V.load()
    C = CD.build(bench)
    a = bench.coef[:, 0]
    b = bench.coef[:, 1]
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    C = C.assign(a=a, abs_a=np.abs(a), b=b, abs_b=np.abs(b))
    R = C[rich].reset_index(drop=True)
    print(f"cells total {len(C)}  rich(>= {MIN_METALS} metals) {len(R)}  "
          f"publications {R.publication_id.nunique()}  chemotypes {R.chemotype.nunique()}")

    varlist = list(CD.NUMERIC) + ["max_logD", "mean_logD", "n_metals"]
    rows = []
    for v in varlist:
        for tgt in ("a", "abs_a", "b"):
            d = within_between(R[v].to_numpy(dtype=float), R[tgt].to_numpy(dtype=float),
                               R["publication_id"].to_numpy())
            d.update({"var": v, "target": tgt})
            rows.append(d)
    T = pd.DataFrame(rows)[["var", "target", "rho_between", "rho_within", "n_pub_var",
                            "n_cells_var", "n_pub_scored", "med_pub_rho", "frac_pub_pos"]]
    T.to_csv(HERE / "diag_within_between.csv", index=False)
    for tgt in ("a", "abs_a", "b"):
        print(f"\n=== target {tgt} ===")
        print(T[T.target == tgt].drop(columns=["target"]).round(3).to_string(index=False))

    # categorical: how many publications carry >1 acid identity / diluent class / complexant state
    print("\n=== categorical variation inside a publication (rich cells) ===")
    for v in ("acid_name", "diluent_class", "diluent_name", "complexant", "modifier"):
        g = R.groupby("publication_id")[v].nunique()
        print(f"  {v:16s}  publications varying it: {int((g > 1).sum())} / {len(g)}   "
              f"cells inside those: {int(R.publication_id.isin(g[g > 1].index).sum())}")

    # how big are publications, and how much does the magnitude vary between them
    sz = R.groupby("publication_id").size().sort_values(ascending=False)
    print("\nlargest publications (rich cells):", list(sz.head(8).items()))
    pm = R.groupby("publication_id")["abs_a"].mean()
    ssb = ((pm - R.abs_a.mean()) ** 2 * R.groupby("publication_id").size()).sum()
    sst = ((R.abs_a - R.abs_a.mean()) ** 2).sum()
    print(f"|a|: between-publication share of variance = {ssb / sst:.3f}  "
          f"({R.publication_id.nunique()} publications)")
    pm2 = R.groupby("chemotype")["abs_a"].mean()
    ssb2 = ((pm2 - R.abs_a.mean()) ** 2 * R.groupby("chemotype").size()).sum()
    print(f"|a|: between-chemotype   share of variance = {ssb2 / sst:.3f}")
    C.to_csv(HERE / "cells_conditions.csv", index=False)


if __name__ == "__main__":
    main()
