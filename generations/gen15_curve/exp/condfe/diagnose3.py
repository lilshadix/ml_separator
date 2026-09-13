"""Third diagnostic: the within *slope*, not the within rank correlation.

``diagnose.py`` reports a pooled within-publication Spearman of +0.36 between acid concentration and
|a|, but the within ridge that the arms actually fit returns a *negative* average coefficient.  The
two are not the same estimator: the rank correlation gives every publication's internal ordering the
same weight, while an OLS/ridge weights by the raw variance of the regressor, so a handful of series
with a wide acid range and a flat |a| dominate it.  This script prints, per publication:

  slope_raw   weighted OLS slope of |a| on log10[acid] inside that publication;
  slope_log   the same for log(|a| + eps), which is what the arms model;
  rho         Spearman inside that publication;

and the three pooled estimators (variance-weighted, equally weighted over publications, and the
median slope), so the report can say which one the signal actually is and whether the sign is
stable.  Also repeats the exercise for the single hand-picked physical block.
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
EPS = 0.05


def slopes(g: pd.DataFrame, x: str, y: str) -> float:
    xv = g[x].to_numpy(dtype=float)
    yv = g[y].to_numpy(dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    if len(xv) < 3 or np.std(xv) < 1e-9:
        return np.nan
    return float(np.polyfit(xv, yv, 1)[0])


def main() -> None:
    bench = V.load()
    a = bench.coef[:, 0]
    C = CD.build(bench).assign(a=a, abs_a=np.abs(a), log_abs_a=np.log(np.abs(a) + EPS),
                               b=bench.coef[:, 1])
    R = C[bench.frame.n_metals.to_numpy() >= 5].reset_index(drop=True)

    rows = []
    for p, g in R.groupby("publication_id"):
        if g.log_acid_M.nunique() < 2 or len(g) < 3:
            continue
        rows.append({
            "publication_id": p, "n": len(g), "levels": int(g.log_acid_M.nunique()),
            "extractants": int(g.extractant.nunique()),
            "acid_lo": float(10 ** g.log_acid_M.min()), "acid_hi": float(10 ** g.log_acid_M.max()),
            "sd_log_acid": float(g.log_acid_M.std()), "sd_abs_a": float(g.abs_a.std()),
            "slope_raw": slopes(g, "log_acid_M", "abs_a"),
            "slope_log": slopes(g, "log_acid_M", "log_abs_a"),
            "rho": float(spearmanr(g.log_acid_M, g.abs_a).statistic) if g.abs_a.nunique() > 1 else np.nan,
        })
    T = pd.DataFrame(rows).sort_values("n", ascending=False)
    T.to_csv(HERE / "diag_acid_slopes.csv", index=False)
    print("=== per-publication within slope, |a| on log10[acid] (rich cells, >=3 cells) ===")
    print(T.round(3).to_string(index=False))

    ok = T.slope_raw.notna()
    wvar = T.loc[ok, "sd_log_acid"] ** 2 * T.loc[ok, "n"]
    print(f"\npooled slope_raw: variance-weighted {np.average(T.loc[ok,'slope_raw'], weights=wvar):+.4f}"
          f"   publication-equal {T.loc[ok,'slope_raw'].mean():+.4f}"
          f"   median {T.loc[ok,'slope_raw'].median():+.4f}"
          f"   fraction positive {float((T.loc[ok,'slope_raw']>0).mean()):.2f}  ({int(ok.sum())} pubs)")
    okl = T.slope_log.notna()
    wvl = T.loc[okl, "sd_log_acid"] ** 2 * T.loc[okl, "n"]
    print(f"pooled slope_log: variance-weighted {np.average(T.loc[okl,'slope_log'], weights=wvl):+.4f}"
          f"   publication-equal {T.loc[okl,'slope_log'].mean():+.4f}"
          f"   median {T.loc[okl,'slope_log'].median():+.4f}"
          f"   fraction positive {float((T.loc[okl,'slope_log']>0).mean()):.2f}")

    # single-extractant series only: the cleanest possible within comparison
    S = T[T.extractants == 1]
    print(f"\nsingle-extractant series only ({len(S)} publications, {int(S.n.sum())} cells):"
          f"  slope_raw median {S.slope_raw.median():+.4f}  fraction positive "
          f"{float((S.slope_raw>0).mean()):.2f}   rho median {S.rho.median():+.3f}")

    # is the sign of the within slope predictable from anything cheap?
    print("\n=== correlates of the per-publication slope sign ===")
    for c in ("n", "levels", "extractants", "acid_lo", "acid_hi", "sd_abs_a"):
        v = T.loc[ok, c].to_numpy(dtype=float)
        s = T.loc[ok, "slope_raw"].to_numpy(dtype=float)
        print(f"  slope_raw vs {c:12s} spearman {spearmanr(v, s).statistic:+.3f}")


if __name__ == "__main__":
    main()
