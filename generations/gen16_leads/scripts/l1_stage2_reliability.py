"""Reliability of the Stage 2 cycle-corrected slope — does the null have power this time?

Stage 1's null was uninformative because the registered `SPECIES` slope had jackknife reliability
0.000: with a free per-series parameter absorbing the trend, the estimator could not detect any
effect, so its |rho| = 0.08 said nothing about chemistry.  `DECISION_REPORT.md` §12 makes measuring
this a standing rule before a correlation is interpreted, so it is applied here to Stage 2 before
the Stage 2 verdict is written.

Two diagnostics, the same two the refuters used:

* **split-half** — fit the slope on the odd-indexed metals of a series and on the even-indexed
  ones, and correlate the two halves over extractants.  A slope that cannot reproduce itself
  cannot correlate with anything.
* **jackknife** — leave one metal out at a time, take the standard error of the slope, and compare
  it with the between-extractant spread.  reliability = max(0, 1 - mean(SE^2) / var(slope)).

Run:  .venv/Scripts/python.exe gen16_leads/scripts/l1_stage2_reliability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from gen16 import bootstrap  # noqa: E402,F401
from gen16 import l1_cycle as L  # noqa: E402

OUT = L.RESULTS
MIN_HALF = 3          # metals needed in a half before a slope is taken from it


def slope(r: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(r) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(r[ok]) < 1e-9:
        return np.nan
    return float(np.polyfit(r[ok], y[ok], 1)[0])


def per_extractant(long: pd.DataFrame, value: str) -> pd.DataFrame:
    """Split-half and jackknife diagnostics of the within-series slope of ``value``."""
    rows = []
    for ext, g in long.groupby("extractant"):
        g = g.sort_values("r")
        r = g["r"].to_numpy(dtype=float)
        y = g[value].to_numpy(dtype=float)
        ok = np.isfinite(r) & np.isfinite(y)
        r, y = r[ok], y[ok]
        n = len(r)
        if n < 2 * MIN_HALF:
            continue
        odd, even = np.arange(n) % 2 == 1, np.arange(n) % 2 == 0
        jk = [slope(np.delete(r, i), np.delete(y, i)) for i in range(n)] if n >= 3 else []
        jk = [v for v in jk if np.isfinite(v)]
        se = (np.sqrt((n - 1) / n * np.sum((np.array(jk) - np.mean(jk)) ** 2))
              if len(jk) >= 3 else np.nan)
        rows.append({"extractant": ext, "n_metals": n,
                     "slope_full": slope(r, y),
                     "slope_odd": slope(r[odd], y[odd]) if odd.sum() >= MIN_HALF else np.nan,
                     "slope_even": slope(r[even], y[even]) if even.sum() >= MIN_HALF else np.nan,
                     "jk_se": se})
    return pd.DataFrame(rows)


def summarise(d: pd.DataFrame, label: str) -> dict:
    h = d.dropna(subset=["slope_odd", "slope_even"])
    sh = stats.spearmanr(h["slope_odd"], h["slope_even"]).statistic if len(h) >= 8 else np.nan
    shp = stats.pearsonr(h["slope_odd"], h["slope_even"]).statistic if len(h) >= 8 else np.nan
    f = d.dropna(subset=["slope_full", "jk_se"])
    var = float(np.var(f["slope_full"], ddof=1)) if len(f) > 2 else np.nan
    mse = float(np.mean(f["jk_se"] ** 2)) if len(f) else np.nan
    rel = max(0.0, 1.0 - mse / var) if np.isfinite(var) and var > 0 else np.nan
    return {"quantity": label, "n_extractants": len(d), "n_split_half": len(h),
            "split_half_spearman": sh, "split_half_pearson": shp,
            "between_sd": float(np.sqrt(var)) if np.isfinite(var) else np.nan,
            "mean_jackknife_se": float(np.sqrt(mse)) if np.isfinite(mse) else np.nan,
            "reliability": rel,
            "max_attainable_abs_rho": float(np.sqrt(rel)) if np.isfinite(rel) else np.nan}


def main() -> int:
    long = L.stage2_long() if hasattr(L, "stage2_long") else None
    if long is None:
        # rebuild from the committed stage-2 slopes' inputs
        from l1_stage2_cycle import cycle_energies, load_references   # noqa: E402
        rows, _ = L.load_energy_rows()
        manifest = pd.read_csv(L.RESULTS / "reference_species" / "manifest.csv")
        ref = load_references(L.RESULTS / "reference_species" / "reference_energies.csv")
        rows = cycle_energies(rows, ref, manifest)
        long = rows.rename(columns={"canonical_smiles": "extractant"})
    out = []
    for value, label in (("dE_eV", "Stage 2: cycle-corrected dE (true reference energies)"),
                         ("complex_total_energy_eV", "Stage 1 comparator: raw complex total energy")):
        if value not in long.columns:
            print("missing column", value)
            continue
        d = per_extractant(long, value)
        d.to_csv(OUT / f"stage2_reliability_{value}.csv", index=False)
        out.append(summarise(d, label))
    T = pd.DataFrame(out)
    T.to_csv(OUT / "stage2_reliability.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
