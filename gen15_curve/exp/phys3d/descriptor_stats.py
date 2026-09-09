"""Per-descriptor association of the phys3d block with the curve, guarded against the corpus's
three standing confounds.

The unit is the *extractant* (82 of them have a well-determined curve), not the cell: the block is
constant within an extractant, so a cell-level correlation would just re-weight ligands by how
often they were published.  For each descriptor three targets are reported:

    a       the radius coefficient with its sign  (direction x magnitude)
    |a|     the magnitude, where every remaining error now lives
    b       the curvature

and for each, four numbers:

    rho             Spearman over the extractants that have the descriptor
    p               its two-sided p-value
    LOCO min/max    the same rho recomputed 45 times, each time dropping one chemotype
    rho | n_metals  partial Spearman controlling the number of metals the *experiment*
                    measured, which is +0.49 with |a| and is this corpus's nastiest confound
                    (a ligand studied over the whole series is also a ligand someone thought
                    was worth a whole-series study)

Nothing here touches the bench folds; it is a description of the corpus, not a score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gen15_curve"))
from gen15 import valuebench as V  # noqa: E402

MIN_METALS = 5


def extractant_table() -> pd.DataFrame:
    bench = V.load()
    f = bench.frame.copy()
    f["a"] = bench.coef[:, 0]
    f["b"] = bench.coef[:, 1]
    rich = f["n_metals"].to_numpy() >= MIN_METALS
    r = f[rich]
    g = r.groupby("extractant").agg(a=("a", "mean"), b=("b", "mean"),
                                    n_metals=("n_metals", "mean"), n_cells=("a", "size"),
                                    chemotype=("chemotype", "first"))
    g["abs_a"] = g["a"].abs()
    return g.reset_index()


def _rho(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan, int(ok.sum())
    r = stats.spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue), int(ok.sum())


def _partial_rho(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman between x and y with z partialled out, on ranks (the usual definition)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return np.nan
    rx, ry, rz = (stats.rankdata(v[ok]) for v in (x, y, z))
    A = np.c_[np.ones(ok.sum()), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def _between_within(x: np.ndarray, y: np.ndarray, chem: np.ndarray) -> tuple[float, float, float]:
    """Split an association into the part that transfers across chemotypes and the part that does not.

    Designs B and BP hold out whole chemotypes, so only the *between*-chemotype part of a
    correlation can ever be used.  ``icc`` is the share of the descriptor's variance that lies
    between chemotypes: a descriptor that is nearly constant inside a chemotype (icc near 1) is
    a chemotype label in disguise; one with icc near 0 cannot transfer at all.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, c = x[ok], y[ok], chem[ok]
    if len(x) < 20:
        return np.nan, np.nan, np.nan
    d = pd.DataFrame({"x": x, "y": y, "c": c})
    m = d.groupby("c").mean()
    n = d.groupby("c").size()
    keep = m.index[n >= 1]
    bet = stats.spearmanr(m.loc[keep, "x"], m.loc[keep, "y"]).statistic if len(keep) >= 8 else np.nan
    d["xw"] = d["x"] - d.groupby("c")["x"].transform("mean")
    d["yw"] = d["y"] - d.groupby("c")["y"].transform("mean")
    multi = d[d.groupby("c")["x"].transform("size") > 1]
    wit = stats.spearmanr(multi["xw"], multi["yw"]).statistic if len(multi) >= 20 else np.nan
    grand = d["x"].var()
    icc = float(1.0 - d["xw"].var() / grand) if grand > 0 else np.nan
    return float(bet) if np.isfinite(bet) else np.nan, \
        float(wit) if np.isfinite(wit) else np.nan, icc


def stats_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    chem = df["chemotype"].to_numpy()
    nm = df["n_metals"].to_numpy(dtype=float)
    for c in cols:
        x = df[c].to_numpy(dtype=float)
        rec = {"descriptor": c.replace("phys3d__", "")}
        for tgt in ("a", "abs_a", "b"):
            y = df[tgt].to_numpy(dtype=float)
            rho, p, n = _rho(x, y)
            loco = []
            for ch in np.unique(chem):
                m = chem != ch
                rr, _, nn = _rho(x[m], y[m])
                if np.isfinite(rr):
                    loco.append(rr)
            rec[f"n_{tgt}"] = n
            rec[f"rho_{tgt}"] = rho
            rec[f"p_{tgt}"] = p
            rec[f"loco_lo_{tgt}"] = min(loco) if loco else np.nan
            rec[f"loco_hi_{tgt}"] = max(loco) if loco else np.nan
            rec[f"part_{tgt}"] = _partial_rho(x, y, nm)
            bet, wit, icc = _between_within(x, y, chem)
            rec[f"between_{tgt}"] = bet
            rec[f"within_{tgt}"] = wit
            rec["icc_chemotype"] = icc
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    df = extractant_table()
    blk = pd.read_parquet(HERE / "phys3d_block.parquet")
    df = df.merge(blk, on="extractant", how="left")
    cols = [c for c in blk.columns if c.startswith("phys3d__")]
    print(f"extractants with a well-determined curve: {len(df)}")
    print("descriptor coverage:")
    print(df[cols].notna().sum().to_string())
    rho_nm = _rho(df["n_metals"].to_numpy(float), df["abs_a"].to_numpy(float))
    print(f"\nconfound check  rho(n_metals, |a|) = {rho_nm[0]:.3f}  p={rho_nm[1]:.2g}  n={rho_nm[2]}")
    rho_nc = _rho(df["n_cells"].to_numpy(float), df["abs_a"].to_numpy(float))
    print(f"confound check  rho(n_cells,  |a|) = {rho_nc[0]:.3f}  p={rho_nc[1]:.2g}  n={rho_nc[2]}")
    t = stats_table(df, cols)
    t.to_csv(HERE / "descriptor_stats.csv", index=False)
    for tgt, label in (("a", "a  (signed radius coefficient)"),
                       ("abs_a", "|a|  (magnitude)"), ("b", "b  (curvature)")):
        show = t[["descriptor", f"n_{tgt}", f"rho_{tgt}", f"p_{tgt}",
                  f"loco_lo_{tgt}", f"loco_hi_{tgt}", f"part_{tgt}",
                  f"between_{tgt}", "icc_chemotype"]].copy()
        show.columns = ["descriptor", "n", "rho", "p", "loco_lo", "loco_hi", "rho|n_metals",
                        "rho_between_chemo", "icc_chemo"]
        show = show.sort_values("rho", key=lambda s: -s.abs())
        print(f"\n=== {label} ===")
        print(show.round(3).to_string(index=False))
    df.to_parquet(HERE / "extractant_targets.parquet", index=False)


if __name__ == "__main__":
    main()
