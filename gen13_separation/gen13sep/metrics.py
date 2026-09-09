"""Metrics on pairwise separation factors.  Every number carries its unit.

The atom is the **pair table**: one row per (split seed, held-out cell, metal A,
metal B) with the observed ``log SF = log D(A) - log D(B)`` (A lighter) and one
prediction column per arm.  From it:

* ``per_extractant`` — one row per (split seed, extractant, arm): MAE over all of the
  extractant's pairs (pair-pooled within the extractant), over adjacent pairs (true
  neighbours ``dZ = 1`` plus Nd–Sm across the promethium gap) and over far pairs
  (``dZ >= 5``); sign accuracy over pairs with ``|log SF| >= 0.3`` (about 0.9 pair-noise
  sd, see DATA_AUDIT §3); ``pair_spearman`` = mean per-cell Spearman between observed and
  predicted *pairwise contrasts* for cells with >= 4 metals.  The curve-level Spearman
  (observed vs predicted centred curve over a cell's metals) is computed by the analysis
  script from the saved curves and reported as ``curve_spearman``.
* ``summarise`` — macro over extractants within seed, then mean and sd over seeds;
  chemotype-macro; pooled over pairs.  Pooled numbers are descriptive only: one
  chemotype holds 375 of 521 cells.

The "heavier always preferred" rule is run as an arm (``B4_HEAVIER_ALWAYS``) so that its
sign accuracy is scored with the same unit and weighting as every other arm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metals import ATOMIC_NUMBER, LANTHANIDES

FAR_MIN_DZ = 5
STRONG_SIGN_THRESHOLD = 0.3
MIN_METALS_FOR_SPEARMAN = 4


def is_adjacent(pairs: pd.DataFrame) -> pd.Series:
    """True neighbours (dZ = 1) plus the Nd-Sm pair that straddles promethium."""
    return (pairs["dZ"] == 1) | ((pairs["A"] == "Nd") & (pairs["B"] == "Sm"))


def pair_rows_for_cell(y_row: np.ndarray) -> list[tuple[int, int, float]]:
    obs = np.flatnonzero(~np.isnan(y_row))
    return [(a, b, float(y_row[a] - y_row[b])) for a in obs for b in obs if a < b]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def per_extractant(pairs: pd.DataFrame, arms: list[str], curve_spearman: pd.DataFrame | None = None) -> pd.DataFrame:
    """``curve_spearman``: optional long table (split_seed, cell_id, arm, curve_spearman) from the analysis script."""
    rows = []
    adj_all = is_adjacent(pairs)
    for (seed, ext), block in pairs.groupby(["split_seed", "extractant"], sort=True):
        adj = adj_all.loc[block.index]
        far = block["dZ"] >= FAR_MIN_DZ
        strong = block["y"].abs() >= STRONG_SIGN_THRESHOLD
        base = {"split_seed": int(seed), "extractant": ext, "chemotype": block["chemotype"].iat[0],
                "n_pairs": int(len(block)), "n_cells": int(block["cell_id"].nunique()),
                "n_adjacent": int(adj.sum()), "n_far": int(far.sum()), "n_strong": int(strong.sum()),
                "y_abs_mean": float(block["y"].abs().mean())}
        for arm in arms:
            err = (block["y"] - block[arm]).abs()
            rec = dict(base, arm=arm,
                       mae_all=float(err.mean()),
                       mae_adjacent=float(err[adj].mean()) if adj.any() else np.nan,
                       mae_far=float(err[far].mean()) if far.any() else np.nan,
                       sign_acc_strong=float((np.sign(block.loc[strong, "y"]) == np.sign(block.loc[strong, arm])).mean()) if strong.any() else np.nan)
            sps = []
            for cell_id, cell in block.groupby("cell_id"):
                if cell["n_metals"].iat[0] >= MIN_METALS_FOR_SPEARMAN:
                    sps.append(_spearman(cell["y"].to_numpy(), cell[arm].to_numpy()))
            sps = np.asarray(sps, dtype=float)
            rec["pair_spearman"] = float(np.nanmean(sps)) if sps.size and np.isfinite(sps).any() else np.nan
            rec["n_cells_spearman"] = int(np.isfinite(sps).sum()) if sps.size else 0
            if curve_spearman is not None:
                cs = curve_spearman[(curve_spearman["split_seed"] == seed) & (curve_spearman["arm"] == arm)
                                    & (curve_spearman["cell_id"].isin(block["cell_id"].unique()))]["curve_spearman"]
                rec["curve_spearman"] = float(cs.mean()) if cs.notna().any() else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def summarise(per_ext: pd.DataFrame, pairs: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    out = []
    adj = is_adjacent(pairs)
    far = pairs["dZ"] >= FAR_MIN_DZ
    strong = pairs["y"].abs() >= STRONG_SIGN_THRESHOLD
    for arm in arms:
        sub = per_ext[per_ext["arm"] == arm]
        by_seed = sub.groupby("split_seed")
        macro = by_seed["mae_all"].mean()
        chem = sub.groupby(["split_seed", "chemotype"])["mae_all"].mean().groupby(level=0).mean()
        pooled = (pairs["y"] - pairs[arm]).abs()
        rec = {
            "arm": arm,
            "macro_mae_extractant": float(macro.mean()),
            "macro_mae_extractant_seed_sd": float(macro.std(ddof=1)) if len(macro) > 1 else np.nan,
            "macro_mae_chemotype": float(chem.mean()),
            "macro_mae_adjacent": float(by_seed["mae_adjacent"].mean().mean()),
            "macro_mae_far": float(by_seed["mae_far"].mean().mean()),
            "macro_sign_acc_strong": float(by_seed["sign_acc_strong"].mean().mean()),
            "macro_pair_spearman": float(by_seed["pair_spearman"].mean().mean()),
            "pooled_mae": float(pooled.mean()),
            "pooled_mae_adjacent": float(pooled[adj].mean()),
            "pooled_mae_far": float(pooled[far].mean()),
            "pooled_sign_acc_strong": float((np.sign(pairs.loc[strong, "y"]) == np.sign(pairs.loc[strong, arm])).mean()),
            "n_units_mae_all": int(sub["extractant"].nunique()),
            "n_units_adjacent": int(sub.dropna(subset=["mae_adjacent"])["extractant"].nunique()),
            "n_units_far": int(sub.dropna(subset=["mae_far"])["extractant"].nunique()),
            "n_units_sign": int(sub.dropna(subset=["sign_acc_strong"])["extractant"].nunique()),
            "n_seeds": int(sub["split_seed"].nunique()),
        }
        if "curve_spearman" in sub.columns:
            rec["macro_curve_spearman"] = float(by_seed["curve_spearman"].mean().mean())
        out.append(rec)
    return pd.DataFrame(out).sort_values("macro_mae_extractant").reset_index(drop=True)


def heavier_always_sign_accuracy(pairs: pd.DataFrame, *, macro: bool = True) -> float:
    """Sign accuracy of 'heavier always preferred' on strong pairs; macro = one vote per (seed, extractant)."""
    strong = pairs["y"].abs() >= STRONG_SIGN_THRESHOLD
    hit = (pairs.loc[strong, "y"] < 0).astype(float)
    if not macro:
        return float(hit.mean())
    return float(hit.groupby([pairs.loc[strong, "split_seed"], pairs.loc[strong, "extractant"]]).mean().mean())


def effective_sample_size(counts) -> float:
    p = np.asarray(list(counts), dtype=float)
    p = p / p.sum()
    return float(1.0 / (p ** 2).sum())
