"""Pooled and grouped metrics on query rows.  Every undefined case is returned as NaN with a
reason, never as 0 or 1, and R² is never clipped.

Input: a long table with one row per (design, split_seed, policy, draw, k, cell_id, A, B) and
columns ``y`` (observed) and ``pred`` (one arm's prediction).  Pooled R² is
``1 - sum (y - p)^2 / sum (y - mean y)^2`` over the rows given; a zero prediction counts 0.5 in the
direction metrics (declared in the pre-registration, identical for every arm).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIRECTION_THRESHOLDS: tuple[float, ...] = (0.0, 0.1, 0.3)


def r2(y: np.ndarray, p: np.ndarray) -> tuple[float, str]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return float("nan"), "fewer than two observations"
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0.0:
        return float("nan"), "constant target"
    return float(1.0 - ((y - p) ** 2).sum() / sst), ""


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.abs(y[ok] - p[ok]).mean()) if ok.any() else float("nan")


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(((y[ok] - p[ok]) ** 2).mean())) if ok.any() else float("nan")


def direction_accuracy(y, p, threshold: float = 0.0) -> tuple[float, int]:
    """Sign agreement on rows with |y| > threshold (strictly > for 0, >= otherwise); a zero
    prediction scores 0.5.  Returns (accuracy, n rows used)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    sel = ok & ((np.abs(y) > 0) if threshold == 0 else (np.abs(y) >= threshold))
    if not sel.any():
        return float("nan"), 0
    sp = np.sign(p[sel])
    score = np.where(sp == 0, 0.5, (sp == np.sign(y[sel])).astype(float))
    return float(score.mean()), int(sel.sum())


def pooled(table: pd.DataFrame, pred: str = "pred") -> dict:
    y, p = table["y"].to_numpy(float), table[pred].to_numpy(float)
    r, why = r2(y, p)
    out = {"n_rows": int(np.isfinite(y).sum()), "n_pred_finite": int((np.isfinite(y) & np.isfinite(p)).sum()),
           "n_cells": int(table["cell_id"].nunique()),
           "n_extractants": int(table["extractant"].nunique()), "n_chemotypes": int(table["chemotype"].nunique()),
           "R2": r, "R2_reason": why, "MAE": mae(y, p), "RMSE": rmse(y, p),
           "y_sd": float(np.nanstd(y)) if np.isfinite(y).sum() > 1 else float("nan"),
           "MAE_FLAT": float(np.nanmean(np.abs(y)))}
    for t in DIRECTION_THRESHOLDS:
        acc, n = direction_accuracy(y, p, t)
        out[f"dir_acc_{t:g}"] = acc
        out[f"n_dir_{t:g}"] = n
    # extractant-macro MAE (continuity with gen13-gen16): mean over extractants of their pair-pooled MAE
    e = table.assign(_e=(table["y"] - table[pred]).abs())
    per = e.groupby("extractant")["_e"].mean()
    out["MAE_macro_extractant"] = float(per.mean())
    per_c = e.groupby(["chemotype", "extractant"])["_e"].mean().groupby(level=0).mean()
    out["MAE_macro_chemotype"] = float(per_c.mean())
    return out


def per_pair_type(table: pd.DataFrame, pred: str = "pred") -> pd.DataFrame:
    rows = []
    for (a, b), blk in table.groupby(["A", "B"], sort=False):
        r, why = r2(blk["y"], blk[pred])
        rows.append({"pair": f"{a}/{b}", "A": a, "B": b, "n_rows": int(len(blk)),
                     "n_extractants": int(blk["extractant"].nunique()), "R2": r, "R2_reason": why,
                     "MAE": mae(blk["y"], blk[pred]), "RMSE": rmse(blk["y"], blk[pred]),
                     "y_sd": float(blk["y"].std(ddof=0)), "MAE_FLAT": float(blk["y"].abs().mean()),
                     "dir_acc_0": direction_accuracy(blk["y"], blk[pred], 0.0)[0]})
    return pd.DataFrame(rows)


def spearman_within_cells(table: pd.DataFrame, pred: str = "pred", min_queries: int = 4) -> float:
    from scipy.stats import spearmanr
    vals = []
    for _, blk in table.groupby("cell_id"):
        if len(blk) >= min_queries and blk["y"].nunique() > 1 and blk[pred].nunique() > 1:
            vals.append(float(spearmanr(blk["y"], blk[pred]).statistic))
    return float(np.nanmean(vals)) if vals else float("nan")
