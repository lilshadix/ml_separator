"""Confound guards for every arm this experiment scored.

The corpus has three known ways to manufacture a correlation, and the protocol requires each to be
addressed explicitly:

* **publication identity** -- handled by the design axis itself (BP masks the training cells that
  share a publication with the held-out ones); the guard here is that a gain must keep its sign
  from B/BR/BQ through to BP.
* **the number of metals a cell measured** (Spearman +0.49 with |a|) -- a magnitude model can
  look good simply by reproducing how many metals a laboratory bothered to measure.  Reported as
  the out-of-fold Spearman between the predicted magnitude and ``n_metals``, beside the Spearman
  between the predicted and the true magnitude.
* **chemotype** -- reported as leave-one-chemotype-out stability of the direction accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K                                   # noqa: E402
from common import V                                 # noqa: E402

OUT = K.OUT


def _sp(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def magnitude_confounds(rec: pd.DataFrame) -> pd.DataFrame:
    b = rec[rec.n_metals >= K.MIN_METALS].copy()
    b["mag_pred"] = b.a_pred.abs()
    b["mag_true"] = b.amp.abs()
    rows = []
    for (d, arm), g in b.groupby(["design", "arm"]):
        rows.append({"design": d, "arm": arm, "n_cells": len(g),
                     "sp_pred_vs_true_mag": _sp(g.mag_pred, g.mag_true),
                     "sp_pred_vs_n_metals": _sp(g.mag_pred, g.n_metals),
                     "sp_true_vs_n_metals": _sp(g.mag_true, g.n_metals),
                     "pred_mag_sd": float(g.mag_pred.std())})
    return pd.DataFrame(rows)


def direction_loco(rec: pd.DataFrame) -> pd.DataFrame:
    """Direction macro accuracy with one chemotype dropped at a time; min/max over the 45."""
    b = rec[rec.n_metals >= K.MIN_METALS].copy()
    b["hit"] = ((b.a_pred < 0).astype(int) == (b.amp < 0).astype(int)).astype(float)
    per_seed = b.groupby(["design", "arm", "split_seed", "extractant", "chemotype"])["hit"].mean()
    per_ext = per_seed.reset_index().groupby(["design", "arm", "extractant",
                                              "chemotype"])["hit"].mean().reset_index()
    rows = []
    for (d, arm), g in per_ext.groupby(["design", "arm"]):
        full = float(g.hit.mean())
        vals = [float(g[g.chemotype != c].hit.mean()) for c in sorted(set(g.chemotype))]
        rows.append({"design": d, "arm": arm, "dir_macro_acc": full,
                     "loco_min": float(np.min(vals)), "loco_max": float(np.max(vals)),
                     "n_chemotypes": int(g.chemotype.nunique())})
    return pd.DataFrame(rows)


def main() -> None:
    parts = []
    for i, f in enumerate(("cheap_predictions.csv.gz", "tabpfn_predictions.csv.gz",
                           "control_predictions.csv.gz", "tpcontrol_predictions.csv.gz")):
        p = OUT / f
        if p.exists():
            r = pd.read_csv(p)
            if i:
                r = r[~r.arm.isin(["FLAT", "G14"])]
            parts.append(r)
    rec = pd.concat(parts, ignore_index=True)
    rec = rec.drop_duplicates(subset=["design", "arm", "split_seed", "fold", "cell_index"],
                              keep="first")
    mc = magnitude_confounds(rec)
    lo = direction_loco(rec)
    mc.to_csv(OUT / "confound_magnitude.csv", index=False)
    lo.to_csv(OUT / "confound_direction_loco.csv", index=False)
    print("=== magnitude confounds, design BP ===")
    print(mc[mc.design == "BP"].round(4).to_string(index=False))
    print("\n=== direction accuracy, leave-one-chemotype-out, design BP ===")
    print(lo[lo.design == "BP"].round(4).to_string(index=False))
    print("\n=== magnitude confounds, all designs (pred vs n_metals) ===")
    print(mc.pivot(index="arm", columns="design",
                   values="sp_pred_vs_n_metals")[[d for d in V.DESIGNS
                                                  if d in set(mc.design)]].round(3).to_string())


if __name__ == "__main__":
    main()
