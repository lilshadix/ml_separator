"""Shared plumbing for the gen15 `tabpfn` experiment.

Everything here is deliberately small and explicit:

* ``fold_data`` pulls the four things every learner in this experiment needs out of a
  ``valuebench.Ctx`` -- the rich (>= 5 metals) training rows gen14 uses for the direction, the
  chemotype-balanced weights, the three targets (direction bit, log magnitude, curvature) and the
  held-out rows.
* ``balanced_resample`` turns those weights into duplicated rows, because TabPFN has no
  ``sample_weight``.  Systematic (not multinomial) resampling, so it is deterministic and adds no
  Monte-Carlo noise on top of the fold plan.
* ``inner_folds`` re-deals the repository's own chemotype-grouped splitter *inside* the training
  fold, which is the only place any hyperparameter in this experiment is allowed to be chosen.
* ``Recorder`` collects the predicted radius coefficient of every held-out cell so that the
  direction macro accuracy (gen14's yardstick: 0.821 under BP) can be computed from the same run
  that produces the MAE table, with no second pass over the folds.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
for p in (ROOT / "generations" / "gen15_curve", ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gen15 import valuebench as V  # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS  # noqa: E402

from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402

EPS = 0.05            # gen15's own floor inside log|a|, kept identical
MIN_METALS = 5
OUT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
@dataclass
class FoldData:
    rtr: np.ndarray          # rich training cell indices
    w: np.ndarray            # chemotype-balanced weights over rtr
    groups: np.ndarray       # chemotype of every rich training cell
    y_dir: np.ndarray        # 1 = heavy-selective (a < 0)
    y_logmag: np.ndarray     # log(|a| + EPS)
    y_curv: np.ndarray       # b
    test: np.ndarray


def fold_data(ctx) -> FoldData:
    rtr = ctx.rich_train()
    w = ctx.rich_weights()
    a = ctx.amp[rtr]
    return FoldData(rtr=rtr, w=w, groups=ctx.bench.groups[rtr],
                    y_dir=(a < 0).astype(int), y_logmag=np.log(np.abs(a) + EPS),
                    y_curv=ctx.cur[rtr], test=ctx.test)


def balanced_resample(w: np.ndarray, n: int | None = None) -> np.ndarray:
    """Systematic resampling: deterministic row indices whose empirical law is ``w``."""
    w = np.asarray(w, dtype=float)
    n = len(w) if n is None else int(n)
    c = np.cumsum(w) / w.sum()
    u = (np.arange(n) + 0.5) / n
    return np.clip(np.searchsorted(c, u), 0, len(w) - 1)


def inner_folds(groups: np.ndarray, seed: int, n_splits: int = 4):
    """The repository's chemotype-grouped splitter, re-dealt inside a training fold."""
    return list(seeded_group_kfold(np.asarray(groups, dtype=object), n_splits, int(seed)))


def median_impute(Xtr: np.ndarray, *others: np.ndarray):
    """Median imputation fitted on the training rows only (gen14's convention)."""
    med = np.nanmedian(np.where(np.isfinite(Xtr), Xtr, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def fix(X):
        X = np.asarray(X, dtype=float).copy()
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(med, np.nonzero(bad)[1])
        return X
    return (fix(Xtr), *(fix(o) for o in others))


# --------------------------------------------------------------------------------------
class Recorder:
    """Per-arm log of the predicted radius coefficient of every held-out cell."""

    def __init__(self):
        self.rows: list[dict] = []

    def log(self, ctx, name: str, a_pred: np.ndarray) -> None:
        fr = ctx.bench.frame
        for j, ci in enumerate(ctx.test):
            self.rows.append({"design": ctx.design, "arm": name, "split_seed": ctx.seed,
                              "fold": ctx.fold, "cell_index": int(ci),
                              "extractant": fr.extractant.iat[ci],
                              "chemotype": fr.chemotype.iat[ci],
                              "n_metals": int(fr.n_metals.iat[ci]),
                              "amp": float(ctx.amp[ci]), "a_pred": float(a_pred[j])})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def direction_board(rec: pd.DataFrame) -> pd.DataFrame:
    """Gen14's direction yardstick: macro accuracy over extractants, well-determined cells only.

    One vote per extractant, averaged over split seeds first, exactly as
    ``gen14.dirbench.unit_hits`` does, so 0.821 (gen14 logistic under BP) and 0.559 (always-heavy)
    are directly comparable numbers.
    """
    b = rec[rec.n_metals >= MIN_METALS].copy()
    b["hit"] = ((b.a_pred < 0).astype(int) == (b.amp < 0).astype(int)).astype(float)
    per_seed = b.groupby(["design", "arm", "split_seed", "extractant"])["hit"].mean().reset_index()
    per_ext = per_seed.groupby(["design", "arm", "extractant"])["hit"].mean().reset_index()
    out = per_ext.groupby(["design", "arm"])["hit"].mean().reset_index()
    out = out.rename(columns={"hit": "dir_macro_acc"})
    n = per_ext.groupby(["design", "arm"])["extractant"].nunique().reset_index(name="n_units")
    return out.merge(n, on=["design", "arm"])


def wide_dir(board: pd.DataFrame) -> pd.DataFrame:
    w = board.pivot(index="arm", columns="design", values="dir_macro_acc")
    return w[[d for d in V.DESIGNS if d in w.columns]]
