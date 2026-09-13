"""A fast bench for the one quantity that carries the error: the lanthanide-axis amplitude.

Stage-2 established that 82-87 % of the pairwise squared error of every Gen13 arm is error in the
coefficient of the standardised Shannon radius, and that a model which got that one number right
would score 0.21 extractant-macro MAE against the 0.48-0.54 the arms actually score.  It also
established that the only feature block set which transfers across laboratories is the compact
chemistry one (conditions + physchem + donors + coordination), and that the effective number of
training units for a chemotype-level scalar is about a dozen.

Every existing arm answers this with the same estimator: 400 extremely randomised trees on ~2 500
columns, refitted 25 times.  That is a poor match to a dozen effective units, and it takes minutes
per configuration, so the space of alternatives has never been searched.

This module makes the search cheap.  It freezes everything the arms freeze -- the cohort, the fold
plan, the physics basis, the per-cell ridge that turns a centred curve into coefficients, and the
extractant-macro scoring -- and exposes exactly one pluggable function:

    fit_predict(X_train, coef_train, weights_train, groups_train, X_test, seed) -> coef_test

so that a candidate is a few lines rather than an arm class.  A full 5-seed, 25-fold evaluation of a
kernel or linear model takes seconds; of a small forest, under a minute.  Scores are computed by
``gen13sep.metrics`` on byte-identical pairs, so a number from this bench is directly comparable
with the locked leaderboards.

Nothing here writes to the locked run directories, and no arm defined here can become the primary
endpoint: the bench exists to find candidates worth running through the real ladder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .basis import fit_all_coefficients, physics_basis_matrix
from .cohort import build_cohort
from .features import build_features
from .metals import ATOMIC_NUMBER, LANTHANIDES
from .metrics import pair_rows_for_cell, per_extractant, summarise
from .models import chemotype_balanced_weights
from .splits import all_folds

Z_VEC = np.array([ATOMIC_NUMBER[m] for m in LANTHANIDES])
LEAN_BLOCKS: tuple[str, ...] = ("COND", "MASSACT", "PHYSCHEM", "DONORS", "COORD")
CHEM_BLOCKS: tuple[str, ...] = ("PHYSCHEM", "DONORS", "COORD")
ALL_BLOCKS: tuple[str, ...] = ("COND", "MASSACT", "PHYSCHEM", "DONORS", "ECFP", "LIG2D", "COORD")

#: signature of a candidate: training features, training coefficients, per-cell weights, chemotype
#: labels, test features and the fold's model seed -> predicted test coefficients
FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class BenchData:
    frame: pd.DataFrame
    Y: np.ndarray                     # cells x 14 raw log D, NaN where unmeasured
    basis: np.ndarray                 # k x 14 physics basis, rows at norm sqrt(14)
    coef: np.ndarray                  # cells x k ridge coefficients of the centred curve
    n_obs: np.ndarray                 # metals observed per cell
    frames: dict                      # block name -> DataFrame of that block's columns
    groups: np.ndarray                # chemotype per cell

    def matrix(self, blocks: Sequence[str]) -> np.ndarray:
        return np.hstack([self.frames[b].to_numpy(dtype=float) for b in blocks])

    def columns(self, blocks: Sequence[str]) -> list[str]:
        out: list[str] = []
        for b in blocks:
            out.extend(self.frames[b].columns)
        return out


def load_bench(key_mode: str = "exact", basis_names: tuple[str, ...] = ("radius", "radius_sq"),
               blocks: Sequence[str] = ALL_BLOCKS) -> BenchData:
    """Build the frozen inputs once.  Takes about a minute; reuse the result for every candidate."""
    cohort = build_cohort(key_mode)
    features = build_features(cohort)
    frame = cohort.frame
    Y = cohort.target_matrix
    basis = physics_basis_matrix(basis_names)
    centred = Y - np.nanmean(Y, axis=1, keepdims=True)
    coef = fit_all_coefficients(centred, basis)
    frames = {b: features.matrix([b]) for b in blocks}
    return BenchData(frame=frame, Y=Y, basis=basis, coef=coef,
                     n_obs=(~np.isnan(Y)).sum(axis=1).astype(float),
                     frames=frames, groups=frame["chemotype"].to_numpy())


def cell_weights(groups: np.ndarray, n_obs: np.ndarray, *, mode: str = "balanced",
                 k: float = 2.0) -> np.ndarray:
    """Per-cell training weight.  ``balanced`` is the locked convention (equal weight per
    chemotype); ``reliability`` additionally down-weights cells whose coefficient is poorly
    determined, renormalised inside each chemotype so chemotype shares are untouched."""
    base = chemotype_balanced_weights(groups)
    if mode == "balanced":
        return base
    if mode != "reliability":
        raise ValueError(f"unknown weight mode {mode!r}")
    raw = base * (n_obs / (n_obs + k))
    per = pd.DataFrame({"g": groups, "base": base, "raw": raw}).groupby("g")[["base", "raw"]].transform("sum")
    return raw * (per["base"].to_numpy() / np.maximum(per["raw"].to_numpy(), 1e-12))


def _pair_frame(frame: pd.DataFrame, Y: np.ndarray, test_index: np.ndarray,
                seed: int, fold: int) -> pd.DataFrame:
    rows = []
    for local, ci in enumerate(test_index):
        for a, b, y in pair_rows_for_cell(Y[ci]):
            rows.append({"split_seed": seed, "fold": fold, "cell_local": local,
                         "cell_id": frame["cell_id"].iat[ci], "extractant": frame["extractant"].iat[ci],
                         "chemotype": frame["chemotype"].iat[ci],
                         "n_metals": int(frame["n_metals"].iat[ci]),
                         "A": LANTHANIDES[a], "B": LANTHANIDES[b], "ia": a, "ib": b,
                         "dZ": int(Z_VEC[b] - Z_VEC[a]), "y": y})
    return pd.DataFrame(rows)


def run_candidate(bench: BenchData, name: str, fit_predict: FitPredict, *,
                  blocks: Sequence[str] = LEAN_BLOCKS, design: str = "B",
                  seeds: Sequence[int] | None = None, weight_mode: str = "balanced") -> pd.DataFrame:
    """Score one candidate on the frozen fold plan; returns the long pair table with predictions."""
    X = bench.matrix(blocks)
    folds = all_folds(bench.frame, design=design, seeds=seeds) if seeds is not None \
        else all_folds(bench.frame, design=design)
    parts = []
    for f in folds:
        tr, te = f.train_index, f.test_index
        pairs = _pair_frame(bench.frame, bench.Y, te, f.seed, f.fold)
        if pairs.empty:
            continue
        # weights are built on the training subset, exactly as every locked arm does
        w_tr = cell_weights(bench.groups[tr], bench.n_obs[tr], mode=weight_mode)
        pred_coef = fit_predict(X[tr], bench.coef[tr], w_tr,
                                bench.groups[tr], X[te], f.model_seed)
        pred_coef = np.asarray(pred_coef, dtype=float).reshape(len(te), bench.basis.shape[0])
        curve = pred_coef @ bench.basis
        ia = pairs["ia"].to_numpy(); ib = pairs["ib"].to_numpy(); rows = pairs["cell_local"].to_numpy()
        t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
        t["prediction"] = curve[rows, ia] - curve[rows, ib]
        parts.append(t)
    table = pd.concat(parts, ignore_index=True)
    table = table.rename(columns={"prediction": name})
    return table


def score(table: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    return summarise(per_extractant(table, list(names)), table, list(names))


def compare(bench: BenchData, candidates: dict[str, FitPredict], *,
            blocks: Sequence[str] = LEAN_BLOCKS, design: str = "B",
            seeds: Sequence[int] | None = None, weight_mode: str = "balanced",
            block_overrides: dict[str, Sequence[str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run several candidates on identical folds and return (leaderboard, merged pair table)."""
    key = ["split_seed", "fold", "cell_id", "extractant", "chemotype", "n_metals", "A", "B", "dZ", "y"]
    merged: pd.DataFrame | None = None
    for name, fn in candidates.items():
        blk = (block_overrides or {}).get(name, blocks)
        t = run_candidate(bench, name, fn, blocks=blk, design=design, seeds=seeds,
                          weight_mode=weight_mode)
        merged = t if merged is None else merged.merge(t, on=key, how="inner", validate="one_to_one")
    names = list(candidates)
    return score(merged, names), merged
