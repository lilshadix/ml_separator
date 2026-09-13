"""Gen15 bench: score *any* curve predictor on the programme's endpoint, under all five designs.

Gen14 closed the direction question and localised every remaining error in the magnitude of the
radius coefficient.  Its bench could only express a candidate as ``(probability heavy, magnitude)``,
which is the right object for the direction and the wrong one for anything that wants to look at the
conditions, the level of log D, the measured metal set, or a second basis coefficient.

Here a candidate is the general thing:

    arm(ctx) -> coef  of shape (n_test, k)      # k = 2, the radius and radius-squared coefficients

with ``ctx`` carrying the whole fold -- the training and test indices, every feature block, the
observed log D matrix, the frozen weights -- so an arm can be a two-stage model, an oracle, or a
model that uses a quantity only some deployments have.  Scoring is the frozen gen13 pipeline
(``metrics.per_extractant`` / ``summarise``) on byte-identical pairs, and the paired inference is
gen13's chemotype-blocked bootstrap, so a number here is comparable with every locked table.

Nothing here writes into a locked run directory.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gen13sep.amplitude_bench import (ALL_BLOCKS, LEAN_BLOCKS, BenchData,  # noqa: E402
                                      _pair_frame, cell_weights)
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import load, feature_sets  # noqa: E402

DESIGNS: tuple[str, ...] = ("B", "BR", "BQ", "A", "BP")
MIN_METALS = 5
RESULTS = ROOT / "generations" / "gen15_curve" / "results"


@dataclass
class Ctx:
    """Everything one fold of one design makes available to an arm."""
    bench: BenchData
    design: str
    seed: int
    fold: int
    train: np.ndarray            # training cell indices (already masked by the design)
    test: np.ndarray             # held-out cell indices
    w: np.ndarray                # chemotype-balanced weight of every training cell
    model_seed: int
    fs: dict                     # named column-index sets over the LEAN block matrix
    X: np.ndarray                # LEAN block matrix, all cells
    rich: np.ndarray             # boolean, cell curve well determined (>= 5 metals)

    # --- convenience -------------------------------------------------------------------
    @property
    def amp(self) -> np.ndarray:
        return self.bench.coef[:, 0]

    @property
    def cur(self) -> np.ndarray:
        return self.bench.coef[:, 1]

    def feat(self, name: str) -> np.ndarray:
        return self.X[:, self.fs[name]]

    def rich_train(self) -> np.ndarray:
        return self.train[self.rich[self.train]]

    def rich_weights(self) -> np.ndarray:
        rtr = self.rich_train()
        return cell_weights(self.bench.groups[rtr], self.bench.n_obs[rtr])

    def train_mean_curvature(self) -> float:
        return float(np.average(self.cur[self.train], weights=self.w))

    def train_mean_amplitude(self) -> float:
        return float(np.average(self.amp[self.train], weights=self.w))

    def train_mean_magnitude(self) -> float:
        rtr = self.rich_train()
        return float(np.average(np.abs(self.amp[rtr]), weights=self.rich_weights()))


Arm = Callable[[Ctx], np.ndarray]


def run_arms(bench: BenchData, arms: dict[str, Arm], design: str, *,
             seeds: Sequence[int] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Fit every arm on every fold of ``design`` and return the long pair table."""
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    folds = all_folds(bench.frame, design=design) if seeds is None \
        else all_folds(bench.frame, design=design, seeds=seeds)
    parts, timing = [], {k: 0.0 for k in arms}
    for f in folds:
        pairs = _pair_frame(bench.frame, bench.Y, f.test_index, f.seed, f.fold)
        if pairs.empty:
            continue
        ctx = Ctx(bench=bench, design=design, seed=f.seed, fold=f.fold, train=f.train_index,
                  test=f.test_index, w=cell_weights(bench.groups[f.train_index],
                                                    bench.n_obs[f.train_index]),
                  model_seed=f.model_seed, fs=fs, X=X, rich=rich)
        ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
        t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
        for name, arm in arms.items():
            t0 = time.time()
            coef = np.asarray(arm(ctx), dtype=float).reshape(len(f.test_index), bench.basis.shape[0])
            curve = coef @ bench.basis
            t[name] = curve[loc, ia] - curve[loc, ib]
            timing[name] += time.time() - t0
        parts.append(t)
    if verbose:
        slow = sorted(timing.items(), key=lambda kv: -kv[1])[:4]
        print("   [%s] %s" % (design, "  ".join(f"{k} {v:.1f}s" for k, v in slow)), flush=True)
    return pd.concat(parts, ignore_index=True)


def board(table: pd.DataFrame, arms: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-extractant table and the summary board, frozen gen13 metrics."""
    pe = per_extractant(table, list(arms))
    return pe, summarise(pe, table, list(arms))


def contrasts(pe: pd.DataFrame, comps: dict[str, tuple[str, str]], *, value: str = "mae_all",
              replicates: int = 10_000) -> pd.DataFrame:
    return paired_contrasts(pe, comps, value=value, replicates=replicates)


def score(bench: BenchData, arms: dict[str, Arm], designs: Sequence[str] = DESIGNS, *,
          comps: dict[str, tuple[str, str]] | None = None,
          value: str = "mae_all", seeds: Sequence[int] | None = None,
          verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The five-design table the evaluation protocol requires, in one call."""
    boards, cons, tables = [], [], {}
    for d in designs:
        t0 = time.time()
        table = run_arms(bench, arms, d, seeds=seeds, verbose=verbose)
        pe, bd = board(table, list(arms))
        bd.insert(0, "design", d)
        boards.append(bd)
        tables[d] = (table, pe)
        if comps:
            c = contrasts(pe, comps, value=value)
            if len(c):
                c.insert(0, "design", d)
                cons.append(c)
        if verbose:
            print(f"  design {d} done in {time.time() - t0:.0f}s", flush=True)
    B = pd.concat(boards, ignore_index=True)
    C = pd.concat(cons, ignore_index=True) if cons else pd.DataFrame()
    return B, C, tables


def wide(B: pd.DataFrame, value: str = "macro_mae_extractant") -> pd.DataFrame:
    """arm x design matrix of one summary column, in the canonical design order."""
    w = B.pivot(index="arm", columns="design", values=value)
    return w[[d for d in DESIGNS if d in w.columns]]
