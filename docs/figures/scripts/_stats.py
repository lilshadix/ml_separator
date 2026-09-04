"""Resampling helpers used by the figure scripts.

The paired machinery is the repository's own
``lanthanide_separation.gen8.inference.paired_chemotype_bootstrap`` — imported, not
re-implemented, so a figure and the decision reports use identical arithmetic.  The one
addition here is a *one-sample* chemotype-block bootstrap of a mean, which the reports
did not need (they only ever quote paired deltas) but a figure with error bands does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen8.inference import (  # noqa: E402
    DEFAULT_REPLICATES, DEFAULT_SEED, paired_chemotype_bootstrap,
)

__all__ = ["paired_chemotype_bootstrap", "block_bootstrap_mean", "block_bootstrap_stat",
           "DEFAULT_REPLICATES", "DEFAULT_SEED"]


def block_bootstrap_mean(values: pd.Series, blocks: pd.Series, *,
                         replicates: int = DEFAULT_REPLICATES,
                         seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """Mean of ``values`` with a 95 % CI from resampling whole ``blocks``.

    ``values`` is one number per experimental unit (extractant); ``blocks`` is that
    unit's Tanimoto-0.7 chemotype.  Blocks are resampled with replacement and every
    member of a drawn block is taken, which is the same block structure
    ``paired_chemotype_bootstrap`` uses, so a mean and a paired delta in the same
    figure are resampled the same way.
    """
    v = np.asarray(values, dtype=float)
    b = np.asarray(blocks).astype(str)
    finite = np.isfinite(v)
    v, b = v[finite], b[finite]
    names = sorted(set(b))
    members = [np.flatnonzero(b == name) for name in names]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(replicates, len(names)))
    draws = np.empty(replicates)
    for i, row in enumerate(picks):
        idx = np.concatenate([members[j] for j in row])
        draws[i] = v[idx].mean()
    return float(v.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def block_bootstrap_stat(values: pd.Series, blocks: pd.Series, statistic="mean", *,
                         replicates: int = DEFAULT_REPLICATES,
                         seed: int = DEFAULT_SEED) -> tuple[float, float, float]:
    """``block_bootstrap_mean`` for any statistic ("mean", "median" or a callable).

    Same block structure and same RNG stream, so a median and a mean in neighbouring
    panels are resampled identically.
    """
    fn = {"mean": np.mean, "median": np.median}.get(statistic, statistic)
    v = np.asarray(values, dtype=float)
    b = np.asarray(blocks).astype(str)
    finite = np.isfinite(v)
    v, b = v[finite], b[finite]
    names = sorted(set(b))
    members = [np.flatnonzero(b == name) for name in names]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(replicates, len(names)))
    draws = np.empty(replicates)
    for i, row in enumerate(picks):
        draws[i] = fn(v[np.concatenate([members[j] for j in row])])
    return float(fn(v)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
