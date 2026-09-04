"""Paired inference with the held-out chemotype as the unit of independence.

The scoring unit in this project is the ligand (or its ECFP cluster), but the
*independence* unit is the Tanimoto chemotype the fold plan actually held out:
two ligands in the same chemotype were held out together, trained against the same
reduced training set, and their errors are correlated.  Resampling ligands rather
than chemotypes understates the width — measured on this cohort at 1.65x — so
every interval here resamples **chemotypes**, and every member ligand travels with
its chemotype.

For acquisition comparisons the pairing is stricter still and is enforced by the
caller: the same ligand, the same repeat, the same candidate pool and the same
evaluation rows must appear on both sides, so that a difference between two
policies cannot be a difference in what they were scored on.

Reports, per comparison: the point estimate, a percentile interval, a BCa interval
(bias-corrected and accelerated, which matters here because the per-ligand delta
distribution is strongly right-skewed), the block macro (chemotype-weighted rather
than ligand-weighted), how many units improved, and the per-seed agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_REPLICATES = 5000
DEFAULT_SEED = 8675309


def _bca_interval(draws: np.ndarray, point: float, values: np.ndarray,
                  blocks: Sequence[np.ndarray]) -> tuple[float, float]:
    """Bias-corrected and accelerated interval; jackknife over the blocks."""
    from scipy.stats import norm

    finite = draws[np.isfinite(draws)]
    if finite.size < 50:
        return float("nan"), float("nan")
    proportion = float(np.mean(finite < point))
    proportion = min(max(proportion, 1e-6), 1 - 1e-6)
    z0 = float(norm.ppf(proportion))

    means = []
    for j in range(len(blocks)):
        keep = np.concatenate([blocks[i] for i in range(len(blocks)) if i != j]) \
            if len(blocks) > 1 else blocks[0]
        subset = values[keep]
        subset = subset[np.isfinite(subset)]
        means.append(subset.mean() if subset.size else np.nan)
    means = np.asarray(means, dtype=float)
    means = means[np.isfinite(means)]
    if means.size < 3:
        return float("nan"), float("nan")
    centred = means.mean() - means
    denominator = 6.0 * (float((centred ** 2).sum()) ** 1.5)
    acceleration = float((centred ** 3).sum()) / denominator if denominator > 0 else 0.0

    out = []
    for alpha in (0.025, 0.975):
        z = norm.ppf(alpha)
        adjusted = z0 + (z0 + z) / max(1e-9, (1 - acceleration * (z0 + z)))
        out.append(float(np.nanquantile(finite, float(norm.cdf(adjusted)))))
    return out[0], out[1]


@dataclass(frozen=True)
class PairedResult:
    comparison: str
    point: float
    ci_low: float
    ci_high: float
    bca_low: float
    bca_high: float
    block_macro: float
    n_units: int
    units_improved: int
    seeds_positive: int
    n_seeds: int

    def as_dict(self) -> dict:
        return {"comparison": self.comparison, "point": self.point,
                "ci95_low": self.ci_low, "ci95_high": self.ci_high,
                "bca_low": self.bca_low, "bca_high": self.bca_high,
                "block_macro": self.block_macro, "n_units": self.n_units,
                "units_improved": self.units_improved,
                "seeds_positive": self.seeds_positive, "n_seeds": self.n_seeds}


def paired_chemotype_bootstrap(
    detail: pd.DataFrame,
    comparisons: Mapping[str, tuple[str, str]],
    *,
    arm_column: str = "arm",
    unit_column: str = "extractant",
    block_column: str = "tanimoto_cluster",
    seed_column: str = "split_seed",
    value_column: str = "mae",
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """``reference - candidate`` per unit, resampled over blocks.  Positive = better.

    ``detail`` is long: one row per (arm, unit, ...) observation of ``value_column``.
    Observations are averaged to one number per (arm, unit) before the bootstrap, so
    a ligand measured under twelve repeats does not become twelve independent units.
    """
    if detail.empty:
        raise ValueError("detail is empty")
    per_unit = detail.groupby([arm_column, unit_column])[value_column].mean().unstack(arm_column)
    block_of = detail.drop_duplicates(unit_column).set_index(unit_column)[block_column]
    units = list(per_unit.index)
    block_names = sorted(set(block_of.reindex(units).astype(str)))
    members = {b: np.array([i for i, u in enumerate(units) if str(block_of.get(u)) == b], dtype=int)
               for b in block_names}
    member_index = [members[b] for b in block_names]

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(block_names), size=(replicates, len(block_names)))
    take = [np.concatenate([member_index[j] for j in row]) for row in picks]

    per_seed = None
    if seed_column in detail.columns:
        per_seed = detail.groupby([arm_column, seed_column])[value_column].mean()

    rows: list[dict] = []
    for label, (reference, candidate) in comparisons.items():
        if reference not in per_unit.columns or candidate not in per_unit.columns:
            continue
        delta = (per_unit[reference] - per_unit[candidate]).to_numpy(dtype=float)
        finite = np.isfinite(delta)
        if finite.sum() < 3:
            continue
        filled = np.where(finite, delta, 0.0)
        counts = finite.astype(float)
        sums = np.array([filled[t].sum() for t in take])
        ns = np.array([counts[t].sum() for t in take])
        draws = np.divide(sums, ns, out=np.full_like(sums, np.nan), where=ns > 0)
        point = float(delta[finite].mean())
        bca_low, bca_high = _bca_interval(draws, point, delta, member_index)

        block_values = []
        for indices in member_index:
            values = delta[indices]
            values = values[np.isfinite(values)]
            if values.size:
                block_values.append(values.mean())
        seeds_positive, n_seeds = 0, 0
        if per_seed is not None and reference in per_seed.index.get_level_values(0) \
                and candidate in per_seed.index.get_level_values(0):
            a, b = per_seed.loc[reference], per_seed.loc[candidate]
            shared = a.index.intersection(b.index)
            n_seeds = len(shared)
            seeds_positive = int(((a.loc[shared] - b.loc[shared]) > 0).sum())
        rows.append(PairedResult(
            comparison=label, point=point,
            ci_low=float(np.nanquantile(draws, 0.025)),
            ci_high=float(np.nanquantile(draws, 0.975)),
            bca_low=bca_low, bca_high=bca_high,
            block_macro=float(np.mean(block_values)) if block_values else float("nan"),
            n_units=int(finite.sum()), units_improved=int((delta[finite] > 0).sum()),
            seeds_positive=seeds_positive, n_seeds=n_seeds).as_dict())
    return pd.DataFrame(rows)
