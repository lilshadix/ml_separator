"""Paired block bootstrap, power, and influence diagnostics.

The resampling unit is the **chemotype**, never the row and never the extractant.
Two extractants inside one held-out Tanimoto super-cluster were held out together
and their errors are not independent; the repository measured that resampling the
wrong unit understates the interval by a factor of 1.65 on this corpus.

Every comparison is paired: the arms predict the same held-out extractants, one
index matrix is drawn and shared by every comparison and every statistic, so the
intervals are mutually comparable and do not depend on dictionary order.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

BOOTSTRAP_SEED = 8675309
REPLICATES = 10_000


def _matched(per_unit: pd.DataFrame, arms: Sequence[str], statistic: str) -> pd.DataFrame:
    if len(set(arms)) != len(arms):
        raise ValueError(f"a comparison needs two distinct arms, got {list(arms)}")
    wide = per_unit.pivot_table(index="unit", columns="arm", values=statistic, aggfunc="first")
    missing = [a for a in arms if a not in wide.columns]
    if missing:
        raise KeyError(f"arms absent from the per-unit table: {missing}")
    return wide[list(arms)]


def unit_table(per_extractant_frames: Mapping[str, pd.DataFrame],
               *, statistic: str = "mae") -> pd.DataFrame:
    """Collapse per-(seed, extractant) tables to one value per (arm, extractant).

    Averaging over seeds *before* the bootstrap is deliberate: the five split
    seeds re-partition the same 183 extractants, so they are not five independent
    experiments.  Per-seed direction is reported separately as a consistency
    check, never as evidence of significance.
    """
    rows: list[pd.DataFrame] = []
    for arm, frame in per_extractant_frames.items():
        block = (frame.groupby("extractant")
                 .agg(value=(statistic, "mean"), chemotype=("chemotype", "first"),
                      n_rows=("n_rows", "mean"), n_seeds=("split_seed", "nunique"))
                 .reset_index())
        block["arm"] = arm
        rows.append(block.rename(columns={"extractant": "unit", "value": statistic}))
    out = pd.concat(rows, ignore_index=True)
    counts = out.groupby("arm")["unit"].nunique()
    if counts.nunique() != 1:
        raise ValueError(f"arms do not share their evaluation units: {counts.to_dict()}")
    return out


def paired_bootstrap(per_unit: pd.DataFrame, comparisons: Mapping[str, tuple[str, str]],
                     *, statistic: str = "mae", replicates: int = REPLICATES,
                     seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """``delta = reference - candidate``; positive means the candidate is better.

    Chemotype blocks are resampled with replacement and every member extractant
    travels with its block.  BCa is reported alongside the percentile interval
    because the percentile interval is anti-conservative for a skewed delta.
    """
    units = sorted(per_unit["unit"].unique())
    block_of = (per_unit.drop_duplicates("unit").set_index("unit")["chemotype"]
                .astype(str).to_dict())
    blocks = sorted({block_of[u] for u in units})
    members = {b: np.array([i for i, u in enumerate(units) if block_of[u] == b]) for b in blocks}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(blocks), size=(replicates, len(blocks)))
    take = [np.concatenate([members[blocks[j]] for j in row]) for row in picks]

    records: list[dict] = []
    for label, (reference, candidate) in comparisons.items():
        wide = _matched(per_unit, [reference, candidate], statistic).reindex(units)
        delta = (wide[reference] - wide[candidate]).to_numpy(dtype=float)
        finite = np.isfinite(delta)
        if not finite.any():
            continue
        filled = np.where(finite, delta, 0.0)
        counts = finite.astype(float)
        draws = np.array([filled[t].sum() / max(counts[t].sum(), 1e-12) for t in take])
        point = float(delta[finite].mean())
        low, high = np.quantile(draws, [0.025, 0.975])
        records.append({
            "comparison": label, "reference": reference, "candidate": candidate,
            "statistic": statistic, "point_delta": point,
            "ci95_low": float(low), "ci95_high": float(high),
            **_bca(delta, finite, members, blocks, draws, point),
            "bootstrap_se": float(draws.std(ddof=1)),
            "p_two_sided": float(2 * min((draws <= 0).mean(), (draws >= 0).mean())),
            "units_improved": int((delta[finite] > 0).sum()),
            "units_total": int(finite.sum()),
            "bootstrap_blocks": len(blocks), "replicates": replicates,
        })
    return pd.DataFrame(records)


def _bca(delta: np.ndarray, finite: np.ndarray, members: Mapping[str, np.ndarray],
         blocks: Sequence[str], draws: np.ndarray, point: float) -> dict:
    """Bias-corrected and accelerated interval, jackknifed over blocks."""
    from scipy.stats import norm
    share = float((draws < point).mean())
    share = min(max(share, 1.0 / len(draws)), 1.0 - 1.0 / len(draws))
    z0 = float(norm.ppf(share))
    filled = np.where(finite, delta, 0.0)
    counts = finite.astype(float)
    jack = []
    for held in blocks:
        keep = np.concatenate([members[b] for b in blocks if b != held]) if len(blocks) > 1 else None
        if keep is None or counts[keep].sum() == 0:
            continue
        jack.append(filled[keep].sum() / counts[keep].sum())
    jack = np.asarray(jack, dtype=float)
    if len(jack) < 3:
        return {"bca_low": np.nan, "bca_high": np.nan, "bca_excludes_zero": False}
    centred = jack.mean() - jack
    denom = 6.0 * (float((centred ** 2).sum()) ** 1.5)
    acceleration = float((centred ** 3).sum()) / denom if denom > 0 else 0.0
    out = {}
    for name, alpha in (("bca_low", 0.025), ("bca_high", 0.975)):
        z = norm.ppf(alpha)
        adjusted = z0 + (z0 + z) / max(1e-12, (1 - acceleration * (z0 + z)))
        out[name] = float(np.quantile(draws, float(norm.cdf(adjusted))))
    out["bca_excludes_zero"] = bool(out["bca_low"] > 0 or out["bca_high"] < 0)
    return out


def minimum_detectable_effect(per_unit: pd.DataFrame, reference: str, candidate: str,
                              *, statistic: str = "mae", power: float = 0.80,
                              replicates: int = REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict:
    """The smallest paired difference this design could resolve, from the real SE.

    Computed from a *null-centred* bootstrap of the observed paired difference:
    ``MDE = (z_{1-a/2} + z_power) * SE``.  Reported before any comparison is
    interpreted, so a non-significant result can be read as "underpowered" or
    "small" rather than "equivalent".
    """
    from scipy.stats import norm
    table = paired_bootstrap(per_unit, {"mde": (reference, candidate)},
                             statistic=statistic, replicates=replicates, seed=seed)
    se = float(table["bootstrap_se"].iloc[0])
    return {"reference": reference, "candidate": candidate, "statistic": statistic,
            "bootstrap_se": se, "power": power,
            "mde": float((norm.ppf(0.975) + norm.ppf(power)) * se),
            "n_blocks": int(table["bootstrap_blocks"].iloc[0]),
            "n_units": int(table["units_total"].iloc[0])}


def leave_one_chemotype_out(per_unit: pd.DataFrame, reference: str, candidate: str,
                            *, statistic: str = "mae") -> pd.DataFrame:
    """Recompute the paired delta with each chemotype removed in turn.

    A result that one large chemical family determines is a result about that
    family.  On this cohort the largest chemotype holds half the rows, so this
    diagnostic is mandatory rather than optional.
    """
    units = sorted(per_unit["unit"].unique())
    block_of = per_unit.drop_duplicates("unit").set_index("unit")["chemotype"].astype(str)
    wide = _matched(per_unit, [reference, candidate], statistic).reindex(units)
    delta = (wide[reference] - wide[candidate])
    full = float(delta.mean(skipna=True))
    rows = []
    for chemotype in sorted(set(block_of)):
        keep = [u for u in units if block_of[u] != chemotype]
        held = [u for u in units if block_of[u] == chemotype]
        rows.append({"chemotype_removed": chemotype, "n_units_removed": len(held),
                     "delta_without": float(delta.loc[keep].mean(skipna=True)),
                     "delta_full": full,
                     "influence": float(delta.loc[keep].mean(skipna=True) - full)})
    return pd.DataFrame(rows).sort_values("influence", key=np.abs, ascending=False,
                                          ignore_index=True)
