"""Paired inference with the held-out chemotype as the unit of independence.

Unit of scoring: the extractant (one vote each, its value averaged over split seeds).
Unit of resampling: the frozen chemotype; every member extractant travels with it.  One
index matrix of ``replicates`` block draws (seed 8675309) is shared by every comparison
in a call, and the percentile interval, the BCa interval (jackknife over blocks, the
repository's ``gen8.inference._bca_interval``) and the two-sided bootstrap p all come
from those same draws.

``passes_P1`` is the pre-registered rule in full: point >= margin (0.02), percentile
and BCa intervals exclude 0, p < 0.05, >= 4 of 5 seeds positive, and no single
chemotype flips the sign under leave-one-chemotype-out.  ``passes_intervals`` is the
interval/p part alone.  ``mde_80`` is the minimum detectable delta at 80 % power,
``2.80 * bootstrap sd`` (two-sided 5 %).

Higher-is-better metrics (sign accuracy, Spearman) are negated before the delta so that
"positive favours the candidate" holds for every value column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths  # noqa: F401
from lanthanide_separation.gen8.inference import _bca_interval  # noqa: E402

REPLICATES = 10_000
SEED = 8675309
MARGIN = 0.02
MIN_SEEDS_POSITIVE = 4
HIGHER_IS_BETTER: frozenset[str] = frozenset({"sign_acc_strong", "pair_spearman", "curve_spearman"})


def _per_unit(per_ext: pd.DataFrame, value: str) -> tuple[pd.DataFrame, pd.Series]:
    sign = -1.0 if value in HIGHER_IS_BETTER else 1.0
    table = (per_ext.assign(_v=sign * per_ext[value]).groupby(["arm", "extractant"])["_v"].mean().unstack("arm"))
    block_of = per_ext.drop_duplicates("extractant").set_index("extractant")["chemotype"].astype(str)
    return table, block_of.reindex(table.index)


def paired_contrasts(per_ext: pd.DataFrame, comparisons: dict[str, tuple[str, str]], *,
                     value: str = "mae_all", replicates: int = REPLICATES, seed: int = SEED,
                     margin: float = MARGIN) -> pd.DataFrame:
    """``reference - candidate`` per extractant on ``value``; positive = candidate better."""
    per_unit, block_of = _per_unit(per_ext, value)
    units = list(per_unit.index)
    names = sorted(set(block_of))
    if len(names) < 2:
        return pd.DataFrame()
    members = [np.array([i for i, u in enumerate(units) if block_of.iloc[i] == b], dtype=int) for b in names]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(names), size=(replicates, len(names)))
    take = [np.concatenate([members[j] for j in row]) for row in picks]
    seeds = per_ext["split_seed"].unique()
    rows = []
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
        bca_low, bca_high = _bca_interval(draws, point, delta, members)
        fin = draws[np.isfinite(draws)]
        p = float(2 * min((fin <= 0).mean(), (fin >= 0).mean()))
        se = float(fin.std(ddof=1))
        # per-seed sign agreement
        per_seed = []
        for s in seeds:
            sub = per_ext[per_ext["split_seed"] == s]
            sign = -1.0 if value in HIGHER_IS_BETTER else 1.0
            t = sub.assign(_v=sign * sub[value]).groupby(["arm", "extractant"])["_v"].mean().unstack("arm")
            if reference in t.columns and candidate in t.columns:
                d = (t[reference] - t[candidate]).dropna()
                per_seed.append(float(d.mean()))
        seeds_positive = int(sum(d > 0 for d in per_seed))
        # leave-one-chemotype-out sign stability
        loco = []
        for b, idx in zip(names, members):
            keep = np.ones(len(delta), dtype=bool); keep[idx] = False
            vals = delta[keep & finite]
            loco.append(float(vals.mean()) if vals.size else np.nan)
        loco = np.asarray(loco)
        loco_stable = bool(np.all(np.sign(loco[np.isfinite(loco)]) == np.sign(point))) if point != 0 else False
        block_values = [float(delta[idx][np.isfinite(delta[idx])].mean()) for idx in members if np.isfinite(delta[idx]).any()]
        passes_intervals = bool((np.nanquantile(draws, 0.025) > 0) and (bca_low > 0) and (p < 0.05))
        rows.append({
            "comparison": label, "value": value, "reference": reference, "candidate": candidate,
            "point": point, "ci95_low": float(np.nanquantile(draws, 0.025)), "ci95_high": float(np.nanquantile(draws, 0.975)),
            "bca_low": bca_low, "bca_high": bca_high, "p_two_sided": min(1.0, p), "bootstrap_se": se, "mde_80": 2.80 * se,
            "block_macro": float(np.mean(block_values)) if block_values else np.nan,
            "n_units": int(finite.sum()), "units_improved": int((delta[finite] > 0).sum()),
            "seeds_positive": seeds_positive, "n_seeds": len(per_seed),
            "loco_min": float(np.nanmin(loco)), "loco_max": float(np.nanmax(loco)), "loco_sign_stable": loco_stable,
            "margin": margin, "passes_intervals": passes_intervals,
            "passes_P1": bool(passes_intervals and point >= margin and seeds_positive >= MIN_SEEDS_POSITIVE and loco_stable),
        })
    return pd.DataFrame(rows)


def leave_one_chemotype_out(per_ext: pd.DataFrame, reference: str, candidate: str, *, value: str = "mae_all") -> pd.DataFrame:
    per_unit, block_of = _per_unit(per_ext, value)
    delta = (per_unit[reference] - per_unit[candidate]).dropna()
    blocks = block_of.reindex(delta.index)
    rows = [{"dropped_chemotype": "(none)", "delta": float(delta.mean()), "n_units": int(len(delta))}]
    for b in sorted(blocks.unique()):
        keep = blocks != b
        rows.append({"dropped_chemotype": b, "delta": float(delta[keep].mean()), "n_units": int(keep.sum())})
    return pd.DataFrame(rows)
