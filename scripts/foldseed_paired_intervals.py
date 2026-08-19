"""Fold-seed-paired intervals for Experiment F (and a reusable helper).

The runner's per-checkpoint bootstrap pools every fold-seed's test rows into one
frame and resamples chemotype blocks — a row-weighted view in which the three
fold-seeds that hold out the diglycolamide chemotype (3,700–4,000 test rows each)
carry ~75 % of every interval.  That is a legitimate reading but not the only one,
and for one comparison (novelty × uncertainty vs max-min) the two weightings
disagree in sign.  This script computes the complementary view: **one unit = one
fold-seed**, each weighted equally, the policy difference paired within the
fold-seed and replicate-averaged, with a percentile and a BCa interval over
fold-seeds.  Both views are reported in the results document; a claim is called
established only where they agree.

Usage::

    .venv/bin/python scripts/foldseed_paired_intervals.py runs/gen6_expF_3seed
    .venv/bin/python scripts/foldseed_paired_intervals.py runs/gen6_expF_3seed_rowweight
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

COMPARISONS = (
    ("maxmin", "random"), ("diversity_x_uncertainty", "random"), ("offset_uncertainty", "random"),
    ("uncertainty", "random"), ("same_chemotype_first", "random"),
    ("diversity_x_uncertainty", "maxmin"), ("offset_uncertainty", "maxmin"),
)
STATISTICS = {"macro_mae": "macro_mae", "hard": "hard_nn0.4__macro_mae", "offset": "offset_mae"}


def bca_interval(values: np.ndarray, *, replicates: int = 4000, seed: int = 8675309) -> tuple[float, float, float, float]:
    """(percentile low, high, BCa low, high) for the mean of a paired-difference vector."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.array([x[rng.integers(0, n, n)].mean() for _ in range(replicates)])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    theta = x.mean()
    from scipy.stats import norm
    z0 = norm.ppf((draws < theta).mean().clip(1e-6, 1 - 1e-6))
    jack = np.array([np.delete(x, i).mean() for i in range(n)])
    diff = jack.mean() - jack
    denom = 6.0 * (diff ** 2).sum() ** 1.5
    a = (diff ** 3).sum() / denom if denom > 0 else 0.0
    def adj(alpha):
        z = norm.ppf(alpha)
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    blo, bhi = np.quantile(draws, [adj(0.025), adj(0.975)])
    return (float(lo), float(hi), float(blo), float(bhi))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[5, 8, 12, 16, 20, 25, 30])
    p.add_argument("--exclude-short-starts", action="store_true",
                   help="drop fold-seeds whose start cohort is short (the DGA-holdout folds)")
    args = p.parse_args(argv)
    curves = pd.read_csv(args.run_dir / "acquisition_curves.csv")
    curves = curves[curves["budget_per_ligand"].astype(str) == "3"]
    if args.exclude_short_starts and (args.run_dir / "start_cohorts.csv").exists():
        starts = pd.read_csv(args.run_dir / "start_cohorts.csv")
        short = set(map(tuple, starts.loc[starts["start_is_short"], ["split_seed", "fold"]].to_numpy()))
        curves = curves[[(s, f) not in short for s, f in zip(curves["split_seed"], curves["fold"])]]
    # replicate-mean per (seed, fold, policy, checkpoint)
    keys = ["split_seed", "fold", "policy", "checkpoint"]
    stat_cols = [c for c in STATISTICS.values() if c in curves.columns]
    mean = curves.groupby(keys)[stat_cols].mean().reset_index()
    rows = []
    for cand, ref in COMPARISONS:
        for k in args.checkpoints:
            a = mean[(mean.policy == cand) & (mean.checkpoint == k)].set_index(["split_seed", "fold"])
            b = mean[(mean.policy == ref) & (mean.checkpoint == k)].set_index(["split_seed", "fold"])
            if a.empty or b.empty:
                continue
            joined = b.join(a, lsuffix="_ref", rsuffix="_cand", how="inner")
            for name, col in STATISTICS.items():
                if col not in stat_cols:
                    continue
                delta = (joined[f"{col}_ref"] - joined[f"{col}_cand"]).to_numpy()
                lo, hi, blo, bhi = bca_interval(delta)
                rows.append({"comparison": f"{cand}_vs_{ref}", "checkpoint": k, "statistic": name,
                             "n_foldseeds": int(np.isfinite(delta).sum()),
                             "point_delta": float(np.nanmean(delta)),
                             "n_positive": int((delta > 0).sum()),
                             "ci95_low": lo, "ci95_high": hi, "bca_low": blo, "bca_high": bhi})
    out = pd.DataFrame(rows)
    suffix = "_excl_short" if args.exclude_short_starts else ""
    out.to_csv(args.run_dir / f"foldseed_paired_contrasts{suffix}.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
