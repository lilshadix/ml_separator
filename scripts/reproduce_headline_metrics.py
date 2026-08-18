#!/usr/bin/env python
"""Recompute the headline pair-model metrics from a committed OOF file.

This is a deliberately *independent*, dependency-light re-implementation of the
metric formulas (pandas/numpy only, no project imports) so that a reader can
check the numbers in ``docs/metrics_reproduction_20260818.md`` without
retraining anything.  It reads ``oof_predictions.csv`` from a gen4 run,
recomputes every headline statistic per split seed, averages over seeds the
way ``scripts/run_gen4_candidates.py::aggregate`` does, and prints the result
next to the committed ``leaderboard.csv`` values.

Usage::

    python scripts/reproduce_headline_metrics.py \
        runs/gen4_candidates_20260817T003449Z/oof_predictions.csv \
        --arms A2_refit A2_refit_TP PAIRMEAN_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "log_SF_A_over_B"
R2_MIN_ROWS = 8  # gen3_metrics.R2_MIN_GROUP_ROWS
R2_MIN_STD = 0.1  # gen3_metrics.R2_MIN_GROUP_TARGET_STD


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot if ss_tot > 0 else float("nan")


def guarded_r2(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < R2_MIN_ROWS or float(np.std(y)) < R2_MIN_STD:
        return float("nan")
    return r2(y, p)


def metrics_one_seed(block: pd.DataFrame, arm: str) -> dict[str, float]:
    y = block[TARGET].to_numpy(dtype=float)
    p = block[f"prediction_{arm}"].to_numpy(dtype=float)
    err = np.abs(y - p)

    # (1) equal-extractant macro MAE: MAE inside each extractant, then an
    #     UNWEIGHTED mean over extractants (34 of them).  Primary metric.
    per_ext_mae = pd.Series(err).groupby(block["extractant"].to_numpy()).mean()
    # (2) pooled (micro) MAE: plain mean over all 6,699 rows.
    # (3) pooled R2 over all rows.
    # (4) sign accuracy over all rows.
    # (5) guarded per-extractant R2 -> median / mean over eligible extractants.
    per_ext_r2 = pd.Series(
        {
            e: guarded_r2(g[TARGET].to_numpy(dtype=float), g[f"prediction_{arm}"].to_numpy(dtype=float))
            for e, g in block.groupby("extractant", sort=False)
        }
    ).dropna()
    adjacent = (block["pair__Z_B"].to_numpy(dtype=float) - block["pair__Z_A"].to_numpy(dtype=float)) == 1.0
    return {
        "equal_extractant_macro_mae": float(per_ext_mae.mean()),
        "pooled_micro_mae": float(err.mean()),
        "pooled_r2": r2(y, p),
        "sign_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
        "median_extractant_r2": float(per_ext_r2.median()),
        "extractants_in_r2_metrics": float(len(per_ext_r2)),
        "adjacent_ln_mae": float(err[adjacent].mean()),
        "nonadjacent_ln_mae": float(err[~adjacent].mean()),
        "prediction_dispersion_ratio": float(np.std(p) / np.std(y)),
    }


def paired_extractant_bootstrap(
    oof: pd.DataFrame, reference: str, candidate: str, replicates: int, seed: int
) -> dict[str, float]:
    """Paired cluster bootstrap over extractants on the pooled 5-seed OOF frame."""
    y = oof[TARGET].to_numpy(dtype=float)
    ref_err = np.abs(y - oof[f"prediction_{reference}"].to_numpy(dtype=float))
    cand_err = np.abs(y - oof[f"prediction_{candidate}"].to_numpy(dtype=float))
    ext = oof["extractant"].to_numpy()
    order = pd.unique(ext)  # first-appearance order, as in gen3_metrics
    delta = np.array(
        [ref_err[ext == e].mean() - cand_err[ext == e].mean() for e in order], dtype=float
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for i in range(replicates):
        idx = rng.integers(0, len(order), size=len(order))
        draws[i] = delta[idx].mean()
    return {
        "point_delta_mae": float(delta.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "p_candidate_better": float(np.mean(draws > 0)),
        "extractants_improved": int(np.sum(delta > 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oof", type=Path, help="gen4 oof_predictions.csv (all split seeds stacked)")
    ap.add_argument("--arms", nargs="+", default=["A2_refit", "A2_refit_TP", "PAIRMEAN_baseline"])
    ap.add_argument("--baseline", default="A2_refit")
    ap.add_argument("--replicates", type=int, default=10_000)
    ap.add_argument("--bootstrap-seed", type=int, default=8675309)
    args = ap.parse_args()

    oof = pd.read_csv(args.oof)
    seeds = sorted(oof["split_seed"].unique())
    print(f"rows={len(oof)}  seeds={seeds}  rows/seed={len(oof)//len(seeds)}  "
          f"extractants={oof['extractant'].nunique()}")

    per_seed = pd.DataFrame(
        [
            {"arm": arm, "split_seed": int(seed), **metrics_one_seed(block, arm)}
            for seed, block in oof.groupby("split_seed", sort=True)
            for arm in args.arms
        ]
    )
    leaderboard = per_seed.drop(columns="split_seed").groupby("arm").mean()
    leaderboard["macro_mae_split_sd"] = per_seed.groupby("arm")["equal_extractant_macro_mae"].std()

    committed_path = args.oof.parent / "leaderboard.csv"
    committed = pd.read_csv(committed_path).set_index("arm") if committed_path.exists() else None

    pd.set_option("display.width", 200)
    print("\n== per-seed equal-extractant macro MAE ==")
    print(per_seed.pivot(index="arm", columns="split_seed", values="equal_extractant_macro_mae").round(6).to_string())
    print("\n== 5-seed means (recomputed here) ==")
    print(leaderboard.round(6).to_string())
    if committed is not None:
        cols = [c for c in leaderboard.columns if c in committed.columns]
        diff = (leaderboard[cols] - committed.loc[leaderboard.index, cols]).abs().max().max()
        print(f"\nmax |recomputed - committed leaderboard.csv| over {len(cols)} columns: {diff:.2e}")

    print(f"\n== paired extractant bootstrap vs {args.baseline} "
          f"(pooled 5-seed OOF, {args.replicates} draws, seed {args.bootstrap_seed}) ==")
    for arm in args.arms:
        if arm == args.baseline:
            continue
        b = paired_extractant_bootstrap(oof, args.baseline, arm, args.replicates, args.bootstrap_seed)
        print(f"{args.baseline}_vs_{arm}: delta={b['point_delta_mae']:+.6f}  "
              f"CI95=[{b['ci95_low']:+.6f}, {b['ci95_high']:+.6f}]  "
              f"p_better={b['p_candidate_better']:.4f}  improved={b['extractants_improved']}/"
              f"{oof['extractant'].nunique()}")


if __name__ == "__main__":
    main()
