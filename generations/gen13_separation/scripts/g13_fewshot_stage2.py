"""Stage-2 one-pair calibration on a finished ladder's saved curves.

    .venv/Scripts/python.exe generations/gen13_separation/scripts/g13_fewshot_stage2.py --label B_primary \
        --arms C_DIRECT_ROW,X_ENS_DIRECT+LOWRANK_K2,M_SELECTED,B1_MEAN_CURVE

Writes ``metrics/<label>/fewshot_stage2_leaderboard.csv`` (extractant-macro MAE before and after
the correction, per arm and per support design), ``fewshot_stage2_by_support_dz.csv`` and the long
pair table ``fewshot_stage2_pairs.parquet``, plus a chemotype-blocked bootstrap of the gain in
``bootstrap/<label>/fewshot_stage2_contrasts.csv``.

The correction is the conditional mean of the residual curve given one measured separation factor,
with the residual covariance estimated leave-chemotype-out within each split seed
(``gen13sep.fewshot_stage2``).  ``support_design = random`` reproduces the locked protocol's draw so
the number is comparable with ``metrics/<label>/fewshot_leaderboard.csv``; ``widest`` measures the
pair with the largest atomic-number gap, which is the pair a laboratory would choose.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen13sep import paths  # noqa: E402
from gen13sep.cohort import build_cohort  # noqa: E402
from gen13sep.fewshot_stage2 import (evaluate_one_pair_conditional, evaluate_support_budget,  # noqa: E402
                                     macro_scores)
from gen13sep.inference import paired_contrasts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="B_primary")
    ap.add_argument("--arms", default="C_DIRECT_ROW,X_ENS_DIRECT+LOWRANK_K2,M_SELECTED,B1_MEAN_CURVE")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--shrink", type=float, default=0.25)
    ap.add_argument("--noise-var", type=float, default=0.09)
    ap.add_argument("--replicates", type=int, default=10_000)
    ap.add_argument("--budget", type=int, default=0,
                    help="if > 0, run only the measurement-budget sweep up to this many pairs")
    args = ap.parse_args()

    cohort = build_cohort("exact")
    curve_dir = paths.PREDICTION_DIR / args.label / "curves"
    arms = [a for a in args.arms.split(",") if (curve_dir / f"{a}.parquet").exists()]
    missing = [a for a in args.arms.split(",") if a not in arms]
    if missing:
        print(f"skipping arms without saved curves: {missing}", flush=True)
    if not arms:
        raise SystemExit(f"no saved curves under {curve_dir}")
    frames = {a: pd.read_parquet(curve_dir / f"{a}.parquet") for a in arms}
    basis_path = paths.PREDICTION_DIR / args.label / "bases.json"
    basis_lookup = None
    if basis_path.exists():
        import json
        with open(basis_path, encoding="utf-8") as fh:
            basis_lookup = {(b["arm"], int(b["split_seed"]), int(b["fold"])): np.asarray(b["basis"])
                            for b in json.load(fh)}

    metric_dir_early = paths.METRIC_DIR / args.label
    metric_dir_early.mkdir(parents=True, exist_ok=True)
    if args.budget > 0:
        budget = evaluate_support_budget(frames, cohort.frame, arms=arms, k_max=args.budget,
                                         shrink=args.shrink, noise_var=args.noise_var)
        budget.to_csv(metric_dir_early / "fewshot_stage2_budget.csv", index=False)
        print(budget.round(4).to_string(index=False), flush=True)
        return

    table = evaluate_one_pair_conditional(frames, cohort.frame, arms=arms, repeats=args.repeats,
                                          shrink=args.shrink, noise_var=args.noise_var,
                                          basis_lookup=basis_lookup)
    metric_dir = paths.METRIC_DIR / args.label
    metric_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(metric_dir / "fewshot_stage2_pairs.parquet", index=False)
    board = macro_scores(table)
    board.to_csv(metric_dir / "fewshot_stage2_leaderboard.csv", index=False)
    print(board.round(4).to_string(index=False), flush=True)

    adapters = [c for c in ("zero_shot", "conditional", "basis_shift", "rescale", "no_model_linear")
                if c in table.columns and not table[c].isna().all()]
    errs = table.assign(**{f"err__{c}": (table["y"] - table[c]).abs() for c in adapters})
    by_dz = errs.groupby(["arm", "support_design", "support_dz"])[[f"err__{c}" for c in adapters]].agg(["mean", "size"])
    by_dz.to_csv(metric_dir / "fewshot_stage2_by_support_dz.csv")

    parts = []
    for (arm, design), block in errs.groupby(["arm", "support_design"], sort=True):
        per = (block.groupby(["split_seed", "extractant", "chemotype"])[[f"err__{c}" for c in adapters]]
               .mean().reset_index())
        long = pd.concat([per.assign(arm=c, mae_all=per[f"err__{c}"]) for c in adapters],
                         ignore_index=True)[["split_seed", "extractant", "chemotype", "arm", "mae_all"]]
        comparisons = {f"{arm}|{design}|vs_{ref}": (ref, "conditional")
                       for ref in adapters if ref != "conditional"}
        res = paired_contrasts(long, comparisons, value="mae_all", replicates=args.replicates)
        if len(res):
            res.insert(1, "arm_scored", arm)
            res.insert(2, "support_design", design)
            parts.append(res)
    contrasts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    boot_dir = paths.BOOTSTRAP_DIR / args.label
    boot_dir.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(boot_dir / "fewshot_stage2_contrasts.csv", index=False)
    print(contrasts.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
