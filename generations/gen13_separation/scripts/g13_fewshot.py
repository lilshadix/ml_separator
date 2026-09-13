"""One-pair calibration study (H5) on a finished ladder's saved curves.

    .venv/Scripts/python.exe generations/gen13_separation/scripts/g13_fewshot.py --label B_primary --arms M_SELECTED,C_DIRECT_ROW,B1_MEAN_CURVE

For every held-out cell with >= 3 metals, one observed pair is drawn (5 repeats, deterministic
per cell), every arm's curve is adapted, and the remaining pairs are scored.  Outputs
metrics/<label>/fewshot_*.csv and bootstrap/<label>/fewshot_contrasts.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen13sep import paths  # noqa: E402
from gen13sep.cohort import build_cohort  # noqa: E402
from gen13sep.fewshot import evaluate_one_pair  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="B_primary")
    ap.add_argument("--arms", default="M_SELECTED,C_DIRECT_ROW,B1_MEAN_CURVE")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--replicates", type=int, default=10_000)
    args = ap.parse_args()

    cohort = build_cohort("exact")
    frame = cohort.frame.set_index("cell_id")
    pred_dir = paths.PREDICTION_DIR / args.label
    with open(pred_dir / "bases.json", encoding="utf-8") as fh:
        bases_raw = json.load(fh)
    bases = {(b["arm"], b["split_seed"], b["fold"]): np.asarray(b["basis"]) for b in bases_raw}
    arms = args.arms.split(",")
    curves = {a: pd.read_parquet(pred_dir / "curves" / f"{a}.parquet") for a in arms}
    ccols = [f"c__{m}" for m in LANTHANIDES]
    ycols = [f"logD__{m}" for m in LANTHANIDES]

    tables = []
    for (seed, fold), block in curves[arms[0]].groupby(["split_seed", "fold"]):
        cell_ids = block["cell_id"].tolist()
        Y = frame.loc[cell_ids, ycols].to_numpy(dtype=float)
        cp = {}
        bs = {}
        for a in arms:
            sub = curves[a][(curves[a]["split_seed"] == seed) & (curves[a]["fold"] == fold)].set_index("cell_id")
            cp[a] = sub.loc[cell_ids, ccols].to_numpy(dtype=float)
            bs[a] = bases.get((a, seed, fold))
        t = evaluate_one_pair(cp, bs, Y, cell_ids, frame.loc[cell_ids, "extractant"].tolist(),
                              frame.loc[cell_ids, "chemotype"].tolist(), seed=int(seed), repeats=args.repeats)
        t["fold"] = fold
        tables.append(t)
    detail = pd.concat(tables, ignore_index=True)
    metric_dir = paths.METRIC_DIR / args.label; metric_dir.mkdir(parents=True, exist_ok=True)
    boot_dir = paths.BOOTSTRAP_DIR / args.label; boot_dir.mkdir(parents=True, exist_ok=True)
    detail["abs_err"] = (detail["y"] - detail["prediction"]).abs()
    detail["variant"] = detail["arm"] + "|" + detail["adapter"]
    per_ext = (detail.groupby(["variant", "split_seed", "extractant", "chemotype"])["abs_err"].mean()
               .rename("mae_all").reset_index().rename(columns={"variant": "arm"}))
    board = (per_ext.groupby(["arm", "split_seed"])["mae_all"].mean().groupby(level=0)
             .agg(["mean", "std"]).rename(columns={"mean": "macro_mae_extractant", "std": "seed_sd"})
             .sort_values("macro_mae_extractant"))
    board.to_csv(metric_dir / "fewshot_leaderboard.csv")
    per_ext.to_csv(metric_dir / "fewshot_per_extractant.csv", index=False)
    comps = {}
    have = set(per_ext["arm"])
    # S6 primary (Addendum 1 item 8): M_SELECTED + basis_shift vs zero-shot and vs the no-model line
    if "M_SELECTED|basis_shift" in have:
        comps["S6a_M_SELECTED_basis_shift_vs_zero_shot"] = ("M_SELECTED|zero_shot", "M_SELECTED|basis_shift")
        comps["S6b_M_SELECTED_basis_shift_vs_no_model_linear"] = ("M_SELECTED|no_model_linear", "M_SELECTED|basis_shift")
    for a in arms:
        comps[f"X_{a}|rescale_vs_zero_shot"] = (f"{a}|zero_shot", f"{a}|rescale")
        comps[f"X_{a}|rescale_vs_no_model_linear"] = (f"{a}|no_model_linear", f"{a}|rescale")
        if f"{a}|basis_shift" in have and a != "M_SELECTED":
            comps[f"X_{a}|basis_shift_vs_zero_shot"] = (f"{a}|zero_shot", f"{a}|basis_shift")
            comps[f"X_{a}|basis_shift_vs_no_model_linear"] = (f"{a}|no_model_linear", f"{a}|basis_shift")
    boot = paired_contrasts(per_ext, comps, value="mae_all", replicates=args.replicates)
    boot["status"] = ["registered" if c.startswith("S6") else "exploratory" for c in boot["comparison"]]
    # by support gap (adjacent support = index gap 1, i.e. neighbours in the 14-metal list)
    gap = (detail.groupby(["variant", "split_seed", "extractant", "chemotype", detail["support_index_gap"].le(1).map({True: "adjacent_support", False: "far_support"})])["abs_err"].mean()
           .rename("mae_all").reset_index().rename(columns={"variant": "arm", "support_index_gap": "support"}))
    gap.to_csv(metric_dir / "fewshot_by_support.csv", index=False)
    boot.to_csv(boot_dir / "fewshot_contrasts.csv", index=False)
    print(board.round(4).to_string())
    print(boot[["comparison", "status", "point", "ci95_low", "ci95_high", "bca_low", "bca_high", "p_two_sided",
                "seeds_positive", "n_seeds", "n_units", "passes_P1"]].round(4).to_string(index=False))
    print("cells per seed:", int(detail.groupby("split_seed")["cell_id"].nunique().mean()),
          "extractants:", detail["extractant"].nunique())


if __name__ == "__main__":
    main()
