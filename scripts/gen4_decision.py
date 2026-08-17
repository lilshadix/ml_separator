#!/usr/bin/env python3
"""Apply the pre-declared gen4 decision rule to a confirmation-run OOF file.

Selection of the candidate set happened on split seed 104729 only; the other
four seeds are confirmation seeds.  A candidate "improves on A2_refit" when

1. its 5-seed mean equal-extractant macro MAE is lower than A2_refit's,
2. its paired delta (A2_refit minus candidate) is positive in >= 3 of the 4
   confirmation seeds,
3. the extractant-unit paired bootstrap 95% CI on the confirmation seeds
   excludes zero, with Holm correction across the frozen candidate set.

Transitive projection is reported as a separate, uniform post-processing step
(candidate vs candidate_TP), never used to pick between candidate families.
Usage: gen4_decision.py <oof_predictions.csv> [--selection-seed 104729]
        [--candidates PRIOR PRIOR_lig2d A2_lig2d SCALE_s50] [--controls ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from lanthanide_separation.gen3_metrics import (  # noqa: E402
    equal_group_macro_mae,
    paired_extractant_bootstrap,
)
from lanthanide_separation.pairs import PAIR_TARGET_COLUMN  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("oof", type=Path)
    parser.add_argument("--selection-seed", type=int, default=104729)
    parser.add_argument("--baseline", default="A2_refit")
    parser.add_argument("--candidates", nargs="+", default=["PRIOR", "PRIOR_lig2d", "A2_lig2d", "SCALE_s50"])
    parser.add_argument("--extra", nargs="*", default=[], help="arms to report without gating (controls, TP twins)")
    parser.add_argument("--replicates", type=int, default=10000)
    args = parser.parse_args()

    oof = pd.read_csv(args.oof)
    truth_col = PAIR_TARGET_COLUMN
    seeds = sorted(oof["split_seed"].unique())
    confirmation = [s for s in seeds if s != args.selection_seed]
    arms = [args.baseline, *args.candidates, *args.extra]
    arms = [a for a in arms if f"prediction_{a}" in oof.columns]

    # per-seed macro MAE
    rows = []
    for seed, block in oof.groupby("split_seed"):
        for arm in arms:
            rows.append({"seed": int(seed), "arm": arm,
                         "macro": equal_group_macro_mae(block[truth_col], block[f"prediction_{arm}"], block["extractant"])})
    per_seed = pd.DataFrame(rows).pivot(index="arm", columns="seed", values="macro").loc[arms]
    base = per_seed.loc[args.baseline]
    delta = base - per_seed  # positive = candidate better
    table = pd.DataFrame({
        "mean_macro_5seeds": per_seed.mean(axis=1),
        "mean_macro_confirm": per_seed[confirmation].mean(axis=1),
        "delta_mean_5seeds": delta.mean(axis=1),
        "delta_mean_confirm": delta[confirmation].mean(axis=1),
        "positive_confirm_seeds": (delta[confirmation] > 0).sum(axis=1),
        "positive_all_seeds": (delta > 0).sum(axis=1),
    })

    conf = oof[oof["split_seed"].isin(confirmation)]
    comparisons = {f"{args.baseline}_vs_{a}": (args.baseline, a) for a in arms if a != args.baseline}
    boot = paired_extractant_bootstrap(conf, comparisons, replicates=args.replicates, seed=8675309).set_index("candidate")
    table["boot_delta_confirm"] = boot["point_delta_mae"]
    table["ci95_low"] = boot["ci95_low"]
    table["ci95_high"] = boot["ci95_high"]
    table["p_better"] = boot["p_candidate_better"]
    table["extractants_improved"] = boot["extractants_improved"].astype("Int64")

    # Holm across the frozen candidate set (two-sided p from bootstrap: 2*min(p, 1-p))
    cand = [a for a in args.candidates if a in table.index]
    p_two = {a: 2.0 * min(table.loc[a, "p_better"], 1.0 - table.loc[a, "p_better"]) for a in cand}
    order = sorted(cand, key=lambda a: p_two[a])
    m = len(order)
    holm = {}
    running = 0.0
    for i, a in enumerate(order):
        adj = min(1.0, (m - i) * p_two[a])
        running = max(running, adj)
        holm[a] = running
    table["holm_p_two_sided"] = pd.Series(holm)
    table["passes_rule"] = [
        bool(a in cand and table.loc[a, "mean_macro_5seeds"] < table.loc[args.baseline, "mean_macro_5seeds"]
             and table.loc[a, "positive_confirm_seeds"] >= 3 and table.loc[a, "ci95_low"] > 0 and holm.get(a, 1.0) < 0.05)
        for a in table.index
    ]
    pd.set_option("display.width", 250)
    print(f"seeds: {seeds}; selection seed {args.selection_seed}; confirmation seeds {confirmation}")
    print("\nper-seed macro MAE:")
    print(per_seed.round(4).to_string())
    print("\ndecision table (delta = A2_refit - candidate; positive = better):")
    print(table.round(4).to_string())
    out = args.oof.parent / "decision_table.csv"
    table.to_csv(out)
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
