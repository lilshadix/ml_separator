"""Leave-one-component-out on the gen7 candidate (brief §16).

A final score says a model works; an ablation says *which part* works.  Each arm here
removes exactly one component from the full candidate and changes nothing else — same
folds, same seeds, same test rows — so the delta is attributable.

The components, in the brief's order:

``recovered``    the experimental variables rescued from the SAFE exports
``indicators``   the missing-value indicator columns (their own line, because they
                 are worth more than the ligand descriptors and half of them are
                 publication reporting conventions)
``massaction``   the mass-action log-concentration block
``ligand``       *all* ligand chemistry — fingerprint, descriptors, donors, physics.
                 The single most important ablation in the table: it turns the model
                 into ``NULL_metal_cond`` and shows how much chemistry is worth at all
``ligphys``      the ligand-diluent coupling terms
``metal_repr``   the continuous metal descriptors, leaving only the symbol's one-hot
``conditions``   the experimental conditions

Two arms in the other direction (add-one-in from the no-ligand model) are included
too, because with an effect this small a leave-one-out on a saturated model and an
add-one-in on an empty one can disagree, and if they do that is the finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.metrics import (  # noqa: E402
    paired_unit_bootstrap, per_unit_statistics,
)
from lanthanide_separation.gen7.contenders import Tabular, extratrees  # noqa: E402
from lanthanide_separation.gen7.harness import (  # noqa: E402
    DEFAULT_SEEDS, evaluate_contender, leaderboard, load_cohort, score_oof,
)

FULL = ("METAL", "COND", "ECFP", "MASSACTION", "LIG2D_EXT", "DONORS", "PHYSCHEM",
        "LIGPHYS", "RECOVERED")
LIGAND_BLOCKS = {"ECFP", "LIG2D_EXT", "DONORS", "PHYSCHEM", "LIGPHYS"}

REMOVALS: dict[str, set[str]] = {
    "minus_recovered": {"RECOVERED"},
    "minus_massaction": {"MASSACTION"},
    "minus_ligand_all": LIGAND_BLOCKS,
    "minus_ligphys": {"LIGPHYS"},
    "minus_ecfp": {"ECFP"},
    "minus_lig2d": {"LIG2D_EXT"},
    "minus_donors": {"DONORS", "PHYSCHEM"},
    "minus_metal": {"METAL"},
    "minus_conditions": {"COND"},
}
ADDITIONS: dict[str, tuple[str, ...]] = {
    "add_none": ("METAL", "COND"),
    "add_massaction": ("METAL", "COND", "MASSACTION"),
    "add_donors": ("METAL", "COND", "MASSACTION", "DONORS", "PHYSCHEM"),
    "add_ligphys": ("METAL", "COND", "MASSACTION", "DONORS", "PHYSCHEM", "LIGPHYS"),
    "add_ecfp": ("METAL", "COND", "MASSACTION", "DONORS", "PHYSCHEM", "LIGPHYS", "ECFP"),
    "add_recovered": FULL,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--indicators", action="store_true", default=True)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "runs" / "gen7_architecture" / "ablations")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    seeds = DEFAULT_SEEDS[: max(1, min(5, args.seeds))]
    available = [b for b in FULL if b in cohort.blocks]
    print(f"full model blocks: {available}", flush=True)

    contenders = [Tabular(tuple(available), extratrees, "ABL_full", add_indicator=True)]
    contenders.append(Tabular(tuple(available), extratrees, "ABL_minus_indicators",
                              add_indicator=False))
    for name, removed in REMOVALS.items():
        blocks = tuple(b for b in available if b not in removed)
        if not blocks or blocks == tuple(available):
            continue
        contenders.append(Tabular(blocks, extratrees, f"ABL_{name}", add_indicator=True))
    for name, blocks in ADDITIONS.items():
        blocks = tuple(b for b in blocks if b in cohort.blocks)
        contenders.append(Tabular(blocks, extratrees, f"ABL_{name}", add_indicator=True))

    parts = []
    for contender in contenders:
        try:
            parts.append(evaluate_contender(contender, cohort, seeds=seeds))
        except Exception as error:
            print(f"  [{contender.name}] FAILED {type(error).__name__}: {error}", flush=True)
    oof = pd.concat(parts, ignore_index=True)
    oof.to_parquet(args.out / "oof_predictions.parquet", index=False)
    scores = score_oof(oof)
    scores.to_csv(args.out / "scores_by_seed.csv", index=False)
    board = leaderboard(scores)
    board.to_csv(args.out / "leaderboard.csv", index=False)

    # paired bootstrap of every ablation against the full model, on seed 0
    seed0 = oof[oof["split_seed"] == seeds[0]]
    models = sorted(seed0["model"].unique())
    wide = seed0[seed0["model"] == models[0]][
        ["row_id", "extractant", "ecfp_cluster", "tanimoto_cluster", "log_D"]].reset_index(drop=True)
    for model in models:
        part = seed0[seed0["model"] == model][["row_id", "prediction"]]
        wide = wide.merge(part.rename(columns={"prediction": f"prediction_{model}"}),
                          on="row_id", validate="one_to_one")
    per_unit = per_unit_statistics(wide, models)
    block_of_unit = (wide.drop_duplicates("ecfp_cluster")
                     .set_index(wide.drop_duplicates("ecfp_cluster")["ecfp_cluster"].astype(str))
                     ["tanimoto_cluster"].astype(str).to_dict())
    comparisons = {m: ("ABL_full", m) for m in models if m != "ABL_full"}
    bootstrap = paired_unit_bootstrap(per_unit, comparisons, block_of_unit=block_of_unit)
    bootstrap.to_csv(args.out / "ablation_bootstrap.csv", index=False)

    pd.set_option("display.width", 210)
    columns = [c for c in ["model", "macro_mae", "macro_mae_sd", "offset_mae", "shape_mae",
                           "nn_lt_0_4__macro_mae"] if c in board.columns]
    print(board[columns].round(4).to_string(index=False))
    # Sign convention, stated because it is easy to invert: ``paired_unit_bootstrap``
    # computes ``reference - candidate`` with ABL_full as reference, so a POSITIVE delta
    # means the ablated arm is BETTER — i.e. removing that component HELPED and the
    # component was hurting.
    print("\nvs ABL_full (POSITIVE = removing the component HELPED; negative = it was carrying weight):")
    view = bootstrap[bootstrap["statistic"] == "mae"][
        ["candidate", "point_delta", "ci95_low", "ci95_high", "block_macro_delta",
         "units_improved", "units_total"]]
    print(view.round(4).to_string(index=False))
    (args.out / "summary.json").write_text(json.dumps(
        {"seeds": [int(s) for s in seeds], "blocks": available}, indent=2))
    print(f"\nartifacts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
