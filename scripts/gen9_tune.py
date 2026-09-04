#!/usr/bin/env python
"""Choose the booster's hyperparameters inside training folds, once, and freeze them.

``CurveBoost`` has a learner and an objective, and gen9's whole claim is about the
objective.  If each loss weight were allowed its own hyperparameters, "the shape arm
is better" would be indistinguishable from "the shape arm got a better tree depth",
so this script selects one configuration and every arm — control and shape alike —
then uses it unchanged.

Scored by **inner group k-fold over Tanimoto chemotypes inside the outer training
partition**.  No outer test row is touched, so nothing here can inform a held-out
number; the folds it uses are the same ``seeded_group_kfold`` the outer plan uses,
one level down.

Tuned on the ``ROW_ONLY`` objective on purpose.  Tuning on a shape arm would pick
the learner that best exploits the curve terms and hand the shape arms an advantage
the control never had.  Tuning on the control asks the honest question — what is
the best this learner does at the *current* objective — and that is the number gen9
has to beat.

The grid always contains :data:`FROZEN_EQUIVALENT`, the one-stage configuration
that is algebraically ``REC_ecfp_plus_recovered``, so the table shows what the
frozen model scores under the identical inner protocol rather than asking anyone to
take the comparison on trust.

Writes ``runs/gen9_shape/boost_config.json`` and ``boost_tuning.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402
from lanthanide_separation.gen7.harness import DEFAULT_SEEDS, build_folds, load_cohort  # noqa: E402
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN, group_balanced_weights  # noqa: E402
from lanthanide_separation.gen9.metrics import macro_mae  # noqa: E402
from lanthanide_separation.gen9.objective import (  # noqa: E402
    FROZEN_EQUIVALENT, SQUARED_LOSS_DELTA, BoostConfig, CurveBoost,
)
from lanthanide_separation.gen9.train import build_design  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen9_shape"

#: Small and motivated rather than exhaustive.  The axes that matter for a boosted
#: forest here are how far each stage is allowed to walk (``learning_rate`` against
#: ``n_stages``), how much averaging each step gets (``trees_per_stage``), and how
#: heavy-tailed the row loss treats the corrupted decades the data audit keeps
#: finding (``row_delta``).  ``max_features`` and ``min_samples_leaf`` are held at
#: the frozen arm's values so the one-stage configuration really is that arm.
CANDIDATES: tuple[BoostConfig, ...] = (
    FROZEN_EQUIVALENT,
    BoostConfig(n_stages=1, trees_per_stage=400, learning_rate=1.0, row_delta=1.0),
    BoostConfig(n_stages=4, trees_per_stage=100, learning_rate=0.5, row_delta=1.0),
    BoostConfig(n_stages=10, trees_per_stage=60, learning_rate=0.3, row_delta=1.0),
    BoostConfig(n_stages=10, trees_per_stage=60, learning_rate=0.3, row_delta=3.0),
    BoostConfig(n_stages=10, trees_per_stage=60, learning_rate=0.3, row_delta=SQUARED_LOSS_DELTA),
    BoostConfig(n_stages=16, trees_per_stage=50, learning_rate=0.25, row_delta=1.0),
    BoostConfig(n_stages=25, trees_per_stage=40, learning_rate=0.2, row_delta=1.0),
    BoostConfig(n_stages=25, trees_per_stage=40, learning_rate=0.2, row_delta=3.0),
    BoostConfig(n_stages=40, trees_per_stage=30, learning_rate=0.15, row_delta=1.0),
    BoostConfig(n_stages=10, trees_per_stage=120, learning_rate=0.3, row_delta=1.0),
    BoostConfig(n_stages=25, trees_per_stage=40, learning_rate=0.2, row_delta=1.0,
                min_samples_leaf=5),
)


def label(config: BoostConfig) -> str:
    return (f"s{config.n_stages}x{config.trees_per_stage}_lr{config.learning_rate:g}"
            f"_d{'sq' if config.row_delta >= SQUARED_LOSS_DELTA else f'{config.row_delta:g}'}"
            f"_leaf{config.min_samples_leaf}")


def score_config(config: BoostConfig, cached, model_seed: int) -> tuple[float, float, float]:
    scores, seconds = [], time.time()
    for frame_b, x_a, x_b, weight, y_a, y_b in cached:
        model = CurveBoost(config=config, lambda_delta=0.0, lambda_span=0.0,
                           random_state=model_seed)
        model.fit(x_a, y_a, sample_weight=weight, record_history=False)
        scores.append(macro_mae(frame_b.assign(prediction=model.predict(x_b))))
    return float(np.mean(scores)), float(np.std(scores)), time.time() - seconds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--outer-folds", type=int, nargs="*", default=[0, 2])
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--out", type=Path, default=OUT / "boost_config.json")
    args = parser.parse_args(argv)

    cohort = load_cohort()
    frame = cohort.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    folds = build_folds(frame, args.seed)

    cached = []
    for index in args.outer_folds:
        fold = folds[index]
        outer = frame.iloc[fold.train_index].reset_index(drop=True)
        y_outer = target[fold.train_index]
        groups = outer["tanimoto_cluster"].astype(str).to_numpy()
        for inner_train, inner_test in seeded_group_kfold(groups, args.inner_splits, args.seed):
            a, b = outer.iloc[inner_train], outer.iloc[inner_test]
            x_a, x_b = build_design(cohort, a, b)
            cached.append((b, x_a, x_b, group_balanced_weights(a["ecfp_cluster"]),
                           y_outer[inner_train], y_outer[inner_test]))
    model_seed = folds[args.outer_folds[0]].model_seed
    print(f"{len(cached)} inner splits over outer folds {args.outer_folds} of seed {args.seed}; "
          f"{len(CANDIDATES)} configurations")

    records = []
    for config in CANDIDATES:
        mean, sd, seconds = score_config(config, cached, model_seed)
        records.append({"label": label(config), **config.as_dict(),
                        "inner_macro_mae": mean, "inner_sd": sd, "seconds": seconds,
                        "is_frozen_equivalent": config == FROZEN_EQUIVALENT})
        print(f"  {mean:.4f} (sd {sd:.4f})  {label(config)}  ({seconds:.0f}s)"
              f"{'   <- algebraically the frozen arm' if config == FROZEN_EQUIVALENT else ''}",
              flush=True)

    table = pd.DataFrame.from_records(records).sort_values("inner_macro_mae")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out.with_name("boost_tuning.csv"), index=False)

    best_index = int(np.argmin([r["inner_macro_mae"] for r in records]))
    chosen = CANDIDATES[best_index]
    frozen_score = float(next(r["inner_macro_mae"] for r in records if r["is_frozen_equivalent"]))
    args.out.write_text(json.dumps({
        "config": chosen.as_dict(),
        "label": label(chosen),
        "selected_by": {
            "protocol": "inner seeded_group_kfold over tanimoto_cluster inside the outer "
                        "training partition; no outer test row is used",
            "objective": "ROW_ONLY (lambda_delta = lambda_span = 0)",
            "split_seed": int(args.seed), "outer_folds": list(args.outer_folds),
            "inner_splits": int(args.inner_splits),
            "inner_macro_mae": float(records[best_index]["inner_macro_mae"]),
            "frozen_equivalent_inner_macro_mae": frozen_score,
            "n_configurations": len(CANDIDATES),
        },
    }, indent=2))
    print(f"\nchose {label(chosen)} at {records[best_index]['inner_macro_mae']:.4f} "
          f"(frozen-equivalent configuration: {frozen_score:.4f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
