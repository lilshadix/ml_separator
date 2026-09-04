#!/usr/bin/env python
"""Run the gen8 §5 Conditional Neural Process arms against offset correction.

The only comparison that matters here is paired: every CNP variant and every
reference adapter is scored on the *same* held-out ligand, the *same* repeat, the
*same* candidate pool and the *same* evaluation rows, with the *same* k measured
values handed to it.  Any difference in the table is a difference in method.

The fold plan is the frozen one — ``seeded_group_kfold`` over the Tanimoto
chemotype — and the script asserts that its own folds agree row-for-row with the
``fold`` column of the frozen model's out-of-fold predictions before anything is
trained.  If they ever disagreed, a "training" fold would silently contain the
held-out chemotype and every number below would be meaningless.

Usage::

    python scripts/gen8_cnp.py --seeds 104729 --repeats 6 --steps 2000
    python scripts/gen8_cnp.py --seeds all --repeats 12 --steps 2500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402
from lanthanide_separation.gen8 import evaluate as gen8_evaluate  # noqa: E402
from lanthanide_separation.gen8.adapters import default_adapters  # noqa: E402
from lanthanide_separation.gen8.cnp import (  # noqa: E402
    LIGAND_COLUMNS, MASSACTION_COLUMNS, METAL_COLUMNS, RECOVERED_COLUMNS,
    build_cnp_adapters, parameter_count,
)

CACHE = ROOT / "runs" / "gen7_architecture" / "cache"
FINALISTS = ROOT / "runs" / "gen7_architecture" / "finalists"
OUT = ROOT / "runs" / "gen8_architecture" / "cnp"
MODEL = "REC_ecfp_plus_recovered"
#: The frozen gen5/gen6/gen7 model seed formula.  Fixed at 42, independent of split.
MODEL_SEED = lambda fold: 42 + fold * 1009 + 9999991  # noqa: E731


def load_cohort() -> pd.DataFrame:
    """Identity + the feature columns the CNP and the reference adapters need.

    A reduced frame on purpose: the driver merges the whole cohort onto 26k
    out-of-fold rows, and carrying 2,048 ECFP bits through that join costs half a
    gigabyte for columns no adapter in this study reads.
    """
    head = pd.read_parquet(CACHE / "cohort.parquet", columns=["row_id"])
    all_columns = pd.read_parquet(CACHE / "cohort.parquet").columns  # cheap enough once
    cond = [c for c in all_columns if c.startswith("cond__")]
    wanted = ["row_id"] + list(METAL_COLUMNS) + list(MASSACTION_COLUMNS) + cond \
        + [c for c in LIGAND_COLUMNS if c in all_columns]
    wanted = list(dict.fromkeys(c for c in wanted if c in all_columns))
    cohort = pd.read_parquet(CACHE / "cohort.parquet", columns=wanted)
    recovered = pd.read_parquet(CACHE / "recovered_cells.parquet",
                                columns=["row_id"] + list(RECOVERED_COLUMNS))
    cohort = cohort.merge(recovered, on="row_id", how="left", validate="one_to_one")
    assert cohort["row_id"].is_unique
    assert len(cohort) == len(head)
    return cohort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="104729")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-ligands", type=int, default=24)
    parser.add_argument("--policies", default="RANDOM")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tag", default="")
    parser.add_argument("--no-attentive", action="store_true")
    args = parser.parse_args()

    oof = pd.read_parquet(FINALISTS / "oof_predictions.parquet")
    oof = oof[oof["model"] == MODEL].reset_index(drop=True)
    seeds = sorted(oof["split_seed"].unique().tolist()) if args.seeds == "all" \
        else [int(s) for s in args.seeds.split(",")]
    oof = oof[oof["split_seed"].isin(seeds)].reset_index(drop=True)

    cohort = load_cohort()
    identity = pd.read_parquet(CACHE / "cohort.parquet",
                               columns=["row_id", "extractant", "tanimoto_cluster", "log_D"])
    groups = identity["tanimoto_cluster"].astype(str).to_numpy()

    # ---- the fold plan must be byte-identical to the one the frozen model used --
    plan: dict[tuple[int, int], np.ndarray] = {}
    for split_seed in seeds:
        marked = oof[oof["split_seed"] == split_seed].set_index("row_id")["fold"].to_dict()
        for fold, (train_index, test_index) in enumerate(
                seeded_group_kfold(groups, 5, int(split_seed))):
            observed = np.array([marked[r] for r in identity["row_id"].to_numpy()[test_index]])
            if not (observed == fold).all():
                raise SystemExit(f"fold plan disagrees with the OOF frame at "
                                 f"seed {split_seed} fold {fold}")
            plan[(int(split_seed), fold)] = train_index
    print(f"fold plan verified against {MODEL} for seeds {seeds}", flush=True)

    prediction_by_seed = {
        int(s): oof[oof["split_seed"] == s].set_index("row_id")["prediction"]
        for s in seeds}

    def fold_trainer(split_seed: int, fold: int):
        index = plan[(int(split_seed), int(fold))]
        frame = cohort.iloc[index].copy()
        frame["extractant"] = identity["extractant"].to_numpy()[index]
        frame["tanimoto_cluster"] = identity["tanimoto_cluster"].to_numpy()[index]
        frame["prediction"] = prediction_by_seed[int(split_seed)].reindex(
            frame["row_id"]).to_numpy()
        y = identity["log_D"].to_numpy(dtype=float)[index]
        return frame, y, MODEL_SEED(int(fold))

    adapters = build_cnp_adapters(steps=args.steps, batch_ligands=args.batch_ligands,
                                  include_attentive=not args.no_attentive,
                                  threads=args.threads)
    adapters = list(adapters) + list(default_adapters())
    print(f"{len(adapters)} adapters: {[a.name for a in adapters]}", flush=True)

    started = time.time()
    detail = gen8_evaluate.evaluate_fewshot(
        oof, cohort, adapters,
        policies=tuple(p for p in args.policies.split(",") if p),
        repeats=args.repeats, fold_trainer=fold_trainer, verbose=True)
    print(f"elapsed {time.time() - started:.1f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"seeds{'-'.join(str(s) for s in seeds)}_r{args.repeats}"
    path = OUT / f"cnp_detail_{tag}.parquet"
    detail.to_parquet(path, index=False)

    audit = {a.name: {"parameters": parameter_count(a),
                      "predict_fallbacks": int(getattr(a, "failures", 0)),
                      "last_error": getattr(a, "last_error", ""),
                      "last_fold_best_step": int(getattr(a, "history", (0,))[0])}
             for a in adapters if hasattr(a, "net")}
    (OUT / f"cnp_params_{tag}.json").write_text(json.dumps(audit, indent=2))
    for name, row in audit.items():
        print(f"  {name}: {row['parameters']:,} params, "
              f"{row['predict_fallbacks']} fallbacks {row['last_error']}", flush=True)
    fallbacks = sum(r["predict_fallbacks"] for r in audit.values())
    if fallbacks:
        print(f"WARNING: {fallbacks} predict() calls fell back to the frozen prediction",
              flush=True)

    summary = gen8_evaluate.summarise(detail)
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(summary.to_string(index=False))
    print(f"\nwrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
