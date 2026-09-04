#!/usr/bin/env python
"""PHASES 2 and 4 — feature access, explicit level+shape, and axis representation.

One driver, one arm registry, gen7's evaluation contract unchanged.  Each stage is
a *predeclared* list of arms; nothing is added after a number is read, and the
registry is the record of what was declared.

``--stage feature_access``   (Phase 2A/2B)
    Is the recomposition necessary, or does the monolith simply fail to reach five
    context columns among 2,160?  The same forest with ``max_features`` over a small
    fixed grid, and with the five columns replicated so a random split subset
    reaches them — the learner untouched, only the access changed.

``--stage level_shape``      (Phase 2C/2D)
    The explicit level + zero-mean shape model, the additive cross-fitted residual
    correction (with its in-sample trap control), and the two-branch model with
    its one interaction form, against the recomposition they must match.

``--stage axis_representation``  (Phase 4)
    The recomposition with each predeclared design-relative representation: gen9's
    endpoint-based columns, rank, local spacing, the hybrid, and the curve-window
    variant of gen9's own columns.

Every arm is run at five seeds, writes a gen7-schema OOF parquet, passes row
accounting, and has its fold-0 determinism measured by refitting twice.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import DEFAULT_SEEDS  # noqa: E402
from lanthanide_separation.gen10.architectures import (  # noqa: E402
    LevelShapeModel, MonolithModel, RecomposedModel, ResidualShapeModel, TwoBranchModel,
    prime_raw_cache,
)
from lanthanide_separation.gen10.setcontext import SetRecomposedModel  # noqa: E402
from lanthanide_separation.gen10.runner import (  # noqa: E402
    audit_folds, default_membership, determinism_probe, finalise, prepared_cohort, run_arm,
)

OUT = REPO_ROOT / "runs" / "gen10_final"

#: Arm registry.  Name -> factory.  Declared before any stage ran.
ARMS: dict[str, callable] = {
    # --- anchors: the gen9 arms, re-expressed and asserted identical ------------
    "FROZEN": lambda: MonolithModel(name="REC_ecfp_plus_recovered", representation_name="NONE"),
    "REL_MONOLITH": lambda: MonolithModel(
        name="GEN9_REL_MONOLITH", representation_name="GEN9", gen9_compat=True),
    "SHAPE_RECOMPOSED": lambda: RecomposedModel(
        name="GEN9_SHAPE_RECOMPOSED", representation_name="GEN9", gen9_compat=True),
    # --- Phase 2B: feature access ----------------------------------------------
    "MF_SQRT": lambda: MonolithModel(
        name="GEN10_MF_SQRT", representation_name="GEN9", gen9_compat=True, max_features="sqrt"),
    "MF_025": lambda: MonolithModel(
        name="GEN10_MF_025", representation_name="GEN9", gen9_compat=True, max_features=0.25),
    "MF_050": lambda: MonolithModel(
        name="GEN10_MF_050", representation_name="GEN9", gen9_compat=True, max_features=0.5),
    "MF_100": lambda: MonolithModel(
        name="GEN10_MF_100", representation_name="GEN9", gen9_compat=True, max_features=1.0),
    "REP_8": lambda: MonolithModel(
        name="GEN10_REP_8", representation_name="GEN9", gen9_compat=True, context_replication=8),
    "REP_32": lambda: MonolithModel(
        name="GEN10_REP_32", representation_name="GEN9", gen9_compat=True,
        context_replication=32),
    # --- Phase 2C/2D: architectures ---------------------------------------------
    "LEVEL_SHAPE": lambda: LevelShapeModel(name="GEN10_LEVEL_SHAPE", representation_name="GEN9"),
    "RESIDUAL_SHAPE": lambda: ResidualShapeModel(
        name="GEN10_RESIDUAL_SHAPE", representation_name="GEN9", cross_fit=True),
    "RESIDUAL_SHAPE_INSAMPLE": lambda: ResidualShapeModel(
        name="GEN10_RESIDUAL_SHAPE_INSAMPLE", representation_name="GEN9", cross_fit=False),
    "TWO_BRANCH": lambda: TwoBranchModel(
        name="GEN10_TWO_BRANCH", representation_name="GEN9", interaction="none"),
    "TWO_BRANCH_INTERACT": lambda: TwoBranchModel(
        name="GEN10_TWO_BRANCH_INTERACT", representation_name="GEN9",
        interaction="base_prediction"),
    # --- Phase 4: axis representation, on the recomposition ---------------------
    "REC_GEN9_PLUS": lambda: RecomposedModel(
        name="GEN10_REC_GEN9_PLUS", representation_name="GEN9_PLUS"),
    "REC_RANK": lambda: RecomposedModel(name="GEN10_REC_RANK", representation_name="RANK"),
    "REC_LOCAL": lambda: RecomposedModel(name="GEN10_REC_LOCAL", representation_name="LOCAL"),
    "REC_HYBRID": lambda: RecomposedModel(
        name="GEN10_REC_HYBRID", representation_name="HYBRID"),
    "REC_ALL": lambda: RecomposedModel(name="GEN10_REC_ALL", representation_name="ALL"),
    "REC_GEN9_CURVEWINDOW": lambda: RecomposedModel(
        name="GEN10_REC_GEN9_CURVEWINDOW", representation_name="GEN9", window="curve"),
    # --- Phase 3: learned set representation, contingent on Phases 1 and 2 -------
    "SET_MEAN": lambda: SetRecomposedModel(name="GEN10_SET_MEAN", pooling="mean"),
    "SET_ATTENTION": lambda: SetRecomposedModel(name="GEN10_SET_ATTENTION", pooling="attention"),
    # --- the same representations on the monolith, for Phase 1's comparison -----
    "MONO_HYBRID": lambda: MonolithModel(
        name="GEN10_MONO_HYBRID", representation_name="HYBRID"),
    "MONO_RANK": lambda: MonolithModel(name="GEN10_MONO_RANK", representation_name="RANK"),
}

STAGES: dict[str, tuple[str, ...]] = {
    "feature_access": ("FROZEN", "REL_MONOLITH", "MF_SQRT", "MF_025", "MF_050", "MF_100",
                       "REP_8", "REP_32"),
    "level_shape": ("SHAPE_RECOMPOSED", "LEVEL_SHAPE", "RESIDUAL_SHAPE",
                    "RESIDUAL_SHAPE_INSAMPLE", "TWO_BRANCH", "TWO_BRANCH_INTERACT"),
    "axis_representation": ("REC_GEN9_PLUS", "REC_RANK", "REC_LOCAL", "REC_HYBRID",
                            "REC_ALL", "REC_GEN9_CURVEWINDOW", "MONO_HYBRID", "MONO_RANK"),
    "set_context": ("SET_MEAN", "SET_ATTENTION"),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--arms", nargs="*", default=None,
                        help="override the stage's arm list (names from the registry)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-determinism", action="store_true")
    args = parser.parse_args(argv)

    names = list(args.arms or STAGES[args.stage])
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; registry has {sorted(ARMS)}")
    seeds = list(DEFAULT_SEEDS[:max(1, min(5, args.seeds))])
    out_dir = args.out or (OUT / args.stage)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = prepared_cohort()
    prime_raw_cache(cohort)
    print(f"cohort {cohort.frame.shape}  fingerprint {cohort.fingerprint}", flush=True)
    membership = default_membership()
    audit = audit_folds(cohort, seeds, membership, out_dir)
    print(f"fold audit clean; curves straddling a boundary: {audit['n_curves_straddling']}; "
          f"rows outside a closed partition: {audit['max_rows_outside_partition']}", flush=True)

    manifest = {"stage": args.stage, "seeds": seeds, "arms": [], "audit": audit,
                "cohort_fingerprint": cohort.fingerprint}
    frames, probes = [], []
    for name in names:
        factory = ARMS[name]
        model = factory()
        spec = {"registry_name": name, "model_name": model.name,
                "class": type(model).__name__,
                "parameters": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                   else str(v))
                               for k, v in vars(model).items()
                               if k not in ("diagnostics", "cohort_membership")}}
        print(f"\n=== {name} -> {model.name} ({type(model).__name__}) ===", flush=True)
        started = time.time()
        frames.append(run_arm(model, cohort, seeds, out_dir))
        spec["seconds"] = round(time.time() - started, 1)
        if not args.no_determinism:
            probe = determinism_probe(factory, cohort)
            probe["arm"] = model.name
            probes.append(probe)
            spec["determinism"] = probe
            print(f"  determinism fold 0: max |delta| {probe['max_abs_delta']:.2e} "
                  f"({probe['n_rows_moved']} rows moved)", flush=True)
            if not probe["within_tolerance"]:
                raise SystemExit(f"{model.name} is not reproducible: {probe}")
        manifest["arms"].append(spec)

    board = finalise(frames, cohort, out_dir)
    if probes:
        pd.DataFrame(probes).to_csv(out_dir / "determinism.csv", index=False)
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("\n" + board.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
