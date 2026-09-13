"""Run the stage-2 exploratory ladder (``S2_*``) on the frozen cohort and fold plan.

    .venv/Scripts/python.exe generations/gen13_separation/scripts/g13_run_stage2.py --label S2_main
    .venv/Scripts/python.exe generations/gen13_separation/scripts/g13_run_stage2.py --label S2_lean \
        --blocks COND,MASSACT,PHYSCHEM,DONORS,COORD
    .venv/Scripts/python.exe generations/gen13_separation/scripts/g13_run_stage2.py --label S2_smoke --seeds 104729

The references ``C_DIRECT_ROW`` and ``M_PHYSICS_radius+radius_sq`` are always included so that
every stage-2 contrast is paired pair-for-pair inside one run; ``--arms`` narrows the ladder.
Score the result with ``g13_analysis.py --label <label>`` exactly as for the locked runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen13sep import paths  # noqa: E402
from gen13sep.arms_stage2 import (stage2_arms, stage3_arms, stage3_weight_arms,  # noqa: E402
                                  stage4_arms)
from gen13sep.cohort import build_cohort  # noqa: E402
from gen13sep.features import BLOCK_ORDER, build_features  # noqa: E402
from gen13sep.models import DirectRowArm, MeanCurveArm, PhysicsBasisArm  # noqa: E402
from gen13sep.runner import RunSpec, run_ladder  # noqa: E402
from gen13sep.splits import SPLIT_SEEDS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="S2_main")
    ap.add_argument("--design", default="B", choices=["A", "B"])
    ap.add_argument("--key-mode", default="exact", choices=["exact", "relaxed", "series"])
    ap.add_argument("--blocks", default=",".join(BLOCK_ORDER))
    ap.add_argument("--arms", default=None, help="comma-separated arm names; default = the whole stage-2 ladder")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SPLIT_SEEDS))
    args = ap.parse_args()

    t0 = time.time()
    cohort = build_cohort(args.key_mode)
    features = build_features(cohort)
    references = [MeanCurveArm(), DirectRowArm(),
                  PhysicsBasisArm(("radius", "radius_sq"), name="M_PHYSICS_radius+radius_sq")]
    ladder = references + stage2_arms() + stage3_arms() + stage3_weight_arms() + stage4_arms()
    seen: set[str] = set()
    ladder = [a for a in ladder if not (a.name in seen or seen.add(a.name))]
    if args.arms:
        wanted = set(args.arms.split(","))
        ladder = [a for a in ladder if a.name in wanted]
        missing = wanted - {a.name for a in ladder}
        if missing:
            raise SystemExit(f"unknown arms: {sorted(missing)}")
    spec = RunSpec(design=args.design, blocks=tuple(args.blocks.split(",")),
                   seeds=tuple(int(s) for s in args.seeds.split(",")), key_mode=args.key_mode,
                   label=args.label)
    print(f"cohort {cohort.fingerprint()} cells {len(cohort.frame)} | blocks {spec.blocks} | "
          f"arms {[a.name for a in ladder]}", flush=True)
    run_ladder(cohort, features, ladder, spec)
    print(f"done in {time.time() - t0:.0f}s -> {paths.PREDICTION_DIR / args.label}")


if __name__ == "__main__":
    main()
