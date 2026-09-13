"""Run the Gen13 separation ladder on the frozen cohort and folds.

    .venv/Scripts/python.exe gen13_separation/scripts/g13_run_ladder.py --label B_primary
    .venv/Scripts/python.exe gen13_separation/scripts/g13_run_ladder.py --label B_ecfp_only --blocks COND,MASSACT,ECFP
    .venv/Scripts/python.exe gen13_separation/scripts/g13_run_ladder.py --arms M_LOWRANK_K2,C_DIRECT_ROW --seeds 104729
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen13sep import paths  # noqa: E402
from gen13sep.cohort import build_cohort, write_cohort  # noqa: E402
from gen13sep.features import BLOCK_ORDER, build_features  # noqa: E402
from gen13sep.models import AntisymmetricPairArm, default_ladder, exploratory_arms, helper_arms, selected_arm, v2_arms  # noqa: E402
from gen13sep.runner import RunSpec, run_ladder  # noqa: E402
from gen13sep.splits import SPLIT_SEEDS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="B_primary")
    ap.add_argument("--design", default="B", choices=["A", "B", "BP", "BR", "BQ"])
    ap.add_argument("--key-mode", default="exact", choices=["exact", "relaxed", "series"])
    ap.add_argument("--blocks", default=",".join(BLOCK_ORDER))
    ap.add_argument("--arms", default=None, help="comma-separated arm names; default = full ladder")
    ap.add_argument("--with-pair-arm", action="store_true", help="add the slow antisymmetric pair control")
    ap.add_argument("--with-exploratory", action="store_true", help="add the X_ exploratory arms")
    ap.add_argument("--with-3d", action="store_true", help="append the RESP3D block and run the 3D helper arms only")
    ap.add_argument("--with-logk", action="store_true", help="append the LOGK prior block and run its helper arms only")
    ap.add_argument("--v2", action="store_true", help="run the gen13.1 exploratory ladder (block subsets, bagging, gain calibration)")
    ap.add_argument("--with-stage2", action="store_true", help="append the second session's stage-2/3/4 arms (gen13sep.arms_stage2)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SPLIT_SEEDS))
    ap.add_argument("--coordination", default="frozen", choices=["frozen", "posthoc"])
    args = ap.parse_args()

    t0 = time.time()
    cohort = build_cohort(args.key_mode)
    write_cohort(cohort)
    features = build_features(cohort, coordination=args.coordination, with_resp3d=args.with_3d, with_logk=args.with_logk)
    ladder = default_ladder() + [selected_arm()]
    if args.v2:
        ladder = [a for a in default_ladder() if a.name in ("B1_MEAN_CURVE", "C_DIRECT_ROW", "M_PHYSICS_radius+radius_sq", "M_LOWRANK_K2")] + v2_arms()
    if args.with_stage2:
        from gen13sep.arms_stage2 import stage2_arms, stage3_arms, stage4_arms
        seen = {a.name for a in ladder}
        for a in stage4_arms() + stage2_arms() + stage3_arms():
            if a.name not in seen:
                ladder.append(a); seen.add(a.name)
    if args.with_3d or args.with_logk:
        ladder = helper_arms("RESP3D") if args.with_3d else []
        if args.with_logk:
            ladder = ladder + [a for a in helper_arms("LOGK") if not (args.with_3d and a.name.startswith("M_PHYSICS"))]
    if args.with_exploratory:
        ladder += exploratory_arms()
    if args.with_pair_arm:
        ladder.append(AntisymmetricPairArm())
    if args.arms:
        wanted = set(args.arms.split(","))
        ladder = [a for a in ladder if a.name in wanted]
        missing = wanted - {a.name for a in ladder}
        if missing:
            raise SystemExit(f"unknown arms: {sorted(missing)}")
    blocks = tuple(args.blocks.split(",")) + (("RESP3D",) if args.with_3d else ()) + (("LOGK",) if args.with_logk else ())
    spec = RunSpec(design=args.design, blocks=blocks,
                   seeds=tuple(int(s) for s in args.seeds.split(",")), key_mode=args.key_mode,
                   label=args.label)
    print(f"cohort {cohort.fingerprint()} cells {len(cohort.frame)} | blocks {spec.blocks} | arms {[a.name for a in ladder]}")
    run_ladder(cohort, features, ladder, spec)
    print(f"done in {time.time() - t0:.0f}s -> {paths.PREDICTION_DIR / args.label}")


if __name__ == "__main__":
    main()
