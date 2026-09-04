"""Run gen11 transfer arms under gen10's frozen evaluation contract.

Usage::

    PYTHONPATH=src python scripts/gen11_run.py --stage primary
    PYTHONPATH=src python scripts/gen11_run.py --stage mechanisms
    PYTHONPATH=src python scripts/gen11_run.py --stage all --seeds 104729

The stages are ordered by what they can rule out.  ``primary`` establishes
whether *any* auxiliary composition moves the benchmark and, crucially, runs the
control under both metal representations so that a transfer effect can be told
apart from a metal-representation effect.  Everything after it only matters if
``primary`` shows something, except the negative controls, which matter most when
it does.

Every arm writes its own OOF parquet in the gen8/gen9/gen10 schema, so the k-shot
frontier is produced afterwards by ``gen10_final_locked.py`` pointed at
``runs/gen11_transfer/arms`` — gen11 does not re-implement the adapter.
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
from lanthanide_separation.gen10.runner import prepared_cohort  # noqa: E402
from lanthanide_separation.gen11.arms import MATCHED_SEEDS  # noqa: E402
from lanthanide_separation.gen11.overlap import build_overlap_map  # noqa: E402
from lanthanide_separation.gen11.pools import build_pool, write_pool_audit  # noqa: E402
from lanthanide_separation.gen11.runner import RunSpec, leaderboard, run_spec  # noqa: E402
from lanthanide_separation.gen11.transfer import nesting_delta  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen11_transfer"
ARMS_DIR = OUT / "arms"
AUX_FEATURES = OUT / "featurizer" / "aux_features.parquet"

#: The headline composition sweep, plus the full control decomposition.
#:
#: Auxiliary arms cannot use the frozen design: ``FROZEN_3`` leaves 98.4 % of
#: auxiliary rows with no metal coordinate, and ``FROZEN_8``'s annotation-coupled
#: MASSACTION terms turn into an "is auxiliary" flag.  Both changes therefore
#: apply to every auxiliary arm — which means the control must be run at all four
#: corners, or "auxiliary data helped" would be indistinguishable from "the design
#: change helped".  The four controls cost four runs and buy the whole
#: interpretation.
CONTROL_GRID: tuple[RunSpec, ...] = (
    # the anchor: bit-for-bit gen10
    RunSpec("A_GEN10_CONTROL", metal_scheme="FROZEN_3", massaction_scheme="FROZEN_8"),
    # each design change alone
    RunSpec("A_GEN10_CONTROL", metal_scheme="GENERAL", massaction_scheme="FROZEN_8"),
    RunSpec("A_GEN10_CONTROL", metal_scheme="FROZEN_3", massaction_scheme="ANNOTATION_SAFE"),
    # the design every auxiliary arm actually uses — the comparison baseline
    RunSpec("A_GEN10_CONTROL", metal_scheme="GENERAL", massaction_scheme="ANNOTATION_SAFE"),
)

PRIMARY: tuple[RunSpec, ...] = CONTROL_GRID + tuple(
    RunSpec(arm, metal_scheme="GENERAL", massaction_scheme="ANNOTATION_SAFE")
    for arm in ("B_LN_EXPANDED", "C_LN_PLUS_ACTINIDES",
                "D_LN_PLUS_NON_ACTINIDE", "E_LN_PLUS_ALL")
)

#: §6 — the three mechanisms must not be collapsed into one experiment.
MECHANISM_ARMS: tuple[RunSpec, ...] = tuple(
    RunSpec(arm, mechanism=mechanism, metal_scheme="GENERAL")
    for arm in ("C_LN_PLUS_ACTINIDES", "E_LN_PLUS_ALL")
    for mechanism in ("SHARED_SHAPE", "SHARED_LEVEL", "AUX_PRETRAIN")
)

#: §12 — an improvement must survive reasonable weighting, chosen in advance.
WEIGHTING_ARMS: tuple[RunSpec, ...] = tuple(
    RunSpec("C_LN_PLUS_ACTINIDES", weighting=weighting, metal_scheme="GENERAL")
    for weighting in ("ROW", "CHEMOTYPE", "METAL_BALANCED")
) + tuple(
    RunSpec("C_LN_PLUS_ACTINIDES", aux_lambda=lam, metal_scheme="GENERAL")
    for lam in (0.25, 0.5, 2.0)
)

#: §19 — these must show no benefit.  If one does, a gain is an artefact.
CONTROL_ARMS: tuple[RunSpec, ...] = tuple(
    RunSpec("C_LN_PLUS_ACTINIDES", control=control, metal_scheme="GENERAL")
    for control in ("PERMUTED_AUX_TARGET", "SHUFFLED_METAL_LABELS")
)

#: §5 F/G — chemical relevance vs row count, at equal n, averaged over draws.
MATCHED_ARMS: tuple[RunSpec, ...] = tuple(
    RunSpec(arm, metal_scheme="GENERAL", matched_seed=seed)
    for arm in ("F_ACTINIDES_ONLY_MATCHED", "G_RANDOM_AUX_MATCHED")
    for seed in MATCHED_SEEDS
)

#: §13 — publication blocking on the composition that matters most.
POLICY_ARMS: tuple[RunSpec, ...] = (
    RunSpec("C_LN_PLUS_ACTINIDES", policy="PUBLICATION_BLOCKED", metal_scheme="GENERAL"),
    RunSpec("E_LN_PLUS_ALL", policy="PUBLICATION_BLOCKED", metal_scheme="GENERAL"),
)

STAGES = {
    "primary": PRIMARY,
    "mechanisms": MECHANISM_ARMS,
    "weighting": WEIGHTING_ARMS,
    "controls": CONTROL_ARMS,
    "matched": MATCHED_ARMS,
    "policy": POLICY_ARMS,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="primary",
                        choices=[*STAGES, "all"], help="which pre-registered block to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", type=Path, default=ARMS_DIR)
    parser.add_argument("--aux", type=Path, default=AUX_FEATURES)
    parser.add_argument("--skip-existing", action="store_true",
                        help="do not refit an arm whose OOF parquet is already on disk")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    specs = ([s for stage in STAGES.values() for s in stage] if args.stage == "all"
             else list(STAGES[args.stage]))
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"gen11 stage={args.stage}  arms={len(specs)}  seeds={args.seeds}", flush=True)
    for spec in specs:
        print("   ", spec.name)
    if args.dry_run:
        return 0

    cohort = prepared_cohort()
    overlap = build_overlap_map(cohort.frame)
    if not args.aux.exists():
        raise SystemExit(f"auxiliary features not found: {args.aux}\n"
                         "run the featurizer first; gen11 refuses to invent them")
    aux_features = pd.read_parquet(args.aux)

    # The control must still *be* the control.  Asserted before anything is fitted,
    # because a nesting failure invalidates every comparison drawn afterwards.
    nesting = nesting_delta(cohort, seed=args.seeds[0])
    print(f"nesting vs gen10 RecomposedModel: max |delta| {nesting['max_abs_delta']:.3g}", flush=True)
    if nesting["max_abs_delta"] > 1e-12:
        raise SystemExit(f"gen11 no longer nests gen10: {nesting}")

    pools = {}
    for policy in sorted({spec.policy for spec in specs}):
        build = build_pool(cohort, overlap, aux_features, policy=policy, seeds=args.seeds)
        summary = write_pool_audit(build, OUT / "overlap" / f"pool_{policy.lower()}")
        print(f"pool[{policy}]: {json.dumps(summary)}", flush=True)
        pools[policy] = build.pool

    frames, started = [], time.time()
    for index, spec in enumerate(specs, start=1):
        target = args.out / f"oof_{spec.name.replace('|', '__')}.parquet"
        if args.skip_existing and target.exists():
            print(f"[{index}/{len(specs)}] {spec.name} — cached", flush=True)
            frames.append(pd.read_parquet(target))
            continue
        print(f"[{index}/{len(specs)}] {spec.name}", flush=True)
        frames.append(run_spec(spec, cohort, pools[spec.policy], args.out, seeds=args.seeds))

    board = leaderboard(frames, args.out)
    print(f"\n=== gen11 {args.stage} leaderboard ({time.time() - started:.0f}s) ===")
    print(board.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
