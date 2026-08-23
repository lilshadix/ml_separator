#!/usr/bin/env python
"""GEN9-A — train the global model with a within-curve objective.

Runs one or more :class:`ShapeArm` configurations through gen7's evaluation
contract — same cohort, same fold plan, same design matrix, same model seeds — and
writes an out-of-fold prediction table per arm in the schema every downstream gen8
tool already reads.  That last point is deliberate: a gen9 arm's OOF parquet drops
straight into ``evaluate_fewshot``, ``curve_statistics`` and the bootstrap without
a single adapter being rewritten, so a gen9 number and a gen8 number are the same
arithmetic on the same rows.

Three stages, run separately so nothing is decided before it is measured:

``--stage phase1``
    the brief's §4 screen: ``ROW_ONLY``, ``ROW_ADJACENT``, ``ROW_ENDPOINT``,
    ``ROW_RANDOM_PAIR``, ``ROW_MULTISCALE`` at one loss weight, few seeds.  The
    question is only *which sampler moves the flattening*, so it is scored on slope
    and span, not on macro MAE.
``--stage grid``
    the §6 weight sweep for whichever samplers survived, still on the screening
    seeds.  Development, labelled as such.
``--stage finalists``
    the locked five-seed evaluation of the one or two arms chosen above, plus the
    ``FROZEN_ET`` anchor and the ``A0_ROW_ONLY`` control that every shape claim is
    quoted against.

Every arm writes ``diagnostics.json`` carrying the per-fold pair audit (how many
pairs, per axis, and the largest share any one curve holds) and the loss history
(``L_row``, ``lambda_delta L_delta``, ``lambda_span L_span`` every ten iterations),
because "the shape arm won by switching the row loss off" and "the shape arm won
because one 40-point titration dominated the gradient" are both results about the
implementation rather than about chemistry, and both are invisible in a leaderboard.
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

from lanthanide_separation.gen7.harness import (  # noqa: E402
    DEFAULT_SEEDS, build_folds, evaluate_contender, load_cohort, score_oof,
)
from lanthanide_separation.gen9.audit import (  # noqa: E402
    assert_fold_audit_clean, assert_row_accounting_clean, curve_pair_audit,
    fold_audit, row_accounting,
)
from lanthanide_separation.gen9.objective import (  # noqa: E402
    LAMBDA_DELTA_EXTENSION, LAMBDA_DELTA_GRID, LAMBDA_SPAN_GRID, BoostConfig,
    ShapeArm, phase1_arms,
)
from lanthanide_separation.gen9.train import (  # noqa: E402
    CurveBoostContender, FrozenExtraTrees, ShapeRecomposed, load_membership,
)

OUT = REPO_ROOT / "runs" / "gen9_shape"
CONFIG_PATH = OUT / "boost_config.json"


def load_config() -> BoostConfig:
    """The learner hyperparameters, selected once inside training folds and frozen.

    Written by ``scripts/gen9_tune.py``.  Refusing to fall back to a default is
    deliberate: an arm trained under different hyperparameters from the arm it is
    compared with is not a comparison, and a silent default is exactly how that
    happens.
    """
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"{CONFIG_PATH} is missing — run `scripts/gen9_tune.py` first so every "
            "arm shares hyperparameters chosen inside training folds only")
    payload = json.loads(CONFIG_PATH.read_text())
    return BoostConfig(**payload["config"])


def stage_arms(stage: str, *, sampler: str, lambda_delta: float) -> list[ShapeArm]:
    if stage == "phase1":
        return phase1_arms(lambda_delta=lambda_delta)
    if stage == "grid":
        arms = []
        for d in LAMBDA_DELTA_GRID:
            for s in LAMBDA_SPAN_GRID:
                tag = f"G_{sampler.replace('ROW_', '')}_d{d:g}_s{s:g}".replace(".", "")
                arms.append(ShapeArm(tag, sampler, float(d), float(s)))
        return arms
    if stage == "grid_strong":
        # The upward extension declared in ``objective.LAMBDA_DELTA_EXTENSION``,
        # plus the cluster-matched pair weighting.  Both are EXPLORATORY: they were
        # written after the phase-1 screen was read.
        arms = []
        for d in LAMBDA_DELTA_EXTENSION:
            for balance in ("curve", "cluster"):
                tag = f"X_{sampler.replace('ROW_', '')}_d{d:g}_{balance}".replace(".", "")
                arms.append(ShapeArm(tag, sampler, float(d), 0.0, balance=balance))
        for d in (1.0, 2.0):
            arms.append(ShapeArm(f"X_{sampler.replace('ROW_', '')}_d{d:g}_span01_cluster"
                                 .replace(".", ""), sampler, float(d), 0.1, balance="cluster"))
        return arms
    if stage == "axes":
        return [
            ShapeArm(f"AX_{name}", sampler, lambda_delta, 0.0, axis_set=name)
            for name in ("L", "LH", "LHM", "LHMT")
        ] + [ShapeArm("AX_LIGAND_BALANCE", sampler, lambda_delta, 0.0, balance="ligand"),
             ShapeArm("AX_CLUSTER_BALANCE", sampler, lambda_delta, 0.0, balance="cluster")]
    raise ValueError(f"unknown stage {stage!r}")


def run_arm(contender, cohort, seeds, out_dir: Path, *, verbose: bool = True) -> pd.DataFrame:
    started = time.time()
    oof = evaluate_contender(contender, cohort, seeds=seeds, verbose=verbose)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out_dir / f"oof_{contender.name}.parquet", index=False)
    if hasattr(contender, "diagnostics") and contender.diagnostics:
        (out_dir / f"diagnostics_{contender.name}.json").write_text(
            json.dumps(contender.diagnostics, indent=1, default=float))
    if verbose:
        print(f"  [{contender.name}] {time.time() - started:.0f}s total", flush=True)
    return oof


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="phase1",
                        choices=("phase1", "grid", "grid_strong", "axes", "finalists"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sampler", default="ROW_MULTISCALE")
    parser.add_argument("--lambda-delta", type=float, default=0.5)
    parser.add_argument("--arms", nargs="*", default=None,
                        help="finalists stage: NAME:SAMPLER:LAMBDA_DELTA:LAMBDA_SPAN[:AXIS_SET]")
    parser.add_argument("--with-control", action="store_true",
                        help="also run A0_ROW_ONLY, the paired control every shape arm is quoted against")
    parser.add_argument("--with-frozen", action="store_true",
                        help="also refit the frozen ExtraTrees arm as the longitudinal anchor")
    parser.add_argument("--with-recomposed", action="store_true",
                        help="also run the EXPLORATORY relative-position recomposition")
    parser.add_argument("--with-relative-monolith", action="store_true",
                        help="also run the frozen arm with the relative-position columns appended")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    seeds = DEFAULT_SEEDS[:max(1, min(5, args.seeds))]
    tag = args.tag or args.stage
    out_dir = args.out or (OUT / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    print(f"cohort {cohort.frame.shape}  fingerprint {cohort.fingerprint}")
    if cohort.fingerprint != "bed178ec1a7a82b0":
        raise SystemExit(f"cohort fingerprint moved: {cohort.fingerprint}")
    membership = load_membership()
    config = load_config()
    print(f"boost config: {config.as_dict()}")

    # --- fold and curve boundary audit, once, before anything is trained -----
    audits, straddles = [], []
    for seed in seeds:
        folds = build_folds(cohort.frame, seed)
        audits.append(fold_audit(cohort.frame, folds))
        straddles.append(curve_pair_audit(membership, folds, cohort.frame))
    fold_table = pd.concat(audits, ignore_index=True)
    straddle_table = pd.concat(straddles, ignore_index=True)
    assert_fold_audit_clean(fold_table)
    fold_table.to_csv(out_dir / "fold_audit.csv", index=False)
    straddle_table.to_csv(out_dir / "curve_boundary_audit.csv", index=False)
    n_straddle = int(straddle_table["n_curves_straddling_train_test"].sum())
    print(f"fold audit clean; curves straddling a fold boundary: {n_straddle}")
    if n_straddle:
        raise SystemExit("a curve spans the train/test boundary — training pairs would leak")

    overrides: dict[str, dict] = {}
    if args.stage == "finalists":
        if not args.arms and not (args.with_frozen or args.with_recomposed
                                  or args.with_relative_monolith):
            raise SystemExit("--stage finalists needs --arms NAME:SAMPLER:LD:LS[:AXES], "
                             "or --with-frozen / --with-recomposed on their own")
        args.arms = args.arms or []
        arms = []
        for spec in args.arms:
            parts = spec.split(":")
            if len(parts) < 4:
                raise SystemExit(
                    f"arm spec {spec!r} needs at least NAME:SAMPLER:LAMBDA_DELTA:LAMBDA_SPAN")
            arms.append(ShapeArm(parts[0], parts[1], float(parts[2]), float(parts[3]),
                                 axis_set=parts[4] if len(parts) > 4 else "LHM",
                                 balance=parts[5] if len(parts) > 5 else "curve"))
            if len(parts) > 6:
                # A per-arm Huber transition on the delta term.  Everything else in
                # the learner stays identical across arms; this one knob is exposed
                # because the transition point decides whether a curve five log units
                # too flat is pushed harder than one half a unit too flat.
                overrides[arms[-1].name] = {"pair_delta": float(parts[6])}
    else:
        arms = stage_arms(args.stage, sampler=args.sampler, lambda_delta=args.lambda_delta)

    manifest = {"stage": args.stage, "seeds": list(seeds), "tag": tag,
                "cohort_fingerprint": cohort.fingerprint,
                "boost_config": config.as_dict(),
                "arms": [a.as_dict() for a in arms], "results": []}

    if args.with_control and not any(a.name == "A0_ROW_ONLY" for a in arms):
        arms = [ShapeArm("A0_ROW_ONLY", "ROW_ONLY", 0.0, 0.0)] + arms

    frames = []
    if args.with_frozen:
        frozen = FrozenExtraTrees()
        frames.append(run_arm(frozen, cohort, seeds, out_dir))
        manifest["results"].append({"arm": frozen.name, "kind": "frozen_reference"})
    if args.with_relative_monolith:
        print("\n=== REL_MONOLITH (EXPLORATORY: frozen arm + relative-position columns) ===")
        relative = FrozenExtraTrees(with_relative=True, membership=membership)
        frames.append(run_arm(relative, cohort, seeds, out_dir))
        manifest["results"].append({"arm": relative.name, "kind": "relative_monolith_exploratory"})
    if args.with_recomposed:
        print("\n=== SHAPE_RECOMPOSED (EXPLORATORY: relative-position recomposition) ===")
        recomposed = ShapeRecomposed(membership=membership)
        frames.append(run_arm(recomposed, cohort, seeds, out_dir))
        manifest["results"].append({"arm": recomposed.name, "kind": "recomposed_exploratory",
                                    "diagnostics": recomposed.diagnostics[:2]})

    for arm in arms:
        print(f"\n=== {arm.name}  ({arm.strategy}, lambda_delta={arm.lambda_delta}, "
              f"lambda_span={arm.lambda_span}, axes={arm.axis_set}, balance={arm.balance}) ===")
        arm_config = config
        if arm.name in overrides:
            arm_config = BoostConfig(**{**config.as_dict(), **overrides[arm.name]})
            print(f"    per-arm learner override: {overrides[arm.name]}")
        contender = CurveBoostContender(arm=arm, config=arm_config, membership=membership)
        oof = run_arm(contender, cohort, seeds, out_dir)
        frames.append(oof)
        pair_audits = [d["pair_audit"] for d in contender.diagnostics]
        manifest["results"].append({
            "arm": contender.name, "kind": "shape",
            "spec": arm.as_dict(), "config_override": overrides.get(arm.name, {}),
            "median_pairs": float(np.median([p.get("n_pairs", 0) for p in pair_audits])),
            "max_curve_weight_share": float(np.max(
                [p.get("max_curve_weight_share", 0.0) for p in pair_audits])),
            "pairs_by_axis": pair_audits[0].get("pairs_by_axis", {}),
        })

    combined = pd.concat(frames, ignore_index=True)
    accounting = row_accounting(combined, expected_rows=len(cohort.frame))
    accounting.to_csv(out_dir / "row_accounting.csv", index=False)
    assert_row_accounting_clean(accounting)
    print("\nrow accounting clean — no arm lost or duplicated a row")

    combined.to_parquet(out_dir / "oof_all.parquet", index=False)
    scores = score_oof(combined)
    scores.to_csv(out_dir / "scores_by_seed.csv", index=False)
    board = scores.groupby("model")[["macro_mae", "offset_mae", "shape_mae", "pooled_mae"]].mean()
    board["macro_sd"] = scores.groupby("model")["macro_mae"].std()
    board = board.sort_values("macro_mae")
    board.to_csv(out_dir / "leaderboard.csv")
    print("\n" + board.to_string())

    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
