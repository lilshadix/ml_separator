"""Which auxiliary metals help unseen lanthanide chemistry, and which hurt it.

Brief §14.  A single pooled delta cannot distinguish an archive that transfers
from one in which two metals cancel, so this driver takes the pool apart.

Usage::

    PYTHONPATH=src python scripts/gen11_negative_transfer.py --stage plan
    PYTHONPATH=src python scripts/gen11_negative_transfer.py --stage equivalence
    PYTHONPATH=src python scripts/gen11_negative_transfer.py --stage fit --seeds 104729
    PYTHONPATH=src python scripts/gen11_negative_transfer.py --stage analyse

Four things are true of every number it writes.

**The baseline is design-matched.**  Every auxiliary arm runs at metal scheme
``GENERAL`` and MASSACTION ``ANNOTATION_SAFE`` because the frozen alternatives
are a provenance flag on auxiliary rows (``runner.ANNOTATION_COUPLED_MASSACTION``).
Comparing such an arm to the bit-for-bit gen10 anchor would therefore charge the
design change to the auxiliary data.  The headline reference is
``A_GEN10_CONTROL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1``; the anchor is
reported too, as its own row, labelled as the design effect it is.

**The full-pool result and the ablations appear side by side.**  §14 permits
excluding a harmful group only in a pre-declared follow-up ablation, never
retroactively from the headline, so ``metal_group_ablations.csv`` carries a
``source`` column and the full-pool row is always present.

**Leave-one-metal-out is run reduced, and the reduction is stated.**  A LOMO arm
is a full refit of a nearly-full pool, so the sweep costs ``n_groups`` fits
whatever the group sizes are.  The default here is a single split seed, which
buys the per-metal ranking and the blocked chemotype bootstrap but *cannot* speak
to split-seed variation — the quantity §17 cares about.  Everything computed from
the pre-registered arms (composition ablations, chemotype deltas, per-ligand
regressions) keeps all five seeds; only the LOMO sweep is reduced, and each row
carries its own ``n_seeds``.

**A missing arm is reported, not skipped silently.**  The primary run writes into
``runs/gen11_transfer/arms`` while this runs, so the driver operates on whatever
exists and names what was absent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lanthanide_separation.gen7.harness import DEFAULT_SEEDS  # noqa: E402
from lanthanide_separation.gen10.runner import prepared_cohort  # noqa: E402
from lanthanide_separation.gen11 import negative as neg  # noqa: E402
from lanthanide_separation.gen11.overlap import build_overlap_map  # noqa: E402
from lanthanide_separation.gen11.pools import build_pool  # noqa: E402
from lanthanide_separation.gen11.runner import RunSpec, run_spec  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen11_transfer"
PRIMARY_ARMS = OUT / "arms"
NEG = OUT / "negative"
NEG_ARMS = NEG / "arms"
AUX_FEATURES = OUT / "featurizer" / "aux_features.parquet"

#: The design corner every auxiliary arm sits at, and therefore the only honest
#: control for one.  Written once so no table can quietly use a different one.
DESIGN = dict(mechanism="JOINT", metal_scheme="GENERAL",
              massaction_scheme="ANNOTATION_SAFE", weighting="HIERARCHICAL", aux_lambda=1.0)

CONTROL_MATCHED = RunSpec("A_GEN10_CONTROL", **DESIGN).name
CONTROL_ANCHOR = RunSpec("A_GEN10_CONTROL", mechanism="JOINT", metal_scheme="FROZEN_3",
                         massaction_scheme="FROZEN_8", weighting="HIERARCHICAL").name
FULL_POOL = RunSpec("E_LN_PLUS_ALL", **DESIGN).name
COMPOSITION_ARMS = {
    "B_LN_EXPANDED": RunSpec("B_LN_EXPANDED", **DESIGN).name,
    "C_LN_PLUS_ACTINIDES": RunSpec("C_LN_PLUS_ACTINIDES", **DESIGN).name,
    "D_LN_PLUS_NON_ACTINIDE": RunSpec("D_LN_PLUS_NON_ACTINIDE", **DESIGN).name,
    "E_LN_PLUS_ALL": FULL_POOL,
}
DESIGN_GRID = {
    "metal_scheme_only": RunSpec("A_GEN10_CONTROL", metal_scheme="GENERAL",
                                 massaction_scheme="FROZEN_8").name,
    "massaction_only": RunSpec("A_GEN10_CONTROL", metal_scheme="FROZEN_3",
                               massaction_scheme="ANNOTATION_SAFE").name,
    "both": CONTROL_MATCHED,
}


# --------------------------------------------------------------------------- #
# Arm discovery
# --------------------------------------------------------------------------- #

def oof_path(directory: Path, name: str) -> Path:
    return directory / f"oof_{name.replace('|', '__')}.parquet"


def load_arm(name: str, *, seeds: list[int] | None = None) -> pd.DataFrame | None:
    """The OOF frame for ``name`` from either arm directory, or ``None``.

    The negative directory is searched *after* the primary one so that a locally
    refitted stand-in never shadows the arm the primary run produced.
    """
    for directory in (PRIMARY_ARMS, NEG_ARMS):
        path = oof_path(directory, name)
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if seeds is not None and not set(seeds) <= set(frame["split_seed"].unique()):
            continue
        return frame
    return None


def align(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[int]]:
    """Restrict every arm to the split seeds all of them share.

    An inner join on ``(seed, row_id)`` would do this silently; doing it here
    makes ``n_seeds`` a reported property of each comparison instead of an
    invisible one.
    """
    common = None
    for frame in frames.values():
        seeds = set(int(s) for s in frame["split_seed"].unique())
        common = seeds if common is None else (common & seeds)
    common = sorted(common or [])
    return ({k: v[v["split_seed"].isin(common)].copy() for k, v in frames.items()}, common)


# --------------------------------------------------------------------------- #
# Pool + groups
# --------------------------------------------------------------------------- #

def build(seeds: list[int]):
    cohort = prepared_cohort()
    if not AUX_FEATURES.exists():
        raise SystemExit(f"auxiliary features not found: {AUX_FEATURES}")
    overlap = build_overlap_map(cohort.frame)
    aux_features = pd.read_parquet(AUX_FEATURES)
    build_result = build_pool(cohort, overlap, aux_features, policy="HEADLINE", seeds=seeds)
    return cohort, build_result.pool


def plan(pool) -> tuple[tuple[neg.MetalGroup, ...], pd.DataFrame]:
    groups = neg.metal_groups(pool.features)
    groups = groups + (neg.actinide_group(pool.features),)
    table = neg.groups_table(groups, n_pool=len(pool.features))
    return groups, table


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def stage_equivalence(seeds: list[int]) -> dict:
    """Is one seed of a five-seed run the same as a one-seed run of that arm?

    The reduced LOMO sweep pairs single-seed arms against the primary run's
    multi-seed reference, which is only legitimate if the answer is yes.  The
    harness derives folds from the split seed alone and the model seed from the
    fold index alone, so it should be — and this refits the frozen anchor at one
    seed and checks, because "should be" is not a measurement.  Cost: one fit.
    """
    reference = load_arm(CONTROL_ANCHOR)
    if reference is None:
        return {"status": "SKIPPED", "reason": f"{CONTROL_ANCHOR} not on disk yet"}
    seed = seeds[0]
    if seed not in set(int(s) for s in reference["split_seed"].unique()):
        return {"status": "SKIPPED", "reason": f"seed {seed} absent from the anchor"}

    scratch = NEG / "equivalence"
    spec = neg.LabelledRunSpec("A_GEN10_CONTROL", mechanism="JOINT", metal_scheme="FROZEN_3",
                               massaction_scheme="FROZEN_8", weighting="HIERARCHICAL",
                               label="SEEDSLICE")
    path = oof_path(scratch, spec.name)
    if path.exists():
        candidate = pd.read_parquet(path)
    else:
        cohort, pool = build([seed])
        candidate = run_spec(spec, cohort, pool.restrict(np.zeros(len(pool.features), bool)),
                             scratch, seeds=[seed], verbose=True)

    left = reference[reference["split_seed"] == seed].set_index("row_id")["prediction"]
    right = candidate.set_index("row_id")["prediction"]
    joined = left.to_frame("five").join(right.to_frame("one"), how="inner")
    delta = (joined["five"] - joined["one"]).abs()
    result = {"status": "OK", "seed": int(seed), "n_rows": int(len(joined)),
              "max_abs_delta": float(delta.max()), "n_rows_moved": int((delta > 0).sum())}
    if len(joined) != len(right):
        result["status"] = "ROW_MISMATCH"
    elif result["max_abs_delta"] > 1e-12:
        result["status"] = "NOT_EQUIVALENT"
    return result


def stage_fit(seeds: list[int], *, skip_existing: bool, dry_run: bool,
              groups_wanted: set[str] | None = None) -> dict:
    """Fit every leave-one-group-out arm, plus the size-matched random removals."""
    cohort, pool = build(seeds)
    groups, table = plan(pool)
    NEG.mkdir(parents=True, exist_ok=True)
    table.to_csv(NEG / "metal_groups.csv", index=False)

    # Sharing eight cores with the primary run costs roughly 1,000 s per arm, so
    # this sweep will very likely be stopped by the clock rather than by finishing.
    # The order is therefore chosen so that **every prefix is interpretable**, not
    # so the largest effects come first: a ranking of per-metal deltas with no
    # scale attached to it is not a result.  The scale-setting arms come fourth
    # and fifth, before the remaining metals:
    #
    #   1. the largest metal — the arm most able to move anything;
    #   2. actinides as one block — §14's actinide-vs-non-actinide contrast, and
    #      incidentally the cheapest arm in the sweep (its pool is ~10 % of full);
    #   3. the smallest group — a removal too small to change the chemistry, so
    #      its delta is the sweep's refit noise floor;
    #   4. the size-matched random removal — the control that separates "this
    #      metal" from "this many rows";
    #   5. the rest of the metals §14 names, then the sparse remainders.
    largest = max((g for g in groups if g.kind == "metal"), key=lambda g: g.n_cells)

    def rank(group: neg.MetalGroup) -> tuple[int, int]:
        if group.key == largest.key:
            return (0, 0)
        if group.kind == "category":
            return (1, 0)
        if group.key == "other_actinide":
            return (2, 0)
        if group.kind == "metal" and group.category == "actinide":
            return (4, -group.n_cells)
        return (5, -group.n_cells)

    candidates: list[tuple[tuple[int, int], str, str, object]] = [
        (rank(group), group.key, group.label, neg.drop_metals(pool, group.metals))
        for group in groups
    ]
    # The first random removal sits at priority 3, right after the noise floor and
    # ahead of every remaining metal; the rest trail at the end.
    for index, draw in enumerate(neg.RANDOM_REMOVAL_SEEDS):
        key = f"RANDOM_{largest.n_cells}_d{draw}"
        candidates.append(((3 if index == 0 else 6, index), key, f"LOMO_{key}",
                           neg.drop_random_matched(pool, largest.n_cells, seed=draw)))

    jobs: list[tuple[str, neg.LabelledRunSpec, object]] = [
        (key, neg.LabelledRunSpec("E_LN_PLUS_ALL", **DESIGN, label=label), restricted)
        for _, key, label, restricted in sorted(candidates, key=lambda c: c[0])
    ]

    if groups_wanted:
        jobs = [job for job in jobs if job[0] in groups_wanted]

    # The full pool is the reference for every LOMO arm, so nothing in the sweep
    # is interpretable without it; it is refitted here if the primary run has not
    # reached it.  The design-matched *control* is deliberately not refitted: it
    # is the primary run's own arm 4, it is needed only for the headline
    # comparison in ``--stage analyse``, and a one-seed stand-in would be a
    # strictly worse duplicate of a five-seed arm that is already on its way.
    # ``--stage equivalence`` measures whether reusing the primary's slice is
    # legitimate and records the answer in ``seed_slice_equivalence.json``; run it
    # before trusting a mixed-seed comparison.
    if load_arm(FULL_POOL, seeds=seeds) is None:
        jobs.insert(0, (f"REFERENCE:{FULL_POOL}",
                        neg.LabelledRunSpec("E_LN_PLUS_ALL", **DESIGN), pool))

    print(f"gen11 negative-transfer fit: {len(jobs)} arms x {len(seeds)} seed(s)", flush=True)
    for key, spec, restricted in jobs:
        print(f"    {key:32s} pool={len(restricted.features):5d}  {spec.name}", flush=True)
    if dry_run:
        return {"planned": len(jobs), "dry_run": True}

    NEG_ARMS.mkdir(parents=True, exist_ok=True)
    done, started = [], time.time()
    for index, (key, spec, restricted) in enumerate(jobs, start=1):
        target = oof_path(NEG_ARMS, spec.name)
        if skip_existing and target.exists():
            print(f"[{index}/{len(jobs)}] {key} — cached", flush=True)
            done.append(key)
            continue
        print(f"[{index}/{len(jobs)}] {key} ({time.time() - started:.0f}s elapsed)", flush=True)
        run_spec(spec, cohort, restricted, NEG_ARMS, seeds=seeds, verbose=True)
        done.append(key)
    return {"fitted": done, "seconds": round(time.time() - started, 1),
            "seeds": [int(s) for s in seeds]}


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _contribution_row(group_row: dict, frames: dict, lomo: str, full: str,
                      replicates: int) -> dict:
    """One per-metal contribution: ``MAE(without g) - MAE(full pool)``.

    ``reference`` is the arm *without* the group, so ``reference - candidate`` is
    the group's contribution with no sign flip anywhere downstream.
    """
    aligned, seeds = align({lomo: frames[lomo], full: frames[full]})
    summary = neg.comparison_summary(aligned, reference=lomo, candidate=full,
                                     replicates=replicates)
    signs = neg.per_seed_signs(aligned, reference=lomo, candidate=full)
    chem = neg.chemotype_deltas(aligned, reference=lomo, candidate=full)
    ligands = neg.ligand_deltas(aligned, reference=lomo, candidate=full)
    row = dict(group_row)
    row.update({
        "seeds": "|".join(str(s) for s in seeds),
        "n_seeds": len(seeds),
        "full_pool_macro_mae": summary["macro_mae_candidate"],
        "without_group_macro_mae": summary["macro_mae_reference"],
        "contribution_macro_mae": summary["macro_mae_delta"],
        "contribution_level_mae": summary["level_mae_delta"],
        "contribution_shape_mae": summary["shape_mae_delta"],
        "ci95_low": summary.get("boot_mae_ci95_low"),
        "ci95_high": summary.get("boot_mae_ci95_high"),
        "p_group_not_helpful": summary.get("boot_mae_p_worse_one_sided"),
        "units_group_helps": summary.get("boot_mae_units_improved"),
        "units_total": summary.get("boot_mae_units_total"),
        "n_seeds_group_helps": signs["n_seeds_positive"],
        "n_seeds_group_hurts": signs["n_seeds_negative"],
        "chemotypes_group_helps": int((chem["macro_mae_delta"] > 0).sum()),
        "chemotypes_group_hurts": int((chem["macro_mae_delta"] < 0).sum()),
        "ligands_group_helps": int((ligands["mae_delta"] > 0).sum()),
        "ligands_group_hurts": int((ligands["mae_delta"] < 0).sum()),
        # ``ligands`` is sorted ascending on ``mae_delta = without_group - full``.
        # The last row is the ligand that most needs the group; the first is the
        # ligand the group most damages — the nameable victim §14 asks for.
        "worst_ligand_when_group_removed": str(ligands.iloc[-1]["extractant"])
        if len(ligands) else "",
        "worst_ligand_when_group_present": str(ligands.iloc[0]["extractant"])
        if len(ligands) else "",
        "worst_ligand_when_group_present_delta": float(ligands.iloc[0]["mae_delta"])
        if len(ligands) else float("nan"),
        "direction": ("HELPS" if summary["macro_mae_delta"] > 0
                      else "HURTS" if summary["macro_mae_delta"] < 0 else "FLAT"),
    })
    return row


def stage_analyse(replicates: int) -> dict:
    NEG.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "replicates": replicates}
    missing: list[str] = []

    groups_file = NEG / "metal_groups.csv"
    if not groups_file.exists():
        raise SystemExit("run --stage plan or --stage fit first: metal_groups.csv is missing")
    groups = pd.read_csv(groups_file)

    full = load_arm(FULL_POOL)
    control = load_arm(CONTROL_MATCHED)
    anchor = load_arm(CONTROL_ANCHOR)
    if full is None:
        missing.append(FULL_POOL)
    if control is None:
        missing.append(CONTROL_MATCHED)

    # ---------------------------------------------------------------- #
    # 1. per-metal contribution (LOMO)
    # ---------------------------------------------------------------- #
    # ``GROUP_all_actinide`` is a *union* of partition members, not a member.  It
    # belongs in the ablation table; including it here would count Am, U, Th, Pu,
    # Np and Cm twice in a table whose rows are supposed to be disjoint.
    partition = groups[groups["kind"] != "category"]
    n_pool = int(partition["n_cells"].sum())
    contributions: list[dict] = []
    if full is not None:
        for _, group in partition.iterrows():
            name = f"{FULL_POOL}|LOMO_{group['group']}"
            frame = load_arm(name)
            if frame is None:
                missing.append(name)
                continue
            contributions.append(_contribution_row(
                {"group": group["group"], "kind": group["kind"],
                 "metal_category": group["metal_category"], "metals": group["metals"],
                 "n_cells": int(group["n_cells"]), "pool_share": float(group["pool_share"])},
                {name: frame, FULL_POOL: full}, name, FULL_POOL, replicates))
        for path in sorted(NEG_ARMS.glob("*LOMO_RANDOM_*.parquet")):
            frame = pd.read_parquet(path)
            name = str(frame["model"].iloc[0])
            key = name.rsplit("|LOMO_", 1)[-1]
            n_cells = int(key.split("_")[1])
            contributions.append(_contribution_row(
                {"group": key, "kind": "random_matched", "metal_category": "n/a",
                 "metals": "", "n_cells": n_cells,
                 "pool_share": n_cells / n_pool if n_pool else float("nan")},
                {name: frame, FULL_POOL: full}, name, FULL_POOL, replicates))

    contribution_table = pd.DataFrame(contributions)
    if not contribution_table.empty:
        contribution_table = contribution_table.sort_values("contribution_macro_mae")
    contribution_table.to_csv(NEG / "per_metal_contribution.csv", index=False)
    report["per_metal_contribution_rows"] = int(len(contribution_table))
    report["noise_floor"] = neg.noise_floor(contribution_table) if len(contribution_table) else {}
    report["size_control"] = neg.size_control(contribution_table) if len(contribution_table) else {}

    # ---------------------------------------------------------------- #
    # 2. metal-group ablations
    # ---------------------------------------------------------------- #
    ablations: list[dict] = []

    def add(source: str, label: str, reference_name: str, candidate_name: str,
            reference: pd.DataFrame, candidate: pd.DataFrame, note: str) -> None:
        aligned, seeds = align({reference_name: reference, candidate_name: candidate})
        summary = neg.comparison_summary(aligned, reference=reference_name,
                                         candidate=candidate_name, replicates=replicates)
        signs = neg.per_seed_signs(aligned, reference=reference_name, candidate=candidate_name)
        ablations.append({
            "source": source, "label": label, "note": note,
            "reference_arm": reference_name, "candidate_arm": candidate_name,
            "seeds": "|".join(str(s) for s in seeds), "n_seeds": len(seeds),
            "macro_mae_reference": summary["macro_mae_reference"],
            "macro_mae_candidate": summary["macro_mae_candidate"],
            "macro_mae_delta": summary["macro_mae_delta"],
            "level_mae_delta": summary["level_mae_delta"],
            "shape_mae_delta": summary["shape_mae_delta"],
            "ci95_low": summary.get("boot_mae_ci95_low"),
            "ci95_high": summary.get("boot_mae_ci95_high"),
            "p_worse_one_sided": summary.get("boot_mae_p_worse_one_sided"),
            "units_improved": summary.get("boot_mae_units_improved"),
            "units_total": summary.get("boot_mae_units_total"),
            "n_seeds_positive": signs["n_seeds_positive"],
            "n_seeds_negative": signs["n_seeds_negative"],
            "level_ci95_low": summary.get("boot_offset_mae_ci95_low"),
            "level_ci95_high": summary.get("boot_offset_mae_ci95_high"),
            "shape_ci95_low": summary.get("boot_shape_mae_ci95_low"),
            "shape_ci95_high": summary.get("boot_shape_mae_ci95_high"),
        })

    if control is not None:
        for key, name in COMPOSITION_ARMS.items():
            frame = load_arm(name)
            if frame is None:
                missing.append(name)
                continue
            add("preregistered_arm", key, CONTROL_MATCHED, name, control, frame,
                "auxiliary composition vs the design-matched control; "
                "positive delta = auxiliary data helped")
    if anchor is not None:
        for key, name in DESIGN_GRID.items():
            frame = load_arm(name)
            if frame is None:
                missing.append(name)
                continue
            add("design_control", key, CONTROL_ANCHOR, name, anchor, frame,
                "no auxiliary rows in either arm; this is the design change alone, "
                "and it must be subtracted before reading any auxiliary arm")

    if full is not None:
        actinide_name = f"{FULL_POOL}|LOMO_{neg.ALL_ACTINIDE_KEY}"
        actinide = load_arm(actinide_name)
        if actinide is not None:
            add("lomo_group", "drop_all_actinides", actinide_name, FULL_POOL, actinide, full,
                "leave-one-group-out from the full pool; "
                "positive delta = the actinides helped")
        majors = groups[(groups["metal_category"] == "actinide")
                        & (groups["kind"] == "metal")]["group"].tolist()
        for group in majors:
            name = f"{FULL_POOL}|LOMO_{group}"
            frame = load_arm(name)
            if frame is None:
                continue
            add("lomo_within_actinide", f"drop_{group}", name, FULL_POOL, frame, full,
                "leave-one-metal-out from the full pool; positive delta = the metal helped")
        for group in groups[groups["metal_category"].isin(
                ["lanthanide", "alkaline_earth", "transition_metal",
                 "rare_earth_non_lanthanide", "post_transition_metal"])]["group"]:
            name = f"{FULL_POOL}|LOMO_{group}"
            frame = load_arm(name)
            if frame is None:
                continue
            add("lomo_non_actinide", f"drop_{group}", name, FULL_POOL, frame, full,
                "leave-one-metal-out from the full pool; positive delta = the group helped")

    ablation_table = pd.DataFrame(ablations)
    ablation_table.to_csv(NEG / "metal_group_ablations.csv", index=False)
    report["metal_group_ablation_rows"] = int(len(ablation_table))

    # ---------------------------------------------------------------- #
    # 3/4/5. chemotype deltas, level-vs-shape, per-ligand regressions
    # ---------------------------------------------------------------- #
    pairs: list[tuple[str, str, str, pd.DataFrame, pd.DataFrame, bool]] = []
    if control is not None and full is not None:
        pairs.append(("HEADLINE_full_pool_vs_control", CONTROL_MATCHED, FULL_POOL,
                      control, full, True))
    if control is not None:
        for key, name in COMPOSITION_ARMS.items():
            if name == FULL_POOL:
                continue
            frame = load_arm(name)
            if frame is not None:
                pairs.append((f"ARM_{key}_vs_control", CONTROL_MATCHED, name,
                              control, frame, False))
    if full is not None:
        for _, group in groups.iterrows():
            name = f"{FULL_POOL}|LOMO_{group['group']}"
            frame = load_arm(name)
            if frame is not None:
                pairs.append((f"LOMO_{group['group']}", name, FULL_POOL, frame, full, False))

    chemotype_rows, ligand_rows, level_shape_rows = [], [], []
    for label, ref_name, cand_name, ref, cand, headline in pairs:
        aligned, seeds = align({ref_name: ref, cand_name: cand})
        chem = neg.chemotype_deltas(aligned, reference=ref_name, candidate=cand_name)
        chem.insert(0, "comparison", label)
        chem.insert(1, "n_seeds", len(seeds))
        chem["rank"] = np.arange(1, len(chem) + 1)
        chemotype_rows.append(chem)

        ligands = neg.ligand_deltas(aligned, reference=ref_name, candidate=cand_name)
        # ``ligand_deltas`` already carries its own per-ligand ``n_seeds``.
        ligands.insert(0, "comparison", label)
        ligands["rank_worst_first"] = np.arange(1, len(ligands) + 1)
        ligands["is_headline_comparison"] = headline
        # The headline keeps every ligand — a negative result has to be auditable
        # in full.  Other comparisons keep the tail that names the regression.
        ligand_rows.append(ligands if headline else ligands.head(15))

        summary = neg.comparison_summary(aligned, reference=ref_name, candidate=cand_name,
                                         replicates=replicates)
        level_shape_rows.append({
            "comparison": label, "reference_arm": ref_name, "candidate_arm": cand_name,
            "n_seeds": len(seeds),
            "macro_mae_reference": summary["macro_mae_reference"],
            "macro_mae_candidate": summary["macro_mae_candidate"],
            "macro_mae_delta": summary["macro_mae_delta"],
            "level_mae_reference": summary["level_mae_reference"],
            "level_mae_candidate": summary["level_mae_candidate"],
            "level_mae_delta": summary["level_mae_delta"],
            "shape_mae_reference": summary["shape_mae_reference"],
            "shape_mae_candidate": summary["shape_mae_candidate"],
            "shape_mae_delta": summary["shape_mae_delta"],
            "boot_offset_mae_delta": summary.get("boot_offset_mae_delta"),
            "boot_offset_ci95_low": summary.get("boot_offset_mae_ci95_low"),
            "boot_offset_ci95_high": summary.get("boot_offset_mae_ci95_high"),
            "boot_shape_mae_delta": summary.get("boot_shape_mae_delta"),
            "boot_shape_ci95_low": summary.get("boot_shape_mae_ci95_low"),
            "boot_shape_ci95_high": summary.get("boot_shape_mae_ci95_high"),
            "chemotypes_improved": int((chem["macro_mae_delta"] > 0).sum()),
            "chemotypes_worsened": int((chem["macro_mae_delta"] < 0).sum()),
            "ligands_improved": int((ligands["mae_delta"] > 0).sum()),
            "ligands_worsened": int((ligands["mae_delta"] < 0).sum()),
        })

    if chemotype_rows:
        pd.concat(chemotype_rows, ignore_index=True).to_csv(
            NEG / "chemotype_deltas.csv", index=False)
    if ligand_rows:
        pd.concat(ligand_rows, ignore_index=True).to_csv(
            NEG / "worst_regressions.csv", index=False)
    level_shape = pd.DataFrame(level_shape_rows)
    level_shape.to_csv(NEG / "level_vs_shape.csv", index=False)
    report["comparisons"] = int(len(level_shape))

    # Two checks that can fail.  The auxiliary weight share must be lam/(1+lam)
    # in every arm — that is what makes a LOMO delta a redistribution rather than
    # a smaller pool, and therefore what makes the arms comparable to each other.
    lam = float(DESIGN["aux_lambda"])
    shares = neg.weight_share_audit(NEG_ARMS, expected=lam / (1.0 + lam))
    if not shares.empty:
        shares.to_csv(NEG / "weight_share_audit.csv", index=False)
        report["weight_share_arms_as_designed"] = int(shares["as_designed"].sum())
        report["weight_share_arms_total"] = int(len(shares))
        report["weight_share_expected"] = lam / (1.0 + lam)
        off = shares[~shares["as_designed"]]
        report["weight_share_violations"] = off["arm"].tolist()

    if control is not None and full is not None:
        aligned, _ = align({CONTROL_MATCHED: control, FULL_POOL: full})
        report["decompose_agreement"] = neg.assert_matches_decompose(
            aligned, reference=CONTROL_MATCHED, candidate=FULL_POOL)

    report["missing_arms"] = sorted(set(missing))
    (NEG / "negative_summary.json").write_text(json.dumps(report, indent=2, default=str))
    write_readme(report, contribution_table, ablation_table, level_shape, groups)
    return report


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #

def write_readme(report: dict, contributions: pd.DataFrame, ablations: pd.DataFrame,
                 level_shape: pd.DataFrame, groups: pd.DataFrame) -> None:
    def table(frame: pd.DataFrame, columns: list[str], limit: int = 40) -> str:
        """A markdown pipe table.

        Hand-rolled rather than ``DataFrame.to_markdown`` because that needs
        ``tabulate``, which this environment does not have — and installing a
        package to format a report is exactly the kind of environment drift that
        silently shifted every number in an earlier generation.
        """
        available = [c for c in columns if c in frame.columns]
        if frame.empty or not available:
            return "_(no arms on disk yet)_"
        block = frame[available].head(limit)
        lines = ["| " + " | ".join(available) + " |",
                 "| " + " | ".join("---" for _ in available) + " |"]
        for _, row in block.iterrows():
            cells = []
            for column in available:
                value = row[column]
                cells.append(f"{value:.4f}" if isinstance(value, (float, np.floating))
                             and np.isfinite(value) else str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    floor = report.get("noise_floor", {})
    floor_line = (f"largest |contribution| among chemically-inert removals: "
                  f"**{floor['floor_abs_max']:.4f}** over {floor['n_floor_arms']} arms "
                  f"({floor.get('floor_arms', '')})"
                  if floor.get("n_floor_arms") else "_not yet computable_")
    n_lomo_seeds = int(contributions["n_seeds"].max()) if len(contributions) else 0

    if "weight_share_arms_total" in report:
        weight_share_line = (
            f"{report['weight_share_arms_as_designed']} of "
            f"{report['weight_share_arms_total']} arms at the expected "
            f"{report['weight_share_expected']:.3f}")
        if report.get("weight_share_violations"):
            weight_share_line += (" — VIOLATIONS: "
                                  + ", ".join(report["weight_share_violations"]))
    else:
        weight_share_line = "not yet computable"

    size = report.get("size_control", {})
    if size.get("n_random_arms") and "matched_metal" in size:
        size_line = (
            f"{size['n_random_arms']} random removal(s) of {size['random_n_cells']} cells: "
            f"contribution {size['random_contribution_mean']:+.4f} "
            f"(range {size['random_contribution_min']:+.4f} .. "
            f"{size['random_contribution_max']:+.4f}).  "
            f"**{size['matched_metal']}** at the same n contributes "
            f"{size['matched_metal_contribution']:+.4f}, an excess of "
            f"**{size['excess_over_random']:+.4f}** over a size-matched random removal.")
    elif size.get("n_random_arms"):
        size_line = (f"{size['n_random_arms']} random removal(s) of "
                     f"{size['random_n_cells']} cells: contribution "
                     f"{size['random_contribution_mean']:+.4f}; the metal they are matched to "
                     f"is not yet on disk.")
    else:
        size_line = "_no size-matched random removal on disk yet_"

    equivalence = NEG / "seed_slice_equivalence.json"
    equivalence_line = "_not run_"
    if equivalence.exists():
        payload = json.loads(equivalence.read_text())
        equivalence_line = (
            f"status **{payload.get('status')}**, max |delta| "
            f"{payload.get('max_abs_delta', float('nan')):.3g} over "
            f"{payload.get('n_rows', 0)} rows"
            if payload.get("status") == "OK" else f"status **{payload.get('status')}** "
            f"({payload.get('reason', '')})")

    headline = level_shape[level_shape["comparison"] == "HEADLINE_full_pool_vs_control"] \
        if "comparison" in level_shape.columns else pd.DataFrame()
    if len(headline):
        row = headline.iloc[0]
        headline_block = (
            f"Full auxiliary pool against the design-matched control, "
            f"{int(row['n_seeds'])} seed(s):\n\n"
            f"| quantity | control | full pool | delta (control - full) |\n"
            f"| --- | --- | --- | --- |\n"
            f"| macro MAE | {row['macro_mae_reference']:.4f} | "
            f"{row['macro_mae_candidate']:.4f} | {row['macro_mae_delta']:+.4f} |\n"
            f"| level MAE | {row['level_mae_reference']:.4f} | "
            f"{row['level_mae_candidate']:.4f} | {row['level_mae_delta']:+.4f} |\n"
            f"| shape MAE | {row['shape_mae_reference']:.4f} | "
            f"{row['shape_mae_candidate']:.4f} | {row['shape_mae_delta']:+.4f} |\n\n"
            f"Chemotypes improved {int(row['chemotypes_improved'])} / worsened "
            f"{int(row['chemotypes_worsened'])}; ligands improved "
            f"{int(row['ligands_improved'])} / worsened {int(row['ligands_worsened'])}.  "
            f"A positive delta means the auxiliary pool helped.")
    else:
        headline_block = "_the full-pool arm or its design-matched control is not on disk yet_"

    text = f"""# gen11 §14 — negative transfer and per-metal contribution

Generated {report.get('generated')}.  Every number here is computed by
`scripts/gen11_negative_transfer.py`; none is carried over from another run.

## Headline, reported before any ablation

{headline_block}

## Sign convention

    contribution(g) = macro_MAE(pool without g) - macro_MAE(full pool)

    contribution > 0  ->  removing g made prediction worse   ->  g HELPED
    contribution < 0  ->  removing g made prediction better  ->  g HURT  (negative transfer)

For the composition arms in `metal_group_ablations.csv` the convention is the
usual one, `reference - candidate`, so a positive `macro_mae_delta` means the
candidate arm is better than its reference.

## What "removing a metal" does

`HIERARCHICAL` weighting at `aux_lambda = 1` sets the auxiliary block's *total*
sample-weight mass to the lanthanide core's, whatever the block contains.  A LOMO
arm therefore does not train on less auxiliary influence — it redistributes the
removed metal's mass over the metals that remain.  `contribution(g)` answers
**"is g worth its place in the pool"**, not "is more auxiliary data better"; the
latter is the headline above.  `weight_share_audit.csv` checks that the
renormalisation actually held in every fitted arm ({weight_share_line}); if it
did not, the contributions are not comparable across arms and nothing below
should be read.

## The baseline

Auxiliary arms cannot run at the frozen design: `FROZEN_3` leaves 98.4 % of
auxiliary rows without a metal coordinate and `FROZEN_8`'s annotation-coupled
MASSACTION terms become an "is this an auxiliary row" flag.  The comparison
baseline is therefore the control at the *same* design corner,
`{CONTROL_MATCHED}`.  The bit-for-bit gen10 anchor `{CONTROL_ANCHOR}` appears in
`metal_group_ablations.csv` under `source=design_control`, which is the design
change measured on its own, with no auxiliary rows in either arm.  Read it first;
it must be subtracted before any auxiliary arm means anything.

## What was reduced

The leave-one-metal-out sweep refits the whole model once per group, and the cost
does not depend on the group's size — dropping 22 cells costs what dropping 1,841
costs.  It was run at **{n_lomo_seeds} split seed(s)** rather than five, because
the machine was simultaneously running the primary arm sweep.  That buys the
per-metal ranking and the chemotype-blocked bootstrap inside those seeds.  It
**cannot** show split-seed direction consistency, which is the §17 requirement,
so `n_seeds_group_helps` / `n_seeds_group_hurts` in
`per_metal_contribution.csv` are bounded by that seed count and must not be read
as five-seed agreement.  Everything derived from the pre-registered composition
arms keeps all five seeds; each row carries its own `n_seeds`.

Pairing a one-seed LOMO arm against a five-seed reference is only legitimate if
one seed of a five-seed run is bit-identical to a one-seed run of the same arm.
That was measured, not assumed, by refitting the frozen anchor at a single seed
and differencing: {equivalence_line} — the thread-order floor, i.e. identical.

## Noise floor

Removing a group that is a fraction of a percent of the pool cannot change the
chemistry, so the spread of those deltas is the sweep's own refit noise.
{floor_line}.  A per-metal contribution below that magnitude is not a
measurement.

## Size control

The pool is 90 % actinide and its largest metal is a third of it, so "removing
that metal hurts" and "removing a third of the training rows hurts" are the same
sentence until a random removal of the same n is run.  The random arms
(`kind=random_matched`) are that control; they are deliberately **not** folded
into the noise floor, because removing a third of the pool is a real
intervention.

{size_line}

## Per-metal contribution

{table(contributions, ['group', 'kind', 'n_cells', 'pool_share', 'contribution_macro_mae',
                       'ci95_low', 'ci95_high', 'contribution_level_mae',
                       'contribution_shape_mae', 'chemotypes_group_helps',
                       'chemotypes_group_hurts', 'direction'])}

Full table: `per_metal_contribution.csv`.

## Metal-group ablations

{table(ablations, ['source', 'label', 'n_seeds', 'macro_mae_delta', 'ci95_low', 'ci95_high',
                   'level_mae_delta', 'shape_mae_delta', 'units_improved', 'units_total'])}

Full table: `metal_group_ablations.csv`.  **The full-pool result and every
ablation are reported side by side.**  §14 permits excluding a harmful metal
group only as a pre-declared follow-up ablation; no row here may be substituted
for the headline.

## Level versus shape

{table(level_shape, ['comparison', 'n_seeds', 'macro_mae_delta', 'level_mae_delta',
                     'shape_mae_delta', 'chemotypes_improved', 'chemotypes_worsened',
                     'ligands_improved', 'ligands_worsened'])}

Full table: `level_vs_shape.csv`.  The level is the per-(seed, ligand) mean
residual and the shape the deviation about it; gen7-gen10 showed they move
independently, so a macro gain that is entirely level and a macro gain that is
entirely shape are different results.

## Group definition

A metal gets its own leave-one-out arm at >= {neg.MIN_CELLS_FOR_OWN_GROUP} pool
cells; below that it joins its `metal_category` remainder.  The rule is declared
in `gen11/negative.py` before the sweep runs, and the groups partition the pool
exactly (checked in `metal_groups`).

`n_cells` counts **pool cells**, i.e. after `pools.prepare_auxiliary` averages
replicates onto one row per (extractant, condition, metal) — the same
granularity a cohort training row has.  It is therefore smaller than the archive
record counts the brief quotes (Am 1841, U 995, Th 761, Pu 709, Np 432, Cm 136):
5,438 records reduce to 5,096 cells.  `GROUP_all_actinide` is a union of
partition members, not a member; it appears in the ablation table and is
deliberately absent from `per_metal_contribution.csv`, whose rows must be
disjoint.

{table(groups, ['group', 'kind', 'metal_category', 'metals', 'n_cells', 'pool_share'])}

## Files

| file | what |
| --- | --- |
| `per_metal_contribution.csv` | leave-one-metal-out contribution per group, with blocked bootstrap CI |
| `metal_group_ablations.csv` | pre-registered composition arms, the design-control grid, and the group-level LOMO |
| `chemotype_deltas.csv` | per-chemotype delta for every comparison, ranked worst first |
| `worst_regressions.csv` | per-ligand deltas; all ligands for the headline, worst 15 elsewhere |
| `level_vs_shape.csv` | the level/shape split of every comparison |
| `metal_groups.csv` | the partition |
| `weight_share_audit.csv` | auxiliary weight share per arm; it must be lam/(1+lam) everywhere or the arms are not comparable |
| `seed_slice_equivalence.json` | proof that a one-seed run is a slice of a five-seed run |
| `negative_summary.json` | machine-readable summary, including which arms were absent |

## Arms absent when this ran

{chr(10).join('- `' + m + '`' for m in report.get('missing_arms', [])) or '_none_'}
"""
    (NEG / "README.md").write_text(text)


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["plan", "equivalence", "fit", "analyse", "all"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[DEFAULT_SEEDS[0]],
                        help="split seeds for the LOMO sweep (default: one, see README)")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--refit", dest="skip_existing", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cpu", type=int, default=0,
                        help="cap joblib's worker count so a concurrent primary run survives")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="restrict the LOMO sweep to these group keys (default: all)")
    args = parser.parse_args(argv)

    if args.max_cpu > 0:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(args.max_cpu)

    if args.stage in ("plan", "all"):
        cohort, pool = build(args.seeds)
        groups, table = plan(pool)
        NEG.mkdir(parents=True, exist_ok=True)
        table.to_csv(NEG / "metal_groups.csv", index=False)
        print(table.to_string(index=False))
        print(f"pool cells {len(pool.features)}  groups {len(groups)}")
        if args.stage == "plan":
            return 0

    if args.stage in ("equivalence", "all"):
        result = stage_equivalence(args.seeds)
        print("seed-slice equivalence:", json.dumps(result))
        (NEG / "seed_slice_equivalence.json").write_text(json.dumps(result, indent=2))
        if result.get("status") not in ("OK", "SKIPPED"):
            raise SystemExit(f"single-seed runs are not a slice of multi-seed runs: {result}")
        if args.stage == "equivalence":
            return 0

    if args.stage in ("fit", "all"):
        print(json.dumps(stage_fit(args.seeds, skip_existing=args.skip_existing,
                                   dry_run=args.dry_run,
                                   groups_wanted=set(args.groups) if args.groups else None),
                         default=str))
        if args.stage == "fit":
            return 0

    if args.stage in ("analyse", "all"):
        report = stage_analyse(args.replicates)
        print(json.dumps({k: v for k, v in report.items() if k != "noise_floor"},
                         indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
