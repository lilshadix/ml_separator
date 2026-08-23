#!/usr/bin/env python
"""The k-shot frontier and the attribution chain.

Reads the detail tables ``gen9_kshot.py`` writes and answers the question the brief
(§22) insists must not be collapsed into a single best number: **which component
bought what.**

The chain is walked in order, each step changing exactly one thing:

```text
OLD GLOBAL + CENTRAL     + OLD ADAPTER
NEW GLOBAL + CENTRAL     + OLD ADAPTER      <- the shape objective
OLD GLOBAL + LEARNED ACQ + OLD ADAPTER      <- the acquisition learner
NEW GLOBAL + LEARNED ACQ + OLD ADAPTER      <- are they additive?
NEW GLOBAL + LEARNED ACQ + NEW ADAPTER      <- series-local calibration
```

Two cohorts are always reported and always named. **Table A** is maximal coverage:
every ligand eligible at each k, so the population changes with k. **Table B** is the
common cohort — the identical ligand set at every k — and the adaptation curve is
read off Table B, because a curve whose population shrinks as k grows can fall for
reasons that have nothing to do with calibration.

Before any number is quoted, two audits run and can fail the script: a pairing audit
(arms called paired must share fold, ligand, repeat, pool size and evaluation size)
and a row-accounting audit (no arm may look better because its difficult rows
vanished).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen9.audit import pairing_audit  # noqa: E402

RUN = REPO_ROOT / "runs" / "gen9_shape"
COMMON_MIN_POOL = 5
K_VALUES = (0, 1, 2, 3, 5)
FAILURE_CLASSES = (REPO_ROOT / "runs" / "gen8_architecture" / "case_studies"
                   / "ligand_failure_classes.csv")

#: gen8's common-cohort frontier, for the longitudinal column.
GEN8_FRONTIER = {0: 1.0605, 1: 0.6674, 2: 0.5879, 3: 0.5230, 5: 0.4743}


def label_arms(detail: pd.DataFrame) -> pd.DataFrame:
    work = detail.copy()
    work["arm"] = np.where(work["adapter"] == "ZERO_SHOT_REF", "ZERO_SHOT",
                           work["adapter"] + "@" + work["policy"])
    if "global_model" not in work.columns:
        work["global_model"] = work.get("model", "UNKNOWN")
    work["full_arm"] = work["global_model"] + " | " + work["arm"]
    return work


def common_cohort(detail: pd.DataFrame, *, min_pool: int = COMMON_MIN_POOL) -> set:
    """Ligands with enough pool rows in every repeat of every seed — gen8's Table B."""
    ok = detail.groupby("extractant").apply(
        lambda block: bool((block["n_pool"] >= min_pool).all() and (block["n_eval"] >= 2).all()),
        include_groups=False)
    return set(ok.index[ok])


def macro_table(detail: pd.DataFrame, ligands: set | None) -> pd.DataFrame:
    work = detail if ligands is None else detail[detail["extractant"].isin(ligands)]
    per_ligand = work.groupby(["full_arm", "global_model", "arm", "k", "extractant"])[
        "mae"].mean().reset_index()
    return per_ligand.groupby(["full_arm", "global_model", "arm", "k"]).agg(
        mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()


def row_audit(detail: pd.DataFrame) -> pd.DataFrame:
    """Per arm: how many units it produced, and whether any metric went non-finite."""
    rows = []
    for arm, block in detail.groupby("full_arm", sort=True):
        rows.append({
            "full_arm": arm,
            "n_records": int(len(block)),
            "n_ligands": int(block["extractant"].nunique()),
            "n_units": int(block[["split_seed", "fold", "extractant", "repeat", "k"]]
                           .drop_duplicates().shape[0]),
            "nan_mae": int(block["mae"].isna().sum()),
            "inf_mae": int(np.isinf(block["mae"].to_numpy(dtype=float)).sum()),
            "min_n_eval": int(block["n_eval"].min()),
            "min_n_pool": int(block["n_pool"].min()),
        })
    return pd.DataFrame.from_records(rows)


def build_chain(table: pd.DataFrame, spec: list[dict], k: int) -> pd.DataFrame:
    rows = []
    keyed = table[table["k"] == k].set_index("full_arm")["mae"]
    counts = table[table["k"] == k].set_index("full_arm")["n_ligands"]
    for step in spec:
        arm = f"{step['model']} | {step['arm']}"
        rows.append({"step": step["step"], "full_arm": arm, "k": k,
                     "mae": float(keyed.get(arm, np.nan)),
                     "n_ligands": int(counts.get(arm, 0))})
    frame = pd.DataFrame.from_records(rows)
    frame["delta_from_previous"] = frame["mae"].shift(1) - frame["mae"]
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, nargs="+", required=True)
    parser.add_argument("--old-model", default="REC_ecfp_plus_recovered")
    parser.add_argument("--new-model", default=None)
    parser.add_argument("--old-adapter", default="OFFSET_K1")
    parser.add_argument("--new-adapter", default="SERIES_MAP")
    parser.add_argument("--old-policy", default="CENTRAL")
    parser.add_argument("--new-policy", default="LEARNED_RANK")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=RUN / "frontier")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    detail = pd.concat([pd.read_parquet(p) for p in args.detail], ignore_index=True)
    detail = label_arms(detail)
    detail = detail.drop_duplicates(["global_model", "split_seed", "fold", "extractant",
                                     "repeat", "policy", "adapter", "k"])
    models = sorted(detail["global_model"].unique())
    new_model = args.new_model or next((m for m in models if m.startswith("GEN9_")
                                        and "ROW_ONLY" not in m), args.old_model)
    print(f"{len(detail):,} records; global models {models}; new = {new_model}")

    audit = row_audit(detail)
    audit.to_csv(args.out / "row_audit.csv", index=False)
    bad = audit[(audit["nan_mae"] > 0) | (audit["inf_mae"] > 0)]
    if len(bad):
        raise SystemExit(f"non-finite MAE in some arms:\n{bad}")
    spread = audit["n_units"].nunique()
    print(f"row audit: {len(audit)} arms, {spread} distinct unit counts "
          f"({'uniform' if spread == 1 else 'NOT uniform — see row_audit.csv'})")

    cohort = common_cohort(detail)
    print(f"common cohort: {len(cohort)} ligands")
    table_a = macro_table(detail, None)
    table_b = macro_table(detail, cohort)
    table_a.to_csv(args.out / "table_a_maximal_coverage.csv", index=False)
    table_b.to_csv(args.out / "table_b_common_cohort.csv", index=False)

    print("\n--- best deployable arms per k, common cohort ---")
    deployable = table_b[~table_b["arm"].str.contains("ORACLE")]
    for k in K_VALUES:
        block = deployable[deployable["k"] == k].nsmallest(4, "mae")
        if len(block):
            print(f"k={k}  gen8 {GEN8_FRONTIER.get(k, float('nan')):.4f}")
            print(block[["full_arm", "mae", "n_ligands"]].to_string(index=False))

    # --- attribution chain -------------------------------------------------
    spec = [
        {"step": "OLD GLOBAL + CENTRAL + OLD ADAPTER",
         "model": args.old_model, "arm": f"{args.old_adapter}@{args.old_policy}"},
        {"step": "NEW GLOBAL + CENTRAL + OLD ADAPTER",
         "model": new_model, "arm": f"{args.old_adapter}@{args.old_policy}"},
        {"step": "OLD GLOBAL + LEARNED ACQ + OLD ADAPTER",
         "model": args.old_model, "arm": f"{args.old_adapter}@{args.new_policy}"},
        {"step": "NEW GLOBAL + LEARNED ACQ + OLD ADAPTER",
         "model": new_model, "arm": f"{args.old_adapter}@{args.new_policy}"},
        {"step": "NEW GLOBAL + LEARNED ACQ + NEW ADAPTER",
         "model": new_model, "arm": f"{args.new_adapter}@{args.new_policy}"},
    ]
    chains = [build_chain(table_b, spec, k) for k in (1, 2)]
    chain = pd.concat(chains, ignore_index=True)
    chain.to_csv(args.out / "attribution_chain.csv", index=False)
    print("\n--- attribution chain (common cohort) ---")
    print(chain.to_string(index=False))

    # --- adaptation curve --------------------------------------------------
    # The curve a reader should see is the *best deployable arm per global model at
    # each k* — the frontier — not the fixed arms of the attribution chain, which are
    # held constant on purpose and are therefore not anybody's best choice.
    deployable = table_b[~table_b["arm"].str.contains("ORACLE")]
    frontier = (deployable.sort_values("mae")
                .groupby(["global_model", "k"], as_index=False).first())
    keep = [args.old_model, new_model, "GEN9_REL_MONOLITH", "GEN9_A0_ROW_ONLY"]
    curve = frontier[frontier["global_model"].isin(keep)][
        ["global_model", "arm", "k", "mae"]].rename(columns={"global_model": "arm",
                                                             "arm": "best_adapter_policy"})
    curve.to_csv(args.out / "adaptation_curve.csv", index=False)
    frontier.to_csv(args.out / "frontier_by_model.csv", index=False)

    # --- paired intervals --------------------------------------------------
    per_unit = detail[detail["extractant"].isin(cohort)].groupby(
        ["full_arm", "extractant", "tanimoto_cluster", "split_seed", "k"])[
        "mae"].mean().reset_index().rename(columns={"full_arm": "arm"})
    bootstraps = []
    for k in (1, 2, 5):
        block = per_unit[per_unit["k"] == k]
        arms = set(block["arm"])
        comparisons = {}
        for i in range(1, len(spec)):
            a = f"{spec[i - 1]['model']} | {spec[i - 1]['arm']}"
            b = f"{spec[i]['model']} | {spec[i]['arm']}"
            if a in arms and b in arms:
                comparisons[f"{spec[i]['step']} vs {spec[i - 1]['step']}"] = (a, b)
        first = f"{spec[0]['model']} | {spec[0]['arm']}"
        last = f"{spec[-1]['model']} | {spec[-1]['arm']}"
        if first in arms and last in arms:
            comparisons["FINAL vs BASELINE"] = (first, last)
        if not comparisons:
            continue
        try:
            table = paired_chemotype_bootstrap(block, comparisons, replicates=args.replicates)
        except ValueError as error:
            print(f"  (k={k} bootstrap skipped: {error})")
            continue
        table.insert(0, "k", k)
        bootstraps.append(table)
    if bootstraps:
        frame = pd.concat(bootstraps, ignore_index=True)
        frame.to_csv(args.out / "frontier_bootstrap.csv", index=False)
        print("\n--- paired chemotype-blocked intervals (positive = candidate better) ---")
        print(frame[["k", "comparison", "point", "ci95_low", "ci95_high", "n_units",
                     "units_improved", "seeds_positive"]].to_string(index=False))

    # --- pairing audit -----------------------------------------------------
    chain_arms = [f"{s['model']} | {s['arm']}" for s in spec]
    present = [a for a in chain_arms if a in set(detail["full_arm"])]
    if len(present) > 1:
        pairing = pairing_audit(detail[detail["k"] == 1], present, arm_column="full_arm",
                                keys=("split_seed", "fold", "extractant", "repeat",
                                      "n_pool", "n_eval"))
        pairing.to_csv(args.out / "pairing_audit.csv", index=False)
        unpaired = pairing[~pairing["identical"]]
        print(f"\npairing audit: {len(pairing)} pairs, {len(unpaired)} not identical")
        if len(unpaired):
            print(unpaired[["arm_a", "arm_b", "n_units_a", "n_units_b", "n_shared"]]
                  .to_string(index=False))

    # --- failure-class strata ---------------------------------------------
    if FAILURE_CLASSES.exists():
        classes = pd.read_csv(FAILURE_CLASSES)[["extractant", "failure_class"]]
        joined = detail.merge(classes, on="extractant", how="left")
        joined = joined[joined["extractant"].isin(cohort)]
        strata = joined.groupby(["full_arm", "k", "failure_class", "extractant"])[
            "mae"].mean().reset_index()
        strata = strata.groupby(["full_arm", "k", "failure_class"]).agg(
            mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()
        strata.to_csv(args.out / "failure_class_frontier.csv", index=False)
        focus = strata[(strata["k"].isin([0, 1])) & (strata["full_arm"].isin(chain_arms))]
        if len(focus):
            print("\n--- by gen8 failure class (locked strata) ---")
            print(focus.pivot_table(index=["full_arm", "k"], columns="failure_class",
                                    values="mae").to_string())

    (args.out / "frontier.json").write_text(json.dumps({
        "sources": [str(p) for p in args.detail], "models": models,
        "new_model": new_model, "common_cohort_size": len(cohort),
        "n_records": int(len(detail)), "gen8_reference": GEN8_FRONTIER,
        "chain": spec, "replicates": int(args.replicates),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
