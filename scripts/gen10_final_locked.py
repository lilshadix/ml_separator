#!/usr/bin/env python
"""PHASE 11 — the locked evaluation of the finalists, and the frozen pipeline.

Everything here is scored on artefacts the earlier phases already wrote — the
finalists' OOF parquets — under gen8's few-shot protocol and gen9's shape metrics,
so no number in the final report comes from a code path that was not already
audited.  The candidate set is fixed by ``--finalists`` (registry names from the
Phase 2/4 stages) and the gen9 anchors are always carried.

Produces, under ``runs/gen10_final/final_locked``:

``kshot_detail.parquet``      every (model, adapter, policy, k, ligand, seed, repeat)
``frontier_common.csv``       Table B: the 99-ligand common cohort, every arm, every k
``frontier_best.csv``         the best *deployable* arm per model per k, and the
                              frozen-rule pipeline's number beside it
``pipeline_frontier.json``    the k = 0/1/2/3/5 numbers of the pipeline with its
                              deterministic rules (no per-seed cherry-picking)
``shape_by_axis.csv``         the eight shape endpoints per model per axis
``shape_bootstrap.csv``       paired, chemotype-blocked intervals vs the anchors
``strata.csv``                FROZEN / QUARANTINED / CORRECTED, RESIDUAL_SHAPE /
                              PURE_LEVEL / ALREADY_GOOD, mismatch / clean
``macro_bootstrap.csv``       macro-MAE intervals vs the anchors per cohort
``final_config.json``         the exact configuration of the frozen pipeline
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
from lanthanide_separation.gen8.adapters import default_adapters  # noqa: E402
from lanthanide_separation.gen8.evaluate import evaluate_fewshot  # noqa: E402
from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.slope_restore import build_slope_adapters  # noqa: E402
from lanthanide_separation.gen9.acquisition import register_extra_policies  # noqa: E402
from lanthanide_separation.gen9.metrics import (  # noqa: E402
    curve_shape_table, macro_mae, slope_distribution, summarise_shape,
)
from lanthanide_separation.gen10.adaptation import CorrectedSeriesAdapter  # noqa: E402

GEN9 = REPO_ROOT / "runs" / "gen9_shape"
GEN8 = REPO_ROOT / "runs" / "gen8_architecture"
GEN7 = REPO_ROOT / "runs" / "gen7_architecture"
GEN10 = REPO_ROOT / "runs" / "gen10_final"
OUT = GEN10 / "final_locked"
SEEDS = (104729, 130363, 155921, 196613, 262147)
COMMON_MIN_POOL = 5
K_VALUES = (0, 1, 2, 3, 5)
POLICIES = ("CENTRAL_THEN_SPREAD", "MEDOID", "D_OPTIMAL", "RANDOM")
ANCHORS = {"REC_ecfp_plus_recovered": GEN9 / "finalists" / "oof_all.parquet",
           "GEN9_A0_ROW_ONLY": GEN9 / "finalists" / "oof_all.parquet",
           "GEN9_REL_MONOLITH": GEN9 / "relmono" / "oof_all.parquet",
           "GEN9_SHAPE_RECOMPOSED": GEN9 / "recomposed" / "oof_all.parquet"}
GEN8_FRONTIER = {0: 1.0605, 1: 0.6674, 2: 0.5879, 3: 0.5230, 5: 0.4743}
GEN9_FRONTIER = {0: 1.0358, 1: 0.6539, 2: 0.5746, 3: 0.5109, 5: 0.4675}


def make_fold_trainer(cohort: pd.DataFrame, n_splits: int = 5):
    groups = cohort["tanimoto_cluster"].astype(str).to_numpy()
    target = cohort["log_D"].to_numpy(dtype=float)
    cache: dict = {}

    def trainer(split_seed: int, fold: int):
        if split_seed not in cache:
            cache[split_seed] = list(seeded_group_kfold(groups, n_splits, int(split_seed)))
        train_index, _ = cache[split_seed][int(fold)]
        return cohort.iloc[train_index], target[train_index], 42 + int(fold) * 1009 + 9_999_991

    return trainer


def locate_oof(name: str, stages: list[Path]) -> Path:
    if name in ANCHORS:
        return ANCHORS[name]
    for stage in stages:
        candidate = stage / f"oof_{name}.parquet"
        if candidate.exists():
            return candidate
    raise SystemExit(f"no OOF parquet for {name} under {stages}")


def common_cohort(detail: pd.DataFrame, *, min_pool: int = COMMON_MIN_POOL) -> set:
    ok = detail.groupby("extractant").apply(
        lambda block: bool((block["n_pool"] >= min_pool).all() and (block["n_eval"] >= 2).all()),
        include_groups=False)
    return set(ok.index[ok])


def pipeline_rule(model: str, series_adapter: str | None) -> dict[int, tuple[str, str]]:
    """The frozen, deterministic k -> (adapter, policy) rule."""
    later = series_adapter or "OFFSET_K3"
    return {0: ("ZERO_SHOT", "NONE"),
            1: ("SLOPE_L_s1_K1", "CENTRAL_THEN_SPREAD"),
            2: ("SLOPE_L_s1_K3" if series_adapter is None else later, "CENTRAL_THEN_SPREAD"),
            3: ("SLOPE_L_s1_K3" if series_adapter is None else later, "CENTRAL_THEN_SPREAD"),
            5: ("SLOPE_L_s1_K3" if series_adapter is None else later, "CENTRAL_THEN_SPREAD")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalists", nargs="+", required=True,
                        help="model names (OOF 'model' column) of the gen10 finalists")
    parser.add_argument("--stages", nargs="*", type=Path, default=[
        GEN10 / "feature_access", GEN10 / "level_shape", GEN10 / "axis_representation",
        GEN10 / "set_context"])
    parser.add_argument("--series-adapter", default=None,
                        help="SERIES_ML / SERIES_INNER / SERIES_FIXED if Phase 5 passed; else OFFSET_K3")
    parser.add_argument("--selected", default=None, help="the model being frozen")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--skip-kshot", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    register_extra_policies()

    cohort = pd.read_parquet(GEN7 / "cache" / "cohort.parquet")
    membership = pd.read_parquet(GEN9 / "curves" / "curve_membership.parquet")
    models = list(dict.fromkeys(list(ANCHORS) + list(args.finalists)))
    frames = []
    for name in models:
        path = locate_oof(name, args.stages)
        oof = pd.read_parquet(path)
        oof = oof[(oof["model"] == name) & (oof["split_seed"].isin(SEEDS))]
        if oof.empty:
            raise SystemExit(f"{name}: no rows in {path}")
        frames.append(oof.assign(source=str(path)))
    oof_all = pd.concat(frames, ignore_index=True).drop_duplicates(["model", "split_seed", "row_id"])
    counts = oof_all.groupby(["model", "split_seed"]).size()
    if not (counts == len(cohort)).all():
        raise SystemExit(f"row accounting failed:\n{counts[counts != len(cohort)]}")
    oof_all.to_parquet(args.out / "oof_finalists.parquet", index=False)

    # --- k-shot frontier --------------------------------------------------------
    detail_path = args.out / "kshot_detail.parquet"
    if args.skip_kshot and detail_path.exists():
        detail = pd.read_parquet(detail_path)
    else:
        parts = []
        for name in models:
            oof = oof_all[oof_all["model"] == name].copy()
            adapters = list(default_adapters())
            adapters += [a for a in build_slope_adapters(strengths=(1.0,))
                         if a.name in ("SLOPE_L_s1_K1", "SLOPE_L_s1_K3")]
            if args.series_adapter:
                adapters.append(CorrectedSeriesAdapter(
                    oof=oof, estimator=args.series_adapter.replace("SERIES_", "")))
            started = time.time()
            part = evaluate_fewshot(oof, cohort, adapters, policies=list(POLICIES),
                                    with_oracle_for=("OFFSET_K1",), repeats=args.repeats,
                                    fold_trainer=make_fold_trainer(cohort), verbose=False)
            part["global_model"] = name
            parts.append(part)
            print(f"  [{name}] {len(part):,} k-shot records ({time.time() - started:.0f}s)",
                  flush=True)
        detail = pd.concat(parts, ignore_index=True)
        detail.to_parquet(detail_path, index=False)
    detail["arm"] = np.where(detail["adapter"] == "ZERO_SHOT_REF", "ZERO_SHOT",
                             detail["adapter"] + "@" + detail["policy"])
    cohort_ligands = common_cohort(detail)
    common = detail[detail["extractant"].isin(cohort_ligands)]
    per_ligand = common.groupby(["global_model", "arm", "adapter", "policy", "k", "extractant"])[
        "mae"].mean().reset_index()
    table_b = per_ligand.groupby(["global_model", "arm", "adapter", "policy", "k"]).agg(
        mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()
    table_b.to_csv(args.out / "frontier_common.csv", index=False)

    rows = []
    for model in models:
        rule = pipeline_rule(model, args.series_adapter)
        for k in K_VALUES:
            block = table_b[(table_b["global_model"] == model) & (table_b["k"] == k)]
            deployable = block[~block["policy"].str.startswith("ORACLE")]
            best = deployable.sort_values("mae").iloc[0]
            adapter, policy = rule[k]
            fixed = block[(block["adapter"] == adapter) & (block["policy"] == policy)] if k else \
                block[block["arm"] == "ZERO_SHOT"]
            rows.append({"global_model": model, "k": k,
                         "best_deployable_arm": best["arm"], "best_deployable_mae": float(best["mae"]),
                         "pipeline_arm": f"{adapter}@{policy}" if k else "ZERO_SHOT",
                         "pipeline_mae": float(fixed["mae"].iloc[0]) if len(fixed) else np.nan,
                         "gen8_frontier": GEN8_FRONTIER[k], "gen9_frontier": GEN9_FRONTIER[k],
                         "n_ligands": int(best["n_ligands"])})
    frontier = pd.DataFrame(rows)
    frontier.to_csv(args.out / "frontier_best.csv", index=False)

    # --- k-shot paired bootstraps vs the gen9 anchor, same adapter, same policy ---
    chemotype = cohort.drop_duplicates("extractant").set_index("extractant")["tanimoto_cluster"]
    common = common.assign(tanimoto_cluster=common["extractant"].map(chemotype).astype(str),
                           full_arm=common["global_model"] + " | " + common["arm"])
    boots = []
    for k in K_VALUES:
        block = common[common["k"] == k]
        rule = pipeline_rule("x", args.series_adapter)
        adapter, policy = rule[k]
        arm = "ZERO_SHOT" if k == 0 else f"{adapter}@{policy}"
        comparisons = {}
        for model in args.finalists:
            for anchor in ("GEN9_SHAPE_RECOMPOSED", "REC_ecfp_plus_recovered"):
                comparisons[f"{model}_vs_{anchor}@k{k}"] = (f"{anchor} | {arm}", f"{model} | {arm}")
        table = paired_chemotype_bootstrap(block, comparisons, arm_column="full_arm")
        table.insert(0, "k", k)
        boots.append(table)
    pd.concat(boots, ignore_index=True).to_csv(args.out / "kshot_bootstrap.csv", index=False)

    # --- shape metrics, by cohort -----------------------------------------------
    definitions = json.loads((GEN8 / "case_studies" / "dmdphpda_cohorts.json").read_text())
    quarantined = set(str(r) for r in definitions["quarantined"])
    corrected = {str(e["row_id"]): float(e["log_D_corrected"]) for e in definitions["corrected"]}
    classes = pd.read_csv(GEN8 / "case_studies" / "ligand_failure_classes.csv")[
        ["extractant", "failure_class"]].set_index("extractant")["failure_class"]
    row_cohorts = pd.read_csv(GEN10 / "data_ceiling" / "row_cohorts.csv").set_index("row_id")["cohort"]

    shape_rows, curve_tables, strata_rows = [], [], []
    for data_cohort in ("FROZEN", "QUARANTINED", "CORRECTED"):
        work = oof_all.copy()
        if data_cohort == "QUARANTINED":
            work = work[~work["row_id"].astype(str).isin(quarantined)]
        elif data_cohort == "CORRECTED":
            mask = work["row_id"].astype(str).isin(corrected)
            work.loc[mask, "log_D"] = work.loc[mask, "row_id"].astype(str).map(corrected)
        for (model, seed), block in work.groupby(["model", "split_seed"]):
            table = curve_shape_table(block, membership)
            table["model"], table["split_seed"], table["data_cohort"] = model, seed, data_cohort
            curve_tables.append(table)
            summary = summarise_shape(table)
            summary.insert(0, "split_seed", seed)
            summary.insert(0, "model", model)
            summary.insert(0, "data_cohort", data_cohort)
            shape_rows.append(summary)
            steep = slope_distribution(table)
            # strata
            err = (block["log_D"] - block["prediction"]).abs()
            strata = {"ALL": np.ones(len(block), dtype=bool)}
            klass = block["extractant"].map(classes)
            for name in ("RESIDUAL_SHAPE", "PURE_LEVEL", "ALREADY_GOOD", "PARTIAL_LEVEL"):
                strata[name] = (klass == name).to_numpy()
            rc = block["row_id"].astype(str).map(row_cohorts)
            strata["MISMATCH_ROWS"] = (rc == "B_MISMATCH").to_numpy()
            strata["CLEAN_ROWS"] = (rc == "A_CONSISTENT").to_numpy()
            for name, mask in strata.items():
                if mask.sum() == 0:
                    continue
                sub = block[mask]
                strata_rows.append({
                    "data_cohort": data_cohort, "model": model, "split_seed": seed,
                    "stratum": name, "n_rows": int(mask.sum()),
                    "n_ligands": int(sub["extractant"].nunique()),
                    "macro_mae": macro_mae(sub), "pooled_mae": float(err[mask].mean()),
                    "ligand_macro_mae": float(err[mask].groupby(sub["extractant"]).mean().mean())})
    shape = pd.concat(shape_rows, ignore_index=True)
    shape_mean = shape.groupby(["data_cohort", "model", "axis_label"]).mean(numeric_only=True).reset_index()
    shape_mean.to_csv(args.out / "shape_by_axis.csv", index=False)
    curves = pd.concat(curve_tables, ignore_index=True)
    curves.to_parquet(args.out / "curve_shape.parquet", index=False)
    strata = pd.DataFrame(strata_rows)
    strata_mean = strata.groupby(["data_cohort", "model", "stratum"]).agg(
        macro_mae=("macro_mae", "mean"), macro_sd=("macro_mae", "std"),
        pooled_mae=("pooled_mae", "mean"), ligand_macro_mae=("ligand_macro_mae", "mean"),
        n_rows=("n_rows", "first"), n_ligands=("n_ligands", "first")).reset_index()
    strata_mean.to_csv(args.out / "strata.csv", index=False)

    # --- shape bootstraps (FROZEN, extractant / acid / metal), vs anchors --------
    frozen_curves = curves[curves["data_cohort"] == "FROZEN"]
    shape_boots = []
    for axis in ("extractant", "acid", "metal_series"):
        block = frozen_curves[frozen_curves["axis_label"] == axis].copy()
        block["span_recovery_distance"] = (block["span_recovery"] - 1.0).abs()
        for metric, sign in (("shape_mae", -1), ("slope_abs_error", -1),
                             ("span_recovery_distance", -1), ("row_mae", -1), ("spearman", 1)):
            long = block[["model", "curve_id", "split_seed", "extractant", "tanimoto_cluster",
                          metric]].rename(columns={metric: "mae"})
            long["mae"] = sign * long["mae"] * -1  # lower-is-better orientation for the bootstrap
            comparisons = {}
            for model in args.finalists:
                for anchor in ("GEN9_SHAPE_RECOMPOSED", "REC_ecfp_plus_recovered", "GEN9_A0_ROW_ONLY"):
                    comparisons[f"{model}_vs_{anchor}"] = (anchor, model)
            table = paired_chemotype_bootstrap(long, comparisons, arm_column="model",
                                               unit_column="curve_id")
            table.insert(0, "metric", metric)
            table.insert(0, "axis", axis)
            shape_boots.append(table)
    pd.concat(shape_boots, ignore_index=True).to_csv(args.out / "shape_bootstrap.csv", index=False)

    # --- macro bootstraps vs anchors, per data cohort ----------------------------
    macro_boots = []
    for data_cohort in ("FROZEN", "QUARANTINED", "CORRECTED"):
        work = oof_all.copy()
        if data_cohort == "QUARANTINED":
            work = work[~work["row_id"].astype(str).isin(quarantined)]
        elif data_cohort == "CORRECTED":
            mask = work["row_id"].astype(str).isin(corrected)
            work.loc[mask, "log_D"] = work.loc[mask, "row_id"].astype(str).map(corrected)
        work["mae"] = (work["log_D"] - work["prediction"]).abs()
        per = work.groupby(["model", "split_seed", "extractant", "tanimoto_cluster"])["mae"].mean().reset_index()
        comparisons = {}
        for model in args.finalists:
            for anchor in ("GEN9_SHAPE_RECOMPOSED", "REC_ecfp_plus_recovered"):
                comparisons[f"{model}_vs_{anchor}"] = (anchor, model)
        table = paired_chemotype_bootstrap(per, comparisons, arm_column="model")
        table.insert(0, "data_cohort", data_cohort)
        macro_boots.append(table)
    pd.concat(macro_boots, ignore_index=True).to_csv(args.out / "macro_bootstrap.csv", index=False)

    # --- the frozen pipeline ------------------------------------------------------
    selected = args.selected or args.finalists[0]
    rule = pipeline_rule(selected, args.series_adapter)
    pipeline = {int(k): float(frontier[(frontier["global_model"] == selected)
                                       & (frontier["k"] == k)]["pipeline_mae"].iloc[0])
                for k in K_VALUES}
    config = {
        "selected_global_model": selected,
        "rules": {str(k): {"adapter": a, "policy": p} for k, (a, p) in rule.items()},
        "first_point": "CENTRAL_THEN_SPREAD (medoid of standardised condition axes, then spread)",
        "series_adapter": args.series_adapter or "none (OFFSET_K3 via SLOPE_L_s1_K3)",
        "common_cohort_size": len(cohort_ligands),
        "frontier_common_cohort": pipeline,
        "gen8_frontier": GEN8_FRONTIER, "gen9_frontier": GEN9_FRONTIER,
        "finalists": args.finalists, "anchors": list(ANCHORS),
        "seeds": list(SEEDS), "repeats": args.repeats, "policies": list(POLICIES),
    }
    (args.out / "pipeline_frontier.json").write_text(json.dumps(config, indent=2))
    print("\n=== common-cohort frontier (pipeline rule) ===")
    print(frontier.pivot(index="k", columns="global_model", values="pipeline_mae").round(4).to_string())
    print("\n=== best deployable arm per k ===")
    print(frontier[["global_model", "k", "best_deployable_arm", "best_deployable_mae"]]
          .pivot(index="k", columns="global_model", values="best_deployable_mae").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
