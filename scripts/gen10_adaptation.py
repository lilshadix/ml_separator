#!/usr/bin/env python
"""PHASE 5 — can a corrected series-local prior beat OFFSET_K3?

gen8's ``evaluate_fewshot`` unchanged — same pool/evaluation draw, same pairing key
``(seed, repeat, ligand, n_rows)`` — over a global model's OOF table, with every
adapter the question needs on the same rows:

* ``OFFSET_K1`` / ``OFFSET_K3``           gen8's references (fixed ridge 4.0)
* ``SERIES_MAP`` / ``SERIES_MAP_NOSERIES`` gen9's hierarchy with the old estimator
* ``SERIES_FIXED``                         gen9's design, lambda 4.0 everywhere
* ``SERIES_INNER``                         penalties selected per fold and per k
* ``SERIES_ML`` / ``SERIES_ML_NOSERIES``   marginal-likelihood variance components
* ``SLOPE_L_s1_K3``                        gen8's slope repair, the frontier's adapter

Scored at k = 0, 1, 2, 3, 5 so the k = 1 identity is *verified* rather than assumed
(every adapter above must agree there to machine precision), and decided at
k = 2, 3, 5.  The pre-registered bar, from the brief: the corrected series-local
model must beat ``OFFSET_K3`` by >= 0.01 macro MAE on at least two of k = 2, 3, 5
with a consistent seed direction.  If not, ``OFFSET_K3`` is retained permanently
and this direction is closed.
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
from lanthanide_separation.gen9.series_adapter import build_series_adapters  # noqa: E402
from lanthanide_separation.gen10.adaptation import build_corrected_adapters  # noqa: E402

COHORT_PATH = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
GEN9 = REPO_ROOT / "runs" / "gen9_shape"
OUT = REPO_ROOT / "runs" / "gen10_final" / "adaptation"
SEEDS = (104729, 130363, 155921, 196613, 262147)
POLICIES = ("CENTRAL_THEN_SPREAD", "MEDOID", "D_OPTIMAL")
#: The gen9 global models the frontier is built on, and where their OOF lives.
DEFAULT_OOF = {
    "GEN9_SHAPE_RECOMPOSED": GEN9 / "recomposed" / "oof_all.parquet",
    "GEN9_REL_MONOLITH": GEN9 / "relmono" / "oof_all.parquet",
}
REFERENCE_ADAPTER = "OFFSET_K3"
CANDIDATES = ("SERIES_FIXED", "SERIES_INNER", "SERIES_ML", "SERIES_ML_NOSERIES",
              "SERIES_MAP", "SERIES_MAP_NOSERIES", "SLOPE_L_s1_K3", "OFFSET_K1")
DECISION_K = (2, 3, 5)
SUCCESS_MARGIN = 0.01


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_OOF))
    parser.add_argument("--oof", nargs="*", default=None,
                        help="MODEL=PATH overrides for OOF parquets")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--policies", nargs="*", default=list(POLICIES))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    seeds = SEEDS[:max(1, min(5, args.seeds))]
    args.out.mkdir(parents=True, exist_ok=True)
    register_extra_policies()
    cohort = pd.read_parquet(COHORT_PATH)
    sources = dict(DEFAULT_OOF)
    for spec in args.oof or []:
        name, path = spec.split("=", 1)
        sources[name] = Path(path)

    frames, priors, manifest = [], [], {"seeds": list(seeds), "repeats": args.repeats,
                                        "policies": list(args.policies), "arms": []}
    for model in args.models:
        oof = pd.read_parquet(sources[model])
        oof = oof[(oof["model"] == model) & (oof["split_seed"].isin(seeds))].copy()
        if oof.empty:
            raise SystemExit(f"no OOF rows for {model} in {sources[model]}")
        adapters = list(default_adapters())
        adapters += [a for a in build_slope_adapters(strengths=(1.0,)) if a.name == "SLOPE_L_s1_K3"]
        adapters += build_series_adapters(oof)
        corrected = build_corrected_adapters(oof)
        adapters += corrected

        started = time.time()
        detail = evaluate_fewshot(oof, cohort, adapters, policies=list(args.policies),
                                  with_oracle_for=(), repeats=args.repeats,
                                  fold_trainer=make_fold_trainer(cohort), verbose=True)
        detail["global_model"] = model
        frames.append(detail)
        for adapter in corrected:
            for (seed, fold), prior in adapter.priors.items():
                record = {"global_model": model, "adapter": adapter.name,
                          "split_seed": seed, "fold": fold, "method": prior.get("method")}
                if "penalty" in prior:
                    record.update({f"penalty_{k}": v for k, v in prior["penalty"].items()})
                if "tau" in prior:
                    record.update({f"tau_{k}": v for k, v in prior["tau"].items()})
                    record["sigma"] = prior.get("sigma")
                    record["converged"] = prior.get("converged")
                    record.update({f"tau_moment_{k}": v
                                   for k, v in prior.get("tau_moment_debiased", {}).items()})
                if "per_k" in prior:
                    for k, choice in prior["per_k"].items():
                        record[f"k{k}_series"] = choice["series"]
                        record[f"k{k}_response"] = choice["response"]
                priors.append(record)
        manifest["arms"].append({"model": model, "adapters": [a.name for a in adapters],
                                 "n_records": int(len(detail)),
                                 "seconds": round(time.time() - started, 1)})
        print(f"[{model}] {len(detail):,} records in {time.time() - started:.0f}s", flush=True)

    detail = pd.concat(frames, ignore_index=True)
    detail.to_parquet(args.out / "detail.parquet", index=False)
    pd.DataFrame(priors).to_csv(args.out / "priors.csv", index=False)

    # --- the k = 1 identity, verified ----------------------------------------
    k1 = detail[(detail["k"] == 1) & detail["adapter"].isin(
        ["OFFSET_K1", "OFFSET_K2", "OFFSET_K3", "SERIES_MAP", "SERIES_FIXED",
         "SERIES_INNER", "SERIES_ML", "SERIES_ML_NOSERIES"])]
    keys = ["global_model", "policy", "split_seed", "fold", "extractant", "repeat"]
    spread = k1.groupby(keys)["mae"].agg(lambda s: float(np.ptp(s.to_numpy())))
    identity = {"max_spread_at_k1": float(spread.max()), "n_units": int(len(spread))}
    print(f"k = 1 identity: max spread across adapters {identity['max_spread_at_k1']:.2e}")

    # --- summary: one ligand, one vote ----------------------------------------
    detail["arm"] = detail["adapter"] + "@" + detail["policy"]
    per_ligand = (detail.groupby(["global_model", "adapter", "policy", "k", "extractant"])["mae"]
                  .mean().reset_index())
    summary = (per_ligand.groupby(["global_model", "adapter", "policy", "k"])
               .agg(mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index())
    summary.to_csv(args.out / "summary.csv", index=False)

    # --- paired, chemotype-blocked bootstrap against OFFSET_K3 ----------------
    chemotype = cohort.drop_duplicates("extractant").set_index("extractant")["tanimoto_cluster"]
    detail["tanimoto_cluster"] = detail["extractant"].map(chemotype).astype(str)
    rows = []
    for (model, policy, k), block in detail[detail["k"].isin(DECISION_K)].groupby(
            ["global_model", "policy", "k"]):
        comparisons = {f"{c}_vs_{REFERENCE_ADAPTER}": (REFERENCE_ADAPTER, c)
                       for c in CANDIDATES if c in set(block["adapter"])}
        table = paired_chemotype_bootstrap(block, comparisons, arm_column="adapter")
        table.insert(0, "k", int(k))
        table.insert(0, "policy", policy)
        table.insert(0, "global_model", model)
        rows.append(table)
    bootstrap = pd.concat(rows, ignore_index=True)
    bootstrap.to_csv(args.out / "bootstrap_vs_offset_k3.csv", index=False)

    # --- the decision rule, applied mechanically ------------------------------
    verdicts = []
    point_col = "point" if "point" in bootstrap.columns else bootstrap.columns[4]
    for (model, policy, candidate), block in bootstrap.groupby(
            ["global_model", "policy", "comparison"]):
        wins = block[block[point_col] >= SUCCESS_MARGIN]
        consistent = block.get("seeds_positive")
        verdicts.append({
            "global_model": model, "policy": policy, "comparison": candidate,
            "k_with_gain_ge_0.01": sorted(int(k) for k in wins["k"]),
            "n_k_with_gain": int(len(wins)),
            "all_seeds_positive_on_those_k": bool(
                (wins["seeds_positive"] == wins["n_seeds"]).all()) if len(wins) and
            "seeds_positive" in wins and "n_seeds" in wins else None,
            "passes": bool(len(wins) >= 2 and
                           ("seeds_positive" not in wins or
                            (wins["seeds_positive"] >= wins["n_seeds"] - 1).all())),
        })
    verdict_table = pd.DataFrame(verdicts)
    verdict_table.to_csv(args.out / "verdict.csv", index=False)
    manifest["k1_identity"] = identity
    manifest["decision_rule"] = {"reference": REFERENCE_ADAPTER, "margin": SUCCESS_MARGIN,
                                 "k": list(DECISION_K), "min_k_passing": 2}
    manifest["any_series_arm_passes"] = bool(verdict_table[
        verdict_table["comparison"].str.startswith("SERIES_")]["passes"].any())
    (args.out / "run.json").write_text(json.dumps(manifest, indent=2, default=str))

    for model in args.models:
        print(f"\n--- {model}: macro MAE by adapter (CENTRAL_THEN_SPREAD) ---")
        block = summary[(summary["global_model"] == model)
                        & (summary["policy"] == "CENTRAL_THEN_SPREAD")]
        print(block.pivot(index="adapter", columns="k", values="mae").round(4).to_string())
    print("\n--- verdicts ---")
    print(verdict_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
