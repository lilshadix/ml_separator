#!/usr/bin/env python
"""Score any global model — gen8's frozen one or a gen9 shape arm — on the k-shot frontier.

Reads an out-of-fold prediction table and runs gen8's ``evaluate_fewshot`` over it
unchanged: the same pool/evaluation draw, the same policies, the same adapters, the
same ``(seed, repeat, ligand, n_rows)`` pairing key.  That is the whole reason gen9's
arms emit gen7-schema OOF parquets — a gen9 model and gen8's frozen model are then
compared on byte-identical evaluation rows without a line of new evaluation code,
and the paired bootstrap between them is genuinely paired.

The comparison this script exists to make is the one the brief calls mandatory
(§11).  Four things have to be told apart:

1. the old global model;
2. the old global model plus gen8's post-hoc slope repair;
3. the new shape-trained model;
4. the new shape-trained model plus the same post-hoc repair.

If (3) reaches (2) then the repair has become intrinsic.  If (4) still beats (3)
the repair is adding something training did not.  If (3) beats (4) the repair is now
double-counting a correction the model already applies — which would be a result in
itself, and one that only shows up if all four are run.

``--with-series`` adds gen9's hierarchical series-local adapter, ``--with-slope``
gen8's repair, ``--with-learned`` the fold-local learned acquisition policy
(fitted here, on training-fold ligands only).
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
from lanthanide_separation.gen8.kshot import POLICIES  # noqa: E402
from lanthanide_separation.gen8.slope_restore import build_slope_adapters  # noqa: E402
from lanthanide_separation.gen9.acquisition import (  # noqa: E402
    FEATURE_GROUPS, BlendedAcquisition, LearnedPolicy, PairwiseAcquisition,
    ScalarAcquisition, build_dataset, register_extra_policies,
)
from lanthanide_separation.gen9.series_adapter import build_series_adapters  # noqa: E402

COHORT_PATH = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
MEMBERSHIP_PATH = REPO_ROOT / "runs" / "gen9_shape" / "curves" / "curve_membership.parquet"
OUT = REPO_ROOT / "runs" / "gen9_shape" / "kshot"
SEEDS = (104729, 130363, 155921, 196613, 262147)
DEFAULT_POLICIES = ("RANDOM", "CENTRAL", "MEDOID", "MAX_PREDICTIVE_VARIANCE",
                    "CENTRAL_THEN_SPREAD", "D_OPTIMAL", "FARTHEST_FROM_EXISTING")


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


def fit_learned_policies(oof: pd.DataFrame, membership: pd.DataFrame, *,
                         seeds, repeats: int, feature_set: str,
                         kinds=("scalar", "rank", "blend"),
                         label: str = "oracle_deviation") -> dict:
    """One acquisition model per (split seed, fold), trained on that fold's *other* folds.

    A ligand held out in fold ``f`` never contributes a training example to the
    model that will choose its first experiment, because every training block comes
    from a fold other than ``f`` — and the fold plan is chemotype-blocked, so its
    whole chemotype is absent too.
    """
    columns = tuple(FEATURE_GROUPS[feature_set])
    models: dict[str, dict] = {kind: {} for kind in kinds}
    audits = []
    for seed in seeds:
        for fold in sorted(oof[oof["split_seed"] == seed]["fold"].unique()):
            train_oof = oof[(oof["split_seed"] == seed) & (oof["fold"] != fold)]
            dataset = build_dataset(train_oof, membership=membership,
                                    repeats=repeats, seed=20260821)
            if dataset.features.empty:
                raise SystemExit(f"no acquisition training data for seed {seed} fold {fold}")
            audits.append({"split_seed": int(seed), "fold": int(fold),
                           "feature_set": feature_set, **dataset.audit})
            rank = PairwiseAcquisition(columns=columns, target=label,
                                       random_state=42 + int(fold)).fit(
                dataset.features, dataset.labels)
            if "scalar" in kinds:
                models["scalar"][(int(seed), int(fold))] = ScalarAcquisition(
                    columns=columns, target=label,
                    random_state=42 + int(fold)).fit(dataset.features, dataset.labels)
            if "rank" in kinds:
                models["rank"][(int(seed), int(fold))] = rank
            alpha = float("nan")
            if "blend" in kinds:
                # alpha is chosen on ligands the *inner* ranker never saw, then applied
                # to the full-data ranker; see BlendedAcquisition.fit_alpha.
                left, right = BlendedAcquisition.ligand_halves(dataset.features)
                inner = PairwiseAcquisition(columns=columns, target=label,
                                            random_state=42 + int(fold)).fit(
                    dataset.features[left], dataset.labels[left])
                blend = BlendedAcquisition(inner=inner, columns=columns).fit_alpha(
                    dataset.features[right], dataset.labels[right])
                blend.inner = rank
                models["blend"][(int(seed), int(fold))] = blend
                alpha = blend.alpha
                audits[-1]["blend_alpha"] = alpha
                audits[-1]["blend_anchor"] = blend.anchor
            print(f"  acquisition fitted: seed {seed} fold {fold} "
                  f"({dataset.audit['n_candidates']:,} candidates, "
                  f"{dataset.audit['n_ligands']} ligands, identity exact "
                  f"{dataset.audit['identity_exact_share']:.3f}, blend alpha {alpha:.2f})",
                  flush=True)
    return {"models": models, "columns": columns, "audit": pd.DataFrame(audits)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, nargs="+", required=True,
                        help="one or more OOF parquets in gen7 schema")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--policies", nargs="*", default=list(DEFAULT_POLICIES))
    parser.add_argument("--with-slope", action="store_true")
    parser.add_argument("--with-series", action="store_true")
    parser.add_argument("--with-learned", action="store_true")
    parser.add_argument("--learned-features", default="geometry+prediction",
                        choices=sorted(FEATURE_GROUPS))
    parser.add_argument("--acq-repeats", type=int, default=8)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="gen9")
    args = parser.parse_args(argv)

    seeds = SEEDS[:max(1, min(5, args.seeds))]
    args.out.mkdir(parents=True, exist_ok=True)
    register_extra_policies()
    cohort = pd.read_parquet(COHORT_PATH)
    membership = pd.read_parquet(MEMBERSHIP_PATH)
    oof_all = pd.concat([pd.read_parquet(p) for p in args.oof], ignore_index=True)
    oof_all = oof_all.drop_duplicates(["model", "split_seed", "row_id"])
    oof_all = oof_all[oof_all["split_seed"].isin(seeds)]
    models = args.models or sorted(oof_all["model"].unique())
    print(f"models: {models}")

    frames, manifest = [], {"oof": [str(p) for p in args.oof], "models": models,
                            "seeds": list(seeds),
                            "repeats": args.repeats, "policies": list(args.policies),
                            "arms": []}
    for model in models:
        oof = oof_all[oof_all["model"] == model].copy()
        if oof.empty:
            raise SystemExit(f"no rows for model {model}")
        adapters = list(default_adapters())
        if args.with_slope:
            adapters += [a for a in build_slope_adapters(strengths=(1.0,))
                         if a.name in ("SLOPE_L_s1_K1", "SLOPE_L_s1_K3")]
        if args.with_series:
            adapters += build_series_adapters(oof)

        policies = list(args.policies)
        if args.with_learned:
            print(f"fitting learned acquisition for {model} "
                  f"(features={args.learned_features})")
            fitted = fit_learned_policies(oof, membership, seeds=seeds,
                                          repeats=args.acq_repeats,
                                          feature_set=args.learned_features)
            fitted["audit"].to_csv(args.out / f"acquisition_audit_{model}.csv", index=False)
            for kind, tag in (("scalar", "LEARNED_SCALAR"), ("rank", "LEARNED_RANK"),
                              ("blend", "LEARNED_BLEND")):
                POLICIES[tag] = LearnedPolicy(models=fitted["models"][kind],
                                              columns=fitted["columns"], name=tag,
                                              membership=membership)
                policies.append(tag)

        started = time.time()
        detail = evaluate_fewshot(oof, cohort, adapters, policies=policies,
                                  with_oracle_for=("OFFSET_K1",), repeats=args.repeats,
                                  fold_trainer=make_fold_trainer(cohort), verbose=True)
        detail["global_model"] = model
        frames.append(detail)
        manifest["arms"].append({
            "model": model, "adapters": [a.name for a in adapters],
            "policies": policies, "n_records": int(len(detail)),
            "seconds": time.time() - started})
        print(f"[{model}] {len(detail):,} records in {time.time() - started:.0f}s")

    detail = pd.concat(frames, ignore_index=True)
    detail.to_parquet(args.out / f"{args.tag}_detail.parquet", index=False)
    (args.out / f"{args.tag}_run.json").write_text(json.dumps(manifest, indent=2, default=float))

    detail["arm"] = np.where(detail["adapter"] == "ZERO_SHOT_REF", "ZERO_SHOT",
                             detail["adapter"] + "@" + detail["policy"])
    per_ligand = detail.groupby(["global_model", "arm", "k", "extractant"])["mae"].mean().reset_index()
    summary = per_ligand.groupby(["global_model", "arm", "k"]).agg(
        mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()
    summary.to_csv(args.out / f"{args.tag}_summary.csv", index=False)
    for model in models:
        block = summary[(summary["global_model"] == model) & (summary["k"].isin([0, 1, 2, 5]))]
        print(f"\n--- {model} ---")
        print(block.sort_values(["k", "mae"]).groupby("k").head(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
