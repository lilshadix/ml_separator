#!/usr/bin/env python
"""GEN9-B — can the one-shot oracle's rule be predicted before the measurement?

gen8 left this exactly posed.  Under one-shot offset calibration the best candidate
is ``argmin_i |r_i - median(r)|``; the rule is known, and it is useless, because
``r_i`` is what you learn *by running the experiment*.  gen9 asks whether that
deviation can be predicted from geometry and the model's own predictions.

The script does four things, in this order, because each one is only meaningful if
the one before it held.

**1. Re-derive the identity under gen9's protocol.**  gen8 verified it to 6.7e-16
in its *exhaustive* protocol, where candidates and scored rows are the same points.
gen9 selects from a pool and scores on a disjoint evaluation set, and the proof does
not transfer: minimising a convex function over a discrete set need not pick the
point nearest its unconstrained minimiser once the two sets differ.  So the identity
is measured — exactness rate, regret of following the surrogate, and rank
correlation — and reported before any model is trained on it.  This is why two
labels are carried: the brief's ``oracle_deviation`` and the realised one-shot MAE.

**2. Train fold-locally.**  One model per (split seed, fold), on blocks drawn from
that seed's *other* folds.  The fold plan is chemotype-blocked, so a held-out
ligand's whole chemotype is absent from the model that will choose its first
experiment.

**3. Score against the baselines that matter.**  The comparison is
``LEARNED vs CENTRAL/MEDOID``, not learned vs random — random was beaten in gen8 and
beating it again would say nothing.  Every policy sees the identical pool and is
scored on the identical evaluation rows.

**4. Ablate and stress.**  Feature groups on and off, ligand descriptors real and
*shuffled* (§15: if shuffling changes nothing, the chemistry was not doing the
work), and perturbed candidate pools (§17: a policy that only works when the exact
medoid is in the pool is not a policy).
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

from lanthanide_separation.gen8.inference import paired_chemotype_bootstrap  # noqa: E402
from lanthanide_separation.gen8.kshot import (  # noqa: E402
    POLICIES, POLICY_AXES, PolicyContext, stable_hash,
)
from lanthanide_separation.gen8.protocols import _standardised_axes, make_p2_split  # noqa: E402
from lanthanide_separation.gen9.acquisition import (  # noqa: E402
    FEATURE_GROUPS, FEATURE_PROVENANCE, BlendedAcquisition, PairwiseAcquisition,
    ScalarAcquisition, build_dataset, candidate_features, oracle_labels,
    register_extra_policies, verify_oracle_identity,
)

COHORT_PATH = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
MEMBERSHIP_PATH = REPO_ROOT / "runs" / "gen9_shape" / "curves" / "curve_membership.parquet"
OUT = REPO_ROOT / "runs" / "gen9_shape" / "acquisition"
SEEDS = (104729, 130363, 155921, 196613, 262147)

BASELINES = ("RANDOM", "CENTRAL", "MEDOID", "MEDIAN_PREDICTION", "MID_ACID",
             "FARTHEST_FROM_EXISTING", "MAX_CONDITION_COVERAGE", "MIN_GP_DESIGN_VAR",
             "MIN_PREDICTION", "MAX_PREDICTION", "MAX_ENSEMBLE_SD", "MIN_ENSEMBLE_SD")

#: The ablations of brief §15.  ``shuffled_ligand`` is the control that decides
#: whether a molecular-descriptor gain is chemistry or capacity.
ABLATIONS = ("geometry", "prediction", "curve", "geometry+prediction", "full",
             "full_shuffled")

STRESS = ("full_pool", "half_pool", "no_medoid", "high_acid_only", "low_extractant_only",
          "metal_subset", "sparse_pool")


def attach_conditions(oof: pd.DataFrame, cohort_path: Path = COHORT_PATH) -> pd.DataFrame:
    """Join the condition columns onto the out-of-fold table, and prove they arrived.

    The OOF parquet carries identity, truth and prediction — **not** the conditions.
    ``evaluate_fewshot`` merges the cohort on before it does anything; a script that
    reads the OOF directly must do the same, and the first version of this one did
    not.  The failure was silent and total: ``_standardised_axes`` returns zeros for
    a missing column, so every geometry-driven policy — MEDOID, CENTRAL, MID_ACID,
    FARTHEST, COVERAGE and any learned policy using geometry features — saw an
    all-zero condition space, tied on every candidate, and returned the first index.
    They scored *identically*, which is what made it visible.

    Hence the assertions: a join that silently produced a constant axis would put the
    whole acquisition study back where it started.
    """
    cohort = pd.read_parquet(cohort_path)
    needed = [c for c in POLICY_AXES if c in cohort.columns]
    missing = [c for c in POLICY_AXES if c not in cohort.columns]
    if missing:
        raise SystemExit(f"cohort is missing policy axes {missing}")
    extra = [c for c in ("series_id", "condition_id", "metal_symbol", "lanthanide_index",
                         "cond__acid_concentration_M", "cond__extractant_concentration_M")
             if c in cohort.columns and c not in needed]
    columns = ["row_id"] + needed + extra
    before = len(oof)
    merged = oof.merge(cohort[columns], on="row_id", how="left", validate="many_to_one",
                       suffixes=("", "__cohort"))
    if len(merged) != before:
        raise AssertionError("the cohort join changed the OOF row count")
    for column in needed:
        values = pd.to_numeric(merged[column], errors="coerce")
        if values.notna().sum() == 0:
            raise AssertionError(f"policy axis {column!r} is entirely missing after the join")
        if float(values.std(skipna=True) or 0.0) <= 0:
            raise AssertionError(
                f"policy axis {column!r} is constant after the join — every geometric "
                "policy would degenerate to 'the first candidate'")
    return merged


# --------------------------------------------------------------------------- #
# pool perturbations (brief §17)
# --------------------------------------------------------------------------- #

def perturb_pool(pool: np.ndarray, block: pd.DataFrame, axes: np.ndarray,
                 mode: str, rng: np.random.Generator) -> np.ndarray:
    """A candidate list an experimentalist might actually hand over.

    Every mode is a *restriction* of the original pool, so the evaluation rows are
    untouched and the comparison across modes stays paired on the ligand.
    """
    if mode == "full_pool" or len(pool) < 4:
        return pool
    sub = axes[pool]
    if mode == "half_pool":
        keep = rng.choice(len(pool), size=max(2, len(pool) // 2), replace=False)
        return np.sort(pool[keep])
    if mode == "no_medoid":
        cost = np.abs(sub[:, None, :] - sub[None, :, :]).sum(axis=(1, 2))
        return np.sort(np.delete(pool, int(np.argmin(cost))))
    if mode == "high_acid_only":
        order = np.argsort(sub[:, 0])
        return np.sort(pool[order[len(order) // 2:]])
    if mode == "low_extractant_only":
        order = np.argsort(sub[:, 1])
        return np.sort(pool[order[:max(2, len(order) // 2)]])
    if mode == "metal_subset":
        metals = block["metal_symbol"].astype(str).to_numpy()[pool]
        unique = sorted(set(metals))
        keep = set(unique[: max(1, len(unique) // 2)])
        chosen = np.array([p for p, m in zip(pool, metals) if m in keep])
        return np.sort(chosen) if len(chosen) >= 2 else pool
    if mode == "sparse_pool":
        keep = np.linspace(0, len(pool) - 1, max(3, len(pool) // 3)).astype(int)
        return np.sort(pool[np.unique(keep)])
    raise ValueError(f"unknown pool perturbation {mode!r}")


# --------------------------------------------------------------------------- #
# one-shot evaluation
# --------------------------------------------------------------------------- #

def one_shot_records(block: pd.DataFrame, membership: pd.DataFrame, *, policies: dict,
                     repeats: int, seed: int, pool_modes=("full_pool",),
                     min_rows: int = 4) -> list[dict]:
    """Every policy's k=1 choice and its consequences, on identical rows."""
    n = len(block)
    if n < min_rows:
        return []
    ligand = str(block["extractant"].iloc[0])
    truth = block["log_D"].to_numpy(dtype=float)
    prediction = block["prediction"].to_numpy(dtype=float)
    axes = _standardised_axes(block)
    uncertainty = (block["extra__prediction_sd"].to_numpy(dtype=float)
                   if "extra__prediction_sd" in block.columns else np.full(n, np.nan))
    residual = truth - prediction
    records: list[dict] = []
    for repeat in range(repeats):
        rng = np.random.default_rng((seed, repeat, stable_hash(ligand)))
        split = make_p2_split(n, rng)
        if len(split.pool) < 2 or len(split.evaluation) < 2:
            continue
        zero_shot = float(np.abs(residual[split.evaluation]).mean())
        base = {"extractant": ligand, "tanimoto_cluster": str(block["tanimoto_cluster"].iloc[0]),
                "ecfp_cluster": str(block["ecfp_cluster"].iloc[0]),
                "split_seed": int(block["split_seed"].iloc[0]),
                "fold": int(block["fold"].iloc[0]), "repeat": repeat, "n_rows": n,
                "n_eval": int(len(split.evaluation)), "zero_shot": zero_shot,
                "nn_train_tanimoto": float(block["nn_train_tanimoto"].iloc[0])
                if "nn_train_tanimoto" in block.columns else np.nan}
        for mode in pool_modes:
            pool = perturb_pool(split.pool, block, axes,
                                mode, np.random.default_rng((seed, repeat, 17)))
            if len(pool) < 2:
                continue
            labels = oracle_labels(truth, prediction, pool, split.evaluation)
            realised = labels["realised_mae"].to_numpy()
            order = np.argsort(realised)
            rank_of = {int(c): int(np.flatnonzero(order == i)[0])
                       for i, c in enumerate(labels["candidate"].to_numpy())}
            oracle = float(realised.min())
            worst = float(realised.max())
            surrogate = int(labels["candidate"].iloc[int(labels["oracle_deviation"].to_numpy().argmin())])
            context = PolicyContext(
                block=block, prediction=prediction, pool=np.asarray(pool, dtype=int),
                evaluation=split.evaluation, axes=axes, uncertainty=uncertainty,
                disagreement=np.full(n, np.nan),
                rng=np.random.default_rng((seed, repeat, 7)), truth=None)
            pool_base = {**base, "pool_mode": mode, "n_pool": int(len(pool)),
                         "oracle_mae": oracle, "worst_mae": worst,
                         "identity_gap": float(realised[int(labels["oracle_deviation"]
                                                            .to_numpy().argmin())] - oracle)}
            # The candidate feature matrix depends on (block, pool) and nothing else,
            # so it is built once and shared by every learned model.  Recomputing it
            # per model tripled the study's runtime for byte-identical numbers — and
            # worse, three models that saw different features would not be comparable.
            features = None
            if any(not callable(p) for p in policies.values()):
                features = candidate_features(block, prediction,
                                              np.asarray(pool, dtype=int),
                                              membership=membership)
            for name, policy in policies.items():
                if not callable(policy):
                    scores = np.asarray(policy.score(features), dtype=float)
                    scores = np.where(np.isfinite(scores), scores, np.inf)
                    choice = int(np.asarray(pool, dtype=int)[int(np.argmin(scores))])
                else:
                    context.rng = np.random.default_rng((seed, repeat, stable_hash(name)))
                    choice = int(policy(context, []))
                mae = float(np.abs(residual[split.evaluation] - residual[choice]).mean())
                records.append({
                    **pool_base, "policy": name, "selected": choice,
                    "mae": mae, "regret": mae - oracle,
                    "gap_recovered": ((zero_shot - mae) / (zero_shot - oracle))
                    if zero_shot > oracle + 1e-9 else np.nan,
                    "rank_percentile": rank_of.get(choice, np.nan) / max(len(pool) - 1, 1),
                    "harms": bool(mae > zero_shot),
                    "matches_surrogate": bool(choice == surrogate)})
            # the non-deployable references, on the identical pool
            for name, value, chosen in (("ORACLE", oracle, int(labels["candidate"]
                                                               .iloc[int(realised.argmin())])),
                                        ("SURROGATE_ORACLE",
                                         float(realised[int(labels["oracle_deviation"]
                                                            .to_numpy().argmin())]), surrogate)):
                records.append({
                    **pool_base, "policy": name, "selected": chosen, "mae": value,
                    "regret": value - oracle,
                    "gap_recovered": ((zero_shot - value) / (zero_shot - oracle))
                    if zero_shot > oracle + 1e-9 else np.nan,
                    "rank_percentile": rank_of.get(chosen, np.nan) / max(len(pool) - 1, 1),
                    "harms": bool(value > zero_shot), "matches_surrogate": True})
    return records


# --------------------------------------------------------------------------- #

def training_dataset(oof: pd.DataFrame, membership, *, seed: int, fold: int,
                     repeats: int, cache: dict):
    """The fold's acquisition training blocks, built once and reused by every ablation.

    The feature *matrix* is the same for every feature group — only which columns a
    learner is allowed to read changes — so rebuilding it per ablation was five times
    the work for identical numbers, and the identical numbers matter: two ablations
    that saw different training blocks would not be comparable.
    """
    key = (int(seed), int(fold))
    if key not in cache:
        train_oof = oof[(oof["split_seed"] == seed) & (oof["fold"] != fold)]
        dataset = build_dataset(train_oof, membership=membership, repeats=repeats,
                                seed=20260821)
        if dataset.features.empty:
            raise SystemExit(f"no acquisition training data for seed {seed} fold {fold}")
        cache[key] = dataset
        print(f"  dataset seed {seed} fold {fold}: {dataset.audit['n_candidates']:,} "
              f"candidates over {dataset.audit['n_ligands']} ligands, identity exact "
              f"{dataset.audit['identity_exact_share']:.3f}", flush=True)
    return cache[key]


def fit_fold_models(dataset, *, fold: int, feature_set: str, label: str,
                    shuffle_ligand: bool = False):
    columns = tuple(FEATURE_GROUPS[feature_set])
    features = dataset.features
    if shuffle_ligand:
        # The §15 control: destroy the correspondence between a candidate and its
        # curve context while keeping the marginal distribution intact.  If the
        # shuffled version scores the same, the context was not doing the work.
        rng = np.random.default_rng(9001 + fold)
        features = features.copy()
        order = rng.permutation(len(features))
        for column in FEATURE_GROUPS["curve"]:
            features[column] = features[column].to_numpy()[order]
    rank = PairwiseAcquisition(columns=columns, target=label,
                               random_state=42 + fold).fit(features, dataset.labels)
    scalar = ScalarAcquisition(columns=columns, target=label, n_estimators=300,
                               random_state=42 + fold).fit(features, dataset.labels)
    # alpha on ligands the inner ranker never saw, then applied to the full-data one
    left, right = BlendedAcquisition.ligand_halves(features)
    inner = PairwiseAcquisition(columns=columns, target=label,
                                random_state=42 + fold).fit(features[left],
                                                            dataset.labels[left])
    blend = BlendedAcquisition(inner=inner, columns=columns).fit_alpha(
        features[right], dataset.labels[right])
    blend.inner = rank
    audit = {**dataset.audit, "blend_alpha": blend.alpha, "blend_anchor": blend.anchor,
             "blend_selection": json.dumps(blend.selection.get("means", {}))}
    return {"rank": rank, "scalar": scalar, "blend": blend}, audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path,
                        default=REPO_ROOT / "runs" / "gen7_architecture" / "finalists"
                        / "oof_predictions.parquet")
    parser.add_argument("--model", default="REC_ecfp_plus_recovered")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--train-repeats", type=int, default=8)
    parser.add_argument("--label", default="oracle_deviation",
                        choices=("oracle_deviation", "realised_mae", "regret"))
    parser.add_argument("--ablations", nargs="*", default=list(ABLATIONS))
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--tag", default="primary")
    args = parser.parse_args(argv)

    seeds = SEEDS[:max(1, min(5, args.seeds))]
    args.out.mkdir(parents=True, exist_ok=True)
    register_extra_policies()

    membership = pd.read_parquet(MEMBERSHIP_PATH)
    oof_all = pd.read_parquet(args.oof)
    oof = oof_all[(oof_all["model"] == args.model) & (oof_all["split_seed"].isin(seeds))].copy()
    if oof.empty:
        raise SystemExit(f"no OOF rows for {args.model}")
    oof = attach_conditions(oof)
    print(f"{args.model}: {len(oof):,} OOF rows, {oof['extractant'].nunique()} ligands")

    pd.DataFrame(list(FEATURE_PROVENANCE.values())).to_csv(
        args.out / "feature_provenance.csv", index=False)

    # --- 1. the identity, measured under this protocol --------------------
    print("\n--- oracle identity under the gen9 pool/evaluation split ---")
    identity_rows = []
    for seed in seeds:
        block_seed = oof[oof["split_seed"] == seed]
        for ligand, block in block_seed.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            n = len(block)
            if n < 4:
                continue
            truth = block["log_D"].to_numpy(dtype=float)
            prediction = block["prediction"].to_numpy(dtype=float)
            for repeat in range(args.repeats):
                split = make_p2_split(n, np.random.default_rng(
                    (20260820, repeat, stable_hash(str(ligand)))))
                if len(split.pool) < 2 or len(split.evaluation) < 2:
                    continue
                labels = oracle_labels(truth, prediction, split.pool, split.evaluation)
                identity_rows.append({"split_seed": int(seed), "extractant": str(ligand),
                                      "repeat": repeat, **verify_oracle_identity(labels)})
    identity = pd.DataFrame(identity_rows)
    identity.to_csv(args.out / "oracle_identity.csv", index=False)
    identity_summary = {
        "n_blocks": int(len(identity)),
        "exact_share": float(identity["exact"].mean()),
        "max_gap": float(identity["gap"].max()),
        "median_gap": float(identity["gap"].median()),
        "mean_gap": float(identity["gap"].mean()),
        "q95_gap": float(identity["gap"].quantile(0.95)),
        "median_spearman": float(identity["spearman"].median()),
        "q05_spearman": float(identity["spearman"].quantile(0.05)),
    }
    print(json.dumps(identity_summary, indent=2))

    # --- 2/3. fit, then score ---------------------------------------------
    all_records: list[pd.DataFrame] = []
    audits = []
    pool_modes = list(STRESS) if args.stress else ["full_pool"]
    dataset_cache: dict = {}
    print("\nbuilding the fold-local acquisition training sets")
    for seed in seeds:
        for fold in sorted(oof[oof["split_seed"] == seed]["fold"].unique()):
            training_dataset(oof, membership, seed=int(seed), fold=int(fold),
                             repeats=args.train_repeats, cache=dataset_cache)

    for feature_set in args.ablations:
        started = time.time()
        per_fold: dict = {}
        for seed in seeds:
            for fold in sorted(oof[oof["split_seed"] == seed]["fold"].unique()):
                dataset = dataset_cache[(int(seed), int(fold))]
                models, audit = fit_fold_models(
                    dataset, fold=int(fold), feature_set=feature_set, label=args.label,
                    shuffle_ligand=feature_set.endswith("_shuffled"))
                per_fold[(int(seed), int(fold))] = models
                audits.append({"feature_set": feature_set, "split_seed": int(seed),
                               "fold": int(fold), **audit})
        records = []
        for (seed, fold), block_ff in oof.groupby(["split_seed", "fold"], sort=True):
            models = per_fold[(int(seed), int(fold))]
            columns = tuple(FEATURE_GROUPS[feature_set])
            policies = {}
            if feature_set == args.ablations[0]:
                policies.update({name: POLICIES[name] for name in BASELINES})
            for kind in ("rank", "scalar", "blend"):
                # The fitted model itself, not a policy wrapper: this script only ever
                # asks for the first point, so the wrapper's fold lookup and per-call
                # feature build are pure overhead here.  ``LearnedPolicy`` is still the
                # object that goes into ``evaluate_fewshot``, where sequential picks
                # need it.
                policies[f"LEARNED_{kind.upper()}[{feature_set}]"] = models[kind]
            for ligand, block in block_ff.groupby("extractant", sort=True):
                records += one_shot_records(block.reset_index(drop=True), membership,
                                            policies=policies, repeats=args.repeats,
                                            seed=20260820, pool_modes=pool_modes)
        frame = pd.DataFrame(records)
        frame["feature_set"] = feature_set
        all_records.append(frame)
        print(f"  [{feature_set}] {len(frame):,} records in {time.time() - started:.0f}s",
              flush=True)

    detail = pd.concat(all_records, ignore_index=True)
    detail = detail.drop_duplicates(["split_seed", "fold", "extractant", "repeat",
                                     "pool_mode", "policy"])
    detail.to_parquet(args.out / f"{args.tag}_detail.parquet", index=False)
    pd.DataFrame(audits).to_csv(args.out / f"{args.tag}_training_audit.csv", index=False)

    # --- summary -----------------------------------------------------------
    full = detail[detail["pool_mode"] == "full_pool"]
    per_ligand = full.groupby(["policy", "extractant", "tanimoto_cluster"]).agg(
        mae=("mae", "mean"), regret=("regret", "mean"),
        gap_recovered=("gap_recovered", "mean"),
        rank_percentile=("rank_percentile", "mean"),
        harms=("harms", "mean"), matches_surrogate=("matches_surrogate", "mean"),
        zero_shot=("zero_shot", "mean")).reset_index()
    summary = per_ligand.groupby("policy").agg(
        mae=("mae", "mean"), regret=("regret", "mean"),
        gap_recovered=("gap_recovered", "mean"), rank_percentile=("rank_percentile", "mean"),
        frac_harmed=("harms", "mean"), matches_surrogate=("matches_surrogate", "mean"),
        n_ligands=("extractant", "nunique")).reset_index().sort_values("mae")
    summary.to_csv(args.out / f"{args.tag}_summary.csv", index=False)
    print("\n--- one-shot policy comparison (macro over ligands) ---")
    print(summary.to_string(index=False))

    # --- paired intervals against the baselines that matter ----------------
    per_unit = full.groupby(["policy", "extractant", "tanimoto_cluster", "split_seed"])[
        "mae"].mean().reset_index().rename(columns={"policy": "arm"})
    learned = [p for p in summary["policy"] if p.startswith("LEARNED")]
    comparisons = {}
    for candidate in learned:
        for reference in ("MEDOID", "CENTRAL", "RANDOM", "MIN_GP_DESIGN_VAR"):
            comparisons[f"{candidate} vs {reference}"] = (reference, candidate)
    best = learned[0] if learned else None
    if best:
        comparisons[f"ORACLE vs {best}"] = (best, "ORACLE")
    if comparisons:
        bootstrap = paired_chemotype_bootstrap(per_unit, comparisons,
                                               replicates=args.replicates)
        bootstrap.to_csv(args.out / f"{args.tag}_bootstrap.csv", index=False)
        print("\n--- paired chemotype-blocked intervals (positive = candidate better) ---")
        print(bootstrap[["comparison", "point", "ci95_low", "ci95_high", "n_units",
                         "units_improved", "seeds_positive"]].to_string(index=False))

    if args.stress:
        stress = detail.groupby(["policy", "pool_mode"]).agg(
            mae=("mae", "mean"), regret=("regret", "mean"),
            gap_recovered=("gap_recovered", "mean"), n=("mae", "size")).reset_index()
        stress.to_csv(args.out / f"{args.tag}_stress.csv", index=False)
        print("\n--- candidate-pool stress (mean regret to oracle) ---")
        pivot = stress.pivot(index="policy", columns="pool_mode", values="regret")
        print(pivot.to_string())

    (args.out / f"{args.tag}_run.json").write_text(json.dumps({
        "model": args.model, "seeds": list(seeds), "repeats": args.repeats,
        "train_repeats": args.train_repeats, "label": args.label,
        "ablations": list(args.ablations), "stress": bool(args.stress),
        "identity": identity_summary, "n_records": int(len(detail)),
    }, indent=2, default=float))
    return 0


class _ResolvedFold(dict):
    """A total mapping holding the one model this script's fold loop already picked.

    ``LearnedPolicy`` resolves its model by ``(split_seed, fold)`` because in the
    ``evaluate_fewshot`` path a single policy object serves every fold.  Here the
    fold loop is outside the policy, so the lookup is deliberately total.  It is a
    named type rather than a ``defaultdict`` so a genuine "no model fitted for this
    fold" in the *other* path still raises instead of being silently absorbed.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model

    def get(self, _key, _default=None):
        return self._model


def _make_policy(model, columns, membership):
    from lanthanide_separation.gen9.acquisition import LearnedPolicy

    return LearnedPolicy(models=_ResolvedFold(model), columns=columns,
                         membership=membership, name="LEARNED")


if __name__ == "__main__":
    raise SystemExit(main())
