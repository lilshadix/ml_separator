"""gen5: predict the extraction level ``log D`` (not just the separation factor).

The pair model answers *"which of two lanthanides does this ligand prefer?"*.
This one answers *"how much of the metal is extracted at these conditions?"* —
the quantity that decides whether a ligand is usable at all, and which the pair
target cancels out by construction.

Three regimes, because "does the model work" has three different answers:

* ``unseen_ligand`` — folds grouped on the **ECFP cluster**, not the extractant.
  The repo audit marks bit-identical ECFP homologs across a fold boundary as
  CRITICAL; on this cohort 92 extractants collapse to 75 clusters.
* ``unseen_conditions`` — folds grouped on ``condition_id``, so the ligand is
  known and whole experiments are held out. This is the practically common case
  (you have the ligand, you are choosing acidity).
* ``kshot`` — post-hoc on the ``unseen_ligand`` out-of-fold predictions: if k
  measurements of the new ligand exist, recalibrate on them. Reuses the gen4
  k-shot calibrators.

Every regime carries the controls the pair studies learned to demand: per-block
shuffle twins, model-free nulls, macro-by-cluster as the decision metric (pooled
is reported but is dominated by the largest ligand), and a cluster bootstrap.

Example::

    .venv/bin/python scripts/run_gen5_levels.py --quick
    .venv/bin/python scripts/run_gen5_levels.py --split-seeds 104729 130363 155921 196613 262147
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.kshot_calibration import (  # noqa: E402
    apply_calibrator, fit_calibrator,
)
from lanthanide_separation.levels import (  # noqa: E402
    LEARNERS, LEVEL_ARMS, LEVEL_BASELINE_ARM, LEVEL_DEFAULT_ARMS, LEVEL_TARGET_COLUMN,
    LevelForestParameters, LevelRegressor, build_level_dataset, level_metric_table,
    paired_group_bootstrap, replicate_noise_floor, shuffle_block,
)

DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
DESCRIPTOR_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_2d_descriptors.parquet"
#: Arms that also get the alternative learners (hgb, ridge) — the ones a reader
#: will actually compare; running every arm x every learner is not informative.
LEARNER_VARIANT_ARMS: tuple[str, ...] = ("MC", "MC_ecfp", "MC_lig2d_ext", "MC_all2d")
DEFAULT_SEEDS = [104729, 130363, 155921, 196613, 262147]
#: Regimes, from most to least demanding.
#:   unseen_chemotype  — hold out Tanimoto≥0.7 super-clusters (strict "new chemistry")
#:   unseen_ligand     — hold out bit-identical ECFP clusters (close homologues may remain)
#:   unseen_series     — known ligand, hold out a whole measurement series (acid/diluent setting)
#:   unseen_conditions — known ligand, hold out single condition vectors (titration interpolation)
REGIMES = ("unseen_chemotype", "unseen_ligand", "unseen_series", "unseen_conditions")
REGIME_GROUP = {"unseen_chemotype": "tanimoto_cluster", "unseen_ligand": "ecfp_cluster",
                "unseen_series": "series_id", "unseen_conditions": "condition_id"}
LIGAND_REGIMES = ("unseen_chemotype", "unseen_ligand")
#: Blocks whose shuffled twin is worth running: if an arm keeps its advantage
#: after its own block is permuted, the block was never carrying signal.
SHUFFLE_ARMS = {"MC_ecfp_SHUF": ("MC_ecfp", "ECFP"),
                "MC_lig2d_ext_SHUF": ("MC_lig2d_ext", "LIG2D_EXT"),
                "MC_donors_SHUF": ("MC_donors", "DONORS"),
                "MC_cond_SHUF": ("MC", "COND")}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--descriptors", type=Path, default=DESCRIPTOR_PATH,
                   help="gen4 extended-2D descriptor parquet (LIG2D_EXT block); pass '' to skip")
    p.add_argument("--arms", nargs="+", default=list(LEVEL_DEFAULT_ARMS), choices=list(LEVEL_ARMS))
    p.add_argument("--learners", nargs="+", default=["extratrees", "hgb", "ridge"], choices=list(LEARNERS),
                   help="extra learners run on LEARNER_VARIANT_ARMS (extratrees is always the main learner)")
    p.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    p.add_argument("--split-seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-features", type=float, default=0.30)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--min-rows-per-extractant", type=int, default=10)
    p.add_argument("--replicate-policy", default="mean", choices=["mean", "unique", "all"])
    p.add_argument("--log-d-floor", type=float, default=-6.0)
    p.add_argument("--no-shuffle-controls", action="store_true")
    p.add_argument("--kshot", nargs="*", type=int, default=[1, 2, 3, 5],
                   help="k values for the post-hoc calibration on unseen_ligand OOF")
    p.add_argument("--kshot-draws", type=int, default=20)
    p.add_argument("--replicates", type=int, default=5000)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--quick", action="store_true", help="1 seed, 3 folds, 80 trees, a small arm subset")
    return p.parse_args(argv)


def _fold_groups(frame: pd.DataFrame, regime: str) -> np.ndarray:
    if regime not in REGIME_GROUP:
        raise ValueError(f"unknown regime {regime!r}")
    return frame[REGIME_GROUP[regime]].astype(str).to_numpy()


def _seeded_group_kfold(groups: np.ndarray, n_splits: int, seed: int):
    """Randomised grouped K-fold: shuffle the groups, deal them round-robin.

    ``sklearn.GroupKFold`` is greedy-by-size and therefore nearly deterministic
    (82 % of rows kept the same fold across five "seeds"), so seed-to-seed spread
    was not split variance.  Round-robin dealing of a shuffled group order gives
    genuinely different partitions; the largest group can still only sit in one
    fold, so that fold is bigger — accepted and disclosed.
    """
    names = np.unique(groups)
    rng = np.random.default_rng(seed)
    dealt = rng.permutation(names)
    fold_of = {g: i % n_splits for i, g in enumerate(dealt)}
    labels = np.array([fold_of[g] for g in groups])
    for k in range(n_splits):
        te = np.flatnonzero(labels == k)
        tr = np.flatnonzero(labels != k)
        if te.size == 0:
            continue
        yield tr, te



def _nearest_condition_copy(train, test, cond_cols, *, fallback, global_mean: float) -> np.ndarray:
    """Model-free null for known-ligand regimes: copy log D from the training row of
    the same (extractant, metal) whose condition vector is nearest (standardised
    L1 over the numeric condition columns).  Falls back to the extractant mean,
    then the global mean, when the ligand+metal is unseen."""
    cols = list(cond_cols)
    tr_x = train[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    te_x = test[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    scale = np.nanstd(tr_x, axis=0); scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    tr_x = np.nan_to_num(tr_x / scale, nan=0.0); te_x = np.nan_to_num(te_x / scale, nan=0.0)
    tr_key = (train["extractant"].astype(str) + "|" + train["metal_symbol"].astype(str)).to_numpy()
    te_key = (test["extractant"].astype(str) + "|" + test["metal_symbol"].astype(str)).to_numpy()
    y_tr = train[LEVEL_TARGET_COLUMN].to_numpy(float)
    idx_by_key: dict[str, np.ndarray] = {}
    for i, k in enumerate(tr_key):
        idx_by_key.setdefault(k, []).append(i)
    out = np.empty(len(test))
    ext_test = test["extractant"].to_numpy()
    for j, k in enumerate(te_key):
        cand = idx_by_key.get(k)
        if cand:
            cand = np.asarray(cand)
            d = np.abs(tr_x[cand] - te_x[j]).sum(axis=1)
            out[j] = y_tr[cand[int(np.argmin(d))]]
        else:
            out[j] = fallback.get(ext_test[j], global_mean)
    return out



def _nearest_train_tanimoto(train, test, ecfp_cols) -> np.ndarray:
    """Per test row: max Tanimoto between its ligand and any training ligand."""
    cols = list(ecfp_cols)
    tr = train.drop_duplicates("extractant")[["extractant"] + cols]
    te = test.drop_duplicates("extractant")[["extractant"] + cols]
    a = te[cols].to_numpy().astype(int); b = tr[cols].to_numpy().astype(int)
    inter = a @ b.T
    ca, cb = a.sum(1), b.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = np.where((ca[:, None] + cb[None, :] - inter) > 0, inter / (ca[:, None] + cb[None, :] - inter), 0.0)
    best = dict(zip(te["extractant"], tan.max(axis=1) if tan.size else np.zeros(len(te))))
    return test["extractant"].map(best).to_numpy(float)


def learner_variant_names(arms: list[str], learners: list[str]) -> dict[str, tuple[str, str]]:
    """``{"MC_ecfp@hgb": ("MC_ecfp", "hgb"), ...}`` for the non-default learners."""
    out: dict[str, tuple[str, str]] = {}
    for arm in LEARNER_VARIANT_ARMS:
        if arm not in arms:
            continue
        for learner in learners:
            if learner != "extratrees":
                out[f"{arm}@{learner}"] = (arm, learner)
    return out


def evaluate_regime(
    data, frame: pd.DataFrame, *, regime: str, arms: list[str], seeds: list[int], folds: int,
    params: LevelForestParameters, shuffle_controls: bool, learners: list[str], log,
) -> pd.DataFrame:
    """Out-of-fold predictions for every (arm, seed) under one regime."""
    groups = _fold_groups(frame, regime)
    variants = learner_variant_names(arms, learners)
    out = []
    for seed in seeds:
        oof = frame[list(("row_id", "extractant", "ecfp_cluster", "condition_id", "metal_symbol",
                          "metal_Z", "n_replicates", LEVEL_TARGET_COLUMN))].copy()
        oof["split_seed"] = seed
        oof["outer_fold"] = -1
        for arm in arms:
            oof[f"prediction_{arm}"] = np.nan
        for name in variants:
            oof[f"prediction_{name}"] = np.nan
        if shuffle_controls:
            for name in SHUFFLE_ARMS:
                oof[f"prediction_{name}"] = np.nan
        oof["prediction_NULL_metal_mean"] = np.nan
        oof["prediction_NULL_global_mean"] = np.nan
        oof["prediction_NULL_extractant_mean"] = np.nan
        oof["prediction_NULL_nearest_condition"] = np.nan
        oof["nn_train_tanimoto"] = np.nan

        for fold, (tr, te) in enumerate(_seeded_group_kfold(groups, folds, seed)):
            train, test = frame.iloc[tr], frame.iloc[te]
            oof.iloc[te, oof.columns.get_loc("outer_fold")] = fold
            y_tr = train[LEVEL_TARGET_COLUMN].to_numpy(float)
            # model-free nulls, fitted on training rows only
            oof.iloc[te, oof.columns.get_loc("prediction_NULL_global_mean")] = float(y_tr.mean())
            metal_mean = train.groupby("metal_symbol")[LEVEL_TARGET_COLUMN].mean()
            oof.iloc[te, oof.columns.get_loc("prediction_NULL_metal_mean")] = (
                test["metal_symbol"].map(metal_mean).fillna(float(y_tr.mean())).to_numpy())
            # For known-ligand regimes the honest nulls are per-ligand: its training
            # mean, and the nearest training condition of the same ligand+metal.
            ext_mean = train.groupby("extractant")[LEVEL_TARGET_COLUMN].mean()
            oof.iloc[te, oof.columns.get_loc("prediction_NULL_extractant_mean")] = (
                test["extractant"].map(ext_mean).fillna(float(y_tr.mean())).to_numpy())
            oof.iloc[te, oof.columns.get_loc("prediction_NULL_nearest_condition")] = (
                _nearest_condition_copy(train, test, data.blocks["COND"], fallback=ext_mean, global_mean=float(y_tr.mean())))

            fold_params = LevelForestParameters(
                n_estimators=params.n_estimators, max_features=params.max_features,
                min_samples_leaf=params.min_samples_leaf,
                random_state=params.random_state + fold * 1009 + 9_999_991, n_jobs=params.n_jobs)
            for arm in arms:
                cols = data.arm_columns(arm)
                model = LevelRegressor(cols, fold_params).fit(train, y_tr, groups=train["ecfp_cluster"])
                oof.iloc[te, oof.columns.get_loc(f"prediction_{arm}")] = model.predict(test)
            for name, (arm, learner) in variants.items():
                vp = LevelForestParameters(**{**fold_params.__dict__, "learner": learner})
                model = LevelRegressor(data.arm_columns(arm), vp).fit(train, y_tr, groups=train["ecfp_cluster"])
                oof.iloc[te, oof.columns.get_loc(f"prediction_{name}")] = model.predict(test)
            if shuffle_controls:
                for name, (base_arm, block) in SHUFFLE_ARMS.items():
                    if base_arm not in arms:
                        continue
                    rng = np.random.default_rng(seed + fold * 7919 + 424_242)
                    block_cols = data.blocks[block]
                    train_s = shuffle_block(train, block_cols, rng=rng)
                    model = LevelRegressor(data.arm_columns(base_arm), fold_params).fit(
                        train_s, y_tr, groups=train["ecfp_cluster"])
                    oof.iloc[te, oof.columns.get_loc(f"prediction_{name}")] = model.predict(test)
            if regime in LIGAND_REGIMES:
                # how close is each held-out ligand to its nearest TRAINING ligand?
                oof.iloc[te, oof.columns.get_loc("nn_train_tanimoto")] = _nearest_train_tanimoto(
                    train, test, data.blocks["ECFP"])
            log(f"  {regime} seed {seed} fold {fold}: train {len(train)} test {len(test)}")
        oof["regime"] = regime
        out.append(oof)
    return pd.concat(out, ignore_index=True)


def kshot_on_levels(oof: pd.DataFrame, arms: list[str], ks: list[int], *, draws: int, seed: int) -> pd.DataFrame:
    """Post-hoc k-shot calibration of level predictions, per extractant.

    Levels carry no exact within-cell additivity, so the pair study's
    "determined query row" trap does not arise.  The honest transfer setting
    is still enforced: ``mae_free`` scores only query rows whose condition is
    not represented in the support.  Only ``offset`` / ``scale`` / ``affine``
    are meaningful for a level — ``trend`` (a ΔZ slope) is a pair-model notion.
    """
    forms = ("offset", "scale", "affine")
    ext_names = sorted(oof["extractant"].unique())
    ext_code = {e: i for i, e in enumerate(ext_names)}   # process-independent, unlike hash()
    rows: list[dict] = []
    k_max = max(ks)
    for regime in sorted(oof.regime.unique()):
        for seed_val in sorted(int(s) for s in oof.split_seed.unique()):
            sub = oof[(oof.regime == regime) & (oof.split_seed == seed_val)]
            for arm in arms:
                for ext, g in sub.groupby("extractant"):
                    p = g[f"prediction_{arm}"].to_numpy(float)
                    y = g[LEVEL_TARGET_COLUMN].to_numpy(float)
                    cond = g["condition_id"].to_numpy()
                    n = len(g)
                    if n < k_max + 10:
                        continue
                    for draw in range(draws):
                        rng = np.random.default_rng([seed, seed_val, ext_code[ext], draw])
                        order = rng.permutation(n)
                        pool, query = order[:k_max], order[k_max:]
                        free = ~np.isin(cond[query], cond[pool])
                        base = {"regime": regime, "split_seed": seed_val, "arm": arm, "extractant": ext,
                                "ecfp_cluster": g["ecfp_cluster"].iloc[0], "draw": draw,
                                "n_query": int(len(query)), "n_free": int(free.sum())}
                        err0 = np.abs(p[query] - y[query])
                        rows.append({**base, "form": "none", "k": 0, "mae": float(err0.mean()),
                                     "mae_free": float(err0[free].mean()) if free.any() else np.nan,
                                     "fit_rejected": 0})
                        # Model-free null: predict every query row as the mean of the k
                        # measured values (uses the measurements, ignores the model).
                        for k in ks:
                            m_k = float(y[pool[:k]].mean())
                            errn = np.abs(m_k - y[query])
                            rows.append({**base, "form": "NULL_kmean", "k": int(k), "mae": float(errn.mean()),
                                         "mae_free": float(errn[free].mean()) if free.any() else np.nan,
                                         "fit_rejected": 0})
                        for form in forms:
                            for k in ks:
                                a, b, rej = fit_calibrator(form, p[pool[:k]], y[pool[:k]], None, 1.0)
                                pred = apply_calibrator(form, a, b, p[query], None)
                                err = np.abs(pred - y[query])
                                rows.append({**base, "form": form, "k": int(k), "mae": float(err.mean()),
                                             "mae_free": float(err[free].mean()) if free.any() else np.nan,
                                             "fit_rejected": int(rej)})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / "runs" / f"gen5_levels_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    arms = list(args.arms)
    seeds = list(args.split_seeds)
    folds = args.folds
    n_estimators = args.n_estimators
    shuffle_controls = not args.no_shuffle_controls
    learners = list(args.learners)
    if args.quick:
        arms = [a for a in ("A_metal", "A_cond", "A_ecfp", "MC", "MC_ecfp", "MC_lig2d_ext")
                if a in arms]
        seeds, folds, n_estimators = seeds[:1], 3, 80
        learners = [l for l in learners if l in ("extratrees", "ridge")]
        args.kshot = [1, 3]
        args.kshot_draws = 5

    source = pd.read_parquet(args.dataset)
    descriptors = None
    if args.descriptors and str(args.descriptors) and Path(args.descriptors).exists():
        descriptors = pd.read_parquet(args.descriptors)
    elif args.descriptors and str(args.descriptors):
        log(f"WARNING: descriptor parquet not found at {args.descriptors}; LIG2D_EXT arms will be skipped")
    data = build_level_dataset(
        source, min_rows_per_extractant=args.min_rows_per_extractant,
        replicate_policy=args.replicate_policy, drop_below_log_d=args.log_d_floor,
        ligand_descriptors=descriptors)
    if "LIG2D_EXT" not in data.blocks:
        arms = [a for a in arms if "LIG2D_EXT" not in LEVEL_ARMS[a]]
    frame = data.frame
    noise = replicate_noise_floor(source)
    log(f"cohort {data.audit['rows']} rows, {data.audit['extractants']} extractants, "
        f"{data.audit['ecfp_clusters']} ECFP clusters, {data.audit['conditions']} conditions; "
        f"target sd {data.audit['target_sd']:.3f}")
    log(f"blocks {data.audit['block_sizes']}; arms {arms}; seeds {seeds}; folds {folds}; trees {n_estimators}")
    log(f"replicate noise floor: {noise}")

    params = LevelForestParameters(
        n_estimators=n_estimators, max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf, random_state=args.model_seed, n_jobs=args.n_jobs)

    t0 = time.time()
    oof_parts = []
    for regime in args.regimes:
        log(f"regime {regime}")
        oof_parts.append(evaluate_regime(
            data, frame, regime=regime, arms=arms, seeds=seeds, folds=folds,
            params=params, shuffle_controls=shuffle_controls, learners=learners, log=log))
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    log(f"fitting done in {time.time() - t0:.1f}s")

    scored_arms = arms + list(learner_variant_names(arms, learners)) \
        + ([n for n, (b, _) in SHUFFLE_ARMS.items() if b in arms] if shuffle_controls else []) \
        + ["NULL_metal_mean", "NULL_global_mean", "NULL_extractant_mean", "NULL_nearest_condition"]
    overall_parts, group_parts, boot_parts = [], [], []
    for (regime, seed), block in oof.groupby(["regime", "split_seed"]):
        over, per_group = level_metric_table(block, scored_arms, baseline_arm=LEVEL_BASELINE_ARM)
        over.insert(0, "split_seed", seed); over.insert(0, "regime", regime)
        per_group.insert(0, "split_seed", seed); per_group.insert(0, "regime", regime)
        overall_parts.append(over); group_parts.append(per_group)
    for regime, block in oof.groupby("regime"):
        comparisons = {f"{LEVEL_BASELINE_ARM}_vs_{a}": (LEVEL_BASELINE_ARM, a)
                       for a in scored_arms if a != LEVEL_BASELINE_ARM}
        b = paired_group_bootstrap(block, comparisons, replicates=args.replicates)
        b.insert(0, "regime", regime)
        boot_parts.append(b)
    overall = pd.concat(overall_parts, ignore_index=True)
    per_group = pd.concat(group_parts, ignore_index=True)
    boot = pd.concat(boot_parts, ignore_index=True)
    overall.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    per_group.to_csv(output_dir / "per_cluster_metrics.csv", index=False)
    boot.to_csv(output_dir / "paired_bootstrap.csv", index=False)

    leaderboard = (overall.groupby(["regime", "arm"], sort=False)
                   .agg(macro_mae=("macro_mae", "mean"), macro_mae_sd=("macro_mae", "std"),
                        pooled_mae=("pooled_mae", "mean"), pooled_r2=("pooled_r2", "mean"),
                        within_lig_r2=("within_ligand_r2", "mean"),
                        dispersion=("prediction_dispersion_ratio", "mean"),
                        within_0_5=("frac_within_0_5_log", "mean"), within_1_0=("frac_within_1_log", "mean"),
                        n_rows=("n_rows", "first"), n_groups=("n_groups", "first"))
                   .reset_index().sort_values(["regime", "macro_mae"]))
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)

    kshot = pd.DataFrame()
    if args.kshot:
        log("k-shot calibration on the unseen_ligand OOF")
        ul = oof[oof.regime == "unseen_ligand"]
        if not ul.empty:
            kshot = kshot_on_levels(ul, [a for a in arms if a == LEVEL_BASELINE_ARM] or arms[:1],
                                    list(args.kshot), draws=args.kshot_draws, seed=20260817)
            if not kshot.empty:
                kshot.to_csv(output_dir / "kshot_levels.csv", index=False)

    lines: list[str] = []
    A = lines.append
    A(f"gen5 level (log D) study — {stamp}")
    A(f"cohort: {data.audit['rows']} rows, {data.audit['extractants']} extractants, "
      f"{data.audit['ecfp_clusters']} ECFP clusters, {data.audit['conditions']} conditions, "
      f"{data.audit['metals']} metals; target sd {data.audit['target_sd']:.3f}")
    A(f"filters: min_rows_per_extractant={args.min_rows_per_extractant}, "
      f"replicate_policy={args.replicate_policy}, log_D floor {args.log_d_floor}")
    A(f"seeds {seeds}; {folds} folds; {n_estimators} trees; arms {arms}")
    A(f"replicate reproducibility over {noise.get('replicated_cells', 0)} replicated cells: median within-cell sd "
      f"{noise.get('median_within_cell_sd', float('nan')):.3f} (pooled {noise.get('pooled_within_cell_sd', float('nan')):.3f}; "
      f"{noise.get('cells_with_sd_over_1', 0)} cells with sd > 1 are probably not true repeats). "
      f"Implied MAE floor vs a single new measurement ≈ {noise.get('mae_floor_median_sd', float('nan')):.2f} (median) / "
      f"{noise.get('mae_floor_pooled_sd', float('nan')):.2f} (pooled).")
    A("")
    A("READING THIS REPORT")
    A("  * macro_mae weights each ECFP cluster equally (one ligand = one vote); pooled_mae is")
    A("    row-weighted and is dominated by the largest ligand. They can disagree — the macro is primary.")
    A("  * folds are grouped on ECFP CLUSTER, not extractant: 92 extractants collapse to 75 clusters,")
    A("    and grouping on the extractant would put bit-identical fingerprints on both sides of a fold.")
    A("  * *_SHUF arms are the same arm with its own feature block permuted. An arm that does not beat")
    A("    its shuffled twin is not using that block.")
    A("  * NULL_metal_mean / NULL_global_mean use no ligand information at all.")
    A("")
    A("LEADERBOARD (mean over seeds; lower macro_mae is better)")
    A(leaderboard.round(4).to_string(index=False))
    A("")
    # Per property family: what it carries alone, and what it adds once metal + conditions are known.
    A("PROPERTY FAMILIES — alone (A_*) and added to metal+conditions (MC_*), macro MAE by regime")
    fam_rows = []
    families = [("physchem", "A_physchem", "MC_physchem"), ("ecfp", "A_ecfp", "MC_ecfp"),
                ("lig2d_ext", "A_lig2d_ext", "MC_lig2d_ext"), ("donors", "A_donors", "MC_donors"),
                ("complex_phys", "A_complex_phys", "MC_complex_phys"), ("polyhedron", "A_polyhedron", "MC_polyhedron"),
                ("cond (vs metal-only)", "A_cond", "MC"), ("all2d", None, "MC_all2d"), ("all3d", None, "MC_all3d"),
                ("everything", None, "MC_everything")]
    lb = leaderboard.set_index(["regime", "arm"])["macro_mae"]
    for regime in leaderboard.regime.unique():
        mc = lb.get((regime, "MC"), np.nan)
        for fam, alone, on_mc in families:
            fam_rows.append({
                "regime": regime, "family": fam,
                "alone": lb.get((regime, alone), np.nan) if alone else np.nan,
                "MC": mc, "MC_plus_family": lb.get((regime, on_mc), np.nan),
                "gain_over_MC": mc - lb.get((regime, on_mc), np.nan),
            })
    fam_table = pd.DataFrame(fam_rows)
    fam_table.to_csv(output_dir / "property_families.csv", index=False)
    A(fam_table.round(4).to_string(index=False))
    A("")
    A("PAIRED CLUSTER BOOTSTRAP vs " + LEVEL_BASELINE_ARM + " (positive delta = candidate better)")
    A(boot[["regime", "candidate", "point_delta_mae", "ci95_low", "ci95_high",
            "p_worse_one_sided", "groups_improved", "groups_total"]].round(4).to_string(index=False))
    if not kshot.empty:
        A("")
        A("K-SHOT ON LEVELS (unseen ligand). mae_free = query rows whose condition is NOT in the support")
        A("  (the honest number); mae = all query rows (same-experiment contaminated). NULL_kmean = mean of the")
        A("  k measured values, no model — the model must beat this, not k=0.")
        ks = (kshot.groupby(["arm", "form", "k"]).agg(
            mae_free=("mae_free", "mean"), mae_all=("mae", "mean"),
            n_extractants=("extractant", "nunique"), n_ext_with_free=("mae_free", lambda s: int(s.notna().sum() > 0)),
            reject=("fit_rejected", "mean")).reset_index())
        A(ks.round(4).to_string(index=False))
    report = "\n".join(lines)
    (output_dir / "decision_report.txt").write_text(report + "\n")
    print(report)

    (output_dir / "summary.json").write_text(json.dumps({
        "stamp": stamp, "cohort_audit": data.audit, "noise_floor": noise,
        "arms": arms, "scored_arms": scored_arms, "regimes": args.regimes, "seeds": seeds,
        "folds": folds, "n_estimators": n_estimators, "max_features": args.max_features,
        "min_samples_leaf": args.min_samples_leaf, "baseline_arm": LEVEL_BASELINE_ARM,
        "leaderboard": leaderboard.to_dict("records"), "bootstrap": boot.to_dict("records"),
        "note": "gen5 level (log D) study; exploratory unless run under a frozen protocol",
    }, indent=2, default=float))
    log(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
