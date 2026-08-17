#!/usr/bin/env python3
"""Evaluate generation-4 candidate estimators on the frozen gen3 outer folds.

This is an *exploratory* harness, not a frozen protocol run: it reuses the
cohort, the A2 column set, the leave-extractants-out outer folds and the
model-seed stream of a completed gen3 primary run so that every candidate is
scored on exactly the rows and folds that produced the champion's out-of-fold
predictions.  The champion is refitted locally with the same hyperparameters
and seeds (``A2_refit``) so that comparisons are apples-to-apples on this
platform; the run's own A2 predictions are carried along as ``A2_run`` for
reference.

Candidates (see ``lanthanide_separation.gen4_candidates``):

* ``A2_refit``      – champion refit (baseline of this harness);
* ``A2_lig2d``      – champion + extended 2D ligand descriptors;
* ``HIER``          – level (extractant x pair) + within-cell deviation;
* ``SCALE``         – learned selectivity scale x pair trend + residual;
* ``*_lig2d``       – the same with the extended descriptors;
* ``A2_ens``        – model-seed ensemble of the champion (variance reference);
* every arm also gets a ``<arm>_TP`` column: the transitive projection of its
  predictions inside each (extractant, condition) cell.

Outputs per split seed ``oof_predictions.csv`` and, aggregated,
``leaderboard.csv`` (macro MAE with per-seed deltas vs ``A2_refit``),
``per_seed_metrics.csv``, ``per_extractant_metrics.csv``,
``paired_bootstrap.csv``, ``pair_level_r2.csv`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lanthanide_separation.evaluation import (  # noqa: E402
    AntisymmetricExtraTreesRegressor,
)
from lanthanide_separation.feature_registry import build_feature_registry  # noqa: E402
from lanthanide_separation.gen3_metrics import (  # noqa: E402
    arm_metric_table,
    equal_group_macro_mae,
    paired_extractant_bootstrap,
)
from lanthanide_separation.gen4_candidates import (  # noqa: E402
    ForestParameters,
    HierarchicalPairRegressor,
    PriorAugmentedPairRegressor,
    ScaleTrendPairRegressor,
    attach_ligand_descriptors,
    transitive_projection,
)
from lanthanide_separation.pairs import (  # noqa: E402
    PAIR_TARGET_COLUMN,
    build_lanthanide_pair_dataset,
)

DATASET_PATH = REPO_ROOT / "dataset with 3D structures/dataset.parquet"
DESCRIPTOR_PATH = REPO_ROOT / "dataset with 3D structures/ligand_2d_descriptors.parquet"
DEFAULT_RUN_DIR = REPO_ROOT / "runs/gen3_primary_20260815T181555Z"
BASELINE = "A2_refit"
RUN_REFERENCE = "A2_run"
PAIRMEAN = "PAIRMEAN_baseline"
ALL_ARMS = (
    "A2_refit",
    "A2_lig2d",
    "HIER",
    "HIER_lig2d",
    "SCALE",
    "SCALE_s50",
    "SCALE_lig2d",
    "SCALE_s50_lig2d",
    "PRIOR",
    "PRIOR_lig2d",
    "PRIOR_TRENDONLY",
    "A2_lig2d_SHUF",
    "A2_ens",
)
IDENTITY_COLUMNS = (
    "pair_id",
    "extractant",
    "extractant_family",
    "condition_id",
    "pair_label",
    "metal_A",
    "metal_B",
    "pair__Z_A",
    "pair__Z_B",
    PAIR_TARGET_COLUMN,
)


def _run_directory_for_seed(run_dir: Path, split_seed: int) -> Path:
    matches = sorted(run_dir.glob(f"run_*_split_{split_seed}_model_*"))
    if not matches:
        raise FileNotFoundError(f"no run directory for split seed {split_seed} under {run_dir}")
    return matches[0]


def _a2_parameters_by_fold(run_seed_dir: Path) -> dict[int, dict[str, float]]:
    tuning = pd.read_csv(run_seed_dir / "inner_cv_tuning.csv")
    selected = tuning[tuning["arm"].eq("A2_current_champion") & tuning["selected"].astype(bool)]
    return {
        int(row.outer_fold): {
            "max_features": float(row.max_features),
            "min_samples_leaf": int(row.min_samples_leaf),
        }
        for row in selected.itertuples()
    }


def _pair_mean_baseline(
    frame: pd.DataFrame, target: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Leave-fold-out equal-extractant pair-label mean (no ligand features)."""

    out = np.full(len(frame), np.nan)
    labels = frame["pair_label"].astype(str).to_numpy()
    extractant = frame["extractant"].astype(str).to_numpy()
    for fold in np.unique(folds):
        train = folds != fold
        table = pd.DataFrame({"pair": labels[train], "e": extractant[train], "y": target[train]})
        means = table.groupby(["pair", "e"])["y"].mean().groupby(level="pair").mean()
        test = ~train
        mapped = pd.Series(labels[test]).map(means).to_numpy(dtype=float)
        # Pair labels absent from this training fold fall back to the training
        # fold's own equal-extractant mean (never touches other folds).
        mapped = np.where(np.isfinite(mapped), mapped, float(means.mean()))
        out[test] = mapped
    return out


def _within_pair_r2(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> float:
    """R2 after removing per-(fold, pair_label) means from truth and prediction."""

    key = frame["outer_fold"].astype(str) + "|" + frame["pair_label"].astype(str)
    t = pd.Series(truth).groupby(key.to_numpy()).transform(lambda s: s - s.mean()).to_numpy()
    p = pd.Series(prediction).groupby(key.to_numpy()).transform(lambda s: s - s.mean()).to_numpy()
    total = float(np.sum(t**2))
    return 1.0 - float(np.sum((t - p) ** 2)) / total if total > 0 else float("nan")


def _pair_level_r2(frame: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> pd.Series:
    rows = {}
    for label, block in frame.assign(_y=truth, _p=prediction).groupby("pair_label", sort=False):
        y = block["_y"].to_numpy()
        p = block["_p"].to_numpy()
        total = float(np.sum((y - y.mean()) ** 2))
        rows[str(label)] = 1.0 - float(np.sum((y - p) ** 2)) / total if total > 0 else np.nan
    return pd.Series(rows, dtype=float)


def evaluate_split_seed(
    frame: pd.DataFrame,
    a2_columns: tuple[str, ...],
    lig_columns: tuple[str, ...],
    *,
    run_dir: Path,
    split_seed: int,
    model_seed: int,
    arms: tuple[str, ...],
    n_estimators: int,
    n_jobs: int,
    ensemble_seeds: tuple[int, ...],
    log,
) -> pd.DataFrame:
    run_seed_dir = _run_directory_for_seed(run_dir, split_seed)
    run_oof = pd.read_csv(
        run_seed_dir / "oof_predictions.csv",
        usecols=["pair_id", "outer_fold", "prediction_A2_current_champion"],
    )
    merged = frame[["pair_id"]].merge(run_oof, on="pair_id", how="left")
    if len(merged) != len(frame) or merged["outer_fold"].isna().any():
        raise AssertionError("cohort rows missing from, or duplicated in, the reference run OOF file")
    folds = merged["outer_fold"].to_numpy(dtype=int)
    reference = merged["prediction_A2_current_champion"].to_numpy(dtype=float)
    a2_params = _a2_parameters_by_fold(run_seed_dir)

    target = frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
    groups = frame["extractant"].astype(str).to_numpy()
    extended = tuple(a2_columns) + tuple(lig_columns)
    predictions: dict[str, np.ndarray] = {arm: np.full(len(frame), np.nan) for arm in arms}
    scale_diag = np.full(len(frame), np.nan)

    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = ~train
        train_frame = frame[train]
        test_frame = frame[test]
        y_train = target[train]
        g_train = groups[train]
        if fold not in a2_params:
            raise KeyError(f"no selected A2 parameters for outer fold {fold} in {run_seed_dir}")
        params = a2_params[fold]
        outer_seed = int(model_seed) + int(fold) * 1009 + 9_999_991
        forest = ForestParameters(
            n_estimators=n_estimators,
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
        )
        t0 = time.time()
        for arm in arms:
            t1 = time.time()
            if arm in ("A2_refit", "A2_lig2d", "A2_lig2d_SHUF"):
                columns = a2_columns if arm == "A2_refit" else extended
                fit_frame, predict_frame = train_frame, test_frame
                if arm == "A2_lig2d_SHUF":
                    # Attribution control: every ligand (training and held-out
                    # alike) receives the descriptor vector of a randomly chosen
                    # OTHER ligand, with one permutation per outer fold.  Row
                    # count, column count and value distribution are identical
                    # to A2_lig2d; only the ligand->descriptor correspondence
                    # is destroyed.
                    ligands = np.array(sorted(frame["extractant"].astype(str).unique()))
                    rng = np.random.default_rng(outer_seed + 424_242)
                    perm = rng.permutation(len(ligands))
                    while np.any(perm == np.arange(len(ligands))):
                        perm = rng.permutation(len(ligands))
                    donor = dict(zip(ligands, ligands[perm]))
                    table = frame.drop_duplicates("extractant").set_index("extractant")[list(lig_columns)]
                    shuffled = frame["extractant"].astype(str).map(donor)  # donor ligand per row
                    shuffled_values = table.loc[shuffled.to_numpy()].to_numpy()
                    shuffled_frame = frame.copy()
                    shuffled_frame[list(lig_columns)] = shuffled_values
                    fit_frame, predict_frame = shuffled_frame[train], shuffled_frame[test]
                model = AntisymmetricExtraTreesRegressor(
                    columns, random_state=outer_seed, n_jobs=n_jobs, **forest.as_kwargs()
                )
                model.fit(fit_frame, y_train, g_train)
                predictions[arm][test] = model.predict(predict_frame)
            elif arm in ("HIER", "HIER_lig2d"):
                columns = a2_columns if arm == "HIER" else extended
                model = HierarchicalPairRegressor(
                    columns, level=forest, deviation=forest, random_state=outer_seed, n_jobs=n_jobs
                )
                model.fit(train_frame, y_train, g_train)
                predictions[arm][test] = model.predict(test_frame)
            elif arm.startswith("SCALE"):
                # SCALE[_sNN][_lig2d]: NN = percent shrinkage of the predicted
                # scale toward the training mean (s100 = pure pair-trend prior).
                match = re.fullmatch(r"SCALE(?:_s(\d{1,3}))?(_lig2d)?", arm)
                if match is None:
                    raise ValueError(f"unknown arm {arm!r}")
                shrinkage = float(match.group(1)) / 100.0 if match.group(1) else 0.0
                columns = extended if match.group(2) else a2_columns
                model = ScaleTrendPairRegressor(
                    columns,
                    scale=ForestParameters(n_estimators=n_estimators, max_features=0.35, min_samples_leaf=2),
                    residual=forest,
                    random_state=outer_seed,
                    n_jobs=n_jobs,
                    scale_shrinkage=shrinkage,
                )
                model.fit(train_frame, y_train, g_train)
                predictions[arm][test] = model.predict(test_frame)
                if np.isnan(scale_diag[test]).all():
                    scale_diag[test] = model.predict_scale(test_frame)
            elif arm in ("PRIOR", "PRIOR_lig2d", "PRIOR_TRENDONLY"):
                columns = extended if arm.endswith("_lig2d") else a2_columns
                model = PriorAugmentedPairRegressor(
                    columns,
                    forest=forest,
                    scale=ForestParameters(n_estimators=n_estimators, max_features=0.35, min_samples_leaf=2),
                    random_state=outer_seed,
                    n_jobs=n_jobs,
                    mode="trend_only" if arm == "PRIOR_TRENDONLY" else "scale",
                )
                model.fit(train_frame, y_train, g_train)
                predictions[arm][test] = model.predict(test_frame)
            elif arm == "A2_ens":
                stack = []
                for k, seed in enumerate(ensemble_seeds):
                    model = AntisymmetricExtraTreesRegressor(
                        a2_columns,
                        random_state=int(seed) + int(fold) * 1009 + 9_999_991,
                        n_jobs=n_jobs,
                        **forest.as_kwargs(),
                    )
                    model.fit(train_frame, y_train, g_train)
                    stack.append(model.predict(test_frame))
                predictions[arm][test] = np.mean(stack, axis=0)
            else:
                raise ValueError(f"unknown arm {arm!r}")
            log(f"  seed {split_seed} fold {fold} {arm:12s} {time.time() - t1:6.1f}s")
        log(f"seed {split_seed} fold {fold} done in {time.time() - t0:.1f}s")

    out = frame[list(IDENTITY_COLUMNS)].copy()
    out["outer_fold"] = folds
    out["split_seed"] = int(split_seed)
    out["model_seed"] = int(model_seed)
    out[f"prediction_{RUN_REFERENCE}"] = reference
    out[f"prediction_{PAIRMEAN}"] = _pair_mean_baseline(frame, target, folds)
    for arm in arms:
        out[f"prediction_{arm}"] = predictions[arm]
        out[f"prediction_{arm}_TP"] = transitive_projection(frame, predictions[arm])
    out[f"prediction_{RUN_REFERENCE}_TP"] = transitive_projection(frame, reference)
    out["scale_SCALE_predicted"] = scale_diag
    return out


def aggregate(oof: pd.DataFrame, arms_all: list[str], output_dir: Path, bootstrap_replicates: int) -> dict[str, Any]:
    per_seed_rows = []
    per_extractant_rows = []
    pair_r2_rows = []
    for seed, block in oof.groupby("split_seed", sort=True):
        overall, per_extractant, _ = arm_metric_table(block, arms_all, baseline_arm=BASELINE)
        overall["split_seed"] = int(seed)
        truth = block[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        overall["within_pair_r2"] = [
            _within_pair_r2(block, truth, block[f"prediction_{arm}"].to_numpy(dtype=float))
            for arm in overall["arm"]
        ]
        overall["median_pair_level_r2"] = [
            float(_pair_level_r2(block, truth, block[f"prediction_{arm}"].to_numpy(dtype=float)).median())
            for arm in overall["arm"]
        ]
        per_seed_rows.append(overall)
        per_extractant["split_seed"] = int(seed)
        per_extractant_rows.append(per_extractant)
        for arm in arms_all:
            series = _pair_level_r2(block, truth, block[f"prediction_{arm}"].to_numpy(dtype=float))
            pair_r2_rows.append(series.rename(arm).to_frame().assign(split_seed=int(seed)).reset_index(names="pair_label").melt(id_vars=["pair_label", "split_seed"], var_name="arm", value_name="r2"))
    per_seed = pd.concat(per_seed_rows, ignore_index=True)
    per_extractant = pd.concat(per_extractant_rows, ignore_index=True)
    pair_r2 = pd.concat(pair_r2_rows, ignore_index=True)

    metric_columns = [
        "equal_extractant_macro_mae",
        "pooled_micro_mae",
        "median_extractant_mae",
        "worst_quartile_extractant_mae",
        "pooled_r2",
        "equal_extractant_macro_r2",
        "median_extractant_r2",
        "pearson_squared_scale_free",
        "prediction_dispersion_ratio",
        "adjacent_ln_mae",
        "nonadjacent_ln_mae",
        "sign_accuracy",
        "within_pair_r2",
        "median_pair_level_r2",
    ]
    leaderboard = per_seed.groupby("arm")[metric_columns].mean()
    leaderboard["macro_mae_split_sd"] = per_seed.groupby("arm")["equal_extractant_macro_mae"].std()
    baseline_by_seed = per_seed[per_seed["arm"].eq(BASELINE)].set_index("split_seed")["equal_extractant_macro_mae"]
    deltas = []
    for arm, block in per_seed.groupby("arm"):
        d = baseline_by_seed.loc[block["split_seed"]].to_numpy() - block["equal_extractant_macro_mae"].to_numpy()
        deltas.append({"arm": arm, "mean_delta_vs_A2_refit": float(np.mean(d)), "positive_split_seeds": int(np.sum(d > 0)), "split_seed_count": int(len(d)), "per_seed_delta": ";".join(f"{v:+.4f}" for v in d)})
    leaderboard = leaderboard.join(pd.DataFrame(deltas).set_index("arm")).sort_values("equal_extractant_macro_mae").reset_index()

    comparisons = {f"{BASELINE}_vs_{arm}": (BASELINE, arm) for arm in arms_all if arm != BASELINE}
    bootstrap = paired_extractant_bootstrap(oof, comparisons, replicates=bootstrap_replicates, seed=8675309)

    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    per_extractant.to_csv(output_dir / "per_extractant_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    pair_r2.to_csv(output_dir / "pair_level_r2.csv", index=False)
    return {
        "leaderboard": leaderboard.to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[104729, 130363, 155921, 196613, 262147])
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--arms", nargs="+", default=list(ALL_ARMS))
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--ensemble-seeds", type=int, nargs="+", default=[42, 7, 137])
    parser.add_argument("--descriptors", type=Path, default=DESCRIPTOR_PATH)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="smoke test: 30 trees, first seed only")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=Warning, module="pandas")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / "runs" / f"gen4_candidates_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(message: str) -> None:
        line = f"[{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    n_estimators = 30 if args.quick else int(args.n_estimators)
    seeds = args.split_seeds[:1] if args.quick else list(args.split_seeds)
    arms = tuple(args.arms)

    source = pd.read_parquet(DATASET_PATH)
    pair_data = build_lanthanide_pair_dataset(
        source,
        pair_scope="all",
        require_geometry=True,
        replicate_policy="unique",
        quarantine_known_bad=True,
        require_complete_conditions=True,
        delta3d_feature_set="compact-invariant",
    )
    frame = pair_data.frame.reset_index(drop=True)
    if "extractant_family" not in frame.columns:
        # The gen3 adapter maps extractant_group -> family, and in this source
        # extractant_group == canonical_smiles for every row, so family is the
        # extractant itself; the metric tables only need the column to exist.
        frame["extractant_family"] = frame["extractant"].astype(str)
    registry = build_feature_registry(pair_data)
    a2_columns = tuple(registry.ablation_columns("A2"))
    lig_columns: tuple[str, ...] = ()
    needs_descriptors = any("_lig2d" in arm for arm in arms)
    if needs_descriptors:
        if not args.descriptors.exists():
            raise FileNotFoundError(f"descriptor table {args.descriptors} not found; build it or drop the *_lig2d arms")
        frame, lig_columns = attach_ligand_descriptors(frame, pd.read_parquet(args.descriptors))
        coverage = frame[list(lig_columns[:1])].notna().mean().iloc[0]
        log(f"attached {len(lig_columns)} ligand descriptor columns; ligand coverage {coverage:.3f}")
    log(f"cohort {len(frame)} pairs, {frame['extractant'].nunique()} extractants, {len(a2_columns)} A2 columns; arms {arms}; seeds {seeds}; trees {n_estimators}")

    frames = []
    for split_seed in seeds:
        frames.append(
            evaluate_split_seed(
                frame,
                a2_columns,
                lig_columns,
                run_dir=args.run_dir,
                split_seed=int(split_seed),
                model_seed=int(args.model_seed),
                arms=arms,
                n_estimators=n_estimators,
                n_jobs=int(args.n_jobs),
                ensemble_seeds=tuple(int(s) for s in args.ensemble_seeds),
                log=log,
            )
        )
        frames[-1].to_csv(output_dir / f"oof_predictions_seed{split_seed}.csv", index=False)
        block = frames[-1]
        truth = block[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        groups = block["extractant"].to_numpy()
        for arm in [RUN_REFERENCE, PAIRMEAN, *arms, *[f"{a}_TP" for a in arms]]:
            log(f"seed {split_seed} macro MAE {arm:16s} {equal_group_macro_mae(truth, block[f'prediction_{arm}'].to_numpy(dtype=float), groups):.4f}")

    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    arms_all = [BASELINE] + [a for a in arms if a != BASELINE] + [f"{a}_TP" for a in arms] + [RUN_REFERENCE, f"{RUN_REFERENCE}_TP", PAIRMEAN]
    summary = aggregate(oof, arms_all, output_dir, int(args.bootstrap_replicates))
    summary.update(
        {
            "run_dir": str(args.run_dir),
            "split_seeds": seeds,
            "model_seed": int(args.model_seed),
            "arms": list(arms),
            "n_estimators": n_estimators,
            "n_ligand_descriptor_columns": len(lig_columns),
            "baseline_arm": BASELINE,
            "note": "exploratory harness on the frozen gen3 outer folds; not a protocol run",
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    lb = pd.read_csv(output_dir / "leaderboard.csv")
    log("\n" + lb[["arm", "equal_extractant_macro_mae", "macro_mae_split_sd", "mean_delta_vs_A2_refit", "positive_split_seeds", "pooled_micro_mae", "pooled_r2", "median_extractant_r2", "within_pair_r2", "prediction_dispersion_ratio"]].round(4).to_string(index=False))
    log(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
