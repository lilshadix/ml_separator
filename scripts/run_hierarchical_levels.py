"""gen6 Experiment C — which component of the level fails to transfer to a new ligand?

Phase 1 measured *that* the per-ligand level is what a model gets wrong on new
chemistry.  Experiment C asks *which piece of the physical decomposition*

.. code-block:: text

    log D(l, m, c)  ~=  alpha_l  +  F_cond(l, c)  +  F_metal(l, m, c)  +  eps

is responsible, by building estimators in which the pieces are separately fitted
and can be swapped for their true values one at a time, and by asking whether
that structure buys anything a monolithic forest does not already have.

Eight models are scored on identical test rows in every regime:

===================  =========================================================
``MONO_ET``          the Experiment A EXPANDED arm (ExtraTrees on
                     METAL+COND+LIG2D_EXT+MASSACTION).  Under
                     ``unseen_chemotype`` in a FULL run it must reproduce
                     Experiment A's ``prediction_EXPANDED`` **bit-for-bit**; the
                     run stops if it does not.
``MONO_RIDGE``       plain ridge on C1's compact design — the linear control
``C1_HIER_RIDGE``    the same design plus a ridge-penalised random intercept per
                     training ligand (zero for a held-out ligand)
``C2_TWO_STAGE``     Stage A on the (ligand, condition) cell mean + Stage B on
                     the **cross-fitted** residual
``C2_TRUECENTRE``    protocol amendment: Stage B on the TRUE within-cell
                     departure.  Chosen after a one-seed preview, so every
                     C2_TRUECENTRE contrast is labelled EXPLORATORY
``C3_SHARED_RIDGE``  C1's design stacked with within-cell pair-difference rows
``ORACLE_LEVEL``     true cell mean + predicted departure
``ORACLE_METAL``     predicted cell mean + true departure
===================  =========================================================

Traps this runner is written around:

1. **The oracles are attribution devices, not models.**  Both read test labels.
   The decisive quantity is the *difference of the two oracle gains*, which is
   scored directly as the contrast ``("ORACLE_METAL", "ORACLE_LEVEL")`` so the
   shared ``C2_TWO_STAGE`` reference cancels exactly instead of being subtracted
   by hand from two intervals.
2. **A singleton (ligand, condition) cell makes the level oracle trivially
   informative**, so hypothesis C1 is scored on test cells with
   ``cell_n_metals >= 2`` only.  The all-rows version is reported beside it and
   is never the verdict.
3. **The inner CV must be grouped like the outer regime.**  ``tune_ridge`` and
   the Stage A cross-fit both receive the regime's own group column
   (``tanimoto_cluster`` / ``ecfp_cluster`` / ``series_id``), otherwise the
   penalty search discovers a ligand intercept that fits training beautifully and
   transfers nothing.
4. **"Reproduces Experiment A" must be checked, not assumed.**  The check is
   fail-closed in a FULL run and explicitly SKIPPED (never silently passed) in
   ``--pilot``, where 120 trees cannot equal Experiment A's 400.
5. **Antisymmetry is measured here, not taken from the module.**
   ``hierarchical.pair_consistency`` returns a hard-coded ``antisymmetry_max``
   of 0.0 (the value is never assigned in its loop), so C4's antisymmetry half
   would be vacuous if it were quoted from there.  This runner recomputes it
   from the row predictions; the module's transitivity number is used as-is.

Example::

    .venv/bin/python scripts/run_hierarchical_levels.py --pilot
    .venv/bin/python scripts/run_hierarchical_levels.py --pilot --regimes unseen_series
    .venv/bin/python scripts/run_hierarchical_levels.py \
        --regimes unseen_chemotype unseen_ligand unseen_series \
        --split-seeds 104729 130363 155921 196613 262147
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6 import GEN6_LAYER  # noqa: E402
from lanthanide_separation.gen6.chemistry import (  # noqa: E402
    ChemistryMap, build_chemistry_map, cluster_manifest,
)
from lanthanide_separation.gen6.cohorts import (  # noqa: E402
    BASE_MIN_CELLS, EXPANDED_ARM, EXPANDED_MIN_CELLS, assert_split_integrity, diversity_splits,
    seeded_group_kfold,
)
from lanthanide_separation.gen6.hierarchical import (  # noqa: E402
    FIXED_PENALTY_GRID, FOREST_BLOCKS, LIGAND_PENALTY_GRID, MODEL_NAMES, RIDGE_BASE_BLOCKS,
    HierarchicalRidge, derived_pairs, fit_two_stage, pair_consistency, pair_difference_rows,
    pair_label_mean_null, pair_metrics, predict_two_stage, tune_ridge,
)
from lanthanide_separation.gen6.manifest import (  # noqa: E402
    RunManifest, sha256_frame, sha256_json, validate_run, write_success,
)
from lanthanide_separation.gen6.metrics import (  # noqa: E402
    HARD_CHEMISTRY_THRESHOLDS, UNIT_STATISTICS, gen6_metric_table, paired_unit_bootstrap,
    per_unit_statistics,
)
from lanthanide_separation.levels import (  # noqa: E402
    LEVEL_IDENTITY_COLUMNS, LEVEL_TARGET_COLUMN, LevelData, LevelForestParameters, LevelRegressor,
    build_level_dataset, group_balanced_weights,
)

DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
DESCRIPTOR_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_2d_descriptors.parquet"
#: Experiment A's five-seed run: the OOF frame ``MONO_ET`` must reproduce.
REFERENCE_OOF_PATH = REPO_ROOT / "runs" / "gen6_expA_5seed" / "oof_predictions.parquet"
#: Experiment A's champion feature set, whose EXPANDED arm is MONO_ET.
REFERENCE_FEATURE_SET = "MC_lig2d_ext_massaction"
REFERENCE_ARM = EXPANDED_ARM

#: The five split seeds every gen5/gen6 study uses.  They re-partition the same
#: ligands, so they measure split sensitivity — not independent replication.
DEFAULT_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
#: gen5's fold seed formula, reproduced exactly (see run_gen5_levels.evaluate_regime).
FOLD_SEED_STRIDE = 1009
FOLD_SEED_OFFSET = 9_999_991

#: The three regimes of the pre-registration, and the column each holds out.
REGIMES: tuple[str, ...] = ("unseen_chemotype", "unseen_ligand", "unseen_series")
PRIMARY_REGIME = "unseen_chemotype"
REGIME_GROUP_COLUMN: Mapping[str, str] = {
    "unseen_chemotype": "tanimoto_cluster",
    "unseen_ligand": "ecfp_cluster",
    "unseen_series": "series_id",
}
#: What must NOT be shared between a fold's train and test side, per regime.
#: ``unseen_series`` deliberately keeps the ligand in training — that is the
#: regime whose whole point is that C1's random intercept is live — so listing
#: ``extractant`` there would flag the design itself as a leak.
REGIME_LEAKAGE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "unseen_chemotype": ("extractant", "ecfp_cluster", "tanimoto_cluster"),
    "unseen_ligand": ("extractant", "ecfp_cluster"),
    "unseen_series": ("series_id",),
}

#: Scoring endpoints.  ``multi_metal_cells`` is the population hypothesis C1 is
#: pre-registered on; ``all`` is reported beside it and decides nothing.
ENDPOINT_ALL = "all"
ENDPOINT_MULTI = "multi_metal_cells"

#: Statistics carried through the per-unit bootstrap.  ``mae`` averaged over ECFP
#: clusters *is* the study's primary macro MAE.
BOOTSTRAP_STATISTICS: tuple[str, ...] = tuple(UNIT_STATISTICS)
#: The pre-registered C3 margin: "does not beat MONO_ET by more than this".
C3_MARGIN = 0.02
#: C4's machine-precision bound.
IDENTITY_TOLERANCE = 1e-9

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "oof_predictions.parquet", "per_ligand_metrics.csv", "arm_metrics.csv",
    "hard_chemistry_metrics.csv", "contrasts.csv", "contrast_summary.csv",
    "component_attribution.csv", "penalties.csv", "pair_metrics.csv",
    "pair_predictions.parquet", "split_manifest.json", "decision_report.md", "summary.json",
    "manifest.json",
)


# --------------------------------------------------------------------------- #
# Contrasts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Contrast:
    """One comparison.

    ``point_delta = statistic(reference) - statistic(candidate)``, so a **positive
    delta always means the candidate is better** (lower error).  The pair is
    ordered so that positive is the direction the hypothesis predicts.
    """

    name: str
    reference: str
    candidate: str
    preregistered: bool
    question: str


CONTRASTS: tuple[Contrast, ...] = (
    Contrast("C1_HIER_RIDGE_vs_MONO_RIDGE", "MONO_RIDGE", "C1_HIER_RIDGE", True,
             "hypothesis C2: does partial pooling help where the ligand is known, "
             "and vanish where it is not?"),
    Contrast("C2_TWO_STAGE_vs_MONO_ET", "MONO_ET", "C2_TWO_STAGE", True,
             "hypothesis C3: does the two-stage structure substitute for chemical coverage?"),
    Contrast("C2_TRUECENTRE_vs_MONO_ET", "MONO_ET", "C2_TRUECENTRE", False,
             "hypothesis C3, amendment — EXPLORATORY: the variant was chosen after a "
             "one-seed preview, so this contrast is exploratory-confirmed at best"),
    Contrast("C3_SHARED_RIDGE_vs_C1_HIER_RIDGE", "C1_HIER_RIDGE", "C3_SHARED_RIDGE", False,
             "descriptive: what does stacking the pair-difference rows cost the level fit?"),
    Contrast("level_oracle_gain", "C2_TWO_STAGE", "ORACLE_LEVEL", True,
             "hypothesis C1: error removed by handing the model the true cell mean"),
    Contrast("metal_oracle_gain", "C2_TWO_STAGE", "ORACLE_METAL", True,
             "hypothesis C1: error removed by handing the model the true metal departure"),
    Contrast("level_minus_metal_gain", "ORACLE_METAL", "ORACLE_LEVEL", True,
             "hypothesis C1, DECISIVE: the difference of the two oracle gains "
             "(the shared C2_TWO_STAGE reference cancels exactly)"),
)
DECISIVE_CONTRAST = "level_minus_metal_gain"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--descriptors", type=Path, default=DESCRIPTOR_PATH,
                   help="gen4 extended-2D descriptor parquet (LIG2D_EXT block); pass '' to skip")
    p.add_argument("--chemistry-map", type=Path, default=None,
                   help="frozen chemistry_map.parquet; rebuilt from the source table when omitted")
    p.add_argument("--eval-min-cells", type=int, default=EXPANDED_MIN_CELLS,
                   help="eligibility of the ONE shared cohort (3 = the gen6 EXPANDED cohort)")
    p.add_argument("--base-min-cells", type=int, default=BASE_MIN_CELLS,
                   help="only used to build the chemotype folds exactly as Experiment A did")
    p.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=list(REGIMES))
    p.add_argument("--split-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-features", type=float, default=0.30)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--replicate-policy", default="mean", choices=["mean", "unique", "all"])
    p.add_argument("--log-d-floor", type=float, default=-6.0)
    p.add_argument("--thresholds", nargs="+", type=float, default=list(HARD_CHEMISTRY_THRESHOLDS),
                   help="hard-chemistry nearest-neighbour Tanimoto cut-offs (to the fold's training "
                        "ligands)")
    p.add_argument("--replicates", type=int, default=5000, help="bootstrap replicates")
    p.add_argument("--bootstrap-seed", type=int, default=8675309)
    p.add_argument("--reference-oof", type=Path, default=REFERENCE_OOF_PATH,
                   help="Experiment A OOF parquet that MONO_ET must reproduce under "
                        "unseen_chemotype in a FULL run")
    p.add_argument("--reference-feature-set", default=REFERENCE_FEATURE_SET)
    p.add_argument("--provenance-state-json", type=Path, default=None,
                   help="provenance audit state to record in the manifest; without it the manifest "
                        "records {'status': 'not_audited_in_this_run'} rather than a guess")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--pilot", action="store_true",
                   help="smoke test: one seed, unseen_chemotype only, 120 trees; the MONO_ET "
                        "reproduction check is SKIPPED because 120 trees cannot equal 400")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FoldSpec:
    """One outer fold of one regime: positional indices into the shared cohort."""

    fold: int
    regime: str
    seed: int
    train_index: np.ndarray
    test_index: np.ndarray
    held_out_groups: tuple[str, ...]


def regime_folds(frame: pd.DataFrame, *, regime: str, seed: int, folds: int,
                 base_min_cells: int = BASE_MIN_CELLS) -> list[FoldSpec]:
    """The outer folds of one regime.

    ``unseen_chemotype`` goes through :func:`diversity_splits` with its default
    arms and takes the ``EXPANDED`` training index — not because the arms matter
    here (they do not; the partition depends only on the group column, the fold
    count and the seed) but because that is literally the call Experiment A made,
    and MONO_ET has to land on byte-identical training rows for the reproduction
    check to mean anything.  The other two regimes are plain grouped K-folds on
    the column the regime holds out.
    """
    if regime not in REGIME_GROUP_COLUMN:
        raise ValueError(f"unknown regime {regime!r}; choose from {REGIMES}")
    group_column = REGIME_GROUP_COLUMN[regime]
    groups = frame[group_column].astype(str).to_numpy()
    out: list[FoldSpec] = []
    if regime == PRIMARY_REGIME:
        for split in diversity_splits(frame, group_column=group_column, n_splits=folds,
                                      seed=seed, base_min_cells=base_min_cells):
            out.append(FoldSpec(fold=split.fold, regime=regime, seed=int(seed),
                                train_index=np.asarray(split.train_index_by_arm[EXPANDED_ARM]),
                                test_index=np.asarray(split.test_index),
                                held_out_groups=tuple(split.held_out_groups)))
        return out
    for fold, (train_index, test_index) in enumerate(seeded_group_kfold(groups, folds, seed)):
        out.append(FoldSpec(fold=fold, regime=regime, seed=int(seed),
                            train_index=np.asarray(train_index),
                            test_index=np.asarray(test_index),
                            held_out_groups=tuple(sorted(set(groups[test_index])))))
    return out


def regime_integrity(frame: pd.DataFrame, splits: Sequence[FoldSpec], *, regime: str) -> dict:
    """Prove the regime's hold-out actually holds out, or say where it does not.

    The leakage columns differ by regime on purpose: under ``unseen_series`` the
    ligand *is* in training and flagging that would flag the design.
    """
    columns = REGIME_LEAKAGE_COLUMNS[regime]
    report: dict = {"regime": regime, "ok": True, "leakage_columns": list(columns), "folds": []}
    covered: set[int] = set()
    for split in splits:
        entry: dict = {"fold": int(split.fold), "n_test_rows": int(split.test_index.size),
                       "n_train_rows": int(split.train_index.size), "leaks": {}}
        if set(split.train_index.tolist()) & set(split.test_index.tolist()):
            entry["leaks"]["row_overlap"] = True
            report["ok"] = False
        for column in columns:
            values = frame[column].astype(str).to_numpy()
            shared = set(values[split.train_index]) & set(values[split.test_index])
            if shared:
                entry["leaks"][column] = sorted(shared)[:5]
                report["ok"] = False
        covered |= set(split.test_index.tolist())
        report["folds"].append(entry)
    report["all_rows_tested_once"] = bool(len(covered) == len(frame))
    report["n_rows_never_tested"] = int(len(frame) - len(covered))
    if not report["all_rows_tested_once"]:
        report["ok"] = False
    return report


def cells_never_split_across_folds(frame: pd.DataFrame, splits: Sequence[FoldSpec]) -> dict:
    """A (ligand, condition) cell must live in exactly one fold.

    Everything about Experiment C is defined per cell — Stage A's target, the
    oracles, ``cell_n_metals``, the multi-metal endpoint — and every one of those
    numbers would be computed on a truncated cell if the folds ever cut one in
    half.  It cannot happen for these three group columns, so this is a cheap
    guard against a fourth regime being added carelessly later.
    """
    key = (frame["extractant"].astype(str) + "|" + frame["condition_id"].astype(str)).to_numpy()
    fold_of = np.full(len(frame), -1, dtype=int)
    for split in splits:
        fold_of[split.test_index] = split.fold
    table = pd.DataFrame({"cell": key, "fold": fold_of})
    per_cell = table.groupby("cell")["fold"].nunique()
    split_cells = per_cell[per_cell > 1]
    return {"ok": bool(split_cells.empty), "n_cells": int(len(per_cell)),
            "n_cells_split_across_folds": int(len(split_cells)),
            "examples": [str(c) for c in split_cells.index[:5]]}


# --------------------------------------------------------------------------- #
# One fold: all eight models
# --------------------------------------------------------------------------- #

def fit_fold(
    data: LevelData, *, train: pd.DataFrame, test: pd.DataFrame,
    params: LevelForestParameters, group_column: str, split_seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Fit and predict every model of one fold.

    Returns ``(predictions, extras, record)`` where ``extras`` carries the
    two-stage internals (Â, B̂, the true cell mean and the cell's metal count) and
    ``record`` carries the chosen penalties, the C1 component attribution and the
    cross-fit audit.
    """
    y_train = train[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    weights = group_balanced_weights(train["ecfp_cluster"].astype(str))
    train_ligands = train["extractant"].astype(str)
    test_ligands = test["extractant"].astype(str)
    predictions: dict[str, np.ndarray] = {}
    record: dict = {}

    # --- MONO_ET: the Experiment A EXPANDED arm, reached the same way ---------
    forest_columns = data.block_columns(FOREST_BLOCKS)
    mono = LevelRegressor(forest_columns, params).fit(train, y_train, groups=train["ecfp_cluster"])
    predictions["MONO_ET"] = mono.predict(test)

    # --- MONO_RIDGE: plain ridge, penalties from inner CV grouped like the outer regime
    plain = tune_ridge(train, data, group_column=group_column, hierarchical=False, seed=split_seed)
    plain_design = plain["design"]
    mono_ridge = HierarchicalRidge(plain_design, plain["lambda_fixed"], None).fit(
        plain_design.transform(train), y_train, weights=weights)
    predictions["MONO_RIDGE"] = mono_ridge.predict(plain_design.transform(test))

    # --- C1: the same design plus a penalised intercept per TRAINING ligand ---
    hier = tune_ridge(train, data, group_column=group_column, hierarchical=True, seed=split_seed)
    design = hier["design"]
    x_train = design.transform(train)
    x_test = design.transform(test)
    c1 = HierarchicalRidge(design, hier["lambda_fixed"], hier["lambda_ligand"]).fit(
        x_train, y_train, weights=weights, ligands=train_ligands)
    components = c1.predict_components(x_test, test_ligands)
    predictions["C1_HIER_RIDGE"] = components["prediction"].to_numpy(dtype=float)

    # --- C3: C1's penalties, C1's design, plus the within-cell difference rows -
    extra_x, extra_y, extra_w, pair_rows = pair_difference_rows(train, x_train, weights)
    c3 = HierarchicalRidge(design, hier["lambda_fixed"], hier["lambda_ligand"]).fit(
        x_train, y_train, weights=weights, ligands=train_ligands,
        extra_x=extra_x, extra_y=extra_y, extra_w=extra_w)
    predictions["C3_SHARED_RIDGE"] = c3.predict(x_test, test_ligands)

    # --- C2 and the two oracles ------------------------------------------------
    two_stage = fit_two_stage(train, data, params=params, group_column=group_column,
                              seed=split_seed)
    two = predict_two_stage(two_stage, test, data)
    for name in ("C2_TWO_STAGE", "C2_TRUECENTRE", "ORACLE_LEVEL", "ORACLE_METAL"):
        predictions[name] = two[f"prediction_{name}"].to_numpy(dtype=float)

    missing = [name for name in MODEL_NAMES if name not in predictions]
    if missing:
        raise RuntimeError(f"models not produced by this fold: {missing}")
    non_finite = {name: int((~np.isfinite(values)).sum()) for name, values in predictions.items()}
    if any(non_finite.values()):
        raise RuntimeError(
            f"non-finite predictions in this fold: { {k: v for k, v in non_finite.items() if v} }. "
            f"The usual cause is fewer than 10 multi-metal training rows, which leaves "
            f"stage_b_truecentre unfitted and C2_TRUECENTRE / ORACLE_LEVEL NaN.")

    extras = two[["stage_a", "stage_b", "stage_b_truecentre", "cell_true_mean",
                  "cell_n_metals"]].copy()

    # --- what the run has to be able to show afterwards ------------------------
    attribution = (components.drop(columns=["prediction"]).abs().mean()
                   .rename("mean_abs_contribution").to_frame())
    attribution["mean_contribution"] = components.drop(columns=["prediction"]).mean()
    attribution.index.name = "component"
    record["attribution"] = attribution.reset_index()
    record["penalties"] = [
        {"model": "MONO_RIDGE", "lambda_fixed": float(plain["lambda_fixed"]),
         "lambda_ligand": None,
         "inner_macro_mae": float(plain["inner_table"].iloc[0]["inner_macro_mae"]),
         "n_grid_points": int(len(plain["inner_table"]))},
        {"model": "C1_HIER_RIDGE", "lambda_fixed": float(hier["lambda_fixed"]),
         "lambda_ligand": None if hier["lambda_ligand"] is None else float(hier["lambda_ligand"]),
         "inner_macro_mae": float(hier["inner_table"].iloc[0]["inner_macro_mae"]),
         "n_grid_points": int(len(hier["inner_table"]))},
        {"model": "C3_SHARED_RIDGE", "lambda_fixed": float(hier["lambda_fixed"]),
         "lambda_ligand": None if hier["lambda_ligand"] is None else float(hier["lambda_ligand"]),
         "inner_macro_mae": float("nan"), "n_grid_points": 0},
    ]
    record["n_pair_difference_rows"] = int(len(pair_rows))
    record["n_design_columns"] = int(x_train.shape[1])
    record["n_train_ligands"] = int(train_ligands.nunique())
    record["n_ligand_intercepts"] = int(len(c1.gamma_))
    audit = two_stage.crossfit_audit
    # The residual Stage B trained on must come from a Stage A fit that never saw
    # the row's own cell.  The module records which inner fold produced each cell's
    # prediction; a row and its cell must agree, and no group may straddle two
    # inner folds.  Checked here so the runner's own artifact carries the evidence.
    cell_fold = dict(zip(audit["cell_keys"], audit["cell_inner_fold"]))
    row_ok = all(cell_fold[k] == f for k, f in zip(audit["row_cell_keys"], audit["row_inner_fold"]))
    record["crossfit"] = {
        "ok": bool(row_ok and int(np.min(audit["row_inner_fold"])) >= 0),
        "n_cells": int(audit["n_cells"]),
        "n_cells_multi_metal": int(audit["n_cells_multi_metal"]),
        "crossfit_folds": int(audit["crossfit_folds"]),
        "row_fold_matches_cell_fold": bool(row_ok),
    }
    return predictions, extras, record


# --------------------------------------------------------------------------- #
# One (regime, seed)
# --------------------------------------------------------------------------- #

@dataclass
class RegimeSeedResult:
    """Out-of-fold predictions for one (regime, split seed) and their audit."""

    regime: str
    seed: int
    oof: pd.DataFrame
    pairs: pd.DataFrame
    penalties: pd.DataFrame
    attribution: pd.DataFrame
    fold_records: list[dict] = field(default_factory=list)
    integrity: dict = field(default_factory=dict)
    cell_audit: dict = field(default_factory=dict)
    fit_seconds: float = 0.0


def evaluate_regime_seed(
    data: LevelData,
    *,
    chemistry: ChemistryMap,
    regime: str,
    seed: int,
    folds: int,
    params: LevelForestParameters,
    base_min_cells: int,
    log: Callable[[str], None],
) -> RegimeSeedResult:
    """Fit every model on every fold of one (regime, split seed).

    One OOF row per cohort row, with one ``prediction_<MODEL>`` column per model,
    so "every model was scored on the same rows" is structural rather than
    something a later join has to preserve.
    """
    frame = data.frame
    n = len(frame)
    group_column = REGIME_GROUP_COLUMN[regime]
    splits = regime_folds(frame, regime=regime, seed=seed, folds=folds,
                          base_min_cells=base_min_cells)
    integrity = regime_integrity(frame, splits, regime=regime)
    if regime == PRIMARY_REGIME:
        # Reuse Experiment A's own tested integrity checker on the same objects.
        integrity["diversity_splits_integrity"] = assert_split_integrity(
            frame, diversity_splits(frame, group_column=group_column, n_splits=folds,
                                    seed=seed, base_min_cells=base_min_cells))
        integrity["ok"] = bool(integrity["ok"]
                               and integrity["diversity_splits_integrity"]["ok"])
    cell_audit = cells_never_split_across_folds(frame, splits)

    predictions = {name: np.full(n, np.nan) for name in MODEL_NAMES}
    extras = {name: np.full(n, np.nan) for name in
              ("stage_a", "stage_b", "stage_b_truecentre", "cell_true_mean", "cell_n_metals")}
    fold_of = np.full(n, -1, dtype=int)
    nn_train = np.full(n, np.nan)
    extractants = frame["extractant"].astype(str).to_numpy()
    superclusters = frame["tanimoto_cluster"].astype(str).to_numpy()
    row_ids = frame["row_id"].astype(str).to_numpy()

    fold_records: list[dict] = []
    penalty_rows: list[dict] = []
    attribution_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []
    started = time.time()

    for split in splits:
        fold_started = time.time()
        train = frame.iloc[split.train_index]
        test = frame.iloc[split.test_index]
        fold_of[split.test_index] = split.fold

        fold_params = LevelForestParameters(
            n_estimators=params.n_estimators, max_features=params.max_features,
            min_samples_leaf=params.min_samples_leaf,
            random_state=params.random_state + split.fold * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET,
            n_jobs=params.n_jobs, learner=params.learner)

        fold_predictions, fold_extras, record = fit_fold(
            data, train=train, test=test, params=fold_params, group_column=group_column,
            split_seed=int(seed))
        for name, values in fold_predictions.items():
            predictions[name][split.test_index] = values
        for name in extras:
            extras[name][split.test_index] = fold_extras[name].to_numpy(dtype=float)

        # --- distance of each test ligand to THIS fold's training chemistry -----
        # ``exclude_self=False`` on purpose: under ``unseen_series`` the test
        # ligand IS a training ligand, and its honest distance to training
        # chemistry is zero (similarity 1.0).  Excluding itself would report the
        # next-nearest ligand and make a seen ligand look like new chemistry.
        test_extractants = sorted(set(extractants[split.test_index]))
        train_extractants = sorted(set(extractants[split.train_index]))
        neighbours = chemistry.nearest_neighbour(test_extractants, train_extractants,
                                                 exclude_self=False)
        lookup = neighbours.set_index("extractant")["nn_tanimoto"]
        nn_train[split.test_index] = test["extractant"].astype(str).map(lookup).to_numpy(dtype=float)

        # --- derived pairs, inside this test fold ------------------------------
        scored_test = test.copy()
        for name, values in fold_predictions.items():
            scored_test[f"prediction_{name}"] = values
        prediction_columns = [f"prediction_{name}" for name in MODEL_NAMES]
        pairs = derived_pairs(scored_test, prediction_columns)
        if not pairs.empty:
            pairs = pairs.copy()
            pairs["pair_NULL"] = pair_label_mean_null(train, pairs)
            pairs.insert(0, "outer_fold", int(split.fold))
            pairs.insert(0, "split_seed", int(seed))
            pairs.insert(0, "regime", regime)
            pair_parts.append(pairs)

        for entry in record["penalties"]:
            penalty_rows.append({"regime": regime, "split_seed": int(seed),
                                 "fold": int(split.fold),
                                 "n_train_ligands": record["n_train_ligands"],
                                 "n_design_columns": record["n_design_columns"], **entry})
        part = record["attribution"].copy()
        part.insert(0, "fold", int(split.fold))
        part.insert(0, "split_seed", int(seed))
        part.insert(0, "regime", regime)
        attribution_parts.append(part)

        fold_records.append({
            "fold": int(split.fold),
            "split_seed": int(seed),
            "regime": regime,
            "group_column": group_column,
            "test_row_ids_sha256": sha256_json(sorted(row_ids[split.test_index].tolist())),
            "test_extractants": sorted(set(extractants[split.test_index])),
            "test_superclusters": sorted(set(superclusters[split.test_index])),
            "train_extractants_by_arm": {"ALL_MODELS": sorted(set(extractants[split.train_index]))},
            "train_superclusters_by_arm": {
                "ALL_MODELS": sorted(set(superclusters[split.train_index]))},
            "n_test_rows": int(split.test_index.size),
            "n_train_rows_by_arm": {"ALL_MODELS": int(split.train_index.size)},
            "held_out_groups": list(split.held_out_groups),
            "n_train_ligands": record["n_train_ligands"],
            "n_ligand_intercepts": record["n_ligand_intercepts"],
            "n_pair_difference_rows": record["n_pair_difference_rows"],
            "crossfit": record["crossfit"],
            "seconds": round(time.time() - fold_started, 2),
        })
        log(f"  {regime} seed {seed} fold {split.fold}: test {split.test_index.size} rows / "
            f"{len(test_extractants)} ligands, train {split.train_index.size} rows / "
            f"{record['n_train_ligands']} ligands; lambda_ligand="
            f"{record['penalties'][1]['lambda_ligand']}, "
            f"{record['n_pair_difference_rows']} pair rows, "
            f"{record['crossfit']['n_cells_multi_metal']}/{record['crossfit']['n_cells']} "
            f"multi-metal training cells, {time.time() - fold_started:.1f}s")

    unpredicted = {name: int(np.isnan(v).sum()) for name, v in predictions.items()}
    if any(unpredicted.values()):
        raise RuntimeError(f"rows left unpredicted (folds did not cover the cohort): {unpredicted}")

    oof = frame[list(LEVEL_IDENTITY_COLUMNS)].copy()
    oof.insert(0, "split_seed", int(seed))
    oof.insert(0, "regime", regime)
    oof["outer_fold"] = fold_of
    oof["nn_train_tanimoto"] = nn_train
    for name in ("cell_n_metals", "cell_true_mean", "stage_a", "stage_b", "stage_b_truecentre"):
        oof[name] = extras[name]
    oof["cell_n_metals"] = oof["cell_n_metals"].astype(int)
    for name in MODEL_NAMES:
        oof[f"prediction_{name}"] = predictions[name]

    return RegimeSeedResult(
        regime=regime, seed=int(seed), oof=oof,
        pairs=pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame(),
        penalties=pd.DataFrame(penalty_rows),
        attribution=pd.concat(attribution_parts, ignore_index=True) if attribution_parts
        else pd.DataFrame(),
        fold_records=fold_records, integrity=integrity, cell_audit=cell_audit,
        fit_seconds=time.time() - started)


# --------------------------------------------------------------------------- #
# The Experiment A reproduction check
# --------------------------------------------------------------------------- #

def assert_mono_et_matches_experiment_a(
    oof: pd.DataFrame,
    *,
    reference_path: Path,
    feature_set: str = REFERENCE_FEATURE_SET,
    reference_arm: str = REFERENCE_ARM,
    regime: str = PRIMARY_REGIME,
    tolerance: float = IDENTITY_TOLERANCE,
    pilot: bool = False,
) -> dict:
    """MONO_ET must equal Experiment A's EXPANDED arm, per row_id and per seed.

    Fail-closed: a missing file, a missing feature set, a seed with no reference
    rows, a row that is not in the reference, or any difference at or above
    ``tolerance`` all return ``ok = False``.  Under ``--pilot`` the check is
    SKIPPED and says so — 120 trees cannot reproduce a 400-tree forest, and
    recording that as a pass would be the exact false assurance this repo has
    been bitten by before.
    """
    block = oof[oof["regime"] == regime]
    if block.empty:
        return {"ok": True, "status": "NOT_APPLICABLE",
                "reason": f"regime {regime!r} was not run, so there is nothing to reproduce; "
                          f"this run therefore carries no reproduction evidence at all.",
                "n_rows_compared": 0}
    if pilot:
        return {"ok": True, "status": "SKIPPED",
                "reason": "--pilot fits 120 trees; Experiment A fitted 400, so the identity "
                          "cannot hold. SKIPPED, not passed: the check is only evidence in a "
                          "FULL run on the same machine.",
                "n_rows_compared": 0}
    if not Path(reference_path).exists():
        return {"ok": False, "status": "MISSING_REFERENCE",
                "reason": f"reference OOF {reference_path} does not exist", "n_rows_compared": 0}
    reference = pd.read_parquet(reference_path)
    column = f"prediction_{reference_arm}"
    if column not in reference.columns:
        return {"ok": False, "status": "MISSING_COLUMN",
                "reason": f"{reference_path} has no column {column!r}", "n_rows_compared": 0}
    reference = reference[reference["feature_set"].astype(str) == feature_set]
    if reference.empty:
        return {"ok": False, "status": "MISSING_FEATURE_SET",
                "reason": f"{reference_path} holds no rows for feature_set {feature_set!r}",
                "n_rows_compared": 0}

    per_seed: dict[str, dict] = {}
    ok = True
    worst = 0.0
    compared = 0
    for seed, ours in block.groupby("split_seed"):
        theirs = reference[reference["split_seed"] == seed]
        if theirs.empty:
            per_seed[str(seed)] = {"ok": False, "reason": "seed absent from the reference OOF"}
            ok = False
            continue
        merged = ours[["row_id", "prediction_MONO_ET"]].merge(
            theirs[["row_id", column]], on="row_id", how="left", validate="one_to_one")
        unmatched = int(merged[column].isna().sum())
        difference = (merged["prediction_MONO_ET"] - merged[column]).abs()
        max_abs = float(difference.max()) if len(difference) else float("nan")
        seed_ok = bool(unmatched == 0 and np.isfinite(max_abs) and max_abs < tolerance)
        per_seed[str(seed)] = {
            "ok": seed_ok, "n_rows": int(len(merged)), "n_unmatched_row_ids": unmatched,
            "max_abs_diff": max_abs,
            "n_rows_above_tolerance": int((difference >= tolerance).sum()),
        }
        ok = ok and seed_ok
        compared += int(len(merged))
        if np.isfinite(max_abs):
            worst = max(worst, max_abs)
    return {"ok": bool(ok), "status": "CHECKED" if ok else "MISMATCH",
            "reference_path": str(reference_path), "feature_set": feature_set,
            "reference_arm": reference_arm, "tolerance": float(tolerance),
            "max_abs_diff": float(worst), "n_rows_compared": int(compared), "per_seed": per_seed}


# --------------------------------------------------------------------------- #
# Pairs: the C4 identities
# --------------------------------------------------------------------------- #

def antisymmetry_residual(fold_pairs: pd.DataFrame, *, fold_rows: pd.DataFrame,
                          prediction_column: str,
                          target_column: str = LEVEL_TARGET_COLUMN) -> float:
    """max |ŷ(A,B) + ŷ(B,A)| recomputed from one fold's row predictions.

    ``hierarchical.pair_consistency`` reports ``antisymmetry_max`` as a literal
    0.0 — the variable is initialised and returned but never assigned inside its
    loop — so quoting it would make C4's antisymmetry half vacuous.  Here the
    reverse orientation is rebuilt from the same two row predictions and added to
    the forward one, which is what the hypothesis actually claims.

    ``row_a``/``row_b`` are *positions* inside the fold's test frame, so the
    alignment between the pair table and ``fold_rows`` is load-bearing and is
    verified against the pair's own truth before anything is measured.
    """
    if fold_pairs.empty:
        return 0.0
    a = fold_pairs["row_a"].to_numpy(dtype=int)
    b = fold_pairs["row_b"].to_numpy(dtype=int)
    y = fold_rows[target_column].to_numpy(dtype=float)
    if a.max() >= len(y) or b.max() >= len(y):
        raise RuntimeError("pair row positions fall outside the fold's OOF rows")
    truth_gap = float(np.max(np.abs((y[a] - y[b]) - fold_pairs["truth"].to_numpy(dtype=float))))
    if truth_gap > 1e-12:
        raise RuntimeError(
            f"pair row positions do not align with the fold's OOF rows "
            f"(truth mismatch {truth_gap:.3e}); the antisymmetry number would be meaningless")
    p = fold_rows[prediction_column].to_numpy(dtype=float)
    forward = fold_pairs[f"pair_{prediction_column}"].to_numpy(dtype=float)
    reverse = p[b] - p[a]
    return float(np.max(np.abs(forward + reverse)))


def pair_tables(pairs: pd.DataFrame, oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-(regime, seed, fold, model) pair metrics, and the C4 identity audit."""
    if pairs.empty:
        return pd.DataFrame(), {"ok": False, "reason": "no within-cell metal pairs were formed",
                                "max_antisymmetry": float("nan"),
                                "max_transitivity": float("nan"), "n_triples": 0, "n_pairs": 0}
    rows: list[dict] = []
    max_anti = 0.0
    max_trans = 0.0
    n_triples = 0
    columns = [f"prediction_{name}" for name in MODEL_NAMES]
    for (regime, seed, fold), block in pairs.groupby(["regime", "split_seed", "outer_fold"],
                                                     sort=True):
        fold_rows = oof[(oof["regime"] == regime) & (oof["split_seed"] == seed)
                        & (oof["outer_fold"] == fold)]
        for column in columns + ["NULL"]:
            pair_column = f"pair_{column}" if column != "NULL" else "pair_NULL"
            metrics = pair_metrics(block, pair_column)
            consistency = pair_consistency(block, pair_column)
            anti = (antisymmetry_residual(block, fold_rows=fold_rows, prediction_column=column)
                    if column != "NULL" else float("nan"))
            rows.append({"regime": regime, "split_seed": int(seed), "fold": int(fold),
                         "model": column.replace("prediction_", ""), **metrics,
                         "transitivity_max": consistency["transitivity_max"],
                         "n_triples": consistency["n_triples"],
                         "antisymmetry_max": anti})
            if column != "NULL":
                max_anti = max(max_anti, anti)
                max_trans = max(max_trans, float(consistency["transitivity_max"]))
                # Every model sees the same pairs, so counting the triples once per
                # fold — not once per (fold, model) — keeps ``n_triples`` a count of
                # triples rather than a count of checks.
                if column == columns[0]:
                    n_triples += int(consistency["n_triples"])
    audit = {
        "ok": bool(max_anti < IDENTITY_TOLERANCE and max_trans < IDENTITY_TOLERANCE
                   and n_triples > 0),
        "max_antisymmetry": float(max_anti), "max_transitivity": float(max_trans),
        "n_triples": int(n_triples), "n_pairs": int(len(pairs)),
        "n_identity_checks": int(n_triples * len(columns)),
        "tolerance": IDENTITY_TOLERANCE,
        "note": "antisymmetry is recomputed here from the row predictions; "
                "hierarchical.pair_consistency returns a hard-coded 0.0 for it",
    }
    return pd.DataFrame(rows), audit


# --------------------------------------------------------------------------- #
# Endpoints and contrasts
# --------------------------------------------------------------------------- #

def endpoint_subsets(predictions: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """``[("all", frame), ("multi_metal_cells", subset)]`` — the two populations.

    Membership depends only on ``cell_n_metals``, a property of the row's cell in
    the shared cohort, never of the model.
    """
    return [(ENDPOINT_ALL, predictions),
            (ENDPOINT_MULTI, predictions[predictions["cell_n_metals"] >= 2])]


def unit_blocks(predictions: pd.DataFrame, *, unit_column: str = "ecfp_cluster",
                block_column: str = "tanimoto_cluster") -> dict[str, str]:
    """ECFP cluster -> the Tanimoto super-cluster the chemotype folds held out.

    Scoring unit and independence unit differ (131 clusters nest inside 79
    super-clusters on this cohort); resampling the scoring unit would understate
    the interval width.  The same block map is used in every regime, because it
    describes the *chemistry*, not the fold algorithm.
    """
    pairs = predictions[[unit_column, block_column]].astype(str).drop_duplicates()
    return dict(zip(pairs[unit_column], pairs[block_column]))


def contrast_frame(per_unit: pd.DataFrame, *, block_of_unit: Mapping[str, str],
                   replicates: int, seed: int) -> pd.DataFrame:
    """Paired bootstrap for every contrast whose two models are present."""
    models = set(per_unit["arm"].unique())
    comparisons = {c.name: (c.reference, c.candidate) for c in CONTRASTS
                   if c.reference in models and c.candidate in models}
    if not comparisons:
        return pd.DataFrame()
    table = paired_unit_bootstrap(per_unit, comparisons, statistics=BOOTSTRAP_STATISTICS,
                                  block_of_unit=dict(block_of_unit), replicates=replicates,
                                  seed=seed)
    registry = {c.name: c for c in CONTRASTS}
    if not table.empty:
        table["preregistered"] = table["comparison"].map(lambda n: registry[n].preregistered)
        table["question"] = table["comparison"].map(lambda n: registry[n].question)
    return table


def pool_per_unit_over_seeds(per_unit_by_seed: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Average each unit's statistics over the split seeds, then bootstrap that.

    A per-seed interval answers "is this fold partition's effect real"; averaging
    the unit statistics over seeds first answers "is the effect real for this
    cohort", which is the question the generation is about.  Both are reported.
    """
    if not per_unit_by_seed:
        return pd.DataFrame()
    stacked = pd.concat(per_unit_by_seed, ignore_index=True)
    numeric = [c for c in stacked.columns if c not in ("unit", "arm")]
    pooled = stacked.groupby(["unit", "arm"], as_index=False)[numeric].mean()
    pooled["n_seeds"] = stacked.groupby(["unit", "arm"]).size().to_numpy()
    return pooled


POOLED_RENAME: Mapping[str, str] = {
    "point_delta": "pooled_point_delta", "ci95_low": "pooled_ci95_low",
    "ci95_high": "pooled_ci95_high", "p_worse_one_sided": "pooled_p_worse_one_sided",
    "units_improved": "pooled_units_improved", "units_total": "pooled_units_total",
    "bootstrap_blocks": "pooled_blocks", "bca_low": "pooled_bca_low",
    "bca_high": "pooled_bca_high", "cluster_robust_low": "pooled_cluster_robust_low",
    "cluster_robust_high": "pooled_cluster_robust_high",
    "block_macro_delta": "pooled_block_macro_delta",
}


# --------------------------------------------------------------------------- #
# Penalties
# --------------------------------------------------------------------------- #

def lambda_frequency_table(penalties: pd.DataFrame, *, model: str = "C1_HIER_RIDGE") -> pd.DataFrame:
    """How often the inner CV chose each ligand penalty, per regime.

    This is the single most informative diagnostic C1 produces: the ligand
    penalty is what decides whether a training ligand's intercept is estimated or
    shrunk away, and the inner CV chooses it under the *same* hold-out as the
    outer regime.  A regime that always lands on 1e6 has discovered, on its own,
    that only the descriptor prior transfers.
    """
    if penalties.empty:
        return pd.DataFrame()
    block = penalties[penalties["model"] == model].copy()
    if block.empty:
        return pd.DataFrame()
    block["lambda_ligand"] = block["lambda_ligand"].apply(
        lambda v: "none" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:g}")
    table = (block.groupby(["regime", "lambda_ligand"]).size().rename("n_folds").reset_index())
    totals = table.groupby("regime")["n_folds"].transform("sum")
    table["share_of_folds"] = table["n_folds"] / totals
    fixed = (block.groupby("regime")["lambda_fixed"].agg(["median", "min", "max"])
             .rename(columns={"median": "lambda_fixed_median", "min": "lambda_fixed_min",
                              "max": "lambda_fixed_max"}).reset_index())
    return table.merge(fixed, on="regime", how="left")


# --------------------------------------------------------------------------- #
# Hypothesis scoring
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Verdict:
    hypothesis: str
    statement: str
    pass_condition: str
    verdict: str
    evidence: str
    falsifier: str


def _seed_rule(seeds_positive: int, n_seeds: int) -> bool | None:
    """The house rule: ">= 4/5 split seeds in the same direction". None = not evaluable."""
    if n_seeds < 2:
        return None
    return seeds_positive >= math.ceil(0.8 * n_seeds)


def _verdict(passed: bool, contradicted: bool, *, seed_ok: bool | None) -> str:
    """PASS only on the pre-registered condition; FAIL only on evidence against it.

    A confidence interval that straddles zero is *not* evidence that the effect is
    absent, so it scores INCONCLUSIVE rather than FAIL.
    """
    if contradicted:
        return "FAIL"
    if passed and seed_ok is True:
        return "PASS"
    return "INCONCLUSIVE"


def _row(summary: pd.DataFrame, *, regime: str, endpoint: str, comparison: str,
         statistic: str) -> pd.Series | None:
    if summary is None or summary.empty:
        return None
    sub = summary[(summary["regime"] == regime) & (summary["endpoint"] == endpoint)
                  & (summary["comparison"] == comparison) & (summary["statistic"] == statistic)]
    return None if sub.empty else sub.iloc[0]


def _fmt(row: pd.Series | None, *, unit: str = "log units") -> str:
    """Every number with its interval, its n and its grouping — never bare."""
    if row is None:
        return "not evaluable (this regime/endpoint/contrast was not run)"
    parts = [f"Δ = {row['pooled_point_delta']:+.4f} {unit}",
             f"percentile CI95 [{row['pooled_ci95_low']:+.4f}, {row['pooled_ci95_high']:+.4f}]"]
    if "pooled_bca_low" in row.index and pd.notna(row.get("pooled_bca_low")):
        parts.append(f"**BCa CI95 [{row['pooled_bca_low']:+.4f}, {row['pooled_bca_high']:+.4f}]**")
    if "pooled_cluster_robust_low" in row.index and pd.notna(row.get("pooled_cluster_robust_low")):
        parts.append(f"cluster-robust [{row['pooled_cluster_robust_low']:+.4f}, "
                     f"{row['pooled_cluster_robust_high']:+.4f}]")
    if "pooled_block_macro_delta" in row.index and pd.notna(row.get("pooled_block_macro_delta")):
        parts.append(f"block-macro Δ {row['pooled_block_macro_delta']:+.4f}")
    tail = (f"{int(row['pooled_units_total'])} ECFP-cluster units in "
            f"{int(row['pooled_blocks'])} chemotype bootstrap blocks")
    if pd.notna(row.get("n_seeds")):
        tail += (f"; mean over seeds {row['mean_point_delta']:+.4f}, "
                 f"{int(row['seeds_positive'])}/{int(row['n_seeds'])} seeds positive")
    return "; ".join(parts) + f" ({tail})"


def _bca_low(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    value = row.get("pooled_bca_low")
    return float(value) if pd.notna(value) else None


def _bca_high(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    value = row.get("pooled_bca_high")
    return float(value) if pd.notna(value) else None


def score_hypotheses(
    summary: pd.DataFrame, *, pair_audit: Mapping[str, object], regimes: Sequence[str],
    n_seeds: int, margin: float = C3_MARGIN,
) -> list[Verdict]:
    """Score C1–C4 against their pre-registered pass conditions, and nothing else."""
    verdicts: list[Verdict] = []

    # ---- C1: which component fails to transfer ------------------------------
    decisive = _row(summary, regime=PRIMARY_REGIME, endpoint=ENDPOINT_MULTI,
                    comparison=DECISIVE_CONTRAST, statistic="mae")
    level = _row(summary, regime=PRIMARY_REGIME, endpoint=ENDPOINT_MULTI,
                 comparison="level_oracle_gain", statistic="mae")
    metal = _row(summary, regime=PRIMARY_REGIME, endpoint=ENDPOINT_MULTI,
                 comparison="metal_oracle_gain", statistic="mae")
    decisive_all = _row(summary, regime=PRIMARY_REGIME, endpoint=ENDPOINT_ALL,
                        comparison=DECISIVE_CONTRAST, statistic="mae")
    low, high = _bca_low(decisive), _bca_high(decisive)
    seed_ok = None if decisive is None else _seed_rule(int(decisive["seeds_positive"]),
                                                       int(decisive["n_seeds"]))
    verdicts.append(Verdict(
        "C1", "The component that fails to transfer to a new chemotype is the level, "
              "not the metal response.",
        f"under {PRIMARY_REGIME}, on test cells with cell_n_metals >= 2: "
        f"MAE(ORACLE_METAL) − MAE(ORACLE_LEVEL) > 0 with **BCa CI95 low > 0** over chemotype "
        f"blocks (this delta is exactly the level oracle's gain minus the metal oracle's, "
        f"because both are measured against the same C2_TWO_STAGE reference), plus the house "
        f"seed clause (>= 4/5 seeds positive; a single-seed pilot cannot evaluate it).",
        _verdict(decisive is not None and low is not None and low > 0
                 and decisive["pooled_point_delta"] > 0,
                 decisive is not None and high is not None and high <= 0, seed_ok=seed_ok),
        f"DECISIVE, multi-metal test cells — level gain minus metal gain: {_fmt(decisive)}. "
        f"Components: level oracle gain {_fmt(level)}; metal oracle gain {_fmt(metal)}. "
        f"All-rows version (reported, never the verdict): {_fmt(decisive_all)}",
        "If the metal oracle removes as much error as the level oracle on multi-metal cells "
        "(BCa CI95 covering or below zero), the level is NOT the component that fails, and "
        "Phase 1's attribution — on which this whole generation's plan rests — is wrong.",
    ))

    # ---- C2 (hypothesis): partial pooling helps exactly where the ligand is known
    series_row = _row(summary, regime="unseen_series", endpoint=ENDPOINT_ALL,
                      comparison="C1_HIER_RIDGE_vs_MONO_RIDGE", statistic="mae")
    chemo_row = _row(summary, regime=PRIMARY_REGIME, endpoint=ENDPOINT_ALL,
                     comparison="C1_HIER_RIDGE_vs_MONO_RIDGE", statistic="mae")
    series_low = _bca_low(series_row)
    series_high = _bca_high(series_row)
    spans_zero = (chemo_row is not None
                  and chemo_row["pooled_ci95_low"] <= 0 <= chemo_row["pooled_ci95_high"])
    both_ran = series_row is not None and chemo_row is not None
    seed_ok_2 = None if series_row is None else _seed_rule(int(series_row["seeds_positive"]),
                                                           int(series_row["n_seeds"]))
    verdicts.append(Verdict(
        "C2", "Partial pooling helps exactly where the ligand is known, and nowhere else.",
        "C1_HIER_RIDGE beats MONO_RIDGE under unseen_series (BCa CI95 low > 0) AND the two are "
        "indistinguishable under unseen_chemotype (CI95 spans 0). Both clauses are required.",
        _verdict(both_ran and series_low is not None and series_low > 0 and spans_zero,
                 both_ran and series_high is not None and series_high <= 0, seed_ok=seed_ok_2)
        if both_ran else "INCONCLUSIVE",
        (f"unseen_series (ligand IS in training, so the intercept is live): {_fmt(series_row)}. "
         f"unseen_chemotype (intercept is zero by construction): {_fmt(chemo_row)}; "
         f"CI95 spans zero there: {spans_zero}."
         + ("" if both_ran else
            f" NOT EVALUABLE: regimes run were {list(regimes)}; the hypothesis needs both "
            f"unseen_series and unseen_chemotype.")),
        "If the hierarchical ridge also beats plain ridge on a NEW chemotype, the intercepts are "
        "not what is helping — something else in the fit is — and the mixed-effects reading of "
        "C1 is wrong. If it fails to beat plain ridge even on a seen ligand, the partial-pooling "
        "machinery is buying nothing anywhere.",
    ))

    # ---- C3: structure does not substitute for coverage ---------------------
    c3_row = _row(summary, regime=PRIMARY_REGIME, endpoint=ENDPOINT_ALL,
                  comparison="C2_TWO_STAGE_vs_MONO_ET", statistic="mae")
    c3_true = _row(summary, regime=PRIMARY_REGIME, endpoint=ENDPOINT_ALL,
                   comparison="C2_TRUECENTRE_vs_MONO_ET", statistic="mae")
    c3_low = _bca_low(c3_row)
    beats = (c3_row is not None and c3_low is not None and c3_low > 0
             and c3_row["pooled_point_delta"] > margin)
    true_low = _bca_low(c3_true)
    true_beats = (c3_true is not None and true_low is not None and true_low > 0
                  and c3_true["pooled_point_delta"] > margin)
    # C3 is a NEGATIVE hypothesis ("no beat"), so a count of positive seeds is the
    # wrong instrument in either direction: a 0/5 no-beat must not read as
    # INCONCLUSIVE (the first draft did that), and a 5/5 set of sub-margin beats is
    # still "no beat".  The pre-registered condition is a property of the pooled
    # interval; the only seed-related requirement kept is that a single-seed pilot
    # cannot PASS, because one partition cannot support a claim of absence.
    seed_ok_3 = None if c3_row is None else int(c3_row["n_seeds"]) >= 2
    verdicts.append(Verdict(
        "C3", "Model structure does not substitute for chemical coverage.",
        f"C2_TWO_STAGE does NOT beat MONO_ET under {PRIMARY_REGIME} by more than {margin:g} "
        f"macro MAE with BCa CI95 low > 0. PASS = no such beat over >= 2 split seeds; FAIL = the "
        f"beat happened, which is the surprise worth reporting. A single-seed pilot scores "
        f"INCONCLUSIVE because one partition cannot support a claim of absence.",
        _verdict(c3_row is not None and not beats, beats, seed_ok=seed_ok_3),
        (f"C2_TWO_STAGE vs MONO_ET, all test rows: {_fmt(c3_row)}; beat by more than "
         f"{margin:g} with BCa low > 0: {beats}. "
         f"EXPLORATORY (amendment; the variant was chosen after a one-seed preview, so this line "
         f"is exploratory-confirmed at best and is NOT part of the verdict) — "
         f"C2_TRUECENTRE vs MONO_ET: {_fmt(c3_true)}; beat: {true_beats}."),
        "A C2 that beats MONO_ET materially on new chemistry falsifies the generation's reading: "
        "it would say model structure, not coverage, was the lever, contradicting Phase 1's "
        "case A.",
    ))

    # ---- C4: the derived pair identities ------------------------------------
    anti = float(pair_audit.get("max_antisymmetry", float("nan")))
    trans = float(pair_audit.get("max_transitivity", float("nan")))
    n_triples = int(pair_audit.get("n_triples", 0))
    exact = bool(pair_audit.get("ok", False))
    verdicts.append(Verdict(
        "C4", "Level-derived pair predictions keep exact antisymmetry and transitivity.",
        f"max |ŷ(A,B) + ŷ(B,A)| and max |ŷ(A,B) + ŷ(B,C) − ŷ(A,C)| below {IDENTITY_TOLERANCE:g} "
        f"on every test fold of every regime and seed, with at least one triple actually "
        f"present (an empty pair table would satisfy the bound vacuously). This is an algebraic "
        f"identity, so no seed clause applies.",
        "PASS" if exact else "FAIL",
        f"max antisymmetry residual {anti:.3e} (recomputed in this runner from the row "
        f"predictions, because hierarchical.pair_consistency returns a hard-coded 0.0 for it); "
        f"max transitivity residual {trans:.3e} over {n_triples} within-cell triples and "
        f"{int(pair_audit.get('n_pairs', 0))} pairs, checked for each of the eight models "
        f"({int(pair_audit.get('n_identity_checks', 0))} triple checks in total).",
        "Any residual at or above 1e-9 means the pair predictions were not derived as a "
        "difference of two level predictions somewhere in the pipeline — which would silently "
        "reintroduce the orientation bug the gen3/gen4 pair models were built to avoid.",
    ))
    return verdicts


# --------------------------------------------------------------------------- #
# Checks the run must satisfy before it may claim success
# --------------------------------------------------------------------------- #

def build_checks(
    *, results: Sequence[RegimeSeedResult], oof: pd.DataFrame, reproduction: Mapping[str, object],
    pair_audit: Mapping[str, object], cohort_extractants: set[str],
    chemistry_extractants: set[str], pilot: bool,
) -> dict:
    """Assertions on the written artifacts, evaluated before ``validate_run``.

    ``validate_run`` is fail-closed on the result, so a broken run produces
    ``_FAILED.json`` and a non-zero exit rather than a plausible-looking report.
    """
    checks: dict = {
        "regime_split_integrity": {
            "ok": bool(all(r.integrity.get("ok", False) for r in results)),
            "per_run": [{"regime": r.regime, "split_seed": r.seed, "ok": r.integrity.get("ok"),
                         "all_rows_tested_once": r.integrity.get("all_rows_tested_once")}
                        for r in results]},
        "cells_never_split_across_folds": {
            "ok": bool(all(r.cell_audit.get("ok", False) for r in results)),
            "per_run": [{"regime": r.regime, "split_seed": r.seed, **r.cell_audit}
                        for r in results]},
        "crossfit_provenance": {
            "ok": bool(all(record["crossfit"]["ok"] for r in results
                           for record in r.fold_records)),
            "note": "every Stage B training residual came from a Stage A model fitted without "
                    "that row's own cell, and no group straddles two inner folds"},
        "no_missing_predictions": {
            "ok": bool(all(np.isfinite(oof[f"prediction_{m}"].to_numpy(dtype=float)).all()
                           for m in MODEL_NAMES)),
            "n_non_finite": {m: int((~np.isfinite(oof[f"prediction_{m}"].to_numpy(dtype=float)))
                                    .sum()) for m in MODEL_NAMES}},
        "similarity_column_complete": {
            "ok": bool(oof["nn_train_tanimoto"].notna().all()),
            "n_missing": int(oof["nn_train_tanimoto"].isna().sum()),
            "note": "a NaN here would be silently dropped from every hard-chemistry endpoint "
                    "INCLUDING 'all', because NaN < inf is False"},
        "chemistry_map_covers_cohort": {
            "ok": bool(set(cohort_extractants) <= set(chemistry_extractants))},
        "pair_identities_exact": dict(pair_audit),
    }
    # The check's KEY says what happened, so nobody reading validation.json can
    # mistake a skipped reproduction for a passed one.  A genuine failure always
    # lands under the plain name, where it fails the run.
    status = str(reproduction.get("status", "")).upper()
    key = {"SKIPPED": "mono_et_reproduction_SKIPPED_in_pilot",
           "NOT_APPLICABLE": "mono_et_reproduction_NOT_APPLICABLE_regime_not_run",
           }.get(status if reproduction.get("ok") else "", "mono_et_reproduces_experiment_a")
    checks[key] = {**dict(reproduction), "pilot": bool(pilot)}
    return checks


def finalise_run(
    output_dir: Path, *, manifest: RunManifest, checks: Mapping[str, object],
    required_artifacts: Sequence[str] = REQUIRED_ARTIFACTS,
) -> tuple[dict, Path | None]:
    """Write the manifest, validate the directory, and mark success only if it validated."""
    payload = manifest.write(output_dir)
    validation = validate_run(output_dir, manifest=payload,
                              required_artifacts=list(required_artifacts), checks=checks)
    success = write_success(output_dir, manifest=payload, validation=validation)
    manifest.write(output_dir)
    return validation, success


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _table(frame: pd.DataFrame, columns: Sequence[str] | None = None, digits: int = 4) -> str:
    if frame is None or frame.empty:
        return "```\n(no rows)\n```"
    view = frame if columns is None else frame[[c for c in columns if c in frame.columns]]
    return "```\n" + view.round(digits).to_string(index=False) + "\n```"


def render_report(
    *, stamp: str, args: argparse.Namespace, audit: dict, arm_metrics: pd.DataFrame,
    hard_metrics: pd.DataFrame, contrast_summary: pd.DataFrame, per_seed: pd.DataFrame,
    verdicts: Sequence[Verdict], lambda_table: pd.DataFrame,
    attribution: pd.DataFrame, pair_summary: pd.DataFrame, pair_audit: Mapping[str, object],
    reproduction: Mapping[str, object], endpoint_counts: pd.DataFrame, fit_seconds: float,
) -> str:
    """The decision report, written for a chemist who expects to be misled."""
    lines: list[str] = []
    A = lines.append
    A("# gen6 Experiment C — which component of the level fails to transfer?")
    A("")
    A(f"**Run** `gen6_hierarchical_{stamp}`"
      + ("  ·  **PILOT** (one seed, one regime, reduced trees — a smoke test, not evidence)"
         if args.pilot else ""))
    A("")
    A("## What was actually done")
    A("")
    A(f"* One shared cohort, built once at `min_cells = {args.eval_min_cells}`: "
      f"**{audit['rows']} rows, {audit['extractants']} extractants, {audit['ecfp_clusters']} ECFP "
      f"clusters, {audit['tanimoto_clusters']} Tanimoto-0.7 chemotypes**, {audit['conditions']} "
      f"conditions, {audit['series']} series, {audit['metals']} metals; target sd "
      f"{audit['target_sd']:.3f} log units.")
    A(f"* Regimes: {', '.join(args.regimes)} — held out on "
      f"{ {r: REGIME_GROUP_COLUMN[r] for r in args.regimes} }. "
      f"{args.folds} folds × {len(args.split_seeds)} split seeds {list(args.split_seeds)}.")
    A(f"* Eight models, all fold-local, all scored on byte-identical test rows: "
      f"{', '.join(MODEL_NAMES)}.")
    A(f"* Forest models: ExtraTrees, {args.n_estimators} trees, max_features "
      f"{args.max_features}, min_samples_leaf {args.min_samples_leaf}, fold seed "
      f"`model_seed + fold*{FOLD_SEED_STRIDE} + {FOLD_SEED_OFFSET}` (the gen5 formula, reused so "
      f"MONO_ET reproduces Experiment A rather than approximating it). Sample weights: "
      f"`group_balanced_weights` on the training rows' ECFP cluster, in every model.")
    A(f"* Ridge penalties by inner grouped CV, 3 folds grouped on **the regime's own column** "
      f"(not always chemotype): λ_fixed ∈ {list(FIXED_PENALTY_GRID)}, "
      f"λ_ligand ∈ {list(LIGAND_PENALTY_GRID)}. The linear design is the compact one "
      f"({', '.join(RIDGE_BASE_BLOCKS)} + the declared interaction block); LIG2D_EXT is "
      f"deliberately **not** in it, so MONO_RIDGE/C1/C3 are a family of their own and are not "
      f"a like-for-like comparison against MONO_ET.")
    A(f"* Fitting took {fit_seconds:.0f} s.")
    A("")
    A("### Does MONO_ET reproduce Experiment A's EXPANDED arm?")
    A("")
    status = str(reproduction.get("status"))
    if status == "SKIPPED":
        A(f"**SKIPPED** — {reproduction.get('reason')}")
    elif status == "NOT_APPLICABLE":
        A(f"**NOT APPLICABLE** — {reproduction.get('reason')}")
    elif reproduction.get("ok"):
        A(f"**YES.** Max absolute difference `{reproduction.get('max_abs_diff'):.3e}` over "
          f"{reproduction.get('n_rows_compared')} row_ids, per seed, against "
          f"`{reproduction.get('reference_path')}` "
          f"(feature_set `{reproduction.get('feature_set')}`, arm "
          f"`{reproduction.get('reference_arm')}`), tolerance "
          f"{reproduction.get('tolerance')}. Per seed: "
          f"{json.dumps(reproduction.get('per_seed'), default=str)}")
    else:
        A(f"**NO — the run is void.** {reproduction.get('reason', '')} "
          f"status `{status}`, max abs diff `{reproduction.get('max_abs_diff')}`, per seed "
          f"{json.dumps(reproduction.get('per_seed'), default=str)}")
    A("")
    A("## Leaderboard — one row per model per regime (mean over seeds)")
    A("")
    A("Macro MAE is primary: one ECFP cluster, one vote. `ORACLE_LEVEL` and `ORACLE_METAL` read "
      "test labels and are **not models** — they are there to be subtracted from each other. "
      "Pooled numbers are printed and never used for selection.")
    A("")
    A(_table(arm_metrics, ["regime", "arm", "macro_mae", "offset_mae", "shape_mae", "shape_r2",
                           "pooled_mae", "median_ligand_mae", "worst_quartile_ligand_mae",
                           "frac_within_1_log", "n_ligands", "n_ecfp_clusters", "n_superclusters",
                           "n_macro_units", "n_eff_pooled_rows"]))
    A("")
    A("## Hypothesis C1 — the attribution")
    A("")
    A("`point_delta = statistic(reference) − statistic(candidate)`, so **positive = the candidate "
      "is better**. The two oracle gains share the same `C2_TWO_STAGE` reference, so their "
      "difference is measured directly as `level_minus_metal_gain = "
      "(ORACLE_METAL, ORACLE_LEVEL)` and the reference cancels exactly instead of being "
      "subtracted by hand from two intervals. The bootstrap scores ECFP clusters and resamples "
      "**Tanimoto chemotypes** — the blocks the chemotype folds actually held out.")
    A("")
    A("Rows counted per endpoint (this is the n behind every interval below):")
    A("")
    A(_table(endpoint_counts, digits=1))
    A("")
    oracle = contrast_summary[contrast_summary["comparison"].isin(
        ("level_oracle_gain", "metal_oracle_gain", DECISIVE_CONTRAST))] \
        if not contrast_summary.empty else contrast_summary
    A(_table(oracle, ["regime", "endpoint", "comparison", "statistic", "pooled_point_delta",
                      "pooled_bca_low", "pooled_bca_high", "pooled_ci95_low", "pooled_ci95_high",
                      "pooled_cluster_robust_low", "pooled_cluster_robust_high",
                      "pooled_block_macro_delta", "mean_point_delta", "seeds_positive", "n_seeds",
                      "pooled_units_total", "pooled_blocks"]))
    A("")
    A("## Every contrast, every regime, every endpoint")
    A("")
    A(_table(contrast_summary, ["regime", "endpoint", "comparison", "statistic",
                                "pooled_point_delta", "pooled_bca_low", "pooled_bca_high",
                                "pooled_ci95_low", "pooled_ci95_high", "mean_point_delta",
                                "seeds_positive", "n_seeds", "pooled_units_total",
                                "preregistered"]))
    A("")
    A("Per-seed intervals (the same contrasts, one split partition at a time) are in "
      "`contrasts.csv`; the primary-regime `mae` rows are here:")
    A("")
    A(_table(per_seed, ["regime", "endpoint", "split_seed", "comparison", "point_delta",
                        "bca_low", "bca_high", "units_improved", "units_total"]))
    A("")
    A("## What the inner CV chose for the ligand penalty")
    A("")
    A("λ_ligand is the whole mechanism of C1: small = each training ligand gets its own estimated "
      "intercept, `1e6` = the intercepts are shrunk to nothing and the model falls back on the "
      "descriptor prior, which is all a held-out ligand can ever get. The inner CV is grouped on "
      "the **regime's own column**, so this table is the model discovering, per regime, whether "
      "ligand intercepts transfer.")
    A("")
    A(_table(lambda_table, digits=3))
    A("")
    A("## Where C1's prediction comes from (mean |contribution| per block, test rows)")
    A("")
    A(_table(attribution, digits=4))
    A("")
    A("## Derived pair predictions — the C4 measurement")
    A("")
    A("Pair predictions are formed **only** as `ŷ(A) − ŷ(B)` with A the lighter metal, inside a "
      "test cell. `NULL` is the leave-fold-out pair-label mean, the strongest honest pair "
      "alternative established in gen3/gen4. No verdict depends on these numbers except C4.")
    A("")
    A("Two rows will look wrong and are not. `ORACLE_METAL` scores **exactly zero** pair error: "
      "it is `Â(cell) + (y − ȳ_cell)`, and Â is constant inside a cell, so the difference of two "
      "of its rows is `y_A − y_B` by construction — a structural check that the oracle is the "
      "swap it claims to be, not a model that solved the pair task. For the same reason "
      "`ORACLE_LEVEL` and `C2_TRUECENTRE` are identical here: they differ only by a per-cell "
      "constant, which cancels in every pair.")
    A("")
    A(_table(pair_summary, ["regime", "model", "n_pairs", "pair_macro_mae", "pair_pooled_mae",
                            "pair_sign_accuracy", "antisymmetry_max", "transitivity_max",
                            "n_triples"], digits=6))
    A("")
    A(f"Identity residuals over every fold (the two columns above are printed in full here "
      f"because the table rounds them to zero): antisymmetry "
      f"{float(pair_audit.get('max_antisymmetry', float('nan'))):.3e}, transitivity "
      f"{float(pair_audit.get('max_transitivity', float('nan'))):.3e}, over "
      f"{int(pair_audit.get('n_triples', 0))} within-cell triples and "
      f"{int(pair_audit.get('n_pairs', 0))} pairs — one check per model, "
      f"{int(pair_audit.get('n_identity_checks', 0))} in total. `NULL` is a pair-label mean and "
      f"is NOT derived as a difference of two level predictions, so its transitivity residual is "
      f"large by construction; that is what makes the zeros above non-vacuous.")
    A("")
    A("## Hard chemistry — descriptive, not a protected endpoint here")
    A("")
    A("Bins are cut on `nn_train_tanimoto`: the maximum Tanimoto from the test ligand to the "
      "**training ligands of its own fold**, computed with `exclude_self=False`. Under "
      "`unseen_series` the test ligand *is* in training, so its distance is 1.0 by construction "
      "and the hard bins are empty — that is the correct answer, not a bug.")
    A("")
    A(_table(hard_metrics, ["regime", "endpoint", "arm", "n_rows", "n_ligands", "n_ecfp_clusters",
                            "macro_mae", "offset_mae", "shape_mae", "frac_within_1_log"]))
    A("")
    A("## Pre-registered hypotheses (protocol Experiment C)")
    A("")
    A("Scoring rule, stated so a verdict cannot be over-read: **PASS** = the pre-registered "
      "condition is met, including the seed-agreement clause; **FAIL** = the interval excludes "
      "the predicted direction (evidence *against*); **INCONCLUSIVE** = the interval straddles "
      "zero, or the seed clause cannot be evaluated (a single-seed pilot). An INCONCLUSIVE is "
      "not a weak PASS.")
    for v in verdicts:
        A("")
        A(f"**{v.hypothesis} — {v.verdict}**  ·  {v.statement}")
        A("")
        A(f"* pass condition: {v.pass_condition}")
        A(f"* measured: {v.evidence}")
        A(f"* what would falsify this: {v.falsifier}")
    A("")
    A("## How to break this result")
    A("")
    A("* **The oracles are not models and the gains are not achievable errors.** Both read test "
      "labels. C1 compares two *attributions*; it says nothing about how much error a deployable "
      "model could remove. Anyone quoting `ORACLE_LEVEL`'s MAE as an achievable number has "
      "misread the table.")
    A("* **The two oracles are not symmetric halves of one model.** `ORACLE_METAL` = Â + true "
      "departure uses Stage A's *predicted* level; `ORACLE_LEVEL` = true cell mean + B̂ uses the "
      "**truecentre** Stage B. So the decisive delta is `Stage A's level error − truecentre "
      "Stage B's departure error`, which is the quantity C1 is about, but it is not "
      "'the same model with one input replaced' in both directions. Re-run with `ORACLE_LEVEL` "
      "built from the cross-fitted Stage B to see whether the sign survives.")
    A("* **Multi-metal cells are a different test population.** Only 521 of 2,405 cells hold two "
      "or more metals. The C1 endpoint restricts to them because a singleton cell makes the "
      "level oracle trivially informative — but that restriction also changes which ligands and "
      "conditions are being scored. The all-rows row is printed beside every C1 number for that "
      "reason.")
    A("* **The ridge family is deliberately low-capacity.** MONO_RIDGE, C1 and C3 share a compact "
      "design without the 206 LIG2D_EXT columns. A C1-beats-MONO_RIDGE result is a statement "
      "about partial pooling inside that family, never a statement that C1 is competitive with "
      "the forest.")
    A("* **Seeds are not replicates.** The split seeds re-partition the same ligands. The paired "
      "bootstrap over chemotype blocks is the load-bearing statistic; seed agreement is a second "
      "requirement, not independent evidence.")
    A("* **Multiplicity.** C1–C4 are the only protected claims. The tables report many more "
      "contrasts (comparisons × statistics × endpoints × regimes), none corrected. Treat every "
      "interval outside C1–C4 as descriptive — and `C2_TRUECENTRE` as exploratory even where it "
      "appears inside C3, because the variant was chosen after a one-seed preview.")
    A("* **Do not quote the percentile interval alone.** One chemotype holds ~21 % of the scoring "
      "units on this cohort; the percentile interval's real one-sided Type-I rate was measured at "
      "~12.7 % at the `all` endpoint. BCa is the interval the verdicts use, and the "
      "cluster-robust and block-macro columns are printed beside it.")
    A("* **`unseen_series` is the easy regime.** The ligand is in training and every test row's "
      "nearest training neighbour is itself. A model that wins there has learned nothing about "
      "new chemistry.")
    A("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else \
        REPO_ROOT / "runs" / f"gen6_hierarchical_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    if args.pilot:
        args.split_seeds = list(args.split_seeds)[:1]
        args.n_estimators = min(args.n_estimators, 120)
        if list(args.regimes) == list(REGIMES):
            args.regimes = [PRIMARY_REGIME]
        log(f"PILOT: one split seed, {args.n_estimators} trees, regimes {list(args.regimes)} — a "
            f"smoke test, not evidence. The MONO_ET reproduction check is SKIPPED (120 trees "
            f"cannot equal Experiment A's 400).")

    # --- data -----------------------------------------------------------------
    source = pd.read_parquet(args.dataset)
    descriptors = None
    if args.descriptors and str(args.descriptors) and Path(args.descriptors).exists():
        descriptors = pd.read_parquet(args.descriptors)
    elif args.descriptors and str(args.descriptors):
        log(f"WARNING: descriptor parquet not found at {args.descriptors}; LIG2D_EXT is absent "
            f"and MONO_ET cannot reproduce Experiment A")

    data = build_level_dataset(
        source, min_rows_per_extractant=args.eval_min_cells,
        replicate_policy=args.replicate_policy, drop_below_log_d=args.log_d_floor,
        ligand_descriptors=descriptors)
    frame = data.frame
    audit = data.audit
    log(f"shared cohort: {audit['rows']} rows, {audit['extractants']} extractants, "
        f"{audit['ecfp_clusters']} ECFP clusters, {audit['tanimoto_clusters']} chemotypes, "
        f"{audit['series']} series; target sd {audit['target_sd']:.3f}")

    cells = frame.groupby(["extractant", "condition_id"], sort=False)[LEVEL_TARGET_COLUMN].size()
    log(f"cells: {len(cells)} (ligand, condition) cells, {int((cells >= 2).sum())} with >= 2 "
        f"metals covering {int(cells[cells >= 2].sum())} rows — the only rows that carry metal "
        f"information, and the population hypothesis C1 is scored on")

    # --- frozen chemistry -----------------------------------------------------
    if args.chemistry_map and Path(args.chemistry_map).exists():
        chemistry = ChemistryMap.from_parquet(args.chemistry_map)
        chemistry_source = str(args.chemistry_map)
        log(f"frozen chemistry map loaded from {args.chemistry_map} ({len(chemistry)} extractants)")
    else:
        chemistry = build_chemistry_map(source, ligand_descriptors=descriptors)
        chemistry_source = "rebuilt from the source table in this run"
        log(f"chemistry map built over all {len(chemistry)} extractants: "
            f"{chemistry.audit['n_ecfp_clusters']} ECFP clusters, "
            f"{chemistry.audit['n_superclusters']} chemotypes, rdkit="
            f"{chemistry.audit['rdkit_available']}")

    params = LevelForestParameters(
        n_estimators=args.n_estimators, max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf, random_state=args.model_seed, n_jobs=args.n_jobs)
    feature_sets = {"FOREST_" + "_".join(FOREST_BLOCKS): data.block_columns(FOREST_BLOCKS)}

    # --- fit ------------------------------------------------------------------
    started = time.time()
    results: list[RegimeSeedResult] = []
    for regime in args.regimes:
        for seed in args.split_seeds:
            log(f"regime {regime} seed {seed} (group column "
                f"{REGIME_GROUP_COLUMN[regime]!r}) …")
            result = evaluate_regime_seed(
                data, chemistry=chemistry, regime=regime, seed=int(seed), folds=args.folds,
                params=params, base_min_cells=args.base_min_cells, log=log)
            log(f"  done in {result.fit_seconds:.1f}s; integrity ok="
                f"{result.integrity['ok']}, cells intact={result.cell_audit['ok']}")
            results.append(result)
    fit_seconds = time.time() - started
    oof = pd.concat([r.oof for r in results], ignore_index=True)
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    log(f"fitting done in {fit_seconds:.1f}s; OOF {len(oof)} rows × {len(MODEL_NAMES)} models")

    # --- the reproduction check ------------------------------------------------
    reproduction = assert_mono_et_matches_experiment_a(
        oof, reference_path=Path(args.reference_oof), feature_set=args.reference_feature_set,
        pilot=bool(args.pilot))
    log(f"MONO_ET vs Experiment A: status={reproduction['status']} ok={reproduction['ok']} "
        f"max_abs_diff={reproduction.get('max_abs_diff')} "
        f"n_rows={reproduction.get('n_rows_compared')}")

    # --- pairs -----------------------------------------------------------------
    pairs = pd.concat([r.pairs for r in results if not r.pairs.empty], ignore_index=True) \
        if any(not r.pairs.empty for r in results) else pd.DataFrame()
    pair_metrics_table, pair_audit = pair_tables(pairs, oof)
    log(f"pairs: {pair_audit['n_pairs']} within-cell metal pairs, "
        f"{pair_audit['n_triples']} triples; antisymmetry {pair_audit['max_antisymmetry']:.3e}, "
        f"transitivity {pair_audit['max_transitivity']:.3e}")

    # --- metrics ---------------------------------------------------------------
    arm_parts, ligand_parts, hard_parts, contrast_parts = [], [], [], []
    endpoint_rows: list[dict] = []
    pooled_inputs: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for result in results:
        predictions = result.oof
        overall, per_ligand, hard = gen6_metric_table(
            predictions, list(MODEL_NAMES), similarity_column="nn_train_tanimoto",
            thresholds=tuple(args.thresholds))
        for table in (overall, per_ligand, hard):
            table.insert(0, "split_seed", result.seed)
            table.insert(0, "regime", result.regime)
        arm_parts.append(overall)
        ligand_parts.append(per_ligand)
        hard_parts.append(hard)

        blocks = unit_blocks(predictions)
        for endpoint, subset in endpoint_subsets(predictions):
            n_units = int(subset["ecfp_cluster"].nunique()) if len(subset) else 0
            endpoint_rows.append({
                "regime": result.regime, "split_seed": result.seed, "endpoint": endpoint,
                "n_rows": int(len(subset)),
                "n_ligands": int(subset["extractant"].nunique()) if len(subset) else 0,
                "n_cells": int(subset.groupby(["extractant", "condition_id"]).ngroups)
                if len(subset) else 0,
                "n_ecfp_clusters": n_units,
                "n_chemotype_blocks": int(subset["tanimoto_cluster"].nunique()) if len(subset) else 0,
            })
            if n_units < 2:
                log(f"  {result.regime} seed {result.seed} endpoint {endpoint}: {n_units} "
                    f"scoring unit(s) — bootstrap skipped, not silently pooled")
                continue
            per_unit = per_unit_statistics(subset, list(MODEL_NAMES))
            pooled_inputs.setdefault((result.regime, endpoint), []).append(per_unit)
            table = contrast_frame(per_unit, block_of_unit=blocks, replicates=args.replicates,
                                   seed=args.bootstrap_seed)
            if not table.empty:
                table.insert(0, "endpoint", endpoint)
                table.insert(0, "split_seed", result.seed)
                table.insert(0, "regime", result.regime)
                contrast_parts.append(table)

    arm_metrics = pd.concat(arm_parts, ignore_index=True)
    per_ligand_metrics = pd.concat(ligand_parts, ignore_index=True)
    hard_metrics = pd.concat(hard_parts, ignore_index=True)
    contrasts = pd.concat(contrast_parts, ignore_index=True) if contrast_parts else pd.DataFrame()
    endpoint_counts = pd.DataFrame(endpoint_rows)

    # --- pooled-over-seeds bootstrap and the contrast summary ------------------
    summary_rows: list[pd.DataFrame] = []
    for (regime, endpoint), parts in pooled_inputs.items():
        pooled = pool_per_unit_over_seeds(parts)
        blocks = unit_blocks(oof[oof["regime"] == regime])
        table = contrast_frame(pooled, block_of_unit=blocks, replicates=args.replicates,
                               seed=args.bootstrap_seed)
        if table.empty:
            continue
        table = table.rename(columns=dict(POOLED_RENAME))
        table.insert(0, "endpoint", endpoint)
        table.insert(0, "regime", regime)
        summary_rows.append(table)
    contrast_summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

    if not contrast_summary.empty and not contrasts.empty:
        per_seed_agg = (contrasts.groupby(["regime", "endpoint", "comparison", "statistic"])
                        .agg(mean_point_delta=("point_delta", "mean"),
                             min_point_delta=("point_delta", "min"),
                             max_point_delta=("point_delta", "max"),
                             seeds_positive=("point_delta", lambda s: int((s > 0).sum())),
                             n_seeds=("point_delta", "size")).reset_index())
        contrast_summary = contrast_summary.merge(
            per_seed_agg, on=["regime", "endpoint", "comparison", "statistic"], how="left")
    for column in ("regime", "endpoint", "comparison", "statistic", "pooled_point_delta",
                   "pooled_ci95_low", "pooled_ci95_high", "pooled_bca_low", "pooled_bca_high",
                   "mean_point_delta", "seeds_positive", "n_seeds", "pooled_units_total",
                   "pooled_blocks"):
        if column not in contrast_summary.columns:
            contrast_summary[column] = pd.Series(dtype=float)

    # --- penalties, attribution -------------------------------------------------
    penalties = pd.concat([r.penalties for r in results if not r.penalties.empty],
                          ignore_index=True) if results else pd.DataFrame()
    lambda_table = lambda_frequency_table(penalties)
    attribution = pd.concat([r.attribution for r in results if not r.attribution.empty],
                            ignore_index=True) if results else pd.DataFrame()
    attribution_summary = (attribution.groupby(["regime", "component"], as_index=False)
                           .agg(mean_abs_contribution=("mean_abs_contribution", "mean"),
                                mean_contribution=("mean_contribution", "mean"),
                                n_folds=("fold", "size"))
                           .sort_values(["regime", "mean_abs_contribution"], ascending=[True, False],
                                        ignore_index=True)) if not attribution.empty \
        else pd.DataFrame()

    # --- hypotheses --------------------------------------------------------------
    verdicts = score_hypotheses(contrast_summary, pair_audit=pair_audit, regimes=args.regimes,
                                n_seeds=len(args.split_seeds))

    # --- write the tables ---------------------------------------------------------
    arm_metrics.to_csv(output_dir / "arm_metrics.csv", index=False)
    per_ligand_metrics.to_csv(output_dir / "per_ligand_metrics.csv", index=False)
    hard_metrics.to_csv(output_dir / "hard_chemistry_metrics.csv", index=False)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    contrast_summary.to_csv(output_dir / "contrast_summary.csv", index=False)
    endpoint_counts.to_csv(output_dir / "endpoint_counts.csv", index=False)
    penalties.to_csv(output_dir / "penalties.csv", index=False)
    attribution.to_csv(output_dir / "component_attribution.csv", index=False)
    pair_metrics_table.to_csv(output_dir / "pair_metrics.csv", index=False)
    if pairs.empty:
        pd.DataFrame(columns=["regime", "split_seed", "outer_fold", "cell"]).to_parquet(
            output_dir / "pair_predictions.parquet", index=False)
    else:
        pairs.to_parquet(output_dir / "pair_predictions.parquet", index=False)
    if not lambda_table.empty:
        lambda_table.to_csv(output_dir / "lambda_frequency.csv", index=False)

    fold_records = [record for result in results for record in result.fold_records]
    split_definition = {
        "shared_cohort_min_cells": int(args.eval_min_cells),
        "regimes": list(args.regimes),
        "group_column_by_regime": {r: REGIME_GROUP_COLUMN[r] for r in args.regimes},
        "n_splits": int(args.folds),
        "split_seeds": [int(s) for s in args.split_seeds],
        "fold_algorithm": "unseen_chemotype: cohorts.diversity_splits (EXPANDED training index, "
                          "so MONO_ET reproduces Experiment A); other regimes: "
                          "cohorts.seeded_group_kfold on the regime's group column",
        "fold_random_state_formula": f"model_seed + fold * {FOLD_SEED_STRIDE} + {FOLD_SEED_OFFSET}",
        "arms": ["ALL_MODELS"],
        "arm_note": "every model of a fold trains on the same rows; the arm key exists only "
                    "because the manifest schema is per-arm",
        "models": list(MODEL_NAMES),
        "identical_test_rows": True,
        "weighting": "cluster (group_balanced_weights on the training rows' ecfp_cluster)",
        "inner_cv_group_column": "the regime's own group column (tune_ridge and the Stage A "
                                 "cross-fit both)",
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(
        {"definition": split_definition, "folds": fold_records,
         "integrity": [{"regime": r.regime, "split_seed": r.seed, **r.integrity}
                       for r in results],
         "cell_audit": [{"regime": r.regime, "split_seed": r.seed, **r.cell_audit}
                        for r in results]},
        indent=2, default=str) + "\n")

    # --- checks -------------------------------------------------------------------
    checks = build_checks(
        results=results, oof=oof, reproduction=reproduction, pair_audit=pair_audit,
        cohort_extractants=set(frame["extractant"].astype(str)),
        chemistry_extractants=set(chemistry.extractants), pilot=bool(args.pilot))

    # --- manifest -------------------------------------------------------------------
    provenance_state: dict = {"status": "not_audited_in_this_run"}
    if args.provenance_state_json and Path(args.provenance_state_json).exists():
        provenance_state = json.loads(Path(args.provenance_state_json).read_text())
    elif args.provenance_state_json:
        provenance_state = {"status": "provenance_state_file_missing",
                            "path": str(args.provenance_state_json)}

    manifest = RunManifest(layer=GEN6_LAYER, run_id=f"gen6_hierarchical_{stamp}")
    manifest.record_dataset(dataset_path=args.dataset, source_frame=source,
                            descriptor_path=args.descriptors if descriptors is not None else None,
                            descriptor_frame=descriptors)
    manifest.record_code([REPO_ROOT / "src" / "lanthanide_separation" / "gen6",
                          REPO_ROOT / "src" / "lanthanide_separation" / "levels.py",
                          Path(__file__).resolve()], repo_root=REPO_ROOT)
    manifest.record_features(feature_sets=feature_sets)
    manifest.record_split(definition=split_definition, folds=fold_records)
    manifest.record_chemistry(definition={
        **cluster_manifest(chemistry)["definition"],
        "source": chemistry_source,
        "chemistry_table_sha256": sha256_frame(chemistry.table),
        "n_extractants": len(chemistry),
        "audit": chemistry.audit,
    })
    manifest.record_provenance(state=provenance_state)
    manifest.record_preprocessing([
        {"step": "build_level_dataset", "min_rows_per_extractant": int(args.eval_min_cells),
         "replicate_policy": args.replicate_policy, "drop_below_log_d": float(args.log_d_floor),
         "ligand_descriptors": bool(descriptors is not None), "cohort_audit": audit},
        {"step": "ridge design", "blocks": list(RIDGE_BASE_BLOCKS),
         "detail": "fold-local median imputation + standardisation, plus the declared "
                   "interaction block (donor×radius, logL×donor, cond×radius)"},
        {"step": "LevelRegressor pipeline",
         "detail": "DropAllNaNColumns -> SimpleImputer(median, add_indicator) -> ExtraTrees"},
        {"step": "sample weighting", "mode": "cluster",
         "detail": "group_balanced_weights on the training rows' ecfp_cluster, every model"},
        {"step": "penalty selection", "fixed_grid": list(FIXED_PENALTY_GRID),
         "ligand_grid": list(LIGAND_PENALTY_GRID),
         "detail": "inner grouped CV, 3 folds, grouped on the regime's own column"},
    ])
    manifest.record("model_seed", int(args.model_seed))
    manifest.record("split_seeds", [int(s) for s in args.split_seeds])
    manifest.record("cohort_sha256", sha256_frame(frame, sort_rows_by=["row_id"]))
    manifest.record("models", list(MODEL_NAMES))
    manifest.record("contrasts", [c.__dict__ for c in CONTRASTS])
    manifest.record("regimes", list(args.regimes))
    manifest.record("pilot", bool(args.pilot))
    manifest.record("bootstrap", {"replicates": int(args.replicates),
                                  "seed": int(args.bootstrap_seed),
                                  "scoring_unit": "ecfp_cluster",
                                  "resample_block": "tanimoto_cluster"})
    manifest.record("endpoints", [ENDPOINT_ALL, ENDPOINT_MULTI])
    manifest.record("hard_chemistry_thresholds", [float(t) for t in args.thresholds])
    manifest.record("mono_et_reproduction", dict(reproduction))
    manifest.record("pair_identities", dict(pair_audit))

    # --- report ---------------------------------------------------------------------
    report = render_report(
        stamp=stamp, args=args, audit=audit,
        arm_metrics=(arm_metrics.groupby(["regime", "arm"], as_index=False, sort=False)
                     .mean(numeric_only=True)),
        hard_metrics=(hard_metrics.groupby(["regime", "endpoint", "arm"], as_index=False,
                                           sort=False).mean(numeric_only=True)),
        contrast_summary=contrast_summary,
        per_seed=(contrasts[(contrasts["statistic"] == "mae")] if not contrasts.empty
                  else contrasts),
        verdicts=verdicts, lambda_table=lambda_table,
        attribution=attribution_summary,
        pair_summary=(pair_metrics_table.groupby(["regime", "model"], as_index=False, sort=False)
                      .agg(n_pairs=("n_pairs", "sum"),
                           pair_macro_mae=("pair_macro_mae", "mean"),
                           pair_pooled_mae=("pair_pooled_mae", "mean"),
                           pair_sign_accuracy=("pair_sign_accuracy", "mean"),
                           antisymmetry_max=("antisymmetry_max", "max"),
                           transitivity_max=("transitivity_max", "max"),
                           n_triples=("n_triples", "sum"))
                      if not pair_metrics_table.empty else pair_metrics_table),
        pair_audit=pair_audit, reproduction=reproduction,
        endpoint_counts=(endpoint_counts.groupby(["regime", "endpoint"], as_index=False,
                                                 sort=False).mean(numeric_only=True)
                         if not endpoint_counts.empty else endpoint_counts),
        fit_seconds=fit_seconds)
    (output_dir / "decision_report.md").write_text(report)

    (output_dir / "summary.json").write_text(json.dumps({
        "run_id": f"gen6_hierarchical_{stamp}",
        "layer": GEN6_LAYER,
        "pilot": bool(args.pilot),
        "cohort_audit": audit,
        "regimes": list(args.regimes),
        "models": list(MODEL_NAMES),
        "split_seeds": [int(s) for s in args.split_seeds],
        "folds": int(args.folds),
        "n_estimators": int(args.n_estimators),
        "fit_seconds": fit_seconds,
        "mono_et_reproduction": reproduction,
        "pair_identities": pair_audit,
        "endpoint_counts": endpoint_counts.to_dict("records"),
        "arm_metrics": arm_metrics.to_dict("records"),
        "contrast_summary": contrast_summary.to_dict("records"),
        "lambda_frequency": lambda_table.to_dict("records") if not lambda_table.empty else [],
        "component_attribution": attribution_summary.to_dict("records")
        if not attribution_summary.empty else [],
        "hypotheses": [v.__dict__ for v in verdicts],
        "checks": checks,
    }, indent=2, default=str) + "\n")

    validation, success = finalise_run(output_dir, manifest=manifest, checks=checks)

    print(report)
    for v in verdicts:
        log(f"{v.hypothesis}: {v.verdict}")
    if success is None:
        log(f"VALIDATION FAILED: {validation['failed_checks']} "
            f"missing_keys={validation['missing_manifest_keys']} "
            f"missing_artifacts={validation['missing_artifacts']}; wrote _FAILED.json")
        return 1
    log(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
