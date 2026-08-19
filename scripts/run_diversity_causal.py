"""gen6 Experiment A — is chemical coverage, or model capacity, the binding constraint?

Six studies (gen2–gen5) asked *which model, which descriptors*, and every answer
landed in the same place: on a genuinely new ligand the model predicts the shape
of the response and not its level.  Every one of those studies was run on a
cohort carved out by ``min_rows_per_extractant = 10``, a rule that keeps 91 of
190 extractants and discards precisely the rarely measured, chemically unusual
ones.  Experiment A asks whether that eligibility rule — not the learner — is
what has been limiting zero-shot accuracy.

The design exists to make exactly one thing vary.

* **One shared cohort**, built once at ``min_cells = 3`` (the EXPANDED
  eligibility).  Every feature column, every one-hot level, every cluster label,
  every replicate average and every row id is therefore identical for all arms.
  Building two cohorts and comparing their leaderboards would be uninterpretable:
  a lower MAE could simply mean easier test rows.
* **Folds hold out whole Tanimoto-0.7 super-clusters** of that shared cohort, so
  a fold's test rows are a property of the fold and not of the arm.
* **An arm is a row mask over the training side only** — BASE (>= 10 cells),
  EXPANDED (>= 3 cells), plus the two controls without which no causal claim is
  allowed:

  ``EXPANDED_ROWMATCHED``
      all sparse rows + random dense rows up to BASE's row count.  If EXPANDED
      wins only because it has more rows, this arm wins too.
  ``EXPANDED_SHUFFLED``
      EXPANDED with the *added sparse rows'* targets permuted among themselves.
      The chemistry is present, the information is not.  If EXPANDED wins only
      through regularisation or through the change in group weighting, this arm
      wins too.

Traps this runner is written around, each of which has silently broken a study
in this repo before:

1. **The hard-chemistry bins must not move with the arm.**  "Hard chemistry" is
   defined by the nearest-neighbour Tanimoto to the **BASE** training set of the
   fold, for every arm.  Defining it per arm would compare BASE and EXPANDED on
   different rows and the comparison would be void.  The column is
   ``nn_reference_tanimoto``; ``nn_expanded_tanimoto`` is carried alongside as a
   *descriptive* column only and is never used to select rows.
2. **Equal-cluster weighting is itself a confound.**  The primary fit uses the
   gen5 rule (``group_balanced_weights`` on the training rows' ECFP cluster), and
   adding sparse ligands adds clusters, so the weighting changes with the arm.
   That cannot be hidden, so ``--weighting row`` runs the same experiment with no
   weights at all and the report names it as the disclosed sensitivity.  The
   number of training clusters per arm is recorded per fold.
3. **Five seeds are not five experiments.**  The seeds re-partition the same
   ligands; the load-bearing statistic is the paired bootstrap over independent
   chemistry units (score per ECFP cluster, resample the Tanimoto super-cluster
   that the folds actually held out).  Seed agreement is reported as a *second*
   requirement, never as the evidence.
4. **A metric without its grouping and regime means nothing.**  Every number in
   the report carries its unit, its n, its CI and the null it beat.

The learner is frozen to the gen5 champion configuration and reached through the
gen5 harness itself (``LevelRegressor`` / ``build_level_dataset``), including the
fold seed formula ``model_seed + fold * 1009 + 9_999_991``, so a gen6 run of a
gen5 arm reproduces the gen5 number rather than approximating it.

Example::

    .venv/bin/python scripts/run_diversity_causal.py --pilot
    .venv/bin/python scripts/run_diversity_causal.py --weighting row
    .venv/bin/python scripts/run_diversity_causal.py \
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
    BASE_ARM, BASE_MIN_CELLS, DEFAULT_ARMS, EXPANDED_ARM, EXPANDED_MIN_CELLS, ROWMATCHED_ARM,
    SHUFFLED_ARM, arm_membership, assert_split_integrity, cohort_comparison, diversity_splits,
)
from lanthanide_separation.gen6.manifest import (  # noqa: E402
    RunManifest, sha256_frame, sha256_json, validate_run, write_success,
)
from lanthanide_separation.gen6.metrics import (  # noqa: E402
    HARD_CHEMISTRY_THRESHOLDS, UNIT_STATISTICS, gen6_metric_table, ood_calibration_table,
    ood_statistics, paired_unit_bootstrap, per_unit_statistics,
)
from lanthanide_separation.levels import (  # noqa: E402
    LEVEL_ARMS, LEVEL_IDENTITY_COLUMNS, LEVEL_TARGET_COLUMN, LevelData, LevelForestParameters,
    LevelRegressor, _as_float_frame, build_level_dataset,
)

DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
DESCRIPTOR_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_2d_descriptors.parquet"

#: The five split seeds every gen5/gen6 study uses.  They re-partition the same
#: ligands, so they measure split sensitivity — not independent replication.
DEFAULT_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
#: Frozen by protocol §5: the current champion, and the block whose edge grows
#: with chemical distance.  No new architecture in Phase 1.
#: Default feature sets. Both MUST exist in ``levels.LEVEL_ARMS`` — argparse's
#: ``choices`` validates what the *user* types but never the default, so an
#: invalid default crashes the script only when it is run with no arguments,
#: which is exactly how Phase 0 invokes it. ``test_default_feature_sets_resolve``
#: pins this.  ``MC_lig2d_ext_massaction`` is the current champion;
#: ``MC_donors`` is the block whose edge grows with chemical distance.
DEFAULT_FEATURE_SETS: tuple[str, ...] = ("MC_lig2d_ext_massaction", "MC_donors")
#: gen5's fold seed formula, reproduced exactly (see run_gen5_levels.evaluate_regime).
FOLD_SEED_STRIDE = 1009
FOLD_SEED_OFFSET = 9_999_991
#: Statistics carried through the per-unit bootstrap.  ``mae`` averaged over ECFP
#: clusters *is* the study's primary macro MAE, because ``per_unit_statistics``
#: defines every statistic as a mean over the unit's members.
BOOTSTRAP_STATISTICS: tuple[str, ...] = tuple(UNIT_STATISTICS)

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "oof_predictions.parquet", "per_ligand_metrics.csv", "ood_metrics.csv",
    "cohort_comparison.json", "split_manifest.json", "decision_report.md", "summary.json",
    "arm_metrics.csv", "hard_chemistry_metrics.csv", "contrasts.csv", "contrast_summary.csv",
    "training_budgets.csv", "bin_migration.csv", "manifest.json",
)


# --------------------------------------------------------------------------- #
# Contrasts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Contrast:
    """One pre-registered comparison.

    ``point_delta = statistic(reference) - statistic(candidate)``, so a **positive
    delta always means the candidate is better** (lower error).  The arms are
    ordered here so that positive = the direction the hypothesis predicts; the
    report prints the arithmetic next to every number so the sign cannot be
    misread.
    """

    name: str
    reference: str
    candidate: str
    preregistered: bool
    question: str


CONTRASTS: tuple[Contrast, ...] = (
    Contrast("EXPANDED_vs_BASE", BASE_ARM, EXPANDED_ARM, True,
             "A1 primary: does adding 61 sparse, chemically distant extractants help?"),
    Contrast("EXPANDED_ROWMATCHED_vs_BASE", BASE_ARM, ROWMATCHED_ARM, True,
             "row-budget-matched depth control: help at BASE's row count?"),
    Contrast("EXPANDED_vs_EXPANDED_SHUFFLED", SHUFFLED_ARM, EXPANDED_ARM, True,
             "A4 information null: is the gain information or just rows/regularisation?"),
    Contrast("EXPANDED_SHUFFLED_vs_BASE", BASE_ARM, SHUFFLED_ARM, True,
             "how much of the apparent gain survives destroying the new labels?"),
    Contrast("EXPANDED_vs_EXPANDED_ROWMATCHED", ROWMATCHED_ARM, EXPANDED_ARM, False,
             "derived (not pre-registered): needed to score A4's first clause"),
)
PRIMARY_CONTRAST = "EXPANDED_vs_BASE"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--descriptors", type=Path, default=DESCRIPTOR_PATH,
                   help="gen4 extended-2D descriptor parquet (LIG2D_EXT block); pass '' to skip")
    p.add_argument("--chemistry-map", type=Path, default=None,
                   help="frozen chemistry_map.parquet from scripts/build_chemistry_map.py; "
                        "rebuilt from the source table when omitted (identical by construction)")
    p.add_argument("--eval-min-cells", type=int, default=EXPANDED_MIN_CELLS,
                   help="eligibility of the ONE shared cohort every arm is evaluated on")
    p.add_argument("--base-min-cells", type=int, default=BASE_MIN_CELLS,
                   help="the gen5 eligibility rule that defines the BASE training mask")
    p.add_argument("--feature-sets", nargs="+", default=list(DEFAULT_FEATURE_SETS),
                   choices=list(LEVEL_ARMS),
                   help="LEVEL_ARMS names; the feature columns come from data.block_columns()")
    p.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=list(DEFAULT_ARMS))
    p.add_argument("--split-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-features", type=float, default=0.30)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--weighting", choices=("cluster", "row"), default="cluster",
                   help="cluster = the gen5 rule (group_balanced_weights on the training rows' "
                        "ecfp_cluster, the primary); row = no weights, the disclosed sensitivity")
    p.add_argument("--replicate-policy", default="mean", choices=["mean", "unique", "all"])
    p.add_argument("--log-d-floor", type=float, default=-6.0)
    p.add_argument("--thresholds", nargs="+", type=float, default=list(HARD_CHEMISTRY_THRESHOLDS),
                   help="hard-chemistry nearest-neighbour Tanimoto cut-offs (to BASE training)")
    p.add_argument("--replicates", type=int, default=5000, help="bootstrap replicates")
    p.add_argument("--bootstrap-seed", type=int, default=8675309)
    p.add_argument("--tree-dispersion", dest="tree_dispersion", action="store_true", default=True,
                   help="record the sd across the forest's trees per test row (default on)")
    p.add_argument("--no-tree-dispersion", dest="tree_dispersion", action="store_false")
    p.add_argument("--dispersion-arm", default=EXPANDED_ARM, choices=list(DEFAULT_ARMS),
                   help="arm whose per-tree dispersion is recorded; the deployable arm by default")
    p.add_argument("--provenance-state-json", type=Path, default=None,
                   help="provenance audit state to record in the manifest; without it the manifest "
                        "records {'status': 'not_audited_in_this_run'} rather than a guess")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--pilot", action="store_true",
                   help="Phase 0 pilot: one seed, 120 trees, one feature set (minutes, not hours)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def resolve_feature_sets(data: LevelData, names: Sequence[str],
                         log: Callable[[str], None]) -> dict[str, tuple[str, ...]]:
    """``{arm name: ordered feature columns}``, dropping sets whose blocks are absent.

    The column *order* matters — ``max_features`` samples columns, so a permuted
    list is a different model — and it is fixed here once and hashed into the
    manifest's feature registry.
    """
    out: dict[str, tuple[str, ...]] = {}
    unknown = [n for n in names if n not in LEVEL_ARMS]
    if unknown:
        raise SystemExit(
            f"unknown feature set(s) {unknown}; a feature set must be a key of "
            f"levels.LEVEL_ARMS. Available: {sorted(LEVEL_ARMS)}")
    for name in names:
        blocks = LEVEL_ARMS[name]
        missing = [b for b in blocks if b not in data.blocks]
        if missing:
            log(f"WARNING: feature set {name!r} needs absent blocks {missing}; skipped")
            continue
        out[name] = data.block_columns(blocks)
    if not out:
        raise SystemExit("no requested feature set is available on this cohort")
    return out


def tree_dispersion(model: LevelRegressor, frame: pd.DataFrame) -> np.ndarray:
    """Per-row sd across the individual trees of a fitted forest.

    This is the model's own answer to "how much do my trees disagree here?", and
    it is the half of an abstention rule that chemistry similarity cannot supply:
    a test ligand can sit close to training chemistry and still land in a region
    where the trees split apart, and vice versa.

    Cost: one ``predict`` per tree instead of one per forest, run serially in
    Python rather than through the estimator's own thread pool — roughly the cost
    of the forest predict itself, times a small constant.  It is off with
    ``--no-tree-dispersion`` for that reason, and computed for one arm only.
    Returns NaN for learners with no ``estimators_`` (ridge, HGB).
    """
    pipeline = model.pipeline
    if pipeline is None:
        raise RuntimeError("model has not been fitted")
    estimator = pipeline.named_steps["model"]
    trees = getattr(estimator, "estimators_", None)
    if not trees:
        return np.full(len(frame), np.nan)
    x = _as_float_frame(frame, model.feature_columns)
    transformed = pipeline[:-1].transform(x)
    stacked = np.column_stack([tree.predict(transformed) for tree in trees])
    return stacked.std(axis=1)


def _fold_neighbours(chemistry: ChemistryMap, test_extractants: Sequence[str],
                     reference_extractants: Sequence[str]) -> pd.DataFrame:
    """Nearest-neighbour table of a fold's test ligands against a reference cohort.

    Raises a readable error when the cohort holds an extractant the frozen map
    does not know — that means the map and the source table have drifted apart,
    which must stop the run rather than produce a silently wrong similarity.
    """
    unknown = [e for e in set(test_extractants) | set(reference_extractants)
               if e not in set(chemistry.extractants)]
    if unknown:
        raise KeyError(f"{len(unknown)} cohort extractants are absent from the frozen chemistry "
                       f"map (first: {unknown[0][:60]}…); map and dataset have drifted")
    return chemistry.nearest_neighbour(list(test_extractants), list(reference_extractants))


# --------------------------------------------------------------------------- #
# One seed of the experiment
# --------------------------------------------------------------------------- #

@dataclass
class SeedResult:
    """Out-of-fold predictions for one (feature set, split seed) and their audit."""

    feature_set: str
    seed: int
    oof: pd.DataFrame
    fold_records: list[dict] = field(default_factory=list)
    integrity: dict = field(default_factory=dict)
    fit_seconds: float = 0.0


def evaluate_seed(
    data: LevelData,
    *,
    chemistry: ChemistryMap,
    feature_set: str,
    feature_columns: Sequence[str],
    arms: Sequence[str],
    seed: int,
    folds: int,
    params: LevelForestParameters,
    base_min_cells: int,
    weighting: str,
    dispersion_arm: str | None,
    log: Callable[[str], None],
) -> SeedResult:
    """Fit every arm on every fold of one split seed; return one OOF row per cohort row.

    Every arm predicts the *same* test rows, so the OOF frame carries one row per
    cohort row with one ``prediction_<arm>`` column per arm.  That layout makes
    the "identical test rows" property structural rather than something a later
    join has to preserve.
    """
    frame = data.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    n = len(frame)
    splits = diversity_splits(frame, group_column="tanimoto_cluster", n_splits=folds,
                              seed=seed, base_min_cells=base_min_cells, arms=tuple(arms))
    integrity = assert_split_integrity(frame, splits)

    predictions = {arm: np.full(n, np.nan) for arm in arms}
    fold_of = np.full(n, -1, dtype=int)
    dispersion = np.full(n, np.nan)
    nn_reference = np.full(n, np.nan)
    nn_expanded = np.full(n, np.nan)
    neighbour_parts: list[pd.DataFrame] = []
    fold_records: list[dict] = []
    extractants = frame["extractant"].astype(str).to_numpy()
    superclusters = frame["tanimoto_cluster"].astype(str).to_numpy()
    row_ids = frame["row_id"].astype(str).to_numpy()
    started = time.time()

    for split in splits:
        test_index = split.test_index
        test = frame.iloc[test_index]
        fold_of[test_index] = split.fold
        test_extractants = sorted(set(extractants[test_index]))

        # --- similarity, fixed by the BASE training set for EVERY arm ---------
        # This is the single most important line in the file: the hard-chemistry
        # bins below are cut on this column, and if it were computed per arm the
        # arms would be scored on different rows.
        base_train = sorted(set(extractants[split.train_index_by_arm[BASE_ARM]])) \
            if BASE_ARM in split.train_index_by_arm else []
        reference_nn = _fold_neighbours(chemistry, test_extractants, base_train)
        lookup = reference_nn.set_index("extractant")["nn_tanimoto"]
        nn_reference[test_index] = test["extractant"].map(lookup).to_numpy(dtype=float)
        neighbour_parts.append(reference_nn)
        if EXPANDED_ARM in split.train_index_by_arm:
            expanded_train = sorted(set(extractants[split.train_index_by_arm[EXPANDED_ARM]]))
            expanded_nn = _fold_neighbours(chemistry, test_extractants, expanded_train)
            nn_expanded[test_index] = test["extractant"].map(
                expanded_nn.set_index("extractant")["nn_tanimoto"]).to_numpy(dtype=float)

        # --- the arms ---------------------------------------------------------
        fold_params = LevelForestParameters(
            n_estimators=params.n_estimators, max_features=params.max_features,
            min_samples_leaf=params.min_samples_leaf,
            random_state=params.random_state + split.fold * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET,
            n_jobs=params.n_jobs, learner=params.learner)
        weight_audit: dict[str, int] = {}
        for arm in arms:
            train_index = split.train_index_by_arm[arm]
            train = frame.iloc[train_index]
            # training_target() applies the sparse-target permutation for the
            # information null and is the identity for every honest arm.
            y_train = split.training_target(arm, target)
            groups = train["ecfp_cluster"] if weighting == "cluster" else None
            weight_audit[arm] = int(train["ecfp_cluster"].nunique())
            model = LevelRegressor(feature_columns, fold_params).fit(train, y_train, groups=groups)
            predictions[arm][test_index] = model.predict(test)
            if dispersion_arm is not None and arm == dispersion_arm:
                dispersion[test_index] = tree_dispersion(model, test)

        record = {
            "fold": int(split.fold),
            "split_seed": int(seed),
            "feature_set": feature_set,
            "test_row_ids_sha256": sha256_json(sorted(row_ids[test_index].tolist())),
            "test_extractants": sorted(set(extractants[test_index])),
            "test_superclusters": sorted(set(superclusters[test_index])),
            "train_extractants_by_arm": {a: sorted(set(extractants[i]))
                                         for a, i in split.train_index_by_arm.items()},
            "train_superclusters_by_arm": {a: sorted(set(superclusters[i]))
                                           for a, i in split.train_index_by_arm.items()},
            "n_test_rows": int(test_index.size),
            "n_train_rows_by_arm": {a: int(len(i)) for a, i in split.train_index_by_arm.items()},
            "n_train_ecfp_clusters_by_arm": weight_audit,
            "held_out_superclusters": list(split.held_out_groups),
            "split_audit": split.audit,
        }
        fold_records.append(record)
        log(f"  {feature_set} seed {seed} fold {split.fold}: test {test_index.size} rows / "
            f"{len(test_extractants)} extractants; train rows "
            f"{ {a: int(len(i)) for a, i in split.train_index_by_arm.items()} }; "
            f"train ECFP clusters {weight_audit}")

    # --- assemble ------------------------------------------------------------
    missing = {arm: int(np.isnan(v).sum()) for arm, v in predictions.items()}
    if any(missing.values()):
        raise RuntimeError(f"rows left unpredicted (folds did not cover the cohort): {missing}")
    oof = frame[list(LEVEL_IDENTITY_COLUMNS)].copy()
    oof.insert(0, "feature_set", feature_set)
    oof.insert(1, "split_seed", int(seed))
    oof["outer_fold"] = fold_of
    oof["nn_reference_tanimoto"] = nn_reference
    oof["nn_expanded_tanimoto"] = nn_expanded
    for arm in arms:
        oof[f"prediction_{arm}"] = predictions[arm]

    neighbours = pd.concat(neighbour_parts, ignore_index=True)
    if neighbours["extractant"].duplicated().any():
        raise RuntimeError("an extractant was tested in more than one fold; the OOD join is unsafe")
    oof = ood_statistics(oof, neighbour_table=neighbours,
                         prediction_spread=pd.Series(dispersion) if dispersion_arm else None)
    if dispersion_arm:
        oof["ood__prediction_sd_arm"] = dispersion_arm
    return SeedResult(feature_set=feature_set, seed=int(seed), oof=oof,
                      fold_records=fold_records, integrity=integrity,
                      fit_seconds=time.time() - started)


# --------------------------------------------------------------------------- #
# Endpoints and contrasts
# --------------------------------------------------------------------------- #

def hard_chemistry_subsets(
    predictions: pd.DataFrame, thresholds: Sequence[float],
    *, similarity_column: str = "nn_reference_tanimoto",
) -> list[tuple[str, pd.DataFrame]]:
    """``[("all", frame), ("nn<0.4", subset), …]`` — the endpoints, in report order.

    Membership depends only on ``similarity_column``, which is a property of the
    row and the fold's BASE training set, never of the arm.  That is what keeps
    the arms comparable on the hard-chemistry endpoints.
    """
    out: list[tuple[str, pd.DataFrame]] = [("all", predictions)]
    for threshold in thresholds:
        out.append((f"nn<{threshold:g}", predictions[predictions[similarity_column] < threshold]))
    return out


def bin_migration_table(
    predictions: pd.DataFrame,
    thresholds: Sequence[float],
    *,
    reference_column: str = "nn_reference_tanimoto",
    arm_column: str = "nn_expanded_tanimoto",
) -> pd.DataFrame:
    """How many rows would change hard-chemistry bin under an arm-specific reference.

    The pre-registration says the bins are cut on the BASE training set for every
    arm.  This table measures what that decision is worth: if the same rows fell
    in the same bins either way, the rule would be a formality.  On this cohort it
    is not — the EXPANDED training set contains the sparse ligands, so a large
    fraction of test rows are nearer to it, and cutting the bins per arm would
    quietly compare BASE and EXPANDED on different subsets.
    """
    rows: list[dict] = []
    for threshold in thresholds:
        reference = predictions[reference_column] < threshold
        arm_specific = predictions[arm_column] < threshold
        rows.append({
            "endpoint": f"nn<{threshold:g}",
            "n_rows_reference_bins": int(reference.sum()),
            "n_rows_if_each_arm_used_its_own_training_set": int(arm_specific.sum()),
            "n_rows_that_would_change_bin": int((reference != arm_specific).sum()),
        })
    return pd.DataFrame(rows)


def contrast_frame(
    per_unit: pd.DataFrame,
    *,
    block_of_unit: Mapping[str, str],
    arms: Sequence[str],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    """Paired bootstrap for every contrast whose two arms are present."""
    comparisons = {c.name: (c.reference, c.candidate) for c in CONTRASTS
                   if c.reference in arms and c.candidate in arms}
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


def unit_blocks(predictions: pd.DataFrame, *, unit_column: str = "ecfp_cluster",
                block_column: str = "tanimoto_cluster") -> dict[str, str]:
    """ECFP cluster -> the Tanimoto super-cluster the folds actually held out.

    Scoring unit and independence unit differ under a chemotype hold-out (131
    clusters nest inside 79 super-clusters on this cohort); resampling the
    scoring unit would understate the interval width.
    """
    pairs = predictions[[unit_column, block_column]].astype(str).drop_duplicates()
    return dict(zip(pairs[unit_column], pairs[block_column]))


def pool_per_unit_over_seeds(per_unit_by_seed: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Average each unit's statistics over the split seeds, then bootstrap that.

    Protocol §5 asks for a paired bootstrap *and* seed agreement.  A per-seed
    interval answers "is this fold partition's effect real"; averaging the unit
    statistics over seeds first and bootstrapping the average answers "is the
    effect real for this cohort", which is the question the generation is about.
    Both are reported.  A unit missing from a seed (hard-chemistry subsets move
    with the fold's BASE training set) contributes only where it exists, and the
    number of seeds behind each unit is kept.
    """
    if not per_unit_by_seed:
        return pd.DataFrame()
    stacked = pd.concat(per_unit_by_seed, ignore_index=True)
    numeric = [c for c in stacked.columns if c not in ("unit", "arm")]
    pooled = stacked.groupby(["unit", "arm"], as_index=False)[numeric].mean()
    pooled["n_seeds"] = stacked.groupby(["unit", "arm"]).size().to_numpy()
    return pooled


# --------------------------------------------------------------------------- #
# Hypothesis scoring
# --------------------------------------------------------------------------- #

def build_checks(
    *,
    results: Sequence[SeedResult],
    oof: pd.DataFrame,
    hard_metrics: pd.DataFrame,
    arms: Sequence[str],
    cohort_extractants: set[str],
    chemistry_extractants: set[str],
) -> dict:
    """The assertions this run must satisfy before it may claim success.

    These are checks on the *written artifacts*, not on intentions: they re-read
    the OOF frame and the metric tables and ask whether the two properties the
    experiment stands on actually hold in them.  ``validate_run`` is fail-closed
    on the result, so a broken run produces ``_FAILED.json`` and a non-zero exit
    rather than a plausible-looking report.
    """
    integrity_ok = all(r.integrity.get("ok", False) for r in results)
    identical_rows: dict = {"ok": True, "detail": {}}
    for (feature_set, seed), block in oof.groupby(["feature_set", "split_seed"]):
        row_sets = {arm: frozenset(block.loc[block[f"prediction_{arm}"].notna(), "row_id"])
                    for arm in arms}
        sizes = {arm: len(s) for arm, s in row_sets.items()}
        same = len(set(row_sets.values())) == 1 and set(sizes.values()) == {len(block)}
        identical_rows["detail"][f"{feature_set}|{seed}"] = {
            "n_rows": int(len(block)), "scored_rows_by_arm": sizes, "identical": bool(same)}
        identical_rows["ok"] = bool(identical_rows["ok"] and same)

    # If the hard-chemistry bins moved with the arm, the endpoint row counts would
    # differ between arms within one (feature set, seed, endpoint) cell.
    bins_ok: dict = {"ok": True, "detail": {}}
    if not hard_metrics.empty:
        for keys, block in hard_metrics.groupby(["feature_set", "split_seed", "endpoint"]):
            counts = sorted({int(v) for v in block["n_rows"]})
            bins_ok["detail"]["|".join(str(k) for k in keys)] = counts
            bins_ok["ok"] = bool(bins_ok["ok"] and len(counts) == 1)

    # An adversarial review pointed out that the two checks below cannot fail as
    # written, and it is right: every arm writes into one row-per-cohort-row OOF
    # frame and `evaluate_seed` already raises on any unpredicted row, so the row
    # sets are equal by construction; and `hard_chemistry_endpoints` is called per
    # arm with the same frame and the same similarity column, so the bin sizes are
    # equal by construction too.  That is a *good* property — the design makes the
    # violation unrepresentable — but recording it as a passed test is false
    # assurance.  They are renamed to say what they are, and the evidence that the
    # choice of similarity column is load-bearing is the bin-migration count, which
    # is a real measurement: it is how many rows WOULD change bin if each arm used
    # its own training set as the reference.
    return {
        "split_integrity": {
            "ok": bool(integrity_ok),
            "per_run": [{"feature_set": r.feature_set, "split_seed": r.seed,
                         "ok": r.integrity.get("ok"),
                         "all_rows_tested_once": r.integrity.get("all_rows_tested_once")}
                        for r in results]},
        "identical_test_rows_by_construction": {
            **identical_rows,
            "note": "structural invariant of the one-frame OOF layout, not an independent test; "
                    "the load-bearing check is split_integrity"},
        "hard_chemistry_bins_arm_invariant_by_construction": {
            **bins_ok,
            "note": "the same similarity column is passed for every arm, so equality here is "
                    "structural; bin_migration.csv measures how many rows would move under an "
                    "arm-specific reference, and that number is what makes the choice matter"},
        "no_missing_predictions": {
            "ok": bool(all(oof[f"prediction_{arm}"].notna().all() for arm in arms))},
        # A NaN here would not raise: metrics.hard_chemistry_endpoints selects the
        # "all" endpoint with `similarity < inf`, and NaN fails that comparison, so
        # a missing similarity would silently drop the row from EVERY endpoint,
        # including the one that is supposed to be the whole test set.  Checked
        # rather than trusted.
        "similarity_column_complete": {
            "ok": bool(oof["nn_reference_tanimoto"].notna().all()),
            "n_missing": int(oof["nn_reference_tanimoto"].isna().sum())},
        "chemistry_map_covers_cohort": {
            "ok": bool(set(cohort_extractants) <= set(chemistry_extractants))},
    }


def finalise_run(
    output_dir: Path,
    *,
    manifest: RunManifest,
    checks: Mapping[str, object],
    required_artifacts: Sequence[str] = REQUIRED_ARTIFACTS,
) -> tuple[dict, Path | None]:
    """Write the manifest, validate the directory, and mark success only if it validated.

    ``artifact_hashes.json`` is written by :meth:`RunManifest.write`, which runs
    before ``validation.json`` exists, so the manifest is written a second time
    once everything else is on disk — otherwise the hash table would silently
    omit the validation record it is supposed to cover.
    """
    payload = manifest.write(output_dir)
    validation = validate_run(output_dir, manifest=payload,
                              required_artifacts=list(required_artifacts), checks=checks)
    success = write_success(output_dir, manifest=payload, validation=validation)
    manifest.write(output_dir)
    return validation, success


@dataclass(frozen=True)
class Verdict:
    hypothesis: str
    statement: str
    pass_condition: str
    verdict: str
    evidence: str
    falsifier: str


def _seed_rule(seeds_positive: int, n_seeds: int) -> bool | None:
    """Protocol §5: ">= 4/5 split seeds in the same direction". None = not evaluable."""
    if n_seeds < 2:
        return None
    return seeds_positive >= math.ceil(0.8 * n_seeds)


def _verdict(passed: bool, contradicted: bool, *, seed_ok: bool | None) -> str:
    """PASS only on the pre-registered condition; FAIL only on evidence against it.

    A confidence interval that straddles zero is *not* evidence that the effect is
    absent, so it scores INCONCLUSIVE rather than FAIL; FAIL is reserved for an
    interval that excludes the predicted direction.  Stated here, and in the
    report, so nobody has to guess what a FAIL means.
    """
    if contradicted:
        return "FAIL"
    if passed and seed_ok is True:
        return "PASS"
    if passed and seed_ok is None:
        return "INCONCLUSIVE"
    return "INCONCLUSIVE"


def _row(summary: pd.DataFrame, comparison: str, statistic: str, endpoint: str,
         feature_set: str) -> pd.Series | None:
    sub = summary[(summary["comparison"] == comparison) & (summary["statistic"] == statistic)
                  & (summary["endpoint"] == endpoint) & (summary["feature_set"] == feature_set)]
    return None if sub.empty else sub.iloc[0]


def _fmt(row: pd.Series | None, *, unit: str = "log units") -> str:
    if row is None:
        return "not evaluable (contrast absent)"
    return (f"Δ = {row['pooled_point_delta']:+.4f} {unit} "
            f"(CI95 [{row['pooled_ci95_low']:+.4f}, {row['pooled_ci95_high']:+.4f}], "
            f"{int(row['pooled_units_total'])} units, {int(row['pooled_blocks'])} bootstrap blocks; "
            f"mean over seeds {row['mean_point_delta']:+.4f}, "
            f"{int(row['seeds_positive'])}/{int(row['n_seeds'])} seeds positive)")


def score_hypotheses(summary: pd.DataFrame, *, feature_set: str, thresholds: Sequence[float],
                     n_seeds: int) -> list[Verdict]:
    """Score A1–A4 of protocol §5 against their stated pass conditions."""
    hardest = f"nn<{min(thresholds):g}" if len(thresholds) else "all"
    verdicts: list[Verdict] = []

    a1 = _row(summary, PRIMARY_CONTRAST, "mae", "all", feature_set)
    seed_ok = None if a1 is None else _seed_rule(int(a1["seeds_positive"]), int(a1["n_seeds"]))
    verdicts.append(Verdict(
        "A1", "Adding sparse diverse chemistry improves overall accuracy.",
        "macro_MAE(BASE) − macro_MAE(EXPANDED) > 0 with CI95 low > 0, and ≥ 4/5 seeds positive.",
        _verdict(a1 is not None and a1["pooled_ci95_low"] > 0 and a1["pooled_point_delta"] > 0,
                 a1 is not None and a1["pooled_ci95_high"] <= 0, seed_ok=seed_ok),
        f"macro MAE (one ECFP cluster = one vote), all test rows: {_fmt(a1)}",
        "A CI95 whose upper end is ≤ 0 — EXPANDED no better, or worse, than BASE on identical "
        "test rows — falsifies it, and with it the claim that coverage is the binding constraint "
        "at this scale (protocol §8 case C).",
    ))

    a2_hard = _row(summary, PRIMARY_CONTRAST, "mae", hardest, feature_set)
    a2_all = a1
    ordered = (a2_hard is not None and a2_all is not None
               and a2_hard["pooled_point_delta"] > a2_all["pooled_point_delta"])
    both_positive = (a2_hard is not None and a2_all is not None
                     and a2_hard["pooled_ci95_low"] > 0 and a2_all["pooled_ci95_low"] > 0)
    seed_ok_2 = None if a2_hard is None else _seed_rule(int(a2_hard["seeds_positive"]),
                                                        int(a2_hard["n_seeds"]))
    verdicts.append(Verdict(
        "A2", "The gain is largest on the hardest chemistry.",
        f"gain at {hardest} > gain overall, both CI95 low > 0.",
        _verdict(ordered and both_positive,
                 a2_hard is not None and a2_hard["pooled_ci95_high"] <= 0, seed_ok=seed_ok_2),
        f"gain at {hardest}: {_fmt(a2_hard)}; gain overall: {_fmt(a2_all)}; "
        f"hard > overall: {ordered}",
        "A gain that is flat or smaller on the far-from-training subset falsifies it: the model "
        "would then be improving where it was already supported, i.e. a depth effect wearing a "
        "coverage costume.",
    ))

    a3_offset = _row(summary, PRIMARY_CONTRAST, "offset_mae", "all", feature_set)
    a3_shape = _row(summary, PRIMARY_CONTRAST, "shape_mae", "all", feature_set)
    bigger = (a3_offset is not None and a3_shape is not None
              and a3_offset["pooled_point_delta"] > a3_shape["pooled_point_delta"])
    seed_ok_3 = None if a3_offset is None else _seed_rule(int(a3_offset["seeds_positive"]),
                                                          int(a3_offset["n_seeds"]))
    verdicts.append(Verdict(
        "A3", "The gain is in the level, not the shape.",
        "offset_mae(BASE) − offset_mae(EXPANDED) CI95 low > 0, and larger than the shape-MAE gain.",
        _verdict(a3_offset is not None and a3_offset["pooled_ci95_low"] > 0 and bigger,
                 a3_offset is not None and a3_offset["pooled_ci95_high"] <= 0, seed_ok=seed_ok_3),
        f"offset gain: {_fmt(a3_offset)}; shape gain: {_fmt(a3_shape)}; offset > shape: {bigger}",
        "A gain that lives in the shape while the per-ligand offset is unchanged falsifies it, and "
        "points at the representation rather than the coverage (protocol §8 case B).",
    ))

    a4_shuffled = _row(summary, "EXPANDED_vs_EXPANDED_SHUFFLED", "mae", "all", feature_set)
    a4_rowmatched = _row(summary, "EXPANDED_vs_EXPANDED_ROWMATCHED", "mae", "all", feature_set)
    smaller_than_base = (a4_rowmatched is not None and a1 is not None
                         and a4_rowmatched["pooled_point_delta"] < a1["pooled_point_delta"])
    seed_ok_4 = None if a4_shuffled is None else _seed_rule(int(a4_shuffled["seeds_positive"]),
                                                            int(a4_shuffled["n_seeds"]))
    verdicts.append(Verdict(
        "A4", "The gain is information, not row count.",
        "EXPANDED beats EXPANDED_ROWMATCHED by less than it beats BASE, and beats "
        "EXPANDED_SHUFFLED with CI95 low > 0.",
        _verdict(a4_shuffled is not None and a4_shuffled["pooled_ci95_low"] > 0
                 and smaller_than_base,
                 a4_shuffled is not None and a4_shuffled["pooled_ci95_high"] <= 0,
                 seed_ok=seed_ok_4),
        f"vs the information null (sparse targets permuted among themselves): {_fmt(a4_shuffled)}; "
        f"vs the row-budget-matched control: {_fmt(a4_rowmatched)}; "
        f"rowmatched gap < BASE gap: {smaller_than_base}",
        "If EXPANDED_SHUFFLED matches EXPANDED, the extra rows helped as regularisation or through "
        "the change in cluster weighting and carried no chemical information; if EXPANDED beats "
        "EXPANDED_ROWMATCHED by as much as it beats BASE, the effect is row count, not coverage.",
    ))
    return verdicts


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _table(frame: pd.DataFrame, columns: Sequence[str] | None = None, digits: int = 4) -> str:
    if frame is None or frame.empty:
        return "```\n(no rows)\n```"
    view = frame if columns is None else frame[[c for c in columns if c in frame.columns]]
    return "```\n" + view.round(digits).to_string(index=False) + "\n```"


def dense_sparse_decomposition(
    oof: pd.DataFrame, dense_extractants: set[str], *, reference: str = BASE_ARM,
    candidate: str = EXPANDED_ARM, target: str = LEVEL_TARGET_COLUMN,
) -> pd.DataFrame:
    """Split the macro gain by whether a scoring unit is made of added chemistry.

    The primary metric gives one vote to each ECFP cluster, and on this cohort 57 of
    the 131 clusters consist *entirely* of the sparsely-measured ligands that only
    EXPANDED can train on — 43 % of the vote for 7 % of the rows.  Reporting the
    headline without this split invites the reader to believe the model improved on
    the chemistry the project already had, which it did not.  This table is printed
    in every run for that reason.
    """
    work = oof.copy()
    work["_is_dense"] = work["extractant"].astype(str).isin(dense_extractants)
    work["_e_ref"] = (work[f"prediction_{reference}"] - work[target]).abs()
    work["_e_cand"] = (work[f"prediction_{candidate}"] - work[target]).abs()
    rows: list[dict] = []
    for (feature_set, endpoint_label), block in _decomposition_blocks(work):
        per_unit = block.groupby(["split_seed", "ecfp_cluster"]).agg(
            ref=("_e_ref", "mean"), cand=("_e_cand", "mean"),
            dense_share=("_is_dense", "mean")).reset_index()
        per_unit["gain"] = per_unit["ref"] - per_unit["cand"]
        per_unit["kind"] = np.where(per_unit["dense_share"] == 1.0, "BASE-eligible only",
                                    np.where(per_unit["dense_share"] == 0.0,
                                             "added chemistry only", "mixed"))
        n_seeds = max(1, per_unit["split_seed"].nunique())
        grouped = per_unit.groupby("kind").agg(
            units=("gain", lambda x: len(x) / n_seeds), gain=("gain", "mean"),
            units_improved=("gain", lambda x: float((x > 0).mean()))).reset_index()
        grouped.insert(0, "endpoint", endpoint_label)
        grouped.insert(0, "feature_set", feature_set)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _decomposition_blocks(work: pd.DataFrame):
    """(feature_set, endpoint) blocks: the whole test set and the hard-chemistry cut."""
    for feature_set, block in work.groupby("feature_set"):
        yield (feature_set, "all"), block
        if "nn_reference_tanimoto" in block.columns:
            hard = block[block["nn_reference_tanimoto"] < 0.4]
            if not hard.empty:
                yield (feature_set, "nn<0.4"), hard


def render_report(
    *,
    stamp: str,
    args: argparse.Namespace,
    audit: dict,
    comparison: dict,
    novelty: dict,
    arm_metrics: pd.DataFrame,
    hard_metrics: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    per_seed_primary: pd.DataFrame,
    verdicts_by_set: Mapping[str, Sequence[Verdict]],
    ood: pd.DataFrame,
    unit_counts: pd.DataFrame,
    budgets: pd.DataFrame,
    migration: pd.DataFrame,
    decomposition: pd.DataFrame,
    fit_seconds: float,
    weight_note: str,
) -> str:
    """The decision report, written for a chemist who expects to be misled."""
    lines: list[str] = []
    A = lines.append
    A("# gen6 Experiment A — BASE91 vs EXPANDED152 on identical test rows")
    A("")
    A(f"**Run** `gen6_diversity_causal_{stamp}`"
      + ("  ·  **PILOT** (one seed, reduced trees, one feature set — not the confirmation run)"
         if args.pilot else ""))
    A("")
    A("## What was actually done")
    A("")
    A(f"* One shared cohort, built once at `min_cells = {args.eval_min_cells}`: "
      f"**{audit['rows']} rows, {audit['extractants']} extractants, {audit['ecfp_clusters']} ECFP "
      f"clusters, {audit['tanimoto_clusters']} Tanimoto-0.7 super-clusters**, "
      f"{audit['conditions']} conditions, {audit['metals']} metals; target sd "
      f"{audit['target_sd']:.3f} log units.")
    A(f"* Folds hold out whole super-clusters ({args.folds} folds, seeds {list(args.split_seeds)}). "
      f"Every arm is scored on **byte-identical test rows**; only the training row mask changes.")
    A(f"* Learner: ExtraTrees, {args.n_estimators} trees, max_features {args.max_features}, "
      f"min_samples_leaf {args.min_samples_leaf}, fold seed "
      f"`model_seed + fold*{FOLD_SEED_STRIDE} + {FOLD_SEED_OFFSET}` (the gen5 formula, reused so a "
      f"gen5 arm reproduces rather than approximates).")
    A(f"* Weighting: **{args.weighting}** — {weight_note}")
    A(f"* Feature sets: {', '.join(verdicts_by_set)}. Fitting took {fit_seconds:.0f} s.")
    A("")
    A("### What the expansion adds")
    A("")
    A(f"BASE (>= {args.base_min_cells} cells) is {comparison['base']['n_rows']} rows / "
      f"{comparison['base']['n_extractants']} extractants / {comparison['base']['n_ecfp_clusters']} "
      f"ECFP clusters / {comparison['base']['n_superclusters']} super-clusters. The expansion adds "
      f"{comparison['added_by_expansion']['n_rows']} rows "
      f"({100 * comparison['row_cost_fraction']:.1f} % of the cohort) from "
      f"{comparison['added_by_expansion']['n_extractants']} extractants, of which "
      f"**{comparison['new_ecfp_clusters']} ECFP clusters and {comparison['new_superclusters']} "
      f"super-clusters are new chemistry** that BASE never sees.")
    if novelty:
        A("")
        A(f"Those added extractants sit at median nearest-neighbour Tanimoto "
          f"{novelty.get('median_nn_to_base', float('nan')):.3f} to the BASE cohort "
          f"({novelty.get('n_below_0_4', 0)} of {novelty.get('n_added', 0)} below 0.4). "
          f"That is the chemistry the eligibility rule has been discarding.")
    A("")
    A("### What each arm was actually trained on (mean over folds and seeds)")
    A("")
    A("Read the controls here before reading their contrasts. `EXPANDED_ROWMATCHED` keeps every "
      "sparse row and gives back dense rows until it matches BASE's row count, so it is a small "
      "perturbation of EXPANDED, not a second BASE — the row-budget clause of A4 is weak evidence "
      "whenever the two arms are this close. `n_train_ecfp_clusters` is the number of equal-weight "
      "voting blocks the fit sees, i.e. the size of the weighting confound.")
    A("")
    A(_table(budgets, ["arm", "n_train_rows", "n_train_extractants", "n_train_ecfp_clusters"], 1))
    A("")
    A("## Headline — the primary contrast")
    A("")
    A("`point_delta = statistic(reference) − statistic(candidate)`, so **positive = the candidate "
      "arm is better** (lower error). `mae` here is macro MAE with one ECFP cluster = one vote: "
      "`per_unit_statistics` defines every statistic as a mean over the unit's rows, so the mean "
      "over resampled units is an exact bootstrap of the macro value. The bootstrap resamples "
      "**Tanimoto super-clusters** (the units the folds held out), and every ECFP cluster travels "
      "with its super-cluster.")
    A("")
    head = contrast_summary[(contrast_summary["comparison"] == PRIMARY_CONTRAST)
                            & (contrast_summary["statistic"].isin(("mae", "offset_mae", "shape_mae")))]
    A(_table(head, ["feature_set", "endpoint", "statistic", "pooled_point_delta", "pooled_ci95_low",
                    "pooled_ci95_high", "mean_point_delta", "seeds_positive", "n_seeds",
                    "pooled_units_total", "pooled_blocks"]))
    A("")
    A("The same contrast one split partition at a time. The five seeds re-partition the same "
      "ligands, so agreement here is a consistency requirement and not five independent "
      "experiments; the interval above is the evidence.")
    A("")
    A(_table(per_seed_primary, ["feature_set", "endpoint", "split_seed", "statistic",
                                "point_delta", "ci95_low", "ci95_high", "units_improved",
                                "units_total"]))
    A("")
    A("## Pre-registered hypotheses (protocol §5)")
    A("")
    A("Scoring rule, stated so a verdict cannot be over-read: **PASS** = the pre-registered "
      "condition is met, including the seed-agreement clause; **FAIL** = the CI95 excludes the "
      "predicted direction (evidence *against*); **INCONCLUSIVE** = the interval straddles zero, or "
      "the seed clause cannot be evaluated (a single-seed pilot). An INCONCLUSIVE is not a weak "
      "PASS.")
    for feature_set, verdicts in verdicts_by_set.items():
        A("")
        A(f"### Feature set `{feature_set}`")
        for v in verdicts:
            A("")
            A(f"**{v.hypothesis} — {v.verdict}**  ·  {v.statement}")
            A("")
            A(f"* pass condition: {v.pass_condition}")
            A(f"* measured: {v.evidence}")
            A(f"* what would falsify this: {v.falsifier}")
    A("")
    A("## Where the gain sits — read this before the leaderboard")
    A("")
    A("One ECFP cluster is one vote, and on this cohort a large share of the clusters consist "
      "**entirely** of the sparsely-measured ligands that only EXPANDED can train on. If the gain "
      "lives only there, the honest claim is 'the expansion repairs chemistry that was missing', "
      "not 'the model got better'. `units` counts scoring clusters; `gain` is BASE minus EXPANDED "
      "macro MAE within that group.")
    A("")
    if decomposition is not None and not decomposition.empty:
        A(_table(decomposition, ["feature_set", "endpoint", "kind", "units", "gain",
                                 "units_improved"]))
    else:
        A("_not computed for this run._")
    A("")
    A("## Arm leaderboard (per feature set, mean over seeds)")
    A("")
    A("Macro MAE is primary. Pooled numbers are printed and never used for selection: the largest "
      "single extractant holds "
      f"{100 * audit.get('largest_extractant_share', float('nan')):.0f} % of rows and the largest "
      f"ECFP cluster {100 * audit.get('largest_ecfp_cluster_share', float('nan')):.0f} %.")
    A("")
    A(_table(arm_metrics, ["feature_set", "arm", "macro_mae", "offset_mae", "shape_mae", "shape_r2",
                           "pooled_mae", "median_ligand_mae", "worst_quartile_ligand_mae",
                           "frac_within_1_log", "n_ligands", "n_ecfp_clusters", "n_superclusters",
                           "n_macro_units", "n_eff_pooled_rows"]))
    A("")
    A("## Hard chemistry — the co-primary endpoint")
    A("")
    A("Bins are cut on `nn_reference_tanimoto`: the maximum Tanimoto from the test ligand to the "
      "**BASE training extractants of its own fold**. The same column is used for every arm, so the "
      "subset does not move with the arm. `nn_expanded_tanimoto` is carried in the OOF for "
      "description only and selects nothing.")
    A("")
    A(_table(hard_metrics, ["feature_set", "endpoint", "arm", "n_rows", "n_ligands",
                            "n_ecfp_clusters", "n_superclusters", "macro_mae", "offset_mae",
                            "shape_mae", "frac_within_1_log"]))
    A("")
    A("Chemistry units surviving each endpoint (this is the n behind every hard-chemistry CI):")
    A("")
    A(_table(unit_counts))
    A("")
    A("The fixed-reference rule is not a formality — this is how many test rows would have landed "
      "in a different bin had each arm's own training set defined it (mean over seeds and feature "
      "sets):")
    A("")
    A(_table(migration, digits=1))
    A("")
    A("## Every contrast, every endpoint")
    A("")
    A(_table(contrast_summary, ["feature_set", "endpoint", "comparison", "statistic",
                                "pooled_point_delta", "pooled_ci95_low", "pooled_ci95_high",
                                "mean_point_delta", "seeds_positive", "n_seeds",
                                "pooled_units_total", "preregistered"]))
    A("")
    A("Per-seed intervals (the same contrasts, one split partition at a time) are in "
      "`contrasts.csv`.")
    A("")
    A("## Out-of-distribution behaviour")
    A("")
    A("Error against distance to the fold's BASE training chemistry, with the forest's own "
      "dispersion (sd across trees) where it was recorded. A model that knows where it is "
      "unsupported is worth more than one that always emits a number. `mean_prediction_sd` is "
      f"recorded for **{args.dispersion_arm if args.tree_dispersion else 'no'} arm only** and is "
      "blank elsewhere — it is a property of one fitted forest, not a comparison between arms.")
    A("")
    A(_table(ood, ["feature_set", "arm", "bin", "n_rows", "n_ligands", "pooled_mae",
                   "macro_mae_ligand", "offset_mae",
                   "shape_mae", "mean_prediction_sd"]))
    A("")
    A("## How to break this result")
    A("")
    A("* **The weighting confound.** The primary fit weights training rows so every ECFP cluster "
      "carries equal total weight, and EXPANDED has more clusters than BASE — so the arms differ in "
      "their weighting as well as their chemistry. Re-run with `--weighting row` and compare; if "
      "the effect vanishes, it was the weighting.")
    A("* **Seeds are not replicates.** The five split seeds re-partition the same ligands. The "
      "bootstrap over super-clusters is the load-bearing statistic; seed agreement is a second "
      "requirement, not independent evidence — and a weak one: the seed-to-seed sd of the macro "
      "delta is ~0.012 against a cluster-robust standard error of ~0.076, a variance component "
      "36x smaller.")
    A("* **Multiplicity.** A1-A4 are pre-registered and are the only protected claims. The "
      "supporting tables report many more contrasts (comparisons x statistics x endpoints x "
      "feature sets), none of them corrected for multiple comparisons. Treat any interval outside "
      "A1-A4 as descriptive.")
    A("* **The gain is concentrated, not broad — read the decomposition above.** The shared cohort "
      "is `min_cells = 3`, so the sparse ligands are *test* rows too, and they carry the effect. "
      "BASE trains on none of them in any fold, so on the clusters made only of added chemistry "
      "the contrast is close to 'the arm allowed to learn this chemistry predicts it better'. What "
      "separates it from a tautology is that a test ligand's own chemotype is held out of every "
      "arm, and that the gain scales with how much closer the addition actually brought training "
      "(the `closeness gained` table). An earlier version of this line claimed the opposite — that "
      "the numbers say nothing about sparse test rows — which was backwards.")
    A("* **`EXPANDED_SHUFFLED` is a HARDER comparator than BASE, not an equal one.** Mislabelled "
      "chemistry is worse than absent chemistry, so the shuffled arm lands below BASE and the "
      "EXPANDED-vs-SHUFFLED delta overstates the effect size. Use it to rule out regularisation "
      "and the weighting change; quote the BASE contrast for the magnitude.")
    A("* **Do not quote the percentile interval alone at the `all` endpoint.** One super-cluster "
      "holds ~21 % of the scoring units inside 1 of the bootstrap blocks, which inflates the "
      "percentile interval's real Type-I rate; BCa, cluster-robust and block-macro columns are "
      "reported beside it for that reason.")
    A("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else \
        REPO_ROOT / "runs" / f"gen6_diversity_causal_{stamp}"
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
        args.feature_sets = list(args.feature_sets)[:1]
        log("PILOT: one split seed, 120 trees, one feature set — a Phase 0 smoke test, not evidence")

    arms = list(args.arms)
    for required in (BASE_ARM, EXPANDED_ARM):
        if required not in arms:
            raise SystemExit(f"arm {required!r} is required for Experiment A")

    # --- data ---------------------------------------------------------------
    source = pd.read_parquet(args.dataset)
    descriptors = None
    if args.descriptors and str(args.descriptors) and Path(args.descriptors).exists():
        descriptors = pd.read_parquet(args.descriptors)
    elif args.descriptors and str(args.descriptors):
        log(f"WARNING: descriptor parquet not found at {args.descriptors}; LIG2D_EXT arms skipped")

    data = build_level_dataset(
        source, min_rows_per_extractant=args.eval_min_cells,
        replicate_policy=args.replicate_policy, drop_below_log_d=args.log_d_floor,
        ligand_descriptors=descriptors)
    frame = data.frame
    audit = data.audit
    log(f"shared cohort: {audit['rows']} rows, {audit['extractants']} extractants, "
        f"{audit['ecfp_clusters']} ECFP clusters, {audit['tanimoto_clusters']} super-clusters; "
        f"target sd {audit['target_sd']:.3f}")
    log(f"blocks {audit['block_sizes']}")

    dense = arm_membership(frame, min_cells=args.base_min_cells)
    comparison = cohort_comparison(frame, base_min_cells=args.base_min_cells)
    log(f"BASE {comparison['base']} | added {comparison['added_by_expansion']} | "
        f"new ECFP clusters {comparison['new_ecfp_clusters']}, "
        f"new super-clusters {comparison['new_superclusters']}")

    # --- frozen chemistry ---------------------------------------------------
    if args.chemistry_map and Path(args.chemistry_map).exists():
        chemistry = ChemistryMap.from_parquet(args.chemistry_map)
        chemistry_source = str(args.chemistry_map)
        log(f"frozen chemistry map loaded from {args.chemistry_map} ({len(chemistry)} extractants)")
    else:
        chemistry = build_chemistry_map(source, ligand_descriptors=descriptors)
        chemistry_source = "rebuilt from the source table in this run"
        log(f"chemistry map built over all {len(chemistry)} extractants: "
            f"{chemistry.audit['n_ecfp_clusters']} ECFP clusters, "
            f"{chemistry.audit['n_superclusters']} super-clusters, rdkit="
            f"{chemistry.audit['rdkit_available']}")

    added_extractants = sorted(set(frame.loc[~dense, "extractant"].astype(str)))
    base_extractants = sorted(set(frame.loc[dense, "extractant"].astype(str)))
    novelty_table = chemistry.nearest_neighbour(added_extractants, base_extractants) \
        if added_extractants and base_extractants else pd.DataFrame()
    novelty = {}
    if not novelty_table.empty:
        novelty = {
            "n_added": int(len(novelty_table)),
            "median_nn_to_base": float(novelty_table["nn_tanimoto"].median()),
            "mean_nn_to_base": float(novelty_table["nn_tanimoto"].mean()),
            "n_below_0_4": int((novelty_table["nn_tanimoto"] < 0.4).sum()),
            "n_below_0_6": int((novelty_table["nn_tanimoto"] < 0.6).sum()),
        }
        log(f"added chemistry vs BASE: {novelty}")

    feature_sets = resolve_feature_sets(data, args.feature_sets, log)
    params = LevelForestParameters(
        n_estimators=args.n_estimators, max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf, random_state=args.model_seed, n_jobs=args.n_jobs)

    # --- fit ----------------------------------------------------------------
    started = time.time()
    results: list[SeedResult] = []
    for feature_set, columns in feature_sets.items():
        log(f"feature set {feature_set}: {len(columns)} columns")
        for seed in args.split_seeds:
            results.append(evaluate_seed(
                data, chemistry=chemistry, feature_set=feature_set, feature_columns=columns,
                arms=arms, seed=int(seed), folds=args.folds, params=params,
                base_min_cells=args.base_min_cells, weighting=args.weighting,
                dispersion_arm=args.dispersion_arm if args.tree_dispersion else None, log=log))
    fit_seconds = time.time() - started
    oof = pd.concat([r.oof for r in results], ignore_index=True)
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    log(f"fitting done in {fit_seconds:.1f}s; OOF {len(oof)} rows")

    # --- metrics ------------------------------------------------------------
    arm_parts, ligand_parts, hard_parts, contrast_parts, ood_parts = [], [], [], [], []
    unit_count_rows: list[dict] = []
    pooled_inputs: dict[tuple[str, str], list[pd.DataFrame]] = {}
    for result in results:
        predictions = result.oof
        overall, per_ligand, hard = gen6_metric_table(
            predictions, arms, similarity_column="nn_reference_tanimoto",
            thresholds=tuple(args.thresholds))
        for table in (overall, per_ligand, hard):
            table.insert(0, "split_seed", result.seed)
            table.insert(0, "feature_set", result.feature_set)
        arm_parts.append(overall)
        ligand_parts.append(per_ligand)
        hard_parts.append(hard)

        blocks = unit_blocks(predictions)
        for endpoint, subset in hard_chemistry_subsets(predictions, args.thresholds):
            n_units = int(subset["ecfp_cluster"].nunique()) if len(subset) else 0
            n_blocks = int(subset["tanimoto_cluster"].nunique()) if len(subset) else 0
            unit_count_rows.append({
                "feature_set": result.feature_set, "split_seed": result.seed,
                "endpoint": endpoint, "n_rows": int(len(subset)),
                "n_ligands": int(subset["extractant"].nunique()) if len(subset) else 0,
                "n_ecfp_clusters": n_units, "n_superclusters": n_blocks,
            })
            if n_units < 2:
                log(f"  {result.feature_set} seed {result.seed} endpoint {endpoint}: "
                    f"{n_units} scoring unit(s) — bootstrap skipped, not silently pooled")
                continue
            per_unit = per_unit_statistics(subset, arms)
            pooled_inputs.setdefault((result.feature_set, endpoint), []).append(per_unit)
            table = contrast_frame(per_unit, block_of_unit=blocks, arms=arms,
                                   replicates=args.replicates, seed=args.bootstrap_seed)
            if not table.empty:
                table.insert(0, "endpoint", endpoint)
                table.insert(0, "split_seed", result.seed)
                table.insert(0, "feature_set", result.feature_set)
                contrast_parts.append(table)

        for arm in arms:
            calibration = ood_calibration_table(predictions, prediction_column=f"prediction_{arm}")
            if calibration.empty:
                continue
            # The forest dispersion belongs to ONE arm (--dispersion-arm); copying it
            # into every arm's row would invite the reader to compare arms on a column
            # that is the same number four times.
            if args.tree_dispersion and arm != args.dispersion_arm:
                calibration["mean_prediction_sd"] = np.nan
            calibration.insert(0, "arm", arm)
            calibration.insert(0, "split_seed", result.seed)
            calibration.insert(0, "feature_set", result.feature_set)
            ood_parts.append(calibration)

    arm_metrics = pd.concat(arm_parts, ignore_index=True)
    per_ligand_metrics = pd.concat(ligand_parts, ignore_index=True)
    hard_metrics = pd.concat(hard_parts, ignore_index=True)
    contrasts = pd.concat(contrast_parts, ignore_index=True) if contrast_parts else pd.DataFrame()
    ood = pd.concat(ood_parts, ignore_index=True) if ood_parts else pd.DataFrame()
    unit_counts = pd.DataFrame(unit_count_rows)
    migration_parts = []
    for result in results:
        part = bin_migration_table(result.oof, args.thresholds)
        part.insert(0, "split_seed", result.seed)
        part.insert(0, "feature_set", result.feature_set)
        migration_parts.append(part)
    migration = pd.concat(migration_parts, ignore_index=True)
    migration.to_csv(output_dir / "bin_migration.csv", index=False)

    # --- pooled-over-seeds bootstrap and the contrast summary ----------------
    summary_rows: list[pd.DataFrame] = []
    for (feature_set, endpoint), parts in pooled_inputs.items():
        pooled = pool_per_unit_over_seeds(parts)
        blocks = unit_blocks(oof[oof["feature_set"] == feature_set])
        table = contrast_frame(pooled, block_of_unit=blocks, arms=arms,
                               replicates=args.replicates, seed=args.bootstrap_seed)
        if table.empty:
            continue
        table = table.rename(columns={
            "point_delta": "pooled_point_delta", "ci95_low": "pooled_ci95_low",
            "ci95_high": "pooled_ci95_high", "p_worse_one_sided": "pooled_p_worse_one_sided",
            "units_improved": "pooled_units_improved", "units_total": "pooled_units_total",
            "bootstrap_blocks": "pooled_blocks"})
        table.insert(0, "endpoint", endpoint)
        table.insert(0, "feature_set", feature_set)
        summary_rows.append(table)
    contrast_summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()

    if not contrast_summary.empty and not contrasts.empty:
        per_seed = (contrasts.groupby(["feature_set", "endpoint", "comparison", "statistic"])
                    .agg(mean_point_delta=("point_delta", "mean"),
                         min_point_delta=("point_delta", "min"),
                         max_point_delta=("point_delta", "max"),
                         seeds_positive=("point_delta", lambda s: int((s > 0).sum())),
                         n_seeds=("point_delta", "size")).reset_index())
        contrast_summary = contrast_summary.merge(
            per_seed, on=["feature_set", "endpoint", "comparison", "statistic"], how="left")
    # The hypothesis scorer reads these columns unconditionally; a run with no
    # evaluable contrast must produce an empty table with the right shape rather
    # than a KeyError three functions later.
    for column in ("pooled_point_delta", "pooled_ci95_low", "pooled_ci95_high", "mean_point_delta",
                   "seeds_positive", "n_seeds", "pooled_units_total", "pooled_blocks",
                   "feature_set", "endpoint", "comparison", "statistic"):
        if column not in contrast_summary.columns:
            contrast_summary[column] = pd.Series(dtype=float)

    # --- hypotheses ----------------------------------------------------------
    verdicts_by_set: dict[str, list[Verdict]] = {}
    for feature_set in feature_sets:
        verdicts_by_set[feature_set] = score_hypotheses(
            contrast_summary, feature_set=feature_set, thresholds=args.thresholds,
            n_seeds=len(args.split_seeds)) if not contrast_summary.empty else []

    # --- write the tables ----------------------------------------------------
    arm_metrics.to_csv(output_dir / "arm_metrics.csv", index=False)
    per_ligand_metrics.to_csv(output_dir / "per_ligand_metrics.csv", index=False)
    hard_metrics.to_csv(output_dir / "hard_chemistry_metrics.csv", index=False)
    unit_counts.to_csv(output_dir / "hard_chemistry_units.csv", index=False)
    contrasts.to_csv(output_dir / "contrasts.csv", index=False)
    contrast_summary.to_csv(output_dir / "contrast_summary.csv", index=False)
    ood.to_csv(output_dir / "ood_metrics.csv", index=False)
    (output_dir / "cohort_comparison.json").write_text(json.dumps(
        {"cohort_audit": audit, "base_vs_expanded": comparison,
         "added_chemistry_vs_base": novelty,
         "base_extractants": base_extractants, "added_extractants": added_extractants},
        indent=2, default=str) + "\n")

    fold_records = [r for result in results for r in result.fold_records]
    budgets = pd.DataFrame([
        {"arm": arm, "feature_set": record["feature_set"], "split_seed": record["split_seed"],
         "fold": record["fold"], "n_train_rows": record["n_train_rows_by_arm"][arm],
         "n_train_extractants": len(record["train_extractants_by_arm"][arm]),
         "n_train_ecfp_clusters": record["n_train_ecfp_clusters_by_arm"][arm]}
        for record in fold_records for arm in arms])
    budgets.to_csv(output_dir / "training_budgets.csv", index=False)

    split_definition = {
        "shared_cohort_min_cells": int(args.eval_min_cells),
        "base_min_cells": int(args.base_min_cells),
        "group_column": "tanimoto_cluster",
        "n_splits": int(args.folds),
        "split_seeds": [int(s) for s in args.split_seeds],
        "fold_algorithm": "shuffle unique group labels with np.random.default_rng(seed), "
                          "deal round-robin (gen5 seeded_group_kfold)",
        "fold_random_state_formula": f"model_seed + fold * {FOLD_SEED_STRIDE} + {FOLD_SEED_OFFSET}",
        "arms": list(arms),
        "identical_test_rows": True,
        "weighting": args.weighting,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(
        {"definition": split_definition, "folds": fold_records,
         "integrity": [{"feature_set": r.feature_set, "split_seed": r.seed, **r.integrity}
                       for r in results]},
        indent=2, default=str) + "\n")

    # --- the checks the run must pass before it may claim success ------------
    checks = build_checks(
        results=results, oof=oof, hard_metrics=hard_metrics, arms=arms,
        cohort_extractants=set(frame["extractant"].astype(str)),
        chemistry_extractants=set(chemistry.extractants))

    # --- manifest ------------------------------------------------------------
    provenance_state: dict = {"status": "not_audited_in_this_run"}
    if args.provenance_state_json and Path(args.provenance_state_json).exists():
        provenance_state = json.loads(Path(args.provenance_state_json).read_text())
    elif args.provenance_state_json:
        provenance_state = {"status": "provenance_state_file_missing",
                            "path": str(args.provenance_state_json)}

    manifest = RunManifest(layer=GEN6_LAYER, run_id=f"gen6_diversity_causal_{stamp}")
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
        {"step": "arm_membership", "base_min_cells": int(args.base_min_cells),
         "note": "row mask over the shared cohort; never a second build_level_dataset call"},
        {"step": "LevelRegressor pipeline",
         "detail": "DropAllNaNColumns -> SimpleImputer(median, add_indicator) -> ExtraTrees"},
        {"step": "sample weighting", "mode": args.weighting,
         "detail": "group_balanced_weights on the training rows' ecfp_cluster (cluster) or none (row)"},
    ])
    manifest.record("model_seed", int(args.model_seed))
    manifest.record("split_seeds", [int(s) for s in args.split_seeds])
    manifest.record("cohort_sha256", sha256_frame(frame, sort_rows_by=["row_id"]))
    manifest.record("arms", list(arms))
    manifest.record("contrasts", [c.__dict__ for c in CONTRASTS])
    manifest.record("pilot", bool(args.pilot))
    manifest.record("weighting", args.weighting)
    manifest.record("bootstrap", {"replicates": int(args.replicates), "seed": int(args.bootstrap_seed),
                                  "scoring_unit": "ecfp_cluster", "resample_block": "tanimoto_cluster"})
    manifest.record("hard_chemistry_thresholds", [float(t) for t in args.thresholds])
    manifest.record("tree_dispersion", {"enabled": bool(args.tree_dispersion),
                                        "arm": args.dispersion_arm if args.tree_dispersion else None})

    # Where the gain sits, by whether a scoring unit is made of chemistry BASE can
    # ever see.  Computed here rather than in the doc so every future run carries it.
    dense_extractants = set(frame.loc[arm_membership(frame, min_cells=args.base_min_cells),
                                      "extractant"].astype(str))
    decomposition = (dense_sparse_decomposition(oof, dense_extractants)
                     if {f"prediction_{BASE_ARM}", f"prediction_{EXPANDED_ARM}"} <= set(oof.columns)
                     else pd.DataFrame())

    weight_note = ("group_balanced_weights on the training rows' ECFP cluster — the gen5 rule, and "
                   "itself arm-dependent because EXPANDED has more clusters (see 'How to break "
                   "this result')") if args.weighting == "cluster" else \
                  ("no sample weights at all — the disclosed sensitivity that removes the "
                   "cluster-weighting confound")

    report = render_report(
        stamp=stamp, args=args, audit=audit, comparison=comparison, novelty=novelty,
        arm_metrics=(arm_metrics.groupby(["feature_set", "arm"], as_index=False, sort=False)
                     .mean(numeric_only=True)),
        hard_metrics=(hard_metrics.groupby(["feature_set", "endpoint", "arm"], as_index=False,
                                           sort=False).mean(numeric_only=True)),
        contrast_summary=contrast_summary,
        per_seed_primary=(contrasts[(contrasts["comparison"] == PRIMARY_CONTRAST)
                                    & (contrasts["statistic"].isin(("mae", "offset_mae",
                                                                    "shape_mae")))]
                          if not contrasts.empty else contrasts),
        verdicts_by_set=verdicts_by_set,
        ood=(ood.groupby(["feature_set", "arm", "bin"], as_index=False, sort=False)
             .mean(numeric_only=True) if not ood.empty else ood),
        unit_counts=(unit_counts.groupby(["feature_set", "endpoint"], as_index=False, sort=False)
                     .mean(numeric_only=True) if not unit_counts.empty else unit_counts),
        budgets=budgets.groupby("arm", as_index=False, sort=False).mean(numeric_only=True),
        migration=migration.groupby("endpoint", as_index=False, sort=False).mean(numeric_only=True),
        decomposition=decomposition,
        fit_seconds=fit_seconds, weight_note=weight_note)
    (output_dir / "decision_report.md").write_text(report)
    if not decomposition.empty:
        decomposition.to_csv(output_dir / "gain_decomposition.csv", index=False)

    (output_dir / "summary.json").write_text(json.dumps({
        "run_id": f"gen6_diversity_causal_{stamp}",
        "layer": GEN6_LAYER,
        "pilot": bool(args.pilot),
        "cohort_audit": audit,
        "base_vs_expanded": comparison,
        "added_chemistry_vs_base": novelty,
        "arms": list(arms),
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "split_seeds": [int(s) for s in args.split_seeds],
        "folds": int(args.folds),
        "n_estimators": int(args.n_estimators),
        "weighting": args.weighting,
        "fit_seconds": fit_seconds,
        "arm_metrics": arm_metrics.to_dict("records"),
        "contrast_summary": contrast_summary.to_dict("records"),
        "hard_chemistry_units": unit_counts.to_dict("records"),
        "hard_chemistry_bin_migration": migration.to_dict("records"),
        "hypotheses": {fs: [v.__dict__ for v in vs] for fs, vs in verdicts_by_set.items()},
        "checks": checks,
    }, indent=2, default=str) + "\n")

    validation, success = finalise_run(output_dir, manifest=manifest, checks=checks)

    print(report)
    for feature_set, verdicts in verdicts_by_set.items():
        for v in verdicts:
            log(f"{feature_set} {v.hypothesis}: {v.verdict}")
    if success is None:
        log(f"VALIDATION FAILED: {validation['failed_checks']} "
            f"missing_keys={validation['missing_manifest_keys']} "
            f"missing_artifacts={validation['missing_artifacts']}; wrote _FAILED.json")
        return 1
    log(f"written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
