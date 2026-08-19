"""gen6 Experiment B — diversity versus depth at an equal training-row budget.

Experiment A asks whether adding 61 sparse, chemically distant extractants to the
training cohort helps.  A sceptic can answer "of course it does, you gave it more
rows".  Experiment B removes that answer: every arm here is allowed **exactly the
same number of training rows**, and the only thing that changes is *which rows an
acquisition policy chooses to spend the budget on*.

Four label-free policies spend the same budget in different currencies:

``depth``
    the best-measured ligands first — what six generations of this project have
    actually been doing, because those are the ligands that clear
    ``min_rows_per_extractant``;
``diversity``
    one ligand from each Tanimoto-0.7 super-cluster in turn, so a previously
    absent chemotype enters before a second member of a present one;
``maxmin``
    greedy max-min walk of Tanimoto space (the classical "widest" heuristic —
    the protocol's required active-learning control);
``random``
    the null every acquisition heuristic has to beat.

The design decisions that make the curve readable, and the traps behind them:

* **One shared cohort, one set of folds.**  The cohort is built once at
  ``min_cells = 3`` and folds hold out whole super-clusters, exactly as in
  Experiment A and gen5.  A budget changes the training rows only; the test rows
  of a fold are a property of the fold.  Comparing two *cohorts* would confound
  "which rows did we train on" with "which rows did we score on".
* **A policy is a distribution over training sets, not one training set.**  Each
  (policy, budget) is therefore repeated with ``--draws`` independent acquisition
  RNG draws and the curve reports the mean and the spread.  A single draw of
  ``diversity`` beating a single draw of ``depth`` is not evidence.
* **The draws are nested across budgets, not re-drawn.**  The acquisition RNG is
  seeded on (split seed, fold, policy, draw) and deliberately **not** on the
  budget, so within one draw the 500-row training set is essentially the 250-row
  one plus more chemistry.  Re-drawing per budget would add pure sampling noise
  to the slope, which is precisely what B3 has to measure.
* **The hard-chemistry endpoint has no single reference here.**  In Experiment A
  every arm is scored on bins fixed by the BASE training set.  In Experiment B
  each curve point has its *own* training set, so "distance to training" moves
  with the point.  Both readings are reported and named accordingly:
  ``hard_own_train_nn<t>__*`` uses the similarity of the test ligand to **that
  point's own** training set (the honest description of what that model saw), and
  ``hard_vs_BASEtrain_nn<t>__*`` uses the frozen, arm-independent BASE bin so that
  two policies are compared on *identical rows*.  Only the second one may be used
  for a paired contrast; the first one is a description, not a comparison.
* **Budgets can exceed a fold's training pool.**  The largest super-cluster is
  ~64 % of rows, so the fold that holds it out trains on far fewer rows than the
  others.  Where a budget exceeds the pool every policy collapses to the same
  training set and the contrast is structurally zero for that fold.  This is
  recorded per fold (``budget_met``) and disclosed in the report rather than
  averaged away.
* **At budget ``all`` every policy and every draw must give the identical model.**
  That is asserted at run time (``all_budget_points_identical``) — it is a free
  end-to-end check that the acquisition layer is not perturbing anything else.

Pre-registered hypotheses scored here (protocol §5, Experiment B):

======  =============================================================  ================================
H       statement                                                      pass condition
======  =============================================================  ================================
**B1**  at equal row budget, diversity beats depth                     CI95 low > 0 at >= 2 consecutive budgets
**B2**  the advantage is larger on hard chemistry than overall         gain at nn < 0.4 > gain overall, both CI95 low > 0
**B3**  depth saturates                                                depth slope between the two largest budgets within noise of zero
======  =============================================================  ================================

Example::

    .venv/bin/python scripts/run_diversity_learning_curve.py --pilot
    .venv/bin/python scripts/run_diversity_learning_curve.py \\
        --budgets 250 500 1000 2000 3000 all --draws 3 \\
        --split-seeds 104729 130363 155921 196613 262147
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.chemistry import (  # noqa: E402
    ChemistryMap, build_chemistry_map, cluster_manifest,
)
from lanthanide_separation.gen6.cohorts import (  # noqa: E402
    ACQUISITION_POLICIES, BASE_ARM, BASE_MIN_CELLS, EXPANDED_ARM, EXPANDED_MIN_CELLS,
    acquisition_audit, assert_split_integrity, diversity_splits, row_budget_subsample,
)
from lanthanide_separation.gen6.manifest import (  # noqa: E402
    RunManifest, sha256_frame, sha256_json, sha256_text, validate_run, write_success,
)
from lanthanide_separation.gen6.metrics import (  # noqa: E402
    HARD_CHEMISTRY_THRESHOLDS, gen6_metric_table, hard_chemistry_endpoints,
    paired_unit_bootstrap, per_unit_statistics,
)
from lanthanide_separation.levels import (  # noqa: E402
    LEVEL_ARMS, LEVEL_TARGET_COLUMN, LevelForestParameters, LevelRegressor, build_level_dataset,
)

DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
DESCRIPTOR_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_2d_descriptors.parquet"

#: The gen5 champion feature set (METAL + COND + LIG2D_EXT + MASSACTION).  Phase 1
#: freezes the learner and the features; only the training rows may vary.
CHAMPION_ARM = "MC_lig2d_ext_massaction"
#: The gen5 split seeds, reused so folds are comparable across the generation.
DEFAULT_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
DEFAULT_BUDGETS: tuple[str, ...] = ("250", "500", "1000", "2000", "3000", "all")
#: Salt for the acquisition RNG; keeps acquisition draws disjoint from the fold
#: RNG stream and from Experiment A's row-matching stream.
ACQUISITION_SEED_SALT = 606_060_607
#: Statistics the paired bootstrap resamples (all are per-unit means, so a mean
#: over resampled units is an exact bootstrap of the macro value).
BOOTSTRAP_STATISTICS: tuple[str, ...] = ("mae", "offset_mae", "shape_mae")

POLICY_CODE: dict[str, int] = {policy: i for i, policy in enumerate(ACQUISITION_POLICIES)}


# --------------------------------------------------------------------------- #
# Curve points
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CurvePoint:
    """One (policy, budget, acquisition draw) cell of the learning curve."""

    policy: str
    budget: int | None          # None == "all rows in the fold's training pool"
    draw: int

    @property
    def budget_label(self) -> str:
        return "all" if self.budget is None else str(self.budget)

    @property
    def arm(self) -> str:
        """Label without the draw — the unit the paired bootstrap compares."""
        return f"{self.policy}@{self.budget_label}"

    @property
    def label(self) -> str:
        return f"{self.arm}#d{self.draw}"


def parse_budget(token: str) -> int | None:
    """``"all"`` -> ``None`` (the whole training pool); otherwise a positive int."""
    text = str(token).strip().lower()
    if text == "all":
        return None
    value = int(text)
    if value <= 0:
        raise ValueError(f"budget must be positive or 'all', got {token!r}")
    return value


def curve_points(policies, budgets, draws: int) -> list[CurvePoint]:
    """Every (policy, budget, draw) cell, in a deterministic order."""
    return [CurvePoint(policy=p, budget=b, draw=d)
            for p in policies for b in budgets for d in range(draws)]


def acquisition_rng(split_seed: int, fold: int, policy: str, draw: int) -> np.random.Generator:
    """RNG for one acquisition draw.

    The budget is deliberately **absent** from the seed: within one draw every
    budget then walks the same policy ordering, so the curve is nested and its
    slope is not contaminated by re-drawing the acquisition order at each budget.
    """
    return np.random.default_rng(
        [int(split_seed), int(fold), POLICY_CODE[policy], int(draw), ACQUISITION_SEED_SALT])


# --------------------------------------------------------------------------- #
# Chemistry helpers
# --------------------------------------------------------------------------- #

def max_similarity_to_reference(
    chemistry: ChemistryMap, query, reference, *, exclude_self: bool = True,
) -> np.ndarray:
    """Max Tanimoto from each query extractant to any reference extractant.

    Numerically identical to
    ``ChemistryMap.nearest_neighbour(query, reference)["nn_tanimoto"]`` — asserted
    in ``tests/test_gen6_learning_curve.py`` so the two cannot drift — but
    vectorised.  The public method resolves every name through ``tuple.index``
    inside a per-query Python loop, which is O(n_query · n_reference · n_all) and
    would dominate this run, where the column is rebuilt once per
    (fold, policy, budget, draw).

    An empty reference gives 0.0, never NaN: "nothing similar in training" is a
    legitimate and important state, and a NaN there would quietly drop the
    hardest ligands out of every downstream average.
    """
    position = {name: i for i, name in enumerate(chemistry.extractants)}
    q_names = [str(x) for x in query]
    r_names = [str(x) for x in reference]
    if not q_names:
        return np.zeros(0, dtype=float)
    missing = [n for n in set(q_names) | set(r_names) if n not in position]
    if missing:
        raise KeyError(f"{len(missing)} extractants are absent from the frozen chemistry map, "
                       f"e.g. {missing[:2]}")
    if not r_names:
        return np.zeros(len(q_names), dtype=float)
    block = chemistry.similarity[np.ix_([position[n] for n in q_names],
                                        [position[n] for n in r_names])].astype(float)
    if exclude_self:
        same = np.equal.outer(np.array(q_names, dtype=object), np.array(r_names, dtype=object))
        block = np.where(same, -1.0, block)
    best = block.max(axis=1)
    return np.where(best < 0.0, 0.0, best)


def select_training_rows(
    frame: pd.DataFrame, pool: np.ndarray, point: CurvePoint, *,
    chemistry: ChemistryMap, split_seed: int, fold: int,
) -> np.ndarray:
    """Draw one training set for one curve point from the fold's full training pool.

    Delegates to :func:`gen6.cohorts.row_budget_subsample`; this wrapper only owns
    the RNG derivation and the similarity plumbing for ``maxmin``.
    """
    return row_budget_subsample(
        frame, pool, budget=point.budget, policy=point.policy,
        rng=acquisition_rng(split_seed, fold, point.policy, point.draw),
        similarity=chemistry.similarity, similarity_names=chemistry.extractants,
        supercluster_column="tanimoto_cluster")


def selection_is_label_free(
    frame: pd.DataFrame, pool: np.ndarray, points, *,
    chemistry: ChemistryMap, split_seed: int, fold: int, seed: int = 20260819,
) -> dict:
    """Run-time proof that no policy peeked at ``log_D``.

    Permutes the target column of the cohort frame and re-runs every selection.
    A policy that used the target would move; all four must return byte-identical
    row indices.  The unit test asserts the same property on synthetic data; this
    check asserts it on the *real* cohort inside the run that publishes the
    numbers, which is where it actually matters.
    """
    permuted = frame.copy()
    rng = np.random.default_rng(seed)
    permuted[LEVEL_TARGET_COLUMN] = frame[LEVEL_TARGET_COLUMN].to_numpy()[rng.permutation(len(frame))]
    moved: list[str] = []
    for point in points:
        a = select_training_rows(frame, pool, point, chemistry=chemistry,
                                 split_seed=split_seed, fold=fold)
        b = select_training_rows(permuted, pool, point, chemistry=chemistry,
                                 split_seed=split_seed, fold=fold)
        if not np.array_equal(a, b):
            moved.append(point.label)
    return {"ok": not moved, "n_points_checked": len(list(points)), "points_that_moved": moved,
            "fold": int(fold), "split_seed": int(split_seed)}


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

class FoldModelCache:
    """Fit-and-predict with memoisation on the *training row set*.

    Different curve points frequently resolve to the identical training set — at
    budget ``all`` every policy and draw does, and any budget that exceeds a
    fold's pool does too.  Fitting them separately would burn minutes and produce
    bit-identical predictions (the learner's seed depends on the fold, not on the
    point), so the cache is a pure speed-up, not an approximation: the key is the
    exact sorted training index.
    """

    def __init__(self, feature_columns, params: LevelForestParameters) -> None:
        self.feature_columns = tuple(feature_columns)
        self.params = params
        self._cache: dict[tuple, np.ndarray] = {}
        self.n_fits = 0
        self.n_hits = 0

    @staticmethod
    def selection_key(index: np.ndarray) -> str:
        return sha256_text(",".join(str(int(i)) for i in np.asarray(index)))

    def predict(
        self, frame: pd.DataFrame, train_index: np.ndarray, test_index: np.ndarray, *,
        split_seed: int, fold: int,
    ) -> np.ndarray:
        key = (int(split_seed), int(fold), self.selection_key(train_index))
        cached = self._cache.get(key)
        if cached is not None:
            self.n_hits += 1
            return cached
        train = frame.iloc[train_index]
        test = frame.iloc[test_index]
        # Fold seed formula copied from gen5 (run_gen5_levels.evaluate_regime) so a
        # point at budget "all" reproduces the Experiment A EXPANDED arm exactly.
        fold_params = LevelForestParameters(
            n_estimators=self.params.n_estimators, max_features=self.params.max_features,
            min_samples_leaf=self.params.min_samples_leaf,
            random_state=self.params.random_state + fold * 1009 + 9_999_991,
            n_jobs=self.params.n_jobs)
        model = LevelRegressor(self.feature_columns, fold_params).fit(
            train, train[LEVEL_TARGET_COLUMN].to_numpy(float), groups=train["ecfp_cluster"])
        prediction = np.asarray(model.predict(test), dtype=float)
        self._cache[key] = prediction
        self.n_fits += 1
        return prediction


# --------------------------------------------------------------------------- #
# Scoring one curve point
# --------------------------------------------------------------------------- #

IDENTITY_COLUMNS = ("row_id", "extractant", "ecfp_cluster", "tanimoto_cluster",
                    "metal_symbol", LEVEL_TARGET_COLUMN)

#: Overall statistics copied straight out of ``gen6.metrics.gen6_metric_table``.
OVERALL_KEYS: tuple[str, ...] = (
    "macro_mae", "offset_mae", "shape_mae", "shape_r2", "macro_mae_ligand", "pooled_mae",
    "pooled_r2", "offset_share_of_sse", "median_ligand_mae", "worst_quartile_ligand_mae",
    "rank_spearman", "sign_accuracy", "frac_within_0_5_log", "frac_within_1_log",
    "prediction_dispersion_ratio", "n_rows", "n_ligands", "n_ecfp_clusters", "n_superclusters",
    "n_macro_units",
    "n_eff_pooled_rows",
)
HARD_KEYS: tuple[str, ...] = ("macro_mae", "offset_mae", "shape_mae", "macro_mae_ligand",
                              "n_rows", "n_ligands", "n_ecfp_clusters", "n_superclusters")


def score_point(
    identity: pd.DataFrame, prediction: np.ndarray, nn_own: np.ndarray, nn_base: np.ndarray,
    *, thresholds=HARD_CHEMISTRY_THRESHOLDS,
) -> dict:
    """All endpoints of one curve point, over the fold-aggregated out-of-fold rows.

    Two hard-chemistry families are returned and they answer different questions:

    ``hard_own_train_nn<t>``
        similarity to **this point's own** training set.  It describes what the
        model in front of you actually faced, and it is the honest learning-curve
        endpoint — but the row subset moves with the point, so two policies'
        numbers here are not computed on the same rows and must not be subtracted.
    ``hard_vs_BASEtrain_nn<t>``
        similarity to the fold's BASE (>= 10 cells) training rows: frozen,
        policy-independent, identical rows for every point.  This is the bin the
        paired contrasts use.
    """
    frame = identity.copy()
    frame["prediction_point"] = prediction
    frame["nn_own_train_tanimoto"] = nn_own
    frame["nn_base_train_tanimoto"] = nn_base
    overall, _, hard_own = gen6_metric_table(
        frame, ["point"], similarity_column="nn_own_train_tanimoto", thresholds=thresholds)
    hard_base = hard_chemistry_endpoints(
        frame, prediction_column="prediction_point",
        similarity_column="nn_base_train_tanimoto", thresholds=thresholds)

    row: dict = {key: overall.iloc[0][key] for key in OVERALL_KEYS if key in overall.columns}
    for table, prefix in ((hard_own, "hard_own_train"), (hard_base, "hard_vs_BASEtrain")):
        for _, entry in table.iterrows():
            if entry["endpoint"] == "all":
                continue
            tag = str(entry["endpoint"]).replace("<", "")     # "nn<0.4" -> "nn0.4"
            for key in HARD_KEYS:
                if key in table.columns:
                    row[f"{prefix}_{tag}__{key}"] = entry[key]
    return row


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DATASET_PATH)
    p.add_argument("--descriptors", type=Path, default=DESCRIPTOR_PATH,
                   help="gen4 extended-2D descriptor parquet (the LIG2D_EXT block)")
    p.add_argument("--chemistry-map", type=Path, default=None,
                   help="frozen chemistry_map.parquet from scripts/build_chemistry_map.py; "
                        "rebuilt in-process when omitted")
    p.add_argument("--feature-set", default=CHAMPION_ARM, choices=list(LEVEL_ARMS),
                   help="frozen for Phase 1; only the training rows may vary")
    p.add_argument("--policies", nargs="+", default=list(ACQUISITION_POLICIES),
                   choices=list(ACQUISITION_POLICIES))
    p.add_argument("--budgets", nargs="+", default=list(DEFAULT_BUDGETS),
                   help="training-row budgets; 'all' means the fold's whole training pool")
    p.add_argument("--draws", type=int, default=3,
                   help="independent acquisition RNG draws per (policy, budget)")
    p.add_argument("--split-seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--eval-min-cells", type=int, default=EXPANDED_MIN_CELLS,
                   help="shared-cohort eligibility; 3 is the gen6 EXPANDED rule")
    p.add_argument("--base-min-cells", type=int, default=BASE_MIN_CELLS,
                   help="the gen5 rule, used only to define the frozen hard-chemistry bins")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-features", type=float, default=0.30)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--replicate-policy", default="mean", choices=["mean", "unique", "all"])
    p.add_argument("--log-d-floor", type=float, default=-6.0)
    p.add_argument("--replicates", type=int, default=5000, help="bootstrap replicates")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--pilot", action="store_true",
                   help="fast smoke: one split seed, one draw, budgets 500/2000/all, 120 trees")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or REPO_ROOT / "runs" / f"gen6_learning_curve_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    seeds = list(args.split_seeds)
    policies = list(args.policies)
    budgets = [parse_budget(b) for b in args.budgets]
    draws = int(args.draws)
    n_estimators = int(args.n_estimators)
    replicates = int(args.replicates)
    if args.pilot:
        seeds, draws, n_estimators = seeds[:1], 1, 120
        budgets = [500, 2000, None]
        replicates = min(replicates, 2000)
    if draws < 1:
        raise SystemExit("--draws must be at least 1: a policy is a distribution over training "
                         "sets, and zero draws would produce an empty curve")
    # Deterministic budget order, smallest first, 'all' last — the report reads as
    # a curve and B3 needs "the two largest budgets" to be unambiguous.
    budgets = sorted({b for b in budgets}, key=lambda b: (b is None, b if b is not None else 0))
    points = curve_points(policies, budgets, draws)

    log(f"gen6 Experiment B — diversity vs depth at equal row budget ({stamp})")
    log(f"policies {policies}; budgets {[('all' if b is None else b) for b in budgets]}; "
        f"draws {draws}; seeds {seeds}; folds {args.folds}; trees {n_estimators}")

    # ---- data ------------------------------------------------------------- #
    source = pd.read_parquet(args.dataset)
    # ``--descriptors ''`` is the documented way to run without the gen4 extended-2D
    # table; argparse turns it into Path('.'), which *exists*, so an existence test
    # alone would try to read a directory as parquet.  Require an actual file.
    descriptor_path: Path | None = None
    if args.descriptors is not None and str(args.descriptors) not in ("", "."):
        descriptor_path = Path(args.descriptors)
    descriptors = None
    if descriptor_path is not None and descriptor_path.is_file():
        descriptors = pd.read_parquet(descriptor_path)
    else:
        log(f"WARNING: no descriptor parquet at {args.descriptors!s}; the LIG2D_EXT block will be "
            f"absent and any arm that needs it will stop the run")

    if args.chemistry_map is not None:
        chemistry = ChemistryMap.from_parquet(args.chemistry_map)
        chemistry_source = str(args.chemistry_map)
    else:
        chemistry = build_chemistry_map(source, ligand_descriptors=descriptors)
        chemistry_source = "built in-process from the source table"
    log(f"chemistry map: {len(chemistry)} extractants, "
        f"{chemistry.audit['n_ecfp_clusters']} ECFP clusters, "
        f"{chemistry.audit['n_superclusters']} super-clusters ({chemistry_source})")

    data = build_level_dataset(
        source, min_rows_per_extractant=args.eval_min_cells,
        replicate_policy=args.replicate_policy, drop_below_log_d=args.log_d_floor,
        ligand_descriptors=descriptors)
    frame = data.frame.reset_index(drop=True)
    feature_columns = data.arm_columns(args.feature_set)
    log(f"shared cohort: {data.audit['rows']} rows, {data.audit['extractants']} extractants, "
        f"{data.audit['ecfp_clusters']} ECFP clusters, {data.audit['tanimoto_clusters']} super-clusters; "
        f"target sd {data.audit['target_sd']:.3f}")
    log(f"feature set {args.feature_set}: {len(feature_columns)} columns "
        f"from blocks {list(LEVEL_ARMS[args.feature_set])}")
    missing_blocks = [b for b in LEVEL_ARMS[args.feature_set] if b not in data.blocks]
    if missing_blocks:
        raise SystemExit(f"feature set {args.feature_set} needs blocks {missing_blocks}, "
                         f"which this cohort does not have: {sorted(data.blocks)}")

    identity = frame[list(IDENTITY_COLUMNS)].copy()
    params = LevelForestParameters(
        n_estimators=n_estimators, max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf, random_state=args.model_seed, n_jobs=args.n_jobs)

    # ---- fit every (seed, fold, point) ------------------------------------ #
    t0 = time.time()
    curve_rows: list[dict] = []
    audit_rows: list[dict] = []
    fold_records: list[dict] = []
    selection_hashes: dict[str, dict[str, str]] = {}
    per_unit_parts: list[pd.DataFrame] = []
    per_unit_hard_parts: list[pd.DataFrame] = []
    oof_parts: list[pd.DataFrame] = []
    label_free_check: dict = {"ok": None, "status": "not run"}
    integrity: dict = {}
    short_budget_folds: list[dict] = []

    for seed in seeds:
        splits = diversity_splits(
            frame, group_column="tanimoto_cluster", n_splits=args.folds, seed=seed,
            base_min_cells=args.base_min_cells, arms=(BASE_ARM, EXPANDED_ARM))
        report = assert_split_integrity(frame, splits)
        integrity[str(seed)] = report
        if not report["ok"]:
            log(f"WARNING: split integrity failed for seed {seed}: see validation.json")

        prediction_by_point = {p.label: np.full(len(frame), np.nan) for p in points}
        nn_own_by_point = {p.label: np.full(len(frame), np.nan) for p in points}
        nn_base = np.full(len(frame), np.nan)
        cache = FoldModelCache(feature_columns, params)

        for split in splits:
            fold = split.fold
            pool = split.train_index_by_arm[EXPANDED_ARM]
            base_train = split.train_index_by_arm[BASE_ARM]
            test_index = split.test_index
            test_ext = frame["extractant"].to_numpy()[test_index]
            base_ext = pd.unique(frame["extractant"].to_numpy()[base_train])
            nn_base[test_index] = max_similarity_to_reference(chemistry, test_ext, base_ext)

            if label_free_check["ok"] is None:
                # Once per run, on the real cohort: permute the target and prove
                # that not one selection moves.
                label_free_check = selection_is_label_free(
                    frame, pool, points, chemistry=chemistry, split_seed=seed, fold=fold)
                log(f"label-free selection check on seed {seed} fold {fold}: "
                    f"{'PASS' if label_free_check['ok'] else 'FAIL ' + str(label_free_check)}")

            fold_selection_hashes: dict[str, str] = {}
            for point in points:
                selection = select_training_rows(
                    frame, pool, point, chemistry=chemistry, split_seed=seed, fold=fold)
                fold_selection_hashes[point.label] = cache.selection_key(selection)
                prediction_by_point[point.label][test_index] = cache.predict(
                    frame, selection, test_index, split_seed=seed, fold=fold)
                train_ext = pd.unique(frame["extractant"].to_numpy()[selection])
                nn_own_by_point[point.label][test_index] = max_similarity_to_reference(
                    chemistry, test_ext, train_ext)

                audit = acquisition_audit(frame, selection)
                budget_met = point.budget is None or len(selection) == point.budget
                if not budget_met:
                    short_budget_folds.append(
                        {"split_seed": seed, "fold": fold, "policy": point.policy,
                         "budget": point.budget_label, "draw": point.draw,
                         "pool_rows": int(len(pool)), "selected_rows": int(len(selection))})
                audit_rows.append({
                    "split_seed": seed, "fold": fold, "policy": point.policy,
                    "budget": point.budget_label, "draw": point.draw,
                    "pool_rows": int(len(pool)), "budget_met": bool(budget_met),
                    "selection_sha256": fold_selection_hashes[point.label], **audit})

            selection_hashes[f"seed{seed}_fold{fold}"] = fold_selection_hashes
            fold_records.append({
                "fold": int(fold), "split_seed": int(seed),
                "test_row_ids_sha256": sha256_text("|".join(
                    sorted(frame["row_id"].to_numpy()[test_index]))),
                "test_extractants": sorted(set(frame["extractant"].to_numpy()[test_index])),
                "test_superclusters": list(split.held_out_groups),
                "train_extractants_by_arm": {
                    "POOL": sorted(set(frame["extractant"].to_numpy()[pool])),
                    BASE_ARM: sorted(set(frame["extractant"].to_numpy()[base_train]))},
                "train_superclusters_by_arm": {
                    "POOL": sorted(set(frame["tanimoto_cluster"].to_numpy()[pool])),
                    BASE_ARM: sorted(set(frame["tanimoto_cluster"].to_numpy()[base_train]))},
                "n_test_rows": int(test_index.size),
                "n_train_rows_by_arm": {"POOL": int(pool.size), BASE_ARM: int(base_train.size)},
                "train_row_ids_sha256_by_arm": {
                    "POOL": sha256_text("|".join(sorted(frame["row_id"].to_numpy()[pool]))),
                    BASE_ARM: sha256_text("|".join(sorted(frame["row_id"].to_numpy()[base_train])))},
                "n_curve_points": len(points),
            })
            log(f"  seed {seed} fold {fold}: test {test_index.size} rows, pool {pool.size} rows, "
                f"{len(points)} points, {cache.n_fits} fits so far ({cache.n_hits} cache hits)")

        # ---- score this seed's out-of-fold predictions --------------------- #
        wide = identity.copy()
        wide["split_seed"] = seed
        wide["nn_base_train_tanimoto"] = nn_base
        for point in points:
            wide[f"prediction_{point.label}"] = prediction_by_point[point.label]
            wide[f"nn_own_{point.label}"] = nn_own_by_point[point.label]
        oof_parts.append(wide)

        for point in points:
            prediction = prediction_by_point[point.label]
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"{int((~np.isfinite(prediction)).sum())} rows were never "
                                   f"predicted for point {point.label} (seed {seed})")
            row = {"split_seed": seed, "policy": point.policy, "budget": point.budget_label,
                   "budget_rows": np.nan if point.budget is None else int(point.budget),
                   "draw": point.draw, "feature_set": args.feature_set,
                   "n_folds": len(splits)}
            audit_here = [a for a in audit_rows
                          if a["split_seed"] == seed and a["policy"] == point.policy
                          and a["budget"] == point.budget_label and a["draw"] == point.draw]
            row.update({
                "train_rows_total": int(sum(a["n_rows"] for a in audit_here)),
                "train_rows_mean": float(np.mean([a["n_rows"] for a in audit_here])),
                "train_extractants_mean": float(np.mean([a["n_extractants"] for a in audit_here])),
                "train_ecfp_clusters_mean": float(np.mean([a["n_ecfp_clusters"] for a in audit_here])),
                "train_superclusters_mean": float(np.mean([a["n_superclusters"] for a in audit_here])),
                "train_largest_extractant_share_mean": float(
                    np.mean([a["largest_extractant_share"] for a in audit_here])),
                "budget_met_all_folds": bool(all(a["budget_met"] for a in audit_here)),
            })
            row.update(score_point(identity, prediction, nn_own_by_point[point.label], nn_base))
            curve_rows.append(row)

        # ---- per-unit tables for the paired bootstrap ---------------------- #
        arms = [p.label for p in points]
        per_unit = per_unit_statistics(wide, arms, unit_column="ecfp_cluster")
        per_unit["split_seed"] = seed
        per_unit_parts.append(per_unit)
        hard_rows = wide[wide["nn_base_train_tanimoto"] < HARD_CHEMISTRY_THRESHOLDS[0]]
        if len(hard_rows):
            per_unit_hard = per_unit_statistics(hard_rows, arms, unit_column="ecfp_cluster")
            per_unit_hard["split_seed"] = seed
            per_unit_hard_parts.append(per_unit_hard)
        log(f"seed {seed} scored: {cache.n_fits} fits, {cache.n_hits} cache hits, "
            f"{time.time() - t0:.0f}s elapsed")

    curve = pd.DataFrame(curve_rows)
    audit_table = pd.DataFrame(audit_rows)
    oof = pd.concat(oof_parts, ignore_index=True)
    log(f"fitting and scoring done in {time.time() - t0:.1f}s")

    # ---- paired bootstrap, DIVERSITY vs DEPTH at each budget --------------- #
    # The per-unit error of a *policy* is averaged over draws (and split seeds)
    # first: a policy is a distribution over training sets, so its expected
    # per-unit error is the mean over draws.  Averaging the *predictions* instead
    # would build an ensemble, which is a different (and better) model than any
    # policy actually produces.
    def collapse(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        table = pd.concat(parts, ignore_index=True)
        table["policy_budget"] = table["arm"].str.split("#d").str[0]
        return (table.groupby(["unit", "policy_budget"], as_index=False)[list(BOOTSTRAP_STATISTICS)]
                .mean().rename(columns={"policy_budget": "arm"}))

    unit_to_block = (frame.drop_duplicates("ecfp_cluster")
                     .set_index("ecfp_cluster")["tanimoto_cluster"].astype(str).to_dict())
    comparisons: dict[str, tuple[str, str]] = {}
    budget_labels = ["all" if b is None else str(b) for b in budgets]
    for label in budget_labels:
        if "depth" in policies and "diversity" in policies:
            comparisons[f"diversity_vs_depth@{label}"] = (f"depth@{label}", f"diversity@{label}")
        if "random" in policies and "diversity" in policies:
            comparisons[f"diversity_vs_random@{label}"] = (f"random@{label}", f"diversity@{label}")
        if "maxmin" in policies and "depth" in policies:
            comparisons[f"maxmin_vs_depth@{label}"] = (f"depth@{label}", f"maxmin@{label}")
    # B3: is a curve still moving between its two largest budgets?  The
    # pre-registered reading uses the two largest budgets as given, which usually
    # means "3000 versus everything available".  ``all`` is a *fold-dependent*
    # budget — the fold that holds out the giant super-cluster trains on a small
    # fraction of the rows the other folds get (the realised pool sizes are in
    # acquisition_audit.csv) — so the same slope is also computed between the two
    # largest *fixed* budgets, where the x spacing is known.  Both are reported;
    # only the pre-registered one is scored.
    finite_labels = [label for label in budget_labels if label != "all"]
    slope_pairs = []
    if len(budget_labels) >= 2:
        slope_pairs.append((budget_labels[-2], budget_labels[-1]))
    if len(finite_labels) >= 2:
        slope_pairs.append((finite_labels[-2], finite_labels[-1]))
    for lo, hi in slope_pairs:
        for policy in ("depth", "diversity"):
            if policy in policies:
                comparisons[f"{policy}_slope_{lo}_to_{hi}"] = (f"{policy}@{lo}", f"{policy}@{hi}")

    boot_parts: list[pd.DataFrame] = []
    for parts, endpoint in ((per_unit_parts, "overall"),
                            (per_unit_hard_parts, f"hard_vs_BASEtrain_nn<{HARD_CHEMISTRY_THRESHOLDS[0]:g}")):
        collapsed = collapse(parts)
        if collapsed.empty or not comparisons:
            continue
        table = paired_unit_bootstrap(
            collapsed, comparisons, statistics=BOOTSTRAP_STATISTICS,
            block_of_unit=unit_to_block, replicates=replicates)
        table.insert(0, "endpoint", endpoint)
        boot_parts.append(table)
    bootstrap = pd.concat(boot_parts, ignore_index=True) if boot_parts else pd.DataFrame()

    # ---- artifacts --------------------------------------------------------- #
    curve.to_csv(output_dir / "diversity_learning_curve.csv", index=False)
    audit_table.to_csv(output_dir / "acquisition_audit.csv", index=False)
    if not bootstrap.empty:
        bootstrap.to_csv(output_dir / "learning_curve_bootstrap.csv", index=False)
    oof.to_parquet(output_dir / "oof_predictions.parquet", index=False)
    (output_dir / "acquisition_selections.json").write_text(
        json.dumps(selection_hashes, indent=2) + "\n")

    # ---- pre-registered scoring ------------------------------------------- #
    verdicts = score_hypotheses(curve, bootstrap, budget_labels, policies)
    checks = build_checks(curve, audit_table, integrity, label_free_check,
                          short_budget_folds, budgets, policies, draws, seeds)
    report = decision_report(
        stamp=stamp, args=args, data=data, chemistry=chemistry, curve=curve,
        audit_table=audit_table, bootstrap=bootstrap, verdicts=verdicts, checks=checks,
        budget_labels=budget_labels, policies=policies, draws=draws, seeds=seeds,
        feature_columns=feature_columns, short_budget_folds=short_budget_folds,
        n_estimators=n_estimators, replicates=replicates)
    (output_dir / "decision_report.md").write_text(report + "\n")
    print(report)

    (output_dir / "summary.json").write_text(json.dumps({
        "stamp": stamp, "experiment": "gen6_B_diversity_vs_depth",
        "cohort_audit": data.audit, "chemistry_audit": chemistry.audit,
        "feature_set": args.feature_set, "n_feature_columns": len(feature_columns),
        "policies": policies, "budgets": budget_labels, "draws": draws, "seeds": seeds,
        "folds": args.folds, "n_estimators": n_estimators, "replicates": replicates,
        "hypotheses": verdicts, "checks": checks,
        "curve": curve.to_dict("records"),
        "bootstrap": bootstrap.to_dict("records") if not bootstrap.empty else [],
    }, indent=2, default=float) + "\n")

    provenance_available = importlib.util.find_spec(
        "lanthanide_separation.gen6.provenance") is not None
    manifest = RunManifest(layer="gen6", run_id=f"gen6_learning_curve_{stamp}")
    manifest.record_dataset(dataset_path=args.dataset, source_frame=source,
                            descriptor_path=descriptor_path, descriptor_frame=descriptors)
    manifest.record_code([REPO_ROOT / "src" / "lanthanide_separation" / "gen6",
                          REPO_ROOT / "src" / "lanthanide_separation" / "levels.py",
                          Path(__file__).resolve()], repo_root=REPO_ROOT)
    manifest.record_features(feature_sets={args.feature_set: list(feature_columns)})
    manifest.record_chemistry(definition={
        **cluster_manifest(chemistry)["definition"],
        "source": chemistry_source,
        "chemistry_table_sha256": sha256_frame(chemistry.table),
        "n_extractants": len(chemistry), "audit": chemistry.audit})
    manifest.record_provenance(state={
        "variant": "legacy_compatible",
        "definition": "every row of the shared cohort; no provenance-based row filter is applied "
                      "in Experiment B, because the acquisition contrast is within-cohort",
        "provenance_module_available": bool(provenance_available)})
    manifest.record_preprocessing([
        {"step": "build_level_dataset", "min_rows_per_extractant": int(args.eval_min_cells),
         "replicate_policy": args.replicate_policy, "drop_below_log_d": float(args.log_d_floor),
         "ligand_descriptors": bool(descriptors is not None)},
        {"step": "LevelRegressor", "learner": "extratrees", "n_estimators": n_estimators,
         "max_features": float(args.max_features), "min_samples_leaf": int(args.min_samples_leaf),
         "imputation": "median + missing indicator, fitted inside the fold",
         "sample_weights": "group_balanced_weights on the training rows' ecfp_cluster"},
    ])
    manifest.record_split(
        definition={
            "experiment": "B_diversity_vs_depth",
            "cohort": f"one shared cohort at min_cells={args.eval_min_cells}",
            "fold_algorithm": "gen5 seeded_group_kfold: shuffle unique group labels, deal round-robin",
            "group_column": "tanimoto_cluster", "n_splits": int(args.folds),
            "training_pool": f"{EXPANDED_ARM} (every training-fold row)",
            "hard_chemistry_reference": f"{BASE_ARM} training rows (>= {args.base_min_cells} cells)",
            "policies": policies, "budgets": budget_labels, "draws": draws,
            "acquisition_rng": "default_rng([split_seed, fold, policy_code, draw, "
                               f"{ACQUISITION_SEED_SALT}]) — budget deliberately excluded so the "
                               "curve is nested within a draw",
            "model_fold_seed": "model_seed + fold*1009 + 9999991 (identical to gen5)",
        },
        folds=fold_records)
    manifest.record_many({"model_seed": int(args.model_seed), "split_seeds": [int(s) for s in seeds],
                          "hypotheses": verdicts,
                          "selection_hashes_sha256": sha256_json(selection_hashes)})
    payload = manifest.write(output_dir)

    validation = validate_run(output_dir, manifest=payload, required_artifacts=[
        "diversity_learning_curve.csv", "acquisition_audit.csv", "oof_predictions.parquet",
        "acquisition_selections.json", "decision_report.md", "summary.json", "manifest.json",
    ] + (["learning_curve_bootstrap.csv"] if not bootstrap.empty else []), checks=checks)
    success = write_success(output_dir, manifest=payload, validation=validation)
    log(f"validation ok={validation['ok']}; "
        f"{'_SUCCESS.json' if success else '_FAILED.json'} written")
    log(f"written to {output_dir}")
    return 0 if validation["ok"] else 1


# --------------------------------------------------------------------------- #
# Hypothesis scoring, checks and report
# --------------------------------------------------------------------------- #

def _boot_row(bootstrap: pd.DataFrame, endpoint: str, comparison: str, statistic: str = "mae"):
    if bootstrap.empty:
        return None
    hit = bootstrap[(bootstrap["endpoint"] == endpoint) & (bootstrap["comparison"] == comparison)
                    & (bootstrap["statistic"] == statistic)]
    return None if hit.empty else hit.iloc[0]


def score_hypotheses(curve: pd.DataFrame, bootstrap: pd.DataFrame, budget_labels, policies) -> dict:
    """Score B1-B3 exactly as pre-registered in protocol §5.  No post-hoc thresholds."""
    hard_endpoint = f"hard_vs_BASEtrain_nn<{HARD_CHEMISTRY_THRESHOLDS[0]:g}"
    # The mechanism travels with the verdict: how many ligands each policy could
    # afford at that budget is what makes a gain believable or tautological.
    bought = None
    if "train_extractants_mean" in curve.columns:
        bought = curve.groupby(["budget", "policy"])["train_extractants_mean"].mean()
    b1_budgets: list[str] = []
    detail: list[dict] = []
    for label in budget_labels:
        row = _boot_row(bootstrap, "overall", f"diversity_vs_depth@{label}")
        hard = _boot_row(bootstrap, hard_endpoint, f"diversity_vs_depth@{label}")
        versus_random = _boot_row(bootstrap, "overall", f"diversity_vs_random@{label}")
        entry = {"budget": label,
                 "gain_overall": None if row is None else float(row["point_delta"]),
                 "ci95_low_overall": None if row is None else float(row["ci95_low"]),
                 "ci95_high_overall": None if row is None else float(row["ci95_high"]),
                 "gain_hard": None if hard is None else float(hard["point_delta"]),
                 "ci95_low_hard": None if hard is None else float(hard["ci95_low"]),
                 "ci95_high_hard": None if hard is None else float(hard["ci95_high"]),
                 # the required null: beating DEPTH is easy at a small budget,
                 # because DEPTH buys almost no ligands.  Beating RANDOM is the
                 # claim that the *heuristic* is doing something.
                 "gain_vs_random": None if versus_random is None else float(versus_random["point_delta"]),
                 "ci95_low_vs_random": None if versus_random is None else float(versus_random["ci95_low"]),
                 "beats_random": bool(versus_random is not None and versus_random["ci95_low"] > 0),
                 "extractants_bought_depth": None if bought is None else float(
                     bought.get((label, "depth"), np.nan)),
                 "extractants_bought_diversity": None if bought is None else float(
                     bought.get((label, "diversity"), np.nan))}
        entry["passes_b1"] = bool(row is not None and row["ci95_low"] > 0)
        entry["passes_b2"] = bool(
            row is not None and hard is not None and hard["ci95_low"] > 0
            and row["ci95_low"] > 0 and hard["point_delta"] > row["point_delta"])
        detail.append(entry)
        if entry["passes_b1"]:
            b1_budgets.append(label)

    consecutive = 0
    best_run: list[str] = []
    run: list[str] = []
    for entry in detail:
        if entry["passes_b1"]:
            run.append(entry["budget"])
            if len(run) > consecutive:
                consecutive, best_run = len(run), list(run)
        else:
            run = []

    slope = None
    if len(budget_labels) >= 2:
        slope = _boot_row(bootstrap, "overall",
                          f"depth_slope_{budget_labels[-2]}_to_{budget_labels[-1]}")
    finite_labels = [label for label in budget_labels if label != "all"]
    fixed_slope = None
    if len(finite_labels) >= 2:
        fixed_slope = _boot_row(bootstrap, "overall",
                                f"depth_slope_{finite_labels[-2]}_to_{finite_labels[-1]}")
    b3 = {
        "statement": "the DEPTH curve slope between the two largest budgets is within noise of zero",
        "budgets": budget_labels[-2:] if len(budget_labels) >= 2 else budget_labels,
        "delta_macro_mae": None if slope is None else float(slope["point_delta"]),
        "ci95_low": None if slope is None else float(slope["ci95_low"]),
        "ci95_high": None if slope is None else float(slope["ci95_high"]),
        # "within noise of zero" == the 95 % interval of the change contains zero.
        "verdict": ("UNEVALUABLE" if slope is None else
                    ("PASS (saturated)" if slope["ci95_low"] <= 0 <= slope["ci95_high"]
                     else "FAIL (still improving)")),
        # Disclosed sensitivity: "all" is a fold-dependent budget, so the same
        # slope is also measured between the two largest fixed budgets.
        "fixed_budget_sensitivity": None if fixed_slope is None else {
            "budgets": finite_labels[-2:],
            "delta_macro_mae": float(fixed_slope["point_delta"]),
            "ci95_low": float(fixed_slope["ci95_low"]),
            "ci95_high": float(fixed_slope["ci95_high"]),
            "verdict": ("PASS (saturated)" if fixed_slope["ci95_low"] <= 0 <= fixed_slope["ci95_high"]
                        else "FAIL (still improving)")},
    }
    # A hypothesis with no contrast behind it is UNEVALUABLE, not FAILED: the gen5
    # run's two silently missing regimes are the reason this distinction is made
    # explicit everywhere in gen6.
    evaluable = any(d["gain_overall"] is not None for d in detail)
    return {
        "B1": {"statement": "at equal row budget, DIVERSITY beats DEPTH",
               "pass_condition": "CI95 low > 0 at >= 2 consecutive budgets",
               "budgets_with_ci_low_above_zero": b1_budgets,
               "longest_consecutive_run": best_run,
               "verdict": ("UNEVALUABLE (no DIVERSITY-vs-DEPTH contrast was run)" if not evaluable
                           else "PASS" if consecutive >= 2 else "FAIL")},
        "B2": {"statement": "the DIVERSITY advantage is larger on hard chemistry than overall",
               "pass_condition": f"gain at {hard_endpoint} > gain overall, both CI95 low > 0",
               "budgets_passing": [d["budget"] for d in detail if d["passes_b2"]],
               "verdict": ("UNEVALUABLE (no DIVERSITY-vs-DEPTH contrast was run)" if not evaluable
                           else "PASS" if any(d["passes_b2"] for d in detail) else "FAIL")},
        "B3": b3,
        "per_budget": detail,
    }


def build_checks(curve, audit_table, integrity, label_free_check, short_budget_folds,
                 budgets, policies, draws, seeds) -> dict:
    """Run-time assertions handed to ``validate_run`` (fail-closed)."""
    expected_rows = len(seeds) * len(policies) * len(budgets) * draws
    # Every point-fold whose budget did not exceed the pool must hit the budget exactly.
    honest_shortfalls = all(row["selected_rows"] == row["pool_rows"] for row in short_budget_folds)
    all_points = audit_table[audit_table["budget"] == "all"]
    all_identical = True
    if not all_points.empty:
        subset = curve[curve["budget"] == "all"]
        all_identical = bool(subset.groupby("split_seed")["macro_mae"].nunique().le(1).all())
    return {
        "curve_shape": {
            "ok": len(curve) == expected_rows, "n_rows": int(len(curve)),
            "expected": int(expected_rows),
            "note": "one row per (split seed, policy, budget, draw), folds aggregated"},
        "budgets_met_exactly": {
            "ok": bool(honest_shortfalls),
            "n_point_folds_short": len(short_budget_folds),
            "note": "a shortfall is only legitimate when the fold's whole training pool is "
                    "smaller than the budget; then the selection must equal the pool"},
        "all_budget_points_identical": {
            "ok": bool(all_identical),
            "note": "at budget 'all' every policy and draw trains on the fold's whole pool, so "
                    "their metrics must coincide exactly"},
        "no_missing_predictions": {
            "ok": bool(curve["n_rows"].notna().all() and (curve["n_rows"] > 0).all())},
        "label_free_selection": label_free_check,
        "split_integrity": {"ok": all(r["ok"] for r in integrity.values()),
                            "per_seed": {k: v["ok"] for k, v in integrity.items()}},
    }


def _pivot(curve: pd.DataFrame, value: str, budget_labels, policies) -> pd.DataFrame:
    """Budget x policy table of a metric, averaged over draws and split seeds."""
    if value not in curve.columns:
        return pd.DataFrame()
    table = (curve.pivot_table(index="budget", columns="policy", values=value, aggfunc="mean")
             .reindex(index=[b for b in budget_labels], columns=[p for p in policies]))
    return table


def decision_report(*, stamp, args, data, chemistry, curve, audit_table, bootstrap, verdicts,
                    checks, budget_labels, policies, draws, seeds, feature_columns,
                    short_budget_folds, n_estimators, replicates) -> str:
    lines: list[str] = []
    A = lines.append
    A(f"# gen6 Experiment B — diversity versus depth at equal row budget ({stamp})")
    A("")
    A("**Question.** Experiment A can be dismissed with \"you just gave it more rows\". Here every "
      "arm gets the *same* number of training rows and only the acquisition policy differs, so a "
      "difference can only come from *which chemistry* the budget was spent on.")
    A("")
    A("## What was run")
    A("")
    A(f"* shared cohort (min_cells = {args.eval_min_cells}): **{data.audit['rows']} rows**, "
      f"{data.audit['extractants']} extractants, {data.audit['ecfp_clusters']} ECFP clusters, "
      f"{data.audit['tanimoto_clusters']} Tanimoto-0.7 super-clusters, "
      f"{data.audit['metals']} metals; target sd {data.audit['target_sd']:.3f}")
    A(f"* frozen chemistry map over {len(chemistry)} extractants "
      f"({chemistry.audit['n_ecfp_clusters']} ECFP clusters, "
      f"{chemistry.audit['n_superclusters']} super-clusters, threshold "
      f"{chemistry.audit['supercluster_threshold']})")
    A(f"* learner: ExtraTrees, {n_estimators} trees, max_features {args.max_features}, "
      f"min_samples_leaf {args.min_samples_leaf}; feature set `{args.feature_set}` "
      f"({len(feature_columns)} columns) — frozen, exactly as gen5 left it")
    A(f"* folds: {args.folds} held-out Tanimoto-0.7 super-clusters, gen5 fold algorithm, "
      f"split seeds {list(seeds)}")
    A(f"* policies {list(policies)}; budgets {budget_labels}; {draws} acquisition draw(s) per point")
    A(f"* every metric below is out-of-fold over the whole cohort "
      f"({int(curve['n_rows'].iloc[0])} rows, {int(curve['n_ligands'].iloc[0])} ligands, "
      f"{int(curve['n_ecfp_clusters'].iloc[0])} ECFP clusters, "
      f"n_eff(pooled rows) {float(curve['n_eff_pooled_rows'].iloc[0]):.1f})")
    A("")
    A("## How to read this")
    A("")
    A("* **macro MAE** weights one ECFP cluster as one vote (the pre-registered primary metric). "
      "Pooled MAE is reported in the CSV but never used for selection: the largest ligand is "
      f"{100 * data.audit.get('largest_extractant_share', float('nan')):.0f} % of rows.")
    A("* **offset vs shape**: `offset_mae` is the per-ligand level error (the failure six "
      "generations could not fix), `shape_mae` the within-ligand response shape. They are "
      "orthogonal components of the same squared error.")
    A("* **two hard-chemistry readings, deliberately**. `hard_own_train_nn<t>__*` measures the "
      "test ligand's distance to *that point's own* training set — the honest description of what "
      "that model saw, but the row subset moves with the point, so those numbers may not be "
      "subtracted between policies. `hard_vs_BASEtrain_nn<t>__*` uses the frozen BASE (>= "
      f"{args.base_min_cells} cells) bin, identical rows for every point; **only that one is used "
      "for the paired contrasts and for B2**.")
    A("* a **draw** is one realisation of a policy. Draws are nested across budgets within a draw "
      "(the acquisition RNG does not see the budget), so the curve's slope is not sampling noise.")
    A("* at budget `all` every policy and draw is the same training set; those rows are printed "
      "for completeness and are an internal consistency check "
      f"(`all_budget_points_identical` = {checks['all_budget_points_identical']['ok']}).")
    A("")
    A("## What a budget buys in chemistry (mean over folds, draws and seeds)")
    A("")
    for value, title in (("train_extractants_mean", "extractants in training"),
                         ("train_superclusters_mean", "Tanimoto-0.7 super-clusters in training"),
                         ("train_ecfp_clusters_mean", "ECFP clusters in training")):
        table = _pivot(curve, value, budget_labels, policies)
        if table.empty:
            continue
        A(f"**{title}**")
        A("")
        A("```")
        A(table.round(2).to_string())
        A("```")
        A("")
    A("This table is the mechanism, not decoration: if `diversity` and `depth` buy the same "
      "chemistry at a budget, no difference in accuracy is expected there.")
    A("")
    A("## Learning curve — macro MAE (one ECFP cluster = one vote; lower is better)")
    A("")
    A("```")
    A(_pivot(curve, "macro_mae", budget_labels, policies).round(4).to_string())
    A("```")
    A("")
    spread = (curve.groupby(["budget", "policy"])["macro_mae"].std()
              .unstack("policy").reindex(index=budget_labels, columns=policies))
    if draws * len(seeds) < 2 or spread.isna().all().all():
        A(f"_Spread over draws is undefined here: {draws} acquisition draw(s) x {len(seeds)} split "
          f"seed(s) gives a single realisation per point. A policy is a distribution over training "
          f"sets, so a one-draw run is a smoke test, not evidence._")
    else:
        A(f"Spread over the {draws} acquisition draw(s) x {len(seeds)} split seed(s) "
          f"(sd of macro MAE):")
        A("")
        A("```")
        A(spread.round(4).to_string())
        A("```")
    A("")
    for value, title in (("offset_mae", "offset MAE (per-ligand level error)"),
                         ("shape_mae", "shape MAE (within-ligand response shape)"),
                         ("median_ligand_mae", "median ligand MAE"),
                         ("worst_quartile_ligand_mae", "worst-quartile ligand MAE"),
                         (f"hard_vs_BASEtrain_nn{HARD_CHEMISTRY_THRESHOLDS[0]:g}__macro_mae",
                          f"macro MAE on hard chemistry (nn to BASE training < "
                          f"{HARD_CHEMISTRY_THRESHOLDS[0]:g}; identical rows for every policy)"),
                         (f"hard_own_train_nn{HARD_CHEMISTRY_THRESHOLDS[0]:g}__macro_mae",
                          f"macro MAE on rows far from THIS POINT'S OWN training set "
                          f"(nn < {HARD_CHEMISTRY_THRESHOLDS[0]:g}; the row subset moves with the "
                          f"point — descriptive only)")):
        table = _pivot(curve, value, budget_labels, policies)
        if table.empty or table.isna().all().all():
            continue
        A(f"**{title}**")
        A("")
        A("```")
        A(table.round(4).to_string())
        A("```")
        A("")
    A("## Paired bootstrap over independent chemistry units")
    A("")
    A(f"Scoring unit = ECFP cluster; resampling unit = Tanimoto-0.7 super-cluster (the block the "
      f"folds actually held out), {replicates} replicates, one shared index matrix for every "
      f"comparison. Positive delta = the candidate policy is better than the reference. The "
      f"per-unit error of a policy is averaged over draws and split seeds first, because a policy "
      f"is a distribution over training sets.")
    A("")
    if bootstrap.empty:
        A("_No comparison could be formed (too few policies or budgets)._")
    else:
        show = bootstrap[bootstrap["statistic"].isin(["mae", "offset_mae"])]
        A("```")
        A(show[["endpoint", "comparison", "statistic", "point_delta", "ci95_low", "ci95_high",
                "p_worse_one_sided", "units_improved", "units_total", "bootstrap_blocks"]]
          .round(4).to_string(index=False))
        A("```")
    A("")
    A("## Pre-registered hypotheses (protocol §5)")
    A("")
    for key in ("B1", "B2"):
        entry = verdicts[key]
        A(f"**{key} — {entry['statement']}**  →  `{entry['verdict']}`")
        A("")
        A(f"* pass condition: {entry['pass_condition']}")
        if key == "B1":
            A(f"* budgets with CI95 low > 0: {entry['budgets_with_ci_low_above_zero'] or 'none'}; "
              f"longest consecutive run: {entry['longest_consecutive_run'] or 'none'}")
        else:
            A(f"* budgets passing: {entry['budgets_passing'] or 'none'}")
        A("")
    A("Per budget, with the mechanism and the required null beside the verdict "
      "(gains are macro MAE, positive = DIVERSITY better):")
    A("")
    detail = pd.DataFrame(verdicts["per_budget"])
    if not detail.empty:
        columns = [c for c in ("budget", "extractants_bought_depth", "extractants_bought_diversity",
                               "gain_overall", "ci95_low_overall", "gain_hard", "ci95_low_hard",
                               "gain_vs_random", "ci95_low_vs_random", "beats_random",
                               "passes_b1", "passes_b2") if c in detail.columns]
        A("```")
        A(detail[columns].round(4).to_string(index=False))
        A("```")
        A("")
    A("**Read the `vs_random` column before celebrating.** Protocol §6 requires an acquisition "
      "claim to beat *random acquisition* as well as the incumbent heuristic. Beating DEPTH at a "
      "small budget is close to tautological — DEPTH spends the whole budget on the few "
      "best-measured ligands (see the extractants-bought columns), so it is a one-ligand model. "
      "The load-bearing question is whether DIVERSITY beats RANDOM at the same budget.")
    A("")
    b3 = verdicts["B3"]
    A(f"**B3 — {b3['statement']}**  →  `{b3['verdict']}`")
    A("")
    A(f"* budgets compared: {b3['budgets']}; change in macro MAE "
      f"{b3['delta_macro_mae'] if b3['delta_macro_mae'] is None else round(b3['delta_macro_mae'], 4)} "
      f"(CI95 [{b3['ci95_low'] if b3['ci95_low'] is None else round(b3['ci95_low'], 4)}, "
      f"{b3['ci95_high'] if b3['ci95_high'] is None else round(b3['ci95_high'], 4)}])")
    A("* \"within noise of zero\" is read as: the 95 % interval of the change contains zero.")
    if b3.get("fixed_budget_sensitivity"):
        sensitivity = b3["fixed_budget_sensitivity"]
        A(f"* disclosed sensitivity — `all` is a **fold-dependent** budget (the fold holding the "
          f"giant super-cluster trains on far fewer rows than the others), so the same slope "
          f"between the two largest *fixed* budgets {sensitivity['budgets']} is "
          f"{round(sensitivity['delta_macro_mae'], 4)} (CI95 "
          f"[{round(sensitivity['ci95_low'], 4)}, {round(sensitivity['ci95_high'], 4)}]) → "
          f"`{sensitivity['verdict']}`. Only the pre-registered reading above is scored.")
    A("")
    A("### What would falsify these claims")
    A("")
    A("* **B1** dies if `diversity` and `random` are indistinguishable — that would say the budget, "
      "not its allocation, is what matters. That contrast is in the bootstrap table above.")
    A("* **B1** also dies if the advantage disappears once the per-unit errors are averaged over "
      "draws: a single lucky acquisition draw is not a policy.")
    A("* **B2** dies if the gain is flat in nearest-neighbour distance; then the policy is buying "
      "general regularisation, not coverage of new chemistry.")
    A("* **B3** dies if the depth curve is still falling between the two largest budgets, in which "
      "case the comparison is budget-limited and the whole curve should be re-run further out.")
    A("")
    A("## Caveats recorded by this run")
    A("")
    if short_budget_folds:
        by_budget: dict[str, int] = {}
        for row in short_budget_folds:
            by_budget[row["budget"]] = by_budget.get(row["budget"], 0) + 1
        distinct = sorted({(row["split_seed"], row["fold"], row["pool_rows"])
                           for row in short_budget_folds})
        A(f"* **{len(short_budget_folds)} point-folds could not spend their budget** "
          f"({len(distinct)} distinct (seed, fold) combinations, pools "
          f"{[d[2] for d in distinct]} rows) because the "
          f"fold's whole training pool was smaller: {by_budget} point-folds per budget. "
          f"In those folds every policy trains "
          f"on the identical pool, so the contrast is structurally zero there and the reported gain "
          f"at those budgets is diluted towards zero. The largest super-cluster is "
          f"{100 * data.audit.get('largest_tanimoto_cluster_share', float('nan')):.0f} % of rows, so "
          f"the fold that holds it out has by far the smallest pool.")
    else:
        A("* every point-fold spent its budget exactly.")
    A(f"* label-free acquisition was verified at run time on the real cohort "
      f"(`label_free_selection` = {checks['label_free_selection'].get('ok')}): permuting `log_D` "
      f"moved none of the {checks['label_free_selection'].get('n_points_checked')} selections.")
    A(f"* split integrity (identical test rows, no extractant / ECFP / super-cluster shared between "
      f"a fold's test and training rows): {checks['split_integrity']['ok']}.")
    A("* the acquisition policies operate on **whole ligands** (a ligand enters, then the next), "
      "with only the ligand that straddles the budget contributing a random row subset. A policy "
      "that could pick individual rows would be a different experiment.")
    A("* this run fixes the feature set to one family; a policy that helps only the champion "
      "features would be an artefact of the representation, so a second family "
      "(`MC_donors_massaction`) is a separate invocation.")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
