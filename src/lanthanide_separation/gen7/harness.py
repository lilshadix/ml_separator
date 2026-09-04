"""The gen7 evaluation contract: one cohort, one fold plan, one metric module.

Every gen7 contender — tree, kernel, neural, ensemble — is a function that maps
``(train frame, train target, test frame, context)`` to a prediction vector.  The
harness owns everything else, which is the only way a leaderboard across a dozen
architectures means anything:

* **one shared cohort**, built once at ``min_cells = 3`` through the gen5 builder
  (5,248 rows / 152 extractants / 131 ECFP clusters / 79 Tanimoto chemotypes), so
  every feature column, one-hot level, cluster label and replicate average is
  identical for every contender;
* **one fold plan** — ``seeded_group_kfold`` over ``tanimoto_cluster``, the gen5
  algorithm reused byte-for-byte, so a gen7 run of the gen6 champion reproduces
  the gen6 number rather than approximating it;
* **one metric module** — :mod:`gen6.metrics`, unchanged, so macro MAE still means
  "one ECFP cluster, one vote" and offset/shape still satisfy the SSE identity.

The contender never sees the test target.  It sees the test *frame*, because a
transductive method (a GP, a k-NN offset) legitimately needs the test features —
and because withholding them would forbid half the architectures the brief asks
for.  :func:`assert_no_target_leak` proves the test target column is absent from
what the contender is handed.

``nn_train_tanimoto`` is the maximum Tanimoto from a test ligand to the fold's
*training* ligands.  Because every gen7 contender trains on the same rows, this
column is a property of the fold and not of the arm, so the hard-chemistry
endpoints cut on it compare arms on identical rows.  ``nn_base_tanimoto`` (the
same quantity against the ``>= 10 cells`` subset) is carried alongside so gen7
numbers can be read against gen6's tables, and is never used for selection.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from ..gen6.chemistry import ChemistryMap, build_chemistry_map
from ..gen6.cohorts import BASE_MIN_CELLS, arm_membership, seeded_group_kfold
from ..gen6.metrics import gen6_metric_table, per_unit_statistics, paired_unit_bootstrap
from ..levels import (
    LEVEL_ARMS, LEVEL_IDENTITY_COLUMNS, LEVEL_TARGET_COLUMN, LevelData, build_level_dataset,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
DESCRIPTOR_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_2d_descriptors.parquet"
EMBEDDING_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_pretrained_embeddings.parquet"
CACHE_DIR = REPO_ROOT / "runs" / "gen7_architecture" / "cache"

#: The gen6 cohort: ``min_cells = 3``, 152 extractants.  Frozen for gen7.
GEN7_MIN_CELLS = 3
#: The five split seeds every gen5/gen6/gen7 study uses.  They re-partition the
#: same ligands, so they measure split sensitivity — not independent replication.
DEFAULT_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
N_SPLITS = 5
#: gen5's fold seed formula, reproduced exactly.
FOLD_SEED_STRIDE = 1009
FOLD_SEED_OFFSET = 9_999_991
#: gen5/gen6's model seed base.  Deliberately *independent* of the split seed:
#: the split seed re-partitions the chemistry, the model seed re-randomises the
#: learner, and confounding them would make "seed spread" mean two things at once.
MODEL_SEED_BASE = 42
#: The gen6 champion and the surprise winner of gen6 Experiment A, in that order.
GEN6_REFERENCE_ARM = "MC_lig2d_ext_massaction"
GEN6_BEST_ARM = "MC_donors"


# --------------------------------------------------------------------------- #
# Cohort
# --------------------------------------------------------------------------- #

@dataclass
class Gen7Cohort:
    """The shared, frozen evaluation cohort plus the chemistry map behind it."""

    data: LevelData
    chemistry: ChemistryMap
    fingerprint: str

    @property
    def frame(self) -> pd.DataFrame:
        return self.data.frame

    @property
    def blocks(self) -> Mapping[str, tuple[str, ...]]:
        return self.data.blocks

    def arm_columns(self, arm: str) -> tuple[str, ...]:
        return self.data.arm_columns(arm)

    def block_columns(self, blocks: Sequence[str]) -> tuple[str, ...]:
        return self.data.block_columns(blocks)


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Digest of the *evaluation contract*: which rows exist and what their target is.

    Deliberately independent of the column count.  Adding a feature block must not
    change this hash, because the contract gen7 enforces is "every contender is
    scored on byte-identical rows with byte-identical targets" — not "every
    contender sees the same columns", which would forbid the comparison the
    generation exists to make.  Column inventory is recorded separately in the run
    manifest.
    """
    key = frame[["row_id", LEVEL_TARGET_COLUMN]].sort_values("row_id")
    payload = "|".join(f"{r}:{v:.10g}" for r, v in zip(key["row_id"], key[LEVEL_TARGET_COLUMN]))
    payload += f"||rows={len(frame)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


#: Blocks gen7 adds on top of the gen5 registry.  ``EMB_*`` are frozen pretrained
#: molecular embeddings (target-free, one row per canonical SMILES); ``RECOVERED``
#: are the experimental variables rescued from the upstream SAFE exports.
GEN7_EMBEDDING_BLOCKS: tuple[str, ...] = (
    "EMB_CHEMBERTA", "EMB_CHEMBERTA_MLM", "EMB_MOLFORMER",
    "EMB_CHEMBERTA_CLS", "EMB_MOLFORMER_CLS")
GEN7_EXTRA_BLOCKS: tuple[str, ...] = GEN7_EMBEDDING_BLOCKS + ("RECOVERED",)


def _attach_gen7_blocks(data: LevelData) -> LevelData:
    """Join the pretrained embeddings and the recovered variables onto the cohort.

    Both joins are *additive and row-preserving*: the embeddings key on
    ``extractant`` (= canonical SMILES) and the recovered table on ``row_id``, so
    no fold, cluster label or replicate average can move.  A missing table is not
    an error — the corresponding blocks simply do not appear, and an arm that asks
    for them fails loudly rather than silently training on nothing.
    """
    frame = data.frame
    blocks = dict(data.blocks)

    if EMBEDDING_PATH.exists():
        table = pd.read_parquet(EMBEDDING_PATH)
        table = table.rename(columns={"canonical_smiles": "extractant"}).drop_duplicates("extractant")
        before = len(frame)
        frame = frame.merge(table, on="extractant", how="left", validate="many_to_one")
        if len(frame) != before:
            raise AssertionError("embedding join changed the row count")
        mapping = {
            "EMB_CHEMBERTA": "emb__chemberta__mean_",
            "EMB_CHEMBERTA_MLM": "emb__chemberta_mlm__mean_",
            "EMB_MOLFORMER": "emb__molformer__mean_",
            "EMB_CHEMBERTA_CLS": "emb__chemberta__cls_",
            "EMB_MOLFORMER_CLS": "emb__molformer__cls_",
        }
        for name, prefix in mapping.items():
            columns = tuple(c for c in frame.columns if c.startswith(prefix))
            if columns:
                blocks[name] = columns

    recovered_cache = CACHE_DIR / "recovered_cells.parquet"
    if recovered_cache.exists():
        table = pd.read_parquet(recovered_cache)
        keep = ["row_id"] + [c for c in table.columns if c.startswith("rec__")]
        nuisance = [c for c in table.columns if c.startswith("nuisance__")]
        before = len(frame)
        frame = frame.merge(table[keep + nuisance], on="row_id", how="left", validate="one_to_one")
        if len(frame) != before:
            raise AssertionError("recovered join changed the row count")
        columns = tuple(c for c in frame.columns if c.startswith("rec__"))
        # Columns with no observed value at all carry nothing; drop them from the
        # block so a learner is not handed a wall of NaN.
        columns = tuple(c for c in columns if frame[c].notna().any())
        if columns:
            blocks["RECOVERED"] = columns

    # METALPHYS: the lanthanide series is *structured*, and the bundle's METAL block
    # (atomic number, series index, ionic radius) is monotone in all three, so a tree
    # can only cut it into intervals.  This adds the two things that are not monotone
    # — the 4f count's distance from the half-filled shell at Gd (the tetrad effect)
    # and a radial-basis expansion of the ionic radius — so neighbouring metals share
    # strength while the model can still bend sharply where the chemistry does.
    from .hierarchical_nn import metal_physical_features
    metal_matrix = metal_physical_features(frame["metal_symbol"].astype(str).tolist())
    metal_names = ("radius", "radius_sq", "series_position", "series_position_sq",
                   "n_f", "n_f_norm", "n_f_from_half", "n_f_from_half_sq") + tuple(
        f"rbf_{i:02d}" for i in range(metal_matrix.shape[1] - 8))
    metal_columns = tuple(f"metalphys__{n}" for n in metal_names)
    frame = frame.assign(**{c: metal_matrix[:, i] for i, c in enumerate(metal_columns)})
    blocks["METALPHYS"] = metal_columns
    # A one-hot metal identity, so "structured continuous" can be measured against
    # "14 unrelated categories" rather than assumed better than it.
    onehot = pd.get_dummies(frame["metal_symbol"].astype(str), prefix="metalonehot_")
    frame = frame.assign(**{c: onehot[c].astype(float) for c in onehot.columns})
    blocks["METAL_ONEHOT"] = tuple(onehot.columns)

    # LIGPHYS is built last because its coupling terms need the recovered solvent
    # physics to exist; without it the block still forms, minus those columns.
    from .ligand_physics import attach_ligand_physics
    frame, ligphys = attach_ligand_physics(frame)
    ligphys = tuple(c for c in ligphys if frame[c].notna().any())
    if ligphys:
        blocks["LIGPHYS"] = ligphys

    return LevelData(frame=frame, blocks=blocks, audit=data.audit)


def load_cohort(*, use_cache: bool = True, extra_features: pd.DataFrame | None = None,
                with_gen7_blocks: bool = True) -> Gen7Cohort:
    """Build (or load) the frozen gen7 cohort.

    ``extra_features`` is an optional table keyed on ``row_id`` carrying gen7's
    recovered experimental variables; it is joined *after* the cohort is built so
    that adding it cannot move a fold, a cluster label or a replicate average.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "cohort.parquet"
    blocks_path = CACHE_DIR / "cohort_blocks.json"
    audit_path = CACHE_DIR / "cohort_audit.json"
    if use_cache and cache.exists() and blocks_path.exists():
        frame = pd.read_parquet(cache)
        blocks = {k: tuple(v) for k, v in json.loads(blocks_path.read_text()).items()}
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        data = LevelData(frame=frame, blocks=blocks, audit=audit)
    else:
        source = pd.read_parquet(DATASET_PATH)
        descriptors = pd.read_parquet(DESCRIPTOR_PATH) if DESCRIPTOR_PATH.exists() else None
        data = build_level_dataset(
            source, min_rows_per_extractant=GEN7_MIN_CELLS, ligand_descriptors=descriptors)
        data.frame.to_parquet(cache, index=False)
        blocks_path.write_text(json.dumps({k: list(v) for k, v in data.blocks.items()}, indent=2))
        audit_path.write_text(json.dumps(data.audit, indent=2, default=str))

    chem_cache = CACHE_DIR / "chemistry_map.parquet"
    if use_cache and chem_cache.exists():
        chemistry = ChemistryMap.from_parquet(chem_cache)
    else:
        source = pd.read_parquet(DATASET_PATH)
        descriptors = pd.read_parquet(DESCRIPTOR_PATH) if DESCRIPTOR_PATH.exists() else None
        chemistry = build_chemistry_map(source, ligand_descriptors=descriptors)
        chemistry.to_parquet(chem_cache)

    if with_gen7_blocks:
        data = _attach_gen7_blocks(data)

    if extra_features is not None:
        if "row_id" not in extra_features.columns:
            raise KeyError("extra_features must be keyed on row_id")
        before = len(data.frame)
        merged = data.frame.merge(extra_features, on="row_id", how="left", validate="one_to_one")
        if len(merged) != before:
            raise AssertionError("extra_features join changed the row count")
        data = LevelData(frame=merged, blocks=data.blocks, audit=data.audit)

    return Gen7Cohort(data=data, chemistry=chemistry, fingerprint=_frame_fingerprint(data.frame))


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Fold:
    """One outer fold of one seed.  Identical for every contender."""

    seed: int
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    held_out_chemotypes: tuple[str, ...]
    model_seed_base: int = MODEL_SEED_BASE

    @property
    def model_seed(self) -> int:
        """gen5's formula, reproduced exactly: ``model_seed + fold*1009 + 9_999_991``."""
        return int(self.model_seed_base + self.fold * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET)


def build_folds(frame: pd.DataFrame, seed: int, *, n_splits: int = N_SPLITS,
                model_seed_base: int = MODEL_SEED_BASE) -> list[Fold]:
    groups = frame["tanimoto_cluster"].astype(str).to_numpy()
    out: list[Fold] = []
    for k, (train_index, test_index) in enumerate(seeded_group_kfold(groups, n_splits, seed)):
        out.append(Fold(seed=int(seed), fold=k, train_index=train_index, test_index=test_index,
                        held_out_chemotypes=tuple(sorted(set(groups[test_index]))),
                        model_seed_base=int(model_seed_base)))
    return out


def assert_fold_integrity(frame: pd.DataFrame, folds: Sequence[Fold]) -> dict:
    """No extractant, ECFP cluster or chemotype may cross a fold boundary."""
    report: dict = {"ok": True, "folds": []}
    covered: set[int] = set()
    columns = [c for c in ("extractant", "ecfp_cluster", "tanimoto_cluster") if c in frame.columns]
    arrays = {c: frame[c].astype(str).to_numpy() for c in columns}
    for fold in folds:
        entry: dict = {"fold": fold.fold, "seed": fold.seed,
                       "n_train": int(fold.train_index.size), "n_test": int(fold.test_index.size),
                       "leaks": {}}
        if set(fold.train_index.tolist()) & set(fold.test_index.tolist()):
            entry["leaks"]["row_overlap"] = True
            report["ok"] = False
        for column, values in arrays.items():
            shared = set(values[fold.train_index]) & set(values[fold.test_index])
            if shared:
                entry["leaks"][column] = sorted(shared)[:5]
                report["ok"] = False
        covered |= set(fold.test_index.tolist())
        report["folds"].append(entry)
    report["all_rows_tested_once"] = len(covered) == len(frame)
    if not report["all_rows_tested_once"]:
        report["ok"] = False
        report["n_rows_never_tested"] = int(len(frame) - len(covered))
    return report


# --------------------------------------------------------------------------- #
# Contender protocol
# --------------------------------------------------------------------------- #

@dataclass
class FoldContext:
    """Everything a contender may legitimately use, and nothing else."""

    fold: Fold
    cohort: Gen7Cohort
    feature_columns: tuple[str, ...]
    model_seed: int
    #: Per-row diagnostics a contender may write back (prediction sd, offset, shape…).
    extras: dict = field(default_factory=dict)


class Contender(Protocol):
    """``fit_predict`` returns one prediction per test row, in test-row order."""

    name: str

    def fit_predict(self, train: pd.DataFrame, y_train: np.ndarray,
                    test: pd.DataFrame, context: FoldContext) -> np.ndarray: ...


def assert_no_target_leak(test: pd.DataFrame) -> None:
    if LEVEL_TARGET_COLUMN in test.columns:
        raise AssertionError(
            f"the test frame handed to a contender still carries {LEVEL_TARGET_COLUMN!r}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _fold_neighbours(chemistry: ChemistryMap, query: Sequence[str], reference: Sequence[str]) -> pd.Series:
    known = set(chemistry.extractants)
    q = [x for x in query if x in known]
    r = [x for x in reference if x in known]
    table = chemistry.nearest_neighbour(q, r, exclude_self=True)
    return table.set_index("extractant")["nn_tanimoto"]


def evaluate_contender(
    contender: Contender,
    cohort: Gen7Cohort,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_splits: int = N_SPLITS,
    feature_columns: Sequence[str] | None = None,
    model_seed_base: int = MODEL_SEED_BASE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Out-of-fold predictions for one contender over every seed and fold.

    Returns a long frame with one row per (seed, cohort row): the identity
    columns, the truth, ``prediction``, ``fold``, ``nn_train_tanimoto`` and
    ``nn_base_tanimoto``.  Scoring is done separately by :func:`score_oof` so that
    a contender can be re-scored without being re-fitted.
    """
    frame = cohort.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    extractants = frame["extractant"].astype(str).to_numpy()
    dense_mask = arm_membership(frame, min_cells=BASE_MIN_CELLS)
    columns = tuple(feature_columns) if feature_columns is not None else ()
    parts: list[pd.DataFrame] = []
    identity = [c for c in LEVEL_IDENTITY_COLUMNS if c in frame.columns]

    for seed in seeds:
        folds = build_folds(frame, seed, n_splits=n_splits, model_seed_base=model_seed_base)
        integrity = assert_fold_integrity(frame, folds)
        if not integrity["ok"]:
            raise AssertionError(f"fold integrity failed for seed {seed}: {integrity}")
        prediction = np.full(len(frame), np.nan)
        fold_of = np.full(len(frame), -1, dtype=int)
        nn_train = np.full(len(frame), np.nan)
        nn_base = np.full(len(frame), np.nan)
        extras: dict[str, np.ndarray] = {}
        started = time.time()
        for fold in folds:
            train = frame.iloc[fold.train_index]
            test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
            assert_no_target_leak(test)
            context = FoldContext(fold=fold, cohort=cohort, feature_columns=columns,
                                  model_seed=fold.model_seed)
            values = np.asarray(
                contender.fit_predict(train, target[fold.train_index], test, context), dtype=float)
            if values.shape != (len(fold.test_index),):
                raise ValueError(
                    f"{contender.name} returned {values.shape}, expected {(len(fold.test_index),)}")
            if not np.isfinite(values).all():
                raise ValueError(f"{contender.name} returned non-finite predictions on fold {fold.fold}")
            prediction[fold.test_index] = values
            fold_of[fold.test_index] = fold.fold

            test_names = sorted(set(extractants[fold.test_index]))
            train_names = sorted(set(extractants[fold.train_index]))
            base_names = sorted(set(extractants[fold.train_index[dense_mask[fold.train_index]]]))
            lookup_train = _fold_neighbours(cohort.chemistry, test_names, train_names)
            lookup_base = _fold_neighbours(cohort.chemistry, test_names, base_names)
            names = pd.Series(extractants[fold.test_index])
            nn_train[fold.test_index] = names.map(lookup_train).to_numpy(dtype=float)
            nn_base[fold.test_index] = names.map(lookup_base).to_numpy(dtype=float)

            for key, value in context.extras.items():
                arr = extras.setdefault(key, np.full(len(frame), np.nan))
                values = np.asarray(value, dtype=float)
                if values.shape != (len(fold.test_index),):
                    # Name the key.  A bare numpy broadcast error here says only
                    # "shape (22,) could not be broadcast to (382,)", which sends the
                    # reader looking at the predictions rather than at a diagnostic
                    # column that was accidentally per-ligand instead of per-row.
                    raise ValueError(
                        f"{contender.name}: extras[{key!r}] has shape {values.shape}, "
                        f"expected one value per test ROW {(len(fold.test_index),)} "
                        f"(fold {fold.fold}, {len(set(extractants[fold.test_index]))} "
                        f"distinct ligands — a per-ligand array is the usual cause)")
                arr[fold.test_index] = values

        if np.isnan(prediction).any():
            raise AssertionError(f"{contender.name}: {int(np.isnan(prediction).sum())} rows unpredicted")
        block = frame[identity].copy()
        block["prediction"] = prediction
        block["fold"] = fold_of
        block["split_seed"] = int(seed)
        block["nn_train_tanimoto"] = nn_train
        block["nn_base_tanimoto"] = nn_base
        block["model"] = contender.name
        for key, arr in extras.items():
            block[f"extra__{key}"] = arr
        parts.append(block)
        if verbose:
            mae = float(np.abs(prediction - target).mean())
            print(f"  [{contender.name}] seed {seed}: pooled MAE {mae:.4f} "
                  f"({time.time() - started:.1f}s)", flush=True)
    return pd.concat(parts, ignore_index=True)


def score_oof(
    oof: pd.DataFrame,
    *,
    similarity_column: str = "nn_train_tanimoto",
    per_seed: bool = True,
) -> pd.DataFrame:
    """Score an OOF frame with the gen6 metric module, one row per (model, seed)."""
    rows: list[pd.DataFrame] = []
    keys = ["model", "split_seed"] if per_seed else ["model"]
    for key, block in oof.groupby(keys, sort=True):
        work = block.rename(columns={"prediction": "prediction_arm"})
        overall, _, hard = gen6_metric_table(
            work, ["arm"], similarity_column=similarity_column, prediction_prefix="prediction_")
        overall = overall.drop(columns=["arm"])
        values = key if isinstance(key, tuple) else (key,)
        for name, value in zip(keys, values):
            overall.insert(0, name, value)
        if not hard.empty:
            for endpoint in ("nn<0.4", "nn<0.6"):
                sub = hard[hard["endpoint"] == endpoint]
                if len(sub):
                    tag = endpoint.replace("nn<", "nn_lt_").replace(".", "_")
                    overall[f"{tag}__macro_mae"] = float(sub["macro_mae"].iloc[0])
                    overall[f"{tag}__offset_mae"] = float(sub["offset_mae"].iloc[0])
                    overall[f"{tag}__shape_mae"] = float(sub["shape_mae"].iloc[0])
                    overall[f"{tag}__n_rows"] = int(sub["n_rows"].iloc[0])
        rows.append(overall)
    return pd.concat(rows, ignore_index=True)


def leaderboard(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds, with the seed spread beside it — never a single seed.

    The spread columns are computed from the *unselected* groupby: re-indexing an
    already-column-selected ``DataFrameGroupBy`` raises ``IndexError: Column(s)
    already selected`` on pandas 3, which is how this silently killed a completed
    twenty-contender sweep at the final aggregation step.
    """
    numeric = [c for c in scores.select_dtypes("number").columns if c != "split_seed"]
    grouped = scores.groupby("model")
    out = grouped[numeric].mean()
    for column in ("macro_mae", "offset_mae", "shape_mae"):
        if column in numeric:
            out[f"{column}_sd"] = grouped[column].std()
            out[f"{column}_min"] = grouped[column].min()
            out[f"{column}_max"] = grouped[column].max()
    out["n_seeds"] = grouped.size()
    return out.sort_values("macro_mae").reset_index()
