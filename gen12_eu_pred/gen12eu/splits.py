"""Fold construction.  Row-random splitting does not exist in this module.

Two split families, both grouped so that a held-out unit is never partly in
training:

``B`` — **chemotype hold-out (primary).**  Groups are the frozen gen6
single-linkage Tanimoto-0.7 super-clusters computed over all 190 bundle
structures.  Using the *frozen* map rather than an Eu-local one is the
conservative choice: a structure outside the Eu cohort can only ever *merge* two
Eu clusters, never split one, so the frozen labels make the hold-out stricter
(measured: 97 frozen chemotypes over the Eu structures, largest 46 extractants,
against 97 Eu-local ones, largest 36).

``A`` — **exact-extractant hold-out (secondary).**  Groups are canonical SMILES.
Reported as a sensitivity, never as the headline, because this cohort contains
homologue series and stereoisomer sets whose 2,048-bit achiral Morgan
fingerprints are bit-identical: an "unseen" extractant can have a training
partner at Tanimoto 1.0.  The repository's own leakage audit rates that CRITICAL.

The splitter is the repository's ``seeded_group_kfold`` — shuffle the unique
group labels, deal round robin — reused unchanged so that Gen12 folds are built
by the same algorithm as gen5-gen11's.  sklearn's ``GroupKFold`` is greedy by
size and was measured to leave 82 % of rows in the same fold across nominally
different seeds.

Inner validation is nested: within each outer training set the same grouped
splitter carves a validation block from the *training* chemotypes only.  No
hyperparameter, no early-stopping decision and no model choice ever sees an
outer test row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from . import paths  # noqa: F401
from .chemistry import fingerprint_matrix, max_similarity_to_reference
from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402

#: gen11's split seeds, reused so that split-to-split variation is comparable
#: across generations.  The cohort and the task differ, so the *folds* are new.
SPLIT_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
N_SPLITS = 5
N_INNER_SPLITS = 4
#: Fixed, independent of the split seed — gen5-gen11's convention.
MODEL_SEED_BASE = 42
FOLD_SEED_STRIDE = 1009
FOLD_SEED_OFFSET = 9_999_991
GROUP_COLUMNS = {"B": "chemotype", "A": "extractant"}


@dataclass(frozen=True)
class Fold:
    design: str
    seed: int
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    inner_train_index: np.ndarray
    inner_validation_index: np.ndarray
    held_out_groups: tuple[str, ...]

    @property
    def model_seed(self) -> int:
        return int(MODEL_SEED_BASE + self.fold * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET)


def build_folds(frame: pd.DataFrame, *, design: str = "B", seed: int,
                n_splits: int = N_SPLITS, n_inner: int = N_INNER_SPLITS) -> list[Fold]:
    if design not in GROUP_COLUMNS:
        raise KeyError(f"unknown split design {design!r}; have {sorted(GROUP_COLUMNS)}")
    groups = frame[GROUP_COLUMNS[design]].astype(str).to_numpy()
    folds: list[Fold] = []
    for k, (train_index, test_index) in enumerate(seeded_group_kfold(groups, n_splits, seed)):
        inner_groups = groups[train_index]
        # A deterministic inner seed that is a function of the outer split, so the
        # validation block is reproducible and independent of iteration order.
        inner_seed = int(seed) * 31 + k * 7919 + 17
        inner_train_rel, inner_validation_rel = next(
            iter(seeded_group_kfold(inner_groups, n_inner, inner_seed)))
        folds.append(Fold(
            design=design, seed=int(seed), fold=k,
            train_index=train_index, test_index=test_index,
            inner_train_index=train_index[inner_train_rel],
            inner_validation_index=train_index[inner_validation_rel],
            held_out_groups=tuple(sorted(set(groups[test_index]))),
        ))
    return folds


def all_folds(frame: pd.DataFrame, *, design: str = "B",
              seeds: Sequence[int] = SPLIT_SEEDS) -> list[Fold]:
    out: list[Fold] = []
    for seed in seeds:
        out.extend(build_folds(frame, design=design, seed=seed))
    return out


def assert_fold_integrity(frame: pd.DataFrame, folds: Sequence[Fold]) -> dict:
    """No extractant, ECFP cluster or chemotype may cross a train/test boundary.

    Checked for *every* design: even the exact-extractant design must not leak an
    ECFP cluster silently, so the overlap is measured and reported rather than
    assumed away.
    """
    columns = [c for c in ("extractant", "ecfp_cluster", "chemotype") if c in frame.columns]
    arrays = {c: frame[c].astype(str).to_numpy() for c in columns}
    report: dict = {"ok": True, "folds": [], "overlap_counts": {c: 0 for c in columns}}
    tested_per_seed: dict[int, set[int]] = {}
    for fold in folds:
        entry = {"design": fold.design, "seed": fold.seed, "fold": fold.fold,
                 "n_train": int(fold.train_index.size), "n_test": int(fold.test_index.size),
                 "n_inner_train": int(fold.inner_train_index.size),
                 "n_inner_validation": int(fold.inner_validation_index.size),
                 "leaks": {}}
        if set(fold.train_index.tolist()) & set(fold.test_index.tolist()):
            entry["leaks"]["row_overlap"] = True
            report["ok"] = False
        if set(fold.inner_validation_index.tolist()) & set(fold.test_index.tolist()):
            entry["leaks"]["inner_validation_in_test"] = True
            report["ok"] = False
        if set(fold.inner_train_index.tolist()) & set(fold.inner_validation_index.tolist()):
            entry["leaks"]["inner_overlap"] = True
            report["ok"] = False
        for column, values in arrays.items():
            shared = set(values[fold.train_index]) & set(values[fold.test_index])
            if shared:
                report["overlap_counts"][column] += len(shared)
                entry["leaks"][column] = sorted(shared)[:3]
                # Only the design's own grouping column is fatal; a weaker design
                # leaking a stricter unit is the measured cost of that design.
                if column == GROUP_COLUMNS[fold.design]:
                    report["ok"] = False
            inner_shared = (set(values[fold.inner_train_index])
                            & set(values[fold.inner_validation_index]))
            if inner_shared and column == GROUP_COLUMNS[fold.design]:
                entry["leaks"][f"inner_{column}"] = sorted(inner_shared)[:3]
                report["ok"] = False
        tested_per_seed.setdefault(fold.seed, set()).update(fold.test_index.tolist())
        report["folds"].append(entry)
    report["every_row_tested_once_per_seed"] = {
        int(seed): len(tested) == len(frame) for seed, tested in tested_per_seed.items()}
    if not all(report["every_row_tested_once_per_seed"].values()):
        report["ok"] = False
    return report


def similarity_table(frame: pd.DataFrame, folds: Sequence[Fold]) -> pd.DataFrame:
    """``max_train_tanimoto`` per (seed, fold, held-out extractant).  Target-free.

    Similarity is measured to that fold's *training* extractants only — never to
    the rest of the test set and never to the whole cohort — so the band a test
    extractant lands in is a property of the split, not of the corpus.
    """
    names, bits = fingerprint_matrix(frame)
    extractants = frame["extractant"].astype(str).to_numpy()
    rows: list[pd.DataFrame] = []
    for fold in folds:
        test_names = sorted(set(extractants[fold.test_index]))
        train_names = sorted(set(extractants[fold.train_index]))
        table = max_similarity_to_reference(test_names, train_names, names, bits)
        table.insert(0, "fold", fold.fold)
        table.insert(0, "split_seed", fold.seed)
        table.insert(0, "design", fold.design)
        table["n_train_extractants"] = len(train_names)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def iter_train_test(frame: pd.DataFrame, folds: Sequence[Fold]) -> Iterator[tuple[Fold, pd.DataFrame, pd.DataFrame]]:
    for fold in folds:
        yield fold, frame.iloc[fold.train_index], frame.iloc[fold.test_index]
