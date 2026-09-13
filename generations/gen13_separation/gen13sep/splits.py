"""Chemotype hold-out folds over cells, built with the repository's own splitter.

Design B (primary): the held-out unit is the frozen gen6 single-linkage Tanimoto-0.7
chemotype.  Every cell of every extractant in a held-out chemotype leaves training
together, so no bit-identical fingerprint, homologue or stereoisomer of a test
extractant is in training.  Design A (exact extractant, sensitivity only) is kept for
the same reason gen12 kept it: to price the leak a naive split would report.

Seeds and the round-robin dealing are gen11/gen12's, so split-to-split variation is
comparable across generations.  Inner validation, when a learner needs it, carves a
validation block from the *training* chemotypes only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from . import paths  # noqa: F401  (inserts src/ into sys.path)
from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402

SPLIT_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
N_SPLITS = 5
N_INNER_SPLITS = 4
MODEL_SEED_BASE = 42
FOLD_SEED_STRIDE = 1009
FOLD_SEED_OFFSET = 9_999_991
GROUP_COLUMNS = {"B": "chemotype", "A": "extractant", "BP": "chemotype", "BR": "chemotype", "BQ": "chemotype"}
#: design BP = chemotype hold-out (as B) with a publication-aware training mask: every training
#: cell whose publication also appears among the held-out cells is dropped, so the 64 condition
#: columns cannot act as a laboratory fingerprint (they identify a cell's publication with 94 %
#: 1-NN accuracy).  Inner splits are re-dealt on the masked training set.
PUBLICATION_MASKED_DESIGNS = ("BP",)
#: design BR = the size-matched control for BP: the same number of training cells as BP removes
#: is removed at random (seeded per fold), ignoring publications.  BP - BR isolates the
#: laboratory-fingerprint effect from the loss of training data.
RANDOM_MASKED_DESIGNS = ("BR",)
#: design BQ = the structure-matched control for BP: whole *training* publications (never the
#: held-out ones) are dropped at random until at least as many cells **and** at least as many
#: distinct condition series are gone as BP removes, so the control loses the same amount of
#: within-publication condition variation and not merely the same number of scattered cells.
RANDOM_PUBLICATION_MASKED_DESIGNS = ("BQ",)


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
        raise KeyError(f"unknown split design {design!r}")
    groups = frame[GROUP_COLUMNS[design]].astype(str).to_numpy()
    publications = frame["publication_id"].astype(str).to_numpy() if "publication_id" in frame.columns else None
    series = frame["condition_key"].astype(str).to_numpy() if "condition_key" in frame.columns else None
    folds: list[Fold] = []
    for k, (train_index, test_index) in enumerate(seeded_group_kfold(groups, n_splits, seed)):
        if design in PUBLICATION_MASKED_DESIGNS or design in RANDOM_MASKED_DESIGNS or design in RANDOM_PUBLICATION_MASKED_DESIGNS:
            if publications is None:
                raise KeyError("designs BP/BR/BQ need a publication_id column")
            held = set(publications[test_index])
            keep = np.array([publications[i] not in held for i in train_index], dtype=bool)
            n_drop = int((~keep).sum())
            if design in RANDOM_MASKED_DESIGNS:
                rng = np.random.default_rng(int(seed) * 7 + k * 101 + 3)
                drop = rng.choice(len(train_index), size=n_drop, replace=False)
                keep = np.ones(len(train_index), dtype=bool); keep[drop] = False
            elif design in RANDOM_PUBLICATION_MASKED_DESIGNS:
                if series is None:
                    raise KeyError("design BQ needs a condition_key column")
                rng = np.random.default_rng(int(seed) * 11 + k * 131 + 5)
                train_pubs = publications[train_index]
                train_series = series[train_index]
                # what BP would remove: cells and distinct condition series
                bp_mask = ~keep
                n_series_drop = len(set(zip(train_pubs[bp_mask], train_series[bp_mask])))
                candidates = [p for p in np.unique(train_pubs) if p not in held]
                rng.shuffle(candidates)
                sizes = {p: int((train_pubs == p).sum()) for p in candidates}
                n_series = {p: len(set(train_series[train_pubs == p])) for p in candidates}
                # greedy closest-fit so the control does not overshoot the amount of data BP removes
                dropped, n_gone, s_gone = set(), 0, 0
                remaining = list(candidates)
                while remaining and (n_gone < n_drop or s_gone < n_series_drop):
                    need = max(n_drop - n_gone, 1)
                    pub = min(remaining, key=lambda q: (abs(sizes[q] - need), sizes[q]))
                    remaining.remove(pub)
                    dropped.add(pub); n_gone += sizes[pub]; s_gone += n_series[pub]
                keep = np.array([p not in dropped for p in train_pubs], dtype=bool)
            train_index = train_index[keep]
        inner_groups = groups[train_index]
        inner_seed = int(seed) * 31 + k * 7919 + 17
        inner_train_rel, inner_validation_rel = next(iter(seeded_group_kfold(inner_groups, n_inner, inner_seed)))
        folds.append(Fold(design=design, seed=int(seed), fold=k,
                          train_index=train_index, test_index=test_index,
                          inner_train_index=train_index[inner_train_rel],
                          inner_validation_index=train_index[inner_validation_rel],
                          held_out_groups=tuple(sorted(set(groups[test_index])))))
    return folds


def all_folds(frame: pd.DataFrame, *, design: str = "B",
              seeds: Sequence[int] = SPLIT_SEEDS) -> list[Fold]:
    out: list[Fold] = []
    for seed in seeds:
        out.extend(build_folds(frame, design=design, seed=seed))
    return out


def assert_fold_integrity(frame: pd.DataFrame, folds: Sequence[Fold]) -> dict:
    """No extractant, ECFP cluster or chemotype may cross a train/test boundary in design B."""
    columns = [c for c in ("extractant", "ecfp_cluster", "chemotype") if c in frame.columns]
    arrays = {c: frame[c].astype(str).to_numpy() for c in columns}
    overlaps = {c: 0 for c in columns}
    for fold in folds:
        if np.intersect1d(fold.train_index, fold.test_index).size:
            raise AssertionError("train/test index overlap")
        for c in columns:
            overlaps[c] += len(set(arrays[c][fold.train_index]) & set(arrays[c][fold.test_index]))
        inner = set(fold.inner_train_index) & set(fold.inner_validation_index)
        if inner:
            raise AssertionError("inner train/validation overlap")
        if not set(fold.inner_train_index) | set(fold.inner_validation_index) <= set(fold.train_index):
            raise AssertionError("inner folds leave the outer training set")
    by_design = {f.design for f in folds}
    if by_design <= {"B", "BP", "BR", "BQ"} and any(v for v in overlaps.values()):
        raise AssertionError(f"design {by_design} leaks: {overlaps}")
    pub_overlap = 0
    if by_design == {"BP"} and "publication_id" in frame.columns:
        pubs = frame["publication_id"].astype(str).to_numpy()
        for fold in folds:
            pub_overlap += len(set(pubs[fold.train_index]) & set(pubs[fold.test_index]))
        if pub_overlap:
            raise AssertionError(f"design BP leaks publications: {pub_overlap}")
    return {"overlaps": overlaps, "publication_overlap": pub_overlap, "n_folds": len(folds)}


def fold_plan(frame: pd.DataFrame, folds: Sequence[Fold]) -> pd.DataFrame:
    rows = []
    for f in folds:
        for i in f.test_index:
            rows.append({"design": f.design, "split_seed": f.seed, "fold": f.fold,
                         "cell_id": frame["cell_id"].iat[i]})
    return pd.DataFrame(rows)


def max_train_tanimoto(frame: pd.DataFrame, folds: Sequence[Fold], fingerprints: np.ndarray) -> pd.DataFrame:
    """Per (seed, fold, test cell): max Tanimoto of its extractant to any training extractant."""
    fp = fingerprints.astype(bool)
    rows = []
    for f in folds:
        tr_ext = frame["extractant"].to_numpy()[f.train_index]
        _, first = np.unique(tr_ext, return_index=True)
        tr_fp = fp[f.train_index][first]
        te_fp = fp[f.test_index]
        inter = te_fp.astype(np.int32) @ tr_fp.astype(np.int32).T
        union = te_fp.sum(1)[:, None] + tr_fp.sum(1)[None, :] - inter
        sim = (inter / np.maximum(union, 1)).max(axis=1)
        for i, s in zip(f.test_index, sim):
            rows.append({"split_seed": f.seed, "fold": f.fold, "cell_id": frame["cell_id"].iat[i],
                         "max_train_tanimoto": float(s)})
    out = pd.DataFrame(rows)
    out["band"] = pd.cut(out["max_train_tanimoto"], [-0.01, 0.40, 0.60, 1.01], labels=["far", "mid", "near"]).astype(str)
    return out


def iter_train_test(frame: pd.DataFrame, folds: Sequence[Fold]) -> Iterator[tuple[Fold, np.ndarray, np.ndarray]]:
    for f in folds:
        yield f, f.train_index, f.test_index
