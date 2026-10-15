"""Fold plans for gen20: the frozen gen13 designs plus the copy-group-masked variant ``BPG``.

* ``A, B, BR, BQ, BP`` are imported from ``gen13sep.splits.all_folds`` unchanged.
* ``BPG`` starts from BP and additionally drops every training cell whose gen19
  cross-publication **copy group** (``data_audit/leakage_publication_components.csv``,
  ``group_cross_publication_copy``) appears among the held-out cells.  The splits audit found 45
  gen13 cells (5 publications, 3 groups, 21 extractants, 5 chemotypes) whose publication shares a
  copy group with another publication, so a series copied between two papers can survive the raw
  BP mask.  BPG is a registered sensitivity; BP remains the selecting design for continuity.
* ``assert_design_integrity`` re-checks every fold set of one design (gen13's check only fires
  the publication test for pure-BP sets, and the frozen benches never call it).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import bootstrap  # noqa: F401
from gen13sep.splits import Fold, SPLIT_SEEDS, all_folds, assert_fold_integrity  # noqa: E402
from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402

GEN13_DESIGNS: tuple[str, ...] = ("A", "B", "BR", "BQ", "BP")
DESIGNS: tuple[str, ...] = GEN13_DESIGNS + ("BPG",)
COPY_GROUPS_CSV = (bootstrap.ROOT / "generations" / "gen19_chem_transfer" / "data_audit"
                   / "leakage_publication_components.csv")
N_INNER = 4


def inner_seed(seed: int, fold: int) -> int:
    """gen13's inner-split seed formula (``splits.py``: ``seed*31 + k*7919 + 17``)."""
    return int(seed) * 31 + int(fold) * 7919 + 17


def copy_group_map() -> dict[str, str]:
    t = pd.read_csv(COPY_GROUPS_CSV)
    pub_col = "g19_publication_id" if "g19_publication_id" in t.columns else t.columns[0]
    return dict(zip(t[pub_col].astype(str), t["group_cross_publication_copy"].astype(str)))


def copy_groups_of(frame: pd.DataFrame) -> np.ndarray:
    m = copy_group_map()
    pubs = frame["publication_id"].astype(str).to_numpy()
    missing = sorted({p for p in pubs if p not in m})
    if missing:
        raise KeyError(f"{len(missing)} cohort publications have no copy group: {missing[:5]}")
    return np.array([m[p] for p in pubs])


def _redeal_inner(frame: pd.DataFrame, fold: Fold, train_index: np.ndarray) -> Fold:
    groups = frame["chemotype"].astype(str).to_numpy()[train_index]
    tr_rel, va_rel = next(iter(seeded_group_kfold(groups, N_INNER, inner_seed(fold.seed, fold.fold))))
    return replace(fold, train_index=train_index, inner_train_index=train_index[tr_rel],
                   inner_validation_index=train_index[va_rel])


def bpg_folds(frame: pd.DataFrame, seeds=SPLIT_SEEDS) -> list[Fold]:
    groups = copy_groups_of(frame)
    out = []
    for f in all_folds(frame, design="BP", seeds=seeds):
        held = set(groups[f.test_index])
        keep = np.array([groups[i] not in held for i in f.train_index], dtype=bool)
        out.append(replace(_redeal_inner(frame, f, f.train_index[keep]), design="BPG"))
    return out


def folds_for(frame: pd.DataFrame, design: str, seeds=SPLIT_SEEDS) -> list[Fold]:
    if design in GEN13_DESIGNS:
        folds = all_folds(frame, design=design, seeds=seeds)
    elif design == "BPG":
        folds = bpg_folds(frame, seeds)
    else:
        raise KeyError(design)
    assert_design_integrity(frame, folds, design)
    return folds


def assert_design_integrity(frame: pd.DataFrame, folds, design: str) -> dict:
    """Zero chemotype crossing for B-family designs, zero extractant crossing for A, zero
    publication crossing for BP/BPG, zero copy-group crossing for BPG, inner folds inside train."""
    chem = frame["chemotype"].astype(str).to_numpy()
    ext = frame["extractant"].astype(str).to_numpy()
    pub = frame["publication_id"].astype(str).to_numpy()
    grp = copy_groups_of(frame) if design == "BPG" else None
    rep = {"design": design, "n_folds": len(folds), "chemotype_overlap": 0, "extractant_overlap": 0,
           "publication_overlap": 0, "copy_group_overlap": 0, "min_train": None, "max_train": None}
    sizes = []
    for f in folds:
        if f.design != design:
            raise AssertionError(f"fold design {f.design} != {design}")
        tr, te = f.train_index, f.test_index
        if np.intersect1d(tr, te).size:
            raise AssertionError("train/test overlap")
        if not (set(f.inner_train_index) | set(f.inner_validation_index)) <= set(tr):
            raise AssertionError("inner folds leave the training set")
        if set(f.inner_train_index) & set(f.inner_validation_index):
            raise AssertionError("inner train/validation overlap")
        rep["extractant_overlap"] += len(set(ext[tr]) & set(ext[te]))
        if design != "A":
            rep["chemotype_overlap"] += len(set(chem[tr]) & set(chem[te]))
        if design in ("BP", "BPG"):
            rep["publication_overlap"] += len(set(pub[tr]) & set(pub[te]))
        if design == "BPG":
            rep["copy_group_overlap"] += len(set(grp[tr]) & set(grp[te]))
        sizes.append(len(tr))
    rep["min_train"], rep["max_train"] = int(min(sizes)), int(max(sizes))
    if rep["extractant_overlap"] or rep["chemotype_overlap"] or rep["publication_overlap"] or rep["copy_group_overlap"]:
        raise AssertionError(f"design {design} leaks: {rep}")
    if design in GEN13_DESIGNS:
        assert_fold_integrity(frame, folds)
    return rep


def fold_record(frame: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    for f in folds:
        for i in f.test_index:
            rows.append({"design": f.design, "split_seed": f.seed, "fold": f.fold, "role": "test",
                         "cell_id": frame["cell_id"].iat[i]})
        for i in f.train_index:
            rows.append({"design": f.design, "split_seed": f.seed, "fold": f.fold, "role": "train",
                         "cell_id": frame["cell_id"].iat[i]})
    return pd.DataFrame(rows)
