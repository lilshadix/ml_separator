"""LT-02: the 4-fold chemotype inner cross-fit re-applies the design mask inside every inner fold: held-out
publications are removed (BP), copy groups too (BPG); inner folds stay inside the outer training set; an
inner fold with < 20 masked inner-training cells is flagged skipped (IMPLEMENTATION_PLAN.md §4)."""
from __future__ import annotations

import numpy as np
import pytest

from gen201 import folds as FO
from gen20k import splits as GS
from gen20k.cov import inner_mask
from lanthanide_separation.gen6.cohorts import seeded_group_kfold


def _arrays(bench):
    f = bench.frame
    return (f["publication_id"].astype(str).to_numpy(), f["chemotype"].astype(str).to_numpy(),
            GS.copy_groups_of(f))


@pytest.mark.parametrize("design", FO.DESIGNS)
def test_inner_folds_partition_and_mask(bench, folds, design):
    pub, chem, grp = _arrays(bench)
    for fo in folds[design]:
        tr = set(np.asarray(fo.train_index).tolist())
        inner = FO.inner_folds(bench, design, fo)
        assert len(inner) == FO.N_INNER == 4
        val_union = []
        for inn in inner:
            itr, iva, raw = set(inn.train_pos.tolist()), set(inn.val_pos.tolist()), set(inn.train_pos_unmasked.tolist())
            assert itr <= raw <= tr and iva <= tr
            assert not (itr & iva) and not (raw & iva)
            assert raw | iva == tr                                    # dealing covers the training fold
            assert not (set(chem[list(raw)]) & set(chem[list(iva)]))  # chemotype grouping (design A included)
            if design in ("BP", "BPG"):
                assert not (set(pub[list(itr)]) & set(pub[list(iva)]))
            else:
                assert itr == raw                                     # no inner mask outside BP/BPG
            if design == "BPG":
                assert not (set(grp[list(itr)]) & set(grp[list(iva)]))
            assert inn.skipped == (len(itr) < FO.MIN_INNER_TRAIN or len(iva) == 0)
            val_union.extend(iva)
        assert sorted(val_union) == sorted(tr)                        # every training cell validated once


def test_inner_folds_equal_gen20k_cov_split(bench, folds):
    """The dealing is gen20k.cov.inner_oof_curves' split: seeded_group_kfold(bench.groups[train], 4,
    inner_seed(seed, fold)) with gen20k.cov.inner_mask, in the same order."""
    for fo in folds["BP"][:5] + folds["BPG"][:5]:
        tr = np.asarray(fo.train_index)
        ref = list(seeded_group_kfold(bench.groups[tr].astype(str), GS.N_INNER, GS.inner_seed(fo.seed, fo.fold)))
        got = FO.inner_folds(bench, fo.design, fo)
        for (a, b), inn in zip(ref, got):
            assert sorted(inner_mask(bench, fo.design, tr[a], tr[b]).tolist()) == inn.train_pos.tolist()
            assert sorted(tr[b].tolist()) == inn.val_pos.tolist()


def test_bp_inner_mask_removes_publications_that_the_unmasked_split_would_leak(bench, folds):
    pub, _, _ = _arrays(bench)
    removed = 0
    for fo in folds["BP"]:
        for inn in FO.inner_folds(bench, "BP", fo):
            gone = np.setdiff1d(inn.train_pos_unmasked, inn.train_pos)
            removed += len(gone)
            held = set(pub[inn.val_pos])
            assert all(pub[i] in held for i in gone)
    assert removed > 0


def test_skipped_inner_fold_is_the_documented_one(bench, folds):
    sk = [(fo.seed, fo.fold, inn.j, len(inn.train_pos)) for fo in folds["BP"] for inn in FO.inner_folds(bench, "BP", fo)
          if inn.skipped]
    assert sk == [(155921, 2, 3, 18)]
    assert not [1 for fo in folds["B"] for inn in FO.inner_folds(bench, "B", fo) if inn.skipped]


def test_inner_folds_refuse_a_foreign_design(bench, folds):
    with pytest.raises(ValueError):
        FO.inner_folds(bench, "B", folds["BP"][0])
