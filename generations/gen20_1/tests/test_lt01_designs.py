"""LT-01: no forbidden overlap in the six designs; each cell is tested exactly once per seed in every design;
one gen20k.splits.folds_for call per design (IMPLEMENTATION_PLAN.md §4)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gen201 import folds as FO
from gen20k import splits as GS


@pytest.mark.parametrize("design", FO.DESIGNS)
def test_no_forbidden_overlap(bench, folds, design):
    F = folds[design]
    assert len(F) == 25
    assert {(f.seed, f.fold) for f in F} == {(s, k) for s in FO.SEEDS for k in range(5)}
    assert all(f.design == design for f in F)
    rep = GS.assert_design_integrity(bench.frame, F, design)
    assert rep["extractant_overlap"] == 0
    if design != "A":
        assert rep["chemotype_overlap"] == 0
    if design in ("BP", "BPG"):
        assert rep["publication_overlap"] == 0
    if design == "BPG":
        assert rep["copy_group_overlap"] == 0
    for f in F:
        assert np.intersect1d(f.train_index, f.test_index).size == 0


@pytest.mark.parametrize("design", FO.DESIGNS)
def test_each_cell_tested_once_per_seed(bench, folds, design):
    n = len(bench.frame)
    assert FO.tested_once_per_seed({design: folds[design]}, n)[design]
    for s in FO.SEEDS:
        held = np.concatenate([np.asarray(f.test_index) for f in folds[design] if f.seed == s])
        assert len(held) == n and len(set(held.tolist())) == n


def test_b_family_test_sets_equal_bp_and_a_differs(folds):
    g = FO.gate_d(folds)
    assert g["pass"], g
    assert g["n_equal_to_BP"] == {"B": 25, "BR": 25, "BQ": 25, "BPG": 25, "A": 0}


def test_masks_only_shrink_training(folds):
    """BP, BR, BQ and BPG drop training cells from B's (same test cells); BPG's training set is inside BP's."""
    b = {(f.seed, f.fold): set(f.train_index.tolist()) for f in folds["B"]}
    bp = {(f.seed, f.fold): set(f.train_index.tolist()) for f in folds["BP"]}
    for d in ("BP", "BR", "BQ", "BPG"):
        for f in folds[d]:
            assert set(f.train_index.tolist()) <= b[(f.seed, f.fold)]
    for f in folds["BPG"]:
        assert set(f.train_index.tolist()) <= bp[(f.seed, f.fold)]


def test_build_calls_folds_for_once_per_design(bench, monkeypatch):
    calls = []
    real = GS.folds_for

    def spy(frame, design, seeds=GS.SPLIT_SEEDS):
        calls.append(design)
        return real(frame, design, seeds=seeds)
    monkeypatch.setattr(GS, "folds_for", spy)
    out = FO.build(bench, seeds=(FO.SEEDS[0],))
    assert calls == list(FO.DESIGNS)
    assert set(out) == set(FO.DESIGNS)


def test_injected_test_cell_in_training_is_caught(bench, folds):
    f = folds["BP"][0]
    bad = replace(f, train_index=np.sort(np.r_[f.train_index, f.test_index[:1]]))
    with pytest.raises(AssertionError):
        GS.assert_design_integrity(bench.frame, [bad] + folds["BP"][1:], "BP")


def test_injected_publication_crossing_is_caught(bench, folds):
    """A BP training cell that shares a publication with a held-out cell is a leak."""
    pub = bench.frame["publication_id"].astype(str).to_numpy()
    f = folds["BP"][0]
    held = set(pub[f.test_index])
    crossing = [i for i in folds["B"][0].train_index if pub[i] in held]
    assert crossing, "design B keeps publication-sharing training cells in this fold"
    bad = replace(f, train_index=np.sort(np.r_[f.train_index, crossing[:1]]))
    with pytest.raises(AssertionError):
        GS.assert_design_integrity(bench.frame, [bad] + folds["BP"][1:], "BP")
