"""LT-13 continuity hard gate (IMPLEMENTATION_PLAN.md §4; PRE_REGISTRATION.md §4.4): on 2 BP folds, with the real
400-tree configuration,

* RowForest(target='raw', K4), centred per cell after prediction, equals the in-process gen13 ``DirectRowArm`` on
  LEAN209 (gen13sep/models.py:240-269) to 1e-12;
* RowForest(target='centred', K4), its 14-metal output centred per cell, equals gen13
  ``CentredRowArm(metal_features='basic', aggregate='mean', weights='balanced')`` on LEAN209 to 1e-12.

Predictions only; no metric."""
from __future__ import annotations

import numpy as np
import pytest

import gen13sep.models as M13
from gen13sep.arms_stage2 import CentredRowArm
from gen201 import data as D
from gen201 import folds as FO
from gen201 import learners as L

#: the two giant BP folds (fewest training cells, so the 400-tree fits stay fast)
FOLDS = ((104729, 0), (155921, 3))


@pytest.fixture(scope="module")
def two_folds(bench):
    F = FO.build(bench, seeds=(104729, 155921), designs=("BP",))["BP"]
    out = [f for f in F if (int(f.seed), int(f.fold)) in FOLDS]
    assert len(out) == 2
    return out


def _ctx(bench, X, f):
    tr = np.asarray(f.train_index)
    return M13.FitContext(X_train=X[tr], Y_train=np.asarray(bench.Y, float)[tr], groups_train=bench.groups[tr],
                          fingerprints_train=D.ecfp(bench)[tr], seed=f.model_seed)


@pytest.mark.parametrize("which", [0, 1])
def test_raw_k4_equals_gen13_direct_row(bench, two_folds, which):
    assert L.N_ESTIMATORS == 400 and L.CONFIGS["K4"] == {"min_samples_leaf": 2, "max_features": 0.5}
    f = two_folds[which]
    X = D.X_lean(bench)
    te = np.asarray(f.test_index)
    ref = M13.DirectRowArm().fit(_ctx(bench, X, f)).predict_curves(X[te])
    rf = L.RowForest("raw", "K4", f.model_seed).fit(X, bench.Y, bench.groups, f.train_index)
    P = rf.predict(X, te)
    got = P - P.mean(axis=1, keepdims=True)
    assert rf.n_estimators == 400
    assert np.max(np.abs(got - ref)) <= 1e-12
    # the raw output is not centred (A predicts absolute log D)
    assert np.max(np.abs(P.mean(axis=1))) > 1e-3


@pytest.mark.parametrize("which", [0, 1])
def test_centred_k4_equals_gen13_centred_row_basic(bench, two_folds, which):
    f = two_folds[which]
    X = D.X_lean(bench)
    te = np.asarray(f.test_index)
    arm = CentredRowArm(metal_features="basic", aggregate="mean", weights="balanced")
    ref = arm.fit(_ctx(bench, X, f)).predict_curves(X[te])
    rf = L.RowForest("centred", "K4", f.model_seed).fit(X, bench.Y, bench.groups, f.train_index)
    P = rf.predict(X, te)
    got = P - P.mean(axis=1, keepdims=True)
    assert np.max(np.abs(got - ref)) <= 1e-12
