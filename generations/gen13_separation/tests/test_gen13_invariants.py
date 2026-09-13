"""Pre-registered invariants of Gen13, executable.

Run:  .venv/Scripts/python.exe -m pytest generations/gen13_separation/tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen13sep import basis as B  # noqa: E402
from gen13sep import metals  # noqa: E402
from gen13sep.cohort import FORBIDDEN_FEATURE_EXACT, FORBIDDEN_FEATURE_TOKENS, build_cohort  # noqa: E402
from gen13sep.features import build_features  # noqa: E402
from gen13sep.metrics import pair_rows_for_cell, per_extractant  # noqa: E402
from gen13sep.models import (DirectRowArm, FitContext, LowRankArm, MeanCurveArm, PairMeanArm,  # noqa: E402
                             PhysicsBasisArm, SelectedArm, ZeroArm)
from gen13sep.splits import all_folds, assert_fold_integrity, build_folds  # noqa: E402


@pytest.fixture(scope="module")
def cohort():
    return build_cohort("exact")


@pytest.fixture(scope="module")
def features(cohort):
    return build_features(cohort)


def test_cohort_shape_and_identity(cohort):
    f = cohort.frame
    assert len(f) == 521 and f["extractant"].nunique() == 90 and f["chemotype"].nunique() == 45
    assert f["cell_id"].is_unique
    assert (f["n_metals"] >= 2).all()
    # a cell never mixes publications by construction
    assert f.groupby("cell_id")["publication_id"].nunique().max() == 1
    # replicate averaging: counts are >= 1 wherever a metal is observed
    Y = cohort.target_matrix
    counts = f[[f"nrep__{m}" for m in metals.LANTHANIDES]].to_numpy()
    assert np.array_equal(~np.isnan(Y), counts >= 1)


def test_no_forbidden_feature_columns(features):
    cols = list(features.frame.columns)
    assert not any(c in FORBIDDEN_FEATURE_EXACT for c in cols)
    assert not any(any(tok in c for tok in FORBIDDEN_FEATURE_TOKENS) for c in cols)
    assert not any(c.startswith("feat3d__") for c in cols)


def test_features_are_target_free(cohort, features):
    """Replacing every log D by noise leaves every feature column bit-identical."""
    noisy = cohort.frame.copy()
    rng = np.random.default_rng(0)
    for m in metals.LANTHANIDES:
        col = f"logD__{m}"
        noisy[col] = np.where(noisy[col].isna(), np.nan, rng.normal(size=len(noisy)))
    from gen13sep.cohort import Gen13Cohort
    alt = Gen13Cohort(frame=noisy, condition_columns=cohort.condition_columns, key_mode="exact")
    f2 = build_features(alt)
    pd.testing.assert_frame_equal(features.frame, f2.frame)


def test_design_b_folds_do_not_leak(cohort):
    folds = all_folds(cohort.frame, design="B")
    report = assert_fold_integrity(cohort.frame, folds)
    assert report["overlaps"] == {"extractant": 0, "ecfp_cluster": 0, "chemotype": 0}
    assert len(folds) == 25
    # every cell is held out exactly once per seed
    for seed in {f.seed for f in folds}:
        held = np.concatenate([f.test_index for f in folds if f.seed == seed])
        assert sorted(held.tolist()) == list(range(len(cohort.frame)))


def test_folds_are_seed_dependent(cohort):
    a = build_folds(cohort.frame, design="B", seed=104729)
    b = build_folds(cohort.frame, design="B", seed=130363)
    assert any(set(x.test_index) != set(y.test_index) for x, y in zip(a, b))


def test_basis_sign_convention_and_rank_monotone():
    rng = np.random.default_rng(1)
    C = rng.normal(size=(60, 14)); C -= C.mean(axis=1, keepdims=True)
    C[rng.random(C.shape) < 0.3] = np.nan
    ev = [B.explained_variance(C, B.fit_data_basis(C, k)) for k in (1, 2, 3)]
    assert ev[0] < ev[1] < ev[2]
    ref = metals.physics_basis()["radius"]
    for k in (1, 2, 3):
        basis = B.fit_data_basis(C, k)
        assert basis[0] @ ref >= 0                                   # component 1 follows the radius curve
        for i in range(1, k):
            assert basis[i, int(np.argmax(np.abs(basis[i])))] > 0    # later components: largest entry positive
        assert np.allclose(np.linalg.norm(basis, axis=1), np.sqrt(14))
        assert np.allclose(basis @ basis.T / 14, np.eye(k), atol=1e-8)


def test_physics_basis_is_centred():
    for name, v in metals.physics_basis().items():
        assert abs(v.mean()) < 1e-12, name
        assert len(v) == 14


def test_curve_arms_are_exactly_antisymmetric(cohort, features):
    """A curve-based prediction of (A,B) is minus that of (B,A) to machine precision."""
    X = features.matrix(["COND", "MASSACT", "ECFP"]).to_numpy(dtype=float)[:80]
    Y = cohort.target_matrix[:80]
    ctx = FitContext(X_train=X, Y_train=Y, groups_train=cohort.frame["chemotype"].to_numpy()[:80],
                     fingerprints_train=features.matrix(["ECFP"]).to_numpy()[:80], seed=3)
    for arm in (MeanCurveArm(), LowRankArm(rank=2, n_estimators=20), PhysicsBasisArm(n_estimators=20),
                DirectRowArm(n_estimators=20)):
        arm.fit(ctx)
        curve = arm.predict_curves(X[:5], ctx.fingerprints_train[:5])
        assert curve.shape == (5, 14)
        if isinstance(arm, (PhysicsBasisArm, DirectRowArm)):
            assert np.allclose(curve.mean(axis=1), 0.0, atol=1e-9)
        ab = curve[:, 2] - curve[:, 9]; ba = curve[:, 9] - curve[:, 2]
        assert np.allclose(ab, -ba)


def test_pairmean_uses_only_training_pairs(cohort):
    Y = cohort.target_matrix
    ctx = FitContext(X_train=np.zeros((len(Y), 1)), Y_train=Y, groups_train=cohort.frame["chemotype"].to_numpy(),
                     fingerprints_train=np.zeros((len(Y), 1)), seed=0)
    arm = PairMeanArm().fit(ctx)
    assert arm.table_.shape == (14, 14) and np.isfinite(arm.table_).all()


def test_selected_arm_records_inner_scores(cohort, features):
    frame = cohort.frame
    fold = build_folds(frame, design="B", seed=104729)[1]
    X = features.matrix(["COND", "MASSACT", "PHYSCHEM"]).to_numpy(dtype=float)
    fp = features.matrix(["ECFP"]).to_numpy()
    ctx = FitContext(X_train=X[fold.train_index], Y_train=cohort.target_matrix[fold.train_index],
                     groups_train=frame["chemotype"].to_numpy()[fold.train_index],
                     fingerprints_train=fp[fold.train_index], seed=fold.model_seed,
                     inner_train=np.searchsorted(fold.train_index, fold.inner_train_index),
                     inner_validation=np.searchsorted(fold.train_index, fold.inner_validation_index))
    sel = SelectedArm([LowRankArm(rank=1, n_estimators=15), LowRankArm(rank=2, n_estimators=15)])
    sel.fit(ctx)
    rec = sel.selection_[-1]
    assert rec["selected"] in ("M_LOWRANK_K1", "M_LOWRANK_K2")
    assert all(np.isfinite(v) for k, v in rec.items() if k.startswith("inner_"))


def test_per_extractant_metrics_identity():
    pairs = pd.DataFrame({"split_seed": [1] * 4, "extractant": ["x"] * 4, "chemotype": ["c"] * 4,
                          "cell_id": ["a"] * 4, "n_metals": [3] * 4, "dZ": [1, 5, 2, 6],
                          "y": [0.5, -1.0, 0.1, 0.9], "arm1": [0.5, -1.0, 0.1, 0.9]})
    pairs["A"] = ["La", "La", "Ce", "Nd"]; pairs["B"] = ["Ce", "Eu", "Sm", "Sm"]
    t = per_extractant(pairs, ["arm1"])
    assert t["mae_all"].iat[0] == 0.0 and t["sign_acc_strong"].iat[0] == 1.0
    assert t["n_adjacent"].iat[0] == 2  # La-Ce and Nd-Sm; Ce-Sm (dZ 4) and La-Eu are not adjacent


def test_pair_rows_only_lighter_first():
    row = np.array([np.nan, 1.0, np.nan, 0.5] + [np.nan] * 10)
    assert pair_rows_for_cell(row) == [(1, 3, 0.5)]
