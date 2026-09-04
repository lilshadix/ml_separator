"""Contract tests for the gen8 curve baselines.

The important one is :func:`test_no_target_leak`: every adapter is asked to predict
twice for the same block, once with the block's ``log_D`` column intact and once with
it replaced by nonsense, and the two answers must be bit-identical.  An adapter that
peeked at the target of a row it was not allowed to measure cannot pass it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lanthanide_separation.gen8.adapters import AdaptContext          # noqa: E402
from lanthanide_separation.gen8.curve_baselines import (              # noqa: E402
    CurveShape, LocalCurveGP, build_curve_adapters, natural_spline_basis,
)

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def fold():
    from gen8_curve_baselines import COHORT, MEMBERSHIP, OOF, load, make_fold_trainer, slim_features
    if not (COHORT.exists() and OOF.exists() and MEMBERSHIP.exists()):
        pytest.skip("gen7 cache / gen8 series artefacts not present")
    prepared, oof = load(COHORT, OOF, MEMBERSHIP)
    oof = oof[(oof["split_seed"] == 104729) & (oof["fold"] == 0)]
    trainer = make_fold_trainer(prepared, oof, verify=False)
    columns = [c for c in slim_features(prepared) if c != "row_id"]
    merged = oof.merge(prepared[["row_id", *columns]], on="row_id", how="left",
                       validate="many_to_one")
    return trainer, merged


def _one_block(merged: pd.DataFrame) -> pd.DataFrame:
    sizes = merged.groupby("extractant").size()
    ligand = sizes[(sizes >= 12) & (sizes <= 200)].index[0]
    return merged[merged["extractant"] == ligand].reset_index(drop=True)


def test_fold_plan_matches_the_frozen_oof():
    from gen8_curve_baselines import COHORT, MEMBERSHIP, OOF, load, make_fold_trainer
    if not (COHORT.exists() and OOF.exists() and MEMBERSHIP.exists()):
        pytest.skip("artefacts not present")
    prepared, oof = load(COHORT, OOF, MEMBERSHIP)
    make_fold_trainer(prepared, oof, verify=True)      # asserts internally


def test_no_target_leak(fold):
    trainer, merged = fold
    train, y_train, model_seed = trainer(104729, 0)
    adapters = build_curve_adapters()
    for adapter in adapters:
        adapter.fit_fold(train, y_train, split_seed=104729, fold=0, model_seed=model_seed)
    block = _one_block(merged)
    truth = block["log_D"].to_numpy(dtype=float)
    prediction = block["prediction"].to_numpy(dtype=float)
    context = AdaptContext(split_seed=104729, fold=0, extractant=block["extractant"].iloc[0])
    selected = np.array([0, 3, 7], dtype=int)
    poisoned = block.copy()
    poisoned["log_D"] = np.random.default_rng(0).normal(50.0, 10.0, len(block))
    for adapter in adapters:
        honest = np.asarray(adapter.predict(block, prediction, selected, truth[selected], context),
                            dtype=float)
        # a fresh cache, so the poisoned call cannot be answered from the honest one
        adapter._cache = {}
        peeked = np.asarray(adapter.predict(poisoned, prediction, selected, truth[selected],
                                            context), dtype=float)
        assert np.array_equal(honest, peeked), f"{adapter.name} read the target column"
        assert np.isfinite(honest).all(), f"{adapter.name} returned a non-finite value"


def test_k0_is_reachable_and_deterministic(fold):
    trainer, merged = fold
    train, y_train, model_seed = trainer(104729, 0)
    block = _one_block(merged)
    prediction = block["prediction"].to_numpy(dtype=float)
    context = AdaptContext(split_seed=104729, fold=0, extractant=block["extractant"].iloc[0])
    empty = np.array([], dtype=int)
    for adapter in build_curve_adapters():
        adapter.fit_fold(train, y_train, split_seed=104729, fold=0, model_seed=model_seed)
        first = np.asarray(adapter.predict(block, prediction, empty, np.array([]), context))
        adapter._cache = {}
        second = np.asarray(adapter.predict(block, prediction, empty, np.array([]), context))
        assert np.isfinite(first).all()
        assert np.array_equal(first, second)


def test_shape_only_ignores_the_measurements(fold):
    trainer, merged = fold
    train, y_train, model_seed = trainer(104729, 0)
    adapter = CurveShape(mode="spline", adapt="none")
    adapter.fit_fold(train, y_train, split_seed=104729, fold=0, model_seed=model_seed)
    block = _one_block(merged)
    truth = block["log_D"].to_numpy(dtype=float)
    prediction = block["prediction"].to_numpy(dtype=float)
    context = AdaptContext(split_seed=104729, fold=0, extractant=block["extractant"].iloc[0])
    empty = np.asarray(adapter.predict(block, prediction, np.array([], dtype=int),
                                       np.array([]), context))
    with_k = np.asarray(adapter.predict(block, prediction, np.array([1, 2]), truth[[1, 2]],
                                        context))
    assert np.array_equal(empty, with_k)


def test_correction_preserves_the_claiming_curve_level(fold):
    """Within the curve that *claims* a row, the correction sums to zero exactly.

    That is the invariant the whole design rests on: the shape correction may not
    move a curve's level, because the level is what the k measurements are for.  It
    holds for the curve that claims the row — the highest-priority axis the row sits
    on.  A row claimed by its extractant titration is deliberately *not* re-levelled
    inside its acid titration as well; that shift is the extractant physics showing
    up on the other axis, not a leak of level information.
    """
    from lanthanide_separation.gen8.curve_baselines import CURVE_COLUMN
    trainer, merged = fold
    train, y_train, model_seed = trainer(104729, 0)
    adapter = CurveShape(mode="spline", adapt="none")
    adapter.fit_fold(train, y_train, split_seed=104729, fold=0, model_seed=model_seed)
    checked = 0
    for ligand, block in merged.groupby("extractant"):
        block = block.reset_index(drop=True)
        prediction = block["prediction"].to_numpy(dtype=float)
        context = AdaptContext(split_seed=104729, fold=0, extractant=ligand)
        correction = adapter._correction(block, prediction, context)
        higher = np.zeros(len(block), dtype=bool)
        for axis in adapter.priority:
            curve = block[CURVE_COLUMN.format(axis=axis)].to_numpy(dtype=object)
            on_axis = curve != ""
            for name in np.unique(curve[on_axis]):
                rows = curve == name
                if higher[rows].any():          # partly claimed by a higher axis
                    continue
                assert abs(float(correction[rows].mean())) < 1e-9
                checked += 1
            higher |= on_axis
    assert checked > 20


def test_natural_spline_is_linear_beyond_the_boundary_knots():
    knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x = np.array([5.0, 6.0, 7.0, 8.0])
    basis = natural_spline_basis(x, knots)
    second = np.diff(basis, n=2, axis=0)
    assert np.abs(second).max() < 1e-8


def test_gp_reduces_to_offset_correction_at_k_equals_one(fold):
    trainer, merged = fold
    train, y_train, model_seed = trainer(104729, 0)
    gp = LocalCurveGP()
    gp.fit_fold(train, y_train, split_seed=104729, fold=0, model_seed=model_seed)
    block = _one_block(merged)
    truth = block["log_D"].to_numpy(dtype=float)
    prediction = block["prediction"].to_numpy(dtype=float)
    context = AdaptContext(split_seed=104729, fold=0, extractant=block["extractant"].iloc[0])
    selected = np.array([4], dtype=int)
    values = np.asarray(gp.predict(block, prediction, selected, truth[selected], context))
    offset = prediction + (truth[selected][0] - prediction[selected][0])
    assert np.abs(values - offset).max() < 5e-3
