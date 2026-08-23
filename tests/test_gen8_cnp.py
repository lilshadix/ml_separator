"""Contract tests for the gen8 Conditional Neural Process adapters.

The interesting ones are the two that would invalidate a result rather than just
break a run: that ``predict`` cannot see a target it was not handed, and that the
anchored residual variant's *initialisation* is exactly offset correction — the
second is what licenses reading every trained number as "the departure from
OFFSET_K1 that training bought".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen8.adapters import AdaptContext, RidgeOffset
from lanthanide_separation.gen8.cnp import (
    CNPAdapter, ZeroContextView, build_cnp_adapters, parameter_count,
)

torch = pytest.importorskip("torch")


def _frame(n_ligands: int = 24, n_rows: int = 12, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ligands):
        level = rng.normal(0.0, 1.2)
        for j in range(n_rows):
            acid = rng.normal(0.0, 0.6)
            rows.append({
                "extractant": f"L{i}", "tanimoto_cluster": f"C{i // 2}",
                "lanthanide_index": rng.integers(1, 16),
                "Atomic Number_metal": 57 + j % 14, "Ionic Radius_metal": 1.0 + 0.01 * j,
                "massact__log10_cond__acid_concentration_M": acid,
                "massact__log10_cond__extractant_concentration_M": rng.normal(-1, 0.4),
                "cond__temperature_C": 25.0, "cond__acid__hno3": 1.0,
                "DENTATE": 4, "MolWt": 300 + 10 * i, "donor__n_total": 4, "TPSA": 60.0,
                "log_D": level + 1.5 * acid + rng.normal(0, 0.3),
            })
    frame = pd.DataFrame(rows)
    frame["prediction"] = frame["log_D"] - np.repeat(
        rng.normal(0.0, 1.0, n_ligands), n_rows) + rng.normal(0, 0.2, len(frame))
    return frame, frame["log_D"].to_numpy(dtype=float)


def _fit(**kwargs) -> CNPAdapter:
    frame, y = _frame()
    adapter = CNPAdapter(name="T", steps=kwargs.pop("steps", 40), batch_ligands=8,
                         val_episodes=3, eval_every=20, **kwargs)
    adapter.fit_fold(frame, y, split_seed=1, fold=0, model_seed=123)
    return adapter


def test_predict_ignores_the_target_column():
    """Corrupting ``log_D`` on the block must not move a single prediction.

    This is the leakage guard the whole gen8 protocol rests on: the driver hands
    the adapter a frame that *does* carry the target, and the adapter is only
    allowed to read ``observed``.
    """
    adapter = _fit(residual=True, anchor=True, relative=True)
    block, truth = _frame(n_ligands=1, n_rows=10, seed=5)
    context = AdaptContext(split_seed=1, fold=0, extractant="L0")
    prediction = block["prediction"].to_numpy(dtype=float)
    selected = np.array([0, 3])
    honest = adapter.predict(block, prediction, selected, truth[selected], context)

    poisoned = block.copy()
    poisoned["log_D"] = 999.0
    adapter._cache.clear()
    again = adapter.predict(poisoned, prediction, selected, truth[selected], context)
    assert np.allclose(honest, again)


def test_anchored_residual_initialisation_is_offset_correction():
    """With no training steps the anchored variant *is* ``OFFSET_K1``, exactly."""
    adapter = _fit(residual=True, anchor=True, relative=True, steps=0)
    block, truth = _frame(n_ligands=1, n_rows=10, seed=7)
    context = AdaptContext(split_seed=1, fold=0, extractant="L0")
    prediction = block["prediction"].to_numpy(dtype=float)
    reference = RidgeOffset(mode="K1")
    for k in (1, 2, 3, 5):
        selected = np.arange(k)
        theirs = reference.predict(block, prediction, selected, truth[selected], context)
        ours = adapter.predict(block, prediction, selected, truth[selected], context)
        assert np.allclose(ours, theirs, atol=1e-5)
    # ...and at k = 0 it is the frozen prediction untouched.
    empty = np.zeros(0, dtype=int)
    assert np.allclose(adapter.predict(block, prediction, empty, np.zeros(0), context), prediction)


def test_same_weights_serve_every_k_and_stay_finite():
    adapter = _fit(residual=True, anchor=True, relative=True)
    block, truth = _frame(n_ligands=1, n_rows=10, seed=11)
    context = AdaptContext(split_seed=1, fold=0, extractant="L0")
    prediction = block["prediction"].to_numpy(dtype=float)
    for k in (0, 1, 2, 3, 5):
        selected = np.arange(k)
        values = adapter.predict(block, prediction, selected, truth[selected], context)
        assert values.shape == (len(block),)
        assert np.isfinite(values).all()


def test_fit_is_deterministic_given_the_model_seed():
    frame, y = _frame()
    block, truth = _frame(n_ligands=1, n_rows=8, seed=3)
    context = AdaptContext(split_seed=1, fold=0, extractant="L0")
    prediction = block["prediction"].to_numpy(dtype=float)
    outputs = []
    for _ in range(2):
        adapter = CNPAdapter(name="T", steps=40, batch_ligands=8, val_episodes=3,
                             eval_every=20, residual=True, anchor=True, relative=True)
        adapter.fit_fold(frame, y, split_seed=1, fold=0, model_seed=99)
        outputs.append(adapter.predict(block, prediction, np.array([0, 2]),
                                       truth[np.array([0, 2])], context))
    assert np.allclose(outputs[0], outputs[1])


def test_permutation_invariance_of_the_context():
    """The CNP aggregation is a mean, so the order of the measurements cannot matter."""
    adapter = _fit(residual=True, anchor=True, relative=True)
    block, truth = _frame(n_ligands=1, n_rows=10, seed=13)
    context = AdaptContext(split_seed=1, fold=0, extractant="L0")
    prediction = block["prediction"].to_numpy(dtype=float)
    forward = np.array([1, 4, 6])
    backward = forward[::-1]
    a = adapter.predict(block, prediction, forward, truth[forward], context)
    b = adapter.predict(block, prediction, backward, truth[backward], context)
    assert np.allclose(a, b, atol=1e-5)


def test_zero_context_view_is_constant_in_k():
    adapter = _fit(residual=True, anchor=True, relative=True)
    view = ZeroContextView(parent=adapter)
    block, truth = _frame(n_ligands=1, n_rows=10, seed=17)
    context = AdaptContext(split_seed=1, fold=0, extractant="L0")
    prediction = block["prediction"].to_numpy(dtype=float)
    one = view.predict(block, prediction, np.array([0]), truth[:1], context)
    five = view.predict(block, prediction, np.arange(5), truth[:5], context)
    assert np.allclose(one, five)
    assert view.name.endswith("@k0")


def test_every_built_adapter_is_under_the_parameter_cap():
    frame, y = _frame()
    for adapter in build_cnp_adapters(steps=1, batch_ligands=4, with_zero_context=False):
        adapter.val_episodes = 2
        adapter.fit_fold(frame, y, split_seed=1, fold=0, model_seed=5)
        assert 0 < parameter_count(adapter) < 100_000, adapter.name


def test_every_built_arm_ignores_the_target_column():
    """The blindness guard, extended from one arm to all of them.

    ``test_predict_ignores_the_target_column`` covers only the anchored residual
    configuration.  The absolute family, the shrinkage head and the attentive
    branch take different paths through ``forward``, and the k = 0 path takes a
    different one again, so each is checked here.  The driver hands every adapter a
    frame that carries ``log_D``; none of them may move when it is destroyed.
    """
    frame, y = _frame()
    block, truth = _frame(n_ligands=1, n_rows=10, seed=23)
    context = AdaptContext(split_seed=1, fold=0, extractant="L0")
    prediction = block["prediction"].to_numpy(dtype=float)
    poisoned = block.copy()
    poisoned["log_D"] = np.nan
    for adapter in build_cnp_adapters(steps=30, batch_ligands=6, with_zero_context=False):
        adapter.val_episodes = 2
        adapter.eval_every = 15
        adapter.fit_fold(frame, y, split_seed=1, fold=0, model_seed=7)
        for k in (0, 1, 2, 3, 5):
            selected = np.arange(k)
            honest = adapter.predict(block, prediction, selected, truth[selected], context)
            adapter._cache.clear()
            again = adapter.predict(poisoned, prediction, selected, truth[selected], context)
            adapter._cache.clear()
            assert np.allclose(honest, again), (adapter.name, k)
        assert adapter.failures == 0 and adapter.unfitted == 0, adapter.name


def test_an_unfitted_fold_is_counted_not_silent():
    """A fold that never fits scores the arm as pure zero-shot; it must be visible.

    ``predict`` returns the frozen prediction when ``_fitted`` is False, which is
    the contract-mandated finite fallback — but it is indistinguishable in the
    score from "this method chose not to move", so the count is the only evidence.
    """
    adapter = CNPAdapter(name="T", steps=1, batch_ligands=4, val_episodes=1)
    block, truth = _frame(n_ligands=1, n_rows=6, seed=29)
    context = AdaptContext(split_seed=1, fold=0, extractant="L0")
    prediction = block["prediction"].to_numpy(dtype=float)
    values = adapter.predict(block, prediction, np.array([0]), truth[:1], context)
    assert np.array_equal(values, prediction)
    assert adapter.unfitted == 1
