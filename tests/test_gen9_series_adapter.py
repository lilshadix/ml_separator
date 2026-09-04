"""The hierarchical adapter: shrinkage behaviour, fold-locality, and leakage.

gen7 lost a k=2 arm to an unrestricted affine fit that produced 1e12-scale
coefficients, and gen8 established that four shrunk coefficients beat one free
one.  gen9 replaces gen8's hand-set ridge with penalties estimated from each fold's
own training ligands, which is better only if it *keeps* the property that made
gen8's version work: at k=1 the slopes must not move.

These tests pin that behaviour numerically rather than trusting that a ridge will
do the right thing, and they check the two ways the estimated prior could go wrong
— by being computed over the whole corpus instead of the fold, and by being
computed from in-sample residuals, which are near zero and would report that no
coefficient ever varies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen8.adapters import AdaptContext, RidgeOffset
from lanthanide_separation.gen9.series_adapter import (
    PENALTY_CEILING, PENALTY_FLOOR, RESPONSE_COLUMNS, HierarchicalSeriesAdapter,
    build_design, build_series_adapters, estimate_prior, family_of, ridge_map,
)


def make_block(n: int = 18, seed: int = 0, ligand: str = "LIG", n_series: int = 3,
               level: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    acid = rng.uniform(-1, 1, size=n)
    extr = rng.uniform(-2, 0, size=n)
    metal = rng.integers(0, 14, size=n).astype(float)
    series = np.array([f"{ligand}_s{i % n_series}" for i in range(n)])
    series_effect = {s: rng.normal(0, 0.6) for s in set(series)}
    truth = (level + np.array([series_effect[s] for s in series])
             + 1.5 * acid + 2.5 * extr + 0.08 * metal + rng.normal(0, 0.3, size=n))
    return pd.DataFrame({
        "row_id": [f"{ligand}_{i}" for i in range(n)], "extractant": ligand,
        "tanimoto_cluster": f"tan_{ligand}", "ecfp_cluster": f"ecfp_{ligand}",
        "series_id": series, "split_seed": 104729, "fold": 0,
        "massact__log10_cond__acid_concentration_M": acid,
        "massact__log10_cond__extractant_concentration_M": extr,
        "lanthanide_index": metal, "cond__temperature_C": 25.0,
        "log_D": truth, "prediction": 0.2 * acid + 0.2 * extr, "model": "TEST"})


def make_oof(n_ligands: int = 10, fold_of=lambda i: i % 5) -> pd.DataFrame:
    frames = []
    for i in range(n_ligands):
        block = make_block(n=18, seed=i, ligand=f"L{i}", level=np.random.default_rng(i).normal(0, 2))
        block["fold"] = fold_of(i)
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# design
# --------------------------------------------------------------------------- #

def test_design_is_intercept_series_deviations_and_standardised_responses():
    block = make_block(n=12, n_series=3)
    design, names = build_design(block)
    assert names[0] == "intercept"
    assert sum(n.startswith("series::") for n in names) == 3
    assert [n for n in names if n.startswith("response::")] == [
        f"response::{c}" for c in RESPONSE_COLUMNS]
    np.testing.assert_allclose(design[:, 0], 1.0)
    for j, name in enumerate(names):
        if name.startswith("series::"):
            assert design[:, j].mean() == pytest.approx(0.0, abs=1e-12), (
                "series columns must be deviations, so the intercept is the ligand level")


def test_a_single_series_block_emits_no_series_column():
    block = make_block(n=9, n_series=1)
    _, names = build_design(block)
    assert not any(n.startswith("series::") for n in names)


def test_the_design_reads_no_target():
    block = make_block(n=12)
    clean, _ = build_design(block)
    poisoned = block.copy()
    poisoned["log_D"] = 1e6
    dirty, _ = build_design(poisoned)
    np.testing.assert_array_equal(clean, dirty)


def test_family_assignment():
    assert family_of("intercept") == "intercept"
    assert family_of("series::abc") == "series"
    assert family_of("response::lanthanide_index") == "response"


# --------------------------------------------------------------------------- #
# 6. k = 1 shrinkage collapses the slope updates
# --------------------------------------------------------------------------- #

def test_one_measurement_moves_the_level_and_essentially_nothing_else():
    """The gen8 property, re-established under estimated penalties."""
    block = make_block(n=20, seed=4)
    oof = make_oof(12)
    adapter = HierarchicalSeriesAdapter(oof=oof)
    adapter.fit_fold(pd.DataFrame({"extractant": [f"L{i}" for i in range(12)]}),
                     np.zeros(12), split_seed=104729, fold=0, model_seed=1)
    prediction = block["prediction"].to_numpy(dtype=float)
    truth = block["log_D"].to_numpy(dtype=float)
    selected = np.array([3])
    context = AdaptContext(split_seed=104729, fold=0, extractant="LIG")
    adjusted = adapter.predict(block, prediction, selected, truth[selected], context)

    shift = adjusted - prediction
    assert np.std(shift) < 0.35 * abs(np.mean(shift)) + 1e-9, (
        "one measurement is moving the response shape, not just the level; "
        f"level {np.mean(shift):.3f} spread {np.std(shift):.3f}")


def test_more_measurements_release_the_response_coefficients():
    block = make_block(n=24, seed=6)
    oof = make_oof(14)
    adapter = HierarchicalSeriesAdapter(oof=oof)
    adapter.fit_fold(pd.DataFrame({"extractant": [f"L{i}" for i in range(14)]}),
                     np.zeros(14), split_seed=104729, fold=0, model_seed=1)
    prediction = block["prediction"].to_numpy(dtype=float)
    truth = block["log_D"].to_numpy(dtype=float)
    context = AdaptContext(split_seed=104729, fold=0, extractant="LIG")
    spreads = []
    for k in (1, 2, 3, 5, 8):
        selected = np.arange(k)
        adjusted = adapter.predict(block, prediction, selected, truth[selected], context)
        spreads.append(float(np.std(adjusted - prediction)))
    assert spreads[-1] > spreads[0], (
        "the hierarchy never releases its slopes; it has become a fixed offset")


def test_the_fit_never_explodes():
    """gen7's failure mode: an unrestricted affine map at k=2 scoring 3.32."""
    block = make_block(n=30, seed=8)
    oof = make_oof(12)
    adapter = HierarchicalSeriesAdapter(oof=oof)
    adapter.fit_fold(pd.DataFrame({"extractant": [f"L{i}" for i in range(12)]}),
                     np.zeros(12), split_seed=104729, fold=0, model_seed=1)
    prediction = block["prediction"].to_numpy(dtype=float)
    truth = block["log_D"].to_numpy(dtype=float)
    context = AdaptContext(split_seed=104729, fold=0, extractant="LIG")
    for k in (1, 2, 3):
        adjusted = adapter.predict(block, prediction, np.arange(k), truth[:k], context)
        assert np.all(np.isfinite(adjusted))
        assert np.abs(adjusted - prediction).max() < 50.0


def test_an_infinite_penalty_reduces_to_plain_offset_correction():
    """The hierarchy must nest ``OFFSET_K1``; a ceiling penalty is that limit."""
    block = make_block(n=16, seed=2)
    adapter = HierarchicalSeriesAdapter(oof=make_oof(4), name="X")
    adapter.priors[(104729, 0)] = {
        "sigma": 1.0, "tau": {"series": 0.0, "response": 0.0},
        "penalty": {"series": 1e12, "response": 1e12}, "n_ligands": 0,
        "n_coefficients": {}}
    prediction = block["prediction"].to_numpy(dtype=float)
    truth = block["log_D"].to_numpy(dtype=float)
    context = AdaptContext(split_seed=104729, fold=0, extractant="LIG")
    selected = np.array([1, 5, 9])
    ours = adapter.predict(block, prediction, selected, truth[selected], context)
    theirs = RidgeOffset(mode="K1").predict(block, prediction, selected, truth[selected], context)
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-6)


def test_zero_measurements_returns_the_prediction_unchanged():
    block = make_block(n=10)
    adapter = HierarchicalSeriesAdapter(oof=make_oof(4))
    prediction = block["prediction"].to_numpy(dtype=float)
    context = AdaptContext(split_seed=104729, fold=0, extractant="LIG")
    np.testing.assert_array_equal(
        adapter.predict(block, prediction, np.array([], dtype=int), np.array([]), context),
        prediction)


# --------------------------------------------------------------------------- #
# 5. corrupting hidden targets cannot change an adapted prediction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("adapter_index", [0, 1, 2])
def test_unselected_targets_cannot_move_a_prediction(adapter_index):
    oof = make_oof(12)
    adapter = build_series_adapters(oof)[adapter_index]
    adapter.fit_fold(pd.DataFrame({"extractant": [f"L{i}" for i in range(12)]}),
                     np.zeros(12), split_seed=104729, fold=0, model_seed=1)
    block = make_block(n=20, seed=42, ligand="HELD")
    prediction = block["prediction"].to_numpy(dtype=float)
    truth = block["log_D"].to_numpy(dtype=float)
    selected = np.array([2, 7, 11])
    context = AdaptContext(split_seed=104729, fold=0, extractant="HELD")
    clean = adapter.predict(block, prediction, selected, truth[selected], context)

    poisoned = block.copy()
    values = truth.copy()
    mask = np.ones(len(block), dtype=bool)
    mask[selected] = False
    values[mask] = np.random.default_rng(0).normal(1e4, 1e3, size=int(mask.sum()))
    poisoned["log_D"] = values
    dirty = adapter.predict(poisoned, prediction, selected, truth[selected], context)
    np.testing.assert_allclose(clean, dirty, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# 10/13. the prior is fold-local
# --------------------------------------------------------------------------- #

def test_the_prior_excludes_the_folds_own_test_ligands():
    oof = make_oof(10, fold_of=lambda i: 0 if i < 3 else 1 + (i % 4))
    adapter = HierarchicalSeriesAdapter(oof=oof)
    train_ligands = [f"L{i}" for i in range(3, 10)]
    adapter.fit_fold(pd.DataFrame({"extractant": train_ligands}), np.zeros(len(train_ligands)),
                     split_seed=104729, fold=0, model_seed=1)
    prior = adapter.priors[(104729, 0)]
    assert prior["n_ligands"] == len(train_ligands), (
        "the prior was estimated over a different ligand set than the fold's training half")


def test_two_folds_get_different_priors():
    """A corpus-wide prior computed once before CV would make these identical."""
    oof = make_oof(20, fold_of=lambda i: i % 4)
    adapter = HierarchicalSeriesAdapter(oof=oof)
    for fold in (0, 1):
        ligands = [f"L{i}" for i in range(20) if i % 4 != fold]
        adapter.fit_fold(pd.DataFrame({"extractant": ligands}), np.zeros(len(ligands)),
                         split_seed=104729, fold=fold, model_seed=1)
    a = adapter.priors[(104729, 0)]["penalty"]
    b = adapter.priors[(104729, 1)]["penalty"]
    assert a != b, "every fold produced the same prior — it is not fold-local"


def test_estimated_penalties_stay_inside_their_declared_bounds():
    oof = make_oof(16)
    prior = estimate_prior(oof)
    for family, value in prior["penalty"].items():
        assert PENALTY_FLOOR <= value <= PENALTY_CEILING, f"{family} penalty {value} out of range"


def test_a_degenerate_prior_falls_back_to_the_ceiling_not_to_zero():
    """``tau -> 0`` must switch a family *off*, never make it free."""
    identical = pd.concat([make_block(n=12, seed=0, ligand=f"L{i}") for i in range(4)],
                          ignore_index=True)
    identical["log_D"] = identical["prediction"]     # zero residual everywhere
    prior = estimate_prior(identical)
    assert prior["penalty"]["response"] == pytest.approx(PENALTY_CEILING)


def test_ridge_map_is_a_penalised_normal_equation():
    rng = np.random.default_rng(0)
    design = rng.normal(size=(20, 3))
    residual = rng.normal(size=20)
    penalties = np.array([0.0, 2.0, 5.0])
    beta = ridge_map(design, residual, penalties)
    expected = np.linalg.solve(design.T @ design + np.diag(penalties), design.T @ residual)
    np.testing.assert_allclose(beta, expected)
