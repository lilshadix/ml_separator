"""Tests for the generation-4 candidate estimators."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen4_candidates import (
    ForestParameters,
    HierarchicalPairRegressor,
    ScaleTrendPairRegressor,
    attach_ligand_descriptors,
    transitive_projection,
)
from lanthanide_separation.pairs import PAIR_TARGET_COLUMN

METALS = {"La": (57, 1.16), "Nd": (60, 1.109), "Eu": (63, 1.066), "Dy": (66, 1.027), "Lu": (71, 0.977)}


def _synthetic_pairs(seed: int = 0, n_extractants: int = 6, n_conditions: int = 3) -> pd.DataFrame:
    """Small pair frame with y = s(ligand, condition) * (Z_B - Z_A) / 10 + noise."""

    rng = np.random.default_rng(seed)
    rows = []
    ligand_feature = rng.normal(size=n_extractants)
    condition_feature = rng.normal(size=n_conditions)
    for e in range(n_extractants):
        for c in range(n_conditions):
            scale = 0.6 + 0.4 * ligand_feature[e] + 0.15 * condition_feature[c]
            for (a, (za, ra)), (b, (zb, rb)) in itertools.combinations(METALS.items(), 2):
                y = scale * (zb - za) / 10.0 + rng.normal(scale=0.03)
                rows.append(
                    {
                        "pair_id": f"e{e}-c{c}-{a}{b}",
                        "extractant": f"SMILES{e}",
                        "condition_id": f"cond{c}",
                        "extractant_family": f"fam{e % 2}",
                        "metal_A": a,
                        "metal_B": b,
                        "pair_label": f"{a}-{b}",
                        "base__lig": ligand_feature[e],
                        "base__lig_noise": rng.normal(),
                        "base__cond": condition_feature[c],
                        "pair__Z_A": float(za),
                        "pair__Z_B": float(zb),
                        "pair__Z_mean": (za + zb) / 2.0,
                        "pair__delta_Z": float(zb - za),
                        "pair__ionic_radius_A": ra,
                        "pair__ionic_radius_B": rb,
                        "pair__ionic_radius_mean": (ra + rb) / 2.0,
                        "pair__delta_ionic_radius": rb - ra,
                        PAIR_TARGET_COLUMN: y,
                    }
                )
    return pd.DataFrame(rows)


FEATURES = (
    "base__lig",
    "base__lig_noise",
    "base__cond",
    "pair__Z_A",
    "pair__Z_B",
    "pair__Z_mean",
    "pair__delta_Z",
    "pair__ionic_radius_A",
    "pair__ionic_radius_B",
    "pair__ionic_radius_mean",
    "pair__delta_ionic_radius",
)


def _swap(frame: pd.DataFrame) -> pd.DataFrame:
    swapped = frame.copy()
    swapped[["pair__Z_A", "pair__Z_B"]] = frame[["pair__Z_B", "pair__Z_A"]].to_numpy()
    swapped[["pair__ionic_radius_A", "pair__ionic_radius_B"]] = frame[
        ["pair__ionic_radius_B", "pair__ionic_radius_A"]
    ].to_numpy()
    swapped["pair__delta_Z"] = -frame["pair__delta_Z"]
    swapped["pair__delta_ionic_radius"] = -frame["pair__delta_ionic_radius"]
    swapped[["metal_A", "metal_B"]] = frame[["metal_B", "metal_A"]].to_numpy()
    return swapped


def _split(frame: pd.DataFrame, held_out: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = frame["extractant"].eq(held_out)
    return frame[~test].reset_index(drop=True), frame[test].reset_index(drop=True)


SMALL = ForestParameters(n_estimators=40, max_features=0.7, min_samples_leaf=2)


# ---------------------------------------------------------------------------
# transitive projection
# ---------------------------------------------------------------------------


def test_transitive_projection_is_identity_on_transitive_input() -> None:
    frame = _synthetic_pairs()
    truth = frame[PAIR_TARGET_COLUMN].to_numpy()
    scores = {m: z / 10.0 for m, (z, _) in METALS.items()}
    exact = np.array([scores[b] - scores[a] for a, b in zip(frame.metal_A, frame.metal_B)])
    projected = transitive_projection(frame, exact)
    assert np.allclose(projected, exact, atol=1e-10)
    # labels are never read: perturbing them changes nothing
    frame2 = frame.assign(**{PAIR_TARGET_COLUMN: truth * 0.0 + 99.0})
    assert np.allclose(transitive_projection(frame2, exact), projected)


def test_transitive_projection_makes_predictions_consistent() -> None:
    frame = _synthetic_pairs(seed=3)
    rng = np.random.default_rng(1)
    noisy = frame[PAIR_TARGET_COLUMN].to_numpy() + rng.normal(scale=0.2, size=len(frame))
    projected = transitive_projection(frame, noisy)
    assert projected.shape == noisy.shape
    # within every cell y(A,B) + y(B,C) == y(A,C)
    table = frame.assign(p=projected)
    for _, cell in table.groupby(["extractant", "condition_id"]):
        lookup = {(a, b): v for a, b, v in zip(cell.metal_A, cell.metal_B, cell.p)}
        for a, b, c in itertools.combinations(list(METALS), 3):
            assert lookup[(a, b)] + lookup[(b, c)] == pytest.approx(lookup[(a, c)], abs=1e-9)
    # projection is a least-squares smoother: it never moves further from the
    # truth in aggregate on additive-noise input
    truth = frame[PAIR_TARGET_COLUMN].to_numpy()
    assert np.mean(np.abs(projected - truth)) < np.mean(np.abs(noisy - truth))


# ---------------------------------------------------------------------------
# hierarchical model
# ---------------------------------------------------------------------------


def test_hierarchical_regressor_fits_predicts_and_is_antisymmetric() -> None:
    frame = _synthetic_pairs(seed=5)
    train, test = _split(frame, "SMILES0")
    model = HierarchicalPairRegressor(FEATURES, level=SMALL, deviation=SMALL, random_state=1, n_jobs=1)
    model.fit(train, train[PAIR_TARGET_COLUMN].to_numpy(), train["extractant"].to_numpy())
    pred = model.predict(test)
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()
    swapped = model.predict(_swap(test))
    assert np.allclose(swapped, -pred, atol=1e-10)
    truth = test[PAIR_TARGET_COLUMN].to_numpy()
    baseline = np.mean(np.abs(truth - truth.mean()))
    assert np.mean(np.abs(truth - pred)) < baseline


def test_hierarchical_regressor_ignores_test_labels() -> None:
    frame = _synthetic_pairs(seed=6)
    train, test = _split(frame, "SMILES1")
    model = HierarchicalPairRegressor(FEATURES, level=SMALL, deviation=SMALL, random_state=2, n_jobs=1)
    model.fit(train, train[PAIR_TARGET_COLUMN].to_numpy(), train["extractant"].to_numpy())
    pred = model.predict(test)
    corrupted = test.assign(**{PAIR_TARGET_COLUMN: 1e6})
    assert np.allclose(model.predict(corrupted), pred)


# ---------------------------------------------------------------------------
# scale x trend model
# ---------------------------------------------------------------------------


def test_scale_trend_regressor_recovers_scale_and_is_antisymmetric() -> None:
    frame = _synthetic_pairs(seed=7, n_extractants=8)
    train, test = _split(frame, "SMILES2")
    model = ScaleTrendPairRegressor(FEATURES, scale=SMALL, residual=SMALL, random_state=3, n_jobs=1)
    model.fit(train, train[PAIR_TARGET_COLUMN].to_numpy(), train["extractant"].to_numpy())
    assert model.fitted_scales_ is not None
    assert np.isfinite(model.fitted_scales_.to_numpy()).all()
    pred = model.predict(test)
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()
    assert np.allclose(model.predict(_swap(test)), -pred, atol=1e-10)
    # the fitted per-cell scales track the generating scale up to the pair-trend
    # normalisation: cells of a strongly selective ligand get larger slopes
    scales = model.fitted_scales_
    ligand_of_cell = pd.Series(scales.index.str.split("\x1f").str[0], index=scales.index)
    per_ligand = scales.groupby(ligand_of_cell.to_numpy()).mean()
    generating = train.groupby("extractant")["base__lig"].first().loc[per_ligand.index]
    assert np.corrcoef(per_ligand.to_numpy(), generating.to_numpy())[0, 1] > 0.9
    truth = test[PAIR_TARGET_COLUMN].to_numpy()
    assert np.mean(np.abs(truth - pred)) < np.mean(np.abs(truth - truth.mean()))


def test_scale_trend_handles_unseen_pair_label_via_span_fallback() -> None:
    frame = _synthetic_pairs(seed=8)
    train, test = _split(frame, "SMILES3")
    train = train[~train["pair_label"].eq("La-Lu")].reset_index(drop=True)
    model = ScaleTrendPairRegressor(FEATURES, scale=SMALL, residual=SMALL, random_state=4, n_jobs=1)
    model.fit(train, train[PAIR_TARGET_COLUMN].to_numpy(), train["extractant"].to_numpy())
    pred = model.predict(test)
    assert np.isfinite(pred).all()
    la_lu = test["pair_label"].eq("La-Lu").to_numpy()
    # unseen span still receives a large negative-to-positive oriented trend
    assert np.all(pred[la_lu] > 0.0)


# ---------------------------------------------------------------------------
# descriptor join
# ---------------------------------------------------------------------------


def test_attach_ligand_descriptors_joins_on_smiles() -> None:
    frame = _synthetic_pairs(seed=9, n_extractants=3)
    table = pd.DataFrame(
        {
            "canonical_smiles": ["SMILES0", "SMILES1"],
            "lig2d__hc__n_amide_N": [2.0, 1.0],
            "lig2d__rd__MolWt": [500.0, 300.0],
            "unrelated": [1, 2],
        }
    )
    merged, columns = attach_ligand_descriptors(frame, table)
    assert columns == ("lig2d__hc__n_amide_N", "lig2d__rd__MolWt")
    assert len(merged) == len(frame)
    assert "unrelated" not in merged.columns
    assert merged.loc[merged.extractant.eq("SMILES1"), "lig2d__rd__MolWt"].eq(300.0).all()
    assert merged.loc[merged.extractant.eq("SMILES2"), "lig2d__rd__MolWt"].isna().all()
    with pytest.raises(ValueError):
        attach_ligand_descriptors(frame, table[["canonical_smiles", "unrelated"]])


# ---------------------------------------------------------------------------
# prior-augmented model
# ---------------------------------------------------------------------------


def test_prior_augmented_regressor_predicts_and_is_antisymmetric() -> None:
    from lanthanide_separation.gen4_candidates import PriorAugmentedPairRegressor

    frame = _synthetic_pairs(seed=11, n_extractants=8)
    train, test = _split(frame, "SMILES4")
    model = PriorAugmentedPairRegressor(FEATURES, forest=SMALL, scale=SMALL, random_state=5, n_jobs=1)
    model.fit(train, train[PAIR_TARGET_COLUMN].to_numpy(), train["extractant"].to_numpy())
    pred = model.predict(test)
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()
    assert np.allclose(model.predict(_swap(test)), -pred, atol=1e-10)
    truth = test[PAIR_TARGET_COLUMN].to_numpy()
    assert np.mean(np.abs(truth - pred)) < np.mean(np.abs(truth - truth.mean()))
    # the prior columns are not left behind on the caller's frame
    assert "lig2d__prior_scale" not in test.columns


def test_scale_trend_and_prior_are_reproducible_across_instances() -> None:
    """Two identical fits must give bit-identical predictions (the residual
    target must not inherit thread-order noise from the scale forests)."""
    from lanthanide_separation.gen4_candidates import PriorAugmentedPairRegressor

    frame = _synthetic_pairs(seed=12, n_extractants=8)
    train, test = _split(frame, "SMILES5")
    outs = []
    for _ in range(2):
        model = ScaleTrendPairRegressor(FEATURES, scale=SMALL, residual=SMALL, random_state=9, n_jobs=-1, scale_shrinkage=0.5)
        model.fit(train, train[PAIR_TARGET_COLUMN].to_numpy(), train["extractant"].to_numpy())
        outs.append(model.predict(test))
    assert np.allclose(outs[0], outs[1], rtol=0.0, atol=1e-12)
    outs = []
    for _ in range(2):
        model = PriorAugmentedPairRegressor(FEATURES, forest=SMALL, scale=SMALL, random_state=9, n_jobs=-1)
        model.fit(train, train[PAIR_TARGET_COLUMN].to_numpy(), train["extractant"].to_numpy())
        outs.append(model.predict(test))
    assert np.allclose(outs[0], outs[1], rtol=0.0, atol=1e-12)
