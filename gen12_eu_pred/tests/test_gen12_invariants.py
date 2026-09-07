"""The twelve pre-registered invariants, as executable checks.

These do not test that a model is good.  They test the properties that make every
Gen12 number mean what the decision report says it means, and each has a way of
failing silently that would leave the tables looking perfectly reasonable.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GEN12 = Path(__file__).resolve().parents[1]
if str(GEN12) not in sys.path:
    sys.path.insert(0, str(GEN12))

from gen12eu import splits  # noqa: E402
from gen12eu.chemistry import ECFP_COLUMNS, band_of, fingerprint_matrix, tanimoto_matrix  # noqa: E402
from gen12eu.cohort import (  # noqa: E402
    CONSTANT_METAL_COLUMNS, FORBIDDEN_FEATURE_TOKENS, TARGET, build_cohort,
)
from gen12eu.fewshot import support_draw  # noqa: E402
from gen12eu.metrics import per_extractant  # noqa: E402
from gen12eu.preprocess import FoldPreprocessor  # noqa: E402


@pytest.fixture(scope="module")
def cohort():
    return build_cohort()


@pytest.fixture(scope="module")
def folds_b(cohort):
    return splits.all_folds(cohort.frame, design="B")


# --- 1, 2: hold-out integrity ---------------------------------------------- #

def test_no_extractant_or_chemotype_crosses_a_chemotype_fold(cohort, folds_b):
    report = splits.assert_fold_integrity(cohort.frame, folds_b)
    assert report["ok"]
    assert report["overlap_counts"] == {"extractant": 0, "ecfp_cluster": 0, "chemotype": 0}


def test_exact_extractant_design_holds_extractants_but_is_recorded_as_leaky(cohort):
    """Design A must be clean on its own unit and *measured* on the stricter ones.

    The point of the test is not that A is safe — it is not — but that its leakage
    is a reported number rather than an unexamined assumption.
    """
    folds = splits.all_folds(cohort.frame, design="A")
    report = splits.assert_fold_integrity(cohort.frame, folds)
    assert report["overlap_counts"]["extractant"] == 0
    assert report["overlap_counts"]["ecfp_cluster"] > 0, (
        "design A is expected to leak bit-identical fingerprints; if it no longer "
        "does, the audit's CRITICAL finding needs revisiting rather than silently dropping")


def test_chemotype_holdout_caps_similarity_below_the_clustering_threshold(cohort, folds_b):
    similarity = splits.similarity_table(cohort.frame, folds_b)
    assert similarity["max_train_tanimoto"].max() < 0.70


def test_every_row_is_tested_exactly_once_per_seed(cohort, folds_b):
    for seed in splits.SPLIT_SEEDS:
        tested = np.concatenate([f.test_index for f in folds_b if f.seed == seed])
        assert len(tested) == len(np.unique(tested)) == len(cohort.frame)


# --- 4, 9: preprocessing and selection never see the test ------------------- #

def test_inner_validation_is_disjoint_from_the_outer_test(folds_b):
    for fold in folds_b:
        assert not set(fold.inner_validation_index) & set(fold.test_index)
        assert not set(fold.inner_train_index) & set(fold.inner_validation_index)
        assert set(fold.inner_train_index) | set(fold.inner_validation_index) <= set(fold.train_index)


def test_preprocessor_statistics_depend_only_on_the_training_rows(cohort, folds_b):
    """Corrupting every test row must not move a single fitted statistic."""
    frame = cohort.frame
    fold = folds_b[0]
    columns = cohort.blocks["COND"] + cohort.blocks["MASSACT"]
    train = frame.iloc[fold.train_index]
    fitted = FoldPreprocessor(columns, standardise=True).fit(train)
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
    corrupted = frame.copy()
    corrupted[numeric] = corrupted[numeric].astype(float)
    corrupted.iloc[fold.test_index, [corrupted.columns.get_loc(c) for c in numeric]] = 1e6
    again = FoldPreprocessor(columns, standardise=True).fit(corrupted.iloc[fold.train_index])
    assert np.allclose(fitted.medians_, again.medians_, equal_nan=True)
    assert np.allclose(fitted.centre_, again.centre_)
    assert np.allclose(fitted.scale_, again.scale_)


# --- 5, 10: the target cannot reach a feature or be altered ----------------- #

def test_no_feature_column_matches_a_forbidden_token(cohort):
    for name, columns in cohort.blocks.items():
        for column in columns:
            for token in FORBIDDEN_FEATURE_TOKENS:
                assert token not in column, f"{column!r} in block {name!r}"


def test_metal_identity_never_reaches_a_feature_block(cohort):
    everything = {c for columns in cohort.blocks.values() for c in columns}
    assert not everything & set(CONSTANT_METAL_COLUMNS)


def test_cohort_structure_is_independent_of_the_target(monkeypatch):
    """Rebuild with a corrupted target; every identity and feature column must match.

    This is the decisive check that no group, no split, no band and no feature is
    a function of ``log_D``.
    """
    import gen12eu.cohort as module
    honest = module.build_cohort()
    original = pd.read_parquet
    rng = np.random.default_rng(0)

    def poisoned(path, *args, **kwargs):
        frame = original(path, *args, **kwargs)
        if TARGET in frame.columns:
            frame = frame.copy()
            frame[TARGET] = rng.normal(size=len(frame))
        return frame

    monkeypatch.setattr(module.pd, "read_parquet", poisoned)
    corrupted = module.build_cohort(verify_hash=False)
    assert list(corrupted.frame["row_id"]) == list(honest.frame["row_id"])
    for column in ("extractant", "chemotype", "ecfp_cluster", "condition_id", "series_id"):
        assert (corrupted.frame[column].to_numpy() == honest.frame[column].to_numpy()).all()
    for name, columns in honest.blocks.items():
        assert corrupted.blocks[name] == columns
        left = corrupted.frame[list(columns)].to_numpy(dtype=float)
        right = honest.frame[list(columns)].to_numpy(dtype=float)
        assert np.allclose(left, right, equal_nan=True), name


# --- 11: bands are similarity, never target -------------------------------- #

def test_bands_depend_only_on_similarity(cohort, folds_b):
    similarity = splits.similarity_table(cohort.frame, folds_b)
    bands = band_of(similarity["max_train_tanimoto"])
    assert set(bands) <= {"far", "mid", "near"}
    for band, low, high in ((("far", -np.inf, 0.4)), ("mid", 0.4, 0.6), ("near", 0.6, np.inf)):
        values = similarity.loc[bands.to_numpy() == band, "max_train_tanimoto"]
        assert ((values > low) & (values <= high)).all()


def test_similarity_is_measured_against_training_only(cohort, folds_b):
    """A test extractant's similarity must not change if other test rows change."""
    frame = cohort.frame
    fold = folds_b[0]
    names, bits = fingerprint_matrix(frame)
    extractants = frame["extractant"].astype(str).to_numpy()
    test_names = sorted(set(extractants[fold.test_index]))
    train_names = sorted(set(extractants[fold.train_index]))
    index = {n: i for i, n in enumerate(names)}
    full = tanimoto_matrix(bits[[index[n] for n in test_names]],
                           bits[[index[n] for n in train_names]]).max(axis=1)
    table = splits.similarity_table(frame, [fold])
    recorded = table.set_index("extractant").loc[test_names, "max_train_tanimoto"].to_numpy()
    assert np.allclose(full, recorded)


# --- 6, 7, 8: few-shot draws --------------------------------------------- #

def test_support_and_query_are_disjoint_and_cover_the_extractant():
    for k in (1, 2, 3, 5):
        support = support_draw("CCO", 12, k, repeat=3, seed=104729)
        query = np.setdiff1d(np.arange(12), support)
        assert len(support) == k
        assert not set(support) & set(query)
        assert len(support) + len(query) == 12


def test_support_draw_is_identical_across_processes():
    """gen8's ``hash()`` trap: the draw must not depend on PYTHONHASHSEED."""
    code = ("import sys; sys.path.insert(0, %r);"
            "from gen12eu.fewshot import support_draw;"
            "print(list(support_draw('TODGA-like', 17, 3, 2, 130363)))" % str(GEN12))
    outputs = set()
    for hashseed in ("0", "1", "12345"):
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"})
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"draw depends on the hash seed: {outputs}"


def test_support_draw_does_not_depend_on_the_model():
    """Two arms must receive byte-identical support sets."""
    a = support_draw("X", 9, 2, 0, 155921)
    b = support_draw("X", 9, 2, 0, 155921)
    assert (a == b).all()


# --- 12: matched evaluation units ------------------------------------------ #

def test_per_extractant_centring_is_per_seed(cohort):
    """A seed-pooled centring would score between-seed level wobble as shape.

    Constructed so that the two seeds have opposite level errors and zero shape
    error: the correct decomposition returns shape 0, a seed-pooled one does not.
    """
    frame = pd.DataFrame({
        "split_seed": [1, 1, 2, 2], "extractant": ["A"] * 4,
        "chemotype": ["c"] * 4, TARGET: [0.0, 1.0, 0.0, 1.0],
        "prediction": [1.0, 2.0, -1.0, 0.0],
    })
    units = per_extractant(frame)
    assert np.allclose(units["shape_mae"], 0.0)
    assert sorted(np.round(units["bias"], 6)) == [-1.0, 1.0]


def test_level_shape_identity_holds_exactly(cohort):
    rng = np.random.default_rng(7)
    n = 40
    frame = pd.DataFrame({
        "split_seed": rng.integers(0, 3, n), "extractant": rng.choice(list("ABCD"), n),
        "chemotype": "c", TARGET: rng.normal(size=n), "prediction": rng.normal(size=n)})
    units = per_extractant(frame)
    assert np.allclose(units["sse"], units["sse_centred"] + units["sse_offset"])


# --- few-shot adapter algebra ---------------------------------------------- #

def test_the_two_preregistered_kshot_nulls_are_the_same_estimator():
    """``m + mean(y_S - m) = mean(y_S)``: the corpus level cancels exactly.

    The pre-registration asked for two nulls and they collapse to one.  The test
    records the identity so the decision report cannot quote them as independent
    corroboration.
    """
    from gen12eu.fewshot import GlobalMeanShiftedAdapter, NoModelAdapter
    rng = np.random.default_rng(3)
    truth = rng.normal(size=11)
    prediction = rng.normal(size=11)
    support, query = np.array([1, 4, 7]), np.array([0, 2, 3, 5, 6, 8, 9, 10])
    a = NoModelAdapter().apply(prediction, truth, support, query)
    b = GlobalMeanShiftedAdapter(global_mean=float(truth.mean())).apply(
        prediction, truth, support, query)
    assert np.allclose(a, b)


def test_offset_adapter_ignores_unselected_targets():
    """Corrupting every query target must not move a single adapted prediction."""
    from gen12eu.fewshot import OffsetAdapter
    rng = np.random.default_rng(11)
    truth = rng.normal(size=14)
    prediction = rng.normal(size=14)
    support = np.array([2, 5])
    query = np.setdiff1d(np.arange(14), support)
    adapter = OffsetAdapter(ratio=1.7)
    honest = adapter.apply(prediction, truth, support, query)
    poisoned = truth.copy()
    poisoned[query] = 999.0
    assert np.allclose(honest, adapter.apply(prediction, poisoned, support, query))


# --- a column that is missing inside one fold ------------------------------- #

def test_all_missing_training_column_is_dropped_without_warning():
    """gen5 lost a whole run to this: a column observed in the cohort but empty in
    one training fold.  The preprocessor must drop it silently and deterministically,
    and must still impute the columns that do have values."""
    import warnings
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(20, 5))
    raw[:, 2] = np.nan
    raw[rng.random((20, 5)) < 0.2] = np.nan
    frame = pd.DataFrame(raw, columns=[f"c{i}" for i in range(5)])
    observed = np.isfinite(raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = np.nanmedian(np.where(observed, raw, np.nan), axis=0)
    legacy = np.where(np.isfinite(legacy), legacy, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pre = FoldPreprocessor(tuple(frame.columns)).fit(frame)
        block = pre.transform(frame)
    assert np.allclose(pre.medians_, legacy)
    assert "c2" not in pre.names_
    assert np.isfinite(block).all()


# --- 3: cross-lanthanide exclusion ------------------------------------------ #

def test_strict_multiln_pool_excludes_held_out_chemistry_under_every_metal():
    """The arm's whole validity rests on this: an Eu test extractant must not
    reappear in training under Nd, La or Dy.  The audit measured 4,148 non-Eu rows
    sitting on Eu cohort extractants, so a filter that only removed Eu rows would
    leave almost all of them."""
    from gen12eu.multiln import MultiLanthanideContender, build_auxiliary
    cohort = build_cohort()
    frame = cohort.frame
    auxiliary = build_auxiliary(frame.columns)
    folds = splits.build_folds(frame, design="B", seed=104729)
    fold = folds[0]
    test = frame.iloc[fold.test_index]
    train = frame.iloc[fold.train_index]

    strict = MultiLanthanideContender(auxiliary=auxiliary, policy="STRICT")
    pool, audit = strict._pool(train, test, seed=1)
    assert audit["aux_test_extractant_overlap"] == 0
    assert not set(pool["extractant"].astype(str)) & set(test["extractant"].astype(str))
    assert not set(pool["chemotype"].astype(str)) & set(test["chemotype"].astype(str))

    leaky = MultiLanthanideContender(auxiliary=auxiliary, policy="LEAKY_NO_FILTER")
    _, leaky_audit = leaky._pool(train, test, seed=1)
    assert leaky_audit["aux_test_extractant_overlap"] > 0, (
        "the leaky control must actually leak; it exists to measure the size of the "
        "artefact a naive multi-metal analysis reports")


def test_multiln_controls_change_only_what_they_claim_to():
    """PERMUTED_METAL keeps every target; SHUFFLED_TARGET keeps every feature."""
    from gen12eu.multiln import METAL_COLUMNS, MultiLanthanideContender, build_auxiliary
    cohort = build_cohort()
    frame = cohort.frame
    auxiliary = build_auxiliary(frame.columns)
    folds = splits.build_folds(frame, design="B", seed=104729)
    fold = folds[0]
    train, test = frame.iloc[fold.train_index], frame.iloc[fold.test_index]

    base, _ = MultiLanthanideContender(auxiliary=auxiliary, policy="STRICT")._pool(train, test, 1)
    permuted, _ = MultiLanthanideContender(
        auxiliary=auxiliary, policy="PERMUTED_METAL")._pool(train, test, 1)
    shuffled, _ = MultiLanthanideContender(
        auxiliary=auxiliary, policy="SHUFFLED_TARGET")._pool(train, test, 1)

    assert len(base) == len(permuted) == len(shuffled)
    assert sorted(permuted[TARGET].round(9)) == sorted(base[TARGET].round(9))
    assert sorted(shuffled[TARGET].round(9)) == sorted(base[TARGET].round(9))
    for column in METAL_COLUMNS:
        assert sorted(permuted[column].dropna()) == sorted(base[column].dropna())
    # the shuffled-target control must not have touched a feature
    assert np.allclose(shuffled[list(METAL_COLUMNS)].to_numpy(dtype=float),
                       base[list(METAL_COLUMNS)].to_numpy(dtype=float), equal_nan=True)


# --- the heavy-tailed-descriptor guard -------------------------------------- #

def test_training_range_clamp_bounds_held_out_inputs(cohort, folds_b):
    """A descriptor spanning 1e8 to 3.8e29 must not reach a network as 7e16.

    Without the clamp, a chemotype hold-out that removes the extreme molecule
    leaves a training sd of 5.3e12 and standardises the held-out value to 7.3e16;
    one D-MPNN arm diverged in 8 of 25 folds because of it.  The clamp must bound
    every held-out standardised value while leaving in-range values untouched.
    """
    frame = cohort.frame
    columns = tuple(cohort.blocks["LIG2D"]) + tuple(cohort.blocks["PHYSCHEM"])
    fold = folds_b[0]
    train = frame.iloc[fold.inner_train_index]
    validation = frame.iloc[fold.inner_validation_index]
    plain = FoldPreprocessor(columns, standardise=True).fit(train).transform(validation)
    guarded = FoldPreprocessor(columns, standardise=True,
                               clip_to_train_range=True).fit(train).transform(validation)
    assert np.isfinite(guarded).all()
    assert np.abs(guarded).max() <= np.abs(plain).max()
    assert np.abs(guarded).max() < 1e3

    # in-range rows are untouched: the training rows themselves cannot be clamped
    a = FoldPreprocessor(columns, standardise=True).fit(train).transform(train)
    b = FoldPreprocessor(columns, standardise=True,
                         clip_to_train_range=True).fit(train).transform(train)
    assert np.allclose(a, b)


def test_ipc_is_the_column_that_needs_the_guard():
    """Records the mechanism, so a future reader does not rediscover it."""
    cohort = build_cohort()
    ipc = cohort.frame["lig2d__rd__Ipc"].astype(float)
    assert ipc.max() / max(ipc.min(), 1.0) > 1e15, (
        "lig2d__rd__Ipc no longer spans many orders of magnitude; the guard's "
        "justification in PRE_REGISTRATION.md Addendum 1 needs revisiting")
