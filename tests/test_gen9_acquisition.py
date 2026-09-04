"""Acquisition: the oracle identity, the feature boundary, and the corruption tests.

gen9-B learns to imitate a rule that is defined in terms of a measurement nobody
has taken yet.  That makes it the part of gen9 most exposed to leakage, and the
leak would not be visible in any summary table — a policy that peeked would simply
look unusually good.

So three things are tested rather than argued.

**The identity is re-derived, not inherited.**  gen8 verified
``argmin |r_i - median(r)| == argmin realised MAE`` under its *exhaustive* protocol,
where the candidate set and the scored set are the same rows.  gen9 selects from a
pool and scores on a disjoint evaluation set, and that proof does not transfer for
free: minimising a convex function over a discrete set need not pick the point
nearest its unconstrained minimiser when the two sets differ.  The tests below
establish where the identity holds exactly and where it only holds approximately,
so the report can say which.

**The feature boundary is machine-checked.**  ``FEATURE_PROVENANCE`` declares, per
column, whether it reads a target.  A test walks it, so adding a feature that reads
``log_D`` and forgetting to declare it fails the suite rather than the review.

**Corruption cannot move a choice.**  Replace every candidate's truth with noise;
the selected candidate must be identical.  This is the gen8 test, and it is the one
that would actually catch a leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen8.kshot import POLICIES, PolicyContext, stable_hash
from lanthanide_separation.gen8.protocols import _standardised_axes, make_p2_split
from lanthanide_separation.gen9.acquisition import (
    FEATURE_GROUPS, FEATURE_PROVENANCE, LearnedPolicy, PairwiseAcquisition,
    ScalarAcquisition, build_dataset, candidate_features, oracle_labels,
    verify_oracle_identity,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def synthetic_block(n: int = 20, seed: int = 0, ligand: str = "LIG", **overrides) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    acid = rng.uniform(-1, 1, size=n)
    extr = rng.uniform(-2, 0, size=n)
    metal = rng.integers(0, 14, size=n).astype(float)
    truth = 1.5 * acid + 2.5 * extr + 0.1 * metal + rng.normal(0, 0.4, size=n)
    frame = pd.DataFrame({
        "row_id": [f"{ligand}_{i}" for i in range(n)],
        "extractant": ligand, "tanimoto_cluster": f"tan_{ligand}",
        "ecfp_cluster": f"ecfp_{ligand}", "series_id": [f"{ligand}_s{i % 3}" for i in range(n)],
        "split_seed": 104729, "fold": 0, "nn_train_tanimoto": 0.4,
        "massact__log10_cond__acid_concentration_M": acid,
        "massact__log10_cond__extractant_concentration_M": extr,
        "lanthanide_index": metal, "cond__temperature_C": 25.0,
        "log_D": truth,
        "prediction": 0.2 * acid + 0.15 * extr + 0.02 * metal,
        "model": "TEST",
    })
    for key, value in overrides.items():
        frame[key] = value
    return frame


def make_context(block, pool, evaluation, truth=None):
    prediction = block["prediction"].to_numpy(dtype=float)
    return PolicyContext(
        block=block, prediction=prediction, pool=np.asarray(pool),
        evaluation=np.asarray(evaluation), axes=_standardised_axes(block),
        uncertainty=np.full(len(block), np.nan), disagreement=np.full(len(block), np.nan),
        rng=np.random.default_rng(0), truth=truth)


# --------------------------------------------------------------------------- #
# 14. the oracle identity, re-derived under the gen9 protocol
# --------------------------------------------------------------------------- #

def test_the_identity_is_exact_when_the_pool_is_the_scored_set():
    """gen8's setting: candidates and scored rows coincide.  Must be exact."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(4, 25))
        residual = rng.normal(0, 1.5, size=n)
        truth = residual.copy()
        prediction = np.zeros(n)
        index = np.arange(n)
        labels = oracle_labels(truth, prediction, index, index)
        realised = np.array([np.abs(residual[index != i] - residual[i]).mean()
                             for i in index])
        chosen = int(labels["oracle_deviation"].to_numpy().argmin())
        assert realised[chosen] == pytest.approx(realised.min(), abs=1e-12)


def test_the_identity_is_approximate_when_pool_and_evaluation_differ():
    """Under gen9's disjoint split the identity is a good surrogate, not a theorem.

    This is a *difference from gen8*, and it is the reason gen9 carries a second
    supervised label.  gen8 verified ``argmin |r_i - median(r)|`` against the
    realised oracle to 6.7e-16 — but in its exhaustive protocol the candidates and
    the scored rows are the same points, which makes the objective a convex function
    evaluated at its own sample and puts the minimum on a data point.  Split the two
    sets and that no longer follows: the candidate nearest the median residual and
    the candidate minimising ``mean_e |r_e - r_i|`` can differ when the evaluation
    residuals are asymmetric about their median.

    Measured here rather than assumed, so the report can quote a rate.  The
    exactness rate is about a coin flip; the *regret* of following the surrogate is
    tiny in the median and the ranking correlation is high, which is what justifies
    training a ranker on it — and why the realised-MAE label is also carried and
    compared rather than assumed redundant.
    """
    rng = np.random.default_rng(11)
    exact, gaps, rhos = 0, [], []
    trials = 300
    for trial in range(trials):
        block = synthetic_block(n=int(rng.integers(8, 30)), seed=trial)
        n = len(block)
        split = make_p2_split(n, np.random.default_rng((20260821, trial, 5)))
        labels = oracle_labels(block["log_D"].to_numpy(dtype=float),
                               block["prediction"].to_numpy(dtype=float),
                               split.pool, split.evaluation)
        check = verify_oracle_identity(labels)
        exact += int(check["exact"])
        gaps.append(check["gap"])
        if np.isfinite(check["spearman"]):
            rhos.append(check["spearman"])
    rate = exact / trials
    assert 0.35 < rate < 0.75, (
        f"exactness rate {rate:.2%} is outside the measured band; either the protocol "
        "or the label definition has moved")
    assert float(np.median(gaps)) < 0.02, "the surrogate's median regret has grown"
    assert float(np.median(rhos)) > 0.85, "oracle_deviation must still rank candidates well"


def test_realised_mae_and_regret_are_consistent():
    block = synthetic_block(n=16, seed=4)
    split = make_p2_split(16, np.random.default_rng((1, 2, 3)))
    labels = oracle_labels(block["log_D"].to_numpy(dtype=float),
                           block["prediction"].to_numpy(dtype=float),
                           split.pool, split.evaluation)
    assert labels["regret"].min() == pytest.approx(0.0, abs=1e-12)
    assert (labels["regret"] >= -1e-12).all()


# --------------------------------------------------------------------------- #
# 16. feature provenance
# --------------------------------------------------------------------------- #

def test_no_deployable_feature_reads_a_target():
    for group in ("geometry", "prediction", "curve", "geometry+prediction", "full"):
        for name in FEATURE_GROUPS[group]:
            entry = FEATURE_PROVENANCE[name]
            assert entry["uses_target"] is False, f"{name} is declared to read a target"
            assert entry["available_pre_measurement"] is True, f"{name} is not knowable in advance"


def test_every_emitted_column_is_declared():
    block = synthetic_block(n=18, seed=7)
    pool = np.arange(0, 18, 2)
    features = candidate_features(block, block["prediction"].to_numpy(dtype=float), pool)
    emitted = set(features.columns) - {"candidate"}
    undeclared = emitted - set(FEATURE_PROVENANCE)
    assert not undeclared, f"undeclared acquisition features: {sorted(undeclared)}"


def test_features_do_not_change_when_every_target_is_destroyed():
    """The decisive check: features are a function of conditions and predictions."""
    block = synthetic_block(n=22, seed=9)
    pool = np.arange(0, 22, 2)
    prediction = block["prediction"].to_numpy(dtype=float)
    clean = candidate_features(block, prediction, pool)
    poisoned = block.copy()
    poisoned["log_D"] = np.random.default_rng(0).normal(500, 100, size=len(block))
    dirty = candidate_features(poisoned, prediction, pool)
    pd.testing.assert_frame_equal(clean, dirty)


# --------------------------------------------------------------------------- #
# 8. candidate normalisation uses the candidate pool only
# --------------------------------------------------------------------------- #

def test_normalisation_uses_the_pool_and_not_the_whole_block():
    """Percentiles must be percentiles *within the candidate list*.

    If they were computed over the whole ligand block they would silently encode
    the evaluation rows' positions, which is information the experimentalist does
    not have when choosing between the conditions in front of them.
    """
    block = synthetic_block(n=30, seed=13)
    prediction = block["prediction"].to_numpy(dtype=float)
    pool = np.arange(0, 10)
    features = candidate_features(block, prediction, pool)
    for column in ("pos_acid", "pos_extractant", "pred_pos"):
        values = np.sort(features[column].to_numpy())
        assert values.min() == pytest.approx(1.0 / len(pool), abs=1e-9)
        assert values.max() == pytest.approx(1.0, abs=1e-9)


def test_changing_rows_outside_the_pool_cannot_move_a_pool_feature():
    block = synthetic_block(n=30, seed=17)
    prediction = block["prediction"].to_numpy(dtype=float)
    pool = np.arange(0, 10)
    clean = candidate_features(block, prediction, pool)
    moved = block.copy()
    outside = np.arange(10, 30)
    moved.loc[outside, "massact__log10_cond__acid_concentration_M"] += 50.0
    moved_prediction = prediction.copy()
    moved_prediction[outside] += 100.0
    dirty = candidate_features(moved, moved_prediction, pool)
    for column in ("pos_acid", "pred_pos", "dist_medoid_rank", "pred_dev_from_median"):
        np.testing.assert_allclose(clean[column], dirty[column], rtol=0, atol=1e-9,
                                   err_msg=f"{column} depends on rows outside the pool")


# --------------------------------------------------------------------------- #
# 4. corruption cannot move a selection
# --------------------------------------------------------------------------- #

def _fit_policy(kind="rank", feature_set="geometry+prediction"):
    frames = []
    for i in range(14):
        frames.append(synthetic_block(n=18, seed=100 + i, ligand=f"L{i}"))
    oof = pd.concat(frames, ignore_index=True)
    dataset = build_dataset(oof, membership=None, repeats=4, seed=20260821)
    columns = tuple(FEATURE_GROUPS[feature_set])
    model = (PairwiseAcquisition(columns=columns) if kind == "rank"
             else ScalarAcquisition(columns=columns, n_estimators=40))
    model.fit(dataset.features, dataset.labels)
    return LearnedPolicy(models={(104729, 0): model}, columns=columns, name=f"LEARNED_{kind}")


@pytest.mark.parametrize("kind", ["rank", "scalar"])
def test_corrupting_hidden_targets_cannot_change_the_selection(kind):
    policy = _fit_policy(kind)
    block = synthetic_block(n=24, seed=999, ligand="HELD")
    split = make_p2_split(24, np.random.default_rng((20260820, 0, stable_hash("HELD"))))
    clean = policy(make_context(block, split.pool, split.evaluation), [])

    rng = np.random.default_rng(5)
    for _ in range(5):
        poisoned = block.copy()
        poisoned["log_D"] = rng.normal(1000, 500, size=len(block))
        assert policy(make_context(poisoned, split.pool, split.evaluation), []) == clean


@pytest.mark.parametrize("kind", ["rank", "scalar"])
def test_the_learned_policy_never_reads_the_evaluation_indices(kind):
    """Which rows will be scored is not information a first-point rule may use."""
    policy = _fit_policy(kind)
    block = synthetic_block(n=24, seed=321, ligand="HELD")
    split = make_p2_split(24, np.random.default_rng((20260820, 1, stable_hash("HELD"))))
    reference = policy(make_context(block, split.pool, split.evaluation), [])
    shuffled = np.random.default_rng(2).permutation(split.evaluation)
    assert policy(make_context(block, split.pool, shuffled), []) == reference
    subset = split.evaluation[:2]
    assert policy(make_context(block, split.pool, subset), []) == reference


def test_the_learned_policy_selects_from_the_pool_and_never_repeats():
    policy = _fit_policy("rank")
    block = synthetic_block(n=24, seed=77, ligand="HELD")
    split = make_p2_split(24, np.random.default_rng((20260820, 2, stable_hash("HELD"))))
    chosen: list[int] = []
    for _ in range(5):
        pick = policy(make_context(block, split.pool, split.evaluation), chosen)
        assert pick in set(split.pool.tolist())
        assert pick not in chosen
        chosen.append(pick)


def test_a_missing_fold_model_raises_rather_than_degrading_silently():
    """A fold with no fitted model must not quietly become a different policy."""
    policy = _fit_policy("rank")
    block = synthetic_block(n=12, seed=5, ligand="HELD")
    block["fold"] = 3
    split = make_p2_split(12, np.random.default_rng((1, 1, 1)))
    with pytest.raises(KeyError, match="no acquisition model fitted"):
        policy(make_context(block, split.pool, split.evaluation), [])


def test_later_picks_delegate_to_a_spread_policy():
    """gen8's first-point/later-point asymmetry, preserved by construction."""
    policy = _fit_policy("rank")
    block = synthetic_block(n=24, seed=88, ligand="HELD")
    split = make_p2_split(24, np.random.default_rng((3, 3, 3)))
    context = make_context(block, split.pool, split.evaluation)
    first = policy(context, [])
    second = policy(context, [first])
    expected = POLICIES[policy.later_policy](context, [first])
    assert second == expected


# --------------------------------------------------------------------------- #
# dataset construction
# --------------------------------------------------------------------------- #

def test_the_training_dataset_never_contains_a_held_out_ligand():
    frames = [synthetic_block(n=16, seed=200 + i, ligand=f"T{i}") for i in range(6)]
    oof = pd.concat(frames, ignore_index=True)
    dataset = build_dataset(oof, membership=None, repeats=3, seed=20260821,
                            ligands=[f"T{i}" for i in range(4)])
    assert set(dataset.features["extractant"]) == {"T0", "T1", "T2", "T3"}


def test_dataset_audit_reports_the_identity_rate():
    frames = [synthetic_block(n=16, seed=300 + i, ligand=f"U{i}") for i in range(6)]
    oof = pd.concat(frames, ignore_index=True)
    dataset = build_dataset(oof, membership=None, repeats=3, seed=20260821)
    for key in ("identity_exact_share", "identity_max_gap", "identity_spearman_median"):
        assert key in dataset.audit and np.isfinite(dataset.audit[key])


def test_scaling_is_fitted_on_the_training_features_only():
    """A scaler refitted at deployment would import the held-out pool's moments."""
    frames = [synthetic_block(n=16, seed=400 + i, ligand=f"V{i}") for i in range(8)]
    oof = pd.concat(frames, ignore_index=True)
    dataset = build_dataset(oof, membership=None, repeats=3, seed=20260821)
    columns = tuple(FEATURE_GROUPS["geometry"])
    model = PairwiseAcquisition(columns=columns).fit(dataset.features, dataset.labels)
    before = model._mean.copy()
    block = synthetic_block(n=20, seed=555, ligand="HELD")
    model.score(candidate_features(block, block["prediction"].to_numpy(dtype=float),
                                   np.arange(20)))
    np.testing.assert_array_equal(model._mean, before)
