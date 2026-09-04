"""Phase 0 — freeze, reproduce, and the metamorphic tests the brief requires.

Fast tests run on synthetic frames; tests marked ``slow`` need the built cohort
and the gen9 curve table on disk and refit real forests.  The slow ones are the
ones that matter for the freeze — exact nesting of the gen9 arms inside the gen10
classes, and determinism in-process and across interpreters with different
``PYTHONHASHSEED`` values — and they are what ``scripts/gen10_phase0.py`` runs
before any new arm is allowed to train.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen10 import features as F  # noqa: E402
from lanthanide_separation.gen10.consistency import build_variants, eligible_curves  # noqa: E402
from lanthanide_separation.gen10.perturb import (  # noqa: E402
    assert_no_target, recompute_derived, synthesise_points,
)
from lanthanide_separation.gen10.querycurves import (  # noqa: E402
    assert_partition_closure, query_membership, restricted_membership, sort_membership,
)
from lanthanide_separation.gen10.runner import DETERMINISM_TOLERANCE  # noqa: E402

COHORT = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
MEMBERSHIP = REPO_ROOT / "runs" / "gen9_shape" / "curves" / "curve_membership.parquet"
needs_cohort = pytest.mark.skipif(not (COHORT.exists() and MEMBERSHIP.exists()),
                                  reason="needs the built cohort and gen9 curve table")
slow = pytest.mark.slow


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #

def synthetic_ligand(n_points: int = 7, *, extractant: str = "L", series: str = "S",
                     metal: str = "Eu", log_low: float = -2.0, log_high: float = 0.0,
                     acid: float = 1.0) -> pd.DataFrame:
    """One extractant titration: ``n_points`` rows, extractant concentration varying."""
    values = np.logspace(log_low, log_high, n_points)
    frame = pd.DataFrame({
        "row_id": [f"{extractant}-{series}-{i}" for i in range(n_points)],
        "extractant": extractant, "series_id": series, "metal_symbol": metal,
        "lanthanide_index": 6, "metal_Z": 63,
        "cond__acid_concentration_M": acid,
        "cond__extractant_concentration_M": values,
        "cond__metal_concentration_mM": 1.0,
        "cond__temperature_C": 25.0, "cond__contact_time_min": 30.0,
        "DENTATE": 3, "coreCN": 9,
        "massact__log10_cond__acid_concentration_M": np.log10(acid),
        "massact__log10_cond__contact_time_min": np.log10(30.0),
        "massact__log10_cond__extractant_concentration_M": np.log10(values),
        "massact__log10_cond__metal_concentration_mM": 0.0,
        "massact__log10_cond__temperature_C": np.log10(25.0),
        "massact__logL_x_DENTATE": np.log10(values) * 3,
        "massact__logL_x_coreCN": np.log10(values) * 9,
        "massact__logL_x_logH": np.log10(values) * np.log10(acid),
    })
    return frame


# --------------------------------------------------------------------------- #
# Query-scoped curves
# --------------------------------------------------------------------------- #

def test_query_membership_finds_the_titration():
    frame = synthetic_ligand(6)
    membership = query_membership(frame)
    assert membership["curve_id"].nunique() == 1
    assert set(membership["row_id"]) == set(frame["row_id"])
    assert (membership["axis_label"] == "extractant").all()
    np.testing.assert_allclose(np.sort(membership["axis_value"]),
                               np.log10(np.sort(frame["cond__extractant_concentration_M"])))


def test_partition_closure_catches_a_straddling_curve():
    frame = synthetic_ligand(6)
    membership = query_membership(frame)
    assert_partition_closure(membership, frame["row_id"], name="whole")
    with pytest.raises(AssertionError):
        assert_partition_closure(membership, frame["row_id"].iloc[:3], name="half")


@needs_cohort
def test_per_ligand_reconstruction_equals_the_cohort_table():
    """The deployment object and the evaluation object agree on a closed partition."""
    cohort = pd.read_parquet(COHORT)
    membership = pd.read_parquet(MEMBERSHIP)
    from lanthanide_separation.gen9.curves import DEFAULT_AXES
    mismatches = []
    for ligand, rows in cohort.groupby("extractant"):
        rebuilt = sort_membership(query_membership(rows))
        stored = sort_membership(restricted_membership(
            membership[membership["axis"].isin(DEFAULT_AXES)], rows["row_id"]))
        if not rebuilt.equals(stored):
            mismatches.append(ligand)
    assert not mismatches, f"{len(mismatches)} ligands rebuild differently: {mismatches[:3]}"


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

@needs_cohort
def test_gen10_features_reproduce_gen9_columns_bit_for_bit():
    from lanthanide_separation.gen9.relative import RELATIVE_COLUMNS, relative_position_features

    cohort = pd.read_parquet(COHORT)
    membership = pd.read_parquet(MEMBERSHIP)
    gen9 = relative_position_features(cohort, membership).set_index("row_id")[list(RELATIVE_COLUMNS)]
    gen10 = F.as_gen9_frame(F.query_features(cohort, membership, gen9_compat=True)).set_index("row_id")
    gen9 = gen9.reindex(gen10.index)
    np.testing.assert_array_equal(gen10.to_numpy(dtype=float), gen9.to_numpy(dtype=float))


def test_every_feature_has_a_declared_sensitivity_class_and_sets_are_closed():
    assert set(F.ALL_COLUMNS) == set(F.SENSITIVITY)
    for name, columns in F.FEATURE_SETS.items():
        assert set(columns) <= set(F.ALL_COLUMNS), name
    assert not F.representation("RANK").has_endpoint_terms
    assert not F.representation("LOCAL").has_endpoint_terms
    assert not F.representation("HYBRID").has_endpoint_terms
    assert F.representation("GEN9").has_endpoint_terms


def test_non_constant_condition_coordinates_stay_non_constant():
    """Brief, metamorphic test 2: preprocessing must not flatten a varying axis."""
    from lanthanide_separation.gen8.protocols import _standardised_axes

    frame = synthetic_ligand(8)
    features = F.query_features(frame, query_membership(frame))
    varying = ["q10__axis_value", "q10__position", "q10__rank_pct", "q10__offset_from_median"]
    for column in varying:
        assert features[column].nunique() == len(frame), column
    axes = _standardised_axes(frame)
    extractant_axis = axes[:, 1] if axes.shape[1] > 1 else axes[:, 0]
    assert np.ptp(axes, axis=0).max() > 0
    assert len(np.unique(np.round(extractant_axis, 9))) == len(frame)


def test_rank_and_local_features_ignore_a_far_decoy_but_endpoint_features_do_not():
    """The sensitivity table is a prediction; check it on a synthetic design."""
    frame = synthetic_ligand(6)
    decoy = synthesise_points(frame.iloc[0], "cond__extractant_concentration_M", [3.0])
    with_decoy = pd.concat([frame, decoy], ignore_index=True)
    before = F.query_features(frame, query_membership(frame)).set_index("row_id")
    after = F.query_features(with_decoy, query_membership(with_decoy)).set_index("row_id")
    shared = list(frame["row_id"])
    for column in ("q10__gap_below", "q10__gap_above", "q10__min_gap"):
        # Interior points keep their neighbours; only the top endpoint's `above`
        # gap changes, because the decoy *is* now its neighbour above.
        interior = shared[1:-1]
        np.testing.assert_allclose(before.loc[interior, column], after.loc[interior, column])
    # rank-based columns move by O(1/n), endpoint-based ones rescale wholesale
    rank_shift = np.abs(before.loc[shared, "q10__rank_pct"] - after.loc[shared, "q10__rank_pct"]).max()
    position_shift = np.abs(before.loc[shared, "q10__position"] - after.loc[shared, "q10__position"]).max()
    assert position_shift > 0.5
    assert rank_shift < 0.2
    assert (after.loc[shared, "q10__window_width"] > before.loc[shared, "q10__window_width"]).all()


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #

def test_synthesised_points_keep_derived_columns_consistent():
    frame = synthetic_ligand(5)
    new = synthesise_points(frame.iloc[0], "cond__extractant_concentration_M", [-0.5, 0.5])
    assert new["row_id"].str.startswith("syn:").all()
    np.testing.assert_allclose(new["massact__log10_cond__extractant_concentration_M"], [-0.5, 0.5])
    np.testing.assert_allclose(new["massact__logL_x_DENTATE"], [-1.5, 1.5])
    np.testing.assert_allclose(new["massact__logL_x_logH"], [0.0, 0.0])
    massact = [c for c in frame.columns if c.startswith("massact__")]
    np.testing.assert_array_equal(recompute_derived(frame)[massact].to_numpy(dtype=float),
                                  frame[massact].to_numpy(dtype=float))


def test_synthetic_row_ids_are_process_independent():
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from tests.test_gen10_phase0 import synthetic_ligand\n"
            "from lanthanide_separation.gen10.perturb import synthesise_points\n"
            "f = synthetic_ligand(4)\n"
            "print(synthesise_points(f.iloc[0], 'cond__extractant_concentration_M', [1.5])"
            "['row_id'].iloc[0])" % str(REPO_ROOT))
    outputs = set()
    for hashseed in ("0", "1", "12345"):
        env = {"PYTHONHASHSEED": hashseed, "PATH": os.environ.get("PATH", ""),
               "PYTHONPATH": str(REPO_ROOT / "src")}
        outputs.add(subprocess.check_output([sys.executable, "-c", code], env=env,
                                            cwd=REPO_ROOT, text=True).strip())
    assert len(outputs) == 1


def test_a_query_may_not_carry_a_target():
    frame = synthetic_ligand(4).assign(log_D=1.0)
    with pytest.raises(AssertionError):
        assert_no_target(frame)


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #

def test_variants_share_rows_with_the_reference_only_where_claimed():
    frame = synthetic_ligand(7)
    membership = eligible_curves(frame)
    curve_id = membership["curve_id"].iloc[0]
    variants = build_variants(frame, curve_id, "cond__extractant_concentration_M", membership,
                              rng=np.random.default_rng(0))
    families = {v.family for v in variants}
    assert {"REFERENCE", "CONTEXT", "NESTED", "PERMUTE", "SPARSE", "DENSITY", "EXTEND",
            "DECOY", "SHIFT"} <= families
    reference = variants[0].frame
    for variant in variants:
        assert "log_D" not in variant.frame.columns
        present = set(variant.frame["row_id"].astype(str))
        if variant.family != "SHIFT":
            assert set(variant.compare_ids) <= present
        if variant.family == "DENSITY":
            # window unchanged, interior points added
            values = np.log10(variant.frame["cond__extractant_concentration_M"])
            assert values.min() == pytest.approx(np.log10(reference["cond__extractant_concentration_M"]).min())
            assert values.max() == pytest.approx(np.log10(reference["cond__extractant_concentration_M"]).max())
        if variant.family == "PERMUTE":
            assert sorted(variant.frame["row_id"]) == sorted(reference["row_id"])


# --------------------------------------------------------------------------- #
# Acquisition metamorphics
# --------------------------------------------------------------------------- #

def _policy_context(frame: pd.DataFrame, pool: np.ndarray):
    from lanthanide_separation.gen8.kshot import PolicyContext
    from lanthanide_separation.gen8.protocols import _standardised_axes

    n = len(frame)
    return PolicyContext(block=frame, prediction=np.zeros(n), pool=pool,
                         evaluation=np.array([], dtype=int), axes=_standardised_axes(frame),
                         uncertainty=np.full(n, np.nan), disagreement=np.full(n, np.nan),
                         rng=np.random.default_rng(0), truth=None)


def test_medoid_and_farthest_disagree_on_an_asymmetric_pool():
    """Brief, metamorphic test 1."""
    from lanthanide_separation.gen8.kshot import POLICIES
    from lanthanide_separation.gen9.acquisition import register_extra_policies

    register_extra_policies()
    # five points bunched at the low end and one far out at the high end
    frame = synthetic_ligand(6)
    frame.loc[:4, "cond__extractant_concentration_M"] = [0.010, 0.011, 0.012, 0.013, 0.014]
    frame.loc[5, "cond__extractant_concentration_M"] = 1.0
    frame = recompute_derived(frame)
    pool = np.arange(len(frame))
    context = _policy_context(frame, pool)
    medoid = POLICIES["MEDOID"](context, [])
    farthest_first = POLICIES["FARTHEST_FROM_EXISTING"](context, [])
    assert medoid in {1, 2, 3}, "the medoid must sit inside the bunch"
    assert farthest_first != medoid
    # and after the medoid is taken, FARTHEST must jump to the outlier
    context = _policy_context(frame, pool)
    assert POLICIES["FARTHEST_FROM_EXISTING"](context, [medoid]) == 5


def test_corrupting_targets_changes_neither_selection_nor_gen10_prediction(tmp_path):
    """Brief, metamorphic test 3 (selection half; prediction half is slow)."""
    from lanthanide_separation.gen8.kshot import POLICIES

    frame = synthetic_ligand(8)
    pool = np.arange(len(frame))
    picks = {}
    for label, corruption in (("clean", None), ("extreme", 1e4), ("missing", "drop")):
        work = frame.copy()
        if corruption == "drop":
            pass
        elif corruption is not None:
            work["log_D"] = corruption
        context = _policy_context(work, pool)
        picks[label] = [POLICIES[p](context, []) for p in ("MEDOID", "CENTRAL",
                                                            "FARTHEST_FROM_EXISTING")]
    assert picks["clean"] == picks["extreme"] == picks["missing"]


# --------------------------------------------------------------------------- #
# Slow: exact nesting, determinism, equivariance, target corruption
# --------------------------------------------------------------------------- #

def _fold0(cohort_obj):
    from lanthanide_separation.gen7.harness import FoldContext, build_folds
    from lanthanide_separation.levels import LEVEL_TARGET_COLUMN

    frame = cohort_obj.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    fold = build_folds(frame, 104729)[0]
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
    context = FoldContext(fold=fold, cohort=cohort_obj, feature_columns=(),
                          model_seed=fold.model_seed)
    return train, target[fold.train_index], test, context, target[fold.test_index]


@pytest.fixture(scope="module")
def real_fold():
    if not (COHORT.exists() and MEMBERSHIP.exists()):
        pytest.skip("needs the built cohort and gen9 curve table")
    from lanthanide_separation.gen7.harness import load_cohort
    from lanthanide_separation.gen10.architectures import prime_raw_cache

    cohort = load_cohort()
    prime_raw_cache(cohort)
    return cohort, _fold0(cohort)


@slow
def test_gen10_arms_nest_the_gen9_arms_exactly(real_fold):
    from lanthanide_separation.gen9.train import FrozenExtraTrees, ShapeRecomposed, load_membership
    from lanthanide_separation.gen10.architectures import MonolithModel, RecomposedModel

    cohort, (train, y, test, context, _) = real_fold
    membership = load_membership()
    pairs = [
        (FrozenExtraTrees(), MonolithModel(representation_name="NONE")),
        (FrozenExtraTrees(with_relative=True, membership=membership),
         MonolithModel(representation_name="GEN9", gen9_compat=True)),
        (ShapeRecomposed(membership=membership),
         RecomposedModel(representation_name="GEN9", gen9_compat=True)),
    ]
    for gen9_arm, gen10_arm in pairs:
        a = gen9_arm.fit_predict(train, y, test, context)
        b = gen10_arm.fit(train, y, context).predict(test)
        assert np.abs(a - b).max() <= DETERMINISM_TOLERANCE, type(gen10_arm).__name__


@slow
@pytest.mark.parametrize("factory", [
    lambda: __import__("lanthanide_separation.gen10.architectures", fromlist=["x"]).MonolithModel(
        representation_name="GEN9", gen9_compat=True),
    lambda: __import__("lanthanide_separation.gen10.architectures", fromlist=["x"]).RecomposedModel(
        representation_name="HYBRID"),
    lambda: __import__("lanthanide_separation.gen10.architectures", fromlist=["x"]).LevelShapeModel(),
    lambda: __import__("lanthanide_separation.gen10.architectures", fromlist=["x"]).ResidualShapeModel(
        n_estimators=60),
])
def test_same_process_refit_is_deterministic(real_fold, factory):
    cohort, (train, y, test, context, _) = real_fold
    a = factory().fit(train, y, context).predict(test)
    b = factory().fit(train, y, context).predict(test)
    assert np.abs(a - b).max() <= DETERMINISM_TOLERANCE


@slow
def test_subprocesses_with_different_hashseeds_agree(tmp_path):
    """Brief: run in clean subprocesses; fail if identical configurations differ."""
    code = f"""
import sys, json, numpy as np
sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); sys.path.insert(0, {str(REPO_ROOT)!r})
from lanthanide_separation.gen7.harness import load_cohort
from lanthanide_separation.gen10.architectures import RecomposedModel, prime_raw_cache
from tests.test_gen10_phase0 import _fold0
cohort = load_cohort(); prime_raw_cache(cohort)
train, y, test, context, _ = _fold0(cohort)
p = RecomposedModel(representation_name="GEN9", gen9_compat=True, n_estimators=80).fit(
    train, y, context).predict(test)
print(json.dumps([round(float(v), 9) for v in p[:50]]))
"""
    outputs = []
    for hashseed in ("0", "4242"):
        env = {"PYTHONHASHSEED": hashseed, "PATH": os.environ.get("PATH", "")}
        out = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=REPO_ROOT,
                                      text=True)
        outputs.append(np.asarray(json.loads(out.strip().splitlines()[-1])))
    assert np.abs(outputs[0] - outputs[1]).max() <= 1e-9


@slow
def test_permuting_query_rows_permutes_predictions(real_fold):
    """Brief, metamorphic test 4: equivariance, at the float floor."""
    from lanthanide_separation.gen10.architectures import RecomposedModel

    cohort, (train, y, test, context, _) = real_fold
    fitted = RecomposedModel(representation_name="GEN9", gen9_compat=True).fit(train, y, context)
    ligand = test["extractant"].value_counts().index[0]
    query = test[test["extractant"] == ligand].reset_index(drop=True)
    permutation = np.random.default_rng(3).permutation(len(query))
    a = fitted.predict(query)
    b = fitted.predict(query.iloc[permutation].reset_index(drop=True))
    assert np.abs(a[permutation] - b).max() <= DETERMINISM_TOLERANCE


@slow
def test_held_out_targets_cannot_reach_a_gen10_prediction(real_fold):
    """Brief, metamorphic test 3: drop or corrupt held-out targets; predictions fixed."""
    from lanthanide_separation.gen10.architectures import LevelShapeModel, RecomposedModel

    cohort, (train, y, test, context, _) = real_fold
    for cls in (RecomposedModel, LevelShapeModel):
        fitted = cls(n_estimators=80).fit(train, y, context)
        clean = fitted.predict(test)
        corrupted = fitted.predict(test.assign(log_D=np.random.default_rng(1).normal(1e4, 1e3, len(test))))
        assert np.abs(clean - corrupted).max() <= DETERMINISM_TOLERANCE, cls.__name__


# --------------------------------------------------------------------------- #
# Slow: the learned set representation keeps the invariants it promises
# --------------------------------------------------------------------------- #

@slow
def test_set_encoder_is_deterministic_equivariant_and_target_free(real_fold):
    from lanthanide_separation.gen10.setcontext import SetRecomposedModel

    cohort, (train, y, test, context, _) = real_fold
    model = SetRecomposedModel(pooling="mean", epochs=60, inner_validation=False)
    fitted = model.fit(train, y, context)
    a = fitted.predict(test)
    b = SetRecomposedModel(pooling="mean", epochs=60, inner_validation=False).fit(
        train, y, context).predict(test)
    assert np.abs(a - b).max() <= DETERMINISM_TOLERANCE
    ligand = test["extractant"].value_counts().index[0]
    query = test[test["extractant"] == ligand].reset_index(drop=True)
    permutation = np.random.default_rng(5).permutation(len(query))
    pa = fitted.predict(query)
    pb = fitted.predict(query.iloc[permutation].reset_index(drop=True))
    assert np.abs(pa[permutation] - pb).max() <= 1e-5          # float32 head
    corrupted = fitted.predict(test.assign(log_D=1e4))
    assert np.abs(a - corrupted).max() <= DETERMINISM_TOLERANCE
    # the shape it adds is zero-mean over every curve: level equals the base level
    from lanthanide_separation.gen10.architectures import MonolithModel
    base = MonolithModel(representation_name="NONE").fit(train, y, context).predict(query)
    from lanthanide_separation.gen10.querycurves import primary_curve, query_membership
    primary = primary_curve(query_membership(query)).set_index("row_id")["curve_id"]
    curve = query["row_id"].astype(str).map(primary)
    for cid, rows in query.groupby(curve).groups.items():
        index = np.asarray(rows, dtype=int)
        assert abs(pa[index].mean() - base[index].mean()) <= 1e-6
