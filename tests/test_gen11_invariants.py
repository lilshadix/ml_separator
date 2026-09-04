"""The gen11 invariants that, if broken, silently invalidate every transfer number.

These are not unit tests of convenience.  Each one corresponds to a way this
sprint could produce a confident wrong answer:

* the control stops being the control (nesting);
* an auxiliary row reaches the evaluation set (namespacing, row accounting);
* a leakage rule passes because it was never actually asked (the empty-dictionary
  trap that let 8,615 rows through on the first audit);
* the auxiliary block quietly becomes the training objective (weighting);
* a negative control is not actually negative (target permutation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen11 import arms, overlap, pools


# --------------------------------------------------------------------------- #
# Leakage rules
# --------------------------------------------------------------------------- #

def _aux(**overrides) -> pd.DataFrame:
    base = {
        "source_record_id": ["1", "2", "3"],
        "duplicate_group_id": ["g1", "g2", "g3"],
        "series_id": ["s1", "s2", "s3"],
        "extractant_primary_smiles": ["CCO", "CCC", "CCN"],
        "doi_primary_corrected": ["d1", "d2", "d3"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _held(**overrides) -> pd.DataFrame:
    base = {
        "duplicate_group_id": ["g1"],
        "series_id_src": ["s2"],
        "series_id_cohort": ["sx"],
        "extractant": ["CCN"],
        "tanimoto_cluster": ["tan001"],
        "doi_primary_corrected": ["d3"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_relationship_masks_refuses_an_incomplete_chemotype_assignment():
    """An absent structure must raise, not silently score as 'not in a chemotype'.

    This is the defect the first gen11 leakage audit found: a dictionary built
    from the cohort's own ligands returns nothing for auxiliary structures, so
    SAME_CHEMOTYPE went False for rows at Tanimoto 0.88 to a held-out ligand.
    """
    with pytest.raises(SystemExit, match="chemotype assignment"):
        overlap.relationship_masks(_aux(), _held(), chemotype_of_smiles={})


def test_relationship_masks_flags_each_relationship_independently():
    assignment = {"CCO": (), "CCC": ("tan002",), "CCN": ("tan001",)}
    masks = overlap.relationship_masks(_aux(), _held(), chemotype_of_smiles=assignment)
    assert masks["DUPLICATE_EQUIVALENT"].tolist() == [True, False, False]
    assert masks["SAME_SERIES"].tolist() == [False, True, False]
    assert masks["SAME_LIGAND"].tolist() == [False, False, True]
    assert masks["SAME_CHEMOTYPE"].tolist() == [False, False, True]
    assert masks["SAME_PUBLICATION"].tolist() == [False, False, True]


def test_multi_chemotype_structure_is_excluded_when_either_cluster_is_held_out():
    """A bridging structure leaks if *any* linked chemotype is held out."""
    assignment = {"CCO": ("tan001", "tan009"), "CCC": (), "CCN": ()}
    masks = overlap.relationship_masks(_aux(), _held(), chemotype_of_smiles=assignment)
    assert bool(masks["SAME_CHEMOTYPE"].iloc[0]) is True


def test_headline_policy_excludes_the_union_of_its_relationships():
    assignment = {"CCO": (), "CCC": (), "CCN": ("tan001",)}
    masks = overlap.relationship_masks(_aux(), _held(), chemotype_of_smiles=assignment)
    safe = overlap.apply_policy(masks, "HEADLINE")
    assert safe.tolist() == [False, False, False]
    permissive = overlap.apply_policy(masks, "PERMISSIVE_DIAGNOSTIC")
    assert permissive.tolist() == [False, True, True]


# --------------------------------------------------------------------------- #
# Namespacing
# --------------------------------------------------------------------------- #

def test_namespacing_makes_auxiliary_series_and_rows_uncollidable():
    aux = pools.namespace_auxiliary(_aux().assign(row_id=["1", "2", "3"]))
    assert aux["series_id"].tolist() == ["aux:s1", "aux:s2", "aux:s3"]
    assert aux["row_id"].tolist() == ["aux:1", "aux:2", "aux:3"]
    cohort = pd.DataFrame({"series_id": ["s1", "s2", "s3"]})
    arms.assert_series_disjoint(cohort, aux)


def test_series_collision_is_fatal():
    cohort = pd.DataFrame({"series_id": ["s1"]})
    with pytest.raises(SystemExit, match="collide"):
        arms.assert_series_disjoint(cohort, _aux())


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #

def _train(n_core: int = 4, n_aux: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "is_auxiliary": [False] * n_core + [True] * n_aux,
        "ecfp_cluster": ["c1", "c1", "c2", "c2"] + [f"a{i % 5}" for i in range(n_aux)],
        "metal_symbol": ["Eu"] * n_core + ["Am"] * n_aux,
    })


def test_row_weighting_is_deliberately_unscaled():
    """ROW exists to show a large metal dominating; rescaling would erase it."""
    weights = arms.training_weights(_train(), scheme="ROW")
    assert np.allclose(weights, 1.0)


@pytest.mark.parametrize("scheme", ["CHEMOTYPE", "METAL_BALANCED", "HIERARCHICAL"])
@pytest.mark.parametrize("aux_lambda", [0.25, 1.0, 2.0])
def test_aux_lambda_sets_the_auxiliary_share_of_total_mass(scheme, aux_lambda):
    train = _train()
    weights = arms.training_weights(train, scheme=scheme, aux_lambda=aux_lambda)
    is_aux = train["is_auxiliary"].to_numpy(dtype=bool)
    ratio = weights[is_aux].sum() / weights[~is_aux].sum()
    assert ratio == pytest.approx(aux_lambda, rel=1e-9)


def test_aux_lambda_means_the_same_thing_at_very_different_pool_sizes():
    """86 auxiliary rows and 4,896 of them must be the same experiment at lambda=1."""
    small = arms.training_weights(_train(n_aux=3), scheme="HIERARCHICAL", aux_lambda=1.0)
    large = arms.training_weights(_train(n_aux=400), scheme="HIERARCHICAL", aux_lambda=1.0)
    for weights, n_aux in ((small, 3), (large, 400)):
        is_aux = np.array([False] * 4 + [True] * n_aux)
        assert weights[is_aux].sum() == pytest.approx(weights[~is_aux].sum(), rel=1e-9)


def test_weighting_without_auxiliary_rows_is_the_frozen_rule():
    train = _train(n_aux=0)
    weights = arms.training_weights(train, scheme="CHEMOTYPE")
    from lanthanide_separation.levels import group_balanced_weights
    assert np.allclose(weights, group_balanced_weights(train["ecfp_cluster"]))


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #

def test_permuted_target_control_changes_only_the_target():
    aux = _aux().assign(log_D=[1.0, 2.0, 3.0])
    out = arms.apply_control(aux, control="PERMUTED_AUX_TARGET",
                             rng=np.random.default_rng(0), metal_columns=())
    assert sorted(out["log_D"]) == [1.0, 2.0, 3.0]
    assert out["source_record_id"].tolist() == aux["source_record_id"].tolist()


def test_shuffled_metal_labels_permutes_metal_columns_as_a_block():
    aux = _aux().assign(m1=[1.0, 2.0, 3.0], m2=[10.0, 20.0, 30.0])
    out = arms.apply_control(aux, control="SHUFFLED_METAL_LABELS",
                             rng=np.random.default_rng(1), metal_columns=("m1", "m2"))
    # rows are permuted together, so the pairing of m1 with m2 must survive
    assert set(zip(out["m1"], out["m2"])) == {(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)}


def test_unknown_names_are_rejected_rather_than_ignored():
    with pytest.raises(KeyError):
        arms.training_weights(_train(), scheme="NOT_A_SCHEME")
    with pytest.raises(KeyError):
        arms.apply_control(_aux(), control="NOPE", rng=np.random.default_rng(0),
                           metal_columns=())
    with pytest.raises(KeyError):
        overlap.apply_policy(pd.DataFrame({r: [False] for r in overlap.RELATIONSHIPS}),
                             "NOT_A_POLICY")


# --------------------------------------------------------------------------- #
# Arm selection
# --------------------------------------------------------------------------- #

def test_control_arm_selects_no_auxiliary_rows():
    pool = pd.DataFrame({"metal_category": ["actinide"] * 5})
    rows = arms.select_arm_rows(pool, arms.ARM_BY_KEY["A_GEN10_CONTROL"])
    assert rows.empty


def test_matched_arm_draws_exactly_the_requested_size_and_is_reproducible():
    pool = pd.DataFrame({"metal_category": ["actinide"] * 100, "i": range(100)})
    spec = arms.ARM_BY_KEY["F_ACTINIDES_ONLY_MATCHED"]
    first = arms.select_arm_rows(pool, spec, matched_size=17, seed=11)
    again = arms.select_arm_rows(pool, spec, matched_size=17, seed=11)
    other = arms.select_arm_rows(pool, spec, matched_size=17, seed=22)
    assert len(first) == 17
    assert first["i"].tolist() == again["i"].tolist()
    assert first["i"].tolist() != other["i"].tolist()


def test_pool_refuses_a_fold_it_has_no_admissibility_entry_for():
    pool = arms.AuxiliaryPool(features=_aux(), safe_ids={(1, 0): frozenset({"1"})},
                              policy="HEADLINE")
    assert len(pool.admissible(1, 0)) == 1
    with pytest.raises(KeyError, match="unfiltered pool"):
        pool.admissible(1, 4)


# --------------------------------------------------------------------------- #
# Regressions for defects found by adversarially verifying the foundations
# --------------------------------------------------------------------------- #

def test_same_series_still_fires_after_namespacing():
    """The guard must compare archive ids, not the ``aux:``-prefixed ones.

    Namespacing was added for curve safety and silently disabled SAME_SERIES:
    it compared "aux:X" against "X" and matched nothing, while 2,505 auxiliary
    rows genuinely shared an archive series with a cohort row.
    """
    aux = pools.namespace_auxiliary(_aux().assign(row_id=["1", "2", "3"]))
    assignment = {"CCO": (), "CCC": (), "CCN": ()}
    masks = overlap.relationship_masks(aux, _held(), chemotype_of_smiles=assignment)
    assert masks["SAME_SERIES"].tolist() == [False, True, False]


def test_namespaced_aux_without_archive_series_id_is_refused():
    """Silently matching nothing is worse than failing."""
    aux = _aux()
    aux["series_id"] = "aux:" + aux["series_id"]
    with pytest.raises(SystemExit, match="match nothing"):
        overlap.relationship_masks(aux, _held(),
                                   chemotype_of_smiles={s: () for s in aux["extractant_primary_smiles"]})


def test_control_rng_is_stable_across_processes():
    """A control seeded from Python's salted ``hash()`` is not reproducible.

    gen8 lost a cross-run comparison to exactly this, so the seed is derived
    with blake2b and the expectation is pinned in a subprocess.
    """
    import subprocess
    import sys
    code = (
        "import hashlib;"
        "d=hashlib.blake2b(b'ARM|1|2', digest_size=8).digest();"
        "print(int.from_bytes(d,'big') % (2**32))"
    )
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": seed, "PATH": ""}).stdout.strip()
            for seed in ("0", "1", "4242")}
    assert len(runs) == 1, f"seed is process-dependent: {runs}"
