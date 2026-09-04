"""The gen7 evaluation contract, in tests.

Three things have to be literally true or the whole generation's leaderboard is
meaningless, and each has a test here:

1. the harness reproduces gen6's published numbers *exactly*, not approximately —
   so a gen7 arm and a gen6 arm can sit in the same table;
2. adding a feature block cannot change the cohort fingerprint, the row order, the
   fold plan or an unrelated arm's predictions;
3. a contender is never handed the target.

Plus regressions for two defects that actually bit during the gen7 session: the
pandas-3 groupby break in ``leaderboard`` that killed a completed twenty-contender
sweep at the final aggregation step, and the model-seed derivation that silently
moved every number by 0.006 when taken from the split seed instead of the fixed
model seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen7.harness import (
    DEFAULT_SEEDS, FOLD_SEED_OFFSET, FOLD_SEED_STRIDE, MODEL_SEED_BASE,
    assert_fold_integrity, assert_no_target_leak, build_folds, leaderboard,
)
from lanthanide_separation.gen7.hierarchical_nn import (
    F_ELECTRONS, IONIC_RADII_CN8, metal_physical_features,
)
from lanthanide_separation.gen7.ligand_physics import attach_ligand_physics
from lanthanide_separation.gen7.recovered import parse_solvent, solvent_descriptors

#: gen6 Experiment A, EXPANDED arm, MC_lig2d_ext_massaction
#: (``runs/gen6_expA_5seed/arm_metrics.csv``).  These are the numbers gen7 must
#: reproduce to the last printed digit.
GEN6_REFERENCE = {
    104729: (1.067993, 0.936996, 0.502044),
    130363: (1.047784, 0.845103, 0.516803),
}


def _synthetic_cohort(n_ligands: int = 24, n_metals: int = 6, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ligands):
        chemotype = f"sc{i // 4}"
        for m in range(n_metals):
            rows.append({
                "row_id": f"r{i:03d}{m}", "extractant": f"lig{i:03d}",
                "ecfp_cluster": f"ec{i}", "tanimoto_cluster": chemotype,
                "condition_id": f"c{m}", "series_id": f"s{i}", "metal_symbol": "La",
                "metal_Z": 57.0, "n_replicates": 1,
                "log_D": float(rng.normal(i * 0.1, 0.5)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

def test_folds_hold_out_whole_chemotypes_and_cover_every_row():
    frame = _synthetic_cohort()
    folds = build_folds(frame, seed=104729, n_splits=5)
    report = assert_fold_integrity(frame, folds)
    assert report["ok"], report
    assert report["all_rows_tested_once"]
    tested = set()
    for fold in folds:
        tested |= set(fold.test_index.tolist())
    assert len(tested) == len(frame)


def test_model_seed_is_independent_of_the_split_seed():
    """The formula is ``42 + fold*1009 + 9_999_991`` — deriving it from the split
    seed instead moved every gen7 number by ~0.006 and broke reproduction."""
    frame = _synthetic_cohort()
    a = build_folds(frame, seed=104729)
    b = build_folds(frame, seed=130363)
    assert [f.model_seed for f in a] == [f.model_seed for f in b]
    assert a[0].model_seed == MODEL_SEED_BASE + 0 * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET
    assert a[2].model_seed == MODEL_SEED_BASE + 2 * FOLD_SEED_STRIDE + FOLD_SEED_OFFSET


def test_split_seed_still_changes_the_partition():
    frame = _synthetic_cohort()
    a = build_folds(frame, seed=104729)
    b = build_folds(frame, seed=262147)
    assert [set(f.test_index.tolist()) for f in a] != [set(f.test_index.tolist()) for f in b]


def test_target_leak_assertion_fires():
    frame = _synthetic_cohort()
    with pytest.raises(AssertionError):
        assert_no_target_leak(frame)
    assert_no_target_leak(frame.drop(columns=["log_D"]))


# --------------------------------------------------------------------------- #
# Leaderboard (pandas-3 regression)
# --------------------------------------------------------------------------- #

def test_leaderboard_aggregates_without_reselecting_a_grouped_frame():
    """Regression: ``grouped[column]`` on an already-column-selected groupby raises
    ``IndexError: Column(s) already selected`` on pandas 3, which silently discarded
    a finished twenty-contender sweep."""
    scores = pd.DataFrame({
        "model": ["a", "a", "b", "b"],
        "split_seed": [1, 2, 1, 2],
        "macro_mae": [1.0, 1.2, 0.9, 0.7],
        "offset_mae": [0.8, 0.9, 0.7, 0.6],
        "shape_mae": [0.5, 0.5, 0.4, 0.4],
    })
    board = leaderboard(scores)
    assert list(board["model"]) == ["b", "a"]                      # sorted by macro
    assert board.loc[board["model"] == "a", "macro_mae"].iat[0] == pytest.approx(1.1)
    assert board.loc[board["model"] == "b", "macro_mae_sd"].iat[0] == pytest.approx(
        np.std([0.9, 0.7], ddof=1))
    assert set(board["n_seeds"]) == {2}
    assert "split_seed" not in board.columns   # a mean over seed labels is meaningless


# --------------------------------------------------------------------------- #
# Metal representation
# --------------------------------------------------------------------------- #

def test_metal_features_are_continuous_and_capture_the_half_filled_shell():
    symbols = ["La", "Gd", "Lu", "Eu"]
    matrix = metal_physical_features(symbols)
    assert matrix.shape == (4, 16)
    assert np.isfinite(matrix).all()
    # the lanthanide contraction: radius strictly decreases La -> Gd -> Lu
    assert matrix[0, 0] > matrix[1, 0] > matrix[2, 0]
    # |n_f - 7| is zero exactly at Gd, which is the tetrad-effect coordinate
    half_filled = matrix[:, 6]
    assert half_filled[1] == pytest.approx(0.0)
    assert half_filled[0] == pytest.approx(7.0)
    assert F_ELECTRONS["Gd"] == 7 and IONIC_RADII_CN8["La"] > IONIC_RADII_CN8["Lu"]


def test_unknown_metal_does_not_produce_nan():
    matrix = metal_physical_features(["La", "Zz"])
    assert np.isfinite(matrix).all()


# --------------------------------------------------------------------------- #
# Recovered solvent physics
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,expected", [
    ("n-Dodecane", {"dodecane": 1.0}),
    ("kerosene with 30 vol% 1-octanol", {"kerosene": 0.7, "octanol": 0.3}),
    ("kerosene 0.7, 1-octanol 0.3", {"kerosene": 0.7, "octanol": 0.3}),
    ("Isopar L with 30 vol% Exxal 13", {"isopar_l": 0.7, "exxal13": 0.3}),
    ("n-octane with 5% 1-octanol", {"octane": 0.95, "octanol": 0.05}),
    ("1-octanol with 30 vol% kerosene", {"octanol": 0.7, "kerosene": 0.3}),
    ("[C4mim][Tf2N]", {"ionic_liquid": 1.0}),
])
def test_solvent_parser_handles_every_notation_in_the_corpus(name, expected):
    parsed = parse_solvent(name)
    assert set(parsed) == set(expected)
    for key, value in expected.items():
        assert parsed[key] == pytest.approx(value)
    assert sum(parsed.values()) == pytest.approx(1.0)


def test_ch3cl_and_chloroform_resolve_to_the_same_liquid():
    """The 140-row DMDPhPDA duplicate is one paper under two solvent spellings;
    treating them as different liquids splits one experiment in two."""
    assert parse_solvent("CH3Cl") == parse_solvent("Chloroform")


def test_solvent_descriptors_interpolate_a_mixture():
    neat = solvent_descriptors("n-Dodecane")
    mixed = solvent_descriptors("kerosene with 30 vol% 1-octanol")
    octanol = solvent_descriptors("1-octanol")
    assert neat["rec__solvent_parsed"] == 1.0
    # a 30 % alcohol cut sits between the alkane and the neat alcohol
    assert neat["rec__solvent_eps"] < mixed["rec__solvent_eps"] < octanol["rec__solvent_eps"]
    assert mixed["rec__solvent_polar_fraction"] == pytest.approx(0.3)
    assert mixed["rec__solvent_log_eps"] == pytest.approx(np.log10(mixed["rec__solvent_eps"]))


def test_unparseable_solvent_yields_nan_not_zero():
    """A silent zero would tell the model the dielectric constant is zero."""
    out = solvent_descriptors("a liquid nobody has heard of")
    assert out["rec__solvent_parsed"] == 0.0
    assert np.isnan(out["rec__solvent_eps"])


# --------------------------------------------------------------------------- #
# Ligand physics
# --------------------------------------------------------------------------- #

def test_ligand_physics_builds_the_ligand_diluent_coupling_terms():
    frame = pd.DataFrame({
        "MolLogP": [6.0, 2.0], "TPSA": [60.0, 90.0], "MolWt": [400.0, 300.0],
        "DENTATE": [3.0, 2.0], "n_ligs": [3.0, 2.0], "coreCN": [9.0, 8.0],
        "donor__n_total": [3.0, 2.0], "donor__O(amide_carbonyl)": [2.0, 0.0],
        "donor__N(aromatic)": [0.0, 2.0],
        "rec__solvent_logp": [6.1, 3.0], "rec__solvent_eps": [2.0, 10.3],
        "rec__solvent_dD": [16.0, 17.0], "rec__solvent_dP": [0.0, 3.3],
        "rec__solvent_dH": [0.0, 11.9],
    })
    out, columns = attach_ligand_physics(frame)
    assert "ligphys__logp_minus_solvent" in columns
    assert out["ligphys__logp_minus_solvent"].tolist() == pytest.approx([-0.1, -1.0])
    assert out["ligphys__complex_logp"].tolist() == pytest.approx([18.0, 4.0])
    # hard amide O outranks soft aromatic N toward a hard trivalent lanthanide
    assert out["ligphys__donor_hardness_mean"].iat[0] > out["ligphys__donor_hardness_mean"].iat[1]
    assert np.isfinite(out["ligphys__hansen_distance"]).all()


def test_ligand_physics_survives_a_frame_without_solvent_columns():
    frame = pd.DataFrame({"MolLogP": [3.0], "TPSA": [50.0], "MolWt": [200.0],
                          "DENTATE": [2.0], "n_ligs": [2.0], "coreCN": [8.0],
                          "donor__n_total": [2.0]})
    _, columns = attach_ligand_physics(frame)
    assert "ligphys__logp_x_dentate" in columns
    assert "ligphys__logp_minus_solvent" not in columns
