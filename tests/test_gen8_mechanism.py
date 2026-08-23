"""Contract tests for the gen8 mechanistic ligand representation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen8.mechanism import (
    MECHANISM_COLUMNS,
    mechanism_distance,
    mechanism_features,
    n_unparsed,
)

TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
TBP = "CCCCOP(=O)(OCCCC)OCCCC"
HDEHP = "CCCCC(CC)COP(=O)(O)OCC(CC)CCCC"
CMPO = "CCCCCCCCP(=O)(CC(=O)N(CC(C)C)CC(C)C)c1ccccc1"
BTBP = "CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC"
CYANEX301 = "CC(C)CC(C)CP(=S)(S)CC(C)CC(C)C"
ALIQUAT = "CCCCCCCCCC[N+](C)(CCCCCCCCCC)CCCCCCCCCC.[Cl-]"
GARBAGE = "this is not a smiles((("

ALL_GOOD = [TODGA, TBP, HDEHP, CMPO, BTBP, CYANEX301, ALIQUAT]


# --------------------------------------------------------------------------
# shape / compactness
# --------------------------------------------------------------------------
def test_column_count_is_compact() -> None:
    """The whole point of gen8's representation is that it is small.

    It is contrasted against a 2048-bit ECFP and a 206-column descriptor table,
    both of which are known to be useless or harmful for the ligand level.
    """
    assert 10 <= len(MECHANISM_COLUMNS) <= 40
    assert len(set(MECHANISM_COLUMNS)) == len(MECHANISM_COLUMNS)
    assert all(c.startswith("mech__") for c in MECHANISM_COLUMNS)


def test_frame_shape_and_columns() -> None:
    frame = mechanism_features(ALL_GOOD)
    assert list(frame.columns) == list(MECHANISM_COLUMNS)
    assert len(frame) == len(ALL_GOOD)
    assert list(frame.index) == list(range(len(ALL_GOOD)))
    assert all(pd.api.types.is_float_dtype(frame[c]) for c in frame.columns)


def test_empty_input() -> None:
    frame = mechanism_features([])
    assert list(frame.columns) == list(MECHANISM_COLUMNS)
    assert len(frame) == 0


# --------------------------------------------------------------------------
# determinism and order preservation
# --------------------------------------------------------------------------
def test_determinism() -> None:
    a = mechanism_features(ALL_GOOD)
    b = mechanism_features(ALL_GOOD)
    pd.testing.assert_frame_equal(a, b)


def test_order_preserved() -> None:
    """Row i must describe smiles i, whatever order the inputs arrive in."""
    forward = mechanism_features(ALL_GOOD)
    reverse = mechanism_features(list(reversed(ALL_GOOD)))
    pd.testing.assert_frame_equal(
        forward.reset_index(drop=True),
        reverse.iloc[::-1].reset_index(drop=True),
    )
    # and a single-molecule call must equal the corresponding row of a batch
    for i, smi in enumerate(ALL_GOOD):
        single = mechanism_features([smi]).iloc[0]
        pd.testing.assert_series_equal(single, forward.iloc[i], check_names=False)


def test_duplicate_inputs_give_identical_rows() -> None:
    frame = mechanism_features([TODGA, TBP, TODGA])
    pd.testing.assert_series_equal(frame.iloc[0], frame.iloc[2], check_names=False)


# --------------------------------------------------------------------------
# hand-checked chemistry
# --------------------------------------------------------------------------
def test_todga_hand_checked() -> None:
    """TODGA: a diglycolamide -- two amide carbonyls, one central ether O."""
    row = mechanism_features([TODGA]).iloc[0]
    assert row["mech__n_amide_O"] == 2
    assert row["mech__n_ether_O"] == 1
    assert row["mech__n_S_donor"] == 0
    assert row["mech__n_P"] == 0
    assert row["mech__n_O_donor"] == 3
    assert row["mech__n_donor_total"] == 3
    assert row["mech__frac_O_donor"] == 1.0
    # the two amide N are delocalised and must NOT be counted as donors
    assert row["mech__n_amine_N"] == 0
    assert row["mech__n_N_donor"] == 0
    # neutral solvating extractant: no exchangeable proton, no ionic site
    assert row["mech__n_acidic_H"] == 0
    assert row["mech__is_neutral_extractant"] == 1
    # all three donors chelate; the amide O pair sits 4 bonds apart (O-C-C-O-C-C-O
    # gives min 3 between an amide O and the ether O)
    assert row["mech__denticity_proxy"] == 3
    assert row["mech__donor_dist_min"] == 3


def test_phosphorus_chemistry() -> None:
    """P=O vs P=S is the axis the existing 13-column donor census cannot see."""
    tbp = mechanism_features([TBP]).iloc[0]
    assert tbp["mech__n_phosphoryl_O"] == 1
    assert tbp["mech__n_thiophosphoryl_S"] == 0
    assert tbp["mech__n_P"] == 1

    cyanex = mechanism_features([CYANEX301]).iloc[0]
    assert cyanex["mech__n_phosphoryl_O"] == 0
    assert cyanex["mech__n_thiophosphoryl_S"] == 1
    assert cyanex["mech__n_P"] == 1
    # soft donors: the softness index must order them apart
    assert cyanex["mech__softness_mean"] > tbp["mech__softness_mean"]


def test_charge_class_separates_solvating_from_cation_exchange() -> None:
    hdehp = mechanism_features([HDEHP]).iloc[0]
    assert hdehp["mech__n_acidic_H"] == 1          # the P-OH proton
    assert hdehp["mech__is_neutral_extractant"] == 0
    assert hdehp["mech__n_phosphoryl_O"] == 1

    aliquat = mechanism_features([ALIQUAT]).iloc[0]
    assert aliquat["mech__is_neutral_extractant"] == 0   # quaternary ammonium salt
    assert aliquat["mech__n_donor_total"] == 0           # N+ has no lone pair

    tbp = mechanism_features([TBP]).iloc[0]
    assert tbp["mech__is_neutral_extractant"] == 1


def test_softness_ordering_across_donor_families() -> None:
    frame = mechanism_features([HDEHP, TODGA, BTBP, CYANEX301])
    soft = frame["mech__softness_mean"].to_numpy()
    # acidic phosphate O < amide/ether O < aromatic N < thiophosphoryl S
    assert soft[0] < soft[1] < soft[2] < soft[3]


def test_btbp_aromatic_nitrogen_and_rigidity() -> None:
    row = mechanism_features([BTBP]).iloc[0]
    assert row["mech__n_aromatic_N"] == 8
    assert row["mech__n_N_donor"] == 8
    assert row["mech__n_O_donor"] == 0
    assert row["mech__frac_N_donor"] == 1.0
    assert row["mech__ring_count"] == 4
    # a preorganised polyaromatic is more rigid than a floppy diglycolamide
    assert row["mech__frac_rotatable"] < mechanism_features([TODGA]).iloc[0]["mech__frac_rotatable"]


def test_cmpo_bidentate_spacing() -> None:
    """CMPO chelates through P=O and the amide O across a 5-membered ring."""
    row = mechanism_features([CMPO]).iloc[0]
    assert row["mech__n_phosphoryl_O"] == 1
    assert row["mech__n_amide_O"] == 1
    assert row["mech__denticity_proxy"] == 2
    assert row["mech__n_chelate_pairs"] == 1
    assert row["mech__donor_dist_min"] == 4


def test_pyrrole_type_nitrogen_is_not_a_donor() -> None:
    """Regression: only a pyridine-type aromatic N donates.

    A pyrrole-type N puts its lone pair in the aromatic sextet and cannot
    coordinate -- and it is pyrrole-type whether the third connection is an H
    (pyrrole, NH-azoles) or a carbon (N-alkyl pyrazole/triazole, carbazole).
    An earlier version keyed only on the H and so counted every N-substituted
    azole N as a donor; six cohort extractants were affected.
    """
    cases = {
        "c1ccncc1": 1,                  # pyridine: the one real donor
        "c1cc[nH]c1": 0,                # pyrrole NH
        "Cn1cccc1": 0,                  # N-methylpyrrole -- no H, still no lone pair
        "Cn1cccn1": 1,                  # N-methylpyrazole: N1 no, N2 yes
        "CCn1cc(C)nn1": 2,              # N-alkyl 1,2,3-triazole: N1 no, N2/N3 yes
        "CCn1c2ccccc2c2ccccc21": 0,     # N-alkyl carbazole
        "c1c[nH]cn1": 1,                # imidazole: NH no, =N- yes
    }
    frame = mechanism_features(list(cases))
    for i, (smi, expected) in enumerate(cases.items()):
        assert frame.iloc[i]["mech__n_aromatic_N"] == expected, smi
        assert frame.iloc[i]["mech__n_N_donor"] == expected, smi
    # and the motif this must NOT break: BTBP's eight pyridine/triazine N
    assert mechanism_features([BTBP]).iloc[0]["mech__n_aromatic_N"] == 8


def test_ester_class_fires_on_a_real_ester() -> None:
    """``mech__n_ester_O`` is zero across the cohort; prove that is a true
    null (no ester chemotype present) and not a dead classifier."""
    row = mechanism_features(["CCOC(C)=O"]).iloc[0]        # ethyl acetate
    assert row["mech__n_ester_O"] == 1
    assert row["mech__n_ether_O"] == 1                     # the alkoxy O
    assert row["mech__n_carboxyl_O"] == 0
    # a real carboxylic acid must go to carboxyl_O, not ester_O
    acid = mechanism_features(["CC(=O)O"]).iloc[0]
    assert acid["mech__n_ester_O"] == 0
    assert acid["mech__n_carboxyl_O"] == 2
    assert acid["mech__n_acidic_H"] == 1


def test_rdkit_log_suppression_is_scoped_not_global(capfd) -> None:
    """Suppressing RDKit's log must be scoped to the parse loop.

    A module-level ``RDLogger.DisableLog`` silences parse and sanitisation
    warnings for every *other* module in the process as a side effect of
    importing this one.  Inside the featuriser the log is quiet (we report a
    count instead); outside it RDKit must still complain on stderr.
    """
    from rdkit import Chem as _Chem

    frame = mechanism_features([GARBAGE, "C((("])
    assert n_unparsed(frame) == 2
    assert "SMILES Parse Error" not in capfd.readouterr().err  # quiet inside

    _Chem.MolFromSmiles("C(((")                                # loud outside
    assert "SMILES Parse Error" in capfd.readouterr().err


def test_never_touches_the_target() -> None:
    """The featuriser is a pure function of SMILES: no experimental input."""
    frame = mechanism_features(ALL_GOOD)
    assert not any("log_d" in c.lower() or "logd" in c.lower() for c in frame.columns)


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [GARBAGE, "", "C(((", None])
def test_unparseable_smiles_gives_nan_row(bad) -> None:
    frame = mechanism_features([TODGA, bad, TBP])
    assert frame.iloc[1].isna().all()
    assert n_unparsed(frame) == 1
    assert frame.attrs["n_unparsed"] == 1
    # the good rows are unaffected
    assert frame.iloc[0].notna().all()
    assert frame.iloc[2].notna().all()
    assert frame.iloc[0]["mech__n_amide_O"] == 2


def test_all_good_reports_zero_unparsed() -> None:
    assert n_unparsed(mechanism_features(ALL_GOOD)) == 0


# --------------------------------------------------------------------------
# distance
# --------------------------------------------------------------------------
def _standardise(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > 0, std, 1.0)
    return (values - mean) / std


def test_distance_shape_and_metric_properties() -> None:
    matrix = _standardise(mechanism_features(ALL_GOOD))
    dist = mechanism_distance(matrix, matrix)
    assert dist.shape == (len(ALL_GOOD), len(ALL_GOOD))
    assert np.allclose(np.diag(dist), 0.0, atol=1e-9)
    assert np.allclose(dist, dist.T, atol=1e-9)
    assert (dist >= -1e-12).all()
    # triangle inequality on every triple
    n = dist.shape[0]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert dist[i, j] <= dist[i, k] + dist[k, j] + 1e-9


def test_distance_matches_euclidean_when_complete() -> None:
    matrix = _standardise(mechanism_features(ALL_GOOD))
    expected = np.sqrt(((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(-1))
    assert np.allclose(mechanism_distance(matrix, matrix), expected, atol=1e-9)


def test_distance_rectangular_and_deterministic() -> None:
    a = _standardise(mechanism_features(ALL_GOOD))
    b = a[:3]
    dist = mechanism_distance(a, b)
    assert dist.shape == (len(ALL_GOOD), 3)
    assert np.allclose(dist, mechanism_distance(a, b))


def test_distance_propagates_nan_rows() -> None:
    frame = mechanism_features([TODGA, GARBAGE, TBP])
    matrix = _standardise(frame)
    dist = mechanism_distance(matrix, matrix)
    assert np.isnan(dist[1]).all()
    assert np.isnan(dist[:, 1]).all()
    assert np.isfinite(dist[0, 2])


def test_distance_dimension_mismatch_raises() -> None:
    a = np.zeros((2, 5))
    b = np.zeros((3, 4))
    with pytest.raises(ValueError):
        mechanism_distance(a, b)
