"""Known-answer tests for the hand-crafted 2D ligand descriptors.

The descriptors are exact graph quantities, so each test states the chemistry
it checks and asserts the integer answer.  RDKit is only available in the
conda environment; the module skips cleanly elsewhere.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile

import pytest

pytest.importorskip("rdkit")

from lanthanide_separation.ligand_descriptors import (  # noqa: E402
    HAND_CRAFTED_DESCRIPTOR_NAMES,
    HAND_CRAFTED_PREFIX,
    RDKIT_PREFIX,
    SMILES_COLUMN,
    build_descriptor_table,
    hand_crafted_descriptors,
    load_descriptor_table,
    mol_from_smiles,
    rdkit_descriptor_row,
)


TODGA = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
METHYL_OCTYL_DGA = "CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC"
TEHDGA = "CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC"
PHEN_ARYL_AMIDE = "CCCCCCCCN(C(=O)c1ccc2ccc3cccnc3c2n1)c1ccc(C)cc1"
DIMETHYL_TODGA_CIS = "CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@@H](C)C(=O)N(CCCCCCCC)CCCCCCCC"
DIMETHYL_TODGA_TRANS = "CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@H](C)C(=O)N(CCCCCCCC)CCCCCCCC"
CROWN_18_6 = "C1COCCOCCOCCOCCOCCO1"


def _hc(smiles: str) -> dict[str, float]:
    mol = mol_from_smiles(smiles)
    assert mol is not None, smiles
    return hand_crafted_descriptors(mol)


def _assert_values(actual: dict[str, float], expected: dict[str, float]) -> None:
    for name, value in expected.items():
        assert name in actual, name
        assert math.isclose(actual[name], value, abs_tol=1e-9), (
            f"{name}: expected {value}, got {actual[name]}"
        )


def test_schema_is_fixed_and_none_gives_all_nan() -> None:
    values = hand_crafted_descriptors(None)
    assert tuple(values) == HAND_CRAFTED_DESCRIPTOR_NAMES
    assert all(math.isnan(value) for value in values.values())
    populated = _hc(TODGA)
    assert tuple(populated) == HAND_CRAFTED_DESCRIPTOR_NAMES


def test_todga_symmetric_tetraoctyl() -> None:
    _assert_values(
        _hc(TODGA),
        {
            "n_amide_N": 2,
            "n_thioamide_N": 0,
            "n_NH_amide": 0,
            "n_ether_O": 1,
            "n_carbonyl_O": 2,
            "n_amine_N": 0,
            "n_donor_atoms": 3,
            "n_N_substituents": 4,
            "chain_len_max": 8,
            "chain_len_min": 8,
            "chain_len_mean": 8,
            "chain_len_asymmetry": 0,
            "n_branched_substituents": 0,
            "n_methyl_on_N": 0,
            "n_aryl_on_N": 0,
            "n_C_in_N_substituents": 32,
            "subst_heavy_max": 8,
            "frac_heavy_in_largest_N_substituent": 8 / 41,
            "is_symmetric_amide": 1,
            "is_symmetric_across_N": 1,
            "n_substituted_alpha_carbons": 0,
            "n_stereocentres": 0,
            "longest_carbon_chain": 8,
            "n_dga_motifs": 1,
            "n_malonamide_motifs": 0,
            "amide_carbonyl_min_path": 4,
            "n_heavy_atoms": 41,
            "n_carbons": 36,
            "n_aromatic_atoms": 0,
        },
    )


def test_methyl_octyl_dga_is_unsymmetrical_per_substituent() -> None:
    values = _hc(METHYL_OCTYL_DGA)
    _assert_values(
        values,
        {
            "n_amide_N": 2,
            "n_ether_O": 1,
            "n_carbonyl_O": 2,
            "chain_len_max": 8,
            "chain_len_min": 1,
            "chain_len_mean": 4.5,
            "chain_len_asymmetry": 7,
            "n_methyl_on_N": 2,
            "n_branched_substituents": 0,
            "is_symmetric_amide": 0,
            # Both nitrogens carry the same {methyl, octyl} pair.
            "is_symmetric_across_N": 1,
            "n_C_in_N_substituents": 18,
            "longest_carbon_chain": 8,
        },
    )


def test_tehdga_beta_branching_and_stereocentres() -> None:
    values = _hc(TEHDGA)
    _assert_values(
        values,
        {
            "n_amide_N": 2,
            # 2-ethylhexyl: CH2-CH(Et)(Bu) -> longest chain from N is 6.
            "chain_len_max": 6,
            "chain_len_min": 6,
            "n_branched_substituents": 4,
            "n_alpha_branched_substituents": 0,
            "n_beta_branched_substituents": 4,
            "n_methyl_on_N": 0,
            "is_symmetric_amide": 1,
            "n_stereocentres": 4,
            "n_assigned_stereocentres": 0,
            "n_C_in_N_substituents": 32,
            # Ethyl-CH-butyl is the longest sp3 path in the molecule.
            "longest_carbon_chain": 7,
            "n_substituted_alpha_carbons": 0,
        },
    )
    assert math.isnan(values["assigned_cip_homochiral"])


def test_aryl_octyl_phenanthroline_amide() -> None:
    _assert_values(
        _hc(PHEN_ARYL_AMIDE),
        {
            "n_amide_N": 1,
            "n_N_substituents": 2,
            "n_aryl_on_N": 1,
            "n_methyl_on_N": 0,
            "chain_len_max": 8,
            "chain_len_min": 0,
            "chain_len_asymmetry": 8,
            "is_symmetric_amide": 0,
            "n_aromatic_N": 2,
            # Every non-amide nitrogen counts as a potential donor.
            "n_amine_N": 2,
            "n_carbonyl_O": 1,
            "n_ether_O": 0,
            "n_donor_atoms": 3,
            "n_aromatic_atoms": 20,
            "n_dga_motifs": 0,
        },
    )


def test_backbone_methyls_and_relative_configuration() -> None:
    cis = _hc(DIMETHYL_TODGA_CIS)
    trans = _hc(DIMETHYL_TODGA_TRANS)
    for values in (cis, trans):
        _assert_values(
            values,
            {
                "n_substituted_alpha_carbons": 2,
                "n_stereocentres": 2,
                "n_assigned_stereocentres": 2,
                "chain_len_max": 8,
                "chain_len_min": 8,
                "n_dga_motifs": 1,
            },
        )
    # The two diastereomers differ only in whether the CIP labels agree.
    assert {cis["assigned_cip_homochiral"], trans["assigned_cip_homochiral"]} == {0.0, 1.0}


def test_no_amide_ligand_has_nan_chain_statistics_and_ether_donors() -> None:
    values = _hc(CROWN_18_6)
    _assert_values(
        values,
        {
            "n_amide_N": 0,
            "n_N_substituents": 0,
            "n_ether_O": 6,
            "n_carbonyl_O": 0,
            "n_donor_atoms": 6,
            "n_methyl_on_N": 0,
            "n_branched_substituents": 0,
            "subst_heavy_max": 0,
            "frac_heavy_in_largest_N_substituent": 0,
            "longest_carbon_chain": 2,
        },
    )
    for name in ("chain_len_max", "chain_len_min", "chain_len_mean", "is_symmetric_amide"):
        assert math.isnan(values[name]), name


def test_rdkit_row_covers_desclist_and_is_finite_for_todga() -> None:
    from rdkit.Chem import Descriptors

    row = rdkit_descriptor_row(mol_from_smiles(TODGA))
    assert set(row) == {name for name, _ in Descriptors.descList}
    assert all(math.isfinite(value) for value in row.values())
    assert all(math.isnan(value) for value in rdkit_descriptor_row(None).values())


def test_build_and_load_table_roundtrip() -> None:
    smiles = [TODGA, METHYL_OCTYL_DGA, TODGA, "not a smiles", CROWN_18_6]
    frame = build_descriptor_table(smiles)
    # Unique, first-appearance order; the unparsable entry is kept as NaN.
    assert frame[SMILES_COLUMN].tolist() == [TODGA, METHYL_OCTYL_DGA, "not a smiles", CROWN_18_6]
    assert frame.attrs["unparsed_smiles"] == ["not a smiles"]
    hc_columns = [column for column in frame.columns if column.startswith(HAND_CRAFTED_PREFIX)]
    rd_columns = [column for column in frame.columns if column.startswith(RDKIT_PREFIX)]
    assert len(hc_columns) == len(HAND_CRAFTED_DESCRIPTOR_NAMES)
    assert rd_columns and frame.attrs["dropped_rdkit_columns"]
    assert set(frame.attrs["dropped_rdkit_columns"]).isdisjoint(frame.columns)
    assert frame.loc[2, hc_columns].isna().all()
    assert frame.loc[0, f"{HAND_CRAFTED_PREFIX}chain_len_max"] == 8

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ligand_2d_descriptors.parquet"
        frame.to_parquet(path, index=False)
        loaded = load_descriptor_table(path)
    assert loaded.shape == frame.shape
    assert loaded[SMILES_COLUMN].tolist() == frame[SMILES_COLUMN].tolist()
    assert loaded[f"{HAND_CRAFTED_PREFIX}n_methyl_on_N"].tolist()[1] == 2
