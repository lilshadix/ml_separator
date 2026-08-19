"""Tests for the frozen all-190 chemistry map.

The two properties that matter: the map never touches the target, and the frozen
partition restricted to a cohort equals that cohort's own partition (otherwise
freezing it would silently move historical fold boundaries).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lanthanide_separation.gen6.chemistry import (
    ChemistryMap, DONOR_TYPE_VOCAB, build_chemistry_map, cluster_manifest, donor_census,
    novelty_of_cohort, partition_stability, tanimoto_matrix,
)
from lanthanide_separation.levels import ecfp_cluster_labels, tanimoto_cluster_labels


def _source(n_extractants: int = 6, n_bits: int = 32, seed: int = 3) -> pd.DataFrame:
    """A synthetic source table with the columns the map reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_extractants):
        bits = np.zeros(n_bits, dtype=int)
        bits[rng.choice(n_bits, size=8, replace=False)] = 1
        if i % 2 == 1:                       # make odd ligands near-copies of the previous one
            previous = rows[-1]
            bits = np.array([previous[f"ecfp_{b}"] for b in range(n_bits)])
            flip = rng.choice(n_bits, size=1)
            bits[flip] = 1 - bits[flip]
        for j in range(4):
            row = {
                "canonical_smiles": f"SMILES{i}",
                "extractant_name": f"L{i}",
                "metal_symbol": ["Nd", "Eu"][j % 2],
                "cond__acid_concentration_M": float(j),
                "cond__diluent__kerosene": 1.0,
                "log_D": float(rng.normal()),
                "DONOR_TYPES": json.dumps(["O(amide_carbonyl)", "O(ether)", "O(amide_carbonyl)"]),
                "DENTATE": 3, "coreCN": 9, "n_ligs": 3, "n_fill": 0,
                "MolWt": 400.0 + i, "TPSA": 50.0, "NumHDonors": 0.0, "NumHAcceptors": 4.0,
                "NumRotatableBonds": 12.0, "NumAromaticRings": 0.0, "NumAliphaticRings": 0.0,
                "RingCount": 0.0, "FractionCSP3": 0.9, "MolLogP": 6.0,
            }
            row.update({f"ecfp_{b}": int(bits[b]) for b in range(n_bits)})
            rows.append(row)
    return pd.DataFrame(rows)


def _descriptors(n_extractants: int = 6) -> pd.DataFrame:
    """Motif columns of the kind frozen in ligand_2d_descriptors.parquet."""
    records = []
    for i in range(n_extractants):
        records.append({
            "canonical_smiles": f"SMILES{i}",
            "lig2d__hc__n_dga_motifs": 1.0 if i < 2 else 0.0,
            "lig2d__hc__n_malonamide_motifs": 1.0 if i == 2 else 0.0,
            "lig2d__hc__n_phosphoryl_O": 1.0 if i == 3 else 0.0,
            "lig2d__hc__n_aromatic_N": 3.0 if i == 4 else 0.0,
            "lig2d__hc__n_ether_O": 2.0 if i == 5 else 0.0,
            "lig2d__hc__n_amide_N": 2.0 if i < 3 else 0.0,
            "lig2d__hc__n_carbonyl_O": 2.0 if i < 3 else 0.0,
            "lig2d__hc__n_hydroxyl_O": 0.0,
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

def test_tanimoto_matrix_matches_the_definition():
    bits = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]])
    similarity = tanimoto_matrix(bits)
    assert similarity[0, 0] == pytest.approx(1.0)
    assert similarity[0, 1] == pytest.approx(1 / 3)      # |∩| = 1, |∪| = 3
    assert similarity[0, 2] == pytest.approx(0.0)        # empty fingerprint
    assert similarity[2, 2] == pytest.approx(1.0)        # diagonal forced to 1
    np.testing.assert_allclose(similarity, similarity.T)


def test_tanimoto_matrix_is_symmetric_and_bounded_on_random_bits():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(12, 64))
    similarity = tanimoto_matrix(bits)
    np.testing.assert_allclose(similarity, similarity.T)
    assert similarity.min() >= 0.0 and similarity.max() <= 1.0


# --------------------------------------------------------------------------- #
# Target independence
# --------------------------------------------------------------------------- #

def test_map_is_independent_of_the_target():
    source = _source()
    blinded = source.assign(log_D=np.nan)
    a = build_chemistry_map(source, ligand_descriptors=_descriptors())
    b = build_chemistry_map(blinded, ligand_descriptors=_descriptors())
    pd.testing.assert_frame_equal(a.table, b.table)
    np.testing.assert_array_equal(a.similarity, b.similarity)


def test_map_is_independent_of_row_order():
    source = _source()
    shuffled = source.sample(frac=1.0, random_state=1).reset_index(drop=True)
    a = build_chemistry_map(source, ligand_descriptors=_descriptors())
    b = build_chemistry_map(shuffled, ligand_descriptors=_descriptors())
    pd.testing.assert_frame_equal(a.table, b.table)
    np.testing.assert_allclose(a.similarity, b.similarity)


def test_map_requires_identity_and_fingerprints():
    with pytest.raises(KeyError, match="canonical_smiles"):
        build_chemistry_map(pd.DataFrame({"x": [1]}))
    with pytest.raises(KeyError, match="fingerprint"):
        build_chemistry_map(pd.DataFrame({"canonical_smiles": ["A"]}))


# --------------------------------------------------------------------------- #
# Clusters
# --------------------------------------------------------------------------- #

def test_frozen_labels_reproduce_the_harness_partition():
    """The map must induce the same partition as levels.* on the same frame."""
    source = _source(n_extractants=8)
    fingerprints = [c for c in source.columns if c.startswith("ecfp_")]
    harness = source.assign(
        extractant=source["canonical_smiles"],
        ecfp_cluster=ecfp_cluster_labels(source, fingerprints),
        tanimoto_cluster=tanimoto_cluster_labels(source, fingerprints),
    )
    chemistry = build_chemistry_map(source)
    report = partition_stability(chemistry, harness)
    assert report["ok"] is True
    assert report["ecfp"]["identical_partition"] is True
    assert report["supercluster"]["identical_partition"] is True


def test_partition_stability_detects_a_disagreement():
    source = _source(n_extractants=6)
    chemistry = build_chemistry_map(source)
    harness = source.assign(extractant=source["canonical_smiles"],
                            ecfp_cluster="everything-in-one",
                            tanimoto_cluster="everything-in-one")
    report = partition_stability(chemistry, harness)
    assert report["ok"] is False
    assert report["ecfp"]["local_groups_spanning_multiple_frozen"] >= 1


def test_supercluster_sizes_and_nearest_neighbour_columns():
    chemistry = build_chemistry_map(_source(n_extractants=6))
    table = chemistry.table
    assert (table["chem__supercluster_size"] >= 1).all()
    assert table["chem__nn_within_all_tanimoto"].between(0.0, 1.0).all()
    # a ligand is never its own nearest neighbour
    assert (table["chem__nn_within_all_partner"] != table["extractant"]).all()
    # the near-copies built by the fixture find each other
    assert table["chem__nn_within_all_tanimoto"].max() > 0.8


# --------------------------------------------------------------------------- #
# Donor census and families
# --------------------------------------------------------------------------- #

def test_donor_census_counts_repeated_types():
    census = donor_census(json.dumps(["O(amide_carbonyl)", "O(ether)", "O(amide_carbonyl)"]))
    assert census["chem__donor__O(amide_carbonyl)"] == 2.0
    assert census["chem__donor__O(ether)"] == 1.0
    assert census["chem__donor__n_total"] == 3.0
    assert set(census) == {f"chem__donor__{t}" for t in DONOR_TYPE_VOCAB} | {"chem__donor__n_total"}


def test_donor_census_survives_missing_and_malformed_input():
    for bad in (None, float("nan"), "not json", 17):
        census = donor_census(bad)
        assert np.isnan(census["chem__donor__n_total"])


def test_families_come_from_the_frozen_motif_columns_when_available():
    chemistry = build_chemistry_map(_source(), ligand_descriptors=_descriptors())
    families = chemistry.table.set_index("extractant")["chem__family"].to_dict()
    assert families["SMILES0"] == "diglycolamide"
    assert families["SMILES2"] == "malonamide"
    assert families["SMILES3"] == "phosphoryl"
    assert families["SMILES4"] == "n_heterocyclic_polydentate"
    assert families["SMILES5"] == "podand_ether"
    assert set(chemistry.table["chem__family_source"]) == {"lig2d_hc_motifs"}


def test_family_source_is_recorded_when_motifs_are_absent():
    chemistry = build_chemistry_map(_source())
    assert set(chemistry.table["chem__family_source"]) <= {
        "smiles_heuristic", "donor_census", "donor_census_fallback"}
    assert chemistry.audit["family_source_counts"]


# --------------------------------------------------------------------------- #
# Nearest neighbour against a reference cohort
# --------------------------------------------------------------------------- #

def test_nearest_neighbour_against_a_reference():
    chemistry = build_chemistry_map(_source(n_extractants=6))
    reference = ["SMILES0", "SMILES1"]
    query = ["SMILES2", "SMILES3"]
    table = chemistry.nearest_neighbour(query, reference)
    assert list(table["extractant"]) == query
    assert (table["n_reference"] == 2).all()
    assert table["nn_tanimoto"].between(0.0, 1.0).all()
    assert table["nn_partner"].isin(reference).all()
    for column in ("n_above_0_5", "n_above_0_7", "n_above_0_8", "supercluster_support"):
        assert column in table.columns


def test_empty_reference_gives_zero_not_nan():
    """'Nothing similar in training' must be a number, or the hardest ligands vanish."""
    chemistry = build_chemistry_map(_source(n_extractants=4))
    table = chemistry.nearest_neighbour(["SMILES0"], [])
    assert table.loc[0, "nn_tanimoto"] == 0.0
    assert table.loc[0, "n_above_0_5"] == 0
    assert table.loc[0, "supercluster_support"] == 0
    assert table["nn_tanimoto"].notna().all()


def test_self_is_excluded_from_its_own_reference():
    chemistry = build_chemistry_map(_source(n_extractants=4))
    included = chemistry.nearest_neighbour(["SMILES0"], ["SMILES0"], exclude_self=True)
    assert included.loc[0, "nn_tanimoto"] == 0.0
    kept = chemistry.nearest_neighbour(["SMILES0"], ["SMILES0"], exclude_self=False)
    assert kept.loc[0, "nn_tanimoto"] == pytest.approx(1.0)


def test_novelty_of_cohort_wrapper_matches_nearest_neighbour():
    chemistry = build_chemistry_map(_source(n_extractants=5))
    a = novelty_of_cohort(chemistry, query=["SMILES1"], reference=["SMILES2", "SMILES3"])
    b = chemistry.nearest_neighbour(["SMILES1"], ["SMILES2", "SMILES3"])
    pd.testing.assert_frame_equal(a, b)


def test_index_of_rejects_unknown_extractants():
    chemistry = build_chemistry_map(_source(n_extractants=3))
    with pytest.raises(KeyError, match="unknown extractant"):
        chemistry.index_of("not-a-ligand")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def test_round_trip_through_parquet_restores_the_similarity_matrix(tmp_path):
    chemistry = build_chemistry_map(_source(), ligand_descriptors=_descriptors())
    path, sidecar = chemistry.to_parquet(tmp_path / "chemistry_map.parquet")
    assert path.is_file() and sidecar.is_file()
    restored = ChemistryMap.from_parquet(path)
    pd.testing.assert_frame_equal(restored.table, chemistry.table)
    np.testing.assert_allclose(restored.similarity, chemistry.similarity)
    assert restored.extractants == chemistry.extractants
    assert restored.audit["n_superclusters"] == chemistry.audit["n_superclusters"]


def test_from_parquet_fails_loudly_without_the_sidecar(tmp_path):
    chemistry = build_chemistry_map(_source())
    path, sidecar = chemistry.to_parquet(tmp_path / "chemistry_map.parquet")
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="sidecar"):
        ChemistryMap.from_parquet(path)


def test_similarity_npz_deliverable(tmp_path):
    chemistry = build_chemistry_map(_source())
    path = chemistry.save_similarity_npz(tmp_path / "nearest_neighbor_matrix.npz")
    with np.load(path, allow_pickle=True) as payload:
        np.testing.assert_allclose(payload["similarity"], chemistry.similarity)
        assert list(payload["extractants"]) == list(chemistry.extractants)


def test_cluster_manifest_lists_every_extractant_once():
    chemistry = build_chemistry_map(_source(n_extractants=6))
    manifest = cluster_manifest(chemistry)
    members = [e for group in manifest["superclusters"].values() for e in group]
    assert sorted(members) == sorted(chemistry.extractants)
    assert manifest["definition"]["target_independent"] is True
    assert manifest["audit"]["n_extractants"] == len(chemistry)


def test_canonical_labels_are_row_order_independent_and_dense():
    """A frozen artifact gets hashed, so its labels may not depend on parquet row order."""
    source = _source(n_extractants=8)
    shuffled = source.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = build_chemistry_map(source).table.set_index("extractant")
    b = build_chemistry_map(shuffled).table.set_index("extractant")
    assert a["chem__supercluster"].to_dict() == b["chem__supercluster"].to_dict()
    assert a["chem__ecfp_cluster"].to_dict() == b["chem__ecfp_cluster"].to_dict()
    labels = sorted(a["chem__supercluster"].unique())
    assert labels == [f"sc{i:03d}" for i in range(len(labels))]
    # the raw fingerprint hash is kept alongside, so the harness label is recoverable
    assert a["chem__ecfp_fingerprint_sha1"].str.len().eq(16).all()


def test_partition_stability_alone_cannot_prove_the_freeze_is_safe():
    """`build_level_dataset` labels chemotypes BEFORE filtering rows, so comparing
    against those labels is near-vacuous. `freeze_is_conservative` re-clusters."""
    from lanthanide_separation.gen6.chemistry import cohort_local_partition, freeze_is_conservative
    source = _source(n_extractants=8)
    chemistry = build_chemistry_map(source)
    subset = ["SMILES0", "SMILES1", "SMILES2", "SMILES3"]
    local = cohort_local_partition(source, subset)
    assert set(local["extractant"]) == set(subset)
    assert {"local_ecfp_cluster", "local_supercluster"} <= set(local.columns)

    report = freeze_is_conservative(chemistry, source, subset)
    # the only direction that would be dangerous is a frozen group SPLITTING a
    # local one — that would put related chemistry on both sides of a fold
    assert report["supercluster"]["local_groups_the_frozen_map_splits"] == 0
    assert report["ecfp"]["local_groups_the_frozen_map_splits"] == 0
    assert report["ok"] is True


def test_cohort_local_partition_rejects_an_empty_cohort():
    from lanthanide_separation.gen6.chemistry import cohort_local_partition
    with pytest.raises(ValueError, match="none of the requested"):
        cohort_local_partition(_source(), ["not-a-ligand"])


def test_shipped_similarity_matrix_rebuilds_the_maps_own_chemotypes(tmp_path):
    """The standalone .npz must reproduce the partition it describes.

    Two traps live here. float32 puts pairs at exactly Tanimoto 7/10 on the wrong
    side of the threshold, so the matrix is float64; and the linkage cut is
    `1.0 - threshold` (0.30000000000000004), not the literal 0.3, which would
    separate those same pairs.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    chemistry = build_chemistry_map(_source(n_extractants=10))
    assert chemistry.similarity.dtype == np.float64
    path = chemistry.save_similarity_npz(tmp_path / "nearest_neighbor_matrix.npz")
    with np.load(path, allow_pickle=True) as payload:
        similarity = payload["similarity"]
        cut = float(payload["linkage_distance_threshold"])
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(linkage(squareform(distance, checks=False), method="single"),
                      t=cut, criterion="distance")
    assert len(set(labels)) == chemistry.audit["n_superclusters"]


def test_extractant_name_is_the_most_common_not_the_first():
    """One SMILES can carry several trade names; taking the first row labelled the
    most-measured ligand in this dataset with a name from 6 of its 1,585 rows."""
    source = _source(n_extractants=2, n_bits=32)
    names = source["extractant_name"].to_numpy().copy()
    names[0] = "RARE"                       # first row of SMILES0 only
    source = source.assign(extractant_name=names)
    table = build_chemistry_map(source).table.set_index("extractant")
    assert table.loc["SMILES0", "extractant_name"] == "L0"      # the majority name
    assert table.loc["SMILES0", "n_extractant_names"] == 2
    assert "RARE" in table.loc["SMILES0", "extractant_name_alternatives"]
