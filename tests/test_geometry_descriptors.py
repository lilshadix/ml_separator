from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from lanthanide_separation.geometry_descriptors import (
    CORE_DESCRIPTOR_NAMES,
    ENCLOSURE_PREFIX,
    LIGAND_FIELD_PREFIX,
    MetalSiteDescriptorBuilder,
    attach_geometry_descriptors,
    enclosure_descriptors,
    fibonacci_directions,
    ligand_field_descriptors,
    permute_descriptors,
    validate_blocks,
)


def octahedron(scale: float = 2.4) -> np.ndarray:
    return scale * np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )


def random_rotation(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1.0
    return matrix


def write_archive(path: Path) -> None:
    """Two single-metal complexes with six donors and a few shell atoms."""

    graphs = []
    for scale, metal_z in ((2.40, 58), (2.36, 59)):
        donors = octahedron(scale)
        shell = 4.5 * np.array([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.3]]) / np.sqrt(2.0)
        coordinates = np.vstack([np.zeros((1, 3)), donors, shell])
        atomic_numbers = np.array([metal_z] + [8] * 6 + [6, 6])
        is_metal = np.array([1] + [0] * 8, dtype=np.int8)
        is_donor = np.array([0] + [1] * 6 + [0, 0], dtype=np.int8)
        graphs.append((coordinates, atomic_numbers, is_metal, is_donor))

    node_ptr = np.cumsum([0] + [len(item[0]) for item in graphs])
    np.savez(
        path,
        coordinates=np.vstack([item[0] for item in graphs]).astype(np.float32),
        atomic_numbers=np.concatenate([item[1] for item in graphs]).astype(np.int16),
        is_metal=np.concatenate([item[2] for item in graphs]),
        is_coord_donor=np.concatenate([item[3] for item in graphs]),
        node_ptr=node_ptr.astype(np.int64),
        build_ids=np.array(["geom_a", "geom_b"]),
    )


class LigandFieldTests(unittest.TestCase):
    def test_invariants_are_unchanged_by_rotation_and_relabelling(self) -> None:
        donors = octahedron()
        reference = ligand_field_descriptors(donors)
        rotated = donors @ random_rotation(7).transpose()
        permuted = rotated[np.random.default_rng(3).permutation(len(rotated))]
        candidate = ligand_field_descriptors(permuted)
        for name, value in reference.items():
            self.assertAlmostEqual(value, candidate[name], places=8, msg=name)

    def test_octahedron_has_no_rank_two_field(self) -> None:
        # A regular octahedron is the textbook case with a vanishing rank-2
        # crystal-field term and non-vanishing rank-4 and rank-6 terms.
        features = ligand_field_descriptors(octahedron())
        self.assertAlmostEqual(features["field_strength_k2"], 0.0, places=10)
        self.assertGreater(features["field_strength_k4"], 0.0)
        self.assertGreater(features["field_strength_k6"], 0.0)

    def test_contracted_shell_raises_the_field_strength(self) -> None:
        loose = ligand_field_descriptors(octahedron(2.60))
        tight = ligand_field_descriptors(octahedron(2.40))
        # Shrinking the shell must strengthen every rank that the polyhedron
        # actually supports, and the purely angular shape terms must not move
        # because the polyhedron itself is unchanged.
        for degree in (4, 6):
            self.assertGreater(
                tight[f"field_strength_k{degree}"], loose[f"field_strength_k{degree}"]
            )
        for degree in (2, 4, 6):
            self.assertAlmostEqual(
                tight[f"field_shape_k{degree}"],
                loose[f"field_shape_k{degree}"],
                places=8,
            )
        self.assertLess(tight["effective_field_radius"], loose["effective_field_radius"])

    def test_distinct_polyhedra_are_distinguished(self) -> None:
        cube = np.array(
            [
                [x, y, z]
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ]
        )
        cube = 2.4 * cube / np.linalg.norm(cube[0])
        antiprism = []
        for index in range(4):
            angle = index * np.pi / 2.0
            antiprism.append([np.cos(angle), np.sin(angle), 0.7])
            antiprism.append(
                [np.cos(angle + np.pi / 4.0), np.sin(angle + np.pi / 4.0), -0.7]
            )
        antiprism = np.asarray(antiprism)
        antiprism = 2.4 * antiprism / np.linalg.norm(antiprism, axis=1, keepdims=True)
        cube_features = ligand_field_descriptors(cube)
        antiprism_features = ligand_field_descriptors(antiprism)
        self.assertNotAlmostEqual(
            cube_features["field_shape_k4"],
            antiprism_features["field_shape_k4"],
            places=3,
        )

    def test_degenerate_shells_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            ligand_field_descriptors(octahedron()[:2])
        collapsed = octahedron()
        collapsed[0] = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            ligand_field_descriptors(collapsed)


class EnclosureTests(unittest.TestCase):
    def test_directions_are_unit_vectors_summing_to_nearly_zero(self) -> None:
        directions = fibonacci_directions(512)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
        self.assertLess(float(np.linalg.norm(directions.mean(axis=0))), 1e-2)

    def test_a_denser_shell_buries_more_of_the_metal(self) -> None:
        directions = fibonacci_directions(512)
        sparse = np.vstack([np.zeros(3), octahedron(2.4)])
        # The second shell must point somewhere new: a collinear copy hides
        # behind the first one and blocks no additional solid angle.
        outer = octahedron(3.2) @ random_rotation(23).transpose()
        dense = np.vstack([np.zeros(3), octahedron(2.4), outer])
        numbers_sparse = np.array([58] + [8] * 6)
        numbers_dense = np.array([58] + [8] * 12)
        sparse_features = enclosure_descriptors(sparse, numbers_sparse, 0, directions)
        dense_features = enclosure_descriptors(dense, numbers_dense, 0, directions)
        self.assertGreater(
            dense_features["buried_fraction_r5p0"],
            sparse_features["buried_fraction_r5p0"],
        )
        self.assertLessEqual(dense_features["open_fraction"], sparse_features["open_fraction"])

    def test_enclosure_is_rotation_invariant(self) -> None:
        directions = fibonacci_directions(2048)
        coordinates = np.vstack([np.zeros(3), octahedron(2.4)])
        numbers = np.array([58] + [8] * 6)
        reference = enclosure_descriptors(coordinates, numbers, 0, directions)
        rotated = coordinates @ random_rotation(11).transpose()
        candidate = enclosure_descriptors(rotated, numbers, 0, directions)
        for name, value in reference.items():
            # Ray sampling is a fixed grid, so rotation invariance is numerical
            # rather than exact; the tolerance reflects the grid resolution.
            self.assertAlmostEqual(value, candidate[name], places=1, msg=name)


class BuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "vr.npz"
        write_archive(self.path)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_core_profile_emits_exactly_the_declared_columns(self) -> None:
        descriptors = MetalSiteDescriptorBuilder(self.path, profile="core").build()
        expected = [
            f"{LIGAND_FIELD_PREFIX}{name}" for name in CORE_DESCRIPTOR_NAMES["ligand_field"]
        ] + [f"{ENCLOSURE_PREFIX}{name}" for name in CORE_DESCRIPTOR_NAMES["enclosure"]]
        self.assertEqual(list(descriptors.columns), expected)
        self.assertEqual(len(descriptors.frame), 2)
        self.assertEqual(descriptors.audit["profile"], "core")

    def test_full_profile_is_a_strict_superset(self) -> None:
        core = MetalSiteDescriptorBuilder(self.path, profile="core").build()
        full = MetalSiteDescriptorBuilder(self.path, profile="full").build()
        self.assertTrue(set(core.columns) < set(full.columns))

    def test_single_block_selection(self) -> None:
        descriptors = MetalSiteDescriptorBuilder(
            self.path, blocks=("ligand_field",)
        ).build()
        self.assertTrue(
            all(name.startswith(LIGAND_FIELD_PREFIX) for name in descriptors.columns)
        )

    def test_unknown_block_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_blocks(("ligand_field", "not_a_block"))

    def test_permutation_preserves_marginals_and_is_recorded(self) -> None:
        descriptors = MetalSiteDescriptorBuilder(self.path).build()
        permuted = permute_descriptors(descriptors, mode="free", seed=5)
        columns = list(descriptors.columns)
        np.testing.assert_allclose(
            np.sort(descriptors.frame[columns].to_numpy(), axis=0),
            np.sort(permuted.frame[columns].to_numpy(), axis=0),
        )
        self.assertEqual(permuted.audit["descriptor_permutation_mode"], "free")
        self.assertEqual(permuted.audit["descriptor_permutation_seed"], 5)
        self.assertEqual(
            list(permuted.frame["geometry_feature_build_id"]),
            list(descriptors.frame["geometry_feature_build_id"]),
        )

    def test_metal_preserving_permutation_never_crosses_elements(self) -> None:
        descriptors = MetalSiteDescriptorBuilder(self.path).build()
        permuted = permute_descriptors(descriptors, mode="metal-preserving", seed=5)
        # The fixture has one geometry per element, so this permutation is the
        # identity; the point is that values never migrate across elements.
        pd.testing.assert_frame_equal(descriptors.frame, permuted.frame)

    def test_none_mode_returns_the_input_unchanged(self) -> None:
        descriptors = MetalSiteDescriptorBuilder(self.path).build()
        self.assertIs(permute_descriptors(descriptors, mode="none", seed=1), descriptors)


class AttachTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary.name) / "vr.npz"
        write_archive(self.path)
        self.descriptors = MetalSiteDescriptorBuilder(self.path).build()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def source(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "geometry_ok": [True, True, False],
                "geometry_feature_build_id": ["geom_a", "geom_b", None],
                "vr_graph_index": [0, 1, None],
            }
        )

    def test_rows_without_geometry_keep_missing_descriptors(self) -> None:
        merged, audit = attach_geometry_descriptors(self.source(), self.descriptors)
        self.assertEqual(audit["rows_with_descriptors"], 2)
        self.assertEqual(audit["rows_with_accepted_geometry"], 2)
        self.assertTrue(merged.loc[2, list(self.descriptors.columns)].isna().all())

    def test_index_drift_fails_closed(self) -> None:
        source = self.source()
        source.loc[1, "vr_graph_index"] = 0
        with self.assertRaises(ValueError):
            attach_geometry_descriptors(source, self.descriptors)

    def test_unknown_accepted_geometry_fails_closed(self) -> None:
        source = self.source()
        source.loc[1, "geometry_feature_build_id"] = "geom_missing"
        with self.assertRaises(ValueError):
            attach_geometry_descriptors(source, self.descriptors)


if __name__ == "__main__":
    unittest.main()
