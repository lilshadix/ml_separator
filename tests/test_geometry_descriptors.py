from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from lanthanide_separation.geometry_descriptors import (
    COORDINATION_SHAPE_PREFIX,
    CORE_DESCRIPTOR_NAMES,
    ENCLOSURE_PREFIX,
    GLOBAL_GEOMETRY_PREFIX,
    LIGAND_FIELD_PREFIX,
    MetalSiteDescriptorBuilder,
    attach_geometry_descriptors,
    coordination_shape_descriptors,
    enclosure_descriptors,
    fibonacci_directions,
    global_shape_descriptors,
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


class GlobalShapeTests(unittest.TestCase):
    def test_invariants_are_unchanged_by_translation_rotation_and_relabelling(
        self,
    ) -> None:
        coordinates = np.array(
            [
                [-1.2, 0.4, 0.1],
                [0.3, -0.8, 1.5],
                [1.7, 0.2, -0.4],
                [-0.1, 1.9, 0.7],
                [0.6, -1.1, -1.3],
                [-1.5, -0.6, 0.9],
            ]
        )
        reference = global_shape_descriptors(coordinates)
        permutation = np.random.default_rng(17).permutation(len(coordinates))
        transformed = (
            coordinates[permutation] @ random_rotation(13).transpose()
            + np.array([8.0, -3.5, 11.2])
        )
        candidate = global_shape_descriptors(transformed)
        self.assertEqual(set(reference), set(candidate))
        for name, value in reference.items():
            self.assertAlmostEqual(value, candidate[name], places=11, msg=name)

    def test_regular_octahedron_has_known_isotropic_shape_and_hull(self) -> None:
        scale = 2.4
        features = global_shape_descriptors(octahedron(scale))
        self.assertAlmostEqual(features["radius_of_gyration"], scale, places=12)
        for index in (1, 2, 3):
            self.assertAlmostEqual(
                features[f"principal_variance_{index}"], scale**2 / 3.0, places=12
            )
            self.assertAlmostEqual(
                features[f"principal_moment_of_inertia_{index}"],
                2.0 * scale**2 / 3.0,
                places=12,
            )
        self.assertAlmostEqual(features["normalized_asphericity"], 0.0, places=12)
        self.assertAlmostEqual(
            features["relative_shape_anisotropy"], 0.0, places=12
        )
        self.assertAlmostEqual(features["eccentricity"], 0.0, places=12)
        self.assertAlmostEqual(
            features["convex_hull_volume"], 4.0 * scale**3 / 3.0, places=11
        )
        self.assertAlmostEqual(
            features["convex_hull_surface_area"],
            4.0 * np.sqrt(3.0) * scale**2,
            places=11,
        )

    def test_linear_cloud_keeps_shape_metrics_and_marks_hull_unavailable(self) -> None:
        coordinates = np.array(
            [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        features = global_shape_descriptors(coordinates)
        self.assertAlmostEqual(features["normalized_asphericity"], 1.0, places=12)
        self.assertAlmostEqual(
            features["relative_shape_anisotropy"], 1.0, places=12
        )
        self.assertAlmostEqual(features["eccentricity"], 1.0, places=12)
        self.assertEqual(features["convex_hull_volume"], 0.0)
        self.assertEqual(features["convex_hull_surface_area"], 0.0)

    def test_invalid_or_collapsed_coordinates_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            global_shape_descriptors(np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            global_shape_descriptors(np.zeros((4, 3)))
        invalid = octahedron()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            global_shape_descriptors(invalid)


class CoordinationShapeTests(unittest.TestCase):
    def test_exact_invariants_survive_rigid_motion_and_donor_relabelling(
        self,
    ) -> None:
        metal = np.array([1.7, -2.1, 0.8])
        donor_positions = octahedron(2.4) + metal
        reference = coordination_shape_descriptors(donor_positions - metal)

        rotation = random_rotation(29)
        translation = np.array([-8.0, 3.5, 12.2])
        permutation = np.random.default_rng(31).permutation(len(donor_positions))
        transformed_metal = metal @ rotation.transpose() + translation
        transformed_donors = (
            donor_positions[permutation] @ rotation.transpose() + translation
        )
        candidate = coordination_shape_descriptors(
            transformed_donors - transformed_metal
        )

        self.assertEqual(set(reference), set(candidate))
        for name, value in reference.items():
            self.assertAlmostEqual(value, candidate[name], places=11, msg=name)

    def test_regular_octahedron_has_known_shape_and_polyhedron_measures(self) -> None:
        scale = 2.4
        features = coordination_shape_descriptors(octahedron(scale))
        for name in (
            "radial_distortion_coefficient",
            "radial_range_fraction",
            "metal_offset_from_donor_centroid_fraction",
            "coordination_normalized_asphericity",
            "coordination_relative_shape_anisotropy",
            "coordination_eccentricity",
            "directional_relative_shape_anisotropy",
            "directional_inversion_imbalance",
        ):
            self.assertAlmostEqual(features[name], 0.0, places=12, msg=name)
        self.assertAlmostEqual(
            features["coordination_polyhedron_volume"],
            4.0 * scale**3 / 3.0,
            places=11,
        )
        self.assertAlmostEqual(
            features["coordination_polyhedron_surface_area"],
            4.0 * np.sqrt(3.0) * scale**2,
            places=11,
        )

    def test_radial_and_off_center_distortions_are_detected(self) -> None:
        donors = octahedron(2.4)
        donors[0] *= 1.2
        donors += np.array([0.25, -0.05, 0.1])
        features = coordination_shape_descriptors(donors)
        self.assertGreater(features["radial_distortion_coefficient"], 0.0)
        self.assertGreater(features["radial_range_fraction"], 0.0)
        self.assertGreater(
            features["metal_offset_from_donor_centroid_fraction"], 0.0
        )
        self.assertGreater(features["directional_inversion_imbalance"], 0.0)

    def test_degenerate_hull_is_zero_but_invalid_shells_fail(self) -> None:
        square = 2.4 * np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        features = coordination_shape_descriptors(square)
        self.assertEqual(features["coordination_polyhedron_volume"], 0.0)
        self.assertEqual(features["coordination_polyhedron_surface_area"], 0.0)

        with self.assertRaises(ValueError):
            coordination_shape_descriptors(octahedron()[:2])
        collapsed = octahedron()
        collapsed[0] = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            coordination_shape_descriptors(collapsed)
        invalid = octahedron()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            coordination_shape_descriptors(invalid)


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
        ] + [
            f"{ENCLOSURE_PREFIX}{name}" for name in CORE_DESCRIPTOR_NAMES["enclosure"]
        ] + [
            f"{GLOBAL_GEOMETRY_PREFIX}{name}"
            for name in CORE_DESCRIPTOR_NAMES["global_shape"]
        ] + [
            f"{COORDINATION_SHAPE_PREFIX}{name}"
            for name in CORE_DESCRIPTOR_NAMES["coordination_shape"]
        ]
        self.assertEqual(list(descriptors.columns), expected)
        self.assertEqual(len(descriptors.frame), 2)
        self.assertEqual(descriptors.audit["profile"], "core")
        self.assertEqual(descriptors.audit["global_shape_hull_available_count"], 2)
        self.assertEqual(descriptors.audit["global_shape_hull_unavailable_count"], 0)
        self.assertEqual(
            descriptors.audit["global_shape_coordinate_selection"],
            "all atoms including metal",
        )
        self.assertEqual(
            descriptors.audit["coordination_shape_hull_available_count"], 2
        )
        self.assertEqual(
            descriptors.audit["coordination_shape_hull_unavailable_count"], 0
        )
        self.assertIsNone(
            descriptors.audit["coordination_shape_reference_geometry"]
        )
        self.assertIsNone(descriptors.audit["coordination_shape_sampling_grid"])

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

    def test_global_shape_block_has_explicit_namespace(self) -> None:
        descriptors = MetalSiteDescriptorBuilder(
            self.path, blocks=("global_shape",)
        ).build()
        self.assertEqual(
            list(descriptors.columns),
            [
                f"{GLOBAL_GEOMETRY_PREFIX}{name}"
                for name in CORE_DESCRIPTOR_NAMES["global_shape"]
            ],
        )

    def test_coordination_shape_block_has_explicit_namespace(self) -> None:
        descriptors = MetalSiteDescriptorBuilder(
            self.path, blocks=("coordination_shape",)
        ).build()
        self.assertEqual(
            list(descriptors.columns),
            [
                f"{COORDINATION_SHAPE_PREFIX}{name}"
                for name in CORE_DESCRIPTOR_NAMES["coordination_shape"]
            ],
        )
        self.assertEqual(
            descriptors.audit["coordination_shape_invariances"],
            ["translation", "orthogonal_transform", "donor_permutation"],
        )

    def test_degenerate_coordination_hull_is_identified_by_build_id(self) -> None:
        planar_path = Path(self._temporary.name) / "planar_vr.npz"
        donors = 2.4 * np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        )
        coordinates = np.vstack([np.zeros((1, 3)), donors])
        np.savez(
            planar_path,
            coordinates=coordinates.astype(np.float32),
            atomic_numbers=np.array([58, 8, 8, 8, 8], dtype=np.int16),
            is_metal=np.array([1, 0, 0, 0, 0], dtype=np.int8),
            is_coord_donor=np.array([0, 1, 1, 1, 1], dtype=np.int8),
            node_ptr=np.array([0, 5], dtype=np.int64),
            build_ids=np.array(["planar_geom"]),
        )

        descriptors = MetalSiteDescriptorBuilder(
            planar_path, blocks=("coordination_shape",)
        ).build()
        self.assertEqual(
            descriptors.frame[
                f"{COORDINATION_SHAPE_PREFIX}coordination_polyhedron_volume"
            ].iloc[0],
            0.0,
        )
        self.assertEqual(
            descriptors.frame[
                f"{COORDINATION_SHAPE_PREFIX}coordination_polyhedron_surface_area"
            ].iloc[0],
            0.0,
        )
        self.assertEqual(
            descriptors.audit["coordination_shape_hull_available_count"], 0
        )
        self.assertEqual(
            descriptors.audit["coordination_shape_hull_unavailable_count"], 1
        )
        self.assertEqual(
            descriptors.audit["coordination_shape_hull_unavailable_build_ids"],
            ["planar_geom"],
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
