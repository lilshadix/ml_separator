from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from lanthanide_separation.feature_registry import (
    ABLATION_FAMILIES,
    FEATURE_FAMILIES,
    LOCAL_3D_SUBBLOCKS,
    build_feature_registry,
    write_feature_registry_json,
)
from lanthanide_separation.pairs import PairDataset


CONDITION = "base__cond__temperature_C"
LN = "pair__Z_A"
TWO_D_SCALAR = "base__MolWt"
TWO_D_ECFP = "base__ecfp_17"
GLOBAL_3D = "delta3d__feat3d__global_geometry__radius_of_gyration"
ELECTRONIC_3D = "delta3d__feat3d__complex_physical__dipole_magnitude"
DISTANCE_3D = "delta3d__feat3d__complex_physical__ln_donor_distance_mean"
ANGLE_3D = "delta3d__feat3d__derived_invariant__donor_angle_mean_deg"
COORDINATION_3D = "delta3d__feat3d__complex_physical__coordination_number"
SHAPE_3D = "delta3d__feat3d__ligand_field__field_shape_k2"
STERIC_3D = "delta3d__feat3d__enclosure__buried_fraction_r3p5"
COORDINATION_DISTORTION_3D = (
    "delta3d__feat3d__coordination_shape__radial_distortion_coefficient"
)
COORDINATION_VOLUME_3D = (
    "delta3d__feat3d__coordination_shape__coordination_polyhedron_volume"
)


def pair_dataset(
    *,
    baseline_columns: tuple[str, ...] = (
        LN,
        CONDITION,
        TWO_D_SCALAR,
        TWO_D_ECFP,
    ),
    delta3d_columns: tuple[str, ...] = (
        ELECTRONIC_3D,
        GLOBAL_3D,
        DISTANCE_3D,
        ANGLE_3D,
        COORDINATION_3D,
    ),
    descriptor_columns: tuple[str, ...] = (SHAPE_3D, STERIC_3D),
) -> PairDataset:
    columns = baseline_columns + delta3d_columns + descriptor_columns
    return PairDataset(
        frame=pd.DataFrame({column: [0.0] for column in set(columns)}),
        baseline_columns=baseline_columns,
        delta3d_columns=delta3d_columns,
        descriptor_columns=descriptor_columns,
        audit={"pair_scope": "all"},
        quarantine=pd.DataFrame(),
    )


class FeatureRegistryTests(unittest.TestCase):
    def test_every_feature_has_one_family_and_local_subblock(self) -> None:
        registry = build_feature_registry(pair_dataset())

        self.assertEqual(tuple(registry.families), FEATURE_FAMILIES)
        self.assertEqual(registry.families["CONDITIONS"], (CONDITION,))
        self.assertEqual(registry.families["LN"], (LN,))
        self.assertEqual(registry.families["2D"], (TWO_D_SCALAR, TWO_D_ECFP))
        self.assertEqual(registry.families["3D_GLOBAL"], (GLOBAL_3D,))
        self.assertEqual(
            registry.excluded_non_geometric_columns, (ELECTRONIC_3D,)
        )
        self.assertEqual(
            registry.families["3D_LOCAL"],
            (DISTANCE_3D, ANGLE_3D, COORDINATION_3D, SHAPE_3D, STERIC_3D),
        )
        self.assertEqual(tuple(registry.local_3d_subblocks), LOCAL_3D_SUBBLOCKS)
        self.assertEqual(registry.local_3d_subblocks["D1"], (DISTANCE_3D,))
        self.assertEqual(registry.local_3d_subblocks["D2"], (ANGLE_3D,))
        self.assertEqual(registry.local_3d_subblocks["D3"], (COORDINATION_3D,))
        self.assertEqual(registry.local_3d_subblocks["D4"], (SHAPE_3D,))
        self.assertEqual(registry.local_3d_subblocks["D5"], (STERIC_3D,))

        assigned = [
            column for columns in registry.families.values() for column in columns
        ]
        self.assertCountEqual(assigned, registry.columns)
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_main_and_descriptor_block_feature_sets_are_deterministic(self) -> None:
        registry = build_feature_registry(pair_dataset())

        self.assertEqual(tuple(registry.feature_sets()), tuple(ABLATION_FAMILIES))
        self.assertEqual(registry.ablation_columns("A0"), (CONDITION,))
        self.assertEqual(registry.ablation_columns("A1"), (LN, CONDITION))
        self.assertEqual(
            registry.ablation_columns("A2"),
            (LN, CONDITION, TWO_D_SCALAR, TWO_D_ECFP),
        )
        self.assertEqual(
            registry.ablation_columns("A5"),
            (
                LN,
                CONDITION,
                TWO_D_SCALAR,
                TWO_D_ECFP,
                DISTANCE_3D,
                ANGLE_3D,
                COORDINATION_3D,
                SHAPE_3D,
                STERIC_3D,
            ),
        )
        self.assertEqual(registry.ablation_columns("A6"), registry.columns)
        self.assertEqual(
            registry.local_3d_block_feature_sets()["A2+D4"],
            (LN, CONDITION, TWO_D_SCALAR, TWO_D_ECFP, SHAPE_3D),
        )
        self.assertEqual(
            registry.local_3d_block_feature_sets()["A2+D1-D5"],
            registry.ablation_columns("A5"),
        )

    def test_json_payload_is_complete_and_writer_is_deterministic(self) -> None:
        registry = build_feature_registry(pair_dataset())
        payload = registry.to_dict()

        self.assertEqual(payload["pair_scope"], "all")
        self.assertEqual(payload["feature_contract"]["column_count"], 10)
        self.assertEqual(payload["feature_contract"]["source_column_count"], 11)
        self.assertEqual(
            payload["feature_contract"]["excluded_non_geometric_column_count"], 1
        )
        self.assertEqual(len(payload["feature_contract"]["sha256"]), 64)
        self.assertTrue(payload["audit"]["all_model_features_assigned_exactly_once"])
        self.assertTrue(payload["audit"]["local_3d_features_assigned_exactly_once"])
        self.assertTrue(payload["audit"]["all_source_columns_accounted_for"])
        self.assertEqual(payload["families"]["3D_LOCAL"]["column_count"], 5)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "feature_registry.json"
            write_feature_registry_json(registry, path)
            first = path.read_bytes()
            write_feature_registry_json(registry, path)
            second = path.read_bytes()
            self.assertEqual(first, second)
            self.assertEqual(json.loads(first), payload)

    def test_unknown_or_forbidden_columns_fail_closed(self) -> None:
        unknown = pair_dataset(baseline_columns=("base__mystery_descriptor",))
        with self.assertRaisesRegex(ValueError, "no explicit feature-family"):
            build_feature_registry(unknown)

        forbidden = pair_dataset(baseline_columns=("base__log_D_extractant_mean",))
        with self.assertRaisesRegex(ValueError, "Forbidden target/identifier"):
            build_feature_registry(forbidden)

        unknown_local = pair_dataset(
            baseline_columns=(),
            delta3d_columns=(
                "delta3d__feat3d__complex_physical__unreviewed_quantity",
            ),
            descriptor_columns=(),
        )
        with self.assertRaisesRegex(ValueError, "no explicit feature-family"):
            build_feature_registry(unknown_local)

    def test_duplicate_or_missing_contract_columns_fail_closed(self) -> None:
        duplicate = pair_dataset(
            baseline_columns=(LN, LN),
            delta3d_columns=(),
            descriptor_columns=(),
        )
        with self.assertRaisesRegex(ValueError, "duplicate columns"):
            build_feature_registry(duplicate)

        missing = PairDataset(
            frame=pd.DataFrame({LN: [57.0]}),
            baseline_columns=(LN, CONDITION),
            delta3d_columns=(),
            descriptor_columns=(),
            audit={"pair_scope": "all"},
            quarantine=pd.DataFrame(),
        )
        with self.assertRaisesRegex(ValueError, "absent from its frame"):
            build_feature_registry(missing)

    def test_all_ranked_and_future_global_names_have_explicit_assignments(self) -> None:
        global_energy = "delta3d__feat3d__complex_physical__binding_energy_eV"
        global_shape = "delta3d__feat3d__global_geometry__radius_of_gyration"
        raw_distance = "delta3d__feat3d__polyhedron_scalars__next_donor_dist"
        raw_angle = "delta3d__feat3d__polyhedron__donor_angle_01_02_deg"
        donor_identity = "delta3d__feat3d__polyhedron__donor_atomic_number_01"
        dataset = pair_dataset(
            baseline_columns=(),
            delta3d_columns=(
                global_energy,
                global_shape,
                raw_distance,
                raw_angle,
                donor_identity,
            ),
            descriptor_columns=(),
        )

        registry = build_feature_registry(dataset)

        self.assertEqual(registry.columns_for_family("3D_GLOBAL"), (global_shape,))
        self.assertEqual(registry.excluded_non_geometric_columns, (global_energy,))
        self.assertEqual(registry.columns_for_local_3d_subblock("D1"), (raw_distance,))
        self.assertEqual(registry.columns_for_local_3d_subblock("D2"), (raw_angle,))
        self.assertEqual(
            registry.columns_for_local_3d_subblock("D3"), (donor_identity,)
        )

    def test_exact_coordination_shape_namespace_populates_d4_and_d5(self) -> None:
        dataset = pair_dataset(
            baseline_columns=(),
            delta3d_columns=(),
            descriptor_columns=(COORDINATION_DISTORTION_3D, COORDINATION_VOLUME_3D),
        )
        registry = build_feature_registry(dataset)

        self.assertEqual(
            registry.columns_for_local_3d_subblock("D4"),
            (COORDINATION_DISTORTION_3D,),
        )
        self.assertEqual(
            registry.columns_for_local_3d_subblock("D5"),
            (COORDINATION_VOLUME_3D,),
        )
        self.assertEqual(
            registry.columns_for_family("3D_LOCAL"),
            (COORDINATION_DISTORTION_3D, COORDINATION_VOLUME_3D),
        )

        unknown_shape = pair_dataset(
            baseline_columns=(),
            delta3d_columns=(),
            descriptor_columns=(
                "delta3d__feat3d__coordination_shape__unreviewed_metric",
            ),
        )
        with self.assertRaisesRegex(ValueError, "no explicit feature-family"):
            build_feature_registry(unknown_shape)


if __name__ == "__main__":
    unittest.main()
