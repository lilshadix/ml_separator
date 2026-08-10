"""Contract tests for the second-generation pair-response and electronic blocks.

The extension blocks are only admissible if they cannot break the guarantees the
frozen benchmark rests on.  These tests check the guarantees directly:

* every emitted column declares a swap parity, and it is the right one;
* symmetric features really are invariant and antisymmetric ones really flip;
* the frozen A0--A6 column sets are untouched when extensions are present;
* an undeclared namespace fails closed rather than being silently accepted.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lanthanide_separation.electronic import (
    COMPLEX_ELECTRONIC_SOURCES,
    DERIVED_COMPLEX_ELECTRONIC_NAME,
    ELECTRONIC_EVEN_PREFIX,
    ELECTRONIC_ODD_PREFIX,
    audit_electronic_sources,
    electronic_column_names,
    energy_column_name,
    pair_electronic_features,
)
from lanthanide_separation.feature_registry import (
    ABLATION_FAMILIES,
    EXTENSION_ABLATION_FAMILIES,
    SENSITIVITY_2D_ARM_FAMILIES,
    SENSITIVITY_2D_ARMS,
    SENSITIVITY_2D_SHUFFLE_BLOCKS,
    build_feature_registry,
)
from lanthanide_separation.pair_response import (
    PAIR_RESPONSE_EVEN_PREFIX,
    PAIR_RESPONSE_ODD_PREFIX,
    PAIR_RESPONSE_QUANTITIES,
    pair_response_column_names,
    pair_response_features,
    pair_response_subblock,
)
from lanthanide_separation.pairs import (
    build_lanthanide_pair_dataset,
    reverse_pair_features,
)

from lanthanide_separation.pairs import LANTHANIDE_Z


def source_row(
    *,
    ligand: str,
    metal: str,
    log_d: float,
    acid: float = 1.0,
    core_cn: int = 9,
) -> dict[str, object]:
    """One synthetic row carrying every column the extension blocks read."""

    z = LANTHANIDE_Z[metal]
    step = z - 57
    radius = 1.16 - 0.015 * step
    distance_mean = 2.50 - 0.010 * step
    return {
        "canonical_smiles": ligand,
        "extractant_name": ligand,
        "extractant_group": ligand,
        "metal": metal,
        "log_D": log_d,
        "geometry_ok": True,
        "geometry_key": f"{metal}|{ligand}|{core_cn}",
        "build_id": f"build-{metal}-{ligand}",
        "geometry_feature_build_id": f"feature-{metal}-{ligand}",
        "vr_graph_index": step,
        "safe_exp_id": f"{metal}_SAFE:{ligand}:{acid}:{log_d}",
        "Ionic Radius_metal": radius,
        "cond__acid_concentration_M": acid,
        "cond__temperature_C": 25.0,
        "MolWt": 300.0,
        "ecfp_0": 1,
        "ecfp_1": 0,
        # Complex composition, used to gate the energetic contrast.
        "coreCN": core_cn,
        "n_ligs": 3,
        "inner_sphere_anion": "nitrate",
        "fill_ligand": "nitrate",
        "n_fill": 0,
        "metal_ox": 3,
        # Geometric sources for the pair-response block.
        "feat3d__complex_physical__coordination_number": float(core_cn),
        "feat3d__complex_physical__ln_donor_distance_mean": distance_mean,
        "feat3d__complex_physical__ln_donor_distance_min": distance_mean - 0.08,
        "feat3d__complex_physical__ln_donor_distance_max": distance_mean + 0.09,
        "feat3d__complex_physical__ln_donor_distance_std": 0.05 + 0.001 * step,
        "feat3d__polyhedron__donor_angle_01_02_deg": 90.0 + step,
        "feat3d__polyhedron__donor_angle_01_03_deg": 140.0 - 0.5 * step,
        "feat3d__polyhedron__donor_angle_02_03_deg": 72.0 + 0.25 * step,
        "feat3d__polyhedron__ln_donor_distance_01": distance_mean,
        "feat3d__polyhedron__ln_donor_distance_02": distance_mean + 0.05,
        # Electronic sources.
        "feat3d__complex_physical__metal_partial_charge": 0.80 + 0.002 * step,
        "feat3d__complex_physical__donor_partial_charge_mean": -0.42 - 0.001 * step,
        "feat3d__complex_physical__donor_partial_charge_std": 0.06,
        "feat3d__complex_physical__donor_partial_charge_min": -0.50,
        "feat3d__complex_physical__donor_partial_charge_max": -0.31,
        "feat3d__complex_physical__dipole_magnitude": 3.1 + 0.02 * step,
        "feat3d__complex_physical__complex_total_energy_eV": -5000.0 - 12.0 * step,
        "feat3d__complex_physical__complex_free_energy_eV": -5000.0 - 12.0 * step,
    }


def make_source_frame() -> pd.DataFrame:
    """A small multi-extractant, multi-metal cohort with complete features."""

    rows: list[dict[str, object]] = []
    for ligand_index, ligand in enumerate(("L1", "L2", "L3")):
        for metal_index, metal in enumerate(("La", "Ce", "Pr", "Nd", "Sm")):
            rows.append(
                source_row(
                    ligand=ligand,
                    metal=metal,
                    log_d=0.5 * metal_index - 0.3 * ligand_index,
                    # One extractant changes coordination number across the
                    # series, so the composition gate is genuinely exercised.
                    core_cn=8 if (ligand == "L3" and metal in {"Nd", "Sm"}) else 9,
                )
            )
    return pd.DataFrame(rows)


def _values(offset: float) -> dict[str, float]:
    return {
        quantity.name: 2.0 + offset + 0.1 * index
        for index, quantity in enumerate(PAIR_RESPONSE_QUANTITIES)
    }


class PairResponseParityTests(unittest.TestCase):
    def test_every_column_declares_a_parity(self) -> None:
        for column in pair_response_column_names():
            self.assertTrue(
                column.startswith(
                    (PAIR_RESPONSE_ODD_PREFIX, PAIR_RESPONSE_EVEN_PREFIX)
                ),
                msg=column,
            )

    def test_odd_columns_flip_and_even_columns_hold_under_swap(self) -> None:
        radius = 0.05
        forward = pair_response_features(
            values_a=_values(0.0), values_b=_values(0.3), radius_difference=radius
        )
        reverse = pair_response_features(
            values_a=_values(0.3), values_b=_values(0.0), radius_difference=-radius
        )
        for column, value in forward.items():
            if column.startswith(PAIR_RESPONSE_ODD_PREFIX):
                if "excess" in column:
                    # excess = delta_g - delta_r; both terms are odd, so the whole
                    # quantity is odd only when the radius difference flips too,
                    # which is exactly what the swap does.
                    self.assertAlmostEqual(reverse[column], -value, places=12, msg=column)
                else:
                    self.assertAlmostEqual(reverse[column], -value, places=12, msg=column)
            else:
                self.assertAlmostEqual(reverse[column], value, places=12, msg=column)

    def test_compliance_is_the_shell_response_per_unit_contraction(self) -> None:
        values_a = _values(0.0)
        values_b = dict(values_a)
        name = PAIR_RESPONSE_QUANTITIES[0].name
        values_b[name] = values_a[name] - 0.02
        record = pair_response_features(
            values_a=values_a, values_b=values_b, radius_difference=0.04
        )
        self.assertAlmostEqual(
            record[f"{PAIR_RESPONSE_EVEN_PREFIX}compliance_{name}"], 0.5, places=12
        )
        self.assertAlmostEqual(
            record[f"{PAIR_RESPONSE_ODD_PREFIX}excess_{name}"], -0.02, places=12
        )

    def test_unusable_radius_difference_is_declared_missing(self) -> None:
        record = pair_response_features(
            values_a=_values(0.0), values_b=_values(0.1), radius_difference=0.0
        )
        compliance = [
            value
            for column, value in record.items()
            if column.startswith(f"{PAIR_RESPONSE_EVEN_PREFIX}compliance_")
        ]
        self.assertTrue(all(np.isnan(value) for value in compliance))

    def test_every_column_maps_to_a_declared_subblock(self) -> None:
        for column in pair_response_column_names():
            self.assertIn(pair_response_subblock(column), {"D1", "D2", "D3", "D4", "D5"})


class ElectronicParityTests(unittest.TestCase):
    def _electronic_values(self, offset: float) -> dict[str, float]:
        values = {
            source.name: 0.5 + offset + 0.05 * index
            for index, source in enumerate(COMPLEX_ELECTRONIC_SOURCES)
        }
        values[DERIVED_COMPLEX_ELECTRONIC_NAME] = values["q_metal"] - values["q_donor_mean"]
        values["complex_total_energy"] = -5000.0 + offset
        return values

    def test_every_column_declares_a_parity(self) -> None:
        for column in electronic_column_names():
            self.assertTrue(
                column.startswith((ELECTRONIC_ODD_PREFIX, ELECTRONIC_EVEN_PREFIX)),
                msg=column,
            )

    def test_odd_columns_flip_and_even_columns_hold_under_swap(self) -> None:
        forward = pair_electronic_features(
            values_a=self._electronic_values(0.0),
            values_b=self._electronic_values(0.2),
            radius_difference=0.05,
            composition_matches=True,
        )
        reverse = pair_electronic_features(
            values_a=self._electronic_values(0.2),
            values_b=self._electronic_values(0.0),
            radius_difference=-0.05,
            composition_matches=True,
        )
        for column, value in forward.items():
            expected = -value if column.startswith(ELECTRONIC_ODD_PREFIX) else value
            self.assertAlmostEqual(reverse[column], expected, places=9, msg=column)

    def test_energy_is_withheld_when_composition_differs(self) -> None:
        record = pair_electronic_features(
            values_a=self._electronic_values(0.0),
            values_b=self._electronic_values(0.2),
            radius_difference=0.05,
            composition_matches=False,
        )
        self.assertTrue(np.isnan(record[energy_column_name()]))

    def test_provenance_audit_declares_unavailable_descriptors(self) -> None:
        frame = pd.DataFrame(
            {
                "geometry_ok": [True, True],
                "canonical_smiles": ["CC", "CC"],
                "metal": ["La", "Ce"],
                "feat3d__complex_physical__metal_partial_charge": [0.8, 0.81],
                "feat3d__complex_physical__complex_total_energy_eV": [-10.0, -11.0],
                "feat3d__complex_physical__complex_free_energy_eV": [-10.0, -11.0],
                "feat3d__complex_physical__homo_eV": [np.nan, np.nan],
            }
        )
        audit = audit_electronic_sources(frame)
        self.assertEqual(
            audit["available_sources"]["q_metal"]["status"], "available"
        )
        self.assertEqual(len(audit["duplicate_columns"]), 1)
        self.assertIn("homo_eV", audit["unavailable_descriptors"])
        self.assertIn(
            "ligand_level_electronic", audit["unavailable_descriptors"]
        )
        self.assertIn("NOT a binding", audit["energy_definition"]["assumptions"])


class ExtensionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = make_source_frame()

    def _build(self, **kwargs):
        return build_lanthanide_pair_dataset(self.source, pair_scope="all", **kwargs)

    def test_frozen_contract_is_unchanged_by_the_extensions(self) -> None:
        frozen = self._build()
        extended = self._build(include_pair_response_3d=True, include_electronic=True)
        self.assertEqual(
            frozen.audit["cohort_sha256"], extended.audit["cohort_sha256"]
        )
        frozen_registry = build_feature_registry(frozen)
        extended_registry = build_feature_registry(extended)
        for arm in ABLATION_FAMILIES:
            self.assertEqual(
                frozen_registry.ablation_columns(arm),
                extended_registry.ablation_columns(arm),
                msg=arm,
            )

    def test_extension_blocks_are_absent_unless_requested(self) -> None:
        frozen = self._build()
        self.assertEqual(frozen.pair_response_columns, ())
        self.assertEqual(frozen.electronic_columns, ())
        self.assertEqual(frozen.extension_columns, ())

    def test_registry_assigns_every_extension_column_exactly_once(self) -> None:
        extended = self._build(include_pair_response_3d=True, include_electronic=True)
        registry = build_feature_registry(extended)
        columns = registry.columns
        self.assertEqual(len(columns), len(set(columns)))
        families = registry.families
        self.assertIn("3D_PAIR_RESPONSE", families)
        self.assertIn("ELEC_COMPLEX", families)
        self.assertIn("ELEC_PAIR", families)
        self.assertIn("ELEC_ENERGY", families)
        for arm in EXTENSION_ABLATION_FAMILIES:
            self.assertGreater(len(registry.ablation_columns(arm)), 0, msg=arm)

    def test_g1_is_an_alias_of_the_prespecified_a5_arm(self) -> None:
        extended = self._build(include_pair_response_3d=True, include_electronic=True)
        registry = build_feature_registry(extended)
        self.assertEqual(
            registry.ablation_columns("G1"), registry.ablation_columns("A5")
        )

    def test_swap_reverses_odd_and_preserves_even_extension_columns(self) -> None:
        extended = self._build(include_pair_response_3d=True, include_electronic=True)
        frame = extended.frame
        reversed_frame = reverse_pair_features(frame)
        odd = [c for c in extended.extension_columns if "__odd__" in c]
        even = [c for c in extended.extension_columns if "__even__" in c]
        self.assertTrue(odd and even)
        np.testing.assert_allclose(
            reversed_frame[odd].to_numpy(dtype=float),
            -frame[odd].to_numpy(dtype=float),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            reversed_frame[even].to_numpy(dtype=float),
            frame[even].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_double_swap_is_the_identity(self) -> None:
        extended = self._build(include_pair_response_3d=True, include_electronic=True)
        frame = extended.frame
        columns = list(extended.all_model_columns)
        twice = reverse_pair_features(reverse_pair_features(frame))
        np.testing.assert_allclose(
            twice[columns].to_numpy(dtype=float),
            frame[columns].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_undeclared_extension_namespace_fails_closed(self) -> None:
        extended = self._build(include_pair_response_3d=True)
        frame = extended.frame.copy()
        frame["pair3d__mystery_quantity"] = 1.0
        with self.assertRaises(ValueError):
            reverse_pair_features(frame)


class TwoDSensitivityLadderTests(unittest.TestCase):
    """STEP 12: the secondary 2D-representation ladder.

    The point of these arms is to ask whether a geometry or electronic block
    only looks useless because ~2 000 fingerprint bits can memorise ligand
    identity.  They are only interpretable if the representation cut is exactly
    what it claims to be, and if A2 itself is untouched.
    """

    def setUp(self) -> None:
        self.extended = build_lanthanide_pair_dataset(
            make_source_frame(),
            pair_scope="all",
            include_pair_response_3d=True,
            include_electronic=True,
        )
        self.registry = build_feature_registry(self.extended)

    def test_descriptor_and_fingerprint_halves_partition_the_2d_family(self) -> None:
        descriptors = set(self.registry.two_d_descriptor_columns)
        fingerprint = set(self.registry.two_d_ecfp_columns)
        family = set(self.registry.columns_for_families(("2D",)))
        self.assertTrue(descriptors)
        self.assertTrue(fingerprint)
        self.assertEqual(descriptors & fingerprint, set())
        self.assertEqual(descriptors | fingerprint, family)

    def test_s1_drops_every_fingerprint_bit_and_keeps_the_descriptors(self) -> None:
        columns = set(self.registry.sensitivity_2d_arm_columns("S1"))
        self.assertEqual(columns & set(self.registry.two_d_ecfp_columns), set())
        self.assertTrue(set(self.registry.two_d_descriptor_columns) <= columns)
        self.assertTrue(
            set(self.registry.columns_for_families(("CONDITIONS", "LN"))) <= columns
        )

    def test_s2_keeps_only_the_fingerprint(self) -> None:
        columns = set(self.registry.sensitivity_2d_arm_columns("S2"))
        self.assertEqual(columns & set(self.registry.two_d_descriptor_columns), set())
        self.assertTrue(set(self.registry.two_d_ecfp_columns) <= columns)

    def test_s1_and_s2_partition_the_a2_column_set(self) -> None:
        s1 = set(self.registry.sensitivity_2d_arm_columns("S1"))
        s2 = set(self.registry.sensitivity_2d_arm_columns("S2"))
        self.assertEqual(s1 | s2, set(self.registry.ablation_columns("A2")))

    def test_block_arms_are_s1_plus_exactly_their_declared_block(self) -> None:
        s1 = set(self.registry.sensitivity_2d_arm_columns("S1"))
        for arm, families in SENSITIVITY_2D_ARM_FAMILIES.items():
            block = set(self.registry.columns_for_families(families))
            self.assertEqual(
                set(self.registry.sensitivity_2d_arm_columns(arm)),
                s1 | block,
                msg=arm,
            )

    def test_the_ladder_never_changes_the_frozen_arms(self) -> None:
        frozen = build_feature_registry(
            build_lanthanide_pair_dataset(make_source_frame(), pair_scope="all")
        )
        for arm in ABLATION_FAMILIES:
            self.assertEqual(
                frozen.ablation_columns(arm),
                self.registry.ablation_columns(arm),
                msg=arm,
            )

    def test_columns_stay_in_contract_order(self) -> None:
        order = {column: index for index, column in enumerate(self.registry.columns)}
        for arm in SENSITIVITY_2D_ARMS:
            columns = self.registry.sensitivity_2d_arm_columns(arm)
            positions = [order[column] for column in columns]
            self.assertEqual(positions, sorted(positions), msg=arm)

    def test_unknown_sensitivity_arm_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.registry.sensitivity_2d_arm_columns("S99")

    def test_block_arm_without_its_family_fails_closed(self) -> None:
        bare = build_feature_registry(
            build_lanthanide_pair_dataset(make_source_frame(), pair_scope="all")
        )
        with self.assertRaises(ValueError):
            bare.sensitivity_2d_arm_columns("S4")
        # ... and it is simply not offered rather than silently degraded.
        self.assertNotIn("S4", bare.sensitivity_2d_feature_sets())
        self.assertIn("S1", bare.sensitivity_2d_feature_sets())

    def test_every_declared_block_arm_has_a_permutation_control(self) -> None:
        # A block that adds columns on top of S1 makes the same metal- or
        # complex-specific claim as its A2-referenced twin, so it needs the same
        # control; S3 reuses the frozen A5_SHUFFLED block and is excluded.
        for arm in SENSITIVITY_2D_ARM_FAMILIES:
            if arm == "S3":
                continue
            self.assertIn(arm, SENSITIVITY_2D_SHUFFLE_BLOCKS, msg=arm)
            controlled, families = SENSITIVITY_2D_SHUFFLE_BLOCKS[arm]
            self.assertEqual(controlled, arm)
            self.assertEqual(tuple(families), SENSITIVITY_2D_ARM_FAMILIES[arm])


if __name__ == "__main__":
    unittest.main()
