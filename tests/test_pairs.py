from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lanthanide_separation.pairs import (
    KNOWN_CENSORED_SAFE_IDS,
    PAIR_TARGET_COLUMN,
    TODGA_SMILES,
    build_adjacent_pair_dataset,
    reverse_pair_features,
)


def source_row(
    *,
    ligand: str,
    name: str,
    metal: str,
    log_d: float,
    acid: float = 1.0,
    safe_id: str | None = None,
    geometry_ok: bool = True,
) -> dict[str, object]:
    z = {
        "La": 57,
        "Ce": 58,
        "Pr": 59,
        "Nd": 60,
        "Sm": 62,
        "Eu": 63,
    }[metal]
    return {
        "canonical_smiles": ligand,
        "LIGAND_SMILES": ligand,
        "extractant_name": name,
        "extractant_group": ligand,
        "metal": metal,
        "log_D": log_d,
        "geometry_ok": geometry_ok,
        "geometry_key": f"{z}|{ligand}|nitrate",
        "build_id": f"build-{metal}-{ligand}",
        "geometry_feature_build_id": f"feature-{metal}-{ligand}",
        "vr_graph_index": z - 57,
        "safe_exp_id": safe_id or f"{metal}_SAFE:{acid}:{log_d}",
        "Ionic Radius_metal": {"La": 1.16, "Ce": 1.143, "Pr": 1.126, "Nd": 1.109,
                                "Sm": 1.079, "Eu": 1.066}[metal],
        "cond__acid_concentration_M": acid,
        "cond__temperature_C": 25.0,
        "MolWt": 300.0,
        "ecfp_0": 1,
        "ecfp_1": 0,
        "feat3d__complex_physical__ln_donor_distance_mean": 2.5 - (z - 57) * 0.01,
        "feat3d__complex_physical__dipole_x": 99.0,
        "feat3d__polyhedron__donor_atomic_number_01": 8.0,
        "feat3d__polyhedron__donor_angle_01_02_deg": 90.0 + (z - 57),
        "feat3d__polyhedron__donor_angle_01_03_deg": 0.0,
        "feat3d__polyhedron__ln_donor_distance_01_A": 2.5 - (z - 57) * 0.01,
        "feat3d__polyhedron__ln_donor_distance_02_A": 0.0,
    }


class PairBuildingTests(unittest.TestCase):
    def test_only_true_atomic_number_neighbors_and_exact_conditions(self) -> None:
        rows = [
            source_row(ligand="L1", name="L1", metal="La", log_d=1.0),
            source_row(ligand="L1", name="L1", metal="Ce", log_d=2.0),
            # Same ligand but a different acid concentration: no cross-condition pair.
            source_row(ligand="L1", name="L1", metal="Ce", log_d=9.0, acid=2.0),
            # Nd-Sm must not be called adjacent because Pm (Z=61) is between them.
            source_row(ligand="L1", name="L1", metal="Nd", log_d=3.0),
            source_row(ligand="L1", name="L1", metal="Sm", log_d=4.0),
        ]
        result = build_adjacent_pair_dataset(pd.DataFrame(rows))
        self.assertEqual(result.frame["pair_label"].tolist(), ["La-Ce"])
        self.assertAlmostEqual(result.frame.iloc[0][PAIR_TARGET_COLUMN], -1.0)
        self.assertAlmostEqual(result.frame.iloc[0]["log_D_B"], 2.0)
        self.assertNotIn(
            "delta3d__feat3d__complex_physical__dipole_x", result.delta3d_columns
        )
        self.assertNotIn(
            "delta3d__feat3d__polyhedron__donor_atomic_number_01",
            result.delta3d_columns,
        )
        self.assertIn(
            "delta3d__feat3d__derived_invariant__donor_angle_mean_deg",
            result.delta3d_columns,
        )
        self.assertIn(
            "delta3d__feat3d__derived_invariant__donor_angle_legendre_p2_mean",
            result.delta3d_columns,
        )
        self.assertIn(
            "delta3d__feat3d__derived_invariant__donor_distance_median",
            result.delta3d_columns,
        )
        self.assertEqual(len(result.audit["cohort_sha256"]), 64)

        all_ranked = build_adjacent_pair_dataset(
            pd.DataFrame(rows), delta3d_feature_set="all-ranked"
        )
        self.assertIn(
            "delta3d__feat3d__polyhedron__donor_atomic_number_01",
            all_ranked.delta3d_columns,
        )

    def test_replicates_are_median_aggregated_before_pairing(self) -> None:
        rows = [
            source_row(ligand="L1", name="L1", metal="La", log_d=1.0),
            source_row(ligand="L1", name="L1", metal="Ce", log_d=2.0),
            source_row(ligand="L1", name="L1", metal="Ce", log_d=4.0),
        ]
        result = build_adjacent_pair_dataset(pd.DataFrame(rows), replicate_policy="median")
        pair = result.frame.iloc[0]
        self.assertEqual(pair["n_replicates_B"], 2)
        self.assertAlmostEqual(pair["log_D_B"], 3.0)
        self.assertAlmostEqual(pair[PAIR_TARGET_COLUMN], -2.0)
        self.assertEqual(result.audit["replicated_cells_with_target_conflict"], 1)

    def test_target_extreme_quarantine_is_opt_in_and_audited(self) -> None:
        censored_id = sorted(KNOWN_CENSORED_SAFE_IDS)[0]
        rows = [
            source_row(ligand="L1", name="L1", metal="La", log_d=1.0),
            source_row(ligand="L1", name="L1", metal="Ce", log_d=2.0),
            source_row(ligand=TODGA_SMILES, name="PHEN-6OH", metal="Eu", log_d=0.5),
            source_row(
                ligand="BAD", name="BAD", metal="Pr", log_d=-12.467, safe_id=censored_id
            ),
        ]
        default_result = build_adjacent_pair_dataset(pd.DataFrame(rows))
        self.assertEqual(len(default_result.quarantine), 1)

        result = build_adjacent_pair_dataset(
            pd.DataFrame(rows), quarantine_known_censored_targets=True
        )
        self.assertEqual(len(result.quarantine), 2)
        self.assertEqual(result.audit["quarantined_rows"], 2)

    def test_missing_conditions_do_not_match_by_default(self) -> None:
        rows = [
            source_row(ligand="L1", name="L1", metal="La", log_d=1.0, acid=np.nan),
            source_row(ligand="L1", name="L1", metal="Ce", log_d=2.0, acid=np.nan),
        ]
        with self.assertRaisesRegex(ValueError, "No true-adjacent"):
            build_adjacent_pair_dataset(pd.DataFrame(rows))

        sensitivity = build_adjacent_pair_dataset(
            pd.DataFrame(rows), require_complete_conditions=False
        )
        self.assertEqual(len(sensitivity.frame), 1)

    def test_inconsistent_ecfp_for_one_extractant_fails_closed(self) -> None:
        rows = [
            source_row(ligand="L1", name="L1", metal="La", log_d=1.0),
            source_row(ligand="L1", name="L1", metal="Ce", log_d=2.0),
        ]
        rows[1]["ecfp_0"] = 0
        rows[1]["ecfp_1"] = 1
        with self.assertRaisesRegex(ValueError, "multiple ECFP fingerprints"):
            build_adjacent_pair_dataset(pd.DataFrame(rows))

    def test_missing_vr_index_is_allowed_only_without_accepted_geometry(self) -> None:
        rows = [
            source_row(ligand="L1", name="L1", metal="La", log_d=1.0),
            source_row(ligand="L1", name="L1", metal="Ce", log_d=2.0),
            source_row(
                ligand="unavailable",
                name="unavailable",
                metal="Sm",
                log_d=0.0,
                geometry_ok=False,
            ),
        ]
        rows[-1]["vr_graph_index"] = np.nan
        result = build_adjacent_pair_dataset(pd.DataFrame(rows))
        self.assertEqual(result.frame["pair_label"].tolist(), ["La-Ce"])

        rows[0]["vr_graph_index"] = np.nan
        with self.assertRaisesRegex(ValueError, "geometry_ok=True"):
            build_adjacent_pair_dataset(pd.DataFrame(rows))

    def test_reverse_pair_features_swaps_and_negates_contrasts(self) -> None:
        frame = pd.DataFrame(
            {
                "pair__Z_A": [57.0],
                "pair__Z_B": [58.0],
                "pair__delta_Z": [-1.0],
                "pair__ionic_radius_A": [1.16],
                "pair__ionic_radius_B": [1.143],
                "pair__delta_ionic_radius": [0.017],
                "pair__Z_mean": [57.5],
                "base__condition": [3.0],
                "delta3d__distance": [0.02],
            }
        )
        reversed_frame = reverse_pair_features(frame)
        self.assertEqual(reversed_frame.iloc[0]["pair__Z_A"], 58.0)
        self.assertEqual(reversed_frame.iloc[0]["pair__Z_B"], 57.0)
        self.assertAlmostEqual(reversed_frame.iloc[0]["pair__delta_Z"], 1.0)
        self.assertAlmostEqual(reversed_frame.iloc[0]["delta3d__distance"], -0.02)
        self.assertEqual(reversed_frame.iloc[0]["base__condition"], 3.0)


if __name__ == "__main__":
    unittest.main()
