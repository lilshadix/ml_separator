from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from lanthanide_separation.ablation import _training_shuffle, run_ablation_benchmark
from lanthanide_separation.feature_registry import build_feature_registry
from lanthanide_separation.pairs import PAIR_TARGET_COLUMN, PairDataset


def synthetic_ablation_dataset() -> PairDataset:
    rows: list[dict[str, object]] = []
    pair_types = (
        ("La", "Ce", 57.0, 58.0),
        ("La", "Pr", 57.0, 59.0),
        ("Ce", "Pr", 58.0, 59.0),
    )
    for extractant_index in range(8):
        for pair_index, (metal_a, metal_b, z_a, z_b) in enumerate(pair_types):
            local_3d = 0.08 * extractant_index + 0.03 * pair_index
            for condition_index in range(2):
                target = 0.65 * local_3d + 0.02 * condition_index
                rows.append(
                    {
                        "pair_id": (
                            f"p-{extractant_index}-{pair_index}-{condition_index}"
                        ),
                        "condition_id": f"c-{extractant_index}-{condition_index}",
                        "extractant": f"ligand-{extractant_index}",
                        # Deliberately shared across extractants: this is a
                        # recorded stricter-sensitivity collision, not literal
                        # extractant/provenance leakage in the primary mode.
                        "ecfp_exact_cluster": "shared-ecfp-cluster",
                        "pair_label": f"{metal_a}-{metal_b}",
                        "metal_A": metal_a,
                        "metal_B": metal_b,
                        "source_id_A": f"source-{extractant_index}-{metal_a}",
                        "source_id_B": f"source-{extractant_index}-{metal_b}",
                        "geometry_key_A": f"geometry-{extractant_index}-{metal_a}",
                        "geometry_key_B": f"geometry-{extractant_index}-{metal_b}",
                        "geometry_feature_build_id_A": (
                            f"feature-build-{extractant_index}-{metal_a}"
                        ),
                        "geometry_feature_build_id_B": (
                            f"feature-build-{extractant_index}-{metal_b}"
                        ),
                        "vr_graph_index_A": extractant_index * 10 + pair_index,
                        "vr_graph_index_B": 100 + extractant_index * 10 + pair_index,
                        "geometry_ok_A": True,
                        "geometry_ok_B": True,
                        "geometry_qc_class_A": "OK",
                        "geometry_qc_class_B": "OK",
                        PAIR_TARGET_COLUMN: target,
                        "pair__Z_A": z_a,
                        "pair__Z_B": z_b,
                        "pair__Z_mean": (z_a + z_b) / 2.0,
                        "pair__delta_Z": z_a - z_b,
                        "pair__ionic_radius_A": 1.20 - 0.01 * (z_a - 57.0),
                        "pair__ionic_radius_B": 1.20 - 0.01 * (z_b - 57.0),
                        "pair__ionic_radius_mean": 1.15,
                        "pair__delta_ionic_radius": 0.01 * (z_b - z_a),
                        "base__cond__temperature_C": 20.0 + condition_index,
                        "base__MolWt": 300.0 + extractant_index,
                        (
                            "delta3d__feat3d__global_geometry__"
                            "radius_of_gyration"
                        ): 0.1 * extractant_index,
                        (
                            "delta3d__feat3d__complex_physical__"
                            "ln_donor_distance_mean"
                        ): local_3d,
                    }
                )
    frame = pd.DataFrame(rows)
    baseline_columns = (
        "pair__Z_A",
        "pair__Z_B",
        "pair__Z_mean",
        "pair__delta_Z",
        "pair__ionic_radius_A",
        "pair__ionic_radius_B",
        "pair__ionic_radius_mean",
        "pair__delta_ionic_radius",
        "base__cond__temperature_C",
        "base__MolWt",
    )
    delta3d_columns = (
        "delta3d__feat3d__global_geometry__radius_of_gyration",
        "delta3d__feat3d__complex_physical__ln_donor_distance_mean",
    )
    return PairDataset(
        frame=frame,
        baseline_columns=baseline_columns,
        delta3d_columns=delta3d_columns,
        descriptor_columns=(),
        audit={"pair_scope": "all", "pairs_excluded_missing_geometry": 4},
        quarantine=pd.DataFrame(),
    )


class AblationTests(unittest.TestCase):
    def test_training_shuffle_moves_complete_repeated_complex_groups_only(self) -> None:
        pair_data = synthetic_ablation_dataset()
        frame = pair_data.frame.iloc[:16].copy()
        local_column = (
            "delta3d__feat3d__complex_physical__ln_donor_distance_mean"
        )
        protected = frame[
            [PAIR_TARGET_COLUMN, "base__cond__temperature_C", "base__MolWt"]
        ].copy()

        shuffled, audit = _training_shuffle(frame, (local_column,), seed=71)

        pd.testing.assert_frame_equal(
            shuffled[protected.columns].reset_index(drop=True),
            protected.reset_index(drop=True),
        )
        self.assertTrue(audit["donors_from_training_only"])
        self.assertTrue(audit["complete_block_moved_together"])
        self.assertEqual(audit["test_rows_touched"] if "test_rows_touched" in audit else 0, 0)
        repeated = shuffled.groupby(
            ["geometry_key_A", "geometry_key_B"], dropna=False
        )[local_column].nunique(dropna=False)
        self.assertEqual(int(repeated.max()), 1)

    def test_nested_all_ablation_contract_is_paired_and_leakage_safe(self) -> None:
        pair_data = synthetic_ablation_dataset()
        registry = build_feature_registry(pair_data)
        result = run_ablation_benchmark(
            pair_data,
            registry,
            group_column="extractant",
            outer_folds=2,
            inner_folds=2,
            n_estimators=3,
            n_jobs=1,
            seed=13,
            split_seed=101,
            shuffle_seeds=(17,),
            include_block_ablations=True,
            n_bootstrap=20,
            parameter_grid=({"max_features": 1.0, "min_samples_leaf": 1},),
        )

        main = {f"A{index}" for index in range(7)}
        observed = {
            column.removeprefix("prediction_")
            for column in result.predictions
            if column.startswith("prediction_")
        }
        self.assertTrue(main.issubset(observed))
        self.assertIn("A5_SHUFFLED_s17", observed)
        self.assertIn("A2+D1", observed)
        self.assertEqual(len(result.fold_assignments), len(pair_data.frame))
        self.assertEqual(
            len(result.fold_memberships), 2 * len(pair_data.frame)
        )
        self.assertTrue(result.leakage_audit["passed"])
        self.assertIn(
            "delta_macro_group_mae",
            result.summary["paired_group_bootstrap"]["comparisons"]["A2_vs_A5"],
        )
        for fold in result.leakage_audit["outer_folds"]:
            self.assertEqual(fold["extractant_overlap"], 0)
            self.assertEqual(fold["source_id_overlap"], 0)
            self.assertGreater(fold["ecfp_exact_cluster_overlap"], 0)
            self.assertEqual(
                fold["record_only_sensitivity_fields"],
                ["ecfp_exact_cluster_overlap"],
            )

        a5_prediction = result.predictions["prediction_A5"].to_numpy(dtype=float)
        expected_oof_r2 = r2_score(
            pair_data.frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float), a5_prediction
        )
        observed_oof_r2 = result.per_ablation_metrics.set_index("ablation").loc[
            "A5", "r2"
        ]
        self.assertAlmostEqual(float(observed_oof_r2), float(expected_oof_r2))

        a2_a5 = result.paired_deltas[
            result.paired_deltas["comparison"].eq("A2_vs_A5")
        ]
        self.assertEqual(len(a2_a5), 2)
        for row in a2_a5.itertuples(index=False):
            fold_metrics = result.fold_metrics[
                result.fold_metrics["outer_fold"].eq(row.outer_fold)
            ].set_index("ablation")
            self.assertAlmostEqual(
                row.delta_mae,
                fold_metrics.loc["A2", "mae"] - fold_metrics.loc["A5", "mae"],
            )

        self.assertEqual(
            set(result.per_lanthanide_metrics["lanthanide"]), {"La", "Ce", "Pr"}
        )
        self.assertFalse(result.geometry_qc_summary["high_vs_low_estimable"])
        self.assertEqual(
            result.geometry_qc_summary["failed_geometries_in_model_cohort"], 0
        )
        self.assertTrue(
            result.preprocessing_audit["outer_test_rows_seen_during_fit"].eq(0).all()
        )
        self.assertTrue(result.shuffle_audit["test_rows_touched"].eq(0).all())
        self.assertTrue(result.shuffle_audit["complete_block_moved_together"].all())


if __name__ == "__main__":
    unittest.main()
