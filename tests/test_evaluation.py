from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lanthanide_separation.evaluation import (
    AntisymmetricExtraTreesRegressor,
    _group_folds,
    nested_group_benchmark,
)
from lanthanide_separation.pairs import PAIR_TARGET_COLUMN, PairDataset, reverse_pair_features


def synthetic_pair_dataset() -> PairDataset:
    rng = np.random.default_rng(7)
    rows = []
    for extractant_index in range(8):
        for pair_index in range(3):
            geometry_delta = 0.05 * (extractant_index - 3.5) + 0.02 * pair_index
            target = 0.3 * geometry_delta + rng.normal(0.0, 0.005)
            rows.append(
                {
                    "pair_id": f"p-{extractant_index}-{pair_index}",
                    "condition_id": f"c-{extractant_index}-{pair_index}",
                    "extractant": f"ligand-{extractant_index}",
                    "ecfp_exact_cluster": f"cluster-{extractant_index // 2}",
                    "pair_label": "La-Ce",
                    "metal_A": "La",
                    "metal_B": "Ce",
                    "n_replicates_A": 1,
                    "n_replicates_B": 1,
                    "source_id_A": f"source-a-{extractant_index}-{pair_index}",
                    "source_id_B": f"source-b-{extractant_index}-{pair_index}",
                    "geometry_key_A": f"geometry-a-{extractant_index}-{pair_index}",
                    "geometry_key_B": f"geometry-b-{extractant_index}-{pair_index}",
                    PAIR_TARGET_COLUMN: target,
                    "pair__Z_A": 57.0,
                    "pair__Z_B": 58.0,
                    "pair__Z_mean": 57.5,
                    "pair__delta_Z": -1.0,
                    "pair__ionic_radius_A": 1.16,
                    "pair__ionic_radius_B": 1.143,
                    "pair__ionic_radius_mean": 1.1515,
                    "pair__delta_ionic_radius": 0.017,
                    "base__condition": float(pair_index),
                    "delta3d__distance": geometry_delta,
                }
            )
    frame = pd.DataFrame(rows)
    baseline = (
        "pair__Z_A",
        "pair__Z_B",
        "pair__Z_mean",
        "pair__delta_Z",
        "pair__ionic_radius_A",
        "pair__ionic_radius_B",
        "pair__ionic_radius_mean",
        "pair__delta_ionic_radius",
        "base__condition",
    )
    return PairDataset(
        frame=frame,
        baseline_columns=baseline,
        delta3d_columns=("delta3d__distance",),
        audit={},
        quarantine=pd.DataFrame(),
    )


class EvaluationTests(unittest.TestCase):
    def test_antisymmetric_prediction_is_exact(self) -> None:
        pair_data = synthetic_pair_dataset()
        model = AntisymmetricExtraTreesRegressor(
            pair_data.full_columns, n_estimators=12, random_state=3, n_jobs=1
        )
        frame = pair_data.frame
        model.fit(frame, frame[PAIR_TARGET_COLUMN], frame["extractant"])
        forward = model.predict(frame)
        backward = model.predict(reverse_pair_features(frame))
        np.testing.assert_allclose(forward, -backward, atol=1e-12)

    def test_nested_group_benchmark_has_no_group_overlap(self) -> None:
        pair_data = synthetic_pair_dataset()
        result = nested_group_benchmark(
            pair_data,
            group_column="extractant",
            outer_folds=4,
            inner_folds=2,
            n_estimators=8,
            n_jobs=1,
            n_bootstrap=20,
            parameter_grid=({"max_features": 1.0, "min_samples_leaf": 1},),
            seed=11,
            split_seed=101,
            fit_final_model=False,
        )
        self.assertTrue(result.summary["leakage_audit"]["passed"])
        self.assertTrue(
            all(
                fold["group_overlap"] == 0
                for fold in result.summary["leakage_audit"]["outer_folds"]
            )
        )
        self.assertFalse(result.predictions["prediction_delta3d"].isna().any())
        self.assertEqual(
            set(result.fold_assignments["outer_fold"].unique().tolist()), {0, 1, 2, 3}
        )
        self.assertIsNone(result.final_model)
        self.assertFalse(result.summary["final_deployment_parameters"]["fitted"])

    def test_matched_2d_control_is_identical_when_no_delta3d_exists(self) -> None:
        pair_data = synthetic_pair_dataset()
        no_3d = PairDataset(
            frame=pair_data.frame,
            baseline_columns=pair_data.baseline_columns,
            delta3d_columns=(),
            audit={},
            quarantine=pd.DataFrame(),
        )
        result = nested_group_benchmark(
            no_3d,
            group_column="extractant",
            outer_folds=4,
            inner_folds=2,
            n_estimators=8,
            n_jobs=1,
            n_bootstrap=20,
            parameter_grid=({"max_features": 1.0, "min_samples_leaf": 1},),
            seed=13,
            split_seed=103,
            fit_final_model=False,
        )
        np.testing.assert_allclose(
            result.predictions["prediction_2d_second_unshrunk"],
            result.predictions["prediction_delta3d_unshrunk"],
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            result.predictions["prediction_2d_ensemble"],
            result.predictions["prediction_delta3d"],
            atol=0.0,
            rtol=0.0,
        )

    def test_seeded_group_folds_are_reproducible_and_group_disjoint(self) -> None:
        groups = np.asarray([f"g-{index}" for index in range(20)], dtype=object)
        strata = np.asarray(["La-Ce"] * 20, dtype=object)
        first = _group_folds(
            groups,
            strata,
            5,
            split_seed=17,
            allow_fewer_splits=False,
        )
        repeated = _group_folds(
            groups,
            strata,
            5,
            split_seed=17,
            allow_fewer_splits=False,
        )
        changed = _group_folds(
            groups,
            strata,
            5,
            split_seed=19,
            allow_fewer_splits=False,
        )
        for (train, test), (train_repeat, test_repeat) in zip(first, repeated, strict=True):
            np.testing.assert_array_equal(train, train_repeat)
            np.testing.assert_array_equal(test, test_repeat)
            self.assertFalse(set(groups[train]) & set(groups[test]))
        first_assignment = np.concatenate([test for _, test in first])
        changed_assignment = np.concatenate([test for _, test in changed])
        self.assertFalse(np.array_equal(first_assignment, changed_assignment))


if __name__ == "__main__":
    unittest.main()
