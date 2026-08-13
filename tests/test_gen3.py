from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from lanthanide_separation.feature_registry import build_feature_registry
from lanthanide_separation.gen3_data import (
    COMPLEX_ELECTRONIC_NAMES,
    build_gen3_feature_adapter,
)
from lanthanide_separation.gen3_metrics import paired_extractant_bootstrap
from lanthanide_separation.gen3_evaluation import _training_label_shuffle
from lanthanide_separation.gen3_models import (
    AntisymmetricCorrectionRegressor,
    LatentDifferenceFit,
    _new_shared_score_network,
    fold_local_group_weights,
)
from lanthanide_separation.gen3_protocol import (
    load_gen3_protocol,
    validate_seed_pair,
)
from lanthanide_separation.pairs import (
    PAIR_TARGET_COLUMN,
    build_lanthanide_pair_dataset,
    reverse_pair_features,
)
from scripts.aggregate_gen3_runs import BASELINE_ARM, _decision_table, _expected_arms
from tests.test_extension_features import make_source_frame


class Gen3ProtocolTests(unittest.TestCase):
    def test_frozen_primary_and_confirmation_seed_plan(self) -> None:
        protocol = load_gen3_protocol("gen3_protocol.json")
        self.assertEqual(
            validate_seed_pair(protocol, split_seed=104729, model_seed=42),
            "primary_split_seed",
        )
        self.assertEqual(
            validate_seed_pair(protocol, split_seed=104729, model_seed=7),
            "secondary_model_seed_confirmation",
        )
        with self.assertRaisesRegex(ValueError, "absent from the frozen protocol"):
            validate_seed_pair(protocol, split_seed=999, model_seed=42)

    def test_h1_and_h3_search_budgets_are_fixed_and_bounded(self) -> None:
        protocol = load_gen3_protocol("gen3_protocol.json")
        h1 = protocol["h1"]
        arm_count = len(h1["weighting_schemes"]) * len(h1["losses"])
        candidate_evaluations = (
            arm_count * len(h1["screening_candidates"])
            + h1["expanded_shortlist_weight_loss_arms"]
            * len(h1["expanded_candidates"])
        )
        self.assertEqual(candidate_evaluations, 24)
        self.assertEqual(h1["inner_catboost_fits_per_outer_fold"], 72)
        self.assertEqual(len(protocol["h3"]["search_candidates"]), 2)
        self.assertEqual(protocol["h3"]["training"]["max_epochs"], 80)


class Gen3WeightTests(unittest.TestCase):
    def test_all_schemes_are_mean_one_and_group_equal_equalizes_totals(self) -> None:
        groups = np.asarray(["large"] * 9 + ["small"], dtype=object)
        for scheme in ("none", "group_sqrt", "group_equal"):
            weights = fold_local_group_weights(groups, scheme)
            self.assertAlmostEqual(float(weights.mean()), 1.0)
            self.assertTrue(np.isfinite(weights).all())
        equal = fold_local_group_weights(groups, "group_equal")
        self.assertAlmostEqual(float(equal[:9].sum()), float(equal[9:].sum()))

    def test_counts_are_scoped_to_supplied_fit_rows(self) -> None:
        full = fold_local_group_weights(["A", "A", "B", "B"], "group_equal")
        subset = fold_local_group_weights(["A", "A", "B"], "group_equal")
        self.assertAlmostEqual(float(full[0]), float(full[2]))
        self.assertNotAlmostEqual(float(subset[0]), float(subset[2]))
        self.assertAlmostEqual(float(subset.mean()), 1.0)


class Gen3AdapterTests(unittest.TestCase):
    def test_compact_electronic_view_and_shoulders_are_target_independent(self) -> None:
        source = make_source_frame()
        pair_data = build_lanthanide_pair_dataset(source, include_electronic=True)
        registry = build_feature_registry(pair_data)
        adapter = build_gen3_feature_adapter(
            pair_data,
            registry,
            source,
            reference_feature_registry=registry.to_dict(),
        )
        self.assertEqual(len(adapter.e3_raw_columns), 21)
        self.assertEqual(len(adapter.e3_compact_columns), 21)
        self.assertFalse(any("log_d" in column.lower() for column in adapter.e3_compact_columns))
        row = adapter.frame.iloc[0]
        for name in COMPLEX_ELECTRONIC_NAMES:
            delta = float(row[f"elec__odd__delta_{name}"])
            mean = float(row[f"elec__even__mean_{name}"])
            self.assertAlmostEqual(float(row[f"gen3__elec_A__{name}"]), mean + delta / 2)
            self.assertAlmostEqual(float(row[f"gen3__elec_B__{name}"]), mean - delta / 2)


class Gen3ModelConstraintTests(unittest.TestCase):
    def test_label_shuffle_is_deterministic_and_within_pair_type(self) -> None:
        frame = pd.DataFrame(
            {"pair_label": ["La-Ce", "La-Ce", "La-Ce", "Ce-Pr", "Ce-Pr"]}
        )
        target = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0])
        first, audit = _training_label_shuffle(target, frame, seed=17)
        second, _ = _training_label_shuffle(target, frame, seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(sorted(first[:3]), [1.0, 2.0, 3.0])
        self.assertEqual(sorted(first[3:]), [10.0, 20.0])
        self.assertEqual(audit["outer_test_labels_touched"], 0)
        self.assertTrue(audit["training_only"])

    def test_residual_ridge_is_exactly_antisymmetric(self) -> None:
        frame = pd.DataFrame(
            {
                "pair__Z_A": [57.0, 58.0, 57.0, 59.0],
                "pair__Z_B": [58.0, 59.0, 59.0, 60.0],
                "elec__odd__delta_q": [-0.1, -0.2, -0.3, -0.4],
                "elec__even__absdelta_q": [0.1, 0.2, 0.3, 0.4],
            }
        )
        columns = ("elec__odd__delta_q", "elec__even__absdelta_q")
        model = AntisymmetricCorrectionRegressor(
            columns,
            kind="ridge",
            parameters={"alpha": 1.0},
            random_state=3,
            n_jobs=1,
        )
        model.fit(frame, [-0.3, -0.4, -0.5, -0.6], ["a", "a", "b", "b"])
        forward = model.predict(frame)
        reverse = model.predict(reverse_pair_features(frame))
        np.testing.assert_allclose(forward, -reverse, atol=1e-12, rtol=0.0)

    def test_shared_g_guarantees_antisymmetry_and_transitivity(self) -> None:
        import torch

        torch.manual_seed(5)
        # Same context, three shoulders A/B/C encoded in AB, BC, and AC rows.
        frame = pd.DataFrame(
            {
                "context_A": [0.25, 0.25, 0.25],
                "z_A": [57.0, 58.0, 57.0],
                "context_B": [0.25, 0.25, 0.25],
                "z_B": [58.0, 59.0, 59.0],
            }
        )
        a_columns = ("context_A", "z_A")
        b_columns = ("context_B", "z_B")
        a = frame[list(a_columns)].to_numpy(dtype=float)
        b = frame[list(b_columns)].to_numpy(dtype=float)
        imputer = SimpleImputer(strategy="median").fit(np.vstack([a, b]))
        scaler = StandardScaler().fit(imputer.transform(np.vstack([a, b])))
        network = _new_shared_score_network(2, (8, 4), 0.0)
        fitted = LatentDifferenceFit(
            imputer=imputer,
            scaler=scaler,
            model=network,
            a_columns=a_columns,
            b_columns=b_columns,
            epochs_trained=0,
        )
        prediction = fitted.predict(frame)
        self.assertAlmostEqual(float(prediction[0] + prediction[1]), float(prediction[2]), places=6)
        score_a, score_b = fitted.score_shoulders(frame)
        np.testing.assert_allclose(prediction, score_a - score_b, atol=1e-7)
        np.testing.assert_allclose(prediction, -(score_b - score_a), atol=1e-7)


class Gen3BootstrapTests(unittest.TestCase):
    def test_extractant_draw_multiplicity_is_not_collapsed(self) -> None:
        frame = pd.DataFrame(
            {
                "extractant": ["large", "large", "large", "small"],
                PAIR_TARGET_COLUMN: [0.0, 0.0, 0.0, 0.0],
                "prediction_base": [1.0, 1.0, 1.0, 4.0],
                "prediction_candidate": [0.0, 0.0, 0.0, 3.0],
            }
        )
        seed = 17
        result = paired_extractant_bootstrap(
            frame,
            {"comparison": ("base", "candidate")},
            replicates=1,
            seed=seed,
        ).iloc[0]
        # Both group deltas equal +1 here, so a repeated draw must still yield
        # exactly +1 rather than becoming row-weighted after label collapse.
        self.assertAlmostEqual(float(result["point_delta_mae"]), 1.0)
        self.assertAlmostEqual(float(result["ci95_low"]), 1.0)
        self.assertTrue(bool(result["multiplicity_preserved"]))


class Gen3AggregateDecisionTests(unittest.TestCase):
    def test_expected_arm_contract_and_four_of_five_win_rule(self) -> None:
        protocol = load_gen3_protocol("gen3_protocol.json")
        arms = _expected_arms(protocol)
        self.assertEqual(arms[0], BASELINE_ARM)
        self.assertEqual(len(arms), 20)

        seeds = protocol["splits"]["split_seeds"]
        rows = [
            {
                "arm": BASELINE_ARM,
                "split_seed": seed,
                "equal_extractant_macro_mae": 1.0,
            }
            for seed in seeds
        ]
        rows.extend(
            {
                "arm": "H1_CATBOOST_GROUP_EQUAL_MAE",
                "split_seed": seed,
                "equal_extractant_macro_mae": 0.9 if index < 4 else 1.1,
            }
            for index, seed in enumerate(seeds)
        )
        decision = _decision_table(
            pd.DataFrame(rows),
            pd.DataFrame(columns=["arm", "split_seed", "lambda"]),
            protocol,
        ).set_index("arm")
        candidate = decision.loc["H1_CATBOOST_GROUP_EQUAL_MAE"]
        self.assertEqual(int(candidate["positive_split_seeds"]), 4)
        self.assertTrue(bool(candidate["win_rule_passed"]))
        self.assertTrue(bool(candidate["eligible_for_champion"]))


if __name__ == "__main__":
    unittest.main()
