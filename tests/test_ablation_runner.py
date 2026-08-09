from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from lanthanide_separation.feature_registry import build_feature_registry
from lanthanide_separation.pairs import PairDataset


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ablation_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_ablation_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _registry():
    columns = (
        "base__cond__temperature_C",
        "pair__Z_A",
        "base__MolWt",
        "delta3d__feat3d__global_geometry__radius_of_gyration",
        "delta3d__feat3d__complex_physical__ln_donor_distance_mean",
    )
    pair_data = PairDataset(
        frame=pd.DataFrame({column: [0.0] for column in columns}),
        baseline_columns=columns[:3],
        delta3d_columns=columns[3:],
        descriptor_columns=(),
        audit={"pair_scope": "all"},
        quarantine=pd.DataFrame(),
    )
    return build_feature_registry(pair_data)


class AblationRunnerTests(unittest.TestCase):
    def test_primary_defaults_are_all_pairs_extractant_holdout_and_exact_shape(self) -> None:
        args = runner.validate_args(runner.parse_args([]))

        self.assertEqual(args.pair_scope, "all")
        self.assertEqual(args.group_mode, "extractant")
        self.assertEqual(
            args.descriptor_blocks,
            ("global_shape", "coordination_shape"),
        )
        self.assertEqual(args.shuffle_seed_values, runner.DEFAULT_SHUFFLE_SEEDS)
        self.assertFalse(args.no_block_ablations)

    def test_quick_mode_is_bounded_and_descriptor_blocks_are_canonical(self) -> None:
        args = runner.validate_args(
            runner.parse_args(
                [
                    "--quick",
                    "--trees",
                    "500",
                    "--bootstrap",
                    "5000",
                    "--shuffle-seeds",
                    "9,3,7",
                    "--geometry-descriptor-blocks",
                    "global_shape,enclosure",
                ]
            )
        )

        self.assertEqual(args.trees, 48)
        self.assertEqual(args.bootstrap, 100)
        self.assertEqual(args.shuffle_seed_values, (9,))
        self.assertEqual(args.descriptor_blocks, ("enclosure", "global_shape"))

    def test_invalid_seeds_blocks_and_digest_fail_before_run_creation(self) -> None:
        cases = (
            ["--shuffle-seeds", "1,1"],
            ["--shuffle-seeds", "x"],
            ["--geometry-descriptor-blocks", "mystery"],
            ["--model-seed", "-1"],
            ["--expected-dataset-sha256", "not-a-hash"],
        )
        for argv in cases:
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                runner.validate_args(runner.parse_args(argv))

    def test_json_and_csv_writers_are_atomic_and_json_has_no_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            json_path = root / "audit.json"
            csv_path = root / "table.csv"

            runner.write_json_atomic(
                {"native": float("nan"), "numpy": np.float64(np.inf), "ok": 3.0},
                json_path,
            )
            runner.write_csv_atomic(pd.DataFrame({"value": [1, 2]}), csv_path)

            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"native": None, "numpy": None, "ok": 3.0},
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(csv_path), pd.DataFrame({"value": [1, 2]})
            )
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_report_marks_empty_local_blocks_unavailable_not_duplicate_models(self) -> None:
        registry = _registry()
        pair_data = SimpleNamespace(frame=pd.DataFrame(index=range(3)))
        args = runner.validate_args(
            runner.parse_args(["--geometry-descriptor-blocks", "none"])
        )
        tables = {
            "per_ablation_metrics": pd.DataFrame(
                [
                    {
                        "ablation": "A2",
                        "mae": 1.0,
                        "rmse": 1.1,
                        "r2": 0.2,
                        "spearman": 0.3,
                    },
                    {
                        "ablation": "A5",
                        "mae": 0.9,
                        "rmse": 1.0,
                        "r2": 0.25,
                        "spearman": 0.35,
                    },
                ]
            ),
            "paired_deltas": pd.DataFrame(
                [{"comparison": "A2_vs_A5", "delta_MAE": 0.1}]
            ),
            "per_extractant_comparison": pd.DataFrame(),
            "per_lanthanide_comparison": pd.DataFrame(),
        }

        report = runner.build_report(
            args=args,
            pair_data=pair_data,
            registry=registry,
            tables=tables,
        )

        self.assertIn("unavailable and were skipped, not fit as duplicate", report)
        self.assertIn("D2, D3, D4, D5", report)
        self.assertIn("primary comparison is A2", report)

    def test_evaluator_tables_are_normalized_to_stable_artifact_contract(self) -> None:
        registry = _registry()
        identities = {
            "pair_id": ["p1", "p2"],
            "extractant": ["x1", "x2"],
            "pair_label": ["La-Ce", "La-Pr"],
            "metal_A": ["La", "La"],
            "metal_B": ["Ce", "Pr"],
            "outer_fold": [0, 1],
            "log_SF_A_over_B": [1.0, -0.5],
            "prediction_A2": [0.8, -0.4],
            "prediction_A5": [0.9, -0.6],
        }
        group_metrics = pd.DataFrame(
            [
                {
                    "extractant_id": extractant,
                    "ablation": ablation,
                    "n_rows": 2,
                    "mae": mae,
                    "rmse": mae,
                    "spearman": 0.5,
                }
                for extractant in ("x1", "x2")
                for ablation, mae in (("A2", 1.0), ("A5", 0.75))
            ]
        )
        metal_metrics = pd.DataFrame(
            [
                {
                    "lanthanide": metal,
                    "ablation": ablation,
                    "n_pair_memberships": 2,
                    "mae": mae,
                    "rmse": mae,
                    "spearman": 0.5,
                }
                for metal in ("La", "Ce")
                for ablation, mae in (("A2", 1.0), ("A5", 0.75))
            ]
        )
        evaluated = {
            name: {"columns": list(registry.ablation_columns(name))}
            for name in ("A2", "A5")
        }
        result = SimpleNamespace(
            predictions=pd.DataFrame(identities),
            fold_metrics=pd.DataFrame(),
            fold_assignments=pd.DataFrame(),
            fold_memberships=pd.DataFrame(),
            inner_fold_assignments=pd.DataFrame(),
            tuning_results=pd.DataFrame(),
            preprocessing_audit=pd.DataFrame(),
            shuffle_audit=pd.DataFrame(),
            per_ablation_metrics=pd.DataFrame(),
            paired_deltas=pd.DataFrame(),
            per_extractant_metrics=group_metrics,
            per_lanthanide_metrics=metal_metrics,
            feature_audit={"evaluated_feature_sets": evaluated},
        )

        tables = runner.prepare_artifact_tables(result, registry)

        self.assertEqual(len(tables["predictions"]), 4)
        self.assertEqual(
            set(tables["predictions"]["ablation"]), {"A2", "A5"}
        )
        self.assertTrue(
            {
                "pair_id",
                "extractant",
                "pair_label",
                "metal_A",
                "metal_B",
                "outer_fold",
                "ablation",
                "y_true",
                "y_pred",
            }.issubset(tables["predictions"].columns)
        )
        self.assertTrue(
            np.allclose(tables["per_extractant_comparison"]["delta_MAE"], 0.25)
        )
        self.assertTrue(
            np.allclose(tables["per_lanthanide_comparison"]["delta_MAE"], 0.25)
        )
        self.assertEqual(
            list(tables["shuffle_audit"].columns)[:2], ["ablation", "outer_fold"]
        )
        self.assertIn("family", tables["selected_features"].columns)


if __name__ == "__main__":
    unittest.main()
