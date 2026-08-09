from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
AGGREGATOR = REPO_ROOT / "scripts" / "aggregate_ablation_runs.py"

REQUIRED_PLACEHOLDERS = (
    "metrics.json",
    "fold_metrics.csv",
    "fold_memberships.csv",
    "inner_fold_assignments.csv",
    "inner_cv_tuning.csv",
    "preprocessing_audit.json",
    "preprocessing_parameters.csv",
    "selected_features.csv",
    "shuffle_audit.csv",
    "geometry_descriptor_audit.json",
    "geometry_qc_summary.json",
    "per_ablation_metrics.csv",
    "paired_ablation_deltas.csv",
    "per_extractant_comparison.csv",
    "per_lanthanide_comparison.csv",
    "report.md",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _object_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _refresh_terminal_manifests(run_dir: Path) -> None:
    success_path = run_dir / "_SUCCESS.json"
    manifest_path = run_dir / "artifact_hashes.json"
    success_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    hashes = {
        path.name: _sha256(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name not in {"_SUCCESS.json", "artifact_hashes.json"}
    }
    _write_json(
        manifest_path,
        {"schema_version": "1.0", "algorithm": "sha256", "sha256": hashes},
    )
    _write_json(
        success_path,
        {
            "status": "complete",
            "summary_sha256": _sha256(run_dir / "summary.json"),
            "validation_sha256": _sha256(run_dir / "validation.json"),
            "artifact_hashes_sha256": _sha256(manifest_path),
        },
    )


def _make_run(root: Path, *, model_seed: int, prediction_offset: float) -> Path:
    run_dir = root / f"run_model_{model_seed}"
    run_dir.mkdir()
    pairs = [
        ("p0", "E0", "La-Ce", "La", "Ce", 0, -1.0, "high_confidence"),
        ("p1", "E0", "La-Pr", "La", "Pr", 0, -0.2, "high_confidence"),
        ("p2", "E1", "Ce-Pr", "Ce", "Pr", 1, 0.4, "questionable_or_recovered"),
        ("p3", "E1", "Ce-Nd", "Ce", "Nd", 1, 1.3, "questionable_or_recovered"),
        ("p4", "E2", "Pr-Nd", "Pr", "Nd", 2, -0.6, "high_confidence"),
        ("p5", "E2", "La-Nd", "La", "Nd", 2, 0.9, "high_confidence"),
    ]
    ablation_bias = {
        "A0": 0.30,
        "A1": 0.25,
        "A2": 0.20,
        "A3": 0.18,
        "A4": 0.16,
        "A5": 0.08,
        "A6": 0.07,
        "A5_SHUFFLED_s1009": 0.21,
        "A2+D1": 0.12,
    }
    rows = []
    for pair_id, extractant, label, metal_a, metal_b, fold, truth, qc in pairs:
        direction = -1.0 if int(pair_id[1:]) % 2 else 1.0
        for ablation, bias in ablation_bias.items():
            rows.append(
                {
                    "pair_id": pair_id,
                    "extractant": extractant,
                    "pair_label": label,
                    "metal_A": metal_a,
                    "metal_B": metal_b,
                    "outer_fold": fold,
                    "geometry_qc_pair": qc,
                    "ablation": ablation,
                    "y_true": truth,
                    "y_pred": truth + direction * (bias + prediction_offset),
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "oof_predictions.csv", index=False)
    assignments = pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "extractant": extractant,
                "outer_fold": fold,
                "outer_split_seed": 77,
            }
            for pair_id, extractant, _, _, _, fold, _, _ in pairs
        ]
    )
    assignments.to_csv(run_dir / "fold_assignments.csv", index=False)

    registry = {
        "schema_version": "1.0",
        "pair_scope": "all",
        "assignments": [
            {"column": "base__temperature", "family": "CONDITIONS"},
            {"column": "pair__Z_A", "family": "LN"},
        ],
    }
    _write_json(run_dir / "feature_registry.json", registry)
    pair_audit = {"pair_scope": "all", "cohort_sha256": "b" * 64, "pair_rows": 6}
    _write_json(run_dir / "pair_build_audit.json", pair_audit)
    _write_json(run_dir / "leakage_audit.json", {"passed": True})
    _write_json(
        run_dir / "feature_audit.json",
        {
            "all_registry_features_assigned_once": True,
            "target_or_identifier_features_present": False,
        },
    )
    _write_json(run_dir / "validation.json", {"passed": True})
    for name in REQUIRED_PLACEHOLDERS:
        path = run_dir / name
        if path.suffix == ".json":
            _write_json(path, {})
        elif path.suffix == ".csv":
            path.write_text("placeholder\n", encoding="utf-8")
        else:
            path.write_text("synthetic fixture\n", encoding="utf-8")
    protocol = {
        "schema_version": "1.0",
        "pair_scope": "all",
        "group_mode": "extractant",
        "split_seed": 77,
        "outer_folds": 3,
        "inner_folds": 2,
        "trees": 48,
        "parameter_grid": [{"max_features": 1.0, "min_samples_leaf": 1}],
        "quick_mode": True,
        "geometry_descriptor_blocks": ["global_shape"],
    }
    protocol_hash = _object_sha256(protocol)
    software = {
        "python": "3.synthetic",
        "numpy": "synthetic",
        "pandas": "synthetic",
        "scipy": "synthetic",
        "scikit_learn": "synthetic",
    }
    _write_json(
        run_dir / "run_config.json",
        {
            "dataset_sha256": "a" * 64,
            "vr_asset_sha256": "d" * 64,
            "arguments": {
                "pair_scope": "all",
                "group_mode": "extractant",
                "model_seed": model_seed,
                "split_seed": 77,
            },
            "scientific_protocol": protocol,
            "scientific_protocol_sha256": protocol_hash,
            "software": software,
        },
    )
    summary = {
        "status": "complete",
        "dataset_sha256": "a" * 64,
        "cohort_sha256": "b" * 64,
        "pair_scope": "all",
        "group_mode": "extractant",
        "model_seed": model_seed,
        "split_seed": 77,
        "vr_asset_sha256": "d" * 64,
        "feature_registry_sha256": _sha256(run_dir / "feature_registry.json"),
        "fold_assignments_sha256": _sha256(run_dir / "fold_assignments.csv"),
        "fold_memberships_sha256": _sha256(run_dir / "fold_memberships.csv"),
        "inner_fold_assignments_sha256": _sha256(
            run_dir / "inner_fold_assignments.csv"
        ),
        "geometry_descriptor_audit_sha256": _sha256(
            run_dir / "geometry_descriptor_audit.json"
        ),
        "scientific_protocol": protocol,
        "scientific_protocol_sha256": protocol_hash,
        "implementation_sha256": {"ablation.py": "c" * 64},
        "software": software,
    }
    _write_json(run_dir / "summary.json", summary)
    _refresh_terminal_manifests(run_dir)
    return run_dir


class AblationAggregationTest(unittest.TestCase):
    def _invoke(self, root: Path, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(AGGREGATOR),
                "--run-root",
                str(root),
                "--output-dir",
                str(output),
                "--expected-runs",
                "2",
                "--expected-pair-scope",
                "all",
                "--bootstrap-replicates",
                "30",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_aggregates_exactly_aligned_hash_verified_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            root.mkdir()
            _make_run(root, model_seed=11, prediction_offset=0.00)
            _make_run(root, model_seed=29, prediction_offset=0.01)
            output = root / "aggregate_test"

            completed = self._invoke(root, output)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            success = json.loads((output / "_SUCCESS.json").read_text(encoding="utf-8"))
            self.assertEqual(success["status"], "complete")
            self.assertEqual(
                success["aggregate_implementation_sha256"], _sha256(AGGREGATOR)
            )
            self.assertEqual(success["bootstrap_replicates"], 30)
            self.assertEqual(success["bootstrap_seed"], 8675309)
            validation = json.loads(
                (output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(validation["passed"])
            cross_seed = pd.read_csv(output / "cross_seed_oof_predictions.csv")
            self.assertEqual(len(cross_seed), 6)
            self.assertIn("prediction_A5_mean", cross_seed.columns)
            summary = json.loads(
                (output / "aggregate_summary.json").read_text(encoding="utf-8")
            )
            primary = summary["paired_comparisons"]["A2_vs_A5"]
            self.assertGreater(
                primary["cross_seed_oof_ensemble_delta"]["delta_mae"], 0.0
            )
            self.assertIn(
                "delta_macro_extractant_mae",
                primary["paired_extractant_bootstrap"]["deltas"],
            )
            self.assertEqual(primary["held_out_extractants"]["n"], 3)
            self.assertIn("A2_vs_A5_delta_mae", pd.read_csv(
                output / "per_lanthanide_comparison.csv"
            ).columns)
            per_lanthanide = pd.read_csv(output / "per_lanthanide_comparison.csv")
            ce = per_lanthanide.set_index("lanthanide").loc["Ce"]
            ce_a = cross_seed[cross_seed["metal_A"].eq("Ce")]
            ce_b = cross_seed[cross_seed["metal_B"].eq("Ce")]
            oriented_truth = np.concatenate(
                [ce_a["y_true"].to_numpy(), -ce_b["y_true"].to_numpy()]
            )
            oriented_prediction = np.concatenate(
                [
                    ce_a["prediction_A5_mean"].to_numpy(),
                    -ce_b["prediction_A5_mean"].to_numpy(),
                ]
            )
            self.assertAlmostEqual(
                float(ce["A5_r2"]),
                float(r2_score(oriented_truth, oriented_prediction)),
            )

    def test_rejects_one_tampered_hashed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            root.mkdir()
            _make_run(root, model_seed=11, prediction_offset=0.00)
            tampered = _make_run(root, model_seed=29, prediction_offset=0.01)
            (tampered / "metrics.json").write_text("{}\nchanged\n", encoding="utf-8")
            output = root / "aggregate_tampered"

            completed = self._invoke(root, output)

            self.assertEqual(completed.returncode, 2)
            self.assertFalse((output / "_SUCCESS.json").exists())
            validation = json.loads(
                (output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertFalse(validation["passed"])
            self.assertTrue(
                any(
                    "artifact hash mismatch: metrics.json" in error
                    for error in validation["errors"]
                )
            )

    def test_rejects_different_fold_membership_contract_even_when_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            root.mkdir()
            _make_run(root, model_seed=11, prediction_offset=0.00)
            changed = _make_run(root, model_seed=29, prediction_offset=0.01)
            (changed / "fold_memberships.csv").write_text(
                "placeholder\nchanged-but-internally-hashed\n", encoding="utf-8"
            )
            _refresh_terminal_manifests(changed)
            output = root / "aggregate_different_folds"

            completed = self._invoke(root, output)

            self.assertEqual(completed.returncode, 2)
            validation = json.loads(
                (output / "validation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                any(
                    "fold_memberships_sha256" in error
                    for error in validation["errors"]
                )
            )

    def test_rejects_different_inner_fold_plan_even_when_fully_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            root.mkdir()
            _make_run(root, model_seed=11, prediction_offset=0.00)
            changed = _make_run(root, model_seed=29, prediction_offset=0.01)
            inner_path = changed / "inner_fold_assignments.csv"
            inner_path.write_text(
                "placeholder\nchanged-but-internally-hashed\n", encoding="utf-8"
            )
            summary = json.loads((changed / "summary.json").read_text())
            summary["inner_fold_assignments_sha256"] = _sha256(inner_path)
            _write_json(changed / "summary.json", summary)
            _refresh_terminal_manifests(changed)
            output = root / "aggregate_different_inner_folds"

            completed = self._invoke(root, output)

            self.assertEqual(completed.returncode, 2)
            validation = json.loads((output / "validation.json").read_text())
            self.assertTrue(
                any(
                    "Runs do not share one inner_fold_assignments_sha256" in error
                    for error in validation["errors"]
                )
            )

    def test_rejects_different_scientific_protocol_even_when_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            root.mkdir()
            _make_run(root, model_seed=11, prediction_offset=0.00)
            changed = _make_run(root, model_seed=29, prediction_offset=0.01)
            run_config = json.loads((changed / "run_config.json").read_text())
            summary = json.loads((changed / "summary.json").read_text())
            run_config["scientific_protocol"]["trees"] = 200
            summary["scientific_protocol"]["trees"] = 200
            changed_hash = _object_sha256(summary["scientific_protocol"])
            run_config["scientific_protocol_sha256"] = changed_hash
            summary["scientific_protocol_sha256"] = changed_hash
            _write_json(changed / "run_config.json", run_config)
            _write_json(changed / "summary.json", summary)
            _refresh_terminal_manifests(changed)

            output = root / "aggregate_different_protocol"
            completed = self._invoke(root, output)

            self.assertEqual(completed.returncode, 2)
            validation = json.loads((output / "validation.json").read_text())
            self.assertTrue(
                any(
                    "scientific_protocol_sha256" in error
                    for error in validation["errors"]
                )
            )

    def test_rejects_different_vr_asset_even_when_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            root.mkdir()
            _make_run(root, model_seed=11, prediction_offset=0.00)
            changed = _make_run(root, model_seed=29, prediction_offset=0.01)
            run_config = json.loads((changed / "run_config.json").read_text())
            summary = json.loads((changed / "summary.json").read_text())
            run_config["vr_asset_sha256"] = "e" * 64
            summary["vr_asset_sha256"] = "e" * 64
            _write_json(changed / "run_config.json", run_config)
            _write_json(changed / "summary.json", summary)
            _refresh_terminal_manifests(changed)

            output = root / "aggregate_different_vr"
            completed = self._invoke(root, output)

            self.assertEqual(completed.returncode, 2)
            validation = json.loads((output / "validation.json").read_text())
            self.assertTrue(
                any("vr_asset_sha256" in error for error in validation["errors"])
            )

    def test_rejects_different_software_even_when_rehashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            root.mkdir()
            _make_run(root, model_seed=11, prediction_offset=0.00)
            changed = _make_run(root, model_seed=29, prediction_offset=0.01)
            run_config = json.loads((changed / "run_config.json").read_text())
            summary = json.loads((changed / "summary.json").read_text())
            run_config["software"]["scikit_learn"] = "different"
            summary["software"]["scikit_learn"] = "different"
            _write_json(changed / "run_config.json", run_config)
            _write_json(changed / "summary.json", summary)
            _refresh_terminal_manifests(changed)

            output = root / "aggregate_different_software"
            completed = self._invoke(root, output)

            self.assertEqual(completed.returncode, 2)
            validation = json.loads((output / "validation.json").read_text())
            self.assertTrue(
                any("software_sha256" in error for error in validation["errors"])
            )


if __name__ == "__main__":
    unittest.main()
