from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = PROJECT_ROOT / "slurm" / "submit_ablation_multiseed.sh"
WORKER_SCRIPT = PROJECT_ROOT / "slurm" / "ablation_multiseed.slurm"
AGGREGATE_SCRIPT = PROJECT_ROOT / "slurm" / "aggregate_ablation.slurm"


class AblationSlurmContractTests(unittest.TestCase):
    def test_scripts_parse_with_stock_bash(self) -> None:
        for script in (SUBMIT_SCRIPT, WORKER_SCRIPT, AGGREGATE_SCRIPT):
            with self.subTest(script=script.name):
                completed = subprocess.run(
                    ["/bin/bash", "-n", str(script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_openmp_contract_precedes_worker_setup(self) -> None:
        required = "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}"
        cases = ((WORKER_SCRIPT, "8"), (AGGREGATE_SCRIPT, "1"))
        for script, fallback in cases:
            with self.subTest(script=script.name):
                lines = script.read_text(encoding="utf-8").splitlines()
                header = lines[:40]
                self.assertIn(
                    f"export SLURM_CPUS_PER_TASK=${{SLURM_CPUS_PER_TASK:-{fallback}}}",
                    header,
                )
                self.assertIn(required, header)

    def test_dry_run_uses_exact_shape_defaults_without_creating_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_root = Path(temporary_directory) / "must_not_exist"
            environment = {
                **os.environ,
                "DRY_RUN": "1",
                "RUN_ROOT": str(run_root),
                "PYTHON_BIN": sys.executable,
            }
            completed = subprocess.run(
                ["/bin/bash", str(SUBMIT_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(run_root.exists())
            self.assertIn(
                "Descriptor blocks: global_shape,coordination_shape", completed.stdout
            )
            self.assertIn("Donor-composition counts enabled: 1", completed.stdout)
            self.assertIn("--partition=campus", completed.stdout)
            self.assertIn("--account=acf-utk0011", completed.stdout)

    def test_literal_partition_and_account_placeholders_fail_before_run_creation(self) -> None:
        cases = (("PARTITION", "actual_partition"), ("ACCOUNT", "actual_account"))
        for variable, value in cases:
            with (
                self.subTest(variable=variable),
                tempfile.TemporaryDirectory() as temporary_directory,
            ):
                run_root = Path(temporary_directory) / "must_not_exist"
                environment = {
                    **os.environ,
                    "DRY_RUN": "1",
                    "RUN_ROOT": str(run_root),
                    "PYTHON_BIN": sys.executable,
                    variable: value,
                }
                completed = subprocess.run(
                    ["/bin/bash", str(SUBMIT_SCRIPT)],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("Placeholder", completed.stderr)
                self.assertFalse(run_root.exists())

    def test_resume_rejects_protocol_drift_before_another_sbatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            fake_bin = temporary_root / "bin"
            fake_bin.mkdir()
            call_log = temporary_root / "sbatch.calls"
            sbatch = fake_bin / "sbatch"
            sbatch.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$SBATCH_CALL_LOG\"\nprintf '12345\\n'\n",
                encoding="utf-8",
            )
            flock = fake_bin / "flock"
            flock.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            sbatch.chmod(0o755)
            flock.chmod(0o755)

            run_root = temporary_root / "run_root"
            dataset_path = temporary_root / "dataset.parquet"
            vr_asset_path = temporary_root / "vietoris_rips_inputs.npz"
            dataset_path.write_bytes(b"frozen-dataset-v1")
            vr_asset_path.write_bytes(b"frozen-vr-asset-v1")
            base_environment = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                "SBATCH_CALL_LOG": str(call_log),
                "DRY_RUN": "0",
                "RUN_ROOT": str(run_root),
                "PROJECT_DIR": str(PROJECT_ROOT),
                "PYTHON_BIN": sys.executable,
                "DATASET_PATH": str(dataset_path),
                "VR_ASSET_PATH": str(vr_asset_path),
            }
            first = subprocess.run(
                ["/bin/bash", str(SUBMIT_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=base_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)

            contract_path = run_root / "experiment_contract.tsv"
            self.assertTrue(contract_path.is_file())
            with contract_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                set(rows[0]),
                {
                    "schema_version",
                    "pair_scope",
                    "group_mode",
                    "split_seed",
                    "outer_folds",
                    "inner_folds",
                    "trees",
                    "bootstrap",
                    "descriptor_blocks",
                    "descriptor_profile",
                    "shuffle_seeds",
                    "include_block_ablations",
                    "quick_mode",
                    "delta3d_feature_set",
                    "replicate_policy",
                    "include_donor_composition_counts",
                    "aggregate_bootstrap_replicates",
                    "aggregate_bootstrap_seed",
                    "dataset_path",
                    "dataset_sha256",
                    "vr_asset_path",
                    "vr_asset_sha256",
                    # Generation-2 switches. They belong to the immutable
                    # contract so a resume cannot silently change which arms a
                    # run root contains.
                    "include_pair_response_3d",
                    "include_electronic",
                    "include_extension_arms",
                    "include_symmetric_3d",
                    "include_reference_baselines",
                    "include_2d_sensitivity",
                    "extension_shuffle_seeds",
                    "prespecified_arms",
                },
            )
            self.assertEqual(rows[0]["descriptor_blocks"], "global_shape,coordination_shape")
            # An unset environment must still describe the frozen protocol.
            self.assertEqual(rows[0]["include_pair_response_3d"], "0")
            self.assertEqual(rows[0]["include_electronic"], "0")
            self.assertEqual(rows[0]["include_extension_arms"], "0")
            self.assertEqual(rows[0]["include_2d_sensitivity"], "0")
            self.assertEqual(rows[0]["extension_shuffle_seeds"], "none")
            self.assertEqual(rows[0]["prespecified_arms"], "A0-A6")
            self.assertEqual(rows[0]["aggregate_bootstrap_replicates"], "2000")
            self.assertEqual(rows[0]["aggregate_bootstrap_seed"], "8675309")
            self.assertEqual(rows[0]["dataset_path"], str(dataset_path.resolve()))
            self.assertEqual(
                rows[0]["dataset_sha256"], hashlib.sha256(dataset_path.read_bytes()).hexdigest()
            )
            self.assertEqual(rows[0]["vr_asset_path"], str(vr_asset_path.resolve()))
            self.assertEqual(
                rows[0]["vr_asset_sha256"],
                hashlib.sha256(vr_asset_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(len(call_log.read_text(encoding="utf-8").splitlines()), 2)

            alternate_dataset_path = temporary_root / "same-content-new-path.parquet"
            alternate_dataset_path.write_bytes(dataset_path.read_bytes())
            path_drift_environment = {
                **base_environment,
                "RESUME_RUN_ROOT": "1",
                "DATASET_PATH": str(alternate_dataset_path),
            }
            path_drift = subprocess.run(
                ["/bin/bash", str(SUBMIT_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=path_drift_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(path_drift.returncode, 2)
            self.assertIn("immutable experiment contract", path_drift.stderr)

            dataset_path.write_bytes(b"frozen-dataset-v2")
            content_drift_environment = {
                **base_environment,
                "RESUME_RUN_ROOT": "1",
            }
            content_drift = subprocess.run(
                ["/bin/bash", str(SUBMIT_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=content_drift_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(content_drift.returncode, 2)
            self.assertIn("immutable experiment contract", content_drift.stderr)

            worker_content_drift = subprocess.run(
                ["/bin/bash", str(WORKER_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=base_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(worker_content_drift.returncode, 0)
            self.assertIn(
                "Array environment disagrees with immutable experiment contract",
                worker_content_drift.stderr,
            )

            aggregate_content_drift = subprocess.run(
                ["/bin/bash", str(AGGREGATE_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=base_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(aggregate_content_drift.returncode, 0)
            self.assertIn(
                "Aggregate environment disagrees with immutable experiment contract",
                aggregate_content_drift.stderr,
            )
            dataset_path.write_bytes(b"frozen-dataset-v1")
            self.assertEqual(len(call_log.read_text(encoding="utf-8").splitlines()), 2)

            drifted_environment = {
                **base_environment,
                "RESUME_RUN_ROOT": "1",
                "AGGREGATE_BOOTSTRAP": "2001",
            }
            resumed = subprocess.run(
                ["/bin/bash", str(SUBMIT_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=drifted_environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(resumed.returncode, 2)
            self.assertIn("immutable experiment contract", resumed.stderr)
            self.assertEqual(len(call_log.read_text(encoding="utf-8").splitlines()), 2)

            aggregate = subprocess.run(
                ["/bin/bash", str(AGGREGATE_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=drifted_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(aggregate.returncode, 0)
            self.assertIn(
                "Aggregate environment disagrees with immutable experiment contract",
                aggregate.stderr,
            )
            self.assertEqual(len(call_log.read_text(encoding="utf-8").splitlines()), 2)

            protocol = {
                "pair_scope": "all",
                "group_mode": "extractant",
                "split_seed": 104729,
                "outer_folds": 5,
                "inner_folds": 3,
                "trees": 200,
                "bootstrap_replicates": 1000,
                "geometry_descriptor_blocks": ["global_shape", "coordination_shape"],
                "descriptor_profile": "core",
                "shuffle_seeds": [1009, 2017, 3019],
                "include_block_ablations": True,
                "quick_mode": False,
                "delta3d_feature_set": "compact-invariant",
                "replicate_policy": "unique",
                "include_donor_composition_counts": True,
            }
            model_seeds = (42, 7, 137, 2027, 9001)
            for index, model_seed in enumerate(model_seeds):
                run_dir = run_root / f"run_{index}_model_{model_seed}_split_104729"
                run_dir.mkdir()
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "pair_scope": "all",
                            "group_mode": "extractant",
                            "split_seed": 104729,
                            "model_seed": model_seed,
                            "dataset_path": str(dataset_path.resolve()),
                            "dataset_sha256": hashlib.sha256(
                                dataset_path.read_bytes()
                            ).hexdigest(),
                            "vr_asset_path": str(vr_asset_path.resolve()),
                            "vr_asset_sha256": hashlib.sha256(
                                vr_asset_path.read_bytes()
                            ).hexdigest(),
                            "scientific_protocol": protocol,
                        }
                    ),
                    encoding="utf-8",
                )

            aggregate_dir = run_root / "aggregate"
            aggregate_dir.mkdir()
            aggregate_summary_path = aggregate_dir / "aggregate_summary.json"
            validation_path = aggregate_dir / "validation.json"
            success_path = aggregate_dir / "_SUCCESS.json"
            aggregate_summary = {
                "pair_scope": "all",
                "run_count": 5,
                "split_seed": 104729,
                "model_seeds": list(model_seeds),
                "paired_comparisons": {
                    name: {
                        "paired_extractant_bootstrap": {
                            "replicates": 2000,
                            "seed": 8675309 + comparison_index * 1009,
                        }
                    }
                    for comparison_index, name in enumerate(
                        ("A2_vs_A5", "A2_vs_A6", "A3_vs_A2")
                    )
                },
            }
            validation_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "aggregate_implementation_sha256": hashlib.sha256(
                            (PROJECT_ROOT / "scripts" / "aggregate_ablation_runs.py").read_bytes()
                        ).hexdigest(),
                        "bootstrap_replicates": 2000,
                        "bootstrap_seed": 8675309,
                        "fingerprints": {
                            "group_mode": ["extractant"],
                            "dataset_sha256": [
                                hashlib.sha256(dataset_path.read_bytes()).hexdigest()
                            ],
                            "vr_asset_sha256": [
                                hashlib.sha256(vr_asset_path.read_bytes()).hexdigest()
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            def publish_aggregate_summary() -> None:
                aggregate_summary_path.write_text(
                    json.dumps(aggregate_summary), encoding="utf-8"
                )
                digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                success_path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "pair_scope": "all",
                            "aggregate_implementation_sha256": hashlib.sha256(
                                (
                                    PROJECT_ROOT
                                    / "scripts"
                                    / "aggregate_ablation_runs.py"
                                ).read_bytes()
                            ).hexdigest(),
                            "bootstrap_replicates": 2000,
                            "bootstrap_seed": 8675309,
                            "aggregate_summary_sha256": digest(aggregate_summary_path),
                            "artifact_sha256": {
                                "aggregate_summary.json": digest(aggregate_summary_path),
                                "validation.json": digest(validation_path),
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            publish_aggregate_summary()
            valid_resume = subprocess.run(
                ["/bin/bash", str(AGGREGATE_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=base_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(valid_resume.returncode, 0, valid_resume.stderr)
            self.assertIn("Validated aggregate already exists", valid_resume.stdout)

            aggregate_summary["paired_comparisons"]["A2_vs_A5"][
                "paired_extractant_bootstrap"
            ]["replicates"] = 2001
            publish_aggregate_summary()
            invalid_resume = subprocess.run(
                ["/bin/bash", str(AGGREGATE_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=base_environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(invalid_resume.returncode, 2)
            self.assertIn(
                "Refusing to overwrite corrupt or incomplete aggregate",
                invalid_resume.stderr,
            )


if __name__ == "__main__":
    unittest.main()
