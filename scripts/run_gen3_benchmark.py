#!/usr/bin/env python3
"""Run one hash-validated split/model-seed task from the frozen gen3 protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lanthanide_separation.feature_registry import build_feature_registry  # noqa: E402
from lanthanide_separation.gen3_data import build_gen3_feature_adapter  # noqa: E402
from lanthanide_separation.gen3_evaluation import run_gen3_benchmark  # noqa: E402
from lanthanide_separation.gen3_protocol import (  # noqa: E402
    file_sha256,
    load_gen3_protocol,
    object_sha256,
    validate_seed_pair,
    verify_frozen_inputs,
)
from lanthanide_separation.pairs import build_lanthanide_pair_dataset  # noqa: E402


DEFAULT_PROTOCOL = REPO_ROOT / "gen3_protocol.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Non-scientific smoke budget; output is marked quick and cannot aggregate.",
    )
    return parser.parse_args(argv)


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_text(payload: Any) -> str:
    return json.dumps(
        _json_safe(payload),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(payload: Any, path: Path) -> None:
    write_text_atomic(_json_text(payload), path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _markdown_table(frame: pd.DataFrame, maximum_rows: int = 40) -> str:
    view = frame.head(maximum_rows).copy()
    if view.empty:
        return "_No rows._"
    headers = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}" if np.isfinite(value) else "NA")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(result: Any, *, protocol_sha256: str, validation: dict[str, Any]) -> str:
    leaderboard = result.per_seed_metrics.sort_values(
        "equal_extractant_macro_mae", kind="stable"
    )[
        [
            "arm",
            "equal_extractant_macro_mae",
            "pooled_micro_mae",
            "pooled_r2",
            "extractants_improved_vs_A2",
        ]
    ]
    h1 = leaderboard[leaderboard["arm"].str.startswith("H1_")]
    h2 = leaderboard[leaderboard["arm"].str.startswith("H2_")]
    h3 = leaderboard[leaderboard["arm"].str.startswith("H3_")]
    lambdas = result.selected_lambdas[
        ["outer_fold", "arm", "lambda", "correction_kind"]
    ]
    return "\n".join(
        [
            "# Primary leaderboard",
            "",
            "Primary selection metric: **equal-extractant macro MAE**. Each held-out "
            "extractant has one vote. Lower is better.",
            "",
            _markdown_table(leaderboard),
            "",
            "Pooled MAE and pooled R2 in this report are row-weighted descriptive "
            "statistics and are never used for champion selection.",
            "",
            "## Split-seed robustness",
            "",
            f"This task evaluates split seed `{result.run_metadata['split_seed']}` and model "
            f"seed `{result.run_metadata['model_seed']}`. Multi-split robustness is computed "
            "only by the aggregate command after all five primary split tasks validate.",
            "",
            "## H1 - learner and weighting",
            "",
            _markdown_table(h1),
            "",
            "## H2 - E3 residual signal",
            "",
            _markdown_table(h2),
            "",
            "Selected nested lambdas (zero is the automatic Stage-1 fallback):",
            "",
            _markdown_table(lambdas),
            "",
            "Every correction residual was built from a Stage-1 out-of-fold prediction. "
            "Real and shuffled arms use the same model-selection procedure; E3 shuffling "
            "touches training rows only.",
            "",
            "## H3 - antisymmetric latent-difference model",
            "",
            _markdown_table(h3),
            "",
            "H3 has no pair head: every prediction is exactly `g(A) - g(B)`.",
            "",
            "## Negative controls",
            "",
            _markdown_table(
                result.paired_bootstrap[
                    result.paired_bootstrap["comparison"].str.contains("SHUFFLED")
                ]
            ),
            "",
            "## Per-extractant breakdown",
            "",
            _markdown_table(
                result.per_extractant_metrics.sort_values(
                    ["arm", "mae"], ascending=[True, False]
                ),
                maximum_rows=60,
            ),
            "",
            "## Pooled-vs-macro inversion",
            "",
            "Any ranking difference between `pooled_micro_mae` and "
            "`equal_extractant_macro_mae` is expected to reflect unequal pair-row counts; "
            "only the macro ranking is eligible for the decision.",
            "",
            "## Decision",
            "",
            "No per-task champion is declared. The frozen multi-split decision rule is "
            "applied only after five primary split seeds validate. A negative result is valid.",
            "",
            f"Protocol SHA-256: `{protocol_sha256}`",
            "",
            f"Validation passed: `{validation['passed']}`",
            "",
        ]
    )


def _software() -> dict[str, Any]:
    import joblib
    import pyarrow

    try:
        import catboost

        catboost_version = catboost.__version__
    except ImportError:
        catboost_version = None
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "joblib": joblib.__version__,
        "catboost": catboost_version,
        "torch": torch_version,
        "platform": platform.platform(),
    }


def _software_contract(
    observed: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    expected = protocol["reproducibility_environment"]
    keys = (
        "python",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "pyarrow",
        "joblib",
        "torch",
        "catboost",
    )
    mismatches = {
        key: {"expected": str(expected[key]), "observed": str(observed.get(key))}
        for key in keys
        if str(observed.get(key)) != str(expected[key])
    }
    platform_matches = "Linux" in str(observed.get("platform")) and "x86_64" in str(
        observed.get("platform")
    )
    if not platform_matches:
        mismatches["platform"] = {
            "expected": str(expected["operating_system"]),
            "observed": str(observed.get("platform")),
        }
    return {"passed": not mismatches, "mismatches": mismatches}


def _implementation_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "scripts" / "aggregate_gen3_runs.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "gen3_protocol.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "gen3_data.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "gen3_models.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "gen3_metrics.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "gen3_evaluation.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "pairs.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "evaluation.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "ablation.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "feature_registry.py",
        REPO_ROOT / "src" / "lanthanide_separation" / "electronic.py",
    )
    return {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in paths
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs cannot be zero.")
    protocol_path = args.protocol.expanduser().resolve()
    protocol = load_gen3_protocol(protocol_path)
    run_role = validate_seed_pair(
        protocol, split_seed=args.split_seed, model_seed=args.model_seed
    )
    frozen_hashes = verify_frozen_inputs(protocol, REPO_ROOT)
    software = _software()
    software_contract = _software_contract(software, protocol)
    if not args.quick and not software_contract["passed"]:
        raise SystemExit(
            "Scientific gen3 execution requires the exact frozen software/platform "
            f"contract: {software_contract['mismatches']}"
        )
    dataset_path = (
        args.dataset.expanduser().resolve()
        if args.dataset is not None
        else REPO_ROOT / protocol["immutable_inputs"]["dataset_path"]
    )
    if file_sha256(dataset_path) != protocol["immutable_inputs"]["dataset_sha256"]:
        raise SystemExit("The requested dataset does not match the frozen protocol hash.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_text_atomic("Generation-3 task has not completed.\n", output_dir / "_INCOMPLETE")
    protocol_sha256 = file_sha256(protocol_path)
    shutil.copyfile(protocol_path, output_dir / "gen3_protocol.json")
    run_config = {
        "status": "incomplete",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            "dataset": str(dataset_path),
            "output_dir": str(output_dir),
            "split_seed": int(args.split_seed),
            "model_seed": int(args.model_seed),
            "n_jobs": int(args.n_jobs),
            "quick": bool(args.quick),
        },
        "run_role": run_role,
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": _implementation_hashes(),
        "software": software,
        "software_contract": software_contract,
        "host": socket.gethostname(),
    }
    write_json_atomic(run_config, output_dir / "run_config.json")

    source = pd.read_parquet(dataset_path)
    pair_data = build_lanthanide_pair_dataset(
        source,
        pair_scope="all",
        require_geometry=True,
        replicate_policy="unique",
        quarantine_known_bad=True,
        quarantine_known_censored_targets=False,
        require_complete_conditions=True,
        delta3d_feature_set="compact-invariant",
        include_symmetric_3d=False,
        include_pair_response_3d=False,
        include_electronic=True,
    )
    expected_inputs = protocol["immutable_inputs"]
    if pair_data.audit["cohort_sha256"] != expected_inputs["pair_cohort_sha256"]:
        raise RuntimeError("Generated pair cohort differs from the frozen gen2 cohort.")
    if len(pair_data.frame) != int(expected_inputs["pair_rows"]):
        raise RuntimeError("Generated pair row count differs from the frozen protocol.")
    registry = build_feature_registry(pair_data)
    reference_registry = REPO_ROOT / expected_inputs["gen2_feature_registry_path"]
    adapter = build_gen3_feature_adapter(
        pair_data,
        registry,
        source,
        reference_feature_registry=reference_registry,
    )
    if len(adapter.a2_columns) != int(expected_inputs["a2_feature_count"]):
        raise RuntimeError("A2 feature count differs from the frozen protocol.")
    if len(adapter.e3_raw_columns) != int(expected_inputs["e3_raw_feature_count"]):
        raise RuntimeError("E3 feature count differs from the frozen protocol.")

    write_json_atomic(pair_data.audit, output_dir / "pair_build_audit.json")
    write_csv_atomic(pair_data.quarantine, output_dir / "quarantined_source_rows.csv")
    write_json_atomic(
        {
            **frozen_hashes,
            "pair_cohort_sha256": pair_data.audit["cohort_sha256"],
            "protocol_sha256": protocol_sha256,
        },
        output_dir / "dataset_hashes.json",
    )
    write_json_atomic(
        {
            **adapter.hashes,
            "a2_feature_count": len(adapter.a2_columns),
            "e3_raw_feature_count": len(adapter.e3_raw_columns),
            "e3_compact_feature_count": len(adapter.e3_compact_columns),
            "generated_feature_contract_sha256": registry.feature_contract_sha256,
            "frozen_gen2_feature_contract_sha256": expected_inputs[
                "gen2_feature_contract_sha256"
            ],
            "frozen_gen2_feature_registry_file_sha256": expected_inputs[
                "gen2_feature_registry_file_sha256"
            ],
        },
        output_dir / "feature_hashes.json",
    )

    result = run_gen3_benchmark(
        adapter,
        protocol,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        run_role=run_role,
        n_jobs=args.n_jobs,
        quick=args.quick,
    )
    tables = {
        "oof_predictions.csv": result.predictions,
        "split_assignments.csv": result.split_assignments,
        "inner_fold_assignments.csv": result.inner_fold_assignments,
        "inner_cv_tuning.csv": result.tuning_results,
        "training_weight_audit.csv": result.weight_audit,
        "residual_crossfit_audit.csv": result.residual_audit,
        "shuffle_audit.csv": result.shuffle_audit,
        "e3_stability.csv": result.e3_stability,
        "per_seed_metrics.csv": result.per_seed_metrics,
        "per_extractant_metrics.csv": result.per_extractant_metrics,
        "extractant_family_metrics.csv": result.family_metrics,
        "paired_bootstrap.csv": result.paired_bootstrap,
        "selected_stage1.csv": result.selected_stage1,
        "selected_lambdas.csv": result.selected_lambdas,
    }
    for name, table in tables.items():
        write_csv_atomic(table, output_dir / name)
    write_json_atomic(result.leakage_audit, output_dir / "leakage_audit.json")
    write_json_atomic(result.run_metadata, output_dir / "run_metadata.json")

    weights_valid = bool(
        np.allclose(result.weight_audit["mean_weight_before_swap"], 1.0)
    )
    expected_arms = set(result.run_metadata["arms"])
    observed_arms = set(result.per_seed_metrics["arm"])
    validation = {
        "passed": bool(
            result.leakage_audit["passed"]
            and result.leakage_audit["stage1_residuals_out_of_fold_only"]
            and weights_valid
            and expected_arms == observed_arms
            and not result.predictions.filter(like="prediction_").isna().any().any()
        ),
        "quick": bool(args.quick),
        "scientific_result_eligible": not bool(args.quick),
        "software_contract_passed": bool(software_contract["passed"]),
        "software_contract_enforced_for_scientific_run": True,
        "run_role": run_role,
        "protocol_sha256": protocol_sha256,
        "dataset_sha256": frozen_hashes["dataset_sha256"],
        "pair_cohort_sha256": pair_data.audit["cohort_sha256"],
        "expected_arms": sorted(expected_arms),
        "observed_arms": sorted(observed_arms),
        "weights_mean_one": weights_valid,
        "leakage_audit_passed": bool(result.leakage_audit["passed"]),
        "stage1_residuals_out_of_fold_only": bool(
            result.leakage_audit["stage1_residuals_out_of_fold_only"]
        ),
        "pooled_metrics_used_for_selection": False,
        "errors": [],
    }
    if not validation["passed"]:
        validation["errors"].append("One or more fail-closed result checks failed.")
    write_json_atomic(validation, output_dir / "validation.json")
    write_text_atomic(
        build_report(result, protocol_sha256=protocol_sha256, validation=validation),
        output_dir / "report.md",
    )
    if not validation["passed"]:
        return 2

    run_config["status"] = "complete"
    run_config["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(run_config, output_dir / "run_config.json")
    artifact_names = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"_INCOMPLETE", "artifact_hashes.json"}
    )
    artifact_hashes = {name: file_sha256(output_dir / name) for name in artifact_names}
    write_json_atomic(artifact_hashes, output_dir / "artifact_hashes.json")
    (output_dir / "_INCOMPLETE").unlink()
    write_json_atomic(
        {
            "status": "complete",
            "quick": bool(args.quick),
            "run_role": run_role,
            "protocol_sha256": protocol_sha256,
            "validation_sha256": file_sha256(output_dir / "validation.json"),
            "artifact_hashes_sha256": file_sha256(output_dir / "artifact_hashes.json"),
            "artifact_count": len(artifact_hashes),
        },
        output_dir / "_SUCCESS.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
