#!/usr/bin/env python3
"""Run the leakage-safe adjacent-lanthanide Delta3D benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lanthanide_separation.evaluation import (  # noqa: E402
    DEFAULT_PARAMETER_GRID,
    nested_group_benchmark,
)
from lanthanide_separation.pairs import build_adjacent_pair_dataset  # noqa: E402


DEFAULT_DATASET = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"


def _default_n_jobs() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value is None:
        return -1
    try:
        parsed = int(value)
    except ValueError:
        return -1
    return parsed if parsed > 0 else -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Model/forest and bootstrap seed; independent of --split-seed.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for shuffled outer and inner held-out-group assignments.",
    )
    parser.add_argument("--n-jobs", type=int, default=_default_n_jobs())
    parser.add_argument(
        "--expected-dataset-sha256",
        default=None,
        help="Fail before creating a run directory if the parquet hash differs.",
    )
    parser.add_argument(
        "--delta3d-feature-set",
        choices=("compact-invariant", "all-ranked"),
        default="compact-invariant",
        help="Primary stable shell summaries or all donor-rank tabular features.",
    )
    parser.add_argument(
        "--group-mode",
        choices=("extractant", "ecfp-exact-cluster"),
        default="ecfp-exact-cluster",
        help="Strict exact-ECFP holdout (default) or literal canonical-SMILES holdout.",
    )
    parser.add_argument(
        "--replicate-policy",
        choices=("median", "unique"),
        default="unique",
        help="Aggregate repeated condition/metal cells by median or require one measurement.",
    )
    parser.add_argument(
        "--allow-incomplete-conditions",
        action="store_true",
        help="Sensitivity only: allow missing cond__ values to match each other.",
    )
    parser.add_argument(
        "--quarantine-known-censored-targets",
        action="store_true",
        help="Sensitivity only: exclude three globally identified target-extreme rows.",
    )
    parser.add_argument(
        "--no-default-quarantine",
        action="store_true",
        help="Disable the provenance-backed TODGA structure/name quarantine.",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Skip full-data deployment tuning/model; recommended for multi-seed arrays.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one fixed tree configuration; useful only for a smoke run.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise SystemExit("--outer-folds and --inner-folds must both be at least 2.")
    if args.trees < 1:
        raise SystemExit("--trees must be positive.")
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be positive.")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs cannot be zero.")
    if args.expected_dataset_sha256 is not None:
        expected = args.expected_dataset_sha256.strip().lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            raise SystemExit("--expected-dataset-sha256 must be a 64-character hex digest.")
        args.expected_dataset_sha256 = expected


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_text_atomic(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(payload: Any, path: Path) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n"
    write_text_atomic(text, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_output_dir(requested: Path | None, args: argparse.Namespace) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        group_slug = args.group_mode.replace("-", "_")
        requested = (
            REPO_ROOT
            / "runs"
            / f"{stamp}_delta3d_{group_slug}_m{args.seed}_s{args.split_seed}"
        )
    requested = requested.expanduser().resolve()
    requested.mkdir(parents=True, exist_ok=False)
    return requested


def slurm_metadata() -> dict[str, str | None]:
    names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_SUBMIT_DIR",
    )
    return {**{name: os.environ.get(name) for name in names}, "hostname": socket.gethostname()}


def main() -> int:
    args = parse_args()
    validate_args(args)
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    dataset_sha256 = sha256_file(dataset_path)
    if (
        args.expected_dataset_sha256 is not None
        and dataset_sha256 != args.expected_dataset_sha256
    ):
        raise SystemExit(
            "Dataset SHA-256 mismatch: "
            f"expected {args.expected_dataset_sha256}, observed {dataset_sha256}"
        )

    output_dir = make_output_dir(args.output_dir, args)
    started_at = datetime.now(timezone.utc).isoformat()
    group_column = "extractant" if args.group_mode == "extractant" else "ecfp_exact_cluster"
    parameter_grid = (
        ({"max_features": 0.70, "min_samples_leaf": 2},)
        if args.quick
        else DEFAULT_PARAMETER_GRID
    )
    write_json(
        {
            "status": "incomplete",
            "started_at_utc": started_at,
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "arguments": vars(args),
            "slurm": slurm_metadata(),
        },
        output_dir / "run_config.json",
    )
    write_text_atomic("Run has not completed.\n", output_dir / "_INCOMPLETE")

    print(f"Dataset: {dataset_path}", flush=True)
    print(f"Dataset SHA-256: {dataset_sha256}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print("Reading parquet...", flush=True)
    source = pd.read_parquet(dataset_path)
    pair_data = build_adjacent_pair_dataset(
        source,
        require_geometry=True,
        replicate_policy=args.replicate_policy,
        quarantine_known_bad=not args.no_default_quarantine,
        quarantine_known_censored_targets=args.quarantine_known_censored_targets,
        require_complete_conditions=not args.allow_incomplete_conditions,
        delta3d_feature_set=args.delta3d_feature_set,
    )
    print(
        f"Pairs: {len(pair_data.frame)} | extractants: "
        f"{pair_data.frame['extractant'].nunique()} | Delta3D: {len(pair_data.delta3d_columns)}",
        flush=True,
    )

    if not pair_data.quarantine.empty:
        pair_data.quarantine.to_csv(output_dir / "quarantined_source_rows.csv", index=False)
    write_json(pair_data.audit, output_dir / "pair_build_audit.json")
    write_json(
        {
            "delta3d_feature_set": args.delta3d_feature_set,
            "cohort_sha256": pair_data.audit["cohort_sha256"],
            "baseline_columns": list(pair_data.baseline_columns),
            "delta3d_columns": list(pair_data.delta3d_columns),
            "forbidden_model_inputs": [
                "D",
                "log_D",
                "safe_exp_id",
                "build_id",
                "geometry_key",
                "paths",
                "asset indices",
                "global sample weights",
            ],
        },
        output_dir / "feature_contract.json",
    )

    print("Running nested group-held-out benchmark...", flush=True)
    result = nested_group_benchmark(
        pair_data,
        group_column=group_column,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        n_estimators=args.trees,
        n_jobs=args.n_jobs,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        split_seed=args.split_seed,
        fit_final_model=not args.evaluation_only,
        parameter_grid=parameter_grid,
    )

    result.predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    result.fold_assignments.to_csv(output_dir / "fold_assignments.csv", index=False)
    result.tuning_results.to_csv(output_dir / "inner_cv_tuning.csv", index=False)
    result.per_extractant_metrics.to_csv(
        output_dir / "per_extractant_metrics.csv", index=False
    )
    result.per_pair_metrics.to_csv(output_dir / "per_pair_type_metrics.csv", index=False)
    deployment_model_saved = result.final_model is not None
    if result.final_model is not None:
        result.feature_importances.to_csv(
            output_dir / "full_forest_impurity_importances.csv", index=False
        )
        joblib.dump(result.final_model, output_dir / "delta3d_model.joblib")

    implementation_sha256 = {
        "scripts/run_delta3d_benchmark.py": sha256_file(Path(__file__).resolve()),
        "src/lanthanide_separation/pairs.py": sha256_file(
            SRC_ROOT / "lanthanide_separation" / "pairs.py"
        ),
        "src/lanthanide_separation/evaluation.py": sha256_file(
            SRC_ROOT / "lanthanide_separation" / "evaluation.py"
        ),
    }
    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "cohort_sha256": pair_data.audit["cohort_sha256"],
        "implementation_sha256": implementation_sha256,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "slurm": slurm_metadata(),
        "output_dir": str(output_dir),
        "arguments": vars(args),
        "pair_build_audit": pair_data.audit,
        "benchmark": result.summary,
        "artifacts": {"deployment_model_saved": deployment_model_saved},
    }

    # summary.json and _SUCCESS are deliberately last: aggregators can reject
    # preempted jobs without mistaking a partial directory for a completed run.
    write_json(run_manifest, output_dir / "summary.json")
    (output_dir / "_INCOMPLETE").unlink()
    write_json(
        {
            "status": "complete",
            "summary_sha256": sha256_file(output_dir / "summary.json"),
        },
        output_dir / "_SUCCESS.json",
    )

    baseline = result.summary["metrics"]["baseline"]
    control = result.summary["metrics"]["2d_ensemble"]
    delta3d = result.summary["metrics"]["delta3d"]
    improvement = result.summary["improvements"]["primary_delta3d_vs_2d_ensemble"]
    ci = result.summary["paired_group_bootstrap"]["comparisons"][
        "delta3d_vs_2d_ensemble"
    ]["r2_gain"]
    print("\nOOF results", flush=True)
    print(
        f"baseline R2={baseline['r2']:.4f}; matched 2D ensemble R2={control['r2']:.4f}; "
        f"Delta3D R2={delta3d['r2']:.4f}",
        flush=True,
    )
    print(
        f"Delta3D vs 2D-ensemble R2 gain={improvement['r2_gain']:+.4f}; "
        f"group-bootstrap 95% CI [{ci['ci95_low']:+.4f}, {ci['ci95_high']:+.4f}]",
        flush=True,
    )
    print(f"Saved: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
