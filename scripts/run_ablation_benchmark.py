#!/usr/bin/env python3
"""Run the leakage-safe all-pairs A0--A6 feature-family benchmark.

This is the auditable entry point for the scientific 2D-versus-3D ablation.  It
builds one pair cohort, freezes one outer leave-extractants-out split, and asks
``lanthanide_separation.ablation`` to reuse that split for every feature set and
negative control.  A run is complete only after every declared artifact is
written, hashed, validated, and the final ``_SUCCESS.json`` marker is emitted.
"""

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
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lanthanide_separation.cohort import build_cohort_report  # noqa: E402
from lanthanide_separation.electronic import (  # noqa: E402
    audit_electronic_sources,
)
from lanthanide_separation.feature_registry import (  # noqa: E402
    ABLATION_FAMILIES,
    EXTENSION_ABLATION_FAMILIES,
    EXTENSION_SHUFFLE_BLOCKS,
    LOCAL_3D_SUBBLOCKS,
    SENSITIVITY_2D_ARM_FAMILIES,
    SENSITIVITY_2D_ARMS,
    SENSITIVITY_2D_SHUFFLE_BLOCKS,
    build_feature_registry,
)
from lanthanide_separation.geometry_descriptors import (  # noqa: E402
    DESCRIPTOR_BLOCKS,
    DESCRIPTOR_PROFILES,
    MetalSiteDescriptorBuilder,
    attach_geometry_descriptors,
    validate_blocks,
)
from lanthanide_separation.novelty import novelty_report  # noqa: E402
from lanthanide_separation.pairs import (  # noqa: E402
    PAIR_SCOPES,
    PAIR_TARGET_COLUMN,
    build_lanthanide_pair_dataset,
)


DEFAULT_DATASET = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
DEFAULT_VR_ASSETS = (
    REPO_ROOT / "dataset with 3D structures" / "features" / "vietoris_rips_inputs.npz"
)
DEFAULT_SHUFFLE_SEEDS: tuple[int, ...] = (1009, 2017, 3019)

TABLE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("predictions", "oof_predictions.csv"),
    ("fold_metrics", "fold_metrics.csv"),
    ("fold_assignments", "fold_assignments.csv"),
    ("fold_memberships", "fold_memberships.csv"),
    ("inner_fold_assignments", "inner_fold_assignments.csv"),
    ("tuning_results", "inner_cv_tuning.csv"),
    ("preprocessing_parameters", "preprocessing_parameters.csv"),
    ("selected_features", "selected_features.csv"),
    ("shuffle_audit", "shuffle_audit.csv"),
    ("per_ablation_metrics", "per_ablation_metrics.csv"),
    ("paired_deltas", "paired_ablation_deltas.csv"),
    ("per_extractant_comparison", "per_extractant_comparison.csv"),
    ("per_lanthanide_comparison", "per_lanthanide_comparison.csv"),
)

JSON_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("preprocessing_audit", "preprocessing_audit.json"),
    ("geometry_qc_summary", "geometry_qc_summary.json"),
    ("leakage_audit", "leakage_audit.json"),
    ("feature_audit", "feature_audit.json"),
)


def _default_n_jobs() -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw is None:
        return -1
    try:
        value = int(raw)
    except ValueError:
        return -1
    return value if value > 0 else -1


def _parse_int_list(raw: str, *, flag: str) -> tuple[int, ...]:
    text = str(raw).strip().lower()
    if text in {"", "none", "off"}:
        return ()
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        try:
            value = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"{flag} must be 'none' or comma-separated integers."
            ) from error
        if value < 0:
            raise argparse.ArgumentTypeError(f"{flag} seeds must be nonnegative.")
        values.append(value)
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{flag} seeds must be unique.")
    return tuple(values)


def _parse_descriptor_blocks(raw: str) -> tuple[str, ...]:
    text = str(raw).strip().lower()
    if text in {"", "none", "off"}:
        return ()
    try:
        return validate_blocks(part.strip() for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--pair-scope",
        choices=PAIR_SCOPES,
        default="all",
        help=(
            "Use every observed condition-matched lanthanide pair (default), or "
            "the historical adjacent-only sensitivity cohort."
        ),
    )
    parser.add_argument(
        "--group-mode",
        choices=("extractant", "ecfp-exact-cluster"),
        default="extractant",
        help=(
            "Outer and inner CV group. Literal extractant holdout is the primary "
            "protocol; exact-ECFP clusters are a stricter sensitivity."
        ),
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--model-seed",
        "--seed",
        dest="model_seed",
        type=int,
        default=42,
        help="Fixed estimator/bootstrap seed (alias: --seed).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Fixed seed used once to create the shared grouped folds.",
    )
    parser.add_argument(
        "--shuffle-seeds",
        default=",".join(str(seed) for seed in DEFAULT_SHUFFLE_SEEDS),
        help=(
            "Comma-separated training-only 3D permutation seeds, or 'none'. "
            "Quick mode retains only the first requested seed."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=_default_n_jobs())
    parser.add_argument(
        "--delta3d-feature-set",
        choices=("compact-invariant", "all-ranked"),
        default="compact-invariant",
    )
    parser.add_argument("--vr-assets", type=Path, default=DEFAULT_VR_ASSETS)
    parser.add_argument(
        "--geometry-descriptor-blocks",
        default="global_shape,coordination_shape",
        help=(
            "Comma-separated VR-derived blocks or 'none'. The pre-specified exact "
            "coordinate-only default is global_shape,coordination_shape; "
            "ligand_field and enclosure are explicit opt-ins. "
            f"Available: {','.join(DESCRIPTOR_BLOCKS)}."
        ),
    )
    parser.add_argument(
        "--descriptor-profile",
        choices=DESCRIPTOR_PROFILES,
        default="core",
    )
    parser.add_argument(
        "--no-block-ablations",
        action="store_true",
        help="Skip exploratory A2+D1...D5 models; main A0--A6 still run.",
    )
    parser.add_argument(
        "--reference-baselines",
        action="store_true",
        help=(
            "Add the trivial (antisymmetric constant) and regularized linear "
            "reference arms B0/B1/B2 on the same folds."
        ),
    )
    parser.add_argument(
        "--symmetric-3d",
        action="store_true",
        help=(
            "Emit sym3d__X = (X_A + X_B)/2 and evaluate the declared A5s/A6s "
            "arms in addition to -- never instead of -- A5/A6. The delta-only "
            "contract cancels the absolute coordination environment; this "
            "restores it while preserving exact A/B antisymmetry."
        ),
    )
    parser.add_argument(
        "--pair-response-3d",
        action="store_true",
        help=(
            "Emit the declared pair3d__ block (relative, magnitude, "
            "ionic-radius normalised and excess forms of the metal-substitution "
            "response of the coordination shell)."
        ),
    )
    parser.add_argument(
        "--electronic",
        action="store_true",
        help=(
            "Emit the declared elec__ block of complex-level and pair-response "
            "xTB quantities. Never added to A0--A6; it forms its own arms."
        ),
    )
    parser.add_argument(
        "--extension-arms",
        action="store_true",
        help=(
            "Evaluate the second-generation G/E/C ladder on the same folds as "
            "the pre-specified arms. Requires the matching feature blocks."
        ),
    )
    parser.add_argument(
        "--two-d-sensitivity",
        action="store_true",
        help=(
            "Evaluate the secondary S1..S5 2D-representation ladder (RDKit-only "
            "and ECFP-only baselines, and the geometry/electronic blocks re-added "
            "on top of the RDKit-only baseline) on the same folds. Diagnostic "
            "for ECFP redundancy; it never modifies A2 or the primary claim."
        ),
    )
    parser.add_argument(
        "--extension-shuffle-seeds",
        default="",
        help=(
            "Comma-separated seeds for the block-permutation negative controls "
            "of the extension arms (G2/E2/E3). Empty disables them."
        ),
    )
    parser.add_argument(
        "--prespecified-arms",
        default="",
        help=(
            "Comma-separated subset of A0..A6 to evaluate; A2 is mandatory. "
            "Empty runs the whole pre-specified ladder. Subsetting only limits "
            "which frozen arms this run recomputes; it never changes them."
        ),
    )
    parser.add_argument(
        "--replicate-policy",
        choices=("median", "unique"),
        default="unique",
    )
    parser.add_argument("--allow-incomplete-conditions", action="store_true")
    parser.add_argument("--quarantine-known-censored-targets", action="store_true")
    parser.add_argument("--no-default-quarantine", action="store_true")
    parser.add_argument("--expected-dataset-sha256", default=None)
    parser.add_argument("--expected-vr-sha256", default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke mode: one fixed hyperparameter candidate, at most 48 trees, "
            "100 bootstraps, and one shuffle seed."
        ),
    )
    return parser.parse_args(argv)


def _validate_digest(value: str | None, flag: str) -> str | None:
    if value is None:
        return None
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit(f"{flag} must be a 64-character hexadecimal digest.")
    return digest


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise SystemExit("--outer-folds and --inner-folds must both be at least 2.")
    if args.trees < 1:
        raise SystemExit("--trees must be positive.")
    if args.bootstrap < 1:
        raise SystemExit("--bootstrap must be positive.")
    if args.model_seed < 0 or args.split_seed < 0:
        raise SystemExit("--model-seed and --split-seed must be nonnegative.")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs cannot be zero.")
    try:
        args.shuffle_seed_values = _parse_int_list(
            args.shuffle_seeds, flag="--shuffle-seeds"
        )
        args.extension_shuffle_seed_values = _parse_int_list(
            args.extension_shuffle_seeds, flag="--extension-shuffle-seeds"
        )
        args.descriptor_blocks = _parse_descriptor_blocks(
            args.geometry_descriptor_blocks
        )
    except argparse.ArgumentTypeError as error:
        raise SystemExit(str(error)) from error
    args.prespecified_arm_values = tuple(
        item.strip()
        for item in str(args.prespecified_arms).split(",")
        if item.strip()
    )
    if args.prespecified_arm_values:
        unknown = [
            arm for arm in args.prespecified_arm_values if arm not in ABLATION_FAMILIES
        ]
        if unknown:
            raise SystemExit(
                f"--prespecified-arms accepts only {tuple(ABLATION_FAMILIES)}; "
                f"got {unknown}."
            )
        if "A2" not in args.prespecified_arm_values:
            raise SystemExit("--prespecified-arms must include A2, the reference arm.")
    if args.extension_arms and not (args.pair_response_3d or args.electronic):
        raise SystemExit(
            "--extension-arms requires --pair-response-3d and/or --electronic."
        )
    if args.extension_shuffle_seed_values and not args.extension_arms:
        raise SystemExit("--extension-shuffle-seeds requires --extension-arms.")
    if args.two_d_sensitivity and args.prespecified_arm_values:
        # S1..S5 are defined relative to the full A2 column set; subsetting the
        # frozen ladder would leave the sensitivity contrast without its
        # reference.
        raise SystemExit(
            "--two-d-sensitivity cannot be combined with --prespecified-arms."
        )
    args.expected_dataset_sha256 = _validate_digest(
        args.expected_dataset_sha256, "--expected-dataset-sha256"
    )
    args.expected_vr_sha256 = _validate_digest(
        args.expected_vr_sha256, "--expected-vr-sha256"
    )
    if args.expected_vr_sha256 is not None and not args.descriptor_blocks:
        raise SystemExit(
            "--expected-vr-sha256 requires at least one --geometry-descriptor-blocks."
        )
    if args.quick:
        args.trees = min(args.trees, 48)
        args.bootstrap = min(args.bootstrap, 100)
        args.shuffle_seed_values = args.shuffle_seed_values[:1]
        args.extension_shuffle_seed_values = args.extension_shuffle_seed_values[:1]
    return args


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sanitize_json(value: Any) -> Any:
    """Recursively replace non-finite/native pandas scalars before encoding."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_sanitize_json(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _sanitize_json(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _sanitize_json(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _json_text(payload: Any) -> str:
    return json.dumps(
        _sanitize_json(payload),
        indent=2,
        ensure_ascii=False,
        sort_keys=False,
        allow_nan=False,
        default=_json_default,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_object(value: Any) -> str:
    payload = json.dumps(
        _sanitize_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scientific_protocol_payload(
    args: argparse.Namespace,
    parameter_grid: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Canonical non-seed scientific contract shared by all model-seed runs."""

    return {
        "schema_version": "1.0",
        "primary_comparison": "A2_vs_A5",
        "secondary_comparison": "A2_vs_A6",
        "substitution_comparison": "A3_vs_A2",
        "primary_metric": "mae",
        "primary_inference_statistic": "equal-extractant macro delta MAE",
        "bootstrap_unit": "held-out extractant",
        "pair_scope": args.pair_scope,
        "group_mode": args.group_mode,
        "outer_folds": int(args.outer_folds),
        "inner_folds": int(args.inner_folds),
        "split_seed": int(args.split_seed),
        "trees": int(args.trees),
        "parameter_grid": [dict(item) for item in parameter_grid],
        "bootstrap_replicates": int(args.bootstrap),
        "shuffle_seeds": [int(value) for value in args.shuffle_seed_values],
        "extension_shuffle_seeds": [
            int(value) for value in args.extension_shuffle_seed_values
        ],
        "prespecified_arms": list(args.prespecified_arm_values) or list(ABLATION_FAMILIES),
        "include_pair_response_3d": bool(args.pair_response_3d),
        "include_electronic": bool(args.electronic),
        "include_extension_arms": bool(args.extension_arms),
        "include_2d_sensitivity_arms": bool(args.two_d_sensitivity),
        "include_symmetric_3d": bool(args.symmetric_3d),
        "include_reference_baselines": bool(args.reference_baselines),
        "delta3d_feature_set": args.delta3d_feature_set,
        "geometry_descriptor_blocks": list(args.descriptor_blocks),
        "descriptor_profile": args.descriptor_profile,
        "include_block_ablations": not bool(args.no_block_ablations),
        "include_donor_composition_counts": True,
        "replicate_policy": args.replicate_policy,
        "require_complete_conditions": not bool(args.allow_incomplete_conditions),
        "quarantine_known_censored_targets": bool(
            args.quarantine_known_censored_targets
        ),
        "default_quarantine_enabled": not bool(args.no_default_quarantine),
        "quick_mode": bool(args.quick),
        "outer_split_generated_once": True,
        "outer_test_groups_unseen_in_inner_cv": True,
        "model_seed_is_the_only_varying_training_field": True,
    }


def _extension_arm_is_buildable(arm: str, args: argparse.Namespace) -> bool:
    """True when this run emitted every feature family the arm needs."""

    if arm == "G4":
        return bool(args.descriptor_blocks)
    families = set(EXTENSION_ABLATION_FAMILIES[arm])
    if "3D_PAIR_RESPONSE" in families and not args.pair_response_3d:
        return False
    if {"ELEC_COMPLEX", "ELEC_PAIR"} & families and not args.electronic:
        return False
    return True


def _sensitivity_arm_is_buildable(arm: str, args: argparse.Namespace) -> bool:
    """True when this run emitted every feature family the S-arm needs."""

    families = set(SENSITIVITY_2D_ARM_FAMILIES.get(arm, ()))
    if "3D_PAIR_RESPONSE" in families and not args.pair_response_3d:
        return False
    if "ELEC_PAIR" in families and not args.electronic:
        return False
    return True


def _as_frame(value: Any, *, field_name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.rename(field_name).reset_index()
    if isinstance(value, (list, tuple)):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame([value])
    raise TypeError(
        f"Evaluator result field {field_name!r} must be table-like; "
        f"received {type(value).__name__}."
    )


def make_output_dir(requested: Path | None, args: argparse.Namespace) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        group_slug = args.group_mode.replace("-", "_")
        requested = REPO_ROOT / "runs" / (
            f"ablation_{stamp}_{args.pair_scope}_{group_slug}_"
            f"m{args.model_seed}_s{args.split_seed}"
        )
    output = requested.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def _slurm_metadata() -> dict[str, str | None]:
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
    return {
        **{name: os.environ.get(name) for name in names},
        "hostname": socket.gethostname(),
    }


def _software_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def _display_float(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not np.isfinite(numeric) else f"{numeric:.6f}"


def _markdown_table(frame: pd.DataFrame, *, maximum_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows available._"
    shown = frame.head(maximum_rows).copy()
    shown = shown.map(_display_float)
    headers = [str(column) for column in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    if len(frame) > maximum_rows:
        lines.append(f"\n_Only the first {maximum_rows} of {len(frame)} rows are shown._")
    return "\n".join(lines)


def _block_availability(registry: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block in LOCAL_3D_SUBBLOCKS:
        columns = registry.columns_for_local_3d_subblock(block)
        rows.append(
            {
                "block": block,
                "column_count": len(columns),
                "status": "evaluated_if_enabled" if columns else "unavailable_skipped",
            }
        )
    return pd.DataFrame(rows)


def _long_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize evaluator-wide predictions to one auditable row per arm/pair."""

    prediction_columns = [
        str(column)
        for column in predictions.columns
        if str(column).startswith("prediction_")
    ]
    if not prediction_columns:
        raise ValueError("Evaluator predictions contain no prediction_<ablation> columns.")
    target_column = "log_SF_A_over_B"
    if target_column not in predictions:
        raise ValueError(f"Evaluator predictions are missing target column {target_column!r}.")
    identity_columns = [
        column for column in predictions.columns if column not in prediction_columns
    ]
    long = predictions.melt(
        id_vars=identity_columns,
        value_vars=prediction_columns,
        var_name="ablation",
        value_name="y_pred",
    )
    long["ablation"] = long["ablation"].str.removeprefix("prediction_")
    long = long.rename(columns={target_column: "y_true"})
    preferred = [
        "pair_id",
        "condition_id",
        "extractant",
        "ecfp_exact_cluster",
        "pair_label",
        "metal_A",
        "metal_B",
        "source_id_A",
        "source_id_B",
        "geometry_key_A",
        "geometry_key_B",
        "geometry_qc_class_A",
        "geometry_qc_class_B",
        "geometry_qc_pair",
        "outer_fold",
        "ablation",
        "y_true",
        "y_pred",
    ]
    ordered = [column for column in preferred if column in long]
    ordered.extend(column for column in long.columns if column not in ordered)
    result = long.loc[:, ordered].sort_values(
        ["ablation", "outer_fold", "pair_id"], kind="stable"
    )
    if result[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("Normalized OOF predictions contain missing target/prediction values.")
    return result.reset_index(drop=True)


def _selected_features_table(registry: Any, feature_audit: dict[str, Any]) -> pd.DataFrame:
    assignments = {
        assignment.column: assignment for assignment in registry.assignments
    }
    evaluated = feature_audit.get("evaluated_feature_sets", {})
    rows: list[dict[str, Any]] = []
    for ablation, contract in evaluated.items():
        columns = tuple(str(column) for column in contract.get("columns", ()))
        for position, column in enumerate(columns):
            assignment = assignments.get(column)
            if assignment is None:
                raise ValueError(
                    f"Evaluator selected feature {column!r} absent from feature registry."
                )
            rows.append(
                {
                    "ablation": str(ablation),
                    "feature_position": int(position),
                    "feature": column,
                    "family": assignment.family,
                    "local_3d_subblock": assignment.local_3d_subblock,
                    "selection_method": "pre_specified_registry_contract",
                }
            )
    if not rows:
        raise ValueError("Evaluator feature audit contains no evaluated feature sets.")
    return pd.DataFrame(rows)


def _a2_a5_comparison(
    metrics: pd.DataFrame,
    *,
    id_column: str,
    output_id_column: str,
) -> pd.DataFrame:
    """Convert long per-group metrics into the requested paired A2/A5 table."""

    required = {id_column, "ablation", "mae", "rmse", "spearman"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Per-group metrics are missing columns: {missing}")
    a2 = metrics[metrics["ablation"].eq("A2")].copy()
    a5 = metrics[metrics["ablation"].eq("A5")].copy()
    if a2.empty or a5.empty:
        raise ValueError("Per-group metrics must contain both A2 and A5 rows.")
    count_column = next(
        (
            column
            for column in ("n_pair_memberships", "n_rows", "n_samples", "n")
            if column in metrics
        ),
        None,
    )
    columns = [id_column, "mae", "rmse", "spearman"]
    if count_column is not None:
        columns.append(count_column)
    merged = a2.loc[:, columns].merge(
        a5.loc[:, columns],
        on=id_column,
        how="inner",
        validate="one_to_one",
        suffixes=("_A2", "_A5"),
    )
    result = pd.DataFrame({output_id_column: merged[id_column].astype(str)})
    if count_column is not None:
        left_count = merged[f"{count_column}_A2"].to_numpy(dtype=int)
        right_count = merged[f"{count_column}_A5"].to_numpy(dtype=int)
        if not np.array_equal(left_count, right_count):
            raise ValueError("A2 and A5 group sample counts differ unexpectedly.")
        result["n_samples"] = left_count
    result["MAE_2D"] = merged["mae_A2"].to_numpy(dtype=float)
    result["MAE_2D3D"] = merged["mae_A5"].to_numpy(dtype=float)
    result["delta_MAE"] = result["MAE_2D"] - result["MAE_2D3D"]
    result["RMSE_2D"] = merged["rmse_A2"].to_numpy(dtype=float)
    result["RMSE_2D3D"] = merged["rmse_A5"].to_numpy(dtype=float)
    result["delta_RMSE"] = result["RMSE_2D"] - result["RMSE_2D3D"]
    result["Spearman_2D"] = merged["spearman_A2"].to_numpy(dtype=float)
    result["Spearman_2D3D"] = merged["spearman_A5"].to_numpy(dtype=float)
    result["delta_Spearman"] = result["Spearman_2D3D"] - result["Spearman_2D"]
    return result.sort_values(output_id_column, kind="stable").reset_index(drop=True)


def prepare_artifact_tables(result: Any, registry: Any) -> dict[str, pd.DataFrame]:
    """Map evaluator-native fields to the stable per-run CSV contract."""

    per_extractant = _as_frame(
        result.per_extractant_metrics, field_name="per_extractant_metrics"
    )
    per_lanthanide = _as_frame(
        result.per_lanthanide_metrics, field_name="per_lanthanide_metrics"
    )
    shuffle_audit = _as_frame(result.shuffle_audit, field_name="shuffle_audit")
    if shuffle_audit.empty and not len(shuffle_audit.columns):
        shuffle_audit = pd.DataFrame(
            columns=(
                "ablation",
                "outer_fold",
                "inner_fold",
                "scope",
                "shuffle_seed",
                "test_rows_touched",
            )
        )
    return {
        "predictions": _long_oof_predictions(
            _as_frame(result.predictions, field_name="predictions")
        ),
        "fold_metrics": _as_frame(result.fold_metrics, field_name="fold_metrics"),
        "fold_assignments": _as_frame(
            result.fold_assignments, field_name="fold_assignments"
        ),
        "fold_memberships": _as_frame(
            result.fold_memberships, field_name="fold_memberships"
        ),
        "inner_fold_assignments": _as_frame(
            result.inner_fold_assignments, field_name="inner_fold_assignments"
        ),
        "tuning_results": _as_frame(
            result.tuning_results, field_name="tuning_results"
        ),
        "preprocessing_parameters": _as_frame(
            result.preprocessing_audit, field_name="preprocessing_audit"
        ),
        "selected_features": _selected_features_table(registry, result.feature_audit),
        "shuffle_audit": shuffle_audit,
        "per_ablation_metrics": _as_frame(
            result.per_ablation_metrics, field_name="per_ablation_metrics"
        ),
        "paired_deltas": _as_frame(result.paired_deltas, field_name="paired_deltas"),
        "per_extractant_comparison": _a2_a5_comparison(
            per_extractant,
            id_column="extractant_id",
            output_id_column="extractant_id",
        ),
        "per_lanthanide_comparison": _a2_a5_comparison(
            per_lanthanide,
            id_column="lanthanide",
            output_id_column="Ln",
        ),
    }


def metrics_payload(result: Any, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Machine-readable aggregate, fold, paired, and bootstrap metric contract."""

    return {
        "primary_metric": "mae",
        "oof_metrics": tables["per_ablation_metrics"].to_dict(orient="records"),
        "fold_metrics": tables["fold_metrics"].to_dict(orient="records"),
        "paired_fold_deltas": tables["paired_deltas"].to_dict(orient="records"),
        "paired_delta_summary": result.summary.get("paired_delta_summary", {}),
        "paired_group_bootstrap": result.summary.get("paired_group_bootstrap", {}),
        "aggregate_r2_note": (
            "OOF R2 is computed from concatenated cross-fitted predictions, not from "
            "the mean of outer-fold R2 values."
        ),
    }


def _scientific_answer_lines(
    tables: dict[str, pd.DataFrame],
    *,
    result: Any | None,
) -> list[str]:
    """Answer the pre-specified report questions from this run without spin."""

    metrics = tables["per_ablation_metrics"]
    if "ablation" not in metrics:
        return ["The evaluator did not emit an ablation label; conclusions are unavailable."]
    lookup = {
        str(row["ablation"]): row
        for row in metrics.to_dict(orient="records")
    }
    a2 = lookup.get("A2")
    a5 = lookup.get("A5")
    a6 = lookup.get("A6")
    if a2 is None or a5 is None:
        return ["A2 or A5 is absent; the primary scientific comparison is unavailable."]

    absolute = float(a2["mae"]) - float(a5["mae"])
    relative = 100.0 * absolute / float(a2["mae"]) if float(a2["mae"]) != 0.0 else np.nan
    bootstrap_interval: dict[str, Any] = {}
    if result is not None:
        bootstrap_interval = (
            result.summary.get("paired_group_bootstrap", {})
            .get("comparisons", {})
            .get("A2_vs_A5", {})
            .get("delta_macro_group_mae", {})
        )
    ci_low = bootstrap_interval.get("ci95_low")
    ci_high = bootstrap_interval.get("ci95_high")
    if ci_low is not None and ci_high is not None and float(ci_low) > 0.0:
        conclusion = (
            "supported by a positive equal-extractant paired-bootstrap 95% interval"
        )
    elif ci_low is not None and ci_high is not None and float(ci_high) < 0.0:
        conclusion = (
            "evidence favors A2; the equal-extractant paired interval is entirely "
            "negative"
        )
    elif ci_low is not None and ci_high is not None:
        conclusion = (
            "inconclusive; the equal-extractant paired 95% interval includes zero"
        )
    else:
        conclusion = "not significance-assessable from this run's bootstrap output"

    extractants = tables["per_extractant_comparison"]
    extractant_wins = int((extractants["delta_MAE"] > 0.0).sum()) if len(extractants) else 0
    extractant_fraction = (
        100.0 * extractant_wins / len(extractants) if len(extractants) else np.nan
    )
    macro_extractant_delta = (
        float(pd.to_numeric(extractants["delta_MAE"], errors="coerce").mean())
        if len(extractants)
        else np.nan
    )

    local_blocks: list[tuple[str, float]] = []
    for name, row in lookup.items():
        if name.startswith("A2+D") and name != "A2+D1-D5":
            local_blocks.append((name, float(a2["mae"]) - float(row["mae"])))
    local_blocks.sort(key=lambda item: item[1], reverse=True)
    block_answer = (
        ", ".join(f"{name}: delta_MAE={gain:+.6f}" for name, gain in local_blocks)
        if local_blocks
        else "no non-empty D1-D5 exploratory blocks were evaluated"
    )

    shuffled = [
        (name, float(a2["mae"]) - float(row["mae"]))
        for name, row in lookup.items()
        if name.startswith("A5_SHUFFLED")
    ]
    shuffle_answer = (
        ", ".join(f"{name}: delta_MAE={gain:+.6f}" for name, gain in shuffled)
        if shuffled
        else "shuffle control disabled"
    )

    qc = result.geometry_qc_summary if result is not None else {}
    if qc.get("high_vs_low_estimable"):
        qc_answer = "both high- and low-confidence strata are estimable; see QC table"
    else:
        qc_answer = (
            "not estimable because the accepted common cohort does not contain both "
            "quality strata"
        )

    lanthanides = tables["per_lanthanide_comparison"]
    top_lanthanides = (
        lanthanides.nlargest(min(3, len(lanthanides)), "delta_MAE")
        if len(lanthanides)
        else lanthanides
    )
    lanthanide_answer = (
        ", ".join(
            f"{row.Ln} ({float(row.delta_MAE):+.6f})"
            for row in top_lanthanides.itertuples(index=False)
        )
        if len(top_lanthanides)
        else "unavailable"
    )
    top_extractants = (
        extractants.nlargest(min(3, len(extractants)), "delta_MAE")
        if len(extractants)
        else extractants
    )
    extractant_answer = (
        "; ".join(
            f"{row.extractant_id} ({float(row.delta_MAE):+.6f})"
            for row in top_extractants.itertuples(index=False)
        )
        if len(top_extractants)
        else "unavailable"
    )

    metric_deltas = {
        "MAE": float(a2["mae"]) - float(a5["mae"]),
        "RMSE": float(a2["rmse"]) - float(a5["rmse"]),
        "R2": float(a5["r2"]) - float(a2["r2"]),
        "Spearman": float(a5["spearman"]) - float(a2["spearman"]),
    }
    consistency = ", ".join(
        f"{name}={value:+.6f}" for name, value in metric_deltas.items()
    )
    secondary = (
        f"A2-vs-A6 delta_MAE={float(a2['mae']) - float(a6['mae']):+.6f}"
        if a6 is not None
        else "A6 unavailable"
    )

    ci_text = (
        f" (equal-extractant macro delta_MAE={macro_extractant_delta:+.6f}; "
        f"95% CI [{float(ci_low):+.6f}, {float(ci_high):+.6f}])"
        if ci_low is not None and ci_high is not None
        else ""
    )
    return [
        f"1. **Does 3D improve unseen-extractant prediction?** {conclusion}{ci_text}.",
        f"2. **Absolute MAE improvement:** {absolute:+.6f} log units (A2 minus A5).",
        f"3. **Relative MAE improvement:** {_display_float(relative)}% versus A2.",
        f"4. **Held-out extractant breadth:** {extractant_wins}/{len(extractants)} "
        f"({_display_float(extractant_fraction)}%) have positive delta_MAE.",
        f"5. **Descriptor blocks:** {block_answer}. Secondary complete-3D result: {secondary}.",
        f"6. **Training-only 3D shuffle:** {shuffle_answer}.",
        f"7. **Geometry-quality dependence:** {qc_answer}.",
        f"8. **Largest lanthanide gains:** {lanthanide_answer}. Largest exact-extractant "
        f"gains: {extractant_answer}. No separate ligand-family label was inferred.",
        "9. **Across seeds:** this directory is one fixed model/split seed; consistency "
        "across seeds must be answered by the aggregate run, not inferred here.",
        f"10. **Metric consistency (positive favors A5):** {consistency}.",
    ]


def build_report(
    *,
    args: argparse.Namespace,
    pair_data: Any,
    registry: Any,
    tables: dict[str, pd.DataFrame],
    result: Any | None = None,
) -> str:
    per_ablation = tables["per_ablation_metrics"]
    paired = tables["paired_deltas"]
    per_extractant = tables["per_extractant_comparison"]
    per_lanthanide = tables["per_lanthanide_comparison"]
    blocks = _block_availability(registry)
    skipped = blocks.loc[blocks["status"].eq("unavailable_skipped"), "block"].tolist()
    skipped_text = ", ".join(skipped) if skipped else "none"
    shuffle_text = (
        ", ".join(str(seed) for seed in args.shuffle_seed_values)
        if args.shuffle_seed_values
        else "disabled"
    )
    excluded_non_geometry = tuple(
        getattr(registry, "excluded_non_geometric_columns", ())
    )

    lines = [
        "# Leakage-safe lanthanide feature-ablation report",
        "",
        "## Scope and protocol",
        "",
        f"- Pair scope: `{args.pair_scope}` ({len(pair_data.frame):,} condition-matched pairs).",
        f"- Held-out group: `{args.group_mode}`; split seed `{args.split_seed}`.",
        f"- Model seed: `{args.model_seed}`; outer/inner folds: "
        f"`{args.outer_folds}/{args.inner_folds}`.",
        f"- Main ablations: `{', '.join(ABLATION_FAMILIES)}` on exactly the same outer folds.",
        f"- Training-only 3D shuffle seeds: `{shuffle_text}`.",
        f"- VR descriptor blocks: `{', '.join(args.descriptor_blocks) or 'none'}`.",
        f"- Non-geometric xTB/electronic columns excluded from A0–A6: "
        f"`{len(excluded_non_geometry)}`.",
        "",
        "The primary comparison is A2 (conditions + lanthanide + 2D) versus A5 "
        "(A2 + local 3D). A2 versus A6 is secondary. A3 versus A2 tests whether "
        "local coordination geometry can substitute for ligand 2D information.",
        "The excluded column names and reasons are frozen in `feature_registry.json`; "
        "therefore the primary A2-vs-A5 contrast is not confounded by dipole or "
        "partial-charge features.",
        "",
        "## Direct answers",
        "",
        *_scientific_answer_lines(tables, result=result),
        "",
        "## Aggregate OOF metrics by ablation",
        "",
        _markdown_table(per_ablation),
        "",
        "## Paired fold improvements",
        "",
        "Positive `delta_MAE` and `delta_RMSE` mean the named candidate model wins; "
        "positive `delta_R2` and `delta_Spearman` also favor the candidate. The "
        "reference/candidate columns define direction explicitly.",
        "",
        _markdown_table(paired),
        "",
        "## Geometry-shuffle negative control",
        "",
        "Shuffling is performed only inside each outer-training fold and never "
        "uses outer-test labels or descriptors. Compare A5 to A2 and every "
        "`A5_SHUFFLED` row above. Meaningful structure-specific signal is supported "
        "only when correct A5 improves while shuffled A5 falls back toward A2.",
        "",
        "## Descriptor-block availability",
        "",
        _markdown_table(blocks),
        "",
        f"Empty local blocks are unavailable and were skipped, not fit as duplicate "
        f"A2 models. Unavailable blocks in this run: `{skipped_text}`. The "
        "pre-specified exact coordinate-only blocks are `global_shape` and "
        "`coordination_shape`; `ligand_field` and approximate ray-based `enclosure` "
        "require explicit CLI selection.",
        "",
        "## Held-out extractant comparison",
        "",
        _markdown_table(per_extractant),
        "",
        "## Lanthanide comparison",
        "",
        _markdown_table(per_lanthanide),
        "",
        "## Geometry quality",
        "",
        "The machine-readable `geometry_qc_summary.json` states which quality strata "
        "are actually represented. Failed geometries are excluded by pair construction "
        "and are never silently imputed into a 3D arm. A high-versus-low-confidence "
        "claim is unavailable when either stratum has no eligible pairs.",
        "",
        "## Audit interpretation",
        "",
        "This run does not declare a positive 3D result merely from one metric. Review "
        "the paired A2-vs-A5 MAE first, then RMSE, R2, Spearman, extractant win fraction, "
        "shuffle controls, geometry quality, and seed-to-seed aggregation. `validation.json` "
        "and `leakage_audit.json` must both pass before scientific interpretation.",
        "",
    ]
    return "\n".join(lines)


def _audit_passed(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, dict):
        for key in ("passed", "all_checks_passed", "valid"):
            if key in value:
                return bool(value[key])
    return default


def validate_result_contract(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    registry: Any,
    result: Any,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    required_files = {
        "run_config.json",
        "summary.json",
        "metrics.json",
        "oof_predictions.csv",
        "fold_metrics.csv",
        "fold_assignments.csv",
        "fold_memberships.csv",
        "inner_fold_assignments.csv",
        "inner_cv_tuning.csv",
        "preprocessing_audit.json",
        "preprocessing_parameters.csv",
        "selected_features.csv",
        "shuffle_audit.csv",
        "feature_registry.json",
        "pair_build_audit.json",
        "geometry_descriptor_audit.json",
        "geometry_qc_summary.json",
        "leakage_audit.json",
        "feature_audit.json",
        "per_ablation_metrics.csv",
        "paired_ablation_deltas.csv",
        "per_extractant_comparison.csv",
        "per_lanthanide_comparison.csv",
        "report.md",
    }
    missing = sorted(name for name in required_files if not (output_dir / name).is_file())
    checks["required_artifacts_present"] = not missing
    checks["missing_required_artifacts"] = missing

    checks["leakage_audit_passed"] = _audit_passed(result.leakage_audit)
    checks["feature_audit_passed"] = bool(
        result.feature_audit.get("all_registry_features_assigned_once", False)
        and not result.feature_audit.get("target_or_identifier_features_present", True)
    )
    registry_payload = registry.to_dict()
    checks["registry_features_uniquely_assigned"] = bool(
        registry_payload["audit"]["all_model_features_assigned_exactly_once"]
    )
    checks["registry_source_columns_accounted_for"] = bool(
        registry_payload["audit"]["all_source_columns_accounted_for"]
    )
    checks["registry_has_no_target_or_identifier_features"] = not bool(
        registry_payload["audit"]["target_or_identifier_features_present"]
    )

    predictions = tables["predictions"]
    required_prediction_columns = {
        "pair_id",
        "extractant",
        "pair_label",
        "metal_A",
        "metal_B",
        "outer_fold",
        "ablation",
        "y_true",
        "y_pred",
    }
    missing_prediction_columns = sorted(
        required_prediction_columns - set(predictions.columns)
    )
    checks["oof_identity_contract_complete"] = not missing_prediction_columns
    checks["missing_oof_identity_columns"] = missing_prediction_columns
    ablation_column = next(
        (name for name in ("ablation", "feature_set", "model") if name in predictions),
        None,
    )
    observed: set[str] = set()
    if ablation_column is not None:
        observed = set(predictions[ablation_column].dropna().astype(str))
    # A run may deliberately recompute only a subset of the frozen ladder; what
    # must always hold is that every arm the run declared was actually produced.
    expected_main = set(args.prespecified_arm_values or ABLATION_FAMILIES)
    missing_main = sorted(expected_main - observed)
    checks["all_main_ablations_present"] = not missing_main
    checks["missing_main_ablations"] = missing_main
    checks["main_ablations_requested"] = sorted(expected_main)
    checks["full_prespecified_ladder_evaluated"] = not sorted(
        set(ABLATION_FAMILIES) - observed
    )

    expected_shuffle_count = len(args.shuffle_seed_values)
    observed_shuffle = sorted(name for name in observed if name.startswith("A5_SHUFFLED"))
    checks["shuffle_controls_present"] = len(observed_shuffle) == expected_shuffle_count
    checks["expected_shuffle_control_count"] = expected_shuffle_count
    checks["observed_shuffle_controls"] = observed_shuffle

    expected_extension_arms = (
        set(EXTENSION_ABLATION_FAMILIES) | {"G4"} if args.extension_arms else set()
    )
    missing_extension = sorted(
        arm
        for arm in expected_extension_arms
        if arm not in observed and _extension_arm_is_buildable(arm, args)
    )
    checks["extension_arms_present"] = not missing_extension
    checks["missing_extension_arms"] = missing_extension
    observed_extension_controls = sorted(
        name
        for name in observed
        if name.endswith(tuple(f"_SHUFFLED_s{s}" for s in args.extension_shuffle_seed_values))
        and not name.startswith("A5_SHUFFLED")
    )
    expected_sensitivity_arms = (
        set(SENSITIVITY_2D_ARMS) if args.two_d_sensitivity else set()
    )
    missing_sensitivity = sorted(
        arm
        for arm in expected_sensitivity_arms
        if arm not in observed and _sensitivity_arm_is_buildable(arm, args)
    )
    checks["sensitivity_2d_arms_present"] = not missing_sensitivity
    checks["missing_2d_sensitivity_arms"] = missing_sensitivity

    # G4 selects its block by namespace rather than by feature family, so it is
    # not in EXTENSION_SHUFFLE_BLOCKS but does get a control when it is present.
    # The S-ladder blocks make the same claim as their A2-referenced twins and
    # are counted the same way.
    controllable_arms = (
        set(EXTENSION_SHUFFLE_BLOCKS) | {"G4"} | set(SENSITIVITY_2D_SHUFFLE_BLOCKS)
    ) & observed
    checks["expected_extension_control_count"] = len(
        args.extension_shuffle_seed_values
    ) * len(controllable_arms)
    checks["observed_extension_controls"] = observed_extension_controls
    checks["extension_controls_present"] = bool(
        len(observed_extension_controls)
        == checks["expected_extension_control_count"]
    )

    assignments = tables["fold_assignments"]
    pair_key = next(
        (name for name in ("pair_id", "row_id") if name in assignments.columns), None
    )
    fold_key = next(
        (name for name in ("outer_fold", "fold") if name in assignments.columns), None
    )
    checks["fold_assignments_have_row_and_fold"] = pair_key is not None and fold_key is not None
    checks["one_test_fold_per_row"] = bool(
        pair_key is not None
        and fold_key is not None
        and not assignments[pair_key].duplicated().any()
        and assignments[fold_key].notna().all()
    )

    memberships = tables["fold_memberships"]
    membership_column = next(
        (name for name in ("assignment", "membership", "split") if name in memberships),
        None,
    )
    checks["fold_memberships_include_train_and_test"] = bool(
        membership_column is not None
        and {"train", "test"}.issubset(
            set(memberships[membership_column].dropna().astype(str).str.lower())
        )
    )

    checks["passed"] = all(
        bool(value)
        for key, value in checks.items()
        if key
        in {
            "required_artifacts_present",
            "leakage_audit_passed",
            "feature_audit_passed",
            "registry_features_uniquely_assigned",
            "registry_source_columns_accounted_for",
            "registry_has_no_target_or_identifier_features",
            "all_main_ablations_present",
            "shuffle_controls_present",
            # A declared extension arm without its permutation control cannot be
            # interpreted, so both gate the run.
            "extension_arms_present",
            "extension_controls_present",
            "sensitivity_2d_arms_present",
            "oof_identity_contract_complete",
            "fold_assignments_have_row_and_fold",
            "one_test_fold_per_row",
            "fold_memberships_include_train_and_test",
        }
    )
    return checks


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "scripts/run_ablation_benchmark.py": Path(__file__).resolve(),
        "src/lanthanide_separation/ablation.py": (
            SRC_ROOT / "lanthanide_separation" / "ablation.py"
        ),
        "src/lanthanide_separation/feature_registry.py": (
            SRC_ROOT / "lanthanide_separation" / "feature_registry.py"
        ),
        "src/lanthanide_separation/pairs.py": (
            SRC_ROOT / "lanthanide_separation" / "pairs.py"
        ),
        "src/lanthanide_separation/geometry_descriptors.py": (
            SRC_ROOT / "lanthanide_separation" / "geometry_descriptors.py"
        ),
        "src/lanthanide_separation/evaluation.py": (
            SRC_ROOT / "lanthanide_separation" / "evaluation.py"
        ),
    }
    return {
        name: sha256_file(path)
        for name, path in paths.items()
        if path.is_file()
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = validate_args(parse_args(argv))
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    dataset_sha256 = sha256_file(dataset_path)
    if (
        args.expected_dataset_sha256 is not None
        and dataset_sha256 != args.expected_dataset_sha256
    ):
        raise SystemExit(
            "Dataset SHA-256 mismatch: expected "
            f"{args.expected_dataset_sha256}, observed {dataset_sha256}."
        )

    vr_path: Path | None = None
    vr_sha256: str | None = None
    if args.descriptor_blocks:
        vr_path = args.vr_assets.expanduser().resolve()
        if not vr_path.is_file():
            raise SystemExit(f"VR coordinate asset not found: {vr_path}")
        vr_sha256 = sha256_file(vr_path)
        if args.expected_vr_sha256 is not None and vr_sha256 != args.expected_vr_sha256:
            raise SystemExit(
                "VR asset SHA-256 mismatch: expected "
                f"{args.expected_vr_sha256}, observed {vr_sha256}."
            )

    # Import only after argument and immutable input validation.  This keeps
    # ``--help`` and runner-unit tests usable while the evaluator is optional.
    from lanthanide_separation.ablation import (  # noqa: PLC0415
        DEFAULT_PARAMETER_GRID,
        run_ablation_benchmark,
    )

    output_dir = make_output_dir(args.output_dir, args)
    started_at = datetime.now(timezone.utc).isoformat()
    write_text_atomic("Run has not completed.\n", output_dir / "_INCOMPLETE")

    parameter_grid: Iterable[dict[str, Any]] = DEFAULT_PARAMETER_GRID
    if args.quick:
        parameter_grid = ({"max_features": 0.70, "min_samples_leaf": 2},)
    scientific_protocol = scientific_protocol_payload(args, parameter_grid)
    scientific_protocol_sha256 = sha256_object(scientific_protocol)

    run_config = {
        "schema_version": "1.0",
        "status": "configured",
        "started_at_utc": started_at,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "vr_asset_path": str(vr_path) if vr_path is not None else None,
        "vr_asset_sha256": vr_sha256,
        "arguments": {
            **vars(args),
            "shuffle_seed_values": list(args.shuffle_seed_values),
            "descriptor_blocks": list(args.descriptor_blocks),
        },
        "parameter_grid": list(parameter_grid),
        "scientific_protocol": scientific_protocol,
        "scientific_protocol_sha256": scientific_protocol_sha256,
        "software": _software_metadata(),
        "slurm": _slurm_metadata(),
    }
    write_json_atomic(run_config, output_dir / "run_config.json")

    print(f"Dataset: {dataset_path}", flush=True)
    print(f"Dataset SHA-256: {dataset_sha256}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print("Reading dataset...", flush=True)
    source = pd.read_parquet(dataset_path)

    descriptor_audit: dict[str, Any] = {
        "requested_blocks": list(args.descriptor_blocks),
        "status": "not_requested",
        "descriptor_column_count": 0,
    }
    if args.descriptor_blocks:
        print(
            "Computing VR-derived descriptors: "
            + ", ".join(args.descriptor_blocks),
            flush=True,
        )
        descriptors = MetalSiteDescriptorBuilder(
            vr_path,
            blocks=args.descriptor_blocks,
            profile=args.descriptor_profile,
        ).build()
        source, descriptor_audit = attach_geometry_descriptors(source, descriptors)
        descriptor_audit = {"status": "computed", **descriptor_audit}
    write_json_atomic(descriptor_audit, output_dir / "geometry_descriptor_audit.json")

    pair_data = build_lanthanide_pair_dataset(
        source,
        pair_scope=args.pair_scope,
        require_geometry=True,
        replicate_policy=args.replicate_policy,
        quarantine_known_bad=not args.no_default_quarantine,
        quarantine_known_censored_targets=args.quarantine_known_censored_targets,
        require_complete_conditions=not args.allow_incomplete_conditions,
        delta3d_feature_set=args.delta3d_feature_set,
        geometry_descriptor_blocks=args.descriptor_blocks,
        include_donor_composition_counts=True,
        include_symmetric_3d=args.symmetric_3d,
        include_pair_response_3d=args.pair_response_3d,
        include_electronic=args.electronic,
    )
    registry = build_feature_registry(pair_data)
    write_json_atomic(pair_data.audit, output_dir / "pair_build_audit.json")
    # Provenance and cohort accounting are written before any model is fitted so
    # they cannot be shaped by a result.
    write_json_atomic(
        audit_electronic_sources(source), output_dir / "electronic_provenance.json"
    )
    cohort = build_cohort_report(source, pair_data, registry)
    write_json_atomic(cohort, output_dir / "cohort_report.json")
    write_csv_atomic(
        pd.DataFrame(cohort["feature_missingness"]),
        output_dir / "feature_missingness.csv",
    )
    write_json_atomic(registry.to_dict(), output_dir / "feature_registry.json")
    if not pair_data.quarantine.empty:
        write_csv_atomic(pair_data.quarantine, output_dir / "quarantined_source_rows.csv")

    print(
        f"Pairs: {len(pair_data.frame):,}; scope={args.pair_scope}; "
        f"extractants={pair_data.frame['extractant'].nunique()}; "
        f"features={len(registry.columns)}",
        flush=True,
    )
    group_column = "extractant" if args.group_mode == "extractant" else "ecfp_exact_cluster"
    print("Freezing grouped outer folds and running A0--A6...", flush=True)
    result = run_ablation_benchmark(
        pair_data,
        registry=registry,
        group_column=group_column,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        n_estimators=args.trees,
        n_jobs=args.n_jobs,
        n_bootstrap=args.bootstrap,
        seed=args.model_seed,
        split_seed=args.split_seed,
        shuffle_seeds=args.shuffle_seed_values,
        include_block_ablations=not args.no_block_ablations,
        include_reference_baselines=args.reference_baselines,
        include_symmetric_arms=args.symmetric_3d,
        include_extension_arms=args.extension_arms,
        include_2d_sensitivity_arms=args.two_d_sensitivity,
        extension_shuffle_seeds=args.extension_shuffle_seed_values,
        prespecified_arms=args.prespecified_arm_values or None,
        parameter_grid=tuple(parameter_grid),
    )
    for fold in result.leakage_audit.get("outer_folds", []):
        held_out = fold.get("held_out_extractants", [])
        print(
            f"Outer fold {fold.get('outer_fold')}: held-out extractants "
            + ", ".join(str(value) for value in held_out),
            flush=True,
        )

    # Declared secondary analysis: does an arm's advantage survive when the
    # held-out ligand is genuinely structurally new?  It re-reads the frozen OOF
    # predictions and never touches the primary split.
    print("Running the structural-novelty sensitivity analysis...", flush=True)
    novelty_comparisons = [
        ("A2", arm)
        for arm in result.summary["feature_counts"]
        if arm != "A2" and f"prediction_{arm}" in result.predictions.columns
    ]
    novelty = novelty_report(
        pair_data.frame.reset_index(drop=True),
        result.predictions,
        result.fold_assignments,
        novelty_comparisons,
        target_column=PAIR_TARGET_COLUMN,
        group_column="extractant",
        n_bootstrap=args.bootstrap,
        seed=args.model_seed + 8_675_309,
    )
    write_json_atomic(novelty, output_dir / "structural_novelty.json")
    write_csv_atomic(
        pd.DataFrame(novelty["stratified_deltas"]),
        output_dir / "structural_novelty_deltas.csv",
    )
    write_csv_atomic(
        pd.DataFrame(novelty["per_extractant_novelty"]),
        output_dir / "structural_novelty_per_extractant.csv",
    )

    tables = prepare_artifact_tables(result, registry)
    write_json_atomic(metrics_payload(result, tables), output_dir / "metrics.json")
    for field_name, filename in TABLE_ARTIFACTS:
        write_csv_atomic(tables[field_name], output_dir / filename)
    write_json_atomic(
        {
            "fit_scope": "one outer-training subset per ablation/fold",
            "learned_transform": "fold-local median imputation",
            "outer_test_rows_seen_during_fit": 0,
            "records": tables["preprocessing_parameters"].to_dict(orient="records"),
        },
        output_dir / "preprocessing_audit.json",
    )
    for field_name, filename in JSON_ARTIFACTS:
        if field_name == "preprocessing_audit":
            continue
        write_json_atomic(getattr(result, field_name), output_dir / filename)

    report = build_report(
        args=args,
        pair_data=pair_data,
        registry=registry,
        tables=tables,
        result=result,
    )
    write_text_atomic(report, output_dir / "report.md")

    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "vr_asset_path": str(vr_path) if vr_path is not None else None,
        "vr_asset_sha256": vr_sha256,
        "cohort_sha256": pair_data.audit["cohort_sha256"],
        "pair_scope": args.pair_scope,
        "group_mode": args.group_mode,
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "shuffle_seeds": list(args.shuffle_seed_values),
        "feature_registry_sha256": sha256_file(output_dir / "feature_registry.json"),
        "fold_assignments_sha256": sha256_file(output_dir / "fold_assignments.csv"),
        "fold_memberships_sha256": sha256_file(output_dir / "fold_memberships.csv"),
        "inner_fold_assignments_sha256": sha256_file(
            output_dir / "inner_fold_assignments.csv"
        ),
        "geometry_descriptor_audit_sha256": sha256_file(
            output_dir / "geometry_descriptor_audit.json"
        ),
        "scientific_protocol": scientific_protocol,
        "scientific_protocol_sha256": scientific_protocol_sha256,
        "implementation_sha256": _implementation_hashes(),
        "pair_build_audit": pair_data.audit,
        "benchmark": result.summary,
        "software": _software_metadata(),
        "slurm": _slurm_metadata(),
    }
    # summary is written before validation because it is part of the required
    # contract that validation independently checks.
    write_json_atomic(summary, output_dir / "summary.json")

    validation = validate_result_contract(
        output_dir=output_dir,
        args=args,
        registry=registry,
        result=result,
        tables=tables,
    )
    write_json_atomic(validation, output_dir / "validation.json")
    if not validation["passed"]:
        raise RuntimeError(
            "Ablation result contract failed validation; inspect "
            f"{output_dir / 'validation.json'}."
        )

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
        and path.name not in {"_INCOMPLETE", "_SUCCESS.json", "artifact_hashes.json"}
    }
    write_json_atomic(
        {
            "schema_version": "1.0",
            "algorithm": "sha256",
            "sha256": artifact_hashes,
        },
        output_dir / "artifact_hashes.json",
    )

    # Success is committed last.  A preempted or invalid run always retains
    # _INCOMPLETE and can never be mistaken for an aggregation-ready bundle.
    (output_dir / "_INCOMPLETE").unlink()
    write_json_atomic(
        {
            "status": "complete",
            "summary_sha256": sha256_file(output_dir / "summary.json"),
            "validation_sha256": sha256_file(output_dir / "validation.json"),
            "artifact_hashes_sha256": sha256_file(output_dir / "artifact_hashes.json"),
        },
        output_dir / "_SUCCESS.json",
    )

    print("Validation: PASS", flush=True)
    print(f"Saved complete run bundle: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
