#!/usr/bin/env python3
"""Fail-closed multi-seed aggregation and OOF ensembling for simplicial runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lanthanide_separation.evaluation import (  # noqa: E402
    regression_metrics,
)
from lanthanide_separation.pairs import PAIR_TARGET_COLUMN  # noqa: E402


METRICS = (
    "r2_gain",
    "group_balanced_r2_gain",
    "mae_reduction",
    "macro_group_mae_reduction",
    "sign_accuracy_gain",
)
REQUIRED_ARTIFACTS = (
    "summary.json",
    "_SUCCESS.json",
    "run_config.json",
    "pair_build_audit.json",
    "feature_contract.json",
    "asset_contract.json",
    "oof_predictions.csv",
    "fold_assignments.csv",
    "inner_cv_tuning.csv",
    "simplicial_training_history.csv",
    "per_extractant_metrics.csv",
    "per_pair_type_metrics.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed-plan", type=Path, default=None)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument(
        "--expected-pair-scope",
        choices=("all", "adjacent"),
        required=True,
        help="Fail unless every run uses this explicitly requested pair cohort scope.",
    )
    parser.add_argument("--ensemble-bootstrap", type=int, default=5000)
    parser.add_argument("--ensemble-bootstrap-seed", type=int, default=8675309)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text_atomic(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(payload: Any, path: Path) -> None:
    write_text_atomic(canonical_json(payload) + "\n", path)


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def read_seed_plan(path: Path) -> dict[int, tuple[int, int]]:
    plan: dict[int, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = ["task_index", "model_seed", "split_seed"]
        if reader.fieldnames != expected_fields:
            raise ValueError(f"Seed plan header must be {expected_fields}.")
        for row in reader:
            task_index = int(row["task_index"])
            model_seed = int(row["model_seed"])
            split_seed = int(row["split_seed"])
            if task_index in plan:
                raise ValueError(f"Duplicate task index in seed plan: {task_index}")
            if min(task_index, model_seed, split_seed) < 0:
                raise ValueError("Seed plan values must be nonnegative integers.")
            if model_seed > 4_000_000_000 or split_seed > 4_000_000_000:
                raise ValueError("Seed plan seeds must not exceed 4000000000.")
            plan[task_index] = (model_seed, split_seed)
    if not plan or set(plan) != set(range(len(plan))):
        raise ValueError("Seed plan task indices must be contiguous from zero.")
    if len({pair[0] for pair in plan.values()}) != len(plan):
        raise ValueError("Seed plan model seeds must be unique.")
    if len({pair[1] for pair in plan.values()}) != len(plan):
        raise ValueError("Seed plan split seeds must be unique.")
    return plan


def normalized_protocol(summary: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(nested(summary, "arguments"))
    for key in (
        "dataset",
        "vr_assets",
        "row_geometry_map",
        "output_dir",
        "logical_output_dir",
        "seed",
        "split_seed",
        "n_jobs",
        "torch_threads",
    ):
        arguments.pop(key, None)
    protocol = dict(nested(summary, "benchmark", "protocol"))
    protocol.pop("model_seed", None)
    protocol.pop("split_seed", None)
    return {"arguments": arguments, "benchmark_protocol": protocol}


def normalized_asset_contract(contract: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(contract)
    normalized.pop("vr_asset_path", None)
    normalized.pop("row_geometry_map_path", None)
    return normalized


def normalized_software(summary: dict[str, Any]) -> dict[str, Any]:
    software = nested(summary, "software")
    if not isinstance(software, dict):
        raise TypeError("software must be a JSON object")
    keys = (
        "python",
        "numpy",
        "pandas",
        "scikit_learn",
        "torch",
        "device_type",
        "cuda_version",
    )
    missing = [key for key in keys if key not in software]
    if missing:
        raise KeyError(f"software.{','.join(missing)}")
    return {key: software[key] for key in keys}


def metric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
        "min": min(values),
        "max": max(values),
        "positive_count": sum(value > 0.0 for value in values),
    }


def improvement_metrics(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    reference_metrics = regression_metrics(truth, reference, groups)
    candidate_metrics = regression_metrics(truth, candidate, groups)
    return {
        "r2_gain": float(candidate_metrics["r2"] - reference_metrics["r2"]),
        "group_balanced_r2_gain": float(
            candidate_metrics["group_balanced_r2"]
            - reference_metrics["group_balanced_r2"]
        ),
        "mae_reduction": float(reference_metrics["mae"] - candidate_metrics["mae"]),
        "macro_group_mae_reduction": float(
            reference_metrics["macro_group_mae"]
            - candidate_metrics["macro_group_mae"]
        ),
        "sign_accuracy_gain": float(
            candidate_metrics["sign_accuracy"] - reference_metrics["sign_accuracy"]
        ),
    }


def paired_group_bootstrap(
    frame: pd.DataFrame,
    *,
    group_column: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_groups = frame[group_column].drop_duplicates().to_numpy()
    group_indices = {
        group: frame.index[frame[group_column].eq(group)].to_numpy()
        for group in unique_groups
    }
    draws = {metric: [] for metric in METRICS}
    for _ in range(int(replicates)):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        sampled = frame.loc[sampled_indices]
        truth = sampled[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
        reference = sampled["prediction_delta3d"].to_numpy(dtype=float)
        candidate = sampled["prediction_simplicial"].to_numpy(dtype=float)
        if np.var(truth) <= 0.0:
            continue
        weights = np.concatenate(
            [
                np.full(len(group_indices[group]), 1.0 / len(group_indices[group]))
                for group in sampled_groups
            ]
        )
        draws["r2_gain"].append(
            float(r2_score(truth, candidate) - r2_score(truth, reference))
        )
        draws["group_balanced_r2_gain"].append(
            float(
                r2_score(truth, candidate, sample_weight=weights)
                - r2_score(truth, reference, sample_weight=weights)
            )
        )
        draws["mae_reduction"].append(
            float(
                mean_absolute_error(truth, reference)
                - mean_absolute_error(truth, candidate)
            )
        )
        draws["sign_accuracy_gain"].append(
            float(
                np.mean(np.sign(truth) == np.sign(candidate))
                - np.mean(np.sign(truth) == np.sign(reference))
            )
        )
        group_differences = []
        for group in sampled_groups:
            group_frame = frame.loc[group_indices[group]]
            group_truth = group_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
            group_differences.append(
                mean_absolute_error(
                    group_truth,
                    group_frame["prediction_delta3d"].to_numpy(dtype=float),
                )
                - mean_absolute_error(
                    group_truth,
                    group_frame["prediction_simplicial"].to_numpy(dtype=float),
                )
            )
        draws["macro_group_mae_reduction"].append(float(np.mean(group_differences)))

    intervals: dict[str, Any] = {}
    for metric, values in draws.items():
        array = np.asarray(values, dtype=float)
        if not len(array):
            raise ValueError("Cross-seed ensemble bootstrap produced zero valid draws.")
        intervals[metric] = {
            "mean": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }
    return {
        "method": "paired cluster bootstrap of cross-seed averaged OOF predictions",
        "reference_model": "cross-seed adaptive tabular Delta3D OOF ensemble",
        "candidate_model": "cross-seed guarded simplicial OOF ensemble",
        "replicates": len(draws["r2_gain"]),
        "metrics": intervals,
    }


def report_markdown(payload: dict[str, Any]) -> str:
    rows = payload["runs"]
    aggregate = payload["across_seed_improvements"]
    ensemble = payload["cross_seed_oof_ensemble"]
    lines = [
        "# Simplicial multi-seed result",
        "",
        "Validation: PASSED",
        "",
        f"Pair scope: `{payload['validation']['expected_pair_scope']}`.",
        "",
        "Primary comparison: guarded SNN hybrid minus adaptive tabular Delta3D.",
        "",
        "| model seed | split seed | Delta R2 | Delta balanced R2 | macro-MAE reduction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_seed']} | {row['split_seed']} | "
            f"{row['r2_gain']:+.6f} | {row['group_balanced_r2_gain']:+.6f} | "
            f"{row['macro_group_mae_reduction']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Across-seed distribution",
            "",
            "| metric | mean | std | min | max | positive runs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in METRICS:
        item = aggregate[metric]
        std = "NA" if item["sample_std"] is None else f"{item['sample_std']:.6f}"
        lines.append(
            f"| {metric} | {item['mean']:+.6f} | {std} | {item['min']:+.6f} | "
            f"{item['max']:+.6f} | {item['positive_count']}/{item['n']} |"
        )
    lines.extend(
        [
            "",
            "## Cross-seed OOF ensemble",
            "",
            "Averaging is performed by pair_id only after every constituent prediction was "
            "generated with that pair's entire held-out group excluded from training.",
            "",
        ]
    )
    for metric in METRICS:
        point = ensemble["improvements"][metric]
        interval = ensemble["paired_group_bootstrap"]["metrics"][metric]
        lines.append(
            f"- {metric}: {point:+.6f}; 95% CI "
            f"[{interval['ci95_low']:+.6f}, {interval['ci95_high']:+.6f}]"
        )
    lines.extend(
        [
            "",
            "Per-run fixed-OOF confidence bounds were not averaged. The ensemble interval was "
            "recomputed from the aligned cross-seed OOF predictions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.expected_runs is not None and args.expected_runs < 2:
        raise SystemExit("--expected-runs must be at least 2.")
    if args.ensemble_bootstrap < 1:
        raise SystemExit("--ensemble-bootstrap must be positive.")
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Run root not found: {run_root}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "aggregate"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_text_atomic("Aggregation has not completed.\n", output_dir / "_INCOMPLETE")

    errors: list[str] = []
    seed_plan_path = (
        args.seed_plan.expanduser().resolve()
        if args.seed_plan is not None
        else run_root / "seed_plan.tsv"
    )
    seed_plan: dict[int, tuple[int, int]] = {}
    seed_plan_sha256: str | None = None
    try:
        if not seed_plan_path.is_file():
            raise ValueError(f"Seed plan not found: {seed_plan_path}")
        seed_plan = read_seed_plan(seed_plan_path)
        seed_plan_sha256 = file_sha256(seed_plan_path)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"Invalid immutable seed plan: {exc}")
    if args.expected_runs is not None and seed_plan and len(seed_plan) != args.expected_runs:
        errors.append(
            f"Seed plan contains {len(seed_plan)} tasks, expected {args.expected_runs}."
        )
    # Only atomically promoted final seed directories are scientific inputs.
    # Interrupted attempt directories under attempts/ are deliberately ignored.
    summary_paths = sorted(run_root.glob("run_*/summary.json"))
    if args.expected_runs is not None and len(summary_paths) != args.expected_runs:
        errors.append(f"Expected {args.expected_runs} summaries, found {len(summary_paths)}.")
    if len(summary_paths) < 2:
        errors.append("At least two summaries are required.")

    summaries: list[tuple[Path, dict[str, Any]]] = []
    for path in summary_paths:
        run_dir = path.parent
        missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            errors.append(f"{run_dir.name}: missing {missing}")
            continue
        try:
            summary = read_json(path)
            success = read_json(run_dir / "_SUCCESS.json")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{run_dir.name}: invalid JSON: {exc}")
            continue
        if success.get("summary_sha256") != file_sha256(path):
            errors.append(f"{run_dir.name}: invalid _SUCCESS summary hash")
            continue
        if success.get("status") != "complete":
            errors.append(f"{run_dir.name}: _SUCCESS status is not complete")
            continue
        if (run_dir / "_INCOMPLETE").exists():
            errors.append(f"{run_dir.name}: stale _INCOMPLETE marker")
            continue
        try:
            declared_hashes = nested(summary, "artifacts", "sha256")
            if not isinstance(declared_hashes, dict):
                raise TypeError("artifacts.sha256 must be a JSON object")
            required_hashed = set(REQUIRED_ARTIFACTS) - {"summary.json", "_SUCCESS.json"}
            missing_hashes = sorted(required_hashed - set(declared_hashes))
            if missing_hashes:
                raise ValueError(f"artifact hashes missing {missing_hashes}")
            for name in sorted(required_hashed):
                if str(declared_hashes[name]) != file_sha256(run_dir / name):
                    raise ValueError(f"artifact hash mismatch: {name}")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            errors.append(f"{run_dir.name}: {exc}")
            continue
        summaries.append((path, summary))
    if args.expected_runs is not None and len(summaries) != args.expected_runs:
        errors.append(f"Only {len(summaries)} completed runs validated structurally.")

    fingerprints: dict[str, set[str]] = {
        "dataset": set(),
        "cohort": set(),
        "assets": set(),
        "implementation": set(),
        "protocol": set(),
        "pair_audit": set(),
        "feature_contract": set(),
        "asset_contract": set(),
        "software": set(),
    }
    seed_pairs: set[tuple[int, int]] = set()
    model_seeds: set[int] = set()
    split_seeds: set[int] = set()
    task_indices: set[int] = set()
    run_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    group_column: str | None = None
    observed_pair_scopes: set[str] = set()

    for path, summary in summaries:
        run_name = path.parent.name
        try:
            pair_scope = str(nested(summary, "arguments", "pair_scope"))
            observed_pair_scopes.add(pair_scope)
            if pair_scope != args.expected_pair_scope:
                raise ValueError(
                    "pair scope mismatch: "
                    f"expected {args.expected_pair_scope!r}, observed {pair_scope!r}"
                )
            if not bool(nested(summary, "arguments", "evaluation_only")):
                raise ValueError("evaluation_only must be true")
            if not bool(nested(summary, "benchmark", "leakage_audit", "passed")):
                raise ValueError("leakage audit failed")
            model_seed = int(nested(summary, "arguments", "seed"))
            split_seed = int(nested(summary, "arguments", "split_seed"))
            name_match = re.fullmatch(
                r"run_(\d+)_model_(-?\d+)_split_(-?\d+)", path.parent.name
            )
            if name_match is None:
                raise ValueError("final run directory name does not match the seed contract")
            if int(name_match.group(2)) != model_seed or int(name_match.group(3)) != split_seed:
                raise ValueError("directory seeds disagree with summary arguments")
            task_index = int(name_match.group(1))
            if task_index in task_indices:
                raise ValueError(f"duplicate task index {task_index}")
            task_indices.add(task_index)
            if seed_plan and seed_plan.get(task_index) != (model_seed, split_seed):
                raise ValueError(
                    f"task {task_index} seeds disagree with immutable seed plan"
                )
            pair = (model_seed, split_seed)
            if pair in seed_pairs:
                raise ValueError(f"duplicate seed pair {pair}")
            seed_pairs.add(pair)
            model_seeds.add(model_seed)
            split_seeds.add(split_seed)
            current_group = str(nested(summary, "benchmark", "protocol", "group_column"))
            if group_column is None:
                group_column = current_group
            elif group_column != current_group:
                raise ValueError("group column differs across runs")

            fingerprints["dataset"].add(str(nested(summary, "dataset_sha256")))
            fingerprints["cohort"].add(str(nested(summary, "cohort_sha256")))
            fingerprints["assets"].add(object_sha256(nested(summary, "asset_sha256")))
            fingerprints["implementation"].add(
                object_sha256(nested(summary, "implementation_sha256"))
            )
            fingerprints["protocol"].add(object_sha256(normalized_protocol(summary)))
            fingerprints["pair_audit"].add(
                object_sha256(nested(summary, "pair_build_audit"))
            )
            fingerprints["feature_contract"].add(
                object_sha256(read_json(path.parent / "feature_contract.json"))
            )
            fingerprints["asset_contract"].add(
                object_sha256(
                    normalized_asset_contract(
                        read_json(path.parent / "asset_contract.json")
                    )
                )
            )
            fingerprints["software"].add(object_sha256(normalized_software(summary)))

            improvement = nested(
                summary,
                "benchmark",
                "improvements",
                "primary_simplicial_vs_delta3d",
            )
            row: dict[str, Any] = {
                "run_dir": str(path.parent),
                "model_seed": model_seed,
                "split_seed": split_seed,
            }
            for metric in METRICS:
                row[metric] = finite(improvement[metric], f"{run_name}.{metric}")
                interval = nested(
                    summary,
                    "benchmark",
                    "paired_group_bootstrap",
                    "comparisons",
                    "simplicial_vs_delta3d",
                    metric,
                )
                low = finite(interval["ci95_low"], f"{run_name}.{metric}.low")
                high = finite(interval["ci95_high"], f"{run_name}.{metric}.high")
                if low > high:
                    raise ValueError(f"{metric} CI has low > high")
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            predictions = pd.read_csv(path.parent / "oof_predictions.csv")
            required_columns = {
                "pair_id",
                PAIR_TARGET_COLUMN,
                current_group,
                "pair_label",
                "prediction_2d_ensemble",
                "prediction_delta3d",
                "prediction_simplicial",
            }
            missing_columns = sorted(required_columns - set(predictions.columns))
            if missing_columns:
                raise ValueError(f"OOF file misses {missing_columns}")
            if predictions["pair_id"].duplicated().any():
                raise ValueError("OOF pair_id is not unique")
            for column in ("pair_id", current_group, "pair_label"):
                if predictions[column].isna().any():
                    raise ValueError(f"OOF {column} contains null values")
                normalized_text = predictions[column].astype(str).str.strip()
                if normalized_text.eq("").any():
                    raise ValueError(f"OOF {column} contains empty values")
                predictions[column] = normalized_text
            for column in (
                PAIR_TARGET_COLUMN,
                "prediction_2d_ensemble",
                "prediction_delta3d",
                "prediction_simplicial",
            ):
                numeric = pd.to_numeric(predictions[column], errors="raise").to_numpy(
                    dtype=float
                )
                if not np.isfinite(numeric).all():
                    raise ValueError(f"OOF {column} contains non-finite values")
                predictions[column] = numeric
            predictions = predictions.sort_values("pair_id", kind="stable").reset_index(drop=True)
            run_rows.append(row)
            prediction_frames.append(predictions)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"{run_name}: {exc}")

    for label, values in fingerprints.items():
        if len(values) > 1:
            errors.append(f"Runs have different {label} fingerprints.")
    if len(run_rows) != len(summaries):
        errors.append("One or more summaries failed semantic validation.")
    if len(model_seeds) != len(run_rows) or len(split_seeds) != len(run_rows):
        errors.append("Model seeds and split seeds must each be unique.")
    if seed_plan and task_indices != set(seed_plan):
        errors.append(
            "Final task indices do not exactly match immutable seed plan: "
            f"found={sorted(task_indices)}, expected={sorted(seed_plan)}."
        )

    if prediction_frames:
        reference = prediction_frames[0]
        for index, frame in enumerate(prediction_frames[1:], start=1):
            if not reference["pair_id"].equals(frame["pair_id"]):
                errors.append(f"OOF pair IDs differ in run index {index}.")
                continue
            if not np.allclose(
                reference[PAIR_TARGET_COLUMN].to_numpy(dtype=float),
                frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            ):
                errors.append(f"OOF targets differ in run index {index}.")
            if group_column and not reference[group_column].astype(str).equals(
                frame[group_column].astype(str)
            ):
                errors.append(f"OOF groups differ in run index {index}.")
            if not reference["pair_label"].equals(frame["pair_label"]):
                errors.append(f"OOF pair labels differ in run index {index}.")

    validation = {
        "passed": not errors,
        "expected_pair_scope": args.expected_pair_scope,
        "observed_pair_scopes": sorted(observed_pair_scopes),
        "expected_runs": args.expected_runs,
        "discovered_summaries": len(summary_paths),
        "validated_runs": len(run_rows),
        "errors": errors,
        "fingerprints": {key: sorted(values) for key, values in fingerprints.items()},
        "seed_plan_path": str(seed_plan_path),
        "seed_plan_sha256": seed_plan_sha256,
    }
    validation["combined_sha256"] = object_sha256(
        {
            "expected_pair_scope": args.expected_pair_scope,
            "observed_pair_scopes": sorted(observed_pair_scopes),
            "fingerprints": validation["fingerprints"],
            "seed_plan_sha256": seed_plan_sha256,
        }
    )
    if errors:
        write_json_atomic(validation, output_dir / "validation.json")
        text = (
            "# Simplicial aggregation failed\n\n"
            f"Expected pair scope: `{args.expected_pair_scope}`.\n\n"
            + "\n".join(f"- {error}" for error in errors)
        )
        write_text_atomic(text + "\n", output_dir / "report.md")
        return 2

    assert group_column is not None
    run_rows.sort(key=lambda row: (row["split_seed"], row["model_seed"]))
    across_seed = {
        metric: metric_summary([float(row[metric]) for row in run_rows])
        for metric in METRICS
    }
    ensemble_frame = prediction_frames[0][
        ["pair_id", group_column, "pair_label", PAIR_TARGET_COLUMN]
    ].copy()
    ensemble_frame["prediction_2d_ensemble"] = np.mean(
        [frame["prediction_2d_ensemble"].to_numpy(dtype=float) for frame in prediction_frames],
        axis=0,
    )
    ensemble_frame["prediction_delta3d"] = np.mean(
        [frame["prediction_delta3d"].to_numpy(dtype=float) for frame in prediction_frames],
        axis=0,
    )
    ensemble_frame["prediction_simplicial"] = np.mean(
        [frame["prediction_simplicial"].to_numpy(dtype=float) for frame in prediction_frames],
        axis=0,
    )
    truth = ensemble_frame[PAIR_TARGET_COLUMN].to_numpy(dtype=float)
    groups = ensemble_frame[group_column].astype(str).to_numpy()
    ensemble_improvements = improvement_metrics(
        truth,
        ensemble_frame["prediction_delta3d"].to_numpy(dtype=float),
        ensemble_frame["prediction_simplicial"].to_numpy(dtype=float),
        groups,
    )
    ensemble_bootstrap = paired_group_bootstrap(
        ensemble_frame,
        group_column=group_column,
        replicates=args.ensemble_bootstrap,
        seed=args.ensemble_bootstrap_seed,
    )
    # PASSED is persisted only after all aligned OOF calculations and the
    # recomputed bootstrap complete successfully.
    write_json_atomic(validation, output_dir / "validation.json")
    payload = {
        "validation": validation,
        "runs": run_rows,
        "across_seed_improvements": across_seed,
        "cross_seed_oof_ensemble": {
            "n_runs": len(run_rows),
            "improvements": ensemble_improvements,
            "paired_group_bootstrap": ensemble_bootstrap,
            "guardrail": (
                "Every constituent pair prediction is OOF under its own seeded group split; "
                "ensemble CI is recomputed and per-run CI bounds are never averaged."
            ),
        },
    }
    pd.DataFrame(run_rows).to_csv(output_dir / "per_run_metrics.csv", index=False)
    ensemble_frame.to_csv(output_dir / "cross_seed_oof_predictions.csv", index=False)
    write_json_atomic(payload, output_dir / "aggregate_summary.json")
    write_text_atomic(report_markdown(payload), output_dir / "report.md")
    (output_dir / "_INCOMPLETE").unlink()
    write_json_atomic(
        {
            "status": "complete",
            "pair_scope": args.expected_pair_scope,
            "aggregate_summary_sha256": file_sha256(output_dir / "aggregate_summary.json"),
            "artifact_sha256": {
                name: file_sha256(output_dir / name)
                for name in (
                    "cross_seed_oof_predictions.csv",
                    "per_run_metrics.csv",
                    "report.md",
                    "validation.json",
                )
            },
        },
        output_dir / "_SUCCESS.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
