#!/usr/bin/env python3
"""Fail-closed aggregation of completed multi-seed Delta3D benchmark runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


PRIMARY_COMPARISON = "primary_delta3d_vs_2d_ensemble"
BOOTSTRAP_COMPARISON = "delta3d_vs_2d_ensemble"

# Each entry names the improvement key, the bootstrap key, the two model
# branches whose OOF metrics are tabulated, and the protocol sentence that must
# be present in every run being pooled.
COMPARISONS: dict[str, dict[str, Any]] = {
    "delta3d_vs_2d_ensemble": {
        "improvement_key": "primary_delta3d_vs_2d_ensemble",
        "bootstrap_key": "delta3d_vs_2d_ensemble",
        "models": ("2d_ensemble", "delta3d"),
        "protocol_sentinel": "2D+2D",
        "description": "adaptive Delta3D blend minus matched adaptive 2D+2D blend",
    },
    "extended_vs_delta3d": {
        "improvement_key": "primary_extended_vs_delta3d",
        "bootstrap_key": "extended_vs_delta3d",
        "models": ("delta3d", "delta3d_extended"),
        "protocol_sentinel": "2D+2D",
        "description": (
            "adaptive Delta3D+metal-site-descriptor blend minus adaptive Delta3D blend"
        ),
    },
}
IMPROVEMENT_METRICS = (
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
    "feature_contract.json",
    "pair_build_audit.json",
    "oof_predictions.csv",
    "fold_assignments.csv",
    "inner_cv_tuning.csv",
    "per_extractant_metrics.csv",
    "per_pair_type_metrics.csv",
)
DESCRIPTOR_SEED_PLAN_FIELDS = (
    "arm",
    "task_index",
    "model_seed",
    "split_seed",
    "descriptor_permutation",
    "permutation_seed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument(
        "--expected-pair-scope",
        choices=("all", "adjacent"),
        required=True,
        help="Fail unless every run uses this explicitly requested pair cohort scope.",
    )
    parser.add_argument(
        "--descriptor-arm",
        choices=("real", "null"),
        default=None,
        help=(
            "Expected descriptor arm. By default this is inferred fail-closed from an "
            "arm_real or arm_null run-root basename."
        ),
    )
    parser.add_argument(
        "--descriptor-seed-plan",
        type=Path,
        default=None,
        help=(
            "Immutable descriptor seed_plan.tsv. Defaults to the parent of the arm "
            "run root."
        ),
    )
    parser.add_argument(
        "--comparison",
        choices=tuple(COMPARISONS),
        default="delta3d_vs_2d_ensemble",
        help=(
            "Which pre-declared contrast to aggregate. 'extended_vs_delta3d' pools the "
            "metal-site descriptor arm and requires every run to carry it."
        ),
    )
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


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def read_descriptor_seed_plan(path: Path) -> list[dict[str, Any]]:
    """Read and validate the immutable real/null descriptor seed contract."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != DESCRIPTOR_SEED_PLAN_FIELDS:
            raise ValueError(
                "Descriptor seed plan header must be "
                f"{list(DESCRIPTOR_SEED_PLAN_FIELDS)}."
            )
        for source_row in reader:
            arm = str(source_row["arm"])
            if arm not in {"real", "null"}:
                raise ValueError(f"Unknown descriptor arm in seed plan: {arm!r}.")
            permutation = str(source_row["descriptor_permutation"]).strip()
            if not permutation:
                raise ValueError("Descriptor seed plan contains an empty permutation mode.")
            try:
                task_index = int(source_row["task_index"])
                model_seed = int(source_row["model_seed"])
                split_seed = int(source_row["split_seed"])
                permutation_seed = int(source_row["permutation_seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Descriptor seed plan contains a non-integer seed field.") from exc
            if min(task_index, model_seed, split_seed, permutation_seed) < 0:
                raise ValueError("Descriptor seed plan integers must be nonnegative.")
            if max(model_seed, split_seed, permutation_seed) > 4_000_000_000:
                raise ValueError("Descriptor seed plan seeds must not exceed 4000000000.")
            rows.append(
                {
                    "arm": arm,
                    "task_index": task_index,
                    "model_seed": model_seed,
                    "split_seed": split_seed,
                    "descriptor_permutation": permutation,
                    "descriptor_permutation_seed": permutation_seed,
                }
            )

    if not rows:
        raise ValueError("Descriptor seed plan is empty.")
    task_indices = [int(row["task_index"]) for row in rows]
    if len(set(task_indices)) != len(task_indices):
        raise ValueError("Descriptor seed plan task indices are not unique.")
    if set(task_indices) != set(range(len(rows))):
        raise ValueError("Descriptor seed plan task indices must be contiguous from zero.")

    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ("real", "null")}
    if not by_arm["real"] or not by_arm["null"]:
        raise ValueError("Descriptor seed plan must contain both real and null arms.")
    for arm, arm_rows in by_arm.items():
        model_split = [
            (int(row["model_seed"]), int(row["split_seed"])) for row in arm_rows
        ]
        if len(set(model_split)) != len(model_split):
            raise ValueError(f"Descriptor seed plan has duplicate {arm} model/split pairs.")
    real_plan = {
        (int(row["model_seed"]), int(row["split_seed"])) for row in by_arm["real"]
    }
    null_plan = {
        (int(row["model_seed"]), int(row["split_seed"])) for row in by_arm["null"]
    }
    if real_plan != null_plan:
        raise ValueError("Real and null arms do not share one model/split seed plan.")
    if any(
        row["descriptor_permutation"] != "none"
        or int(row["descriptor_permutation_seed"]) != 0
        for row in by_arm["real"]
    ):
        raise ValueError("Real seed-plan rows must use permutation=none and seed=0.")
    if any(row["descriptor_permutation"] == "none" for row in by_arm["null"]):
        raise ValueError("Null seed-plan rows must use a non-none permutation.")
    null_modes = {str(row["descriptor_permutation"]) for row in by_arm["null"]}
    if len(null_modes) != 1:
        raise ValueError("Null seed-plan rows must share one permutation mode.")
    if not null_modes <= {"free", "metal-preserving"}:
        raise ValueError(
            f"Unsupported null descriptor permutation mode: {sorted(null_modes)}."
        )
    null_seeds = [int(row["descriptor_permutation_seed"]) for row in by_arm["null"]]
    if len(set(null_seeds)) != len(null_seeds):
        raise ValueError("Null descriptor permutation seeds must be unique.")
    return sorted(rows, key=lambda row: int(row["task_index"]))


def normalized_protocol(summary: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(nested(summary, "arguments"))
    # Permutation mode and seed define the descriptor arm, not the common model
    # protocol. They are validated independently against the immutable seed plan.
    for key in (
        "dataset",
        "output_dir",
        "seed",
        "split_seed",
        "n_jobs",
        "descriptor_permutation",
        "descriptor_permutation_seed",
    ):
        arguments.pop(key, None)
    protocol = dict(nested(summary, "benchmark", "protocol"))
    for key in ("model_seed", "split_seed"):
        protocol.pop(key, None)
    return {"arguments": arguments, "benchmark_protocol": protocol}


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    array = [float(value) for value in values]
    return {
        "n": len(array),
        "mean": statistics.fmean(array),
        "median": statistics.median(array),
        "sample_std": statistics.stdev(array) if len(array) >= 2 else None,
        "min": min(array),
        "max": max(array),
        "positive_count": sum(value > 0.0 for value in array),
        "nonpositive_count": sum(value <= 0.0 for value in array),
    }


def csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def failure_report(
    errors: list[str], warnings: list[str], expected_pair_scope: str
) -> str:
    lines = [
        "# Delta3D multi-seed aggregation",
        "",
        "Status: FAILED (fail-closed)",
        "",
        f"Expected pair scope: `{expected_pair_scope}`.",
        "",
    ]
    lines.extend(["## Validation errors", ""])
    lines.extend(f"- {error}" for error in errors)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "No aggregate scientific estimate was produced because the run set was not compatible.",
            "",
        ]
    )
    return "\n".join(lines)


def success_report(
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    lines = [
        "# Delta3D multi-seed aggregation",
        "",
        "Status: PASSED",
        "",
        f"Pair scope: `{validation['expected_pair_scope']}`.",
        "",
        f"Comparison: {validation['comparison_description']}.",
        "",
        "All runs have identical dataset, exact pair cohort, implementation, feature contract, "
        "and protocol fingerprints. Model and split seeds are intentionally different.",
        "",
    ]
    if validation.get("descriptor_arm"):
        lines.extend(
            [
                f"Descriptor arm: `{validation['descriptor_arm']}`; immutable seed plan: "
                f"`{validation['descriptor_seed_plan']['semantic_sha256']}`.",
                "",
            ]
        )
    permutations = sorted({str(row.get("descriptor_permutation", "")) for row in rows} - {""})
    if permutations:
        lines.extend(
            [
                f"Descriptor permutation arm: {', '.join(permutations)}; "
                f"descriptor columns: {rows[0].get('descriptor_column_count')}.",
                "",
            ]
        )
        if permutations != ["none"]:
            lines.extend(
                [
                    "**This is a negative-control aggregate.** Descriptor values were detached "
                    "from their geometry, so the numbers below measure how much any block of "
                    "continuous columns of this width moves the metrics. A real descriptor "
                    "effect must exceed this reference, not merely exceed zero.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Runs",
            "",
            "| model seed | split seed | Delta R2 | Delta balanced R2 | macro-MAE reduction | R2 CI |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {model_seed} | {split_seed} | {r2} | {balanced} | {macro} | [{low}, {high}] |".format(
                model_seed=row["model_seed"],
                split_seed=row["split_seed"],
                r2=fmt(row["improvement_r2_gain"]),
                balanced=fmt(row["improvement_group_balanced_r2_gain"]),
                macro=fmt(row["improvement_macro_group_mae_reduction"]),
                low=fmt(row["bootstrap_r2_gain_ci95_low"]),
                high=fmt(row["bootstrap_r2_gain_ci95_high"]),
            )
        )

    lines.extend(
        [
            "",
            "## Across seeded splits and model initializations",
            "",
            "| improvement | mean | median | std | min | max | positive runs | CI > 0 | CI crosses 0 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in IMPROVEMENT_METRICS:
        item = aggregate[metric]
        lines.append(
            "| {metric} | {mean} | {median} | {std} | {minimum} | {maximum} | "
            "{positive}/{n} | {strict}/{n} | {crosses}/{n} |".format(
                metric=metric,
                mean=fmt(item["mean"]),
                median=fmt(item["median"]),
                std=fmt(item["sample_std"]),
                minimum=fmt(item["min"]),
                maximum=fmt(item["max"]),
                positive=item["positive_count"],
                strict=item["bootstrap_ci_strict_positive_count"],
                crosses=item["bootstrap_ci_crosses_zero_count"],
                n=item["n"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "The table summarizes variability across prespecified model/split seeds. Each reported "
            "bootstrap interval is conditional on that run's fixed OOF predictions. The interval "
            "bounds are not averaged into a new confidence interval.",
            "",
            f"Validation fingerprint: `{validation['combined_validation_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Run root not found: {run_root}")
    if args.expected_runs is not None and args.expected_runs < 2:
        raise SystemExit("--expected-runs must be at least 2.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "aggregate"
    )
    comparison = COMPARISONS[args.comparison]
    required_artifacts = REQUIRED_ARTIFACTS
    if args.comparison == "extended_vs_delta3d":
        required_artifacts = REQUIRED_ARTIFACTS + ("geometry_descriptor_audit.json",)

    output_dir.mkdir(parents=True, exist_ok=False)
    errors: list[str] = []
    warnings: list[str] = []
    summaries: list[tuple[Path, dict[str, Any]]] = []
    descriptor_contracts: set[str] = set()
    descriptor_arm: str | None = None
    descriptor_seed_plan_path: Path | None = None
    descriptor_seed_plan_sha256: str | None = None
    descriptor_seed_plan_fingerprint: str | None = None
    descriptor_seed_plan_rows: list[dict[str, Any]] = []
    if args.comparison == "extended_vs_delta3d":
        descriptor_arm = args.descriptor_arm
        if descriptor_arm is None:
            if run_root.name in {"arm_real", "arm_null"}:
                descriptor_arm = run_root.name.removeprefix("arm_")
            else:
                errors.append(
                    "Cannot infer descriptor arm from run root; use --descriptor-arm "
                    "or name it arm_real/arm_null."
                )
        descriptor_seed_plan_path = (
            args.descriptor_seed_plan.expanduser().resolve()
            if args.descriptor_seed_plan is not None
            else run_root.parent / "seed_plan.tsv"
        )
        try:
            if not descriptor_seed_plan_path.is_file():
                raise ValueError(
                    f"descriptor seed plan not found: {descriptor_seed_plan_path}"
                )
            descriptor_seed_plan_rows = read_descriptor_seed_plan(
                descriptor_seed_plan_path
            )
            descriptor_seed_plan_sha256 = file_sha256(descriptor_seed_plan_path)
            descriptor_seed_plan_fingerprint = object_sha256(
                descriptor_seed_plan_rows
            )
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"Invalid immutable descriptor seed plan: {exc}")
    elif args.descriptor_arm is not None or args.descriptor_seed_plan is not None:
        errors.append(
            "--descriptor-arm/--descriptor-seed-plan require "
            "--comparison extended_vs_delta3d."
        )

    summary_paths = sorted(
        path
        for path in run_root.rglob("summary.json")
        if output_dir not in path.parents
    )
    if args.expected_runs is not None and len(summary_paths) != args.expected_runs:
        errors.append(
            f"Expected {args.expected_runs} summary files, found {len(summary_paths)}."
        )
    if len(summary_paths) < 2:
        errors.append("At least two completed run summaries are required.")

    for summary_path in summary_paths:
        run_dir = summary_path.parent
        missing = [name for name in required_artifacts if not (run_dir / name).is_file()]
        if missing:
            errors.append(f"{run_dir.name}: missing artifacts {missing}")
            continue
        try:
            summary = read_json(summary_path)
            success = read_json(run_dir / "_SUCCESS.json")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{run_dir.name}: invalid JSON: {exc}")
            continue
        observed_summary_hash = file_sha256(summary_path)
        if success.get("summary_sha256") != observed_summary_hash:
            errors.append(f"{run_dir.name}: _SUCCESS summary hash mismatch")
            continue
        summaries.append((summary_path, summary))

    if args.expected_runs is not None and len(summaries) != args.expected_runs:
        errors.append(
            f"Expected {args.expected_runs} validated completed runs, found {len(summaries)}."
        )

    dataset_hashes: set[str] = set()
    cohort_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    protocol_hashes: set[str] = set()
    audit_hashes: set[str] = set()
    feature_contract_hashes: set[str] = set()
    software_fingerprints: set[str] = set()
    observed_pair_scopes: set[str] = set()
    run_keys: set[tuple[int, int]] = set()
    model_seeds: set[int] = set()
    split_seeds: set[int] = set()
    rows: list[dict[str, Any]] = []

    for summary_path, summary in summaries:
        run_name = summary_path.parent.name
        try:
            pair_scope = str(nested(summary, "arguments", "pair_scope"))
            observed_pair_scopes.add(pair_scope)
            if pair_scope != args.expected_pair_scope:
                raise ValueError(
                    "pair scope mismatch: "
                    f"expected {args.expected_pair_scope!r}, observed {pair_scope!r}"
                )
            dataset_hash = str(nested(summary, "dataset_sha256"))
            cohort_hash = str(nested(summary, "cohort_sha256"))
            implementation = nested(summary, "implementation_sha256")
            if not isinstance(implementation, dict) or not implementation:
                raise ValueError("implementation_sha256 is empty")
            if any(not isinstance(value, str) or not value for value in implementation.values()):
                raise ValueError("implementation_sha256 contains an empty hash")
            protocol = normalized_protocol(summary)
            audit = nested(summary, "pair_build_audit")
            feature_contract = read_json(summary_path.parent / "feature_contract.json")
            model_seed = int(nested(summary, "arguments", "seed"))
            split_seed = int(nested(summary, "arguments", "split_seed"))
            run_key = (model_seed, split_seed)
            if run_key in run_keys:
                raise ValueError(f"duplicate model/split seed pair {run_key}")
            run_keys.add(run_key)
            model_seeds.add(model_seed)
            split_seeds.add(split_seed)

            if bool(nested(summary, "arguments", "quick")):
                raise ValueError("quick=True is not allowed in the scientific aggregate")
            if not bool(nested(summary, "arguments", "evaluation_only")):
                warnings.append(f"{run_name}: deployment fit was enabled but is not aggregated")
            if not bool(nested(summary, "benchmark", "leakage_audit", "passed")):
                raise ValueError("leakage audit did not pass")
            primary_text = str(nested(summary, "benchmark", "protocol", "primary_comparison"))
            if comparison["protocol_sentinel"] not in primary_text:
                raise ValueError("matched 2D+2D primary comparison is missing")

            if args.comparison == "extended_vs_delta3d":
                model_names = nested(summary, "benchmark", "model_names")
                if "delta3d_extended" not in list(model_names):
                    raise ValueError("run has no metal-site descriptor arm")
                # Pooling runs whose descriptor block, profile or permutation
                # differ would silently average different experiments.
                descriptor_contracts.add(
                    canonical_json(
                        {
                            "blocks": list(
                                nested(summary, "arguments", "descriptor_blocks")
                            ),
                            "profile": str(
                                nested(summary, "arguments", "descriptor_profile")
                            ),
                            "permutation": str(
                                nested(summary, "arguments", "descriptor_permutation")
                            ),
                            "columns": list(
                                nested(summary, "geometry_descriptor_audit", "descriptor_columns")
                            ),
                        }
                    )
                )
                row_permutation_seed = int(
                    nested(summary, "arguments", "descriptor_permutation_seed")
                )

            dataset_hashes.add(dataset_hash)
            cohort_hashes.add(cohort_hash)
            implementation_hashes.add(object_sha256(implementation))
            protocol_hashes.add(object_sha256(protocol))
            audit_hashes.add(object_sha256(audit))
            feature_contract_hashes.add(object_sha256(feature_contract))
            software_fingerprints.add(object_sha256(nested(summary, "software")))

            row: dict[str, Any] = {
                "run_dir": str(summary_path.parent),
                "created_at_utc": summary.get("created_at_utc"),
                "model_seed": model_seed,
                "split_seed": split_seed,
                "bootstrap_replicates": int(
                    nested(summary, "benchmark", "paired_group_bootstrap", "replicates")
                ),
                "pair_rows": int(nested(summary, "pair_build_audit", "pair_rows")),
                "pair_extractants": int(
                    nested(summary, "pair_build_audit", "pair_extractants")
                ),
                "pair_ecfp_exact_clusters": int(
                    nested(summary, "pair_build_audit", "pair_ecfp_exact_clusters")
                ),
                "fold_assignments_sha256": file_sha256(
                    summary_path.parent / "fold_assignments.csv"
                ),
                "leakage_passed": True,
            }
            if args.comparison == "extended_vs_delta3d":
                row["descriptor_permutation"] = str(
                    nested(summary, "arguments", "descriptor_permutation")
                )
                row["descriptor_permutation_seed"] = row_permutation_seed
                row["descriptor_column_count"] = int(
                    nested(
                        summary,
                        "benchmark",
                        "feature_counts",
                        "metal_site_descriptors_added",
                    )
                )
            for model_name in comparison["models"]:
                for metric in (
                    "r2",
                    "group_balanced_r2",
                    "mae",
                    "macro_group_mae",
                    "sign_accuracy",
                ):
                    label = f"benchmark.metrics.{model_name}.{metric}"
                    row[f"{model_name}_{metric}"] = finite_number(
                        nested(summary, "benchmark", "metrics", model_name, metric), label
                    )

            for metric in IMPROVEMENT_METRICS:
                point = finite_number(
                    nested(
                        summary,
                        "benchmark",
                        "improvements",
                        comparison["improvement_key"],
                        metric,
                    ),
                    f"{run_name}.{metric}",
                )
                interval = nested(
                    summary,
                    "benchmark",
                    "paired_group_bootstrap",
                    "comparisons",
                    comparison["bootstrap_key"],
                    metric,
                )
                mean = finite_number(interval["mean"], f"{run_name}.{metric}.mean")
                low = finite_number(interval["ci95_low"], f"{run_name}.{metric}.ci95_low")
                high = finite_number(interval["ci95_high"], f"{run_name}.{metric}.ci95_high")
                if low > high:
                    raise ValueError(f"{metric} interval has low > high")
                row[f"improvement_{metric}"] = point
                row[f"bootstrap_{metric}_mean"] = mean
                row[f"bootstrap_{metric}_ci95_low"] = low
                row[f"bootstrap_{metric}_ci95_high"] = high
                row[f"bootstrap_{metric}_ci_crosses_zero"] = low <= 0.0 <= high
            rows.append(row)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"{run_name}: {exc}")

    for label, values in (
        ("dataset SHA-256", dataset_hashes),
        ("cohort SHA-256", cohort_hashes),
        ("implementation fingerprint", implementation_hashes),
        ("protocol fingerprint", protocol_hashes),
        ("pair audit fingerprint", audit_hashes),
        ("feature contract fingerprint", feature_contract_hashes),
    ):
        if len(values) > 1:
            errors.append(f"Runs do not share one {label}.")
    if summaries and len(rows) != len(summaries):
        errors.append("One or more completed summaries failed metric validation.")
    if len(model_seeds) != len(rows):
        errors.append("Model seeds must be unique across the prespecified runs.")
    if len(split_seeds) != len(rows):
        errors.append("Split seeds must be unique across the prespecified runs.")
    if len(software_fingerprints) > 1:
        warnings.append("Software/platform metadata differs across runs.")
    if len(descriptor_contracts) > 1:
        errors.append(
            "Runs do not share one metal-site descriptor contract "
            "(block, profile or permutation differs); they are different experiments."
        )
    descriptor_arm_contract: dict[str, Any] | None = None
    if args.comparison == "extended_vs_delta3d" and rows:
        permutations = {str(row["descriptor_permutation"]) for row in rows}
        permutation_seeds = [int(row["descriptor_permutation_seed"]) for row in rows]
        if any(seed < 0 for seed in permutation_seeds):
            errors.append("Descriptor permutation seeds must be nonnegative.")
        if descriptor_arm == "real":
            if permutations != {"none"} or any(seed != 0 for seed in permutation_seeds):
                errors.append(
                    "Real descriptor arm must use permutation=none and seed=0 in every run."
                )
        elif descriptor_arm == "null":
            if not permutations or "none" in permutations:
                errors.append("Null descriptor arm must use a non-none permutation.")
            if len(permutation_seeds) != len(set(permutation_seeds)):
                errors.append("Null descriptor permutation seeds must be unique.")

        observed_arm_rows = sorted(
            (
                int(row["model_seed"]),
                int(row["split_seed"]),
                str(row["descriptor_permutation"]),
                int(row["descriptor_permutation_seed"]),
            )
            for row in rows
        )
        expected_arm_rows = sorted(
            (
                int(row["model_seed"]),
                int(row["split_seed"]),
                str(row["descriptor_permutation"]),
                int(row["descriptor_permutation_seed"]),
            )
            for row in descriptor_seed_plan_rows
            if row["arm"] == descriptor_arm
        )
        if descriptor_seed_plan_rows and observed_arm_rows != expected_arm_rows:
            errors.append(
                f"Observed {descriptor_arm} runs do not exactly match immutable "
                "descriptor seed-plan rows."
            )
        if (
            args.expected_runs is not None
            and expected_arm_rows
            and len(expected_arm_rows) != args.expected_runs
        ):
            errors.append(
                "Descriptor seed plan arm cardinality does not match --expected-runs: "
                f"plan={len(expected_arm_rows)}, expected={args.expected_runs}."
            )
        descriptor_arm_contract = {
            "arm": descriptor_arm,
            "observed_rows": [list(row) for row in observed_arm_rows],
            "expected_rows": [list(row) for row in expected_arm_rows],
            "validated_against_seed_plan": bool(
                descriptor_seed_plan_rows and observed_arm_rows == expected_arm_rows
            ),
            "sha256": object_sha256(
                {"arm": descriptor_arm, "rows": observed_arm_rows}
            ),
        }
        if len(permutations) == 1 and permutations != {"none"}:
            warnings.append(
                "Every run in this aggregate is a permuted negative control; the "
                "result describes the descriptor-width artefact, not chemistry."
            )

    validation = {
        "passed": not errors,
        "expected_pair_scope": args.expected_pair_scope,
        "observed_pair_scopes": sorted(observed_pair_scopes),
        "comparison": args.comparison,
        "comparison_description": comparison["description"],
        "descriptor_contract": sorted(descriptor_contracts),
        "descriptor_arm": descriptor_arm,
        "descriptor_arm_contract": descriptor_arm_contract,
        "descriptor_seed_plan": {
            "path": str(descriptor_seed_plan_path)
            if descriptor_seed_plan_path is not None
            else None,
            "file_sha256": descriptor_seed_plan_sha256,
            "semantic_sha256": descriptor_seed_plan_fingerprint,
            "validated": bool(
                descriptor_seed_plan_fingerprint
                and descriptor_arm_contract
                and descriptor_arm_contract["validated_against_seed_plan"]
            ),
        }
        if args.comparison == "extended_vs_delta3d"
        else None,
        "expected_runs": args.expected_runs,
        "discovered_summaries": len(summary_paths),
        "validated_completed_runs": len(rows),
        "errors": errors,
        "warnings": warnings,
        "fingerprints": {
            "dataset_sha256": sorted(dataset_hashes),
            "cohort_sha256": sorted(cohort_hashes),
            "implementation": sorted(implementation_hashes),
            "protocol": sorted(protocol_hashes),
            "pair_audit": sorted(audit_hashes),
            "feature_contract": sorted(feature_contract_hashes),
            "descriptor_seed_plan": (
                [descriptor_seed_plan_fingerprint]
                if descriptor_seed_plan_fingerprint is not None
                else []
            ),
        },
    }
    validation["combined_validation_sha256"] = object_sha256(
        {
            "expected_pair_scope": args.expected_pair_scope,
            "observed_pair_scopes": sorted(observed_pair_scopes),
            "descriptor_arm": descriptor_arm,
            "descriptor_arm_contract": descriptor_arm_contract,
            "fingerprints": validation["fingerprints"],
        }
    )
    write_json_atomic(validation, output_dir / "validation.json")

    if errors:
        write_text_atomic(
            failure_report(errors, warnings, args.expected_pair_scope),
            output_dir / "report.md",
        )
        return 2

    rows.sort(key=lambda row: (int(row["split_seed"]), int(row["model_seed"])))
    aggregate: dict[str, Any] = {}
    for metric in IMPROVEMENT_METRICS:
        item = describe(row[f"improvement_{metric}"] for row in rows)
        item["bootstrap_ci_strict_positive_count"] = sum(
            row[f"bootstrap_{metric}_ci95_low"] > 0.0 for row in rows
        )
        item["bootstrap_ci_strict_negative_count"] = sum(
            row[f"bootstrap_{metric}_ci95_high"] < 0.0 for row in rows
        )
        item["bootstrap_ci_crosses_zero_count"] = sum(
            bool(row[f"bootstrap_{metric}_ci_crosses_zero"]) for row in rows
        )
        aggregate[metric] = item

    payload = {
        "validation": validation,
        "runs": rows,
        "aggregate_improvements": aggregate,
        "uncertainty_guardrail": (
            "Across-run summaries describe prespecified split/model-seed variability. "
            "Per-run fixed-OOF bootstrap intervals remain separate and are not averaged."
        ),
    }
    write_text_atomic(csv_text(rows), output_dir / "per_run_metrics.csv")
    write_json_atomic(payload, output_dir / "aggregate_summary.json")
    write_text_atomic(
        success_report(rows, aggregate, validation), output_dir / "report.md"
    )
    write_json_atomic(
        {
            "status": "complete",
            "pair_scope": args.expected_pair_scope,
            "aggregate_summary_sha256": file_sha256(output_dir / "aggregate_summary.json"),
        },
        output_dir / "_SUCCESS.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
