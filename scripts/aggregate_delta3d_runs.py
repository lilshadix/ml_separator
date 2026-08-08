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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-runs", type=int, default=None)
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


def normalized_protocol(summary: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(nested(summary, "arguments"))
    for key in ("dataset", "output_dir", "seed", "split_seed", "n_jobs"):
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


def failure_report(errors: list[str], warnings: list[str]) -> str:
    lines = ["# Delta3D multi-seed aggregation", "", "Status: FAILED (fail-closed)", ""]
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
        f"Comparison: {validation['comparison_description']}.",
        "",
        "All runs have identical dataset, exact pair cohort, implementation, feature contract, "
        "and protocol fingerprints. Model and split seeds are intentionally different.",
        "",
    ]
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
    run_keys: set[tuple[int, int]] = set()
    model_seeds: set[int] = set()
    split_seeds: set[int] = set()
    rows: list[dict[str, Any]] = []

    for summary_path, summary in summaries:
        run_name = summary_path.parent.name
        try:
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
    if args.comparison == "extended_vs_delta3d" and rows:
        permutations = {str(row["descriptor_permutation"]) for row in rows}
        if len(permutations) == 1 and permutations != {"none"}:
            warnings.append(
                "Every run in this aggregate is a permuted negative control; the "
                "result describes the descriptor-width artefact, not chemistry."
            )

    validation = {
        "passed": not errors,
        "comparison": args.comparison,
        "comparison_description": comparison["description"],
        "descriptor_contract": sorted(descriptor_contracts),
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
        },
    }
    validation["combined_validation_sha256"] = object_sha256(validation["fingerprints"])
    write_json_atomic(validation, output_dir / "validation.json")

    if errors:
        write_text_atomic(failure_report(errors, warnings), output_dir / "report.md")
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
            "aggregate_summary_sha256": file_sha256(output_dir / "aggregate_summary.json"),
        },
        output_dir / "_SUCCESS.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
