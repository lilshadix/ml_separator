#!/usr/bin/env python3
"""Compare the real metal-site descriptor arm against its permuted null arm.

A descriptor block is only evidence of chemistry if its effect exceeds what an
equally wide block of meaningless continuous columns achieves on the same folds.
This script reads two validated aggregates produced by
``aggregate_delta3d_runs.py --comparison extended_vs_delta3d`` and reports the
real effect, the null effect, and their difference. It is fail-closed: if either
aggregate did not pass, or if the two arms do not share a cohort, protocol and
seed plan, no verdict is produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


IMPROVEMENT_METRICS = (
    "r2_gain",
    "group_balanced_r2_gain",
    "mae_reduction",
    "macro_group_mae_reduction",
    "sign_accuracy_gain",
)
# Group-level endpoints are primary: the dominant exact-ECFP cluster holds
# roughly half the pairs, so row-weighted R2 alone can be carried by one
# chemistry family.
PRIMARY_METRICS = ("group_balanced_r2_gain", "macro_group_mae_reduction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-aggregate", type=Path, required=True)
    parser.add_argument("--null-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text_atomic(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(payload: Any, path: Path) -> None:
    write_text_atomic(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", path)


def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:+.6f}"


def arm_values(payload: dict[str, Any], metric: str) -> list[float]:
    return [float(row[f"improvement_{metric}"]) for row in payload["runs"]]


def seed_plan(payload: dict[str, Any]) -> list[tuple[int, int]]:
    return sorted(
        (int(row["model_seed"]), int(row["split_seed"])) for row in payload["runs"]
    )


def failure_report(errors: list[str]) -> str:
    lines = [
        "# Metal-site descriptor arm comparison",
        "",
        "Status: FAILED (fail-closed)",
        "",
        "## Validation errors",
        "",
    ]
    lines.extend(f"- {error}" for error in errors)
    lines.extend(
        [
            "",
            "No verdict was produced because the two arms are not comparable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}
    for arm, path in (("real", args.real_aggregate), ("null", args.null_aggregate)):
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            errors.append(f"{arm} aggregate not found: {resolved}")
            continue
        try:
            payloads[arm] = read_json(resolved)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{arm} aggregate is unreadable: {exc}")

    if len(payloads) == 2:
        for arm, payload in payloads.items():
            validation = payload.get("validation", {})
            if not validation.get("passed"):
                errors.append(f"{arm} aggregate did not pass validation")
            if validation.get("comparison") != "extended_vs_delta3d":
                errors.append(f"{arm} aggregate is not the descriptor comparison")
            permutations = {
                str(row.get("descriptor_permutation")) for row in payload.get("runs", [])
            }
            if arm == "real" and permutations != {"none"}:
                errors.append(f"real arm carries a permutation: {sorted(permutations)}")
            if arm == "null" and (permutations == {"none"} or not permutations):
                errors.append("null arm is not permuted")

        real, null = payloads["real"], payloads["null"]
        if seed_plan(real) != seed_plan(null):
            errors.append("The two arms do not share one model/split seed plan.")
        for label, key in (
            ("cohort", "cohort_sha256"),
            ("protocol", "protocol"),
            ("dataset", "dataset_sha256"),
        ):
            real_value = real["validation"]["fingerprints"].get(key)
            null_value = null["validation"]["fingerprints"].get(key)
            if real_value != null_value:
                errors.append(f"The two arms do not share one {label} fingerprint.")

    if errors:
        write_text_atomic(failure_report(errors), output_dir / "arm_comparison.md")
        write_json_atomic(
            {"passed": False, "errors": errors}, output_dir / "arm_comparison.json"
        )
        return 2

    real, null = payloads["real"], payloads["null"]
    comparison: dict[str, Any] = {}
    for metric in IMPROVEMENT_METRICS:
        real_values = arm_values(real, metric)
        null_values = arm_values(null, metric)
        # Both arms use the same folds and the same seed plan, so the runs pair
        # up one to one and the paired difference is the quantity of interest.
        paired = [a - b for a, b in zip(real_values, null_values, strict=True)]
        comparison[metric] = {
            "real_mean": statistics.fmean(real_values),
            "null_mean": statistics.fmean(null_values),
            "excess_mean": statistics.fmean(paired),
            "excess_std": statistics.stdev(paired) if len(paired) > 1 else None,
            "real_positive_runs": sum(value > 0.0 for value in real_values),
            "excess_positive_runs": sum(value > 0.0 for value in paired),
            "n_runs": len(paired),
        }

    exceeds_null = {
        metric: bool(
            comparison[metric]["excess_mean"] > 0.0
            and comparison[metric]["excess_positive_runs"] == comparison[metric]["n_runs"]
        )
        for metric in PRIMARY_METRICS
    }
    verdict_passed = all(exceeds_null.values())
    verdict = (
        "The descriptor block exceeds its permuted null on every primary "
        "group-level endpoint in every run."
        if verdict_passed
        else "The descriptor block does not exceed its permuted null on all primary "
        "group-level endpoints; the raw gain is not distinguishable from the "
        "effect of adding continuous columns of the same width."
    )

    payload = {
        "passed": True,
        "verdict_supports_descriptor_block": verdict_passed,
        "verdict": verdict,
        "primary_metrics": list(PRIMARY_METRICS),
        "primary_metric_exceeds_null": exceeds_null,
        "n_runs": comparison[PRIMARY_METRICS[0]]["n_runs"],
        "descriptor_contract": real["validation"].get("descriptor_contract"),
        "null_descriptor_contract": null["validation"].get("descriptor_contract"),
        "metrics": comparison,
        "guardrail": (
            "The null arm is one permutation realisation per seed, not an exhaustive "
            "permutation distribution. A block that only narrowly exceeds the null "
            "should be re-tested with additional permutation seeds before it is "
            "reported."
        ),
    }
    write_json_atomic(payload, output_dir / "arm_comparison.json")

    lines = [
        "# Metal-site descriptor arm comparison",
        "",
        "Status: PASSED",
        "",
        f"Verdict: {verdict}",
        "",
        f"Runs per arm: {payload['n_runs']}. Primary endpoints: "
        f"{', '.join(PRIMARY_METRICS)}.",
        "",
        "| metric | real | permuted null | excess (paired) | excess std | excess positive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in IMPROVEMENT_METRICS:
        item = comparison[metric]
        lines.append(
            "| {metric} | {real} | {null} | {excess} | {std} | {positive}/{n} |".format(
                metric=metric,
                real=fmt(item["real_mean"]),
                null=fmt(item["null_mean"]),
                excess=fmt(item["excess_mean"]),
                std=fmt(item["excess_std"]),
                positive=item["excess_positive_runs"],
                n=item["n_runs"],
            )
        )
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "The `real` column is the gain of Delta3D plus the descriptor block over "
            "Delta3D alone. The `permuted null` column is the same gain when the "
            "descriptor values are detached from the geometry they describe, which "
            "isolates the effect of merely widening the feature matrix. Only the "
            "`excess` column carries a chemical claim.",
            "",
            payload["guardrail"],
            "",
        ]
    )
    write_text_atomic("\n".join(lines), output_dir / "arm_comparison.md")
    write_json_atomic(
        {
            "status": "complete",
            "arm_comparison_sha256": hashlib.sha256(
                (output_dir / "arm_comparison.json").read_bytes()
            ).hexdigest(),
        },
        output_dir / "_SUCCESS.json",
    )
    print(f"Verdict: {verdict}")
    for metric in PRIMARY_METRICS:
        item = comparison[metric]
        print(
            f"{metric}: real={item['real_mean']:+.6f} null={item['null_mean']:+.6f} "
            f"excess={item['excess_mean']:+.6f} "
            f"({item['excess_positive_runs']}/{item['n_runs']} runs positive)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
