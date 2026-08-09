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


def descriptor_arm_errors(arm: str, payload: dict[str, Any]) -> list[str]:
    """Validate the arm-specific contract without weakening the common protocol."""

    errors: list[str] = []
    validation = payload.get("validation", {})
    runs = payload.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return [f"{arm} aggregate has no run rows"]

    expected_scope = validation.get("expected_pair_scope")
    observed_scopes = validation.get("observed_pair_scopes")
    if expected_scope not in {"all", "adjacent"}:
        errors.append(f"{arm} aggregate has no valid expected pair scope")
    if observed_scopes != [expected_scope]:
        errors.append(
            f"{arm} aggregate pair-scope validation is inconsistent: "
            f"expected={expected_scope!r}, observed={observed_scopes!r}"
        )

    modes: list[str] = []
    seeds: list[int] = []
    contract_rows: list[tuple[int, int, str, int]] = []
    for index, row in enumerate(runs):
        if not isinstance(row, dict):
            errors.append(f"{arm} run row {index} is not an object")
            continue
        modes.append(str(row.get("descriptor_permutation", "")))
        try:
            seed = int(row["descriptor_permutation_seed"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{arm} run row {index} has an invalid permutation seed")
            continue
        if seed < 0:
            errors.append(f"{arm} run row {index} has a negative permutation seed")
        seeds.append(seed)
        try:
            contract_rows.append(
                (
                    int(row["model_seed"]),
                    int(row["split_seed"]),
                    modes[-1],
                    seed,
                )
            )
        except (KeyError, TypeError, ValueError):
            errors.append(f"{arm} run row {index} has an invalid model/split seed")

    mode_set = set(modes)
    if arm == "real":
        if mode_set != {"none"} or len(seeds) != len(runs) or any(seed != 0 for seed in seeds):
            errors.append("real arm must use permutation=none and seed=0 in every run")
    else:
        if not mode_set or "" in mode_set or "none" in mode_set:
            errors.append("null arm must use a non-none permutation in every run")
        if len(mode_set) != 1:
            errors.append("null arm must use one shared permutation mode")
        if not mode_set <= {"free", "metal-preserving"}:
            errors.append(f"null arm uses an unsupported permutation: {sorted(mode_set)}")
        if len(seeds) != len(runs) or len(set(seeds)) != len(seeds):
            errors.append("null arm permutation seeds must be present and unique")

    arm_contract = validation.get("descriptor_arm_contract", {})
    if not isinstance(arm_contract, dict):
        errors.append(f"{arm} aggregate descriptor arm contract is missing")
    else:
        if arm_contract.get("arm") != arm:
            errors.append(f"{arm} aggregate is labelled as arm={arm_contract.get('arm')!r}")
        if not arm_contract.get("validated_against_seed_plan"):
            errors.append(f"{arm} aggregate was not validated against its descriptor seed plan")
        normalized_rows = [list(row) for row in sorted(contract_rows)]
        if arm_contract.get("observed_rows") != normalized_rows:
            errors.append(f"{arm} aggregate arm contract disagrees with its run rows")
        if arm_contract.get("expected_rows") != normalized_rows:
            errors.append(f"{arm} aggregate run rows disagree with its expected seed-plan rows")
        expected_contract_sha256 = hashlib.sha256(
            canonical_json({"arm": arm, "rows": sorted(contract_rows)}).encode("utf-8")
        ).hexdigest()
        if arm_contract.get("sha256") != expected_contract_sha256:
            errors.append(f"{arm} aggregate arm-contract fingerprint is invalid")
    seed_contract = validation.get("descriptor_seed_plan", {})
    if not isinstance(seed_contract, dict) or not seed_contract.get("validated"):
        errors.append(f"{arm} aggregate has no validated immutable descriptor seed plan")
    if not isinstance(seed_contract, dict) or not seed_contract.get("semantic_sha256"):
        errors.append(f"{arm} aggregate has no descriptor seed-plan fingerprint")
    elif validation.get("fingerprints", {}).get("descriptor_seed_plan") != [
        seed_contract["semantic_sha256"]
    ]:
        errors.append(f"{arm} aggregate seed-plan fingerprints are inconsistent")
    return errors


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
            errors.extend(descriptor_arm_errors(arm, payload))

        real, null = payloads["real"], payloads["null"]
        try:
            if seed_plan(real) != seed_plan(null):
                errors.append("The two arms do not share one model/split seed plan.")
        except (KeyError, TypeError, ValueError):
            errors.append("The two arms do not carry a valid model/split seed plan.")
        real_scope = real.get("validation", {}).get("expected_pair_scope")
        null_scope = null.get("validation", {}).get("expected_pair_scope")
        if real_scope != null_scope:
            errors.append(
                "The two arms do not share one expected pair scope: "
                f"real={real_scope!r}, null={null_scope!r}."
            )
        for label, key in (
            ("cohort", "cohort_sha256"),
            ("protocol", "protocol"),
            ("dataset", "dataset_sha256"),
            ("immutable descriptor seed plan", "descriptor_seed_plan"),
        ):
            real_value = real.get("validation", {}).get("fingerprints", {}).get(key)
            null_value = null.get("validation", {}).get("fingerprints", {}).get(key)
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
        "pair_scope": real["validation"]["expected_pair_scope"],
        "descriptor_seed_plan_sha256": real["validation"]["descriptor_seed_plan"][
            "semantic_sha256"
        ],
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
