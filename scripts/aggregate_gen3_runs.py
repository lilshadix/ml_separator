#!/usr/bin/env python3
"""Validate and aggregate the five frozen primary generation-3 split tasks."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lanthanide_separation.gen3_protocol import (  # noqa: E402
    file_sha256,
    load_gen3_protocol,
)


BASELINE_ARM = "A2_current_champion"


def _expected_arms(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    h1 = tuple(
        f"H1_CATBOOST_{weighting.upper()}_{loss.upper()}"
        for weighting in protocol["h1"]["weighting_schemes"]
        for loss in protocol["h1"]["losses"]
    )
    h2 = tuple(protocol["h2"]["full_e3_arms"]) + tuple(
        protocol["h2"]["stability_followup"]["arms"]
    )
    h3 = tuple(protocol["h3"]["arms"]) + tuple(
        protocol["h3"]["label_shuffle_control_arms"]
    )
    return (BASELINE_ARM,) + h1 + h2 + h3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "gen3_protocol.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(payload: Any, path: Path) -> None:
    write_text_atomic(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        path,
    )


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_run(
    run_dir: Path, protocol: Mapping[str, Any], protocol_sha256: str
) -> dict[str, Any]:
    success = read_json(run_dir / "_SUCCESS.json")
    validation = read_json(run_dir / "validation.json")
    metadata = read_json(run_dir / "run_metadata.json")
    run_config = read_json(run_dir / "run_config.json")
    dataset_hashes = read_json(run_dir / "dataset_hashes.json")
    feature_hashes = read_json(run_dir / "feature_hashes.json")
    leakage = read_json(run_dir / "leakage_audit.json")
    hashes = read_json(run_dir / "artifact_hashes.json")
    required_artifacts = {
        "gen3_protocol.json",
        "dataset_hashes.json",
        "feature_hashes.json",
        "split_assignments.csv",
        "inner_fold_assignments.csv",
        "oof_predictions.csv",
        "per_extractant_metrics.csv",
        "per_seed_metrics.csv",
        "paired_bootstrap.csv",
        "e3_stability.csv",
        "leakage_audit.json",
        "validation.json",
        "report.md",
    }
    missing_artifacts = sorted(required_artifacts - set(hashes))
    if missing_artifacts:
        raise ValueError(f"required hashed artifacts are absent: {missing_artifacts}")
    if success.get("status") != "complete" or validation.get("passed") is not True:
        raise ValueError("run is not complete and validation-passed")
    if run_config.get("status") != "complete":
        raise ValueError("run_config is not marked complete")
    if run_config.get("software_contract", {}).get("passed") is not True:
        raise ValueError("run software contract did not pass")
    if bool(success.get("quick")) or bool(validation.get("quick")):
        raise ValueError("quick smoke output is ineligible for scientific aggregation")
    if success.get("protocol_sha256") != protocol_sha256:
        raise ValueError("protocol SHA-256 mismatch")
    inputs = protocol["immutable_inputs"]
    if dataset_hashes.get("dataset_sha256") != inputs["dataset_sha256"]:
        raise ValueError("dataset SHA-256 differs from the frozen protocol")
    if dataset_hashes.get("pair_cohort_sha256") != inputs["pair_cohort_sha256"]:
        raise ValueError("pair cohort SHA-256 differs from the frozen protocol")
    if feature_hashes.get("a2_feature_count") != inputs["a2_feature_count"]:
        raise ValueError("A2 feature count differs from the frozen protocol")
    if feature_hashes.get("e3_raw_feature_count") != inputs["e3_raw_feature_count"]:
        raise ValueError("E3 feature count differs from the frozen protocol")
    if feature_hashes.get("frozen_gen2_feature_contract_sha256") != inputs[
        "gen2_feature_contract_sha256"
    ]:
        raise ValueError("frozen gen2 feature-contract SHA-256 mismatch")
    if leakage.get("passed") is not True or leakage.get(
        "stage1_residuals_out_of_fold_only"
    ) is not True:
        raise ValueError("leakage/residual audit did not pass")
    if tuple(metadata.get("arms", ())) != _expected_arms(protocol):
        raise ValueError("run arm contract differs from the frozen protocol")
    software = run_config.get("software")
    if not isinstance(software, Mapping):
        raise ValueError("run_config.software is missing")
    environment = protocol["reproducibility_environment"]
    for key in (
        "python",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "pyarrow",
        "joblib",
        "torch",
        "catboost",
    ):
        if str(software.get(key)) != str(environment.get(key)):
            raise ValueError(
                f"software version mismatch for {key}: expected "
                f"{environment.get(key)!r}, observed {software.get(key)!r}"
            )
    if "Linux" not in str(software.get("platform")) or "x86_64" not in str(
        software.get("platform")
    ):
        raise ValueError("scientific gen3 runs require the frozen Linux x86_64 platform")
    implementation = run_config.get("implementation_sha256")
    if not isinstance(implementation, Mapping) or not implementation:
        raise ValueError("implementation_sha256 is missing from run_config")
    for name, digest in implementation.items():
        if len(str(digest)) != 64:
            raise ValueError(f"invalid implementation digest: {name}")
        current_path = REPO_ROOT / str(name)
        if not current_path.is_file() or file_sha256(current_path) != str(digest):
            raise ValueError(f"current implementation differs from run contract: {name}")
    if file_sha256(run_dir / "gen3_protocol.json") != protocol_sha256:
        raise ValueError("copied gen3_protocol.json differs from the frozen protocol")
    for name, expected_hash in hashes.items():
        path = run_dir / str(name)
        if not path.is_file():
            raise ValueError(f"declared artifact is absent: {name}")
        if file_sha256(path) != str(expected_hash):
            raise ValueError(f"artifact hash mismatch: {name}")
    if file_sha256(run_dir / "artifact_hashes.json") != success.get(
        "artifact_hashes_sha256"
    ):
        raise ValueError("artifact_hashes.json self-declaration mismatch")
    if file_sha256(run_dir / "validation.json") != success.get("validation_sha256"):
        raise ValueError("validation.json self-declaration mismatch")
    if int(success.get("artifact_count", -1)) != len(hashes):
        raise ValueError("artifact count differs from _SUCCESS.json")
    metrics = pd.read_csv(run_dir / "per_seed_metrics.csv")
    predictions = pd.read_csv(run_dir / "oof_predictions.csv")
    per_extractant = pd.read_csv(run_dir / "per_extractant_metrics.csv")
    family_metrics = pd.read_csv(run_dir / "extractant_family_metrics.csv")
    stability = pd.read_csv(run_dir / "e3_stability.csv")
    lambdas = pd.read_csv(run_dir / "selected_lambdas.csv")
    split_assignments = pd.read_csv(run_dir / "split_assignments.csv")
    inner_fold_assignments = pd.read_csv(run_dir / "inner_fold_assignments.csv")
    if metrics["arm"].duplicated().any():
        raise ValueError("per_seed_metrics has duplicate arms")
    if tuple(metrics["arm"]) != _expected_arms(protocol):
        raise ValueError("per_seed_metrics arm order/contract is invalid")
    if len(predictions) != 6699 or predictions["pair_id"].duplicated().any():
        raise ValueError("OOF prediction row contract is not 6,699 unique pairs")
    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "validation": validation,
        "implementation_sha256": dict(implementation),
        "dataset_hashes": dataset_hashes,
        "feature_hashes": feature_hashes,
        "metrics": metrics,
        "predictions": predictions,
        "per_extractant": per_extractant,
        "family_metrics": family_metrics,
        "stability": stability,
        "lambdas": lambdas,
        "split_assignments": split_assignments,
        "inner_fold_assignments": inner_fold_assignments,
    }


def _paired_multisplit_bootstrap(
    per_extractant: pd.DataFrame,
    comparisons: Mapping[str, tuple[str, str]],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    extractants = sorted(per_extractant["extractant"].astype(str).unique())
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for comparison, (reference_arm, candidate_arm) in comparisons.items():
        reference = per_extractant[per_extractant["arm"].eq(reference_arm)][
            ["split_seed", "extractant", "mae"]
        ].rename(columns={"mae": "reference_mae"})
        candidate = per_extractant[per_extractant["arm"].eq(candidate_arm)][
            ["split_seed", "extractant", "mae"]
        ].rename(columns={"mae": "candidate_mae"})
        paired = reference.merge(
            candidate, on=["split_seed", "extractant"], validate="one_to_one"
        )
        paired["delta"] = paired["reference_mae"] - paired["candidate_mae"]
        group_delta = (
            paired.groupby("extractant", sort=True)["delta"]
            .mean()
            .reindex(extractants)
            .to_numpy(dtype=float)
        )
        draws = np.empty(int(replicates), dtype=float)
        for index in range(int(replicates)):
            sampled = rng.integers(0, len(group_delta), size=len(group_delta))
            draws[index] = float(group_delta[sampled].mean())
        rows.append(
            {
                "comparison": comparison,
                "reference": reference_arm,
                "candidate": candidate_arm,
                "point_delta_mae": float(group_delta.mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "p_candidate_better": float(np.mean(draws > 0.0)),
                "extractants_improved": int(np.sum(group_delta > 0.0)),
                "extractants_total": int(len(group_delta)),
                "split_seeds": int(paired["split_seed"].nunique()),
                "bootstrap_replicates": int(replicates),
                "bootstrap_seed": int(seed),
                "bootstrap_unit": "extractant",
                "split_seed_weighting": "one split seed one vote within extractant",
                "multiplicity_preserved": True,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_stability(stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    real = stability[
        stability["selection_source_arm"].eq("H2_E3_RAW_REAL")
    ].copy()
    for feature, group in real.groupby("feature", sort=False):
        per_split = group.groupby("split_seed", sort=True).agg(
            selection_frequency=("selection_frequency", "mean"),
            selected=("selected", "mean"),
        )
        rows.append(
            {
                "feature": str(feature),
                "selection_frequency": float(group["selection_frequency"].mean()),
                "mean_signed_importance_or_coefficient": float(
                    group["mean_signed_importance_or_coefficient"].mean()
                ),
                "fold_consistency": float(group["fold_consistency"].mean()),
                "split_seed_consistency": float(
                    np.mean(per_split["selection_frequency"].to_numpy() >= 0.6)
                ),
                "selected_outer_fold_fraction": float(group["selected"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_frequency", "fold_consistency"], ascending=False
    )


def _markdown_table(frame: pd.DataFrame, maximum_rows: int = 50) -> str:
    view = frame.head(maximum_rows)
    if view.empty:
        return "_No rows._"
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        values = [
            f"{value:.6f}" if isinstance(value, float) and np.isfinite(value) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _decision_table(
    metrics: pd.DataFrame,
    lambdas: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    baseline_by_seed = metrics[metrics["arm"].eq(BASELINE_ARM)].set_index("split_seed")
    rows: list[dict[str, Any]] = []
    control_pairs = {
        "H2_E3_RAW_REAL": "H2_E3_RAW_SHUFFLED",
        "H2_E3_COMPACT_REAL": "H2_E3_COMPACT_SHUFFLED",
        "H2_E3_STABLE_REAL": "H2_E3_STABLE_SHUFFLED",
        "H3_A2": "H3_A2_LABEL_SHUFFLED",
        "H3_A2_ELEC": "H3_A2_ELEC_LABEL_SHUFFLED",
    }
    never_champion = set(control_pairs.values())
    for arm, group in metrics.groupby("arm", sort=False):
        if arm == BASELINE_ARM:
            continue
        aligned = group.set_index("split_seed")
        delta = (
            baseline_by_seed["equal_extractant_macro_mae"]
            - aligned["equal_extractant_macro_mae"]
        )
        positive_seeds = int(np.sum(delta > 0.0))
        mean_improves = bool(
            aligned["equal_extractant_macro_mae"].mean()
            < baseline_by_seed["equal_extractant_macro_mae"].mean()
        )
        win_rule = bool(mean_improves and positive_seeds >= 4)
        residual_gate = True
        if arm.startswith("H2_"):
            arm_lambdas = lambdas[lambdas["arm"].eq(arm)]
            nonzero_by_seed = arm_lambdas.groupby("split_seed")["lambda"].apply(
                lambda values: float(np.mean(values > 0.0)) >= 0.5
            )
            residual_gate = int(nonzero_by_seed.sum()) >= 4
        control_gate = True
        control = control_pairs.get(str(arm))
        if control is not None:
            control_metrics = metrics[metrics["arm"].eq(control)].set_index("split_seed")
            real_better = (
                control_metrics["equal_extractant_macro_mae"]
                - aligned["equal_extractant_macro_mae"]
            )
            control_gate = bool(real_better.mean() > 0.0 and np.sum(real_better > 0.0) >= 4)
        label_shuffle_gate = True
        if str(arm) in {"H3_A2", "H3_A2_ELEC"} and win_rule:
            label_shuffle_gate = control_gate
        eligible = bool(
            win_rule
            and residual_gate
            and control_gate
            and label_shuffle_gate
            and arm not in never_champion
        )
        rows.append(
            {
                "arm": arm,
                "mean_equal_extractant_macro_mae": float(
                    aligned["equal_extractant_macro_mae"].mean()
                ),
                "mean_delta_mae_baseline_minus_candidate": float(delta.mean()),
                "positive_split_seeds": positive_seeds,
                "split_seed_count": int(len(delta)),
                "win_rule_passed": win_rule,
                "h2_nonzero_lambda_gate": residual_gate,
                "real_beats_shuffled_gate": control_gate,
                "h3_label_shuffle_gate": label_shuffle_gate,
                "eligible_for_champion": eligible,
            }
        )
    return pd.DataFrame(rows)


def _report(
    leaderboard: pd.DataFrame,
    decision: pd.DataFrame,
    bootstrap: pd.DataFrame,
    per_extractant: pd.DataFrame,
    family_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    champion: str,
) -> str:
    h1 = leaderboard[leaderboard["arm"].str.startswith("H1_")]
    h2 = leaderboard[leaderboard["arm"].str.startswith("H2_")]
    h3 = leaderboard[leaderboard["arm"].str.startswith("H3_")]
    controls = decision[decision["arm"].str.contains("SHUFFLED")]
    pooled_order = leaderboard.sort_values("pooled_micro_mae")["arm"].tolist()
    macro_order = leaderboard["arm"].tolist()
    inversion = pooled_order != macro_order
    return "\n".join(
        [
            "# Primary leaderboard",
            "",
            "Five predeclared leave-extractants-out split seeds; one split seed has one "
            "vote. Primary metric is equal-extractant macro MAE (lower is better).",
            "",
            _markdown_table(leaderboard),
            "",
            "Pooled MAE and pooled R2 are row-weighted descriptive statistics. They were "
            "not used for hyperparameter, arm, or champion selection.",
            "",
            "## Split-seed robustness",
            "",
            _markdown_table(decision),
            "",
            "## H1 - learner and weighting",
            "",
            _markdown_table(h1),
            "",
            "## H2 - E3 residual signal",
            "",
            _markdown_table(h2),
            "",
            "Nested stability selection summary:",
            "",
            _markdown_table(stability),
            "",
            "## H3 - antisymmetric latent-difference model",
            "",
            _markdown_table(h3),
            "",
            "Each H3 arm was run with its predeclared training-only label-shuffle "
            "control; a real H3 arm is eligible only when that gate passes.",
            "",
            "## Negative controls",
            "",
            _markdown_table(controls),
            "",
            "Paired extractant bootstrap:",
            "",
            _markdown_table(bootstrap),
            "",
            "## Per-extractant breakdown",
            "",
            _markdown_table(
                per_extractant.sort_values(
                    ["arm", "mean_mae"], ascending=[True, False]
                ),
                maximum_rows=80,
            ),
            "",
            "Extractant-family macro summaries:",
            "",
            _markdown_table(family_metrics, maximum_rows=80),
            "",
            "## Pooled-vs-macro inversion",
            "",
            f"Pooled and macro complete rankings differ: `{inversion}`.",
            "",
            "## Decision",
            "",
            f"Frozen-protocol champion: **{champion}**.",
            "",
            "If no candidate is eligible, retaining A2 is the predeclared valid negative "
            "result; no score was optimized by changing the split protocol.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    protocol = load_gen3_protocol(protocol_path)
    protocol_sha256 = file_sha256(protocol_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "aggregate"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_text_atomic("Generation-3 aggregation has not completed.\n", output_dir / "_INCOMPLETE")
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    run_dirs = sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir() and (path / "_SUCCESS.json").is_file()
    )
    for run_dir in run_dirs:
        try:
            records.append(_validate_run(run_dir, protocol, protocol_sha256))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            errors.append(f"{run_dir.name}: {error}")

    primary_seed_set = set(int(value) for value in protocol["splits"]["split_seeds"])
    primary_model_seed = int(protocol["splits"]["primary_model_seed"])
    primary_records = [
        record
        for record in records
        if record["metadata"]["run_role"] == "primary_split_seed"
    ]
    observed_pairs = [
        (
            int(record["metadata"]["split_seed"]),
            int(record["metadata"]["model_seed"]),
        )
        for record in primary_records
    ]
    expected_pairs = sorted((seed, primary_model_seed) for seed in primary_seed_set)
    if sorted(observed_pairs) != expected_pairs:
        errors.append(
            f"Primary split/model plan mismatch: expected {expected_pairs}, observed "
            f"{sorted(observed_pairs)}"
        )
    if records:
        arm_contracts = {tuple(record["metadata"]["arms"]) for record in records}
        if len(arm_contracts) != 1:
            errors.append("Run arm contracts differ.")
        implementation_contracts = {
            json.dumps(record["implementation_sha256"], sort_keys=True)
            for record in records
        }
        if len(implementation_contracts) != 1:
            errors.append("Run implementation SHA-256 contracts differ.")
        dataset_contracts = {
            json.dumps(record["dataset_hashes"], sort_keys=True)
            for record in records
        }
        feature_contracts = {
            json.dumps(record["feature_hashes"], sort_keys=True)
            for record in records
        }
        if len(dataset_contracts) != 1:
            errors.append("Run dataset hash contracts differ.")
        if len(feature_contracts) != 1:
            errors.append("Run feature hash contracts differ.")
        pair_contracts = {
            tuple(record["predictions"].sort_values("pair_id")["pair_id"])
            for record in primary_records
        }
        if len(pair_contracts) != 1:
            errors.append("Primary runs do not share identical pair rows.")

    validation = {
        "passed": not errors,
        "protocol_sha256": protocol_sha256,
        "discovered_runs": len(run_dirs),
        "hash_validated_runs": len(records),
        "primary_runs": len(primary_records),
        "expected_primary_seed_model_pairs": expected_pairs,
        "observed_primary_seed_model_pairs": sorted(observed_pairs),
        "errors": errors,
    }
    if errors:
        write_json_atomic(validation, output_dir / "validation.json")
        write_text_atomic(
            "# Primary leaderboard\n\nStatus: **FAILED (fail-closed)**\n\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\n",
            output_dir / "report.md",
        )
        return 2

    metric_frames: list[pd.DataFrame] = []
    extractant_frames: list[pd.DataFrame] = []
    family_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    lambda_frames: list[pd.DataFrame] = []
    split_assignment_frames: list[pd.DataFrame] = []
    inner_assignment_frames: list[pd.DataFrame] = []
    for record in primary_records:
        split_seed = int(record["metadata"]["split_seed"])
        model_seed = int(record["metadata"]["model_seed"])
        for frame in (
            record["metrics"],
            record["per_extractant"],
            record["family_metrics"],
            record["predictions"],
            record["stability"],
            record["lambdas"],
            record["split_assignments"],
            record["inner_fold_assignments"],
        ):
            frame["split_seed"] = split_seed
            frame["model_seed"] = model_seed
        metric_frames.append(record["metrics"])
        extractant_frames.append(record["per_extractant"])
        family_frames.append(record["family_metrics"])
        prediction_frames.append(record["predictions"])
        stability_frames.append(record["stability"])
        lambda_frames.append(record["lambdas"])
        split_assignment_frames.append(record["split_assignments"])
        inner_assignment_frames.append(record["inner_fold_assignments"])
    per_seed = pd.concat(metric_frames, ignore_index=True)
    per_extractant_seed = pd.concat(extractant_frames, ignore_index=True)
    family_seed = pd.concat(family_frames, ignore_index=True)
    cross_seed_oof = pd.concat(prediction_frames, ignore_index=True)
    stability_raw = pd.concat(stability_frames, ignore_index=True)
    lambdas = pd.concat(lambda_frames, ignore_index=True)
    split_assignments = pd.concat(split_assignment_frames, ignore_index=True)
    inner_fold_assignments = pd.concat(inner_assignment_frames, ignore_index=True)
    arms = tuple(primary_records[0]["metadata"]["arms"])

    aggregate = (
        per_seed.groupby("arm", sort=False)
        .agg(
            equal_extractant_macro_mae=("equal_extractant_macro_mae", "mean"),
            equal_extractant_macro_mae_split_sd=("equal_extractant_macro_mae", "std"),
            pooled_micro_mae=("pooled_micro_mae", "mean"),
            median_extractant_mae=("median_extractant_mae", "mean"),
            worst_quartile_extractant_mae=("worst_quartile_extractant_mae", "mean"),
            fraction_extractants_improved_vs_A2=(
                "fraction_extractants_improved_vs_A2",
                "mean",
            ),
            pooled_r2=("pooled_r2", "mean"),
            adjacent_ln_mae=("adjacent_ln_mae", "mean"),
            nonadjacent_ln_mae=("nonadjacent_ln_mae", "mean"),
            sign_accuracy=("sign_accuracy", "mean"),
            split_seed_count=("split_seed", "nunique"),
        )
        .reset_index()
        .sort_values("equal_extractant_macro_mae", kind="stable")
    )
    per_extractant = (
        per_extractant_seed.groupby(
            ["extractant", "extractant_family", "arm"], sort=False
        )
        .agg(
            mean_mae=("mae", "mean"),
            split_seed_sd_mae=("mae", "std"),
            mean_delta_mae_baseline_minus_candidate=(
                "delta_mae_baseline_minus_candidate",
                "mean",
            ),
            split_seeds_improved=("improved_vs_baseline", "sum"),
            split_seed_count=("split_seed", "nunique"),
        )
        .reset_index()
    )
    family_metrics = (
        family_seed.groupby(["arm", "extractant_family"], sort=False)
        .agg(
            macro_mae=("macro_mae", "mean"),
            macro_mae_split_sd=("macro_mae", "std"),
            n_extractants=("n_extractants", "max"),
            fraction_extractants_improved_vs_A2=(
                "fraction_extractants_improved_vs_A2",
                "mean",
            ),
            split_seed_count=("split_seed", "nunique"),
        )
        .reset_index()
    )
    comparisons = {
        f"{BASELINE_ARM}_vs_{arm}": (BASELINE_ARM, arm)
        for arm in arms
        if arm != BASELINE_ARM
    }
    comparisons.update(
        {
            "H2_E3_RAW_SHUFFLED_vs_H2_E3_RAW_REAL": (
                "H2_E3_RAW_SHUFFLED",
                "H2_E3_RAW_REAL",
            ),
            "H2_E3_COMPACT_SHUFFLED_vs_H2_E3_COMPACT_REAL": (
                "H2_E3_COMPACT_SHUFFLED",
                "H2_E3_COMPACT_REAL",
            ),
            "H2_E3_STABLE_SHUFFLED_vs_H2_E3_STABLE_REAL": (
                "H2_E3_STABLE_SHUFFLED",
                "H2_E3_STABLE_REAL",
            ),
            "H3_A2_LABEL_SHUFFLED_vs_H3_A2": (
                "H3_A2_LABEL_SHUFFLED",
                "H3_A2",
            ),
            "H3_A2_ELEC_LABEL_SHUFFLED_vs_H3_A2_ELEC": (
                "H3_A2_ELEC_LABEL_SHUFFLED",
                "H3_A2_ELEC",
            ),
        }
    )
    bootstrap = _paired_multisplit_bootstrap(
        per_extractant_seed,
        comparisons,
        replicates=int(protocol["statistics"]["cluster_bootstrap_replicates"]),
        seed=int(protocol["statistics"]["bootstrap_seed"]),
    )
    stability = _aggregate_stability(stability_raw)
    decision = _decision_table(per_seed, lambdas, protocol)
    eligible = decision[decision["eligible_for_champion"]]
    champion = (
        BASELINE_ARM
        if eligible.empty
        else str(
            eligible.sort_values("mean_equal_extractant_macro_mae").iloc[0]["arm"]
        )
    )
    summary = {
        "protocol_sha256": protocol_sha256,
        "primary_split_seeds": sorted(primary_seed_set),
        "primary_model_seed": primary_model_seed,
        "primary_metric": "equal_extractant_macro_mae",
        "champion": champion,
        "valid_negative_result": champion == BASELINE_ARM,
        "pooled_metrics_row_weighted": True,
        "pooled_metrics_used_for_selection": False,
        "validation": validation,
    }
    report = _report(
        aggregate,
        decision,
        bootstrap,
        per_extractant,
        family_metrics,
        stability,
        champion,
    )

    write_text_atomic(
        protocol_path.read_text(encoding="utf-8"),
        output_dir / "gen3_protocol.json",
    )
    write_json_atomic(
        primary_records[0]["dataset_hashes"], output_dir / "dataset_hashes.json"
    )
    write_json_atomic(
        primary_records[0]["feature_hashes"], output_dir / "feature_hashes.json"
    )
    write_json_atomic(validation, output_dir / "validation.json")
    write_csv_atomic(cross_seed_oof, output_dir / "oof_predictions.csv")
    write_csv_atomic(split_assignments, output_dir / "split_assignments.csv")
    write_csv_atomic(
        inner_fold_assignments, output_dir / "inner_fold_assignments.csv"
    )
    write_csv_atomic(per_seed, output_dir / "per_seed_metrics.csv")
    write_csv_atomic(aggregate, output_dir / "aggregate_metrics.csv")
    write_csv_atomic(per_extractant, output_dir / "per_extractant_metrics.csv")
    write_csv_atomic(
        family_metrics, output_dir / "extractant_family_metrics.csv"
    )
    write_csv_atomic(bootstrap, output_dir / "paired_bootstrap.csv")
    write_csv_atomic(stability, output_dir / "e3_stability.csv")
    write_csv_atomic(decision, output_dir / "decision_rules.csv")
    write_json_atomic(summary, output_dir / "aggregate_summary.json")
    write_text_atomic(report, output_dir / "report.md")
    artifact_names = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"_INCOMPLETE", "artifact_hashes.json"}
    )
    write_json_atomic(
        {name: file_sha256(output_dir / name) for name in artifact_names},
        output_dir / "artifact_hashes.json",
    )
    (output_dir / "_INCOMPLETE").unlink()
    write_json_atomic(
        {
            "status": "complete",
            "protocol_sha256": protocol_sha256,
            "champion": champion,
            "validation_sha256": file_sha256(output_dir / "validation.json"),
            "aggregate_summary_sha256": file_sha256(
                output_dir / "aggregate_summary.json"
            ),
            "artifact_hashes_sha256": file_sha256(
                output_dir / "artifact_hashes.json"
            ),
        },
        output_dir / "_SUCCESS.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
