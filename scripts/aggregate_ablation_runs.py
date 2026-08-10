#!/usr/bin/env python3
"""Fail-closed aggregation of leakage-safe, multi-seed ablation runs.

The aggregator deliberately recomputes every published metric from aligned OOF
predictions.  A run is admitted only when its terminal success marker, artifact
manifest, scientific fingerprints, fold assignment and prediction cohort all
validate.  This prevents a partially copied or scientifically incompatible run
from being averaged into the final result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CORE_ABLATIONS: tuple[str, ...] = tuple(f"A{index}" for index in range(7))
PRIMARY_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("A2", "A5"),
    ("A2", "A6"),
    ("A3", "A2"),
)
METRIC_NAMES: tuple[str, ...] = ("mae", "rmse", "r2", "pearson", "spearman")

# These files are the per-run audit contract written by run_ablation_benchmark.
# Training history is intentionally absent because the current ExtraTrees model
# has no iterative training history; all inner-CV candidates are persisted in
# inner_cv_tuning.csv instead.
REQUIRED_RUN_ARTIFACTS: tuple[str, ...] = (
    "_SUCCESS.json",
    "summary.json",
    "run_config.json",
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
    "validation.json",
    "artifact_hashes.json",
)

OUTPUT_ARTIFACTS: tuple[str, ...] = (
    "aggregate_summary.json",
    "cross_seed_oof_predictions.csv",
    "per_run_metrics.csv",
    "per_ablation_metrics.csv",
    "paired_ablation_deltas.csv",
    "per_extractant_comparison.csv",
    "per_lanthanide_comparison.csv",
    "report.md",
    "validation.json",
)

_SAFE_ABLATION = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.-]*\Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-runs", type=int, default=None)
    parser.add_argument(
        "--expected-pair-scope",
        choices=("all", "adjacent"),
        required=True,
        help="Reject runs outside this explicitly requested pair cohort.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Held-out-extractant bootstrap replicates for paired ensemble deltas.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=8675309)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite values to strict JSON values."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def write_text_atomic(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(payload: Any, path: Path) -> None:
    write_text_atomic(canonical_json(payload) + "\n", path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def nested_optional(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def first_value(
    payloads: Sequence[Mapping[str, Any]],
    paths: Sequence[tuple[str, ...]],
    label: str,
) -> Any:
    for payload in payloads:
        for path in paths:
            value = nested_optional(payload, *path)
            if value is not None:
                return value
    raise KeyError(label)


def require_hash(value: Any, label: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return text


def require_seed(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not an integer seed")
    result = int(value)
    if result < 0 or result > 4_000_000_000:
        raise ValueError(f"{label} is outside [0, 4000000000]")
    return result


def artifact_hash_mapping(payload: Mapping[str, Any]) -> dict[str, str]:
    """Accept the runner's named hash object while rejecting ambiguous values."""

    candidate: Any = payload
    for key in ("sha256", "artifact_sha256", "artifacts"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidate = value
            break
    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError("artifact_hashes.json has no non-empty hash mapping")
    result: dict[str, str] = {}
    for raw_name, raw_hash in candidate.items():
        name = str(raw_name)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in {
            "_SUCCESS.json",
            "artifact_hashes.json",
        }:
            raise ValueError(f"unsafe or self-referential artifact name: {name!r}")
        result[relative.as_posix()] = require_hash(raw_hash, f"artifact hash {name}")
    return result


def validate_hashed_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify terminal markers and every material file in one completed run."""

    missing = [name for name in REQUIRED_RUN_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing required artifacts: {missing}")
    if (run_dir / "_INCOMPLETE").exists():
        raise ValueError("stale _INCOMPLETE marker")

    success = read_json(run_dir / "_SUCCESS.json")
    if success.get("status") != "complete":
        raise ValueError("_SUCCESS status is not complete")
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "artifact_hashes.json"
    if require_hash(success.get("summary_sha256"), "_SUCCESS.summary_sha256") != file_sha256(
        summary_path
    ):
        raise ValueError("_SUCCESS summary hash mismatch")
    if require_hash(
        success.get("artifact_hashes_sha256"), "_SUCCESS.artifact_hashes_sha256"
    ) != file_sha256(manifest_path):
        raise ValueError("_SUCCESS artifact-manifest hash mismatch")

    validation_path = run_dir / "validation.json"
    declared_validation_hash = success.get("validation_sha256")
    if declared_validation_hash is not None and require_hash(
        declared_validation_hash, "_SUCCESS.validation_sha256"
    ) != file_sha256(validation_path):
        raise ValueError("_SUCCESS validation hash mismatch")

    declared = artifact_hash_mapping(read_json(manifest_path))
    observed = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name not in {"_SUCCESS.json", "artifact_hashes.json", "_INCOMPLETE"}
    }
    # summary.json is authenticated by _SUCCESS and may also be listed in the
    # manifest. Every other regular file must have exactly one declared hash.
    missing_hashes = sorted((observed - {"summary.json"}) - set(declared))
    unknown_hashes = sorted(set(declared) - observed)
    if missing_hashes:
        raise ValueError(f"unhashed artifacts: {missing_hashes}")
    if unknown_hashes:
        raise ValueError(f"hash manifest names missing files: {unknown_hashes}")
    for name, expected_hash in sorted(declared.items()):
        if file_sha256(run_dir / name) != expected_hash:
            raise ValueError(f"artifact hash mismatch: {name}")

    validation = read_json(validation_path)
    if validation.get("passed") is not True:
        raise ValueError("per-run validation.json did not pass")
    leakage_audit = read_json(run_dir / "leakage_audit.json")
    if leakage_audit.get("passed") is not True:
        raise ValueError("leakage_audit.json did not pass")
    feature_audit = read_json(run_dir / "feature_audit.json")
    feature_passed = feature_audit.get("passed") is True or bool(
        feature_audit.get("all_registry_features_assigned_once", False)
        and not feature_audit.get("target_or_identifier_features_present", True)
    )
    if not feature_passed:
        raise ValueError("feature_audit.json did not pass")
    summary = read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("summary status is not complete")
    return summary, read_json(run_dir / "run_config.json")


def _rename_first(frame: pd.DataFrame, canonical: str, aliases: Iterable[str]) -> None:
    if canonical in frame.columns:
        return
    for alias in aliases:
        if alias in frame.columns:
            frame.rename(columns={alias: canonical}, inplace=True)
            return


def normalize_oof(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Validate long-form OOF rows and return metadata plus wide predictions."""

    frame = pd.read_csv(path)
    _rename_first(frame, "extractant", ("extractant_id", "group_id", "group"))
    _rename_first(frame, "y_true", ("target", "log_SF_A_over_B"))
    _rename_first(frame, "y_pred", ("prediction", "oof_prediction"))
    _rename_first(frame, "ablation", ("model", "feature_set"))
    required = {
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
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF predictions miss columns {missing}")
    if frame.empty:
        raise ValueError("OOF predictions are empty")

    text_columns = (
        "pair_id",
        "extractant",
        "pair_label",
        "metal_A",
        "metal_B",
        "ablation",
    )
    for column in text_columns:
        if frame[column].isna().any():
            raise ValueError(f"OOF {column} contains null values")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"OOF {column} contains empty values")
    invalid_names = sorted(
        name for name in frame["ablation"].unique() if _SAFE_ABLATION.fullmatch(name) is None
    )
    if invalid_names:
        raise ValueError(f"OOF contains unsafe ablation names: {invalid_names}")

    for column in ("outer_fold",):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if (frame[column] < 0).any():
            raise ValueError(f"OOF {column} contains a negative value")
    for column in ("y_true", "y_pred"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"OOF {column} contains non-finite values")
    if frame.duplicated(["pair_id", "ablation"]).any():
        raise ValueError("OOF contains duplicate pair_id/ablation rows")

    ablations = tuple(sorted(frame["ablation"].unique(), key=ablation_sort_key))
    missing_core = sorted(set(CORE_ABLATIONS) - set(ablations))
    if missing_core:
        raise ValueError(f"OOF is missing pre-specified ablations {missing_core}")
    counts = frame.groupby("ablation", sort=False)["pair_id"].nunique()
    expected_pairs = frame["pair_id"].nunique()
    incomplete = counts[counts != expected_pairs]
    if not incomplete.empty:
        raise ValueError(
            "OOF ablations do not cover one identical pair cohort: "
            f"{incomplete.to_dict()} versus {expected_pairs}"
        )

    optional_metadata = tuple(
        column
        for column in (
            "geometry_qc_class",
            "geometry_qc_tier",
            "geometry_qc_pair",
            "geometry_quality",
            "condition_id",
        )
        if column in frame.columns
    )
    metadata_columns = (
        "pair_id",
        "extractant",
        "pair_label",
        "metal_A",
        "metal_B",
        "outer_fold",
        "y_true",
        *optional_metadata,
    )
    for column in metadata_columns[1:]:
        per_pair = frame.groupby("pair_id", sort=False, dropna=False)[column].nunique(
            dropna=False
        )
        if (per_pair != 1).any():
            raise ValueError(f"OOF {column} changes between ablations for one pair")
    metadata = (
        frame.loc[:, list(metadata_columns)]
        .drop_duplicates("pair_id")
        .sort_values("pair_id", kind="stable")
        .reset_index(drop=True)
    )
    wide = frame.pivot(index="pair_id", columns="ablation", values="y_pred")
    wide = wide.reindex(index=metadata["pair_id"], columns=list(ablations))
    if wide.isna().any().any():
        raise ValueError("OOF prediction matrix contains missing cells")
    wide.index.name = "pair_id"
    return metadata, wide.reset_index(drop=True), ablations


def validate_fold_assignments(path: Path, metadata: pd.DataFrame, split_seed: int) -> None:
    frame = pd.read_csv(path)
    _rename_first(frame, "extractant", ("extractant_id", "group_id", "group"))
    required = {"pair_id", "extractant", "outer_fold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fold assignments miss {missing}")
    if frame["pair_id"].duplicated().any():
        raise ValueError("fold assignments contain duplicate pair IDs")
    frame["pair_id"] = frame["pair_id"].astype(str).str.strip()
    frame["extractant"] = frame["extractant"].astype(str).str.strip()
    frame["outer_fold"] = pd.to_numeric(frame["outer_fold"], errors="raise").astype(int)
    if "outer_split_seed" in frame.columns:
        observed = set(
            pd.to_numeric(frame["outer_split_seed"], errors="raise").astype(int).tolist()
        )
        if observed != {split_seed}:
            raise ValueError(
                f"fold assignment split seed mismatch: observed={sorted(observed)}"
            )
    reference = metadata.loc[:, ["pair_id", "extractant", "outer_fold"]]
    frame = frame.loc[:, ["pair_id", "extractant", "outer_fold"]].sort_values(
        "pair_id", kind="stable"
    )
    frame.reset_index(drop=True, inplace=True)
    if not frame.equals(reference.reset_index(drop=True)):
        raise ValueError("fold assignments disagree with OOF row metadata")


def ablation_sort_key(name: str) -> tuple[int, int, str]:
    if name in CORE_ABLATIONS:
        return (0, int(name[1:]), name)
    if name.startswith("A5_SHUFFLED"):
        return (1, 0, name)
    if name.startswith("A2+D"):
        return (2, 0, name)
    # Generation-2 ladder, then its permutation controls, then the secondary
    # 2D-representation ladder, then the reference baselines.
    for position, prefix in enumerate(("G", "E", "C", "S", "B")):
        if name.startswith(prefix):
            return (3 + position, int("_SHUFFLED_s" in name), name)
    return (8, 0, name)


def aligned_exact(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    for column in (
        "pair_id",
        "extractant",
        "pair_label",
        "metal_A",
        "metal_B",
        "outer_fold",
    ):
        if not reference[column].equals(candidate[column]):
            raise ValueError(f"{label}: aligned OOF {column} values differ")
    if not np.array_equal(
        reference["y_true"].to_numpy(dtype=float),
        candidate["y_true"].to_numpy(dtype=float),
    ):
        raise ValueError(f"{label}: aligned OOF truth values differ")


def correlation(truth: np.ndarray, prediction: np.ndarray, *, ranked: bool) -> float:
    left = pd.Series(truth)
    right = pd.Series(prediction)
    if ranked:
        left = left.rank(method="average")
        right = right.rank(method="average")
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    if len(left_values) < 2 or np.std(left_values) == 0 or np.std(right_values) == 0:
        return float("nan")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if len(truth) != len(prediction) or len(truth) == 0:
        raise ValueError("metric arrays must have equal non-zero length")
    residual = truth - prediction
    squared = residual**2
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        "n_samples": int(len(truth)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(squared))),
        "r2": float(1.0 - np.sum(squared) / denominator) if denominator > 0 else float("nan"),
        "pearson": correlation(truth, prediction, ranked=False),
        "spearman": correlation(truth, prediction, ranked=True),
    }


def delta_metrics(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return {
        "delta_mae": float(reference["mae"] - candidate["mae"]),
        "delta_rmse": float(reference["rmse"] - candidate["rmse"]),
        "delta_r2": float(candidate["r2"] - reference["r2"]),
        "delta_pearson": float(candidate["pearson"] - reference["pearson"]),
        "delta_spearman": float(candidate["spearman"] - reference["spearman"]),
    }


def describe(values: Iterable[Any]) -> dict[str, Any]:
    finite_values = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite_values:
        return {"n": 0, "mean": None, "median": None, "sample_std": None, "min": None, "max": None}
    return {
        "n": len(finite_values),
        "mean": statistics.fmean(finite_values),
        "median": statistics.median(finite_values),
        "sample_std": statistics.stdev(finite_values) if len(finite_values) >= 2 else None,
        "min": min(finite_values),
        "max": max(finite_values),
    }


def comparison_contract(ablations: Sequence[str]) -> tuple[tuple[str, str], ...]:
    available = set(ablations)
    comparisons = [pair for pair in PRIMARY_COMPARISONS if set(pair) <= available]
    comparisons.extend(
        ("A2", name)
        for name in ablations
        if name.startswith("A5_SHUFFLED") or name.startswith("A2+D")
    )
    # Second-generation ladder: every declared arm against the 2D baseline, and
    # every permutation control against the arm it controls.  Without the second
    # contrast a block that merely adds capacity looks like a real gain.
    if "A2" in available:
        comparisons.extend(
            ("A2", name)
            for name in ablations
            if name not in CORE_ABLATIONS
            and not name.startswith(("A5_SHUFFLED", "A2+D"))
            and name != "A2"
        )
    # Secondary 2D-representation ladder: the blocks re-added on top of the
    # RDKit-only baseline are scored against that baseline, not against A2.
    if "S1" in available:
        comparisons.extend(
            ("S1", name)
            for name in ablations
            if name in {"S3", "S4", "S5"}
        )
    for name in ablations:
        if "_SHUFFLED_s" not in name or name.startswith("A5_SHUFFLED"):
            continue
        controlled = name.split("_SHUFFLED_s", 1)[0]
        if controlled in available:
            comparisons.append((name, controlled))
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for pair in comparisons:
        if pair not in seen and set(pair) <= available and pair[0] != pair[1]:
            seen.add(pair)
            ordered.append(pair)
    return tuple(ordered)


def compute_run_tables(
    records: Sequence[dict[str, Any]],
    comparisons: Sequence[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for record in records:
        metadata = record["metadata"]
        wide = record["predictions"]
        truth = metadata["y_true"].to_numpy(dtype=float)
        run_row: dict[str, Any] = {
            "run_dir": str(record["run_dir"]),
            "model_seed": record["model_seed"],
            "split_seed": record["split_seed"],
            "pair_scope": record["pair_scope"],
            "n_pairs": len(metadata),
            "n_extractants": int(metadata["extractant"].nunique()),
            "n_outer_folds": int(metadata["outer_fold"].nunique()),
        }
        all_metrics: dict[str, dict[str, Any]] = {}
        for ablation in record["ablations"]:
            prediction = wide[ablation].to_numpy(dtype=float)
            metrics = regression_metrics(truth, prediction)
            all_metrics[ablation] = metrics
            fold_metrics = []
            for fold, indices in metadata.groupby("outer_fold", sort=True).groups.items():
                selection = np.asarray(list(indices), dtype=int)
                item = regression_metrics(truth[selection], prediction[selection])
                item["outer_fold"] = int(fold)
                fold_metrics.append(item)
            row: dict[str, Any] = {
                "run_dir": str(record["run_dir"]),
                "model_seed": record["model_seed"],
                "split_seed": record["split_seed"],
                "ablation": ablation,
                **metrics,
            }
            for metric_name in METRIC_NAMES:
                summary = describe(item[metric_name] for item in fold_metrics)
                row[f"fold_{metric_name}_mean"] = summary["mean"]
                row[f"fold_{metric_name}_sample_std"] = summary["sample_std"]
            ablation_rows.append(row)
            for metric_name in METRIC_NAMES:
                run_row[f"{ablation}_{metric_name}"] = metrics[metric_name]

        folds = sorted(metadata["outer_fold"].unique().tolist())
        for reference, candidate in comparisons:
            comparison = f"{reference}_vs_{candidate}"
            delta = delta_metrics(all_metrics[reference], all_metrics[candidate])
            for key, value in delta.items():
                run_row[f"{comparison}_{key}"] = value
            delta_rows.append(
                {
                    "run_dir": str(record["run_dir"]),
                    "model_seed": record["model_seed"],
                    "split_seed": record["split_seed"],
                    "scope": "complete_oof",
                    "outer_fold": None,
                    "comparison": comparison,
                    "reference": reference,
                    "candidate": candidate,
                    "n_samples": len(metadata),
                    **delta,
                    "candidate_wins_mae": delta["delta_mae"] > 0,
                }
            )
            for fold in folds:
                mask = metadata["outer_fold"].to_numpy(dtype=int) == int(fold)
                ref_metric = regression_metrics(
                    truth[mask], wide.loc[mask, reference].to_numpy(dtype=float)
                )
                cand_metric = regression_metrics(
                    truth[mask], wide.loc[mask, candidate].to_numpy(dtype=float)
                )
                fold_delta = delta_metrics(ref_metric, cand_metric)
                delta_rows.append(
                    {
                        "run_dir": str(record["run_dir"]),
                        "model_seed": record["model_seed"],
                        "split_seed": record["split_seed"],
                        "scope": "outer_fold",
                        "outer_fold": int(fold),
                        "comparison": comparison,
                        "reference": reference,
                        "candidate": candidate,
                        "n_samples": int(mask.sum()),
                        **fold_delta,
                        "candidate_wins_mae": fold_delta["delta_mae"] > 0,
                    }
                )
        run_rows.append(run_row)
    return pd.DataFrame(run_rows), pd.DataFrame(ablation_rows), pd.DataFrame(delta_rows)


def build_cross_seed_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    metadata = records[0]["metadata"].copy()
    result = metadata.copy()
    for ablation in records[0]["ablations"]:
        arrays = []
        for record in records:
            values = record["predictions"][ablation].to_numpy(dtype=float)
            arrays.append(values)
            result[f"prediction_{ablation}_model_seed_{record['model_seed']}"] = values
        result[f"prediction_{ablation}_mean"] = np.mean(arrays, axis=0)
    return result


def ensemble_prediction(frame: pd.DataFrame, ablation: str) -> np.ndarray:
    return frame[f"prediction_{ablation}_mean"].to_numpy(dtype=float)


def paired_extractant_bootstrap(
    frame: pd.DataFrame,
    reference: str,
    candidate: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    truth = frame["y_true"].to_numpy(dtype=float)
    reference_prediction = ensemble_prediction(frame, reference)
    candidate_prediction = ensemble_prediction(frame, candidate)
    groups = frame["extractant"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {f"delta_{name}": [] for name in METRIC_NAMES}
    draws["delta_macro_extractant_mae"] = []
    for _ in range(replicates):
        selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected_indices = np.concatenate([indices[group] for group in selected_groups])
        ref_metrics = regression_metrics(
            truth[selected_indices], reference_prediction[selected_indices]
        )
        cand_metrics = regression_metrics(
            truth[selected_indices], candidate_prediction[selected_indices]
        )
        delta = delta_metrics(ref_metrics, cand_metrics)
        for name, value in delta.items():
            if math.isfinite(value):
                draws[name].append(value)
        group_deltas = [
            float(
                np.mean(np.abs(truth[indices[group]] - reference_prediction[indices[group]]))
                - np.mean(
                    np.abs(truth[indices[group]] - candidate_prediction[indices[group]])
                )
            )
            for group in selected_groups
        ]
        draws["delta_macro_extractant_mae"].append(float(np.mean(group_deltas)))
    point = delta_metrics(
        regression_metrics(truth, reference_prediction),
        regression_metrics(truth, candidate_prediction),
    )
    point["delta_macro_extractant_mae"] = float(
        np.mean(
            [
                np.mean(
                    np.abs(truth[indices[group]] - reference_prediction[indices[group]])
                )
                - np.mean(
                    np.abs(truth[indices[group]] - candidate_prediction[indices[group]])
                )
                for group in unique_groups
            ]
        )
    )
    intervals: dict[str, Any] = {}
    for name, values in draws.items():
        array = np.asarray(values, dtype=float)
        intervals[name] = {
            "point": point[name],
            "bootstrap_mean": float(np.mean(array)) if len(array) else None,
            "ci95_low": float(np.quantile(array, 0.025)) if len(array) else None,
            "ci95_high": float(np.quantile(array, 0.975)) if len(array) else None,
            "valid_replicates": int(len(array)),
        }
    return {
        "method": (
            "paired cluster bootstrap over held-out extractants and fixed cross-seed "
            "OOF predictions; delta_macro_extractant_mae gives every sampled "
            "extractant equal weight"
        ),
        "replicates": replicates,
        "seed": seed,
        "n_extractants": int(len(unique_groups)),
        "scope_limitation": (
            "Conditional on the frozen folds and completed model seeds; no models are refit "
            "inside bootstrap replicates."
        ),
        "deltas": intervals,
    }


def group_comparison_table(
    frame: pd.DataFrame,
    group_column: str,
    output_column: str,
    ablations: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, group_frame in frame.groupby(group_column, sort=True, dropna=False):
        truth = group_frame["y_true"].to_numpy(dtype=float)
        row: dict[str, Any] = {output_column: group, "n_samples": int(len(group_frame))}
        metrics: dict[str, dict[str, Any]] = {}
        for ablation in ablations:
            item = regression_metrics(truth, ensemble_prediction(group_frame, ablation))
            metrics[ablation] = item
            for metric_name in METRIC_NAMES:
                row[f"{ablation}_{metric_name}"] = item[metric_name]
        for reference, candidate in comparisons:
            comparison = f"{reference}_vs_{candidate}"
            for name, value in delta_metrics(metrics[reference], metrics[candidate]).items():
                row[f"{comparison}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def lanthanide_comparison_table(
    frame: pd.DataFrame,
    ablations: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
) -> pd.DataFrame:
    metals = sorted(set(frame["metal_A"].astype(str)) | set(frame["metal_B"].astype(str)))
    rows: list[dict[str, Any]] = []
    for metal in metals:
        side_a = frame.loc[frame["metal_A"].astype(str).eq(metal)]
        side_b = frame.loc[frame["metal_B"].astype(str).eq(metal)]
        truth = np.concatenate(
            [
                side_a["y_true"].to_numpy(dtype=float),
                -side_b["y_true"].to_numpy(dtype=float),
            ]
        )
        row: dict[str, Any] = {"lanthanide": metal, "n_samples": int(len(truth))}
        metrics: dict[str, dict[str, Any]] = {}
        for ablation in ablations:
            prediction = np.concatenate(
                [
                    ensemble_prediction(side_a, ablation),
                    -ensemble_prediction(side_b, ablation),
                ]
            )
            item = regression_metrics(truth, prediction)
            metrics[ablation] = item
            for metric_name in METRIC_NAMES:
                row[f"{ablation}_{metric_name}"] = item[metric_name]
        for reference, candidate in comparisons:
            comparison = f"{reference}_vs_{candidate}"
            for name, value in delta_metrics(metrics[reference], metrics[candidate]).items():
                row[f"{comparison}_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_payload(
    validation: dict[str, Any],
    records: Sequence[dict[str, Any]],
    per_ablation: pd.DataFrame,
    paired_deltas: pd.DataFrame,
    cross_seed: pd.DataFrame,
    per_extractant: pd.DataFrame,
    per_lanthanide: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    ablation_summary: dict[str, Any] = {}
    truth = cross_seed["y_true"].to_numpy(dtype=float)
    for ablation in records[0]["ablations"]:
        rows = per_ablation.loc[per_ablation["ablation"] == ablation]
        ablation_summary[ablation] = {
            "across_model_seeds": {
                metric: describe(rows[metric].tolist()) for metric in METRIC_NAMES
            },
            "cross_seed_oof_ensemble": regression_metrics(
                truth, ensemble_prediction(cross_seed, ablation)
            ),
        }

    paired_summary: dict[str, Any] = {}
    for comparison_index, (reference, candidate) in enumerate(comparisons):
        name = f"{reference}_vs_{candidate}"
        oof_rows = paired_deltas.loc[
            (paired_deltas["comparison"] == name)
            & (paired_deltas["scope"] == "complete_oof")
        ]
        fold_rows = paired_deltas.loc[
            (paired_deltas["comparison"] == name)
            & (paired_deltas["scope"] == "outer_fold")
        ]
        extractant_delta_column = f"{name}_delta_mae"
        extractant_deltas = pd.to_numeric(
            per_extractant[extractant_delta_column], errors="coerce"
        )
        ensemble_delta = delta_metrics(
            regression_metrics(truth, ensemble_prediction(cross_seed, reference)),
            regression_metrics(truth, ensemble_prediction(cross_seed, candidate)),
        )
        bootstrap = paired_extractant_bootstrap(
            cross_seed,
            reference,
            candidate,
            # Confidence intervals are pre-specified for the three scientific
            # contrasts. D1--D5 and individual shuffle controls are exploratory
            # point estimates and must not multiply the inferential search.
            replicates=(
                bootstrap_replicates
                if (reference, candidate) in PRIMARY_COMPARISONS
                else 0
            ),
            seed=bootstrap_seed + comparison_index * 1009,
        )
        paired_summary[name] = {
            "reference": reference,
            "candidate": candidate,
            "positive_delta_definition": (
                "Positive delta_mae/delta_rmse means lower candidate error; positive "
                "delta_r2/delta_pearson/delta_spearman means higher candidate agreement."
            ),
            "across_model_seed_complete_oof_deltas": {
                name_: describe(oof_rows[name_].tolist())
                for name_ in (
                    "delta_mae",
                    "delta_rmse",
                    "delta_r2",
                    "delta_pearson",
                    "delta_spearman",
                )
            },
            "descriptive_seed_fold_deltas": {
                name_: describe(fold_rows[name_].tolist())
                for name_ in (
                    "delta_mae",
                    "delta_rmse",
                    "delta_r2",
                    "delta_pearson",
                    "delta_spearman",
                )
            },
            "cross_seed_oof_ensemble_delta": ensemble_delta,
            "held_out_extractants": {
                "n": int(extractant_deltas.notna().sum()),
                "improved_count": int((extractant_deltas > 0).sum()),
                "improved_fraction": float((extractant_deltas > 0).mean()),
                "delta_mae": describe(extractant_deltas.tolist()),
            },
            "paired_extractant_bootstrap": bootstrap,
        }

    block_rows = []
    for name, item in paired_summary.items():
        if item["candidate"].startswith("A2+D"):
            block_rows.append(
                {
                    "comparison": name,
                    "candidate": item["candidate"],
                    "delta_mae": item["cross_seed_oof_ensemble_delta"]["delta_mae"],
                }
            )
    block_rows.sort(key=lambda row: row["delta_mae"], reverse=True)

    qc_summary: dict[str, Any] | None = None
    qc_column = next(
        (
            column
            for column in (
                "geometry_qc_class",
                "geometry_qc_tier",
                "geometry_qc_pair",
                "geometry_quality",
            )
            if column in cross_seed.columns
        ),
        None,
    )
    if qc_column is not None:
        qc_table = group_comparison_table(
            cross_seed,
            qc_column,
            qc_column,
            records[0]["ablations"],
            comparisons,
        )
        qc_summary = {
            "column": qc_column,
            "rows": json_safe(qc_table.to_dict(orient="records")),
        }

    return {
        "schema_version": "1.0",
        "validation": validation,
        "run_count": len(records),
        "model_seeds": sorted(int(record["model_seed"]) for record in records),
        "split_seed": int(records[0]["split_seed"]),
        "pair_scope": records[0]["pair_scope"],
        "ablation_order": list(records[0]["ablations"]),
        "per_ablation": ablation_summary,
        "paired_comparisons": paired_summary,
        "descriptor_block_ranking_by_cross_seed_delta_mae": block_rows,
        "geometry_qc_comparison": qc_summary,
        "uncertainty_guardrail": (
            "Fold rows sharing a split are correlated and are descriptive. Confidence "
            "intervals use paired resampling of held-out extractants from aligned fixed OOF "
            "predictions; the primary macro-MAE interval gives each sampled extractant one "
            "vote and does not include model-refitting or split uncertainty."
        ),
        "lanthanide_attribution_guardrail": (
            "Each all-pairs observation contributes once to metal_A and once to metal_B in "
            "per-lanthanide summaries; those summaries are therefore overlapping views."
        ),
    }


def fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "NA"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(result):
        return "NA"
    return f"{100.0 * result:.1f}%" if percent else f"{result:+.6f}"


def report_markdown(
    payload: Mapping[str, Any],
    per_extractant: pd.DataFrame,
    per_lanthanide: pd.DataFrame,
) -> str:
    primary = payload["paired_comparisons"]["A2_vs_A5"]
    primary_delta = primary["cross_seed_oof_ensemble_delta"]
    interval = primary["paired_extractant_bootstrap"]["deltas"][
        "delta_macro_extractant_mae"
    ]
    a2_mae = payload["per_ablation"]["A2"]["cross_seed_oof_ensemble"]["mae"]
    relative = primary_delta["delta_mae"] / a2_mae if a2_mae > 0 else None
    if interval["ci95_low"] is not None and interval["ci95_low"] > 0:
        conclusion = "Yes under the conditional equal-extractant bootstrap."
    elif interval["ci95_high"] is not None and interval["ci95_high"] < 0:
        conclusion = (
            "No; A5 is worse than A2 under the conditional equal-extractant "
            "bootstrap."
        )
    else:
        conclusion = (
            "Inconclusive; the paired equal-extractant interval includes zero."
        )

    top_extractants = per_extractant.sort_values(
        "A2_vs_A5_delta_mae", ascending=False
    ).head(5)
    top_metals = per_lanthanide.sort_values("A2_vs_A5_delta_mae", ascending=False).head(5)
    block_ranking = payload["descriptor_block_ranking_by_cross_seed_delta_mae"]
    shuffle_items = [
        item
        for name, item in payload["paired_comparisons"].items()
        if item["candidate"].startswith("A5_SHUFFLED")
    ]

    lines = [
        "# Leakage-safe lanthanide ablation aggregate",
        "",
        "Status: **PASSED**",
        "",
        f"Pair scope: `{payload['pair_scope']}`; model seeds: "
        f"{', '.join(str(seed) for seed in payload['model_seeds'])}; one frozen split seed: "
        f"`{payload['split_seed']}`.",
        "",
        "All run artifacts were hash-verified, feature registries and fold assignments were "
        "identical, and OOF rows aligned exactly by pair ID, truth, extractant and fold.",
        "",
        "## Prespecified A2 versus A5 result",
        "",
        f"1. **Does local 3D improve unseen-extractant prediction?** {conclusion}",
        f"2. **Absolute MAE improvement:** {fmt(primary_delta['delta_mae'])} log units.",
        f"3. **Relative MAE improvement:** {fmt(relative, percent=True)} versus A2.",
        "4. **Held-out extractants improved:** "
        f"{primary['held_out_extractants']['improved_count']}/"
        f"{primary['held_out_extractants']['n']} "
        f"({fmt(primary['held_out_extractants']['improved_fraction'], percent=True)}).",
        "5. **Equal-extractant macro delta MAE and bootstrap 95% CI:** "
        f"{fmt(interval['point'])} "
        f"[{fmt(interval['ci95_low'])}, {fmt(interval['ci95_high'])}].",
        "",
        "Positive error deltas mean the candidate reduced error; positive correlation/R2 "
        "deltas mean the candidate increased agreement.",
        "",
        "| comparison | delta MAE | delta RMSE | delta R2 | delta Pearson | delta Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in payload["paired_comparisons"].items():
        if name not in {"A2_vs_A5", "A2_vs_A6", "A3_vs_A2"}:
            continue
        delta = item["cross_seed_oof_ensemble_delta"]
        lines.append(
            f"| {name} | {fmt(delta['delta_mae'])} | {fmt(delta['delta_rmse'])} | "
            f"{fmt(delta['delta_r2'])} | {fmt(delta['delta_pearson'])} | "
            f"{fmt(delta['delta_spearman'])} |"
        )

    lines.extend(["", "## Descriptor and negative-control checks", ""])
    if block_ranking:
        lines.append(
            "Descriptor-block ranking by cross-seed OOF delta MAE: "
            + ", ".join(
                f"{row['candidate']} ({fmt(row['delta_mae'])})" for row in block_ranking
            )
            + "."
        )
    else:
        lines.append("No exploratory A2+D1--D5 runs were present.")
    lines.append("")
    if shuffle_items:
        shuffled_deltas = [
            item["cross_seed_oof_ensemble_delta"]["delta_mae"] for item in shuffle_items
        ]
        shuffled_mean = statistics.fmean(shuffled_deltas)
        lines.append(
            "A5 shuffled-geometry controls: mean delta MAE "
            f"{fmt(shuffled_mean)} across {len(shuffled_deltas)} shuffle variants, versus "
            f"{fmt(primary_delta['delta_mae'])} for correctly matched A5 geometry. "
            "No categorical 'approximately equal' claim is made without a predeclared "
            "equivalence margin."
        )
    else:
        lines.append("No A5_SHUFFLED negative-control prediction was present.")
    lines.append("")
    qc_comparison = payload.get("geometry_qc_comparison")
    if qc_comparison is None:
        lines.append(
            "Geometry-quality-stratified improvement was unavailable in OOF metadata; the "
            "per-run geometry QC artifacts remain hash-verified."
        )
    else:
        qc_rows = {
            str(row[qc_comparison["column"]]): row for row in qc_comparison["rows"]
        }
        high = qc_rows.get("high_confidence")
        low = qc_rows.get("questionable_or_recovered")
        if high is not None and low is not None:
            high_delta = high["A2_vs_A5_delta_mae"]
            low_delta = low["A2_vs_A5_delta_mae"]
            relation = "stronger" if high_delta > low_delta else "not stronger"
            lines.append(
                "Geometry-quality check: A2-vs-A5 delta MAE is "
                f"{fmt(high_delta)} for high-confidence pairs and {fmt(low_delta)} for "
                f"questionable/recovered pairs; high-confidence improvement is {relation}."
            )
        else:
            lines.append(
                "A high-versus-questionable geometry comparison is not estimable from the "
                "available QC classes; class-specific rows remain in aggregate_summary.json."
            )

    lines.extend(
        [
            "",
            "## Where A5 helps most",
            "",
            "Top extractants by A2-minus-A5 MAE:",
            "",
        ]
    )
    lines.extend(
        f"- `{row['extractant_id']}`: n={int(row['n_samples'])}, "
        f"delta MAE={fmt(row['A2_vs_A5_delta_mae'])}"
        for _, row in top_extractants.iterrows()
    )
    lines.extend(["", "Top lanthanides by overlapping all-pairs attribution:", ""])
    lines.extend(
        f"- `{row['lanthanide']}`: n={int(row['n_samples'])}, "
        f"delta MAE={fmt(row['A2_vs_A5_delta_mae'])}"
        for _, row in top_metals.iterrows()
    )

    fold_rows = payload["paired_comparisons"]["A2_vs_A5"][
        "descriptive_seed_fold_deltas"
    ]["delta_mae"]
    seed_rows = payload["paired_comparisons"]["A2_vs_A5"][
        "across_model_seed_complete_oof_deltas"
    ]["delta_mae"]
    lines.extend(
        [
            "",
            "## Consistency and interpretation",
            "",
            f"Across model seeds, delta MAE mean={fmt(seed_rows['mean'])}, "
            f"sample SD={fmt(seed_rows['sample_std'])}, range=[{fmt(seed_rows['min'])}, "
            f"{fmt(seed_rows['max'])}]. Across correlated seed-fold cells, mean="
            f"{fmt(fold_rows['mean'])} and sample SD={fmt(fold_rows['sample_std'])}.",
            "",
            "The conclusion must be read across MAE, RMSE, R2, Pearson and Spearman in the "
            "comparison table, not from R2 alone. Fold summaries are descriptive because the "
            "same held-out folds recur across model seeds.",
            "",
            payload["uncertainty_guardrail"],
            "",
            f"Validation fingerprint: `{payload['validation']['combined_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def failure_report(errors: Sequence[str], expected_pair_scope: str) -> str:
    return "\n".join(
        [
            "# Leakage-safe ablation aggregation",
            "",
            "Status: **FAILED (fail-closed)**",
            "",
            f"Expected pair scope: `{expected_pair_scope}`.",
            "",
            "No aggregate scientific estimate was emitted because validation failed:",
            "",
            *(f"- {error}" for error in errors),
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    aggregate_implementation_sha256 = file_sha256(Path(__file__).resolve())
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise SystemExit(f"Run root not found: {run_root}")
    if args.expected_runs is not None and args.expected_runs < 2:
        raise SystemExit("--expected-runs must be at least 2")
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    if args.bootstrap_seed < 0 or args.bootstrap_seed > 4_000_000_000:
        raise SystemExit("--bootstrap-seed must be in [0, 4000000000]")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "aggregate"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_text_atomic("Aggregation has not completed.\n", output_dir / "_INCOMPLETE")

    errors: list[str] = []
    run_dirs = sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir()
        and path.name.startswith("run_")
        and path != output_dir
        and (path / "summary.json").is_file()
    )
    if args.expected_runs is not None and len(run_dirs) != args.expected_runs:
        errors.append(f"Expected {args.expected_runs} run directories, found {len(run_dirs)}.")
    if len(run_dirs) < 2:
        errors.append("At least two completed model-seed runs are required.")

    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            summary, run_config = validate_hashed_run(run_dir)
            pair_audit = read_json(run_dir / "pair_build_audit.json")
            config_arguments = run_config.get("arguments")
            if not isinstance(config_arguments, Mapping):
                raise ValueError("run_config.arguments must be a mapping")
            config_protocol = run_config.get("scientific_protocol")
            summary_protocol = summary.get("scientific_protocol")
            if not isinstance(config_protocol, Mapping) or not isinstance(
                summary_protocol, Mapping
            ):
                raise ValueError("scientific_protocol is missing from run config/summary")
            if canonical_json(config_protocol) != canonical_json(summary_protocol):
                raise ValueError("run config and summary scientific_protocol differ")
            scientific_protocol_hash = object_sha256(summary_protocol)
            for container_name, container in (
                ("run_config", run_config),
                ("summary", summary),
            ):
                declared_protocol_hash = require_hash(
                    container.get("scientific_protocol_sha256"),
                    f"{container_name}.scientific_protocol_sha256",
                )
                if declared_protocol_hash != scientific_protocol_hash:
                    raise ValueError(
                        f"{container_name} scientific_protocol_sha256 mismatch"
                    )

            summary_dataset_hash = require_hash(
                summary.get("dataset_sha256"), "summary.dataset_sha256"
            )
            config_dataset_hash = require_hash(
                run_config.get("dataset_sha256"), "run_config.dataset_sha256"
            )
            if summary_dataset_hash != config_dataset_hash:
                raise ValueError("run config and summary dataset_sha256 differ")
            dataset_hash = summary_dataset_hash

            summary_cohort_hash = require_hash(
                summary.get("cohort_sha256"), "summary.cohort_sha256"
            )
            audit_cohort_hash = require_hash(
                pair_audit.get("cohort_sha256"), "pair_build_audit.cohort_sha256"
            )
            if summary_cohort_hash != audit_cohort_hash:
                raise ValueError("summary and pair-build cohort_sha256 differ")
            cohort_hash = summary_cohort_hash

            pair_scope_values = {
                "summary": summary.get("pair_scope"),
                "run_config": config_arguments.get("pair_scope"),
                "scientific_protocol": summary_protocol.get("pair_scope"),
                "pair_build_audit": pair_audit.get("pair_scope"),
            }
            if any(value is None for value in pair_scope_values.values()):
                raise ValueError(f"pair_scope is missing: {pair_scope_values}")
            pair_scope = str(pair_scope_values["summary"])
            if any(str(value) != pair_scope for value in pair_scope_values.values()):
                raise ValueError(f"run-internal pair_scope mismatch: {pair_scope_values}")
            if pair_scope != args.expected_pair_scope:
                raise ValueError(
                    f"pair scope mismatch: expected {args.expected_pair_scope!r}, "
                    f"observed {pair_scope!r}"
                )
            group_mode_values = {
                "summary": summary.get("group_mode"),
                "run_config": config_arguments.get("group_mode"),
                "scientific_protocol": summary_protocol.get("group_mode"),
            }
            if any(value is None for value in group_mode_values.values()):
                raise ValueError(f"group_mode is missing: {group_mode_values}")
            group_mode = str(group_mode_values["summary"])
            if any(str(value) != group_mode for value in group_mode_values.values()):
                raise ValueError(f"run-internal group_mode mismatch: {group_mode_values}")
            if group_mode != "extractant":
                raise ValueError(
                    "primary aggregation requires literal leave-extractants-out "
                    f"group_mode='extractant'; observed {group_mode!r}"
                )
            summary_model_seed = require_seed(
                summary.get("model_seed"), "summary.model_seed"
            )
            config_model_seed = require_seed(
                config_arguments.get("model_seed"), "run_config.arguments.model_seed"
            )
            if summary_model_seed != config_model_seed:
                raise ValueError("run config and summary model_seed differ")
            model_seed = summary_model_seed

            split_seed_values = {
                "summary": require_seed(summary.get("split_seed"), "summary.split_seed"),
                "run_config": require_seed(
                    config_arguments.get("split_seed"),
                    "run_config.arguments.split_seed",
                ),
                "scientific_protocol": require_seed(
                    summary_protocol.get("split_seed"),
                    "scientific_protocol.split_seed",
                ),
            }
            split_seed = split_seed_values["summary"]
            if any(value != split_seed for value in split_seed_values.values()):
                raise ValueError(f"run-internal split_seed mismatch: {split_seed_values}")
            feature_registry_hash = file_sha256(run_dir / "feature_registry.json")
            fold_assignments_hash = file_sha256(run_dir / "fold_assignments.csv")
            fold_memberships_hash = file_sha256(run_dir / "fold_memberships.csv")
            inner_fold_assignments_hash = file_sha256(
                run_dir / "inner_fold_assignments.csv"
            )
            geometry_descriptor_audit_hash = file_sha256(
                run_dir / "geometry_descriptor_audit.json"
            )
            summary_vr_hash = summary.get("vr_asset_sha256")
            config_vr_hash = run_config.get("vr_asset_sha256")
            if summary_vr_hash != config_vr_hash:
                raise ValueError("run config and summary vr_asset_sha256 differ")
            vr_asset_hash = (
                None
                if summary_vr_hash is None
                else require_hash(summary_vr_hash, "vr_asset_sha256")
            )
            summary_software = summary.get("software")
            config_software = run_config.get("software")
            if not isinstance(summary_software, Mapping) or not isinstance(
                config_software, Mapping
            ):
                raise ValueError("software metadata must be a mapping in config/summary")
            if canonical_json(summary_software) != canonical_json(config_software):
                raise ValueError("run config and summary software metadata differ")
            software_hash = object_sha256(summary_software)
            implementation = summary.get("implementation_sha256")
            if not isinstance(implementation, Mapping) or not implementation:
                raise ValueError("implementation_sha256 must be a non-empty mapping")
            for implementation_name, digest in implementation.items():
                require_hash(digest, f"implementation_sha256[{implementation_name!r}]")
            implementation_hash = object_sha256(implementation)
            for declared_name, observed_hash in (
                ("feature_registry_sha256", feature_registry_hash),
                ("fold_assignments_sha256", fold_assignments_hash),
                ("fold_memberships_sha256", fold_memberships_hash),
                ("inner_fold_assignments_sha256", inner_fold_assignments_hash),
                (
                    "geometry_descriptor_audit_sha256",
                    geometry_descriptor_audit_hash,
                ),
            ):
                declared_value = require_hash(summary.get(declared_name), declared_name)
                if declared_value != observed_hash:
                    raise ValueError(f"declared {declared_name} mismatch")

            metadata, predictions, ablations = normalize_oof(
                run_dir / "oof_predictions.csv"
            )
            validate_fold_assignments(
                run_dir / "fold_assignments.csv", metadata, split_seed
            )
            records.append(
                {
                    "run_dir": run_dir,
                    "dataset_sha256": dataset_hash,
                    "cohort_sha256": cohort_hash,
                    "feature_registry_sha256": feature_registry_hash,
                    "fold_assignments_sha256": fold_assignments_hash,
                    "fold_memberships_sha256": fold_memberships_hash,
                    "inner_fold_assignments_sha256": inner_fold_assignments_hash,
                    "geometry_descriptor_audit_sha256": geometry_descriptor_audit_hash,
                    "scientific_protocol_sha256": scientific_protocol_hash,
                    "vr_asset_sha256": vr_asset_hash,
                    "software_sha256": software_hash,
                    "pair_scope": pair_scope,
                    "group_mode": group_mode,
                    "model_seed": model_seed,
                    "split_seed": split_seed,
                    "implementation_sha256": implementation_hash,
                    "metadata": metadata,
                    "predictions": predictions,
                    "ablations": ablations,
                }
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{run_dir.name}: {exc}")

    fingerprints = {
        "dataset_sha256": sorted({record["dataset_sha256"] for record in records}),
        "cohort_sha256": sorted({record["cohort_sha256"] for record in records}),
        "feature_registry_sha256": sorted(
            {record["feature_registry_sha256"] for record in records}
        ),
        "fold_assignments_sha256": sorted(
            {record["fold_assignments_sha256"] for record in records}
        ),
        "fold_memberships_sha256": sorted(
            {record["fold_memberships_sha256"] for record in records}
        ),
        "inner_fold_assignments_sha256": sorted(
            {record["inner_fold_assignments_sha256"] for record in records}
        ),
        "geometry_descriptor_audit_sha256": sorted(
            {record["geometry_descriptor_audit_sha256"] for record in records}
        ),
        "scientific_protocol_sha256": sorted(
            {record["scientific_protocol_sha256"] for record in records}
        ),
        "vr_asset_sha256": sorted(
            {
                record["vr_asset_sha256"] or "none"
                for record in records
            }
        ),
        "software_sha256": sorted(
            {record["software_sha256"] for record in records}
        ),
        "pair_scope": sorted({record["pair_scope"] for record in records}),
        "group_mode": sorted({record["group_mode"] for record in records}),
        "split_seed": sorted({record["split_seed"] for record in records}),
        "implementation_sha256": sorted(
            {record["implementation_sha256"] for record in records}
        ),
    }
    for label, values in fingerprints.items():
        if len(values) > 1:
            errors.append(f"Runs do not share one {label}: {values}")
    model_seeds = [int(record["model_seed"]) for record in records]
    if len(model_seeds) != len(set(model_seeds)):
        errors.append("Model seeds must be unique across runs.")
    if args.expected_runs is not None and len(records) != args.expected_runs:
        errors.append(
            f"Expected {args.expected_runs} hash-validated runs, found {len(records)}."
        )
    if records:
        reference_metadata = records[0]["metadata"]
        reference_ablations = records[0]["ablations"]
        for record in records[1:]:
            try:
                aligned_exact(reference_metadata, record["metadata"], record["run_dir"].name)
                if record["ablations"] != reference_ablations:
                    raise ValueError(
                        f"{record['run_dir'].name}: OOF ablation contracts differ"
                    )
            except ValueError as exc:
                errors.append(str(exc))

    validation: dict[str, Any] = {
        "passed": not errors,
        "aggregate_implementation_sha256": aggregate_implementation_sha256,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "expected_pair_scope": args.expected_pair_scope,
        "expected_runs": args.expected_runs,
        "discovered_run_dirs": len(run_dirs),
        "hash_validated_runs": len(records),
        "errors": errors,
        "fingerprints": fingerprints,
        "model_seeds": sorted(model_seeds),
    }
    validation["combined_sha256"] = object_sha256(
        {
            "expected_pair_scope": args.expected_pair_scope,
            "aggregate_implementation_sha256": aggregate_implementation_sha256,
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_seed": int(args.bootstrap_seed),
            "fingerprints": fingerprints,
            "model_seeds": sorted(model_seeds),
        }
    )
    if errors:
        write_json_atomic(validation, output_dir / "validation.json")
        write_text_atomic(
            failure_report(errors, args.expected_pair_scope), output_dir / "report.md"
        )
        return 2

    comparisons = comparison_contract(records[0]["ablations"])
    per_run, per_ablation, paired_deltas = compute_run_tables(records, comparisons)
    per_run.sort_values("model_seed", inplace=True, ignore_index=True)
    per_ablation.sort_values(
        ["ablation", "model_seed"],
        key=lambda series: (
            series.map(ablation_sort_key) if series.name == "ablation" else series
        ),
        inplace=True,
        ignore_index=True,
    )
    paired_deltas.sort_values(
        ["comparison", "scope", "outer_fold", "model_seed"],
        inplace=True,
        ignore_index=True,
        na_position="first",
    )
    cross_seed = build_cross_seed_frame(records)
    per_extractant = group_comparison_table(
        cross_seed,
        "extractant",
        "extractant_id",
        records[0]["ablations"],
        comparisons,
    )
    per_lanthanide = lanthanide_comparison_table(
        cross_seed, records[0]["ablations"], comparisons
    )
    payload = aggregate_payload(
        validation,
        records,
        per_ablation,
        paired_deltas,
        cross_seed,
        per_extractant,
        per_lanthanide,
        comparisons,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report = report_markdown(payload, per_extractant, per_lanthanide)

    # PASSED is persisted only after all aligned OOF recomputation succeeds.
    write_json_atomic(validation, output_dir / "validation.json")
    write_csv_atomic(cross_seed, output_dir / "cross_seed_oof_predictions.csv")
    write_csv_atomic(per_run, output_dir / "per_run_metrics.csv")
    write_csv_atomic(per_ablation, output_dir / "per_ablation_metrics.csv")
    write_csv_atomic(paired_deltas, output_dir / "paired_ablation_deltas.csv")
    write_csv_atomic(per_extractant, output_dir / "per_extractant_comparison.csv")
    write_csv_atomic(per_lanthanide, output_dir / "per_lanthanide_comparison.csv")
    write_json_atomic(payload, output_dir / "aggregate_summary.json")
    write_text_atomic(report, output_dir / "report.md")
    (output_dir / "_INCOMPLETE").unlink()
    write_json_atomic(
        {
            "status": "complete",
            "pair_scope": args.expected_pair_scope,
            "aggregate_implementation_sha256": aggregate_implementation_sha256,
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "bootstrap_seed": int(args.bootstrap_seed),
            "aggregate_summary_sha256": file_sha256(
                output_dir / "aggregate_summary.json"
            ),
            "artifact_sha256": {
                name: file_sha256(output_dir / name) for name in OUTPUT_ARTIFACTS
            },
        },
        output_dir / "_SUCCESS.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
