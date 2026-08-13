"""Frozen protocol handling for the additive generation-3 experiment layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


GEN3_PROTOCOL_SCHEMA_VERSION = "gen3.1"


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_gen3_protocol(path: Path | str) -> dict[str, Any]:
    protocol_path = Path(path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_gen3_protocol(payload)
    return payload


def validate_gen3_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != GEN3_PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"Expected protocol schema {GEN3_PROTOCOL_SCHEMA_VERSION!r}; "
            f"observed {protocol.get('schema_version')!r}."
        )
    if protocol.get("status") != "frozen_before_experiments":
        raise ValueError("The gen3 protocol is not marked frozen_before_experiments.")
    environment = protocol.get("reproducibility_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("The exact gen3 reproducibility_environment is missing.")
    for key in ("operating_system", "python", "requirements_file", "catboost", "torch"):
        if not str(environment.get(key, "")):
            raise ValueError(f"reproducibility_environment.{key} is missing.")

    inputs = protocol.get("immutable_inputs")
    splits = protocol.get("splits")
    metrics = protocol.get("metrics")
    if not all(isinstance(value, Mapping) for value in (inputs, splits, metrics)):
        raise ValueError("Protocol immutable_inputs, splits, and metrics must be mappings.")
    for key in (
        "dataset_sha256",
        "pair_cohort_sha256",
        "gen2_feature_registry_file_sha256",
        "gen2_feature_contract_sha256",
    ):
        value = str(inputs.get(key, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"immutable_inputs.{key} is not a SHA-256 digest.")

    split_seeds = tuple(int(value) for value in splits.get("split_seeds", ()))
    if len(split_seeds) < 5 or len(set(split_seeds)) != len(split_seeds):
        raise ValueError("At least five unique predeclared split seeds are required.")
    if split_seeds[0] != 104729:
        raise ValueError("The first split seed must remain the frozen 104729 split.")
    if any(value < 0 for value in split_seeds):
        raise ValueError("Split seeds must be nonnegative.")
    if metrics.get("primary") != "equal_extractant_macro_mae":
        raise ValueError("Generation 3 must select on equal_extractant_macro_mae.")
    if protocol.get("statistics", {}).get("bootstrap_unit") != "extractant":
        raise ValueError("The paired bootstrap unit must be extractant.")
    never_select = set(protocol.get("decision_rules", {}).get("never_select_on", ()))
    if not {"pooled_micro_mae", "pooled_r2"}.issubset(never_select):
        raise ValueError("Pooled metrics must be explicitly forbidden for selection.")

    h1 = protocol.get("h1", {})
    screening_candidates = tuple(h1.get("screening_candidates", ()))
    expanded_candidates = tuple(h1.get("expanded_candidates", ()))
    if int(h1.get("screening_budget_per_weight_loss_arm", -1)) != len(
        screening_candidates
    ):
        raise ValueError("The declared H1 screening budget must equal its candidate count.")
    if int(h1.get("expanded_budget_per_shortlisted_arm", -1)) != len(
        expanded_candidates
    ):
        raise ValueError("The declared H1 expanded budget must equal its candidate count.")
    h1_arm_count = len(h1.get("weighting_schemes", {})) * len(h1.get("losses", {}))
    shortlist_count = int(h1.get("expanded_shortlist_weight_loss_arms", -1))
    if not 1 <= shortlist_count <= h1_arm_count:
        raise ValueError("The H1 expanded shortlist must be within the arm count.")
    expected_candidate_evaluations = (
        h1_arm_count * len(screening_candidates)
        + shortlist_count * len(expanded_candidates)
    )
    if int(h1.get("inner_candidate_evaluations_per_outer_fold", -1)) != (
        expected_candidate_evaluations
    ):
        raise ValueError("The declared H1 per-fold candidate budget is inconsistent.")
    expected_fits = expected_candidate_evaluations * int(splits["inner_folds"])
    if int(h1.get("inner_catboost_fits_per_outer_fold", -1)) != expected_fits:
        raise ValueError("The declared H1 per-fold fit budget is inconsistent.")
    candidates = screening_candidates + expanded_candidates
    if not screening_candidates or not expanded_candidates:
        raise ValueError("Both H1 search phases require fixed candidates.")
    for candidate in candidates:
        depth = int(candidate["depth"])
        learning_rate = float(candidate["learning_rate"])
        iterations = int(candidate["iterations"])
        if not 4 <= depth <= 9:
            raise ValueError("H1 CatBoost depth must be in [4, 9].")
        if not 0.01 <= learning_rate <= 0.10:
            raise ValueError("H1 CatBoost learning_rate must be in [0.01, 0.10].")
        if not 500 <= iterations <= 3000:
            raise ValueError("H1 CatBoost iterations must be in [500, 3000].")

    lambdas = tuple(float(value) for value in protocol.get("h2", {}).get("lambda_grid", ()))
    if not lambdas or lambdas[0] != 0.0 or any(value < 0.0 or value > 1.0 for value in lambdas):
        raise ValueError("H2 lambda_grid must start at zero and remain in [0, 1].")

    h3 = protocol.get("h3", {})
    if len(tuple(h3.get("search_candidates", ()))) != 2:
        raise ValueError("H3 must retain its fixed two-candidate search budget.")
    training = h3.get("training", {})
    if int(training.get("max_epochs", 0)) != 80:
        raise ValueError("H3 max_epochs must remain at the frozen budget of 80.")


def verify_frozen_inputs(protocol: Mapping[str, Any], repo_root: Path | str) -> dict[str, str]:
    root = Path(repo_root)
    inputs = protocol["immutable_inputs"]
    dataset = root / str(inputs["dataset_path"])
    feature_registry = root / str(inputs["gen2_feature_registry_path"])
    observed = {
        "dataset_sha256": file_sha256(dataset),
        "gen2_feature_registry_file_sha256": file_sha256(feature_registry),
    }
    expected = {
        "dataset_sha256": str(inputs["dataset_sha256"]),
        "gen2_feature_registry_file_sha256": str(
            inputs["gen2_feature_registry_file_sha256"]
        ),
    }
    mismatches = {
        key: {"expected": expected[key], "observed": value}
        for key, value in observed.items()
        if value != expected[key]
    }
    if mismatches:
        raise ValueError(f"Frozen gen3 input hash mismatch: {mismatches}")
    return observed


def validate_seed_pair(
    protocol: Mapping[str, Any], *, split_seed: int, model_seed: int
) -> str:
    splits = protocol["splits"]
    primary_split_seeds = tuple(int(value) for value in splits["split_seeds"])
    primary_model_seed = int(splits["primary_model_seed"])
    confirmation = tuple(int(value) for value in splits["confirmation_model_seeds_on_split_104729"])
    if int(split_seed) in primary_split_seeds and int(model_seed) == primary_model_seed:
        return "primary_split_seed"
    if int(split_seed) == primary_split_seeds[0] and int(model_seed) in confirmation:
        return "secondary_model_seed_confirmation"
    raise ValueError(
        f"model_seed={model_seed}, split_seed={split_seed} is absent from the frozen protocol."
    )
