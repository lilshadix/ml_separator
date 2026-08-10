#!/usr/bin/env python3
"""Run the guarded metal-centred simplicial neural benchmark."""

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

import numpy as np
import pandas as pd
import sklearn
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lanthanide_separation.deep_evaluation import (  # noqa: E402
    CONTEXT_MODES,
    SimplicialTrainingConfig,
    nested_simplicial_benchmark,
    validate_vr_pair_links,
)
from lanthanide_separation.evaluation import DEFAULT_PARAMETER_GRID  # noqa: E402
from lanthanide_separation.pairs import (  # noqa: E402
    PAIR_SCOPES,
    build_lanthanide_pair_dataset,
)
from lanthanide_separation.simplicial import (  # noqa: E402
    SIMPLEX_ORDERS,
    VietorisRipsStore,
)


DEFAULT_BUNDLE = REPO_ROOT / "dataset with 3D structures"
DEFAULT_DATASET = DEFAULT_BUNDLE / "dataset.parquet"
DEFAULT_VR_ASSET = DEFAULT_BUNDLE / "features" / "vietoris_rips_inputs.npz"
DEFAULT_ROW_MAP = DEFAULT_BUNDLE / "row_geometry_map.csv"
MAX_BASE_SEED = 4_000_000_000


def _default_n_jobs() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        return max(1, int(value)) if value is not None else -1
    except ValueError:
        return -1


def _default_torch_threads() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        if value is not None:
            return max(1, int(value))
    except ValueError:
        pass
    return max(1, int(os.cpu_count() or 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--vr-assets", type=Path, default=DEFAULT_VR_ASSET)
    parser.add_argument("--row-geometry-map", type=Path, default=DEFAULT_ROW_MAP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pair-scope",
        choices=PAIR_SCOPES,
        default="all",
        help=(
            "'all' evaluates every observed lanthanide pair under matched conditions; "
            "'adjacent' reproduces the historical nearest-neighbour cohort."
        ),
    )
    parser.add_argument(
        "--logical-output-dir",
        type=Path,
        default=None,
        help="Final path recorded in provenance when output-dir is a temporary attempt path.",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="Forest/neural seed.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=_default_n_jobs())
    parser.add_argument("--torch-threads", type=int, default=_default_torch_threads())
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--determinism",
        choices=("strict", "warn"),
        default="strict",
        help="CPU primary uses strict; CUDA scatter may require explicit warn mode.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--initializations", type=int, default=3)
    parser.add_argument("--max-pairs-per-batch", type=int, default=64)
    parser.add_argument("--max-simplices-per-batch", type=int, default=60_000)
    parser.add_argument("--gate-z", type=float, default=0.50)
    parser.add_argument(
        "--simplex-order",
        choices=SIMPLEX_ORDERS,
        default="nodes_edges_triangles",
        help=(
            "Declared encoder ladder: distance-only geometry, nodes+distances, "
            "nodes+edges, or the full nodes+edges+triangles network (default). "
            "Every rung keeps the same weight shapes, so a difference is "
            "attributable to simplicial order rather than to capacity."
        ),
    )
    parser.add_argument(
        "--context-mode",
        choices=CONTEXT_MODES,
        default="full",
        help=(
            "'conditions_only' is the learned-geometry-only arm: no ligand 2D "
            "descriptor reaches the network, so the encoded structure has to "
            "carry the ligand signal. Conditions and metal identity are kept "
            "because the target is a condition-matched difference."
        ),
    )
    parser.add_argument(
        "--geometry-null-seeds",
        default="",
        help=(
            "Comma-separated seeds for the learned-geometry permutation control. "
            "Each seed trains the same network on training-fold-permuted "
            "geometry and scores it on the untouched held-out rows. Empty "
            "disables the control."
        ),
    )
    parser.add_argument(
        "--shell-mode",
        choices=("coordination", "radius"),
        default="coordination",
        help="Primary keeps Ln plus flagged donors and only Ln-donor-donor triangles.",
    )
    parser.add_argument("--radius-angstrom", type=float, default=3.10)
    parser.add_argument(
        "--max-filtration-angstrom",
        type=float,
        default=4.00,
        help="Frozen RBF support; 4.0 A matches the bundled VR construction cutoff.",
    )
    parser.add_argument("--max-edges-per-complex", type=int, default=512)
    parser.add_argument("--max-triangles-per-complex", type=int, default=512)
    parser.add_argument(
        "--use-partial-charges",
        action="store_true",
        help="Sensitivity only; primary geometry model excludes charge availability.",
    )
    parser.add_argument(
        "--group-mode",
        choices=("extractant", "ecfp-exact-cluster"),
        default="ecfp-exact-cluster",
    )
    parser.add_argument(
        "--replicate-policy", choices=("unique", "median"), default="unique"
    )
    parser.add_argument("--allow-incomplete-conditions", action="store_true")
    parser.add_argument("--quarantine-known-censored-targets", action="store_true")
    parser.add_argument("--no-default-quarantine", action="store_true")
    parser.add_argument(
        "--evaluation-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only nested OOF evaluation is implemented; no deployment model is fit.",
    )
    parser.add_argument("--expected-dataset-sha256", default=None)
    parser.add_argument("--expected-vr-sha256", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "outer_folds",
        "inner_folds",
        "trees",
        "bootstrap",
        "torch_threads",
        "hidden_dim",
        "layers",
        "epochs",
        "initializations",
        "max_pairs_per_batch",
        "max_simplices_per_batch",
        "max_edges_per_complex",
        "max_triangles_per_complex",
    )
    for name in positive_integer_names:
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise SystemExit("Outer and inner folds must be at least 2.")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs cannot be zero.")
    if args.hidden_dim < 4:
        raise SystemExit("--hidden-dim must be at least 4.")
    for name in ("seed", "split_seed"):
        value = int(getattr(args, name))
        if value < 0 or value > MAX_BASE_SEED:
            raise SystemExit(
                f"--{name.replace('_', '-')} must be between 0 and {MAX_BASE_SEED}."
            )
    if not 0.0 <= args.dropout < 1.0:
        raise SystemExit("--dropout must be in [0, 1).")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise SystemExit("Learning rate must be positive and weight decay nonnegative.")
    if (
        args.radius_angstrom <= 0.0
        or args.max_filtration_angstrom <= 0.0
        or args.gate_z < 0.0
    ):
        raise SystemExit("Radius/filtration scale must be positive and gate-z nonnegative.")
    if args.replicate_policy != "unique":
        raise SystemExit(
            "The simplicial benchmark requires --replicate-policy=unique; median "
            "aggregation cannot select one VR graph without mixing geometry provenance."
        )
    if not args.evaluation_only:
        raise SystemExit("Deployment fitting is not implemented; use --evaluation-only.")
    raw_null_seeds = str(args.geometry_null_seeds).strip()
    seed_values: list[int] = []
    if raw_null_seeds:
        for token in raw_null_seeds.split(","):
            token = token.strip()
            if not token.isdigit():
                raise SystemExit(
                    "--geometry-null-seeds must be comma-separated nonnegative "
                    f"integers; got {token!r}."
                )
            value = int(token)
            if value > MAX_BASE_SEED:
                raise SystemExit(
                    f"--geometry-null-seeds entries must not exceed {MAX_BASE_SEED}."
                )
            seed_values.append(value)
        if len(set(seed_values)) != len(seed_values):
            raise SystemExit("--geometry-null-seeds must be unique.")
    args.geometry_null_seed_values = tuple(seed_values)
    for name in ("expected_dataset_sha256", "expected_vr_sha256"):
        value = getattr(args, name)
        if value is None:
            continue
        value = value.strip().lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise SystemExit(f"--{name.replace('_', '-')} must be a SHA-256 hex digest.")
        setattr(args, name, value)


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_text_atomic(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(payload: Any, path: Path) -> None:
    write_text_atomic(
        json.dumps(
            json_compatible(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        path,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slurm_metadata() -> dict[str, str | None]:
    keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_SUBMIT_DIR",
        "CUDA_VISIBLE_DEVICES",
    )
    return {**{key: os.environ.get(key) for key in keys}, "hostname": socket.gethostname()}


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device=cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strict_boolean_series(series: pd.Series, *, label: str) -> pd.Series:
    def parse(value: Any) -> bool:
        if pd.isna(value):
            raise ValueError(f"{label} contains a null value.")
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            return bool(int(value))
        if isinstance(value, (float, np.floating)) and float(value) in (0.0, 1.0):
            return bool(int(value))
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"{label} contains an invalid boolean value: {value!r}")

    return series.map(parse).astype(bool)


def validate_row_map(
    pair_frame: pd.DataFrame,
    row_map_path: Path,
    store: VietorisRipsStore,
) -> None:
    mapping = pd.read_csv(
        row_map_path,
        usecols=(
            "safe_exp_id",
            "geometry_key",
            "geometry_ok",
            "geometry_feature_build_id",
            "vr_graph_index",
        ),
    )
    if mapping["safe_exp_id"].isna().any():
        raise ValueError("row_geometry_map.safe_exp_id contains null values.")
    mapping["safe_exp_id"] = mapping["safe_exp_id"].astype(str).str.strip()
    if mapping["safe_exp_id"].eq("").any():
        raise ValueError("row_geometry_map.safe_exp_id contains empty values.")
    if mapping["safe_exp_id"].duplicated().any():
        raise ValueError("row_geometry_map.safe_exp_id must be unique.")
    mapping["geometry_ok"] = strict_boolean_series(
        mapping["geometry_ok"], label="row_geometry_map.geometry_ok"
    )
    raw_vr_indices = mapping["vr_graph_index"]
    vr_index_provided = (
        raw_vr_indices.notna()
        & raw_vr_indices.astype("string").str.strip().ne("").fillna(False)
    ).to_numpy(dtype=bool)
    vr_indices = pd.to_numeric(
        raw_vr_indices.where(vr_index_provided), errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    valid_vr_index = (
        np.isfinite(vr_indices)
        & np.equal(vr_indices, np.floor(vr_indices))
        & (vr_indices >= 0)
    )
    if np.any(vr_index_provided & ~valid_vr_index):
        raise ValueError(
            "Every supplied row_geometry_map.vr_graph_index must be a finite "
            "nonnegative integer."
        )
    geometry_ok = mapping["geometry_ok"].to_numpy(dtype=bool)
    if np.any(geometry_ok & ~valid_vr_index):
        raise ValueError(
            "Every geometry_ok=True row in row_geometry_map must have a finite "
            "nonnegative vr_graph_index."
        )
    mapping["vr_graph_index"] = pd.array(
        np.where(valid_vr_index, vr_indices, np.nan), dtype="Int64"
    )
    mapping = mapping.set_index("safe_exp_id", verify_integrity=True)
    for side in ("A", "B"):
        source_ids = pair_frame[f"source_id_{side}"].astype(str)
        missing = sorted(set(source_ids) - set(mapping.index.astype(str)))
        if missing:
            raise ValueError(f"row_geometry_map lacks source IDs: {missing[:5]}")
        aligned = mapping.loc[source_ids]
        if not aligned["geometry_ok"].all():
            raise ValueError(f"Side {side} contains a row without accepted geometry.")
        comparisons = (
            (
                aligned["geometry_key"].astype(str).to_numpy(),
                pair_frame[f"geometry_key_{side}"].astype(str).to_numpy(),
                "geometry_key",
            ),
            (
                aligned["geometry_feature_build_id"].astype(str).to_numpy(),
                pair_frame[f"geometry_feature_build_id_{side}"].astype(str).to_numpy(),
                "geometry_feature_build_id",
            ),
            (
                pd.to_numeric(aligned["vr_graph_index"]).to_numpy(dtype=np.int64),
                pd.to_numeric(pair_frame[f"vr_graph_index_{side}"]).to_numpy(dtype=np.int64),
                "vr_graph_index",
            ),
        )
        for observed, expected, label in comparisons:
            if not np.array_equal(observed, expected):
                raise ValueError(f"Pair/{row_map_path.name} mismatch for {label} on side {side}.")
    validate_vr_pair_links(pair_frame, store)


def graph_statistics(pair_frame: pd.DataFrame, store: VietorisRipsStore) -> dict[str, Any]:
    build_ids = sorted(
        set(pair_frame["geometry_feature_build_id_A"].astype(str))
        | set(pair_frame["geometry_feature_build_id_B"].astype(str))
    )
    node_counts: list[int] = []
    edge_counts: list[int] = []
    triangle_counts: list[int] = []
    maximum_edge_filtration = 0.0
    maximum_triangle_side = 0.0
    for build_id in build_ids:
        graph = store.graph(build_id)
        node_counts.append(len(graph.atomic_numbers))
        edge_counts.append(graph.edge_index.shape[1])
        triangle_counts.append(graph.triangle_index.shape[1])
        if len(graph.edge_filtration):
            maximum_edge_filtration = max(
                maximum_edge_filtration, float(np.max(graph.edge_filtration))
            )
        if len(graph.triangle_shape_features):
            maximum_triangle_side = max(
                maximum_triangle_side,
                float(np.max(graph.triangle_shape_features[:, :3])),
            )

    def describe(values: list[int]) -> dict[str, float | int]:
        array = np.asarray(values, dtype=float)
        return {
            "min": int(array.min()),
            "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)),
            "max": int(array.max()),
        }

    return {
        "required_unique_graphs": len(build_ids),
        "nodes": describe(node_counts),
        "edges": describe(edge_counts),
        "triangles": describe(triangle_counts),
        "maximum_edge_filtration_angstrom": maximum_edge_filtration,
        "maximum_triangle_side_angstrom": maximum_triangle_side,
        "closure_policy": (
            "complete selected edge set; distinct triangle vertices; every triangle "
            "boundary is present and born no later than the triangle; caps hard-fail"
        ),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    dataset_path = args.dataset.expanduser().resolve()
    vr_path = args.vr_assets.expanduser().resolve()
    row_map_path = args.row_geometry_map.expanduser().resolve()
    for label, path in (
        ("dataset", dataset_path),
        ("VR asset", vr_path),
        ("row geometry map", row_map_path),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    dataset_sha256 = sha256_file(dataset_path)
    vr_sha256 = sha256_file(vr_path)
    if args.expected_dataset_sha256 and dataset_sha256 != args.expected_dataset_sha256:
        raise SystemExit("Dataset SHA-256 does not match --expected-dataset-sha256.")
    if args.expected_vr_sha256 and vr_sha256 != args.expected_vr_sha256:
        raise SystemExit("VR SHA-256 does not match --expected-vr-sha256.")

    output_dir = args.output_dir.expanduser().resolve()
    logical_output_dir = (
        args.logical_output_dir.expanduser().resolve()
        if args.logical_output_dir is not None
        else output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(
        {
            "status": "incomplete",
            "started_at_utc": started_at,
            "dataset_sha256": dataset_sha256,
            "vr_asset_sha256": vr_sha256,
            "logical_output_dir": str(logical_output_dir),
            "staging_output_dir": str(output_dir),
            "arguments": vars(args),
            "slurm": slurm_metadata(),
        },
        output_dir / "run_config.json",
    )
    write_text_atomic("Run has not completed.\n", output_dir / "_INCOMPLETE")

    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)
    print(f"Dataset: {dataset_path}", flush=True)
    print(f"VR asset: {vr_path}", flush=True)
    source = pd.read_parquet(dataset_path)
    pair_data = build_lanthanide_pair_dataset(
        source,
        pair_scope=args.pair_scope,
        require_geometry=True,
        replicate_policy=args.replicate_policy,
        quarantine_known_bad=not args.no_default_quarantine,
        quarantine_known_censored_targets=args.quarantine_known_censored_targets,
        require_complete_conditions=not args.allow_incomplete_conditions,
        delta3d_feature_set="compact-invariant",
    )
    store = VietorisRipsStore(
        vr_path,
        radius_angstrom=args.radius_angstrom,
        max_edges=args.max_edges_per_complex,
        max_triangles=args.max_triangles_per_complex,
        use_partial_charges=args.use_partial_charges,
        shell_mode=args.shell_mode,
    )
    validate_row_map(pair_data.frame, row_map_path, store)
    asset_statistics = graph_statistics(pair_data.frame, store)
    observed_filtration_maximum = max(
        float(asset_statistics["maximum_edge_filtration_angstrom"]),
        float(asset_statistics["maximum_triangle_side_angstrom"]),
    )
    if observed_filtration_maximum > args.max_filtration_angstrom + 1e-5:
        raise ValueError(
            "--max-filtration-angstrom does not cover the selected VR geometry: "
            f"observed {observed_filtration_maximum:.6f} A, configured "
            f"{args.max_filtration_angstrom:.6f} A."
        )
    asset_contract = {
        "vr_asset_path": str(vr_path),
        "vr_asset_sha256": vr_sha256,
        "row_geometry_map_path": str(row_map_path),
        "row_geometry_map_sha256": sha256_file(row_map_path),
        "mapping_chain": (
            "source_id -> row_geometry_map.safe_exp_id -> vr_graph_index -> "
            "VR build_ids[index] == geometry_feature_build_id"
        ),
        "shell_mode": args.shell_mode,
        "radius_angstrom": args.radius_angstrom,
        "rbf_max_filtration_angstrom": args.max_filtration_angstrom,
        "use_partial_charges": args.use_partial_charges,
        "statistics": asset_statistics,
    }
    write_json(pair_data.audit, output_dir / "pair_build_audit.json")
    write_json(asset_contract, output_dir / "asset_contract.json")
    write_json(
        {
            "pair_scope": args.pair_scope,
            "simplicial_input": "0/1/2-simplex coordination subcomplex",
            "triangle_invariants": [
                "three sorted side lengths",
                "area divided by coordination-radius squared",
                "scale-free triangle quality",
                "cosine of donor-metal-donor angle",
                "metal-membership flag",
            ],
            "message_passing": (
                "shared node-to-edge and node-to-triangle incidence updates with "
                "metal, donor mean/dispersion, direct triangle and all-node pooling"
            ),
            "rbf_max_filtration_angstrom": args.max_filtration_angstrom,
            "cohort_sha256": pair_data.audit["cohort_sha256"],
            "forbidden_model_inputs": [
                "targets",
                "source/build/geometry IDs",
                "asset indices",
                "paths",
                "ECFP identity in the neural branch",
                "partial charge availability in the primary profile",
            ],
        },
        output_dir / "feature_contract.json",
    )
    if not pair_data.quarantine.empty:
        pair_data.quarantine.to_csv(output_dir / "quarantined_source_rows.csv", index=False)

    config = SimplicialTrainingConfig(
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
        max_filtration=args.max_filtration_angstrom,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        initializations=args.initializations,
        max_pairs_per_batch=args.max_pairs_per_batch,
        max_simplices_per_batch=args.max_simplices_per_batch,
        gate_z=args.gate_z,
        strict_determinism=args.determinism == "strict",
        simplex_order=args.simplex_order,
        context_mode=args.context_mode,
        geometry_null_seeds=tuple(args.geometry_null_seed_values),
    )
    group_column = "extractant" if args.group_mode == "extractant" else "ecfp_exact_cluster"
    result = nested_simplicial_benchmark(
        pair_data,
        store,
        group_column=group_column,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        n_estimators=args.trees,
        n_jobs=args.n_jobs,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        split_seed=args.split_seed,
        device=device,
        training_config=config,
        parameter_grid=DEFAULT_PARAMETER_GRID,
    )

    result.predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    result.fold_assignments.to_csv(output_dir / "fold_assignments.csv", index=False)
    result.tuning_results.to_csv(output_dir / "inner_cv_tuning.csv", index=False)
    result.training_history.to_csv(output_dir / "simplicial_training_history.csv", index=False)
    result.per_extractant_metrics.to_csv(
        output_dir / "per_extractant_metrics.csv", index=False
    )
    result.per_pair_metrics.to_csv(output_dir / "per_pair_type_metrics.csv", index=False)

    artifact_names = [
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
    ]
    if (output_dir / "quarantined_source_rows.csv").is_file():
        artifact_names.append("quarantined_source_rows.csv")
    artifact_sha256 = {
        name: sha256_file(output_dir / name) for name in sorted(artifact_names)
    }

    implementation_paths = (
        Path(__file__).resolve(),
        SRC_ROOT / "lanthanide_separation" / "pairs.py",
        SRC_ROOT / "lanthanide_separation" / "evaluation.py",
        SRC_ROOT / "lanthanide_separation" / "simplicial.py",
        SRC_ROOT / "lanthanide_separation" / "deep_evaluation.py",
    )
    implementation_sha256 = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in implementation_paths
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "cohort_sha256": pair_data.audit["cohort_sha256"],
        "asset_sha256": {
            "vietoris_rips_inputs.npz": vr_sha256,
            "row_geometry_map.csv": asset_contract["row_geometry_map_sha256"],
        },
        "implementation_sha256": implementation_sha256,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device_type": device.type,
            "cuda_version": torch.version.cuda,
        },
        "slurm": slurm_metadata(),
        "output_dir": str(logical_output_dir),
        "staging_output_dir": str(output_dir),
        "arguments": vars(args),
        "pair_build_audit": pair_data.audit,
        "asset_contract": asset_contract,
        "benchmark": result.summary,
        "artifacts": {
            "deployment_model_saved": False,
            "sha256": artifact_sha256,
        },
    }
    write_json(summary, output_dir / "summary.json")
    (output_dir / "_INCOMPLETE").unlink()
    write_json(
        {
            "status": "complete",
            "summary_sha256": sha256_file(output_dir / "summary.json"),
        },
        output_dir / "_SUCCESS.json",
    )

    primary = result.summary["improvements"]["primary_simplicial_vs_delta3d"]
    interval = result.summary["paired_group_bootstrap"]["comparisons"][
        "simplicial_vs_delta3d"
    ]["r2_gain"]
    print(
        f"Simplicial hybrid vs tabular Delta3D: Delta R2={primary['r2_gain']:+.5f}; "
        f"95% fixed-OOF group CI=[{interval['ci95_low']:+.5f}, "
        f"{interval['ci95_high']:+.5f}]",
        flush=True,
    )
    print(f"Saved: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
