"""Models and evaluation tools for lanthanide solvent-extraction selectivity."""

from .ablation import AblationBenchmarkResult, run_ablation_benchmark
from .feature_registry import FeatureRegistry, build_feature_registry
from .pairs import (
    PAIR_SCOPES,
    PairDataset,
    build_adjacent_pair_dataset,
    build_lanthanide_pair_dataset,
)

__all__ = [
    "AblationBenchmarkResult",
    "FeatureRegistry",
    "PAIR_SCOPES",
    "PairDataset",
    "build_adjacent_pair_dataset",
    "build_feature_registry",
    "build_lanthanide_pair_dataset",
    "run_ablation_benchmark",
]
