"""Data access for the refined ml_separator figures.

Thin wrapper over the verified helpers that already live in ``figures/scripts`` — path
resolution (main checkout first, then the gen10 finalisation worktree), the repository's
own paired chemotype bootstrap, and the derived k-shot tables.  Nothing here recomputes a
metric; the refinement pass is aesthetic and must read the same numbers the metric audit
verified (``figures/METRIC_AUDIT.md``, 67/67 PASS).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_FIGSCRIPTS = REPO / "figures" / "scripts"
if str(_FIGSCRIPTS) not in sys.path:
    sys.path.insert(0, str(_FIGSCRIPTS))

import _paths as _p          # noqa: E402  path resolution across checkout and worktree
import _stats as _s          # noqa: E402  the repository's own bootstraps

run = _p.run
DERIVED = _p.DERIVED
paired_chemotype_bootstrap = _s.paired_chemotype_bootstrap
block_bootstrap_mean = _s.block_bootstrap_mean
block_bootstrap_stat = _s.block_bootstrap_stat

SEEDS = (104729, 130363, 155921, 196613, 262147)


def derived(name: str, builder: str | None = None) -> Path:
    """Path to a derived table, building it once if it is missing."""
    path = DERIVED / name
    if not path.exists():
        if builder is None:
            raise FileNotFoundError(f"{path} is missing and no builder was given")
        subprocess.run([sys.executable, str(_FIGSCRIPTS / builder)], check=True)
    return path


def kshot_per_ligand() -> pd.DataFrame:
    """Mean MAE per (global model, adapter, policy, k, extractant) on the common cohort."""
    return pd.read_csv(derived("kshot_per_ligand.csv", "prepare_kshot_tables.py"))


def kshot_per_seed() -> pd.DataFrame:
    return pd.read_csv(derived("kshot_per_seed.csv", "prepare_kshot_tables.py"))


def curve_shape(axis: str | None = "extractant") -> pd.DataFrame:
    frame = pd.read_parquet(run("gen9_shape/shape/curve_shape.parquet"))
    return frame if axis is None else frame[frame.axis_label == axis].copy()
