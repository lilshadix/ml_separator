"""Fingerprint similarity, chemotypes and the near/mid/far bands.

Everything here is target-free by construction: no function accepts ``log_D``.
The ECFP bits are the bundle's pre-computed Morgan radius-2 / 2048-bit / achiral
fingerprints (verified bit-exact against RDKit on all 190 structures, see
DATA_AUDIT.md §6).  Chemotypes are the repository's single-linkage Tanimoto-0.7
super-clusters (:func:`lanthanide_separation.levels.tanimoto_cluster_labels`).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from . import paths  # noqa: F401  (puts src/ on sys.path)
from lanthanide_separation.levels import (  # noqa: E402
    TANIMOTO_CLUSTER_THRESHOLD, ecfp_cluster_labels, tanimoto_cluster_labels,
)

ECFP_COLUMNS: tuple[str, ...] = tuple(f"ecfp_{i}" for i in range(2048))
#: gen10/gen11 bands, reused unchanged (pre-registered before any Gen12 model ran).
BANDS: tuple[tuple[str, float, float], ...] = (
    ("far", -np.inf, 0.40),
    ("mid", 0.40, 0.60),
    ("near", 0.60, np.inf),
)


def fingerprint_matrix(frame: pd.DataFrame, key: str = "extractant") -> tuple[list[str], np.ndarray]:
    """One binary fingerprint per distinct ``key`` value, asserted consistent."""
    block = frame[[key, *ECFP_COLUMNS]].drop_duplicates()
    per = block.groupby(key).size()
    if (per > 1).any():
        raise ValueError(f"{int((per > 1).sum())} extractants carry more than one fingerprint")
    block = block.set_index(key)
    bits = block[list(ECFP_COLUMNS)].to_numpy()
    if not np.isin(bits, (0, 1)).all():
        raise ValueError("ECFP block must be binary")
    return list(block.index.astype(str)), bits.astype(np.uint8)


def tanimoto_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    a = a.astype(np.float64)
    b = a if b is None else b.astype(np.float64)
    inter = a @ b.T
    ca, cb = a.sum(1), b.sum(1)
    union = ca[:, None] + cb[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return np.clip(sim, 0.0, 1.0)


def max_similarity_to_reference(query_names: Sequence[str], reference_names: Sequence[str],
                                names: Sequence[str], bits: np.ndarray) -> pd.DataFrame:
    """``max_train_tanimoto`` and the nearest reference structure for each query."""
    index = {n: i for i, n in enumerate(names)}
    q = np.array([index[n] for n in query_names]); r = np.array([index[n] for n in reference_names])
    sim = tanimoto_matrix(bits[q], bits[r])
    best = sim.argmax(1)
    return pd.DataFrame({
        "extractant": list(query_names),
        "max_train_tanimoto": sim[np.arange(len(q)), best],
        "nearest_train_extractant": [reference_names[j] for j in best],
        "n_train_above_0_7": (sim >= 0.7).sum(1),
        "n_train_above_0_5": (sim >= 0.5).sum(1),
    })


def band_of(similarity: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(np.asarray(similarity, dtype=float))
    out = pd.Series(index=s.index, dtype=object)
    for name, low, high in BANDS:
        out[(s > low) & (s <= high)] = name
    if out.isna().any():
        raise ValueError("a similarity fell outside every band (NaN?)")
    return out


def ecfp_clusters(frame: pd.DataFrame) -> pd.Series:
    return ecfp_cluster_labels(frame, ECFP_COLUMNS)


def eu_local_chemotypes(frame: pd.DataFrame, threshold: float = TANIMOTO_CLUSTER_THRESHOLD) -> pd.Series:
    """Single-linkage chemotypes computed on the Eu structures alone (diagnostic only)."""
    return tanimoto_cluster_labels(frame, ECFP_COLUMNS, threshold=threshold)


def frozen_chemotypes(extractants: pd.Series) -> pd.DataFrame:
    """The gen6 frozen all-190 chemistry map: supercluster, ECFP cluster, family."""
    cm = pd.read_parquet(paths.CHEMISTRY_MAP_PARQUET)
    cm = cm.set_index("extractant")
    missing = sorted(set(extractants) - set(cm.index))
    if missing:
        raise KeyError(f"{len(missing)} extractants are absent from the frozen chemistry map")
    sub = cm.loc[extractants.astype(str)]
    return pd.DataFrame({
        "chemotype": sub["chem__supercluster"].to_numpy(),
        "chem_family": sub["chem__family"].to_numpy(),
        "chem_family_source": sub["chem__family_source"].to_numpy(),
        "frozen_ecfp_cluster": sub["chem__ecfp_cluster"].to_numpy(),
    }, index=extractants.index)
