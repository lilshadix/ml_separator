"""Generation-3 adapters over the unchanged generation-2 pair cohort."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .electronic import COMPLEX_ELECTRONIC_SOURCES, DERIVED_COMPLEX_ELECTRONIC_NAME
from .feature_registry import FeatureRegistry
from .pairs import PairDataset


COMPLEX_ELECTRONIC_NAMES: tuple[str, ...] = tuple(
    source.name for source in COMPLEX_ELECTRONIC_SOURCES
) + (DERIVED_COMPLEX_ELECTRONIC_NAME,)


@dataclass(frozen=True)
class Gen3FeatureAdapter:
    """Exact A2/E3 contracts plus shoulder-specific latent-difference inputs."""

    frame: pd.DataFrame
    a2_columns: tuple[str, ...]
    e3_raw_columns: tuple[str, ...]
    e3_compact_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    shoulder_a_columns: tuple[str, ...]
    shoulder_b_columns: tuple[str, ...]
    electronic_shoulder_a_columns: tuple[str, ...]
    electronic_shoulder_b_columns: tuple[str, ...]
    hashes: dict[str, str]

    @property
    def h3_a2_a_columns(self) -> tuple[str, ...]:
        return self.context_columns + self.shoulder_a_columns

    @property
    def h3_a2_b_columns(self) -> tuple[str, ...]:
        return self.context_columns + self.shoulder_b_columns

    @property
    def h3_elec_a_columns(self) -> tuple[str, ...]:
        return self.h3_a2_a_columns + self.electronic_shoulder_a_columns

    @property
    def h3_elec_b_columns(self) -> tuple[str, ...]:
        return self.h3_a2_b_columns + self.electronic_shoulder_b_columns


def _column_hash(columns: tuple[str, ...]) -> str:
    payload = json.dumps(list(columns), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reference_columns(
    feature_registry_payload: Mapping[str, Any], family: str
) -> tuple[str, ...]:
    families = feature_registry_payload.get("families", {})
    if family not in families:
        raise ValueError(f"Reference feature registry has no {family!r} family.")
    columns = tuple(str(value) for value in families[family].get("columns", ()))
    declared = int(families[family].get("column_count", -1))
    if len(columns) != declared:
        raise ValueError(f"Reference {family} column count is inconsistent.")
    return columns


def build_gen3_feature_adapter(
    pair_data: PairDataset,
    registry: FeatureRegistry,
    source: pd.DataFrame,
    *,
    reference_feature_registry: Path | str | Mapping[str, Any],
) -> Gen3FeatureAdapter:
    """Build target-independent views without changing rows or gen2 features."""

    if isinstance(reference_feature_registry, Mapping):
        reference = dict(reference_feature_registry)
    else:
        reference = json.loads(
            Path(reference_feature_registry).read_text(encoding="utf-8")
        )
    a2_columns = registry.ablation_columns("A2")
    e3_raw_columns = registry.columns_for_family("ELEC_PAIR")
    reference_a2 = tuple(
        str(value) for value in reference["ablations"]["A2"].get("columns", ())
    )
    if not reference_a2:
        reference_a2 = tuple(
            column
            for family in ("CONDITIONS", "LN", "2D")
            for column in _reference_columns(reference, family)
        )
        reference_order = tuple(
            str(item["column"])
            for item in reference.get("assignments", ())
            if str(item.get("family")) in {"CONDITIONS", "LN", "2D"}
        )
        if reference_order:
            reference_a2 = reference_order
    reference_e3 = _reference_columns(reference, "ELEC_PAIR")
    if a2_columns != reference_a2:
        raise ValueError("Generated A2 columns differ from the frozen gen2 registry.")
    if e3_raw_columns != reference_e3:
        raise ValueError("Generated E3 columns differ from the frozen gen2 registry.")

    frame = pair_data.frame.reset_index(drop=True).copy()
    family_map = (
        source.loc[:, ["canonical_smiles", "extractant_group"]]
        .drop_duplicates()
        .set_index("canonical_smiles")["extractant_group"]
        .astype("string")
    )
    if family_map.index.has_duplicates:
        raise ValueError("One extractant maps to multiple extractant_group labels.")
    frame["extractant_family"] = frame["extractant"].map(family_map)
    if frame["extractant_family"].isna().any():
        missing = sorted(frame.loc[frame["extractant_family"].isna(), "extractant"].unique())
        raise ValueError(f"Missing extractant family labels: {missing[:5]}")

    compact_columns: list[str] = []
    elec_a_columns: list[str] = []
    elec_b_columns: list[str] = []
    compact_payload: dict[str, np.ndarray] = {}
    shoulder_payload: dict[str, np.ndarray] = {}
    for name in COMPLEX_ELECTRONIC_NAMES:
        delta_column = f"elec__odd__delta_{name}"
        abs_column = f"elec__even__absdelta_{name}"
        mean_column = f"elec__even__mean_{name}"
        for column in (delta_column, abs_column, mean_column):
            if column not in frame.columns:
                raise ValueError(f"Electronic adapter source column is absent: {column}")
        compact_columns.extend((delta_column, abs_column, mean_column))
        compact_payload[delta_column] = frame[delta_column].to_numpy(dtype=float)
        compact_payload[abs_column] = frame[abs_column].to_numpy(dtype=float)
        compact_payload[mean_column] = frame[mean_column].to_numpy(dtype=float)
        a_column = f"gen3__elec_A__{name}"
        b_column = f"gen3__elec_B__{name}"
        delta = frame[delta_column].to_numpy(dtype=float)
        mean = frame[mean_column].to_numpy(dtype=float)
        shoulder_payload[a_column] = mean + 0.5 * delta
        shoulder_payload[b_column] = mean - 0.5 * delta
        elec_a_columns.append(a_column)
        elec_b_columns.append(b_column)

    compact_frame = pd.DataFrame(compact_payload, index=frame.index)
    # Existing columns are reused rather than overwritten; the compact view is
    # an ordered selection of delta, magnitude, and mean quantities.
    for column in compact_frame:
        if not np.allclose(
            frame[column].to_numpy(dtype=float),
            compact_frame[column].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise AssertionError(f"Compact E3 reconstruction changed {column}.")
    frame = pd.concat([frame, pd.DataFrame(shoulder_payload, index=frame.index)], axis=1)

    context_columns = tuple(
        column
        for column in a2_columns
        if column.startswith("base__cond__")
        or column.startswith("base__ecfp_")
        or column.startswith("base__") and not column.startswith("base__cond__")
    )
    # Fail closed: A2 must partition into shared ligand/condition context and
    # exactly the eight pair-level lanthanide columns.
    ln_columns = tuple(column for column in a2_columns if column.startswith("pair__"))
    if len(context_columns) + len(ln_columns) != len(a2_columns) or len(ln_columns) != 8:
        raise AssertionError("A2 did not partition into context plus eight Ln columns.")

    shoulder_a_columns = ("pair__Z_A", "pair__ionic_radius_A")
    shoulder_b_columns = ("pair__Z_B", "pair__ionic_radius_B")
    if not set(shoulder_a_columns + shoulder_b_columns).issubset(frame.columns):
        raise AssertionError("Pair shoulders cannot be reconstructed from A2.")

    hashes = {
        "a2_columns_sha256": _column_hash(a2_columns),
        "e3_raw_columns_sha256": _column_hash(e3_raw_columns),
        "e3_compact_columns_sha256": _column_hash(tuple(compact_columns)),
        "h3_context_columns_sha256": _column_hash(context_columns),
        "h3_electronic_names_sha256": _column_hash(COMPLEX_ELECTRONIC_NAMES),
    }
    return Gen3FeatureAdapter(
        frame=frame,
        a2_columns=a2_columns,
        e3_raw_columns=e3_raw_columns,
        e3_compact_columns=tuple(compact_columns),
        context_columns=context_columns,
        shoulder_a_columns=shoulder_a_columns,
        shoulder_b_columns=shoulder_b_columns,
        electronic_shoulder_a_columns=tuple(elec_a_columns),
        electronic_shoulder_b_columns=tuple(elec_b_columns),
        hashes=hashes,
    )

