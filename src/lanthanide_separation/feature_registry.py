"""Explicit, leakage-auditable feature families for paired models.

The pair builder exposes model columns in three historical blocks
(``baseline_columns``, ``delta3d_columns`` and ``descriptor_columns``).  Those
blocks are useful for reproducing earlier benchmarks, but they do not answer
the scientific ablation question directly.  This module assigns every model
column to exactly one of the pre-specified families:

``CONDITIONS``, ``LN``, ``2D``, ``3D_GLOBAL`` or ``3D_LOCAL``.

Local 3D columns are additionally partitioned into the deterministic D1--D5
descriptor blocks.  Classification is deliberately fail-closed: a new column
whose meaning is not covered by the explicit rules raises an error instead of
silently changing an ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .pairs import MOLECULAR_DESCRIPTOR_COLUMNS, PairDataset


FEATURE_REGISTRY_SCHEMA_VERSION = "1.0"

FEATURE_FAMILIES: tuple[str, ...] = (
    "CONDITIONS",
    "LN",
    "2D",
    "3D_GLOBAL",
    "3D_LOCAL",
)

FEATURE_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "CONDITIONS": "Experimental conditions available at prediction time.",
    "LN": "Lanthanide identities and deterministic physical descriptors.",
    "2D": "Ligand molecular descriptors and ECFP fingerprint bits.",
    "3D_GLOBAL": "Target-independent global complex properties derived from 3D structures.",
    "3D_LOCAL": "Target-independent metal-centred coordination descriptors.",
}

LOCAL_3D_SUBBLOCKS: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5")

LOCAL_3D_SUBBLOCK_NAMES: dict[str, str] = {
    "D1": "distance",
    "D2": "angular",
    "D3": "coordination_and_donor_composition",
    "D4": "shape_and_distortion",
    "D5": "steric_and_volume",
}

ABLATION_FAMILIES: dict[str, tuple[str, ...]] = {
    "A0": ("CONDITIONS",),
    "A1": ("CONDITIONS", "LN"),
    "A2": ("CONDITIONS", "LN", "2D"),
    "A3": ("CONDITIONS", "LN", "3D_LOCAL"),
    "A4": ("CONDITIONS", "LN", "2D", "3D_GLOBAL"),
    "A5": ("CONDITIONS", "LN", "2D", "3D_LOCAL"),
    "A6": FEATURE_FAMILIES,
}

# PairDataset.baseline_columns currently carries exactly these metal identity
# and deterministic physical contrasts.  Keeping an allowlist prevents a future
# pair-level target proxy from being treated as an LN feature by prefix alone.
LN_COLUMNS: frozenset[str] = frozenset(
    {
        "pair__Z_A",
        "pair__Z_B",
        "pair__Z_mean",
        "pair__delta_Z",
        "pair__ionic_radius_A",
        "pair__ionic_radius_B",
        "pair__ionic_radius_mean",
        "pair__delta_ionic_radius",
    }
)

TWO_D_DESCRIPTOR_COLUMNS: frozenset[str] = frozenset(
    f"base__{column}" for column in MOLECULAR_DESCRIPTOR_COLUMNS
)

_ECFP_PATTERN = re.compile(r"base__ecfp_[0-9]+\Z")
_THREE_D_PREFIX = "delta3d__feat3d__"
_COMPLEX_PHYSICAL_PREFIX = f"{_THREE_D_PREFIX}complex_physical__"
_POLYHEDRON_PREFIX = f"{_THREE_D_PREFIX}polyhedron__"
_POLYHEDRON_SCALARS_PREFIX = f"{_THREE_D_PREFIX}polyhedron_scalars__"
_DERIVED_INVARIANT_PREFIX = f"{_THREE_D_PREFIX}derived_invariant__"
_LIGAND_FIELD_PREFIX = f"{_THREE_D_PREFIX}ligand_field__"
_ENCLOSURE_PREFIX = f"{_THREE_D_PREFIX}enclosure__"
_COORDINATION_SHAPE_PREFIX = f"{_THREE_D_PREFIX}coordination_shape__"

_COORDINATION_SHAPE_D4_NAMES: frozenset[str] = frozenset(
    {
        "radial_distortion_coefficient",
        "radial_range_fraction",
        "metal_offset_from_donor_centroid_fraction",
        "coordination_normalized_asphericity",
        "coordination_relative_shape_anisotropy",
        "coordination_eccentricity",
        "directional_relative_shape_anisotropy",
        "directional_inversion_imbalance",
    }
)
_COORDINATION_SHAPE_D5_NAMES: frozenset[str] = frozenset(
    {
        "coordination_polyhedron_volume",
        "coordination_polyhedron_surface_area",
    }
)

# Electronic/xTB quantities can be useful predictors, but they are not
# coordinate geometry.  The A0--A6 question is deliberately geometry-specific,
# so these source-contract columns are audited and excluded from every ablation
# rather than silently relabelled as 3D geometry.
_NON_GEOMETRIC_COMPLEX_PHYSICAL_FRAGMENTS: tuple[str, ...] = (
    "energy",
    "dipole",
    "homo",
    "lumo",
    "partial_charge",
)

# Independently computed global geometry blocks may be added later without
# conflating them with metal-centred polyhedron descriptors.  The namespace
# must still be explicit; arbitrary unknown feat3d columns fail closed.
_GLOBAL_3D_PREFIXES: tuple[str, ...] = (
    f"{_THREE_D_PREFIX}global__",
    f"{_THREE_D_PREFIX}global_geometry__",
    f"{_THREE_D_PREFIX}complex_global__",
)

_FORBIDDEN_FEATURE_FRAGMENTS: tuple[str, ...] = (
    "log_d",
    "log_sf",
    "safe_exp_id",
    "build_id",
    "geometry_key",
    "sample_weight",
    "asset_index",
    "xyz_path",
)


@dataclass(frozen=True)
class FeatureAssignment:
    """One model feature and its unique scientific family assignment."""

    column: str
    family: str
    local_3d_subblock: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "column": self.column,
            "family": self.family,
            "local_3d_subblock": self.local_3d_subblock,
        }


@dataclass(frozen=True)
class FeatureRegistry:
    """Frozen, serialisable feature-family contract for one PairDataset."""

    assignments: tuple[FeatureAssignment, ...]
    excluded_non_geometric_columns: tuple[str, ...]
    source_column_count: int
    feature_contract_sha256: str
    pair_scope: str | None

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(assignment.column for assignment in self.assignments)

    @property
    def families(self) -> dict[str, tuple[str, ...]]:
        """Return family columns in the frozen PairDataset contract order."""

        return {
            family: self.columns_for_family(family) for family in FEATURE_FAMILIES
        }

    @property
    def local_3d_subblocks(self) -> dict[str, tuple[str, ...]]:
        """Return D1--D5 columns in the frozen PairDataset contract order."""

        return {
            subblock: self.columns_for_local_3d_subblock(subblock)
            for subblock in LOCAL_3D_SUBBLOCKS
        }

    def columns_for_family(self, family: str) -> tuple[str, ...]:
        if family not in FEATURE_FAMILIES:
            raise ValueError(
                f"Unknown feature family {family!r}; expected one of {FEATURE_FAMILIES}."
            )
        return tuple(
            assignment.column
            for assignment in self.assignments
            if assignment.family == family
        )

    def columns_for_local_3d_subblock(self, subblock: str) -> tuple[str, ...]:
        if subblock not in LOCAL_3D_SUBBLOCKS:
            raise ValueError(
                f"Unknown local 3D sub-block {subblock!r}; "
                f"expected one of {LOCAL_3D_SUBBLOCKS}."
            )
        return tuple(
            assignment.column
            for assignment in self.assignments
            if assignment.local_3d_subblock == subblock
        )

    def ablation_columns(self, name: str) -> tuple[str, ...]:
        """Return columns for one pre-specified A0--A6 ablation."""

        if name not in ABLATION_FAMILIES:
            raise ValueError(
                f"Unknown ablation {name!r}; expected one of "
                f"{tuple(ABLATION_FAMILIES)}."
            )
        included = set(ABLATION_FAMILIES[name])
        return tuple(
            assignment.column
            for assignment in self.assignments
            if assignment.family in included
        )

    def feature_sets(self) -> dict[str, tuple[str, ...]]:
        """Return all main ablation feature sets in canonical A0--A6 order."""

        return {
            name: self.ablation_columns(name) for name in ABLATION_FAMILIES
        }

    def local_3d_block_ablation_columns(self, subblock: str) -> tuple[str, ...]:
        """Return the exploratory ``A2 + Dn`` descriptor-block contract."""

        local_columns = set(self.columns_for_local_3d_subblock(subblock))
        baseline_columns = set(self.ablation_columns("A2"))
        included = baseline_columns | local_columns
        return tuple(column for column in self.columns if column in included)

    def local_3d_block_feature_sets(self) -> dict[str, tuple[str, ...]]:
        """Return A2+D1 ... A2+D5 and the complete A2+D1-D5 set."""

        result = {
            f"A2+{subblock}": self.local_3d_block_ablation_columns(subblock)
            for subblock in LOCAL_3D_SUBBLOCKS
        }
        result["A2+D1-D5"] = self.ablation_columns("A5")
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return the complete machine-readable ``feature_registry.json`` payload."""

        families: dict[str, dict[str, Any]] = {}
        for family in FEATURE_FAMILIES:
            columns = self.columns_for_family(family)
            families[family] = {
                "description": FEATURE_FAMILY_DESCRIPTIONS[family],
                "column_count": len(columns),
                "columns": list(columns),
            }

        local_subblocks: dict[str, dict[str, Any]] = {}
        for subblock in LOCAL_3D_SUBBLOCKS:
            columns = self.columns_for_local_3d_subblock(subblock)
            local_subblocks[subblock] = {
                "name": LOCAL_3D_SUBBLOCK_NAMES[subblock],
                "column_count": len(columns),
                "columns": list(columns),
            }

        local_columns = self.columns_for_family("3D_LOCAL")
        partitioned_local_columns = tuple(
            column
            for subblock in LOCAL_3D_SUBBLOCKS
            for column in self.columns_for_local_3d_subblock(subblock)
        )
        return {
            "schema_version": FEATURE_REGISTRY_SCHEMA_VERSION,
            "pair_scope": self.pair_scope,
            "feature_contract": {
                "source": "geometry-eligible subset of PairDataset.extended_columns",
                "column_count": len(self.assignments),
                "source_column_count": int(self.source_column_count),
                "excluded_non_geometric_column_count": len(
                    self.excluded_non_geometric_columns
                ),
                "sha256": self.feature_contract_sha256,
            },
            "excluded_from_ablation_inputs": [
                {
                    "column": column,
                    "reason": (
                        "target-independent xTB/electronic quantity; excluded so A2-vs-A5 "
                        "and A2-vs-A6 isolate coordinate geometry"
                    ),
                }
                for column in self.excluded_non_geometric_columns
            ],
            "family_order": list(FEATURE_FAMILIES),
            "families": families,
            "local_3d_subblock_order": list(LOCAL_3D_SUBBLOCKS),
            "local_3d_subblocks": local_subblocks,
            "ablation_order": list(ABLATION_FAMILIES),
            "ablations": {
                name: {
                    "families": list(families),
                    "column_count": len(self.ablation_columns(name)),
                }
                for name, families in ABLATION_FAMILIES.items()
            },
            "local_3d_block_ablations": {
                name: {"column_count": len(columns)}
                for name, columns in self.local_3d_block_feature_sets().items()
            },
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "audit": {
                "all_model_features_assigned_exactly_once": (
                    len(self.columns) == len(set(self.columns))
                ),
                "all_source_columns_accounted_for": (
                    len(self.columns) + len(self.excluded_non_geometric_columns)
                    == int(self.source_column_count)
                    and not set(self.columns) & set(self.excluded_non_geometric_columns)
                ),
                "local_3d_features_assigned_exactly_once": (
                    len(local_columns)
                    == len(partitioned_local_columns)
                    == len(set(partitioned_local_columns))
                    and set(local_columns) == set(partitioned_local_columns)
                ),
                "target_or_identifier_features_present": False,
            },
        }


def _contains_any(name: str, fragments: tuple[str, ...]) -> bool:
    return any(fragment in name for fragment in fragments)


def _classify_local_3d(column: str) -> str | None:
    """Return D1--D5 for an explicitly recognised metal-centred column."""

    if column.startswith(_COORDINATION_SHAPE_PREFIX):
        name = column.removeprefix(_COORDINATION_SHAPE_PREFIX)
        if name in _COORDINATION_SHAPE_D4_NAMES:
            return "D4"
        if name in _COORDINATION_SHAPE_D5_NAMES:
            return "D5"
        # The block is intentionally fail-closed.  Adding a new shape column
        # must include a deliberate D4/D5 registry decision.
        return None

    if column.startswith(_ENCLOSURE_PREFIX):
        # The enclosure block measures solid-angle shielding and access to the
        # metal.  Contact-distance values stay with the steric block so the
        # descriptor-block ablation does not split one physical construction.
        return "D5"

    if column.startswith(_LIGAND_FIELD_PREFIX):
        name = column.removeprefix(_LIGAND_FIELD_PREFIX)
        if name == "effective_field_radius" or name.startswith("radial_moment_"):
            return "D1"
        if name.startswith("field_strength_") or name.startswith("field_shape_"):
            return "D4"
        return None

    local_prefixes = (
        _COMPLEX_PHYSICAL_PREFIX,
        _POLYHEDRON_PREFIX,
        _POLYHEDRON_SCALARS_PREFIX,
        _DERIVED_INVARIANT_PREFIX,
    )
    prefix = next((value for value in local_prefixes if column.startswith(value)), None)
    if prefix is None:
        return None
    name = column.removeprefix(prefix).lower()

    if _contains_any(
        name,
        (
            "volume",
            "surface_area",
            "steric",
            "buried_fraction",
            "open_fraction",
            "enclosure",
        ),
    ):
        return "D5"
    if _contains_any(
        name,
        (
            "donor_angle",
            "bite_angle",
            "angle_mean",
            "angle_std",
            "angle_min",
            "angle_max",
            "angle_q",
            "angle_median",
            "angle_legendre",
        ),
    ):
        return "D2"
    if _contains_any(
        name,
        (
            "ln_donor_distance",
            "donor_dist",
            "donor_gap",
            "distance",
            "shell_clearance",
            "coordination_sphere_radius",
            "metal_from_donor_centroid",
        ),
    ):
        return "D1"
    if _contains_any(
        name,
        (
            "continuous_shape",
            "shape_measure",
            "shape_score",
            "distortion",
            "asphericity",
        ),
    ):
        return "D4"
    if _contains_any(
        name,
        (
            "coordination_number",
            "observed_donors",
            "donor_count",
            "donor_atomic_number",
            "donor_element",
            "donor_partial_charge",
            "metal_partial_charge",
        ),
    ):
        return "D3"
    return None


def _classify_column(column: str) -> FeatureAssignment:
    lowered = column.lower()
    forbidden = [
        fragment for fragment in _FORBIDDEN_FEATURE_FRAGMENTS if fragment in lowered
    ]
    if forbidden:
        raise ValueError(
            f"Forbidden target/identifier fragment(s) {forbidden} in model feature "
            f"{column!r}."
        )

    if column in LN_COLUMNS:
        return FeatureAssignment(column=column, family="LN")
    if column.startswith("base__cond__"):
        return FeatureAssignment(column=column, family="CONDITIONS")
    if column in TWO_D_DESCRIPTOR_COLUMNS or _ECFP_PATTERN.fullmatch(column):
        return FeatureAssignment(column=column, family="2D")

    if column.startswith(_GLOBAL_3D_PREFIXES):
        return FeatureAssignment(column=column, family="3D_GLOBAL")

    local_subblock = _classify_local_3d(column)
    if local_subblock is not None:
        return FeatureAssignment(
            column=column,
            family="3D_LOCAL",
            local_3d_subblock=local_subblock,
        )

    raise ValueError(
        f"Model feature {column!r} has no explicit feature-family assignment. "
        "Update the feature registry deliberately before using this column."
    )


def build_feature_registry(pair_data: PairDataset) -> FeatureRegistry:
    """Build and validate the unique family assignment for a PairDataset."""

    source_columns = tuple(str(column) for column in pair_data.extended_columns)
    duplicates = sorted(
        {column for column in source_columns if source_columns.count(column) > 1}
    )
    if duplicates:
        raise ValueError(f"PairDataset model contract contains duplicate columns: {duplicates}")

    missing_from_frame = sorted(set(source_columns) - set(pair_data.frame.columns))
    if missing_from_frame:
        raise ValueError(
            "PairDataset model columns are absent from its frame: "
            f"{missing_from_frame}"
        )

    def is_non_geometric_electronic(column: str) -> bool:
        if not column.startswith(_COMPLEX_PHYSICAL_PREFIX):
            return False
        name = column.removeprefix(_COMPLEX_PHYSICAL_PREFIX).lower()
        return any(
            fragment in name
            for fragment in _NON_GEOMETRIC_COMPLEX_PHYSICAL_FRAGMENTS
        )

    excluded_non_geometric_columns = tuple(
        column for column in source_columns if is_non_geometric_electronic(column)
    )
    excluded_non_geometric_set = set(excluded_non_geometric_columns)
    columns = tuple(
        column
        for column in source_columns
        if column not in excluded_non_geometric_set
    )
    assignments = tuple(_classify_column(column) for column in columns)
    if len(assignments) != len(columns) or len({a.column for a in assignments}) != len(columns):
        raise AssertionError("Feature registry assignment is not one-to-one.")

    local_assignments = [a for a in assignments if a.family == "3D_LOCAL"]
    if any(a.local_3d_subblock not in LOCAL_3D_SUBBLOCKS for a in local_assignments):
        raise AssertionError("A 3D_LOCAL feature has no valid D1--D5 assignment.")
    if any(
        a.local_3d_subblock is not None
        for a in assignments
        if a.family != "3D_LOCAL"
    ):
        raise AssertionError("A non-local feature was assigned a local 3D sub-block.")

    contract_payload = json.dumps(
        {
            "source_columns": list(source_columns),
            "geometry_ablation_input_columns": list(columns),
            "excluded_non_geometric_columns": list(excluded_non_geometric_columns),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw_scope = pair_data.audit.get("pair_scope")
    pair_scope = None if raw_scope is None else str(raw_scope)
    return FeatureRegistry(
        assignments=assignments,
        excluded_non_geometric_columns=excluded_non_geometric_columns,
        source_column_count=len(source_columns),
        feature_contract_sha256=hashlib.sha256(contract_payload).hexdigest(),
        pair_scope=pair_scope,
    )


def write_feature_registry_json(
    registry: FeatureRegistry,
    path: Path | str,
) -> Path:
    """Write a deterministic, human-readable ``feature_registry.json`` artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(registry.to_dict(), indent=2, ensure_ascii=False, sort_keys=False)
        + "\n",
        encoding="utf-8",
    )
    return output_path
