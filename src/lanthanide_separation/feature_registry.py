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
from typing import Any, Iterable

from .geometry_descriptors import BLOCK_PREFIXES
from .electronic import (
    ELECTRONIC_EVEN_PREFIX,
    ELECTRONIC_ODD_PREFIX,
    ELECTRONIC_PREFIX,
    UNAVAILABLE_ELECTRONIC_DESCRIPTORS,
)
from .pair_response import (
    PAIR_RESPONSE_EVEN_PREFIX,
    PAIR_RESPONSE_ODD_PREFIX,
    PAIR_RESPONSE_PREFIX,
    pair_response_subblock,
)
from .pairs import MOLECULAR_DESCRIPTOR_COLUMNS, PairDataset


FEATURE_REGISTRY_SCHEMA_VERSION = "1.0"

FEATURE_FAMILIES: tuple[str, ...] = (
    "CONDITIONS",
    "LN",
    "2D",
    "3D_GLOBAL",
    "3D_LOCAL",
)

# Swap-symmetric 3D partners.  They are deliberately kept out of FEATURE_FAMILIES
# so that the frozen A0--A6 ablations are bit-for-bit unchanged when the
# symmetric block is present in the frame.
SYMMETRIC_FEATURE_FAMILIES: tuple[str, ...] = ("3D_GLOBAL_SYM", "3D_LOCAL_SYM")

# Second-generation extension families.  Like the symmetric block they are kept
# out of FEATURE_FAMILIES so that A0--A6 is bit-for-bit unchanged whenever the
# extension columns happen to be present in the frame.
EXTENSION_FEATURE_FAMILIES: tuple[str, ...] = (
    "3D_PAIR_RESPONSE",
    "ELEC_COMPLEX",
    "ELEC_PAIR",
    "ELEC_ENERGY",
)

ALL_FEATURE_FAMILIES: tuple[str, ...] = (
    FEATURE_FAMILIES + SYMMETRIC_FEATURE_FAMILIES + EXTENSION_FEATURE_FAMILIES
)

FEATURE_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "CONDITIONS": "Experimental conditions available at prediction time.",
    "LN": "Lanthanide identities and deterministic physical descriptors.",
    "2D": "Ligand molecular descriptors and ECFP fingerprint bits.",
    "3D_GLOBAL": "Target-independent global complex properties derived from 3D structures.",
    "3D_LOCAL": "Target-independent metal-centred coordination descriptors.",
    "3D_GLOBAL_SYM": "Swap-symmetric mean of the global 3D block: (X_A + X_B) / 2.",
    "3D_LOCAL_SYM": "Swap-symmetric mean of the metal-centred 3D block: (X_A + X_B) / 2.",
    "3D_PAIR_RESPONSE": (
        "Declared pair-response geometry: relative, magnitude, ionic-radius "
        "normalised and excess forms of how the coordination shell answers the "
        "metal substitution."
    ),
    "ELEC_COMPLEX": (
        "Complex-level xTB electronic environment as a swap-symmetric mean of "
        "the two complexes."
    ),
    "ELEC_PAIR": (
        "Pair-response xTB electronic block: signed and unsigned charge and "
        "dipole differences and their ionic-radius normalised responses. "
        "Complete on every modelled pair."
    ),
    "ELEC_ENERGY": (
        "Composition-gated total-energy contrast, kept in its own family "
        "because it is undefined for the 45% of pairs whose two complexes "
        "differ in composition, and that missingness is itself associated with "
        "the target magnitude."
    ),
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

# Declared extension arms.  The frozen A0--A6 contract encodes 3D only as the
# antisymmetric A-B contrast, which cancels the absolute coordination
# environment (donor composition and coordination number are identically zero
# or near-zero for most pairs).  The symmetric partner ``sym3d__X`` restores
# that information while remaining invariant under the A/B swap, so exact
# prediction antisymmetry is preserved.  These arms are reported *alongside*
# A5/A6, never as replacements.
SYMMETRIC_ABLATION_FAMILIES: dict[str, tuple[str, ...]] = {
    "A5s": ("CONDITIONS", "LN", "2D", "3D_LOCAL", "3D_LOCAL_SYM"),
    "A6s": FEATURE_FAMILIES + ("3D_LOCAL_SYM", "3D_GLOBAL_SYM"),
}

# Second-generation ladder.  ``A2`` is the reference in every comparison and the
# geometry/electronic blocks are added one at a time before any combination, so
# each contrast stays interpretable.  ``G1`` is deliberately the same column set
# as the pre-specified ``A5``; it is listed as an alias rather than duplicated
# so the two ladders cannot silently diverge.
EXTENSION_ARM_ALIASES: dict[str, str] = {"G1": "A5"}

# The independently computed metal-site descriptor blocks live inside the
# 3D_LOCAL / 3D_GLOBAL families (that is where they belong scientifically), so
# the G4 arm selects them by their declared namespace rather than by family.
METAL_SITE_DESCRIPTOR_PREFIXES: tuple[str, ...] = tuple(
    f"delta3d__{prefix}" for prefix in BLOCK_PREFIXES.values()
)

EXTENSION_ABLATION_FAMILIES: dict[str, tuple[str, ...]] = {
    "G2": ("CONDITIONS", "LN", "2D", "3D_PAIR_RESPONSE"),
    "E2": ("CONDITIONS", "LN", "2D", "ELEC_COMPLEX"),
    "E3": ("CONDITIONS", "LN", "2D", "ELEC_PAIR"),
    "E4": ("CONDITIONS", "LN", "2D", "ELEC_ENERGY"),
    "C1": ("CONDITIONS", "LN", "2D", "3D_LOCAL", "3D_PAIR_RESPONSE"),
    "C2": ("CONDITIONS", "LN", "2D", "ELEC_COMPLEX", "ELEC_PAIR", "ELEC_ENERGY"),
    "C3": (
        "CONDITIONS",
        "LN",
        "2D",
        "3D_LOCAL",
        "3D_PAIR_RESPONSE",
        "ELEC_COMPLEX",
        "ELEC_PAIR",
        "ELEC_ENERGY",
    ),
}

# Blocks whose permutation control is meaningful, mapped to the arm they test.
# STEP 10 of the protocol: any block claiming metal- or complex-specific
# information must be shown to beat its own shuffled twin, not merely A2.
EXTENSION_SHUFFLE_BLOCKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "G2": ("G2", ("3D_PAIR_RESPONSE",)),
    "E3": ("E3", ("ELEC_PAIR",)),
    "E2": ("E2", ("ELEC_COMPLEX",)),
    "E4": ("E4", ("ELEC_ENERGY",)),
}

# Arms the brief asks about that cannot be built from the bundled assets.
UNAVAILABLE_EXTENSION_ARMS: dict[str, str] = {
    "E1": UNAVAILABLE_ELECTRONIC_DESCRIPTORS["ligand_level_electronic"],
}

# 2D-representation sensitivity ladder (STEP 12).  ``A2`` carries ~2 058 columns
# of which all but ~40 are ECFP bits, so a 40-column geometry or electronic
# block can be diluted -- and, worse, the fingerprint can memorise ligand
# identity outright.  These arms shrink the 2D representation and re-ask the
# same incremental question; they are strictly secondary and never replace the
# primary ``A2``-referenced claim.  ``S1``/``S2`` split the 2D family by
# namespace rather than by feature family, so the frozen ``2D`` family and the
# A0--A6 contract are untouched.
#
#   S1  RDKit molecular descriptors only (no fingerprint)
#   S2  ECFP bits only (no descriptors)
#   S3  S1 + existing local 3D          -- the S-ladder mirror of A5/G1
#   S4  S1 + pair-response 3D           -- the mirror of G2
#   S5  S1 + pair-response electronic   -- the mirror of E3
#
# Reading rule, fixed in advance: if a block helps in the S-ladder but not
# against A2, that is evidence of *redundancy with ECFP*, not evidence that the
# production model should drop the fingerprint.
SENSITIVITY_2D_ARM_FAMILIES: dict[str, tuple[str, ...]] = {
    "S3": ("3D_LOCAL",),
    "S4": ("3D_PAIR_RESPONSE",),
    "S5": ("ELEC_PAIR",),
}

SENSITIVITY_2D_ARMS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")

SENSITIVITY_2D_ARM_DESCRIPTIONS: dict[str, str] = {
    "S1": "Conditions + Ln + RDKit 2D descriptors only; ECFP bits removed.",
    "S2": "Conditions + Ln + ECFP bits only; RDKit 2D descriptors removed.",
    "S3": "S1 + the existing metal-centred local 3D block.",
    "S4": "S1 + the declared pair-response geometry block.",
    "S5": "S1 + the declared pair-response electronic block.",
}

# The blocks added on top of S1 make the same metal-/complex-specific claim as
# their A2-referenced twins, so they need the same permutation control.
SENSITIVITY_2D_SHUFFLE_BLOCKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "S4": ("S4", ("3D_PAIR_RESPONSE",)),
    "S5": ("S5", ("ELEC_PAIR",)),
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
    degenerate_columns: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(assignment.column for assignment in self.assignments)

    @property
    def families(self) -> dict[str, tuple[str, ...]]:
        """Return family columns in the frozen PairDataset contract order."""

        return {
            family: self.columns_for_family(family)
            for family in self.active_families
        }

    @property
    def local_3d_subblocks(self) -> dict[str, tuple[str, ...]]:
        """Return D1--D5 columns in the frozen PairDataset contract order."""

        return {
            subblock: self.columns_for_local_3d_subblock(subblock)
            for subblock in LOCAL_3D_SUBBLOCKS
        }

    def columns_for_family(self, family: str) -> tuple[str, ...]:
        if family not in ALL_FEATURE_FAMILIES:
            raise ValueError(
                f"Unknown feature family {family!r}; expected one of {ALL_FEATURE_FAMILIES}."
            )
        return tuple(
            assignment.column
            for assignment in self.assignments
            if assignment.family == family
        )

    @property
    def has_symmetric_3d(self) -> bool:
        """True when the frame carries the declared sym3d extension block."""

        return any(
            assignment.family in SYMMETRIC_FEATURE_FAMILIES
            for assignment in self.assignments
        )

    @property
    def present_extension_families(self) -> tuple[str, ...]:
        """Second-generation families actually carried by this cohort."""

        present = {assignment.family for assignment in self.assignments}
        return tuple(
            family for family in EXTENSION_FEATURE_FAMILIES if family in present
        )

    @property
    def active_families(self) -> tuple[str, ...]:
        """Core families, extended only by blocks actually present in the frame.

        With every extension off this is exactly ``FEATURE_FAMILIES``, so the
        frozen A0--A6 contract and its serialised form are unchanged.
        """

        families = list(FEATURE_FAMILIES)
        if self.has_symmetric_3d:
            families.extend(SYMMETRIC_FEATURE_FAMILIES)
        families.extend(self.present_extension_families)
        return tuple(families)

    def columns_for_local_3d_subblock(
        self, subblock: str, *, family: str = "3D_LOCAL"
    ) -> tuple[str, ...]:
        if subblock not in LOCAL_3D_SUBBLOCKS:
            raise ValueError(
                f"Unknown local 3D sub-block {subblock!r}; "
                f"expected one of {LOCAL_3D_SUBBLOCKS}."
            )
        # Symmetric partners carry the same D1--D5 label, so the family filter is
        # what keeps the delta partition exhaustive and non-overlapping.
        return tuple(
            assignment.column
            for assignment in self.assignments
            if assignment.local_3d_subblock == subblock and assignment.family == family
        )

    def ablation_columns(self, name: str) -> tuple[str, ...]:
        """Return columns for a pre-specified, symmetric or extension arm."""

        resolved = EXTENSION_ARM_ALIASES.get(name, name)
        if resolved == "G4":
            return self.metal_site_arm_columns()
        specification = {
            **ABLATION_FAMILIES,
            **SYMMETRIC_ABLATION_FAMILIES,
            **EXTENSION_ABLATION_FAMILIES,
        }
        if resolved not in specification:
            raise ValueError(
                f"Unknown ablation {name!r}; expected one of "
                f"{tuple(specification) + tuple(EXTENSION_ARM_ALIASES)}."
            )
        included = set(specification[resolved])
        missing = sorted(included - set(self.active_families))
        if missing:
            raise ValueError(
                f"Ablation {name!r} needs feature families {missing} that this "
                "cohort does not carry; build the pair dataset with the "
                "matching extension enabled."
            )
        return tuple(
            assignment.column
            for assignment in self.assignments
            if assignment.family in included
        )

    @property
    def metal_site_descriptor_columns(self) -> tuple[str, ...]:
        """Independently computed VR descriptor columns, in contract order."""

        return tuple(
            assignment.column
            for assignment in self.assignments
            if assignment.column.startswith(METAL_SITE_DESCRIPTOR_PREFIXES)
        )

    def metal_site_arm_columns(self) -> tuple[str, ...]:
        """``G4`` = A2 plus the metal-site descriptor block."""

        descriptors = set(self.metal_site_descriptor_columns)
        if not descriptors:
            raise ValueError(
                "G4 needs metal-site descriptor columns; build the pair dataset "
                "with geometry_descriptor_blocks."
            )
        included = set(self.ablation_columns("A2")) | descriptors
        return tuple(column for column in self.columns if column in included)

    @property
    def two_d_descriptor_columns(self) -> tuple[str, ...]:
        """RDKit molecular-descriptor half of the 2D family, in contract order."""

        return tuple(
            assignment.column
            for assignment in self.assignments
            if assignment.family == "2D"
            and assignment.column in TWO_D_DESCRIPTOR_COLUMNS
        )

    @property
    def two_d_ecfp_columns(self) -> tuple[str, ...]:
        """ECFP-bit half of the 2D family, in contract order."""

        return tuple(
            assignment.column
            for assignment in self.assignments
            if assignment.family == "2D"
            and _ECFP_PATTERN.fullmatch(assignment.column) is not None
        )

    def sensitivity_2d_arm_columns(self, name: str) -> tuple[str, ...]:
        """Return the columns of one declared 2D-representation sensitivity arm.

        The 2D family is partitioned by namespace here; every column keeps the
        family label it already carries, so nothing in the frozen A0--A6
        contract or its serialised form changes.
        """

        if name not in SENSITIVITY_2D_ARMS:
            raise ValueError(
                f"Unknown 2D sensitivity arm {name!r}; expected one of "
                f"{SENSITIVITY_2D_ARMS}."
            )
        descriptors = self.two_d_descriptor_columns
        fingerprint = self.two_d_ecfp_columns
        if not descriptors:
            raise ValueError(
                "The 2D sensitivity ladder needs RDKit descriptor columns; this "
                "cohort carries none."
            )
        if not fingerprint:
            raise ValueError(
                "The 2D sensitivity ladder needs ECFP columns; this cohort "
                "carries none."
            )
        conditions_and_ln = set(self.columns_for_families(("CONDITIONS", "LN")))
        if name == "S2":
            included = conditions_and_ln | set(fingerprint)
        else:
            included = conditions_and_ln | set(descriptors)
            for family in SENSITIVITY_2D_ARM_FAMILIES.get(name, ()):
                if family not in self.active_families:
                    raise ValueError(
                        f"Sensitivity arm {name!r} needs feature family "
                        f"{family!r} that this cohort does not carry; build the "
                        "pair dataset with the matching extension enabled."
                    )
                included |= set(self.columns_for_families((family,)))
        return tuple(column for column in self.columns if column in included)

    def sensitivity_2d_feature_sets(self) -> dict[str, tuple[str, ...]]:
        """Return every 2D-sensitivity arm this cohort can actually evaluate."""

        if not self.two_d_descriptor_columns or not self.two_d_ecfp_columns:
            return {}
        available = set(self.active_families)
        sets: dict[str, tuple[str, ...]] = {}
        for name in SENSITIVITY_2D_ARMS:
            required = set(SENSITIVITY_2D_ARM_FAMILIES.get(name, ()))
            if required <= available:
                sets[name] = self.sensitivity_2d_arm_columns(name)
        return sets

    def extension_feature_sets(self) -> dict[str, tuple[str, ...]]:
        """Return every declared G/E/C arm this cohort can actually evaluate."""

        available = set(self.active_families)
        sets = {
            name: self.ablation_columns(name)
            for name, families in EXTENSION_ABLATION_FAMILIES.items()
            if set(families) <= available
        }
        if self.metal_site_descriptor_columns:
            sets["G4"] = self.metal_site_arm_columns()
        return dict(sorted(sets.items()))

    def columns_for_families(self, families: Iterable[str]) -> tuple[str, ...]:
        """Return the contract-ordered columns of a set of families."""

        included = set(families)
        unknown = sorted(included - set(ALL_FEATURE_FAMILIES))
        if unknown:
            raise ValueError(f"Unknown feature families: {unknown}")
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

    def symmetric_feature_sets(self) -> dict[str, tuple[str, ...]]:
        """Return the declared A5s/A6s arms, or nothing when sym3d is absent."""

        if not self.has_symmetric_3d:
            return {}
        return {
            name: self.ablation_columns(name) for name in SYMMETRIC_ABLATION_FAMILIES
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
        for family in self.active_families:
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
            "degenerate_zero_variance_columns": {
                "count": len(self.degenerate_columns),
                "columns": list(self.degenerate_columns),
                "interpretation": (
                    "Constant across the whole cohort, therefore carrying no "
                    "information. Reported as unavailable so per-block feature "
                    "counts and importance normalisation are not overstated."
                ),
                "by_family": {
                    family: sorted(
                        column
                        for column in self.degenerate_columns
                        if column in set(self.columns_for_family(family))
                    )
                    for family in self.active_families
                    if any(
                        column in set(self.columns_for_family(family))
                        for column in self.degenerate_columns
                    )
                },
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
                for name, families in {
                    **ABLATION_FAMILIES,
                    **(
                        SYMMETRIC_ABLATION_FAMILIES if self.has_symmetric_3d else {}
                    ),
                    **{
                        name: EXTENSION_ABLATION_FAMILIES.get(
                            name, ("CONDITIONS", "LN", "2D", "METAL_SITE_DESCRIPTORS")
                        )
                        for name in self.extension_feature_sets()
                    },
                }.items()
            },
            "declared_symmetric_ablation_order": (
                list(SYMMETRIC_ABLATION_FAMILIES) if self.has_symmetric_3d else []
            ),
            "declared_extension_ablation_order": list(self.extension_feature_sets()),
            "extension_arm_aliases": {
                name: target
                for name, target in EXTENSION_ARM_ALIASES.items()
                if target in ABLATION_FAMILIES
            },
            "unavailable_extension_arms": [
                {"arm": arm, "reason": reason}
                for arm, reason in UNAVAILABLE_EXTENSION_ARMS.items()
            ],
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


_SYMMETRIC_PREFIX = "sym3d__"


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

    if column.startswith(PAIR_RESPONSE_PREFIX):
        if not column.startswith(
            (PAIR_RESPONSE_ODD_PREFIX, PAIR_RESPONSE_EVEN_PREFIX)
        ):
            raise ValueError(
                f"Pair-response feature {column!r} declares no swap parity; use "
                "the odd/even namespace so the A/B swap stays exact."
            )
        return FeatureAssignment(
            column=column,
            family="3D_PAIR_RESPONSE",
            local_3d_subblock=pair_response_subblock(column),
        )

    if column.startswith(ELECTRONIC_PREFIX):
        if not column.startswith((ELECTRONIC_ODD_PREFIX, ELECTRONIC_EVEN_PREFIX)):
            raise ValueError(
                f"Electronic feature {column!r} declares no swap parity; use the "
                "odd/even namespace so the A/B swap stays exact."
            )
        if column.startswith(f"{ELECTRONIC_EVEN_PREFIX}mean_"):
            family = "ELEC_COMPLEX"
        elif "total_energy" in column:
            family = "ELEC_ENERGY"
        else:
            family = "ELEC_PAIR"
        return FeatureAssignment(column=column, family=family)

    if column.startswith(_SYMMETRIC_PREFIX):
        # Classify the symmetric partner exactly like its delta counterpart, then
        # relabel the family.  A symmetric column with no delta counterpart is a
        # registry error, not something to absorb silently.
        delta_equivalent = "delta3d__" + column.removeprefix(_SYMMETRIC_PREFIX)
        partner = _classify_column(delta_equivalent)
        if partner.family not in {"3D_LOCAL", "3D_GLOBAL"}:
            raise ValueError(
                f"Symmetric feature {column!r} maps to non-3D family {partner.family!r}."
            )
        return FeatureAssignment(
            column=column,
            family=f"{partner.family}_SYM",
            local_3d_subblock=partner.local_3d_subblock,
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

    source_columns = tuple(
        str(column)
        for column in (
            pair_data.all_model_columns
            if pair_data.extension_columns
            else pair_data.extended_columns
        )
    )
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

    symmetric_complex_physical_prefix = (
        f"{_SYMMETRIC_PREFIX}{_COMPLEX_PHYSICAL_PREFIX.removeprefix('delta3d__')}"
    )

    def is_non_geometric_electronic(column: str) -> bool:
        # The symmetric block mirrors every delta column, so the xTB/electronic
        # exclusion has to recognise both namespaces or A5s/A6s would smuggle in
        # exactly the quantities A5/A6 deliberately exclude.
        for prefix in (_COMPLEX_PHYSICAL_PREFIX, symmetric_complex_physical_prefix):
            if column.startswith(prefix):
                name = column.removeprefix(prefix).lower()
                return any(
                    fragment in name
                    for fragment in _NON_GEOMETRIC_COMPLEX_PHYSICAL_FRAGMENTS
                )
        return False

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

    # A symmetric partner inherits its delta counterpart's D1--D5 label, so both
    # local families legitimately carry a sub-block.
    subblock_families = {"3D_LOCAL", "3D_LOCAL_SYM", "3D_PAIR_RESPONSE"}
    local_assignments = [a for a in assignments if a.family in subblock_families]
    if any(a.local_3d_subblock not in LOCAL_3D_SUBBLOCKS for a in local_assignments):
        raise AssertionError("A 3D_LOCAL feature has no valid D1--D5 assignment.")
    if any(
        a.local_3d_subblock is not None
        for a in assignments
        if a.family not in subblock_families
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

    # A constant column is not a weak feature, it is an unavailable one.  The
    # brief requires descriptors that cannot be computed to be declared rather
    # than silently emitted, so record them.  They are kept in the contract (a
    # tree ignores them) but reported so block sizes are not overstated.
    values = pair_data.frame.loc[:, list(columns)]
    degenerate_columns = tuple(
        column
        for column in columns
        if values[column].nunique(dropna=True) <= 1
    )

    return FeatureRegistry(
        assignments=assignments,
        excluded_non_geometric_columns=excluded_non_geometric_columns,
        source_column_count=len(source_columns),
        feature_contract_sha256=hashlib.sha256(contract_payload).hexdigest(),
        pair_scope=pair_scope,
        degenerate_columns=degenerate_columns,
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
