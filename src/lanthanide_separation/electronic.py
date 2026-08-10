"""Provenance-audited electronic and energetic descriptors.

The A0--A6 geometry ablation deliberately excludes every xTB/electronic column so
that ``A2`` versus ``A5`` isolates coordinate geometry.  This module does not
change that decision.  It declares the electronic quantities as their own
experimental hypothesis, with three hard rules:

1. **Nothing is invented.**  A descriptor that cannot be computed from the
   bundled assets is listed in :data:`UNAVAILABLE_ELECTRONIC_DESCRIPTORS` with
   the recorded reason, not silently approximated.
2. **Every emitted column has a documented equation, unit and parity** under the
   A/B swap.  Parity is encoded in the column namespace (``elec__odd__`` /
   ``elec__even__``) so the swap machinery cannot get it wrong by accident.
3. **Energies are total energies, never binding energies.**  The bundle has no
   free-ion or free-ligand references (see
   ``features/feature_blocks_manifest.json``), so no complexation energy can be
   defined.  The one energetic contrast that *is* definable is gated on exact
   composition matching between the two complexes of a pair.

Provenance of the source values: ``features/complex_physical_scalars.parquet``,
computed once per QC-accepted geometry (``geometry_key``) from the bundled
GFN2-xTB extended-XYZ payloads and joined onto rows by ``safe_exp_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


ELECTRONIC_ODD_PREFIX = "elec__odd__"
ELECTRONIC_EVEN_PREFIX = "elec__even__"
ELECTRONIC_PREFIX = "elec__"

# Composition fields that must agree between the two complexes of a pair before
# a total-energy difference can be interpreted at all.  If the two complexes do
# not contain the same ligand stoichiometry, the difference also contains the
# energy of the extra/absent fragments and is not a metal-substitution contrast.
COMPOSITION_MATCH_COLUMNS: tuple[str, ...] = (
    "coreCN",
    "n_ligs",
    "inner_sphere_anion",
    "fill_ligand",
    "n_fill",
    "metal_ox",
)

# Ionic-radius denominators below this are treated as unusable rather than
# producing an unbounded compliance ratio.
MINIMUM_RADIUS_DIFFERENCE_ANGSTROM = 1e-3


@dataclass(frozen=True)
class ElectronicSource:
    """One metal-dependent complex-level electronic quantity."""

    name: str
    column: str
    units: str
    description: str


# Metal-dependent, complex-level quantities that exist and are non-degenerate in
# this bundle.  ``feat3d__complex_physical__complex_free_energy_eV`` is
# deliberately absent: it is bit-identical to the total-energy column (verified
# in :func:`audit_electronic_sources`), so carrying both would double-count one
# measurement.
COMPLEX_ELECTRONIC_SOURCES: tuple[ElectronicSource, ...] = (
    ElectronicSource(
        "q_metal",
        "feat3d__complex_physical__metal_partial_charge",
        "e",
        "GFN2-xTB partial charge on the lanthanide centre.",
    ),
    ElectronicSource(
        "q_donor_mean",
        "feat3d__complex_physical__donor_partial_charge_mean",
        "e",
        "Mean GFN2-xTB partial charge over the selected coordinating donors.",
    ),
    ElectronicSource(
        "q_donor_std",
        "feat3d__complex_physical__donor_partial_charge_std",
        "e",
        "Population standard deviation of the donor partial charges.",
    ),
    ElectronicSource(
        "q_donor_min",
        "feat3d__complex_physical__donor_partial_charge_min",
        "e",
        "Most negative donor partial charge.",
    ),
    ElectronicSource(
        "q_donor_max",
        "feat3d__complex_physical__donor_partial_charge_max",
        "e",
        "Least negative donor partial charge.",
    ),
    ElectronicSource(
        "dipole_magnitude",
        "feat3d__complex_physical__dipole_magnitude",
        "debye",
        "Norm of the complex dipole vector; the frame-dependent components "
        "dipole_x/y/z are never used.",
    ),
)

# Derived from the sources above rather than read from the bundle.
DERIVED_COMPLEX_ELECTRONIC_NAME = "metal_minus_donor_charge"
DERIVED_COMPLEX_ELECTRONIC_DESCRIPTION = (
    "q_metal - q_donor_mean: the metal/donor charge separation, i.e. how "
    "ionic the first coordination shell is in the GFN2-xTB population analysis."
)

ENERGY_SOURCE = ElectronicSource(
    "complex_total_energy",
    "feat3d__complex_physical__complex_total_energy_eV",
    "eV",
    "GFN2-xTB total energy of the whole complex as bundled.",
)

ENERGY_EQUATION = (
    "elec__odd__delta_complex_total_energy_eV = "
    "E_total(complex with metal A) - E_total(complex with metal B), "
    "emitted only when both complexes share coreCN, n_ligs, "
    "inner_sphere_anion, fill_ligand, n_fill and metal_ox."
)
ENERGY_ASSUMPTIONS = (
    "This is a total-energy difference, NOT a binding, complexation or "
    "interaction energy: the bundle contains no free-ion and no free-ligand "
    "reference calculation, so no thermodynamic cycle can be closed. The "
    "difference retains an unknown additive constant per metal pair (the "
    "free-ion reference difference). That constant is bounded in this cohort: "
    "the between-metal-pair spread of mean dE is smaller than the within-pair "
    "ligand-dependent spread, so the column is not merely a metal-pair label. "
    "It is still redundant with the LN family by construction and must not be "
    "read as a stability constant."
)

# Descriptors the brief asks about that genuinely cannot be built here.  The
# reasons are quoted from features/feature_blocks_manifest.json, which lists
# them under "uncomputed_null_columns".
UNAVAILABLE_ELECTRONIC_DESCRIPTORS: dict[str, str] = {
    "homo_eV": (
        "All-null in the bundle; the manifest records that the existing extxyz "
        "payloads lack individual frontier-orbital energies."
    ),
    "lumo_eV": "All-null in the bundle; same reason as homo_eV.",
    "homo_lumo_gap_eV": "All-null in the bundle; requires homo_eV and lumo_eV.",
    "binding_energy_eV": (
        "All-null in the bundle; requires free-ion, free-ligand and free-fill "
        "reference xTB energies that were queued "
        "(features/xtb_reference_calculation_queue.csv, calculation_status="
        "not_run_requires_reference_xtb) but never run."
    ),
    "strain_energy_eV": (
        "All-null in the bundle; requires a free-ligand reference geometry and "
        "single point at the bound conformation."
    ),
    "geometry_xtb_energy_eV": "Row-level column present but all-null in the bundle.",
    "ligand_level_electronic": (
        "No free-ligand electronic calculation exists in the bundle. Every "
        "available electronic quantity is a property of the metal complex, so "
        "a ligand-only electronic block (ligand dipole, ligand HOMO/LUMO, "
        "ligand partial charges) cannot be constructed. The E1 arm is "
        "therefore declared unavailable rather than filled with complex-level "
        "values relabelled as ligand-level."
    ),
    "solvation_energy": (
        "No implicit-solvent single points are bundled and the extraction "
        "conditions are not encoded in the geometry assets."
    ),
    "interaction_energy": (
        "Requires fragment references (ion plus ligands at the complex "
        "geometry); see binding_energy_eV."
    ),
}


def electronic_source_columns() -> tuple[str, ...]:
    """Every dataset column this module reads."""

    return tuple(source.column for source in COMPLEX_ELECTRONIC_SOURCES) + (
        ENERGY_SOURCE.column,
    )


def audit_electronic_sources(frame: pd.DataFrame) -> dict[str, Any]:
    """Trace the provenance and usability of every electronic source column.

    The audit never trusts a column merely because it exists: it records
    non-null counts, distinct-value counts, whether the values actually vary
    with the metal inside one extractant, and whether two nominally different
    columns are bit-identical duplicates.
    """

    available: dict[str, Any] = {}
    for source in (*COMPLEX_ELECTRONIC_SOURCES, ENERGY_SOURCE):
        if source.column not in frame.columns:
            available[source.name] = {
                "column": source.column,
                "status": "absent_from_dataset",
            }
            continue
        values = pd.to_numeric(frame[source.column], errors="coerce")
        geometry_ok = (
            frame["geometry_ok"].fillna(False).astype(bool)
            if "geometry_ok" in frame.columns
            else pd.Series(True, index=frame.index)
        )
        varies_with_metal = None
        if {"canonical_smiles", "metal"}.issubset(frame.columns):
            grouped = values.groupby(
                [frame["canonical_smiles"], frame["metal"]], dropna=False
            ).mean()
            per_extractant = grouped.groupby(level=0).nunique()
            varies_with_metal = int((per_extractant > 1).sum())
        available[source.name] = {
            "column": source.column,
            "units": source.units,
            "description": source.description,
            "status": "available" if values.notna().any() else "all_null",
            "non_null_rows": int(values.notna().sum()),
            "non_null_rows_with_accepted_geometry": int(
                values[geometry_ok].notna().sum()
            ),
            "distinct_values": int(values.nunique(dropna=True)),
            "extractants_where_value_varies_across_metals": varies_with_metal,
        }

    duplicates: list[dict[str, str]] = []
    free_energy = "feat3d__complex_physical__complex_free_energy_eV"
    if {ENERGY_SOURCE.column, free_energy}.issubset(frame.columns):
        left = pd.to_numeric(frame[ENERGY_SOURCE.column], errors="coerce")
        right = pd.to_numeric(frame[free_energy], errors="coerce")
        both = left.notna() & right.notna()
        if bool(both.any()) and bool((left[both] == right[both]).all()):
            duplicates.append(
                {
                    "columns": f"{ENERGY_SOURCE.column} == {free_energy}",
                    "finding": (
                        "bit-identical on every row where both are observed; the "
                        "bundle reports one energy under two names, so only the "
                        "total-energy column is used"
                    ),
                }
            )

    unavailable = {
        name: {
            "reason": reason,
            "all_null_in_dataset": bool(
                name in frame.columns
                and not pd.to_numeric(frame[name], errors="coerce").notna().any()
            )
            or any(
                column.endswith(name)
                and not pd.to_numeric(frame[column], errors="coerce").notna().any()
                for column in frame.columns
                if column.endswith(name)
            ),
        }
        for name, reason in UNAVAILABLE_ELECTRONIC_DESCRIPTORS.items()
    }

    return {
        "provenance": {
            "source_asset": "features/complex_physical_scalars.parquet",
            "method": "GFN2-xTB scalars bundled with each QC-accepted geometry",
            "join_keys": ["safe_exp_id", "geometry_key"],
            "note": (
                "Values were not recomputed here. They are used exactly as "
                "bundled and are audited, not trusted a priori."
            ),
        },
        "available_sources": available,
        "duplicate_columns": duplicates,
        "unavailable_descriptors": unavailable,
        "energy_definition": {
            "equation": ENERGY_EQUATION,
            "units": "eV",
            "assumptions": ENERGY_ASSUMPTIONS,
            "composition_match_columns": list(COMPOSITION_MATCH_COLUMNS),
        },
    }


def pair_electronic_features(
    *,
    values_a: dict[str, float],
    values_b: dict[str, float],
    radius_difference: float,
    composition_matches: bool,
) -> dict[str, float]:
    """Return one pair's electronic block with parity encoded in the names.

    ``elec__odd__X`` changes sign when A and B are swapped; ``elec__even__X``
    does not.  :func:`lanthanide_separation.pairs.reverse_pair_features` relies
    on exactly this namespace split, and the registry refuses any ``elec__``
    column that declares neither parity.
    """

    record: dict[str, float] = {}
    names = [source.name for source in COMPLEX_ELECTRONIC_SOURCES]
    names.append(DERIVED_COMPLEX_ELECTRONIC_NAME)

    usable_radius = abs(float(radius_difference)) >= MINIMUM_RADIUS_DIFFERENCE_ANGSTROM
    for name in names:
        value_a = values_a.get(name, np.nan)
        value_b = values_b.get(name, np.nan)
        observed = bool(np.isfinite(value_a) and np.isfinite(value_b))
        difference = float(value_a - value_b) if observed else np.nan
        # Complex-level absolute environment (E2): symmetric mean.
        record[f"{ELECTRONIC_EVEN_PREFIX}mean_{name}"] = (
            float((value_a + value_b) / 2.0) if observed else np.nan
        )
        # Pair response (E3): signed difference, its magnitude, and the
        # response per unit lanthanide-contraction step.
        record[f"{ELECTRONIC_ODD_PREFIX}delta_{name}"] = difference
        record[f"{ELECTRONIC_EVEN_PREFIX}absdelta_{name}"] = (
            abs(difference) if observed else np.nan
        )
        record[f"{ELECTRONIC_EVEN_PREFIX}compliance_{name}"] = (
            float(difference / radius_difference)
            if observed and usable_radius
            else np.nan
        )

    energy_a = values_a.get(ENERGY_SOURCE.name, np.nan)
    energy_b = values_b.get(ENERGY_SOURCE.name, np.nan)
    energy_observed = bool(
        composition_matches and np.isfinite(energy_a) and np.isfinite(energy_b)
    )
    record[f"{ELECTRONIC_ODD_PREFIX}delta_complex_total_energy_eV"] = (
        float(energy_a - energy_b) if energy_observed else np.nan
    )
    return record


def electronic_column_names() -> tuple[str, ...]:
    """The exact, ordered electronic contract emitted per pair."""

    probe = pair_electronic_features(
        values_a={},
        values_b={},
        radius_difference=1.0,
        composition_matches=False,
    )
    return tuple(probe)


def split_electronic_columns(
    columns: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition an electronic block into (complex-level, pair-response).

    ``ELEC_COMPLEX`` is the swap-symmetric mean block -- the absolute electronic
    environment of the two complexes.  ``ELEC_PAIR`` is everything describing
    how that environment *responds* to the metal substitution.
    """

    complex_level: list[str] = []
    pair_level: list[str] = []
    for column in columns:
        name = str(column)
        if not name.startswith(ELECTRONIC_PREFIX):
            raise ValueError(f"Not an electronic column: {name!r}")
        if name.startswith(f"{ELECTRONIC_EVEN_PREFIX}mean_"):
            complex_level.append(name)
        else:
            pair_level.append(name)
    return tuple(complex_level), tuple(pair_level)


def energy_column_name() -> str:
    """The single composition-gated energetic contrast this bundle supports."""

    return f"{ELECTRONIC_ODD_PREFIX}delta_complex_total_energy_eV"
