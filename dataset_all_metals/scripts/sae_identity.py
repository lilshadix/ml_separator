"""Experimental identity keys and the numeric tolerance policy.

An identity key is built from *scientifically meaningful* variables only.  Row
number, archive record id, DOI, citation text, entry author and source-file
order are deliberately excluded: two measurements must not count as different
science merely because the archive filed them twice.

Numeric comparison uses significant-figure quantisation rather than a chained
tolerance.  Chaining (``a~b``, ``b~c`` therefore ``a~c``) is not transitive and
would make the grouping depend on row order; quantisation is order-independent
and therefore reproducible, at the cost of a boundary effect that the tolerance
sweep in ``stage03`` quantifies.
"""

from __future__ import annotations

import hashlib
import math

# --------------------------------------------------------------------------
# Numeric field policy
# --------------------------------------------------------------------------
# significant figures kept when a value enters an identity key.
# level -> significant digits.  ``exact`` keeps the full float repr.
TOLERANCE_LEVELS: dict[str, int | None] = {
    "exact": None,   # bit-for-bit float equality
    "sig12": 12,     # removes pure float-serialisation noise only
    "sig9": 9,
    "sig6": 6,       # default: 1 ppm relative
    "sig4": 4,       # 0.01% relative
    "sig3": 3,       # 0.1% relative
}

DEFAULT_TOLERANCE = "sig6"

# Documented unit handling for every numeric field that enters an identity key.
NUMERIC_FIELD_POLICY: dict[str, dict[str, str]] = {
    "extractant_concentrations_M": {
        "original_unit": "M (string suffix ' M')", "canonical_unit": "M",
        "parser": "sae_normalize.parse_quantity(kind='concentration')",
        "conversion": "none (already molar)", "tolerance": "significant-figure quantisation",
    },
    "acid_concentration_M": {
        "original_unit": "M", "canonical_unit": "M",
        "parser": "sae_normalize.parse_quantity(kind='concentration')",
        "conversion": "none", "tolerance": "significant-figure quantisation",
    },
    "acid_concentration_organic_M": {
        "original_unit": "M", "canonical_unit": "M",
        "parser": "sae_normalize.parse_quantity(kind='concentration')",
        "conversion": "none", "tolerance": "significant-figure quantisation",
    },
    "metal_concentration_M": {
        "original_unit": "mM (column named *_mM)", "canonical_unit": "M",
        "parser": "sae_normalize.parse_quantity(kind='concentration', default_unit='mM')",
        "conversion": "x 1e-3", "tolerance": "significant-figure quantisation",
    },
    "nitrate_concentration_M": {
        "original_unit": "M (comments 'nitrate concentration(M)')", "canonical_unit": "M",
        "parser": "sae_normalize.parse_quantity(kind='concentration')",
        "conversion": "none", "tolerance": "significant-figure quantisation",
    },
    "modifier_concentration_M": {
        "original_unit": "M", "canonical_unit": "M",
        "parser": "sae_normalize.parse_quantity(kind='concentration')",
        "conversion": "none", "tolerance": "significant-figure quantisation",
    },
    "complexant_concentration_M": {
        "original_unit": "M (unitless number in comments)", "canonical_unit": "M",
        "parser": "sae_normalize.parse_quantity(kind='concentration', default_unit='M')",
        "conversion": "none", "tolerance": "significant-figure quantisation",
    },
    "holdback_concentration_M": {
        "original_unit": "M (unitless number in comments)", "canonical_unit": "M",
        "parser": "sae_normalize.parse_quantity(kind='concentration', default_unit='M')",
        "conversion": "none", "tolerance": "significant-figure quantisation",
    },
    "temperature_C": {
        "original_unit": "C (obsTempUnit is 'C' for all 45,289 filled rows)",
        "canonical_unit": "C",
        "parser": "sae_normalize.parse_quantity(kind='temperature')",
        "conversion": "K->C and F->C supported; not exercised by this export",
        "tolerance": "significant-figure quantisation",
    },
    "contact_time_min": {
        "original_unit": "min", "canonical_unit": "min",
        "parser": "sae_normalize.parse_quantity(kind='time', default_unit='min')",
        "conversion": "h x 60, s / 60", "tolerance": "significant-figure quantisation",
    },
    "shaking_time_min": {
        "original_unit": "min", "canonical_unit": "min",
        "parser": "sae_normalize.parse_quantity(kind='time', default_unit='min')",
        "conversion": "h x 60, s / 60", "tolerance": "significant-figure quantisation",
    },
    "phase_ratio_org_aq": {
        "original_unit": "dimensionless (volType='Volume Ratio')", "canonical_unit": "ratio",
        "parser": "sae_normalize.parse_quantity(kind='ratio')",
        "conversion": "none", "tolerance": "significant-figure quantisation",
    },
    "log_D": {
        "original_unit": "dimensionless D (obsDvaluesValue)", "canonical_unit": "log10(D)",
        "parser": "log10 of parsed D; non-positive D flagged, never logged",
        "conversion": "log10", "tolerance": "significant-figure quantisation",
    },
}


def quantize(value, level: str = DEFAULT_TOLERANCE):
    """Round ``value`` to the significant figures implied by ``level``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    digits = TOLERANCE_LEVELS[level]
    if digits is None:
        return repr(number)
    if number == 0.0:
        return "0"
    return f"{number:.{digits - 1}e}"


# --------------------------------------------------------------------------
# Identity key
# --------------------------------------------------------------------------
# Scientific variables only.  Anything not in this list cannot make two rows
# scientifically distinct.
IDENTITY_FIELDS_CATEGORICAL = (
    "metal_symbol",
    "metal_oxidation_state",
    "extractant_system_key",
    "acid_signature",
    "solvent_key",
    "modifier_name",
    # The full aqueous-complexant set, order-invariant and concentration-aware.
    # Keying on a single complexant's structure lost the second component and
    # its concentration, which merged genuine concentration sweeps.
    "complexant_signature",
    "holdback_smiles_canonical",
)

IDENTITY_FIELDS_NUMERIC = (
    "extractant_concentration_signature",
    "acid_concentration_M",
    "acid_concentration_organic_M",
    "metal_concentration_M",
    "nitrate_concentration_M",
    "modifier_concentration_M",
    "holdback_concentration_M",
    "temperature_C",
    "contact_time_min",
    "shaking_time_min",
    "phase_ratio_org_aq",
)

# Fields that must be present for a grouping decision to be trustworthy.
REQUIRED_FOR_DECISION = ("metal_symbol", "extractant_system_key", "log_D")


def component_signature(smiles_list, conc_list, level: str) -> str:
    """Order-invariant signature of a multi-component extractant system.

    Components are sorted by canonical SMILES so that ``"A, B"`` and ``"B, A"``
    -- which the archive uses interchangeably -- produce the same key.
    """
    def as_list(value):
        # Parquet round-trips list columns as numpy arrays, for which
        # ``value or []`` raises; test for None explicitly instead.
        if value is None:
            return []
        return list(value)

    smiles = as_list(smiles_list)
    concentrations = as_list(conc_list)
    width = max(len(smiles), len(concentrations))
    pairs = []
    for index in range(width):
        structure = smiles[index] if index < len(smiles) else None
        concentration = concentrations[index] if index < len(concentrations) else None
        pairs.append((structure if structure else "?",
                      quantize(concentration, level) or "?"))
    return ";".join(f"{s}@{c}" for s, c in sorted(pairs))


def identity_tuple(row, level: str = DEFAULT_TOLERANCE) -> tuple:
    parts = []
    for field in IDENTITY_FIELDS_CATEGORICAL:
        value = row.get(field)
        parts.append("" if value is None else str(value))
    for field in IDENTITY_FIELDS_NUMERIC:
        if field == "extractant_concentration_signature":
            parts.append(component_signature(
                row.get("extractant_smiles_canonical"),
                row.get("extractant_concentrations_M"), level))
        else:
            parts.append(str(quantize(row.get(field), level)))
    return tuple(parts)


def identity_hash(row, level: str = DEFAULT_TOLERANCE) -> str:
    blob = "\x1f".join(identity_tuple(row, level))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def value_key(row, level: str = DEFAULT_TOLERANCE) -> str:
    return str(quantize(row.get("log_D"), level))


def significant_digits(text) -> int:
    """How many significant digits the archive actually wrote down.

    Used to tell a copied value (identical to 10+ digits) from a coincidentally
    equal one (``1.0`` vs ``1.0``), which is the difference between a duplicated
    record and a possible independent replicate.
    """
    if text is None:
        return 0
    cleaned = str(text).strip().lstrip("+-")
    if not cleaned:
        return 0
    cleaned = cleaned.split("e")[0].split("E")[0]
    digits = cleaned.replace(".", "").lstrip("0")
    return len(digits.rstrip("0")) if digits else 0
