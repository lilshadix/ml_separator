"""Explicit pair-response geometry: how one cavity answers a metal swap.

The frozen ``delta3d__X = X_A - X_B`` contract already encodes a difference, but
only a raw one.  The target is a *difference*, and the physically interesting
quantity is not how much a distance changed in absolute terms -- it is how much
it changed **relative to the change the free ion itself underwent**.  A rigid,
preorganised cavity resists the lanthanide contraction and its donors move less
than the ionic radius does; a floppy one tracks it.  That contrast is what
separates two metals with the same ligand, and no column in A0--A6 expresses it.

Four transforms are declared for each base geometric quantity ``g``:

======================  ==================================  ======  ==============
column                  definition                          parity  available for
======================  ==================================  ======  ==============
``odd__delta_g``        ``g_A - g_B``                       odd     all
``odd__reldelta_g``     ``(g_A-g_B) / mean(|g_A|,|g_B|)``   odd     all
``even__absdelta_g``    ``|g_A - g_B|``                     even    all
``even__compliance_g``  ``(g_A-g_B) / (r_A - r_B)``         even    all
``odd__excess_g``       ``(g_A-g_B) - (r_A - r_B)``         odd     lengths only
======================  ==================================  ======  ==============

``r`` is the tabulated Shannon ionic radius already carried by the LN family, so
no fitted quantity enters.  ``excess`` is dimensionally meaningful only when
``g`` is a length, and is emitted only for those quantities.  ``compliance`` is
even because it is a ratio of two odd quantities, and it is the descriptor that
directly measures cavity rigidity: a value near 1 means the shell follows the
ion exactly, near 0 means the cavity is rigid.

Parity is carried in the column namespace so that the A/B swap machinery cannot
silently mis-handle a new column: ``pair3d__odd__*`` is negated on swap,
``pair3d__even__*`` is left untouched, and anything else under ``pair3d__``
fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PAIR_RESPONSE_ODD_PREFIX = "pair3d__odd__"
PAIR_RESPONSE_EVEN_PREFIX = "pair3d__even__"
PAIR_RESPONSE_PREFIX = "pair3d__"

MINIMUM_RADIUS_DIFFERENCE_ANGSTROM = 1e-3
MINIMUM_RELATIVE_SCALE = 1e-9


@dataclass(frozen=True)
class PairResponseQuantity:
    """One metal-dependent geometric quantity and its declared dimension."""

    name: str
    column: str
    dimension: str
    subblock: str
    description: str

    @property
    def is_length(self) -> bool:
        return self.dimension == "angstrom"


# The declared core set.  It is deliberately small: the cohort has 34 extractants
# and a Kish effective group count near five, so a wide block buys variance, not
# resolution.  Every entry is a permutation-invariant, frame-independent
# metal-centred quantity that already exists in the compact Delta3D contract.
PAIR_RESPONSE_QUANTITIES: tuple[PairResponseQuantity, ...] = (
    PairResponseQuantity(
        "ln_donor_distance_mean",
        "feat3d__complex_physical__ln_donor_distance_mean",
        "angstrom",
        "D1",
        "Mean Ln-donor bond length over the selected coordination shell.",
    ),
    PairResponseQuantity(
        "ln_donor_distance_min",
        "feat3d__complex_physical__ln_donor_distance_min",
        "angstrom",
        "D1",
        "Shortest Ln-donor bond length.",
    ),
    PairResponseQuantity(
        "ln_donor_distance_max",
        "feat3d__complex_physical__ln_donor_distance_max",
        "angstrom",
        "D1",
        "Longest Ln-donor bond length.",
    ),
    PairResponseQuantity(
        "ln_donor_distance_std",
        "feat3d__complex_physical__ln_donor_distance_std",
        "angstrom",
        "D1",
        "Dispersion of the Ln-donor bond lengths.",
    ),
    PairResponseQuantity(
        "shell_clearance_mean",
        "feat3d__derived_invariant__shell_clearance_mean",
        "angstrom",
        "D1",
        "Mean Ln-donor distance minus the tabulated ionic radius.",
    ),
    PairResponseQuantity(
        "shell_distance_span",
        "feat3d__derived_invariant__shell_distance_span",
        "angstrom",
        "D1",
        "Longest minus shortest Ln-donor distance.",
    ),
    PairResponseQuantity(
        "coordination_number",
        "feat3d__complex_physical__coordination_number",
        "count",
        "D3",
        "Number of donors assigned to the first coordination sphere.",
    ),
    PairResponseQuantity(
        "donor_angle_mean_deg",
        "feat3d__derived_invariant__donor_angle_mean_deg",
        "degree",
        "D2",
        "Mean donor-Ln-donor angle over unordered donor pairs.",
    ),
    PairResponseQuantity(
        "donor_angle_std_deg",
        "feat3d__derived_invariant__donor_angle_std_deg",
        "degree",
        "D2",
        "Dispersion of the donor-Ln-donor angles.",
    ),
    PairResponseQuantity(
        "donor_angle_legendre_p2_mean",
        "feat3d__derived_invariant__donor_angle_legendre_p2_mean",
        "dimensionless",
        "D2",
        "Second Legendre moment of the donor-Ln-donor angle distribution; the "
        "leading crystal-field shape term in a point-charge treatment.",
    ),
)


def pair_response_source_columns() -> tuple[str, ...]:
    """Every ``feat3d__`` column this block reads."""

    return tuple(quantity.column for quantity in PAIR_RESPONSE_QUANTITIES)


def pair_response_features(
    *,
    values_a: dict[str, float],
    values_b: dict[str, float],
    radius_difference: float,
) -> dict[str, float]:
    """Return one pair's declared pair-response geometry block."""

    record: dict[str, float] = {}
    usable_radius = abs(float(radius_difference)) >= MINIMUM_RADIUS_DIFFERENCE_ANGSTROM
    for quantity in PAIR_RESPONSE_QUANTITIES:
        value_a = values_a.get(quantity.name, np.nan)
        value_b = values_b.get(quantity.name, np.nan)
        observed = bool(np.isfinite(value_a) and np.isfinite(value_b))
        difference = float(value_a - value_b) if observed else np.nan
        scale = (abs(float(value_a)) + abs(float(value_b))) / 2.0 if observed else np.nan

        record[f"{PAIR_RESPONSE_ODD_PREFIX}delta_{quantity.name}"] = difference
        record[f"{PAIR_RESPONSE_ODD_PREFIX}reldelta_{quantity.name}"] = (
            float(difference / scale)
            if observed and np.isfinite(scale) and scale > MINIMUM_RELATIVE_SCALE
            else np.nan
        )
        record[f"{PAIR_RESPONSE_EVEN_PREFIX}absdelta_{quantity.name}"] = (
            abs(difference) if observed else np.nan
        )
        record[f"{PAIR_RESPONSE_EVEN_PREFIX}compliance_{quantity.name}"] = (
            float(difference / radius_difference)
            if observed and usable_radius
            else np.nan
        )
        if quantity.is_length:
            record[f"{PAIR_RESPONSE_ODD_PREFIX}excess_{quantity.name}"] = (
                float(difference - float(radius_difference)) if observed else np.nan
            )
    return record


def pair_response_column_names() -> tuple[str, ...]:
    """The exact, ordered pair-response contract emitted per pair."""

    probe = pair_response_features(
        values_a={},
        values_b={},
        radius_difference=1.0,
    )
    return tuple(probe)


def pair_response_subblock(column: str) -> str:
    """Return the D1--D5 sub-block the column inherits from its base quantity."""

    name = str(column)
    for prefix in (PAIR_RESPONSE_ODD_PREFIX, PAIR_RESPONSE_EVEN_PREFIX):
        if not name.startswith(prefix):
            continue
        remainder = name.removeprefix(prefix)
        for quantity in PAIR_RESPONSE_QUANTITIES:
            if remainder.endswith(f"_{quantity.name}"):
                return quantity.subblock
    raise ValueError(f"No declared pair-response quantity behind {column!r}.")
