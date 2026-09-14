"""Chemistry reference tables for the multi-metal SAE dataset.

Everything in this module is *reference data*, not data derived from the archive.
It is kept separate from the parsers so that a reviewer can audit the chemistry
in one place.

Design rule followed throughout: a value is only recorded when it can be stated
with high confidence from a standard source.  Anything uncertain is left ``None``
and surfaced in the manual-review queue rather than guessed, because a silently
wrong ionic radius or oxidation state would propagate into every downstream
model without ever raising an error.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Element identity
# --------------------------------------------------------------------------

ATOMIC_NUMBER: dict[str, int] = {
    "Sc": 21, "Cr": 24, "Fe": 26, "Sr": 38, "Y": 39, "Zr": 40, "Mo": 42,
    "Tc": 43, "Ru": 44, "Pd": 46, "Cd": 48, "In": 49, "Ba": 56, "Ca": 20,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63,
    "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Pb": 82, "Bi": 83, "Th": 90, "Pa": 91, "U": 92,
    "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Cf": 98,
}

# The 15 elements La..Lu.  Y and Sc are chemically "rare earths" but are NOT
# lanthanides; gen10's ``lanthanide_index`` covers La..Lu only, so we keep the
# same definition to stay comparable with the frozen benchmark.
LANTHANIDES: tuple[str, ...] = (
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
)

ACTINIDES: tuple[str, ...] = ("Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Cf")

# gen10 defines lanthanide_index = Z - 56 (La -> 1 ... Lu -> 15).  Verified
# against "dataset with 3D structures/dataset.parquet".
LANTHANIDE_INDEX_OFFSET = 56


def lanthanide_index(symbol: str) -> int | None:
    if symbol in LANTHANIDES:
        return ATOMIC_NUMBER[symbol] - LANTHANIDE_INDEX_OFFSET
    return None


def metal_category(symbol: str) -> str:
    if symbol in LANTHANIDES:
        return "lanthanide"
    if symbol in ACTINIDES:
        return "actinide"
    if symbol in ("Y", "Sc"):
        return "rare_earth_non_lanthanide"
    if symbol in ("Ca", "Sr", "Ba"):
        return "alkaline_earth"
    if symbol in ("Cr", "Fe", "Zr", "Mo", "Tc", "Ru", "Pd", "Hf"):
        return "transition_metal"
    if symbol in ("Cd", "In", "Pb", "Bi"):
        return "post_transition_metal"
    return "other"


# --------------------------------------------------------------------------
# Oxidation state
# --------------------------------------------------------------------------

ROMAN_TO_INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}

# Oxidation states that are chemically accessible in aqueous solvent-extraction
# systems.  Used ONLY to flag impossible combinations for review -- never to
# overwrite what the archive recorded.
PLAUSIBLE_OX: dict[str, frozenset[int]] = {
    "Ca": frozenset({2}), "Sr": frozenset({2}), "Ba": frozenset({2}),
    "Sc": frozenset({3}), "Y": frozenset({3}), "In": frozenset({3}),
    "Bi": frozenset({3, 5}), "Cd": frozenset({2}), "Pb": frozenset({2, 4}),
    "Zr": frozenset({4}), "Hf": frozenset({4}),
    "Cr": frozenset({2, 3, 6}), "Fe": frozenset({2, 3}),
    "Mo": frozenset({3, 4, 5, 6}), "Tc": frozenset({4, 5, 7}),
    "Ru": frozenset({2, 3, 4, 6, 8}), "Pd": frozenset({2, 4}),
    "Th": frozenset({4}), "Pa": frozenset({4, 5}),
    "U": frozenset({3, 4, 5, 6}), "Np": frozenset({3, 4, 5, 6, 7}),
    "Pu": frozenset({3, 4, 5, 6}), "Am": frozenset({3, 4, 5, 6}),
    "Cm": frozenset({3, 4}), "Cf": frozenset({2, 3, 4}),
}
# Every lanthanide is accessible as +3; Ce also +4, Eu/Sm/Yb also +2.
for _ln in LANTHANIDES:
    PLAUSIBLE_OX[_ln] = frozenset({3})
PLAUSIBLE_OX["Ce"] = frozenset({3, 4})
PLAUSIBLE_OX["Eu"] = frozenset({2, 3})
PLAUSIBLE_OX["Sm"] = frozenset({2, 3})
PLAUSIBLE_OX["Yb"] = frozenset({2, 3})

# Metal tokens in the archive that are species, not bare elements.
# UO2(2+) (uranyl) is by definition uranium(VI), so recording the oxidation
# state here is a definition, not an inference.
SPECIES_TOKENS: dict[str, tuple[str, int, str]] = {
    "UO2+2": ("U", 6, "uranyl_UO2_2plus"),
}


# --------------------------------------------------------------------------
# Ionic radii
# --------------------------------------------------------------------------
# Shannon (1976) effective ionic radii, coordination number 8, in angstrom.
# CN = 8 is used because that is what the frozen gen10 table uses: its Ln(III)
# values reproduce these numbers exactly (La 1.160 ... Lu 0.977).
#
# Only (element, oxidation state) pairs that Shannon tabulates at CN = 8 and
# that we could confirm are listed.  Everything else is deliberately absent so
# that it shows up as "requires curation" instead of as a plausible-looking
# number.
#
# Present for the actinides: Th(IV), Pa(V), U(IV), U(VI), Np(IV), Pu(IV) and
# Am(III).  Am(III) is included because Shannon tabulates it at CN = 8; the
# heavier trans-americium ions (Cm, Cf) are NOT, and neither are the higher
# actinide oxidation states Np(V), Np(VI), Pu(III), Pu(VI) or Pa(IV).
# Also absent: the square-planar and highly charged d-block ions Pd(II),
# Tc(VII), Mo, Ru and Cr, none of which Shannon gives at CN = 8.
SHANNON_CN8: dict[tuple[str, int], float] = {
    ("La", 3): 1.160, ("Ce", 3): 1.143, ("Pr", 3): 1.126, ("Nd", 3): 1.109,
    ("Pm", 3): 1.093, ("Sm", 3): 1.079, ("Eu", 3): 1.066, ("Gd", 3): 1.053,
    ("Tb", 3): 1.040, ("Dy", 3): 1.027, ("Ho", 3): 1.015, ("Er", 3): 1.004,
    ("Tm", 3): 0.994, ("Yb", 3): 0.985, ("Lu", 3): 0.977,
    ("Ce", 4): 0.970,
    ("Y", 3): 1.019, ("Sc", 3): 0.870,
    ("Ca", 2): 1.120, ("Sr", 2): 1.260, ("Ba", 2): 1.420,
    ("Cd", 2): 1.100, ("Pb", 2): 1.290, ("In", 3): 0.920, ("Bi", 3): 1.170,
    ("Zr", 4): 0.840, ("Hf", 4): 0.830,
    ("Th", 4): 1.050, ("U", 4): 1.000, ("U", 6): 0.860,
    ("Np", 4): 0.980, ("Pu", 4): 0.960, ("Am", 3): 1.090, ("Pa", 5): 0.910,
}

IONIC_RADIUS_SOURCE = "Shannon (1976) effective ionic radii, CN=8"


def ionic_radius_cn8(symbol: str, ox: int | None) -> tuple[float | None, str]:
    """Return ``(radius_or_None, status)``.

    ``status`` is one of ``ok`` / ``unknown_oxidation_state`` /
    ``requires_curation`` so the audit can count exactly how many rows a
    radius-dependent model would silently drop.
    """
    if ox is None:
        return None, "unknown_oxidation_state"
    value = SHANNON_CN8.get((symbol, ox))
    if value is None:
        return None, "requires_curation"
    return value, "ok"


# --------------------------------------------------------------------------
# Extractant structures absent from the archive
# --------------------------------------------------------------------------
# TBP and DHOA carry no SMILES in any of their single-component records, so an
# earlier draft of this module hard-coded them from external sources.  That is
# no longer needed and the table has been removed: both structures are now
# recovered from the archive's own two-component records by elimination
# (``stage02_normalize.deduce_by_elimination``), which is stronger evidence and
# leaves no second, contradictory provenance story in the codebase.
#
# The deduction is only as reliable as the archive's SMILES column: it faithfully
# reproduces whatever the archive recorded, including one structure known to be
# wrong (``Br-Cosan``, which the archive gives as a chloroacetanilide rather than
# a cobalt bis(dicarbollide)).  That case is preserved as recorded and raised in
# the manual-review queue rather than silently corrected.
