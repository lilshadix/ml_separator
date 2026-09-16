"""``chemistry/metals.py`` -- metal identity, alias normalisation and the metal descriptor table.

Brief section 4.1 (metal representation), 15 (shared + series + oxidation + element structure) and
27 ("descriptor consistency", "metal alias normalization").

A *metal* in gen19 is an **element + oxidation state** (``Nd(III)``, ``U(VI)``), never a bare
symbol: the archive holds U(IV) and U(VI), Pu(III)/(IV)/(VI), Np(IV)/(V)/(VI) and Am(III)/(VI), and
a leave-metal-out fold that grouped on the symbol alone would silently mix different chemistry.

Everything in this module is **reference data or definition**, never derived from ``log_D``.
Design rules:

* every value column ``c`` of :func:`build_descriptor_table` has a companion ``c_source``;
  a missing value carries ``"NA_REASON: ..."`` in its source column instead of a citation;
* a literature value is only written when it could be stated with confidence (see
  ``descriptors/metals_sources.md`` for the verification status of every column);
* rows whose oxidation state the archive did not record get the element-level columns only
  (:data:`ELEMENT_LEVEL_COLUMNS`); ion-level columns stay NA.

Public API: :func:`normalize_metal`, :func:`parse_oxidation_state`, :func:`metal_state_label`,
:func:`default_species_form`, :func:`ion_configuration`, :func:`build_descriptor_table`,
:func:`series_keys`.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------------------------- #
# Element identity (definitional)
# --------------------------------------------------------------------------------------------- #

#: All 118 element symbols in atomic-number order (index + 1 = Z).  Used only to decide whether a
#: label names a real element; descriptor rows exist only for :data:`ELEMENTS`.
PERIODIC_SYMBOLS: tuple[str, ...] = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
)
ATOMIC_NUMBER: dict[str, int] = {s: i + 1 for i, s in enumerate(PERIODIC_SYMBOLS)}

LANTHANIDES: tuple[str, ...] = ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
                                "Er", "Tm", "Yb", "Lu")
ACTINIDES: tuple[str, ...] = ("Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
                              "Md", "No", "Lr")
#: An(III) rows added for series smoothness: Ac..Cf where Shannon (1976) tabulates M(3+) (Th(III)
#: is not tabulated and is not an aqueous state, so it is left out).
AN3_SERIES: tuple[str, ...] = ("Ac", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf")

#: symbol -> (name, period, group or None, block, category).  Group is None for the f-block by
#: gen19 convention (the task specification); La, Lu and Ac are labelled block "f" as series
#: members (same convention as gen11 ``metalrep.PERIODIC_TABLE``).
_ELEMENT_ROWS: dict[str, tuple[str, int, int | None, str, str]] = {
    "Ca": ("calcium", 4, 2, "s", "alkaline_earth"),
    "Sc": ("scandium", 4, 3, "d", "rare_earth_non_lanthanide"),
    "Cr": ("chromium", 4, 6, "d", "transition_metal"),
    "Fe": ("iron", 4, 8, "d", "transition_metal"),
    "Sr": ("strontium", 5, 2, "s", "alkaline_earth"),
    "Y": ("yttrium", 5, 3, "d", "rare_earth_non_lanthanide"),
    "Zr": ("zirconium", 5, 4, "d", "transition_metal"),
    "Mo": ("molybdenum", 5, 6, "d", "transition_metal"),
    "Tc": ("technetium", 5, 7, "d", "transition_metal"),
    "Ru": ("ruthenium", 5, 8, "d", "transition_metal"),
    "Pd": ("palladium", 5, 10, "d", "transition_metal"),
    "Cd": ("cadmium", 5, 12, "d", "post_transition_metal"),
    "In": ("indium", 5, 13, "p", "post_transition_metal"),
    "Ba": ("barium", 6, 2, "s", "alkaline_earth"),
    "Hf": ("hafnium", 6, 4, "d", "transition_metal"),
    "Pb": ("lead", 6, 14, "p", "post_transition_metal"),
    "Bi": ("bismuth", 6, 15, "p", "post_transition_metal"),
}
for _s, _n in zip(LANTHANIDES, ("lanthanum", "cerium", "praseodymium", "neodymium", "promethium",
                                "samarium", "europium", "gadolinium", "terbium", "dysprosium",
                                "holmium", "erbium", "thulium", "ytterbium", "lutetium")):
    _ELEMENT_ROWS[_s] = (_n, 6, None, "f", "lanthanide")
for _s, _n in zip(ACTINIDES[:10], ("actinium", "thorium", "protactinium", "uranium", "neptunium",
                                   "plutonium", "americium", "curium", "berkelium", "californium")):
    _ELEMENT_ROWS[_s] = (_n, 7, None, "f", "actinide")

#: Elements that may receive descriptor rows.
ELEMENTS: tuple[str, ...] = tuple(sorted(_ELEMENT_ROWS, key=lambda s: ATOMIC_NUMBER[s]))
ELEMENT_NAMES: dict[str, str] = {s: v[0] for s, v in _ELEMENT_ROWS.items()}
_NAME_TO_SYMBOL: dict[str, str] = {v: k for k, v in ELEMENT_NAMES.items()}

CATEGORIES: tuple[str, ...] = ("lanthanide", "actinide", "transition_metal", "post_transition_metal",
                               "alkaline_earth", "rare_earth_non_lanthanide")

# --------------------------------------------------------------------------------------------- #
# Oxidation states
# --------------------------------------------------------------------------------------------- #

ROMAN: dict[int, str] = {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII",
                         8: "VIII"}
_ROMAN_TO_INT: dict[str, int] = {v: k for k, v in ROMAN.items() if k > 0}

#: Oxidation states accessible in aqueous solvent-extraction media.  Copied from the archive build
#: (``dataset_all_metals/scripts/sae_chem.py`` ``PLAUSIBLE_OX``) so gen19 flags exactly what the
#: archive flags (asserted by ``scripts/g19_build_metals.py``); Ac {3} and Bk {3, 4} added for the
#: An(III) series rows.  Used only to label, never to rewrite a recorded state.
ACCESSIBLE_STATES: dict[str, frozenset[int]] = {
    "Ca": frozenset({2}), "Sr": frozenset({2}), "Ba": frozenset({2}),
    "Sc": frozenset({3}), "Y": frozenset({3}), "In": frozenset({3}),
    "Bi": frozenset({3, 5}), "Cd": frozenset({2}), "Pb": frozenset({2, 4}),
    "Zr": frozenset({4}), "Hf": frozenset({4}),
    "Cr": frozenset({2, 3, 6}), "Fe": frozenset({2, 3}),
    "Mo": frozenset({3, 4, 5, 6}), "Tc": frozenset({4, 5, 7}),
    "Ru": frozenset({2, 3, 4, 6, 8}), "Pd": frozenset({2, 4}),
    "Ac": frozenset({3}), "Th": frozenset({4}), "Pa": frozenset({4, 5}),
    "U": frozenset({3, 4, 5, 6}), "Np": frozenset({3, 4, 5, 6, 7}),
    "Pu": frozenset({3, 4, 5, 6}), "Am": frozenset({3, 4, 5, 6}),
    "Cm": frozenset({3, 4}), "Bk": frozenset({3, 4}), "Cf": frozenset({2, 3, 4}),
    **{ln: frozenset({3}) for ln in LANTHANIDES},
    "Ce": frozenset({3, 4}), "Eu": frozenset({2, 3}), "Sm": frozenset({2, 3}), "Yb": frozenset({2, 3}),
}

#: Actinyl dioxo cations: element -> species name.  An(V) = AnO2(+), An(VI) = AnO2(2+).
ACTINYL_NAMES: dict[str, str] = {"U": "uranyl", "Np": "neptunyl", "Pu": "plutonyl", "Am": "americyl"}


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value)) if not isinstance(value, (str, bytes)) else False
    except (TypeError, ValueError):
        return False


def parse_oxidation_state(value: Any) -> int | None:
    """``"III"`` / ``"iii"`` / ``3`` / ``3.0`` / ``"3"`` / ``"+3"`` / ``"3+"`` / ``"(III)"`` /
    ``"+++"`` -> ``3``; ``None`` / NaN / ``""`` -> ``None``.  Raises ``ValueError`` otherwise."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"not an oxidation state: {value!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        if not f.is_integer() or not 0 <= f <= 8:
            raise ValueError(f"not an integral oxidation state 0..8: {value!r}")
        return int(f)
    text = str(value).strip()
    for ch in "()[]{}^ ":
        text = text.replace(ch, "")
    if text == "":
        return None
    if re.fullmatch(r"\++", text):
        n = len(text)
    elif (m := re.fullmatch(r"\+?(\d)\+?", text)) is not None:
        n = int(m.group(1))
    elif (m := re.fullmatch(r"\+?([IVXivx]+)\+?", text)) is not None and m.group(1).upper() in _ROMAN_TO_INT:
        n = _ROMAN_TO_INT[m.group(1).upper()]
    else:
        raise ValueError(f"cannot parse oxidation state {value!r}")
    if not 0 <= n <= 8:
        raise ValueError(f"oxidation state out of range 0..8: {value!r}")
    return n


def _parse_charge(text: str) -> int | None:
    """Charge suffix of a species label: ``"+2"``, ``"2+"``, ``"(2+)"``, ``"++"``, ``"+"`` -> int."""
    t = text
    for ch in "()[]{}^ ":
        t = t.replace(ch, "")
    if t == "":
        return None
    if re.fullmatch(r"\++", t):
        return len(t)
    m = re.fullmatch(r"\+(\d)|(\d)\+", t)
    if m is None:
        raise ValueError(f"cannot parse species charge {text!r}")
    return int(m.group(1) or m.group(2))


def default_species_form(symbol: str, oxidation_state: int | None) -> str | None:
    """Species a recorded ``(symbol, state)`` stands for in aqueous extraction media.

    ``None`` when the state is unknown; ``uranyl``/``neptunyl``/``plutonyl``/``americyl`` for
    An(V)/An(VI) of U, Np, Pu, Am (the dioxo cation *is* the aqueous form of those states);
    ``pertechnetate`` for Tc(VII) (an anion, TcO4-); ``protactinium_v_oxo`` for Pa(V) (hydrolysed
    oxo species, not a clean dioxo cation); for a state outside :data:`ACCESSIBLE_STATES`
    ``non_aqueous_state`` when Shannon (1976) tabulates the ion (a real ion in solids, e.g. the
    Pa(III) series row) and ``implausible_state`` otherwise (e.g. the archive's Sr(III));
    otherwise ``free_ion`` (the formal cation -- aqueous complexation and hydrolysis are condition
    properties, not metal properties)."""
    if oxidation_state is None:
        return None
    if symbol in ACTINYL_NAMES and oxidation_state in (5, 6):
        return ACTINYL_NAMES[symbol]
    if (symbol, oxidation_state) == ("Tc", 7):
        return "pertechnetate"
    if (symbol, oxidation_state) == ("Pa", 5):
        return "protactinium_v_oxo"
    allowed = ACCESSIBLE_STATES.get(symbol)
    if allowed is not None and oxidation_state not in allowed:
        # A real ion Shannon tabulates in solids (Pa(III)) is not an archive typo (Sr(III)).
        return "non_aqueous_state" if (symbol, oxidation_state) in SHANNON_RADII else "implausible_state"
    return "free_ion"


def _canonical_symbol(token: str) -> str | None:
    cand = token[:1].upper() + token[1:].lower()
    return cand if cand in ATOMIC_NUMBER else None


def normalize_metal(label: Any, oxidation_state: Any = None) -> tuple[str, int | None, str | None]:
    """Resolve a metal label to ``(symbol, oxidation_state or None, species_form or None)``.

    Accepted labels (whitespace and case tolerant): ``"Nd"``, ``"Nd(III)"``, ``"Nd(3)"``,
    ``"Nd(3+)"``, ``"Nd3+"``, ``"Nd+3"``, ``"Nd III"``, ``"Nd+++"``, ``"neodymium"``,
    actinyl labels ``"UO2+2"``, ``"UO2(2+)"``, ``"UO2^2+"``, ``"UO22+"``, ``"UO2++"`` (-> U(VI)),
    ``"NpO2+"`` (-> Np(V)), ``"PuO2+2"`` (-> Pu(VI)), and ``"uranyl"`` (-> U(VI)).

    ``oxidation_state`` is an optional separately recorded state (the archive keeps
    ``metal_raw`` and ``metal_oxidation_state_raw`` apart); it must agree with any state carried
    by the label.  Raises ``ValueError`` on an unknown element symbol, an actinyl without a
    charge, an unparsable state, or a conflict.
    """
    if _is_missing(label):
        raise ValueError("empty metal label")
    text = str(label).strip()
    if not text:
        raise ValueError("empty metal label")
    extra = parse_oxidation_state(oxidation_state)

    actinyl = re.fullmatch(r"([A-Za-z]{1,2})[Oo]2\s*(.*)", text)
    if text.lower() == "uranyl":
        symbol, ox = "U", 6
    elif actinyl is not None and _canonical_symbol(actinyl.group(1)) in ACTINYL_NAMES:
        charge = _parse_charge(actinyl.group(2))
        if charge is None:
            raise ValueError(f"actinyl label without a charge is ambiguous: {label!r}")
        if charge not in (1, 2):
            raise ValueError(f"actinyl charge must be +1 or +2: {label!r}")
        symbol, ox = _canonical_symbol(actinyl.group(1)), 4 + charge
    else:
        letters = re.match(r"[A-Za-z]+", text)
        if letters is None:
            raise ValueError(f"unrecognised metal label {label!r}")
        word, tail = letters.group(0), text[letters.end():].strip()
        if word.lower() in _NAME_TO_SYMBOL:
            symbol = _NAME_TO_SYMBOL[word.lower()]
        elif len(word) <= 2 and _canonical_symbol(word) is not None:
            symbol = _canonical_symbol(word)
        else:
            # Glued roman numerals ("NdIII", "BIII") are ambiguous (B(III) or Bi(II)?): refuse.
            raise ValueError(f"unknown element symbol in metal label {label!r}")
        if tail.startswith("-") and len(tail) > 1 and tail[1] not in "0123456789":
            tail = tail[1:]                               # "Nd-III" separator, not a negative charge
        ox = parse_oxidation_state(tail) if tail else None

    if symbol not in ATOMIC_NUMBER:
        raise ValueError(f"unknown element symbol in metal label {label!r}")
    if extra is not None:
        if ox is not None and ox != extra:
            raise ValueError(f"label {label!r} implies oxidation state {ox}, record says {extra}")
        ox = extra
    return symbol, ox, default_species_form(symbol, ox)


def metal_state_label(symbol: str, oxidation_state: Any) -> str | None:
    """``("Nd", 3)`` -> ``"Nd(III)"``; ``None`` when the state is unknown (as ``g19_metal_state``)."""
    ox = parse_oxidation_state(oxidation_state)
    return None if ox is None else f"{symbol}({ROMAN[ox]})"


# --------------------------------------------------------------------------------------------- #
# Electron configuration of the ion (ionic model, definitional arithmetic)
# --------------------------------------------------------------------------------------------- #

_P_BLOCK_IONS: dict[tuple[str, int], tuple[str, int, int]] = {
    ("In", 3): ("[Kr] 4d10", 0, 10),
    ("Pb", 2): ("[Xe] 4f14 5d10 6s2", 14, 10),
    ("Pb", 4): ("[Xe] 4f14 5d10", 14, 10),
    ("Bi", 3): ("[Xe] 4f14 5d10 6s2", 14, 10),
    ("Bi", 5): ("[Xe] 4f14 5d10", 14, 10),
}
#: Noble-gas core of a cation, keyed by the period of its element (Ca -> [Ar], Pd -> [Kr]).
_NOBLE_CORE = {4: "[Ar]", 5: "[Kr]", 6: "[Xe]", 7: "[Rn]"}


def ion_configuration(symbol: str, oxidation_state: int) -> tuple[str, int, int] | None:
    """``(configuration, f_count, d_count)`` of the ion in the ionic model, or ``None``.

    f_count = electrons in the (n-2)f subshell and d_count = electrons in the (n-1)d subshell,
    n = period: Ln(z+) ``[Xe] 4f^(Z-54-z)``; An(z+) ``[Rn] 5f^(Z-86-z)``; group 3-12 d-block
    ``core (n-1)d^(group-z)`` (period-6 core includes 4f14); s-block cations the preceding noble
    gas; p-block ions from an explicit table.  ``None`` for implausible or non-physical counts.
    """
    if default_species_form(symbol, oxidation_state) == "implausible_state":
        return None
    z = ATOMIC_NUMBER[symbol]
    _, period, group, block, category = _ELEMENT_ROWS[symbol]
    if category == "lanthanide":
        f = z - 54 - oxidation_state
        return (f"[Xe] 4f{f}" if f > 0 else "[Xe]", f, 0) if 0 <= f <= 14 else None
    if category == "actinide":
        f = z - 86 - oxidation_state
        return (f"[Rn] 5f{f}" if f > 0 else "[Rn]", f, 0) if 0 <= f <= 14 else None
    if block == "s":
        return (_NOBLE_CORE[period], 0, 0) if oxidation_state == group else None
    if block == "d" and group is not None:
        d = group - oxidation_state
        if not 0 <= d <= 10:
            return None
        core = _NOBLE_CORE[period] + (" 4f14" if period == 6 else "")
        f = 14 if period == 6 else 0
        return (core + (f" {period - 1}d{d}" if d > 0 else ""), f, d)
    return _P_BLOCK_IONS.get((symbol, oxidation_state))


# --------------------------------------------------------------------------------------------- #
# Literature values
# --------------------------------------------------------------------------------------------- #

_CN_ROMAN = {6: "VI", 8: "VIII", 9: "IX"}
SHANNON_REF = "Shannon, R. D. Acta Cryst. A32 (1976) 751-767, Table 1, effective ionic radius (IR)"

#: Shannon (1976) effective ionic radii (angstrom) keyed (symbol, state) -> {CN: radius}.
#: Only coordination numbers Shannon tabulates for that ion are present.
SHANNON_RADII: dict[tuple[str, int], dict[int, float]] = {
    ("La", 3): {6: 1.032, 8: 1.160, 9: 1.216},
    ("Ce", 3): {6: 1.01, 8: 1.143, 9: 1.196},
    ("Ce", 4): {6: 0.87, 8: 0.97},
    ("Pr", 3): {6: 0.99, 8: 1.126, 9: 1.179},
    ("Nd", 3): {6: 0.983, 8: 1.109, 9: 1.163},
    ("Pm", 3): {6: 0.97, 8: 1.093, 9: 1.144},
    ("Sm", 3): {6: 0.958, 8: 1.079, 9: 1.132},
    ("Eu", 3): {6: 0.947, 8: 1.066, 9: 1.120},
    ("Gd", 3): {6: 0.938, 8: 1.053, 9: 1.107},
    ("Tb", 3): {6: 0.923, 8: 1.040, 9: 1.095},
    ("Dy", 3): {6: 0.912, 8: 1.027, 9: 1.083},
    ("Ho", 3): {6: 0.901, 8: 1.015, 9: 1.072},
    ("Er", 3): {6: 0.890, 8: 1.004, 9: 1.062},
    ("Tm", 3): {6: 0.880, 8: 0.994, 9: 1.052},
    ("Yb", 3): {6: 0.868, 8: 0.985, 9: 1.042},
    ("Lu", 3): {6: 0.861, 8: 0.977, 9: 1.032},
    ("Y", 3): {6: 0.900, 8: 1.019, 9: 1.075},
    ("Sc", 3): {6: 0.745, 8: 0.870},
    ("Ca", 2): {6: 1.00, 8: 1.12, 9: 1.18},
    ("Sr", 2): {6: 1.18, 8: 1.26, 9: 1.31},
    ("Ba", 2): {6: 1.35, 8: 1.42, 9: 1.47},
    ("Cd", 2): {6: 0.95, 8: 1.10},
    ("Pb", 2): {6: 1.19, 8: 1.29, 9: 1.35},
    ("In", 3): {6: 0.800, 8: 0.92},
    ("Bi", 3): {6: 1.03, 8: 1.17},
    ("Zr", 4): {6: 0.72, 8: 0.84, 9: 0.89},
    ("Hf", 4): {6: 0.71, 8: 0.83},
    ("Pd", 2): {6: 0.86},
    ("Pd", 4): {6: 0.615},
    ("Tc", 7): {6: 0.56},
    ("Ac", 3): {6: 1.12},
    ("Th", 4): {6: 0.94, 8: 1.05, 9: 1.09},
    ("Pa", 3): {6: 1.04},
    ("Pa", 5): {6: 0.78, 8: 0.91, 9: 0.95},
    ("U", 3): {6: 1.025},
    ("U", 4): {6: 0.89, 8: 1.00, 9: 1.05},
    ("U", 6): {6: 0.73, 8: 0.86},
    ("Np", 3): {6: 1.01},
    ("Np", 4): {6: 0.87, 8: 0.98},
    ("Np", 5): {6: 0.75},
    ("Np", 6): {6: 0.72},
    ("Pu", 3): {6: 1.00},
    ("Pu", 4): {6: 0.86, 8: 0.96},
    ("Pu", 6): {6: 0.71},
    ("Am", 3): {6: 0.975, 8: 1.09},
    ("Cm", 3): {6: 0.97},
    ("Bk", 3): {6: 0.96},
    ("Cf", 3): {6: 0.95},
}

EN_REF = ("Pauling scale; Allred, J. Inorg. Nucl. Chem. 17 (1961) 215 and Huheey, Keiter & Keiter, "
          "Inorganic Chemistry 4th ed. (1993), as quoted by WebElements; transcribed from Wikipedia "
          "'Electronegativities of the elements (data page)', accessed 2026-09-14")
#: Pauling electronegativity of the ELEMENT.
PAULING_EN: dict[str, float] = {
    "Ca": 1.00, "Sc": 1.36, "Cr": 1.66, "Fe": 1.83, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Mo": 2.16,
    "Tc": 1.9, "Ru": 2.2, "Pd": 2.20, "Cd": 1.69, "In": 1.78, "Ba": 0.89,
    "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17, "Gd": 1.20, "Dy": 1.22, "Ho": 1.23,
    "Er": 1.24, "Tm": 1.25, "Lu": 1.27, "Hf": 1.3, "Pb": 2.33, "Bi": 2.02,
    "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.3, "Cm": 1.3,
    "Bk": 1.3, "Cf": 1.3,
}
#: The data page gives one value per element without its oxidation state, but Allred (1961) computed
#: state-specific values for a few elements.  Wikipedia 'Electronegativity' (accessed 2026-09-15) states
#: that lead "conforms better to trends if it is quoted for the +2 state with a Pauling value of 1.87
#: instead of the +4 state", i.e. the tabulated 2.33 is Pb(IV).  State rows listed here replace the
#: element value; the element row keeps the tabulated value with the note below.
PAULING_EN_STATE_REF = ("Allred, J. Inorg. Nucl. Chem. 17 (1961) 215, state-specific value as quoted by Wikipedia "
                        "'Electronegativity' (accessed 2026-09-15): the +2 state of lead is 1.87, the tabulated "
                        "element value 2.33 is the +4 state")
PAULING_EN_STATE: dict[tuple[str, int], float] = {("Pb", 2): 1.87}
#: element rows whose tabulated value belongs to a specific (or unverified) oxidation state
PAULING_EN_ELEMENT_NOTE: dict[str, str] = {
    "Pb": "; NOTE: 2.33 is Allred's Pb(IV) value (Wikipedia 'Electronegativity'); the Pb(II) state row uses 1.87",
    "Fe": ("; NOTE: the source does not state the oxidation state of this value (INFERRED, not verified: Allred's "
           "M(II) value); extraction chemistry is Fe(III) -- state-independent use is not justified"),
    "Cr": ("; NOTE: the source does not state the oxidation state of this value (INFERRED, not verified: Allred's "
           "M(II) value); extraction chemistry is Cr(III) -- state-independent use is not justified"),
    "Mo": ("; NOTE: the source does not state the oxidation state of this value (INFERRED, not verified: Allred's "
           "M(II) value); extraction chemistry is Mo(VI) -- state-independent use is not justified"),
}
PAULING_EN_NA_REASON: dict[str, str] = {
    "Pm": "NA_REASON: the source table gives only a range (1.10-1.20); no single literature value",
    "Eu": "NA_REASON: the source table gives only a range (1.1-1.2); no single literature value",
    "Yb": "NA_REASON: the source table gives only a range (1.1-1.2); no single literature value",
    "Tb": ("NA_REASON: not in Allred (1961); secondary tables give a one-significant-figure estimate "
           "(1.1, also seen as 1.2) that breaks the Gd 1.20 -> Dy 1.22 trend; not certain"),
}

HSAB_EXPLICIT = ("Pearson, J. Am. Chem. Soc. 85 (1963) 3533, hard/borderline/soft acid table as "
                 "reproduced in Wikipedia 'HSAB theory' (accessed 2026-09-14)")
HSAB_RECALLED = ("Pearson, J. Am. Chem. Soc. 85 (1963) 3533, Table I (standard textbook list; not "
                 "re-checked against the printed table in this build)")
HSAB_ANALOGY = "INFERRED by analogy within Pearson (1963) classes: "
#: (symbol, state) -> (class, basis tag, source text)
HSAB: dict[tuple[str, int], tuple[str, str, str]] = {
    **{(ln, 3): ("hard", "explicit", HSAB_EXPLICIT + " ('lanthanides Ln3+')") for ln in LANTHANIDES},
    ("Sc", 3): ("hard", "explicit", HSAB_EXPLICIT),
    ("Th", 4): ("hard", "explicit", HSAB_EXPLICIT),
    ("U", 4): ("hard", "explicit", HSAB_EXPLICIT),
    ("Pb", 2): ("borderline", "explicit", HSAB_EXPLICIT),
    ("Pd", 2): ("soft", "explicit", HSAB_EXPLICIT),
    ("Ca", 2): ("hard", "recalled", HSAB_RECALLED),
    ("Sr", 2): ("hard", "recalled", HSAB_RECALLED),
    ("In", 3): ("hard", "recalled", HSAB_RECALLED),
    ("Zr", 4): ("hard", "recalled", HSAB_RECALLED),
    ("Pu", 4): ("hard", "recalled", HSAB_RECALLED),
    ("U", 6): ("hard", "recalled", HSAB_RECALLED + " (UO2 2+)"),
    ("Cd", 2): ("soft", "recalled", HSAB_RECALLED),
    ("Bi", 3): ("borderline", "recalled", HSAB_RECALLED),
    ("Ba", 2): ("hard", "analogy", HSAB_ANALOGY + "alkaline-earth cation like Ca2+, Sr2+"),
    ("Y", 3): ("hard", "analogy", HSAB_ANALOGY + "group-3 M3+ like Sc3+, La3+"),
    ("Hf", 4): ("hard", "analogy", HSAB_ANALOGY + "group-4 M4+ like Zr4+"),
    ("Np", 4): ("hard", "analogy", HSAB_ANALOGY + "An4+ like Th4+, U4+, Pu4+"),
    **{(an, 3): ("hard", "analogy", HSAB_ANALOGY + "An3+ like Ln3+ (trivalent f-element cation)")
       for an in AN3_SERIES},
    **{(an, 6): ("hard", "analogy", HSAB_ANALOGY + "AnO2 2+ like UO2 2+") for an in ("Np", "Pu", "Am")},
    ("Np", 5): ("hard", "analogy", HSAB_ANALOGY + "actinyl dioxo cation like UO2 2+ (weakest analogy: "
                "NpO2+ effective charge ~2.2)"),
}
HSAB_NA_REASON: dict[tuple[str, int], str] = {
    ("Tc", 7): "NA_REASON: Tc(VII) exists as the TcO4- anion; an acid class is not meaningful",
    ("Pa", 5): "NA_REASON: no standard classification; Pa(V) is a hydrolysed oxo species, not a clean cation",
    ("Pd", 4): "NA_REASON: not in Pearson's lists; soft by analogy to Pt4+ is plausible but not standard",
}

POLARIZABILITY_NA_REASON = (
    "NA_REASON: Schwerdtfeger & Nagle, Mol. Phys. 117 (2019) 1200, doi:10.1080/00268976.2018.1535143 "
    "(2018 table) could not be opened during this build (tandfonline/ingentaconnect HTTP 403; "
    "ctcp.massey.ac.nz and github.com unresolvable from the build host); values are not "
    "transcribed from memory")

CHOPPIN_REF = ("Choppin & Rao, Radiochim. Acta 37 (1984) 143-146, doi:10.1524/ract.1984.37.3.143 "
               "(NpO2+ 2.2, UO2 2+ 3.2; as quoted in OSTI 1366418)")
#: Effective cationic charge of actinyl species (Choppin electrostatic scale).
EFFECTIVE_CHARGE_ACTINYL: dict[tuple[str, int], float] = {("U", 6): 3.2, ("Np", 5): 2.2}

# --------------------------------------------------------------------------------------------- #
# The descriptor table
# --------------------------------------------------------------------------------------------- #

VALUE_COLUMNS: tuple[str, ...] = (
    "Z", "name", "period", "group", "block", "category", "series", "series_index",
    "oxidation_state", "formal_charge", "species_form", "species_charge", "effective_charge",
    "f_electron_count", "d_electron_count", "electron_configuration",
    "radius_cn6_A", "radius_cn8_A", "radius_cn9_A",
    "electronegativity_pauling", "hsab_class", "polarizability_atom_au",
)
#: Properties of the element (identical on every row of a symbol, filled on NA-state rows).
ELEMENT_LEVEL_COLUMNS: tuple[str, ...] = (
    "Z", "name", "period", "group", "block", "category", "series", "series_index",
    "electronegativity_pauling", "polarizability_atom_au",
)
ION_LEVEL_COLUMNS: tuple[str, ...] = tuple(c for c in VALUE_COLUMNS
                                           if c not in ELEMENT_LEVEL_COLUMNS and c != "oxidation_state")
META_COLUMNS: tuple[str, ...] = ("symbol", "oxidation_state", "metal_state_label", "category_state_key",
                                 "row_origin", "in_archive", "state_plausible", "hsab_class_basis")
INT_COLUMNS: tuple[str, ...] = ("Z", "period", "group", "series_index", "oxidation_state",
                                "formal_charge", "species_charge", "f_electron_count", "d_electron_count")
FLOAT_COLUMNS: tuple[str, ...] = ("effective_charge", "radius_cn6_A", "radius_cn8_A", "radius_cn9_A",
                                  "electronegativity_pauling", "polarizability_atom_au")

NA_STATE_REASON = "NA_REASON: oxidation state not recorded; ion-level descriptor undefined"
IMPLAUSIBLE_REASON = "NA_REASON: recorded oxidation state is not chemically accessible (archive flag)"


def series_keys() -> list[tuple[str, int]]:
    """The Ln(III) La..Lu (with Pm) and An(III) Ac..Cf keys added for series smoothness."""
    return [(s, 3) for s in LANTHANIDES] + [(s, 3) for s in AN3_SERIES]


def _element_values(symbol: str) -> dict[str, tuple[Any, str]]:
    name, period, group, block, category = _ELEMENT_ROWS[symbol]
    z = ATOMIC_NUMBER[symbol]
    if category == "lanthanide":
        series, sidx = "Ln", z - 57
    elif category == "actinide":
        series, sidx = "An", z - 89
    else:
        series, sidx = "none", None
    out: dict[str, tuple[Any, str]] = {
        "Z": (z, "definition: IUPAC atomic number"),
        "name": (name, "definition: IUPAC English element name"),
        "period": (period, "definition: IUPAC periodic table row"),
        "group": (group, "definition: IUPAC group 1-18") if group is not None else
                 (None, "NA_REASON: f-block element; group left NA by gen19 convention"),
        "block": (block, "definition: Aufbau block; La/Lu/Ac labelled f as series members "
                         "(convention of gen11 metalrep.PERIODIC_TABLE)"),
        "category": (category, "gen19 convention = dataset_all_metals/scripts/sae_chem.py metal_category "
                               "(Y, Sc rare_earth_non_lanthanide; Cd post_transition_metal)"),
        "series": (series, "definition: Ln = La-Lu (Z 57-71), An = Ac-Lr (Z 89-103), else none"),
        "series_index": (sidx, "definition: Ln Z-57 (La=0..Lu=14); An Z-89 (Ac=0..Cf=9)")
                        if sidx is not None else (None, "NA_REASON: not a lanthanide or actinide"),
    }
    if symbol in PAULING_EN:
        out["electronegativity_pauling"] = (PAULING_EN[symbol], EN_REF + PAULING_EN_ELEMENT_NOTE.get(symbol, ""))
    else:
        out["electronegativity_pauling"] = (None, PAULING_EN_NA_REASON.get(
            symbol, "NA_REASON: no value transcribed for this element"))
    out["polarizability_atom_au"] = (None, POLARIZABILITY_NA_REASON)
    return out


def _ion_values(symbol: str, ox: int | None) -> dict[str, tuple[Any, str]]:
    if ox is None:
        return {c: (None, NA_STATE_REASON) for c in ION_LEVEL_COLUMNS}
    species = default_species_form(symbol, ox)
    if species == "implausible_state":
        out = {c: (None, IMPLAUSIBLE_REASON) for c in ION_LEVEL_COLUMNS}
        out["species_form"] = ("implausible_state", "gen19 metals.default_species_form: state outside "
                               "ACCESSIBLE_STATES (= archive PLAUSIBLE_OX)")
        return out
    out: dict[str, tuple[Any, str]] = {
        "formal_charge": (ox, "definition: formal ionic charge = oxidation state"),
    }
    if species in ("free_ion", "non_aqueous_state"):
        out["species_form"] = (species, "definition: formal bare cation M(z+)" if species == "free_ion" else
                               "gen19 metals.default_species_form: state outside ACCESSIBLE_STATES but "
                               "tabulated by Shannon (1976) in solids; formal bare cation, not an aqueous "
                               "extraction species")
        out["species_charge"] = (ox, "definition: charge of the bare cation = oxidation state")
        out["effective_charge"] = (float(ox), "definition on the Choppin electrostatic scale: bare cation "
                                              "Z_eff = z (" + CHOPPIN_REF + ")")
    elif species in ACTINYL_NAMES.values():
        charge = ox - 4
        out["species_form"] = (species, f"definition: An({ROMAN[ox]}) is the dioxo cation AnO2({charge}+) "
                                        "in aqueous media")
        out["species_charge"] = (charge, "definition: AnO2(+) for An(V), AnO2(2+) for An(VI)")
        if (symbol, ox) in EFFECTIVE_CHARGE_ACTINYL:
            out["effective_charge"] = (EFFECTIVE_CHARGE_ACTINYL[(symbol, ox)], CHOPPIN_REF)
        else:
            out["effective_charge"] = (None, "NA_REASON: Choppin & Rao (1984) give NpO2+ 2.2 and UO2 2+ 3.2 "
                                             "only; class-wide transfer not verified in this build")
    elif species == "pertechnetate":
        out["species_form"] = (species, "definition: Tc(VII) is the TcO4- oxoanion in aqueous media")
        out["species_charge"] = (-1, "definition: TcO4- (same as gen11 OXO_SPECIES_CHARGE)")
        out["effective_charge"] = (None, "NA_REASON: anion; no cationic effective charge")
    elif species == "protactinium_v_oxo":
        out["species_form"] = (species, "Pa(V) hydrolyses to oxo/hydroxo species (e.g. PaO(OH)2+), not a "
                                        "clean dioxo cation (gen11 metalrep.py OXO_SPECIES_CHARGE note)")
        out["species_charge"] = (None, "NA_REASON: Pa(V) aqueous speciation is a mixture; no single charge")
        out["effective_charge"] = (None, "NA_REASON: no citable effective charge for Pa(V)")
    conf = ion_configuration(symbol, ox)
    if conf is None:
        for c in ("electron_configuration", "f_electron_count", "d_electron_count"):
            out[c] = (None, "NA_REASON: no configuration rule for this ion")
    else:
        src = ("definition (ionic model): noble-gas core + (n-2)f/(n-1)d occupancy from Z and "
               "oxidation state; f = Z-54-z (Ln), Z-86-z (An); d = group-z (d-block)")
        out["electron_configuration"] = (conf[0], src)
        out["f_electron_count"] = (conf[1], src)
        out["d_electron_count"] = (conf[2], src)
    radii = SHANNON_RADII.get((symbol, ox), {})
    for cn in (6, 8, 9):
        col = f"radius_cn{cn}_A"
        if cn in radii:
            out[col] = (radii[cn], f"{SHANNON_REF}, CN={_CN_ROMAN[cn]}")
        else:
            out[col] = (None, f"NA_REASON: Shannon (1976) does not tabulate {symbol}({ROMAN[ox]}) at CN={cn}")
    if (symbol, ox) in HSAB:
        cls, basis, src = HSAB[(symbol, ox)]
        out["hsab_class"] = (cls, src)
        out["_hsab_basis"] = (basis, "")
    else:
        out["hsab_class"] = (None, HSAB_NA_REASON.get((symbol, ox), "NA_REASON: no standard HSAB class"))
    return out


def build_descriptor_table(keys: Iterable[tuple[str, Any]],
                           archive_keys: Iterable[tuple[str, Any]] = ()) -> pd.DataFrame:
    """One row per unique ``(symbol, oxidation_state)`` in ``keys`` (state may be ``None``/NaN),
    plus the Ln(III)/An(III) series rows.  ``archive_keys`` marks which rows the archive holds."""
    def norm(k: tuple[str, Any]) -> tuple[str, int | None]:
        sym, ox = k
        if sym not in _ELEMENT_ROWS:
            raise KeyError(f"no descriptor element row for {sym!r}")
        return sym, parse_oxidation_state(ox)

    arch = {norm(k) for k in archive_keys}
    ln3 = {(s, 3) for s in LANTHANIDES}
    an3 = {(s, 3) for s in AN3_SERIES}
    all_keys = {norm(k) for k in keys} | arch | ln3 | an3
    rows = []
    for sym, ox in sorted(all_keys, key=lambda k: (ATOMIC_NUMBER[k[0]], -1 if k[1] is None else k[1])):
        vals = {**_element_values(sym), **_ion_values(sym, ox)}
        if ox is not None and (sym, ox) in PAULING_EN_STATE:
            vals["electronegativity_pauling"] = (PAULING_EN_STATE[(sym, ox)], PAULING_EN_STATE_REF)
        origin = [tag for tag, s in (("archive", arch), ("ln3_series", ln3), ("an3_series", an3))
                  if (sym, ox) in s]
        allowed = ACCESSIBLE_STATES.get(sym)
        row: dict[str, Any] = {
            "symbol": sym,
            "oxidation_state": ox,
            "metal_state_label": metal_state_label(sym, ox),
            "category_state_key": None if ox is None else f"{_ELEMENT_ROWS[sym][4]}({ROMAN[ox]})",
            "row_origin": "+".join(origin),
            "in_archive": (sym, ox) in arch,
            "state_plausible": None if (ox is None or allowed is None) else (ox in allowed),
            "hsab_class_basis": vals.pop("_hsab_basis", (None, ""))[0],
        }
        for col in VALUE_COLUMNS:
            if col == "oxidation_state":
                if ox is None:
                    src = "NA_REASON: archive did not record an oxidation state for these rows"
                elif (sym, ox) in arch:
                    src = "archive: dataset_all_metals master_clean metal_oxidation_state"
                else:
                    src = "definition: trivalent series row (Ln(III) La-Lu / An(III) Ac-Cf)"
                row["oxidation_state_source"] = src
                continue
            value, source = vals[col]
            row[col] = value
            row[f"{col}_source"] = source
        rows.append(row)
    ordered = list(META_COLUMNS)
    for col in VALUE_COLUMNS:
        if col == "oxidation_state":
            ordered.append("oxidation_state_source")
        else:
            ordered += [col, f"{col}_source"]
    df = pd.DataFrame(rows)[ordered]
    for c in INT_COLUMNS:
        df[c] = pd.array([None if _is_missing(v) else int(v) for v in df[c]], dtype="Int64")
    for c in FLOAT_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["in_archive"] = df["in_archive"].astype(bool)
    df["state_plausible"] = pd.array(list(df["state_plausible"]), dtype="boolean")
    return df.reset_index(drop=True)


def descriptor_key(symbol: Any, oxidation_state: Any) -> str:
    """Join key ``"Nd|3"`` / ``"Nd|NA"`` used to attach the table to archive rows."""
    ox = parse_oxidation_state(oxidation_state)
    return f"{symbol}|{'NA' if ox is None else ox}"
