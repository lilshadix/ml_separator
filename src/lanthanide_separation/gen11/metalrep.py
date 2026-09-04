"""What the gen10 metal block does when it meets 39 metals it was never written for.

The frozen ``METAL`` block is three columns — ``Atomic Number_metal``,
``lanthanide_index``, ``Ionic Radius_metal`` — and gen10 obtains all three from a
**14-entry dictionary keyed by lanthanide symbol**
(``LANTHANIDE_DESCRIPTORS``, ``scripts/dataset/build_dataset.py:127``), attached with

    df.merge(metal_features, on="metal_symbol", how="left", validate="many_to_one")

A left merge on a 14-key table is the whole problem.  An Am row does not fail; it
silently acquires three NaNs, and gen10's design matrix ends in
``SimpleImputer(strategy="median", add_indicator=True)``, so those NaNs become the
*median lanthanide*.  Worse, ``add_indicator`` defaults to ``features="missing-only"``:
it emits an indicator only for columns that were missing **in the fold it was fitted
on**.  An arm whose training fold is lanthanide-only therefore produces no indicator
at all, and every actinide row arrives at the tree wearing a middle-lanthanide
costume with nothing to say it is a costume.  ``lanthanide_index`` is the sharpest
case: it is literally ``Z - 56``, so imputing it for americium asserts "this is
element 56 + 8 = gadolinium" while ``Atomic Number_metal`` simultaneously says 95.
The two columns then disagree by 31 protons on the same row.

This module does three things and no modelling:

1. audits the archive's metal columns per metal, over the full archive and over the
   A_model_ready auxiliary rows, so the row cost of each feature is a number;
2. checks the archive's own lanthanide values against ``LANTHANIDE_DESCRIPTORS``,
   which decides whether archive columns may be used directly for lanthanide rows;
3. supplies four pre-registered representations, each a pure ``build(records)``.

The pre-registration matters: which metal representation to use is a *choice*, and
choosing it after seeing transfer results would be the same p-hacking gen11 exists
to avoid.  The four are fixed here, before any arm runs.

Nothing in this module reads ``log_D``, ``D_value`` or any provenance column, and
the builders never touch the target — :func:`assert_no_target_leakage` enforces the
first half mechanically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "runs" / "gen11_transfer" / "metalrep"

# --------------------------------------------------------------------------- #
# Frozen reference data
# --------------------------------------------------------------------------- #

#: Verbatim copy of ``LANTHANIDE_DESCRIPTORS`` from the frozen bundle builder
#: (``scripts/dataset/build_dataset.py:127``).  Copied rather than imported
#: because that repository is not a dependency of this one and the frozen bundle
#: must stay reproducible from this tree alone.  Note Pm is absent: the builder
#: had no promethium rows, so a Pm row would left-merge to NaN even though Pm is
#: a lanthanide.  ``lanthanide_index`` is Z - 56.
LANTHANIDE_DESCRIPTORS: Mapping[str, Mapping[str, float]] = {
    "La": {"Atomic Number_metal": 57, "lanthanide_index": 1, "Ionic Radius_metal": 1.160},
    "Ce": {"Atomic Number_metal": 58, "lanthanide_index": 2, "Ionic Radius_metal": 1.143},
    "Pr": {"Atomic Number_metal": 59, "lanthanide_index": 3, "Ionic Radius_metal": 1.126},
    "Nd": {"Atomic Number_metal": 60, "lanthanide_index": 4, "Ionic Radius_metal": 1.109},
    "Sm": {"Atomic Number_metal": 62, "lanthanide_index": 6, "Ionic Radius_metal": 1.079},
    "Eu": {"Atomic Number_metal": 63, "lanthanide_index": 7, "Ionic Radius_metal": 1.066},
    "Gd": {"Atomic Number_metal": 64, "lanthanide_index": 8, "Ionic Radius_metal": 1.053},
    "Tb": {"Atomic Number_metal": 65, "lanthanide_index": 9, "Ionic Radius_metal": 1.040},
    "Dy": {"Atomic Number_metal": 66, "lanthanide_index": 10, "Ionic Radius_metal": 1.027},
    "Ho": {"Atomic Number_metal": 67, "lanthanide_index": 11, "Ionic Radius_metal": 1.015},
    "Er": {"Atomic Number_metal": 68, "lanthanide_index": 12, "Ionic Radius_metal": 1.004},
    "Tm": {"Atomic Number_metal": 69, "lanthanide_index": 13, "Ionic Radius_metal": 0.994},
    "Yb": {"Atomic Number_metal": 70, "lanthanide_index": 14, "Ionic Radius_metal": 0.985},
    "Lu": {"Atomic Number_metal": 71, "lanthanide_index": 15, "Ionic Radius_metal": 0.977},
}

FROZEN_METAL_COLUMNS: tuple[str, ...] = (
    "Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal",
)

#: Shannon (1976) CN=8 trivalent radii, Å, for the fifteen lanthanides.  Used
#: only as the flagged fallback described in :func:`build_general`; identical to
#: the ``Ionic Radius_metal`` entries above wherever both exist (asserted by
#: :func:`frozen_dict_agreement`), extended with Pm.
LN3_RADIUS_CN8: Mapping[str, float] = {
    "La": 1.160, "Ce": 1.143, "Pr": 1.126, "Nd": 1.109, "Pm": 1.093, "Sm": 1.079,
    "Eu": 1.066, "Gd": 1.053, "Tb": 1.040, "Dy": 1.027, "Ho": 1.015, "Er": 1.004,
    "Tm": 0.994, "Yb": 0.985, "Lu": 0.977,
}

#: Position in the periodic table for every metal symbol the archive contains.
#: ``(period, group, block)``.  This is external reference data, not anything
#: derived from the measurements.  ``group`` for the f-block is the conventional
#: group-3 slot the series occupies; La/Ac-type elements with a formally d1
#: ground state are still labelled ``f`` so that the whole series is one class.
PERIODIC_TABLE: Mapping[str, tuple[int, int, str]] = {
    "Ca": (4, 2, "s"), "Sc": (4, 3, "d"), "Cr": (4, 6, "d"), "Fe": (4, 8, "d"),
    "Sr": (5, 2, "s"), "Y": (5, 3, "d"), "Zr": (5, 4, "d"), "Mo": (5, 6, "d"),
    "Tc": (5, 7, "d"), "Ru": (5, 8, "d"), "Pd": (5, 10, "d"), "Cd": (5, 12, "d"),
    "In": (5, 13, "p"),
    "Ba": (6, 2, "s"),
    "La": (6, 3, "f"), "Ce": (6, 3, "f"), "Pr": (6, 3, "f"), "Nd": (6, 3, "f"),
    "Pm": (6, 3, "f"), "Sm": (6, 3, "f"), "Eu": (6, 3, "f"), "Gd": (6, 3, "f"),
    "Tb": (6, 3, "f"), "Dy": (6, 3, "f"), "Ho": (6, 3, "f"), "Er": (6, 3, "f"),
    "Tm": (6, 3, "f"), "Yb": (6, 3, "f"), "Lu": (6, 3, "f"),
    "Hf": (6, 4, "d"), "Pb": (6, 14, "p"), "Bi": (6, 15, "p"),
    "Th": (7, 3, "f"), "Pa": (7, 3, "f"), "U": (7, 3, "f"), "Np": (7, 3, "f"),
    "Pu": (7, 3, "f"), "Am": (7, 3, "f"), "Cm": (7, 3, "f"), "Cf": (7, 3, "f"),
}

BLOCKS: tuple[str, ...] = ("s", "d", "p", "f")

#: Net charge of the aqueous species when the recorded oxidation state does not
#: sit on a bare aquo cation.  Every entry is a textbook speciation fact for
#: nitric/hydrochloric extraction media, keyed on ``(symbol, oxidation_state)``
#: so it can only fire where the archive *recorded* that state — no oxidation
#: state is ever invented here.
#:
#:   * An(V)  -> AnO2(+)    net +1   (uranyl/neptunyl/plutonyl-type dioxo cation)
#:   * An(VI) -> AnO2(2+)   net +2
#:   * Tc(VII)-> TcO4(-)    net -1   (pertechnetate is an anion, not a 7+ cation)
#:
#: Pa(V) is deliberately absent: protactinium(V) hydrolyses to a mixture that is
#: not a clean dioxo cation, so its charge stays at the recorded oxidation state
#: and the README lists it as a known limitation (15 auxiliary rows).
OXO_SPECIES_CHARGE: Mapping[tuple[str, float], float] = {
    **{(s, 5.0): 1.0 for s in ("U", "Np", "Pu", "Am")},
    **{(s, 6.0): 2.0 for s in ("U", "Np", "Pu", "Am")},
    ("Tc", 7.0): -1.0,
}

#: Noble-gas core size used by the f-count rule below.
_LN_CORE_Z = 54   # Xe, so 4f^n for Ln(n+) is Z - 54 - n
_AN_CORE_Z = 86   # Rn, so 5f^n for An(n+) is Z - 86 - n

#: Per-scheme input contract.  ``FROZEN_3`` needs only the symbol — that is the
#: entire point of it — so it can be rebuilt straight from the frozen cohort and
#: checked against gen10's own columns (:func:`verify_frozen_reproduction`).
REQUIRED_INPUT_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "FROZEN_3": ("metal_symbol",),
    "FROZEN_3_INDICATED": ("metal_symbol",),
    "GENERAL": ("metal_symbol", "ionic_radius_cn8_A", "metal_oxidation_state"),
    "NO_RADIUS": ("metal_symbol", "ionic_radius_cn8_A", "metal_oxidation_state"),
}

#: Columns a metal representation may never contain, however it is built.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "log_d", "logd", "d_value", "target", "doi", "source_record", "exp_id",
    "series_id", "duplicate_group",
)


# --------------------------------------------------------------------------- #
# Deterministic element chemistry
# --------------------------------------------------------------------------- #

def f_electron_count(symbol: str | float, oxidation_state: float) -> float:
    """Valence f-electron count of the ion, or NaN when it is not determined.

    The rule is arithmetic on the noble-gas core, not a lookup table, and it is
    externally deterministic given ``(Z, oxidation state)``:

    ==================  ==========================  ==================
    element range       rule                        check
    ==================  ==========================  ==================
    57 <= Z <= 71 (Ln)  ``f = Z - 54 - n``          La(III) -> 0, Lu(III) -> 14
    89 <= Z <= 103 (An) ``f = Z - 86 - n``          Th(IV) -> 0, Am(III) -> 6, U(VI) -> 0
    Z < 57              ``f = 0``                   no f shell occupied
    71 < Z < 89         ``f = 14``                  filled 4f core (Hf, Pb, Bi)
    ==================  ==========================  ==================

    For the f-block the count depends on the oxidation state, so it is **NaN when
    the archive did not record one** — that is the honest answer and the reason
    :func:`build_general` ships a missingness indicator beside it.  Counts are
    clipped to ``[0, 14]``; a value outside that range means the recorded
    oxidation state is not chemically possible and is reported, not silently
    fixed.
    """
    if not isinstance(symbol, str) or symbol not in PERIODIC_TABLE:
        return float("nan")
    z = float(ATOMIC_NUMBER[symbol])
    if 57 <= z <= 71:
        if not np.isfinite(oxidation_state):
            return float("nan")
        return float(np.clip(z - _LN_CORE_Z - oxidation_state, 0.0, 14.0))
    if 89 <= z <= 103:
        if not np.isfinite(oxidation_state):
            return float("nan")
        return float(np.clip(z - _AN_CORE_Z - oxidation_state, 0.0, 14.0))
    if z < 57:
        return 0.0
    return 14.0


def effective_charge(symbol: str | float, oxidation_state: float) -> float:
    """Net charge of the aqueous metal species, or NaN if no state was recorded.

    Equal to the oxidation state except on the ``OXO_SPECIES_CHARGE`` entries
    documented above.  Never invents an oxidation state: a NaN in gives a NaN out.
    """
    if not np.isfinite(oxidation_state):
        return float("nan")
    if isinstance(symbol, str):
        override = OXO_SPECIES_CHARGE.get((symbol, float(oxidation_state)))
        if override is not None:
            return float(override)
    return float(oxidation_state)


def is_oxo_species(symbol: str | float, oxidation_state: float) -> float:
    """1.0 where ``OXO_SPECIES_CHARGE`` fires, 0.0 where a bare cation, NaN if unknown."""
    if not np.isfinite(oxidation_state):
        return float("nan")
    if not isinstance(symbol, str):
        return float("nan")
    return 1.0 if (symbol, float(oxidation_state)) in OXO_SPECIES_CHARGE else 0.0


#: symbol -> Z, from the archive's own ``atomic_number`` column but frozen here so
#: the builders do not depend on a row happening to carry it.
ATOMIC_NUMBER: Mapping[str, int] = {
    "Ca": 20, "Sc": 21, "Cr": 24, "Fe": 26, "Sr": 38, "Y": 39, "Zr": 40, "Mo": 42,
    "Tc": 43, "Ru": 44, "Pd": 46, "Cd": 48, "In": 49, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Pb": 82, "Bi": 83,
    "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Cf": 98,
}


# --------------------------------------------------------------------------- #
# The four pre-registered representations
# --------------------------------------------------------------------------- #

def _validate(records: pd.DataFrame, scheme: str) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS[scheme] if c not in records.columns]
    if missing:
        raise KeyError(f"{scheme} input is missing required columns: {missing}")


def assert_no_target_leakage(frame: pd.DataFrame) -> None:
    """A representation carrying a provenance or target column is a build error."""
    bad = [c for c in frame.columns
           if any(sub in c.lower() for sub in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise AssertionError(f"metal representation leaks non-metal columns: {bad}")


FROZEN_3_COLUMNS: tuple[str, ...] = FROZEN_METAL_COLUMNS


def build_frozen_3(records: pd.DataFrame) -> pd.DataFrame:
    """gen10's block reproduced literally: a left merge on the 14-entry dictionary.

    Columns
    -------
    ``Atomic Number_metal``   Z, from the dictionary — **NaN for any metal not in it**
    ``lanthanide_index``      Z - 56 — NaN for any metal not in the dictionary
    ``Ionic Radius_metal``    Shannon CN=8 Ln(III) radius — NaN outside the dictionary

    This is deliberately *not* "atomic number for everything".  gen10 does not read
    an element table; it reads a dictionary of fourteen lanthanides, so a Pm row —
    a lanthanide! — and every actinide row leave with three NaNs.  Reproducing that
    faithfully is what makes the row-loss table an audit rather than an opinion.
    """
    _validate(records, "FROZEN_3")
    symbol = records["metal_symbol"]
    out = pd.DataFrame(index=records.index)
    for col in FROZEN_METAL_COLUMNS:
        lookup = {k: float(v[col]) for k, v in LANTHANIDE_DESCRIPTORS.items()}
        out[col] = symbol.map(lookup).astype(float)
    assert_no_target_leakage(out)
    return out


FROZEN_3_INDICATED_COLUMNS: tuple[str, ...] = FROZEN_METAL_COLUMNS + (
    "metal_atomic_number_is_missing",
    "metal_lanthanide_index_is_missing",
    "metal_ionic_radius_is_missing",
    "metal_not_in_frozen_dict",
)


def build_frozen_3_indicated(records: pd.DataFrame) -> pd.DataFrame:
    """:func:`build_frozen_3` plus explicit missingness, so nothing is imputed mutely.

    The three per-column indicators exist because gen10's
    ``SimpleImputer(add_indicator=True)`` uses ``features="missing-only"`` and
    therefore emits *no* indicator when the fitting fold had no missing metal
    values — precisely the lanthanide-only control arm.  ``metal_not_in_frozen_dict``
    is the aggregate flag: under the dictionary merge all three go missing
    together, so it is currently redundant with the three, but it stays distinct
    because it names the *reason* rather than the symptom.
    """
    base = build_frozen_3(records)
    out = base.copy()
    out["metal_atomic_number_is_missing"] = base["Atomic Number_metal"].isna().astype(float)
    out["metal_lanthanide_index_is_missing"] = base["lanthanide_index"].isna().astype(float)
    out["metal_ionic_radius_is_missing"] = base["Ionic Radius_metal"].isna().astype(float)
    known = records["metal_symbol"].isin(set(LANTHANIDE_DESCRIPTORS))
    out["metal_not_in_frozen_dict"] = (~known).astype(float)
    assert_no_target_leakage(out)
    return out[list(FROZEN_3_INDICATED_COLUMNS)]


GENERAL_COLUMNS: tuple[str, ...] = (
    "metal_atomic_number",
    "metal_oxidation_state",
    "metal_effective_charge",
    "metal_is_oxo_species",
    "metal_ionic_radius_cn8_A",
    "metal_f_electron_count",
    "metal_is_lanthanide",
    "metal_is_actinide",
    "metal_period",
    "metal_group",
    "metal_block_is_s",
    "metal_block_is_d",
    "metal_block_is_p",
    "metal_block_is_f",
    "metal_oxidation_state_is_missing",
    "metal_ionic_radius_is_missing",
    "metal_ionic_radius_is_assumed_trivalent",
    "metal_f_electron_count_is_missing",
    "metal_symbol_is_unknown",
)


def build_general(records: pd.DataFrame) -> pd.DataFrame:
    """A representation that is defined across the periodic table, with no Ln coordinate.

    Every column is either an element fact (Z, period, group, block), a recorded
    measurement property (oxidation state, radius), a deterministic function of the
    two (charge, f-count), or a missingness indicator.  There is **no**
    ``lanthanide_index``: a coordinate defined as ``Z - 56`` cannot mean anything for
    an actinide, and the lanthanide-contraction information it was carrying is
    already in ``metal_ionic_radius_cn8_A`` and ``metal_f_electron_count``, both of
    which extend to the 5f series by the same physics.

    Columns
    -------
    ``metal_atomic_number``       Z from :data:`ATOMIC_NUMBER`; NaN for an unidentified metal.
    ``metal_oxidation_state``     exactly as the archive recorded it; **never inferred**.
    ``metal_effective_charge``    :func:`effective_charge` (oxidation state, or the
                                  documented actinyl / pertechnetate net charge).
    ``metal_is_oxo_species``      1 where that override fired.
    ``metal_ionic_radius_cn8_A``  archive Shannon CN=8 radius for the recorded
                                  oxidation state.  **One fallback**: a lanthanide row
                                  whose oxidation state was not recorded takes the
                                  trivalent CN=8 radius, flagged by
                                  ``metal_ionic_radius_is_assumed_trivalent``.  This
                                  is exactly the assumption gen10 makes silently for
                                  all 5,992 bundle rows; here it is a column an
                                  ablation can switch off.  It is *externally*
                                  deterministic for La, Pr, Nd, Pm, Gd, Tb, Dy, Ho,
                                  Er, Tm and Lu, which have no other accessible
                                  aqueous state, and an *assumption* for Ce, Sm, Eu
                                  and Yb, which do — see the README.
    ``metal_f_electron_count``    :func:`f_electron_count`; NaN when the f-block
                                  oxidation state is unrecorded.
    ``metal_is_lanthanide``       1 for 57 <= Z <= 71.
    ``metal_is_actinide``         1 for 89 <= Z <= 103.
    ``metal_period``/``metal_group``  IUPAC position; f-block takes the group-3 slot.
    ``metal_block_is_{s,d,p,f}``  one-hot block.
    ``*_is_missing``              explicit indicators; the point of the scheme.
    ``metal_symbol_is_unknown``   1 where the archive never resolved a metal at all.
    """
    _validate(records, "GENERAL")
    symbol = records["metal_symbol"]
    known = symbol.isin(PERIODIC_TABLE)
    ox = pd.to_numeric(records["metal_oxidation_state"], errors="coerce").astype(float)

    out = pd.DataFrame(index=records.index)
    out["metal_atomic_number"] = symbol.map(ATOMIC_NUMBER).astype(float)
    out["metal_oxidation_state"] = ox
    out["metal_effective_charge"] = [
        effective_charge(s, o) for s, o in zip(symbol, ox)]
    out["metal_is_oxo_species"] = [
        is_oxo_species(s, o) for s, o in zip(symbol, ox)]

    radius = pd.to_numeric(records["ionic_radius_cn8_A"], errors="coerce").astype(float)
    z = out["metal_atomic_number"]
    is_ln = (z >= 57) & (z <= 71)
    fallback_ok = radius.isna() & is_ln & symbol.isin(LN3_RADIUS_CN8)
    filled = radius.where(~fallback_ok, symbol.map(LN3_RADIUS_CN8).astype(float))
    out["metal_ionic_radius_cn8_A"] = filled
    out["metal_f_electron_count"] = [
        f_electron_count(s, o) for s, o in zip(symbol, ox)]

    out["metal_is_lanthanide"] = np.where(known, is_ln.astype(float), np.nan)
    is_an = (z >= 89) & (z <= 103)
    out["metal_is_actinide"] = np.where(known, is_an.astype(float), np.nan)

    period = symbol.map({k: v[0] for k, v in PERIODIC_TABLE.items()}).astype(float)
    group = symbol.map({k: v[1] for k, v in PERIODIC_TABLE.items()}).astype(float)
    block = symbol.map({k: v[2] for k, v in PERIODIC_TABLE.items()})
    out["metal_period"] = period
    out["metal_group"] = group
    for b in BLOCKS:
        out[f"metal_block_is_{b}"] = np.where(known, (block == b).astype(float), np.nan)

    out["metal_oxidation_state_is_missing"] = ox.isna().astype(float)
    out["metal_ionic_radius_is_missing"] = filled.isna().astype(float)
    out["metal_ionic_radius_is_assumed_trivalent"] = fallback_ok.astype(float)
    out["metal_f_electron_count_is_missing"] = out["metal_f_electron_count"].isna().astype(float)
    out["metal_symbol_is_unknown"] = (~known).astype(float)

    assert_no_target_leakage(out)
    return out[list(GENERAL_COLUMNS)]


#: Every column of :data:`GENERAL_COLUMNS` that depends on an ionic radius.
RADIUS_DEPENDENT_COLUMNS: tuple[str, ...] = (
    "metal_ionic_radius_cn8_A",
    "metal_ionic_radius_is_missing",
    "metal_ionic_radius_is_assumed_trivalent",
)

NO_RADIUS_COLUMNS: tuple[str, ...] = tuple(
    c for c in GENERAL_COLUMNS if c not in RADIUS_DEPENDENT_COLUMNS)


def build_no_radius(records: pd.DataFrame) -> pd.DataFrame:
    """:func:`build_general` with every radius-dependent column deleted.

    This is the §7 "with and without ionic radius" comparison.  It matters more
    than an ordinary ablation because the radius is the one metal feature that is
    *structurally* absent for whole metals rather than scattered rows: nine of the
    forty archive metals have no tabulated CN=8 radius at any oxidation state, so
    a radius-dependent model cannot place them at all, and a median impute would
    place all nine on top of each other.  If NO_RADIUS matches GENERAL, those nine
    metals are usable; if it does not, the radius is load-bearing and the nine are
    a coverage limit that has to be stated.
    """
    return build_general(records)[list(NO_RADIUS_COLUMNS)]


#: name -> (builder, declared column list).  The registry other gen11 modules use.
REPRESENTATIONS: Mapping[str, tuple] = {
    "FROZEN_3": (build_frozen_3, FROZEN_3_COLUMNS),
    "FROZEN_3_INDICATED": (build_frozen_3_indicated, FROZEN_3_INDICATED_COLUMNS),
    "GENERAL": (build_general, GENERAL_COLUMNS),
    "NO_RADIUS": (build_no_radius, NO_RADIUS_COLUMNS),
}

#: Columns of each scheme that actually carry information (indicators excluded).
#: A row with all of these missing has *no usable representation* under that scheme.
_INFORMATIVE: Mapping[str, tuple[str, ...]] = {
    "FROZEN_3": FROZEN_3_COLUMNS,
    "FROZEN_3_INDICATED": FROZEN_METAL_COLUMNS,
    "GENERAL": tuple(c for c in GENERAL_COLUMNS if not c.endswith(
        ("_is_missing", "_is_unknown", "_is_assumed_trivalent"))),
    "NO_RADIUS": tuple(c for c in NO_RADIUS_COLUMNS if not c.endswith(
        ("_is_missing", "_is_unknown", "_is_assumed_trivalent"))),
}


def build(records: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Dispatch by name; raises on an unknown scheme rather than defaulting."""
    if scheme not in REPRESENTATIONS:
        raise KeyError(f"unknown metal representation {scheme!r}; have {sorted(REPRESENTATIONS)}")
    builder, columns = REPRESENTATIONS[scheme]
    out = builder(records)
    if tuple(out.columns) != tuple(columns):
        raise AssertionError(
            f"{scheme} produced {tuple(out.columns)}, declared {tuple(columns)}")
    return out


def verify_frozen_reproduction(cohort_frame: pd.DataFrame) -> dict:
    """``FROZEN_3`` must rebuild gen10's own METAL columns bit-for-bit.

    Without this the audit would be an argument about a dictionary rather than a
    measurement of the pipeline: if the reproduction is not exact, either the
    copied ``LANTHANIDE_DESCRIPTORS`` has drifted from the builder's or the frozen
    bundle's METAL block came from somewhere else, and both invalidate every
    row-loss number below.
    """
    rebuilt = build_frozen_3(cohort_frame)
    actual = cohort_frame[list(FROZEN_METAL_COLUMNS)].astype(float)
    delta = (rebuilt.to_numpy(dtype=float) - actual.to_numpy(dtype=float))
    worst = float(np.nanmax(np.abs(delta))) if delta.size else 0.0
    nan_mismatch = int((rebuilt.isna().to_numpy() != actual.isna().to_numpy()).sum())
    if worst != 0.0 or nan_mismatch:
        raise SystemExit(
            f"FROZEN_3 does not reproduce the gen10 METAL block "
            f"(max |delta| {worst:.3g}, {nan_mismatch} NaN-pattern mismatches)")
    return {"cohort_rows": int(len(cohort_frame)), "max_abs_delta": worst,
            "nan_pattern_mismatches": nan_mismatch}


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #

#: Provenance-free metal columns the audit needs but ``overlap.ARCHIVE_IDENTITY_COLUMNS``
#: does not carry.  Read separately rather than by widening that allow-list, which is
#: frozen: the allow-list is a leakage control and gen11 does not get to edit it.
_EXTRA_METAL_COLUMNS: tuple[str, ...] = (
    "source_record_id", "metal_oxidation_state_source",
    "metal_oxidation_state_plausible", "metal_species_form",
)


def _auxiliary_a_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(full archive, A_model_ready auxiliary rows, overlap audit)."""
    from ..gen10.runner import prepared_cohort
    from .overlap import ARCHIVE_CLEAN, build_overlap_map

    cohort = prepared_cohort()
    overlap = build_overlap_map(cohort.frame)
    extra = pd.read_parquet(ARCHIVE_CLEAN, columns=list(_EXTRA_METAL_COLUMNS))
    extra["source_record_id"] = extra["source_record_id"].astype(str)
    archive = overlap.archive.merge(extra, on="source_record_id",
                                    how="left", validate="one_to_one")
    if len(archive) != len(overlap.archive):
        raise SystemExit("metal-column join changed the archive row count")
    frozen_ids = set(overlap.exact["source_record_id"])
    aux = archive[~archive["source_record_id"].isin(frozen_ids)]
    aux_a = aux[aux["model_readiness"] == "A_model_ready"]
    return archive, aux_a, overlap.audit


def _coverage_block(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per-metal counts of every metal field, over whatever subset is passed."""
    ox = pd.to_numeric(frame["metal_oxidation_state"], errors="coerce")
    grouped = frame.assign(_ox=ox).groupby(
        frame["metal_symbol"].fillna("<unknown>"), dropna=False)
    out = pd.DataFrame({
        f"{prefix}_rows": grouped.size(),
        f"{prefix}_atomic_number_n": grouped["atomic_number"].count(),
        f"{prefix}_oxidation_state_n": grouped["_ox"].count(),
        f"{prefix}_ionic_radius_n": grouped["ionic_radius_cn8_A"].count(),
        f"{prefix}_lanthanide_index_n": grouped["lanthanide_index"].count(),
    })
    out[f"{prefix}_oxidation_states"] = grouped["_ox"].apply(
        lambda s: "|".join(f"{v:g}" for v in sorted(set(s.dropna()))))
    out[f"{prefix}_ox_state_sources"] = grouped["metal_oxidation_state_source"].apply(
        lambda s: "|".join(sorted(set(s.dropna().astype(str)))))
    out[f"{prefix}_ox_state_implausible_rows"] = grouped[
        "metal_oxidation_state_plausible"].apply(
        lambda s: int((s.dropna().astype(str).str.lower() == "false").sum()))
    out[f"{prefix}_ionic_radius_status"] = grouped["ionic_radius_status"].apply(
        lambda s: "|".join(sorted(set(s.dropna().astype(str)))))
    return out


def metal_coverage_table(archive: pd.DataFrame, aux_a: pd.DataFrame) -> pd.DataFrame:
    """Per-metal coverage over the full archive and over A_model_ready auxiliary rows.

    Charge is reported as the set of ``effective_charge`` values the metal takes,
    which is the column ``GENERAL`` actually feeds a model, not the raw oxidation
    state — the two differ for exactly the actinyl and pertechnetate rows.
    """
    full = _coverage_block(archive, "archive")
    aux = _coverage_block(aux_a, "auxA")
    table = full.join(aux, how="left")
    for col in table.columns:
        if col.endswith(("_rows", "_n")):
            table[col] = table[col].fillna(0).astype(int)
        else:
            table[col] = table[col].fillna("")

    symbols = pd.Series(table.index, index=table.index)
    table.insert(0, "metal_symbol", symbols)
    table.insert(1, "atomic_number", symbols.map(ATOMIC_NUMBER).astype("Float64"))
    cat = archive.groupby(archive["metal_symbol"].fillna("<unknown>"))["metal_category"].first()
    table.insert(2, "metal_category", cat.reindex(table.index).fillna("<none>"))
    table.insert(3, "is_lanthanide", symbols.map(
        lambda s: bool(57 <= ATOMIC_NUMBER.get(s, -1) <= 71)))
    table.insert(4, "is_actinide", symbols.map(
        lambda s: bool(89 <= ATOMIC_NUMBER.get(s, -1) <= 103)))
    table.insert(5, "in_frozen_dict", symbols.isin(set(LANTHANIDE_DESCRIPTORS)))
    table.insert(6, "period", symbols.map(
        {k: v[0] for k, v in PERIODIC_TABLE.items()}).astype("Float64"))
    table.insert(7, "group", symbols.map(
        {k: v[1] for k, v in PERIODIC_TABLE.items()}).astype("Float64"))
    table.insert(8, "block", symbols.map(
        {k: v[2] for k, v in PERIODIC_TABLE.items()}).fillna(""))

    # Charges, from the archive's own oxidation states through the documented rule.
    charges = {}
    fcounts = {}
    for sym, sub in archive.groupby(archive["metal_symbol"].fillna("<unknown>")):
        ox = pd.to_numeric(sub["metal_oxidation_state"], errors="coerce")
        vals = sorted({effective_charge(sym, o) for o in ox.dropna().unique()})
        charges[sym] = "|".join(f"{v:g}" for v in vals)
        fv = sorted({f_electron_count(sym, o) for o in ox.dropna().unique()}) if len(ox.dropna()) \
            else [f_electron_count(sym, float("nan"))]
        fcounts[sym] = "|".join(f"{v:g}" for v in fv if np.isfinite(v))
    table["effective_charges"] = pd.Series(charges).reindex(table.index).fillna("")
    table["f_electron_counts"] = pd.Series(fcounts).reindex(table.index).fillna("")
    return table.sort_values("archive_rows", ascending=False).reset_index(drop=True)


def crosscheck_published_coverage(table: pd.DataFrame) -> pd.DataFrame:
    """Compare the recomputed table with ``dataset_all_metals/reports/metal_coverage.csv``.

    The published report is the archive builder's own accounting.  If gen11's
    recomputation disagrees with it, one of the two is wrong and no downstream
    number is trustworthy, so the disagreement is written out rather than asserted
    away.
    """
    path = REPO_ROOT / "dataset_all_metals" / "reports" / "metal_coverage.csv"
    published = pd.read_csv(path)
    published["metal_symbol"] = published["metal_symbol"].fillna("<unknown>")
    merged = table.merge(
        published[["metal_symbol", "records", "atomic_number", "rows_with_ionic_radius",
                   "category", "ionic_radius_status"]],
        on="metal_symbol", how="outer", suffixes=("_gen11", "_published"), indicator=True)
    merged["rows_match"] = merged["archive_rows"] == merged["records"]
    merged["radius_rows_match"] = merged["archive_ionic_radius_n"] == merged["rows_with_ionic_radius"]
    merged["Z_match"] = (
        merged["atomic_number_gen11"].astype("Float64")
        == merged["atomic_number_published"].astype("Float64"))
    return merged


def frozen_dict_agreement(archive: pd.DataFrame) -> pd.DataFrame:
    """Does the archive reproduce ``LANTHANIDE_DESCRIPTORS`` for the fourteen lanthanides?

    Three questions, kept apart because they have different answers:

    * do the *values* agree where both sides have one (atomic number, index, radius)?
    * does the archive have a value everywhere gen10 does?
    * per metal, how many archive rows would gain a NaN radius by switching to the
      archive column, because the archive refuses a radius when the oxidation state
      was never recorded and gen10 simply assumes trivalency?

    Deltas are reported to more than the three decimals the brief asks for, so a
    rounding-scale disagreement is still visible.
    """
    rows = []
    for symbol, desc in LANTHANIDE_DESCRIPTORS.items():
        sub = archive[archive["metal_symbol"] == symbol]
        radius = pd.to_numeric(sub["ionic_radius_cn8_A"], errors="coerce")
        z = pd.to_numeric(sub["atomic_number"], errors="coerce")
        idx = pd.to_numeric(sub["lanthanide_index"], errors="coerce")
        uniq_r = sorted(set(radius.dropna().round(9)))
        rows.append({
            "metal_symbol": symbol,
            "archive_rows": int(len(sub)),
            "frozen_atomic_number": float(desc["Atomic Number_metal"]),
            "archive_atomic_number_values": "|".join(f"{v:g}" for v in sorted(set(z.dropna()))),
            "atomic_number_delta": round(
                float(abs(z.dropna() - desc["Atomic Number_metal"]).max()) if z.notna().any()
                else float("nan"), 3),
            "frozen_lanthanide_index": float(desc["lanthanide_index"]),
            "archive_lanthanide_index_values": "|".join(f"{v:g}" for v in sorted(set(idx.dropna()))),
            "lanthanide_index_delta": round(
                float(abs(idx.dropna() - desc["lanthanide_index"]).max()) if idx.notna().any()
                else float("nan"), 3),
            "frozen_ionic_radius": float(desc["Ionic Radius_metal"]),
            "archive_ionic_radius_values": "|".join(f"{v:.6g}" for v in uniq_r),
            "ionic_radius_delta": round(
                float(abs(radius.dropna() - desc["Ionic Radius_metal"]).max())
                if radius.notna().any() else float("nan"), 3),
            "archive_ionic_radius_missing_rows": int(radius.isna().sum()),
            "archive_lanthanide_index_missing_rows": int(idx.isna().sum()),
            "archive_atomic_number_missing_rows": int(z.isna().sum()),
        })
    frame = pd.DataFrame(rows)
    frame["values_agree"] = (
        (frame["atomic_number_delta"].fillna(0) == 0)
        & (frame["lanthanide_index_delta"].fillna(0) == 0)
        & (frame["ionic_radius_delta"].fillna(0) == 0))
    frame["coverage_agrees"] = frame["archive_ionic_radius_missing_rows"] == 0
    return frame


def row_loss_by_feature(aux_a: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """How many rows each metal feature costs, per scheme, if reused unchanged.

    "Cost" is counted two ways because gen10 does both: a NaN either survives to
    the median imputer (silent, wrong, and undetectable when the fitting fold had
    no NaNs) or, in any pipeline that drops incomplete rows, deletes the row.
    Both numbers are reported per feature and per scheme; the brief forbids the
    second and this table is the evidence about the first.
    """
    frames = {"auxA": aux_a, "archive": archive}
    records = []
    for scope, frame in frames.items():
        n = len(frame)
        for scheme in REPRESENTATIONS:
            built = build(frame, scheme)
            informative = list(_INFORMATIVE[scheme])
            for col in built.columns:
                nan = int(built[col].isna().sum())
                records.append({
                    "scope": scope, "scheme": scheme, "feature": col,
                    "is_informative": col in informative,
                    "rows": n, "rows_missing": nan,
                    "frac_missing": round(nan / n, 6) if n else float("nan"),
                })
            info = built[informative]
            all_missing = int(info.isna().all(axis=1).sum())
            any_missing = int(info.isna().any(axis=1).sum())
            records.append({
                "scope": scope, "scheme": scheme, "feature": "<ALL informative missing>",
                "is_informative": False, "rows": n, "rows_missing": all_missing,
                "frac_missing": round(all_missing / n, 6) if n else float("nan"),
            })
            records.append({
                "scope": scope, "scheme": scheme, "feature": "<ANY informative missing>",
                "is_informative": False, "rows": n, "rows_missing": any_missing,
                "frac_missing": round(any_missing / n, 6) if n else float("nan"),
            })
    return pd.DataFrame(records)


def unusable_metals(aux_a: pd.DataFrame) -> pd.DataFrame:
    """Per scheme and metal: rows with no informative metal feature at all."""
    records = []
    symbols = aux_a["metal_symbol"].fillna("<unknown>")
    for scheme in REPRESENTATIONS:
        built = build(aux_a, scheme)
        info = built[list(_INFORMATIVE[scheme])]
        dead = info.isna().all(axis=1)
        partial = info.isna().any(axis=1) & ~dead
        grouped = pd.DataFrame({"symbol": symbols.to_numpy(),
                                "dead": dead.to_numpy(), "partial": partial.to_numpy()})
        for sym, sub in grouped.groupby("symbol"):
            records.append({
                "scheme": scheme, "metal_symbol": sym, "auxA_rows": int(len(sub)),
                "rows_no_representation": int(sub["dead"].sum()),
                "rows_partial_representation": int(sub["partial"].sum()),
            })
    return pd.DataFrame(records)


def imputation_damage(cohort_frame: pd.DataFrame, aux_a: pd.DataFrame) -> dict:
    """What a lanthanide-fitted median imputer actually writes onto an actinide row.

    Reproduces gen10's ``SimpleImputer(strategy="median", add_indicator=True)`` fitted
    on the frozen cohort's METAL block, then transforms the auxiliary rows under
    FROZEN_3.  The interesting output is not the imputed values but the indicator
    count: ``add_indicator`` defaults to ``features="missing-only"``, so a fold with
    no missing metal values produces zero indicator columns and the substitution is
    invisible downstream.
    """
    from sklearn.impute import SimpleImputer

    train = cohort_frame[list(FROZEN_METAL_COLUMNS)].astype(float).to_numpy()
    imp = SimpleImputer(strategy="median", add_indicator=True).fit(train)
    aux_block = build_frozen_3(aux_a).to_numpy(dtype=float)
    transformed = imp.transform(aux_block)
    medians = {c: float(v) for c, v in zip(FROZEN_METAL_COLUMNS, imp.statistics_)}
    nearest = min(LANTHANIDE_DESCRIPTORS,
                  key=lambda s: abs(LANTHANIDE_DESCRIPTORS[s]["lanthanide_index"]
                                    - medians["lanthanide_index"]))
    return {
        "cohort_rows_fitted_on": int(len(train)),
        "cohort_metal_block_nan_cells": int(np.isnan(train).sum()),
        "indicator_columns_emitted": int(transformed.shape[1] - len(FROZEN_METAL_COLUMNS)),
        "median_imputed_values": medians,
        "auxA_rows_receiving_them": int(
            build_frozen_3(aux_a)["Atomic Number_metal"].isna().sum()),
        "element_the_imputed_index_names": nearest,
        "note": ("add_indicator=True emits features='missing-only'; a lanthanide-only "
                 "fitting fold has no missing METAL cells, so no indicator is created "
                 "and every auxiliary non-lanthanide row is silently relabelled"),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from ..gen10.runner import prepared_cohort

    archive, aux_a, overlap_audit = _auxiliary_a_rows()
    cohort = prepared_cohort()

    coverage = metal_coverage_table(archive, aux_a)
    coverage.to_csv(OUT_DIR / "metal_coverage.csv", index=False)

    cross = crosscheck_published_coverage(coverage)
    cross.to_csv(OUT_DIR / "metal_coverage_crosscheck.csv", index=False)

    agreement = frozen_dict_agreement(archive)
    agreement.to_csv(OUT_DIR / "frozen_dict_agreement.csv", index=False)

    loss = row_loss_by_feature(aux_a, archive)
    loss.to_csv(OUT_DIR / "row_loss_by_feature.csv", index=False)

    dead = unusable_metals(aux_a)
    dead.to_csv(OUT_DIR / "unusable_metals.csv", index=False)

    damage = imputation_damage(cohort.frame, aux_a)

    schemes = {}
    for name, (_, columns) in REPRESENTATIONS.items():
        built = build(aux_a, name)
        schemes[name] = {
            "columns": list(columns),
            "n_columns": len(columns),
            "informative_columns": list(_INFORMATIVE[name]),
            "indicator_columns": [c for c in columns if c not in _INFORMATIVE[name]],
            "auxA_rows": int(len(aux_a)),
            "auxA_rows_with_no_representation": int(
                built[list(_INFORMATIVE[name])].isna().all(axis=1).sum()),
            "auxA_rows_with_partial_representation": int(
                built[list(_INFORMATIVE[name])].isna().any(axis=1).sum()
                - built[list(_INFORMATIVE[name])].isna().all(axis=1).sum()),
        }
    payload = {
        "overlap_audit": overlap_audit,
        "frozen_block_reproduction": verify_frozen_reproduction(cohort.frame),
        "archive_rows": int(len(archive)),
        "auxA_rows": int(len(aux_a)),
        "distinct_metals_archive": int(archive["metal_symbol"].nunique()),
        "distinct_metals_auxA": int(aux_a["metal_symbol"].nunique()),
        "representations": schemes,
        "imputation_damage": damage,
        "crosscheck_all_row_counts_match": bool(cross["rows_match"].all()),
        "crosscheck_all_radius_counts_match": bool(cross["radius_rows_match"].all()),
        "crosscheck_all_Z_match": bool(cross["Z_match"].fillna(True).all()),
        "frozen_dict_all_values_agree": bool(agreement["values_agree"].all()),
        "frozen_dict_metals_with_missing_archive_radius": int(
            (~agreement["coverage_agrees"]).sum()),
    }
    (OUT_DIR / "representations.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
