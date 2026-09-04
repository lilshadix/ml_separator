"""data-v2: structural-absence flags and the recovered-variable bundle.

Why this module exists
----------------------
The gen7 cohort stores a missing condition value as ``NaN`` and lets whatever
imputer is downstream fill it in.  That collapses two different facts into one
symbol:

* **structural absence** — "this experiment had no phase modifier".  The
  concentration is not unknown, it is *zero*, and the operator knew that before
  the measurement was taken.
* **unrecorded** — "the value was not written down".  The quantity existed; the
  record is silent about it.

What gen7's indicator result does and does not license
------------------------------------------------------
gen7 measured missing-value *indicators* at +0.067 macro MAE, the largest
confirmed effect of that generation — but it also localised that effect, and the
localisation matters here.  On gen7's own arms, ``IND_noligand_all`` (1.0903)
versus ``IND_noligand_none`` (1.0876) shows the indicators contribute **nothing**
once the ligand descriptors are removed; the whole +0.067 comes from the eight
partially-missing ``lig2d__hc__*`` columns, where "null" encodes *this molecule
has no amide*.  Every family in this module is a **condition** family, and gen7
explicitly ruled the condition-side missingness (``cond__contact_time_min``) out
as the source of that gain.

So this module is **not** an extrapolation of the +0.067 result and must not be
sold as one.  Its claim is narrower and does not rest on a prior effect size: a
``NaN`` that means "0 M modifier, and the operator knew it" is a different fact
from a ``NaN`` that means "not written down", the record can tell them apart on
part of the corpus, and an imputer that fills both with a median destroys the
first.  Whether separating them moves the metric is an open question for the
ablation, not something gen7 already answered.

The evidence is not hypothetical.  ``geom_cond__modifier_class=none`` — a gen7
feature — is 1 on 5,006 of the 5,248 cohort rows, and **1,093 of those rows have
a phase modifier**: it is written into the diluent name ("kerosene with 30 vol%
1-octanol") instead of the modifier field.  The solvent parse in
``gen7.recovered`` finds 122 more inside ``cond__diluent__other``.  A flag built
from the modifier column alone is wrong on 24 % of the rows it calls "none".

The three states
----------------
Every condition family gets three mutually exclusive indicator columns::

    sflag__<family>__present      the family has a recorded member / value
    sflag__<family>__absent       the experiment structurally had none
    sflag__<family>__unrecorded   the record is silent; we cannot tell

Exactly one of the three is 1 in every row (asserted in the tests).  The state
codes are :data:`UNRECORDED`, :data:`ABSENT`, :data:`PRESENT`.

Per-family rules, and why they differ
-------------------------------------
``has_additive``
    ``cond__additive__*`` is a one-hot vocabulary over the upstream
    ``Phase_Modifier_Name`` field.  All-zero therefore means "the field was
    blank".  This is the one blank-means-absence reading with a check behind it:
    on the cohort the block agrees with ``rec__has_phase_modifier`` on
    5,248/5,248 rows.  **That agreement is a vocabulary-closure check, not
    independent corroboration** — ``rec__has_phase_modifier`` is
    ``Phase_Modifier_Name.notna()``, i.e. the same upstream column the one-hot
    was built from.  What it rules out is a modifier name falling outside the
    nine-level vocabulary and being silently dropped to all-zero; it cannot rule
    out the upstream field itself being blank when a modifier was used.  That
    residual risk is why ``has_additive`` is labelled *bookkeeping* and the
    chemical question is answered by ``has_phase_modifier`` below.  All-zero ⇒
    ABSENT.

``has_phase_modifier``
    The chemical question, not the bookkeeping one.  PRESENT if the additive
    block fires, **or** the upstream modifier field was populated, **or** a
    modifier concentration was recovered, **or** the diluent is a mixture with a
    polar (alcohol / TBP) component — the case ``geom_cond__modifier_class``
    gets wrong.  ABSENT requires the *recovered* evidence per row
    (``rec__has_phase_modifier`` finite **and** the solvent parse having run):
    those are the only two sources that can see a modifier hidden inside
    ``cond__diluent__other``, and 122 cohort rows are exactly that case.  The
    additive block and the diluent one-hot names can therefore raise PRESENT but
    never license ABSENT — without the recovered join the family reports
    UNRECORDED rather than a confident, wrong absence.  A neat alcohol diluent
    (218 rows with solvent polar fraction 1.0 — 205 of them literally
    ``cond__diluent__1_octanol``, the rest inside ``cond__diluent__other``) is
    the *diluent*, not a modifier, and is not counted.

``has_acid``
    Deliberately *more conservative* than ``has_additive``: the blank-field
    convention could not be verified against an independent upstream column for
    acids, and every experiment has an aqueous phase.  An all-zero acid block
    with a positive recorded acid concentration is PRESENT with acid class
    ``unrecorded``; with a recorded zero it is ABSENT; with nothing at all it is
    UNRECORDED, not ABSENT.

``has_shaking_time`` / ``has_contact_time`` / ``has_temperature`` /
``has_metal_concentration`` / ``has_extractant_concentration``
    A blank here is UNRECORDED.  Shaking happens, a temperature exists, a metal
    is present — the physical quantity cannot be structurally absent.  ABSENT is
    reserved for a value positively recorded as zero, and on the real cohort
    those columns are all-zero (reported honestly rather than manufactured).
    The two timing columns are perfectly complementary in the corpus (1,023 rows
    carry a shaking time, 2,117 carry no contact time, and *no* row carries
    both), so ``sflag__timing_descriptor__*`` records which descriptor the record
    used instead of pretending the other one is structurally absent.

``has_aqueous_complexant``
    ``Holdback_Agent_Name`` is empty in all 48,138 upstream rows, so an aqueous
    masking agent is only ever visible through the extractant *name*.  PRESENT
    when the recovered name evidence matches a water-soluble complexant
    signature; ABSENT when the structure carries exactly one name corpus-wide
    (no ambiguity, no evidence); UNRECORDED when the structure has several names
    but this row shows no aqueous signature.

Sentinel policy (the flags come first)
--------------------------------------
This module **never imputes and never chooses a sentinel**.  It emits the flags
from the *pre-sentinel* frame — ``NaN`` as it comes off the cohort parquet — so
that the flag is a fact about the record and not an artefact of a fill value.
Order matters and is part of the contract: run
:func:`attach_recovered_v2`, then :func:`attach_structural_flags`, then any
imputer.  Handing a pre-imputed frame to :func:`attach_structural_flags` does not
error — nothing in the data can tell a fill value from a measurement — it simply
reports every filled row as PRESENT, which is why the flags must be computed
first.

The policy a downstream model should apply, materialised by the optional helper
:func:`apply_sentinel_policy` into ``sentinel__*`` columns:

===============  ==========================================================
state            numeric value handed to the model
===============  ==========================================================
PRESENT          the recorded value, untouched
ABSENT           ``0.0`` — a modifier that is not there is at 0 M, and that
                 is a measured fact, not a guess
UNRECORDED       ``NaN`` — left for the model's own missing-value handling,
                 with ``sflag__…__unrecorded`` telling it why
===============  ==========================================================

Second species
--------------
``second_species_present`` is a first-class column (see
:func:`attach_recovered_v2`), not a derived afterthought: 414 raw bundle rows
name an extractant that is not the modal name of their own canonical SMILES,
which means an additional, unrecorded chemical species was in the experiment.
``second_species_class`` coarsens the evidence into ``aqueous_complexant`` /
``name_mismatch_other`` / ``none`` / ``unknown`` following the classification in
:mod:`lanthanide_separation.gen7.recovered`.

``log_D`` is never read and never written by anything in this module; both
public functions assert it came through untouched.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RECOVERED_CELLS_PATH = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "recovered_cells.parquet"

TARGET_COLUMN = "log_D"

# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #

UNRECORDED = 0
ABSENT = 1
PRESENT = 2

STATE_SUFFIXES: Mapping[int, str] = {UNRECORDED: "unrecorded", ABSENT: "absent", PRESENT: "present"}

#: Condition families that get the three-state treatment, in emission order.
FLAG_FAMILIES: tuple[str, ...] = (
    "has_acid",
    "has_acid_concentration",
    "has_additive",
    "has_phase_modifier",
    "has_phase_modifier_concentration",
    "has_shaking_time",
    "has_contact_time",
    "has_temperature",
    "has_metal_concentration",
    "has_extractant_concentration",
    "has_aqueous_complexant",
)

#: family -> the numeric column whose sentinel the family's state governs.
FAMILY_VALUE_COLUMNS: Mapping[str, str] = {
    "has_acid_concentration": "cond__acid_concentration_M",
    "has_phase_modifier_concentration": "rec__phase_modifier_concentration_M",
    "has_shaking_time": "rec__shaking_time_min",
    "has_contact_time": "cond__contact_time_min",
    "has_temperature": "cond__temperature_C",
    "has_metal_concentration": "cond__metal_concentration_mM",
    "has_extractant_concentration": "cond__extractant_concentration_M",
}

ACID_CLASSES: tuple[str, ...] = (
    "nitrate", "chloride", "sulfate", "perchlorate", "carboxylate",
    "other", "none", "unrecorded",
)

#: ``cond__acid__<key>`` -> class.  ``hno3_oxalic_acid`` is a nitrate medium with
#: a carboxylate complexant; gen7's ``geom_cond__acid_class`` calls it nitrate and
#: this module agrees, flagging the mixture separately in ``sflag__acid_mixed``.
ACID_CLASS_BY_KEY: Mapping[str, str] = {
    "hno3": "nitrate",
    "hno3_oxalic_acid": "nitrate",
    "hcl": "chloride",
    "hbr": "chloride",
    "h2so4": "sulfate",
    "hclo4": "perchlorate",
    "citric_acid": "carboxylate",
    "lactic_acid": "carboxylate",
    "malonic_acid": "carboxylate",
    "tartaric_acid": "carboxylate",
    "oxalic_acid": "carboxylate",
    "acetic_acid": "carboxylate",
    "glycolic_acid": "carboxylate",
    "edta": "carboxylate",
    "dtpa": "carboxylate",
}

#: acid keys naming more than one acid
_MIXED_ACID_KEYS: frozenset[str] = frozenset({"hno3_oxalic_acid"})

TIMING_DESCRIPTORS: tuple[str, ...] = ("contact", "shaking", "both", "none")

SECOND_SPECIES_CLASSES: tuple[str, ...] = (
    "aqueous_complexant", "name_mismatch_other", "none", "unknown",
)

#: String columns :func:`attach_recovered_v2` adds — labels for reporting, never
#: model input.  The analogue of :data:`STRUCTURAL_LABEL_COLUMNS` on the join side.
RECOVERED_LABEL_COLUMNS: tuple[str, ...] = ("second_species_class",)

#: A diluent one-hot name encoding a *mixture*: either "A with N vol% B" or the
#: "A 0.7, B 0.3" notation, both slugified by the bundle builder.
_MIXTURE_NAME = re.compile(r"(?:_with_)|(?:_0_\d)")

#: Polar components that act as phase modifiers when they are the minor part of
#: a mixed diluent.
_POLAR_TOKENS: tuple[str, ...] = (
    "octanol", "decanol", "dodecanol", "nonanol", "heptanol", "hexanol",
    "exxal", "isodecyl", "isodecanol", "alcohol", "tbp", "dhoa", "dohya",
)


def _flag_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for family in FLAG_FAMILIES:
        for state in (PRESENT, ABSENT, UNRECORDED):
            columns.append(f"sflag__{family}__{STATE_SUFFIXES[state]}")
    columns.append("sflag__acid_class")
    columns.extend(f"sflag__acid_class__{name}" for name in ACID_CLASSES)
    columns.append("sflag__acid_mixed")
    columns.extend(f"sflag__timing_descriptor__{name}" for name in TIMING_DESCRIPTORS)
    columns.append("sflag__second_species_present")
    columns.append("sflag__n_names_for_structure")
    return tuple(columns)


#: Every column :func:`attach_structural_flags` emits, in emission order.
STRUCTURAL_FLAG_COLUMNS: tuple[str, ...] = _flag_columns()

#: The label columns inside :data:`STRUCTURAL_FLAG_COLUMNS` — strings, not model
#: input.  Everything else in the tuple is numeric and free of ``NaN`` except
#: ``sflag__n_names_for_structure``, which is a count and is ``NaN`` when the
#: recovered name evidence is not joined.
STRUCTURAL_LABEL_COLUMNS: tuple[str, ...] = ("sflag__acid_class",)

STRUCTURAL_FLAG_NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    c for c in STRUCTURAL_FLAG_COLUMNS if c not in STRUCTURAL_LABEL_COLUMNS)


# --------------------------------------------------------------------------- #
# Evidence helpers
# --------------------------------------------------------------------------- #

def _block(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in frame.columns if c.startswith(prefix)]


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    """The column as float, or ``None`` when the frame does not carry it."""
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _value_state(values: np.ndarray | None, n: int, *, zero_is_absent: bool = True) -> np.ndarray:
    """PRESENT for a finite non-zero value, ABSENT for a recorded 0, else UNRECORDED."""
    if values is None:
        return np.full(n, UNRECORDED, dtype=np.int8)
    state = np.full(n, UNRECORDED, dtype=np.int8)
    finite = np.isfinite(values)
    state[finite] = PRESENT
    if zero_is_absent:
        state[finite & (values == 0.0)] = ABSENT
    return state


def _mixed_polar_diluent(frame: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """Rows whose organic phase is a mixture containing a polar modifier.

    Two independent sources, unioned: the recovered solvent parse (which sees
    through ``cond__diluent__other``) and the slugified diluent one-hot name.
    Returns ``(is_mixed, evidence_available)``.
    """
    n = len(frame)
    mixed = np.zeros(n, dtype=bool)
    available = False

    n_components = _numeric(frame, "rec__solvent_n_components")
    polar_fraction = _numeric(frame, "rec__solvent_polar_fraction")
    if n_components is not None and polar_fraction is not None:
        available = True
        with np.errstate(invalid="ignore"):
            mixed |= ((n_components > 1) & (polar_fraction > 0.0) & (polar_fraction < 1.0)
                      & np.isfinite(polar_fraction))

    diluents = _block(frame, "cond__diluent__")
    if diluents:
        available = True
        for column in diluents:
            slug = column[len("cond__diluent__"):]
            if _MIXTURE_NAME.search(slug) and any(token in slug for token in _POLAR_TOKENS):
                mixed |= frame[column].to_numpy(dtype=float) > 0
    return mixed, available


def _acid_evidence(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """``(state, class_label, mixed, block_available)`` for the acid identity."""
    n = len(frame)
    columns = _block(frame, "cond__acid__")
    concentration = _numeric(frame, "cond__acid_concentration_M")

    state = np.full(n, UNRECORDED, dtype=np.int8)
    labels = np.full(n, "unrecorded", dtype=object)
    mixed = np.zeros(n, dtype=np.int8)

    if not columns:
        if concentration is not None:
            positive = np.isfinite(concentration) & (concentration > 0)
            state[positive] = PRESENT
            zero = np.isfinite(concentration) & (concentration == 0)
            state[zero] = ABSENT
            labels[zero] = "none"
        return state, labels, mixed, False

    hot = frame[columns].to_numpy(dtype=float) > 0
    n_hot = hot.sum(axis=1)
    keys = [c[len("cond__acid__"):] for c in columns]

    single = n_hot == 1
    if single.any():
        index = hot[single].argmax(axis=1)
        chosen = np.array([keys[i] for i in index], dtype=object)
        labels[single] = np.array(
            [ACID_CLASS_BY_KEY.get(k, "other") for k in chosen], dtype=object)
        mixed[single] = np.array([1 if k in _MIXED_ACID_KEYS else 0 for k in chosen], dtype=np.int8)
    state[n_hot >= 1] = PRESENT
    labels[n_hot > 1] = "other"
    mixed[n_hot > 1] = 1

    none_hot = n_hot == 0
    if none_hot.any():
        if concentration is None:
            # no way to tell "neutral aqueous phase" from "acid not written down"
            state[none_hot] = UNRECORDED
        else:
            positive = none_hot & np.isfinite(concentration) & (concentration > 0)
            state[positive] = PRESENT          # an acid was there; its identity was not
            labels[positive] = "unrecorded"
            zero = none_hot & np.isfinite(concentration) & (concentration == 0)
            state[zero] = ABSENT
            labels[zero] = "none"
    return state, labels, mixed, True


def _second_species_evidence(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(present, class_label, n_names)`` from the recovered name evidence.

    ``rec__name_mismatch`` is the cell mean of a per-replicate 0/1 flag, so any
    value above 0 means at least one replicate of the cell named a species that
    is not the modal name of its own canonical SMILES.
    """
    n = len(frame)
    mismatch = _numeric(frame, "rec__name_mismatch")
    aqueous = _numeric(frame, "rec__aqueous_complexant")
    n_names = _numeric(frame, "rec__n_names_for_structure")

    present = np.zeros(n, dtype=np.int8)
    labels = np.full(n, "unknown", dtype=object)
    if mismatch is None:
        return present, labels, (np.full(n, np.nan) if n_names is None else n_names)

    known = np.isfinite(mismatch)
    hit = known & (mismatch > 0)
    present[hit] = 1
    labels[known & ~hit] = "none"
    if aqueous is None:
        labels[hit] = "name_mismatch_other"
    else:
        aqueous_hit = hit & np.isfinite(aqueous) & (aqueous > 0)
        labels[hit & ~aqueous_hit] = "name_mismatch_other"
        labels[aqueous_hit] = "aqueous_complexant"
    return present, labels, (np.full(n, np.nan) if n_names is None else n_names)


def _one_hot(labels: np.ndarray, categories: Sequence[str], prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}{name}": (labels == name).astype(np.int8) for name in categories}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def attach_structural_flags(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Attach the ``sflag__*`` structural-absence block.

    Row-preserving and index-preserving; deterministic; idempotent (a second call
    recomputes the same values from the same evidence).  Emits exactly
    :data:`STRUCTURAL_FLAG_COLUMNS`, whatever evidence the frame happens to
    carry — a family with no evidence at all is UNRECORDED everywhere, never
    silently ABSENT.

    The frame must be *pre-sentinel*: missing numerics still ``NaN``.  Nothing
    here reads or writes ``log_D``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("attach_structural_flags expects a DataFrame")
    n = len(frame)
    original_index = frame.index
    target_before = frame[TARGET_COLUMN].copy() if TARGET_COLUMN in frame.columns else None

    states: dict[str, np.ndarray] = {}

    # --- acid identity ------------------------------------------------------ #
    acid_state, acid_labels, acid_mixed, _acid_block = _acid_evidence(frame)
    states["has_acid"] = acid_state

    # --- acid concentration: structurally 0 when there is no acid ----------- #
    acid_concentration = _numeric(frame, "cond__acid_concentration_M")
    conc_state = _value_state(acid_concentration, n)
    conc_state[(conc_state == UNRECORDED) & (acid_state == ABSENT)] = ABSENT
    states["has_acid_concentration"] = conc_state

    # --- recorded additive vocabulary --------------------------------------- #
    additive_columns = _block(frame, "cond__additive__")
    if additive_columns:
        hot = frame[additive_columns].to_numpy(dtype=float).sum(axis=1) > 0
        additive_state = np.where(hot, PRESENT, ABSENT).astype(np.int8)
    else:
        additive_state = np.full(n, UNRECORDED, dtype=np.int8)
    states["has_additive"] = additive_state

    # --- the chemical phase modifier ---------------------------------------- #
    mixed, _mixture_evidence = _mixed_polar_diluent(frame)
    recovered_has = _numeric(frame, "rec__has_phase_modifier")
    modifier_concentration = _numeric(frame, "rec__phase_modifier_concentration_M")

    modifier_present = (additive_state == PRESENT) | mixed
    if recovered_has is not None:
        modifier_present |= np.isfinite(recovered_has) & (recovered_has > 0)
    if modifier_concentration is not None:
        modifier_present |= np.isfinite(modifier_concentration) & (modifier_concentration > 0)

    # ABSENT is a positive claim and needs evidence that could have seen a
    # modifier hidden inside ``cond__diluent__other``.  Only the recovered
    # upstream field and the solvent parse can; the additive block and the
    # diluent one-hot names raise PRESENT but never license ABSENT, so a frame
    # without the recovered join reports UNRECORDED instead of a wrong absence.
    modifier_evidence = np.zeros(n, dtype=bool)
    polar_fraction = _numeric(frame, "rec__solvent_polar_fraction")
    if recovered_has is not None and polar_fraction is not None:
        modifier_evidence = np.isfinite(recovered_has) & np.isfinite(polar_fraction)

    modifier_state = np.full(n, UNRECORDED, dtype=np.int8)
    modifier_state[modifier_evidence] = ABSENT
    modifier_state[modifier_present] = PRESENT
    states["has_phase_modifier"] = modifier_state

    modifier_conc_state = _value_state(modifier_concentration, n)
    modifier_conc_state[(modifier_conc_state == UNRECORDED) & (modifier_state == ABSENT)] = ABSENT
    states["has_phase_modifier_concentration"] = modifier_conc_state

    # --- times, temperature, concentrations --------------------------------- #
    shaking = _numeric(frame, "rec__shaking_time_min")
    states["has_shaking_time"] = _value_state(shaking, n)
    contact = _numeric(frame, "cond__contact_time_min")
    states["has_contact_time"] = _value_state(contact, n)
    # a temperature always exists; a blank is silence, never structural absence
    states["has_temperature"] = _value_state(_numeric(frame, "cond__temperature_C"), n,
                                             zero_is_absent=False)
    states["has_metal_concentration"] = _value_state(
        _numeric(frame, "cond__metal_concentration_mM"), n)
    states["has_extractant_concentration"] = _value_state(
        _numeric(frame, "cond__extractant_concentration_M"), n)

    # --- aqueous complexant / second species -------------------------------- #
    second_present, second_labels, n_names = _second_species_evidence(frame)
    aqueous = _numeric(frame, "rec__aqueous_complexant")
    aqueous_state = np.full(n, UNRECORDED, dtype=np.int8)
    if n_names is not None:
        # a structure with exactly one name corpus-wide leaves no room for an
        # unrecorded co-species
        aqueous_state[np.isfinite(n_names) & (n_names <= 1)] = ABSENT
    if aqueous is not None:
        aqueous_state[np.isfinite(aqueous) & (aqueous > 0)] = PRESENT
    states["has_aqueous_complexant"] = aqueous_state

    # --- assemble ----------------------------------------------------------- #
    new: dict[str, np.ndarray] = {}
    for family in FLAG_FAMILIES:
        state = states[family]
        for code in (PRESENT, ABSENT, UNRECORDED):
            new[f"sflag__{family}__{STATE_SUFFIXES[code]}"] = (state == code).astype(np.int8)

    new["sflag__acid_class"] = acid_labels
    new.update(_one_hot(acid_labels, ACID_CLASSES, "sflag__acid_class__"))
    new["sflag__acid_mixed"] = acid_mixed.astype(np.int8)

    contact_present = states["has_contact_time"] == PRESENT
    shaking_present = states["has_shaking_time"] == PRESENT
    timing = np.where(contact_present & shaking_present, "both",
                      np.where(contact_present, "contact",
                               np.where(shaking_present, "shaking", "none"))).astype(object)
    new.update(_one_hot(timing, TIMING_DESCRIPTORS, "sflag__timing_descriptor__"))

    new["sflag__second_species_present"] = second_present.astype(np.int8)
    new["sflag__n_names_for_structure"] = (np.full(n, np.nan) if n_names is None
                                           else np.asarray(n_names, dtype=float))

    out = frame.copy()
    for column in STRUCTURAL_FLAG_COLUMNS:
        out[column] = pd.Series(new[column], index=original_index)

    _assert_row_preserving(frame, out, "attach_structural_flags")
    _assert_target_untouched(target_before, out)
    return out, STRUCTURAL_FLAG_COLUMNS


def attach_recovered_v2(
    frame: pd.DataFrame,
    *,
    path: Path | str = RECOVERED_CELLS_PATH,
    include_nuisance: bool = False,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Join the recovered-variable table on ``row_id`` and derive second species.

    Returns ``(frame, {"RECOVERED_V2": (...), "SECOND_SPECIES": (...)})``.  The
    join is a left join validated many-to-one: the row count and the index are
    asserted unchanged, so a duplicated or missing ``row_id`` upstream raises
    instead of silently changing the cohort.

    ``RECOVERED_V2`` holds the recovered *features* (solvent physics, phase
    modifier concentration, shaking time, organic-phase acid).  ``SECOND_SPECIES``
    holds the name evidence plus the first-class ``second_species_present`` and
    the ``second_species_class__*`` one-hots.  The ``nuisance__*`` keys — DOI,
    batch, curator, extractant name — are never features and are joined only when
    ``include_nuisance=True``, for grouping and deconfounding.
    """
    if "row_id" not in frame.columns:
        raise KeyError("attach_recovered_v2 needs a 'row_id' column to join on")
    already = [c for c in frame.columns if c.startswith("rec__")]
    if already:
        # pandas would suffix these to ``rec__*_x`` / ``rec__*_y`` and the
        # returned block names would point at columns that no longer exist.
        raise ValueError(
            f"attach_recovered_v2 was handed a frame that already carries {len(already)} "
            f"'rec__' columns (e.g. {already[:3]}); joining again would silently suffix them")
    recovered = pd.read_parquet(path)
    if "row_id" not in recovered.columns:
        raise KeyError(f"{path} has no 'row_id' column")
    if recovered["row_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate row_id values; the join would not be one-to-one")
    unmatched = ~frame["row_id"].isin(set(recovered["row_id"]))
    if bool(unmatched.any()):
        # a left join would fill these with NaN and every downstream flag would
        # quietly fall back to the weakest evidence available.
        missing = frame.loc[unmatched, "row_id"].unique()
        raise ValueError(
            f"{int(unmatched.sum())} of {len(frame)} row_id values are not in {path} "
            f"(e.g. {list(missing[:3])}); the join would silently NaN-fill them")

    target_before = frame[TARGET_COLUMN].copy() if TARGET_COLUMN in frame.columns else None
    original_index = frame.index

    feature_columns = [c for c in recovered.columns if c.startswith("rec__")]
    nuisance_columns = [c for c in recovered.columns if c.startswith("nuisance__")]
    keep = ["row_id"] + feature_columns + (nuisance_columns if include_nuisance else [])

    joined = frame.merge(recovered[keep], on="row_id", how="left", validate="many_to_one")
    joined.index = original_index
    _assert_row_preserving(frame, joined, "attach_recovered_v2")

    second_present, second_labels, _ = _second_species_evidence(joined)
    joined["second_species_present"] = pd.Series(second_present.astype(np.int8), index=original_index)
    joined["second_species_class"] = pd.Series(second_labels, index=original_index)
    for name, values in _one_hot(second_labels, SECOND_SPECIES_CLASSES,
                                 "second_species_class__").items():
        joined[name] = pd.Series(values, index=original_index)

    evidence = ("rec__name_mismatch", "rec__aqueous_complexant", "rec__n_names_for_structure")
    blocks = {
        "RECOVERED_V2": tuple(c for c in feature_columns if c not in evidence),
        "SECOND_SPECIES": tuple(c for c in evidence if c in joined.columns)
        + ("second_species_present",)
        + tuple(f"second_species_class__{name}" for name in SECOND_SPECIES_CLASSES),
    }
    _assert_target_untouched(target_before, joined)
    return joined, blocks


def apply_sentinel_policy(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Materialise the documented sentinel policy into ``sentinel__*`` columns.

    Requires :func:`attach_structural_flags` to have run first — the flag is a
    fact about the record and must exist before any fill value is chosen.  For
    each family with a bound value column: PRESENT keeps the value, ABSENT
    becomes ``0.0``, UNRECORDED stays ``NaN``.
    """
    missing = [c for c in STRUCTURAL_FLAG_NUMERIC_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError("apply_sentinel_policy requires attach_structural_flags first; "
                       f"missing {len(missing)} flag columns, e.g. {missing[:3]}")
    out = frame.copy()
    created: list[str] = []
    for family, column in FAMILY_VALUE_COLUMNS.items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float).copy()
        absent = out[f"sflag__{family}__absent"].to_numpy(dtype=bool)
        unrecorded = out[f"sflag__{family}__unrecorded"].to_numpy(dtype=bool)
        values[absent] = 0.0
        values[unrecorded] = np.nan
        name = f"sentinel__{column}"
        out[name] = pd.Series(values, index=out.index)
        created.append(name)
    return out, tuple(created)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _assert_row_preserving(before: pd.DataFrame, after: pd.DataFrame, where: str) -> None:
    if len(after) != len(before):
        raise AssertionError(f"{where} changed the row count: {len(before)} -> {len(after)}")
    if not after.index.equals(before.index):
        raise AssertionError(f"{where} changed the row index")


def _assert_target_untouched(before: pd.Series | None, after: pd.DataFrame) -> None:
    if before is None:
        return
    if TARGET_COLUMN not in after.columns:
        raise AssertionError(f"{TARGET_COLUMN} disappeared")
    if not after[TARGET_COLUMN].equals(before):
        raise AssertionError(f"{TARGET_COLUMN} was modified")


def flag_prevalence(frame: pd.DataFrame) -> pd.DataFrame:
    """Long-form prevalence table of the flag block, for reporting."""
    rows = []
    for column in STRUCTURAL_FLAG_NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        rows.append({
            "column": column,
            "n_nonzero": int((series > 0).sum()),
            "prevalence": float((series > 0).mean()),
            # a fixed schema means some states are empty on any given cohort;
            # say so here rather than letting a constant column look informative
            "constant": bool(series.nunique(dropna=False) <= 1),
        })
    return pd.DataFrame(rows)
