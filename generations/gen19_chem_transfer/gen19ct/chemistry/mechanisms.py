"""``chemistry/mechanisms.py`` -- extraction-mechanism labels for components and systems (brief section 5).

Known chemistry takes priority over heuristics: a component's mechanism comes from its structural
FAMILY (:data:`FAMILY_MECHANISM`, a fixed chemistry table), not from a learned or descriptor-based
rule.  Only a component whose family is ``other`` falls back, and it falls back to ``UNKNOWN`` -- no
acidity-based guess is made.

Component labels (what the molecule does when it is the organic extractant)
    ``ACIDIC_CATION_EXCHANGE``  P-OH / P-SH / COOH acids, metallacarborane anions (H+ form)
    ``NEUTRAL_SOLVATING``       TBP, TOPO, CMPO, amides (mono-, malon-, diglycol-), podands, crowns
    ``ION_PAIR_BASIC``          amines, quaternary ammonium, ionic liquids
    ``CHELATING``               beta-diketones, hydroxyoximes, aminopolycarboxylic acids
    ``SOFT_N_DONOR``            neutral aromatic N-donors (BTP/BTBP/BTPhen, pyridine carboxamides)
    ``UNKNOWN``                 family ``other`` / inorganic salts

A hydrophilic aqueous agent (family overlay ``hydrophilic_aqueous_agent``) keeps the label of its
*core* family (TEDGA -> NEUTRAL_SOLVATING) and is marked ``is_hydrophilic_agent``; the system rules
below decide whether it acts as an extractant or as an aqueous agent in a given row.

System labels (one row = one set of organic extractants, plus row context)
    1. No organic extractant -> ``UNKNOWN`` (basis ``NO_EXTRACTANT_RECORDED``).
    2. A hydrophilic agent recorded in the organic-extractant slot next to at least one lipophilic
       extractant is treated as an aqueous agent (flag ``HYDROPHILIC_AGENT_IN_EXTRACTANT_SLOT``; the
       ``ST*.json`` pattern, e.g. ``TODGA, TWE-18``).  When every recorded extractant is hydrophilic
       they stay extractants (flag ``HYDROPHILIC_ONLY_EXTRACTANT``; e.g. TEDGA in nitrobenzene) -- unless
       every one of them is an ionisable hydrophilic acid or chelator (mechanism ``ACIDIC_CATION_EXCHANGE``
       or ``CHELATING``: HEDTA, CDTA, thiodiglycolic acid).  Such a molecule cannot be the organic-phase
       extractant (insoluble in an alkane diluent), so the recorded structure is an aqueous agent in the
       extractant slot and the real extractant is not recorded: ``UNKNOWN``, basis and flag
       ``AQUEOUS_AGENT_RECORDED_AS_EXTRACTANT``.
    3. A phase modifier whose mechanism is ``ACIDIC_CATION_EXCHANGE`` or ``CHELATING`` (e.g. HDEHP
       recorded as a modifier of TODGA) is promoted to a co-extractant (flag
       ``ACIDIC_MODIFIER_AS_COEXTRACTANT``).  Neutral modifiers (TBP, DHOA, alcohols) only set
       ``has_phase_modifier`` -- the archive's role assignment is respected.  The system *table* keeps
       these rows apart: a system's label is voted over its rows without promoted co-extractants, and the
       co-extractant rows are counted separately (``g19_build_extractants``), because an acidic
       co-extractant (HDEHP with TODGA or DOHyA) adds a cation-exchange route and changes the chemistry
       of the row.
    4. One distinct extractant -> its component label.
    5. Two or more distinct extractants, with A = {ACIDIC_CATION_EXCHANGE, CHELATING} and
       N = {NEUTRAL_SOLVATING, SOFT_N_DONOR}:
       any of A and any of N -> ``SYNERGISTIC``; all in N -> ``MIXED_NEUTRAL``; all in A ->
       ``MIXED_ACIDIC`` (extension label, documented); anything else (an ``ION_PAIR_BASIC`` or
       ``UNKNOWN`` member) -> ``UNKNOWN`` with basis ``UNCLASSIFIED_COMBINATION``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

COMPONENT_MECHANISMS = ("ACIDIC_CATION_EXCHANGE", "NEUTRAL_SOLVATING", "ION_PAIR_BASIC", "CHELATING",
                        "SOFT_N_DONOR", "UNKNOWN")
SYSTEM_MECHANISMS = COMPONENT_MECHANISMS[:-1] + ("SYNERGISTIC", "MIXED_NEUTRAL", "MIXED_ACIDIC", "UNKNOWN")
ACIDIC_SET = frozenset({"ACIDIC_CATION_EXCHANGE", "CHELATING"})
NEUTRAL_SET = frozenset({"NEUTRAL_SOLVATING", "SOFT_N_DONOR"})

#: family -> (mechanism, note).  Chemistry table, not a fitted rule.
FAMILY_MECHANISM: dict[str, tuple[str, str]] = {
    "phosphoric_acid": ("ACIDIC_CATION_EXCHANGE", "dialkyl phosphoric acids exchange H+ for M(n+) (D2EHPA)"),
    "phosphonic_acid_monoester": ("ACIDIC_CATION_EXCHANGE", "PC88A / P507 cation exchange"),
    "phosphonic_acid": ("ACIDIC_CATION_EXCHANGE", "RP(O)(OH)2 cation exchange"),
    "phosphinic_acid": ("ACIDIC_CATION_EXCHANGE", "Cyanex 272 cation exchange"),
    "thiophosphorus_acid": ("ACIDIC_CATION_EXCHANGE", "thio-P acids (Cyanex 302 type) cation exchange"),
    "dithiophosphinic_acid": ("ACIDIC_CATION_EXCHANGE", "Cyanex 301 type soft-S acidic exchange"),
    "carboxylic_acid": ("ACIDIC_CATION_EXCHANGE", "carboxylic acids cation exchange"),
    "metallacarborane_anion": ("ACIDIC_CATION_EXCHANGE", "COSAN in H+ form: lipophilic-anion cation exchange"),
    "aminopolycarboxylic_acid": ("CHELATING", "polyaminocarboxylates chelate (normally aqueous holdback)"),
    "acylpyrazolone": ("CHELATING", "4-acyl-5-pyrazolone (HPMBP) chelates as its enolate, acidic"),
    "hydroxyquinoline": ("CHELATING", "8-hydroxyquinoline N,O chelate as its phenolate"),
    "hydroxyketone_catechol": ("CHELATING", "alpha-hydroxy-ketone / catechol O,O chelate (quercetin)"),
    "beta_diketone": ("CHELATING", "enolisable beta-diketone chelates as its anion (HTTA)"),
    "hydroxyoxime": ("CHELATING", "hydroxyoxime chelates as its anion (LIX)"),
    "neutral_organophosphate": ("NEUTRAL_SOLVATING", "TBP solvates neutral metal-nitrate species"),
    "phosphonate_phosphinate_neutral": ("NEUTRAL_SOLVATING", "neutral P=O solvation"),
    "phosphine_oxide": ("NEUTRAL_SOLVATING", "TOPO solvation"),
    "carbamoylmethylphosphine_oxide": ("NEUTRAL_SOLVATING", "CMPO bidentate neutral solvation"),
    "amide_phosphoryl_hybrid": ("NEUTRAL_SOLVATING", "neutral P=O / C=O donor hybrid"),
    "diglycolamide": ("NEUTRAL_SOLVATING", "tridentate O,O,O neutral solvation (TODGA)"),
    "malonamide": ("NEUTRAL_SOLVATING", "bidentate diamide neutral solvation (DMDOHEMA)"),
    "monoamide": ("NEUTRAL_SOLVATING", "monoamide neutral solvation (DEHiBA)"),
    "polyamide_other": ("NEUTRAL_SOLVATING", "neutral amide solvation"),
    "amino_polyamide": ("NEUTRAL_SOLVATING", "tripodal amide; the central amine may protonate (INFERRED: solvation dominant)"),
    "podand_ether": ("NEUTRAL_SOLVATING", "open-chain polyether amide (DOODA) neutral solvation"),
    "crown_ether_calixarene": ("NEUTRAL_SOLVATING", "macrocycle forms a neutral complex with co-extracted anions"),
    "sulfur_donor_other": ("NEUTRAL_SOLVATING", "neutral soft S donor (phosphine sulfide / thioamide)"),
    "aliphatic_alcohol": ("NEUTRAL_SOLVATING", "weak O donor; normally a phase modifier"),
    "n_heterocyclic": ("SOFT_N_DONOR", "neutral aromatic N donor (BTBP / BTPhen)"),
    "pyridine_carboxamide": ("SOFT_N_DONOR", "mixed aromatic-N / amide-O neutral donor"),
    "amine_basic": ("ION_PAIR_BASIC", "protonated amine extracts anionic metal complexes"),
    "quaternary_ammonium": ("ION_PAIR_BASIC", "Aliquat 336 anion exchange"),
    "ionic_liquid": ("ION_PAIR_BASIC", "organic cation / anion exchange"),
    "electrolyte_salt": ("UNKNOWN", "inorganic salt, not an extractant"),
    "other": ("UNKNOWN", "no family rule fired"),
}


def family_mechanism(family: str | None) -> tuple[str, str]:
    """``(mechanism, basis)`` for a family label; ``UNKNOWN`` for unmapped / missing labels."""
    if family is None or family not in FAMILY_MECHANISM:
        return "UNKNOWN", f"FAMILY_NOT_MAPPED:{family}"
    mech, _ = FAMILY_MECHANISM[family]
    return mech, f"FAMILY:{family}"


def component_mechanism(family: str, core_family: str | None = None) -> dict[str, Any]:
    """Mechanism of one component structure.  ``family`` may be an overlay label
    (``hydrophilic_aqueous_agent`` / ``electrolyte_salt``); the core family then decides."""
    hydrophilic = family == "hydrophilic_aqueous_agent"
    label = core_family if hydrophilic and core_family else family
    mech, basis = family_mechanism(label)
    return {"mechanism": mech, "mechanism_basis": basis, "is_hydrophilic_agent": hydrophilic,
            "is_electrolyte_salt": family == "electrolyte_salt"}


@dataclass(frozen=True)
class Entry:
    """One chemically active component in a row, as the system rules see it."""
    key: str                      # structure key (canonical SMILES) or ``name:<name>``
    mechanism: str
    family: str                   # core family used for system_family
    is_hydrophilic_agent: bool = False
    role: str = "organic_extractant"
    basis: str = ""


@dataclass
class SystemMechanism:
    mechanism: str
    basis: str
    effective_extractants: tuple[str, ...]
    system_family: str
    flags: tuple[str, ...] = field(default_factory=tuple)


def system_mechanism(extractants: Iterable[Entry], modifiers: Iterable[Entry] = ()) -> SystemMechanism:
    """Apply the system rules of the module docstring to one row's components."""
    ext = list(extractants)
    flags: list[str] = []
    if not ext:
        return SystemMechanism("UNKNOWN", "NO_EXTRACTANT_RECORDED", (), "none", ())
    lipo = [e for e in ext if not e.is_hydrophilic_agent]
    if lipo and len(lipo) < len(ext):
        flags.append("HYDROPHILIC_AGENT_IN_EXTRACTANT_SLOT")
        ext = lipo
    elif not lipo:
        flags.append("HYDROPHILIC_ONLY_EXTRACTANT")
        if all(e.mechanism in ACIDIC_SET for e in ext):
            flags.append("AQUEOUS_AGENT_RECORDED_AS_EXTRACTANT")
            keys = tuple(sorted({e.key for e in ext}))
            family = "+".join(sorted({e.family for e in ext}))
            return SystemMechanism("UNKNOWN", "AQUEOUS_AGENT_RECORDED_AS_EXTRACTANT", keys, family, tuple(flags))
    promoted = [m for m in modifiers if m.mechanism in ACIDIC_SET]
    if promoted:
        flags.append("ACIDIC_MODIFIER_AS_COEXTRACTANT")
    uniq: dict[str, Entry] = {}
    for e in ext + promoted:
        uniq.setdefault(e.key, e)
    entries = [uniq[k] for k in sorted(uniq)]
    family = "+".join(sorted({e.family for e in entries}))
    mechs = {e.mechanism for e in entries}
    keys = tuple(e.key for e in entries)
    if len(entries) == 1:
        e = entries[0]
        return SystemMechanism(e.mechanism, "SINGLE:" + (e.basis or f"FAMILY:{e.family}"), keys, family,
                               tuple(flags))
    combo = "+".join(sorted(mechs))
    if mechs & ACIDIC_SET and mechs & NEUTRAL_SET and mechs <= (ACIDIC_SET | NEUTRAL_SET):
        return SystemMechanism("SYNERGISTIC", "COMBINATION:" + combo, keys, family, tuple(flags))
    if mechs <= NEUTRAL_SET:
        return SystemMechanism("MIXED_NEUTRAL", "COMBINATION:" + combo, keys, family, tuple(flags))
    if mechs <= ACIDIC_SET:
        return SystemMechanism("MIXED_ACIDIC", "COMBINATION:" + combo, keys, family, tuple(flags))
    return SystemMechanism("UNKNOWN", "UNCLASSIFIED_COMBINATION:" + combo, keys, family, tuple(flags))


def vote(labels: Mapping[str, int]) -> tuple[str, str]:
    """Majority label from ``{label: n_rows}`` and a ``label:n`` vote string (ties -> sorted first,
    with ``_TIE`` noted in the string)."""
    if not labels:
        return "UNKNOWN", ""
    items = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [k for k, v in items if v == items[0][1]]
    votes = ";".join(f"{k}:{v}" for k, v in items)
    return top[0], votes + (";_TIE" if len(top) > 1 else "")
