"""Stage 2 -- normalise experimental identity, preserving every raw value.

Two archive quirks drive the design here.

1.  The structure columns are more trustworthy than the name columns.  A single
    canonical SMILES (TODGA's) is attached to 22 unrelated extractant *names*
    because one sub-source recorded the water-soluble masking agent's name next
    to the organic extractant's structure.  Identity is therefore keyed on
    structure, and every name/structure disagreement is flagged rather than
    resolved.
2.  Multi-component systems are stored as parallel comma-separated lists whose
    element order is not stable between the name, SMILES and concentration
    columns.  Components are resolved through a consensus name->structure map
    built from the 14,459 unambiguous single-component records, and only fall
    back to positional pairing when that lookup fails.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sae_chem as CHEM                                        # noqa: E402
import sae_normalize as NORM                                   # noqa: E402
from sae_paths import ROOT, INTERMEDIATE_DIR, AUDIT_DIR, ensure_dirs  # noqa: E402
from sae_structures import canonicalize, RDKIT_VERSION          # noqa: E402


#: Cached Crossref lookups, produced once by ``fetch_crossref.py``.  Reading a
#: file rather than the network is what keeps the pipeline deterministic and
#: runnable offline.  Absent cache -> the reference_* columns are simply empty.
CROSSREF_CACHE = ROOT / "cache" / "crossref_metadata.json"


def load_crossref() -> dict[str, dict]:
    if not CROSSREF_CACHE.exists():
        return {}
    return json.loads(CROSSREF_CACHE.read_text())


#: |log10 D| beyond which a reported distribution ratio is treated as a
#: digitisation artefact rather than a measurement.  Flag only -- never a filter.
EXTREME_LOG_D = 6.0


# --------------------------------------------------------------------------
# Consensus name -> structure map
# --------------------------------------------------------------------------

def build_name_structure_consensus(records: pd.DataFrame) -> tuple[dict[str, str], dict]:
    """Map extractant name -> canonical SMILES using unambiguous records only.

    Only records carrying exactly one name and one SMILES contribute, because
    those cannot be mis-paired by construction.
    """
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for name_field, smiles_field in zip(records["Extractant_Name"], records["Extractant_SMILES"]):
        names = NORM.split_components(name_field)
        smiles = NORM.split_components(smiles_field)
        if len(names) != 1 or len(smiles) != 1:
            continue
        structure = canonicalize(smiles[0])
        if structure.status == "ok":
            votes[NORM.clean_text(names[0])][structure.canonical] += 1

    consensus, ambiguous = {}, {}
    for name, counter in votes.items():
        if len(counter) == 1:
            consensus[name] = next(iter(counter))
        else:
            ranked = counter.most_common()
            # Only accept a winner that is at least 5x more common than the
            # runner-up; otherwise the name is genuinely contested.
            if ranked[0][1] >= 5 * ranked[1][1]:
                consensus[name] = ranked[0][0]
            ambiguous[name] = {s: int(n) for s, n in ranked}

    return consensus, {
        "names_with_consensus": len(consensus),
        "names_contested": len(ambiguous),
        "contested_detail": ambiguous,
    }


def deduce_by_elimination(records: pd.DataFrame,
                          consensus: dict[str, str]) -> tuple[dict[str, str], dict]:
    """Recover structures for names the archive never gives a SMILES for.

    ``TBP`` and ``DHOA`` carry no SMILES in any of their 972 / 505 single-
    component records, yet both appear inside two-component records such as
    ``"TODGA, TBP"`` whose SMILES list *does* contain them.  When every other
    name in such a record already has a consensus structure, the remaining
    SMILES must belong to the remaining name.  That is a deduction from the
    archive's own contents, not an external assumption, so the recovered
    structure is tagged ``deduced_by_elimination`` and can be audited.
    """
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for name_field, smiles_field in zip(records["Extractant_Name"], records["Extractant_SMILES"]):
        names = [NORM.clean_text(n) for n in NORM.split_components(name_field)]
        smiles = NORM.split_components(smiles_field)
        if len(names) < 2 or len(names) != len(smiles):
            continue
        canon = [canonicalize(s) for s in smiles]
        if any(c.status != "ok" for c in canon):
            continue
        pool = [c.canonical for c in canon]
        unknown = [n for n in names if n not in consensus]
        if len(unknown) != 1:
            continue
        for name in names:
            if name in consensus and consensus[name] in pool:
                pool.remove(consensus[name])
        if len(pool) == 1:
            votes[unknown[0]][pool[0]] += 1

    deduced, rejected = {}, {}
    for name, counter in votes.items():
        if len(counter) == 1:
            deduced[name] = next(iter(counter))
        else:
            rejected[name] = {k: int(v) for k, v in counter.most_common()}
    return deduced, {
        "names_deduced": sorted(deduced),
        "deduced_structures": deduced,
        "names_rejected_ambiguous": rejected,
    }


def build_structure_name_index(consensus: dict[str, str],
                               records: pd.DataFrame) -> dict[str, list[str]]:
    """canonical SMILES -> names using it, most-used name first."""
    usage = collections.Counter(
        NORM.clean_text(n)
        for field in records["Extractant_Name"]
        for n in NORM.split_components(field)
    )
    index: dict[str, list[str]] = collections.defaultdict(list)
    for name, smiles in consensus.items():
        index[smiles].append(name)
    return {k: sorted(v, key=lambda n: (-usage.get(n, 0), n)) for k, v in index.items()}


# --------------------------------------------------------------------------
# Component resolution
# --------------------------------------------------------------------------

#: Names whose structure the archive never states directly and which were
#: recovered by elimination.  Populated by main() before any row is built.
DEDUCED_NAMES: set[str] = set()


def resolve_extractants(name_field, smiles_field, conc_field,
                        consensus: dict[str, str], log: NORM.NormalizationLog) -> dict:
    """Return the organic-extractant component list plus alignment diagnostics."""
    names = NORM.split_components(name_field)
    smiles = NORM.split_components(smiles_field)
    concs = NORM.split_components(conc_field)

    canon = [canonicalize(s) for s in smiles]
    canon_ok = [c.canonical for c in canon if c.status == "ok"]
    failures = [c for c in canon if c.status == "parse_failure"]

    n = max(len(names), len(smiles), len(concs))
    components, flags = [], []

    # Prefer the consensus structure for each *name*; this removes the
    # name/SMILES ordering ambiguity entirely whenever it succeeds.
    consensus_hits = [consensus.get(NORM.clean_text(x)) for x in names]
    positional_set = {c for c in canon_ok}
    consensus_set = {c for c in consensus_hits if c}

    use_consensus = (
        len(names) == len(smiles)
        and len(names) > 0
        and all(consensus_hits)
        and consensus_set == positional_set
    )
    if use_consensus and len(names) > 1:
        log.record("extractant.component_order_resolved_by_consensus", name_field, smiles_field)
    if (not use_consensus) and len(names) > 1 and consensus_set and consensus_set != positional_set:
        flags.append("component_name_structure_mismatch")
        log.record("extractant.component_name_structure_mismatch", name_field, smiles_field)

    for i in range(n):
        name = names[i] if i < len(names) else None
        raw_smiles = smiles[i] if i < len(smiles) else None
        structure = None
        source = None
        if use_consensus and i < len(consensus_hits):
            key = NORM.clean_text(name) if name else None
            structure = consensus_hits[i]
            source = ("deduced_by_elimination" if key in DEDUCED_NAMES else "name_consensus")
        elif i < len(canon) and canon[i].status == "ok":
            structure, source = canon[i].canonical, "positional_smiles"
        elif name is not None and consensus.get(NORM.clean_text(name)):
            key = NORM.clean_text(name)
            # Separate tag for the four structures the archive never states
            # directly.  Previously these shared the "name_consensus" tag with
            # 14,942 ordinary archive-confirmed lookups, so the documented
            # "exclude by structure_source" recipe would have removed 88% of
            # the dataset instead of these four.
            source = ("deduced_by_elimination" if key in DEDUCED_NAMES
                      else "name_consensus_fallback")
            structure = consensus[key]
            log.record("extractant.structure_filled_from_name_consensus", name, structure)

        quantity = NORM.parse_quantity(concs[i], "concentration", "M") if i < len(concs) else None
        components.append({
            "role": "organic_extractant",
            "name": name,
            "smiles_raw": raw_smiles,
            "smiles_canonical": structure,
            "structure_source": source,
            "concentration_M": quantity.value if quantity else None,
            "concentration_raw": concs[i] if i < len(concs) else None,
        })

    if len(names) != len(smiles) and names and smiles:
        flags.append("component_list_length_mismatch")
        log.record("extractant.list_length_mismatch", f"{name_field} || {smiles_field}", None)

    return {
        "components": components,
        "flags": flags,
        "n_names": len(names),
        "n_smiles": len(smiles),
        "n_conc": len(concs),
        "parse_failures": [f.raw for f in failures],
    }


def resolve_aqueous_list(role, name_field, smiles_field, conc_field, log) -> list[dict]:
    """Expand a semicolon-separated aqueous-additive field into components.

    166 records record two aqueous complexants as ``"DTPA; NaNO3"`` with the
    parallel concentrations ``"0.02;1"``.  Passing that whole string to the
    single-quantity parser produced ``None``, which silently deleted the
    complexant concentration from the identity key -- and merged a DTPA
    concentration sweep into one "exact duplicate", folding away the very
    variable the source study was varying.  Each component is now resolved on
    its own, and any component whose concentration will not parse is flagged
    rather than dropped.
    """
    names = NORM.split_components(name_field, separators=";")
    smiles = NORM.split_components(smiles_field, separators=";")
    concs = NORM.split_components(conc_field, separators=";")
    width = max(len(names), len(smiles), len(concs))
    if width == 0:
        return []

    components = []
    for index in range(width):
        name = names[index] if index < len(names) else None
        # The archive supplies a structure only for the first (organic)
        # component when two are listed; the rest stay name-only.
        raw_smiles = smiles[index] if index < len(smiles) else None
        raw_conc = concs[index] if index < len(concs) else None
        structure = canonicalize(raw_smiles)
        quantity = NORM.parse_quantity(raw_conc, "concentration", "M")
        if raw_conc is not None and quantity is None:
            log.record(f"{role}.concentration_unparsed", raw_conc, None)
        if name is None and structure.status != "ok" and quantity is None:
            continue
        components.append({
            "role": role,
            "name": NORM.clean_text(name),
            "smiles_raw": structure.raw,
            "smiles_canonical": structure.canonical,
            "structure_source": "archive" if structure.status == "ok" else None,
            "concentration_M": quantity.value if quantity else None,
            "concentration_raw": NORM.clean_text(raw_conc),
        })
    if width > 1:
        log.record(f"{role}.multi_component_expanded", name_field, f"{width} components")
    return components


def _aqueous_component(role, name, smiles, conc, log) -> dict | None:
    name = NORM.clean_text(name)
    structure = canonicalize(smiles)
    quantity = NORM.parse_quantity(conc, "concentration", "M")
    if name is None and structure.status != "ok" and quantity is None:
        return None
    # A recorded concentration of exactly zero means the agent was absent in
    # this run; it is kept as an explicit zero rather than dropped.
    return {
        "role": role,
        "name": name,
        "smiles_raw": structure.raw,
        "smiles_canonical": structure.canonical,
        "structure_source": "archive" if structure.status == "ok" else None,
        "concentration_M": quantity.value if quantity else None,
        "concentration_raw": NORM.clean_text(conc),
    }


def classify_system(components: list[dict]) -> tuple[str, int]:
    """Label the multi-component character of the system."""
    active = [c for c in components
              if not (c["role"] != "organic_extractant"
                      and c["concentration_M"] is not None and c["concentration_M"] == 0.0)]
    organics = [c for c in active if c["role"] == "organic_extractant"]
    modifiers = [c for c in active if c["role"] == "phase_modifier"]
    aqueous = [c for c in active if c["role"] in ("aqueous_complexant", "aqueous_holdback")]

    # The archive has no SMILES column for phase modifiers at all, so a missing
    # modifier structure says nothing about this record's quality.  Only an
    # unresolved *extractant* structure downgrades the label.
    unresolved = any(c["smiles_canonical"] is None for c in organics)
    if not organics:
        # 226 records name no extractant at all.  Calling them
        # "SINGLE_EXTRACTANT" claimed a component that does not exist.
        return "NO_EXTRACTANT_RECORDED", len(active)
    if len(organics) > 1:
        label = "SYNERGISTIC_TWO_EXTRACTANT" if len(organics) == 2 else "MULTI_EXTRACTANT"
    elif aqueous:
        label = "EXTRACTANT_PLUS_AQUEOUS_AGENT"
    elif modifiers:
        label = "EXTRACTANT_PLUS_MODIFIER"
    else:
        label = "SINGLE_EXTRACTANT"
    if unresolved:
        label += "_STRUCTURE_UNRESOLVED"
    return label, len(active)


# --------------------------------------------------------------------------
# Metal
# --------------------------------------------------------------------------

def resolve_metal(metal_raw, ox_raw, log: NORM.NormalizationLog) -> dict:
    token = NORM.clean_text(metal_raw)
    species = None
    symbol = None
    ox = None
    ox_source = None

    if token is not None:
        if token in CHEM.SPECIES_TOKENS:
            symbol, ox, species = CHEM.SPECIES_TOKENS[token]
            ox_source = "species_definition"
            log.record("metal.species_token_expanded", token, f"{symbol}({ox})")
        elif token in CHEM.ATOMIC_NUMBER:
            symbol = token
        else:
            log.record("metal.unrecognised_token", token, None)

    ox_text = NORM.clean_text(ox_raw)
    if ox_text is not None:
        parsed = CHEM.ROMAN_TO_INT.get(ox_text.upper())
        if parsed is not None:
            if ox is not None and parsed != ox:
                log.record("metal.oxidation_state_conflicts_with_species", f"{token}/{ox_text}", str(ox))
            else:
                ox, ox_source = parsed, "archive"
        else:
            log.record("metal.oxidation_state_unparsed", ox_text, None)

    radius, radius_status = (None, "unknown_metal")
    plausible = None
    if symbol is not None:
        radius, radius_status = CHEM.ionic_radius_cn8(symbol, ox)
        if ox is not None:
            allowed = CHEM.PLAUSIBLE_OX.get(symbol)
            plausible = None if allowed is None else (ox in allowed)
            if plausible is False:
                log.record("metal.implausible_oxidation_state", f"{symbol}({ox})", None)

    return {
        "metal_symbol": symbol,
        "metal_raw": token,
        "metal_species_form": species,
        "metal_oxidation_state": ox,
        "metal_oxidation_state_raw": ox_text,
        "metal_oxidation_state_source": ox_source,
        "atomic_number": CHEM.ATOMIC_NUMBER.get(symbol) if symbol else None,
        "is_lanthanide": (symbol in CHEM.LANTHANIDES) if symbol else None,
        "lanthanide_index": CHEM.lanthanide_index(symbol) if symbol else None,
        "metal_category": CHEM.metal_category(symbol) if symbol else None,
        "ionic_radius_cn8_A": radius,
        "ionic_radius_status": radius_status,
        "metal_oxidation_state_plausible": plausible,
    }


# --------------------------------------------------------------------------
# Row builder
# --------------------------------------------------------------------------

def build_row(rec, consensus, structure_names, crossref, log: NORM.NormalizationLog) -> dict:
    comments = NORM.parse_comments(rec["comments_description"])
    flags: list[str] = []

    metal = resolve_metal(rec["Metal_Name"], rec["Metal_Oxidation_state"], log)

    extract = resolve_extractants(
        rec["Extractant_Name"], rec["Extractant_SMILES"],
        rec["Extractant_Concentration_M"], consensus, log)
    components = list(extract["components"])
    flags.extend(extract["flags"])

    modifier = _aqueous_component(
        "phase_modifier", rec["Phase_Modifier_Name"], None,
        rec["Phase_Modifier_Concentration_M"], log)
    if modifier:
        components.append(modifier)

    complexants = resolve_aqueous_list(
        "aqueous_complexant", comments.get("Complexant_Name"),
        comments.get("Complexant_SMILES"), comments.get("Complexant_Concentration_M"), log)
    components.extend(complexants)
    complexant = complexants[0] if complexants else None

    holdback = _aqueous_component(
        "aqueous_holdback", rec["Holdback_Agent_Name"] or None,
        comments.get("Holdback_Agent_SMILES"),
        comments.get("Holdback_Agent_Concentration_M") or rec["Holdback_Agent_Concentration_M"], log)
    if holdback:
        components.append(holdback)

    # Name/structure disagreement against the global consensus.
    conflicting: list[dict] = []
    for comp in components:
        if comp["role"] != "organic_extractant":
            continue
        name = NORM.clean_text(comp["name"])
        expected = consensus.get(name) if name else None
        if expected and comp["smiles_canonical"] and expected != comp["smiles_canonical"]:
            flags.append("extractant_name_structure_conflict")
            # Record WHICH component disagrees.  The flag is per-record, so a
            # two-component row was previously filed under its first-listed
            # name (usually TODGA, whose own structure is perfectly
            # consistent) instead of the component actually in conflict.
            conflicting.append({
                "name": name,
                "structure_in_record": comp["smiles_canonical"],
                "structure_expected": expected,
            })
            log.record("extractant.name_structure_conflict", f"{name} -> {comp['smiles_canonical']}", expected)
        # A structure shared by several names is only suspicious for the names
        # that are NOT its dominant owner.  Flagging every TODGA row because 21
        # other labels borrowed TODGA's SMILES would bury the real signal.
        shared = structure_names.get(comp["smiles_canonical"] or "", [])
        if len(shared) > 1 and name and name != shared[0]:
            flags.append("structure_shared_by_multiple_names")

    system_class, n_active = classify_system(components)

    organics = [c for c in components if c["role"] == "organic_extractant"]
    resolved = sorted(c["smiles_canonical"] for c in organics if c["smiles_canonical"])
    extractant_system_key = "|".join(resolved) if resolved else None

    acids = NORM.normalize_acids(rec["Acid_Name"], log)
    solvent_components, solvent_pattern = NORM.parse_solvent(rec["Solvent_Name"], log)

    acid_conc = NORM.parse_quantity(rec["Acid_Concentration_M"], "concentration", "M")
    acid_org = NORM.parse_quantity(rec["Acid_Concentration_Organic_M"], "concentration", "M")
    # Metal concentration is declared in mM by the column name.
    metal_conc = NORM.parse_quantity(rec["Metal_Concentration_mM"], "concentration", "mM")
    if metal_conc is None and comments.get("Metal_Concentration_mM"):
        metal_conc = NORM.parse_quantity(comments["Metal_Concentration_mM"], "concentration", "mM")
        if metal_conc is not None:
            log.record("metal_concentration.recovered_from_comments",
                       comments["Metal_Concentration_mM"], metal_conc.value)
    temperature = NORM.parse_quantity(rec["obsTempsValue"], "temperature")
    contact = NORM.parse_quantity(rec["Contact_Time_min"], "time", "min")
    shaking = NORM.parse_quantity(rec["Shaking_Time_min"], "time", "min")
    phase_ratio = NORM.parse_quantity(rec["volValue"], "ratio")
    nitrate = NORM.parse_quantity(comments.get("nitrate concentration(M)"), "concentration", "M")

    d_value = NORM.parse_quantity(rec["obsDvaluesValue"], "ratio")
    d_val = d_value.value if d_value else None
    log_d = None
    if d_val is not None:
        if d_val > 0:
            import math
            log_d = math.log10(d_val)
            # Distribution ratios in solvent extraction are not measurable
            # beyond roughly six orders of magnitude either way.  Values past
            # that are digitisation artefacts, not measurements -- three
            # records share the bit-identical D = 3.41060513164848e-13 across
            # three *different* extractants read off the same figure.  The
            # value is kept exactly as recorded and flagged, never clipped.
            if abs(log_d) > EXTREME_LOG_D:
                flags.append("implausible_extreme_D")
                log.record("target.implausible_extreme_D", rec["obsDvaluesValue"], log_d)
        else:
            flags.append("non_positive_D")
            log.record("target.non_positive_D", rec["obsDvaluesValue"], None)

    dois, other_refs = NORM.normalize_dois(rec["DOI"])
    archive_citation = (NORM.ARCHIVE_SELF_CITATION_DOI
                        if NORM.ARCHIVE_SELF_CITATION_DOI in dois else None)
    source_dois = [d for d in dois if d != NORM.ARCHIVE_SELF_CITATION_DOI]

    row = {
        # ---- provenance (never a predictive feature) ----------------------
        "canonical_measurement_id": rec["canonical_measurement_id"],
        "source_record_id": rec["source_record_id"],
        "raw_row_ids": list(rec["raw_row_ids"]),
        "export_source_files": list(rec["export_source_files"]),
        "export_metals_queried": list(rec["export_metals_queried"]),
        "export_fanout_size": int(rec["export_fanout_size"]),
        "representative_raw_row_id": rec["raw_row_id"],
        "source_file": rec["source_file"],
        "source_line_number": int(rec["source_line_number"]),
        # doi_primary is the *measurement's* source, so the archive's own
        # self-citation never fills it; that is tracked separately.
        "doi_primary": source_dois[0] if source_dois else None,
        "doi_source_all": source_dois,
        "doi_all": dois,
        "archive_citation_doi": archive_citation,
        "has_source_reference": bool(source_dois or other_refs),
        "reference_other": other_refs,
        "entry_author": NORM.clean_text(rec["entry_author"]),
        "addition_date": NORM.clean_text(rec["addition_date"]),
        "publication_year": comments.get("Publication_Year"),
        "publication_title": comments.get("Title"),
        "publication_authors": comments.get("Authors"),
        "data_location": comments.get("Data Location"),
        "sub_source_file": comments.get("file"),
        "aqueous_phase_metals_declared": comments.get("Aqueous Phase Metals"),
        "n_metals_declared": comments.get("No. of Metals"),
        "n_extractants_declared": comments.get("No. of Extractants"),
        "comments_raw": NORM.clean_text(rec["comments_description"]),
        "ini_comp_raw": NORM.clean_text(rec["ini_comp"]),
        "extractant_name_raw": NORM.clean_text(rec["Extractant_Name"]),
        "extractant_smiles_raw": NORM.clean_text(rec["Extractant_SMILES"]),
        "solvent_name_raw": NORM.clean_text(rec["Solvent_Name"]),
        "acid_name_raw": NORM.clean_text(rec["Acid_Name"]),
        # Raw passthroughs for every field that enters an identity key, so the
        # duplicate ladder can compare "raw text" against "parsed value" on
        # exactly the same field set.
        "modifier_name_raw": NORM.clean_text(rec["Phase_Modifier_Name"]),
        "modifier_concentration_raw": NORM.clean_text(rec["Phase_Modifier_Concentration_M"]),
        "acid_concentration_organic_raw": NORM.clean_text(rec["Acid_Concentration_Organic_M"]),
        "phase_ratio_raw": NORM.clean_text(rec["volValue"]),
        "complexant_name_raw": comments.get("Complexant_Name"),
        "complexant_smiles_raw": comments.get("Complexant_SMILES"),
        "complexant_concentration_raw": comments.get("Complexant_Concentration_M"),
        "holdback_smiles_raw": comments.get("Holdback_Agent_SMILES"),
        "holdback_concentration_raw": comments.get("Holdback_Agent_Concentration_M"),
        "nitrate_concentration_raw": comments.get("nitrate concentration(M)"),
        "extractant_concentration_raw": NORM.clean_text(rec["Extractant_Concentration_M"]),

        # ---- chemical system ----------------------------------------------
        "components": components,
        "system_component_class": system_class,
        "n_chemically_active_components": n_active,
        "n_organic_extractants": len(organics),
        "extractant_system_key": extractant_system_key,
        "extractant_names": [c["name"] for c in organics],
        "extractant_smiles_canonical": [c["smiles_canonical"] for c in organics],
        "extractant_concentrations_M": [c["concentration_M"] for c in organics],
        "extractant_primary_smiles": organics[0]["smiles_canonical"] if organics else None,
        "extractant_primary_name": organics[0]["name"] if organics else None,
        "extractant_primary_concentration_M": organics[0]["concentration_M"] if organics else None,
        "modifier_name": modifier["name"] if modifier else None,
        "modifier_concentration_M": modifier["concentration_M"] if modifier else None,
        "complexant_name": complexant["name"] if complexant else None,
        "complexant_smiles_canonical": complexant["smiles_canonical"] if complexant else None,
        "complexant_concentration_M": complexant["concentration_M"] if complexant else None,
        "n_complexants": len(complexants),
        "complexant_names": [c["name"] for c in complexants],
        "complexant_smiles_all": [c["smiles_canonical"] for c in complexants],
        "complexant_concentrations_M": [c["concentration_M"] for c in complexants],
        # Order-invariant signature of the full aqueous-complexant set; this is
        # what the identity key uses, so a two-component complexant system can
        # no longer collapse into a one-component one.
        "complexant_signature": "|".join(sorted(
            f"{(c['name'] or c['smiles_canonical'] or '?')}@{c['concentration_M']}"
            for c in complexants)) or None,
        "holdback_name": holdback["name"] if holdback else None,
        "holdback_smiles_canonical": holdback["smiles_canonical"] if holdback else None,
        "holdback_concentration_M": holdback["concentration_M"] if holdback else None,

        # ---- aqueous phase --------------------------------------------------
        "acid_names": acids,
        "acid_primary": acids[0] if acids else None,
        "acid_anion": CHEM_ANION(acids),
        "acid_concentration_M": acid_conc.value if acid_conc else None,
        "acid_concentration_raw": NORM.clean_text(rec["Acid_Concentration_M"]),
        "acid_concentration_organic_M": acid_org.value if acid_org else None,
        "nitrate_concentration_M": nitrate.value if nitrate else None,

        # ---- diluent --------------------------------------------------------
        "solvent_components": [c.name for c in solvent_components],
        "solvent_fractions": [c.fraction for c in solvent_components],
        "solvent_key": "|".join(f"{c.name}:{c.fraction:g}" if c.fraction is not None else c.name
                                for c in solvent_components) or None,
        "solvent_primary": solvent_components[0].name if solvent_components else None,
        "solvent_n_components": len(solvent_components),
        "solvent_pattern": solvent_pattern,

        # ---- conditions -----------------------------------------------------
        "metal_concentration_M": metal_conc.value if metal_conc else None,
        "metal_concentration_raw": NORM.clean_text(rec["Metal_Concentration_mM"]),
        "temperature_C": temperature.value if temperature else None,
        "temperature_raw": NORM.clean_text(rec["obsTempsValue"]),
        "contact_time_min": contact.value if contact else None,
        "contact_time_raw": NORM.clean_text(rec["Contact_Time_min"]),
        "shaking_time_min": shaking.value if shaking else None,
        "shaking_time_raw": NORM.clean_text(rec["Shaking_Time_min"]),
        "phase_ratio_org_aq": phase_ratio.value if phase_ratio else None,

        # ---- target ---------------------------------------------------------
        "D_value": d_val,
        "D_raw": NORM.clean_text(rec["obsDvaluesValue"]),
        "log_D": log_d,
    }
    row.update(metal)

    # Authoritative bibliographic metadata, added alongside -- never on top of --
    # whatever the archive recorded.  The archive's own publication_* values are
    # left exactly as found so the two can be compared.
    corrected, correction = NORM.correct_doi(row["doi_primary"])
    row["doi_primary_corrected"] = corrected
    row["doi_correction_rule"] = correction["rule"] if correction else None
    row["doi_correction_evidence"] = correction["evidence"] if correction else None
    if correction:
        flags.append("doi_repaired")
        log.record("reference.doi_repaired", row["doi_primary"], corrected)
    entry = crossref.get(corrected) if corrected else None
    if entry is not None and entry.get("status") == "ok":
        row.update({
            "reference_title": entry.get("crossref_title"),
            "reference_year": entry.get("crossref_year"),
            "reference_journal": entry.get("crossref_journal"),
            "reference_authors": entry.get("crossref_authors"),
            "reference_publisher": entry.get("crossref_publisher"),
            "reference_url": entry.get("crossref_url"),
            "reference_metadata_source": "crossref",
        })
    else:
        row.update({
            "reference_title": comments.get("Title"),
            "reference_year": comments.get("Publication_Year"),
            "reference_journal": None, "reference_authors": comments.get("Authors"),
            "reference_publisher": None, "reference_url": None,
            "reference_metadata_source": "archive_comments" if comments.get("Title") else None,
        })
        if corrected and entry is not None:
            flags.append("doi_does_not_resolve")
            log.record("reference.doi_does_not_resolve", row["doi_primary"], entry.get("status"))

    if row["metal_symbol"] is None:
        flags.append("metal_unresolved")
    if row["log_D"] is None:
        flags.append("target_missing")
    if row["extractant_primary_smiles"] is None:
        flags.append("extractant_structure_unresolved")
    if row["metal_oxidation_state_plausible"] is False:
        flags.append("implausible_oxidation_state")
    if extract["parse_failures"]:
        flags.append("rdkit_parse_failure")

    row["flags"] = sorted(set(flags))
    row["conflicting_component_names"] = [c["name"] for c in conflicting]
    row["conflicting_component_detail"] = [
        f"{c['name']}: record has {c['structure_in_record']}, archive consensus is {c['structure_expected']}"
        for c in conflicting
    ]
    row["rdkit_parse_failures"] = extract["parse_failures"]
    return row


def CHEM_ANION(acids: list[str]) -> str | None:
    anions = [CHEM_ACID_ANION.get(a) for a in acids]
    anions = [a for a in anions if a]
    return "|".join(sorted(set(anions))) if anions else None


CHEM_ACID_ANION = NORM.ACID_ANION


def main() -> None:
    ensure_dirs()
    records = pd.read_parquet(INTERMEDIATE_DIR / "records_collapsed.parquet")
    log = NORM.NormalizationLog()

    consensus, consensus_report = build_name_structure_consensus(records)
    deduced, deduction_report = deduce_by_elimination(records, consensus)
    # Deduced entries never overwrite a directly observed consensus.
    for name, smiles in deduced.items():
        consensus.setdefault(name, smiles)
    DEDUCED_NAMES.clear()
    DEDUCED_NAMES.update(deduced)
    structure_names = build_structure_name_index(consensus, records)

    crossref = load_crossref()
    rows = [build_row(rec, consensus, structure_names, crossref, log)
            for rec in records.to_dict("records")]
    normalized = pd.DataFrame(rows)

    if len(normalized) != len(records):
        raise AssertionError("stage 2 must not add or drop records")

    normalized.to_parquet(INTERMEDIATE_DIR / "records_normalized.parquet", index=False)

    payload = {
        "rdkit_version": RDKIT_VERSION,
        "crossref_cache": {
            "path": str(CROSSREF_CACHE),
            "dois_cached": len(crossref),
            "dois_resolved": sum(1 for v in crossref.values() if v.get("status") == "ok"),
            "dois_unresolved": sorted(k for k, v in crossref.items() if v.get("status") != "ok"),
        },
        "records_in": int(len(records)),
        "records_out": int(len(normalized)),
        "name_structure_consensus": {k: v for k, v in consensus_report.items()
                                     if k != "contested_detail"},
        "structure_deduction_by_elimination": deduction_report,
        "contested_names": consensus_report["contested_detail"],
        "normalization": log.to_dict(),
    }
    (AUDIT_DIR / "stage02_normalization_log.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    print(json.dumps({
        "records": len(normalized),
        "consensus": payload["name_structure_consensus"],
        "deduced_by_elimination": deduction_report["deduced_structures"],
        "flag_counts": dict(collections.Counter(
            f for fl in normalized["flags"] for f in fl).most_common()),
        "system_classes": normalized["system_component_class"].value_counts().to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
