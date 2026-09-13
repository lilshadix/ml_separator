"""``systems.py`` — the extraction-systems database (DESIGN.md section 3).

Layout addendum WB0 (``addenda/WB0.md``): the schema dataclasses of DESIGN.md section 3.2 (and
every other enum / protocol / dataclass of the design) live in ``gen18proc.types`` and are
re-exported here, so that both ``from gen18proc.systems import SystemEntry`` and
``from gen18proc.types import SystemEntry`` name the same class.  WB1 adds the system key /
``system_id`` hash, the validator (V1-V17, W1-W4), ``load_system`` / ``write_system`` and
``load_registry`` below the marker line.

Units and conventions: see ``gen18proc.types``.  The JSON entry records the monomer formal
concentration ``[HA]_T``; the loader converts to the dimer basis ``[(HA)2]_T = [HA]_T / 2``.
"""
from __future__ import annotations

from .types import (
    ASSUMED_LABEL,
    INADMISSIBLE_FLAGS,
    OOD_FLAGS,
    REGIME_STATUSES,
    SOURCE_KINDS,
    ActivityModel,
    ApplicabilityDomain,
    AqStream,
    CascadeResult,
    CascadeSpec,
    CationExchangeParams,
    ComplexantSpec,
    DEval,
    DistributionRecord,
    DModel,
    Flag,
    LigandSpec,
    Mechanism,
    MediumSpec,
    ModelBuildError,
    OrgStream,
    PhaseBehaviour,
    ProcessMetrics,
    Provenance,
    ProvStatus,
    SolvatingParams,
    Source,
    Sourced,
    StageDiagnostics,
    StageState,
    Stoichiometry,
    StreamRecord,
    SystemEntry,
    SystemValidationError,
    Violation,
)

__all__ = [
    "ASSUMED_LABEL", "INADMISSIBLE_FLAGS", "OOD_FLAGS", "REGIME_STATUSES", "SOURCE_KINDS",
    "ActivityModel", "ApplicabilityDomain", "AqStream", "CascadeResult", "CascadeSpec",
    "CationExchangeParams", "ComplexantSpec", "DEval", "DistributionRecord", "DModel", "Flag",
    "LigandSpec", "Mechanism", "MediumSpec", "ModelBuildError", "OrgStream", "PhaseBehaviour",
    "ProcessMetrics", "Provenance", "ProvStatus", "SolvatingParams", "Source", "Sourced",
    "StageDiagnostics", "StageState", "Stoichiometry", "StreamRecord", "SystemEntry",
    "SystemValidationError", "Violation",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = "gen18.1"
"""The only admissible ``SystemEntry.schema_version`` (validator V1)."""


# --- WB1 extends below ---
# system_key / system_id (section 3.1), validate_entry (3.6), load_system / write_system,
# load_registry.  Keep every name above unchanged; new public names are appended to __all__.
#
# Sources of the vocabularies below: bundle columns read on 2026-09-13 (``geom_cond__acid_class``,
# ``geom_cond__diluent_family``); DESIGN.md sections 3.1-3.6 for everything else.

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from collections.abc import Iterator, Mapping  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pandas as pd  # noqa: E402

__all__ += [
    "ACID_CLASSES", "ANIONS", "DILUENT_FAMILIES", "FAMILIES", "ORIGINS", "LIGAND_ROLES",
    "AGGREGATIONS", "STREAM_KINDS", "DUPLICATE_FLAGS", "K_REPORTED_AS", "TEMPERATURE_BANDS",
    "DEFAULT_BAND", "ApplicabilityMapping", "temperature_band", "system_key", "system_id",
    "entry_to_json", "entry_from_json", "iter_sourced", "validate_entry", "load_system",
    "write_system", "load_registry", "registry_row",
]

ACID_CLASSES = ("nitrate", "chloride", "sulfate", "perchlorate", "carboxylate")
ANIONS = ACID_CLASSES
DILUENT_FAMILIES = ("aliphatic_hydrocarbon", "aromatic", "other", "chlorinated",
                    "alcohol_modifier", "nitroaromatic", "ionic_liquid")
"""Bundle vocabulary of ``geom_cond__diluent_family`` (5860 rows after quarantine; V17)."""
FAMILIES = ("acidic_organophosphorus", "diglycolamide", "phen_carboxamide", "n_donor",
            "carboxylic_acid", "amine", "other")
ORIGINS = ("corpus", "literature", "mixed", "hypothetical")
LIGAND_ROLES = ("extractant", "synergist", "modifier")
AGGREGATIONS = ("monomer", "dimer")
STREAM_KINDS = ("scrub", "strip")
DUPLICATE_FLAGS = (None, "UNIT_SLIP_DUPLICATE", "TIED_D")
K_REPORTED_AS = ("log_k", "pH50")
GEN15_TOKENS = ("gen15", "deploy_g15")

#: Fixed temperature bands of DESIGN.md section 3.4: ``lo <= T < hi`` in degC.
TEMPERATURE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<20C", -math.inf, 20.0), ("20-30C", 20.0, 30.0), ("30-40C", 30.0, 40.0),
    ("40-50C", 40.0, 50.0), (">=50C", 50.0, math.inf),
)
DEFAULT_BAND = "20-30C"
"""Band assigned to a record with NaN temperature (section 3.4; the row count is reported)."""


def temperature_band(temperature_C: float | None) -> str:
    """Band string for a temperature in degC; None / NaN -> ``DEFAULT_BAND``."""
    if temperature_C is None or (isinstance(temperature_C, float) and math.isnan(temperature_C)):
        return DEFAULT_BAND
    t = float(temperature_C)
    for name, lo, hi in TEMPERATURE_BANDS:
        if lo <= t < hi:
            return name
    raise ValueError(f"temperature {temperature_C!r} matches no band")  # unreachable


class ApplicabilityMapping(dict):
    """``dict[str, ApplicabilityDomain]`` that also carries the hand-entered metadata keys of a
    literature domain (``_status``, ``_assumed_label``, ``_range_note``; DESIGN.md section 3.7)
    in ``meta[key]`` so that ``write_system`` reproduces them byte for byte.  Corpus-built
    domains have no metadata."""

    def __init__(self, domains: Mapping[str, ApplicabilityDomain] | None = None,
                 meta: Mapping[str, Mapping[str, Any]] | None = None):
        super().__init__(domains or {})
        self.meta: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (meta or {}).items()}


# =============================================================================================
# 3.1  System identity
# =============================================================================================

def system_key(entry: SystemEntry) -> tuple:
    """The five-part key of DESIGN.md section 3.1 built from an entry.

    (sorted canonical SMILES of ligands with role extractant / synergist, acid class, diluent
    family, sorted modifier *names*, sorted canonical SMILES of aqueous complexants; a complexant
    without SMILES contributes its name)."""
    ligands = sorted(lig.canonical_smiles for lig in entry.organic_ligands
                     if lig.role in ("extractant", "synergist"))
    modifiers = sorted(lig.name for lig in entry.organic_ligands if lig.role == "modifier")
    complexants = sorted((c.canonical_smiles or c.name) for c in entry.aqueous_complexants)
    return (tuple(ligands), entry.medium.acid_class, str(entry.diluent.get("family")),
            tuple(modifiers), tuple(complexants))


def system_id(key_or_entry: tuple | SystemEntry) -> str:
    """``"sys_" + blake2b(json.dumps(list(key), separators=(",",":"), ensure_ascii=True), 8)``."""
    key = system_key(key_or_entry) if isinstance(key_or_entry, SystemEntry) else key_or_entry
    payload = json.dumps([list(k) if isinstance(k, tuple) else k for k in key],
                         separators=(",", ":"), ensure_ascii=True).encode()
    return "sys_" + hashlib.blake2b(payload, digest_size=8).hexdigest()


# =============================================================================================
# JSON (de)serialisation of SystemEntry (shape of DESIGN.md sections 3.7 / 3.8)
# =============================================================================================

def _json_float(x: Any) -> Any:
    """numpy scalars -> Python; NaN -> None (JSON has no NaN)."""
    if x is None:
        return None
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _pair(v: tuple[float, float] | None) -> list | None:
    return None if v is None else [_json_float(v[0]), _json_float(v[1])]


def _pair_from(v: Any) -> tuple[float, float] | None:
    return None if v is None else (v[0], v[1])


def _sourced_tuple_to_json(items: tuple[Sourced, ...]) -> list:
    return [s.to_json() for s in items]


def _sourced_tuple_from(items: Any) -> tuple[Sourced, ...]:
    return tuple(Sourced.from_json(o) for o in (items or ()))


def _stoich_to_json(s: Stoichiometry) -> dict:
    return {"ligands_per_metal": s.ligands_per_metal.to_json(),
            "protons_released_per_metal": s.protons_released_per_metal.to_json(),
            "anions_per_metal": s.anions_per_metal.to_json()}


def _stoich_from(o: Mapping) -> Stoichiometry:
    return Stoichiometry(Sourced.from_json(o["ligands_per_metal"]),
                         Sourced.from_json(o["protons_released_per_metal"]),
                         Sourced.from_json(o["anions_per_metal"]))


def _ligand_to_json(lig: LigandSpec) -> dict:
    return {"name": lig.name, "smiles": lig.smiles, "canonical_smiles": lig.canonical_smiles,
            "role": lig.role, "mechanism": None if lig.mechanism is None else lig.mechanism.value,
            "concentration": lig.concentration.to_json(), "aggregation": lig.aggregation,
            "scaffold_id": lig.scaffold_id, "variant_tag": lig.variant_tag,
            "stoichiometry": _stoich_to_json(lig.stoichiometry)}


def _ligand_from(o: Mapping) -> LigandSpec:
    mech = o.get("mechanism")
    return LigandSpec(name=o["name"], smiles=o["smiles"], canonical_smiles=o["canonical_smiles"],
                      role=o["role"], mechanism=None if mech is None else Mechanism(mech),
                      concentration=Sourced.from_json(o["concentration"]),
                      aggregation=o["aggregation"], scaffold_id=o.get("scaffold_id"),
                      variant_tag=o.get("variant_tag"),
                      stoichiometry=_stoich_from(o["stoichiometry"]))


def _complexant_to_json(c: ComplexantSpec) -> dict:
    return {"name": c.name, "smiles": c.smiles, "canonical_smiles": c.canonical_smiles,
            "concentration": c.concentration.to_json(),
            "log_beta": {m: _sourced_tuple_to_json(v) for m, v in c.log_beta.items()},
            "protonation_logk": _sourced_tuple_to_json(c.protonation_logk),
            "regeneration_fraction": c.regeneration_fraction.to_json()}


def _complexant_from(o: Mapping) -> ComplexantSpec:
    return ComplexantSpec(
        name=o["name"], smiles=o.get("smiles"), canonical_smiles=o.get("canonical_smiles"),
        concentration=Sourced.from_json(o["concentration"]),
        log_beta={m: _sourced_tuple_from(v) for m, v in (o.get("log_beta") or {}).items()},
        protonation_logk=_sourced_tuple_from(o.get("protonation_logk")),
        regeneration_fraction=Sourced.from_json(o["regeneration_fraction"]))


def _medium_to_json(m: MediumSpec) -> dict:
    return {"acid": m.acid, "acid_class": m.acid_class, "anion": m.anion,
            "salting_agent": m.salting_agent, "salting_anion_M": m.salting_anion_M.to_json(),
            "ionic_strength_M": m.ionic_strength_M.to_json(),
            "temperature_C": m.temperature_C.to_json()}


def _medium_from(o: Mapping) -> MediumSpec:
    return MediumSpec(acid=o["acid"], acid_class=o["acid_class"], anion=o["anion"],
                      salting_agent=o.get("salting_agent"),
                      salting_anion_M=Sourced.from_json(o["salting_anion_M"]),
                      ionic_strength_M=Sourced.from_json(o["ionic_strength_M"]),
                      temperature_C=Sourced.from_json(o["temperature_C"]))


def _params_to_json(p: CationExchangeParams | SolvatingParams) -> dict:
    common = {"medium_anion": p.medium_anion, "temperature_band": p.temperature_band,
              "log_k": {m: s.to_json() for m, s in p.log_k.items()},
              "delta_h_kj_mol": None if p.delta_h_kj_mol is None else p.delta_h_kj_mol.to_json()}
    if isinstance(p, CationExchangeParams):
        return {**common, "model_type": "cation_exchange", "a_dimer": p.a_dimer.to_json(),
                "b_proton": p.b_proton.to_json(), "k_reported_as": p.k_reported_as,
                "beta_anion": None if p.beta_anion is None else
                {m: _sourced_tuple_to_json(v) for m, v in p.beta_anion.items()},
                "saponification_degree_studied": _pair(p.saponification_degree_studied)}
    return {**common, "model_type": "solvating", "n_solvation": p.n_solvation.to_json(),
            "p_anion": p.p_anion.to_json(), "p_h": p.p_h.to_json(),
            "k_acid_uptake": p.k_acid_uptake.to_json()}


def _params_from(o: Mapping) -> CationExchangeParams | SolvatingParams:
    dh = o.get("delta_h_kj_mol")
    common = dict(medium_anion=o["medium_anion"], temperature_band=o["temperature_band"],
                  log_k={m: Sourced.from_json(s) for m, s in (o.get("log_k") or {}).items()},
                  delta_h_kj_mol=None if dh is None else Sourced.from_json(dh))
    model_type = o.get("model_type")
    if model_type == "cation_exchange":
        ba = o.get("beta_anion")
        return CationExchangeParams(
            **common, a_dimer=Sourced.from_json(o["a_dimer"]),
            b_proton=Sourced.from_json(o["b_proton"]),
            k_reported_as=o.get("k_reported_as", "log_k"),
            beta_anion=None if ba is None else {m: _sourced_tuple_from(v) for m, v in ba.items()},
            saponification_degree_studied=_pair_from(o.get("saponification_degree_studied")))
    if model_type == "solvating":
        return SolvatingParams(**common, n_solvation=Sourced.from_json(o["n_solvation"]),
                               p_anion=Sourced.from_json(o["p_anion"]),
                               p_h=Sourced.from_json(o["p_h"]),
                               k_acid_uptake=Sourced.from_json(o["k_acid_uptake"]))
    raise ValueError(f"unknown params model_type {model_type!r}")


_PHASE_FIELDS = ("loc_metal_M", "loc_acid_M", "third_phase_observed", "disengagement_s",
                 "ligand_loss_mol_per_L_aq", "max_loading_fraction_studied")


def _phase_to_json(p: PhaseBehaviour) -> dict:
    out = {f: getattr(p, f).to_json() for f in _PHASE_FIELDS}
    out["regenerability_note"] = p.regenerability_note
    return out


def _phase_from(o: Mapping) -> PhaseBehaviour:
    return PhaseBehaviour(**{f: Sourced.from_json(o[f]) for f in _PHASE_FIELDS},
                          regenerability_note=o.get("regenerability_note") or "")


def _stream_to_json(s: StreamRecord) -> dict:
    return {"kind": s.kind, "acid_M": s.acid_M.to_json(), "anion_M": s.anion_M.to_json(),
            "complexant_M": s.complexant_M.to_json(),
            "metals_mM": {m: v.to_json() for m, v in s.metals_mM.items()},
            "oa_ratio": s.oa_ratio.to_json(), "stages": s.stages.to_json(),
            "fraction_removed": {m: v.to_json() for m, v in s.fraction_removed.items()},
            "note": s.note}


def _stream_from(o: Mapping) -> StreamRecord:
    return StreamRecord(
        kind=o["kind"], acid_M=Sourced.from_json(o["acid_M"]),
        anion_M=Sourced.from_json(o["anion_M"]), complexant_M=Sourced.from_json(o["complexant_M"]),
        metals_mM={m: Sourced.from_json(v) for m, v in (o.get("metals_mM") or {}).items()},
        oa_ratio=Sourced.from_json(o["oa_ratio"]), stages=Sourced.from_json(o["stages"]),
        fraction_removed={m: Sourced.from_json(v)
                          for m, v in (o.get("fraction_removed") or {}).items()},
        note=o.get("note") or "")


_RECORD_FIELDS = ("record_id", "metal", "d", "log_d", "acid_nominal_M", "acid_eq_M", "anion_M",
                  "ligand_M", "complexant_M", "metals_initial_mM", "oa_ratio", "temperature_C",
                  "contact_time_min", "diluent_name", "publication_id", "experiment_series_id",
                  "replicate_id", "loading_series_id", "is_tracer", "fit_eligible",
                  "fit_ineligible_reason", "duplicate_flag")


def _record_to_json(r: DistributionRecord) -> dict:
    out: dict[str, Any] = {}
    for f in _RECORD_FIELDS:
        v = getattr(r, f)
        if isinstance(v, Mapping):
            v = {k: _json_float(x) for k, x in v.items()}
        else:
            v = _json_float(v)
        out[f] = v
    out["provenance"] = r.provenance.to_json()
    return out


def _record_from(o: Mapping) -> DistributionRecord:
    return DistributionRecord(
        record_id=o["record_id"], metal=o["metal"], d=float(o["d"]), log_d=float(o["log_d"]),
        acid_nominal_M=o.get("acid_nominal_M"), acid_eq_M=o.get("acid_eq_M"),
        anion_M=o.get("anion_M"), ligand_M=dict(o.get("ligand_M") or {}),
        complexant_M=o.get("complexant_M"),
        metals_initial_mM=dict(o.get("metals_initial_mM") or {}), oa_ratio=o.get("oa_ratio"),
        temperature_C=o.get("temperature_C"), contact_time_min=o.get("contact_time_min"),
        diluent_name=o.get("diluent_name"), publication_id=o.get("publication_id"),
        experiment_series_id=o.get("experiment_series_id"), replicate_id=o.get("replicate_id"),
        loading_series_id=o.get("loading_series_id"), is_tracer=bool(o.get("is_tracer", False)),
        fit_eligible=bool(o.get("fit_eligible", False)),
        fit_ineligible_reason=o.get("fit_ineligible_reason"),
        duplicate_flag=o.get("duplicate_flag"),
        provenance=Provenance.from_json(o["provenance"]))


_DOMAIN_META_KEYS = ("_status", "_assumed_label", "_range_note")


def _domain_to_json(d: ApplicabilityDomain, meta: Mapping[str, Any] | None) -> dict:
    out = {"anion": d.anion, "diluent_family": d.diluent_family, "modifiers": list(d.modifiers),
           "log_acid": _pair(d.log_acid),
           "log_ligand": {k: _pair(v) for k, v in d.log_ligand.items()},
           "log_metal_total_mM": _pair(d.log_metal_total_mM),
           "loading_fraction": _pair(d.loading_fraction),
           "log_complexant": _pair(d.log_complexant),
           "oa_ratio": _pair(d.oa_ratio), "temperature_C": _pair(d.temperature_C),
           "saponification_degree": _pair(d.saponification_degree),
           "hull_vertices": None if d.hull_vertices is None else
           [[_json_float(x), _json_float(y)] for x, y in d.hull_vertices],
           "n_records": int(d.n_records), "n_publications": int(d.n_publications)}
    if meta:
        out.update({k: v for k, v in meta.items() if k in _DOMAIN_META_KEYS})
    return out


def _domain_from(o: Mapping) -> tuple[ApplicabilityDomain, dict[str, Any]]:
    hv = o.get("hull_vertices")
    dom = ApplicabilityDomain(
        anion=o["anion"], diluent_family=o["diluent_family"],
        modifiers=tuple(o.get("modifiers") or ()), log_acid=_pair_from(o["log_acid"]),
        log_ligand={k: _pair_from(v) for k, v in (o.get("log_ligand") or {}).items()},
        log_metal_total_mM=_pair_from(o.get("log_metal_total_mM")),
        loading_fraction=_pair_from(o.get("loading_fraction")),
        log_complexant=_pair_from(o.get("log_complexant")),
        oa_ratio=_pair_from(o.get("oa_ratio")), temperature_C=_pair_from(o.get("temperature_C")),
        saponification_degree=_pair_from(o.get("saponification_degree")),
        hull_vertices=None if hv is None else tuple((x, y) for x, y in hv),
        n_records=int(o.get("n_records", 0)), n_publications=int(o.get("n_publications", 0)))
    meta = {k: o[k] for k in _DOMAIN_META_KEYS if k in o}
    return dom, meta


def entry_to_json(entry: SystemEntry) -> dict:
    """The JSON object of ``systems/<system_id>.json`` (DESIGN.md sections 3.7 / 3.8)."""
    meta = getattr(entry.applicability, "meta", {})
    return {
        "schema_version": entry.schema_version, "system_id": entry.system_id,
        "name": entry.name, "family": entry.family, "origin": entry.origin,
        "organic_ligands": [_ligand_to_json(lig) for lig in entry.organic_ligands],
        "aqueous_complexants": [_complexant_to_json(c) for c in entry.aqueous_complexants],
        "diluent": dict(entry.diluent), "medium": _medium_to_json(entry.medium),
        "params": {lig: {band: _params_to_json(p) for band, p in bands.items()}
                   for lig, bands in entry.params.items()},
        "phase": _phase_to_json(entry.phase),
        "stream_records": [_stream_to_json(s) for s in entry.stream_records],
        "oxidation_state_routes": [dict(r) for r in entry.oxidation_state_routes],
        "direction_prior": None if entry.direction_prior is None else dict(entry.direction_prior),
        "applicability": {k: _domain_to_json(d, meta.get(k))
                          for k, d in entry.applicability.items()},
        "records": [_record_to_json(r) for r in entry.records],
        "notes": entry.notes,
    }


def entry_from_json(obj: Mapping) -> SystemEntry:
    """Inverse of ``entry_to_json``; raises ``ValueError`` / ``KeyError`` on a malformed object
    (``load_system`` turns vocabulary problems into V16 violations first)."""
    domains, meta = {}, {}
    for k, o in (obj.get("applicability") or {}).items():
        domains[k], meta[k] = _domain_from(o)
    return SystemEntry(
        schema_version=obj.get("schema_version", ""), system_id=obj["system_id"],
        name=obj["name"], family=obj["family"], origin=obj["origin"],
        organic_ligands=tuple(_ligand_from(o) for o in obj.get("organic_ligands") or ()),
        aqueous_complexants=tuple(_complexant_from(o)
                                  for o in obj.get("aqueous_complexants") or ()),
        diluent=dict(obj.get("diluent") or {}), medium=_medium_from(obj["medium"]),
        params={lig: {band: _params_from(p) for band, p in bands.items()}
                for lig, bands in (obj.get("params") or {}).items()},
        phase=_phase_from(obj["phase"]),
        stream_records=tuple(_stream_from(s) for s in obj.get("stream_records") or ()),
        oxidation_state_routes=tuple(dict(r) for r in obj.get("oxidation_state_routes") or ()),
        direction_prior=obj.get("direction_prior"),
        applicability=ApplicabilityMapping(domains, meta),
        records=tuple(_record_from(r) for r in obj.get("records") or ()),
        notes=obj.get("notes") or "")


# =============================================================================================
# 3.6  Validator
# =============================================================================================

def iter_sourced(entry: SystemEntry) -> Iterator[tuple[str, Sourced]]:
    """Every ``Sourced`` of an entry with its dotted path (records' provenances excluded)."""
    for i, lig in enumerate(entry.organic_ligands):
        p = f"organic_ligands[{i}]"
        yield f"{p}.concentration", lig.concentration
        st = lig.stoichiometry
        yield f"{p}.stoichiometry.ligands_per_metal", st.ligands_per_metal
        yield f"{p}.stoichiometry.protons_released_per_metal", st.protons_released_per_metal
        yield f"{p}.stoichiometry.anions_per_metal", st.anions_per_metal
    for i, c in enumerate(entry.aqueous_complexants):
        p = f"aqueous_complexants[{i}]"
        yield f"{p}.concentration", c.concentration
        for m, items in c.log_beta.items():
            for j, s in enumerate(items):
                yield f"{p}.log_beta.{m}[{j}]", s
        for j, s in enumerate(c.protonation_logk):
            yield f"{p}.protonation_logk[{j}]", s
        yield f"{p}.regeneration_fraction", c.regeneration_fraction
    yield "medium.salting_anion_M", entry.medium.salting_anion_M
    yield "medium.ionic_strength_M", entry.medium.ionic_strength_M
    yield "medium.temperature_C", entry.medium.temperature_C
    for lig, bands in entry.params.items():
        for band, blk in bands.items():
            p = f"params.{lig}.{band}"
            for m, s in blk.log_k.items():
                yield f"{p}.log_k.{m}", s
            if blk.delta_h_kj_mol is not None:
                yield f"{p}.delta_h_kj_mol", blk.delta_h_kj_mol
            if isinstance(blk, CationExchangeParams):
                yield f"{p}.a_dimer", blk.a_dimer
                yield f"{p}.b_proton", blk.b_proton
                for m, items in (blk.beta_anion or {}).items():
                    for j, s in enumerate(items):
                        yield f"{p}.beta_anion.{m}[{j}]", s
            else:
                yield f"{p}.n_solvation", blk.n_solvation
                yield f"{p}.p_anion", blk.p_anion
                yield f"{p}.p_h", blk.p_h
                yield f"{p}.k_acid_uptake", blk.k_acid_uptake
    for f in _PHASE_FIELDS:
        yield f"phase.{f}", getattr(entry.phase, f)
    for i, s in enumerate(entry.stream_records):
        p = f"stream_records[{i}]"
        for f in ("acid_M", "anion_M", "complexant_M", "oa_ratio", "stages"):
            yield f"{p}.{f}", getattr(s, f)
        for m, val in s.metals_mM.items():
            yield f"{p}.metals_mM.{m}", val
        for m, val in s.fraction_removed.items():
            yield f"{p}.fraction_removed.{m}", val


def _canonical(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger  # local import: the module imports without RDKit
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def _provenance_violations(path: str, prov: Provenance, value: Any) -> list[Violation]:
    """V2, V3, V4, V6, V9, V16 for one provenance (``value`` is the quantity's value)."""
    out: list[Violation] = []

    def err(rule: str, msg: str) -> None:
        out.append(Violation("error", path, f"{rule}: {msg}"))

    if not isinstance(prov.status, ProvStatus):
        err("V16", f"status {prov.status!r} outside ProvStatus")
        return out
    src = prov.source
    if src.kind not in SOURCE_KINDS:
        err("V16", f"source.kind {src.kind!r} outside {SOURCE_KINDS}")
    st = prov.status
    if value is not None and st is ProvStatus.UNKNOWN:
        err("V2", "non-null value with status unknown")
    if st in (ProvStatus.LITERATURE, ProvStatus.MEASURED_LITERATURE):
        if not src.doi or not src.locator:
            err("V3", f"status {st.value} requires a DOI and a locator")
    if st is ProvStatus.MEASURED_CORPUS:
        if not src.safe_exp_ids or not src.publication_id:
            err("V3", "status measured_corpus requires safe_exp_ids and a publication_id")
    if st is ProvStatus.FITTED_FROM_CORPUS:
        if not prov.model_id or not prov.fit_manifest_sha256:
            err("V3", "status fitted_from_corpus requires model_id and fit_manifest_sha256")
        if prov.range is None:
            err("V3", "status fitted_from_corpus requires a range")
    if st is ProvStatus.ASSUMED:
        if prov.assumed_label != ASSUMED_LABEL:
            err("V4", f"assumed without assumed_label == {ASSUMED_LABEL!r}")
        if src.kind == "doi":
            if not src.doi:
                err("V4", "assumed with source.kind doi but no DOI")
        elif src.kind == "none":
            if not (prov.note or src.locator):
                err("V4", "assumed with source.kind none needs a note (or locator) saying why")
        else:
            err("V4", "assumed requires source.kind doi or none naming where the number must "
                "come from")
        if prov.range is None:
            err("V9", "assumed numeric without a range")
    for tok in GEN15_TOKENS:
        if (prov.model_id and tok in prov.model_id) or (src.locator and tok in src.locator):
            err("V6", f"provenance names {tok!r}: the gen15 direction model is never a D source")
            break
    return out


def validate_entry(entry: SystemEntry) -> list[Violation]:
    """Rules V1-V17 (level ``error``) and W1-W4 (level ``warning``) of DESIGN.md section 3.6."""
    v: list[Violation] = []

    def err(path: str, rule: str, msg: str) -> None:
        v.append(Violation("error", path, f"{rule}: {msg}"))

    def warn(path: str, rule: str, msg: str) -> None:
        v.append(Violation("warning", path, f"{rule}: {msg}"))

    # V1
    if entry.schema_version != SCHEMA_VERSION:
        err("schema_version", "V1", f"unknown schema_version {entry.schema_version!r}")
    # V16 vocabularies
    if entry.family not in FAMILIES:
        err("family", "V16", f"{entry.family!r} outside {FAMILIES}")
    if entry.origin not in ORIGINS:
        err("origin", "V16", f"{entry.origin!r} outside {ORIGINS}")
    if entry.medium.anion not in ANIONS:
        err("medium.anion", "V16", f"{entry.medium.anion!r} outside {ANIONS}")
    if entry.medium.acid_class not in ACID_CLASSES:
        err("medium.acid_class", "V16", f"{entry.medium.acid_class!r} outside {ACID_CLASSES}")
    # V17
    fam = entry.diluent.get("family") if isinstance(entry.diluent, Mapping) else None
    if fam not in DILUENT_FAMILIES:
        err("diluent.family", "V17", f"{fam!r} outside the bundle vocabulary {DILUENT_FAMILIES}")
    # per-Sourced rules
    for path, s in iter_sourced(entry):
        v.extend(_provenance_violations(path, s.provenance, s.value))
    # ligands: V10, V14, V16
    for i, lig in enumerate(entry.organic_ligands):
        p = f"organic_ligands[{i}]"
        if lig.role not in LIGAND_ROLES:
            err(f"{p}.role", "V16", f"{lig.role!r} outside {LIGAND_ROLES}")
        if lig.aggregation not in AGGREGATIONS:
            err(f"{p}.aggregation", "V16", f"{lig.aggregation!r} outside {AGGREGATIONS}")
        if lig.mechanism is not None and not isinstance(lig.mechanism, Mechanism):
            err(f"{p}.mechanism", "V16", f"{lig.mechanism!r} outside Mechanism")
        can = _canonical(lig.smiles)
        if can is None:
            err(f"{p}.smiles", "V10", f"RDKit cannot parse {lig.smiles!r}")
        elif can != lig.canonical_smiles:
            err(f"{p}.canonical_smiles", "V10",
                f"canonical form {can!r} differs from recorded {lig.canonical_smiles!r}")
        if lig.mechanism is Mechanism.CATION_EXCHANGE:
            prot = lig.stoichiometry.protons_released_per_metal
            literature = prot.status in (ProvStatus.LITERATURE, ProvStatus.MEASURED_LITERATURE)
            if (prot.value != 3 or lig.aggregation != "dimer") and not literature:
                err(f"{p}.stoichiometry", "V14",
                    "cation exchange requires protons_released_per_metal == 3 and aggregation "
                    "dimer unless the value carries a literature source")
    for i, c in enumerate(entry.aqueous_complexants):
        p = f"aqueous_complexants[{i}]"
        if c.smiles is not None:
            can = _canonical(c.smiles)
            if can is None:
                err(f"{p}.smiles", "V10", f"RDKit cannot parse {c.smiles!r}")
            elif can != c.canonical_smiles:
                err(f"{p}.canonical_smiles", "V10",
                    f"canonical form {can!r} differs from recorded {c.canonical_smiles!r}")
    # V7, V8, V16 (k_reported_as), W4
    record_metals = {r.metal for r in entry.records}
    stream_metals = {m for s in entry.stream_records for m in s.metals_mM}
    for lig, bands in entry.params.items():
        for band, blk in bands.items():
            p = f"params.{lig}.{band}"
            if blk.medium_anion != entry.medium.anion:
                err(f"{p}.medium_anion", "V7",
                    f"{blk.medium_anion!r} != medium.anion {entry.medium.anion!r}")
            missing = sorted((record_metals | stream_metals) - set(blk.log_k))
            if missing:
                err(f"{p}.log_k", "V8", f"missing log_k for metals {missing}")
            if isinstance(blk, CationExchangeParams):
                if blk.k_reported_as not in K_REPORTED_AS:
                    err(f"{p}.k_reported_as", "V16",
                        f"{blk.k_reported_as!r} outside {K_REPORTED_AS}")
            elif blk.k_acid_uptake.value is None:
                warn(f"{p}.k_acid_uptake", "W4", "k_acid_uptake is null for a solvating ligand "
                     "(ACID_UPTAKE_UNMODELLED above 1 M acid)")
    # V5 phase and hand-entered domains
    for f in _PHASE_FIELDS:
        s = getattr(entry.phase, f)
        if s.value is not None and s.status is ProvStatus.ASSUMED:
            err(f"phase.{f}", "V5", "phase values stay null unless measured or literature")
    meta = getattr(entry.applicability, "meta", {})
    for key, dom in entry.applicability.items():
        m = meta.get(key, {})
        status = m.get("_status")
        if status == "assumed":
            if m.get("_assumed_label") != ASSUMED_LABEL or not m.get("_range_note"):
                err(f"applicability.{key}", "V5", "hand-entered domain with _status assumed needs "
                    f"_assumed_label {ASSUMED_LABEL!r} and a non-empty _range_note")
        elif status is not None and status not in {s.value for s in ProvStatus}:
            err(f"applicability.{key}", "V16", f"_status {status!r} outside ProvStatus")
        elif status is None and dom.n_records == 0 and dom.hull_vertices is None:
            err(f"applicability.{key}", "V5", "hand-entered domain (no records, no hull) without "
                "a _status; enter _status assumed with a _range_note or build it from records")
    # V11
    recomputed = system_id(entry)
    if entry.system_id != recomputed:
        err("system_id", "V11", f"{entry.system_id!r} != recomputed {recomputed!r}")
    # records: V12, V13, V16, provenance rules, W3
    seen: set[str] = set()
    no_pub = 0
    for i, r in enumerate(entry.records):
        p = f"records[{i}]"
        if r.record_id in seen:
            err(f"{p}.record_id", "V12", f"duplicate record_id {r.record_id!r}")
        seen.add(r.record_id)
        if not (r.d > 0) or not math.isfinite(r.log_d):
            err(f"{p}", "V13", f"d={r.d!r}, log_d={r.log_d!r}")
        if r.duplicate_flag not in DUPLICATE_FLAGS:
            err(f"{p}.duplicate_flag", "V16", f"{r.duplicate_flag!r} outside {DUPLICATE_FLAGS}")
        v.extend(_provenance_violations(f"{p}.provenance", r.provenance, r.d))
        if r.publication_id is None:
            no_pub += 1
    if no_pub:
        warn("records", "W3", f"{no_pub} records without publication_id")
    # V15
    if entry.aqueous_complexants and record_metals:
        for i, c in enumerate(entry.aqueous_complexants):
            missing = sorted(record_metals - set(c.log_beta))
            if missing:
                err(f"aqueous_complexants[{i}].log_beta", "V15",
                    f"no log_beta for record metals {missing}")
    # stream kinds (V16)
    for i, s in enumerate(entry.stream_records):
        if s.kind not in STREAM_KINDS:
            err(f"stream_records[{i}].kind", "V16", f"{s.kind!r} outside {STREAM_KINDS}")
    # W1, W2
    param_metals = {m for bands in entry.params.values() for blk in bands.values()
                    for m in blk.log_k}
    if len(record_metals | stream_metals | param_metals) < 2:
        warn("records", "W1", "fewer than 2 metals in the entry")
    if entry.medium.temperature_C.value is None and not any(
            r.temperature_C is not None for r in entry.records):
        warn("medium.temperature_C", "W2", "no temperature anywhere in the entry")
    return v


def _prevalidate_json(obj: Mapping) -> list[Violation]:
    """Vocabulary checks (V16) on the raw JSON so that ``entry_from_json`` cannot raise on an
    enum value; run by ``load_system`` before construction."""
    out: list[Violation] = []
    statuses = {s.value for s in ProvStatus}
    mechanisms = {m.value for m in Mechanism} | {None}

    def walk(o: Any, path: str) -> None:
        if isinstance(o, Mapping):
            if "status" in o and "value" in o and o.get("status") not in statuses:
                out.append(Violation("error", path,
                                     f"V16: status {o.get('status')!r} outside ProvStatus"))
            if "kind" in o and "safe_exp_ids" in o and o.get("kind") not in SOURCE_KINDS:
                out.append(Violation("error", path,
                                     f"V16: source.kind {o.get('kind')!r} outside {SOURCE_KINDS}"))
            if "mechanism" in o and "role" in o and o.get("mechanism") not in mechanisms:
                out.append(Violation("error", path,
                                     f"V16: mechanism {o.get('mechanism')!r} outside Mechanism"))
            if "model_type" in o and o.get("model_type") not in ("cation_exchange", "solvating"):
                out.append(Violation("error", path, f"V16: model_type {o.get('model_type')!r}"))
            for k, val in o.items():
                walk(val, f"{path}.{k}" if path else str(k))
        elif isinstance(o, (list, tuple)):
            for i, val in enumerate(o):
                walk(val, f"{path}[{i}]")

    walk(obj, "")
    return out


def load_system(path: str | Path) -> SystemEntry:
    """Read ``systems/<id>.json``; raises ``SystemValidationError`` on any error-level rule."""
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    pre = _prevalidate_json(obj)
    if any(x.level == "error" for x in pre):
        raise SystemValidationError(pre)
    entry = entry_from_json(obj)
    violations = validate_entry(entry)
    if any(x.level == "error" for x in violations):
        raise SystemValidationError(violations)
    return entry


def write_system(entry: SystemEntry, path: str | Path) -> Path:
    """Serialise with ``sort_keys=True, indent=2, ensure_ascii=True``; floats by ``repr``
    (``json.dumps`` default); NaN is refused (``allow_nan=False``); trailing newline; LF."""
    path = Path(path)
    text = json.dumps(entry_to_json(entry), sort_keys=True, indent=2, ensure_ascii=True,
                      allow_nan=False)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    return path


_REGISTRY_COLUMNS = ("system_id", "name", "family", "mechanism", "origin", "n_records",
                     "n_publications", "metals", "scaffold_ids")


def registry_row(obj: Mapping) -> dict:
    """Registry summary of one entry JSON object (used by ``g18_build_db.py``)."""
    ligs = obj.get("organic_ligands") or []
    extractants = [lig for lig in ligs if lig.get("role") in ("extractant", "synergist")]
    mechs = sorted({lig.get("mechanism") for lig in extractants if lig.get("mechanism")})
    metals = sorted({r["metal"] for r in obj.get("records") or []})
    pubs = {r.get("publication_id") for r in obj.get("records") or []} - {None}
    return {
        "system_id": obj["system_id"], "name": obj["name"], "family": obj["family"],
        "mechanism": ";".join(mechs), "origin": obj["origin"],
        "n_records": len(obj.get("records") or []), "n_publications": len(pubs),
        "metals": ";".join(metals),
        "scaffold_ids": ";".join(sorted({lig.get("scaffold_id") for lig in ligs
                                         if lig.get("scaffold_id")})),
        "ligands": ";".join(lig["name"] for lig in extractants),
        "acid_class": obj["medium"]["acid_class"],
        "diluent_family": obj["diluent"].get("family"),
        "modifiers": ";".join(lig["name"] for lig in ligs if lig.get("role") == "modifier"),
    }


def load_registry(systems_dir: str | Path) -> pd.DataFrame:
    """One row per ``sys_*.json`` in ``systems_dir`` (system_id, name, family, mechanism,
    origin, n_records, n_publications, metals, scaffold_ids, plus ligands / acid_class /
    diluent_family / modifiers).  Reads ``registry.json`` when present, else scans the files
    (JSON only, no validation).  Sorted by ``system_id``."""
    systems_dir = Path(systems_dir)
    reg = systems_dir / "registry.json"
    if reg.exists():
        with open(reg, encoding="utf-8") as fh:
            rows = json.load(fh)["systems"]
    else:
        rows = []
        for p in sorted(systems_dir.glob("sys_*.json")):
            with open(p, encoding="utf-8") as fh:
                rows.append(registry_row(json.load(fh)))
    extra = [c for c in (rows[0].keys() if rows else ()) if c not in _REGISTRY_COLUMNS]
    df = pd.DataFrame(rows, columns=list(_REGISTRY_COLUMNS) + extra)
    return df.sort_values("system_id").reset_index(drop=True)
