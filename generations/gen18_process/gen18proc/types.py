"""Every enum, protocol and frozen dataclass of the gen18 design, transcribed from DESIGN.md.

Sections transcribed (names and fields are the contract; other work blocks code against them):

* 1.2 ``ProvStatus``; 1.3 ``Source`` / ``Provenance`` / ``Sourced`` with the JSON shape of the
  database (``Sourced.to_json`` is the flat object shown in DESIGN.md section 1.3; ``value: null``
  is the only way to say "not known"); 1.4 ``Flag`` plus ``INADMISSIBLE_FLAGS`` / ``OOD_FLAGS``.
* 3.2 the extraction-systems database: ``Mechanism``, ``LigandSpec``, ``Stoichiometry``,
  ``ComplexantSpec``, ``MediumSpec``, ``CationExchangeParams``, ``SolvatingParams``,
  ``PhaseBehaviour``, ``StreamRecord``, ``DistributionRecord``, ``ApplicabilityDomain``,
  ``SystemEntry``; 3.6 ``Violation``, ``SystemValidationError``; 5.7 ``ModelBuildError``.
* 5.1 ``StageState``, ``DEval``, ``DModel``; 5.6 ``ActivityModel``.
* 6.1 ``AqStream``, ``OrgStream``; 6.2 ``StageDiagnostics``.
* 7.1 ``CascadeSpec``; 7.9 ``CascadeResult``; 8 ``ProcessMetrics``.

Units (DESIGN.md section 1.1): concentrations mol/L in the phase named; ``metals_mM`` /
``metals_initial_mM`` fields are mM and are converted at the boundary (``mM * 1e-3``); flows L/h;
O/A is the volumetric ratio V_org / V_aq; temperature degC; time h (``contact_time_min`` in
minutes, ``disengagement_s`` in seconds); ``log`` means log10.  Ligand concentrations of dimeric
acidic extractants are **dimer** concentrations ``[(HA)2]`` inside the code; the JSON entry records
the monomer formal concentration ``[HA]_T`` and the loader converts ``[(HA)2]_T = [HA]_T / 2``.

Every dataclass is frozen and uses plain ``dict`` / ``tuple`` / ``Mapping`` field types so that
instances are hashable where their fields are and serialise without surprises.  This module
imports nothing heavier than the standard library.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    # 1.2-1.4
    "ProvStatus", "Source", "Provenance", "Sourced", "Flag", "INADMISSIBLE_FLAGS", "OOD_FLAGS",
    "REGIME_STATUSES",
    # 3.2, 3.6, 5.7
    "Mechanism", "Stoichiometry", "LigandSpec", "ComplexantSpec", "MediumSpec",
    "CationExchangeParams", "SolvatingParams", "PhaseBehaviour", "StreamRecord",
    "DistributionRecord", "ApplicabilityDomain", "SystemEntry", "Violation",
    "SystemValidationError", "ModelBuildError",
    # 5.1, 5.6
    "StageState", "DEval", "DModel", "ActivityModel",
    # 6.1, 6.2
    "AqStream", "OrgStream", "StageDiagnostics",
    # 7.1, 7.9, 8
    "CascadeSpec", "CascadeResult", "ProcessMetrics",
]


# =============================================================================================
# 1.2  Provenance statuses
# =============================================================================================

class ProvStatus(str, Enum):
    """Provenance status of every numeric quantity (DESIGN.md section 1.2).

    Rules enforced by the validator (section 3.6): a non-null value with ``UNKNOWN`` is an error;
    ``LITERATURE`` and ``MEASURED_LITERATURE`` require a DOI and a locator; ``MEASURED_CORPUS``
    requires ``safe_exp_ids`` and a ``publication_id``; ``FITTED_FROM_CORPUS`` requires
    ``model_id``, ``fit_manifest_sha256`` and a ``range``; ``ASSUMED`` requires
    ``assumed_label == "ASSUMED_PLACEHOLDER"``, a ``range`` and a source naming the DOI (or "none").
    """

    MEASURED_CORPUS = "measured_corpus"
    MEASURED_LITERATURE = "measured_literature"
    FITTED_FROM_CORPUS = "fitted_from_corpus"
    LITERATURE = "literature"
    ASSUMED = "assumed"
    UNKNOWN = "unknown"


ASSUMED_LABEL = "ASSUMED_PLACEHOLDER"
"""The only admissible ``assumed_label`` for ``ProvStatus.ASSUMED`` (validator V4)."""

SOURCE_KINDS = ("corpus", "doi", "none", "model")
"""Admissible ``Source.kind`` values (DESIGN.md section 1.3)."""


# =============================================================================================
# 1.3  Source / Provenance / Sourced
# =============================================================================================

def _range_from_json(obj: Any) -> tuple[float, float] | None:
    if obj is None:
        return None
    lo, hi = obj
    return (lo, hi)


def _range_to_json(rng: tuple[float, float] | None) -> list | None:
    return None if rng is None else [rng[0], rng[1]]


@dataclass(frozen=True)
class Source:
    """Where a number comes from (DESIGN.md section 1.3).

    ``kind`` in ``{"corpus", "doi", "none", "model"}``; ``doi`` and ``locator`` (table / figure /
    passage) for literature; ``publication_id`` (gen6) and ``safe_exp_ids`` (bundle rows) for
    corpus-measured or corpus-fitted quantities.
    """

    kind: str
    doi: str | None = None
    locator: str | None = None
    publication_id: str | None = None
    safe_exp_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "doi": self.doi,
            "locator": self.locator,
            "publication_id": self.publication_id,
            "safe_exp_ids": list(self.safe_exp_ids),
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any] | None) -> "Source":
        if obj is None:
            return cls(kind="none")
        return cls(
            kind=str(obj.get("kind", "none")),
            doi=obj.get("doi"),
            locator=obj.get("locator"),
            publication_id=obj.get("publication_id"),
            safe_exp_ids=tuple(obj.get("safe_exp_ids") or ()),
        )


@dataclass(frozen=True)
class Provenance:
    """Status plus the evidence for it (DESIGN.md section 1.3).

    ``range`` is the declared (lo, hi) interval in the units of the quantity; mandatory for
    ``assumed`` and ``fitted_from_corpus``.  ``model_id`` / ``fit_manifest_sha256`` identify a
    corpus fit.  ``note`` is free text (derivations of placeholder windows live here).
    """

    status: ProvStatus
    source: Source = field(default_factory=lambda: Source(kind="none"))
    range: tuple[float, float] | None = None
    assumed_label: str | None = None
    model_id: str | None = None
    fit_manifest_sha256: str | None = None
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "source": self.source.to_json(),
            "range": _range_to_json(self.range),
            "assumed_label": self.assumed_label,
            "model_id": self.model_id,
            "fit_manifest_sha256": self.fit_manifest_sha256,
            "note": self.note,
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Provenance":
        return cls(
            status=ProvStatus(obj.get("status", "unknown")),
            source=Source.from_json(obj.get("source")),
            range=_range_from_json(obj.get("range")),
            assumed_label=obj.get("assumed_label"),
            model_id=obj.get("model_id"),
            fit_manifest_sha256=obj.get("fit_manifest_sha256"),
            note=obj.get("note") or "",
        )


@dataclass(frozen=True)
class Sourced:
    """A numeric quantity with unit and provenance (DESIGN.md section 1.3).

    ``value: None`` means "not known" and is the only way to say so.  ``unit`` is a UCUM-like
    string (``"mol/L"``, ``"1"``, ``"kJ/mol"``, ``"s"``, ``"mol/L_aq"``, ``"Cel"``).  The JSON
    form is the flat object of section 1.3: ``value, unit, status, assumed_label, range, source,
    model_id, fit_manifest_sha256, note``.
    """

    value: float | None
    unit: str
    provenance: Provenance

    # convenience views (read-only) -------------------------------------------------------
    @property
    def status(self) -> ProvStatus:
        return self.provenance.status

    @property
    def range(self) -> tuple[float, float] | None:
        return self.provenance.range

    @property
    def source(self) -> Source:
        return self.provenance.source

    def to_json(self) -> dict[str, Any]:
        prov = self.provenance.to_json()
        return {
            "value": self.value,
            "unit": self.unit,
            "status": prov["status"],
            "assumed_label": prov["assumed_label"],
            "range": prov["range"],
            "source": prov["source"],
            "model_id": prov["model_id"],
            "fit_manifest_sha256": prov["fit_manifest_sha256"],
            "note": prov["note"],
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Sourced":
        return cls(
            value=obj.get("value"),
            unit=str(obj.get("unit", "1")),
            provenance=Provenance.from_json(obj),
        )

    @classmethod
    def unknown(cls, unit: str, *, note: str = "", source: Source | None = None) -> "Sourced":
        """A ``value: null`` object with status ``unknown`` (the shape used for every unmeasured
        phase / stream field of the example entries)."""
        return cls(None, unit, Provenance(ProvStatus.UNKNOWN, source or Source(kind="none"),
                                          note=note))


# =============================================================================================
# 1.4  Flags
# =============================================================================================

class Flag(str, Enum):
    """Typed chemistry / applicability / solver flags (DESIGN.md section 1.4).

    Free strings are a validator error (V16).
    """

    # chemistry / model validity
    HIGH_LOADING = "HIGH_LOADING"                       # loading fraction > 0.5 on any ligand
    LOADING_CAP_HIT = "LOADING_CAP_HIT"                 # free ligand < 1e-6 * total on any ligand
    THIRD_PHASE_RISK = "THIRD_PHASE_RISK"               # sourced LOC exceeded
    PHASE_BEHAVIOUR_UNKNOWN = "PHASE_BEHAVIOUR_UNKNOWN"  # LOC null and loading fraction > 0.3
    ACID_UPTAKE_UNMODELLED = "ACID_UPTAKE_UNMODELLED"   # solvating, K_H null, aqueous acid > 1 M
    MEDIUM_STRENGTH_UNMODELLED = "MEDIUM_STRENGTH_UNMODELLED"  # anion varies, no beta_anion
    ANION_ACID_CONFOUNDED = "ANION_ACID_CONFOUNDED"     # corpus p_eff applied where [anion] != [H+]
    MIXED_ORGANIC_UNMODELLED = "MIXED_ORGANIC_UNMODELLED"  # two extractant-role ligands
    OA_ASSUMED = "OA_ASSUMED"                           # parameter fitted on corpus rows, no O/A
    ALKALI_EXCESS = "ALKALI_EXCESS"                     # base exceeded protons in a stage
    SAPONIFICATION_RANGE_UNKNOWN = "SAPONIFICATION_RANGE_UNKNOWN"
    EQUILIBRIUM_ACID_ASSUMED_NOMINAL = "EQUILIBRIUM_ACID_ASSUMED_NOMINAL"  # fit uses nominal acid
    # applicability
    OOD_ACID = "OOD_ACID"
    OOD_LIGAND = "OOD_LIGAND"
    OOD_METAL = "OOD_METAL"
    OOD_LOADING = "OOD_LOADING"
    OOD_ANION = "OOD_ANION"
    OOD_COMPLEXANT = "OOD_COMPLEXANT"
    OOD_OA = "OOD_OA"
    OOD_TEMPERATURE = "OOD_TEMPERATURE"
    OOD_DILUENT = "OOD_DILUENT"
    OOD_MODIFIER = "OOD_MODIFIER"
    OOD_HULL = "OOD_HULL"                               # outside the (log acid, log ligand) hull
    # solver
    NOT_CONVERGED = "NOT_CONVERGED"
    JACOBIAN_SINGULAR = "JACOBIAN_SINGULAR"
    COST_INCOMPLETE = "COST_INCOMPLETE"
    LIGAND_LOSS_NOT_MEASURED = "LIGAND_LOSS_NOT_MEASURED"


INADMISSIBLE_FLAGS: frozenset[Flag] = frozenset({
    Flag.LOADING_CAP_HIT, Flag.THIRD_PHASE_RISK, Flag.ALKALI_EXCESS, Flag.NOT_CONVERGED,
})
"""Flags that make a regime ``INADMISSIBLE`` (DESIGN.md section 1.4)."""

OOD_FLAGS: frozenset[Flag] = frozenset(f for f in Flag if f.name.startswith("OOD_"))
"""Applicability flags; any of them makes a regime ``OUT_OF_DOMAIN`` (DESIGN.md section 1.4)."""

REGIME_STATUSES = ("IN_DOMAIN", "IN_DOMAIN_WITH_CAVEATS", "OUT_OF_DOMAIN", "INADMISSIBLE")
"""Regime status vocabulary (DESIGN.md sections 1.4 and 7.9)."""


# =============================================================================================
# 3.2  The extraction-systems database
# =============================================================================================

class Mechanism(str, Enum):
    """Extraction mechanism of an organic ligand (DESIGN.md section 3.2)."""

    CATION_EXCHANGE = "cation_exchange"
    SOLVATING = "solvating"
    ANION_EXCHANGE = "anion_exchange"       # schema only


@dataclass(frozen=True)
class Stoichiometry:
    """Fixed stoichiometry per metal (DESIGN.md section 3.2).

    ``ligands_per_metal`` q: 3 dimers (cation exchange) or n (solvating);
    ``protons_released_per_metal`` p: 3 (cation exchange), 0 (solvating);
    ``anions_per_metal`` z: 0 (cation exchange), 3 (solvating nitrate).  Unit ``"1"``.
    """

    ligands_per_metal: Sourced
    protons_released_per_metal: Sourced
    anions_per_metal: Sourced


@dataclass(frozen=True)
class LigandSpec:
    """One organic ligand of a system (DESIGN.md section 3.2).

    ``concentration`` is the formal **monomer** concentration, mol/L organic; ``aggregation`` in
    ``{"monomer", "dimer"}``; ``role`` in ``{"extractant", "synergist", "modifier"}``
    (``mechanism`` is None for modifiers); ``scaffold_id`` / ``variant_tag`` link series A-D
    variants across systems (section 3.3).
    """

    name: str
    smiles: str
    canonical_smiles: str
    role: str
    mechanism: Mechanism | None
    concentration: Sourced
    aggregation: str
    scaffold_id: str | None
    variant_tag: str | None
    stoichiometry: Stoichiometry


@dataclass(frozen=True)
class ComplexantSpec:
    """Aqueous hold-back complexant (DESIGN.md section 3.2).

    ``concentration`` mol/L aqueous, total; ``log_beta[metal]`` = cumulative log beta_1, log
    beta_2, ... (M^-m); ``protonation_logk`` cumulative log K_H,j; ``regeneration_fraction``
    recovered fraction per pass (``assumed`` with a range when unknown).
    """

    name: str
    smiles: str | None
    canonical_smiles: str | None
    concentration: Sourced
    log_beta: Mapping[str, tuple[Sourced, ...]]
    protonation_logk: tuple[Sourced, ...]
    regeneration_fraction: Sourced


@dataclass(frozen=True)
class MediumSpec:
    """Aqueous medium (DESIGN.md section 3.2).

    ``anion`` in ``{"nitrate", "chloride", "sulfate", "perchlorate", "carboxylate"}``;
    ``salting_anion_M`` and ``ionic_strength_M`` mol/L; ``temperature_C`` degC.
    """

    acid: str
    acid_class: str
    anion: str
    salting_agent: str | None
    salting_anion_M: Sourced
    ionic_strength_M: Sourced
    temperature_C: Sourced


@dataclass(frozen=True)
class CationExchangeParams:
    """Mass-action parameters of a cation-exchange ligand (DESIGN.md sections 3.2 and 5.2 a).

    ``log D0_i = log K_i + a_dimer log[(HA)2]_f - b_proton log[H+]``; ``log_k`` per metal on the
    **dimer**, concentration basis; ``k_reported_as`` in ``{"log_k", "pH50"}`` (pH50 converts to
    log K only in the tracer limit, section 4.7); ``beta_anion`` per-metal anion complexation
    (None -> not modelled); ``medium_anion`` must equal ``MediumSpec.anion`` (V7).
    """

    medium_anion: str
    temperature_band: str
    log_k: Mapping[str, Sourced]
    a_dimer: Sourced
    b_proton: Sourced
    k_reported_as: str
    beta_anion: Mapping[str, tuple[Sourced, ...]] | None
    saponification_degree_studied: tuple[float, float] | None
    delta_h_kj_mol: Sourced | None


@dataclass(frozen=True)
class SolvatingParams:
    """Mass-action parameters of a solvating ligand (DESIGN.md sections 3.2 and 5.2 b).

    ``log D0_i = log K_i + n_solvation log[L]_f + p_anion log[anion] + p_h log[H+]``; ``log_k``
    per metal at [anion] and [L] in mol/L; ``k_acid_uptake`` K_H for HNO3 + L <=> HNO3.L in
    (mol/L)^-2 (``value None`` -> ``ACID_UPTAKE_UNMODELLED`` above 1 M acid).
    """

    medium_anion: str
    temperature_band: str
    log_k: Mapping[str, Sourced]
    n_solvation: Sourced
    p_anion: Sourced
    p_h: Sourced
    k_acid_uptake: Sourced
    delta_h_kj_mol: Sourced | None


@dataclass(frozen=True)
class PhaseBehaviour:
    """Phase-behaviour data; each value None unless measured or literature (validator V5).

    ``loc_metal_M`` / ``loc_acid_M`` limiting organic concentrations mol/L; ``disengagement_s``
    seconds; ``ligand_loss_mol_per_L_aq`` mol per litre of aqueous contacted;
    ``max_loading_fraction_studied`` dimensionless.
    """

    loc_metal_M: Sourced
    loc_acid_M: Sourced
    third_phase_observed: Sourced
    disengagement_s: Sourced
    ligand_loss_mol_per_L_aq: Sourced
    max_loading_fraction_studied: Sourced
    regenerability_note: str


@dataclass(frozen=True)
class StreamRecord:
    """A scrub or strip liquor as reported by the source (DESIGN.md section 3.2).

    ``kind`` in ``{"scrub", "strip"}``; concentrations mol/L except ``metals_mM`` (mM);
    ``oa_ratio`` volumetric; ``stages`` count; ``fraction_removed`` per metal (0-1).
    """

    kind: str
    acid_M: Sourced
    anion_M: Sourced
    complexant_M: Sourced
    metals_mM: Mapping[str, Sourced]
    oa_ratio: Sourced
    stages: Sourced
    fraction_removed: Mapping[str, Sourced]
    note: str


@dataclass(frozen=True)
class DistributionRecord:
    """One measured D (DESIGN.md section 3.2); corpus rows are kept one per bundle row.

    ``record_id``: corpus ``safe_exp_id``; literature ``"doi:<doi>#<locator>#<n>"``.  ``d`` and
    ``log_d`` (log10) copied unchanged.  ``acid_nominal_M`` / ``acid_eq_M`` / ``anion_M`` /
    ``complexant_M`` mol/L; ``ligand_M`` per organic ligand name, formal concentration mol/L;
    ``metals_initial_mM`` mM (corpus semantics ASSUMED initial aqueous); ``oa_ratio`` None in the
    corpus; ``duplicate_flag`` in ``{None, "UNIT_SLIP_DUPLICATE", "TIED_D"}``.
    """

    record_id: str
    metal: str
    d: float
    log_d: float
    acid_nominal_M: float | None
    acid_eq_M: float | None
    anion_M: float | None
    ligand_M: Mapping[str, float]
    complexant_M: float | None
    metals_initial_mM: Mapping[str, float]
    oa_ratio: float | None
    temperature_C: float | None
    contact_time_min: float | None
    diluent_name: str | None
    publication_id: str | None
    experiment_series_id: str | None
    replicate_id: str | None
    loading_series_id: str | None
    is_tracer: bool
    fit_eligible: bool
    fit_ineligible_reason: str | None
    duplicate_flag: str | None
    provenance: Provenance


@dataclass(frozen=True)
class ApplicabilityDomain:
    """Intervals actually studied for one parameter set (DESIGN.md sections 3.2 and 5.5).

    Concentration axes are log10 (``log_acid`` of mol/L, ``log_ligand`` per ligand of mol/L,
    ``log_metal_total_mM`` of mM, ``log_complexant`` of mol/L); ``loading_fraction``, ``oa_ratio``,
    ``saponification_degree`` linear; ``temperature_C`` degC; ``hull_vertices`` in
    (log acid, log primary ligand) (None for hand-entered literature domains: box only).
    """

    anion: str
    diluent_family: str
    modifiers: tuple[str, ...]
    log_acid: tuple[float, float]
    log_ligand: Mapping[str, tuple[float, float]]
    log_metal_total_mM: tuple[float, float] | None
    loading_fraction: tuple[float, float] | None
    log_complexant: tuple[float, float] | None
    oa_ratio: tuple[float, float] | None
    temperature_C: tuple[float, float] | None
    saponification_degree: tuple[float, float] | None
    hull_vertices: tuple[tuple[float, float], ...] | None
    n_records: int
    n_publications: int


@dataclass(frozen=True)
class SystemEntry:
    """One extraction system of the database (DESIGN.md section 3.2; ``systems/<id>.json``).

    ``family`` in ``{acidic_organophosphorus, diglycolamide, phen_carboxamide, n_donor,
    carboxylic_acid, amine, other}``; ``origin`` in ``{corpus, literature, mixed, hypothetical}``;
    ``params[ligand name][temperature band]``; ``applicability`` keyed ``"<ligand>|<band>"``;
    ``diluent`` = ``{"name": str|None, "family": str, "components": [{"name", "vol_fraction"}]}``;
    ``oxidation_state_routes`` items ``{"metal", "from", "to", "method", "doi", "status"}``;
    ``direction_prior`` = ``{"pair", "sign", "source", "note"}`` (gen15 pre-screen only, never a D
    source).
    """

    schema_version: str
    system_id: str
    name: str
    family: str
    origin: str
    organic_ligands: tuple[LigandSpec, ...]
    aqueous_complexants: tuple[ComplexantSpec, ...]
    diluent: Mapping
    medium: MediumSpec
    params: Mapping[str, Mapping[str, CationExchangeParams | SolvatingParams]]
    phase: PhaseBehaviour
    stream_records: tuple[StreamRecord, ...]
    oxidation_state_routes: tuple[Mapping, ...]
    direction_prior: Mapping | None
    applicability: Mapping[str, ApplicabilityDomain]
    records: tuple[DistributionRecord, ...]
    notes: str


# ---- 3.6 validator results and 5.7 model-build refusal ---------------------------------------

@dataclass(frozen=True)
class Violation:
    """One validator finding (DESIGN.md section 3.6): ``level`` in ``{"error", "warning"}``,
    ``path`` a dotted / bracketed location inside the entry, ``message`` free text naming the rule
    (V1-V17, W1-W4)."""

    level: str
    path: str
    message: str


class SystemValidationError(ValueError):
    """Raised by ``load_system`` when ``validate_entry`` returns any error-level violation."""

    def __init__(self, violations: list[Violation] | tuple[Violation, ...]):
        self.violations: tuple[Violation, ...] = tuple(violations)
        errors = [v for v in self.violations if v.level == "error"]
        lines = [f"{v.level}: {v.path}: {v.message}" for v in self.violations]
        super().__init__(f"{len(errors)} error(s) in system entry\n" + "\n".join(lines))


@dataclass(frozen=True)
class ModelBuildError:
    """Returned (never raised) by ``build_system_model`` when it refuses (DESIGN.md section 5.7):
    cross-anion transfer, a missing ``log_k``, or a complexant without ``log_beta`` for a feed
    metal.  Callers test ``isinstance(result, ModelBuildError)``."""

    reason: str


# =============================================================================================
# 5.1 / 5.6  D-model protocol
# =============================================================================================

@dataclass(frozen=True)
class StageState:
    """Candidate equilibrium of one stage (DESIGN.md section 5.1).

    ``v_aq_L`` / ``v_org_L`` per-hour phase volumes (flows, L/h); ``x_total`` aqueous total metal
    mol/L (free + complexed); ``y`` organic metal mol/L summed over ligands; ``y_by_ligand``
    ligand -> metal -> mol/L; ``h`` aqueous [H+], ``anion`` aqueous [anion], ``c_free`` free
    deprotonated complexant, all mol/L; ``ligand_total`` / ``ligand_free`` per ligand mol/L organic
    (dimer basis for dimers); ``acid_in_org`` per ligand mol/L organic (HNO3.L); ``alkali_reserve``
    mol/L organic; ``temperature_C`` degC.
    """

    v_aq_L: float
    v_org_L: float
    x_total: dict[str, float]
    y: dict[str, float]
    y_by_ligand: dict[str, dict[str, float]]
    h: float
    anion: float
    c_free: float
    ligand_total: dict[str, float]
    ligand_free: dict[str, float]
    acid_in_org: dict[str, float]
    alkali_reserve: float
    temperature_C: float


@dataclass(frozen=True)
class DEval:
    """One D evaluation (DESIGN.md section 5.1): effective (complexant-corrected) ``log_d`` /
    ``d``; analytic partials ``d log10 D / d ln(.)`` with respect to free ligand, [H+], anion and
    free complexant; chemistry / applicability ``flags``; per-axis ``ood_distance`` (log10 units,
    0 inside)."""

    log_d: float
    d: float
    dlogd_dlnL: float
    dlogd_dlnh: float
    dlogd_dlnanion: float
    dlogd_dlnc: float
    flags: frozenset[Flag]
    ood_distance: dict[str, float]


@runtime_checkable
class ActivityModel(Protocol):
    """Non-ideality hook (DESIGN.md section 5.6).  Default behaviour: unit activity coefficients
    and ``organic_free_ligand`` returning ``state.ligand_free[ligand]``.  ``EffectiveCapacity(phi)``
    (WB2) is the only implementation and is exploratory."""

    def aqueous_gamma(self, state: StageState) -> dict[str, float]: ...

    def organic_free_ligand(self, state: StageState, ligand: str) -> float: ...


@runtime_checkable
class DModel(Protocol):
    """Distribution model of one ligand for a set of metals (DESIGN.md section 5.1).

    ``q`` / ``p`` / ``z`` per metal: ligands, protons released and anions transported per metal.
    ``evaluate`` never raises on a physically odd state; it returns a ``DEval`` with flags.
    """

    ligand: str
    mechanism: Mechanism
    metals: tuple[str, ...]
    q: Mapping[str, float]
    p: Mapping[str, float]
    z: Mapping[str, float]
    domain: ApplicabilityDomain | None
    provenance: Provenance

    def evaluate(self, metal: str, state: StageState,
                 activity: ActivityModel | None = None) -> DEval: ...


# =============================================================================================
# 6.1 / 6.2  Streams and stage diagnostics
# =============================================================================================

@dataclass(frozen=True)
class AqStream:
    """Aqueous stream (DESIGN.md section 6.1).  ``flow_L_h`` L/h; ``metals`` total (free +
    complexed) mol/L; ``h`` [H+], ``anion``, ``complexant_total``, ``sodium`` mol/L; ``labels``
    origin sub-species metal -> label -> mol/L (section 7.8), None when not computed."""

    flow_L_h: float
    metals: dict[str, float]
    h: float
    anion: float
    complexant_total: float
    sodium: float
    labels: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True)
class OrgStream:
    """Organic stream (DESIGN.md section 6.1).  ``flow_L_h`` L/h; ``metals`` mol/L summed over
    ligands; ``metals_by_ligand`` ligand -> metal -> mol/L; ``ligand_total`` / ``ligand_free`` per
    ligand mol/L (dimer basis for dimers); ``acid_in_org`` per ligand mol/L (HNO3.L);
    ``alkali_reserve`` mol/L organic; ``labels`` as for ``AqStream``."""

    flow_L_h: float
    metals: dict[str, float]
    metals_by_ligand: dict[str, dict[str, float]]
    ligand_total: dict[str, float]
    ligand_free: dict[str, float]
    acid_in_org: dict[str, float]
    alkali_reserve: float
    labels: dict[str, dict[str, float]] | None = None


@dataclass(frozen=True)
class StageDiagnostics:
    """Output of ``solve_stage`` besides the two streams (DESIGN.md section 6.2).

    ``d`` effective D per metal; ``d_by_ligand`` ligand -> metal -> D; ``loading_fraction`` per
    ligand (lambda_k, section 5.4); ``iterations``; ``residual_max`` scaled; ``branch`` 1 (acid
    balance) or 2 (alkali excess, h = H_MIN); ``flags``; ``ood_distance`` per axis (max over the
    metals); ``status`` in ``{"converged", "failed"}``.
    """

    d: dict[str, float]
    d_by_ligand: dict[str, dict[str, float]]
    loading_fraction: dict[str, float]
    iterations: int
    residual_max: float
    branch: int
    flags: frozenset[Flag]
    ood_distance: dict[str, float]
    status: str


# =============================================================================================
# 7.1 / 7.9  Cascade specification and result
# =============================================================================================

@dataclass(frozen=True)
class CascadeSpec:
    """Countercurrent cascade specification (DESIGN.md section 7.1).

    Stage counts ``n_ext`` (>= 1), ``n_scr``, ``n_str`` (0 allowed); ``feed`` (flow A), ``scrub``
    (flow S; may carry complexant and/or the target metal), ``strip`` (flow W); ``organic_flow_L_h``
    O; ``ligand_total`` L_T per ligand, mol/L organic (dimer basis for dimers);
    ``saponification_degree`` s in [0, 1] (B_org at the organic inlet of stage 0 = s * [HA]_T,
    monomer basis; 0 for solvating); ``feed_stage`` j_F and ``scrub_return_stage`` j_R in
    ``[0, n_ext - 1]`` (None -> ``n_ext - 1``); ``f_bleed`` fraction of organic replaced by
    ``fresh_organic`` per pass; ``target`` metal for the origin-labelled pass (section 7.8).
    Validation (``ValueError`` on bad flows, stage indices, degree, unknown metals) is performed by
    ``cascade.solve_cascade``, not by this dataclass.
    """

    n_ext: int
    n_scr: int
    n_str: int
    feed: AqStream
    scrub: AqStream
    strip: AqStream
    organic_flow_L_h: float
    ligand_total: dict[str, float]
    saponification_degree: float
    feed_stage: int | None = None
    scrub_return_stage: int | None = None
    f_bleed: float = 0.0
    fresh_organic: OrgStream | None = None
    target: str = ""


@dataclass(frozen=True)
class CascadeResult:
    """Result of ``solve_cascade`` (DESIGN.md section 7.9).

    ``status`` in ``{"converged_newton", "converged_ss", "failed", "invalid_spec"}``; per-stage
    streams and diagnostics; the named outlet streams (``product`` is the strip liquor, or the
    loaded organic when ``n_str == 0``); ``origin`` the four quantities of section 7.8 plus label
    sums (None when no target); ``balances`` relative in-minus-out per ledger and their maximum
    ``balance_rel_max``; ``flags`` / ``ood_distance`` union over stages, max distance per axis;
    ``regime_status`` in ``REGIME_STATUSES``; ``assumptions`` the static idealisation tuple.  A
    failed result carries NaN metrics and is never raised.
    """

    status: str
    stages_aq: tuple[AqStream, ...]
    stages_org: tuple[OrgStream, ...]
    diagnostics: tuple[StageDiagnostics, ...]
    raffinate: AqStream
    product: AqStream | OrgStream
    scrub_raffinate: AqStream | None
    loaded_organic: OrgStream
    stripped_organic: OrgStream
    origin: Mapping[str, float] | None
    residual_max: float
    iterations: int
    balance_rel_max: float
    balances: Mapping[str, float]
    flags: frozenset[Flag]
    ood_distance: Mapping[str, float]
    regime_status: str
    assumptions: tuple[str, ...]


# =============================================================================================
# 8  Process metrics
# =============================================================================================

@dataclass(frozen=True)
class ProcessMetrics:
    """Output of ``metrics.compute_metrics`` (DESIGN.md section 8); every numeric field is None /
    NaN when ``result.status == "failed"``.

    Purities are fractions of the target T among metals in the product on mol / mass / M2O3 oxide
    basis; ``recovery_from_feed`` (headline), ``recovery_total``, ``scrub_target_return`` (None
    when the scrub carries no T) and ``net_product_mol_h`` (mol/h) are the origin-labelled
    quantities of section 7.8; ``enrichment_factor[I]`` = (n_T/n_I)_product / (n_T/n_I)_feed;
    ``sf_tracer[section][I]`` = D_T/D_I at the section's tracer conditions; ``sf_by_stage[j][I]``;
    stage counts and flow ratios (``oa_ext`` = O/A, ``s_over_a`` = S/A, ``w_over_a`` = W/A);
    ``throughput_mol_T_per_h_per_L_org`` = n_T,product / O, ``throughput_kg_oxide_per_h``;
    ``max_loading_fraction`` per ligand over stages; ``phase`` items copied from the entry as
    ``Sourced`` (``loc_metal_M``, ``disengagement_s``, ``ligand_loss``, ``third_phase_observed``,
    ``regenerability_note``); ``consumption`` mapping of reagent -> amount (mol/h or L/h, and per
    mol T and per kg oxide, keys named with their unit suffix by WB3);
    ``cost_proxy_per_kg_oxide`` from ``config/prices.json`` (NaN with ``COST_INCOMPLETE`` when any
    item is missing); ``on_spec``; ``regime_status``; ``flags``; ``ood_distance`` per axis.
    """

    purity_mol: float | None
    purity_mass: float | None
    purity_oxide: float | None
    recovery_from_feed: float | None
    recovery_total: float | None
    scrub_target_return: float | None
    net_product_mol_h: float | None
    enrichment_factor: Mapping[str, float]
    sf_tracer: Mapping[str, Mapping[str, float]]
    sf_by_stage: tuple[Mapping[str, float], ...]
    n_stages_total: int
    n_ext: int
    n_scr: int
    n_str: int
    oa_ext: float
    s_over_a: float
    w_over_a: float
    throughput_mol_T_per_h_per_L_org: float | None
    throughput_kg_oxide_per_h: float | None
    max_loading_fraction: Mapping[str, float]
    phase: Mapping[str, Any]
    consumption: Mapping[str, float]
    cost_proxy_per_kg_oxide: float | None
    on_spec: bool | None
    regime_status: str
    flags: frozenset[Flag]
    ood_distance: Mapping[str, float]
