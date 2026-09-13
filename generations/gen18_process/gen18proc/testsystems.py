"""``testsystems.py`` — shared synthetic systems for the test suite (DESIGN.md section 12.1).

EVERY NUMBER IN THIS MODULE IS A SYNTHETIC TEST FIXTURE, NOT A LITERATURE VALUE.  The fixtures
resemble a PC88A-like cation-exchange system in chloride and a TODGA-like solvating system in
nitrate only in their mechanism and order of magnitude; the ``Sourced`` objects carry status
``assumed`` with ``ASSUMED_PLACEHOLDER`` and the note ``"synthetic test fixture"`` so that no
value can be mistaken for a database entry.

Exact content (section 12.1):

* ``two_metal_cation_exchange()``: metals Pr, Nd; ``log_k = {"Pr": -2.10, "Nd": -1.95}`` (dimer
  basis), ``a = b = 3``, ``q = 3``, ``p = 3``, ``z = 0``, chloride, ``[HA]_T = 0.8`` (dimer 0.4),
  no complexant; domain box log acid [-3, 1], log ligand [-2, 0.5].
* ``two_metal_solvating()``: metals Pr, Nd; ``log_k = {"Pr": 1.50, "Nd": 1.80}``, ``n = 2.7``,
  ``p_anion = 2.0``, ``p_h = 0``, ``K_H = 0.5`` L^2/mol^2, ``z = 3``, nitrate, ``L_T = 0.1``.
* ``constant_d_system(log_d)``; ``complexant_spec()``: Pr ``log_beta = (3.0,)``, Nd
  ``log_beta = (2.0,)``, protonation ``log K_H = (2.0,)``, regeneration 0.9 (assumed, [0.5, 1]).
* ``feed_prnd()``: chloride, Pr 0.025 M, Nd 0.075 M, ``h = 0.01``, anion 0.31, flow 1.0 L/h.

Extras for the other work blocks: ``feed_prnd_nitrate()`` (the same metals in 3 M HNO3),
``lean_organic(system, ...)`` (an ``OrgStream`` with the fixture's ligand total and no metal),
and the ligand-name / ligand-total constants ``CE_LIGAND``, ``CE_LT_DIMER``, ``SOLV_LIGAND``,
``SOLV_LT``, ``CONSTANT_LIGAND``.  Units: mol/L, L/h, degC.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .dmodel import ASSUMPTIONS, ConstantD, SystemModel, build_system_model
from .types import (
    ASSUMED_LABEL,
    ActivityModel,
    ApplicabilityDomain,
    AqStream,
    CationExchangeParams,
    ComplexantSpec,
    LigandSpec,
    Mechanism,
    MediumSpec,
    ModelBuildError,
    OrgStream,
    PhaseBehaviour,
    Provenance,
    ProvStatus,
    SolvatingParams,
    Source,
    Sourced,
    Stoichiometry,
    SystemEntry,
)

__all__ = [
    "CE_LIGAND", "CE_LT_DIMER", "CE_HA_TOTAL", "SOLV_LIGAND", "SOLV_LT", "CONSTANT_LIGAND",
    "two_metal_cation_exchange", "two_metal_solvating", "constant_d_system", "complexant_spec",
    "feed_prnd", "feed_prnd_nitrate", "lean_organic", "cation_exchange_entry", "solvating_entry",
    "assumed",
]

_NOTE = "synthetic test fixture (DESIGN.md section 12.1); not a literature value"

CE_LIGAND = "PC88A"
"""Ligand name of the cation-exchange fixture (synthetic parameters)."""
CE_HA_TOTAL = 0.8
"""Formal monomer concentration [HA]_T of the cation-exchange fixture, mol/L."""
CE_LT_DIMER = CE_HA_TOTAL / 2.0
"""Dimer total [(HA)2]_T = [HA]_T / 2 used as ``ligand_total`` in the code, mol/L."""
SOLV_LIGAND = "TODGA"
"""Ligand name of the solvating fixture (synthetic parameters)."""
SOLV_LT = 0.1
"""Ligand total of the solvating fixture, mol/L."""
CONSTANT_LIGAND = "constant"
"""Ligand name used by ``constant_d_system``."""


def assumed(value: float | None, unit: str = "1", rng: tuple[float, float] | None = None, *,
            note: str = _NOTE) -> Sourced:
    """An ``assumed`` / ``ASSUMED_PLACEHOLDER`` fixture value with a declared range."""
    if rng is None and value is not None:
        rng = (float(value), float(value))
    return Sourced(value, unit, Provenance(ProvStatus.ASSUMED, Source(kind="none", locator=note),
                                           rng, ASSUMED_LABEL, note=note))


def _unknown_phase() -> PhaseBehaviour:
    return PhaseBehaviour(
        loc_metal_M=Sourced.unknown("mol/L"), loc_acid_M=Sourced.unknown("mol/L"),
        third_phase_observed=Sourced.unknown("1"), disengagement_s=Sourced.unknown("s"),
        ligand_loss_mol_per_L_aq=Sourced.unknown("mol/L_aq"),
        max_loading_fraction_studied=Sourced.unknown("1"), regenerability_note="")


def _medium(acid: str, acid_class: str, anion: str) -> MediumSpec:
    return MediumSpec(acid=acid, acid_class=acid_class, anion=anion, salting_agent=None,
                      salting_anion_M=Sourced.unknown("mol/L"),
                      ionic_strength_M=Sourced.unknown("mol/L"),
                      temperature_C=assumed(25.0, "Cel", (20.0, 30.0)))


def _ligand(name: str, smiles: str, mechanism: Mechanism, aggregation: str, q: float, p: float,
            z: float, concentration: float) -> LigandSpec:
    return LigandSpec(
        name=name, smiles=smiles, canonical_smiles=smiles, role="extractant",
        mechanism=mechanism, concentration=assumed(concentration, "mol/L"),
        aggregation=aggregation, scaffold_id=None, variant_tag=None,
        stoichiometry=Stoichiometry(ligands_per_metal=assumed(q),
                                    protons_released_per_metal=assumed(p),
                                    anions_per_metal=assumed(z)))


def _domain(anion: str, ligand: str) -> ApplicabilityDomain:
    return ApplicabilityDomain(
        anion=anion, diluent_family="aliphatic_hydrocarbon", modifiers=(),
        log_acid=(-3.0, 1.0), log_ligand={ligand: (-2.0, 0.5)}, log_metal_total_mM=None,
        loading_fraction=None, log_complexant=None, oa_ratio=None, temperature_C=None,
        saponification_degree=None, hull_vertices=None, n_records=0, n_publications=0)


def _entry(system_id: str, name: str, family: str, ligands: tuple[LigandSpec, ...],
           medium: MediumSpec, params: Mapping[str, Any], applicability: Mapping[str, Any],
           complexants: tuple[ComplexantSpec, ...] = ()) -> SystemEntry:
    return SystemEntry(
        schema_version="gen18.1", system_id=system_id, name=name, family=family,
        origin="hypothetical", organic_ligands=ligands, aqueous_complexants=complexants,
        diluent={"name": None, "family": "aliphatic_hydrocarbon", "components": []},
        medium=medium, params=params, phase=_unknown_phase(), stream_records=(),
        oxidation_state_routes=(), direction_prior=None, applicability=applicability,
        records=(), notes=_NOTE)


# ---------------------------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------------------------

def cation_exchange_entry(log_k: Mapping[str, float] | None = None, *, a: float = 3.0,
                          b: float = 3.0) -> SystemEntry:
    """The synthetic PC88A-like chloride entry (section 12.1) as a ``SystemEntry``."""
    logk = dict(log_k) if log_k is not None else {"Pr": -2.10, "Nd": -1.95}
    params = CationExchangeParams(
        medium_anion="chloride", temperature_band="20-30C",
        log_k={m: assumed(v, "1", (v - 1.0, v + 1.0)) for m, v in logk.items()},
        a_dimer=assumed(a, "1", (2.0, 3.0)), b_proton=assumed(b, "1", (2.0, 3.0)),
        k_reported_as="log_k", beta_anion=None, saponification_degree_studied=None,
        delta_h_kj_mol=None)
    ligand = _ligand(CE_LIGAND, "CCCCC(CC)COP(=O)(O)CC(CC)CCCC", Mechanism.CATION_EXCHANGE,
                     "dimer", 3.0, 3.0, 0.0, CE_HA_TOTAL)
    return _entry("sys_test_cation_exchange", "synthetic cation-exchange fixture",
                  "acidic_organophosphorus", (ligand,), _medium("HCl", "chloride", "chloride"),
                  {CE_LIGAND: {"20-30C": params}},
                  {f"{CE_LIGAND}|20-30C": _domain("chloride", CE_LIGAND)})


def solvating_entry(log_k: Mapping[str, float] | None = None, *, n: float = 2.7,
                    p_anion: float = 2.0, p_h: float = 0.0, k_acid_uptake: float | None = 0.5,
                    ) -> SystemEntry:
    """The synthetic TODGA-like nitrate entry (section 12.1) as a ``SystemEntry``."""
    logk = dict(log_k) if log_k is not None else {"Pr": 1.50, "Nd": 1.80}
    kh = (assumed(k_acid_uptake, "L2/mol2", (0.0, 2.0)) if k_acid_uptake is not None
          else Sourced.unknown("L2/mol2", note="K_H unknown in this fixture"))
    params = SolvatingParams(
        medium_anion="nitrate", temperature_band="20-30C",
        log_k={m: assumed(v, "1", (v - 1.0, v + 1.0)) for m, v in logk.items()},
        n_solvation=assumed(n, "1", (2.0, 3.5)), p_anion=assumed(p_anion, "1", (1.0, 3.0)),
        p_h=assumed(p_h, "1", (0.0, 1.0)), k_acid_uptake=kh, delta_h_kj_mol=None)
    ligand = _ligand(SOLV_LIGAND, "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC",
                     Mechanism.SOLVATING, "monomer", n, 0.0, 3.0, SOLV_LT)
    return _entry("sys_test_solvating", "synthetic solvating fixture", "diglycolamide",
                  (ligand,), _medium("HNO3", "nitrate", "nitrate"),
                  {SOLV_LIGAND: {"20-30C": params}},
                  {f"{SOLV_LIGAND}|20-30C": _domain("nitrate", SOLV_LIGAND)})


# ---------------------------------------------------------------------------------------------
# system models
# ---------------------------------------------------------------------------------------------

def _build(entry: SystemEntry, anion: str, complexant: ComplexantSpec | None,
           activity: ActivityModel | None, temperature_C: float,
           parameter_draw: Mapping[str, float] | None) -> SystemModel:
    model = build_system_model(entry, feed_anion=anion, temperature_C=temperature_C,
                               params_source="literature", complexant=complexant,
                               activity=activity, parameter_draw=parameter_draw)
    if isinstance(model, ModelBuildError):
        raise RuntimeError(f"fixture failed to build: {model.reason}")
    return model


def two_metal_cation_exchange(*, complexant: ComplexantSpec | None = None,
                              activity: ActivityModel | None = None,
                              log_k: Mapping[str, float] | None = None,
                              temperature_C: float = 25.0,
                              parameter_draw: Mapping[str, float] | None = None) -> SystemModel:
    """Pr/Nd cation-exchange fixture (chloride; ligand ``CE_LIGAND``, dimer total
    ``CE_LT_DIMER``)."""
    return _build(cation_exchange_entry(log_k), "chloride", complexant, activity, temperature_C,
                  parameter_draw)


def two_metal_solvating(*, complexant: ComplexantSpec | None = None,
                        activity: ActivityModel | None = None,
                        log_k: Mapping[str, float] | None = None,
                        k_acid_uptake: float | None = 0.5, temperature_C: float = 25.0,
                        parameter_draw: Mapping[str, float] | None = None) -> SystemModel:
    """Pr/Nd solvating fixture (nitrate; ligand ``SOLV_LIGAND``, total ``SOLV_LT``)."""
    return _build(solvating_entry(log_k, k_acid_uptake=k_acid_uptake), "nitrate", complexant,
                  activity, temperature_C, parameter_draw)


def constant_d_system(log_d: Mapping[str, float], *, q: Any = 0.0, p: Any = 0.0, z: Any = 0.0,
                      temperature_C: float = 25.0) -> SystemModel:
    """A ``SystemModel`` with one ``ConstantD`` ligand named ``CONSTANT_LIGAND``."""
    model = ConstantD(dict(log_d), q=q, p=p, z=z, ligand=CONSTANT_LIGAND)
    return SystemModel(entry=None, dmodels={CONSTANT_LIGAND: model}, complexant=None,
                       activity=None, temperature_C=temperature_C, assumptions=ASSUMPTIONS,
                       params_source="constant")


def complexant_spec(*, log_beta: Mapping[str, tuple[float, ...]] | None = None,
                    protonation_logk: tuple[float, ...] = (2.0,),
                    concentration: float = 0.05) -> ComplexantSpec:
    """Synthetic hold-back complexant: Pr log beta 3.0, Nd 2.0, log K_H 2.0, regeneration 0.9."""
    lb = dict(log_beta) if log_beta is not None else {"Pr": (3.0,), "Nd": (2.0,)}
    return ComplexantSpec(
        name="synthetic_complexant", smiles=None, canonical_smiles=None,
        concentration=assumed(concentration, "mol/L", (0.0, 0.5)),
        log_beta={m: tuple(assumed(v, "1", (v - 1.0, v + 1.0)) for v in vals)
                  for m, vals in lb.items()},
        protonation_logk=tuple(assumed(v, "1", (v - 1.0, v + 1.0)) for v in protonation_logk),
        regeneration_fraction=assumed(0.9, "1", (0.5, 1.0)))


# ---------------------------------------------------------------------------------------------
# streams
# ---------------------------------------------------------------------------------------------

def feed_prnd(*, flow_L_h: float = 1.0, complexant_total: float = 0.0) -> AqStream:
    """Chloride feed: Pr 0.025 M, Nd 0.075 M, [H+] 0.01, chloride 0.31, flow 1.0 L/h."""
    return AqStream(flow_L_h=flow_L_h, metals={"Pr": 0.025, "Nd": 0.075}, h=0.01, anion=0.31,
                    complexant_total=complexant_total, sodium=0.0)


def feed_prnd_nitrate(*, flow_L_h: float = 1.0, complexant_total: float = 0.0,
                      acid_M: float = 3.0) -> AqStream:
    """Nitrate feed for the solvating fixture: Pr 0.025 M, Nd 0.075 M in ``acid_M`` HNO3."""
    return AqStream(flow_L_h=flow_L_h, metals={"Pr": 0.025, "Nd": 0.075}, h=acid_M, anion=acid_M,
                    complexant_total=complexant_total, sodium=0.0)


def lean_organic(system: SystemModel, *, flow_L_h: float = 1.0,
                 ligand_total: Mapping[str, float] | None = None,
                 alkali_reserve: float = 0.0, metals: Mapping[str, float] | None = None,
                 ) -> OrgStream:
    """An organic stream with the fixture's ligand total (``CE_LT_DIMER`` for the cation-exchange
    ligand, ``SOLV_LT`` for the solvating one, 1.0 for a constant-D ligand) and no metal unless
    ``metals`` is given (attributed to the first ligand)."""
    totals: dict[str, float] = {}
    for name in system.ligands:
        if ligand_total is not None and name in ligand_total:
            totals[name] = float(ligand_total[name])
        elif name == CE_LIGAND:
            totals[name] = CE_LT_DIMER
        elif name == SOLV_LIGAND:
            totals[name] = SOLV_LT
        else:
            totals[name] = 1.0
    names = tuple(system.metals)
    y = {m: 0.0 for m in names}
    if metals:
        y.update({m: float(v) for m, v in metals.items()})
    by_ligand = {name: ({m: 0.0 for m in y} if i else dict(y))
                 for i, name in enumerate(system.ligands)}
    return OrgStream(flow_L_h=flow_L_h, metals=y, metals_by_ligand=by_ligand,
                     ligand_total=totals, ligand_free=dict(totals),
                     acid_in_org={name: 0.0 for name in totals}, alkali_reserve=alkali_reserve)
