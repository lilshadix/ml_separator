"""``literature.py`` — chemical identities, placeholder builders and the literature entries
(DESIGN.md section 4.7; numbers from ``LITERATURE_NOTES.md`` S1, S2, S4 and the task document).

Nothing here is a measurement made in this repository.  Every numeric value of the two Pr/Nd
entries is ``assumed`` with ``ASSUMED_PLACEHOLDER``, a range and the DOI where a person must
verify it (open item U4).  ``EXTRACTANT_SMILES`` and ``MODIFIER_SMILES`` are identity tables
(SMILES of named reagents), not literature numbers.

Units: concentrations mol/L; ``log`` is log10; pH is -log10[H+] (concentration basis);
temperatures degC.  The cation-exchange log K is on the **dimer** basis
(``log D = log K + a log[(HA)2]_f - b log[H+]``, DESIGN.md section 5.2 a).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from .systems import (
    ASSUMED_LABEL,
    ComplexantSpec,
    Provenance,
    ProvStatus,
    Source,
    Sourced,
    SystemEntry,
    entry_from_json,
    system_id,
)

__all__ = [
    "EXTRACTANT_SMILES", "MODIFIER_SMILES", "SF_ND_PR_PLACEHOLDER", "PHASE_LITERATURE",
    "placeholder", "derive_logk_from_extraction", "ph50_to_logk", "pc88a_prnd_entry",
    "cyanex272_prnd_entry", "todga_hydrophilic_complexant_placeholder", "literature_entries",
]

# ---- identity tables (SMILES as written; the canonical form is recomputed by RDKit) ---------
EXTRACTANT_SMILES: dict[str, str] = {
    "HDEHP": "CCCCC(CC)COP(=O)(O)OCC(CC)CCCC",          # di(2-ethylhexyl) phosphoric acid
    "D2EHPA": "CCCCC(CC)COP(=O)(O)OCC(CC)CCCC",         # same molecule, industrial name
    "TBP": "CCCCOP(=O)(OCCCC)OCCCC",                    # tributyl phosphate
    "DHOA": "CCCCCCN(CCCCCC)C(=O)CCCCCCC",              # N,N-dihexyloctanamide
    "DOHyA": "CCCCCCCCN(CCCCCCCC)C(=O)CO",              # N,N-dioctyl-2-hydroxyacetamide (bundle)
    "PC88A": "CCCCC(CC)COP(=O)(O)CC(CC)CCCC",           # 2-ethylhexyl hydrogen 2-ethylhexyl-
    #                                                    phosphonate, HEH[EHP]
    "Cyanex 272": "CC(C)(C)CC(C)CP(=O)(O)CC(C)CC(C)(C)C",  # bis(2,4,4-trimethylpentyl)phosphinic
    #                                                        acid
    "TODGA": "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC",
}
"""Identity only.  Bundle additive columns map ``hdehp -> HDEHP``, ``tbp -> TBP``,
``dhoa -> DHOA``, ``dohya -> DOHyA`` (DESIGN.md section 4.2 step 3)."""

MODIFIER_SMILES: dict[str, str] = {
    "1_octanol": "CCCCCCCCO", "octanol": "CCCCCCCCO", "isodecanol": "CC(C)CCCCCCCO",
    "isodecyl": "CC(C)CCCCCCCO", "isododecanol": "CC(C)CCCCCCCCCO",
}
"""Representative isomers of the alcohol additives of the bundle (identity only; the system
key uses the modifier *name*, not this SMILES)."""

SF_ND_PR_PLACEHOLDER: dict[str, dict] = {
    "sys_29976921e156a0a0": {"central": 1.4, "range": (1.3, 1.5), "status": "assumed",
                             "note": "EP2388344A1 PC-88A SF(Nd/Pr) 1.4 in kerosene (verified); "
                                     "Banda 2014 max about 1.5 (cited by the task document)"},
    "sys_f02db527a94a5e86": {"central": 1.25, "sweep": (1.1, 1.2, 1.3, 1.4), "range": (1.1, 1.4),
                             "status": "assumed",
                             "note": "no Cyanex 272 Nd/Pr value found on 2026-09-13; assumed "
                                     "sweep (LITERATURE_NOTES.md section 3)"},
}
"""Machine-readable SF(Nd/Pr) placeholder windows used by the case study's draws (DESIGN.md
section 13.2); the same numbers are in the ``log_k.Pr`` notes of the entries."""


def _src(doi: str | None, locator: str | None, kind: str | None = None) -> dict:
    return {"kind": kind or ("doi" if doi else "none"), "doi": doi, "locator": locator,
            "publication_id": None, "safe_exp_ids": []}


def _unknown(unit: str, doi: str | None = None, locator: str | None = None, note: str = "") -> dict:
    return {"value": None, "unit": unit, "status": "unknown", "assumed_label": None,
            "range": None, "source": _src(doi, locator), "model_id": None,
            "fit_manifest_sha256": None, "note": note}


def _assumed(value, unit: str, rng, doi: str | None, locator: str | None, note: str = "") -> dict:
    return {"value": value, "unit": unit, "status": "assumed", "assumed_label": ASSUMED_LABEL,
            "range": list(rng), "source": _src(doi, locator), "model_id": None,
            "fit_manifest_sha256": None, "note": note}


def placeholder(value: float | None, unit: str, range: tuple[float, float],  # noqa: A002
                doi: str | None, locator: str | None, note: str = "") -> Sourced:
    """An ``assumed`` quantity with ``ASSUMED_PLACEHOLDER``, a range and a source naming the DOI
    (``kind: "doi"``) or ``kind: "none"`` with the locator / note saying why (validator V4)."""
    return Sourced(value, unit, Provenance(
        ProvStatus.ASSUMED, Source(kind="doi" if doi else "none", doi=doi, locator=locator),
        range=(range[0], range[1]), assumed_label=ASSUMED_LABEL, note=note))


def derive_logk_from_extraction(fraction_extracted: Mapping[str, float],
                                metals_initial_mM: Mapping[str, float], ha_total_M: float,
                                oa: float, ph_eq_range: tuple[float, float], a_dimer: float,
                                b_proton: float, s_ha: float = 3.0,
                                ) -> Mapping[str, tuple[float, float]]:
    """Placeholder log K windows (dimer basis) from a reported fraction extracted.

    ``y_i = E_i x0_i / oa`` (mol/L organic, ``x0_i = mM * 1e-3``); ``[(HA)2]_f = ha_total / 2
    - s_ha * sum_i y_i``; ``log K_i = log10(E_i / (1 - E_i)) - log10(oa) - a log10[(HA)2]_f
    + b (-pH)`` evaluated at both pH bounds; returns ``{metal: (lo, hi)}``.  Concentration
    basis, ideal exponents; a placeholder, never a literature value."""
    y_sum = sum(e * metals_initial_mM[m] * 1e-3 / oa for m, e in fraction_extracted.items())
    dimer_free = ha_total_M / 2.0 - s_ha * y_sum
    if dimer_free <= 0:
        raise ValueError("free dimer concentration <= 0: loading exceeds the model cap")
    out: dict[str, tuple[float, float]] = {}
    for m, e in fraction_extracted.items():
        if not 0 < e < 1:
            raise ValueError(f"fraction extracted of {m} must be in (0, 1), got {e}")
        base = math.log10(e / (1 - e)) - math.log10(oa) - a_dimer * math.log10(dimer_free)
        vals = (base - b_proton * ph_eq_range[0], base - b_proton * ph_eq_range[1])
        out[m] = (min(vals), max(vals))
    return out


def ph50_to_logk(pH50: float, ha_total_M: float, a: float, b: float) -> float:
    """``log K = -b pH50 - a log10(ha_total / 2)``: valid only in the tracer limit (no loading,
    ``[(HA)2]_f = [HA]_T / 2``) and at O/A 1; sets ``k_reported_as = "pH50"``."""
    return -b * pH50 - a * math.log10(ha_total_M / 2.0)


# ---- the two literature entries of DESIGN.md section 3.7 ------------------------------------
_S2 = "10.1038/s41598-020-74041-9"
_BANDA = "10.1016/j.jiec.2014.03.002"
_THAKUR = "10.1016/0304-386X(93)90084-Q"


def _phase_block() -> dict:
    return {
        "loc_metal_M": _unknown("mol/L"), "loc_acid_M": _unknown("mol/L"),
        "third_phase_observed": _unknown("1"), "disengagement_s": _unknown("s"),
        "ligand_loss_mol_per_L_aq": _unknown("mol/L_aq"),
        "max_loading_fraction_studied": _unknown("1"),
        "regenerability_note": "stripped organic returns as HA; re-saponification per pass is a "
                               "design variable"}


def _strip_stream_placeholder() -> dict:
    return {
        "kind": "strip",
        "acid_M": _unknown("mol/L", _S2, "0.5 mol/L oxalic acid strips > 99.9 % Nd from loaded "
                           "D2EHPA, 2 stages, O/A 4 (D2EHPA, not PC88A)",
                           "recorded for orientation only; not used by the model"),
        "anion_M": _unknown("mol/L"), "complexant_M": _unknown("mol/L"), "metals_mM": {},
        "oa_ratio": _unknown("1"), "stages": _unknown("1"), "fraction_removed": {},
        "note": "placeholder stream record; strip liquor not transcribed"}


def _medium_chloride() -> dict:
    return {"acid": "HCl", "acid_class": "chloride", "anion": "chloride", "salting_agent": None,
            "salting_anion_M": _unknown("mol/L"), "ionic_strength_M": _unknown("mol/L"),
            "temperature_C": _assumed(25.0, "Cel", (20.0, 30.0), _S2, "298 K")}


def _cation_exchange_ligand(name: str, smiles: str, canonical: str, scaffold: str) -> dict:
    return {
        "name": name, "smiles": smiles, "canonical_smiles": canonical, "role": "extractant",
        "mechanism": "cation_exchange", "aggregation": "dimer", "scaffold_id": scaffold,
        "variant_tag": None,
        "concentration": _assumed(0.8, "mol/L", (0.2, 1.5), _S2,
                                  "extractant concentration series, 0.8 mol/L point",
                                  "formal monomer concentration used in S2; the case sweeps "
                                  "the range as a design variable"),
        "stoichiometry": {
            "ligands_per_metal": _assumed(3.0, "1", (3.0, 3.0), _S2, "mechanism nRE3+ + n(HA2)org",
                                          "dimers per Ln3+, ideal dilute-regime stoichiometry"),
            "protons_released_per_metal": _assumed(3.0, "1", (3.0, 3.0), _S2, "mechanism"),
            "anions_per_metal": _assumed(0.0, "1", (0.0, 0.0), None,
                                         "cation exchange transports no anion")}}


def _applicability_s2(ligand: str, log_acid: tuple[float, float]) -> dict:
    return {f"{ligand}|20-30C": {
        "anion": "chloride", "diluent_family": "aliphatic_hydrocarbon", "modifiers": [],
        "log_acid": list(log_acid), "log_ligand": {ligand: [-0.699, 0.0]},
        "log_metal_total_mM": [1.0, 1.7], "loading_fraction": [0.0, 0.2], "log_complexant": None,
        "oa_ratio": [1.0, 1.0], "temperature_C": [25.0, 25.0], "saponification_degree": None,
        "hull_vertices": None, "n_records": 0, "n_publications": 0,
        "_status": "assumed", "_assumed_label": ASSUMED_LABEL,
        "_range_note": "box transcribed from S2 conditions (equilibrium pH "
                       f"{-log_acid[1]:.2f}-{-log_acid[0]:.2f}, 0.2-1.0 M extractant, 10-45 mM "
                       "metal); to be replaced by the transcribed Banda/Thakur condition ranges"}}


PC88A_ENTRY_JSON: dict = {
    "schema_version": "gen18.1",
    "system_id": "sys_29976921e156a0a0",
    "name": "PC88A (HEH[EHP]) in aliphatic diluent, chloride medium",
    "family": "acidic_organophosphorus",
    "origin": "literature",
    "organic_ligands": [_cation_exchange_ligand(
        "PC88A", EXTRACTANT_SMILES["PC88A"], "CCCCC(CC)COP(=O)(O)CC(CC)CCCC",
        "phosphonic_monoester")],
    "aqueous_complexants": [],
    "diluent": {"name": "kerosene", "family": "aliphatic_hydrocarbon",
                "components": [{"name": "kerosene", "vol_fraction": 1.0}]},
    "medium": _medium_chloride(),
    "params": {"PC88A": {"20-30C": {
        "model_type": "cation_exchange", "medium_anion": "chloride", "temperature_band": "20-30C",
        "k_reported_as": "log_k",
        "log_k": {
            "Nd": _assumed(-1.95, "1", (-2.9, -1.0), _S2,
                           "Nd 64 % extracted at 0.8 M PC88A, initial pH 4.0, equilibrium pH "
                           "1.02-1.42, 1500 mg/L each Nd/Tb/Dy, A/O 1",
                           "derive_logk_from_extraction: log D_Nd = 0.250; sum [M]_org = 24.3 mM; "
                           "[(HA)2]_f = 0.400 - 3*0.0243 = 0.327 M; a = b = 3; pH_eq in "
                           "[1.02, 1.42] gives [-2.55, -1.35]; widened by 0.3 for exponent "
                           "uncertainty; with b = 2.22 (S2 slope) the window is [-1.45, -0.56]. "
                           f"Verify against Banda 2014 doi {_BANDA} and Thakur 1993 doi "
                           f"{_THAKUR}."),
            "Pr": _assumed(-2.10, "1", (-3.08, -1.11), _BANDA,
                           "maximum SF(Nd/Pr) about 1.5 (task document); patent EP2388344A1: "
                           "PC-88A SF(Nd/Pr) 1.4 in kerosene",
                           "log K_Pr = log K_Nd - log10 SF(Nd/Pr), SF placeholder range [1.3, 1.5] "
                           "(log 0.114-0.176), central 1.4 (log 0.146)")},
        "a_dimer": _assumed(3.0, "1", (2.0, 3.0), _S2, "log D vs log[extractant] slopes 2-3",
                            "ideal 3"),
        "b_proton": _assumed(3.0, "1", (2.0, 3.0), _S2, "log D vs pH slope, PC 88A Nd 2.22",
                             "ideal 3; S2 measured 2.22 for Nd"),
        "beta_anion": None,
        "saponification_degree_studied": None,
        "delta_h_kj_mol": _unknown(
            "kJ/mol", _S2, "thermodynamics section: endothermic, values not transcribed")}}},
    "phase": _phase_block(),
    "stream_records": [_strip_stream_placeholder()],
    "oxidation_state_routes": [],
    "direction_prior": None,
    "applicability": _applicability_s2("PC88A", (-1.42, -1.02)),
    "records": [],
    "notes": "Corpus has no PC88A rows. Consistency targets, never validation: Thakur 1993 (doi "
             f"{_THAKUR}) > 5 kg Nd2O3 at 97 % purity, > 85 % recovery, counter-current PC88A; "
             "EP2388344A1 PC-88A Nd/Pr circuit 72 extraction + 72 scrub + 8 strip stages at "
             "SF 1.4.",
}

_CYANEX_CANDIDATES = ("10.1016/j.hydromet.2014.09.015", "10.1016/j.jre.2017.09.016",
                      "10.1016/j.mineng.2013.10.021")
_CYANEX_NOTE = ("no Cyanex 272 Nd/Pr value found on 2026-09-13; candidate sources doi "
                + ", doi ".join(_CYANEX_CANDIDATES))

CYANEX272_ENTRY_JSON: dict = {
    "schema_version": "gen18.1",
    "system_id": "sys_f02db527a94a5e86",
    "name": "Cyanex 272 (bis(2,4,4-trimethylpentyl)phosphinic acid) in aliphatic diluent, "
            "chloride medium",
    "family": "acidic_organophosphorus",
    "origin": "literature",
    "organic_ligands": [_cation_exchange_ligand(
        "Cyanex 272", EXTRACTANT_SMILES["Cyanex 272"], "CC(CC(C)(C)C)CP(=O)(O)CC(C)CC(C)(C)C",
        "phosphinic_acid")],
    "aqueous_complexants": [],
    "diluent": {"name": "kerosene", "family": "aliphatic_hydrocarbon",
                "components": [{"name": "kerosene", "vol_fraction": 1.0}]},
    "medium": _medium_chloride(),
    "params": {"Cyanex 272": {"20-30C": {
        "model_type": "cation_exchange", "medium_anion": "chloride", "temperature_band": "20-30C",
        "k_reported_as": "log_k",
        "log_k": {
            "Nd": _assumed(-3.45, "1", (-4.5, -2.4), _S2,
                           "Nd 27 % extracted at 0.8 M Cyanex 272, initial pH 4.0, equilibrium pH "
                           "1.22-1.70, 1500 mg/L each Nd/Tb/Dy, A/O 1",
                           "derive_logk_from_extraction: log D_Nd = -0.432; sum [M]_org = 15.6 mM; "
                           "[(HA)2]_f = 0.400 - 3*0.0156 = 0.353 M; a = b = 3; pH_eq in "
                           "[1.22, 1.70] gives [-4.18, -2.74]; widened by 0.3 for exponent "
                           "uncertainty; with b = 2.0 (S2 slope) the window is [-2.48, -1.52]."),
            "Pr": {**_assumed(-3.55, "1", (-4.65, -2.36), None,
                              "SF(Nd/Pr) assumed sweep {1.1, 1.2, 1.3, 1.4}; no source found",
                              "log K_Pr = log K_Nd - log10 SF(Nd/Pr) with SF an assumed sweep "
                              "{1.1, 1.2, 1.3, 1.4} (log 0.041-0.146), central 1.25 (log 0.097); "
                              + _CYANEX_NOTE)}},
        "a_dimer": _assumed(3.0, "1", (2.0, 3.0), _S2, "log D vs log[extractant] slopes 2-3",
                            "ideal 3"),
        "b_proton": _assumed(3.0, "1", (2.0, 3.0), _S2, "log D vs pH slope, Cyanex 272 Nd 2.0",
                             "ideal 3; S2 measured 2.0 for Nd"),
        "beta_anion": None,
        "saponification_degree_studied": None,
        "delta_h_kj_mol": _unknown(
            "kJ/mol", _S2, "thermodynamics section: endothermic, values not transcribed")}}},
    "phase": _phase_block(),
    "stream_records": [_strip_stream_placeholder()],
    "oxidation_state_routes": [],
    "direction_prior": None,
    "applicability": _applicability_s2("Cyanex 272", (-1.70, -1.22)),
    "records": [],
    "notes": "Corpus has no Cyanex 272 rows. " + _CYANEX_NOTE + ". Weakest and least selective "
             "of the three acidic organophosphorus extractants for light lanthanides in chloride "
             "(search snippets only, not_found at source level; LITERATURE_NOTES.md section 3).",
}


def pc88a_prnd_entry() -> SystemEntry:
    """DESIGN.md section 3.7 example entry 1 (``sys_29976921e156a0a0``); every number
    ``ASSUMED_PLACEHOLDER``."""
    return entry_from_json(PC88A_ENTRY_JSON)


def cyanex272_prnd_entry() -> SystemEntry:
    """The Cyanex 272 entry (``sys_f02db527a94a5e86``) of the paragraph after DESIGN.md
    section 3.7; ``log_k.Pr`` from an assumed SF sweep, ``source.kind = "none"``."""
    return entry_from_json(CYANEX272_ENTRY_JSON)


def literature_entries() -> tuple[SystemEntry, ...]:
    """Every literature entry ``g18_build_db.py`` writes (ids recomputed by the validator)."""
    entries = (pc88a_prnd_entry(), cyanex272_prnd_entry())
    for e in entries:
        assert system_id(e) == e.system_id, (e.system_id, system_id(e))
    return entries


def todga_hydrophilic_complexant_placeholder() -> ComplexantSpec:
    """Hold-back complexant for the DGA + aqueous-ligand exploration (DESIGN.md section 13.5).

    ``log_beta`` values are None with the declared sweep ranges (``log beta_Nd`` in [1, 4];
    ``log beta_Pr = log beta_Nd + delta`` with ``delta`` in [0, 1.5], hence [1, 5.5]);
    protonation ``log K_H`` in [1, 3]; regeneration fraction 0.9 in [0.5, 1.0].  Open item U3:
    the candidate source is the "logK Ln-series SDFs" set; nothing here is a sourced beta."""
    none_src = "no sourced beta set for a hydrophilic DGA / sulfonated BLPhen in nitrate (U3)"
    return ComplexantSpec(
        name="hydrophilic_DGA_placeholder", smiles=None, canonical_smiles=None,
        concentration=placeholder(0.05, "mol/L", (0.001, 0.2), None, none_src,
                                  "design variable of the case (scrub_complexant_M)"),
        log_beta={"Nd": (placeholder(None, "1", (1.0, 4.0), None, none_src,
                                     "log beta_1 sweep; value None until sourced"),),
                  "Pr": (placeholder(None, "1", (1.0, 5.5), None, none_src,
                                     "log beta_Nd + delta, delta in [0, 1.5]"),)},
        protonation_logk=(placeholder(None, "1", (1.0, 3.0), None, none_src,
                                      "cumulative log K_H,1 sweep"),),
        regeneration_fraction=placeholder(0.9, "1", (0.5, 1.0), None, none_src,
                                          "recovered fraction per pass"))


# ---- phase-behaviour literature merged into corpus entries at build time --------------------
PHASE_LITERATURE: dict[str, dict[str, Sourced]] = {
    "sys_5cb78e5000d40860": {
        "loc_metal_M": Sourced(0.008, "mol/L", Provenance(
            ProvStatus.LITERATURE,
            Source(kind="doi", doi="10.1081/SEI-120016073",
                   locator="abstract (OpenAlex): 0.1 M TODGA-n-dodecane, 0.008 M Nd(III), "
                           "aqueous 3 M HNO3"),
            note="Tachimori, Sasaki, Suzuki, SEIE 20(6) 687-699 (2002). Applies only at 0.1 M "
                 "TODGA / n-dodecane / 3 M HNO3 / Nd; the loading capacity scales with TODGA "
                 "concentration, acidity, temperature, alkane chain length and aqueous anion "
                 "(same abstract); temperature not stated; full text not accessible "
                 "(addenda/ORCHESTRATOR_prefit_20260913.md, U6).")),
    },
}
"""Per ``system_id``: phase fields replaced in the corpus entry by ``g18_build_db.py``."""
