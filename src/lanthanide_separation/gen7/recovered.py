"""Variables that exist upstream and were thrown away on the way into the bundle.

The brief's hypothesis "World C" is that much of the apparently irreducible error
is missing-variable error.  This module tests it by going back to
``lanthanide_dataset_builder/raw_data/*_SAFE.csv`` — which joins 5,992/5,992 to
the bundle on ``safe_exp_id`` — and recovering everything recoverable.

What the audit actually found, because the answer matters more than the wish:

* ``Holdback_Agent_Name`` / ``Holdback_Agent_Concentration_M`` — **0 % populated**.
  The gen6 warning that some ``extractant_name`` values encode holdback agents may
  still be true, but the holdback column cannot confirm or supply it: it is empty
  in every one of the 48,138 upstream rows.
* ``Radiolytic_Dosage_kGy``, ``f_Metal_Concentration_mM``, ``thirdType``,
  ``thirdValue`` — also **0 % populated**.
* ``ini_comp`` — 100 % populated and 1,832 unique, but it is a *concatenation of
  columns already present* (``acid, extractant, metal, solvent``).  900 rows carry
  a fifth token, and in every case it is the second half of a solvent mixture that
  ``Solvent_Name`` already names.  It carries no new variable.
* ``Metal_Oxidation_state`` — two values, ``"III "`` and ``" "``.  Constant.
* ``volValue`` — 56.5 % populated, single value ``1.0``.  Constant.

What *is* genuinely recoverable and genuinely absent downstream:

``SOLVENT`` (the substantial one)
    The bundle one-hots the diluent into 34 levels and dumps 43 of the 83 upstream
    solvents into ``cond__diluent__other`` (305 rows).  Worse, a one-hot cannot
    express that 1-octanol and 1-decanol are nearly the same liquid, so 34
    independent columns each learn their own offset from their own rows.  This
    module parses ``Solvent_Name`` into base components with volume fractions and
    emits volume-weighted physical descriptors — dielectric constant, dipole
    moment, logP, molar volume, Hansen parameters, aromatic/halogen/hydroxyl
    content.  Those are properties of a liquid, known before any measurement, so
    they are safe at inference time and they interpolate to a diluent never seen.

``rec__phase_modifier_concentration_M``
    The bundle keeps ``cond__additive__<name>`` but drops the concentration.  265
    rows have one.

``rec__shaking_time_min``
    17.8 % populated upstream, absent downstream (only ``contact_time`` survived).

``rec__acid_concentration_organic_M``
    0.6 % populated.  Kept for completeness; expected to do nothing.

``publication`` / ``batch``
    ``DOI`` gives 115 publications over the raw bundle (111 inside the evaluation
    cohort).  ``comments_description`` ("Data Location: Table 3") splits a publication
    into the individual table or figure the numbers were read from — 694 distinct
    (DOI, location) batches in the bundle, 541 in the cohort, which is the finest
    experimental-session proxy the record supports.  **Neither is a deployment
    feature**: a new ligand has no DOI.  They exist here as nuisance variables for
    the variance decomposition and the deconfounding experiments, and
    :func:`recovered_feature_columns` never returns them.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_RAW = Path(os.environ.get(
    "LANTHANIDE_UPSTREAM_RAW",
    Path.home() / "PycharmProjects" / "lanthanide_dataset_builder" / "raw_data"))

#: Upstream columns confirmed empty over all 48,138 rows — recorded so a future
#: reader does not spend the afternoon rediscovering it.
EMPTY_UPSTREAM_COLUMNS: tuple[str, ...] = (
    "Holdback_Agent_Name", "Holdback_Agent_Concentration_M", "Radiolytic_Dosage_kGy",
    "f_Metal_Concentration_mM", "thirdType", "thirdValue",
)

# --------------------------------------------------------------------------- #
# Solvent physics
# --------------------------------------------------------------------------- #

#: Base solvent components with the properties that govern a distribution ratio.
#: ``eps`` static dielectric constant (25 °C), ``mu`` gas-phase dipole moment (D),
#: ``logp`` octanol/water partition coefficient of the liquid itself, ``vm`` molar
#: volume (cm³/mol), ``dD``/``dP``/``dH`` Hansen dispersion / polar / H-bonding
#: parameters (MPa^0.5).  Aliphatic diluents (kerosene, TPH, Isopar L, hydrogenated
#: tetrapropene) are technical mixtures and take dodecane-like values; that
#: approximation is exactly what a chemist means by "an aliphatic diluent" and is
#: recorded as such in ``rec__solvent_is_technical``.
SOLVENT_COMPONENTS: Mapping[str, dict] = {
    "dodecane":        dict(eps=2.01, mu=0.0,  logp=6.10, vm=228.6, dD=16.0, dP=0.0, dH=0.0, arom=0, hal=0, oh=0, technical=0),
    "kerosene":        dict(eps=2.00, mu=0.0,  logp=6.50, vm=230.0, dD=16.0, dP=0.0, dH=0.0, arom=0, hal=0, oh=0, technical=1),
    "tph":             dict(eps=2.00, mu=0.0,  logp=6.50, vm=230.0, dD=16.0, dP=0.0, dH=0.0, arom=0, hal=0, oh=0, technical=1),
    "isopar_l":        dict(eps=2.00, mu=0.0,  logp=6.30, vm=220.0, dD=15.8, dP=0.0, dH=0.0, arom=0, hal=0, oh=0, technical=1),
    "htp":             dict(eps=2.00, mu=0.0,  logp=6.40, vm=225.0, dD=15.9, dP=0.0, dH=0.0, arom=0, hal=0, oh=0, technical=1),
    "octane":          dict(eps=1.95, mu=0.0,  logp=4.50, vm=163.5, dD=15.5, dP=0.0, dH=0.0, arom=0, hal=0, oh=0, technical=0),
    "hexane":          dict(eps=1.88, mu=0.0,  logp=3.90, vm=131.6, dD=14.9, dP=0.0, dH=0.0, arom=0, hal=0, oh=0, technical=0),
    "octanol":         dict(eps=10.30, mu=1.68, logp=3.00, vm=158.4, dD=17.0, dP=3.3, dH=11.9, arom=0, hal=0, oh=1, technical=0),
    "decanol":         dict(eps=8.10, mu=1.68, logp=4.00, vm=191.6, dD=17.0, dP=2.6, dH=10.0, arom=0, hal=0, oh=1, technical=0),
    "dodecanol":       dict(eps=5.80, mu=1.68, logp=5.00, vm=224.0, dD=16.0, dP=2.6, dH=9.3,  arom=0, hal=0, oh=1, technical=0),
    "exxal13":         dict(eps=4.50, mu=1.65, logp=5.50, vm=245.0, dD=16.4, dP=2.8, dH=9.0,  arom=0, hal=0, oh=1, technical=1),
    "isodecanol":      dict(eps=7.30, mu=1.68, logp=4.00, vm=195.0, dD=16.7, dP=2.7, dH=10.2, arom=0, hal=0, oh=1, technical=1),
    "hexanol":         dict(eps=13.03, mu=1.65, logp=2.03, vm=125.2, dD=15.9, dP=5.8, dH=12.5, arom=0, hal=0, oh=1, technical=0),
    "heptanol":        dict(eps=11.75, mu=1.65, logp=2.62, vm=141.9, dD=16.0, dP=5.3, dH=11.7, arom=0, hal=0, oh=1, technical=0),
    "nonanol":         dict(eps=8.83, mu=1.65, logp=3.77, vm=175.0, dD=16.0, dP=4.5, dH=11.0, arom=0, hal=0, oh=1, technical=0),
    "toluene":         dict(eps=2.38, mu=0.36, logp=2.73, vm=106.9, dD=18.0, dP=1.4, dH=2.0,  arom=1, hal=0, oh=0, technical=0),
    "benzene":         dict(eps=2.27, mu=0.0,  logp=2.13, vm=89.4,  dD=18.4, dP=0.0, dH=2.0,  arom=1, hal=0, oh=0, technical=0),
    "tbub":            dict(eps=2.36, mu=0.35, logp=4.11, vm=155.5, dD=17.4, dP=1.1, dH=1.2,  arom=1, hal=0, oh=0, technical=0),
    "dipb":            dict(eps=2.24, mu=0.0,  logp=5.05, vm=185.0, dD=17.3, dP=0.5, dH=1.0,  arom=1, hal=0, oh=0, technical=0),
    "anisole":         dict(eps=4.30, mu=1.38, logp=2.11, vm=109.2, dD=17.8, dP=4.4, dH=6.9,  arom=1, hal=0, oh=0, technical=0),
    "nitrobenzene":    dict(eps=34.8, mu=4.22, logp=1.85, vm=102.7, dD=20.0, dP=8.6, dH=4.1,  arom=1, hal=0, oh=0, technical=0),
    "mnbtf":           dict(eps=21.0, mu=3.80, logp=3.10, vm=145.0, dD=18.5, dP=8.0, dH=3.0,  arom=1, hal=1, oh=0, technical=0),
    "nphe":            dict(eps=14.0, mu=3.60, logp=4.60, vm=205.0, dD=18.0, dP=6.5, dH=4.0,  arom=1, hal=0, oh=0, technical=0),
    "benzaldehyde":    dict(eps=17.8, mu=3.00, logp=1.48, vm=101.5, dD=19.4, dP=7.4, dH=5.3,  arom=1, hal=0, oh=0, technical=0),
    "ptms":            dict(eps=44.0, mu=4.50, logp=1.50, vm=145.0, dD=19.0, dP=15.0, dH=4.0, arom=1, hal=1, oh=0, technical=0),
    "chloroform":      dict(eps=4.81, mu=1.04, logp=1.97, vm=80.7,  dD=17.8, dP=3.1, dH=5.7,  arom=0, hal=1, oh=0, technical=0),
    "dcm":             dict(eps=8.93, mu=1.60, logp=1.25, vm=64.5,  dD=18.2, dP=6.3, dH=6.1,  arom=0, hal=1, oh=0, technical=0),
    "dce":             dict(eps=10.36, mu=1.80, logp=1.48, vm=79.4, dD=19.0, dP=7.4, dH=4.1,  arom=0, hal=1, oh=0, technical=0),
    "ccl4":            dict(eps=2.24, mu=0.0,  logp=2.83, vm=97.1,  dD=17.8, dP=0.0, dH=0.6,  arom=0, hal=1, oh=0, technical=0),
    "tetrachloroethane": dict(eps=8.20, mu=1.32, logp=2.39, vm=105.2, dD=18.8, dP=5.1, dH=5.3, arom=0, hal=1, oh=0, technical=0),
    "tetrachloroethylene": dict(eps=2.30, mu=0.0, logp=3.40, vm=102.7, dD=19.0, dP=6.5, dH=2.9, arom=0, hal=1, oh=0, technical=0),
    "trichloroethylene": dict(eps=3.42, mu=0.80, logp=2.42, vm=90.1, dD=18.0, dP=3.1, dH=5.3, arom=0, hal=1, oh=0, technical=0),
    "cyclohexanone":   dict(eps=15.6, mu=3.08, logp=0.81, vm=104.2, dD=17.8, dP=6.3, dH=5.1,  arom=0, hal=0, oh=0, technical=0),
    "dmso":            dict(eps=46.7, mu=3.96, logp=-1.35, vm=71.3, dD=18.4, dP=16.4, dH=10.2, arom=0, hal=0, oh=0, technical=0),
    "diethylether":    dict(eps=4.27, mu=1.15, logp=0.89, vm=104.7, dD=14.5, dP=2.9, dH=5.1,  arom=0, hal=0, oh=0, technical=0),
    "ionic_liquid":    dict(eps=12.0, mu=0.0,  logp=0.50, vm=290.0, dD=18.0, dP=12.0, dH=6.0, arom=1, hal=1, oh=0, technical=1),
    "tbp":             dict(eps=8.34, mu=3.07, logp=4.00, vm=273.8, dD=16.3, dP=6.3, dH=4.3,  arom=0, hal=0, oh=0, technical=0),
}

#: Longest-match-first patterns mapping a raw name fragment to a component.
_COMPONENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"1,1,2,2-tetrachloroethane", "tetrachloroethane"),
    (r"tetrachloroethylene", "tetrachloroethylene"),
    (r"trichloroethylene", "trichloroethylene"),
    (r"tetrachloroethane", "tetrachloroethane"),
    (r"tetrachloromethane", "ccl4"),
    (r"1,2-dichloroethane", "dce"),
    (r"dichloromethane", "dcm"),
    # "CH3Cl" is chloromethane by formula, but in this corpus it is chloroform: the
    # 140-row DMDPhPDA duplicate is one paper entered twice by two curators, one
    # writing "CH3Cl" and the other "Chloroform" for the same experiments.  Mapping
    # it to dichloromethane would split one liquid into two.
    (r"ch3cl", "chloroform"),
    (r"chcl3", "chloroform"),
    (r"chloroform", "chloroform"),
    (r"ccl4", "ccl4"),
    (r"meta-nitrobenzotrifluoride", "mnbtf"),
    (r"phenyl trifluoromethyl sulfone", "ptms"),
    (r"nitrobenzene", "nitrobenzene"),
    (r"benzaldehyde", "benzaldehyde"),
    (r"2-nitrophenyl hexyl ether", "nphe"),
    (r"2-nphe", "nphe"),
    (r"nphe", "nphe"),
    (r"anisole", "anisole"),
    (r"1,4-diisopropylbenzene", "dipb"),
    (r"4-diisopropylbenzene", "dipb"),
    (r"diisopropylbenzene", "dipb"),
    (r"dipb", "dipb"),
    (r"tert-butylbenzene", "tbub"),
    (r"tbub", "tbub"),
    (r"toluene", "toluene"),
    (r"benzene", "benzene"),
    (r"cyclohexanone", "cyclohexanone"),
    (r"dmso", "dmso"),
    (r"diethylether", "diethylether"),
    (r"\[c4mim\]\[tf2n\]", "ionic_liquid"),
    (r"iso-decanol", "isodecanol"),
    (r"isodecanol", "isodecanol"),
    (r"isodecyl", "isodecanol"),
    (r"isododecanol", "dodecanol"),
    (r"exxal 13", "exxal13"),
    (r"exxal13", "exxal13"),
    (r"1-dodecanol", "dodecanol"),
    (r"1-decanol", "decanol"),
    (r"1-nonanol", "nonanol"),
    (r"1-heptanol", "heptanol"),
    (r"1-hexanol", "hexanol"),
    (r"n-octanol", "octanol"),
    (r"1-octanol", "octanol"),
    (r"octanol", "octanol"),
    (r"hydrogenated tetrapropene", "htp"),
    (r"hydrogenated tetrapropylene", "htp"),
    (r"isopar l", "isopar_l"),
    (r"isoparl", "isopar_l"),
    (r"sulfonated kerosene", "kerosene"),
    (r"avkerosene", "kerosene"),
    (r"tphkerosene", "kerosene"),
    (r"kerosene \(solvent 70\)", "kerosene"),
    (r"kerosene", "kerosene"),
    (r"total petroleum hydrocarbons", "tph"),
    (r"tph", "tph"),
    (r"n-dodecane", "dodecane"),
    (r"dodecane", "dodecane"),
    (r"n-octane", "octane"),
    (r"octane", "octane"),
    (r"n-hexane", "hexane"),
    (r"hexane", "hexane"),
    (r"tbp", "tbp"),
)


def _match_component(fragment: str) -> str | None:
    text = fragment.strip().lower()
    if not text:
        return None
    for pattern, component in _COMPONENT_PATTERNS:
        if re.search(pattern, text):
            return component
    return None


def parse_solvent(name: str) -> dict[str, float]:
    """``Solvent_Name`` -> component volume fractions summing to 1.

    Handles the three notations the upstream table actually uses::

        "n-Dodecane"                            -> {dodecane: 1.0}
        "kerosene with 30 vol% 1-octanol"       -> {kerosene: 0.7, octanol: 0.3}
        "kerosene 0.7, 1-octanol 0.3"           -> {kerosene: 0.7, octanol: 0.3}

    An unparseable fragment is dropped and the remaining fractions renormalised;
    a wholly unparseable name returns ``{}``, which downstream becomes NaN rather
    than a silent zero.
    """
    if not isinstance(name, str) or not name.strip():
        return {}
    text = name.strip()
    fractions: dict[str, float] = {}

    # "A with N vol% B"  /  "A with N% B"
    with_match = re.search(r"^(.*?)\s+with\s+([\d.]+)\s*(?:vol)?%\s*(.*)$", text, flags=re.I)
    if with_match:
        host, percent, guest = with_match.groups()
        try:
            share = float(percent) / 100.0
        except ValueError:
            share = np.nan
        if np.isfinite(share):
            host_key, guest_key = _match_component(host), _match_component(guest)
            if host_key:
                fractions[host_key] = fractions.get(host_key, 0.0) + (1.0 - share)
            if guest_key:
                fractions[guest_key] = fractions.get(guest_key, 0.0) + share
            return _normalise(fractions)

    # "A 0.7, B 0.3"  — comma-separated fragments each ending in a fraction
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        parsed: list[tuple[str, float]] = []
        ok = True
        for part in parts:
            fraction_match = re.search(r"^(.*?)[\s]+([01]?\.\d+|1(?:\.0+)?)$", part)
            if not fraction_match:
                ok = False
                break
            body, value = fraction_match.groups()
            key = _match_component(body)
            if key is None:
                ok = False
                break
            parsed.append((key, float(value)))
        if ok and parsed:
            for key, value in parsed:
                fractions[key] = fractions.get(key, 0.0) + value
            return _normalise(fractions)

    key = _match_component(text)
    return {key: 1.0} if key else {}


def _normalise(fractions: dict[str, float]) -> dict[str, float]:
    total = sum(fractions.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in fractions.items()}


SOLVENT_PROPERTIES: tuple[str, ...] = ("eps", "mu", "logp", "vm", "dD", "dP", "dH",
                                       "arom", "hal", "oh", "technical")


def solvent_descriptors(name: str) -> dict[str, float]:
    """Volume-weighted physical descriptors of a (possibly mixed) diluent.

    Also emits ``rec__solvent_log_eps`` — a distribution ratio responds to
    polarity roughly logarithmically, and the dielectric constants here span
    1.88 to 46.7, so the tree should be handed the logarithm as well as the value
    (the same argument the MASSACTION block makes about concentrations).
    """
    fractions = parse_solvent(name)
    out = {f"rec__solvent_{p}": np.nan for p in SOLVENT_PROPERTIES}
    out["rec__solvent_n_components"] = np.nan
    out["rec__solvent_log_eps"] = np.nan
    out["rec__solvent_polar_fraction"] = np.nan
    out["rec__solvent_parsed"] = 0.0
    if not fractions:
        return out
    known = {k: v for k, v in fractions.items() if k in SOLVENT_COMPONENTS}
    if not known:
        return out
    weight = sum(known.values())
    for prop in SOLVENT_PROPERTIES:
        out[f"rec__solvent_{prop}"] = float(
            sum(SOLVENT_COMPONENTS[k][prop] * v for k, v in known.items()) / weight)
    out["rec__solvent_n_components"] = float(len(known))
    eps = out["rec__solvent_eps"]
    out["rec__solvent_log_eps"] = float(np.log10(eps)) if eps and eps > 0 else np.nan
    # the modifier fraction: the alcohol/TBP share of the organic phase, which is
    # what a chemist calls "the modifier level" and what suppresses third-phase
    # formation
    out["rec__solvent_polar_fraction"] = float(
        sum(v for k, v in known.items() if SOLVENT_COMPONENTS[k]["oh"] or k == "tbp") / weight)
    out["rec__solvent_parsed"] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Upstream join
# --------------------------------------------------------------------------- #

def load_upstream(raw_dir: Path | str = UPSTREAM_RAW) -> pd.DataFrame:
    """Concatenate every ``*_SAFE.csv`` and build the ``safe_exp_id`` join key."""
    files = sorted(glob.glob(str(Path(raw_dir) / "*_SAFE.csv")))
    if not files:
        raise FileNotFoundError(f"no *_SAFE.csv under {raw_dir}")
    parts = []
    for path in files:
        stem = Path(path).stem
        table = pd.read_csv(path, low_memory=False)
        table["safe_exp_id"] = stem + ":" + table["exp_id"].astype(str)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def _numeric_with_unit(series: pd.Series) -> pd.Series:
    """``"0.5 M"`` -> 0.5;  ``" "`` -> NaN.  Never a silent zero."""
    text = series.astype(str).str.strip()
    numbers = text.str.extract(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
    return pd.to_numeric(numbers, errors="coerce")


def build_recovered_table(source: pd.DataFrame, *, raw_dir: Path | str = UPSTREAM_RAW) -> pd.DataFrame:
    """One row per bundle row: recovered features plus the nuisance keys.

    ``source`` is the raw bundle (5,992 rows) — the join happens *before* the
    level cohort is built and averaged, so a recovered variable that differs
    within an averaged cell is detectable rather than silently collapsed.
    """
    upstream = load_upstream(raw_dir)
    keep = ["safe_exp_id", "Solvent_Name", "Phase_Modifier_Name", "Phase_Modifier_Concentration_M",
            "Shaking_Time_min", "Contact_Time_min", "Acid_Concentration_Organic_M",
            "Metal_Oxidation_state", "DOI", "entry_author", "addition_date",
            "comments_description", "volValue", "ini_comp", "Extractant_Name"]
    upstream = upstream[[c for c in keep if c in upstream.columns]].drop_duplicates("safe_exp_id")
    merged = source[["safe_exp_id"]].merge(upstream, on="safe_exp_id", how="left", validate="one_to_one")
    if merged["Solvent_Name"].isna().all():
        raise AssertionError("the upstream join produced no solvent names; check safe_exp_id")

    # --- solvent physics ---------------------------------------------------- #
    unique_names = merged["Solvent_Name"].dropna().unique()
    lookup = {name: solvent_descriptors(name) for name in unique_names}
    blank = {k: np.nan for k in next(iter(lookup.values()))} if lookup else {}
    solvent = pd.DataFrame(
        [lookup.get(n, blank) if isinstance(n, str) else blank for n in merged["Solvent_Name"]],
        index=merged.index)

    out = pd.concat([merged[["safe_exp_id"]], solvent], axis=1)

    # --- scalar recoveries -------------------------------------------------- #
    out["rec__phase_modifier_concentration_M"] = _numeric_with_unit(
        merged["Phase_Modifier_Concentration_M"])
    out["rec__has_phase_modifier"] = merged["Phase_Modifier_Name"].notna().astype(float)
    out["rec__shaking_time_min"] = _numeric_with_unit(merged["Shaking_Time_min"])
    out["rec__has_shaking_time"] = merged["Shaking_Time_min"].notna().astype(float)
    out["rec__acid_concentration_organic_M"] = _numeric_with_unit(
        merged["Acid_Concentration_Organic_M"])

    # --- the unmodelled-second-species flag --------------------------------- #
    # gen6 warned that 17 canonical SMILES carry several ``extractant_name`` values
    # and that some names look like aqueous holdback agents.  ``Holdback_Agent_Name``
    # is empty upstream, so the warning cannot be confirmed from the column it
    # belongs in — but it *can* be confirmed from the names.  One SMILES (TODGA,
    # 1,714 rows) carries 21 names, and 20 of them are other chemicals entirely:
    # SO3-Ph-BTP, (PhSO3Na)2-BTBP, the TWE- series, PHEN-6OH, PyTri-diol, CITAM.
    # Those are water-soluble complexants added to the *aqueous* phase alongside
    # TODGA; the record kept the aqueous agent's name and left the organic
    # extractant's SMILES, so the row describes a two-species experiment that the
    # model sees as a plain TODGA row.
    #
    # 414 rows across the bundle carry a name that is not the modal name of their
    # own structure.  The flag says "an additional, unrecorded species was present",
    # which is a property of the experiment the operator knows before measuring, so
    # it is safe at inference.  The modal name is derived from *names only* — never
    # from ``log_D`` — which puts it in the same target-free class as the frozen
    # chemistry map.
    names = merged["Extractant_Name"].fillna("unknown").astype(str)
    structures = source["canonical_smiles"].astype(str).to_numpy() \
        if "canonical_smiles" in source.columns else np.array(["?"] * len(merged))
    modal = pd.Series(names.to_numpy(), index=structures).groupby(level=0).agg(
        lambda s: s.value_counts().idxmax())
    expected = pd.Series(structures).map(modal)
    out["rec__name_mismatch"] = (names.to_numpy() != expected.to_numpy()).astype(float)
    out["rec__n_names_for_structure"] = pd.Series(structures).map(
        pd.Series(names.to_numpy(), index=structures).groupby(level=0).nunique()).to_numpy()
    lowered = names.str.lower()
    # water-soluble complexant signatures: sulfonate salts, the TWE/PyTri/PHEN-ol
    # families and the named aqueous masking agents seen in this corpus
    aqueous = (lowered.str.contains("so3") | lowered.str.contains("phso3na")
               | lowered.str.contains("twe-") | lowered.str.contains("pytri")
               | lowered.str.contains("phen-6oh") | lowered.str.contains("phen-dialcohol")
               | lowered.str.contains("citam") | lowered.str.contains("dtpa")
               | lowered.str.contains("edta") | lowered.str.contains("hedta"))
    out["rec__aqueous_complexant"] = (aqueous.to_numpy()
                                      & (out["rec__name_mismatch"].to_numpy() > 0)).astype(float)

    # --- nuisance keys (NEVER features) ------------------------------------- #
    out["nuisance__doi"] = merged["DOI"].fillna("unknown").astype(str)
    location = (merged["comments_description"].fillna("").astype(str)
                .str.replace(r"^Data Location:\s*", "", regex=True).str.strip())
    out["nuisance__data_location"] = location.replace("", "unknown")
    out["nuisance__batch"] = out["nuisance__doi"] + " || " + out["nuisance__data_location"]
    out["nuisance__entry_author"] = merged["entry_author"].fillna("unknown").astype(str)
    out["nuisance__addition_date"] = merged["addition_date"].astype(str)
    out["nuisance__extractant_name"] = merged["Extractant_Name"].fillna("unknown").astype(str)
    return out


def recovered_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Only ``rec__*``.  The nuisance keys are deliberately excluded."""
    return tuple(c for c in frame.columns if c.startswith("rec__"))


def coverage_report(recovered: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in recovered.columns:
        if column == "safe_exp_id":
            continue
        series = recovered[column]
        rows.append({
            "column": column,
            "kind": "feature" if column.startswith("rec__") else "nuisance",
            "coverage": float(series.notna().mean()),
            "n_unique": int(series.nunique(dropna=True)),
            "dtype": str(series.dtype),
        })
    return pd.DataFrame(rows).sort_values(["kind", "column"], ignore_index=True)


# --------------------------------------------------------------------------- #
# Aggregation to the level cohort's row (the "cell")
# --------------------------------------------------------------------------- #

def recovered_cell_table(source: pd.DataFrame, *, raw_dir: Path | str = UPSTREAM_RAW,
                         drop_below_log_d: float | None = -6.0) -> pd.DataFrame:
    """Recovered variables keyed on the level cohort's ``row_id``.

    A level row is a *cell* — one (extractant, condition, metal) — and the cohort
    builder averages the replicates inside it.  The recovered variables must be
    aggregated the same way or the join would not be one-to-one.  Numeric features
    take the cell mean; the nuisance keys take the modal value and carry a
    ``*_n_distinct`` count, because a cell whose rows come from two publications is
    exactly the case the batch analysis needs to see rather than have smoothed away.

    The ``row_id`` is recomputed with the cohort builder's own hash, so the join
    key cannot drift from :func:`levels.build_level_dataset`.
    """
    import hashlib

    from ..levels import LEVEL_TARGET_COLUMN, condition_labels

    frame = source.copy()
    if drop_below_log_d is not None and LEVEL_TARGET_COLUMN in frame.columns:
        frame = frame[frame[LEVEL_TARGET_COLUMN] > drop_below_log_d]
    condition_columns = [c for c in frame.columns if c.startswith("cond__")]
    frame = frame.assign(
        extractant=frame["canonical_smiles"].astype(str),
        metal_symbol=frame["metal_symbol"].astype(str),
        condition_id=condition_labels(frame, condition_columns))
    frame = frame.assign(row_id=[
        hashlib.sha1(f"{e}|{c}|{m}".encode()).hexdigest()[:16]
        for e, c, m in zip(frame["extractant"], frame["condition_id"], frame["metal_symbol"])])

    recovered = build_recovered_table(frame, raw_dir=raw_dir)
    joined = pd.concat([frame[["row_id"]].reset_index(drop=True),
                        recovered.drop(columns=["safe_exp_id"]).reset_index(drop=True)], axis=1)

    features = [c for c in joined.columns if c.startswith("rec__")]
    nuisance = [c for c in joined.columns if c.startswith("nuisance__")]
    grouped = joined.groupby("row_id", sort=False)
    out = grouped[features].mean()
    for column in nuisance:
        modes = grouped[column].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "unknown")
        out[column] = modes
        out[f"{column}__n_distinct"] = grouped[column].nunique()
    return out.reset_index()
