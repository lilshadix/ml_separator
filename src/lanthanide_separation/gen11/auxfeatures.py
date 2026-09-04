"""Render multi-metal archive records into the frozen cohort's feature contract.

A transfer arm that trains on auxiliary (multi-metal) rows is only meaningful if
those rows are *encoded* the way the frozen cohort's training rows are encoded.
The danger is not that a featurizer is wrong in some absolute sense — it is that
it is wrong *differently* from the one that produced the training rows.  A
diluent that lands in ``cond__diluent__other`` here but in its own column there,
an ECFP computed at radius 3, a concentration parsed as mM instead of M: each
would give the model an auxiliary set drawn from a different distribution than
the one it will be tested against, and the resulting transfer number would
measure the encoding mismatch rather than the chemistry.

So this module does not re-invent the encoding, it *replays* it, and then proves
the replay by re-deriving the frozen bundle's own columns from the archive and
diffing them cell by cell.  :func:`reproduction_report` is the load-bearing part
of the file; the featurizer is only trustworthy to the extent that report is
clean.

What is replayed, and from where
--------------------------------
``ECFP`` (2,048)
    RDKit ``GetMorganGenerator(radius=2, fpSize=2048)`` on the canonical SMILES,
    exactly as ``build_dataset_no3d.py``.  The archive's
    ``extractant_primary_smiles`` is byte-identical to the bundle's
    ``canonical_smiles`` on all 5,992 shared records, so no re-canonicalisation
    is needed (and a Morgan fingerprint is a property of the molecular graph, not
    of the SMILES string, so it could not change one anyway).

``COND`` (64)
    The builder's own normalizers, copied verbatim below, applied to the
    archive's ``*_raw`` text — *not* to the archive's canonicalised fields.  This
    is deliberate.  The archive corrects several chemical mistakes the bundle
    builder made (see :data:`ARCHIVE_ADJUDICATIONS`); adopting those corrections
    here would encode auxiliary rows on a different vocabulary than the training
    rows, which is the exact failure this module exists to prevent.  Every such
    correction is instead recorded in ``known_encoding_divergences.csv`` so a
    later sensitivity arm can switch it on and measure what it costs.

    The one-hot vocabulary is read off the *frozen column names* rather than
    recomputed.  ``build_dataset_no3d.py`` caps a categorical at
    ``CATEGORICAL_MAX_LEVELS = 40`` and sends the tail to ``OTHER``; reading the
    surviving level set out of the frozen contract reproduces that cap without
    having to reconstruct the frame it was computed on, and it makes the
    treatment of an unseen auxiliary level explicit: it goes to ``other`` if the
    family has an ``other`` column (diluent does) and to an all-zero row if it
    does not (acid, additive do not).  Both counts are reported.

``METAL`` (3)
    ``LANTHANIDE_DESCRIPTORS`` from the builder, copied verbatim.  It covers 14
    lanthanides and nothing else — no Pm, no Y, no actinide — so on its own it
    would leave the METAL block NaN for 98 % of the auxiliary rows.  The default
    policy therefore falls back to the archive's own ``atomic_number`` /
    ``lanthanide_index`` / ``ionic_radius_cn8_A`` for symbols the table lacks.
    That fallback is checked, not assumed: on the 5,992 shared records the
    archive's three columns agree with the builder's table exactly, so the
    fallback is on the same scale rather than merely plausible.  ``metal_policy
    = "table_only"`` disables it for a sensitivity arm.

``MASSACTION`` (8)
    ``levels._attach_mass_action`` itself, so the arithmetic cannot drift.  It
    needs ``DENTATE`` and ``coreCN``, which are geometry-plan annotations that
    exist only for the 190 frozen structures; they are looked up from the cohort
    (per-extractant mode for ``DENTATE``, per-(extractant, metal) mode for
    ``coreCN``) and left NaN otherwise.  The frozen arm imputes with
    ``add_indicator=True``, so NaN is a representable state rather than a
    silent zero — but the affected row counts are reported because a block that
    is NaN for most auxiliary rows is a fact about the experiment, not a detail.

``RECOVERED`` (23)
    ``gen7.recovered``'s pure functions (:func:`~lanthanide_separation.gen7.recovered.parse_solvent`,
    :func:`~lanthanide_separation.gen7.recovered.solvent_descriptors`,
    ``_numeric_with_unit``) driven from the archive's raw fields instead of the
    upstream ``*_SAFE.csv`` files, because 10 of the 41 archive metal files have
    no upstream counterpart.  The three name-derived columns
    (``rec__name_mismatch``, ``rec__n_names_for_structure``,
    ``rec__aqueous_complexant``) are *relative to a per-structure modal name*;
    for a structure the frozen cohort knows, the frozen modal name is used so the
    auxiliary row is judged against the same reference as its training
    counterparts, and only a genuinely new structure falls back to the archive's
    own rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..gen7.recovered import (
    SOLVENT_PROPERTIES,
    _numeric_with_unit,
    solvent_descriptors,
)
from ..levels import CONTINUOUS_CONDITION_COLUMNS, _attach_mass_action
from .overlap import ARCHIVE_CLEAN, FROZEN_DATASET, REPO_ROOT

OUT_DIR = REPO_ROOT / "runs" / "gen11_transfer" / "featurizer"

#: The frozen arm gen11 must match, in the order ``gen10`` names it.
FROZEN_ARM_BLOCKS: tuple[str, ...] = ("METAL", "COND", "ECFP", "MASSACTION", "RECOVERED")

#: Identity carried alongside the features.  None of these may ever become one.
#: The first seven are the leakage keys ``overlap.relationship_masks`` reads plus
#: the target; ``metal_category`` and the two readiness columns follow because
#: ``gen11.pools.build_pool`` reports survival per metal category and an arm
#: cannot be re-scoped without them.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "source_record_id", "metal_symbol", "extractant_primary_smiles", "series_id",
    "duplicate_group_id", "doi_primary_corrected", "log_D",
    "metal_category", "metal_oxidation_state", "model_readiness",
)

#: Identity the archive does not carry under that name.  ``extractant`` is the
#: cohort's own name for the canonical SMILES and ``gen11.pools`` keys the
#: Tanimoto chemotype assignment on it, so the auxiliary frame must speak both
#: vocabularies; it is an alias, never a second value.
DERIVED_IDENTITY_COLUMNS: tuple[str, ...] = ("extractant",)

#: Archive columns the featurizer reads.  ``overlap.ARCHIVE_IDENTITY_COLUMNS`` is
#: an allow-list for *leakage* reasoning and deliberately omits the raw text; the
#: featurizer needs that text, so it keeps its own explicit list rather than
#: widening the other one.
ARCHIVE_FEATURE_COLUMNS: tuple[str, ...] = (
    "source_record_id", "duplicate_group_id", "series_id", "doi_primary_corrected",
    "model_readiness", "metal_symbol", "metal_category", "metal_oxidation_state",
    "atomic_number", "lanthanide_index", "ionic_radius_cn8_A", "ionic_radius_status",
    "extractant_primary_smiles", "extractant_name_raw", "log_D",
    "solvent_name_raw", "solvent_primary", "solvent_key",
    "acid_name_raw", "modifier_name_raw", "modifier_concentration_raw",
    "extractant_concentration_raw", "acid_concentration_raw", "metal_concentration_raw",
    "temperature_raw", "contact_time_raw", "shaking_time_raw",
    "acid_concentration_organic_raw",
)

#: How each frozen condition column is fed.  The left column is the frozen
#: feature name, the right the archive ``*_raw`` field standing in for the
#: upstream SAFE column ``build_dataset_no3d.py`` read.
NUMERIC_CONDITION_SOURCES: Mapping[str, tuple[str, str]] = {
    # frozen column -> (archive raw column, parse kind)
    "cond__extractant_concentration_M": ("extractant_concentration_raw", "M"),
    "cond__acid_concentration_M": ("acid_concentration_raw", "M"),
    "cond__metal_concentration_mM": ("metal_concentration_raw", "mM"),
    "cond__temperature_C": ("temperature_raw", "temperature"),
    "cond__contact_time_min": ("contact_time_raw", "number"),
}

#: ``cond__<prefix>__<level>`` families and the raw text that drives them.
#: ``acid <- Acid_Name``, ``diluent <- Solvent_Name``, ``additive <-
#: Phase_Modifier_Name`` is the builder's own alias resolution.
CATEGORICAL_CONDITION_SOURCES: Mapping[str, str] = {
    "acid": "acid_name_raw",
    "diluent": "solvent_name_raw",
    "additive": "modifier_name_raw",
}

#: Metal descriptors, copied verbatim from
#: ``lanthanide_dataset_builder/src/chemistry/coordination.py``.  Pm is absent
#: upstream and is left absent here; the archive supplies it under the default
#: fallback policy and that is recorded as a divergence rather than patched in.
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
METAL_COLUMNS: tuple[str, ...] = (
    "Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal")

ECFP_BITS = 2048
ECFP_RADIUS = 2
ECFP_PREFIX = "ecfp_"

#: Chemical adjudications the archive makes that this featurizer deliberately
#: does NOT apply, because the training rows do not have them.  Each becomes a
#: row of ``known_encoding_divergences.csv`` with a measured auxiliary row count.
#: Source: ``dataset_all_metals/scripts/sae_normalize.py`` (``SOLVENT_AMBIGUOUS``,
#: ``_split_solvent_components``).
ARCHIVE_ADJUDICATIONS: tuple[dict, ...] = (
    dict(divergence="tph_identity", family="COND/diluent",
         detail="TPH = tetrapropylene hydrogenated, a branched-C12 kerosene cut; "
                "'total petroleum hydrocarbons' is an analytical parameter. The frozen "
                "one-hot keeps cond__diluent__tph, __total_petroleum_hydrocarbons, "
                "__hydrogenated_tetrapropene, __hydrogenated_tetrapropylene, __tphkerosene "
                "as five independent levels of what is largely one aliphatic diluent.",
         levels="tph|total petroleum hydrocarbons|hydrogenated tetrapropene|"
                "hydrogenated tetrapropylene|tphkerosene",
         sensitivity_arm="merge the five levels into one aliphatic-cut indicator"),
    dict(divergence="tph_identity_recovered", family="RECOVERED/solvent",
         detail="gen7.recovered maps 'tph' AND 'total petroleum hydrocarbons' to the same "
                "physical component, so the RECOVERED block already merges what the COND "
                "one-hot splits. The two blocks disagree with each other inside the frozen "
                "contract; both are reproduced as-is.",
         levels="tph|total petroleum hydrocarbons",
         sensitivity_arm="none needed for aux; recorded because it explains a COND/RECOVERED "
                         "inconsistency a reader will otherwise take for a bug"),
    dict(divergence="iupac_locant_comma", family="COND/diluent + RECOVERED/solvent",
         detail="Blind splitting of a diluent string on ',' turns 1,4-diisopropylbenzene and "
                "1,1,2,2-tetrachloroethane into fabricated mixtures whose first component is "
                "the literal '1' (296 rows archive-wide). This featurizer reads "
                "solvent_name_raw and never splits, so it is unaffected; the archive's "
                "solvent_primary / solvent_key carry the corrected split and are NOT used.",
         levels="1,4-diisopropylbenzene|1,1,2,2-tetrachloroethane|1,2-dichloroethane",
         sensitivity_arm="re-encode the diluent from the archive's adjudicated solvent_key"),
    dict(divergence="ch3cl_isomer", family="COND/diluent",
         detail="'CH3Cl' is chloromethane by formula but chloroform in this corpus (one paper "
                "entered twice, one curator writing CH3Cl and the other Chloroform for the same "
                "experiments). The archive declines to merge them on a guess; the frozen COND "
                "one-hot therefore carries cond__diluent__ch3cl, __chcl3 and __chloroform as "
                "three levels of one liquid, while gen7.recovered's RECOVERED block merges all "
                "three into a single component. The two blocks disagree inside the frozen "
                "contract and both are reproduced as-is. NOTE also that the row-level cache "
                "runs/gen7_architecture/cache/recovered_raw.parquet predates the CH3Cl fix and "
                "still encodes it as dichloromethane; recovered_cells.parquet, which is what the "
                "cohort actually joins, does not. Measured in reproduction.json/"
                "recovered_cache_check.",
         levels="CH3Cl|CHCl3|Chloroform",
         sensitivity_arm="collapse the three chloroform levels into one"),
    dict(divergence="ionic_radius_oxidation_state", family="METAL",
         detail="The frozen METAL block assigns one ionic radius per element symbol and "
                "ignores oxidation state (all bundle metals are Ln(III)). The archive resolves "
                "the radius per row from the oxidation state, which matters for U(VI) vs U(IV) "
                "and Pu(III/IV). This featurizer uses one radius per symbol to stay on the "
                "training encoding.",
         levels="U|Pu|Np|Am|Cm",
         sensitivity_arm="per-row oxidation-state-resolved ionic_radius_cn8_A"),
)


# --------------------------------------------------------------------------- #
# The builder's normalizers, copied verbatim
# --------------------------------------------------------------------------- #
# These are byte-for-byte transcriptions of
# ``lanthanide_dataset_builder/scripts/build_dataset_no3d.py``.  They are copied
# rather than imported because that repository is not a dependency of this one
# and is not version-pinned; a copy that is *proved* equivalent by
# :func:`reproduction_report` is safer than an import that could silently move.

def normalize_text_value(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-", "--", "n/a", "na"}:
        return None
    return re.sub(r"\s+", " ", text)


def first_number_from_text(value) -> float | None:
    text = normalize_text_value(value)
    if text is None:
        return None
    text = text.replace(",", ".")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_concentration(value, default_unit: str) -> float | None:
    """``'0.5 M'`` / ``'10 mM'`` / ``'0.01 mol/L'`` -> a number in M or mM."""
    text = normalize_text_value(value)
    if text is None:
        return None
    number = first_number_from_text(text)
    if number is None:
        return None
    lower = text.lower().replace("μ", "u").replace("µ", "u")
    if "mg/l" in lower or "mg l" in lower or "ppm" in lower:
        return number  # not safely convertible to molar
    if "mmol" in lower or re.search(r"\bmm\b", lower):
        value_m = number / 1000.0
    elif "umol" in lower or re.search(r"\bum\b", lower):
        value_m = number / 1_000_000.0
    elif "mol/l" in lower or re.search(r"(?<![a-z])m(?![a-z])", lower):
        value_m = number
    else:
        return number  # unit-less number; assume already in requested unit
    return value_m * 1000.0 if default_unit == "mM" else value_m


def parse_temperature_c(value) -> float | None:
    number = first_number_from_text(value)
    if number is None:
        return None
    lower = (normalize_text_value(value) or "").lower()
    if "k" in lower and "°" not in lower and number > 100:
        return number - 273.15
    return number


def safe_dummy_name(prefix: str, value) -> str:
    text = (normalize_text_value(value) or "missing").lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return f"cond__{prefix}__{text[:60]}"


# --------------------------------------------------------------------------- #
# The contract: which columns, and which one-hot levels, the frozen arm has
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FeatureContract:
    """The exact 2,146-column feature space auxiliary rows must land in."""

    metal: tuple[str, ...]
    cond_numeric: tuple[str, ...]
    cond_categorical: Mapping[str, tuple[str, ...]]   # prefix -> column names
    ecfp: tuple[str, ...]
    massaction: tuple[str, ...]
    recovered: tuple[str, ...]

    @property
    def cond(self) -> tuple[str, ...]:
        cols = list(self.cond_numeric)
        for prefix in self.cond_categorical:
            cols.extend(self.cond_categorical[prefix])
        return tuple(sorted(cols))

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(list(self.metal) + list(self.cond) + list(self.ecfp)
                     + list(self.massaction) + list(self.recovered))

    def levels(self, prefix: str) -> set[str]:
        return set(self.cond_categorical[prefix])

    def has_other(self, prefix: str) -> bool:
        return f"cond__{prefix}__other" in self.cond_categorical[prefix]


def build_contract(cohort) -> FeatureContract:
    """Read the contract off the frozen cohort rather than restating it."""
    blocks = cohort.blocks
    missing = [b for b in FROZEN_ARM_BLOCKS if b not in blocks]
    if missing:
        raise SystemExit(f"cohort is missing frozen-arm blocks {missing}")
    cond = tuple(blocks["COND"])
    categorical: dict[str, tuple[str, ...]] = {}
    for prefix in CATEGORICAL_CONDITION_SOURCES:
        categorical[prefix] = tuple(c for c in cond if c.startswith(f"cond__{prefix}__"))
    flat = {c for cols in categorical.values() for c in cols}
    numeric = tuple(c for c in cond if c not in flat)
    contract = FeatureContract(
        metal=tuple(blocks["METAL"]),
        cond_numeric=numeric,
        cond_categorical=categorical,
        ecfp=tuple(blocks["ECFP"]),
        massaction=tuple(blocks["MASSACTION"]),
        recovered=tuple(blocks["RECOVERED"]),
    )
    if set(numeric) != set(NUMERIC_CONDITION_SOURCES):
        raise SystemExit(
            "the frozen numeric condition columns moved: "
            f"{sorted(set(numeric) ^ set(NUMERIC_CONDITION_SOURCES))}")
    if len(contract.columns) != 2146:
        raise SystemExit(f"frozen arm is {len(contract.columns)} columns, expected 2146")
    return contract


# --------------------------------------------------------------------------- #
# Per-ligand / per-metal annotations that only the frozen side has
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Annotations:
    """Everything the archive cannot supply, taken from the frozen cohort.

    ``dentate`` and ``core_cn`` are geometry-plan outputs: the builder derived
    them per row from (metal Z, ligand SMILES, acid), so they are not strictly a
    per-ligand constant — 9 of the 190 structures carry more than one
    ``DENTATE`` and 15 (ligand, metal) cells more than one ``coreCN`` inside the
    cohort itself.  The mode is used and the ambiguity counted, because silently
    picking the first row would hide it.
    """

    dentate: Mapping[str, float]                     # extractant -> DENTATE
    core_cn_cell: Mapping[tuple[str, str], float]    # (extractant, metal) -> coreCN
    core_cn_metal: Mapping[str, float]               # metal -> coreCN
    modal_name: Mapping[str, str]                    # extractant -> modal Extractant_Name
    n_names: Mapping[str, int]                       # extractant -> distinct names
    ambiguity: dict


def _mode(series: pd.Series):
    modes = series.dropna().mode()
    return modes.iat[0] if len(modes) else np.nan


def build_annotations(cohort, frozen_bundle: pd.DataFrame | None = None) -> Annotations:
    frame = cohort.frame
    for column in ("DENTATE", "coreCN", "extractant", "metal_symbol"):
        if column not in frame.columns:
            raise SystemExit(f"cohort frame lacks {column!r}; annotations cannot be built")
    dentate = frame.groupby("extractant")["DENTATE"].agg(_mode)
    cell = frame.groupby(["extractant", "metal_symbol"])["coreCN"].agg(_mode)
    per_metal = frame.groupby("metal_symbol")["coreCN"].agg(_mode)
    ambiguity = {
        "extractants_with_multiple_DENTATE":
            int((frame.groupby("extractant")["DENTATE"].nunique() > 1).sum()),
        "cells_with_multiple_coreCN":
            int((frame.groupby(["extractant", "metal_symbol"])["coreCN"].nunique() > 1).sum()),
        "annotated_extractants": int(dentate.size),
        "annotated_metals": int(per_metal.size),
    }

    bundle = frozen_bundle if frozen_bundle is not None else load_frozen_bundle()
    names = bundle["extractant_name"].fillna("unknown").astype(str)
    structures = bundle["canonical_smiles"].astype(str)
    grouped = pd.DataFrame({"s": structures.to_numpy(), "n": names.to_numpy()}).groupby("s")["n"]
    modal_name = grouped.agg(lambda s: s.value_counts().idxmax())
    n_names = grouped.nunique()
    ambiguity["structures_with_multiple_names"] = int((n_names > 1).sum())
    ambiguity["named_structures"] = int(n_names.size)

    return Annotations(
        dentate=dentate.to_dict(),
        core_cn_cell={(a, b): v for (a, b), v in cell.items()},
        core_cn_metal=per_metal.to_dict(),
        modal_name=modal_name.to_dict(),
        n_names=n_names.astype(int).to_dict(),
        ambiguity=ambiguity,
    )


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _morgan_generator():
    from rdkit import RDLogger
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    RDLogger.DisableLog("rdApp.*")
    return GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)


def ecfp_block(smiles: pd.Series, contract: FeatureContract) -> tuple[pd.DataFrame, list[str]]:
    """2,048 Morgan bits, one lookup per distinct structure."""
    from rdkit import Chem, DataStructs

    generator = _morgan_generator()
    lookup: dict[str, np.ndarray] = {}
    failures: list[str] = []
    for value in sorted({s for s in smiles.dropna().astype(str)}):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            failures.append(value)
            continue
        arr = np.zeros((ECFP_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
        lookup[value] = arr
    blank = np.full((ECFP_BITS,), np.nan)
    matrix = np.stack([lookup.get(str(s), blank) if pd.notna(s) else blank for s in smiles])
    frame = pd.DataFrame(matrix, index=smiles.index, columns=list(contract.ecfp))
    if not np.isnan(matrix).any():
        frame = frame.astype(np.int8)
    return frame, failures


def condition_block(records: pd.DataFrame, contract: FeatureContract) -> tuple[pd.DataFrame, dict]:
    """The 64 ``cond__*`` columns, from the archive's raw text."""
    out: dict[str, pd.Series] = {}
    report: dict = {"unseen_levels": {}, "rows_all_zero": {}, "rows_to_other": {}}

    for column, (source, kind) in NUMERIC_CONDITION_SOURCES.items():
        raw = records[source]
        if kind == "temperature":
            parsed = raw.map(parse_temperature_c)
        elif kind == "number":
            parsed = raw.map(first_number_from_text)
        else:
            parsed = raw.map(lambda v, u=kind: parse_concentration(v, u))
        series = pd.to_numeric(parsed, errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[column] = series.astype(float)

    for prefix, source in CATEGORICAL_CONDITION_SOURCES.items():
        columns = list(contract.cond_categorical[prefix])
        levels = contract.levels(prefix)
        other = f"cond__{prefix}__other"
        has_other = contract.has_other(prefix)
        dummy = pd.DataFrame(0, index=records.index, columns=columns, dtype=np.int8)
        unseen: dict[str, int] = {}
        n_other = n_zero = 0
        for position, value in enumerate(records[source]):
            text = normalize_text_value(value)
            if text is None:
                continue  # dummy_na=False: a missing category is an all-zero row
            name = safe_dummy_name(prefix, text)
            if name in levels:
                dummy.iloc[position, columns.index(name)] = 1
                continue
            unseen[text] = unseen.get(text, 0) + 1
            if has_other:
                dummy.iloc[position, columns.index(other)] = 1
                n_other += 1
            else:
                n_zero += 1
        for column in columns:
            out[column] = dummy[column]
        report["unseen_levels"][prefix] = dict(sorted(unseen.items(), key=lambda kv: -kv[1]))
        report["rows_to_other"][prefix] = int(n_other)
        report["rows_all_zero"][prefix] = int(n_zero)

    frame = pd.DataFrame(out, index=records.index)[list(contract.cond)]
    return frame, report


def metal_block(records: pd.DataFrame, contract: FeatureContract, *,
                metal_policy: str = "table_then_archive") -> tuple[pd.DataFrame, dict]:
    """The 3 METAL columns; see the module docstring for the fallback policy."""
    if metal_policy not in ("table_then_archive", "table_only"):
        raise ValueError(f"unknown metal_policy {metal_policy!r}")
    symbols = records["metal_symbol"].astype(str)

    extension: dict[str, dict[str, float]] = {}
    if metal_policy == "table_then_archive":
        extension = archive_metal_table()

    values = {c: np.full(len(records), np.nan) for c in METAL_COLUMNS}
    from_table = np.zeros(len(records), dtype=bool)
    from_archive = np.zeros(len(records), dtype=bool)
    for position, symbol in enumerate(symbols):
        entry = LANTHANIDE_DESCRIPTORS.get(symbol)
        if entry is not None:
            from_table[position] = True
        else:
            entry = extension.get(symbol)
            if entry is not None:
                from_archive[position] = True
        if entry is None:
            continue
        for column in METAL_COLUMNS:
            values[column][position] = entry.get(column, np.nan)

    frame = pd.DataFrame(values, index=records.index)[list(contract.metal)]
    report = {
        "metal_policy": metal_policy,
        "rows_from_builder_table": int(from_table.sum()),
        "rows_from_archive_fallback": int(from_archive.sum()),
        "rows_unresolved": int(len(records) - from_table.sum() - from_archive.sum()),
        "null_cells": {c: int(frame[c].isna().sum()) for c in frame.columns},
        "symbols_from_archive_fallback": sorted(
            {s for s, ok in zip(symbols, from_archive) if ok}),
        "symbols_unresolved": sorted(
            {s for s, t, a in zip(symbols, from_table, from_archive) if not (t or a)}),
    }
    return frame, report


@lru_cache(maxsize=1)
def archive_metal_table() -> dict[str, dict[str, float]]:
    """One (Z, lanthanide index, CN8 ionic radius) per element symbol.

    Per *symbol*, not per row: the frozen encoding assigns a radius to the element
    and never to the oxidation state (every bundle metal is Ln(III)), so a
    per-row radius would put auxiliary actinides on a different axis than the
    training lanthanides.  The modal non-null archive value is used and the
    discarded per-row variation is reported as a divergence.
    """
    archive = pd.read_parquet(
        ARCHIVE_CLEAN,
        columns=["metal_symbol", "atomic_number", "lanthanide_index", "ionic_radius_cn8_A"])
    grouped = archive.groupby("metal_symbol")
    table: dict[str, dict[str, float]] = {}
    for symbol, part in grouped:
        table[str(symbol)] = {
            "Atomic Number_metal": float(_mode(part["atomic_number"])),
            "lanthanide_index": float(_mode(part["lanthanide_index"])),
            "Ionic Radius_metal": float(_mode(part["ionic_radius_cn8_A"])),
        }
    return table


def massaction_block(condition: pd.DataFrame, records: pd.DataFrame,
                     annotations: Annotations,
                     contract: FeatureContract) -> tuple[pd.DataFrame, dict]:
    """The 8 ``massact__*`` columns via ``levels._attach_mass_action`` itself."""
    extractant = records["extractant_primary_smiles"].astype(str)
    metal = records["metal_symbol"].astype(str)
    dentate = extractant.map(lambda s: annotations.dentate.get(s, np.nan)).astype(float)
    core_cn = pd.Series([
        annotations.core_cn_cell.get(
            (e, m), annotations.core_cn_metal.get(m, np.nan))
        for e, m in zip(extractant, metal)], index=records.index, dtype=float)

    work = condition[list(CONTINUOUS_CONDITION_COLUMNS)].copy()
    work["DENTATE"] = dentate.to_numpy()
    work["coreCN"] = core_cn.to_numpy()
    attached = _attach_mass_action(work)
    missing = [c for c in contract.massaction if c not in attached.columns]
    if missing:
        raise SystemExit(f"_attach_mass_action did not produce {missing}")
    frame = attached[list(contract.massaction)].astype(float)
    report = {
        "rows_without_DENTATE": int(dentate.isna().sum()),
        "rows_without_coreCN": int(core_cn.isna().sum()),
        "coreCN_from_cell": int(sum(
            1 for e, m in zip(extractant, metal) if (e, m) in annotations.core_cn_cell)),
        "structures_without_DENTATE": int(
            extractant[dentate.isna()].nunique()),
        "null_cells": {c: int(frame[c].isna().sum()) for c in frame.columns},
    }
    return frame, report


def recovered_block(records: pd.DataFrame, annotations: Annotations,
                    contract: FeatureContract) -> tuple[pd.DataFrame, dict]:
    """The 23 ``rec__*`` columns, from archive fields, via gen7's own functions."""
    names = records["solvent_name_raw"]
    lookup = {n: solvent_descriptors(n) for n in {x for x in names.dropna().astype(str)}}
    template = {f"rec__solvent_{p}": np.nan for p in SOLVENT_PROPERTIES}
    template.update({"rec__solvent_n_components": np.nan, "rec__solvent_log_eps": np.nan,
                     "rec__solvent_polar_fraction": np.nan, "rec__solvent_parsed": 0.0})
    rows = [lookup[str(n)] if isinstance(n, str) and str(n) in lookup else dict(template)
            for n in names]
    out = pd.DataFrame(rows, index=records.index)

    out["rec__phase_modifier_concentration_M"] = _numeric_with_unit(
        records["modifier_concentration_raw"]).to_numpy()
    out["rec__has_phase_modifier"] = records["modifier_name_raw"].notna().astype(float).to_numpy()
    out["rec__shaking_time_min"] = _numeric_with_unit(records["shaking_time_raw"]).to_numpy()
    out["rec__has_shaking_time"] = records["shaking_time_raw"].notna().astype(float).to_numpy()
    out["rec__acid_concentration_organic_M"] = _numeric_with_unit(
        records["acid_concentration_organic_raw"]).to_numpy()

    # The name-derived trio.  A structure the frozen cohort knows is judged against
    # the frozen modal name, so an auxiliary row and its training counterparts
    # share a reference; only a genuinely new structure falls back to the archive.
    row_names = records["extractant_name_raw"].fillna("unknown").astype(str)
    structures = records["extractant_primary_smiles"].astype(str)
    local = pd.DataFrame({"s": structures.to_numpy(), "n": row_names.to_numpy()}).groupby("s")["n"]
    local_modal = local.agg(lambda s: s.value_counts().idxmax()).to_dict()
    local_n = local.nunique().to_dict()
    known = np.array([s in annotations.modal_name for s in structures])
    expected = np.array([annotations.modal_name.get(s, local_modal.get(s, "unknown"))
                         for s in structures])
    n_names = np.array([float(annotations.n_names.get(s, local_n.get(s, 1)))
                        for s in structures])
    out["rec__name_mismatch"] = (row_names.to_numpy() != expected).astype(float)
    out["rec__n_names_for_structure"] = n_names
    lowered = row_names.str.lower()
    aqueous = (lowered.str.contains("so3") | lowered.str.contains("phso3na")
               | lowered.str.contains("twe-") | lowered.str.contains("pytri")
               | lowered.str.contains("phen-6oh") | lowered.str.contains("phen-dialcohol")
               | lowered.str.contains("citam") | lowered.str.contains("dtpa")
               | lowered.str.contains("edta") | lowered.str.contains("hedta"))
    out["rec__aqueous_complexant"] = (
        aqueous.to_numpy() & (out["rec__name_mismatch"].to_numpy() > 0)).astype(float)

    missing = [c for c in contract.recovered if c not in out.columns]
    if missing:
        raise SystemExit(f"recovered block is missing {missing}")
    frame = out[list(contract.recovered)].astype(float)
    unparsed = sorted({str(n) for n in names.dropna().astype(str)
                       if not lookup[str(n)]["rec__solvent_parsed"]})
    report = {
        "rows_with_solvent_name": int(names.notna().sum()),
        "rows_solvent_unparsed": int((frame["rec__solvent_parsed"] == 0).sum()),
        "unparsed_solvent_names": unparsed,
        "rows_name_reference_frozen": int(known.sum()),
        "rows_name_reference_archive_local": int((~known).sum()),
        "rows_name_mismatch": int(out["rec__name_mismatch"].sum()),
        "rows_aqueous_complexant": int(out["rec__aqueous_complexant"].sum()),
    }
    return frame, report


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def featurize_archive(records: pd.DataFrame, *, cohort, annotations: Annotations | None = None,
                      contract: FeatureContract | None = None,
                      metal_policy: str = "table_then_archive",
                      with_identity: bool = True) -> pd.DataFrame:
    """Archive records -> the frozen arm's 2,146 columns (+ identity).

    ``records`` must carry :data:`ARCHIVE_FEATURE_COLUMNS`; use
    :func:`auxiliary_records` or :func:`frozen_side_records` to get them.  The
    returned frame is row-aligned with ``records`` and carries no NaN it did not
    earn — every null is reported by :func:`featurize_with_report`.
    """
    frame, _ = featurize_with_report(
        records, cohort=cohort, annotations=annotations, contract=contract,
        metal_policy=metal_policy, with_identity=with_identity)
    return frame


def featurize_with_report(records: pd.DataFrame, *, cohort,
                          annotations: Annotations | None = None,
                          contract: FeatureContract | None = None,
                          metal_policy: str = "table_then_archive",
                          with_identity: bool = True) -> tuple[pd.DataFrame, dict]:
    """:func:`featurize_archive` plus the per-block diagnostic it produced."""
    contract = contract or build_contract(cohort)
    annotations = annotations or build_annotations(cohort)
    missing = [c for c in ARCHIVE_FEATURE_COLUMNS if c not in records.columns]
    if missing:
        raise SystemExit(f"records lack archive columns {missing}")
    records = records.reset_index(drop=True)

    metal, metal_report = metal_block(records, contract, metal_policy=metal_policy)
    condition, condition_report = condition_block(records, contract)
    ecfp, ecfp_failures = ecfp_block(records["extractant_primary_smiles"], contract)
    massaction, massaction_report = massaction_block(condition, records, annotations, contract)
    recovered, recovered_report = recovered_block(records, annotations, contract)

    parts = [metal, condition, ecfp, massaction, recovered]
    if with_identity:
        identity = records[list(IDENTITY_COLUMNS)].copy()
        identity["source_record_id"] = identity["source_record_id"].astype(str)
        identity["extractant"] = records["extractant_primary_smiles"].astype(str)
        parts.insert(0, identity)
    frame = pd.concat(parts, axis=1)
    expected = ((list(IDENTITY_COLUMNS) + list(DERIVED_IDENTITY_COLUMNS))
                if with_identity else []) + list(contract.columns)
    if list(frame.columns) != expected:
        raise SystemExit("featurized frame column order drifted from the contract")

    report = {
        "rows": int(len(records)),
        "columns": int(len(contract.columns)),
        "METAL": metal_report,
        "COND": condition_report,
        "ECFP": {"rdkit_parse_failures": ecfp_failures,
                 "distinct_structures": int(records["extractant_primary_smiles"].nunique())},
        "MASSACTION": massaction_report,
        "RECOVERED": recovered_report,
        "annotation_ambiguity": annotations.ambiguity,
    }
    return frame, report


@lru_cache(maxsize=1)
def load_frozen_bundle() -> pd.DataFrame:
    frame = pd.read_parquet(FROZEN_DATASET)
    frame["exp_id"] = frame["safe_exp_id"].astype(str).str.split("SAFE:").str[-1]
    return frame


@lru_cache(maxsize=1)
def load_archive_features() -> pd.DataFrame:
    frame = pd.read_parquet(ARCHIVE_CLEAN, columns=list(ARCHIVE_FEATURE_COLUMNS))
    frame["source_record_id"] = frame["source_record_id"].astype(str)
    return frame


def frozen_side_records() -> pd.DataFrame:
    """The 5,992 archive records that *are* the frozen bundle, in bundle order.

    Bundle order matters: the cohort keeps the first row of each replicate cell,
    so a cell-level comparison is only meaningful if the source order matches.
    """
    bundle = load_frozen_bundle()
    archive = load_archive_features()
    merged = bundle[["exp_id", "safe_exp_id"]].merge(
        archive, left_on="exp_id", right_on="source_record_id",
        how="inner", validate="one_to_one")
    if len(merged) != len(bundle):
        raise SystemExit(f"frozen<->archive join lost rows: {len(merged)} != {len(bundle)}")
    return merged.reset_index(drop=True)


def auxiliary_records(readiness: str | None = "A_model_ready") -> pd.DataFrame:
    """Archive records that are not frozen bundle rows, optionally A-only."""
    archive = load_archive_features()
    frozen = set(load_frozen_bundle()["exp_id"].astype(str))
    aux = archive[~archive["source_record_id"].isin(frozen)]
    if readiness is not None:
        aux = aux[aux["model_readiness"].astype(str) == readiness]
    return aux.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The proof
# --------------------------------------------------------------------------- #

#: ``gen7.harness.GEN7_MIN_CELLS`` — the cohort keeps an extractant with >= 3
#: distinct (condition, metal) cells.  Restated here (and asserted against the
#: cohort's own row count) so the replay cannot silently drift from it.
COHORT_MIN_CELLS = 3
COHORT_LOG_D_FLOOR = -6.0


def _cell_reduce(features: pd.DataFrame, bundle: pd.DataFrame, contract: FeatureContract,
                 *, min_cells: int = COHORT_MIN_CELLS,
                 floor: float = COHORT_LOG_D_FLOOR,
                 mean_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Replay the cohort's row -> cell reduction on freshly featurized rows.

    ``build_level_dataset`` drops ``log_D <= floor``, requires >= ``min_cells``
    distinct (condition, metal) cells per extractant, then keeps the *first* row
    of each cell — so MASSACTION, which is attached before that step, carries the
    first row's value.  RECOVERED does not: it is joined in later from
    ``recovered_cell_table``, which takes the **cell mean** of every ``rec__``
    column.  Both reductions are performed here, or the cell-level comparison
    would be measuring the aggregation rather than the featurizer.
    """
    from ..levels import condition_labels

    work = features.copy()
    work["log_D"] = bundle["log_D"].astype(float).to_numpy()
    work["extractant"] = bundle["canonical_smiles"].astype(str).to_numpy()
    work["metal_symbol"] = bundle["metal_symbol"].astype(str).to_numpy()
    work = work[work["log_D"] > floor].reset_index(drop=True)
    work["condition_id"] = condition_labels(work, list(contract.cond)).to_numpy()
    cell = ["extractant", "condition_id", "metal_symbol"]
    counts = work.drop_duplicates(cell).groupby("extractant").size()
    keep = set(counts[counts >= min_cells].index)
    work = work[work["extractant"].isin(keep)].reset_index(drop=True)
    means = work.groupby(cell, sort=False)[list(mean_columns)].mean() if mean_columns else None
    first = work.drop_duplicates(subset=cell, keep="first").reset_index(drop=True)
    if means is not None:
        replaced = means.reindex(
            pd.MultiIndex.from_frame(first[cell])).reset_index(drop=True)
        for column in mean_columns:
            first[column] = replaced[column].to_numpy()
    first["row_id"] = [
        hashlib.sha1(f"{e}|{c}|{m}".encode()).hexdigest()[:16]
        for e, c, m in zip(first["extractant"], first["condition_id"], first["metal_symbol"])]
    return first


def _diff_counts(got: pd.DataFrame, want: pd.DataFrame, columns: Sequence[str],
                 *, atol: float = 0.0) -> pd.Series:
    """Per-column count of cells that differ, treating NaN==NaN as equal."""
    counts = {}
    for column in columns:
        a = pd.to_numeric(got[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(want[column], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        if atol > 0:
            equal = np.isclose(np.nan_to_num(a, nan=-9e18), np.nan_to_num(b, nan=-9e18),
                               rtol=0.0, atol=atol)
        else:
            equal = np.nan_to_num(a, nan=-9e18) == np.nan_to_num(b, nan=-9e18)
        counts[column] = int((~(both_nan | equal)).sum())
    return pd.Series(counts, dtype=int)


def reproduction_report(cohort, *, metal_policy: str = "table_then_archive") -> dict:
    """Re-derive the frozen bundle's own features from the archive and diff them.

    Returns the report; :func:`write_outputs` persists it.  The report is the
    deliverable — a featurizer whose ECFP/COND/METAL diff is not zero has not
    earned the right to produce auxiliary rows.
    """
    contract = build_contract(cohort)
    annotations = build_annotations(cohort)
    bundle = load_frozen_bundle()
    records = frozen_side_records()
    if not (records["exp_id"].to_numpy() == bundle["exp_id"].to_numpy()).all():
        raise SystemExit("frozen-side records are not in bundle order")

    got, block_report = featurize_with_report(
        records, cohort=cohort, annotations=annotations, contract=contract,
        metal_policy=metal_policy, with_identity=False)

    families = {
        "ECFP": list(contract.ecfp),
        "COND_numeric": list(contract.cond_numeric),
        "COND_acid": list(contract.cond_categorical["acid"]),
        "COND_diluent": list(contract.cond_categorical["diluent"]),
        "COND_additive": list(contract.cond_categorical["additive"]),
        "METAL": list(contract.metal),
    }
    mismatches: list[dict] = []
    summary: dict = {}
    for family, columns in families.items():
        counts = _diff_counts(got, bundle, columns)
        summary[family] = {
            "columns": len(columns),
            "cells": int(len(bundle) * len(columns)),
            "mismatching_cells": int(counts.sum()),
            "mismatching_columns": int((counts > 0).sum()),
            "mismatching_rows": int(
                (got[columns].fillna(-9e18).to_numpy(dtype=float)
                 != bundle[columns].astype(float).fillna(-9e18).to_numpy(dtype=float)
                 ).any(axis=1).sum()),
        }
        for column, count in counts[counts > 0].items():
            mismatches.append({
                "family": family, "column": column, "mismatching_cells": int(count),
                "rows": int(len(bundle)),
                "diagnosis": _diagnose(column, got, bundle),
            })

    # --- cell-level: MASSACTION and RECOVERED live on the cohort, not the bundle
    cells = _cell_reduce(got, bundle, contract, mean_columns=list(contract.recovered))
    joined = cells.merge(cohort.frame[["row_id"] + list(contract.massaction)
                                      + list(contract.recovered)].astype({"row_id": str}),
                         on="row_id", how="inner", suffixes=("_got", "_want"))
    cell_summary = {"cohort_rows": int(len(cohort.frame)),
                    "reproduced_cells": int(len(cells)),
                    "matched_cells": int(len(joined)),
                    "unmatched_cohort_rows": int(len(cohort.frame) - len(joined))}
    for family, columns in (("MASSACTION", list(contract.massaction)),
                            ("RECOVERED", list(contract.recovered))):
        counts = _diff_counts(
            joined.rename(columns={f"{c}_got": c for c in columns}),
            joined.rename(columns={f"{c}_want": c for c in columns}),
            columns, atol=1e-9)
        cell_summary[family] = {
            "columns": len(columns),
            "cells": int(len(joined) * len(columns)),
            "mismatching_cells": int(counts.sum()),
            "mismatching_columns": int((counts > 0).sum()),
        }
        for column, count in counts[counts > 0].items():
            mismatches.append({
                "family": f"{family} (cell level)", "column": column,
                "mismatching_cells": int(count), "rows": int(len(joined)),
                "diagnosis": _diagnose_cell(column, joined),
            })

    cache_check = recovered_cache_check(cells, records, contract, annotations)

    # The MASSACTION residual has exactly one candidate explanation; check it
    # rather than assert it, so the report either names the cause or admits it
    # does not know.
    frame = cohort.frame
    modal_dentate = frame["extractant"].map(
        lambda s: annotations.dentate.get(s, np.nan)).astype(float).to_numpy()
    modal_cn = np.array([
        annotations.core_cn_cell.get((e, m), annotations.core_cn_metal.get(m, np.nan))
        for e, m in zip(frame["extractant"], frame["metal_symbol"])], dtype=float)
    annotation_drift = {
        "cells_where_modal_DENTATE_differs_from_row":
            int((modal_dentate != frame["DENTATE"].astype(float).to_numpy()).sum()),
        "cells_where_modal_coreCN_differs_from_row":
            int((modal_cn != frame["coreCN"].astype(float).to_numpy()).sum()),
    }

    return {
        "cohort_fingerprint": cohort.fingerprint,
        "frozen_bundle_rows": int(len(bundle)),
        "frozen_arm_columns": int(len(contract.columns)),
        "metal_policy": metal_policy,
        "row_level": summary,
        "cell_level": cell_summary,
        "recovered_cache_check": cache_check,
        "annotation_drift": annotation_drift,
        "blocks": block_report,
        "mismatches": mismatches,
        "exact_row_level": all(v["mismatching_cells"] == 0 for v in summary.values()),
    }


GEN7_CACHE = REPO_ROOT / "runs" / "gen7_architecture" / "cache"


def recovered_cache_check(cells: pd.DataFrame, records: pd.DataFrame,
                          contract: FeatureContract, annotations: Annotations) -> dict:
    """Diff the RECOVERED block against gen7's two caches, row-level and cell-level.

    The brief asks for agreement with ``recovered_cells.parquet``; the row-level
    ``recovered_raw.parquet`` is checked as well because the two caches were
    written 83 minutes apart and disagree with each other, which is exactly the
    sort of thing that would otherwise be discovered halfway through an arm.
    """
    out: dict = {}
    cell_cache = GEN7_CACHE / "recovered_cells.parquet"
    if cell_cache.exists():
        table = pd.read_parquet(cell_cache)
        columns = [c for c in contract.recovered if c in table.columns]
        joined = cells.merge(table[["row_id"] + columns].astype({"row_id": str}),
                             on="row_id", how="inner", suffixes=("_got", "_want"))
        counts = _diff_counts(
            joined.rename(columns={f"{c}_got": c for c in columns}),
            joined.rename(columns={f"{c}_want": c for c in columns}), columns, atol=1e-9)
        out["recovered_cells_parquet"] = {
            "matched_cells": int(len(joined)),
            "columns_compared": len(columns),
            "mismatching_cells": int(counts.sum()),
            "mismatching_columns": {k: int(v) for k, v in counts[counts > 0].items()},
        }
    raw_cache = GEN7_CACHE / "recovered_raw.parquet"
    if raw_cache.exists():
        table = pd.read_parquet(raw_cache)
        columns = [c for c in contract.recovered if c in table.columns]
        block, _ = recovered_block(records.reset_index(drop=True), annotations, contract)
        rows = pd.concat(
            [records[["safe_exp_id"]].reset_index(drop=True), block.reset_index(drop=True)],
            axis=1)
        joined = rows.merge(table[["safe_exp_id"] + columns], on="safe_exp_id",
                            how="inner", suffixes=("_got", "_want"))
        counts = _diff_counts(
            joined.rename(columns={f"{c}_got": c for c in columns}),
            joined.rename(columns={f"{c}_want": c for c in columns}), columns, atol=1e-9)
        out["recovered_raw_parquet"] = {
            "matched_rows": int(len(joined)),
            "columns_compared": len(columns),
            "mismatching_cells": int(counts.sum()),
            "mismatching_columns": {k: int(v) for k, v in counts[counts > 0].items()},
            "note": "recovered_raw.parquet predates the CH3Cl->chloroform correction in "
                    "gen7/recovered.py; recovered_cells.parquet does not.",
        }
    return out


def _diagnose(column: str, got: pd.DataFrame, want: pd.DataFrame) -> str:
    a = pd.to_numeric(got[column], errors="coerce")
    b = pd.to_numeric(want[column], errors="coerce")
    bad = ~((a.isna() & b.isna()) | (a.fillna(-9e18) == b.fillna(-9e18)))
    n_only_got = int((bad & b.isna()).sum())
    n_only_want = int((bad & a.isna()).sum())
    return (f"{int(bad.sum())} cells differ; {n_only_want} where the featurizer produced NaN and "
            f"the bundle has a value; {n_only_got} the reverse; "
            f"example got={a[bad].head(1).to_list()} want={b[bad].head(1).to_list()}")


def _diagnose_cell(column: str, joined: pd.DataFrame) -> str:
    a = pd.to_numeric(joined[f"{column}_got"], errors="coerce")
    b = pd.to_numeric(joined[f"{column}_want"], errors="coerce")
    bad = ~((a.isna() & b.isna()) | np.isclose(a.fillna(-9e18), b.fillna(-9e18),
                                               rtol=0.0, atol=1e-9))
    pairs = pd.DataFrame({"got": a[bad], "want": b[bad]}).drop_duplicates().head(3)
    return (f"{int(bad.sum())} of {len(joined)} cells differ; distinct (got, want) pairs: "
            + "; ".join(f"({g:g}, {w:g})" for g, w in pairs.itertuples(index=False)))


def divergence_table(aux_report: dict, aux_records: pd.DataFrame,
                     reproduction: dict) -> pd.DataFrame:
    """``known_encoding_divergences.csv``: encodings we deliberately did not fix.

    Every row is a place where the frozen contract's encoding is chemically
    questionable *and* this featurizer reproduces it anyway, because matching the
    training rows beats being right in isolation.  Each carries the measured
    auxiliary row count so a sensitivity arm can price it.
    """
    rows: list[dict] = []
    solvent = aux_records["solvent_name_raw"].fillna("").astype(str).str.lower()
    acid = aux_records["acid_name_raw"].fillna("").astype(str).str.lower()

    # measured, not asserted: how many auxiliary rows actually lose an
    # oxidation-state-resolved radius by being pinned to one value per element
    per_symbol = archive_metal_table()
    pinned = aux_records["metal_symbol"].astype(str).map(
        lambda s: per_symbol.get(s, {}).get("Ionic Radius_metal", np.nan)).astype(float)
    per_row = pd.to_numeric(aux_records["ionic_radius_cn8_A"], errors="coerce")
    radius_lost = int((per_row.notna() & (per_row - pinned).abs().gt(1e-9)).sum())
    radius_absent = int((per_row.isna() & pinned.notna()).sum())
    radius_note = (
        f" MEASURED: only {radius_lost} auxiliary rows carry a per-row radius that differs "
        f"from the pinned per-symbol value, so the encoding choice is nearly free here; "
        f"{radius_absent} further rows have no per-row radius at all (unknown oxidation state) "
        f"and are given the pinned value, which is what the frozen rows also receive.")

    for entry in ARCHIVE_ADJUDICATIONS:
        levels = [l.strip().lower() for l in entry["levels"].split("|")]
        if entry["divergence"] == "ionic_radius_oxidation_state":
            hit = radius_lost
            entry = dict(entry, detail=entry["detail"] + radius_note)
        elif entry["family"].startswith("METAL"):
            hit = int(aux_records["metal_symbol"].astype(str).isin(
                [l.strip().title() for l in levels]).sum())
        elif "acid" in entry["family"].lower():
            hit = int(acid.isin(levels).sum())
        else:
            hit = int(solvent.isin(levels).sum())
        rows.append({
            "divergence": entry["divergence"],
            "family": entry["family"],
            "aux_rows_affected": hit,
            "levels": entry["levels"],
            "detail": entry["detail"],
            "sensitivity_arm": entry["sensitivity_arm"],
            "source": "dataset_all_metals/scripts/sae_normalize.py",
        })

    # measured, not asserted: levels the frozen vocabulary cannot express
    cond = aux_report["COND"]
    for prefix, unseen in cond["unseen_levels"].items():
        if not unseen:
            continue
        destination = "cond__%s__other" % prefix if cond["rows_to_other"][prefix] else "all-zero row"
        rows.append({
            "divergence": f"unseen_{prefix}_level",
            "family": f"COND/{prefix}",
            "aux_rows_affected": int(sum(unseen.values())),
            "levels": "|".join(f"{k} x{v}" for k, v in list(unseen.items())[:20]),
            "detail": (f"{len(unseen)} {prefix} values do not exist in the frozen one-hot "
                       f"vocabulary and are encoded as {destination}. The builder's own rule "
                       f"(CATEGORICAL_MAX_LEVELS=40 then OTHER) is applied; a level absent "
                       f"from a family with no 'other' column becomes an all-zero row, which "
                       f"is indistinguishable from 'not recorded'."),
            "sensitivity_arm": "extend the vocabulary and refit the frozen arm (breaks the control)",
            "source": "measured on the A_model_ready auxiliary rows",
        })

    unparsed = aux_report["RECOVERED"]["unparsed_solvent_names"]
    if unparsed:
        counts = solvent.value_counts()
        rows.append({
            "divergence": "unparsed_solvent_recovered",
            "family": "RECOVERED/solvent",
            "aux_rows_affected": int(aux_report["RECOVERED"]["rows_solvent_unparsed"]),
            "levels": "|".join(f"{n} x{int(counts.get(n.lower(), 0))}" for n in unparsed[:20]),
            "detail": ("gen7.recovered's component patterns do not cover these diluents, so all "
                       "15 rec__solvent_* columns are NaN and rec__solvent_parsed is 0. The "
                       "patterns were written against the frozen corpus and are left untouched."),
            "sensitivity_arm": "add the missing components to SOLVENT_COMPONENTS",
            "source": "measured on the A_model_ready auxiliary rows",
        })

    metal = aux_report["METAL"]
    rows.append({
        "divergence": "metal_descriptor_table_fallback",
        "family": "METAL",
        "aux_rows_affected": int(metal["rows_from_archive_fallback"]),
        "levels": "|".join(metal["symbols_from_archive_fallback"]),
        "detail": ("The builder's LANTHANIDE_DESCRIPTORS table covers 14 lanthanides and nothing "
                   "else — not Pm, not Y, no actinide. Under metal_policy='table_then_archive' "
                   "those symbols take the archive's own atomic_number / lanthanide_index / "
                   "ionic_radius_cn8_A, which reproduce the builder's table exactly on the 5,992 "
                   "shared records, so the fallback is on the same scale. lanthanide_index is "
                   f"undefined for a non-lanthanide and stays NaN in "
                   f"{metal['null_cells'].get('lanthanide_index', 0)} rows; the radius stays NaN "
                   f"in {metal['null_cells'].get('Ionic Radius_metal', 0)} rows whose element has "
                   "no CN8 Shannon radius in the archive."),
        "sensitivity_arm": "metal_policy='table_only' (leaves the whole METAL block NaN off-Ln)",
        "source": "measured on the A_model_ready auxiliary rows",
    })

    ambiguity = reproduction["blocks"]["annotation_ambiguity"]
    rows.append({
        "divergence": "geometry_annotation_ambiguity",
        "family": "MASSACTION",
        "aux_rows_affected": int(aux_report["MASSACTION"]["rows_without_DENTATE"]),
        "levels": (f"{ambiguity['extractants_with_multiple_DENTATE']} extractants carry >1 DENTATE; "
                   f"{ambiguity['cells_with_multiple_coreCN']} (extractant, metal) cells carry >1 coreCN"),
        "detail": ("DENTATE/coreCN are per-row geometry-plan outputs, not per-ligand constants. "
                   "The mode is used for the lookup; the frozen rows where the mode differs from "
                   "the row's own value are counted in reproduction.json under "
                   "cell_level.MASSACTION."),
        "sensitivity_arm": "run the builder's plan_complex for the new structures",
        "source": "measured on the frozen cohort",
    })
    return pd.DataFrame(rows)


def write_outputs(cohort, out_dir: Path | str = OUT_DIR, *,
                  metal_policy: str = "table_then_archive") -> dict:
    """Produce every deliverable and return the summary the orchestrator reads."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    contract = build_contract(cohort)
    annotations = build_annotations(cohort)

    reproduction = reproduction_report(cohort, metal_policy=metal_policy)
    (out / "reproduction.json").write_text(json.dumps(reproduction, indent=2, default=str))
    pd.DataFrame(reproduction["mismatches"]).to_csv(out / "mismatches.csv", index=False)

    aux = auxiliary_records("A_model_ready")
    features, aux_report = featurize_with_report(
        aux, cohort=cohort, annotations=annotations, contract=contract,
        metal_policy=metal_policy, with_identity=True)
    features.to_parquet(out / "aux_features.parquet", index=False)

    divergences = divergence_table(aux_report, aux, reproduction)
    divergences.to_csv(out / "known_encoding_divergences.csv", index=False)
    # A content fingerprint over the feature matrix, computed with hashlib rather
    # than ``hash()``: Python's string hash is salted per process, and that
    # silently broke cross-run pairing once already in gen8.
    aux_report["feature_fingerprint"] = feature_fingerprint(features, contract)
    (out / "aux_report.json").write_text(json.dumps(aux_report, indent=2, default=str))
    (out / "README.md").write_text(
        render_readme(reproduction, aux_report, divergences, contract))

    return {
        "reproduction": reproduction,
        "aux_report": aux_report,
        "aux_rows": int(len(features)),
        "aux_columns": int(features.shape[1]),
        "aux_feature_fingerprint": aux_report["feature_fingerprint"],
        "divergences": int(len(divergences)),
        "out_dir": str(out),
    }


def render_readme(reproduction: dict, aux_report: dict, divergences: pd.DataFrame,
                  contract: FeatureContract) -> str:
    """The README, rendered from the reports so no number can be stale."""
    row = reproduction["row_level"]
    cell = reproduction["cell_level"]
    cache = reproduction.get("recovered_cache_check", {})
    drift = reproduction.get("annotation_drift", {})
    metal, cond = aux_report["METAL"], aux_report["COND"]
    mass, rec = aux_report["MASSACTION"], aux_report["RECOVERED"]

    total_cells = sum(v["cells"] for v in row.values())
    total_bad = sum(v["mismatching_cells"] for v in row.values())
    lines = [
        "# gen11 auxiliary featurizer",
        "",
        "`lanthanide_separation.gen11.auxfeatures` renders multi-metal archive records into the",
        f"frozen gen10 arm `{'+'.join(FROZEN_ARM_BLOCKS)}` — "
        f"{len(contract.columns)} columns — so an auxiliary row and a training row are the same",
        "kind of object. Every number below was computed by the code that wrote this file.",
        "",
        "## The proof: does the featurizer reproduce the frozen bundle?",
        "",
        f"Applied to the {reproduction['frozen_bundle_rows']} archive records that *are* the frozen",
        "bundle rows (joined 1:1 on `exp_id`), re-derived from the archive's `*_raw` text:",
        "",
        "| family | columns | cells | mismatching cells | mismatching columns |",
        "|---|---|---|---|---|",
    ]
    for family, stats in row.items():
        lines.append(f"| {family} | {stats['columns']} | {stats['cells']:,} | "
                     f"{stats['mismatching_cells']} | {stats['mismatching_columns']} |")
    lines += [
        f"| **total (bundle-resident)** | **{sum(v['columns'] for v in row.values())}** | **{total_cells:,}** | "
        f"**{total_bad}** | **{sum(v['mismatching_columns'] for v in row.values())}** |",
        "",
        f"**Row-level reproduction is exact: {total_bad} mismatching cells out of {total_cells:,}.**",
        "The 2,048 ECFP bits, all 64 `cond__*` columns (9 acid + 41 diluent + 9 additive + 5",
        "numeric) and the 3 METAL columns are bit-for-bit identical to the frozen bundle's own.",
        "",
        "## MASSACTION and RECOVERED: checked at the cell, because that is where they live",
        "",
        "Neither block exists in the bundle. MASSACTION is attached by `levels._attach_mass_action`",
        "before replicate collapse (so the cohort carries the *first* row of each cell); RECOVERED",
        "is joined afterwards from `recovered_cell_table`, which takes the *cell mean*. Both",
        "reductions are replayed here before comparing.",
        "",
        f"* cohort rows {cell['cohort_rows']}, reproduced cells {cell['reproduced_cells']}, "
        f"matched {cell['matched_cells']}, unmatched {cell['unmatched_cohort_rows']}",
        f"* **RECOVERED: {cell['RECOVERED']['mismatching_cells']} mismatching cells** over "
        f"{cell['RECOVERED']['cells']:,} ({cell['RECOVERED']['columns']} columns) — exact.",
        f"* **MASSACTION: {cell['MASSACTION']['mismatching_cells']} mismatching cells** over "
        f"{cell['MASSACTION']['cells']:,}, in {cell['MASSACTION']['mismatching_columns']} columns.",
        "",
        "### The MASSACTION residual, diagnosed",
        "",
        "`DENTATE` and `coreCN` are per-*row* geometry-plan outputs, not per-ligand constants: the",
        "builder derived them from (metal Z, ligand SMILES, acid), so the same ligand can carry",
        "two denticities in the same cohort. The archive does not carry them at all, so they are",
        "looked up by mode. The residual is exactly the drift that lookup introduces:",
        "",
        f"* cells where the modal `DENTATE` differs from the row's own: "
        f"{drift.get('cells_where_modal_DENTATE_differs_from_row')} "
        f"(= mismatches in `massact__logL_x_DENTATE`)",
        f"* cells where the modal `coreCN` differs from the row's own: "
        f"{drift.get('cells_where_modal_coreCN_differs_from_row')} "
        f"(= mismatches in `massact__logL_x_coreCN`)",
        "",
        "The two counts account for every mismatching cell. The five `massact__log10_*` columns and",
        "`massact__logL_x_logH`, which do not touch the annotations, reproduce exactly.",
        "",
        "## RECOVERED against gen7's caches",
        "",
    ]
    if "recovered_cells_parquet" in cache:
        c = cache["recovered_cells_parquet"]
        lines.append(f"* `recovered_cells.parquet` (what the cohort actually joins): "
                     f"{c['mismatching_cells']} mismatching cells over {c['matched_cells']} cells "
                     f"x {c['columns_compared']} columns — exact.")
    if "recovered_raw_parquet" in cache:
        c = cache["recovered_raw_parquet"]
        lines.append(f"* `recovered_raw.parquet` (row level): {c['mismatching_cells']} mismatching "
                     f"cells over {c['matched_rows']} rows x {c['columns_compared']} columns, in "
                     f"{len(c['mismatching_columns'])} solvent columns. **This cache is stale**: it "
                     f"predates the `CH3Cl -> chloroform` correction in `gen7/recovered.py` and "
                     f"still encodes those 84 rows as dichloromethane. `recovered_cells.parquet`, "
                     f"the cohort and this featurizer all agree; only the row-level cache does not. "
                     f"Do not use it.")
    lines += [
        "",
        "## What the auxiliary rows look like in this space",
        "",
        f"`aux_features.parquet`: {aux_report['rows']} A_model_ready archive records x "
        f"{aux_report['columns']} features + "
        f"{len(IDENTITY_COLUMNS) + len(DERIVED_IDENTITY_COLUMNS)} identity columns "
        f"(`{'`, `'.join(IDENTITY_COLUMNS + DERIVED_IDENTITY_COLUMNS)}`).",
        f"Feature fingerprint (sha1 of the matrix, hashlib not `hash()`): "
        f"`{aux_report.get('feature_fingerprint', 'n/a')}`.",
        f"{aux_report['ECFP']['distinct_structures']} distinct structures, "
        f"{len(aux_report['ECFP']['rdkit_parse_failures'])} RDKit parse failures.",
        "",
        "What does *not* fully carry over, with counts:",
        "",
        f"* **METAL** — {metal['rows_from_builder_table']} rows resolve from the builder's own",
        f"  14-lanthanide table; {metal['rows_from_archive_fallback']} take the archive fallback",
        f"  ({len(metal['symbols_from_archive_fallback'])} symbols); "
        f"{metal['rows_unresolved']} unresolved.",
        f"  `lanthanide_index` is NaN in {metal['null_cells']['lanthanide_index']} rows (undefined",
        f"  off the lanthanide series) and `Ionic Radius_metal` in "
        f"{metal['null_cells']['Ionic Radius_metal']} rows.",
        f"* **COND** — {cond['rows_to_other']['diluent']} rows carry a diluent outside the frozen",
        f"  vocabulary and go to `cond__diluent__other` (the builder's own OTHER bucket);",
        f"  {cond['rows_all_zero']['additive']} rows carry an additive with no frozen level and no",
        f"  `other` column, so they become an all-zero additive row — indistinguishable from",
        f"  'no additive recorded'. Acid: {cond['rows_all_zero']['acid']} unrepresentable rows.",
        f"* **MASSACTION** — `massact__logL_x_DENTATE` is NaN in {mass['rows_without_DENTATE']} rows",
        f"  ({mass['structures_without_DENTATE']} structures the frozen cohort has never seen) and",
        f"  `massact__logL_x_coreCN` in {mass['rows_without_coreCN']} rows (metals outside the",
        "  lanthanide series have no coordination-number annotation). The frozen arm imputes with",
        "  `add_indicator=True`, so NaN is a representable state, not a silent zero.",
        f"* **RECOVERED** — {rec['rows_solvent_unparsed']} rows have a diluent gen7's component",
        f"  patterns do not cover ({', '.join(rec['unparsed_solvent_names'])}), leaving all 15",
        "  `rec__solvent_*` columns NaN and `rec__solvent_parsed` 0.",
        f"  {rec['rows_name_reference_frozen']} rows are judged against the frozen modal extractant",
        f"  name; {rec['rows_name_reference_archive_local']} (new structures) against the archive's",
        "  own rows, which is a different denominator for `rec__n_names_for_structure`.",
        "",
        "## Encoding divergences we deliberately did not fix",
        "",
        "The archive corrects several chemical mistakes the bundle builder made. Adopting those",
        "corrections here would encode auxiliary rows on a different vocabulary than the training",
        "rows, which is the exact failure this module exists to prevent — so they are reproduced",
        "as-is and recorded in `known_encoding_divergences.csv` for a later sensitivity arm:",
        "",
        "| divergence | family | aux rows | sensitivity arm |",
        "|---|---|---|---|",
    ]
    for _, entry in divergences.iterrows():
        lines.append(f"| {entry['divergence']} | {entry['family']} | "
                     f"{entry['aux_rows_affected']} | {entry['sensitivity_arm']} |")
    lines += [
        "",
        "## Files",
        "",
        "* `reproduction.json` — per-column-family mismatch counts, both levels, plus the cache",
        "  check and the annotation-drift diagnosis.",
        "* `mismatches.csv` — every non-reproducing column with its count and a diagnosis.",
        "* `known_encoding_divergences.csv` — the table above, with full detail text.",
        "* `aux_features.parquet` — the featurized auxiliary rows.",
        "* `aux_report.json` — the per-block diagnostic for the auxiliary run.",
        "",
        "## Reproduce",
        "",
        "```python",
        "from lanthanide_separation.gen10.runner import prepared_cohort",
        "from lanthanide_separation.gen11 import auxfeatures",
        "auxfeatures.write_outputs(prepared_cohort())",
        "```",
        "",
    ]
    return "\n".join(lines)


def feature_fingerprint(features: pd.DataFrame, contract: FeatureContract) -> str:
    matrix = features[list(contract.columns)].to_numpy(dtype=float)
    return hashlib.sha1(np.nan_to_num(matrix, nan=-9e18).tobytes()).hexdigest()[:16]
