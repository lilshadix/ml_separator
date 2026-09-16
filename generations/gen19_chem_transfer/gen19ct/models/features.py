"""``models/features.py`` -- shared, train-only feature builders for the learned arms (B5 = M0, FLAT_CAT, B6 / B6r0,
B8, M1-M7).

Registered text (``preregistration.md``, sealed 2026-09-15): section 2 "Features" (no ``load.PROVENANCE_COLUMNS``
column, publication / study id, DOI, data location, row id or free-text comment is ever a feature; every fitted
preprocessing step is fitted on the training rows of the fold it serves) and "Acid-molarity semantics" (the acid-grid
flag); section 3.5 (``d_desc``, standardised over outer-training systems); section 5 (B5 metal / extractant /
condition blocks, B6 ``z_m`` / ``z_s``, B8 METAL / COND / LIG2D_EXT / MASSACTION with ECFP-cluster weights and the
TOPO39 direction inputs, FLAT_CAT); section 6 (M1: ``e_m = e_shared(z_m) + e_series + e_ox + e_element``,
``e_l = W_l z_l + delta_l``, a condition encoder).  Brief section 12: scaling, imputation, PCA and embeddings are
fitted on train only.  Nothing in this module reads ``log_D`` and nothing here fits or scores a model.

Blocks (raw columns; every output column is prefixed ``<block>__``)
------------------------------------------------------------------
``metal`` (:class:`MetalBlock`, ``descriptors/metals.csv``, section 5 B5 metal block)
    numeric ``Z, formal_charge, radius_cn6_A, radius_cn8_A, radius_cn9_A, f_electron_count, d_electron_count,
    electronegativity_pauling``; categorical ``category, series, hsab_class``.  ``polarizability_atom_au`` is excluded
    (all NA).  An unknown-state (X(?)) row gets the element-level columns only (``Z``, ``category``, ``series``,
    ``electronegativity_pauling``); every ion-level value stays NaN -- no state is imputed.  Embedding ids of M1
    (vocabularies fitted on the training rows, index 0 reserved for a value unseen in training): ``metal_state_id``,
    ``series_id`` (``Ln`` / ``An`` / ``other``), ``ox_id`` (``<NA>`` for X(?) rows), ``element_id``, ``category_id``.
``extractant`` (:class:`ExtractantBlock`, ``descriptors/extractant_systems.csv`` + ``extractant_components.csv``)
    numeric ``n_components`` (``n_organic_extractants``), the 19 ``donor_*`` counts, ``n_donor_sites`` and
    ``denticity_proxy`` of the primary extractant, the concentration-share-weighted ``mw, mol_logp, tpsa,
    rotatable_bonds`` (shares = the row's ``extractant_concentrations_sorted_M`` over the system's components in key
    order; equal shares when the tuple does not match, flagged), and ``pca_00`` ... ``pca_15``: a 16-component PCA of
    the Morgan COUNT fingerprint (radius 2, 2048) of the primary extractant, fitted on the outer-training SYSTEMS --
    each training system enters once, not row-weighted, in sorted key order; numpy SVD of the centred count matrix,
    sign fixed so the largest-|loading| entry of each component is positive (equal to sklearn ``PCA(svd_solver="full")``
    up to that sign, tested); a fit with fewer than 17 systems pads the missing components with 0.  Categorical
    ``family`` (``system_family``), ``mechanism``.  Id ``system_id`` (free offsets ``delta_l`` / ``b'_s``).
``condition`` (:class:`ConditionBlock`, ``gen19ct.data.normalize.condition_vector``)
    numeric ``log10_acid_M, log10_extractant_M`` (primary), ``log10_metal_M, phase_ratio_org_aq`` (O/A, as recorded),
    ``temperature_C, contact_time_min, modifier_present, log10_modifier_M, complexant_present, log10_complexant_M``
    (log10 of the summed aqueous-complexant concentrations), ``acid_M_log10_grid`` (0/1); categorical ``acid_anion``,
    ``diluent_family``.
``flat_cat`` (:class:`FlatCatBlock`): categorical ``g19_metal_state``, ``extractant_system_key`` (FLAT_CAT).
``b8_metal`` (:class:`B8MetalBlock`): ``Z, lanthanide_index`` (``series_index`` of La..Lu, NaN otherwise),
    ``radius_cn8_A``.
``lig2d_ext`` (:class:`Lig2dExtBlock`): the 2,048-bit ECFP4 (Morgan r2 bit vector) of the primary extractant as a wide
    ``uint8`` matrix (dense, or scipy CSR) plus numeric ``n_components``.
``massaction`` (:class:`MassActionBlock`): ``log10_acid_M, log10_extractant_M, log10_metal_M`` and
    ``n0_x_log10_extractant_M`` with ``n0 = 3.0`` (the gen18 prior of section 5 B7).

Encodings (:class:`FeatureSet`, fitted on the rows passed to ``fit`` only)
--------------------------------------------------------------------------
``categorical="string"``  tokens as ``object`` columns listed in ``FeatureMatrix.categorical_columns`` (CatBoost
                          ``cat_features``; ``one_hot_max_size`` 10 one-hot-encodes the <= 10-level ones); a missing
                          value is ``<NA>``, a token absent from the training vocabulary is ``<UNSEEN>``.
``categorical="onehot"``  one 0/1 column per training token; an unseen token is all zeros.
``categorical="drop"``    categoricals omitted.
``missing="nan"``         NaN stays NaN (CatBoost, ExtraTrees).
``missing="impute"``      training median (row-weighted; 0.0 for a column all-NaN in training, recorded) and, with
                          ``missing_indicators=True``, a ``__missing`` 0/1 column for every column with a missing value
                          in the training rows (the sklearn ``SimpleImputer(add_indicator=True)`` rule).
``standardize=True``      training mean / SD (ddof 0, SD <= 0 -> 1) of every numeric output column after imputation.
Statistics are summed over sorted values, so a fit does not depend on the row order (digest-tested).

Guards
------
:func:`assert_feature_columns_allowed` raises for any provenance column, id column, target column or name token
(:data:`FORBIDDEN_SOURCE_COLUMNS`, :data:`FORBIDDEN_NAME_TOKENS`).  :class:`FeatureSet` checks every block's
``source_columns`` at construction and hands blocks a VIEW holding only those columns (plus the condition-vector
inputs), so a provenance or target column cannot physically reach a block; every output column name is checked again
at ``transform``.  ``transform`` never refits: it recomputes the fitted-state digest and raises if it changed.

Also here: :class:`EcfpClusters` (bit-identical primary-extractant fingerprint groups of the training systems, gen6
``levels.ecfp_cluster_labels`` labels, gen7 ``group_balanced_weights`` weights), TOPO39 (:func:`topo39_table`,
:func:`topo39_for_systems`, :func:`topo39_coverage`, :class:`Topo39Prep`) and :class:`DDescScaler` (section 3.5).
Readings where the registration is silent are listed in :data:`REGISTRATION_CHOICES`.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from gen19ct import paths
from gen19ct.chemistry import ligands as LG
from gen19ct.chemistry import metals as MET
from gen19ct.chemistry import support_graph as SG
from gen19ct.data import load as LOAD
from gen19ct.data import normalize as N

RDLogger.DisableLog("rdApp.*")

SCHEMA = "gen19.features.v1"
METALS_CSV = paths.DESCRIPTORS_DIR / "metals.csv"
SYSTEMS_CSV = paths.DESCRIPTORS_DIR / "extractant_systems.csv"
COMPONENTS_CSV = paths.DESCRIPTORS_DIR / "extractant_components.csv"
#: gen12.2's frozen coordination-topology table (v1.1.0); gen13 ``features.build_features(coordination="frozen")``
COORDINATION_PARQUET = paths.REPO_ROOT / "generations" / "gen12_2_eu_pred" / "features" / "coordination_descriptors.parquet"

NA_TOKEN = "<NA>"
UNSEEN_TOKEN = "<UNSEEN>"
#: embedding index of a value absent from the training vocabulary
RESERVED_INDEX = 0
N_PCA = 16
FP_RADIUS = 2
FP_BITS = 2048
#: the gen18 extractant-slope prior of section 5 B7 (``models.mass_action.N_PRIOR``), used by B8 MASSACTION
N0 = 3.0

METAL_NUMERIC: tuple[str, ...] = ("Z", "formal_charge", "radius_cn6_A", "radius_cn8_A", "radius_cn9_A",
                                  "f_electron_count", "d_electron_count", "electronegativity_pauling")
METAL_CATEGORICAL: tuple[str, ...] = ("category", "series", "hsab_class")
METAL_EXCLUDED: tuple[str, ...] = ("polarizability_atom_au",)
#: element-level columns an unknown-state row may carry (``chemistry.metals.ELEMENT_LEVEL_COLUMNS`` among the above)
METAL_ELEMENT_LEVEL: tuple[str, ...] = tuple(c for c in METAL_NUMERIC + METAL_CATEGORICAL + ("series_index",)
                                             if c in MET.ELEMENT_LEVEL_COLUMNS)
DONOR_COLUMNS: tuple[str, ...] = (
    "donor_amide_carbonyl_O", "donor_thioamide_S", "donor_ketone_O", "donor_ether_O", "donor_phosphoryl_O",
    "donor_P_OH_acidic_O", "donor_P_S", "donor_oxime_N", "donor_hydroxyl_O", "donor_aromatic_N_pyridine_type",
    "donor_triazine_N", "donor_amine_N_tertiary", "donor_amine_N_primary_secondary", "donor_ammonium_N_quaternary",
    "donor_thioether_S", "donor_thiophene_S", "donor_N_oxide_O", "donor_sulfonate_O", "donor_carboxylic_acid_groups",
)
DENTICITY_COLUMNS: tuple[str, ...] = ("n_donor_sites", "denticity_proxy")
SHARE_WEIGHTED: tuple[str, ...] = ("mw", "mol_logp", "tpsa", "rotatable_bonds")
D_DESC_COLUMNS: tuple[str, ...] = ("mw", "mol_logp", "rotatable_bonds")
PCA_COLUMNS: tuple[str, ...] = tuple(f"pca_{i:02d}" for i in range(N_PCA))
CONDITION_NUMERIC: tuple[str, ...] = ("log10_acid_M", "log10_extractant_M", "log10_metal_M", "phase_ratio_org_aq",
                                      "temperature_C", "contact_time_min", "modifier_present", "log10_modifier_M",
                                      "complexant_present", "log10_complexant_M", "acid_M_log10_grid")
CONDITION_CATEGORICAL: tuple[str, ...] = ("acid_anion", "diluent_family")
TOPO39_PREFIXES: tuple[str, ...] = ("coord__dist__", "coord__arm__")
SERIES_TOKENS = {"Ln": "Ln", "An": "An"}

METAL_STATE_COL = SG.METAL_COL
ELEMENT_COL = SG.ELEMENT_COL
SYSTEM_COL = SG.SYSTEM_COL

#: id / target columns that are never features, besides ``load.PROVENANCE_COLUMNS``
ID_COLUMNS: tuple[str, ...] = ("canonical_measurement_id", "row_id", "pub_group", "group_cross_publication_copy",
                               "group_near_duplicate_key", "group_compilation_doi", "g19_publication_id",
                               "g19_publication_status", "g19_publication_refs", "g19_study_id", "g19_bundle_exp_id",
                               "g19_tier", "exp_id", "source_record_id", "safe_exp_id", "record_id", "system_id",
                               "component_ids", "nn_row_id", "nn_canonical_measurement_id")
TARGET_COLUMNS: tuple[str, ...] = ("log_D", "D", "D_value", "logD", "log_D_raw", "D_raw", "mean_logD", "y", "target")
FORBIDDEN_SOURCE_COLUMNS: frozenset[str] = frozenset(LOAD.PROVENANCE_COLUMNS) | frozenset(ID_COLUMNS) | \
    frozenset(TARGET_COLUMNS)
#: case-insensitive substrings no feature name or source column may contain
FORBIDDEN_NAME_TOKENS: tuple[str, ...] = ("doi", "publication", "pub_group", "study_id", "measurement_id", "row_id",
                                          "record_id", "exp_id", "data_location", "comment", "author", "citation",
                                          "reference", "log_d", "source_file", "duplicate_group")

#: where the registration is silent, the reading implemented here (reported with every run that uses the features)
REGISTRATION_CHOICES: dict[str, str] = {
    "statistics_unit": "medians, means and SDs of scaling / imputation are row-weighted over the rows passed to fit "
                       "(section 2 'fitted on the training rows'); only the fingerprint PCA (section 5: 'fitted on "
                       "outer-training systems') and d_desc (section 3.5) are system-weighted",
    "pca_input": "the Morgan count fingerprint of the PRIMARY extractant of the system (the structure B3l, s4, B8 "
                 "LIG2D_EXT and TOPO39 use), not a sum over components",
    "pca_components_short": "fewer than 17 training systems: min(16, n_systems - 1) components, the rest 0",
    "donor_counts": "donor_* counts, n_donor_sites and denticity_proxy of the primary extractant; only mw, mol_logp, "
                    "tpsa and rotatable_bonds are concentration-share weighted (section 5 lists the weighting for those)",
    "share_weights": "shares = extractant_concentrations_sorted_M / its sum over the system's components in key order; "
                     "equal shares when the tuple length differs or a value is missing / <= 0 (flagged; no MODEL row); "
                     "a component with a missing descriptor makes the weighted value NaN",
    "unknown_state_metal": "X(?) rows carry element-level metal columns (Z, category, series, Pauling EN); ion-level "
                           "columns NaN, ox token <NA>; no state imputed",
    "condition_raw_scale": "O/A, temperature and contact time as recorded (section 5 logs only the concentrations); "
                           "contact_time_min only (shaking_time_min is not listed)",
    "complexant": "aqueous_complexant role only (holdback agents are not named by section 5); log10 of the summed "
                  "complexant concentrations; presence = complexant_structure_key present",
    "modifier": "presence = modifier_name present; log10_modifier_M NaN when absent or unrecorded",
    "categorical_native": "CatBoost receives category / series / hsab_class, family, mechanism, acid_anion, "
                          "diluent_family as categorical tokens (<NA> missing, <UNSEEN> outside the training "
                          "vocabulary); one_hot_max_size 10 one-hot-encodes the <= 10-level ones",
    "e_series": "Ln / An / other (orchestrator task F); brief section 15 lists lanthanide / actinide / transition / "
                "post-transition / other -- category_id carries the metals.csv category for that reading",
    "flat_cat_unknown_state": "FLAT_CAT's g19_metal_state is the <NA> token for X(?) rows (the column as recorded)",
    "b8_lanthanide_index": "metals.csv series_index for La..Lu (0..14), NaN for every other metal (actinides included)",
    "b8_massaction": "log10 acid, extractant (primary) and metal M plus n0 * log10[L] (n0 = 3.0); modifier and "
                     "complexant concentrations stay in COND only",
    "b8_fp_missing": "a system without a parseable primary structure gets an all-zero LIG2D_EXT row (flagged)",
    "b6_preset": "B6 / B6r0 use one-hot categoricals, training-median imputation WITHOUT missing indicators "
                 "(section 5: 'training medians imputed') and standardisation",
    "ecfp_cluster": "bit-identical 2048-bit fingerprints of the primary extractant share a label (sha1 of the int8 "
                    "bits, first 16 hex, gen6); a system without a fingerprint is its own group; weights 1 / (training "
                    "rows of the row's cluster) (gen7 group_balanced_weights)",
    "topo39_source": "the frozen gen12.2 table only (183 bundle extractants; section 5: 'TOPO39 exists only for "
                     "bundle extractants'), matched on the RDKit canonical primary-extractant SMILES; not recomputed",
    "system_static_source": "static system labels from descriptors/extractant_systems.csv; a system absent from it "
                            "(synthetic queries only) takes the row's system_family / mechanism / "
                            "primary_extractant_smiles and components parsed from the key "
                            "(ligands.describe_structure when a component is not in extractant_components.csv)",
}


# --------------------------------------------------------------------------------------------- #
# guards and digests
# --------------------------------------------------------------------------------------------- #

def forbidden_reason(column: str) -> str | None:
    """Why ``column`` may not be a feature or a feature source (None when it may)."""
    c = str(column)
    base = c.split("__", 1)[1] if "__" in c else c
    for cand in (c, base, base.split("=", 1)[0]):
        if cand in FORBIDDEN_SOURCE_COLUMNS:
            return f"{c!r} is a provenance / id / target column ({cand!r})"
    low = c.lower()
    for tok in FORBIDDEN_NAME_TOKENS:
        if tok in low:
            return f"{c!r} contains the forbidden token {tok!r}"
    return None


def assert_feature_columns_allowed(columns: Iterable[str], what: str = "feature columns") -> None:
    """Raise ``ValueError`` when any column is a provenance column, a publication / study / row id, a target or a
    name carrying a forbidden token (prereg section 2 'Features')."""
    bad = [r for r in (forbidden_reason(c) for c in columns) if r is not None]
    if bad:
        raise ValueError(f"{what}: forbidden columns would enter the features: {bad[:6]}")


def _canon(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {"__map__": [[str(k), _canon(v)] for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))]}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [_canon(v) for v in obj.tolist()]
        arr = np.ascontiguousarray(obj)
        return {"__nd__": str(arr.dtype), "shape": list(arr.shape), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return {"__f__": float(obj).hex()}
    if obj is None or isinstance(obj, str):
        return obj
    raise TypeError(f"state digest: unsupported type {type(obj)!r}")


def state_digest(state: Mapping[str, Any]) -> str:
    """SHA-256 of a fitted state (floats by exact hex, arrays by dtype / shape / bytes, maps by sorted key)."""
    text = json.dumps(_canon(state), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _missing(v: Any) -> bool:
    return SG._missing(v)


def _token(v: Any) -> str:
    return NA_TOKEN if _missing(v) else str(v)


def _sorted_mean(x: np.ndarray) -> float:
    x = np.sort(x[np.isfinite(x)])
    return float(np.mean(x)) if len(x) else float("nan")


def _sorted_std(x: np.ndarray) -> float:
    x = np.sort(x[np.isfinite(x)])
    if not len(x):
        return float("nan")
    m = float(np.mean(x))
    return float(np.sqrt(np.mean(np.sort((x - m) ** 2))))


# --------------------------------------------------------------------------------------------- #
# static chemistry (descriptor tables; never fitted, never targets)
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StaticTables:
    metals: pd.DataFrame               # indexed by metal_state_label
    metal_elements: pd.DataFrame       # element-level columns, one row per symbol
    systems: pd.DataFrame              # indexed by extractant_system_key
    components_by_id: pd.DataFrame
    components_by_smiles: pd.DataFrame
    digests: dict


@lru_cache(maxsize=1)
def static_tables() -> StaticTables:
    """The three descriptor tables (cached) and their LF-normalised digests."""
    m = pd.read_csv(METALS_CSV)
    states = m[m["metal_state_label"].notna()].set_index("metal_state_label", drop=False)
    elem = m.sort_values(["symbol", "oxidation_state"], na_position="first").drop_duplicates("symbol")
    elem = elem.set_index("symbol", drop=False)[[c for c in MET.ELEMENT_LEVEL_COLUMNS if c in m.columns]]
    s = pd.read_csv(SYSTEMS_CSV).set_index("extractant_system_key", drop=False)
    c = pd.read_csv(COMPONENTS_CSV)
    missing = [k for k in DONOR_COLUMNS + DENTICITY_COLUMNS + SHARE_WEIGHTED if k not in c.columns]
    if missing:
        raise KeyError(f"extractant_components.csv lacks {missing}")
    by_id = c.set_index("component_id", drop=False)
    st = c[(c["record_type"] == "STRUCTURE") & c["smiles_canonical"].notna()]
    by_smi = st.set_index("smiles_canonical", drop=False)
    dig = {p.name: paths.digests(p)["sha256_lf"] for p in (METALS_CSV, SYSTEMS_CSV, COMPONENTS_CSV)}
    return StaticTables(states, elem, s, by_id, by_smi, dig)


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


@lru_cache(maxsize=4096)
def metal_record(state: str | None, element: str | None) -> dict[str, Any]:
    """Metal descriptors of one row: ``METAL_NUMERIC`` / ``METAL_CATEGORICAL`` / ``series_index`` values, plus the
    M1 tokens ``series_token`` (Ln / An / other) and ``ox_token``.  An unknown state keeps only element-level values."""
    tab = static_tables()
    out: dict[str, Any] = {c: float("nan") for c in METAL_NUMERIC + ("series_index",)}
    out.update({c: None for c in METAL_CATEGORICAL})
    out.update(state_token=_token(state), element_token=_token(element), ox_token=NA_TOKEN, source="none")
    if not _missing(state):
        label = str(state)
        sym, ox, _ = MET.normalize_metal(label)
        out["element_token"] = sym or out["element_token"]
        out["ox_token"] = NA_TOKEN if ox is None else str(int(ox))
        if label in tab.metals.index:
            row = tab.metals.loc[label]
            for c in METAL_NUMERIC + ("series_index",):
                out[c] = _num(row[c])
            for c in METAL_CATEGORICAL:
                out[c] = None if _missing(row[c]) else str(row[c])
            out["source"] = "metals.csv:state"
        else:                                   # a state absent from metals.csv (synthetic): the support-graph table
            p = SG.metal_properties(label)
            out.update(Z=_num(p["Z"]), formal_charge=_num(p["charge"]), radius_cn8_A=_num(p["r_cn8"]),
                       radius_cn6_A=_num(p["r_cn6"]), category=p["category"], series=p["series"],
                       source="support_graph.metal_properties")
    elif not _missing(element):
        sym = str(element)
        if sym in tab.metal_elements.index:
            row = tab.metal_elements.loc[sym]
            for c in METAL_ELEMENT_LEVEL:
                if c in METAL_CATEGORICAL:
                    out[c] = None if _missing(row[c]) else str(row[c])
                else:
                    out[c] = _num(row[c])
            out["source"] = "metals.csv:element_level"
        else:
            p = SG.metal_properties(f"{sym}(?)")
            out.update(Z=_num(p["Z"]), category=p["category"], series=p["series"], source="support_graph:element")
    ser = out["series"]
    out["series_token"] = SERIES_TOKENS.get(ser, "other") if ser is not None else NA_TOKEN
    return out


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


@lru_cache(maxsize=4096)
def _fingerprints(smiles: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if _missing(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    bits = np.asarray(_MORGAN.GetFingerprintAsNumPy(mol), dtype=np.uint8)
    counts = np.asarray(_MORGAN.GetCountFingerprintAsNumPy(mol), dtype=np.float64)
    bits.setflags(write=False)
    counts.setflags(write=False)
    return bits, counts


def ecfp_bits(smiles: str | None) -> np.ndarray | None:
    """2,048-bit Morgan r2 (ECFP4) bit vector as ``uint8`` (None when RDKit cannot parse the SMILES)."""
    fp = _fingerprints(smiles)
    return None if fp is None else fp[0]


def morgan_counts(smiles: str | None) -> np.ndarray | None:
    """Morgan r2 count fingerprint folded to 2,048 bins (``float64``; None when unparseable)."""
    fp = _fingerprints(smiles)
    return None if fp is None else fp[1]


@lru_cache(maxsize=1024)
def _component_from_smiles(smiles: str) -> dict[str, float]:
    tab = static_tables()
    cols = DONOR_COLUMNS + DENTICITY_COLUMNS + SHARE_WEIGHTED
    if smiles in tab.components_by_smiles.index:
        row = tab.components_by_smiles.loc[smiles]
        return {c: _num(row[c]) for c in cols}
    rec = LG.describe_structure(smiles) if not smiles.startswith("name:") else {"parse_ok": False}
    return {c: (_num(rec.get(c)) if rec.get("parse_ok") else float("nan")) for c in cols}


@dataclass(frozen=True)
class SystemRecord:
    key: str
    in_table: bool
    family: str | None
    mechanism: str | None
    n_components: float
    primary_smiles: str | None
    component_smiles: tuple[str, ...]
    component_desc: np.ndarray          # n_components x len(SHARE_WEIGHTED)
    primary_donors: np.ndarray          # len(DONOR_COLUMNS + DENTICITY_COLUMNS)


def system_record(key: str, fallback: Mapping[str, Any] | None = None) -> SystemRecord:
    """Static chemistry of one extractant system (section 5 extractant block inputs)."""
    return _system_record(key, tuple(sorted((fallback or {}).items())))


@lru_cache(maxsize=2048)
def _system_record(key: str, fallback_items: tuple) -> SystemRecord:
    tab = static_tables()
    fb = dict(fallback_items)
    if key in tab.systems.index:
        r = tab.systems.loc[key]
        ids = [] if _missing(r["component_ids"]) else str(r["component_ids"]).split("|")
        comp_smiles = []
        for cid in ids:
            c = tab.components_by_id.loc[cid] if cid in tab.components_by_id.index else None
            smi = None if c is None or _missing(c["smiles_canonical"]) else str(c["smiles_canonical"])
            comp_smiles.append(smi if smi is not None else f"name:{cid}")
        family = None if _missing(r["system_family"]) else str(r["system_family"])
        mech = None if _missing(r["mechanism"]) else str(r["mechanism"])
        n_comp = _num(r["n_organic_extractants"])
        primary = None if _missing(r["primary_extractant_smiles"]) else str(r["primary_extractant_smiles"])
        in_table = True
    else:
        comp_smiles = [p for p in str(key).split("|") if p]
        family = None if _missing(fb.get("system_family")) else str(fb["system_family"])
        mech = None if _missing(fb.get("mechanism")) else str(fb["mechanism"])
        n_comp = float(len(comp_smiles))
        primary = fb.get("primary_extractant_smiles")
        primary = (comp_smiles[0] if comp_smiles else None) if _missing(primary) else str(primary)
        in_table = False
    desc = np.array([[_component_from_smiles(s)[c] for c in SHARE_WEIGHTED] for s in comp_smiles], dtype=float) \
        if comp_smiles else np.zeros((0, len(SHARE_WEIGHTED)))
    if primary is not None:
        pd_ = _component_from_smiles(primary)
        donors = np.array([pd_[c] for c in DONOR_COLUMNS + DENTICITY_COLUMNS], dtype=float)
    else:
        donors = np.full(len(DONOR_COLUMNS + DENTICITY_COLUMNS), np.nan)
    desc.setflags(write=False)
    donors.setflags(write=False)
    return SystemRecord(str(key), in_table, family, mech, n_comp, primary, tuple(comp_smiles), desc, donors)


def _row_fallbacks(view: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """``{system: {system_family, mechanism, primary_extractant_smiles}}`` from the rows, for systems absent from the
    descriptor table only."""
    tab = static_tables()
    cols = [c for c in ("system_family", "mechanism", "primary_extractant_smiles") if c in view.columns]
    out: dict[str, dict[str, Any]] = {}
    if not cols:
        return out
    keys = view[SYSTEM_COL].to_numpy(dtype=object)
    absent = np.array([not _missing(k) and str(k) not in tab.systems.index for k in keys], dtype=bool)
    if not absent.any():
        return out
    sub = view.loc[absent, [SYSTEM_COL] + cols]
    for k, grp in sub.groupby(SYSTEM_COL, sort=True):
        rec = {}
        for c in cols:
            vals = [v for v in grp[c].to_numpy(dtype=object) if not _missing(v)]
            rec[c] = str(vals[0]) if vals else None
        out[str(k)] = rec
    return out


# --------------------------------------------------------------------------------------------- #
# fitted helpers
# --------------------------------------------------------------------------------------------- #

class Vocabulary:
    """Sorted training tokens; ``index`` maps a token to ``1 + position`` and anything unseen to
    :data:`RESERVED_INDEX` (0)."""

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self._pos: dict[str, int] = {}

    def fit(self, tokens: Iterable[str]) -> "Vocabulary":
        self.tokens = sorted({str(t) for t in tokens})
        self._pos = {t: i + 1 for i, t in enumerate(self.tokens)}
        return self

    def index(self, tokens: Iterable[str]) -> np.ndarray:
        return np.array([self._pos.get(str(t), RESERVED_INDEX) for t in tokens], dtype=np.int64)

    def encode(self, tokens: Iterable[str]) -> np.ndarray:
        return np.array([str(t) if str(t) in self._pos else UNSEEN_TOKEN for t in tokens], dtype=object)

    @property
    def size(self) -> int:
        """Embedding table size including the reserved index."""
        return len(self.tokens) + 1

    def state(self) -> list[str]:
        return list(self.tokens)


@dataclass
class BlockRaw:
    numeric: pd.DataFrame
    categorical: pd.DataFrame
    ids: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: pd.DataFrame | None = None
    wide: dict[str, tuple[Any, tuple[str, ...]]] = field(default_factory=dict)


class Block:
    """A feature block: ``fit(view, cv)`` fits block-specific state on training rows, ``raw(view, cv)`` returns raw
    columns for any rows (never refitting).  ``source_columns`` are the only row columns the block reads."""

    name: str = "block"
    source_columns: tuple[str, ...] = ()
    needs_condition_vector: bool = False

    def fit(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> None:
        return None

    def raw(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> BlockRaw:
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        return {}


class MetalBlock(Block):
    """Section 5 B5 metal block and the M1 metal ids (module docstring)."""

    name = "metal"
    source_columns = (METAL_STATE_COL, ELEMENT_COL)
    ID_NAMES = ("metal_state_id", "series_id", "ox_id", "element_id", "category_id")

    def __init__(self) -> None:
        self.vocab: dict[str, Vocabulary] = {}

    def _records(self, view: pd.DataFrame) -> list[dict[str, Any]]:
        st = view[METAL_STATE_COL].to_numpy(dtype=object)
        el = view[ELEMENT_COL].to_numpy(dtype=object)
        return [metal_record(None if _missing(a) else str(a), None if _missing(b) else str(b)) for a, b in zip(st, el)]

    @staticmethod
    def _id_tokens(recs: list[dict[str, Any]]) -> dict[str, list[str]]:
        return {"metal_state_id": [r["state_token"] for r in recs], "series_id": [r["series_token"] for r in recs],
                "ox_id": [r["ox_token"] for r in recs], "element_id": [r["element_token"] for r in recs],
                "category_id": [_token(r["category"]) for r in recs]}

    def fit(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> None:
        toks = self._id_tokens(self._records(view))
        self.vocab = {k: Vocabulary().fit(v) for k, v in toks.items()}

    def raw(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> BlockRaw:
        recs = self._records(view)
        num = pd.DataFrame({c: np.array([r[c] for r in recs], dtype=float) for c in METAL_NUMERIC}, index=view.index)
        cat = pd.DataFrame({c: np.array([_token(r[c]) for r in recs], dtype=object) for c in METAL_CATEGORICAL},
                           index=view.index)
        toks = self._id_tokens(recs)
        ids = {k: self.vocab[k].index(v) for k, v in toks.items()}
        diag = pd.DataFrame({"metal_state_seen": ids["metal_state_id"] != RESERVED_INDEX,
                             "metal_descriptor_source": np.array([r["source"] for r in recs], dtype=object)},
                            index=view.index)
        return BlockRaw(num, cat, ids, diag)

    def state(self) -> dict[str, Any]:
        return {"vocab": {k: v.state() for k, v in self.vocab.items()}}


class ExtractantBlock(Block):
    """Section 5 B5 extractant block and the system vocabulary for free offsets (module docstring)."""

    name = "extractant"
    source_columns = (SYSTEM_COL, "system_family", "mechanism", "primary_extractant_smiles")
    needs_condition_vector = True

    def __init__(self, n_components: int = N_PCA) -> None:
        self.n_components = int(n_components)
        self.system_vocab = Vocabulary()
        self.pca_mean: np.ndarray | None = None
        self.pca_components: np.ndarray | None = None
        self.pca_explained_variance: np.ndarray | None = None
        self.pca_n_fitted = 0
        self.pca_systems: list[str] = []

    @staticmethod
    def _keys(view: pd.DataFrame) -> np.ndarray:
        return np.array([_token(k) for k in view[SYSTEM_COL].to_numpy(dtype=object)], dtype=object)

    def _records(self, view: pd.DataFrame) -> dict[str, SystemRecord]:
        fb = _row_fallbacks(view)
        return {k: system_record(k, fb.get(k)) for k in sorted(set(self._keys(view))) if k != NA_TOKEN}

    def fit(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> None:
        keys = self._keys(view)
        self.system_vocab = Vocabulary().fit(keys)
        recs = self._records(view)
        systems, rows = [], []
        for k in sorted(recs):                                  # each training system once, sorted by key
            c = morgan_counts(recs[k].primary_smiles)
            if c is not None:
                systems.append(k)
                rows.append(c)
        self.pca_systems = systems
        d = FP_BITS
        comps = np.zeros((self.n_components, d))
        var = np.zeros(self.n_components)
        mean = np.zeros(d)
        k_fit = 0
        if systems:
            X = np.vstack(rows)
            mean = X.mean(axis=0)
            k_fit = min(self.n_components, max(len(systems) - 1, 0))
            if k_fit:
                _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
                vt = vt[:k_fit].copy()
                for i in range(k_fit):                          # deterministic sign: largest |loading| positive
                    j = int(np.argmax(np.abs(vt[i])))
                    if vt[i, j] < 0:
                        vt[i] = -vt[i]
                comps[:k_fit] = vt
                var[:k_fit] = (s[:k_fit] ** 2) / max(len(systems) - 1, 1)
        self.pca_mean, self.pca_components, self.pca_explained_variance, self.pca_n_fitted = mean, comps, var, k_fit

    def pca_scores(self, smiles: Sequence[str | None]) -> np.ndarray:
        """16 PCA scores per structure (NaN rows for an unparseable / missing structure); never refits."""
        if self.pca_components is None:
            raise RuntimeError("ExtractantBlock: fit first")
        out = np.full((len(smiles), self.n_components), np.nan)
        for i, s in enumerate(smiles):
            c = morgan_counts(s)
            if c is not None:
                out[i] = (c - self.pca_mean) @ self.pca_components.T
                out[i, self.pca_n_fitted:] = 0.0
        return out

    def raw(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> BlockRaw:
        if cv is None:
            raise ValueError("ExtractantBlock needs the condition vector (concentration shares)")
        keys = self._keys(view)
        recs = self._records(view)
        n = len(view)
        concs = cv["extractant_concentrations_sorted_M"].to_numpy(dtype=object)
        weighted = np.full((n, len(SHARE_WEIGHTED)), np.nan)
        donors = np.full((n, len(DONOR_COLUMNS + DENTICITY_COLUMNS)), np.nan)
        n_comp = np.full(n, np.nan)
        fallback = np.zeros(n, dtype=bool)
        cache: dict[tuple, tuple[np.ndarray, bool]] = {}
        sys_list = sorted(recs)
        pca_by_sys = dict(zip(sys_list, self.pca_scores([recs[k].primary_smiles for k in sys_list])))
        pca = np.full((n, self.n_components), np.nan)
        for i in range(n):
            k = keys[i]
            if k == NA_TOKEN:
                continue
            rec = recs[k]
            n_comp[i] = rec.n_components
            donors[i] = rec.primary_donors
            pca[i] = pca_by_sys[k]
            t = tuple(float(x) for x in concs[i]) if isinstance(concs[i], (tuple, list, np.ndarray)) else ()
            ck = (k, t)
            hit = cache.get(ck)
            if hit is None:
                m = len(rec.component_smiles)
                arr = np.asarray(t, dtype=float)
                ok = m > 0 and len(arr) == m and np.isfinite(arr).all() and (arr > 0).all()
                w = arr / arr.sum() if ok else (np.full(m, 1.0 / m) if m else np.zeros(0))
                val = (w[:, None] * rec.component_desc).sum(axis=0) if m else np.full(len(SHARE_WEIGHTED), np.nan)
                if m and not np.isfinite(rec.component_desc).all():
                    val = np.full(len(SHARE_WEIGHTED), np.nan)
                hit = (val, not ok)
                cache[ck] = hit
            weighted[i], fallback[i] = hit
        data = {"n_components": n_comp}
        data.update({c: donors[:, j] for j, c in enumerate(DONOR_COLUMNS + DENTICITY_COLUMNS)})
        data.update({f"w_{c}": weighted[:, j] for j, c in enumerate(SHARE_WEIGHTED)})
        data.update({PCA_COLUMNS[j]: pca[:, j] for j in range(self.n_components)})
        num = pd.DataFrame(data, index=view.index)
        cat = pd.DataFrame({
            "family": np.array([_token(recs[k].family) if k != NA_TOKEN else NA_TOKEN for k in keys], dtype=object),
            "mechanism": np.array([_token(recs[k].mechanism) if k != NA_TOKEN else NA_TOKEN for k in keys], dtype=object),
        }, index=view.index)
        ids = {"system_id": self.system_vocab.index(keys)}
        fp_missing = np.array([k == NA_TOKEN or morgan_counts(recs[k].primary_smiles) is None for k in keys], dtype=bool)
        diag = pd.DataFrame({"system_seen": ids["system_id"] != RESERVED_INDEX, "share_fallback": fallback,
                             "primary_fingerprint_missing": fp_missing,
                             "system_in_descriptor_table": np.array([k != NA_TOKEN and recs[k].in_table for k in keys])},
                            index=view.index)
        return BlockRaw(num, cat, ids, diag)

    def state(self) -> dict[str, Any]:
        return {"system_vocab": self.system_vocab.state(), "pca_systems": list(self.pca_systems),
                "pca_mean": self.pca_mean, "pca_components": self.pca_components,
                "pca_explained_variance": self.pca_explained_variance, "pca_n_fitted": self.pca_n_fitted,
                "n_components": self.n_components}


def _log10_pos(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x > 0)
    out[ok] = np.log10(x[ok])
    return out


class ConditionBlock(Block):
    """Section 5 B5 condition block (module docstring); no fitted state of its own."""

    name = "condition"
    source_columns = ()
    needs_condition_vector = True

    def raw(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> BlockRaw:
        if cv is None:
            raise ValueError("ConditionBlock needs the condition vector")
        cx = cv["complexant_concentrations_sorted_M"].to_numpy(dtype=object)
        cx_sum = np.array([float(np.sum(t)) if isinstance(t, (tuple, list, np.ndarray)) and len(t)
                           and np.isfinite(np.asarray(t, dtype=float)).all() else np.nan for t in cx], dtype=float)
        num = pd.DataFrame({
            "log10_acid_M": cv["log10_acid_M"].to_numpy(dtype=float),
            "log10_extractant_M": cv["log10_extractant_primary_M"].to_numpy(dtype=float),
            "log10_metal_M": cv["log10_metal_M"].to_numpy(dtype=float),
            "phase_ratio_org_aq": cv["phase_ratio_org_aq"].to_numpy(dtype=float),
            "temperature_C": cv["temperature_C"].to_numpy(dtype=float),
            "contact_time_min": cv["contact_time_min"].to_numpy(dtype=float),
            "modifier_present": np.array([0.0 if _missing(v) else 1.0 for v in cv["modifier_name"].to_numpy(dtype=object)]),
            "log10_modifier_M": _log10_pos(cv["modifier_concentration_M"].to_numpy(dtype=float)),
            "complexant_present": np.array([0.0 if _missing(v) else 1.0
                                            for v in cv["complexant_structure_key"].to_numpy(dtype=object)]),
            "log10_complexant_M": _log10_pos(cx_sum),
            "acid_M_log10_grid": cv["acid_M_log10_grid"].to_numpy(dtype=bool).astype(float),
        }, index=view.index)
        for c in ("log10_acid_M", "log10_extractant_M", "log10_metal_M", "phase_ratio_org_aq", "temperature_C",
                  "contact_time_min"):
            v = num[c].to_numpy(dtype=float, copy=True)
            v[~np.isfinite(v)] = np.nan
            num[c] = v
        cat = pd.DataFrame({c: np.array([_token(v) for v in cv[c].to_numpy(dtype=object)], dtype=object)
                            for c in CONDITION_CATEGORICAL}, index=view.index)
        return BlockRaw(num, cat)


class FlatCatBlock(Block):
    """FLAT_CAT's categorical inputs (section 5): ``g19_metal_state`` and ``extractant_system_key``."""

    name = "flat_cat"
    source_columns = (METAL_STATE_COL, SYSTEM_COL)

    def raw(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> BlockRaw:
        cat = pd.DataFrame({c: np.array([_token(v) for v in view[c].to_numpy(dtype=object)], dtype=object)
                            for c in (METAL_STATE_COL, SYSTEM_COL)}, index=view.index)
        return BlockRaw(pd.DataFrame(index=view.index), cat)


class B8MetalBlock(Block):
    """B8 METAL: ``Z``, lanthanide index (La..Lu = 0..14, NaN otherwise), CN8 radius."""

    name = "b8_metal"
    source_columns = (METAL_STATE_COL, ELEMENT_COL)

    def raw(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> BlockRaw:
        st = view[METAL_STATE_COL].to_numpy(dtype=object)
        el = view[ELEMENT_COL].to_numpy(dtype=object)
        recs = [metal_record(None if _missing(a) else str(a), None if _missing(b) else str(b)) for a, b in zip(st, el)]
        num = pd.DataFrame({
            "Z": np.array([r["Z"] for r in recs], dtype=float),
            "lanthanide_index": np.array([r["series_index"] if r["series_token"] == "Ln" else np.nan for r in recs],
                                         dtype=float),
            "radius_cn8_A": np.array([r["radius_cn8_A"] for r in recs], dtype=float),
        }, index=view.index)
        return BlockRaw(num, pd.DataFrame(index=view.index))


class MassActionBlock(Block):
    """B8 MASSACTION: log10 concentrations and ``n0 * log10[L]`` (n0 = 3.0)."""

    name = "massaction"
    source_columns = ()
    needs_condition_vector = True

    def raw(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> BlockRaw:
        if cv is None:
            raise ValueError("MassActionBlock needs the condition vector")
        la = cv["log10_acid_M"].to_numpy(dtype=float)
        le = cv["log10_extractant_primary_M"].to_numpy(dtype=float)
        lm = cv["log10_metal_M"].to_numpy(dtype=float)
        num = pd.DataFrame({"log10_acid_M": la, "log10_extractant_M": le, "log10_metal_M": lm,
                            "n0_x_log10_extractant_M": N0 * le}, index=view.index)
        return BlockRaw(num, pd.DataFrame(index=view.index))


class Lig2dExtBlock(Block):
    """B8 LIG2D_EXT: 2,048-bit ECFP4 of the primary extractant (wide ``uint8``) plus the component count."""

    name = "lig2d_ext"
    source_columns = (SYSTEM_COL, "primary_extractant_smiles")

    def __init__(self, fmt: str = "dense") -> None:
        if fmt not in ("dense", "sparse"):
            raise ValueError(f"ecfp format {fmt!r}")
        self.fmt = fmt

    def raw(self, view: pd.DataFrame, cv: pd.DataFrame | None) -> BlockRaw:
        keys = np.array([_token(k) for k in view[SYSTEM_COL].to_numpy(dtype=object)], dtype=object)
        fb = _row_fallbacks(view)
        uniq = sorted(set(keys) - {NA_TOKEN})
        recs = {k: system_record(k, fb.get(k)) for k in uniq}
        code = {k: i for i, k in enumerate(uniq)}
        U = np.zeros((len(uniq), FP_BITS), dtype=np.uint8)
        missing_u = np.ones(len(uniq), dtype=bool)
        for k, i in code.items():
            b = ecfp_bits(recs[k].primary_smiles)
            if b is not None:
                U[i] = b
                missing_u[i] = False
        pos = np.array([code.get(k, -1) for k in keys], dtype=np.int64)
        cols = tuple(f"ecfp_{j:04d}" for j in range(FP_BITS))
        if self.fmt == "dense":
            bits = np.zeros((len(view), FP_BITS), dtype=np.uint8)
            ok = pos >= 0
            bits[ok] = U[pos[ok]]
        else:
            from scipy import sparse
            if not len(uniq):
                bits = sparse.csr_matrix((len(view), FP_BITS), dtype=np.uint8)
            else:
                bits = sparse.csr_matrix(U)[np.where(pos >= 0, pos, 0)]
                if (pos < 0).any():
                    bits = (sparse.diags((pos >= 0).astype(np.uint8)) @ bits).astype(np.uint8).tocsr()
                    bits.eliminate_zeros()
        num = pd.DataFrame({"n_components": np.array([recs[k].n_components if k != NA_TOKEN else np.nan for k in keys],
                                                     dtype=float)}, index=view.index)
        fp_missing = np.ones(len(view), dtype=bool)
        if len(uniq):
            fp_missing = np.where(pos >= 0, missing_u[np.maximum(pos, 0)], True)
        diag = pd.DataFrame({"lig2d_fingerprint_missing": fp_missing}, index=view.index)
        return BlockRaw(num, pd.DataFrame(index=view.index), {}, diag, {"ecfp": (bits, cols)})

    def state(self) -> dict[str, Any]:
        return {"format": self.fmt}


BLOCK_TYPES: dict[str, type[Block]] = {
    "metal": MetalBlock, "extractant": ExtractantBlock, "condition": ConditionBlock, "flat_cat": FlatCatBlock,
    "b8_metal": B8MetalBlock, "massaction": MassActionBlock, "lig2d_ext": Lig2dExtBlock,
}


class _Prep:
    """Generic, fitted encoding of one block's raw output (module docstring, "Encodings")."""

    def __init__(self, block: str, categorical: str, missing: str, missing_indicators: bool, standardize: bool):
        self.block, self.categorical, self.missing = block, categorical, missing
        self.missing_indicators, self.standardize = bool(missing_indicators), bool(standardize)
        self.vocab: dict[str, Vocabulary] = {}
        self.medians: dict[str, float] = {}
        self.all_nan_in_training: list[str] = []
        self.indicator_for: list[str] = []
        self.mean: dict[str, float] = {}
        self.sd: dict[str, float] = {}
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []

    def _name(self, col: str) -> str:
        return f"{self.block}__{col}"

    def _encode(self, raw: BlockRaw) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Columns in order: each numeric column (followed by its ``__missing`` indicator), then the categoricals."""
        parts: dict[str, np.ndarray] = {}
        indicators: list[str] = []
        for c in self.numeric_cols:
            v = raw.numeric[c].to_numpy(dtype=float).copy()
            v[~np.isfinite(v)] = np.nan
            if self.missing == "impute":
                miss = np.isnan(v)
                v[miss] = self.medians[c]
                parts[self._name(c)] = v
                if self.missing_indicators and c in self.indicator_for:
                    parts[self._name(c) + "__missing"] = miss.astype(float)
                    indicators.append(self._name(c) + "__missing")
            else:
                parts[self._name(c)] = v
        cats: list[str] = []
        for c in self.categorical_cols:
            toks = raw.categorical[c].to_numpy(dtype=object)
            voc = self.vocab[c]
            if self.categorical == "string":
                parts[self._name(c)] = voc.encode(toks)
                cats.append(self._name(c))
            else:                                               # onehot ("drop" has no categorical_cols)
                ii = voc.index(toks)
                for j, t in enumerate(voc.tokens):
                    parts[f"{self._name(c)}={t}"] = (ii == j + 1).astype(float)
        frame = pd.DataFrame({k: (pd.Series(v, index=raw.numeric.index, dtype=object) if k in cats else v)
                              for k, v in parts.items()}, index=raw.numeric.index)
        for c in cats:                                      # pandas 3 would infer the str dtype; CatBoost gets object
            if frame[c].dtype != object:
                frame[c] = frame[c].astype(object)
        return frame, cats, indicators

    def fit(self, raw: BlockRaw) -> "_Prep":
        self.numeric_cols = list(raw.numeric.columns)
        self.categorical_cols = list(raw.categorical.columns) if self.categorical != "drop" else []
        self.vocab = {c: Vocabulary().fit(raw.categorical[c].to_numpy(dtype=object)) for c in self.categorical_cols}
        self.medians, self.all_nan_in_training, self.indicator_for = {}, [], []
        for c in self.numeric_cols:
            v = raw.numeric[c].to_numpy(dtype=float)
            fin = v[np.isfinite(v)]
            if len(fin) < len(v):
                self.indicator_for.append(c)
            if len(fin):
                self.medians[c] = float(np.median(np.sort(fin)))
            else:
                self.medians[c] = 0.0
                self.all_nan_in_training.append(c)
        self.mean, self.sd = {}, {}
        if self.standardize:
            frame, cats, _ = self._encode(raw)
            for c in frame.columns:
                if c in cats:
                    continue
                v = frame[c].to_numpy(dtype=float)
                m, s = _sorted_mean(v), _sorted_std(v)
                self.mean[c] = 0.0 if not np.isfinite(m) else m
                self.sd[c] = 1.0 if (not np.isfinite(s) or s <= 0) else s
        return self

    def transform(self, raw: BlockRaw) -> tuple[pd.DataFrame, list[str], list[str]]:
        missing = [c for c in self.numeric_cols if c not in raw.numeric.columns] + \
                  [c for c in self.categorical_cols if c not in raw.categorical.columns]
        if missing:
            raise KeyError(f"{self.block}: raw columns missing at transform {missing}")
        frame, cats, indicators = self._encode(raw)
        if self.standardize:
            for c in frame.columns:
                if c in self.mean:
                    frame[c] = (frame[c].to_numpy(dtype=float) - self.mean[c]) / self.sd[c]
        return frame, cats, indicators

    def state(self) -> dict[str, Any]:
        return {"numeric_cols": self.numeric_cols, "categorical_cols": self.categorical_cols,
                "vocab": {k: v.state() for k, v in self.vocab.items()}, "medians": self.medians,
                "all_nan_in_training": self.all_nan_in_training, "indicator_for": self.indicator_for,
                "mean": self.mean, "sd": self.sd,
                "config": [self.categorical, self.missing, self.missing_indicators, self.standardize]}


# --------------------------------------------------------------------------------------------- #
# the facade
# --------------------------------------------------------------------------------------------- #

@dataclass
class FeatureMatrix:
    """What :meth:`FeatureSet.transform` returns.  ``frame`` holds every tabular feature (``float64``; ``object``
    tokens for ``categorical_columns``); ``wide`` holds the LIG2D_EXT bit block (``uint8`` dense or CSR); ``ids`` the
    embedding indices (0 = unseen); ``diagnostics`` per-row flags that are never features."""

    frame: pd.DataFrame
    categorical_columns: tuple[str, ...]
    indicator_columns: tuple[str, ...]
    block_columns: dict[str, tuple[str, ...]]
    wide: dict[str, tuple[Any, tuple[str, ...]]]
    ids: dict[str, np.ndarray]
    diagnostics: pd.DataFrame
    state_digest: str

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.frame.columns)

    def cat_feature_indices(self) -> list[int]:
        cols = list(self.frame.columns)
        return [cols.index(c) for c in self.categorical_columns]

    def wide_columns(self) -> tuple[str, ...]:
        return tuple(c for _, (_, cols) in sorted(self.wide.items()) for c in cols)

    def numeric_array(self, dtype: Any = np.float32, include_wide: bool = True) -> np.ndarray:
        """Dense ``frame`` (+ wide blocks) as one numeric array; refuses when categorical tokens are present."""
        if self.categorical_columns:
            raise TypeError("numeric_array: categorical token columns present (use categorical='onehot' or 'drop')")
        parts = [self.frame.to_numpy(dtype=dtype)]
        if include_wide:
            for _, (mat, _) in sorted(self.wide.items()):
                parts.append(mat.toarray().astype(dtype) if hasattr(mat, "toarray") else np.asarray(mat, dtype=dtype))
        return np.hstack(parts) if len(parts) > 1 else parts[0]


#: registered arm -> FeatureSet configuration (module docstring; REGISTRATION_CHOICES["b6_preset"])
ARM_PRESETS: dict[str, dict[str, Any]] = {
    "B5": dict(blocks=("metal", "extractant", "condition"), categorical="string", missing="nan"),
    "M0": dict(blocks=("metal", "extractant", "condition"), categorical="string", missing="nan"),
    "FLAT_CAT": dict(blocks=("flat_cat", "condition"), categorical="string", missing="nan"),
    "B6": dict(blocks=("metal", "extractant", "condition"), categorical="onehot", missing="impute",
               missing_indicators=False, standardize=True),
    "B6r0": dict(blocks=("metal", "extractant", "condition"), categorical="onehot", missing="impute",
                 missing_indicators=False, standardize=True),
    "B8": dict(blocks=("b8_metal", "condition", "lig2d_ext", "massaction"), categorical="onehot", missing="nan",
               ecfp_format="dense"),
    "M1": dict(blocks=("metal", "extractant", "condition"), categorical="onehot", missing="impute",
               missing_indicators=True, standardize=True),
}


class FeatureSet:
    """Fit on training rows, transform any rows, never refit on transform (module docstring)."""

    def __init__(self, blocks: Sequence[str | Block] = ("metal", "extractant", "condition"), *,
                 categorical: str = "string", missing: str = "nan", missing_indicators: bool = True,
                 standardize: bool = False, ecfp_format: str = "dense", name: str | None = None):
        if categorical not in ("string", "onehot", "drop"):
            raise ValueError(f"categorical {categorical!r}")
        if missing not in ("nan", "impute"):
            raise ValueError(f"missing {missing!r}")
        if ecfp_format not in ("dense", "sparse"):
            raise ValueError(f"ecfp_format {ecfp_format!r}")
        self.config = {"categorical": categorical, "missing": missing, "missing_indicators": bool(missing_indicators),
                       "standardize": bool(standardize), "ecfp_format": ecfp_format}
        self.name = name
        self.blocks: list[Block] = []
        for b in blocks:
            if isinstance(b, Block):
                self.blocks.append(b)
            elif b == "lig2d_ext":
                self.blocks.append(Lig2dExtBlock(ecfp_format))
            elif b in BLOCK_TYPES:
                self.blocks.append(BLOCK_TYPES[b]())
            else:
                raise KeyError(f"unknown block {b!r}; have {sorted(BLOCK_TYPES)}")
        names = [b.name for b in self.blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate blocks {names}")
        assert_feature_columns_allowed(self.source_columns, "block source columns")
        self._preps: dict[str, _Prep] = {}
        self._digest: str | None = None
        self.n_train_rows = 0

    @classmethod
    def for_arm(cls, arm: str) -> "FeatureSet":
        if arm not in ARM_PRESETS:
            raise KeyError(f"no feature preset for {arm!r}; have {sorted(ARM_PRESETS)}")
        return cls(**ARM_PRESETS[arm], name=arm)

    @property
    def needs_condition_vector(self) -> bool:
        return any(b.needs_condition_vector for b in self.blocks)

    @property
    def source_columns(self) -> tuple[str, ...]:
        cols: list[str] = []
        for b in self.blocks:
            cols += list(b.source_columns)
        if self.needs_condition_vector:
            cols += list(N.REQUIRED_COLUMNS)
        return tuple(dict.fromkeys(cols))

    def _view(self, rows: pd.DataFrame, cv: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        if not rows.index.is_unique:
            raise ValueError("FeatureSet: the row index must be unique")
        required = [c for b in self.blocks for c in b.source_columns
                    if c in (METAL_STATE_COL, ELEMENT_COL, SYSTEM_COL)]
        missing = [c for c in dict.fromkeys(required) if c not in rows.columns]
        if missing:
            raise KeyError(f"FeatureSet: row columns missing {missing}")
        view = rows.loc[:, [c for c in self.source_columns if c in rows.columns]]
        assert_feature_columns_allowed(view.columns, "row columns handed to the blocks")
        if self.needs_condition_vector:
            if cv is None:
                cv = N.condition_vector(view)
            else:
                if not cv.index.is_unique or not rows.index.isin(cv.index).all():
                    raise ValueError("FeatureSet: cv does not cover the rows")
                cv = cv.loc[rows.index]
        else:
            cv = None
        return view, cv

    def fit(self, train_rows: pd.DataFrame, cv: pd.DataFrame | None = None) -> "FeatureSet":
        """Fit every block and encoding on ``train_rows`` only (``cv``: optional precomputed
        ``normalize.condition_vector`` covering the rows -- a row-wise, target-free transform)."""
        if not len(train_rows):
            raise ValueError("FeatureSet.fit: no training rows")
        view, cvv = self._view(train_rows, cv)
        self._preps = {}
        for b in self.blocks:
            b.fit(view, cvv)
            raw = b.raw(view, cvv)
            self._preps[b.name] = _Prep(b.name, self.config["categorical"], self.config["missing"],
                                        self.config["missing_indicators"], self.config["standardize"]).fit(raw)
        self.n_train_rows = int(len(train_rows))
        self._digest = state_digest(self.state())
        return self

    @property
    def fitted(self) -> bool:
        return self._digest is not None

    def state(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "config": self.config, "blocks": [b.name for b in self.blocks],
                "block_state": {b.name: b.state() for b in self.blocks},
                "prep_state": {k: v.state() for k, v in self._preps.items()},
                "n_train_rows": self.n_train_rows, "static_tables": static_tables().digests}

    @property
    def state_digest(self) -> str:
        if self._digest is None:
            raise RuntimeError("FeatureSet: fit first")
        return self._digest

    def transform(self, rows: pd.DataFrame, cv: pd.DataFrame | None = None) -> FeatureMatrix:
        """Features of ``rows`` under the fitted state; raises if the state changed (transform never refits)."""
        if self._digest is None:
            raise RuntimeError("FeatureSet: fit first")
        view, cvv = self._view(rows, cv)
        frames, cats, inds, ids, diags, wide = [], [], [], {}, [], {}
        block_cols: dict[str, tuple[str, ...]] = {}
        for b in self.blocks:
            raw = b.raw(view, cvv)
            fr, c, i = self._preps[b.name].transform(raw)
            frames.append(fr)
            cats += c
            inds += i
            block_cols[b.name] = tuple(fr.columns)
            for k, v in raw.ids.items():
                if k in ids:
                    raise ValueError(f"duplicate id {k}")
                ids[k] = v
            if raw.diagnostics is not None:
                diags.append(raw.diagnostics)
            for k, v in raw.wide.items():
                wide[f"{b.name}__{k}"] = v
        frame = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=rows.index)
        if frame.columns.duplicated().any():
            raise ValueError("FeatureSet: duplicate feature columns")
        assert_feature_columns_allowed(frame.columns, "output feature columns")
        assert_feature_columns_allowed([c for _, (_, cols) in wide.items() for c in cols], "wide feature columns")
        if state_digest(self.state()) != self._digest:
            raise AssertionError("FeatureSet: the fitted state changed during transform")
        diag = pd.concat(diags, axis=1) if diags else pd.DataFrame(index=rows.index)
        return FeatureMatrix(frame, tuple(cats), tuple(inds), block_cols, wide, ids, diag, self._digest)

    def fit_transform(self, train_rows: pd.DataFrame, cv: pd.DataFrame | None = None) -> FeatureMatrix:
        return self.fit(train_rows, cv).transform(train_rows, cv)

    def id_sizes(self) -> dict[str, int]:
        """Embedding table sizes (reserved index included) of every fitted vocabulary id."""
        out: dict[str, int] = {}
        for b in self.blocks:
            if isinstance(b, MetalBlock):
                out.update({k: v.size for k, v in b.vocab.items()})
            if isinstance(b, ExtractantBlock):
                out["system_id"] = b.system_vocab.size
        return out


# --------------------------------------------------------------------------------------------- #
# ECFP clusters (B8 sample weights)
# --------------------------------------------------------------------------------------------- #

def ecfp_cluster_label(bits: np.ndarray) -> str:
    """gen6 ``levels.ecfp_cluster_labels``: first 16 hex of sha1 over the int8 bit row."""
    return hashlib.sha1(np.ascontiguousarray(np.asarray(bits).astype(np.int8)).tobytes()).hexdigest()[:16]


def system_cluster_label(system: str, fallback: Mapping[str, Any] | None = None) -> str:
    rec = system_record(system, fallback)
    b = ecfp_bits(rec.primary_smiles)
    if b is None:
        return "nofp_" + hashlib.sha1(str(system).encode("utf-8")).hexdigest()[:16]
    return ecfp_cluster_label(b)


class EcfpClusters:
    """Bit-identical primary-extractant fingerprint groups of the training systems and the balanced sample weights
    of the training rows (section 5 B8; :data:`REGISTRATION_CHOICES` ``ecfp_cluster``)."""

    source_columns = (SYSTEM_COL, "primary_extractant_smiles")

    def __init__(self) -> None:
        self.system_labels: dict[str, str] = {}
        self.cluster_rows: dict[str, int] = {}
        self._weights: pd.Series | None = None

    def labels(self, rows: pd.DataFrame) -> np.ndarray:
        view = rows.loc[:, [c for c in self.source_columns if c in rows.columns]]
        fb = _row_fallbacks(view)
        keys = [_token(k) for k in view[SYSTEM_COL].to_numpy(dtype=object)]
        cache = {k: system_cluster_label(k, fb.get(k)) for k in sorted(set(keys)) if k != NA_TOKEN}
        return np.array([cache.get(k, NA_TOKEN) for k in keys], dtype=object)

    def fit(self, train_rows: pd.DataFrame) -> "EcfpClusters":
        lab = self.labels(train_rows)
        keys = [_token(k) for k in train_rows[SYSTEM_COL].to_numpy(dtype=object)]
        self.system_labels = dict(sorted({k: l for k, l in zip(keys, lab)}.items()))
        u, inv, cnt = np.unique(lab.astype(str), return_inverse=True, return_counts=True)
        self.cluster_rows = {str(a): int(b) for a, b in zip(u, cnt)}
        self._weights = pd.Series((1.0 / cnt[inv]).astype(float), index=train_rows.index, name="ecfp_cluster_weight")
        return self

    def sample_weights(self, index: Iterable[Any] | None = None) -> np.ndarray:
        """Balanced weights of the fitted training rows (in ``index`` order when given)."""
        if self._weights is None:
            raise RuntimeError("EcfpClusters: fit first")
        if index is None:
            return self._weights.to_numpy()
        idx = pd.Index(list(index))
        if not idx.isin(self._weights.index).all():
            raise KeyError("sample_weights: rows outside the fitted training rows")
        return self._weights.loc[idx].to_numpy()

    def state(self) -> dict[str, Any]:
        return {"system_labels": self.system_labels, "cluster_rows": self.cluster_rows}


# --------------------------------------------------------------------------------------------- #
# TOPO39 (B8 direction inputs)
# --------------------------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def topo39_table() -> pd.DataFrame:
    """The 39 TOPO39 columns of the frozen gen12.2 coordination table (``coord__dist__*`` + ``coord__arm__*`` in file
    order, gen14 ``dirbench.feature_sets``), indexed by RDKit canonical SMILES."""
    p = pd.read_parquet(COORDINATION_PARQUET)
    if p.index.name != "extractant":
        p = p.set_index("extractant")
    cols = [c for c in p.columns if c.startswith(TOPO39_PREFIXES)]
    if len(cols) != 39:
        raise AssertionError(f"TOPO39: {len(cols)} columns, expected 39")
    canon = [LG.canonical_smiles(s) for s in p.index]
    if any(c is None for c in canon) or len(set(canon)) != len(canon):
        raise AssertionError("TOPO39: the table's SMILES do not canonicalise one-to-one")
    out = p[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    out.index = pd.Index(canon, name="canonical_smiles")
    return out


def topo39_columns() -> tuple[str, ...]:
    return tuple(topo39_table().columns)


def topo39_for_smiles(smiles: Sequence[str | None]) -> pd.DataFrame:
    """TOPO39 rows for structures (NaN rows where the table has none), in input order."""
    tab = topo39_table()
    canon = [None if _missing(s) else LG.canonical_smiles(str(s)) for s in smiles]
    out = tab.reindex([c if c is not None else "__none__" for c in canon])
    out.index = pd.RangeIndex(len(smiles))
    return out


def topo39_for_systems(systems: Sequence[str]) -> pd.DataFrame:
    """TOPO39 by the primary extractant of each system (index = system key); a ``topo39_available`` column is added."""
    keys = [str(s) for s in systems]
    out = topo39_for_smiles([system_record(k).primary_smiles for k in keys])
    out.index = pd.Index(keys, name=SYSTEM_COL)
    out["topo39_available"] = out[list(topo39_columns())].notna().all(axis=1)
    return out


def topo39_coverage(rows: pd.DataFrame, min_ln3_states: int = 5) -> dict[str, Any]:
    """Coverage of TOPO39 over the systems, known-state cells, Ln(III) cells and rows of ``rows`` (target-free), and
    over the systems with >= ``min_ln3_states`` distinct Ln(III) states (the section 5 B8 direction training rule)."""
    st = rows[METAL_STATE_COL].to_numpy(dtype=object)
    sy = rows[SYSTEM_COL].to_numpy(dtype=object)
    systems = sorted({str(s) for s in sy if not _missing(s)})
    avail = topo39_for_systems(systems)["topo39_available"].to_dict()
    known = np.array([not _missing(a) for a in st])
    cells = {(str(a), str(b)) for a, b, k in zip(st, sy, known) if k and not _missing(b)}
    ln3 = {c for c in cells if metal_record(c[0], None)["series_token"] == "Ln" and c[0].endswith("(III)")}
    ln3_by_sys: dict[str, set[str]] = {}
    for m, s in ln3:
        ln3_by_sys.setdefault(s, set()).add(m)
    rich = sorted(s for s, ms in ln3_by_sys.items() if len(ms) >= min_ln3_states)
    row_ok = np.array([not _missing(s) and bool(avail[str(s)]) for s in sy])
    return {
        "n_systems": len(systems), "n_systems_topo39": int(sum(avail.values())),
        "n_rows": int(len(rows)), "n_rows_topo39": int(row_ok.sum()),
        "n_cells_known_state": len(cells), "n_cells_topo39": int(sum(avail[s] for _, s in cells)),
        "n_ln3_cells": len(ln3), "n_ln3_cells_topo39": int(sum(avail[s] for _, s in ln3)),
        f"n_systems_ge{min_ln3_states}_ln3_states": len(rich),
        f"n_systems_ge{min_ln3_states}_ln3_states_topo39": int(sum(avail[s] for s in rich)),
        "source": paths.rel(COORDINATION_PARQUET), "n_table_structures": int(len(topo39_table())),
    }


class Topo39Prep:
    """Median imputation and standardisation of TOPO39 fitted on a list of training systems, each system once
    (section 5 B8 direction: 'standardised, median-imputed'); systems without TOPO39 are not fit units."""

    def __init__(self) -> None:
        self.median: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.systems: list[str] = []

    def fit(self, systems: Sequence[str]) -> "Topo39Prep":
        t = topo39_for_systems(sorted(set(map(str, systems))))
        t = t[t["topo39_available"]]
        self.systems = list(t.index)
        X = t[list(topo39_columns())].to_numpy(dtype=float)
        if not len(X):
            raise ValueError("Topo39Prep.fit: no training system has TOPO39")
        self.median = np.array([float(np.median(np.sort(c[np.isfinite(c)]))) if np.isfinite(c).any() else 0.0
                                for c in X.T])
        Xi = np.where(np.isfinite(X), X, self.median)
        self.mean = np.array([_sorted_mean(c) for c in Xi.T])
        sd = np.array([_sorted_std(c) for c in Xi.T])
        self.sd = np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
        return self

    def transform(self, systems: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(standardised matrix, available mask)``; rows of systems without TOPO39 are NaN."""
        if self.median is None:
            raise RuntimeError("Topo39Prep: fit first")
        t = topo39_for_systems([str(s) for s in systems])
        X = t[list(topo39_columns())].to_numpy(dtype=float)
        ok = t["topo39_available"].to_numpy(dtype=bool)
        Xi = (np.where(np.isfinite(X), X, self.median) - self.mean) / self.sd
        Xi[~ok] = np.nan
        return Xi, ok

    def state(self) -> dict[str, Any]:
        return {"systems": self.systems, "median": self.median, "mean": self.mean, "sd": self.sd}


# --------------------------------------------------------------------------------------------- #
# d_desc (section 3.5)
# --------------------------------------------------------------------------------------------- #

class DDescScaler:
    """``d_desc`` = Euclidean distance in (``mw``, ``mol_logp``, ``rotatable_bonds``) of the primary extractant,
    standardised over the training systems (each once; ``nanmean`` / ``nanstd``, SD <= 0 -> 1) -- the scale of
    ``baselines.LookupEngine._d_desc_scale`` (tested equal)."""

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    @staticmethod
    def vector(system: str) -> np.ndarray:
        rec = system_record(str(system))
        if rec.primary_smiles is None:
            return np.full(len(D_DESC_COLUMNS), np.nan)
        tab = static_tables().components_by_smiles
        if rec.primary_smiles not in tab.index:
            return np.full(len(D_DESC_COLUMNS), np.nan)
        return np.array([_num(tab.loc[rec.primary_smiles, c]) for c in D_DESC_COLUMNS], dtype=float)

    def fit(self, train_rows: pd.DataFrame) -> "DDescScaler":
        systems = sorted({str(s) for s in train_rows[SYSTEM_COL].to_numpy(dtype=object) if not _missing(s)})
        D = np.vstack([self.vector(s) for s in systems]) if systems else np.zeros((0, len(D_DESC_COLUMNS)))
        with np.errstate(all="ignore"):
            mu = np.nanmean(D, axis=0) if len(D) else np.full(len(D_DESC_COLUMNS), np.nan)
            sd = np.nanstd(D, axis=0) if len(D) else np.full(len(D_DESC_COLUMNS), np.nan)
        self.mu, self.sd = mu, np.where(~np.isfinite(sd) | (sd <= 0), 1.0, sd)
        return self

    def distance(self, system_a: str, system_b: str) -> float:
        if self.mu is None:
            raise RuntimeError("DDescScaler: fit first")
        za = (self.vector(system_a) - self.mu) / self.sd
        zb = (self.vector(system_b) - self.mu) / self.sd
        d = float(np.sqrt(np.sum((za - zb) ** 2)))
        return d if np.isfinite(d) else float("nan")
