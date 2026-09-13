"""``ingest.py`` — frozen bundle + gen6 provenance -> the per-system database (DESIGN.md 4.1-4.6).

Pipeline (``ingest_corpus``): bundle SHA check; column projection (section 4.1); merge with the
provenance table one-to-one on ``safe_exp_id``; gen13 quarantine (``gen13sep.cohort.
apply_quarantine``); one-hot decoding (acid, diluent, additives -> synergists / modifiers);
system key (section 3.1); fit eligibility; tied-D groups and the unit-slip rule (section 4.6);
publication-aware loading series (section 4.5); applicability domains per (ligand, band)
(section 5.5, box + convex hull, built locally); one ``SystemEntry`` per key written with
``systems.write_system``; flat mirrors ``corpus_records.csv``, ``series.csv``,
``duplicates.csv``, ``exclusions.csv``, ``registry.json``, ``INDEX.md``.

Units: bundle ``cond__acid_concentration_M`` and ``cond__extractant_concentration_M`` mol/L,
``cond__metal_concentration_mM`` mM (semantics ASSUMED initial aqueous concentration of the
row's single metal, section 4.4), ``cond__temperature_C`` degC, ``cond__contact_time_min`` min;
``log_D`` is log10.  O/A is not in the bundle (``oa_ratio`` None on every record).

Declared assumptions (status ``assumed`` on every derived parameter, PRE_REGISTRATION.md
section 2): metal concentration = initial aqueous of the row's metal; O/A = 1 for loading
fractions; nominal acid = equilibrium acidity; in HNO3 media aqueous nitrate = nominal acid.
``anion_M`` is set to the nominal acid concentration for every acid class (for polyprotic and
weak acids it is the formal acid concentration, not a free-anion value).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import paths
from .literature import EXTRACTANT_SMILES, MODIFIER_SMILES
from .systems import (
    DEFAULT_BAND,
    SCHEMA_VERSION,
    ApplicabilityDomain,
    ApplicabilityMapping,
    DistributionRecord,
    LigandSpec,
    Mechanism,
    MediumSpec,
    PhaseBehaviour,
    Provenance,
    ProvStatus,
    Source,
    Sourced,
    Stoichiometry,
    SystemEntry,
    entry_to_json,
    registry_row,
    system_id,
    temperature_band,
    validate_entry,
    write_system,
)

__all__ = [
    "BUNDLE_BASE_COLUMNS", "PROVENANCE_COLUMNS", "ATOMIC_MASS_G_MOL", "SYNERGIST_ADDITIVES",
    "ACID_NAMES", "LOADING_N_PRIOR", "IngestAudit", "bundle_columns", "load_corpus",
    "decode_onehots", "classify_ligand", "assign_systems", "mark_fit_eligibility",
    "find_duplicates", "assign_loading_series", "replicate_groups", "build_domains",
    "build_entry", "ingest_corpus",
]

BUNDLE_BASE_COLUMNS: tuple[str, ...] = (
    "safe_exp_id", "metal", "canonical_smiles", "extractant_name", "D", "log_D",
    "geom_cond__acid_class", "geom_cond__diluent_family",
)
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "safe_exp_id", "publication_id", "experiment_series_id", "replicate_id", "condition_id",
    "cell_id",
)
CONTINUOUS = ("cond__acid_concentration_M", "cond__extractant_concentration_M",
              "cond__metal_concentration_mM", "cond__temperature_C")

#: IUPAC conventional atomic weights (g/mol), used only by the unit-slip ratio test.
ATOMIC_MASS_G_MOL: dict[str, float] = {
    "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.36, "Eu": 151.964,
    "Gd": 157.25, "Tb": 158.925, "Dy": 162.500, "Ho": 164.930, "Er": 167.259, "Tm": 168.934,
    "Yb": 173.045, "Lu": 174.967,
}
UNIT_FACTORS = (1000.0,)                     # M <-> mM (1/1000 is the same ratio inverted)
UNIT_RATIO_TOL = 1e-3                        # within 0.1 %

SYNERGIST_ADDITIVES: dict[str, str] = {"hdehp": "HDEHP", "tbp": "TBP", "dhoa": "DHOA",
                                       "dohya": "DOHyA"}
ACID_NAMES: dict[str, str] = {
    "hno3": "HNO3", "hcl": "HCl", "h2so4": "H2SO4", "hclo4": "HClO4",
    "citric_acid": "citric acid", "lactic_acid": "lactic acid", "malonic_acid": "malonic acid",
    "tartaric_acid": "tartaric acid", "hno3_oxalic_acid": "HNO3 + oxalic acid",
}
LOADING_N_PRIOR = 3.0
"""Ligands per metal used for the domain's loading-fraction axis (DESIGN.md section 5.5)."""

_MECHANISM_DOI = "10.1016/j.jiec.2014.03.002"    # the brief's reference for the mechanism classes
_S2_DOI = "10.1038/s41598-020-74041-9"


@dataclass(frozen=True)
class IngestAudit:
    """Counts returned by ``ingest_corpus`` (DESIGN.md section 4.2 step 7, plus extras)."""

    bundle_sha256: str
    rows_in: int
    todga_name_mismatch_rows: int
    sentinel_rows: int
    rows_fit_ineligible: int
    n_systems: int
    n_records: int
    n_publications: int
    n_loading_series_publication_aware: int
    n_loading_series_publication_blind: int
    n_unit_slip_rows: int
    n_tied_d_groups: int
    replicate_groups: int
    replicate_sd_median: float
    # extras (superset; see addenda/WB1.md)
    rows_after_quarantine: int = 0
    fit_ineligible_reasons: dict[str, int] = field(default_factory=dict)
    n_unit_slip_groups: int = 0
    n_tied_d_rows: int = 0
    n_tied_d_publications: int = 0
    n_tied_d_tier_groups: int = 0
    n_tied_d_tier_rows: int = 0
    n_nan_metal_rows: int = 0
    n_nan_temperature_rows: int = 0
    band_counts: dict[str, int] = field(default_factory=dict)
    loading_series_ids: tuple[str, ...] = ()
    publication_blind_series_ids: tuple[str, ...] = ()
    n_two_ligand_systems: int = 0
    n_systems_with_modifiers: int = 0
    n_literature_entries: int = 0

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["loading_series_ids"] = list(self.loading_series_ids)
        out["publication_blind_series_ids"] = list(self.publication_blind_series_ids)
        return out


# =============================================================================================
# 4.1 / 4.2 steps 1-2: columns, join, quarantine
# =============================================================================================

def bundle_columns() -> list[str]:
    """The base columns plus every ``cond__*`` column of the bundle schema (about 80)."""
    schema = pq.read_schema(str(paths.BUNDLE_PARQUET))
    cond = [c for c in schema.names if c.startswith("cond__")]
    return list(BUNDLE_BASE_COLUMNS) + cond


def load_corpus() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Bundle SHA check, projected read, one-to-one provenance merge, gen13 quarantine.

    Returns ``(frame, exclusions, quarantine_audit)``; ``exclusions`` has ``safe_exp_id`` and
    ``reason in {todga_name_mismatch, sentinel_logD_le_-6}``."""
    from gen13sep.cohort import LOG_D_FLOOR, TODGA_SMILES  # gen13 on sys.path via paths

    sha = paths.assert_bundle_unchanged()
    bundle = pd.read_parquet(paths.BUNDLE_PARQUET, columns=bundle_columns())
    prov = pd.read_parquet(paths.GEN6_PROVENANCE_PARQUET, columns=list(PROVENANCE_COLUMNS))
    frame = bundle.merge(prov, on="safe_exp_id", how="left", validate="one_to_one")
    if frame["publication_id"].isna().any():
        raise RuntimeError("publication_id missing for some bundle rows")
    name = frame["extractant_name"].astype(str).str.upper().str.replace(" ", "", regex=False)
    todga_mismatch = (frame["canonical_smiles"] == TODGA_SMILES) & (name != "TODGA")
    sentinel = frame["log_D"] <= LOG_D_FLOOR
    exclusions = pd.concat([
        pd.DataFrame({"safe_exp_id": frame.loc[todga_mismatch, "safe_exp_id"],
                      "reason": "todga_name_mismatch"}),
        pd.DataFrame({"safe_exp_id": frame.loc[sentinel & ~todga_mismatch, "safe_exp_id"],
                      "reason": "sentinel_logD_le_-6"}),
    ]).sort_values("safe_exp_id").reset_index(drop=True)
    kept = frame.loc[~todga_mismatch & ~sentinel].copy()
    audit = {"bundle_sha256": sha, "rows_in": int(len(frame)),
             "todga_name_mismatch_rows": int(todga_mismatch.sum()),
             "sentinel_rows": int(sentinel.sum()), "rows_out": int(len(kept))}
    return kept.reset_index(drop=True), exclusions, audit


# =============================================================================================
# 4.2 step 3: one-hot decoding
# =============================================================================================

def _onehot_name(frame: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in frame.columns if c.startswith(prefix)]
    block = frame[cols].to_numpy()
    names = np.array([c[len(prefix):] for c in cols], dtype=object)
    out = np.full(len(frame), None, dtype=object)
    hit = block.sum(axis=1) > 0
    out[hit] = names[block[hit].argmax(axis=1)]
    return pd.Series(out, index=frame.index)


def decode_onehots(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds ``acid_name``, ``diluent_name``, ``synergists`` (tuple of names), ``modifiers``."""
    frame = frame.copy()
    frame["acid_name"] = _onehot_name(frame, "cond__acid__")
    frame["diluent_name"] = _onehot_name(frame, "cond__diluent__")
    add_cols = [c for c in frame.columns if c.startswith("cond__additive__")]
    syn, mod = [], []
    block = frame[add_cols].to_numpy()
    keys = [c[len("cond__additive__"):] for c in add_cols]
    for row in block:
        present = [k for k, v in zip(keys, row) if v == 1]
        syn.append(tuple(sorted(SYNERGIST_ADDITIVES[k] for k in present
                                if k in SYNERGIST_ADDITIVES)))
        mod.append(tuple(sorted(k for k in present if k not in SYNERGIST_ADDITIVES)))
    frame["synergists"] = syn
    frame["modifiers"] = mod
    return frame


# =============================================================================================
# ligand classification (family, mechanism, aggregation) by substructure
# =============================================================================================

_SMARTS = {
    "acid_P_OH": "[PX4](=O)[OX2H1]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
    "dga": "[NX3][CX3](=O)[CH2]O[CH2][CX3](=O)[NX3]",
    "amide": "[NX3][CX3]=O",
    "phenanthroline": "c1cnc2c(c1)ccc1cccnc12",
    "aromatic_n": "[n]",
    "amine": "[NX3;!$(N[C,S,P]=O);!$(N-a)]",
}


def classify_ligand(smiles: str) -> tuple[str, Mechanism, str]:
    """``(family, mechanism, aggregation)`` from RDKit substructures (metadata; documented in
    addenda/WB1.md).  Acidic P-OH / carboxylic / sulfonic -> cation exchange, dimer; else
    solvating, monomer."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "other", Mechanism.SOLVATING, "monomer"

    def has(key: str) -> bool:
        return mol.HasSubstructMatch(Chem.MolFromSmarts(_SMARTS[key]))

    if has("acid_P_OH"):
        return "acidic_organophosphorus", Mechanism.CATION_EXCHANGE, "dimer"
    if has("carboxylic_acid"):
        return "carboxylic_acid", Mechanism.CATION_EXCHANGE, "dimer"
    if has("sulfonic_acid"):
        return "other", Mechanism.CATION_EXCHANGE, "dimer"
    if has("dga"):
        return "diglycolamide", Mechanism.SOLVATING, "monomer"
    if has("phenanthroline") and has("amide"):
        return "phen_carboxamide", Mechanism.SOLVATING, "monomer"
    if has("aromatic_n") and not has("amide"):
        return "n_donor", Mechanism.SOLVATING, "monomer"
    if has("amine") and not has("amide") and not has("aromatic_n"):
        return "amine", Mechanism.SOLVATING, "monomer"
    return "other", Mechanism.SOLVATING, "monomer"


def _canonical(smiles: str) -> str:
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


# =============================================================================================
# 4.2 step 4: system key
# =============================================================================================

def assign_systems(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds ``system_id`` (section 3.1) and ``ligand_name`` (one name per SMILES: the most
    frequent ``extractant_name``, ties by sorted order)."""
    frame = frame.copy()
    syn_canon = {n: _canonical(EXTRACTANT_SMILES[n]) for n in SYNERGIST_ADDITIVES.values()}
    ids = []
    cols = ["canonical_smiles", "geom_cond__acid_class", "geom_cond__diluent_family",
            "synergists", "modifiers"]
    for smi, acid_class, fam, syn, mod in frame[cols].itertuples(index=False):
        ligands = tuple(sorted([smi] + [syn_canon[s] for s in syn]))
        ids.append(system_id((ligands, acid_class, fam, tuple(mod), ())))
    frame["system_id"] = ids
    counts = frame.groupby(["canonical_smiles", "extractant_name"]).size().reset_index(name="n")
    counts = counts.sort_values(["canonical_smiles", "n", "extractant_name"],
                                ascending=[True, False, True])
    name_of = (counts.drop_duplicates("canonical_smiles").set_index("canonical_smiles")
               ["extractant_name"])
    frame["ligand_name"] = frame["canonical_smiles"].map(name_of)
    return frame


# =============================================================================================
# 4.2 step 5: fit eligibility (before the duplicate rule)
# =============================================================================================

def mark_fit_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    acid = frame["cond__acid_concentration_M"]
    ext = frame["cond__extractant_concentration_M"]
    reason = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    reason[acid.isna() | (acid <= 0)] = "acid_nan_or_nonpositive"
    reason[(ext.isna() | (ext <= 0)) & reason.isna()] = "extractant_nan_or_zero"
    frame["fit_ineligible_reason"] = reason
    frame["fit_eligible"] = reason.isna()
    frame["duplicate_flag"] = None
    return frame


# =============================================================================================
# 4.6 duplicates
# =============================================================================================

def _cond_string(frame: pd.DataFrame) -> pd.Series:
    def fmt(v: Any) -> str:
        return "NA" if pd.isna(v) else format(float(v), ".9g")
    return frame[list(CONTINUOUS)].apply(lambda r: "|".join(fmt(v) for v in r), axis=1)


def _ratio_is_unit_factor(a: float, b: float, metal: str) -> bool:
    if not (a > 0 and b > 0):
        return False
    r = max(a, b) / min(a, b)
    factors = UNIT_FACTORS + (ATOMIC_MASS_G_MOL.get(metal, math.nan),)
    return any(abs(r / f - 1.0) <= UNIT_RATIO_TOL for f in factors if f == f)


def _series_key(row: pd.Series) -> tuple:
    return (row["system_id"], row["metal"], row["publication_id"],
            row["cond__acid_concentration_M"], row["cond__extractant_concentration_M"])


def find_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tied groups within (publication, SMILES, metal): D equal to 6 significant digits at
    different (acid, extractant, metal concentration, temperature).  Unit-slip tier: a pair
    whose metal concentrations differ by a unit factor (ratio within 0.1 % of the metal's molar
    mass, i.e. g/L versus mM, or 1000, M versus mM; the other conditions may also differ,
    addenda/WB1.md A3) is a copy; the copy kept is the one closer to the least-squares line
    through the clean points of its own series (>= 3 clean points), the other gets
    ``UNIT_SLIP_DUPLICATE`` and ``fit_eligible = False``; when only one side has a series with
    >= 3 clean points that side is kept; when neither has, both copies are flagged.  All other
    tied rows get ``TIED_D`` and stay eligible.

    Returns ``(frame, duplicates)`` where ``duplicates`` has one row per tied row."""
    frame = frame.copy()
    frame["_cond"] = _cond_string(frame)
    frame["_d6"] = frame["D"].map(lambda x: format(float(x), ".6g"))
    grp_cols = ["publication_id", "canonical_smiles", "metal", "_d6"]
    g = frame.groupby(grp_cols)
    stats = g["_cond"].agg(["size", "nunique"])
    tied_keys = stats[(stats["size"] >= 2) & (stats["nunique"] > 1)].index
    tied = (frame.set_index(grp_cols).loc[tied_keys].reset_index() if len(tied_keys)
            else frame.iloc[0:0])
    tied_ids = set(tied["safe_exp_id"])
    # clean points per series (not in any tied group, metal concentration present)
    clean = frame[~frame["safe_exp_id"].isin(tied_ids)
                  & frame["cond__metal_concentration_mM"].notna()
                  & frame["fit_eligible"]]
    clean_by_series: dict[tuple, pd.DataFrame] = {k: v for k, v in clean.groupby(
        ["system_id", "metal", "publication_id", "cond__acid_concentration_M",
         "cond__extractant_concentration_M"])}

    def residual(row: pd.Series) -> float | None:
        pts = clean_by_series.get(_series_key(row))
        if pts is None or len(pts) < 3:
            return None
        x = np.log10(pts["cond__metal_concentration_mM"].to_numpy(dtype=float))
        y = pts["log_D"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        return float(row["log_D"] - (slope * math.log10(row["cond__metal_concentration_mM"])
                                     + intercept))

    flag: dict[str, str] = {}
    tier: dict[str, str] = {}
    kept: dict[str, bool] = {}
    resid: dict[str, float | None] = {}
    group_id: dict[str, str] = {}
    for key, grp in tied.groupby(grp_cols, sort=True):
        gid = hashlib.blake2b("|".join(map(str, key)).encode(), digest_size=8).hexdigest()
        rows = grp.sort_values("safe_exp_id").to_dict("records")
        for r in rows:
            group_id[r["safe_exp_id"]] = gid
            tier.setdefault(r["safe_exp_id"], "tied_d")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a["_cond"] == b["_cond"]:
                    continue
                # unit factor on the metal-concentration axis only (addenda/WB1.md A3): the
                # keep-rule compares against the loading series in (log mM, log D)
                va, vb = a["cond__metal_concentration_mM"], b["cond__metal_concentration_mM"]
                if pd.isna(va) or pd.isna(vb) or va == vb:
                    continue
                if not _ratio_is_unit_factor(va, vb, a["metal"]):
                    continue
                ra, rb = residual(pd.Series(a)), residual(pd.Series(b))
                resid[a["safe_exp_id"]], resid[b["safe_exp_id"]] = ra, rb
                tier[a["safe_exp_id"]] = tier[b["safe_exp_id"]] = "unit_slip"
                if ra is None and rb is None:
                    losers = (a, b)
                elif ra is None:
                    losers = (a,)
                elif rb is None:
                    losers = (b,)
                else:
                    losers = (a,) if abs(ra) > abs(rb) else (b,)
                for r in (a, b):
                    kept.setdefault(r["safe_exp_id"], True)
                for r in losers:
                    flag[r["safe_exp_id"]] = "UNIT_SLIP_DUPLICATE"
                    kept[r["safe_exp_id"]] = False
    for sid in tied_ids:
        if tier.get(sid) == "tied_d":
            flag[sid] = "TIED_D"
            kept[sid] = True
    frame["duplicate_flag"] = frame["safe_exp_id"].map(flag).astype(object).where(
        frame["safe_exp_id"].isin(flag), None)
    slip = frame["duplicate_flag"] == "UNIT_SLIP_DUPLICATE"
    frame.loc[slip, "fit_eligible"] = False
    no_reason = slip & frame["fit_ineligible_reason"].isna()
    frame.loc[no_reason, "fit_ineligible_reason"] = "UNIT_SLIP_DUPLICATE"
    dup = tied[["safe_exp_id", "publication_id", "canonical_smiles", "metal", "D", "log_D",
                "system_id", *CONTINUOUS]].copy()
    dup.insert(0, "group_id", dup["safe_exp_id"].map(group_id))
    dup["tier"] = dup["safe_exp_id"].map(tier)
    dup["duplicate_flag"] = dup["safe_exp_id"].map(flag)
    dup["kept"] = dup["safe_exp_id"].map(kept)
    dup["series_residual"] = dup["safe_exp_id"].map(resid)
    dup = dup.sort_values(["group_id", "safe_exp_id"]).reset_index(drop=True)
    return frame.drop(columns=["_cond", "_d6"]), dup


# =============================================================================================
# 4.5 loading series
# =============================================================================================

def _fmt(x: float) -> str:
    return format(float(x), "g")


def assign_loading_series(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Publication-aware series on the records (``loading_series_id``, ``is_tracer``) and a
    ``series`` table holding both definitions (``definition`` in {publication_aware,
    publication_blind}); ids ``ls_<system_id>_<metal>_<publication_id>_<acid>_<ligand>`` and
    ``lsb_<system_id>_<metal>_<acid>_<ligand>``."""
    frame = frame.copy()
    frame["loading_series_id"] = None
    frame["is_tracer"] = False
    elig = frame[frame["fit_eligible"] & frame["cond__metal_concentration_mM"].notna()]
    rows = []
    aware_cols = ["system_id", "metal", "publication_id", "cond__acid_concentration_M",
                  "cond__extractant_concentration_M"]
    for key, grp in elig.groupby(aware_cols, sort=True):
        if grp["cond__metal_concentration_mM"].nunique() < 3:
            continue
        sid = f"ls_{key[0]}_{key[1]}_{key[2]}_{_fmt(key[3])}_{_fmt(key[4])}"
        frame.loc[grp.index, "loading_series_id"] = sid
        tracer = grp.sort_values(["cond__metal_concentration_mM", "safe_exp_id"]).index[0]
        frame.loc[tracer, "is_tracer"] = True
        rows.append(_series_row(sid, "publication_aware", key[0], key[1], key[2], key[3], key[4],
                                grp, frame.loc[tracer]))
    blind_cols = ["system_id", "metal", "cond__acid_concentration_M",
                  "cond__extractant_concentration_M"]
    for key, grp in elig.groupby(blind_cols, sort=True):
        if grp["cond__metal_concentration_mM"].nunique() < 3:
            continue
        sid = f"lsb_{key[0]}_{key[1]}_{_fmt(key[2])}_{_fmt(key[3])}"
        tracer = grp.sort_values(["cond__metal_concentration_mM", "safe_exp_id"]).index[0]
        rows.append(_series_row(sid, "publication_blind", key[0], key[1],
                                ";".join(sorted(grp["publication_id"].unique())), key[2], key[3],
                                grp, frame.loc[tracer]))
    series = pd.DataFrame(rows, columns=[
        "loading_series_id", "definition", "system_id", "ligand_name", "metal", "publication_id",
        "acid_nominal_M", "ligand_M", "n_points", "n_distinct_mM", "mM_min", "mM_max",
        "log_d_tracer", "log_d_min", "log_d_max", "log_d_range", "loading_active",
        "tracer_record_id", "record_ids"])
    return frame, series


def _series_row(sid, definition, system, metal, pub, acid, ext, grp, tracer) -> list:
    rng = float(grp["log_D"].max() - grp["log_D"].min())
    return [sid, definition, system, str(grp["ligand_name"].iloc[0]), metal, pub, float(acid),
            float(ext), int(len(grp)), int(grp["cond__metal_concentration_mM"].nunique()),
            float(grp["cond__metal_concentration_mM"].min()),
            float(grp["cond__metal_concentration_mM"].max()), float(tracer["log_D"]),
            float(grp["log_D"].min()), float(grp["log_D"].max()), rng, bool(rng >= 0.3),
            str(tracer["safe_exp_id"]), ";".join(sorted(grp["safe_exp_id"]))]


# =============================================================================================
# replicate groups (exact 64-column key + publication + SMILES + metal)
# =============================================================================================

def replicate_groups(frame: pd.DataFrame) -> tuple[pd.Series, int, float]:
    """Returns ``(replicate_group_id per row, n groups with >= 2 rows, median within-group sd
    of log D over those groups)``."""
    from gen13sep.cohort import _condition_key

    cond = [c for c in frame.columns if c.startswith("cond__")]
    ck = _condition_key(frame, cond)
    key = (ck + "|" + frame["publication_id"].astype(str) + "|" + frame["canonical_smiles"]
           + "|" + frame["metal"])
    gid = key.map(lambda s: hashlib.blake2b(s.encode(), digest_size=8).hexdigest())
    stats = frame.assign(_g=gid).groupby("_g")["log_D"].agg(["size", "std"])
    multi = stats[stats["size"] >= 2]
    return gid, int(len(multi)), float(multi["std"].median()) if len(multi) else math.nan


# =============================================================================================
# 5.5 applicability domains (box + hull), built locally per (ligand, band)
# =============================================================================================

def _hull_vertices(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    pts = np.unique(points, axis=0)
    if len(pts) == 1:
        return ((float(pts[0, 0]), float(pts[0, 1])),)
    centred = pts - pts.mean(axis=0)
    if np.linalg.matrix_rank(centred, tol=1e-9) < 2:
        u = np.linalg.svd(centred, full_matrices=False)[2][0]
        proj = centred @ u
        lo, hi = pts[int(np.argmin(proj))], pts[int(np.argmax(proj))]
        return ((float(lo[0]), float(lo[1])), (float(hi[0]), float(hi[1])))
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts, qhull_options="QJ")
    return tuple((float(pts[i, 0]), float(pts[i, 1])) for i in hull.vertices)


def _span(values: pd.Series) -> tuple[float, float] | None:
    v = values.dropna()
    return None if v.empty else (float(v.min()), float(v.max()))


def build_domains(records: pd.DataFrame, ligand: str, *, anion: str, diluent_family: str,
                  modifiers: tuple[str, ...]) -> ApplicabilityMapping:
    """One ``ApplicabilityDomain`` per temperature band from the fit-eligible rows of one
    system (``records`` is the flat frame of that system).  Loading fraction =
    ``LOADING_N_PRIOR * mM * 1e-3 / ligand_M`` (ASSUMED O/A 1); hull in (log acid, log ligand)."""
    out = ApplicabilityMapping()
    elig = records[records["fit_eligible"]].copy()
    if elig.empty:
        return out
    elig["_band"] = elig["cond__temperature_C"].map(temperature_band)
    for band, grp in elig.groupby("_band", sort=True):
        la = np.log10(grp["cond__acid_concentration_M"].to_numpy(dtype=float))
        le = np.log10(grp["cond__extractant_concentration_M"].to_numpy(dtype=float))
        mm = grp["cond__metal_concentration_mM"]
        loading = LOADING_N_PRIOR * mm * 1e-3 / grp["cond__extractant_concentration_M"]
        out[f"{ligand}|{band}"] = ApplicabilityDomain(
            anion=anion, diluent_family=diluent_family, modifiers=modifiers,
            log_acid=(float(la.min()), float(la.max())),
            log_ligand={ligand: (float(le.min()), float(le.max()))},
            log_metal_total_mM=_span(np.log10(mm[mm > 0])) if (mm > 0).any() else None,
            loading_fraction=_span(loading), log_complexant=None, oa_ratio=None,
            temperature_C=_span(grp["cond__temperature_C"]), saponification_degree=None,
            hull_vertices=_hull_vertices(np.column_stack([la, le])),
            n_records=int(len(grp)), n_publications=int(grp["publication_id"].nunique()))
    return out


# =============================================================================================
# entries
# =============================================================================================

def _corpus_src(locator: str) -> Source:
    return Source(kind="corpus", locator=locator)


def _unknown(unit: str, *, locator: str | None = None, kind: str = "none",
             note: str = "") -> Sourced:
    return Sourced.unknown(unit, note=note, source=Source(kind=kind, locator=locator))


def _lit(value: float, unit: str, doi: str, locator: str, note: str = "") -> Sourced:
    return Sourced(value, unit, Provenance(ProvStatus.LITERATURE, Source("doi", doi, locator),
                                           note=note))


def _assumed(value: float, unit: str, rng: tuple[float, float], doi: str | None, locator: str,
             note: str = "") -> Sourced:
    return Sourced(value, unit, Provenance(
        ProvStatus.ASSUMED, Source("doi" if doi else "none", doi, locator), range=rng,
        assumed_label="ASSUMED_PLACEHOLDER", note=note))


def _stoichiometry(mechanism: Mechanism, anion: str) -> Stoichiometry:
    if mechanism is Mechanism.CATION_EXCHANGE:
        return Stoichiometry(
            ligands_per_metal=_assumed(
                3.0, "1", (2.0, 3.0), _S2_DOI,
                "mechanism nRE3+ + n(HA2)org <=> nRE(HA2)n,org + nH+; log D vs log[extractant] "
                "slopes 2-3", "dimers per Ln3+; ideal 3; refined by a fit when one exists"),
            protons_released_per_metal=_lit(3.0, "1", _S2_DOI,
                                            "mechanism nRE3+ + n(HA2)org <=> nRE(HA2)n,org + nH+",
                                            "one proton per HA2 unit, three per Ln3+"),
            anions_per_metal=_assumed(0.0, "1", (0.0, 0.0), None,
                                      "cation exchange transports no anion"))
    z_known = anion in ("nitrate", "chloride", "perchlorate")
    return Stoichiometry(
        ligands_per_metal=_unknown(
            "1", locator="set from the fitted n_solvation by g18_fit_dmodels.py"),
        protons_released_per_metal=_lit(
            0.0, "1", _MECHANISM_DOI,
            "solvating mechanism Ln3+ + 3 X- + n L (BRIEF.md section 3b; textbook mechanism, "
            "DOI is the brief's reference for the mechanism class)",
            "no proton release for neutral solvating extractants"),
        anions_per_metal=_lit(3.0, "1", _MECHANISM_DOI, "as above", f"LnX3.L_n, X = {anion}")
        if z_known else _unknown("1", locator=f"anions per metal not fixed for {anion} media"))


def _ligand_spec(name: str, smiles: str, canonical: str, role: str, anion: str,
                 concentration: Sourced) -> LigandSpec:
    if role == "modifier":
        return LigandSpec(name=name, smiles=smiles, canonical_smiles=canonical, role=role,
                          mechanism=None, concentration=concentration, aggregation="monomer",
                          scaffold_id=None, variant_tag=None,
                          stoichiometry=Stoichiometry(_unknown("1"), _unknown("1"), _unknown("1")))
    family, mech, agg = classify_ligand(canonical)
    scaffold = "DGA_core" if family == "diglycolamide" else None
    return LigandSpec(name=name, smiles=smiles, canonical_smiles=canonical, role=role,
                      mechanism=mech, concentration=concentration, aggregation=agg,
                      scaffold_id=scaffold, variant_tag=None,
                      stoichiometry=_stoichiometry(mech, anion))


def _record(row: Mapping[str, Any], ligand: str) -> DistributionRecord:
    def f(x: Any) -> float | None:
        return None if x is None or pd.isna(x) else float(x)

    def s(x: Any) -> str | None:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)

    acid = f(row["cond__acid_concentration_M"])
    ext = f(row["cond__extractant_concentration_M"])
    mm = f(row["cond__metal_concentration_mM"])
    temp = f(row["cond__temperature_C"])
    notes = []
    if mm is not None:
        notes.append("metal concentration semantics ASSUMED initial aqueous; O/A not reported")
    if temp is None:
        notes.append("temperature_assumed_20_30C")
    return DistributionRecord(
        record_id=str(row["safe_exp_id"]), metal=str(row["metal"]), d=float(row["D"]),
        log_d=float(row["log_D"]), acid_nominal_M=acid, acid_eq_M=None, anion_M=acid,
        ligand_M={} if ext is None else {ligand: ext}, complexant_M=None,
        metals_initial_mM={} if mm is None else {str(row["metal"]): mm}, oa_ratio=None,
        temperature_C=temp, contact_time_min=f(row["cond__contact_time_min"]),
        diluent_name=s(row["diluent_name"]), publication_id=s(row["publication_id"]),
        experiment_series_id=s(row["experiment_series_id"]), replicate_id=s(row["replicate_id"]),
        loading_series_id=s(row["loading_series_id"]), is_tracer=bool(row["is_tracer"]),
        fit_eligible=bool(row["fit_eligible"]),
        fit_ineligible_reason=s(row["fit_ineligible_reason"]),
        duplicate_flag=s(row["duplicate_flag"]),
        provenance=Provenance(ProvStatus.MEASURED_CORPUS,
                              Source("corpus", locator="dataset.parquet row",
                                     publication_id=s(row["publication_id"]),
                                     safe_exp_ids=(str(row["safe_exp_id"]),)),
                              note="; ".join(notes)))


def build_entry(sid: str, rows: pd.DataFrame) -> SystemEntry:
    """One corpus ``SystemEntry`` from the flat rows of one system (sorted by record id)."""
    rows = rows.sort_values("safe_exp_id")
    first = rows.iloc[0]
    smiles = str(first["canonical_smiles"])
    ligand = str(first["ligand_name"])
    acid_class = str(first["geom_cond__acid_class"])
    dil_family = str(first["geom_cond__diluent_family"])
    synergists = tuple(first["synergists"])
    modifiers = tuple(first["modifiers"])
    anion = acid_class
    ligands = [_ligand_spec(ligand, smiles, smiles, "extractant", anion, Sourced.unknown(
        "mol/L", note="varies per record; see records[].ligand_M",
        source=_corpus_src("per record: cond__extractant_concentration_M")))]
    for syn in synergists:
        conc = _unknown("mol/L", locator="additive present; concentration not in the bundle")
        ligands.append(_ligand_spec(syn, EXTRACTANT_SMILES[syn],
                                    _canonical(EXTRACTANT_SMILES[syn]), "synergist", anion, conc))
    for mod in modifiers:
        conc = _unknown("mol/L", locator="alcohol additive; volume fraction not in the bundle")
        ligands.append(_ligand_spec(mod, MODIFIER_SMILES[mod], _canonical(MODIFIER_SMILES[mod]),
                                    "modifier", anion, conc))
    family = classify_ligand(smiles)[0]
    acids = sorted(set(ACID_NAMES.get(a, str(a)) for a in rows["acid_name"] if a is not None))
    diluents = sorted(set(d for d in rows["diluent_name"] if d is not None))
    temps = rows["cond__temperature_C"].dropna()
    medium = MediumSpec(
        acid=acids[0] if len(acids) == 1 else "mixed: " + "; ".join(acids), acid_class=acid_class,
        anion=anion, salting_agent=None,
        salting_anion_M=_unknown("mol/L", kind="corpus",
                                 locator="no salting-agent column in the bundle",
                                 note="no salting additive recorded; the cascade uses anion = "
                                      "nominal acid for corpus systems (section 4.3)"),
        ionic_strength_M=_unknown("mol/L"),
        temperature_C=_unknown("Cel", kind="corpus", locator="per record: cond__temperature_C",
                               note=(f"{temps.min():g}-{temps.max():g} C across records; parameter "
                                     "sets are per temperature band") if len(temps) else
                               "no temperature recorded; records assigned to band 20-30C"))
    phase = PhaseBehaviour(
        loc_metal_M=_unknown("mol/L", note="third-phase limits stay null until a DOI is entered"),
        loc_acid_M=_unknown("mol/L"), third_phase_observed=_unknown("1"),
        disengagement_s=_unknown("s"), ligand_loss_mol_per_L_aq=_unknown("mol/L_aq"),
        max_loading_fraction_studied=_unknown(
            "1", kind="corpus", locator="computed by the fitter from loading-series records"),
        regenerability_note="")
    records = tuple(_record(r, ligand) for r in rows.to_dict("records"))
    domains = build_domains(rows, ligand, anion=anion, diluent_family=dil_family,
                            modifiers=modifiers)
    n_pub = rows["publication_id"].nunique()
    metals = sorted(rows["metal"].unique())
    additive_txt = "no additive" if not (synergists or modifiers) else "with " + ", ".join(
        [f"{s} (synergist)" for s in synergists] + [f"{m} (modifier)" for m in modifiers])
    name = (f"{' + '.join([ligand, *synergists])} in {dil_family.replace('_', ' ')}, "
            f"{acid_class} medium, {additive_txt}")
    entry = SystemEntry(
        schema_version=SCHEMA_VERSION, system_id=sid, name=name, family=family, origin="corpus",
        organic_ligands=tuple(ligands), aqueous_complexants=(),
        diluent={"name": diluents[0] if len(diluents) == 1 else None, "family": dil_family,
                 "components": []},
        medium=medium, params={}, phase=phase, stream_records=(), oxidation_state_routes=(),
        direction_prior=None, applicability=domains, records=records,
        notes=f"{len(records)} records, {n_pub} publications, {len(metals)} metals; "
              f"diluents: {', '.join(diluents) if diluents else 'unknown'}.")
    assert system_id(entry) == sid, (sid, system_id(entry), name)
    return entry


# =============================================================================================
# flat mirror and the driver
# =============================================================================================

_RECORD_CSV_COLUMNS = [
    "system_id", "record_id", "metal", "d", "log_d", "acid_nominal_M", "acid_eq_M", "anion_M",
    "ligand_name", "ligand_M", "synergists", "modifiers", "complexant_M", "metal_initial_mM",
    "oa_ratio", "temperature_C", "temperature_band", "contact_time_min", "diluent_name",
    "diluent_family", "acid_class", "anion", "acid_name", "canonical_smiles", "extractant_name",
    "publication_id", "experiment_series_id", "replicate_id", "replicate_group_id",
    "loading_series_id", "is_tracer", "fit_eligible", "fit_ineligible_reason", "duplicate_flag",
]


def _flat_records(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "system_id": frame["system_id"], "record_id": frame["safe_exp_id"], "metal": frame["metal"],
        "d": frame["D"], "log_d": frame["log_D"],
        "acid_nominal_M": frame["cond__acid_concentration_M"], "acid_eq_M": np.nan,
        "anion_M": frame["cond__acid_concentration_M"], "ligand_name": frame["ligand_name"],
        "ligand_M": frame["cond__extractant_concentration_M"],
        "synergists": frame["synergists"].map(";".join),
        "modifiers": frame["modifiers"].map(";".join),
        "complexant_M": np.nan, "metal_initial_mM": frame["cond__metal_concentration_mM"],
        "oa_ratio": np.nan, "temperature_C": frame["cond__temperature_C"],
        "temperature_band": frame["cond__temperature_C"].map(temperature_band),
        "contact_time_min": frame["cond__contact_time_min"],
        "diluent_name": frame["diluent_name"],
        "diluent_family": frame["geom_cond__diluent_family"],
        "acid_class": frame["geom_cond__acid_class"], "anion": frame["geom_cond__acid_class"],
        "acid_name": frame["acid_name"], "canonical_smiles": frame["canonical_smiles"],
        "extractant_name": frame["extractant_name"], "publication_id": frame["publication_id"],
        "experiment_series_id": frame["experiment_series_id"],
        "replicate_id": frame["replicate_id"], "replicate_group_id": frame["replicate_group_id"],
        "loading_series_id": frame["loading_series_id"], "is_tracer": frame["is_tracer"],
        "fit_eligible": frame["fit_eligible"],
        "fit_ineligible_reason": frame["fit_ineligible_reason"],
        "duplicate_flag": frame["duplicate_flag"],
    })[_RECORD_CSV_COLUMNS]
    return out.sort_values(["system_id", "record_id"]).reset_index(drop=True)


def ingest_corpus(systems_dir: str | Path, *, seed: int = 18) -> IngestAudit:
    """The algorithm of DESIGN.md section 4.2 (``seed`` is unused: nothing here is stochastic;
    kept for the interface).  Writes the corpus systems and mirrors into ``systems_dir``."""
    del seed
    systems_dir = Path(systems_dir)
    systems_dir.mkdir(parents=True, exist_ok=True)
    frame, exclusions, qa = load_corpus()
    frame = decode_onehots(frame)
    frame = assign_systems(frame)
    frame = mark_fit_eligibility(frame)
    frame, duplicates = find_duplicates(frame)
    frame, series = assign_loading_series(frame)
    frame["replicate_group_id"], n_rep, sd_med = replicate_groups(frame)

    for old in systems_dir.glob("sys_*.json"):
        old.unlink()
    registry_rows = []
    n_two, n_mod = 0, 0
    for sid, rows in frame.groupby("system_id", sort=True):
        entry = build_entry(sid, rows)
        errors = [v for v in validate_entry(entry) if v.level == "error"]
        if errors:
            raise RuntimeError(f"{sid}: {len(errors)} validator errors: "
                               + "; ".join(f"{v.path}: {v.message}" for v in errors[:5]))
        write_system(entry, systems_dir / f"{sid}.json")
        registry_rows.append(registry_row(entry_to_json(entry)))
        n_two += sum(lig.role == "synergist" for lig in entry.organic_ligands) > 0
        n_mod += any(lig.role == "modifier" for lig in entry.organic_ligands)

    flat = _flat_records(frame)
    flat.to_csv(systems_dir / "corpus_records.csv", index=False, lineterminator="\n")
    exclusions.to_csv(systems_dir / "exclusions.csv", index=False, lineterminator="\n")
    series.to_csv(systems_dir / "series.csv", index=False, lineterminator="\n")
    duplicates.to_csv(systems_dir / "duplicates.csv", index=False, lineterminator="\n")
    with open(systems_dir / "registry.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"schema_version": SCHEMA_VERSION, "systems": registry_rows}, fh, indent=2,
                  sort_keys=True)
        fh.write("\n")
    _write_index(systems_dir, registry_rows)

    reasons = frame["fit_ineligible_reason"].value_counts().to_dict()
    aware = series[series["definition"] == "publication_aware"]
    blind = series[series["definition"] == "publication_blind"]
    slip_rows = int((frame["duplicate_flag"] == "UNIT_SLIP_DUPLICATE").sum())
    tied_all = duplicates.groupby("group_id").size()
    tier_groups = duplicates.groupby("group_id")["tier"].first()
    bands = frame["cond__temperature_C"].map(temperature_band).value_counts().to_dict()
    return IngestAudit(
        bundle_sha256=qa["bundle_sha256"], rows_in=qa["rows_in"],
        todga_name_mismatch_rows=qa["todga_name_mismatch_rows"], sentinel_rows=qa["sentinel_rows"],
        rows_fit_ineligible=int((~frame["fit_eligible"]).sum()),
        n_systems=int(frame["system_id"].nunique()),
        n_records=int(len(frame)), n_publications=int(frame["publication_id"].nunique()),
        n_loading_series_publication_aware=int(len(aware)),
        n_loading_series_publication_blind=int(len(blind)), n_unit_slip_rows=slip_rows,
        n_tied_d_groups=int(len(tied_all)), replicate_groups=n_rep, replicate_sd_median=sd_med,
        rows_after_quarantine=qa["rows_out"],
        fit_ineligible_reasons={str(k): int(v) for k, v in reasons.items()},
        n_unit_slip_groups=int((tier_groups == "unit_slip").sum()),
        n_tied_d_rows=int(len(duplicates)),
        n_tied_d_publications=int(duplicates["publication_id"].nunique()),
        n_tied_d_tier_groups=int((tier_groups == "tied_d").sum()),
        n_tied_d_tier_rows=int((duplicates["tier"] == "tied_d").sum()),
        n_nan_metal_rows=int(frame["cond__metal_concentration_mM"].isna().sum()),
        n_nan_temperature_rows=int(frame["cond__temperature_C"].isna().sum()),
        band_counts={str(k): int(v) for k, v in sorted(bands.items())},
        loading_series_ids=tuple(aware["loading_series_id"]),
        publication_blind_series_ids=tuple(blind["loading_series_id"]),
        n_two_ligand_systems=int(n_two), n_systems_with_modifiers=int(n_mod))


def _write_index(systems_dir: Path, rows: list[dict]) -> None:
    lines = ["# systems/ index (written by gen18proc.ingest / scripts/g18_build_db.py)", "",
             "| system_id | name | family | origin | mechanism | records | publications | metals |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["system_id"]):
        cells = [r[k] for k in ("system_id", "name", "family", "origin", "mechanism",
                                "n_records", "n_publications", "metals")]
        lines.append("| " + " | ".join(str(x) for x in cells) + " |")
    with open(systems_dir / "INDEX.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
