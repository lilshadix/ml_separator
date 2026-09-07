"""The Gen12 europium cohort: one row per (extractant, condition) cell.

Design decisions, all made before any model was fitted and all recorded in
``DATA_AUDIT.md``:

* **Metal is fixed to Eu.** Every metal-identity column (atomic number,
  lanthanide index, ionic radius) is constant and is therefore *excluded* from
  every feature block rather than passed as a zero-variance column.
* **The target is the bundle's ``log_D``** — base-10 logarithm of the
  distribution ratio, dimensionless.  It is finite on every Eu row; no floor,
  no censoring rule and no transformation is applied.
* **Identity is structural.**  The unit of chemistry is ``canonical_smiles``;
  the extractant *name* is never an identity key, because 16 Eu structures carry
  more than one name and one of them carries 21.
* **The repository's provenance-backed quarantine is applied**: rows whose
  structure, fingerprint and 3D complex are TODGA while the recorded name is a
  different ligand (``lanthanide_separation.pairs.apply_default_quarantine``).
* **Replicates are averaged per cell**, matching gen7-gen11's ``replicate_policy
  = "mean"``, so a condition measured eight times does not enter as an eight-row
  cluster.
* **No minimum-rows filter.**  gen5 measured that ``min_rows = 10`` discarded 37
  of its 47 chemically novel extractants; on a study whose whole subject is
  unseen chemistry that filter would remove the phenomenon.  Few-shot
  eligibility is a *declared sub-cohort*, never a cohort-wide filter.

Nothing in this module reads ``log_D`` to define an identity, a group, a split
or a feature.  ``tests/test_cohort_contract.py`` proves it by rebuilding the
cohort with the target replaced by noise and asserting every non-target column
is unchanged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import paths
from .chemistry import ECFP_COLUMNS, ecfp_clusters, frozen_chemotypes
from lanthanide_separation.levels import (  # noqa: E402
    CONTINUOUS_CONDITION_COLUMNS, condition_labels, series_labels,
)
from lanthanide_separation.pairs import TODGA_SMILES  # noqa: E402

TARGET = "log_D"
METAL = "Eu"

#: Columns that identify a row or its provenance.  Never a feature, asserted.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "row_id", "extractant", "extractant_name", "ecfp_cluster", "chemotype", "chem_family",
    "condition_id", "series_id", "n_replicates", "safe_exp_id_first", "archive_record_ids",
    "doi", "archive_system_class", "archive_duplicate_class", "review_flags",
    "name_conflict", "geometry_ok_any", TARGET,
)
#: Anything matching these may never enter a feature matrix.
FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "log_D", "D_value", "log_SF", "safe_exp_id", "build_id", "geometry_key", "xyz_path",
    "mol2_path", "sample_weight", "asset_index", "doi", "archive_", "review_", "row_id",
    "extractant_name", "_index", "series_id", "condition_id", "chemotype", "ecfp_cluster",
)
#: Metal-identity columns — constant at a single lanthanide, dropped on purpose.
CONSTANT_METAL_COLUMNS: tuple[str, ...] = (
    "Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal", "metal_ox",
    "sample_weight_inv_metal_freq",
)
#: The ten generic RDKit scalars the frozen bundle carries.
PHYSCHEM_COLUMNS: tuple[str, ...] = (
    "MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "NumAromaticRings", "NumAliphaticRings", "RingCount", "FractionCSP3", "MolLogP",
)
#: Complex-specification scalars (denticity, coordination number, stoichiometry).
COMPLEX_SPEC_COLUMNS: tuple[str, ...] = ("DENTATE", "coreCN", "n_ligs", "n_fill")


@dataclass
class Gen12Cohort:
    frame: pd.DataFrame
    blocks: dict[str, tuple[str, ...]]
    audit: dict = field(default_factory=dict)

    def columns(self, *block_names: str) -> tuple[str, ...]:
        out: list[str] = []
        for name in block_names:
            if name not in self.blocks:
                raise KeyError(f"unknown block {name!r}; have {sorted(self.blocks)}")
            out.extend(self.blocks[name])
        seen: dict[str, None] = {}
        for c in out:
            seen.setdefault(c, None)
        return tuple(seen)

    @property
    def fingerprint(self) -> str:
        payload = pd.util.hash_pandas_object(
            self.frame[sorted(self.frame.columns)], index=False).to_numpy().tobytes()
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _donor_census(frame: pd.DataFrame) -> pd.DataFrame:
    """``DONOR_TYPES`` JSON list -> ``donor__<type>`` counts. Target-free."""
    vocab = ["O(amide_carbonyl)", "O(ether)", "N(aromatic)", "N(amine)", "S(donor)",
             "O(hydroxyl)", "O(ester_carbonyl)", "O(carbonyl)", "P(phosphoryl)", "O(phosphoryl)"]
    rows = []
    for raw in frame["DONOR_TYPES"].tolist():
        out = {f"donor__{t}": 0.0 for t in vocab}
        out["donor__n_total"] = np.nan
        items = None
        if isinstance(raw, str):
            try:
                items = json.loads(raw)
            except ValueError:
                items = None
        elif isinstance(raw, (list, tuple, np.ndarray)):
            items = list(raw)
        if items is not None:
            for t in items:
                key = f"donor__{t}"
                if key in out:
                    out[key] += 1.0
            out["donor__n_total"] = float(len(items))
        rows.append(out)
    return pd.DataFrame(rows, index=frame.index)


def _mass_action(frame: pd.DataFrame) -> pd.DataFrame:
    """The extraction mass-action law in the form a tree can split on.

    ``log D = log K_ex + n log[L] + m log[NO3-]``.  The metal terms of gen5's
    block are dropped (constant at Eu); the ligand-stoichiometry products are
    kept because ``n`` tracks denticity and coordination number.
    """
    new: dict[str, np.ndarray] = {}
    logs: dict[str, pd.Series] = {}
    for col in CONTINUOUS_CONDITION_COLUMNS:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        positive = values.where(values > 0)
        if positive.notna().sum() == 0:
            continue
        logs[col] = np.log10(positive)
        new[f"massact__log10_{col}"] = logs[col].to_numpy(dtype=float)
    log_l = logs.get("cond__extractant_concentration_M")
    log_h = logs.get("cond__acid_concentration_M")
    if log_l is not None:
        for col in ("DENTATE", "coreCN"):
            if col in frame.columns:
                new[f"massact__logL_x_{col}"] = (
                    log_l * pd.to_numeric(frame[col], errors="coerce")).to_numpy(dtype=float)
        if log_h is not None:
            new["massact__logL_x_logH"] = (log_l * log_h).to_numpy(dtype=float)
    return pd.DataFrame(new, index=frame.index)


def _review_flags(records: pd.Series) -> pd.Series:
    """Data-quality review categories attached to each cell, as a ``;``-joined string."""
    queue = pd.read_csv(paths.REPO_ROOT / "dataset_all_metals" / "audit" / "manual_review_queue.csv")
    queue = queue.assign(ids=queue["canonical_measurement_ids"].astype(str).str.split(";"))
    long = queue.explode("ids")
    long["record"] = long["ids"].astype(str).str.strip().str.replace("SAE:", "", regex=False)
    by_record: dict[str, set[str]] = {}
    for record, category in zip(long["record"], long["category"]):
        by_record.setdefault(record, set()).add(str(category))
    out = []
    for ids in records:
        flags: set[str] = set()
        for record in str(ids).split(";"):
            flags |= by_record.get(record.strip(), set())
        out.append(";".join(sorted(flags)))
    return pd.Series(out, index=records.index, name="review_flags")


def build_cohort(*, verify_hash: bool = True, apply_quarantine: bool = True) -> Gen12Cohort:
    """Build the frozen Gen12 Eu cohort from the immutable bundle."""
    if verify_hash:
        digest = paths.sha256_of(paths.BUNDLE_PARQUET)
        if digest != paths.BUNDLE_SHA256:
            raise RuntimeError(
                f"bundle SHA-256 drift: {digest} != {paths.BUNDLE_SHA256}. "
                "Every Gen12 number is defined against the pinned bundle; refusing to build.")
    source = pd.read_parquet(paths.BUNDLE_PARQUET)
    audit: dict = {
        "bundle_sha256": paths.BUNDLE_SHA256,
        "bundle_rows": int(len(source)),
        "bundle_columns": int(source.shape[1]),
        "bundle_extractants": int(source["canonical_smiles"].nunique()),
        "bundle_metals": sorted(source["metal"].astype(str).unique()),
    }

    frame = source[source["metal"].astype(str) == METAL].copy()
    audit["eu_rows_raw"] = int(len(frame))
    audit["eu_extractants_raw"] = int(frame["canonical_smiles"].nunique())
    audit["eu_names_raw"] = int(frame["extractant_name"].nunique())
    audit["eu_target_finite"] = int(np.isfinite(frame[TARGET]).sum())

    name_counts = frame.groupby("canonical_smiles")["extractant_name"].nunique()
    audit["structures_with_multiple_names"] = int((name_counts > 1).sum())

    wrong_todga = frame["canonical_smiles"].eq(TODGA_SMILES) & ~frame["extractant_name"].eq("TODGA")
    audit["quarantine_rule"] = "todga_structure_assigned_to_different_extractant"
    audit["quarantine_rows"] = int(wrong_todga.sum())
    audit["quarantine_names"] = sorted(frame.loc[wrong_todga, "extractant_name"].astype(str).unique())
    if apply_quarantine:
        frame = frame.loc[~wrong_todga].copy()
    audit["eu_rows_after_quarantine"] = int(len(frame))

    condition_columns = tuple(c for c in frame.columns if c.startswith("cond__"))
    frame = frame.assign(
        extractant=frame["canonical_smiles"].astype(str),
        condition_id=condition_labels(frame, condition_columns),
        ecfp_cluster=ecfp_clusters(frame),
        series_id=series_labels(
            frame.assign(extractant=frame["canonical_smiles"].astype(str)), condition_columns),
    )
    chem = frozen_chemotypes(frame["extractant"])
    frame = frame.assign(chemotype=chem["chemotype"].to_numpy(),
                         chem_family=chem["chem_family"].to_numpy())

    # ---- descriptors -------------------------------------------------------
    lig2d = pd.read_parquet(paths.LIG2D_PARQUET).drop_duplicates("canonical_smiles")
    lig2d_columns = [c for c in lig2d.columns if c.startswith("lig2d__")]
    frame = frame.merge(lig2d[["canonical_smiles", *lig2d_columns]], on="canonical_smiles",
                        how="left", validate="many_to_one")
    audit["lig2d_coverage"] = float(frame[lig2d_columns[0]].notna().mean())
    frame = pd.concat([frame, _donor_census(frame), _mass_action(frame)], axis=1)

    # ---- provenance / audit columns (never features) -----------------------
    frame["_archive_record"] = frame["safe_exp_id"].astype(str).str.split("SAFE:").str[-1]
    archive = pd.read_parquet(paths.ARCHIVE_MASTER_PARQUET, columns=[
        "source_record_id", "doi_primary_corrected", "system_component_class",
        "duplicate_class", "in_value_conflict"])
    archive = archive.drop_duplicates("source_record_id").set_index("source_record_id")
    joined = archive.reindex(frame["_archive_record"].to_numpy())
    audit["archive_join_rate"] = float(joined["doi_primary_corrected"].notna().mean())
    frame = frame.assign(
        doi=joined["doi_primary_corrected"].to_numpy(),
        archive_system_class=joined["system_component_class"].to_numpy(),
        archive_duplicate_class=joined["duplicate_class"].to_numpy(),
    )
    frame["review_flags"] = _review_flags(frame["_archive_record"]).to_numpy()
    frame["name_conflict"] = frame.groupby("extractant")["extractant_name"].transform("nunique") > 1

    # ---- replicate collapse: one row per (extractant, condition) cell -------
    cell = ["extractant", "condition_id"]
    sizes = frame.groupby(cell)[TARGET].transform("size")
    spread = frame.groupby(cell)[TARGET].transform(lambda s: float(s.max() - s.min()))
    frame = frame.assign(n_replicates=sizes.astype(int), replicate_spread=spread.astype(float))
    frame[TARGET] = frame.groupby(cell)[TARGET].transform("mean")
    aggregated = {
        "safe_exp_id_first": frame.groupby(cell)["safe_exp_id"].transform("first"),
        "archive_record_ids": frame.groupby(cell)["_archive_record"].transform(
            lambda s: ";".join(sorted(set(s.astype(str))))),
        "geometry_ok_any": frame.groupby(cell)["geometry_ok"].transform(
            lambda s: bool(np.asarray(s).any())),
        "review_flags": frame.groupby(cell)["review_flags"].transform(
            lambda s: ";".join(sorted({f for v in s for f in str(v).split(";") if f}))),
        "doi": frame.groupby(cell)["doi"].transform("first"),
    }
    frame = frame.assign(**aggregated)
    audit["cells_before_collapse"] = int(len(frame))
    audit["replicated_cells"] = int(frame.drop_duplicates(cell)["n_replicates"].gt(1).sum())
    frame = frame.drop_duplicates(subset=cell, keep="first").reset_index(drop=True)
    audit["rows"] = int(len(frame))

    frame["row_id"] = [
        hashlib.blake2b(f"{e}|{c}".encode(), digest_size=8).hexdigest()
        for e, c in zip(frame["extractant"], frame["condition_id"])
    ]
    if frame["row_id"].duplicated().any():
        raise AssertionError("row_id is not unique")

    # ---- feature blocks ----------------------------------------------------
    blocks: dict[str, tuple[str, ...]] = {
        "COND": condition_columns,
        "MASSACT": tuple(c for c in frame.columns if c.startswith("massact__")),
        "ECFP": ECFP_COLUMNS,
        "PHYSCHEM": tuple(c for c in PHYSCHEM_COLUMNS if c in frame.columns),
        "DONORS": tuple(c for c in frame.columns if c.startswith("donor__"))
                  + tuple(c for c in COMPLEX_SPEC_COLUMNS if c in frame.columns),
        "LIG2D": tuple(lig2d_columns),
    }
    # Columns with no observed value carry nothing and would only make the imputer warn.
    dropped: dict[str, list[str]] = {}
    for name, cols in list(blocks.items()):
        empty = [c for c in cols if frame[c].isna().all()]
        constant = [c for c in cols if c not in empty and frame[c].nunique(dropna=True) <= 1]
        if empty:
            dropped[name] = empty
        blocks[name] = tuple(c for c in cols if c not in set(empty))
        audit.setdefault("constant_columns_per_block", {})[name] = len(constant)
    audit["all_null_columns_dropped"] = dropped

    for name, cols in blocks.items():
        for column in cols:
            for token in FORBIDDEN_FEATURE_TOKENS:
                if token in column:
                    raise AssertionError(f"forbidden column {column!r} in block {name!r}")
    for column in CONSTANT_METAL_COLUMNS:
        if any(column in cols for cols in blocks.values()):
            raise AssertionError(f"metal-identity column {column!r} reached a feature block")

    keep = list(dict.fromkeys(
        [c for c in IDENTITY_COLUMNS if c in frame.columns]
        + ["replicate_spread"]
        + [c for cols in blocks.values() for c in cols]))
    frame = frame[keep]

    audit.update({
        "extractants": int(frame["extractant"].nunique()),
        "ecfp_clusters": int(frame["ecfp_cluster"].nunique()),
        "chemotypes": int(frame["chemotype"].nunique()),
        "conditions": int(frame["condition_id"].nunique()),
        "series": int(frame["series_id"].nunique()),
        "target_mean": float(frame[TARGET].mean()),
        "target_sd": float(frame[TARGET].std()),
        "target_min": float(frame[TARGET].min()),
        "target_max": float(frame[TARGET].max()),
        "largest_extractant_share": float(frame["extractant"].value_counts(normalize=True).iloc[0]),
        "largest_chemotype_share": float(frame["chemotype"].value_counts(normalize=True).iloc[0]),
        "block_sizes": {k: len(v) for k, v in blocks.items()},
    })
    return Gen12Cohort(frame=frame, blocks=blocks, audit=audit)
