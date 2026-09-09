"""The Gen13 separation cohort: one row per multi-metal *cell*.

A cell is one extractant measured under one exactly matched condition vector for
two or more lanthanides.  The cell is the unit on which a separation factor is
observed, because ``log SF(A/B) = log D(A) - log D(B)`` is only meaningful when
both metals share the extractant, the aqueous and organic phases, the acid, the
temperature and the contact time.

Design decisions, made before any model was fitted:

* **Condition key is the exact 64-column bundle key with NaN treated as a
  value.**  gen2-gen4 required every condition to be *present* and that single
  filter collapsed 190 extractants to 34.  Two rows recorded with the same
  missing fields come from the same table of the same paper; treating "not
  reported" as a matchable value recovers 90 extractants with at least one
  multi-metal cell (``scripts/audit/gen13_sf_data_audit_part1.out.txt``).  The
  gen6 provenance ``publication_id`` is added to the key so that a cell can
  never mix two publications: 7 of 521 exact cells did, and they are split.
* **Replicates are averaged per (cell, metal)** and their count and spread are
  kept, because the replicate noise floor (median within-replicate sd 0.237) is
  of the same size as an adjacent-lanthanide separation factor and every
  headline table must print it.
* **The target is the centred lanthanide-axis curve.**  Per cell we keep the 14
  per-metal ``log_D`` means; every pairwise ``log SF`` is a difference of two of
  them.  Nothing else about the level enters the target.
* **The repository's TODGA quarantine is applied** (``pairs.apply_default_quarantine``
  semantics: structure is TODGA, name is not).  Three sentinel rows at
  ``log_D <= -6`` are dropped as in gen5-gen12.
* **Chemotype and ECFP cluster come from the frozen gen6 map** over all 190
  bundle structures, so the chemotype hold-out is the same object as in
  gen6-gen12.
* **No minimum-cell filter.**  Extractants with one two-metal cell stay.

Nothing here reads ``log_D`` to define an identity, a group, a split or a feature.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import paths
from .metals import LANTHANIDES, ATOMIC_NUMBER

TARGET = "log_D"
TODGA_SMILES = "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"
LOG_D_FLOOR = -6.0          # gen5-gen12 sentinel rule: three rows at log D <= -6 are dropped
MIN_METALS_PER_CELL = 2

#: Continuous condition columns whose *presence* differs between otherwise identical
#: series.  ``RELAXED`` drops them from the key (sensitivity only).
RELAXABLE_CONDITION_COLUMNS: tuple[str, ...] = (
    "cond__contact_time_min", "cond__metal_concentration_mM",
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "cell_id", "extractant", "extractant_name", "ecfp_cluster", "chemotype", "chem_family",
    "publication_id", "experiment_series_ids", "condition_key", "n_metals", "metals",
    "n_rows", "n_replicated_metals", "replicate_sd_median", "safe_exp_ids",
)
FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "log_D", "logD__", "nrep__", "repsd__", "D_value", "log_SF", "safe_exp_id", "build_id",
    "geometry_key", "xyz_path", "mol2_path", "sample_weight", "asset_index", "publication_id",
    "series_id", "cell_id", "chemotype", "ecfp_cluster", "extractant_name", "condition_key",
)
#: Exact column names that are identities, not features (substring rules would hit
#: ``cond__extractant_concentration_M``).
FORBIDDEN_FEATURE_EXACT: tuple[str, ...] = ("extractant", "metals", "chem_family")


@dataclass(frozen=True)
class Gen13Cohort:
    frame: pd.DataFrame
    condition_columns: tuple[str, ...]
    key_mode: str
    audit: dict = field(default_factory=dict)

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(f"logD__{m}" for m in LANTHANIDES)

    @property
    def target_matrix(self) -> np.ndarray:
        """Cells x 14 lanthanides, NaN where the metal was not measured in the cell."""
        return self.frame[list(self.target_columns)].to_numpy(dtype=float)

    def fingerprint(self) -> str:
        cols = sorted(c for c in self.frame.columns if c != "safe_exp_ids")
        payload = pd.util.hash_pandas_object(self.frame[cols], index=False).to_numpy().tobytes()
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


CONDITION_KEY_SIGNIFICANT_DIGITS = 9


def _condition_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Exact key with NaN as a value; floats are rounded to 9 significant digits so that
    representation noise (10.0 vs 9.9999999999) cannot split a cell."""
    parts = []
    for col in columns:
        v = frame[col]
        if pd.api.types.is_float_dtype(v):
            v = v.map(lambda x: "NA" if pd.isna(x) else format(float(x), f".{CONDITION_KEY_SIGNIFICANT_DIGITS}g"))
        else:
            v = v.astype(object).where(v.notna(), "NA").astype(str)
        parts.append(v)
    return pd.concat(parts, axis=1).agg("|".join, axis=1)


def load_bundle() -> pd.DataFrame:
    paths.assert_bundle_unchanged()
    return pd.read_parquet(paths.BUNDLE_PARQUET)


def apply_quarantine(bundle: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """TODGA structure under a foreign name, and the log D sentinel floor."""
    name = bundle["extractant_name"].astype(str).str.upper().str.replace(" ", "", regex=False)
    todga_mismatch = (bundle["canonical_smiles"] == TODGA_SMILES) & (name != "TODGA")
    sentinel = bundle[TARGET] <= LOG_D_FLOOR
    kept = bundle.loc[~todga_mismatch & ~sentinel].copy()
    audit = {
        "rows_in": int(len(bundle)),
        "todga_name_mismatch_rows": int(todga_mismatch.sum()),
        "sentinel_rows": int(sentinel.sum()),
        "rows_out": int(len(kept)),
    }
    return kept, audit


def build_cohort(key_mode: str = "exact", *, require_publication: bool = True) -> Gen13Cohort:
    """Build the cell table.  ``key_mode`` is ``exact`` (primary), ``relaxed`` or ``series`` (sensitivities)."""
    if key_mode not in ("exact", "relaxed", "series"):
        raise ValueError(f"key_mode must be exact, relaxed or series, got {key_mode!r}")
    bundle = load_bundle()
    bundle, quarantine_audit = apply_quarantine(bundle)
    cond = [c for c in bundle.columns if c.startswith("cond__")]
    key_cols = [c for c in cond if key_mode == "exact" or c not in RELAXABLE_CONDITION_COLUMNS]

    provenance = pd.read_parquet(paths.GEN6_PROVENANCE_PARQUET)[
        ["safe_exp_id", "publication_id", "experiment_series_id"]]
    bundle = bundle.merge(provenance, on="safe_exp_id", how="left", validate="one_to_one")
    if require_publication and bundle["publication_id"].isna().any():
        raise RuntimeError("publication_id missing for some bundle rows")

    bundle["condition_key"] = _condition_key(bundle, key_cols)
    if key_mode == "series":
        # sensitivity: rows with identical recorded conditions but different reconstructed
        # experiment series are kept apart (the audit's 'hidden axis' replicates)
        bundle["condition_key"] = bundle["condition_key"] + "|series=" + bundle["experiment_series_id"].astype(str)
    bundle["cell_key"] = (bundle["canonical_smiles"] + "@@" + bundle["publication_id"].astype(str)
                          + "@@" + bundle["condition_key"])
    bundle["Z"] = bundle["metal"].map(ATOMIC_NUMBER)
    if bundle["Z"].isna().any():
        raise RuntimeError(f"unknown metal symbols: {sorted(set(bundle.loc[bundle.Z.isna(), 'metal']))}")

    per = bundle.groupby(["cell_key", "metal"], sort=True)[TARGET].agg(["mean", "size", "std"])
    n_metals = per.groupby(level=0).size()
    keep = n_metals[n_metals >= MIN_METALS_PER_CELL].index
    per = per.loc[keep]
    means = per["mean"].unstack("metal").reindex(columns=LANTHANIDES)
    counts = per["size"].unstack("metal").reindex(columns=LANTHANIDES)
    sds = per["std"].unstack("metal").reindex(columns=LANTHANIDES)
    means.columns = [f"logD__{m}" for m in LANTHANIDES]
    counts.columns = [f"nrep__{m}" for m in LANTHANIDES]
    sds.columns = [f"repsd__{m}" for m in LANTHANIDES]

    cells = bundle[bundle["cell_key"].isin(keep)]
    first = cells.sort_values("safe_exp_id").groupby("cell_key", sort=True).first()
    meta = pd.DataFrame(index=first.index)
    meta["extractant"] = first["canonical_smiles"]
    meta["extractant_name"] = first["extractant_name"]
    meta["publication_id"] = first["publication_id"]
    meta["condition_key"] = first["condition_key"]
    meta["experiment_series_ids"] = cells.groupby("cell_key")["experiment_series_id"].agg(
        lambda s: ";".join(sorted(set(map(str, s)))))
    meta["safe_exp_ids"] = cells.groupby("cell_key")["safe_exp_id"].agg(
        lambda s: ";".join(sorted(map(str, s))))
    meta["n_rows"] = cells.groupby("cell_key").size()
    meta["n_metals"] = means.notna().sum(axis=1)
    meta["metals"] = means.apply(lambda r: ";".join(m for m in LANTHANIDES if pd.notna(r[f"logD__{m}"])), axis=1)
    meta["n_replicated_metals"] = (counts > 1).sum(axis=1)
    meta["replicate_sd_median"] = sds.median(axis=1)
    # per-cell condition values: identical across the cell's rows for key columns; columns
    # relaxed out of the key are averaged when they agree and set to NaN when they differ
    condition_values = first[cond].apply(pd.to_numeric, errors="coerce")
    relaxed_cols = [col for col in cond if col not in key_cols]
    if relaxed_cols:
        agg = cells.groupby("cell_key")[relaxed_cols].agg(
            lambda v: float(pd.to_numeric(v, errors="coerce").mean()) if pd.to_numeric(v, errors="coerce").nunique(dropna=True) <= 1 else np.nan)
        condition_values[relaxed_cols] = agg.reindex(condition_values.index)
    # Architector recipe columns vary per row for a few extractants: take the mode per cell
    recipe = cells.groupby("cell_key")[["DENTATE", "coreCN"]].agg(
        lambda s: pd.to_numeric(s, errors="coerce").mode().iloc[0] if pd.to_numeric(s, errors="coerce").notna().any() else np.nan)
    recipe.columns = ["recipe__DENTATE", "recipe__coreCN"]

    chem = pd.read_parquet(paths.CHEMISTRY_MAP_PARQUET).set_index("extractant")
    for col, src in (("chemotype", "chem__supercluster"), ("ecfp_cluster", "chem__ecfp_cluster"), ("chem_family", "chem__family")):
        mapped = meta["extractant"].map(chem[src])
        if mapped.isna().any():
            missing = sorted(set(meta.loc[mapped.isna(), "extractant"]))
            raise RuntimeError(f"{len(missing)} extractants missing {src} in the frozen chemistry map")
        meta[col] = mapped.astype(str)

    frame = pd.concat([meta, means, counts, sds, condition_values, recipe], axis=1)
    frame = frame.sort_values(["extractant", "publication_id", "condition_key"]).reset_index(drop=True)
    frame.insert(0, "cell_id", [hashlib.blake2b(k.encode(), digest_size=8).hexdigest()
                                for k in (frame["extractant"] + "@@" + frame["publication_id"].astype(str)
                                          + "@@" + frame["condition_key"])])
    if frame["cell_id"].duplicated().any():
        raise RuntimeError("cell_id collision")

    audit = {
        "key_mode": key_mode,
        "key_columns": key_cols,
        "quarantine": quarantine_audit,
        "n_cells": int(len(frame)),
        "n_extractants": int(frame["extractant"].nunique()),
        "n_chemotypes": int(frame["chemotype"].nunique()),
        "n_ecfp_clusters": int(frame["ecfp_cluster"].nunique()),
        "n_publications": int(frame["publication_id"].nunique()),
        "rows_in_cells": int(frame["n_rows"].sum()),
        "metals_per_cell": {int(k): int(v) for k, v in frame["n_metals"].value_counts().sort_index().items()},
        "cells_with_all_14": int((frame["n_metals"] == 14).sum()),
        "replicate_sd_median_per_cell_metal": float(np.nanmedian(sds.to_numpy(dtype=float))),
        "n_replicated_cell_metal": int(np.isfinite(sds.to_numpy(dtype=float)).sum()),
        "replicate_sd_median_over_replicated_cells": float(frame["replicate_sd_median"].median()),
        "largest_chemotype_cells": int(frame["chemotype"].value_counts().iloc[0]),
        "largest_chemotype_extractants": int(frame.groupby("chemotype")["extractant"].nunique().max()),
    }
    return Gen13Cohort(frame=frame, condition_columns=tuple(cond), key_mode=key_mode, audit=audit)


def write_cohort(cohort: Gen13Cohort, name: str = "cohort") -> dict:
    out = paths.MANIFEST_DIR / f"{name}_{cohort.key_mode}.parquet"
    cohort.frame.to_parquet(out, index=False)
    record = {"path": str(out.relative_to(paths.REPO_ROOT)), "fingerprint": cohort.fingerprint(),
              "sha256": paths.sha256_of(out), **cohort.audit}
    with open(paths.MANIFEST_DIR / f"{name}_{cohort.key_mode}.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record
