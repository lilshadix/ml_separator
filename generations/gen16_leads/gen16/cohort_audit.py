"""L6 — cohort audit: which extractants gen13's cohort construction throws away, and why.

This module NEVER writes the frozen cohort.  It re-implements
``gen13_separation/gen13sep/cohort.py::build_cohort`` step for step, with one switch per
design decision, so that each rule can be relaxed **one at a time** and the resulting
cohort described.  ``build_cohort_switched()`` with all switches at their frozen defaults
must reproduce the frozen build exactly: 521 cells / 90 extractants / 45 chemotypes and
fingerprint ``4c3c6628ea0be949``.  ``verify_frozen()`` asserts that.

The frozen rules, in the order ``build_cohort`` applies them:

1. ``todga_quarantine``  — rows whose structure is TODGA but whose name is not are dropped
   (129 rows).
2. ``sentinel_drop``     — rows with ``log_D <= -6`` are dropped (3 rows).
3. ``require_publication`` — every bundle row must carry a gen6 ``publication_id``.
4. ``publication_in_key`` — ``publication_id`` is part of the cell key, so a cell can never
   mix two publications.
5. ``key_mode``          — the condition key: ``exact`` (all 64 ``cond__`` columns, NaN as a
   value), ``relaxed`` (drop contact time and metal concentration), ``series`` (exact plus
   the reconstructed experiment series id), or ``none`` (no condition columns at all — the
   maximal relaxation, chemically wrong but it bounds what the key can possibly cost).
6. ``min_metals``        — a cell needs ``MIN_METALS_PER_CELL = 2`` lanthanides.
7. ``require_chemotype`` — every kept extractant must appear in the frozen gen6 chemistry
   map (``chem__supercluster``).  ``build_cohort`` raises if one does not.

Nothing here reads ``log_D`` to define an identity, a group or a split.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GEN16_ROOT = Path(__file__).resolve().parents[1]          # gen16_leads/
REPO_ROOT = GEN16_ROOT.parent
_GEN13 = REPO_ROOT / "gen13_separation"
if str(_GEN13) not in sys.path:
    sys.path.insert(0, str(_GEN13))

from gen13sep import paths                                    # noqa: E402
from gen13sep.cohort import (                                 # noqa: E402
    CONDITION_KEY_SIGNIFICANT_DIGITS, LOG_D_FLOOR, MIN_METALS_PER_CELL,
    RELAXABLE_CONDITION_COLUMNS, TARGET, TODGA_SMILES, _condition_key,
)
from gen13sep.metals import ATOMIC_NUMBER, LANTHANIDES        # noqa: E402
from gen13sep.metrics import effective_sample_size            # noqa: E402

FROZEN_FINGERPRINT = "4c3c6628ea0be949"
FROZEN_CELLS, FROZEN_EXTRACTANTS, FROZEN_CHEMOTYPES = 521, 90, 45

KEY_MODES = ("exact", "relaxed", "series", "none")

#: The exclusion rules, in the order they are tested for a given extractant.
RULE_ORDER = (
    "todga_name_quarantine",
    "sentinel_log_D_le_-6",
    "missing_publication_id",
    "publication_in_key_splits_metals",
    "exact_condition_key_splits_metals",
    "missing_chemotype_in_gen6_map",
    "single_metal_in_bundle",
    "kept",
)


# --------------------------------------------------------------------------------------
# the switched rebuild
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Switches:
    """One flag per frozen design decision.  All-default == the frozen build."""
    todga_quarantine: bool = True
    sentinel_drop: bool = True
    publication_in_key: bool = True
    key_mode: str = "exact"
    require_chemotype: bool = True
    min_metals: int = MIN_METALS_PER_CELL

    def label(self) -> str:
        d = Switches()
        diffs = [f"{f}={getattr(self, f)!r}" for f in
                 ("todga_quarantine", "sentinel_drop", "publication_in_key",
                  "key_mode", "require_chemotype", "min_metals")
                 if getattr(self, f) != getattr(d, f)]
        return "FROZEN" if not diffs else "+".join(diffs)


@dataclass(frozen=True)
class SwitchedCohort:
    frame: pd.DataFrame
    switches: Switches
    audit: dict

    def fingerprint(self) -> str:
        """Identical recipe to ``Gen13Cohort.fingerprint``."""
        cols = sorted(c for c in self.frame.columns if c != "safe_exp_ids")
        payload = pd.util.hash_pandas_object(self.frame[cols], index=False).to_numpy().tobytes()
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


def load_bundle_raw() -> pd.DataFrame:
    """The frozen bundle, sha256-checked, with gen6 provenance merged on ``safe_exp_id``."""
    paths.assert_bundle_unchanged()
    bundle = pd.read_parquet(paths.BUNDLE_PARQUET)
    provenance = pd.read_parquet(paths.GEN6_PROVENANCE_PARQUET)[
        ["safe_exp_id", "publication_id", "experiment_series_id"]]
    return bundle.merge(provenance, on="safe_exp_id", how="left", validate="one_to_one")


def _apply_quarantine(bundle: pd.DataFrame, sw: Switches) -> tuple[pd.DataFrame, dict]:
    name = bundle["extractant_name"].astype(str).str.upper().str.replace(" ", "", regex=False)
    todga_mismatch = (bundle["canonical_smiles"] == TODGA_SMILES) & (name != "TODGA")
    sentinel = bundle[TARGET] <= LOG_D_FLOOR
    drop = pd.Series(False, index=bundle.index)
    if sw.todga_quarantine:
        drop |= todga_mismatch
    if sw.sentinel_drop:
        drop |= sentinel
    kept = bundle.loc[~drop].copy()
    audit = {"rows_in": int(len(bundle)),
             "todga_name_mismatch_rows": int(todga_mismatch.sum()),
             "sentinel_rows": int(sentinel.sum()),
             "rows_out": int(len(kept))}
    return kept, audit


def cell_keys(bundle: pd.DataFrame, sw: Switches) -> pd.DataFrame:
    """Add ``condition_key`` and ``cell_key`` to a (already quarantined) bundle copy."""
    cond = [c for c in bundle.columns if c.startswith("cond__")]
    if sw.key_mode == "none":
        key_cols: list[str] = []
    elif sw.key_mode == "exact":
        key_cols = list(cond)
    else:
        # NOTE: build_cohort's expression is
        #   [c for c in cond if key_mode == "exact" or c not in RELAXABLE_CONDITION_COLUMNS]
        # so ``series`` uses the RELAXED column set (contact time and metal concentration
        # are dropped) and then appends the experiment-series id.  It is not exact+series.
        key_cols = [c for c in cond if c not in RELAXABLE_CONDITION_COLUMNS]
    if key_cols:
        bundle["condition_key"] = _condition_key(bundle, key_cols)
    else:
        bundle["condition_key"] = "ALL"
    if sw.key_mode == "series":
        bundle["condition_key"] = (bundle["condition_key"] + "|series="
                                   + bundle["experiment_series_id"].astype(str))
    pub = bundle["publication_id"].astype(str) if sw.publication_in_key else "ALLPUB"
    bundle["cell_key"] = bundle["canonical_smiles"] + "@@" + pub + "@@" + bundle["condition_key"]
    bundle.attrs["key_cols"] = key_cols
    bundle.attrs["cond_cols"] = cond
    return bundle


def build_cohort_switched(sw: Switches = Switches(), *, bundle: pd.DataFrame | None = None
                          ) -> SwitchedCohort:
    """``gen13sep.cohort.build_cohort`` with one switch per rule.  Defaults == frozen."""
    if sw.key_mode not in KEY_MODES:
        raise ValueError(f"key_mode must be one of {KEY_MODES}, got {sw.key_mode!r}")
    raw = load_bundle_raw() if bundle is None else bundle.copy()
    b, quarantine_audit = _apply_quarantine(raw, sw)
    if b["publication_id"].isna().any():
        raise RuntimeError("publication_id missing for some bundle rows")
    b = cell_keys(b, sw)
    cond, key_cols = b.attrs["cond_cols"], b.attrs["key_cols"]

    b["Z"] = b["metal"].map(ATOMIC_NUMBER)
    if b["Z"].isna().any():
        raise RuntimeError(f"unknown metal symbols: {sorted(set(b.loc[b.Z.isna(), 'metal']))}")

    per = b.groupby(["cell_key", "metal"], sort=True)[TARGET].agg(["mean", "size", "std"])
    n_metals = per.groupby(level=0).size()
    keep = n_metals[n_metals >= sw.min_metals].index
    per = per.loc[keep]
    means = per["mean"].unstack("metal").reindex(columns=LANTHANIDES)
    counts = per["size"].unstack("metal").reindex(columns=LANTHANIDES)
    sds = per["std"].unstack("metal").reindex(columns=LANTHANIDES)
    means.columns = [f"logD__{m}" for m in LANTHANIDES]
    counts.columns = [f"nrep__{m}" for m in LANTHANIDES]
    sds.columns = [f"repsd__{m}" for m in LANTHANIDES]

    cells = b[b["cell_key"].isin(keep)]
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
    meta["metals"] = means.apply(
        lambda r: ";".join(m for m in LANTHANIDES if pd.notna(r[f"logD__{m}"])), axis=1)
    meta["n_replicated_metals"] = (counts > 1).sum(axis=1)
    meta["replicate_sd_median"] = sds.median(axis=1)
    meta["n_publications_in_cell"] = cells.groupby("cell_key")["publication_id"].nunique()

    condition_values = first[cond].apply(pd.to_numeric, errors="coerce")
    relaxed_cols = [col for col in cond if col not in key_cols]
    if relaxed_cols:
        agg = cells.groupby("cell_key")[relaxed_cols].agg(
            lambda v: float(pd.to_numeric(v, errors="coerce").mean())
            if pd.to_numeric(v, errors="coerce").nunique(dropna=True) <= 1 else np.nan)
        condition_values[relaxed_cols] = agg.reindex(condition_values.index)
    recipe = cells.groupby("cell_key")[["DENTATE", "coreCN"]].agg(
        lambda s: pd.to_numeric(s, errors="coerce").mode().iloc[0]
        if pd.to_numeric(s, errors="coerce").notna().any() else np.nan)
    recipe.columns = ["recipe__DENTATE", "recipe__coreCN"]

    chem = pd.read_parquet(paths.CHEMISTRY_MAP_PARQUET).set_index("extractant")
    for col, src in (("chemotype", "chem__supercluster"), ("ecfp_cluster", "chem__ecfp_cluster"),
                     ("chem_family", "chem__family")):
        mapped = meta["extractant"].map(chem[src])
        if mapped.isna().any():
            missing = sorted(set(meta.loc[mapped.isna(), "extractant"]))
            if sw.require_chemotype:
                raise RuntimeError(f"{len(missing)} extractants missing {src} "
                                   "in the frozen chemistry map")
            mapped = mapped.where(mapped.notna(),
                                  "unmapped__" + meta["extractant"].map(
                                      lambda s: hashlib.blake2b(s.encode(), digest_size=4).hexdigest()))
        meta[col] = mapped.astype(str)

    frame = pd.concat([meta, means, counts, sds, condition_values, recipe], axis=1)
    frame = frame.sort_values(["extractant", "publication_id", "condition_key"]).reset_index(drop=True)
    frame.insert(0, "cell_id", [hashlib.blake2b(k.encode(), digest_size=8).hexdigest()
                                for k in (frame["extractant"] + "@@"
                                          + frame["publication_id"].astype(str) + "@@"
                                          + frame["condition_key"])])
    if frame["cell_id"].duplicated().any():
        raise RuntimeError("cell_id collision")

    audit = {
        "switches": sw.label(),
        "key_mode": sw.key_mode,
        "n_key_columns": len(key_cols),
        "quarantine": quarantine_audit,
        "n_cells": int(len(frame)),
        "n_extractants": int(frame["extractant"].nunique()),
        "n_chemotypes": int(frame["chemotype"].nunique()),
        "n_ecfp_clusters": int(frame["ecfp_cluster"].nunique()),
        "n_publications": int(frame["publication_id"].nunique()),
        "rows_in_cells": int(frame["n_rows"].sum()),
        "cells_with_all_14": int((frame["n_metals"] == 14).sum()),
        "cells_mixing_publications": int((frame["n_publications_in_cell"] > 1).sum()),
        "kish_n_eff_chemotypes_by_extractant": kish_by_extractant(frame),
        "kish_n_eff_chemotypes_by_cell": kish_by_cell(frame),
    }
    return SwitchedCohort(frame=frame, switches=sw, audit=audit)


# --------------------------------------------------------------------------------------
# Kish effective sample size — exactly as gen13 computes it
# --------------------------------------------------------------------------------------
def kish_by_extractant(frame: pd.DataFrame) -> float:
    """gen13's published 11.7: ``effective_sample_size`` over the number of distinct
    extractants per chemotype (``scripts/g13_analysis.py``)."""
    counts = frame.groupby("chemotype")["extractant"].nunique()
    return float(effective_sample_size(counts.to_numpy()))


def kish_by_cell(frame: pd.DataFrame) -> float:
    """The same statistic over chemotype *cell* counts (a different, smaller number)."""
    counts = frame["chemotype"].value_counts()
    return float(effective_sample_size(counts.to_numpy()))


# --------------------------------------------------------------------------------------
# per-extractant exclusion tracing
# --------------------------------------------------------------------------------------
def trace_exclusions(bundle: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per bundle extractant, with the FIRST frozen rule that excludes it.

    The trace follows ``build_cohort``'s own order, so the reported rule is the rule that
    actually does the excluding; the boolean ``would_enter_if_<rule>_relaxed`` columns say
    whether relaxing that one rule alone would let the extractant in.
    """
    raw = load_bundle_raw() if bundle is None else bundle.copy()
    name = raw["extractant_name"].astype(str).str.upper().str.replace(" ", "", regex=False)
    raw["_todga_mismatch"] = (raw["canonical_smiles"] == TODGA_SMILES) & (name != "TODGA")
    raw["_sentinel"] = raw[TARGET] <= LOG_D_FLOOR

    chem = pd.read_parquet(paths.CHEMISTRY_MAP_PARQUET).set_index("extractant")

    def n_cells_ge2(sub: pd.DataFrame, sw: Switches) -> int:
        if sub.empty:
            return 0
        k = cell_keys(sub.copy(), sw)
        g = k.groupby(["cell_key", "metal"]).size().groupby(level=0).size()
        return int((g >= 2).sum())

    rows = []
    for smi, sub in raw.groupby("canonical_smiles", sort=True):
        surviving = sub.loc[~sub["_todga_mismatch"] & ~sub["_sentinel"]]
        rec = {
            "extractant": smi,
            "extractant_name": str(sub["extractant_name"].iloc[0]),
            "n_rows_bundle": int(len(sub)),
            "n_lanthanides_bundle": int(sub["metal"].nunique()),
            "n_lanthanides_after_quarantine": int(surviving["metal"].nunique()),
            "n_rows_todga_quarantined": int(sub["_todga_mismatch"].sum()),
            "n_rows_sentinel": int(sub["_sentinel"].sum()),
            "n_missing_publication_id": int(sub["publication_id"].isna().sum()),
            "n_publications": int(sub["publication_id"].nunique(dropna=True)),
            "in_chem_map": bool(smi in chem.index),
            "chemotype": str(chem["chem__supercluster"].get(smi, "unmapped")),
            "chem_family": str(chem["chem__family"].get(smi, "unmapped")),
        }
        rec["n_cells_frozen"] = n_cells_ge2(surviving, Switches())
        rec["n_cells_no_pub_in_key"] = n_cells_ge2(surviving, Switches(publication_in_key=False))
        rec["n_cells_key_relaxed"] = n_cells_ge2(surviving, Switches(key_mode="relaxed"))
        rec["n_cells_key_series"] = n_cells_ge2(surviving, Switches(key_mode="series"))
        rec["n_cells_key_none"] = n_cells_ge2(surviving, Switches(key_mode="none"))
        rec["n_cells_no_quarantine"] = n_cells_ge2(sub, Switches())

        # first rule that excludes, in build_cohort's own order
        if rec["n_cells_frozen"] > 0:
            rule = "kept"
        elif rec["n_lanthanides_bundle"] < 2:
            rule = "single_metal_in_bundle"
        elif rec["n_lanthanides_after_quarantine"] < 2 and rec["n_rows_todga_quarantined"] > 0:
            rule = "todga_name_quarantine"
        elif rec["n_lanthanides_after_quarantine"] < 2 and rec["n_rows_sentinel"] > 0:
            rule = "sentinel_log_D_le_-6"
        elif rec["n_missing_publication_id"] > 0:
            rule = "missing_publication_id"
        elif rec["n_cells_no_pub_in_key"] > 0:
            rule = "publication_in_key_splits_metals"
        else:
            rule = "exact_condition_key_splits_metals"
        if rule != "kept" and not rec["in_chem_map"]:
            rule = "missing_chemotype_in_gen6_map"
        rec["exclusion_rule"] = rule
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["kept"] = out["exclusion_rule"] == "kept"
    return out.sort_values(["exclusion_rule", "extractant"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# ECFP4 Tanimoto to the kept cohort
# --------------------------------------------------------------------------------------
def morgan_bits(smiles_list, radius: int = 2, n_bits: int = 2048):
    """RDKit Morgan (ECFP4) bit vectors; returns (list of ExplicitBitVect, list of failures)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, bad = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(None)
            bad.append(s)
        else:
            fps.append(gen.GetFingerprint(mol))
    return fps, bad


def nn_tanimoto(query_smiles, reference_smiles) -> pd.DataFrame:
    """Nearest-neighbour ECFP4 Tanimoto of each query to the reference set."""
    from rdkit import DataStructs
    qfp, qbad = morgan_bits(list(query_smiles))
    rfp, rbad = morgan_bits(list(reference_smiles))
    ref_ok = [(s, f) for s, f in zip(reference_smiles, rfp) if f is not None]
    rows = []
    for s, f in zip(query_smiles, qfp):
        if f is None:
            rows.append({"extractant": s, "max_tanimoto_to_kept": np.nan,
                         "nn_kept_extractant": None})
            continue
        sims = DataStructs.BulkTanimotoSimilarity(f, [x[1] for x in ref_ok])
        j = int(np.argmax(sims))
        rows.append({"extractant": s, "max_tanimoto_to_kept": float(sims[j]),
                     "nn_kept_extractant": ref_ok[j][0]})
    df = pd.DataFrame(rows)
    df.attrs["rdkit_failures"] = sorted(set(qbad) | set(rbad))
    return df


def single_linkage_clusters(smiles_list, threshold: float = 0.7) -> dict:
    """Single-linkage Tanimoto clustering of a SMILES set at ``threshold`` (the frozen
    chemotype recipe).  Returns {smiles: cluster index}."""
    from rdkit import DataStructs
    smis = list(smiles_list)
    fps, _ = morgan_bits(smis)
    ok = [i for i, f in enumerate(fps) if f is not None]
    parent = {i: i for i in ok}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ii, i in enumerate(ok):
        if ii + 1 >= len(ok):
            break
        others = ok[ii + 1:]
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in others])
        for j, s in zip(others, sims):
            if s >= threshold:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[rb] = ra
    return {smis[i]: find(i) for i in ok}


# --------------------------------------------------------------------------------------
def verify_frozen() -> dict:
    """All-default switches must reproduce the frozen cohort exactly."""
    c = build_cohort_switched(Switches())
    got = {"n_cells": c.audit["n_cells"], "n_extractants": c.audit["n_extractants"],
           "n_chemotypes": c.audit["n_chemotypes"], "fingerprint": None}
    # the frozen frame does not carry n_publications_in_cell; drop it for the fingerprint
    frame = c.frame.drop(columns=["n_publications_in_cell"])
    cols = sorted(x for x in frame.columns if x != "safe_exp_ids")
    payload = pd.util.hash_pandas_object(frame[cols], index=False).to_numpy().tobytes()
    got["fingerprint"] = hashlib.blake2b(payload, digest_size=8).hexdigest()
    ok = (got["n_cells"] == FROZEN_CELLS and got["n_extractants"] == FROZEN_EXTRACTANTS
          and got["n_chemotypes"] == FROZEN_CHEMOTYPES
          and got["fingerprint"] == FROZEN_FINGERPRINT)
    got["matches_frozen"] = bool(ok)
    got["kish_by_extractant"] = c.audit["kish_n_eff_chemotypes_by_extractant"]
    got["kish_by_cell"] = c.audit["kish_n_eff_chemotypes_by_cell"]
    return got
