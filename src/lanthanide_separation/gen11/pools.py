"""Turning the archive into a per-fold auxiliary pool a model is allowed to see.

This is where :mod:`.overlap`'s relationships become an actual set of admissible
row ids, and it is deliberately the only path by which an auxiliary row can
reach :mod:`.transfer`.

The step that deserves explanation is the chemotype assignment.  gen10's folds
are single-linkage Tanimoto clusters at 0.7 over the cohort's 152 structures, so
"is this auxiliary ligand inside a held-out chemotype?" is not answerable by a
lookup — most auxiliary structures were never clustered at all, and 53 of them
are absent from the frozen bundle entirely.  The rule used here is the strict
one:

    an auxiliary structure is treated as belonging to every cohort chemotype
    containing a structure it resembles at Tanimoto >= 0.7.

It errs toward exclusion (a structure that bridges two chemotypes is excluded
when *either* is held out), which is the safe direction for a leakage filter,
and it makes "genuinely new chemistry" a measured property — a structure below
0.7 to every cohort ligand is in no chemotype and is never excluded on chemotype
grounds, because there is no held-out cluster it could be leaking.

The auxiliary series namespace (``aux:``) is applied here rather than in the
featurizer so that every consumer of a pool gets it, including the audits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..gen7.harness import DEFAULT_SEEDS, build_folds
from ..levels import TANIMOTO_CLUSTER_THRESHOLD
from .arms import AuxiliaryPool
from .overlap import OverlapMap, POLICIES, RELATIONSHIPS, apply_policy, cohort_identity, relationship_masks


#: An ECFP *bit* column, as opposed to ``ecfp_cluster``, which is identity.
_ECFP_BIT = re.compile(r"ecfp_\d+")


def _bits(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return np.ascontiguousarray(frame[list(columns)].to_numpy(dtype=np.int8))


def chemotype_assignment(cohort_frame: pd.DataFrame, aux: pd.DataFrame, *,
                         threshold: float = TANIMOTO_CLUSTER_THRESHOLD) -> pd.DataFrame:
    """For each auxiliary structure: the cohort chemotypes it is >= ``threshold`` to.

    Returns one row per distinct auxiliary structure with ``max_tanimoto`` (the
    chemotype-distance quantity the composition tables report) and
    ``chemotypes`` (a tuple, possibly empty).  Computed on the ECFP columns the
    featurizer produced, which are the cohort's own bits for the 190 shared
    structures — verified by the featurizer's reproduction test, so a Tanimoto
    computed here is the same number gen10 would have computed.
    """
    # ``ecfp_cluster`` is an *identity* column that shares the block's prefix; a
    # naive prefix scan picks it up and then the shapes disagree by one.
    ecfp = [c for c in cohort_frame.columns if _ECFP_BIT.fullmatch(c)]
    missing = [c for c in ecfp if c not in aux.columns]
    if missing:
        raise SystemExit(f"auxiliary frame lacks {len(missing)} ecfp columns; "
                         "chemotype assignment would be silently wrong")

    cohort_unique = cohort_frame.drop_duplicates("extractant")
    cohort_bits = _bits(cohort_unique, ecfp).astype(np.int32)
    cohort_chemotype = cohort_unique["tanimoto_cluster"].astype(str).to_numpy()

    aux_unique = aux.drop_duplicates("extractant")
    aux_bits = _bits(aux_unique, ecfp).astype(np.int32)

    inter = aux_bits @ cohort_bits.T
    denom = aux_bits.sum(axis=1)[:, None] + cohort_bits.sum(axis=1)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        tanimoto = np.where(denom > 0, inter / denom, 0.0)

    records = []
    for i, smiles in enumerate(aux_unique["extractant"].astype(str).to_numpy()):
        row = tanimoto[i]
        linked = sorted(set(cohort_chemotype[row >= threshold]))
        records.append({
            "extractant": smiles,
            "max_tanimoto": float(row.max()) if row.size else 0.0,
            "nearest_chemotype": cohort_chemotype[int(row.argmax())] if row.size else None,
            "chemotypes": tuple(linked),
            "n_chemotypes_linked": len(linked),
            "is_new_chemistry": len(linked) == 0,
        })
    return pd.DataFrame.from_records(records)


def namespace_auxiliary(aux: pd.DataFrame) -> pd.DataFrame:
    """Give auxiliary rows cohort-shaped identities that cannot collide.

    Two jobs, both done here so every consumer of a pool gets them.

    *Namespacing.*  ``series_id`` is the grouping root for every curve and
    ``row_id`` keys the design-relative features, so both are prefixed ``aux:``.
    Without it an auxiliary point silently joins a cohort titration.

    *Naming.*  The archive calls a structure ``extractant_primary_smiles``; the
    cohort calls it ``extractant``, and the curve, weighting and chemotype code
    all read the cohort's name.  ``ecfp_cluster`` is derived the way the cohort
    derives it — from the fingerprint bits, not from the SMILES — so two distinct
    SMILES with identical ECFP land in one cluster here exactly as they do there,
    which is what makes the one-cluster-one-vote weighting mean the same thing on
    both sides of the concatenation.
    """
    out = aux.copy()
    # The un-namespaced archive id is KEPT, because the leakage rule and the curve
    # rule need opposite things from it and satisfying one silently broke the
    # other: prefixing ``series_id`` made SAME_SERIES compare "aux:X" against "X",
    # so the guard matched nothing while 2,505 auxiliary rows genuinely shared an
    # archive series with a cohort row.  ``archive_series_id`` is what leakage
    # compares; ``series_id`` is what curves group on.
    out["archive_series_id"] = out["series_id"].astype(str)
    out["series_id"] = "aux:" + out["series_id"].astype(str)
    out["row_id"] = "aux:" + out["source_record_id"].astype(str)
    if "extractant" not in out.columns and "extractant_primary_smiles" in out.columns:
        out["extractant"] = out["extractant_primary_smiles"].astype(str)
    bits = [c for c in out.columns if _ECFP_BIT.fullmatch(c)]
    if "ecfp_cluster" not in out.columns and bits:
        from ..levels import ecfp_cluster_labels
        # Prefixed, so the cohort's own clusters keep exactly the weights gen10
        # computed for them: an auxiliary row never joins — and so never dilutes —
        # a cohort ligand's single vote.
        out["ecfp_cluster"] = "aux:" + ecfp_cluster_labels(out, bits).astype(str)
    if "metal_Z" not in out.columns and "Atomic Number_metal" in out.columns:
        out["metal_Z"] = out["Atomic Number_metal"].astype(float)
    return out


def prepare_auxiliary(aux_features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Make auxiliary rows the *same kind of object* as cohort training rows.

    Two corrections, both found by adversarially verifying the featurizer, and
    both of which would otherwise bias every arm in the direction of a spurious
    effect.

    **Granularity.**  The cohort is one row per (extractant, condition, metal)
    cell with replicates averaged (``replicate_policy="mean"``); the featurizer
    emits one row per archive record.  Left alone, 5,438 auxiliary rows cover only
    ~5,100 cells, and a single Th cell enters the fit 24 times — 24x the weight a
    cohort cell carries, with a within-cell spread of up to 5.3 log units.  The
    same reduction is therefore applied here, using the cohort's own
    ``condition_labels``, so one auxiliary cell is one training unit.

    **Ionic radius.**  The featurizer fills a missing radius with the element's
    modal value, which gives 947 rows a number the archive explicitly declines to
    state (``ionic_radius_status`` is ``unknown_oxidation_state`` or
    ``requires_curation``).  §7 forbids exactly this.  The archive's own value and
    status are joined back on and the fabricated entries are returned to NaN,
    where the missingness indicator can carry them honestly.
    """
    from ..levels import condition_labels
    from .overlap import load_archive

    audit: dict = {"input_rows": int(len(aux_features))}
    frame = aux_features.copy()
    frame["source_record_id"] = frame["source_record_id"].astype(str)

    archive = load_archive()[["source_record_id", "ionic_radius_cn8_A", "ionic_radius_status"]]
    frame = frame.merge(archive, on="source_record_id", how="left", validate="many_to_one")

    fabricated = frame["ionic_radius_cn8_A"].isna() & frame["Ionic Radius_metal"].notna()
    audit["ionic_radius_fabricated_rows_nulled"] = int(fabricated.sum())
    audit["ionic_radius_status_when_nulled"] = (
        frame.loc[fabricated, "ionic_radius_status"].value_counts().to_dict())
    frame.loc[fabricated, "Ionic Radius_metal"] = np.nan

    condition_columns = tuple(c for c in frame.columns if c.startswith("cond__"))
    frame["condition_id"] = condition_labels(frame, condition_columns).to_numpy()
    cell = ["extractant_primary_smiles", "condition_id", "metal_symbol"]
    sizes = frame.groupby(cell, dropna=False)["log_D"].transform("size")
    frame["log_D"] = frame.groupby(cell, dropna=False)["log_D"].transform("mean")
    frame["n_replicates"] = sizes.astype(int)
    reduced = frame.drop_duplicates(subset=cell, keep="first").reset_index(drop=True)

    audit.update({
        "cells": int(len(reduced)),
        "rows_collapsed": int(len(frame) - len(reduced)),
        "cells_with_replicates": int((reduced["n_replicates"] > 1).sum()),
        "largest_cell": int(reduced["n_replicates"].max()),
        "replicate_policy": "mean (the cohort's own policy)",
    })
    return reduced, audit


@dataclass(frozen=True)
class PoolBuild:
    pool: AuxiliaryPool
    assignment: pd.DataFrame
    survival: pd.DataFrame
    relationship_counts: pd.DataFrame
    prepare_audit: dict = field(default_factory=dict)


def build_pool(cohort, overlap: OverlapMap, aux_features: pd.DataFrame, *,
               policy: str = "HEADLINE",
               seeds: Sequence[int] = DEFAULT_SEEDS) -> PoolBuild:
    """Per-fold admissible auxiliary ids under ``policy``, with the audit tables.

    The returned ``survival`` table is the number the arms depend on: if a fold
    admits no auxiliary rows for an arm, that arm has nothing to learn from and
    its "no effect" result means something different from a measured null.
    """
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}; have {sorted(POLICIES)}")
    frame = cohort.frame
    prepared, prepare_audit = prepare_auxiliary(aux_features)
    aux = namespace_auxiliary(prepared)
    assignment = chemotype_assignment(frame, aux)
    # Every auxiliary structure gets an entry, including the genuinely new ones
    # whose entry is empty.  An absent key and an empty one mean opposite things
    # to a leakage filter, and conflating them is what let 8,615 rows through.
    chemotype_of = {record["extractant"]: tuple(record["chemotypes"])
                    for record in assignment.to_dict("records")}

    identity = cohort_identity(frame, overlap)
    safe: dict[tuple[int, int], frozenset[str]] = {}
    survival_rows, relationship_rows = [], []

    for seed in seeds:
        for fold in build_folds(frame, seed):
            held_row_ids = set(frame.iloc[fold.test_index]["row_id"].astype(str))
            held = identity[identity["row_id"].isin(held_row_ids)]
            masks = relationship_masks(aux, held, chemotype_of_smiles=chemotype_of)
            keep = apply_policy(masks, policy)
            ids = frozenset(aux.loc[keep, "source_record_id"].astype(str))
            safe[(int(seed), int(fold.fold))] = ids

            for relationship in RELATIONSHIPS:
                relationship_rows.append({
                    "split_seed": int(seed), "fold": int(fold.fold),
                    "relationship": relationship,
                    "n_excluded": int(masks[relationship].sum()),
                })
            kept = aux.loc[keep]
            survival_rows.append({
                "split_seed": int(seed), "fold": int(fold.fold), "policy": policy,
                "n_admissible": int(keep.sum()),
                "n_pool": int(len(aux)),
                **{f"n_{category}": int((kept["metal_category"] == category).sum())
                   for category in sorted(aux["metal_category"].dropna().unique())},
            })

    pool = AuxiliaryPool(features=aux, safe_ids=safe, policy=policy)
    return PoolBuild(pool=pool, assignment=assignment,
                     survival=pd.DataFrame(survival_rows),
                     relationship_counts=pd.DataFrame(relationship_rows),
                     prepare_audit=prepare_audit)


def write_pool_audit(build: PoolBuild, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    build.assignment.assign(
        chemotypes=build.assignment["chemotypes"].map(lambda t: "|".join(t))
    ).to_csv(out_dir / "chemotype_assignment.csv", index=False)
    build.survival.to_csv(out_dir / "pool_survival.csv", index=False)
    build.relationship_counts.to_csv(out_dir / "pool_relationship_counts.csv", index=False)
    summary = {
        "policy": build.pool.policy,
        **{f"prepare_{k}": v for k, v in build.prepare_audit.items()},
        "pool_rows": int(len(build.pool.features)),
        "distinct_structures": int(build.assignment["extractant"].nunique()),
        "structures_new_chemistry": int(build.assignment["is_new_chemistry"].sum()),
        "median_max_tanimoto": float(build.assignment["max_tanimoto"].median()),
        "min_admissible_over_folds": int(build.survival["n_admissible"].min()),
        "max_admissible_over_folds": int(build.survival["n_admissible"].max()),
        "mean_admissible_over_folds": float(build.survival["n_admissible"].mean()),
    }
    (out_dir / "pool_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
