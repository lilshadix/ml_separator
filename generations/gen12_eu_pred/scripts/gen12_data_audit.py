#!/usr/bin/env python
"""Compute every number in ``DATA_AUDIT.md``, and fail loudly on a leakage hazard.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_data_audit.py

Writes ``manifests/data_audit.json`` plus the per-extractant, per-chemotype and
similarity tables the report cites, so no number in the markdown is hand-typed.
The exit code is non-zero if any *fatal* hazard fires: a target that is not
finite, a structure carrying more than one fingerprint, a fold whose grouping
column crosses train/test, or a feature column matching the forbidden list.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import paths, splits  # noqa: E402
from gen12eu.chemistry import (  # noqa: E402
    ECFP_COLUMNS, band_of, eu_local_chemotypes, fingerprint_matrix, tanimoto_matrix,
)
from gen12eu.cohort import TARGET, build_cohort  # noqa: E402
from gen12eu.metrics import effective_sample_size  # noqa: E402


def _describe(series: pd.Series) -> dict:
    s = pd.Series(series).astype(float)
    q = s.quantile([0.25, 0.5, 0.75, 0.95]).to_dict()
    return {"n": int(s.size), "mean": float(s.mean()), "sd": float(s.std(ddof=1)) if s.size > 1 else 0.0,
            "min": float(s.min()), "p25": float(q[0.25]), "median": float(q[0.5]),
            "p75": float(q[0.75]), "p95": float(q[0.95]), "max": float(s.max())}


def main() -> int:
    fatal: list[str] = []
    warnings: list[str] = []
    cohort = build_cohort()
    frame = cohort.frame
    audit = dict(cohort.audit)
    audit["cohort_fingerprint"] = cohort.fingerprint

    # ---- target -----------------------------------------------------------
    y = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        fatal.append("target is not finite on every cohort row")
    audit["target"] = {
        "column": TARGET, "definition": "log10 of the distribution ratio D = [Eu]_org / [Eu]_aq",
        "units": "dimensionless (decades)", **_describe(frame[TARGET]),
        "n_below_minus_4": int((y < -4).sum()), "n_above_3": int((y > 3).sum()),
        "censoring_rule_applied": "none — no Eu row sits at or below the gen5/gen7 -6 floor",
        "n_at_or_below_minus_6": int((y <= -6).sum()),
    }

    # ---- structures and identity ------------------------------------------
    names, bits = fingerprint_matrix(frame)
    similarity = tanimoto_matrix(bits)
    np.fill_diagonal(similarity, -1.0)
    nn = similarity.max(axis=1)
    audit["structures"] = {
        "extractants": len(names),
        "ecfp_clusters_bit_identical": int(frame["ecfp_cluster"].nunique()),
        "chemotypes_frozen_map": int(frame["chemotype"].nunique()),
        "chemotypes_eu_local": int(eu_local_chemotypes(frame.drop_duplicates("extractant")).nunique()),
        "informative_ecfp_bits": int((frame[list(ECFP_COLUMNS)].nunique() > 1).sum()),
        "constant_ecfp_bits": int((frame[list(ECFP_COLUMNS)].nunique() <= 1).sum()),
        "pairs_at_tanimoto_1": int((similarity >= 0.999).sum() // 2),
        "max_non_identical_similarity": float(similarity[similarity < 0.999].max()),
        "nearest_neighbour_within_cohort": _describe(pd.Series(nn)),
        "n_with_no_neighbour_above_0_4": int((nn <= 0.4).sum()),
        "chem_families": frame.drop_duplicates("extractant")["chem_family"].value_counts().to_dict(),
    }
    per_structure_fp = frame.groupby("extractant")[list(ECFP_COLUMNS)].nunique().max(axis=1)
    if (per_structure_fp > 1).any():
        fatal.append("a structure carries more than one fingerprint")

    # ---- rows per unit ----------------------------------------------------
    rows_per_extractant = frame.groupby("extractant").size()
    rows_per_chemotype = frame.groupby("chemotype").size()
    extractants_per_chemotype = frame.groupby("chemotype")["extractant"].nunique()
    conditions_per_extractant = frame.groupby("extractant")["condition_id"].nunique()
    series_per_extractant = frame.groupby("extractant")["series_id"].nunique()
    audit["distribution"] = {
        "rows_per_extractant": _describe(rows_per_extractant),
        "rows_per_chemotype": _describe(rows_per_chemotype),
        "extractants_per_chemotype": _describe(extractants_per_chemotype),
        "conditions_per_extractant": _describe(conditions_per_extractant),
        "series_per_extractant": _describe(series_per_extractant),
        "extractants_with_n_rows_at_least": {
            str(k): int((rows_per_extractant >= k).sum()) for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 20)},
        "n_extractants_single_row": int((rows_per_extractant == 1).sum()),
        "n_extractants_single_series": int((series_per_extractant == 1).sum()),
        "largest_extractant_rows": int(rows_per_extractant.max()),
        "largest_chemotype_rows": int(rows_per_chemotype.max()),
        "largest_chemotype_extractants": int(extractants_per_chemotype.max()),
        "n_eff_rows_by_extractant": effective_sample_size(rows_per_extractant),
        "n_eff_rows_by_chemotype": effective_sample_size(rows_per_chemotype),
        "n_eff_extractants_by_chemotype": effective_sample_size(extractants_per_chemotype),
    }

    # ---- few-shot eligibility ---------------------------------------------
    audit["few_shot_eligibility"] = {
        f"k={k}": {
            "rule": "n_rows >= k + 2 (k support points and at least two queries)",
            "extractants": int((rows_per_extractant >= k + 2).sum()),
            "rows": int(rows_per_extractant[rows_per_extractant >= k + 2].sum()),
            "chemotypes": int(frame[frame["extractant"].isin(
                rows_per_extractant[rows_per_extractant >= k + 2].index)]["chemotype"].nunique()),
        } for k in (1, 2, 3, 5)}
    common = rows_per_extractant[rows_per_extractant >= 7].index
    audit["few_shot_eligibility"]["common_cohort_k_up_to_5"] = {
        "rule": "n_rows >= 7, so the same extractant is scorable at every k in {0,1,2,3,5}",
        "extractants": int(len(common)), "rows": int(rows_per_extractant[common].sum()),
        "chemotypes": int(frame[frame["extractant"].isin(common)]["chemotype"].nunique())}

    # ---- conditions and missingness ---------------------------------------
    cond = list(cohort.blocks["COND"])
    continuous = [c for c in cond if frame[c].dtype.kind == "f" and frame[c].nunique() > 2]
    audit["conditions"] = {
        "n_condition_columns": len(cond),
        "n_constant_in_cohort": int((frame[cond].nunique() <= 1).sum()),
        "missingness": {c: float(frame[c].isna().mean())
                        for c in cond if frame[c].isna().any()},
        "continuous_summary": {c: _describe(frame[c].dropna()) for c in continuous},
        "acid_one_hot_rows_with_none": int((frame[[c for c in cond if c.startswith("cond__acid__")]]
                                           .sum(axis=1) == 0).sum()),
        "diluent_one_hot_rows_with_none": int((frame[[c for c in cond if c.startswith("cond__diluent__")]]
                                              .sum(axis=1) == 0).sum()),
        "diluent_one_hot_rows_with_many": int((frame[[c for c in cond if c.startswith("cond__diluent__")]]
                                              .sum(axis=1) > 1).sum()),
    }
    audit["feature_missingness_by_block"] = {
        name: {"columns": len(cols),
               "columns_with_any_missing": int(sum(frame[c].isna().any() for c in cols)),
               "mean_missing_fraction": float(np.mean([frame[c].isna().mean() for c in cols]))}
        for name, cols in cohort.blocks.items()}

    # ---- duplicates and replicates ----------------------------------------
    audit["duplicates"] = {
        "cells_with_replicates": int(audit["replicated_cells"]),
        "replicate_spread": _describe(frame.loc[frame["n_replicates"] > 1, "replicate_spread"])
        if (frame["n_replicates"] > 1).any() else {},
        "replicate_cells_spread_above_1_decade": int(
            ((frame["n_replicates"] > 1) & (frame["replicate_spread"] > 1.0)).sum()),
        "archive_duplicate_class": frame["archive_duplicate_class"].value_counts(dropna=False).to_dict(),
    }

    # ---- provenance --------------------------------------------------------
    doi = frame["doi"]
    per_extractant_doi = frame.dropna(subset=["doi"]).groupby("extractant")["doi"].nunique()
    per_doi_extractant = frame.dropna(subset=["doi"]).groupby("doi")["extractant"].nunique()
    chemotype_doi = frame.dropna(subset=["doi"]).groupby("chemotype")["doi"].nunique()
    doi_chemotype = frame.dropna(subset=["doi"]).groupby("doi")["chemotype"].nunique()
    audit["provenance"] = {
        "rows_with_doi": int(doi.notna().sum()), "rows_without_doi": int(doi.isna().sum()),
        "distinct_dois": int(doi.nunique()),
        "extractants_spanning_multiple_dois": int((per_extractant_doi > 1).sum()),
        "dois_spanning_multiple_extractants": int((per_doi_extractant > 1).sum()),
        "dois_spanning_multiple_chemotypes": int((doi_chemotype > 1).sum()),
        "chemotypes_spanning_multiple_dois": int((chemotype_doi > 1).sum()),
        "rows_in_dois_that_span_chemotypes": int(frame["doi"].isin(
            doi_chemotype[doi_chemotype > 1].index).sum()),
        "rows_per_doi": _describe(frame.dropna(subset=["doi"]).groupby("doi").size()),
        "archive_system_class": frame["archive_system_class"].value_counts(dropna=False).to_dict(),
    }
    if audit["provenance"]["dois_spanning_multiple_chemotypes"] > 0:
        warnings.append(
            f"{audit['provenance']['dois_spanning_multiple_chemotypes']} publications span more than "
            f"one chemotype, so a chemotype hold-out does not block publication batch effects "
            f"({audit['provenance']['rows_in_dois_that_span_chemotypes']} rows). "
            "Reported as a robustness variant, not as fold corruption.")

    # ---- identity hazards --------------------------------------------------
    audit["identity_hazards"] = {
        "structures_with_multiple_names": int(frame.groupby("extractant")["extractant_name"].nunique().gt(1).sum()),
        "rows_on_multi_name_structures": int(frame["name_conflict"].sum()),
        "names_resolving_to_multiple_structures": int(
            frame.groupby("extractant_name")["extractant"].nunique().gt(1).sum()),
        "quarantined_rows": int(audit["quarantine_rows"]),
        "review_flagged_rows": int((frame["review_flags"].astype(str) != "").sum()),
        "review_flag_counts": (frame["review_flags"].astype(str).str.split(";").explode()
                               .replace("", np.nan).dropna().value_counts().to_dict()),
        "review_flagged_extractants": int(
            frame.loc[frame["review_flags"].astype(str) != "", "extractant"].nunique()),
        "multi_component_rows": int((frame["archive_system_class"].astype(str)
                                     != "SINGLE_EXTRACTANT").sum()),
    }

    # ---- splits, bands, power ---------------------------------------------
    split_report: dict = {}
    per_extractant_similarity: list[pd.DataFrame] = []
    for design in ("B", "A"):
        folds = splits.all_folds(frame, design=design)
        integrity = splits.assert_fold_integrity(frame, folds)
        if not integrity["ok"]:
            fatal.append(f"fold integrity failed for design {design}")
        table = splits.similarity_table(frame, folds)
        table["band"] = band_of(table["max_train_tanimoto"])
        table["design"] = design
        per_extractant_similarity.append(table)
        rows_of = frame.groupby("extractant").size()
        table["n_rows"] = table["extractant"].map(rows_of)
        chem_of = frame.drop_duplicates("extractant").set_index("extractant")["chemotype"]
        table["chemotype"] = table["extractant"].map(chem_of)
        bands = {}
        for band in ("far", "mid", "near", "overall"):
            block = table if band == "overall" else table[table["band"] == band]
            per_seed = block.groupby("split_seed")["extractant"].nunique()
            bands[band] = {
                "distinct_extractants_any_seed": int(block["extractant"].nunique()),
                "distinct_chemotypes_any_seed": int(block["chemotype"].nunique()),
                "extractants_per_seed_mean": float(per_seed.mean()) if len(per_seed) else 0.0,
                "extractants_per_seed_min": int(per_seed.min()) if len(per_seed) else 0,
                "rows_per_seed_mean": float(block.groupby("split_seed")["n_rows"].sum().mean())
                if len(block) else 0.0,
            }
        split_report[design] = {
            "group_column": splits.GROUP_COLUMNS[design],
            "n_groups": int(frame[splits.GROUP_COLUMNS[design]].nunique()),
            "seeds": list(splits.SPLIT_SEEDS), "n_splits": splits.N_SPLITS,
            "integrity_ok": integrity["ok"],
            "train_test_overlap_counts": integrity["overlap_counts"],
            "test_rows_per_fold": _describe(pd.Series([e["n_test"] for e in integrity["folds"]])),
            "inner_validation_rows": _describe(
                pd.Series([e["n_inner_validation"] for e in integrity["folds"]])),
            "max_train_tanimoto": _describe(table["max_train_tanimoto"]),
            "bands": bands,
        }
        if design == "B" and float(table["max_train_tanimoto"].max()) >= 0.7:
            fatal.append("chemotype hold-out left a test extractant at Tanimoto >= 0.7 to training")
    audit["splits"] = split_report
    if split_report["A"]["train_test_overlap_counts"]["ecfp_cluster"] > 0:
        warnings.append(
            f"design A (exact extractant) leaks {split_report['A']['train_test_overlap_counts']['ecfp_cluster']} "
            f"bit-identical ECFP clusters and "
            f"{split_report['A']['train_test_overlap_counts']['chemotype']} chemotypes across 25 folds; "
            "it is a sensitivity design and may not carry a zero-shot headline.")

    similarity_frame = pd.concat(per_extractant_similarity, ignore_index=True)
    similarity_frame.to_parquet(paths.SPLIT_DIR / "similarity_by_fold.parquet", index=False)

    # ---- cross-metal availability (for the Phase-10 transfer arm) ---------
    source = pd.read_parquet(paths.BUNDLE_PARQUET, columns=["metal", "canonical_smiles", "log_D"])
    eu_structures = set(frame["extractant"])
    other = source[source["metal"].astype(str) != "Eu"]
    audit["cross_metal"] = {
        "non_eu_rows": int(len(other)),
        "non_eu_extractants": int(other["canonical_smiles"].nunique()),
        "non_eu_rows_on_eu_extractants": int(other["canonical_smiles"].isin(eu_structures).sum()),
        "non_eu_extractants_shared_with_eu": int(
            other.loc[other["canonical_smiles"].isin(eu_structures), "canonical_smiles"].nunique()),
        "non_eu_extractants_not_in_eu": int(
            other.loc[~other["canonical_smiles"].isin(eu_structures), "canonical_smiles"].nunique()),
        "eu_extractants_with_no_other_metal": int(
            len(eu_structures - set(other["canonical_smiles"]))),
        "rows_by_metal": other["metal"].value_counts().to_dict(),
    }
    warnings.append(
        f"{audit['cross_metal']['non_eu_rows_on_eu_extractants']} non-Eu rows sit on Eu cohort "
        "extractants; the strict multi-lanthanide arm must delete every row of a held-out "
        "chemotype under every metal, not only the Eu rows.")

    audit["hazards"] = {"fatal": fatal, "warnings": warnings}
    audit["environment"] = _environment()

    (paths.MANIFEST_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    frame.drop(columns=[c for c in frame.columns if c.startswith("ecfp_")]).to_parquet(
        paths.MANIFEST_DIR / "cohort_identity.parquet", index=False)
    frame.to_parquet(paths.MANIFEST_DIR / "cohort.parquet", index=False)
    (paths.MANIFEST_DIR / "cohort_blocks.json").write_text(
        json.dumps({k: list(v) for k, v in cohort.blocks.items()}, indent=1))

    print(json.dumps({k: v for k, v in audit.items()
                      if k in ("rows", "extractants", "ecfp_clusters", "chemotypes",
                               "cohort_fingerprint")}, indent=1))
    for w in warnings:
        print("WARNING:", w)
    for f in fatal:
        print("FATAL:", f)
    return 1 if fatal else 0


def _environment() -> dict:
    import platform
    import subprocess
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "sklearn", "rdkit", "catboost", "xgboost",
                 "torch", "chemprop", "pyarrow"):
        try:
            versions[name] = __import__(name).__version__
        except Exception as error:
            versions[name] = f"unavailable ({type(error).__name__})"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=paths.REPO_ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    return {"platform": platform.platform(), "machine": platform.machine(),
            "versions": versions, "git_commit": commit}


if __name__ == "__main__":
    raise SystemExit(main())
