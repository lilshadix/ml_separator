"""``data/load.py`` -- the one loader every gen19 script uses for the multi-metal archive.

Source: ``dataset_all_metals/clean/master_clean.parquet`` (16,770 SAFE records, one row per archive
``exp_id``; schema in ``dataset_all_metals/reports/schema.md``).  The loader never drops a row and
never rewrites an archive column; it only *adds* derived columns, all prefixed ``g19_``:

``g19_publication_id`` / ``g19_publication_status`` / ``g19_publication_refs``
    The publication key used by every leave-publication-out fold (brief V1).  Rule, in order:

    1. ``DOI`` -- :func:`canonical_publication` (the gen6 rule, ``src/lanthanide_separation/gen6/
       provenance.py:148``) over ``doi_all`` with the SAFE self-citation removed.  For bundle rows
       this reproduces the gen6/gen18 ``pub_`` ids (checked in ``tests/test_load.py``).
    2. ``REPORT_SUBSOURCE`` -- no DOI, but a non-DOI reference (CORDIS project page, OSTI, INIS,
       thesis) and an archive ``sub_source_file``: key over the references only (as gen6 did), so
       the whole CORDIS project (ACSEPT, 211267) is ONE publication group -- conservative for
       folds.  The finer per-``ST*.json`` key is ``g19_study_id``.
    3. ``REPORT`` -- non-DOI reference and no sub-source file.
    4. ``UNRESOLVED`` -- nothing identifies the study; every such row gets the single shared key
       ``pub_unresolved`` so a fold can never split an unidentified study across train and test.

``g19_metal`` / ``g19_ox``
    ``metal_symbol`` and ``metal_oxidation_state`` (``UO2+2`` is already expanded to U(VI) by the
    archive).  ``g19_metal_state`` = ``"Nd(III)"``-style label, ``None`` when the state is unknown.

``g19_tier``
    ``TARGET`` rows (a finite ``log_D`` and a resolved metal) that are canonical (not a class A/B
    duplicate member) and model-ready A or B -> ``"MODEL"``; other rows with a target and a metal ->
    ``"TARGET_ONLY"``; everything else -> ``"NO_TARGET"``.  Tiers are labels, not filters.

``g19_bundle_exp_id``
    ``source_record_id`` as the integer ``exp_id`` that the frozen lanthanide bundle encodes as
    ``<stem>_SAFE:<exp_id>`` (joins 1:1; ``overlap.py:131``).

Provenance columns (DOI, source ids, ``g19_publication_*``) must never be model features
(``schema.md`` line 5).  :data:`PROVENANCE_COLUMNS` lists them for the leakage guards.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

from gen19ct import paths

SAFE_DATABASE_SELF_CITATION = "10.1021/jacs.5c19738"
_DOI_PREFIXES = (
    "https://www.doi.org/", "http://www.doi.org/", "https://dx.doi.org/", "http://dx.doi.org/",
    "https://doi.org/", "http://doi.org/", "www.doi.org/", "dx.doi.org/", "doi.org/", "doi:", "doi ",
)
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s,;]+")

PUBLICATION_STATUSES = ("DOI", "REPORT_SUBSOURCE", "REPORT", "UNRESOLVED")
UNRESOLVED_PUBLICATION = "pub_unresolved"

PROVENANCE_COLUMNS: tuple[str, ...] = (
    "canonical_measurement_id", "source_record_id", "raw_row_ids", "export_source_files",
    "export_metals_queried", "export_fanout_size", "representative_raw_row_id", "source_file",
    "source_line_number", "doi_primary", "doi_source_all", "doi_all", "archive_citation_doi",
    "has_source_reference", "reference_other", "entry_author", "addition_date", "publication_year",
    "publication_title", "publication_authors", "data_location", "sub_source_file", "comments_raw",
    "ini_comp_raw", "reference_title", "reference_year", "reference_journal", "reference_authors",
    "reference_publisher", "reference_url", "reference_metadata_source", "doi_primary_corrected",
    "doi_correction_rule", "doi_correction_evidence", "duplicate_group_id", "group_representative_id",
    "identity_hash", "series_id",
    # the rest of the schema.md "Provenance" section: raw strings the parsed features come from
    "aqueous_phase_metals_declared", "n_metals_declared", "n_extractants_declared",
    "extractant_name_raw", "extractant_smiles_raw", "solvent_name_raw", "acid_name_raw",
    "modifier_name_raw", "modifier_concentration_raw", "acid_concentration_organic_raw",
    "phase_ratio_raw", "complexant_name_raw", "complexant_smiles_raw", "complexant_concentration_raw",
    "holdback_smiles_raw", "holdback_concentration_raw", "nitrate_concentration_raw",
    "extractant_concentration_raw", "acid_concentration_raw", "metal_concentration_raw",
    "temperature_raw", "contact_time_raw", "shaking_time_raw", "metal_raw", "metal_oxidation_state_raw",
    "D_raw", "duplicate_class", "duplicate_class_reason", "duplicate_group_size", "is_canonical_row",
    "flags", "rdkit_parse_failures",
    # target-derived quality columns (computed from log D, so they leak the target)
    "in_value_conflict",
    "g19_publication_id", "g19_publication_status", "g19_publication_refs", "g19_study_id",
    "g19_bundle_exp_id",
)


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value if v is not None and str(v) != "nan"]
    return [str(value)]


def canonical_publication(references: Iterable[str]) -> tuple[str | None, tuple[str, ...]]:
    """The gen6 rule: strip URL prefixes, lowercase, pull out embedded DOIs, drop the SAFE
    self-citation, sort, sha1[:10].  ``(None, ())`` when nothing is left."""
    tokens: list[str] = []
    for ref in references:
        for token in re.split(r"[,;]\s*", str(ref)):
            token = token.strip().strip("'\"")
            if not token:
                continue
            lowered = token.lower().strip()
            for prefix in _DOI_PREFIXES:
                if lowered.startswith(prefix):
                    lowered = lowered[len(prefix):]
                    break
            embedded = _DOI_PATTERN.search(lowered)
            if embedded:
                lowered = embedded.group(0)
            lowered = lowered.strip().rstrip("/.,;)")
            if not lowered or SAFE_DATABASE_SELF_CITATION in lowered:
                continue
            tokens.append(lowered)
    if not tokens:
        return None, ()
    unique = tuple(sorted(set(tokens)))
    return "pub_" + hashlib.sha1("|".join(unique).encode()).hexdigest()[:10], unique


def publication_key(doi_all: object, reference_other: object, sub_source_file: object
                    ) -> tuple[str, str, tuple[str, ...]]:
    """``(publication_id, status, references)`` by the four-step rule in the module docstring."""
    pid, refs = canonical_publication(_as_list(doi_all))
    if pid is not None:
        return pid, "DOI", refs
    others = [r for r in _as_list(reference_other) if r.strip()]
    sub = None if sub_source_file is None or (isinstance(sub_source_file, float)
                                              and np.isnan(sub_source_file)) else str(sub_source_file)
    if others:
        pid, refs = canonical_publication(others)
        if pid is not None:
            return pid, ("REPORT_SUBSOURCE" if sub else "REPORT"), refs
    return UNRESOLVED_PUBLICATION, "UNRESOLVED", ()


def study_key(publication_id: str, sub_source_file: object) -> str:
    """A finer key than the publication: the publication plus the archive ``sub_source_file``
    (one CORDIS project fronts ~100 ``ST*.json`` studies).  Equal to ``publication_id`` when
    there is no sub-source file.  Folds group on the *coarser* publication key (conservative);
    the study key is for diagnostics and pair generation only."""
    if sub_source_file is None or (isinstance(sub_source_file, float) and np.isnan(sub_source_file)):
        return publication_id
    return "stu_" + hashlib.sha1(f"{publication_id}@@{sub_source_file}".encode()).hexdigest()[:10]


def assert_archive_unchanged() -> str:
    """Refuse to run on an archive whose master parquet differs from the pinned digest."""
    d = paths.digests(paths.ARCHIVE_MASTER)["sha256"]
    if d != paths.ARCHIVE_MASTER_SHA256:
        raise RuntimeError(f"master_clean.parquet sha256 {d} != pinned {paths.ARCHIVE_MASTER_SHA256}")
    return d


@lru_cache(maxsize=1)
def _load_cached() -> pd.DataFrame:
    assert_archive_unchanged()
    df = pd.read_parquet(paths.ARCHIVE_MASTER)
    keys = [publication_key(a, b, c) for a, b, c in
            zip(df["doi_all"], df["reference_other"], df["sub_source_file"])]
    df["g19_publication_id"] = [k[0] for k in keys]
    df["g19_publication_status"] = [k[1] for k in keys]
    df["g19_publication_refs"] = [" | ".join(k[2]) for k in keys]
    df["g19_study_id"] = [study_key(p, s) for p, s in zip(df["g19_publication_id"], df["sub_source_file"])]
    df["g19_metal"] = df["metal_symbol"]
    df["g19_ox"] = df["metal_oxidation_state"]
    state = []
    for m, o in zip(df["metal_symbol"], df["metal_oxidation_state"]):
        if m is None or (isinstance(m, float) and np.isnan(m)) or pd.isna(o):
            state.append(None)
        else:
            state.append(f"{m}({_roman(int(o))})")
    df["g19_metal_state"] = state
    has_target = df["log_D"].notna() & np.isfinite(df["log_D"].astype(float)) & df["metal_symbol"].notna()
    model = has_target & df["is_canonical_row"].astype(bool) & df["model_readiness"].isin(
        ["A_model_ready", "B_usable_with_caveats"])
    df["g19_tier"] = np.where(model, "MODEL", np.where(has_target, "TARGET_ONLY", "NO_TARGET"))
    df["g19_bundle_exp_id"] = pd.to_numeric(df["source_record_id"], errors="coerce").astype("Int64")
    return df


def _roman(n: int) -> str:
    return {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}.get(n, str(n))


def load_archive(copy: bool = True) -> pd.DataFrame:
    """The full archive with the ``g19_`` columns added.  Returns a copy by default so callers
    cannot mutate the cache."""
    df = _load_cached()
    return df.copy() if copy else df


def load_model_rows(copy: bool = True) -> pd.DataFrame:
    """Rows with ``g19_tier == "MODEL"`` (canonical, model-ready A/B, finite log D, metal)."""
    df = _load_cached()
    out = df[df["g19_tier"] == "MODEL"]
    return out.copy() if copy else out


def bundle_publication_map() -> pd.DataFrame:
    """gen6 provenance for the 5,992 bundle rows: ``safe_exp_id, exp_id, publication_id``."""
    p = pd.read_parquet(paths.GEN6_PROVENANCE, columns=["safe_exp_id", "metal_symbol", "publication_id",
                                                        "publication_references"])
    p["exp_id"] = pd.to_numeric(p["safe_exp_id"].str.split(":").str[-1], errors="coerce").astype("Int64")
    return p
