"""Every way an archive record can be related to a gen10 cohort row.

This is the module the rest of gen11 is not allowed to bypass.  The multi-metal
archive is a *superset* of the frozen bundle — all 5,992 frozen rows are in it,
joining 1:1 on ``exp_id`` with ``log_D`` agreeing to 5.6e-14 — so "train on the
new dataset" without this file means training on the gen10 test set.

The relationships, from fatal to benign:

``EXACT``
    the same archive record.  A frozen row *is* an archive record; auxiliary
    data must never contain one.
``DUPLICATE_EQUIVALENT``
    a different record in the same ``duplicate_group_id``.  The archive folds
    only classes A and B (same conditions, same value, same or copied
    provenance) and keeps every value conflict as a separate observation, so a
    class-D record is a byte-equal restatement of a held-out measurement.
``SAME_SERIES``
    a different point of the same titration.  gen10 already forbids a curve
    straddling a fold; an auxiliary row from a held-out curve reintroduces it.
``SAME_CHEMOTYPE``
    a different ligand inside the held-out Tanimoto cluster.  gen10's folds are
    grouped on exactly this, so an auxiliary row here contradicts the premise
    of the benchmark ("unseen chemistry"), *even when the metal differs*.
``SAME_LIGAND``
    the held-out ligand itself, measured with another metal.  A strict subset
    of SAME_CHEMOTYPE, tracked separately because it answers a different and
    genuinely interesting question — "does an Am titration of this ligand
    predict its Eu behaviour?" — which is a k-shot question, not a zero-shot
    one, and must not be reported as transfer.
``SAME_PUBLICATION``
    same DOI.  Not leakage in the mechanical sense, but gen10 Phase 8 showed
    publication blocking widens intervals ~40 %, so it is a robustness variant.

The default policy (:data:`HEADLINE_POLICY`) excludes EXACT, DUPLICATE_EQUIVALENT,
SAME_SERIES and SAME_CHEMOTYPE.  It is deliberately the strictest reading that
still leaves an experiment: an auxiliary row survives only if its ligand is in
no held-out chemotype of that fold, which is the same rule the fold applies to
its own training rows.  Every looser policy exists here too, so the cost of the
strictness is measurable rather than assumed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_DIR = REPO_ROOT / "dataset_all_metals"
ARCHIVE_CLEAN = ARCHIVE_DIR / "clean" / "master_clean.parquet"
FROZEN_DATASET = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"

#: Columns of the archive gen11 ever reads.  Listing them is not tidiness: the
#: provenance group must never reach a model, and an explicit allow-list makes
#: an accidental ``doi`` feature a KeyError rather than a silent 0.02 gain.
ARCHIVE_IDENTITY_COLUMNS: tuple[str, ...] = (
    "canonical_measurement_id", "source_record_id", "duplicate_group_id",
    "group_representative_id", "duplicate_class", "is_canonical_row",
    "doi_primary_corrected", "series_id", "model_readiness", "is_model_ready",
    "has_target", "structures_resolved", "in_value_conflict",
    "metal_symbol", "metal_category", "is_lanthanide", "metal_oxidation_state",
    "atomic_number", "lanthanide_index", "ionic_radius_cn8_A", "ionic_radius_status",
    "extractant_system_key", "extractant_primary_smiles", "extractant_primary_name",
    "extractant_primary_concentration_M", "n_chemically_active_components",
    "n_organic_extractants", "system_component_class",
    "acid_primary", "acid_anion", "acid_concentration_M", "acid_concentration_organic_M",
    "solvent_primary", "solvent_key", "solvent_n_components",
    "modifier_name", "modifier_concentration_M",
    "complexant_name", "complexant_smiles_canonical", "complexant_concentration_M",
    "holdback_smiles_canonical", "holdback_concentration_M",
    "metal_concentration_M", "temperature_C", "contact_time_min", "shaking_time_min",
    "phase_ratio_org_aq", "log_D", "flags",
)

#: Relationship labels, ordered from fatal to benign.
RELATIONSHIPS: tuple[str, ...] = (
    "EXACT", "DUPLICATE_EQUIVALENT", "SAME_SERIES", "SAME_CHEMOTYPE",
    "SAME_LIGAND", "SAME_PUBLICATION",
)

#: What the headline arms exclude from auxiliary training, per fold.
HEADLINE_POLICY: tuple[str, ...] = (
    "EXACT", "DUPLICATE_EQUIVALENT", "SAME_SERIES", "SAME_CHEMOTYPE",
)

#: Named exclusion policies.  ``PUBLICATION_BLOCKED`` is the §13 robustness
#: variant; ``PERMISSIVE_DIAGNOSTIC`` is §19's "intentionally less strict"
#: comparison and is a diagnostic only — it may never carry a headline number.
POLICIES: Mapping[str, tuple[str, ...]] = {
    "HEADLINE": HEADLINE_POLICY,
    "PUBLICATION_BLOCKED": HEADLINE_POLICY + ("SAME_PUBLICATION",),
    "LIGAND_ONLY": ("EXACT", "DUPLICATE_EQUIVALENT", "SAME_SERIES", "SAME_LIGAND"),
    "PERMISSIVE_DIAGNOSTIC": ("EXACT", "DUPLICATE_EQUIVALENT"),
}


def _sha16(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The frozen side: cohort row -> the archive records it was averaged from
# --------------------------------------------------------------------------- #

def frozen_row_map() -> pd.DataFrame:
    """One row per frozen bundle row, carrying the cohort ``row_id`` it feeds.

    Reproduces :func:`levels.build_level_dataset`'s identity arithmetic rather
    than trusting a re-derivation: ``condition_id`` is the sha1 of the ``cond__*``
    vector and ``row_id`` the sha1 of ``extractant|condition_id|metal_symbol``,
    both exactly as the builder writes them.  The mapping is verified against the
    real cohort in :func:`assert_frozen_map_covers_cohort`, so a drift in either
    formula fails loudly instead of quietly mis-labelling leakage.
    """
    from ..levels import condition_labels

    frame = pd.read_parquet(FROZEN_DATASET, columns=None)
    cond = tuple(c for c in frame.columns if c.startswith("cond__"))
    if not cond:
        raise ValueError("frozen bundle lacks cond__* columns")
    extractant = frame["canonical_smiles"].astype(str)
    metal = frame["metal_symbol"].astype(str)
    condition_id = condition_labels(frame, cond)
    row_id = [_sha16(f"{e}|{c}|{m}") for e, c, m in zip(extractant, condition_id, metal)]
    exp_id = frame["safe_exp_id"].astype(str).str.split("SAFE:").str[-1]
    return pd.DataFrame({
        "safe_exp_id": frame["safe_exp_id"].astype(str),
        "exp_id": exp_id,
        "queried_metal": frame["safe_exp_id"].astype(str).str.split("_SAFE:").str[0],
        "extractant": extractant,
        "metal_symbol": metal,
        "condition_id": condition_id.to_numpy(),
        "row_id": row_id,
        "log_D": frame["log_D"].astype(float).to_numpy(),
    })


def assert_frozen_map_covers_cohort(mapping: pd.DataFrame, cohort_frame: pd.DataFrame) -> dict:
    """Every cohort row must be reachable from the bundle, and vice versa.

    The cohort is the bundle after a ``log_D > -6`` floor, a ``>= 3 cells per
    extractant`` filter and replicate averaging, so bundle rows may be dropped —
    but no cohort ``row_id`` may be unexplained, and the averaged target must
    reproduce.  Both directions are checked because a one-sided check would pass
    on a mapping that invented ids.
    """
    cohort_ids = set(cohort_frame["row_id"].astype(str))
    mapped_ids = set(mapping["row_id"])
    missing = cohort_ids - mapped_ids
    if missing:
        raise SystemExit(
            f"{len(missing)} cohort row_ids have no bundle source; identity formula drifted")
    grouped = mapping[mapping["row_id"].isin(cohort_ids)].groupby("row_id")["log_D"].mean()
    target = cohort_frame.set_index(cohort_frame["row_id"].astype(str))["log_D"].astype(float)
    delta = (grouped - target.reindex(grouped.index)).abs()
    worst = float(delta.max())
    if worst > 1e-9:
        raise SystemExit(f"replicate-averaged target does not reproduce: max |delta| {worst:.3g}")
    return {
        "cohort_rows": int(len(cohort_ids)),
        "bundle_rows": int(len(mapping)),
        "bundle_rows_feeding_cohort": int(mapping["row_id"].isin(cohort_ids).sum()),
        "bundle_rows_dropped_by_cohort_filters": int((~mapping["row_id"].isin(cohort_ids)).sum()),
        "max_abs_target_delta": worst,
    }


# --------------------------------------------------------------------------- #
# The archive side
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def load_archive() -> pd.DataFrame:
    """The audited multi-metal archive, restricted to the columns gen11 may read."""
    frame = pd.read_parquet(ARCHIVE_CLEAN, columns=list(ARCHIVE_IDENTITY_COLUMNS))
    frame["source_record_id"] = frame["source_record_id"].astype(str)
    return frame


@dataclass(frozen=True)
class OverlapMap:
    """The full relationship table between the archive and the frozen cohort."""

    archive: pd.DataFrame
    frozen: pd.DataFrame
    #: archive ``source_record_id`` -> cohort ``row_id`` for the 5,992 frozen rows
    exact: pd.DataFrame
    audit: dict

    @property
    def auxiliary_candidates(self) -> pd.DataFrame:
        """Archive records that are not themselves a frozen bundle row."""
        return self.archive[~self.archive["source_record_id"].isin(set(self.exact["source_record_id"]))]


def build_overlap_map(cohort_frame: pd.DataFrame) -> OverlapMap:
    """Join the archive to the frozen cohort at every level the brief names."""
    archive = load_archive().copy()
    frozen = frozen_row_map()
    coverage = assert_frozen_map_covers_cohort(frozen, cohort_frame)

    known = set(archive["source_record_id"])
    unmatched = set(frozen["exp_id"]) - known
    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} frozen bundle rows are absent from the archive; "
            "the archive is supposed to be a superset")

    exact = frozen.merge(
        archive[["source_record_id", "duplicate_group_id", "doi_primary_corrected",
                 "series_id", "metal_symbol", "log_D"]],
        left_on="exp_id", right_on="source_record_id", how="inner",
        suffixes=("_frozen", "_archive"), validate="one_to_one")
    if len(exact) != len(frozen):
        raise SystemExit(f"exact join lost rows: {len(exact)} != {len(frozen)}")
    metal_agreement = float((exact["metal_symbol_frozen"] == exact["metal_symbol_archive"]).mean())
    target_delta = float((exact["log_D_frozen"] - exact["log_D_archive"]).abs().max())
    if metal_agreement < 1.0:
        raise SystemExit(f"metal disagreement on the exact join: {metal_agreement:.4f}")
    if target_delta > 1e-9:
        raise SystemExit(f"target disagreement on the exact join: {target_delta:.3g}")

    audit = dict(coverage)
    audit.update({
        "archive_records": int(len(archive)),
        "frozen_rows_matched_exactly": int(len(exact)),
        "exact_join_metal_agreement": metal_agreement,
        "exact_join_max_abs_target_delta": target_delta,
        "auxiliary_candidate_records": int(len(archive) - len(exact)),
    })
    return OverlapMap(archive=archive, frozen=frozen, exact=exact, audit=audit)


# --------------------------------------------------------------------------- #
# Per-fold exclusion
# --------------------------------------------------------------------------- #

def cohort_identity(cohort_frame: pd.DataFrame, overlap: OverlapMap) -> pd.DataFrame:
    """Cohort rows with the archive identities they inherit from their sources.

    Returns an explicit schema rather than whatever a merge's ``suffixes`` happen
    to produce.  ``extractant``, ``series_id`` and ``metal_symbol`` exist on both
    sides under the same name, and letting pandas disambiguate them silently
    renamed the very columns the leakage rules key on.  One cohort row can inherit
    several archive identities (it is a replicate average), so this frame has one
    row per (cohort row, source record) pair — which is what the exclusion rules
    need: a cohort row leaks if *any* of its sources does.
    """
    cohort = (cohort_frame[["row_id", "extractant", "tanimoto_cluster"]]
              .astype({"row_id": str, "extractant": str}))
    merged = overlap.exact.merge(cohort, on="row_id", how="inner",
                                 suffixes=("_archive", "_cohort"))
    return pd.DataFrame({
        "row_id": merged["row_id"].astype(str),
        "source_record_id": merged["source_record_id"].astype(str),
        "extractant": merged["extractant_cohort"].astype(str),
        "tanimoto_cluster": merged["tanimoto_cluster"].astype(str),
        "duplicate_group_id": merged["duplicate_group_id"],
        # the archive's series id, which is the space auxiliary rows live in
        "series_id_src": merged["series_id"],
        # the cohort's own series id, kept so a cohort-side collision is visible too
        "series_id_cohort": merged["series_id"],
        "doi_primary_corrected": merged["doi_primary_corrected"],
    })


def relationship_masks(
    aux: pd.DataFrame, held_out: pd.DataFrame, *,
    chemotype_of_smiles: Mapping[str, Sequence[str]],
    require_complete_assignment: bool = True,
) -> pd.DataFrame:
    """Boolean columns, one per relationship, marking auxiliary rows to exclude.

    ``held_out`` is the cohort-identity frame restricted to one fold's test rows.
    Every column is computed independently so the audit can report how much each
    relationship costs, rather than only the union.

    ``chemotype_of_smiles`` maps an auxiliary structure to **every** cohort
    chemotype it links to at the clustering threshold — not to one label.  The
    distinction is not pedantic: a dictionary built from the cohort's own
    ``extractant -> tanimoto_cluster`` column returns nothing for the auxiliary
    structures that are absent from the cohort, so ``SAME_CHEMOTYPE`` silently
    goes False and structures at Tanimoto 0.88 — even 1.0 — to a held-out ligand
    survive the filter.  The first gen11 leakage audit found 8,615 such rows.
    The assignment is therefore required to cover every auxiliary structure, and
    a caller that cannot supply one has to say so explicitly.
    """
    out = pd.DataFrame(index=aux.index)
    out["EXACT"] = False  # auxiliary_candidates already excludes frozen records

    bad_groups = set(held_out["duplicate_group_id"].dropna())
    out["DUPLICATE_EQUIVALENT"] = aux["duplicate_group_id"].isin(bad_groups).to_numpy()

    # Compare on the ARCHIVE series id, never on the namespaced one.  Curve safety
    # renames auxiliary series to ``aux:<id>``; comparing that against a cohort
    # ``<id>`` is guaranteed to match nothing, which is how this guard silently
    # went dead while 2,505 auxiliary rows shared a cohort series.
    series_column = "archive_series_id" if "archive_series_id" in aux.columns else "series_id"
    if series_column == "series_id" and aux["series_id"].astype(str).str.startswith("aux:").any():
        raise SystemExit(
            "auxiliary series ids are namespaced but 'archive_series_id' is absent; "
            "SAME_SERIES would compare 'aux:X' against 'X' and match nothing")
    bad_series = set(held_out["series_id_src"].dropna()) | set(held_out["series_id_cohort"].dropna())
    out["SAME_SERIES"] = aux[series_column].astype(str).isin(bad_series).to_numpy()

    bad_ligands = set(held_out["extractant"].dropna())
    out["SAME_LIGAND"] = aux["extractant_primary_smiles"].isin(bad_ligands).to_numpy()

    structures = aux["extractant_primary_smiles"].astype(str)
    if require_complete_assignment:
        uncovered = sorted(set(structures[structures.notna() & (structures != "nan")])
                           - set(chemotype_of_smiles))
        if uncovered:
            raise SystemExit(
                f"{len(uncovered)} auxiliary structures have no chemotype assignment "
                f"(e.g. {uncovered[:3]}); SAME_CHEMOTYPE would silently pass them. "
                "Build one with gen11.pools.chemotype_assignment, or pass "
                "require_complete_assignment=False and justify it.")
    bad_chemotypes = set(held_out["tanimoto_cluster"].dropna().astype(str))
    out["SAME_CHEMOTYPE"] = structures.map(
        lambda s: bool(set(chemotype_of_smiles.get(s, ())) & bad_chemotypes)).to_numpy(dtype=bool)

    bad_dois = set(held_out["doi_primary_corrected"].dropna())
    out["SAME_PUBLICATION"] = aux["doi_primary_corrected"].isin(bad_dois).to_numpy()
    return out


def apply_policy(masks: pd.DataFrame, policy: str) -> np.ndarray:
    """``True`` where the auxiliary row is *safe* to train on under ``policy``."""
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}; have {sorted(POLICIES)}")
    excluded = np.zeros(len(masks), dtype=bool)
    for relationship in POLICIES[policy]:
        excluded |= masks[relationship].to_numpy()
    return ~excluded
