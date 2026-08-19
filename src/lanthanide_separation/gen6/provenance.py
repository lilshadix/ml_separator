"""Experimental provenance: which study each measurement came from, and what that costs.

Every generation of this project has carried the same disclaimer — the bundle has
no ``publication_id``, so condition-matched rows might silently combine
measurements from unrelated studies — and none could quantify it.  This module
settles the question instead of repeating it.

**The bundle really cannot answer it.**  Every ``reference`` column in
``dataset with 3D structures/provenance/*.csv`` is 100 % empty, there is no DOI
anywhere in the repository, and the best proxy that can be built from bundle
information alone (contiguous ``safe_exp_id`` blocks cut at each change of
ligand) reaches only ARI ≈ 0.55 against the truth, splitting about half the
publications.

**The upstream tables can.**  ``safe_exp_id`` decomposes exactly into
``{source_file_stem}_SAFE:{exp_id}`` and joins one-to-one onto the raw SAFE
exports of the sibling dataset-builder project, which carry ``DOI``,
``entry_author``, ``addition_date`` and a dozen experimental columns the bundle
drops.  The join was verified on all 5,992 rows, cross-checked against ``D`` and
``metal_symbol``.  Provenance was stripped deliberately (the builder blacklists
``DOI``/``publication``/``exp_id`` from ever becoming a feature) — a sound
anti-leakage rule that also removed the ability to audit study boundaries.

So this module reconstructs provenance **when the upstream tables are reachable**
and records an honest ``unavailable`` when they are not.  It never guesses: every
identifier carries a status from :data:`PROVENANCE_STATUS`, and a surrogate is
labelled a surrogate.

Three things it exists to measure, all of which change how later results must be
read:

* *how much averaging crosses study boundaries* — the level cohort averages
  replicated (extractant, condition, metal) cells, and if a cell spans two
  publications that average is a merge of two experiments;
* *how much of the fold structure crosses study boundaries* — a "held-out
  series" that appears in two papers is not held out at all;
* *whether the so-called replicates are replicates* — they are not.  The
  upstream dedup key already includes ``D``, so anything with identical
  conditions *and* identical ``D`` was collapsed upstream.  What survives as a
  "repeated cell" differs in a variable the bundle does not encode (a free-text
  holdback agent, a solvent collapsed into ``diluent__other``, an unrecorded
  shaking time).  The gen5 "noise floor" computed from those cells is therefore
  not a reproducibility floor; it is the spread of *different experiments*.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..levels import LEVEL_TARGET_COLUMN, condition_labels, series_labels

#: Status of a reconstructed identifier.  Anything not ``reconstructed`` must be
#: read as an approximation and is labelled as such in every report.
PROVENANCE_STATUS: tuple[str, ...] = ("reconstructed", "surrogate", "ambiguous", "unavailable")

#: Where the upstream SAFE exports live when the sibling project is checked out.
#: Overridable from the CLI; absence is a normal, reportable state, not an error.
DEFAULT_UPSTREAM_DIRECTORIES: tuple[Path, ...] = (
    Path.home() / "PycharmProjects" / "lanthanide_dataset_builder" / "raw_data",
)

#: The SAFE database's own citation, attached to ~80 % of rows in addition to the
#: primary reference.  It identifies the compilation, not the study, so it is
#: removed before a publication identity is formed — otherwise four fifths of the
#: dataset would look like one publication.
SAFE_DATABASE_SELF_CITATION = "10.1021/jacs.5c19738"

#: Raw upstream columns worth carrying into the audit: they are experimental
#: variables the bundle's ``cond__*`` encoding drops or flattens, and they are the
#: hidden axis behind the "replicate" spread.
UPSTREAM_PHYSICAL_CONDITION_COLUMNS: tuple[str, ...] = (
    "Solvent_Name", "f_Solvent_Name", "Phase_Modifier_Name", "Phase_Modifier_Concentration_M",
    "Holdback_Agent_Name", "Holdback_Agent_Concentration_M", "Shaking_Time_min",
    "Acid_Concentration_Organic_M", "Radiolytic_Dosage_kGy", "Metal_Oxidation_state",
    "ini_comp",
)
#: Free-text bookkeeping, NOT an experimental variable.  ``comments_description``
#: is the paper's own figure/table caption ("Fig 2 …", "Table 4 …").  It must be
#: counted separately: a genuine replicate reported in two figures differs here
#: too, so "the caption differs" is not evidence of a different experiment.
#: Conflating the two turns 79 cells with a real hidden variable into 287.
UPSTREAM_DESCRIPTIVE_COLUMNS: tuple[str, ...] = ("comments_description", "comments_obsType")
UPSTREAM_HIDDEN_CONDITION_COLUMNS: tuple[str, ...] = (
    UPSTREAM_PHYSICAL_CONDITION_COLUMNS + UPSTREAM_DESCRIPTIVE_COLUMNS
)

#: Notation variants of the same DOI seen in the raw exports.  Missing any of
#: them splits one paper into two ``pub_`` ids, which then makes every cell
#: containing both spellings look as if it crossed a study boundary — 13 of the
#: 16 "multi-publication" level cells in the first draft were exactly that.
_DOI_PREFIXES = (
    "https://www.doi.org/", "http://www.doi.org/", "https://dx.doi.org/", "http://dx.doi.org/",
    "https://doi.org/", "http://doi.org/", "www.doi.org/", "dx.doi.org/", "doi.org/",
    "doi:", "doi ",
)
#: A DOI embedded anywhere in a token.  One reference in the bundle is a mangled
#: citation blob in which the DOI is buried among page numbers and a copyright
#: line; pulling the DOI out of it is more faithful than treating the fragments
#: as four separate publications.
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s,;]+")


# --------------------------------------------------------------------------- #
# Upstream tables
# --------------------------------------------------------------------------- #

def find_upstream_directory(candidates: Iterable[Path] = DEFAULT_UPSTREAM_DIRECTORIES) -> Path | None:
    for candidate in candidates:
        path = Path(candidate)
        if path.is_dir() and any(path.glob("*_SAFE.csv")):
            return path
    return None


def load_upstream_tables(directory: str | Path) -> pd.DataFrame:
    """Concatenate the raw ``*_SAFE.csv`` exports and rebuild ``safe_exp_id``.

    ``safe_exp_id`` in the bundle is ``{file stem}_SAFE:{exp_id}``; the stem is
    the export file, **not** the metal (66 % of rows have a stem different from
    their own ``metal_symbol``, because a Eu measurement co-reported in an Am
    study lives in ``Am_SAFE.csv``).
    """
    root = Path(directory)
    files = sorted(root.glob("*_SAFE.csv"))
    if not files:
        raise FileNotFoundError(f"no *_SAFE.csv under {root}")
    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frame["upstream_source_file"] = path.name
        frame["upstream_stem"] = path.name[: -len("_SAFE.csv")]
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    if "exp_id" not in raw.columns:
        raise KeyError(f"upstream tables under {root} have no exp_id column")
    raw["safe_exp_id"] = raw["upstream_stem"].astype(str) + "_SAFE:" + raw["exp_id"].astype(str)
    return raw


def canonical_publication(reference: object) -> tuple[str | None, tuple[str, ...]]:
    """Normalise a raw reference string into ``(publication_id, references)``.

    The raw field is a comma-separated list of URLs/DOIs that always includes the
    SAFE compilation's own citation.  Normalisation strips URL prefixes, lowercases,
    removes the self-citation, sorts what remains and hashes it, so two rows citing
    the same paper in different notations get the same id.  A row left with nothing
    but the self-citation has **no identifiable study** and returns ``None``.
    """
    if reference is None or (isinstance(reference, float) and np.isnan(reference)):
        return None, ()
    text = str(reference)
    tokens: list[str] = []
    for token in re.split(r"[,;]\s*", text):
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
    digest = hashlib.sha1("|".join(unique).encode()).hexdigest()[:10]
    return f"pub_{digest}", unique


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ProvenanceAudit:
    """Per-row provenance plus the summary the protocol requires."""

    table: pd.DataFrame
    audit: dict = field(default_factory=dict)

    @property
    def state(self) -> dict:
        """Compact state for a run manifest — what was known, and how well."""
        return {
            "publication_status": self.audit.get("publication_id_status"),
            "series_status": self.audit.get("experiment_series_id_status"),
            "replicate_status": self.audit.get("replicate_id_status"),
            "upstream_directory": self.audit.get("upstream_directory"),
            "n_publications": self.audit.get("n_publications"),
            "fraction_ambiguous_provenance": self.audit.get("fraction_ambiguous_provenance"),
            "provenance_strict_is_legacy": self.audit.get("provenance_strict_is_legacy"),
        }

    def to_json(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.audit, indent=2, default=_default) + "\n")
        return out


def _default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _describe(counts: pd.Series) -> dict:
    if counts.empty:
        return {"n": 0}
    return {
        "n": int(counts.size), "min": int(counts.min()), "q25": float(counts.quantile(0.25)),
        "median": float(counts.median()), "q75": float(counts.quantile(0.75)),
        "max": int(counts.max()), "mean": float(counts.mean()),
    }


def _group_purity(frame: pd.DataFrame, keys: Sequence[str], *, label: str) -> dict:
    """How much of a grouping crosses publication boundaries."""
    if "publication_id" not in frame.columns:
        return {"status": "unavailable"}
    known = frame[frame["publication_id"].notna()]
    if known.empty:
        return {"status": "unavailable"}
    grouped = known.groupby(list(keys), dropna=False)["publication_id"].nunique()
    sizes = known.groupby(list(keys), dropna=False).size()
    mixed = grouped[grouped > 1]
    return {
        "label": label,
        "n_groups": int(grouped.size),
        "n_groups_multi_publication": int(mixed.size),
        "n_rows_in_multi_publication_groups": int(sizes[mixed.index].sum()) if mixed.size else 0,
        "fraction_rows_in_multi_publication_groups":
            float(sizes[mixed.index].sum() / len(known)) if mixed.size else 0.0,
    }


def reconstruct_provenance(
    source: pd.DataFrame,
    *,
    upstream_directory: str | Path | None = None,
    upstream: pd.DataFrame | None = None,
    include_curator: bool = True,
) -> ProvenanceAudit:
    """Reconstruct publication / series / experiment / replicate identity.

    ``upstream`` (or ``upstream_directory``) supplies the raw SAFE exports.  With
    neither, the function still returns a complete audit: ``experiment_id`` is
    always available (``safe_exp_id`` is row-unique), a bundle-only surrogate
    series is built, and publication identity is reported as ``unavailable`` with
    the reason — never invented.
    """
    if "safe_exp_id" not in source.columns:
        raise KeyError("source frame needs safe_exp_id to reconstruct provenance")

    condition_columns = [c for c in source.columns if c.startswith("cond__")]
    frame = pd.DataFrame({
        "safe_exp_id": source["safe_exp_id"].astype(str),
        "extractant": source["canonical_smiles"].astype(str),
        "metal_symbol": source["metal_symbol"].astype(str) if "metal_symbol" in source.columns else "",
        LEVEL_TARGET_COLUMN: source[LEVEL_TARGET_COLUMN].astype(float)
        if LEVEL_TARGET_COLUMN in source.columns else np.nan,
    })
    if condition_columns:
        frame["condition_id"] = condition_labels(source, condition_columns).to_numpy()
        # The bundle-only surrogate series is *exactly* levels.series_labels — one
        # extractant under one categorical condition setting — so that a comparison
        # against the reconstructed series measures the reconstruction, not a
        # difference in how the surrogate was defined.
        surrogate_source = source.assign(extractant=frame["extractant"].to_numpy())
        surrogate_series = series_labels(surrogate_source, condition_columns).to_numpy()
    else:
        frame["condition_id"] = "unknown"
        surrogate_series = frame["extractant"].to_numpy()
    # `experiment_id` is the one identifier the bundle keeps: safe_exp_id is the
    # upstream row id and is unique on every row of this table.
    frame["experiment_id"] = frame["safe_exp_id"]
    frame["experiment_id_status"] = "reconstructed"

    audit: dict = {
        "n_rows": int(len(frame)),
        "n_extractants": int(frame["extractant"].nunique()),
        "experiment_id_status": "reconstructed",
        "experiment_id_definition": "safe_exp_id — the upstream row id, unique on every row",
        "n_experiment_ids": int(frame["experiment_id"].nunique()),
        "bundle_contains_bibliographic_data": False,
        "bundle_reference_columns_checked": [
            "provenance/geometry_specs.csv:reference", "provenance/current_qc_supplement.csv:reference",
            "provenance/current_geometry_index.csv:reference", "accepted_geometries.csv",
            "row_geometry_map.csv", "bundle_manifest.json",
        ],
    }

    # -- upstream join ------------------------------------------------------- #
    if upstream is None:
        directory = Path(upstream_directory) if upstream_directory else find_upstream_directory()
        if directory is not None:
            try:
                upstream = load_upstream_tables(directory)
                audit["upstream_directory"] = str(directory)
            except (FileNotFoundError, KeyError) as error:
                audit["upstream_directory"] = None
                audit["upstream_error"] = str(error)
        else:
            audit["upstream_directory"] = None
    else:
        audit["upstream_directory"] = str(upstream_directory) if upstream_directory else "supplied"

    if upstream is not None and "safe_exp_id" in upstream.columns:
        keep = ["safe_exp_id", "DOI", "entry_author", "addition_date", "upstream_source_file"]
        keep += [c for c in UPSTREAM_HIDDEN_CONDITION_COLUMNS if c in upstream.columns]
        keep = [c for c in dict.fromkeys(keep) if c in upstream.columns]
        joined = frame.merge(upstream[keep].drop_duplicates("safe_exp_id"),
                             on="safe_exp_id", how="left")
        matched = int(joined["DOI"].notna().sum()) if "DOI" in joined.columns else 0
        audit["upstream_rows"] = int(len(upstream))
        audit["upstream_join_matched_rows"] = matched
        audit["upstream_join_fraction"] = float(matched / len(frame)) if len(frame) else 0.0
        frame = joined
    else:
        audit["upstream_rows"] = 0
        audit["upstream_join_matched_rows"] = 0
        audit["upstream_join_fraction"] = 0.0

    # -- publication identity ------------------------------------------------ #
    if "DOI" in frame.columns and frame["DOI"].notna().any():
        parsed = [canonical_publication(v) for v in frame["DOI"]]
        frame["publication_id"] = [p for p, _ in parsed]
        frame["publication_references"] = ["|".join(r) for _, r in parsed]
        frame["publication_id_status"] = np.where(
            frame["publication_id"].notna(), "reconstructed", "ambiguous")
        ambiguous = float(frame["publication_id"].isna().mean())
        audit["publication_id_status"] = "reconstructed"
        audit["publication_id_definition"] = (
            "canonical set of DOIs from the upstream SAFE export, excluding the SAFE "
            f"database self-citation {SAFE_DATABASE_SELF_CITATION}, hashed to pub_<sha1[:10]>")
        audit["n_publications"] = int(frame["publication_id"].nunique())
        # A row may cite more than one paper, and a *set* of references is its own
        # id.  Report the distinct individual references too, so the two numbers
        # cannot be confused: they differ exactly by the co-cited combinations.
        tokens = {t for _, refs in parsed for t in refs}
        audit["n_reference_tokens"] = len(tokens)
        audit["n_publication_ids_with_multiple_references"] = int(
            frame.loc[frame["publication_id"].notna(), "publication_references"]
            .str.contains(r"\|").groupby(frame["publication_id"]).any().sum())
        audit["fraction_ambiguous_provenance"] = ambiguous
        known = frame[frame["publication_id"].notna()]
        audit["rows_per_publication"] = _describe(known.groupby("publication_id").size())
        audit["extractants_per_publication"] = _describe(
            known.groupby("publication_id")["extractant"].nunique())
        audit["metals_per_publication"] = _describe(
            known.groupby("publication_id")["metal_symbol"].nunique())
        audit["publications_per_extractant"] = _describe(
            known.groupby("extractant")["publication_id"].nunique())
        shared = known.groupby("extractant")["publication_id"].nunique()
        audit["n_extractants_in_multiple_publications"] = int((shared > 1).sum())
        audit["rows_of_extractants_in_multiple_publications"] = int(
            known[known["extractant"].isin(shared[shared > 1].index)].shape[0])
        if include_curator and "entry_author" in frame.columns:
            audit["curator_row_counts"] = frame["entry_author"].value_counts().to_dict()
        if "addition_date" in frame.columns:
            audit["n_addition_timestamps"] = int(frame["addition_date"].nunique())
    else:
        frame["publication_id"] = pd.Series([None] * len(frame), dtype=object)
        frame["publication_references"] = ""
        frame["publication_id_status"] = "unavailable"
        audit["publication_id_status"] = "unavailable"
        audit["publication_id_definition"] = None
        audit["publication_id_unavailable_reason"] = (
            "the bundle carries no bibliographic field (every provenance `reference` column is "
            "empty and no DOI exists anywhere in it); the upstream SAFE exports were not reachable, "
            "so publication identity cannot be established. It is NOT approximated: the best "
            "bundle-only proxy reaches ARI ~0.55 and splits about half of the studies.")
        audit["n_publications"] = None
        audit["fraction_ambiguous_provenance"] = 1.0
        audit["rows_per_publication"] = None
        audit["extractants_per_publication"] = None
        audit["metals_per_publication"] = None

    # -- series identity ------------------------------------------------------ #
    if "comments_description" in frame.columns and frame["comments_description"].notna().any() \
            and frame["publication_id"].notna().any():
        frame["experiment_series_id"] = _hash_key(
            frame.assign(_pub=frame["publication_id"].astype(str),
                         _desc=frame["comments_description"].astype(str)), ["_pub", "_desc"])
        frame["experiment_series_id_status"] = np.where(
            frame["publication_id"].notna() & frame["comments_description"].notna(),
            "reconstructed", "surrogate")
        fallback = frame["experiment_series_id_status"] == "surrogate"
        frame.loc[fallback, "experiment_series_id"] = pd.Series(surrogate_series, index=frame.index)[fallback]
        audit["experiment_series_id_status"] = "reconstructed"
        audit["experiment_series_id_definition"] = (
            "(publication_id, comments_description) from the upstream export. The description is "
            "free text such as 'Fig 2 …' or 'Table 4 …', so a series is only as well defined as the "
            "paper's own figure labelling; rows without a description fall back to the bundle surrogate.")
    else:
        frame["experiment_series_id"] = pd.Series(surrogate_series, index=frame.index)
        frame["experiment_series_id_status"] = "surrogate"
        audit["experiment_series_id_status"] = "surrogate"
        audit["experiment_series_id_definition"] = (
            "SURROGATE: one extractant x one categorical condition setting (the definition "
            "levels.series_labels already uses). It is not a study's measurement series.")
    audit["n_series"] = int(frame["experiment_series_id"].nunique())
    audit["conditions_per_series"] = _describe(
        frame.groupby("experiment_series_id")["condition_id"].nunique())
    audit["metals_per_series"] = _describe(
        frame.groupby("experiment_series_id")["metal_symbol"].nunique())
    audit["rows_per_series"] = _describe(frame.groupby("experiment_series_id").size())

    # -- cells and "replicates" ----------------------------------------------- #
    cell_keys = ["extractant", "condition_id", "metal_symbol"]
    frame["cell_id"] = _hash_key(frame, cell_keys)
    cell_sizes = frame.groupby("cell_id").size()
    repeated = cell_sizes[cell_sizes > 1]
    audit["n_cells"] = int(cell_sizes.size)
    audit["n_cells_repeated"] = int(repeated.size)
    audit["n_rows_in_repeated_cells"] = int(cell_sizes[repeated.index].sum()) if repeated.size else 0
    if repeated.size and LEVEL_TARGET_COLUMN in frame.columns:
        spread = (frame[frame["cell_id"].isin(repeated.index)]
                  .groupby("cell_id")[LEVEL_TARGET_COLUMN].agg(lambda s: float(s.max() - s.min())))
        audit["repeated_cell_log_d_range"] = {
            "median": float(spread.median()), "mean": float(spread.mean()),
            "p90": float(spread.quantile(0.9)), "max": float(spread.max()),
            "fraction_over_0_5": float((spread > 0.5).mean()),
            "fraction_exactly_zero": float((spread == 0).mean()),
        }
    # Are the repeated cells true replicates?  Only if nothing else distinguishes
    # them.  Any upstream column that varies inside a repeated cell is a hidden
    # experimental axis, which makes the cell a set of *different* experiments.
    hidden = [c for c in UPSTREAM_HIDDEN_CONDITION_COLUMNS if c in frame.columns]
    if repeated.size and hidden:
        block = frame[frame["cell_id"].isin(repeated.index)]
        varying = {}
        for column in hidden:
            per_cell = block.groupby("cell_id")[column].nunique(dropna=False)
            varying[column] = {"cells_varying": int((per_cell > 1).sum()),
                               "fraction_of_repeated_cells": float((per_cell > 1).mean())}
        audit["hidden_axes_inside_repeated_cells"] = varying

        def _varies(columns) -> np.ndarray:
            flags = np.zeros(len(repeated), dtype=bool)
            for name in columns:
                if name not in block.columns:
                    continue
                per_cell = block.groupby("cell_id")[name].nunique(dropna=False).reindex(repeated.index)
                flags |= (per_cell > 1).to_numpy()
            return flags

        physical = _varies(UPSTREAM_PHYSICAL_CONDITION_COLUMNS)
        descriptive = _varies(UPSTREAM_DESCRIPTIVE_COLUMNS)
        any_varying = physical | descriptive
        audit["repeated_cells_with_a_hidden_axis"] = int(any_varying.sum())
        audit["repeated_cells_with_a_PHYSICAL_hidden_axis"] = int(physical.sum())
        audit["repeated_cells_differing_only_in_free_text"] = int((descriptive & ~physical).sum())
        audit["repeated_cells_that_look_like_true_replicates"] = int((~any_varying).sum())
        audit["repeated_cells_that_look_like_true_replicates_physical_only"] = int((~physical).sum())
        # Spread PER SUBSET.  Quoting the whole-set range next to the "true repeat"
        # count invites the reader to attach one to the other; they are different
        # populations, and here they point in opposite directions.
        cell_range = (block.groupby("cell_id")[LEVEL_TARGET_COLUMN]
                      .agg(lambda s: float(s.max() - s.min())).reindex(repeated.index))
        subsets = {"physical_axis_varies": physical,
                   "free_text_only": descriptive & ~physical,
                   "no_recoverable_difference": ~any_varying}
        audit["repeated_cell_log_d_range_by_subset"] = {
            name: ({"n": int(mask.sum()),
                    "median": float(cell_range[mask].median()),
                    "max": float(cell_range[mask].max())} if mask.any() else {"n": 0})
            for name, mask in subsets.items()}
        frame["replicate_id"] = frame["cell_id"]
        frame["replicate_id_status"] = "surrogate"
        audit["replicate_id_status"] = "surrogate"
        audit["replicate_id_definition"] = (
            "SURROGATE only. Identical conditions WITH identical D were collapsed upstream (the "
            "dedup key includes D), so a cell that repeats here differs in something. Read the two "
            "counts separately: repeated_cells_with_a_PHYSICAL_hidden_axis is the defensible "
            "'different experiment' count; repeated_cells_differing_only_in_free_text differ only in "
            "the paper's own figure/table caption, which a genuine replicate reported twice also "
            "does. Neither count licenses a claim about the measurement noise floor on its own -- see "
            "repeated_cell_log_d_range_by_subset, where the cells with NO recoverable difference "
            "scatter more than the ones with one.")
    else:
        frame["replicate_id"] = frame["cell_id"]
        frame["replicate_id_status"] = "unavailable" if not hidden else "surrogate"
        audit["replicate_id_status"] = frame["replicate_id_status"].iloc[0] if len(frame) else "unavailable"
        audit["replicate_id_definition"] = (
            "Not reconstructible from the bundle; upstream columns needed to test whether repeated "
            "cells are replicates were not reachable.")

    # -- how much of the modelling structure crosses study boundaries --------- #
    # Purity is reported for the groupings the *harness* actually uses.  The
    # reconstructed series is publication-keyed by construction and would score a
    # meaningless 0 %, so the row that matters is the bundle surrogate — the
    # definition gen5's `unseen_series` regime ran on.
    frame["_surrogate_series"] = pd.Series(surrogate_series, index=frame.index)
    audit["merged_across_publication_boundaries"] = {
        "level_cell": _group_purity(frame, ["cell_id"], label="extractant x condition x metal (averaged in gen5)"),
        "surrogate_series": _group_purity(
            frame, ["_surrogate_series"],
            label="levels.series_labels — the CV group gen5's unseen_series used"),
        "extractant_condition": _group_purity(
            frame, ["extractant", "condition_id"], label="the pair-cohort key (log_SF)"),
        "extractant": _group_purity(frame, ["extractant"],
                                    label="the CV group of unseen_ligand (same ligand, several papers)"),
    }
    merged = audit["merged_across_publication_boundaries"]["level_cell"]
    audit["fraction_merged_across_publication_boundaries"] = (
        merged.get("fraction_rows_in_multi_publication_groups")
        if isinstance(merged, dict) else None)

    # -- the two dataset variants --------------------------------------------- #
    # These are not definitions on paper: `strict_cell_id` is materialised on every
    # row, so a downstream cohort builder can group by it instead of `cell_id` and
    # actually avoid averaging two studies together.  An earlier version emitted the
    # definition string and a mask that was True everywhere, which is not a variant.
    frame["legacy_compatible"] = True
    if audit["publication_id_status"] == "reconstructed":
        strict_mask = frame["publication_id"].notna()
        frame["provenance_strict"] = strict_mask
        frame["strict_cell_id"] = _hash_key(
            frame.assign(_pub=frame["publication_id"].astype(str)),
            ["extractant", "condition_id", "metal_symbol", "_pub"])
        n_legacy_cells = int(frame["cell_id"].nunique())
        n_strict_cells = int(frame["strict_cell_id"].nunique())
        audit["provenance_strict_definition"] = (
            "rows with an identifiable primary publication, with the averaging cell keyed by "
            "(extractant, condition, metal, publication_id) instead of (extractant, condition, "
            "metal). Materialised as the `strict_cell_id` column; a cohort builder that groups on "
            "it never averages two studies into one target value.")
        audit["provenance_strict_is_legacy"] = bool(strict_mask.all()) and (
            n_strict_cells == n_legacy_cells)
        audit["provenance_strict_rows"] = int(strict_mask.sum())
        audit["provenance_strict_rows_dropped"] = int((~strict_mask).sum())
        audit["provenance_strict_cells"] = n_strict_cells
        audit["legacy_cells"] = n_legacy_cells
        audit["cells_split_by_strict_keying"] = n_strict_cells - n_legacy_cells
    else:
        frame["provenance_strict"] = True
        frame["strict_cell_id"] = frame["cell_id"]
        audit["provenance_strict_cells"] = int(frame["cell_id"].nunique())
        audit["legacy_cells"] = int(frame["cell_id"].nunique())
        audit["cells_split_by_strict_keying"] = 0
        audit["provenance_strict_definition"] = (
            "IDENTICAL TO LEGACY. Nothing stricter is defensible without publication identity: "
            "every bundle-only surrogate either splits real studies or merges them, so filtering on "
            "one would trade an unknown bias for a known one.")
        audit["provenance_strict_is_legacy"] = True
        audit["provenance_strict_rows"] = int(len(frame))
        audit["provenance_strict_rows_dropped"] = 0

    return ProvenanceAudit(table=frame, audit=audit)


def _hash_key(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    view = frame[list(columns)].to_numpy(dtype=object)
    rows = ("|".join("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in row)
            for row in view)
    return pd.Series([hashlib.sha1(r.encode()).hexdigest()[:16] for r in rows], index=frame.index)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def provenance_audit_report(audit: ProvenanceAudit) -> str:
    """Human-readable provenance report — status first, numbers second."""
    a = audit.audit
    lines: list[str] = []
    add = lines.append
    add("# Provenance audit")
    add("")
    add(f"* rows: {a['n_rows']}, extractants: {a['n_extractants']}")
    add(f"* upstream tables: {a.get('upstream_directory') or 'NOT REACHABLE'}"
        + (f" ({a.get('upstream_join_matched_rows')}/{a['n_rows']} rows joined, "
           f"{100 * a.get('upstream_join_fraction', 0):.2f} %)" if a.get("upstream_rows") else ""))
    add("")
    add("| identifier | status | definition |")
    add("|---|---|---|")
    for key, label in (("experiment_id", "experiment_id"), ("publication_id", "publication_id"),
                       ("experiment_series_id", "experiment_series_id"), ("replicate_id", "replicate_id")):
        definition = a.get(f"{key}_definition") or a.get(f"{key}_unavailable_reason") or "—"
        add(f"| `{label}` | **{a.get(f'{key}_status')}** | {str(definition).replace(chr(10), ' ')} |")
    add("")
    add("## Counts")
    add(f"* publications: {a.get('n_publications')}")
    add(f"* series: {a.get('n_series')}  |  cells: {a.get('n_cells')}  |  "
        f"experiments: {a.get('n_experiment_ids')}")
    for key in ("rows_per_publication", "extractants_per_publication", "metals_per_publication",
                "rows_per_series", "conditions_per_series", "metals_per_series"):
        value = a.get(key)
        if isinstance(value, dict) and value.get("n"):
            add(f"* {key}: median {value['median']:.0f}, q25–q75 {value['q25']:.0f}–{value['q75']:.0f}, "
                f"max {value['max']} (n={value['n']})")
    if a.get("n_extractants_in_multiple_publications") is not None:
        add(f"* extractants appearing in more than one publication: "
            f"{a['n_extractants_in_multiple_publications']} "
            f"({a.get('rows_of_extractants_in_multiple_publications')} rows)")
    add("")
    add("## How much of the modelling structure crosses study boundaries")
    add("| grouping | groups | groups spanning >1 publication | rows affected |")
    add("|---|---|---|---|")
    for key, value in (a.get("merged_across_publication_boundaries") or {}).items():
        if not isinstance(value, dict) or value.get("status") == "unavailable":
            add(f"| {key} | — | unavailable | — |")
            continue
        add(f"| {value.get('label', key)} | {value['n_groups']} | "
            f"{value['n_groups_multi_publication']} | {value['n_rows_in_multi_publication_groups']} "
            f"({100 * value['fraction_rows_in_multi_publication_groups']:.2f} %) |")
    add("")
    add("## Are the repeated cells replicates?")
    add(f"* repeated cells: {a.get('n_cells_repeated')} holding {a.get('n_rows_in_repeated_cells')} rows")
    spread = a.get("repeated_cell_log_d_range")
    if spread:
        add(f"* their log D range: median {spread['median']:.3f}, mean {spread['mean']:.3f}, "
            f"p90 {spread['p90']:.3f}, max {spread['max']:.3f}; "
            f"{100 * spread['fraction_over_0_5']:.1f} % exceed 0.5 log units and only "
            f"{100 * spread['fraction_exactly_zero']:.1f} % are exactly zero")
    if a.get("repeated_cells_with_a_hidden_axis") is not None:
        add(f"* cells differing in a **physical** variable the bundle drops: "
            f"**{a.get('repeated_cells_with_a_PHYSICAL_hidden_axis')}** of {a.get('n_cells_repeated')}")
        add(f"* cells differing **only in the paper's own figure/table caption**: "
            f"{a.get('repeated_cells_differing_only_in_free_text')} — a genuine replicate reported in "
            f"two figures looks exactly like this, so these are NOT evidence of a different experiment")
        add(f"* cells with no recoverable difference at all: "
            f"{a.get('repeated_cells_that_look_like_true_replicates')} "
            f"(counting only physical axes, {a.get('repeated_cells_that_look_like_true_replicates_physical_only')} "
            f"would qualify)")
        add("* per-axis breakdown (cells in which the upstream column varies):")
        for column, stats in (a.get("hidden_axes_inside_repeated_cells") or {}).items():
            if stats["cells_varying"]:
                add(f"    * `{column}`: {stats['cells_varying']} cells "
                    f"({100 * stats['fraction_of_repeated_cells']:.1f} %)")
        by_subset = a.get("repeated_cell_log_d_range_by_subset") or {}
        if by_subset:
            add("")
            add("| subset | cells | median log D range | max |")
            add("|---|---|---|---|")
            for name, stats in by_subset.items():
                if not stats.get("n"):
                    continue
                add(f"| {name.replace('_', ' ')} | {stats['n']} | {stats['median']:.3f} | "
                    f"{stats['max']:.3f} |")
        add("")
        add("  **How far this goes, and no further.** A quarter of the repeated cells differ in a "
            "recoverable *physical* variable, so for those the published noise floor is partly "
            "missing-feature error rather than measurement error. It does not follow that the floor "
            "is wrong overall: most repeated cells differ only in a caption, and the cells with no "
            "recoverable difference at all scatter *more* than the ones with a hidden axis. The "
            "defensible statement is that the floor is contaminated and its true value is unknown, "
            "not that it is an artefact.")
    add("")
    add("## Dataset variants")
    add(f"* `legacy_compatible`: {a['n_rows']} rows (everything).")
    add(f"* `provenance_strict`: {a.get('provenance_strict_rows')} rows "
        f"({a.get('provenance_strict_rows_dropped')} dropped). {a.get('provenance_strict_definition')}")
    add(f"* averaging cells: legacy {a.get('legacy_cells')} -> strict "
        f"{a.get('provenance_strict_cells')} (**{a.get('cells_split_by_strict_keying')}** cells split "
        f"apart because they mixed two studies)")
    add(f"* strict == legacy: **{a.get('provenance_strict_is_legacy')}**")
    return "\n".join(lines) + "\n"
