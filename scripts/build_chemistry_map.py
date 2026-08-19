"""Freeze one description of chemical space for the whole gen6 generation.

Every study this repo has run carved a cohort out of the same 190 extractants
with an eligibility rule (``min_rows_per_extractant``) and then re-derived its
ECFP and Tanimoto cluster labels *inside* that cohort.  Two cohorts labelled that
way cannot be compared: "super-cluster 7" is a different object in each, and the
gen6 experiments are built entirely on comparing a 91-extractant training set
against a 152-extractant one on *identical* test rows.  So gen6 computes
chemistry once, over all 190 extractants, target-independently, and freezes it
into an artifact that every later run hashes into its manifest.

This script is the freezer.  What it must convince a sceptical reader of:

1. **The freeze changes no historical fold boundary.**  If the all-190 partition,
   restricted to a cohort, disagreed with that cohort's own partition, then
   adopting the frozen labels would silently move ligands across folds and every
   gen6-vs-gen5 comparison would be contaminated.  The report therefore prints
   the *full* :func:`~lanthanide_separation.gen6.chemistry.partition_stability`
   check against both cohorts (``min_rows`` 10 and 3) — the crosstab-derived
   split/merge counts, not a boolean.  A reader who only sees "stable: True" has
   been given a claim, not evidence.

2. **What the ``min_rows >= 10`` rule actually discarded.**  Phase 0 of the
   protocol asks explicitly which extractants enter at ``min_rows = 3`` but not
   at 10, how far they sit from the dense cohort in Tanimoto space, and how many
   ECFP clusters and chemotypes exist *only* because of them.  That table is the
   quantitative statement of the generation's primary hypothesis (coverage, not
   capacity, is the binding constraint), so it is computed here rather than
   asserted later.

Traps this script is written around, all of them things that have bitten this
project before:

* **Cluster-label strings are not stable; partitions are.**  ``fcluster``
  numbers clusters in input order, so the frozen map renumbers canonically
  (``chem__`` prefix) and never overwrites a cohort-local label.  Reproduction
  runs must keep the cohort-local strings; this artifact is for *comparison*
  across cohorts.
* **Two different "n_cells".**  The chemistry map counts cells over the raw
  source table.  ``build_level_dataset`` first drops rows at or below the
  ``log_D`` detection floor and only then applies the eligibility rule, so its
  cell counts are lower.  The eligibility rule is defined on the *cohort's*
  counts, so the entrant table reports the cohort number as definitional and the
  map's number beside it.  Reporting only one of them would misstate which
  ligands the rule discards.
* **A nearest-neighbour similarity of "none" is 0.0, not NaN.**  An extractant
  with no neighbour in the reference cohort is the most important case in the
  study; a NaN there would quietly drop it out of every average.
* **``log_D`` is never read by the map.**  The cohorts are built with the target
  present because ``build_level_dataset`` needs it, but the map itself is
  target-independent and ``tests/test_gen6_chemistry.py`` proves it.

Nothing is fitted here.  The run is cheap (~1 minute, dominated by hashing the
2,261-column source frame) and writes the standard gen6 provenance set so that
``scripts/aggregate_gen6.py`` can check any later run against it.

Example::

    .venv/bin/python scripts/build_chemistry_map.py
    .venv/bin/python scripts/build_chemistry_map.py --output-dir runs/gen6_chemistry_test \\
        --cohort-min-cells 10 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen6.chemistry import (  # noqa: E402
    DEFAULT_SUPERCLUSTER_THRESHOLD, ChemistryMap, build_chemistry_map, cluster_manifest,
    partition_stability,
)
from lanthanide_separation.gen6.cohorts import (  # noqa: E402
    BASE_MIN_CELLS, EXPANDED_MIN_CELLS, cells_per_extractant, cohort_comparison,
)
from lanthanide_separation.gen6.manifest import (  # noqa: E402
    RunManifest, sha256_file, sha256_frame, validate_run, write_success,
)
from lanthanide_separation.levels import build_level_dataset  # noqa: E402

DATASET_PATH = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
DESCRIPTOR_PATH = REPO_ROOT / "dataset with 3D structures" / "ligand_2d_descriptors.parquet"

#: Tanimoto cut points the protocol reports novelty at.  0.4 and 0.6 are the
#: pre-registered "hard chemistry" endpoints; 0.5 and 0.7 are printed because a
#: distribution summarised at only its two decision points is not a distribution.
NOVELTY_THRESHOLDS: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7)
#: Quantiles printed for every similarity distribution in the report.
REPORT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

#: Artifacts this run must have written before it may claim success.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "chemistry_map.parquet",
    "chemistry_map.parquet.similarity.npz",
    "chemistry_cluster_manifest.json",
    "nearest_neighbor_matrix.npz",
    "chemistry_report.md",
    "cohort_entrants.csv",
    "summary.json",
    "manifest.json",
)

#: Source files whose content defines this run's behaviour.
CODE_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve(),
    REPO_ROOT / "src" / "lanthanide_separation" / "gen6",
    REPO_ROOT / "src" / "lanthanide_separation" / "levels.py",
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", type=Path, default=DATASET_PATH,
                   help="source parquet; the map covers EVERY extractant in it, eligible or not")
    # Deliberately a string, not a Path: ``Path("")`` silently becomes ``Path(".")``,
    # which exists, so a ``--descriptors ''`` meant as "none" would be handed to
    # ``read_parquet`` as the working directory.  Parse the emptiness first.
    p.add_argument("--descriptors", type=str, default=str(DESCRIPTOR_PATH),
                   help="gen4 ligand_2d_descriptors parquet; supplies the rdkit-derived motif "
                        "counts the family assignment prefers. Pass '' to force the SMILES fallback.")
    p.add_argument("--threshold", type=float, default=DEFAULT_SUPERCLUSTER_THRESHOLD,
                   help="single-linkage Tanimoto threshold defining a super-cluster (chemotype)")
    p.add_argument("--cohort-min-cells", nargs="*", type=int,
                   default=[BASE_MIN_CELLS, EXPANDED_MIN_CELLS],
                   help="eligibility rules to run the partition-stability check against; "
                        "pass no values to skip cohort building entirely")
    p.add_argument("--replicate-policy", default="mean", choices=["mean", "unique", "all"],
                   help="passed to build_level_dataset; must match the runs this map will serve")
    p.add_argument("--log-d-floor", type=float, default=-6.0,
                   help="passed to build_level_dataset; affects the cohorts, never the map")
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Small report helpers
# --------------------------------------------------------------------------- #

def markdown_table(frame: pd.DataFrame, *, floatfmt: str = "{:.4g}") -> str:
    """Render a frame as a GitHub pipe table.

    ``DataFrame.to_markdown`` needs ``tabulate``, which is not a dependency of
    this repo, so the table is built by hand.  Kept deliberately dumb: no
    alignment tricks, no truncation — a report that hides a row is worse than an
    ugly one.
    """
    if frame.empty:
        return "_(no rows)_"

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—" if value is None or np.isnan(value) else str(value)
        if isinstance(value, (float, np.floating)):
            return floatfmt.format(float(value))
        if isinstance(value, (bool, np.bool_)):
            return "yes" if bool(value) else "**no**"
        return str(value)

    header = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def distribution_summary(values: Sequence[float], *,
                         thresholds: Sequence[float] = NOVELTY_THRESHOLDS) -> dict[str, Any]:
    """Quantiles plus counts below each novelty threshold, with the n behind them."""
    array = np.asarray([float(v) for v in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "quantiles": {}, "counts_below": {}, "fraction_below": {},
                "min": None, "max": None, "mean": None}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "quantiles": {f"q{int(q * 100):02d}": float(np.quantile(array, q)) for q in REPORT_QUANTILES},
        "counts_below": {f"<{t:g}": int((array < t).sum()) for t in thresholds},
        "fraction_below": {f"<{t:g}": float((array < t).mean()) for t in thresholds},
    }


def distribution_frame(summary: dict[str, Any], label: str) -> pd.DataFrame:
    """One-row rendering of :func:`distribution_summary` for the report."""
    row: dict[str, Any] = {"distribution": label, "n": summary.get("n", 0)}
    row.update(summary.get("quantiles", {}))
    row["min"] = summary.get("min")
    row["max"] = summary.get("max")
    for key, value in summary.get("counts_below", {}).items():
        row[f"n {key}"] = value
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- #
# Cohorts
# --------------------------------------------------------------------------- #

@dataclass
class CohortView:
    """A cohort built by the gen5 harness, or an honest record of why it was not.

    A cohort that fails to build must not take the whole run down: the frozen map
    is the deliverable, and the stability check is a claim *about* the map that
    can be reported as unavailable.  ``status`` is what the report prints.
    """

    min_cells: int
    status: str
    frame: pd.DataFrame | None = None
    audit: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "BUILT"

    @property
    def extractants(self) -> tuple[str, ...]:
        if self.frame is None:
            return ()
        return tuple(sorted(self.frame["extractant"].astype(str).unique()))


def build_cohort(source: pd.DataFrame, *, min_cells: int, descriptors: pd.DataFrame | None,
                 replicate_policy: str, log_d_floor: float | None, log: Callable[[str], None]) -> CohortView:
    """``build_level_dataset`` at one eligibility rule, reusing the gen5 harness verbatim."""
    try:
        data = build_level_dataset(
            source, min_rows_per_extractant=min_cells, replicate_policy=replicate_policy,
            drop_below_log_d=log_d_floor, ligand_descriptors=descriptors)
    except Exception as error:  # noqa: BLE001 - the report must survive a bad cohort
        log(f"  cohort min_cells={min_cells}: FAILED ({type(error).__name__}: {error})")
        return CohortView(min_cells=min_cells, status="FAILED", error=f"{type(error).__name__}: {error}")
    log(f"  cohort min_cells={min_cells}: {data.audit['rows']} rows, "
        f"{data.audit['extractants']} extractants, {data.audit['ecfp_clusters']} ECFP clusters, "
        f"{data.audit['tanimoto_clusters']} super-clusters")
    return CohortView(min_cells=min_cells, status="BUILT", frame=data.frame, audit=dict(data.audit))


# --------------------------------------------------------------------------- #
# Phase 0: who enters at min_cells = 3 but not at 10
# --------------------------------------------------------------------------- #

def entrant_table(
    chemistry: ChemistryMap, expanded: CohortView, base: CohortView,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The extractants the ``>= 10 cells`` rule discards, and how novel they are.

    Returns ``(table, audit)``.  ``table`` has one row per entrant with its name,
    the cohort cell count that the eligibility rule actually tested, the map's raw
    source cell count, the frozen cluster labels, and the maximum Tanimoto to the
    *dense* cohort — the number that decides whether the entrant is new chemistry
    or a near-duplicate of something already in training.

    The novelty is measured against the BASE (>= 10 cells) extractant set, fixed
    once, exactly as Experiment A's hard-chemistry bins will be, so the two
    numbers are the same number.
    """
    audit: dict[str, Any] = {"status": "OK"}
    if not (expanded.ok and base.ok):
        audit["status"] = "SKIPPED"
        audit["reason"] = (f"expanded cohort status {expanded.status}, base cohort status {base.status}; "
                           "the entrant comparison needs both")
        return pd.DataFrame(), audit

    expanded_set = set(expanded.extractants)
    base_set = set(base.extractants)
    audit["base_is_subset_of_expanded"] = bool(base_set <= expanded_set)
    # If this ever failed, the "relax the rule" framing would be wrong: the two
    # cohorts would not nest and the comparison would not be about added chemistry.
    audit["extractants_in_base_only"] = sorted(base_set - expanded_set)
    entrants = sorted(expanded_set - base_set)
    audit["n_entrants"] = len(entrants)
    audit["n_base_extractants"] = len(base_set)
    audit["n_expanded_extractants"] = len(expanded_set)
    if not entrants:
        audit["status"] = "EMPTY"
        return pd.DataFrame(), audit

    neighbours = chemistry.nearest_neighbour(entrants, sorted(base_set))
    cohort_cells = cells_per_extractant(expanded.frame)
    map_table = chemistry.table.set_index("extractant")

    rows: list[dict[str, Any]] = []
    for name in entrants:
        neighbour = neighbours.loc[neighbours["extractant"] == name].iloc[0]
        mapped = map_table.loc[name] if name in map_table.index else None
        rows.append({
            "extractant_name": ("" if mapped is None else str(mapped.get("extractant_name", "") or "")),
            "cohort_n_cells": int(cohort_cells.get(name, 0)),
            "source_n_cells": (None if mapped is None else int(mapped["n_cells"])),
            "family": (None if mapped is None else str(mapped["chem__family"])),
            "chem__ecfp_cluster": (None if mapped is None else str(mapped["chem__ecfp_cluster"])),
            "chem__supercluster": (None if mapped is None else str(mapped["chem__supercluster"])),
            "max_tanimoto_to_base": float(neighbour["nn_tanimoto"]),
            "nearest_base_partner": neighbour["nn_partner"],
            "n_base_above_0_7": int(neighbour.get("n_above_0_7", 0)),
            "extractant": name,
        })
    table = pd.DataFrame(rows).sort_values("max_tanimoto_to_base", ignore_index=True)

    # Clusters that exist ONLY because of the entrants, judged on the frozen
    # labels restricted to the shared (min_cells = 3) cohort.  Using the frozen
    # labels rather than either cohort's local ones is the whole point of the map:
    # "new super-cluster" has to mean the same thing on both sides.
    frozen = chemistry.table.set_index("extractant")
    base_ecfp = {str(frozen.loc[e, "chem__ecfp_cluster"]) for e in base_set if e in frozen.index}
    base_super = {str(frozen.loc[e, "chem__supercluster"]) for e in base_set if e in frozen.index}
    entrant_ecfp = {str(frozen.loc[e, "chem__ecfp_cluster"]) for e in entrants if e in frozen.index}
    entrant_super = {str(frozen.loc[e, "chem__supercluster"]) for e in entrants if e in frozen.index}
    new_ecfp = sorted(entrant_ecfp - base_ecfp)
    new_super = sorted(entrant_super - base_super)

    similarity = distribution_summary(table["max_tanimoto_to_base"])
    frozen_entrant_ecfp = table["chem__ecfp_cluster"].astype(str)
    audit.update({
        # 61 entrants do not mean 61 new chemistry units: some are bit-identical to
        # a ligand already in BASE (Tanimoto 1.0, different SMILES, same fingerprint)
        # and add depth rather than coverage.  Counting them separately is what
        # keeps the "the expansion buys chemistry" claim honest.
        "n_entrants_tanimoto_1_to_base": int((table["max_tanimoto_to_base"] >= 1.0).sum()),
        "n_entrants_above_0_8_to_base": int((table["max_tanimoto_to_base"] >= 0.8).sum()),
        "n_entrants_in_a_base_ecfp_cluster": int(frozen_entrant_ecfp.isin(base_ecfp).sum()),
        "n_entrants_with_name": int((table["extractant_name"].astype(str).str.len() > 0).sum()),
        "cohort_cells_min": int(table["cohort_n_cells"].min()),
        "cohort_cells_max": int(table["cohort_n_cells"].max()),
        "cohort_cells_total": int(table["cohort_n_cells"].sum()),
        "max_tanimoto_to_base": similarity,
        "n_ecfp_clusters_only_from_entrants": len(new_ecfp),
        "n_superclusters_only_from_entrants": len(new_super),
        "ecfp_clusters_only_from_entrants": new_ecfp,
        "superclusters_only_from_entrants": new_super,
        "n_base_ecfp_clusters": len(base_ecfp),
        "n_base_superclusters": len(base_super),
        "entrant_family_counts": table["family"].value_counts().to_dict(),
        "n_entrants_sharing_a_base_supercluster": int(len(entrant_super & base_super)),
    })
    return table, audit


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def family_table(chemistry: ChemistryMap) -> pd.DataFrame:
    """Scaffold families with the counts that make them interpretable."""
    table = chemistry.table
    rows: list[dict[str, Any]] = []
    for family, block in table.groupby("chem__family", sort=True):
        rows.append({
            "family": str(family),
            "n_extractants": int(len(block)),
            "n_ecfp_clusters": int(block["chem__ecfp_cluster"].nunique()),
            "n_superclusters": int(block["chem__supercluster"].nunique()),
            "median_n_cells": float(block["n_cells"].median()),
            "n_cells_total": int(block["n_cells"].sum()),
            "median_nn_within_all": float(block["chem__nn_within_all_tanimoto"].median()),
            "source": "/".join(sorted(block["chem__family_source"].unique())),
        })
    return pd.DataFrame(rows).sort_values("n_extractants", ascending=False, ignore_index=True)


def stability_frame(check: dict[str, Any], min_cells: int) -> pd.DataFrame:
    """Flatten one :func:`partition_stability` result into printable rows."""
    rows: list[dict[str, Any]] = []
    for level in ("ecfp", "supercluster"):
        block = check.get(level, {})
        if "status" in block:
            rows.append({"cohort": f"min_cells={min_cells}", "level": level,
                         "n_local": None, "n_frozen_restricted": None,
                         "local_spanning_multiple_frozen (splits)": None,
                         "frozen_spanning_multiple_local (merges)": None,
                         "identical_partition": block["status"]})
            continue
        rows.append({
            "cohort": f"min_cells={min_cells}",
            "level": level,
            "n_local": block.get("n_local"),
            "n_frozen_restricted": block.get("n_frozen_restricted"),
            "local_spanning_multiple_frozen (splits)": block.get("local_groups_spanning_multiple_frozen"),
            "frozen_spanning_multiple_local (merges)": block.get("frozen_groups_spanning_multiple_local"),
            "identical_partition": block.get("identical_partition"),
        })
    return pd.DataFrame(rows)


def render_report(
    *,
    stamp: str,
    args: argparse.Namespace,
    chemistry: ChemistryMap,
    cohorts: dict[int, CohortView],
    stability: dict[int, dict[str, Any]],
    comparison: dict[str, Any] | None,
    entrants: pd.DataFrame,
    entrant_audit: dict[str, Any],
    source_shape: tuple[int, int],
    dataset_sha: str,
    descriptor_path: Path | None,
    elapsed: float,
) -> str:
    """The human-readable ``chemistry_report.md``, written for a sceptical chemist."""
    audit = chemistry.audit
    out: list[str] = []
    A = out.append

    A(f"# gen6 frozen chemistry map — {stamp}")
    A("")
    A("One target-independent description of chemical space, computed over **every** extractant in "
      "the source table (including the ones no cohort is eligible for), and frozen so that every "
      "later gen6 run can hash it into its manifest and mean the same thing by "
      "\"super-cluster\".")
    A("")
    A(f"* source table: `{args.dataset}` — {source_shape[0]} rows x {source_shape[1]} columns, "
      f"sha256 `{dataset_sha[:16]}…`")
    A(f"* descriptor table: `{descriptor_path if descriptor_path else '(none requested)'}` — "
      f"{'supplied' if audit.get('descriptor_table_supplied') else 'NOT supplied'}")
    A(f"* rdkit importable: {audit.get('rdkit_available')} (version {audit.get('rdkit_version')}). "
      "Family assignment prefers the frozen rdkit-derived motif counts in the descriptor parquet; "
      "the SMILES-substring fallback is coarse and is labelled per row in `chem__family_source`.")
    A(f"* wall clock: {elapsed:.1f} s. Nothing is fitted here.")
    A("")

    A("## 1. Scope of the map")
    A("")
    scope = pd.DataFrame([{
        "n_extractants": audit["n_extractants"],
        "n_ecfp_clusters": audit["n_ecfp_clusters"],
        "n_superclusters": audit["n_superclusters"],
        "supercluster_threshold": audit["supercluster_threshold"],
        "n_fingerprint_bits": audit["n_fingerprint_columns"],
        "largest_supercluster (extractants)": audit["largest_supercluster_extractants"],
    }])
    A(markdown_table(scope))
    A("")
    A("Definitions (frozen for the generation): an **extractant** is a `canonical_smiles`; an "
      "**ECFP cluster** is a group of bit-identical fingerprints; a **super-cluster** (chemotype) is a "
      f"single-linkage group at Tanimoto >= {audit['supercluster_threshold']:g}. Labels carry a "
      "`chem__` prefix so they can never be confused with a cohort-local label.")
    A("")

    A("## 2. Scaffold families")
    A("")
    A(markdown_table(family_table(chemistry)))
    A("")
    A(f"Family sources: {json.dumps(audit.get('family_source_counts', {}))}. A family label is a "
      "convenience for reading the tables; no metric in this generation is defined on it.")
    A("")

    A("## 3. Nearest-neighbour similarity")
    A("")
    A("`nn_within_all` is, for each extractant, the maximum Tanimoto to any *other* extractant in the "
      "full 190. It is the honest measure of how isolated a ligand is before any cohort filter is "
      "applied. The pairwise row is the full upper triangle of the similarity matrix and is dominated "
      "by unrelated pairs, so read it as background, not as a novelty statistic.")
    A("")
    nn_summary = distribution_summary(chemistry.table["chem__nn_within_all_tanimoto"])
    upper = chemistry.similarity[np.triu_indices(len(chemistry), k=1)]
    pair_summary = distribution_summary(upper)
    A(markdown_table(pd.concat([
        distribution_frame(nn_summary, "nn_within_all (per extractant)"),
        distribution_frame(pair_summary, "all pairwise Tanimoto"),
    ], ignore_index=True)))
    A("")
    sizes = chemistry.table["chem__supercluster"].value_counts()
    A(f"Singleton chemotypes: {int((sizes == 1).sum())} of {int(sizes.size)} super-clusters hold exactly "
      f"one extractant; the largest holds {int(sizes.max())}. "
      f"{int((chemistry.table['chem__nn_within_all_tanimoto'] < 0.4).sum())} of "
      f"{len(chemistry)} extractants have **no** neighbour above Tanimoto 0.4 anywhere in the 190.")
    A("")

    A("## 4. Partition stability — is the freeze safe?")
    A("")
    A("The claim being tested: *the all-190 partition, restricted to a cohort, equals that cohort's own "
      "partition.* If it did not, adopting the frozen labels would move ligands across historical fold "
      "boundaries and every gen6-vs-gen5 comparison would be contaminated. Two partitions of the same "
      "set agree exactly when no local group spans two frozen groups (**splits** = 0) and no frozen "
      "group spans two local ones (**merges** = 0). Label strings are ignored.")
    A("")
    stability_rows = [stability_frame(check, min_cells) for min_cells, check in sorted(stability.items())]
    if stability_rows:
        A(markdown_table(pd.concat(stability_rows, ignore_index=True)))
    else:
        A("_No cohort was built, so the stability check did not run._")
    A("")
    for min_cells, check in sorted(stability.items()):
        A(f"* `min_cells = {min_cells}`: {check.get('n_extractants')} extractants matched into the frozen "
          f"map, {check.get('unmapped_extractants')} unmapped, overall ok = **{check.get('ok')}**.")
    for min_cells, cohort in sorted(cohorts.items()):
        if not cohort.ok:
            A(f"* `min_cells = {min_cells}`: cohort status **{cohort.status}** — {cohort.error}. "
              "The stability claim is UNVERIFIED for this cohort.")
    A("")
    A("**What would falsify this:** a single non-zero split or merge count. Any non-zero entry means the "
      "frozen labels must not be substituted for cohort-local ones, and every downstream run must be "
      "re-audited before it may quote a super-cluster.")
    A("")

    A("## 5. What the eligibility rule costs — BASE (>= 10 cells) vs EXPANDED (>= 3 cells)")
    A("")
    if comparison is None:
        A("_The shared `min_cells = 3` cohort was not built, so the comparison is unavailable._")
    else:
        rows = []
        for label in ("base", "added_by_expansion", "expanded"):
            block = comparison[label]
            rows.append({"side": label, **{k: block[k] for k in
                                           ("n_rows", "n_extractants", "n_ecfp_clusters", "n_superclusters")}})
        A(markdown_table(pd.DataFrame(rows)))
        A("")
        A(f"* ECFP clusters that exist only in the added rows: **{comparison['new_ecfp_clusters']}**")
        A(f"* super-clusters that exist only in the added rows: **{comparison['new_superclusters']}**")
        A(f"* rows sitting in those new super-clusters: {comparison['rows_in_new_superclusters']}")
        A(f"* the added rows are {100 * comparison['row_cost_fraction']:.1f} % of the shared cohort — "
          "the expansion buys chemistry, not volume.")
        A("")
        A("These counts use the *cohort-local* labels of the shared `min_cells = 3` frame, which is the "
          "frame every Experiment A arm will be masked out of, so both sides are labelled identically "
          "by construction.")
    A("")

    A("## 6. Phase 0 question — exactly who enters at min_cells = 3 but not at 10")
    A("")
    if entrant_audit.get("status") != "OK":
        A(f"_Unavailable: {entrant_audit.get('reason', entrant_audit.get('status'))}._")
    else:
        A(f"**{entrant_audit['n_entrants']} extractants** enter the shared cohort at `min_cells = 3` "
          f"and are excluded at `min_cells = 10` "
          f"({entrant_audit['n_base_extractants']} -> {entrant_audit['n_expanded_extractants']} "
          f"extractants). BASE is a subset of EXPANDED: **{entrant_audit['base_is_subset_of_expanded']}**.")
        A("")
        A(f"They contribute {entrant_audit['cohort_cells_total']} cohort cells in total "
          f"({entrant_audit['cohort_cells_min']}–{entrant_audit['cohort_cells_max']} each), and "
          f"{entrant_audit['n_entrants_with_name']} of {entrant_audit['n_entrants']} carry a "
          "human-readable name in the source table.")
        A("")
        A("### 6a. How novel are they? (max Tanimoto to the >= 10-cell cohort)")
        A("")
        A(markdown_table(distribution_frame(
            entrant_audit["max_tanimoto_to_base"], "entrant max Tanimoto to BASE")))
        A("")
        counts = entrant_audit["max_tanimoto_to_base"]["counts_below"]
        fractions = entrant_audit["max_tanimoto_to_base"]["fraction_below"]
        A(markdown_table(pd.DataFrame([
            {"threshold": key, "n_entrants": counts[key],
             "fraction_of_entrants": fractions[key]} for key in counts])))
        A("")
        A("A value below 0.4 is the protocol's **hard chemistry** endpoint: the dense cohort contains "
          "nothing chemically close, so a model trained on BASE alone is extrapolating.")
        A("")
        A("### 6b. New chemistry units they bring (frozen labels)")
        A("")
        A(markdown_table(pd.DataFrame([{
            "ECFP clusters present only via entrants": entrant_audit["n_ecfp_clusters_only_from_entrants"],
            "super-clusters present only via entrants": entrant_audit["n_superclusters_only_from_entrants"],
            "ECFP clusters in BASE": entrant_audit["n_base_ecfp_clusters"],
            "super-clusters in BASE": entrant_audit["n_base_superclusters"],
            "entrant chemotypes shared with BASE": entrant_audit["n_entrants_sharing_a_base_supercluster"],
        }])))
        A("")
        A(f"Not every entrant is new chemistry: {entrant_audit['n_entrants_in_a_base_ecfp_cluster']} of "
          f"{entrant_audit['n_entrants']} already sit in an ECFP cluster that BASE contains "
          f"({entrant_audit['n_entrants_tanimoto_1_to_base']} are bit-identical to a BASE ligand at "
          f"Tanimoto 1.0, i.e. a different SMILES with the same fingerprint), and "
          f"{entrant_audit['n_entrants_above_0_8_to_base']} sit at Tanimoto >= 0.8 of something BASE "
          "already has. Those add depth, not coverage, and are the reason the expansion buys "
          f"{entrant_audit['n_superclusters_only_from_entrants']} new chemotypes rather than "
          f"{entrant_audit['n_entrants']}.")
        A("")
        A(f"Entrant families: {json.dumps(entrant_audit['entrant_family_counts'])}")
        A("")
        A("### 6c. The entrants themselves")
        A("")
        A("`cohort_n_cells` is the count the eligibility rule actually tested (after the `log_D` floor); "
          "`source_n_cells` is the raw count in the source table. They differ where a ligand has rows at "
          "or below the detection floor, and the *cohort* number is the definitional one.")
        A("")
        display = entrants[["extractant_name", "family", "cohort_n_cells", "source_n_cells",
                            "max_tanimoto_to_base", "nearest_base_partner", "chem__supercluster"]].copy()
        display["nearest_base_partner"] = display["nearest_base_partner"].map(
            lambda s: (str(s)[:32] + "…") if s is not None and len(str(s)) > 32 else s)
        display.insert(0, "#", np.arange(1, len(display) + 1))
        A(markdown_table(display))
        A("")
        A("The full table, including the canonical SMILES of every entrant and its nearest BASE partner, "
          "is `cohort_entrants.csv`.")
    A("")

    A("## 7. Limits of this artifact")
    A("")
    A("* The map describes chemistry, not measurability. `n_cells` is a property of the experiment "
      "record; nothing here reads `log_D`.")
    A("* ECFP clusters are bit-identical groups of a **precomputed** 2,048-bit fingerprint taken from the "
      "source table. Two different SMILES can share one, and the map does not re-derive fingerprints.")
    A("* Single-linkage chaining means a super-cluster can hold two members below the threshold from "
      "each other. That is deliberate — it makes the hold-out strictly harder — but a super-cluster is "
      "not a ball of radius 0.3.")
    A("* The SMILES-substring family fallback is a reading aid only; if `chem__family_source` says "
      "`smiles_heuristic`, do not quote that family as a curated assignment.")
    A("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "runs" / f"gen6_chemistry_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.txt"

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    log(f"gen6 chemistry map -> {output_dir}")
    source = pd.read_parquet(args.dataset)
    log(f"source {args.dataset}: {len(source)} rows, "
        f"{source['canonical_smiles'].nunique()} extractants, {source.shape[1]} columns")

    descriptors = None
    descriptor_path = Path(args.descriptors) if str(args.descriptors).strip() else None
    if descriptor_path is not None and descriptor_path.is_file():
        descriptors = pd.read_parquet(descriptor_path)
        log(f"descriptors {descriptor_path}: {len(descriptors)} rows")
    elif descriptor_path is not None:
        log(f"WARNING: descriptor parquet not found at {descriptor_path}; families fall back to the "
            "SMILES heuristic and are labelled as such")

    chemistry = build_chemistry_map(source, threshold=args.threshold, ligand_descriptors=descriptors)
    log(f"map: {chemistry.audit['n_extractants']} extractants, "
        f"{chemistry.audit['n_ecfp_clusters']} ECFP clusters, "
        f"{chemistry.audit['n_superclusters']} super-clusters at Tanimoto >= {args.threshold}")

    # -- cohorts and the stability check ----------------------------------- #
    cohorts: dict[int, CohortView] = {}
    stability: dict[int, dict[str, Any]] = {}
    for min_cells in sorted({int(v) for v in (args.cohort_min_cells or [])}, reverse=True):
        cohort = build_cohort(source, min_cells=min_cells, descriptors=descriptors,
                              replicate_policy=args.replicate_policy, log_d_floor=args.log_d_floor,
                              log=log)
        cohorts[min_cells] = cohort
        if cohort.ok:
            stability[min_cells] = partition_stability(chemistry, cohort.frame)
            log(f"  partition stability min_cells={min_cells}: ok={stability[min_cells]['ok']}")

    expanded = cohorts.get(EXPANDED_MIN_CELLS, CohortView(min_cells=EXPANDED_MIN_CELLS, status="NOT_REQUESTED"))
    base = cohorts.get(BASE_MIN_CELLS, CohortView(min_cells=BASE_MIN_CELLS, status="NOT_REQUESTED"))

    comparison: dict[str, Any] | None = None
    if expanded.ok:
        comparison = cohort_comparison(expanded.frame, base_min_cells=BASE_MIN_CELLS)
        log(f"  cohort comparison: base {comparison['base']['n_extractants']} ext / "
            f"{comparison['base']['n_superclusters']} super, expansion adds "
            f"{comparison['added_by_expansion']['n_extractants']} ext, "
            f"{comparison['new_superclusters']} new super-clusters")

    entrants, entrant_audit = entrant_table(chemistry, expanded, base)
    log(f"  entrants (min_cells 3 but not 10): {entrant_audit.get('n_entrants', 'unavailable')}")

    # -- artifacts ---------------------------------------------------------- #
    map_path, sidecar_path = chemistry.to_parquet(output_dir / "chemistry_map.parquet")
    manifest_json = cluster_manifest(chemistry)
    (output_dir / "chemistry_cluster_manifest.json").write_text(
        json.dumps(manifest_json, indent=2, default=str) + "\n")
    npz_path = chemistry.save_similarity_npz(output_dir / "nearest_neighbor_matrix.npz")
    # Always write the entrant CSV, even when empty: a downstream reader must be
    # able to tell "no entrants" from "the step did not run", and the presence of
    # an empty file with a header says the first.
    entrant_columns = ["extractant", "extractant_name", "family", "cohort_n_cells", "source_n_cells",
                       "chem__ecfp_cluster", "chem__supercluster", "max_tanimoto_to_base",
                       "nearest_base_partner", "n_base_above_0_7"]
    (entrants.reindex(columns=entrant_columns) if not entrants.empty
     else pd.DataFrame(columns=entrant_columns)).to_csv(output_dir / "cohort_entrants.csv", index=False)

    report = render_report(
        stamp=stamp, args=args, chemistry=chemistry, cohorts=cohorts, stability=stability,
        comparison=comparison, entrants=entrants, entrant_audit=entrant_audit,
        source_shape=(int(source.shape[0]), int(source.shape[1])),
        dataset_sha=sha256_file(args.dataset), descriptor_path=descriptor_path,
        elapsed=time.time() - started)

    summary = {
        "stamp": stamp,
        "layer": "gen6",
        "kind": "chemistry_map",
        "threshold": float(args.threshold),
        "chemistry_audit": chemistry.audit,
        "cohorts": {str(k): {"status": v.status, "error": v.error, "audit": v.audit}
                    for k, v in cohorts.items()},
        "partition_stability": {str(k): v for k, v in stability.items()},
        "cohort_comparison": comparison,
        "entrants": entrant_audit,
        "note": "target-independent; nothing in this run reads log_D",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    # -- provenance --------------------------------------------------------- #
    manifest = RunManifest(layer="gen6", run_id=f"gen6_chemistry_{stamp}")
    manifest.record_dataset(dataset_path=args.dataset, source_frame=source,
                            descriptor_path=descriptor_path,
                            descriptor_frame=descriptors)
    manifest.record_code(CODE_PATHS, repo_root=REPO_ROOT)
    # The "feature registry" of a run that fits nothing is the fingerprint column
    # list the whole map is derived from.  Recording it makes the required key
    # meaningful rather than a placeholder: a source table with a different
    # fingerprint width would produce a different hash and fail aggregation.
    fingerprint_columns = [c for c in source.columns
                           if c.startswith("ecfp_") and c[len("ecfp_"):].isdigit()]
    manifest.record_features(feature_sets={"ecfp_fingerprint": fingerprint_columns})
    manifest.record_chemistry(definition={
        **manifest_json["definition"],
        "threshold": float(args.threshold),
        "n_extractants": chemistry.audit["n_extractants"],
        "n_ecfp_clusters": chemistry.audit["n_ecfp_clusters"],
        "n_superclusters": chemistry.audit["n_superclusters"],
        "chemistry_map_sha256": sha256_frame(chemistry.table, sort_rows_by=["extractant"]),
    })
    manifest.record_provenance(state={
        "variant": "legacy_compatible",
        "reason": "the map covers every extractant in the source table; no provenance filter applies",
    })
    # A map is not a model and has no folds.  The keys are still recorded, with
    # their emptiness explained, because validate_run is fail-closed on the whole
    # required list and a silently missing key is exactly what it exists to catch.
    manifest.record_split(definition={
        "kind": "none",
        "reason": "the chemistry map is split-independent; no rows are held out",
        "cohorts_checked": {str(k): {"min_cells": k, "status": v.status,
                                     "n_extractants": v.audit.get("extractants"),
                                     "n_rows": v.audit.get("rows")} for k, v in cohorts.items()},
    }, folds=[])
    manifest.record_preprocessing([
        {"step": "read source parquet", "path": str(args.dataset)},
        {"step": "build_chemistry_map", "threshold": float(args.threshold),
         "descriptors": str(descriptor_path) if descriptor_path else None},
        {"step": "build_level_dataset for the stability check",
         "min_rows_per_extractant": sorted(cohorts), "replicate_policy": args.replicate_policy,
         "drop_below_log_d": args.log_d_floor},
    ])
    manifest.record("model_seed", None)
    manifest.record("model_seed_status", "no model is fitted by this run")
    manifest.record("split_seeds", [])
    manifest.record("split_seeds_status", "no split is drawn by this run")
    manifest.record("cohort_definition", {
        "shared_cohort_min_cells": EXPANDED_MIN_CELLS,
        "base_min_cells": BASE_MIN_CELLS,
        "replicate_policy": args.replicate_policy,
        "drop_below_log_d": args.log_d_floor,
        "built": {str(k): v.status for k, v in cohorts.items()},
    })
    manifest.record("artifact_roles", {
        "chemistry_map.parquet": "one row per extractant; frozen chem__ labels",
        "chemistry_map.parquet.similarity.npz": "dense Tanimoto matrix sidecar of the map",
        "nearest_neighbor_matrix.npz": "standalone similarity matrix deliverable",
        "chemistry_cluster_manifest.json": "cluster definition plus realised group membership",
        "cohort_entrants.csv": "extractants eligible at min_cells=3 but not at 10",
        "chemistry_report.md": "human-readable audit",
    })

    (output_dir / "chemistry_report.md").write_text(report)
    payload = manifest.write(output_dir)

    checks: dict[str, Any] = {
        "map_covers_every_source_extractant": {
            "ok": chemistry.audit["n_extractants"] == int(source["canonical_smiles"].nunique()),
            "n_map": chemistry.audit["n_extractants"],
            "n_source": int(source["canonical_smiles"].nunique()),
        },
        "similarity_matrix_square_and_symmetric": {
            "ok": bool(chemistry.similarity.shape == (len(chemistry), len(chemistry))
                       and np.allclose(chemistry.similarity, chemistry.similarity.T, atol=1e-6)),
            "shape": list(chemistry.similarity.shape),
        },
        "map_round_trips": {"ok": False},
    }
    reloaded = ChemistryMap.from_parquet(map_path)
    checks["map_round_trips"] = {
        "ok": bool(reloaded.extractants == chemistry.extractants
                   and np.array_equal(reloaded.similarity, chemistry.similarity)
                   and sha256_frame(reloaded.table, sort_rows_by=["extractant"])
                   == sha256_frame(chemistry.table, sort_rows_by=["extractant"])),
        "n_extractants": len(reloaded),
    }
    for min_cells, check in stability.items():
        checks[f"partition_stability_min_cells_{min_cells}"] = check
    if not stability:
        # No cohort was built, so the load-bearing claim of the artifact is
        # unproven.  That is a failed validation, not a silent pass.
        checks["partition_stability_ran"] = {
            "ok": False, "reason": "no cohort was built, so the freeze is unverified"}

    validation = validate_run(output_dir, manifest=payload, required_artifacts=REQUIRED_ARTIFACTS,
                              checks=checks)
    success = write_success(output_dir, manifest=payload, validation=validation)
    log(f"validation ok={validation['ok']}"
        + ("" if validation["ok"] else f"; failed={validation['failed_checks']} "
                                       f"missing={validation['missing_artifacts']}"))

    print()
    print(report)
    log(f"written to {output_dir} in {time.time() - started:.1f}s "
        f"({'_SUCCESS.json' if success else '_FAILED.json'})")
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
