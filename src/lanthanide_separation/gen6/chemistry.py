"""The frozen all-190 chemistry map — one description of chemical space for the whole generation.

Every cohort this project has trained on was carved out of the same 190
extractants by an eligibility rule, and every study then re-derived its cluster
labels *inside* its own cohort.  That is fine until two cohorts must be compared,
at which point the labels are no longer the same objects and "the same
super-cluster" stops meaning anything.  Gen6 therefore computes chemistry once,
over all 190 extractants, target-independently, and freezes it.

Two properties make the freeze safe, both measured rather than assumed, and the
first needs stating carefully because the obvious version of it is vacuous:

* **The freeze is conservative, never finer.**  ``levels.build_level_dataset``
  computes ``tanimoto_cluster`` *before* it applies the ``min_rows`` filter, so
  the labels a cohort carries were already derived from all 190 extractants.
  Comparing the frozen map against *those* (:func:`partition_stability`) is
  therefore a genuine check only at the bit-identical level, where a
  fingerprint's identity cannot depend on which other molecules are present; at
  the super-cluster level it is close to a tautology, and it is retained only
  because it is the check that matters for *reproducing gen5* — gen5's folds used
  exactly those labels.  The honest comparison, :func:`freeze_is_conservative`,
  re-runs single linkage on the cohort alone.  Measured: the frozen map **splits
  no cohort-local group** (0 at both levels, both cohorts) but does **merge**
  three of them at ``min_cells = 10`` (45 from-scratch chemotypes become 40) and
  one at ``min_cells = 3`` (80 become 79), because a ligand outside the cohort
  bridges them.  Merging makes a held-out chemotype *larger* and the hold-out
  *stricter*; splitting would be the dangerous direction and does not occur.
* the *label strings* still differ between a frozen map and a cohort-local
  labelling, and fold assignment permutes ``np.unique(labels)``, so a
  reproduction run must keep the cohort-local strings.  The frozen labels are
  therefore exposed under a ``chem__`` prefix and never overwrite the columns the
  gen5 harness produces.

The map also carries what a coordination chemist would ask for first — the
donor-atom census, denticity, and a scaffold family — because the gen5 run found
the 13-column donor census to be the *only* block whose advantage grows as the
test ligand gets less similar to training chemistry.  Families are assigned from
the rdkit-derived motif counts already frozen in
``ligand_2d_descriptors.parquet`` when that table is supplied; from rdkit
directly if it happens to be installed; and from a declared SMILES-substring
heuristic otherwise.  Which route was used is recorded per extractant in
``chem__family_source``, because a family label of unknown provenance is worse
than none.

Nothing here reads ``log_D``.  :func:`build_chemistry_map` is given the source
table only for the identity, fingerprint and coverage columns, and
``tests/test_gen6_chemistry.py`` proves the map is unchanged when the target is
replaced by NaN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..levels import (
    LEVEL_DONOR_SCALAR_COLUMNS, LEVEL_LIGAND_SCALAR_COLUMNS, TANIMOTO_CLUSTER_THRESHOLD,
    condition_labels, ecfp_cluster_labels, tanimoto_cluster_labels,
)

FINGERPRINT_COLUMN_PREFIX = "ecfp_"
DEFAULT_SUPERCLUSTER_THRESHOLD = TANIMOTO_CLUSTER_THRESHOLD
#: Neighbour-count thresholds carried with every prediction for the OOD layer.
NEIGHBOUR_THRESHOLDS: tuple[float, ...] = (0.5, 0.7, 0.8)
#: Sidecar holding the dense similarity matrix beside a saved map.
SIMILARITY_SUFFIX = ".similarity.npz"

#: Donor types the dataset's ``DONOR_TYPES`` JSON list can contain.
DONOR_TYPE_VOCAB: tuple[str, ...] = (
    "O(amide_carbonyl)", "O(ether)", "N(aromatic)", "N(amine)", "S(donor)",
    "O(hydroxyl)", "O(ester_carbonyl)", "O(carbonyl)",
)

#: Scaffold families, in decision order.  The first matching rule wins, so the
#: order encodes chemical priority: a diglycolamide is a diglycolamide even
#: though it also contains amide and ether oxygens.
#: ``amide_other`` is deliberately not called "monoamide": it holds every amide
#: that is neither a diglycolamide nor a malonamide, and that includes genuine
#: di- and tri-amides (DOODA, NTAamide, succinamides, picolinamides).  Calling it
#: monoamide was a misnomer that a reader would take as a chemical claim.
FAMILY_ORDER: tuple[str, ...] = (
    "diglycolamide", "malonamide", "phosphoryl", "n_heterocyclic_polydentate",
    "podand_ether", "amide_other", "sulfur_donor", "hydroxyl_acid", "other",
)


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

def canonical_group_labels(members: Mapping[str, str], *, prefix: str) -> dict[str, str]:
    """Relabel groups deterministically as ``<prefix>000`` … , ordered by first member.

    ``levels.tanimoto_cluster_labels`` numbers its clusters in the order the rows
    happen to arrive (scipy's ``fcluster`` numbering depends on input order), so
    the same chemistry can come back as ``tan002`` or ``tan003`` depending on how
    the parquet was sorted.  The *partition* is stable; the strings are not.  A
    frozen artifact that gets hashed into a manifest cannot have unstable
    strings, so the map renumbers by the alphabetically first member extractant.

    ``members`` maps extractant -> raw group label; the return maps raw label ->
    canonical label.
    """
    by_group: dict[str, list[str]] = {}
    for extractant, group in members.items():
        by_group.setdefault(str(group), []).append(str(extractant))
    ordered = sorted(by_group, key=lambda g: min(by_group[g]))
    width = max(3, len(str(max(0, len(ordered) - 1))))
    return {group: f"{prefix}{i:0{width}d}" for i, group in enumerate(ordered)}


def tanimoto_matrix(bits: np.ndarray) -> np.ndarray:
    """Dense pairwise Tanimoto over a binary fingerprint matrix (rows = molecules)."""
    x = np.ascontiguousarray(np.asarray(bits) > 0).astype(np.int32)
    intersection = x @ x.T
    counts = x.sum(axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, intersection / union, 0.0)
    np.fill_diagonal(similarity, 1.0)
    # float64, deliberately.  An earlier version returned float32, and
    # float32(0.7) = 0.69999999 < 0.7, so four pairs sitting at exactly 7/10 fell
    # on the wrong side of the threshold: rebuilding the clustering from the
    # shipped matrix gave 99 chemotypes where the map itself (computed in float64
    # by levels.tanimoto_cluster_labels) has 98.  190x190 float64 is 289 kB.
    return similarity.astype(np.float64)


# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #

def _family_from_motifs(row: Mapping[str, float], census: Mapping[str, float] | None = None) -> str | None:
    """Family from the rdkit-derived motif counts, with a donor-census fallback.

    Returns ``None`` when the motif columns are degenerate for this molecule, so
    the caller can fall back rather than file it under ``other``.  Without that,
    ``other`` becomes a detector-failure bucket masquerading as a chemical class:
    two 2,6-pyridine-dicarboxamides land there because their descriptor row has
    ``n_carbonyl_O = 2`` but ``n_amide_N = 0``, and four more have
    ``n_donor_atoms = 0`` while the ``DONOR_TYPES`` census lists 2–6 donors.
    """
    def count(name: str) -> float:
        value = row.get(f"lig2d__hc__{name}")
        return float(value) if value is not None and not pd.isna(value) else 0.0

    if count("n_dga_motifs") >= 1:
        return "diglycolamide"
    if count("n_malonamide_motifs") >= 1:
        return "malonamide"
    if count("n_phosphoryl_O") >= 1:
        return "phosphoryl"
    if count("n_aromatic_N") >= 2:
        return "n_heterocyclic_polydentate"
    if count("n_ether_O") >= 2 and count("n_amide_N") == 0:
        return "podand_ether"
    if count("n_amide_N") >= 1 and count("n_carbonyl_O") >= 1:
        return "amide_other"
    if count("n_hydroxyl_O") >= 1:
        return "hydroxyl_acid"
    # the motif columns say nothing about this molecule -- fall back rather than
    # assert a family, and let the caller record which route was used
    if census:
        return _family_from_donors(census)
    return None


def _family_from_donors(census: Mapping[str, float]) -> str | None:
    """Coarse family from the ``DONOR_TYPES`` census, used only as a fallback."""
    def n(key: str) -> float:
        value = census.get(f"chem__donor__{key}")
        return float(value) if value is not None and not pd.isna(value) else 0.0

    if n("S(donor)") >= 1:
        return "sulfur_donor"
    if n("N(aromatic)") >= 2:
        return "n_heterocyclic_polydentate"
    if n("O(amide_carbonyl)") >= 1 and n("O(ether)") >= 1:
        return "diglycolamide"
    if n("O(amide_carbonyl)") >= 1:
        return "amide_other"
    if n("O(ether)") >= 2:
        return "podand_ether"
    if n("O(hydroxyl)") >= 1:
        return "hydroxyl_acid"
    return None


def _family_from_smiles(smiles: str) -> str:
    """Declared fallback heuristic.  Coarse, and labelled as such in the audit.

    It reads the canonical SMILES string only.  ``C(=O)COCC(=O)`` is the
    diglycolamide backbone; ``C(=O)CC(=O)`` the malonamide one; ``P(=O)`` a
    phosphoryl donor; ``n`` an aromatic nitrogen.  This route is used only when
    neither the frozen motif columns nor rdkit are available, and it must never
    be quoted as a curated family assignment.
    """
    text = smiles
    if "C(=O)COCC(=O)" in text or "C(=O)COC" in text and "C(=O)" in text.split("COC", 1)[-1]:
        return "diglycolamide"
    if "C(=O)CC(=O)" in text:
        return "malonamide"
    if "P(=O)" in text or "P(=S)" in text:
        return "phosphoryl"
    if text.count("n") >= 2:
        return "n_heterocyclic_polydentate"
    if text.count("OC") >= 2 and "N" not in text:
        return "podand_ether"
    if "C(=O)N" in text or "NC(=O)" in text:
        return "monoamide"
    if "O)O" in text or text.endswith("O"):
        return "hydroxyl_acid"
    return "other"


def rdkit_available() -> tuple[bool, str | None]:
    try:
        import rdkit  # noqa: F401
    except ImportError:
        return False, None
    return True, getattr(__import__("rdkit"), "__version__", "unknown")


# --------------------------------------------------------------------------- #
# Donor census
# --------------------------------------------------------------------------- #

def donor_census(raw) -> dict[str, float]:
    """``DONOR_TYPES`` JSON list -> ``chem__donor__<type>`` counts plus a total."""
    out = {f"chem__donor__{t}": 0.0 for t in DONOR_TYPE_VOCAB}
    out["chem__donor__n_total"] = np.nan
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return out
    try:
        items = json.loads(raw) if isinstance(raw, str) else list(raw)
    except (ValueError, TypeError):
        return out
    for item in items:
        key = f"chem__donor__{item}"
        if key in out:
            out[key] += 1.0
    out["chem__donor__n_total"] = float(len(items))
    return out


# --------------------------------------------------------------------------- #
# The map
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChemistryMap:
    """One row per extractant plus the dense Tanimoto matrix behind it."""

    table: pd.DataFrame
    similarity: np.ndarray
    extractants: tuple[str, ...]
    audit: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.extractants)

    def index_of(self, extractant: str) -> int:
        try:
            return self.extractants.index(str(extractant))
        except ValueError as error:
            raise KeyError(f"unknown extractant {extractant!r}") from error

    def similarity_between(self, query: Sequence[str], reference: Sequence[str]) -> np.ndarray:
        rows = [self.index_of(q) for q in query]
        cols = [self.index_of(r) for r in reference]
        if not rows or not cols:
            return np.zeros((len(rows), len(cols)), dtype=np.float32)
        return self.similarity[np.ix_(rows, cols)]

    def nearest_neighbour(
        self,
        query: Sequence[str],
        reference: Sequence[str],
        *,
        exclude_self: bool = True,
        thresholds: Sequence[float] = NEIGHBOUR_THRESHOLDS,
    ) -> pd.DataFrame:
        """Similarity of each query extractant to a *reference* cohort.

        With an empty reference (or one that is entirely the query itself) the
        answer is ``nn_tanimoto = 0.0`` and zero neighbours — never NaN and never
        an exception, because "nothing similar in training" is a legitimate and
        important state, and a NaN there would quietly drop the hardest ligands
        out of every downstream average.
        """
        query = [str(q) for q in query]
        reference = [str(r) for r in reference]
        rows: list[dict] = []
        supercluster = self.table.set_index("extractant")["chem__supercluster"]
        reference_superclusters = pd.Series(
            [supercluster.get(r) for r in reference], dtype=object).value_counts()
        for name in query:
            candidates = [r for r in reference if not (exclude_self and r == name)]
            record: dict = {"extractant": name, "n_reference": len(candidates)}
            if candidates:
                sims = self.similarity_between([name], candidates)[0]
                best = int(np.argmax(sims))
                record["nn_tanimoto"] = float(sims[best])
                record["nn_partner"] = candidates[best]
                record["mean_top5_tanimoto"] = float(np.mean(np.sort(sims)[::-1][:5]))
                for threshold in thresholds:
                    record[f"n_above_{str(threshold).replace('.', '_')}"] = int((sims >= threshold).sum())
            else:
                record["nn_tanimoto"] = 0.0
                record["nn_partner"] = None
                record["mean_top5_tanimoto"] = 0.0
                for threshold in thresholds:
                    record[f"n_above_{str(threshold).replace('.', '_')}"] = 0
            own = supercluster.get(name)
            support = int(reference_superclusters.get(own, 0))
            if exclude_self and name in reference:
                support = max(0, support - 1)
            record["supercluster_support"] = support
            rows.append(record)
        return pd.DataFrame(rows)

    # -- persistence -------------------------------------------------------- #
    def to_parquet(self, path: str | Path) -> tuple[Path, Path]:
        """Write ``chemistry_map.parquet`` plus its similarity sidecar."""
        table_path = Path(path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(table_path, index=False)
        sidecar = table_path.with_suffix(table_path.suffix + SIMILARITY_SUFFIX)
        np.savez_compressed(sidecar, similarity=self.similarity,
                            extractants=np.array(self.extractants, dtype=object),
                            audit=np.array(json.dumps(self.audit, default=str)))
        return table_path, sidecar

    @classmethod
    def from_parquet(cls, path: str | Path) -> "ChemistryMap":
        table_path = Path(path)
        table = pd.read_parquet(table_path)
        sidecar = table_path.with_suffix(table_path.suffix + SIMILARITY_SUFFIX)
        if not sidecar.exists():
            raise FileNotFoundError(f"similarity sidecar missing: {sidecar}")
        with np.load(sidecar, allow_pickle=True) as payload:
            similarity = payload["similarity"]
            extractants = tuple(str(x) for x in payload["extractants"])
            audit = json.loads(str(payload["audit"]))
        return cls(table=table, similarity=similarity, extractants=extractants, audit=audit)

    def save_similarity_npz(self, path: str | Path) -> Path:
        """Standalone ``nearest_neighbor_matrix.npz`` deliverable.

        Rebuilding the chemotypes from this matrix reproduces the map's own
        partition **only if the same threshold expression is used**: single
        linkage at ``t = 1.0 - threshold``, which is ``0.30000000000000004``, not
        the literal ``0.3``.  Pairs sitting at exactly Tanimoto 7/10 fall on
        opposite sides of those two values and the rebuild comes back with 99
        groups instead of 98.  The stored array also carries the threshold for
        that reason.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, similarity=self.similarity,
                            extractants=np.array(self.extractants, dtype=object),
                            supercluster_threshold=np.array(
                                self.audit.get("supercluster_threshold",
                                               DEFAULT_SUPERCLUSTER_THRESHOLD)),
                            linkage_distance_threshold=np.array(
                                1.0 - float(self.audit.get("supercluster_threshold",
                                                           DEFAULT_SUPERCLUSTER_THRESHOLD))))
        return out


def build_chemistry_map(
    source: pd.DataFrame,
    *,
    threshold: float = DEFAULT_SUPERCLUSTER_THRESHOLD,
    ligand_descriptors: pd.DataFrame | None = None,
) -> ChemistryMap:
    """Freeze chemical space over every extractant in ``source``.

    Uses only identity, fingerprint, coordination and coverage information — never
    the target.  Coverage columns (``n_rows``, ``n_cells``, ``n_metals``,
    ``n_conditions``) describe how much data exists, which is a property of the
    experiment record, not of ``log_D``.
    """
    if "canonical_smiles" not in source.columns:
        raise KeyError("source frame needs a canonical_smiles column")
    fingerprint_columns = [c for c in source.columns if c.startswith(FINGERPRINT_COLUMN_PREFIX)
                           and c[len(FINGERPRINT_COLUMN_PREFIX):].isdigit()]
    if not fingerprint_columns:
        raise KeyError("source frame has no ecfp_<int> fingerprint columns")

    frame = source.copy()
    frame["extractant"] = frame["canonical_smiles"].astype(str)
    condition_columns = [c for c in frame.columns if c.startswith("cond__")]
    if condition_columns:
        frame["condition_id"] = condition_labels(frame, condition_columns)
    else:
        frame["condition_id"] = "unknown"

    # cluster labels through the harness's own functions, so the frozen map and
    # the cohort-local labelling can only ever differ in the label *strings*
    frame["_ecfp_cluster"] = ecfp_cluster_labels(frame, fingerprint_columns)
    frame["_supercluster"] = tanimoto_cluster_labels(frame, fingerprint_columns, threshold=threshold)

    first = frame.drop_duplicates("extractant").set_index("extractant")
    extractants = tuple(sorted(frame["extractant"].unique()))
    # canonical, row-order-independent labels (see canonical_group_labels)
    raw_ecfp = first.loc[list(extractants), "_ecfp_cluster"].astype(str).to_dict()
    raw_super = first.loc[list(extractants), "_supercluster"].astype(str).to_dict()
    ecfp_relabel = canonical_group_labels(raw_ecfp, prefix="ec")
    super_relabel = canonical_group_labels(raw_super, prefix="sc")
    bits = first.loc[list(extractants), fingerprint_columns].to_numpy()
    similarity = tanimoto_matrix(bits)

    coverage = frame.groupby("extractant").agg(
        n_rows=("extractant", "size"),
        n_metals=("metal_symbol", "nunique") if "metal_symbol" in frame.columns else ("extractant", "size"),
        n_conditions=("condition_id", "nunique"),
    )
    if {"condition_id", "metal_symbol"} <= set(frame.columns):
        cells = (frame.drop_duplicates(["extractant", "condition_id", "metal_symbol"])
                 .groupby("extractant").size().rename("n_cells"))
    else:
        cells = coverage["n_rows"].rename("n_cells")

    if "extractant_name" in frame.columns:
        counted = (frame.groupby(["extractant", "extractant_name"]).size()
                   .rename("n").reset_index().sort_values(["extractant", "n"],
                                                          ascending=[True, False]))
        name_of = {k: str(v) for k, v in
                   counted.drop_duplicates("extractant").set_index("extractant")["extractant_name"].items()}
        name_counts = counted.groupby("extractant")["extractant_name"].nunique().to_dict()
        alternatives = (counted.groupby("extractant")["extractant_name"]
                        .apply(lambda x: "|".join(map(str, list(x)[1:6]))).to_dict())
    else:
        name_of, name_counts, alternatives = {}, {}, {}

    records: list[dict] = []
    descriptor_lookup: dict[str, dict] = {}
    if ligand_descriptors is not None and "canonical_smiles" in ligand_descriptors.columns:
        descriptor_lookup = (ligand_descriptors.drop_duplicates("canonical_smiles")
                             .set_index("canonical_smiles").to_dict("index"))
    has_rdkit, rdkit_version = rdkit_available()

    for name in extractants:
        row = first.loc[name]
        record: dict = {
            "extractant": name,
            # The MOST COMMON name, not the first row's.  `drop_duplicates` took the
            # first row, which labelled TODGA (1,585 rows) as "CITAM" (6 rows) —
            # 17 of the 190 SMILES carry more than one name, because the CORDIS
            # block records the aqueous holdback agent in the name field while the
            # SMILES stays TODGA.  The count of alternatives is carried too, so the
            # conflation is visible rather than hidden behind one label.
            "extractant_name": name_of.get(name, ""),
            "n_extractant_names": int(name_counts.get(name, 0)),
            "extractant_name_alternatives": alternatives.get(name, ""),
            "n_rows": int(coverage.loc[name, "n_rows"]),
            "n_cells": int(cells.loc[name]),
            "n_metals": int(coverage.loc[name, "n_metals"]),
            "n_conditions": int(coverage.loc[name, "n_conditions"]),
            "chem__ecfp_cluster": ecfp_relabel[str(row["_ecfp_cluster"])],
            "chem__supercluster": super_relabel[str(row["_supercluster"])],
            "chem__ecfp_fingerprint_sha1": str(row["_ecfp_cluster"]),
            "chem__n_fingerprint_bits": int(np.sum(bits[extractants.index(name)] > 0)),
        }
        record.update(donor_census(row.get("DONOR_TYPES")))
        for column, target in (("DENTATE", "chem__dentate"), ("coreCN", "chem__core_cn"),
                               ("n_ligs", "chem__n_ligands"), ("n_fill", "chem__n_fill")):
            if column in first.columns:
                value = row.get(column)
                record[target] = float(value) if value is not None and not pd.isna(value) else np.nan
        for column in LEVEL_LIGAND_SCALAR_COLUMNS:
            if column in first.columns:
                value = row.get(column)
                record[f"chem__physchem__{column}"] = float(value) if not pd.isna(value) else np.nan
        descriptors = descriptor_lookup.get(name)
        family, source = None, None
        if descriptors is not None and any(k.startswith("lig2d__hc__") for k in descriptors):
            family = _family_from_motifs(descriptors)
            source = "lig2d_hc_motifs"
            if family is None:
                family = _family_from_donors(record)
                source = "donor_census_fallback"
        if family is None:
            family = _family_from_donors(record)
            source = "donor_census" if family else None
        if family is None:
            family = _family_from_smiles(name)
            source = "smiles_heuristic"
        record["chem__family"] = family
        record["chem__family_source"] = source
        records.append(record)

    table = pd.DataFrame(records)
    sizes = table["chem__supercluster"].value_counts()
    table["chem__supercluster_size"] = table["chem__supercluster"].map(sizes).astype(int)
    off_diagonal = similarity.copy()
    np.fill_diagonal(off_diagonal, -1.0)
    best = off_diagonal.argmax(axis=1)
    table["chem__nn_within_all_tanimoto"] = [float(off_diagonal[i, best[i]]) for i in range(len(extractants))]
    table["chem__nn_within_all_partner"] = [extractants[int(best[i])] for i in range(len(extractants))]

    upper = similarity[np.triu_indices(len(extractants), k=1)]
    audit = {
        "n_extractants": len(extractants),
        "n_ecfp_clusters": int(table["chem__ecfp_cluster"].nunique()),
        "n_superclusters": int(table["chem__supercluster"].nunique()),
        "supercluster_threshold": float(threshold),
        "n_fingerprint_columns": len(fingerprint_columns),
        "rdkit_available": has_rdkit,
        "rdkit_version": rdkit_version,
        "family_counts": table["chem__family"].value_counts().to_dict(),
        "family_source_counts": table["chem__family_source"].value_counts().to_dict(),
        "similarity_quantiles": {
            q: float(np.quantile(upper, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        } if upper.size else {},
        "nn_within_all_quantiles": {
            q: float(np.quantile(table["chem__nn_within_all_tanimoto"], q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "largest_supercluster_extractants": int(table["chem__supercluster_size"].max()),
        "descriptor_table_supplied": ligand_descriptors is not None,
        "extractants_with_multiple_names": int(sum(1 for v in name_counts.values() if v > 1)),
        "max_names_on_one_extractant": int(max(name_counts.values())) if name_counts else 0,
    }
    return ChemistryMap(table=table, similarity=similarity, extractants=extractants, audit=audit)


# --------------------------------------------------------------------------- #
# Stability of the freeze
# --------------------------------------------------------------------------- #

def partition_stability(
    chemistry: ChemistryMap, cohort_frame: pd.DataFrame,
    *, local_cluster_column: str = "ecfp_cluster",
    local_supercluster_column: str = "tanimoto_cluster",
) -> dict:
    """Does the frozen partition, restricted to a cohort, equal that cohort's own?

    Two partitions of the same set agree when no local group spans two frozen
    groups **and** no frozen group spans two local ones.  Label strings are
    ignored — only the induced partition matters.  This is the check that makes
    it safe to use one frozen map across cohorts that were historically labelled
    independently.
    """
    local = cohort_frame.drop_duplicates("extractant")[
        ["extractant"] + [c for c in (local_cluster_column, local_supercluster_column)
                          if c in cohort_frame.columns]]
    merged = local.merge(chemistry.table[["extractant", "chem__ecfp_cluster", "chem__supercluster"]],
                         on="extractant", how="left", validate="one_to_one")
    unmapped = int(merged["chem__ecfp_cluster"].isna().sum())
    report: dict = {"n_extractants": int(len(merged)), "unmapped_extractants": unmapped, "ok": unmapped == 0}
    for local_column, frozen_column, label in (
        (local_cluster_column, "chem__ecfp_cluster", "ecfp"),
        (local_supercluster_column, "chem__supercluster", "supercluster"),
    ):
        if local_column not in merged.columns:
            report[label] = {"status": "local labels absent"}
            continue
        crosstab = pd.crosstab(merged[local_column].astype(str), merged[frozen_column].astype(str))
        splits = int(((crosstab > 0).sum(axis=1) > 1).sum())
        merges = int(((crosstab > 0).sum(axis=0) > 1).sum())
        report[label] = {
            "n_local": int(merged[local_column].nunique()),
            "n_frozen_restricted": int(merged[frozen_column].nunique()),
            "local_groups_spanning_multiple_frozen": splits,
            "frozen_groups_spanning_multiple_local": merges,
            "identical_partition": splits == 0 and merges == 0,
        }
        report["ok"] = report["ok"] and report[label]["identical_partition"]
    return report


def cohort_local_partition(
    source: pd.DataFrame, extractants: Sequence[str], *,
    threshold: float = DEFAULT_SUPERCLUSTER_THRESHOLD,
) -> pd.DataFrame:
    """Re-cluster **only** the given extractants, from scratch.

    This exists because :func:`partition_stability` alone cannot answer the
    question it appears to answer.  ``levels.build_level_dataset`` computes
    ``tanimoto_cluster`` *before* applying the ``min_rows`` filter, so the labels
    it carries were already derived from every extractant in the source table.
    Comparing the frozen all-190 map against those is close to a tautology at the
    super-cluster level (it is a genuine check at the bit-identical level, where
    a fingerprint's identity cannot depend on which other molecules are present).

    The honest comparison re-runs single linkage on the cohort alone, which is
    what a study that had only ever seen those ligands would have computed.
    """
    fingerprints = [c for c in source.columns if c.startswith(FINGERPRINT_COLUMN_PREFIX)
                    and c[len(FINGERPRINT_COLUMN_PREFIX):].isdigit()]
    keep = {str(e) for e in extractants}
    subset = source[source["canonical_smiles"].astype(str).isin(keep)]
    if subset.empty:
        raise ValueError("none of the requested extractants are in the source frame")
    return pd.DataFrame({
        "extractant": subset["canonical_smiles"].astype(str).to_numpy(),
        "local_ecfp_cluster": ecfp_cluster_labels(subset, fingerprints).to_numpy(),
        "local_supercluster": tanimoto_cluster_labels(
            subset, fingerprints, threshold=threshold).to_numpy(),
    }).drop_duplicates("extractant").reset_index(drop=True)


def freeze_is_conservative(
    chemistry: ChemistryMap, source: pd.DataFrame, extractants: Sequence[str],
    *, threshold: float = DEFAULT_SUPERCLUSTER_THRESHOLD,
) -> dict:
    """Compare the frozen map against a genuine from-scratch clustering of a cohort.

    The property that actually matters is not "identical" but **never finer**: a
    frozen group may *merge* two groups a cohort-local clustering would separate
    (a ligand outside the cohort bridges them), which makes a held-out chemotype
    larger and the hold-out stricter.  A frozen group that *split* a local one
    would be the dangerous direction — it would put chemistry on both sides of a
    fold — and is reported as ``splits``.
    """
    local = cohort_local_partition(source, extractants, threshold=threshold)
    merged = local.merge(chemistry.table[["extractant", "chem__ecfp_cluster", "chem__supercluster"]],
                         on="extractant", how="left", validate="one_to_one")
    report: dict = {"n_extractants": int(len(merged)), "threshold": float(threshold)}
    for local_column, frozen_column, label in (
        ("local_ecfp_cluster", "chem__ecfp_cluster", "ecfp"),
        ("local_supercluster", "chem__supercluster", "supercluster"),
    ):
        crosstab = pd.crosstab(merged[local_column].astype(str), merged[frozen_column].astype(str))
        splits = int(((crosstab > 0).sum(axis=1) > 1).sum())
        merges = int(((crosstab > 0).sum(axis=0) > 1).sum())
        report[label] = {
            "n_local_from_scratch": int(merged[local_column].nunique()),
            "n_frozen_restricted": int(merged[frozen_column].nunique()),
            "local_groups_the_frozen_map_splits": splits,
            "frozen_groups_merging_several_local": merges,
            "identical_partition": splits == 0 and merges == 0,
            "conservative": splits == 0,
        }
    report["ok"] = all(report[k]["conservative"] for k in ("ecfp", "supercluster"))
    return report


def cluster_manifest(chemistry: ChemistryMap) -> dict:
    """``chemistry_cluster_manifest.json``: the definition plus the realised groups."""
    grouped = (chemistry.table.groupby("chem__supercluster")["extractant"].apply(list).to_dict())
    ecfp_groups = (chemistry.table.groupby("chem__ecfp_cluster")["extractant"].apply(list).to_dict())
    return {
        "definition": {
            "identity": "canonical_smiles",
            "fingerprint": "precomputed 2048-bit ECFP columns from the source table",
            "ecfp_cluster": "sha1 of the bit vector (bit-identical fingerprints)",
            "supercluster": f"single-linkage agglomeration at Tanimoto >= "
                            f"{chemistry.audit.get('supercluster_threshold')}",
            "target_independent": True,
        },
        "audit": chemistry.audit,
        "superclusters": {k: sorted(v) for k, v in grouped.items()},
        "ecfp_clusters": {k: sorted(v) for k, v in ecfp_groups.items()},
    }


def novelty_of_cohort(
    chemistry: ChemistryMap, *, query: Iterable[str], reference: Iterable[str],
) -> pd.DataFrame:
    """Convenience wrapper: nearest-neighbour table of one cohort against another."""
    return chemistry.nearest_neighbour(list(query), list(reference))
