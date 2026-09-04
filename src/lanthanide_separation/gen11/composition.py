"""What each gen11 arm is actually made of — new chemistry, or the same chemistry again?

The transfer brief's §9 exists because a row count is the easiest number in this
project to mislead yourself with.  The archive offers ~5.4k auxiliary
model-ready records against a 5,248-row cohort: a nominal doubling.  Whether
that is worth anything depends entirely on *where* the rows sit.  gen6 already
measured the shape of this trap once — chemical coverage, not capacity, is the
bottleneck — so an arm that adds five thousand rows of one actinide campaign on
ligands the cohort already contains is a null result waiting to be
misattributed to the model.

This module answers the question with counts rather than adjectives:

``arm_composition``
    rows / extractant systems / structures / chemotypes / metals / publications
    / series, for the auxiliary pool alone *and* for the training pool the arm
    would actually see (cohort ∪ auxiliary).  The cohort side is expressed in
    the same archive records the auxiliary side is, via
    :func:`overlap.build_overlap_map`, so the two halves are commensurable
    instead of being counted by two different rules.

``chemotype_distance``
    for every auxiliary structure, the maximum Tanimoto to any *cohort*
    structure.  This is what turns "new chemotypes" from a claim into a
    measurement.  A structure is mapped onto a cohort chemotype when that
    maximum reaches the same 0.7 single-linkage threshold
    (``levels.TANIMOTO_CLUSTER_THRESHOLD``) that defined the cohort's 79
    clusters in the first place; below it the structure is in *no* cohort
    chemotype and is genuinely new chemistry.  Using a second, looser rule here
    would let an arm "add chemotypes" that the fold-grouping would still treat
    as seen.

``curve_inventory_by_arm``
    the archive's own curve definition (stage04: one series, exactly one axis
    varying, everything else fixed, ``MIN_CURVE_POINTS = 3``) recomputed on each
    arm's record subset — because an arm that samples rows can hold most of a
    titration without holding a usable curve, and gen8/gen9 showed the curve,
    not the row, is the unit that carries shape information.

``concentration_diagnostics``
    the repetition side: largest-publication / largest-system / largest-metal
    row share, and the effective number of independent units
    (``exp`` of the Shannon entropy of the row shares) for publication,
    structure and chemotype.  gen2's macro/micro trap was one extractant
    holding 43.6 % of pairs; these columns are there so the same trap cannot be
    re-entered silently.

The near / mid / far bucketing is **not** invented here.  It is read back from
gen10's own cut points (see :func:`gen10_distance_terciles`) so that a gen11
"far" structure means what a gen10 "far" ligand meant.

The arm pools (:func:`build_arm_pools`) are defined here because the composition
table is the first thing that needs them; ``gen11.arms`` should import them
rather than restate them, so a sampler change cannot make the published
composition table describe a different experiment from the one that ran.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..levels import TANIMOTO_CLUSTER_THRESHOLD
from .overlap import ARCHIVE_DIR, REPO_ROOT, OverlapMap, build_overlap_map

OUT_DIR = REPO_ROOT / "runs" / "gen11_transfer" / "composition"
CURVE_INVENTORY = ARCHIVE_DIR / "audit" / "curve_inventory.parquet"
CURVE_MEMBERSHIP = ARCHIVE_DIR / "intermediate" / "curve_membership.parquet"
GEN10_ADAPTATION_CURVES = (
    REPO_ROOT / "runs" / "gen10_final" / "budget_simulation" / "adaptation_curves.csv"
)

#: The dataset's own fingerprint.  Verified bit-for-bit against the cohort's
#: ``ecfp_*`` columns by :func:`assert_fingerprint_reproduces_cohort` on every
#: run; radius 3 and radius 1 both miss all 152 ligands, so this is not a
#: convention that can drift unnoticed.
ECFP_RADIUS = 2
ECFP_BITS = 2048

#: stage04's rule, restated so the arm-restricted recomputation is the same rule.
MIN_CURVE_POINTS = 3

#: Auxiliary rows must clear the archive's own readiness tier.  B rows carry
#: caveats the frozen cohort's rows do not, so mixing them in would confound
#: "more data" with "worse data".
AUX_READINESS = "A_model_ready"

#: Deterministic sampler seed for the two size-matched arms.  Fixed before any
#: result was seen, as the brief requires.
MATCH_SEED = 11

#: Curve axes reported separately, in the brief's language.
CURVE_AXIS_GROUPS: Mapping[str, str] = {
    "extractant": "extractant_titration",
    "acid": "acid_titration",
    "metal_series": "metal_series",
}

ARM_NAMES: tuple[str, ...] = (
    "A_GEN10_CONTROL",
    "B_LN_EXPANDED",
    "C_LN_PLUS_ACTINIDES",
    "D_LN_PLUS_NON_ACTINIDE",
    "E_LN_PLUS_ALL",
    "F_ACTINIDES_ONLY_MATCHED",
    "G_RANDOM_AUX_MATCHED",
)


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #

def _morgan_generator():
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)


def ecfp_matrix(smiles: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Binary fingerprints for ``smiles``; second element lists what rdkit rejected."""
    from rdkit import Chem, rdBase

    generator = _morgan_generator()
    rows: list[np.ndarray] = []
    unparsed: list[str] = []
    for item in smiles:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(str(item))
        if mol is None:
            unparsed.append(str(item))
            rows.append(np.zeros(ECFP_BITS, dtype=np.int8))
            continue
        rows.append(np.asarray(generator.GetFingerprint(mol), dtype=np.int8))
    return np.vstack(rows) if rows else np.zeros((0, ECFP_BITS), np.int8), unparsed


def ecfp_columns(frame: pd.DataFrame) -> list[str]:
    """``ecfp_<int>`` columns in *bit* order, not in the file's lexicographic order.

    Reading them in column order gives ``ecfp_0, ecfp_1, ecfp_10, …`` on some
    writers, which silently permutes the fingerprint and makes every Tanimoto
    wrong by a plausible-looking amount rather than by an obvious one.
    """
    cols = [c for c in frame.columns if str(c).startswith("ecfp_") and str(c)[5:].isdigit()]
    return sorted(cols, key=lambda c: int(str(c)[5:]))


def assert_fingerprint_reproduces_cohort(cohort_frame: pd.DataFrame) -> dict:
    """The generator must rebuild the cohort's own ECFP block bit-for-bit.

    Every auxiliary Tanimoto in this module is computed against the cohort with
    a *recomputed* fingerprint.  If the recomputation is not the dataset's own
    fingerprint, "new chemistry" is measured on a different metric from the one
    the folds were grouped with, and the whole table is decorative.
    """
    columns = ecfp_columns(cohort_frame)
    if len(columns) != ECFP_BITS:
        raise SystemExit(f"cohort has {len(columns)} ecfp bits, expected {ECFP_BITS}")
    ligands = cohort_frame.drop_duplicates("extractant").set_index("extractant")[columns]
    stored = ligands.to_numpy().astype(np.int8)
    recomputed, unparsed = ecfp_matrix(list(ligands.index.astype(str)))
    if unparsed:
        raise SystemExit(f"rdkit could not parse {len(unparsed)} cohort SMILES")
    exact = int((recomputed == stored).all(axis=1).sum())
    if exact != len(ligands):
        raise SystemExit(
            f"recomputed ECFP matches only {exact}/{len(ligands)} cohort ligands; "
            f"radius/bit convention drifted")
    return {"cohort_ligands": int(len(ligands)), "bit_exact_ligands": exact}


def tanimoto_to_reference(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """``(n_query, n_reference)`` Tanimoto between two binary fingerprint matrices."""
    q = query.astype(np.float64)
    r = reference.astype(np.float64)
    inter = q @ r.T
    union = q.sum(axis=1)[:, None] + r.sum(axis=1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0.0, inter / union, 0.0)
    return np.clip(sim, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# gen10's near / mid / far, read back rather than re-invented
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DistanceBuckets:
    """gen10's Tanimoto-distance terciles, with the file that defines them."""

    near_max_distance: float
    mid_max_distance: float
    source: str
    definition: str
    n_reference_ligands: int

    def label(self, max_tanimoto: np.ndarray) -> np.ndarray:
        distance = 1.0 - np.asarray(max_tanimoto, dtype=float)
        out = np.where(distance <= self.near_max_distance, "near",
                       np.where(distance <= self.mid_max_distance, "mid", "far"))
        return out.astype(object)


def gen10_distance_terciles(path: Path = GEN10_ADAPTATION_CURVES) -> DistanceBuckets:
    """Recover gen10's bucket boundaries from gen10's own held-out ligand table.

    ``scripts/gen10_budget_simulation.py:156`` reads

        ``gains["distance_tercile"] = pd.qcut(1.0 - curves["nn_train_tanimoto"], 3,
                                              labels=["near", "mid", "far"])``

    — equal-frequency terciles of Tanimoto *distance* to the nearest structure
    in that fold's training set, over the 99 common-cohort ligands in
    ``runs/gen10_final/budget_simulation/adaptation_curves.csv``.  The cut points
    are therefore data-defined, not constants, so they are recomputed from that
    frozen file here instead of being typed in from a report.
    """
    curves = pd.read_csv(path)
    distance = 1.0 - curves["nn_train_tanimoto"].astype(float)
    lower, upper = (float(x) for x in np.quantile(distance, [1.0 / 3.0, 2.0 / 3.0]))
    return DistanceBuckets(
        near_max_distance=lower,
        mid_max_distance=upper,
        source="scripts/gen10_budget_simulation.py:156",
        definition=(
            'pd.qcut(1.0 - nn_train_tanimoto, 3, labels=["near", "mid", "far"]); '
            "nn_train_tanimoto = max Tanimoto (Morgan r=2, 2048 bits) from the held-out "
            "ligand to the fold's training ligands"
        ),
        n_reference_ligands=int(len(curves)),
    )


# --------------------------------------------------------------------------- #
# Record sets
# --------------------------------------------------------------------------- #

def cohort_record_ids(cohort_frame: pd.DataFrame, overlap: OverlapMap) -> set[str]:
    """The archive records the 5,248 cohort rows were averaged from.

    The cohort is the bundle after filtering and replicate averaging, so this is
    a subset of the 5,992 exact matches, not all of them.  Counting the dropped
    bundle rows as cohort content would overstate the control arm.
    """
    keep = set(cohort_frame["row_id"].astype(str))
    exact = overlap.exact
    return set(exact.loc[exact["row_id"].isin(keep), "source_record_id"].astype(str))


def auxiliary_pool(overlap: OverlapMap) -> pd.DataFrame:
    """Model-ready archive records that are not frozen bundle rows."""
    aux = overlap.auxiliary_candidates
    return aux[aux["model_readiness"] == AUX_READINESS].copy()


def build_arm_pools(aux: pd.DataFrame, *, seed: int = MATCH_SEED) -> dict[str, pd.Index]:
    """Auxiliary record indices per arm.

    ``F`` and ``G`` are size-matched to ``D`` at the *row* level, which is the
    brief's wording and the only matching that makes the row counts literally
    equal.  It does fragment titrations — a randomly chosen half of a curve is
    not a curve — and that cost is not hidden: ``curve_inventory_by_arm``
    recomputes usable curves on exactly these subsets, so the fragmentation
    shows up as a number rather than as a caveat.

    Sampling is over records sorted by ``canonical_measurement_id`` with a fixed
    ``numpy`` generator, so the pools are reproducible across processes (gen8's
    salted-``hash()`` trap: never let set iteration order decide a split).
    """
    frame = aux.sort_values("canonical_measurement_id")
    category = frame["metal_category"].astype(str)
    is_lanthanide = category.eq("lanthanide")
    is_actinide = category.eq("actinide")

    lanthanide = frame.index[is_lanthanide]
    actinide = frame.index[is_actinide]
    non_actinide_non_lanthanide = frame.index[~is_lanthanide & ~is_actinide]
    everything = frame.index
    non_lanthanide = frame.index[~is_lanthanide]

    target = len(non_actinide_non_lanthanide)
    rng = np.random.default_rng(seed)
    matched_actinide = pd.Index(rng.choice(np.asarray(actinide), size=target, replace=False))
    rng = np.random.default_rng(seed)
    matched_random = pd.Index(rng.choice(np.asarray(non_lanthanide), size=target, replace=False))

    pools = {
        "A_GEN10_CONTROL": pd.Index([], dtype=frame.index.dtype),
        "B_LN_EXPANDED": lanthanide,
        "C_LN_PLUS_ACTINIDES": actinide,
        "D_LN_PLUS_NON_ACTINIDE": non_actinide_non_lanthanide,
        "E_LN_PLUS_ALL": everything,
        "F_ACTINIDES_ONLY_MATCHED": matched_actinide,
        "G_RANDOM_AUX_MATCHED": matched_random,
    }
    if tuple(pools) != ARM_NAMES:
        raise SystemExit(f"arm set drifted from ARM_NAMES: {tuple(pools)}")
    for name in ("F_ACTINIDES_ONLY_MATCHED", "G_RANDOM_AUX_MATCHED"):
        if len(pools[name]) != target:
            raise SystemExit(f"{name} is not size-matched to D: {len(pools[name])} != {target}")
    return pools


# --------------------------------------------------------------------------- #
# Chemotype distance
# --------------------------------------------------------------------------- #

def chemotype_distance(
    cohort_frame: pd.DataFrame, aux: pd.DataFrame, buckets: DistanceBuckets,
    *, threshold: float = TANIMOTO_CLUSTER_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """One row per auxiliary structure: how far it is from the cohort's chemistry.

    ``assigned_chemotype`` follows the cohort's own single-linkage rule: the
    structure joins the cluster of its nearest cohort structure when that
    similarity reaches ``threshold`` (0.7), and is ``NEW`` otherwise.  The NEW
    structures are then clustered among themselves at the same threshold, so
    "genuinely new chemistry" is reported as a number of new chemotypes and not
    only as a number of new SMILES.
    """
    columns = ecfp_columns(cohort_frame)
    cohort_lig = cohort_frame.drop_duplicates("extractant").set_index("extractant")
    cohort_bits = cohort_lig[columns].to_numpy().astype(np.int8)
    cohort_names = list(cohort_lig.index.astype(str))
    cohort_cluster = cohort_lig["tanimoto_cluster"].astype(str).to_numpy()

    structures = (
        aux.groupby(aux["extractant_primary_smiles"].astype(str))
        .agg(aux_rows=("canonical_measurement_id", "size"),
             n_metals=("metal_symbol", "nunique"),
             n_publications=("doi_primary_corrected", "nunique"),
             n_series=("series_id", "nunique"),
             n_systems=("extractant_system_key", "nunique"))
        .reset_index()
        .rename(columns={"extractant_primary_smiles": "structure"})
        .sort_values("structure")
        .reset_index(drop=True)
    )
    metals = (aux.groupby(aux["extractant_primary_smiles"].astype(str))["metal_symbol"]
              .apply(lambda s: "|".join(sorted(set(s.astype(str))))))
    categories = (aux.groupby(aux["extractant_primary_smiles"].astype(str))["metal_category"]
                  .apply(lambda s: "|".join(sorted(set(s.astype(str))))))
    structures["metals"] = structures["structure"].map(metals)
    structures["metal_categories"] = structures["structure"].map(categories)

    aux_bits, unparsed = ecfp_matrix(structures["structure"].tolist())
    similarity = tanimoto_to_reference(aux_bits, cohort_bits)
    best = similarity.argmax(axis=1)
    max_tanimoto = similarity[np.arange(len(structures)), best]

    structures["max_tanimoto_to_cohort"] = max_tanimoto
    structures["nearest_cohort_structure"] = [cohort_names[i] for i in best]
    structures["nearest_cohort_chemotype"] = [cohort_cluster[i] for i in best]
    structures["in_cohort"] = structures["structure"].isin(set(cohort_names))
    structures["rdkit_unparsed"] = structures["structure"].isin(set(unparsed))
    joined = max_tanimoto >= threshold
    structures["assigned_chemotype"] = np.where(
        joined, structures["nearest_cohort_chemotype"], "NEW")
    structures["is_new_chemistry"] = ~joined
    structures["gen10_distance_bucket"] = buckets.label(max_tanimoto)

    # New chemotypes: single linkage at the same threshold, among the NEW ones only.
    new_mask = structures["is_new_chemistry"].to_numpy()
    new_labels = np.array(["" for _ in range(len(structures))], dtype=object)
    n_new_clusters = 0
    if new_mask.sum() == 1:
        new_labels[new_mask] = "new000"
        n_new_clusters = 1
    elif new_mask.sum() > 1:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        sub = aux_bits[new_mask]
        dist = 1.0 - tanimoto_to_reference(sub, sub)
        np.fill_diagonal(dist, 0.0)
        dist = np.minimum(dist, dist.T)
        labels = fcluster(linkage(squareform(dist, checks=False), method="single"),
                          t=1.0 - threshold, criterion="distance")
        new_labels[new_mask] = [f"new{int(v):03d}" for v in labels]
        n_new_clusters = int(len(set(labels)))
    structures["new_chemotype_id"] = new_labels
    structures.loc[~new_mask, "new_chemotype_id"] = ""

    audit = {
        "aux_structures": int(len(structures)),
        "aux_structures_already_in_cohort": int(structures["in_cohort"].sum()),
        "aux_structures_mapped_to_cohort_chemotype": int(joined.sum()),
        "aux_structures_in_no_cohort_chemotype": int((~joined).sum()),
        "new_chemotypes_among_them": n_new_clusters,
        "rows_on_new_chemistry": int(structures.loc[~joined, "aux_rows"].sum()),
        "rows_total": int(structures["aux_rows"].sum()),
        "chemotype_threshold": float(threshold),
        "rdkit_unparsed_structures": int(len(unparsed)),
        "median_max_tanimoto_to_cohort": float(np.median(max_tanimoto)),
    }
    return structures, audit


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #

def _curve_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = pd.read_parquet(CURVE_MEMBERSHIP)
    inventory = pd.read_parquet(CURVE_INVENTORY)
    return membership, inventory


_CURVE_STAT_COLUMNS = ["curve_id", "axis_label", "n_points", "n_distinct", "span",
                       "usable", "usable_distinct_axis"]


def curve_stats_on(
    membership: pd.DataFrame, records: Iterable[str], log_d: Mapping[str, float],
) -> pd.DataFrame:
    """stage04's usability rule, recomputed on an arbitrary subset of records.

    ``usable`` is the archive's own definition, restated verbatim
    (``dataset_all_metals/scripts/stage04_series.py:205-222``): a curve is
    usable when at least ``MIN_CURVE_POINTS = 3`` of its member records carry a
    finite axis coordinate *and* a finite ``log_D``, and those coordinates have
    non-zero range.  Recomputing it over every canonical record returns the
    archive's 2,331 exactly, which is what
    :func:`assert_curve_rule_reproduces_archive` enforces.

    ``usable_distinct_axis`` additionally demands three *distinct* coordinates.
    stage04 does not check that, because ``build_curves`` guaranteed it at
    construction; on an arm's subset the guarantee is gone.  Over the full
    archive the two differ by 5 curves (four metal series and one metal
    concentration where three records sit on two coordinates), so the strict
    column is reported alongside rather than substituted for the archive rule.
    """
    keep = membership[membership["canonical_measurement_id"].isin(set(records))].copy()
    if keep.empty:
        return pd.DataFrame(columns=_CURVE_STAT_COLUMNS)
    keep["log_D"] = keep["canonical_measurement_id"].map(log_d).astype(float)
    keep = keep[np.isfinite(keep["log_D"].to_numpy()) & np.isfinite(keep["axis_value"].to_numpy())]
    if keep.empty:
        return pd.DataFrame(columns=_CURVE_STAT_COLUMNS)
    grouped = keep.groupby(["curve_id", "axis_label"], sort=False)["axis_value"]
    stats = pd.DataFrame({
        "n_points": grouped.size(),
        "n_distinct": grouped.nunique(),
        "span": grouped.max() - grouped.min(),
    }).reset_index()
    enough = (stats["n_points"] >= MIN_CURVE_POINTS) & (stats["span"] > 0.0)
    stats["usable"] = enough
    stats["usable_distinct_axis"] = enough & (stats["n_distinct"] >= MIN_CURVE_POINTS)
    return stats


def assert_curve_rule_reproduces_archive(membership: pd.DataFrame, inventory: pd.DataFrame,
                                         archive: pd.DataFrame) -> dict:
    """Re-deriving usability over *all* canonical records must return the archive's own set.

    Compared as *sets* of ``curve_id``, not as counts: two different rules can
    agree on a total while disagreeing on which curves they admit, and it is the
    membership that the per-arm subsetting depends on.
    """
    log_d = dict(zip(archive["canonical_measurement_id"].astype(str),
                     archive["log_D"].astype(float)))
    stats = curve_stats_on(membership, log_d.keys(), log_d)
    mine = set(stats.loc[stats["usable"], "curve_id"])
    theirs = set(inventory.loc[inventory["usable"].astype(bool), "curve_id"])
    if mine != theirs:
        raise SystemExit(
            f"curve usability rule drifted: {len(mine ^ theirs)} curves differ from the "
            f"archive's own inventory")
    return {
        "archive_curves_total": int(len(inventory)),
        "archive_curves_usable": int(len(theirs)),
        "recomputed_curves_usable_all_records": int(len(mine)),
        "curve_id_sets_identical": True,
        "strict_distinct_axis_rule_usable": int(stats["usable_distinct_axis"].sum()),
    }


def curve_inventory_by_arm(
    arm_records: Mapping[str, set[str]], membership: pd.DataFrame, archive: pd.DataFrame,
    structures: pd.DataFrame | None = None, cohort_structures: Iterable[str] = (),
) -> pd.DataFrame:
    """Usable curves per arm, split by axis and — when ``structures`` is given —
    by how novel the curve's ligand is.

    The novelty split is the point of the table.  gen10's shape evidence was
    thin because it rested on 25 ligands in 8 chemotypes; auxiliary curves are
    only a fix for that if they sit on chemistry the cohort does not already
    have, and "curves added" without that split cannot tell the two apart.
    """
    log_d = dict(zip(archive["canonical_measurement_id"].astype(str),
                     archive["log_D"].astype(float)))
    smiles_of_record = dict(zip(archive["canonical_measurement_id"].astype(str),
                                archive["extractant_primary_smiles"].astype(str)))
    novelty_of_structure: dict[str, str] = {}
    new_chemistry: set[str] = set()
    if structures is not None:
        novelty_of_structure = dict(zip(structures["structure"],
                                        structures["gen10_distance_bucket"]))
        # A cohort structure is not "near" its own chemistry, it *is* its own
        # chemistry; giving it a distance tercile would inflate ``near``.
        novelty_of_structure.update({str(s): "cohort" for s in cohort_structures})
        new_chemistry = set(structures.loc[structures["is_new_chemistry"], "structure"])

    rows = []
    for arm, records in arm_records.items():
        stats = curve_stats_on(membership, records, log_d)
        usable = stats[stats["usable"]] if len(stats) else stats
        counts = usable["axis_label"].value_counts() if len(usable) else pd.Series(dtype=int)
        row = {"arm": arm, "records": len(records), "usable_curves": int(len(usable)),
               "usable_curves_distinct_axis_rule": int(stats["usable_distinct_axis"].sum())
               if len(stats) else 0}
        for label, name in CURVE_AXIS_GROUPS.items():
            row[f"curves_{name}"] = int(counts.get(label, 0))
        row["curves_other_axes"] = int(len(usable) - sum(
            row[f"curves_{name}"] for name in CURVE_AXIS_GROUPS.values()))
        row["curve_points_median"] = (float(usable["n_points"].median()) if len(usable)
                                      else float("nan"))
        members = membership[membership["curve_id"].isin(set(usable["curve_id"]))
                             & membership["canonical_measurement_id"].isin(records)] \
            if len(usable) else membership.iloc[:0]
        row["rows_on_a_usable_curve"] = int(members["canonical_measurement_id"].nunique())

        if structures is not None:
            curve_structure = (
                members.assign(structure=members["canonical_measurement_id"].map(smiles_of_record))
                .groupby("curve_id")["structure"].first()
            ) if len(members) else pd.Series(dtype=str)
            row["curves_on_new_chemistry"] = int(curve_structure.isin(new_chemistry).sum())
            bucket = curve_structure.map(novelty_of_structure)
            for name in ("cohort", "near", "mid", "far"):
                row[f"curves_{name}"] = int((bucket == name).sum())
            row["curves_unclassified_structure"] = int(bucket.isna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Composition and concentration
# --------------------------------------------------------------------------- #

def _effective_units(labels: pd.Series) -> float:
    """``exp`` of the Shannon entropy of the row shares — "how many units, really".

    Missing labels are dropped rather than pooled into one bucket: a shared
    ``NaN`` publication would otherwise read as a single enormous publication
    and make the concentration look worse than it is.
    """
    counts = labels.dropna().astype(str).value_counts()
    if counts.empty:
        return float("nan")
    p = counts.to_numpy(dtype=float) / counts.sum()
    entropy = float(-(p * np.log(p)).sum())
    return float(math.exp(entropy))


def _largest_share(labels: pd.Series) -> float:
    counts = labels.dropna().astype(str).value_counts()
    if counts.empty:
        return float("nan")
    return float(counts.iloc[0] / counts.sum())


def _block_counts(records: pd.DataFrame, prefix: str) -> dict:
    return {
        f"{prefix}rows": int(len(records)),
        f"{prefix}extractant_systems": int(records["extractant_system_key"].nunique()),
        f"{prefix}structures": int(records["extractant_primary_smiles"].nunique()),
        f"{prefix}metals": int(records["metal_symbol"].nunique()),
        f"{prefix}publications": int(records["doi_primary_corrected"].nunique()),
        f"{prefix}rows_without_doi": int(records["doi_primary_corrected"].isna().sum()),
        f"{prefix}series": int(records["series_id"].nunique()),
    }


def chemotype_labeller(structures: pd.DataFrame, cohort_frame: pd.DataFrame):
    """One chemotype label per record, on the cohort's clustering wherever possible.

    A cohort structure keeps its own ``tanimoto_cluster``.  An auxiliary
    structure inherits the cluster of its nearest cohort structure when it is
    within the 0.7 threshold, and otherwise carries the ``new###`` id of the
    cluster it forms with the other unmapped auxiliary structures.  Any label
    that cannot be resolved stays ``NaN`` rather than being pooled, so a lookup
    failure cannot masquerade as one giant chemotype.
    """
    chemotype_of = dict(zip(structures["structure"], structures["assigned_chemotype"]))
    new_id_of = dict(zip(structures["structure"], structures["new_chemotype_id"]))
    cohort_chemotype_of = dict(zip(cohort_frame["extractant"].astype(str),
                                   cohort_frame["tanimoto_cluster"].astype(str)))

    def label(frame: pd.DataFrame) -> pd.Series:
        smiles = frame["extractant_primary_smiles"].astype(str)
        known = smiles.map(cohort_chemotype_of)
        mapped = smiles.map(chemotype_of)
        mapped = mapped.where(mapped != "NEW", smiles.map(new_id_of))
        return known.fillna(mapped)

    return label


def arm_composition(
    arms: Mapping[str, pd.Index], aux: pd.DataFrame, cohort_records: pd.DataFrame,
    structures: pd.DataFrame, cohort_frame: pd.DataFrame,
) -> pd.DataFrame:
    chemotype_of = dict(zip(structures["structure"], structures["assigned_chemotype"]))
    new_id_of = dict(zip(structures["structure"], structures["new_chemotype_id"]))
    bucket_of = dict(zip(structures["structure"], structures["gen10_distance_bucket"]))
    tanimoto_of = dict(zip(structures["structure"], structures["max_tanimoto_to_cohort"]))
    label = chemotype_labeller(structures, cohort_frame)

    rows = []
    for arm, index in arms.items():
        pool = aux.loc[index]
        smiles = pool["extractant_primary_smiles"].astype(str)
        assigned = smiles.map(chemotype_of)
        new_ids = smiles.map(new_id_of)
        record = {"arm": arm}
        record.update(_block_counts(pool, "aux_"))
        record["aux_cohort_chemotypes_hit"] = int(
            assigned[assigned != "NEW"].nunique()) if len(pool) else 0
        record["aux_structures_in_no_cohort_chemotype"] = int(
            smiles[assigned == "NEW"].nunique()) if len(pool) else 0
        record["aux_new_chemotypes"] = int(
            new_ids[assigned == "NEW"].nunique()) if len(pool) else 0
        record["aux_rows_on_new_chemistry"] = int((assigned == "NEW").sum()) if len(pool) else 0
        record["aux_structures_already_in_cohort"] = int(
            smiles[smiles.isin(set(cohort_frame["extractant"].astype(str)))].nunique()
        ) if len(pool) else 0

        # gen10's own near/mid/far terciles, carried per row rather than per
        # structure: 25 "far" structures matter much less than they sound if
        # they hold 300 of 5,438 rows.
        bucket = smiles.map(bucket_of)
        for name in ("near", "mid", "far"):
            record[f"aux_rows_{name}"] = int((bucket == name).sum()) if len(pool) else 0
            record[f"aux_structures_{name}"] = int(
                smiles[bucket == name].nunique()) if len(pool) else 0
        record["aux_row_weighted_median_max_tanimoto"] = (
            float(smiles.map(tanimoto_of).median()) if len(pool) else float("nan"))
        # Tanimoto 1.0 is not "similar", it is the same fingerprint: the model
        # cannot tell such a row's ligand from a cohort ligand at all.
        identical = smiles.map(tanimoto_of) >= 0.999 if len(pool) else pd.Series(dtype=bool)
        record["aux_rows_fingerprint_identical_to_cohort"] = int(identical.sum()) if len(pool) else 0
        record["aux_structures_fingerprint_identical_to_cohort"] = int(
            smiles[identical].nunique()) if len(pool) else 0

        total = pd.concat([cohort_records, pool], ignore_index=True)
        record.update(_block_counts(total, "total_"))
        total_labels = label(total)
        record["total_chemotypes"] = int(total_labels.nunique())
        record["total_chemotypes_unresolved_rows"] = int(total_labels.isna().sum())
        record["total_rows_delta_vs_control"] = int(len(total) - len(cohort_records))
        rows.append(record)
    return pd.DataFrame(rows)


def concentration_diagnostics(
    arms: Mapping[str, pd.Index], aux: pd.DataFrame, cohort_records: pd.DataFrame,
    structures: pd.DataFrame, cohort_frame: pd.DataFrame,
) -> pd.DataFrame:
    chemotype_labels = chemotype_labeller(structures, cohort_frame)

    rows = []
    for scope in ("aux", "total"):
        for arm, index in arms.items():
            pool = aux.loc[index]
            frame = pool if scope == "aux" else pd.concat([cohort_records, pool],
                                                          ignore_index=True)
            if frame.empty:
                rows.append({"arm": arm, "scope": scope, "rows": 0})
                continue
            rows.append({
                "arm": arm, "scope": scope, "rows": int(len(frame)),
                "largest_publication_share": _largest_share(frame["doi_primary_corrected"]),
                "largest_extractant_system_share": _largest_share(frame["extractant_system_key"]),
                "largest_metal_share": _largest_share(frame["metal_symbol"]),
                "largest_structure_share": _largest_share(frame["extractant_primary_smiles"]),
                "largest_series_share": _largest_share(frame["series_id"]),
                "effective_publications": _effective_units(frame["doi_primary_corrected"]),
                "effective_structures": _effective_units(frame["extractant_primary_smiles"]),
                "effective_chemotypes": _effective_units(chemotype_labels(frame)),
                "effective_metals": _effective_units(frame["metal_symbol"]),
                "effective_series": _effective_units(frame["series_id"]),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def build(out_dir: Path = OUT_DIR) -> dict:
    from ..gen10.runner import prepared_cohort

    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = prepared_cohort()
    frame = cohort.frame
    fingerprint_audit = assert_fingerprint_reproduces_cohort(frame)

    overlap = build_overlap_map(frame)
    archive = overlap.archive
    aux = auxiliary_pool(overlap)
    cohort_ids = cohort_record_ids(frame, overlap)
    cohort_records = archive[archive["source_record_id"].isin(cohort_ids)].copy()

    buckets = gen10_distance_terciles()
    structures, chem_audit = chemotype_distance(frame, aux, buckets)
    arms = build_arm_pools(aux)

    membership, inventory = _curve_tables()
    curve_check = assert_curve_rule_reproduces_archive(membership, inventory, archive)

    arm_records = {}
    for arm, index in arms.items():
        pool_ids = set(aux.loc[index, "canonical_measurement_id"].astype(str))
        cohort_measurement_ids = set(cohort_records["canonical_measurement_id"].astype(str))
        arm_records[arm] = cohort_measurement_ids | pool_ids

    cohort_smiles = set(frame["extractant"].astype(str))
    composition = arm_composition(arms, aux, cohort_records, structures, frame)
    curves = curve_inventory_by_arm(arm_records, membership, archive, structures, cohort_smiles)
    aux_only_curves = curve_inventory_by_arm(
        {arm: set(aux.loc[index, "canonical_measurement_id"].astype(str))
         for arm, index in arms.items()}, membership, archive, structures, cohort_smiles)
    aux_only_curves.insert(1, "scope", "auxiliary_only")
    curves.insert(1, "scope", "cohort_plus_auxiliary")
    curve_table = pd.concat([curves, aux_only_curves], ignore_index=True)

    concentration = concentration_diagnostics(arms, aux, cohort_records, structures, frame)

    composition.to_csv(out_dir / "arm_composition.csv", index=False)
    curve_table.to_csv(out_dir / "curve_inventory_by_arm.csv", index=False)
    structures.to_csv(out_dir / "chemotype_distance.csv", index=False)
    concentration.to_csv(out_dir / "concentration_diagnostics.csv", index=False)

    audit = {
        "cohort_rows": int(len(frame)),
        "cohort_records": int(len(cohort_records)),
        "cohort_structures": int(frame["extractant"].nunique()),
        "cohort_chemotypes": int(frame["tanimoto_cluster"].nunique()),
        "auxiliary_pool_rows": int(len(aux)),
        "fingerprint": fingerprint_audit,
        "chemotype": chem_audit,
        "curve_rule_check": curve_check,
        "distance_buckets": {
            "near_max_distance": buckets.near_max_distance,
            "mid_max_distance": buckets.mid_max_distance,
            "near_min_tanimoto": 1.0 - buckets.near_max_distance,
            "mid_min_tanimoto": 1.0 - buckets.mid_max_distance,
            "source": buckets.source,
            "definition": buckets.definition,
            "n_reference_ligands": buckets.n_reference_ligands,
        },
        "overlap": overlap.audit,
        "match_seed": MATCH_SEED,
    }
    (out_dir / "composition_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    return audit


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(build(), indent=2, default=str))
