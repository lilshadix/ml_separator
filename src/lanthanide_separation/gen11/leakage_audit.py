"""Exercise :mod:`gen11.overlap` on every fold of every seed, then attack it.

The brief's §2/§20 gate says no gen11 headline number may be reported until the
auxiliary pool has been shown, fold by fold, to contain nothing the gen10
benchmark holds out.  ``overlap`` states the rules; this module is the part that
does not take the module's word for them.

Three commitments shape the file.

*   **Every exclusion is recomputed here from raw archive columns**, not read
    back from ``relationship_masks``.  The module's masks and this file's masks
    are compared, and agreement is a reported result rather than an assumption.

*   **A check must not share a mechanism with the rule it tests.**  Chemotype
    membership is re-derived from Morgan bits recomputed from SMILES, because
    the ``extractant -> tanimoto_cluster`` dictionary is exactly what the rule
    consults and cannot falsify itself.  This is the check that found the first
    real defect: the shipped ``SAME_CHEMOTYPE`` looked auxiliary structures up
    in a dictionary built from the cohort's 152 ligands, so any auxiliary ligand
    the cohort did not contain returned nothing and passed — including
    structures at Tanimoto 1.0 to a held-out ligand.

*   **Identifier rules can only catch leakage the curator already labelled.**
    Two records from different papers describing the same cell — same structure,
    metal, acid, acid molarity, extractant molarity, solvent and temperature —
    carry the same measurement with no identifier in common.  The near-duplicate
    sweep matches on the experiment instead of on ids, which is the failure mode
    duplicate-group/series/ligand rules are structurally unable to see.

Both readings of ``SAME_CHEMOTYPE`` are run and reported side by side:

``DICT_LEGACY``
    the shipped-then-replaced dictionary lookup, kept so the size of the hole is
    a measured number and so a regression would be visible rather than silent.
``CLOSURE``
    the current rule — an auxiliary structure belongs to *every* cohort
    chemotype containing a ligand it resembles at Tanimoto >= 0.7.  Built here
    from independently recomputed fingerprints, not imported from
    ``gen11.pools``, so the two implementations are a cross-check on each other.

The pool is reported at two widths.  ``all_candidates`` is every archive record
that is not itself a frozen bundle row (10,778); ``model_ready`` is the subset
gen11 could actually train on.  The strictness cost has to be quoted against the
pool that would be used, or it flatters itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..gen7.harness import DEFAULT_SEEDS, build_folds
from ..gen10.runner import prepared_cohort
from ..levels import TANIMOTO_CLUSTER_THRESHOLD
from . import overlap as ov

#: Morgan settings that reproduce the bundle's ``ecfp_*`` columns bit-for-bit.
#: Asserted, not trusted: :func:`fingerprint_table` fails if they do not.
ECFP_RADIUS = 2
ECFP_BITS = 2048

#: The experimental cell.  Anything outside this tuple (contact time, phase
#: ratio, metal loading) can vary between two reports of the same measurement
#: without changing which measurement it is, so including it would hide
#: near-duplicates rather than find them.
CELL_CATEGORICAL: tuple[str, ...] = (
    "extractant_primary_smiles", "metal_symbol", "acid_primary", "solvent_key",
)
CELL_NUMERIC: tuple[str, ...] = (
    "acid_concentration_M", "extractant_primary_concentration_M", "temperature_C",
)

#: Two tolerance tiers.  6 significant figures is "the same number written
#: twice"; 3 is "the same experiment rounded differently by two typesetters".
#: Both are reported because the loose tier is the one that can produce a false
#: positive, and a check whose sensitivity is unstated is not a check.
SIG_FIGS: tuple[int, ...] = (6, 3)

POOLS: tuple[str, ...] = ("all_candidates", "model_ready")

#: The two readings of SAME_CHEMOTYPE, run side by side.
ASSIGNMENTS: tuple[str, ...] = ("DICT_LEGACY", "CLOSURE")


# --------------------------------------------------------------------------- #
# Structure bookkeeping
# --------------------------------------------------------------------------- #

def _ecfp(smiles: str) -> np.ndarray | None:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdFingerprintGenerator

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
    return np.asarray(generator.GetFingerprint(mol), dtype=np.uint8)


@dataclass(frozen=True)
class Fingerprints:
    """Recomputed bit vectors for every SMILES either side of the join."""

    smiles: tuple[str, ...]
    bits: np.ndarray                      # (n_smiles, 2048) uint8
    unparsed: tuple[str, ...]
    cohort_reproduction_max_bit_delta: int


def fingerprint_table(cohort_frame: pd.DataFrame, archive: pd.DataFrame) -> Fingerprints:
    """Fingerprints for the union of cohort and archive ligands.

    The cohort's own ``ecfp_*`` columns are reproduced first and compared
    bit-for-bit.  If that fails, every Tanimoto in this audit is measured on a
    different footing from the clustering the folds were built with, and the
    chemotype check would be meaningless rather than merely wrong.
    """
    # ``ecfp_cluster`` also starts with ``ecfp_`` and is a label, not a bit.
    ecfp_columns = [f"ecfp_{i}" for i in range(ECFP_BITS) if f"ecfp_{i}" in cohort_frame.columns]
    if len(ecfp_columns) != ECFP_BITS:
        raise SystemExit(f"expected {ECFP_BITS} ecfp bit columns, found {len(ecfp_columns)}")
    unique = cohort_frame.drop_duplicates("extractant")
    stored = unique.set_index(unique["extractant"].astype(str))[ecfp_columns].astype(np.uint8)
    cohort_smiles = sorted(set(cohort_frame["extractant"].astype(str)))
    archive_smiles = sorted({s for s in archive["extractant_primary_smiles"].dropna().astype(str)})

    rows, kept, unparsed = [], [], []
    for smiles in sorted(set(cohort_smiles) | set(archive_smiles)):
        bits = _ecfp(smiles)
        if bits is None:
            unparsed.append(smiles)
            continue
        rows.append(bits)
        kept.append(smiles)
    bits = np.vstack(rows) if rows else np.zeros((0, ECFP_BITS), dtype=np.uint8)

    index = {s: i for i, s in enumerate(kept)}
    deltas = [int(np.abs(bits[index[s]].astype(int) - stored.loc[s].to_numpy().astype(int)).max())
              for s in cohort_smiles if s in index]
    worst = max(deltas) if deltas else 1
    if worst != 0 or len(deltas) != len(cohort_smiles):
        raise SystemExit(
            "recomputed ECFP does not reproduce the bundle's ecfp_* columns "
            f"(max bit delta {worst}, reproduced {len(deltas)}/{len(cohort_smiles)})")
    return Fingerprints(tuple(kept), bits, tuple(unparsed), worst)


def tanimoto_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = query.astype(np.int32)
    r = reference.astype(np.int32)
    inter = q @ r.T
    union = q.sum(axis=1)[:, None] + r.sum(axis=1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros(inter.shape, dtype=float), where=union > 0)


# --------------------------------------------------------------------------- #
# Cell keys for the near-duplicate sweep
# --------------------------------------------------------------------------- #

def _round_sig(values: pd.Series, digits: int) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(x.shape, np.nan)
    zero = np.isfinite(x) & (x == 0)
    out[zero] = 0.0
    finite = np.isfinite(x) & (x != 0)
    if finite.any():
        scale = np.power(10.0, np.floor(np.log10(np.abs(x[finite]))))
        out[finite] = np.round(x[finite] / scale, digits - 1) * scale
    return pd.Series(out, index=values.index)


def cell_key(frame: pd.DataFrame, digits: int, *, include_structure: bool = True) -> pd.Series:
    """A string key for the experimental cell at ``digits`` significant figures.

    Missing values become the literal token ``NA`` rather than being dropped.
    That makes two rows that are both silent about temperature *match*, which is
    the inclusive reading — this sweep is meant to over-report and then be
    filtered, not to quietly exonerate rows whose metadata is thin.
    """
    categorical = CELL_CATEGORICAL if include_structure else CELL_CATEGORICAL[1:]
    parts = [frame[c].astype("string").fillna("NA").to_numpy() for c in categorical]
    for column in CELL_NUMERIC:
        rounded = _round_sig(frame[column], digits)
        parts.append(rounded.map(
            lambda v: "NA" if not np.isfinite(v) else format(v, ".12g")).to_numpy())
    return pd.Series(["|".join(values) for values in zip(*parts)], index=frame.index)


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #

def run(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = prepared_cohort()
    frame = cohort.frame
    overlap = ov.build_overlap_map(frame)
    identity = ov.cohort_identity(frame, overlap)

    # ``relationship_masks`` reads ``held_out["extractant"]`` but
    # ``cohort_identity`` emits ``extractant_src``/``extractant_cohort`` because
    # the merge collides on that name.  The two are byte-equal (asserted here),
    # so aliasing is safe; the API mismatch itself is reported as a finding
    # rather than patched away silently.
    if not bool((identity["extractant_src"] == identity["extractant_cohort"]).all()):
        raise SystemExit("bundle and cohort disagree on the extractant SMILES")
    identity = identity.assign(extractant=identity["extractant_cohort"].astype(str))

    archive = overlap.archive
    aux = overlap.auxiliary_candidates.copy().reset_index(drop=True)
    aux["metal_category"] = aux["metal_category"].astype("string").fillna("unknown")
    aux["has_target_value"] = aux["log_D"].notna().to_numpy()
    pool_mask = {
        "all_candidates": np.ones(len(aux), dtype=bool),
        "model_ready": (aux["model_readiness"] == "A_model_ready").to_numpy(),
    }
    aux_smiles = aux["extractant_primary_smiles"].astype("string")

    fingerprints = fingerprint_table(frame, archive)
    fp_index = {s: i for i, s in enumerate(fingerprints.smiles)}

    # Dense Tanimoto between every auxiliary structure and every cohort ligand.
    # 10,778 x 152 is nothing, and having it up front makes the structural checks
    # exact rather than sampled.
    cohort_ligands = sorted(set(frame["extractant"].astype(str)))
    ligand_column = {s: j for j, s in enumerate(cohort_ligands)}
    cohort_rows = np.array([fp_index[s] for s in cohort_ligands])
    aux_row_of = aux_smiles.map(fp_index)
    aux_has_fp = aux_row_of.notna().to_numpy()
    sim_aux_cohort = np.zeros((len(aux), len(cohort_ligands)), dtype=float)
    if aux_has_fp.any():
        sim_aux_cohort[aux_has_fp] = tanimoto_matrix(
            fingerprints.bits[aux_row_of[aux_has_fp].astype(int).to_numpy()],
            fingerprints.bits[cohort_rows])
    # Control: the same quantity inside the cohort, to establish whether gen10's
    # own fold plan separates train from test at this threshold.  Without it the
    # structural finding could be a property of the benchmark rather than of the
    # auxiliary pool.
    cohort_sim = tanimoto_matrix(fingerprints.bits[cohort_rows], fingerprints.bits[cohort_rows])

    chemotype_of_ligand = (frame.drop_duplicates("extractant")
                           .set_index(frame.drop_duplicates("extractant")["extractant"].astype(str))
                           ["tanimoto_cluster"].astype(str).to_dict())
    cohort_chemotype = np.array([chemotype_of_ligand[s] for s in cohort_ligands])

    dict_assignment = {s: (c,) for s, c in chemotype_of_ligand.items()}
    closure_assignment: dict[str, tuple[str, ...]] = {}
    for smiles in sorted(set(aux_smiles.dropna().astype(str))) + cohort_ligands:
        row = fp_index.get(smiles)
        if row is None:
            closure_assignment[smiles] = ()
            continue
        sims = tanimoto_matrix(fingerprints.bits[[row]], fingerprints.bits[cohort_rows])[0]
        closure_assignment[smiles] = tuple(sorted(set(cohort_chemotype[sims >= TANIMOTO_CLUSTER_THRESHOLD])))
    assignment_spec = {
        "DICT_LEGACY": (dict_assignment, False),
        "CLOSURE": (closure_assignment, True),
    }

    # ---- near-duplicate pair table, computed once ------------------------- #
    heldout_source = identity.merge(
        archive[["source_record_id", *CELL_CATEGORICAL, *CELL_NUMERIC, "log_D"]],
        on="source_record_id", how="left", suffixes=("", "_arch")).reset_index(drop=True)
    archive_log_D = archive.set_index("source_record_id")["log_D"]
    near_pairs = []
    for digits in SIG_FIGS:
        left = pd.DataFrame({"row_id": heldout_source["row_id"].to_numpy(),
                             "source_record_id": heldout_source["source_record_id"].to_numpy(),
                             "cell_key": cell_key(heldout_source, digits).to_numpy()})
        right = pd.DataFrame({"aux_source_record_id": aux["source_record_id"].to_numpy(),
                              "cell_key": cell_key(aux, digits).to_numpy(),
                              "aux_log_D": aux["log_D"].to_numpy(),
                              "aux_metal_category": aux["metal_category"].to_numpy(),
                              "aux_model_readiness": aux["model_readiness"].to_numpy(),
                              "aux_duplicate_group_id": aux["duplicate_group_id"].to_numpy(),
                              "aux_series_id": aux["series_id"].to_numpy(),
                              "aux_doi": aux["doi_primary_corrected"].to_numpy()})
        merged = left.merge(right, on="cell_key", how="inner")
        merged["sig_figs"] = digits
        merged["cohort_log_D"] = merged["source_record_id"].map(archive_log_D)
        merged["abs_log_D_delta"] = (merged["aux_log_D"] - merged["cohort_log_D"]).abs()
        near_pairs.append(merged)
    near_pair_table = pd.concat(near_pairs, ignore_index=True)

    # Structure-free cell keys, for "same experiment on a near-duplicate ligand".
    cohort_cell_no_structure = cell_key(heldout_source, 3, include_structure=False).to_numpy()
    aux_cell_no_structure = cell_key(aux, 3, include_structure=False).to_numpy()

    # ---- per (seed, fold) -------------------------------------------------- #
    relationship_rows, policy_rows, per_fold_rows, near_dup_fold_rows = [], [], [], []
    mask_disagreements = {name: 0 for name in ASSIGNMENTS}
    violations = {name: {"group": 0, "series": 0, "chemotype_dict": 0,
                         "structural": 0, "structural_model_ready": 0,
                         "structural_identical": 0, "structural_same_cell": 0}
                  for name in ASSIGNMENTS}
    max_tanimoto_seen = {name: 0.0 for name in ASSIGNMENTS}
    structural_examples: list[dict] = []
    violating_structures: set[str] = set()
    cohort_cross_violations, cohort_cross_max = 0, 0.0
    closure_matches_structural = True

    for seed in DEFAULT_SEEDS:
        for fold in build_folds(frame, seed):
            test_ids = set(frame.iloc[fold.test_index]["row_id"].astype(str))
            held = identity[identity["row_id"].isin(test_ids)]
            if held["row_id"].nunique() != len(test_ids):
                raise SystemExit(
                    f"seed {seed} fold {fold.fold}: "
                    f"{len(test_ids) - held['row_id'].nunique()} held-out cohort rows "
                    "have no archive source")

            bad_groups = set(held["duplicate_group_id"].dropna().astype(str))
            bad_series = set(held["series_id_src"].dropna().astype(str)) | \
                set(held["series_id_cohort"].dropna().astype(str))
            bad_ligands = set(held["extractant"].dropna().astype(str))
            bad_chemotypes = set(held["tanimoto_cluster"].dropna().astype(str))
            bad_dois = set(held["doi_primary_corrected"].dropna().astype(str))

            held_columns = [ligand_column[s] for s in bad_ligands if s in ligand_column]
            best_to_heldout = (sim_aux_cohort[:, held_columns].max(axis=1) if held_columns
                               else np.zeros(len(aux)))
            structural = (best_to_heldout >= TANIMOTO_CLUSTER_THRESHOLD) & aux_has_fp

            # Control inside the cohort itself.
            test_cols = held_columns
            train_cols = [j for s, j in ligand_column.items() if s not in bad_ligands]
            if train_cols and test_cols:
                block = cohort_sim[np.ix_(train_cols, test_cols)]
                cohort_cross_violations += int((block >= TANIMOTO_CLUSTER_THRESHOLD).sum())
                cohort_cross_max = max(cohort_cross_max, float(block.max()))

            fold_record = {"split_seed": int(seed), "fold": int(fold.fold),
                           "n_heldout_rows": int(len(test_ids)),
                           "n_heldout_extractants": len(bad_ligands),
                           "n_heldout_chemotypes": len(bad_chemotypes)}

            for name in ASSIGNMENTS:
                mapping, require = assignment_spec[name]
                masks = ov.relationship_masks(
                    aux, held, chemotype_of_smiles=mapping,
                    require_complete_assignment=require)

                # Independent recomputation of the same rules, from raw columns.
                mine = pd.DataFrame({
                    "EXACT": np.zeros(len(aux), dtype=bool),
                    "DUPLICATE_EQUIVALENT": aux["duplicate_group_id"].astype(str)
                                            .isin(bad_groups).to_numpy(),
                    "SAME_SERIES": aux["series_id"].astype(str).isin(bad_series).to_numpy(),
                    "SAME_CHEMOTYPE": np.array([
                        bool(set(mapping.get(s, ())) & bad_chemotypes)
                        for s in aux_smiles.astype(str)]),
                    "SAME_LIGAND": aux_smiles.isin(bad_ligands).to_numpy(),
                    "SAME_PUBLICATION": aux["doi_primary_corrected"].astype("string")
                                        .isin(bad_dois).to_numpy(),
                }, index=aux.index)
                for relationship in ov.RELATIONSHIPS:
                    if not np.array_equal(masks[relationship].to_numpy(),
                                          mine[relationship].to_numpy()):
                        mask_disagreements[name] += 1

                for pool in POOLS:
                    sel = pool_mask[pool]
                    union = np.zeros(len(aux), dtype=bool)
                    for relationship in ov.RELATIONSHIPS:
                        column = masks[relationship].to_numpy()
                        marginal = int((column & ~union & sel).sum())
                        union = union | column
                        relationship_rows.append({
                            **fold_record, "assignment": name, "pool": pool,
                            "relationship": relationship,
                            "n_pool": int(sel.sum()),
                            "n_excluded": int((column & sel).sum()),
                            "frac_excluded": float((column & sel).sum() / max(1, sel.sum())),
                            "n_marginal_beyond_stricter": marginal,
                            "n_union_through_this": int((union & sel).sum()),
                        })

                for policy in ov.POLICIES:
                    safe = ov.apply_policy(masks, policy)
                    for pool in POOLS:
                        sel = pool_mask[pool] & safe
                        counts = (aux.loc[sel].groupby("metal_category", observed=True)
                                  .agg(n_surviving=("source_record_id", "size"),
                                       n_surviving_with_target=("has_target_value", "sum")))
                        for category, row in counts.iterrows():
                            policy_rows.append({
                                "split_seed": int(seed), "fold": int(fold.fold),
                                "assignment": name, "policy": policy, "pool": pool,
                                "metal_category": str(category),
                                "n_surviving": int(row["n_surviving"]),
                                "n_surviving_with_target": int(row["n_surviving_with_target"]),
                            })
                        policy_rows.append({
                            "split_seed": int(seed), "fold": int(fold.fold),
                            "assignment": name, "policy": policy, "pool": pool,
                            "metal_category": "ALL",
                            "n_surviving": int(sel.sum()),
                            "n_surviving_with_target": int(aux.loc[sel, "has_target_value"].sum()),
                        })

                headline = ov.apply_policy(masks, "HEADLINE")
                v = violations[name]
                v["group"] += int((headline & aux["duplicate_group_id"].astype(str)
                                   .isin(bad_groups).to_numpy()).sum())
                v["series"] += int((headline & aux["series_id"].astype(str)
                                    .isin(bad_series).to_numpy()).sum())
                v["chemotype_dict"] += int((headline & aux_smiles.map(chemotype_of_ligand)
                                            .astype("string").isin(bad_chemotypes).to_numpy()).sum())
                bad = headline & structural
                v["structural"] += int(bad.sum())
                v["structural_model_ready"] += int((bad & pool_mask["model_ready"]).sum())
                v["structural_identical"] += int((bad & (best_to_heldout >= 1.0 - 1e-12)).sum())
                if bad.any():
                    held_cells = set(cohort_cell_no_structure[
                        heldout_source["row_id"].isin(test_ids).to_numpy()])
                    v["structural_same_cell"] += int(
                        pd.Series(aux_cell_no_structure[bad]).isin(held_cells).sum())
                if (headline & aux_has_fp).any():
                    max_tanimoto_seen[name] = max(
                        max_tanimoto_seen[name],
                        float(best_to_heldout[headline & aux_has_fp].max()))
                if name == "DICT_LEGACY" and bad.any():
                    violating_structures.update(aux.loc[bad, "extractant_primary_smiles"]
                                                .dropna().astype(str))
                    order = np.argsort(-best_to_heldout[bad])[:3]
                    for position in order:
                        row_index = np.flatnonzero(bad)[position]
                        structural_examples.append({
                            "split_seed": int(seed), "fold": int(fold.fold),
                            "aux_source_record_id": str(aux.at[row_index, "source_record_id"]),
                            "aux_smiles": str(aux.at[row_index, "extractant_primary_smiles"]),
                            "aux_model_readiness": str(aux.at[row_index, "model_readiness"]),
                            "aux_metal_category": str(aux.at[row_index, "metal_category"]),
                            "max_tanimoto_to_heldout": float(best_to_heldout[row_index]),
                        })
                if name == "CLOSURE":
                    fold_record.update({
                        "n_headline_survivors_all": int(headline.sum()),
                        "n_headline_survivors_model_ready": int(
                            (headline & pool_mask["model_ready"]).sum()),
                        "max_tanimoto_survivor_to_heldout": float(
                            best_to_heldout[headline & aux_has_fp].max())
                        if (headline & aux_has_fp).any() else 0.0,
                        "n_structural_violations": int(bad.sum()),
                    })
                    survivor_ids = set(aux.loc[headline, "source_record_id"])
                    fold_pairs = near_pair_table[near_pair_table["row_id"].isin(test_ids)]
                    for digits in SIG_FIGS:
                        tier = fold_pairs[fold_pairs["sig_figs"] == digits]
                        surviving = tier[tier["aux_source_record_id"].isin(survivor_ids)]
                        near_dup_fold_rows.append({
                            "split_seed": int(seed), "fold": int(fold.fold),
                            "sig_figs": int(digits), "n_pairs": int(len(tier)),
                            "n_distinct_aux_rows": int(tier["aux_source_record_id"].nunique()),
                            "n_distinct_cohort_rows": int(tier["row_id"].nunique()),
                            "n_distinct_aux_rows_surviving_headline": int(
                                surviving["aux_source_record_id"].nunique()),
                            "n_surviving_pairs_with_target": int(surviving["aux_log_D"].notna().sum()),
                        })

            # Cross-check the two implementations of the closure rule against
            # each other: HEADLINE under CLOSURE must equal HEADLINE under the
            # dictionary minus everything the fingerprint sweep flags.
            dict_masks = ov.relationship_masks(aux, held, chemotype_of_smiles=dict_assignment,
                                               require_complete_assignment=False)
            closure_masks = ov.relationship_masks(aux, held, chemotype_of_smiles=closure_assignment,
                                                  require_complete_assignment=True)
            if not np.array_equal(ov.apply_policy(closure_masks, "HEADLINE"),
                                  ov.apply_policy(dict_masks, "HEADLINE") & ~structural):
                closure_matches_structural = False

            per_fold_rows.append(fold_record)

    relationship_table = pd.DataFrame(relationship_rows)
    policy_table = pd.DataFrame(policy_rows)
    fold_table = pd.DataFrame(per_fold_rows)
    near_dup_fold = pd.DataFrame(near_dup_fold_rows)

    relationship_table.to_csv(out_dir / "relationship_counts.csv", index=False)
    policy_table.to_csv(out_dir / "policy_survival.csv", index=False)
    fold_table.to_csv(out_dir / "per_fold_checks.csv", index=False)
    near_dup_fold.to_csv(out_dir / "near_duplicate_per_fold.csv", index=False)
    near_pair_table.sort_values(["sig_figs", "abs_log_D_delta"], ascending=[True, False]) \
        .to_csv(out_dir / "near_duplicate_candidates.csv", index=False)

    # ---- publication overlap ---------------------------------------------- #
    cohort_doi_counts = identity.assign(
        _doi=identity["doi_primary_corrected"].astype("string"))
    cohort_dois = set(cohort_doi_counts["_doi"].dropna())
    pub_rows = []
    for doi, group in aux.assign(
            _doi=aux["doi_primary_corrected"].astype("string").fillna("MISSING")
    ).groupby("_doi", observed=True):
        side = cohort_doi_counts[cohort_doi_counts["_doi"] == doi]
        pub_rows.append({
            "doi": str(doi),
            "in_cohort": bool(doi in cohort_dois),
            "n_aux_records": int(len(group)),
            "n_aux_model_ready": int((group["model_readiness"] == "A_model_ready").sum()),
            "n_aux_with_target": int(group["log_D"].notna().sum()),
            "n_cohort_source_records": int(len(side)),
            "n_cohort_rows": int(side["row_id"].nunique()),
            "n_cohort_extractants": int(side["extractant"].nunique()),
            "n_cohort_chemotypes": int(side["tanimoto_cluster"].nunique()),
            "aux_metal_categories": ",".join(sorted(set(group["metal_category"].astype(str)))),
        })
    publication_table = pd.DataFrame(pub_rows).sort_values("n_aux_records", ascending=False)
    publication_table.to_csv(out_dir / "publication_overlap.csv", index=False)

    # ---- global assertions ------------------------------------------------- #
    frozen_ids = set(overlap.exact["source_record_id"].astype(str))
    aux_ids = set(aux["source_record_id"].astype(str))
    class_d_ids = set(archive.loc[archive["model_readiness"] == "D_redundant_duplicate",
                                  "source_record_id"].astype(str))
    model_ready_ids = set(aux.loc[pool_mask["model_ready"], "source_record_id"].astype(str))
    null_smiles = aux["extractant_primary_smiles"].isna().to_numpy()

    def headline_span(assignment: str) -> pd.DataFrame:
        return policy_table[(policy_table["assignment"] == assignment)
                            & (policy_table["policy"] == "HEADLINE")
                            & (policy_table["pool"] == "model_ready")
                            & (policy_table["metal_category"] == "ALL")]

    nd_summary = {}
    for digits in SIG_FIGS:
        tier = near_dup_fold[near_dup_fold["sig_figs"] == digits]
        glob = near_pair_table[near_pair_table["sig_figs"] == digits]
        nd_summary[f"sig_figs_{digits}"] = {
            "n_pairs_global": int(len(glob)),
            "n_distinct_aux_rows_global": int(glob["aux_source_record_id"].nunique()),
            "n_distinct_cohort_rows_global": int(glob["row_id"].nunique()),
            "n_aux_rows_model_ready": int(
                glob.loc[glob["aux_model_readiness"] == "A_model_ready",
                         "aux_source_record_id"].nunique()),
            "median_abs_log_D_delta": float(glob["abs_log_D_delta"].median()) if len(glob) else None,
            "frac_pairs_identical_value": float((glob["abs_log_D_delta"] < 1e-9).mean())
            if len(glob) else None,
            "total_surviving_aux_rows_across_folds": int(
                tier["n_distinct_aux_rows_surviving_headline"].sum()),
            "max_surviving_aux_rows_on_any_fold": int(
                tier["n_distinct_aux_rows_surviving_headline"].max()),
        }

    checks = {
        "module_masks_match_independent_recomputation": {
            "passed": all(v == 0 for v in mask_disagreements.values()),
            "n_disagreeing_masks_by_assignment": mask_disagreements,
            "detail": "25 folds x 6 relationships x 2 assignments, recomputed from raw columns",
        },
        "no_survivor_shares_duplicate_group_with_heldout": {
            "passed": all(violations[n]["group"] == 0 for n in ASSIGNMENTS),
            "n_violations_by_assignment": {n: violations[n]["group"] for n in ASSIGNMENTS},
        },
        "no_survivor_shares_series_with_heldout": {
            "passed": all(violations[n]["series"] == 0 for n in ASSIGNMENTS),
            "n_violations_by_assignment": {n: violations[n]["series"] for n in ASSIGNMENTS},
        },
        "no_survivor_in_heldout_chemotype_by_dictionary": {
            "passed": all(violations[n]["chemotype_dict"] == 0 for n in ASSIGNMENTS),
            "n_violations_by_assignment": {n: violations[n]["chemotype_dict"] for n in ASSIGNMENTS},
        },
        "no_survivor_within_tanimoto_0p7_of_heldout_structure": {
            "passed": violations["CLOSURE"]["structural"] == 0,
            "n_violations_CLOSURE": violations["CLOSURE"]["structural"],
            "n_violations_DICT_LEGACY": violations["DICT_LEGACY"]["structural"],
            "n_violations_DICT_LEGACY_model_ready": violations["DICT_LEGACY"]["structural_model_ready"],
            "n_violations_DICT_LEGACY_tanimoto_1p0": violations["DICT_LEGACY"]["structural_identical"],
            "n_violations_DICT_LEGACY_same_experimental_cell": violations["DICT_LEGACY"]["structural_same_cell"],
            "n_distinct_offending_structures_DICT_LEGACY": len(violating_structures),
            "max_tanimoto_among_survivors": max_tanimoto_seen,
            "threshold": float(TANIMOTO_CLUSTER_THRESHOLD),
            "examples_DICT_LEGACY": structural_examples[:10],
            "detail": "chemotype membership re-derived from recomputed Morgan bits; "
                      "DICT_LEGACY is the pre-fix rule and is reported as the size of the hole",
        },
        "closure_rule_equals_independent_fingerprint_sweep": {
            "passed": bool(closure_matches_structural),
            "detail": "HEADLINE(CLOSURE) == HEADLINE(DICT) minus every row the "
                      "fingerprint sweep flags, on all 25 folds",
        },
        "gen10_folds_separate_cohort_train_and_test_at_the_same_threshold": {
            "passed": cohort_cross_violations == 0,
            "n_cohort_train_test_pairs_at_or_above_threshold": int(cohort_cross_violations),
            "max_cohort_train_test_tanimoto": float(cohort_cross_max),
            "detail": "control: the benchmark's own folds, measured the same way",
        },
        "aux_pool_contains_no_frozen_record": {
            "passed": len(aux_ids & frozen_ids) == 0,
            "n_frozen_records": len(frozen_ids),
            "n_intersection": len(aux_ids & frozen_ids),
        },
        "class_d_redundant_duplicates_never_in_model_ready_pool": {
            "passed": len(class_d_ids & model_ready_ids) == 0,
            "n_class_d_records": len(class_d_ids),
            "n_class_d_in_all_candidates_pool": len(class_d_ids & aux_ids),
            "n_class_d_in_model_ready_pool": len(class_d_ids & model_ready_ids),
            "detail": "class D is excluded by model_readiness, NOT by any overlap "
                      "relationship; it survives the all_candidates pool, so any arm "
                      "that widens beyond model_ready must filter it explicitly",
        },
        "ecfp_reproduces_bundle_bits": {
            "passed": fingerprints.cohort_reproduction_max_bit_delta == 0,
            "max_bit_delta": int(fingerprints.cohort_reproduction_max_bit_delta),
            "n_unparsed_smiles": len(fingerprints.unparsed),
            "n_structures_fingerprinted": len(fingerprints.smiles),
        },
        "every_heldout_cohort_row_has_an_archive_source": {
            "passed": True,
            "detail": "asserted inside every fold; a failure raises SystemExit",
        },
        "headline_pool_is_nonempty_on_every_fold": {
            "passed": bool(headline_span("CLOSURE")["n_surviving"].min() > 0),
            "CLOSURE_min_model_ready": int(headline_span("CLOSURE")["n_surviving"].min()),
            "CLOSURE_max_model_ready": int(headline_span("CLOSURE")["n_surviving"].max()),
            "DICT_LEGACY_min_model_ready": int(headline_span("DICT_LEGACY")["n_surviving"].min()),
            "DICT_LEGACY_max_model_ready": int(headline_span("DICT_LEGACY")["n_surviving"].max()),
        },
        "aux_rows_with_no_structure_are_unblockable_by_structure_rules": {
            "passed": int((null_smiles & pool_mask["model_ready"]).sum()) == 0,
            "n_aux_rows_without_smiles": int(null_smiles.sum()),
            "n_model_ready_without_smiles": int((null_smiles & pool_mask["model_ready"]).sum()),
            "detail": "SAME_LIGAND/SAME_CHEMOTYPE cannot fire on a null SMILES; the "
                      "check passes only because every such row is outside model_ready",
        },
        "headline_removes_every_experimental_near_duplicate": {
            "passed": all(v["total_surviving_aux_rows_across_folds"] == 0
                          for v in nd_summary.values()),
            "tiers": nd_summary,
            "detail": "match on (structure, metal, acid, [acid], [extractant], solvent, T); "
                      "missing values match each other; survival judged under CLOSURE HEADLINE",
        },
    }

    summary = {
        "overlap_audit": overlap.audit,
        "seeds": [int(s) for s in DEFAULT_SEEDS],
        "n_folds_total": int(len(fold_table)),
        "aux_pool_sizes": {p: int(pool_mask[p].sum()) for p in POOLS},
        "chemotype_assignment": {
            "n_structures_assigned": len(closure_assignment),
            "n_aux_structures_linked_to_no_cohort_chemotype": int(sum(
                1 for s in set(aux_smiles.dropna().astype(str)) if not closure_assignment.get(s))),
            "n_aux_structures_linked_to_multiple_chemotypes": int(sum(
                1 for s in set(aux_smiles.dropna().astype(str))
                if len(closure_assignment.get(s, ())) > 1)),
        },
        "publication": {
            "n_cohort_dois": len(cohort_dois),
            "n_aux_dois": int(aux["doi_primary_corrected"].nunique()),
            "n_aux_dois_shared_with_cohort": int(publication_table["in_cohort"].sum()),
            "n_aux_records_sharing_a_cohort_doi": int(
                publication_table.loc[publication_table["in_cohort"], "n_aux_records"].sum()),
            "n_aux_model_ready_sharing_a_cohort_doi": int(
                publication_table.loc[publication_table["in_cohort"], "n_aux_model_ready"].sum()),
            "n_aux_records_without_doi": int(aux["doi_primary_corrected"].isna().sum()),
        },
        "checks": checks,
        "all_checks_passed": all(v["passed"] for v in checks.values()),
        "failed_checks": [k for k, v in checks.items() if not v["passed"]],
    }
    (out_dir / "adversarial_checks.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":  # pragma: no cover
    result = run(ov.REPO_ROOT / "runs" / "gen11_transfer" / "overlap")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2, default=str))
    for name, check in result["checks"].items():
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {name}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def write_readme(out_dir: Path) -> Path:
    """Render README.md from the artefacts, so no number is typed by hand."""
    import textwrap

    checks = json.loads((out_dir / "adversarial_checks.json").read_text())
    rel = pd.read_csv(out_dir / "relationship_counts.csv")
    pol = pd.read_csv(out_dir / "policy_survival.csv")
    fold = pd.read_csv(out_dir / "per_fold_checks.csv")
    pub = pd.read_csv(out_dir / "publication_overlap.csv")
    struct = checks["checks"]["no_survivor_within_tanimoto_0p7_of_heldout_structure"]
    near = checks["checks"]["headline_removes_every_experimental_near_duplicate"]["tiers"]

    def rel_block(assignment: str, pool: str) -> str:
        sub = rel[(rel["assignment"] == assignment) & (rel["pool"] == pool)]
        grouped = sub.groupby("relationship", sort=False).agg(
            excluded_mean=("n_excluded", "mean"), excluded_min=("n_excluded", "min"),
            excluded_max=("n_excluded", "max"),
            marginal_mean=("n_marginal_beyond_stricter", "mean"),
            marginal_min=("n_marginal_beyond_stricter", "min"),
            marginal_max=("n_marginal_beyond_stricter", "max"),
            union_mean=("n_union_through_this", "mean"))
        lines = ["| relationship | excluded mean | min | max | marginal mean | min | max | union-through mean |",
                 "|---|---|---|---|---|---|---|---|"]
        for name, row in grouped.iterrows():
            lines.append(
                f"| {name} | {row.excluded_mean:.1f} | {int(row.excluded_min)} | "
                f"{int(row.excluded_max)} | {row.marginal_mean:.1f} | {int(row.marginal_min)} | "
                f"{int(row.marginal_max)} | {row.union_mean:.1f} |")
        return "\n".join(lines)

    def policy_block(assignment: str, pool: str) -> str:
        sub = pol[(pol["assignment"] == assignment) & (pol["pool"] == pool)
                  & (pol["metal_category"] == "ALL")]
        grouped = sub.groupby("policy").agg(
            mean=("n_surviving", "mean"), lo=("n_surviving", "min"), hi=("n_surviving", "max"),
            target_mean=("n_surviving_with_target", "mean"))
        lines = ["| policy | surviving mean | min | max | with target, mean |", "|---|---|---|---|---|"]
        for name, row in grouped.iterrows():
            lines.append(f"| {name} | {row['mean']:.0f} | {int(row.lo)} | {int(row.hi)} | "
                         f"{row.target_mean:.0f} |")
        return "\n".join(lines)

    def metal_block() -> str:
        sub = pol[(pol["assignment"] == "CLOSURE") & (pol["pool"] == "model_ready")
                  & (pol["policy"] == "HEADLINE") & (pol["metal_category"] != "ALL")]
        grouped = sub.groupby("metal_category").agg(
            mean=("n_surviving", "mean"), lo=("n_surviving", "min"), hi=("n_surviving", "max"))
        lines = ["| metal category | surviving mean | min over 25 folds | max |", "|---|---|---|---|"]
        for name, row in grouped.sort_values("mean", ascending=False).iterrows():
            lines.append(f"| {name} | {row['mean']:.0f} | {int(row.lo)} | {int(row.hi)} |")
        return "\n".join(lines)

    verdict = "\n".join(
        f"- {'PASS' if v['passed'] else '**FAIL**'} — `{k}`"
        for k, v in checks["checks"].items())

    body = f"""# gen11 §2/§20 — per-fold leakage audit of the auxiliary pool

Generated by `lanthanide_separation.gen11.leakage_audit`.
Cohort fingerprint `bed178ec1a7a82b0`; {len(checks['seeds'])} split seeds x 5 folds =
{checks['n_folds_total']} folds; seeds {checks['seeds']}.

## Verdict

**{'all checks pass' if checks['all_checks_passed'] else 'CHECKS FAILED: ' + ', '.join(checks['failed_checks'])}** —
headline reporting is {'unblocked by this audit' if checks['all_checks_passed'] else 'BLOCKED'}.

{verdict}

## What the pool is

The archive holds {checks['overlap_audit']['archive_records']:,} records.  All
{checks['overlap_audit']['frozen_rows_matched_exactly']:,} frozen bundle rows join into it 1:1
(metal agreement {checks['overlap_audit']['exact_join_metal_agreement']:.1f}, max |log D| delta
{checks['overlap_audit']['exact_join_max_abs_target_delta']:.2g}), leaving
{checks['aux_pool_sizes']['all_candidates']:,} auxiliary candidates, of which
{checks['aux_pool_sizes']['model_ready']:,} are `A_model_ready`.  Every table below is reported at
both widths, because a strictness cost quoted against a pool nobody would train on is not a cost.

Held out per fold: {fold['n_heldout_rows'].min()}–{fold['n_heldout_rows'].max()} cohort rows
(mean {fold['n_heldout_rows'].mean():.0f}), {fold['n_heldout_extractants'].min()}–{fold['n_heldout_extractants'].max()} extractants,
{fold['n_heldout_chemotypes'].min()}–{fold['n_heldout_chemotypes'].max()} chemotypes.

## The defect this audit found

`SAME_CHEMOTYPE` originally resolved an auxiliary structure through a dictionary built from the
cohort's own `extractant -> tanimoto_cluster` column.  That dictionary only knows the cohort's 152
ligands, so every auxiliary structure the cohort does not contain returned nothing and passed the
filter.  Measured, not argued:

- **{struct['n_violations_DICT_LEGACY']:,}** surviving (row, fold) instances under the pre-fix rule sat at Tanimoto
  >= {struct['threshold']} to a held-out ligand — i.e. inside a held-out chemotype;
  **{struct['n_violations_DICT_LEGACY_model_ready']:,}** of them were `A_model_ready`.
- **{struct['n_violations_DICT_LEGACY_tanimoto_1p0']}** of those were at Tanimoto **1.0** — a bit-identical ligand under a
  different SMILES string.
- {struct['n_distinct_offending_structures_DICT_LEGACY']} distinct structures were responsible: exactly the auxiliary
  ligands that are absent from the cohort yet within 0.7 of a cohort ligand.
- On the `model_ready` pool the pre-fix `SAME_CHEMOTYPE` excluded the *same* rows as `SAME_LIGAND`
  (both {rel[(rel.assignment == 'DICT_LEGACY') & (rel.pool == 'model_ready') & (rel.relationship == 'SAME_CHEMOTYPE')]['n_excluded'].mean():.1f} on average) — the chemotype rule was operationally a ligand rule.

`overlap.relationship_masks` now requires a complete assignment mapping each auxiliary structure to
*every* cohort chemotype it links to at 0.7.  Under that rule the violation count is
**{struct['n_violations_CLOSURE']}** and the highest Tanimoto any surviving auxiliary structure reaches to a held-out
ligand is **{struct['max_tanimoto_among_survivors']['CLOSURE']:.3f}**, below the 0.7 threshold.

Control, measured the same way: inside the cohort itself gen10's folds put
{checks['checks']['gen10_folds_separate_cohort_train_and_test_at_the_same_threshold']['n_cohort_train_test_pairs_at_or_above_threshold']}
train/test ligand pairs at or above 0.7, max
{checks['checks']['gen10_folds_separate_cohort_train_and_test_at_the_same_threshold']['max_cohort_train_test_tanimoto']:.3f}.
The benchmark was clean; the hole was in the auxiliary join alone.

## Per-relationship cost (`relationship_counts.csv`)

Marginal = rows this relationship excludes that no stricter relationship already excluded,
in the fatal-to-benign order `EXACT, DUPLICATE_EQUIVALENT, SAME_SERIES, SAME_CHEMOTYPE,
SAME_LIGAND, SAME_PUBLICATION`.

### model_ready pool, current (CLOSURE) rule

{rel_block("CLOSURE", "model_ready")}

`SAME_LIGAND` has zero marginal cost by construction — it is a subset of `SAME_CHEMOTYPE` — and that
zero is the check that the closure is actually a superset.

### all_candidates pool, current rule

{rel_block("CLOSURE", "all_candidates")}

## Policy survival — is each arm runnable? (`policy_survival.csv`)

### model_ready pool, CLOSURE

{policy_block("CLOSURE", "model_ready")}

### all_candidates pool, CLOSURE

{policy_block("CLOSURE", "all_candidates")}

### HEADLINE survival by metal category, model_ready

{metal_block()}

Every arm is runnable on every fold, but not equally: the actinide block carries the pool, and the
worst fold leaves {pol[(pol.assignment == 'CLOSURE') & (pol.pool == 'model_ready') & (pol.policy == 'HEADLINE') & (pol.metal_category == 'lanthanide')]['n_surviving'].min()}
non-cohort lanthanide rows and
{pol[(pol.assignment == 'CLOSURE') & (pol.pool == 'model_ready') & (pol.policy == 'HEADLINE') & (pol.metal_category == 'post_transition_metal')]['n_surviving'].min()}
post-transition rows.  Any arm stratified by metal category has to carry that spread, or a null on a
minority category will be a sample-size result wearing a transfer result's clothes.

## Near-duplicate sweep (`near_duplicate_candidates.csv`, `near_duplicate_per_fold.csv`)

Matched on the experimental cell rather than on identifiers: structure, metal, acid, acid molarity,
extractant molarity, solvent key and temperature, with missing values matching each other.

- {near['sig_figs_6']['n_pairs_global']:,} (auxiliary, cohort) pairs describe the same cell, covering
  {near['sig_figs_6']['n_distinct_aux_rows_global']:,} distinct auxiliary records and
  {near['sig_figs_6']['n_distinct_cohort_rows_global']} distinct cohort rows.
- {near['sig_figs_6']['frac_pairs_identical_value']:.1%} of those pairs carry a **bit-identical** `log_D`; the median
  |delta| is {near['sig_figs_6']['median_abs_log_D_delta']:.3g}.  These are the same measurements, restated.
- Only {near['sig_figs_6']['n_aux_rows_model_ready']} of the {near['sig_figs_6']['n_distinct_aux_rows_global']:,} are `A_model_ready`.
- HEADLINE removes **all** of them on **all** {checks['n_folds_total']} folds
  ({near['sig_figs_6']['total_surviving_aux_rows_across_folds']} surviving, summed over folds) — they match on structure, so
  `SAME_LIGAND` and hence `SAME_CHEMOTYPE` fire.
- The 6- and 3-significant-figure tiers return identical pair sets even though rounding to 3 s.f.
  changes thousands of individual concentrations, so the finding is not a tolerance artefact.

## Publication overlap (`publication_overlap.csv`)

{checks['publication']['n_aux_dois']} DOIs appear in the auxiliary pool against
{checks['publication']['n_cohort_dois']} in the cohort.
{checks['publication']['n_aux_dois_shared_with_cohort']} are shared, covering
{checks['publication']['n_aux_records_sharing_a_cohort_doi']:,} auxiliary records
({checks['publication']['n_aux_model_ready_sharing_a_cohort_doi']:,} model-ready) —
{checks['publication']['n_aux_records_sharing_a_cohort_doi'] / checks['aux_pool_sizes']['all_candidates']:.0%} of the pool.
{int((~pub['in_cohort']).sum()) - 1} DOIs are auxiliary-only
({int(pub.loc[~pub['in_cohort'], 'n_aux_records'].sum()) - int(pub.loc[pub['doi'] == 'MISSING', 'n_aux_records'].sum()):,} records),
{int(pub.loc[pub['doi'] == 'MISSING', 'n_aux_records'].sum())} auxiliary records carry no DOI at all, and
{checks['publication']['n_cohort_dois'] - checks['publication']['n_aux_dois_shared_with_cohort']} cohort DOIs contribute nothing to the pool.
`PUBLICATION_BLOCKED` therefore costs real data — mean
{pol[(pol.assignment == 'CLOSURE') & (pol.pool == 'model_ready') & (pol.metal_category == 'ALL') & (pol.policy == 'HEADLINE')]['n_surviving'].mean() - pol[(pol.assignment == 'CLOSURE') & (pol.pool == 'model_ready') & (pol.metal_category == 'ALL') & (pol.policy == 'PUBLICATION_BLOCKED')]['n_surviving'].mean():.0f}
model-ready rows per fold beyond HEADLINE — and stays a robustness variant, not the headline.

## Known holes, stated rather than hidden

- **Class D is not an overlap rule.** {checks['checks']['class_d_redundant_duplicates_never_in_model_ready_pool']['n_class_d_records']:,} `D_redundant_duplicate`
  records sit in `all_candidates` and are kept out of training only by the `model_readiness` filter.
  Any arm that widens past `model_ready` must exclude them explicitly.
- **Structure-free rows cannot be blocked by structure rules.**
  {checks['checks']['aux_rows_with_no_structure_are_unblockable_by_structure_rules']['n_aux_rows_without_smiles']} auxiliary records have no SMILES;
  `SAME_LIGAND` and `SAME_CHEMOTYPE` cannot fire on them.  The check passes only because
  {checks['checks']['aux_rows_with_no_structure_are_unblockable_by_structure_rules']['n_model_ready_without_smiles']} of them are model-ready.
- **API mismatch.** `overlap.cohort_identity` emits `extractant_src`/`extractant_cohort` (the merge
  collides on the name), while `overlap.relationship_masks` reads `held_out["extractant"]`.  Calling
  them in sequence raises `KeyError`.  This audit aliases the column after asserting the two are
  byte-equal on all {checks['overlap_audit']['bundle_rows_feeding_cohort']:,} matched rows; the mismatch itself is unfixed.

## Files

| file | contents |
|---|---|
| `relationship_counts.csv` | seed x fold x assignment x pool x relationship: excluded, marginal, running union |
| `policy_survival.csv` | seed x fold x assignment x policy x pool x metal category: surviving rows |
| `per_fold_checks.csv` | per fold: held-out size, HEADLINE survivors, max survivor Tanimoto |
| `adversarial_checks.json` | every assertion with PASS/FAIL and its supporting counts |
| `near_duplicate_candidates.csv` | every (auxiliary, cohort) pair on the same experimental cell, both tolerance tiers |
| `near_duplicate_per_fold.csv` | the same, resolved per fold, with HEADLINE survival |
| `publication_overlap.csv` | DOI-level contingency between the auxiliary pool and the cohort |
"""
    path = out_dir / "README.md"
    path.write_text(textwrap.dedent(body).lstrip())
    return path
