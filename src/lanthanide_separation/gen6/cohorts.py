"""BASE91 vs EXPANDED152: change the training chemistry and nothing else.

A ``min_rows=10`` versus ``min_rows=3`` leaderboard would be uninterpretable,
because relaxing the eligibility rule changes the *test* population as well as
the training one — a lower MAE could simply mean easier test rows.  The design
here fixes the evaluation first and varies only what the model is allowed to
learn from:

* one shared cohort is built once at ``min_cells = 3`` (5,248 rows, 152
  extractants), so every feature column, every one-hot level, every cluster
  label and every replicate average is **identical** for all arms;
* folds hold out whole Tanimoto-0.7 super-clusters of that shared cohort, so a
  fold's test rows are a property of the fold, not of the arm;
* an arm is a **row mask over the training side only**.

That leaves exactly one difference between BASE and EXPANDED — whether the
sparse, chemically distant extractants are visible during training — which is
the causal question the generation exists to answer.

Two controls come with it, and neither is optional:

``EXPANDED_ROWMATCHED``
    All sparse rows plus randomly chosen dense rows up to BASE's row count.  If
    EXPANDED wins only because it has 7.5 % more rows, this arm wins too.

``EXPANDED_SHUFFLED``
    EXPANDED's rows, with the *added* sparse rows' targets permuted among
    themselves.  The chemistry is present, the information is not.  If EXPANDED
    wins only through regularisation or through the change in group weighting,
    this arm wins too.

The fold algorithm is deliberately byte-identical to gen5's
(``shuffle the unique group labels, deal them round-robin``): sklearn's
``GroupKFold`` is greedy-by-size and kept 82 % of rows in the same fold across
five "seeds", which is why gen5 replaced it, and reusing the replacement is what
lets a gen6 run reproduce a gen5 number exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..levels import LEVEL_TARGET_COLUMN

#: Eligibility of the gen5 cohort: the rule that kept 91 of 190 extractants.
BASE_MIN_CELLS = 10
#: Eligibility of the gen6 cohort: 152 extractants for +7.5 % rows.
EXPANDED_MIN_CELLS = 3

BASE_ARM = "BASE"
EXPANDED_ARM = "EXPANDED"
ROWMATCHED_ARM = "EXPANDED_ROWMATCHED"
SHUFFLED_ARM = "EXPANDED_SHUFFLED"
DEFAULT_ARMS: tuple[str, ...] = (BASE_ARM, EXPANDED_ARM, ROWMATCHED_ARM, SHUFFLED_ARM)

#: Acquisition policies for the diversity-versus-depth learning curve.  All are
#: label-free: they may look at chemistry and at how many rows exist, never at
#: ``log_D``.
ACQUISITION_POLICIES: tuple[str, ...] = ("depth", "diversity", "random", "maxmin")


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #

def cells_per_extractant(frame: pd.DataFrame) -> pd.Series:
    """Unique (condition, metal) cells per extractant — the unit the filter counts.

    On a cohort built with ``replicate_policy="mean"`` one row *is* one cell, but
    counting the cell columns explicitly keeps the function correct on a raw
    frame too, and keeps it honest about what "10 rows" meant in gen5.
    """
    required = {"extractant", "condition_id", "metal_symbol"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"frame lacks cell columns {sorted(missing)}")
    return (frame.drop_duplicates(["extractant", "condition_id", "metal_symbol"])
            .groupby("extractant").size().sort_index())


def eligible_extractants(frame: pd.DataFrame, *, min_cells: int) -> tuple[str, ...]:
    counts = cells_per_extractant(frame)
    return tuple(counts[counts >= int(min_cells)].index.astype(str))


def arm_membership(frame: pd.DataFrame, *, min_cells: int) -> np.ndarray:
    """Boolean row mask: rows whose extractant clears ``min_cells``."""
    keep = set(eligible_extractants(frame, min_cells=min_cells))
    return frame["extractant"].astype(str).isin(keep).to_numpy()


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #

def seeded_group_kfold(groups: Sequence, n_splits: int, seed: int):
    """Randomised grouped K-fold: shuffle unique group labels, deal round-robin.

    Identical to the gen5 implementation (``scripts/run_gen5_levels.py``), which
    cannot be imported from a package module.  Kept here as the canonical copy;
    :mod:`tests.test_gen6_cohorts` re-derives the reference algorithm and asserts
    equality, so the two cannot drift silently.

    The largest group can only sit in one fold, so that fold is larger.  That is
    accepted and disclosed rather than balanced away — balancing would break the
    group hold-out.
    """
    labels = np.asarray([str(g) for g in groups])
    names = np.unique(labels)
    rng = np.random.default_rng(seed)
    dealt = rng.permutation(names)
    fold_of = {g: i % n_splits for i, g in enumerate(dealt)}
    assignment = np.array([fold_of[g] for g in labels])
    for k in range(n_splits):
        test = np.flatnonzero(assignment == k)
        train = np.flatnonzero(assignment != k)
        if test.size == 0:
            continue
        yield train, test


@dataclass(frozen=True)
class DiversitySplit:
    """One outer fold: identical test rows, arm-specific training rows.

    ``train_index_by_arm`` holds positional indices into the shared cohort frame.
    ``target_permutation_by_arm`` is ``None`` for every honest arm; for the
    information null it is a permutation *of positions within that arm's training
    index*, applied to the target only.
    """

    fold: int
    seed: int
    test_index: np.ndarray
    train_index_by_arm: dict[str, np.ndarray]
    target_permutation_by_arm: dict[str, np.ndarray | None]
    held_out_groups: tuple[str, ...]
    audit: dict = field(default_factory=dict)

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(self.train_index_by_arm)

    def training_target(self, arm: str, target: np.ndarray) -> np.ndarray:
        """The target vector the arm trains on (permuted only for a null arm)."""
        index = self.train_index_by_arm[arm]
        permutation = self.target_permutation_by_arm.get(arm)
        y = np.asarray(target, dtype=float)[index]
        return y if permutation is None else y[permutation]


def _sparse_permutation(
    train_index: np.ndarray, dense_mask_of_rows: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Identity permutation except on the sparse rows, which swap among themselves."""
    permutation = np.arange(len(train_index))
    sparse_positions = np.flatnonzero(~dense_mask_of_rows[train_index])
    if sparse_positions.size > 1:
        permutation[sparse_positions] = sparse_positions[rng.permutation(sparse_positions.size)]
    return permutation


def diversity_splits(
    frame: pd.DataFrame,
    *,
    group_column: str = "tanimoto_cluster",
    n_splits: int = 5,
    seed: int = 104729,
    base_min_cells: int = BASE_MIN_CELLS,
    arms: Sequence[str] = DEFAULT_ARMS,
    rowmatch_seed_offset: int = 700_001,
    shuffle_seed_offset: int = 900_001,
) -> list[DiversitySplit]:
    """Build the fixed-test BASE/EXPANDED folds over one shared cohort.

    ``frame`` must be the shared cohort (built once at ``min_cells = 3``).  Every
    arm sees the same test rows; only the training mask differs.

    ``EXPANDED_ROWMATCHED`` keeps every sparse training row and then draws dense
    training rows without replacement until it matches BASE's training row count.
    If the sparse rows alone already exceed that count (they do not on this
    cohort, but a future cohort could), the arm keeps all sparse rows and no
    dense rows, and the audit records the overshoot rather than silently
    dropping sparse chemistry.
    """
    unknown = [a for a in arms if a not in DEFAULT_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; choose from {DEFAULT_ARMS}")
    if group_column not in frame.columns:
        raise KeyError(f"frame lacks the fold-grouping column {group_column!r}")

    dense_mask = arm_membership(frame, min_cells=base_min_cells)
    groups = frame[group_column].astype(str).to_numpy()
    extractants = frame["extractant"].astype(str).to_numpy()
    splits: list[DiversitySplit] = []

    for fold, (train_all, test_index) in enumerate(seeded_group_kfold(groups, n_splits, seed)):
        train_by_arm: dict[str, np.ndarray] = {}
        permutation_by_arm: dict[str, np.ndarray | None] = {}

        base_index = train_all[dense_mask[train_all]]
        expanded_index = train_all
        sparse_index = train_all[~dense_mask[train_all]]

        if BASE_ARM in arms:
            train_by_arm[BASE_ARM] = base_index
            permutation_by_arm[BASE_ARM] = None
        if EXPANDED_ARM in arms:
            train_by_arm[EXPANDED_ARM] = expanded_index
            permutation_by_arm[EXPANDED_ARM] = None
        overshoot = 0
        if ROWMATCHED_ARM in arms:
            rng = np.random.default_rng(seed + fold * 1013 + rowmatch_seed_offset)
            budget = len(base_index) - len(sparse_index)
            if budget < 0:
                overshoot = -budget
                chosen_dense = np.empty(0, dtype=int)
            else:
                chosen_dense = rng.choice(base_index, size=min(budget, len(base_index)), replace=False)
            matched = np.sort(np.concatenate([sparse_index, chosen_dense]))
            train_by_arm[ROWMATCHED_ARM] = matched
            permutation_by_arm[ROWMATCHED_ARM] = None
        if SHUFFLED_ARM in arms:
            rng = np.random.default_rng(seed + fold * 1013 + shuffle_seed_offset)
            train_by_arm[SHUFFLED_ARM] = expanded_index
            permutation_by_arm[SHUFFLED_ARM] = _sparse_permutation(expanded_index, dense_mask, rng)

        held_out = tuple(sorted(set(groups[test_index])))
        audit = {
            "n_test_rows": int(test_index.size),
            "n_test_extractants": int(pd.unique(extractants[test_index]).size),
            "n_train_rows_by_arm": {a: int(len(i)) for a, i in train_by_arm.items()},
            "n_train_extractants_by_arm": {a: int(pd.unique(extractants[i]).size)
                                           for a, i in train_by_arm.items()},
            "n_sparse_train_rows": int(sparse_index.size),
            "n_sparse_train_extractants": int(pd.unique(extractants[sparse_index]).size),
            "rowmatch_overshoot_rows": int(overshoot),
        }
        splits.append(DiversitySplit(
            fold=fold, seed=seed, test_index=test_index, train_index_by_arm=train_by_arm,
            target_permutation_by_arm=permutation_by_arm, held_out_groups=held_out, audit=audit))
    return splits


def assert_split_integrity(
    frame: pd.DataFrame,
    splits: Sequence[DiversitySplit],
    *,
    group_column: str = "tanimoto_cluster",
    identity_columns: Sequence[str] = ("extractant", "ecfp_cluster", "tanimoto_cluster"),
) -> dict:
    """Prove the two properties the experiment stands on, or raise.

    1. **Identical test rows** — the test row ids of a fold do not depend on the arm
       (trivially true by construction here, and checked anyway because the whole
       comparison is void if it ever stops being true).
    2. **No chemistry leakage** — no extractant, ECFP cluster or Tanimoto
       super-cluster appears both in a fold's test rows and in any arm's training
       rows.

    Returns an audit dict suitable for ``validation.json``.
    """
    row_ids = frame["row_id"].astype(str).to_numpy() if "row_id" in frame.columns else None
    report: dict = {"folds": [], "ok": True}
    covered: set[int] = set()
    for split in splits:
        test_ids = set(row_ids[split.test_index]) if row_ids is not None else set(split.test_index.tolist())
        entry: dict = {"fold": split.fold, "n_test_rows": int(split.test_index.size), "leaks": {}}
        for arm, index in split.train_index_by_arm.items():
            if set(index.tolist()) & set(split.test_index.tolist()):
                entry["leaks"][arm] = {"row_overlap": True}
                report["ok"] = False
            for column in identity_columns:
                if column not in frame.columns:
                    continue
                shared = (set(frame[column].astype(str).to_numpy()[index])
                          & set(frame[column].astype(str).to_numpy()[split.test_index]))
                if shared:
                    entry["leaks"].setdefault(arm, {})[column] = sorted(shared)[:5]
                    report["ok"] = False
        # BASE must be a subset of EXPANDED for the contrast to be about added chemistry
        if BASE_ARM in split.train_index_by_arm and EXPANDED_ARM in split.train_index_by_arm:
            base = set(split.train_index_by_arm[BASE_ARM].tolist())
            expanded = set(split.train_index_by_arm[EXPANDED_ARM].tolist())
            entry["base_subset_of_expanded"] = base <= expanded
            if not base <= expanded:
                report["ok"] = False
        entry["test_row_id_count"] = len(test_ids)
        covered |= set(split.test_index.tolist())
        report["folds"].append(entry)
    report["all_rows_tested_once"] = len(covered) == len(frame)
    report["n_rows_never_tested"] = int(len(frame) - len(covered))
    if not report["all_rows_tested_once"]:
        report["ok"] = False
    return report


# --------------------------------------------------------------------------- #
# Acquisition policies (Experiment B)
# --------------------------------------------------------------------------- #

def _maxmin_order(
    candidates: Sequence[str], similarity: np.ndarray, names: Sequence[str],
    *, rng: np.random.Generator,
) -> list[str]:
    """Greedy max-min Tanimoto: each step takes the ligand farthest from the chosen set."""
    position = {name: i for i, name in enumerate(names)}
    pool = [c for c in candidates if c in position]
    if not pool:
        return []
    order = [pool[int(rng.integers(len(pool)))]]
    remaining = [c for c in pool if c != order[0]]
    while remaining:
        chosen_idx = [position[c] for c in order]
        best, best_score = None, None
        for candidate in remaining:
            score = float(np.max(similarity[position[candidate], chosen_idx]))
            if best_score is None or score < best_score:
                best, best_score = candidate, score
        order.append(best)
        remaining.remove(best)
    return order


def ligand_acquisition_order(
    frame: pd.DataFrame,
    index: np.ndarray,
    *,
    policy: str,
    rng: np.random.Generator,
    similarity: np.ndarray | None = None,
    similarity_names: Sequence[str] | None = None,
    supercluster_column: str = "tanimoto_cluster",
) -> list[str]:
    """Order the candidate extractants of ``index`` under an acquisition policy.

    Label-free by construction: the policies read row counts and chemistry, never
    ``log_D``.  ``depth`` takes the best-measured ligands first (what the project
    has actually been doing); ``diversity`` deals one ligand from each
    super-cluster in turn, so previously absent chemotypes enter first; ``maxmin``
    walks the Tanimoto space greedily; ``random`` is the null.
    """
    if policy not in ACQUISITION_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {ACQUISITION_POLICIES}")
    sub = frame.iloc[index]
    counts = sub.groupby("extractant").size()
    candidates = list(counts.index.astype(str))
    if policy == "random":
        return [candidates[i] for i in rng.permutation(len(candidates))]
    if policy == "depth":
        shuffled = pd.Series(rng.random(len(counts)), index=counts.index)  # deterministic tie-break
        return list(pd.DataFrame({"n": counts, "t": shuffled})
                    .sort_values(["n", "t"], ascending=[False, True]).index.astype(str))
    if policy == "maxmin":
        if similarity is None or similarity_names is None:
            raise ValueError("maxmin needs a similarity matrix and its row names")
        return _maxmin_order(candidates, similarity, list(similarity_names), rng=rng)
    # diversity: round-robin over super-clusters, largest ligand first inside each
    if supercluster_column not in sub.columns:
        raise KeyError(f"diversity policy needs {supercluster_column!r}")
    by_cluster: dict[str, list[str]] = {}
    for cluster, block in sub.groupby(supercluster_column):
        members = block.groupby("extractant").size().sort_values(ascending=False)
        by_cluster[str(cluster)] = list(members.index.astype(str))
    cluster_order = [str(c) for c in np.array(sorted(by_cluster))[rng.permutation(len(by_cluster))]]
    order: list[str] = []
    while any(by_cluster[c] for c in cluster_order):
        for cluster in cluster_order:
            if by_cluster[cluster]:
                order.append(by_cluster[cluster].pop(0))
    return order


def row_budget_subsample(
    frame: pd.DataFrame,
    index: np.ndarray,
    *,
    budget: int | None,
    policy: str,
    rng: np.random.Generator,
    similarity: np.ndarray | None = None,
    similarity_names: Sequence[str] | None = None,
    supercluster_column: str = "tanimoto_cluster",
) -> np.ndarray:
    """Take up to ``budget`` training rows from ``index`` under an acquisition policy.

    Ligands enter whole, in the policy's order, until the budget would be
    exceeded; the ligand that straddles the boundary contributes a random subset
    of its rows so that budgets match exactly across policies.  ``budget=None``
    means "all rows" (the right-hand end of the learning curve).
    """
    if budget is None or budget >= len(index):
        return np.sort(np.asarray(index, dtype=int))
    if budget <= 0:
        return np.empty(0, dtype=int)
    order = ligand_acquisition_order(
        frame, index, policy=policy, rng=rng, similarity=similarity,
        similarity_names=similarity_names, supercluster_column=supercluster_column)
    extractants = frame["extractant"].astype(str).to_numpy()
    rows_by_extractant: dict[str, np.ndarray] = {}
    for name in order:
        rows_by_extractant[name] = index[extractants[index] == name]
    taken: list[np.ndarray] = []
    total = 0
    for name in order:
        rows = rows_by_extractant[name]
        if total + len(rows) <= budget:
            taken.append(rows)
            total += len(rows)
        else:
            remaining = budget - total
            if remaining > 0:
                taken.append(rng.choice(rows, size=remaining, replace=False))
                total = budget
            break
    if not taken:
        return np.empty(0, dtype=int)
    return np.sort(np.concatenate(taken))


def acquisition_audit(
    frame: pd.DataFrame, index: np.ndarray, *, supercluster_column: str = "tanimoto_cluster",
) -> dict:
    """What chemistry a training subset actually covers — printed beside every curve point."""
    sub = frame.iloc[index]
    return {
        "n_rows": int(len(sub)),
        "n_extractants": int(sub["extractant"].nunique()),
        "n_ecfp_clusters": int(sub["ecfp_cluster"].nunique()) if "ecfp_cluster" in sub.columns else None,
        "n_superclusters": int(sub[supercluster_column].nunique())
        if supercluster_column in sub.columns else None,
        "largest_extractant_share": float(sub["extractant"].value_counts(normalize=True).iloc[0])
        if len(sub) else float("nan"),
        "target_sd": float(sub[LEVEL_TARGET_COLUMN].std()) if LEVEL_TARGET_COLUMN in sub.columns else None,
    }


def cohort_comparison(
    frame: pd.DataFrame,
    *,
    base_min_cells: int = BASE_MIN_CELLS,
    supercluster_column: str = "tanimoto_cluster",
) -> dict:
    """What EXPANDED adds over BASE: extractants, clusters, super-clusters, rows.

    This is the "what did the eligibility rule cost us" table, computed on the
    shared cohort so both sides use identical labels.
    """
    dense = arm_membership(frame, min_cells=base_min_cells)
    base, sparse = frame[dense], frame[~dense]

    def describe(block: pd.DataFrame) -> dict:
        return {
            "n_rows": int(len(block)),
            "n_extractants": int(block["extractant"].nunique()),
            "n_ecfp_clusters": int(block["ecfp_cluster"].nunique()),
            "n_superclusters": int(block[supercluster_column].nunique()),
        }

    base_clusters = set(base["ecfp_cluster"].astype(str))
    base_super = set(base[supercluster_column].astype(str))
    sparse_clusters = set(sparse["ecfp_cluster"].astype(str))
    sparse_super = set(sparse[supercluster_column].astype(str))
    return {
        "base": describe(base),
        "added_by_expansion": describe(sparse),
        "expanded": describe(frame),
        "new_ecfp_clusters": len(sparse_clusters - base_clusters),
        "new_superclusters": len(sparse_super - base_super),
        "rows_in_new_superclusters": int(
            sparse[sparse[supercluster_column].astype(str).isin(sparse_super - base_super)].shape[0]),
        "row_cost_fraction": float(len(sparse) / len(frame)) if len(frame) else float("nan"),
    }
