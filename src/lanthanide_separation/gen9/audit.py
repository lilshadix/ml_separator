"""Audit artefacts — the checks that turn "the script completed" into evidence.

The follow-up brief's premise is that a gen9 run should be assumed wrong until its
data flow has been shown correct, and it names the specific failure modes this
project has already paid for: a salted hash that silently un-paired two runs, a
row with a missing abscissa emitted into a curve, dropped NaN predictions that made
an arm look better because its hard rows vanished, and comparisons labelled paired
that were joined on different evaluation sets.

None of those announce themselves.  Every one of them produces a completed script
and a plausible table.  So each gets a function here that writes a *file* — a
number a reader can check — rather than an assertion that passes silently.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Train / test boundaries
# --------------------------------------------------------------------------- #

def fold_audit(frame: pd.DataFrame, folds: Sequence, *,
               ligand_column: str = "extractant",
               block_column: str = "tanimoto_cluster") -> pd.DataFrame:
    """One row per fold: sizes and every forbidden overlap, which must all be zero."""
    ligands = frame[ligand_column].astype(str).to_numpy()
    blocks = frame[block_column].astype(str).to_numpy()
    clusters = (frame["ecfp_cluster"].astype(str).to_numpy()
                if "ecfp_cluster" in frame.columns else blocks)
    records = []
    for fold in folds:
        train, test = fold.train_index, fold.test_index
        records.append({
            "split_seed": int(fold.seed), "fold": int(fold.fold),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "train_ligands": int(len(set(ligands[train]))),
            "test_ligands": int(len(set(ligands[test]))),
            "train_chemotypes": int(len(set(blocks[train]))),
            "test_chemotypes": int(len(set(blocks[test]))),
            "overlap_rows": int(len(set(train.tolist()) & set(test.tolist()))),
            "overlap_ligands": int(len(set(ligands[train]) & set(ligands[test]))),
            "overlap_chemotypes": int(len(set(blocks[train]) & set(blocks[test]))),
            "overlap_ecfp_clusters": int(len(set(clusters[train]) & set(clusters[test]))),
        })
    return pd.DataFrame.from_records(records)


def assert_fold_audit_clean(audit: pd.DataFrame) -> None:
    forbidden = ["overlap_rows", "overlap_ligands", "overlap_chemotypes", "overlap_ecfp_clusters"]
    bad = audit[(audit[forbidden] > 0).any(axis=1)]
    if len(bad):
        raise AssertionError(f"fold boundary violated:\n{bad[['split_seed', 'fold'] + forbidden]}")


def curve_pair_audit(membership: pd.DataFrame, folds: Sequence,
                     frame: pd.DataFrame) -> pd.DataFrame:
    """Could any curve straddle a fold boundary?

    Under a chemotype hold-out a curve belongs to one series which belongs to one
    ligand, so the answer should be no — but "should" is what an audit is for, and
    an inner validation split *can* cut a ligand in half, which is where a straddle
    would first appear.
    """
    row_ids = frame["row_id"].astype(str).to_numpy()
    records = []
    for fold in folds:
        train = set(row_ids[fold.train_index])
        test = set(row_ids[fold.test_index])
        sub = membership.assign(_side=np.where(
            membership["row_id"].astype(str).isin(train), "train",
            np.where(membership["row_id"].astype(str).isin(test), "test", "absent")))
        sides = sub.groupby("curve_id")["_side"].nunique()
        mixed = sub.groupby("curve_id")["_side"].agg(
            lambda s: bool({"train", "test"} <= set(s)))
        records.append({
            "split_seed": int(fold.seed), "fold": int(fold.fold),
            "n_curves": int(len(sides)),
            "n_curves_pure": int((sides == 1).sum()),
            "n_curves_straddling_train_test": int(mixed.sum()),
        })
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Row accounting
# --------------------------------------------------------------------------- #

def row_accounting(oof: pd.DataFrame, expected_rows: int, *,
                   arm_column: str = "model",
                   prediction_column: str = "prediction",
                   truth_column: str = "log_D") -> pd.DataFrame:
    """Per arm per seed: nothing may quietly disappear.

    "The candidate looked better because its difficult rows were dropped" is the
    single easiest way to manufacture a gen9 result, and an inner join further down
    the pipeline hides it perfectly.
    """
    records = []
    for key, block in oof.groupby([arm_column, "split_seed"], sort=True):
        prediction = block[prediction_column].to_numpy(dtype=float)
        truth = block[truth_column].to_numpy(dtype=float)
        records.append({
            arm_column: key[0], "split_seed": int(key[1]),
            "expected_rows": int(expected_rows), "returned_rows": int(len(block)),
            "missing_rows": int(expected_rows - block["row_id"].nunique()),
            "duplicate_rows": int(len(block) - block["row_id"].nunique()),
            "nan_predictions": int(np.isnan(prediction).sum()),
            "inf_predictions": int(np.isinf(prediction).sum()),
            "nan_truth": int(np.isnan(truth).sum()),
            "scored_rows": int((np.isfinite(prediction) & np.isfinite(truth)).sum()),
        })
    return pd.DataFrame.from_records(records)


def assert_row_accounting_clean(accounting: pd.DataFrame) -> None:
    forbidden = ["missing_rows", "duplicate_rows", "nan_predictions",
                 "inf_predictions", "nan_truth"]
    bad = accounting[(accounting[forbidden] > 0).any(axis=1)]
    if len(bad):
        raise AssertionError(f"rows went missing:\n{bad}")


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #

PAIRING_KEYS: tuple[str, ...] = ("split_seed", "fold", "extractant", "repeat",
                                 "policy", "k", "n_pool", "n_eval")


def pairing_audit(detail: pd.DataFrame, arms: Sequence[str], *,
                  arm_column: str = "arm",
                  keys: Sequence[str] = PAIRING_KEYS) -> pd.DataFrame:
    """Do these arms really occupy the same units?

    Reports, for every pair of arms, how many units each has, how many they share
    and how many are unique to one — so a comparison labelled paired can be
    checked against a file instead of against a docstring.  ``policy`` is included
    in the key deliberately: two acquisition policies may select different *rows*,
    but they must still be compared over the same ligands, repeats and pool sizes,
    and a difference in ``n_pool`` between two arms means the pools were not the
    same pools.
    """
    usable = [k for k in keys if k in detail.columns]
    units = {}
    for arm in arms:
        block = detail[detail[arm_column] == arm]
        units[arm] = set(map(tuple, block[usable].astype(str).to_numpy().tolist()))
    records = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            shared = units[a] & units[b]
            records.append({
                "arm_a": a, "arm_b": b, "keys": "|".join(usable),
                "n_units_a": len(units[a]), "n_units_b": len(units[b]),
                "n_shared": len(shared),
                "n_only_a": len(units[a] - units[b]),
                "n_only_b": len(units[b] - units[a]),
                "identical": bool(units[a] == units[b] and units[a]),
            })
    return pd.DataFrame.from_records(records)


def assert_identical_units(detail: pd.DataFrame, arms: Sequence[str], **kwargs) -> pd.DataFrame:
    audit = pairing_audit(detail, list(arms), **kwargs)
    bad = audit[~audit["identical"]]
    if len(bad):
        raise AssertionError(
            "arms claimed to be paired do not share their units:\n"
            f"{bad[['arm_a', 'arm_b', 'n_units_a', 'n_units_b', 'n_shared']]}")
    return audit


def paired_frames(a: pd.DataFrame, b: pd.DataFrame, *,
                  keys: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two frames on ``keys`` and refuse rather than inner-join silently."""
    ka = a[list(keys)].astype(str).agg("|".join, axis=1)
    kb = b[list(keys)].astype(str).agg("|".join, axis=1)
    if set(ka) != set(kb):
        raise AssertionError(
            f"frames are not paired: {len(set(ka) - set(kb))} units only in the first, "
            f"{len(set(kb) - set(ka))} only in the second")
    left = a.assign(_key=ka).sort_values("_key").reset_index(drop=True)
    right = b.assign(_key=kb).sort_values("_key").reset_index(drop=True)
    return left.drop(columns="_key"), right.drop(columns="_key")
