"""``folds/random_split.py`` -- V0, the row-level random split (pre-registration section 3.6; diagnostic only).

Five folds per discovery seed over every MODEL row.  A row's fold is drawn from the seed's ``V0`` stream:
the sorted ``canonical_measurement_id`` list is permuted and dealt round robin, so the five folds differ in
size by at most one row.  ``V6_TARGET_ROWS`` are hidden with their fold (they are trained on in the other
four) and never scored; X(?) and Sr(III) rows are never scored either (:func:`io.scorable_mask`).

V0 has no ``fold_isolation_check`` level (the pre-registration's guard list is V1-V6): near-duplicates cross a
random split by design, which is why V0 is never a claim.  The builder asserts the partition only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gen19ct.folds import io as FI

DESIGN, VARIANT, SCHEME = "V0", "rows", "random5"
N_SPLITS = 5


def v0_folds(frame: pd.DataFrame, seed: int, v6_mask: pd.Series, n_splits: int = N_SPLITS) -> list[FI.Fold]:
    ids = np.asarray(sorted(frame[FI.ROW_ID].astype(str)), dtype=object)
    rng = FI.seed_rng(seed, "V0")
    label = np.empty(len(ids), dtype=int)
    label[rng.permutation(len(ids))] = np.arange(len(ids)) % n_splits
    ok = set(frame.loc[FI.scorable_mask(frame, v6_mask).to_numpy(), FI.ROW_ID].astype(str))
    folds = []
    for k in range(n_splits):
        hid = ids[label == k].tolist()
        folds.append(FI.make_fold(design=DESIGN, variant=VARIANT, scheme=SCHEME, fold_id=f"s{seed}_f{k}", half="NA",
                                  seed=seed, hidden=hid, scored=[r for r in hid if r in ok], unit_type="row",
                                  units=[f"random_fold_{k}"]))
    return folds


def check_partition(folds: list[FI.Fold], frame: pd.DataFrame) -> None:
    """Every MODEL row is hidden in exactly one fold of each seed."""
    all_ids = set(frame[FI.ROW_ID].astype(str))
    for seed in sorted({f.seed for f in folds}):
        seen: list[str] = [r for f in folds if f.seed == seed for r in f.hidden_row_ids]
        if len(seen) != len(set(seen)) or set(seen) != all_ids:
            raise AssertionError(f"V0 seed {seed}: folds do not partition the MODEL rows")
