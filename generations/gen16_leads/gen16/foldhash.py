"""SHA-256 fingerprint of the five-design fold plan.

``hash()`` is salted per process and silently broke cross-run pairing in gen8, so split
reproducibility is checked with a content hash, and checked in a *subprocess*
(``scripts/g16_fold_hash.py``), never only in-session.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from gen16 import bootstrap  # noqa: F401  (sys.path + thread cap)
from gen13sep.splits import all_folds  # noqa: E402
from gen15.valuebench import DESIGNS  # noqa: E402


def fold_plan_digest(frame: pd.DataFrame, designs=DESIGNS) -> dict:
    """Hashes over every fold of every design, in canonical design order (B, BR, BQ, A, BP).

    ``sha256``            over the concatenated *sorted test-index* arrays of every fold (int64).
    ``sha256_train_test`` the same, with each fold's sorted train index appended after its test
                          index -- a stricter fingerprint kept alongside the required one.
    """
    h_test, h_full = hashlib.sha256(), hashlib.sha256()
    per_design, n_folds = {}, {}
    for d in designs:
        h_d = hashlib.sha256()
        folds = all_folds(frame, design=d)
        for f in folds:
            te = np.sort(np.asarray(f.test_index, dtype=np.int64)).tobytes()
            tr = np.sort(np.asarray(f.train_index, dtype=np.int64)).tobytes()
            h_d.update(te)
            h_test.update(te)
            h_full.update(te)
            h_full.update(tr)
        per_design[d] = h_d.hexdigest()
        n_folds[d] = len(folds)
    return {"sha256": h_test.hexdigest(), "sha256_train_test": h_full.hexdigest(),
            "per_design": per_design, "n_folds": n_folds, "designs": list(designs)}
