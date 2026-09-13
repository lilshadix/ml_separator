"""The bridge to Gen12: cohort, folds, similarity and frozen predictions, unchanged.

Everything Gen12.2 compares against is loaded through this module so that the
"same frozen design-B held-out chemistry" invariant is a property of the code
path and not of discipline.  The fold plan is rebuilt with Gen12's own builder
and then *asserted* against the fold plan Gen12 wrote to disk, row id by row id.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import paths  # noqa: F401  (puts gen12_eu_pred/ and src/ on sys.path)
from gen12eu import splits  # noqa: E402
from gen12eu.chemistry import band_of  # noqa: E402
from gen12eu.cohort import Gen12Cohort, build_cohort  # noqa: E402

TARGET = "log_D"


@dataclass
class FrozenDesignB:
    cohort: Gen12Cohort
    folds: list
    similarity: pd.DataFrame

    @property
    def frame(self) -> pd.DataFrame:
        return self.cohort.frame


@lru_cache(maxsize=1)
def load_design_b() -> FrozenDesignB:
    """Rebuild the Gen12 cohort and design-B folds and prove they match Gen12's plan."""
    cohort = build_cohort()
    if cohort.fingerprint != paths.GEN12_COHORT_FINGERPRINT:
        raise RuntimeError(f"cohort fingerprint {cohort.fingerprint} != Gen12's "
                           f"{paths.GEN12_COHORT_FINGERPRINT}; refusing to compare")
    folds = splits.all_folds(cohort.frame, design="B")
    plan = json.loads(paths.GEN12_FOLD_PLAN_B.read_text())
    if plan["cohort_fingerprint"] != cohort.fingerprint:
        raise RuntimeError("Gen12 fold plan was written against a different cohort")
    if len(plan["folds"]) != len(folds):
        raise RuntimeError("fold count differs from Gen12's plan")
    for fold, planned in zip(folds, plan["folds"]):
        if fold.seed != planned["seed"] or fold.fold != planned["fold"]:
            raise RuntimeError("fold order differs from Gen12's plan")
        ids = cohort.frame.iloc[fold.test_index]["row_id"].tolist()
        if ids != planned["test_row_ids"]:
            raise RuntimeError(f"seed {fold.seed} fold {fold.fold}: test rows differ from Gen12")
        if list(fold.held_out_groups) != planned["held_out_groups"]:
            raise RuntimeError(f"seed {fold.seed} fold {fold.fold}: held-out chemotypes differ")
    integrity = splits.assert_fold_integrity(cohort.frame, folds)
    if not integrity["ok"]:
        raise RuntimeError("design-B fold integrity failed")
    similarity = pd.read_parquet(paths.GEN12_PREDICTIONS / "B" / "similarity.parquet")
    rebuilt = splits.similarity_table(cohort.frame, folds)
    merged = similarity.merge(rebuilt, on=["design", "split_seed", "fold", "extractant"],
                              suffixes=("", "_rebuilt"), validate="one_to_one")
    if not np.allclose(merged["max_train_tanimoto"], merged["max_train_tanimoto_rebuilt"]):
        raise RuntimeError("similarity table differs from Gen12's")
    similarity = similarity.copy()
    similarity["band"] = band_of(similarity["max_train_tanimoto"]).to_numpy()
    return FrozenDesignB(cohort=cohort, folds=folds, similarity=similarity)


def gen12_predictions(arm: str) -> pd.DataFrame:
    """A frozen Gen12 design-B prediction frame, by arm name, from whichever directory holds it."""
    for sub in ("B", "B_ablation", "B_multiln"):
        path = paths.GEN12_PREDICTIONS / sub / f"{arm}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            if frame["arm"].iloc[0] != arm:
                raise RuntimeError(f"{path} carries arm {frame['arm'].iloc[0]!r}")
            return frame
    raise FileNotFoundError(arm)


def training_rows(fd: FrozenDesignB, fold) -> pd.DataFrame:
    return fd.frame.iloc[fold.train_index]


def test_rows(fd: FrozenDesignB, fold) -> pd.DataFrame:
    return fd.frame.iloc[fold.test_index]
