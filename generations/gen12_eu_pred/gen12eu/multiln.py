"""Cross-lanthanide transfer: does knowing Nd, La and Dy help predict Eu?

The whole difficulty of this arm is one number from the audit: **4,148 of the
4,426 non-Eu rows sit on extractants already in the Eu cohort**, and only 7
non-Eu extractants are chemistry the Eu cohort does not have.  A multi-metal
model trained naively has therefore already seen almost every test molecule under
a different metal, and its "unseen extractant" score is not one.

So the strict arm deletes, per fold, **every auxiliary row of every held-out
chemotype under every metal** — not merely the held-out Eu rows.  The filter uses
the frozen chemotype label, a property of the structure computed without the
target; a deployment knows which molecule it is asking about, so this is not
privileged information, and the exclusion can only remove training data.

Because a transfer gain and a data-volume gain look identical on a leaderboard,
three controls run alongside on byte-identical folds and test rows:

``MATCHED_EU_ONLY``
    Eu rows only, re-weighted to the multi-metal arm's total training mass.  If
    "more effective rows" is the whole story, this arm captures it.
``PERMUTED_METAL``
    the same auxiliary rows with the metal descriptors permuted between rows.
    Keeps rows, chemistry, conditions and targets; destroys only metal identity.
    A gain that survives was never about the metal.
``SHUFFLED_TARGET``
    the same auxiliary rows with their targets permuted.  Keeps every count and
    every feature, destroys the information.  An arm that does not beat this is
    measuring regularisation.

``LEAKY_NO_FILTER`` is included deliberately as the *wrong* analysis — the arm a
naive multi-metal study would report — so that the size of the leak is measured
instead of asserted.  It is labelled non-deployable everywhere it appears.

gen11 is why all of this exists: its actinide arm showed a large, seed-consistent,
level-only gain whose interval touched zero, measured against a control the
auxiliary data itself had forced, with the decisive size-matched comparison never
run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import paths
from .chemistry import frozen_chemotypes
from .cohort import TARGET, _donor_census, _mass_action
from .models import DEFAULT_BLOCKS, TreeContender
from .preprocess import FoldPreprocessor, clip_to_training_range
from lanthanide_separation.levels import condition_labels  # noqa: E402
from lanthanide_separation.pairs import TODGA_SMILES  # noqa: E402

#: Metal descriptors — constant in the Eu cohort, informative once other metals enter.
METAL_COLUMNS: tuple[str, ...] = ("Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal")
POLICIES: tuple[str, ...] = ("STRICT", "MATCHED_EU_ONLY", "PERMUTED_METAL", "SHUFFLED_TARGET",
                             "LEAKY_NO_FILTER")
NON_DEPLOYABLE: frozenset = frozenset({"LEAKY_NO_FILTER"})


def build_auxiliary(cohort_columns) -> pd.DataFrame:
    """Every non-Eu lanthanide row, processed exactly like the Eu cohort."""
    source = pd.read_parquet(paths.BUNDLE_PARQUET)
    frame = source[source["metal"].astype(str) != "Eu"].copy()
    wrong_todga = frame["canonical_smiles"].eq(TODGA_SMILES) & ~frame["extractant_name"].eq("TODGA")
    frame = frame.loc[~wrong_todga].copy()
    condition_columns = tuple(c for c in frame.columns if c.startswith("cond__"))
    frame = frame.assign(extractant=frame["canonical_smiles"].astype(str),
                         condition_id=condition_labels(frame, condition_columns))
    frame = frame.assign(chemotype=frozen_chemotypes(frame["extractant"])["chemotype"].to_numpy())
    lig2d = pd.read_parquet(paths.LIG2D_PARQUET).drop_duplicates("canonical_smiles")
    lig2d_columns = [c for c in lig2d.columns if c.startswith("lig2d__")]
    frame = frame.merge(lig2d[["canonical_smiles", *lig2d_columns]], on="canonical_smiles",
                        how="left", validate="many_to_one")
    frame = pd.concat([frame, _donor_census(frame), _mass_action(frame)], axis=1)
    cell = ["extractant", "condition_id", "metal_symbol"]
    frame[TARGET] = frame.groupby(cell)[TARGET].transform("mean")
    frame = frame.drop_duplicates(subset=cell, keep="first").reset_index(drop=True)
    frame["row_id"] = ["aux:" + "|".join(map(str, r)) for r in frame[cell].to_numpy()]
    keep = [c for c in cohort_columns if c in frame.columns]
    extra = [c for c in (*METAL_COLUMNS, "metal_symbol") if c in frame.columns]
    return frame[list(dict.fromkeys(keep + extra))]


def attach_metal_columns(eu_frame: pd.DataFrame) -> pd.DataFrame:
    """Give the Eu cohort the metal descriptors it dropped, at their Eu values."""
    return eu_frame.assign(**{"Atomic Number_metal": 63.0, "lanthanide_index": 7.0,
                              "Ionic Radius_metal": 1.066, "metal_symbol": "Eu"})


@dataclass
class MultiLanthanideContender:
    """The Tier-1 champion's learner, trained on a policy-selected row pool.

    The Eu training rows, the folds, the test rows and the learner are identical
    in every policy, so a leaderboard difference is the policy and nothing else.
    ``reference_mass`` is the multi-metal arm's total training mass, supplied so
    that ``MATCHED_EU_ONLY`` can match it exactly.
    """

    auxiliary: pd.DataFrame
    policy: str = "STRICT"
    family: str = "extratrees"
    name: str = ""
    tier: str = "T4"
    blocks: tuple[str, ...] = DEFAULT_BLOCKS
    reference_mass: dict = field(default_factory=dict)
    selected_: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise KeyError(f"unknown policy {self.policy!r}")
        if not self.name:
            self.name = f"T4_{self.policy}"

    def _pool(self, train, test, seed):
        held_out = set(test["chemotype"].astype(str))
        aux = self.auxiliary
        if self.policy != "LEAKY_NO_FILTER":
            aux = aux[~aux["chemotype"].astype(str).isin(held_out)]
        overlap = set(aux["extractant"].astype(str)) & set(test["extractant"].astype(str))
        audit = {"aux_rows_available": int(len(self.auxiliary)),
                 "aux_rows_admitted": int(len(aux)),
                 "aux_extractants": int(aux["extractant"].nunique()),
                 "held_out_chemotypes": len(held_out),
                 "aux_test_extractant_overlap": len(overlap)}
        if self.policy != "LEAKY_NO_FILTER" and overlap:
            raise AssertionError(
                f"{len(overlap)} test extractants survive in the auxiliary pool under "
                f"policy {self.policy!r}; the strict filter has failed")
        rng = np.random.default_rng(seed)
        aux = aux.copy()
        if self.policy == "PERMUTED_METAL":
            order = rng.permutation(len(aux))
            for column in (*METAL_COLUMNS, "metal_symbol"):
                aux[column] = aux[column].to_numpy()[order]
        elif self.policy == "SHUFFLED_TARGET":
            aux[TARGET] = aux[TARGET].to_numpy()[rng.permutation(len(aux))]
        elif self.policy == "MATCHED_EU_ONLY":
            aux = aux.iloc[:0]
        eu = attach_metal_columns(train)
        pool = pd.concat([eu, aux], ignore_index=True) if len(aux) else eu
        audit["pool_rows"] = int(len(pool))
        return pool, audit

    @staticmethod
    def _weights(pool: pd.DataFrame, target_mass: float | None) -> np.ndarray:
        counts = pool["extractant"].astype(str).value_counts()
        weights = pool["extractant"].astype(str).map(lambda e: 1.0 / float(counts[e])).to_numpy(float)
        weights = weights * (len(weights) / weights.sum())
        if target_mass is not None and weights.sum() > 0:
            weights = weights * (float(target_mass) / weights.sum())
        return weights

    def fit_predict(self, train, y_train, test, context):
        pool, audit = self._pool(train, test, context.model_seed)
        columns = list(context.columns(*self.blocks)) + list(METAL_COLUMNS)
        key = (context.split_seed, context.fold)
        target_mass = self.reference_mass.get(key) if self.policy == "MATCHED_EU_ONLY" else None
        weights = self._weights(pool, target_mass)
        pre = FoldPreprocessor(tuple(columns)).fit(pool)
        model = TreeContender(family=self.family, blocks=self.blocks)._fit(
            pre.transform(pool), pool[TARGET].to_numpy(dtype=float), weights,
            {"max_features": 0.30, "min_samples_leaf": 2}, context.model_seed)
        self.selected_ = {**audit, "training_mass": float(weights.sum()),
                          "matched_to_mass": target_mass}
        if self.policy == "STRICT":
            self.reference_mass[key] = float(weights.sum())
        prediction = model.predict(pre.transform(attach_metal_columns(test)))
        return clip_to_training_range(prediction, pool[TARGET].to_numpy(dtype=float))
