"""The pre-registered arms, auxiliary pools, weighting schemes and controls.

Everything a result could be accused of being chosen after the fact lives here
and is fixed before any arm runs: which rows each arm may see, how they are
weighted, how the size-matched controls are drawn, and which negative controls
must come out flat.

Two decisions are worth stating in full, because they shape every number.

**Auxiliary rows are filtered per fold, from a pre-computed table.**  A fold
holds out Tanimoto chemotypes; an auxiliary row is admissible only if it is
related to none of the held-out rows under the fold's policy
(:mod:`.overlap`).  That table is built once, audited once, and looked up by
``(split_seed, fold)`` at fit time.  A missing key is an error rather than an
empty filter — a silently unfiltered fold is exactly the failure this design
exists to prevent.

**Auxiliary series are namespaced.**  Curves are grouped on ``series_id``
before anything else (``gen8.series.build_curve_table``), so an auxiliary row
carrying a cohort series id would join a cohort curve and an Am point would
land inside a lanthanide metal series.  Auxiliary rows therefore carry
``aux:<archive series id>``, which cannot collide, and
:func:`assert_series_disjoint` proves it.

The size-matched arms (F, G) are the reason this file has a sampler at all.
The archive's auxiliary pool is ~90 % actinide, so "actinides help" and "more
rows help" are confounded unless the comparison is drawn at equal n.  Matched
draws are repeated (``MATCHED_REPEATS``) with a fixed seed sequence so the
comparison is against a distribution, not one lucky sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..levels import group_balanced_weights

#: Metal categories as the archive labels them.
ACTINIDE = "actinide"
LANTHANIDE = "lanthanide"
NON_ACTINIDE_OTHER: tuple[str, ...] = (
    "alkaline_earth", "transition_metal", "rare_earth_non_lanthanide",
    "post_transition_metal",
)

#: How many independent matched draws each size-matched arm averages over.
MATCHED_REPEATS = 5
#: Fixed, so a matched draw is reproducible and cannot be re-rolled after seeing a result.
MATCHED_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArmSpec:
    """One pre-registered auxiliary composition."""

    key: str
    description: str
    #: metal categories admitted; ``None`` means "every category".
    categories: tuple[str, ...] | None
    #: draw a subset size-matched to this arm's pool, instead of taking all rows
    match_size_to: str | None = None
    #: sample the matched subset from these categories (defaults to ``categories``)
    match_pool: tuple[str, ...] | None = None

    @property
    def is_control(self) -> bool:
        return self.categories is not None and len(self.categories) == 0


#: Brief §5.  ``A`` is the frozen gen10 control and must reproduce it exactly.
ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("A_GEN10_CONTROL", "frozen gen10 pipeline, no auxiliary rows", ()),
    ArmSpec("B_LN_EXPANDED", "auxiliary lanthanide observations only", (LANTHANIDE,)),
    ArmSpec("C_LN_PLUS_ACTINIDES", "auxiliary actinide observations", (ACTINIDE,)),
    ArmSpec("D_LN_PLUS_NON_ACTINIDE", "auxiliary non-lanthanide non-actinide metals",
            NON_ACTINIDE_OTHER),
    ArmSpec("E_LN_PLUS_ALL", "every safe auxiliary row", None),
    ArmSpec("F_ACTINIDES_ONLY_MATCHED",
            "actinide subset size-matched to D — chemical relevance at equal n",
            (ACTINIDE,), match_size_to="D_LN_PLUS_NON_ACTINIDE"),
    ArmSpec("G_RANDOM_AUX_MATCHED",
            "random non-lanthanide subset size-matched to D — does row count alone explain it",
            (ACTINIDE,) + NON_ACTINIDE_OTHER, match_size_to="D_LN_PLUS_NON_ACTINIDE"),
)

ARM_BY_KEY: Mapping[str, ArmSpec] = {a.key: a for a in ARMS}


# --------------------------------------------------------------------------- #
# Weighting (brief §12)
# --------------------------------------------------------------------------- #

#: Names of the weighting schemes.  Fixed in advance; never selected per fold.
WEIGHTINGS: tuple[str, ...] = ("ROW", "CHEMOTYPE", "METAL_BALANCED", "HIERARCHICAL")


def training_weights(train: pd.DataFrame, *, scheme: str, aux_lambda: float = 1.0) -> np.ndarray:
    """Sample weights for an augmented training frame.

    ``train`` must carry ``ecfp_cluster``, ``metal_symbol`` and the boolean
    ``is_auxiliary``.  ``aux_lambda`` scales the auxiliary block's *total* mass
    relative to the lanthanide core, which is the knob that stops 4,896 actinide
    rows from silently becoming the objective.  It is a declared constant of an
    arm, never fitted.

    * ``ROW`` — every row weight 1.  The row-weighted comparison the brief asks
      for, and the one that lets a large metal dominate.
    * ``CHEMOTYPE`` — gen5..gen10's rule, one ECFP cluster one vote, applied to
      the augmented frame.
    * ``METAL_BALANCED`` — equal total mass per metal, so Am cannot outvote Eu.
    * ``HIERARCHICAL`` — one chemotype one vote *within* the lanthanide core and
      *within* the auxiliary block separately, then the two blocks combined at
      the declared ratio.  This is the scheme that preserves gen10's philosophy
      while admitting auxiliary rows, and it is the default for headline arms.
    """
    if scheme not in WEIGHTINGS:
        raise KeyError(f"unknown weighting {scheme!r}; have {WEIGHTINGS}")
    is_aux = train["is_auxiliary"].to_numpy(dtype=bool)
    if scheme == "ROW":
        # Deliberately unscaled: "one row one vote" is the scheme whose entire
        # purpose is to show what happens when a metal with thousands of rows is
        # allowed to dominate.  Rescaling it would erase the arm.
        return np.ones(len(train), dtype=float)
    if scheme == "CHEMOTYPE":
        weights = group_balanced_weights(train["ecfp_cluster"])
    elif scheme == "METAL_BALANCED":
        weights = group_balanced_weights(train["metal_symbol"])
    else:  # HIERARCHICAL
        weights = np.zeros(len(train), dtype=float)
        for mask in (~is_aux, is_aux):
            if mask.any():
                weights[mask] = group_balanced_weights(train.loc[mask, "ecfp_cluster"])
    return _scale_auxiliary(weights, is_aux, aux_lambda)


def _scale_auxiliary(weights: np.ndarray, is_aux: np.ndarray, aux_lambda: float) -> np.ndarray:
    """Set the auxiliary block's total mass to ``aux_lambda`` x the core's.

    Normalising the *total* rather than each row is what makes ``aux_lambda``
    mean the same thing at 86 auxiliary rows and at 4,896 of them; otherwise the
    same lambda would be two different experiments in arms B and C.
    """
    out = np.asarray(weights, dtype=float).copy()
    core_mass = float(out[~is_aux].sum())
    aux_mass = float(out[is_aux].sum())
    if aux_mass <= 0 or core_mass <= 0:
        return out
    out[is_aux] *= (aux_lambda * core_mass) / aux_mass
    return out


# --------------------------------------------------------------------------- #
# Negative controls (brief §19)
# --------------------------------------------------------------------------- #

#: Controls that must show **no** benefit.  If one does, the gain is an artefact.
CONTROLS: tuple[str, ...] = (
    "NONE",
    "PERMUTED_AUX_TARGET",     # auxiliary log_D shuffled within the auxiliary block
    "SHUFFLED_METAL_LABELS",   # auxiliary metal features shuffled across auxiliary rows
)


def apply_control(aux: pd.DataFrame, *, control: str, rng: np.random.Generator,
                  metal_columns: Sequence[str]) -> pd.DataFrame:
    """Corrupt the auxiliary block in the way ``control`` names.

    Deliberately corrupts *only* the auxiliary rows and leaves their count,
    conditions and chemistry intact, so a benefit that survives the corruption
    cannot be attributed to the information the corruption destroyed.
    """
    if control not in CONTROLS:
        raise KeyError(f"unknown control {control!r}; have {CONTROLS}")
    if control == "NONE" or aux.empty:
        return aux
    out = aux.copy()
    if control == "PERMUTED_AUX_TARGET":
        out["log_D"] = rng.permutation(out["log_D"].to_numpy())
        return out
    columns = [c for c in metal_columns if c in out.columns]
    if columns:
        order = rng.permutation(len(out))
        out[columns] = out[columns].to_numpy()[order]
    return out


# --------------------------------------------------------------------------- #
# Pools
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AuxiliaryPool:
    """Featurized auxiliary rows plus the per-fold admissibility table.

    ``safe_ids`` maps ``(split_seed, fold)`` to the archive record ids that the
    fold's policy admits.  It is computed once by the leakage audit, so what a
    model may train on is a reviewed artefact rather than a runtime decision.
    """

    features: pd.DataFrame
    safe_ids: Mapping[tuple[int, int], frozenset[str]]
    policy: str
    id_column: str = "source_record_id"

    def admissible(self, split_seed: int, fold: int) -> pd.DataFrame:
        key = (int(split_seed), int(fold))
        if key not in self.safe_ids:
            raise KeyError(
                f"no auxiliary admissibility entry for seed {split_seed} fold {fold}; "
                "refusing to train on an unfiltered pool")
        keep = self.safe_ids[key]
        return self.features[self.features[self.id_column].isin(keep)]

    def restrict(self, mask: np.ndarray | pd.Series) -> "AuxiliaryPool":
        return replace(self, features=self.features.loc[np.asarray(mask, dtype=bool)])


def select_arm_rows(pool: pd.DataFrame, spec: ArmSpec, *,
                    matched_size: int | None = None, seed: int = 0) -> pd.DataFrame:
    """The auxiliary rows one arm draws from an already fold-filtered pool."""
    if spec.categories is not None and len(spec.categories) == 0:
        return pool.iloc[:0]
    frame = pool
    categories = spec.match_pool or spec.categories
    if categories is not None:
        frame = frame[frame["metal_category"].isin(list(categories))]
    if spec.match_size_to is None or matched_size is None:
        return frame
    if len(frame) <= matched_size:
        return frame
    rng = np.random.default_rng(seed)
    take = rng.choice(len(frame), size=int(matched_size), replace=False)
    return frame.iloc[np.sort(take)]


def assert_series_disjoint(cohort_frame: pd.DataFrame, aux: pd.DataFrame) -> None:
    """No auxiliary row may share a ``series_id`` with the cohort.

    Curves are grouped on ``series_id`` first, so a shared id silently merges an
    auxiliary point into a cohort titration — the exact failure the ``aux:``
    namespace exists to prevent.  Checked rather than trusted.
    """
    if aux.empty:
        return
    shared = set(aux["series_id"].astype(str)) & set(cohort_frame["series_id"].astype(str))
    if shared:
        raise SystemExit(
            f"{len(shared)} auxiliary series ids collide with cohort series ids; "
            f"example {sorted(shared)[:3]}")
