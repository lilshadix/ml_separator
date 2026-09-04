"""The three transfer mechanisms, expressed in gen10's query contract.

The brief (§6) insists these be kept apart, and the architecture makes that easy
for a reason worth stating.  gen10's frozen model is a *recomposition*: a level
forest predicts each curve's height, a shape forest predicts the within-curve
deviation, and the two are recombined mean-preservingly.  gen7–gen10 established
that these two halves fail for different reasons — the level of an unseen ligand
is essentially unpredictable from structure (0.52 of 0.97 macro MAE is a
per-ligand constant), while the shape is learnable but evidenced by only 25
ligands over 8 chemotypes.

That gives the transfer question a sharper form than "does more data help":

* auxiliary metals plausibly teach **shape** — how log D responds to extractant
  and acid concentration is mass-action physics that does not care whether the
  cation is Eu or Am;
* they implausibly teach a lanthanide's **level**, which is the quantity the
  auxiliary metal does not share.

So the mechanisms are not three arbitrary implementations.  They are a
decomposition of where auxiliary information is allowed to act:

``JOINT``
    both forests see auxiliary rows.  The naive "train on everything" reading.
``SHARED_SHAPE``
    the shape forest sees auxiliary rows, the level forest does not.  The
    hypothesis above, stated as a model.
``SHARED_LEVEL``
    the complement — level sees auxiliary rows, shape does not.  Included
    because a hypothesis that is never given the chance to fail is not tested.
``AUX_PRETRAIN``
    an auxiliary-only forest is fitted first and its prediction becomes one extra
    column for a lanthanide-only model.  This is §6.2's "pretrain then fit" in
    the only form a forest supports honestly.  No cross-fitting is needed and
    none is faked: the auxiliary forest never sees a lanthanide row, so its
    prediction on a training row is already out-of-sample with respect to the
    target being fitted.

Every mechanism reduces to gen10's ``RecomposedModel`` exactly when the
auxiliary block is empty.  :func:`nesting_delta` asserts it, at the thread-order
floor, and the arm-A control in the runner is that assertion applied to the
whole evaluation rather than to one fold.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..gen7.harness import FoldContext
from ..gen9.curves import DEFAULT_AXES
from ..gen9.train import FROZEN_BLOCKS
from ..gen10.architectures import (
    CLIP_MULTIPLIER, DesignBuilder, Representation, _curve_means, _curve_of, _forest,
    _recompose, _tile, matrix, query_features, query_membership, representation,
)
from .arms import AuxiliaryPool, ArmSpec, apply_control, assert_series_disjoint, training_weights

#: The mechanisms, in the order the report presents them.
MECHANISMS: tuple[str, ...] = ("JOINT", "SHARED_SHAPE", "SHARED_LEVEL", "AUX_PRETRAIN")


@dataclass
class _FittedTransfer:
    """A fitted transfer arm.  ``predict`` is gen10's recomposition, unchanged."""

    design: DesignBuilder
    spec: Representation
    axes: tuple[str, ...]
    lo: float
    hi: float
    base: object
    shape: object
    gen9_compat: bool = True
    aux_model: object | None = None
    aux_design: DesignBuilder | None = None

    def _membership(self, query: pd.DataFrame) -> pd.DataFrame:
        return query_membership(query, axes=self.axes)

    def _context(self, query: pd.DataFrame, membership: pd.DataFrame) -> np.ndarray:
        if not self.spec.columns:
            return np.zeros((len(query), 0), dtype=float)
        features = query_features(query, membership, axes=self.axes,
                                  window=self.spec.window, gen9_compat=self.gen9_compat)
        return _tile(matrix(features, query["row_id"].astype(str).to_numpy(), self.spec), 1)

    def _static(self, query: pd.DataFrame) -> np.ndarray:
        static = self.design.transform(query)
        if self.aux_model is None:
            return static
        # The auxiliary forest's opinion, as one extra column.  It is appended
        # after the design transform so the lanthanide model's imputation and
        # column order are untouched by the mechanism.
        aux = np.asarray(self.aux_model.predict(self.aux_design.transform(query)), dtype=float)
        return np.hstack([static, aux.reshape(-1, 1)])

    def _clip(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, self.lo, self.hi)

    def predict(self, query: pd.DataFrame) -> np.ndarray:
        membership = self._membership(query)
        static = self._static(query)
        base = self._clip(np.asarray(self.base.predict(static), dtype=float))
        context = self._context(query, membership)
        shape = np.asarray(self.shape.predict(np.hstack([static, context])), dtype=float)
        curve = _curve_of(query, membership, self.axes)
        return self._clip(_recompose(base, shape, curve))

    def set_predict_jobs(self, n_jobs: int) -> "_FittedTransfer":
        for attribute in ("base", "shape", "aux_model"):
            forest = getattr(self, attribute, None)
            if forest is not None and hasattr(forest, "n_jobs"):
                forest.n_jobs = int(n_jobs)
        return self


@dataclass
class TransferRecomposedModel:
    """gen10's ``GEN9_SHAPE_RECOMPOSED`` with an auxiliary block admitted under a plan.

    The forest hyper-parameters, the clip rule, the design builder, the curve
    geometry and the recomposition are gen10's own objects, imported rather than
    restated, so this class can only differ from the frozen model in *which rows
    reach which forest* — which is exactly the variable gen11 is studying.

    ``clip_from_core`` keeps the output range a property of the lanthanide
    training rows.  An actinide's log D range is not the lanthanide range, and
    letting it widen the clip would change the frozen model's behaviour on rows
    the auxiliary data says nothing about.
    """

    name: str
    arm: ArmSpec
    pool: AuxiliaryPool
    mechanism: str = "JOINT"
    weighting: str = "HIERARCHICAL"
    aux_lambda: float = 1.0
    control: str = "NONE"
    matched_size: int | None = None
    matched_seed: int = 0
    metal_columns: tuple[str, ...] = ()
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    n_estimators: int = 400
    max_features: object = 0.30
    min_samples_leaf: int = 2
    axes: tuple[str, ...] = DEFAULT_AXES
    representation_name: str = "GEN9"
    window: str = "primary"
    gen9_compat: bool = True
    clip_from_core: bool = True
    diagnostics: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISMS:
            raise KeyError(f"unknown mechanism {self.mechanism!r}; have {MECHANISMS}")

    @property
    def spec(self) -> Representation:
        return representation(self.representation_name, window=self.window)

    # ------------------------------------------------------------------ #

    def _auxiliary_for_fold(self, context: FoldContext) -> pd.DataFrame:
        from .arms import ARM_BY_KEY, select_arm_rows

        fold = context.fold
        admissible = self.pool.admissible(fold.seed, fold.fold)

        # A size-matched arm matches the *fold's* admissible count of its
        # reference arm, not a global constant.  The leakage filter removes
        # different numbers of rows in different folds, so a fixed size would
        # make F larger than D on some folds and smaller on others — and the
        # whole point of F is that n is equal.
        matched_size = self.matched_size
        if self.arm.match_size_to is not None and matched_size is None:
            reference = ARM_BY_KEY[self.arm.match_size_to]
            matched_size = len(select_arm_rows(admissible, reference))
        rows = select_arm_rows(admissible, self.arm, matched_size=matched_size,
                               seed=self.matched_seed)
        if rows.empty:
            return rows
        # blake2b, never ``hash()``: Python salts the hash of a str per process, so
        # a control seeded from it draws a different permutation on every run and
        # cannot be reproduced or paired across processes.  gen8 lost a
        # cross-run comparison to exactly this.
        digest = hashlib.blake2b(
            f"{self.name}|{int(fold.seed)}|{int(fold.fold)}".encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big") % (2 ** 32))
        rows = apply_control(rows, control=self.control, rng=rng,
                             metal_columns=self.metal_columns)
        assert_series_disjoint(context.cohort.frame, rows)
        return rows

    def _augmentation_columns(self, context: FoldContext) -> list[str]:
        """Exactly the columns the augmented frame has to carry.

        The cohort frame is 5,642 columns wide because it accumulates every block
        any generation ever tested — 3D geometry, polyhedra, four pretrained
        embedding families.  An auxiliary row has none of those and never could:
        they are properties of a 3D structure the archive does not have.  Demanding
        the full width would fail every arm; carrying it would concatenate 3,500
        all-NaN columns onto every fit.

        So the contract is the *used* set: the design blocks, the identity columns
        the curve and weighting code reads, and the curve axes.  A column outside
        it cannot influence a prediction, and a missing one raises rather than
        arriving as a silent NaN.
        """
        needed: list[str] = list(context.cohort.block_columns(tuple(self.blocks)))
        for column in ("row_id", "series_id", "ecfp_cluster", "extractant",
                       "metal_symbol", "metal_Z", "lanthanide_index", "log_D",
                       "is_auxiliary", *self.axes):
            if column == "metal":       # a curve axis, not a column
                continue
            if column not in needed:
                needed.append(column)
        return needed

    def fit(self, train: pd.DataFrame, y_train: np.ndarray,
            context: FoldContext) -> _FittedTransfer:
        aux = self._auxiliary_for_fold(context)
        core = train.assign(is_auxiliary=False)
        if aux.empty:
            augmented = core
            y_aug = np.asarray(y_train, dtype=float)
        else:
            aux = aux.assign(is_auxiliary=True)
            columns = self._augmentation_columns(context)
            missing = [c for c in columns if c not in aux.columns]
            if missing:
                raise SystemExit(
                    f"auxiliary rows lack {len(missing)} required columns "
                    f"(e.g. {missing[:5]}); the featurizer contract is broken")
            augmented = pd.concat([core[columns], aux[columns]], ignore_index=True)
            y_aug = np.concatenate([np.asarray(y_train, dtype=float),
                                    aux["log_D"].to_numpy(dtype=float)])

        is_aux = augmented["is_auxiliary"].to_numpy(dtype=bool)
        weights = training_weights(augmented, scheme=self.weighting, aux_lambda=self.aux_lambda)

        # The clip range is a property of the lanthanide core by default, so an
        # auxiliary metal cannot widen the answer for rows it says nothing about.
        clip_source = y_aug[~is_aux] if (self.clip_from_core and (~is_aux).any()) else y_aug
        spread = float(clip_source.max() - clip_source.min())
        lo = float(clip_source.min() - CLIP_MULTIPLIER * spread)
        hi = float(clip_source.max() + CLIP_MULTIPLIER * spread)

        aux_model = aux_design = None
        if self.mechanism == "AUX_PRETRAIN" and is_aux.any():
            aux_design = DesignBuilder(blocks=tuple(self.blocks),
                                       add_indicator=self.add_indicator).fit(
                context.cohort, augmented[is_aux])
            aux_model = _forest(context.model_seed + 2, n_estimators=self.n_estimators,
                                max_features=self.max_features,
                                min_samples_leaf=self.min_samples_leaf)
            aux_model.fit(aux_design.transform(augmented[is_aux]), y_aug[is_aux],
                          sample_weight=weights[is_aux])

        # Which rows each forest is allowed to see.  This is the mechanism.
        if self.mechanism in ("JOINT",):
            level_mask = shape_mask = np.ones(len(augmented), dtype=bool)
        elif self.mechanism == "SHARED_SHAPE":
            level_mask, shape_mask = ~is_aux, np.ones(len(augmented), dtype=bool)
        elif self.mechanism == "SHARED_LEVEL":
            level_mask, shape_mask = np.ones(len(augmented), dtype=bool), ~is_aux
        else:  # AUX_PRETRAIN — the lanthanide model is fitted on core rows only
            level_mask = shape_mask = ~is_aux

        design_rows = augmented[level_mask]
        design = DesignBuilder(blocks=tuple(self.blocks),
                               add_indicator=self.add_indicator).fit(context.cohort, design_rows)

        fitted = _FittedTransfer(
            design=design, spec=self.spec, axes=tuple(self.axes), lo=lo, hi=hi,
            base=None, shape=None, gen9_compat=self.gen9_compat,
            aux_model=aux_model, aux_design=aux_design)

        static_all = fitted._static(augmented)
        base = _forest(context.model_seed, n_estimators=self.n_estimators,
                       max_features=self.max_features, min_samples_leaf=self.min_samples_leaf)
        base.fit(static_all[level_mask], y_aug[level_mask], sample_weight=weights[level_mask])

        # Curve geometry is computed on the rows the shape forest actually sees,
        # so an auxiliary titration forms its own curve and is centred on its own
        # mean — never on a cohort curve's.
        shape_rows = augmented[shape_mask]
        membership = query_membership(shape_rows, axes=tuple(self.axes))
        curve = _curve_of(shape_rows, membership, tuple(self.axes))
        centred = y_aug[shape_mask] - _curve_means(curve, y_aug[shape_mask])
        usable = np.isfinite(centred)
        x_shape = np.hstack([static_all[shape_mask], fitted._context(shape_rows, membership)])
        shape = _forest(context.model_seed + 1, n_estimators=self.n_estimators,
                        max_features=self.max_features, min_samples_leaf=self.min_samples_leaf)
        shape.fit(x_shape[usable], centred[usable],
                  sample_weight=weights[shape_mask][usable])

        fitted.base, fitted.shape = base, shape
        self.diagnostics.append({
            "arm": self.name, "mechanism": self.mechanism, "weighting": self.weighting,
            "aux_lambda": float(self.aux_lambda), "control": self.control,
            "split_seed": int(context.fold.seed), "fold": int(context.fold.fold),
            "n_core": int((~is_aux).sum()), "n_aux": int(is_aux.sum()),
            "n_level_train": int(level_mask.sum()), "n_shape_train": int(usable.sum()),
            "aux_weight_share": float(weights[is_aux].sum() / weights.sum()) if is_aux.any() else 0.0,
            "clip_lo": lo, "clip_hi": hi,
        })
        return fitted


def nesting_delta(cohort, *, seed: int, fold_index: int = 0) -> dict:
    """Prove the transfer class *is* gen10's model when the auxiliary block is empty.

    gen10 asserted the same property of itself against gen9 (``nests gen9 arms at
    1e-15``); repeating it here is what makes arm A a control rather than a
    reimplementation that happens to score similarly.
    """
    from ..gen7.harness import FoldContext, build_folds
    from ..gen10.architectures import RecomposedModel
    from ..levels import LEVEL_TARGET_COLUMN
    from .arms import ARM_BY_KEY

    frame = cohort.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    fold = build_folds(frame, seed)[fold_index]
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
    context = FoldContext(fold=fold, cohort=cohort, feature_columns=(),
                          model_seed=fold.model_seed)
    y = target[fold.train_index]

    reference = RecomposedModel(representation_name="GEN9", gen9_compat=True).fit(
        train, y, context).predict(test)
    empty = AuxiliaryPool(
        features=pd.DataFrame(columns=list(frame.columns) + ["metal_category", "source_record_id"]),
        safe_ids={(int(fold.seed), int(fold.fold)): frozenset()}, policy="HEADLINE")
    candidate = TransferRecomposedModel(
        name="A_GEN10_CONTROL", arm=ARM_BY_KEY["A_GEN10_CONTROL"], pool=empty).fit(
        train, y, context).predict(test)
    delta = np.abs(np.asarray(reference) - np.asarray(candidate))
    return {"split_seed": int(seed), "fold": int(fold_index), "n_rows": int(len(delta)),
            "max_abs_delta": float(delta.max()), "mean_abs_delta": float(delta.mean()),
            "n_rows_moved": int((delta > 0).sum())}
