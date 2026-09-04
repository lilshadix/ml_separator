"""The gen9 contender: same cohort, same folds, same design matrix, new objective.

This module exists to make one comparison airtight.  ``REC_ecfp_plus_recovered``
— gen7's champion and gen8's frozen model — is an ``ExtraTreesRegressor`` on
``METAL + COND + ECFP + MASSACTION + RECOVERED`` with median imputation and
missingness indicators, cluster-balanced sample weights, and predictions clipped
to 1.5x the training range.  :class:`CurveBoostContender` reproduces every one of
those choices and changes exactly one thing: how the trees are grown.

That matters because the gen9 claim is about an *objective*, and the cheapest way
to get a wrong answer would be to compare a shape-trained booster against a
differently-parameterised forest and attribute the difference to the loss.  So the
design matrix is built by :func:`build_design`, which is a line-for-line
transcription of ``gen7.contenders.Tabular.fit_predict``'s preprocessing, and the
control arm ``A0_ROW_ONLY`` is the same booster with the curve terms switched off.

Three arms therefore appear in every gen9 table:

``FROZEN_ET``
    the gen7/gen8 model itself, refitted through this path, which must reproduce
    macro MAE 0.9807 — the longitudinal anchor;
``A0_ROW_ONLY``
    the booster with ``lambda = 0``: the *paired* control for every shape arm, and
    the arm that tells us what changing learner costs before any shape term is
    added;
``A1..A4`` and the weight grid
    the shape arms.

A gain quoted against ``FROZEN_ET`` confounds learner and objective.  A gain quoted
against ``A0_ROW_ONLY`` does not, and that is the one the report leads with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..gen7.harness import FoldContext, REPO_ROOT
from ..levels import group_balanced_weights
from .curves import AXIS_SETS, CurvePairs, build_pairs
from .relative import RELATIVE_COLUMNS
from .objective import BoostConfig, CurveBoost, ShapeArm

#: The frozen arm's feature blocks, in the order ``suites._finalists`` lists them.
FROZEN_BLOCKS: tuple[str, ...] = ("METAL", "COND", "ECFP", "MASSACTION", "RECOVERED")
FROZEN_ARM_NAME = "REC_ecfp_plus_recovered"

#: gen9's own recomputation of the curve geometry, written by ``gen9_curves.py``.
GEN9_MEMBERSHIP_PATH = REPO_ROOT / "runs" / "gen9_shape" / "curves" / "curve_membership.parquet"
#: gen8's, kept only so the two can be asserted identical.
GEN8_MEMBERSHIP_PATH = REPO_ROOT / "runs" / "gen8_architecture" / "series" / "curve_membership.parquet"


def build_design(cohort, train: pd.DataFrame, test: pd.DataFrame, *,
                 blocks: Sequence[str] = FROZEN_BLOCKS,
                 add_indicator: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The frozen arm's preprocessing, fitted on the training fold only.

    Median imputation and the missingness indicators are *not* cosmetic here: gen7
    measured the indicators worth more macro MAE than every ligand descriptor
    combined, because two of them encode publication reporting conventions.
    Dropping them to "simplify" gen9 would move the baseline.
    """
    from sklearn.impute import SimpleImputer

    from ..levels import _as_float_frame

    columns = cohort.block_columns(tuple(blocks))
    x_train = _as_float_frame(train, columns).to_numpy()
    x_test = _as_float_frame(test, columns).to_numpy()
    keep = ~np.all(np.isnan(x_train), axis=0)
    x_train, x_test = x_train[:, keep], x_test[:, keep]
    imputer = SimpleImputer(strategy="median", add_indicator=add_indicator).fit(x_train)
    return imputer.transform(x_train), imputer.transform(x_test)


def load_membership(path: Path | None = None) -> pd.DataFrame:
    """gen9's own curve reconstruction, recomputed if it has not been cached."""
    target = Path(path) if path is not None else GEN9_MEMBERSHIP_PATH
    if target.exists():
        return pd.read_parquet(target)
    raise FileNotFoundError(
        f"{target} is missing — run `scripts/gen9_curves.py` first so the curve "
        "geometry gen9 trains on is the one gen9 recomputed and audited")


@dataclass
class CurveBoostContender:
    """gen7 ``Contender`` protocol, gen9 objective.

    ``membership`` is the whole-cohort curve table; the *restriction to the
    training partition happens inside* :meth:`fit_predict`, once, so there is a
    single place a leak could be introduced and a single place a test can watch.
    """

    arm: ShapeArm
    config: BoostConfig = field(default_factory=BoostConfig)
    membership: pd.DataFrame | None = None
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    name: str = ""
    #: Filled per fold so a caller can write the loss traces and pair audits out.
    diagnostics: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"GEN9_{self.arm.name}"
        if self.membership is None:
            self.membership = load_membership()

    def fit_predict(self, train: pd.DataFrame, y_train: np.ndarray,
                    test: pd.DataFrame, context: FoldContext) -> np.ndarray:
        x_train, x_test = build_design(context.cohort, train, test,
                                       blocks=self.blocks, add_indicator=self.add_indicator)
        weights = group_balanced_weights(train["ecfp_cluster"])
        train_row_ids = train["row_id"].astype(str).to_numpy()

        pairs: CurvePairs | None = None
        if self.arm.strategy != "ROW_ONLY":
            ligand_of_row = dict(zip(train_row_ids, train["extractant"].astype(str).to_numpy()))
            # The row loss's own weights, so ``balance="cluster"`` can put the two
            # terms of the objective on the same scale for every row.
            weight_of_row = dict(zip(train_row_ids, weights.astype(float)))
            pairs = build_pairs(
                self.membership, train_row_ids,
                strategy=self.arm.strategy,
                axes=AXIS_SETS[self.arm.axis_set],
                rng=np.random.default_rng(context.model_seed),
                ligand_of_row=ligand_of_row,
                weight_of_row=weight_of_row,
                balance=self.arm.balance,
            )
            # The only rows the pair builder was given are training rows, so this
            # cannot fail — which is exactly why it is asserted rather than assumed.
            held_out = set(test["row_id"].astype(str).to_numpy())
            if held_out & set(train_row_ids):
                raise AssertionError("train and test row_ids overlap")

        model = CurveBoost(config=self.config, lambda_delta=self.arm.lambda_delta,
                           lambda_span=self.arm.lambda_span,
                           random_state=context.model_seed)
        model.fit(x_train, y_train, sample_weight=weights, pairs=pairs)

        record = {
            "arm": self.arm.name, "split_seed": context.fold.seed, "fold": context.fold.fold,
            "model_seed": context.model_seed, "n_train": int(len(train)),
            "n_features": int(x_train.shape[1]),
            "pair_audit": pairs.audit if pairs is not None else {"strategy": "ROW_ONLY", "n_pairs": 0},
            "history": model.history.to_dict("records") if model.history is not None else [],
        }
        self.diagnostics.append(record)
        # Written back as ``extra__prediction_sd``, the column gen8's uncertainty
        # policies read.  Without it they degrade to random on gen9 arms and the
        # comparison table becomes ragged for a reason unrelated to the result.
        context.extras["prediction_sd"] = model.prediction_spread(x_test)
        return model.predict(x_test)


@dataclass
class FrozenExtraTrees:
    """``REC_ecfp_plus_recovered``, refitted through gen9's own design builder.

    Its number is the reproduction check of brief §1: if this does not land on
    gen7's 0.9807 the environment or the cohort has drifted and nothing downstream
    is comparable.
    """

    name: str = FROZEN_ARM_NAME
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    n_estimators: int = 400
    max_features: float = 0.30
    min_samples_leaf: int = 2

    #: Append the five relative-position columns to the design.  The simplest
    #: possible version of the gen9 finding, and the one that separates "the
    #: features were missing" from "the recomposition was needed": if the monolith
    #: recovers the shape once it is told where on its own curve a row sits, no
    #: architecture change is required at all.  EXPLORATORY.
    with_relative: bool = False
    membership: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.with_relative and self.name == FROZEN_ARM_NAME:
            self.name = "GEN9_REL_MONOLITH"

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        from sklearn.ensemble import ExtraTreesRegressor

        x_train, x_test = build_design(context.cohort, train, test,
                                       blocks=self.blocks, add_indicator=self.add_indicator)
        if self.with_relative:
            from .relative import relative_position_features

            relative = relative_position_features(
                pd.concat([train, test], ignore_index=True),
                self.membership if self.membership is not None else load_membership())
            relative = relative.set_index("row_id")[list(RELATIVE_COLUMNS)]
            x_train = np.hstack([x_train, relative.reindex(
                train["row_id"].astype(str).to_numpy()).to_numpy(dtype=float)])
            x_test = np.hstack([x_test, relative.reindex(
                test["row_id"].astype(str).to_numpy()).to_numpy(dtype=float)])
        model = ExtraTreesRegressor(
            n_estimators=self.n_estimators, max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf, random_state=context.model_seed, n_jobs=-1)
        model.fit(x_train, y_train, sample_weight=group_balanced_weights(train["ecfp_cluster"]))
        prediction = np.asarray(model.predict(x_test), dtype=float)
        spread = float(y_train.max() - y_train.min())
        return np.clip(prediction, y_train.min() - 0.5 * spread, y_train.max() + 0.5 * spread)


def write_diagnostics(contender: CurveBoostContender, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contender.diagnostics, indent=1, default=float))


@dataclass
class ShapeRecomposed:
    """Keep the monolith's level, replace its within-curve shape with a learned one.

    The deployable form of the relative-position finding, and structurally the same
    move gen8's post-hoc slope repair makes — **mean-preserving** — with the
    median-slope prior replaced by a model:

    1. fit the frozen arm on the training fold; call it ``base``;
    2. fit a second model of the same family on the **curve-centred** target
       ``y - mean(y over the curve)``, computed from training rows only, with the
       relative-position columns appended to the design;
    3. predict ``mean_over_curve(base) + shape(row)`` for every row that sits on a
       curve, and plain ``base`` for every row that does not.

    Step 3 is where the mean preservation lives: the curve's *level* is whatever the
    monolith said it was, averaged over the curve, so nothing about the level changes
    and none of the accuracy the monolith bought by drawing flat is spent.  Only the
    shape around that level is replaced.

    Nothing here reads a target of a held-out row.  The curve means in step 2 are
    training means; the curve mean in step 3 is a mean of *predictions*; the
    relative-position columns are functions of the condition list, which the gen7
    harness hands a contender by design and the gen8 protocol assumes the user
    supplies.  ``tests/test_gen9_curve_objective.py`` corrupts every held-out target
    and demands bit-identical output.

    **EXPLORATORY** — written after the pre-registered sweep was read.
    """

    membership: pd.DataFrame | None = None
    blocks: tuple[str, ...] = FROZEN_BLOCKS
    add_indicator: bool = True
    n_estimators: int = 400
    max_features: float = 0.30
    min_samples_leaf: int = 2
    axes: tuple[str, ...] = ()
    name: str = "GEN9_SHAPE_RECOMPOSED"
    diagnostics: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.membership is None:
            self.membership = load_membership()
        if not self.axes:
            from .curves import DEFAULT_AXES
            self.axes = DEFAULT_AXES

    def _forest(self, seed: int):
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=self.n_estimators, max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf, random_state=seed, n_jobs=-1)

    def fit_predict(self, train: pd.DataFrame, y_train: np.ndarray,
                    test: pd.DataFrame, context: FoldContext) -> np.ndarray:
        from .relative import curve_membership_map, relative_position_features

        x_train, x_test = build_design(context.cohort, train, test,
                                       blocks=self.blocks, add_indicator=self.add_indicator)
        weights = group_balanced_weights(train["ecfp_cluster"])

        # --- 1. the monolith, unchanged --------------------------------------
        base = self._forest(context.model_seed).fit(x_train, y_train, sample_weight=weights)
        spread = float(y_train.max() - y_train.min())
        base_test = np.clip(base.predict(x_test), y_train.min() - 0.5 * spread,
                            y_train.max() + 0.5 * spread)

        # --- 2. the shape model, on curve-centred targets ---------------------
        relative = relative_position_features(
            pd.concat([train, test], ignore_index=True), self.membership, axes=self.axes)
        relative = relative.set_index("row_id")[list(RELATIVE_COLUMNS)]
        rel_train = relative.reindex(train["row_id"].astype(str).to_numpy()).to_numpy(dtype=float)
        rel_test = relative.reindex(test["row_id"].astype(str).to_numpy()).to_numpy(dtype=float)

        curve_train = curve_membership_map(train, self.membership, axes=self.axes).to_numpy()
        block = pd.DataFrame({"curve": curve_train, "y": y_train})
        on_curve = pd.notna(block["curve"]).to_numpy()
        means = block[on_curve].groupby("curve")["y"].mean()
        centred = np.where(on_curve,
                           block["y"].to_numpy() - block["curve"].map(means).to_numpy(dtype=float),
                           np.nan)
        usable = np.isfinite(centred)
        shape = self._forest(context.model_seed + 1).fit(
            np.hstack([x_train, rel_train])[usable], centred[usable],
            sample_weight=weights[usable])
        shape_test = shape.predict(np.hstack([x_test, rel_test]))

        # --- 3. recompose, preserving each curve's predicted level -------------
        curve_test = curve_membership_map(test, self.membership, axes=self.axes).to_numpy()
        out = base_test.copy()
        frame = pd.DataFrame({"curve": curve_test, "base": base_test, "shape": shape_test})
        for curve_id, rows in frame[pd.notna(frame["curve"])].groupby("curve").groups.items():
            index = np.asarray(rows, dtype=int)
            level = float(frame["base"].to_numpy()[index].mean())
            centred_prediction = frame["shape"].to_numpy()[index]
            out[index] = level + centred_prediction - centred_prediction.mean()

        self.diagnostics.append({
            "arm": self.name, "split_seed": context.fold.seed, "fold": context.fold.fold,
            "n_train": int(len(train)), "n_shape_train": int(usable.sum()),
            "n_test_on_curve": int(pd.notna(frame["curve"]).sum()),
            "n_test": int(len(test)),
        })
        return np.clip(out, y_train.min() - 0.5 * spread, y_train.max() + 0.5 * spread)
