"""Contenders that change the *target* or the *decomposition*, not the learner.

Three ideas from the brief live here, all of which reshape what the model is asked
to fit rather than what fits it:

:class:`BatchDeconfounded` (brief §11)
    Publication identity explains 42 % of ``log_D`` variance on its own and the
    finer (DOI, table/figure) batch explains 64 % — but a *new* ligand has no DOI,
    so a batch feature is worthless at inference.  The useful move is the opposite
    one: estimate the batch offsets on the **training** side and subtract them, so
    the model learns chemistry from a target that has had the study-to-study
    calibration taken out of it.  The offsets are shrunk toward zero by
    ``n/(n+λ)``, the standard random-intercept estimator, so a two-row batch cannot
    invent a large correction.  Nothing about the test rows is used, and the
    prediction is deliberately *not* re-offset — there is no batch to re-offset it
    to.

:class:`TwoStageCellMean` (brief §3, and the gen6 trap it repairs)
    gen6 measured that a Stage-B model on the *cross-fitted residual* is worse than
    the monolith (1.25 vs 1.08) because Stage B ends up learning Stage A's error,
    while centring on the true cell mean instead reaches 1.04.  This implements the
    working variant: Stage A predicts the (ligand, series) level; Stage B predicts
    the deviation from the **observed** level on the training side, never from
    Stage A's prediction of it.  At test time the two are added, which is where the
    level error enters — and that is the honest accounting.

:class:`OffsetShapeTree`
    The same decomposition as the hierarchical network but with trees: one model
    for the per-ligand level fitted on **one row per ligand** (so a ligand with
    1,500 rows does not outvote one with three), and one for the within-ligand
    centred response.  It isolates "does the decomposition help?" from "does a
    neural network help?", which a single hierarchical-net result cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..levels import LEVEL_TARGET_COLUMN, _as_float_frame, group_balanced_weights
from .contenders import CHAMPION_BLOCKS, extratrees
from .harness import FoldContext


def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    x = _as_float_frame(frame, columns).to_numpy()
    return x


def _prepare(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = ~np.all(np.isnan(train_x), axis=0)
    train_x, test_x = train_x[:, keep], test_x[:, keep]
    median = np.nan_to_num(np.nanmedian(train_x, axis=0))
    return (np.where(np.isfinite(train_x), train_x, median),
            np.where(np.isfinite(test_x), test_x, median))


@dataclass
class BatchDeconfounded:
    """Subtract shrunken per-batch intercepts from the training target."""

    blocks: tuple[str, ...] = CHAMPION_BLOCKS
    estimator: Callable[[int], object] = extratrees
    shrinkage: float = 5.0
    batch_column: str = "nuisance__batch"
    name: str = "DECONF_batch"
    weighting: str = "cluster"

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        columns = context.cohort.block_columns(self.blocks)
        x_train, x_test = _prepare(_matrix(train, columns), _matrix(test, columns))
        y = np.asarray(y_train, dtype=float)

        if self.batch_column in train.columns:
            # Centre within LIGAND first: a batch offset is only identifiable
            # relative to the other batches of the same ligand.  A batch that is a
            # ligand's only batch therefore gets an offset of exactly zero, which
            # is correct — there is nothing to compare it with.
            ligand = train["extractant"].astype(str).to_numpy()
            batch = train[self.batch_column].astype(str).to_numpy()
            centred = y - pd.Series(y).groupby(ligand).transform("mean").to_numpy()
            frame = pd.DataFrame({"batch": batch, "ligand": ligand, "r": centred})
            stats = frame.groupby(["ligand", "batch"])["r"].agg(["mean", "size"])
            shrunk = stats["mean"] * stats["size"] / (stats["size"] + self.shrinkage)
            key = pd.MultiIndex.from_arrays([ligand, batch])
            offsets = shrunk.reindex(key).to_numpy()
            offsets = np.where(np.isfinite(offsets), offsets, 0.0)
            y_fit = y - offsets
            context.extras["deconfound_offset_sd"] = np.full(len(test), float(np.std(offsets)))
        else:
            y_fit = y

        model = self.estimator(context.model_seed)
        weights = (group_balanced_weights(train["ecfp_cluster"])
                   if self.weighting == "cluster" else None)
        model.fit(x_train, y_fit, sample_weight=weights)
        prediction = np.asarray(model.predict(x_test), dtype=float)
        span = float(y.max() - y.min())
        return np.clip(prediction, y.min() - 0.5 * span, y.max() + 0.5 * span)


@dataclass
class OffsetShapeTree:
    """Level head on one row per ligand; response head on the centred target."""

    level_blocks: tuple[str, ...] = ("DONORS", "PHYSCHEM", "LIG2D_EXT")
    response_blocks: tuple[str, ...] = ("METAL", "COND", "MASSACTION", "DONORS")
    estimator: Callable[[int], object] = extratrees
    name: str = "DECOMP_offset_shape_tree"
    level_unit: str = "ligand"          # ligand | ligand_series
    weighting: str = "cluster"

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        y = np.asarray(y_train, dtype=float)
        cohort = context.cohort
        level_columns = cohort.block_columns(
            tuple(b for b in self.level_blocks if b in cohort.blocks))
        response_columns = cohort.block_columns(
            tuple(b for b in self.response_blocks if b in cohort.blocks))

        ligand = train["extractant"].astype(str).to_numpy()
        level = pd.Series(y).groupby(ligand).transform("mean").to_numpy()

        # --- level head: ONE row per training ligand ------------------------- #
        first = pd.DataFrame({"ligand": ligand}).drop_duplicates().index.to_numpy()
        level_train = train.iloc[first]
        level_target = pd.Series(y).groupby(ligand).mean().reindex(
            level_train["extractant"].astype(str)).to_numpy()
        lx_train, lx_test = _prepare(_matrix(level_train, level_columns),
                                     _matrix(test, level_columns))
        level_model = self.estimator(context.model_seed)
        level_model.fit(lx_train, level_target)
        level_prediction = np.asarray(level_model.predict(lx_test), dtype=float)

        # --- response head: the OBSERVED centred target, never Stage A's error - #
        rx_train, rx_test = _prepare(_matrix(train, response_columns),
                                     _matrix(test, response_columns))
        response_model = self.estimator(context.model_seed + 1)
        weights = (group_balanced_weights(train["ecfp_cluster"])
                   if self.weighting == "cluster" else None)
        response_model.fit(rx_train, y - level, sample_weight=weights)
        response_prediction = np.asarray(response_model.predict(rx_test), dtype=float)

        context.extras["offset_hat"] = level_prediction
        prediction = level_prediction + response_prediction
        span = float(y.max() - y.min())
        return np.clip(prediction, y.min() - 0.5 * span, y.max() + 0.5 * span)


@dataclass
class PairMeanNull:
    """The no-model null gen5's k-shot study established: predict the global mean.

    Kept as a contender rather than a footnote because several generations of this
    project have reported R² values that sit *below* this line, and a leaderboard
    that does not carry its own null invites that mistake again.
    """

    name: str = "NULL_pair_mean"

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        return np.full(len(test), float(np.median(np.asarray(y_train, dtype=float))))


@dataclass
class InnerSelected:
    """Choose among candidate arms on an inner chemotype split — never on test rows.

    The gen7 learner sweep found ``MC_ecfp_massaction`` (macro 0.998) ahead of the
    arm gen6 pre-registered as champion (``MC_lig2d_ext_massaction``, 1.047).
    Quoting the winner would be selection on the outer test set: three seeds of the
    same 79 chemotypes chose it, and the margin between the top arms (0.01–0.05) is
    the same size as their seed spread (0.02–0.04).

    So the selection is made a *part of the model*.  For each outer fold, the
    training rows are split again by whole Tanimoto chemotypes; every candidate is
    fitted on the inner-training part and scored on the inner-held-out part with the
    same macro rule (one ECFP cluster, one vote); the winner is refitted on the full
    outer training set and applied to the outer test rows.  The outer test rows take
    no part in the choice, so the resulting number is honest in a way a hand-picked
    arm is not — and it is also what a deployment would actually do.

    ``n_inner`` inner repeats are averaged before choosing, because a single inner
    split of 79 chemotypes is itself noisy.
    """

    candidates: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("ecfp_massaction", ("METAL", "COND", "ECFP", "MASSACTION")),
        ("lig2d_massaction", ("METAL", "COND", "LIG2D_EXT", "MASSACTION")),
        ("donors", ("METAL", "COND", "DONORS")),
        ("donors_massaction", ("METAL", "COND", "DONORS", "MASSACTION")),
        ("metalphys_ecfp", ("METALPHYS", "COND", "ECFP", "MASSACTION")),
        ("everything", ("METAL", "METALPHYS", "COND", "ECFP", "LIG2D_EXT", "DONORS",
                        "PHYSCHEM", "LIGPHYS", "MASSACTION", "RECOVERED")),
    )
    estimator: Callable[[int], object] = extratrees
    #: Two inner repeats, not three.  Seven candidates x 3 repeats x 5 folds x 5 seeds
    #: is 525 model fits, several of them over 2,048 fingerprint columns, and the
    #: selector was taking longer than every other contender combined.  Two repeats
    #: still average away most of the inner-split noise; the cost is disclosed here
    #: rather than buried.
    n_inner: int = 2
    add_indicator: bool = True
    name: str = "SELECT_inner"
    weighting: str = "cluster"

    def _fit_one(self, blocks, train, y, test, cohort, seed):
        from sklearn.impute import SimpleImputer
        columns = cohort.block_columns(tuple(b for b in blocks if b in cohort.blocks))
        x_train, x_test = _prepare(_matrix(train, columns), _matrix(test, columns))
        if self.add_indicator:
            imputer = SimpleImputer(strategy="median", add_indicator=True).fit(
                _matrix(train, columns))
            x_train = imputer.transform(_matrix(train, columns))
            x_test = imputer.transform(_matrix(test, columns))
        model = self.estimator(seed)
        weights = (group_balanced_weights(train["ecfp_cluster"])
                   if self.weighting == "cluster" else None)
        model.fit(x_train, y, sample_weight=weights)
        return np.asarray(model.predict(x_test), dtype=float)

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        cohort = context.cohort
        y = np.asarray(y_train, dtype=float)
        chemotypes = train["tanimoto_cluster"].astype(str).to_numpy()
        unique = np.unique(chemotypes)
        scores = {name: [] for name, _ in self.candidates}

        for repeat in range(self.n_inner):
            rng = np.random.default_rng(context.model_seed + 31 * repeat)
            held = set(rng.permutation(unique)[: max(1, len(unique) // 5)].tolist())
            valid = np.array([g in held for g in chemotypes])
            fit = ~valid
            if not valid.any() or fit.sum() < 30:
                continue
            inner_train, inner_valid = train.iloc[fit], train.iloc[valid]
            truth = y[valid]
            clusters = inner_valid["ecfp_cluster"].astype(str).to_numpy()
            for name, blocks in self.candidates:
                try:
                    prediction = self._fit_one(blocks, inner_train, y[fit], inner_valid,
                                               cohort, context.model_seed + repeat)
                except Exception:
                    continue
                error = pd.Series(np.abs(prediction - truth)).groupby(clusters).mean()
                scores[name].append(float(error.mean()))

        ranked = [(float(np.mean(v)), k) for k, v in scores.items() if v]
        if not ranked:
            chosen = self.candidates[0][0]
        else:
            ranked.sort()
            chosen = ranked[0][1]
        blocks = dict(self.candidates)[chosen]
        context.extras["selected_arm_index"] = np.full(
            len(test), float([n for n, _ in self.candidates].index(chosen)))
        prediction = self._fit_one(blocks, train, y, test, cohort, context.model_seed)
        span = float(y.max() - y.min())
        return np.clip(prediction, y.min() - 0.5 * span, y.max() + 0.5 * span)


@dataclass
class Oracle:
    """Arms that are *given* something no deployed model could know — to bound headroom.

    These are not contenders.  They exist because "how much is left to win?" is a
    question about the data, and the only honest way to answer it is to hand a model
    the answer and see how much it still gets wrong.

    ``kind="level"``
        The test ligand's **true mean** ``log_D`` (computed from its own held-out
        rows) replaces the predicted level; the response is still learned from the
        training rows.  Its ``offset_mae`` is zero by construction, so what remains
        is pure shape error.  The macro MAE of this arm is the floor that any
        perfect level model would reach, and the gap from the real model to it is
        the entire prize on offer for better chemistry.

    ``kind="batch"``
        The test row's ``(DOI, table/figure)`` batch mean residual is given.  A
        deployed model has no DOI, so this is unattainable — but the difference
        between it and the level oracle separates "level we could learn from
        chemistry" from "study-to-study calibration we could not".

    ``kind="cell"``
        The true mean of the row's own ``(ligand, condition, metal)`` cell.  Since
        the cohort is already replicate-averaged this is effectively perfect
        prediction, and it exists as the sanity check that the harness scores an
        oracle at ~0.

    Every oracle is labelled ``ORACLE_*`` and must never appear in a leaderboard
    without that prefix.
    """

    kind: str = "level"
    blocks: tuple[str, ...] = ("METAL", "COND", "MASSACTION", "ECFP")
    estimator: Callable[[int], object] = extratrees
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"ORACLE_{self.kind}"

    def fit_predict(self, train, y_train, test, context: FoldContext) -> np.ndarray:
        cohort = context.cohort
        y = np.asarray(y_train, dtype=float)
        frame = cohort.frame
        truth = frame[LEVEL_TARGET_COLUMN].to_numpy(float)
        test_index = context.fold.test_index

        if self.kind == "cell":
            return truth[test_index]

        columns = cohort.block_columns(tuple(b for b in self.blocks if b in cohort.blocks))
        x_train, x_test = _prepare(_matrix(train, columns), _matrix(test, columns))
        ligand = train["extractant"].astype(str).to_numpy()
        level = pd.Series(y).groupby(ligand).transform("mean").to_numpy()
        model = self.estimator(context.model_seed)
        model.fit(x_train, y - level,
                  sample_weight=group_balanced_weights(train["ecfp_cluster"]))
        response = np.asarray(model.predict(x_test), dtype=float)

        test_ligand = frame["extractant"].astype(str).to_numpy()[test_index]
        if self.kind == "level":
            given = pd.Series(truth[test_index]).groupby(test_ligand).transform("mean").to_numpy()
        elif self.kind == "batch":
            if "nuisance__batch" not in frame.columns:
                raise KeyError("batch oracle needs nuisance__batch")
            key = (pd.Series(test_ligand) + "||"
                   + pd.Series(frame["nuisance__batch"].astype(str).to_numpy()[test_index]))
            given = pd.Series(truth[test_index]).groupby(key.to_numpy()).transform("mean").to_numpy()
        else:
            raise ValueError(f"unknown oracle kind {self.kind!r}")

        # the oracle supplies the LEVEL; the response is still the model's own, and
        # its own mean over the ligand is removed so the two cannot double-count
        centred = response - pd.Series(response).groupby(test_ligand).transform("mean").to_numpy() \
            if self.kind == "level" else response - pd.Series(response).groupby(
                key.to_numpy()).transform("mean").to_numpy()
        return given + centred
