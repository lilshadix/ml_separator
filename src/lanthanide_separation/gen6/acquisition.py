"""Experiment F — which ligand should have been measured next?

Phase 1 established that the next measurement dollar should buy breadth, not
depth.  This module asks the follow-up: *which* distant ligand.  It replays the
project's history as a sequential decision: start from a deliberately narrow,
deep cohort (the ten most-measured ligands of one chemotype — the situation this
project was in for five generations), treat every other training ligand as an
unlabelled pool, and let an acquisition rule choose the next ligand, revealing
only a few of its measurements each time.  A fixed, untouched test set of
held-out chemotypes scores every step.

Every policy is **label-free for the candidate** by construction: a policy's
score function receives the acquired data, the pool's *feature* rows with the
target column removed, and a similarity oracle — nothing else.  The test suite
proves it by blanking the pool's ``log_D`` and asserting that every ordering is
unchanged.

Policies:

``random``                    uniform over the pool (the null every claim is against)
``maxmin``                    farthest from everything acquired, by min Tanimoto
``uncertainty``               largest mean across-tree sd of the current forest on
                              the candidate's rows
``diversity_x_uncertainty``   rank product of the two above
``offset_uncertainty``        largest across-tree sd of the candidate's *predicted
                              ligand mean* — uncertainty about the level, which is
                              the quantity Phase 1 says is missing
``same_chemotype_first``      closest to the acquired set — "another DGA
                              analogue", what the field did

The model is the gen5 ``LevelRegressor`` (ExtraTrees); per-tree predictions come
straight from the fitted forest's estimators, run through the same fold-local
imputer, so the uncertainty is the model's own, not a proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..levels import LEVEL_TARGET_COLUMN, LevelData, LevelForestParameters, LevelRegressor
from .chemistry import ChemistryMap

POLICIES: tuple[str, ...] = (
    "random", "maxmin", "uncertainty", "diversity_x_uncertainty", "offset_uncertainty",
    "same_chemotype_first",
)
#: Policies that need the current model to score the pool (refit every step).
MODEL_POLICIES: frozenset[str] = frozenset({"uncertainty", "diversity_x_uncertainty", "offset_uncertainty"})
DEFAULT_CHECKPOINTS: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 16, 20, 25, 30)


# --------------------------------------------------------------------------- #
# Start cohort and pool
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StartCohort:
    extractants: tuple[str, ...]
    chemotype: str
    row_index: np.ndarray          # positions into the shared frame
    pool_extractants: tuple[str, ...]
    audit: dict


def start_cohort(
    frame: pd.DataFrame, train_index: np.ndarray, *, n_ligands: int = 10,
    group_column: str = "tanimoto_cluster",
) -> StartCohort:
    """The ``n_ligands`` most-measured training ligands of the largest training chemotype.

    Narrow by construction.  When the largest chemotype has fewer ligands than
    ``n_ligands`` (the fold that holds out the diglycolamides), the start is
    smaller and the audit says so; it is not padded from other chemotypes, because
    that would make the start broad.
    """
    train = frame.iloc[train_index]
    chemotype = str(train[group_column].value_counts().index[0])
    block = train[train[group_column].astype(str) == chemotype]
    ranked = block["extractant"].astype(str).value_counts()
    chosen = tuple(ranked.index[:n_ligands])
    mask = train["extractant"].astype(str).isin(chosen).to_numpy()
    pool = tuple(sorted(set(train["extractant"].astype(str)) - set(chosen)))
    return StartCohort(
        extractants=chosen, chemotype=chemotype, row_index=train_index[mask], pool_extractants=pool,
        audit={"n_start_ligands": len(chosen), "n_start_rows": int(mask.sum()),
               "start_chemotype": chemotype, "largest_chemotype_ligands": int(len(ranked)),
               "n_pool_ligands": len(pool),
               "n_pool_rows": int(len(train) - mask.sum()),
               "start_is_short": bool(len(chosen) < n_ligands)})


def reveal_rows(
    frame: pd.DataFrame, train_index: np.ndarray, extractant: str, *, budget: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """The rows of ``extractant`` that an acquisition reveals: ``budget`` at random
    (seeded) or all of them."""
    rows = train_index[frame["extractant"].astype(str).to_numpy()[train_index] == str(extractant)]
    if budget is None or budget >= len(rows):
        return np.sort(rows)
    return np.sort(rng.choice(rows, size=int(budget), replace=False))


# --------------------------------------------------------------------------- #
# Model-side uncertainty
# --------------------------------------------------------------------------- #

def per_tree_predictions(model: LevelRegressor, frame: pd.DataFrame) -> np.ndarray:
    """(n_trees, n_rows) predictions through the model's own fold-local pipeline."""
    if model.pipeline is None:
        raise RuntimeError("model has not been fitted")
    from ..levels import _as_float_frame
    x = _as_float_frame(frame, model.feature_columns)
    transformed = x
    for _, step in model.pipeline.steps[:-1]:
        transformed = step.transform(transformed)
    forest = model.pipeline.named_steps["model"]
    if not hasattr(forest, "estimators_"):
        raise TypeError("per-tree predictions need a forest learner")
    return np.vstack([tree.predict(transformed) for tree in forest.estimators_])


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #

@dataclass
class PolicyContext:
    """Everything a policy may look at.  Note what is *not* here: the pool's target."""

    acquired: tuple[str, ...]
    pool_features: Mapping[str, pd.DataFrame]       # extractant -> its rows, target column removed
    similarity: Callable[[str, Sequence[str]], np.ndarray]   # (candidate, reference) -> sims
    model: LevelRegressor | None
    rng: np.random.Generator


def _assert_label_free(context: PolicyContext) -> None:
    for name, block in context.pool_features.items():
        if LEVEL_TARGET_COLUMN in block.columns:
            raise ValueError(f"pool features for {name!r} carry the target column; a policy may not see it")


def score_random(candidates: Sequence[str], context: PolicyContext) -> dict[str, float]:
    return {c: float(context.rng.random()) for c in candidates}


def score_maxmin(candidates: Sequence[str], context: PolicyContext) -> dict[str, float]:
    """Distance to the nearest acquired ligand: larger = more novel."""
    reference = list(context.acquired)
    out = {}
    for c in candidates:
        sims = context.similarity(c, reference)
        out[c] = float(1.0 - sims.max()) if len(sims) else 1.0
    return out


def score_same_chemotype_first(candidates: Sequence[str], context: PolicyContext) -> dict[str, float]:
    """The opposite of max-min: prefer what is closest to what is already held."""
    return {c: -v for c, v in score_maxmin(candidates, context).items()}


def _pool_tree_predictions(
    candidates: Sequence[str], context: PolicyContext,
) -> dict[str, np.ndarray]:
    """(n_trees, n_rows) per candidate, computed in ONE pass over the pool.

    Running the forest once per ligand was 10x slower for the same answer; the
    pool's rows are stacked, pushed through the pipeline once, and split back.
    """
    if context.model is None:
        raise ValueError("uncertainty policies need a fitted model in the context")
    blocks = [context.pool_features[c] for c in candidates]
    sizes = [len(b) for b in blocks]
    if not blocks:
        return {}
    stacked = pd.concat(blocks, axis=0, ignore_index=True)
    trees = per_tree_predictions(context.model, stacked)        # (n_trees, total_rows)
    out: dict[str, np.ndarray] = {}
    offset = 0
    for name, size in zip(candidates, sizes):
        out[name] = trees[:, offset:offset + size]
        offset += size
    return out


def score_uncertainty(candidates: Sequence[str], context: PolicyContext) -> dict[str, float]:
    """Mean across-tree sd over the candidate's rows under the current model."""
    trees = _pool_tree_predictions(candidates, context)
    return {c: float(trees[c].std(axis=0).mean()) for c in candidates}


def score_offset_uncertainty(candidates: Sequence[str], context: PolicyContext) -> dict[str, float]:
    """Across-tree sd of the candidate's *predicted ligand mean*: how unsure the
    forest is about the level of this ligand, averaged over its rows."""
    trees = _pool_tree_predictions(candidates, context)
    return {c: float(trees[c].mean(axis=1).std()) for c in candidates}


def score_diversity_x_uncertainty(candidates: Sequence[str], context: PolicyContext) -> dict[str, float]:
    """Rank product of novelty and uncertainty (ranks so scales do not matter)."""
    novelty = score_maxmin(candidates, context)
    uncertain = score_uncertainty(candidates, context)
    order_n = pd.Series(novelty).rank(pct=True)
    order_u = pd.Series(uncertain).rank(pct=True)
    return {c: float(order_n[c] * order_u[c]) for c in candidates}


SCORERS: dict[str, Callable[[Sequence[str], PolicyContext], dict[str, float]]] = {
    "random": score_random,
    "maxmin": score_maxmin,
    "uncertainty": score_uncertainty,
    "diversity_x_uncertainty": score_diversity_x_uncertainty,
    "offset_uncertainty": score_offset_uncertainty,
    "same_chemotype_first": score_same_chemotype_first,
}


def choose_next(policy: str, candidates: Sequence[str], context: PolicyContext) -> str:
    """Argmax of the policy's score with a seeded, deterministic tie-break."""
    if policy not in SCORERS:
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
    _assert_label_free(context)
    scores = SCORERS[policy](list(candidates), context)
    ordered = sorted(candidates)
    best = max(ordered, key=lambda c: (scores[c], -ordered.index(c)))
    # break exact ties randomly but reproducibly
    tied = [c for c in ordered if scores[c] == scores[best]]
    if len(tied) > 1:
        best = tied[int(context.rng.integers(len(tied)))]
    return best


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #

@dataclass
class AcquisitionTrace:
    policy: str
    steps: pd.DataFrame          # one row per acquisition: step, extractant, n_rows_revealed, ...
    checkpoints: pd.DataFrame    # one row per checkpoint: metrics on the fixed test set
    predictions: dict[int, np.ndarray]   # checkpoint -> test predictions
    audit: dict = field(default_factory=dict)


def run_acquisition(
    frame: pd.DataFrame,
    data: LevelData,
    *,
    start: StartCohort,
    train_index: np.ndarray,
    test_index: np.ndarray,
    policy: str,
    feature_columns: Sequence[str],
    params: LevelForestParameters,
    chemistry: ChemistryMap,
    budget_per_ligand: int | None = 3,
    n_steps: int = 30,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
    seed: int = 0,
    evaluate: Callable[[np.ndarray, np.ndarray, np.ndarray], dict] | None = None,
    group_weighting: bool = True,
) -> AcquisitionTrace:
    """Replay one acquisition sequence and score the fixed test set at checkpoints.

    ``evaluate(prediction, train_rows, acquired_ligands)`` returns the metric dict
    for one checkpoint; it is injected so the runner owns the metric definitions.
    The model is refit after every acquisition for policies that score with it,
    and only at checkpoints otherwise — the checkpoint evaluation always uses a
    model fitted on exactly the rows revealed so far.
    """
    rng = np.random.default_rng(seed)
    extractant_of_row = frame["extractant"].astype(str).to_numpy()
    pool = list(start.pool_extractants)
    acquired: list[str] = list(start.extractants)
    revealed = np.array(start.row_index, dtype=int)
    test = frame.iloc[test_index]
    y = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)

    # pool features WITHOUT the target: the structural guarantee of label-freeness
    pool_rows = {name: train_index[extractant_of_row[train_index] == name] for name in pool}
    pool_features = {name: frame.iloc[rows].drop(columns=[LEVEL_TARGET_COLUMN])
                     for name, rows in pool_rows.items()}

    def similarity(candidate: str, reference: Sequence[str]) -> np.ndarray:
        if not reference:
            return np.zeros(0)
        return chemistry.similarity_between([candidate], list(reference))[0]

    def fit_current() -> LevelRegressor:
        # ``group_weighting`` is the gen5 rule (equal total weight per ECFP cluster).
        # Under it a newly acquired 3-row ligand is a new cluster with a full vote, so
        # after 30 acquisitions the 90 new rows carry ~83 % of the training weight —
        # which is why the study must be rerun with row weighting as a sensitivity:
        # the comparison BETWEEN policies shares the rule, the effect SIZE does not.
        train = frame.iloc[revealed]
        return LevelRegressor(feature_columns, params).fit(
            train, y[revealed], groups=train["ecfp_cluster"] if group_weighting else None)

    needs_model = policy in MODEL_POLICIES
    step_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    predictions: dict[int, np.ndarray] = {}
    checkpoint_set = {int(c) for c in checkpoints}

    def record_checkpoint(step: int, model: LevelRegressor) -> None:
        pred = model.predict(test)
        predictions[step] = pred
        record = {"step": step, "n_train_rows": int(len(revealed)), "n_acquired": step,
                  "n_train_ligands": len(acquired)}
        if evaluate is not None:
            record.update(evaluate(pred, revealed, np.array(acquired)))
        checkpoint_rows.append(record)

    # Checkpoint 0 is the start cohort alone.  The model fitted here is also what
    # a model-based policy uses to choose step 1.
    model = fit_current()
    record_checkpoint(0, model)

    for step in range(1, n_steps + 1):
        if not pool:
            break
        context = PolicyContext(acquired=tuple(acquired), pool_features=pool_features,
                                similarity=similarity, model=model if needs_model else None,
                                rng=rng)
        chosen = choose_next(policy, pool, context)
        new_rows = reveal_rows(frame, train_index, chosen, budget=budget_per_ligand, rng=rng)
        nn_to_previous = similarity(chosen, acquired)
        revealed = np.concatenate([revealed, new_rows])
        acquired.append(chosen)
        pool.remove(chosen)
        pool_features.pop(chosen, None)
        step_rows.append({"step": step, "extractant": chosen, "n_rows_revealed": int(len(new_rows)),
                          "n_train_rows": int(len(revealed)),
                          "nn_to_acquired_before": float(nn_to_previous.max()) if len(nn_to_previous) else np.nan,
                          "chemotype": str(frame["tanimoto_cluster"].iloc[new_rows[0]]) if len(new_rows) else ""})
        # Model-based policies refit after every acquisition (the refit model
        # chooses the next step); the others refit only when a checkpoint needs
        # scoring.  Either way a checkpoint is scored by a model fitted on exactly
        # the rows revealed so far.
        if needs_model or step in checkpoint_set:
            model = fit_current()
        if step in checkpoint_set:
            record_checkpoint(step, model)

    return AcquisitionTrace(
        policy=policy, steps=pd.DataFrame(step_rows), checkpoints=pd.DataFrame(checkpoint_rows),
        predictions=predictions,
        audit={"budget_per_ligand": budget_per_ligand, "n_steps_requested": n_steps,
               "n_steps_run": len(step_rows), "pool_exhausted": not pool, "seed": seed,
               **start.audit})
