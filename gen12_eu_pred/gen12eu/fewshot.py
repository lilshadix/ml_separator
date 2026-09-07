"""Few-shot adaptation: what k measurements of a genuinely new extractant buy.

Everything here happens *after* a zero-shot model has emitted an out-of-fold
prediction for every row of a held-out extractant, so nothing can leak: the model
never saw the extractant, and the adapter reads only the targets of the rows it
was explicitly allowed to measure.

Three properties make the numbers comparable, and each is enforced rather than
assumed:

* **The support/query split is a pure function of ``(seed, repeat, extractant,
  n_rows)``**, through BLAKE2b.  Python's ``hash`` is salted per interpreter
  process; in gen8 that silently broke cross-run pairing while every table
  claimed the runs were paired.  A test re-derives a draw in a subprocess under a
  different ``PYTHONHASHSEED`` and demands the same answer.
* **A support row is never scored.**  Queries are the complement of the support
  set, always.
* **Every arm is handed byte-identical support and query sets**, because the draw
  depends only on the extractant and the repeat, never on the model.

Two adapters, and the nulls that make them meaningful.  Adapter A is a
shrinkage-corrected level offset — deliberately the simplest thing that could
work, so that "most of the few-shot gain is just learning the level" is a
testable statement rather than a slogan.  The shrinkage constant is estimated on
the *training* folds' out-of-fold residuals, never on the held-out extractant.
The required nulls are the repository's own: predicting the mean of what you
measured (``NO_MODEL``), and the corpus-level mean shifted by the same k points.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

TARGET = "log_D"
K_VALUES: tuple[int, ...] = (0, 1, 2, 3, 5)
DEFAULT_REPEATS = 20
#: Extractants with at least this many rows are scorable at every k in K_VALUES.
COMMON_COHORT_MIN_ROWS = 7


def stable_hash(text: str) -> int:
    """Process-independent 31-bit hash.  See the module docstring."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(),
                          "big") % (2 ** 31)


def support_draw(extractant: str, n_rows: int, k: int, repeat: int, seed: int) -> np.ndarray:
    """Positional indices of the k support rows.  Identical for every arm."""
    if k <= 0:
        return np.zeros(0, dtype=int)
    if k >= n_rows:
        raise ValueError(f"k={k} needs fewer than n_rows={n_rows}")
    rng = np.random.default_rng((int(seed), int(repeat), stable_hash(extractant), int(n_rows)))
    return np.sort(rng.choice(n_rows, size=k, replace=False))


def shrinkage_from_training(oof: pd.DataFrame) -> dict:
    """Estimate the level/noise variance ratio from *training-fold* residuals only.

    The James-Stein style estimate the adapter needs is ``lambda_k = k / (k +
    sigma^2 / tau^2)``, where ``tau^2`` is the between-extractant spread of the
    level error and ``sigma^2`` the within-extractant residual variance.  Both are
    measured on the out-of-fold residuals of extractants *other than* the one
    being adapted, which is what makes the shrinkage deployable.
    """
    residual = oof["prediction"] - oof[TARGET]
    per = residual.groupby([oof["split_seed"], oof["extractant"]])
    level = per.mean()
    within = per.var(ddof=1).dropna()
    tau2 = float(level.var(ddof=1))
    sigma2 = float(within.mean()) if len(within) else tau2
    ratio = sigma2 / tau2 if tau2 > 1e-9 else 1.0
    return {"tau2": tau2, "sigma2": sigma2, "ratio": float(ratio),
            "n_extractant_seeds": int(len(level))}


@dataclass
class OffsetAdapter:
    """Adapter A: shrinkage-corrected level offset, ``f_k(x) = f(x) + b_hat``."""

    name: str = "A_OFFSET_SHRUNK"
    ratio: float = 1.0
    trainable: bool = False

    def apply(self, prediction: np.ndarray, truth: np.ndarray, support: np.ndarray,
              query: np.ndarray) -> np.ndarray:
        if support.size == 0:
            return prediction[query]
        residual = float(np.mean(truth[support] - prediction[support]))
        weight = support.size / (support.size + self.ratio)
        return prediction[query] + weight * residual


@dataclass
class PlainOffsetAdapter:
    """Adapter A0: the unshrunk offset — the thing gen7 mistook for a degrees-of-freedom problem."""

    name: str = "A0_OFFSET_PLAIN"
    trainable: bool = False

    def apply(self, prediction, truth, support, query):
        if support.size == 0:
            return prediction[query]
        return prediction[query] + float(np.mean(truth[support] - prediction[support]))


@dataclass
class NoModelAdapter:
    """The null that killed gen5's k-shot story: predict the mean of what you measured."""

    name: str = "NULL_NO_MODEL"
    trainable: bool = False

    def apply(self, prediction, truth, support, query):
        if support.size == 0:
            return np.full(query.size, np.nan)
        return np.full(query.size, float(np.mean(truth[support])))


@dataclass
class GlobalMeanShiftedAdapter:
    """Retained only to *demonstrate* that it is not a second null.

    The pre-registration asked for two k-shot nulls: a no-model fit on the k
    points, and the corpus level shifted by the same k points.  They are the same
    estimator.  Writing the second one out,
    ``m + mean(y_S - m) = mean(y_S)``: the corpus level cancels exactly, so it
    reproduces ``NoModelAdapter`` to machine precision for every k, every draw and
    every extractant.  This was found by running both and seeing identical
    columns, and it is kept here as an executable note rather than deleted, with
    a test asserting the identity.  The second null's *role* — a chemistry-aware
    level that also gets the k measurements — is filled instead by applying the
    offset adapter to the nearest-training-chemistry level arm (``B3``).
    """

    name: str = "NULL_GLOBALMEAN_KSHOT"
    global_mean: float = 0.0
    trainable: bool = False

    def apply(self, prediction, truth, support, query):
        base = np.full(query.size, self.global_mean)
        if support.size == 0:
            return base
        return base + float(np.mean(truth[support] - self.global_mean))


def evaluate(oof: pd.DataFrame, adapters: Sequence, *, k_values: Sequence[int] = K_VALUES,
             repeats: int = DEFAULT_REPEATS, seed: int = 20260905,
             min_queries: int = 2) -> pd.DataFrame:
    """Run every (adapter, k, repeat) on every held-out extractant of every seed.

    Returns one row per (arm, adapter, k, split_seed, extractant, repeat) with the
    query MAE and the level/shape split of the query error, centred within that
    (seed, extractant) block only.
    """
    records: list[dict] = []
    for (arm, split_seed, extractant), block in oof.groupby(
            ["arm", "split_seed", "extractant"], sort=True):
        block = block.sort_values("row_id")
        truth = block[TARGET].to_numpy(dtype=float)
        prediction = block["prediction"].to_numpy(dtype=float)
        n = len(block)
        chemotype = str(block["chemotype"].iloc[0])
        band = str(block["band"].iloc[0]) if "band" in block.columns else ""
        for k in k_values:
            if k > 0 and n - k < min_queries:
                continue
            draws = 1 if k == 0 else repeats
            for repeat in range(draws):
                support = support_draw(str(extractant), n, k, repeat, split_seed)
                query = np.setdiff1d(np.arange(n), support)
                if query.size < min_queries:
                    continue
                assert not set(support) & set(query), "a support row was scored as a query"
                for adapter in adapters:
                    adapted = adapter.apply(prediction, truth, support, query)
                    residual = adapted - truth[query]
                    if not np.isfinite(residual).all():
                        continue
                    offset = float(residual.mean())
                    records.append({
                        "arm": arm, "adapter": adapter.name, "k": k, "split_seed": int(split_seed),
                        "extractant": str(extractant), "chemotype": chemotype, "band": band,
                        "repeat": repeat, "n_rows": n, "n_query": int(query.size),
                        "mae": float(np.abs(residual).mean()),
                        "offset_abs": abs(offset),
                        "shape_mae": float(np.abs(residual - offset).mean())
                        if query.size >= 2 else np.nan,
                        "support_rows": ";".join(block["row_id"].to_numpy()[support]),
                    })
    return pd.DataFrame(records)


def learning_curve(detail: pd.DataFrame, *, cohort: Sequence[str] | None = None) -> pd.DataFrame:
    """Macro MAE against k, averaged over repeats then extractants then seeds.

    Restricted to a fixed cohort when one is given.  A curve drawn on a cohort
    that shrinks with k is not a learning curve, so the common cohort is the
    default presentation and the per-k cohorts are reported beside it.
    """
    block = detail if cohort is None else detail[detail["extractant"].isin(set(cohort))]
    per_extractant = (block.groupby(["arm", "adapter", "k", "split_seed", "extractant"])
                      [["mae", "offset_abs", "shape_mae"]].mean().reset_index())
    per_seed = (per_extractant.groupby(["arm", "adapter", "k", "split_seed"])
                [["mae", "offset_abs", "shape_mae"]].mean().reset_index())
    curve = (per_seed.groupby(["arm", "adapter", "k"])
             [["mae", "offset_abs", "shape_mae"]].mean().reset_index())
    counts = (per_extractant.groupby(["arm", "adapter", "k"])
              .agg(n_extractants=("extractant", "nunique"),
                   n_seeds=("split_seed", "nunique")).reset_index())
    spread = (per_seed.groupby(["arm", "adapter", "k"])["mae"].std(ddof=0)
              .rename("mae_sd_over_seeds").reset_index())
    return curve.merge(counts, on=["arm", "adapter", "k"]).merge(
        spread, on=["arm", "adapter", "k"])


def common_cohort(oof: pd.DataFrame, *, min_rows: int = COMMON_COHORT_MIN_ROWS) -> list[str]:
    counts = oof.drop_duplicates(["extractant", "row_id"]).groupby("extractant").size()
    return sorted(counts[counts >= min_rows].index.astype(str))
