"""The gen8 evaluation driver: policies x adapters x k, on one frozen fold plan.

One function owns the whole contract so that no experiment can quietly relax it:

* the fold plan is gen5's ``seeded_group_kfold`` over the Tanimoto chemotype,
  reused byte-for-byte, so a held-out ligand's *entire chemotype* is unseen;
* a trainable adapter is fitted on that fold's training rows and nothing else;
* within a held-out ligand, each repeat splits the rows once into a candidate
  **pool** and a disjoint **evaluation set**, and *every* policy and *every*
  adapter sees the identical pool and is scored on the identical evaluation rows;
* an adapter is handed the targets of the selected rows only.

The output is one long frame — (model, seed, fold, ligand, repeat, policy,
adapter, k) -> mae plus the level/shape decomposition — which every table and
every bootstrap in the study is aggregated from.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .adapters import AdaptContext, is_trainable
from .kshot import NON_DEPLOYABLE, POLICIES, PolicyContext, make_oracle_policy, stable_hash
from .protocols import K_VALUES, POOL_CAP, _standardised_axes, make_p2_split


#: Row cap for the O(n^2) within-ligand sign accuracy.  One ligand carries 1,488
#: rows and the metric is recomputed for every (policy, adapter, k, repeat); the
#: subsample is deterministic in the row order, so the number is stable.
SIGN_ACCURACY_CAP = 120


def _sign_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Share of within-ligand row pairs ordered correctly.  Ties in the truth drop out.

    A pair tied in the *prediction* counts as one half, the usual convention, so a
    constant predictor scores 0.5 rather than 0 — which matters here because a
    perfectly calibrated level with no shape at all is exactly a constant.
    """
    n = len(truth)
    if n < 2:
        return float("nan")
    if n > SIGN_ACCURACY_CAP:
        take = np.linspace(0, n - 1, SIGN_ACCURACY_CAP).astype(int)
        truth, prediction = truth[take], prediction[take]
    dy = np.subtract.outer(truth, truth)
    dp = np.subtract.outer(prediction, prediction)
    iu = np.triu_indices(len(truth), k=1)
    dy, dp = dy[iu], dp[iu]
    orderable = dy != 0
    if not orderable.any():
        return float("nan")
    dy, dp = dy[orderable], dp[orderable]
    ties = dp == 0
    agree = np.sign(dy) == np.sign(dp)
    return float((agree & ~ties).sum() + 0.5 * ties.sum()) / float(orderable.sum())


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """MAE plus the gen6 level/shape split, on one ligand's evaluation rows."""
    residual = prediction - truth
    offset = float(residual.mean())
    centred = residual - offset
    out = {"mae": float(np.abs(residual).mean()),
           "offset": abs(offset),
           "shape_mae": float(np.abs(centred).mean())}
    if len(truth) >= 3 and np.ptp(truth) > 0 and np.ptp(prediction) > 0:
        from scipy.stats import spearmanr
        out["spearman"] = float(spearmanr(truth, prediction).statistic)
    else:
        out["spearman"] = float("nan")
    out["sign_accuracy"] = _sign_accuracy(truth, prediction)
    out["within_0_5"] = float((np.abs(residual) <= 0.5).mean())
    out["within_1_0"] = float((np.abs(residual) <= 1.0).mean())
    return out


def evaluate_fewshot(
    oof: pd.DataFrame,
    cohort: pd.DataFrame,
    adapters: Sequence,
    *,
    policies: Sequence[str] = ("RANDOM", "CENTRAL"),
    with_oracle_for: Sequence[str] = (),
    k_values: Sequence[int] = K_VALUES,
    repeats: int = 12,
    seed: int = 20260820,
    min_rows: int = 4,
    pool_cap: int = POOL_CAP,
    fold_trainer: Callable[[int, int], tuple[pd.DataFrame, np.ndarray, int]] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every (policy, adapter, k) on every held-out ligand of every fold.

    ``oof`` must carry one model's out-of-fold predictions with ``row_id``,
    ``split_seed``, ``fold``, ``extractant`` and ``log_D``.  ``cohort`` supplies the
    feature columns; the join is on ``row_id`` and is asserted row-preserving.

    ``fold_trainer(split_seed, fold)`` returns ``(train_frame, y_train, model_seed)``
    for the trainable adapters.  It is required as soon as any adapter is trainable,
    and its absence with a trainable adapter present is an error rather than a
    silent zero-shot fallback.
    """
    feature_columns = [c for c in cohort.columns if c not in ("log_D",)]
    merged = oof.merge(cohort[feature_columns], on="row_id", how="left",
                       validate="many_to_one", suffixes=("", "__cohort"))
    assert len(merged) == len(oof), "cohort join changed the row count"

    trainable = [a for a in adapters if is_trainable(a)]
    if trainable and fold_trainer is None:
        raise ValueError(
            f"{[a.name for a in trainable]} need per-fold training data; pass fold_trainer")

    records: list[dict] = []
    for (split_seed, fold), fold_block in merged.groupby(["split_seed", "fold"], sort=True):
        started = time.time()
        if trainable:
            train, y_train, model_seed = fold_trainer(int(split_seed), int(fold))
            for adapter in trainable:
                adapter.fit_fold(train, y_train, split_seed=int(split_seed), fold=int(fold),
                                 model_seed=int(model_seed))
        for ligand, block in fold_block.groupby("extractant", sort=True):
            block = block.reset_index(drop=True)
            n = len(block)
            if n < min_rows:
                continue
            truth = block["log_D"].to_numpy(dtype=float)
            prediction = block["prediction"].to_numpy(dtype=float)
            axes = _standardised_axes(block)
            uncertainty = (block["extra__prediction_sd"].to_numpy(dtype=float)
                           if "extra__prediction_sd" in block.columns else np.full(n, np.nan))
            disagreement = (block["disagreement"].to_numpy(dtype=float)
                            if "disagreement" in block.columns else np.full(n, np.nan))
            context = AdaptContext(split_seed=int(split_seed), fold=int(fold), extractant=ligand)

            for repeat in range(repeats):
                rng = np.random.default_rng((seed, repeat, stable_hash(ligand)))
                split = make_p2_split(n, rng, pool_cap=pool_cap)
                if len(split.pool) < 1 or len(split.evaluation) < 2:
                    continue
                base = {"model": block["model"].iloc[0], "split_seed": int(split_seed),
                        "fold": int(fold), "extractant": ligand,
                        "tanimoto_cluster": block["tanimoto_cluster"].iloc[0],
                        "ecfp_cluster": block["ecfp_cluster"].iloc[0],
                        "nn_train_tanimoto": float(block["nn_train_tanimoto"].iloc[0]),
                        "repeat": repeat, "n_rows": n, "n_pool": int(len(split.pool)),
                        "n_eval": int(len(split.evaluation))}

                # zero-shot, once per repeat, independent of policy and k
                records.append({**base, "policy": "NONE", "adapter": "ZERO_SHOT_REF", "k": 0,
                                "deployable": True,
                                **_metrics(truth[split.evaluation], prediction[split.evaluation])})

                sequences: dict[str, list[int]] = {}
                for name in policies:
                    policy_context = PolicyContext(
                        block=block, prediction=prediction, pool=split.pool,
                        evaluation=split.evaluation, axes=axes, uncertainty=uncertainty,
                        disagreement=disagreement,
                        rng=np.random.default_rng((seed, repeat, stable_hash(name))),
                        truth=None)
                    chosen: list[int] = []
                    for _ in range(min(max(k_values), len(split.pool))):
                        chosen.append(int(POLICIES[name](policy_context, chosen)))
                    sequences[name] = chosen
                for adapter_name in with_oracle_for:
                    adapter = next(a for a in adapters if a.name == adapter_name)
                    policy_context = PolicyContext(
                        block=block, prediction=prediction, pool=split.pool,
                        evaluation=split.evaluation, axes=axes, uncertainty=uncertainty,
                        disagreement=disagreement,
                        rng=np.random.default_rng((seed, repeat, 7)), truth=truth)
                    chosen = []
                    for _ in range(min(max(k_values), len(split.pool))):
                        chosen.append(int(_oracle_pick(adapter, policy_context, chosen, context)))
                    sequences[f"ORACLE[{adapter_name}]"] = chosen

                for policy_name, sequence in sequences.items():
                    for k in k_values:
                        if k < 1 or k > len(sequence):
                            continue
                        selected = np.asarray(sequence[:k], dtype=int)
                        for adapter in adapters:
                            values = adapter.predict(block, prediction, selected,
                                                     truth[selected], context)
                            values = np.asarray(values, dtype=float)
                            if not np.isfinite(values[split.evaluation]).all():
                                continue
                            records.append({
                                **base, "policy": policy_name, "adapter": adapter.name, "k": k,
                                "deployable": not policy_name.startswith("ORACLE"),
                                **_metrics(truth[split.evaluation], values[split.evaluation])})
        if verbose:
            print(f"  seed {split_seed} fold {fold}: {time.time() - started:.1f}s "
                  f"({len(records):,} records)", flush=True)
    return pd.DataFrame(records)


def _oracle_pick(adapter, context: PolicyContext, selected: list[int],
                 adapt_context: AdaptContext) -> int:
    """Exhaustive best next point **for this adapter**.  Non-deployable by construction."""
    truth = context.truth
    assert truth is not None
    taken = set(selected)
    remaining = [i for i in context.pool if i not in taken]
    best, best_score = remaining[0], np.inf
    for candidate in remaining:
        chosen = np.asarray(selected + [int(candidate)], dtype=int)
        values = np.asarray(adapter.predict(context.block, context.prediction, chosen,
                                            truth[chosen], adapt_context), dtype=float)
        score = float(np.abs(values[context.evaluation] - truth[context.evaluation]).mean())
        if score < best_score:
            best, best_score = int(candidate), score
    return best


def summarise(detail: pd.DataFrame, keys: Sequence[str] = ("adapter", "policy", "k"),
              metric: str = "mae") -> pd.DataFrame:
    """One ligand, one vote — the macro convention every generation of this repo uses."""
    keys = list(keys)
    per_ligand = detail.groupby(keys + ["extractant"])[metric].mean().reset_index()
    out = per_ligand.groupby(keys).agg(**{metric: (metric, "mean"),
                                          "n_ligands": ("extractant", "nunique")})
    return out.reset_index().sort_values(metric)
