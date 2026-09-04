"""The two k-shot protocols, and the fairness rules that make them comparable.

See :mod:`.kshot` for the adaptation modes and acquisition policies these drive.

``P1`` — *gen7-compatible, exhaustive.*  Every row of the held-out ligand is a
candidate.  For each candidate set the calibration is fitted on it and scored on
**all other rows of that ligand**.  ``RANDOM`` is then the mean over candidate
sets and ``ORACLE`` the minimum — the same ligand, the same model, the same
arithmetic, differing only in whether the choice was lucky.  This is the protocol
that answers "how much room is there between random and optimal selection", and
its ``k = 1`` numbers are directly comparable to ``runs/gen7_architecture/kshot``.

``P2`` — *acquisition-fair.*  Each repeat splits the ligand's rows into a
**candidate pool** and a disjoint **evaluation set**.  Every policy selects from
the identical pool, sequentially, and every policy is scored on the identical
evaluation rows.  Without this two policies that pick different points would be
scored on different remainders, which is the comparison the brief forbids (§28).

Both protocols obey the same invariants, asserted in code:

* the model that produced ``prediction`` never saw the ligand (guaranteed
  upstream by the chemotype-blocked fold plan);
* a row used for calibration is never scored;
* no policy other than ``ORACLE`` is handed a candidate's target, and ``ORACLE``
  is labelled non-deployable wherever it appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .kshot import (
    Calibration, NON_DEPLOYABLE, POLICIES, PolicyContext, POLICY_AXES,
    apply_no_model, design_matrix, fit_calibration, make_oracle_policy, ridge_fit,
    stable_hash,
)

#: Adaptation modes evaluated everywhere, in increasing order of freedom.
MODES: tuple[str, ...] = ("K1", "K2", "K3")
#: k values the whole study reports.
K_VALUES: tuple[int, ...] = (0, 1, 2, 3, 5)
#: Candidate-pool cap per repeat.  Greedy policies are O(pool^2) per pick and one
#: ligand carries 1,488 rows; the cap is applied identically to every policy so it
#: cannot advantage one, and is logged so the truncation is never silent.
POOL_CAP = 48
#: P1 exhaustive candidate cap, same reasoning.  Beyond this the candidate set is
#: a deterministic subsample seeded on the ligand name.
P1_CANDIDATE_CAP = 400


def _standardised_axes(block: pd.DataFrame) -> np.ndarray:
    """Condition coordinates a policy navigates in.  Features only, never targets."""
    columns = []
    for name in POLICY_AXES:
        values = block[name].to_numpy(dtype=float) if name in block.columns \
            else np.zeros(len(block))
        finite = np.isfinite(values)
        if finite.sum() >= 2:
            centre, scale = float(values[finite].mean()), float(values[finite].std())
            column = np.where(finite, values - centre, 0.0)
            column = column / scale if scale > 1e-9 else np.zeros_like(column)
        else:
            column = np.zeros_like(values)
        columns.append(column)
    return np.vstack(columns).T


# --------------------------------------------------------------------------- #
# P1 — exhaustive, gen7-compatible
# --------------------------------------------------------------------------- #

def _k1_scores_exhaustive(residual: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """MAE on all other rows for every single-candidate offset correction.

    Under ``K1`` the whole calibration collapses to ``delta = r_i``, so the score
    of candidate *i* is ``mean_{j != i} |r_j - r_i|`` — the mean absolute deviation
    of the residual field from that one point.  Computing it in closed form is not
    an optimisation, it is the statement of what one-shot offset calibration *is*:
    the best possible single point is the one nearest the ligand's median residual,
    and the gap to random is a property of the residual spread.
    """
    n = len(residual)
    diff = np.abs(residual[candidates][:, None] - residual[None, :])
    total = diff.sum(axis=1)
    # remove the self term (which is zero) and divide by n-1
    return total / (n - 1)


def evaluate_p1(block: pd.DataFrame, *, modes: Sequence[str] = MODES,
                k_values: Sequence[int] = K_VALUES, penalty: float,
                draws: int, rng: np.random.Generator,
                candidate_cap: int = P1_CANDIDATE_CAP) -> list[dict]:
    """Exhaustive-at-k=1, sampled-above: RANDOM / ORACLE / WORST for one ligand."""
    truth = block["log_D"].to_numpy(dtype=float)
    prediction = block["prediction"].to_numpy(dtype=float)
    residual = truth - prediction
    n = len(block)
    records: list[dict] = []
    zero_shot = float(np.abs(residual).mean())
    base = {"extractant": block["extractant"].iloc[0], "n_rows": n, "zero_shot": zero_shot}

    # The ORACLE_level bound: the single best *constant* offset, chosen with full
    # knowledge of every target.  Not achievable by any number of measurements
    # unless one of them lands exactly on the optimum.
    grid = np.median(residual)
    records.append({**base, "protocol": "P1", "mode": "ORACLE_level", "policy": "ORACLE",
                    "k": -1, "deployable": False,
                    "mae": float(np.abs(residual - grid).mean()), "n_scored": n})

    candidates = np.arange(n)
    if n > candidate_cap:
        sub = np.random.default_rng(stable_hash(str(block["extractant"].iloc[0])))
        candidates = np.sort(sub.choice(n, size=candidate_cap, replace=False))

    for mode in modes:
        for k in k_values:
            if k == 0:
                records.append({**base, "protocol": "P1", "mode": mode, "policy": "ZERO_SHOT",
                                "k": 0, "deployable": True, "mae": zero_shot, "n_scored": n})
                continue
            if k >= n:
                continue
            if k == 1 and mode == "K1":
                scores = _k1_scores_exhaustive(residual, candidates)
                no_model = np.array([np.abs(truth[np.arange(n) != i] - truth[i]).mean()
                                     for i in candidates])
            else:
                scores_list, no_model_list = [], []
                sets = _candidate_sets(candidates, k, draws, rng, n)
                for chosen in sets:
                    mask = np.ones(n, dtype=bool)
                    mask[chosen] = False
                    calibration = fit_calibration(block, prediction, truth, chosen, mode,
                                                  penalty=penalty)
                    adjusted = calibration.apply(block, prediction)
                    scores_list.append(float(np.abs(adjusted[mask] - truth[mask]).mean()))
                    no_model_list.append(float(np.abs(
                        apply_no_model(truth, chosen, n)[mask] - truth[mask]).mean()))
                scores = np.asarray(scores_list)
                no_model = np.asarray(no_model_list)
            records.append({**base, "protocol": "P1", "mode": mode, "policy": "RANDOM", "k": k,
                            "deployable": True, "mae": float(scores.mean()), "n_scored": n - k,
                            "n_candidate_sets": int(len(scores))})
            records.append({**base, "protocol": "P1", "mode": mode, "policy": "ORACLE", "k": k,
                            "deployable": False, "mae": float(scores.min()), "n_scored": n - k,
                            "n_candidate_sets": int(len(scores))})
            records.append({**base, "protocol": "P1", "mode": mode, "policy": "WORST", "k": k,
                            "deployable": False, "mae": float(scores.max()), "n_scored": n - k,
                            "n_candidate_sets": int(len(scores))})
            if mode == modes[0]:
                records.append({**base, "protocol": "P1", "mode": "NO_MODEL", "policy": "RANDOM",
                                "k": k, "deployable": True, "mae": float(no_model.mean()),
                                "n_scored": n - k, "n_candidate_sets": int(len(no_model))})
    return records


def _candidate_sets(candidates: np.ndarray, k: int, draws: int,
                    rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """Every k-subset when that is cheap, a random sample of them otherwise."""
    from itertools import combinations
    from math import comb
    if k == 1:
        return [np.array([c]) for c in candidates]
    if len(candidates) <= 24 and comb(len(candidates), k) <= max(draws, 200):
        return [np.asarray(c) for c in combinations(candidates, k)]
    return [rng.choice(candidates, size=k, replace=False) for _ in range(draws)]


# --------------------------------------------------------------------------- #
# P2 — acquisition-fair
# --------------------------------------------------------------------------- #

@dataclass
class P2Split:
    pool: np.ndarray
    evaluation: np.ndarray


def make_p2_split(n: int, rng: np.random.Generator, *, pool_cap: int = POOL_CAP) -> P2Split:
    """Half the ligand's rows may be measured; the other half is scored.  Disjoint."""
    order = rng.permutation(n)
    n_eval = max(2, n // 2)
    evaluation = np.sort(order[:n_eval])
    pool = np.sort(order[n_eval:])
    if len(pool) > pool_cap:
        pool = np.sort(rng.choice(pool, size=pool_cap, replace=False))
    return P2Split(pool=pool, evaluation=evaluation)


def evaluate_p2(block: pd.DataFrame, *, modes: Sequence[str] = MODES,
                k_values: Sequence[int] = K_VALUES, penalty: float,
                repeats: int, seed: int, policies: Sequence[str] | None = None,
                with_oracle: bool = True, pool_cap: int = POOL_CAP) -> list[dict]:
    """Sequential acquisition under identical pools and identical evaluation rows."""
    truth = block["log_D"].to_numpy(dtype=float)
    prediction = block["prediction"].to_numpy(dtype=float)
    n = len(block)
    max_k = max(k_values)
    axes = _standardised_axes(block)
    uncertainty = (block["uncertainty"].to_numpy(dtype=float) if "uncertainty" in block.columns
                   else np.full(n, np.nan))
    disagreement = (block["disagreement"].to_numpy(dtype=float) if "disagreement" in block.columns
                    else np.full(n, np.nan))
    names = list(policies) if policies is not None else list(POLICIES)
    records: list[dict] = []
    ligand = block["extractant"].iloc[0]

    for repeat in range(repeats):
        rng = np.random.default_rng((seed, repeat, stable_hash(ligand)))
        split = make_p2_split(n, rng, pool_cap=pool_cap)
        if len(split.pool) < 1 or len(split.evaluation) < 2:
            continue
        base = {"extractant": ligand, "n_rows": n, "repeat": repeat,
                "n_pool": int(len(split.pool)), "n_eval": int(len(split.evaluation))}
        zero_shot = float(np.abs(prediction[split.evaluation] - truth[split.evaluation]).mean())
        records.append({**base, "protocol": "P2", "mode": "K0", "policy": "ZERO_SHOT", "k": 0,
                        "deployable": True, "mae": zero_shot, "selected": ""})
        best_constant = float(np.abs(
            truth[split.evaluation] - prediction[split.evaluation]
            - np.median(truth - prediction)).mean())
        records.append({**base, "protocol": "P2", "mode": "ORACLE_level", "policy": "ORACLE",
                        "k": -1, "deployable": False, "mae": best_constant, "selected": ""})

        # A deployable policy's *selection* does not depend on the adaptation mode —
        # it never sees a target, so nothing about the fitting rule reaches it.  The
        # sequence is therefore chosen once and scored under every mode, which is not
        # only 3x cheaper but is the stronger comparison: the modes are then compared
        # on byte-identical measured rows.  ORACLE is the exception, because its
        # choice is by definition the best choice *for a given fitting rule*.
        sequences: dict[str, list[int]] = {}
        for policy_name in names:
            policy = POLICIES[policy_name]
            context = PolicyContext(
                block=block, prediction=prediction, pool=split.pool,
                evaluation=split.evaluation, axes=axes, uncertainty=uncertainty,
                disagreement=disagreement,
                rng=np.random.default_rng((seed, repeat, stable_hash(policy_name))),
                truth=None)
            selected: list[int] = []
            for _ in range(min(max_k, len(split.pool))):
                selected.append(int(policy(context, selected)))
            sequences[policy_name] = selected

        for mode in modes:
            per_mode = dict(sequences)
            if with_oracle:
                oracle = make_oracle_policy(mode, penalty)
                context = PolicyContext(
                    block=block, prediction=prediction, pool=split.pool,
                    evaluation=split.evaluation, axes=axes, uncertainty=uncertainty,
                    disagreement=disagreement,
                    rng=np.random.default_rng((seed, repeat, 7)), truth=truth)
                selected = []
                for _ in range(min(max_k, len(split.pool))):
                    selected.append(int(oracle(context, selected)))
                per_mode["ORACLE"] = selected
            for policy_name, sequence in per_mode.items():
                for k in k_values:
                    if k < 1 or k > len(sequence):
                        continue
                    chosen = np.asarray(sequence[:k], dtype=int)
                    calibration = fit_calibration(block, prediction, truth, chosen, mode,
                                                  penalty=penalty)
                    adjusted = calibration.apply(block, prediction)
                    mae = float(np.abs(adjusted[split.evaluation] - truth[split.evaluation]).mean())
                    records.append({**base, "protocol": "P2", "mode": mode, "policy": policy_name,
                                    "k": k, "deployable": policy_name not in NON_DEPLOYABLE,
                                    "mae": mae, "selected": ",".join(map(str, chosen))})
                    if mode == modes[0]:
                        null = apply_no_model(truth, chosen, n)
                        records.append({
                            **base, "protocol": "P2", "mode": "NO_MODEL", "policy": policy_name,
                            "k": k, "deployable": policy_name not in NON_DEPLOYABLE,
                            "mae": float(np.abs(null[split.evaluation] - truth[split.evaluation]).mean()),
                            "selected": ",".join(map(str, chosen))})
    return records
