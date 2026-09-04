"""Series-local adaptation with an honest prior — Phase 5.

gen9's hierarchy was right about the *structure* and wrong about the *estimator*.
``SERIES_MAP`` beats ``SERIES_MAP_NOSERIES`` at every k, so the series term earns
its place; but it loses to gen8's hand-set ``OFFSET_K3`` (ridge 4.0) by 0.055 to
0.096, and the diagnosis is mechanical.  ``estimate_prior`` fits each training
ligand's residuals with a ridge of its own, takes the spread of the **fitted**
coefficients as ``tau``, and those coefficients are already shrunk — so ``tau`` is
underestimated, ``sigma^2 / tau^2`` is inflated, and the response coefficients are
shrunk about twice as hard as 4.0 shrinks them.  The prior measured a shrunk
quantity and treated it as the truth.

Three estimators, and only three, as the brief asks:

``FIXED``
    ``lambda = 4.0`` on every penalised column, series columns included.  The
    reference.  With the series columns removed it *is* ``OFFSET_K3``.
``INNER``
    the penalty pair ``(lambda_series, lambda_response)`` chosen per fold from a
    small fixed grid by simulating the k-shot protocol on the fold's **training**
    ligands and their out-of-fold residuals — a held-out ligand never informs the
    penalty that will be applied to it.  Selected per ``k``, because ``k`` is
    known at deployment and a penalty that is right for two points need not be
    right for five.
``ML``
    the corrected hierarchy.  ``tau`` per family and ``sigma`` are the maximisers
    of the **marginal likelihood** of the training ligands' residuals under
    ``beta_family ~ N(0, tau_f^2)``, ``eps ~ N(0, sigma^2)``, with the intercept
    a fixed effect.  The marginal likelihood integrates the coefficients out, so
    nothing shrunk is ever mistaken for a population quantity — which is exactly
    the defect it replaces.  A method-of-moments de-biased ``tau`` is computed
    alongside and written to the diagnostics as a cross-check, never used.

Everything else — the design, the centring of the series columns, the MAP solve,
the adapter protocol — is gen9's, unchanged, so a difference between these arms is
a difference in the penalty and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..gen8.kshot import DEFAULT_RIDGE, stable_hash
from ..gen8.protocols import make_p2_split
from ..gen9.series_adapter import (
    FAMILIES, PENALTY_CEILING, PENALTY_FLOOR, RESPONSE_COLUMNS, build_design, family_of,
    ridge_map,
)

#: The inner-selection grid.  Small and fixed — declared before the study ran.
SERIES_GRID: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
RESPONSE_GRID: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
#: k values the inner selection simulates.  k = 1 is algebraically identical for
#: every adapter (a ridge with an unpenalised intercept on one point is
#: ``beta = [r, 0, ...]``), so it is not selected for.
INNER_K: tuple[int, ...] = (2, 3, 5)
INNER_REPEATS = 4
MIN_LIGAND_ROWS = 6


def _penalty_vector(names: Sequence[str], series: float, response: float) -> np.ndarray:
    out = np.empty(len(names))
    for j, name in enumerate(names):
        family = family_of(name)
        out[j] = 0.0 if family == "intercept" else (series if family == "series" else response)
    return out


def _ligand_blocks(train_oof: pd.DataFrame, *, use_series: bool,
                   min_rows: int = MIN_LIGAND_ROWS) -> list[dict]:
    """Per training ligand: design, names, residual — built once and reused."""
    blocks = []
    for ligand, block in train_oof.groupby("extractant", sort=True):
        block = block.reset_index(drop=True)
        if len(block) < min_rows:
            continue
        residual = (block["log_D"].to_numpy(dtype=float)
                    - block["prediction"].to_numpy(dtype=float))
        design, names = build_design(block, use_series=use_series)
        if design.shape[1] >= len(block):
            continue
        blocks.append({"ligand": str(ligand), "design": design, "names": names,
                       "residual": residual, "n": int(len(block))})
    return blocks


# --------------------------------------------------------------------------- #
# INNER: penalty selected by simulating the protocol on training ligands
# --------------------------------------------------------------------------- #

def _simulate_mae(block: dict, series: float, response: float, k: int,
                  rng: np.random.Generator, repeats: int) -> float:
    """Mean MAE on disjoint evaluation rows after fitting on ``k`` pool rows."""
    design, residual, n = block["design"], block["residual"], block["n"]
    penalties = _penalty_vector(block["names"], series, response)
    scores = []
    for _ in range(repeats):
        split = make_p2_split(n, rng)
        if len(split.pool) < k or len(split.evaluation) < 2:
            continue
        # CENTRAL_THEN_SPREAD is the frontier's policy; a random draw from the pool
        # is the cheaper stand-in that does not privilege any penalty, because no
        # penalty can influence which rows are chosen.
        chosen = np.sort(rng.choice(split.pool, size=k, replace=False))
        beta = ridge_map(design[chosen], residual[chosen], penalties)
        corrected = residual - design @ beta
        scores.append(float(np.abs(corrected[split.evaluation]).mean()))
    return float(np.mean(scores)) if scores else float("nan")


def select_inner_penalties(train_oof: pd.DataFrame, *, use_series: bool = True,
                           seed: int = 0, repeats: int = INNER_REPEATS,
                           k_values: Sequence[int] = INNER_K) -> dict:
    """Per ``k``, the grid point with the lowest simulated one-ligand-one-vote MAE."""
    blocks = _ligand_blocks(train_oof, use_series=use_series)
    table: list[dict] = []
    for k in k_values:
        for series in (SERIES_GRID if use_series else (0.0,)):
            for response in RESPONSE_GRID:
                per_ligand = []
                for block in blocks:
                    rng = np.random.default_rng(
                        [int(seed), int(k), stable_hash(block["ligand"]),
                         int(series * 100), int(response * 100)])
                    value = _simulate_mae(block, series, response, k, rng, repeats)
                    if np.isfinite(value):
                        per_ligand.append(value)
                table.append({"k": int(k), "series": float(series),
                              "response": float(response),
                              "mae": float(np.mean(per_ligand)) if per_ligand else np.nan,
                              "n_ligands": len(per_ligand)})
    grid = pd.DataFrame.from_records(table)
    chosen: dict[int, dict] = {}
    for k, block in grid.groupby("k"):
        best = block.sort_values(["mae", "response", "series"]).iloc[0]
        chosen[int(k)] = {"series": float(best["series"]), "response": float(best["response"]),
                          "mae": float(best["mae"])}
    return {"chosen": chosen, "grid": grid, "n_ligands": len(blocks)}


# --------------------------------------------------------------------------- #
# ML: marginal-likelihood variance components
# --------------------------------------------------------------------------- #

def _negative_log_marginal(params: np.ndarray, blocks: list[dict], families: Sequence[str]
                           ) -> float:
    """-log p(residuals | sigma, tau) with the coefficients integrated out.

    Per ligand: ``r ~ N(X_fixed b, X_random diag(tau^2) X_random' + sigma^2 I)``.
    The intercept is a fixed effect profiled out analytically (GLS).  Evaluated
    through the Woodbury identity so the cost is in the number of random-effect
    columns, not in the number of rows.
    """
    log_sigma, *log_taus = params
    sigma2 = float(np.exp(2.0 * log_sigma))
    tau2 = {family: float(np.exp(2.0 * value)) for family, value in zip(families, log_taus)}
    total = 0.0
    for block in blocks:
        design, names, r = block["design"], block["names"], block["residual"]
        n = len(r)
        random_cols = [j for j, name in enumerate(names) if family_of(name) != "intercept"]
        lam = np.array([tau2[family_of(names[j])] for j in random_cols])
        z = design[:, random_cols] * np.sqrt(lam)[None, :]
        # V = sigma2 I + Z Z';  V^{-1} = (I - Z (sigma2 I + Z'Z)^{-1} Z') / sigma2
        inner = sigma2 * np.eye(len(random_cols)) + z.T @ z
        try:
            inner_chol = np.linalg.cholesky(inner)
        except np.linalg.LinAlgError:
            return 1e30
        log_det_v = n * np.log(sigma2) + 2.0 * np.log(np.diag(inner_chol)).sum() \
            - len(random_cols) * np.log(sigma2)

        def v_inv(m: np.ndarray) -> np.ndarray:
            t = np.linalg.solve(inner_chol, z.T @ m)
            t = np.linalg.solve(inner_chol.T, t)
            return (m - z @ t) / sigma2

        x = design[:, 0]
        vinv_x = v_inv(x)
        b = float(vinv_x @ r) / float(vinv_x @ x)
        e = r - x * b
        quad = float(e @ v_inv(e))
        total += 0.5 * (log_det_v + quad + n * np.log(2.0 * np.pi))
    return total


def estimate_ml_prior(train_oof: pd.DataFrame, *, use_series: bool = True,
                      families: Sequence[str] = FAMILIES) -> dict:
    """``sigma`` and per-family ``tau`` by maximum marginal likelihood."""
    from scipy.optimize import minimize

    blocks = _ligand_blocks(train_oof, use_series=use_series)
    active = [f for f in families if any(
        family_of(n) == f for block in blocks for n in block["names"])]
    if not blocks or not active:
        return {"sigma": 1.0, "tau": {f: float("nan") for f in families},
                "penalty": {f: DEFAULT_RIDGE for f in families}, "n_ligands": len(blocks),
                "converged": False, "method": "ML"}
    pooled_sd = float(np.std(np.concatenate([b["residual"] for b in blocks])))
    start = np.array([np.log(max(pooled_sd, 1e-3))] + [np.log(0.5 * max(pooled_sd, 1e-3))]
                     * len(active))
    result = minimize(_negative_log_marginal, start, args=(blocks, active),
                      method="L-BFGS-B", bounds=[(-6.0, 4.0)] * len(start))
    sigma = float(np.exp(result.x[0]))
    taus = {f: float(np.exp(v)) for f, v in zip(active, result.x[1:])}
    penalties = {}
    for f in families:
        tau = taus.get(f, float("nan"))
        if not np.isfinite(tau) or tau <= 1e-6:
            penalties[f] = PENALTY_CEILING
        else:
            penalties[f] = float(np.clip(sigma ** 2 / tau ** 2, PENALTY_FLOOR, PENALTY_CEILING))
    # Method-of-moments cross-check: var(beta_hat) - mean posterior variance,
    # under the fitted sigma.  Reported, never used.
    moments = {}
    for f in active:
        values, posterior = [], []
        for block in blocks:
            design, names, r = block["design"], block["names"], block["residual"]
            pen = _penalty_vector(names, penalties["series"], penalties["response"])
            gram = design.T @ design + np.diag(pen)
            beta = np.linalg.solve(gram, design.T @ r)
            cov = sigma ** 2 * np.linalg.inv(gram)
            for j, name in enumerate(names):
                if family_of(name) == f:
                    values.append(beta[j])
                    posterior.append(cov[j, j])
        if len(values) >= 3:
            moments[f] = float(np.sqrt(max(np.var(values) - np.mean(posterior), 0.0)))
    return {"sigma": sigma, "tau": {f: taus.get(f, float("nan")) for f in families},
            "penalty": penalties, "n_ligands": len(blocks),
            "converged": bool(result.success), "nll": float(result.fun),
            "tau_moment_debiased": moments, "method": "ML"}


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #

@dataclass
class CorrectedSeriesAdapter:
    """gen8 ``Adapter`` protocol; gen9's design; a corrected penalty estimator.

    ``estimator`` is one of ``FIXED``, ``INNER``, ``ML``.  ``fit_fold`` reads from
    ``oof`` only the rows of *other* folds of the same split seed — the fold's
    training chemistry, with predictions that are out-of-sample for them.
    """

    oof: pd.DataFrame
    estimator: str = "ML"
    name: str = ""
    use_series: bool = True
    fixed_series: float = DEFAULT_RIDGE
    fixed_response: float = DEFAULT_RIDGE
    fallback_penalty: float = DEFAULT_RIDGE
    priors: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.estimator not in ("FIXED", "INNER", "ML"):
            raise ValueError(f"unknown estimator {self.estimator!r}")
        if not self.name:
            suffix = "" if self.use_series else "_NOSERIES"
            self.name = f"SERIES_{self.estimator}{suffix}"

    def fit_fold(self, train: pd.DataFrame, y_train: np.ndarray, *,
                 split_seed: int, fold: int, model_seed: int) -> None:
        key = (int(split_seed), int(fold))
        if self.estimator == "FIXED":
            self.priors[key] = {"penalty": {"series": self.fixed_series,
                                            "response": self.fixed_response},
                                "method": "FIXED"}
            return
        subset = self.oof[(self.oof["split_seed"] == int(split_seed))
                          & (self.oof["fold"] != int(fold))]
        held_out = set(train["extractant"].astype(str))
        subset = subset[subset["extractant"].astype(str).isin(held_out)]
        if self.estimator == "ML":
            self.priors[key] = estimate_ml_prior(subset, use_series=self.use_series)
        else:
            chosen = select_inner_penalties(subset, use_series=self.use_series,
                                            seed=int(model_seed))
            self.priors[key] = {"per_k": chosen["chosen"], "method": "INNER",
                                "n_ligands": chosen["n_ligands"],
                                "grid": chosen["grid"].to_dict("records")}

    def _penalties(self, names: Sequence[str], prior: dict | None, k: int) -> np.ndarray:
        if prior is None:
            return _penalty_vector(names, self.fallback_penalty, self.fallback_penalty)
        if prior.get("method") == "INNER":
            per_k = prior["per_k"]
            # k outside the simulated set borrows the nearest simulated k.
            nearest = min(per_k, key=lambda kk: (abs(kk - k), kk))
            choice = per_k[nearest]
            return _penalty_vector(names, choice["series"], choice["response"])
        return _penalty_vector(names, prior["penalty"]["series"], prior["penalty"]["response"])

    def predict(self, block: pd.DataFrame, prediction: np.ndarray, selected: np.ndarray,
                observed: np.ndarray, context) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=float)
        if len(selected) == 0:
            return prediction
        design, names = build_design(block, use_series=self.use_series,
                                     response_columns=RESPONSE_COLUMNS)
        prior = self.priors.get((int(context.split_seed), int(context.fold)))
        penalties = self._penalties(names, prior, int(len(selected)))
        residual = np.asarray(observed, dtype=float) - prediction[selected]
        beta = ridge_map(design[selected], residual, penalties)
        return prediction + design @ beta


def build_corrected_adapters(oof: pd.DataFrame) -> list:
    """The three estimators with the series term, plus ML without it."""
    return [
        CorrectedSeriesAdapter(oof=oof, estimator="FIXED"),
        CorrectedSeriesAdapter(oof=oof, estimator="INNER"),
        CorrectedSeriesAdapter(oof=oof, estimator="ML"),
        CorrectedSeriesAdapter(oof=oof, estimator="ML", use_series=False),
    ]
