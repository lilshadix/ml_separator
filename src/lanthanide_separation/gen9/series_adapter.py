"""Series-local few-shot calibration with an empirical-Bayes hierarchy.

gen8's one failed falsification condition is the design brief for this module:
**a ligand does not have one level, it has a level per series.**  The transfer
matrix is strongly positive on the diagonal and mostly negative off it — one
acid-titration point does not calibrate the same ligand's metal selectivity in a
different diluent.  gen8's ``OFFSET_K1`` nevertheless fits a single intercept for
the whole ligand, which is why measuring in one series can make another worse.

gen9's adapter fits, on the ``k`` measured rows,

.. code-block:: text

    correction(row) = mu_ligand
                    + delta_series[s(row)]
                    + c_acid  * z_acid(row)
                    + c_extr  * z_extr(row)
                    + c_metal * z_metal(row)

with ``mu_ligand`` unpenalised and everything else shrunk toward zero — so
``delta_series`` is a *deviation* from the ligand level and the whole thing
degenerates gracefully to plain offset correction when the data cannot support
more.  That degeneration is the point: gen7 lost a k=2 arm to an unrestricted
affine fit that produced 1e12-scale coefficients, and gen8 established that four
shrunk coefficients beat one free one.  gen9 keeps the shrinkage and removes the
one thing gen8 tuned by hand.

**Where the penalties come from.**  Not from a schedule chosen to look sensible at
each k.  Each fold estimates, from *its own training ligands only*, how much these
coefficients actually vary between ligands (``tau``) and how much scatter is left
inside a ligand once they are fitted (``sigma``).  The MAP ridge penalty is then
``sigma^2 / tau^2`` per coefficient family, which is a measurement rather than a
knob.  k-dependence needs no rule at all: with one measurement the slope columns
carry almost no leverage and the posterior stays at the prior; with five they
move.  That is the same mechanism gen8 named "shrinkage, not degrees of freedom",
made explicit.

Nothing here reads a target the protocol did not hand over: ``fit_fold`` sees the
fold's training ligands, ``predict`` sees ``observed = truth[selected]``.  Series
identity is a *condition* label — acid, diluent, additive — known before any
measurement is made, so conditioning on it is not leakage; it is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from ..gen8.kshot import DEFAULT_RIDGE

#: The continuous response columns the adapter is allowed to bend, on the scale
#: the mass-action law is linear in.  Identical to gen8's ``K3_DESIGN`` so the two
#: adapters differ in *hierarchy*, not in which physics they can express.
RESPONSE_COLUMNS: tuple[str, ...] = (
    "lanthanide_index",
    "massact__log10_cond__acid_concentration_M",
    "massact__log10_cond__extractant_concentration_M",
)

#: Penalty families.  The intercept is never penalised.
FAMILIES: tuple[str, ...] = ("series", "response")

#: Floor and ceiling on an estimated penalty.  The floor stops a fold with three
#: near-identical training ligands from concluding that a coefficient family is
#: free; the ceiling stops a degenerate ``tau -> 0`` from switching a family off
#: entirely, which would silently turn the hierarchy back into ``OFFSET_K1``
#: without saying so.
PENALTY_FLOOR = 0.25
PENALTY_CEILING = 400.0


def _standardise_within_block(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros_like(values, dtype=float)
    centre = float(values[finite].mean())
    scale = float(values[finite].std())
    out = np.where(finite, values - centre, 0.0)
    return out / scale if scale > 1e-9 else np.zeros_like(out)


def build_design(block: pd.DataFrame, *, use_series: bool = True,
                 response_columns: tuple[str, ...] = RESPONSE_COLUMNS
                 ) -> tuple[np.ndarray, list[str]]:
    """``[1 | series deviations | standardised response columns]``, target-free.

    Series columns are centred so that a row's series column sums to zero across
    the block — the coding that makes ``mu_ligand`` the ligand's level rather than
    the level of whichever series happens to be listed first.
    """
    n = len(block)
    columns: list[np.ndarray] = [np.ones(n)]
    names: list[str] = ["intercept"]
    if use_series and "series_id" in block.columns:
        labels = block["series_id"].astype(str).to_numpy()
        unique = sorted(set(labels))
        if len(unique) > 1:
            for label in unique:
                indicator = (labels == label).astype(float)
                columns.append(indicator - indicator.mean())
                names.append(f"series::{label}")
    for name in response_columns:
        values = (block[name].to_numpy(dtype=float) if name in block.columns
                  else np.zeros(n))
        columns.append(_standardise_within_block(values))
        names.append(f"response::{name}")
    return np.vstack(columns).T, names


def family_of(name: str) -> str:
    if name == "intercept":
        return "intercept"
    return "series" if name.startswith("series::") else "response"


def ridge_map(design: np.ndarray, residual: np.ndarray, penalties: np.ndarray) -> np.ndarray:
    """Ridge with a per-column penalty; falls back to min-norm rather than raising."""
    gram = design.T @ design + np.diag(penalties)
    moment = design.T @ residual
    try:
        return np.linalg.solve(gram, moment)
    except np.linalg.LinAlgError:                        # pragma: no cover
        return np.linalg.lstsq(design, residual, rcond=None)[0]


# --------------------------------------------------------------------------- #
# Fold-local empirical prior
# --------------------------------------------------------------------------- #

def estimate_prior(train_oof: pd.DataFrame, *, use_series: bool = True,
                   min_rows: int = 6, ridge: float = 1.0) -> dict:
    """How much do these coefficients vary between ligands, inside this fold?

    Fitted on the fold's *training* ligands using their own out-of-fold residuals,
    which is the only honest estimate available: an in-sample residual on a ligand
    the model was trained on is near zero and would report a prior of zero
    variance, freezing every coefficient at the prior mean forever.
    """
    per_family: dict[str, list[float]] = {name: [] for name in FAMILIES}
    sigmas: list[float] = []
    n_ligands = 0
    for _, block in train_oof.groupby("extractant", sort=True):
        block = block.reset_index(drop=True)
        if len(block) < min_rows:
            continue
        residual = (block["log_D"].to_numpy(dtype=float)
                    - block["prediction"].to_numpy(dtype=float))
        design, names = build_design(block, use_series=use_series)
        if design.shape[1] >= len(block):
            continue
        penalties = np.full(design.shape[1], ridge)
        penalties[0] = 0.0
        beta = ridge_map(design, residual, penalties)
        fitted = design @ beta
        sigmas.append(float(np.std(residual - fitted)))
        for name, value in zip(names, beta):
            family = family_of(name)
            if family in per_family:
                per_family[family].append(float(value))
        n_ligands += 1

    sigma = float(np.median(sigmas)) if sigmas else 1.0
    penalties: dict[str, float] = {}
    taus: dict[str, float] = {}
    for family in FAMILIES:
        values = np.asarray(per_family[family], dtype=float)
        tau = float(np.std(values)) if len(values) >= 3 else float("nan")
        taus[family] = tau
        if not np.isfinite(tau) or tau <= 1e-6:
            penalties[family] = PENALTY_CEILING
        else:
            penalties[family] = float(np.clip((sigma ** 2) / (tau ** 2),
                                              PENALTY_FLOOR, PENALTY_CEILING))
    return {"sigma": sigma, "tau": taus, "penalty": penalties,
            "n_ligands": n_ligands, "n_coefficients": {f: len(per_family[f]) for f in FAMILIES}}


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #

@dataclass
class HierarchicalSeriesAdapter:
    """gen8 ``Adapter`` protocol; MAP hierarchy instead of a flat ridge.

    ``oof`` is the frozen model's out-of-fold table.  ``fit_fold`` reads from it
    only the rows of *other* folds of the same split seed — those ligands are the
    fold's training chemistry and their predictions are out-of-sample for them, so
    the prior is both fold-local and honest.
    """

    oof: pd.DataFrame
    name: str = "SERIES_MAP"
    use_series: bool = True
    fallback_penalty: float = DEFAULT_RIDGE
    #: Set False for the ablation that removes the hierarchy and keeps the
    #: empirical prior, which separates "series-local" from "better-shrunk".
    response: bool = True
    priors: dict = field(default_factory=dict, repr=False)

    def fit_fold(self, train: pd.DataFrame, y_train: np.ndarray, *,
                 split_seed: int, fold: int, model_seed: int) -> None:
        subset = self.oof[(self.oof["split_seed"] == int(split_seed))
                          & (self.oof["fold"] != int(fold))]
        held_out = set(train["extractant"].astype(str))
        # Belt and braces: the fold's own test ligands must not be in the prior.
        subset = subset[subset["extractant"].astype(str).isin(held_out)]
        self.priors[(int(split_seed), int(fold))] = estimate_prior(
            subset, use_series=self.use_series)

    def predict(self, block: pd.DataFrame, prediction: np.ndarray, selected: np.ndarray,
                observed: np.ndarray, context) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=float)
        if len(selected) == 0:
            return prediction
        response_columns = RESPONSE_COLUMNS if self.response else ()
        design, names = build_design(block, use_series=self.use_series,
                                     response_columns=response_columns)
        prior = self.priors.get((int(context.split_seed), int(context.fold)))
        penalties = np.empty(design.shape[1])
        for j, name in enumerate(names):
            family = family_of(name)
            if family == "intercept":
                penalties[j] = 0.0
            elif prior is None:
                penalties[j] = self.fallback_penalty
            else:
                penalties[j] = prior["penalty"][family]
        residual = np.asarray(observed, dtype=float) - prediction[selected]
        beta = ridge_map(design[selected], residual, penalties)
        return prediction + design @ beta


def build_series_adapters(oof: pd.DataFrame) -> list:
    """The gen9-C arm and the two ablations that say which half of it works."""
    return [
        HierarchicalSeriesAdapter(oof=oof, name="SERIES_MAP"),
        HierarchicalSeriesAdapter(oof=oof, name="SERIES_MAP_NOSERIES", use_series=False),
        HierarchicalSeriesAdapter(oof=oof, name="SERIES_MAP_LEVELONLY",
                                  use_series=True, response=False),
    ]
