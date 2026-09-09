"""One-pair calibration: what a single measured separation factor buys.

Deployment question: a new extractant has been measured for two lanthanides under
one condition (one ``log SF``).  How much better is the whole predicted series?

Adapters operate on a held-out cell's predicted centred curve ``c_pred`` (14,) and
one observed pair ``(a, b, y_ab)``:

* ``no_model_linear`` — ignore the model: assume ``log D`` is linear in the
  standardised Shannon radius with the slope set by the measured pair.
* ``rescale`` — multiply ``c_pred`` by the ratio ``y_ab / (c_pred[a] - c_pred[b])``,
  shrunk toward 1 (a gain adjuster; undefined when the predicted contrast is ~0,
  in which case the prediction is kept).
* ``basis_shift`` — move along the basis direction that changes the measured
  contrast: ``c_new = c_pred + lam * r * d / (d.d)`` with ``d = basis[:, a] - basis[:, b]``
  projected back through the basis (ridge toward the prediction), ``r`` the residual.

Support pairs are drawn deterministically per (cell, repeat) and the same draw is
used for every arm.  Query pairs exclude the support pair.  Scoring is the same
pair-level macro MAE as zero-shot.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .metals import LANTHANIDES, SHANNON_RADIUS_CN8

R_Z = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES])
R_Z = (R_Z - R_Z.mean()) / R_Z.std()


def stable_hash(text: str) -> int:
    return int(hashlib.blake2b(text.encode(), digest_size=8).hexdigest(), 16)


def support_pair(cell_id: str, pairs: list[tuple[int, int]], repeat: int, seed: int) -> tuple[int, int]:
    rng = np.random.default_rng(stable_hash(f"{cell_id}|{repeat}|{seed}"))
    return pairs[int(rng.integers(0, len(pairs)))]


def adapt_no_model_linear(a: int, b: int, y_ab: float) -> np.ndarray:
    slope = y_ab / (R_Z[a] - R_Z[b])
    return slope * R_Z


def adapt_rescale(c_pred: np.ndarray, a: int, b: int, y_ab: float, *, shrink: float = 0.5,
                  min_contrast: float = 0.3) -> np.ndarray:
    """Gain adjuster, gated at the pair-noise floor (0.3) and never allowed to flip the sign."""
    contrast = c_pred[a] - c_pred[b]
    if abs(contrast) < min_contrast:
        return c_pred
    gain = y_ab / contrast
    gain = 1.0 + shrink * (gain - 1.0)
    gain = float(np.clip(gain, 0.0, 3.0))
    return gain * c_pred


def adapt_basis_shift(c_pred: np.ndarray, basis: np.ndarray, a: int, b: int, y_ab: float, *,
                      lam: float = 0.7) -> np.ndarray:
    d = basis[:, a] - basis[:, b]
    r = y_ab - (c_pred[a] - c_pred[b])
    if float(d @ d) < 1e-12:
        return c_pred
    delta_coef = lam * r * d / float(d @ d)
    return c_pred + delta_coef @ basis


def evaluate_one_pair(curves_pred: dict[str, np.ndarray], bases: dict[str, np.ndarray | None],
                      Y_test: np.ndarray, cell_ids: list[str], extractants: list[str],
                      chemotypes: list[str], *, seed: int, repeats: int = 5,
                      min_metals: int = 3) -> pd.DataFrame:
    """Long table: (arm, adapter, repeat, cell, extractant, query pair, y, prediction)."""
    rows = []
    for i in range(len(Y_test)):
        obs = np.flatnonzero(~np.isnan(Y_test[i]))
        if len(obs) < min_metals:
            continue
        pairs = [(a, b) for a in obs for b in obs if a < b]
        for rep in range(repeats):
            a, b = support_pair(cell_ids[i], pairs, rep, seed)
            y_ab = Y_test[i, a] - Y_test[i, b]
            queries = [(p, q) for p, q in pairs if (p, q) != (a, b)]
            for arm, C in curves_pred.items():
                c0 = C[i]
                variants = {
                    "zero_shot": c0,
                    "rescale": adapt_rescale(c0, a, b, y_ab),
                    "no_model_linear": adapt_no_model_linear(a, b, y_ab),
                }
                if bases.get(arm) is not None:
                    variants["basis_shift"] = adapt_basis_shift(c0, bases[arm], a, b, y_ab)
                support_dz = int(abs(b - a))
                for adapter, c in variants.items():
                    for p, q in queries:
                        rows.append({"arm": arm, "adapter": adapter, "repeat": rep, "split_seed": seed,
                                     "cell_id": cell_ids[i], "extractant": extractants[i], "chemotype": chemotypes[i],
                                     "A": LANTHANIDES[p], "B": LANTHANIDES[q], "support": f"{LANTHANIDES[a]}-{LANTHANIDES[b]}",
                                     "support_index_gap": support_dz,
                                     "y": float(Y_test[i, p] - Y_test[i, q]), "prediction": float(c[p] - c[q])})
    return pd.DataFrame(rows)
