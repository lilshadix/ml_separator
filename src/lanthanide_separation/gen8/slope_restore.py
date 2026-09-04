"""Restore the response slope the frozen model shrinks away — a target-free repair.

The curve audit (``scripts/gen8_slopes.py``) measured something no previous
generation reported: on a held-out ligand the model does not get the *magnitude*
of the response wrong at the margin, it gets it wrong by a factor.

.. code-block:: text

    curve type            true median slope    predicted median slope   ratio
    extractant conc.            2.57                    0.12             0.05
    acid conc.                  1.66                    0.36             0.22
    lanthanide series           0.084                   0.040            0.38

An ensemble of trees fitted across 152 ligands and asked about a chemotype it has
never seen returns something close to the conditional mean over training ligands,
and averaging many ligands whose curves sit at different heights flattens the
common slope.  The predicted extractant titration rises by 0.1 log units where the
measured one rises by 2.1.  *That is not a level error and one measurement will not
fix it* — it is a shape error, and it is systematic and one-signed.

The repair is available without measuring anything, because the model's own
predicted slope is a function of the *features* alone:

1. fit the frozen model's **predictions** for the held-out ligand against the
   mass-action design — this uses no target of any kind, only the model's output;
2. compare each fitted slope with the **prior slope for that axis**, estimated from
   the training ligands' measured curves inside the fold;
3. add back the difference, shrunk, along each axis the ligand actually varies in.

The level is deliberately left untouched: re-centring on the block mean means the
correction changes the *shape* and cannot change the ligand's mean prediction, so
it composes cleanly with any k-shot level calibration and the two effects stay
separable in the results table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: (design column, minimum distinct values before the axis is considered varied).
SLOPE_AXES: tuple[tuple[str, int], ...] = (
    ("massact__log10_cond__extractant_concentration_M", 3),
    ("massact__log10_cond__acid_concentration_M", 3),
    ("lanthanide_index", 3),
)
#: How far to move toward the prior slope.  1.0 replaces the model's slope outright;
#: the default is deliberately below that because the prior is a *median over
#: training ligands* and a ligand whose true slope is genuinely small should not be
#: dragged all the way to the median.
DEFAULT_STRENGTH = 0.8
#: Ridge penalty when fitting the model's own predicted slope.  Small: the design is
#: a handful of columns and the response is the model's own smooth output.
FIT_PENALTY = 1e-3


from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MEMBERSHIP_PATH = REPO_ROOT / "runs" / "gen8_architecture" / "series" / "curve_membership.parquet"
CURVE_PATH = REPO_ROOT / "runs" / "gen8_architecture" / "series" / "curve_table.parquet"

#: Axes the repair acts on, as they are labelled in the curve membership table.
REPAIR_AXES: tuple[str, ...] = (
    "cond__extractant_concentration_M",
    "cond__acid_concentration_M",
    "metal",
)
#: A curve needs this many points before its slope is estimated or repaired.
MIN_CURVE_POINTS = 3
#: ...and an axis needs this many training curves before its prior is trusted.
MIN_PRIOR_CURVES = 8
#: How far to move toward the prior slope.  1.0 replaces the model's slope outright.
DEFAULT_STRENGTH = 0.5

def _fit_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """OLS slope of y on x, or None when the abscissa carries no spread."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_CURVE_POINTS:
        return None
    x, y = x[ok], y[ok]
    if len(np.unique(np.round(x, 9))) < MIN_CURVE_POINTS or np.ptp(x) <= 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def training_slope_priors(curves: pd.DataFrame, training_ligands: set) -> dict[str, float]:
    """Median measured slope per axis over the fold's **training** curves.

    Measured *per curve* — one axis varying, everything else held fixed — because
    that is the quantity the audit measured and the quantity the repair injects.
    A regression pooling a ligand's whole row block would mix the between-series
    level differences into the slope and give a different, smaller number.
    """
    sub = curves[curves["extractant"].isin(training_ligands)]
    sub = sub[sub["n_points"] >= MIN_CURVE_POINTS]
    return {axis: float(block["slope"].median())
            for axis, block in sub.groupby("axis", sort=True) if len(block) >= MIN_PRIOR_CURVES}


@dataclass
class SlopeRestore:
    """Adapter: repair the shape with a transferable slope prior, then calibrate the level.

    ``inner`` is the level-calibration adapter applied *after* the shape repair, so
    the results table can attribute a gain to the repair rather than to the
    calibration it is stacked on.  The repair is applied **curve by curve**: within
    one titration of one held-out ligand the model's own predicted slope is fitted
    (from predictions alone — no target of any kind), compared with the training
    prior for that axis, and the difference added back, re-centred on that curve so
    the ligand's level cannot move.
    """

    inner: object | None = None
    strength: float = DEFAULT_STRENGTH
    name: str = "SLOPE_RESTORE"
    axes: tuple = REPAIR_AXES
    membership: pd.DataFrame | None = None
    curves: pd.DataFrame | None = None
    _fold_priors: dict = field(default_factory=dict, repr=False)
    _index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.membership is None:
            self.membership = pd.read_parquet(MEMBERSHIP_PATH)
        if self.curves is None:
            self.curves = pd.read_parquet(CURVE_PATH)
        if not self._index:
            for axis in self.axes:
                sub = self.membership[self.membership["axis"] == axis]
                self._index[axis] = (
                    sub.set_index("row_id")["curve_id"].to_dict(),
                    sub.set_index("row_id")["axis_value"].to_dict())

    def fit_fold(self, train: pd.DataFrame, y_train: np.ndarray, *, split_seed: int,
                 fold: int, model_seed: int) -> None:
        ligands = set(train["extractant"].astype(str))
        self._fold_priors[(int(split_seed), int(fold))] = training_slope_priors(
            self.curves, ligands)

    def _repair(self, block: pd.DataFrame, prediction: np.ndarray, key) -> np.ndarray:
        priors = self._fold_priors.get(key)
        if not priors:
            return prediction
        row_ids = block["row_id"].to_numpy()
        out = np.asarray(prediction, dtype=float).copy()
        for axis in self.axes:
            prior = priors.get(axis)
            if prior is None:
                continue
            curve_of, value_of = self._index[axis]
            curve_ids = np.array([curve_of.get(r, "") for r in row_ids], dtype=object)
            values = np.array([value_of.get(r, np.nan) for r in row_ids], dtype=float)
            for curve_id in {c for c in curve_ids if c}:
                mask = curve_ids == curve_id
                if mask.sum() < MIN_CURVE_POINTS:
                    continue
                x = values[mask]
                fitted = _fit_slope(x, out[mask])
                if fitted is None:
                    continue
                delta = self.strength * (prior - fitted) * (x - np.nanmean(x))
                # Re-centred on the curve, so the repair changes the curve's shape
                # and cannot change its mean: level and shape stay separable.
                out[mask] = out[mask] + delta - np.nanmean(delta)
        return out

    def predict(self, block, prediction, selected, observed, context):
        repaired = self._repair(block, np.asarray(prediction, dtype=float),
                                (int(context.split_seed), int(context.fold)))
        if self.inner is None or len(selected) == 0:
            return repaired
        return self.inner.predict(block, repaired, selected, observed, context)


def build_slope_adapters(strengths=(0.5, 1.0), penalty: float = 4.0,
                         axis_sets: dict | None = None) -> list:
    """Adapters for the axis subsets the curve diagnostic separates.

    The repair is *not* uniformly good across axes, and pretending it is would hide
    the result.  Measured per curve on held-out ligands, moving the model's slope to
    the training prior takes extractant-titration shape MAE from 0.555 to 0.194
    (1,096 of 1,205 curves improve), acid from 0.553 to 0.509 (1,690 of 2,480), and
    the lanthanide series from 0.289 to 0.288 — nothing.  So the axis subsets are
    evaluated as separate arms rather than assumed equivalent.
    """
    from .adapters import RidgeOffset

    if axis_sets is None:
        axis_sets = {
            "L": ("cond__extractant_concentration_M",),
            "LH": ("cond__extractant_concentration_M", "cond__acid_concentration_M"),
            "LHM": REPAIR_AXES,
        }
    membership = pd.read_parquet(MEMBERSHIP_PATH)
    curves = pd.read_parquet(CURVE_PATH)
    out: list = []
    for label, axes in axis_sets.items():
        for strength in strengths:
            tag = f"{label}_s{f'{strength:g}'.replace('.', '')}"
            shared = dict(strength=strength, axes=tuple(axes), membership=membership,
                          curves=curves)
            out.append(SlopeRestore(inner=None, name=f"SLOPE_{tag}", **shared))
            for mode in ("K1", "K3"):
                out.append(SlopeRestore(inner=RidgeOffset(mode=mode, penalty=penalty),
                                        name=f"SLOPE_{tag}_{mode}", **shared))
    return out
