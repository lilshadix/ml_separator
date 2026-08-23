"""Where a row sits on its own curve — and why that is what was missing.

gen9's shape objective reduced dynamic-range compression on every axis and did not
restore the slopes, and raising its weight bought range only by paying accuracy in
proportion.  The obvious next question is whether the *level* was competing with the
shape, so the obvious next experiment is to delete the level: train the same learner,
on the same design and the same folds, to predict the **curve-centred** target
``y - mean(y over the row's curve)``.

That model, with nothing to do but predict shape, recovers **2.5 %** of the extractant
range — *worse* than the monolith.  Which rules out competition and points at
something more basic.

**The diagnosis.**  ``log D`` at a given *absolute* extractant concentration is
dominated by the ligand's level.  The within-curve response is only interpretable
relative to the curve's own measurement window, and different titrations span
different windows: 0.03 M is the top of one series and the bottom of another.  So the
conditional mean of the centred response, given absolute conditions alone, is close to
zero everywhere — and a flat prediction is the *correct* answer to the question the
model was being asked.  It was never asked where on its own curve the row sat.

Adding that — four columns, all functions of the condition list — moves every shape
metric at once (single seed, five folds, shape-only model):

============================  ==========  ==========  ==============
extractant axis               monolith    shape-only  + relative pos
============================  ==========  ==========  ==============
predicted slope (median)           0.115       0.051         **0.580**
slope MAE                          2.436       2.532         **1.966**
span recovery (median)             0.053       0.025         **0.209**
shape MAE                          0.663       0.691         **0.578**
within-curve Spearman              0.650       0.418         **0.803**
linear R2 of the prediction        0.862       0.730         **0.932**
============================  ==========  ==========  ==============

**Is this legitimate?**  Yes, and the reason is worth stating rather than assuming.
These columns read the *conditions* of the held-out ligand's own rows and no target of
any kind.  The gen7 harness explicitly hands a contender the test **frame** — features
only, with the target column dropped and an assertion that it is gone — because a
transductive method legitimately needs the candidate conditions; that is also exactly
what the gen8 deployment protocol assumes, since the user supplies the list of
conditions they are choosing between.  It is the same information gen9's acquisition
features already use as pool percentiles, and the same information gen8's post-hoc
slope repair uses when it fits a curve.  What none of them use is a measurement.

Everything here is **EXPLORATORY**: it was written after the pre-registered sweep was
read.  It is reported as a mechanism and a gen10 direction, never as a confirmed
pre-registered effect.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .curves import DEFAULT_AXES

#: Emitted column names, in order.  Prefixed so they cannot collide with a cohort
#: column and so a leakage audit can find them by pattern.
RELATIVE_COLUMNS: tuple[str, ...] = (
    "rel__position",          # 0 at the window's low end, 1 at its high end
    "rel__offset_from_mean",  # signed distance from the window's centre, axis units
    "rel__window_width",      # how wide the titration is, axis units
    "rel__n_points",          # how many points it has
    "rel__is_endpoint",       # 1 if the row is the lowest or highest point
)


def primary_curve(membership: pd.DataFrame, *,
                  axes: Sequence[str] = DEFAULT_AXES) -> pd.DataFrame:
    """One curve per row — the longest it belongs to, ties broken by ``curve_id``.

    A row can sit on an acid titration and a lanthanide series at once.  The longest
    curve is chosen because that is where its position is most informative, and the
    tie-break is deterministic so two runs describe the same row the same way.
    """
    work = membership[membership["axis"].isin(tuple(axes))].copy()
    work["_size"] = work.groupby("curve_id")["row_id"].transform("size")
    work = work.sort_values(["row_id", "_size", "curve_id"], ascending=[True, False, True])
    return work.drop_duplicates("row_id")[
        ["row_id", "curve_id", "axis", "axis_label", "axis_value"]]


def relative_position_features(frame: pd.DataFrame, membership: pd.DataFrame, *,
                               axes: Sequence[str] = DEFAULT_AXES) -> pd.DataFrame:
    """One row per cohort row: where it sits on its own curve.  Target-free.

    Rows that lie on no supervised curve get zeros — the honest encoding of "this
    row has no titration to be positioned within", and the same value the model
    would learn for a degenerate one-point curve.
    """
    primary = primary_curve(membership, axes=axes).set_index("row_id")
    stats = primary.groupby("curve_id")["axis_value"].agg(["min", "max", "size"])

    row_ids = frame["row_id"].astype(str).to_numpy()
    curve = primary["curve_id"].reindex(row_ids).to_numpy()
    value = primary["axis_value"].reindex(row_ids).to_numpy(dtype=float)
    low = stats["min"].reindex(pd.Index(curve)).to_numpy(dtype=float)
    high = stats["max"].reindex(pd.Index(curve)).to_numpy(dtype=float)
    count = stats["size"].reindex(pd.Index(curve)).to_numpy(dtype=float)

    width = high - low
    usable = np.isfinite(width) & (width > 1e-12) & np.isfinite(value)
    position = np.where(usable, (value - low) / np.where(usable, width, 1.0), 0.0)
    centre = np.where(usable, value - 0.5 * (low + high), 0.0)
    endpoint = np.where(usable & ((position <= 1e-9) | (position >= 1 - 1e-9)), 1.0, 0.0)

    out = pd.DataFrame({
        "row_id": row_ids,
        "rel__position": position,
        "rel__offset_from_mean": centre,
        "rel__window_width": np.where(usable, width, 0.0),
        "rel__n_points": np.where(np.isfinite(count), count, 0.0),
        "rel__is_endpoint": endpoint,
    })
    return out


def curve_membership_map(frame: pd.DataFrame, membership: pd.DataFrame, *,
                         axes: Sequence[str] = DEFAULT_AXES) -> pd.Series:
    """``row_id -> curve_id`` for the row's primary curve, ``NaN`` when it has none."""
    primary = primary_curve(membership, axes=axes).set_index("row_id")["curve_id"]
    return pd.Series(primary.reindex(frame["row_id"].astype(str).to_numpy()).to_numpy(),
                     index=frame.index, name="curve_id")
