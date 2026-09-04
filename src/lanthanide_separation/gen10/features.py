"""Design-relative coordinates, and how much each one depends on the question asked.

gen9 answered "where does this row sit on its own curve" with five columns, four of
which are functions of the window's **endpoints**::

    position          = (v - min) / (max - min)
    offset_from_mid   = v - (min + max) / 2
    window_width      = max - min
    is_endpoint       = v in {min, max}

Every one of those moves when the user adds a candidate outside the measured range,
and moves for *every other point at once*.  Ask about {0.01, 0.02, 0.03} and 0.03
has position 1.0; ask about {0.01, 0.02, 0.03, 0.30} — the same three experiments
plus one more you were considering — and 0.03 has position 0.06.  The model is
handed a different question about an identical experiment.

That is a **prediction, not a complaint**, and it is what makes gen10's Phase 4 an
experiment rather than a sweep.  Each column below is labelled with its sensitivity
class, declared before the consistency benchmark was run:

``ABSOLUTE``
    no dependence on the query set at all.  Perfectly stable, and by itself
    exactly the coordinate system gen9 proved insufficient.
``LOCAL``
    depends only on the nearest measured neighbours.  A decoy point three decades
    away cannot move it; an *interpolating* point can, and should — a titration
    sampled twice as densely really is a different design.
``RANK``
    depends on the ordering of the whole set, so one added point moves it by
    ``O(1/n)`` and never rescales anything.
``MOMENT``
    depends on the mean / median / IQR of the set: bounded influence, degrading
    gracefully, but not local.
``ENDPOINT``
    depends on ``min`` and ``max``.  Unbounded influence: one candidate can double
    the width and halve every position.  gen9's representation is four-fifths this
    class.

The claim gen10 tests is that the ``ENDPOINT`` class is where the query-set
instability lives, that ``RANK``/``LOCAL``/``MOMENT`` carry the same information
about *where a point sits in its design*, and that a representation built from them
keeps gen9's shape gain while losing gen9's fragility.  If the instability turns
out to be small, the simpler gen9 representation stays and this module is a
measurement rather than a change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..gen9.curves import DEFAULT_AXES
from ..gen9.relative import RELATIVE_COLUMNS
from .querycurves import window_statistics

#: Sensitivity class of every emitted column.  Declared here, before the
#: consistency benchmark, so the benchmark can be read as a test of this table.
SENSITIVITY: Mapping[str, str] = {
    "q10__on_curve": "RANK",
    "q10__axis_value": "ABSOLUTE",
    "q10__axis_is_metal": "ABSOLUTE",
    "q10__position": "ENDPOINT",
    "q10__offset_from_mid": "ENDPOINT",
    "q10__window_width": "ENDPOINT",
    "q10__n_points": "RANK",
    "q10__is_endpoint": "ENDPOINT",
    "q10__dist_to_boundary": "ENDPOINT",
    "q10__rank_pct": "RANK",
    "q10__centered_rank": "RANK",
    "q10__ordinal": "RANK",
    "q10__rank_from_end": "RANK",
    "q10__offset_from_median": "MOMENT",
    "q10__z_within_window": "MOMENT",
    "q10__iqr": "MOMENT",
    "q10__gap_below": "LOCAL",
    "q10__gap_above": "LOCAL",
    "q10__min_gap": "LOCAL",
    "q10__median_gap": "LOCAL",
}

#: Every column this module can emit, in a fixed order.
ALL_COLUMNS: tuple[str, ...] = tuple(SENSITIVITY)

#: gen9's five columns, under gen10 names, in gen9's order.  The mapping exists so
#: :func:`as_gen9_frame` can prove the two implementations agree numerically.
GEN9_EQUIVALENT: Mapping[str, str] = {
    "rel__position": "q10__position",
    "rel__offset_from_mean": "q10__offset_from_mid",
    "rel__window_width": "q10__window_width",
    "rel__n_points": "q10__n_points",
    "rel__is_endpoint": "q10__is_endpoint",
}

#: The predeclared representation families Phase 4 compares.  Small and fixed: the
#: brief forbids a sweep, and a sweep over coordinate systems is exactly how one
#: manufactures a win.
FEATURE_SETS: Mapping[str, tuple[str, ...]] = {
    # gen9, exactly.  The control every gen10 representation is quoted against.
    "GEN9": tuple(GEN9_EQUIVALENT.values()),
    # gen9 plus the two things it left out: an explicit "this row is on no curve"
    # indicator (gen9 encodes that as position 0, which collides with a genuine
    # lowest point) and the absolute coordinate of the axis being positioned in.
    "GEN9_PLUS": tuple(GEN9_EQUIVALENT.values()) + (
        "q10__on_curve", "q10__axis_value", "q10__axis_is_metal"),
    # Rank only: ordering information, no scale, no endpoints.
    "RANK": ("q10__on_curve", "q10__axis_value", "q10__axis_is_metal",
             "q10__rank_pct", "q10__centered_rank", "q10__ordinal",
             "q10__rank_from_end", "q10__n_points"),
    # Local geometry: how far the neighbours are, how tight the design is here.
    "LOCAL": ("q10__on_curve", "q10__axis_value", "q10__axis_is_metal",
              "q10__offset_from_median", "q10__gap_below", "q10__gap_above",
              "q10__min_gap", "q10__median_gap", "q10__iqr", "q10__n_points"),
    # The brief's hybrid: absolute + rank + local spacing, no endpoint term.
    "HYBRID": ("q10__on_curve", "q10__axis_value", "q10__axis_is_metal",
               "q10__rank_pct", "q10__centered_rank", "q10__rank_from_end",
               "q10__offset_from_median", "q10__z_within_window",
               "q10__gap_below", "q10__gap_above", "q10__min_gap",
               "q10__median_gap", "q10__iqr", "q10__n_points"),
    # Everything, including the endpoint columns.  The upper bound on what the
    # representation can buy, and the arm that says whether dropping the endpoint
    # class costs accuracy.
    "ALL": ALL_COLUMNS,
    # No design context at all — the A0 control for the representation question.
    "NONE": (),
}


def _gaps(values: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance to the nearest measured point below and above, on the axis.

    At an endpoint there is no neighbour on one side.  Mirroring the other side —
    rather than emitting a sentinel — keeps the column continuous and keeps the
    *meaning* ("how dense is the design where I am standing") intact; a sentinel
    would let a tree split on "is an endpoint", which is the ENDPOINT class this
    representation exists to avoid re-entering through the back door.
    """
    if len(values) < 2:
        return np.zeros_like(position), np.zeros_like(position)
    index = np.searchsorted(values, position, side="left")
    index = np.clip(index, 0, len(values) - 1)
    below = np.where(index > 0, position - values[np.maximum(index - 1, 0)], np.nan)
    above = np.where(index < len(values) - 1,
                     values[np.minimum(index + 1, len(values) - 1)] - position, np.nan)
    below = np.where(np.isfinite(below), below, above)
    above = np.where(np.isfinite(above), above, below)
    return np.nan_to_num(below, nan=0.0), np.nan_to_num(above, nan=0.0)


def query_features(frame: pd.DataFrame, membership: pd.DataFrame, *,
                   axes: Sequence[str] = DEFAULT_AXES,
                   window: str = "primary",
                   gen9_compat: bool = False) -> pd.DataFrame:
    """One row per row of ``frame``: where it sits inside the design ``membership``.

    Target-free by construction — every input is a condition column or a function
    of the condition list.  Rows on no supervised curve get zeros and
    ``q10__on_curve = 0``; the indicator is the one thing gen9's encoding lacked,
    because a row off every curve and a row at the bottom of its window both scored
    ``position = 0`` and the model could not tell them apart.

    ``gen9_compat`` reproduces one gen9 quirk exactly: a row whose primary-curve
    *group* has collapsed to a single point has zero window width and is therefore
    unpositionable, but gen9 still writes ``rel__n_points = 1`` for it while
    zeroing the other four columns.  Nineteen cohort rows are affected.  The flag
    exists so ``GEN10_REL_MONOLITH`` can be asserted bit-identical to
    ``GEN9_REL_MONOLITH``; every gen10 arm uses the corrected semantics, and the
    difference between the two is measured rather than waved through.
    """
    row_ids = frame["row_id"].astype(str).to_numpy()
    n = len(row_ids)
    out = {name: np.zeros(n, dtype=float) for name in ALL_COLUMNS}
    if n == 0 or membership.empty:
        return pd.DataFrame({"row_id": row_ids, **out})

    primary, stats = window_statistics(membership, axes=axes, window=window)
    if primary.empty:
        return pd.DataFrame({"row_id": row_ids, **out})
    primary = primary.set_index("row_id")
    curve = primary["curve_id"].reindex(row_ids).to_numpy()
    value = primary["axis_value"].reindex(row_ids).to_numpy(dtype=float)
    label = primary["axis_label"].reindex(row_ids).to_numpy()

    lookup = {cid: (float(row["min"]), float(row["max"]), float(row["size"]),
                    np.asarray(row["values"], dtype=float))
              for cid, row in stats.iterrows()}

    on_curve = np.zeros(n, dtype=float)
    for i in range(n):
        cid = curve[i]
        if cid is None or (isinstance(cid, float) and not np.isfinite(cid)) or cid not in lookup:
            continue
        low, high, size, values = lookup[cid]
        v = value[i]
        if not np.isfinite(v) or not np.isfinite(low) or not np.isfinite(high):
            continue
        width = high - low
        if not (width > 1e-12):
            if gen9_compat and np.isfinite(size):
                out["q10__n_points"][i] = size
            continue
        on_curve[i] = 1.0
        out["q10__axis_value"][i] = v
        out["q10__axis_is_metal"][i] = 1.0 if str(label[i]) == "metal_series" else 0.0
        # --- ENDPOINT class -------------------------------------------------
        position = (v - low) / width
        out["q10__position"][i] = position
        out["q10__offset_from_mid"][i] = v - 0.5 * (low + high)
        out["q10__window_width"][i] = width
        out["q10__is_endpoint"][i] = 1.0 if (position <= 1e-9 or position >= 1 - 1e-9) else 0.0
        out["q10__dist_to_boundary"][i] = float(min(v - low, high - v))
        out["q10__n_points"][i] = size
        # --- RANK class ------------------------------------------------------
        distinct = values
        m = len(distinct)
        ordinal = float(np.searchsorted(distinct, v, side="left"))
        ordinal = float(min(ordinal, m - 1))
        out["q10__ordinal"][i] = ordinal
        out["q10__rank_pct"][i] = ordinal / (m - 1) if m > 1 else 0.0
        out["q10__centered_rank"][i] = out["q10__rank_pct"][i] - 0.5
        out["q10__rank_from_end"][i] = float(min(ordinal, (m - 1) - ordinal))
        # --- MOMENT class ----------------------------------------------------
        median = float(np.median(distinct))
        out["q10__offset_from_median"][i] = v - median
        sd = float(np.std(distinct))
        out["q10__z_within_window"][i] = (v - float(np.mean(distinct))) / sd if sd > 1e-12 else 0.0
        out["q10__iqr"][i] = float(np.subtract(*np.percentile(distinct, [75, 25]))) if m > 1 else 0.0
        # --- LOCAL class -----------------------------------------------------
        below, above = _gaps(distinct, np.asarray([v], dtype=float))
        out["q10__gap_below"][i] = float(below[0])
        out["q10__gap_above"][i] = float(above[0])
        out["q10__min_gap"][i] = float(min(below[0], above[0]))
        out["q10__median_gap"][i] = float(np.median(np.diff(distinct))) if m > 1 else 0.0
    out["q10__on_curve"] = on_curve
    return pd.DataFrame({"row_id": row_ids, **out})


def as_gen9_frame(features: pd.DataFrame) -> pd.DataFrame:
    """gen10 features renamed to gen9's five columns, for the equivalence test."""
    inverse = {v: k for k, v in GEN9_EQUIVALENT.items()}
    keep = ["row_id"] + list(GEN9_EQUIVALENT.values())
    return features[keep].rename(columns=inverse)[["row_id"] + list(RELATIVE_COLUMNS)]


@dataclass(frozen=True)
class Representation:
    """A named subset of :data:`ALL_COLUMNS`, plus how the window is defined."""

    name: str
    columns: tuple[str, ...]
    window: str = "primary"

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(sorted({SENSITIVITY[c] for c in self.columns}))

    @property
    def has_endpoint_terms(self) -> bool:
        return any(SENSITIVITY[c] == "ENDPOINT" for c in self.columns)

    def as_dict(self) -> dict:
        return {"name": self.name, "columns": list(self.columns), "window": self.window,
                "classes": list(self.classes), "endpoint_terms": self.has_endpoint_terms}


def representation(name: str, *, window: str = "primary") -> Representation:
    if name not in FEATURE_SETS:
        raise KeyError(f"unknown representation {name!r}; have {sorted(FEATURE_SETS)}")
    return Representation(name=name, columns=tuple(FEATURE_SETS[name]), window=window)


def matrix(features: pd.DataFrame, row_ids: Sequence[str],
           spec: Representation) -> np.ndarray:
    """``(len(row_ids), len(spec.columns))`` float array, aligned on ``row_id``."""
    if not spec.columns:
        return np.zeros((len(row_ids), 0), dtype=float)
    indexed = features.set_index("row_id")
    return indexed.reindex(np.asarray(row_ids, dtype=object))[
        list(spec.columns)].to_numpy(dtype=float)
