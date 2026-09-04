"""Series and curve reconstruction — what the model is actually supposed to represent.

gen7 established that most remaining error is a per-ligand *level* offset and that
*shape* transfers far better than level.  gen8's premise is that the shape is a
**function**, not a bag of rows: a ligand's ``log D`` is measured along titration
series in which exactly one experimental variable is swept while everything else
is held fixed.  Before any architecture can exploit that, the series have to be
reconstructed from the flat table, because the bundle carries no explicit "this is
an acid titration" label.

Two units, deliberately distinct:

``series``
    the gen5 definition, reused byte-for-byte: one extractant × one setting of the
    *categorical* conditions (acid identity, diluent, additive).  Everything that
    can vary inside a series is the metal and the five continuous columns.  A
    series is what a paper's Table 2 usually is: a grid, not a curve.

``curve``
    the new unit.  A maximal subset of one series in which **exactly one axis
    varies** and every other axis is fixed.  ``(series, metal, all continuous
    except acid) -> acid curve``.  This is the object a titration is, the object a
    slope is defined on, and the object a Neural Process should treat as a task.

The distinction matters because a series with 376 rows is a metal × acid grid, and
fitting one smooth curve through it — the brief's explicit warning — would be
fitting a curve through a surface.  A grid decomposes into many curves, one per
axis, and every one of them is honest.

Nothing in this module reads ``log_D`` to *define* a series or a curve: the
grouping is a function of the condition columns and the metal alone.  The target
is carried along only so the audit can report what was measured.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..levels import CONTINUOUS_CONDITION_COLUMNS

#: The continuous experimental axes, in the order the audit reports them.
CONTINUOUS_AXES: tuple[str, ...] = tuple(CONTINUOUS_CONDITION_COLUMNS)

#: Human names for the axes, used for the curve-type labels in the audit.
AXIS_LABEL: dict[str, str] = {
    "cond__acid_concentration_M": "acid",
    "cond__extractant_concentration_M": "extractant",
    "cond__metal_concentration_mM": "metal_concentration",
    "cond__temperature_C": "temperature",
    "cond__contact_time_min": "contact_time",
    "metal": "metal_series",
}

#: Every axis a curve may run along.  ``metal`` is ordered by lanthanide index, not
#: by an arbitrary category code, which is why it can carry a slope at all.
CURVE_AXES: tuple[str, ...] = CONTINUOUS_AXES + ("metal",)

#: A curve needs this many distinct points on its axis before a slope means anything.
MIN_CURVE_POINTS = 3
#: ...and this many before curvature (a second derivative) means anything.
MIN_CURVATURE_POINTS = 4

#: Rounding applied to a continuous column before it is used as a *grouping* key.
#: Without it, ``3.0`` and ``2.9999999`` split one curve in two.  Applied only to
#: the grouping key, never to the value the slope is computed from.
GROUP_ROUND_DECIMALS = 6


def _axis_values(frame: pd.DataFrame, axis: str) -> np.ndarray:
    """The numeric coordinate of one axis, on the scale a slope should be taken in.

    Concentrations are read on the **log10** scale, because the mass-action law is
    linear in log concentration (``log D = log K + n log[L] + m log[H+]``); a slope
    taken on the linear scale would not be the stoichiometric coefficient the
    chemistry defines, and would not be comparable between a 0.01 M and a 1 M
    titration.  Temperature, time and the metal are read on their natural scale.
    """
    if axis == "metal":
        return frame["lanthanide_index"].to_numpy(dtype=float) if "lanthanide_index" in frame.columns \
            else frame["metal_Z"].to_numpy(dtype=float)
    values = pd.to_numeric(frame[axis], errors="coerce").to_numpy(dtype=float)
    if axis in ("cond__acid_concentration_M", "cond__extractant_concentration_M",
                "cond__metal_concentration_mM"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(np.where(values > 0, values, np.nan)), np.nan)
    return values


def _group_key(frame: pd.DataFrame, held: Sequence[str]) -> pd.Series:
    """A stable hash of every axis that is *held fixed* for a candidate curve."""
    parts: list[np.ndarray] = []
    for axis in held:
        if axis == "metal":
            parts.append(np.asarray([str(v) for v in frame["metal_symbol"].to_numpy()], dtype=object))
        else:
            values = pd.to_numeric(frame[axis], errors="coerce").round(GROUP_ROUND_DECIMALS)
            # `.astype(str)` on a pandas-3 Series can leave real floats in an object
            # array (the same trap `levels.condition_labels` documents), so the
            # stringification goes through a plain Python list comprehension.
            parts.append(np.asarray([repr(v) for v in values.to_numpy(dtype=float)], dtype=object))
    if not parts:
        return pd.Series(["_"] * len(frame), index=frame.index)
    stacked = ["|".join(row) for row in zip(*parts)]
    return pd.Series(stacked, index=frame.index)


def _n_distinct(frame: pd.DataFrame, axis: str) -> int:
    if axis == "metal":
        return int(frame["metal_symbol"].nunique(dropna=False))
    return int(pd.to_numeric(frame[axis], errors="coerce").round(GROUP_ROUND_DECIMALS).nunique(dropna=False))


# --------------------------------------------------------------------------- #
# Curve extraction
# --------------------------------------------------------------------------- #

def build_curve_table(frame: pd.DataFrame, *, min_points: int = MIN_CURVE_POINTS) -> pd.DataFrame:
    """One row per (row_id, curve) membership.  A row may belong to several curves.

    A row sitting at the intersection of an acid titration and a metal series
    belongs to both, which is correct and is the whole reason cross-series
    transfer (brief §18) is a measurable question.  ``curve_id`` is a stable hash,
    so the table can be recomputed and joined without carrying state.
    """
    records: list[dict] = []
    for series_id, block in frame.groupby("series_id", sort=True):
        for axis in CURVE_AXES:
            if _n_distinct(block, axis) < min_points:
                continue
            held = [a for a in CURVE_AXES if a != axis]
            keys = _group_key(block, held)
            for key, sub in block.groupby(keys, sort=True):
                if _n_distinct(sub, axis) < min_points:
                    continue
                coord = _axis_values(sub, axis)
                if np.isfinite(coord).sum() < min_points:
                    continue
                digest = hashlib.sha1(f"{series_id}|{axis}|{key}".encode()).hexdigest()[:16]
                # A row whose coordinate on this axis is missing is *not on this
                # curve*: it has no abscissa, so it can carry no slope and cannot be
                # repaired along it.  Emitting it anyway put NaN coordinates into the
                # membership table, which silently propagated into every downstream
                # slope fit and correction.
                finite = np.isfinite(coord)
                n_points = int(finite.sum())
                for row_id, value, ok in zip(sub["row_id"].to_numpy(), coord, finite):
                    if not ok:
                        continue
                    records.append({
                        "curve_id": digest, "series_id": series_id, "axis": axis,
                        "axis_label": AXIS_LABEL[axis], "row_id": row_id,
                        "axis_value": float(value), "n_points": n_points,
                    })
    if not records:
        return pd.DataFrame(columns=["curve_id", "series_id", "axis", "axis_label",
                                     "row_id", "axis_value", "n_points"])
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Curve statistics — slope, curvature, monotonicity
# --------------------------------------------------------------------------- #

def curve_statistics(frame: pd.DataFrame, membership: pd.DataFrame,
                     *, target: str = "log_D", value_column: str | None = None) -> pd.DataFrame:
    """Per-curve slope / curvature / monotonicity of ``target`` along the axis.

    ``value_column`` overrides the column the statistics are taken of, so the same
    routine scores a *prediction* curve against a *truth* curve (brief §19).

    The slope is an ordinary least-squares fit on the axis coordinate (log10 for
    concentrations, see :func:`_axis_values`); the curvature is the quadratic
    coefficient of a degree-2 fit, defined only when the curve has at least
    ``MIN_CURVATURE_POINTS`` distinct abscissae.  A turning point is reported when
    the quadratic vertex falls strictly inside the measured range.
    """
    column = value_column or target
    values = frame.set_index("row_id")[column]
    records: list[dict] = []
    for curve_id, block in membership.groupby("curve_id", sort=True):
        x = block["axis_value"].to_numpy(dtype=float)
        y = values.reindex(block["row_id"].to_numpy()).to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < MIN_CURVE_POINTS or np.ptp(x) <= 0:
            continue
        order = np.argsort(x)
        x, y = x[order], y[order]
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = float(1.0 - (residual ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
        record = {
            "curve_id": curve_id, "series_id": block["series_id"].iloc[0],
            "axis": block["axis"].iloc[0], "axis_label": block["axis_label"].iloc[0],
            "n_points": int(len(x)), "n_distinct_x": int(len(np.unique(x))),
            "x_min": float(x.min()), "x_max": float(x.max()), "x_span": float(np.ptp(x)),
            "y_min": float(y.min()), "y_max": float(y.max()), "y_span": float(np.ptp(y)),
            "slope": float(slope), "intercept": float(intercept), "linear_r2": r2,
            "monotone_increasing": bool(np.all(np.diff(y) >= -1e-9)),
            "monotone_decreasing": bool(np.all(np.diff(y) <= 1e-9)),
        }
        if len(np.unique(x)) >= MIN_CURVATURE_POINTS:
            a, b, _ = np.polyfit(x, y, 2)
            record["curvature"] = float(a)
            if abs(a) > 1e-12:
                vertex = float(-b / (2 * a))
                record["turning_point"] = vertex if x.min() < vertex < x.max() else float("nan")
            else:
                record["turning_point"] = float("nan")
        else:
            record["curvature"] = float("nan")
            record["turning_point"] = float("nan")
        records.append(record)
    return pd.DataFrame.from_records(records) if records else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Series classification
# --------------------------------------------------------------------------- #

#: Order in which a series is assigned its type when several axes vary.  A series
#: is named after the axis carrying the *most* curve rows; ties break by this order.
CLASSIFICATION_PRIORITY: tuple[str, ...] = (
    "cond__acid_concentration_M", "cond__extractant_concentration_M", "metal",
    "cond__metal_concentration_mM", "cond__temperature_C", "cond__contact_time_min")


def classify_series(frame: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """One row per series: what varies, what is fixed, and which curve type it is.

    ``series_type`` is one of ``acid_sweep``, ``extractant_sweep``, ``metal_series``,
    ``metal_concentration_sweep``, ``temperature_sweep``, ``contact_time_sweep``,
    ``grid`` (more than one axis carries curves), ``single_point`` (one row) or
    ``unusable`` (rows exist but no axis reaches ``MIN_CURVE_POINTS``).

    ``grid`` is *not* a failure mode — it is the common case for a paper reporting a
    metal series at several acidities — and it is exactly the case where forcing one
    smooth curve would be wrong.  It is decomposed into curves, not discarded.
    """
    by_curve = membership.drop_duplicates("curve_id").set_index("curve_id")
    rows_per_axis = membership.groupby(["series_id", "axis"]).size().unstack(fill_value=0)
    curves_per_axis = membership.drop_duplicates("curve_id").groupby(
        ["series_id", "axis"]).size().unstack(fill_value=0)
    records: list[dict] = []
    for series_id, block in frame.groupby("series_id", sort=True):
        varying = [a for a in CURVE_AXES if _n_distinct(block, a) > 1]
        axes_with_curves = [a for a in CURVE_AXES
                            if series_id in rows_per_axis.index and rows_per_axis.get(a, {}).get(series_id, 0) > 0]
        if len(block) == 1:
            series_type = "single_point"
        elif not axes_with_curves:
            series_type = "unusable"
        elif len(axes_with_curves) == 1:
            axis = axes_with_curves[0]
            series_type = {"metal": "metal_series"}.get(axis, f"{AXIS_LABEL[axis]}_sweep")
        else:
            series_type = "grid"
        primary = None
        if axes_with_curves:
            counts = {a: int(rows_per_axis.loc[series_id, a]) for a in axes_with_curves}
            best = max(counts.values())
            tied = [a for a, c in counts.items() if c == best]
            primary = next(a for a in CLASSIFICATION_PRIORITY if a in tied)
        fixed = [a for a in CURVE_AXES if _n_distinct(block, a) == 1]
        records.append({
            "series_id": series_id,
            "extractant": block["extractant"].iloc[0],
            "tanimoto_cluster": block["tanimoto_cluster"].iloc[0],
            "n_rows": int(len(block)),
            "n_metals": int(block["metal_symbol"].nunique()),
            "metals": ",".join(sorted(set(block["metal_symbol"].astype(str)))),
            "series_type": series_type,
            "primary_axis": AXIS_LABEL[primary] if primary else "",
            "varying_axes": ",".join(AXIS_LABEL[a] for a in varying),
            "fixed_axes": ",".join(AXIS_LABEL[a] for a in fixed),
            "n_curves": int(sum(int(curves_per_axis.loc[series_id, a]) for a in axes_with_curves))
            if series_id in curves_per_axis.index else 0,
            "curve_rows": int(membership[membership["series_id"] == series_id]["row_id"].nunique()),
            "log_d_min": float(block["log_D"].min()), "log_d_max": float(block["log_D"].max()),
            "log_d_span": float(block["log_D"].max() - block["log_D"].min()),
        })
    return pd.DataFrame.from_records(records)


@dataclass(frozen=True)
class SeriesReconstruction:
    """The three tables gen8 builds once and every downstream experiment reads."""

    membership: pd.DataFrame     # row_id x curve_id, with the axis coordinate
    curves: pd.DataFrame         # one row per curve: slope, curvature, span
    series: pd.DataFrame         # one row per series: type, axes, counts

    def curve_rows(self) -> set:
        return set(self.membership["row_id"])


def reconstruct(frame: pd.DataFrame, *, min_points: int = MIN_CURVE_POINTS) -> SeriesReconstruction:
    membership = build_curve_table(frame, min_points=min_points)
    curves = curve_statistics(frame, membership)
    series = classify_series(frame, membership)
    if not curves.empty:
        series_of_curve = membership.drop_duplicates("curve_id").set_index("curve_id")["series_id"]
        curves = curves.assign(extractant=curves["curve_id"].map(
            series_of_curve.map(series.set_index("series_id")["extractant"])))
    return SeriesReconstruction(membership=membership, curves=curves, series=series)
