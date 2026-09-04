"""Curve geometry rebuilt from the frame in front of you — the deployment object.

gen9 trains and predicts against ``runs/gen9_shape/curves/curve_membership.parquet``,
a table built once over the whole cohort.  That is correct for the *evaluation*,
because the fold plan is chemotype-blocked and :func:`assert_partition_closure`
proves a curve never straddles a fold boundary — so a held-out ligand's curves are
built from the held-out ligand's own rows and nothing else.

It is not what happens at deployment.  There the user hands over a list of
conditions and asks for predictions, and the curves are whatever *that list*
decomposes into.  If those two objects can differ, every gen9 shape number is a
statement about a design the user did not necessarily propose.  This module makes
the deployment object explicit so the difference can be measured instead of
assumed away.

**The equivalence, measured rather than argued.**  ``build_curve_table`` groups by
``series_id`` first, and a series belongs to exactly one extractant, so restricting
the cohort to one ligand's rows and rebuilding gives byte-identical curves — 143 of
143 ligands, checked in ``tests/test_gen10_querycurves.py``.  The cohort table and
the deployment table therefore agree **when the user supplies the whole published
design**, and :mod:`.consistency` is the study of what happens when they do not.

**One inherited quirk, reproduced deliberately and reported.**
:func:`lanthanide_separation.gen9.relative.relative_position_features` computes a
curve's window from the rows whose *primary* curve it is, not from every row on the
curve.  A row sitting on a 9-point acid titration and a 5-point lanthanide series
counts toward the acid window only.  The 5-point series' window is then computed
from however many of its rows had no longer curve — which can be none of them, in
which case the curve vanishes from the statistics entirely.  gen10 keeps the
behaviour under ``window="primary"`` so the gen9 arms reproduce exactly, and offers
``window="curve"`` — the window of the whole curve — as the obvious repair, tested
as an arm rather than assumed better.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..gen8.series import build_curve_table
from ..gen9.curves import DEFAULT_AXES, MIN_CURVE_POINTS

#: Columns a membership table always carries, in this order.
MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "curve_id", "series_id", "axis", "axis_label", "row_id", "axis_value", "n_points")

#: How a curve's measurement window is defined.
#:
#: ``primary``
#:     rows whose *primary* (longest) curve this is.  gen9's behaviour, kept so the
#:     gen9 arms nest exactly.  A curve can lose most of its rows to a longer one.
#: ``curve``
#:     every row on the curve.  What "the window of this titration" means in
#:     English, and what a reader of the gen9 report would assume.
WINDOW_MODES: tuple[str, ...] = ("primary", "curve")


def empty_membership() -> pd.DataFrame:
    """A membership table with no rows but the right columns and dtypes."""
    return pd.DataFrame({
        "curve_id": pd.Series(dtype=object), "series_id": pd.Series(dtype=object),
        "axis": pd.Series(dtype=object), "axis_label": pd.Series(dtype=object),
        "row_id": pd.Series(dtype=object), "axis_value": pd.Series(dtype=float),
        "n_points": pd.Series(dtype=int)})


def query_membership(frame: pd.DataFrame, *, axes: Sequence[str] = DEFAULT_AXES,
                     min_points: int = MIN_CURVE_POINTS) -> pd.DataFrame:
    """Curve membership for exactly the rows in ``frame``.

    The deployment-time object: what a titration *is*, given the conditions the
    user actually proposed.  Restricted to ``axes`` because a curve on an axis the
    model was never taught to read carries no representation.
    """
    if len(frame) == 0:
        return empty_membership()
    table = build_curve_table(frame, min_points=min_points)
    if table.empty:
        return empty_membership()
    table = table[table["axis"].isin(tuple(axes))]
    if table.empty:
        return empty_membership()
    return table[list(MEMBERSHIP_COLUMNS)].reset_index(drop=True)


def primary_curve(membership: pd.DataFrame, *,
                  axes: Sequence[str] = DEFAULT_AXES) -> pd.DataFrame:
    """One curve per row — the longest it belongs to, ties broken by ``curve_id``.

    Transcribed from :func:`gen9.relative.primary_curve` so the two cannot drift.
    """
    work = membership[membership["axis"].isin(tuple(axes))].copy()
    if work.empty:
        return pd.DataFrame(columns=["row_id", "curve_id", "axis", "axis_label", "axis_value"])
    work["_size"] = work.groupby("curve_id")["row_id"].transform("size")
    work = work.sort_values(["row_id", "_size", "curve_id"], ascending=[True, False, True])
    return work.drop_duplicates("row_id")[
        ["row_id", "curve_id", "axis", "axis_label", "axis_value"]]


def window_statistics(membership: pd.DataFrame, *, axes: Sequence[str] = DEFAULT_AXES,
                      window: str = "primary") -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(primary, stats)``: each row's primary curve, and each curve's window.

    ``stats`` is indexed by ``curve_id`` and carries ``min`` / ``max`` / ``size``
    plus the sorted distinct abscissae as ``values``, which the rank and spacing
    representations need.  ``window`` selects which rows define the window; see
    :data:`WINDOW_MODES`.
    """
    if window not in WINDOW_MODES:
        raise ValueError(f"unknown window mode {window!r}; have {WINDOW_MODES}")
    primary = primary_curve(membership, axes=axes)
    if primary.empty:
        stats = pd.DataFrame(columns=["min", "max", "size", "values"])
        stats.index.name = "curve_id"
        return primary, stats
    source = primary if window == "primary" else \
        membership[membership["axis"].isin(tuple(axes))]
    grouped = source.groupby("curve_id")["axis_value"]
    stats = grouped.agg(["min", "max", "size"])
    stats["values"] = grouped.apply(lambda s: np.sort(np.unique(s.to_numpy(dtype=float))))
    return primary, stats


def assert_partition_closure(membership: pd.DataFrame, partition_row_ids: Sequence[str],
                             *, name: str = "partition") -> dict:
    """Every curve touching the partition must lie **entirely** inside it.

    The invariant that makes the cohort-wide table and the query-scoped table agree
    on a held-out fold.  Returns an audit dict; raises when it fails, because a
    curve that straddles the boundary means a training row is defining the window a
    held-out row is positioned in.
    """
    inside = set(str(r) for r in partition_row_ids)
    touching = membership[membership["row_id"].astype(str).isin(inside)]["curve_id"].unique()
    block = membership[membership["curve_id"].isin(touching)]
    outside = block[~block["row_id"].astype(str).isin(inside)]
    audit = {"partition": name, "n_rows": len(inside),
             "n_curves_touching": int(len(touching)),
             "n_rows_outside": int(len(outside)),
             "n_curves_straddling": int(outside["curve_id"].nunique())}
    if audit["n_curves_straddling"]:
        raise AssertionError(
            f"{audit['n_curves_straddling']} curves straddle the {name} boundary; "
            "a held-out row's window would be defined by a training row")
    return audit


def restricted_membership(membership: pd.DataFrame,
                          row_ids: Sequence[str]) -> pd.DataFrame:
    """The cohort table cut down to ``row_ids``, keeping every column.

    Used to show that the cohort table and :func:`query_membership` agree on a
    closed partition; the equality is asserted in the test suite, never assumed.
    """
    keep = set(str(r) for r in row_ids)
    out = membership[membership["row_id"].astype(str).isin(keep)]
    return out[list(MEMBERSHIP_COLUMNS)].reset_index(drop=True)


def sort_membership(table: pd.DataFrame) -> pd.DataFrame:
    """Canonical ordering, so two tables can be compared with ``.equals``."""
    if table.empty:
        return empty_membership()
    out = table[list(MEMBERSHIP_COLUMNS)].copy()
    for column in ("curve_id", "series_id", "axis", "axis_label", "row_id"):
        out[column] = out[column].astype(str)
    out["axis_value"] = out["axis_value"].astype(float)
    out["n_points"] = out["n_points"].astype(int)
    return out.sort_values(["curve_id", "row_id"]).reset_index(drop=True)
