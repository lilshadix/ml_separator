"""Query-set consistency — does the answer depend on the question you did not ask?

This is gen10's highest-priority diagnostic, and it exists because gen9's whole
result rests on a representation that is a function of the candidate design.  If
predicting ``log D`` at 0.03 M depends on whether the user also listed 0.30 M as a
point they were considering, then the model is not a response surface: it is a
function of the question, and two chemists with the same compound and the same
target concentration get different numbers because one of them was more ambitious
about their titration plan.

The benchmark holds the **absolute point fixed** and varies the design around it.
Seven perturbations, each isolating one way a real user's list could differ from
the published one:

``NESTED``
    drop points from one or both ends.  ``Q_small`` is a subset of ``Q_large``;
    compare on the intersection.  This is the commonest real case — the user runs
    a shorter titration than the paper did.
``DENSITY``
    keep ``min`` and ``max``, insert unmeasured interior points.  The window is
    unchanged by construction, so anything that moves is the model reacting to
    *sampling density* rather than to range.
``EXTEND``
    push one boundary out by a fixed number of decades.  The single most
    diagnostic perturbation for the ENDPOINT sensitivity class, because it
    rescales every position at once while changing no measured point.
``PERMUTE``
    reorder the rows.  Must be exactly invariant; anything else is a bug, not a
    property, and it is checked at zero tolerance.
``SPARSE``
    thin the design to 2, 3, 4, 6, 8 points while pinning the endpoints, so
    density falls without the window moving.
``SHIFT``
    translate the whole design along the axis.  Not a consistency requirement —
    a genuinely absolute response *should* move — but the one measurement that
    separates "the model reads relative position" from "the model reads absolute
    concentration", and it is reported as a diagnostic rather than as a failure.
``DECOY``
    add candidate points far outside the region being predicted.  A decoy is
    information about the user's ambitions, not about the chemistry, and the
    honest requirement is that it change nothing.

Every arm is measured against the same variants on the same ligands, and the
**frozen model is carried as a positive control**: it reads no design context at
all, so its shift must be exactly zero.  A benchmark that reports a non-zero shift
for the frozen model is measuring its own plumbing, which is the failure mode gen9
issue 1 was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..gen9.curves import DEFAULT_AXES
from .perturb import SYNTHESISABLE_AXES, assert_no_target, synthesise_points
from .querycurves import query_membership

#: How far outside the window an ``EXTEND`` or ``DECOY`` candidate is placed, in
#: axis units (decades for a concentration).  Declared before the benchmark ran.
EXTENSION_DECADES: tuple[float, ...] = (0.5, 1.0, 2.0)
#: Interior points inserted by ``DENSITY``.
DENSITY_INSERTS: tuple[int, ...] = (1, 3, 7)
#: Retained point counts for ``SPARSE``.
SPARSE_SIZES: tuple[int, ...] = (2, 3, 4, 6, 8)
#: Axis translations for ``SHIFT``.
SHIFT_DECADES: tuple[float, ...] = (1.0,)
#: A curve needs this many measured points before its design can be perturbed
#: meaningfully — below it, "drop the top two points" is the whole curve.
MIN_CURVE_ROWS = 5
#: At most this many curves per ligand enter the benchmark, chosen deterministically
#: and spread across axes.  TODGA alone carries hundreds of curves; without a cap it
#: would dominate the runtime and — since every summary is "one ligand, one vote" —
#: buy nothing for it.  The cap is recorded in every output so the truncation is
#: never silent.
MAX_CURVES_PER_LIGAND = 12

VARIANTS: tuple[str, ...] = (
    "REFERENCE", "CONTEXT", "NESTED", "DENSITY", "EXTEND", "PERMUTE", "SPARSE", "SHIFT",
    "DECOY")


@dataclass(frozen=True)
class QueryVariant:
    """One perturbed candidate design, and which rows it can be compared on."""

    family: str
    label: str
    frame: pd.DataFrame
    #: ``row_id``s present in **both** this design and the reference design; the
    #: only rows on which a difference is a difference rather than a new question.
    compare_ids: tuple[str, ...]
    detail: dict


def _curve_rows(ligand_rows: pd.DataFrame, membership: pd.DataFrame,
                curve_id: str) -> tuple[pd.DataFrame, np.ndarray]:
    block = membership[membership["curve_id"] == curve_id]
    order = np.argsort(block["axis_value"].to_numpy(dtype=float), kind="stable")
    ids = block["row_id"].astype(str).to_numpy()[order]
    values = block["axis_value"].to_numpy(dtype=float)[order]
    indexed = ligand_rows.set_index(ligand_rows["row_id"].astype(str))
    return indexed.loc[list(ids)].reset_index(drop=True), values


def build_variants(ligand_rows: pd.DataFrame, curve_id: str, axis: str,
                   membership: pd.DataFrame, *,
                   rng: np.random.Generator) -> list[QueryVariant]:
    """Every perturbation of one curve's design, with the rest of the ligand kept.

    The rows *not* on the perturbed curve travel unchanged in every variant.  A
    user asking about one titration still owns the rest of their data, and holding
    them fixed means a measured shift is attributable to the curve whose design
    moved.
    """
    assert_no_target(ligand_rows)
    curve_frame, values = _curve_rows(ligand_rows, membership, curve_id)
    curve_ids = curve_frame["row_id"].astype(str).to_numpy()
    # A curve lives inside one series (gen8.series.build_curve_table groups by
    # series_id first), so rows of *other* series can neither join this curve nor
    # change which curve is a row's primary one.  Carrying only the same-series
    # rows keeps every geometric outcome identical and makes the benchmark
    # tractable on the 1,488-row ligand.
    series_id = str(curve_frame["series_id"].iloc[0]) if "series_id" in curve_frame else None
    others = ligand_rows[~ligand_rows["row_id"].astype(str).isin(set(curve_ids))]
    if series_id is not None:
        others = others[others["series_id"].astype(str) == series_id]
    n = len(curve_frame)
    if n < MIN_CURVE_ROWS:
        return []
    low, high = float(values.min()), float(values.max())
    can_synthesise = axis in SYNTHESISABLE_AXES
    template = curve_frame.iloc[0]

    def design(keep: Sequence[int], extra: pd.DataFrame | None = None) -> pd.DataFrame:
        parts = [curve_frame.iloc[list(keep)], others]
        if extra is not None and len(extra):
            parts.append(extra)
        return pd.concat(parts, ignore_index=True)

    out: list[QueryVariant] = []
    everything = tuple(curve_ids)

    # --- 0. the reference design, and the series-restriction check ----------
    # The first variant *is* the published design restricted to this series; every
    # shift is measured against it.  The second is the whole ligand: a model whose
    # answer on this curve changes when rows of an unrelated series are added has
    # broken the "curves live inside one series" invariant the restriction rests
    # on, so it must report exactly zero for every arm and is checked, not assumed.
    reference = design(range(n))
    out.append(QueryVariant("REFERENCE", "published", reference, everything,
                            {"n_rows": int(len(reference)), "width_after": high - low}))
    out.append(QueryVariant("CONTEXT", "whole_ligand", ligand_rows.reset_index(drop=True),
                            everything, {"n_rows": int(len(ligand_rows))}))

    # --- A. nested windows -------------------------------------------------
    for drop_low, drop_high in ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0)):
        keep = list(range(drop_low, n - drop_high))
        if len(keep) < 3:
            continue
        kept = tuple(curve_ids[i] for i in keep)
        out.append(QueryVariant(
            "NESTED", f"drop_lo{drop_low}_hi{drop_high}", design(keep), kept,
            {"n_kept": len(keep), "n_dropped": n - len(keep),
             "width_before": high - low,
             "width_after": float(values[keep].max() - values[keep].min())}))

    # --- D. permutation ----------------------------------------------------
    permutation = rng.permutation(len(reference))
    out.append(QueryVariant(
        "PERMUTE", "shuffled", reference.iloc[permutation].reset_index(drop=True),
        everything, {"n_rows": int(len(reference))}))

    # --- E. sparse grids (endpoints pinned) --------------------------------
    for size in SPARSE_SIZES:
        if size >= n or size < 2:
            continue
        interior = list(range(1, n - 1))
        take = sorted(rng.choice(interior, size=size - 2, replace=False).tolist()) \
            if size > 2 else []
        keep = [0] + take + [n - 1]
        kept = tuple(curve_ids[i] for i in keep)
        out.append(QueryVariant(
            "SPARSE", f"keep{size}", design(keep), kept,
            {"n_kept": len(keep), "width_after": high - low}))

    if not can_synthesise:
        return out

    # --- B. grid density ---------------------------------------------------
    for inserts in DENSITY_INSERTS:
        grid = np.linspace(low, high, inserts + 2)[1:-1]
        # Only points that are not already measured; a duplicate abscissa is a
        # replicate, which is a different question entirely.
        fresh = [g for g in grid if np.min(np.abs(values - g)) > 1e-6]
        if not fresh:
            continue
        extra = synthesise_points(template, axis, fresh)
        out.append(QueryVariant(
            "DENSITY", f"insert{len(fresh)}", design(range(n), extra), everything,
            {"n_inserted": len(fresh), "width_after": high - low}))

    # --- C. window extension -----------------------------------------------
    for decades in EXTENSION_DECADES:
        for side in ("high", "low"):
            value = high + decades if side == "high" else low - decades
            extra = synthesise_points(template, axis, [value])
            out.append(QueryVariant(
                "EXTEND", f"{side}_{decades:g}", design(range(n), extra), everything,
                {"decades": float(decades), "side": side,
                 "width_before": high - low, "width_after": (high - low) + decades}))

    # --- G. decoys, far from anything being predicted ----------------------
    for decades in (2.0, 3.0):
        extra = synthesise_points(template, axis, [high + decades, low - decades])
        out.append(QueryVariant(
            "DECOY", f"both_{decades:g}", design(range(n), extra), everything,
            {"decades": float(decades), "n_decoys": 2,
             "width_after": (high - low) + 2 * decades}))

    # --- F. shifted window (diagnostic, not a consistency requirement) -----
    for decades in SHIFT_DECADES:
        shifted = synthesise_points(template, axis, (values + decades).tolist())
        frame = pd.concat([shifted, others], ignore_index=True)
        out.append(QueryVariant(
            "SHIFT", f"plus_{decades:g}", frame, tuple(shifted["row_id"].astype(str)),
            {"decades": float(decades), "paired_with": list(curve_ids)}))
    return out


def compare_variant(reference: pd.Series, variant_prediction: pd.Series,
                    compare_ids: Sequence[str]) -> dict:
    """Row-level shift between two designs, on the rows they share."""
    ids = [i for i in compare_ids if i in reference.index and i in variant_prediction.index]
    if not ids:
        return {"n_compared": 0}
    a = reference.reindex(ids).to_numpy(dtype=float)
    b = variant_prediction.reindex(ids).to_numpy(dtype=float)
    delta = np.abs(a - b)
    centred = np.abs((a - a.mean()) - (b - b.mean()))
    return {
        "n_compared": len(ids),
        "shift_median": float(np.median(delta)),
        "shift_mean": float(delta.mean()),
        "shift_p90": float(np.quantile(delta, 0.90)),
        "shift_p95": float(np.quantile(delta, 0.95)),
        "shift_max": float(delta.max()),
        # The level of a curve is what one measurement supplies, so a design change
        # that only moves the level is far less damaging than one that changes the
        # shape.  Reported separately rather than folded into one number.
        "shape_shift_median": float(np.median(centred)),
        "shape_shift_max": float(centred.max()),
        "level_shift": float(abs(a.mean() - b.mean())),
        "reference_span": float(np.ptp(a)),
        "variant_span": float(np.ptp(b)),
    }


def shift_for_shifted_window(reference: pd.Series, variant_prediction: pd.Series,
                             original_ids: Sequence[str],
                             shifted_ids: Sequence[str]) -> dict:
    """``SHIFT``: compare the predicted *shape* before and after translation.

    A model reading only relative position predicts the identical centred curve at
    a different absolute location; a model reading only absolute conditions
    predicts something unrelated.  The statistic is therefore the difference of
    curve-centred predictions, paired point-for-point by rank.
    """
    a = reference.reindex(list(original_ids)).to_numpy(dtype=float)
    b = variant_prediction.reindex(list(shifted_ids)).to_numpy(dtype=float)
    if len(a) != len(b) or not len(a):
        return {"n_compared": 0}
    ac, bc = a - a.mean(), b - b.mean()
    return {"n_compared": len(a),
            "shape_shift_median": float(np.median(np.abs(ac - bc))),
            "shape_shift_max": float(np.max(np.abs(ac - bc))),
            "level_shift": float(abs(a.mean() - b.mean())),
            "reference_span": float(np.ptp(a)), "variant_span": float(np.ptp(b))}


def eligible_curves(ligand_rows: pd.DataFrame, *,
                    axes: Sequence[str] = DEFAULT_AXES,
                    max_curves: int | None = MAX_CURVES_PER_LIGAND,
                    rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Curves of one ligand big enough to have their design perturbed.

    When a ligand has more than ``max_curves`` eligible curves, a deterministic
    subset is taken round-robin across axes (so a grid ligand contributes acid,
    extractant *and* lanthanide curves rather than only its most numerous kind),
    ordered by ``curve_id`` when ``rng`` is None and shuffled by ``rng`` otherwise.
    """
    membership = query_membership(ligand_rows, axes=axes)
    if membership.empty:
        return membership
    size = membership.groupby("curve_id")["row_id"].transform("size")
    membership = membership[size >= MIN_CURVE_ROWS]
    if max_curves is None or membership["curve_id"].nunique() <= max_curves:
        return membership
    per_axis: dict[str, list[str]] = {}
    for axis, block in membership.groupby("axis", sort=True):
        ids = sorted(block["curve_id"].unique())
        if rng is not None:
            ids = [ids[i] for i in rng.permutation(len(ids))]
        per_axis[axis] = ids
    chosen: list[str] = []
    while len(chosen) < max_curves and any(per_axis.values()):
        for axis in sorted(per_axis):
            if per_axis[axis] and len(chosen) < max_curves:
                chosen.append(per_axis[axis].pop(0))
    return membership[membership["curve_id"].isin(set(chosen))]
