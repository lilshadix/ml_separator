"""Within-curve training pairs — the object gen9's shape objective is defined on.

gen8 reconstructed the corpus into 1,176 one-axis curves
(:mod:`lanthanide_separation.gen8.series`) and then measured that the frozen model
draws every one of them too flat: the median extractant titration has a true slope
of 2.57 and a predicted slope of 0.12.  gen8 repaired that *after* prediction.
gen9 asks whether the flattening can be removed from the objective instead, and the
unit that makes that possible is a **pair of rows on the same curve**.

For two rows ``i`` and ``j`` of one curve::

    true_delta = y_j - y_i
    pred_delta = f(x_j) - f(x_i)

Matching the two supervises the response *shape* and nothing else: a per-ligand
intercept, a per-series level, and a publication's calibration offset all cancel
exactly, because they enter ``y_i`` and ``y_j`` identically.  That is the whole
point — gen7 and gen8 between them established that the zero-shot level is not
recoverable from structure, so an objective that keeps asking for it spends its
capacity on an impossible problem.

Three rules this module exists to enforce, each of which has a test.

**Fold locality.**  :func:`build_pairs` takes the row ids of one training
partition and will only ever emit pairs whose *both* endpoints are inside it.  A
delta whose two rows straddle the train/test boundary would put a held-out target
into a training loss, which is the leak gen9 is most exposed to.

**No curve may dominate by length.**  A 14-point curve offers 91 pairs and a
4-point curve offers 6.  Weighting pairs uniformly would let one long titration
outvote fifteen short ones, and the resulting model would be "good at the curves
that happen to be densely sampled" rather than "good at curve shape".  Every
sampler therefore emits a bounded number of pairs per curve and the weights are
renormalised so **each curve carries the same total weight** — optionally each
*ligand* instead, which is the stricter balance and is reported either way.

**No zero-baseline pairs.**  Two rows at the same abscissa carry no shape
information: their true delta is measurement noise and their predicted delta is
identically zero for any function of the conditions.  Training on them injects
pure noise into the gradient.  They are dropped, and the count is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..gen8.series import AXIS_LABEL

#: Axes gen9 supervises by default.  Deliberately the same three gen8's post-hoc
#: repair used, so "intrinsic training" and "post-hoc repair" are asked about the
#: same physics rather than about different axis inventories.
#:
#: * ``extractant`` — 241 curves, median true slope 2.31, the strongest target and
#:   the one with an identifiable mass-action meaning;
#: * ``acid`` — 496 curves, median true slope 1.57;
#: * ``metal`` — 386 curves; supervised as *order and shape*, never as a forced
#:   monotone trend, because the lanthanide response genuinely turns over.
#:
#: Excluded on purpose: ``contact_time`` (8 curves, median slope 1e-4 — gen8 shows
#: the axis is flat noise after equilibrium and supervising it would teach the
#: model to fit measurement scatter), ``metal_concentration`` (10 curves) and
#: ``temperature`` (35 curves over 6 ligands — too little support to generalise;
#: available as an explicit variant, never in the default).
DEFAULT_AXES: tuple[str, ...] = (
    "cond__extractant_concentration_M",
    "cond__acid_concentration_M",
    "metal",
)

#: Named axis sets the sweep may select.
AXIS_SETS: Mapping[str, tuple[str, ...]] = {
    "LHM": DEFAULT_AXES,
    "L": ("cond__extractant_concentration_M",),
    "LH": ("cond__extractant_concentration_M", "cond__acid_concentration_M"),
    "LHMT": DEFAULT_AXES + ("cond__temperature_C",),
    "ALL": DEFAULT_AXES + ("cond__temperature_C", "cond__metal_concentration_mM",
                           "cond__contact_time_min"),
}

#: Two abscissae closer than this are the same point.  ``curve_membership`` stores
#: log10 concentrations, so 1e-9 is far below any real titration step.
MIN_ABSCISSA_GAP = 1e-9

#: A curve needs this many rows *inside the partition* before it can contribute.
MIN_CURVE_POINTS = 3

#: The sampling strategies of the brief, in the order the phase-1 screen runs them.
SAMPLERS: tuple[str, ...] = (
    "ROW_ONLY", "ROW_ADJACENT", "ROW_ENDPOINT", "ROW_RANDOM_PAIR", "ROW_MULTISCALE")


@dataclass(frozen=True)
class CurvePairs:
    """Training pairs and per-curve spans, addressed by position in the design matrix.

    ``pairs`` holds *positional* indices into whatever design matrix the caller
    built for the partition, never ``row_id``, so the objective can index numpy
    directly.  ``curve_key`` and ``axis_label`` travel alongside so a diagnostic
    can attribute a gradient back to the chemistry it came from.
    """

    pairs: np.ndarray            # (P, 2) int
    pair_weight: np.ndarray      # (P,) float, sums to 1 over each balance unit
    pair_curve: np.ndarray       # (P,) int index into ``curves``
    pair_axis: np.ndarray        # (P,) object, axis label
    pair_gap: np.ndarray         # (P,) float, |x_j - x_i| on the supervised scale
    curves: tuple                # (curve_id, axis_label, ligand) per curve index
    curve_members: tuple         # tuple of int arrays: positional rows of each curve
    curve_weight: np.ndarray     # (C,) float, for the span term
    audit: dict

    def __len__(self) -> int:
        return int(len(self.pairs))

    @property
    def n_curves(self) -> int:
        return int(len(self.curves))


def _empty(audit: dict | None = None) -> CurvePairs:
    return CurvePairs(
        pairs=np.zeros((0, 2), dtype=int), pair_weight=np.zeros(0), pair_curve=np.zeros(0, dtype=int),
        pair_axis=np.zeros(0, dtype=object), pair_gap=np.zeros(0),
        curves=(), curve_members=(), curve_weight=np.zeros(0), audit=audit or {})


# --------------------------------------------------------------------------- #
# Per-curve pair enumeration
# --------------------------------------------------------------------------- #

def _adjacent(order: np.ndarray) -> list[tuple[int, int]]:
    return [(int(order[i]), int(order[i + 1])) for i in range(len(order) - 1)]


def _endpoint(order: np.ndarray) -> list[tuple[int, int]]:
    return [(int(order[0]), int(order[-1]))]


def _random_pairs(order: np.ndarray, rng: np.random.Generator, budget: int) -> list[tuple[int, int]]:
    n = len(order)
    out: list[tuple[int, int]] = []
    for _ in range(budget):
        a, b = rng.choice(n, size=2, replace=False)
        lo, hi = (a, b) if a < b else (b, a)
        out.append((int(order[lo]), int(order[hi])))
    return out


def _multiscale(order: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Adjacent + medium-lag + endpoint, in equal measure.

    The three scales are not interchangeable.  Adjacent pairs carry the local
    derivative and are what stops a model predicting a staircase; the endpoint
    pair is the only term that speaks directly to dynamic range, which is the
    quantity gen8 measured at 5 % of truth; medium lags interpolate between them
    and stop the objective being satisfied by a curve that is right at its two
    ends and wrong in the middle.
    """
    n = len(order)
    out = _adjacent(order)
    if n >= 4:
        lag = max(2, n // 3)
        out += [(int(order[i]), int(order[min(i + lag, n - 1)]))
                for i in range(0, n - 1, max(1, lag // 2))]
    out += _endpoint(order)
    if n >= 5:
        out += _random_pairs(order, rng, budget=max(1, n // 3))
    return out


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

#: How the total pair weight is shared out.
#:
#: ``curve``
#:     every curve carries the same total weight.  The simplest reading of "no
#:     curve may dominate by being long", and the default.
#: ``ligand``
#:     every *ligand* carries the same total weight, its curves sharing it.  A
#:     ligand measured along fifteen curves then does not outvote one measured
#:     along two.
#: ``cluster``
#:     each curve is weighted by the cluster-balanced row weight of its own rows —
#:     the same weighting the *row* loss uses.  This is not cosmetic.  With
#:     ``curve`` balance the two terms of the objective speak on different scales:
#:     a row in a rare ECFP cluster carries a row weight of ~7 and a delta weight
#:     of ~1, so the row term outguns the shape term precisely on the chemistry
#:     the shape term exists to fix.  Matching the weightings makes ``lambda_delta``
#:     mean the same thing for every row.
BALANCE_MODES: tuple[str, ...] = ("curve", "ligand", "cluster")


def build_pairs(
    membership: pd.DataFrame,
    partition_row_ids: Sequence[str],
    *,
    strategy: str = "ROW_MULTISCALE",
    axes: Sequence[str] = DEFAULT_AXES,
    rng: np.random.Generator | None = None,
    ligand_of_row: Mapping[str, str] | None = None,
    weight_of_row: Mapping[str, float] | None = None,
    balance: str = "curve",
    min_points: int = MIN_CURVE_POINTS,
) -> CurvePairs:
    """Pairs and curves for one training partition.

    ``membership`` is gen8's ``curve_membership`` table (``curve_id``, ``axis``,
    ``axis_label``, ``row_id``, ``axis_value``).  It is a pure function of the
    condition columns — no target ever enters it — so building it once over the
    whole cohort and *then* restricting to the partition is safe.  Restricting is
    the step that matters and it happens here, exactly once.

    ``weight_of_row`` is required by ``balance="cluster"`` and ignored otherwise.
    """
    if strategy not in SAMPLERS:
        raise ValueError(f"unknown sampling strategy {strategy!r}; expected one of {SAMPLERS}")
    if balance not in BALANCE_MODES:
        raise ValueError(f"balance must be one of {BALANCE_MODES}, got {balance!r}")
    if balance == "cluster" and weight_of_row is None:
        raise ValueError("balance='cluster' needs weight_of_row, the row loss's own weights")
    rng = rng if rng is not None else np.random.default_rng(0)

    partition = list(dict.fromkeys(str(r) for r in partition_row_ids))
    position = {row_id: i for i, row_id in enumerate(partition)}
    audit: dict = {
        "strategy": strategy, "balance": balance, "axes": list(axes),
        "n_partition_rows": len(partition),
        "n_membership_rows_total": int(len(membership)),
    }
    if strategy == "ROW_ONLY":
        audit.update({"n_curves": 0, "n_pairs": 0, "n_pairs_dropped_zero_gap": 0})
        return _empty(audit)

    allowed_axes = set(axes)
    work = membership[membership["axis"].isin(allowed_axes)]
    work = work[work["row_id"].astype(str).isin(position)]
    audit["n_membership_rows_in_partition"] = int(len(work))

    curves: list[tuple] = []
    curve_members: list[np.ndarray] = []
    pairs: list[tuple[int, int]] = []
    pair_curve: list[int] = []
    pair_gap: list[float] = []
    per_curve_counts: list[int] = []
    curve_row_weight: list[float] = []
    dropped_zero_gap = 0
    dropped_short = 0

    for curve_id, block in work.groupby("curve_id", sort=True):
        rows = block["row_id"].astype(str).to_numpy()
        idx = np.array([position[r] for r in rows], dtype=int)
        x = block["axis_value"].to_numpy(dtype=float)
        finite = np.isfinite(x)
        idx, x = idx[finite], x[finite]
        if len(idx) < min_points or len(np.unique(np.round(x, 9))) < min_points:
            dropped_short += 1
            continue
        order_local = np.argsort(x, kind="stable")
        # ``order`` indexes into (idx, x); the samplers work on those positions.
        if strategy == "ROW_ADJACENT":
            candidates = _adjacent(order_local)
        elif strategy == "ROW_ENDPOINT":
            candidates = _endpoint(order_local)
        elif strategy == "ROW_RANDOM_PAIR":
            candidates = _random_pairs(order_local, rng, budget=max(1, len(idx) - 1))
        else:
            candidates = _multiscale(order_local, rng)

        kept: list[tuple[int, int]] = []
        gaps: list[float] = []
        for a, b in candidates:
            if a == b:
                dropped_zero_gap += 1
                continue
            gap = abs(float(x[b] - x[a]))
            if gap < MIN_ABSCISSA_GAP:
                # Same abscissa: the true delta is replicate noise and the
                # predicted delta is identically zero for any function of the
                # conditions, so the pair can only add gradient variance.
                dropped_zero_gap += 1
                continue
            kept.append((int(idx[a]), int(idx[b])))
            gaps.append(gap)
        if not kept:
            dropped_short += 1
            continue

        c = len(curves)
        ligand = ""
        if ligand_of_row is not None:
            ligand = str(ligand_of_row.get(rows[0], ""))
        if weight_of_row is not None:
            curve_row_weight.append(
                float(np.mean([weight_of_row.get(r, 1.0) for r in rows])))
        curves.append((str(curve_id), str(block["axis_label"].iloc[0]), ligand))
        curve_members.append(idx[order_local])
        pairs.extend(kept)
        pair_curve.extend([c] * len(kept))
        pair_gap.extend(gaps)
        per_curve_counts.append(len(kept))

    if not pairs:
        audit.update({"n_curves": 0, "n_pairs": 0,
                      "n_pairs_dropped_zero_gap": dropped_zero_gap,
                      "n_curves_dropped_too_short": dropped_short})
        return _empty(audit)

    pair_curve_arr = np.asarray(pair_curve, dtype=int)
    counts = np.asarray(per_curve_counts, dtype=float)

    # --- balance -----------------------------------------------------------
    # Step 1: every curve's pairs share one unit of weight, so a 14-point curve
    # and a 4-point curve speak equally loudly.
    curve_weight = np.ones(len(curves), dtype=float)
    if balance == "ligand":
        ligands = np.asarray([c[2] for c in curves], dtype=object)
        for name in set(ligands.tolist()):
            mask = ligands == name
            curve_weight[mask] = 1.0 / float(mask.sum())
    elif balance == "cluster":
        curve_weight = np.asarray(curve_row_weight, dtype=float)
        curve_weight = np.where(np.isfinite(curve_weight) & (curve_weight > 0),
                                curve_weight, 1.0)
    pair_weight = curve_weight[pair_curve_arr] / counts[pair_curve_arr]

    total = float(pair_weight.sum())
    if total > 0:
        pair_weight = pair_weight / total
    span_weight = curve_weight / float(curve_weight.sum())

    axis_of_curve = np.asarray([c[1] for c in curves], dtype=object)
    audit.update({
        "n_curves": len(curves),
        "n_pairs": len(pairs),
        "n_pairs_dropped_zero_gap": int(dropped_zero_gap),
        "n_curves_dropped_too_short": int(dropped_short),
        "pairs_per_curve_median": float(np.median(counts)),
        "pairs_per_curve_max": int(counts.max()),
        "pairs_by_axis": {a: int((axis_of_curve[pair_curve_arr] == a).sum())
                          for a in sorted(set(axis_of_curve.tolist()))},
        "curves_by_axis": {a: int((axis_of_curve == a).sum())
                           for a in sorted(set(axis_of_curve.tolist()))},
        "max_curve_weight_share": float(
            pd.Series(pair_weight).groupby(pair_curve_arr).sum().max()),
        "n_ligands": int(len({c[2] for c in curves if c[2]})),
        "curve_weight_min": float(curve_weight.min()),
        "curve_weight_max": float(curve_weight.max()),
    })
    return CurvePairs(
        pairs=np.asarray(pairs, dtype=int), pair_weight=pair_weight,
        pair_curve=pair_curve_arr, pair_axis=axis_of_curve[pair_curve_arr],
        pair_gap=np.asarray(pair_gap, dtype=float),
        curves=tuple(curves), curve_members=tuple(curve_members),
        curve_weight=span_weight, audit=audit)


def assert_pairs_inside(pairs: CurvePairs, n_rows: int) -> None:
    """Every endpoint of every pair addresses a row of the partition.

    Cheap, and it is the assertion that would have caught a train/test straddle
    had the restriction in :func:`build_pairs` been written the other way round.
    """
    if len(pairs) == 0:
        return
    flat = pairs.pairs.reshape(-1)
    if flat.min() < 0 or flat.max() >= n_rows:
        raise AssertionError(
            f"curve pair addresses row {flat.min()}..{flat.max()} outside the "
            f"{n_rows}-row training partition")
    for members in pairs.curve_members:
        if len(members) and (members.min() < 0 or members.max() >= n_rows):
            raise AssertionError("curve membership addresses a row outside the partition")


def contribution_report(pairs: CurvePairs) -> pd.DataFrame:
    """Effective weight each curve and each ligand carries — brief §12.

    A weighting bug is invisible in the loss curve and obvious here.
    """
    if len(pairs) == 0:
        return pd.DataFrame(columns=["curve_id", "axis_label", "ligand", "n_pairs", "weight"])
    frame = pd.DataFrame({
        "curve": pairs.pair_curve, "weight": pairs.pair_weight,
        "axis_label": pairs.pair_axis})
    grouped = frame.groupby("curve").agg(n_pairs=("weight", "size"), weight=("weight", "sum"))
    meta = pd.DataFrame(list(pairs.curves), columns=["curve_id", "axis_label", "ligand"])
    out = meta.join(grouped[["n_pairs", "weight"]])
    return out.sort_values("weight", ascending=False).reset_index(drop=True)
