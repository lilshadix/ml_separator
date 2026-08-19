"""Level / shape / offset decomposition — stop compressing every failure into one MAE.

Five generations of this project reported a single MAE per arm and then argued
about what it meant.  The gen5 four-regime run finally separated the two things
that MAE was mixing: on a held-out ligand the model predicts the **shape** of the
response across metals and conditions (shape R² 0.17–0.31) but not its absolute
**level** (deployable within-ligand R² ≤ 0).  Those are different failures with
different cures — more chemistry versus a better representation — so gen6 refuses
to report the sum.

The decomposition, for a held-out ligand *l* with residual ``e_i = ŷ_i − y_i``:

.. code-block:: text

    b_l        = mean_{i in l} e_i                 (the ligand's level offset)
    offset_mae = mean_l |b_l|                      (equal weight per ligand)
    ŷ_c, y_c   = prediction and truth centred on their own per-ligand means
    shape_mae  = mean_l mean_{i in l} |ŷ_c − y_c|

and the exact algebraic identity that ties them to the usual squared error

.. code-block:: text

    SSE_total = SSE_centred + Σ_l n_l · b_l²

which is unit-tested (:func:`decompose_level_shape` returns all three terms).
The identity is why "offset" and "shape" are the right two pieces: they are
orthogonal components of the same error, not two arbitrary summaries.

Three further rules the older harness learned the hard way and this module
enforces:

* **A shape statistic needs at least two rows.** A one-row ligand has
  ``ŷ_c = y_c = 0`` and would contribute a free zero to ``shape_mae``; such
  ligands are excluded from shape statistics and counted separately.  They still
  contribute to ``offset_mae``, where a single residual *is* the level error.
* **Hard-chemistry rows may not disappear into an average.** The endpoints at
  nearest-neighbour Tanimoto < 0.4 and < 0.6 are computed as first-class
  metrics, with their own n.
* **Bootstrap the held-out chemistry unit, not the row.** Scoring can be per
  ECFP cluster while independence lives at the Tanimoto super-cluster; the two
  differ under a chemotype hold-out and resampling the wrong one understates the
  width by 1.65x (measured on this cohort: blocked 0.287 vs unblocked 0.174).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..levels import LEVEL_TARGET_COLUMN

#: Ligands with fewer rows than this contribute no shape statistic (see module docstring).
MIN_ROWS_FOR_SHAPE = 2
#: Ligands with fewer rows than this contribute no rank statistic.
MIN_ROWS_FOR_RANK = 3
#: Row cap per ligand for the O(n²) pairwise sign accuracy.  The largest ligand in
#: this cohort holds ~1,500 rows (2.2 M pairs); the cap keeps the cost bounded and
#: the subsample is deterministic in the ligand name, so the number is stable.
SIGN_ACCURACY_ROW_CAP = 1200
#: Hard-chemistry thresholds on nearest-neighbour Tanimoto to the *reference*
#: training set (the brief's co-primary endpoints).
HARD_CHEMISTRY_THRESHOLDS: tuple[float, ...] = (0.4, 0.6)


# --------------------------------------------------------------------------- #
# Core decomposition
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LevelShapeDecomposition:
    """Per-ligand table plus the scalar summary the reports quote."""

    per_ligand: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def __getitem__(self, key: str):
        return self.summary[key]


def _sign_accuracy_within(y: np.ndarray, p: np.ndarray, *, rng: np.random.Generator) -> tuple[float, int]:
    """Fraction of within-ligand row pairs whose predicted order matches the truth.

    Pairs tied in the truth are excluded (they carry no orderable information);
    pairs tied in the *prediction* count as one half, the usual convention, so a
    constant predictor scores 0.5 rather than 0.
    """
    n = len(y)
    if n > SIGN_ACCURACY_ROW_CAP:
        take = rng.choice(n, size=SIGN_ACCURACY_ROW_CAP, replace=False)
        y, p = y[take], p[take]
        n = SIGN_ACCURACY_ROW_CAP
    if n < 2:
        return float("nan"), 0
    dy = np.subtract.outer(y, y)
    dp = np.subtract.outer(p, p)
    iu = np.triu_indices(n, k=1)
    dy, dp = dy[iu], dp[iu]
    orderable = dy != 0
    total = int(orderable.sum())
    if total == 0:
        return float("nan"), 0
    dy, dp = dy[orderable], dp[orderable]
    agree = np.sign(dy) == np.sign(dp)
    ties = dp == 0
    score = float((agree & ~ties).sum() + 0.5 * ties.sum()) / total
    return score, total


def _rank_correlations(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(spearman, kendall) within one ligand; NaN when either side is constant."""
    if len(y) < MIN_ROWS_FOR_RANK or np.all(y == y[0]) or np.all(p == p[0]):
        return float("nan"), float("nan")
    try:
        from scipy.stats import kendalltau, spearmanr
    except ImportError:  # pragma: no cover - scipy is a hard dependency of the repo
        return float("nan"), float("nan")
    rho = spearmanr(y, p).statistic
    tau = kendalltau(y, p).statistic
    return float(rho), float(tau)


def decompose_level_shape(
    truth: Iterable[float],
    prediction: Iterable[float],
    ligand: Iterable,
    *,
    min_rows_for_shape: int = MIN_ROWS_FOR_SHAPE,
    seed: int = 20260819,
) -> LevelShapeDecomposition:
    """Split the error of each held-out ligand into a level offset and a shape.

    Returns a :class:`LevelShapeDecomposition` whose ``summary`` carries, among
    others, ``offset_mae``, ``shape_mae``, ``shape_r2``, ``macro_mae``
    (equal weight per ligand), ``sse_total``, ``sse_centred``, ``sse_offset``
    (the last three satisfy ``sse_total == sse_centred + sse_offset`` exactly),
    ``rank_spearman``, ``sign_accuracy`` and the counts behind each.

    ``macro_mae`` here is per *ligand*.  The primary metric of the study is macro
    per ECFP **cluster**; use :func:`gen6_metric_table` for that, and read this
    function as the per-ligand diagnostic it is.
    """
    y = np.asarray(list(truth), dtype=float)
    p = np.asarray(list(prediction), dtype=float)
    lig = np.asarray([str(x) for x in ligand])
    if not (len(y) == len(p) == len(lig)):
        raise ValueError("truth, prediction and ligand must have equal length")
    if len(y) == 0:
        raise ValueError("no rows to decompose")
    if not np.isfinite(p).all():
        raise ValueError(f"{int((~np.isfinite(p)).sum())} non-finite predictions")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for name in pd.unique(lig):
        mask = lig == name
        yl, pl = y[mask], p[mask]
        n = int(mask.sum())
        err = pl - yl
        bias = float(err.mean())
        yc, pc = yl - yl.mean(), pl - pl.mean()
        shape_ok = n >= min_rows_for_shape
        sign_acc, n_pairs = _sign_accuracy_within(yl, pl, rng=rng) if shape_ok else (float("nan"), 0)
        rho, tau = _rank_correlations(yl, pl)
        rows.append({
            "extractant": name,
            "n_rows": n,
            "mae": float(np.abs(err).mean()),
            "bias": bias,
            "offset_abs": abs(bias),
            "shape_mae": float(np.abs(pc - yc).mean()) if shape_ok else np.nan,
            "sse": float((err ** 2).sum()),
            "sse_centred": float(((pc - yc) ** 2).sum()),
            "sse_offset": float(n * bias ** 2),
            "sst_centred": float((yc ** 2).sum()),
            "y_sd": float(yl.std(ddof=0)),
            "rank_spearman": rho,
            "rank_kendall": tau,
            "sign_accuracy": sign_acc,
            "n_sign_pairs": n_pairs,
        })
    per_ligand = pd.DataFrame(rows).sort_values("extractant", ignore_index=True)

    shaped = per_ligand[per_ligand["n_rows"] >= min_rows_for_shape]
    sse_total = float(per_ligand["sse"].sum())
    sse_centred = float(per_ligand["sse_centred"].sum())
    sse_offset = float(per_ligand["sse_offset"].sum())
    sst_centred = float(per_ligand["sst_centred"].sum())
    ligand_mae = per_ligand["mae"].to_numpy(dtype=float)
    quartile = np.quantile(ligand_mae, 0.75) if len(ligand_mae) else np.nan
    abs_err = np.abs(p - y)

    summary = {
        # the two headline pieces
        "offset_mae": float(per_ligand["offset_abs"].mean()),
        "shape_mae": float(shaped["shape_mae"].mean()) if len(shaped) else np.nan,
        "shape_r2": (1.0 - sse_centred / sst_centred) if sst_centred > 0 else np.nan,
        "shape_r2_median_ligand": float(
            (1.0 - shaped["sse_centred"] / shaped["sst_centred"].replace(0.0, np.nan)).median()
        ) if len(shaped) else np.nan,
        # error budget — these three satisfy the identity exactly
        "sse_total": sse_total,
        "sse_centred": sse_centred,
        "sse_offset": sse_offset,
        "offset_share_of_sse": (sse_offset / sse_total) if sse_total > 0 else np.nan,
        # per-ligand aggregates
        "macro_mae_ligand": float(ligand_mae.mean()) if len(ligand_mae) else np.nan,
        "pooled_mae": float(abs_err.mean()),
        "median_ligand_mae": float(np.median(ligand_mae)) if len(ligand_mae) else np.nan,
        "worst_quartile_ligand_mae": float(ligand_mae[ligand_mae >= quartile].mean())
        if len(ligand_mae) else np.nan,
        "max_ligand_mae": float(ligand_mae.max()) if len(ligand_mae) else np.nan,
        # ordering quality
        "rank_spearman": float(per_ligand["rank_spearman"].mean(skipna=True)),
        "rank_kendall": float(per_ligand["rank_kendall"].mean(skipna=True)),
        "sign_accuracy": float(
            np.average(per_ligand.loc[per_ligand["n_sign_pairs"] > 0, "sign_accuracy"],
                       weights=per_ligand.loc[per_ligand["n_sign_pairs"] > 0, "n_sign_pairs"])
        ) if (per_ligand["n_sign_pairs"] > 0).any() else np.nan,
        # coverage
        "frac_within_0_5_log": float(np.mean(abs_err <= 0.5)),
        "frac_within_1_log": float(np.mean(abs_err <= 1.0)),
        # counts — never quote a metric without them
        "n_rows": int(len(y)),
        "n_ligands": int(len(per_ligand)),
        "n_ligands_with_shape": int(len(shaped)),
        "n_ligands_rank": int(per_ligand["rank_spearman"].notna().sum()),
    }
    return LevelShapeDecomposition(per_ligand=per_ligand, summary=summary)


def effective_sample_size(counts: Iterable[float]) -> float:
    """Kish effective n over unit sizes: ``(Σw)² / Σw²``.

    With 67 % of rows in one super-cluster, "n = 5,248 rows" is not 5,248
    independent observations; this is the number to print instead.
    """
    w = np.asarray([float(c) for c in counts], dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return float("nan")
    return float(w.sum() ** 2 / np.square(w).sum())


# --------------------------------------------------------------------------- #
# Hard chemistry endpoints
# --------------------------------------------------------------------------- #

def hard_chemistry_endpoints(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    similarity_column: str = "nn_reference_tanimoto",
    ligand_column: str = "extractant",
    cluster_column: str = "ecfp_cluster",
    supercluster_column: str = "tanimoto_cluster",
    target_column: str = LEVEL_TARGET_COLUMN,
    thresholds: Sequence[float] = HARD_CHEMISTRY_THRESHOLDS,
    seed: int = 20260819,
) -> pd.DataFrame:
    """Metrics restricted to test rows far from the *reference* training chemistry.

    The similarity column must be computed against a **fixed** reference cohort
    (in Experiment A: the BASE training set) for every arm, otherwise the subset
    moves with the arm and the arms are no longer compared on the same rows.
    A threshold with no rows yields a row of NaNs with ``n_rows = 0`` — never a
    silent omission.
    """
    if similarity_column not in frame.columns:
        raise KeyError(f"similarity column {similarity_column!r} is absent; "
                       f"have {sorted(frame.columns)[:8]}…")
    # A missing similarity means "distance to training chemistry unknown", which is
    # neither hard nor easy.  It must not be silently swallowed: `NaN < threshold` is
    # False, so a naive filter would drop such rows from every subset *including the
    # one that is supposed to be the whole test set*.  The "all" row therefore uses
    # the unfiltered frame, the threshold rows exclude unknown-distance rows, and the
    # count of them is reported on every row so the loss is visible.
    n_unknown = int(frame[similarity_column].isna().sum())
    out: list[dict] = []
    for threshold in [*thresholds, np.inf]:
        if np.isfinite(threshold):
            subset = frame[frame[similarity_column] < threshold]
        else:
            subset = frame
        label = "all" if not np.isfinite(threshold) else f"nn<{threshold:g}"
        record: dict = {
            "endpoint": label,
            "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
            "n_rows": int(len(subset)),
            "n_ligands": int(subset[ligand_column].nunique()) if len(subset) else 0,
            "n_ecfp_clusters": int(subset[cluster_column].nunique())
            if len(subset) and cluster_column in subset.columns else 0,
            "n_superclusters": int(subset[supercluster_column].nunique())
            if len(subset) and supercluster_column in subset.columns else 0,
            "n_rows_unknown_similarity": n_unknown,
        }
        if len(subset) == 0:
            record.update({k: np.nan for k in
                           ("macro_mae", "macro_mae_ligand", "pooled_mae", "offset_mae",
                            "shape_mae", "shape_r2", "median_ligand_mae",
                            "worst_quartile_ligand_mae", "frac_within_1_log")})
            out.append(record)
            continue
        decomposition = decompose_level_shape(
            subset[target_column], subset[prediction_column], subset[ligand_column], seed=seed)
        record.update({k: decomposition.summary[k] for k in
                       ("macro_mae_ligand", "pooled_mae", "offset_mae", "shape_mae", "shape_r2",
                        "median_ligand_mae", "worst_quartile_ligand_mae", "frac_within_1_log",
                        "frac_within_0_5_log", "rank_spearman", "sign_accuracy")})
        if cluster_column in subset.columns:
            per_cluster = (subset.assign(_e=np.abs(subset[prediction_column] - subset[target_column]))
                           .groupby(cluster_column)["_e"].mean())
            record["macro_mae"] = float(per_cluster.mean())
            record["n_eff_pooled_rows"] = effective_sample_size(
                subset.groupby(cluster_column).size())
            record["n_macro_units"] = int(subset[cluster_column].nunique())
        else:
            record["macro_mae"] = np.nan
            record["n_eff_pooled_rows"] = np.nan
            record["n_macro_units"] = 0
        out.append(record)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# OOD layer
# --------------------------------------------------------------------------- #

def ood_statistics(
    frame: pd.DataFrame,
    *,
    neighbour_table: pd.DataFrame,
    ligand_column: str = "extractant",
    neighbour_key: str = "extractant",
    prediction_spread: Mapping[str, Sequence[float]] | pd.Series | None = None,
) -> pd.DataFrame:
    """Attach per-row out-of-distribution statistics.

    ``neighbour_table`` is the frame returned by
    :meth:`gen6.chemistry.ChemistryMap.nearest_neighbour` for the *training*
    cohort of the arm in question: one row per query extractant with
    ``nn_tanimoto``, ``n_above_0_5``, ``n_above_0_7``, ``n_above_0_8`` and
    ``supercluster_support``.  ``prediction_spread`` optionally carries the
    per-row model dispersion (the sd across the trees of the forest, or across
    folds), which is the second half of an abstention rule: chemistry can be
    close and the model still unsure, and vice versa.

    A model that knows where it is unsupported is more useful than one that
    always emits a number, so these columns travel with every prediction.
    """
    if neighbour_key not in neighbour_table.columns:
        raise KeyError(f"neighbour_table has no key column {neighbour_key!r}; "
                       f"have {list(neighbour_table.columns)}")
    columns = ["nn_tanimoto", "n_above_0_5", "n_above_0_7", "n_above_0_8", "supercluster_support"]
    available = [c for c in columns if c in neighbour_table.columns]
    # Keyed by NAME, never by position: a caller handing over a re-ordered frame
    # would otherwise get a silently wrong join.
    keyed = neighbour_table.set_index(neighbour_key)[available]
    out = frame.copy()
    for column in available:
        out[f"ood__{column}"] = out[ligand_column].map(keyed[column])
    if prediction_spread is not None:
        spread = (pd.Series(prediction_spread) if not isinstance(prediction_spread, pd.Series)
                  else prediction_spread)
        out["ood__prediction_sd"] = np.asarray(spread, dtype=float)
    return out


def ood_calibration_table(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    target_column: str = LEVEL_TARGET_COLUMN,
    similarity_column: str = "ood__nn_tanimoto",
    bins: Sequence[float] = (0.0, 0.4, 0.6, 0.8, 1.0001),
    ligand_column: str = "extractant",
) -> pd.DataFrame:
    """Error as a function of distance to training chemistry — the abstention curve."""
    work = frame.dropna(subset=[similarity_column]).copy()
    if work.empty:
        return pd.DataFrame(columns=["bin", "n_rows", "n_ligands", "pooled_mae",
                                     "macro_mae_ligand", "offset_mae", "shape_mae"])
    work["_bin"] = pd.cut(work[similarity_column], bins=list(bins), right=False, include_lowest=True)
    rows = []
    for label, block in work.groupby("_bin", observed=True):
        decomposition = decompose_level_shape(
            block[target_column], block[prediction_column], block[ligand_column])
        rows.append({
            "bin": str(label),
            "n_rows": int(len(block)),
            "n_ligands": int(block[ligand_column].nunique()),
            # named `pooled_mae`, never a bare `mae`: this is a row mean, while every
            # other MAE in a gen6 report is a macro over chemistry units, and the two
            # differ by ~20 % on the same rows.  A column called "mae" sitting beside
            # macro tables invites exactly that confusion.
            "pooled_mae": float(np.abs(block[prediction_column] - block[target_column]).mean()),
            "macro_mae_ligand": decomposition.summary["macro_mae_ligand"],
            "offset_mae": decomposition.summary["offset_mae"],
            "shape_mae": decomposition.summary["shape_mae"],
            "shape_r2": decomposition.summary["shape_r2"],
            "mean_prediction_sd": float(block["ood__prediction_sd"].mean())
            if "ood__prediction_sd" in block.columns else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Arm-level table and the chemistry-unit bootstrap
# --------------------------------------------------------------------------- #

#: Statistics the per-unit table carries.
#:
#: **Only ``mae`` is an exact bootstrap of the leaderboard's macro value.** For it,
#: the per-unit statistic is a mean over the unit's rows and the macro is the mean
#: over units, so resampling units reproduces the macro exactly.  ``offset_mae``
#: and ``shape_mae`` are means over *ligands* within a unit, so the unit-level and
#: leaderboard-level values differ slightly when units hold different numbers of
#: ligands (measured: 0.004–0.014 on this cohort).  ``median_ligand_mae`` is a
#: median of medians and differs by up to 0.11 — and since 89 % of units hold
#: exactly one ligand it mostly degenerates to ``mae`` anyway, so it is **not** in
#: the default set.  An earlier comment here claimed all four were exact; they are
#: not, and the difference is 18 % of the shape effect.
UNIT_STATISTICS: tuple[str, ...] = ("mae", "offset_mae", "shape_mae")


def per_unit_statistics(
    predictions: pd.DataFrame,
    arms: Sequence[str],
    *,
    unit_column: str = "ecfp_cluster",
    ligand_column: str = "extractant",
    target_column: str = LEVEL_TARGET_COLUMN,
    prediction_prefix: str = "prediction_",
    seed: int = 20260819,
) -> pd.DataFrame:
    """Long table ``(unit, arm, statistic, value)`` — the input to the bootstrap.

    Every statistic is defined as a *mean over the unit's members*, which is what
    makes ``mean over resampled units`` an exact bootstrap of the macro value.
    """
    rows: list[dict] = []
    for unit, block in predictions.groupby(unit_column, sort=True):
        for arm in arms:
            column = f"{prediction_prefix}{arm}"
            if column not in block.columns:
                raise KeyError(f"missing prediction column {column!r}")
            decomposition = decompose_level_shape(
                block[target_column], block[column], block[ligand_column], seed=seed)
            summary = decomposition.summary
            rows.append({
                "unit": str(unit), "arm": arm,
                "mae": float(np.abs(block[column] - block[target_column]).mean()),
                "offset_mae": summary["offset_mae"],
                "shape_mae": summary["shape_mae"],
                "median_ligand_mae": summary["median_ligand_mae"],
                "n_rows": int(len(block)),
                "n_ligands": int(block[ligand_column].nunique()),
            })
    return pd.DataFrame(rows)


def paired_unit_bootstrap(
    per_unit: pd.DataFrame,
    comparisons: Mapping[str, tuple[str, str]],
    *,
    statistics: Sequence[str] = UNIT_STATISTICS,
    block_of_unit: Mapping[str, str] | None = None,
    replicates: int = 5000,
    seed: int = 8675309,
) -> pd.DataFrame:
    """Paired bootstrap of ``reference − candidate`` over independent chemistry units.

    ``per_unit`` comes from :func:`per_unit_statistics`.  ``block_of_unit`` maps
    each scoring unit to the block the folds actually held out (ECFP cluster →
    Tanimoto super-cluster under a chemotype hold-out); the *blocks* are
    resampled and every member unit travels with its block, because scoring
    units inside one held-out block are not independent.  One index matrix is
    drawn and shared by every comparison and statistic, so intervals are
    mutually comparable and do not depend on dict order.

    Units where a statistic is NaN for either arm (e.g. ``shape_mae`` for a unit
    of single-row ligands) are dropped from *that statistic only*, and the count
    that survived is reported.
    """
    if per_unit.empty:
        raise ValueError("per_unit is empty")
    units = sorted(per_unit["unit"].unique())
    blocks = {u: (block_of_unit or {}).get(u, u) for u in units}
    block_names = sorted(set(blocks.values()))
    members = {b: [i for i, u in enumerate(units) if blocks[u] == b] for b in block_names}
    member_index = [np.asarray(members[b], dtype=int) for b in block_names]

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(block_names), size=(replicates, len(block_names)))
    take = [np.concatenate([member_index[j] for j in row]) for row in picks]

    wide = {stat: per_unit.pivot_table(index="unit", columns="arm", values=stat, aggfunc="first")
            .reindex(units) for stat in statistics}

    out: list[dict] = []
    for label, (reference, candidate) in comparisons.items():
        for stat in statistics:
            table = wide[stat]
            if reference not in table.columns or candidate not in table.columns:
                continue
            delta = (table[reference] - table[candidate]).to_numpy(dtype=float)
            finite = np.isfinite(delta)
            if not finite.any():
                continue
            filled = np.where(finite, delta, 0.0)
            counts = finite.astype(float)
            sums = np.array([filled[t].sum() for t in take])
            ns = np.array([counts[t].sum() for t in take])
            draws = np.divide(sums, ns, out=np.full_like(sums, np.nan), where=ns > 0)
            point = float(delta[finite].mean())
            record = {
                "comparison": label, "reference": reference, "candidate": candidate,
                "statistic": stat, "point_delta": point,
                "ci95_low": float(np.nanquantile(draws, 0.025)),
                "ci95_high": float(np.nanquantile(draws, 0.975)),
                "p_worse_one_sided": float((1 + np.nansum(draws <= 0)) / (1 + replicates)),
                "units_improved": int(np.sum(delta[finite] > 0)),
                "units_total": int(finite.sum()),
                "bootstrap_blocks": int(len(block_names)),
                "bootstrap_replicates": int(replicates),
            }
            record.update(_robust_intervals(delta, finite, member_index, draws, point))
            out.append(record)
    return pd.DataFrame(out)


def _robust_intervals(
    delta: np.ndarray, finite: np.ndarray, member_index: Sequence[np.ndarray],
    draws: np.ndarray, point: float,
) -> dict:
    """BCa and cluster-robust companions to the percentile interval.

    The percentile interval is **not** trustworthy here and this is measured, not
    suspected: the scoring unit is the ECFP cluster (131 of them) while the
    resampling unit is the Tanimoto chemotype (79), and one chemotype holds 28 of
    the 131 scoring units — 21 % of the vote inside 1.3 % of the evidence — with a
    delta of the opposite sign to the population.  A double-bootstrap coverage
    simulation puts the real one-sided Type-I rate of the percentile interval at
    ~12.7 % rather than 2.5 % at the ``all`` endpoint (it is well behaved, ~2.9 %,
    on the hard-chemistry subset, where no block dominates).

    So three intervals are reported side by side and the report must not quote the
    percentile one alone:

    * **percentile** — kept for continuity with gen5;
    * **BCa** — bias-corrected and accelerated, jackknifed over *blocks*, which is
      what corrects for exactly this kind of one-block leverage;
    * **cluster-robust t** — a block-weighted mean with a cluster-robust standard
      error and ``n_blocks − 1`` degrees of freedom;
    * **block macro** — the delta with one vote per *block* instead of per unit,
      i.e. voting at the level the folds actually held out.
    """
    from scipy import stats

    values = np.where(finite, delta, 0.0)
    counts = finite.astype(float)
    out: dict = {}

    # --- block-level quantities ------------------------------------------- #
    block_sums, block_ns = [], []
    for members in member_index:
        block_sums.append(values[members].sum())
        block_ns.append(counts[members].sum())
    block_sums = np.asarray(block_sums, dtype=float)
    block_ns = np.asarray(block_ns, dtype=float)
    live = block_ns > 0
    if live.sum() < 2:
        return out
    block_means = np.divide(block_sums, block_ns, out=np.zeros_like(block_sums), where=block_ns > 0)
    out["block_macro_delta"] = float(block_means[live].mean())
    out["n_blocks_positive"] = int((block_means[live] > 0).sum())
    out["largest_block_unit_share"] = float(block_ns.max() / block_ns.sum())

    # --- cluster-robust t on the unit-weighted mean ------------------------ #
    total_n = block_ns.sum()
    residual = block_sums - block_ns * point
    se = float(np.sqrt(np.square(residual).sum()) / total_n) if total_n else np.nan
    dof = int(live.sum() - 1)
    if np.isfinite(se) and se > 0 and dof > 0:
        crit = float(stats.t.ppf(0.975, dof))
        out["cluster_robust_se"] = se
        out["cluster_robust_low"] = point - crit * se
        out["cluster_robust_high"] = point + crit * se
        out["cluster_robust_p_two_sided"] = float(2 * stats.t.sf(abs(point) / se, dof))

    # --- BCa, jackknifed over blocks --------------------------------------- #
    finite_draws = draws[np.isfinite(draws)]
    if finite_draws.size:
        proportion = float((finite_draws < point).mean())
        proportion = min(max(proportion, 1e-6), 1 - 1e-6)
        z0 = float(stats.norm.ppf(proportion))
        jack = np.array([
            (block_sums.sum() - block_sums[i]) / (block_ns.sum() - block_ns[i])
            for i in range(len(block_ns)) if (block_ns.sum() - block_ns[i]) > 0])
        centred = jack.mean() - jack
        denominator = 6.0 * (np.square(centred).sum() ** 1.5)
        acceleration = float((centred ** 3).sum() / denominator) if denominator > 0 else 0.0
        for tag, alpha in (("low", 0.025), ("high", 0.975)):
            z = stats.norm.ppf(alpha)
            adjusted = z0 + (z0 + z) / max(1e-9, 1 - acceleration * (z0 + z))
            out[f"bca_{tag}"] = float(np.quantile(finite_draws, np.clip(stats.norm.cdf(adjusted), 0, 1)))
        out["bca_z0"] = z0
        out["bca_acceleration"] = acceleration
    return out


def gen6_metric_table(
    predictions: pd.DataFrame,
    arms: Sequence[str],
    *,
    cluster_column: str = "ecfp_cluster",
    supercluster_column: str = "tanimoto_cluster",
    ligand_column: str = "extractant",
    target_column: str = LEVEL_TARGET_COLUMN,
    similarity_column: str | None = "nn_reference_tanimoto",
    prediction_prefix: str = "prediction_",
    thresholds: Sequence[float] = HARD_CHEMISTRY_THRESHOLDS,
    seed: int = 20260819,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(overall, per_ligand, hard_chemistry)`` for every arm.

    ``overall`` leads with the study's primary metric — macro MAE with one ECFP
    cluster one vote — and carries the co-primary offset MAE and the counts that
    make it interpretable (``n_ligands``, ``n_ecfp_clusters``, ``n_superclusters``,
    ``n_eff_pooled_rows``).  Pooled numbers are reported but never used for selection: the
    largest ligand holds ~30 % of rows.
    """
    y = predictions[target_column].to_numpy(dtype=float)
    overall_rows: list[dict] = []
    per_ligand_parts: list[pd.DataFrame] = []
    hard_parts: list[pd.DataFrame] = []
    # Kish n over ROWS PER CLUSTER.  This is the effective n of the *pooled*
    # (row-weighted) metric, and it is ~20x smaller than the macro metric's,
    # because macro weights each cluster equally: its unit count is the number of
    # clusters, and the load-bearing independence unit is the bootstrap block.
    # Reporting one number called "n_eff" beside macro_mae understated the macro
    # metric's precision; all three are now named.
    n_eff_pooled = effective_sample_size(predictions.groupby(cluster_column).size())

    for arm in arms:
        column = f"{prediction_prefix}{arm}"
        if column not in predictions.columns:
            raise KeyError(f"missing prediction column {column!r}")
        p = predictions[column].to_numpy(dtype=float)
        if not np.isfinite(p).all():
            raise ValueError(f"arm {arm!r} has {int((~np.isfinite(p)).sum())} missing predictions")
        decomposition = decompose_level_shape(y, p, predictions[ligand_column], seed=seed)
        per_cluster = (predictions.assign(_e=np.abs(p - y))
                       .groupby(cluster_column)["_e"].mean())
        sse = float(((p - y) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        overall_rows.append({
            "arm": arm,
            "macro_mae": float(per_cluster.mean()),
            "offset_mae": decomposition.summary["offset_mae"],
            "shape_mae": decomposition.summary["shape_mae"],
            "shape_r2": decomposition.summary["shape_r2"],
            "macro_mae_ligand": decomposition.summary["macro_mae_ligand"],
            "pooled_mae": decomposition.summary["pooled_mae"],
            "pooled_r2": (1.0 - sse / sst) if sst > 0 else np.nan,
            "offset_share_of_sse": decomposition.summary["offset_share_of_sse"],
            "median_ligand_mae": decomposition.summary["median_ligand_mae"],
            "worst_quartile_ligand_mae": decomposition.summary["worst_quartile_ligand_mae"],
            "rank_spearman": decomposition.summary["rank_spearman"],
            "sign_accuracy": decomposition.summary["sign_accuracy"],
            "frac_within_0_5_log": decomposition.summary["frac_within_0_5_log"],
            "frac_within_1_log": decomposition.summary["frac_within_1_log"],
            "prediction_dispersion_ratio": float(np.std(p) / np.std(y)) if np.std(y) > 0 else np.nan,
            "n_rows": int(len(y)),
            "n_ligands": int(predictions[ligand_column].nunique()),
            "n_ecfp_clusters": int(predictions[cluster_column].nunique()),
            "n_superclusters": int(predictions[supercluster_column].nunique())
            if supercluster_column in predictions.columns else np.nan,
            "n_eff_pooled_rows": n_eff_pooled,
            "n_macro_units": int(predictions[cluster_column].nunique()),
        })
        table = decomposition.per_ligand.assign(arm=arm)
        per_ligand_parts.append(table)
        if similarity_column and similarity_column in predictions.columns:
            hard = hard_chemistry_endpoints(
                predictions, prediction_column=column, similarity_column=similarity_column,
                ligand_column=ligand_column, cluster_column=cluster_column,
                supercluster_column=supercluster_column, target_column=target_column,
                thresholds=thresholds, seed=seed).assign(arm=arm)
            hard_parts.append(hard)

    overall = pd.DataFrame(overall_rows)
    per_ligand = pd.concat(per_ligand_parts, ignore_index=True) if per_ligand_parts else pd.DataFrame()
    hard_chemistry = pd.concat(hard_parts, ignore_index=True) if hard_parts else pd.DataFrame()
    return overall, per_ligand, hard_chemistry
