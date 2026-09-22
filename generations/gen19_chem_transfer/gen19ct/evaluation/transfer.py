"""``evaluation/transfer.py`` -- the statistical procedure and decision rule R19 (section 8) with the
section 9 margin formula and the signal-injection helper.

No model is fitted here.  Every function takes per-unit values (or a frame and a statistic) computed
elsewhere.

Bootstrap (section 8)
---------------------
:func:`paired_cluster_bootstrap` -- ``Delta = metric(comparator) - metric(candidate)`` (sign flipped for a
higher-is-better metric, so a positive Delta always favours the candidate) of the design's macro statistic,
the equal-weight mean over scored units.  Clusters (sorted by their text label) are drawn with replacement,
``numpy.random.default_rng(seed).integers(0, G, size=(n_resamples, G))`` -- one matrix per call -- and every
unit of a drawn cluster enters as often as its cluster was drawn; the macro is recomputed on each resample.
Defaults: 10,000 resamples, seed 19, 95 %.  :func:`unclustered_bootstrap` resamples units (a reference that
never decides); :func:`cluster_bootstrap_statistic` does the same cluster resampling for any statistic of a
frame (e.g. a Spearman across cells, section 9 S1(e) / section 12).

:class:`BootstrapResult` gives the percentile and BCa intervals (BCa: bias from the mid-rank proportion of
draws below the point estimate, acceleration from the jackknife over clusters; undefined -> NaN, never
clipped), the two-sided bootstrap p ``min(1, 2 min(P*(Delta* <= 0), P*(Delta* >= 0)))`` (the gen13
convention), ``mde_80 = 2.80 x bootstrap SD`` (ddof 1), and the leave-one-cluster-out deltas (the same values
feed the BCa jackknife and R19 item 5).

R19 (:func:`r19`)
-----------------
1. point Delta >= margin;
2. percentile and BCa 95 % intervals exclude 0 (lower bound > 0) under every registered cluster unit of the
   design, from a registered bootstrap (10,000 resamples, seed 19, clustered); V6 checks the percentile
   interval only (section 8: BCa unreliable at 13 clusters);
3. two-sided p < 0.05 under **every** registered cluster unit, each from its own registered bootstrap (resolved by
   the orchestrator 2026-09-15, the conservative reading; the item detail prints p per cluster unit);
4. Delta > 0 in >= 4 of the 5 discovery seeds (stage ``discovery``) or 5 of 5 (``confirmation``); VACUOUS for a
   deterministic contrast;
5. no leave-one-cluster-out Delta <= 0 under any registered cluster unit;
6. Delta > 0 in every registered sensitivity of the design (:data:`REGISTERED_SENSITIVITIES`).  A sensitivity
   the caller marks :data:`UNTESTABLE` (e.g. V5-P dropped under its own rule) cannot pass: the verdict is then
   UNDECIDED unless another item fails.  A missing or non-finite sensitivity FAILS.  The wildcard-copy scoring filter
   ``wildcard_copies_excluded_scoring`` (any partner) is a registered sensitivity of V1, V5, V5-P and V5-PAIR (section 2
   resolution, extended to V5-P and V5-PAIR by the section 8 R19 item 6 resolution, 2026-09-15); the strict-partner
   filter stays exploratory (:data:`EXPLORATORY_SENSITIVITIES`, never deciding).

Resolutions of 2026-09-15 implemented here
------------------------------------------
* V0 F1 interval (section 3.6): :func:`seed_mean_cluster_bootstrap` -- seed-mean macro over publication groups, the
  same resampled groups applied to every seed; :func:`f1_check`.
* S1(c) redefined (section 9): :func:`s1c_half` (paired direction contrasts per yardstick and paired logSF MAE gains on
  identical pairs of one half and one fitted run) and :func:`s1c_paired_verdict` (confirmation rule over the 5 withheld
  seeds, :func:`s1c_seed_combination`, + selection-half counterweight).  The superseded ``s1c_check`` raises.  Before any
  pair enters a metric (section 2) :func:`guard_scored_pairs` runs ``leakage.pair_isolation_check``, requires every
  member in the test fold, requires fold-qualified member labels (``fold_id|row_id``) whose fold equals the pair's
  ``fold`` column, and passes the union of the members through the ``V6_TARGET_ROWS`` guard (a mask that does not cover
  every member label is refused).
  "Same fitted folds" (resolved by the orchestrator 2026-09-15, :data:`S1C_REGISTERED_FOLD_READING`): the B3x and B3i
  yardsticks are re-fitted on exactly the batched V5-PAIR folds the candidate is fitted on (closed-form;
  ``gen19ct.models.s1c_yardsticks.refit_lookup_yardsticks``), HEAVIER needs no fit.  :func:`check_s1c_fold_designs`
  refuses a fold design named without its design hash (``<stem>@<design_hash>``; the stem is the same for every seed), a
  yardstick (or pair set) whose outer fold design differs from the candidate's, and a candidate whose fold design is not
  a batched V5-PAIR design.  :func:`s1c_half` refuses pairs of another half; :func:`s1c_paired_verdict` refuses (never
  FAILs) a selection result not scored on the selection half on seed 104729.
  Seed combination at confirmation (:func:`s1c_seed_combination`): per withheld seed the cell-pair-macro Delta_Y, the
  mean over the 5 seeds, the percentile interval of a system-cluster bootstrap of that seed mean (10,000 resamples, seed
  19, the same resampled systems in every seed; :func:`seed_mean_nested_cluster_bootstrap`) and Delta_Y > 0 in 5 of 5
  seeds; the logSF MAE part is the seed mean of the per-seed gains on the identical confirmation pair set.
  Scoring-filter sensitivities of V5-PAIR (R19 item 6) re-score the same pairs through ``exclude_rows``, whose labels
  must be fold-qualified.
* Power check (section 8): :func:`prepare_injected_run` drops X(?) rows from the injected refits of every arm and
  keeps the scored rows identical to the un-injected run; :meth:`InjectedRun.fold_inputs` builds a fold's training rows
  from the fold's HIDDEN rows (input rows minus hidden minus X(?)), so a hidden-but-unscored row can never be refitted.

``verdict``: FAIL if any item fails, else UNDECIDED if an item is untestable, else PASS; ``passes`` is True only
for PASS.

TOST (:func:`tost`): the 90 % interval of Delta under the primary cluster, margin 0.05.  NO_DIFFERENCE when
the interval lies strictly inside (-0.05, 0.05); PASS (non-inferior) when only its lower bound exceeds
-0.05; UNDECIDED otherwise.  By default the interval is the union of the percentile and BCa 90 % intervals.

Readings fixed here (the registration is silent; the conservative option is taken)
------------------------------------------------------------------------------------
* "exclude 0" is read as lower bound > 0 (Delta must favour the candidate), and item 3 as p < 0.05 under
  every registered cluster unit;
* sensitivities of designs other than V5: the censoring-excluded scoring (section 8), the acid-grid-excluded
  rows and the Sr(III)-dropped training (section 2, "every registered contrast" / "every design"), plus the
  design's own section 3 sensitivities;
* Benjamini-Hochberg counts a contrast with a missing p toward m (its adjusted p stays NaN);
* the injected signal is standardised twice: u over metal states and v over systems (mean 0, SD 1), then
  s_row over the rows it is computed on, so kappa is the SD of the injection in log D.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gen19ct.evaluation import metrics as EM

# --------------------------------------------------------------------------------------------- #
# Registered constants
# --------------------------------------------------------------------------------------------- #

N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 19
ALPHA = 0.05
MDE_FACTOR = 2.80
P_THRESHOLD = 0.05
#: equivalence / non-inferiority margin (section 8)
EPSILON = 0.05
TOST_LEVEL = 0.90
#: section 9 margins
MARGIN_FLOOR = 0.05
RHO5 = 0.20
GAMMA5 = 0.05
ETA5 = 0.02
S1E_MAX_SPEARMAN = -0.10
REPLICATE_SD = 0.299
#: noise floor N0 = 0.299 * sqrt(2/pi) (section 9; 0.239 as printed)
N0 = REPLICATE_SD * math.sqrt(2.0 / math.pi)
#: discovery seeds (section 15); R19 item 4 needs one Delta per seed
N_SEEDS = 5
MIN_SEEDS_POSITIVE_DISCOVERY = 4
MIN_SEEDS_POSITIVE_CONFIRMATION = 5
STAGES: tuple[str, ...] = ("discovery", "confirmation")
#: signal-injection amplitudes (section 8) and the underpowered bound
KAPPAS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
KAPPA_MIN_INFORMATIVE = 0.25
#: reliability floor (section 8 "Reliability before correlation")
RELIABILITY_FLOOR = 0.3

UNTESTABLE = "UNTESTABLE"

#: cluster units per design (section 8 table); names are labels, the caller maps them to columns
REGISTERED_CLUSTER_UNITS: dict[str, tuple[str, ...]] = {
    "V5": ("system", "publication_group"), "V5-P": ("system", "publication_group"),
    "V5-PAIR": ("system", "publication_group"), "V1": ("publication_group",), "V2": ("metal_state",),
    "V3": ("system",), "V4": ("system",), "V7": ("system",), "V6": ("system",),
}
#: designs whose item 2 checks the percentile interval only
PERCENTILE_ONLY_DESIGNS: frozenset[str] = frozenset({"V6"})

COMMON_SENSITIVITIES: tuple[str, ...] = ("censoring_candidates_excluded_scoring", "acid_grid_rows_excluded",
                                         "sr_iii_dropped_training")
#: section 2 resolution (2026-09-15): value-matched copies the near-duplicate key cannot see, any partner, excluded
#: from scoring (kept in training) -- registered for V1 and V5, and for V5-P and V5-PAIR by the section 8 R19 item 6
#: resolution of the same day (task X leakage finding VR-05; not score-driven): 12 crossings on 11 scored row entries in
#: each V5-P design, 371 on 359 in each V5-PAIR design (``folds/INDEX.json``, ``folds/wildcard_copy_crossings.csv``).
#: The strict-partner filter stays exploratory for every design.
WILDCARD_COPY_SENSITIVITY = "wildcard_copies_excluded_scoring"
WILDCARD_COPY_STRICT_SENSITIVITY = "wildcard_copies_strict_excluded_scoring"
#: the designs whose R19 item 6 registers the wildcard-copy filter (any partner)
WILDCARD_COPY_DESIGNS: tuple[str, ...] = ("V1", "V5", "V5-P", "V5-PAIR")
REGISTERED_SENSITIVITIES: dict[str, tuple[str, ...]] = {
    "V5": ("loose_setting", "strict_setting", "V5-P", "V5-cell-only", "non_DGA_stratum", "HNO3_only_cells",
           "parent_structure_hiding", "acid_grid_rows_excluded", "censoring_candidates_excluded_scoring",
           "sr_iii_dropped_training", WILDCARD_COPY_SENSITIVITY),
    "V5-P": COMMON_SENSITIVITIES + (WILDCARD_COPY_SENSITIVITY,),
    "V5-PAIR": COMMON_SENSITIVITIES + (WILDCARD_COPY_SENSITIVITY,),
    "V1": COMMON_SENSITIVITIES + ("near_duplicate_key_groups_value_blind", "compilation_doi_groups",
                                  WILDCARD_COPY_SENSITIVITY),
    "V2": COMMON_SENSITIVITIES + ("state_level_hiding",),
    "V3": COMMON_SENSITIVITIES + ("parent_structure_components",),
    "V4": COMMON_SENSITIVITIES,
    "V6": COMMON_SENSITIVITIES + ("seven_system_setting",),
    "V7": COMMON_SENSITIVITIES,
}
#: printed beside R19, never deciding (R19 item 6 lists them as "unregistered, not deciding")
EXPLORATORY_SENSITIVITIES: dict[str, tuple[str, ...]] = {
    d: (WILDCARD_COPY_STRICT_SENSITIVITY,) for d in WILDCARD_COPY_DESIGNS
}
#: sensitivity name -> the scoring-filter label of the prediction tables (re-scoring of existing predictions)
SCORING_FILTER_OF_SENSITIVITY: dict[str, str] = {
    "censoring_candidates_excluded_scoring": "censoring_candidates_excluded_scoring",
    "acid_grid_rows_excluded": "acid_grid_rows_excluded_scoring",
    WILDCARD_COPY_SENSITIVITY: WILDCARD_COPY_SENSITIVITY,
    WILDCARD_COPY_STRICT_SENSITIVITY: WILDCARD_COPY_STRICT_SENSITIVITY,
}


def sensitivity_status(design: str, name: str) -> str:
    """``registered`` (decides R19 item 6), ``exploratory`` (printed, never decides) or ``unregistered``."""
    if design not in REGISTERED_SENSITIVITIES:
        raise ValueError(f"design {design!r} has no sensitivity registry")
    if name in REGISTERED_SENSITIVITIES[design]:
        return "registered"
    if name in EXPLORATORY_SENSITIVITIES.get(design, ()):
        return "exploratory"
    return "unregistered"


def scoring_filters(design: str, status: str = "registered") -> tuple[str, ...]:
    """The scoring-filter labels of the design's ``registered`` or ``exploratory`` sensitivities that re-score
    existing predictions (:data:`SCORING_FILTER_OF_SENSITIVITY`)."""
    if status not in ("registered", "exploratory"):
        raise ValueError("status must be registered or exploratory")
    names = REGISTERED_SENSITIVITIES[design] if status == "registered" else EXPLORATORY_SENSITIVITIES.get(design, ())
    return tuple(SCORING_FILTER_OF_SENSITIVITY[n] for n in names if n in SCORING_FILTER_OF_SENSITIVITY)


# --------------------------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------------------------- #

def draw_cluster_picks(n_clusters: int, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """The ``(n_resamples, n_clusters)`` matrix of drawn cluster positions (with replacement)."""
    if n_clusters < 1 or n_resamples < 1:
        raise ValueError("need at least one cluster and one resample")
    return np.random.default_rng(seed).integers(0, n_clusters, size=(int(n_resamples), int(n_clusters)))


def _cluster_counts(picks: np.ndarray, n_clusters: int) -> np.ndarray:
    b = picks.shape[0]
    flat = (picks + n_clusters * np.arange(b)[:, None]).ravel()
    return np.bincount(flat, minlength=b * n_clusters).reshape(b, n_clusters).astype(float)


@dataclass(frozen=True)
class BootstrapResult:
    """One bootstrap of one statistic under one resampling unit (see module docstring)."""

    contrast: str
    cluster_unit: str
    point: float
    draws: np.ndarray = field(repr=False)
    jackknife: pd.Series = field(repr=False)
    n_units: int
    n_clusters: int
    n_resamples: int
    seed: int
    clustered: bool
    higher_is_better: bool

    @property
    def finite_draws(self) -> np.ndarray:
        return self.draws[np.isfinite(self.draws)]

    @property
    def n_nan_draws(self) -> int:
        return int((~np.isfinite(self.draws)).sum())

    @property
    def sd(self) -> float:
        f = self.finite_draws
        return float(f.std(ddof=1)) if f.size > 1 else float("nan")

    @property
    def mde_80(self) -> float:
        return MDE_FACTOR * self.sd

    @property
    def p_two_sided(self) -> float:
        f = self.finite_draws
        if f.size == 0:
            return float("nan")
        return float(min(1.0, 2.0 * min(np.mean(f <= 0), np.mean(f >= 0))))

    @property
    def loco_deltas(self) -> pd.Series:
        """Leave-one-cluster-out values of the statistic (the jackknife over clusters)."""
        return self.jackknife

    def percentile_interval(self, level: float = 1 - ALPHA) -> tuple[float, float]:
        f = self.finite_draws
        if f.size < 2 or self.n_clusters < 2:
            return float("nan"), float("nan")
        lo, hi = np.quantile(f, [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(lo), float(hi)

    @property
    def bias_z0(self) -> float:
        from scipy.special import ndtri

        f = self.finite_draws
        if f.size == 0 or not np.isfinite(self.point):
            return float("nan")
        prop = (np.sum(f < self.point) + 0.5 * np.sum(f == self.point)) / f.size
        return float(ndtri(prop)) if 0 < prop < 1 else float("nan")

    @property
    def acceleration(self) -> float:
        jk = self.jackknife.to_numpy(dtype=float)
        jk = jk[np.isfinite(jk)]
        if jk.size < 3:
            return float("nan")
        centred = jk.mean() - jk
        den = 6.0 * float((centred ** 2).sum()) ** 1.5
        return float((centred ** 3).sum() / den) if den > 0 else 0.0

    def bca_interval(self, level: float = 1 - ALPHA) -> tuple[float, float]:
        from scipy.special import ndtr, ndtri

        f = self.finite_draws
        z0, acc = self.bias_z0, self.acceleration
        if f.size < 2 or self.n_clusters < 3 or not (np.isfinite(z0) and np.isfinite(acc)):
            return float("nan"), float("nan")
        out = []
        for q in ((1 - level) / 2, 1 - (1 - level) / 2):
            z = float(ndtri(q))
            denom = 1.0 - acc * (z0 + z)
            if denom <= 0:
                return float("nan"), float("nan")
            out.append(float(np.quantile(f, float(ndtr(z0 + (z0 + z) / denom)))))
        return out[0], out[1]

    def record(self, regime: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """One contrast row.  ``regime`` (e.g. design, variant, half, seed_set, status, comparator, candidate,
        averaging_unit) is copied in front; the bootstrap's own seed is ``bootstrap_seed``."""
        plo, phi = self.percentile_interval()
        blo, bhi = self.bca_interval()
        jk = self.jackknife.to_numpy(dtype=float)
        out = dict(regime or {})
        out.update({"contrast": self.contrast, "cluster_unit": self.cluster_unit, "clustered": self.clustered,
                    "decides": self.clustered, "point": self.point, "percentile_low": plo, "percentile_high": phi,
                    "bca_low": blo, "bca_high": bhi, "p_two_sided": self.p_two_sided, "bootstrap_sd": self.sd,
                    "mde_80": self.mde_80, "bias_z0": self.bias_z0, "acceleration": self.acceleration,
                    "loco_min": float(np.nanmin(jk)) if np.isfinite(jk).any() else float("nan"),
                    "loco_max": float(np.nanmax(jk)) if np.isfinite(jk).any() else float("nan"),
                    "n_units": self.n_units, "n_clusters": self.n_clusters, "n_resamples": self.n_resamples,
                    "bootstrap_seed": self.seed, "n_nan_draws": self.n_nan_draws,
                    "higher_is_better": self.higher_is_better})
        return out


def _aligned(comparator: pd.Series, candidate: pd.Series, clusters: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    for name, s in (("comparator", comparator), ("candidate", candidate), ("clusters", clusters)):
        if not isinstance(s, pd.Series):
            raise TypeError(f"{name} must be a pandas Series indexed by unit")
        if s.index.has_duplicates:
            raise ValueError(f"{name} index has duplicate units")
    idx = comparator.index
    if set(idx) != set(candidate.index) or set(idx) != set(clusters.index):
        raise ValueError("paired bootstrap: comparator, candidate and clusters must cover the same units")
    a = comparator.to_numpy(dtype=float)
    b = candidate.reindex(idx).to_numpy(dtype=float)
    c = clusters.reindex(idx)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("paired bootstrap: every unit needs a finite value in both arms (units are paired)")
    if c.isna().any():
        raise ValueError("paired bootstrap: every unit needs a cluster")
    return a, b, [str(x) for x in c.to_numpy(dtype=object)]


def paired_cluster_bootstrap(comparator: pd.Series, candidate: pd.Series, clusters: pd.Series, *,
                             higher_is_better: bool = False, n_resamples: int = N_RESAMPLES,
                             seed: int = BOOTSTRAP_SEED, contrast: str = "", cluster_unit: str = "",
                             _clustered: bool = True) -> BootstrapResult:
    """Paired cluster bootstrap of the macro Delta (see module docstring).  All three Series are indexed by
    scored unit; ``clusters`` gives each unit's cluster label."""
    a, b, labels = _aligned(comparator, candidate, clusters)
    d = (b - a) if higher_is_better else (a - b)
    names = sorted(set(labels))
    code = {n: i for i, n in enumerate(names)}
    cc = np.array([code[x] for x in labels], dtype=int)
    g = len(names)
    s_sum = np.bincount(cc, weights=d, minlength=g)
    n_sum = np.bincount(cc, minlength=g).astype(float)
    picks = draw_cluster_picks(g, n_resamples, seed)
    counts = _cluster_counts(picks, g)
    draws = (counts @ s_sum) / (counts @ n_sum)
    tot_s, tot_n = float(s_sum.sum()), float(n_sum.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        jk = (tot_s - s_sum) / (tot_n - n_sum)
    jk = np.where(tot_n - n_sum > 0, jk, np.nan)
    return BootstrapResult(contrast=contrast, cluster_unit=cluster_unit, point=float(d.mean()), draws=draws,
                           jackknife=pd.Series(jk, index=pd.Index(names, name="left_out_cluster")),
                           n_units=int(len(d)), n_clusters=g, n_resamples=int(n_resamples), seed=int(seed),
                           clustered=_clustered, higher_is_better=bool(higher_is_better))


def unclustered_bootstrap(comparator: pd.Series, candidate: pd.Series, *, higher_is_better: bool = False,
                          n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                          contrast: str = "") -> BootstrapResult:
    """Reference bootstrap over scored units (each unit its own cluster); printed, never decides."""
    clusters = pd.Series([f"unit:{u}" for u in comparator.index], index=comparator.index)
    return paired_cluster_bootstrap(comparator, candidate, clusters, higher_is_better=higher_is_better,
                                    n_resamples=n_resamples, seed=seed, contrast=contrast,
                                    cluster_unit="unit (unclustered reference; never decides)", _clustered=False)


def cluster_bootstrap_statistic(frame: pd.DataFrame, cluster_col: str, statistic: Callable[[pd.DataFrame], float], *,
                                n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED, contrast: str = "",
                                cluster_unit: str = "", higher_is_better: bool = False) -> BootstrapResult:
    """Cluster bootstrap of an arbitrary statistic of ``frame`` (rows of a drawn cluster enter as often as it is
    drawn; the same draw matrix as :func:`paired_cluster_bootstrap` for the same number of clusters and seed).
    The jackknife leaves one cluster out.  ``higher_is_better`` is only recorded: the statistic defines its sign."""
    if cluster_col not in frame.columns:
        raise KeyError(cluster_col)
    if frame[cluster_col].isna().any():
        raise ValueError("every row needs a cluster")
    labels = frame[cluster_col].astype(str).to_numpy()
    names = sorted(set(labels))
    members = [np.flatnonzero(labels == n) for n in names]
    g = len(names)
    picks = draw_cluster_picks(g, n_resamples, seed)
    draws = np.empty(len(picks))
    for r, row in enumerate(picks):
        draws[r] = float(statistic(frame.iloc[np.concatenate([members[j] for j in row])]))
    jk = np.empty(g)
    allpos = np.arange(len(frame))
    for j in range(g):
        keep = np.setdiff1d(allpos, members[j], assume_unique=True)
        jk[j] = float(statistic(frame.iloc[keep])) if keep.size else np.nan
    return BootstrapResult(contrast=contrast, cluster_unit=cluster_unit, point=float(statistic(frame)), draws=draws,
                           jackknife=pd.Series(jk, index=pd.Index(names, name="left_out_cluster")),
                           n_units=int(len(frame)), n_clusters=g, n_resamples=int(n_resamples), seed=int(seed),
                           clustered=True, higher_is_better=bool(higher_is_better))


# --------------------------------------------------------------------------------------------- #
# V0 (diagnostic): the F1 interval (section 3.6, resolved by the orchestrator 2026-09-15)
# --------------------------------------------------------------------------------------------- #

#: V0 averaging and cluster unit (section 3.6 resolution; V0 has no secondary cluster)
V0_CLUSTER_UNIT = "publication_group"


def _seed_table(x: pd.DataFrame, name: str, unit: str = "publication group") -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame: index = {unit}, columns = seeds")
    if x.index.has_duplicates or x.columns.has_duplicates:
        raise ValueError(f"{name}: duplicate {unit}s or seeds")
    return x


def _seed_mean_core(D: np.ndarray, codes: np.ndarray, n_clusters: int, n_resamples: int,
                    seed: int) -> tuple[float, np.ndarray, np.ndarray]:
    """The seed-mean macro of per-unit deltas ``D`` (units x seeds, NaN = unit not scored in that seed) with units
    nested in clusters ``codes`` (0..n_clusters-1): ``(point, draws, jackknife)``.  Each resample draws clusters once and
    applies the drawn clusters to every seed; per seed the macro is the equal-weight mean over the units of the drawn
    clusters (a cluster drawn twice enters twice), a seed without a unit in the resample makes that draw NaN; the
    statistic is the mean over seeds.  The jackknife leaves one cluster out of every seed."""
    present = np.isfinite(D)
    Dz = np.where(present, D, 0.0)
    P = present.astype(float)
    S = np.zeros((n_clusters, D.shape[1]))
    N = np.zeros((n_clusters, D.shape[1]))
    np.add.at(S, codes, Dz)
    np.add.at(N, codes, P)
    picks = draw_cluster_picks(n_clusters, n_resamples, seed)
    counts = _cluster_counts(picks, n_clusters)                         # (B, clusters)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_seed = (counts @ S) / (counts @ N)                          # (B, seeds)
        draws = per_seed.mean(axis=1)
        point_per_seed = S.sum(axis=0) / N.sum(axis=0)
        tot_s, tot_n = S.sum(axis=0)[None, :], N.sum(axis=0)[None, :]
        jk_seed = (tot_s - S) / (tot_n - N)                             # (clusters, seeds): leave that cluster out
    jk = np.where((tot_n - N > 0).all(axis=1), jk_seed.mean(axis=1), np.nan)
    return float(point_per_seed.mean()), draws, jk


def seed_mean_cluster_bootstrap(comparator: pd.DataFrame, candidate: pd.DataFrame, *, higher_is_better: bool = False,
                                n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED, contrast: str = "",
                                cluster_unit: str = V0_CLUSTER_UNIT) -> BootstrapResult:
    """The V0 seed-mean statistic and its publication-group cluster bootstrap (section 3.6 resolution).

    ``comparator`` / ``candidate``: per-unit values (e.g. MAE), index = publication group, one column per discovery
    seed; NaN = the group has no scored row in that seed (the NaN pattern must be identical in both arms).  Per seed,
    Delta_s = the equal-weight mean over that seed's scored groups of (comparator - candidate) (sign flipped for a
    higher-is-better metric); the statistic is the mean of Delta_s over the seeds.  Each resample draws publication
    groups with replacement ONCE (``draw_cluster_picks``: 10,000 resamples, seed 19) and applies the same drawn groups
    to every seed; a seed whose resample holds none of its groups makes that draw NaN.  The jackknife leaves one group
    out of every seed.  F1 reads the percentile 95 % interval."""
    a, b = _seed_table(comparator, "comparator"), _seed_table(candidate, "candidate")
    if not (set(a.index) == set(b.index) and list(a.columns) == list(b.columns)):
        raise ValueError("comparator and candidate must cover the same publication groups and seeds")
    b = b.reindex(a.index)
    A, Bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    if not np.array_equal(np.isfinite(A), np.isfinite(Bv)):
        raise ValueError("comparator and candidate must be scored on the same groups in every seed (paired)")
    if A.shape[1] < 1 or not np.isfinite(A).any(axis=0).all():
        raise ValueError("every seed needs at least one scored publication group")
    order = np.argsort(np.array([str(g) for g in a.index], dtype=object), kind="stable")
    names = [str(a.index[i]) for i in order]
    D = ((Bv - A) if higher_is_better else (A - Bv))[order]           # (groups, seeds); each group its own cluster
    g = len(names)
    point, draws, jk = _seed_mean_core(D, np.arange(g), g, n_resamples, seed)
    return BootstrapResult(contrast=contrast, cluster_unit=cluster_unit, point=point,
                           draws=draws, jackknife=pd.Series(jk, index=pd.Index(names, name="left_out_cluster")),
                           n_units=int(np.isfinite(D).any(axis=1).sum()), n_clusters=g, n_resamples=int(n_resamples),
                           seed=int(seed), clustered=True, higher_is_better=bool(higher_is_better))


def seed_mean_nested_cluster_bootstrap(comparator: pd.DataFrame, candidate: pd.DataFrame, clusters: pd.Series, *,
                                       higher_is_better: bool = False, n_resamples: int = N_RESAMPLES,
                                       seed: int = BOOTSTRAP_SEED, contrast: str = "",
                                       cluster_unit: str = "system") -> BootstrapResult:
    """The seed-mean statistic of scored units NESTED in clusters and its cluster bootstrap (section 9 S1(c) seed
    combination at confirmation: cell pairs nested in systems).

    ``comparator`` / ``candidate``: per-unit values, index = scored unit (e.g. cell pair), one column per seed; NaN = the
    unit is not scored in that seed (the NaN pattern must be identical in both arms).  ``clusters``: unit -> cluster
    label.  Per seed, Delta_s = the equal-weight mean over that seed's scored units of (comparator - candidate) (sign
    flipped for a higher-is-better metric); the statistic is the mean of Delta_s over the seeds.  Each resample draws
    clusters with replacement ONCE (``draw_cluster_picks``: 10,000 resamples, seed 19 by default) and applies the same
    drawn clusters to every seed; every unit of a drawn cluster enters as often as the cluster was drawn.  The jackknife
    leaves one cluster out of every seed.  With one unit per cluster it equals :func:`seed_mean_cluster_bootstrap`."""
    a = _seed_table(comparator, "comparator", "scored unit")
    b = _seed_table(candidate, "candidate", "scored unit")
    if not (set(a.index) == set(b.index) and list(a.columns) == list(b.columns)):
        raise ValueError("comparator and candidate must cover the same units and seeds (in the same seed order)")
    if not isinstance(clusters, pd.Series) or clusters.index.has_duplicates:
        raise TypeError("clusters must be a Series indexed by unique scored unit")
    if not set(a.index) <= set(clusters.index) or clusters.reindex(a.index).isna().any():
        raise ValueError("every scored unit needs a cluster")
    b = b.reindex(a.index)
    A, Bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    if not np.array_equal(np.isfinite(A), np.isfinite(Bv)):
        raise ValueError("comparator and candidate must be scored on the same units in every seed (paired)")
    if A.shape[1] < 1 or not np.isfinite(A).any(axis=0).all():
        raise ValueError("every seed needs at least one scored unit")
    labels = clusters.reindex(a.index).astype(str).to_numpy(dtype=object)
    names = sorted(set(labels))
    code = {n: i for i, n in enumerate(names)}
    codes = np.array([code[x] for x in labels], dtype=int)
    D = (Bv - A) if higher_is_better else (A - Bv)
    point, draws, jk = _seed_mean_core(D, codes, len(names), n_resamples, seed)
    return BootstrapResult(contrast=contrast, cluster_unit=cluster_unit, point=point, draws=draws,
                           jackknife=pd.Series(jk, index=pd.Index(names, name="left_out_cluster")),
                           n_units=int(np.isfinite(D).any(axis=1).sum()), n_clusters=len(names),
                           n_resamples=int(n_resamples), seed=int(seed), clustered=True,
                           higher_is_better=bool(higher_is_better))


def f1_check(v0: BootstrapResult, *, s1a_passed: bool, v1: BootstrapResult) -> dict[str, Any]:
    """Section 10 F1 (gains only on random split): failure when

    * on V0 the candidate beats B3 with the 95 % interval excluding 0 -- point Delta > 0 and the percentile lower
      bound of :func:`seed_mean_cluster_bootstrap` > 0 (section 3.6 resolution);
    * S1(a) fails on V5-primary;
    * and on V1 the Delta (B3 - candidate) interval includes 0 or Delta <= 0.  Reading (INFERRED, not registered
      text): the V1 interval excludes 0 exactly as R19 item 2 reads it, percentile and BCa lower bounds > 0 under
      the publication-group cluster, so it "includes 0" when either does.

    ``v0`` / ``v1`` are the bootstraps of Delta = metric(B3) - metric(candidate) (lower-is-better MAE)."""
    if v0.cluster_unit != V0_CLUSTER_UNIT or not v0.clustered or v0.n_resamples != N_RESAMPLES or v0.seed != BOOTSTRAP_SEED:
        raise ValueError("the V0 F1 interval is the registered publication-group seed-mean bootstrap (10,000, seed 19)")
    plo0, phi0 = v0.percentile_interval()
    v0_beats = bool(np.isfinite(v0.point) and v0.point > 0 and np.isfinite(plo0) and plo0 > 0)
    plo1, phi1 = v1.percentile_interval()
    blo1, bhi1 = v1.bca_interval()
    v1_excludes = bool(np.isfinite(plo1) and plo1 > 0 and np.isfinite(blo1) and blo1 > 0)
    v1_null = bool(not v1_excludes or not (np.isfinite(v1.point) and v1.point > 0))
    failure = bool(v0_beats and not s1a_passed and v1_null)
    return {"failure": failure, "v0_candidate_beats_B3_interval_excludes_0": v0_beats, "s1a_failed": not s1a_passed,
            "v1_interval_includes_0_or_delta_le_0": v1_null, "v0_point": v0.point,
            "v0_percentile_95": (plo0, phi0), "v1_point": v1.point, "v1_percentile_95": (plo1, phi1),
            "v1_bca_95": (blo1, bhi1)}


# --------------------------------------------------------------------------------------------- #
# R19
# --------------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class R19Result:
    design: str
    stage: str
    contrast: str
    point: float
    margin: float
    verdict: str
    items: tuple[dict[str, Any], ...]

    @property
    def passes(self) -> bool:
        return self.verdict == "PASS"

    def item(self, number: int) -> dict[str, Any]:
        return next(i for i in self.items if i["item"] == number)

    def to_frame(self, regime: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """One row per item; ``regime`` fields (variant, half, seed_set, status, ...) are copied in front."""
        base = {k: v for k, v in (regime or {}).items() if k not in ("design", "stage")}
        rows = [{**base, "design": self.design, "stage": self.stage, "contrast": self.contrast, "point": self.point,
                 "margin": self.margin, "verdict": self.verdict, **i} for i in self.items]
        return pd.DataFrame(rows)


def _item(n: int, name: str, status: str, detail: str) -> dict[str, Any]:
    return {"item": n, "name": name, "status": status, "detail": detail}


def r19(*, design: str, stage: str, point: float, margin: float, bootstraps: Mapping[str, BootstrapResult],
        seed_deltas: Sequence[float] | None, deterministic: bool,
        sensitivity_deltas: Mapping[str, float | str],
        loco_deltas: Mapping[str, Sequence[float] | pd.Series] | None = None, contrast: str = "") -> R19Result:
    """Decision rule R19 of section 8 (items 1-6; see module docstring).

    ``bootstraps`` maps every registered cluster unit of the design (:data:`REGISTERED_CLUSTER_UNITS`) to its
    :class:`BootstrapResult`; ``loco_deltas`` (same keys) defaults to each result's leave-one-cluster-out deltas;
    ``seed_deltas`` holds the five per-seed Deltas (ignored when ``deterministic``); ``sensitivity_deltas`` maps
    every registered sensitivity name to its Delta or :data:`UNTESTABLE`."""
    if design not in REGISTERED_CLUSTER_UNITS:
        raise ValueError(f"design {design!r} has no registered cluster unit")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if not (np.isfinite(margin) and margin >= 0):
        raise ValueError("margin must be a finite number >= 0")
    units = REGISTERED_CLUSTER_UNITS[design]
    items: list[dict[str, Any]] = []

    # 1. margin
    ok1 = bool(np.isfinite(point) and point >= margin)
    items.append(_item(1, "point_estimate_at_least_margin", "PASS" if ok1 else "FAIL",
                       f"Delta={point:.6g}, margin={margin:.6g}"))

    # 2. intervals under every registered cluster unit; 3. p < 0.05 under EVERY registered cluster unit (resolved
    #    2026-09-15: the conservative reading) -- each unit needs its own registered bootstrap
    fail2, fail3, p_by_unit = [], [], []
    for u in units:
        br = bootstraps.get(u)
        if br is None:
            fail2.append(f"{u}: no bootstrap")
            fail3.append(f"{u}: no bootstrap")
            continue
        if not br.clustered or br.n_resamples != N_RESAMPLES or br.seed != BOOTSTRAP_SEED:
            msg = (f"{u}: unregistered bootstrap (clustered={br.clustered}, n_resamples={br.n_resamples}, "
                   f"seed={br.seed})")
            fail2.append(msg)
            fail3.append(msg)
            continue
        same_point = bool(np.isfinite(br.point) and math.isclose(br.point, point, rel_tol=1e-9, abs_tol=1e-12))
        if not same_point:
            msg = f"{u}: bootstrap point {br.point:.6g} differs from Delta {point:.6g}"
            fail2.append(msg)
            fail3.append(msg)
        plo, _ = br.percentile_interval()
        if not (np.isfinite(plo) and plo > 0):
            fail2.append(f"{u}: percentile low {plo:.6g}")
        if design not in PERCENTILE_ONLY_DESIGNS:
            blo, _ = br.bca_interval()
            if not (np.isfinite(blo) and blo > 0):
                fail2.append(f"{u}: BCa low {blo:.6g}")
        p = br.p_two_sided
        p_by_unit.append(f"{u}: p={p:.6g}")
        if not (np.isfinite(p) and p < P_THRESHOLD):
            fail3.append(f"{u}: p={p:.6g} (not < {P_THRESHOLD:g})")
    items.append(_item(2, "intervals_exclude_zero_every_cluster_unit", "FAIL" if fail2 else "PASS",
                       "; ".join(fail2) or f"cluster units {list(units)}"
                       + (" (percentile only)" if design in PERCENTILE_ONLY_DESIGNS else "")))
    items.append(_item(3, "two_sided_p_below_0.05_every_cluster_unit", "FAIL" if fail3 else "PASS",
                       "; ".join(fail3) or ("every registered cluster unit: " + "; ".join(p_by_unit))))

    # 4. seeds
    if deterministic:
        items.append(_item(4, "seed_sign_agreement", "VACUOUS", "deterministic contrast: one seed value"))
    else:
        need = MIN_SEEDS_POSITIVE_DISCOVERY if stage == "discovery" else MIN_SEEDS_POSITIVE_CONFIRMATION
        sd = [] if seed_deltas is None else [float(x) for x in seed_deltas]
        if len(sd) != N_SEEDS:
            items.append(_item(4, "seed_sign_agreement", "FAIL", f"expected {N_SEEDS} seed deltas, got {len(sd)}"))
        else:
            n_pos = int(sum(1 for x in sd if np.isfinite(x) and x > 0))
            items.append(_item(4, "seed_sign_agreement", "PASS" if n_pos >= need else "FAIL",
                               f"{n_pos} of {N_SEEDS} seeds positive; {stage} needs {need}"))

    # 5. leave one cluster out
    fail5 = []
    for u in units:
        if loco_deltas is not None and u in loco_deltas:
            vals = np.asarray(list(loco_deltas[u]), dtype=float)
        elif u in bootstraps:
            vals = bootstraps[u].loco_deltas.to_numpy(dtype=float)
        else:
            fail5.append(f"{u}: no leave-one-cluster-out deltas")
            continue
        if vals.size == 0 or not np.isfinite(vals).all():
            fail5.append(f"{u}: {vals.size} deltas, non-finite or empty")
        elif (vals <= 0).any():
            fail5.append(f"{u}: {int((vals <= 0).sum())} of {vals.size} deltas <= 0 (min {vals.min():.6g})")
    items.append(_item(5, "leave_one_cluster_out_stays_positive", "FAIL" if fail5 else "PASS",
                       "; ".join(fail5) or f"cluster units {list(units)}"))

    # 6. registered sensitivities
    fail6, untestable = [], []
    for name in REGISTERED_SENSITIVITIES[design]:
        if name not in sensitivity_deltas:
            fail6.append(f"{name}: missing")
            continue
        v = sensitivity_deltas[name]
        if isinstance(v, str):
            if v == UNTESTABLE:
                untestable.append(name)
            else:
                fail6.append(f"{name}: {v!r}")
            continue
        v = float(v)
        if not (np.isfinite(v) and v > 0):
            fail6.append(f"{name}: Delta={v:.6g}")
    extra = sorted(set(sensitivity_deltas) - set(REGISTERED_SENSITIVITIES[design]))
    detail = "; ".join(fail6 + [f"{n}: UNTESTABLE" for n in untestable]) or "all registered sensitivities positive"
    expl = [n for n in extra if n in EXPLORATORY_SENSITIVITIES.get(design, ())]
    unreg = [n for n in extra if n not in expl]
    if expl:
        detail += f" (exploratory, not deciding: {expl})"
    if unreg:
        detail += f" (unregistered, not deciding: {unreg})"
    status6 = "FAIL" if fail6 else ("UNTESTABLE" if untestable else "PASS")
    items.append(_item(6, "positive_in_every_registered_sensitivity", status6, detail))

    statuses = [i["status"] for i in items]
    verdict = "FAIL" if "FAIL" in statuses else ("UNDECIDED" if "UNTESTABLE" in statuses else "PASS")
    return R19Result(design=design, stage=stage, contrast=contrast, point=float(point), margin=float(margin),
                     verdict=verdict, items=tuple(items))


# --------------------------------------------------------------------------------------------- #
# TOST, multiplicity, margins
# --------------------------------------------------------------------------------------------- #

def tost(result: BootstrapResult, *, epsilon: float = EPSILON, interval: str = "both") -> dict[str, Any]:
    """Equivalence / non-inferiority from the 90 % bootstrap interval of Delta under the primary cluster.

    ``interval``: ``percentile``, ``bca`` or ``both`` (the union of the two; the default)."""
    if not result.clustered:
        raise ValueError("TOST uses the primary-cluster bootstrap, not the unclustered reference")
    if interval not in ("percentile", "bca", "both"):
        raise ValueError("interval must be percentile, bca or both")
    plo, phi = result.percentile_interval(TOST_LEVEL)
    blo, bhi = result.bca_interval(TOST_LEVEL)
    if interval == "percentile":
        lo, hi = plo, phi
    elif interval == "bca":
        lo, hi = blo, bhi
    else:
        vals = [plo, phi, blo, bhi]
        lo, hi = (min(plo, blo), max(phi, bhi)) if all(np.isfinite(vals)) else (float("nan"), float("nan"))
    finite = bool(np.isfinite(lo) and np.isfinite(hi))
    equivalent = finite and lo > -epsilon and hi < epsilon
    non_inferior = finite and lo > -epsilon
    verdict = "NO_DIFFERENCE" if equivalent else ("PASS" if non_inferior else "UNDECIDED")
    return {"verdict": verdict, "non_inferior": non_inferior, "equivalent": equivalent, "low_90": lo, "high_90": hi,
            "epsilon": epsilon, "interval": interval, "point": result.point, "cluster_unit": result.cluster_unit}


def benjamini_hochberg(p: Sequence[float] | np.ndarray, *, nan_counts_toward_m: bool = True) -> np.ndarray:
    """BH step-up adjusted p (monotone, capped at 1).  A NaN p stays NaN; by default it still counts toward m."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ((p[ok] < 0) | (p[ok] > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    k = int(ok.sum())
    if k == 0:
        return out
    m = int(p.size) if nan_counts_toward_m else k
    vals = p[ok]
    order = np.argsort(vals, kind="stable")
    ranked = vals[order] * m / np.arange(1, k + 1)
    adj = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    res = np.empty(k)
    res[order] = adj
    out[ok] = res
    return out


def delta5(l5: float, *, rho5: float = RHO5, n0: float = N0, floor: float = MARGIN_FLOOR) -> float:
    """Section 9 S1(a) margin ``delta5 = max(0.05, rho5 * (L5 - N0))``, ``N0 = 0.299 sqrt(2/pi)``."""
    if not np.isfinite(l5):
        raise ValueError("L5 must be finite")
    return float(max(floor, rho5 * (float(l5) - n0)))


def stronger_lookup(macro_mae: Mapping[str, float], order: Sequence[str] = ("B3x", "B3i")) -> tuple[str, float]:
    """The V5 lookup comparator: the one of B3x / B3i with the lower selection-half macro MAE (a tie keeps the
    first in ``order``)."""
    vals = [(name, float(macro_mae[name])) for name in order]
    if not all(np.isfinite(v) for _, v in vals):
        raise ValueError("both lookup macro MAEs must be finite")
    best = min(vals, key=lambda nv: nv[1])
    return best


def s1c_check(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Superseded: the fixed-number S1(c) reading ("exceeds DIR5") was redefined by the orchestrator on 2026-09-15 as
    a paired contrast on identical pairs of the half being judged.  Use :func:`s1c_half` and
    :func:`s1c_paired_verdict`."""
    raise NotImplementedError("S1(c) was redefined (section 9, 2026-09-15): use s1c_half + s1c_paired_verdict")


# --------------------------------------------------------------------------------------------- #
# S1(c) as redefined (section 9, orchestrator 2026-09-15): paired contrasts on identical pairs of one half
# --------------------------------------------------------------------------------------------- #

#: the three direction yardsticks Y of S1(c)
S1C_YARDSTICKS: tuple[str, ...] = ("HEAVIER", "B3x", "B3i")
#: the lookup whose derived logSF MAE the candidate must beat (the V5 comparator fixed pre-seal: B3i)
S1C_LOOKUP = "B3i"
#: the yardsticks re-fitted on the candidate's batched V5-PAIR folds (closed-form lookups) ...
S1C_REFIT_YARDSTICKS: tuple[str, ...] = ("B3x", "B3i")
#: ... and the rules on the pairs that need no fit (their fold design, when named, is never compared)
S1C_UNFITTED_RULES: tuple[str, ...] = ("HEAVIER", "FLAT")
#: selection-half counterweight: min_Y Delta_Y >= -0.02 on discovery seed 104729
S1C_SELECTION_MIN_DELTA = -0.02
S1C_SELECTION_SEED = 104729
S1C_HALVES: tuple[str, ...] = ("confirmation", "selection")
#: the public discovery seeds (section 15; equal to ``folds.io.DISCOVERY_SEEDS``, tested); the confirmation seeds are
#: withheld and never one of them
DISCOVERY_SEEDS: tuple[int, ...] = (104729, 130363, 155921, 196613, 262147)
#: the entries of ``fold_designs`` an S1(c) half must name: the pair set, the candidate and the two re-fitted lookups
S1C_FOLD_DESIGN_KEYS: tuple[str, ...] = ("pairs", "candidate", "B3x", "B3i")
#: the registered reading of "same fitted folds" (section 9 S1(c), resolved by the orchestrator 2026-09-15; not
#: score-driven)
S1C_READING_REFIT = "yardsticks_refit_on_candidate_batched_folds"
S1C_FOLD_READINGS: dict[str, str] = {
    S1C_READING_REFIT: "the B3x and B3i yardsticks are re-fitted (closed-form) on exactly the batched V5-PAIR folds the "
                       "candidate is fitted on -- in discovery the seed-104729 batched folds of section 3.1, at "
                       "confirmation the batched folds built by the same colouring rule with each withheld seed -- not "
                       "taken from the unbatched pre-seal folds; HEAVIER needs no fit (models.s1c_yardsticks)",
}
#: readings that were put to the orchestrator and are NOT registered (an S1(c) half under one of them is refused)
S1C_UNREGISTERED_FOLD_READINGS: dict[str, str] = {
    "same_pairs": "B3x / B3i kept on the unbatched V5-PAIR folds of section 3.1 and paired with the candidate by row "
                  "pair (task X finding VR-02 alternative; not registered)",
    "same_fold_design": "task X's provisional label of the reading now registered as "
                        f"{S1C_READING_REFIT!r} (superseded name)",
}
#: the registered reading (set by the orchestrator's resolution of 2026-09-15)
S1C_REGISTERED_FOLD_READING: str = S1C_READING_REFIT
#: the design and scheme prefix of the batched V5-PAIR fold files (``folds.cell_holdout.v5pair_batched_folds``)
S1C_BATCHED_DESIGN = "V5PAIR"
S1C_BATCHED_SCHEME_PREFIX = "batched"
#: the separator of fold-qualified row labels (``fold_id|row_id``, the pre-seal ``pair_frames`` convention)
FOLD_LABEL_SEP = "|"
#: the separator of a fold design id ``<stem>@<design_hash>`` (:func:`s1c_fold_design_id`)
FOLD_DESIGN_HASH_SEP = "@"
#: ``folds.io.design_hash``: a lowercase hex SHA-256.  A bare stem cannot identify the folds an arm was fitted on
#: (``folds.io.design_stem`` is the same for every seed; only the fold ids and so the design hash differ), so every
#: fitted S1(c) arm must name its design hash (task X verification, finding 5)
FOLD_DESIGN_HASH_CHARS = frozenset("0123456789abcdef")
FOLD_DESIGN_HASH_LEN = 64


def fold_qualified_label(fold_id: Any, row_id: Any) -> str:
    """``fold_id|row_id``: the label of one prediction of one fold (a row is scored in several V5-PAIR folds)."""
    f, r = str(fold_id), str(row_id)
    if not f or not r or FOLD_LABEL_SEP in f:
        raise ValueError(f"cannot fold-qualify fold {fold_id!r} / row {row_id!r}")
    return f"{f}{FOLD_LABEL_SEP}{r}"


def split_fold_label(label: Any) -> tuple[str, str]:
    """``(fold_id, row_id)`` of a fold-qualified label; raises ``ValueError`` on anything else."""
    if not isinstance(label, str) or FOLD_LABEL_SEP not in label:
        raise ValueError(f"not a fold-qualified label 'fold_id{FOLD_LABEL_SEP}row_id': {label!r}")
    f, r = label.split(FOLD_LABEL_SEP, 1)
    if not f or not r:
        raise ValueError(f"not a fold-qualified label 'fold_id{FOLD_LABEL_SEP}row_id': {label!r}")
    return f, r


def _is_design_hash(design_hash: Any) -> bool:
    return (isinstance(design_hash, str) and len(design_hash) == FOLD_DESIGN_HASH_LEN
            and set(design_hash) <= FOLD_DESIGN_HASH_CHARS)


def s1c_fold_design_id(stem: str, design_hash: str) -> str:
    """The outer fold design id every S1(c) arm names: ``<stem>@<design_hash>`` (``folds.io.design_hash`` of the fold
    file the arm was fitted on: 64 lowercase hex characters)."""
    if not isinstance(stem, str) or not stem or FOLD_DESIGN_HASH_SEP in stem or not _is_design_hash(design_hash):
        raise ValueError(f"bad fold design stem {stem!r} / hash {design_hash!r} (hash: {FOLD_DESIGN_HASH_LEN} lowercase "
                         "hex characters, folds.io.design_hash)")
    return f"{stem}{FOLD_DESIGN_HASH_SEP}{design_hash}"


def split_s1c_fold_design_id(fold_design: Any) -> tuple[str, str]:
    """``(stem, design_hash)`` of a fold design id ``<stem>@<design_hash>``; ``ValueError`` on anything else -- a bare
    stem (the same for every seed, so it cannot show which folds were fitted), an empty or malformed hash."""
    if not isinstance(fold_design, str) or fold_design.count(FOLD_DESIGN_HASH_SEP) != 1:
        raise ValueError(f"not a fold design id '<stem>{FOLD_DESIGN_HASH_SEP}<design_hash>': {fold_design!r}")
    stem, dhash = fold_design.split(FOLD_DESIGN_HASH_SEP)
    if not stem or not _is_design_hash(dhash):
        raise ValueError(f"not a fold design id '<stem>{FOLD_DESIGN_HASH_SEP}<design_hash>' (hash: "
                         f"{FOLD_DESIGN_HASH_LEN} lowercase hex characters): {fold_design!r}")
    return stem, dhash


def is_batched_v5pair_design(fold_design: str) -> bool:
    """Whether a fold design stem ``<stem>`` or a well-formed id ``<stem>@<design_hash>`` names a batched V5-PAIR fold
    file (``V5PAIR__<variant>__batched...``).  A string carrying the separator with an empty or malformed hash is not
    one (False)."""
    s = str(fold_design)
    if FOLD_DESIGN_HASH_SEP in s:
        try:
            s = split_s1c_fold_design_id(s)[0]
        except ValueError:
            return False
    parts = s.split("__")
    return len(parts) == 3 and parts[0] == S1C_BATCHED_DESIGN and parts[2].startswith(S1C_BATCHED_SCHEME_PREFIX)


def guard_scored_pairs(pairs: pd.DataFrame, *, folds: pd.Series, v6_mask: pd.Series, test_label: Any = "test",
                       design: str = "V5-PAIR", what: str = "pair scoring") -> pd.Index:
    """The section 2 checks every pair passes before it enters a metric; returns the union of the member labels.

    * ``leakage.pair_isolation_check`` of ``idx_a`` / ``idx_b`` against ``folds`` (member label -> fold label), and
      both members in ``test_label`` (a training row can never be a scored member);
    * member labels are fold-qualified (``fold_id|row_id``) and the fold part of both equals the pair's ``fold`` column,
      so a prediction of one fold is never paired with a prediction of another (rows are scored in up to 17 V5-PAIR
      folds);
    * ``v6_mask`` (member label -> in ``V6_TARGET_ROWS``) covers every member label and the union passes
      ``metrics.guard_scoring_index`` (``registered.assert_not_scored``)."""
    from gen19ct.data import leakage as LK

    if not isinstance(folds, pd.Series):
        raise TypeError("folds must be a Series: member label -> fold label")
    if not isinstance(v6_mask, pd.Series):
        raise TypeError("v6_mask must be a Series: member label -> in V6_TARGET_ROWS (required before confirmation)")
    for c in ("idx_a", "idx_b", "fold"):
        if c not in pairs.columns:
            raise KeyError(f"{what}: pairs lack column {c!r}")
    LK.pair_isolation_check(pairs, folds, member_cols=("idx_a", "idx_b"))
    in_test = (pairs["idx_a"].map(folds).astype(object).eq(test_label)
               & pairs["idx_b"].map(folds).astype(object).eq(test_label))
    if not bool(in_test.all()):
        raise AssertionError(f"{what}: only {test_label!r}-{test_label!r} pairs are scored; "
                             f"{int((~in_test).sum())} pair(s) have a member in another fold role")
    pf = pairs["fold"].astype(str)
    for col in ("idx_a", "idx_b"):
        lab = pairs[col].astype(str)
        if not lab.str.contains(FOLD_LABEL_SEP, regex=False).all():
            raise ValueError(f"{what}: member labels must be fold-qualified 'fold_id{FOLD_LABEL_SEP}row_id' ({col})")
        bad = lab.str.split(FOLD_LABEL_SEP, n=1).str[0] != pf
        if bad.any():
            raise AssertionError(f"{what}: {int(bad.sum())} pair(s) whose {col} was predicted in another fold than the "
                                 f"pair's (first {lab[bad].iloc[0]!r} in fold {pf[bad].iloc[0]!r})")
    members = pd.Index(pd.unique(np.concatenate([pairs["idx_a"].to_numpy(dtype=object),
                                                 pairs["idx_b"].to_numpy(dtype=object)])))
    missing = members.difference(v6_mask.index)
    if len(missing):
        raise ValueError(f"{what}: v6_mask does not cover {len(missing)} member label(s) (first {missing[0]!r}); a mask "
                         "keyed otherwise would pass the V6 guard vacuously")
    EM.guard_scoring_index(members, v6_mask, design, what=what)
    return members


def _direction_scores(pairs: pd.DataFrame, pred: pd.Series, *, direction_only: bool) -> np.ndarray:
    """Per pair 1 / 1/2 (predicted zero) / 0, NaN when |observed logSF| < 0.3 or the prediction is undefined
    (``pairs.score_pairs`` at the V5-PAIR threshold)."""
    from gen19ct.evaluation import pairs as EP

    sc = EP.score_pairs(pairs, pred, design="V5-PAIR", direction_only=direction_only)
    return sc[f"dir_{EP.DIRECTION_THRESHOLD:g}"].to_numpy(dtype=float)


def _unit_codes(pairs: pd.DataFrame, unit_cols: Sequence[str]) -> tuple[np.ndarray, pd.Index, np.ndarray]:
    codes, keys = EM.unit_index(pairs, unit_cols)
    labels = pd.Index(EM.unit_keys(keys, unit_cols).to_numpy(), name="cell_pair")
    return codes, labels, keys


def _pair_set_sha256(pairs: pd.DataFrame) -> str:
    """SHA-256 of the sorted ``row_a<TAB>row_b<TAB>observed logSF`` lines of a pair frame with the fold part of every
    member label stripped: the identity of a pair set across fitted runs whose fold ids differ (the withheld seeds'
    batched folds)."""
    import hashlib

    ra = [split_fold_label(x)[1] for x in pairs["idx_a"].astype(str)]
    rb = [split_fold_label(x)[1] for x in pairs["idx_b"].astype(str)]
    obs = pairs["logsf_obs"].to_numpy(dtype=float)
    lines = sorted(f"{a}\t{b}\t{float(o)!r}" for a, b, o in zip(ra, rb, obs))
    return hashlib.sha256("".join(f"{x}\n" for x in lines).encode("utf-8")).hexdigest()


def paired_direction_contrast(pairs: pd.DataFrame, candidate_logsf: pd.Series, yardstick_pred: pd.Series, *,
                              yardstick: str, yardstick_direction_only: bool, folds: pd.Series, v6_mask: pd.Series,
                              test_label: Any = "test", unit_cols: Sequence[str] | None = None,
                              system_col: str = EM.SYSTEM_COL, n_resamples: int = N_RESAMPLES,
                              seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Delta_Y = direction accuracy of the candidate - direction accuracy of yardstick Y, both on the SAME comparable
    test-test pairs: those where Y is defined and |observed logSF| >= 0.3; per cell pair, then an equal-weight mean
    over the cell pairs holding at least one such pair (identical for both arms, so the Delta is paired).  A system
    cluster bootstrap (10,000 resamples, seed 19, higher is better) gives the percentile 95 % interval.  The
    candidate must predict every pair.  The pairs pass :func:`guard_scored_pairs` first (``folds``, ``v6_mask``).
    ``per_cell_pair`` holds both accuracies, the system and the pair count of every cell pair (the input of the seed
    combination at confirmation, :func:`s1c_seed_combination`)."""
    from gen19ct.evaluation import pairs as EP

    guard_scored_pairs(pairs, folds=folds, v6_mask=v6_mask, test_label=test_label,
                       what=f"S1(c) direction vs {yardstick}")
    ucols = tuple(EP.CELL_PAIR_COLS if unit_cols is None else unit_cols)
    if not (candidate_logsf.index.equals(pairs.index) and yardstick_pred.index.equals(pairs.index)):
        raise ValueError("candidate and yardstick predictions must be indexed like pairs")
    if not np.isfinite(candidate_logsf.to_numpy(dtype=float)).all():
        raise ValueError("the candidate must predict every pair (a magnitude arm)")
    cand = _direction_scores(pairs, candidate_logsf, direction_only=False)
    yard = _direction_scores(pairs, yardstick_pred, direction_only=yardstick_direction_only)
    keep = np.isfinite(yard)
    out: dict[str, Any] = {"yardstick": yardstick, "n_pairs": int(keep.sum()),
                           "n_pairs_qualifying": int(np.isfinite(cand).sum()),
                           "n_pairs_yardstick_undefined": int((np.isfinite(cand) & ~keep).sum())}
    if not keep.any():
        empty = pd.DataFrame(columns=["candidate", "yardstick", "system", "n_pairs"], index=pd.Index([], name="cell_pair"))
        return {**out, "testable": False, "delta": float("nan"), "candidate_accuracy": float("nan"),
                "yardstick_accuracy": float("nan"), "n_cell_pairs": 0, "n_systems": 0, "percentile_low": float("nan"),
                "percentile_high": float("nan"), "interval_excludes_zero": False, "bootstrap": None,
                "per_cell_pair": empty}
    sub = pairs[keep]
    codes, labels, keys = _unit_codes(sub, ucols)
    n = np.bincount(codes).astype(float)
    acc_c = pd.Series(np.bincount(codes, weights=cand[keep]) / n, index=labels)
    acc_y = pd.Series(np.bincount(codes, weights=yard[keep]) / n, index=labels)
    systems = pd.Series(keys[system_col].astype(str).to_numpy(), index=labels)
    br = paired_cluster_bootstrap(acc_y, acc_c, systems, higher_is_better=True, n_resamples=n_resamples, seed=seed,
                                  contrast=f"direction: candidate - {yardstick}", cluster_unit="system")
    lo, hi = br.percentile_interval()
    per = pd.DataFrame({"candidate": acc_c, "yardstick": acc_y, "system": systems,
                        "n_pairs": pd.Series(n.astype(int), index=labels)})
    return {**out, "testable": br.n_clusters >= 2, "delta": br.point, "candidate_accuracy": float(acc_c.mean()),
            "yardstick_accuracy": float(acc_y.mean()), "n_cell_pairs": int(len(labels)),
            "n_systems": int(br.n_clusters), "percentile_low": lo, "percentile_high": hi,
            "interval_excludes_zero": bool(np.isfinite(lo) and lo > 0), "p_two_sided": br.p_two_sided,
            "bootstrap": br, "per_cell_pair": per}


def paired_logsf_mae_gain(pairs: pd.DataFrame, candidate_logsf: pd.Series, comparator_logsf: pd.Series, *,
                          comparator: str, folds: pd.Series, v6_mask: pd.Series, test_label: Any = "test",
                          unit_cols: Sequence[str] | None = None, system_col: str = EM.SYSTEM_COL,
                          n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Gain = comparator logSF MAE - candidate logSF MAE on the identical pair set (every comparable test-test pair;
    both arms must predict every pair), cell-pair macro.  The system-cluster bootstrap is printed, not deciding
    (S1(c) fixes only the margin eta5 = 0.02 on the point).  The pairs pass :func:`guard_scored_pairs` first.
    ``per_cell_pair`` holds both MAEs, the system and the pair count of every cell pair."""
    from gen19ct.evaluation import pairs as EP

    guard_scored_pairs(pairs, folds=folds, v6_mask=v6_mask, test_label=test_label,
                       what=f"S1(c) logSF MAE vs {comparator}")
    ucols = tuple(EP.CELL_PAIR_COLS if unit_cols is None else unit_cols)
    if not (candidate_logsf.index.equals(pairs.index) and comparator_logsf.index.equals(pairs.index)):
        raise ValueError("candidate and comparator logSF must be indexed like pairs")
    obs = pairs["logsf_obs"].to_numpy(dtype=float)
    c = candidate_logsf.to_numpy(dtype=float)
    k = comparator_logsf.to_numpy(dtype=float)
    if not (np.isfinite(obs).all() and np.isfinite(c).all() and np.isfinite(k).all()):
        raise ValueError("observed, candidate and comparator logSF must be finite on every pair")
    if not len(pairs):
        return {"comparator": comparator, "testable": False, "gain": float("nan"), "n_pairs": 0, "n_cell_pairs": 0,
                "per_cell_pair": pd.DataFrame(columns=["candidate_logsf_mae", "comparator_logsf_mae", "system",
                                                       "n_pairs"], index=pd.Index([], name="cell_pair"))}
    codes, labels, keys = _unit_codes(pairs, ucols)
    n = np.bincount(codes).astype(float)
    mae_c = pd.Series(np.bincount(codes, weights=np.abs(c - obs)) / n, index=labels)
    mae_k = pd.Series(np.bincount(codes, weights=np.abs(k - obs)) / n, index=labels)
    systems = pd.Series(keys[system_col].astype(str).to_numpy(), index=labels)
    br = paired_cluster_bootstrap(mae_k, mae_c, systems, n_resamples=n_resamples, seed=seed,
                                  contrast=f"logSF MAE: {comparator} - candidate", cluster_unit="system")
    lo, hi = br.percentile_interval()
    per = pd.DataFrame({"candidate_logsf_mae": mae_c, "comparator_logsf_mae": mae_k, "system": systems,
                        "n_pairs": pd.Series(n.astype(int), index=labels)})
    return {"comparator": comparator, "testable": True, "gain": br.point, "candidate_logsf_mae": float(mae_c.mean()),
            "comparator_logsf_mae": float(mae_k.mean()), "n_pairs": int(len(pairs)), "n_cell_pairs": int(len(labels)),
            "n_systems": int(br.n_clusters), "percentile_low": lo, "percentile_high": hi, "bootstrap": br,
            "per_cell_pair": per}


def check_s1c_fold_designs(fold_designs: Mapping[str, str],
                           fold_reading: str = S1C_REGISTERED_FOLD_READING) -> dict[str, str]:
    """The outer fold design of every S1(c) arm under the registered "same fitted folds" reading (section 9, resolved
    by the orchestrator 2026-09-15: :data:`S1C_REGISTERED_FOLD_READING`).

    ``fold_designs`` maps ``pairs``, ``candidate``, ``B3x`` and ``B3i`` (:data:`S1C_FOLD_DESIGN_KEYS`) to the fold design
    id each was fitted on (:func:`s1c_fold_design_id`, ``<stem>@<design_hash>``); ``HEAVIER`` / ``FLAT`` may be named and
    are never compared (they need no fit).  Refused:

    * a reading other than the registered one (``ValueError``; :data:`S1C_UNREGISTERED_FOLD_READINGS`);
    * a missing entry or an unknown arm (``ValueError``);
    * an entry that is not a well-formed ``<stem>@<design_hash>`` id -- a bare stem or an empty / malformed hash
      (``ValueError``): the stem of a batched V5-PAIR file is the same for every seed, so only the design hash shows
      that the arms were fitted on the same folds (:func:`split_s1c_fold_design_id`);
    * a candidate whose fold design is not a batched V5-PAIR design (``ValueError``);
    * a yardstick or pair set whose fold design differs from the candidate's -- e.g. B3x / B3i taken from the unbatched
      pre-seal folds, or from the batched folds of another seed (``AssertionError``).

    Returns the compared designs plus ``HEAVIER`` / ``FLAT`` -> ``"no fit"``."""
    if fold_reading != S1C_REGISTERED_FOLD_READING:
        why = S1C_UNREGISTERED_FOLD_READINGS.get(fold_reading)
        raise ValueError(f"S1(c) fold reading {fold_reading!r} is not the registered reading "
                         f"{S1C_REGISTERED_FOLD_READING!r}" + (f": {why}" if why else ""))
    if not isinstance(fold_designs, Mapping):
        raise TypeError("fold_designs must map every S1(c) arm to its outer fold design (stem@design_hash)")
    unknown = sorted(set(map(str, fold_designs)) - set(S1C_FOLD_DESIGN_KEYS) - set(S1C_UNFITTED_RULES))
    if unknown:
        raise ValueError(f"fold_designs names arm(s) {unknown} that are not part of S1(c)")
    missing = [k for k in S1C_FOLD_DESIGN_KEYS if not fold_designs.get(k)]
    if missing:
        raise ValueError(f"fold_designs lacks {missing}: every S1(c) arm names the outer fold design it was fitted on")
    fd = {k: fold_designs[k] for k in S1C_FOLD_DESIGN_KEYS}
    for k, v in fd.items():
        try:
            split_s1c_fold_design_id(v)
        except ValueError as exc:
            raise ValueError(f"S1(c) fold_designs[{k!r}]: every fitted arm and the pair set name the design hash of the "
                             f"fold file they were fitted on ({exc})") from None
    cand = fd["candidate"]
    if not is_batched_v5pair_design(cand):
        raise ValueError(f"S1(c): the candidate's fold design {cand!r} is not a batched V5-PAIR design "
                         f"({S1C_BATCHED_DESIGN}__<variant>__{S1C_BATCHED_SCHEME_PREFIX}...)")
    differ = {k: v for k, v in fd.items() if v != cand}
    if differ:
        raise AssertionError(f"S1(c) 'same fitted folds': different outer fold designs -- {differ} instead of the "
                             f"candidate's {cand!r}; B3x and B3i are re-fitted on exactly the batched V5-PAIR folds the "
                             "candidate is fitted on (models.s1c_yardsticks.refit_lookup_yardsticks)")
    return {**fd, **{r: "no fit (a rule on the pairs)" for r in S1C_UNFITTED_RULES}}


def s1c_half(pairs: pd.DataFrame, candidate_logsf: pd.Series, *, half: str, seed: int | None,
             lookup_logsf: Mapping[str, pd.Series], folds: pd.Series, v6_mask: pd.Series,
             fold_designs: Mapping[str, str], fold_reading: str = S1C_REGISTERED_FOLD_READING,
             test_label: Any = "test", exclude_rows: Iterable[Any] | None = None, scoring_filter: str = "none",
             heavier: pd.Series | None = None, unit_cols: Sequence[str] | None = None,
             system_col: str = EM.SYSTEM_COL, n_resamples: int = N_RESAMPLES,
             bootstrap_seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Every S1(c) component of ONE half and ONE fitted run (one seed's batched V5-PAIR folds): ``pairs`` are that
    half's comparable test-test pairs (V5-PAIR; columns ``idx_a`` / ``idx_b`` = fold-qualified member labels, ``fold``,
    the cell-pair columns, ``logsf_obs`` and ``half``, which must equal ``half`` on every pair -- confirmation-half pairs
    are never scored as the selection half or the reverse; ``models.s1c_yardsticks.select_half`` supplies it),
    ``candidate_logsf`` the candidate's derived logSF, ``lookup_logsf`` the B3x-
    and B3i-derived logSF of the yardsticks re-fitted on the same folds (``pairs.derived_logsf``;
    ``models.s1c_yardsticks``) and ``heavier`` the HEAVIER direction (computed from the pairs when omitted).  ``seed``
    is the seed whose batched folds were fitted (104729 on the selection half in discovery, a withheld seed at
    confirmation).

    Guards (section 2, before anything is scored): :func:`check_s1c_fold_designs` of ``fold_designs`` under the
    registered reading, and :func:`guard_scored_pairs` with ``folds`` (member label -> fold label; members must be
    ``test_label``) and ``v6_mask`` (member label -> in ``V6_TARGET_ROWS``).  ``exclude_rows`` is the scoring-filter hook
    of the R19 item 6 re-scoring sensitivities (``scoring_filter`` names it): fold-qualified member labels
    (``fold_id|row_id``; anything else is refused, since a bare row id would silently exclude nothing); a pair with an
    excluded member is dropped for every arm alike.

    Returns the per-yardstick direction contrasts (:func:`paired_direction_contrast`), their minimum, the logSF MAE gains
    over FLAT and over the B3i-derived logSF (:func:`paired_logsf_mae_gain`) and ``pair_set_sha256`` (the pair set with
    fold parts stripped, :func:`s1c_seed_combination` requires it identical across seeds)."""
    from gen19ct.evaluation import pairs as EP

    if half not in S1C_HALVES:
        raise ValueError(f"half must be one of {S1C_HALVES}")
    if "half" not in pairs.columns:
        raise KeyError(f"S1(c) {half} half: pairs lack the column 'half' (the registered half of every pair; "
                       "models.s1c_yardsticks.yardstick_pair_inputs supplies it)")
    wrong_half = pairs["half"].astype(object).ne(half)
    if bool(wrong_half.any()):
        raise ValueError(f"S1(c) {half} half: {int(wrong_half.sum())} pair(s) belong to another half "
                         f"({sorted(map(str, pd.unique(pairs.loc[wrong_half, 'half'])))}); a half is scored on its own "
                         "pairs only")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise TypeError(f"seed must be an int or None, got {seed!r}")
    missing = [k for k in S1C_REFIT_YARDSTICKS if k not in lookup_logsf]
    if missing:
        raise ValueError(f"lookup_logsf lacks {missing}")
    fd = check_s1c_fold_designs(fold_designs, fold_reading)
    guard_scored_pairs(pairs, folds=folds, v6_mask=v6_mask, test_label=test_label, what=f"S1(c) {half} half")
    n_before = int(len(pairs))
    heavier_all = EP.heavier_direction(pairs) if heavier is None else heavier
    for name, s in (("candidate_logsf", candidate_logsf), ("heavier", heavier_all),
                    *((f"lookup_logsf[{k!r}]", lookup_logsf[k]) for k in S1C_REFIT_YARDSTICKS)):
        if not (isinstance(s, pd.Series) and s.index.equals(pairs.index)):
            raise ValueError(f"{name} must be a Series indexed like pairs")
    n_exclude_labels = 0
    if exclude_rows is not None:
        if isinstance(exclude_rows, (str, bytes)) or not isinstance(exclude_rows, Iterable):
            raise TypeError("exclude_rows must be an iterable of fold-qualified labels 'fold_id|row_id'")
        ex_list = list(exclude_rows)
        for lab in ex_list:
            try:
                split_fold_label(lab)
            except ValueError as exc:
                raise ValueError(f"S1(c) {half} half, scoring filter {scoring_filter!r}: exclude_rows must be "
                                 f"fold-qualified 'fold_id{FOLD_LABEL_SEP}row_id' labels ({exc})") from None
        ex = set(ex_list)
        n_exclude_labels = len(ex)
        keep = ~(pairs["idx_a"].isin(ex) | pairs["idx_b"].isin(ex)).to_numpy(dtype=bool)
        pairs = pairs[keep]
        candidate_logsf, heavier_all = candidate_logsf[keep], heavier_all[keep]
        lookup_logsf = {k: lookup_logsf[k][keep] for k in S1C_REFIT_YARDSTICKS}
    yard = {"HEAVIER": (heavier_all, True)}
    yard.update({k: (lookup_logsf[k], False) for k in S1C_REFIT_YARDSTICKS})
    kw = dict(folds=folds, v6_mask=v6_mask, test_label=test_label, unit_cols=unit_cols, system_col=system_col,
              n_resamples=n_resamples, seed=bootstrap_seed)
    direction = {k: paired_direction_contrast(pairs, candidate_logsf, yard[k][0], yardstick=k,
                                              yardstick_direction_only=yard[k][1], **kw) for k in S1C_YARDSTICKS}
    testable = all(d["testable"] for d in direction.values())
    deltas = {k: d["delta"] for k, d in direction.items()}
    min_y = min(deltas, key=lambda k: deltas[k]) if testable else None
    mae = {"FLAT": paired_logsf_mae_gain(pairs, candidate_logsf, EP.flat_logsf(pairs), comparator="FLAT", **kw),
           S1C_LOOKUP: paired_logsf_mae_gain(pairs, candidate_logsf, lookup_logsf[S1C_LOOKUP], comparator=S1C_LOOKUP,
                                             **kw)}
    return {"half": half, "seed": None if seed is None else int(seed), "n_pairs": int(len(pairs)),
            "n_pairs_excluded_by_filter": n_before - int(len(pairs)), "n_exclude_labels": n_exclude_labels,
            "scoring_filter": scoring_filter, "fold_reading": fold_reading, "fold_designs": fd,
            "fold_design": fd["candidate"], "pair_set_sha256": _pair_set_sha256(pairs), "direction": direction,
            "direction_testable": testable, "min_delta": deltas[min_y] if min_y else float("nan"),
            "min_delta_yardstick": min_y, "logsf_mae": mae}


def _seed_value_table(per_seed: Sequence[pd.DataFrame], seeds: Sequence[int], col: str, what: str) -> pd.DataFrame:
    idx = per_seed[0].index
    for s, t in zip(seeds, per_seed):
        if set(t.index) != set(idx):
            raise ValueError(f"S1(c) seed combination, {what}: seed {s} is scored on other cell pairs than seed "
                             f"{seeds[0]} (the confirmation pair set must be identical in every seed)")
    return pd.DataFrame({s: t[col].reindex(idx).to_numpy(dtype=float) for s, t in zip(seeds, per_seed)}, index=idx)


def _seed_clusters(per_seed: Sequence[pd.DataFrame], what: str) -> pd.Series:
    first = per_seed[0]["system"].astype(str)
    for t in per_seed[1:]:
        if not t["system"].astype(str).reindex(first.index).equals(first):
            raise ValueError(f"S1(c) seed combination, {what}: a cell pair changes system between seeds")
    return first


def s1c_seed_combination(halves: Sequence[Mapping[str, Any]], *, n_resamples: int = N_RESAMPLES,
                         bootstrap_seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Section 9 S1(c) "seed combination at confirmation" (resolved by the orchestrator 2026-09-15).

    ``halves``: one :func:`s1c_half` result per withheld seed, all on the confirmation half.  Refused unless there are
    exactly 5, their seeds are distinct and none is a discovery seed, each was fitted on its own batched V5-PAIR folds
    (every half's ``fold_designs`` re-checked by :func:`check_s1c_fold_designs`, and the 5 candidate design hashes
    distinct), all under the registered reading and one scoring filter, and all on the identical confirmation pair set
    (``pair_set_sha256``).

    For each yardstick Y: Delta_Y = the mean over the 5 seeds of the per-seed cell-pair-macro Delta_Y; its interval is
    the percentile 95 % interval of a system-cluster bootstrap of that seed mean with the same resampled systems applied
    to every seed (:func:`seed_mean_nested_cluster_bootstrap`, 10,000 resamples, seed 19); ``all_seeds_positive`` is
    Delta_Y > 0 in 5 of 5 seeds.  The logSF MAE gains over FLAT and over the B3i-derived logSF are combined the same way
    (the seed mean decides against eta5; the interval is printed)."""
    if isinstance(halves, Mapping) or isinstance(halves, (str, bytes)) or not isinstance(halves, Sequence):
        raise TypeError("the confirmation S1(c) takes one s1c_half result per withheld seed (a sequence of 5)")
    hs = list(halves)
    if len(hs) != N_SEEDS:
        raise ValueError(f"the confirmation S1(c) needs one s1c_half result per withheld seed: {N_SEEDS}, got {len(hs)}")
    if any(h.get("half") != "confirmation" for h in hs):
        raise ValueError("every seed of the S1(c) seed combination must be scored on the confirmation half")
    seeds = [h.get("seed") for h in hs]
    if any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) for s in seeds):
        raise ValueError(f"every confirmation half must carry its withheld seed, got {seeds}")
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != N_SEEDS:
        raise ValueError(f"the withheld seeds must be distinct, got {seeds}")
    if set(seeds) & set(DISCOVERY_SEEDS):
        raise ValueError(f"confirmation is scored on the withheld seeds, not on discovery seeds "
                         f"{sorted(set(seeds) & set(DISCOVERY_SEEDS))}")
    readings = {h.get("fold_reading") for h in hs}
    if readings != {S1C_REGISTERED_FOLD_READING}:
        raise ValueError(f"confirmation halves scored under {sorted(map(str, readings))}; registered reading is "
                         f"{S1C_REGISTERED_FOLD_READING!r}")
    filters = {h.get("scoring_filter") for h in hs}
    if len(filters) != 1:
        raise ValueError(f"the seeds were scored under different scoring filters {sorted(map(str, filters))}")
    if len({h.get("pair_set_sha256") for h in hs}) != 1:
        raise ValueError("the seeds were scored on different pair sets; S1(c) is paired on the identical confirmation "
                         "pair set")
    # every half's fold designs are re-checked (a result dict is not trusted to come from s1c_half): hashed ids, one
    # batched V5-PAIR design per seed for every fitted arm; distinct seeds need distinct design HASHES (the stem is the
    # same for every seed)
    designs = [check_s1c_fold_designs(h.get("fold_designs"), h.get("fold_reading"))["candidate"] for h in hs]
    if len({split_s1c_fold_design_id(d)[1] for d in designs}) != N_SEEDS:
        raise ValueError(f"each withheld seed is fitted on its own batched V5-PAIR folds; fold designs {designs}")
    order = np.argsort(seeds, kind="stable")
    hs = [hs[i] for i in order]
    designs = [designs[i] for i in order]
    seeds = [seeds[i] for i in order]
    boot = dict(n_resamples=n_resamples, seed=bootstrap_seed)

    direction: dict[str, dict[str, Any]] = {}
    for y in S1C_YARDSTICKS:
        per = [h["direction"][y] for h in hs]
        per_seed_delta = {s: float(p["delta"]) for s, p in zip(seeds, per)}
        if not all(p["testable"] for p in per):
            direction[y] = {"yardstick": y, "testable": False, "delta": float("nan"), "per_seed_delta": per_seed_delta,
                            "n_seeds_positive": None, "all_seeds_positive": None, "percentile_low": float("nan"),
                            "percentile_high": float("nan"), "interval_excludes_zero": False, "bootstrap": None}
            continue
        tabs = [p["per_cell_pair"] for p in per]
        comp = _seed_value_table(tabs, seeds, "yardstick", f"direction vs {y}")
        cand = _seed_value_table(tabs, seeds, "candidate", f"direction vs {y}")
        clusters = _seed_clusters(tabs, f"direction vs {y}")
        br = seed_mean_nested_cluster_bootstrap(comp, cand, clusters, higher_is_better=True, cluster_unit="system",
                                                contrast=f"direction: candidate - {y} (mean of {N_SEEDS} seeds)", **boot)
        for s in seeds:
            if not math.isclose(float((cand[s] - comp[s]).mean()), per_seed_delta[s], rel_tol=1e-9, abs_tol=1e-12):
                raise AssertionError(f"S1(c) seed {s}, {y}: per-cell-pair table does not reproduce its Delta")
        lo, hi = br.percentile_interval()
        n_pos = int(sum(1 for v in per_seed_delta.values() if np.isfinite(v) and v > 0))
        direction[y] = {"yardstick": y, "testable": br.n_clusters >= 2, "delta": br.point,
                        "per_seed_delta": per_seed_delta, "n_seeds_positive": n_pos,
                        "all_seeds_positive": n_pos == N_SEEDS, "percentile_low": lo, "percentile_high": hi,
                        "interval_excludes_zero": bool(np.isfinite(lo) and lo > 0), "p_two_sided": br.p_two_sided,
                        "n_cell_pairs": int(len(comp)), "n_systems": int(br.n_clusters), "bootstrap": br}
    testable = all(d["testable"] for d in direction.values())
    deltas = {k: d["delta"] for k, d in direction.items()}
    min_y = min(deltas, key=lambda k: deltas[k]) if testable else None

    mae: dict[str, dict[str, Any]] = {}
    for comp_name in ("FLAT", S1C_LOOKUP):
        per = [h["logsf_mae"][comp_name] for h in hs]
        per_seed_gain = {s: float(p["gain"]) for s, p in zip(seeds, per)}
        if not all(p["testable"] for p in per):
            mae[comp_name] = {"comparator": comp_name, "testable": False, "gain": float("nan"),
                              "per_seed_gain": per_seed_gain, "bootstrap": None}
            continue
        tabs = [p["per_cell_pair"] for p in per]
        k_tab = _seed_value_table(tabs, seeds, "comparator_logsf_mae", f"logSF MAE vs {comp_name}")
        c_tab = _seed_value_table(tabs, seeds, "candidate_logsf_mae", f"logSF MAE vs {comp_name}")
        br = seed_mean_nested_cluster_bootstrap(k_tab, c_tab, _seed_clusters(tabs, f"logSF MAE vs {comp_name}"),
                                                cluster_unit="system",
                                                contrast=f"logSF MAE: {comp_name} - candidate (mean of {N_SEEDS} seeds)",
                                                **boot)
        lo, hi = br.percentile_interval()
        mae[comp_name] = {"comparator": comp_name, "testable": True, "gain": br.point, "per_seed_gain": per_seed_gain,
                          "candidate_logsf_mae": float(c_tab.mean(axis=0).mean()),
                          "comparator_logsf_mae": float(k_tab.mean(axis=0).mean()),
                          "n_seeds_positive": int(sum(1 for v in per_seed_gain.values() if np.isfinite(v) and v > 0)),
                          "percentile_low": lo, "percentile_high": hi, "n_cell_pairs": int(len(k_tab)),
                          "n_systems": int(br.n_clusters), "bootstrap": br}
    return {"half": "confirmation", "seeds": seeds, "n_seeds": N_SEEDS, "fold_reading": S1C_REGISTERED_FOLD_READING,
            "scoring_filter": next(iter(filters)), "pair_set_sha256": hs[0]["pair_set_sha256"],
            "n_pairs": int(hs[0]["n_pairs"]), "fold_designs_by_seed": dict(zip(seeds, designs)),
            "direction": direction, "direction_testable": testable,
            "min_delta": deltas[min_y] if min_y else float("nan"), "min_delta_yardstick": min_y, "logsf_mae": mae}


def s1c_paired_verdict(*, confirmation: Sequence[Mapping[str, Any]] | None, selection: Mapping[str, Any] | None,
                       v5_pair_dropped: bool = False) -> dict[str, Any]:
    """Section 9 S1(c) as redefined on 2026-09-15.  Passes only if both hold:

    * confirmation half, the 5 withheld seeds combined (:func:`s1c_seed_combination`): min_Y Delta_Y >= gamma5 = 0.05
      (Delta_Y = the seed mean), every Delta_Y's system-cluster percentile 95 % interval excludes 0 (lower bound > 0),
      Delta_Y > 0 in 5 of 5 withheld seeds for every Y, and the candidate's logSF MAE is below FLAT's and below the
      B3i-derived logSF MAE by >= eta5 = 0.02 (seed mean) on the identical confirmation pair set;
    * selection half (discovery seed 104729): min_Y Delta_Y >= -0.02.

    ``confirmation`` is the sequence of the 5 per-seed :func:`s1c_half` results, ``selection`` one :func:`s1c_half`
    result.  An evaluation that did not happen as registered is refused (``ValueError``), never recorded as a scientific
    FAIL: a half scored under another "same fitted folds" reading, fold designs that :func:`check_s1c_fold_designs`
    refuses, halves under different scoring filters, a ``selection`` result that was not scored on the selection half or
    not on discovery seed 104729, or a selection run on the fold design of a confirmation seed.  FAIL when a computed
    check fails; UNDECIDED when V5-PAIR was dropped, a half is missing, or a yardstick has no defined qualifying pair;
    PASS otherwise.  Every component is returned."""
    if v5_pair_dropped:
        return {"verdict": "UNDECIDED", "reason": "V5-PAIR dropped (section 3.1): S1(c) untestable, S1 UNDECIDED"}
    if isinstance(confirmation, Mapping):
        raise TypeError("confirmation takes the 5 per-seed s1c_half results (one per withheld seed), not one half")
    if selection is not None and not isinstance(selection, Mapping):
        raise TypeError("selection takes one s1c_half result (the selection half on discovery seed 104729)")
    conf_halves = [] if confirmation is None else list(confirmation)
    for h in conf_halves + ([selection] if selection is not None else []):
        if h.get("fold_reading") != S1C_REGISTERED_FOLD_READING:
            raise ValueError(f"S1(c) half scored under {h.get('fold_reading')!r}, registered reading is "
                             f"{S1C_REGISTERED_FOLD_READING!r}")
    sel_design = None
    if selection is not None:
        if selection.get("half") != "selection":
            raise ValueError(f"the result passed as selection was scored on the {selection.get('half')!r} half; the S1(c) "
                             "counterweight is the selection half")
        sel_seed = selection.get("seed")
        if isinstance(sel_seed, bool) or not isinstance(sel_seed, (int, np.integer)) or int(sel_seed) != S1C_SELECTION_SEED:
            raise ValueError(f"the S1(c) selection half is registered on discovery seed {S1C_SELECTION_SEED} (got "
                             f"{sel_seed!r}); a run on another seed is not the registered evaluation")
        sel_design = check_s1c_fold_designs(selection.get("fold_designs"), selection.get("fold_reading"))["candidate"]
    checks: dict[str, bool | None] = {"fold_reading_registered": True}
    reasons: list[str] = []
    combined = None
    conf_keys = ("confirmation_min_delta_at_least_gamma5", "confirmation_every_direction_interval_excludes_zero",
                 "confirmation_delta_positive_in_5_of_5_seeds", "confirmation_logsf_mae_below_FLAT_by_eta5",
                 f"confirmation_logsf_mae_below_{S1C_LOOKUP}_by_eta5")
    if confirmation is None:
        reasons.append("confirmation-half components missing")
        checks.update(dict.fromkeys(conf_keys))
    else:
        combined = s1c_seed_combination(conf_halves)
        if combined["direction_testable"]:
            d = combined["direction"]
            checks[conf_keys[0]] = bool(combined["min_delta"] >= GAMMA5)
            checks[conf_keys[1]] = all(v["interval_excludes_zero"] for v in d.values())
            checks[conf_keys[2]] = all(bool(v["all_seeds_positive"]) for v in d.values())
        else:
            reasons.append("confirmation half: a yardstick has no defined qualifying pair")
            checks.update(dict.fromkeys(conf_keys[:3]))
        for comp in ("FLAT", S1C_LOOKUP):
            g = combined["logsf_mae"][comp]
            checks[f"confirmation_logsf_mae_below_{comp}_by_eta5"] = (bool(np.isfinite(g["gain"]) and g["gain"] >= ETA5)
                                                                     if g["testable"] else None)
    if selection is None:
        reasons.append("selection-half components missing")
        checks["selection_min_delta_at_least_-0.02"] = None
    else:
        s = selection
        if combined is not None and s.get("scoring_filter") != combined["scoring_filter"]:
            raise ValueError(f"selection half scored under scoring filter {s.get('scoring_filter')!r}, confirmation "
                             f"under {combined['scoring_filter']!r}")
        if combined is not None and sel_design in set(combined["fold_designs_by_seed"].values()):
            raise ValueError(f"the selection half was fitted on {sel_design!r}, the fold design of a withheld seed; it is "
                             f"scored on the seed-{S1C_SELECTION_SEED} batched V5-PAIR folds")
        checks["selection_min_delta_at_least_-0.02"] = (bool(s["min_delta"] >= S1C_SELECTION_MIN_DELTA)
                                                       if s["direction_testable"] else None)
        if not s["direction_testable"]:
            reasons.append("selection half: a yardstick has no defined qualifying pair")
    vals = list(checks.values())
    verdict = "FAIL" if any(v is False for v in vals) else ("UNDECIDED" if any(v is None for v in vals) else "PASS")
    return {"verdict": verdict, "checks": checks, "reasons": reasons, "gamma5": GAMMA5, "eta5": ETA5,
            "selection_min_delta": S1C_SELECTION_MIN_DELTA, "fold_reading_registered": S1C_REGISTERED_FOLD_READING,
            "confirmation_combined": combined, "confirmation": conf_halves if confirmation is not None else None,
            "selection": selection}


def s1e_check(spearman: BootstrapResult, *, components_reliable: bool) -> dict[str, Any]:
    """Section 9 S1(e): Spearman(cell MAE, support_score) <= -0.10 with the system-cluster interval excluding 0
    (upper bounds of the percentile and BCa intervals < 0) and the support components above the reliability
    floor."""
    _, phi = spearman.percentile_interval()
    _, bhi = spearman.bca_interval()
    checks = {"spearman_at_most_-0.10": bool(np.isfinite(spearman.point) and spearman.point <= S1E_MAX_SPEARMAN),
              "interval_excludes_zero": bool(np.isfinite(phi) and phi < 0 and np.isfinite(bhi) and bhi < 0),
              "components_reliable": bool(components_reliable)}
    return {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "point": spearman.point,
            "percentile_high": phi, "bca_high": bhi}


# --------------------------------------------------------------------------------------------- #
# Reliability before correlation (section 8)
# --------------------------------------------------------------------------------------------- #

def spearman_brown(r: float) -> float:
    """Split-half correlation stepped up to full length: ``2r / (1 + r)``."""
    r = float(r)
    return 2.0 * r / (1.0 + r) if np.isfinite(r) and r > -1 else float("nan")


def jackknife_reliability(between_unit_variance: float, mean_within_unit_se2: float) -> float:
    """``between / (between + mean within-unit SE^2)``."""
    b, w = float(between_unit_variance), float(mean_within_unit_se2)
    if not (np.isfinite(b) and np.isfinite(w)) or b < 0 or w < 0 or b + w == 0:
        return float("nan")
    return b / (b + w)


def reliability_gate(reliability: float) -> str:
    """``RELIABLE`` at or above the 0.3 floor; otherwise (NaN included) ``UNDECIDED_UNRELIABLE``."""
    return "RELIABLE" if np.isfinite(reliability) and reliability >= RELIABILITY_FLOOR else "UNDECIDED_UNRELIABLE"


# --------------------------------------------------------------------------------------------- #
# Signal injection (section 8 power check; never data)
# --------------------------------------------------------------------------------------------- #

def _standardise(x: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return x - x.mean() if x.size else x
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def injected_signal(frame: pd.DataFrame, *, seed: int, state_col: str = EM.METAL_STATE_COL,
                    system_col: str = EM.SYSTEM_COL, u_share: Mapping[str, str] | None = None,
                    standardise: bool = True) -> pd.Series:
    """``s_row = u_m v_s`` (section 8).  ``u ~ N(0, 1)`` per metal state (sorted labels) then ``v ~ N(0, 1)`` per
    system (sorted keys), both from ``default_rng(seed)``; standardised over states / systems, the product then
    standardised over the rows with a defined signal.  ``u_share`` maps a state to the state whose u it takes
    (H3: an An(III) state -> its nearest-CN8-radius Ln(III)).  Rows without a state or system get NaN."""
    for c in (state_col, system_col):
        if c not in frame.columns:
            raise KeyError(c)
    states = sorted({str(s) for s in frame[state_col].dropna()})
    systems = sorted({str(s) for s in frame[system_col].dropna()})
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(len(states))
    v = rng.standard_normal(len(systems))
    if standardise:
        u, v = _standardise(u), _standardise(v)
    u_of = dict(zip(states, u))
    v_of = dict(zip(systems, v))
    for s, target in (u_share or {}).items():
        if s in u_of:
            if target not in u_of:
                raise ValueError(f"u_share target {target!r} is not a metal state of the frame")
            u_of[s] = u_of[target]
    out = np.array([u_of[str(m)] * v_of[str(k)] if isinstance(m, str) and isinstance(k, str) else np.nan
                    for m, k in zip(frame[state_col].to_numpy(dtype=object), frame[system_col].to_numpy(dtype=object))],
                   dtype=float)
    if standardise:
        ok = np.isfinite(out)
        out[ok] = _standardise(out[ok])
    return pd.Series(out, index=frame.index, name="injected_signal")


def inject_targets(y: pd.Series, signal: pd.Series, kappa: float) -> pd.Series:
    """``y' = y + kappa * s`` for a registered kappa; applied to training and test rows alike."""
    if not any(math.isclose(kappa, k) for k in KAPPAS):
        raise ValueError(f"kappa must be one of {KAPPAS}")
    if not y.index.equals(signal.index):
        raise ValueError("y and signal must share the row index")
    bad = y.notna() & signal.isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) with a target have no injected signal")
    return (y + float(kappa) * signal).rename("log_D_injected")


#: the power-check verdicts.  Section 8 scopes the check to a contrast "reported as a null", so the NULL labels apply
#: only to a contrast whose un-injected verdict is FAIL; addendum 2's `needs_power` widened the TRIGGER (any primary /
#: H1b / S1(b) / H3 contrast whose full verdict is not PASS), not this vocabulary -- a triggered contrast that PASSES
#: its scope is not a null and must not be labelled one (task X findings V-L3 / V-P04)
POWER_VERDICTS: tuple[str, ...] = ("INFORMATIVE_NULL", "UNDECIDED_UNDERPOWERED", "POWERED_NOT_A_NULL",
                                   "NOT_A_NULL_UNDERPOWERED")


def power_verdict(r19_passes_by_kappa: Mapping[float, bool], *, uninjected_verdict: str | None = None) -> dict[str, Any]:
    """kappa_min = smallest registered kappa at which R19 passes; a null with kappa_min > 0.25 (or none) is
    UNDECIDED (underpowered).

    ``uninjected_verdict`` is the contrast's UN-INJECTED verdict on the reported scope (``verdict_freezing_screen`` /
    ``reported_verdict``).  Only a FAIL is a null: with a PASS the contrast is labelled ``POWERED_NOT_A_NULL`` (or
    ``NOT_A_NULL_UNDERPOWERED``), because "the null is informative" is false of a contrast that has no null.  Left
    ``None`` the caller does not know, and the two null labels are returned as before with
    ``uninjected_is_a_null`` None."""
    missing = [k for k in KAPPAS if not any(math.isclose(k, float(x)) for x in r19_passes_by_kappa)]
    if missing:
        raise ValueError(f"power check needs every registered kappa; missing {missing}")
    passing = sorted(float(k) for k, ok in r19_passes_by_kappa.items() if ok)
    kmin = passing[0] if passing else None
    informative = kmin is not None and kmin <= KAPPA_MIN_INFORMATIVE + 1e-12
    is_null = None if uninjected_verdict in (None, "") else str(uninjected_verdict).upper() == "FAIL"
    if is_null is False:
        verdict = "POWERED_NOT_A_NULL" if informative else "NOT_A_NULL_UNDERPOWERED"
    else:
        verdict = "INFORMATIVE_NULL" if informative else "UNDECIDED_UNDERPOWERED"
    return {"kappa_min": kmin, "verdict": verdict, "informative": bool(informative),
            "uninjected_is_a_null": is_null, "uninjected_verdict": None if uninjected_verdict in (None, "")
            else str(uninjected_verdict)}


@dataclass(frozen=True)
class InjectedRun:
    """One injected power-check dataset (section 8, X(?) rows resolved by the orchestrator 2026-09-15), shared by
    EVERY arm of the contrast.

    ``frame``                 the rows the injected refits may use: the input rows minus every unknown-state (X(?))
                              row, with ``y_col`` replaced by ``y + kappa * s``
    ``dropped_unknown_state`` the X(?) rows removed (they have no state-level u)
    ``scored_index``          the scored rows, identical to the un-injected run's (checked)
    ``signal``                ``s_row`` over the input rows (NaN on X(?) rows)
    ``input_index``           the input rows, in input order (the universe a fold's training rows are taken from)"""

    kappa: float
    seed: int
    y_col: str
    frame: pd.DataFrame = field(repr=False)
    dropped_unknown_state: pd.Index = field(repr=False)
    scored_index: pd.Index = field(repr=False)
    signal: pd.Series = field(repr=False)
    input_index: pd.Index = field(repr=False, default=None)

    def _universe(self) -> pd.Index:
        if self.input_index is None:
            raise ValueError("InjectedRun without input_index: build it with prepare_injected_run")
        return self.input_index

    def training_index(self, hidden_index: Iterable[Any]) -> pd.Index:
        """The training rows of an injected refit of one outer fold: the run's input rows minus the fold's hidden rows
        (every row the design removes, scored or not) minus the X(?) rows (task X, leakage finding VR-03: training is
        derived from the fold's hidden rows, never taken from a caller's training list)."""
        uni = self._universe()
        hid = pd.Index(list(hidden_index))
        if hid.has_duplicates:
            raise ValueError("hidden rows must be unique")
        if not hid.isin(uni).all():
            raise KeyError("hidden rows outside the injected run's input rows")
        return uni[~uni.isin(hid) & ~uni.isin(self.dropped_unknown_state)]

    def fold_inputs(self, hidden_index: Iterable[Any], fold_scored_index: Iterable[Any], *,
                    outer_training_index: Iterable[Any] | None = None,
                    uninjected_scored_index: Iterable[Any] | None = None) -> tuple[pd.DataFrame, pd.Index]:
        """``(injected training frame, scored index)`` of one outer fold, the same for every arm.

        Training is exactly the input rows minus ``hidden_index`` (the fold's hidden rows) minus the X(?) rows
        (:meth:`training_index`).  ``outer_training_index``, when given (the un-injected fold's training rows), must equal
        the input rows minus the hidden rows, so a hidden-but-unscored row (a component-sharing state row, a
        group-masked row) passed as training is refused.  The scored rows must be hidden rows of the fold and scored
        rows of the run (never X(?)) and, when given, identical to the un-injected fold's."""
        tr = self.training_index(hidden_index)
        hid = pd.Index(list(hidden_index))
        uni = self._universe()
        if outer_training_index is not None:
            given = pd.Index(list(outer_training_index))
            want = uni[~uni.isin(hid)]
            if given.has_duplicates or set(given) != set(want):
                raise AssertionError(f"outer training rows are not the input rows minus the fold's hidden rows: "
                                     f"{len(given.difference(want))} extra (hidden rows among them: "
                                     f"{int(given.isin(hid).sum())}), {len(want.difference(given))} missing")
        sc = pd.Index(list(fold_scored_index))
        if uninjected_scored_index is not None:
            assert_same_scored_rows(pd.Index(list(uninjected_scored_index)), sc)
        if not sc.isin(self.scored_index).all():
            raise AssertionError("an injected fold scores a row the un-injected run does not score")
        if not sc.isin(hid).all():
            raise AssertionError("an injected fold scores a row that is not hidden in the fold")
        if sc.isin(tr).any():
            raise AssertionError("a scored row is an injected training row")
        return self.frame.loc[tr], sc


def assert_same_scored_rows(uninjected: pd.Index, injected: pd.Index) -> None:
    """The power check scores exactly the un-injected rows (section 8: "the same scored rows")."""
    a, b = pd.Index(uninjected), pd.Index(injected)
    if a.has_duplicates or b.has_duplicates or set(a) != set(b):
        raise AssertionError(f"injected scored rows differ from the un-injected run: {len(b.difference(a))} added, "
                             f"{len(a.difference(b))} missing")


def prepare_injected_run(frame: pd.DataFrame, *, kappa: float, seed: int, scored_index: Iterable[Any],
                         y_col: str = EM.Y_COL, state_col: str = EM.METAL_STATE_COL, system_col: str = EM.SYSTEM_COL,
                         u_share: Mapping[str, str] | None = None) -> InjectedRun:
    """Build the injected dataset of the section 8 power check.

    The signal ``s_row = u_m v_s`` (:func:`injected_signal`) is drawn on ``frame`` as given (all MODEL rows, so the
    draw does not depend on the drop), then every X(?) row is **dropped from the injected refits** of every arm (they
    have no state-level u) and ``y' = y + kappa * s`` is applied to the remaining rows, training and test alike.
    ``scored_index`` is the un-injected run's scored rows: none may be an X(?) row (X(?) rows are scored in no design,
    section 2), so the injected run scores exactly the same rows."""
    if frame.index.has_duplicates:
        raise ValueError("the row index must be unique")
    for c in (y_col, state_col, system_col):
        if c not in frame.columns:
            raise KeyError(c)
    sc = pd.Index(list(scored_index))
    if not sc.isin(frame.index).all():
        raise KeyError("scored rows outside the frame")
    unknown = frame[state_col].isna().to_numpy(dtype=bool)
    if frame.loc[sc, state_col].isna().any():
        raise AssertionError("an X(?) row is among the scored rows; X(?) rows are scored in no design (section 2)")
    signal = injected_signal(frame, seed=seed, state_col=state_col, system_col=system_col, u_share=u_share)
    kept = frame.loc[~unknown].copy()
    kept[y_col] = inject_targets(pd.to_numeric(kept[y_col], errors="coerce").astype(float), signal.loc[kept.index],
                                 kappa).to_numpy()
    return InjectedRun(kappa=float(kappa), seed=int(seed), y_col=y_col, frame=kept,
                       dropped_unknown_state=frame.index[unknown], scored_index=sc, signal=signal,
                       input_index=frame.index.copy())
