"""gen20.1 inference (IMPLEMENTATION_PLAN.md §6; PRE_REGISTRATION.md §6): chemotype and publication cluster
bootstraps with per-draw recomputation, percentile and BCa intervals, LOCO, the wild-cluster-t test on
per-extractant deltas, and the three MDEs.

Draws.  One count matrix per (unit, sorted cluster list): ``gen20k.inference._draw_counts(G, 10_000, seed)``
with seed 8675309 (chemotype) or 20260930 (publication), cached and shared by every arm, baseline, target and
split seed (LT-17).  Seeds are never resampled: each seed's statistic is recomputed under the same draw and the
draw's value is the mean over seeds (NA if any seed is NA in that draw).

Per-draw recomputation from per-cluster sufficient statistics.  A draw multiplies every row of cluster K by
its count c_K.  Then (n_l rows of extractant l, L_g extractants of chemotype g):

* MET-01 (plan §6, gen20 ``blocked_unit_contrast`` generalised): macro = Σ_l u_l·MAE_l* / Σ_l u_l with
  MAE_l* = Σ_K c_K S_lK / Σ_K c_K n_lK and u_l = Σ_K c_K n_lK / n_l, which equals Σ_i c_i |e_i|/n_l /
  Σ_i c_i/n_l: a ratio of two cluster sums with row weights 1/n_l.  MET-20 is the same on |ỹ − ŷ̃|; MET-03
  uses row weights 1/(n_l·L_g).
* MET-04 / MET-07 (weights recomputed): the extractant weight is the same u_l; within an extractant the
  balanced structure is recomputed on the draw with drawn cluster copies counted as distinct units:
  B_l* = Σ_K c_K A_lK / Σ_K c_K P_lK, A_lK = Σ_{i∈l,K} f_i/(C_lp·N_lpc), P_lK = publications of l in K, and
  BAL*(f) = Σ_l u_l B_l* / Σ_l u_l.  The implied row weights sum to 1 in every draw (LT-22; see
  ``balanced_draw_weights``).  Under chemotype draws this is exactly MET-04 recomputed on the materialised
  resample (copies of a chemotype are distinct extractants); under publication draws P_l* = Σ_{p∈l} c_p.
* R² family: denominators about the draw's own (weighted) mean; MET-06 SST = Σc y² − (Σc y)²/Σc;
  MET-08 / MET-09 / MET-10 / MET-11 from cluster sums (cells are nested in both units).

Intervals.  Percentile (np.quantile, linear) and BCa (gen20k.inference._bca semantics: mid-rank z0 from the
finite draws, acceleration from the LOCO jackknife) at 95 % and 98.75 % for chemotype draws; percentile for
publication draws.  NA draws are kept: −∞ for lower bounds, +∞ for upper bounds, counted unfavourably in the
two-sided p; more than 1 % NA draws makes the interval NA ("degenerate").  A unit with one cluster gives NA
("one cluster"); fewer than 10 clusters is flagged "preliminary".

WCR.  ``gen13sep.wildcluster.wild_cluster_t`` (CR2, null imposed, Rademacher; Webb 6-point when G < 12; 9,999
replications, seed 8675309) on the per-extractant seed-mean deltas clustered by chemotype; mde80_cr =
``detectable(se_cr2, dof_BM, alpha=0.0125)``; mde80_95 = 2.80·se_boot; mde80_dec = (z_0.99375 + z_0.80)·se_boot.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import bootstrap  # noqa: F401
from . import metrics as M
from gen20k import inference as GI  # noqa: E402
from gen13sep import wildcluster as WC  # noqa: E402

REPLICATES = 10_000
UNIT_SEEDS: dict[str, int] = {"chemotype": 8675309, "publication": 20260930}
UNIT_COLUMNS: dict[str, str] = {"chemotype": "chemotype", "publication": "publication_id"}
UNITS: tuple[str, ...] = ("chemotype", "publication")
WCR_REPS = 9_999
WCR_SEED = 8675309
WEBB_BELOW = 12
ALPHA_K = 0.0125
LEVELS: tuple[float, ...] = (0.95, 0.9875)
NA_DRAW_LIMIT = 0.01
PRELIMINARY_BELOW = 10
MDE95_FACTOR = 2.80
Z_DEC = float(norm.ppf(1.0 - ALPHA_K / 2.0) + norm.ppf(0.80))          # 3.3393...

#: statistics where lower is better (contrast Δ = ref − cand) and where higher is better (Δ = cand − ref)
ERROR_STATS = ("MET01", "MET02", "MET03", "MET04_MAE", "MET04_RMSE", "MET05", "MET20")
SCORE_STATS = ("MET06", "MET07", "MET07W", "MET08", "MET09", "MET10", "MET11")
LEVEL_ONLY_STATS = ("MET04_BIAS", "S_B")
#: the direct (metrics.summarise) key of each statistic, for the point self-check
DIRECT_KEY = {"MET01": "MET01_MAE_macro_ext", "MET02": "MET02_MAE_pooled", "MET03": "MET03_MAE_macro_chem",
              "MET04_MAE": "MET04_MAE_bal", "MET04_RMSE": "MET04_RMSE_bal", "MET04_BIAS": "MET04_Bias_bal",
              "MET05": "MET05_RMSE_pooled", "MET06": "R2_pooled", "MET07": "R2_balanced",
              "MET07W": "R2_balanced_within", "MET08": "R2_within_cell", "MET09": "R2_level", "MET10": "R2_0",
              "MET20": "MET20_MAE_macro_ext_within", "S_B": "s_b"}
_REL_TOL = 1e-12


def level_tag(level: float) -> str:
    return f"{level * 100:g}".replace(".", "")


# --------------------------------------------------------------------------------------
# draw matrices (one per (unit, cluster list), shared by everything)
# --------------------------------------------------------------------------------------
_DRAWS: dict = {}
_POINT: dict = {}
_LOCO: dict = {}


def draw_matrix(unit: str, clusters, replicates: int = REPLICATES) -> np.ndarray:
    """``gen20k.inference._draw_counts(G, replicates, seed_of(unit))`` over the sorted cluster list (cached;
    the same object for every caller with the same unit and cluster list)."""
    clusters = tuple(str(c) for c in clusters)
    if list(clusters) != sorted(clusters):
        raise ValueError("the cluster list must be sorted")
    key = (unit, clusters, int(replicates))
    if key not in _DRAWS:
        D = GI._draw_counts(len(clusters), int(replicates), UNIT_SEEDS[unit])
        D.setflags(write=False)
        _DRAWS[key] = D
    return _DRAWS[key]


def _point_w(G: int) -> np.ndarray:
    if G not in _POINT:
        w = np.ones((1, G))
        w.setflags(write=False)
        _POINT[G] = w
    return _POINT[G]


def _loco_w(G: int) -> np.ndarray:
    if G not in _LOCO:
        w = np.ones((G, G)) - np.eye(G)
        w.setflags(write=False)
        _LOCO[G] = w
    return _LOCO[G]


# --------------------------------------------------------------------------------------
# a population prepared for inference
# --------------------------------------------------------------------------------------
class Frame:
    """One scored population of one (design, variant) long table: rows sorted by key, seed-invariant, with the
    per-row weights and cluster codes every statistic needs.  ``arms`` are checked for coverage up front."""

    def __init__(self, table: pd.DataFrame, *, name: str = "", arms=None):
        self.name = name
        self.target = M.target_of(table)
        self.seeds = M.check_seed_invariant(table)
        if arms is not None:
            M.check_coverage(table, list(arms))
        self._parts = [M._sorted_seed(table, s) for s in self.seeds]
        self.rows = self._parts[0]
        r = M.SeedRows(self.rows)
        self.n = r.n
        self.y = r.y
        self.yc = r.y - r.y.mean()                    # shifted for conditioning (SST is shift-invariant)
        self.cell = r.cell
        self.ext, self.ext_names = r.ext, r.ext_names
        self.L = len(r.ext_names)
        self.one = np.ones(self.n)
        self.n_l = r.n_l
        self.n_l_vec = np.bincount(r.ext).astype(float)
        self.v_ext = 1.0 / r.n_l
        self.v_chem = 1.0 / (r.n_l * r.L_g)
        self.yt = r.yt                                 # ỹ (cell-centred y)
        self.ybar_cc = M._group_mean(self.yc, self.cell)   # cell means of the shifted y
        self.w_bal = r.w_bal
        # within-extractant balanced part b_i = 1/(C_lp N_lpc) = L·P_l·w_i
        P_l = pd.Series(self.rows["publication_id"].astype(str).to_numpy()).groupby(self.ext).nunique()
        self.P_l_row = P_l.to_numpy(dtype=float)[self.ext]
        self.b = self.L * self.P_l_row * r.w_bal
        self.ext_chem = pd.DataFrame({"e": self.ext, "g": self.rows["chemotype"].astype(str).to_numpy()}) \
            .drop_duplicates().sort_values("e")["g"].to_numpy()
        self._pred: dict = {}
        self._clusters: dict = {}
        self._bal_cache: dict = {}
        self._seedrows = [r] + [None] * (len(self._parts) - 1)
        self._direct_cache: dict = {}

    def seedrows(self) -> list:
        """metrics.SeedRows of every seed (built once; the row structure is seed-invariant)."""
        for k, p in enumerate(self._parts):
            if self._seedrows[k] is None:
                self._seedrows[k] = M.SeedRows(p)
        return self._seedrows

    # -- inputs ---------------------------------------------------------------------
    def has(self, arm: str) -> bool:
        """'run' = the column exists and is finite somewhere; a partially non-finite column then fails coverage in
        ``pred`` (a run failure), never 'not run'."""
        return any(M.has_arm(p, arm) for p in self._parts)

    def pred(self, arm: str) -> np.ndarray:
        """(n_seeds, n) predictions; coverage asserted on first use."""
        if arm not in self._pred:
            P = np.vstack([p[arm].to_numpy(dtype=float) for p in self._parts])
            if not np.isfinite(P).all():
                raise M.CoverageError(f"{self.name}: arm {arm}: n_pred_finite != n_rows")
            self._pred[arm] = P
        return self._pred[arm]

    def cluster(self, unit: str) -> tuple[np.ndarray, np.ndarray]:
        if unit not in self._clusters:
            names, code = np.unique(self.rows[UNIT_COLUMNS[unit]].astype(str).to_numpy(), return_inverse=True)
            self._clusters[unit] = (names, code.astype(np.int64))
        return self._clusters[unit]

    def clusters(self, unit: str) -> list[str]:
        return [str(x) for x in self.cluster(unit)[0]]

    def csum(self, unit: str, v: np.ndarray) -> np.ndarray:
        names, code = self.cluster(unit)
        return np.bincount(code, weights=v, minlength=len(names))

    # -- MET-04 balanced aggregation --------------------------------------------------
    def _bal_parts(self, unit: str, W: np.ndarray):
        # cache only the module's registered weight matrices (draws, point, LOCO): they live for the whole process,
        # so their id() can never be reused by a temporary array
        cacheable = any(W is v for v in (*_DRAWS.values(), *_POINT.values(), *_LOCO.values()))
        key = (unit, id(W), W.shape)
        if not cacheable or key not in self._bal_cache:
            names, K = self.cluster(unit)
            G, L = len(names), self.L
            lk = K * L + self.ext
            pubs = self.rows["publication_id"].astype(str).to_numpy()
            P_lK = pd.DataFrame({"lk": lk, "p": pubs}).groupby("lk")["p"].nunique()
            P = np.zeros(G * L)
            P[P_lK.index.to_numpy()] = P_lK.to_numpy(dtype=float)
            N = np.bincount(lk, minlength=G * L).astype(float)
            den = W @ P.reshape(G, L)
            u = (W @ N.reshape(G, L)) / self.n_l_vec[None, :]
            parts = (lk, G, L, den, u, u.sum(axis=1))
            if not cacheable:
                return parts
            self._bal_cache[key] = parts
        return self._bal_cache[key]

    def bal(self, unit: str, W: np.ndarray, f: np.ndarray) -> np.ndarray:
        """BAL*(f) = Σ_l u_l B_l*(f) / Σ_l u_l for every row of W (see the module docstring)."""
        lk, G, L, den, u, usum = self._bal_parts(unit, W)
        A = np.bincount(lk, weights=self.b * f, minlength=G * L).reshape(G, L)
        num = W @ A
        with np.errstate(invalid="ignore", divide="ignore"):
            B = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
            return (u * B).sum(axis=1) / usum


def balanced_draw_weights(F: Frame, unit: str, counts: np.ndarray) -> np.ndarray:
    """The total MET-04 weight a draw with cluster counts ``counts`` puts on each original row (all of its copies
    together): w_i* = (u_l/Σu)·c_K/(Σ_K c_K P_lK)·b_i.  Σ_i w_i* = 1 for every draw (LT-22)."""
    W = np.asarray(counts, dtype=float).reshape(1, -1)
    lk, G, L, den, u, usum = F._bal_parts(unit, W)
    _, K = F.cluster(unit)
    c = W[0][K]
    den_row = den[0].reshape(-1)[F.ext]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den_row > 0, (u[0][F.ext] / usum[0]) * c / den_row * F.b, 0.0)


# --------------------------------------------------------------------------------------
# statistics from cluster sums
# --------------------------------------------------------------------------------------
def _r2v(num, den, n, scale=None):
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (n >= 2) & (den > (0.0 if scale is None else _REL_TOL * scale))
        return np.where(ok, 1.0 - num / np.where(ok, den, 1.0), np.nan)


def _one(F: Frame, unit: str, W: np.ndarray, stat: str, arm: str | None, ref: str | None, k: int) -> np.ndarray:
    def agg(v):
        return W @ F.csum(unit, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat == "S_B":
            n, sy, syy = agg(F.one), agg(F.yc), agg(F.yc * F.yc)
            sst = syy - sy * sy / n
            ssb = agg(F.ybar_cc * F.ybar_cc) - sy * sy / n
            return np.where(sst > _REL_TOL * syy, ssb / np.where(sst > 0, sst, 1.0), np.nan)
        p = F.pred(arm)[k]
        e = F.y - p
        if stat == "MET01":
            return agg(F.v_ext * np.abs(e)) / agg(F.v_ext)
        if stat == "MET02":
            return agg(np.abs(e)) / agg(F.one)
        if stat == "MET03":
            return agg(F.v_chem * np.abs(e)) / agg(F.v_chem)
        if stat == "MET05":
            return np.sqrt(agg(e * e) / agg(F.one))
        if stat == "MET06":
            n, sy, syy = agg(F.one), agg(F.yc), agg(F.yc * F.yc)
            return _r2v(agg(e * e), syy - sy * sy / n, n, syy)
        if stat == "MET04_MAE":
            return F.bal(unit, W, np.abs(e))
        if stat == "MET04_RMSE":
            return np.sqrt(F.bal(unit, W, e * e))
        if stat == "MET04_BIAS":
            return F.bal(unit, W, -e)
        if stat == "MET07":
            m, m2 = F.bal(unit, W, F.yc), F.bal(unit, W, F.yc * F.yc)
            return _r2v(F.bal(unit, W, e * e), m2 - m * m, agg(F.one), m2)
        if stat == "MET10":
            return _r2v(agg(e * e), agg(F.y * F.y), agg(F.one))
        if stat == "MET11":
            eY = F.y - F.pred(ref)[k]
            sY = agg(eY * eY)
            return np.where(sY > 0, 1.0 - agg(e * e) / np.where(sY > 0, sY, 1.0), np.nan)
        pt = p - M._group_mean(p, F.cell)
        dw = F.yt - pt
        if stat == "MET07W":
            return _r2v(F.bal(unit, W, dw * dw), F.bal(unit, W, F.yt * F.yt), agg(F.one))
        if stat == "MET08":
            return _r2v(agg(dw * dw), agg(F.yt * F.yt), agg(F.one))
        if stat == "MET09":
            pbar = M._group_mean(p, F.cell) - F.y.mean()
            n, sy = agg(F.one), agg(F.yc)
            sbb = agg(F.ybar_cc * F.ybar_cc)
            return _r2v(agg((F.ybar_cc - pbar) ** 2), sbb - sy * sy / n, n, sbb)
        if stat == "MET20":
            return agg(F.v_ext * np.abs(dw)) / agg(F.v_ext)
    raise ValueError(f"unknown statistic {stat}")


def seed_values(F: Frame, unit: str, W: np.ndarray, stat: str, arm: str | None, ref: str | None = None, *,
                contrast: bool = False) -> np.ndarray:
    """(n_seeds, rows of W): the statistic (or the contrast in the candidate's favour) per seed."""
    out = np.empty((len(F.seeds), W.shape[0]))
    for k in range(len(F.seeds)):
        if not contrast:
            out[k] = _one(F, unit, W, stat, arm, ref, k)
        elif stat in ERROR_STATS:
            out[k] = _one(F, unit, W, stat, ref, None, k) - _one(F, unit, W, stat, arm, None, k)
        elif stat in SCORE_STATS and stat != "MET11":
            out[k] = _one(F, unit, W, stat, arm, None, k) - _one(F, unit, W, stat, ref, None, k)
        else:
            raise ValueError(f"no contrast orientation for {stat}")
    return out


def seed_mean(v: np.ndarray) -> np.ndarray:
    """Mean over seeds per column; NA if any seed is NA in that column."""
    return v.mean(axis=0)


# --------------------------------------------------------------------------------------
# quantiles, BCa, p with NA draws
# --------------------------------------------------------------------------------------
def _lerp(x0: float, x1: float, f: float) -> float:
    d = x1 - x0
    return float(x0 + d * f) if f < 0.5 else float(x1 - d * (1.0 - f))


def quantile_na(arr: np.ndarray, q: float) -> float:
    """np.quantile (linear) for finite arrays; with ±inf entries the same order statistic, returning ±inf when the
    interpolation touches an infinite draw."""
    a = np.asarray(arr, dtype=float)
    if not a.size:
        return float("nan")
    if np.isfinite(a).all():
        return float(np.quantile(a, q))
    s = np.sort(a)
    h = (len(s) - 1) * q
    lo = int(math.floor(h))
    hi = min(lo + 1, len(s) - 1)
    f = h - lo
    x0, x1 = s[lo], s[hi]
    if f == 0 or lo == hi:
        return float(x0)
    if x0 == -np.inf:
        return float("-inf")
    if x1 == np.inf:
        return float("inf")
    return _lerp(float(x0), float(x1), f)


def bca(draws: np.ndarray, point: float, jack: np.ndarray, level: float) -> tuple[float, float]:
    """gen20k.inference._bca semantics (identical when no draw is NA); NA draws are kept as −∞ (lower bound) and
    +∞ (upper bound); more than 1 % NA draws -> NA."""
    draws = np.asarray(draws, dtype=float)
    na = ~np.isfinite(draws)
    fin = draws[~na]
    jk = np.asarray(jack, dtype=float)
    jk = jk[np.isfinite(jk)]
    if not draws.size or na.mean() > NA_DRAW_LIMIT:
        return float("nan"), float("nan")
    if len(fin) < 50 or len(jk) < 3 or not np.isfinite(point):
        return float("nan"), float("nan")
    prop = (np.sum(fin < point) + 0.5 * np.sum(fin == point)) / len(fin)
    if prop <= 0.0 or prop >= 1.0:
        return float("nan"), float("nan")
    z0 = norm.ppf(prop)
    m = jk.mean()
    num = ((m - jk) ** 3).sum()
    den = 6.0 * (((m - jk) ** 2).sum()) ** 1.5
    if den <= 0:
        return float("nan"), float("nan")
    a = num / den
    alpha = (1.0 - level) / 2.0
    lo_arr = np.where(na, -np.inf, draws)
    hi_arr = np.where(na, np.inf, draws)
    out = []
    for q, arr in ((alpha, lo_arr), (1.0 - alpha, hi_arr)):
        z = norm.ppf(q)
        adj = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(quantile_na(arr if na.any() else fin, min(max(adj, 0.0), 1.0)))
    return out[0], out[1]


def p_two_sided(draws: np.ndarray) -> float:
    """Two-sided percentile p, NA draws counted unfavourably (on both sides)."""
    d = np.asarray(draws, dtype=float)
    if not d.size:
        return float("nan")
    na = ~np.isfinite(d)
    fin = d[~na]
    n_na = int(na.sum())
    R = len(d)
    return float(min(1.0, 2.0 * min((np.sum(fin <= 0) + n_na) / R, (np.sum(fin >= 0) + n_na) / R)))


def interval_block(draws: np.ndarray, point: float, jack: np.ndarray | None, prefix: str) -> dict:
    """Percentile (and, with ``jack``, BCa) bounds at 95 % and 98.75 %, the two-sided p, se_boot, NA draw counts."""
    d = np.asarray(draws, dtype=float)
    na = ~np.isfinite(d)
    frac = float(na.mean()) if d.size else float("nan")
    degenerate = bool(d.size and frac > NA_DRAW_LIMIT)
    out = {f"{prefix}_na_draws": int(na.sum()), f"{prefix}_na_frac": frac, f"{prefix}_degenerate": degenerate,
           f"{prefix}_status": "degenerate" if degenerate else "ok"}
    lo_arr, hi_arr = np.where(na, -np.inf, d), np.where(na, np.inf, d)
    for lv in LEVELS:
        tag = level_tag(lv)
        alpha = (1.0 - lv) / 2.0
        if degenerate or not d.size:
            lo = hi = float("nan")
        else:
            lo, hi = quantile_na(lo_arr, alpha), quantile_na(hi_arr, 1.0 - alpha)
        out[f"{prefix}_pct{tag}_low"], out[f"{prefix}_pct{tag}_high"] = lo, hi
        if jack is not None:
            bl, bh = bca(d, point, jack, lv) if not degenerate else (float("nan"), float("nan"))
            out[f"{prefix}_bca{tag}_low"], out[f"{prefix}_bca{tag}_high"] = bl, bh
    fin = d[~na]
    out[f"{prefix}_p_two_sided"] = p_two_sided(d)
    out[f"{prefix}_se"] = float(fin.std(ddof=1)) if fin.size > 1 else float("nan")
    return out


def _na_block(prefix: str, status: str, bca_cols: bool) -> dict:
    out = {f"{prefix}_na_draws": 0, f"{prefix}_na_frac": float("nan"), f"{prefix}_degenerate": True,
           f"{prefix}_status": status, f"{prefix}_p_two_sided": float("nan"), f"{prefix}_se": float("nan")}
    for lv in LEVELS:
        tag = level_tag(lv)
        out[f"{prefix}_pct{tag}_low"] = out[f"{prefix}_pct{tag}_high"] = float("nan")
        if bca_cols:
            out[f"{prefix}_bca{tag}_low"] = out[f"{prefix}_bca{tag}_high"] = float("nan")
    return out


# --------------------------------------------------------------------------------------
# the main entry: a level or a contrast with both units, LOCO and MDEs
# --------------------------------------------------------------------------------------
def infer(F: Frame, stat: str, arm: str | None, ref: str | None = None, *, contrast: bool = False,
          units=UNITS, replicates: int = REPLICATES, keep_draws: bool = False) -> dict:
    """Seed-mean point, per-seed values, chemotype percentile + BCa (LOCO jackknife) and publication percentile
    intervals at 95 % and 98.75 %, two-sided p, se_boot, LOCO over chemotypes, mde80_95 and mde80_dec.

    ``contrast``: Δ in the candidate ``arm``'s favour against ``ref`` (error statistics ref − cand, scores
    cand − ref).  Without ``contrast``, ``ref`` is only used by MET11 (Skill_MSE(arm | ref))."""
    names0, _ = F.cluster("chemotype")
    pt = seed_values(F, "chemotype", _point_w(len(names0)), stat, arm, ref, contrast=contrast)[:, 0]
    point = float(pt.mean())
    out = {"population": F.name, "stat": stat, "arm": arm, "ref": ref, "contrast": bool(contrast),
           "point": point, "per_seed": [float(v) for v in pt], "n_seeds": len(F.seeds),
           "seeds": list(F.seeds), "n_rows": F.n, "n_cells": int(len(np.unique(F.cell))), "n_extractants": F.L}
    if contrast:
        out["seeds_positive"] = int(np.sum(pt > 0))
    draws_out = {}
    for unit in units:
        names, _ = F.cluster(unit)
        G = len(names)
        pre = "chem" if unit == "chemotype" else "pub"
        out[f"{pre}_n_clusters"] = G
        out[f"{pre}_preliminary"] = bool(G < PRELIMINARY_BELOW)
        if G < 2:
            out.update(_na_block(pre, "one cluster", unit == "chemotype"))
            if unit == "chemotype":
                out.update({"loco": {}, "loco_min": float("nan"), "loco_min_cluster": None, "loco_n_na": G,
                            "loco_n_positive": 0, "loco_all_positive": False})
            continue
        D = draw_matrix(unit, names, replicates)
        dr = seed_mean(seed_values(F, unit, D, stat, arm, ref, contrast=contrast))
        jack = None
        if unit == "chemotype":
            jack = seed_mean(seed_values(F, unit, _loco_w(G), stat, arm, ref, contrast=contrast))
            fin = np.isfinite(jack)
            out["loco"] = {str(n): float(v) for n, v in zip(names, jack)}
            out["loco_n_na"] = int((~fin).sum())
            out["loco_min"] = float(jack[fin].min()) if fin.any() else float("nan")
            out["loco_min_cluster"] = str(names[fin][np.argmin(jack[fin])]) if fin.any() else None
            out["loco_n_positive"] = int(np.sum(jack[fin] > 0))
            out["loco_all_positive"] = bool(fin.all() and np.all(jack > 0))
        out.update(interval_block(dr, point, jack, pre))
        if keep_draws:
            draws_out[unit] = dr
    se = out.get("chem_se", float("nan"))
    out["mde80_95"] = MDE95_FACTOR * se if np.isfinite(se) else float("nan")
    out["mde80_dec"] = Z_DEC * se if np.isfinite(se) else float("nan")
    if keep_draws:
        out["_draws"] = draws_out
    return out


def point(F: Frame, stat: str, arm: str | None, ref: str | None = None, *, contrast: bool = False) -> dict:
    """Seed-mean point and per-seed values only (no draws)."""
    names, _ = F.cluster("chemotype")
    pt = seed_values(F, "chemotype", _point_w(len(names)), stat, arm, ref, contrast=contrast)[:, 0]
    return {"point": float(pt.mean()), "per_seed": [float(v) for v in pt]}


# --------------------------------------------------------------------------------------
# WCR on per-extractant deltas, MDEs
# --------------------------------------------------------------------------------------
def per_extractant_deltas(F: Frame, cand: str, ref: str, *, kind: str = "abs") -> pd.DataFrame:
    """Δ_l = MAE_l(ref) − MAE_l(cand), MAE_l the seed mean of the per-seed extractant MAE (|e| for MET-01, |ỹ −
    ŷ̃| for MET-20), with the extractant's chemotype.  mean_l Δ_l equals the MET-01 (MET-20) seed-mean Δ."""
    def mae_l(arm):
        P = F.pred(arm)
        vals = []
        for k in range(len(F.seeds)):
            if kind == "abs":
                a = np.abs(F.y - P[k])
            else:
                a = np.abs(F.yt - (P[k] - M._group_mean(P[k], F.cell)))
            vals.append(np.bincount(F.ext, weights=a) / F.n_l_vec)
        return np.mean(vals, axis=0)
    d = mae_l(ref) - mae_l(cand)
    return pd.DataFrame({"extractant": F.ext_names, "chemotype": F.ext_chem, "delta": d})


def wcr(d, clusters, *, reps: int = WCR_REPS, seed: int = WCR_SEED, alpha: float = ALPHA_K) -> dict:
    """Wild-cluster-t (CR2, null imposed; Rademacher, Webb when G < 12) on per-unit deltas, plus the CR2 SE,
    Bell-McCaffrey dof and mde80_cr = detectable(se_cr2, dof_BM, alpha = 0.0125)."""
    d = np.asarray(d, dtype=float)
    cl = np.asarray(clusters).astype(str)
    ok = np.isfinite(d)
    d, cl = d[ok], cl[ok]
    G = len(np.unique(cl))
    out = {"wcr_mean": float(d.mean()) if d.size else float("nan"), "wcr_n_units": int(d.size),
           "wcr_n_clusters": int(G), "wcr_weights": "webb" if G < WEBB_BELOW else "rademacher",
           "wcr_reps": int(reps), "wcr_seed": int(seed)}
    if d.size < 3 or G < 2:
        out.update({"p_wcr": float("nan"), "p_wcu": float("nan"), "wcr_t_obs": float("nan"),
                    "se_cr2": float("nan"), "dof_bm": float("nan"), "mde80_cr": float("nan"),
                    "wcr_status": "too few units or clusters"})
        return out
    w = WC.wild_cluster_t(d, cl, reps=reps, seed=seed, weights=out["wcr_weights"], se_kind="CR2", ci=False)
    cr = WC.cluster_robust(d, cl)
    mde = WC.detectable(cr.se_cr2, cr.dof_bm, alpha=alpha) if np.isfinite(cr.dof_bm) and cr.se_cr2 > 0 else float("nan")
    out.update({"p_wcr": float(w.p_wcr), "p_wcu": float(w.p_wcu), "wcr_t_obs": float(w.t_obs),
                "se_cr2": float(cr.se_cr2), "dof_bm": float(cr.dof_bm), "kish_ext": float(cr.kish_clusters),
                "mde80_cr": float(mde), "wcr_status": "ok"})
    return out


# --------------------------------------------------------------------------------------
# evidence for the decision rules (rules.py consumes these dicts)
# --------------------------------------------------------------------------------------
def _direct(F: Frame, arm: str) -> dict:
    """metrics.py's direct per-seed metrics and seed means of ``arm`` on the frame's rows (cached per frame)."""
    if arm not in F._direct_cache:
        F._direct_cache[arm] = M.combine_seeds([M.seed_metrics(r, arm) for r in F.seedrows()], F.seeds)
    return F._direct_cache[arm]


def per_type_met01(F: Frame, arm: str) -> dict:
    """Seed-mean MET-01 of ``arm`` within each pair type (log SF) or metal (log D)."""
    col = "pair" if F.target == "logsf" else "metal"
    names, tc = np.unique(F.rows[col].astype(str).to_numpy(), return_inverse=True)
    te = tc * F.L + F.ext
    P = F.pred(arm)
    out = np.zeros(len(names))
    for k in range(len(F.seeds)):
        ae = np.abs(F.y - P[k])
        n_te = np.bincount(te, minlength=len(names) * F.L).reshape(len(names), F.L)
        s_te = np.bincount(te, weights=ae, minlength=len(names) * F.L).reshape(len(names), F.L)
        with np.errstate(invalid="ignore", divide="ignore"):
            mae = s_te / n_te
        out += np.nanmean(np.where(n_te > 0, mae, np.nan), axis=1)
    return {str(n): float(v / len(F.seeds)) for n, v in zip(names, out)}


def contrast_evidence(F: Frame, X: str, Y: str, *, flat: str = "FLAT", replicates: int = REPLICATES,
                      stat: str = "MET01", full: bool = True) -> dict:
    """Everything RULE-C needs for component X vs Y on one population of design BP (Δ positive favours X), plus
    the fixed annotations; (h) is added by the caller.  ``full=False`` skips (i)-(k) and the annotations (NC1 and
    exploratory tests)."""
    kind = "abs" if stat == "MET01" else "within"
    inf = infer(F, stat, X, Y, contrast=True, replicates=replicates)
    sx, sy = _direct(F, X), _direct(F, Y)
    dkey = DIRECT_KEY[stat]
    _selfcheck(inf["point"], sy[dkey] - sx[dkey], f"{F.name} {stat} {X} vs {Y}")
    dl = per_extractant_deltas(F, X, Y, kind=kind)
    w = wcr(dl["delta"].to_numpy(), dl["chemotype"].to_numpy())
    _selfcheck(w["wcr_mean"], inf["point"], f"{F.name} per-extractant mean {X} vs {Y}")
    ev = {"population": F.name, "target": F.target, "X": X, "Y": Y, "stat": stat,
          "delta": inf["point"], "delta_seeds": inf["per_seed"], "seeds_positive": inf["seeds_positive"],
          "met01_X": sx[dkey], "met01_Y": sy[dkey], "n_rows": F.n, "n_extractants": F.L}
    ev.update({k: v for k, v in inf.items() if k.startswith(("chem_", "pub_", "loco", "mde80"))})
    ev.update(w)
    if not full:
        return ev
    # (i) MAE_bal sign, (j) R2_balanced gain with both lower bounds and own R2_balanced, (k) Skill_MSE(X | FLAT)
    ev["delta_mae_bal"] = sy["MET04_MAE_bal"] - sx["MET04_MAE_bal"]
    ev["r2_bal_X"], ev["r2_bal_Y"] = sx["R2_balanced"], sy["R2_balanced"]
    if F.target == "logsf":
        jb = infer(F, "MET07", X, Y, contrast=True, replicates=replicates)
        ev["delta_r2_bal"] = jb["point"]
        ev["delta_r2_bal_chem_low"] = jb["chem_pct9875_low"]
        ev["delta_r2_bal_pub_low"] = jb["pub_pct9875_low"]
        if F.has(flat):
            ev["skill_flat_X"] = point(F, "MET11", X, flat)["point"]
        else:
            ev["skill_flat_X"] = float("nan")
    # fixed annotations (never change a verdict)
    ev["ann_delta_mae_pooled"] = sy["MET02_MAE_pooled"] - sx["MET02_MAE_pooled"]
    ev["ann_delta_mae_macro_chem"] = sy["MET03_MAE_macro_chem"] - sx["MET03_MAE_macro_chem"]
    sk = infer(F, "MET11", X, Y, contrast=False, replicates=replicates, units=("chemotype",))
    ev["ann_skill_mse_XY"] = sk["point"]
    ev["ann_skill_mse_XY_chem_low9875"] = sk["chem_pct9875_low"]
    ev["ann_delta_r2_pooled"] = sx["R2_pooled"] - sy["R2_pooled"]
    return ev


def threshold_evidence(F: Frame, arm: str, *, stat: str = "MET06", replicates: int = REPLICATES) -> dict:
    """RULE-T inputs for ``arm`` on MET-06 (T1, T2, T3) or MET-08 (T1w; condition (vi) then uses the within-cell
    R2_balanced): seed-mean point, per-seed values, chemotype 2.5/97.5th percentiles and 95 % BCa lower bound,
    publication 2.5/97.5th percentiles, 98.75 % lower bounds, LOCO minimum, se_boot, seed-mean R2_balanced."""
    inf = infer(F, stat, arm, replicates=replicates)
    d = _direct(F, arm)
    _selfcheck(inf["point"], d[DIRECT_KEY[stat]], f"{F.name} {stat} {arm}", na_ok=True)
    bal_key = "R2_balanced_within" if stat == "MET08" else "R2_balanced"
    ev = {"population": F.name, "arm": arm, "stat": stat, "point": inf["point"], "per_seed": inf["per_seed"],
          "reason": d[f"{DIRECT_KEY[stat]}_reason"], "n_seeds_defined": d[f"{DIRECT_KEY[stat]}_n_seeds_defined"],
          "r2_bal": d[bal_key], "r2_bal_reason": d[f"{bal_key}_reason"]}
    ev.update({k: v for k, v in inf.items() if k.startswith(("chem_", "pub_", "loco_min", "loco_n"))})
    return ev


def _selfcheck(a: float, b: float, what: str, *, tol: float = 1e-9, na_ok: bool = False) -> None:
    """The sufficient-statistic point must equal the direct (metrics.py) point."""
    if not (np.isfinite(a) and np.isfinite(b)):
        if np.isfinite(a) != np.isfinite(b) and not (na_ok and not np.isfinite(b)):
            raise AssertionError(f"point self-check failed for {what}: {a} vs {b}")
        return
    if abs(a - b) > tol * max(1.0, abs(b)):
        raise AssertionError(f"point self-check failed for {what}: {a} vs {b}")


# --------------------------------------------------------------------------------------
# MET-17 with chemotype / publication intervals
# --------------------------------------------------------------------------------------
def met17_infer(logd: pd.DataFrame, logsf: pd.DataFrame, arm: str = "A", *, replicates: int = REPLICATES) -> dict:
    """MET-17 (λ, κ_adj, ρ_adj, ρ_within_adj) seed means with 95 % chemotype and publication percentile
    intervals, recomputed per draw from cluster sums of both row sets (same cluster lists)."""
    ends = M.endpoint_residuals(logd, logsf, arm)
    seeds = M.check_seed_invariant(logsf)
    dparts = [M._sorted_seed(logd, s) for s in seeds]
    eparts = [ends[ends["split_seed"] == s].reset_index(drop=True) for s in seeds]
    out = {}
    for unit in UNITS:
        col = UNIT_COLUMNS[unit]
        names = sorted(set(dparts[0][col].astype(str)) | set(eparts[0][col].astype(str)))
        pos = {n: i for i, n in enumerate(names)}
        G = len(names)
        cd = np.array([pos[x] for x in dparts[0][col].astype(str)], dtype=np.int64)
        ce = np.array([pos[x] for x in eparts[0][col].astype(str)], dtype=np.int64)
        Ws = {"point": _point_w(G)}
        if G >= 2:
            Ws["draws"] = draw_matrix(unit, names, replicates)
        res = {k: [] for k in Ws}
        for dp, ep in zip(dparts, eparts):
            r = dp["y"].to_numpy(dtype=float) - dp[arm].to_numpy(dtype=float)
            _, cc = np.unique(dp["ci"].to_numpy(), return_inverse=True)
            rbar = M._group_mean(r, cc)
            ra, rb = ep["r_a"].to_numpy(dtype=float), ep["r_b"].to_numpy(dtype=float)
            ta, tb = ep["rt_a"].to_numpy(dtype=float), ep["rt_b"].to_numpy(dtype=float)
            sd = {k: np.bincount(cd, weights=v, minlength=G) for k, v in (("rb2", rbar * rbar), ("r2", r * r))}
            se = {k: np.bincount(ce, weights=v, minlength=G) for k, v in (
                ("d2", (ra - rb) ** 2), ("s2", ra * ra + rb * rb), ("n", np.ones_like(ra)), ("a", ra), ("b", rb),
                ("aa", ra * ra), ("bb", rb * rb), ("ab", ra * rb), ("ta", ta), ("tb", tb), ("tata", ta * ta),
                ("tbtb", tb * tb), ("tatb", ta * tb))}
            for k, W in Ws.items():
                A = {kk: W @ v for kk, v in {**sd, **se}.items()}
                with np.errstate(invalid="ignore", divide="ignore"):
                    lam = A["rb2"] / A["r2"]
                    kap = A["d2"] / A["s2"]
                    rho = _pearson_sums(A["n"], A["a"], A["b"], A["aa"], A["bb"], A["ab"])
                    rhow = _pearson_sums(A["n"], A["ta"], A["tb"], A["tata"], A["tbtb"], A["tatb"])
                res[k].append(np.vstack([lam, kap, rho, rhow]))
        for j, name in enumerate(("lambda", "kappa_adj", "rho_adj", "rho_within_adj")):
            pts = np.array([x[j, 0] for x in res["point"]])
            out[name] = float(pts.mean())
            out[f"{name}_per_seed"] = pts.tolist()
            pre = "chem" if unit == "chemotype" else "pub"
            if "draws" in res:
                dr = np.mean([x[j] for x in res["draws"]], axis=0)
                blk = interval_block(dr, out[name], None, f"{name}_{pre}")
                out[f"{name}_{pre}_pct95_low"] = blk[f"{name}_{pre}_pct95_low"]
                out[f"{name}_{pre}_pct95_high"] = blk[f"{name}_{pre}_pct95_high"]
            else:
                out[f"{name}_{pre}_pct95_low"] = out[f"{name}_{pre}_pct95_high"] = float("nan")
    return out


def _pearson_sums(n, a, b, aa, bb, ab):
    cov = ab - a * b / n
    va = aa - a * a / n
    vb = bb - b * b / n
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt(va * vb)
        return np.where(den > 0, cov / np.where(den > 0, den, 1.0), np.nan)


# --------------------------------------------------------------------------------------
# EX-12: Spearman of per-extractant gains, with split-half reliability
# --------------------------------------------------------------------------------------
def split_half_gain_reliability(F: Frame, cand: str, ref: str) -> dict:
    """Spearman-Brown split-half reliability of the per-extractant MET-01 gain: the cells of each extractant are
    sorted by cell index and dealt alternately into two halves (extractants with < 2 cells are left out); the
    per-extractant gains of the halves are correlated (Pearson) across extractants, r_SB = 2r/(1 + r)."""
    P_c, P_r = F.pred(cand), F.pred(ref)
    cells = F.rows["ci"].to_numpy()
    half = np.zeros(F.n, dtype=int)
    for e in range(F.L):
        idx = np.flatnonzero(F.ext == e)
        uc = np.unique(cells[idx])
        rank = {c: i % 2 for i, c in enumerate(uc)}
        half[idx] = [rank[c] for c in cells[idx]]
    ok_ext = np.array([len(np.unique(cells[F.ext == e])) >= 2 for e in range(F.L)])
    g = []
    for h in (0, 1):
        vals = []
        for k in range(len(F.seeds)):
            m = half == h
            dc = np.abs(F.y - P_c[k])
            dr = np.abs(F.y - P_r[k])
            num_r = np.bincount(F.ext[m], weights=dr[m], minlength=F.L)
            num_c = np.bincount(F.ext[m], weights=dc[m], minlength=F.L)
            cnt = np.bincount(F.ext[m], minlength=F.L).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                vals.append((num_r - num_c) / cnt)
        g.append(np.mean(vals, axis=0))
    a, b = g[0][ok_ext], g[1][ok_ext]
    r = M._pearson(a, b)
    sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")
    return {"n_extractants_split": int(ok_ext.sum()), "split_half_r": r, "spearman_brown": sb}


def spearman_gains(F_d: Frame, a_cand: str, a_ref: str, F_sf: Frame, b_cand: str, b_ref: str,
                   *, min_reliability: float = 0.3) -> dict:
    """EX-12: Spearman correlation across common extractants of A's per-extractant MET-01 gain (over ``a_ref``) and
    B's (over ``b_ref``); read only when both split-half reliabilities are >= 0.3 (else p = 1 enters BH)."""
    from scipy.stats import spearmanr
    ga = per_extractant_deltas(F_d, a_cand, a_ref).set_index("extractant")["delta"]
    gb = per_extractant_deltas(F_sf, b_cand, b_ref).set_index("extractant")["delta"]
    common = ga.index.intersection(gb.index)
    ra = split_half_gain_reliability(F_d, a_cand, a_ref)
    rb = split_half_gain_reliability(F_sf, b_cand, b_ref)
    res = spearmanr(ga[common].to_numpy(), gb[common].to_numpy()) if len(common) >= 3 else None
    rho = float(res.statistic) if res is not None else float("nan")
    p = float(res.pvalue) if res is not None else float("nan")
    readable = bool(np.isfinite(ra["spearman_brown"]) and np.isfinite(rb["spearman_brown"])
                    and ra["spearman_brown"] >= min_reliability and rb["spearman_brown"] >= min_reliability)
    return {"rho": rho, "p_spearman": p, "n_common": int(len(common)), "reliability_A": ra["spearman_brown"],
            "reliability_B": rb["spearman_brown"], "readable": readable,
            "p_for_bh": p if (readable and np.isfinite(p)) else 1.0}
