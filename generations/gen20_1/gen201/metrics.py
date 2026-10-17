"""gen20.1 metrics MET-01..MET-24: point estimates per split seed and 5-seed means on a long table
(IMPLEMENTATION_PLAN.md §5; PRE_REGISTRATION.md §5).  Everything is recomputed from saved long tables
(CONTRACT schemas: logd rows keyed (ci, im), logsf rows keyed (ci, ia, ib), one float64 column per arm).

Conventions (sealed):

* ``n_pred_finite == n_rows`` is asserted before any metric (``check_coverage``); a non-finite prediction or
  target is a run failure (``CoverageError``), never a silently dropped row (gen20k/metrics.py:19-21 drops them).
* Scored rows are seed-invariant (each cell is tested once per seed): ``check_seed_invariant`` asserts that every
  split seed carries the same keys and the same y.  Hence n, y and every R² denominator are seed-invariant, and
  the seed mean of an R² is I4: ``1 - mean_s(SSE_s) / SST`` (equal to the mean of the per-seed R²).
* NA: an R² is NA with a reason when n < 2 ("n < 2") or its denominator is <= 0 ("SST <= 0"); every R² carries
  ``<key>_reason`` and ``<key>_n_seeds_defined``; a seed mean is NA if any seed is NA.  R² is never clipped,
  floored, averaged across folds/extractants/types or replaced by corr².  Skill_MSE is NA when the baseline SSE
  is 0.  Seed means of non-R² statistics are the arithmetic mean of the per-seed values (NA if any seed is NA).
* e = y - ŷ.  MET-04 Bias_bal = Σ w (ŷ - y) (recovery plan sign).  Direction accuracy (MET-14) scores a zero
  prediction 0.5 (t = 0: |y| > 0; t > 0: |y| >= t; gen20k semantics).
* Within-cell quantities centre y and ŷ over the cell's *scored* rows after prediction (ỹ, ŷ̃).

This module computes target statistics (MET-13) only when called; the sealed stages S5/S10 call it on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import bootstrap  # noqa: F401

LOG10_1_5 = float(np.log10(1.5))          # 0.17609...
LOG10_2 = float(np.log10(2.0))            # 0.30103...
DIRECTION_THRESHOLDS: tuple[float, ...] = (0.0, 0.1, 0.3)
LOGD_KEYS: tuple[str, ...] = ("ci", "im")
LOGSF_KEYS: tuple[str, ...] = ("ci", "ia", "ib")
ID_COLUMNS: tuple[str, ...] = ("design", "variant", "split_seed", "fold", "ci", "cell_id", "extractant", "chemotype",
                               "publication_id", "n_metals", "metal", "im", "a", "b", "ia", "ib", "dZ", "pair",
                               "strict", "gen13adj", "y")
NA_FEW = "n < 2"
NA_SST = "SST <= 0"
NA_SKILL = "baseline SSE = 0"
#: R² keys whose seed mean follows I4 (seed-invariant denominator)
R2_KEYS: tuple[str, ...] = ("R2_pooled", "R2_balanced", "R2_balanced_within", "R2_within_cell", "R2_level", "R2_0",
                            "R2_within_system")


class CoverageError(AssertionError):
    """n_pred_finite != n_rows, a non-finite target, a missing or duplicated scored key: a run failure."""


# --------------------------------------------------------------------------------------
# table checks
# --------------------------------------------------------------------------------------
def target_of(table: pd.DataFrame) -> str:
    return "logsf" if "ia" in table.columns else "logd"


def keys_of(target: str) -> tuple[str, ...]:
    return LOGSF_KEYS if target == "logsf" else LOGD_KEYS


def arm_columns(table: pd.DataFrame) -> list[str]:
    """Prediction columns of a long table (everything that is not an identity column)."""
    return [c for c in table.columns if c not in ID_COLUMNS and c != "is_dmd"]


def has_arm(table: pd.DataFrame, arm: str) -> bool:
    """An arm is 'run' when its column exists and is not entirely NaN (an all-NaN column = 'not run')."""
    return arm in table.columns and bool(np.isfinite(table[arm].to_numpy(dtype=float)).any())


def check_coverage(table: pd.DataFrame, arms) -> dict:
    """Assert finite y and n_pred_finite == n_rows for every arm; returns {arm: (n_rows, n_pred_finite)}."""
    y = table["y"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise CoverageError(f"{int((~np.isfinite(y)).sum())} non-finite targets among {len(y)} scored rows")
    out = {}
    for a in arms:
        if a not in table.columns:
            raise CoverageError(f"arm column {a!r} missing")
        p = table[a].to_numpy(dtype=float)
        nf = int(np.isfinite(p).sum())
        if nf != len(p):
            raise CoverageError(f"arm {a}: n_pred_finite {nf} != n_rows {len(p)} (a non-finite prediction is a run failure)")
        out[a] = (int(len(p)), nf)
    return out


def check_seed_invariant(table: pd.DataFrame) -> list:
    """Every split seed carries the same scored keys (no duplicates) and the same y; returns the sorted seeds."""
    keys = list(keys_of(target_of(table)))
    seeds = sorted(table["split_seed"].unique().tolist())
    ref = None
    for s in seeds:
        t = table[table["split_seed"] == s].sort_values(keys)
        k = t[keys].to_numpy(dtype=np.int64)
        if len(k) and (np.diff(k, axis=0) == 0).all(axis=1).any():
            raise CoverageError(f"duplicated scored key in seed {s}")
        cur = (k, t["y"].to_numpy(dtype=float))
        if ref is None:
            ref = cur
        elif ref[0].shape != cur[0].shape or not np.array_equal(ref[0], cur[0]) or not np.array_equal(ref[1], cur[1]):
            raise CoverageError(f"seed {s}: scored keys or y differ from seed {seeds[0]} (rows must be seed-invariant)")
    return seeds


def _sorted_seed(table: pd.DataFrame, s) -> pd.DataFrame:
    return table[table["split_seed"] == s].sort_values(list(keys_of(target_of(table)))).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# row structure
# --------------------------------------------------------------------------------------
def _codes(x) -> tuple[np.ndarray, np.ndarray]:
    names, code = np.unique(np.asarray(x).astype(str), return_inverse=True)
    return names, code.astype(np.int64)


def _group_mean(v: np.ndarray, code: np.ndarray) -> np.ndarray:
    """Per-row mean of ``v`` over its group."""
    s = np.bincount(code, weights=v)
    n = np.bincount(code).astype(float)
    return (s / n)[code]


def _pair_code(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense code of the pair (a, b) of two integer code arrays."""
    if not len(a):
        return np.zeros(0, dtype=np.int64)
    _, code = np.unique(a.astype(np.int64) * (int(b.max()) + 1) + b.astype(np.int64), return_inverse=True)
    return code.astype(np.int64)


def _n_distinct_within(group: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Per row: the number of distinct ``member`` codes in the row's ``group``."""
    pairs = np.unique(np.c_[group, member], axis=0)
    return np.bincount(pairs[:, 0], minlength=int(group.max()) + 1)[group]


def balanced_weights(extractant, publication, cell) -> np.ndarray:
    """MET-04 weights w_i = 1/(L·P_l·C_lp·N_lpc) recomputed on the given rows (recovery plan :123-141): L
    extractants, P_l publications of extractant l among the rows, C_lp cells of l in publication p, N_lpc rows
    of cell c.  Σ w = 1 on any non-empty row set."""
    _, le = _codes(extractant)
    if not len(le):
        return np.zeros(0)
    _, pc = _codes(publication)
    _, lc = _codes(cell)
    lp = _pair_code(le, pc)
    L = len(np.unique(le))
    P_l = _n_distinct_within(le, lp)
    C_lp = _n_distinct_within(lp, lc)
    N_c = np.bincount(lc)[lc]
    return 1.0 / (L * P_l.astype(float) * C_lp.astype(float) * N_c.astype(float))


class SeedRows:
    """One seed's scored rows (sorted by key) with the groupings every metric needs."""

    def __init__(self, t: pd.DataFrame):
        self.t = t
        self.target = target_of(t)
        self.y = t["y"].to_numpy(dtype=float)
        self.n = len(self.y)
        _, self.cell = _codes(t["ci"].to_numpy())
        self.ext_names, self.ext = _codes(t["extractant"])
        self.chem_names, self.chem = _codes(t["chemotype"])
        _, self.pub = _codes(t["publication_id"])
        self.n_c = np.bincount(self.cell)[self.cell].astype(float)
        self.ybar_c = _group_mean(self.y, self.cell)
        self.yt = self.y - self.ybar_c                                # ỹ
        self.w_bal = balanced_weights(t["extractant"], t["publication_id"], t["ci"])
        self.n_l = np.bincount(self.ext)[self.ext].astype(float)
        # chemotype of each extractant and L_g (extractants per chemotype)
        if self.n:
            pairs = np.unique(np.c_[self.ext, self.chem], axis=0)
            if len(pairs) != len(np.unique(self.ext)):
                raise CoverageError("an extractant belongs to more than one chemotype")
            self.ext_chem_code = pairs[np.argsort(pairs[:, 0]), 1]           # chemotype code of extractant e
            self.L_g = np.bincount(pairs[:, 1], minlength=int(self.chem.max()) + 1).astype(float)[self.chem]
        else:
            self.ext_chem_code = np.zeros(0, dtype=np.int64)
            self.L_g = np.zeros(0)
        self._type_code = None

    def pred(self, arm: str) -> np.ndarray:
        return self.t[arm].to_numpy(dtype=float)

    def system_groups(self) -> tuple[np.ndarray, np.ndarray]:
        """(group code of (publication, extractant, type), distinct cells per group) for R2_within_system."""
        if self._type_code is None:
            typ = self.t["pair"] if self.target == "logsf" else self.t["metal"]
            _, tc = _codes(typ.to_numpy())
            g = _pair_code(_pair_code(self.pub, self.ext), tc)
            ncell = np.bincount(np.unique(np.c_[g, self.cell], axis=0)[:, 0], minlength=int(g.max()) + 1)                 if len(g) else np.zeros(0, dtype=int)
            self._type_code = (g, ncell)
        return self._type_code


def _r2(num: float, den: float, n: int) -> tuple[float, str]:
    if n < 2:
        return float("nan"), NA_FEW
    if not den > 0:
        return float("nan"), NA_SST
    return float(1.0 - num / den), ""


def _macro_ext(r: SeedRows, a: np.ndarray) -> float:
    """MET-01 style: (1/L) Σ_l mean_{i∈l} a_i."""
    per = np.bincount(r.ext, weights=a) / np.bincount(r.ext)
    return float(per.mean())


def _macro_chem(r: SeedRows, a: np.ndarray) -> float:
    """MET-03: (1/G) Σ_g (1/L_g) Σ_{l∈g} MAE_l (gen20k/metrics.py:72-73)."""
    per_l = np.bincount(r.ext, weights=a) / np.bincount(r.ext)
    G = int(r.ext_chem_code.max()) + 1
    per_g = np.bincount(r.ext_chem_code, weights=per_l, minlength=G) / np.bincount(r.ext_chem_code, minlength=G)
    return float(per_g[np.bincount(r.ext_chem_code, minlength=G) > 0].mean())


def direction_accuracy(y, p, threshold: float = 0.0) -> tuple[float, int, int]:
    """MET-14: sign agreement on rows with |y| > 0 (t = 0) or |y| >= t; ŷ = 0 exactly scores 0.5.
    Returns (accuracy, n_t, n_zero_pred among those rows); accuracy NA when n_t = 0."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    sel = (np.abs(y) > 0) if threshold == 0 else (np.abs(y) >= threshold)
    n_t = int(sel.sum())
    if not n_t:
        return float("nan"), 0, 0
    sp = np.sign(p[sel])
    score = np.where(sp == 0, 0.5, (sp == np.sign(y[sel])).astype(float))
    return float(score.mean()), n_t, int((p[sel] == 0).sum())


# --------------------------------------------------------------------------------------
# per-seed metrics of one arm
# --------------------------------------------------------------------------------------
def seed_metrics(r: SeedRows, arm: str) -> dict:
    """Every point metric of ``arm`` on one seed's rows (MET-01..10, 12, 14..16, 20)."""
    y, n = r.y, r.n
    p = r.pred(arm)
    e = y - p
    ae, e2 = np.abs(e), e * e
    w = r.w_bal
    out: dict = {"n_rows": n, "n_pred_finite": int(np.isfinite(p).sum())}
    out["MET01_MAE_macro_ext"] = _macro_ext(r, ae)
    out["MET02_MAE_pooled"] = float(ae.mean())
    out["MET03_MAE_macro_chem"] = _macro_chem(r, ae)
    out["MET04_MAE_bal"] = float((w * ae).sum())
    out["MET04_RMSE_bal"] = float(np.sqrt((w * e2).sum()))
    out["MET04_Bias_bal"] = float((w * (p - y)).sum())
    out["MET05_RMSE_pooled"] = float(np.sqrt(e2.mean()))
    sse, sst = float(e2.sum()), float(((y - y.mean()) ** 2).sum())
    _put_r2(out, "R2_pooled", sse, sst, n)
    # MET-07 balanced (MET-04 weights) and its within-cell variant
    yw = float((w * y).sum())
    _put_r2(out, "R2_balanced", float((w * e2).sum()), float((w * (y - yw) ** 2).sum()), n)
    pt = p - _group_mean(p, r.cell)                                   # ŷ̃
    dw = r.yt - pt
    _put_r2(out, "R2_balanced_within", float((w * dw ** 2).sum()), float((w * r.yt ** 2).sum()), n)
    # MET-08 within-cell; MET-09 level with s_b / s_w
    _put_r2(out, "R2_within_cell", float((dw ** 2).sum()), float((r.yt ** 2).sum()), n)
    pbar_c = _group_mean(p, r.cell)
    ssb_err = float(((r.ybar_c - pbar_c) ** 2).sum())                # Σ_c n_c (ȳ_c − ŷ̄_c)²
    ssb = float(((r.ybar_c - y.mean()) ** 2).sum())                  # Σ_c n_c (ȳ_c − ȳ)²
    _put_r2(out, "R2_level", ssb_err, ssb, n)
    out["s_b"] = ssb / sst if sst > 0 else float("nan")
    out["s_w"] = 1.0 - out["s_b"] if sst > 0 else float("nan")
    # MET-10 R2_0 (skill vs FLAT; reported for log SF)
    _put_r2(out, "R2_0", sse, float((y * y).sum()), n)
    # MET-12
    var0 = sst / n if n else float("nan")
    out["MET12_EV_abs"] = var0 - float(e2.mean())
    out["MSE"] = float(e2.mean())
    out["SSE"] = sse
    # MET-14 (log SF only; computed for both, printed for log SF)
    for t in DIRECTION_THRESHOLDS:
        acc, n_t, nz = direction_accuracy(y, p, t)
        out[f"MET14_dir_acc_{t:g}"] = acc
        out[f"MET14_n_{t:g}"] = n_t
        out[f"MET14_n_zero_pred_{t:g}"] = nz
    # MET-15 shares within 1.5x / 2x (pooled and MET-04 weighted)
    out["MET15_within_1p5x"] = float((ae <= LOG10_1_5).mean())
    out["MET15_within_2x"] = float((ae <= LOG10_2).mean())
    out["MET15_within_1p5x_bal"] = float((w * (ae <= LOG10_1_5)).sum())
    out["MET15_within_2x_bal"] = float((w * (ae <= LOG10_2)).sum())
    # MET-16 diagnostics
    out["MET16_bias"] = float((p - y).mean())
    sd_y, sd_p = float(y.std()), float(p.std())
    out["MET16_sd_ratio"] = sd_p / sd_y if sd_y > 0 else float("nan")
    syt = float((r.yt ** 2).sum())
    out["MET16_within_dispersion"] = float(np.sqrt((pt ** 2).sum() / syt)) if syt > 0 else float("nan")
    vp = float(((p - p.mean()) ** 2).sum())
    out["MET16_ols_slope"] = float(((p - p.mean()) * (y - y.mean())).sum() / vp) if vp > 0 else float("nan")
    # MET-20 within-cell extractant-macro MAE
    out["MET20_MAE_macro_ext_within"] = _macro_ext(r, np.abs(dw))
    # MET-21 R2_within_system (point; recovery plan :154-156)
    v, why, num, den = r2_within_system(r, p)
    out["R2_within_system"], out["R2_within_system_reason"] = v, why
    out["R2_within_system_num"], out["R2_within_system_den"] = num, den
    return out


def _put_r2(out: dict, key: str, num: float, den: float, n: int) -> None:
    v, why = _r2(num, den, n)
    out[key], out[f"{key}_reason"], out[f"{key}_num"], out[f"{key}_den"] = v, why, num, den


def r2_within_system(r: SeedRows, p: np.ndarray, *, min_conditions: int = 3) -> tuple[float, str, float, float]:
    """MET-21 R2_within_system: groups (publication, extractant, metal or pair type) with >= 3 distinct cells
    (condition sets); y and ŷ centred within each group; equal weight per group and per cell within a group;
    groups with constant y are excluded (counted in the reason when nothing is left)."""
    g, ncell = r.system_groups()
    size = np.bincount(g, minlength=len(ncell))
    yc = r.y - _group_mean(r.y, g)
    pc = p - _group_mean(p, g)
    sy = np.bincount(g, weights=yc * yc, minlength=len(ncell))
    keep = (ncell >= min_conditions) & (sy > 0)
    if not keep.any():
        return float("nan"), f"no group with >= {min_conditions} condition sets and non-constant y", float("nan"), float("nan")
    wg = np.where(keep, 1.0 / (keep.sum() * np.maximum(size, 1)), 0.0)[g]
    num, den = float((wg * (pc - yc) ** 2).sum()), float((wg * yc * yc).sum())
    v, why = _r2(num, den, int(keep[g].sum()))
    return v, why, num, den


# --------------------------------------------------------------------------------------
# seed means
# --------------------------------------------------------------------------------------
def summarise(table: pd.DataFrame, arm: str, *, check: bool = True) -> dict:
    """Per-seed metrics and seed means of ``arm`` on a population table (all seeds).  R² keys follow I4 with NA
    propagation and carry ``_reason`` and ``_n_seeds_defined``; ``<key>_per_seed`` lists the per-seed values."""
    if check:
        check_coverage(table, [arm])
    seeds = check_seed_invariant(table)
    per = [seed_metrics(SeedRows(_sorted_seed(table, s)), arm) for s in seeds]
    return combine_seeds(per, seeds)


def combine_seeds(per: list[dict], seeds: list) -> dict:
    out: dict = {"n_seeds": len(seeds), "seeds": list(seeds)}
    skip = {f"{k}{suf}" for k in R2_KEYS for suf in ("_reason", "_num", "_den")}
    for k in per[0]:
        if k in skip:
            continue
        vals = [d[k] for d in per]
        out[f"{k}_per_seed"] = vals
        if k in R2_KEYS:
            nums = np.array([d[f"{k}_num"] for d in per], dtype=float)
            dens = np.array([d[f"{k}_den"] for d in per], dtype=float)
            defined = [np.isfinite(v) for v in vals]
            out[f"{k}_n_seeds_defined"] = int(sum(defined))
            reasons = [d[f"{k}_reason"] for d in per if d[f"{k}_reason"]]
            out[f"{k}_reason"] = reasons[0] if reasons else ""
            if not all(defined):
                out[k] = float("nan")
            else:
                if not np.allclose(dens, dens[0], rtol=1e-12, atol=0.0):
                    raise CoverageError(f"{k}: denominator differs across seeds (rows must be seed-invariant)")
                out[k] = float(1.0 - nums.mean() / dens[0])           # I4
        elif isinstance(vals[0], (int, np.integer)) and not isinstance(vals[0], bool):
            out[k] = vals[0] if len(set(vals)) == 1 else float(np.mean(vals))
        else:
            arr = np.asarray(vals, dtype=float)
            out[k] = float(arr.mean()) if np.isfinite(arr).all() else float("nan")
    return out


def skill_mse(table: pd.DataFrame, arm: str, ref: str) -> dict:
    """MET-11 Skill_MSE(X|Y) = 1 − SSE_s(X)/SSE_s(Y) per seed on identical rows; seed mean of the per-seed values
    (NA if any seed is NA, e.g. a zero baseline SSE)."""
    check_coverage(table, [arm, ref])
    seeds = check_seed_invariant(table)
    vals, reasons = [], []
    for s in seeds:
        t = table[table["split_seed"] == s]
        y = t["y"].to_numpy(dtype=float)
        sx = float(((y - t[arm].to_numpy(dtype=float)) ** 2).sum())
        sy = float(((y - t[ref].to_numpy(dtype=float)) ** 2).sum())
        if sy > 0:
            vals.append(1.0 - sx / sy)
        else:
            vals.append(float("nan"))
            reasons.append(NA_SKILL)
    arr = np.asarray(vals, dtype=float)
    return {"Skill_MSE": float(arr.mean()) if np.isfinite(arr).all() else float("nan"),
            "Skill_MSE_per_seed": vals, "Skill_MSE_reason": reasons[0] if reasons else "",
            "Skill_MSE_n_seeds_defined": int(np.isfinite(arr).sum())}


def met12(sx: dict, sy: dict) -> dict:
    """MET-12 between candidate summary ``sx`` and reference summary ``sy`` (seed means): ΔMSE, ΔRMSE (reference
    minus candidate) and the relative MET-01 gain Δ/MET-01(Y)."""
    d01 = sy["MET01_MAE_macro_ext"] - sx["MET01_MAE_macro_ext"]
    return {"EV_abs_X": sx["MET12_EV_abs"], "EV_abs_Y": sy["MET12_EV_abs"],
            "dMSE": sy["MSE"] - sx["MSE"], "dRMSE": sy["MET05_RMSE_pooled"] - sx["MET05_RMSE_pooled"],
            "rel_MAE_gain": d01 / sy["MET01_MAE_macro_ext"] if sy["MET01_MAE_macro_ext"] else float("nan")}


# --------------------------------------------------------------------------------------
# per-extractant MAE (inference unit of MET-01 / MET-20)
# --------------------------------------------------------------------------------------
def per_extractant(table: pd.DataFrame, arm: str, *, kind: str = "abs") -> pd.DataFrame:
    """Seed-mean MAE_l per extractant (kind 'abs' = MET-01's |e|, 'within' = MET-20's |ỹ − ŷ̃|), with its
    chemotype and the per-seed values (columns s<seed>)."""
    seeds = check_seed_invariant(table)
    cols = {}
    chem = None
    for s in seeds:
        r = SeedRows(_sorted_seed(table, s))
        p = r.pred(arm)
        if kind == "abs":
            a = np.abs(r.y - p)
        elif kind == "within":
            a = np.abs(r.yt - (p - _group_mean(p, r.cell)))
        else:
            raise ValueError(kind)
        cols[f"s{s}"] = np.bincount(r.ext, weights=a) / np.bincount(r.ext)
        if chem is None:
            df = pd.DataFrame({"e": r.ext, "g": r.chem_names[r.chem]}).drop_duplicates().sort_values("e")
            chem = df["g"].to_numpy()
            names = r.ext_names
    out = pd.DataFrame(cols, index=pd.Index(names, name="extractant"))
    out["MAE_l"] = out[[f"s{s}" for s in seeds]].mean(axis=1)
    out["chemotype"] = chem
    return out


def per_extractant_delta(table: pd.DataFrame, cand: str, ref: str, *, kind: str = "abs") -> pd.DataFrame:
    """Δ_l = MAE_l(ref) − MAE_l(cand) (seed means; positive favours the candidate) with chemotype."""
    a = per_extractant(table, cand, kind=kind)
    b = per_extractant(table, ref, kind=kind)
    return pd.DataFrame({"delta": b["MAE_l"] - a["MAE_l"], "chemotype": a["chemotype"],
                         "MAE_l_cand": a["MAE_l"], "MAE_l_ref": b["MAE_l"]})


# --------------------------------------------------------------------------------------
# MET-13 card (target statistics; S5 and S10 only)
# --------------------------------------------------------------------------------------
def kish(groups) -> float:
    g = pd.Series(np.ones(len(groups))).groupby(np.asarray(groups).astype(str)).sum()
    return float(g.sum() ** 2 / (g ** 2).sum()) if len(g) else float("nan")


def card(table: pd.DataFrame) -> dict:
    """MET-13 on one seed's rows (rows are seed-invariant): n_rows, cells, extractants, chemotypes, publications,
    Kish over chemotypes (rows) and over chemotypes counting extractants, largest chemotype share, y_mean, y_sd0,
    y_var0, y_rms0 (log SF only), s_b and s_w (log D)."""
    seeds = check_seed_invariant(table)
    t = _sorted_seed(table, seeds[0])
    target = target_of(t)
    y = t["y"].to_numpy(dtype=float)
    vc = t["chemotype"].astype(str).value_counts()
    ext = t[["extractant", "chemotype"]].astype(str).drop_duplicates()
    n = len(y)
    var0 = float(((y - y.mean()) ** 2).mean()) if n else float("nan")
    out = {"target": target, "n_rows": int(n), "n_cells": int(t["ci"].nunique()),
           "n_extractants": int(t["extractant"].nunique()), "n_chemotypes": int(t["chemotype"].nunique()),
           "n_publications": int(t["publication_id"].nunique()),
           "kish_rows": kish(t["chemotype"].to_numpy()), "kish_ext": kish(ext["chemotype"].to_numpy()),
           "largest_chemotype": str(vc.index[0]) if len(vc) else None,
           "largest_chemotype_share": float(vc.iloc[0] / vc.sum()) if len(vc) else float("nan"),
           "y_mean": float(y.mean()) if n else float("nan"), "y_sd0": float(np.sqrt(var0)), "y_var0": var0}
    if target == "logsf":
        out["y_rms0"] = float(np.sqrt((y * y).mean())) if n else float("nan")
    else:
        r = SeedRows(t)
        sst = float(((y - y.mean()) ** 2).sum())
        ssb = float(((r.ybar_c - y.mean()) ** 2).sum())
        out["s_b"] = ssb / sst if sst > 0 else float("nan")
        out["s_w"] = 1.0 - out["s_b"] if sst > 0 else float("nan")
    return out


# --------------------------------------------------------------------------------------
# MET-17 (error cancellation from A to B), MET-18 (per type), MET-21 (pair macro), MET-24 (Pr/Nd)
# --------------------------------------------------------------------------------------
def endpoint_residuals(logd: pd.DataFrame, logsf: pd.DataFrame, arm: str = "A") -> pd.DataFrame:
    """For every pair row of every seed: A's residuals r_a, r_b at the endpoints and their cell-centred values
    (centred over the cell's rows in ``logd``).  Every endpoint must be a scored log D row."""
    rows = []
    for s in check_seed_invariant(logsf):
        d = _sorted_seed(logd, s)
        r = d["y"].to_numpy(dtype=float) - d[arm].to_numpy(dtype=float)
        _, cc = _codes(d["ci"].to_numpy())
        rt = r - _group_mean(r, cc)
        lk = pd.DataFrame({"ci": d["ci"].to_numpy(), "im": d["im"].to_numpy(), "r": r, "rt": rt})
        q = _sorted_seed(logsf, s)[["split_seed", "ci", "ia", "ib", "chemotype", "publication_id", "extractant"]]
        q = q.merge(lk.rename(columns={"im": "ia", "r": "r_a", "rt": "rt_a"}), on=["ci", "ia"], how="left")
        q = q.merge(lk.rename(columns={"im": "ib", "r": "r_b", "rt": "rt_b"}), on=["ci", "ib"], how="left")
        if q[["r_a", "r_b"]].isna().any().any():
            raise CoverageError("a pair endpoint is not a scored log D row")
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


def met17_seed(logd_seed: pd.DataFrame, ends_seed: pd.DataFrame, arm: str = "A") -> dict:
    r = logd_seed["y"].to_numpy(dtype=float) - logd_seed[arm].to_numpy(dtype=float)
    _, cc = _codes(logd_seed["ci"].to_numpy())
    rbar = _group_mean(r, cc)
    ra, rb = ends_seed["r_a"].to_numpy(dtype=float), ends_seed["r_b"].to_numpy(dtype=float)
    ta, tb = ends_seed["rt_a"].to_numpy(dtype=float), ends_seed["rt_b"].to_numpy(dtype=float)
    s2 = float((r * r).sum())
    den = float((ra * ra + rb * rb).sum())
    return {"lambda": float((rbar * rbar).sum() / s2) if s2 > 0 else float("nan"),
            "kappa_adj": float(((ra - rb) ** 2).sum() / den) if den > 0 else float("nan"),
            "rho_adj": _pearson(ra, rb), "rho_within_adj": _pearson(ta, tb)}


def _pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return float((da * db).sum() / den) if den > 0 else float("nan")


def met17(logd: pd.DataFrame, logsf: pd.DataFrame, arm: str = "A") -> dict:
    """MET-17 on A's residuals: λ = Σ_c n_c r̄_c² / Σ r² (log D rows), κ_adj = Σ (r_a − r_b)² / Σ (r_a² + r_b²),
    ρ_adj = Pearson(r_a, r_b), ρ_within_adj on cell-centred residuals; per seed and seed mean."""
    ends = endpoint_residuals(logd, logsf, arm)
    seeds = check_seed_invariant(logsf)
    per = [met17_seed(_sorted_seed(logd, s), ends[ends["split_seed"] == s], arm) for s in seeds]
    out = {}
    for k in per[0]:
        arr = np.array([d[k] for d in per], dtype=float)
        out[k] = float(arr.mean()) if np.isfinite(arr).all() else float("nan")
        out[f"{k}_per_seed"] = arr.tolist()
    return out


def per_type(table: pd.DataFrame, arm: str, simple: str | None = None) -> pd.DataFrame:
    """MET-18: per strict pair type (log SF) or per metal (log D): n, y_sd0, the simple baseline's MAE (FLAT for
    log SF, ``simple`` = BD_METAL for log D, with its RMSE and R²), the arm's MAE, RMSE and within-type R² (I4,
    R2_reason, n_seeds_defined); fewer than 10 extractants -> preliminary."""
    target = target_of(table)
    col = "pair" if target == "logsf" else "metal"
    order = "ia" if target == "logsf" else "im"
    rows = []
    for typ, blk in sorted(table.groupby(col), key=lambda kv: int(kv[1][order].iloc[0])):
        s = summarise(blk, arm, check=False)
        y0 = blk[blk["split_seed"] == blk["split_seed"].iloc[0]]["y"].to_numpy(dtype=float)
        row = {"type": typ, "n_rows": int(len(y0)), "n_extractants": int(blk["extractant"].nunique()),
               "preliminary": bool(blk["extractant"].nunique() < 10),
               "y_sd0": float(y0.std()) if len(y0) else float("nan"),
               "MAE": s["MET02_MAE_pooled"], "MAE_macro_ext": s["MET01_MAE_macro_ext"], "RMSE": s["MET05_RMSE_pooled"],
               "R2": s["R2_pooled"], "R2_reason": s["R2_pooled_reason"],
               "R2_n_seeds_defined": s["R2_pooled_n_seeds_defined"]}
        if target == "logsf":
            row["MAE_FLAT"] = float(np.abs(y0).mean()) if len(y0) else float("nan")
        if simple is not None and simple in blk.columns:
            b = summarise(blk, simple, check=False)
            row.update({f"MAE_{simple}": b["MET02_MAE_pooled"], f"RMSE_{simple}": b["MET05_RMSE_pooled"],
                        f"R2_{simple}": b["R2_pooled"], f"R2_{simple}_reason": b["R2_pooled_reason"]})
        rows.append(row)
    return pd.DataFrame(rows)


def r2_pair_macro(table: pd.DataFrame, arm: str) -> dict:
    """MET-21 R2_pair_macro (recovery plan :151): per type (pair for log SF, metal for log D) the R2_balanced with
    weights recomputed within the type, then the equal mean of the finite values (also the median and each)."""
    target = target_of(table)
    col = "pair" if target == "logsf" else "metal"
    per_type_vals = {}
    for typ, blk in table.groupby(col):
        per_type_vals[str(typ)] = summarise(blk, arm, check=False)["R2_balanced"]
    fin = np.array([v for v in per_type_vals.values() if np.isfinite(v)], dtype=float)
    return {"R2_pair_macro": float(fin.mean()) if fin.size else float("nan"),
            "R2_pair_macro_median": float(np.median(fin)) if fin.size else float("nan"),
            "R2_pair_macro_n_types": int(fin.size), "R2_pair_macro_per_type": per_type_vals,
            "R2_pair_macro_reason": "" if fin.size else "no type with a defined R2_balanced"}


def met24(logsf: pd.DataFrame, arm: str, *, pair: str = "Pr/Nd") -> dict:
    """MET-24 Pr/Nd card on the Pr/Nd rows of a log SF population: MAE (pooled and extractant-macro), R², y_sd0,
    systems (cells), frac_within_2x."""
    blk = logsf[logsf["pair"].astype(str) == pair]
    if not len(blk):
        return {"pair": pair, "n_rows": 0, "reason": "no rows"}
    s = summarise(blk, arm, check=False)
    y0 = blk[blk["split_seed"] == blk["split_seed"].iloc[0]]["y"].to_numpy(dtype=float)
    return {"pair": pair, "n_rows": int(len(y0)), "systems": int(blk["ci"].nunique()),
            "n_extractants": int(blk["extractant"].nunique()), "y_sd0": float(y0.std()),
            "MAE": s["MET02_MAE_pooled"], "MAE_macro_ext": s["MET01_MAE_macro_ext"], "R2": s["R2_pooled"],
            "R2_reason": s["R2_pooled_reason"], "R2_n_seeds_defined": s["R2_pooled_n_seeds_defined"],
            "frac_within_2x": s["MET15_within_2x"]}
