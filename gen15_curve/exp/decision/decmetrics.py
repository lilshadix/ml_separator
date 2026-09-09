"""Decision metrics: what a chemist actually asks a separation model, scored on the gen13 pair table.

The atom is the frozen pair table produced by ``gen15.valuebench.run_arms``: one row per
(split seed, held-out cell, metal A, metal B) with ``y = logD(A) - logD(B)`` (A the lighter metal,
base-10) and one prediction column per arm.

Four questions, four families of function:

Q1  ``sign_bands``      given a target pair, does the model call the direction right?
Q2  ``within_cell``     for one system, can the model rank its pairs and name its best one?
Q3  ``cross_extractant`` given a target pair, can the model pick the extractant that separates it?
Q4  ``calibration`` / ``coverage``  is the predicted size of the separation believable, and does
                        the model know when it does not know?

Two conventions are used throughout and are stated in the report:

* **units.**  Every aggregate is macro: mean within a (split seed, extractant), then over
  extractants, then over the five split seeds -- gen13's own unit, so one 375-cell chemotype
  cannot carry a number.  The reported spread is the sd over the five seeds.
* **ties.**  A constant prediction (FLAT everywhere, MEAN_CURVE across extractants) has no
  ranking.  Rather than dropping it, every rank statistic is the *exact expectation under uniform
  random tie-breaking*: a top-1 hit rate becomes (number of tied picks that are correct) / (number
  tied), a regret becomes the mean over the tied set, and a rank correlation whose predictor is
  constant is 0.  This makes an uninformative arm land exactly on chance instead of on NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

STRONG = 0.3                     # gen13's strong-pair threshold, ~0.9 pair-noise sd
MIN_METALS_RANK = 4              # a cell needs >= 4 metals (6 pairs) to be ranked
MIN_EXT_RANK = 5                 # a metal pair needs >= 5 held-out extractants to be ranked
DZ_BANDS = [(1, 1, "dZ=1"), (2, 4, "dZ 2-4"), (5, 8, "dZ 5-8"), (9, 13, "dZ 9-13")]
Y_BANDS = [(0.3, 0.5, "0.3-0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 2.0, "1.0-2.0"), (2.0, 99.0, ">2.0")]
COVERAGES = (1.0, 0.5, 0.25, 0.10)
#: the pairs the separation literature is actually about (canonicalised light-first)
INDUSTRIAL = [("Nd", "Dy"), ("Pr", "Nd"), ("Sm", "Eu"), ("Eu", "Gd"),
              ("La", "Ce"), ("Dy", "Ho"), ("Er", "Yb")]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _spearman(pred: np.ndarray, obs: np.ndarray) -> float:
    """Spearman rho; 0 (not NaN) when the predictor is constant -- see the tie convention."""
    if len(pred) < 3 or np.ptp(obs) == 0:
        return np.nan
    if np.ptp(pred) == 0:
        return 0.0
    a, b = rankdata(pred), rankdata(obs)
    return float(np.corrcoef(a, b)[0, 1])


def _kendall(pred: np.ndarray, obs: np.ndarray) -> float:
    if len(pred) < 3 or np.ptp(obs) == 0:
        return np.nan
    if np.ptp(pred) == 0:
        return 0.0
    return float(kendalltau(pred, obs, variant="b").statistic)


def _tie_expected_top1(score: np.ndarray, truth_best: np.ndarray) -> float:
    """P(the model's pick is a true best) under uniform tie-breaking among the argmax of score."""
    m = score.max()
    tied = score >= m - 1e-12
    return float(truth_best[tied].mean())


def _tie_expected_value(score: np.ndarray, value: np.ndarray) -> float:
    """Expected ``value`` of the model's pick under uniform tie-breaking among argmax of score."""
    m = score.max()
    tied = score >= m - 1e-12
    return float(value[tied].mean())


def _comb(n: int, k: int) -> float:
    from math import comb
    return float(comb(n, k)) if 0 <= k <= n else 0.0


def _tie_expected_topk(score: np.ndarray, best: np.ndarray, k: int = 3) -> float:
    """P(at least one true best is in the model's top k) under uniform tie-breaking.

    The model's score induces tie-groups; a random total order consistent with it fills the k
    slots from the best group down, drawing uniformly inside the group that straddles slot k.
    """
    n = len(score)
    if n <= k:
        return float(best.sum() > 0)
    order = np.argsort(-score, kind="stable")
    s = score[order]
    b = best[order].astype(int)
    p_none, filled, i = 1.0, 0, 0
    while filled < k and i < n:
        j = i
        while j < n and s[j] >= s[i] - 1e-12:
            j += 1
        size, nb = j - i, int(b[i:j].sum())
        take = min(size, k - filled)
        if take == size:
            if nb:
                return 1.0
        else:
            p_none *= _comb(size - nb, take) / _comb(size, take)
        filled += take
        i = j
    return float(1.0 - p_none)


def _macro(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Mean within (seed, extractant), then over extractants, then over seeds; (mean, sd_seed)."""
    if df.empty or df[col].notna().sum() == 0:
        return np.nan, np.nan
    by = df.groupby(["split_seed", "extractant"])[col].mean()
    per_seed = by.groupby(level=0).mean()
    return float(per_seed.mean()), float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan


def prepare(tab: pd.DataFrame, basis: np.ndarray, lanthanides) -> pd.DataFrame:
    """Attach the radius difference and the two rule-based reference columns."""
    r = {m: float(basis[0][i]) for i, m in enumerate(lanthanides)}
    r2 = {m: float(basis[1][i]) for i, m in enumerate(lanthanides)}
    tab = tab.copy()
    tab["dr"] = tab["A"].map(r).astype(float) - tab["B"].map(r).astype(float)
    tab["dr2"] = tab["A"].map(r2).astype(float) - tab["B"].map(r2).astype(float)
    # 'always prefer the heavier lanthanide': predicted logSF(A-B) < 0, ordered by radius gap
    tab["HEAVIER_ALWAYS"] = -tab["dr"]
    return tab


# --------------------------------------------------------------------------------------
# Q1  pair-direction accuracy
# --------------------------------------------------------------------------------------
def sign_units(tab: pd.DataFrame, arms: list[str], *, thr: float = STRONG) -> pd.DataFrame:
    """Per (split seed, extractant, band, arm) sign-accuracy, the unit every aggregate is built on.

    A prediction of exactly zero is scored as a coin toss (0.5), not as a miss: FLAT declines to
    call a direction, and scoring that as 0 % would flatter every arm that does call one.
    """
    s = tab[tab["y"].abs() >= thr]
    bands = [("all", np.ones(len(s), dtype=bool))]
    bands += [(lab, ((s["dZ"] >= lo) & (s["dZ"] <= hi)).to_numpy()) for lo, hi, lab in DZ_BANDS]
    bands += [(lab, ((s["y"].abs() >= lo) & (s["y"].abs() < hi)).to_numpy())
              for lo, hi, lab in Y_BANDS]
    out = []
    for lab, mask in bands:
        blk = s[mask]
        if blk.empty:
            continue
        for arm in arms:
            hit = (np.sign(blk["y"]) == np.sign(blk[arm])).astype(float).where(blk[arm] != 0.0, 0.5)
            g = (blk.assign(_h=hit).groupby(["split_seed", "extractant", "chemotype"], sort=False)
                 .agg(hit=("_h", "mean"), n=("_h", "size")).reset_index())
            g["band"], g["arm"] = lab, arm
            out.append(g)
    return pd.concat(out, ignore_index=True)


def sign_from_units(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lab, arm), blk in units.groupby(["band", "arm"], sort=False):
        per_seed = blk.groupby("split_seed")["hit"].mean()
        rows.append({"band": lab, "arm": arm, "sign_acc": float(per_seed.mean()),
                     "sd_seed": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan,
                     "n_pairs": int(blk["n"].sum()),
                     "n_extractants": int(blk["extractant"].nunique())})
    return pd.DataFrame(rows)


def sign_bands(tab: pd.DataFrame, arms: list[str], *, thr: float = STRONG) -> pd.DataFrame:
    """Sign accuracy on pairs with |observed log SF| >= thr, overall and by band."""
    return sign_from_units(sign_units(tab, arms, thr=thr))


# --------------------------------------------------------------------------------------
# Q2  within-cell ranking
# --------------------------------------------------------------------------------------
def within_cell(tab: pd.DataFrame, arms: list[str], *,
                min_metals: int = MIN_METALS_RANK) -> pd.DataFrame:
    """Per held-out cell: rank correlation over its pairs, and 'name the best-separated pair'."""
    sub = tab[tab["n_metals"] >= min_metals]
    recs = []
    for (seed, cid), cell in sub.groupby(["split_seed", "cell_id"], sort=False):
        obs = cell["y"].to_numpy()
        aobs = np.abs(obs)
        best = (aobs >= aobs.max() - 1e-12).astype(float)          # true best-separated pair(s)
        order = np.argsort(-aobs)
        top3_truth = np.zeros(len(obs)); top3_truth[order[:3]] = 1.0
        base = {"split_seed": seed, "cell_id": cid, "extractant": cell["extractant"].iat[0],
                "chemotype": cell["chemotype"].iat[0], "n_metals": int(cell["n_metals"].iat[0]),
                "n_pairs": len(obs)}
        for arm in arms:
            p = cell[arm].to_numpy()
            ap = np.abs(p)
            recs.append(dict(base, arm=arm,
                             spearman=_spearman(p, obs),
                             kendall=_kendall(p, obs),
                             top1=_tie_expected_top1(ap, best),
                             top3=_tie_expected_topk(ap, best, 3)))
        n, m = len(obs), int(best.sum())
        recs.append(dict(base, arm="_RANDOM",
                         spearman=0.0, kendall=0.0,
                         top1=float(m / n),
                         top3=float(1.0 - (_comb(n - m, 3) / _comb(n, 3) if n > 3 else 0.0))))
    cells = pd.DataFrame(recs)
    return within_from_cells(cells), cells


def within_from_cells(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, blk in cells.groupby("arm"):
        rec = {"arm": arm, "n_cells": int(blk["cell_id"].nunique())}
        for col in ("spearman", "kendall", "top1", "top3"):
            m, sd = _macro(blk, col)
            rec[col], rec[col + "_sd"] = m, sd
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Q3  cross-extractant selection
# --------------------------------------------------------------------------------------
def cross_extractant(tab: pd.DataFrame, arms: list[str], *, min_ext: int = MIN_EXT_RANK,
                     unit: str = "extractant") -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each metal pair, rank the held-out systems that measured it.

    ``unit='extractant'``: an extractant's several cells (different conditions) are collapsed by
    the median, which is the design question 'which molecule'.  ``unit='cell'`` keeps the
    (extractant, condition) system, which is the procurement question 'which recipe'.
    """
    key = ["split_seed", "A", "B"]
    agg_key = key + ([unit] if unit == "extractant" else ["cell_id", "extractant"])
    g = tab.groupby(agg_key, sort=False)[["y"] + arms].median().reset_index()
    recs = []
    for (seed, a, bmet), blk in g.groupby(key, sort=False):
        n = len(blk)
        if n < min_ext:
            continue
        obs = blk["y"].to_numpy()
        if np.ptp(obs) == 0:
            continue
        hi, lo = obs.max(), obs.min()
        best_hi = (obs >= hi - 1e-12).astype(float)
        best_lo = (obs <= lo + 1e-12).astype(float)
        base = {"split_seed": seed, "A": a, "B": bmet, "n_units": n,
                "obs_spread": float(hi - lo)}
        for arm in arms + ["_RANDOM"]:
            if arm == "_RANDOM":
                p = np.zeros(n)
            else:
                p = blk[arm].to_numpy()
            rec = dict(base, arm=arm, spearman=_spearman(p, obs),
                       # maximise A over B
                       top1_up=_tie_expected_top1(p, best_hi),
                       regret_up=hi - _tie_expected_value(p, obs),
                       # maximise B over A
                       top1_down=_tie_expected_top1(-p, best_lo),
                       regret_down=_tie_expected_value(-p, obs) - lo)
            rec["top1"] = 0.5 * (rec["top1_up"] + rec["top1_down"])
            rec["regret"] = 0.5 * (rec["regret_up"] + rec["regret_down"])
            recs.append(rec)
    per_pair = pd.DataFrame(recs)
    if per_pair.empty:
        return pd.DataFrame(), per_pair
    rows = []
    for arm, blk in per_pair.groupby("arm"):
        rec = {"arm": arm, "n_pair_tasks": int(len(blk)),
               "n_distinct_pairs": int(blk.groupby(["A", "B"]).ngroups),
               "median_n_units": float(blk["n_units"].median())}
        for col in ("spearman", "top1", "regret", "top1_up", "top1_down"):
            per_seed = blk.groupby("split_seed")[col].mean()
            rec[col] = float(per_seed.mean())
            rec[col + "_sd"] = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows), per_pair


def industrial_table(per_pair: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    rows = []
    for a, b in INDUSTRIAL:
        blk = per_pair[(per_pair["A"] == a) & (per_pair["B"] == b)]
        if blk.empty:
            rows.append({"pair": f"{a}/{b}", "arm": "-", "n_units": np.nan, "note": "not ranked"})
            continue
        for arm in arms + ["_RANDOM"]:
            s = blk[blk["arm"] == arm]
            if s.empty:
                continue
            rows.append({"pair": f"{a}/{b}", "arm": arm,
                         "n_units": float(s["n_units"].mean()),
                         "obs_spread": float(s["obs_spread"].mean()),
                         "spearman": float(s["spearman"].mean()),
                         "top1": float(s["top1"].mean()),
                         "regret": float(s["regret"].mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Q4  calibration and coverage
# --------------------------------------------------------------------------------------
def calibration_slope(tab: pd.DataFrame, arm: str) -> tuple[float, float, float, float]:
    """Weighted OLS of observed on predicted log SF; extractants weighted equally.

    Returns (slope, intercept, slope sd over seeds, R^2).
    """
    w = 1.0 / tab.groupby(["split_seed", "extractant"])["y"].transform("size").to_numpy()
    x, y = tab[arm].to_numpy(), tab["y"].to_numpy()
    if np.ptp(x) == 0:
        return np.nan, float(np.average(y, weights=w)), np.nan, np.nan
    A = np.c_[np.ones(len(x)), x] * np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
    pred = beta[0] + beta[1] * x
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    slopes = []
    for seed, blk in tab.groupby("split_seed"):
        ws = 1.0 / blk.groupby("extractant")["y"].transform("size").to_numpy()
        xs, ys = blk[arm].to_numpy(), blk["y"].to_numpy()
        if np.ptp(xs) == 0:
            continue
        As = np.c_[np.ones(len(xs)), xs] * np.sqrt(ws)[:, None]
        bs, *_ = np.linalg.lstsq(As, ys * np.sqrt(ws), rcond=None)
        slopes.append(bs[1])
    sd = float(np.std(slopes, ddof=1)) if len(slopes) > 1 else np.nan
    return float(beta[1]), float(beta[0]), sd, 1.0 - ss_res / ss_tot


def calibration_bins(tab: pd.DataFrame, arm: str, n_bins: int = 8) -> pd.DataFrame:
    x = tab[arm].to_numpy()
    if np.ptp(x) == 0:
        return pd.DataFrame([{"bin": "constant", "n": len(x), "pred_mean": float(x[0]),
                              "obs_mean": float(tab["y"].mean()),
                              "obs_sd": float(tab["y"].std())}])
    q = pd.qcut(pd.Series(x), n_bins, duplicates="drop")
    rows = []
    for b, idx in tab.groupby(q.to_numpy(), observed=True).groups.items():
        blk = tab.loc[idx]
        rows.append({"bin": str(b), "n": len(blk), "pred_mean": float(blk[arm].mean()),
                     "obs_mean": float(blk["y"].mean()), "obs_sd": float(blk["y"].std()),
                     "obs_median": float(blk["y"].median())})
    return pd.DataFrame(rows).sort_values("pred_mean").reset_index(drop=True)


def coverage(tab: pd.DataFrame, arm: str, conf: np.ndarray, *, coverages=COVERAGES,
             thr: float = STRONG, seed: int = 20260909) -> pd.DataFrame:
    """Selective prediction: keep the most confident q of pairs, report MAE and sign accuracy.

    Confidence is massively tied for the deployed arms (G14's |prediction| depends only on the
    metal pair), so ties are broken by a seeded random key -- never by row order, which would
    correlate with the cohort's cell ordering.
    """
    rng = np.random.default_rng(seed)
    t = tab.assign(_c=np.asarray(conf, dtype=float), _j=rng.random(len(tab)))
    t = t.sort_values(["_c", "_j"], ascending=[False, False], kind="stable")
    rows = []
    for q in coverages:
        k = max(1, int(round(q * len(t))))
        keep = t.iloc[:k]
        err = (keep["y"] - keep[arm]).abs()
        m_mae, sd_mae = _macro(keep.assign(_e=err), "_e")
        st = keep[keep["y"].abs() >= thr]
        hit = (np.sign(st["y"]) == np.sign(st[arm])).astype(float)
        hit = hit.where(st[arm] != 0.0, 0.5)
        m_acc, _ = _macro(st.assign(_h=hit), "_h")
        rows.append({"arm": arm, "coverage": q, "n_pairs": int(len(keep)),
                     "mae": m_mae, "mae_sd": sd_mae, "sign_acc": m_acc,
                     "n_strong": int(len(st))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# headline bundle (used for the permutation null)
# --------------------------------------------------------------------------------------
def headline(tab: pd.DataFrame, arm: str) -> dict:
    """The five numbers every arm is judged on, in one pass -- cheap enough to bootstrap."""
    from gen13sep.metrics import per_extractant, summarise
    pe = per_extractant(tab, [arm])
    bd = summarise(pe, tab, [arm]).iloc[0]
    sb = sign_bands(tab, [arm])
    wc, _ = within_cell(tab, [arm])
    ce, _ = cross_extractant(tab, [arm])
    w = wc[wc["arm"] == arm].iloc[0]
    c = ce[ce["arm"] == arm].iloc[0]
    slope, _, _, _ = calibration_slope(tab, arm)
    return {"arm": arm,
            "macro_mae": float(bd["macro_mae_extractant"]),
            "sign_acc": float(sb[(sb["band"] == "all") & (sb["arm"] == arm)]["sign_acc"].iat[0]),
            "cell_spearman": float(w["spearman"]),
            "cell_top1": float(w["top1"]),
            "cross_spearman": float(c["spearman"]),
            "cross_top1": float(c["top1"]),
            "cross_regret": float(c["regret"]),
            "calib_slope": float(slope)}
