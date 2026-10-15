"""Chemotype-blocked inference for pooled statistics, vectorised through per-cluster sums.

For a query table with ``y`` and one or more prediction columns, every pooled statistic gen20
reports (R², MAE, RMSE, and differences of them between two arms or two budgets) is a function of
per-cluster sums: ``n, Σy, Σy², Σ|e|, Σe²`` per arm.  A block-bootstrap draw is a multinomial count
vector over clusters, so 10 000 draws are one matrix product.  R² is recomputed inside every draw
(its denominator changes with the resample), which is the plan's requirement and what
``paired_contrasts`` (per-unit mean only) cannot do.

Intervals: percentile and BCa (mid-rank ``z0``, acceleration from the cluster jackknife, NaN when
degenerate -- gen19's convention).  ``p`` is the two-sided percentile p.  ``loco`` is the
leave-one-cluster-out statistic.  Seeds: the statistic is computed per split seed on that seed's
rows and the *same* cluster draws are applied to every seed; the headline is the mean over seeds
of the per-seed statistic (gen19's nested convention), with the per-seed values printed.
For per-extractant mean contrasts (the gen13 continuity metric) ``unit_contrast`` adds the
wild-cluster-t p from ``gen13sep.wildcluster`` beside the percentile interval, since the
percentile chemotype bootstrap rejects at 7-8 % on this cluster profile (gen13 inference audit).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import bootstrap  # noqa: F401

REPLICATES = 10_000
SEED = 8675309
LEVELS: tuple[float, ...] = (0.95, 0.9875)      # descriptive, and the Bonferroni level for four registered claims


def _sums(table: pd.DataFrame, pred_cols, cluster_col: str) -> tuple[list, np.ndarray]:
    """Per-cluster sums: columns = [n, Sy, Syy] + per arm [Sabs, Ssq]."""
    y = table["y"].to_numpy(float)
    cl = table[cluster_col].astype(str).to_numpy()
    names, code = np.unique(cl, return_inverse=True)
    G = len(names)
    cols = [np.ones_like(y), y, y * y]
    for c in pred_cols:
        e = y - table[c].to_numpy(float)
        cols += [np.abs(e), e * e]
    M = np.stack(cols, axis=1)
    S = np.zeros((G, M.shape[1]))
    np.add.at(S, code, M)
    return list(names), S


def _stats_from_sums(S: np.ndarray, n_arms: int) -> dict[str, np.ndarray]:
    """Pooled statistics for each row of ``S`` (one row per draw, or one row for the point)."""
    n, Sy, Syy = S[..., 0], S[..., 1], S[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        sst = Syy - Sy * Sy / n
        out = {}
        for a in range(n_arms):
            sabs, ssq = S[..., 3 + 2 * a], S[..., 4 + 2 * a]
            out[f"MAE_{a}"] = sabs / n
            out[f"RMSE_{a}"] = np.sqrt(ssq / n)
            r2 = 1.0 - ssq / sst
            out[f"R2_{a}"] = np.where((sst > 0) & (n >= 2), r2, np.nan)
    return out


def _bca(draws: np.ndarray, point: float, jack: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    fin = draws[np.isfinite(draws)]
    jk = jack[np.isfinite(jack)]
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
    out = []
    alpha = (1.0 - level) / 2.0
    for q in (alpha, 1.0 - alpha):
        z = norm.ppf(q)
        adj = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(float(np.quantile(fin, min(max(adj, 0.0), 1.0))))
    return out[0], out[1]


def interval_columns(draws: np.ndarray, point: float, jack: np.ndarray) -> dict:
    """Percentile and BCa bounds at every level in ``LEVELS`` (keys like ci95_low, bca9875_high)."""
    fin = draws[np.isfinite(draws)]
    out = {}
    for lv in LEVELS:
        tag = f"{lv * 100:g}".replace(".", "")
        alpha = (1.0 - lv) / 2.0
        lo, hi = (np.quantile(fin, [alpha, 1.0 - alpha]) if fin.size else (np.nan, np.nan))
        bl, bh = _bca(draws, point, jack, lv)
        out[f"ci{tag}_low"], out[f"ci{tag}_high"], out[f"bca{tag}_low"], out[f"bca{tag}_high"] = float(lo), float(hi), bl, bh
    return out


def _draw_counts(G: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, G, size=(replicates, G))
    counts = np.zeros((replicates, G))
    for g in range(G):
        counts[:, g] = (picks == g).sum(axis=1)
    return counts


def _per_seed_sums(table, arms, cluster_col, seed_col, all_clusters):
    pos = {c: i for i, c in enumerate(all_clusters)}
    out = {}
    for s in sorted(table[seed_col].unique()):
        names, S = _sums(table[table[seed_col] == s], arms, cluster_col)
        full = np.zeros((len(all_clusters), S.shape[1]))
        for nm, row in zip(names, S):
            full[pos[nm]] = row
        out[s] = full
    return out


@dataclass
class PooledContrast:
    name: str
    statistic: str
    point: float
    ci_low: float
    ci_high: float
    bca_low: float
    bca_high: float
    p_two_sided: float
    se: float
    n_clusters: int
    loco_min: float
    loco_max: float
    loco_all_positive: bool
    per_seed: dict = field(default_factory=dict)
    seeds_positive: int = 0
    n_seeds: int = 0

    def row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "per_seed"}
        d["per_seed"] = ";".join(f"{k}:{v:.5f}" for k, v in self.per_seed.items())
        d["mde_80"] = 2.80 * self.se if np.isfinite(self.se) else np.nan
        return d


def pooled_contrasts(table: pd.DataFrame, pairs_of_arms: dict[str, tuple[str, str]], *,
                     statistics=("MAE", "R2", "RMSE"), cluster_col: str = "chemotype",
                     seed_col: str = "split_seed", replicates: int = REPLICATES, seed: int = SEED) -> pd.DataFrame:
    """For each named contrast ``(reference, candidate)`` and statistic, the seed-mean difference
    in the candidate's favour (R²: candidate − reference; MAE/RMSE: reference − candidate), with
    cluster-bootstrap intervals from one shared draw matrix across all contrasts."""
    arms = sorted({a for pr in pairs_of_arms.values() for a in pr})
    idx = {a: i for i, a in enumerate(arms)}
    all_clusters = sorted(table[cluster_col].astype(str).unique())
    G = len(all_clusters)
    counts = _draw_counts(G, replicates, seed)
    per_seed_S = _per_seed_sums(table, arms, cluster_col, seed_col, all_clusters)
    seeds = sorted(per_seed_S)
    rows = []
    for label, (ref, cand) in pairs_of_arms.items():
        for stat in statistics:
            sign = 1.0 if stat == "R2" else -1.0
            seed_point, seed_draws, seed_jack = {}, [], []
            for s in seeds:
                S = per_seed_S[s]
                st = _stats_from_sums(S.sum(axis=0, keepdims=True), len(arms))
                seed_point[s] = float(sign * (st[f"{stat}_{idx[cand]}"] - st[f"{stat}_{idx[ref]}"])[0])
                dr = _stats_from_sums(counts @ S, len(arms))
                seed_draws.append(sign * (dr[f"{stat}_{idx[cand]}"] - dr[f"{stat}_{idx[ref]}"]))
                tot = S.sum(axis=0, keepdims=True)
                sj = _stats_from_sums(tot - S, len(arms))          # leave-one-cluster-out, vectorised
                seed_jack.append(sign * (sj[f"{stat}_{idx[cand]}"] - sj[f"{stat}_{idx[ref]}"]))
            with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", category=RuntimeWarning)
                point = float(np.nanmean(list(seed_point.values())))
                draws = np.nanmean(np.vstack(seed_draws), axis=0)
                jack = np.nanmean(np.vstack(seed_jack), axis=0)
            fin = draws[np.isfinite(draws)]
            lo, hi = (np.quantile(fin, [0.025, 0.975]) if fin.size else (np.nan, np.nan))
            p = float(min(1.0, 2 * min((fin <= 0).mean(), (fin >= 0).mean()))) if fin.size else np.nan
            bl, bh = _bca(draws, point, jack)
            jf = jack[np.isfinite(jack)]
            row = PooledContrast(name=label, statistic=stat, point=point, ci_low=float(lo), ci_high=float(hi),
                                 bca_low=bl, bca_high=bh, p_two_sided=p,
                                 se=float(fin.std(ddof=1)) if fin.size > 1 else np.nan, n_clusters=G,
                                 loco_min=float(jf.min()) if jf.size else np.nan,
                                 loco_max=float(jf.max()) if jf.size else np.nan,
                                 loco_all_positive=bool(jf.size and np.all(jf > 0)),
                                 per_seed=seed_point, seeds_positive=int(sum(v > 0 for v in seed_point.values())),
                                 n_seeds=len(seeds)).row()
            row.update(interval_columns(draws, point, jack))
            rows.append(row)
    return pd.DataFrame(rows)


def pooled_levels(table: pd.DataFrame, arms, *, cluster_col: str = "chemotype", seed_col: str = "split_seed",
                  replicates: int = REPLICATES, seed: int = SEED) -> pd.DataFrame:
    """Seed-mean pooled R²/MAE/RMSE per arm with cluster-bootstrap percentile intervals."""
    arms = list(arms)
    all_clusters = sorted(table[cluster_col].astype(str).unique())
    G = len(all_clusters)
    counts = _draw_counts(G, replicates, seed)
    per_seed = _per_seed_sums(table, arms, cluster_col, seed_col, all_clusters)
    seeds = sorted(per_seed)
    rows = []
    for a_i, a in enumerate(arms):
        for stat in ("R2", "MAE", "RMSE"):
            pts, drs = [], []
            for s in seeds:
                S = per_seed[s]
                pts.append(_stats_from_sums(S.sum(axis=0, keepdims=True), len(arms))[f"{stat}_{a_i}"][0])
                drs.append(_stats_from_sums(counts @ S, len(arms))[f"{stat}_{a_i}"])
            with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", category=RuntimeWarning)
                draws = np.nanmean(np.vstack(drs), axis=0)
                point_val = float(np.nanmean(pts)) if np.isfinite(pts).any() else float("nan")
            fin = draws[np.isfinite(draws)]
            rows.append({"arm": a, "statistic": stat, "point": point_val,
                         "seed_sd": float(np.nanstd(pts, ddof=1)) if len(pts) > 1 else np.nan,
                         "ci_low": float(np.quantile(fin, 0.025)) if fin.size else np.nan,
                         "ci_high": float(np.quantile(fin, 0.975)) if fin.size else np.nan,
                         "per_seed": ";".join(f"{s}:{v:.5f}" for s, v in zip(seeds, pts)), "n_clusters": G})
    return pd.DataFrame(rows)


def unit_contrast(d: np.ndarray, clusters: np.ndarray, *, reps: int = 9999) -> dict:
    """Wild-cluster-t (CR2, Rademacher/Webb) and cluster-robust SE for a per-unit paired delta
    (positive favours the candidate), beside the percentile cluster bootstrap used elsewhere."""
    from gen13sep.wildcluster import cluster_robust, wild_cluster_t
    d = np.asarray(d, float)
    clusters = np.asarray(clusters).astype(str)
    ok = np.isfinite(d)
    d, clusters = d[ok], clusters[ok]
    out = {"point": float(d.mean()), "n_units": int(len(d)), "n_clusters": int(len(set(clusters)))}
    try:
        w = wild_cluster_t(d, clusters, reps=reps, se_kind="CR2")
        wd = w.__dict__ if hasattr(w, "__dict__") else dict(w)
        out.update({f"wcr_{k}": v for k, v in wd.items() if isinstance(v, (int, float, str, bool))})
    except Exception as exc:  # pragma: no cover
        out["wcr_error"] = str(exc)[:200]
    try:
        cr = cluster_robust(d, clusters)
        out.update({f"cr_{k}": v for k, v in cr.__dict__.items() if isinstance(v, (int, float))})
    except Exception as exc:  # pragma: no cover
        out["cr_error"] = str(exc)[:200]
    return out


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    m = len(q)
    order = np.argsort(q, kind="mergesort")
    ranked = q[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def unit_mae_contrast(wide: pd.DataFrame, reference: str, candidate: str, *, cluster_col: str = "chemotype",
                      seed_col: str = "split_seed") -> dict:
    """Per-extractant MAE delta (reference − candidate, seed-mean per extractant) with the
    wild-cluster-t p and cluster-robust SE: the continuity statistic beside the pooled contrast."""
    e = wide.assign(_r=(wide["y"] - wide[reference]).abs(), _c=(wide["y"] - wide[candidate]).abs())
    per = e.groupby([seed_col, "extractant"]).agg(r=("_r", "mean"), c=("_c", "mean"), cl=(cluster_col, "first"))
    per = per.groupby("extractant").agg(r=("r", "mean"), c=("c", "mean"), cl=("cl", "first"))
    d = (per["r"] - per["c"]).to_numpy(float)
    out = unit_contrast(d, per["cl"].to_numpy())
    out.update({"reference": reference, "candidate": candidate, "units_improved": int((d > 0).sum())})
    return out


def blocked_unit_contrast(d: np.ndarray, clusters: np.ndarray, *, replicates: int = REPLICATES, seed: int = SEED) -> dict:
    """Chemotype-blocked bootstrap of a per-unit mean (reference − candidate per extractant, seed-mean),
    at every level in LEVELS, with the jackknife over clusters and the two-sided percentile p."""
    d = np.asarray(d, float)
    clusters = np.asarray(clusters).astype(str)
    ok = np.isfinite(d)
    d, clusters = d[ok], clusters[ok]
    names, code = np.unique(clusters, return_inverse=True)
    G = len(names)
    sums = np.zeros(G); cnt = np.zeros(G)
    np.add.at(sums, code, d); np.add.at(cnt, code, 1.0)
    counts = _draw_counts(G, replicates, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (counts @ sums) / (counts @ cnt)
        jack = (sums.sum() - sums) / (cnt.sum() - cnt)
    point = float(d.mean())
    fin = draws[np.isfinite(draws)]
    out = {"point": point, "n_units": int(len(d)), "n_clusters": G, "units_improved": int((d > 0).sum()),
           "p_two_sided": float(min(1.0, 2 * min((fin <= 0).mean(), (fin >= 0).mean()))) if fin.size else np.nan,
           "se": float(fin.std(ddof=1)) if fin.size > 1 else np.nan,
           "loco_min": float(np.nanmin(jack)), "loco_max": float(np.nanmax(jack)),
           "loco_all_positive": bool(np.all(jack[np.isfinite(jack)] > 0)),
           "kish_clusters": float(cnt.sum() ** 2 / (cnt ** 2).sum())}
    out["mde_80"] = 2.80 * out["se"] if np.isfinite(out["se"]) else np.nan
    out.update(interval_columns(draws, point, jack))
    return out


def macro_mae_contrast(wide: pd.DataFrame, reference: str, candidate: str, *, cluster_col: str = "chemotype",
                       seed_col: str = "split_seed", replicates: int = REPLICATES) -> dict:
    """Extractant-macro MAE contrast: per-extractant MAE (pairs pooled within the extractant), averaged
    over seeds, reference − candidate; chemotype-blocked intervals plus the wild-cluster-t p."""
    e = wide.assign(_r=(wide["y"] - wide[reference]).abs(), _c=(wide["y"] - wide[candidate]).abs())
    per = e.groupby([seed_col, "extractant"]).agg(r=("_r", "mean"), c=("_c", "mean"), cl=(cluster_col, "first"))
    per = per.groupby("extractant").agg(r=("r", "mean"), c=("c", "mean"), cl=("cl", "first"))
    d = (per["r"] - per["c"]).to_numpy(float)
    out = blocked_unit_contrast(d, per["cl"].to_numpy(), replicates=replicates)
    out.update({f"wcr_{k}": v for k, v in unit_contrast(d, per["cl"].to_numpy()).items() if k.startswith("wcr_") or k.startswith("cr_")})
    per_seed = e.groupby([seed_col, "extractant"]).agg(r=("_r", "mean"), c=("_c", "mean")).groupby(level=0).apply(lambda g: float((g["r"] - g["c"]).mean()))
    out.update({"reference": reference, "candidate": candidate, "statistic": "MAE_macro_extractant",
                "macro_reference": float(per["r"].mean()), "macro_candidate": float(per["c"].mean()),
                "seeds_positive": int((per_seed > 0).sum()), "n_seeds": int(len(per_seed)),
                "per_seed": ";".join(f"{k}:{v:.5f}" for k, v in per_seed.items())})
    return out
