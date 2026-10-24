"""Round-4 metrics on top of r3.metrics (METRICS.md).

* R^2 (primary, historical): pooled strict-adjacent R^2 inside each split seed, then the seed mean
  (r3.metrics.per_seed, unchanged).
* Balanced MAE: round 3's chemotype-macro MAE (``MAE_macro_chem``): inside each seed, |error| averaged over
  the rows of each extractant, extractants averaged inside each chemotype, chemotypes averaged with equal
  weight; then the seed mean.  (This evaluation weighting differs from the references' training weighting,
  DATA_CONTRACT.md "Training masks and weights".)
* Clusters for uncertainty: (a) chemotype (round-3 practice); (b) declared "independent source" clusters =
  connected components of chemotype, publication and gen19 cross-publication copy group (``cluster_cpg``).  One
  shared bootstrap draw matrix over clusters for all seeds (gen20k.inference); percentile, BCa and a
  small-cluster cluster-jackknife t(G-1) interval for every paired contrast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as SS

from . import paths  # noqa: F401
from r3 import metrics as MT  # noqa: E402
from gen20k import inference as INF  # noqa: E402

REPLICATES = 2000
CLUSTERINGS = ("chemotype", "cluster_cpg")
DGA_CHEMOTYPE = "sc009"


def add_clusters(t: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Adds ``copy_group`` and ``cluster_cpg`` (union-find over chemotype, publication, copy group)."""
    from gen20k.splits import copy_group_map
    cg = copy_group_map()
    t = t.copy()
    missing = sorted(set(t.publication.astype(str)) - set(cg))
    if missing:
        raise KeyError(f"{len(missing)} publications have no copy group: {missing[:5]}")
    t["copy_group"] = t.publication.astype(str).map(cg)
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def union(a, b):
        par[find(a)] = find(b)
    for c, p, g in t[["chemotype", "publication", "copy_group"]].drop_duplicates().itertuples(index=False):
        union("c:" + str(c), "p:" + str(p))
        union("p:" + str(p), "g:" + str(g))
    t["cluster_cpg"] = t.chemotype.map(lambda c: find("c:" + str(c)))
    return t


def kish(t: pd.DataFrame, cluster: str) -> float:
    """Kish effective number of clusters for the pooled rows of one seed (row shares)."""
    s = t.groupby(cluster).size().to_numpy(float)
    s = s / s.sum()
    return float(1.0 / (s ** 2).sum())


def _jackknife_t(point: float, jk: np.ndarray) -> tuple[float, float, float]:
    """Cluster-jackknife standard error and t(G-1) 95 % interval around the point estimate."""
    jk = jk[np.isfinite(jk)]
    G = len(jk)
    if G < 3:
        return np.nan, np.nan, np.nan
    se = float(np.sqrt((G - 1) / G * ((jk - jk.mean()) ** 2).sum()))
    q = float(SS.t.ppf(0.975, G - 1))
    return se, point - q * se, point + q * se


def balanced_mae_contrast(t: pd.DataFrame, cand: str, ref: str, cluster: str, *, replicates=REPLICATES,
                          seed=INF.SEED) -> dict:
    """ref - cand of the balanced MAE (positive favours the candidate), seed mean, with a cluster bootstrap.
    Every chemotype lies inside one cluster for both clusterings, so a draw reweights whole chemotypes."""
    clusters = sorted(t[cluster].astype(str).unique())
    cidx = {c: i for i, c in enumerate(clusters)}
    counts = INF._draw_counts(len(clusters), replicates, seed)            # (replicates, G)
    pts, draws, jacks = [], [], []
    for s, g in t.groupby("split_seed"):
        g = g.assign(ec=(g[cand] - g.y).abs(), er=(g[ref] - g.y).abs())
        ext = g.groupby(["chemotype", "extractant"])[["ec", "er"]].mean()
        ch = ext.groupby(level=0).mean()
        cl = g.groupby("chemotype")[cluster].first().astype(str).loc[ch.index].map(cidx).to_numpy()
        d = (ch.er - ch.ec).to_numpy()
        pts.append(float(d.mean()))
        w = counts[:, cl]                                                   # (replicates, n_chemotypes)
        draws.append((w * d).sum(1) / np.maximum(w.sum(1), 1e-12))
        jk = []
        for gi in range(len(clusters)):
            keep = cl != gi
            jk.append(float(d[keep].mean()) if keep.any() else np.nan)
        jacks.append(np.array(jk))
    point = float(np.mean(pts))
    dr = np.mean(np.vstack(draws), 0)
    jk = np.nanmean(np.vstack(jacks), 0)
    lo, hi = np.quantile(dr, [0.025, 0.975])
    bl, bh = INF._bca(dr, point, jk)
    se_j, jl, jh = _jackknife_t(point, jk)
    return {"name": f"{cand} - {ref}", "statistic": "MAE_balanced", "cluster": cluster, "point": point,
            "ci_low": float(lo), "ci_high": float(hi), "bca_low": bl, "bca_high": bh,
            "p_two_sided": float(min(1.0, 2 * min((dr <= 0).mean(), (dr >= 0).mean()))), "se": float(dr.std(ddof=1)),
            "jk_se": se_j, "jk_t_low": jl, "jk_t_high": jh, "n_clusters": len(clusters),
            "loco_min": float(np.nanmin(jk)), "loco_max": float(np.nanmax(jk)),
            "loco_all_positive": bool(np.all(jk[np.isfinite(jk)] > 0)),
            "seeds_positive": int(sum(p > 0 for p in pts)), "n_seeds": len(pts)}


def _jackknife_pooled(base: pd.DataFrame, cand: str, ref: str, cluster: str, stat: str) -> np.ndarray:
    """Leave-one-cluster-out seed-mean contrast of a pooled statistic (R2 / MAE / RMSE), as gen20k computes it."""
    arms = [ref, cand]
    clusters = sorted(base[cluster].astype(str).unique())
    ps = INF._per_seed_sums(base.assign(**{cluster: base[cluster].astype(str)}), arms, cluster, "split_seed", clusters)
    sign = 1.0 if stat == "R2" else -1.0
    out = []
    for S in ps.values():
        sj = INF._stats_from_sums(S.sum(0, keepdims=True) - S, 2)
        out.append(sign * (sj[f"{stat}_1"] - sj[f"{stat}_0"]))
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.vstack(out), 0)


def contrasts(t: pd.DataFrame, pairs, *, K=3, k=3, replicates=REPLICATES) -> pd.DataFrame:
    """(candidate, reference) pairs on identical rows under both clusterings: R^2 / MAE / RMSE from
    gen20k.inference.pooled_contrasts, plus the balanced MAE, each with a cluster-jackknife t(G-1) interval.
    Positive always favours the candidate."""
    base = t[(t.k_max >= K) & (t.k == k)].copy()
    out = []
    for cl in CLUSTERINGS:
        arms = sorted({a for pr in pairs for a in pr})
        b = base.copy()
        for a in arms:
            b[f"{a}|k{k}"] = b[a]
        comps = {f"{c} - {r}": (f"{r}|k{k}", f"{c}|k{k}") for c, r in pairs}
        ct = INF.pooled_contrasts(b, comps, replicates=replicates, cluster_col=cl)
        ct["cluster"] = cl
        jrows = []
        for (c, r), row in zip([(c, r) for c, r in pairs for _ in ("MAE", "R2", "RMSE")], ct.itertuples()):
            assert row.name == f"{c} - {r}", (row.name, c, r)
            jk = _jackknife_pooled(base, c, r, cl, row.statistic)
            se_j, jl, jh = _jackknife_t(row.point, jk)
            jrows.append({"jk_se": se_j, "jk_t_low": jl, "jk_t_high": jh})
        ct = pd.concat([ct.reset_index(drop=True), pd.DataFrame(jrows)], axis=1)
        out.append(ct)
        out.append(pd.DataFrame([balanced_mae_contrast(base, c, r, cl, replicates=replicates) for c, r in pairs]))
    res = pd.concat(out, ignore_index=True, sort=False)
    res["excludes_zero_all_three"] = (res.ci_low > 0) & (res.bca_low > 0) & (res.jk_t_low > 0)
    return res


def subsets(base: pd.DataFrame) -> dict:
    out = {"all": base, "sc009": base[base.chemotype == DGA_CHEMOTYPE], "non_sc009": base[base.chemotype != DGA_CHEMOTYPE],
           "one_end_read": base[base.a_read | base.b_read], "both_hidden": base[~(base.a_read | base.b_read)]}
    if "dga_donor" in base:
        out["DGA_donor"] = base[base.dga_donor]
        out["non_DGA_donor"] = base[~base.dga_donor]
    for pair in ("Pr/Nd", "Eu/Gd", "Gd/Tb"):
        A, B = pair.split("/")
        out[pair] = base[(base.A == A) & (base.B == B)]
    return out


def summary(t: pd.DataFrame, arms, *, K=3, k=3) -> pd.DataFrame:
    """Seed-mean statistics on C<K> at budget k, overall and on the declared subsets (METRICS.md)."""
    base = t[(t.k_max >= K) & (t.k == k)]
    rows = []
    for name, sub in subsets(base).items():
        if sub.empty:
            continue
        for a in arms:
            ps = MT.per_seed(sub, a)
            rec = {"subset": name, "arm": a, "K": K, "k": k}
            for c in ("R2", "MAE", "RMSE", "bias", "dir_acc", "MAE_macro_ext", "MAE_macro_chem"):
                rec[c] = float(ps[c].mean())
            rec["R2_per_seed"] = ";".join(f"{v:.4f}" for v in ps.R2)
            rec["R2_min_seed"] = float(ps.R2.min())
            rec["R2_undefined_reason"] = "" if ps.R2.notna().all() else "fewer than 2 rows or zero target variance in a seed"
            rec["n_rows_per_seed"] = float(ps.n_rows.mean())
            rec["n_cells_per_seed"] = float(ps.n_cells.mean())
            rows.append(rec)
    return pd.DataFrame(rows).rename(columns={"MAE_macro_chem": "MAE_balanced"})


def per_publication(t: pd.DataFrame, arms, *, K=3, k=3) -> pd.DataFrame:
    """Per publication: seed-mean pooled R^2 (NaN with reason when undefined), row MAE, rows and cells per seed."""
    base = t[(t.k_max >= K) & (t.k == k)]
    rows = []
    for pub, g in base.groupby("publication"):
        for a in arms:
            r2 = [MT._r2(h.y, h[a]) for _, h in g.groupby("split_seed")]
            rows.append({"publication": pub, "arm": a, "R2": float(np.nanmean(r2)) if np.isfinite(r2).any() else np.nan,
                         "R2_undefined_reason": "" if np.isfinite(r2).all() else "fewer than 2 rows or zero variance",
                         "MAE": float((g[a] - g.y).abs().mean()), "n_rows_per_seed": len(g) / g.split_seed.nunique(),
                         "n_cells": int(g.cell_id.nunique()), "chemotypes": ";".join(sorted(g.chemotype.unique()))})
    return pd.DataFrame(rows)


def levels(t: pd.DataFrame, arms, *, K=3, k=3, replicates=REPLICATES) -> pd.DataFrame:
    """Seed-mean pooled R^2 / MAE / RMSE levels with cluster-bootstrap percentile intervals under both clusterings."""
    base = t[(t.k_max >= K) & (t.k == k)]
    out = []
    for cl in CLUSTERINGS:
        lv = INF.pooled_levels(base, list(arms), cluster_col=cl, replicates=replicates)
        lv["cluster"] = cl
        out.append(lv)
    return pd.concat(out, ignore_index=True)


def calibration_diag(t: pd.DataFrame, arms, *, K=3, k=3) -> pd.DataFrame:
    """Diagnostics of PROTOCOL section 8 item 7 (seed means): raw R^2, RMSE, bias, prediction variance, target
    variance, Pearson^2, and the test-fitted optimal scale b* = cov(y, p) / var(p) -- an ORACLE, never a score."""
    base = t[(t.k_max >= K) & (t.k == k)]
    rows = []
    for a in arms:
        ps = []
        for s, g in base.groupby("split_seed"):
            y, p = g.y.to_numpy(float), g[a].to_numpy(float)
            vp = p.var()
            ps.append({"R2": MT._r2(y, p), "RMSE": float(np.sqrt(((p - y) ** 2).mean())), "bias": float((p - y).mean()),
                       "pred_var": float(vp), "y_var": float(y.var()),
                       "pearson2": float(np.corrcoef(y, p)[0, 1] ** 2) if vp > 0 else np.nan,
                       "oracle_scale": float(((y - y.mean()) * (p - p.mean())).mean() / vp) if vp > 0 else np.nan})
        d = pd.DataFrame(ps).mean()
        rows.append({"arm": a, **d.to_dict()})
    return pd.DataFrame(rows)


def cluster_counts(t: pd.DataFrame, *, K=3, k=3) -> dict:
    base = t[(t.k_max >= K) & (t.k == k)]
    one = base[base.split_seed == base.split_seed.iloc[0]]
    return {cl: {"nominal": int(one[cl].nunique()), "kish_rows": kish(one, cl),
                 "largest_share": float(one.groupby(cl).size().max() / len(one))} for cl in CLUSTERINGS} | \
        {"publication": {"nominal": int(one.publication.nunique())}, "extractant": {"nominal": int(one.extractant.nunique())},
         "cells": int(one.cell_id.nunique()), "rows_per_seed": int(len(one))}
