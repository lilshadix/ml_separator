"""Metrics for round 3 on the long tables of r3.harness.

Primary (historical definition): pooled R^2 of strict-adjacent log SF inside each split seed, then the
mean over seeds; cohort C<K> = rows with k_max >= K at budget k.  Also MAE, RMSE, bias, direction
accuracy (a zero prediction scores 0.5; the constant baseline is FLAT = 0), extractant- and
chemotype-macro MAE, DGA (sc009) vs non-DGA, one-read-end vs both-hidden rows, per pair type, and
paired contrasts with chemotype-blocked bootstrap intervals (gen20 inference; nominal 95 %, about 90 %
real coverage with ~4 effective chemotypes).  Undefined R^2 -> NaN with a reason, never clipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths  # noqa: F401
from g21 import evaluate as E  # noqa: E402

DGA = "sc009"
PAIR_ORDER = ["La/Ce", "Ce/Pr", "Pr/Nd", "Sm/Eu", "Eu/Gd", "Gd/Tb", "Tb/Dy", "Dy/Ho", "Ho/Er", "Er/Tm", "Tm/Yb", "Yb/Lu"]


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = ((y - y.mean()) ** 2).sum()
    if len(y) < 2 or sst <= 0:
        return np.nan
    return 1.0 - ((y - p) ** 2).sum() / sst


def per_seed(t: pd.DataFrame, arm: str) -> pd.DataFrame:
    out = []
    for s, g in t.groupby("split_seed"):
        y, p = g.y.to_numpy(float), g[arm].to_numpy(float)
        e = p - y
        nz = np.abs(y) > 0
        sp = np.sign(p[nz])
        dacc = float(np.where(sp == 0, 0.5, (sp == np.sign(y[nz])).astype(float)).mean()) if nz.any() else np.nan
        mex = g.assign(ae=np.abs(e)).groupby("extractant").ae.mean().mean()
        mch = g.assign(ae=np.abs(e)).groupby(["chemotype", "extractant"]).ae.mean().groupby(level=0).mean().mean()
        out.append({"split_seed": s, "R2": _r2(y, p), "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "bias": e.mean(),
                    "dir_acc": dacc, "MAE_macro_ext": mex, "MAE_macro_chem": mch, "n_rows": len(g), "n_cells": g.cell_id.nunique()})
    return pd.DataFrame(out)


def summary(t: pd.DataFrame, arms, *, K: int = 3, k: int = 3, subsets: bool = True) -> pd.DataFrame:
    """Seed-mean statistics on C<K> at budget k, overall and on declared subsets."""
    base = t[(t.k_max >= K) & (t.k == k)]
    subs = {"all": base}
    if subsets:
        subs.update({"DGA_sc009": base[base.chemotype == DGA], "non_DGA": base[base.chemotype != DGA],
                     "one_end_read": base[base.a_read | base.b_read], "both_hidden": base[~(base.a_read | base.b_read)],
                     "Pr/Nd": base[(base.A == "Pr") & (base.B == "Nd")]})
    rows = []
    for name, sub in subs.items():
        if sub.empty:
            continue
        for a in arms:
            ps = per_seed(sub, a)
            rec = {"subset": name, "arm": a, "K": K, "k": k}
            for c in ("R2", "MAE", "RMSE", "bias", "dir_acc", "MAE_macro_ext", "MAE_macro_chem"):
                rec[c] = float(ps[c].mean())
            rec["R2_per_seed"] = ";".join(f"{v:.4f}" for v in ps.R2)
            rec["R2_min_seed"] = float(ps.R2.min())
            rec["n_rows_per_seed"] = float(ps.n_rows.mean())
            rec["n_cells_per_seed"] = float(ps.n_cells.mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def per_pair(t: pd.DataFrame, arms, *, K=3, k=3) -> pd.DataFrame:
    base = t[(t.k_max >= K) & (t.k == k)]
    rows = []
    for (A, B), g in base.groupby(["A", "B"]):
        for a in arms:
            r2s = [_r2(h.y, h[a]) for _, h in g.groupby("split_seed")]
            e = g[a] - g.y
            rows.append({"pair": f"{A}/{B}", "arm": a, "R2": float(np.nanmean(r2s)), "MAE": float(e.abs().mean()),
                         "bias": float(e.mean()), "y_sd": float(g.y.std(ddof=0)), "n_rows_per_seed": len(g) / g.split_seed.nunique()})
    d = pd.DataFrame(rows)
    d["order"] = d.pair.map({p: i for i, p in enumerate(PAIR_ORDER)})
    return d.sort_values(["order", "arm"]).drop(columns="order")


def contrasts(t: pd.DataFrame, pairs, *, K=3, k=3, replicates=2000) -> pd.DataFrame:
    """(candidate, reference) pairs on identical rows; signed so that positive favours the candidate
    (R2: cand - ref; MAE/RMSE: ref - cand); chemotype-blocked bootstrap + per-seed signs."""
    from gen20k import inference as INF
    base = t[(t.k_max >= K) & (t.k == k)].copy()
    arms = sorted({a for pr in pairs for a in pr})
    for a in arms:
        base[f"{a}|k{k}"] = base[a]
    comps = {f"{c} - {r}": (f"{r}|k{k}", f"{c}|k{k}") for c, r in pairs}
    ct = INF.pooled_contrasts(base, comps, replicates=replicates)
    seeds = []
    for c, r in pairs:
        for s, g in base.groupby("split_seed"):
            y = g.y.to_numpy(float)
            sst = ((y - y.mean()) ** 2).sum()
            ec, er = y - g[c].to_numpy(float), y - g[r].to_numpy(float)
            seeds.append({"name": f"{c} - {r}", "split_seed": s, "dR2": ((er ** 2).sum() - (ec ** 2).sum()) / sst,
                          "dMAE": np.abs(er).mean() - np.abs(ec).mean()})
    sd = pd.DataFrame(seeds).groupby("name").agg(seeds_R2_better=("dR2", lambda v: int((v > 0).sum())),
                                                 seeds_MAE_better=("dMAE", lambda v: int((v > 0).sum())))
    return ct.merge(sd, left_on="name", right_index=True, how="left")


def levels_ci(t: pd.DataFrame, arms, *, K=3, ks=(0, 1, 2, 3), replicates=2000):
    pol = t["policy"].iloc[0]
    lv, _ = E.cohort_levels(t, list(arms), K=K, ks=list(ks), replicates=replicates, policy=pol)
    return lv
