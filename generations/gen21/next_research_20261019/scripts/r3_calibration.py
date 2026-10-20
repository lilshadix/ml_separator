#!/usr/bin/env python
"""Direction D (brief section 9): the model criterion "posterior contrast variance" against the realised
inner-validation loss -- inner cross-fit only.

What it computes
----------------
For every cached outer fold (design, dev seed, fold) and every inner chemotype block j of the OUTER
TRAINING fold (gen21 cache, rec["inner"]):

* HBK@ML: the four HBK kernels (g21.methods.HBKModel, basis B2) are fitted by chemotype-balanced weighted
  ML on block j's inner-training cells; the kernel with the lowest training NLL is kept (exactly
  r3.inner_preds.inner_table's rule, "HBK@ML_inner").
* Every inner-validation cell of block j with k_max >= 3 is read at budget k = 3 with its nested DOPT_Q
  sequence (chosen under block j's own nested covariance); 4 metals read, star support (gen20).
* For every strict-adjacent query (A lighter, |dZ| = 1; none has both ends read):
    pv_hbk   model criterion: posterior variance of the OBSERVABLE contrast y(A) - y(B) given the read
             contrasts under the fitted HBK@ML model.  C = B' S B + tau2 K(l) + sig2 I (the model's own
             cov_full, reading noise included, i.e. the observable joint covariance of the 14 log D),
             Dm = gen20 pair matrix of the support, Sigma_post = C - C Dm' (Dm C Dm')^-1 Dm C,
             pv = d' Sigma_post d with d = e_A - e_B.  The model's prediction (r3.inner_preds.hbk_predict,
             c = 1) is the posterior mean mu + C Dm' (Dm C Dm')^-1 (v - Dm mu) of this same Gaussian
             (checked to 1e-8 on every cell), so pv is the model's own expected squared error of
             that prediction.  The free cell offset cancels in every contrast.
    pv_dq    secondary criterion: the quantity DOPT_Q itself optimises -- latent query variance under
             block j's nested G14 covariance after DOPT_Q's sequential rank-1 updates with pair noise
             2 s2 (gen20k.design._posterior_update), no observable noise added.
    se_hbk   realised squared error of HBK@ML;  se_pwl  realised squared error of PWL_R
             (r3.interp.pwl_levels kind "R").
    polerr   POL_ERR's expected squared PWL(R) residual for this situation (r3.policy.config_class key,
             r3.policy.choose's lookup), table fitted by r3.policy.fit_error_table on block j's
             inner-training cells only, their DOPT_Q masks recomputed under block j's nested covariance
             (r3.corrector's inner-CV convention).
  plus pair type ("Eu/Gd"), read status ("one" end read / "none"), distance to the nearest read, weights.

Summaries per design: row-level Spearman / Pearson of each criterion with each realised loss; group
means per (pair type x read status), per pair type and per status with calibration ratios
mean(se) / mean(criterion) (unweighted and chemotype-weighted); rank agreement of the groups (Spearman,
Kendall); within-pair-type row Spearman; top/bottom quintile MSE ratio; the values the same statistics
would take if the criterion were exactly calibrated (se = pv * chi2_1, simulated); per-seed stability.

Label safety
------------
* Right after loading, the fold record is reduced to rec["inner"] plus the OUTER TRAINING index; the
  outer held-out cells, their read sequences and test_index are dropped before any computation
  (E.assert_test_isolation is run first; it reads metadata only).
* All labels are read from Y_in, a copy of bench.Y in which every row outside the outer training fold
  is NaN; assert_inner_only checks, per block, inner-train and inner-validation are disjoint subsets of
  the outer training fold, disjoint from the outer test index, and every scored cell is a validation
  cell of its block; every scored label must be finite.
* Within block j, fitted quantities (HBK parameters and kernel, POL_ERR table, the masks of its
  training situations) use only block j's inner-training cells; the validation cells' reads come from
  the block's nested covariance; a validation cell contributes its labels only to its own realised
  loss.  No outer test cell or label is touched; no final-seed file is read.

Command (repository root):
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -W ignore \
        generations/gen21/next_research_20261019/scripts/r3_calibration.py --designs BP BPG
Outputs: results/calibration/rows_<design>.parquet, summary_<design>.csv, summary.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R3))

from r3 import paths  # noqa: E402  (puts g21 / gen20k / gen13 on sys.path)
from r3 import corrector as CO  # noqa: E402,F401
from r3 import inner_preds as IP  # noqa: E402
from r3 import interp  # noqa: E402
from r3 import policy as PO  # noqa: E402
from g21 import cache as C  # noqa: E402
from g21 import evaluate as E  # noqa: E402
from g21 import methods as M  # noqa: E402
from gen20k import adapters as AD  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k import design as DS  # noqa: E402
from gen20k.data import LANTHANIDES, N_LN, Z_VEC  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

K = 3
MIN_GROUP = 20                     # groups with fewer rows are listed but left out of group-level correlations
N_SIM = 50                         # draws for the "exactly calibrated" reference
OUT = paths.RES / "calibration"


# ------------------------------------------------------------------------------------------
# label-safety
# ------------------------------------------------------------------------------------------
def reduce_record(bench, rec: dict) -> dict:
    """Keep only the inner blocks and the outer TRAINING index; drop every outer held-out object."""
    E.assert_test_isolation(bench, rec)                     # metadata only (cell / extractant / chemotype / ...)
    red = {"design": rec["design"], "split_seed": int(rec["split_seed"]), "fold": int(rec["fold"]),
           "outer_train": np.asarray(rec["outer"]["train_index"]).copy(),
           "outer_test_ids": frozenset(map(int, rec["outer"]["test_index"])),   # ids only, for the disjointness assert
           "inner": rec["inner"]}
    return red


def assert_inner_only(red: dict, blk: dict) -> None:
    otr = set(map(int, red["outer_train"]))
    ote = red["outer_test_ids"]
    itr, iva = set(map(int, blk["train_index"])), set(map(int, blk["val_index"]))
    if not itr <= otr or not iva <= otr:
        raise AssertionError("inner block reaches outside the outer training fold")
    if itr & iva:
        raise AssertionError("inner-training and inner-validation overlap")
    if (itr | iva) & ote:
        raise AssertionError("inner block touches an outer held-out cell")
    if not {int(c["ci"]) for c in blk["cells"]} <= iva:
        raise AssertionError("a scored cell is not an inner-validation cell of its block")


def masked_labels(bench, red: dict) -> np.ndarray:
    Y_in = np.full_like(np.asarray(bench.Y, float), np.nan)
    tr = red["outer_train"]
    Y_in[tr] = bench.Y[tr]
    return Y_in


# ------------------------------------------------------------------------------------------
# model criterion
# ------------------------------------------------------------------------------------------
def hbk_posterior(mdl, support):
    """Posterior mean and covariance of the observable level vector given the read contrasts."""
    Cf = mdl.cov_full(mdl.S, mdl.tau2, mdl.rho, mdl.sig2)          # observable (reading noise included)
    mu = mdl.B.T @ mdl.m
    Dm = AD.pair_matrix(support.pairs)
    G = Cf @ Dm.T
    Sm = Dm @ G
    mean = mu + G @ np.linalg.solve(Sm, np.asarray(support.values, float) - Dm @ mu)
    cov = Cf - G @ np.linalg.solve(Sm, G.T)
    return mean, cov


def dopt_q_post(blk: dict, seq, k: int) -> np.ndarray:
    post = np.asarray(blk["cov"], float).copy()
    nv = 2.0 * blk["s2"]
    for p in seq[:k]:
        post = DS._posterior_update(post, DS._pair_vector(p), nv)
    return post


def pol_err_table(bench, red: dict, blk: dict, Y_in: np.ndarray) -> dict:
    """r3.policy.fit_error_table on block j's inner-training cells only, their DOPT_Q masks recomputed
    with block j's nested covariance (no mask depends on a covariance that saw block j's validation cells)."""
    itr = set(map(int, blk["train_index"]))
    cells = []
    for ob in red["inner"]:
        for cell in ob["cells"]:
            if cell["ci"] not in itr or cell["k_max"] < K:
                continue
            dz = DS.make_design(E._view(cell), "adjacent", "metal")
            dseq = DS.support_sequence(dz, "DOPT_Q", min(dz.k_max, 6), cov=blk["cov"], noise_var=2 * blk["s2"])
            cells.append(dict(cell, seqs=dict(cell["seqs"], DOPT_Q=dseq)))
    rec2 = {"outer": {"train_index": red["outer_train"]}, "inner": [{"cells": cells}]}
    return PO.fit_error_table(bench, rec2, Y_in, k=K)


# ------------------------------------------------------------------------------------------
# rows
# ------------------------------------------------------------------------------------------
def fold_rows(bench, red: dict, *, with_polerr: bool = True) -> tuple[list[dict], dict]:
    from gen13sep.amplitude_bench import cell_weights
    Y_in = masked_labels(bench, red)
    otr = red["outer_train"]
    wmap = dict(zip(otr.tolist(), cell_weights(bench.groups[otr], bench.n_obs[otr]).tolist()))
    rows, info = [], {}
    for blk in red["inner"]:
        assert_inner_only(red, blk)
        itr = np.asarray(blk["train_index"])
        w = cell_weights(bench.groups[itr], bench.n_obs[itr])
        fits = {kk: M.HBKModel(M.hb_basis(bench, "B2"), kk).fit(Y_in[itr], w) for kk in IP.KERNELS}
        ml = min(IP.KERNELS, key=lambda kk: fits[kk].fit_info["nll"])
        mdl = fits[ml]
        err = pol_err_table(bench, red, blk, Y_in) if with_polerr else None
        info[blk["inner_fold"]] = {"ml_kernel": ml, "nll": {kk: fits[kk].fit_info["nll"] for kk in IP.KERNELS}}
        for cell in blk["cells"]:
            if cell["k_max"] < K:
                continue
            yrow = Y_in[cell["ci"]]
            seq = cell["seqs"]["DOPT_Q"]
            o = E.observe(yrow, cell, seq, K)
            read = frozenset(o.read)
            mean, pcov = hbk_posterior(mdl, o.support)
            ref = IP.hbk_predict(mdl, o.support, 1.0)
            if not np.allclose(mean, ref, rtol=0, atol=1e-8):
                raise AssertionError("posterior mean differs from the HBK prediction")
            pq = dopt_q_post(blk, seq, K)
            pwl = interp.pwl_levels(o.y_masked, cell["metals"], "R")
            nq = len(cell["query"])
            for a, b in cell["query"]:
                if a in read and b in read:
                    raise AssertionError("a scored pair has both ends read")
                y = float(yrow[a] - yrow[b])
                if not np.isfinite(y):
                    raise AssertionError("a scored label is not inside the outer training fold")
                d = np.zeros(N_LN)
                d[a], d[b] = 1.0, -1.0
                status, dmin, extrap = PO.config_class(a, b, read, cell["metals"])
                r = {"design": red["design"], "split_seed": red["split_seed"], "fold": red["fold"],
                     "inner_fold": int(blk["inner_fold"]), "ml_kernel": ml, "ci": int(cell["ci"]), "cell_id": cell["cell_id"],
                     "chemotype": cell["chemotype"], "k": K, "read_set": ",".join(LANTHANIDES[m] for m in sorted(read)),
                     "A": LANTHANIDES[a], "B": LANTHANIDES[b], "ia": int(a), "ib": int(b),
                     "pair": f"{LANTHANIDES[a]}/{LANTHANIDES[b]}", "status": status, "dmin": int(dmin), "extrap": int(extrap),
                     "a_read": a in read, "b_read": b in read, "y": y,
                     "w": wmap[int(cell["ci"])], "wq": wmap[int(cell["ci"])] / nq,
                     "pred_hbk": float(mean[a] - mean[b]), "pred_pwl": float(pwl[a] - pwl[b]),
                     "pv_hbk": float(d @ pcov @ d), "pv_dq": float(d @ pq @ d)}
                r["se_hbk"] = (r["pred_hbk"] - y) ** 2
                r["se_pwl"] = (r["pred_pwl"] - y) ** 2
                if err is not None:
                    key = (int(a), status, int(dmin), int(extrap))
                    r["polerr"] = float(err["table"].get(key, err["class_mean"].get(key[1:], 0.05)))
                rows.append(r)
    return rows, info


def inner_cache_check(t: pd.DataFrame, design: str, seed: int, fold: int, info: dict) -> dict | None:
    """Compare with the round's own inner_table cache (predictions/_inner), if present: same rows, same
    HBK@ML_inner and PWL_R predictions, same ML kernels."""
    p = paths.PRED / "_inner" / f"{design}_{seed}_{fold}_V1.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        inner, lik = pickle.load(fh)
    inner = inner[inner.k == K]
    m = t.merge(inner[["inner_fold", "ci", "A", "B", "HBK@ML_inner", "PWL_R"]].rename(columns={"A": "ia", "B": "ib"}),
                on=["inner_fold", "ci", "ia", "ib"], how="outer", indicator=True)
    return {"rows_ours": int(len(t)), "rows_cache": int(len(inner)), "unmatched": int((m["_merge"] != "both").sum()),
            "max_abs_diff_hbk": float(np.nanmax(np.abs(m["pred_hbk"] - m["HBK@ML_inner"]))),
            "max_abs_diff_pwl": float(np.nanmax(np.abs(m["pred_pwl"] - m["PWL_R"]))),
            "ml_kernel_same": all(lik[j]["ml_kernel"] == info[j]["ml_kernel"] for j in info if j in lik)}


# ------------------------------------------------------------------------------------------
# summaries
# ------------------------------------------------------------------------------------------
PAIRS_CRIT_LOSS = (("pv_hbk", "se_hbk"), ("pv_hbk", "se_pwl"), ("polerr", "se_pwl"), ("polerr", "se_hbk"),
                   ("pv_dq", "se_hbk"), ("pv_dq", "se_pwl"))
GROUPINGS = {"pair_status": ["pair", "status"], "pair": ["pair"], "status": ["status"], "dmin": ["status", "dmin"]}


def _corr(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return {"spearman": None, "pearson": None, "kendall": None, "n": int(ok.sum())}
    return {"spearman": float(stats.spearmanr(x[ok], y[ok])[0]), "pearson": float(stats.pearsonr(x[ok], y[ok])[0]),
            "kendall": float(stats.kendalltau(x[ok], y[ok])[0]), "n": int(ok.sum())}


def _quintile_ratio(c, s) -> float:
    q = pd.qcut(pd.Series(c).rank(method="first"), 5, labels=False)
    g = pd.Series(s).groupby(q.to_numpy()).mean()
    return float(g.iloc[-1] / g.iloc[0])


def group_table(t: pd.DataFrame, crits) -> pd.DataFrame:
    out = []
    for level, keys in GROUPINGS.items():
        for key, g in t.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            r = {"level": level, **dict(zip(keys, key)), "n_rows": len(g), "n_cells": g[["inner_fold", "split_seed", "fold", "ci"]].drop_duplicates().shape[0],
                 "w_sum": float(g.wq.sum())}
            for loss in ("se_hbk", "se_pwl"):
                r[f"mse_{loss[3:]}"] = float(g[loss].mean())
                r[f"wmse_{loss[3:]}"] = float((g.wq * g[loss]).sum() / g.wq.sum())
            for c in crits:
                r[f"mean_{c}"] = float(g[c].mean())
                r[f"wmean_{c}"] = float((g.wq * g[c]).sum() / g.wq.sum())
            r["ratio_hbk"] = r["mse_hbk"] / r["mean_pv_hbk"]
            r["wratio_hbk"] = r["wmse_hbk"] / r["wmean_pv_hbk"]
            if "polerr" in crits:
                r["ratio_polerr_pwl"] = r["mse_pwl"] / r["mean_polerr"]
                r["wratio_polerr_pwl"] = r["wmse_pwl"] / r["wmean_polerr"]
            out.append(r)
    tot = {"level": "overall", "n_rows": len(t), "n_cells": t[["inner_fold", "split_seed", "fold", "ci"]].drop_duplicates().shape[0],
           "w_sum": float(t.wq.sum())}
    for loss in ("se_hbk", "se_pwl"):
        tot[f"mse_{loss[3:]}"] = float(t[loss].mean())
        tot[f"wmse_{loss[3:]}"] = float((t.wq * t[loss]).sum() / t.wq.sum())
    for c in crits:
        tot[f"mean_{c}"] = float(t[c].mean())
        tot[f"wmean_{c}"] = float((t.wq * t[c]).sum() / t.wq.sum())
    tot["ratio_hbk"] = tot["mse_hbk"] / tot["mean_pv_hbk"]
    tot["wratio_hbk"] = tot["wmse_hbk"] / tot["wmean_pv_hbk"]
    if "polerr" in crits:
        tot["ratio_polerr_pwl"] = tot["mse_pwl"] / tot["mean_polerr"]
        tot["wratio_polerr_pwl"] = tot["wmse_pwl"] / tot["wmean_polerr"]
    out.append(tot)
    gt = pd.DataFrame(out)
    # ranks inside each level (1 = largest)
    for level in GROUPINGS:
        m = gt.level == level
        for col in ["mse_hbk", "mse_pwl"] + [f"mean_{c}" for c in crits]:
            gt.loc[m, f"rank_{col}"] = gt.loc[m, col].rank(ascending=False, method="min")
    return gt


def summarise(t: pd.DataFrame, gt: pd.DataFrame, crits) -> dict:
    rng = np.random.default_rng(20261019)
    s: dict = {"n_rows": int(len(t)), "n_cells": int(t[["split_seed", "fold", "inner_fold", "ci"]].drop_duplicates().shape[0])}
    # row level
    s["row_level"] = {f"{c}~{l}": _corr(t[c], t[l]) for c, l in PAIRS_CRIT_LOSS if c in crits}
    s["row_level_quintile_ratio_top_bottom"] = {f"{c}~{l}": _quintile_ratio(t[c], t[l]) for c, l in PAIRS_CRIT_LOSS if c in crits}
    # calibration
    s["calibration"] = {
        "hbk": {"mse": float(t.se_hbk.mean()), "mean_pv": float(t.pv_hbk.mean()), "ratio": float(t.se_hbk.mean() / t.pv_hbk.mean()),
                "wratio": float((t.wq * t.se_hbk).sum() / (t.wq * t.pv_hbk).sum()),
                "median_ratio_se_over_pv": float(np.median(t.se_hbk / t.pv_hbk)),
                "share_rows_se_gt_pv": float((t.se_hbk > t.pv_hbk).mean()),
                "calibrated_reference": {"median_ratio": float(stats.chi2.median(1)), "share_gt": float(stats.chi2.sf(1, 1))}},
        "dopt_q_latent_vs_hbk_loss": {"ratio": float(t.se_hbk.mean() / t.pv_dq.mean())},
    }
    if "polerr" in crits:
        s["calibration"]["polerr_vs_pwl"] = {"mse": float(t.se_pwl.mean()), "mean_polerr": float(t.polerr.mean()),
                                             "ratio": float(t.se_pwl.mean() / t.polerr.mean()),
                                             "wratio": float((t.wq * t.se_pwl).sum() / (t.wq * t.polerr).sum())}
    # group level
    gl = {}
    for level in ("pair_status", "pair", "status"):
        g = gt[(gt.level == level) & (gt.n_rows >= MIN_GROUP)]
        gl[level] = {"n_groups": int(len(g)), **{f"{c}~{l}": _corr(g[f"mean_{c}"], g[f"mse_{l[3:]}"])
                                                 for c, l in PAIRS_CRIT_LOSS if c in crits},
                     **{f"w:{c}~{l}": _corr(g[f"wmean_{c}"], g[f"wmse_{l[3:]}"]) for c, l in PAIRS_CRIT_LOSS if c in crits}}
    s["group_level"] = gl
    # pair-type order
    gp = gt[gt.level == "pair"].copy()
    gp["z"] = gp.pair.map(lambda p: int(Z_VEC[LANTHANIDES.index(p.split("/")[0])]))
    gp = gp.sort_values("z")
    s["pair_order"] = {
        "by_mean_pv_hbk_desc": gp.sort_values("mean_pv_hbk", ascending=False).pair.tolist(),
        "by_mse_hbk_desc": gp.sort_values("mse_hbk", ascending=False).pair.tolist(),
        "by_mse_pwl_desc": gp.sort_values("mse_pwl", ascending=False).pair.tolist(),
        **({"by_mean_polerr_desc": gp.sort_values("mean_polerr", ascending=False).pair.tolist()} if "polerr" in crits else {}),
        "table": gp[["pair", "n_rows", "mean_pv_hbk", "mse_hbk", "ratio_hbk", "mse_pwl"] +
                    (["mean_polerr", "ratio_polerr_pwl"] if "polerr" in crits else []) + ["mean_pv_dq"]].round(5).to_dict("records"),
    }
    # within pair type: does the criterion rank read configurations of the same pair?
    wp = {}
    for c, l in PAIRS_CRIT_LOSS:
        if c not in crits:
            continue
        vals = [stats.spearmanr(g[c], g[l])[0] for _, g in t.groupby("pair") if len(g) >= MIN_GROUP and g[c].std() > 0]
        wp[f"{c}~{l}"] = {"mean": float(np.nanmean(vals)), "min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)),
                          "n_pairs": len(vals)}
    s["within_pair_row_spearman"] = wp
    # exactly-calibrated reference (se = pv * chi2_1) for the HBK criterion
    ref_row, ref_q, ref_grp, ref_pair = [], [], [], []
    for _ in range(N_SIM):
        sim = t.pv_hbk.to_numpy() * rng.standard_normal(len(t)) ** 2
        ref_row.append(stats.spearmanr(t.pv_hbk, sim)[0])
        ref_q.append(_quintile_ratio(t.pv_hbk, sim))
        tt = t.assign(sim=sim)
        g = tt.groupby(["pair", "status"]).agg(n=("sim", "size"), pv=("pv_hbk", "mean"), se=("sim", "mean"))
        g = g[g.n >= MIN_GROUP]
        ref_grp.append(stats.spearmanr(g.pv, g.se)[0])
        g = tt.groupby("pair").agg(pv=("pv_hbk", "mean"), se=("sim", "mean"))
        ref_pair.append(stats.spearmanr(g.pv, g.se)[0])
    s["calibrated_reference_pv_hbk"] = {"row_spearman": float(np.mean(ref_row)), "quintile_ratio": float(np.mean(ref_q)),
                                        "pair_status_spearman": float(np.mean(ref_grp)),
                                        "pair_spearman_mean": float(np.mean(ref_pair)), "pair_spearman_p05": float(np.quantile(ref_pair, 0.05))}
    # per seed stability
    ps = {}
    for seed, g in t.groupby("split_seed"):
        gg = g.groupby("pair").agg(pv=("pv_hbk", "mean"), se=("se_hbk", "mean"), sp=("se_pwl", "mean"),
                                   **({"pe": ("polerr", "mean")} if "polerr" in crits else {}))
        ps[str(seed)] = {"row_spearman_pv_hbk~se_hbk": float(stats.spearmanr(g.pv_hbk, g.se_hbk)[0]),
                         "pair_spearman_pv_hbk~se_hbk": float(stats.spearmanr(gg.pv, gg.se)[0]),
                         "calib_ratio_hbk": float(g.se_hbk.mean() / g.pv_hbk.mean()),
                         **({"pair_spearman_polerr~se_pwl": float(stats.spearmanr(gg.pe, gg.sp)[0]),
                             "calib_ratio_polerr": float(g.se_pwl.mean() / g.polerr.mean())} if "polerr" in crits else {})}
    s["per_seed"] = ps
    s["ml_kernels"] = t.drop_duplicates(["split_seed", "fold", "inner_fold"]).ml_kernel.value_counts().to_dict()
    return s


# ------------------------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--designs", nargs="+", default=["BP", "BPG"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of the dev seeds (default: all five)")
    ap.add_argument("--max-folds", type=int, default=None, help="smoke tests only")
    ap.add_argument("--no-polerr", action="store_true")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args(argv)
    seeds = list(args.seeds or paths.DEV_SEEDS)
    if not set(seeds) <= set(paths.DEV_SEEDS):
        raise SystemExit("only the development seeds may be used")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    bench = D.load_bench()
    crits = ("pv_hbk", "pv_dq") + (() if args.no_polerr else ("polerr",))
    t0 = time.time()
    summary = {"what": "posterior contrast variance (HBK@ML, observable) and POL_ERR's expected PWL(R) error vs the realised "
                       "inner-validation squared error of strict-adjacent log SF; k = 3, nested DOPT_Q reads; inner blocks "
                       "of the outer training folds only",
               "criterion": "pv_hbk = d' (C - C Dm' (Dm C Dm')^-1 Dm C) d, C = HBKModel.cov_full(S, tau2, l, sig2) (observable, "
                            "reading noise included) of the ML kernel fitted on the block's inner-training cells, Dm = gen20 star-support "
                            "pair matrix of the 4 DOPT_Q reads, d = e_A - e_B; its posterior mean equals r3.inner_preds.hbk_predict (c = 1) "
                            "to 1e-8 on every cell. pv_dq = DOPT_Q's own latent criterion (nested G14 cov, rank-1 updates, pair noise 2 s2).",
               "k": K, "min_group_rows": MIN_GROUP, "coverage": {}, "designs": {}, "inner_cache_check": {}}
    for design in args.designs:
        rows = []
        folds = folds_for(bench.frame, design, seeds=seeds)
        if args.max_folds:
            folds = folds[:args.max_folds]
        checks = []
        nblk = 0
        for f in folds:
            t1 = time.time()
            red = reduce_record(bench, C.load(f.design, f.seed, f.fold))
            if "outer" in red or "cells" in red:
                raise AssertionError("outer held-out objects survived the reduction")
            fr, info = fold_rows(bench, red, with_polerr=not args.no_polerr)
            nblk += len(red["inner"])
            ft = pd.DataFrame(fr)
            chk = inner_cache_check(ft, design, f.seed, f.fold, info)
            if chk is not None:
                checks.append({"seed": f.seed, "fold": f.fold, **chk})
            rows += fr
            del red
            print(f"  {design} {f.seed} fold {f.fold}: {len(fr)} rows, kernels "
                  f"{[info[j]['ml_kernel'] for j in sorted(info)]} ({time.time() - t1:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
        t = pd.DataFrame(rows)
        t.to_parquet(out / f"rows_{design}.parquet", index=False)
        gt = group_table(t, crits)
        gt.to_csv(out / f"summary_{design}.csv", index=False, float_format="%.6g")
        summary["designs"][design] = summarise(t, gt, crits)
        summary["coverage"][design] = {"seeds": seeds, "n_outer_folds": len(folds),
                                       "folds": [[int(f.seed), int(f.fold)] for f in folds], "n_inner_blocks": nblk,
                                       "n_rows": int(len(t)), "n_cells": summary["designs"][design]["n_cells"]}
        if checks:
            summary["inner_cache_check"][design] = {
                "n_folds_checked": len(checks), "unmatched_rows": int(sum(c["unmatched"] for c in checks)),
                "max_abs_diff_hbk": float(max(c["max_abs_diff_hbk"] for c in checks)),
                "max_abs_diff_pwl": float(max(c["max_abs_diff_pwl"] for c in checks)),
                "ml_kernel_same_all": bool(all(c["ml_kernel_same"] for c in checks))}
        summary["runtime_s"] = round(time.time() - t0, 1)
        with open(out / "summary.json", "w", encoding="utf-8") as fh:        # rewritten after every design
            json.dump(summary, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({d: {"row": v["row_level"]["pv_hbk~se_hbk"], "calib": v["calibration"]["hbk"]["ratio"]}
                      for d, v in summary["designs"].items()}, indent=1))


if __name__ == "__main__":
    main()
