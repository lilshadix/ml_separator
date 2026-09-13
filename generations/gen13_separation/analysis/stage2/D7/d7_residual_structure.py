"""D7 - what is left after the smooth radius trend.

Fits a smooth (quadratic / cubic) function of the standardised Shannon CN8 radius to
every well-measured cell's centred lanthanide curve, then asks whether the residual
has reproducible non-smooth structure (Gd break, Jorgensen tetrads, element anomalies)
and what that structure is worth in extractant-macro pairwise MAE.

Run from the repo root:
    .venv/Scripts/python.exe gen13_separation/analysis/stage2/D7/d7_residual_structure.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "gen13_separation")
from gen13sep.metals import (  # noqa: E402
    ATOMIC_NUMBER,
    F_COUNT,
    LANTHANIDES,
    SHANNON_RADIUS_CN8,
    jorgensen_e1,
    jorgensen_e3,
)

OUT = "gen13_separation/analysis/stage2/D7"
FIG = "gen13_separation/figures/stage2"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
RNG = np.random.default_rng(20260908)

MIN_METALS = 6          # cells used to define the residual
N_BOOT = 2000
N_SPLIT = 500
N_PERM = 1000

L = list(LANTHANIDES)
NL = len(L)
IDX = {m: i for i, m in enumerate(L)}


# ----------------------------------------------------------------------------- basis
def _std(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


R8 = np.array([SHANNON_RADIUS_CN8[m] for m in L])
Q = np.array([F_COUNT[m] for m in L], float)
Z = _std(R8)                       # standardised Shannon CN8 radius, mean 0 sd 1
Z2 = Z ** 2
Z3 = Z ** 3
X_QUAD = np.column_stack([np.ones(NL), Z, Z2])
X_CUB = np.column_stack([np.ones(NL), Z, Z2, Z3])


def _centre(v):
    v = np.asarray(v, float)
    return v - v.mean()


SHAPES = {
    "gd_break": _centre(_std((Q >= 7).astype(float))),
    "tetrad_e1": _centre(_std(jorgensen_e1(Q))),
    "tetrad_e3": _centre(_std(jorgensen_e3(Q))),
    "eu_anomaly": _centre(_std((np.array(L) == "Eu").astype(float))),
    "ce_anomaly": _centre(_std((np.array(L) == "Ce").astype(float))),
    "yb_anomaly": _centre(_std((np.array(L) == "Yb").astype(float))),
}


# ------------------------------------------------------------------------- residuals
def build_residuals(df, min_metals=MIN_METALS, design=X_QUAD):
    """Return (meta DataFrame, residual matrix n_cells x 14 with NaN, obs mask)."""
    keep = df[df["n_metals"] >= min_metals].reset_index(drop=True)
    yv = keep[[f"logD__{m}" for m in L]].to_numpy(float)
    obs = ~np.isnan(yv)
    resid = np.full_like(yv, np.nan)
    hat_shrink = np.full(len(keep), np.nan)
    for i in range(len(keep)):
        m = obs[i]
        if m.sum() < design.shape[1] + 1:
            obs[i, :] = False
            continue
        Xo = design[m]
        yo = yv[i, m]
        yo = yo - yo.mean()
        beta, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
        r = yo - Xo @ beta
        resid[i, m] = r
        H = Xo @ np.linalg.pinv(Xo)
        hat_shrink[i] = np.trace(np.eye(m.sum()) - H) / m.sum()
    keep = keep.copy()
    keep["hat_shrink"] = hat_shrink
    resid[~obs] = np.nan
    return keep, resid, obs


def project_shape(shape, obs, design=X_QUAD):
    """Per-cell projection of a 14-vector shape into each cell's residual space."""
    out = np.full((obs.shape[0], NL), np.nan)
    for i in range(obs.shape[0]):
        m = obs[i]
        if m.sum() == 0:
            continue
        Xo = design[m]
        s = shape[m] - shape[m].mean()
        beta, *_ = np.linalg.lstsq(Xo, s, rcond=None)
        out[i, m] = s - Xo @ beta
    return out


def col_mean(mat):
    with np.errstate(invalid="ignore"):
        return np.nanmean(mat, axis=0)


# ------------------------------------------------------------------------------ main
def main():
    df = pd.read_parquet("gen13_separation/manifests/cohort_exact.parquet")
    print(f"cohort: {len(df)} cells, {df.extractant.nunique()} extractants, "
          f"{df.chemotype.nunique()} chemotypes, {df.publication_id.nunique()} publications")

    meta, R, OBS = build_residuals(df, MIN_METALS, X_QUAD)
    meta_c, RC, OBSC = build_residuals(df, MIN_METALS, X_CUB)
    n_cells = len(meta)
    print(f"\n[1] cells with >= {MIN_METALS} observed metals: {n_cells} "
          f"({meta.extractant.nunique()} extractants, {meta.chemotype.nunique()} chemotypes, "
          f"{meta.publication_id.nunique()} publications); "
          f"observed cell-metal residuals = {int(OBS.sum())}")

    extr = meta["extractant"].to_numpy()
    chemo = meta["chemotype"].to_numpy()
    pub = meta["publication_id"].to_numpy()

    # ---------------------------------------------------------------- cluster bootstrap
    uex = np.unique(extr)
    ex_rows = {e: np.where(extr == e)[0] for e in uex}

    def boot_indices():
        pick = RNG.choice(len(uex), size=len(uex), replace=True)
        return np.concatenate([ex_rows[uex[k]] for k in pick])

    boot_mean = np.empty((N_BOOT, NL))
    boot_medabs = np.empty((N_BOOT, NL))
    for b in range(N_BOOT):
        idx = boot_indices()
        sub = R[idx]
        boot_mean[b] = col_mean(sub)
        with np.errstate(invalid="ignore"):
            boot_medabs[b] = np.nanmedian(np.abs(sub), axis=0)

    mean_res = col_mean(R)
    med_abs = np.nanmedian(np.abs(R), axis=0)
    n_per_el = np.sum(~np.isnan(R), axis=0)
    se = np.nanstd(boot_mean, axis=0, ddof=1)
    zstat = mean_res / se
    from scipy import stats
    pval = 2 * stats.norm.sf(np.abs(zstat))
    # Holm-Bonferroni over the 14 elements
    order = np.argsort(pval)
    holm = np.empty(NL)
    running = 0.0
    for rank, j in enumerate(order):
        adj = min(1.0, (NL - rank) * pval[j])
        running = max(running, adj)
        holm[j] = running

    mean_res_cub = col_mean(RC)
    med_abs_cub = np.nanmedian(np.abs(RC), axis=0)

    # replicate-noise reference per element (median repsd where measured)
    reps = df[[f"repsd__{m}" for m in L]].to_numpy(float)
    with np.errstate(invalid="ignore"):
        rep_med = np.nanmedian(reps, axis=0)

    t1 = pd.DataFrame({
        "element": L,
        "Z": [ATOMIC_NUMBER[m] for m in L],
        "n_cells": n_per_el,
        "med_abs_resid_quad": med_abs,
        "med_abs_resid_quad_lo": np.nanpercentile(boot_medabs, 2.5, axis=0),
        "med_abs_resid_quad_hi": np.nanpercentile(boot_medabs, 97.5, axis=0),
        "med_abs_resid_cubic": med_abs_cub,
        "mean_resid_quad": mean_res,
        "mean_resid_lo": np.nanpercentile(boot_mean, 2.5, axis=0),
        "mean_resid_hi": np.nanpercentile(boot_mean, 97.5, axis=0),
        "mean_resid_se_clusterboot": se,
        "z_stat": zstat,
        "p_raw": pval,
        "p_holm": holm,
        "mean_resid_cubic": mean_res_cub,
        "median_within_replicate_sd": rep_med,
    })
    t1.to_csv(f"{OUT}/D7_table1_per_element_residual.csv", index=False)
    print("\n[1] per-element residual after the per-cell quadratic in standardised r(CN8):")
    print(t1[["element", "n_cells", "med_abs_resid_quad", "mean_resid_quad",
              "mean_resid_se_clusterboot", "p_holm", "mean_resid_cubic"]].to_string(
        index=False, float_format=lambda v: f"{v: .4f}"))
    sig = t1[t1.p_holm < 0.05]["element"].tolist()
    print(f"    Holm-significant elements (alpha=0.05, 14 tests): {sig}")
    print(f"    pooled median |resid| = {np.nanmedian(np.abs(R)):.4f}  "
          f"(cubic {np.nanmedian(np.abs(RC)):.4f}); "
          f"rms of mean residual curve = {np.sqrt(np.nanmean(mean_res**2)):.4f}")

    # -------------------------------------------------------- [2] candidate shapes
    proj = {k: project_shape(v, OBS) for k, v in SHAPES.items()}
    rows = []
    flat_r = R[OBS]
    tot_ss = np.sum(flat_r ** 2)
    for k, P in proj.items():
        p_mean = col_mean(P)
        ok = np.isfinite(p_mean) & np.isfinite(mean_res)
        r_curve = np.corrcoef(mean_res[ok], p_mean[ok])[0, 1]
        # pooled cell-level regression of residual on the cell-projected shape
        x = P[OBS]
        good = np.isfinite(x) & np.isfinite(flat_r)
        beta = np.sum(x[good] * flat_r[good]) / np.sum(x[good] ** 2)
        ss_exp = beta ** 2 * np.sum(x[good] ** 2)
        # variance of the *mean* residual curve explained
        b2 = np.sum(p_mean[ok] * mean_res[ok]) / np.sum(p_mean[ok] ** 2)
        r2_curve = 1 - np.sum((mean_res[ok] - b2 * p_mean[ok]) ** 2) / np.sum(mean_res[ok] ** 2)
        # cluster-bootstrap CI on the curve correlation
        bc = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = boot_indices()
            mm = col_mean(R[idx])
            pm = col_mean(P[idx])
            o = np.isfinite(mm) & np.isfinite(pm)
            bc[b] = np.corrcoef(mm[o], pm[o])[0, 1] if o.sum() > 3 else np.nan
        rows.append(dict(shape=k, corr_with_mean_resid_curve=r_curve,
                         corr_lo=np.nanpercentile(bc, 2.5), corr_hi=np.nanpercentile(bc, 97.5),
                         beta_pooled=beta,
                         frac_pooled_resid_var_explained=ss_exp / tot_ss,
                         frac_mean_curve_var_explained=r2_curve))
    t2 = pd.DataFrame(rows).sort_values("frac_mean_curve_var_explained", ascending=False)
    # joint model of all three physics shapes
    Xj = np.column_stack([proj[k][OBS] for k in ("gd_break", "tetrad_e1", "tetrad_e3")])
    gj = np.all(np.isfinite(Xj), axis=1) & np.isfinite(flat_r)
    bj, *_ = np.linalg.lstsq(Xj[gj], flat_r[gj], rcond=None)
    ss_j = np.sum((Xj[gj] @ bj) ** 2)
    t2.to_csv(f"{OUT}/D7_table2_shape_match.csv", index=False)
    print("\n[2] match of the mean residual curve to candidate non-smooth shapes")
    print(t2.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print(f"    joint gd_break+tetrad_e1+tetrad_e3 explains "
          f"{ss_j / tot_ss:.4f} of the pooled residual variance")

    # --------------------------------------------------- [3] split-half reproducibility
    def split_half(labels, n_rep=N_SPLIT, permute=False):
        u = np.unique(labels)
        out = []
        for _ in range(n_rep):
            Rw = R
            if permute:  # null: shuffle residuals among observed elements within a cell
                Rw = R.copy()
                for i in range(Rw.shape[0]):
                    m = OBS[i]
                    v = Rw[i, m]
                    Rw[i, m] = RNG.permutation(v)
            perm = RNG.permutation(len(u))
            g1 = set(u[perm[: len(u) // 2]])
            m1 = np.array([lab in g1 for lab in labels])
            a, b = col_mean(Rw[m1]), col_mean(Rw[~m1])
            na = np.sum(~np.isnan(Rw[m1]), axis=0)
            nb = np.sum(~np.isnan(Rw[~m1]), axis=0)
            ok = (na >= 3) & (nb >= 3) & np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 8:
                out.append(np.corrcoef(a[ok], b[ok])[0, 1])
        return np.array(out)

    rows = []
    for name, lab in (("extractant", extr), ("chemotype", chemo), ("publication", pub)):
        c = split_half(lab)
        cn = split_half(lab, n_rep=200, permute=True)
        # Spearman-Brown step-up to the full-sample reliability
        sb = 2 * np.median(c) / (1 + np.median(c))
        rows.append(dict(split_by=name, n_groups=int(np.unique(lab).size),
                         n_splits=len(c), median_split_half_r=float(np.median(c)),
                         lo_2p5=float(np.percentile(c, 2.5)),
                         hi_97p5=float(np.percentile(c, 97.5)),
                         frac_splits_positive=float(np.mean(c > 0)),
                         spearman_brown_reliability=float(sb),
                         null_median_r=float(np.median(cn)),
                         null_p95_r=float(np.percentile(cn, 95))))
        print(f"\n[3] split-half by {name}: median r = {np.median(c):.3f} "
              f"[{np.percentile(c,2.5):.3f}, {np.percentile(c,97.5):.3f}] over {len(c)} splits; "
              f"within-cell-permutation null median {np.median(cn):.3f} "
              f"(p95 {np.percentile(cn,95):.3f}); Spearman-Brown {sb:.3f}")
    t3 = pd.DataFrame(rows)
    t3.to_csv(f"{OUT}/D7_table3_split_half.csv", index=False)

    # ---------------------------------------------------- [4] chemotype dependence
    counts = pd.Series(chemo).value_counts()
    top6 = counts.index[:6].tolist()
    rows = []
    for ct in top6:
        m = chemo == ct
        cm = col_mean(R[m])
        nn = np.sum(~np.isnan(R[m]), axis=0)
        # cluster bootstrap within this chemotype (over its extractants)
        ue = np.unique(extr[m])
        rr = np.where(m)[0]
        er = {e: rr[extr[m] == e] for e in ue}
        bb = np.empty((500, NL))
        for b in range(500):
            pk = RNG.choice(len(ue), size=len(ue), replace=True)
            bb[b] = col_mean(R[np.concatenate([er[ue[k]] for k in pk])])
        for j, el in enumerate(L):
            rows.append(dict(chemotype=ct, n_cells=int(m.sum()),
                             n_extractants=int(len(ue)), element=el,
                             n_cells_el=int(nn[j]), mean_resid=cm[j],
                             lo=np.nanpercentile(bb[:, j], 2.5),
                             hi=np.nanpercentile(bb[:, j], 97.5)))
    t4 = pd.DataFrame(rows)
    t4.to_csv(f"{OUT}/D7_table4_chemotype_curves.csv", index=False)

    # heterogeneity test: permute chemotype labels at the extractant level
    def hetero_stat(chem_lab):
        M = []
        w = []
        for ct in top6:
            m = chem_lab == ct
            M.append(col_mean(R[m]))
            w.append(np.sum(~np.isnan(R[m]), axis=0))
        M = np.array(M)
        w = np.array(w, float)
        ok = np.all(w >= 3, axis=0) & np.all(np.isfinite(M), axis=0)
        if ok.sum() == 0:
            return np.nan
        grand = np.average(M[:, ok], axis=0, weights=w[:, ok])
        return float(np.nansum(w[:, ok] * (M[:, ok] - grand) ** 2) / w[:, ok].sum())

    obs_stat = hetero_stat(chemo)
    ex_to_chem = pd.Series(chemo, index=extr).groupby(level=0).first()
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        shuffled = ex_to_chem.copy()
        shuffled[:] = RNG.permutation(shuffled.values)
        null[b] = hetero_stat(np.array([shuffled[e] for e in extr]))
    p_het = float((np.nansum(null >= obs_stat) + 1) / (np.sum(np.isfinite(null)) + 1))
    print(f"\n[4] top-6 chemotypes {top6} (cells {[int(counts[c]) for c in top6]})")
    piv = t4.pivot(index="element", columns="chemotype", values="mean_resid").reindex(L)
    print(piv.to_string(float_format=lambda v: f"{v: .3f}"))
    print(f"    between-chemotype heterogeneity of the residual curve: "
          f"stat = {obs_stat:.5f}, extractant-level permutation p = {p_het:.4f}")
    cc = piv.corr()
    cc.to_csv(f"{OUT}/D7_table4b_chemotype_curve_correlations.csv")
    off = cc.to_numpy()[np.triu_indices(len(top6), 1)]
    print(f"    pairwise correlation between chemotype residual curves: "
          f"median {np.nanmedian(off):.3f}, range [{np.nanmin(off):.3f}, {np.nanmax(off):.3f}]")
    pd.DataFrame([dict(stat=obs_stat, perm_p=p_het, n_perm=N_PERM,
                       median_pairwise_curve_corr=float(np.nanmedian(off)))]).to_csv(
        f"{OUT}/D7_table4c_heterogeneity_test.csv", index=False)

    # donor set: DENTATE / coreCN
    rows = []
    for col in ("recipe__DENTATE", "recipe__coreCN"):
        vals = meta[col].to_numpy()
        for v in pd.unique(vals):
            m = vals == v
            if m.sum() < 15:
                continue
            cm = col_mean(R[m])
            rows.append(dict(field=col, value=str(v), n_cells=int(m.sum()),
                             **{f"resid__{e}": cm[j] for j, e in enumerate(L)}))
    pd.DataFrame(rows).to_csv(f"{OUT}/D7_table4d_donorset_curves.csv", index=False)

    # ------------------------------------------------------------- [5] the prize
    prize_rows = []

    # (a) oracle-smooth baseline on the residual-defined cells
    def loo_curve(group_labels):
        """Leave-one-group-out mean residual curve, per cell."""
        out = np.full((n_cells, NL), 0.0)
        tot = np.nansum(np.where(np.isnan(R), 0, R), axis=0)
        cnt = np.sum(~np.isnan(R), axis=0)
        for g in np.unique(group_labels):
            m = group_labels == g
            sub = R[m]
            t = tot - np.nansum(np.where(np.isnan(sub), 0, sub), axis=0)
            c = cnt - np.sum(~np.isnan(sub), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[m] = np.where(c > 0, t / np.maximum(c, 1), 0.0)
        return out

    yv = meta[[f"logD__{m}" for m in L]].to_numpy(float)
    loo_ex = loo_curve(extr)
    loo_ch = loo_curve(chemo)

    def pair_mae(curves_pred, cells_mask=None):
        """extractant-macro and pooled MAE of pairwise log SF from per-cell curves."""
        per_ex = {}
        pooled = []
        for i in range(n_cells):
            m = OBS[i]
            if m.sum() < 2:
                continue
            ids = np.where(m)[0]
            yo = yv[i, ids]
            yo = yo - yo.mean()
            pr = curves_pred[i, ids]
            errs = []
            for a, b in itertools.combinations(range(len(ids)), 2):
                errs.append(abs((yo[a] - yo[b]) - (pr[a] - pr[b])))
            per_ex.setdefault(extr[i], []).extend(errs)
            pooled.extend(errs)
        macro = float(np.mean([np.mean(v) for v in per_ex.values()]))
        return macro, float(np.mean(pooled)), len(per_ex)

    smooth = np.full((n_cells, NL), np.nan)
    for i in range(n_cells):
        m = OBS[i]
        if m.sum() == 0:
            continue
        Xo = X_QUAD[m]
        yo = yv[i, m] - yv[i, m].mean()
        beta, *_ = np.linalg.lstsq(Xo, yo, rcond=None)
        smooth[i, m] = Xo @ beta

    base_macro, base_pool, n_ex_used = pair_mae(np.nan_to_num(smooth))
    variants = {
        "smooth_only": np.zeros((n_cells, NL)),
        "plus_mean_resid_insample": np.tile(mean_res, (n_cells, 1)),
        "plus_mean_resid_LOEO_extractant": loo_ex,
        "plus_mean_resid_LOCO_chemotype": loo_ch,
        "plus_own_residual_oracle": np.nan_to_num(R),
    }
    for name, add in variants.items():
        pred = np.nan_to_num(smooth) + np.nan_to_num(add)
        mac, pool, _ = pair_mae(pred)
        prize_rows.append(dict(setting="oracle_smooth_per_cell", variant=name,
                               extractant_macro_mae=mac, pooled_mae=pool,
                               delta_vs_smooth_only=mac - base_macro,
                               n_extractants=n_ex_used, n_cells=n_cells))
        print(f"[5a] {name:34s} macro MAE {mac:.4f}  pooled {pool:.4f}  "
              f"delta {mac - base_macro:+.4f}")

    # (b) real arms on the primary design: add the held-out mean residual curve
    arms = ["C_DIRECT_ROW", "M_SELECTED", "M_PHYSICS_radius+radius_sq",
            "M_PHYSICS_radius+radius_sq+gd_break",
            "M_PHYSICS_radius+radius_sq+tetrad_e1+tetrad_e3",
            "M_LOWRANK_K2", "X_ENS_DIRECT+LOWRANK_K2", "X_ENS_DIRECT+PHYSICS",
            "B1_MEAN_CURVE"]
    # chemotype-held-out mean residual curve, keyed by chemotype
    chem_curve = {}
    tot = np.nansum(np.where(np.isnan(R), 0, R), axis=0)
    cnt = np.sum(~np.isnan(R), axis=0)
    for g in np.unique(chemo):
        m = chemo == g
        sub = R[m]
        t = tot - np.nansum(np.where(np.isnan(sub), 0, sub), axis=0)
        c = cnt - np.sum(~np.isnan(sub), axis=0)
        chem_curve[g] = np.where(c > 0, t / np.maximum(c, 1), 0.0)
    global_curve = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)

    arm_rows = []
    err_curves = {}
    for arm in arms:
        p = f"gen13_separation/predictions/B_primary/{arm}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        cur = np.array([chem_curve.get(c, global_curve) for c in d["chemotype"]])
        ia = np.array([IDX[a] for a in d["A"]])
        ib = np.array([IDX[b] for b in d["B"]])
        add = cur[np.arange(len(d)), ia] - cur[np.arange(len(d)), ib]
        d = d.assign(_add=add)

        def macro(pred_col):
            g = d.groupby(["split_seed", "extractant"], observed=True).apply(
                lambda t: np.mean(np.abs(t["y"] - t[pred_col])), include_groups=False)
            return float(g.groupby(level=0).mean().mean())

        d["_p0"] = d["prediction"]
        d["_p1"] = d["prediction"] + d["_add"]
        m0, m1 = macro("_p0"), macro("_p1")
        # optimal scalar on the residual-curve correction (fitted on all held-out rows -> optimistic)
        res = d["y"] - d["prediction"]
        lam = float(np.sum(res * d["_add"]) / np.sum(d["_add"] ** 2))
        d["_p2"] = d["prediction"] + lam * d["_add"]
        m2 = macro("_p2")
        arm_rows.append(dict(arm=arm, macro_mae=m0, macro_mae_plus_resid_curve=m1,
                             delta_lambda1=m1 - m0, lambda_opt=lam,
                             macro_mae_plus_lambda_opt=m2, delta_lambda_opt=m2 - m0,
                             pooled_mae=float(np.mean(np.abs(d["y"] - d["_p0"])))))
        print(f"[5b] {arm:46s} macro {m0:.4f} -> {m1:.4f} (d={m1-m0:+.4f}); "
              f"lambda_opt={lam:+.3f} -> {m2:.4f} (d={m2-m0:+.4f})")

        # arm error curve per element (least squares over that cell's pairs)
        sd = d[d["split_seed"] == d["split_seed"].iloc[0]]
        ec = {}
        for cid, t in sd.groupby("cell_id", observed=True):
            mets = sorted(set(t["A"]) | set(t["B"]), key=lambda m: IDX[m])
            if len(mets) < MIN_METALS:
                continue
            pos = {m: j for j, m in enumerate(mets)}
            D = np.zeros((len(t), len(mets)))
            D[np.arange(len(t)), [pos[a] for a in t["A"]]] = 1
            D[np.arange(len(t)), [pos[b] for b in t["B"]]] = -1
            e = (t["y"] - t["prediction"]).to_numpy()
            f = np.linalg.pinv(np.vstack([D, np.ones(len(mets))])) @ np.concatenate([e, [0.0]])
            ec[cid] = {m: f[pos[m]] for m in mets}
        M = np.full((len(ec), NL), np.nan)
        for i, (cid, dd) in enumerate(ec.items()):
            for m, v in dd.items():
                M[i, IDX[m]] = v
        err_curves[arm] = col_mean(M)
    t5b = pd.DataFrame(arm_rows)
    t5b.to_csv(f"{OUT}/D7_table5b_arm_prize.csv", index=False)
    pd.DataFrame(prize_rows).to_csv(f"{OUT}/D7_table5a_oracle_prize.csv", index=False)

    ec_df = pd.DataFrame(err_curves, index=L)
    ec_df["observed_mean_resid"] = mean_res
    ec_df.to_csv(f"{OUT}/D7_table5c_arm_error_curves.csv")
    print("\n[5c] mean held-out error curve of each arm (y - pred, per element) vs the "
          "observed non-smooth residual")
    print(ec_df.to_string(float_format=lambda v: f"{v: .3f}"))
    for arm, v in err_curves.items():
        ok = np.isfinite(v) & np.isfinite(mean_res)
        print(f"     corr(mean error curve, observed mean residual) {arm:46s} "
              f"{np.corrcoef(v[ok], mean_res[ok])[0,1]:+.3f}   rms error curve {np.sqrt(np.mean(v[ok]**2)):.4f}")

    # ------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(NL)
    ax[0].axhline(0, color="0.7", lw=0.8)
    ax[0].errorbar(x, mean_res,
                   yerr=[mean_res - np.nanpercentile(boot_mean, 2.5, axis=0),
                         np.nanpercentile(boot_mean, 97.5, axis=0) - mean_res],
                   fmt="o-", capsize=3, color="tab:blue", label="mean residual (quadratic)")
    ax[0].plot(x, mean_res_cub, "s--", color="tab:orange", ms=4, label="mean residual (cubic)")
    ax[0].axvline(IDX["Gd"] - 0.5, color="0.5", ls=":", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(L, rotation=45)
    ax[0].set_ylabel("mean residual (log units)")
    ax[0].set_title(f"Residual after per-cell smooth r(CN8) fit\n({n_cells} cells, "
                    f">= {MIN_METALS} metals)")
    ax[0].legend(fontsize=8)

    for k in ("gd_break", "tetrad_e1", "tetrad_e3", "eu_anomaly"):
        pm = col_mean(proj[k])
        ok = np.isfinite(pm)
        s = np.sum(pm[ok] * mean_res[ok]) / np.sum(pm[ok] ** 2)
        ax[1].plot(x[ok], s * pm[ok], "-", lw=1.4, label=f"{k} (scaled)")
    ax[1].plot(x, mean_res, "ko-", ms=4, label="observed")
    ax[1].axhline(0, color="0.7", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(L, rotation=45)
    ax[1].set_title("Candidate shapes, projected into residual space")
    ax[1].legend(fontsize=7)

    for ct in top6:
        m = chemo == ct
        ax[2].plot(x, col_mean(R[m]), "-o", ms=3, lw=1.1, label=f"{ct} (n={int(m.sum())})")
    ax[2].axhline(0, color="0.7", lw=0.8)
    ax[2].set_xticks(x); ax[2].set_xticklabels(L, rotation=45)
    ax[2].set_title("Mean residual curve by chemotype")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/D7_residual_structure.png", dpi=150)
    print(f"\nwrote {FIG}/D7_residual_structure.png")

    json.dump(dict(n_cells=n_cells, n_extractants=int(meta.extractant.nunique()),
                   n_chemotypes=int(meta.chemotype.nunique()),
                   n_observations=int(OBS.sum()),
                   pooled_median_abs_resid=float(np.nanmedian(np.abs(R))),
                   rms_mean_resid_curve=float(np.sqrt(np.nanmean(mean_res ** 2))),
                   heterogeneity_perm_p=p_het),
              open(f"{OUT}/D7_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
