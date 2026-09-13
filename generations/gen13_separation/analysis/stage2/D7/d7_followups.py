"""D7 follow-ups: noise floor, extra element shapes, reproducible amplitude, honest prize.

Run from the repo root:
    .venv/Scripts/python.exe generations/gen13_separation/analysis/stage2/D7/d7_followups.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "generations/gen13_separation")
sys.path.insert(0, "generations/gen13_separation/analysis/stage2/D7")
from d7_residual_structure import (  # noqa: E402
    IDX, L, NL, OUT, Q, SHAPES, X_CUB, X_QUAD, Z, build_residuals, col_mean,
    project_shape, _centre, _std,
)

RNG = np.random.default_rng(11)
MIN_METALS = 6
LAN = np.array(L)


def main():
    df = pd.read_parquet("generations/gen13_separation/manifests/cohort_exact.parquet")
    meta, R, OBS = build_residuals(df, MIN_METALS, X_QUAD)
    meta_c, RC, OBSC = build_residuals(df, MIN_METALS, X_CUB)
    n_cells = len(meta)
    extr = meta["extractant"].to_numpy()
    chemo = meta["chemotype"].to_numpy()
    yv = meta[[f"logD__{m}" for m in L]].to_numpy(float)
    mean_res = col_mean(R)

    # ---------------------------------------------- A. how much is left at all
    rows = []
    for tag, design, Rm in (("linear", X_QUAD[:, :2], None),
                            ("quadratic", X_QUAD, R), ("cubic", X_CUB, RC)):
        if Rm is None:
            _, Rm, _ = build_residuals(df, MIN_METALS, design)
        ss_res, ss_tot, pcr2 = 0.0, 0.0, []
        for i in range(n_cells):
            m = OBS[i]
            if m.sum() == 0:
                continue
            yo = yv[i, m] - yv[i, m].mean()
            r = Rm[i, m]
            ss_res += np.sum(r ** 2)
            ss_tot += np.sum(yo ** 2)
            if np.sum(yo ** 2) > 0:
                pcr2.append(1 - np.sum(r ** 2) / np.sum(yo ** 2))
        rows.append(dict(model=tag, n_params=design.shape[1],
                         pooled_R2=1 - ss_res / ss_tot,
                         median_per_cell_R2=float(np.median(pcr2)),
                         rms_resid=float(np.sqrt(ss_res / OBS.sum())),
                         median_abs_resid=float(np.nanmedian(np.abs(Rm)))))
    tA = pd.DataFrame(rows)

    # noise floor: sd of the cell-level logD from replicates, where measurable
    rep = meta[[f"repsd__{m}" for m in L]].to_numpy(float)
    nrep = meta[[f"nrep__{m}" for m in L]].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = rep / np.sqrt(np.maximum(nrep, 1))
    have = np.isfinite(sem) & (nrep > 1) & OBS
    tA["median_repsd_of_cell_mean"] = float(np.nanmedian(sem[have]))
    tA["n_repl_measured_cell_metal"] = int(have.sum())
    tA.to_csv(f"{OUT}/D7_table0_smoothness.csv", index=False)
    print("[A] how much of the centred curve the smooth radius model already explains")
    print(tA.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print(f"    replicate SEM of a cell-metal logD (median over {int(have.sum())} "
          f"replicated cell-metals) = {np.nanmedian(sem[have]):.4f}; "
          f"raw replicate sd median = {np.nanmedian(rep[np.isfinite(rep)&(nrep>1)]):.4f}")

    # ------------------------------------- B. reproducible amplitude of the mean curve
    # split-half covariance is an unbiased estimate of the common-signal variance
    uex = np.unique(extr)
    covs, r_halves = [], []
    for _ in range(2000):
        perm = RNG.permutation(len(uex))
        g1 = set(uex[perm[: len(uex) // 2]])
        m1 = np.array([e in g1 for e in extr])
        a, b = col_mean(R[m1]), col_mean(R[~m1])
        na = np.sum(~np.isnan(R[m1]), axis=0)
        nb = np.sum(~np.isnan(R[~m1]), axis=0)
        ok = (na >= 3) & (nb >= 3) & np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 8:
            covs.append(np.mean((a[ok] - a[ok].mean()) * (b[ok] - b[ok].mean())))
            r_halves.append(np.corrcoef(a[ok], b[ok])[0, 1])
    covs = np.array(covs)
    amp = np.sqrt(np.maximum(covs, 0))
    # total residual variance (pooled) and the share carried by the common curve
    tot_var = float(np.nanmean(R[OBS] ** 2))
    common_var = float(np.median(covs))
    tB = pd.DataFrame([dict(
        rms_of_observed_mean_resid_curve=float(np.sqrt(np.nanmean(mean_res ** 2))),
        reproducible_rms_amplitude=float(np.median(amp)),
        reproducible_rms_lo=float(np.percentile(amp, 2.5)),
        reproducible_rms_hi=float(np.percentile(amp, 97.5)),
        median_split_half_r=float(np.median(r_halves)),
        pooled_residual_variance=tot_var,
        common_curve_variance=common_var,
        share_of_residual_variance_that_is_a_common_curve=common_var / tot_var)])
    tB.to_csv(f"{OUT}/D7_table3b_reproducible_amplitude.csv", index=False)
    print("\n[B] reproducible (split-half debiased) amplitude of the common residual curve")
    print(tB.T.to_string(float_format=lambda v: f"{v: .5f}"))

    # --------------------------------------------- C. wider shape catalogue
    shapes = dict(SHAPES)
    for el in L:
        shapes[f"single_{el}"] = _centre(_std((LAN == el).astype(float)))
    shapes["gd_dip"] = _centre(_std(-(LAN == "Gd").astype(float)))
    shapes["quarter_shell_4"] = _centre(_std(np.cos(2 * np.pi * (Q + 0.5) / 3.5)))
    proj = {k: project_shape(v, OBS) for k, v in shapes.items()}
    flat_r = R[OBS]
    tot_ss = float(np.sum(flat_r ** 2))
    rows = []
    for k, P in proj.items():
        pm = col_mean(P)
        ok = np.isfinite(pm) & np.isfinite(mean_res)
        x = P[OBS]
        g = np.isfinite(x)
        beta = float(np.sum(x[g] * flat_r[g]) / np.sum(x[g] ** 2))
        # cluster-bootstrap t on beta
        bb = []
        ex_rows = {e: np.where(extr == e)[0] for e in uex}
        for _ in range(500):
            pick = RNG.choice(len(uex), size=len(uex), replace=True)
            idx = np.concatenate([ex_rows[uex[k2]] for k2 in pick])
            xr, rr = P[idx], R[idx]
            gg = np.isfinite(xr) & np.isfinite(rr)
            bb.append(np.sum(xr[gg] * rr[gg]) / np.sum(xr[gg] ** 2))
        bb = np.array(bb)
        b2 = np.sum(pm[ok] * mean_res[ok]) / np.sum(pm[ok] ** 2)
        rows.append(dict(shape=k, corr_with_mean_resid_curve=float(np.corrcoef(mean_res[ok], pm[ok])[0, 1]),
                         beta_pooled=beta, beta_lo=float(np.percentile(bb, 2.5)),
                         beta_hi=float(np.percentile(bb, 97.5)),
                         frac_pooled_resid_var=beta ** 2 * float(np.sum(x[g] ** 2)) / tot_ss,
                         frac_mean_curve_var=float(1 - np.sum((mean_res[ok] - b2 * pm[ok]) ** 2)
                                                   / np.sum(mean_res[ok] ** 2))))
    tC = pd.DataFrame(rows).sort_values("frac_mean_curve_var", ascending=False)
    tC.to_csv(f"{OUT}/D7_table2b_shape_catalogue.csv", index=False)
    print("\n[C] wider shape catalogue (sorted by share of the mean-curve variance)")
    print(tC.head(12).to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    # ------------------------------- D. honest prize: best lambda on extractant-macro MAE
    tot = np.nansum(np.where(np.isnan(R), 0, R), axis=0)
    cnt = np.sum(~np.isnan(R), axis=0)
    global_curve = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)
    chem_curve = {}
    for g in np.unique(chemo):
        m = chemo == g
        sub = R[m]
        t = tot - np.nansum(np.where(np.isnan(sub), 0, sub), axis=0)
        c = cnt - np.sum(~np.isnan(sub), axis=0)
        chem_curve[g] = np.where(c > 0, t / np.maximum(c, 1), 0.0)

    arms = ["C_DIRECT_ROW", "M_SELECTED", "X_ENS_DIRECT+LOWRANK_K2", "B1_MEAN_CURVE"]
    grid = np.linspace(-1.0, 2.0, 31)
    rows = []
    for arm in arms:
        d = pd.read_parquet(f"generations/gen13_separation/predictions/B_primary/{arm}.parquet")
        cur = np.array([chem_curve.get(c, global_curve) for c in d["chemotype"]])
        ia = np.array([IDX[a] for a in d["A"]])
        ib = np.array([IDX[b] for b in d["B"]])
        add = cur[np.arange(len(d)), ia] - cur[np.arange(len(d)), ib]
        res = (d["y"] - d["prediction"]).to_numpy()
        key = pd.MultiIndex.from_arrays([d["split_seed"], d["extractant"]])
        best = None
        for lam in grid:
            e = np.abs(res - lam * add)
            s = pd.Series(e, index=key).groupby(level=[0, 1]).mean()
            mac = float(s.groupby(level=0).mean().mean())
            if best is None or mac < best[1]:
                best = (lam, mac)
        s0 = pd.Series(np.abs(res), index=key).groupby(level=[0, 1]).mean()
        m0 = float(s0.groupby(level=0).mean().mean())
        rows.append(dict(arm=arm, macro_mae=m0, best_lambda=best[0],
                         macro_mae_at_best_lambda=best[1], delta=best[1] - m0,
                         rms_of_correction=float(np.sqrt(np.mean(add ** 2)))))
        print(f"[D] {arm:26s} macro {m0:.4f}; best lambda {best[0]:+.2f} -> {best[1]:.4f} "
              f"(delta {best[1]-m0:+.4f}); rms of pairwise correction {np.sqrt(np.mean(add**2)):.4f}")
    tD = pd.DataFrame(rows)
    tD.to_csv(f"{OUT}/D7_table5d_best_lambda.csv", index=False)

    # ---- E. ceiling: an oracle that knows the true common residual curve, on top of arms
    #        (fit the 14-element offset vector to minimise held-out macro MAE => upper bound)
    d = pd.read_parquet("generations/gen13_separation/predictions/B_primary/X_ENS_DIRECT+LOWRANK_K2.parquet")
    ia = np.array([IDX[a] for a in d["A"]])
    ib = np.array([IDX[b] for b in d["B"]])
    res = (d["y"] - d["prediction"]).to_numpy()
    D = np.zeros((len(d), NL))
    D[np.arange(len(d)), ia] = 1
    D[np.arange(len(d)), ib] = -1
    f, *_ = np.linalg.lstsq(np.vstack([D, np.ones(NL)]),
                            np.concatenate([res, [0.0]]), rcond=None)
    key = pd.MultiIndex.from_arrays([d["split_seed"], d["extractant"]])
    m0 = float(pd.Series(np.abs(res), index=key).groupby(level=[0, 1]).mean()
               .groupby(level=0).mean().mean())
    corr = D @ f
    m1 = float(pd.Series(np.abs(res - corr), index=key).groupby(level=[0, 1]).mean()
               .groupby(level=0).mean().mean())
    # split the fitted offset into the smooth part and the non-smooth part
    beta, *_ = np.linalg.lstsq(X_QUAD, f, rcond=None)
    f_smooth = X_QUAD @ beta
    f_rough = f - f_smooth
    m_sm = float(pd.Series(np.abs(res - D @ f_smooth), index=key).groupby(level=[0, 1]).mean()
                 .groupby(level=0).mean().mean())
    m_ro = float(pd.Series(np.abs(res - D @ f_rough), index=key).groupby(level=[0, 1]).mean()
                 .groupby(level=0).mean().mean())
    tE = pd.DataFrame([dict(arm="X_ENS_DIRECT+LOWRANK_K2", macro_mae=m0,
                            macro_with_best_fit_14el_offset=m1, delta_full=m1 - m0,
                            macro_with_smooth_part_only=m_sm, delta_smooth=m_sm - m0,
                            macro_with_nonsmooth_part_only=m_ro, delta_nonsmooth=m_ro - m0,
                            rms_offset_full=float(np.sqrt(np.mean(f ** 2))),
                            rms_offset_smooth=float(np.sqrt(np.mean(f_smooth ** 2))),
                            rms_offset_nonsmooth=float(np.sqrt(np.mean(f_rough ** 2))))])
    tE.to_csv(f"{OUT}/D7_table5e_offset_ceiling.csv", index=False)
    print("\n[E] in-sample-optimal global 14-element offset on the best arm (an upper bound)")
    print(tE.T.to_string(float_format=lambda v: f"{v: .4f}"))
    print("    fitted offset per element:",
          {e: round(float(v), 3) for e, v in zip(L, f)})
    print("    non-smooth part of it:   ",
          {e: round(float(v), 3) for e, v in zip(L, f_rough)})

    # ---- F. same prize measured on the oracle-smooth baseline, with a grid on lambda
    def pair_mae(add_curve, lam):
        per_ex = {}
        for i in range(n_cells):
            m = OBS[i]
            if m.sum() < 2:
                continue
            ids = np.where(m)[0]
            r = R[i, ids] - lam * add_curve[i, ids]
            errs = [abs(r[a] - r[b]) for a, b in itertools.combinations(range(len(ids)), 2)]
            per_ex.setdefault(extr[i], []).extend(errs)
        return float(np.mean([np.mean(v) for v in per_ex.values()]))

    loo = np.zeros((n_cells, NL))
    ex_rows = {e: np.where(extr == e)[0] for e in uex}
    for e in uex:
        m = extr == e
        sub = R[m]
        t = tot - np.nansum(np.where(np.isnan(sub), 0, sub), axis=0)
        c = cnt - np.sum(~np.isnan(sub), axis=0)
        loo[m] = np.where(c > 0, t / np.maximum(c, 1), 0.0)
    base = pair_mae(loo, 0.0)
    rows = []
    for lam in np.linspace(0, 2, 21):
        rows.append(dict(lam=lam, macro_mae=pair_mae(loo, lam)))
    tF = pd.DataFrame(rows)
    tF["delta"] = tF["macro_mae"] - base
    tF.to_csv(f"{OUT}/D7_table5f_oracle_smooth_lambda.csv", index=False)
    b = tF.loc[tF["macro_mae"].idxmin()]
    print(f"\n[F] oracle-smooth baseline macro MAE {base:.4f}; best lambda on the "
          f"held-out mean residual curve {b['lam']:.2f} -> {b['macro_mae']:.4f} "
          f"(delta {b['delta']:+.4f})")


if __name__ == "__main__":
    main()
