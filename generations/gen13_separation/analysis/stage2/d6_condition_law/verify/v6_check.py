"""Independent re-derivation of the three load-bearing d6_condition_law numbers.

Written from scratch (the d6_* scripts were not read before this produced its numbers).

N1  within-extractant share of rank-1 amplitude variance   (claim 40.3%, SS 25.89 / 38.33,
    rms 0.377 on 182 cells in 15 extractants)
N2  observed vs predicted within-extractant amplitude sd   (claim obs 0.340; arms 0.009-0.061)
N3  rank-2 condition oracle vs extractant-level curve oracle gain in extractant-macro MAE
    (claim +0.030 vs +0.245 on 169 cells / 12 extractants / 8 arms)

Run:  .venv/Scripts/python.exe generations/gen13_separation/analysis/stage2/d6_condition_law/verify/v6_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/ml_separator_gh")
sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

OUT = ROOT / "generations/gen13_separation/analysis/stage2/d6_condition_law/verify"
OUT.mkdir(parents=True, exist_ok=True)

# standardised Shannon CN8 radius over the 14 lanthanides (same construction as physics_basis)
_r = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES])
RZ = pd.Series((_r - _r.mean()) / _r.std(), index=list(LANTHANIDES))

ARMS = ["C_DIRECT_ROW", "M_SELECTED", "M_PHYSICS_radius+radius_sq", "M_LOWRANK_K2",
        "X_ENS_DIRECT+LOWRANK_K2", "X_ENS_DIRECT+PHYSICS", "B1_MEAN_CURVE", "B3_NN_TANIMOTO"]


# ----------------------------------------------------------------------------- amplitudes
def cell_amplitudes(coh: pd.DataFrame, center_then_no_intercept: bool = False) -> pd.DataFrame:
    """Per-cell quadratic fit of the centred curve on (rz, rz^2).

    amplitude = linear coefficient, curvature = quadratic coefficient.
    Default fit includes an intercept, which makes it identical to fitting the curve
    centred over the METALS OBSERVED IN THAT CELL.  The alternative (centre over the
    observed metals, then fit with no intercept) is computed as a cross-check.
    """
    rows = []
    for _, c in coh.iterrows():
        mets = [m for m in LANTHANIDES if pd.notna(c[f"logD__{m}"])]
        y = np.array([float(c[f"logD__{m}"]) for m in mets])
        x = RZ.loc[mets].to_numpy()
        n = len(mets)
        rec = dict(cell_id=c.cell_id, extractant=c.extractant, extractant_name=c.extractant_name,
                   publication_id=c.publication_id, chemotype=c.chemotype, n_metals=n,
                   rz_span=float(x.max() - x.min()) if n else np.nan,
                   curve_rms=float(np.sqrt(np.mean((y - y.mean()) ** 2))) if n else np.nan,
                   acid_M=float(c["cond__acid_concentration_M"]))
        # replicate noise on each point (fall back to the cell median / cohort median)
        sd = np.array([c[f"repsd__{m}"] for m in mets], dtype=float)
        nrep = np.array([c[f"nrep__{m}"] for m in mets], dtype=float)
        nrep = np.where(np.isfinite(nrep) & (nrep >= 1), nrep, 1.0)
        fallback = c.replicate_sd_median
        if not np.isfinite(fallback):
            fallback = 0.237
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, fallback)
        se_pt = sd / np.sqrt(nrep)
        if n >= 4 and np.ptp(x) > 0:
            if center_then_no_intercept:
                X = np.column_stack([x, x ** 2])
                yy = y - y.mean()
            else:
                X = np.column_stack([np.ones(n), x, x ** 2])
                yy = y
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            resid = yy - X @ beta
            dof = n - X.shape[1]
            XtXi = np.linalg.pinv(X.T @ X)
            # (a) SE from measurement noise propagated through the fit
            cov_meas = XtXi @ (X.T @ np.diag(se_pt ** 2) @ X) @ XtXi
            # (b) SE from the fit residual (classic OLS)
            s2 = float(resid @ resid) / dof if dof > 0 else np.nan
            k = 0 if center_then_no_intercept else 1
            rec["amplitude"] = float(beta[k])
            rec["curvature"] = float(beta[k + 1])
            rec["se_amp_meas"] = float(np.sqrt(max(cov_meas[k, k], 0.0)))
            rec["se_amp_resid"] = float(np.sqrt(s2 * XtXi[k, k])) if np.isfinite(s2) else np.nan
        else:
            rec["amplitude"] = rec["curvature"] = np.nan
            rec["se_amp_meas"] = rec["se_amp_resid"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def ss_decomposition(d: pd.DataFrame, col: str = "amplitude") -> dict:
    grand = d[col].mean()
    g = d.groupby("extractant")[col]
    means, cnt = g.mean(), g.size()
    between = float((cnt * (means - grand) ** 2).sum())
    within = float(((d[col] - d.extractant.map(means)) ** 2).sum())
    return dict(n_cells=len(d), n_extractants=d.extractant.nunique(), within_ss=within,
                between_ss=between, total_ss=within + between,
                within_share=within / (within + between),
                within_rms=float(np.sqrt(within / len(d))))


# ------------------------------------------------------------------ pair-space curve fits
def pair_design(A: pd.Series, B: pd.Series) -> np.ndarray:
    a, b = RZ.reindex(A).to_numpy(), RZ.reindex(B).to_numpy()
    return np.column_stack([a - b, a ** 2 - b ** 2])


def fit_block(U: np.ndarray, r: np.ndarray, rank: int) -> np.ndarray:
    """Least-squares coefficients of r on the first `rank` columns of U (no intercept:
    a pair difference has no level)."""
    Uk = U[:, :rank]
    beta, *_ = np.linalg.lstsq(Uk, r, rcond=None)
    return beta


def main() -> None:
    coh = pd.read_parquet(ROOT / "generations/gen13_separation/manifests/cohort_exact.parquet")
    print(f"cohort {coh.shape}, rz span La-Lu = {RZ.max() - RZ.min():.4f}")

    amp = cell_amplitudes(coh)
    amp_alt = cell_amplitudes(coh, center_then_no_intercept=True)
    amp["amplitude_no_intercept"] = amp_alt["amplitude"].to_numpy()

    ok4 = amp[amp.n_metals >= 4]
    print(f"\n[A] cells with >=4 metals: {len(ok4)}")
    bands = [(0, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 9)]
    band_rows = []
    for lo, hi in bands:
        s = ok4[(ok4.rz_span > lo) & (ok4.rz_span <= hi)]
        band_rows.append(dict(band=f"({lo},{hi}]", n=len(s), amp_sd=s.amplitude.std(),
                              amp_abs_median=s.amplitude.abs().median()))
        print(f"    rz_span {lo}-{hi}: n={len(s):4d}  amplitude sd={s.amplitude.std():8.3f}")
    pd.DataFrame(band_rows).to_csv(OUT / "v6_span_bands.csv", index=False)

    ana = amp[(amp.n_metals >= 4) & (amp.rz_span >= 1.5)].copy()
    ana["in_analysis_set"] = True
    print(f"[A] analysis set (>=4 metals, span>=1.5): {len(ana)} cells "
          f"(claim 254); median SE_meas={ana.se_amp_meas.median():.4f} (claim 0.050), "
          f"median SE_resid={ana.se_amp_resid.median():.4f}")

    cnt = ana.extractant.value_counts()
    sub = ana[ana.extractant.map(cnt) >= 3].copy()
    d = ss_decomposition(sub)
    print(f"\n[N1] >=3 analysis cells per extractant: {d['n_cells']} cells (claim 182) in "
          f"{d['n_extractants']} extractants (claim 15)")
    print(f"[N1] within SS={d['within_ss']:.2f} (claim 25.89)  between SS={d['between_ss']:.2f} "
          f"(claim 38.33)  total={d['total_ss']:.2f} (claim 64.22)")
    print(f"[N1] within share={100*d['within_share']:.1f}% (claim 40.3%)   "
          f"within rms={d['within_rms']:.3f} (claim 0.377)")
    d_alt = ss_decomposition(sub, "amplitude_no_intercept")
    print(f"[N1] no-intercept variant: within SS={d_alt['within_ss']:.2f} "
          f"share={100*d_alt['within_share']:.1f}% rms={d_alt['within_rms']:.3f}")
    pd.DataFrame([dict(variant="quadratic_with_intercept", **d),
                  dict(variant="centred_no_intercept", **d_alt)]).to_csv(
        OUT / "v6_n1_variance_decomposition.csv", index=False)

    # noise floor on the amplitude: rms of the per-cell measurement SE
    print(f"[N1] noise floor: rms of per-cell SE_meas = "
          f"{np.sqrt((sub.se_amp_meas ** 2).mean()):.4f}  "
          f"ratio rms/floor = {d['within_rms'] / np.sqrt((sub.se_amp_meas ** 2).mean()):.2f} "
          f"(claim 7.0)")

    # ------------------------------------------------- acid subset (12 extractants / 169 cells)
    nacid = ana.groupby("extractant").acid_M.nunique()
    acid_extr = sorted(set(cnt[cnt >= 3].index) & set(nacid[nacid >= 2].index))
    acid = ana[ana.extractant.isin(acid_extr)].copy()
    print(f"\n[N2] acid subset: {len(acid)} cells (claim 169) in {len(acid_extr)} extractants "
          f"(claim 12), {acid.chemotype.nunique()} chemotypes (claim 6)")
    obs_sd = acid.groupby("extractant").amplitude.std(ddof=1)
    obs_pooled = float(np.sqrt(((acid.groupby('extractant').amplitude
                                 .transform('mean') - acid.amplitude) ** 2).sum()
                               / (len(acid) - len(acid_extr))))
    print(f"[N2] observed within-extractant amplitude sd: macro over extractants "
          f"{obs_sd.mean():.3f}, pooled {obs_pooled:.3f} (claim 0.340)")

    slopes = []
    for e, g in acid.groupby("extractant"):
        x = np.log10(g.acid_M.to_numpy())
        if len(np.unique(x)) < 2:
            continue
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, g.amplitude.to_numpy(), rcond=None)
        slopes.append(dict(extractant_name=g.extractant_name.iloc[0], n=len(g), slope=beta[1]))
    sl = pd.DataFrame(slopes)
    print(f"[N2] d(amp)/d(log10 acid): median {sl.slope.median():.3f} (claim -0.095), "
          f"mean {sl.slope.mean():.3f} (claim -0.130), sd {sl.slope.std():.3f} (claim 1.072), "
          f"n={len(sl)}")
    sl.to_csv(OUT / "v6_n2_acid_slopes.csv", index=False)

    # ------------------------------------------- predicted amplitude spread + oracle gains
    cells_keep = set(acid.cell_id)
    amp_map = acid.set_index("cell_id")
    spread_rows, oracle_rows = [], []
    for arm in ARMS:
        p = pd.read_parquet(ROOT / f"generations/gen13_separation/predictions/B_primary/{arm}.parquet")
        p = p[p.cell_id.isin(cells_keep)].copy()
        U_all = pair_design(p.A, p.B)
        p["u1"], p["u2"] = U_all[:, 0], U_all[:, 1]
        p["resid"] = p.y - p.prediction

        per_seed_spread, per_seed_corr = [], []
        per_seed_gain = {k: [] for k in ["base", "within_rank1", "within_rank2",
                                         "extr_rank2", "both_rank2"]}
        for seed, ps in p.groupby("split_seed"):
            # per-cell observed / predicted amplitude from the SAME pair-space quadratic fit
            recs = []
            for cid, g in ps.groupby("cell_id"):
                U = g[["u1", "u2"]].to_numpy()
                bo = fit_block(U, g.y.to_numpy(), 2)
                bp = fit_block(U, g.prediction.to_numpy(), 2)
                recs.append(dict(cell_id=cid, extractant=g.extractant.iloc[0],
                                 amp_obs=bo[0], amp_pred=bp[0],
                                 cur_obs=bo[1], cur_pred=bp[1]))
            R = pd.DataFrame(recs)
            gg = R.groupby("extractant")
            sd_o = gg.amp_obs.std(ddof=1)
            sd_p = gg.amp_pred.std(ddof=1)
            cr = gg.apply(lambda t: np.corrcoef(t.amp_obs, t.amp_pred)[0, 1]
                          if len(t) > 2 and t.amp_pred.std() > 1e-12 else np.nan,
                          include_groups=False)
            per_seed_spread.append((sd_o.mean(), sd_p.mean()))
            per_seed_corr.append(cr.mean(skipna=True))

            # oracle corrections, applied in pair space
            ps = ps.copy()
            for rank, name in [(1, "rank1"), (2, "rank2")]:
                cellb = {}
                for cid, g in ps.groupby("cell_id"):
                    cellb[cid] = fit_block(g[["u1", "u2"]].to_numpy(), g.resid.to_numpy(), rank)
                extb = {}
                for e, g in ps.groupby("extractant"):
                    extb[e] = fit_block(g[["u1", "u2"]].to_numpy(), g.resid.to_numpy(), rank)
                Uk = ps[["u1", "u2"]].to_numpy()[:, :rank]
                cb = np.array([cellb[c] for c in ps.cell_id])
                eb = np.array([extb[e] for e in ps.extractant])
                ps[f"corr_cell_{name}"] = (Uk * cb).sum(1)
                ps[f"corr_extr_{name}"] = (Uk * eb).sum(1)

            def macro(err: np.ndarray) -> float:
                t = pd.DataFrame({"extractant": ps.extractant.to_numpy(), "e": err})
                return float(t.groupby("extractant").e.mean().mean())

            r = ps.resid.to_numpy()
            per_seed_gain["base"].append(macro(np.abs(r)))
            per_seed_gain["within_rank1"].append(
                macro(np.abs(r - (ps.corr_cell_rank1 - ps.corr_extr_rank1))))
            per_seed_gain["within_rank2"].append(
                macro(np.abs(r - (ps.corr_cell_rank2 - ps.corr_extr_rank2))))
            per_seed_gain["extr_rank2"].append(macro(np.abs(r - ps.corr_extr_rank2)))
            per_seed_gain["both_rank2"].append(macro(np.abs(r - ps.corr_cell_rank2)))

        so = float(np.mean([a for a, _ in per_seed_spread]))
        sp = float(np.mean([b for _, b in per_seed_spread]))
        spread_rows.append(dict(arm=arm, obs_sd=so, pred_sd=sp, ratio=sp / so,
                                corr_obs_pred=float(np.nanmean(per_seed_corr))))
        m = {k: float(np.mean(v)) for k, v in per_seed_gain.items()}
        oracle_rows.append(dict(arm=arm, macro_mae_base=m["base"],
                                gain_within_rank1=m["base"] - m["within_rank1"],
                                gain_within_rank2=m["base"] - m["within_rank2"],
                                gain_extractant_rank2=m["base"] - m["extr_rank2"],
                                gain_both_rank2=m["base"] - m["both_rank2"]))
        print(f"[N2/N3] {arm:28s} obs_sd={so:.3f} pred_sd={sp:.3f} ({sp/so:5.1%}) "
              f"corr={np.nanmean(per_seed_corr):+.3f} | base={m['base']:.3f} "
              f"within2={m['base']-m['within_rank2']:+.3f} "
              f"extr2={m['base']-m['extr_rank2']:+.3f} both={m['base']-m['both_rank2']:+.3f}")

    S = pd.DataFrame(spread_rows)
    O = pd.DataFrame(oracle_rows)
    S.to_csv(OUT / "v6_n2_amplitude_spread_by_arm.csv", index=False)
    O.to_csv(OUT / "v6_n3_oracle_gains.csv", index=False)
    print(f"\n[N2] mean observed within-extractant amplitude sd over arms/seeds "
          f"{S.obs_sd.mean():.3f} (claim 0.340); predicted sd range "
          f"{S.pred_sd.min():.3f}-{S.pred_sd.max():.3f} (claim 0.009-0.061); "
          f"corr range {S.corr_obs_pred.min():+.3f}..{S.corr_obs_pred.max():+.3f} "
          f"(claim -0.064..+0.083)")
    print(f"[N3] condition (within) rank-2 oracle gain: mean {O.gain_within_rank2.mean():+.3f} "
          f"(claim +0.030), range {O.gain_within_rank2.min():+.3f}..{O.gain_within_rank2.max():+.3f} "
          f"(claim +0.016..+0.050)")
    print(f"[N3] extractant-LEVEL rank-2 oracle gain: mean {O.gain_extractant_rank2.mean():+.3f} "
          f"(claim +0.245), range {O.gain_extractant_rank2.min():+.3f}.."
          f"{O.gain_extractant_rank2.max():+.3f} (claim +0.200..+0.421)")
    print(f"[N3] ratio extractant/condition = "
          f"{O.gain_extractant_rank2.mean()/O.gain_within_rank2.mean():.1f}x (claim 8x)")
    print(f"[N3] rank-1-only within oracle gain: mean {O.gain_within_rank1.mean():+.3f}, "
          f"C_DIRECT_ROW {O.set_index('arm').loc['C_DIRECT_ROW','gain_within_rank1']:+.3f} "
          f"(claim 0.468->0.477, i.e. -0.009)")
    print(f"[N3] both (full cell rank-2) gain: range {O.gain_both_rank2.min():.3f}.."
          f"{O.gain_both_rank2.max():.3f} (claim 0.34-0.56)")

    # -------------------------------------------------- dilution to the full 90-extractant macro
    n_extr_total = 90
    print(f"[N3] condition oracle diluted over all {n_extr_total} extractants: "
          f"{O.gain_within_rank2.mean() * len(acid_extr) / n_extr_total:.4f} (claim 0.004)")

    # ------------------------------------------------------------------- leaderboard cross-check
    lb = pd.read_csv(ROOT / "generations/gen13_separation/metrics/B_primary/leaderboard.csv").set_index("arm")
    ab = pd.read_csv(ROOT / "generations/gen13_separation/metrics/B_abl_cond_only/leaderboard.csv"
                     ).set_index("arm")
    print(f"\n[X] C_DIRECT_ROW primary {lb.loc['C_DIRECT_ROW','macro_mae_extractant']:.4f} vs "
          f"cond-only {ab.loc['C_DIRECT_ROW','macro_mae_extractant']:.4f} "
          f"(delta {lb.loc['C_DIRECT_ROW','macro_mae_extractant']-ab.loc['C_DIRECT_ROW','macro_mae_extractant']:+.4f}; "
          f"claim 0.4953 / 0.4709 / 0.024)")

    amp.to_csv(OUT / "v6_cell_amplitudes.csv", index=False)
    print("\nwrote:", *(f.name for f in sorted(OUT.glob('v6_*.csv'))))


if __name__ == "__main__":
    main()
