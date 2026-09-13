"""Independent re-derivation of the load-bearing numbers in diagnostic D7.

Written from scratch (the D7 agent's own scripts were NOT read before this ran).

Checks
  C1  per-cell smooth fit quality on centred curves (pooled/median R^2, |resid|)
  C2  per-element mean residual (Eu / Gd / Nd / Ho / Lu) + cluster bootstrap
  C3  split-half correlation of the mean residual curve, by extractant
  C4  split-half-debiased reproducible rms amplitude
  C5  arm baselines under the extractant-macro convention + the in-sample
      global 14-element non-smooth offset ceiling, + lambda=1 chemotype-held-out
      mean-residual-curve correction on three arms
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "gen13_separation")
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

ROOT = "gen13_separation"
OUT = os.path.join(ROOT, "analysis", "stage2", "d7_tetrad_residual", "verify")
os.makedirs(OUT, exist_ok=True)

LN = list(LANTHANIDES)
NEL = len(LN)
IDX = {m: i for i, m in enumerate(LN)}

# standardised Shannon CN8 radius over the 14 lanthanides
r_raw = np.array([SHANNON_RADIUS_CN8[m] for m in LN], float)
Z = (r_raw - r_raw.mean()) / r_raw.std(ddof=0)

results = {}


def hdr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ---------------------------------------------------------------- load cohort
coh = pd.read_parquet(os.path.join(ROOT, "manifests", "cohort_exact.parquet"))
print("cohort rows", len(coh))

logD = coh[[f"logD__{m}" for m in LN]].to_numpy(float)          # (521, 14)
nrep = coh[[f"nrep__{m}" for m in LN]].to_numpy(float)
repsd = coh[[f"repsd__{m}" for m in LN]].to_numpy(float)
obs = np.isfinite(logD)
n_obs = obs.sum(1)

# centred curve: subtract the mean over the metals OBSERVED IN THAT CELL
mean_obs = np.where(n_obs > 0, np.nansum(np.where(obs, logD, 0.0), 1) / np.maximum(n_obs, 1), np.nan)
Y = np.where(obs, logD - mean_obs[:, None], np.nan)

# sanity: each row's centred curve sums to ~0 over its observed metals
row_sums = np.nansum(np.where(obs, Y, 0.0), 1)
print("max |sum of centred curve over observed metals| =", np.abs(row_sums).max())

sel = n_obs >= 6
cells = np.where(sel)[0]
print("cells with >=6 observed metals:", sel.sum(), " cell-metal points:", int(obs[sel].sum()))
print("distinct extractants among them:", coh.loc[sel, "extractant"].nunique())
print("distinct publications:", coh.loc[sel, "publication_id"].nunique())
print("distinct chemotypes:", coh.loc[sel, "chemotype"].nunique())


# ------------------------------------------------- C1 smooth-fit quality
def fit_resid(deg):
    """Per-cell OLS of the centred curve on [1, z, ..., z^deg]; return residual matrix."""
    R = np.full((len(coh), NEL), np.nan)
    r2 = np.full(len(coh), np.nan)
    for i in cells:
        m = obs[i]
        X = np.vander(Z[m], deg + 1, increasing=True)
        y = Y[i, m]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        R[i, m] = res
        ss_tot = float((y ** 2).sum())          # curve already centred -> SS about 0
        r2[i] = 1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return R, r2


hdr("C1  smoothness of the centred curve in standardised Shannon CN8 radius")
rows = []
resid_by_deg = {}
for deg, name in [(1, "linear"), (2, "quadratic"), (3, "cubic")]:
    R, r2 = fit_resid(deg)
    resid_by_deg[deg] = R
    mask = np.isfinite(R)
    ss_res = float((R[mask] ** 2).sum())
    ss_tot = float((Y[sel] ** 2)[np.isfinite(Y[sel])].sum())
    pooled_r2 = 1.0 - ss_res / ss_tot
    med_r2 = float(np.nanmedian(r2[sel]))
    med_abs = float(np.median(np.abs(R[mask])))
    rms = float(np.sqrt((R[mask] ** 2).mean()))
    rows.append(dict(degree=deg, name=name, pooled_r2=pooled_r2, median_cell_r2=med_r2,
                     median_abs_resid=med_abs, rms_resid=rms,
                     n_cells=int(sel.sum()), n_points=int(mask.sum())))
    print(f"  {name:10s} pooled R2={pooled_r2:.4f}  median cell R2={med_r2:.4f}  "
          f"median|r|={med_abs:.4f}  rms={rms:.4f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "V_c1_smoothness.csv"), index=False)
results["c1"] = rows

RQ = resid_by_deg[2]        # quadratic residuals, the D7 working object

# replicate SEM reference among the >=6-metal cells
rep_mask = sel[:, None] & obs & np.isfinite(repsd) & (nrep > 1)
sd_vals = repsd[rep_mask]
sem_vals = repsd[rep_mask] / np.sqrt(nrep[rep_mask])
print(f"  replicated cell-metals with nrep>1 in the >=6-metal cells: {rep_mask.sum()}")
print(f"  median replicate sd = {np.median(sd_vals):.4f}   median SEM = {np.median(sem_vals):.4f}")
results["replicate"] = dict(n=int(rep_mask.sum()), median_sd=float(np.median(sd_vals)),
                            median_sem=float(np.median(sem_vals)))


# ------------------------------------------------- C2 per-element mean residual
hdr("C2  per-element mean quadratic residual (cells weighted equally)")
extr = coh["extractant"].to_numpy()
sel_extr = extr[sel]
uniq_extr = np.unique(sel_extr)
Rsel = RQ[sel]
rng = np.random.default_rng(20260908)
B = 2000
# cluster bootstrap over extractants
boot = np.full((B, NEL), np.nan)
groups = [np.where(sel_extr == e)[0] for e in uniq_extr]
for b in range(B):
    pick = rng.integers(0, len(groups), len(groups))
    idx = np.concatenate([groups[k] for k in pick])
    sub = Rsel[idx]
    with np.errstate(invalid="ignore"):
        boot[b] = np.nanmean(sub, axis=0)

el_rows = []
for j, m in enumerate(LN):
    col = Rsel[:, j]
    v = col[np.isfinite(col)]
    mu = float(v.mean())
    se = float(np.nanstd(boot[:, j], ddof=1))
    z = mu / se if se > 0 else np.nan
    from scipy.stats import norm
    p = float(2 * norm.sf(abs(z)))
    el_rows.append(dict(element=m, n_cells=len(v), mean_resid=mu, boot_se=se, z=z, p_raw=p,
                        median_abs_resid=float(np.median(np.abs(v)))))
edf = pd.DataFrame(el_rows)
# Holm-Bonferroni over the 14 elements
order = np.argsort(edf["p_raw"].to_numpy())
adj = np.empty(NEL)
running = 0.0
for k, oi in enumerate(order):
    val = (NEL - k) * edf["p_raw"].to_numpy()[oi]
    running = max(running, val)
    adj[oi] = min(1.0, running)
edf["p_holm"] = adj
edf = edf.sort_values("p_holm")
edf.to_csv(os.path.join(OUT, "V_c2_per_element_residual.csv"), index=False)
print(edf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
results["c2"] = edf.to_dict("records")


# ------------------------------------------------- C3 split-half correlation
hdr("C3  split-half correlation of the mean residual curve")


def mean_curve(rows_idx):
    sub = Rsel[rows_idx]
    with np.errstate(invalid="ignore"):
        return np.nanmean(sub, axis=0)


def split_half(labels, n_draws=500, seed=7):
    r = np.random.default_rng(seed)
    lab = np.asarray(labels)
    u = np.unique(lab)
    gidx = {g: np.where(lab == g)[0] for g in u}
    out = []
    for _ in range(n_draws):
        perm = r.permutation(len(u))
        h1 = u[perm[: len(u) // 2]]
        h2 = u[perm[len(u) // 2:]]
        c1 = mean_curve(np.concatenate([gidx[g] for g in h1]))
        c2 = mean_curve(np.concatenate([gidx[g] for g in h2]))
        ok = np.isfinite(c1) & np.isfinite(c2)
        if ok.sum() >= 3:
            out.append(float(np.corrcoef(c1[ok], c2[ok])[0, 1]))
    return np.array(out)


sh_rows = []
for name, lab in [("extractant", sel_extr),
                  ("publication", coh.loc[sel, "publication_id"].to_numpy()),
                  ("chemotype", coh.loc[sel, "chemotype"].to_numpy())]:
    rs = split_half(lab)
    sh_rows.append(dict(unit=name, n_groups=int(len(np.unique(lab))), n_draws=len(rs),
                        median_r=float(np.median(rs)), lo=float(np.percentile(rs, 2.5)),
                        hi=float(np.percentile(rs, 97.5)),
                        frac_positive=float((rs > 0).mean())))
    print(f"  by {name:12s} median r={np.median(rs):+.3f} "
          f"[{np.percentile(rs,2.5):+.3f}, {np.percentile(rs,97.5):+.3f}] "
          f"({(rs>0).mean()*100:.0f}% positive, {len(np.unique(lab))} groups)")

# within-cell permutation null: shuffle each cell's residuals across its observed metals
rng2 = np.random.default_rng(11)
Rperm = np.full_like(Rsel, np.nan)
for i in range(Rsel.shape[0]):
    ok = np.where(np.isfinite(Rsel[i]))[0]
    Rperm[i, ok] = Rsel[i, rng2.permutation(ok)]
Rtrue = Rsel
Rsel = Rperm
null_med = {}
for name, lab in [("extractant", sel_extr),
                  ("publication", coh.loc[sel, "publication_id"].to_numpy()),
                  ("chemotype", coh.loc[sel, "chemotype"].to_numpy())]:
    rs = split_half(lab, n_draws=300, seed=99)
    null_med[name] = float(np.median(rs))
    print(f"  permutation null by {name:12s} median r={np.median(rs):+.3f}")
Rsel = Rtrue
for row in sh_rows:
    row["perm_null_median_r"] = null_med[row["unit"]]
pd.DataFrame(sh_rows).to_csv(os.path.join(OUT, "V_c3_split_half.csv"), index=False)
results["c3"] = sh_rows


# ------------------------------------------------- C4 reproducible rms amplitude
hdr("C4  split-half-debiased reproducible rms amplitude of the common curve")
# E[<c1, c2>] over independent halves = ||true common curve||^2 (noise is independent)
r4 = np.random.default_rng(4242)
u = uniq_extr
gidx = {g: np.where(sel_extr == g)[0] for g in u}
vals = []
for _ in range(2000):
    perm = r4.permutation(len(u))
    h1 = u[perm[: len(u) // 2]]
    h2 = u[perm[len(u) // 2:]]
    c1 = mean_curve(np.concatenate([gidx[g] for g in h1]))
    c2 = mean_curve(np.concatenate([gidx[g] for g in h2]))
    ok = np.isfinite(c1) & np.isfinite(c2)
    cov = float((c1[ok] * c2[ok]).mean())
    vals.append(np.sqrt(cov) if cov > 0 else -np.sqrt(-cov))
vals = np.array(vals)
amp = float(np.median(vals))
print(f"  reproducible rms amplitude = {amp:.4f} log units "
      f"[{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]  (2000 extractant half-splits)")
full_curve = mean_curve(np.arange(Rsel.shape[0]))
print(f"  raw rms of the full mean residual curve = {np.sqrt(np.nanmean(full_curve**2)):.4f}")
pooled_rms = float(np.sqrt(np.nanmean(RQ[sel][np.isfinite(RQ[sel])] ** 2)))
print(f"  pooled residual rms = {pooled_rms:.4f} -> share of pooled residual variance "
      f"= {amp**2/pooled_rms**2*100:.2f}%")
pd.DataFrame([dict(reproducible_rms=amp, lo=float(np.percentile(vals, 2.5)),
                   hi=float(np.percentile(vals, 97.5)),
                   raw_mean_curve_rms=float(np.sqrt(np.nanmean(full_curve ** 2))),
                   pooled_resid_rms=pooled_rms,
                   share_of_pooled_var_pct=amp ** 2 / pooled_rms ** 2 * 100)]
             ).to_csv(os.path.join(OUT, "V_c4_amplitude.csv"), index=False)
results["c4"] = dict(amp=amp, lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)),
                     pooled_rms=pooled_rms)
np.savetxt(os.path.join(OUT, "V_mean_residual_curve.csv"),
           np.c_[np.arange(NEL), full_curve], delimiter=",", header="element_index,mean_resid",
           comments="")
print("  mean residual curve:", {m: round(float(full_curve[j]), 4) for j, m in enumerate(LN)})


# ------------------------------------------------- C5 arm-level MAE
hdr("C5  arm baselines under the extractant-macro convention, and the prize")


def macro_mae(df, err_col):
    """metric within extractant over its held-out pairs -> mean over extractants -> mean over seeds."""
    per = df.groupby(["split_seed", "extractant"])[err_col].mean()
    per_seed = per.groupby("split_seed").mean()
    return float(per_seed.mean()), per_seed


ARMS = ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "M_SELECTED", "B1_MEAN_CURVE",
        "B4_HEAVIER_ALWAYS", "X_ENS_DIRECT+PHYSICS"]
arm_rows = []
for arm in ARMS:
    p = pd.read_parquet(os.path.join(ROOT, "predictions", "B_primary", f"{arm}.parquet"))
    p["ae"] = (p["y"] - p["prediction"]).abs()
    base, per_seed = macro_mae(p, "ae")
    pooled = float(p["ae"].mean())
    arm_rows.append(dict(arm=arm, macro_mae=base, pooled_mae=pooled,
                         n_rows=len(p), n_extractants=p["extractant"].nunique(),
                         seed_sd=float(per_seed.std(ddof=1))))
    print(f"  {arm:28s} macro={base:.4f}  pooled={pooled:.4f}  "
          f"({p['extractant'].nunique()} extractants, {len(p)} rows)")

# ---- the generous in-sample ceiling: global 14-element offset, non-smooth part only
hdr("C5b  in-sample global 14-element offset, restricted to its non-smooth part")
P = np.vander(Z, 3, increasing=True)                 # [1, z, z^2]
Pproj = P @ np.linalg.pinv(P)                        # smooth projector on the 14-vector


def offset_ceiling(arm):
    p = pd.read_parquet(os.path.join(ROOT, "predictions", "B_primary", f"{arm}.parquet"))
    p["ae"] = (p["y"] - p["prediction"]).abs()
    base, _ = macro_mae(p, "ae")
    ia = p["A"].map(IDX).to_numpy()
    ib = p["B"].map(IDX).to_numpy()
    e = (p["y"] - p["prediction"]).to_numpy()
    out = {}
    for seed, g in p.groupby("split_seed"):
        pass
    # fit one global offset on ALL rows (pooled across seeds), least squares on differences
    D = np.zeros((len(p), NEL))
    D[np.arange(len(p)), ia] += 1.0
    D[np.arange(len(p)), ib] -= 1.0
    o, *_ = np.linalg.lstsq(D, e, rcond=None)
    o = o - o.mean()
    o_smooth = Pproj @ o
    o_ns = o - o_smooth
    for tag, vec in [("full", o), ("nonsmooth", o_ns), ("smooth", o_smooth)]:
        corr = vec[ia] - vec[ib]
        p["ae2"] = np.abs(e - corr)
        m, _ = macro_mae(p, "ae2")
        out[tag] = (m, float(np.sqrt((corr ** 2).mean())))
    return base, out, o, o_ns


ceil_rows = []
for arm in ["X_ENS_DIRECT+LOWRANK_K2", "C_DIRECT_ROW", "B1_MEAN_CURVE"]:
    base, out, o, o_ns = offset_ceiling(arm)
    print(f"  {arm}: base={base:.4f} -> nonsmooth-offset={out['nonsmooth'][0]:.4f} "
          f"(delta {out['nonsmooth'][0]-base:+.4f}, corr rms {out['nonsmooth'][1]:.4f}); "
          f"full-offset={out['full'][0]:.4f} ({out['full'][0]-base:+.4f}); "
          f"smooth-only={out['smooth'][0]:.4f} ({out['smooth'][0]-base:+.4f})")
    ceil_rows.append(dict(arm=arm, base_macro_mae=base,
                          nonsmooth_macro_mae=out["nonsmooth"][0],
                          delta_nonsmooth=out["nonsmooth"][0] - base,
                          nonsmooth_corr_rms=out["nonsmooth"][1],
                          full_macro_mae=out["full"][0], delta_full=out["full"][0] - base,
                          smooth_macro_mae=out["smooth"][0], delta_smooth=out["smooth"][0] - base))
    if arm == "X_ENS_DIRECT+LOWRANK_K2":
        print("   offset (full):", {m: round(float(o[j]), 3) for j, m in enumerate(LN)})
        print("   offset (nonsmooth):", {m: round(float(o_ns[j]), 3) for j, m in enumerate(LN)})
pd.DataFrame(ceil_rows).to_csv(os.path.join(OUT, "V_c5b_offset_ceiling.csv"), index=False)

# ---- lambda=1 chemotype-held-out mean residual curve applied to real arms
hdr("C5c  chemotype-held-out mean residual curve applied to the arms at lambda=1")
cell_chem = coh.set_index("cell_id")["chemotype"].to_dict()
sel_cellids = coh.loc[sel, "cell_id"].to_numpy()
sel_chem = coh.loc[sel, "chemotype"].to_numpy()
# LOCO (leave-one-chemotype-out) mean residual curve
loco = {}
for ch in np.unique(coh["chemotype"]):
    keep = sel_chem != ch
    if keep.sum() == 0:
        loco[ch] = np.zeros(NEL)
        continue
    with np.errstate(invalid="ignore"):
        c = np.nanmean(RQ[sel][keep], axis=0)
    loco[ch] = np.nan_to_num(c)

lam_rows = []
for arm in ["C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2", "B1_MEAN_CURVE"]:
    p = pd.read_parquet(os.path.join(ROOT, "predictions", "B_primary", f"{arm}.parquet"))
    p["ae"] = (p["y"] - p["prediction"]).abs()
    base, _ = macro_mae(p, "ae")
    curves = np.array([loco.get(c, np.zeros(NEL)) for c in p["chemotype"].to_numpy()])
    ia = p["A"].map(IDX).to_numpy()
    ib = p["B"].map(IDX).to_numpy()
    corr = curves[np.arange(len(p)), ia] - curves[np.arange(len(p)), ib]
    e = (p["y"] - p["prediction"]).to_numpy()
    best = (None, None)
    grid = {}
    for lam in np.round(np.arange(-1.0, 2.01, 0.1), 2):
        p["ae2"] = np.abs(e - lam * corr)
        m, _ = macro_mae(p, "ae2")
        grid[float(lam)] = m
        if best[1] is None or m < best[1]:
            best = (float(lam), m)
    print(f"  {arm:28s} base={base:.4f}  lambda=1 -> {grid[1.0]:.4f} ({grid[1.0]-base:+.4f})  "
          f"best lambda={best[0]:+.1f} -> {best[1]:.4f} ({best[1]-base:+.4f})")
    lam_rows.append(dict(arm=arm, base_macro_mae=base, mae_lambda1=grid[1.0],
                         delta_lambda1=grid[1.0] - base, best_lambda=best[0],
                         mae_best=best[1], delta_best=best[1] - base,
                         corr_rms=float(np.sqrt((corr ** 2).mean()))))
pd.DataFrame(lam_rows).to_csv(os.path.join(OUT, "V_c5c_lambda_prize.csv"), index=False)
pd.DataFrame(arm_rows).to_csv(os.path.join(OUT, "V_c5a_arm_baselines.csv"), index=False)

with open(os.path.join(OUT, "V_summary.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nwrote outputs to", OUT)
