"""Independent re-derivation of the d4_amplitude_shape headline numbers.

Written from scratch (the original agent's script was not read before this ran).

What is re-derived
------------------
1. amplitude / shape share of the total pairwise squared error
2. extractant-macro MAE at baseline, with amplitude made exact, with shape made exact
3. oracle global gain g and its extractant-macro MAE gain
4. sd(observed amplitude) / sd(predicted amplitude)
5. single-pair (widest-gap anchor) gain calibration
6. curve-reconstruction residual and non-integrable share of the predicted pairs

Conventions enforced here
-------------------------
* a "unit" is one (split_seed, cell_id); its curve lives ONLY on the metals that
  cell actually observed, and is centred over exactly those metals.
* extractant-macro = MAE inside a (seed, extractant) group, then mean over the
  extractants of a seed, then mean over the 5 seeds.  Every extractant counts
  once regardless of how many pairs it has.
* the amplitude direction u is the Shannon CN8 radius restricted to the cell's
  observed metals, mean-centred over those metals and scaled to unit L2 norm.
  (Any global affine standardisation of the radius cancels out under
  centre-then-normalise, so "standardised" vs "raw" radius cannot change u.)
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = "D:/ml_separator_gh"
PRED = os.path.join(ROOT, "gen13_separation/predictions/B_primary")
OUT = os.path.join(ROOT, "gen13_separation/analysis/stage2/d4_amplitude_vs_shape/verify")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "gen13_separation"))
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

LN_IDX = {m: i for i, m in enumerate(LANTHANIDES)}
RADIUS = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES], float)

ARMS = [
    "C_DIRECT_ROW",
    "M_SELECTED",
    "M_PHYSICS_radius+radius_sq",
    "M_LOWRANK_K2",
    "X_ENS_DIRECT+LOWRANK_K2",
]

GAIN_GRID = np.round(np.arange(0.80, 2.0001, 0.05), 4)  # 25 points


# ----------------------------------------------------------------- helpers
def macro_mae(df, err_col):
    """extractant-macro MAE: mean over extractants inside a seed, then over seeds."""
    per_group = df.groupby(["split_seed", "extractant"])[err_col].mean()
    per_seed = per_group.groupby("split_seed").mean()
    return float(per_seed.mean()), per_group


def macro_mae_from_abs(seed, extr, absval):
    """same thing on raw numpy arrays (used inside the gain sweep)."""
    t = pd.DataFrame({"s": seed, "e": extr, "a": absval})
    g = t.groupby(["s", "e"])["a"].mean()
    return float(g.groupby("s").mean().mean())


def build_units(df):
    """Return per-unit index structures.  One unit = (split_seed, cell_id)."""
    key = df["split_seed"].astype(str) + "|" + df["cell_id"]
    df = df.assign(_key=key)
    order = np.argsort(key.values, kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    ks = df["_key"].values
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    ends = np.r_[starts[1:], len(ks)]
    return df, list(zip(starts, ends))


def unit_geometry(sub):
    """metals of the unit, incidence matrix P, unit amplitude direction u."""
    metals = sorted(set(sub["A"]).union(sub["B"]), key=lambda m: LN_IDX[m])
    m = len(metals)
    pos = {mm: i for i, mm in enumerate(metals)}
    n = len(sub)
    P = np.zeros((n, m))
    P[np.arange(n), [pos[a] for a in sub["A"]]] = 1.0
    P[np.arange(n), [pos[b] for b in sub["B"]]] = -1.0
    r = RADIUS[[LN_IDX[mm] for mm in metals]]
    u = r - r.mean()
    nrm = np.linalg.norm(u)
    u = u / nrm if nrm > 0 else u
    return metals, m, P, u


# ----------------------------------------------------------------- main
def run_arm(arm):
    raw = pd.read_parquet(os.path.join(PRED, arm + ".parquet"))
    df, spans = build_units(raw)

    n_units = len(spans)
    n_pairs = len(df)

    # per-pair outputs
    amp_exact_pred = np.empty(n_pairs)   # prediction with amplitude replaced by observed
    shape_exact_pred = np.empty(n_pairs)  # prediction with shape replaced by observed
    # per-unit outputs
    a_obs = np.empty(n_units)
    a_pred = np.empty(n_units)
    m_of_unit = np.empty(n_units)
    sse_amp = np.empty(n_units)   # ||P (a_err * u)||^2
    sse_shape = np.empty(n_units)  # ||P s||^2
    sse_tot = np.empty(n_units)   # ||P e||^2  (integrable part of the error)
    sse_pairtot = np.empty(n_units)  # ||pred - y||^2  (actual pairwise SSE)
    nonint_pred = np.empty(n_units)   # ||pred - P c_pred||^2
    sse_pred = np.empty(n_units)      # ||pred||^2
    recon_res = np.empty(n_units)     # max |P c_obs - y|
    n_metals_col = np.empty(n_units, dtype=int)
    complete = np.empty(n_units, dtype=bool)

    for i, (s, e) in enumerate(spans):
        sub = df.iloc[s:e]
        metals, m, P, u = unit_geometry(sub)
        y = sub["y"].values.astype(float)
        p = sub["prediction"].values.astype(float)

        # least-squares centred curves.  For a complete graph P'P = m*I on the
        # centred subspace and P'y is automatically centred, so c = P'y / m is
        # exactly the minimum-norm least-squares solution.
        c_obs = P.T @ y / m
        c_pred = P.T @ p / m

        recon_res[i] = np.max(np.abs(P @ c_obs - y))
        nonint_pred[i] = float(np.sum((p - P @ c_pred) ** 2))
        sse_pred[i] = float(np.sum(p ** 2))

        err = c_pred - c_obs
        ao, ap = float(c_obs @ u), float(c_pred @ u)
        a_err = ap - ao
        shape = err - a_err * u

        # P'P = m*I on centred vectors, so ||P v||^2 = m*||v||^2 and the
        # amplitude/shape cross term vanishes exactly.
        sse_amp[i] = m * a_err ** 2
        sse_shape[i] = m * float(shape @ shape)
        sse_tot[i] = m * float(err @ err)
        sse_pairtot[i] = float(np.sum((p - y) ** 2))

        a_obs[i], a_pred[i], m_of_unit[i] = ao, ap, m
        n_metals_col[i] = int(sub["n_metals"].iloc[0])
        complete[i] = (len(sub) == m * (m - 1) // 2) and (m == n_metals_col[i])

        Pu = P @ u
        amp_exact_pred[s:e] = p - a_err * Pu      # kill only the amplitude error
        shape_exact_pred[s:e] = P @ (c_obs + a_err * u)  # keep predicted amplitude

    df["amp_exact"] = amp_exact_pred
    df["shape_exact"] = shape_exact_pred

    units = pd.DataFrame({
        "split_seed": [df["split_seed"].iloc[s] for s, _ in spans],
        "cell_id": [df["cell_id"].iloc[s] for s, _ in spans],
        "extractant": [df["extractant"].iloc[s] for s, _ in spans],
        "m": m_of_unit.astype(int),
        "n_metals": n_metals_col,
        "complete_graph": complete,
        "a_obs": a_obs, "a_pred": a_pred,
        "sse_amp": sse_amp, "sse_shape": sse_shape, "sse_tot": sse_tot,
        "sse_pairtot": sse_pairtot, "nonint_pred": nonint_pred,
        "sse_pred": sse_pred, "recon_res": recon_res,
    })
    return df, units


rows_share, rows_mae, rows_gain, rows_anchor = [], [], [], []
gain_curves = []

for arm in ARMS:
    df, units = run_arm(arm)

    # --- 1. reconstruction / integrability sanity -------------------------
    max_recon = units["recon_res"].max()
    nonint_share = units["nonint_pred"].sum() / units["sse_pred"].sum()
    n_incomplete = int((~units["complete_graph"]).sum())

    # --- 2. amplitude / shape share of pairwise SSE -----------------------
    tot = units["sse_tot"].sum()
    amp_share = units["sse_amp"].sum() / tot
    shp_share = units["sse_shape"].sum() / tot
    pair_sse = units["sse_pairtot"].sum()

    sub4 = units[units["m"] >= 4]
    amp_share4 = sub4["sse_amp"].sum() / sub4["sse_tot"].sum()
    two = units[units["m"] == 2]

    # --- 3. extractant-macro MAE -----------------------------------------
    df["e_base"] = (df["prediction"] - df["y"]).abs()
    df["e_amp"] = (df["amp_exact"] - df["y"]).abs()
    df["e_shp"] = (df["shape_exact"] - df["y"]).abs()
    mae_base, _ = macro_mae(df, "e_base")
    mae_amp, _ = macro_mae(df, "e_amp")
    mae_shp, _ = macro_mae(df, "e_shp")
    pooled_base = float(df["e_base"].mean())

    # --- 4. amplitude dispersion -----------------------------------------
    sd_obs = float(units["a_obs"].std(ddof=1))
    sd_pred = float(units["a_pred"].std(ddof=1))

    # --- 5. oracle global gain -------------------------------------------
    seed = df["split_seed"].values
    extr = df["extractant"].values
    yv = df["y"].values
    pv = df["prediction"].values
    curve = []
    for g in GAIN_GRID:
        curve.append((g, macro_mae_from_abs(seed, extr, np.abs(g * pv - yv))))
    curve = pd.DataFrame(curve, columns=["g", "macro_mae"])
    curve["arm"] = arm
    gain_curves.append(curve)
    best = curve.loc[curve["macro_mae"].idxmin()]
    gain_at_1 = float(curve.loc[np.isclose(curve["g"], 1.0), "macro_mae"].iloc[0])

    # --- 6. single-pair widest-gap anchor calibration ---------------------
    # one anchor pair per (seed, extractant): the largest |dZ|; the anchor pair
    # itself is removed from the evaluation set.
    d = df[["split_seed", "extractant", "cell_id", "dZ", "y", "prediction"]].copy()
    d["absdZ"] = d["dZ"].abs()
    idx_anchor = d.sort_values(["split_seed", "extractant", "absdZ"],
                               ascending=[True, True, False]) \
                   .groupby(["split_seed", "extractant"], sort=False).head(1).index
    anch = d.loc[idx_anchor, ["split_seed", "extractant", "y", "prediction"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        gfit = np.where(np.abs(anch["prediction"].values) > 1e-9,
                        anch["y"].values / anch["prediction"].values, 1.0)
    anch = anch.assign(g_raw=gfit)
    res_anchor = {}
    for lo, hi in [(0.5, 2.0), (0.8, 1.25), (0.25, 4.0)]:
        gg = anch.assign(g=np.clip(anch["g_raw"], lo, hi))[
            ["split_seed", "extractant", "g"]]
        ev = d.drop(index=idx_anchor).merge(gg, on=["split_seed", "extractant"], how="left")
        ev["g"] = ev["g"].fillna(1.0)
        base = macro_mae_from_abs(ev["split_seed"].values, ev["extractant"].values,
                                  np.abs(ev["prediction"].values - ev["y"].values))
        cal = macro_mae_from_abs(ev["split_seed"].values, ev["extractant"].values,
                                 np.abs(ev["g"].values * ev["prediction"].values
                                        - ev["y"].values))
        res_anchor[(lo, hi)] = (base, cal, base - cal, len(ev))
        rows_anchor.append(dict(arm=arm, clip_lo=lo, clip_hi=hi, n_eval_pairs=len(ev),
                                macro_mae_base=base, macro_mae_calibrated=cal,
                                gain=base - cal))

    rows_share.append(dict(
        arm=arm, n_pairs=len(df), n_units=len(units), n_incomplete_units=n_incomplete,
        max_recon_resid=max_recon, nonintegrable_share_of_pred_sse=nonint_share,
        pairwise_sse=pair_sse, integrable_err_sse=tot,
        amplitude_share=amp_share, shape_share=shp_share,
        amplitude_share_m_ge_4=amp_share4, n_units_m_ge_4=len(sub4),
        n_units_m_eq_2=len(two), sse_share_m_eq_2=two["sse_tot"].sum() / tot,
        sd_a_obs=sd_obs, sd_a_pred=sd_pred, sd_ratio=sd_obs / sd_pred))
    rows_mae.append(dict(arm=arm, macro_mae_baseline=mae_base,
                         pooled_mae_baseline=pooled_base,
                         macro_mae_amplitude_exact=mae_amp,
                         macro_mae_shape_exact=mae_shp,
                         gain_amplitude=mae_base - mae_amp,
                         gain_shape=mae_base - mae_shp,
                         pct_removed_amplitude=100 * (mae_base - mae_amp) / mae_base))
    rows_gain.append(dict(arm=arm, macro_mae_g1=gain_at_1, best_g=float(best["g"]),
                          macro_mae_best_g=float(best["macro_mae"]),
                          oracle_gain=gain_at_1 - float(best["macro_mae"])))

    print(f"\n=== {arm}")
    print(f"  pairs={len(df)} units={len(units)} incomplete_units={n_incomplete}")
    print(f"  max|P c_obs - y| = {max_recon:.3e}   nonintegrable/pred SSE = {nonint_share:.3e}")
    print(f"  amplitude share = {100*amp_share:.2f}%   shape share = {100*shp_share:.2f}%"
          f"   (m>=4: {100*amp_share4:.2f}%, {len(sub4)} units)")
    print(f"  macro MAE base={mae_base:.4f} (pooled {pooled_base:.4f})"
          f"  amp-exact={mae_amp:.4f}  shape-exact={mae_shp:.4f}")
    print(f"  sd(a_obs)={sd_obs:.3f} sd(a_pred)={sd_pred:.3f} ratio={sd_obs/sd_pred:.3f}")
    print(f"  oracle best g={best['g']:.2f} macroMAE={best['macro_mae']:.4f}"
          f"  gain={gain_at_1 - float(best['macro_mae']):.4f}")
    for k, v in res_anchor.items():
        print(f"  anchor clip {k}: {v[0]:.4f} -> {v[1]:.4f}  (+{v[2]:.4f}) on {v[3]} pairs")

share = pd.DataFrame(rows_share)
mae = pd.DataFrame(rows_mae)
gain = pd.DataFrame(rows_gain)
anchor = pd.DataFrame(rows_anchor)
share.to_csv(os.path.join(OUT, "v_d4_shares.csv"), index=False)
mae.to_csv(os.path.join(OUT, "v_d4_macro_mae.csv"), index=False)
gain.to_csv(os.path.join(OUT, "v_d4_gain_oracle.csv"), index=False)
anchor.to_csv(os.path.join(OUT, "v_d4_anchor.csv"), index=False)
pd.concat(gain_curves).to_csv(os.path.join(OUT, "v_d4_gain_sweep.csv"), index=False)

# ------------------------------------------------- replicate-noise floor
coh = pd.read_parquet(os.path.join(ROOT, "gen13_separation/manifests/cohort_exact.parquet"))
rep = [c for c in coh.columns if c.startswith("repsd__")]
vals = coh[rep].stack().dropna()
med = float(vals.median())
print(f"\ncohort repsd: n={len(vals)} median logD sd={med:.4f}"
      f"  pair sd={med*np.sqrt(2):.4f}  Gaussian pair MAE={0.7978845608*med*np.sqrt(2):.4f}")
print(f"using programme 0.237: pair MAE={0.7978845608*0.237*np.sqrt(2):.4f}")
pd.DataFrame([dict(n_repsd=len(vals), median_logD_sd=med,
                   pair_sd=med * np.sqrt(2),
                   gauss_pair_mae=0.7978845608 * med * np.sqrt(2),
                   gauss_pair_mae_at_0237=0.7978845608 * 0.237 * np.sqrt(2))]
             ).to_csv(os.path.join(OUT, "v_d4_noise_floor.csv"), index=False)

print("\n---- summary ----")
print(share[["arm", "amplitude_share", "shape_share", "sd_ratio"]].to_string(index=False))
print(mae.to_string(index=False))
print(gain.to_string(index=False))
