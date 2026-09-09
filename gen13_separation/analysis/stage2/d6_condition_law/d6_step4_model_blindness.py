"""D6 step 4: do the B_primary arms move the curve when the conditions move?

For each held-out cell we rebuild a predicted centred curve from the arm's pairwise
predictions (the pair graph is complete, so the sum-zero least-squares curve is
c_m = mean_k pred(m,k)) and take its rank-1 amplitude the same way as for the
observations.  Then, within an extractant that was measured at several acid
concentrations, we compare the observed spread of amplitude across its cells with the
predicted spread.
"""
from __future__ import annotations
import glob
import os
import numpy as np
import pandas as pd
from d6_common import (load_cohort, cell_amplitudes, predicted_curves, RZ, OUT, FIG,
                       ATOMIC_NUMBER,
                       PRED_DIR, MIN_RZ_SPAN)

ARMS = ["C_DIRECT_ROW", "M_SELECTED", "X_ENS_DIRECT+LOWRANK_K2", "X_ENS_DIRECT+PHYSICS",
        "M_LOWRANK_K2", "M_PHYSICS_radius+radius_sq", "B1_MEAN_CURVE", "B3_NN_TANIMOTO"]

df = load_cohort()
obs = df.merge(cell_amplitudes(df), on="cell_id")
obs = obs[obs.rz_span >= MIN_RZ_SPAN].copy()
obs["log_acid"] = np.log10(obs["cond__acid_concentration_M"].astype(float))

# extractants with >= 3 analysis cells spanning >= 2 acid concentrations
grp = obs.dropna(subset=["log_acid"]).groupby("extractant")
QUAL = [k for k, g in grp if len(g) >= 3 and g.log_acid.nunique() >= 2]
qual = obs[obs.extractant.isin(QUAL)].dropna(subset=["log_acid"]).copy()
print(f"qualifying extractants: {len(QUAL)}, cells: {len(qual)}, "
      f"chemotypes: {qual.chemotype.nunique()}")

# pair-geometry factor: an amplitude error da becomes a log-SF error da*|rz(A)-rz(B)|
pairfac = []
for _, r in qual.iterrows():
    ms = r.metal_set.split(",")
    v = [abs(RZ[a] - RZ[b]) for i, a in enumerate(ms) for b in ms[i + 1:]]
    pairfac.append(np.mean(v))
qual["mean_abs_drz"] = pairfac
PF = float(np.mean(pairfac))
print(f"mean |rz(A)-rz(B)| over the pairs of these cells = {PF:.3f} "
      f"(so 1 amplitude unit = {PF:.3f} log-SF units of average pairwise error)")

rows, per_ext = [], []
for arm in ARMS:
    path = os.path.join(PRED_DIR, arm + ".parquet")
    if not os.path.exists(path):
        print(f"  [missing arm {arm}]")
        continue
    pred = pd.read_parquet(path)
    pred = pred[pred.cell_id.isin(set(qual.cell_id))]
    pc = predicted_curves(pred)
    pc = pc[pc.rz_span >= MIN_RZ_SPAN]
    m = pc.merge(qual[["cell_id", "extractant", "extractant_name", "chemotype",
                       "amp", "log_acid", "mean_abs_drz"]], on="cell_id")
    # per (extractant, split_seed): spread of amplitude across the extractant's cells
    recs = []
    for (k, seed), g in m.groupby(["extractant", "split_seed"]):
        if len(g) < 3:
            continue
        recs.append(dict(extractant=k, extractant_name=g.extractant_name.iloc[0],
                         chemotype=g.chemotype.iloc[0], split_seed=seed, n_cells=len(g),
                         sd_obs=float(g.amp.std(ddof=1)),
                         sd_pred=float(g.amp_pred.std(ddof=1)),
                         range_obs=float(g.amp.max() - g.amp.min()),
                         range_pred=float(g.amp_pred.max() - g.amp_pred.min()),
                         corr=float(np.corrcoef(g.amp, g.amp_pred)[0, 1])
                         if g.amp_pred.std() > 1e-9 and g.amp.std() > 1e-9 else np.nan,
                         # slope of predicted amplitude on observed amplitude
                         slope_pred_on_obs=float(np.polyfit(g.amp, g.amp_pred, 1)[0])
                         if g.amp.std() > 1e-9 else np.nan,
                         acid_slope_obs=float(np.polyfit(g.log_acid, g.amp, 1)[0])
                         if g.log_acid.std() > 1e-9 else np.nan,
                         acid_slope_pred=float(np.polyfit(g.log_acid, g.amp_pred, 1)[0])
                         if g.log_acid.std() > 1e-9 else np.nan,
                         # within-extractant amplitude error the arm actually makes
                         rms_within_err=float(np.sqrt(
                             (((g.amp - g.amp.mean()) - (g.amp_pred - g.amp_pred.mean())) ** 2).mean())),
                         mean_abs_drz=float(g.mean_abs_drz.mean())))
    R = pd.DataFrame(recs)
    if R.empty:
        continue
    R["arm"] = arm
    per_ext.append(R)
    # extractant-macro: average over extractants first, then over the 5 split seeds
    bysd = R.groupby(["extractant", "split_seed"]).first().reset_index()
    macro = bysd.groupby("split_seed").mean(numeric_only=True).mean()
    ratio_sd = bysd.groupby("split_seed").apply(
        lambda g: (g.sd_pred / g.sd_obs).mean(), include_groups=False).mean()
    rows.append(dict(arm=arm, n_extractants=int(R.extractant.nunique()),
                     n_cells=int(m.cell_id.nunique()),
                     sd_obs=macro.sd_obs, sd_pred=macro.sd_pred,
                     sd_ratio_pred_over_obs=float(ratio_sd),
                     range_obs=macro.range_obs, range_pred=macro.range_pred,
                     corr_obs_pred=macro["corr"],
                     slope_pred_on_obs=macro.slope_pred_on_obs,
                     acid_slope_obs=macro.acid_slope_obs,
                     acid_slope_pred=macro.acid_slope_pred,
                     rms_within_amp_error=macro.rms_within_err,
                     rms_within_amp_error_in_logSF=macro.rms_within_err * PF))

summary = pd.DataFrame(rows)
summary.to_csv(OUT + "/d6_step4_amplitude_spread_by_arm.csv", index=False)
pd.concat(per_ext, ignore_index=True).to_csv(
    OUT + "/d6_step4_amplitude_spread_per_extractant.csv", index=False)

print("\n=== observed vs predicted spread of the rank-1 amplitude WITHIN an extractant "
      "(extractant-macro over the 5 split seeds) ===")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

best = summary.sort_values("sd_ratio_pred_over_obs")
print("\n=== the gap, stated plainly ===")
for _, r in summary.iterrows():
    print(f"  {r.arm:30s} observed within-extractant amplitude sd {r.sd_obs:.3f}, "
          f"predicted {r.sd_pred:.3f} ({100*r.sd_ratio_pred_over_obs:.0f}% of it); "
          f"corr(obs, pred) = {r.corr_obs_pred:+.3f}; "
          f"observed d(amp)/d(log10 acid) {r.acid_slope_obs:+.3f} vs predicted "
          f"{r.acid_slope_pred:+.3f}")

# --- per-extractant detail for the headline arm --------------------------------------
pe = pd.concat(per_ext, ignore_index=True)
for arm in ["C_DIRECT_ROW", "X_ENS_DIRECT+LOWRANK_K2"]:
    d = pe[pe.arm == arm]
    if d.empty:
        continue
    agg = d.groupby("extractant_name").agg(
        chemotype=("chemotype", "first"), n_cells=("n_cells", "mean"),
        sd_obs=("sd_obs", "mean"), sd_pred=("sd_pred", "mean"),
        corr=("corr", "mean"), acid_slope_obs=("acid_slope_obs", "mean"),
        acid_slope_pred=("acid_slope_pred", "mean")).sort_values("n_cells",
                                                                 ascending=False)
    agg["sd_ratio"] = agg.sd_pred / agg.sd_obs
    print(f"\n=== per-extractant, arm {arm} ===")
    print(agg.to_string(float_format=lambda v: f"{v:.3f}"))

# --- oracle decomposition: how much is the CONDITION part actually worth? ------------
# Split the curve-shape error of an arm into
#   LEVEL   - the extractant's average curve (a chemotype-transfer problem), and
#   WITHIN  - how the curve moves from cell to cell of that extractant (the condition
#             part, the subject of this analysis).
# Correct the arm's pairwise predictions with each in turn.  rank-1 = amplitude only,
# rank-2 = amplitude and curvature together (the curve's two shape coefficients).
def shape_coefs(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, cid), g in pred.groupby(["split_seed", "cell_id"], sort=False):
        ms = sorted(set(g.A) | set(g.B), key=lambda m: ATOMIC_NUMBER[m])
        n = len(ms)
        idx = {m: i for i, m in enumerate(ms)}
        D = np.zeros((n, n))
        a = g.A.map(idx).to_numpy()
        b = g.B.map(idx).to_numpy()
        D[a, b] = g.prediction.to_numpy()
        D[b, a] = -g.prediction.to_numpy()
        cp = D.sum(1) / n
        x = np.array([RZ[m] for m in ms])
        X = np.column_stack([np.ones(n), x, x ** 2])
        bp, *_ = np.linalg.lstsq(X, cp - cp.mean(), rcond=None)
        rows.append(dict(split_seed=seed, cell_id=cid, ap=float(bp[1]),
                         qp=float(bp[2]), rz_span=float(x.max() - x.min())))
    return pd.DataFrame(rows)


print("\n=== oracle decomposition of the curve-shape error, per arm ===")
print("    (macro = extractant-macro MAE in log-SF units on the qualifying cells)")
orc = []
for arm in ARMS:
    pred = pd.read_parquet(os.path.join(PRED_DIR, arm + ".parquet"))
    pred = pred[pred.cell_id.isin(set(qual.cell_id))].copy()
    P = shape_coefs(pred)
    P = P[P.rz_span >= MIN_RZ_SPAN].merge(qual[["cell_id", "amp", "quad"]], on="cell_id")
    p = pred.merge(P.drop(columns=["rz_span"]), on=["split_seed", "cell_id"])
    for c in ("ap", "qp"):
        p[c + "_dev"] = p[c] - p.groupby(["split_seed", "extractant"])[c].transform("mean")
    for c in ("amp", "quad"):
        p[c + "_dev"] = p[c] - p.groupby("extractant")[c].transform("mean")
    p["d1"] = p.A.map(RZ) - p.B.map(RZ)
    p["d2"] = p.A.map(RZ) ** 2 - p.B.map(RZ) ** 2

    def macro(adj):
        e = (p.prediction + adj - p.y).abs()
        return float(e.groupby([p.split_seed, p.extractant]).mean()
                     .groupby(level=0).mean().mean())

    z = 0.0 * p.d1
    row = dict(arm=arm, n_cells=int(p.cell_id.nunique()),
               n_extractants=int(p.extractant.nunique()),
               macro_as_shipped=macro(z),
               macro_within_rank1=macro((p.amp_dev - p.ap_dev) * p.d1),
               macro_within_rank2=macro((p.amp_dev - p.ap_dev) * p.d1
                                        + (p.quad_dev - p.qp_dev) * p.d2),
               macro_level_rank2=macro(((p.amp - p.amp_dev) - (p.ap - p.ap_dev)) * p.d1
                                       + ((p.quad - p.quad_dev) - (p.qp - p.qp_dev)) * p.d2),
               macro_full_rank2=macro((p.amp - p.ap) * p.d1 + (p.quad - p.qp) * p.d2))
    row["gain_within_rank2"] = row["macro_as_shipped"] - row["macro_within_rank2"]
    row["gain_level_rank2"] = row["macro_as_shipped"] - row["macro_level_rank2"]
    row["gain_full_rank2"] = row["macro_as_shipped"] - row["macro_full_rank2"]
    row["gain_within_given_level"] = row["macro_level_rank2"] - row["macro_full_rank2"]
    orc.append(row)
    print(f"  {arm:26s} shipped {row['macro_as_shipped']:.3f} | "
          f"within-oracle rank1 {row['macro_within_rank1']:.3f} "
          f"rank2 {row['macro_within_rank2']:.3f} ({row['gain_within_rank2']:+.3f}) | "
          f"level-oracle {row['macro_level_rank2']:.3f} "
          f"({row['gain_level_rank2']:+.3f}) | both {row['macro_full_rank2']:.3f} "
          f"({row['gain_full_rank2']:+.3f})")
O = pd.DataFrame(orc)
n_ext_total = df.extractant.nunique()
O["gain_within_rank2_scaled_to_full_macro"] = \
    O.gain_within_rank2 * O.n_extractants / n_ext_total
O.to_csv(OUT + "/d6_step4_oracle_decomposition.csv", index=False)
print(f"\n  the within-extractant (condition) oracle is worth "
      f"{O.gain_within_rank2.mean():.3f} log units of extractant-macro MAE averaged "
      f"over {len(O)} arms, on {int(O.n_extractants.iloc[0])} of {n_ext_total} "
      f"extractants; spread over the whole 90-extractant macro that is "
      f"{O.gain_within_rank2_scaled_to_full_macro.mean():.3f}.")
print(f"  the extractant-LEVEL curve oracle is worth "
      f"{O.gain_level_rank2.mean():.3f} on the same cells - "
      f"{O.gain_level_rank2.mean()/max(O.gain_within_rank2.mean(),1e-9):.0f}x larger.")
print("\nwrote d6_step4_amplitude_spread_by_arm.csv, "
      "d6_step4_amplitude_spread_per_extractant.csv, "
      "d6_step4_oracle_decomposition.csv")
