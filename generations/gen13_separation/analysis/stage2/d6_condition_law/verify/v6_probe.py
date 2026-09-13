"""Follow-up probes: (a) what the '0.050 / 0.054 noise floor' actually is,
(b) sensitivity of the rank-2 oracle decomposition to how the extractant reference is formed,
(c) the 1-NN condition-fingerprint accuracies and their chance rates.
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
_r = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES])
RZ = pd.Series((_r - _r.mean()) / _r.std(), index=list(LANTHANIDES))
ARMS = ["C_DIRECT_ROW", "M_SELECTED", "M_PHYSICS_radius+radius_sq", "M_LOWRANK_K2",
        "X_ENS_DIRECT+LOWRANK_K2", "X_ENS_DIRECT+PHYSICS", "B1_MEAN_CURVE", "B3_NN_TANIMOTO"]

coh = pd.read_parquet(ROOT / "generations/gen13_separation/manifests/cohort_exact.parquet")
amp = pd.read_csv(OUT / "v6_cell_amplitudes.csv")
ana = amp[(amp.n_metals >= 4) & (amp.rz_span >= 1.5)]
cnt = ana.extractant.value_counts()
sub = ana[ana.extractant.map(cnt) >= 3]

# --------------------------------------------------------------- (a) what is the noise floor?
print("=== (a) amplitude standard error: three constructions, 182-cell set ===")
print(f"  residual SE of the quadratic fit   median {sub.se_amp_resid.median():.4f}  "
      f"rms {np.sqrt((sub.se_amp_resid**2).mean()):.4f}")
print(f"  replicate-noise propagated SE      median {sub.se_amp_meas.median():.4f}  "
      f"rms {np.sqrt((sub.se_amp_meas**2).mean()):.4f}")
rows = []
for sd0, tag in [(0.237, "cohort median repsd 0.237"), (0.302, "frozen-cohort repsd 0.302")]:
    ses = []
    for _, c in coh.set_index("cell_id").loc[sub.cell_id].reset_index().iterrows():
        mets = [m for m in LANTHANIDES if pd.notna(c[f"logD__{m}"])]
        x = RZ.loc[mets].to_numpy()
        X = np.column_stack([np.ones(len(x)), x, x ** 2])
        ses.append(sd0 * np.sqrt(np.linalg.pinv(X.T @ X)[1, 1]))
    ses = np.array(ses)
    print(f"  flat sd={sd0} propagated           median {np.median(ses):.4f}  "
          f"rms {np.sqrt((ses**2).mean()):.4f}   -> 0.377/median = {0.3772/np.median(ses):.2f}x")
    rows.append(dict(construction=tag, median_se=np.median(ses), rms_se=np.sqrt((ses**2).mean()),
                     ratio_rms_to_median=0.3772 / np.median(ses)))
rows.append(dict(construction="quadratic-fit residual SE (what the summary used)",
                 median_se=sub.se_amp_resid.median(),
                 rms_se=float(np.sqrt((sub.se_amp_resid**2).mean())),
                 ratio_rms_to_median=0.3772 / sub.se_amp_resid.median()))
rows.append(dict(construction="per-cell repsd propagated",
                 median_se=sub.se_amp_meas.median(),
                 rms_se=float(np.sqrt((sub.se_amp_meas**2).mean())),
                 ratio_rms_to_median=0.3772 / sub.se_amp_meas.median()))
pd.DataFrame(rows).to_csv(OUT / "v6_n1_noise_floor_variants.csv", index=False)

# ------------------------------------------- (b) oracle sensitivity to the extractant reference
nacid = ana.groupby("extractant").acid_M.nunique()
acid_extr = sorted(set(cnt[cnt >= 3].index) & set(nacid[nacid >= 2].index))
cells_keep = set(ana[ana.extractant.isin(acid_extr)].cell_id)
print(f"\n=== (b) oracle variants, {len(cells_keep)} cells / {len(acid_extr)} extractants ===")


def pair_design(A, B):
    a, b = RZ.reindex(A).to_numpy(), RZ.reindex(B).to_numpy()
    return np.column_stack([a - b, a ** 2 - b ** 2])


res = []
for arm in ARMS:
    p = pd.read_parquet(ROOT / f"generations/gen13_separation/predictions/B_primary/{arm}.parquet")
    p = p[p.cell_id.isin(cells_keep)].copy()
    U = pair_design(p.A, p.B)
    p["u1"], p["u2"] = U[:, 0], U[:, 1]
    p["resid"] = p.y - p.prediction
    acc = {k: [] for k in ["base", "w2_pooled", "w2_cellmean", "e2_pooled", "e2_cellmean",
                           "w1_sub_of_rank2", "w1_own"]}
    for seed, ps in p.groupby("split_seed"):
        ps = ps.copy()
        cb, idx = {}, {}
        for cid, g in ps.groupby("cell_id"):
            Ug = g[["u1", "u2"]].to_numpy()
            cb[cid] = np.linalg.lstsq(Ug, g.resid.to_numpy(), rcond=None)[0]
            idx[cid] = g.extractant.iloc[0]
        cbdf = pd.DataFrame(cb).T
        cbdf.columns = ["b1", "b2"]
        cbdf["extractant"] = [idx[i] for i in cbdf.index]
        # reference 1: pooled LS over all the extractant's pairs (pair-weighted)
        eb_pooled = {e: np.linalg.lstsq(g[["u1", "u2"]].to_numpy(), g.resid.to_numpy(),
                                        rcond=None)[0] for e, g in ps.groupby("extractant")}
        # reference 2: unweighted mean of the per-cell coefficient vectors (cell-weighted)
        eb_mean = cbdf.groupby("extractant")[["b1", "b2"]].mean().to_dict("index")
        C = np.array([cb[c] for c in ps.cell_id])
        EP = np.array([eb_pooled[e] for e in ps.extractant])
        EM = np.array([[eb_mean[e]["b1"], eb_mean[e]["b2"]] for e in ps.extractant])
        Uk = ps[["u1", "u2"]].to_numpy()
        # rank-1 done two ways
        cb1 = {cid: np.linalg.lstsq(g[["u1"]].to_numpy(), g.resid.to_numpy(), rcond=None)[0]
               for cid, g in ps.groupby("cell_id")}
        eb1 = {e: np.linalg.lstsq(g[["u1"]].to_numpy(), g.resid.to_numpy(), rcond=None)[0]
               for e, g in ps.groupby("extractant")}
        c1 = np.array([cb1[c][0] for c in ps.cell_id])
        e1 = np.array([eb1[e][0] for e in ps.extractant])
        ex = ps.extractant.to_numpy()

        def macro(err):
            return float(pd.DataFrame({"x": ex, "e": err}).groupby("x").e.mean().mean())

        r = ps.resid.to_numpy()
        acc["base"].append(macro(np.abs(r)))
        acc["w2_pooled"].append(macro(np.abs(r - ((Uk * C).sum(1) - (Uk * EP).sum(1)))))
        acc["w2_cellmean"].append(macro(np.abs(r - ((Uk * C).sum(1) - (Uk * EM).sum(1)))))
        acc["e2_pooled"].append(macro(np.abs(r - (Uk * EP).sum(1))))
        acc["e2_cellmean"].append(macro(np.abs(r - (Uk * EM).sum(1))))
        # rank-1 as the AMPLITUDE SLICE of the rank-2 fit (curvature left uncorrected)
        acc["w1_sub_of_rank2"].append(
            macro(np.abs(r - (Uk[:, 0] * (C[:, 0] - EM[:, 0])))))
        # rank-1 as its own best 1-D projection
        acc["w1_own"].append(macro(np.abs(r - Uk[:, 0] * (c1 - e1))))
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    res.append(dict(arm=arm, base=m["base"],
                    gain_within2_pooled_ref=m["base"] - m["w2_pooled"],
                    gain_within2_cellmean_ref=m["base"] - m["w2_cellmean"],
                    gain_extr2_pooled_ref=m["base"] - m["e2_pooled"],
                    gain_extr2_cellmean_ref=m["base"] - m["e2_cellmean"],
                    gain_within1_amp_slice=m["base"] - m["w1_sub_of_rank2"],
                    gain_within1_own=m["base"] - m["w1_own"]))
    print(f"  {arm:28s} base {m['base']:.3f} | within2 pooled {res[-1]['gain_within2_pooled_ref']:+.3f} "
          f"cellmean {res[-1]['gain_within2_cellmean_ref']:+.3f} | extr2 pooled "
          f"{res[-1]['gain_extr2_pooled_ref']:+.3f} cellmean {res[-1]['gain_extr2_cellmean_ref']:+.3f} "
          f"| rank1 slice {res[-1]['gain_within1_amp_slice']:+.3f}")
R = pd.DataFrame(res)
R.to_csv(OUT / "v6_n3_oracle_variants.csv", index=False)
print(f"  MEANS: within2 pooled {R.gain_within2_pooled_ref.mean():+.4f}, "
      f"within2 cellmean {R.gain_within2_cellmean_ref.mean():+.4f} (claim +0.030); "
      f"extr2 pooled {R.gain_extr2_pooled_ref.mean():+.4f}, cellmean "
      f"{R.gain_extr2_cellmean_ref.mean():+.4f} (claim +0.245)")
print(f"  rank-1 amplitude-slice gain: mean {R.gain_within1_amp_slice.mean():+.4f}, "
      f"C_DIRECT_ROW {R.set_index('arm').loc['C_DIRECT_ROW','gain_within1_amp_slice']:+.4f} "
      f"(claim -0.009)")

# --------------------------------------------------- (c) 1-NN condition fingerprint, all 521
print("\n=== (c) 1-NN leave-one-out on the 64 standardised cond__ columns, all 521 cells ===")
cc = [c for c in coh.columns if c.startswith("cond__")]
X = coh[cc].astype(float).to_numpy()
X = np.nan_to_num(X, nan=0.0)
sd = X.std(0)
X = (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)
D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(D, np.inf)
nn = D.argmin(1)
fp = []
for lab in ["publication_id", "chemotype", "extractant"]:
    v = coh[lab].to_numpy()
    acc = float((v[nn] == v).mean())
    n = pd.Series(v).value_counts().to_numpy().astype(float)
    chance = float((n * (n - 1)).sum() / (len(v) * (len(v) - 1)))
    print(f"  {lab:15s} 1-NN LOO accuracy {100*acc:5.1f}%   chance {100*chance:5.1f}%")
    fp.append(dict(label=lab, nn_accuracy=acc, chance=chance))
pd.DataFrame(fp).to_csv(OUT / "v6_n4_fingerprint.csv", index=False)

# duplicate condition vectors
key = coh[cc].astype(float).round(9).astype(str).agg("|".join, axis=1)
grp = coh.assign(k=key).groupby("k")
dups = grp.filter(lambda g: len(g) > 1)
ng = dups.assign(k=key.loc[dups.index]).groupby("k").extractant.nunique()
print(f"  cells sharing an identical 64-col condition vector: {len(dups)} (claim 95) in "
      f"{len(ng)} groups (claim 28); groups with >1 extractant: {int((ng > 1).sum())}")
