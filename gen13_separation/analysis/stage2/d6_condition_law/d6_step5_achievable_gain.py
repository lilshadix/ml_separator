"""D6 step 5: of the amplitude headroom found in step 4, how much is actually reachable?

Two correction heads are tested on top of the shipped arm predictions:
  GLOBAL  - one condition->amplitude-deviation law shared by all extractants, fitted
            leave-one-chemotype-out, so it is legal under the primary chemotype hold-out.
            At test time it only uses the conditions of the held-out cells (inputs, not
            labels), centring each extractant's conditions over its own test cells.
  LOCAL   - an extractant-specific law fitted on the OTHER cells of the same extractant
            (leave-one-cell-out).  Illegal under chemotype hold-out; it stands for a
            deployment where the extractant has already been measured at other acidities.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from d6_common import (load_cohort, cell_amplitudes, predicted_curves, ATOMIC_NUMBER,
                       RZ, OUT, PRED_DIR, MIN_RZ_SPAN)

ARMS = ["C_DIRECT_ROW", "M_SELECTED", "X_ENS_DIRECT+LOWRANK_K2"]
FEATS = ["log_acid", "log_ext", "temp"]

df = load_cohort()
obs = df.merge(cell_amplitudes(df), on="cell_id")
obs = obs[obs.rz_span >= MIN_RZ_SPAN].copy()
obs["log_acid"] = np.log10(obs["cond__acid_concentration_M"].astype(float))
obs["log_ext"] = np.log10(obs["cond__extractant_concentration_M"].astype(float))
obs["temp"] = obs["cond__temperature_C"].astype(float)
for c in FEATS:
    obs[c] = obs[c].fillna(obs.groupby("extractant")[c].transform("median"))
    obs[c] = obs[c].fillna(obs[c].median())

g = obs.dropna(subset=["log_acid"]).groupby("extractant")
QUAL = [k for k, gg in g if len(gg) >= 3 and gg.log_acid.nunique() >= 2]
q = obs[obs.extractant.isin(QUAL)].copy()
for c in FEATS + ["amp"]:
    q[c + "_dev"] = q[c] - q.groupby("extractant")[c].transform("mean")
print(f"{len(q)} cells, {q.extractant.nunique()} extractants, "
      f"{q.chemotype.nunique()} chemotypes")

q["quad_dev"] = q.quad - q.groupby("extractant").quad.transform("mean")
X = q[[c + "_dev" for c in FEATS]].to_numpy(float)
qi = q.reset_index(drop=True)
r2rows = []
for tgt in ("amp_dev", "quad_dev"):
    yv = qi[tgt].to_numpy(float)
    ss_tot = float((yv ** 2).sum())
    # GLOBAL head: leave-one-chemotype-out
    ghat = np.zeros(len(qi))
    coefs = []
    for ct in qi.chemotype.unique():
        tr = (qi.chemotype != ct).to_numpy()
        te = ~tr
        if tr.sum() < 5:
            continue
        b = np.linalg.pinv(X[tr].T @ X[tr] + 1e-6 * np.eye(X.shape[1])) @ (X[tr].T @ yv[tr])
        ghat[te] = X[te] @ b
        coefs.append(dict(target=tgt, held_out_chemotype=ct, n_train=int(tr.sum()),
                          **{f"beta_{c}": float(v) for c, v in zip(FEATS, b)}))
    qi[tgt + "_hat_global"] = ghat
    # LOCAL head: leave-one-cell-out inside the extractant
    lhat = np.zeros(len(qi))
    for k, gg in qi.groupby("extractant"):
        Xi = X[gg.index.to_numpy()]
        yi = yv[gg.index.to_numpy()]
        for j in range(len(gg)):
            m = np.ones(len(gg), bool)
            m[j] = False
            if m.sum() < 2:
                continue
            b = np.linalg.pinv(Xi[m].T @ Xi[m] + np.eye(Xi.shape[1])) @ (Xi[m].T @ yi[m])
            lhat[gg.index[j]] = float(Xi[j] @ b)
    qi[tgt + "_hat_local"] = lhat
    r2rows.append(dict(target=tgt, rms_dev=np.sqrt(ss_tot / len(yv)),
                       r2_global_out_of_chemotype=1 - float(((yv - ghat) ** 2).sum()) / ss_tot,
                       rms_after_global=np.sqrt(float(((yv - ghat) ** 2).sum()) / len(yv)),
                       r2_local_loo=1 - float(((yv - lhat) ** 2).sum()) / ss_tot,
                       rms_after_local=np.sqrt(float(((yv - lhat) ** 2).sum()) / len(yv))))
    if tgt == "amp_dev":
        print("\n=== GLOBAL condition->amplitude-deviation law, "
              "leave-one-chemotype-out (coefficients) ===")
        print(pd.DataFrame(coefs).to_string(index=False,
                                            float_format=lambda v: f"{v:.4f}"))
r2 = pd.DataFrame(r2rows)
r2.to_csv(OUT + "/d6_step5_head_r2.csv", index=False)
print("\n=== how well each head predicts the WITHIN-extractant shape deviation ===")
print(r2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

heads = qi[["cell_id", "extractant", "chemotype", "amp", "amp_dev", "quad", "quad_dev",
            "amp_dev_hat_global", "amp_dev_hat_local",
            "quad_dev_hat_global", "quad_dev_hat_local"]]
heads.to_csv(OUT + "/d6_step5_amplitude_heads.csv", index=False)

# --- apply each head to the shipped pairwise predictions ------------------------------
print("\n=== effect on the shipped extractant-macro MAE (qualifying cells only) ===")
print("    rank-2 correction: both shape coefficients are replaced by the head's")
res = []
for arm in ARMS:
    pred = pd.read_parquet(os.path.join(PRED_DIR, arm + ".parquet"))
    pred = pred[pred.cell_id.isin(set(qi.cell_id))].copy()
    sc = []
    for (seed, cid), gg in pred.groupby(["split_seed", "cell_id"], sort=False):
        ms = sorted(set(gg.A) | set(gg.B), key=lambda m: ATOMIC_NUMBER[m])
        n = len(ms)
        idx = {m: i for i, m in enumerate(ms)}
        D = np.zeros((n, n))
        a = gg.A.map(idx).to_numpy()
        b = gg.B.map(idx).to_numpy()
        D[a, b] = gg.prediction.to_numpy()
        D[b, a] = -gg.prediction.to_numpy()
        cp = D.sum(1) / n
        x = np.array([RZ[m] for m in ms])
        X2 = np.column_stack([np.ones(n), x, x ** 2])
        bp, *_ = np.linalg.lstsq(X2, cp - cp.mean(), rcond=None)
        sc.append(dict(split_seed=seed, cell_id=cid, ap=float(bp[1]), qp=float(bp[2])))
    p = pred.merge(pd.DataFrame(sc), on=["split_seed", "cell_id"]).merge(
        heads.drop(columns=["extractant", "chemotype"]), on="cell_id")
    for c in ("ap", "qp"):
        p[c + "_dev"] = p[c] - p.groupby(["split_seed", "extractant"])[c].transform("mean")
    p["d1"] = p.A.map(RZ) - p.B.map(RZ)
    p["d2"] = p.A.map(RZ) ** 2 - p.B.map(RZ) ** 2
    out = dict(arm=arm, n_cells=int(p.cell_id.nunique()),
               n_extractants=int(p.extractant.nunique()))
    for lab, suf in [("as_shipped", None), ("global_head", "_hat_global"),
                     ("local_head", "_hat_local"), ("oracle", "")]:
        if suf is None:
            e = (p.prediction - p.y).abs()
        else:
            e = (p.prediction + (p["amp_dev" + suf] - p.ap_dev) * p.d1
                 + (p["quad_dev" + suf] - p.qp_dev) * p.d2 - p.y).abs()
        mac = e.groupby([p.split_seed, p.extractant]).mean().groupby(
            level=0).mean().mean()
        out["mae_" + lab] = float(mac)
    for lab in ("global", "local", "oracle"):
        out["gain_" + lab] = out["mae_as_shipped"] - out[f"mae_{lab}_head"] \
            if lab != "oracle" else out["mae_as_shipped"] - out["mae_oracle"]
    res.append(out)
    print(f"  {arm:26s} shipped {out['mae_as_shipped']:.3f} | "
          f"global head {out['mae_global_head']:.3f} ({out['gain_global']:+.3f}) | "
          f"local head {out['mae_local_head']:.3f} ({out['gain_local']:+.3f}) | "
          f"within-oracle {out['mae_oracle']:.3f} ({out['gain_oracle']:+.3f})")
R = pd.DataFrame(res)
R.to_csv(OUT + "/d6_step5_achievable_gain.csv", index=False)

# --- dilute the same numbers to the whole 521-cell design ----------------------------
tot_cells = df.cell_id.nunique()
tot_ext = df.extractant.nunique()
print(f"\nthese gains apply to {R.n_cells.iloc[0]} of {tot_cells} cells "
      f"({R.n_extractants.iloc[0]} of {tot_ext} extractants). Extractants with a single "
      f"cell cannot show a within-extractant amplitude effect at all.")
print("wrote d6_step5_amplitude_heads.csv, d6_step5_achievable_gain.csv")
