"""The two direct transfer tests, both directions, with every guard the protocol demands.

E -> K :  gen14's deployed direction estimator, fitted on the 82 extraction ligands, asked for the
          sign of the aqueous logK slope of 273 independent ligands.
K -> E :  the same estimator fitted on the 273 logK ligands, asked for the extraction direction of
          the 82, scored with gen14's own unit (extractant) and block (chemotype).

Guards: the constant rule on each side; balanced accuracy and AUC, because the logK side is 72 %
heavy; the exact-structure overlap removed; accuracy split by Tanimoto distance to the other set;
leave-one-chemotype-out on the extraction side; a label-permutation null.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for p in (ROOT / "gen15_curve", ROOT / "gen13_separation", ROOT / "gen14_direction"):
    sys.path.insert(0, str(p))

from sklearn.impute import SimpleImputer                    # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.pipeline import make_pipeline                  # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402
from sklearn.metrics import roc_auc_score                   # noqa: E402
from scipy.stats import spearmanr                           # noqa: E402

from gen13sep.models import chemotype_balanced_weights      # noqa: E402
from gen14.dirbench import Blocked                          # noqa: E402

E = pd.read_parquet(HERE / "data" / "side_E_extraction.parquet")
K = pd.read_parquet(HERE / "data" / "side_K_logk.parquet")
TOPO = json.loads((HERE / "data" / "prep_meta.json").read_text())["topo39_columns"]
SHARED = set(E.smiles) & set(K.smiles)
RNG = np.random.default_rng(20260909)


def fit_logistic(X, y, w, C: float = 1.0):
    """gen14's deployed estimator: median impute -> standardise -> L2 logistic, C = 1."""
    m = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                      StandardScaler(),
                      LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
    m.fit(X, y, logisticregression__sample_weight=w)
    return m


def basic(y, p, w=None) -> dict:
    yh = (np.asarray(p) >= 0.5).astype(int)
    y = np.asarray(y)
    acc = float((yh == y).mean())
    per_class = {int(c): float((yh[y == c] == c).mean()) for c in np.unique(y)}
    bal = float(np.mean(list(per_class.values())))
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    maj = float(max(np.mean(y == 0), np.mean(y == 1)))
    return {"n": int(len(y)), "accuracy": acc, "balanced_accuracy": bal, "auc": auc,
            "majority_rule_accuracy": maj, "base_rate_heavy": float(y.mean()),
            "acc_minus_majority": acc - maj, "per_class_recall": per_class,
            "pred_frac_heavy": float(yh.mean())}


res: dict = {}

# ======================================================================================
# E -> K
# ======================================================================================
Xe = E[TOPO].to_numpy(float)
Xk = K[TOPO].to_numpy(float)
ye = E.y_heavy.to_numpy(int)
yk = K.y_heavy.to_numpy(int)
we = chemotype_balanced_weights(E.chemotype.to_numpy())      # gen14's weighting, extractant units

m_e = fit_logistic(Xe, ye, we)
p_k = m_e.predict_proba(Xk)[:, 1]

res["E_to_K"] = {"all": basic(yk, p_k)}
res["E_to_K"]["all"]["spearman_p_vs_slope"] = float(spearmanr(p_k, K.slope).statistic)

# unweighted training variant (does the chemotype balancing carry the result?)
p_k_uw = fit_logistic(Xe, ye, np.ones(len(ye))).predict_proba(Xk)[:, 1]
res["E_to_K"]["unweighted_train"] = basic(yk, p_k_uw)

# drop the 5 structures present on both sides
mk = ~K.smiles.isin(SHARED).to_numpy()
res["E_to_K"]["exact_overlap_removed"] = basic(yk[mk], p_k[mk])

# split by Tanimoto distance to the nearest extraction ligand
tan = K.nn_tanimoto.to_numpy(float)
bins = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.6), (0.6, 1.01)]
res["E_to_K"]["by_tanimoto"] = [
    dict(lo=lo, hi=hi, **basic(yk[s], p_k[s])) for lo, hi in bins
    if (s := (tan >= lo) & (tan < hi)).sum() >= 5]

# restrict to logK series that are well determined, the analogue of "rich" cells
for name, s in {"n_ge_10": K.n.to_numpy() >= 10,
                "r2_ge_0.8": K.r2.to_numpy() >= 0.8,
                "abs_slope_ge_0.2": np.abs(K.slope.to_numpy()) >= 0.2}.items():
    res["E_to_K"][f"subset_{name}"] = basic(yk[s], p_k[s])

# permutation null: shuffle the extraction labels, refit, re-score (200 draws)
perm = []
for _ in range(200):
    yp = RNG.permutation(ye)
    if len(set(yp)) < 2:
        continue
    perm.append(float(((fit_logistic(Xe, yp, we).predict_proba(Xk)[:, 1] >= 0.5).astype(int) == yk).mean()))
perm = np.array(perm)
res["E_to_K"]["permutation_null"] = {"mean": float(perm.mean()), "sd": float(perm.std()),
                                     "q95": float(np.quantile(perm, 0.95)),
                                     "p_one_sided": float((perm >= res["E_to_K"]["all"]["accuracy"]).mean())}

# leave-one-chemotype-out on the *training* side: does one extraction family carry it?
loco = []
for c in sorted(E.chemotype.unique()):
    s = E.chemotype.to_numpy() != c
    if len(set(ye[s])) < 2:
        continue
    pp = fit_logistic(Xe[s], ye[s], chemotype_balanced_weights(E.chemotype.to_numpy()[s])).predict_proba(Xk)[:, 1]
    loco.append(float(((pp >= 0.5).astype(int) == yk).mean()))
res["E_to_K"]["loco_train_accuracy"] = {"min": float(np.min(loco)), "max": float(np.max(loco)),
                                        "mean": float(np.mean(loco)), "n": len(loco)}

# ======================================================================================
# K -> E
# ======================================================================================
wk = np.ones(len(yk))
m_k = fit_logistic(Xk, yk, wk)
p_e = m_k.predict_proba(Xe)[:, 1]

hit = ((p_e >= 0.5).astype(int) == ye).astype(float)
unit = pd.DataFrame({"extractant": E.smiles, "chemotype": E.chemotype, "hit": hit})
boot = Blocked(unit.chemotype)
macro, draws = boot.draws(unit)

always_heavy = (np.ones(len(ye)) == ye).astype(float)
u_ah = unit.assign(hit=always_heavy)
macro_ah, draws_ah = boot.draws(u_ah)
d_unit = unit.assign(hit=hit - always_heavy)
gain, gdraws = boot.draws(d_unit)

res["K_to_E"] = {
    "pooled": basic(ye, p_e),
    "macro_accuracy": macro,
    "macro_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    "always_heavy_macro": macro_ah,
    "always_heavy_ci": [float(np.quantile(draws_ah, 0.025)), float(np.quantile(draws_ah, 0.975))],
    "gain_vs_always_heavy": gain,
    "gain_ci": [float(np.quantile(gdraws, 0.025)), float(np.quantile(gdraws, 0.975))],
    "gain_p_two_sided": float(2 * min((gdraws <= 0).mean(), (gdraws >= 0).mean())),
    "units_better": int((d_unit.hit > 0).sum()), "units_worse": int((d_unit.hit < 0).sum()),
    "n_chemotypes": int(unit.chemotype.nunique()),
    "spearman_p_vs_amp": float(spearmanr(p_e, E.amp).statistic),
}

me = ~E.smiles.isin(SHARED).to_numpy()
res["K_to_E"]["exact_overlap_removed_pooled"] = basic(ye[me], p_e[me])

# weight the logK training set so it is not dominated by one structural family:
# balance by the chemotype of each logK ligand's nearest extraction neighbour
wk2 = chemotype_balanced_weights(K.nn_chemotype.fillna("none").to_numpy())
p_e2 = fit_logistic(Xk, yk, wk2).predict_proba(Xe)[:, 1]
res["K_to_E"]["nnchemotype_balanced_train_pooled"] = basic(ye, p_e2)
h2 = ((p_e2 >= 0.5).astype(int) == ye).astype(float)
res["K_to_E"]["nnchemotype_balanced_train_macro"] = float(boot.draws(unit.assign(hit=h2))[0])

# class-balanced logK training (the logK side is 72 % heavy)
wk3 = np.where(yk == 1, 1.0 / yk.mean(), 1.0 / (1 - yk.mean()))
p_e3 = fit_logistic(Xk, yk, wk3).predict_proba(Xe)[:, 1]
res["K_to_E"]["class_balanced_train_pooled"] = basic(ye, p_e3)
h3 = ((p_e3 >= 0.5).astype(int) == ye).astype(float)
res["K_to_E"]["class_balanced_train_macro"] = float(boot.draws(unit.assign(hit=h3))[0])

# restrict the logK training set to well determined series
sK = (K.n.to_numpy() >= 10) & (K.r2.to_numpy() >= 0.8)
p_e4 = fit_logistic(Xk[sK], yk[sK], np.ones(int(sK.sum()))).predict_proba(Xe)[:, 1]
res["K_to_E"]["rich_logk_train_pooled"] = basic(ye, p_e4)
h4 = ((p_e4 >= 0.5).astype(int) == ye).astype(float)
res["K_to_E"]["rich_logk_train_macro"] = float(boot.draws(unit.assign(hit=h4))[0])
res["K_to_E"]["rich_logk_train_n"] = int(sK.sum())

# split by how close each extraction ligand is to the logK set
tane = E.nn_tanimoto.to_numpy(float)
res["K_to_E"]["by_tanimoto"] = [
    dict(lo=lo, hi=hi, **basic(ye[s], p_e[s])) for lo, hi in [(0.0, 0.3), (0.3, 0.4), (0.4, 1.01)]
    if (s := (tane >= lo) & (tane < hi)).sum() >= 5]

# permutation null on the logK labels
permE = []
for _ in range(200):
    yp = RNG.permutation(yk)
    hh = ((fit_logistic(Xk, yp, wk).predict_proba(Xe)[:, 1] >= 0.5).astype(int) == ye).astype(float)
    permE.append(float(boot.draws(unit.assign(hit=hh), "hit")[0]))
permE = np.array(permE)
res["K_to_E"]["permutation_null_macro"] = {"mean": float(permE.mean()), "sd": float(permE.std()),
                                           "q95": float(np.quantile(permE, 0.95)),
                                           "p_one_sided": float((permE >= macro).mean())}

# ======================================================================================
# a within-corpus reference for the same estimator: does it work AT ALL on its own data?
# 5-fold chemotype-grouped CV on side E, so the two transfer numbers have a ceiling to sit under.
# ======================================================================================
chem = E.chemotype.to_numpy()
uniq = np.array(sorted(set(chem)))
assign = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(uniq))}
blk = np.array([assign[g] for g in chem])
oof = np.zeros(len(ye))
for k in range(5):
    tr, te = blk != k, blk == k
    if len(set(ye[tr])) < 2 or te.sum() == 0:
        oof[te] = ye[tr].mean(); continue
    oof[te] = fit_logistic(Xe[tr], ye[tr], chemotype_balanced_weights(chem[tr])).predict_proba(Xe[te])[:, 1]
hcv = ((oof >= 0.5).astype(int) == ye).astype(float)
res["reference_within_E_chemotype_cv"] = {
    "pooled": basic(ye, oof),
    "macro_accuracy": float(boot.draws(unit.assign(hit=hcv))[0]),
}

# and the logK side's own within-set skill on the SAME 39 columns (its published 0.67 Spearman
# used 2048-bit ECFP + trees; this is what the 39 topology columns alone are worth there)
kf = np.array([i % 5 for i in np.random.default_rng(1).permutation(len(yk))])
oofk = np.zeros(len(yk))
for k in range(5):
    tr, te = kf != k, kf == k
    oofk[te] = fit_logistic(Xk[tr], yk[tr], np.ones(int(tr.sum()))).predict_proba(Xk[te])[:, 1]
res["reference_within_K_cv_on_TOPO39"] = basic(yk, oofk)

(HERE / "transfer.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=1))
