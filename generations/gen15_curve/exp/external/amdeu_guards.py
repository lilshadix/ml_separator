"""Every guard the protocol demands, applied to the Am/Eu transfer signal.

The headline of ``amdeu.py`` is that the gen14 direction model's P(heavy-lanthanide-selective)
is ANTI-correlated with Am/Eu selectivity: Spearman -0.54, AUC 0.83 once the orientation is
flipped, in both transfer directions.  A sign-reversed transfer is exactly the kind of result that
can be manufactured by shared chemistry, so this script tries to kill it:

* exact structural overlap with the extraction set removed (9 ligands);
* accuracy and AUC split by Tanimoto distance to the nearest extraction ligand;
* a nearest-neighbour lookup baseline -- the sign of the Tanimoto-nearest extraction ligand's own
  amplitude, which needs no model at all.  If that matches the model, the transfer is neighbour
  lookup rather than a learned rule;
* leave-one-chemotype-out on the extraction training side;
* a soft-donor-only control (drop every S-containing ligand), since S-donors are both the classic
  Am-selective family and a distinctive topology;
* a label permutation null;
* the ligands with the most replicate rows only, to check the label is not measurement noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
for p in (ROOT / "generations" / "gen15_curve", ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction",
          ROOT / "generations" / "gen12_2_eu_pred"):
    sys.path.insert(0, str(p))

from sklearn.impute import SimpleImputer                    # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.pipeline import make_pipeline                  # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402
from sklearn.metrics import roc_auc_score                   # noqa: E402
from scipy.stats import spearmanr                           # noqa: E402

from gen122.coordination import build_table                 # noqa: E402
from gen13sep.models import chemotype_balanced_weights      # noqa: E402

TOPO = json.loads((HERE / "data" / "prep_meta.json").read_text())["topo39_columns"]
E = pd.read_parquet(HERE / "data" / "side_E_extraction.parquet")
L = pd.read_parquet(HERE / "data" / "side_A_amdeu.parquet")
L = L.join(build_table(L.canon.tolist())[TOPO].reindex(L.canon).reset_index(drop=True))

Xe = E[TOPO].to_numpy(float); ye = E.y_heavy.to_numpy(int)
we = chemotype_balanced_weights(E.chemotype.to_numpy())
Xa = L[TOPO].to_numpy(float); ya = L.y_am.to_numpy(int)
logsf = L.logSF.to_numpy(float)
tan = L.nn_tanimoto.to_numpy(float)
SHARED = set(L.canon) & set(E.smiles)
res: dict = {"n_ligands": int(len(L)), "n_exact_shared": len(SHARED)}


def pipe(C: float = 1.0):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))


def fit_E(X=Xe, y=ye, w=we):
    return pipe().fit(X, y, logisticregression__sample_weight=w)


def report(p_heavy, y, sf, tag) -> dict:
    """Scored on the FLIPPED orientation (P(Am) = 1 - P(heavy)), the one the data selects.

    The flip is declared once here and never re-chosen per subset, so it costs exactly one
    binary degree of freedom for the whole analysis; the permutation null below prices it.
    """
    p_am = 1.0 - np.asarray(p_heavy)
    yh = (p_am >= 0.5).astype(int)
    y = np.asarray(y)
    maj = float(max(np.mean(y == 0), np.mean(y == 1)))
    per = {int(c): float((yh[y == c] == c).mean()) for c in np.unique(y)}
    try:
        auc = float(roc_auc_score(y, p_am))
    except ValueError:
        auc = float("nan")
    return {"tag": tag, "n": int(len(y)), "accuracy_flipped": float((yh == y).mean()),
            "balanced_accuracy_flipped": float(np.mean(list(per.values()))),
            "auc_flipped": auc, "majority_rule": maj, "base_rate_Am": float(y.mean()),
            "spearman_pheavy_vs_logSF": float(spearmanr(p_heavy, sf).statistic)}


m = fit_E()
p_all = m.predict_proba(Xa)[:, 1]
res["headline"] = report(p_all, ya, logsf, "all 85 ligands")

# ---- 1. exact overlap removed -----------------------------------------------------------
keep = ~L.canon.isin(SHARED).to_numpy()
res["exact_overlap_removed"] = report(p_all[keep], ya[keep], logsf[keep],
                                      f"{int(keep.sum())} ligands, {len(SHARED)} shared removed")

# ---- 2. by Tanimoto distance -------------------------------------------------------------
res["by_tanimoto"] = []
for lo, hi in [(0.0, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 0.999), (0.999, 1.01)]:
    s = (tan >= lo) & (tan < hi)
    if s.sum() >= 8:
        r = report(p_all[s], ya[s], logsf[s], f"tanimoto [{lo},{hi})")
        r["lo"], r["hi"] = lo, hi
        res["by_tanimoto"].append(r)

# ---- 3. nearest-neighbour lookup baseline -------------------------------------------------
from rdkit import Chem, DataStructs, RDLogger                # noqa: E402
from rdkit.Chem import AllChem                               # noqa: E402
RDLogger.DisableLog("rdApp.*")


def fp(s):
    mol = Chem.MolFromSmiles(s)
    return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


fe = [fp(s) for s in E.smiles]
ok = [i for i, f in enumerate(fe) if f is not None]
nn_amp, nn_amp_excl = [], []
for s in L.canon:
    f = fp(s)
    sims = np.array(DataStructs.BulkTanimotoSimilarity(f, [fe[i] for i in ok]))
    nn_amp.append(float(E.amp.iat[ok[int(sims.argmax())]]))
    m2 = sims < 0.999                                   # exclude the identical structure
    nn_amp_excl.append(float(E.amp.iat[ok[int(np.flatnonzero(m2)[sims[m2].argmax()])]])
                       if m2.any() else np.nan)
nn_amp = np.array(nn_amp); nn_amp_excl = np.array(nn_amp_excl)
# a heavy-selective neighbour (amp < 0) -> "P(heavy) = 1"
res["nn_lookup_baseline"] = report((nn_amp < 0).astype(float), ya, logsf, "NN extractant amp sign")
res["nn_lookup_baseline_excl_identical"] = report(
    (nn_amp_excl < 0).astype(float), ya, logsf, "NN sign, identical structure excluded")
res["spearman_nn_amp_vs_logSF"] = float(spearmanr(nn_amp_excl, logsf).statistic)

# ---- 4. leave-one-chemotype-out on the extraction training side ----------------------------
loco = []
for c in sorted(E.chemotype.unique()):
    s = E.chemotype.to_numpy() != c
    if len(set(ye[s])) < 2:
        continue
    pp = fit_E(Xe[s], ye[s], chemotype_balanced_weights(E.chemotype.to_numpy()[s])).predict_proba(Xa)[:, 1]
    loco.append({"dropped": c, "auc_flipped": float(roc_auc_score(ya, 1 - pp)),
                 "spearman": float(spearmanr(pp, logsf).statistic)})
res["loco_train"] = {"n": len(loco),
                     "auc_min": float(min(d["auc_flipped"] for d in loco)),
                     "auc_max": float(max(d["auc_flipped"] for d in loco)),
                     "auc_mean": float(np.mean([d["auc_flipped"] for d in loco])),
                     "spearman_min": float(min(d["spearman"] for d in loco)),
                     "spearman_max": float(max(d["spearman"] for d in loco)),
                     "sign_stable": bool(all(d["spearman"] < 0 for d in loco))}

# ---- 5. soft-donor control ----------------------------------------------------------------
hasS = L.has_S.to_numpy(int)
res["no_sulphur_ligands"] = report(p_all[hasS == 0], ya[hasS == 0], logsf[hasS == 0],
                                   f"{int((hasS == 0).sum())} S-free ligands")
res["sulphur_ligands_only"] = report(p_all[hasS == 1], ya[hasS == 1], logsf[hasS == 1],
                                     f"{int((hasS == 1).sum())} S-containing ligands")
# and drop every S-containing ligand from the *extraction training* side too
hasS_e = np.array([int("S" in s or "s" in s) for s in E.smiles])
pS = fit_E(Xe[hasS_e == 0], ye[hasS_e == 0],
           chemotype_balanced_weights(E.chemotype.to_numpy()[hasS_e == 0])).predict_proba(Xa)[:, 1]
res["S_free_training_side"] = report(pS[hasS == 0], ya[hasS == 0], logsf[hasS == 0],
                                     "S-free on both sides")
res["n_S_in_extraction_train"] = int(hasS_e.sum())

# ---- 6. permutation null -------------------------------------------------------------------
rng = np.random.default_rng(4242)
perm_auc, perm_rho = [], []
for _ in range(300):
    yp = rng.permutation(ye)
    if len(set(yp)) < 2:
        continue
    pp = fit_E(Xe, yp, we).predict_proba(Xa)[:, 1]
    perm_auc.append(float(roc_auc_score(ya, 1 - pp)))
    perm_rho.append(float(spearmanr(pp, logsf).statistic))
perm_auc = np.array(perm_auc); perm_rho = np.array(perm_rho)
obs_auc = res["headline"]["auc_flipped"]; obs_rho = res["headline"]["spearman_pheavy_vs_logSF"]
res["permutation_null"] = {
    "auc_mean": float(perm_auc.mean()), "auc_sd": float(perm_auc.std()),
    "auc_q95": float(np.quantile(perm_auc, 0.95)),
    "auc_p_one_sided": float((perm_auc >= obs_auc).mean()),
    "rho_mean": float(perm_rho.mean()), "rho_sd": float(perm_rho.std()),
    "rho_q05": float(np.quantile(perm_rho, 0.05)),
    "rho_p_one_sided": float((perm_rho <= obs_rho).mean()),
    "draws": int(len(perm_auc)),
}

# ---- 7. well-replicated ligands only --------------------------------------------------------
for name, s in {"n_rows_ge_3": L.n_rows.to_numpy() >= 3,
                "abs_logSF_ge_1": np.abs(logsf) >= 1.0}.items():
    res[f"subset_{name}"] = report(p_all[s], ya[s], logsf[s], name)

# ---- 8. bootstrap CI on the Spearman (ligand resampling) --------------------------------------
bs = []
rng2 = np.random.default_rng(7)
for _ in range(5000):
    i = rng2.integers(0, len(ya), len(ya))
    if len(set(ya[i])) < 2:
        continue
    bs.append(float(spearmanr(p_all[i], logsf[i]).statistic))
res["headline"]["spearman_ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

(HERE / "amdeu_guards.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=1))
