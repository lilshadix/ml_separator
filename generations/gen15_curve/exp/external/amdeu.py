"""SECONDARY, clearly-labelled result: the An/Ln (soft-donor) axis, not the lanthanide axis.

Source: CainLiTian/Am_Eu_ligand_design ``data/literature_data.xlsx`` (fetched 2026-09-09,
116 962 bytes, sha256 recorded below).  301 rows of D_Am / D_Eu / SF with conditions.

Am/Eu selectivity is a *different* axis from heavy-vs-light lanthanide selectivity: it is driven by
soft-donor covalency (S, N) rather than by ionic radius.  So this cannot validate the lanthanide
slope.  What it can ask is whether the SAME 39 donor-topology columns that call the lanthanide
direction also carry the Am/Eu direction, i.e. whether one representation serves both axes.
"""
from __future__ import annotations

import hashlib
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
from gen14.dirbench import Blocked                          # noqa: E402

XLSX = HERE / "data" / "am_eu_literature_data.xlsx"
TOPO = json.loads((HERE / "data" / "prep_meta.json").read_text())["topo39_columns"]
res: dict = {"source_sha256": hashlib.sha256(XLSX.read_bytes()).hexdigest(),
             "source_bytes": XLSX.stat().st_size}


def pipe(C: float = 1.0):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))


def basic(y, p) -> dict:
    yh = (np.asarray(p) >= 0.5).astype(int); y = np.asarray(y)
    per = {int(c): float((yh[y == c] == c).mean()) for c in np.unique(y)}
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return {"n": int(len(y)), "accuracy": float((yh == y).mean()),
            "balanced_accuracy": float(np.mean(list(per.values()))), "auc": auc,
            "majority_rule_accuracy": float(max(np.mean(y == 0), np.mean(y == 1))),
            "base_rate_class1": float(y.mean()), "per_class_recall": per,
            "pred_frac_class1": float(yh.mean())}


# ---- load and collapse to one row per ligand ------------------------------------------
raw = pd.read_excel(XLSX, sheet_name="Sheet1")
res["raw_rows"] = int(len(raw))
res["type_counts"] = {str(k): int(v) for k, v in raw["type"].value_counts().items()}

from rdkit import Chem, DataStructs, RDLogger                # noqa: E402
from rdkit.Chem import AllChem                               # noqa: E402
RDLogger.DisableLog("rdApp.*")

raw = raw[raw["type"].astype(str).str.strip().str.lower() == "am/eu"].copy()
raw["SF"] = pd.to_numeric(raw["SF"], errors="coerce")
raw = raw[np.isfinite(raw["SF"]) & (raw["SF"] > 0)]
raw["canon"] = [Chem.MolToSmiles(m) if (m := Chem.MolFromSmiles(str(s))) else None
                for s in raw["SMILES"]]
raw = raw[raw.canon.notna()]
raw["logSF"] = np.log10(raw["SF"])
res["usable_rows"] = int(len(raw))

L = raw.groupby("canon").agg(logSF=("logSF", "median"), n_rows=("logSF", "size"),
                             logSF_sd=("logSF", "std")).reset_index()
L["y_am"] = (L.logSF > 0).astype(int)          # 1 = Am-selective (SF > 1)
res["n_ligands"] = int(len(L))
res["base_rate_Am_selective"] = float(L.y_am.mean())
res["constant_rule_accuracy"] = float(max(L.y_am.mean(), 1 - L.y_am.mean()))
res["logSF_quantiles"] = {q: float(L.logSF.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
res["within_ligand_logSF_sd_median"] = float(L.logSF_sd.median(skipna=True))

# ---- the shared 39 topology columns ----------------------------------------------------
T = build_table(L.canon.tolist())[TOPO]
L = L.join(T.reindex(L.canon).reset_index(drop=True))
Xa = L[TOPO].to_numpy(float)
ya = L.y_am.to_numpy(int)
res["distinct_topology_rows"] = int(len(np.unique(Xa, axis=0)))

E = pd.read_parquet(HERE / "data" / "side_E_extraction.parquet")
Xe = E[TOPO].to_numpy(float); ye = E.y_heavy.to_numpy(int)
we = chemotype_balanced_weights(E.chemotype.to_numpy())

# ---- chemical-space overlap with the extraction set ------------------------------------
def fp(s):
    m = Chem.MolFromSmiles(s)
    return None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)

fe = [f for f in (fp(s) for s in E.smiles) if f is not None]
nn = []
for s in L.canon:
    f = fp(s)
    nn.append(np.nan if f is None else float(max(DataStructs.BulkTanimotoSimilarity(f, fe))))
L["nn_tanimoto"] = nn
res["nn_tanimoto_quantiles"] = {q: float(np.nanquantile(nn, q)) for q in (0.05, 0.5, 0.95, 1.0)}
res["exact_shared_with_extraction"] = int(len(set(L.canon) & set(E.smiles)))

# ======================================================================================
# Test 1: gen14's direction model (trained on heavy-vs-light extraction) -> Am/Eu direction
# ======================================================================================
m_e = pipe().fit(Xe, ye, logisticregression__sample_weight=we)
p_a = m_e.predict_proba(Xa)[:, 1]                       # P(heavy-selective in extraction)
res["E_to_AmEu"] = {"as_is": basic(ya, p_a),
                    "spearman_pheavy_vs_logSF": float(spearmanr(p_a, L.logSF).statistic),
                    "flipped_orientation": basic(ya, 1 - p_a)}
s = np.abs(L.logSF.to_numpy()) >= 0.5                   # SF beyond 3x, either way
res["E_to_AmEu"]["strong_SF_only"] = basic(ya[s], p_a[s])

# ======================================================================================
# Test 2: Am/Eu direction model -> extraction heavy-vs-light direction
# ======================================================================================
m_a = pipe().fit(Xa, ya, logisticregression__sample_weight=np.ones(len(ya)))
p_e = m_a.predict_proba(Xe)[:, 1]                       # P(Am-selective) reused as P(heavy)
unit = pd.DataFrame({"chemotype": E.chemotype, "hit": ((p_e >= 0.5).astype(int) == ye).astype(float)})
boot = Blocked(unit.chemotype)
macro, draws = boot.draws(unit)
ah = unit.assign(hit=(np.ones(len(ye)) == ye).astype(float))
gain, gd = boot.draws(unit.assign(hit=unit.hit.to_numpy() - ah.hit.to_numpy()))
res["AmEu_to_E"] = {
    "pooled": basic(ye, p_e), "macro_accuracy": macro,
    "always_heavy_macro": float(boot.draws(ah)[0]),
    "gain_vs_always_heavy": gain,
    "gain_ci": [float(np.quantile(gd, 0.025)), float(np.quantile(gd, 0.975))],
    "gain_p_two_sided": float(2 * min((gd <= 0).mean(), (gd >= 0).mean())),
    "spearman_pAm_vs_amp": float(spearmanr(p_e, E.amp).statistic),
}

# ======================================================================================
# Reference: is the Am/Eu direction learnable from these 39 columns AT ALL?
# 5-fold CV inside the Am/Eu set -- if this is at chance the transfer test is uninformative.
# ======================================================================================
kf = np.array([i % 5 for i in np.random.default_rng(0).permutation(len(ya))])
oof = np.zeros(len(ya))
for k in range(5):
    tr, te = kf != k, kf == k
    if len(set(ya[tr])) < 2:
        oof[te] = ya[tr].mean(); continue
    oof[te] = pipe().fit(Xa[tr], ya[tr]).predict_proba(Xa[te])[:, 1]
res["reference_within_AmEu_cv"] = basic(ya, oof)
res["reference_within_AmEu_cv"]["spearman_oof_vs_logSF"] = float(spearmanr(oof, L.logSF).statistic)

# a soft-donor sanity control: do S/N-donor ligands dominate the Am-selective class?
L["has_S"] = [int("S" in Chem.MolToSmiles(Chem.MolFromSmiles(s))) for s in L.canon]
res["soft_donor_sanity"] = {
    "frac_ligands_with_S": float(L.has_S.mean()),
    "Am_selective_rate_with_S": float(L[L.has_S == 1].y_am.mean()),
    "Am_selective_rate_without_S": float(L[L.has_S == 0].y_am.mean()),
    "median_logSF_with_S": float(L[L.has_S == 1].logSF.median()),
    "median_logSF_without_S": float(L[L.has_S == 0].logSF.median()),
}

L.drop(columns=TOPO).to_parquet(HERE / "data" / "side_A_amdeu.parquet", index=False)
(HERE / "amdeu.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=1))
