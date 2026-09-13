"""Does the external score rank the 21 distinct extraction topology classes?

A model on the TOPO39 block cannot distinguish two ligands with the same 39 numbers, and the 82
extraction ligands occupy only 21 such rows (Kish effective n 3.5).  So the honest question for any
transfer is whether the external score orders those 21 classes by their mean extraction amplitude.
The permutation here shuffles CLASS labels, not ligands, so it prices the test at its real n.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
for p in (ROOT / "generations" / "gen15_curve", ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen12_2_eu_pred"):
    sys.path.insert(0, str(p))

from scipy.stats import spearmanr                           # noqa: E402
from sklearn.impute import SimpleImputer                    # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.pipeline import make_pipeline                  # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402
from gen122.coordination import build_table                 # noqa: E402
from gen13sep.models import chemotype_balanced_weights      # noqa: E402

TOPO = json.loads((HERE / "data" / "prep_meta.json").read_text())["topo39_columns"]
E = pd.read_parquet(HERE / "data" / "side_E_extraction.parquet")
K = pd.read_parquet(HERE / "data" / "side_K_logk.parquet")
A = pd.read_parquet(HERE / "data" / "side_A_amdeu.parquet")
A = A.join(build_table(A.canon.tolist())[TOPO].reindex(A.canon).reset_index(drop=True))


def pipe():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"))


def classes(df, cols=TOPO):
    return np.array([hash(tuple(np.round(r, 9))) for r in df[cols].to_numpy(float)])


def class_test(score: np.ndarray, value: np.ndarray, cls: np.ndarray, tag: str,
               reps: int = 50_000) -> dict:
    """Spearman between a per-class score and a per-class mean value, permuting classes."""
    d = pd.DataFrame({"c": cls, "s": score, "v": value})
    g = d.groupby("c").agg(s=("s", "mean"), v=("v", "mean"), n=("v", "size"))
    if g.s.nunique() < 3:
        return {"tag": tag, "n_classes": int(len(g)), "note": "score constant across classes"}
    rho = float(spearmanr(g.s, g.v).statistic)
    rng = np.random.default_rng(3)
    null = np.array([spearmanr(g.s, rng.permutation(g.v.to_numpy())).statistic for _ in range(reps)])
    return {"tag": tag, "n_classes": int(len(g)), "n_ligands": int(g.n.sum()),
            "kish_effective_n": float(g.n.sum() ** 2 / (g.n ** 2).sum()),
            "spearman_class": rho,
            "p_two_sided_class_permutation": float(2 * min((null <= rho).mean(), (null >= rho).mean())),
            "null_sd": float(null.std()),
            "spearman_ligand_level": float(spearmanr(score, value).statistic)}


res: dict = {}
Xe = E[TOPO].to_numpy(float); ye = E.y_heavy.to_numpy(int); amp = E.amp.to_numpy(float)
we = chemotype_balanced_weights(E.chemotype.to_numpy())
ce = classes(E)

# --- external logK score, evaluated on the extraction classes ----------------------------
m_k = pipe().fit(K[TOPO].to_numpy(float), K.y_heavy.to_numpy(int))
res["logk_score_on_extraction_classes"] = class_test(
    -m_k.predict_proba(Xe)[:, 1], amp, ce, "logK-trained P(heavy) (negated) vs extraction amp")

# --- reference: the WITHIN-corpus model, chemotype-CV, on the same classes ----------------
chem = E.chemotype.to_numpy()
uniq = np.array(sorted(set(chem)))
assign = {g: i % 5 for i, g in enumerate(np.random.default_rng(0).permutation(uniq))}
blk = np.array([assign[g] for g in chem])
oof = np.zeros(len(ye))
for k in range(5):
    tr, te = blk != k, blk == k
    if len(set(ye[tr])) < 2 or te.sum() == 0:
        oof[te] = ye[tr].mean(); continue
    oof[te] = pipe().fit(Xe[tr], ye[tr],
                         logisticregression__sample_weight=chemotype_balanced_weights(chem[tr])
                         ).predict_proba(Xe[te])[:, 1]
res["within_corpus_cv_on_extraction_classes"] = class_test(
    -oof, amp, ce, "gen14-style chemotype-CV P(heavy) (negated) vs extraction amp")

# --- Am/Eu: the extraction-trained score on the Am/Eu classes -----------------------------
m_e = pipe().fit(Xe, ye, logisticregression__sample_weight=we)
Xa = A[TOPO].to_numpy(float)
res["extraction_score_on_amdeu_classes"] = class_test(
    m_e.predict_proba(Xa)[:, 1], A.logSF.to_numpy(float), classes(A),
    "extraction-trained P(heavy) vs log10 Am/Eu SF (negative rho = the flip)")

# --- Am/Eu: the Am/Eu-trained score on the extraction classes -----------------------------
m_a = pipe().fit(Xa, A.y_am.to_numpy(int))
res["amdeu_score_on_extraction_classes"] = class_test(
    m_a.predict_proba(Xe)[:, 1], amp, ce,
    "AmEu-trained P(Am) vs extraction amp (positive rho = the same flip)")

# --- logK: the extraction-trained score on the logK classes -------------------------------
Xk = K[TOPO].to_numpy(float)
res["extraction_score_on_logk_classes"] = class_test(
    -m_e.predict_proba(Xk)[:, 1], K.slope.to_numpy(float), classes(K),
    "extraction-trained P(heavy) (negated) vs logK slope")

(HERE / "class_level.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res, indent=1))
