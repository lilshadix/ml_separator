"""How many independent units does the 39-column topology block actually resolve, and does the
association survive at that unit?

The 82 extraction ligands occupy only 21 DISTINCT rows of the TOPO39 block.  Everything a model on
these columns can say is therefore one number per distinct row, and a transfer test scored over 82
ligands, or over 273, is scored over far fewer effective units than it looks.  This script:

* counts the distinct rows on each side and their occupancies;
* for every row present on BOTH sides, compares the mean extraction amplitude with the mean
  external quantity (logK slope, or log Am/Eu SF), which is the association at the honest unit;
* prices it with a permutation test that permutes the class labels, i.e. the ~14 units, not the
  hundreds of ligands.
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
from gen122.coordination import build_table                 # noqa: E402

TOPO = json.loads((HERE / "data" / "prep_meta.json").read_text())["topo39_columns"]
E = pd.read_parquet(HERE / "data" / "side_E_extraction.parquet")
K = pd.read_parquet(HERE / "data" / "side_K_logk.parquet")
A = pd.read_parquet(HERE / "data" / "side_A_amdeu.parquet")
A = A.join(build_table(A.canon.tolist())[TOPO].reindex(A.canon).reset_index(drop=True))

res: dict = {}


def key_rows(df: pd.DataFrame) -> np.ndarray:
    """A hashable key per distinct TOPO39 row."""
    return np.array([hash(tuple(np.round(r, 9))) for r in df[TOPO].to_numpy(float)])


def side_summary(df: pd.DataFrame, name: str) -> dict:
    k = key_rows(df)
    _, cnt = np.unique(k, return_counts=True)
    return {"n_ligands": int(len(df)), "n_distinct_topology_rows": int(len(cnt)),
            "largest_class": int(cnt.max()),
            "occupancy_top5": sorted(map(int, cnt))[::-1][:5],
            "effective_n_kish": float(cnt.sum() ** 2 / (cnt ** 2).sum())}


res["sides"] = {"extraction_82": side_summary(E, "E"), "logk_273": side_summary(K, "K"),
                "amdeu_85": side_summary(A, "A")}

E["_k"] = key_rows(E); K["_k"] = key_rows(K); A["_k"] = key_rows(A)


def paired_units(other: pd.DataFrame, col: str, label: str) -> dict:
    e = E.groupby("_k").agg(amp=("amp", "mean"), n_e=("amp", "size"))
    o = other.groupby("_k").agg(v=(col, "mean"), n_o=(col, "size"))
    j = e.join(o, how="inner").dropna()
    if len(j) < 4:
        return {"label": label, "n_shared_units": int(len(j)), "note": "too few shared units"}
    rho = float(spearmanr(j.amp, j.v).statistic)
    r = float(np.corrcoef(j.amp, j.v)[0, 1])
    # sign agreement at the unit level, and its exact binomial-style permutation p
    agree = float(np.mean(np.sign(j.amp) == np.sign(j.v)))
    rng = np.random.default_rng(11)
    null_rho = np.array([spearmanr(j.amp, rng.permutation(j.v.to_numpy())).statistic
                         for _ in range(20000)])
    p = float(2 * min((null_rho <= rho).mean(), (null_rho >= rho).mean()))
    # weighted by how many ligands each unit holds on the extraction side
    wrho = float(spearmanr(np.repeat(j.amp, j.n_e), np.repeat(j.v, j.n_e)).statistic)
    return {"label": label, "n_shared_units": int(len(j)),
            "ligands_covered_extraction": int(j.n_e.sum()),
            "ligands_covered_other": int(j.n_o.sum()),
            "spearman_unit": rho, "pearson_unit": r,
            "spearman_p_permutation": p,
            "spearman_weighted_by_extraction_count": wrho,
            "sign_agreement": agree,
            "units": j.reset_index(drop=True).round(4).to_dict("records")}


res["extraction_vs_logk_slope"] = paired_units(K, "slope", "extraction amp vs aqueous logK slope")
res["extraction_vs_amdeu_logSF"] = paired_units(A, "logSF", "extraction amp vs log10 Am/Eu SF")

# the same association at the LIGAND level for the exactly-shared structures only
sh_k = sorted(set(E.smiles) & set(K.smiles))
sh_a = sorted(set(E.smiles) & set(A.canon))
res["exact_shared_ligands"] = {
    "logk": {"n": len(sh_k),
             "pairs": [{"smiles": s, "amp": float(E[E.smiles == s].amp.iat[0]),
                        "slope": float(K[K.smiles == s].slope.iat[0])} for s in sh_k]},
    "amdeu": {"n": len(sh_a),
              "pairs": [{"smiles": s, "amp": float(E[E.smiles == s].amp.iat[0]),
                         "logSF": float(A[A.canon == s].logSF.iat[0])} for s in sh_a]},
}
if len(sh_a) >= 4:
    aa = np.array([E[E.smiles == s].amp.iat[0] for s in sh_a])
    ss = np.array([A[A.canon == s].logSF.iat[0] for s in sh_a])
    res["exact_shared_ligands"]["amdeu"]["spearman"] = float(spearmanr(aa, ss).statistic)
if len(sh_k) >= 4:
    aa = np.array([E[E.smiles == s].amp.iat[0] for s in sh_k])
    ss = np.array([K[K.smiles == s].slope.iat[0] for s in sh_k])
    res["exact_shared_ligands"]["logk"]["spearman"] = float(spearmanr(aa, ss).statistic)

(HERE / "unit_level.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
printable = {k: (v if k != "sides" else v) for k, v in res.items()}
for k in ("extraction_vs_logk_slope", "extraction_vs_amdeu_logSF"):
    printable[k] = {kk: vv for kk, vv in res[k].items() if kk != "units"}
print(json.dumps(printable, indent=1))
