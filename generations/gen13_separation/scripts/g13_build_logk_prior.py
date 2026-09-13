"""Build the LOGK transfer block: aqueous lanthanide-series stability-constant shape as a prior.

Source: SmartChemDesign/BidentateGNN (MIT) ``Data/logK/<El>.sdf`` — literature log K values
for La–Lu complexes with tags ``logK_<M>_z=3.0_T=<T>_I=<I>``.  For every (ligand, T, I) series
with >= 4 lanthanides we fit ``logK = level + slope * z(r) + curv * (z(r)^2 - mean)`` on the
standardised Shannon radius (the same physics basis Gen13 uses for extraction curves).  A tree
model from ECFP + RDKit scalars to (slope, curv) is fitted on those external ligands only, and
its prediction for each bundle extractant — plus the nearest external ligand's slope/curv and
Tanimoto — becomes the ``logk__*`` block.  Nothing here reads the bundle's log D.

    .venv/Scripts/python.exe generations/gen13_separation/scripts/g13_build_logk_prior.py --sdf-dir <dir with La.sdf ...>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen13sep import paths  # noqa: E402
from gen13sep.metals import LANTHANIDES, SHANNON_RADIUS_CN8  # noqa: E402

OUT_DIR = paths.GEN13_ROOT / "features"


def parse_sdfs(sdf_dir: Path) -> pd.DataFrame:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for fn in sorted(os.listdir(sdf_dir)):
        if not fn.endswith(".sdf"):
            continue
        for mol in Chem.SDMolSupplier(str(sdf_dir / fn), removeHs=False):
            if mol is None:
                continue
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            for prop in mol.GetPropNames():
                m = re.match(r"logK_(\w+)_z=([\d.]+)_T=([\d.]+)_I=([\d.]+)", prop)
                if m:
                    try:
                        v = float(mol.GetProp(prop))
                    except ValueError:
                        continue
                    rows.append(dict(smiles=smi, metal=m.group(1), z=float(m.group(2)), T=float(m.group(3)),
                                     I=float(m.group(4)), logK=v))
    return pd.DataFrame(rows)


def series_table(df: pd.DataFrame, min_ln: int = 4) -> pd.DataFrame:
    r = np.array([SHANNON_RADIUS_CN8[m] for m in LANTHANIDES]); rz = (r - r.mean()) / r.std()
    rz_map = dict(zip(LANTHANIDES, rz)); c2 = float((rz ** 2).mean())
    ln = df[df.metal.isin(LANTHANIDES) & (df.z == 3.0)]
    out = []
    for (smi, T, I), sub in ln.groupby(["smiles", "T", "I"]):
        s = sub.groupby("metal").logK.mean()
        if len(s) < min_ln:
            continue
        x = np.array([rz_map[m] for m in s.index]); y = s.to_numpy(float)
        X = np.c_[np.ones_like(x), x, x ** 2 - c2]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        out.append(dict(smiles=smi, T=T, I=I, n=len(s), slope=beta[1], curv=beta[2],
                        r2=1 - resid.var() / y.var() if y.var() > 0 else np.nan))
    ser = pd.DataFrame(out)
    # one row per ligand: prefer T=25, I=0.1; else the series with most lanthanides; average duplicates
    ser["pref"] = ((ser["T"] == 25) & (ser["I"] == 0.1)).astype(int) * 100 + ser["n"]
    best = ser.sort_values("pref", ascending=False).groupby("smiles").head(1)
    return best.drop(columns="pref").reset_index(drop=True)


def featurise(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    feats = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            feats.append(np.full(2048 + 6, np.nan)); continue
        fp = np.zeros(2048)
        bv = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        for i in bv.GetOnBits():
            fp[i] = 1.0
        desc = [Descriptors.MolWt(m), Descriptors.TPSA(m), Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
                Descriptors.NumRotatableBonds(m), Descriptors.MolLogP(m)]
        feats.append(np.concatenate([fp, desc]))
    return np.vstack(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf-dir", required=True)
    ap.add_argument("--trees", type=int, default=500)
    args = ap.parse_args()
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import KFold

    df = parse_sdfs(Path(args.sdf_dir))
    ser = series_table(df)
    OUT_DIR.mkdir(exist_ok=True)
    ser.to_parquet(OUT_DIR / "logk_external_series.parquet", index=False)
    X = featurise(ser["smiles"].tolist()); Y = ser[["slope", "curv"]].to_numpy(float)
    ok = np.isfinite(X).all(1)
    X, Y, ser = X[ok], Y[ok], ser[ok].reset_index(drop=True)
    # external cross-validated skill of the transfer model (5-fold over ligands)
    cv_pred = np.zeros_like(Y)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = ExtraTreesRegressor(n_estimators=args.trees, max_features=0.5, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X[tr], Y[tr])
        cv_pred[te] = m.predict(X[te])
    from scipy.stats import spearmanr
    cv = {"n_ligands": int(len(ser)), "slope_spearman_cv": float(spearmanr(Y[:, 0], cv_pred[:, 0]).statistic),
          "curv_spearman_cv": float(spearmanr(Y[:, 1], cv_pred[:, 1]).statistic),
          "slope_mae_cv": float(np.abs(Y[:, 0] - cv_pred[:, 0]).mean()), "slope_sd": float(Y[:, 0].std())}
    model = ExtraTreesRegressor(n_estimators=args.trees, max_features=0.5, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(X, Y)

    bundle = pd.read_parquet(paths.BUNDLE_PARQUET, columns=["canonical_smiles"])
    ours = sorted(bundle["canonical_smiles"].unique())
    Xo = featurise(ours)
    pred = model.predict(np.where(np.isfinite(Xo), Xo, 0.0))
    ext_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in ser["smiles"]]
    rows = []
    for i, s in enumerate(ours):
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append({"extractant": s}); continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), ext_fps))
        j = int(sims.argmax())
        rows.append({"extractant": s, "logk__pred_slope": float(pred[i, 0]), "logk__pred_curv": float(pred[i, 1]),
                     "logk__nn_slope": float(ser["slope"].iat[j]), "logk__nn_curv": float(ser["curv"].iat[j]),
                     "logk__nn_tanimoto": float(sims[j]), "logk__nn_r2": float(ser["r2"].iat[j])})
    block = pd.DataFrame(rows).set_index("extractant")
    block.to_parquet(OUT_DIR / "logk_prior.parquet")
    with open(OUT_DIR / "logk_prior.json", "w", encoding="utf-8") as fh:
        json.dump({"source": "SmartChemDesign/BidentateGNN Data/logK/*.sdf (MIT)", "records": int(len(df)),
                   "ligands_with_series": int(len(ser)), "transfer_cv": cv,
                   "columns": [c for c in block.columns], "n_extractants": int(len(block))}, fh, indent=2)
    print(json.dumps(cv, indent=1))
    print(block.describe().round(3).to_string())


if __name__ == "__main__":
    main()
