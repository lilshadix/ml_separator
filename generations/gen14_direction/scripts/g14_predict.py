"""Deployable Gen14 predictor: two numbers per ligand, from one SMILES string.

Gen14 predicts a lanthanide separation curve from a *direction* and a *magnitude*:

    log D(Ln) - mean_Ln log D  =  a * radius(Ln) + b * radius(Ln)^2,     a = s * m

where ``s`` is the direction of selectivity (-1 heavy-selective, +1 light-selective) called by an
L2 logistic regression on the 39 donor-topology columns of the frozen gen12.2 coordination block,
``m`` is the training corpus' mean |amplitude| and ``b`` its mean curvature.  Nothing else is
fitted, and nothing but the SMILES string is needed: the descriptors are bond counts on the 2D
molecular graph, so there is no conformer, no DFT and no measurement in the path.

    # fit on all 289 well-determined cells (writes gen14_direction/models/deploy.joblib)
    .venv/Scripts/python.exe generations/gen14_direction/scripts/g14_predict.py fit

    # predict for one ligand, or for a CSV with a `smiles` column
    .venv/Scripts/python.exe generations/gen14_direction/scripts/g14_predict.py predict --smiles "CCN(CC)C(=O)COCC(=O)N(CC)CC"
    .venv/Scripts/python.exe generations/gen14_direction/scripts/g14_predict.py predict --input new.csv --output pred.csv

What it is worth, out of fold, for a ligand whose whole chemical family *and* every publication
that studied it are absent from training (design BP): the direction is right for 82 % of
extractants, and the resulting curve carries a macro MAE of 0.50 log units over all metal pairs
(0.19 for neighbours), against 0.62 for the corpus mean curve and 0.55 for gen13's 209-column
regression.  Accuracy of the direction call depends strongly on how selective the ligand actually
is -- 93 % where the whole-series contrast exceeds 1.6 log units, 59 % where it is under 0.16,
which is at the measurement noise floor.  The report is `gen14_direction/GEN14_REPORT.md`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "generations" / "gen14_direction"))
sys.path.insert(0, str(ROOT / "generations" / "gen13_separation"))
sys.path.insert(0, str(ROOT / "generations" / "gen12_2_eu_pred"))

from gen14 import dirbench as db  # noqa: E402
from gen14 import models as M  # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen13sep.metals import ATOMIC_NUMBER, LANTHANIDES  # noqa: E402

MODEL = ROOT / "generations" / "gen14_direction" / "models" / "deploy.joblib"


def fit() -> None:
    bench = db.load()
    lean = bench.columns(LEAN_BLOCKS)
    topo = [c for c in lean if c.startswith("coord__dist__") or c.startswith("coord__arm__")]
    cols = np.array([lean.index(c) for c in topo], dtype=int)
    X = bench.matrix(LEAN_BLOCKS)[:, cols]
    amp = bench.coef[:, 0]
    rich = bench.frame.n_metals.to_numpy() >= db.MIN_METALS
    w = cell_weights(bench.groups[rich], bench.n_obs[rich])

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    pipe.fit(X[rich], (amp[rich] < 0).astype(int), logisticregression__sample_weight=w)

    payload = {
        "pipeline": pipe,
        "columns": topo,
        "magnitude": float(np.average(np.abs(amp[rich]), weights=w)),
        "curvature": float(np.average(bench.coef[:, 1],
                                      weights=cell_weights(bench.groups, bench.n_obs))),
        "basis": bench.basis,
        "n_train_cells": int(rich.sum()),
        "n_train_extractants": int(bench.frame.extractant[rich].nunique()),
        "training_topology": np.unique(np.nan_to_num(X[rich], nan=-999.0), axis=0),
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL)
    print(f"fitted on {payload['n_train_cells']} cells / {payload['n_train_extractants']} "
          f"extractants; magnitude {payload['magnitude']:.3f}, curvature {payload['curvature']:.3f}")
    print(f"-> {MODEL}")


def _descriptors(smiles: list[str], columns: list[str]) -> pd.DataFrame:
    from gen122 import coordination
    rows = []
    for s in smiles:
        d = coordination.descriptors_for(s)
        d = {(k if k.startswith("coord__") else f"coord__{k}"): v for k, v in d.items()}
        rows.append({c: d.get(c, np.nan) for c in columns})
    return pd.DataFrame(rows, index=smiles)


def predict(smiles: list[str]) -> pd.DataFrame:
    payload = joblib.load(MODEL)
    X = _descriptors(smiles, payload["columns"]).to_numpy(dtype=float)
    p = payload["pipeline"].predict_proba(X)[:, 1]
    s = np.where(p >= 0.5, -1.0, 1.0)
    a = s * payload["magnitude"]
    curve = np.c_[a, np.full(len(a), payload["curvature"])] @ payload["basis"]
    seen = payload["training_topology"]
    novel = [bool(not (np.abs(seen - np.nan_to_num(row, nan=-999.0)) < 1e-9).all(axis=1).any())
             for row in X]
    out = pd.DataFrame({
        "smiles": smiles,
        "p_heavy_selective": p.round(4),
        "direction": np.where(s < 0, "heavy-selective", "light-selective"),
        "confidence": np.abs(2 * p - 1).round(4),
        "amplitude": a.round(4),
        "topology_unseen_in_training": novel,
    })
    for j, m in enumerate(LANTHANIDES):
        out[f"curve__{m}"] = curve[:, j].round(4)
    z = {m: ATOMIC_NUMBER[m] for m in LANTHANIDES}
    widest = max(((i, j) for i in range(14) for j in range(14) if i < j),
                 key=lambda t: z[LANTHANIDES[t[1]]] - z[LANTHANIDES[t[0]]])
    out["logSF_La_Lu"] = (curve[:, widest[0]] - curve[:, widest[1]]).round(4)
    out["suggested_first_measurement"] = f"{LANTHANIDES[widest[0]]}/{LANTHANIDES[widest[1]]}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("fit")
    pp = sub.add_parser("predict")
    pp.add_argument("--smiles", nargs="*", default=None)
    pp.add_argument("--input", type=Path, default=None)
    pp.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.command == "fit":
        fit()
        return 0
    smiles = list(args.smiles or [])
    if args.input is not None:
        smiles += pd.read_csv(args.input)["smiles"].astype(str).tolist()
    if not smiles:
        ap.error("give --smiles or --input")
    out = predict(smiles)
    if args.output is not None:
        out.to_csv(args.output, index=False)
        print(f"-> {args.output}")
    cols = ["smiles", "p_heavy_selective", "direction", "confidence", "logSF_La_Lu",
            "topology_unseen_in_training"]
    print(out[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
