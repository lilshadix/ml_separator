"""Deployable Gen15 predictor: a lanthanide separation curve from one SMILES, and from measurements.

Gen14 predicts the whole curve from two numbers and nothing but a structure.  Measured out of fold
for a ligand whose chemical family *and* every publication that studied it are absent from training
(design BP), that is worth an extractant-macro MAE of 0.50 log units against 0.59 for predicting no
separation at all -- the direction of selectivity is right 82 % of the time and essentially nothing
else about the curve is known.

Gen15 adds the mode that actually works.  A cell's pair residuals are additive in one per-cell metal
curve, so **one measured separation factor constrains all the others**, and the best linear use of it
is the BLUP with the corpus residual covariance.  On identical held-out pairs under BP:

    k = 0   0.439      structure only
    k = 1   0.231      one measured log SF   (+0.207, p < 1e-4, passes the pre-registered P1 rule)
    k = 2   0.223
    k = 3   0.205
    ceiling 0.182      the cell's own two coefficients, fitted to its complete data

Be honest about what carries that: a straight line drawn through the same measured pair with no
corpus at all scores 0.240, so of the 0.207 the measurement is worth, the model contributes about
0.008 (95 % CI [-0.011, +0.022], p = 0.35).  What the model adds that a line cannot is the *shape*
prior, the choice of which pair to measure next, and a calibrated interval on every prediction.

    # fit on the frozen corpus (writes gen15_curve/models/deploy_g15.joblib)
    .venv/Scripts/python.exe generations/gen15_curve/scripts/g15_predict.py fit

    # structure only
    .venv/Scripts/python.exe generations/gen15_curve/scripts/g15_predict.py predict \
        --smiles "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC"

    # structure plus one measured separation factor, log10 D(La) - log10 D(Lu)
    .venv/Scripts/python.exe generations/gen15_curve/scripts/g15_predict.py predict --smiles "..." \
        --measured "La/Lu=-1.31"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "generations" / "gen15_curve", ROOT / "generations" / "gen14_direction",
          ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen12_2_eu_pred"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen13sep.metals import ATOMIC_NUMBER, LANTHANIDES  # noqa: E402
from gen14 import dirbench as db  # noqa: E402
from gen15 import fewshot as FS  # noqa: E402

MODEL = ROOT / "generations" / "gen15_curve" / "models" / "deploy_g15.joblib"
N_LN = len(LANTHANIDES)


# --------------------------------------------------------------------------------------
def fit() -> None:
    """Fit the direction model, the corpus priors and the residual covariance on all 521 cells."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    bench = db.load()
    lean = bench.columns(LEAN_BLOCKS)
    topo = [c for c in lean if c.startswith("coord__dist__") or c.startswith("coord__arm__")]
    cols = np.array([lean.index(c) for c in topo], dtype=int)
    X = bench.matrix(LEAN_BLOCKS)[:, cols]
    amp = bench.coef[:, 0]
    rich = bench.frame.n_metals.to_numpy() >= db.MIN_METALS
    w = cell_weights(bench.groups[rich], bench.n_obs[rich])

    pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    pipe.fit(X[rich], (amp[rich] < 0).astype(int), logisticregression__sample_weight=w)

    magnitude = float(np.average(np.abs(amp[rich]), weights=w))
    curvature = float(np.average(bench.coef[:, 1], weights=cell_weights(bench.groups, bench.n_obs)))

    # residual curves of the deployed model over the whole corpus -> the correction's covariance.
    # In the honest evaluation this matrix is re-estimated leave-chemotype-out and publication-masked
    # for every held-out cell; here it is the deployment object, fitted on everything.
    p_all = pipe.predict_proba(X)[:, 1]
    a_all = np.where(p_all >= 0.5, -1.0, 1.0) * magnitude
    pred_curves = np.c_[a_all, np.full(len(a_all), curvature)] @ bench.basis
    resid = np.array([FS.centred_residual(bench.Y[i], pred_curves[i]) for i in range(len(bench.Y))])
    cov = FS.residual_covariance(resid)

    payload = {
        "pipeline": pipe, "columns": topo, "magnitude": magnitude, "curvature": curvature,
        "basis": bench.basis, "covariance": cov, "noise_var": FS.NOISE_VAR,
        "metals": list(LANTHANIDES),
        "n_train_cells": int(len(bench.Y)), "n_rich_cells": int(rich.sum()),
        "n_train_extractants": int(bench.frame.extractant.nunique()),
        "training_topology": np.unique(np.nan_to_num(X[rich], nan=-999.0), axis=0),
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, MODEL)
    print(f"fitted on {payload['n_train_cells']} cells "
          f"({payload['n_rich_cells']} well determined) / {payload['n_train_extractants']} extractants")
    print(f"magnitude {magnitude:.3f}  curvature {curvature:.3f}  "
          f"residual sd {np.sqrt(np.diag(cov)).mean():.3f}")
    print(f"-> {MODEL}")


# --------------------------------------------------------------------------------------
def _descriptors(smiles: list[str], columns: list[str]) -> np.ndarray:
    from gen122 import coordination
    rows = []
    for s in smiles:
        d = coordination.descriptors_for(s)
        d = {(k if k.startswith("coord__") else f"coord__{k}"): v for k, v in d.items()}
        rows.append({c: d.get(c, np.nan) for c in columns})
    return pd.DataFrame(rows, index=smiles).to_numpy(dtype=float)


def _parse_measured(items: list[str]) -> list[tuple[int, int, float]]:
    """``"La/Lu=-1.31"`` -> (index of La, index of Lu, -1.31), i.e. log D(La) - log D(Lu)."""
    out = []
    for it in items:
        pair, _, val = it.partition("=")
        a, _, b = pair.partition("/")
        a, b = a.strip(), b.strip()
        if a not in LANTHANIDES or b not in LANTHANIDES:
            raise SystemExit(f"unknown metal in {it!r}; use two of {list(LANTHANIDES)}")
        out.append((LANTHANIDES.index(a), LANTHANIDES.index(b), float(val)))
    return out


def predict_one(smiles: str, measured: list[tuple[int, int, float]] | None = None,
                payload: dict | None = None) -> dict:
    """Curve, every pairwise log SF with an interval, and the pair worth measuring next."""
    payload = payload or joblib.load(MODEL)
    measured = measured or []
    X = _descriptors([smiles], payload["columns"])
    p = float(payload["pipeline"].predict_proba(X)[0, 1])
    s = -1.0 if p >= 0.5 else 1.0
    prior = np.c_[[s * payload["magnitude"]], [payload["curvature"]]] @ payload["basis"]
    prior = prior[0]
    cov, nv = payload["covariance"], payload["noise_var"]
    curve = FS.blup(prior, cov, measured, noise_var=nv)
    post = FS.blup_posterior(cov, measured, noise_var=nv)

    seen = payload["training_topology"]
    novel = bool(not (np.abs(seen - np.nan_to_num(X[0], nan=-999.0)) < 1e-9).all(axis=1).any())

    pairs = []
    for i in range(N_LN):
        for j in range(i + 1, N_LN):
            sd = FS.pair_sd(post, i, j, noise_var=nv)
            v = float(curve[i] - curve[j])
            pairs.append({"A": LANTHANIDES[i], "B": LANTHANIDES[j],
                          "dZ": ATOMIC_NUMBER[LANTHANIDES[j]] - ATOMIC_NUMBER[LANTHANIDES[i]],
                          "log_SF": round(v, 4), "sd": round(sd, 4),
                          "lo90": round(v - 1.645 * sd, 4), "hi90": round(v + 1.645 * sd, 4),
                          "sign_confidence": round(float(abs(v) / sd), 3)})
    P = pd.DataFrame(pairs)

    remaining = [(i, j) for i in range(N_LN) for j in range(i + 1, N_LN)
                 if (i, j) not in {(a, b) for a, b, _ in measured}]
    nxt = FS.pick_support(remaining, 1, "dopt", post, noise_var=nv)
    return {
        "smiles": smiles, "p_heavy_selective": round(p, 4),
        "direction": "heavy-selective" if s < 0 else "light-selective",
        "direction_confidence": round(abs(2 * p - 1), 4),
        "topology_unseen_in_training": novel,
        "n_measurements_used": len(measured),
        "curve": {m: round(float(curve[k]), 4) for k, m in enumerate(LANTHANIDES)},
        "pairs": P,
        "next_measurement": (f"{LANTHANIDES[nxt[0][0]]}/{LANTHANIDES[nxt[0][1]]}" if nxt else None),
    }


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit")
    pp = sub.add_parser("predict")
    pp.add_argument("--smiles", required=True)
    pp.add_argument("--measured", nargs="*", default=[],
                    help='measured separation factors, e.g. "La/Lu=-1.31" (log D(La) - log D(Lu))')
    pp.add_argument("--output", default=None, help="write the full pair table to this CSV")
    pp.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "fit":
        fit()
        return 0

    r = predict_one(args.smiles, _parse_measured(args.measured))
    print(f"{r['smiles']}")
    print(f"  direction   {r['direction']}  (p_heavy {r['p_heavy_selective']}, "
          f"confidence {r['direction_confidence']})")
    print(f"  novel topology: {r['topology_unseen_in_training']}   "
          f"measurements used: {r['n_measurements_used']}")
    print(f"  measure next: {r['next_measurement']}")
    P = r["pairs"].reindex(r["pairs"].log_SF.abs().sort_values(ascending=False).index)
    print(f"\n  the {args.top} pairs it separates best (log SF, 90 % interval):")
    print(P.head(args.top).to_string(index=False))
    if args.output:
        r["pairs"].to_csv(args.output, index=False)
        print(f"\n  full 91-pair table -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
