"""Arms that put the external aqueous-logK ligands into the gen15 curve bench.

Every arm here is a direction call times the training fold's mean magnitude, with the curvature at
the training-fold mean -- i.e. gen14's deployed shape, with only the direction swapped.  That
isolates the question: is there direction information in the external corpus that the extraction
corpus does not already have?

``LOGK_DIR``   direction from a logistic fitted on the 273 logK ligands ALONE.  No extraction label
               of any kind enters, so it is external under every design including BP.
``LOGK_DIR_CB`` the same with class-balanced training weights (the logK side is 72 % heavy, so the
               threshold of the unweighted fit is a free parameter this removes).
``LOGK_NN``    the sign of the nearest logK ligand's slope by ECFP4 Tanimoto -- the cheapest
               possible use of the external data, and the one the earlier feature block used.
``G14_AUG_x``  gen14's own logistic with the external ligands appended to the training fold at
               relative weight x.  The practical question: does external data improve the deployed
               model?
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
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

from gen15.valuebench import Ctx                            # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS            # noqa: E402


@lru_cache(maxsize=1)
def _external() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    K = pd.read_parquet(HERE / "data" / "side_K_logk.parquet")
    topo = json.loads((HERE / "data" / "prep_meta.json").read_text())["topo39_columns"]
    return K[topo].to_numpy(float), K.y_heavy.to_numpy(int), K


def _pipe(C: float = 1.0):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))


@lru_cache(maxsize=4)
def _external_model(balanced: bool, C: float = 1.0):
    Xk, yk, _ = _external()
    w = (np.where(yk == 1, 1.0 / yk.mean(), 1.0 / (1 - yk.mean())) if balanced
         else np.ones(len(yk)))
    m = _pipe(C)
    m.fit(Xk, yk, logisticregression__sample_weight=w)
    return m


def _curve(a, b, n: int) -> np.ndarray:
    return np.c_[np.asarray(a, float), np.full(n, float(b))]


def _topo_index(ctx: Ctx) -> np.ndarray:
    return ctx.fs["TOPO39"]


# --------------------------------------------------------------------------------------
def always_heavy(ctx: Ctx) -> np.ndarray:
    """The constant direction rule (gen14's published yardstick, macro accuracy 0.559).

    This is the cheapest sensible alternative to *any* direction model, and the one an external
    model that mostly predicts "heavy" has to be scored against -- not against FLAT.
    """
    return _curve(np.full(len(ctx.test), -ctx.train_mean_magnitude()),
                  ctx.train_mean_curvature(), len(ctx.test))


def logk_dir(balanced: bool = False, C: float = 1.0):
    def f(ctx: Ctx) -> np.ndarray:
        X = ctx.X[:, _topo_index(ctx)][ctx.test]
        p = _external_model(balanced, C).predict_proba(X)[:, 1]
        sign = np.where(p >= 0.5, -1.0, 1.0)
        return _curve(sign * ctx.train_mean_magnitude(), ctx.train_mean_curvature(), len(ctx.test))
    return f


@lru_cache(maxsize=1)
def _nn_slope_by_extractant() -> dict[str, float]:
    """Slope of the ECFP4-nearest logK ligand, per extraction SMILES (Tanimoto, 2048 bits)."""
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    _, _, K = _external()
    kf, ks = [], []
    for s, sl in zip(K.smiles, K.slope):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            kf.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)); ks.append(float(sl))
    ks = np.array(ks)
    E = pd.read_parquet(HERE / "data" / "side_E_extraction.parquet")
    bench_smiles = set(E.smiles)
    frame = pd.read_parquet(ROOT / "gen13_separation" / "features" / "logk_prior.parquet")
    bench_smiles |= set(frame.index.astype(str))
    out = {}
    for s in sorted(bench_smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048), kf))
        out[s] = float(ks[int(sims.argmax())])
    return out


def logk_nn(ctx: Ctx) -> np.ndarray:
    tbl = _nn_slope_by_extractant()
    ext = ctx.bench.frame.extractant.to_numpy()[ctx.test]
    sl = np.array([tbl.get(e, np.nan) for e in ext], dtype=float)
    sign = np.where(np.isnan(sl), -1.0, np.where(sl < 0, -1.0, 1.0))
    return _curve(sign * ctx.train_mean_magnitude(), ctx.train_mean_curvature(), len(ctx.test))


def g14_augmented(lam: float = 1.0, C: float = 1.0):
    """gen14's logistic, training fold + external ligands at relative total weight ``lam``."""
    def f(ctx: Ctx) -> np.ndarray:
        idx = _topo_index(ctx)
        rtr = ctx.rich_train()
        Xtr = ctx.X[:, idx][rtr]
        w = ctx.rich_weights()
        y = (ctx.amp[rtr] < 0).astype(int)
        Xk, yk, _ = _external()
        wk = np.full(len(yk), lam * w.sum() / len(yk))
        m = _pipe(C)
        m.fit(np.vstack([Xtr, Xk]), np.concatenate([y, yk]),
              logisticregression__sample_weight=np.concatenate([w, wk]))
        p = m.predict_proba(ctx.X[:, idx][ctx.test])[:, 1]
        sign = np.where(p >= 0.5, -1.0, 1.0)
        return _curve(sign * ctx.train_mean_magnitude(), ctx.train_mean_curvature(), len(ctx.test))
    return f
