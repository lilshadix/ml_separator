"""Direction accuracy of every candidate sign rule, in gen14's own units.

The value bench scores a *curve*; gen14's headline direction number (macro accuracy 0.821 over
extractants, against 0.559 for always-heavy) is an upstream quantity.  Reporting it directly makes
the kernel direction arms comparable with the deployed classifier without going through the
pairwise metric, and it is cheap -- no pairs, no bootstrap, just the fold loop.

The unit is the extractant and the resampling block is the chemotype, exactly as gen13 stage 3.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "gen15_curve"))
sys.path.insert(0, str(HERE))

from gen15.valuebench import Ctx, MIN_METALS          # noqa: E402
from gen15 import valuebench as V                     # noqa: E402
from gen15.arms import _logistic_sign                 # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen13sep.splits import all_folds                 # noqa: E402
from gen14.dirbench import feature_sets               # noqa: E402
import kernels as KK                                   # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DESIGNS = ["B", "BR", "BQ", "A", "BP"]


def sign_rules() -> dict:
    r = {
        "G14_TOPO39_LOGIT": lambda c: _logistic_sign(c),
        "ALWAYS_HEAVY": lambda c: np.full(len(c.test), -1.0),
        "TRAIN_MAJORITY": _majority,
    }
    for lam in (0.1, 1.0, 10.0, 100.0):
        r[f"TAN_KLR_l{lam:g}"] = (lambda lam: lambda c: _klr(c, "TAN", lam))(lam)
        r[f"RBF_KLR_l{lam:g}"] = (lambda lam: lambda c: _klr(c, "RBF", lam))(lam)
        r[f"COMBO_KLR_l{lam:g}"] = (lambda lam: lambda c: _klr(c, "COMBO", lam))(lam)
    for C in (0.01, 0.1, 1.0):
        r[f"ECFPz_LOGIT_C{C:g}"] = (lambda C: lambda c: _ecfp(c, C, True))(C)
        r[f"ECFPraw_LOGIT_C{C:g}"] = (lambda C: lambda c: _ecfp(c, C, False))(C)
    for k in (1, 3, 5, 10):
        r[f"KNN_TAN_k{k}"] = (lambda k: lambda c: _knn(c, k))(k)
    return r


def _majority(c: Ctx) -> np.ndarray:
    tr, w = KK._rich(c)
    heavy = float(np.average((c.amp[tr] < 0).astype(float), weights=w))
    return np.full(len(c.test), -1.0 if heavy >= 0.5 else 1.0)


def _klr(c: Ctx, gram: str, lam: float) -> np.ndarray:
    tr, w = KK._rich(c)
    K = KK.GRAMS[gram](c)
    p = KK.wklr(K[np.ix_(tr, tr)], (c.amp[tr] < 0).astype(float), w, K[np.ix_(c.test, tr)], lam)
    return np.where(p >= 0.5, -1.0, 1.0)


def _ecfp(c: Ctx, C: float, scale: bool) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr, w = KK._rich(c)
    E = c.bench.frames["ECFP"].to_numpy(dtype=float)
    y = (c.amp[tr] < 0).astype(int)
    if len(set(y.tolist())) < 2:
        return np.full(len(c.test), -1.0 if y[0] == 1 else 1.0)
    steps = ([StandardScaler()] if scale else []) + [LogisticRegression(C=C, max_iter=5000)]
    m = make_pipeline(*steps)
    m.fit(E[tr], y, logisticregression__sample_weight=KK._norm_w(w))
    return np.where(m.predict_proba(E[c.test])[:, 1] >= 0.5, -1.0, 1.0)


def _knn(c: Ctx, k: int) -> np.ndarray:
    a, _, _ = KK._knn_predict(c, k, "TAN")
    return np.where(a < 0, -1.0, 1.0)


def main() -> None:
    bench = V.load()
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    rules = sign_rules()
    rows = []
    for d in DESIGNS:
        for f in all_folds(bench.frame, design=d):
            ctx = Ctx(bench=bench, design=d, seed=f.seed, fold=f.fold, train=f.train_index,
                      test=f.test_index,
                      w=cell_weights(bench.groups[f.train_index], bench.n_obs[f.train_index]),
                      model_seed=f.model_seed, fs=fs, X=X, rich=rich)
            keep = rich[ctx.test]
            if keep.sum() == 0:
                continue
            truth = np.sign(ctx.amp[ctx.test])
            ext = bench.frame.extractant.to_numpy()[ctx.test]
            chem = bench.frame.chemotype.to_numpy()[ctx.test]
            for name, fn in rules.items():
                s = np.asarray(fn(ctx), dtype=float)
                rows.append(pd.DataFrame({"design": d, "seed": f.seed, "rule": name,
                                          "extractant": ext[keep], "chemotype": chem[keep],
                                          "hit": (s[keep] == truth[keep]).astype(float)}))
        print(f"  {d} done", flush=True)
    t = pd.concat(rows, ignore_index=True)
    per_ext = t.groupby(["design", "rule", "seed", "extractant"])["hit"].mean().reset_index()
    macro = per_ext.groupby(["design", "rule", "seed"])["hit"].mean().groupby(
        level=[0, 1]).mean().unstack(0)
    macro = macro[[d for d in DESIGNS if d in macro.columns]]
    macro.to_csv(OUT / "direction_accuracy.csv")
    print("\n=== macro direction accuracy (extractant unit) ===")
    print(macro.round(4).sort_values("BP", ascending=False).to_string())


if __name__ == "__main__":
    main()
