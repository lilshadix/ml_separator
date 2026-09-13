"""Re-implementation of the original estimator so that attacks change one thing at a time."""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("D:/ml_separator_gh/gen13_separation")
OUT = ROOT / "analysis" / "stage3" / "verify_direction"
sys.path.insert(0, str(ROOT))
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from gen13sep.splits import all_folds
from gen13sep.amplitude_bench import cell_weights

P = pickle.load(open(OUT / "bench_cache.pkl", "rb"))
FRAME, COEF, NOBS, FRAMES, GROUPS, Y14, BASIS = (
    P["frame"], P["coef"], P["n_obs"], P["frames"], P["groups"], P["Y"], P["basis"])
ALLX = pd.concat([FRAMES[b] for b in ("COND","MASSACT","PHYSCHEM","DONORS","ECFP","LIG2D","COORD")], axis=1)
LEAN_COLS = [c for b in ("COND","MASSACT","PHYSCHEM","DONORS","COORD") for c in FRAMES[b].columns]
TOPO_COLS = [c for c in FRAMES["COORD"].columns if c.startswith("coord__dist__") or c.startswith("coord__arm__")]


def et(seed):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         ExtraTreesClassifier(n_estimators=400, max_features=0.5,
                                              min_samples_leaf=2, random_state=seed, n_jobs=4))

def lg(seed):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def run(cols, y, keep, *, design="BP", est="et", frame=None, seeds=None, min_train=40):
    """`keep` is a boolean mask over the 521 cells selecting the scored subset."""
    fr = FRAME if frame is None else frame
    X = ALLX.loc[:, cols].to_numpy(float)
    folds = all_folds(fr, design=design) if seeds is None else all_folds(fr, design=design, seeds=seeds)
    maker = {"et": et, "lg": lg}[est]
    rows = []
    for f in folds:
        tr = f.train_index[keep[f.train_index]]
        te = f.test_index[keep[f.test_index]]
        if len(tr) < min_train or len(te) < 1 or len(set(y[tr])) < 2:
            continue
        w = cell_weights(GROUPS[tr], NOBS[tr])
        m = maker(f.model_seed)
        step = list(m.named_steps)[-1]
        m.fit(X[tr], y[tr], **{f"{step}__sample_weight": w})
        p = m.predict_proba(X[te])[:, 1]
        for j, ci in enumerate(te):
            rows.append({"split_seed": f.seed, "fold": f.fold, "cell_id": fr["cell_id"].iat[ci],
                         "extractant": fr["extractant"].iat[ci], "chemotype": fr["chemotype"].iat[ci],
                         "y": int(y[ci]), "p": float(p[j])})
    return pd.DataFrame(rows)


def units(tab):
    """One row per extractant: mean-over-seeds of the extractant's per-seed cell accuracy."""
    t = tab.assign(hit=(((tab.p >= 0.5).astype(int) == tab.y).astype(float)))
    per_seed = t.groupby(["split_seed", "extractant", "chemotype"])["hit"].mean().reset_index()
    return per_seed.groupby(["extractant", "chemotype"])["hit"].mean().reset_index()


def const_units(tab, value=1):
    t = tab.assign(hit=((tab.y == value).astype(float)))
    per_seed = t.groupby(["split_seed", "extractant", "chemotype"])["hit"].mean().reset_index()
    return per_seed.groupby(["extractant", "chemotype"])["hit"].mean().reset_index()


def boot(u_a, u_b=None, reps=10000, seed=8675309):
    """Paired chemotype-blocked bootstrap of macro accuracy (or of a - b)."""
    rng = np.random.default_rng(seed)
    u_a = u_a.sort_values("extractant").reset_index(drop=True)
    ch = sorted(set(u_a.chemotype))
    mem = [np.flatnonzero(u_a.chemotype.to_numpy() == c) for c in ch]
    a = u_a["hit"].to_numpy()
    if u_b is None:
        b = np.zeros_like(a)
    else:
        u_b = u_b.sort_values("extractant").reset_index(drop=True)
        assert (u_b.extractant.to_numpy() == u_a.extractant.to_numpy()).all()
        b = u_b["hit"].to_numpy()
    picks = rng.integers(0, len(ch), size=(reps, len(ch)))
    d = a - b
    draws = np.array([d[np.concatenate([mem[j] for j in row])].mean() for row in picks])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    frac = float((draws <= 0).mean())
    p = max(2 * min(frac, 1 - frac), 1.0 / reps)
    return {"value": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(p), "n_units": len(a), "n_chemotypes": len(ch)}
