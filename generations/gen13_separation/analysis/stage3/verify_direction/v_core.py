"""Independent re-derivation machinery for the direction-of-selectivity claim.

Nothing here reads the original s3_direction* scripts.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("D:/ml_separator_gh/generations/gen13_separation")
OUT = ROOT / "analysis" / "stage3" / "verify_direction"
sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

LEAN = ("COND", "MASSACT", "PHYSCHEM", "DONORS", "COORD")
CHEM = ("PHYSCHEM", "DONORS", "COORD")


def load():
    return pickle.load(open(OUT / "bench_cache.pkl", "rb"))


def topo_columns(frames):
    return [c for c in frames["COORD"].columns
            if c.startswith("coord__dist__") or c.startswith("coord__arm__")]


def design_matrix(frames, cols, mask):
    X = pd.concat([frames[b] for b in ("COND", "MASSACT", "PHYSCHEM", "DONORS", "ECFP", "LIG2D", "COORD")], axis=1)
    return X.loc[mask, cols].to_numpy(dtype=float)


def block_columns(frames, blocks):
    return [c for b in blocks for c in frames[b].columns]


def ols_coef(Y, basis):
    """Plain least-squares (no ridge) coefficient on the observed metals of each centred row."""
    centred = Y - np.nanmean(Y, axis=1, keepdims=True)
    out = np.full((len(Y), basis.shape[0]), np.nan)
    for i in range(len(Y)):
        m = ~np.isnan(centred[i])
        if m.sum() < basis.shape[0]:
            continue
        B = basis[:, m].T
        out[i] = np.linalg.lstsq(B, centred[i][m], rcond=None)[0]
    return out


def chemotype_weights(groups):
    s = pd.Series(groups)
    n = s.map(s.value_counts())
    w = 1.0 / n.to_numpy(dtype=float)
    return w * (len(groups) / w.sum())


def make_folds(frame, design="BP", seeds=None):
    from gen13sep.splits import all_folds
    f = frame.reset_index(drop=True)
    return all_folds(f, design=design) if seeds is None else all_folds(f, design=design, seeds=seeds)


def _prep(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    A = np.where(np.isnan(Xtr), med, Xtr)
    B = np.where(np.isnan(Xte), med, Xte)
    mu, sd = A.mean(0), A.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (A - mu) / sd, (B - mu) / sd


def fit_predict(kind, Xtr, ytr, wtr, Xte, seed):
    A, B = _prep(Xtr, Xte)
    if len(np.unique(ytr)) < 2:
        return np.full(len(B), ytr[0]), np.full(len(B), float(ytr[0]))
    if kind == "logit":
        m = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
        m.fit(A, ytr, sample_weight=wtr)
        return m.predict(B), m.predict_proba(B)[:, 1]
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=2)
        m.fit(A, ytr, sample_weight=wtr)
        return m.predict(B), m.predict_proba(B)[:, 1]
    raise ValueError(kind)


def run(frame, y, X, groups, kind="logit", design="BP", seeds=None):
    """Returns a long prediction table: one row per (seed, cell)."""
    f = frame.reset_index(drop=True)
    folds = make_folds(f, design=design, seeds=seeds)
    w_all = None
    rows = []
    for fold in folds:
        tr, te = fold.train_index, fold.test_index
        if len(tr) == 0:
            continue
        w = chemotype_weights(groups[tr])
        pred, prob = fit_predict(kind, X[tr], y[tr], w, X[te], fold.model_seed)
        for j, i in enumerate(te):
            rows.append({"split_seed": fold.seed, "fold": fold.fold, "row": int(i),
                         "cell_id": f["cell_id"].iat[i], "extractant": f["extractant"].iat[i],
                         "chemotype": f["chemotype"].iat[i], "y": int(y[i]),
                         "pred": int(pred[j]), "prob": float(prob[j]),
                         "n_train": len(tr)})
    return pd.DataFrame(rows)


def per_extractant_acc(tab, col="pred"):
    """One accuracy per extractant: mean over its cells, averaged over seeds."""
    hit = (tab[col] == tab["y"]).astype(float)
    t = tab.assign(hit=hit)
    per_seed = t.groupby(["split_seed", "extractant"])["hit"].mean().reset_index()
    per_ext = per_seed.groupby("extractant")["hit"].mean()
    chem = t.groupby("extractant")["chemotype"].first()
    return pd.DataFrame({"acc": per_ext, "chemotype": chem})


def macro(tab, col="pred"):
    return float(per_extractant_acc(tab, col)["acc"].mean())


def baseline_table(tab, value=1):
    """`value` is the label of 'heavy-selective'."""
    b = tab.copy()
    b["pred"] = value
    return b


def blocked_bootstrap(pe_model, pe_base, n=4000, seed=20260908):
    """Resample CHEMOTYPES with replacement; paired gain in extractant-macro accuracy."""
    rng = np.random.default_rng(seed)
    chems = np.array(sorted(pe_model["chemotype"].unique()))
    idx = {c: np.flatnonzero(pe_model["chemotype"].to_numpy() == c) for c in chems}
    am = pe_model["acc"].to_numpy(); ab = pe_base["acc"].to_numpy()
    gains, accs, bases = [], [], []
    for _ in range(n):
        pick = rng.choice(len(chems), size=len(chems), replace=True)
        sel = np.concatenate([idx[chems[p]] for p in pick])
        gains.append(am[sel].mean() - ab[sel].mean())
        accs.append(am[sel].mean()); bases.append(ab[sel].mean())
    g = np.asarray(gains)
    lo, hi = np.percentile(g, [2.5, 97.5])
    frac_le0 = float((g <= 0).mean())
    p = float(min(1.0, 2 * min(frac_le0, 1 - frac_le0)))
    p = max(p, 2.0 / n)
    return {"gain": float(am.mean() - ab.mean()), "lo": float(lo), "hi": float(hi),
            "p_two_sided": p, "frac_gain_le_0": frac_le0,
            "acc_lo": float(np.percentile(accs, 2.5)), "acc_hi": float(np.percentile(accs, 97.5))}
