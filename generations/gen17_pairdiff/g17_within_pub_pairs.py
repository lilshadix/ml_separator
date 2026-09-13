"""Gen17 lead probe: within-publication pairwise difference learning (PADRE / DeepDelta).

The claim under test
-------------------
Every pair (i, j) drawn from the *same publication* cancels the publication effect
exactly, so a model of ``delta a = a_i - a_j`` is the machine-learning analogue of
Chamberlain's conditional likelihood / the econometric within-transformation.  If the
64 condition columns carry any transferable signal about the amplitude of the
lanthanide curve, it must show up here, because here it is not confounded with which
paper the row came from.

The leak rule is DeepDelta's (Fralish et al., J. Cheminform. 15:101, 2023): split into
folds FIRST, cross-merge inside each fold, never pair a train cell with a test cell.
Here the fold is a publication, so the rule is automatic -- a within-publication pair
is entirely inside one fold by construction.

What this script prints
-----------------------
1.  The pair census: how many within-publication pairs exist, split by whether the
    extractant changed and whether any condition column changed.
2.  The noise floor of ``delta a`` propagated from the corpus replicate spread.
3.  Leave-one-publication-out MAE of ``delta a`` for several feature sets and two
    estimators, each against the null ``delta a = 0``, with a publication-clustered
    bootstrap of the difference (pairs are massively non-independent -- one paper with
    m cells contributes m(m-1) of them -- so the bootstrap MUST resample publications).
4.  The cleanest contrast in the corpus: same publication, identical condition vector,
    different extractant.  Lab effect and condition effect both cancel exactly, so
    anything left is pure ligand structure.

Run from the repository root:
    .venv/Scripts/python.exe -u gen17_pairdiff/g17_within_pub_pairs.py

Runtime is a few minutes single-threaded on the i7-8750H; it needs ~1.5 GB of RAM
because ``gen14_direction/cache/bench.pkl`` holds the 2048-column ECFP block.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gen13_separation"))

from gen13sep import basis as Bs  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402

BENCH = ROOT / "gen14_direction" / "cache" / "bench.pkl"
OUT = Path(__file__).resolve().parent / "results"
MIN_METALS = 5
N_BOOT = 400


# ----------------------------------------------------------------------------------
def load():
    d = pickle.load(open(BENCH, "rb"))
    frame = d.frame
    coef = d.coef                       # (cells, 2): a = radius, b = radius^2
    centred = Bs.centre_rows(d.Y)
    keep = np.flatnonzero(d.n_obs >= MIN_METALS)
    blocks = {
        "COND": np.nan_to_num(d.frames["COND"].to_numpy(float)),
        "MASSACT": np.nan_to_num(d.frames["MASSACT"].to_numpy(float)),
        "DONORS": np.nan_to_num(d.frames["DONORS"].to_numpy(float)),
    }
    coord = d.frames["COORD"]
    topo_cols = [c for c in coord.columns
                 if c.startswith("coord__dist__") or c.startswith("coord__arm__")]
    blocks["TOPO39"] = np.nan_to_num(coord[topo_cols].to_numpy(float))
    return frame, coef, centred, d.basis, keep, blocks


def pair_table(frame, keep, cond_mat, ext):
    """Every ordered within-publication pair of well-determined cells."""
    pub = frame["publication_id"].to_numpy()
    rows = []
    for p in np.unique(pub[keep]):
        idx = keep[pub[keep] == p]
        if len(idx) < 2:
            continue
        for i in idx:
            for j in idx:
                if i != j:
                    rows.append((p, i, j))
    P = pd.DataFrame(rows, columns=["pub", "i", "j"])
    I, J = P["i"].to_numpy(), P["j"].to_numpy()
    P["same_ext"] = ext[I] == ext[J]
    P["cond_diff"] = (np.abs(cond_mat[I] - cond_mat[J]) > 1e-9).any(axis=1)
    return P


def amplitude_noise(frame, centred, basis, ridge=0.5):
    """Propagate the per-(cell, metal) replicate spread into sd(a)."""
    rs = frame[[f"repsd__{m}" for m in LANTHANIDES]].to_numpy(float)
    nr = frame[[f"nrep__{m}" for m in LANTHANIDES]].to_numpy(float)
    med = float(np.nanmedian(rs[rs > 0]))
    out = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        m = ~np.isnan(centred[i])
        k = int(m.sum())
        if k < 3:
            continue
        Bm = basis[:, m].T
        W = np.linalg.solve(Bm.T @ Bm + ridge * np.eye(basis.shape[0]), Bm.T)
        s = np.where(np.isnan(rs[i, m]) | (rs[i, m] <= 0), med, rs[i, m])
        n = np.where(np.isnan(nr[i, m]) | (nr[i, m] < 1), 1.0, nr[i, m])
        se2 = (s / np.sqrt(n)) ** 2
        v = se2 * (1 - 2 / k) + se2.sum() / k ** 2      # variance after row centring
        out[i] = np.sqrt(float((W[0] ** 2 * v).sum()))
    return out, med


def lopo(X, y, groups, make_model):
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if tr.sum() < 40 or te.sum() == 0:
            continue
        m = make_model()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def clustered_bootstrap(delta_loss, groups, ok, rng, n=N_BOOT):
    gs = np.unique(groups[ok])
    by = {g: np.flatnonzero(ok & (groups == g)) for g in gs}
    draws = [float(np.mean(np.concatenate([delta_loss[by[g]] for g in rng.choice(gs, len(gs), True)])))
             for _ in range(n)]
    return float(np.mean(draws)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ----------------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, coef, centred, basis, keep, blocks = load()
    a = coef[:, 0]
    ext = frame["extractant"].to_numpy()
    CM = np.hstack([blocks["COND"], blocks["MASSACT"]])
    TP, DN = blocks["TOPO39"], blocks["DONORS"]

    sa, med_rep = amplitude_noise(frame, centred, basis)
    print(f"median replicate sd of log D            {med_rep:.4f}")
    print(f"propagated sd(a) per cell (median)      {np.nanmedian(sa):.4f}")
    print(f"=> |delta a| expected from noise alone  {np.nanmedian(sa) * np.sqrt(2) * np.sqrt(2 / np.pi):.4f}")

    P = pair_table(frame, keep, CM, ext)
    I, J = P["i"].to_numpy(), P["j"].to_numpy()
    y = a[I] - a[J]
    grp = P["pub"].to_numpy()
    print(f"\nwithin-publication ordered pairs        {len(P)} "
          f"from {P['pub'].nunique()} publications")
    print(P.groupby(["same_ext", "cond_diff"])
           .apply(lambda d: pd.Series({"n": len(d), "null_MAE": np.abs(a[d.i] - a[d.j]).mean()}),
                  include_groups=False).to_string())
    print(f"largest publication contributes         "
          f"{P['pub'].value_counts().iloc[0] / len(P):.3f} of all pairs")

    strata = {
        "all": np.ones(len(P), bool),
        "sameE/dCond": P["same_ext"].to_numpy() & P["cond_diff"].to_numpy(),
        "diffE/sameC": ~P["same_ext"].to_numpy() & ~P["cond_diff"].to_numpy(),
        "diffE/dCond": ~P["same_ext"].to_numpy() & P["cond_diff"].to_numpy(),
    }
    feature_sets = {
        "dCOND72": CM[I] - CM[J],
        "TOPOij+dCOND": np.hstack([TP[I], TP[J], CM[I] - CM[J]]),
        "dTOPO+dCOND": np.hstack([TP[I] - TP[J], CM[I] - CM[J]]),
        "dTOPO+dDONORS": np.hstack([TP[I] - TP[J], DN[I] - DN[J]]),
    }
    estimators = {
        "ET": lambda: ExtraTreesRegressor(n_estimators=200, min_samples_leaf=8,
                                          random_state=0, n_jobs=2),
        "ridge": lambda: Ridge(alpha=10.0),
    }

    rng = np.random.default_rng(1)
    rows = []
    print("\n--- leave-one-publication-out delta-a, vs the null delta-a = 0 ---")
    for fname, X in feature_sets.items():
        for ename, mk in estimators.items():
            pred = lopo(X, y, grp, mk)
            ok = ~np.isnan(pred)
            dl = np.abs(y - pred) - np.abs(y)
            mean, lo, hi = clustered_bootstrap(dl, grp, ok, rng)
            print(f"{fname:14s} {ename:5s} n={ok.sum():4d} "
                  f"null {np.abs(y[ok]).mean():.4f} model {np.abs(y[ok] - pred[ok]).mean():.4f} "
                  f"delta {dl[ok].mean():+.4f}  pub-boot [{lo:+.4f}, {hi:+.4f}]", flush=True)
            for sname, mask in strata.items():
                m = ok & mask
                if m.sum() < 20:
                    continue
                rows.append({"features": fname, "estimator": ename, "stratum": sname,
                             "n_pairs": int(m.sum()),
                             "null_mae": float(np.abs(y[m]).mean()),
                             "model_mae": float(np.abs(y[m] - pred[m]).mean()),
                             "delta_mae": float(dl[m].mean()),
                             "boot_lo": lo if sname == "all" else np.nan,
                             "boot_hi": hi if sname == "all" else np.nan})
                if sname != "all":
                    print(f"    {sname:12s} n={m.sum():4d} null {np.abs(y[m]).mean():.4f} "
                          f"model {np.abs(y[m] - pred[m]).mean():.4f} delta {dl[m].mean():+.4f}")
    pd.DataFrame(rows).to_csv(OUT / "g17_within_pub_pairs.csv", index=False)
    print(f"\nwrote {OUT / 'g17_within_pub_pairs.csv'}")
    print("\nFALSIFICATION: the lead survives only if some row has delta_mae < 0 on the "
          "'all' stratum with a publication-clustered bootstrap interval that excludes 0.")


if __name__ == "__main__":
    main()
