"""Gen16: anchor regression with the publication as the anchor.

Rothenhaeusler, Meinshausen, Buehlmann & Peters, "Anchor regression: heterogeneous data meets
causality", JRSS-B 83(2):215-246, 2021 (arXiv:1801.06229), define

    b(gamma) = argmin_b  ||(I - Pi_A)(y - Xb)||^2  +  gamma * ||Pi_A (y - Xb)||^2

with Pi_A the projection onto the anchor's column span.  gamma = 0 partials the anchor out (the
within / fixed-effects fit), gamma = 1 is OLS, gamma -> infinity is two-stage least squares, i.e.
the between / instrument limit.  Because Pi_A is a projection the objective equals
||W_gamma (y - Xb)||^2 with

    W_gamma = I - (1 - sqrt(gamma)) Pi_A ,

so the estimator is ordinary least squares on (W_gamma X, W_gamma y): one line of linear algebra
and any existing L2 estimator on top.  Theorem 1 of that paper makes b(gamma) minimax optimal
against shift interventions of bounded strength in span(A) -- and design BP holds out whole
publications, which is exactly a shift to an unseen level of A.

Here A = publication identity of the *training* fold.  Weights are the frozen chemotype-balanced
cell weights, so the projection is taken in the W-inner product, Pi = A (A' W A)^+ A' W, which
keeps ||W_gamma r||_W^2 = ||(I-Pi) r||_W^2 + gamma ||Pi r||_W^2.

The binary arm follows Kook, Sick & Buehlmann, "Distributional anchor regression", Statistics and
Computing 32:39, 2022 (DOI 10.1007/s11222-022-10097-z): penalise the log-likelihood by
xi * ||Pi_A r||^2 / n with r the *score residuals* and xi = (gamma - 1) / 2.  For a binary logit
the score residual w.r.t. an extra intercept is exactly (y - p), so the arm below is their
estimator specialised to a one-parameter transformation model.

Usage:  python gen16_anchor/scripts/g16_anchor.py [design[,design...]] [--quick]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen14_direction"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen13_separation"))

from gen14 import dirbench as db                                              # noqa: E402
from gen14 import models as M                                                 # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame, cell_weights   # noqa: E402
from gen13sep.inference import paired_contrasts                               # noqa: E402
from gen13sep.metrics import per_extractant, summarise                        # noqa: E402
from gen13sep.splits import all_folds                                         # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
GAMMAS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1e4)
ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)


# ----------------------------------------------------------------------------------------
# the anchor transform
# ----------------------------------------------------------------------------------------
def anchor_operator(pub, w):
    """W-orthogonal projection onto the span of the publication indicators, Pi = A(A'WA)^+A'W."""
    levels = np.unique(pub)
    A = (pub[:, None] == levels[None, :]).astype(float)
    AtW = A.T * w[None, :]
    return A @ np.linalg.pinv(AtW @ A) @ AtW


def anchor_transform(Z, gamma, Pi):
    """W_gamma Z with W_gamma = I - (1 - sqrt(gamma)) Pi."""
    return Z - (1.0 - np.sqrt(gamma)) * (Pi @ Z)


# ----------------------------------------------------------------------------------------
# arm 1: anchor ridge on the signed amplitude
# ----------------------------------------------------------------------------------------
def _prep(Xtr, Xte):
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    return sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))


def anchor_ridge(Xtr, y, w, pub, Xte, gamma, alpha):
    """Anchor regression by OLS on transformed data, with an L2 penalty; predicts the amplitude."""
    Ztr, Zte = _prep(Xtr, Xte)
    Pi = anchor_operator(pub, w)
    Zt = anchor_transform(np.hstack([np.ones((len(Ztr), 1)), Ztr]), gamma, Pi)
    yt = anchor_transform(y, gamma, Pi)
    s = np.sqrt(w)[:, None]
    m = Ridge(alpha=alpha, fit_intercept=False).fit(Zt * s, yt * s.ravel())
    b = m.coef_
    return b[0] + Zte @ b[1:]


def anchor_ridge_lopo_alpha(Xtr, y, w, pub, gamma, alphas=ALPHAS):
    """Pick alpha by leave-one-publication-out *inside the training fold* at this gamma."""
    err = np.zeros(len(alphas))
    for p in np.unique(pub):
        te = pub == p
        tr = ~te
        if te.sum() == 0 or tr.sum() < 30 or len(np.unique(pub[tr])) < 3:
            continue
        for i, a in enumerate(alphas):
            pred = anchor_ridge(Xtr[tr], y[tr], w[tr], pub[tr], Xtr[te], gamma, a)
            err[i] += float(np.sum(w[te] * np.abs(pred - y[te])))
    return float(alphas[int(np.argmin(err))])


# ----------------------------------------------------------------------------------------
# arm 2: anchor logistic (Kook et al. score-residual penalty, binary logit)
# ----------------------------------------------------------------------------------------
def anchor_logit(Xtr, y, w, pub, Xte, gamma, lam=1.0):
    """min -sum w loglik / n + lam ||b||^2 / 2n + xi ||Pi r||_W^2 / n, r = y - p, xi = (gamma-1)/2."""
    Ztr, Zte = _prep(Xtr, Xte)
    Pi = anchor_operator(pub, w)
    n = len(Ztr)
    Z = np.hstack([np.ones((n, 1)), Ztr])
    xi = 0.5 * (gamma - 1.0)
    WPi = Pi * w[:, None]          # W Pi ; for a W-orthogonal projection Pi' W Pi = W Pi

    def obj(b):
        eta = np.clip(Z @ b, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        ll = float(np.sum(w * (y * eta - np.logaddexp(0.0, eta)))) / n
        r = y - p
        pen = xi * float(r @ (WPi @ r)) / n
        val = -ll + 0.5 * lam * float(b[1:] @ b[1:]) / n + pen
        g = -(Z.T @ (w * r)) / n
        g[1:] += lam * b[1:] / n
        g -= (2.0 * xi / n) * (Z.T @ ((WPi @ r) * p * (1.0 - p)))
        return val, g

    res = minimize(obj, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9})
    b = res.x
    return 1.0 / (1.0 + np.exp(-np.clip(b[0] + Zte @ b[1:], -35, 35)))


# ----------------------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------------------
def main(designs, quick=False):
    bench = db.load()
    FS = db.feature_sets(bench)
    fr = bench.frame
    amp = bench.coef[:, 0]
    rich = fr.n_metals.to_numpy() >= db.MIN_METALS
    pub_all = fr.publication_id.astype(str).to_numpy()
    X_lean = bench.matrix(LEAN_BLOCKS)
    gammas = (0.0, 0.5, 1.0, 5.0, 100.0) if quick else GAMMAS
    cols = {"COND_MA": np.concatenate([FS["COND64"], FS["MASSACT8"]]),
            "TOPO39": FS["TOPO39"],
            "ALL111": np.concatenate([FS["TOPO39"], FS["COND64"], FS["MASSACT8"]])}

    for design in designs:
        t0 = time.time()
        parts, chosen = [], []
        for f in all_folds(fr, design=design):
            tr, te = f.train_index, f.test_index
            pairs = _pair_frame(fr, bench.Y, te, f.seed, f.fold)
            if pairs.empty:
                continue
            rtr = tr[rich[tr]]
            if len(rtr) < db.MIN_TRAIN or len(np.unique(pub_all[rtr])) < 4:
                continue
            w = cell_weights(bench.groups[tr], bench.n_obs[tr])
            wr = cell_weights(bench.groups[rtr], bench.n_obs[rtr])
            pr = pub_all[rtr]
            bmean = float(np.average(bench.coef[tr, 1], weights=w))
            amean = float(np.average(bench.coef[tr, 0], weights=w))
            magc = float(np.average(np.abs(amp[rtr]), weights=wr))
            flat = np.full(len(te), bmean)
            curves = {}
            for blk, idx in cols.items():
                Xtr, Xte = X_lean[np.ix_(rtr, idx)], X_lean[np.ix_(te, idx)]
                for g in gammas:
                    a_lo = anchor_ridge_lopo_alpha(Xtr, amp[rtr], wr, pr, g)
                    chosen.append({"design": design, "seed": f.seed, "fold": f.fold,
                                   "block": blk, "gamma": g, "alpha": a_lo})
                    ah = anchor_ridge(Xtr, amp[rtr], wr, pr, Xte, g, a_lo)
                    curves[f"ARIDGE_{blk}_g{g:g}"] = np.c_[ah, flat] @ bench.basis
                    curves[f"ASIGN_{blk}_g{g:g}"] = np.c_[np.where(ah < 0, -magc, magc), flat] @ bench.basis
                    if g >= 1.0:
                        p = anchor_logit(Xtr, (amp[rtr] < 0).astype(float), wr, pr, Xte, g)
                        curves[f"ALOGIT_{blk}_g{g:g}"] = np.c_[np.where(p >= 0.5, -magc, magc), flat] @ bench.basis
            pg = M.dir_logistic()(X_lean[np.ix_(rtr, FS["TOPO39"])], amp[rtr], wr,
                                  bench.groups[rtr], X_lean[np.ix_(te, FS["TOPO39"])],
                                  f.model_seed, fr.extractant.to_numpy()[rtr])
            curves["G14_DIR_HARD"] = np.c_[np.where(pg >= 0.5, -magc, magc), flat] @ bench.basis
            curves["MEAN_CURVE"] = np.tile(np.array([amean, bmean]) @ bench.basis, (len(te), 1))
            ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
            t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
            for nm, cv in curves.items():
                t[nm] = cv[loc, ia] - cv[loc, ib]
            parts.append(t)

        table = pd.concat(parts, ignore_index=True)
        names = [c for c in table.columns if c.startswith(("ARIDGE_", "ASIGN_", "ALOGIT_"))] \
            + ["G14_DIR_HARD", "MEAN_CURVE"]
        pe = per_extractant(table, names)
        board = summarise(pe, table, names)
        board.insert(0, "design", design)
        RESULTS.mkdir(parents=True, exist_ok=True)
        board.to_csv(RESULTS / f"g16_anchor_{design}.csv", index=False)
        pd.DataFrame(chosen).to_csv(RESULTS / f"g16_alpha_{design}.csv", index=False)
        print(f"\n=== design {design} ({time.time() - t0:.0f}s) ===")
        print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd",
                     "macro_sign_acc_strong"]].round(4).to_string(index=False))
        comps = {f"{n}_vs_G14": ("G14_DIR_HARD", n) for n in names if n != "G14_DIR_HARD"}
        r = paired_contrasts(pe, comps, value="mae_all", replicates=4000)
        r.insert(0, "design", design)
        r.to_csv(RESULTS / f"g16_anchor_contrasts_{design}.csv", index=False)
        print(r[["comparison", "point", "ci95_low", "ci95_high", "p_two_sided"]].round(4).to_string(index=False))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0].split(",") if args else ["BP"], quick="--quick" in sys.argv)
