"""Gen16: anchor regression with the publication as the anchor -- exact, fast, whole gamma path.

Rothenhaeusler, Meinshausen, Buehlmann & Peters, "Anchor regression: heterogeneous data meets
causality", JRSS-B 83(2):215-246, 2021 (arXiv:1801.06229), eq. (4):

    b(gamma) = argmin_b  E[((I - P_A)(Y - X'b))^2] + gamma * E[((P_A)(Y - X'b))^2]

and their eq. (20) says this is OLS on Xt = (I - Pi_A)X + sqrt(gamma) Pi_A X, likewise for Y.
Their stated limits (paper, Section 2):  b^0 = b_PA (partialling out / within / fixed effects),
b^1 = b_OLS (pooled), b^inf = b_IV (between / two-stage least squares).
NOTE: gamma -> inf is the BETWEEN limit, not the within limit.

Why a rewrite instead of g16_anchor.py: because Pi_A is block diagonal by publication, the two
Gram matrices are SUMS OVER PUBLICATIONS,

    S_w = sum_p sum_{i in p} w_i (z_i - zbar_p)(z_i - zbar_p)'      (within scatter)
    S_b = sum_p omega_p (zbar_p - mu)(zbar_p - mu)'                 (between scatter)

and the anchor-ridge solution for every (gamma, alpha) is

    b(gamma, alpha) = (S_w + gamma S_b + alpha I)^{-1} (c_w + gamma c_b).

So one pass of per-publication sufficient statistics gives the exact leave-one-publication-out
alpha search over the entire gamma grid: drop publication q by subtracting its terms.  That turns
g16_anchor.py's ~166k re-fitted ridges into ~370 eigendecompositions per fold and block, and it
removes a real confound in the naive version -- with alpha held fixed, scaling the design by
sqrt(gamma) silently rescales the effective penalty to alpha/gamma, so a "gamma path" at fixed
alpha is partly just a ridge path.  Alpha is re-selected at every gamma here.

Usage:  python gen16_anchor/scripts/g16_anchor_fast.py [BP,A,B,BR,BQ] [--blocks COND_MA,TOPO39]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gen14_direction"))
sys.path.insert(0, str(ROOT / "gen13_separation"))

from gen14 import dirbench as db                                              # noqa: E402
from gen14 import models as M                                                 # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame, cell_weights   # noqa: E402
from gen13sep.inference import paired_contrasts                               # noqa: E402
from gen13sep.metrics import per_extractant, summarise                        # noqa: E402
from gen13sep.splits import all_folds                                         # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
#: gamma = 0 within/fixed-effects, 1 pooled OLS, large = between/IV limit.
GAMMAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0)
# the weights are normalised to sum 1 and the columns are scaled so that diag(S_w + S_b) = 1, so a
# typical Gram eigenvalue is O(1): the useful alpha range is around 1, not around n.
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


# ----------------------------------------------------------------------------------------
# per-publication sufficient statistics
# ----------------------------------------------------------------------------------------
class PubStats:
    """Everything the anchor fit needs, decomposed by publication so LOPO is a subtraction."""

    def __init__(self, Z: np.ndarray, y: np.ndarray, w: np.ndarray, pub: np.ndarray):
        self.levels, inv = np.unique(pub, return_inverse=True)
        P, d = len(self.levels), Z.shape[1]
        self.omega = np.bincount(inv, weights=w, minlength=P)
        self.zbar = np.zeros((P, d))
        for j in range(d):
            self.zbar[:, j] = np.bincount(inv, weights=w * Z[:, j], minlength=P)
        self.zbar /= np.maximum(self.omega, 1e-12)[:, None]
        self.ybar = np.bincount(inv, weights=w * y, minlength=P) / np.maximum(self.omega, 1e-12)
        self.inv, self.Z, self.y, self.w = inv, Z, y, w
        self.rows = [np.flatnonzero(inv == p) for p in range(P)]
        # only the TOTAL within scatter is stored (d x d); a publication's own contribution is
        # recomputed on demand, so leave-one-publication-out costs O(n_p d^2), not O(P d^2) memory.
        self.Sw_tot = np.zeros((d, d))
        self.cw_tot = np.zeros(d)
        for p in range(P):
            S, c = self._within_of(p)
            self.Sw_tot += S
            self.cw_tot += c

    def _within_of(self, p: int):
        m = self.rows[p]
        C = self.Z[m] - self.zbar[p]
        ww = self.w[m]
        Cw = C * ww[:, None]
        return Cw.T @ C, Cw.T @ (self.y[m] - self.ybar[p])

    def aggregate(self, drop: int | None = None):
        """Grams, centring and scaling with publication ``drop`` left out (None = all)."""
        keep = np.ones(len(self.omega), dtype=bool)
        if drop is not None:
            keep[drop] = False
        om = self.omega[keep]
        Om = om.sum()
        zb, yb = self.zbar[keep], self.ybar[keep]
        mu = (om[:, None] * zb).sum(0) / Om
        ym = float((om * yb).sum() / Om)
        D = zb - mu
        Sw, cw = self.Sw_tot, self.cw_tot
        if drop is not None:
            S, c = self._within_of(drop)
            Sw, cw = Sw - S, cw - c
        Sb = (D * om[:, None]).T @ D
        cb = (D * om[:, None]).T @ (yb - ym)
        sd = np.sqrt(np.maximum(np.diag(Sw + Sb), 0.0) / Om)
        sd[sd < 1e-9] = 1.0
        s = 1.0 / sd
        return (Sw * s[:, None] * s[None, :], Sb * s[:, None] * s[None, :],
                cw * s, cb * s, mu, sd, ym)


def _coefs(Sw, Sb, cw, cb, gamma, alphas):
    """b(gamma, alpha) for every alpha, from one eigendecomposition of S_w + gamma S_b."""
    S = Sw + gamma * Sb
    ev, V = np.linalg.eigh(0.5 * (S + S.T))
    Vc = V.T @ (cw + gamma * cb)
    return [V @ (Vc / (ev + a)) for a in alphas]


def anchor_path(Z, y, w, pub, Zte, gammas=None, alphas=None):
    """For each gamma: pick alpha by exact leave-one-publication-out, then predict the test rows.

    Returns {gamma: (prediction, chosen alpha, lopo weighted MAE at that alpha)}.
    """
    gammas = GAMMAS if gammas is None else gammas
    alphas = ALPHAS if alphas is None else alphas
    st = PubStats(Z, y, w, pub)
    P = len(st.levels)
    usable = P >= 4
    # the LOPO aggregates do not depend on gamma, so build them once for all gammas
    lopo = [(st.aggregate(q), st.rows[q]) for q in range(P)] if usable else []
    out = {}
    for g in gammas:
        err = np.zeros(len(alphas))
        for (Sw, Sb, cw, cb, mu, sd, ym), m in lopo:
            Zq = (st.Z[m] - mu) / sd
            for i, b in enumerate(_coefs(Sw, Sb, cw, cb, g, alphas)):
                err[i] += float(np.sum(st.w[m] * np.abs(ym + Zq @ b - st.y[m])))
        k = int(np.argmin(err)) if usable else len(alphas) // 2
        Sw, Sb, cw, cb, mu, sd, ym = st.aggregate(None)
        b = _coefs(Sw, Sb, cw, cb, g, [alphas[k]])[0]
        out[g] = (ym + ((Zte - mu) / sd) @ b, alphas[k], float(err[k]))
    return out


# ----------------------------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------------------------
def main(designs, blocks):
    bench = db.load()
    FS = db.feature_sets(bench)
    fr = bench.frame
    amp = bench.coef[:, 0]
    rich = fr.n_metals.to_numpy() >= db.MIN_METALS
    pub_all = fr.publication_id.astype(str).to_numpy()
    X = bench.matrix(LEAN_BLOCKS)
    cols = {"COND_MA": np.concatenate([FS["COND64"], FS["MASSACT8"]]),
            "TOPO39": FS["TOPO39"],
            "ALL111": np.concatenate([FS["TOPO39"], FS["COND64"], FS["MASSACT8"]])}
    cols = {k: v for k, v in cols.items() if k in blocks}
    RESULTS.mkdir(parents=True, exist_ok=True)

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
            # LogisticRegression(C=1) balances its penalty against a likelihood that scales with
            # sum(sample_weight), so the G14 baseline must get the UNNORMALISED chemotype weights;
            # normalising them multiplies its effective penalty by n and drives it to the prior.
            wr_raw = cell_weights(bench.groups[rtr], bench.n_obs[rtr])
            wr = wr_raw / wr_raw.sum()
            pr = pub_all[rtr]
            bmean = float(np.average(bench.coef[tr, 1], weights=w))
            amean = float(np.average(bench.coef[tr, 0], weights=w))
            magc = float(np.average(np.abs(amp[rtr]), weights=wr))
            flat = np.full(len(te), bmean)
            curves = {}
            for blk, idx in cols.items():
                Xtr, Xte = X[np.ix_(rtr, idx)], X[np.ix_(te, idx)]
                med = np.nanmedian(np.where(np.isnan(Xtr), np.nan, Xtr), axis=0)
                med = np.nan_to_num(med, nan=0.0)
                Ztr = np.where(np.isnan(Xtr), med, Xtr)
                Zte = np.where(np.isnan(Xte), med, Xte)
                path = anchor_path(Ztr, amp[rtr], wr, pr, Zte)
                for g, (ah, a_lo, lo_err) in path.items():
                    chosen.append({"design": design, "seed": f.seed, "fold": f.fold,
                                   "block": blk, "gamma": g, "alpha": a_lo, "lopo_mae": lo_err})
                    curves[f"ARIDGE_{blk}_g{g:g}"] = np.c_[ah, flat] @ bench.basis
                    curves[f"ASIGN_{blk}_g{g:g}"] = (np.c_[np.where(ah < 0, -magc, magc), flat]
                                                     @ bench.basis)
            pg = M.dir_logistic()(X[np.ix_(rtr, FS["TOPO39"])], amp[rtr], wr_raw,
                                  bench.groups[rtr], X[np.ix_(te, FS["TOPO39"])],
                                  f.model_seed, fr.extractant.to_numpy()[rtr])
            curves["G14_DIR_HARD"] = np.c_[np.where(pg >= 0.5, -magc, magc), flat] @ bench.basis
            curves["MEAN_CURVE"] = np.tile(np.array([amean, bmean]) @ bench.basis, (len(te), 1))
            ia, ib, loc = pairs["ia"].to_numpy(), pairs["ib"].to_numpy(), pairs["cell_local"].to_numpy()
            t = pairs.drop(columns=["ia", "ib", "cell_local"]).copy()
            for nm, cv in curves.items():
                t[nm] = cv[loc, ia] - cv[loc, ib]
            parts.append(t)

        table = pd.concat(parts, ignore_index=True)
        names = [c for c in table.columns if c.startswith(("ARIDGE_", "ASIGN_"))] \
            + ["G14_DIR_HARD", "MEAN_CURVE"]
        pe = per_extractant(table, names)
        board = summarise(pe, table, names)
        board.insert(0, "design", design)
        board.to_csv(RESULTS / f"g16f_anchor_{design}.csv", index=False)
        pd.DataFrame(chosen).to_csv(RESULTS / f"g16f_alpha_{design}.csv", index=False)
        print(f"\n=== design {design} ({time.time() - t0:.0f}s) ===", flush=True)
        print(board[["arm", "macro_mae_extractant", "macro_mae_extractant_seed_sd",
                     "macro_sign_acc_strong"]].round(4).to_string(index=False), flush=True)
        comps = {f"{n}_vs_G14": ("G14_DIR_HARD", n) for n in names if n != "G14_DIR_HARD"}
        r = paired_contrasts(pe, comps, value="mae_all", replicates=4000)
        r.insert(0, "design", design)
        r.to_csv(RESULTS / f"g16f_contrasts_{design}.csv", index=False)
        print(r[["comparison", "point", "ci95_low", "ci95_high", "p_two_sided"]]
              .round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    blk = "COND_MA,TOPO39,ALL111"
    for a in sys.argv[1:]:
        if a.startswith("--blocks"):
            blk = a.split("=", 1)[1] if "=" in a else blk
    main(args[0].split(",") if args else ["BP"], blk.split(","))
