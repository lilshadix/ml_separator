"""gen21 methods.  Each is ``fn(block, cell, obs, k) -> level vector (14,)``; see ``evaluate.py``.

Attempt 1 -- predict the *observable* log SF, not the latent one ("OBS").
    gen20's ridge line and BLUP treat the target as the noise-free latent difference f(a) - f(b) and
    smooth through the readings.  But the scored target is the *observed* y(a) - y(b), and when the
    metal a has been read, its reading y(a) -- including its own measurement error eps(a) -- is
    already in the support.  Under the same Gaussian model the arms already assume (per-metal noise
    s2, R = s2 D D^T), the best predictor of the observable is
        E[y(a) - y(b) | supports] = f_hat(a) - f_hat(b) + E[eps(a) | supports] - E[eps(b) | supports],
    with E[eps | supports] = s2 D^T Var(y_sup)^-1 (y_sup - E y_sup) on read metals and 0 on unread
    ones.  No new parameter: RIDGE2's prior / lambda and BLUP's covariance / s2 are the fold's own.
"""
from __future__ import annotations

import numpy as np

from . import paths  # noqa: F401  (puts the frozen gen20 package on sys.path)
from gen20k import adapters as AD  # noqa: E402
from gen20k import cov as CV  # noqa: E402
from gen20k.data import N_LN  # noqa: E402


def ridge_obs(basis, support, prior, lam, s2):
    """RIDGE2 (Bayesian form: theta ~ N(prior, I/lam), y_sup ~ N(D B' theta, s2 D D')) plus the
    posterior mean of the per-metal noise at read metals.  Returns (latent curve, eps_hat)."""
    prior = np.asarray(prior, float)
    if support.k == 0:
        return prior @ basis, np.zeros(N_LN)
    D = AD.pair_matrix(support.pairs)
    A = D @ basis.T
    y = np.asarray(support.values, float)
    R = s2 * (D @ D.T)
    V = A @ A.T / lam + R
    w = np.linalg.solve(V, y - A @ prior)
    theta = prior + A.T @ w / lam
    eps = s2 * (D.T @ w)
    return theta @ basis, eps


def blup_obs(c0, cov, support, s2):
    if support.k == 0:
        return np.asarray(c0, float).copy(), np.zeros(N_LN)
    D = AD.pair_matrix(support.pairs)
    r = np.asarray(support.values, float) - D @ c0
    S = D @ cov @ D.T + s2 * (D @ D.T)
    w = np.linalg.lstsq(S, r, rcond=None)[0]
    return c0 + cov @ D.T @ w, s2 * (D.T @ w)


def attempt1_methods(bench, *, reselect: bool = False):
    """``reselect``: use the (noise factor, lambda) fitted for the OBS rule on the inner cross-fit
    (``block['fitted']``) instead of gen20's selection."""
    basis = bench.basis

    def _s2_lam(block):
        if reselect:
            f = block["fitted"]
            return block["noise_s2_robust"] * f["noise_factor"], f["ridge_lambda"]
        return block["s2"], block["ridge_lambda"]

    def ridge2_obs(block, cell, obs, k):
        s2, lam = _s2_lam(block)
        c, e = ridge_obs(basis, obs.support, block["prior"], lam, s2)
        return c + e

    def ridge2_same_params(block, cell, obs, k):        # latent RIDGE2 at the re-selected parameters (control)
        s2, lam = _s2_lam(block)
        return AD.ridge_line(basis, obs.support, block["prior"], lam, CV.noise_matrix(obs.support.pairs, s2, "shared"))

    def g14_blup_obs(block, cell, obs, k):
        c, e = blup_obs(cell["zero_g14"], block["cov"], obs.support, block["s2"])
        return c + e

    def flat_blup_obs(block, cell, obs, k):
        c, e = blup_obs(cell["zero_flat"], block["cov"], obs.support, block["s2"])
        return c + e

    if reselect:
        return {"RIDGE2+OBS@RESEL": ridge2_obs, "RIDGE2@RESEL": ridge2_same_params}
    return {"RIDGE2+OBS": ridge2_obs, "G14@BLUP20+OBS": g14_blup_obs, "FLAT@BLUP20+OBS": flat_blup_obs}


NOISE_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)      # gen20's grid
LAMBDA_GRID = (0.3, 1.0, 3.0, 10.0)               # gen20's grid
SELECT_KS = (1, 2, 3)


def attempt1_fit(bench, rec, *, ks=SELECT_KS):
    """Choose (noise factor, lambda) for RIDGE2+OBS on the inner cross-fit of the outer training
    fold: pooled squared error over strict-adjacent queries of inner-validation cells with
    k_max >= 3 at k in ``ks`` (DOPT_Q supports as cached)."""
    from . import evaluate as E
    basis = bench.basis
    best, scores = None, {}
    for f in NOISE_GRID:
        for lam in LAMBDA_GRID:
            sq, n = 0.0, 0
            for blk in rec["inner"]:
                s2 = blk["s2"] / rec["outer"]["noise_factor"] * f      # inner robust s2 times f
                for cell in blk["cells"]:
                    if cell["k_max"] < max(ks):
                        continue
                    seq = cell["seqs"]["DOPT_Q"]
                    for k in ks:
                        obs = E.observe(bench.Y[cell["ci"]], cell, seq, k)
                        c, e = ridge_obs(basis, obs.support, blk["prior"], lam, s2)
                        v = c + e
                        for a, b in cell["query"]:
                            d = (v[a] - v[b]) - (bench.Y[cell["ci"]][a] - bench.Y[cell["ci"]][b])
                            sq += d * d
                            n += 1
            scores[f"{f}|{lam}"] = sq / max(n, 1)
            if best is None or scores[f"{f}|{lam}"] < scores[best]:
                best = f"{f}|{lam}"
    f, lam = best.split("|")
    return {"noise_factor": float(f), "ridge_lambda": float(lam), "inner_mse": scores, "n_inner_cells":
            int(sum(1 for blk in rec["inner"] for c in blk["cells"] if c["k_max"] >= max(ks)))}


# ======================================================================================
# Attempt 2 -- hierarchical empirical-Bayes curve model ("HB")
# ======================================================================================
#   y_cell = mu * 1 + B' theta + delta + eps        (log D over the cell's metals)
#   theta ~ N(m, S_theta)  (shape coefficients),  delta ~ GP, Cov = tau2 * rho^|Z_i - Z_j|,
#   eps ~ N(0, sig2 I) (per-metal reading noise, part of the scored target),  mu free.
# All of (m, S_theta, tau2, rho, sig2) are estimated by chemotype-balanced weighted maximum
# likelihood of the within-cell contrasts of the OUTER TRAINING cells' observed curves.  The
# prediction for a query is the conditional mean of the observable y(a) - y(b) given the read
# contrasts -- the observable rule of attempt 1 falls out of the model (eps is in the cross term).

from scipy.optimize import minimize  # noqa: E402

from gen20k.data import Z_VEC  # noqa: E402

HINGE_Z = 64                                  # Gd: slope change at the half-filled shell


def hb_basis(bench, kind: str) -> np.ndarray:
    B = np.asarray(bench.basis, float)
    if kind == "B2":
        return B
    if kind == "B3H":
        h = np.maximum(0.0, Z_VEC - HINGE_Z).astype(float)
        h = h - h.mean()
        h = h / np.linalg.norm(h) * np.sqrt(N_LN)
        return np.vstack([B, h])
    raise ValueError(kind)


class HBModel:
    def __init__(self, basis: np.ndarray, use_gp: bool = True):
        self.B = basis
        self.p = basis.shape[0]
        self.use_gp = use_gp
        self.dZ = np.abs(Z_VEC[:, None] - Z_VEC[None, :]).astype(float)

    # ---- parameters ----
    def _unpack(self, x):
        p = self.p
        m = x[:p]
        L = np.zeros((p, p))
        L[np.tril_indices(p)] = x[p:p + p * (p + 1) // 2]
        d = np.diag_indices(p)
        L[d] = np.exp(L[d])
        S = L @ L.T
        o = p + p * (p + 1) // 2
        sig2 = np.exp(x[o])
        if self.use_gp:
            tau2, rho = np.exp(x[o + 1]), 1.0 / (1.0 + np.exp(-x[o + 2]))
        else:
            tau2, rho = 0.0, 0.0
        return m, S, tau2, rho, sig2

    def cov_full(self, S, tau2, rho, sig2, *, latent=False):
        C = self.B.T @ S @ self.B
        if self.use_gp and tau2 > 0:
            C = C + tau2 * rho ** self.dZ
        if not latent:
            C = C + sig2 * np.eye(N_LN)
        return C

    def fit(self, Y: np.ndarray, w: np.ndarray, *, min_metals: int = 3):
        ok = np.isfinite(Y).sum(1) >= min_metals
        Y, w = Y[ok], w[ok] / w[ok].mean()
        pats: dict = {}
        for i, row in enumerate(Y):
            key = tuple(np.flatnonzero(np.isfinite(row)))
            pats.setdefault(key, []).append(i)
        self.groups = []
        for key, idx in pats.items():
            M = np.array(key)
            n = len(M)
            P = np.hstack([-np.ones((n - 1, 1)), np.eye(n - 1)])       # differences to the first observed metal
            R = Y[np.ix_(idx, M)] @ P.T                                  # (n_cells, n-1)
            self.groups.append((M, P, R, w[idx]))
        p = self.p
        x0 = np.zeros(p + p * (p + 1) // 2 + (3 if self.use_gp else 1))
        x0[p:p + p * (p + 1) // 2][[sum(range(j + 2)) - 1 for j in range(p)]] = np.log(0.3)
        o = p + p * (p + 1) // 2
        x0[o] = np.log(0.05)
        if self.use_gp:
            x0[o + 1], x0[o + 2] = np.log(0.05), 0.0
        res = minimize(self._nll, x0, method="L-BFGS-B", options={"maxiter": 500})
        self.x = res.x
        self.fit_info = {"success": bool(res.success), "nll": float(res.fun), "n_cells": int(len(Y)), "nit": int(res.nit)}
        self.m, self.S, self.tau2, self.rho, self.sig2 = self._unpack(res.x)
        return self

    def _nll(self, x):
        m, S, tau2, rho, sig2 = self._unpack(x)
        C = self.cov_full(S, tau2, rho, sig2)
        mu = self.B.T @ m
        tot = 0.0
        for M, P, R, w in self.groups:
            Sg = P @ C[np.ix_(M, M)] @ P.T
            try:
                Lc = np.linalg.cholesky(Sg)
            except np.linalg.LinAlgError:
                return 1e12
            r = R - (P @ mu[M])[None, :]
            z = np.linalg.solve(Lc, r.T)
            logdet = 2.0 * np.log(np.diag(Lc)).sum()
            tot += float((w * (0.5 * logdet + 0.5 * (z * z).sum(0))).sum())
        return tot

    def predict(self, support, *, latent: bool = False) -> np.ndarray:
        mu = self.B.T @ self.m
        if support.k == 0:
            return mu
        C = self.cov_full(self.S, self.tau2, self.rho, self.sig2)
        Ccross = self.cov_full(self.S, self.tau2, self.rho, self.sig2, latent=latent)
        D = AD.pair_matrix(support.pairs)
        r = np.asarray(support.values, float) - D @ mu
        w = np.linalg.solve(D @ C @ D.T, r)
        return mu + Ccross @ D.T @ w

    def params(self) -> dict:
        return {"m": self.m.tolist(), "S_theta": self.S.tolist(), "tau2": float(self.tau2), "rho": float(self.rho),
                "sig2": float(self.sig2), **self.fit_info}


HB_VARIANTS = {"HB2": ("B2", False), "HB2+GP": ("B2", True), "HB3H+GP": ("B3H", True)}


def attempt2_fit(bench, rec):
    """ML on the outer training cells only (chemotype-balanced weights, gen13's convention)."""
    from gen13sep.amplitude_bench import cell_weights
    tr = np.asarray(rec["outer"]["train_index"])
    w = cell_weights(bench.groups[tr], bench.n_obs[tr])
    out = {}
    for name, (kind, gp) in HB_VARIANTS.items():
        out[name] = HBModel(hb_basis(bench, kind), use_gp=gp).fit(bench.Y[tr], w)
    return out


def attempt2_methods(bench):
    def make(name, latent=False):
        def fn(block, cell, obs, k):
            return block["fitted"][name].predict(obs.support, latent=latent)
        return fn
    m = {n: make(n) for n in HB_VARIANTS}
    m["HB2+GP@LATENT"] = make("HB2+GP", latent=True)
    m["HB3H+GP@LATENT"] = make("HB3H+GP", latent=True)
    return m


# ======================================================================================
# Attempt 3 -- the extractant-level hierarchy: pool k-shot reads across a held-out extractant's
# other condition cells in the same publication ("HBG", setting P)
# ======================================================================================
#   y_c = mu_c 1 + B'(theta_g + eta_c) + delta_g + nu_c + eps_c        (cells c of group g)
#   theta_g ~ N(m, S_g), eta_c ~ N(0, S_c), delta_g ~ GP(tau_g2, rho), nu_c ~ GP(tau_c2, rho),
#   eps_c ~ N(0, sig2 I); mu_c free per cell.  g = (extractant, publication).
# Fitted by weighted ML of the within-cell contrasts of the OUTER TRAINING groups, jointly per
# group.  Prediction: conditional mean of the observable y_c(a) - y_c(b) given the read contrasts
# of every cell of the group (each cell's own k + 1 reads; nothing a cell did not read).

class HBGroupModel:
    def __init__(self, basis: np.ndarray):
        self.B = basis
        self.p = basis.shape[0]
        self.dZ = np.abs(Z_VEC[:, None] - Z_VEC[None, :]).astype(float)

    def _unpack(self, x):
        p, q = self.p, self.p * (self.p + 1) // 2
        m = x[:p]

        def chol(v):
            L = np.zeros((p, p))
            L[np.tril_indices(p)] = v
            L[np.diag_indices(p)] = np.exp(L[np.diag_indices(p)])
            return L @ L.T
        Sg, Sc = chol(x[p:p + q]), chol(x[p + q:p + 2 * q])
        o = p + 2 * q
        tg, tc, rho, sig2 = np.exp(x[o]), np.exp(x[o + 1]), 1.0 / (1.0 + np.exp(-x[o + 2])), np.exp(x[o + 3])
        return m, Sg, Sc, tg, tc, rho, sig2

    def covs(self, x=None):
        m, Sg, Sc, tg, tc, rho, sig2 = self._unpack(self.x if x is None else x)
        K = rho ** self.dZ
        Cg = self.B.T @ Sg @ self.B + tg * K
        Cc = self.B.T @ Sc @ self.B + tc * K + sig2 * np.eye(N_LN)
        return m, Cg, Cc

    def fit(self, Y, w, groups, *, min_metals: int = 3):
        ok = np.isfinite(Y).sum(1) >= min_metals
        Y, w, groups = Y[ok], w[ok] / w[ok].mean(), np.asarray(groups)[ok]
        self.G = []
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            stack, cell_of, rows = [], [], []
            for j, i in enumerate(idx):
                M = np.flatnonzero(np.isfinite(Y[i]))
                stack.append(M); cell_of.append(np.full(len(M), j))
                rows.append((len(np.concatenate(stack[:-1])) if j else 0, M, Y[i, M]))
            st = np.concatenate(stack)
            co = np.concatenate(cell_of)
            nS = len(st)
            P = np.zeros((nS - len(idx), nS))
            r = np.zeros(nS - len(idx))
            row = 0
            for off, M, yv in rows:
                n = len(M)
                P[row:row + n - 1, off] = -1.0
                P[row:row + n - 1, off + 1:off + n] = np.eye(n - 1)
                r[row:row + n - 1] = yv[1:] - yv[0]
                row += n - 1
            same = co[:, None] == co[None, :]
            self.G.append((st, same, P, r, float(w[idx].mean())))
        p, q = self.p, self.p * (self.p + 1) // 2
        x0 = np.zeros(p + 2 * q + 4)
        diag = [sum(range(j + 2)) - 1 for j in range(p)]
        x0[p:p + q][diag] = np.log(0.3)
        x0[p + q:p + 2 * q][diag] = np.log(0.1)
        o = p + 2 * q
        x0[o:o + 4] = [np.log(0.05), np.log(0.02), 2.0, np.log(0.01)]
        res = minimize(self._nll, x0, method="L-BFGS-B", options={"maxiter": 400})
        self.x = res.x
        self.fit_info = {"success": bool(res.success), "nll": float(res.fun), "n_cells": int(len(Y)),
                         "n_groups": int(len(self.G)), "nit": int(res.nit)}
        return self

    def _nll(self, x):
        m, Cg, Cc = self.covs(x)
        mu = self.B.T @ m
        tot = 0.0
        for st, same, P, r, w in self.G:
            S = Cg[np.ix_(st, st)] + same * Cc[np.ix_(st, st)]
            V = P @ S @ P.T
            try:
                L = np.linalg.cholesky(V)
            except np.linalg.LinAlgError:
                return 1e12
            z = np.linalg.solve(L, r - P @ mu[st])
            tot += w * (np.log(np.diag(L)).sum() + 0.5 * float(z @ z))
        return tot

    def predict(self, target_support, sibling_supports) -> np.ndarray:
        """Level vector (14,) of the target cell given its own and its siblings' support pairs."""
        m, Cg, Cc = self.covs()
        mu = self.B.T @ m
        sups = [target_support] + [s for s in sibling_supports if s.k > 0]
        if all(s.k == 0 for s in sups):
            return mu
        Ds = [AD.pair_matrix(s.pairs) if s.k else np.zeros((0, N_LN)) for s in sups]
        nc = len(sups)
        # stacked latent vector: 14 metals per cell; V = Var(D y) over all supports
        blocks = [[Ds[i] @ (Cg + (Cc if i == j else 0.0)) @ Ds[j].T for j in range(nc)] for i in range(nc)]
        V = np.block(blocks)
        r = np.concatenate([np.asarray(s.values, float) - Ds[i] @ mu for i, s in enumerate(sups) if True])
        cross = np.hstack([(Cg + (Cc if i == 0 else 0.0)) @ Ds[i].T for i in range(nc)])
        w = np.linalg.solve(V, r)
        return mu + cross @ w

    def params(self) -> dict:
        m, Sg, Sc, tg, tc, rho, sig2 = self._unpack(self.x)
        return {"m": m.tolist(), "S_group": Sg.tolist(), "S_cell": Sc.tolist(), "tau2_group": float(tg),
                "tau2_cell": float(tc), "rho": float(rho), "sig2": float(sig2), **self.fit_info}


def attempt3_fit(bench, rec):
    from gen13sep.amplitude_bench import cell_weights
    tr = np.asarray(rec["outer"]["train_index"])
    f = bench.frame
    g = (f["extractant"].astype(str) + "|" + f["publication_id"].astype(str)).to_numpy()[tr]
    w = cell_weights(bench.groups[tr], bench.n_obs[tr])
    return {"HBG": HBGroupModel(np.asarray(bench.basis, float)).fit(bench.Y[tr], w, g)}


def _hbg_fit_torch(model: "HBGroupModel", x0: np.ndarray, maxiter: int = 400):
    """Same objective as ``HBGroupModel._nll`` with exact autograd gradients (float64)."""
    import torch
    torch.set_num_threads(4)
    B = torch.tensor(model.B, dtype=torch.float64)
    dZ = torch.tensor(model.dZ, dtype=torch.float64)
    p, q = model.p, model.p * (model.p + 1) // 2
    tril = torch.tril_indices(p, p)
    groups = [(torch.tensor(st), torch.tensor(same, dtype=torch.float64), torch.tensor(P), torch.tensor(r), w)
              for st, same, P, r, w in model.G]

    def chol(v):
        L = torch.zeros((p, p), dtype=torch.float64)
        L = L.index_put((tril[0], tril[1]), v)
        d = torch.diagonal(L)
        L = L - torch.diag(d) + torch.diag(torch.exp(d))
        return L @ L.T

    def f(xn):
        x = torch.tensor(xn, dtype=torch.float64, requires_grad=True)
        m = x[:p]
        Sg, Sc = chol(x[p:p + q]), chol(x[p + q:p + 2 * q])
        o = p + 2 * q
        tg, tc, rho, sig2 = torch.exp(x[o]), torch.exp(x[o + 1]), torch.sigmoid(x[o + 2]), torch.exp(x[o + 3])
        K = rho ** dZ
        Cg = B.T @ Sg @ B + tg * K
        Cc = B.T @ Sc @ B + tc * K + sig2 * torch.eye(N_LN, dtype=torch.float64)
        mu = B.T @ m
        tot = torch.zeros((), dtype=torch.float64)
        for st, same, P, r, w in groups:
            S = Cg[st][:, st] + same * Cc[st][:, st]
            V = P @ S @ P.T
            L = torch.linalg.cholesky(V)
            z = torch.linalg.solve_triangular(L, (r - P @ mu[st]).unsqueeze(1), upper=False)
            tot = tot + w * (torch.log(torch.diagonal(L)).sum() + 0.5 * (z * z).sum())
        tot.backward()
        return float(tot.detach()), x.grad.detach().numpy().astype(float)

    def safe(xn):
        try:
            return f(xn)
        except Exception:                       # non-PD step: reject it
            return 1e12, np.zeros_like(xn)
    return minimize(safe, x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})


def _hbg_fit(self, Y, w, groups, *, min_metals: int = 3):
    """Build the per-group matrices (as ``HBGroupModel.fit``) and optimise with torch gradients."""
    ok = np.isfinite(Y).sum(1) >= min_metals
    Y, w, groups = Y[ok], w[ok] / w[ok].mean(), np.asarray(groups)[ok]
    self.G = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        Ms = [np.flatnonzero(np.isfinite(Y[i])) for i in idx]
        st = np.concatenate(Ms)
        co = np.concatenate([np.full(len(M), j) for j, M in enumerate(Ms)])
        nS = len(st)
        P = np.zeros((nS - len(idx), nS))
        r = np.zeros(nS - len(idx))
        off = row = 0
        for i, M in zip(idx, Ms):
            n = len(M)
            P[row:row + n - 1, off] = -1.0
            P[row:row + n - 1, off + 1:off + n] = np.eye(n - 1)
            r[row:row + n - 1] = Y[i, M][1:] - Y[i, M][0]
            off += n
            row += n - 1
        self.G.append((st, co[:, None] == co[None, :], P, r, float(w[idx].mean())))
    p, q = self.p, self.p * (self.p + 1) // 2
    x0 = np.zeros(p + 2 * q + 4)
    diag = [sum(range(j + 2)) - 1 for j in range(p)]
    x0[p:p + q][diag] = np.log(0.3)
    x0[p + q:p + 2 * q][diag] = np.log(0.1)
    o = p + 2 * q
    x0[o:o + 4] = [np.log(0.05), np.log(0.02), 2.0, np.log(0.01)]
    res = _hbg_fit_torch(self, x0)
    self.x = res.x
    self.fit_info = {"success": bool(res.success), "nll": float(res.fun), "n_cells": int(len(Y)),
                     "n_groups": int(len(self.G)), "nit": int(res.nit)}
    return self


HBGroupModel.fit = _hbg_fit


def ridge_pool(basis, supports, prior, lam, s2):
    """RIDGE2 with one shape theta shared by a condition series: every cell's star differences
    (offsets cancel within a cell), block-diagonal shared-reading noise, gen20's prior and lambda."""
    sups = [s for s in supports if s.k > 0]
    prior = np.asarray(prior, float)
    if not sups:
        return prior @ basis
    A = np.vstack([AD.pair_matrix(s.pairs) @ basis.T for s in sups])
    y = np.concatenate([np.asarray(s.values, float) for s in sups])
    Rinv = np.zeros((len(y), len(y)))
    o = 0
    for s in sups:
        D = AD.pair_matrix(s.pairs)
        n = len(s.pairs)
        Rinv[o:o + n, o:o + n] = np.linalg.pinv(s2 * (D @ D.T))
        o += n
    H = A.T @ Rinv @ A + lam * np.eye(basis.shape[0])
    coef = np.linalg.solve(H, A.T @ Rinv @ y + lam * prior)
    return coef @ basis


def attempt3_methods(bench):
    from . import evaluate as E
    basis = bench.basis

    def hbg_pool(block, cell, obs, k):
        sib = [o.support for _, o in E.sibling_obs(block, cell, k)]
        return block["fitted"]["HBG"].predict(obs.support, sib)

    def hbg_own(block, cell, obs, k):
        return block["fitted"]["HBG"].predict(obs.support, [])

    def ridge2_pool(block, cell, obs, k):
        sib = [o.support for _, o in E.sibling_obs(block, cell, k)]
        return ridge_pool(basis, [obs.support] + sib, block["prior"], block["ridge_lambda"], block["s2"])

    def hb2gp(block, cell, obs, k):
        return block["fitted"]["HB2+GP"].predict(obs.support)

    return {"HBG@POOL": hbg_pool, "HBG@OWN": hbg_own, "RIDGE2@POOL": ridge2_pool, "HB2+GP": hb2gp}


def attempt3_fit_all(bench, rec):
    from gen13sep.amplitude_bench import cell_weights
    tr = np.asarray(rec["outer"]["train_index"])
    w = cell_weights(bench.groups[tr], bench.n_obs[tr])
    out = attempt3_fit(bench, rec)
    out["HB2+GP"] = HBModel(hb_basis(bench, "B2"), use_gp=True).fit(bench.Y[tr], w)
    return out
