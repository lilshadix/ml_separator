"""Gen15 §7: let the measurement pick the curve shape.

Two gen15 results point at each other and have not been connected.

* §2 measured that a small alphabet of canonical lanthanide curves is real and valuable: assigning a
  held-out cell to the nearest of four prototypes learned from the training fold scores 0.4438
  against 0.5001, +0.056, passing P1 — and that **no model can pick the prototype from structure**
  (the multiclass logistic on donor topology scores 0.610, worse than the constant).
* §5 measured that one measured separation factor is worth +0.207, and that after it the ligand
  prior is worth +0.002: the measurement, not the chemistry, is what identifies the cell.

So use the measurement to pick the prototype.  The prior over a held-out cell's curve becomes a
*mixture* of Gaussians, one per prototype, each with its own residual covariance; a measured log SF
updates the mixture weights by their likelihoods and the prediction is the posterior mixture of the
per-component BLUPs.  The pooled-covariance BLUP of §5 is the K = 1 special case.

The components are shifted so that the prior mixture mean equals the deployed gen14 curve
(``shift=True``).  Without that the k = 0 rung would be the prototype prediction, which §2 already
measured as bad, and the comparison at k >= 1 would be confounded by a worse starting point.  With
it, k = 0 is *identical* to the gen15 arm and every difference is attributable to the measurement
selecting a shape.

The cheap pre-check comes first and can kill the idea in one run: if a single measurement cannot
separate the components' likelihoods, the posterior stays at the prior and nothing can follow.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights
from gen13sep.metals import ATOMIC_NUMBER, LANTHANIDES
from gen13sep.splits import all_folds
from gen14.dirbench import feature_sets

from .fewshot import (NOISE_VAR, blup, centred_residual, naive_line, pick_support,
                      residual_covariance, smooth_covariance)
from .shape import _prototypes
from .valuebench import Ctx, MIN_METALS

N_LN = len(LANTHANIDES)
Z_VEC = np.array([ATOMIC_NUMBER[m] for m in LANTHANIDES])


# --------------------------------------------------------------------------------------
@dataclass
class Components:
    """A learned curve alphabet with a residual covariance and a prior weight per letter."""
    mu: np.ndarray            # K x 14 prototype curves
    cov: list                 # K covariance matrices, 14 x 14
    logdet: np.ndarray        # K
    prior: np.ndarray         # K, sums to 1
    label: np.ndarray         # K, component index of each training cell (diagnostic)


def fit_components(ctx: Ctx, k: int, *, shrink_to_pooled: float = 0.5,
                   min_members: int = 12) -> Components:
    """Prototypes from the training fold, each with its own shrunk residual covariance.

    A component holding 30-120 cells cannot support 105 free covariance parameters, so every
    ``Sigma_c`` is shrunk halfway toward the pooled training covariance — the same device
    ``residual_covariance`` already uses to shrink toward its own diagonal, one level up.
    """
    tr, w = ctx.rich_train(), ctx.rich_weights()
    a, b = ctx.amp[tr], ctx.cur[tr]
    P = _prototypes(a, b, w, k, ctx.model_seed) if k > 1 else np.array(
        [[float(np.average(a, weights=w)), float(np.average(b, weights=w))]])
    scale = np.array([np.abs(a).mean() + 1e-9, np.abs(b).mean() + 1e-9])
    lab = np.abs(np.c_[a, b][:, None, :] / scale - P[None] / scale).sum(2).argmin(axis=1)
    mu = P @ ctx.bench.basis

    resid = np.array([centred_residual(ctx.bench.Y[ci], mu[lab[j]])
                      for j, ci in enumerate(tr)])
    pooled = residual_covariance(resid)
    covs, prior = [], np.zeros(len(P))
    for c in range(len(P)):
        m = lab == c
        prior[c] = float(w[m].sum())
        if m.sum() >= min_members:
            S = residual_covariance(resid[m])
            covs.append((1.0 - shrink_to_pooled) * S + shrink_to_pooled * pooled)
        else:
            covs.append(pooled)
    prior = prior / max(prior.sum(), 1e-12)
    keep = prior > 1e-6
    mu, covs, prior = mu[keep], [c for c, kk in zip(covs, keep) if kk], prior[keep]
    prior = prior / prior.sum()
    logdet = np.array([np.linalg.slogdet(c)[1] for c in covs])
    return Components(mu=mu, cov=covs, logdet=logdet, prior=prior, label=lab)


def _design(supports: Sequence[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    D = np.zeros((len(supports), N_LN))
    y = np.empty(len(supports))
    for i, (a, b, v) in enumerate(supports):
        D[i, a], D[i, b] = 1.0, -1.0
        y[i] = v
    return D, y


def posterior(comp: Components, supports: Sequence[tuple[int, int, float]], shift: np.ndarray,
              *, noise_var: float = NOISE_VAR) -> tuple[np.ndarray, np.ndarray]:
    """Component posterior and per-component BLUP means given the measurements."""
    K = len(comp.prior)
    mus = comp.mu + shift[None, :]
    if not supports:
        return comp.prior.copy(), mus
    D, y = _design(supports)
    ll = np.empty(K)
    out = np.empty_like(mus)
    for c in range(K):
        S = D @ comp.cov[c] @ D.T + noise_var * np.eye(len(supports))
        r = y - D @ mus[c]
        try:
            sol = np.linalg.solve(S, r)
            sign, ld = np.linalg.slogdet(S)
        except np.linalg.LinAlgError:
            ll[c] = -np.inf
            out[c] = mus[c]
            continue
        ll[c] = -0.5 * (float(r @ sol) + ld)
        out[c] = mus[c] + comp.cov[c] @ D.T @ sol
    ll = ll - ll.max()
    g = comp.prior * np.exp(ll)
    s = g.sum()
    return (g / s if s > 1e-300 else comp.prior.copy()), out


def separation_nats(comp: Components, supports: Sequence[tuple[int, int, float]],
                    shift: np.ndarray, *, noise_var: float = NOISE_VAR) -> float:
    """Pre-check: how many nats separate the best and worst component's likelihood.

    Under a nat or so, one measurement cannot tell the prototypes apart and the posterior stays at
    the prior; the whole idea is then dead before any arm is scored.
    """
    if not supports or len(comp.prior) < 2:
        return 0.0
    D, y = _design(supports)
    mus = comp.mu + shift[None, :]
    ll = []
    for c in range(len(comp.prior)):
        S = D @ comp.cov[c] @ D.T + noise_var * np.eye(len(supports))
        r = y - D @ mus[c]
        try:
            ll.append(-0.5 * (float(r @ np.linalg.solve(S, r)) + np.linalg.slogdet(S)[1]))
        except np.linalg.LinAlgError:
            pass
    return float(max(ll) - min(ll)) if len(ll) > 1 else 0.0


# --------------------------------------------------------------------------------------
def evaluate(bench, prior_arm: Callable[[Ctx], np.ndarray], design: str, *,
             ks: Sequence[int] = (0, 1, 2, 3), how: str = "dopt",
             kk: Sequence[int] = (2, 4, 6), shift: bool = True,
             noise_var: float = NOISE_VAR, min_metals: int = 3,
             verbose: bool = True) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Score the mixture against the pooled BLUP on a common pair set, plus the pre-check."""
    t0 = time.time()
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    pub = bench.frame.publication_id.astype(str).to_numpy()
    kmax = max(ks)
    rows, checks, modes = [], [], []

    for f in all_folds(bench.frame, design=design):
        ctx = Ctx(bench=bench, design=design, seed=f.seed, fold=f.fold, train=f.train_index,
                  test=f.test_index, w=cell_weights(bench.groups[f.train_index],
                                                    bench.n_obs[f.train_index]),
                  model_seed=f.model_seed, fs=fs, X=X, rich=rich)
        coef = np.asarray(prior_arm(ctx), dtype=float).reshape(len(f.test_index), 2)
        prior_curves = coef @ bench.basis
        comps = {k: fit_components(ctx, k) for k in kk}
        pooled = comps[1] if 1 in comps else None
        if pooled is None:
            pooled = fit_components(ctx, 1)

        for j, ci in enumerate(f.test_index):
            y = bench.Y[ci]
            obs = np.flatnonzero(~np.isnan(y))
            if len(obs) < min_metals + 1:
                continue
            pairs = [(int(a), int(b)) for i, a in enumerate(obs) for b in obs[i + 1:]]
            if len(pairs) <= kmax:
                continue
            base = prior_curves[j]
            order = pick_support(pairs, kmax, how, pooled.cov[0], noise_var=noise_var)
            held = [p for p in pairs if p not in set(order)]
            if not held:
                continue
            preds: dict[str, np.ndarray] = {}
            for k in ks:
                sup = [(a, b, float(y[a] - y[b])) for a, b in order[:k]]
                # pooled reference: the gen15 arm, K = 1 with a training-fold covariance
                preds[f"POOLED@k{k}"] = blup(base, pooled.cov[0], sup, noise_var=noise_var)
                if k > 0:
                    preds[f"NAIVE@k{k}"] = naive_line(bench.basis, sup)
                for K, comp in comps.items():
                    sh = base - comp.prior @ comp.mu if shift else np.zeros(N_LN)
                    g, mus = posterior(comp, sup, sh, noise_var=noise_var)
                    preds[f"MIX{K}mean@k{k}"] = g @ mus
                    preds[f"MIX{K}hard@k{k}"] = mus[int(np.argmax(g))]
                    # attribution arm: the same component means, but every component given the
                    # POOLED covariance.  If this also loses, the prototype means are what
                    # over-commit; if it wins, the per-component covariances were the problem.
                    flat_cov = Components(mu=comp.mu, cov=[pooled.cov[0]] * len(comp.prior),
                                          logdet=np.zeros(len(comp.prior)), prior=comp.prior,
                                          label=comp.label)
                    gf, musf = posterior(flat_cov, sup, sh, noise_var=noise_var)
                    preds[f"MIX{K}meanPC@k{k}"] = gf @ musf
                    if k == 1:
                        checks.append({"design": design, "K": K, "cell": int(ci),
                                       "chemotype": bench.groups[ci],
                                       "nats": separation_nats(comp, sup, sh, noise_var=noise_var),
                                       "post_max": float(g.max()),
                                       "collapsed": bool(g.max() > 0.9)})
            if not modes:
                modes = list(preds)
            for a, b in held:
                rec = {"split_seed": f.seed, "fold": f.fold,
                       "cell_id": bench.frame.cell_id.iat[ci],
                       "extractant": bench.frame.extractant.iat[ci],
                       "chemotype": bench.frame.chemotype.iat[ci],
                       "n_metals": int(bench.frame.n_metals.iat[ci]),
                       "A": LANTHANIDES[a], "B": LANTHANIDES[b],
                       "dZ": int(Z_VEC[b] - Z_VEC[a]), "y": float(y[a] - y[b])}
                for m, cv in preds.items():
                    rec[m] = float(cv[a] - cv[b])
                rows.append(rec)
    if verbose:
        print(f"   [{design}] mixture {len(rows)} pairs, {len(modes)} modes, "
              f"{time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), modes, pd.DataFrame(checks)
