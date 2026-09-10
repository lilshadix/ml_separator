"""L5: ``gen15.fewshot.evaluate`` with one hook -- the residual-covariance estimator.

This is a copy of the frozen measured-mode route (``gen15_curve/gen15/fewshot.py``, not modified)
with exactly these additions:

1. ``cov_estimators`` -- a dict ``name -> estimator(rows, groups) -> 14 x 14``.  ``cov_for`` calls
   every estimator on the *same* leave-chemotype-out, publication-masked residual rows and returns
   a dict of covariances.  With ``cov_estimators=None`` the single default estimator is
   ``residual_covariance`` and every name, rng draw, support choice and exclusion is byte-identical
   to gen15 (``scripts/l5_verify.py`` checks ``G14@doptk3`` = 0.1700 and ``G14@doptk1`` = 0.2313
   under BP before anything else runs).

2. In multi-estimator mode every estimator's D-optimal supports are chosen with *its own*
   covariance, and the **union** of every (strategy, estimator) support set is excluded from
   scoring for every mode, so all estimators are scored on byte-identical pairs -- the same device
   gen15 §5 used across support strategies.  Modes are named ``<arm>_<EST>@<how>k<k>``,
   ``NAIVE_LINE_<EST>@doptk<k>`` (the line through *that* estimator's D-optimal pairs;
   ``NAIVE_LINE@<how>k<k>`` for the estimator-independent strategies) and ``sd_<EST>@<how>k<k>``.

3. ``mixtures`` -- ``{name: K}`` adds gen15 §7's component-mean arm under the deployed route: the
   K prototypes and prior weights are ``gen15.mixture.fit_components`` on the training fold
   (frozen code), every component is given the *first* estimator's covariance (the leave-chemotype-
   out, publication-masked POOLED one), the components are shifted so the prior mixture mean is the
   reference arm's curve, and the prediction is ``gen15.mixture.posterior``'s weighted BLUP mean.
   Its predictive sd is the mixture's: posterior pair variance + noise + between-component spread.

Bookkeeping change only: the pair table is accumulated as arrays per cell instead of one dict per
row (140+ modes x tens of thousands of pairs would not fit the 8 GB machine as dicts).  The numbers
are the same and the verification script proves it on the default estimator.
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from gen16 import bootstrap  # noqa: F401  (sys.path)
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import feature_sets  # noqa: E402
from gen15.fewshot import (N_LN, NOISE_VAR, Z_VEC, FewShotResult, blup, blup_posterior,  # noqa: E402
                           centred_residual, line_plus_curvature, naive_line, pick_support,
                           residual_covariance, smooth_covariance)
from gen15.mixture import Components, fit_components, posterior as mix_posterior  # noqa: E402
from gen15.valuebench import Ctx, MIN_METALS  # noqa: E402

from gen16.l5_cov import Estimator  # noqa: E402

DEFAULT_ESTIMATOR: dict[str, Estimator] = {"POOLED": lambda rows, groups: residual_covariance(rows)}


def evaluate(bench, arms: dict[str, Callable[[Ctx], np.ndarray]], design: str, *,
             ks: Sequence[int] = (0, 1, 2, 3), how: str = "widest",
             hows: Sequence[str] | None = None,
             cov_kind: str = "empirical", noise_var: float = NOISE_VAR,
             mask_publication: bool = False, per_cell_noise: bool = False,
             min_metals: int = 3,
             cov_estimators: dict[str, Estimator] | None = None,
             mixtures: dict[str, int] | None = None,
             verbose: bool = True) -> FewShotResult:
    """Score every arm at every measurement budget on a FIXED pair set (gen15 semantics).

    See the module docstring for the two additions (``cov_estimators``, ``mixtures``).  Discovery
    seeds are the frozen defaults of ``all_folds``; there is deliberately no ``seeds`` argument.
    """
    t0 = time.time()
    multi = cov_estimators is not None
    ests: dict[str, Estimator] = dict(cov_estimators) if multi else dict(DEFAULT_ESTIMATOR)
    est_names = list(ests)
    est0 = est_names[0]
    mixtures = dict(mixtures) if mixtures else {}
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    kmax = max(ks)
    folds = all_folds(bench.frame, design=design)

    # ---- pass 1: predicted curves for every held-out cell, per arm ----
    curves: dict[str, dict] = {a: {} for a in arms}
    bstats: dict[tuple[int, int], tuple[float, float]] = {}
    comps: dict[tuple[int, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for f in folds:
        ctx = Ctx(bench=bench, design=design, seed=f.seed, fold=f.fold, train=f.train_index,
                  test=f.test_index, w=cell_weights(bench.groups[f.train_index],
                                                    bench.n_obs[f.train_index]),
                  model_seed=f.model_seed, fs=fs, X=X, rich=rich)
        rtr, rw = ctx.rich_train(), ctx.rich_weights()
        a_tr, b_tr = ctx.amp[rtr], ctx.cur[rtr]
        hv, lt = a_tr < 0, a_tr > 0
        bstats[(f.seed, f.fold)] = (
            float(np.average(b_tr[hv], weights=rw[hv])) if hv.sum() >= 10 else ctx.train_mean_curvature(),
            float(np.average(b_tr[lt], weights=rw[lt])) if lt.sum() >= 10 else ctx.train_mean_curvature())
        for name, arm in arms.items():
            coef = np.asarray(arm(ctx), dtype=float).reshape(len(f.test_index), bench.basis.shape[0])
            cv = coef @ bench.basis
            for j, ci in enumerate(f.test_index):
                curves[name].setdefault((f.seed, f.fold), {})[ci] = cv[j]
        if mixtures:
            comps[(f.seed, f.fold)] = {}
            for mname, K in mixtures.items():
                c = fit_components(ctx, K)
                comps[(f.seed, f.fold)][mname] = (c.mu, c.prior)

    # ---- pass 2: leave-chemotype-out residual covariance, per (seed, reference arm) ----
    ref = next(iter(arms))
    pub = bench.frame.publication_id.astype(str).to_numpy()
    resid_by_seed: dict[int, dict[int, np.ndarray]] = {}
    for (seed, fold), d in curves[ref].items():
        for ci, cv in d.items():
            resid_by_seed.setdefault(seed, {})[ci] = centred_residual(bench.Y[ci], cv)
    fallback_cov = smooth_covariance(scale=0.25, length=1.0, nugget=0.05, basis_row=bench.basis[0])

    def cov_for(seed: int, chemotype: str, publication: str) -> dict[str, np.ndarray]:
        if cov_kind == "smooth":
            return {e: fallback_cov for e in est_names}
        keep = [ci for ci in resid_by_seed.get(seed, {})
                if bench.groups[ci] != chemotype
                and not (mask_publication and pub[ci] == publication)]
        if len(keep) < 10:
            return {e: fallback_cov for e in est_names}
        rows = np.array([resid_by_seed[seed][ci] for ci in keep])
        groups = np.array([str(bench.groups[ci]) for ci in keep])
        out = {}
        for e, fn in ests.items():
            emp = fn(rows, groups)                      # <-- the hook
            if cov_kind == "blend":
                s = float(np.trace(emp) / np.trace(fallback_cov))
                emp = 0.5 * emp + 0.5 * s * fallback_cov
            out[e] = emp
        return out

    cov_cache: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    repsd = bench.frame[[f"repsd__{m}" for m in LANTHANIDES]].to_numpy(dtype=float)
    med_repsd = float(np.nanmedian(repsd)) if np.isfinite(repsd).any() else 0.212

    def noise_for(ci: int) -> float:
        if not per_cell_noise:
            return noise_var
        row = repsd[ci]
        s = float(np.nanmedian(row)) if np.isfinite(row).any() else med_repsd
        return float(np.clip(2.0 * s ** 2, 0.02, 0.5))

    def nm(arm: str, est: str) -> str:
        return f"{arm}_{est}" if multi else arm

    # ---- pass 3: correct and score ----
    modes: list[str] = []
    sd_names: list[str] = []
    blocks_pred: list[np.ndarray] = []
    blocks_sd: list[np.ndarray] = []
    meta_rows: list[tuple] = []
    for (seed, fold), d in curves[ref].items():
        for ci in d:
            y = bench.Y[ci]
            obs = np.flatnonzero(~np.isnan(y))
            if len(obs) < min_metals + 1:
                continue
            pairs = [(int(a), int(b)) for i, a in enumerate(obs) for b in obs[i + 1:]]
            if len(pairs) <= kmax:
                continue
            key = (seed, str(bench.groups[ci]), pub[ci] if mask_publication else "")
            if key not in cov_cache:
                cov_cache[key] = cov_for(seed, str(bench.groups[ci]), pub[ci])
            covs = cov_cache[key]
            nv = noise_for(ci)
            rng = np.random.default_rng(abs(hash((int(ci), int(seed)))) % (2 ** 32))
            strategies = list(hows) if hows else [how]
            # support orders: D-optimal depends on the covariance, the others do not
            orders: dict[tuple[str, str | None], list] = {}
            for h in strategies:
                if h == "dopt":
                    for e in est_names:
                        orders[(h, e)] = pick_support(pairs, kmax, h, covs[e], rng=rng, noise_var=nv)
                else:
                    orders[(h, None)] = pick_support(pairs, kmax, h, covs[est0], rng=rng, noise_var=nv)
            excluded = {p for order in orders.values() for p in order}
            held = [p for p in pairs if p not in excluded]
            if not held:
                continue
            bh, bl = bstats[(seed, fold)]
            preds: dict[str, np.ndarray] = {}
            sds: dict[str, np.ndarray] = {}
            A_idx = np.array([a for a, _ in held])
            B_idx = np.array([b for _, b in held])
            for h in strategies:
                tag = "" if not hows else h
                for e in est_names:
                    cov = covs[e]
                    order = orders[(h, e)] if h == "dopt" else orders[(h, None)]
                    for k in ks:
                        sup = [(a, b, float(y[a] - y[b])) for a, b in order[:k]]
                        post = blup_posterior(cov, sup, noise_var=nv)
                        base_var = post[A_idx, A_idx] + post[B_idx, B_idx] - 2.0 * post[A_idx, B_idx]
                        sds[f"sd{'_' + e if multi else ''}@{tag}k{k}"] = \
                            np.sqrt(np.maximum(base_var + nv, 1e-12))
                        for name in arms:
                            preds[f"{nm(name, e)}@{tag}k{k}"] = blup(
                                curves[name][(seed, fold)][ci], cov, sup, noise_var=nv)
                        if k > 0:
                            if h == "dopt" and multi:
                                preds[f"NAIVE_LINE_{e}@{tag}k{k}"] = naive_line(bench.basis, sup)
                            elif e == est0:
                                preds[f"NAIVE_LINE@{tag}k{k}"] = naive_line(bench.basis, sup)
                            if not hows and e == est0:
                                preds[f"LINE_PLUS_CURV@k{k}"] = line_plus_curvature(
                                    bench.basis, sup, bh, bl)
                        if e == est0 and mixtures:
                            base = curves[ref][(seed, fold)][ci]
                            for mname, K in mixtures.items():
                                mu, prior = comps[(seed, fold)][mname]
                                flat = Components(mu=mu, cov=[cov] * len(prior),
                                                  logdet=np.zeros(len(prior)), prior=prior,
                                                  label=np.zeros(0, dtype=int))
                                sh = base - prior @ mu
                                g, mus = mix_posterior(flat, sup, sh, noise_var=nv)
                                mean = g @ mus
                                preds[f"{nm(mname, e)}@{tag}k{k}"] = mean
                                dm = (mus[:, A_idx] - mus[:, B_idx]) - (mean[A_idx] - mean[B_idx])[None, :]
                                between = (g[:, None] * dm ** 2).sum(axis=0)
                                sds[f"sd_{mname}{'_' + e if multi else ''}@{tag}k{k}"] = \
                                    np.sqrt(np.maximum(base_var + nv + between, 1e-12))
            if not modes:
                modes = list(preds)
                sd_names = list(sds)
            C = np.array([preds[m] for m in modes])                      # modes x 14
            blocks_pred.append((C[:, A_idx] - C[:, B_idx]).T)              # held x modes
            blocks_sd.append(np.array([sds[s] for s in sd_names]).T)       # held x sds
            cell_id = bench.frame.cell_id.iat[ci]
            ext = bench.frame.extractant.iat[ci]
            chem = bench.frame.chemotype.iat[ci]
            nmet = int(bench.frame.n_metals.iat[ci])
            for a, b in held:
                meta_rows.append((seed, fold, cell_id, ext, chem, nmet, LANTHANIDES[a],
                                  LANTHANIDES[b], int(Z_VEC[b] - Z_VEC[a]), float(y[a] - y[b])))
    if not meta_rows:
        return FewShotResult(pairs=pd.DataFrame(), modes=[], seconds=time.time() - t0)
    meta = pd.DataFrame(meta_rows, columns=["split_seed", "fold", "cell_id", "extractant",
                                            "chemotype", "n_metals", "A", "B", "dZ", "y"])
    P = pd.DataFrame(np.vstack(blocks_pred), columns=modes)
    S = pd.DataFrame(np.vstack(blocks_sd), columns=sd_names)
    table = pd.concat([meta, P, S], axis=1)
    if verbose:
        print(f"   [{design}] L5 few-shot {len(table)} scored pairs, {len(modes)} modes, "
              f"{len(est_names)} estimators, {time.time() - t0:.0f}s", flush=True)
    return FewShotResult(pairs=table, modes=modes, seconds=time.time() - t0)
