"""L5: gen15 §7's component-mean mixture (``MIX6meanPC``) on two routes.

``gen15.mixture.evaluate`` (frozen, not modified) scores the mixture on the **training route**:
prototypes, prior weights *and the pooled covariance* are fitted on the training fold's own
in-sample residuals, and the D-optimal support is chosen under that training-fold covariance.
Under BP the training fold already contains no cell from any held-out publication (the fold plan
masks them), so the training route has *no* place for a publication leak -- ``audit_training_route``
counts the shared publications per fold and returns 0 for BP.  Under B / BR / BQ / A the training
fold does contain the test publication by design; masking it there would change the design's
meaning, so the training route is left as gen15 ran it and is used here only as the pipeline
reproduction (``evaluate_training``, a verbatim port, must give MIX6meanPC@k3 = 0.1852 and
POOLED@k3 = 0.1922 under BP with how='dopt').

The **deployed route** (``evaluate_deployed``) is the registered comparison: the pooled covariance
is the leave-chemotype-out, publication-masked out-of-fold residual covariance of the deployed
measured mode (identical to ``POOLED`` in ``l5_fewshot``), the support is D-optimal under it, and
every component of the mixture is given that same covariance.  It is ``l5_fewshot.evaluate`` with
``mixtures={'MIX6meanPC': 6}`` and the default estimator, so ``POOLED@doptk<k>`` in its table is
gen15 §5's ``G14@doptk<k>`` on gen15's common scoring set (0.1700 at k = 3 under BP), and
``MIX6meanPC@doptk<k>`` is scored on exactly those pairs.
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from gen16 import bootstrap  # noqa: F401
from gen13sep.amplitude_bench import LEAN_BLOCKS, cell_weights  # noqa: E402
from gen13sep.metals import LANTHANIDES  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import feature_sets  # noqa: E402
from gen15.fewshot import NOISE_VAR, blup, naive_line, pick_support  # noqa: E402
from gen15.mixture import Components, N_LN, Z_VEC, fit_components, posterior, separation_nats  # noqa: E402
from gen15.valuebench import Ctx, MIN_METALS  # noqa: E402
from gen15 import arms as A  # noqa: E402

from gen16 import l5_fewshot as L5  # noqa: E402

HOWS = ("widest", "dopt", "random")


def audit_training_route(bench, design: str) -> pd.DataFrame:
    """Per fold: training cells whose publication appears among the held-out cells (BP: 0)."""
    pub = bench.frame.publication_id.astype(str).to_numpy()
    rows = []
    for f in all_folds(bench.frame, design=design):
        held = set(pub[f.test_index])
        n_shared = int(sum(pub[i] in held for i in f.train_index))
        rows.append({"design": design, "split_seed": f.seed, "fold": f.fold,
                     "n_train": len(f.train_index), "n_test": len(f.test_index),
                     "n_train_cells_sharing_test_publication": n_shared})
    return pd.DataFrame(rows)


def evaluate_training(bench, prior_arm: Callable[[Ctx], np.ndarray], design: str, *,
                      ks: Sequence[int] = (0, 1, 2, 3), how: str = "dopt",
                      kk: Sequence[int] = (6,), shift: bool = True,
                      noise_var: float = NOISE_VAR, min_metals: int = 3,
                      verbose: bool = True) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Verbatim port of ``gen15.mixture.evaluate`` (the training route), default ``kk=(6,)``.

    Only ``MIX{K}meanPC`` and ``POOLED`` / ``NAIVE`` modes are emitted (the per-component-
    covariance arms were measured dead in gen15 §7 and are not re-run); the prototypes, the pooled
    covariance, the support order and the exclusion are byte-identical to gen15's.
    """
    t0 = time.time()
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
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
        pooled = comps[1] if 1 in comps else fit_components(ctx, 1)
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
                preds[f"POOLED@k{k}"] = blup(base, pooled.cov[0], sup, noise_var=noise_var)
                if k > 0:
                    preds[f"NAIVE@k{k}"] = naive_line(bench.basis, sup)
                for K, comp in comps.items():
                    sh = base - comp.prior @ comp.mu if shift else np.zeros(N_LN)
                    flat_cov = Components(mu=comp.mu, cov=[pooled.cov[0]] * len(comp.prior),
                                          logdet=np.zeros(len(comp.prior)), prior=comp.prior,
                                          label=comp.label)
                    gf, musf = posterior(flat_cov, sup, sh, noise_var=noise_var)
                    preds[f"MIX{K}meanPC@k{k}"] = gf @ musf
                    if k == 1:
                        g, _ = posterior(comp, sup, sh, noise_var=noise_var)
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
        print(f"   [{design}] mixture (training route) {len(rows)} pairs, {len(modes)} modes, "
              f"{time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), modes, pd.DataFrame(checks)


def evaluate_deployed(bench, design: str, *, ks: Sequence[int] = (0, 1, 2, 3),
                      hows: Sequence[str] = HOWS, K: int = 6, mask_publication: bool = True,
                      verbose: bool = True):
    """The deployed route: fewshot semantics, publication-masked OOF pooled covariance, MIX6meanPC.

    Returns the ``FewShotResult`` with ``G14@...`` renamed to ``POOLED@...`` so the table reads as
    the registered contrast (``POOLED`` is the G14 prior + BLUP under the deployed covariance).
    """
    r = L5.evaluate(bench, {"G14": A.g14}, design, ks=ks, hows=hows, cov_kind="empirical",
                    mask_publication=mask_publication, mixtures={f"MIX{K}meanPC": K},
                    verbose=verbose)
    ren = {m: m.replace("G14@", "POOLED@") for m in r.modes if m.startswith("G14@")}
    r.pairs.rename(columns=ren, inplace=True)
    r.modes = [ren.get(m, m) for m in r.modes]
    return r
