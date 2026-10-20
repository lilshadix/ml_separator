"""Inner cross-fitted predictions inside one outer training fold (directions B and K).

For every inner chemotype block j of the outer training fold (gen21 cache: design mask re-applied,
nested covariance), every HBK kernel is fitted by maximum likelihood on block j's inner-TRAINING cells
only, and the hidden strict-adjacent queries of block j's inner-VALIDATION cells are predicted from
their own nested DOPT_Q reads, for a small grid of reading-noise multipliers c (sigma^2 -> c sigma^2
at prediction time).  PWL(R) and PWL(Z) are added on the same rows.  Nothing here reads an outer
held-out cell; each inner-validation prediction uses no label of its own block.

Output: a DataFrame, one row per (inner block, cell, k, query), columns y, weight (chemotype-balanced
weight of the cell within the outer training fold), and one column per candidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import interp
from . import paths  # noqa: F401
from g21 import evaluate as E  # noqa: E402
from g21 import methods as M  # noqa: E402

KERNELS = M.KERNELS
NOISE_MULT = (0.5, 1.0, 2.0, 4.0)
KS = (3,)


def hbk_predict(mdl, support, c: float = 1.0) -> np.ndarray:
    mu = mdl.B.T @ mdl.m
    if support.k == 0:
        return mu
    C = mdl.cov_full(mdl.S, mdl.tau2, mdl.rho, c * mdl.sig2)
    from gen20k import adapters as AD
    D = AD.pair_matrix(support.pairs)
    r = np.asarray(support.values, float) - D @ mu
    return mu + C @ D.T @ np.linalg.solve(D @ C @ D.T, r)


def inner_table(bench, rec, *, ks=KS, kernels=KERNELS, noise_mult=NOISE_MULT) -> tuple[pd.DataFrame, dict]:
    from gen13sep.amplitude_bench import cell_weights
    tr_out = np.asarray(rec["outer"]["train_index"])
    wmap = dict(zip(tr_out.tolist(), cell_weights(bench.groups[tr_out], bench.n_obs[tr_out]).tolist()))
    rows, lik = [], {}
    for blk in rec["inner"]:
        itr = np.asarray(blk["train_index"])
        w = cell_weights(bench.groups[itr], bench.n_obs[itr])
        fits = {kname: M.HBKModel(M.hb_basis(bench, "B2"), kname).fit(bench.Y[itr], w) for kname in kernels}
        ml = min(kernels, key=lambda kk: fits[kk].fit_info["nll"])
        lik[blk["inner_fold"]] = {kk: fits[kk].fit_info["nll"] for kk in kernels} | {"ml_kernel": ml}
        for cell in blk["cells"]:
            if cell["k_max"] < max(ks):
                continue
            for k in ks:
                o = E.observe(bench.Y[cell["ci"]], cell, cell["seqs"]["DOPT_Q"], k)
                preds = {"PWL_R": interp.pwl_levels(o.y_masked, cell["metals"], "R"),
                         "PWL_Z": interp.pwl_levels(o.y_masked, cell["metals"], "Z")}
                for kk in kernels:
                    for c in noise_mult:
                        preds[f"HBK[{kk}]x{c}"] = hbk_predict(fits[kk], o.support, c)
                preds["HBK@ML_inner"] = preds[f"HBK[{ml}]x1.0"]
                for a, b in cell["query"]:
                    r = {"inner_fold": blk["inner_fold"], "ci": cell["ci"], "cell_id": cell["cell_id"], "chemotype": cell["chemotype"],
                         "k": k, "A": a, "B": b, "a_read": a in o.read, "b_read": b in o.read,
                         "y": float(bench.Y[cell["ci"]][a] - bench.Y[cell["ci"]][b]), "w": wmap.get(cell["ci"], np.nan)}
                    for n, v in preds.items():
                        r[n] = float(v[a] - v[b])
                    rows.append(r)
    return pd.DataFrame(rows), lik


def task_choice(inner: pd.DataFrame, candidates, *, k: int = 3) -> tuple[str, dict]:
    """Candidate with the lowest weighted squared error of hidden adjacent log SF on the inner-validation
    rows (each cell's weight split over its queries, so cells, not pairs, carry the weight)."""
    t = inner[inner.k == k].copy()
    nq = t.groupby(["inner_fold", "ci"]).y.transform("size")
    t["wq"] = t["w"] / nq
    sc = {c: float((t["wq"] * (t[c] - t.y) ** 2).sum() / t["wq"].sum()) for c in candidates}
    return min(sc, key=sc.get), sc
