"""Label-free support policies added in gen21.

``SERIES_STAGGER`` (attempt 3): the held-out cells of one condition series (same extractant, same
publication) are ordered by a hash of their cell id; cell j reads, greedily, the legal metal read
by the fewest earlier cells of its series (their full sequences), then the one farthest in Z from
its own reads so far, then the lighter one.  For the first cell of a series (and for any singleton)
this is exactly gen20's UNIFORM order (lightest legal, heaviest legal, farthest-point).  Legality is
gen20's: the read set stays an independent set of the cell's adjacency path and k_max stays
reachable, so no query pair of a cell is ever read at both ends *in that cell*.  It reads metal
identities and the series composition only, never a label.
"""
from __future__ import annotations

import hashlib

from gen20k import design as DS  # noqa: E402
from gen20k.data import Z_VEC  # noqa: E402

from .cache import KMAX

_MEMO: dict = {}


def _order_key(cell) -> str:
    return hashlib.sha256(str(cell["cell_id"]).encode()).hexdigest()


def _sequence(cell, cover: dict) -> list[int]:
    from .evaluate import _view
    dz = DS.make_design(_view(cell), "adjacent", "metal")
    target = min(dz.k_max, KMAX) + 1
    order: list[int] = []
    while len(order) < target:
        legal = DS.legal_metals(dz, frozenset(order))
        if not legal:
            break

        def key(m):
            spread = min(abs(int(Z_VEC[m]) - int(Z_VEC[u])) for u in order) if order else 0
            return (-cover.get(m, 0), spread, -int(Z_VEC[m]))
        order.append(max(legal, key=key))
    DS.assert_metal_set(dz, order)
    return order


def series_stagger(block: dict, cell: dict):
    cells = block["cells"]
    memo_key = (id(cells), len(cells), cells[0]["cell_id"] if cells else "")
    seqs = _MEMO.get(memo_key)
    if seqs is None:
        seqs = {}
        groups: dict = {}
        for c in cells:
            groups.setdefault((c["extractant"], c["publication"]), []).append(c)
        for members in groups.values():
            cover: dict = {}
            for c in sorted(members, key=_order_key):
                order = _sequence(c, cover)
                for m in order:
                    cover[m] = cover.get(m, 0) + 1
                seqs[c["ci"]] = DS.star_pairs(order)
        _MEMO[memo_key] = seqs
    return seqs[cell["ci"]]


# --------------------------------------------------------------------------------------
# DOPT_HB (planned as attempt 5; deferred after attempt 4's diagnosis, NOT run by any registered
# attempt): reads chosen to minimise the fitted HB model's posterior variance of the
# observable strict-adjacent query contrasts.  Label-free: it reads the cell's metal identities,
# its query set and the HB model fitted on the outer training fold.  First two metals jointly
# (one reading carries no contrast), then greedy; gen20's legality (independent read set, k_max
# reachable) at every step, so no query is ever read at both ends.
# --------------------------------------------------------------------------------------
import numpy as np  # noqa: E402

from gen20k.data import N_LN  # noqa: E402


def _query_post_var(C: np.ndarray, Q: np.ndarray, read: list[int]) -> float:
    """Sum over queries of Var(q | read contrasts) under the joint covariance C of the level vector."""
    prior = float(np.einsum("ij,jk,ik->", Q, C, Q))
    if len(read) < 2:
        return prior
    D = np.zeros((len(read) - 1, N_LN))
    for i, m in enumerate(read[1:]):
        D[i, read[0]], D[i, m] = 1.0, -1.0
    S = D @ C @ D.T
    G = Q @ C @ D.T
    return prior - float(np.einsum("ij,ji->", G, np.linalg.solve(S, G.T)))


def dopt_hb_sequence(cell: dict, C: np.ndarray) -> list:
    from .evaluate import _view
    dz = DS.make_design(_view(cell), "adjacent", "metal")
    Q = np.zeros((len(dz.query), N_LN))
    for i, (a, b) in enumerate(dz.query):
        Q[i, a], Q[i, b] = 1.0, -1.0
    target = min(dz.k_max, KMAX) + 1
    if target < 2:
        return []
    best = None
    for m0 in DS.legal_metals(dz, frozenset()):
        for m1 in DS.legal_metals(dz, frozenset([m0])):
            v = _query_post_var(C, Q, [m0, m1])
            key = (v, int(Z_VEC[m0]), int(Z_VEC[m1]))
            if best is None or key < best[0]:
                best = (key, [m0, m1])
    order = best[1] if best else []
    while len(order) < target:
        legal = DS.legal_metals(dz, frozenset(order))
        if not legal:
            break
        order.append(min(legal, key=lambda m: (_query_post_var(C, Q, order + [m]), int(Z_VEC[m]))))
    DS.assert_metal_set(dz, order)
    return DS.star_pairs(order)


def make_dopt_hb(model_key: str):
    """Policy closure: ``model_key`` names the fitted HB model in ``block['fitted']`` ('HBK@ML'
    resolves to the fold's maximum-likelihood kernel)."""
    memo: dict = {}

    def policy(block: dict, cell: dict):
        f = block["fitted"]
        key = f"HBK[{f['ml_kernel']}]" if model_key == "HBK@ML" else model_key
        mdl = f[key]
        mk = (id(mdl), cell["ci"])
        if mk not in memo:
            C = mdl.cov_full(mdl.S, mdl.tau2, mdl.rho, mdl.sig2)
            memo[mk] = dopt_hb_sequence(cell, C)
        return memo[mk]
    return policy
