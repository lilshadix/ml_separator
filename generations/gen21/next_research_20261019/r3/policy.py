"""Direction D: a read policy aimed at the realised error of the target differences (POL_ERR).

Error model (fitted on the OUTER TRAINING fold only): for real training cells masked with DOPT_Q,
UNIFORM and RANDOM legal read sets at budget k (r3.corrector.training_situations' masks), the squared
residual of PWL(R) on every hidden strict-adjacent query is tabulated by configuration class
  (pair type, read status [one end read / both hidden], distance from the pair to the nearest read in
   atomic number, capped at 3, extrapolation flag)
as a weighted mean, shrunk toward the class mean over pair types with a pseudo-count of 20 rows.
Policy: for a held-out cell, every legal read set of size k + 1 (an independent set of the cell's
adjacency path that keeps k_max reachable, gen20's rule) is scored by the summed table value over the
cell's fixed queries, and the minimum is read (ties: lexicographic by Z).  It uses the cell's metal
identities, its query set and the train-fitted table only -- never a label of the held-out cell.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import interp
from . import paths  # noqa: F401
from gen20k import design as DS  # noqa: E402
from gen20k.data import N_LN, Z_VEC  # noqa: E402

PSEUDO = 20.0


def config_class(a: int, b: int, read: frozenset, metals) -> tuple:
    status = "one" if (a in read or b in read) else "none"
    rz = sorted(int(Z_VEC[m]) for m in read)
    za, zb = int(Z_VEC[a]), int(Z_VEC[b])
    dmin = 0 if status == "one" else min(min(abs(za - r), abs(zb - r)) for r in rz)
    extrap = int(max(za, zb) > rz[-1] or min(za, zb) < rz[0])
    # dmin: distance in atomic number (the Pm gap counts as 2), capped at 3
    return (status, min(dmin, 3), extrap)


def fit_error_table(bench, rec, Y, *, k: int = 3) -> dict:
    from . import corrector as CO
    from gen13sep.amplitude_bench import cell_weights
    tr = np.asarray(rec["outer"]["train_index"])
    wmap = dict(zip(tr.tolist(), cell_weights(bench.groups[tr], bench.n_obs[tr]).tolist()))
    acc: dict = {}
    for blk in rec["inner"]:
        for cell in blk["cells"]:
            if cell["k_max"] < k or cell["ci"] not in wmap:
                continue
            masks = [frozenset(m for p in cell["seqs"]["DOPT_Q"][:k] for m in p),
                     frozenset(m for p in cell["seqs"]["UNIFORM"][:k] for m in p)]
            masks += CO._random_legal(cell, k, CO.N_RANDOM_MASKS, "r3train")
            wc = wmap[cell["ci"]] / len(masks)
            for read in masks:
                ym = np.full(N_LN, np.nan)
                for m in read:
                    ym[m] = Y[cell["ci"]][m]
                base = interp.pwl_levels(ym, cell["metals"], "R")
                for a, b in cell["query"]:
                    r = (Y[cell["ci"]][a] - Y[cell["ci"]][b]) - (base[a] - base[b])
                    key = (int(a),) + config_class(a, b, read, cell["metals"])
                    s = acc.setdefault(key, [0.0, 0.0, 0])
                    s[0] += wc * r * r
                    s[1] += wc
                    s[2] += 1
    # class means over pair types, then shrink
    cls: dict = {}
    for key, (sw2, sw, n) in acc.items():
        c = cls.setdefault(key[1:], [0.0, 0.0])
        c[0] += sw2
        c[1] += sw
    table = {}
    for key, (sw2, sw, n) in acc.items():
        cm = cls[key[1:]][0] / cls[key[1:]][1]
        lam = n / (n + PSEUDO)
        table[key] = lam * (sw2 / sw) + (1 - lam) * cm
    cls_mean = {c: v[0] / v[1] for c, v in cls.items()}
    return {"table": table, "class_mean": cls_mean, "k": k}


def _legal_sets(cell, size: int):
    from g21.evaluate import _view
    dz = DS.make_design(_view(cell), "adjacent", "metal")
    target = DS.max_independent_size(cell["metals"], dz.query, frozenset())
    q = dz.query
    for S in itertools.combinations(sorted(cell["metals"], key=lambda m: Z_VEC[m]), size):
        s = frozenset(S)
        if any(a in s and b in s for a, b in q):
            continue
        if DS.max_independent_size(cell["metals"], q, s) != target:
            continue
        yield S


def choose(cell, err: dict, k: int):
    best = None
    for S in _legal_sets(cell, k + 1):
        read = frozenset(S)
        tot = 0.0
        for a, b in cell["query"]:
            key = (int(a),) + config_class(a, b, read, cell["metals"])
            tot += err["table"].get(key, err["class_mean"].get(key[1:], 0.05))
        cand = (tot, tuple(int(Z_VEC[m]) for m in S))
        if best is None or cand < best[0]:
            best = (cand, S)
    return list(best[1]) if best else []


class PolErrK:
    """R24: POL_ERR at each evaluated budget k (error table fitted at that budget: fitted['pol_err_k<k>'])."""
    name = "POL_ERR_K"
    per_k = True

    def __init__(self):
        self.memo: dict = {}

    def __call__(self, block, cell, k=None):
        err = block["fitted"][f"pol_err_k{k}"]
        key = (id(err), cell["ci"], k)
        hit = self.memo.get(key)
        if hit is None or hit[0] is not err:        # identity check: an id() can be recycled after a table is freed
            hit = self.memo[key] = (err, DS.star_pairs(choose(cell, err, k)))
        return hit[1]

    def fresh(self):
        return type(self)()


class PolErr:
    """Policy callable for r3.harness: block['fitted']['pol_err'] holds the fold's error table."""
    name = "POL_ERR"

    def __init__(self, k: int = 3):
        self.k = k
        self.memo: dict = {}

    def __call__(self, block, cell):
        err = block["fitted"]["pol_err"]
        key = (id(err), cell["ci"])
        hit = self.memo.get(key)
        if hit is None or hit[0] is not err:        # identity check: an id() can be recycled after a table is freed
            hit = self.memo[key] = (err, DS.star_pairs(choose(cell, err, self.k)))
        return hit[1]

    def fresh(self):
        return type(self)(self.k)
