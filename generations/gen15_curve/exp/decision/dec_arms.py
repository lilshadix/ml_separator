"""Arms used by the decision study.

Two things differ from ``gen15.arms``:

* ``g13_full`` is re-declared with ``n_jobs=2`` (the repo's ``tree_pipeline`` hard-codes ``-1``
  and this machine runs four agents at once).  Extra-trees with a fixed ``random_state`` is
  deterministic in the number of jobs, so the numbers are byte-comparable with the locked ones.
* ``g14_probe`` is gen14's deployed arm that *also* records the logistic's own probability for
  every held-out cell, so a confidence / coverage curve can be drawn from the model's own belief
  rather than only from the size of its prediction.
* ``permuted`` wraps any arm so that, inside the training fold only, the 137 ligand-derived
  columns are permuted between extractants.  Conditions, mass-action columns, weights, groups and
  targets stay attached to their own cell, so the only thing destroyed is the map from ligand
  structure to curve -- the permutation null the protocol asks for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "gen15_curve") not in sys.path:
    sys.path.insert(0, str(ROOT / "gen15_curve"))

from gen15 import arms as A                       # noqa: E402
from gen15.valuebench import Ctx                  # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS  # noqa: E402
from gen14.models import dir_logistic             # noqa: E402

flat = A.flat
mean_curve = A.mean_curve
o_both = A.o_both


def g13_full(ctx: Ctx) -> np.ndarray:
    """Gen13's 209-column extra-trees regression on both coefficients (n_jobs=2)."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        ExtraTreesRegressor(n_estimators=400, max_features=0.5, min_samples_leaf=2,
                            random_state=ctx.model_seed, n_jobs=2),
    )
    m.fit(ctx.X[ctx.train], ctx.bench.coef[ctx.train],
          extratreesregressor__sample_weight=ctx.w)
    return np.asarray(m.predict(ctx.X[ctx.test]), dtype=float).reshape(len(ctx.test), 2)


class G14Probe:
    """Gen14's arm, keeping the logistic's held-out probability of being heavy-selective."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def __call__(self, ctx: Ctx) -> np.ndarray:
        rtr = ctx.rich_train()
        X = ctx.feat("TOPO39")
        p = np.asarray(dir_logistic()(X[rtr], ctx.amp[rtr], ctx.rich_weights(),
                                      ctx.bench.groups[rtr], X[ctx.test], ctx.model_seed,
                                      ctx.bench.frame.extractant.to_numpy()[rtr]), dtype=float)
        sign = np.where(p >= 0.5, -1.0, 1.0)
        mag = ctx.train_mean_magnitude()
        cur = ctx.train_mean_curvature()
        cid = ctx.bench.frame["cell_id"].to_numpy()[ctx.test]
        for k, c in enumerate(cid):
            self.records.append({"split_seed": ctx.seed, "fold": ctx.fold, "cell_id": c,
                                 "p_heavy": float(p[k]), "dir_conf": float(abs(p[k] - 0.5)),
                                 "sign": float(sign[k]),
                                 "train_mean_magnitude": mag, "train_mean_curvature": cur})
        return np.c_[sign * mag, np.full(len(ctx.test), cur)]


# --------------------------------------------------------------------------------------
# permutation null
# --------------------------------------------------------------------------------------
def ligand_columns(bench) -> np.ndarray:
    """Indices into the LEAN matrix of the columns that are a property of the ligand alone."""
    cols = bench.columns(LEAN_BLOCKS)
    return np.array([i for i, c in enumerate(cols) if not c.startswith(("cond__", "massact__"))],
                    dtype=int)


class _PermCtx:
    """A Ctx look-alike whose ``X`` has the ligand block permuted between training extractants."""

    def __init__(self, ctx: Ctx, X: np.ndarray) -> None:
        self._ctx, self.X = ctx, X

    def __getattr__(self, name):          # everything else is the real fold
        return getattr(self._ctx, name)

    def feat(self, name: str) -> np.ndarray:
        return self.X[:, self._ctx.fs[name]]


def permuted(arm, rep: int, lig_cols: np.ndarray):
    """Wrap ``arm`` so the ligand block of the *training* cells is shuffled between extractants.

    The permutation is at the extractant level (all of an extractant's cells receive the same
    substitute structure), so the null keeps the corpus's replication structure -- how many cells
    an extractant has, and which publication each came from -- and destroys only the structure to
    curve map that every candidate in this programme is trying to learn.
    """
    def f(ctx: Ctx) -> np.ndarray:
        ext = ctx.bench.frame["extractant"].to_numpy()
        tr = ctx.train
        uniq = np.unique(ext[tr])
        rng = np.random.default_rng(hash((int(ctx.seed), int(ctx.fold), int(rep))) % (2 ** 32))
        target = uniq[rng.permutation(len(uniq))]
        # a representative cell for each substitute extractant (ligand columns are constant in it)
        rep_row = {e: int(np.flatnonzero(ext == e)[0]) for e in uniq}
        X = ctx.X.copy()
        src = np.array([rep_row[target[np.searchsorted(uniq, ext[i])]] for i in tr])
        X[np.ix_(tr, lig_cols)] = ctx.X[np.ix_(src, lig_cols)]
        return arm(_PermCtx(ctx, X))
    return f
