"""Decompose any magnitude head into the two things it can be doing.

A magnitude model can beat ``G14`` for two completely different reasons, and the programme's own
negative result -- "seven priors for |a| all land within 0.007 of a constant" -- only rules out one
of them.  A model can

* move the **level**: predict a smaller (or larger) constant magnitude than the training fold's
  weighted mean, which is the constant ``G14`` uses.  An absolute-error metric prefers a value near
  the weighted *median* of |a|, and every regressor fitted with an L1 loss on log|a| drifts there on
  its own.  That is not chemistry, it is loss geometry, and it is available for free.
* move the **ranking**: give cell A a larger magnitude than cell B.  That is the thing the
  programme wants and the thing it has never found.

The controls below separate them without touching a held-out label.  For each magnitude head:

``*_LEVEL``  every test cell gets one number -- the training-weighted mean of the head's own
             predictions **on the training rows** -- so the ranking is destroyed and only the level
             survives.
``*_RANK``   the head's test predictions multiplied by ``train_mean_magnitude / that same level``,
             so the level is put back to ``G14``'s and only the ranking survives.

``MAG_MED`` is the cheapest sensible alternative to both: the weighted median of |a| on the rich
training cells, a constant, no features at all.  If ``*_LEVEL`` and ``MAG_MED`` capture the whole
gain, the head has found no chemistry.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K                                   # noqa: E402
from common import V                                 # noqa: E402
import learners as L                                 # noqa: E402
import exparms as E                                  # noqa: E402
from gen15 import arms as GA                         # noqa: E402

OUT = K.OUT


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    v, w = np.asarray(v)[o], np.asarray(w)[o]
    c = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(c, 0.5)])


def mag_median(ctx) -> np.ndarray:
    rtr = ctx.rich_train()
    return np.full(len(ctx.test), _wmedian(np.abs(ctx.amp[rtr]), ctx.rich_weights()))


def _cb_mag_parts(features: str = "TOPO39", iterations: int = 400):
    """Return (level_fn, rank_fn) for the CatBoost log-magnitude head."""
    from catboost import CatBoostRegressor
    cache: dict = {"key": None}

    def _fit(ctx):
        k = (ctx.design, ctx.seed, ctx.fold)
        if cache["key"] == k:
            return cache["val"]
        fd, Xtr, Xte = L._xy(ctx, features)
        m = CatBoostRegressor(iterations=iterations, loss_function="MAE",
                              random_seed=ctx.model_seed, **L._CB_FIXED)
        m.fit(Xtr, fd.y_logmag, sample_weight=fd.w)
        te = np.clip(np.exp(np.asarray(m.predict(Xte), float)) - K.EPS, 0.0, None)
        tr = np.clip(np.exp(np.asarray(m.predict(Xtr), float)) - K.EPS, 0.0, None)
        level = float(np.average(tr, weights=fd.w))
        cache["key"], cache["val"] = k, (te, level)
        return te, level

    def level_fn(ctx):
        _, level = _fit(ctx)
        return np.full(len(ctx.test), level)

    def rank_fn(ctx):
        te, level = _fit(ctx)
        scale = ctx.train_mean_magnitude() / max(level, 1e-9)
        return te * scale

    return level_fn, rank_fn


def main() -> None:
    bench = V.load()
    rec = K.Recorder()
    lvl, rnk = _cb_mag_parts()

    arms = {
        "FLAT": E.reference("FLAT", GA.flat, rec),
        "G14": E.reference("G14", GA.g14, rec),
        "MAG_MED": E.compose("MAG_MED", rec, mag_fn=mag_median),
        "CB_MAG": E.compose("CB_MAG", rec, mag_fn=L.catboost_reg("logmag")),
        "CB_MAG_LEVEL": E.compose("CB_MAG_LEVEL", rec, mag_fn=lvl),
        "CB_MAG_RANK": E.compose("CB_MAG_RANK", rec, mag_fn=rnk),
    }
    cands = [a for a in arms if a not in ("FLAT", "G14")]
    comps = {f"{a}_vs_G14": ("G14", a) for a in cands}
    comps["CB_MAG_vs_MAG_MED"] = ("MAG_MED", "CB_MAG")
    comps["CB_MAG_vs_CB_MAG_LEVEL"] = ("CB_MAG_LEVEL", "CB_MAG")
    # the one contrast that isolates the ranking: same level, spread on or off
    comps["CB_MAG_RANK_vs_CB_MAG_LEVEL"] = ("CB_MAG_LEVEL", "CB_MAG_RANK")
    comps["CB_MAG_LEVEL_vs_MAG_MED"] = ("MAG_MED", "CB_MAG_LEVEL")

    t0 = time.time()
    B, C, _ = V.score(bench, arms, V.DESIGNS, comps=comps)
    print(f"[controls] scored in {(time.time() - t0) / 60:.1f} min", flush=True)

    B.to_csv(OUT / "control_board.csv", index=False)
    C.to_csv(OUT / "control_contrasts.csv", index=False)
    r = rec.frame()
    r.to_csv(OUT / "control_predictions.csv.gz", index=False)

    print("\n=== extractant-macro MAE (lower better) ===", flush=True)
    print(V.wide(B).round(4).to_string(), flush=True)
    print("\n=== predicted magnitude: level and spread, BP ===", flush=True)
    b = r[(r.design == "BP") & (r.n_metals >= K.MIN_METALS)]
    g = b.assign(mag=b.a_pred.abs()).groupby("arm")["mag"].agg(["mean", "std", "min", "max"])
    print(g.round(4).to_string(), flush=True)
    print("\n=== contrasts under BP ===", flush=True)
    print(C[C.design == "BP"][["comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
                               "seeds_positive", "loco_sign_stable",
                               "passes_P1"]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
