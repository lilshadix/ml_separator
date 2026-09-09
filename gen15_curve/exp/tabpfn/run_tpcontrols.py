"""Is TabPFN's magnitude gain a ranking, or only a level?

``TP_MAG`` is the only arm in this experiment whose paired gain over ``G14`` keeps a positive sign
in four of the five designs.  Its out-of-fold Spearman with the true magnitude is **negative**
(-0.108 under BP), which already says it cannot be ranking cells correctly -- but "negative
Spearman" and "no contribution to the metric" are not the same statement, because the endpoint is
an absolute error over metal pairs and not a correlation.  So the claim is tested directly.

Two controls, neither of which sees a held-out label:

``TP_MAG_SHUF``  the same fold's TabPFN magnitudes, randomly permuted across the held-out cells.
                The level, the spread and the entire marginal distribution are preserved exactly;
                only the assignment of a magnitude to a cell is destroyed.  If ``TP_MAG`` and
                ``TP_MAG_SHUF`` score the same, the model contributes nothing beyond its marginal.
``TP_MAG_MEAN`` one number per fold -- the mean of that fold's own predictions.  A transductive
                diagnostic (it reads the held-out *features* in a batch, never a label), so it is
                not a deployable arm; it prices "predict a better constant" on its own.

The same pair is run for the CatBoost head, whose fully honest level control (``CB_MAG_LEVEL`` in
``run_controls.py``, level taken from the model's predictions on the training rows) is already
measured, so the two routes to the same question can be compared.
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


def _shuffled(base_fn):
    def f(ctx):
        v = np.asarray(base_fn(ctx), dtype=float).ravel()
        rng = np.random.default_rng(int(ctx.model_seed) * 1000003 + int(ctx.fold))
        return v[rng.permutation(len(v))]
    return f


def _meaned(base_fn):
    def f(ctx):
        v = np.asarray(base_fn(ctx), dtype=float).ravel()
        return np.full(len(v), float(v.mean()))
    return f


def main() -> None:
    cache = E.TPCache(OUT / "tp_oof.pkl")
    bench = V.load()
    rec = K.Recorder()
    tp_mag = cache.mag_fn()
    cb_mag = L.catboost_reg("logmag")

    arms = {
        "FLAT": E.reference("FLAT", GA.flat, rec),
        "G14": E.reference("G14", GA.g14, rec),
        "TP_MAG": E.compose("TP_MAG", rec, mag_fn=tp_mag),
        "TP_MAG_SHUF": E.compose("TP_MAG_SHUF", rec, mag_fn=_shuffled(tp_mag)),
        "TP_MAG_MEAN": E.compose("TP_MAG_MEAN", rec, mag_fn=_meaned(tp_mag)),
        "CB_MAG": E.compose("CB_MAG", rec, mag_fn=cb_mag),
        "CB_MAG_SHUF": E.compose("CB_MAG_SHUF", rec, mag_fn=_shuffled(cb_mag)),
    }
    comps = {
        "TP_MAG_vs_G14": ("G14", "TP_MAG"),
        "TP_MAG_SHUF_vs_G14": ("G14", "TP_MAG_SHUF"),
        "TP_MAG_MEAN_vs_G14": ("G14", "TP_MAG_MEAN"),
        "TP_MAG_vs_TP_MAG_SHUF": ("TP_MAG_SHUF", "TP_MAG"),
        "TP_MAG_vs_TP_MAG_MEAN": ("TP_MAG_MEAN", "TP_MAG"),
        "CB_MAG_vs_CB_MAG_SHUF": ("CB_MAG_SHUF", "CB_MAG"),
    }

    t0 = time.time()
    B, C, _ = V.score(bench, arms, V.DESIGNS, comps=comps)
    print(f"[tpctl] scored in {(time.time() - t0) / 60:.1f} min", flush=True)

    B.to_csv(OUT / "tpcontrol_board.csv", index=False)
    C.to_csv(OUT / "tpcontrol_contrasts.csv", index=False)
    r = rec.frame()
    r.to_csv(OUT / "tpcontrol_predictions.csv.gz", index=False)

    print("\n=== extractant-macro MAE (lower better) ===", flush=True)
    print(V.wide(B).round(4).to_string(), flush=True)
    print("\n=== predicted magnitude under BP ===", flush=True)
    b = r[(r.design == "BP") & (r.n_metals >= K.MIN_METALS)]
    print(b.assign(mag=b.a_pred.abs()).groupby("arm")["mag"]
          .agg(["mean", "std", "min", "max"]).round(4).to_string(), flush=True)
    print("\n=== paired gain, all designs (positive = first named better) ===", flush=True)
    print(C.pivot(index="comparison", columns="design",
                  values="point")[list(V.DESIGNS)].round(4).to_string(), flush=True)
    print("\n=== full BP inference ===", flush=True)
    print(C[C.design == "BP"][["comparison", "point", "ci95_low", "ci95_high", "mde_80",
                               "p_two_sided", "seeds_positive", "loco_sign_stable",
                               "passes_P1"]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
