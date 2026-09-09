"""Score the TabPFN v2 arms under all five designs, from the precomputed out-of-fold cache.

``tp_precompute.py`` must have been run first; it walks the same fold plan and writes
``tp_oof.pkl``.  This script only composes those predictions into arms and scores them, so the
expensive part is never repeated and the arms sit on byte-identical folds with ``FLAT`` and
``G14``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K                                   # noqa: E402
from common import V                                 # noqa: E402
import exparms as E                                  # noqa: E402
from gen15 import arms as GA                         # noqa: E402

OUT = K.OUT


def main() -> None:
    cache = E.TPCache(OUT / "tp_oof.pkl")
    print(f"[tp] {len(cache.store)} folds in the cache", flush=True)
    bench = V.load()
    rec = K.Recorder()

    arms = {
        "FLAT": E.reference("FLAT", GA.flat, rec),
        "G14": E.reference("G14", GA.g14, rec),
        "TP_DIR39": E.compose("TP_DIR39", rec, dir_fn=cache.dir_fn("p39")),
        "TP_DIR209": E.compose("TP_DIR209", rec, dir_fn=cache.dir_fn("p209")),
        "TP_MAG": E.compose("TP_MAG", rec, mag_fn=cache.mag_fn()),
        "TP_CURV": E.compose("TP_CURV", rec, curv_fn=cache.curv_fn()),
    }

    def tp_all(ctx):
        import numpy as np
        p = cache.dir_fn("p39")(ctx)
        s = np.where(p >= 0.5, -1.0, 1.0)
        a = s * cache.mag_fn()(ctx)
        rec.log(ctx, "TP_ALL", a)
        return np.c_[a, cache.curv_fn()(ctx)]
    arms["TP_ALL"] = tp_all

    cands = [a for a in arms if a not in ("FLAT", "G14")]
    comps = {f"{a}_vs_G14": ("G14", a) for a in cands}
    comps.update({f"{a}_vs_FLAT": ("FLAT", a) for a in cands})
    comps["G14_vs_FLAT"] = ("FLAT", "G14")

    t0 = time.time()
    B, C, tables = V.score(bench, arms, V.DESIGNS, comps=comps)
    print(f"[tp] scored in {(time.time() - t0) / 60:.1f} min", flush=True)

    B.to_csv(OUT / "tabpfn_board.csv", index=False)
    C.to_csv(OUT / "tabpfn_contrasts.csv", index=False)
    r = rec.frame()
    r.to_csv(OUT / "tabpfn_predictions.csv.gz", index=False)
    db = K.direction_board(r)
    db.to_csv(OUT / "tabpfn_direction.csv", index=False)

    print("\n=== extractant-macro MAE (lower better) ===")
    print(V.wide(B).round(4).to_string())
    print("\n=== direction macro accuracy, gen14 yardstick ===")
    print(K.wide_dir(db).round(4).to_string())
    print("\n=== pairwise sign accuracy on strong pairs ===")
    print(V.wide(B, "macro_sign_acc_strong").round(4).to_string())
    print("\n=== pair Spearman ===")
    print(V.wide(B, "macro_pair_spearman").round(4).to_string())
    print("\n=== contrasts under BP ===")
    print(C[C.design == "BP"][["comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
                               "seeds_positive", "loco_sign_stable",
                               "passes_P1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
