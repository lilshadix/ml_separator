"""Score the non-TabPFN learners of this experiment under all five designs.

CatBoost (with and without chemotype-grouped early stopping), a monotone-constrained XGBoost and
its unconstrained twin, an isotonic fit on the bite descriptor with the direction of the constraint
fixed a priori and with it chosen on the training fold, a spline GAM with ``k`` fixed and with
``k`` chosen by inner CV, and the symbolic search.  Every one of them replaces exactly one slot of
gen14's composition, so the table is a like-for-like comparison with ``G14``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K                                   # noqa: E402
from common import V                                 # noqa: E402
import learners as L                                 # noqa: E402
import exparms as E                                  # noqa: E402
from gen15 import arms as GA                         # noqa: E402

OUT = K.OUT


def main() -> None:
    bench = V.load()
    rec = K.Recorder()
    audit: list = []

    arms = {
        "FLAT": E.reference("FLAT", GA.flat, rec),
        "G14": E.reference("G14", GA.g14, rec),
        "CB_DIR": E.compose("CB_DIR", rec, dir_fn=L.catboost_dir("TOPO39", early=False)),
        "CB_DIR_ES": E.compose("CB_DIR_ES", rec, dir_fn=L.catboost_dir("TOPO39", early=True)),
        "CB_DIR209": E.compose("CB_DIR209", rec, dir_fn=L.catboost_dir("LEAN209", early=False)),
        "CB_MAG": E.compose("CB_MAG", rec, mag_fn=L.catboost_reg("logmag")),
        "CB_CURV": E.compose("CB_CURV", rec, curv_fn=L.catboost_reg("curv")),
        "XGB_MONO": E.compose("XGB_MONO", rec, dir_fn=L.xgb_dir("TOPO39", L.BITE)),
        "XGB_FREE": E.compose("XGB_FREE", rec, dir_fn=L.xgb_dir("TOPO39", None)),
        "ISO_PRIOR": E.compose("ISO_PRIOR", rec, dir_fn=L.isotonic_dir(L.BITE, increasing=True)),
        "ISO_PICK": E.compose("ISO_PICK", rec, dir_fn=L.isotonic_dir(L.BITE, increasing=None)),
        "GAM_DIR": E.compose("GAM_DIR", rec, dir_fn=L.gam_dir("TOPO39", k=8)),
        "GAM_DIR_K": E.compose("GAM_DIR_K", rec, dir_fn=L.gam_dir("TOPO39", inner_pick_k=True)),
        "GAM_MAG": E.compose("GAM_MAG", rec, mag_fn=L.gam_reg("logmag")),
        "SYM_DIR": E.compose("SYM_DIR", rec, dir_fn=L.symbolic_search("TOPO39", audit=audit)),
    }
    cands = [a for a in arms if a not in ("FLAT", "G14")]
    comps = {f"{a}_vs_G14": ("G14", a) for a in cands}
    comps.update({f"{a}_vs_FLAT": ("FLAT", a) for a in cands})
    comps["G14_vs_FLAT"] = ("FLAT", "G14")

    t0 = time.time()
    B, C, tables = V.score(bench, arms, V.DESIGNS, comps=comps)
    print(f"[cheap] scored in {(time.time() - t0) / 60:.1f} min", flush=True)

    B.to_csv(OUT / "cheap_board.csv", index=False)
    C.to_csv(OUT / "cheap_contrasts.csv", index=False)
    r = rec.frame()
    r.to_csv(OUT / "cheap_predictions.csv.gz", index=False)
    db = K.direction_board(r)
    db.to_csv(OUT / "cheap_direction.csv", index=False)
    if audit:
        pd.DataFrame(audit).to_csv(OUT / "symbolic_audit.csv", index=False)

    print("\n=== extractant-macro MAE (lower better) ===")
    print(V.wide(B).round(4).to_string())
    print("\n=== direction macro accuracy, gen14 yardstick (higher better) ===")
    print(K.wide_dir(db).round(4).to_string())
    print("\n=== pairwise sign accuracy on strong pairs ===")
    print(V.wide(B, "macro_sign_acc_strong").round(4).to_string())
    print("\n=== pair Spearman ===")
    print(V.wide(B, "macro_pair_spearman").round(4).to_string())
    print("\n=== contrasts under BP ===")
    cbp = C[C.design == "BP"][["comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
                               "seeds_positive", "loco_sign_stable", "passes_P1"]]
    print(cbp.round(4).to_string(index=False))
    if audit:
        a = pd.DataFrame(audit)
        print("\n=== symbolic search: selection bias ===")
        g = a.groupby("design")[["inner_cv_acc", "outer_acc", "bite_outer_acc",
                                 "oracle_best_outer_acc"]].mean()
        print(g.round(4).to_string())


if __name__ == "__main__":
    main()
