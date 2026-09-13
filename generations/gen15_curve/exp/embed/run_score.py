"""Score a named set of embedding arms under all five designs and write the tables.

Usage:  python run_score.py <batch> [designs]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (str(ROOT / "generations" / "gen15_curve"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import arms_embed as E                                      # noqa: E402
import features as FEAT                                     # noqa: E402
from gen15 import arms as A, valuebench as V                 # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def batch_morgan_dir() -> dict:
    d = {"FLAT": A.flat, "G14": A.g14}
    for r in ("morgan2", "morgan3"):
        for C in (0.01, 0.1, 1.0):
            d[f"DIR_{r}_C{C:g}"] = E.direction_arm(r, C=C)
        for k in (8, 16):
            d[f"DIR_{r}_P{k}_C1"] = E.direction_arm(r, C=1.0, n_pca=k)
    return d


def batch_dir(tags: list[str]) -> dict:
    d = {"FLAT": A.flat, "G14": A.g14}
    for t in tags:
        short = t.replace("chemberta_", "cb").replace("molformer_", "mf").replace("__", "_")
        for C in (0.01, 0.1, 1.0):
            d[f"DIR_{short}_C{C:g}"] = E.direction_arm(t, C=C)
    return d


def batch_pca(tags: list[str]) -> dict:
    d = {"FLAT": A.flat, "G14": A.g14}
    for t in tags:
        short = t.replace("chemberta_", "cb").replace("molformer_", "mf").replace("__", "_")
        for k in (8, 16, 20):
            d[f"DIR_{short}_P{k}"] = E.direction_arm(t, C=1.0, n_pca=k)
    return d


def batch_magcurv(tags: list[str]) -> dict:
    d = {"FLAT": A.flat, "G14": A.g14}
    for t in tags:
        short = t.replace("chemberta_", "cb").replace("molformer_", "mf").replace("__", "_")
        d[f"MAG_{short}_ridge"] = E.magnitude_arm(t, how="ridge", alpha=100.0)
        d[f"MAG_{short}_rbf"] = E.magnitude_arm(t, how="rbf", alpha=1.0)
        d[f"MAG_{short}_cos"] = E.magnitude_arm(t, how="cosine", alpha=1.0)
        d[f"CUR_{short}_ridge"] = E.curvature_arm(t, how="ridge", alpha=100.0)
        d[f"CUR_{short}_sign"] = E.curvature_arm(t, how="sign", alpha=1.0)
    return d


def _short(t: str) -> str:
    return t.replace("chemberta_", "cb").replace("molformer_", "mf").replace("__", "_")


#: the representation the magnitude / curvature / composite arms are built on, declared BEFORE any
#: score is looked at: the MTR checkpoint's mean-pooled vector, which is the encoder the repo's own
#: manifest names first and the one a deployment would reach for.  Choosing it on the held-out MAE
#: would be exactly the researcher degree of freedom this programme has been burned by twice.
PRIMARY = "chemberta_mtr__mean"


def batch_main(tags: list[str]) -> dict:
    """The five-design headline set: one arm per role, plus the references.

    ``C = 1`` throughout, gen14's rule ("C = 1 unless something is dramatically better in EVERY
    design"); the C sweep is shown as three extra direction arms so its whole range is visible.
    """
    d = {"FLAT": A.flat, "G14": A.g14, "DIR_morgan2_C1": E.direction_arm("morgan2", C=1.0)}
    for t in tags:
        d[f"DIR_{_short(t)}_C1"] = E.direction_arm(t, C=1.0)
    s = _short(PRIMARY)
    d[f"DIR_{s}_C0.1"] = E.direction_arm(PRIMARY, C=0.1)
    d[f"DIR_{s}_C0.01"] = E.direction_arm(PRIMARY, C=0.01)
    d[f"DIR_{s}_P16"] = E.direction_arm(PRIMARY, C=1.0, n_pca=16)
    d[f"MAG_{s}_ridge"] = E.magnitude_arm(PRIMARY, how="ridge", alpha=100.0)
    d[f"MAG_{s}_cos"] = E.magnitude_arm(PRIMARY, how="cosine", alpha=1.0)
    d[f"CUR_{s}_ridge"] = E.curvature_arm(PRIMARY, how="ridge", alpha=100.0)
    d[f"CUR_{s}_sign"] = E.curvature_arm(PRIMARY, how="sign", alpha=1.0)
    d[f"COMP_{s}"] = E.composite_arm(PRIMARY, C=1.0, how="ridge", alpha=100.0)
    d[f"CONCAT_{s}_P16"] = E.concat_arm(PRIMARY, C=1.0, n_pca=16)
    return d


BATCHES = {
    "morgan_dir": lambda: batch_morgan_dir(),
    "morgan_magcurv": lambda: batch_magcurv(["morgan2"]),
}


def main() -> None:
    name = sys.argv[1]
    designs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(V.DESIGNS)
    if name in BATCHES:
        arms = BATCHES[name]()
    elif name.startswith("dir:"):
        arms = batch_dir(name[4:].split(","))
    elif name.startswith("pca:"):
        arms = batch_pca(name[4:].split(","))
    elif name.startswith("magcurv:"):
        arms = batch_magcurv(name[8:].split(","))
    elif name.startswith("main:"):
        arms = batch_main(name[5:].split(","))
    else:
        raise SystemExit(f"unknown batch {name!r}")
    comps = {f"{k}_vs_G14": ("G14", k) for k in arms if k not in ("G14", "FLAT")}
    comps.update({f"{k}_vs_FLAT": ("FLAT", k) for k in arms if k not in ("G14", "FLAT")})
    bench = V.load()
    t0 = time.time()
    B, C, _ = V.score(bench, arms, designs, comps=comps)
    tag = name.replace(":", "_").replace(",", "+")
    B.to_csv(OUT / f"board_{tag}.csv", index=False)
    if len(C):
        C.to_csv(OUT / f"contrast_{tag}.csv", index=False)
    print(f"\n=== {name}  extractant-macro MAE  ({time.time() - t0:.0f}s) ===")
    print(V.wide(B).round(4).to_string())
    print("\n--- macro_sign_acc_strong ---")
    print(V.wide(B, "macro_sign_acc_strong").round(4).to_string())
    print("\n--- macro_pair_spearman ---")
    print(V.wide(B, "macro_pair_spearman").round(4).to_string())
    if len(C):
        print("\n--- contrasts ---")
        print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
