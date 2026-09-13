"""Capacity controls: what resolution does TOPO39 actually have, and is that enough to memorise?

g14_straw.py asks "could a chemistry-free barcode score what TOPO39 scores?".  This asks the
complementary and, it turns out, sharper question: *how many distinct things can TOPO39 say?*

Measured here: over the 82 scored extractants the 39 donor-topology columns take only 23 distinct
values, and one of those values is shared by 40 of the 90 extractants.  A featurisation that
cannot tell 44 % of the corpus apart cannot be memorising extractant identity -- so the identity
worry that motivates the straw control is structurally impossible for this block, and the straw
control's job changes from "is it identity?" to "is it more than a 23-row lookup table?".

Four arms, all under the frozen gen14 fold plans:

  REAL_TOPO39        the deployed L2 logistic on the 39 columns.
  TOPO_CLASS_LOOKUP  no model at all: a held-out cell inherits the chemotype-weighted majority
                     direction of the training cells whose TOPO39 row is bit-identical to its own;
                     unseen rows fall back to the training majority.  This is the cheapest
                     sensible alternative (feedback-evaluation-protocol) and the arm the logistic
                     has to beat to be worth its 39 coefficients.
  ONEHOT_chemotype   Chuang & Keiser's second control (Science 362:eaat8603) -- the maximally
                     memorising featurisation.  Under B/BR/BQ/BP a held-out chemotype's column is
                     all-zero in training, so this arm MUST collapse to the intercept.  It is a
                     leak assertion on the fold plan itself, not only a straw.
  ONEHOT_publication same, keyed on publication: must collapse under BP alone.
  STRAW_topo_class   a barcode drawn once per TOPO39 equivalence class -- the granularity-matched
                     straw.  If this scores what REAL_TOPO39 scores, TOPO39 is a class label and
                     nothing more; if it collapses, the *values* of the topology vector carry the
                     signal, not merely the partition they induce.

Usage:  python generations/gen14_direction/scripts/g14_capacity.py --designs A,B,BR,BQ,BP --draws 20
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gen13_separation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen14 import dirbench as db                                     # noqa: E402
from gen14 import models as M                                        # noqa: E402
from gen13sep.amplitude_bench import LEAN_BLOCKS                     # noqa: E402
from g14_straw import run_matrix, macro_accuracy, barcode            # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
DESIGNS = ("A", "B", "BR", "BQ", "BP")


def row_key(X: np.ndarray) -> np.ndarray:
    """Integer label of each row's exact TOPO39 value (bit-identical rows share a label)."""
    _, inv = np.unique(np.round(X, 6), axis=0, return_inverse=True)
    return inv


def dir_lookup(Xtr, amp, w, groups, Xte, seed, ext=None):
    """Chemotype-weighted majority direction among training rows identical to the test row.

    No fitting, no regularisation, no hyperparameter.  Ties and unseen rows fall back to the
    training majority, so the arm is always defined.
    """
    ktr = [tuple(np.round(r, 6)) for r in Xtr]
    kte = [tuple(np.round(r, 6)) for r in Xte]
    y = (amp < 0).astype(float)
    fallback = float(np.average(y, weights=w))
    num, den = {}, {}
    for k, yi, wi in zip(ktr, y, w):
        num[k] = num.get(k, 0.0) + wi * yi
        den[k] = den.get(k, 0.0) + wi
    return np.array([num[k] / den[k] if den.get(k, 0.0) > 1e-12 else fallback for k in kte])


def onehot(frame: pd.DataFrame, column: str) -> np.ndarray:
    key = frame[column].astype(str).to_numpy()
    names = np.unique(key)
    lut = {n: i for i, n in enumerate(names)}
    Z = np.zeros((len(key), len(names)))
    Z[np.arange(len(key)), [lut[k] for k in key]] = 1.0
    return Z


def class_barcode(bench, cls: np.ndarray, seed: int, d: int = 39) -> np.ndarray:
    names = np.unique(cls)
    table = np.random.default_rng(seed).standard_normal((len(names), d))
    lut = {n: i for i, n in enumerate(names)}
    return table[np.array([lut[c] for c in cls])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default=",".join(DESIGNS))
    ap.add_argument("--draws", type=int, default=20)
    args = ap.parse_args()

    bench = db.load()
    FS = db.feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)[:, FS["TOPO39"]]
    cls = row_key(X)
    frame = bench.frame
    rich = frame.n_metals.to_numpy() >= db.MIN_METALS

    # ---- the structural diagnostic, before any model is fitted -------------------------
    ext = frame.extractant.astype(str).to_numpy()
    _, first = np.unique(ext, return_index=True)
    print(f"[capacity] {len(first)} extractants -> {len(np.unique(cls[first]))} distinct TOPO39 rows; "
          f"largest class holds {pd.Series(cls[first]).value_counts().iat[0]} of them")

    cand_real = M.candidate(M.dir_logistic(), M.amp_constant)
    cand_look = M.candidate(dir_lookup, M.amp_constant)
    Zc, Zp = onehot(frame, "chemotype"), onehot(frame, "publication_id")

    rows = []
    for design in args.designs.split(","):
        arms = {
            "REAL_TOPO39":       (cand_real, X),
            "TOPO_CLASS_LOOKUP": (cand_look, X),
            "ONEHOT_chemotype":  (cand_real, Zc),
            "ONEHOT_publication": (cand_real, Zp),
            "CONST_heavy":       (M.candidate(M.dir_always_heavy), X),
        }
        for name, (c, mat) in arms.items():
            a = macro_accuracy(run_matrix(bench, name, c, mat, design=design))
            rows.append({"design": design, "arm": name, "mean": a, "sd": 0.0, "n": 1})
            print(f"  {design:3s} {name:20s} {a:.4f}", flush=True)
        accs = [macro_accuracy(run_matrix(bench, "STRAW_topo_class", cand_real,
                                          class_barcode(bench, cls, 20260900 + k), design=design))
                for k in range(args.draws)]
        rows.append({"design": design, "arm": "STRAW_topo_class", "mean": float(np.mean(accs)),
                     "sd": float(np.std(accs, ddof=1)), "n": len(accs),
                     "lo": float(np.min(accs)), "hi": float(np.max(accs))})
        print(f"  {design:3s} {'STRAW_topo_class':20s} {np.mean(accs):.4f} "
              f"+-{np.std(accs, ddof=1):.4f}", flush=True)

    out = pd.DataFrame(rows)
    db.RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(db.RESULTS / "g14_capacity.csv", index=False)
    print("\n" + out.round(4).to_string(index=False))

    # ---- where the accuracy actually comes from: per-TOPO-class decomposition ----------
    oof = run_matrix(bench, "REAL_TOPO39", cand_real, X, design="BP")
    u = oof.cells[oof.cells.n_metals >= db.MIN_METALS].copy()
    u["hit"] = ((u.p >= 0.5).astype(int) == u.y).astype(float)
    u["topo_class"] = cls[u.cell_index.to_numpy()]
    per = (u.groupby(["topo_class", "extractant"])["hit"].mean().reset_index()
             .groupby("topo_class")
             .agg(n_extractants=("extractant", "nunique"), macro_hit=("hit", "mean"))
             .sort_values("n_extractants", ascending=False))
    per["share_of_units"] = per.n_extractants / per.n_extractants.sum()
    per.to_csv(db.RESULTS / "g14_capacity_by_class_BP.csv")
    print("\n=== BP macro accuracy decomposed by TOPO39 equivalence class ===")
    print(per.head(12).round(4).to_string())


if __name__ == "__main__":
    main()
