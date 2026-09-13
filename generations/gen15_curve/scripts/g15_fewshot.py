"""Gen15 step 3: the measurement-assisted ladder, on a fixed pair set, under all five designs.

Every budget k is scored on byte-identical held-out pairs -- the support pairs of the largest budget
are excluded at every k -- so the ladder answers one question, not four easier ones.  Two references
are carried at every rung because they are the only ones that can settle what the *chemistry* is
worth once a measurement exists: ``NAIVE_LINE`` (a radius ramp through the measured pairs, no corpus
at all) and ``FLAT`` (the same BLUP correction applied to a zero prior, i.e. the learned residual
covariance with no ligand information).

Usage:
  python generations/gen15_curve/scripts/g15_fewshot.py [designs] [--how widest|dopt|random]
                                            [--cov empirical|smooth|blend] [--noise 0.09]
                                            [--maskpub] [--cellnoise]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen15 import arms as A  # noqa: E402
from gen15 import fewshot as FS  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
DESIGNS = argv[0].split(",") if argv else list(V.DESIGNS)


def opt(flag: str, default: str) -> str:
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


HOW = opt("--how", "widest")
COV = opt("--cov", "empirical")
NOISE = float(opt("--noise", str(FS.NOISE_VAR)))
MASKPUB = "--maskpub" in sys.argv
CELLNOISE = "--cellnoise" in sys.argv
TAG = f"{HOW}_{COV}_n{NOISE:g}" + ("_maskpub" if MASKPUB else "") + ("_cellnoise" if CELLNOISE else "")

ARMS = {"G14": A.g14, "FLAT": A.flat, "MEANCURVE": A.mean_curve, "OBOTH": A.o_both}
KS = (0, 1, 2, 3)
COMPS = {
    "G14k1_vs_NAIVEk1": ("NAIVE_LINE@k1", "G14@k1"),
    "G14k1_vs_FLATk1": ("FLAT@k1", "G14@k1"),
    "FLATk1_vs_NAIVEk1": ("NAIVE_LINE@k1", "FLAT@k1"),
    "G14k1_vs_G14k0": ("G14@k0", "G14@k1"),
    "G14k3_vs_G14k1": ("G14@k1", "G14@k3"),
    "OBOTHk1_vs_G14k1": ("G14@k1", "OBOTH@k1"),
    "G14k0_vs_FLATk0": ("FLAT@k0", "G14@k0"),
}

if __name__ == "__main__":
    t0 = time.time()
    bench = V.load()
    boards, cons = [], []
    for d in DESIGNS:
        r = FS.evaluate(bench, ARMS, d, ks=KS, how=HOW, cov_kind=COV, noise_var=NOISE,
                        mask_publication=MASKPUB, per_cell_noise=CELLNOISE)
        if r.pairs.empty:
            print(f"  {d}: no scorable cells")
            continue
        pe = per_extractant(r.pairs, r.modes)
        bd = summarise(pe, r.pairs, r.modes)
        bd.insert(0, "design", d)
        boards.append(bd)
        c = paired_contrasts(pe, COMPS, value="mae_all", replicates=10_000)
        if len(c):
            c.insert(0, "design", d)
            cons.append(c)
        V.RESULTS.mkdir(parents=True, exist_ok=True)
        pe.to_parquet(V.RESULTS / f"g15_fewshot_perext_{d}_{TAG}.parquet")
    B = pd.concat(boards, ignore_index=True)
    B.to_csv(V.RESULTS / f"g15_fewshot_board_{TAG}.csv", index=False)
    C = pd.concat(cons, ignore_index=True) if cons else pd.DataFrame()
    if len(C):
        C.to_csv(V.RESULTS / f"g15_fewshot_contrasts_{TAG}.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n=== extractant-macro MAE | support={HOW} cov={COV} noise={NOISE:g}"
          f"{' maskpub' if MASKPUB else ''}{' cellnoise' if CELLNOISE else ''} ===")
    print(V.wide(B).round(4).to_string())
    for col in ("macro_sign_acc_strong", "macro_pair_spearman", "macro_mae_far"):
        print(f"\n=== {col} ===")
        print(V.wide(B, col).round(4).to_string())
    if len(C):
        print("\n=== paired contrasts (positive favours the candidate) ===")
        print(C[["design", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
                 "units_improved", "n_units", "seeds_positive", "loco_sign_stable", "passes_P1"]]
              .round(4).to_string(index=False))
    print(f"\n[g15-fewshot] total {time.time() - t0:.0f}s")
