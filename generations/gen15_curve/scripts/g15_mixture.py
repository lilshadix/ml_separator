"""Gen15 step 7: does a measurement pick the curve prototype?

Runs the pre-check first (how many nats separate the components' likelihoods, and how often the
posterior collapses onto one component), then scores the mixture against the pooled BLUP on a
common pair set.

Usage:  python gen15_curve/scripts/g15_mixture.py [designs] [--how dopt|widest] [--noshift]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen15 import arms as A  # noqa: E402
from gen15 import mixture as MX  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
DESIGNS = argv[0].split(",") if argv else ["BP"]
HOW = "dopt"
for i, a in enumerate(sys.argv):
    if a == "--how" and i + 1 < len(sys.argv):
        HOW = sys.argv[i + 1]
SHIFT = "--noshift" not in sys.argv
KK = (2, 4, 6)
TAG = f"{HOW}{'' if SHIFT else '_noshift'}"

COMPS = {}
for K in KK:
    for k in (1, 2, 3):
        COMPS[f"MIX{K}hard_vs_POOLED@k{k}"] = (f"POOLED@k{k}", f"MIX{K}hard@k{k}")
        COMPS[f"MIX{K}mean_vs_POOLED@k{k}"] = (f"POOLED@k{k}", f"MIX{K}mean@k{k}")
        COMPS[f"MIX{K}meanPC_vs_POOLED@k{k}"] = (f"POOLED@k{k}", f"MIX{K}meanPC@k{k}")
for k in (1, 3):
    COMPS[f"POOLED_vs_NAIVE@k{k}"] = (f"NAIVE@k{k}", f"POOLED@k{k}")

if __name__ == "__main__":
    t0 = time.time()
    bench = V.load()
    boards, cons, allchecks = [], [], []
    for d in DESIGNS:
        table, modes, checks = MX.evaluate(bench, A.g14, d, how=HOW, kk=KK, shift=SHIFT)
        if table.empty:
            print(f"  {d}: nothing scorable")
            continue
        allchecks.append(checks)
        pe = per_extractant(table, modes)
        bd = summarise(pe, table, modes)
        bd.insert(0, "design", d)
        boards.append(bd)
        c = paired_contrasts(pe, COMPS, value="mae_all", replicates=10_000)
        if len(c):
            c.insert(0, "design", d)
            cons.append(c)
    B = pd.concat(boards, ignore_index=True)
    V.RESULTS.mkdir(parents=True, exist_ok=True)
    B.to_csv(V.RESULTS / f"g15_mixture_board_{TAG}.csv", index=False)
    C = pd.concat(cons, ignore_index=True) if cons else pd.DataFrame()
    if len(C):
        C.to_csv(V.RESULTS / f"g15_mixture_contrasts_{TAG}.csv", index=False)
    CH = pd.concat(allchecks, ignore_index=True) if allchecks else pd.DataFrame()
    if len(CH):
        CH.to_csv(V.RESULTS / f"g15_mixture_precheck_{TAG}.csv", index=False)

    pd.set_option("display.width", 220)
    if len(CH):
        print("\n=== PRE-CHECK at k = 1: can one measurement tell the prototypes apart? ===")
        print(CH.groupby(["design", "K"]).agg(
            n=("nats", "size"), median_nats=("nats", "median"),
            q90_nats=("nats", lambda s: s.quantile(0.9)),
            mean_post_max=("post_max", "mean"),
            frac_collapsed=("collapsed", "mean")).round(3).to_string())
        print("  (median nats under ~1 and frac_collapsed near 1 would kill the idea)")
    print("\n=== extractant-macro MAE ===")
    w = V.wide(B)
    order = [f"{m}@k{k}" for m in ["POOLED", "NAIVE"] + [f"MIX{K}{r}" for K in KK
                                                         for r in ("hard", "mean", "meanPC")]
             for k in (0, 1, 2, 3)]
    print(w.reindex([m for m in order if m in w.index]).round(4).to_string())
    print("\n=== macro sign accuracy on strong pairs ===")
    ws = V.wide(B, "macro_sign_acc_strong")
    print(ws.reindex([m for m in order if m in ws.index]).round(4).to_string())
    if len(C):
        print("\n=== paired contrasts (positive favours the candidate) ===")
        print(C[["design", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
                 "units_improved", "n_units", "seeds_positive", "loco_sign_stable", "passes_P1"]]
              .round(4).to_string(index=False))
    print(f"\n[g15-mixture] total {time.time() - t0:.0f}s")
