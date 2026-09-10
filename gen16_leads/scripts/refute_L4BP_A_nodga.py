"""Refuter lens A, step 4: the diglycolamides removed from TRAINING as well as from scoring.

The lead's `contrasts_nodga.csv` drops sc009 from the scoring units only, so every arm is still
*trained* on the 375 diglycolamide cells -- which is the very thing the A-optimal criterion grabs
first.  Here sc009 is removed from the frozen folds on both sides (train and test) before any order
is computed, so no arm ever sees a diglycolamide.  The fold plan itself is untouched: the frozen
(seed, fold) assignment is kept and its indices are filtered.

Arms: AOPT_R, BIGGEST (descending well-determined-cell count), RANDOM (6 draws), RANDCM (the same
permutations with AOPT_R's training-cell count as the budget).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve",
           ROOT / "gen16_leads" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402
from gen13sep.splits import Fold, all_folds  # noqa: E402
from gen14.dirbench import feature_sets, load  # noqa: E402
from gen15.valuebench import MIN_METALS  # noqa: E402
from refute_L4BP_A_rerun import (BUDGETS, N_DRAWS, RNG_BASE, fit, greedy_aopt,  # noqa: E402
                                 predict_pairs, standardise, thin)

OUT = ROOT / "gen16_leads" / "results" / "refutation" / "L4BP" / "A"
DROP = "sc009"


def run(design: str = "BP") -> None:
    bench = load()
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    topo = X[:, fs["TOPO39"]]
    groups = bench.groups.astype(str)
    keep_cell = groups != DROP
    print(f"cohort without {DROP}: {int(keep_cell.sum())} of {len(keep_cell)} cells, "
          f"{bench.frame.extractant[keep_cell].nunique()} extractants")

    parts = {k: [] for k in BUDGETS}
    guard = {}
    t0 = time.time()
    for f0 in all_folds(bench.frame, design=design):
        tr = f0.train_index[keep_cell[f0.train_index]]
        te = f0.test_index[keep_cell[f0.test_index]]
        if len(te) == 0 or len(tr) == 0:
            continue
        f = Fold(design=design, seed=f0.seed, fold=f0.fold, train_index=tr, test_index=te,
                 inner_train_index=tr, inner_validation_index=tr[:0],
                 held_out_groups=f0.held_out_groups)
        pairs = _pair_frame(bench.frame, bench.Y, te, f.seed, f.fold)
        if pairs.empty:
            continue
        g = groups[tr]
        chemos = sorted(set(g))
        if len(chemos) < max(BUDGETS):
            budgets = tuple(k for k in BUDGETS if k <= len(chemos))
        else:
            budgets = BUDGETS
        Z = standardise(topo[tr], topo[tr])
        size_all = pd.Series(g).value_counts()
        size_rich = pd.Series(g[rich[tr]]).value_counts()
        aopt = greedy_aopt(chemos, Z, g, rich[tr])
        biggest = sorted(chemos, key=lambda c: (-int(size_rich.get(c, 0)), c))
        cum = np.cumsum([int(size_all.get(c, 0)) for c in aopt])
        perms = []
        for d in range(N_DRAWS):
            rng = np.random.default_rng(RNG_BASE + int(f.seed) * 97 + f.fold * 13 + d)
            perms.append([chemos[i] for i in rng.permutation(len(chemos))])
        base = pairs.drop(columns=["ia", "ib", "cell_local"])
        for k in budgets:
            t = base.copy()
            for nm, o in (("AOPT_R", aopt), ("BIGGEST", biggest)):
                coef, fb = fit(bench, fs, X, rich, thin(f, bench.groups, set(o[:k])), design)
                t[f"{nm}|0"] = predict_pairs(coef, pairs, bench)
                guard[(nm, k)] = guard.get((nm, k), 0) + int(fb)
            target = int(cum[k - 1])
            for d, p in enumerate(perms):
                coef, fb = fit(bench, fs, X, rich, thin(f, bench.groups, set(p[:k])), design)
                t[f"RANDOM|{d}"] = predict_pairs(coef, pairs, bench)
                guard[("RANDOM", k)] = guard.get(("RANDOM", k), 0) + int(fb)
                cc = np.cumsum([int(size_all.get(c, 0)) for c in p])
                j = min(int(np.searchsorted(cc, target, side="left")) + 1, len(p))
                coef, fb = fit(bench, fs, X, rich, thin(f, bench.groups, set(p[:j])), design)
                t[f"RANDCM|{d}"] = predict_pairs(coef, pairs, bench)
                guard[("RANDCM", k)] = guard.get(("RANDCM", k), 0) + int(fb)
            parts[k].append(t)
    print(f"[{design}] fits done in {time.time() - t0:.0f}s", flush=True)

    pe_parts, curve = [], []
    for k, plist in parts.items():
        if not plist:
            continue
        tab = pd.concat(plist, ignore_index=True)
        arms = [c for c in tab.columns if "|" in c]
        pe = per_extractant(tab, arms)
        s = summarise(pe, tab, arms).set_index("arm")
        pe = pe.assign(design=design, budget=k,
                       order=pe["arm"].str.split("|", regex=False).str[0],
                       draw=pe["arm"].str.split("|", regex=False).str[1].astype(int))
        pe_parts.append(pe)
        for nm in sorted({a.split("|")[0] for a in arms}):
            cols = [a for a in arms if a.startswith(nm + "|")]
            curve.append({"order": nm, "budget": k,
                          "macro_mae": float(s.loc[cols, "macro_mae_extractant"].mean()),
                          "n_flat_fallbacks": int(guard.get((nm, k), 0))})
    PE = pd.concat(pe_parts, ignore_index=True)
    C = pd.DataFrame(curve)
    C.to_csv(OUT / f"nodga_curve_{design}.csv", index=False)
    keys = ["design", "budget", "order", "split_seed", "extractant", "chemotype"]
    DA = PE.groupby(keys)[["mae_all"]].mean().reset_index()
    ABC = (DA.groupby(["design", "order", "split_seed", "extractant", "chemotype"])[["mae_all"]]
           .mean().reset_index())
    comps = {"AOPTR_vs_RANDOM": ("RANDOM", "AOPT_R"),
             "BIGGEST_vs_RANDOM": ("RANDOM", "BIGGEST"),
             "AOPTR_vs_BIGGEST": ("BIGGEST", "AOPT_R"),
             "AOPTR_vs_RANDCM": ("RANDCM", "AOPT_R"),
             "RANDCM_vs_RANDOM": ("RANDOM", "RANDCM")}
    CT = paired_contrasts(ABC.rename(columns={"order": "arm"}), comps, value="mae_all")
    CT.to_csv(OUT / f"nodga_contrasts_{design}.csv", index=False)
    pd.set_option("display.width", 260)
    print("\n=== learning curves, cohort WITHOUT sc009 (train and test) ===")
    print(C.pivot_table(index="order", columns="budget", values="macro_mae").round(4).to_string())
    print("\n=== FLAT-guard firings ===")
    print(C.pivot_table(index="order", columns="budget", values="n_flat_fallbacks").astype(int).to_string())
    print("\n=== contrasts (ABC on mae_all) ===")
    print(CT[["comparison", "point", "ci95_low", "ci95_high", "p_two_sided", "seeds_positive",
              "loco_min", "loco_max", "loco_sign_stable", "passes_P1", "n_units"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BP")
