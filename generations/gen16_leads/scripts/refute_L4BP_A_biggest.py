"""Refuter lens A, step 6: the chemistry-free size control on the remaining designs.

`refute_L4BP_A_rerun.py` ran the full arm set on BP and B.  This runs the four arms that carry the
argument on BR, BQ and A so the control is available in all five designs:

    AOPT_R   the lead's criterion, re-implemented from the pre-registration
    BIGGEST  chemotypes in descending well-determined-cell count -- zero chemistry, zero features
    RANDOM   uniform random order, the refuter's own RNG stream (6 draws)
    RANDCM   the SAME permutations, but the budget at rung k is AOPT_R's training-CELL count,
             not k chemotypes

Endpoint and inference identical to the lead's: ABC over k in (6,9,12,16,20,24) per extractant,
gen13sep.inference.paired_contrasts, margin 0.02.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen15_curve",
           ROOT / "generations" / "gen16_leads" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402
from gen13sep.splits import all_folds  # noqa: E402
from gen14.dirbench import feature_sets, load  # noqa: E402
from gen15.valuebench import MIN_METALS  # noqa: E402
from refute_L4BP_A_rerun import (BUDGETS, N_DRAWS, RNG_BASE, fit, greedy_aopt,  # noqa: E402
                                 predict_pairs, standardise, thin)

OUT = ROOT / "generations" / "gen16_leads" / "results" / "refutation" / "L4BP" / "A"
VALUE = "mae_all"
COMPS = {"AOPTR_vs_RANDOM": ("RANDOM", "AOPT_R"),
         "BIGGEST_vs_RANDOM": ("RANDOM", "BIGGEST"),
         "AOPTR_vs_BIGGEST": ("BIGGEST", "AOPT_R"),
         "AOPTR_vs_RANDCM": ("RANDCM", "AOPT_R"),
         "RANDCM_vs_RANDOM": ("RANDOM", "RANDCM")}


def run(design: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bench = load()
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    topo = X[:, fs["TOPO39"]]
    groups = bench.groups.astype(str)

    parts = {k: [] for k in BUDGETS}
    guard: dict = {}
    t0 = time.time()
    for f in all_folds(bench.frame, design=design):
        pairs = _pair_frame(bench.frame, bench.Y, f.test_index, f.seed, f.fold)
        if pairs.empty:
            continue
        tr = f.train_index
        g = groups[tr]
        chemos = sorted(set(g))
        Z = standardise(topo[tr], topo[tr])
        size_rich = pd.Series(g[rich[tr]]).value_counts()
        size_all = pd.Series(g).value_counts()

        aopt = greedy_aopt(chemos, Z, g, rich[tr])
        biggest = sorted(chemos, key=lambda c: (-int(size_rich.get(c, 0)), c))
        cum = np.cumsum([int(size_all.get(c, 0)) for c in aopt])
        targets = {k: int(cum[k - 1]) for k in BUDGETS}

        perms = []
        for d in range(N_DRAWS):
            rng = np.random.default_rng(RNG_BASE + int(f.seed) * 97 + f.fold * 13 + d)
            perms.append([chemos[i] for i in rng.permutation(len(chemos))])

        base = pairs.drop(columns=["ia", "ib", "cell_local"])
        for k in BUDGETS:
            t = base.copy()
            for nm, o in (("AOPT_R", aopt), ("BIGGEST", biggest)):
                coef, fb = fit(bench, fs, X, rich, thin(f, bench.groups, set(o[:k])), design)
                t[f"{nm}|0"] = predict_pairs(coef, pairs, bench)
                guard[(nm, k)] = guard.get((nm, k), 0) + int(fb)
            for d, p in enumerate(perms):
                coef, fb = fit(bench, fs, X, rich, thin(f, bench.groups, set(p[:k])), design)
                t[f"RANDOM|{d}"] = predict_pairs(coef, pairs, bench)
                guard[("RANDOM", k)] = guard.get(("RANDOM", k), 0) + int(fb)
                cc = np.cumsum([int(size_all.get(c, 0)) for c in p])
                j = min(int(np.searchsorted(cc, targets[k], side="left")) + 1, len(p))
                coef, fb = fit(bench, fs, X, rich, thin(f, bench.groups, set(p[:j])), design)
                t[f"RANDCM|{d}"] = predict_pairs(coef, pairs, bench)
                guard[("RANDCM", k)] = guard.get(("RANDCM", k), 0) + int(fb)
            parts[k].append(t)
    print(f"[{design}] fits {time.time() - t0:.0f}s", flush=True)

    pe_parts, curve = [], []
    for k, plist in parts.items():
        tab = pd.concat(plist, ignore_index=True)
        arms = [c for c in tab.columns if "|" in c]
        pe = per_extractant(tab, arms)
        s = summarise(pe, tab, arms).set_index("arm")
        pe = pe.assign(design=design, budget=k,
                       order=pe["arm"].str.split("|", regex=False).str[0])
        pe_parts.append(pe)
        for nm in sorted({a.split("|")[0] for a in arms}):
            cols = [a for a in arms if a.startswith(nm + "|")]
            curve.append({"design": design, "order": nm, "budget": k,
                          "macro_mae": float(s.loc[cols, "macro_mae_extractant"].mean()),
                          "n_draws": len(cols), "n_flat_fallbacks": int(guard.get((nm, k), 0))})
        print(f"[{design}] budget {k} scored", flush=True)

    PE = pd.concat(pe_parts, ignore_index=True)
    keys = ["budget", "order", "split_seed", "extractant", "chemotype"]
    DA = PE.groupby(keys)[VALUE].mean()
    ABC = DA.groupby(level=["order", "split_seed", "extractant", "chemotype"]).mean().reset_index()
    C = paired_contrasts(ABC.rename(columns={"order": "arm"}), COMPS, value=VALUE)
    C.insert(0, "design", design)
    return pd.DataFrame(curve), C


def main() -> None:
    designs = sys.argv[1:] or ["BR", "BQ", "A"]
    curves, cons = [], []
    for d in designs:
        C, K = run(d)
        curves.append(C)
        cons.append(K)
    CU = pd.concat(curves, ignore_index=True)
    CO = pd.concat(cons, ignore_index=True)
    CU.to_csv(OUT / f"biggest_curve_{'_'.join(designs)}.csv", index=False)
    CO.to_csv(OUT / f"biggest_contrasts_{'_'.join(designs)}.csv", index=False)
    pd.set_option("display.width", 260)
    print("\n=== learning curves (macro MAE) ===")
    print(CU.pivot_table(index=["design", "order"], columns="budget",
                         values="macro_mae").round(4).to_string())
    print("\n=== FLAT-guard firings ===")
    print(CU.pivot_table(index=["design", "order"], columns="budget",
                         values="n_flat_fallbacks").astype(int).to_string())
    print("\n=== contrasts (ABC on mae_all) ===")
    print(CO[["design", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
              "seeds_positive", "loco_min", "loco_max", "loco_sign_stable", "passes_P1",
              "n_units"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
