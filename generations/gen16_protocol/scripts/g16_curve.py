"""Is the limit the model or the corpus?  A learning curve over *chemotypes*, under BP.

GEN14_REPORT asserts that the binding constraint is data.  Nothing in the repository tests it.
This does: for every BP fold the training set is thinned to k randomly chosen chemotypes (the test
set is untouched, so every budget is scored on byte-identical held-out pairs), the deployed arm is
refitted, and the macro MAE is plotted against k.

Three things are read off the curve.
  * the *slope at the full corpus*, tested per extractant with the restricted wild cluster
    bootstrap-t over chemotypes -- if the full budget is not reliably better than the next one
    down, the curve has flattened and "collect more chemotypes" is not supported;
  * a power law MAE = c + a k^-b fitted to the mean curve, whose asymptote c is the best guess at
    what an infinitely diverse corpus of this kind would buy (Viering & Loog, IEEE TPAMI
    45:7799-7819, 2023, on why the power law is the right default shape and where it fails);
  * the noise floor next to both: the replicate sd of the amplitude is 0.237, so any asymptote
    near the oracle-amplitude arm is measuring the corpus, not the model.

Usage:  python gen16_protocol/scripts/g16_curve.py [draws]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "gen13_separation", ROOT / "gen14_direction",
          ROOT / "gen15_curve", ROOT / "gen16_protocol"):
    sys.path.insert(0, str(p))

from gen13sep.amplitude_bench import LEAN_BLOCKS            # noqa: E402
from gen13sep.metrics import per_extractant, summarise      # noqa: E402
from gen13sep.splits import all_folds                       # noqa: E402
from gen14.dirbench import load, feature_sets               # noqa: E402
from gen15 import arms as A                                 # noqa: E402
from gen16.clusterboot import wcr_test                      # noqa: E402
from gen16.designs import run_folds, thin_folds             # noqa: E402

DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BUDGETS = (6, 9, 12, 16, 20, 24, 28, 32, None)      # None = every training chemotype
ARMS = {"G14": A.g14, "MEAN_CURVE": A.mean_curve}   # both are seconds-cheap; no tree arm here
OUT = ROOT / "gen16_protocol" / "results"
OUT.mkdir(parents=True, exist_ok=True)

bench = load()
X = bench.matrix(LEAN_BLOCKS)
fs = feature_sets(bench)
folds = all_folds(bench.frame, design="BP")
full_k = int(np.mean([len(set(bench.groups[f.train_index])) for f in folds]))
print(f"BP: {len(folds)} folds, mean {full_k} training chemotypes, "
      f"mean {int(np.mean([len(f.train_index) for f in folds]))} training cells")

rows, pe_rows = [], []
t0 = time.time()
for k in BUDGETS:
    n_draws = 1 if k is None else DRAWS
    for d in range(n_draws):
        th = [thin_folds(f, bench.groups, k if k is not None else 10 ** 6, d) for f in folds]
        th = [f for f in th if f is not None]
        tab = run_folds(bench, ARMS, th, design="BP", X=X, fs=fs)
        if tab.empty:
            continue
        pe = per_extractant(tab, list(ARMS))
        s = summarise(pe, tab, list(ARMS))
        kk = float(tab.n_train_chemotypes.mean())
        pe = pe.assign(budget=(k if k is not None else full_k), draw=d, k_actual=kk)
        pe_rows.append(pe)
        for _, r in s.iterrows():
            rows.append({"budget": (k if k is not None else full_k), "draw": d,
                         "k_actual": kk, "n_cells": float(tab.n_train_cells.mean()),
                         "arm": r["arm"], "macro_mae": r["macro_mae_extractant"],
                         "macro_mae_far": r["macro_mae_far"],
                         "sign_acc": r["macro_sign_acc_strong"]})
    print(f"  budget {k} done ({time.time() - t0:.0f}s)", flush=True)

curve = pd.DataFrame(rows)
curve.to_csv(OUT / "g16_curve_raw.csv", index=False)
PE = pd.concat(pe_rows, ignore_index=True)
PE.to_parquet(OUT / "g16_curve_per_extractant.parquet", index=False)

agg = (curve.groupby(["arm", "budget"])
            .agg(k_actual=("k_actual", "mean"), n_cells=("n_cells", "mean"),
                 macro_mae=("macro_mae", "mean"), draw_sd=("macro_mae", "std"),
                 macro_mae_far=("macro_mae_far", "mean"),
                 sign_acc=("sign_acc", "mean")).reset_index())

# ---- chemotype cluster bootstrap band on the level, at each budget --------------------
rng = np.random.default_rng(20260909)
bands = []
for (arm, budget), blk in PE.groupby(["arm", "budget"]):
    u = blk.groupby(["extractant", "chemotype"])["mae_all"].mean().reset_index()
    groups = [g["mae_all"].to_numpy() for _, g in u.groupby("chemotype")]
    G = len(groups)
    draws = np.array([np.concatenate([groups[i] for i in rng.integers(0, G, G)]).mean()
                      for _ in range(4000)])
    bands.append({"arm": arm, "budget": budget, "boot_low": float(np.percentile(draws, 2.5)),
                  "boot_high": float(np.percentile(draws, 97.5)), "G": G})
agg = agg.merge(pd.DataFrame(bands), on=["arm", "budget"], how="left")
agg.to_csv(OUT / "g16_curve.csv", index=False)
print("\n=== learning curve over chemotypes, design BP ===")
print(agg.round(4).to_string(index=False))

# ---- is the curve still descending at the full corpus? -------------------------------
print("\n=== paired test: full budget vs the next one down (wild cluster bootstrap-t) ===")
tests = []
for arm in ARMS:
    sub = PE[PE.arm == arm]
    top = sub[sub.budget == full_k].groupby(["extractant", "chemotype"])["mae_all"].mean()
    for k in (BUDGETS[-2], BUDGETS[-3]):
        lo = sub[sub.budget == k].groupby(["extractant", "chemotype"])["mae_all"].mean()
        j = pd.concat([lo.rename("lo"), top.rename("hi")], axis=1).dropna().reset_index()
        d = (j["lo"] - j["hi"]).to_numpy()            # positive = full budget is better
        r = wcr_test(d, j["chemotype"].to_numpy(), comparison=f"{arm}: k={k} -> full", reps=9_999)
        tests.append({"arm": arm, "from_k": k, "to_k": full_k, "gain": r.point,
                      "se_cv3": r.se_cv3, "p_wcr": r.p_wcr, "ci_low": r.ci_low,
                      "ci_high": r.ci_high, "G": r.G, "G_star": r.G_star})
T = pd.DataFrame(tests)
T.to_csv(OUT / "g16_curve_slope_test.csv", index=False)
print(T.round(4).to_string(index=False))

# ---- power-law extrapolation, with a chemotype bootstrap on the asymptote -------------
from scipy.optimize import curve_fit                                        # noqa: E402


def _fit(xs, ys):
    p, _ = curve_fit(lambda n, c, a, b: c + a * n ** (-b), xs, ys,
                     p0=[float(ys.min()) * 0.8, 1.0, 0.5],
                     bounds=([0.0, 0.0, 0.05], [1.5, 50.0, 3.0]), maxfev=40000)
    return p


print("\n=== power-law extrapolation MAE = c + a k^-b ===")
laws = []
for arm in ARMS:
    a = agg[agg.arm == arm].sort_values("k_actual")
    try:
        c, aa, b = _fit(a.k_actual.to_numpy(), a.macro_mae.to_numpy())
    except Exception as e:                                                  # noqa: BLE001
        print(f"  {arm}: fit failed ({e})")
        continue
    boot_c = []
    for _ in range(500):
        y = a.macro_mae.to_numpy() + rng.normal(0, np.nan_to_num(a.draw_sd.to_numpy(), nan=0.005))
        try:
            boot_c.append(_fit(a.k_actual.to_numpy(), y)[0])
        except Exception:                                                   # noqa: BLE001
            pass
    lo, hi = (np.percentile(boot_c, [2.5, 97.5]) if boot_c else (np.nan, np.nan))
    laws.append({"arm": arm, "c_asymptote": c, "c_low": lo, "c_high": hi, "a": aa, "b": b,
                 **{f"pred_{n}": c + aa * n ** (-b) for n in (45, 60, 90, 135, 200)}})
    print(f"  {arm}: MAE = {c:.3f} + {aa:.3f} k^-{b:.3f};  asymptote {c:.3f} "
          f"[{lo:.3f}, {hi:.3f}];  90 chemotypes -> {c + aa * 90 ** (-b):.3f}")
pd.DataFrame(laws).to_csv(OUT / "g16_curve_powerlaw.csv", index=False)
print("\nNoise floor: replicate sd of the amplitude = 0.237; oracle-amplitude arm = 0.322 under BP.")
