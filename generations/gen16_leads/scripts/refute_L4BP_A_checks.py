"""Refuter lens A, step 3: bookkeeping checks that need no refit, plus the guard probe.

1. Are B/BR/BQ/BP scoring byte-identical held-out cells?  (If so the five-design disagreement is
   purely a training-composition effect -- the same thing the acquisition simulation manipulates.)
2. Benjamini-Hochberg recomputed independently over L4's registered family of 20.
3. n_metals stratification of the BP headline (scoring side).
4. Per-draw ABC spread in the refuter's own 6-draw RANDOM, against the lead's 20-draw sd.
5. FLAT-guard probe: what the RANDOM curve does at k=6/9 under BP if the guard is switched off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "generations" / "gen13_separation", ROOT / "generations" / "gen14_direction", ROOT / "generations" / "gen15_curve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame, cell_weights  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402
from gen13sep.splits import Fold, all_folds  # noqa: E402
from gen14.dirbench import feature_sets, load  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15.valuebench import Ctx, MIN_METALS  # noqa: E402

sys.path.insert(0, str(ROOT / "generations" / "gen16_leads" / "scripts"))
from refute_L4BP_A_rerun import greedy_aopt, predict_pairs, standardise, thin  # noqa: E402

OUT = ROOT / "generations" / "gen16_leads" / "results" / "refutation" / "L4BP" / "A"
LEAD = ROOT / "generations" / "gen16_leads" / "results" / "L4"
pd.set_option("display.width", 260)


def bh_independent(p: np.ndarray) -> np.ndarray:
    n = len(p)
    o = np.argsort(p, kind="mergesort")
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(adj, 1.0)
    return out


def main() -> None:
    bench = load()

    # ---- 1. test-index identity across designs --------------------------------------------
    idx = {}
    for d in ("B", "BR", "BQ", "BP", "A"):
        for f in all_folds(bench.frame, design=d):
            idx[(d, f.seed, f.fold)] = (f.test_index, f.train_index)
    keys = sorted({(s, k) for (_, s, k) in idx})
    same_test = all(np.array_equal(idx[("B", s, k)][0], idx[(d, s, k)][0])
                    for d in ("BR", "BQ", "BP") for s, k in keys)
    print(f"1. test sets identical across B/BR/BQ/BP: {same_test}")
    for d in ("B", "BR", "BQ", "BP", "A"):
        n = np.mean([len(idx[(d, s, k)][1]) for s, k in keys])
        print(f"   design {d}: mean training cells {n:.1f}")

    # ---- 2. BH over the registered family ---------------------------------------------------
    ca = pd.read_csv(LEAD / "contrasts_abc.csv")
    reg = ca[ca.family == "registered"].reset_index(drop=True)
    mine = bh_independent(reg.p_two_sided.to_numpy(float))
    print(f"\n2. BH over the registered family of {len(reg)}: "
          f"max |lead - refuter| = {np.abs(mine - reg.p_bh_within_lead_family.to_numpy(float)).max():.2e}")
    r = reg[(reg.design == "BP") & (reg.comparison == "AOPT_vs_RANDOM")]
    print(f"   BP AOPT_vs_RANDOM raw p {float(r.p_two_sided.iloc[0]):.4f}, "
          f"BH(reg, n=20) {mine[r.index[0]]:.4f}")
    expl = ca[ca.family == "exploratory"]
    all_p = pd.concat([ca.p_two_sided,
                       pd.read_csv(LEAD / "contrasts_per_budget.csv").p_two_sided,
                       pd.read_csv(LEAD / "contrasts_cellmatched.csv").p_two_sided,
                       pd.read_csv(LEAD / "contrasts_nodga.csv").p_two_sided]).to_numpy(float)
    print(f"   L4 wrote {len(all_p)} contrasts in total ({len(reg)} registered, {len(expl)} "
          f"exploratory in contrasts_abc alone)")

    # ---- 3. n_metals stratification of the BP headline ---------------------------------------
    ABC = pd.read_parquet(LEAD / "per_extractant_abc.parquet")
    nm = (bench.frame.groupby("extractant")["n_metals"].max().rename("n_metals_max"))
    bp = ABC[ABC.design == "BP"].merge(nm, left_on="extractant", right_index=True, how="left")
    cut = float(nm.median())
    print(f"\n3. n_metals confound (scoring side); median max n_metals per extractant = {cut}")
    for lab, sub in (("n_metals <= median", bp[bp.n_metals_max <= cut]),
                     ("n_metals >  median", bp[bp.n_metals_max > cut]),
                     ("well determined only (>=5)", bp[bp.n_metals_max >= MIN_METALS])):
        c = paired_contrasts(sub.rename(columns={"order": "arm"}),
                             {"AOPT_vs_RANDOM": ("RANDOM", "AOPT")}, value="mae_all")
        if len(c):
            print(f"   {lab:28s} point {c.point.iloc[0]:+.4f} "
                  f"[{c.ci95_low.iloc[0]:+.4f}, {c.ci95_high.iloc[0]:+.4f}] "
                  f"p {c.p_two_sided.iloc[0]:.4f} seeds+ {int(c.seeds_positive.iloc[0])} "
                  f"n_units {int(c.n_units.iloc[0])} P1 {bool(c.passes_P1.iloc[0])}")
    # partial: regress the per-extractant delta on n_metals
    per = (bp.pivot_table(index=["extractant", "n_metals_max"], columns="order",
                          values="mae_all").reset_index())
    delta = (per["RANDOM"] - per["AOPT"]).to_numpy(float)
    from scipy.stats import spearmanr
    ok = np.isfinite(delta)
    print(f"   Spearman(per-extractant AOPT gain, its n_metals) = "
          f"{spearmanr(delta[ok], per.n_metals_max.to_numpy(float)[ok]).statistic:+.3f} (n={ok.sum()})")

    # ---- 4. per-draw ABC spread in the refuter's RANDOM --------------------------------------
    for design in ("BP", "B"):
        f = OUT / f"rerun_abc_{design}.parquet"
        if not f.exists():
            continue
        R = pd.read_parquet(f)
        print(f"\n4. [{design}] refuter ABC (6 draws): "
              + ", ".join(f"{o} {v:.4f}" for o, v in
                          R.groupby("order")["mae_all"].mean().items()))

    # ---- 5. FLAT-guard probe under BP at k = 6 and 9 -----------------------------------------
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    topo = X[:, fs["TOPO39"]]
    groups = bench.groups.astype(str)
    RNG_BASE, N_DRAWS = 20260910, 6
    rows = []
    for f in all_folds(bench.frame, design="BP"):
        tr = f.train_index
        pairs = _pair_frame(bench.frame, bench.Y, f.test_index, f.seed, f.fold)
        if pairs.empty:
            continue
        g = groups[tr]
        chemos = sorted(set(g))
        Z = standardise(topo[tr], topo[tr])
        aopt = greedy_aopt(chemos, Z, g, rich[tr])
        base = pairs.drop(columns=["ia", "ib", "cell_local"])
        for k in (6, 9):
            t = base.copy()
            coef, _ = _fit_raw(bench, fs, X, rich, thin(f, bench.groups, set(aopt[:k])), "BP")
            t[f"AOPT|0"] = predict_pairs(coef, pairs, bench)
            for d in range(N_DRAWS):
                rng = np.random.default_rng(RNG_BASE + int(f.seed) * 97 + f.fold * 13 + d)
                p = [chemos[i] for i in rng.permutation(len(chemos))]
                coef, guarded = _fit_raw(bench, fs, X, rich, thin(f, bench.groups, set(p[:k])), "BP")
                t[f"RANDOM_NOGUARD|{d}"] = predict_pairs(coef, pairs, bench)
            t["budget"] = k
            rows.append(t)
    tab = pd.concat(rows, ignore_index=True)
    print("\n5. FLAT-guard probe (BP, refuter draws, guard OFF for RANDOM):")
    for k in (6, 9):
        sub = tab[tab.budget == k].drop(columns=["budget"])
        arms = [c for c in sub.columns if "|" in c]
        s = summarise(per_extractant(sub, arms), sub, arms).set_index("arm")
        rr = [a for a in arms if a.startswith("RANDOM_NOGUARD")]
        print(f"   k={k}: AOPT {s.loc['AOPT|0', 'macro_mae_extractant']:.4f}   "
              f"RANDOM guard-off {s.loc[rr, 'macro_mae_extractant'].mean():.4f}")


def _fit_raw(bench, fs, X, rich, fold: Fold, design: str):
    """G14 with NO FLAT guard; falls back only when the logistic genuinely cannot be fitted."""
    tr = fold.train_index
    ctx = Ctx(bench=bench, design=design, seed=fold.seed, fold=fold.fold, train=tr,
              test=fold.test_index, w=cell_weights(bench.groups[tr], bench.n_obs[tr]),
              model_seed=fold.model_seed, fs=fs, X=X, rich=rich)
    try:
        return A.g14(ctx), False
    except Exception:
        return A.flat(ctx), True


if __name__ == "__main__":
    main()
