"""Refuter lens A, step 2: independent re-derivation of L4's headline plus size-matched controls.

Written from PRE_REGISTRATION.md section 3 (L4), NOT from gen16/l4_acq.py.  The frozen bench
(gen13sep / gen14 / gen15) is imported; the thinning, the A-optimal criterion, the orders and the
ABC endpoint are re-implemented here so a bookkeeping error in the lead's module would show up as a
disagreement.

Orders scored
    AOPT_R    greedy A-optimal on standardised TOPO39, ridge penalty 1, precision accumulated over
              the *well-determined* (>= 5 metals) cells of the chosen chemotypes  [the lead's reading]
    AOPT_ALL  the same with precision accumulated over *every* chosen training cell
              [the other reading of the pre-registration, which does not say "well determined"]
    BIGGEST   chemotypes in descending well-determined-cell count -- pure size, zero chemistry
    DGAFIRST  the single largest chemotype first, then a uniform random order  (N_DRAWS draws)
    RANDOM    uniform random order (N_DRAWS draws), refuter's own RNG stream
    RANDCM    the SAME random permutations, but the budget is AOPT_R's training-CELL count at k
              rather than k chemotypes -- RANDOM given the same number of cells at every rung

Endpoint: ABC = mean over k in (6,9,12,16,20,24) of [MAE_ref(k) - MAE_cand(k)] per extractant,
stochastic orders averaged over draws; gen13sep.inference.paired_contrasts, margin 0.02.

    PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2 .venv/Scripts/python.exe \
        gen16_leads/scripts/refute_L4BP_A_rerun.py BP
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gen13sep.amplitude_bench import LEAN_BLOCKS, _pair_frame, cell_weights  # noqa: E402
from gen13sep.inference import paired_contrasts  # noqa: E402
from gen13sep.metrics import per_extractant, summarise  # noqa: E402
from gen13sep.splits import Fold, all_folds  # noqa: E402
from gen14.dirbench import feature_sets, load  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen15.valuebench import Ctx, MIN_METALS  # noqa: E402

OUT = ROOT / "gen16_leads" / "results" / "refutation" / "L4BP" / "A"
OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = (6, 9, 12, 16, 20, 24)
N_DRAWS = 6
LAM = 1.0
MIN_RICH_FIT = 10
VALUE_COLS = ["mae_all", "mae_far", "sign_acc_strong", "pair_spearman"]
RNG_BASE = 20260910          # refuter's own stream: NOT the lead's model_seed*1000 + d


# ---------------------------------------------------------------- standardise / criterion
def standardise(X_pool: np.ndarray, X: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        med = np.nanmedian(X_pool, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Zp = np.where(np.isfinite(X_pool), X_pool, med)
    mu, sd = Zp.mean(axis=0), Zp.std(axis=0)
    const = ~(sd > 1e-12)
    Z = np.where(np.isfinite(X), X, med)
    Z = (Z - mu) / np.where(const, 1.0, sd)
    Z[:, const] = 0.0
    return Z


def greedy_aopt(chemos: list[str], Z: np.ndarray, g: np.ndarray, use: np.ndarray) -> list[str]:
    """Add the chemotype minimising sum_i z_i^T (lam I + sum_chosen z z^T)^-1 z_i over the pool."""
    d = Z.shape[1]
    P = LAM * np.eye(d)
    mem = {c: Z[(g == c) & use] for c in chemos}
    chosen, rest = [], sorted(chemos)
    while rest:
        best, bs = None, np.inf
        for c in rest:
            M = mem[c]
            Pn = P + M.T @ M if len(M) else P
            s = float(np.einsum("ij,jk,ik->", Z, np.linalg.inv(Pn), Z))
            if s < bs - 1e-12:
                best, bs = c, s
        chosen.append(best)
        rest.remove(best)
        if len(mem[best]):
            P = P + mem[best].T @ mem[best]
    return chosen


# ---------------------------------------------------------------- fold machinery
def thin(fold: Fold, groups: np.ndarray, keep: set[str]) -> Fold:
    tr = fold.train_index
    sub = tr[np.isin(groups[tr].astype(str), list(keep))]
    return Fold(design=fold.design, seed=fold.seed, fold=fold.fold, train_index=sub,
                test_index=fold.test_index, inner_train_index=sub,
                inner_validation_index=sub[:0], held_out_groups=fold.held_out_groups)


def fit(bench, fs, X, rich, fold: Fold, design: str) -> tuple[np.ndarray, bool]:
    tr = fold.train_index
    ctx = Ctx(bench=bench, design=design, seed=fold.seed, fold=fold.fold, train=tr,
              test=fold.test_index, w=cell_weights(bench.groups[tr], bench.n_obs[tr]),
              model_seed=fold.model_seed, fs=fs, X=X, rich=rich)
    rtr = ctx.rich_train()
    if len(rtr) < MIN_RICH_FIT or len(set((ctx.amp[rtr] < 0).tolist())) < 2:
        return A.flat(ctx), True
    return A.g14(ctx), False


def predict_pairs(coef: np.ndarray, pairs: pd.DataFrame, bench) -> np.ndarray:
    curve = np.asarray(coef, float).reshape(-1, bench.basis.shape[0]) @ bench.basis
    loc = pairs["cell_local"].to_numpy()
    return curve[loc, pairs["ia"].to_numpy()] - curve[loc, pairs["ib"].to_numpy()]


# ---------------------------------------------------------------- main
def run(design: str) -> None:
    bench = load()
    fs = feature_sets(bench)
    X = bench.matrix(LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= MIN_METALS
    topo_all = X[:, fs["TOPO39"]]
    groups = bench.groups.astype(str)

    folds = all_folds(bench.frame, design=design)
    parts = {k: [] for k in BUDGETS}
    guard, order_rows, agree = {}, [], []
    t0 = time.time()
    for f in folds:
        tr = f.train_index
        pairs = _pair_frame(bench.frame, bench.Y, f.test_index, f.seed, f.fold)
        if pairs.empty:
            continue
        g = groups[tr]
        chemos = sorted(set(g))
        Z = standardise(topo_all[tr], topo_all[tr])
        size_rich = pd.Series(g[rich[tr]]).value_counts()
        size_all = pd.Series(g).value_counts()

        aopt_r = greedy_aopt(chemos, Z, g, rich[tr])
        aopt_all = greedy_aopt(chemos, Z, g, np.ones(len(tr), bool))
        biggest = sorted(chemos, key=lambda c: (-int(size_rich.get(c, 0)), c))
        top1 = str(size_all.sort_values(ascending=False).index[0])

        # AOPT_R's training-cell count at every budget -> the cell target for RANDCM
        cum = np.cumsum([int(size_all.get(c, 0)) for c in aopt_r])
        targets = {k: int(cum[k - 1]) for k in BUDGETS}

        orders: dict[tuple[str, int], list[str]] = {("AOPT_R", 0): aopt_r,
                                                    ("AOPT_ALL", 0): aopt_all,
                                                    ("BIGGEST", 0): biggest}
        rand_perms = []
        for d in range(N_DRAWS):
            rng = np.random.default_rng(RNG_BASE + int(f.seed) * 97 + f.fold * 13 + d)
            p = [chemos[i] for i in rng.permutation(len(chemos))]
            rand_perms.append(p)
            orders[("RANDOM", d)] = p
            rest = [c for c in p if c != top1]
            orders[("DGAFIRST", d)] = [top1] + rest
        for key, o in orders.items():
            assert sorted(o) == chemos
            for pos, c in enumerate(o):
                order_rows.append({"design": design, "seed": f.seed, "fold": f.fold,
                                   "order": key[0], "draw": key[1], "position": pos, "chemotype": c})

        base = pairs.drop(columns=["ia", "ib", "cell_local"])
        for k in BUDGETS:
            t = base.copy()
            for (nm, d), o in orders.items():
                coef, fb = fit(bench, fs, X, rich, thin(f, bench.groups, set(o[:k])), design)
                t[f"{nm}|{d}"] = predict_pairs(coef, pairs, bench)
                guard[(nm, k)] = guard.get((nm, k), 0) + int(fb)
            # RANDCM: same permutations, budget = AOPT_R's cell count at k
            for d, p in enumerate(rand_perms):
                c_cum = np.cumsum([int(size_all.get(c, 0)) for c in p])
                j = int(np.searchsorted(c_cum, targets[k], side="left")) + 1
                j = min(j, len(p))
                coef, fb = fit(bench, fs, X, rich, thin(f, bench.groups, set(p[:j])), design)
                t[f"RANDCM|{d}"] = predict_pairs(coef, pairs, bench)
                guard[("RANDCM", k)] = guard.get(("RANDCM", k), 0) + int(fb)
                order_rows.append({"design": design, "seed": f.seed, "fold": f.fold,
                                   "order": "RANDCM_nchem", "draw": d, "position": k,
                                   "chemotype": str(j)})
            parts[k].append(t)
    print(f"[{design}] fits done in {time.time() - t0:.0f}s", flush=True)

    pe_parts, curve = [], []
    for k, plist in parts.items():
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
            curve.append({"design": design, "order": nm, "budget": k,
                          "macro_mae": float(s.loc[cols, "macro_mae_extractant"].mean()),
                          "n_draws": len(cols),
                          "n_flat_fallbacks": int(guard.get((nm, k), 0))})
        print(f"[{design}] budget {k} scored ({len(arms)} arms)", flush=True)

    PE = pd.concat(pe_parts, ignore_index=True)
    C = pd.DataFrame(curve)
    C.to_csv(OUT / f"rerun_curve_{design}.csv", index=False)
    pd.DataFrame(order_rows).to_parquet(OUT / f"rerun_orders_{design}.parquet", index=False)

    keys = ["design", "budget", "order", "split_seed", "extractant", "chemotype"]
    DA = PE.groupby(keys)[VALUE_COLS].mean().reset_index()
    ABC = (DA.groupby(["design", "order", "split_seed", "extractant", "chemotype"])[VALUE_COLS]
           .mean().reset_index())
    ABC.to_parquet(OUT / f"rerun_abc_{design}.parquet", index=False)

    comps = {
        "AOPTR_vs_RANDOM": ("RANDOM", "AOPT_R"),
        "AOPTALL_vs_RANDOM": ("RANDOM", "AOPT_ALL"),
        "BIGGEST_vs_RANDOM": ("RANDOM", "BIGGEST"),
        "DGAFIRST_vs_RANDOM": ("RANDOM", "DGAFIRST"),
        "AOPTR_vs_BIGGEST": ("BIGGEST", "AOPT_R"),
        "AOPTR_vs_DGAFIRST": ("DGAFIRST", "AOPT_R"),
        "AOPTR_vs_RANDCM": ("RANDCM", "AOPT_R"),
        "BIGGEST_vs_RANDCM": ("RANDCM", "BIGGEST"),
        "RANDCM_vs_RANDOM": ("RANDOM", "RANDCM"),
    }
    pe = ABC.rename(columns={"order": "arm"})
    rows = [paired_contrasts(pe, comps, value="mae_all").assign(design=design, endpoint="ABC_mae_all")]
    # the same, with the 23 diglycolamide (sc009) extractants dropped from the SCORING units
    pe_nod = pe[pe.chemotype != "sc009"]
    rows.append(paired_contrasts(pe_nod, comps, value="mae_all")
                .assign(design=design, endpoint="ABC_mae_all_noDGAscored"))
    CT = pd.concat(rows, ignore_index=True)
    CT.to_csv(OUT / f"rerun_contrasts_{design}.csv", index=False)

    pd.set_option("display.width", 260)
    print("\n=== learning curves (macro MAE) ===")
    print(C.pivot_table(index="order", columns="budget", values="macro_mae").round(4).to_string())
    print("\n=== FLAT-guard firings ===")
    print(C.pivot_table(index="order", columns="budget", values="n_flat_fallbacks").astype(int).to_string())
    print("\n=== contrasts (ABC on mae_all) ===")
    print(CT[["endpoint", "comparison", "point", "ci95_low", "ci95_high", "p_two_sided",
              "seeds_positive", "loco_min", "loco_max", "loco_sign_stable", "passes_P1",
              "n_units", "block_macro"]].round(4).to_string(index=False))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BP")
