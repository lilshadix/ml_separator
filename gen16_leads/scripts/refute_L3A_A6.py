"""REFUTER lens A, stage 6: independent chemotype-blocked interval (my own seed), a Monte-Carlo
check of the expected-draws arithmetic itself, and a median-collapse sensitivity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refute_L3A_A import OUT, DESIGNS, STRONG, MIN_EXT, my_tasks, e_random, e_model, seed_macro  # noqa: E402
from refute_L3A_A3 import saved_from_tasks  # noqa: E402


def mc_expected_draws(N, K, reps=200000, seed=7):
    """Simulate the policy itself: uniform random order, stop at the first success."""
    rng = np.random.default_rng(seed)
    succ = np.zeros(N, bool); succ[:K] = True
    tot = 0
    for _ in range(reps):
        order = rng.permutation(N)
        s = succ[order]
        tot += (np.argmax(s) + 1) if s.any() else N
    return tot / reps


def mc_e_model(N, K, kept_mask, succ_mask, reps=200000, seed=11):
    """Simulate: exhaust the kept set in random order, then the deferred set in random order."""
    rng = np.random.default_rng(seed)
    kept = np.flatnonzero(kept_mask); dfr = np.flatnonzero(~kept_mask)
    tot = 0
    for _ in range(reps):
        order = np.concatenate([rng.permutation(kept), rng.permutation(dfr)])
        s = succ_mask[order]
        tot += (np.argmax(s) + 1) if s.any() else len(order)
    return tot / reps


def main() -> int:
    meta = pd.read_parquet(OUT / "cellmeta.parquet")
    chem_of = meta.drop_duplicates("extractant").set_index("extractant")["chemotype"].astype(str).to_dict()
    res = {}

    # --- 1. is (N+1)/(K+1) the right policy cost?  and is the fall-through modelled right?
    mc = []
    rng = np.random.default_rng(3)
    for N, K in ((10, 1), (20, 3), (58, 7), (58, 0)):
        mc.append({"N": N, "K": K, "formula": float(e_random(N, K)),
                   "monte_carlo": mc_expected_draws(N, K, 60000)})
    for N, K, nk, kk in ((20, 3, 8, 2), (20, 3, 8, 0), (58, 7, 25, 1)):
        succ = np.zeros(N, bool); kept = np.zeros(N, bool)
        kept[:nk] = True
        succ[:kk] = True                      # kk successes inside the kept set
        succ[nk:nk + (K - kk)] = True         # the rest deferred
        mc.append({"N": N, "K": K, "Nk": nk, "Kk": kk,
                   "formula": float(e_model(N, K, nk, kk)),
                   "monte_carlo": mc_e_model(N, K, kept, succ, 60000)})
    res["expected_draws_monte_carlo"] = mc
    for r in mc:
        print("  MC", r, flush=True)

    # --- 2. my own chemotype-blocked interval (seed 20260914, not the lead's 8675309)
    blocks = sorted(set(chem_of.values()))
    bidx = {b: j for j, b in enumerate(blocks)}
    for d in DESIGNS:
        tab = pd.read_parquet(OUT / f"pairs_don_{d}.parquet")
        tasks, _ = my_tasks(tab)
        seeds = np.array([t["seed"] for t in tasks])
        calls = [np.sign(t["pred"]).astype(int) for t in tasks]
        point = saved_from_tasks(tasks, seeds, calls)[0]
        rg = np.random.default_rng(20260914)
        picks = rg.integers(0, len(blocks), size=(2000, len(blocks)))
        cnt = np.zeros((2000, len(blocks)))
        for j in range(len(blocks)):
            cnt[:, j] = (picks == j).sum(1)
        draws = np.full((2000, len(tasks)), np.nan)
        for i, t in enumerate(tasks):
            W = cnt[:, [bidx[chem_of[e]] for e in t["ext"]]]
            obs, call = t["obs"], calls[i]
            N = W.sum(1)
            tot = np.zeros(2000)
            for dd in (-1, 1):
                succ = (np.sign(obs) == dd) & (np.abs(obs) >= STRONG)
                kept = call != -dd
                tot += 0.5 * (e_random(N, W @ succ.astype(float))
                              - e_model(N, W @ succ.astype(float), W @ kept.astype(float),
                                        W @ (kept & succ).astype(float)))
            draws[:, i] = np.where(N >= MIN_EXT, tot, np.nan)
        bm = np.array([seed_macro(draws[r], seeds) for r in range(2000)])
        # --- 3. median-collapse sensitivity: majority vote of per-cell calls instead of sign(median)
        g = tab.groupby(["split_seed", "A", "B", "extractant"], sort=True)
        maj = (g["G14"].apply(lambda s: float(np.sign(np.sign(s).sum()))).rename("maj").reset_index())
        med = g[["y", "G14"]].median().reset_index().merge(maj, on=["split_seed", "A", "B", "extractant"])
        mtasks, mcalls = [], []
        for (s_, a, b), blk in med.groupby(["split_seed", "A", "B"], sort=True):
            if len(blk) < MIN_EXT:
                continue
            blk = blk.sort_values("extractant")
            mtasks.append({"seed": int(s_), "A": a, "B": b, "ext": blk["extractant"].to_numpy(),
                           "obs": blk["y"].to_numpy(float), "pred": blk["G14"].to_numpy(float)})
            mcalls.append(blk["maj"].to_numpy().astype(int))
        mseeds = np.array([t["seed"] for t in mtasks])
        maj_saved = saved_from_tasks(mtasks, mseeds, mcalls)[0]
        res[d] = {"point": point,
                  "my_block_ci95": [float(np.nanquantile(bm, .025)), float(np.nanquantile(bm, .975))],
                  "my_block_p_two_sided": float(min(1.0, 2 * min((bm <= 0).mean(), (bm >= 0).mean()))),
                  "saved_majority_vote_call": maj_saved,
                  "n_tasks": len(tasks), "n_distinct_pairs": 91, "n_seeds": 5,
                  "n_extractants": len({e for t in tasks for e in t["ext"]}),
                  "n_chemotypes": len({chem_of[e] for t in tasks for e in t["ext"]})}
        print(f"[{d}] point {point:+.6f} | my blocked CI [{res[d]['my_block_ci95'][0]:+.4f}, "
              f"{res[d]['my_block_ci95'][1]:+.4f}] p {res[d]['my_block_p_two_sided']:.4f} | "
              f"majority-vote call {maj_saved:+.4f}", flush=True)
    (OUT / "checks_stage6.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
