"""REFUTER lens A for CLAIM L3A -- stage 2: the mandatory checks, all offline on the
cached pair tables written by refute_L3A_A.py.  Nothing here imports the lead's code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refute_L3A_A import (OUT, DESIGNS, STRONG, MIN_EXT, my_tasks, e_random, e_model,
                          saved_per_task, seed_macro)  # noqa: E402


def calls_of(tasks):
    return [np.sign(t["pred"]).astype(int) for t in tasks]


def saved_weighted(tasks, seeds, calls, w_of_ext, sel=None):
    """seed-macro saved for a per-extractant multiplicity map."""
    sv, er, em, ok = [], [], [], []
    for i, t in enumerate(tasks):
        obs, call = t["obs"], calls[i]
        w = np.array([w_of_ext.get(e, 0.0) for e in t["ext"]], float)
        N = w.sum()
        if N < MIN_EXT:
            sv.append(np.nan); er.append(np.nan); em.append(np.nan); continue
        s_, r_, m_ = 0.0, 0.0, 0.0
        for d in (-1, 1):
            succ = (np.sign(obs) == d) & (np.abs(obs) >= STRONG)
            kept = call != -d
            K, Nk, Kk = (w * succ).sum(), (w * kept).sum(), (w * (kept & succ)).sum()
            r = e_random(N, K); m = float(e_model(N, K, Nk, Kk))
            s_ += 0.5 * (r - m); r_ += 0.5 * r; m_ += 0.5 * m
        sv.append(s_); er.append(r_); em.append(m_)
    sv, er, em = np.array(sv), np.array(er), np.array(em)
    return seed_macro(sv, seeds, sel), seed_macro(er, seeds, sel), seed_macro(em, seeds, sel)


def perm_null(tasks, seeds, calls, reps, rng, strata=None):
    """Permute the calls within each task (optionally within strata) -> (reps,) seed-macro saved."""
    n_t = len(tasks)
    draws = np.zeros((reps, n_t))
    for i, t in enumerate(tasks):
        obs, call = t["obs"], calls[i]
        n = len(obs)
        if strata is None:
            perm = np.argsort(rng.random((reps, n)), axis=1)
            cp = call[perm]
        else:
            st = strata[i]
            cp = np.tile(call, (reps, 1))
            for s in np.unique(st):
                idx = np.flatnonzero(st == s)
                if len(idx) < 2:
                    continue
                p = idx[np.argsort(rng.random((reps, len(idx))), axis=1)]
                cp[:, idx] = call[p]
        tot = np.zeros(reps)
        for d in (-1, 1):
            succ = (np.sign(obs) == d) & (np.abs(obs) >= STRONG)
            K = int(succ.sum())
            kept = cp != -d
            Nk = kept.sum(1); Kk = (kept & succ[None, :]).sum(1)
            tot += 0.5 * (e_random(n, K) - e_model(n, K, Nk, Kk))
        draws[:, i] = tot
    return np.array([seed_macro(draws[r], seeds) for r in range(reps)])


def main() -> int:
    meta = pd.read_parquet(OUT / "cellmeta.parquet")
    chem_of = meta.drop_duplicates("extractant").set_index("extractant")["chemotype"].astype(str).to_dict()
    nmet_of = meta.groupby("extractant")["n_metals"].max().to_dict()
    res = {}
    rng_master = np.random.default_rng(20260911)   # deliberately NOT the lead's 8675309

    for d in DESIGNS:
        tab = pd.read_parquet(OUT / f"pairs_{d}.parquet")
        tasks, _ = my_tasks(tab)
        seeds = np.array([t["seed"] for t in tasks])
        calls = calls_of(tasks)
        allext = sorted({e for t in tasks for e in t["ext"]})
        one = {e: 1.0 for e in allext}
        R = {}
        base = saved_weighted(tasks, seeds, calls, one)
        R["saved_all"] = base[0]

        # ---- 1. LOCO over chemotypes (diglycolamide check)
        loco = {}
        for c in sorted(set(chem_of[e] for e in allext)):
            w = {e: (0.0 if chem_of[e] == c else 1.0) for e in allext}
            loco[c] = saved_weighted(tasks, seeds, calls, w)[0]
        R["loco"] = loco
        R["loco_min"] = float(np.nanmin(list(loco.values())))
        R["loco_max"] = float(np.nanmax(list(loco.values())))
        R["loco_argmin"] = min(loco, key=lambda k: loco[k])
        R["saved_without_sc009"] = loco.get("sc009", None)
        R["n_ext_sc009"] = int(sum(1 for e in allext if chem_of[e] == "sc009"))
        # sc009 only
        w = {e: (1.0 if chem_of[e] == "sc009" else 0.0) for e in allext}
        sv_only = saved_weighted(tasks, seeds, calls, w)
        R["saved_sc009_only"] = sv_only[0]
        R["e_random_sc009_only"] = sv_only[1]

        # ---- 2. n_metals confound
        nm = {e: float(nmet_of[e]) for e in allext}
        strat_nm = [np.digitize([nm[e] for e in t["ext"]], [5, 8, 11]) for t in tasks]
        strat_ch = [np.array([chem_of[e] for e in t["ext"]]) for t in tasks]
        # correlation of the call with n_metals, and of success with n_metals, pooled
        cc, nn, ss = [], [], []
        for i, t in enumerate(tasks):
            cc.append(calls[i]); nn.append([nm[e] for e in t["ext"]])
            ss.append(((np.abs(t["obs"]) >= STRONG) & (np.sign(t["obs"]) == -1)).astype(float))
        cc = np.concatenate(cc); nn = np.concatenate(nn); ss = np.concatenate(ss)
        from scipy.stats import spearmanr
        R["spearman_call_nmetals"] = float(spearmanr(cc, nn).statistic)
        R["spearman_heavysuccess_nmetals"] = float(spearmanr(ss, nn).statistic)
        # n_metals-stratified saving: split candidates by median n_metals is not meaningful;
        # instead restrict candidate sets to one n_metals band at a time
        for lo, hi, lab in ((0, 4, "nm<=4"), (5, 7, "nm5-7"), (8, 10, "nm8-10"), (11, 99, "nm>=11")):
            w = {e: (1.0 if lo <= nm[e] <= hi else 0.0) for e in allext}
            R[f"saved_{lab}"] = saved_weighted(tasks, seeds, calls, w)[0]
            R[f"next_{lab}"] = int(sum(1 for e in allext if lo <= nm[e] <= hi))

        # ---- 4/5. matched competitors and nulls
        heavy = [np.full(len(t["obs"]), -1, int) for t in tasks]
        light = [np.full(len(t["obs"]), 1, int) for t in tasks]
        R["saved_HEAVIER_ALWAYS"] = saved_weighted(tasks, seeds, heavy, one)[0]
        R["saved_LIGHTER_ALWAYS"] = saved_weighted(tasks, seeds, light, one)[0]
        reps = 2000
        nulls = perm_null(tasks, seeds, calls, reps, np.random.default_rng(int(rng_master.integers(1e9))))
        R["perm_free"] = {"mean": float(nulls.mean()), "sd": float(nulls.std(ddof=1)),
                          "q025": float(np.quantile(nulls, .025)), "q975": float(np.quantile(nulls, .975)),
                          "max": float(nulls.max()),
                          "p": float((nulls >= R["saved_all"]).mean()),
                          "real_minus_shuffled": R["saved_all"] - float(nulls.mean())}
        n_nm = perm_null(tasks, seeds, calls, 500, np.random.default_rng(int(rng_master.integers(1e9))),
                         strata=strat_nm)
        R["perm_nmetals_stratified"] = {"mean": float(n_nm.mean()), "sd": float(n_nm.std(ddof=1)),
                                        "max": float(n_nm.max()),
                                        "p": float((n_nm >= R["saved_all"]).mean()),
                                        "real_minus_shuffled": R["saved_all"] - float(n_nm.mean())}
        n_ch = perm_null(tasks, seeds, calls, 500, np.random.default_rng(int(rng_master.integers(1e9))),
                         strata=strat_ch)
        R["perm_chemotype_stratified"] = {"mean": float(n_ch.mean()), "sd": float(n_ch.std(ddof=1)),
                                          "max": float(n_ch.max()),
                                          "p": float((n_ch >= R["saved_all"]).mean()),
                                          "real_minus_shuffled": R["saved_all"] - float(n_ch.mean())}

        # ---- 6. bookkeeping: identical candidate sets across seeds?  one bit per extractant?
        sig = {}
        for t in tasks:
            sig.setdefault((t["A"], t["B"]), []).append((tuple(t["ext"]), tuple(np.round(t["obs"], 12))))
        R["candidate_sets_identical_across_seeds"] = bool(all(len(set(v)) == 1 for v in sig.values()))
        R["n_distinct_pairs"] = len(sig)
        # how many distinct direction calls does one extractant get across the 91 pairs of a seed?
        per_es = {}
        for i, t in enumerate(tasks):
            for j, e in enumerate(t["ext"]):
                per_es.setdefault((t["seed"], e), set()).add(int(calls[i][j]))
        R["frac_extractant_seed_single_call"] = float(np.mean([len(v) == 1 for v in per_es.values()]))
        per_e = {}
        for (s, e), v in per_es.items():
            per_e.setdefault(e, set()).update(v)
        R["frac_extractant_single_call_across_seeds"] = float(np.mean([len(v) == 1 for v in per_e.values()]))
        R["n_extractants"] = len(allext)
        res[d] = R
        print(f"[{d}] saved {R['saved_all']:+.4f} | no-sc009 {R['saved_without_sc009']:+.4f} "
              f"| sc009-only {R['saved_sc009_only']:+.4f} | perm mean {R['perm_free']['mean']:+.4f} "
              f"p {R['perm_free']['p']:.4f} | real-shuf {R['perm_free']['real_minus_shuffled']:+.4f} "
              f"| nm-strat mean {R['perm_nmetals_stratified']['mean']:+.4f} "
              f"| chem-strat mean {R['perm_chemotype_stratified']['mean']:+.4f}", flush=True)

    (OUT / "checks_stage2.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
