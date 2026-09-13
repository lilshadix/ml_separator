"""REFUTER lens A for CLAIM L3A -- independent re-derivation and bookkeeping audit.

Written from PRE_REGISTRATION.md section 3 (L3a) text, NOT from gen16/l3_decision.py.
Only the frozen bench is imported (gen15.valuebench / gen15.arms / gen13sep.splits); the
task construction, the expected-draws arithmetic, the aggregation, the bootstrap, the
LOCO sweep and the permutation null are all re-written here.

Stage 1 caches the five design pair tables; every later stage is offline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen16 import bootstrap  # noqa: E402
from gen15 import valuebench as V  # noqa: E402
from gen15 import arms as A  # noqa: E402
from gen14.dirbench import load  # noqa: E402

OUT = bootstrap.RESULTS / "refutation" / "L3A" / "A"
OUT.mkdir(parents=True, exist_ok=True)
DESIGNS = ("B", "BR", "BQ", "A", "BP")
STRONG = 0.3      # pre-reg: "|log SF| >= 0.3"
MIN_EXT = 5       # pre-reg: "tasks with < 5 candidates are dropped and counted"


# ---------------------------------------------------------------- stage 1: pair tables
def cache_tables() -> None:
    bench = load()
    meta = bench.frame[["cell_id", "extractant", "chemotype", "n_metals", "publication_id"]].copy()
    meta.to_parquet(OUT / "cellmeta.parquet")
    for d in DESIGNS:
        p = OUT / f"pairs_{d}.parquet"
        if p.exists():
            continue
        t0 = time.time()
        tab = V.run_arms(bench, {"G14": A.g14}, d, verbose=False)
        tab.to_parquet(p)
        print(f"[cache] {d}: {len(tab)} rows in {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- my own task builder
def my_tasks(tab: pd.DataFrame):
    """(seed, unordered pair) -> candidate extractants, median observed y, median G14."""
    g = (tab.groupby(["split_seed", "A", "B", "extractant"], sort=True)[["y", "G14"]]
         .median().reset_index())
    tasks = []
    n_drop = 0
    for (seed, a, b), blk in g.groupby(["split_seed", "A", "B"], sort=True):
        blk = blk.sort_values("extractant")
        if len(blk) < MIN_EXT:
            n_drop += 1
            continue
        tasks.append({"seed": int(seed), "A": a, "B": b,
                      "ext": blk["extractant"].to_numpy(),
                      "obs": blk["y"].to_numpy(dtype=float),
                      "pred": blk["G14"].to_numpy(dtype=float)})
    return tasks, n_drop


def e_random(N, K):
    return (N + 1.0) / (K + 1.0)


def e_model(N, K, Nk, Kk):
    """kept set first; if it holds no success, exhaust it then random over the deferred set."""
    N, K, Nk, Kk = (np.asarray(x, float) for x in (N, K, Nk, Kk))
    return np.where(Kk > 0, (Nk + 1.0) / (Kk + 1.0), Nk + (N - Nk + 1.0) / (K - Kk + 1.0))


def saved_per_task(tasks, calls=None, weights=None):
    """calls: list of per-task call vectors (default sign(G14)).  weights: dict ext->multiplicity.
    Returns per-task (saved, e_random, e_model, valid)."""
    sv, er_, em_, ok = [], [], [], []
    for i, t in enumerate(tasks):
        obs = t["obs"]
        call = np.sign(t["pred"]).astype(int) if calls is None else calls[i]
        w = np.ones(len(obs)) if weights is None else np.array([weights.get(e, 0.0) for e in t["ext"]])
        N = w.sum()
        if N < MIN_EXT:
            sv.append(np.nan); er_.append(np.nan); em_.append(np.nan); ok.append(False); continue
        s_, r_, m_ = [], [], []
        for d in (-1, 1):
            succ = (np.sign(obs) == d) & (np.abs(obs) >= STRONG)
            kept = call != -d
            K = float((w * succ).sum())
            Nk = float((w * kept).sum())
            Kk = float((w * (kept & succ)).sum())
            er = e_random(N, K)
            em = float(e_model(N, K, Nk, Kk))
            s_.append(er - em); r_.append(er); m_.append(em)
        sv.append(0.5 * sum(s_)); er_.append(0.5 * sum(r_)); em_.append(0.5 * sum(m_)); ok.append(True)
    return np.array(sv), np.array(er_), np.array(em_), np.array(ok)


def seed_macro(v, seeds, sel=None):
    v = np.asarray(v, float).copy()
    if sel is not None:
        v = np.where(sel, v, np.nan)
    per = [np.nanmean(v[seeds == s]) if np.isfinite(v[seeds == s]).any() else np.nan
           for s in np.unique(seeds)]
    per = np.array(per, float)
    return float(np.nanmean(per))


def main() -> int:
    cache_tables()
    meta = pd.read_parquet(OUT / "cellmeta.parquet")
    chem_of = meta.drop_duplicates("extractant").set_index("extractant")["chemotype"].astype(str)
    nmet_of = meta.groupby("extractant")["n_metals"].max()
    res = {}

    for d in DESIGNS:
        tab = pd.read_parquet(OUT / f"pairs_{d}.parquet")
        tasks, n_drop = my_tasks(tab)
        seeds = np.array([t["seed"] for t in tasks])
        sv, er, em, ok = saved_per_task(tasks)
        res[d] = {
            "n_tasks": len(tasks), "n_dropped_lt5": n_drop,
            "n_candidate_slots": int(sum(len(t["ext"]) for t in tasks)),
            "median_candidates": float(np.median([len(t["ext"]) for t in tasks])),
            "tasks_per_seed": {int(s): int((seeds == s).sum()) for s in np.unique(seeds)},
            "e_random": seed_macro(er, seeds), "e_model": seed_macro(em, seeds),
            "saved": seed_macro(sv, seeds),
            "saved_plain_task_mean": float(np.nanmean(sv)),
            "saved_frac": seed_macro(sv, seeds) / seed_macro(er, seeds),
            "per_seed": {int(s): seed_macro(sv, seeds, seeds == s) for s in np.unique(seeds)},
        }
        # candidate-set fingerprint (byte-identical units across designs / arms)
        fp = pd.util.hash_pandas_object(
            pd.Series(["|".join([f"{t['seed']}:{t['A']}:{t['B']}"] + list(t["ext"])) for t in tasks]),
            index=False).sum()
        obs_fp = float(np.nansum(np.concatenate([t["obs"] for t in tasks])))
        res[d]["candidate_fingerprint"] = int(fp)
        res[d]["obs_sum"] = obs_fp
        print(f"[{d}] my saved = {res[d]['saved']:+.6f}  E_rand {res[d]['e_random']:.4f} "
              f"E_mod {res[d]['e_model']:.4f}  frac {res[d]['saved_frac']:.4f}  "
              f"tasks {len(tasks)} (dropped {n_drop})", flush=True)

    (OUT / "rederive.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
