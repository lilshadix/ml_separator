"""Precompute TabPFN v2 out-of-fold predictions for every fold of every design.

TabPFN's forward pass costs ~20 s per fold on this CPU, and the four heads this experiment needs
(direction on TOPO39, direction on LEAN209, log magnitude, curvature) would otherwise be refitted
once per arm.  This script walks the fold plan once, caches the four predictions per fold to disk,
and lets the scoring run read them.  It checkpoints after every fold, so a partial run is still
usable and a restart resumes.

Nothing here sees a held-out label: for every fold the model is fitted on the rich (>= 5 metals)
training cells only, resampled to the chemotype-balanced weights (TabPFN takes no ``sample_weight``),
median-imputed on those same rows.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K                                   # noqa: E402
from common import V                                 # noqa: E402
import tabpfn_compat                                 # noqa: E402

from gen13sep.amplitude_bench import cell_weights    # noqa: E402
from gen13sep.splits import all_folds                # noqa: E402
from gen14.dirbench import feature_sets              # noqa: E402

torch.set_num_threads(2)

CACHE = K.OUT / "tp_oof.pkl"
N_EST = 2          # fixed a priori; the CPU forward pass is dominated by fixed cost, not by this
DESIGNS = V.DESIGNS


def _fit_predict(Cls, Reg, Xtr, Xte, y_dir, y_logmag, y_curv, seed, want):
    out = {}
    if "p" in want:
        m = Cls(device="cpu", n_estimators=N_EST, random_state=seed, fit_mode="low_memory")
        m.fit(Xtr, y_dir)
        out["p"] = np.asarray(m.predict_proba(Xte))[:, list(m.classes_).index(1)]
    if "mag" in want:
        m = Reg(device="cpu", n_estimators=N_EST, random_state=seed, fit_mode="low_memory")
        m.fit(Xtr, y_logmag)
        out["mag"] = np.clip(np.exp(np.asarray(m.predict(Xte))) - K.EPS, 0.0, None)
    if "curv" in want:
        m = Reg(device="cpu", n_estimators=N_EST, random_state=seed, fit_mode="low_memory")
        m.fit(Xtr, y_curv)
        out["curv"] = np.asarray(m.predict(Xte), dtype=float)
    return out


def main() -> None:
    Cls, Reg = tabpfn_compat.load()
    bench = V.load()
    fs = feature_sets(bench)
    X = bench.matrix(K.LEAN_BLOCKS)
    rich = bench.frame.n_metals.to_numpy() >= K.MIN_METALS
    amp, cur = bench.coef[:, 0], bench.coef[:, 1]

    store = {}
    if CACHE.exists():
        with CACHE.open("rb") as fh:
            store = pickle.load(fh)
        print(f"[tp] resuming with {len(store)} folds cached", flush=True)

    todo = []
    for d in DESIGNS:
        for f in all_folds(bench.frame, design=d):
            if (d, f.seed, f.fold) not in store:
                todo.append((d, f))
    print(f"[tp] {len(todo)} folds to do", flush=True)

    t_start = time.time()
    for i, (d, f) in enumerate(todo):
        rtr = f.train_index[rich[f.train_index]]
        if len(rtr) < 30 or len(set((amp[rtr] < 0).tolist())) < 2:
            store[(d, f.seed, f.fold)] = {"test": f.test_index, "skip": True}
            continue
        w = cell_weights(bench.groups[rtr], bench.n_obs[rtr])
        take = K.balanced_resample(w)
        rows = rtr[take]
        y_dir = (amp[rows] < 0).astype(int)
        y_mag = np.log(np.abs(amp[rows]) + K.EPS)
        y_cur = cur[rows]
        rec = {"test": f.test_index, "skip": False}
        t0 = time.time()
        for key, cols in (("39", fs["TOPO39"]), ("209", fs["LEAN209"])):
            Xtr, Xte = K.median_impute(X[np.ix_(rows, cols)], X[np.ix_(f.test_index, cols)])
            want = ("p",) if key == "39" else ("p",)
            r = _fit_predict(Cls, Reg, Xtr, Xte, y_dir, y_mag, y_cur, f.model_seed, want)
            rec[f"p{key}"] = r["p"]
        Xtr, Xte = K.median_impute(X[np.ix_(rows, fs["TOPO39"])], X[np.ix_(f.test_index, fs["TOPO39"])])
        r = _fit_predict(Cls, Reg, Xtr, Xte, y_dir, y_mag, y_cur, f.model_seed, ("mag", "curv"))
        rec["mag"], rec["curv"] = r["mag"], r["curv"]
        store[(d, f.seed, f.fold)] = rec
        with CACHE.open("wb") as fh:
            pickle.dump(store, fh, protocol=5)
        el = time.time() - t_start
        print(f"[tp] {i + 1}/{len(todo)} {d} seed={f.seed} fold={f.fold} "
              f"ntr={len(rows)} nte={len(f.test_index)} {time.time() - t0:.0f}s "
              f"(elapsed {el / 60:.1f} min, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min)",
              flush=True)
    print("[tp] done", flush=True)


if __name__ == "__main__":
    main()
