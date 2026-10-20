#!/usr/bin/env python
"""PROTOCOL section 8 item 2 on EVERY fold of both designs for one configuration: the whole fitted chain
(registry.fit: HBK, POL_ERR table, corrector, lambda, ...) is fitted twice, on the real bench and on a bench whose
log D of every cell OUTSIDE the outer training fold is replaced by a constant (bench.Y itself is replaced, so a
closure over bench.Y is seen too).  Every fitted quantity must be identical.  For a POL_ERR configuration the read
sets chosen for every outer test cell must be identical as well (label-free placement).

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_poison_chain.py R23
writes results/leakage/poison_chain_<ID>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3 import harness as H, paths, registry as RG  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def deep_equal(a, b) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    if hasattr(a, "__dict__") and not isinstance(a, type):
        return type(a) is type(b) and deep_equal(vars(a), vars(b))
    return bool(a == b)


def main() -> int:
    cid = sys.argv[1]
    bench = D.load_bench()
    cfg = RG.get(cid, bench)
    assert cfg.version == "V1"
    res, t0 = {"config": cid, "folds": [], "all_identical": True}, time.time()
    for design in ("BP", "BPG"):
        for f in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS)):
            rec = C.load(f.design, f.seed, f.fold)
            tr = np.asarray(rec["outer"]["train_index"])
            Yp = np.where(np.isfinite(bench.Y), 77.0, np.nan)
            Yp[tr] = bench.Y[tr]
            pb = SimpleNamespace(Y=Yp, groups=bench.groups, n_obs=bench.n_obs, basis=bench.basis, frame=bench.frame,
                                 coef=bench.coef, frames=bench.frames)
            a, b = RG.fit(cfg, bench, rec, bench.Y), RG.fit(cfg, pb, rec, Yp)
            same_fit = a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
            same_reads, n_cells = True, 0
            if not isinstance(cfg.policy, str):
                p1, p2 = cfg.policy.fresh(), cfg.policy.fresh()
                b1 = dict(rec["outer"], Y=bench.Y, fitted=a, _rec=rec)
                b2 = dict(rec["outer"], Y=Yp, fitted=b, _rec=rec)
                for cell in RG.cells_of(cfg, rec["outer"]):
                    n_cells += 1
                    if list(H.seq_for(b1, cell, p1, 3)) != list(H.seq_for(b2, cell, p2, 3)):
                        same_reads = False
            ok = bool(same_fit and same_reads)
            res["all_identical"] &= ok
            res["folds"].append({"design": design, "seed": int(f.seed), "fold": int(f.fold), "fit_identical": bool(same_fit),
                                 "read_sets_identical": bool(same_reads), "test_cells_checked": n_cells,
                                 "poisoned_cells": int(np.isfinite(bench.Y).any(1).sum() - len(tr))})
            print(cid, design, f.seed, f.fold, "OK" if ok else "DIFFERS", f"{time.time() - t0:.0f}s", flush=True)
    res["n_folds"] = len(res["folds"])
    out = paths.RES / "leakage" / f"poison_chain_{cid}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(cid, "ALL IDENTICAL" if res["all_identical"] else "FAILED", "->", out)
    return 0 if res["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
