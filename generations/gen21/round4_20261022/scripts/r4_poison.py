#!/usr/bin/env python
"""Fit poisoning on EVERY fold of both designs (PROTOCOL.md section 8 item 2) for a round-4 configuration: the
whole fitted chain (registry.fit) is fitted on the real bench and on r4.leakage.poisoned_bench (log D, cached
coefficients and frame label columns of every cell outside the outer training fold randomised; bench.Y itself is
replaced, so a closure over it is seen).  Every fitted quantity must be bit-identical.
Block C (A4): the poisoned fit also sees the An(III) record set with log D randomised for every record the fold's
union exclusions remove, so a fit that used an excluded An record would differ.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_poison.py REF
writes results/leakage/poison_<config id>.json (S21:S01 -> poison_S21-S01.json)
Sharded (A4):  r4_poison.py S13 --shard 0/3  ... then  r4_poison.py S13 --combine 3  (the combine step requires the
shards to cover exactly the 50 fold-fits once).
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from r4 import an_data as AD, paths, registry as RG  # noqa: E402
from r4 import slots_c as SCc  # noqa: E402
from r4.leakage import deep_equal, poisoned_bench  # noqa: E402
from r3 import configs as CF  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def _out(cid: str, suffix: str = "") -> Path:
    out = paths.RES / "leakage" / f"poison_{cid}{suffix}.json"
    assert ":" not in out.name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    cid = sys.argv[1]
    shard = sys.argv[sys.argv.index("--shard") + 1] if "--shard" in sys.argv else "0/1"
    bench = D.load_bench()
    cfg = RG.get(cid, bench)
    if "--combine" in sys.argv:
        n = int(sys.argv[sys.argv.index("--combine") + 1])
        parts = [json.loads(_out(cfg.cid, f".shard{i}of{n}").read_text()) for i in range(n)]
        folds = [f for q in parts for f in q["folds"]]
        assert len({(f["design"], f["seed"], f["fold"]) for f in folds}) == len(folds) == 50, "shards must cover 50 fold-fits"
        res = {"config": cfg.cid, "requested_id": cid, "folds": folds, "n_folds": len(folds),
               "all_identical": all(f["fit_identical"] for f in folds), "shards": n}
        _out(cfg.cid).write_text(json.dumps(res, indent=1), encoding="utf-8", newline=chr(10))
        print(cid, "ALL IDENTICAL" if res["all_identical"] else "FAILED", "->", _out(cfg.cid))
        return 0 if res["all_identical"] else 1
    i_sh, n_sh = map(int, shard.split("/"))
    res, t0 = {"config": cfg.cid, "requested_id": cid, "folds": [], "all_identical": True}, time.time()
    jobs = [f for design in ("BP", "BPG") for f in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS))]
    for f in jobs[i_sh::n_sh]:
        design = f.design
        rec = C.load(f.design, f.seed, f.fold)
        pb, cols = poisoned_bench(bench, rec["outer"]["train_index"], seed=f.seed * 10 + f.fold)
        a = RG.fit(cfg, bench, rec, bench.Y)
        real_inner, real_an = CF.INNER_CACHE_DIR, AD.an3_records
        an_c = any(p.startswith("c:") for p in cfg.parts)
        if an_c:
            an = real_an()
            keep = set(AD.fold_an(bench, rec).rid)
            rng = np.random.default_rng(f.seed * 10 + f.fold + 7)
            drop = ~an.rid.isin(keep).to_numpy()
            an_p = an.assign(log_D=np.where(drop, rng.normal(0, 3, len(an)), an.log_D.to_numpy()))
        with tempfile.TemporaryDirectory() as tmp:          # the poisoned fit never reads or writes a shared cache
            CF.INNER_CACHE_DIR = Path(tmp)
            if an_c:
                AD.an3_records = lambda: an_p
            try:
                b = RG.fit(cfg, pb, rec, pb.Y)
            finally:
                CF.INNER_CACHE_DIR, AD.an3_records = real_inner, real_an
        ok = a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
        res["all_identical"] &= bool(ok)
        res["folds"].append({"design": design, "seed": int(f.seed), "fold": int(f.fold), "fit_identical": bool(ok),
                             "poisoned_cells": int(bench.Y.shape[0] - len(rec["outer"]["train_index"])),
                             "poisoned_frame_columns": len(cols), "parts": sorted(a),
                             "poisoned_an_records": int(drop.sum()) if an_c else 0})
        print(cid, design, f.seed, f.fold, "OK" if ok else "DIFFERS", f"{time.time() - t0:.0f}s", flush=True)
        SCc._S11_CACHE.clear()                     # block C: do not keep every fold's poisoned bench alive
    res["n_folds"] = len(res["folds"])
    out = _out(cfg.cid, "" if n_sh == 1 else f".shard{i_sh}of{n_sh}")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8", newline=chr(10))
    print(cid, "ALL IDENTICAL" if res["all_identical"] else "FAILED", "->", out)
    return 0 if res["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
