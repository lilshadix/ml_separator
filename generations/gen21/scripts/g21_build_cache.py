#!/usr/bin/env python
"""Build the per-fold context cache (gen20 selection, covariance, supports, zero-shot curves).

    .venv/Scripts/python.exe generations/gen21/scripts/g21_build_cache.py --designs BP,BPG --shard 0/2
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g21 import cache as C  # noqa: E402
from g21 import paths  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="BP,BPG")
    ap.add_argument("--seeds", default=",".join(map(str, paths.DEV_SEEDS)))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None, help="timing probe: build at most this many folds")
    a = ap.parse_args()
    i, n = (int(x) for x in a.shard.split("/"))
    bench = D.load_bench()
    paths.CACHE.mkdir(parents=True, exist_ok=True)
    folds = [f for d in a.designs.split(",") for f in folds_for(bench.frame, d, seeds=[int(s) for s in a.seeds.split(",")])]
    mine = [f for j, f in enumerate(folds) if j % n == i and not C.cache_path(f.design, f.seed, f.fold).exists()]
    if a.limit:
        mine = mine[: a.limit]
    print(f"shard {i}/{n}: {len(mine)} folds to build", flush=True)
    t0 = time.time()
    for j, f in enumerate(mine):
        rec = C.build_fold(bench, f)
        tmp = C.cache_path(f.design, f.seed, f.fold).with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(rec, fh)
        tmp.replace(C.cache_path(f.design, f.seed, f.fold))
        print(f"  [{j + 1}/{len(mine)}] {f.design} {f.seed} {f.fold}: {len(rec['outer']['cells'])} test / "
              f"{sum(len(b["cells"]) for b in rec["inner"])} inner cells, {rec['seconds']:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
    print("EXIT 0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
