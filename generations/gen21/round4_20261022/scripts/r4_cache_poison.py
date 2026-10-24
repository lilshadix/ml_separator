#!/usr/bin/env python
"""Cache poisoning over EVERY non-training cell (PROTOCOL section 8 item 3): each dev fold cache is rebuilt with
g21.cache.build_fold on r4.leakage.poisoned_bench (log D, cached gen13 coefficients and frame label columns of
every cell outside the outer training fold randomised -- held-out cells AND the cells the BP / BPG design mask
drops), then compared field by field with the stored cache (round 3's compare: outer covariance, noise s2,
priors, lambda, noise factor, held-out cells' supports / sequences / zero-shot curves, every inner block).
Round 3's check poisoned held-out cells only.

    .venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_cache_poison.py --shard 0/3
writes results/leakage/cache_poison_nontrain_<shard>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r4 import paths  # noqa: E402
from r4.leakage import poisoned_bench  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402

sys.path.insert(0, str(paths.R3DIR / "scripts"))
from r3_cache_poison import compare  # noqa: E402  (read-only reuse of round 3's comparison)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    bench = D.load_bench()
    jobs = [f for d in ("BP", "BPG") for f in folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS))][i::n]
    report, t0 = {"shard": a.shard, "folds": [], "failures": []}, time.time()
    for f in jobs:
        stored = C.load(f.design, f.seed, f.fold)
        pb, cols = poisoned_bench(bench, f.train_index, seed=f.seed * 10 + f.fold + 7)
        rebuilt = C.build_fold(pb, f)
        bad = compare(stored["outer"], rebuilt["outer"], "outer") + compare(stored["inner"], rebuilt["inner"], "inner")
        n_excl = int(bench.Y.shape[0] - len(f.train_index) - len(f.test_index))
        report["folds"].append({"fold": f"{f.design} {f.seed} {f.fold}", "identical": not bad,
                                "poisoned_test_cells": int(len(f.test_index)), "poisoned_design_excluded_cells": n_excl})
        if bad:
            report["failures"].append({"fold": f"{f.design} {f.seed} {f.fold}", "differences": bad[:10]})
        print(f"{f.design} {f.seed} {f.fold}: {'OK' if not bad else 'DIFF ' + str(bad[:3])} "
              f"(excluded {n_excl}) {time.time() - t0:.0f}s", flush=True)
    report["passed"] = not report["failures"]
    out = paths.RES / "leakage" / f"cache_poison_nontrain_{i}of{n}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("PASSED" if report["passed"] else "FAILED", len(report["folds"]), "folds")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
