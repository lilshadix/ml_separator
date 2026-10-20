#!/usr/bin/env python
"""Leakage check 3 (PROTOCOL §8): rebuild every fold's cache with the held-out cells' labels (log D matrix,
the label-derived coefficient vector, and the logD__/repsd__/replicate columns of the frame) replaced by
random numbers, and compare the allowed cache fields with the stored cache: outer covariance, noise s2,
RIDGE2 prior, ridge lambda, noise factor, every held-out cell's metals / queries / k_max / DOPT_Q and UNIFORM
sequences / zero-shot curves, and every inner block (training indices, covariance, sequences).  All must be
identical (they may only depend on training labels and on label-free metadata).

    .venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_cache_poison.py [--max-folds N]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from r3 import paths  # noqa: E402
from g21 import cache as C  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def poisoned_bench(bench, test_index, rng):
    pb = copy.copy(bench)
    Y = np.array(bench.Y, copy=True)
    te = np.asarray(test_index)
    Y[te] = np.where(np.isfinite(Y[te]), rng.normal(0, 5, size=Y[te].shape), np.nan)
    pb.Y = Y
    coef = np.array(bench.coef, copy=True)
    coef[te] = rng.normal(0, 5, size=coef[te].shape)
    pb.coef = coef
    fr = bench.frame.copy()
    lab_cols = [c for c in fr.columns if c.startswith("logD__") or c.startswith("repsd__")] + ["replicate_sd_median"]
    for c in lab_cols:
        v = fr[c].to_numpy(float).copy()
        sel = np.zeros(len(v), bool)
        sel[te] = True
        sel &= np.isfinite(v)
        v[sel] = rng.normal(0, 5, size=int(sel.sum()))
        fr[c] = v
    pb.frame = fr
    return pb


def compare(a, b, path="") -> list:
    bad = []
    if isinstance(a, dict):
        for k in a:
            if k in ("seconds", "selection_scores"):
                continue
            bad += compare(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        for i, (x, y) in enumerate(zip(a, b)):
            bad += compare(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            bad.append(f"{path}: array differs (max {np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))) if a.shape == b.shape else 'shape'})")
    else:
        if a != b and not (isinstance(a, float) and np.isnan(a) and np.isnan(b)):
            bad.append(f"{path}: {a!r} vs {b!r}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-folds", type=int, default=None)
    a = ap.parse_args()
    bench = D.load_bench()
    rng = np.random.default_rng(20261019)
    report = {"folds": 0, "failures": []}
    for design in ("BP", "BPG"):
        folds = folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS))
        for f in folds[: a.max_folds] if a.max_folds else folds:
            stored = C.load(f.design, f.seed, f.fold)
            pb = poisoned_bench(bench, f.test_index, rng)
            rebuilt = C.build_fold(pb, f)
            # the held-out cells' zero-shot curves and the inner blocks must not depend on held-out labels
            bad = compare(stored["outer"], rebuilt["outer"], "outer") + compare(stored["inner"], rebuilt["inner"], "inner")
            report["folds"] += 1
            if bad:
                report["failures"].append({"fold": f"{design} {f.seed} {f.fold}", "differences": bad[:10]})
            print(f"{design} {f.seed} {f.fold}: {'OK' if not bad else 'DIFF ' + str(bad[:3])}", flush=True)
    report["passed"] = not report["failures"]
    (paths.RES / "leakage").mkdir(parents=True, exist_ok=True)
    (paths.RES / "leakage" / "cache_poison.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("PASSED" if report["passed"] else "FAILED", report["folds"], "folds")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
