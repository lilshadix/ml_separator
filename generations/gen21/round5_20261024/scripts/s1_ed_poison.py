#!/usr/bin/env python
"""Fit poisoning (CLAUDE_CODE_PROMPT.md 8 item 2) for item 1.6's k-specific B1 beta refits on all 50 fold-fits.

For every outer fold of BP and BPG (5 dev seeds x 5 folds) and every code budget k of r5.ed.BETA_KS, the fit
r4.refs.fit_reference(bench, rec, Y, 'B1', k=k) is run on the real bench and on r4.leakage.poisoned_bench (log D, the
cached gen13 coefficients and the frame label columns logD__*, repsd__*, replicate_sd_median of every cell outside the
outer training fold randomised on their NaN pattern; seed split_seed * 10 + fold, round 4's).  Every fitted quantity
(coef, lambda, inner scores, situation / row counts, coverage) must be bit-identical (r4.leakage.deep_equal).  The real
fit is also compared with the beta of results/ED/fits_<design>.csv when that file exists.  One BLAS thread.

Adapted from round4_20261022/scripts/r4_poison.py (LF SHA-256 7f74900c5c5b65e8a0e09106855deb9fef407152ad02d50f8e3c55e3574a63e8)
with its output moved to the round-5 tree; it never runs r4_poison.py and writes only results/leakage/.

Runs only after the Stage-0 seal (r5.sealcheck.require_seal); about 2 x 6 fits per fold-fit; START / END lines in
RUN_RECORD.md and the post-run seal check.  --combine refuses shards from another seal or with other k budgets, and a
combined file covers ED.BETA_KS only when every shard does (s0-mechanics-13).

    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ed_poison.py
    (sharded: --shard 0/3 ... --shard 2/3, then --combine 3; --ks 1,2,4 to restrict the budgets)
writes results/leakage/poison_ED_beta.json (shards: poison_ED_beta.shard<i>of<n>.json)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402

import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from r5 import ed as ED  # noqa: E402
from r5 import sealcheck  # noqa: E402
from r4 import refs as RF  # noqa: E402
from r4.leakage import deep_equal, poisoned_bench  # noqa: E402
from r3 import configs as CF  # noqa: E402
from g21 import cache as GC  # noqa: E402
from gen20k import data as D  # noqa: E402
from gen20k.splits import folds_for  # noqa: E402


def _out(suffix: str = "") -> Path:
    return ED.LEAK / f"poison_ED_beta{suffix}.json"


def _fit(bench, rec, k):
    with threadpool_limits(limits=1):
        return RF.fit_reference(bench, rec, bench.Y, "B1", k=k)


COPIED_FROM = {"generations/gen21/round4_20261022/scripts/r4_poison.py":
               "7f74900c5c5b65e8a0e09106855deb9fef407152ad02d50f8e3c55e3574a63e8"}
ITEM = "ED_POISON"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--combine", type=int, default=0)
    ap.add_argument("--ks", default=",".join(map(str, ED.BETA_KS)))
    args = ap.parse_args(argv)
    tag = sealcheck.require_seal()
    mode = f"combine {args.combine}" if args.combine else f"shard {args.shard}"
    return sealcheck.run_sealed(ITEM, mode, tag, lambda: body(args, tag))


def body(args, tag: str) -> int:
    ks = tuple(int(x) for x in args.ks.split(","))
    if args.combine:
        n = args.combine
        parts = [json.loads(_out(f".shard{i}of{n}").read_bytes()) for i in range(n)]
        if not all(q.get("seal") == tag and list(q.get("ks", [])) == list(ks) for q in parts):
            raise SystemExit(f"shards must carry the current seal {tag} and the same ks {list(ks)}")
        if list(ks) != list(ED.BETA_KS):
            raise SystemExit(f"--combine needs every budget of ED.BETA_KS {list(ED.BETA_KS)} (got {list(ks)})")
        folds = [f for q in parts for f in q["folds"]]
        keys = {(f["design"], f["seed"], f["fold"]) for f in folds}
        if not (len(keys) == len(folds) == 50):
            raise SystemExit("shards must cover the 50 fold-fits exactly once")
        res = {"item": "1.6 (E-D) k-specific beta refits", "seal": tag, "ks": parts[0]["ks"], "folds": folds,
               "n_folds": len(folds), "all_identical": all(f["fit_identical"] for f in folds),
               "matches_main_run": all(f.get("matches_main_run", True) for f in folds), "shards": n,
               "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        paths.write_text_lf(_out(), json.dumps(res, indent=1) + "\n")
        print("ALL IDENTICAL" if res["all_identical"] else "FAILED", "->", _out())
        return 0 if res["all_identical"] else 1
    i_sh, n_sh = map(int, args.shard.split("/"))
    bench = D.load_bench()
    main_fits = {}
    for d in paths.DESIGNS:
        p = ED.OUT / f"fits_{d}.csv"
        if p.exists():
            t = pd.read_csv(p, float_precision="round_trip")       # A1: the default C parser is not round-trip exact
            main_fits.update({(d, int(r.split_seed), int(r.fold), int(r.k)): float(r.beta) for r in t.itertuples()})
    jobs = [f for design in ("BP", "BPG") for f in folds_for(bench.frame, design, seeds=list(paths.DEV_SEEDS))]
    res = {"item": "1.6 (E-D) k-specific beta refits", "seal": tag, "ks": list(ks), "folds": [], "all_identical": True}
    t0 = time.time()
    for f in jobs[i_sh::n_sh]:
        rec = GC.load(f.design, f.seed, f.fold)
        pb, cols = poisoned_bench(bench, rec["outer"]["train_index"], seed=f.seed * 10 + f.fold)
        per_k, ok_all, match = {}, True, True
        real_inner = CF.INNER_CACHE_DIR
        for k in ks:
            a = _fit(bench, rec, k)
            with tempfile.TemporaryDirectory() as tmp:          # the poisoned fit never reads or writes a shared cache
                CF.INNER_CACHE_DIR = Path(tmp)
                try:
                    b = _fit(pb, rec, k)
                finally:
                    CF.INNER_CACHE_DIR = real_inner
            ok = a.keys() == b.keys() and all(deep_equal(a[x], b[x]) for x in a)
            mk = (f.design, int(f.seed), int(f.fold), int(k))
            mm = None if mk not in main_fits or k == ED.K_EA else bool(float(a["coef"][0]) == main_fits[mk])
            per_k[str(k)] = {"fit_identical": bool(ok), "matches_main_run": mm}
            ok_all &= bool(ok)
            match &= mm is not False
        for key in [kk for kk in RF._ORPHANS if kk[0] == id(pb)]:   # do not keep every poisoned bench alive
            RF._ORPHANS.pop(key)
        res["all_identical"] &= ok_all
        res["folds"].append({"design": f.design, "seed": int(f.seed), "fold": int(f.fold), "fit_identical": ok_all,
                             "matches_main_run": match, "per_k": per_k,
                             "poisoned_cells": int(bench.Y.shape[0] - len(rec["outer"]["train_index"])),
                             "poisoned_frame_columns": len(cols)})
        print(f.design, f.seed, f.fold, "OK" if ok_all else "DIFFERS", f"{time.time() - t0:.0f}s", flush=True)
    res["n_folds"] = len(res["folds"])
    res["matches_main_run"] = all(f["matches_main_run"] for f in res["folds"])
    res["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out = _out("" if n_sh == 1 else f".shard{i_sh}of{n_sh}")
    paths.write_text_lf(out, json.dumps(res, indent=1) + "\n")
    print("ALL IDENTICAL" if res["all_identical"] else "FAILED", "->", out)
    print(f"wrote {out}")
    return 0 if res["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
