#!/usr/bin/env python
"""Stage-0 input, brief section 4.3 item 9: beta_frozen, the B1 Gd amplitude refitted on the whole corpus.

Computed before the Stage-0 seal (the brief requires it); it is a training-side fit, not a round-5 score: no fold has a
held-out cell, so nothing is predicted or scored.  Recipe (sealed): one fold record whose outer train_index is every
bench cell, inner blocks built as g21.cache.build_fold builds them (r5.frozen.build_fold_record), design BPG, declared
inner seed gen20k.splits.inner_seed(104729, 0); r4.refs.fit_reference(bench, rec, Y, 'B1', k=3) with the default
LAMBDA_GRID (round 3's grid + infinity).  Logged: lambda, beta, inner scores, coverage.  Sensitivities: design BP (same
recipe), lambda fixed at 0.03 (BPG record), and round 4's fold-mean beta from results/REF/fitted_<design>.json.

Writes cache/frozen/fold_all_{BPG,BP}_104729_0.pkl (when missing; --rebuild rebuilds and requires identical bytes) and
results/stage0/beta_frozen.json.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore \
        generations/gen21/round5_20261024/scripts/s0_beta_frozen.py [--rebuild]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r5 import paths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

from r5 import frozen as FR  # noqa: E402
from gen20k import data as D  # noqa: E402

OUT = FR.STAGE0 / "beta_frozen.json"


def main() -> int:
    from r5 import sealcheck
    sealcheck.refuse_if_sealed("scripts/s0_beta_frozen.py (rewrites sealed Stage-0 inputs)", "--amend" in sys.argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--amend", action="store_true", help="rerun after the seal (an AMENDMENTS.md entry and a reseal follow)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the all-cell records and require identical bytes")
    args = ap.parse_args()
    t0 = time.time()
    bench = D.load_bench()
    print(f"Stage-0 input (brief 4.3 item 9): beta_frozen; bench {len(bench.frame)} cells", flush=True)
    recs, info, checks = {}, {}, {}
    for design in ("BPG", "BP"):
        recs[design], info[design] = FR.load_or_build(bench, design, rebuild=args.rebuild)
        checks[design] = FR.record_checks(bench, recs[design])
        c = checks[design]
        print(f"  {design}: train {c['n_train']} / test {c['n_test']}; inner blocks {c['n_inner_blocks']} "
              f"{c['inner_block_sizes']}; C3 with inner record {c['c3_cells_with_inner_record']}/{c['n_c3_cells']}; "
              f"orphans {c['orphan_cells']}; checks ok {c['ok']}", flush=True)
        if not c["ok"]:
            raise AssertionError(f"{design} all-cell record fails its construction checks: {c}")
    fits = {}
    fits["primary_BPG"] = FR.fit_beta(bench, recs["BPG"])
    fits["sens_design_BP"] = FR.fit_beta(bench, recs["BP"])
    fits["sens_lambda_0.03_BPG"] = FR.fit_beta(bench, recs["BPG"], lam_grid=(FR.LAMBDA_FIXED,))
    for k, f in fits.items():
        sc = ", ".join(f"{kk}: {v:.6f}" for kk, v in f["inner_scores"].items())
        print(f"  {k}: beta {f['beta']:+.6f}, lambda {f['lambda']}, situations {f['n_situations']}, rows {f['n_rows']}, "
              f"C3 used {f['coverage']['train_cells_k3_used']}/{f['coverage']['train_cells_k3_eligible']} "
              f"({f['seconds']:.0f}s)\n      inner scores {{{sc}}}", flush=True)
    r4 = {d: FR.round4_fold_betas(d) for d in ("BPG", "BP")}
    for d, r in r4.items():
        print(f"  round-4 fold mean {d}: {r['mean']:+.6f} (sd {r['sd']:.4f}, range {r['min']:+.4f} .. {r['max']:+.4f}; "
              f"lambda {r['lambda_counts']}; {r['n_folds']} folds)", flush=True)
    prim = fits["primary_BPG"]
    out = {
        "label": "Stage-0 input (CLAUDE_CODE_PROMPT.md 4.3 item 9), computed before the Stage-0 seal; a training-side "
                 "fit on all bench cells, not a round-5 score",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "beta_frozen": prim["beta"], "lambda": prim["lambda"],
        "recipe": {"outer_train_index": "every bench cell (np.arange(n)); test_index empty", "design": FR.PRIMARY_DESIGN,
                   "split_seed": FR.DECLARED_SEED, "fold": FR.DECLARED_FOLD,
                   "inner_seed": "gen20k.splits.inner_seed(104729, 0) = %d" % checks["BPG"]["inner_seed"],
                   "inner_split": "gen20k.splits._redeal_inner (chemotype groups; noise-factor selection)",
                   "inner_blocks": "r5.frozen.inner_blocks = g21.cache.build_fold's inner cross-fit, unchanged",
                   "fit": "r4.refs.fit_reference(bench, rec, bench.Y, 'B1', k=3), default LAMBDA_GRID",
                   "lambda_tie_break": "toward the stronger penalty (r4.refs)",
                   "blas_threads": FR.BLAS_THREADS},
        "fits": fits,
        "sensitivities": {"design_BP": {"beta": fits["sens_design_BP"]["beta"], "lambda": fits["sens_design_BP"]["lambda"]},
                          "lambda_fixed_0.03": {"beta": fits["sens_lambda_0.03_BPG"]["beta"], "lambda": FR.LAMBDA_FIXED},
                          "round4_fold_mean": {d: r4[d]["mean"] for d in r4}},
        "round4_fold_betas": r4,
        "records": {d: {**info[d], "checks": checks[d]} for d in recs},
        "code_lf_sha256": {"r5/frozen.py": paths.sha256_text_lf(paths.R5 / "r5" / "frozen.py"),
                           "scripts/s0_beta_frozen.py": paths.sha256_text_lf(Path(__file__)),
                           "g21/cache.py (copied into r5.frozen)": FR.G21_CACHE_LF_SHA256,
                           "r4/refs.py (imported)": FR.R4_REFS_LF_SHA256},
        "env": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "seconds": round(time.time() - t0, 1),
    }
    for name, got in (("g21/cache.py", paths.sha256_text_lf(paths.G21 / "g21" / "cache.py")),
                      ("r4/refs.py", paths.sha256_text_lf(paths.R4DIR / "r4" / "refs.py"))):
        want = FR.G21_CACHE_LF_SHA256 if name == "g21/cache.py" else FR.R4_REFS_LF_SHA256
        if got != want:
            raise AssertionError(f"{name} changed since r5.frozen was written: {got}")
    paths.write_text_lf(OUT, json.dumps(out, indent=1) + "\n")
    print(f"beta_frozen = {prim['beta']:+.6f} (lambda {prim['lambda']}); BP {fits['sens_design_BP']['beta']:+.6f}; "
          f"lambda 0.03 {fits['sens_lambda_0.03_BPG']['beta']:+.6f}; round-4 fold mean BPG {r4['BPG']['mean']:+.6f} / "
          f"BP {r4['BP']['mean']:+.6f}")
    print(f"wrote {OUT}")
    for d in recs:
        print(f"wrote/kept {paths.R5 / info[d]['path']} (sha256 {info[d]['sha256'][:12]}, {info[d]['status']})")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
