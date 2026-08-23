"""Run a named set of gen7 contenders on the frozen cohort and write the artifacts.

Every contender is scored on byte-identical test rows through
:mod:`gen7.harness`, which reproduces the gen6 numbers exactly (verified: the
``TREE_MC_lig2d_ext_massaction`` arm returns macro 1.067993 / offset 0.936996 /
shape 0.502044 on split seed 104729, matching ``runs/gen6_expA_5seed``).

Usage::

    .venv/bin/python scripts/run_gen7_experiment.py --suite learners --seeds 1
    .venv/bin/python scripts/run_gen7_experiment.py --suite learners --out runs/gen7_architecture/learners
"""

from __future__ import annotations

import os

# Must be set before LightGBM/XGBoost initialise their OpenMP pools; see
# ``contenders.BOOSTER_THREADS`` for the deadlock this prevents.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import (  # noqa: E402
    DEFAULT_SEEDS, evaluate_contender, leaderboard, load_cohort, score_oof,
)
from lanthanide_separation.gen7 import suites  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True, help=f"one of {sorted(suites.SUITES)}")
    p.add_argument("--seeds", type=int, default=5, help="number of split seeds (1-5)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--only", nargs="*", default=None, help="restrict to these contender names")
    p.add_argument("--append", action="store_true",
                   help="merge into an existing output directory instead of overwriting")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.suite not in suites.SUITES:
        raise SystemExit(f"unknown suite {args.suite!r}; have {sorted(suites.SUITES)}")
    seeds = DEFAULT_SEEDS[: max(1, min(5, args.seeds))]
    out = args.out or (REPO_ROOT / "runs" / "gen7_architecture" / args.suite)
    out.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()
    print(f"cohort {len(cohort.frame)} rows / {cohort.frame.extractant.nunique()} extractants "
          f"/ fingerprint {cohort.fingerprint}", flush=True)

    contenders = suites.SUITES[args.suite](cohort)
    if args.only:
        wanted = set(args.only)
        contenders = [c for c in contenders if c.name in wanted]
    print(f"suite {args.suite}: {len(contenders)} contenders", flush=True)

    oof_parts: list[pd.DataFrame] = []
    failures: list[dict] = []
    timings: list[dict] = []
    for contender in contenders:
        started = time.time()
        try:
            oof = evaluate_contender(contender, cohort, seeds=seeds)
            oof_parts.append(oof)
            timings.append({"model": contender.name, "seconds": time.time() - started})
        except Exception as error:  # a failed contender must not kill the suite
            print(f"  [{contender.name}] FAILED: {type(error).__name__}: {error}", flush=True)
            failures.append({"model": contender.name, "error": f"{type(error).__name__}: {error}",
                             "traceback": traceback.format_exc()[-2000:]})

    # Written FIRST: an aggregation bug must never be able to hide which
    # contenders failed.  A previous run lost three failure records this way and
    # they read as "silently absent from the leaderboard" instead.
    if failures:
        (out / "failures.json").write_text(json.dumps(failures, indent=2))
    if not oof_parts:
        raise SystemExit("every contender failed; see failures.json")

    oof = pd.concat(oof_parts, ignore_index=True)
    if args.append and (out / "oof_predictions.parquet").exists():
        previous = pd.read_parquet(out / "oof_predictions.parquet")
        previous = previous[~previous["model"].isin(set(oof["model"]))]
        oof = pd.concat([previous, oof], ignore_index=True)
    oof.to_parquet(out / "oof_predictions.parquet", index=False)

    scores = score_oof(oof)
    scores.to_csv(out / "scores_by_seed.csv", index=False)
    board = leaderboard(scores)
    board.to_csv(out / "leaderboard.csv", index=False)

    # Stamped on every run because a silent dependency change already invalidated a
    # sweep once: ``pip install tabpfn`` pulled scikit-learn 1.9.0 -> 1.6.1 and
    # pandas 3.0.5 -> 2.3.3, which moved the reference arm by 0.0014 macro MAE and
    # made suites run on either side of it non-comparable.  Two runs whose
    # ``library_versions`` differ must not share a leaderboard.
    import sklearn as _sklearn
    library_versions = {"scikit-learn": _sklearn.__version__, "pandas": pd.__version__,
                        "numpy": np.__version__}
    summary = {
        "suite": args.suite,
        "library_versions": library_versions,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_fingerprint": cohort.fingerprint,
        "n_rows": int(len(cohort.frame)),
        "seeds": [int(s) for s in seeds],
        "n_contenders": len(contenders),
        "n_failures": len(failures),
        "timings": timings,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    columns = [c for c in ["model", "macro_mae", "macro_mae_sd", "offset_mae", "shape_mae",
                           "shape_r2", "pooled_mae", "nn_lt_0_4__macro_mae", "n_seeds"]
               if c in board.columns]
    pd.set_option("display.width", 220)
    print()
    print(board[columns].to_string(index=False))
    print(f"\nartifacts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
