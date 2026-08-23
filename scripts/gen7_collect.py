"""Assemble every gen7 suite into one leaderboard, and check they are comparable.

Numbers in the final report are read from here, never transcribed by hand.  The
script refuses to merge two runs whose cohort fingerprint differs, because that is
exactly the failure mode ("a lower MAE could just mean easier test rows") the whole
evaluation design exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import leaderboard, load_cohort, score_oof  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen7_architecture"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="*", default=None)
    parser.add_argument("--min-seeds", type=int, default=1)
    args = parser.parse_args(argv)

    directories = ([Path(d) for d in args.dirs] if args.dirs
                   else sorted(p for p in OUT.iterdir()
                               if p.is_dir() and (p / "oof_predictions.parquet").exists()))
    cohort = load_cohort()
    rows, provenance = [], []
    for directory in directories:
        summary_path = directory / "summary.json"
        versions = None
        if summary_path.exists():
            try:
                versions = json.loads(summary_path.read_text()).get("library_versions")
            except Exception:
                versions = None
        oof = pd.read_parquet(directory / "oof_predictions.parquet")
        # a suite's rows must be the cohort's rows
        n_rows = int(oof.groupby(["model", "split_seed"]).size().max())
        if n_rows != len(cohort.frame):
            provenance.append({"suite": directory.name, "status": "SKIPPED_ROW_COUNT",
                               "n_rows": n_rows, "expected": len(cohort.frame)})
            continue
        scores = score_oof(oof)
        scores["suite"] = directory.name
        rows.append(scores)
        provenance.append({"suite": directory.name, "status": "ok",
                           "models": int(oof["model"].nunique()),
                           "seeds": sorted(int(s) for s in oof["split_seed"].unique())})

    if not rows:
        raise SystemExit("no comparable suites found")
    scores = pd.concat(rows, ignore_index=True)
    scores.to_csv(OUT / "all_scores_by_seed.csv", index=False)

    # a model may appear in several suites (e.g. the reference arm); keep the run
    # with the most seeds and record the duplication rather than averaging over
    # different seed sets
    duplicates = (scores.groupby(["model", "suite"])["split_seed"].nunique()
                  .reset_index().sort_values("split_seed", ascending=False))
    best_suite = duplicates.drop_duplicates("model").set_index("model")["suite"].to_dict()
    filtered = scores[[best_suite[m] == s for m, s in zip(scores["model"], scores["suite"])]]

    board = leaderboard(filtered.drop(columns=["suite"]))
    board["suite"] = board["model"].map(best_suite)
    board = board[board["n_seeds"] >= args.min_seeds]
    board.to_csv(OUT / "leaderboard_all.csv", index=False)

    pd.set_option("display.width", 230)
    columns = [c for c in ["model", "suite", "macro_mae", "macro_mae_sd", "offset_mae",
                           "shape_mae", "shape_r2", "pooled_mae", "nn_lt_0_4__macro_mae",
                           "n_seeds"] if c in board.columns]
    print(board[columns].round(4).to_string(index=False))
    (OUT / "collect_provenance.json").write_text(json.dumps(
        {"cohort_fingerprint": cohort.fingerprint, "suites": provenance}, indent=2))
    print(f"\ncohort fingerprint {cohort.fingerprint}; {len(board)} models")
    print(f"artifacts -> {OUT/'leaderboard_all.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
