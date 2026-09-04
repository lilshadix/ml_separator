"""gen7's evaluation contract, driving a gen10 :class:`~.architectures.QueryModel`.

Nothing here changes how an arm is scored.  ``evaluate_contender`` is gen7's, the
cohort is gen7's cache, the folds are ``seeded_group_kfold`` over the Tanimoto
chemotype at gen5's five seeds, the model seed is ``42 + fold*1009 + 9_999_991``,
and the OOF frame lands in the schema every gen8 and gen9 tool already reads.  A
gen10 arm and a gen9 arm are therefore the same arithmetic on byte-identical rows,
which is the only reason the longitudinal frontier means anything.

What this module adds is the bookkeeping the brief asks for at every stage: the
fold and curve-boundary audits before anything is fitted, row accounting after,
and a determinism check that refits one fold twice and records the spread.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# scikit-learn 1.9 warns, once per parallel predict call, that `delayed` should be
# paired with its own `Parallel`; it is a notice about configuration propagation,
# not about results, and at thousands of predict calls it buries the log.
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

from ..gen7.harness import (
    DEFAULT_SEEDS, build_folds, evaluate_contender, load_cohort, score_oof,
)
from ..gen9.audit import (
    assert_fold_audit_clean, assert_row_accounting_clean, curve_pair_audit, fold_audit,
    row_accounting,
)
from ..gen9.train import load_membership
from ..levels import LEVEL_TARGET_COLUMN
from .architectures import ContenderAdapter
from .querycurves import assert_partition_closure

#: The cohort every generation since gen6 has been scored on.  A mismatch is fatal.
COHORT_FINGERPRINT = "bed178ec1a7a82b0"

#: Two runs of one configuration may differ by at most this, in log units.
#:
#: ``ExtraTreesRegressor(n_jobs=-1)`` sums trees in thread-completion order, so a
#: forest is reproducible to ~1e-15 and no better.  Demanding bit-identity would
#: fail on arithmetic rather than on logic; demanding 1e-9 would have let gen9's
#: issue 6 and gen10's :class:`ResidualShapeModel` defect (3.4e-2) through in the
#: *other* direction — both were caught precisely because the observed spread was
#: compared against the floor rather than against a tolerance chosen to pass.
DETERMINISM_TOLERANCE = 1e-12


def prepared_cohort(*, require_fingerprint: bool = True):
    cohort = load_cohort()
    if require_fingerprint and cohort.fingerprint != COHORT_FINGERPRINT:
        raise SystemExit(
            f"cohort fingerprint moved: {cohort.fingerprint} != {COHORT_FINGERPRINT}")
    return cohort


def audit_folds(cohort, seeds: Sequence[int], membership: pd.DataFrame,
                out_dir: Path) -> dict:
    """Fold integrity, curve-boundary closure and partition closure, before fitting."""
    audits, straddles, closure = [], [], []
    for seed in seeds:
        folds = build_folds(cohort.frame, seed)
        audits.append(fold_audit(cohort.frame, folds))
        straddles.append(curve_pair_audit(membership, folds, cohort.frame))
        for fold in folds:
            ids = cohort.frame.iloc[fold.test_index]["row_id"].astype(str).to_numpy()
            record = assert_partition_closure(membership, ids, name=f"s{seed}f{fold.fold}")
            record.update({"split_seed": int(seed), "fold": int(fold.fold)})
            closure.append(record)
    fold_table = pd.concat(audits, ignore_index=True)
    assert_fold_audit_clean(fold_table)
    straddle_table = pd.concat(straddles, ignore_index=True)
    closure_table = pd.DataFrame.from_records(closure)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(out_dir / "fold_audit.csv", index=False)
    straddle_table.to_csv(out_dir / "curve_boundary_audit.csv", index=False)
    closure_table.to_csv(out_dir / "partition_closure_audit.csv", index=False)
    n_straddle = int(straddle_table["n_curves_straddling_train_test"].sum())
    if n_straddle:
        raise SystemExit(f"{n_straddle} curves straddle a train/test boundary")
    return {"n_curves_straddling": n_straddle,
            "n_folds": int(len(fold_table)),
            "max_rows_outside_partition": int(closure_table["n_rows_outside"].max())}


def run_arm(model, cohort, seeds: Sequence[int], out_dir: Path, *,
            verbose: bool = True) -> pd.DataFrame:
    """Fit and score one arm over every seed and fold; write its OOF parquet."""
    contender = ContenderAdapter(model)
    started = time.time()
    oof = evaluate_contender(contender, cohort, seeds=seeds, verbose=verbose)
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out_dir / f"oof_{contender.name}.parquet", index=False)
    if contender.diagnostics:
        (out_dir / f"diagnostics_{contender.name}.json").write_text(
            json.dumps(contender.diagnostics, indent=1, default=float))
    if verbose:
        print(f"  [{contender.name}] {time.time() - started:.0f}s total", flush=True)
    return oof


def determinism_probe(model_factory, cohort, *, seed: int = DEFAULT_SEEDS[0],
                      fold_index: int = 0) -> dict:
    """Refit one fold twice from scratch and report the spread.

    Deliberately *not* a tolerance check that passes quietly: it returns the
    observed maximum, the number of rows that moved and the fraction, so a report
    can state the floor rather than assert it is small.
    """
    from ..gen7.harness import FoldContext

    frame = cohort.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    fold = build_folds(frame, seed)[fold_index]
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
    context = FoldContext(fold=fold, cohort=cohort, feature_columns=(),
                          model_seed=fold.model_seed)
    y = target[fold.train_index]
    first = model_factory().fit(train, y, context).predict(test)
    second = model_factory().fit(train, y, context).predict(test)
    delta = np.abs(np.asarray(first) - np.asarray(second))
    return {"split_seed": int(seed), "fold": int(fold_index), "n_rows": int(len(delta)),
            "max_abs_delta": float(delta.max()), "mean_abs_delta": float(delta.mean()),
            "n_rows_moved": int((delta > 0).sum()),
            "within_tolerance": bool(delta.max() <= DETERMINISM_TOLERANCE),
            "pooled_mae_first": float(np.abs(first - target[fold.test_index]).mean()),
            "pooled_mae_second": float(np.abs(second - target[fold.test_index]).mean())}


def finalise(frames: Sequence[pd.DataFrame], cohort, out_dir: Path) -> pd.DataFrame:
    """Concatenate, audit row accounting, score, and write the leaderboard."""
    combined = pd.concat(list(frames), ignore_index=True)
    accounting = row_accounting(combined, expected_rows=len(cohort.frame))
    accounting.to_csv(out_dir / "row_accounting.csv", index=False)
    assert_row_accounting_clean(accounting)
    combined.to_parquet(out_dir / "oof_all.parquet", index=False)
    scores = score_oof(combined)
    scores.to_csv(out_dir / "scores_by_seed.csv", index=False)
    board = scores.groupby("model")[["macro_mae", "offset_mae", "shape_mae", "pooled_mae"]].mean()
    board["macro_sd"] = scores.groupby("model")["macro_mae"].std()
    board["n_seeds"] = scores.groupby("model").size()
    board = board.sort_values("macro_mae")
    board.to_csv(out_dir / "leaderboard.csv")
    return board


def default_membership() -> pd.DataFrame:
    return load_membership()
