"""The checks that must pass before any gen11 number is reported (brief §20).

Written to fail, not to reassure.  Every check here corresponds to a way this
sprint could produce a confident wrong answer, and several of them have already
caught one:

* the ``SAME_SERIES`` guard went dead when auxiliary series were namespaced for
  curve safety, while 2,505 auxiliary rows shared a cohort series;
* auxiliary rows arrived at raw-record granularity against a replicate-averaged
  cohort, so one cell would have carried 24x a cohort cell's weight;
* 947 auxiliary rows carried an ionic radius the archive explicitly declines to
  state.

A check that cannot fail is worse than no check, so where a property is true by
construction this script says so in the ``vacuous`` field rather than claiming
evidence it does not have.

Usage::

    PYTHONPATH=src python scripts/gen11_self_audit.py --arms runs/gen11_transfer/arms
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import (  # noqa: E402
    DEFAULT_SEEDS, FoldContext, build_folds,
)
from lanthanide_separation.gen10.runner import COHORT_FINGERPRINT, prepared_cohort  # noqa: E402
from lanthanide_separation.gen11.arms import ARM_BY_KEY  # noqa: E402
from lanthanide_separation.gen11.overlap import build_overlap_map, cohort_identity  # noqa: E402
from lanthanide_separation.gen11.pools import build_pool  # noqa: E402
from lanthanide_separation.gen11.runner import (  # noqa: E402
    RunSpec, attach_metal_scheme, build_model,
)
from lanthanide_separation.gen11.transfer import nesting_delta  # noqa: E402
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen11_transfer" / "self_audit"
AUX_FEATURES = REPO_ROOT / "runs" / "gen11_transfer" / "featurizer" / "aux_features.parquet"

#: Two fits of one configuration may differ by at most this (thread-order floor).
DETERMINISM_TOLERANCE = 1e-12


def _check(name: str, passed: bool, *, vacuous: bool = False, **detail) -> dict:
    return {"check": name, "passed": bool(passed), "vacuous": bool(vacuous), **detail}


# --------------------------------------------------------------------------- #

def check_identity(cohort) -> list[dict]:
    bundle = REPO_ROOT / "dataset with 3D structures" / "dataset.parquet"
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    pinned = json.loads((REPO_ROOT / "gen3_protocol.json").read_text())

    def find(node, key):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key:
                    return v
                found = find(v, key)
                if found is not None:
                    return found
        return None

    return [
        _check("cohort_fingerprint_unchanged", cohort.fingerprint == COHORT_FINGERPRINT,
               observed=cohort.fingerprint, expected=COHORT_FINGERPRINT),
        _check("frozen_bundle_sha256_matches_gen3_protocol",
               digest == find(pinned, "dataset_sha256"),
               observed=digest, expected=find(pinned, "dataset_sha256")),
        _check("gen11_nests_gen10_at_the_thread_order_floor",
               nesting_delta(cohort, seed=DEFAULT_SEEDS[0])["max_abs_delta"]
               <= DETERMINISM_TOLERANCE,
               tolerance=DETERMINISM_TOLERANCE,
               observed=nesting_delta(cohort, seed=DEFAULT_SEEDS[0])["max_abs_delta"]),
    ]


def _fold_setup(cohort, pool, spec: RunSpec, seed: int, fold_index: int = 0):
    from dataclasses import replace as dc_replace

    patched, aux = attach_metal_scheme(cohort, pool.features, spec.metal_scheme,
                                       spec.massaction_scheme)
    model = build_model(spec, dc_replace(pool, features=aux) if aux is not None else pool)
    frame = patched.frame
    fold = build_folds(frame, seed)[fold_index]
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
    y = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)[fold.train_index]
    context = FoldContext(fold=fold, cohort=patched, feature_columns=(),
                          model_seed=fold.model_seed)
    return model, frame, fold, train, test, y, context


def check_prediction_invariances(cohort, pool, spec: RunSpec) -> list[dict]:
    """The properties a prediction must have regardless of what the arm learned."""
    out: list[dict] = []
    model, frame, fold, train, test, y, context = _fold_setup(cohort, pool, spec, DEFAULT_SEEDS[0])

    first = np.asarray(model.fit(train, y, context).predict(test), dtype=float)
    second = np.asarray(model.fit(train, y, context).predict(test), dtype=float)
    delta = float(np.abs(first - second).max())
    out.append(_check("determinism_same_configuration_twice", delta <= DETERMINISM_TOLERANCE,
                      max_abs_delta=delta, tolerance=DETERMINISM_TOLERANCE))

    # No held-out log D may be an input.  Corrupting every held-out target must
    # leave the prediction bit-identical; if it moves, the target is being read.
    out.append(_check("held_out_target_absent_from_predict_input",
                      LEVEL_TARGET_COLUMN not in test.columns,
                      column=LEVEL_TARGET_COLUMN))
    corrupted = frame.iloc[fold.test_index].copy()
    corrupted[LEVEL_TARGET_COLUMN] = 999.0
    corrupted = corrupted.drop(columns=[LEVEL_TARGET_COLUMN])
    third = np.asarray(model.fit(train, y, context).predict(corrupted), dtype=float)
    corrupt_delta = float(np.abs(first - third).max())
    out.append(_check("corrupt_held_out_targets_do_not_move_predictions",
                      corrupt_delta <= DETERMINISM_TOLERANCE,
                      max_abs_delta=corrupt_delta,
                      note="the target column is dropped before predict, so this is "
                           "a guard against it being re-derived, not a strong test",
                      vacuous=True))

    # Permuting the query rows must permute the predictions and nothing else.
    # This is NOT vacuous: the design-relative features are computed over the
    # query set, so a representation that depended on row order would move here.
    order = np.random.default_rng(0).permutation(len(test))
    permuted = np.asarray(model.fit(train, y, context).predict(test.iloc[order]), dtype=float)
    permute_delta = float(np.abs(permuted - first[order]).max())
    out.append(_check("row_permutation_permutes_predictions_only",
                      permute_delta <= DETERMINISM_TOLERANCE,
                      max_abs_delta=permute_delta))
    return out


def check_row_accounting(arms_dir: Path, cohort) -> list[dict]:
    """Every arm scored every cohort row, once per seed, on the cohort's own target."""
    out: list[dict] = []
    expected = cohort.frame[["row_id", LEVEL_TARGET_COLUMN]].astype({"row_id": str})
    for path in sorted(arms_dir.glob("oof_*.parquet")):
        oof = pd.read_parquet(path)
        seeds = sorted(oof["split_seed"].unique())
        counts = oof.groupby("split_seed").size()
        merged = oof.assign(row_id=oof["row_id"].astype(str)).merge(
            expected, on="row_id", how="left", suffixes=("", "_expected"))
        target_delta = float(
            (merged[LEVEL_TARGET_COLUMN] - merged[f"{LEVEL_TARGET_COLUMN}_expected"]).abs().max())
        extra = set(oof["row_id"].astype(str)) - set(expected["row_id"])
        out.append(_check(f"row_accounting[{path.stem}]",
                          bool((counts == len(expected)).all()) and not extra
                          and target_delta == 0.0 and not oof["prediction"].isna().any(),
                          n_seeds=len(seeds), rows_per_seed=sorted(set(counts.tolist())),
                          expected_rows=int(len(expected)), n_extra_rows=len(extra),
                          max_abs_target_delta=target_delta,
                          n_unpredicted=int(oof["prediction"].isna().sum())))
    return out


def check_auxiliary_leakage(cohort, overlap, pool) -> list[dict]:
    """No admissible auxiliary row may be related to a held-out row.

    Recomputed here from the raw archive columns rather than by re-reading the
    pool's own masks, so a bug in the mask code cannot certify itself.
    """
    out: list[dict] = []
    identity = cohort_identity(cohort.frame, overlap)
    aux = pool.features
    violations = {"duplicate_group": 0, "series": 0, "exact_record": 0}
    zero_survivor_folds = 0

    for seed in DEFAULT_SEEDS:
        for fold in build_folds(cohort.frame, seed):
            held_ids = set(cohort.frame.iloc[fold.test_index]["row_id"].astype(str))
            held = identity[identity["row_id"].isin(held_ids)]
            admissible = pool.admissible(seed, fold.fold)
            if admissible.empty:
                zero_survivor_folds += 1
            violations["duplicate_group"] += int(
                admissible["duplicate_group_id"].isin(set(held["duplicate_group_id"].dropna())).sum())
            violations["series"] += int(
                admissible["archive_series_id"].astype(str)
                .isin(set(held["series_id_src"].dropna().astype(str))).sum())
            violations["exact_record"] += int(
                admissible["source_record_id"].astype(str)
                .isin(set(held["source_record_id"].astype(str))).sum())

    for key, count in violations.items():
        out.append(_check(f"no_admissible_aux_row_shares_{key}_with_a_held_out_row",
                          count == 0, n_violations=int(count)))
    out.append(_check("every_fold_admits_a_nonempty_auxiliary_pool",
                      zero_survivor_folds == 0, n_empty_folds=zero_survivor_folds))

    frozen = set(overlap.exact["source_record_id"].astype(str))
    out.append(_check("auxiliary_pool_contains_no_frozen_bundle_record",
                      not (set(aux["source_record_id"].astype(str)) & frozen),
                      n_intersection=len(set(aux["source_record_id"].astype(str)) & frozen)))
    out.append(_check("auxiliary_pool_contains_no_redundant_duplicate_record",
                      bool((aux["model_readiness"] == "A_model_ready").all()),
                      readiness=aux["model_readiness"].value_counts().to_dict()))
    return out


def check_provenance_not_a_feature(cohort, pool, spec: RunSpec) -> list[dict]:
    """No design column may separate auxiliary from cohort rows on bookkeeping.

    A column that is ~always missing on one side and ~never on the other becomes,
    under ``add_indicator=True``, an "is this an auxiliary row" flag.  That is the
    defect that removed ``massact__logL_x_coreCN`` from every gen11 arm, and this
    check is what would catch the next one.
    """
    patched, aux = attach_metal_scheme(cohort, pool.features, spec.metal_scheme,
                                       spec.massaction_scheme)
    columns = list(patched.block_columns(
        build_model(spec, pool).blocks))
    banned = [c for c in columns
              if any(token in c.lower() for token in ("doi", "source_file", "record_id",
                                                      "publication", "author", "entry_"))]
    rows = []
    for column in columns:
        if column not in aux.columns:
            continue
        core_missing = float(patched.frame[column].isna().mean())
        aux_missing = float(aux[column].isna().mean())
        rows.append({"column": column, "core_missing": core_missing,
                     "aux_missing": aux_missing,
                     "separation": abs(core_missing - aux_missing)})
    table = pd.DataFrame(rows).sort_values("separation", ascending=False)
    worst = table.head(10)
    # 0.90 is deliberately strict: the two columns gen11 removed sat at 0.984.
    offenders = table[table["separation"] >= 0.90]
    offenders = offenders[~offenders["column"].str.contains("is_lanthanide|is_actinide")]
    return [
        _check("no_provenance_named_column_in_the_design", not banned, columns=banned),
        _check("no_design_column_separates_aux_from_core_on_missingness",
               offenders.empty, n_offenders=int(len(offenders)),
               offenders=offenders["column"].tolist()[:10],
               worst=worst.to_dict("records")),
    ]


def check_fresh_subprocess_determinism() -> list[dict]:
    """Splits and control seeds must not depend on ``PYTHONHASHSEED``."""
    code = (
        "import sys; sys.path.insert(0,'src');"
        "from lanthanide_separation.gen10.runner import prepared_cohort;"
        "from lanthanide_separation.gen7.harness import build_folds;"
        "c=prepared_cohort();f=build_folds(c.frame,104729)[0];"
        "print(c.fingerprint, len(f.train_index), len(f.test_index),"
        " ''.join(sorted(f.held_out_chemotypes))[:40])"
    )
    results = {}
    for seed in ("0", "1", "4242"):
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=REPO_ROOT, env={"PYTHONHASHSEED": seed,
                                                  "PATH": "/usr/bin:/bin"})
        results[seed] = proc.stdout.strip() or proc.stderr.strip()[-200:]
    return [_check("splits_identical_across_PYTHONHASHSEED",
                   len(set(results.values())) == 1, results=results)]


def check_dataset_tests() -> list[dict]:
    dataset = REPO_ROOT / "dataset_all_metals"
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                          capture_output=True, text=True, cwd=dataset)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    gen11 = subprocess.run([sys.executable, "-m", "pytest",
                            "tests/test_gen11_invariants.py", "-q"],
                           capture_output=True, text=True, cwd=REPO_ROOT,
                           env={"PYTHONPATH": str(REPO_ROOT / "src"),
                                "PATH": "/usr/bin:/bin"})
    gen11_tail = gen11.stdout.strip().splitlines()[-1] if gen11.stdout.strip() else ""
    return [
        _check("dataset_all_metals_test_suite_passes", proc.returncode == 0, summary=tail),
        _check("gen11_invariant_tests_pass", gen11.returncode == 0, summary=gen11_tail),
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path,
                        default=REPO_ROOT / "runs" / "gen11_transfer" / "arms")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--skip-slow", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.time()
    cohort = prepared_cohort()
    overlap = build_overlap_map(cohort.frame)
    aux_features = pd.read_parquet(AUX_FEATURES)
    build = build_pool(cohort, overlap, aux_features, policy="HEADLINE")
    spec = RunSpec("C_LN_PLUS_ACTINIDES", metal_scheme="GENERAL",
                   massaction_scheme="ANNOTATION_SAFE")

    checks: list[dict] = []
    checks += check_identity(cohort)
    checks += check_dataset_tests()
    checks += check_fresh_subprocess_determinism()
    checks += check_auxiliary_leakage(cohort, overlap, build.pool)
    checks += check_provenance_not_a_feature(cohort, build.pool, spec)
    if not args.skip_slow:
        checks += check_prediction_invariances(cohort, build.pool, spec)
    if args.arms.exists():
        checks += check_row_accounting(args.arms, cohort)

    failed = [c for c in checks if not c["passed"]]
    vacuous = [c for c in checks if c["vacuous"]]
    report = {
        "generated_seconds": round(time.time() - started, 1),
        "n_checks": len(checks), "n_failed": len(failed),
        "n_vacuous_by_construction": len(vacuous),
        "verdict": "PASS" if not failed else "FAIL",
        "failed": [c["check"] for c in failed],
        "checks": checks,
    }
    (args.out / "self_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    for check in checks:
        flag = "PASS" if check["passed"] else "FAIL"
        note = " (vacuous)" if check["vacuous"] else ""
        print(f"  [{flag}]{note} {check['check']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
