#!/usr/bin/env python
"""PHASE 0 — freeze and reproduce.  Nothing else runs until this passes.

Consolidates every check the brief lists into one record,
``runs/gen10_final/reproduction/phase0.json``, and exits non-zero on the first
failure:

1. the gen9 manifest verifies with zero drift (154 artefacts);
2. gen9's own Phase-0 script reproduces all 15 gen8 references in this environment;
3. the cohort fingerprint, fold plan and curve geometry are the frozen ones;
4. the gen10 classes nest the three gen9 arms at the float floor;
5. determinism: the same configuration twice in one process and twice in fresh
   subprocesses under different ``PYTHONHASHSEED`` values;
6. the test suites — gen7/8/9 and gen10's metamorphic and regression tests — pass.

Each item records what it compared, not just that it passed, so the self-audit
can quote the floor rather than assert it is small.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen7.harness import DEFAULT_SEEDS, build_folds, FoldContext  # noqa: E402
from lanthanide_separation.gen9.train import (  # noqa: E402
    FrozenExtraTrees, ShapeRecomposed, load_membership,
)
from lanthanide_separation.gen10.architectures import (  # noqa: E402
    MonolithModel, RecomposedModel, prime_raw_cache,
)
from lanthanide_separation.gen10.querycurves import assert_partition_closure  # noqa: E402
from lanthanide_separation.gen10.runner import (  # noqa: E402
    COHORT_FINGERPRINT, DETERMINISM_TOLERANCE, determinism_probe, prepared_cohort,
)
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN  # noqa: E402

OUT = REPO_ROOT / "runs" / "gen10_final" / "reproduction"
PY = sys.executable


def step(record: dict, name: str, ok: bool, **detail) -> None:
    record["checks"].append({"check": name, "pass": bool(ok), **detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail if not ok else ''}", flush=True)
    if not ok:
        (OUT / "phase0.json").write_text(json.dumps(record, indent=2, default=str))
        raise SystemExit(f"Phase 0 failed at: {name}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "checks": [], "environment": {}}
    import numpy, pandas, scipy, sklearn
    record["environment"] = {"python": sys.version.split()[0], "numpy": numpy.__version__,
                             "pandas": pandas.__version__, "scipy": scipy.__version__,
                             "scikit-learn": sklearn.__version__}
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))

    # 1. gen9 manifest
    result = subprocess.run([PY, "scripts/gen9_manifest.py", "--verify"], cwd=REPO_ROOT,
                            env=env, capture_output=True, text=True)
    step(record, "gen9 manifest verifies with zero drift", result.returncode == 0,
         output=result.stdout.strip().splitlines()[-1:] if result.stdout else result.stderr[-500:])

    # 2. gen9's Phase-0 reproduction, rerun into gen10's directory
    rerun = OUT / "gen9_reproduce_rerun" / "reproduction.json"
    if not rerun.exists():
        result = subprocess.run([PY, "scripts/gen9_reproduce_gen8.py", "--out", str(rerun.parent)],
                                cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        (OUT / "gen9_reproduce_rerun.log").write_text(result.stdout + result.stderr)
    payload = json.loads(rerun.read_text()) if rerun.exists() else {}
    checks = payload.get("checks", payload.get("results", []))
    passed = [c for c in checks if c.get("pass", c.get("ok", False))]
    step(record, "gen9 Phase-0 reproduces all gen8 references",
         bool(checks) and len(passed) == len(checks),
         n_checks=len(checks), n_passed=len(passed))

    # 3. frozen cohort, folds, curves
    cohort = prepared_cohort()
    prime_raw_cache(cohort)
    membership = load_membership()
    step(record, "cohort fingerprint is frozen", cohort.fingerprint == COHORT_FINGERPRINT,
         fingerprint=cohort.fingerprint, n_rows=int(len(cohort.frame)))
    step(record, "curve geometry is gen9's (1,176 curves / 7,207 memberships)",
         membership["curve_id"].nunique() == 1176 and len(membership) == 7207,
         n_curves=int(membership["curve_id"].nunique()), n_memberships=int(len(membership)))
    closure_ok, worst = True, 0
    for seed in DEFAULT_SEEDS:
        for fold in build_folds(cohort.frame, seed):
            audit = assert_partition_closure(
                membership, cohort.frame.iloc[fold.test_index]["row_id"].astype(str),
                name=f"s{seed}f{fold.fold}")
            worst = max(worst, audit["n_curves_straddling"])
    step(record, "no curve straddles a fold boundary on any of 25 folds", worst == 0,
         max_straddling=worst)

    # 4. exact nesting on fold 0 of the first seed
    frame = cohort.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)
    fold = build_folds(frame, DEFAULT_SEEDS[0])[0]
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
    context = FoldContext(fold=fold, cohort=cohort, feature_columns=(), model_seed=fold.model_seed)
    y = target[fold.train_index]
    pairs = {
        "REC_ecfp_plus_recovered": (FrozenExtraTrees(), MonolithModel(representation_name="NONE")),
        "GEN9_REL_MONOLITH": (FrozenExtraTrees(with_relative=True, membership=membership),
                              MonolithModel(representation_name="GEN9", gen9_compat=True)),
        "GEN9_SHAPE_RECOMPOSED": (ShapeRecomposed(membership=membership),
                                  RecomposedModel(representation_name="GEN9", gen9_compat=True)),
    }
    for name, (old, new) in pairs.items():
        a = old.fit_predict(train, y, test, context)
        b = new.fit(train, y, context).predict(test)
        delta = float(np.abs(a - b).max())
        step(record, f"gen10 nests {name} at the float floor", delta <= DETERMINISM_TOLERANCE,
             max_abs_delta=delta, tolerance=DETERMINISM_TOLERANCE)

    # 5a. determinism in-process
    for name, factory in (("REL_MONOLITH", lambda: MonolithModel(representation_name="GEN9",
                                                                 gen9_compat=True)),
                          ("SHAPE_RECOMPOSED", lambda: RecomposedModel(representation_name="GEN9",
                                                                       gen9_compat=True))):
        probe = determinism_probe(factory, cohort)
        step(record, f"{name} identical on two in-process refits", probe["within_tolerance"],
             **probe)

    # 5b. determinism across interpreters with different hash seeds
    code = f"""
import sys, json, numpy as np
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
from lanthanide_separation.gen7.harness import load_cohort, build_folds, FoldContext
from lanthanide_separation.gen10.architectures import RecomposedModel, prime_raw_cache
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN
cohort = load_cohort(); prime_raw_cache(cohort); frame = cohort.frame
t = frame[LEVEL_TARGET_COLUMN].to_numpy(float); fold = build_folds(frame, {DEFAULT_SEEDS[0]})[0]
train = frame.iloc[fold.train_index]; test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
ctx = FoldContext(fold=fold, cohort=cohort, feature_columns=(), model_seed=fold.model_seed)
p = RecomposedModel(representation_name="GEN9", gen9_compat=True).fit(train, t[fold.train_index], ctx).predict(test)
print(json.dumps([float(v) for v in p]))
"""
    outputs = {}
    for hashseed in ("0", "4242"):
        sub_env = {"PYTHONHASHSEED": hashseed, "PATH": os.environ.get("PATH", "")}
        out = subprocess.check_output([PY, "-c", code], env=sub_env, cwd=REPO_ROOT, text=True)
        outputs[hashseed] = np.asarray(json.loads(out.strip().splitlines()[-1]))
    delta = float(np.abs(outputs["0"] - outputs["4242"]).max())
    step(record, "SHAPE_RECOMPOSED identical across PYTHONHASHSEED 0 / 4242 subprocesses",
         delta <= DETERMINISM_TOLERANCE, max_abs_delta=delta)

    # 6. the test suites
    tests = ["tests/test_gen10_phase0.py", "tests/test_gen10_regressions.py",
             "tests/test_gen9_acquisition.py", "tests/test_gen9_curve_objective.py",
             "tests/test_gen9_reproducibility.py", "tests/test_gen9_series_adapter.py",
             "tests/test_gen8_harness.py", "tests/test_gen7_harness.py"]
    result = subprocess.run([PY, "-m", "pytest", *tests, "-q", "-p", "no:cacheprovider"],
                            cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    (OUT / "pytest_phase0.log").write_text(result.stdout + result.stderr)
    summary = [l for l in result.stdout.splitlines() if "passed" in l or "failed" in l]
    step(record, "gen7/8/9/10 test suites pass", result.returncode == 0, summary=summary[-1:])

    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["all_pass"] = all(c["pass"] for c in record["checks"])
    (OUT / "phase0.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"\nPHASE 0: {'PASS' if record['all_pass'] else 'FAIL'} "
          f"({len(record['checks'])} checks)")
    return 0 if record["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
