#!/usr/bin/env python
"""Phase 0 — reproduce gen8's reference numbers in *this* environment, or stop.

The follow-up brief opens with a rule the project has learned the hard way: do not
compare a gen9 arm against a gen8 baseline that was produced by a different
effective pipeline.  gen8's own README records a sweep lost to exactly that, and
``environment.json`` records python 3.13.0 against the 3.13.13 running now.

So before any gen9 model is trained, this script rebuilds gen8's headline numbers
from the frozen out-of-fold predictions using the current interpreter, the current
libraries and gen9's own independently-recomputed curve geometry, and compares
them against the stored artefacts with declared tolerances.

Two families are checked.

**The k-shot frontier** — zero-shot, random / central / medoid one-shot, the
oracle, and the slope-repair arms that carry gen8's best 2-, 3- and 5-shot
numbers.  Reproduced by re-running ``evaluate_fewshot`` end to end, not by
re-reading gen8's detail parquet: re-reading would test the analysis code and
nothing else, and the pool/evaluation draw is precisely the thing that has broken
between processes before.

**The flattening pathology** — the median true and predicted slope on the
extractant and acid axes.  This is the number gen9 exists to move, so it has to be
established as reproducible before it is moved.

Exit code 1 if anything is outside tolerance.  Nothing downstream should run.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lanthanide_separation.gen8.adapters import default_adapters  # noqa: E402
from lanthanide_separation.gen8.evaluate import evaluate_fewshot  # noqa: E402
from lanthanide_separation.gen8.series import curve_statistics  # noqa: E402
from lanthanide_separation.gen8.slope_restore import build_slope_adapters  # noqa: E402
from lanthanide_separation.gen6.cohorts import seeded_group_kfold  # noqa: E402

COHORT_PATH = REPO_ROOT / "runs" / "gen7_architecture" / "cache" / "cohort.parquet"
OOF_PATH = REPO_ROOT / "runs" / "gen7_architecture" / "finalists" / "oof_predictions.parquet"
MEMBERSHIP_PATH = REPO_ROOT / "runs" / "gen9_shape" / "curves" / "curve_membership.parquet"
OUT = REPO_ROOT / "runs" / "gen9_shape" / "reproduction"
MODEL = "REC_ecfp_plus_recovered"
SEEDS = (104729, 130363, 155921, 196613, 262147)
COMMON_MIN_POOL = 5

#: What gen8 reported, and how far from it this rerun may land before gen9 stops.
#: 0.005 log units is roughly a fifth of the seed-to-seed spread of these numbers,
#: so it catches a pipeline change while tolerating float and library drift.
REFERENCES: dict[str, dict] = {
    "zero_shot_k0":            {"arm": "ZERO_SHOT",                              "k": 0, "expected": 1.0605, "tol": 0.005},
    "random_1shot":            {"arm": "OFFSET_K1@RANDOM",                       "k": 1, "expected": 0.7927, "tol": 0.005},
    "central_1shot":           {"arm": "OFFSET_K1@CENTRAL",                      "k": 1, "expected": 0.6964, "tol": 0.005},
    "medoid_1shot":            {"arm": "OFFSET_K1@MEDOID",                       "k": 1, "expected": 0.6916, "tol": 0.005},
    "oracle_1shot":            {"arm": "OFFSET_K1@ORACLE[OFFSET_K1]",            "k": 1, "expected": 0.5326, "tol": 0.005},
    "slope_medoid_1shot":      {"arm": "SLOPE_L_s1_K1@MEDOID",                   "k": 1, "expected": 0.6752, "tol": 0.005},
    "best_2shot":              {"arm": "SLOPE_L_s1_K3@MAX_PREDICTIVE_VARIANCE",  "k": 2, "expected": 0.5879, "tol": 0.005},
    "best_3shot":              {"arm": "SLOPE_L_s1_K3@CENTRAL_THEN_SPREAD",      "k": 3, "expected": 0.5230, "tol": 0.005},
    "best_5shot":              {"arm": "SLOPE_L_s1_K3@D_OPTIMAL",                "k": 5, "expected": 0.4743, "tol": 0.005},
    "random_2shot":            {"arm": "OFFSET_K1@RANDOM",                       "k": 2, "expected": 0.7018, "tol": 0.005},
}

#: The pathology, from ``runs/gen8_architecture/functional/slope_accuracy.csv``.
SLOPE_REFERENCES: dict[str, dict] = {
    "extractant_true_median": {"axis": "extractant", "column": "slope_true_median", "expected": 2.574, "tol": 0.05},
    "extractant_pred_median": {"axis": "extractant", "column": "slope_pred_median", "expected": 0.116, "tol": 0.05},
    "acid_true_median":       {"axis": "acid",       "column": "slope_true_median", "expected": 1.656, "tol": 0.05},
    "acid_pred_median":       {"axis": "acid",       "column": "slope_pred_median", "expected": 0.355, "tol": 0.05},
    "extractant_span_ratio":  {"axis": "extractant", "column": "span_ratio_median", "expected": 0.051, "tol": 0.02},
}

POLICIES = ("RANDOM", "CENTRAL", "MEDOID", "MAX_PREDICTIVE_VARIANCE",
            "CENTRAL_THEN_SPREAD", "D_OPTIMAL")


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #

def environment_record() -> dict:
    import numpy, pandas, scipy, sklearn
    versions = {"python": platform.python_version(), "numpy": numpy.__version__,
                "pandas": pandas.__version__, "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__}
    for name in ("torch", "rdkit"):
        try:
            versions[name] = __import__(name).__version__
        except Exception as error:                       # pragma: no cover - reporting only
            versions[name] = f"unavailable ({error})"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                             text=True).strip())
    except Exception:                                     # pragma: no cover
        commit, dirty = "unknown", True
    return {"platform": platform.platform(), "machine": platform.machine(),
            "library_versions": versions, "git_commit": commit, "git_dirty": dirty,
            "pythonhashseed": __import__("os").environ.get("PYTHONHASHSEED", "<unset>")}


def environment_drift() -> dict:
    """What changed against gen8's stamped environment, and whether it can matter."""
    stored = REPO_ROOT / "runs" / "gen8_architecture" / "environment.json"
    if not stored.exists():
        return {"available": False}
    gen8 = json.loads(stored.read_text()).get("library_versions", {})
    now = environment_record()["library_versions"]
    changed = {k: {"gen8": gen8.get(k), "gen9": now.get(k)}
               for k in sorted(set(gen8) | set(now)) if gen8.get(k) != now.get(k)}
    return {"available": True, "changed": changed, "n_changed": len(changed)}


# --------------------------------------------------------------------------- #
# the k-shot frontier
# --------------------------------------------------------------------------- #

def common_cohort(detail: pd.DataFrame, *, min_pool: int = COMMON_MIN_POOL) -> set:
    """gen8's Table B cohort: ligands with enough pool rows in every repeat/seed."""
    eligible = detail.groupby("extractant").apply(
        lambda block: bool((block["n_pool"] >= min_pool).all() and (block["n_eval"] >= 2).all()),
        include_groups=False)
    return set(eligible.index[eligible])


def arm_label(row) -> str:
    return "ZERO_SHOT" if row["adapter"] == "ZERO_SHOT_REF" else f"{row['adapter']}@{row['policy']}"


def macro_table(detail: pd.DataFrame, ligands: set | None = None) -> pd.DataFrame:
    work = detail.copy()
    work["arm"] = [arm_label(r) for _, r in work[["adapter", "policy"]].iterrows()]
    if ligands is not None:
        work = work[work["extractant"].isin(ligands)]
    per_ligand = work.groupby(["arm", "k", "extractant"])["mae"].mean().reset_index()
    out = per_ligand.groupby(["arm", "k"]).agg(
        mae=("mae", "mean"), n_ligands=("extractant", "nunique")).reset_index()
    return out


def run_kshot(seeds, repeats: int, out_dir: Path) -> pd.DataFrame:
    oof = pd.read_parquet(OOF_PATH)
    oof = oof[(oof["model"] == MODEL) & (oof["split_seed"].isin(seeds))].copy()
    if oof.empty:
        raise SystemExit(f"no OOF rows for {MODEL}")
    cohort = pd.read_parquet(COHORT_PATH)

    adapters = list(default_adapters())
    adapters += [a for a in build_slope_adapters(strengths=(1.0,))
                 if a.name in ("SLOPE_L_s1_K1", "SLOPE_L_s1_K3")]

    groups = cohort["tanimoto_cluster"].astype(str).to_numpy()
    target = cohort["log_D"].to_numpy(dtype=float)
    cache: dict = {}

    def fold_trainer(split_seed: int, fold: int):
        if split_seed not in cache:
            cache[split_seed] = list(seeded_group_kfold(groups, 5, int(split_seed)))
        train_index, _ = cache[split_seed][int(fold)]
        return cohort.iloc[train_index], target[train_index], 42 + int(fold) * 1009 + 9_999_991

    started = time.time()
    detail = evaluate_fewshot(
        oof, cohort, adapters, policies=POLICIES, with_oracle_for=("OFFSET_K1",),
        repeats=repeats, fold_trainer=fold_trainer, verbose=True)
    print(f"  evaluate_fewshot: {len(detail):,} records in {time.time() - started:.0f}s")
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(out_dir / "gen8_reproduction_detail.parquet", index=False)
    return detail


# --------------------------------------------------------------------------- #
# the flattening pathology
# --------------------------------------------------------------------------- #

def slope_pathology(seeds, membership: pd.DataFrame, *, min_points: int = 4) -> pd.DataFrame:
    """gen8's ``slope_accuracy.csv``, recomputed from the frozen OOF."""
    oof = pd.read_parquet(OOF_PATH)
    oof = oof[(oof["model"] == MODEL) & (oof["split_seed"].isin(seeds))]
    records = []
    for seed, block in oof.groupby("split_seed", sort=True):
        sub = membership[membership["row_id"].isin(set(block["row_id"]))]
        sub = sub[sub.groupby("curve_id")["row_id"].transform("size") >= min_points]
        truth = curve_statistics(block, sub, value_column="log_D")
        pred = curve_statistics(block, sub, value_column="prediction")
        merged = truth.merge(pred, on="curve_id", suffixes=("_true", "_pred"))
        merged["split_seed"] = seed
        records.append(merged)
    curves = pd.concat(records, ignore_index=True)
    rows = []
    for axis, block in curves.groupby("axis_label_true", sort=True):
        span_ratio = (block["y_span_pred"] / block["y_span_true"].replace(0, np.nan)).median()
        rows.append({
            "axis": axis, "n_curves": int(len(block)),
            "slope_true_median": float(block["slope_true"].median()),
            "slope_pred_median": float(block["slope_pred"].median()),
            "slope_mae": float((block["slope_pred"] - block["slope_true"]).abs().mean()),
            "span_ratio_median": float(span_ratio),
        })
    return pd.DataFrame.from_records(rows)


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--skip-kshot", action="store_true")
    args = parser.parse_args(argv)
    seeds = SEEDS[:max(1, min(5, args.seeds))]
    args.out.mkdir(parents=True, exist_ok=True)

    report: dict = {"seeds": list(seeds), "repeats": args.repeats,
                    "environment": environment_record(),
                    "environment_drift": environment_drift(), "checks": []}
    print("environment drift vs gen8:", json.dumps(report["environment_drift"].get("changed", {})))

    membership = pd.read_parquet(MEMBERSHIP_PATH)

    print("\n--- flattening pathology ---")
    slopes = slope_pathology(seeds, membership)
    slopes.to_csv(args.out / "slope_pathology.csv", index=False)
    print(slopes.to_string(index=False))
    lookup = slopes.set_index("axis")
    for name, spec in SLOPE_REFERENCES.items():
        got = float(lookup.loc[spec["axis"], spec["column"]]) if spec["axis"] in lookup.index else float("nan")
        ok = np.isfinite(got) and abs(got - spec["expected"]) <= spec["tol"]
        report["checks"].append({"family": "slope", "name": name, "expected": spec["expected"],
                                 "observed": got, "tol": spec["tol"], "pass": bool(ok)})
        print(f"  {'PASS' if ok else 'FAIL'}  {name:26s} expected {spec['expected']:+.3f}  got {got:+.3f}")

    if not args.skip_kshot:
        print("\n--- k-shot frontier ---")
        detail = run_kshot(seeds, args.repeats, args.out)
        cohort_ligands = common_cohort(detail)
        table = macro_table(detail, cohort_ligands)
        table.to_csv(args.out / "gen8_reproduction_table_b.csv", index=False)
        print(f"  common cohort: {len(cohort_ligands)} ligands")
        report["common_cohort_size"] = int(len(cohort_ligands))
        keyed = table.set_index(["arm", "k"])["mae"]
        for name, spec in REFERENCES.items():
            key = (spec["arm"], spec["k"])
            got = float(keyed.get(key, np.nan))
            ok = np.isfinite(got) and abs(got - spec["expected"]) <= spec["tol"]
            report["checks"].append({"family": "kshot", "name": name, "arm": spec["arm"],
                                     "k": spec["k"], "expected": spec["expected"],
                                     "observed": got, "tol": spec["tol"], "pass": bool(ok)})
            print(f"  {'PASS' if ok else 'FAIL'}  {name:22s} {spec['arm']:38s} k={spec['k']} "
                  f"expected {spec['expected']:.4f}  got {got:.4f}")

    failed = [c for c in report["checks"] if not c["pass"]]
    report["n_checks"] = len(report["checks"])
    report["n_failed"] = len(failed)
    report["pass"] = not failed
    (args.out / "reproduction.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{report['n_checks'] - report['n_failed']}/{report['n_checks']} checks passed")
    if failed:
        print("FAILED — gen9 must not proceed until these reproduce:")
        for check in failed:
            print(f"  - {check['name']}: expected {check['expected']}, got {check['observed']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
