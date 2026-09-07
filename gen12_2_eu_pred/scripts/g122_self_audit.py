#!/usr/bin/env python
"""Re-check every pre-registered invariant against the artefacts that were produced.

The test suite proves the invariants hold for freshly built objects.  This proves they
hold for the files on disk — which is what a reader of the decision report is trusting.
Exit code is non-zero on any failure.

Usage::

    PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_self_audit.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen122 import coordination, paths  # noqa: E402

TARGET = "log_D"


def row_key(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.sort_values(columns)
    return hashlib.blake2b("|".join(
        "|".join(str(v) for v in row) for row in ordered[columns].to_numpy()).encode(),
        digest_size=16).hexdigest()


def main() -> int:
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str = ""):
        checks.append({"check": name, "status": "pass" if ok else "FAIL",
                       "pass": bool(ok), "detail": detail})

    def skip(name: str, why: str):
        # A missing artefact must be visible as a skip.  Silently dropping the check turns
        # "N/N pass" into a statement about which files happen to exist.
        checks.append({"check": name, "status": "skipped", "pass": True, "detail": why})

    # ---- 1: Gen12 is untouched --------------------------------------------- #
    gen12_manifest = json.loads((paths.GEN12_MANIFESTS / "manifest.json").read_text())
    drift = []
    for name, entry in gen12_manifest["artefacts"].items():
        path = paths.GEN12_ROOT / name
        if not path.exists():
            drift.append(f"{name} missing")
            continue
        if "blake2b_128" not in entry:      # Gen12 records figures by size only
            if entry.get("bytes") not in (None, path.stat().st_size):
                drift.append(f"{name} (size)")
            continue
        digest = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
        if digest != entry["blake2b_128"]:
            drift.append(name)
    record("every Gen12 artefact is byte-identical to its own manifest", not drift,
           "; ".join(drift[:4]))

    # ---- 2: the fold plan and the cohort ------------------------------------ #
    plan = json.loads(paths.GEN12_FOLD_PLAN_B.read_text())
    manifest_path = paths.PREDICTION_DIR / "level" / "LVL_MEAN" / "run_manifest.json"
    if manifest_path.exists():
        run = json.loads(manifest_path.read_text())
        record("the level ladder ran on Gen12's frozen fold plan",
               run["fold_plan_matches_gen12"]["folds_checked"] == len(plan["folds"]),
               f"{run['fold_plan_matches_gen12']['folds_checked']} folds checked")
        record("the level ladder ran against Gen12's cohort fingerprint",
               run["cohort_fingerprint"] == plan["cohort_fingerprint"],
               run["cohort_fingerprint"])
        record("the SMARTS digest recorded by the ladder is the current one",
               run["coordination_spec_sha256"] == coordination.spec_digest())

    # ---- 3: the feature matrix ---------------------------------------------- #
    table = pd.read_parquet(paths.FEATURE_DIR / "coordination_descriptors.parquet")
    audit = json.loads((paths.FEATURE_DIR / "coordination_audit.json").read_text())
    record("the coordination matrix matches its recorded content hash",
           audit["feature_matrix_blake2b"] == paths.blake2b_of_frame(table),
           audit["feature_matrix_blake2b"])
    record("the SMARTS specification matches its recorded digest",
           audit["spec_sha256"] == coordination.spec_digest(), audit["spec_sha256"][:16])
    record("every coordination descriptor is finite",
           bool(np.isfinite(table.to_numpy(dtype=float)).all()))
    record("no coordination column has a Gen12-style heavy tail",
           audit["max_max_over_sd"] < 100.0, f"largest |x|/sd {audit['max_max_over_sd']:.2f}")

    # ---- 4: level arms are matched ------------------------------------------ #
    for tag in ("LVL_COND_RESIDUAL", "LVL_MEAN_xgboost", "LVL_MEAN_catboost"):
        directory = paths.PREDICTION_DIR / "level" / tag.replace("LVL_MEAN_", "LVL_MEAN/")
        alt = paths.PREDICTION_DIR / "level" / tag
        present = directory.exists() or alt.exists()
        record(f"secondary arm set {tag} is present", present,
               "" if present else "not run")

    directory = paths.PREDICTION_DIR / "level" / "LVL_MEAN"
    if directory.exists():
        keys, truths, clipped = set(), [], 0
        for path in sorted(directory.glob("*.parquet")):
            frame = pd.read_parquet(path)
            keys.add(row_key(frame, ["split_seed", "fold", "extractant"]))
            truths.append(frame.sort_values(["split_seed", "fold", "extractant"])
                          ["alpha_true"].to_numpy())
        for path in sorted(directory.glob("*__selection.csv")):
            clipped += int(pd.read_csv(path)["n_clipped"].sum())
        record("all level arms are scored on identical evaluation units", len(keys) == 1,
               f"{len(truths)} arms, {len(keys)} distinct unit sets")
        record("all level arms see the same level target",
               all(np.allclose(v, truths[0]) for v in truths))
        record("no level prediction was clipped", clipped == 0, f"{clipped} clipped values")

    # ---- 5: full arms are matched to Gen12's own rows ----------------------- #
    full = paths.PREDICTION_DIR / "full"
    if full.exists() and list(full.glob("*.parquet")):
        keys = set()
        for path in sorted(full.glob("*.parquet")):
            keys.add(row_key(pd.read_parquet(path), ["split_seed", "fold", "row_id"]))
        gen12 = pd.read_parquet(paths.GEN12_PREDICTIONS / "B_ablation"
                                / "ABL_D_PLUS_LIG2D.parquet")
        keys.add(row_key(gen12, ["split_seed", "fold", "row_id"]))
        record("Gen12.2 full arms are scored on Gen12's exact query rows", len(keys) == 1,
               f"{len(keys)} distinct row sets including Gen12's")

    # ---- 6: the subgroup is structural -------------------------------------- #
    subgroup = pd.read_csv(paths.MANIFEST_DIR / "multi_arm_subgroup.csv", index_col=0)
    rule = table["coord__arm__local_donor_cluster_count"].reindex(subgroup.index) >= 2
    record("MULTI_ARM membership is exactly the frozen structural rule",
           bool(subgroup["MULTI_ARM"].astype(bool).equals(rule)),
           f"{int(subgroup['MULTI_ARM'].sum())} multi-arm extractants")

    # ---- 7: the level definition was settled before the ladder -------------- #
    choice = json.loads((paths.MANIFEST_DIR / "level_definition_choice.json").read_text())
    record("the primary level definition is the pre-registered one",
           choice["decision"]["primary_level_definition"] == "LVL_MEAN",
           choice["decision"]["reason"])
    record("the study records what the un-amended rule would have chosen",
           choice["decision"]["original_rule_would_have_chosen"] == "LVL_COND_RESIDUAL")

    # ---- 8: few-shot draws ---------------------------------------------------- #
    membership_path = paths.METRIC_DIR / "one_shot" / "fewshot_support_membership.parquet"
    if not membership_path.exists():
        skip("support draws are identical across arms and adapters", "one-shot not run")
    if membership_path.exists():
        membership = pd.read_parquet(membership_path)
        sizes = membership.assign(k_actual=membership["support_rows"].astype(str).apply(
            lambda s: 0 if not s else len(s.split(";"))))
        record("support-set size equals k", bool((sizes["k_actual"] == sizes["k"]).all()))
        per_draw = membership[membership["k"] > 0].groupby(
            ["k", "split_seed", "extractant", "repeat"])["support_rows"].nunique()
        record("support draws are identical across arms and adapters",
               bool((per_draw == 1).all()), f"{int((per_draw > 1).sum())} draws differ")

    table_out = pd.DataFrame(checks)
    table_out.to_csv(paths.MANIFEST_DIR / "self_audit.csv", index=False)
    print(table_out[["check", "status", "detail"]].to_string(index=False))
    failed = int((table_out["status"] == "FAIL").sum())
    skipped = int((table_out["status"] == "skipped").sum())
    print(f"\n{len(table_out) - failed - skipped}/{len(table_out)} checks pass, "
          f"{skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
