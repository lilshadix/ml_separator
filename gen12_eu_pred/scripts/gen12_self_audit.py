#!/usr/bin/env python
"""Re-check every pre-registered invariant against the artefacts that were produced.

The test suite proves the invariants hold for freshly built objects.  This proves
they hold for the files on disk — which is the thing a reader of the decision
report is actually trusting.  Exit code is non-zero on any failure.

Usage::

    PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_self_audit.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen12eu import paths  # noqa: E402

TARGET = "log_D"


def _predictions(design: str) -> dict[str, pd.DataFrame]:
    out = {}
    for directory in (paths.PREDICTION_DIR / design,
                      paths.PREDICTION_DIR / f"{design}_ablation",
                      paths.PREDICTION_DIR / f"{design}_multiln"):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.parquet")):
            if path.stem == "similarity":
                continue
            frame = pd.read_parquet(path)
            if "prediction" in frame.columns:
                out[f"{directory.name}/{frame['arm'].iloc[0]}"] = frame
    return out


def main() -> int:
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str = ""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    cohort = pd.read_parquet(paths.MANIFEST_DIR / "cohort.parquet")
    audit = json.loads((paths.MANIFEST_DIR / "data_audit.json").read_text())

    record("cohort target finite", bool(np.isfinite(cohort[TARGET]).all()))
    record("cohort row ids unique", not cohort["row_id"].duplicated().any())
    record("one fingerprint per extractant",
           bool((cohort.groupby("extractant")[[f"ecfp_{i}" for i in range(2048)]]
                 .nunique().max(axis=1) <= 1).all()))

    frames = _predictions("B")
    if frames:
        keys = {}
        for arm, frame in frames.items():
            ordered = frame.sort_values(["split_seed", "fold", "row_id"])
            keys[arm] = hashlib.blake2b(
                "|".join(f"{s}:{f}:{r}" for s, f, r in zip(
                    ordered["split_seed"], ordered["fold"], ordered["row_id"])).encode(),
                digest_size=16).hexdigest()
        record("all design-B arms scored on identical rows", len(set(keys.values())) == 1,
               f"{len(frames)} arms, {len(set(keys.values()))} distinct row sets")

        truth = {}
        for arm, frame in frames.items():
            ordered = frame.sort_values(["split_seed", "fold", "row_id"])
            truth[arm] = ordered[TARGET].to_numpy()
        base = next(iter(truth.values()))
        record("target identical across arms",
               all(np.allclose(v, base) for v in truth.values()))

        # A held-out extractant may never appear in its own fold's training rows.
        plan = json.loads((paths.PREDICTION_DIR / "B" / "fold_plan.json").read_text())
        extractant_of = cohort.set_index("row_id")["extractant"].to_dict()
        chemotype_of = cohort.set_index("row_id")["chemotype"].to_dict()
        all_rows = set(cohort["row_id"])
        leaks = 0
        for fold in plan["folds"]:
            test_rows = set(fold["test_row_ids"])
            train_rows = all_rows - test_rows
            test_chem = {chemotype_of[r] for r in test_rows}
            train_chem = {chemotype_of[r] for r in train_rows}
            leaks += len(test_chem & train_chem)
        record("no chemotype crosses a design-B fold", leaks == 0, f"{leaks} crossings")

        similarity = pd.read_parquet(paths.PREDICTION_DIR / "B" / "similarity.parquet")
        record("chemotype hold-out caps similarity below 0.70",
               float(similarity["max_train_tanimoto"].max()) < 0.70,
               f"max {similarity['max_train_tanimoto'].max():.4f}")
        record("band labels agree with the frozen thresholds",
               bool(((similarity["band"] == "far") == (similarity["max_train_tanimoto"] <= 0.4)).all()))

    support = paths.METRIC_DIR / "B" / "fewshot_support_membership.parquet"
    if support.exists():
        membership = pd.read_parquet(support)
        detail = pd.read_parquet(paths.METRIC_DIR / "B" / "fewshot_detail.parquet")
        sizes = membership.assign(k_actual=membership["support_rows"].astype(str).apply(
            lambda s: 0 if not s else len(s.split(";"))))
        record("support-set size equals k", bool((sizes["k_actual"] == sizes["k"]).all()))
        per_arm = membership[membership["k"] > 0].groupby(
            ["k", "split_seed", "extractant", "repeat"])["support_rows"].nunique()
        record("support draws identical across arms and adapters", bool((per_arm == 1).all()),
               f"{int((per_arm > 1).sum())} draws differ")
        record("query counts never exceed the extractant's rows minus k",
               bool((detail["n_query"] <= detail["n_rows"] - detail["k"]).all()))

    multiln = paths.METRIC_DIR / "B_multiln" / "inventory.json"
    if multiln.exists():
        inventory = json.loads(multiln.read_text())
        strict = [p for p in inventory["per_policy"] if p["policy"] != "LEAKY_NO_FILTER"]
        record("no Eu test extractant survives in a strict auxiliary pool",
               all(p.get("aux_test_extractant_overlap", 0) == 0 for p in strict))
        leaky = [p for p in inventory["per_policy"] if p["policy"] == "LEAKY_NO_FILTER"]
        record("the leaky control does leak, as designed",
               bool(leaky and leaky[0].get("aux_test_extractant_overlap", 0) > 0),
               f"{leaky[0]['aux_test_extractant_overlap']:.1f} test extractants per fold"
               if leaky else "")

    record("data audit reports no fatal hazard", not audit["hazards"]["fatal"],
           "; ".join(audit["hazards"]["fatal"]))

    table = pd.DataFrame(checks)
    table.to_csv(paths.MANIFEST_DIR / "self_audit.csv", index=False)
    print(table.to_string(index=False))
    failed = int((~table["pass"]).sum())
    print(f"\n{len(table) - failed}/{len(table)} checks pass")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
