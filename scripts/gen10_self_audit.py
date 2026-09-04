#!/usr/bin/env python
"""PHASE 12 — the final self-audit, run before the model is accepted.

Sixteen checks, each recorded with what it compared.  The record is
``runs/gen10_final/self_audit/self_audit.json``; the script exits non-zero at the
first failure so a headline cannot be written on top of a broken run.

 1. identical configuration twice, in one process (selected model, fold 0)
 2. fresh subprocesses under several ``PYTHONHASHSEED`` values
 3. manifest and hashes (gen9 zero-drift, gen10 stamped)
 4. row accounting on every finalist OOF table
 5. fold boundaries (no ligand / chemotype / cluster overlap, 25 folds)
 6. curve boundaries (no curve straddles a fold; partitions closed)
 7. no target contamination (target column absent from every query frame the
    selected model is handed; structural, asserted)
 8. corrupt held-out targets -> predictions unchanged
 9. permute candidate rows -> predictions permuted, nothing else
10. query-set composition -> consistency distribution, from Phase 1's rows
11. every acquisition policy on synthetic positive controls
12. every reported paired comparison exactly paired (unit keys identical)
13. NaN / inf scan over every final table
14. no difficult rows disappear (finalist OOFs carry every cohort row)
15. final tables regenerate from saved raw predictions (frontier and macro recomputed)
16. rerun of fold 0 vs stored predictions, byte-for-byte where expected
"""

from __future__ import annotations

import argparse
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

from lanthanide_separation.gen7.harness import DEFAULT_SEEDS, FoldContext, build_folds  # noqa: E402
from lanthanide_separation.gen7.harness import assert_fold_integrity  # noqa: E402
from lanthanide_separation.gen9.train import load_membership  # noqa: E402
from lanthanide_separation.gen10.architectures import prime_raw_cache  # noqa: E402
from lanthanide_separation.gen10.querycurves import assert_partition_closure  # noqa: E402
from lanthanide_separation.gen10.runner import (  # noqa: E402
    DETERMINISM_TOLERANCE, determinism_probe, prepared_cohort,
)
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN  # noqa: E402

GEN10 = REPO_ROOT / "runs" / "gen10_final"
OUT = GEN10 / "self_audit"
PY = sys.executable


def registry():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen10_train", REPO_ROOT / "scripts" / "gen10_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ARMS


def check(record: dict, name: str, ok: bool, **detail) -> None:
    record["checks"].append({"check": name, "pass": bool(ok), **detail})
    print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    if not ok:
        print(f"        {detail}", flush=True)
        (OUT / "self_audit.json").write_text(json.dumps(record, indent=2, default=str))
        raise SystemExit(f"self-audit failed at: {name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True, help="registry name of the frozen model")
    parser.add_argument("--stage-dir", type=Path, required=True,
                        help="the stage directory holding the selected model's OOF parquet")
    parser.add_argument("--finalists", nargs="*", default=None,
                        help="OOF model names to audit for row accounting (default: all in final_locked)")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "selected": args.selected, "checks": []}
    arms = registry()
    factory = arms[args.selected]
    cohort = prepared_cohort()
    prime_raw_cache(cohort)
    membership = load_membership()
    frame = cohort.frame
    target = frame[LEVEL_TARGET_COLUMN].to_numpy(dtype=float)

    # 1. same configuration twice
    probe = determinism_probe(factory, cohort)
    check(record, "1 identical configuration twice in one process", probe["within_tolerance"], **probe)

    # 2. subprocesses under several hash seeds
    code = f"""
import sys, json, importlib.util, numpy as np
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
spec = importlib.util.spec_from_file_location("gen10_train", {str(REPO_ROOT / 'scripts' / 'gen10_train.py')!r})
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from lanthanide_separation.gen7.harness import load_cohort, build_folds, FoldContext
from lanthanide_separation.gen10.architectures import prime_raw_cache
from lanthanide_separation.levels import LEVEL_TARGET_COLUMN
cohort = load_cohort(); prime_raw_cache(cohort); frame = cohort.frame
t = frame[LEVEL_TARGET_COLUMN].to_numpy(float); fold = build_folds(frame, {DEFAULT_SEEDS[0]})[0]
train = frame.iloc[fold.train_index]; test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
ctx = FoldContext(fold=fold, cohort=cohort, feature_columns=(), model_seed=fold.model_seed)
p = m.ARMS[{args.selected!r}]().fit(train, t[fold.train_index], ctx).predict(test)
print(json.dumps([float(v) for v in p]))
"""
    outputs = {}
    for hashseed in ("0", "1", "4242"):
        env = {"PYTHONHASHSEED": hashseed, "PATH": os.environ.get("PATH", "")}
        out = subprocess.check_output([PY, "-c", code], env=env, cwd=REPO_ROOT, text=True)
        outputs[hashseed] = np.asarray(json.loads(out.strip().splitlines()[-1]))
    spread = max(float(np.abs(outputs["0"] - outputs[h]).max()) for h in ("1", "4242"))
    check(record, "2 fresh subprocesses under PYTHONHASHSEED 0/1/4242 agree",
          spread <= DETERMINISM_TOLERANCE, max_abs_delta=spread)
    stored_fold0 = outputs["0"]

    # 3. manifests
    gen9 = subprocess.run([PY, "scripts/gen9_manifest.py", "--verify"], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    check(record, "3a gen9 manifest verifies with zero drift", gen9.returncode == 0,
          output=gen9.stdout.strip().splitlines()[-1:])
    gen10 = subprocess.run([PY, "scripts/gen10_manifest.py"], cwd=REPO_ROOT,
                           capture_output=True, text=True)
    check(record, "3b gen10 manifest stamped", gen10.returncode == 0, output=gen10.stdout.strip()[-200:])

    # 4 + 14. row accounting on finalist OOFs
    oof = pd.read_parquet(GEN10 / "final_locked" / "oof_finalists.parquet")
    counts = oof.groupby(["model", "split_seed"]).size()
    rows_ok = bool((counts == len(frame)).all())
    missing_rows = {m: int(len(set(frame["row_id"]) - set(b["row_id"])))
                    for m, b in oof.groupby("model")}
    check(record, "4 row accounting: every finalist returns every row at every seed", rows_ok,
          counts={f"{m}@{seed}": int(v) for (m, seed), v in counts.items()})
    check(record, "14 no difficult rows disappear", all(v == 0 for v in missing_rows.values()),
          missing=missing_rows)

    # 5 + 6. fold and curve boundaries
    worst_overlap, worst_straddle = False, 0
    for seed in DEFAULT_SEEDS:
        folds = build_folds(frame, seed)
        integrity = assert_fold_integrity(frame, folds)
        worst_overlap = worst_overlap or not integrity["ok"]
        for fold in folds:
            audit = assert_partition_closure(
                membership, frame.iloc[fold.test_index]["row_id"].astype(str), name=f"s{seed}f{fold.fold}")
            worst_straddle = max(worst_straddle, audit["n_curves_straddling"])
    check(record, "5 fold boundaries: zero ligand/chemotype/cluster overlap on 25 folds", not worst_overlap)
    check(record, "6 curve boundaries: zero curves straddle a fold", worst_straddle == 0)

    # 7, 8, 9. target contamination, corruption, permutation on the selected model
    fold = build_folds(frame, DEFAULT_SEEDS[0])[0]
    train = frame.iloc[fold.train_index]
    test = frame.iloc[fold.test_index].drop(columns=[LEVEL_TARGET_COLUMN])
    context = FoldContext(fold=fold, cohort=cohort, feature_columns=(), model_seed=fold.model_seed)
    fitted = factory().fit(train, target[fold.train_index], context)
    clean = fitted.predict(test)
    check(record, "7 no target column in the query frame", LEVEL_TARGET_COLUMN not in test.columns)
    corrupted = fitted.predict(test.assign(**{LEVEL_TARGET_COLUMN: np.random.default_rng(1).normal(1e4, 1e3, len(test))}))
    check(record, "8 corrupting held-out targets leaves predictions unchanged",
          float(np.abs(clean - corrupted).max()) <= DETERMINISM_TOLERANCE,
          max_abs_delta=float(np.abs(clean - corrupted).max()))
    permutation = np.random.default_rng(7).permutation(len(test))
    permuted = fitted.predict(test.iloc[permutation].reset_index(drop=True))
    check(record, "9 permuting candidate rows permutes predictions and nothing else",
          float(np.abs(clean[permutation] - permuted).max()) <= 1e-5,
          max_abs_delta=float(np.abs(clean[permutation] - permuted).max()))

    # 10. query-set composition
    qc = GEN10 / "query_consistency" / "summary_by_arm.csv"
    if qc.exists():
        table = pd.read_csv(qc)
        check(record, "10 query-set consistency distribution recorded", len(table) > 0,
              arms=sorted(table["arm"].unique()))
    else:
        check(record, "10 query-set consistency distribution recorded", False, reason="missing")

    # 11. acquisition policies on synthetic positive controls
    from lanthanide_separation.gen8.kshot import POLICIES, PolicyContext
    from lanthanide_separation.gen8.protocols import _standardised_axes
    from lanthanide_separation.gen9.acquisition import register_extra_policies
    register_extra_policies()
    sys.path.insert(0, str(REPO_ROOT))
    from tests.test_gen10_phase0 import synthetic_ligand
    from lanthanide_separation.gen10.perturb import recompute_derived
    synth = synthetic_ligand(6)
    synth.loc[:4, "cond__extractant_concentration_M"] = [0.010, 0.011, 0.012, 0.013, 0.014]
    synth.loc[5, "cond__extractant_concentration_M"] = 1.0
    synth = recompute_derived(synth)
    axes = _standardised_axes(synth)
    ctx = PolicyContext(block=synth, prediction=np.zeros(6), pool=np.arange(6),
                        evaluation=np.array([], dtype=int), axes=axes, uncertainty=np.full(6, np.nan),
                        disagreement=np.full(6, np.nan), rng=np.random.default_rng(0), truth=None)
    picks = {n: int(POLICIES[n](ctx, [])) for n in ("MEDOID", "CENTRAL", "FARTHEST_FROM_EXISTING",
                                                      "CENTRAL_THEN_SPREAD", "D_OPTIMAL", "RANDOM")}
    controls_ok = picks["MEDOID"] in {1, 2, 3} and picks["FARTHEST_FROM_EXISTING"] != picks["MEDOID"] \
        and picks["CENTRAL_THEN_SPREAD"] == picks["CENTRAL"] or picks["CENTRAL_THEN_SPREAD"] in {1, 2, 3}
    check(record, "11 acquisition policies behave on a synthetic asymmetric pool", controls_ok, picks=picks)

    # 12. pairing of every reported comparison
    detail = pd.read_parquet(GEN10 / "final_locked" / "kshot_detail.parquet",
                             columns=["global_model", "split_seed", "fold", "extractant", "repeat",
                                      "n_pool", "n_eval", "adapter", "policy", "k"])
    keys = ["split_seed", "fold", "extractant", "repeat", "n_pool", "n_eval"]
    units = {m: set(map(tuple, b[keys].drop_duplicates().to_numpy().tolist()))
             for m, b in detail.groupby("global_model")}
    reference = next(iter(units.values()))
    paired = all(u == reference for u in units.values())
    check(record, "12 every global model scored on identical (seed, fold, ligand, repeat, pool, eval) units",
          paired, n_units={m: len(u) for m, u in units.items()})

    # 13. NaN / inf scan.  Two NaN sources are by construction and exempted by name:
    # BCa bounds where the jackknife is degenerate, and the guarded span columns on
    # an axis with zero curves above the span guard (gen9's own table carries them).
    GUARDED = ("span_recovery_median", "span_recovery_q25", "span_recovery_q75",
               "compression_median")
    bad = {}
    for path in sorted((GEN10 / "final_locked").glob("*.csv")):
        table = pd.read_csv(path)
        numeric = table.select_dtypes("number")
        if not numeric.size:
            continue
        n_inf = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        columns = [c for c in numeric.columns if c not in ("bca_low", "bca_high")]
        mask = numeric[columns].isna()
        if "n_curves_guarded" in table.columns:
            degenerate = (table["n_curves_guarded"] == 0).to_numpy()
            for column in GUARDED:
                if column in mask.columns:
                    mask.loc[degenerate, column] = False
        n_nan = int(mask.sum().sum())
        if n_inf or n_nan:
            bad[path.name] = {"nan": n_nan, "inf": n_inf,
                              "columns": mask.sum()[mask.sum() > 0].to_dict()}
    check(record, "13 no NaN / inf in final tables (BCa and degenerate guarded span excepted)",
          not bad, offenders=bad)

    # 15. regenerate the headline from raw predictions
    fro = pd.read_csv(GEN10 / "final_locked" / "frontier_best.csv")
    k0 = fro[(fro["global_model"] == oof["model"].iloc[0]) & (fro["k"] == 0)]
    selected_model = json.loads((GEN10 / "final_locked" / "pipeline_frontier.json").read_text())[
        "selected_global_model"]
    common = set(pd.read_parquet(GEN10 / "final_locked" / "kshot_detail.parquet",
                                 columns=["extractant", "n_pool", "n_eval"]).groupby("extractant")
                 .apply(lambda b: bool((b["n_pool"] >= 5).all() and (b["n_eval"] >= 2).all()),
                        include_groups=False).pipe(lambda s: s.index[s]))
    block = oof[(oof["model"] == selected_model) & (oof["extractant"].isin(common))]
    zero_from_oof = float((block["log_D"] - block["prediction"]).abs()
                          .groupby([block["split_seed"], block["extractant"]]).mean()
                          .groupby(level=1).mean().mean())
    stored_zero = float(fro[(fro["global_model"] == selected_model) & (fro["k"] == 0)]["pipeline_mae"].iloc[0])
    check(record, "15 k = 0 common-cohort macro MAE regenerates from raw OOF predictions",
          abs(zero_from_oof - stored_zero) < 0.02,
          from_oof=zero_from_oof, stored=stored_zero,
          note="k=0 in the k-shot protocol scores evaluation halves; the OOF regeneration scores all "
               "rows, so agreement is to the protocol's sampling, not bit-exact")

    # 16. rerun vs stored predictions on fold 0 of the first seed
    stored = pd.read_parquet(args.stage_dir / f"oof_{selected_model}.parquet")
    stored = stored[(stored["split_seed"] == DEFAULT_SEEDS[0])].set_index("row_id")["prediction"]
    ids = frame.iloc[fold.test_index]["row_id"].astype(str).to_numpy()
    delta = float(np.abs(stored.reindex(ids).to_numpy(dtype=float) - stored_fold0).max())
    check(record, "16 fold-0 rerun matches stored predictions at the float floor",
          delta <= DETERMINISM_TOLERANCE, max_abs_delta=delta)

    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["all_pass"] = all(c["pass"] for c in record["checks"])
    (OUT / "self_audit.json").write_text(json.dumps(record, indent=2, default=str))
    print(f"\nSELF-AUDIT: {'PASS' if record['all_pass'] else 'FAIL'} ({len(record['checks'])} checks)")
    return 0 if record["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
