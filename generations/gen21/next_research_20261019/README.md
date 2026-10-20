# gen21 round 3 — next_research_20261019

Task: `../INSTRUCTIONS FOR NEW IDEAS ROUND 3.md` (Russian). Goal: R² ≥ 0.50 for strict-adjacent
log SF with **four** measured metals (V1, C3, k = 3, DOPT_Q), with a real advantage over interpolating
the same measurements (PWL). Read `PROTOCOL.md`, `EXPERIMENT_REGISTRY.csv`, `ATTEMPTS.md`, then
`FINAL_REPORT_RU.md`. Resume: `RESUME.md`.

**Status (2026-10-19): complete; the single final check has been run** (user's permission; results in
`results/final/` and `FINAL_REPORT_RU.md`): R02 0.4224 / 0.4227, target not reached, gain over PWL not
confirmed; the Gd parameter (R14 / R23, secondary) keeps +0.03 over PWL on the new partition. Dev summary: 0.50 is not reached with four
metals. Under PROTOCOL §7 as written the candidate is R02 (geometry corrector, 0.4189 / 0.4199, gain over
PWL indistinguishable from 0). The one-parameter Gd correction (R14 0.4518 on DOPT_Q; R23 0.4748 on POL_ERR
reads, a separate regime) gains +0.034 / +0.038 over PWL on the same reads but fails the registered §8.6
control (it learns a pair-type-level mean). R02 was frozen with R14 and R23 as secondary configurations.
Read `FINAL_REPORT_RU.md`, then `AMENDMENTS.md` (deviations found by the review). Nothing committed or
pushed (brief §3).

## Layout

```
r3/         paths, interp (PWL by Z / by radius, PCHIP), harness (rows with read sets, policy, budget, data
            version; intervention), corrector (direction A), inner_preds (B / stacking), policy (D, POL_ERR),
            chem (C), configs + registry (one ID = one configuration), metrics
scripts/    r3_manifest.py, r3_run.py (one config x one design, checkpoint/resume), r3_queue.py, leakage checks
            (r3_cache_poison, r3_poison_chain, r3_intervention, r3_repro), analyses (r3_policy_cross, r3_cross2x2,
            r3_ladders, r3_diagnostics, r3_selection_optimism, r3_calibration, r3_table), r3_freeze.py, the
            guarded r3_final_runner.py
tests/      contracts: masks, poisoning, intervention, legality, label-free policy
manifests/  environment, inputs (cohort, log D, folds, cache content), code digests, freeze manifest
predictions/<ID>_<design>.parquet  row-level OOF predictions;  _parts/ per-fold checkpoints
results/<ID>/                       summaries, per-pair, contrasts, levels, fitted choices, intervention
```

## Commands (repository root, Windows venv)

```bash
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_manifest.py
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_run.py --config B0 --design BP
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_run.py --config B0 --design BPG
```

Each further configuration works the same way with its ID from `EXPERIMENT_REGISTRY.csv`. The
gen21 fold cache (`../cache`, rebuilt by `../scripts/g21_build_cache.py`) is reused read-only.
