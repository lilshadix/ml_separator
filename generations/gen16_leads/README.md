# gen16_leads: six pre-registered leads on the frozen gen13 cohort

Gen16 asked whether a disciplined multi-agent search of the frozen gen13 cohort (fingerprint `4c3c6628ea0be949`: 521 cells, 90 extractants, 45 chemotypes, Kish n_eff 11.7) can find a positive, valid, significant result. Six leads were pre-registered: L1, a cycle-corrected GFN2-xTB slope; L2, a leave-pair-out curvature gate; L3, the decision value of the gen14 direction call (L3a measurements saved, L3b within-chemotype ranking, L3c ranking at k = 1); L4, chemotype acquisition order; L5, covariance shrinkage and calibration of the measured mode; L6, a cohort audit. Every lead ran under designs B, BR, BQ, A and BP; BP selects and FLAT is the floor. Each surviving claim went to two blind refuters, then to one confirmation run on five withheld seeds. Dated 2026-09-09 (brief, commit df63929) to 2026-09-10 (commit 9ff67e7); moved from the repository root into `generations/` in commit 155dc6c.

## Status

**Mixed, closed.** One claim confirmed, every other lead closed or failed. `PRE_REGISTRATION.md` is **sealed** (body SHA-256 `d004c303...c48f50e`, no addenda): never edit it. Two in-unit Phase 6 adversarial audits checked the report, but L1 Stage 2 and the 1030/128 multiplicity recount (Phase 7, commit 9ff67e7) came after them and were not audited.

- **L3a, confirmed.** Dropping candidates whose gen14 direction call disagrees with the requested direction cuts the expected measurements to a first useful candidate (observed log SF of the requested sign, |log SF| >= 0.3) from 6.91 to 4.92: +1.988, 95 % CI [+0.443, +3.781], design BP, 5 discovery seeds, 455 tasks. On the withheld seeds: +1.909, CI [+0.360, +3.760], p 0.014, 5/5 seeds.
- **L1, powered null.** Exact xTB cycle: extractant-level Spearman rho −0.084 (set S8, n = 62), CI [−0.297, +0.240], from a slope with jackknife reliability 0.80; family-wise permutation p 0.94.
- **L2, gate fails.** Headroom H = +0.0287, percentile CI [−0.0013, +0.0532] (extractant-macro MAE of log SF, BP, discovery seeds), minimum detectable 0.039 against a 0.02 margin.
- **L4, killed.** A featureless biggest-chemotype-first order reproduces the A-optimal gain (AOPT − SIZE +0.005, p 0.36, BP).
- **L5, no claim.** No contrast reaches the 0.02 margin; the deployed covariance is indefinite in 100 % of 430 estimates and nominal 90 % intervals cover 98.8 % at k = 3 (BP, discovery seeds).
- **L6.** Counterfactual Kish 11.67 → 27.38 is a design-capacity statistic, not a measured gain. Zero-shot skill is still gen15's +0.088 over predicting no separation.

## Read first

1. [DECISION_REPORT.md](DECISION_REPORT.md): headline, §2 the confirmed claim and its scope, §9 supported/not supported, §10 defects, §11 guardrails.
2. [CONFIRMATION.md](CONFIRMATION.md): seed commitment, the one run, what it does not establish.
3. [REFUTATION_LOG.md](REFUTATION_LOG.md): six blind refuters, temptations, process failures.
4. [PRE_REGISTRATION.md](PRE_REGISTRATION.md) (sealed), then [START_HERE.md](START_HERE.md) (the brief that commissioned the run).
5. [STATUS.md](STATUS.md): phase checkpoints, newest first; earlier entries carry superseded verdicts.
6. Lead detail: [results/L1/L1_REPORT.md](results/L1/L1_REPORT.md), [results/L1/L1_STAGE2_HANDOVER.md](results/L1/L1_STAGE2_HANDOVER.md), [results/L2/L2_GATE.md](results/L2/L2_GATE.md), [results/L3/L3_REPORT.md](results/L3/L3_REPORT.md), [results/L4/L4_REPORT.md](results/L4/L4_REPORT.md), [results/L5/L5_REPORT.md](results/L5/L5_REPORT.md), [results/L6_cohort_audit/L6_COHORT_AUDIT.md](results/L6_cohort_audit/L6_COHORT_AUDIT.md), [results/anchors/ANCHORS.md](results/anchors/ANCHORS.md), [results/env/ENV.md](results/env/ENV.md).

## Key files (relative to this directory)

- `gen16/bootstrap.py`: import before any frozen module; puts `generations/gen15_curve`, `gen14_direction` and `gen13_separation` on `sys.path` and caps `LOKY_MAX_CPU_COUNT` at 2.
- `gen16/claims.py` (frozen confirmation claim registry), `gen16/l3_decision.py`, `gen16/l1_cycle.py`, `gen16/l2_lpo.py`, `gen16/l4_acq.py`, `gen16/l5_cov.py`, `gen16/cohort_audit.py`, `gen16/anchors.py`.
- `scripts/`: per-lead runners, `refute_*.py` (refuter checks), `audit_*.py` (Phase 6), protocol guards `g16_prereg_hash.py`, `g16_audit_seeds.py`, `g16_manifest.py`, and `g16_confirm.py` (already executed once; refuses a second run without `--force`).
- `results/confirmation/verdicts.csv` and `discovery_vs_confirmation.csv`; `results/refutation/` (`L3A`, `L4BP`, `L1NULL`); `results/audit/`; per-lead tables under `results/L1` to `results/L6_cohort_audit`; `results/MANIFEST.sha256`.

## How to run (from the repository root)

Do not run these casually: the runners overwrite committed result files. Script docstrings still show the Windows pre-move form (`.venv/Scripts/python.exe gen16_leads/scripts/...`); use the paths below.

```bash
.venv/bin/python generations/gen16_leads/scripts/g16_prereg_hash.py          # verify the seal; never pass --seal/--reseal
.venv/bin/python generations/gen16_leads/scripts/g16_audit_seeds.py          # no discovery file passes seeds= or a confirmation seed
.venv/bin/python generations/gen16_leads/scripts/g16_manifest.py --check     # without --check it rewrites results/MANIFEST.sha256
OMP_NUM_THREADS=2 .venv/bin/python generations/gen16_leads/scripts/g16_anchors.py   # Phase 0 anchors; builds generations/gen14_direction/cache/bench.pkl
.venv/bin/python generations/gen16_leads/scripts/g16_cohort_audit.py        # L6
OMP_NUM_THREADS=2 .venv/bin/python generations/gen16_leads/scripts/l1_stage1.py
.venv/bin/python generations/gen16_leads/scripts/l1_run_references_local.py --workers 6 --xtb /path/to/xtb   # needs xtb 6.7.1pre
OMP_NUM_THREADS=2 .venv/bin/python generations/gen16_leads/scripts/l1_stage2_cycle.py --refs generations/gen16_leads/results/L1/reference_species/reference_energies.csv
.venv/bin/python generations/gen16_leads/scripts/l1_stage2_reliability.py
OMP_NUM_THREADS=2 .venv/bin/python generations/gen16_leads/scripts/l2_gate.py
OMP_NUM_THREADS=2 .venv/bin/python generations/gen16_leads/scripts/l3_decision.py && .venv/bin/python generations/gen16_leads/scripts/l3_report.py
OMP_NUM_THREADS=2 .venv/bin/python generations/gen16_leads/scripts/l4_retro.py && OMP_NUM_THREADS=2 .venv/bin/python generations/gen16_leads/scripts/l4_diag.py && OMP_NUM_THREADS=2 .venv/bin/python generations/gen16_leads/scripts/l4_rank.py
.venv/bin/python generations/gen16_leads/scripts/l5_verify.py && .venv/bin/python generations/gen16_leads/scripts/l5_covrun.py && .venv/bin/python generations/gen16_leads/scripts/l5_direction.py && .venv/bin/python generations/gen16_leads/scripts/l5_summary.py
```

## Tests

```bash
PYTHONPATH=src .venv/bin/python -m pytest generations/gen16_leads/tests -q -m "not slow"
```

42 tests collected after the move (`test_anchors.py` 4, `test_gen16_results.py` 18, `test_l6_cohort_audit.py` 15, `test_protocol.py` 5); two are marked slow, drop `-m "not slow"` to run them. On macOS, 40 of the 42 pass and the two `test_reproduces_frozen_key_mode` cases in `test_l6_cohort_audit.py` fail on the cohort fingerprint; STATUS.md reports passes on the Windows clone. The anchor tests build the gitignored `generations/gen14_direction/cache/bench.pkl` if it is absent.

## Dependencies

- [`generations/gen13_separation`](../gen13_separation/README.md): `gen13sep` splits, `inference.paired_contrasts`, metrics, amplitude bench, cohort, paths.
- [`generations/gen14_direction`](../gen14_direction/README.md): `gen14.dirbench`, `gen14.models`.
- [`generations/gen15_curve`](../gen15_curve/README.md): `gen15.valuebench`, `arms`, `fewshot`, `mixture`; `exp/phys3d`, `exp/decision/decmetrics.py`, `exp/labelerr/s4_floor.py`, `exp/external/data/side_K_logk.parquet`. The deployed `scripts/g15_predict.py` is left unchanged.
- `generations/gen12_2_eu_pred/features` (read transitively through `gen13sep.paths` when the bench is built); `src/`.
- `dataset with 3D structures/` (`dataset.parquet`, `accepted_geometries.csv`, `features/complex_physical_scalars.parquet`, `ligand_2d_descriptors.parquet`); `runs/gen7_architecture/cache/chemistry_map.parquet`; `runs/gen6_provenance/provenance_table.parquet`.
- External: the xtb 6.7.1pre binary, outside the repository, for the L1 reference species only (`reference_energies.csv` is committed).

## Caveats and corrections

- **The confirmed claim is between-laboratory only.** Within one publication it saves +0.0046, CI [−0.016, +0.022] (BP, discovery seeds, 1 230 tasks; interval contains zero in all five designs); within one chemotype +0.334, failing the rule 5/5; rebuilt one fold at a time +0.906 of an E_random of 5.52 (16.4 %) rather than 28.8 %. Guardrail 1 (DECISION_REPORT §11) requires both dents to travel with the number.
- "Confirmed" means L3a's registered rule plus the percentile half of P1: L3 used its own 2 000-replicate percentile bootstrap, so the BCa half was never computed (DECISION_REPORT §10 item 10).
- DECISION_REPORT §1 mixes two multiplicity accountings: 1030 rows (128 registered) in the headline, 940/126 and 860/122 in the paragraph and BH table (`results/audit/verify/bh_all_940.csv` is pre-recount). The confirmed claim's BH q is 0.040 (128) and 0.027 (1030).
- The headline "+0.644 fell to −0.09" pairs S14 raw energies (n = 39) with S8 dE (n = 62); like for like S14 +0.644 → −0.042, S8 +0.392 → −0.093. The L1 null covers only a gas-phase GFN2 total-energy slope over 62 extractants. The L1 verdict changed three times: `results/L1/L1_REPORT.md` §0 "CLOSED" and STATUS Phase 2/4 verdicts are superseded by Stage 2.
- L2 is closed by rule and underpowered by design; the r0 study never ran. The L4 rankings (`results/L4/ranking_pool_bundle.csv`, `ranking_pool_logk.csv`) follow a family-size proxy and are not a purchase list. L5's common scoring set differs from gen15's (POOLED@k3 0.1674 vs 0.1700); its PSD and NOISE_VAR fixes are recommendations, not applied to gen15.
- BP selects: do not quote B or A numbers as selection evidence. MEAN_CURVE (0.622) is not a baseline. The composition-position correlations (steppos_n_H2O up to +0.83) are post hoc refuter findings, not results.
- Stale text: L6_COHORT_AUDIT §2 "91 of the 100 from a single publication" (they span 23 publications); REFUTATION_LOG §4 "larger than every xTB estimator" (false against NAIVE +0.644); `results/L1/stage2_decision.json` records pre-move `gen16_leads/results/...` paths.
- All computation ran on the Windows clone (Python 3.14.5, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.0); bit-identity on macOS is unchecked. Do not upgrade scikit-learn to 1.10 (the frozen G14 arm uses a deprecated `penalty=`).
- Large artefacts are excluded by `.gitignore` here and digested in `results/MANIFEST.sha256`; `results/L4/_dry` is a discarded dry run whose 80 rows are counted in the 1030.
- Unreviewed context: sibling work committed without review (`generations/gen16_protocol/results/g16_size.log`) finds the percentile chemotype-blocked bootstrap mildly anti-conservative (rejection 0.0705-0.0815 at nominal 0.05). `generations/gen16_anchor`, `gen16_protocol` and `gen17_pairdiff` are not part of this unit.
- This unit corrects earlier ones (DECISION_REPORT §10): gen15 §3's +0.644 xTB correlation is a composition artefact; about 0.044 of gen15 §2's in-sample +0.073 curvature gap was self-fitting noise; gen15's deployed covariance is indefinite and over-covers; `gen13sep/arms_stage2.py:330` hashes bytes with salted `hash()`. No later generation report corrects gen16_leads.
