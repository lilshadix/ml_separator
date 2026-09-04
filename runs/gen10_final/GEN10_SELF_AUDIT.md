# GEN10 self-audit

**STATUS: PASS — Phase 0 12/12, final self-audit 17/17, 214 tests, gen9 manifest
154/154 zero drift, gen10 manifest 269 artefacts zero drift.**

*Records: `reproduction/phase0.json`, `self_audit/self_audit.json`,
`reproduction/pytest_phase0.log`, `manifest.json`. Every check below states what
it compared; "the floor" means the ExtraTrees thread-order floor of ~1e-15 log
units, against which every determinism tolerance (1e-12) was set three orders of
magnitude above and nine below any effect reported.*

## 1. Phase 0 — freeze and reproduce (before any new arm trained)

| # | check | result |
|---|---|---|
| 1 | gen9 manifest verifies | 154 recorded / 154 on disk, 0 changed, 0 missing |
| 2 | gen9's Phase-0 script reproduces gen8 in this environment | 15 / 15 references (zero-shot 1.0605, MEDOID 1-shot 0.6916, oracle 0.5326, 2/3/5-shot 0.5879 / 0.5230 / 0.4743, extractant slope 2.574 / 0.116, acid 1.656 / 0.355, span ratio 0.051); zero library drift (Python 3.13.0, numpy 2.5.1, pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.0) |
| 3 | cohort fingerprint | `bed178ec1a7a82b0`, 5,248 rows |
| 4 | curve geometry | 1,176 curves / 7,207 memberships, gen9's table |
| 5 | fold / curve boundaries | 25 folds, 0 curves straddling, every held-out partition closed |
| 6–8 | gen10 nests the gen9 arms | max \|Δ\| 1.3e-15 (`REC_ecfp_plus_recovered`), 1.3e-15 (`GEN9_REL_MONOLITH`), 8.9e-16 (`GEN9_SHAPE_RECOMPOSED`) on fold 0 |
| 9–10 | determinism, same process twice | `REL_MONOLITH` and `SHAPE_RECOMPOSED` at the floor |
| 11 | determinism, fresh subprocesses | `PYTHONHASHSEED` 0 vs 4242: at the floor |
| 12 | test suites | 214 passed (gen7/8/9 harness, reproduction, acquisition, curve objective, series adapter; gen10 Phase-0 metamorphic and regression suites, slow tests included) |

The metamorphic tests the brief required, all present and passing
(`tests/test_gen10_phase0.py`): MEDOID and FARTHEST disagree on an asymmetric
synthetic pool and FARTHEST jumps to the outlier once the medoid is taken;
non-constant condition coordinates stay non-constant through the feature builder
and `_standardised_axes`; removing or corrupting held-out targets (N(1e4, 1e3))
leaves predictions of every gen10 class and every selection unchanged; permuting
query rows permutes predictions at the floor; the learned set encoder is
deterministic, equivariant and zero-mean per curve. One regression test per gen9
issue (`tests/test_gen10_regressions.py`): the condition join (the all-zero-axes
tie reproduced, the assertion that refuses a constant axis), the core-cell
duplicate key (a decade pair hidden by an unreported temperature is found by the
core key and missed by the full key), the leave-one-out adjusted level (NaN, not
zero, when unidentifiable), the float group key, CurveBoost's rounding and its
unrounded base constant, LEARNED_BLEND's inner ligand split; plus gen10's own two
(`stabilise`, cross-fitting) and the builtin-`hash` AST scan extended to gen10 and
the shared gen6/gen8 modules.

## 2. Final self-audit — the sixteen checks on the frozen pipeline

Selected model `SHAPE_RECOMPOSED` (gen9's arm through gen10's class).

| # | check | what was compared | result |
|---|---|---|---|
| 1 | identical configuration twice in one process | fold 0 of seed 104729, 382 rows | max \|Δ\| at the floor; pooled MAE 1.097628 both times |
| 2 | fresh subprocesses, `PYTHONHASHSEED` 0 / 1 / 4242 | the same fold | at the floor |
| 3 | manifests | gen9 `--verify`; gen10 stamped | 154/154 zero drift; 269 artefacts, verify 0 changed |
| 4 | row accounting | every finalist × every seed | 5,248 rows each, 45 (model, seed) blocks |
| 5 | fold boundaries | 25 folds | 0 row / ligand / chemotype / cluster overlap |
| 6 | curve boundaries | 25 folds × 1,176 curves | 0 straddling |
| 7 | no target contamination | the query frame handed to `predict` | `log_D` absent, asserted |
| 8 | corrupted held-out targets | N(1e4, 1e3) written into the query frame | predictions unchanged at the floor |
| 9 | permuted candidate rows | a random permutation of the largest held-out ligand | predictions permuted, nothing else |
| 10 | query-set composition | `query_consistency/summary_by_arm.csv` (5 seeds) and `query_consistency_finalists/` (2 seeds) | recorded; frozen control exactly zero; see decision report §3.1 |
| 11 | acquisition policies on synthetic positive controls | five bunched points and one outlier | MEDOID in the bunch, FARTHEST ≠ MEDOID, FARTHEST → outlier after the medoid |
| 12 | paired comparisons | every global model's (seed, fold, ligand, repeat, pool, eval) units | identical sets across all nine models, 1,426,080 records each |
| 13 | NaN / inf | every CSV in `final_locked/` | none, except BCa bounds where the jackknife is degenerate and the guarded span columns on the contact-time axis (zero curves above the span guard, as in gen9's own table) |
| 14 | no difficult rows disappear | every cohort row in every finalist OOF | 0 missing |
| 15 | tables regenerate from raw predictions | k = 0 common-cohort macro MAE recomputed from `oof_finalists.parquet` | 1.0324 from all rows vs 1.0358 from the protocol's evaluation halves — agreement to the protocol's sampling |
| 16 | rerun vs stored predictions | fold 0 refit vs gen9's stored `oof_GEN9_SHAPE_RECOMPOSED.parquet` | byte-level at the floor |

## 3. What else was verified along the way

* **Determinism of every trained gen10 arm**: `<stage>/determinism.csv`, 26
  arms, all at the floor — including the two arms whose first versions were not
  (decision report §0.1).
* **Exact invariances inside the consistency benchmark**: the `CONTEXT` variant
  (whole ligand vs same-series frame) and the `PERMUTE` variant agree at the floor
  for every arm and every curve, and the frozen model's shift is zero under every
  design change — the benchmark measures the model, not its plumbing.
* **The k = 1 identity**: every ridge adapter (`OFFSET_K1/K2/K3`, `SERIES_*`)
  identical at k = 1 to 3e-14 over 12 repeats × 5 seeds.
* **gen9 numbers inside gen10's runs**: `SERIES_MAP` 0.5959 / 0.5729 / 0.5735 and
  `OFFSET_K3` 0.5409 / 0.4887 / 0.4779 on `REL_MONOLITH` (gen9: 0.596 / 0.573 /
  0.574 and 0.541 / 0.489 / 0.478); every feature-access anchor's per-seed pooled
  MAE equals gen9's log to four decimals.
* **Provenance of every headline number**: `headline_tables/headline.json` lists
  the phase each table came from and the phases that were missing (none).

## 4. What was not done, stated so it is not inferred

* Phase 1 on the finalists ran at **two** seeds (the gen9 arms at five); the
  finalists' consistency numbers are labelled as such.
* Phase 9 simulates allocation over held-out *adaptation curves*; it does not
  retrain with subsets of ligands (gen6 did that side).
* `SERIES_INNER` was not repaired to replay the deployed policy; the brief
  forbids continuing to tune that branch once `ML` cleared the bar.
* No arm combining the slope repair with `SERIES_ML` at k ≥ 2 was run; the
  pipeline uses the repair at k = 1 only, where it is the best deployable arm.
* Nothing was committed to git; the worktree holds gen10 alongside gen7–gen9,
  which were also uncommitted when the sprint began.
