# gen19_chem_transfer — Chemistry Transfer Engine

Generation 19 asks whether unmeasured metal × extractant combinations can be reconstructed from
structure shared across metals, extractants, mechanisms, conditions and publications, with an
uncertainty that is carried into the Gen18 process layer. It learns from the whole SAFE archive
(`dataset_all_metals/`: lanthanides, actinides and other metals, all extractant families) but the
applied target stays Pr/Nd separation. Success is judged only under deliberately hidden chemistry
(leave-publication, leave-metal and missing-cell hold-outs), never on a random split. The brief is
`GEN19_CLAUDE_CODE_INSTRUCTIONS.md`; where this directory departs from it, the departure is listed
under [Conventions](#conventions).

## Status

| Phase (brief §28) | State |
|---|---|
| A — corpus audit | complete: `DATA_AUDIT.md`, tables in `data_audit/`, figures F01, F02, F04–F06 |
| B — feasibility | complete: `FEASIBILITY.md`, `data_audit/feasibility*.{json,csv}`, figure F03 |
| decision | `decisions/D00_corpus_feasibility.md`, `decisions/D01_transfer_signal.md`, `decisions/D02_factorization.md` (brief §29 format; D03–D06 wait on the runners that answer them) |
| verification corrections | applied 2026-09-15 (27 findings of three adversarial verifiers, then 7 residual findings of a second pass): component-aware V5 hiding (made symmetric at the metal-state level by a pre-seal design decision), V1 groups merged on archive duplicate groups, same-charge-first B3x, chemistry label fixes (aqueous agents recorded as extractants, EsPyTri, Br-Cosan, quercetin, name–structure conflicts, HDEHP as co-extractant), acid-medium and censoring counts, `V6_TARGET_ROWS`, selection / confirmation halves, fixed success margins |
| pre-registration | **SEALED 2026-09-15** (`preregistration.md`, footer digest `135842499a86…5641`, `manifests/prereg_sha256.txt`; `preregistration_draft.md` is the frozen draft it was built from). It carries **one POST-HOC addendum below the footer** — addendum 1 (2026-09-15, no learned-model outcome seen), the compute-driven reduction of the discovery plan: the sealed §7 plan was priced at 1,535.7 CPU-h against a 60 h budget, so V5 inner tuning became three *simultaneous* inner folds (one fit per configuration per fold, up to 30 cells hidden at once), discovery runs **seed 104729 only** (R19 item 4 NOT_EVALUATED in discovery, unchanged at confirmation), a learned arm runs only the V5 strict and HNO3-only refits (R19 item 6 = "reduced sensitivity set (addendum 1)", the rest named as not run), and V5-PAIR carries M2 with its M1 prerequisite. Designs, folds, halves, hiding, guards, metrics, comparators, margins, the stop rule and confirmation are unchanged. The text above the footer is never edited; `g19_seal_prereg.py --check` exits 0 and the runner refuses any addendum count but the one it implements. History: the orchestrator resolved every open marker in `preregistration_draft.md` on 2026-09-15; each resolution says whether outcomes had been seen, and only baseline outcomes existed (§20; list in `decisions/D01_transfer_signal.md`). Task X re-verified the code consequences and put two readings back to the orchestrator; both were resolved the same day and are implemented: S1(c) "same fitted folds" — B3x and B3i re-fitted on exactly the batched V5-PAIR folds the candidate is fitted on, HEAVIER needs no fit (`gen19ct.evaluation.transfer.S1C_REGISTERED_FOLD_READING`, `check_s1c_fold_designs`; re-fit helper `gen19ct/models/s1c_yardsticks.py`, unit-tested on a synthetic mini-corpus only) with the seed combination at confirmation (`transfer.s1c_seed_combination`: seed-mean Δ_Y, system-cluster bootstrap of the seed mean with the same resampled systems in every seed, 5 of 5 seeds) — and the wildcard-copy sensitivity registered for V5-P and V5-PAIR as well as V1 and V5 (`transfer.REGISTERED_SENSITIVITIES`; the strict filter stays exploratory). Under the V1 outer-fold unit the pooled remainder fold is one publication-group cluster (`metrics.design_unit_clusters`, `tests/test_resolutions.py`). The verification of that implementation (same day) tightened the S1(c) guards — every fitted arm and the pair set must name `<stem>@<design_hash>` (a bare stem is the same for every seed), `s1c_half` refuses pairs of another half, `s1c_paired_verdict` refuses rather than FAILs a selection result not scored on the selection half on seed 104729 — made marked consistency corrections in `preregistration_draft.md` (§3.1 and §7 item 7: the S1(c) fold exceptions; §3.2 and §8: the V1 unit wording), and put one reading back to the orchestrator, resolved the same day: §9 S2(a) now uses the paired rule of S1(c) (`decisions/D01_transfer_signal.md`). `scripts/g19_seal_prereg.py --check` exits 0: sealed 2026-09-15 (footer 135842499a86…5641) with POST-HOC addendum 1 below the footer. |
| pre-seal Phase C | closed-form baselines only (`scripts/g19_run_preseal.py`, `decisions/D01_transfer_signal.md`); second verification pass applied 2026-09-15 (task X: one inner-design implementation shared by the fold builder and the conformal wrapper, known-state calibration rows, section 13 thresholds written by the fold builder, `gen19ct/evaluation/support.py`, the wildcard-copy leakage audit, exploratory V1-unit and copy sensitivities, a seal gate on unticked boxes); `tables/preseal_pair_summary.csv` rebuilt the same day with `--tables-only` so that its DIR5-alternative rows are exploratory, role side, arm `HEAVIER_alt_half` / `HEAVIER_alt_lnln` (no refit; every other output re-rendered byte-identical, `manifests/g19_run_preseal.json` → `tables_only_rebuild`); after task X the same day every output was re-aggregated from the stored predictions under the resolutions (`--skip-compute`, no refit; `manifests/g19_run_preseal.json` → `aggregation_rebuild`): registered V1 outer-fold unit and wildcard-copy status in the tables, resolved `support_score` for every support job (`scripts/g19_update_support_preseal.py`, now also called by the pre-seal script), `evaluation/preseal/difficulty_resolved.json`, and `difficulty.json` byte-identical to the digest §9 quotes; after the last two resolutions (S1(c) "same fitted folds", wildcard copies on V5-P / V5-PAIR) were implemented and verified, `--skip-compute` ran again (no refit; of the 58 recorded outputs only `difficulty_resolved.json` changed: `resolved_orchestrator_decisions`, the V5-P / V5-PAIR wildcard-copy status and the two reading strings; the manifest's `code_sha256` is current) and a `--tables-only` pass then re-rendered every output byte-identical (`tables_only_rebuild.tables_rewritten` = `[]`) |
| C — baselines B0–B8 on V1/V2/V5 | **complete for every arm addendum 1 schedules**: the closed-form baselines are the pre-seal run; the learned arms B5 (= M0), B6, B6r0, B8, FLAT_CAT, M1 and M2 are fitted and scored on the **selection half, seed 104729 only** (`evaluation/discovery/`). M3–M7 were never fitted: the 60 h budget was exhausted, so the discovery ledger demoted them by rule (`decisions/decisions.json` → `budget.discovery.demoted_now` = M7, M6, M5, M4, M3; `never_demoted` = H1, H1b, H4) and POST-HOC addendum 2 gives the ladder its own 40 h |
| discovery run | **COMPLETE 2026-09-22 04:36, exit 0**, every job 0 errors: 76.6 h wall clock on 2 workers (`decisions/decisions.json` → `budget.discovery.used_hours` 76.5955 against `budget_hours` 60.0, `exhausted` true), 3 invocations, **1,461 registered fold records** over the 7 fitted arms (`manifests/digest_registry.json` → `stages.discovery.n_records`), beside the 488 shared `_support` frames and the S1(c) yardstick refit — 240.3 MB with the run log, excluded from git (see the last row). Plan state as run: `heavy_v5_scheme` `batched_max4`, `heavy_v1_scheme` `exact`, `v5_batched_check` `passed_after_recolour`, `v1_tenfold_check` `failed`, every design's guard mode `nested_certificate` (`decisions/plan_state.json`). The freezing-candidate pass then ran (registry stage `discovery_candidates`): 3 jobs, all markers, 0 folds fitted — the M1 / M2 refits deduplicate into the main plan — and 33 pre-existing fit jobs `skipped_done` |
| discovery runner | built: `scripts/g19_run_discovery.py` (plan, seal gate, resumable per-fold records, cost model) and `scripts/g19_score_discovery.py` (R19, stop rule, ladder, S1 components) implement the sealed §7 plan **as amended by POST-HOC addendum 1**. The seal gate pins the footer digest, the addendum count AND the addendum text (`discovery.REGISTERED_ADDENDA_SHA256`), and every fold record's resume digest carries the addendum digest and the resolved inner design. `--dry-run` writes the job plan to `evaluation/discovery/benchmark/plan_addendum1.txt`. The addendum-1 cost estimate (`--benchmark-addendum1`, fit-only timings on the TRAINING rows of one V5-primary batched fold, no test row predicted) is **measured**: `cost_estimate_addendum1.{json,md}` price the plan at **90.2 CPU-h, 45.1 h wall clock on 2 workers against the 60 h budget -> FITS** (sealed plan: 1535.7 CPU-h / 767.9 h wall); cumulative wall-clock checkpoints by stage: 00_safeguard 0.2 h; 01_B6 0.5 h; 03_B5_FLAT_CAT_B8 15.4 h; 04_M1 17.9 h; 05_M2 19.0 h; 07_s1c_v5pair 27.6 h; 08_refit_sensitivities 45.1 h. Conditional runs (V5-P per heavy arm, further V5-PAIR candidates, the B6 re-colouring) are priced separately and are not in the total. The estimate is ideal wall clock (compute / workers) from single-fold timings |
| scorer | **run, exit 0** (`scripts/g19_score_discovery.py`; the last of 5 recorded passes took 127.4 s, `manifests/run_info/g19_score_discovery.json` → `passes`, and re-scoring never moved a verdict). Records are verified against the **registered** `discovery` code digest `895a6ab9…` of `manifests/digest_registry.json`, never against live code (POST-HOC addendum 2 item 5; the live digest is `d4ddf83c…`): 34 of 34 planned record sets complete = **1,254 folds** (`decisions/decisions.json` → `record_sets_summary`; the 2 remaining requests — B8 V5-strict and B8 V5-HNO3-only — are `not_planned`, addendum 1 item 4 registers those refits for H1 / H1b / H4 and freezing candidates only). Guards: `confirmation_half_read` false, `v6_target_rows_scored` 0, 186,446 row halves re-derived over 45 scoring frames. Outputs: `evaluation/discovery/{contrasts_registered,contrasts_exploratory,r19_items,records_index}.csv` (29 / 33 rows of 38 available contrasts), `decisions/{decisions,stop_rule,plan_state,record_verification}.json`, `tables/discovery_*` |
| stop rule and headline verdicts | **stop = false** (`decisions/stop_rule.json`): M2 vs B3i@V5 **PASS** on R19 items 1, 2, 3, 5 (MAE 0.504 vs 0.767, Δ **+0.263** ≥ δ5 = 0.10569, system-cluster p = 0.0040), B6 vs B3i@V5 **FAIL** (Δ −0.104, p = 0.0550); `pending` and `consequences` empty. Because R19 item 4 is NOT_EVALUATED in discovery (addendum 1 item 3, one seed), the full R19 verdict of the passing contrast is **UNDECIDED**, so nothing is confirmed in discovery. The three freezing candidates (M2 vs B0 / B3i / B6r0@V5) are **`eligible_for_freezing` true** under POST-HOC addendum 3 item 1 — items 1, 2, 3, 5 and 6 PASS, item 4 not evaluated in discovery, no item FAIL (`decisions/plan_state.json` → `freezing_candidates[].eligibility.reason`); the *literal* §3.1 V5-P trigger (`section_3_1_v5p_trigger`) is still false, and addendum 1 item 4 runs no V5-P job either way. Item 4 is evaluated at confirmation exactly as registered, on 5 of 5 withheld seeds. H4 ×3 FAIL, ladder M1 and M2 **not kept** against M0 (= B5), S1(a)/(b)/(c) reported UNDECIDED, S1(d) PASS, S1(e) NOT_EVALUATED. `decisions.json` → `power_check.status` is **not implemented**: the §8 signal-injection check is *required before any of these nulls is reported as a null*. Every number: `decisions/D02_factorization.md`, `tables/discovery_contrasts.md` |
| D–H — models, uncertainty, process | **ladder RUN, exit 0** (`scripts/g19_run_ladder.py`): every step M3–M7 is `not_run` because neither M1 nor M2 is retained (POST-HOC addendum 3 item 3), `retained_final` **M0 (= B5)**, `stop_rule` false, and the ladder's own 40 h budget is **unconsumed** — `budget.used_hours` 0.0 of 40.0, `exhausted` false (`evaluation/ladder/decisions/{ladder,wall_clock}.json`, `tables/ladder_decisions.csv`, D04, D05). The ledger also records each invocation's `process_seconds` beside the budget's `seconds`, so an invocation that ran and skipped every step is distinguishable from one that never ran, and states the discovery run's 76.5955 h against its registered 60 h (task X finding V-L4). **The section 8 power check has run** (`scripts/g19_run_power.py`; `evaluation/power/`, `tables/power_kappa.csv`, `tables/reliability_before_correlation.csv`, `tables/power_per_unit_mae.csv`). **H3 (section 11) has RUN and is SCORED** (`scripts/g19_run_h3.py`; `evaluation/h3/`, D03): **10 of the 12** planned contrast record sets are complete and scored — every WITHOUT and ACT_PERMUTED leg of B5 and B6 on V5 and V2, and the WITHOUT legs on V1 — and the two `ACT_PERMUTED@V1` legs are `INCOMPLETE_GUARD_FAILURE` (25 of 39 folds each), not `NOT_RUN`: the 20 h cap was not reached (5.78 h used) and the legs stopped on a deterministic section 2 isolation-check failure at fold `pub_97510df3a0`, which `scripts/g19_h3_guard_diagnosis.py` traces to the near-duplicate **value** comparison reading the ACT_PERMUTED arm's permuted `log_D`; the fold is not forced. Completeness is a property of the contrast record set, not of the design (`h3.CONTRAST_UNIT_RULE`, a reading of addendum 4 item 2 for which an addendum is REQUESTED). Verdicts **B5 UNDECIDED, B6 UNDECIDED** (no registered power check; 20 H3 contrasts owe one). **F4 (negative actinide transfer) HOLDS** on the conservative reading — either registered 95 % interval on any registered design: the BCa interval of WITHOUT vs WITH on V2 excludes 0 ([+0.0031, +0.0783], Δ +0.0264) while the percentile interval does not, so the percentile-only reading does not hold and the two disagree on V2 (`h3_f4.json → failure`, `failure_percentile`, `failure_bca`). Then confirmation (orchestrator) and process |
| report, figures 7–13 | built, not run: the discovery run is COMPLETE and the **M3–M7 ladder has run** (`h3.ladder_complete` satisfied — every step in a done / skipped status), so neither runner refuses on that gate any more; the deployed predictor is §11's "retained ladder configuration", **M0 (= B5)** by addendum 3 item 2, and every lookup of it in the report and the figures takes `arm_alias` (= B5), the name its records, prediction frames, `_support` files and `tables/discovery_summary.csv` rows carry, while the printed sentences keep M0 (task X finding V-P02). `scripts/g19_make_figures.py` + `gen19ct/evaluation/figures.py` (brief §25 items 7–13) and `scripts/g19_build_report.py` + `gen19ct/evaluation/report.py` (`GEN19_REPORT.md`, `SUMMARY.md`, `decisions/D02_factorization.md`, `tables/claims.json`; brief §20, §22, §29, §34; pre-registration §9, §10, §15–§17, §19). Everything is generated from files: every printed number cites a path and a key (`tables/report_numbers.csv`) and is re-resolved from that source after writing; an absent input prints `not computed (input missing: <path>)`. A dry in-memory build on the current tree re-resolves every one of its 65 pre-seal numbers and lists 21 inputs as missing. `tests/test_report.py` (11 tests, synthetic inputs only). **D02 is already written**: `scripts/g19_write_d02.py` runs the D02 generator (`report.d02_text`) and nothing else, under the same seal gate and the same ledger, because that decision file is answerable from the completed run while the report is not (99 numbers, all re-resolved: `tables/d02_numbers_verification.csv`); `g19_build_report.py` regenerates it with the ladder rows filled in |
| raw fold records | the 3,901 excluded files of the run (240.3 MB: per-fold predictions + provenance blocks under `evaluation/discovery/<arm>/<design>/s<seed>/`, the shared `_support` records, the `_s1c_yardsticks` refits and the 16.3 MB runner log) are **not in version control** — `.gitignore` excludes them and `evaluation/discovery/MANIFEST.sha256` records a SHA-256 and a byte count for every one, written and verified by `scripts/g19_manifest_discovery.py [--check]` (`--check` exits 0 on the current tree). Everything a decision file cites by number is in a small committed file. Same convention as `generations/gen18_process/` and `generations/gen16_leads/` |

**Models have now been fitted, and only on the selection half.** Until the discovery run completed
this README said that no predictive model had been trained; that sentence is gone. What still holds,
and is the thing to keep in mind when reading any number in this directory: every fitted score is
**discovery, optimistically biased** — selection half, seed 104729 — and the confirmation half and
the four withheld seeds have never been read (`decisions/decisions.json` → `guard_counters`:
`confirmation_half_rows_seen` 0, `v6_target_rows_scored` 0). No claim is confirmed, and no
confirmation-half prediction exists. In Phase A/B, which the two audit documents report, `log_D` is
still read only descriptively: to compare values of duplicate and near-duplicate records, to test
whether source metadata leaks the target (`g19_audit_leakage.py`), and for the variance decomposition
of feasibility question 13 (labelled DESCRIPTIVE in `g19_feasibility.py`).

The Phase A/B outputs are byte-reproducible: two consecutive runs of the pipeline were compared
file by file in `manifests/phaseAB_determinism.json` (see [Determinism](#determinism)).

## Reading order

1. `GEN19_CLAUDE_CODE_INSTRUCTIONS.md` — the brief (the task, non-negotiable rules, validation designs V0–V7).
2. [`DATA_AUDIT.md`](DATA_AUDIT.md) — Phase A, items 1–14: what the corpus is, its hashes, counts,
   sparsity, Pr/Nd / lanthanide / actinide coverage, alias collisions, duplicates, near-duplicates,
   which §1.4 provenance fields exist, and a list of known traps.
3. [`FEASIBILITY.md`](FEASIBILITY.md) — Phase B, questions 1–13, the V0–V7 support table, the
   feasibility of the §30 actinide / §31 architecture / §32 priors questions, the §19 Pr/Nd table.
4. [`decisions/`](decisions/) — [`D00_corpus_feasibility.md`](decisions/D00_corpus_feasibility.md) and
   [`D01_transfer_signal.md`](decisions/D01_transfer_signal.md) (pre-seal baseline difficulty):
   question, evidence, metrics, verdict, decision, next action.
5. [`preregistration_draft.md`](preregistration_draft.md) — the draft registration of the Phase C+
   evaluation (hypotheses, hold-outs, metrics, baselines, decision rule, thresholds as formulas of
   Phase C difficulty, deviations from the brief). The remaining placeholders are marked `TO BE FINALISED`
   (the banner and the §15 seed digest).

Every number in the three documents cites the file and key it was copied from; `†` marks a row
tally of a cited table rather than a stored field.

## Layout

```
GEN19_CLAUDE_CODE_INSTRUCTIONS.md   the brief
README.md  DATA_AUDIT.md  FEASIBILITY.md  preregistration_draft.md
decisions/     D00_corpus_feasibility.md  D01_transfer_signal.md
gen19ct/       the package (the brief's src/, see Conventions)
  paths.py       every location read or written; digests() = binary + LF-normalised SHA-256
  manifest.py    Run context manager, write_csv / write_json / write_text (LF, sorted JSON)
  data/          load.py (the one archive loader, g19_* derived columns), normalize.py (condition
                 vectors and keys), provenance.py (brief §1.4 availability), leakage.py (duplicates,
                 near-duplicates, cross-publication copies, fold_isolation_check, pair_isolation_check)
  chemistry/     metals.py (metal descriptors, normalize_metal), ligands.py (components, families,
                 donor sites, named extractants), mechanisms.py (mechanism labels),
                 support_graph.py (metal-state x system graph, SupportIndex, hide_cell / hide_cells with
                 component-aware state-level hiding)
  folds/         registered.py (V6_TARGET_ROWS mask and scoring guard, selection / confirmation halves,
                 parent-structure component map); io.py, cell_holdout.py, source_holdout.py, metal_holdout.py,
                 random_split.py (outer folds and the ONE implementation of each section 7 inner design)
  models/        interface.py (arm protocol, RowTable, split-conformal wrapper and its inner splitters, which
                 take their units from folds/), baselines.py (B0-B4 family), mass_action.py (B7),
                 s1c_yardsticks.py (B3x / B3i re-fitted on batched V5-PAIR folds for the section 9 S1(c) contrast),
                 neural.py (M1 / M2), ladder.py (M3-M7 composed on neural.py: mechanism experts, publication-group
                 offset, pairwise loss, physics penalties, heteroscedastic 5-member ensemble with normalised
                 split conformal; nothing in neural.py is edited)
  evaluation/    metrics.py, transfer.py, calibration.py, pairs.py, support.py (section 13 thresholds,
                 support_score components, domain status), discovery.py (plan, records, scorer core), h3.py,
                 power.py, figures.py (brief section 25 figures 7-13 drawn from frames the runner assembles from
                 files; every figure function returns the input files it used and writes its plotted points to
                 figures/data/F*.csv; skipped with a logged reason when an input is missing), report.py
                 (GEN19_REPORT.md, SUMMARY.md, decisions/D02_factorization.md, tables/claims.json GENERATED FROM
                 FILES through one number ledger: every printed number carries its source path + key in
                 tables/report_numbers.csv and is re-resolved from that source after writing; a missing input prints
                 "not computed (input missing: <path>)", never a placeholder)
  process/       gen18_adapter.py (Gen19 prediction records as a gen18 DModel: PredictionTable, Gen19DModel subclassing
                 gen18proc.dmodel._ModelBase with analytic partials, ASSUMED-with-range provenance, the support gate),
                 monte_carlo.py (64 seeded joint D draws through gen18's cascade on one paired LHS design; per
                 operating-point medians, 5/50/95 %, P(purity), P(recovery), P(both), P(feasible)),
                 robust_optimize.py (support rank -> P(both & feasible) -> P(feasible) -> consumption -> stages ->
                 throughput; S2(d) bootstrap stability; F5 audit).  gen18 is imported read-only, never edited
  losses/        empty package reserved for Phases D-G
scripts/       g19_build_extractants.py  g19_build_metals.py  g19_audit_leakage.py
               g19_audit_corpus.py  g19_feasibility.py       (Phase A/B steps)
               g19_run_phaseAB.py      runs the five steps in order; --determinism
               g19_seal_prereg.py      --check / --seal / --commit-seeds / --verify-seeds
               g19_build_folds.py      registered outer folds, per-fold section 13 thresholds, wildcard-copy audit,
                                       nested-certificate safeguard sample (--merge-index: partial rebuild)
               g19_run_preseal.py      the pre-seal closed-form baseline run (B0-B4, B7, FLAT, HEAVIER, intervals)
               g19_update_support_preseal.py   section 13 s4 update of support_score (also called by g19_run_preseal)
               g19_run_discovery.py    the sealed section 7 plan as amended by POST-HOC addendum 1 (B6 -> M2)
               g19_score_discovery.py  selection-half scorer: R19 contrasts, stop rule, ladder M1 / M2, S1 components
               g19_run_ladder.py       the ladder M3 -> M7 AFTER discovery (refuses until discovery is complete and the
                                       stop rule is decided); records under evaluation/ladder/, decisions D04 / D05
               g19_run_process.py      Phase H: the Gen19-fed gen18 process chain for the TODGA Pr/Nd nitrate case
                                       (refuses until discovery is complete, the stop rule is decided and the
                                       confirmation run says S1 passed; --exploratory = labelled transfer-unsupported);
                                       evaluation/process/, figures F14 / F15, decisions/D06
               g19_make_figures.py     brief section 25 figures 7-13 from the discovery / ladder / power /
                                       confirmation FILES after discovery (refuses until discovery is COMPLETE and
                                       the scorer has run): figures/F07-F13, figures/data/*.csv,
                                       tables/s1e_error_vs_support.csv, evaluation/figures/figures_index.json
               g19_build_report.py     GEN19_REPORT.md, SUMMARY.md, decisions/D02_factorization.md, tables/claims.json,
                                       tables/report_numbers.csv from the files (refuses until discovery is COMPLETE
                                       and the M3-M7 ladder is done); exit 3 if any printed number does not re-resolve
                                       from its cited source
               g19_write_d02.py        decisions/D02_factorization.md ALONE, under the same seal gate, ledger and
                                       re-resolution: g19_build_report.py's D02 generator called before the ladder
                                       exists, editing no report-stage code (tables/d02_numbers*.csv)
               g19_manifest_discovery.py  evaluation/discovery/MANIFEST.sha256: sha256 + bytes for every fold record
                                       the .gitignore excludes; --check verifies (exit 1 on a mismatch, on a missing
                                       file, or when a listed file is not in fact ignored)
tests/         conftest.py, test_load, test_normalize, test_provenance, test_leakage, test_metals,
               test_ligands, test_support_graph, test_seal_prereg, test_manifest,
               test_registered_folds (marker: slow), test_folds, test_baselines, test_metrics,
               test_transfer_stats, test_inner_designs, test_support, test_resolutions, test_neural,
               test_discovery_runner, test_ladder (M3-M7 components and the ladder runner's gate, synthetic only),
               test_process (adapter vs ConstantD, partials vs finite differences, mass balance on every draw,
               Monte Carlo reproducibility, P(both) arithmetic, UNSUPPORTED never wins, S2(d), the runner's gate;
               synthetic tables and gen18's test systems only),
               test_report (the report / figure / D02 generators on synthetic input files shaped like the real
               writers' outputs: every printed number traceable to a cited file or a sealed-text literal, a removed
               input becomes "not computed (input missing: ...)", no number is invented when every input is absent,
               the figures write PNG + data CSV and report their inputs, the two runners' gates refuse)
descriptors/   metals.csv + metals_sources.md, extractant_components.csv, extractant_systems.csv,
               family_rules.json
data_audit/    machine-readable Phase A/B tables (CSV/JSON), one family per script:
               counts / sparsity / matrix_* / *_coverage / columns / dataset_hashes  (g19_audit_corpus)
               leakage_* / metadata_availability                                    (g19_audit_leakage)
               metal_alias_audit / metal_descriptor_coverage                         (g19_build_metals)
               ligand_alias_collisions / family_coverage / named_extractant_presence (g19_build_extractants)
               feasibility.json / feasibility_* (incl. feasibility_halves.csv)       (g19_feasibility)
figures/       F01_observation_matrix, F02_density_by_metal, F03_density_by_family,
               F04_lanthanide_coverage, F05_actinide_coverage, F06_condition_coverage (PNG, dpi 130);
               F07_pred_vs_measured, F08_error_vs_support, F09_uncertainty_calibration, F10_metal_embedding,
               F11_extractant_embedding, F12_prnd_reconstruction, F13_direction_confusion (g19_make_figures, after
               discovery; the pre-seal F07 / F08 are F07_preseal_* / F08_preseal_*), F14 / F15 (g19_run_process);
               data/F*.csv = the plotted points of every generated figure
manifests/     <script>.json (deterministic), run_info/<script>.json (volatile),
               phaseAB_determinism.json (the two-run byte comparison),
               g19_build_folds_incremental.json (partial fold builds; merge.full_build_link supersedes the full
               build's INDEX.json / wildcard_copy_crossings.csv digests)
folds/         registered outer folds, INDEX.json                                     (g19_build_folds)
evaluation/preseal/  difficulty.json (as-run, quoted by §9), difficulty_resolved.json, support_status.json,
               support/<job>__support_score.parquet; tables/ (preseal_*)  figures F07, F08  (g19_run_preseal)
models/ process/   empty output directories for later phases
evaluation/figures/  figures_index.json (which figure was written or skipped, why, from which files)  (g19_make_figures)
evaluation/report/   report_inputs.json (every input the report read: exists / missing)                (g19_build_report)
GEN19_REPORT.md  SUMMARY.md  tables/claims.json  tables/report_numbers.csv  tables/report_numbers_verification.csv
               generated by g19_build_report after discovery (not yet written: discovery is running)
```

Empty directories are not tracked by git. No Phase A/B output is larger than 5 MB (the largest is
`data_audit/leakage_near_duplicates.csv`, 4,779,596 bytes per `manifests/g19_audit_leakage.json`).
`g19_audit_leakage.py` raises rather than write a pair listing above 20 MB (`MAX_CSV_BYTES`). Two pre-seal
outputs are larger (`manifests/g19_run_preseal.json` → `files_over_5MB`; both digests are under `outputs`):
`evaluation/preseal/predictions/V0__rows.parquet` (15,977,984 bytes) is listed in the local `.gitignore`, and
`tables/preseal_summary.csv` (6,151,134 bytes, 6.15 MB) is committed as a cited table, not ignored:
`preregistration_draft.md` §3.2 and `decisions/D01_transfer_signal.md` quote values from it, and
`evaluation/preseal/difficulty_resolved.json` is read from it.

## How to run

Everything runs from the repository root. Plain `python` on this machine is a broken stub; use
the project interpreter. The environment variables are the same for every command:

```
export PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer
```

**Phase A/B pipeline** (five processes in dependency order; about 4.5 minutes per run, per-step
runtimes in `manifests/phaseAB_determinism.json › run_info`):

```
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_phaseAB.py            # all steps
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_phaseAB.py --list     # show the order
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_phaseAB.py --steps g19_audit_leakage,g19_feasibility
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_phaseAB.py --determinism   # two full runs + comparison
```

| # | step | writes | reads from earlier steps |
|---|---|---|---|
| 1 | `g19_build_extractants` | `descriptors/extractant_*.csv`, `family_rules.json`; `data_audit/ligand_alias_collisions.csv`, `family_coverage.csv`, `named_extractant_presence.csv` | — |
| 2 | `g19_build_metals` | `descriptors/metals.csv`, `metals_sources.md`; `data_audit/metal_alias_audit.csv`, `metal_descriptor_coverage.csv` | — |
| 3 | `g19_audit_leakage` | `data_audit/leakage_*.csv`, `leakage_summary.json`, `metadata_availability.csv` | — |
| 4 | `g19_audit_corpus` | `data_audit/counts.json`, `sparsity.json`, `matrix_*.csv`, `*_coverage.csv`, `columns.csv`, `dataset_hashes.csv`; F01, F02, F04–F06 | — |
| 5 | `g19_feasibility` | `data_audit/feasibility.json`, `feasibility_*.csv`; F03 | 1 (`extractant_*.csv`, `named_extractant_presence.csv`), 2 (`metals.csv`), 3 (`leakage_publication_components.csv`) |

Each step can also be run on its own as
`.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/<step>.py`; every threshold of
`g19_feasibility.py` is an argument (`--help`). The runner stops at the first failing step
(exit 2) and, after a normal run, verifies every step manifest's output digests against disk.

**Tests**

```
.venv/Scripts/python.exe -m pytest generations/gen19_chem_transfer/tests -q -p no:cacheprovider                 # all
.venv/Scripts/python.exe -m pytest generations/gen19_chem_transfer/tests -q -m "not slow" -p no:cacheprovider    # fast
.venv/Scripts/python.exe -m pytest generations/gen18_process/tests -q -m "not slow and not validation" -p no:cacheprovider   # gen18 regression
```

The gen19 tests change no file in the tree (checked by hashing every file before and after a full
run; the S1(c) yardstick re-fit test writes its synthetic fold files to pytest's `tmp_path` only). Full-archive sweeps
are marked `slow`: `test_registered_folds.py` runs the registered V1 (104 folds), V2 (23 states), V5 (224 cells) and
V6 (13 systems) splits through `fold_isolation_check`, and other slow tests read the pre-seal outputs. The whole suite
(458 tests, slow included) takes about 350 s (348.8 s on 2026-09-15, 12-thread CPU) and `-m "not slow"` (431 tests)
about 40 s. `test_manifest.py`
also checks that `load.PROVENANCE_COLUMNS` covers the schema's provenance columns and that every JSON output is
strict JSON.
`test_metals.py` rebuilds `descriptors/metals.csv` into a temporary directory and compares it byte for byte;
`test_seal_prereg.py` works on temporary copies and never seals the real draft.

**Pre-registration** (read-only status check; sealing is an orchestrator decision taken after the
Phase C baselines, never by a builder):

```
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_seal_prereg.py --check
```

**Pre-seal Phase C** (closed-form baselines only; the fitting phase took 2,484 s on 2 workers,
`manifests/run_info/g19_run_preseal_jobs.json` → `total_runtime_s_compute`):

```
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_preseal.py                  # fit + score everything
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_preseal.py --skip-compute   # re-aggregate stored predictions
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_preseal.py --tables-only    # rebuild tables/ only (run_info/g19_run_preseal.json → runtime_s)
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_update_support_preseal.py       # the section 13 s4 update alone (idempotent)
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_folds.py --designs V5P_BATCHED,V5PAIR_BATCHED --merge-index   # partial fold rebuild
```

Every mode aggregates under the orchestrator's resolutions of 2026-09-15: the registered V1 outer-fold unit (per-group
reading printed as exploratory, column `unit_reading`), scoring-filter statuses from the R19 registry (the wildcard-copy
filter registered for V1, V5, V5-P and V5-PAIR; the pre-seal run computes it for V5-primary and V1-copy only, whose
scored rows carry one fold each) and `support_score` under the resolved s4 reading (the pre-seal script calls
`g19_update_support_preseal.apply_support_update`). `evaluation/preseal/difficulty.json` is the as-run record whose SHA-256
§9 quotes: a full-data run refuses to change it and writes the resolved numbers to `difficulty_resolved.json`.

`--skip-compute` fits nothing. It requires the stored predictions, support feature files, guard log and inner-guard record
to match `manifests/g19_run_preseal.json`, rewrites every other output, and records under `aggregation_rebuild` the
carried compute-run record and the outputs a later manifest had superseded.

`--tables-only` fits nothing. It refuses unless every non-table output still matches
`manifests/g19_run_preseal.json`. It re-renders every output in a temporary directory, aborts if a
non-table output would change, and writes only the tables whose bytes changed. The compute run's git
HEAD, arguments and code digests are kept under `tables_only_rebuild.compute_run` (and the aggregation run's record under
`aggregation_rebuild`).

The partial fold rebuild asserts that every full-build design file is unchanged and records in
`manifests/g19_build_folds_incremental.json` → `merge.full_build_link` the full-build digests of the two files it
rewrites (`folds/INDEX.json`, `folds/wildcard_copy_crossings.csv`), so the full-build manifest's stale entries for them are
superseded, not unexplained (`tests/test_manifest.py::test_preseal_chain_manifests_match_disk_or_are_superseded`).

**Discovery** (the sealed §7 plan as amended by POST-HOC addendum 1; **run and scored — see
[Status](#status)**: 1,461 registered fold records, 76.6 h, exit 0, selection half and seed 104729 only. The
commands below are how it was run and how a rerun is compared with it, not work still to do.) Every
command first calls `g19_seal_prereg.py --check`, pins the sealed digest `135842499a86…` and refuses
unless the expected number of POST-HOC addenda is below the footer — one while discovery ran, two
since addendum 2, three since addendum 3, taken per stage from `manifests/digest_registry.json`
(`--expect-addenda N` overrides deliberately, and never the addendum text digest); the digest, the
addendum count and the SHA-256 of the addendum text go into every fold record and manifest.

```
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_discovery.py --dry-run
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_discovery.py --benchmark-addendum1
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_discovery.py --workers 2
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_discovery.py --workers 2 --only B6
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_discovery.py --workers 2 --max-hours 6
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_score_discovery.py
```

| command | what it does |
|---|---|
| `--dry-run` | enumerates the plan with fold counts, prices it with the newest benchmark, prints the per-stage cumulative wall clock, and writes `evaluation/discovery/benchmark/plan_addendum1.txt`. Fits nothing |
| `--benchmark-addendum1` | fit-only timings per arm (B6, B5, FLAT_CAT, B8, M1, M2) on the TRAINING rows of one V5-primary batched fold under the simultaneous inner design; reuses the sealed-plan V1 / V2 measurements; writes `cost_estimate_addendum1.json` / `.md` with per-stage checkpoints and the fits-in-60 h verdict. **No test row is predicted** |
| `--benchmark` | the superseded sealed-plan measurement, kept so that the 1,535.7 CPU-h estimate the addendum rests on stays reproducible |
| (no flag) | runs the plan stage by stage, ≤ 2 workers, resumable per fold (a record whose digest matches is skipped); `--only` filters (`B6`, `M2:V5`, `V1`, `safeguard`), `--max-hours` is an operator pause (never a registered demotion), `--steps point` defers the intervals |
| `g19_score_discovery.py` | scores the **selection half only** into `tables/discovery_*.csv`, `evaluation/discovery/contrasts_*.csv` and `decisions/decisions.json` (stop rule, ladder, S1 components, freezing screen, the `addendum_1` block) |

The plan, in order: the nested-certificate safeguard → B6 / B6r0 (V5 exact and batched with the
batched-vs-exact check, V1 exact and ten-fold with the ten-fold check, V2, and the V5 strict /
HNO3-only refits) → the seed-104729 main designs B5 = M0, FLAT_CAT, B8 → M1 → M2 → the stop rule →
the S1(c) V5-PAIR run (M1 prerequisite, M2; B3x / B3i are re-fitted on the same folds by the scorer)
→ the strict / HNO3-only refits of the H1 / H4 arms → conditional freezing-candidate runs →
`M3+ not implemented`. Everything addendum 1 does **not** run — the other four discovery seeds, the
loose / cell-only / parent-structure / Sr(III)-dropped V5 refits, the heavy-arm V5-P runs, the
V1 / V2 refit sensitivities and the comparator-interval jobs — is emitted as a `marker` job naming
it, so nothing is a silent absence. The 60-hour budget and its demotion order (M7 → M3 only) are
unchanged; exhausting it demotes nothing else.

**Ladder M3 -> M7** (`scripts/g19_run_ladder.py`; pre-registration section 6 ladder rows M3-M7 and the
M-model training settings, section 7 items 3-6, section 12, addendum 1). Runs only after discovery: the gate
requires `g19_seal_prereg.py --check` with the registered digests, a COMPLETE discovery run (the progress
ledger `manifests/run_info/g19_run_discovery_progress.json` lists the final-stage marker `M3+ not implemented`
without job errors, `evaluation/discovery/records_index.csv` exists, and every `fit` job of the current plan
has a current record with steps `point` + `intervals` for every fittable fold), the scorer's
`evaluation/discovery/decisions/stop_rule.json` with `stop` in {true, false} and `decisions.json` -> `ladder`
with boolean `kept` for M1 and M2. `--check-only` prints the gate verdict.

```
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_ladder.py --check-only
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_ladder.py --workers 2
```

Steps run in section 7 item 3 order on the seed-104729 main designs discovery gave M2 (identical JobSpecs with
the arm renamed), each searching only its own hyperparameter with the outer fold's retained predecessor values
fixed (M3 none; M4 tau in {0.1, 0.3, 1.0}; M5 lambda_pair in {0.3, 1}; M6 lambda_phys in {0.1, 1}; M7 none);
after each step the section 6 keep / remove rule is evaluated with the scorer's R19 machinery (ladder scope on
V5-primary, TOST epsilon 0.05 on V1 / V2, selection half, seed 104729) and a removed step is skipped by the next
step's chain; an UNDECIDED step (a design frame missing) stops the ladder with status `undecided` -- nothing builds
past it until the input is resolved (task X finding V-05). Under the stop rule M3-M6 are `exploratory_not_run` and
only M7 runs. The 60 h budget counts the discovery wall clock; on exhaustion the remaining steps are demoted (M7 -> M3).
After the H5 block, M3's strict / HNO3-only refit frames are consumed: the H4 / H5 / ladder rows of `M3 vs M2` are
re-evaluated, `contrasts_M3.csv` rewritten and the earlier rows kept as `contrasts_M3_before_refits.csv` (finding
V-04). The M6 physics terms record whether they acted on any row (`hinge_active`, `smoothness_active` in every fit
record); an M6 decision or an H5 M6a / M6b toggle whose term acted on no fold is labelled `vacuous`, never read as a
component on / off comparison (finding VL2-07). The gate's file checks (ledger marker, `records_index.csv`, the
decision files) run before the corpus is loaded (finding VL2-04). Outputs: `evaluation/ladder/<arm>/...`
records in the discovery schema, `evaluation/ladder/decisions/ladder.json` + `contrasts_<step>.csv`,
`evaluation/ladder/M7/metrics.json` (coverage 50 / 80 / 95, width, CRPS, Spearman(|error|, SD), coverage by
domain status, S1(d), F3, "knows when it does not know"), `tables/ladder_decisions.csv`,
`decisions/D04_mechanism_experts.md`, `decisions/D05_uncertainty.md`. Every number is selection-half,
optimistically biased and never confirmed.

**Phase H process integration** (`scripts/g19_run_process.py`; pre-registration section 14 in full, section 13
domain status, section 9 S2(d), section 10 F5; brief sections 1.6, 17-19, 25 items 14-15, 27, 29). The gen18 cascade
solver is not rewritten: `gen19ct/process/gen18_adapter.py` subclasses `gen18proc.dmodel._ModelBase` so that
`solve_cascade` uses the Gen19 D source through the same vectorised `core()` as gen18's own models, with analytic
partials (bilinear table interpolation in log acid, gen18's ideal depletion term in the free ligand). gen18's
`ProvStatus` has no model-predicted value, so Gen19 predictions enter as `ASSUMED` with `ASSUMED_PLACEHOLDER`, a
declared range (the table-wide 95 % interval of log D), `source.kind = "model"` and a `MODEL_DERIVED` note; every
Gen19-side record carries `model_derived = True`. The gate requires `g19_seal_prereg.py --check` with the registered
digests, a COMPLETE discovery run (progress-ledger marker `M3+ not implemented` without job errors,
`records_index.csv`, `wall_clock.json` at the final stage, and `h3.discovery_complete`'s verified record sets), the
scorer's `stop_rule.json` with `stop` decided (`stop = true` refuses: "H7 not run"), and the confirmation run's
`evaluation/confirmation/decisions/confirmation.json` (schema `gen19.confirmation.v1`, keys `S1.passed`,
`S1.deployed_predictor`, `S2.passed`, `V6.run`, `seeds.verified`) saying `S1.passed = true`; `S1.passed = false`
refuses always; a missing decision refuses unless `--exploratory`, and unless S1, S2 and V6 all hold every output is
labelled `transfer-unsupported` (never a headline). `--check-only` prints the gate verdict.

```
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_process.py --check-only
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_process.py            # 64 draws x 1000 LHS points
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_process.py --exploratory --draws 64 --n-lhs 1000
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_process.py --decision-only   # D06 from the files
```

Inputs (`evaluation/process/inputs/`): `todga_prnd_predictions.csv`, the deployed predictor's Gen19 prediction records
for Pr and Nd on a (log acid, log ligand) grid (`gen18_adapter.PREDICTION_COLUMNS`; when absent the runner writes the
exact conditions to predict to `todga_prnd_prediction_request.csv` and refuses), and `prnd_residual_correlation.json`
(`{"rho": r, "n_pairs": n, "source": ...}`, the section 14 Pr/Nd residual correlation from inner-fold comparable-pair
residuals; required in registered mode). The case is section 14's: system `sys_5cb78e5000d40860` (TODGA / nitrate /
aliphatic), feed `gen18 cases/todga_prnd_feed.json`, spec grid `gen18 cases/prnd_spec.json` (purity {0.95, 0.97, 0.99} x
recovery {0.80, 0.85, 0.90}; every value an assumed placeholder), design space `gen18 config/design_spaces.json`
(diglycolamide family, the entry's `TODGA|20-30C` intervals, saponification / complexant / salting anion / dilution /
bleed fixed at 0) intersected with the prediction table's supported box. Monte Carlo: 64 joint draws, each section
14's "one ensemble member plus a residual drawn from the calibrated conformal scale": the prediction record carries the
M7 member means `member_logD_0..4` and the calibrated normalised-conformal multiplier `conformal_q95` (columns of the
request file the deployment step must fill); one member per draw, shared by Pr and Nd, plus one Gaussian residual per
metal with scale `std_logD * conformal_q95 / 1.96` (common over the grid, Pr / Nd correlated with `rho` through a
Gaussian copula). A table without member columns is refused in registered mode; `--exploratory` falls back to the
truncated Gaussian on the record, flagged (`draw_mode`; task X finding V-07). Each draw goes through
`optimize.lhs_pareto` with the SAME LHS seed (paired candidates); per operating point median and 5/50/95 % purity and
recovery, P(purity >= t), P(recovery >= t), P(both), P(both & feasible), P(phase / loading constraints: no
`THIRD_PHASE_RISK`, `LOADING_CAP_HIT`, `HIGH_LOADING`), consumption, stages, throughput. Objective per spec cell, the
sealed order: support rank (any UNSUPPORTED or outside-table prediction = ineligible; CONDITION_ / FAMILY_EXTRAPOLATION
ranks below clean) -> P(both targets) -> P(feasible) -> consumption -> stages -> throughput; the joint P(both & feasible)
is printed beside and does not rank (finding V-06). A recipe whose D-source usage was not tracked (`statuses_used`
absent or empty for a converged candidate) is refused, never ranked as clean (finding VL2-03). Checks: S2(d) (20
bootstrap re-rankings of the draws, seed 19, top recipe kept
= stage counts +- 1 and O/A within 10 %, pass >= 0.80), F5(i) (an UNSUPPORTED winner is an `AssertionError`), F5(ii)
(barred UNSUPPORTED + CONDITION_EXTRAPOLATION vs allowed, per cell; the allowed variant is a second run with
`allow_unsupported=True` when the table carries UNSUPPORTED cells), the gate-lifted re-ranking. Outputs:
`evaluation/process/` (`process_table.csv` byte-identical for the same seed, `operating_points.csv`, `rankings.csv`,
`winners.csv`, `stability.json`, `f5.json`, `summary.json`, `inputs_used.json`), `tables/process_*.csv`, figures
`F14_process_pareto_uncertainty.png` and `F15_probability_of_specification_map.png`,
`decisions/D06_process_integration.md`, `manifests/g19_run_process.json` (its own code digests and the digests of
the gen18 files it imports). Readings that are not literally in section 14 are listed in
`g19_run_process.REGISTRATION_READINGS` and written to `summary.json` -> `readings`. **Nothing has run**: no
confirmation decision, prediction table or correlation file exists yet.

**Order of the post-discovery steps** (each runner refuses to start unless `g19_seal_prereg.py --check` exits 0
with the registered digests and the discovery run is COMPLETE — `h3.discovery_complete`: `wall_clock.json` reached
stage `10_not_implemented`, every stage of the current plan done, every `fit` job's record set verified against the
digests the current code, fold files and plan state produce; the scorer's decision files gate the steps that read
them). Since POST-HOC addendum 2 each of these stages is also a **digest-registry stage**: it is registered in
`manifests/digest_registry.json` from the tree before it first runs, and a record is verified against its stage's
registered digests, never against live code (addendum 2 item 5). **Step 1 has run; step 2 is what runs next.**

1. ~~`g19_score_discovery.py`~~ — **DONE** (exit 0): selection-half R19 contrasts, stop rule, ladder M1 / M2, S1
   components, freezing screen (`evaluation/discovery/contrasts_*.csv`, `decisions/decisions.json`, `stop_rule.json`).
   `decisions/D02_factorization.md` is written from its outputs by `g19_write_d02.py`;
2. **→ `g19_run_ladder.py`** — M3 → M7 under the stop rule, on the ladder's own 40 h budget (addendum 2; no hour spent
   yet, `budget.ladder.ledger_exists` false) (`evaluation/ladder/`, D04, D05). **Required by steps 3, 4 and 7**:
   the configuration deployed for lanthanide prediction is section 11's "retained ladder configuration", read by
   `h3.deployed_configuration` from `evaluation/ladder/decisions/ladder.json` (highest kept step M7 > … > M3, from a
   registered — not stop-rule — run) and then the scorer's M2 > M1 > **M0**; the H3, figure and report
   runners refuse until every step M3–M7 is in a done / skipped status (`h3.ladder_complete`; task X finding V-01).
   **POST-HOC addendum 3 item 2**: the ladder kept M0 and dropped M1 and M2, so the retained — and therefore
   deployed — configuration is **M0 (= B5)**, and addendum 2's remaining order (M2 / B6 by the stop-rule scope, then
   B3i) is reached only when the ladder retains no step at all. Addendum 3 item 3: because neither M1 nor M2 is
   retained, M3–M7 are `not_run` (§6's components belong to the factorised model) *before* the budget check, so the
   ladder's own 40 h are not consumed;
3. `g19_run_h3.py` — the section 11 actinide ablation and F4 (`evaluation/h3/`, D03). With **M0 deployed** (addendum 3
   item 2) the model arms are B5 and B6 — M0 *is* B5, so the deployed arm is also one of §11's transparent references —
   WITH is B5's own discovery record and the three transformed arms are refitted with `h3.FrozenBoosted` at that
   record's frozen CatBoost configuration and tree count per fold, with the discovery runner's cross-fitted
   calibration; a deployed ladder step is refitted instead
   with `h3.FrozenLadder` at its ladder record's configuration, epochs and seeds (M7: members + normalised conformal); a
   closed-form comparator is fitted on the exact leave-one-cell-out folds as in discovery (finding V-03); a refit whose
   inner folds differ from the WITH record's is recorded `not_calibrated_inner_folds_differ_from_the_with_record`
   (finding VL2-05). Completeness is per **contrast record set** (`h3.CONTRAST_UNIT_RULE`): a complete
   `<arm>:<transform>@<design>` leg is scored and carries its verdict, an incomplete one carries `NOT_RUN` when the 20 h
   cap stopped it and `INCOMPLETE_GUARD_FAILURE` otherwise, and contributes no row, delta or verdict input either way.
   `scripts/g19_h3_guard_diagnosis.py` is the read-only companion that names why a leg's guard failed (it fits nothing
   and writes only `evaluation/h3/decisions/isolation_guard_diagnosis.json`);
4. `g19_run_power.py` — the section 8 signal-injection power check for every failed H1 / H1b / H3 contrast (the
   closed-form comparator on the exact folds, both arms asserted to score the un-injected rows; finding V-03) and the
   reliability-before-correlation report, per unit: split-half of the unit's own publication groups where it has ≥ 4,
   the delete-one-publication jackknife for 2–3 groups, single-group units counted, the quantity's reliability the
   minimum over the computed branches (finding V-02); no `V6_TARGET_ROWS` row is read and the observed-logSF amplitude
   uses the selection half only (finding VL2-06) (`evaluation/power/`, `tables/power_kappa.csv`,
   `tables/reliability_before_correlation.csv`);
5. **confirmation (orchestrator, section 15)** — the ≤ 5 frozen claims on the withheld seeds and V6 once; writes
   `evaluation/confirmation/decisions/confirmation.json` (schema `gen19.confirmation.v1`) and, for the report and
   figure 12, the V6 tables (`evaluation/confirmation/v6_rows.csv`, `v6_pairs.csv`, `v6_systems.csv`) — no builder runs
   this step;
6. `g19_run_process.py` — Phase H (gated by the confirmation decision; `evaluation/process/`, F14 / F15, D06);
7. `g19_make_figures.py` then `g19_build_report.py` — figures 7–13, `GEN19_REPORT.md`, `SUMMARY.md`, D02,
   `tables/claims.json`; both require the ladder (step 2) and print every later step as `not computed (input missing:
   ...)` / `not run`. Every runner's first gate is the cheap `wall_clock.json` check (`h3.refuse_unless_cheap_complete`),
   before the feasibility frame or the corpus is loaded (finding VL2-04).

**Figures 7–13 and the report** (`scripts/g19_make_figures.py`, `scripts/g19_build_report.py`; `--check-only` prints
each gate). The figure runner assembles frames through the scorer's verified `Store` (stale records raise) for the
deployed predictor (`h3.deployed_configuration` on `decisions.json`) and the V5 lookup comparator B3i: F07 predicted
vs measured under V5 / V1 / V2, F08 cell error vs `support_score` per domain-status category with the S1(e) statistic
(Spearman, system-cluster bootstrap 10,000 × seed 19) written to `tables/s1e_error_vs_support.csv`, F09 the
reliability diagram by design and coverage by domain status, F10 / F11 the learned embeddings (from
`evaluation/power/embeddings/*.csv`, which nothing writes yet — skipped with the reason until `g19_run_power.py
--include-learned` is implemented), F12 the V6 Pr / Nd reconstruction from the confirmation files only, F13 the
direction confusion matrices on the V5-PAIR pairs (from the verified M2 record + the pairs parquet) and the V6 pairs, one
panel per (design, threshold): |logSF| >= 0.3 on every design (the registered primary reading) plus the additional
>= 0.1 reading on V6 only, flagged `registered_primary` in `figures/data/F13_direction_confusion.csv` (finding V-08).
The report answers the ten questions of brief §34 in order (one sentence first — "Under deliberately hidden
chemistry, the system can / cannot / has not been shown to …" — then the evidence with regime columns: design, half,
seeds, averaging unit, parameter status; both comparators — the constant baseline B0 and the cheapest sensible
alternative B3i, gen13 Addendum 3; the FLAT floor for logSF; R19 verdicts with BH-adjusted p beside the raw p; TOST;
the `reduced sensitivity set (addendum 1)` label; `status: discovery / confirmation / not run` per claim), then the
S1 / S2 / F1–F6 verdict table, what was not run and why (addendum 1, stop rule, demotions), the §17 deviations and the
POST-HOC addenda (parsed from the sealed text) with every runner reading that still needs an addendum, and
reproducibility (git HEAD, sealed and addenda digests, dataset hash, seed commitment, manifests — the manifest and
output counts and the digests are ledger entries citing `manifests/` and the files that hold them, finding VL2-01).
Q10's support category refuses an unknown status token and maps `OUTSIDE_TABLE` to unsupported (finding VL2-02).

```
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_make_figures.py --check-only
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_make_figures.py            # F07-F13 (+ --only F07,F08)
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_report.py --check-only
.venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_report.py            # exit 3 if a number does not re-resolve
```

## Conventions

- **Package name `gen19ct`, not `src/`.** Every generation in this repository owns a uniquely
  named package (`gen13sep`, `gen18proc`, ...) so several can sit on `sys.path` at once; the brief's
  §26 module tree is kept inside it (`gen19ct/data`, `chemistry`, `folds`, `models`, `losses`,
  `evaluation`, `process`). Scripts put `generations/gen19_chem_transfer` on `sys.path` themselves;
  `tests/conftest.py` does the same.
- **Document names.** Human documents are capitalised (`README.md`, `DATA_AUDIT.md`,
  `FEASIBILITY.md`, later `GEN19_REPORT.md`, `SUMMARY.md`) as in gen18; the registration keeps the
  brief's lower-case name (`preregistration_draft.md`, sealed as `preregistration.md` with a
  SHA-256 footer and `manifests/prereg_sha256.txt`). Decisions are `decisions/Dnn_<topic>.md`.
- **One loader, derived columns prefixed `g19_`.** `gen19ct.data.load.load_archive()` never drops
  a row or rewrites an archive column; it adds `g19_publication_id`, `g19_publication_status`,
  `g19_publication_refs`, `g19_study_id`, `g19_metal`, `g19_ox`, `g19_metal_state` (`"Nd(III)"`,
  `None` when the state is unknown — never imputed), `g19_tier` (`MODEL` / `TARGET_ONLY` /
  `NO_TARGET`; a label, not a filter) and `g19_bundle_exp_id`. `load_model_rows()` is the MODEL tier.
  Columns in `load.PROVENANCE_COLUMNS` are never model features.
- **Manifests.** Every script wraps its work in `gen19ct.manifest.Run`, which writes two files:
  `manifests/<script>.json` — deterministic: git HEAD, seed, arguments, archive digest and, for
  every input and output, the repo-relative POSIX path, `sha256` (binary), `sha256_lf` (CRLF→LF
  normalised, text files only) and `bytes`; and `manifests/run_info/<script>.json` — volatile:
  date, runtime, hardware, package versions. A text file verifies when either digest matches
  (`paths.matches`), so a CRLF checkout is not a reproducibility failure (brief §24). Because
  git HEAD is part of the deterministic manifest, a rerun after a new commit changes that one field.
- **Excluded artefacts carry a digest, never a promise.** Bulk run output is kept out of version
  control and its SHA-256 recorded, the convention of `generations/gen18_process/.gitignore` +
  `results/MANIFEST.sha256` and of `generations/gen16_leads/`. Here that is the discovery run's
  3,901 raw fold records and logs (240.3 MB): `.gitignore` excludes
  `evaluation/discovery/*/*/s*/` and `evaluation/discovery/logs/`, and
  `evaluation/discovery/MANIFEST.sha256` lists `sha256  path  bytes` for every excluded file, LF,
  sorted by path. `scripts/g19_manifest_discovery.py` writes it and `--check` verifies it: a changed
  or missing file fails, a newly excluded file that is not yet listed is reported, and a *listed*
  file that git does not in fact ignore fails too (the manifest may not claim cover for a file about
  to be committed). Verification goes through `paths.matches`, so a CRLF checkout of a text record is
  not a reproducibility failure (brief §24). Everything any decision file or the report cites by
  number lives in a small committed CSV, JSON or Markdown file; the manifest is what makes a
  regenerated run comparable with the one those files were written from.
- **Writers.** CSV/JSON/text go through `write_csv` (LF, `index=False`, `%.6g`), `write_json`
  (sorted keys, indent 2, LF) and `write_text`. Figures use matplotlib's Agg backend, PNG at dpi
  130, one scientific question per figure stated in its title.
- **Read-only inputs.** `dataset_all_metals/`, the 14-lanthanide bundle
  (`dataset with 3D structures/dataset.parquet`, read only with `columns=[...]`) and
  `generations/gen18_process/` are never modified; gen18 is imported read-only through
  `paths.add_gen18_to_path()`.
- **Labels.** Anything not verified against a source is labelled INFERRED; a missing
  chemistry value stays missing with an `NA_REASON` / reason column instead of a recalled number.

## Determinism

`scripts/g19_run_phaseAB.py --determinism` runs the five steps twice and compares the SHA-256 of
every file under `data_audit/`, `descriptors/`, `figures/` and every `manifests/<script>.json`
between the runs, with `PYTHONHASHSEED` left unset so hash-order dependence would show. It also
compares run 1 with the tree as it stood before the check (the outputs the documents cite),
verifies every step manifest against disk, lists outputs no manifest claims, and confirms no code
file and no git HEAD changed during the check. The record, with a per-file identical/different
status, is `manifests/phaseAB_determinism.json` (its `run_info` block is volatile).

Result of the check rerun on 2026-09-15 at git HEAD 40f6a75, after the residual corrections and the state-level V5 hiding
(`verdict` and `counts_*` keys of the record):

| comparison | files | identical |
|---|---|---|
| run 1 vs run 2: step manifests | 5 | 5 |
| run 1 vs run 2: `data_audit/` (CSV/JSON) | 50 | 50 |
| run 1 vs run 2: `descriptors/` (4 CSV/JSON + `metals_sources.md`) | 5 | 5 |
| run 1 vs run 2: `figures/` PNG | 6 | 6 |
| tree before the check vs run 1 (all of the above) | 66 | 66 |
| run 1 vs run 2: `manifests/run_info/` (volatile, expected to differ) | 5 | 0 |

Verdict `DETERMINISTIC: true`: every step manifest's output digests match the files on disk,
every file in the output directories is claimed by a step manifest, and none of the 26 code files
under `gen19ct/` and `scripts/g19_*.py` changed during the check. Because the tree before the check
was reproduced exactly, the numbers cited by `DATA_AUDIT.md`, `FEASIBILITY.md`, `D00` and the draft
pre-registration are the pipeline's current output.

## Known open issues

Reported by the Phase A/B agents and not yet resolved in shared code; each is detailed in
`DATA_AUDIT.md` / `FEASIBILITY.md` / `decisions/D00_corpus_feasibility.md`.

- `g19_publication_id` is built from uncorrected DOIs, so one paper can carry two ids (a DOI with a
  stray trailing character, a page-footer-mangled DOI, and rows co-citing a compilation paper).
  Leave-publication-out folds must use the merged groups in
  `data_audit/leakage_publication_components.csv`, not the raw id (the draft pre-registration does).
- Step manifests record git HEAD but not always the digests of the code that produced them, and while
  the gen19 tree is uncommitted HEAD does not pin that code. Narrowed since: the discovery, scoring
  and later runners record their own `code_sha256` / `code_parts`, every fold record carries the code
  digest it was written under, and POST-HOC addendum 2 added `manifests/digest_registry.json`, which
  pins a code digest per stage and logs every later edit to a closure file with its reason. When an
  addendum is appended, the stages with no records yet are re-registered under the new below-footer
  digest (`register --stage … --force`) while `discovery` and `discovery_candidates` keep their
  entries; a record written under a superseded entry of its own stage still verifies against that
  entry (`registry.verify_record` → `matched_entry` `superseded`), so an addendum invalidates nothing.
  The Phase A/B step manifests are still git-HEAD-only.
- Curation items opened by the verification: the five name–structure-conflict structures (Br-Cosan,
  TPDGA malonamide, TDGA, TBADIPIC, NDDIPIC), the stereo-free vs cis/trans Me2-TODGA identity, and the
  censoring heuristic (exact-decade floors/ceilings) that stands in for a missing detection-limit flag.
- Policy decisions still open: the Sr(III) rows the archive marks implausible, acid values that
  look like converted pH, the DMDOHEMA name attached to two structures, empty polarizability.
