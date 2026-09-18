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
| decision | `decisions/D00_corpus_feasibility.md` (brief §29 format) |
| verification corrections | applied 2026-09-15 (27 findings of three adversarial verifiers, then 7 residual findings of a second pass): component-aware V5 hiding (made symmetric at the metal-state level by a pre-seal design decision), V1 groups merged on archive duplicate groups, same-charge-first B3x, chemistry label fixes (aqueous agents recorded as extractants, EsPyTri, Br-Cosan, quercetin, name–structure conflicts, HDEHP as co-extractant), acid-medium and censoring counts, `V6_TARGET_ROWS`, selection / confirmation halves, fixed success margins |
| pre-registration | **SEALED 2026-09-15** (`preregistration.md`, footer digest `135842499a86…5641`, `manifests/prereg_sha256.txt`; `preregistration_draft.md` is the frozen draft it was built from). It carries **one POST-HOC addendum below the footer** — addendum 1 (2026-09-15, no learned-model outcome seen), the compute-driven reduction of the discovery plan: the sealed §7 plan was priced at 1,535.7 CPU-h against a 60 h budget, so V5 inner tuning became three *simultaneous* inner folds (one fit per configuration per fold, up to 30 cells hidden at once), discovery runs **seed 104729 only** (R19 item 4 NOT_EVALUATED in discovery, unchanged at confirmation), a learned arm runs only the V5 strict and HNO3-only refits (R19 item 6 = "reduced sensitivity set (addendum 1)", the rest named as not run), and V5-PAIR carries M2 with its M1 prerequisite. Designs, folds, halves, hiding, guards, metrics, comparators, margins, the stop rule and confirmation are unchanged. The text above the footer is never edited; `g19_seal_prereg.py --check` exits 0 and the runner refuses any addendum count but the one it implements. History: the orchestrator resolved every open marker in `preregistration_draft.md` on 2026-09-15; each resolution says whether outcomes had been seen, and only baseline outcomes existed (§20; list in `decisions/D01_transfer_signal.md`). Task X re-verified the code consequences and put two readings back to the orchestrator; both were resolved the same day and are implemented: S1(c) "same fitted folds" — B3x and B3i re-fitted on exactly the batched V5-PAIR folds the candidate is fitted on, HEAVIER needs no fit (`gen19ct.evaluation.transfer.S1C_REGISTERED_FOLD_READING`, `check_s1c_fold_designs`; re-fit helper `gen19ct/models/s1c_yardsticks.py`, unit-tested on a synthetic mini-corpus only) with the seed combination at confirmation (`transfer.s1c_seed_combination`: seed-mean Δ_Y, system-cluster bootstrap of the seed mean with the same resampled systems in every seed, 5 of 5 seeds) — and the wildcard-copy sensitivity registered for V5-P and V5-PAIR as well as V1 and V5 (`transfer.REGISTERED_SENSITIVITIES`; the strict filter stays exploratory). Under the V1 outer-fold unit the pooled remainder fold is one publication-group cluster (`metrics.design_unit_clusters`, `tests/test_resolutions.py`). The verification of that implementation (same day) tightened the S1(c) guards — every fitted arm and the pair set must name `<stem>@<design_hash>` (a bare stem is the same for every seed), `s1c_half` refuses pairs of another half, `s1c_paired_verdict` refuses rather than FAILs a selection result not scored on the selection half on seed 104729 — made marked consistency corrections in `preregistration_draft.md` (§3.1 and §7 item 7: the S1(c) fold exceptions; §3.2 and §8: the V1 unit wording), and put one reading back to the orchestrator, resolved the same day: §9 S2(a) now uses the paired rule of S1(c) (`decisions/D01_transfer_signal.md`). `scripts/g19_seal_prereg.py --check` exits 0: sealed 2026-09-15 (footer 135842499a86…5641) with POST-HOC addendum 1 below the footer. |
| pre-seal Phase C | closed-form baselines only (`scripts/g19_run_preseal.py`, `decisions/D01_transfer_signal.md`); second verification pass applied 2026-09-15 (task X: one inner-design implementation shared by the fold builder and the conformal wrapper, known-state calibration rows, section 13 thresholds written by the fold builder, `gen19ct/evaluation/support.py`, the wildcard-copy leakage audit, exploratory V1-unit and copy sensitivities, a seal gate on unticked boxes); `tables/preseal_pair_summary.csv` rebuilt the same day with `--tables-only` so that its DIR5-alternative rows are exploratory, role side, arm `HEAVIER_alt_half` / `HEAVIER_alt_lnln` (no refit; every other output re-rendered byte-identical, `manifests/g19_run_preseal.json` → `tables_only_rebuild`); after task X the same day every output was re-aggregated from the stored predictions under the resolutions (`--skip-compute`, no refit; `manifests/g19_run_preseal.json` → `aggregation_rebuild`): registered V1 outer-fold unit and wildcard-copy status in the tables, resolved `support_score` for every support job (`scripts/g19_update_support_preseal.py`, now also called by the pre-seal script), `evaluation/preseal/difficulty_resolved.json`, and `difficulty.json` byte-identical to the digest §9 quotes; after the last two resolutions (S1(c) "same fitted folds", wildcard copies on V5-P / V5-PAIR) were implemented and verified, `--skip-compute` ran again (no refit; of the 58 recorded outputs only `difficulty_resolved.json` changed: `resolved_orchestrator_decisions`, the V5-P / V5-PAIR wildcard-copy status and the two reading strings; the manifest's `code_sha256` is current) and a `--tables-only` pass then re-rendered every output byte-identical (`tables_only_rebuild.tables_rewritten` = `[]`) |
| C — baselines B0–B8 on V1/V2/V5 | not started as a scored run: the closed-form pre-seal baselines exist; the learned arms (B5, B6/B6r0, B8, FLAT_CAT, M0–M2) are **built and planned, not run** |
| discovery runner | built: `scripts/g19_run_discovery.py` (plan, seal gate, resumable per-fold records, cost model) and `scripts/g19_score_discovery.py` (R19, stop rule, ladder, S1 components) implement the sealed §7 plan **as amended by POST-HOC addendum 1**. The seal gate pins the footer digest, the addendum count AND the addendum text (`discovery.REGISTERED_ADDENDA_SHA256`), and every fold record's resume digest carries the addendum digest and the resolved inner design. `--dry-run` writes the job plan to `evaluation/discovery/benchmark/plan_addendum1.txt`. The addendum-1 cost estimate (`--benchmark-addendum1`, fit-only timings on the TRAINING rows of one V5-primary batched fold, no test row predicted) is **measured**: `cost_estimate_addendum1.{json,md}` price the plan at **90.2 CPU-h, 45.1 h wall clock on 2 workers against the 60 h budget -> FITS** (sealed plan: 1535.7 CPU-h / 767.9 h wall); cumulative wall-clock checkpoints by stage: 00_safeguard 0.2 h; 01_B6 0.5 h; 03_B5_FLAT_CAT_B8 15.4 h; 04_M1 17.9 h; 05_M2 19.0 h; 07_s1c_v5pair 27.6 h; 08_refit_sensitivities 45.1 h. Conditional runs (V5-P per heavy arm, further V5-PAIR candidates, the B6 re-colouring) are priced separately and are not in the total. The estimate is ideal wall clock (compute / workers) from single-fold timings |
| D–H — models, uncertainty, process | not started |

**No predictive model has been trained or fitted.** `log_D` is read only descriptively: to compare
values of duplicate and near-duplicate records, to test whether source metadata leaks the target
(`g19_audit_leakage.py`), and for the variance decomposition of feasibility question 13 (labelled
DESCRIPTIVE in `g19_feasibility.py`).

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
                 s1c_yardsticks.py (B3x / B3i re-fitted on batched V5-PAIR folds for the section 9 S1(c) contrast)
  evaluation/    metrics.py, transfer.py, calibration.py, pairs.py, support.py (section 13 thresholds,
                 support_score components, domain status)
  losses/ process/   empty packages reserved for Phases D-H
scripts/       g19_build_extractants.py  g19_build_metals.py  g19_audit_leakage.py
               g19_audit_corpus.py  g19_feasibility.py       (Phase A/B steps)
               g19_run_phaseAB.py      runs the five steps in order; --determinism
               g19_seal_prereg.py      --check / --seal / --commit-seeds / --verify-seeds
               g19_build_folds.py      registered outer folds, per-fold section 13 thresholds, wildcard-copy audit,
                                       nested-certificate safeguard sample (--merge-index: partial rebuild)
               g19_run_preseal.py      the pre-seal closed-form baseline run (B0-B4, B7, FLAT, HEAVIER, intervals)
               g19_update_support_preseal.py   section 13 s4 update of support_score (also called by g19_run_preseal)
tests/         conftest.py, test_load, test_normalize, test_provenance, test_leakage, test_metals,
               test_ligands, test_support_graph, test_seal_prereg, test_manifest,
               test_registered_folds (marker: slow), test_folds, test_baselines, test_metrics,
               test_transfer_stats, test_inner_designs, test_support, test_resolutions
descriptors/   metals.csv + metals_sources.md, extractant_components.csv, extractant_systems.csv,
               family_rules.json
data_audit/    machine-readable Phase A/B tables (CSV/JSON), one family per script:
               counts / sparsity / matrix_* / *_coverage / columns / dataset_hashes  (g19_audit_corpus)
               leakage_* / metadata_availability                                    (g19_audit_leakage)
               metal_alias_audit / metal_descriptor_coverage                         (g19_build_metals)
               ligand_alias_collisions / family_coverage / named_extractant_presence (g19_build_extractants)
               feasibility.json / feasibility_* (incl. feasibility_halves.csv)       (g19_feasibility)
figures/       F01_observation_matrix, F02_density_by_metal, F03_density_by_family,
               F04_lanthanide_coverage, F05_actinide_coverage, F06_condition_coverage (PNG, dpi 130)
manifests/     <script>.json (deterministic), run_info/<script>.json (volatile),
               phaseAB_determinism.json (the two-run byte comparison),
               g19_build_folds_incremental.json (partial fold builds; merge.full_build_link supersedes the full
               build's INDEX.json / wildcard_copy_crossings.csv digests)
folds/         registered outer folds, INDEX.json                                     (g19_build_folds)
evaluation/preseal/  difficulty.json (as-run, quoted by §9), difficulty_resolved.json, support_status.json,
               support/<job>__support_score.parquet; tables/ (preseal_*)  figures F07, F08  (g19_run_preseal)
models/ process/   empty output directories for later phases
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

**Discovery** (the sealed §7 plan as amended by POST-HOC addendum 1; **nothing has been fitted or
scored yet**). Every command first calls `g19_seal_prereg.py --check`, pins the sealed digest
`135842499a86…` and refuses unless exactly one POST-HOC addendum is below the footer
(`--expect-addenda N` overrides deliberately); the digest, the addendum count and the SHA-256 of the
addendum text go into every fold record and manifest.

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
- Step manifests record git HEAD but not the digests of the code that produced them, and while
  the gen19 tree is uncommitted HEAD does not pin that code.
- Curation items opened by the verification: the five name–structure-conflict structures (Br-Cosan,
  TPDGA malonamide, TDGA, TBADIPIC, NDDIPIC), the stereo-free vs cis/trans Me2-TODGA identity, and the
  censoring heuristic (exact-decade floors/ceilings) that stands in for a missing detection-limit flag.
- Policy decisions still open: the Sr(III) rows the archive marks implausible, acid values that
  look like converted pH, the DMDOHEMA name attached to two structures, empty polarizability.
