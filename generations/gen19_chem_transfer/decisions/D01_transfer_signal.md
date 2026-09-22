# D01 — Is there transferable signal before advanced architecture?

*Pre-seal Phase C (pre-registration §0 order of work item 1; brief §28 Phase C, §29). Written 2026-09-15 from
`scripts/g19_run_preseal.py` outputs at git HEAD 40f6a75 (`manifests/g19_run_preseal.json` → `git_head`), and revised
the same day after the second verification pass (task X): the conformal inner designs now take their units from the fold
builder's functions, calibration rows are the scored population, the §13 thresholds are written by the fold builder,
contrast rows are labelled exploratory, and the V1-unit and wildcard-copy sensitivities are printed. The run was repeated
after those corrections; every number below is from the repeated run. Revised again the same day after the orchestrator
resolved every open marker of the draft pre-registration (list below); the V1 unit and the S1(c) text follow those
resolutions, and `tables/preseal_pair_summary.csv` was rebuilt from the stored predictions with
`g19_run_preseal.py --tables-only` to relabel its exploratory HEAVIER rows (no fold was refitted; every other output
re-rendered byte-identical). Revised a third time the same day after task X verified the code consequences of those
resolutions: `g19_run_preseal.py --skip-compute` re-aggregated every output from the stored predictions under the
resolutions (no fold refitted; `manifests/g19_run_preseal.json` → `aggregation_rebuild`), so the tables now carry the
registered V1 outer-fold unit (column `unit_reading`), the registered wildcard-copy status and the resolved support
status. `evaluation/preseal/difficulty.json`, the as-run record whose SHA-256 §9 quotes, re-rendered byte-identical
(`manifests/g19_run_preseal.json` → `difficulty_json.byte_identical_to_section9`); the resolved numbers are in
`evaluation/preseal/difficulty_resolved.json`. After the last two resolutions (S1(c) "same fitted folds"; wildcard
copies on V5-P / V5-PAIR) were implemented and verified, `--skip-compute` ran once more (no fold refitted): of the 58
recorded outputs only `difficulty_resolved.json` changed (`resolved_orchestrator_decisions`, the V5-P / V5-PAIR
wildcard-copy status, the S1(c) and wildcard reading strings), and a `--tables-only` pass re-rendered every output
byte-identical (`manifests/g19_run_preseal.json` → `tables_only_rebuild.tables_rewritten` = `[]`); no number below
changed. Only the closed-form baselines B0–B4 (with B3x,
B3i, B3l, B4x, B4l), B7, the pair yardsticks FLAT and HEAVIER, and split-conformal intervals on them were fitted or
scored (`manifests/g19_run_preseal.json` → `arms`, `pair_yardsticks`, `learned_arms_fitted_or_scored` = `[]`). Nothing
ran on V6; no `V6_TARGET_ROWS` row was scored (`evaluation/preseal/run_checks.json` → `v6_target_rows_scored` = 0,
`forbidden_arm_names_found` = `{}`). Every contrast below is **DESCRIPTIVE (baseline difficulty)**: none is an R19
claim, none was pre-specified as a claim, and none informs a threshold (`tables/preseal_contrasts.csv`: every row has
`status` = exploratory and `decides` = False). Paths are relative to `generations/gen19_chem_transfer/`. Unless marked,
numbers are the **selection half** (§9), macro MAE in log D over the design's averaging unit; confirmation-half values
are printed only where labelled, and they inform nothing.*

## Question

Before any factorised or learned model: do cheap within-system borrowing rules transfer chemistry to data that is
hidden — a missing metal × extractant cell (V5), an unseen metal (V2), an unseen publication (V1) — better than the
global, metal and extractant means? And how much error does the best such rule leave above the replicate noise, i.e.
how much headroom a factorised model could remove?

## Evidence

| file | contents |
|---|---|
| `evaluation/preseal/difficulty.json` | the as-run record (SHA-256 quoted in §9): §9 difficulty numbers, comparator choices, V1/V2/V0 comparators, fallback shares, `readings`, `V1.unit_sensitivities_exploratory`, `V5_sensitivity_prediction_identity`, `wildcard_copy_sensitivity_exploratory` (its V1 and wildcard labels predate the resolutions) |
| `evaluation/preseal/difficulty_resolved.json` | the same pre-seal numbers under the resolutions: `V1.selection` (outer-fold unit), `V1.selection_exploratory_publication_group`, wildcard-copy filters with their status, `sensitivity_status`, `support`, `resolved_orchestrator_decisions` |
| `tables/preseal_summary.csv` (+ `.md`) | every design / variant / half / arm / stratum / scoring filter / unit reading, tidy with regime columns |
| `tables/preseal_contrasts.csv` | paired cluster bootstrap (10,000 resamples, seed 19) of baseline-vs-baseline contrasts, family `DESCRIPTIVE_baseline_difficulty`, columns `unit_variant`, `unit_reading` |
| `tables/preseal_pair_summary.csv`, `evaluation/preseal/pairs/V5PAIR__primary__cell_pairs.csv` | V5-PAIR logSF and direction |
| `tables/preseal_fallback_counts.csv` | fallback level of every prediction |
| `evaluation/preseal/guard_log.csv`, `inner_guard_verification.json` | outer and inner `fold_isolation_check` records |
| `evaluation/preseal/support_status.json`, `domain_status_counts.csv`, `support/<job>__support_score.parquet`, `tables/preseal_coverage_by_domain_status.csv` | §13 support features, the registered `support_score` and labels (`gen19ct/evaluation/support.py`) |
| `folds/INDEX.json` → `leakage_sensitivity_wildcard_copies`, `nested_certificate_safeguard_sample`; `folds/wildcard_copy_crossings.csv` | value-matched copies the near-duplicate key cannot see (DATA_AUDIT.md §13b); the §2 safeguard draw |
| `manifests/g19_run_preseal.json`, `g19_update_support_preseal.json`, `g19_build_folds_incremental.json` | the run records: `aggregation_rebuild`, `support_update`, `merge.full_build_link` |
| `figures/F07_preseal_pred_vs_measured.png`, `figures/F08_preseal_error_vs_support.png` | V5 predictions vs measurements; comparator error vs support |

## Metrics

**V5-primary, exact leave-one-cell-out** (105 cells in 18 systems; `difficulty.json` → `V5.n_cells_selection`,
`V5.n_systems_selection`; values from `V5.macro_mae_selection`)

| B0 | B1 | B2 | B3x | B3i | B3l | B4x | B7 |
|---|---|---|---|---|---|---|---|
| 1.526 | 1.473 | 1.130 | 0.827 | 0.767 | 1.188 | 1.400 | 0.810 |

- **V5 lookup comparator: B3i** (0.767 < B3x 0.827; `V5.lookup_comparator.choice`), fixed here, before any learned
  model is scored. The B3x / B3i pool is the system's training metal states measured with the query's acid anion
  (`readings.B3x_B3i_anion_pool`; the reading implemented before any score, now written into pre-registration §5).
- **L5 = 0.7670, C5 = 1.5262, N0 = 0.2386, L5 − N0 = 0.5284, δ5 = max(0.05, 0.20 × (L5 − N0)) = 0.1057**
  (`V5.L5`, `V5.C5`, `V5.N0`, `V5.L5_minus_N0`, `V5.delta5`). A5 is not computed pre-seal (`V5.A5`).
- Lookup oracle, per-cell minimum over B3x/B3i/B3l/B4x, labelled optimistic: 0.627 (`V5.lookup_oracle_macro_mae_selection`).
- Confirmation half, descriptive: B0 1.120, B3x 0.912, B3i 0.860, B7 0.826 (`V5.macro_mae_confirmation_descriptive`).

Contrasts on V5, **selection half** (Δ = comparator − candidate, > 0 favours the candidate; `tables/preseal_contrasts.csv`,
job `V5__primary`, `unit_variant` registered_unit; 95 % percentile interval under the system cluster (18 clusters), then
under the publication-group cluster (27)):

| candidate vs comparator | Δ | system cluster | publication-group cluster |
|---|---|---|---|
| B3i vs B0 | 0.759 | [0.339, 1.252] | [0.319, 1.231] |
| B3i vs B1 | 0.706 | [0.247, 1.301] | [0.219, 1.284] |
| B3i vs B2 | 0.363 | [0.198, 0.502] | [0.182, 0.532] |
| B3x vs B0 | 0.699 | [0.291, 1.188] | [0.270, 1.157] |
| B3x vs B2 | 0.303 | [0.161, 0.429] | [0.134, 0.457] |
| B3i vs B3x | 0.060 | [0.021, 0.088] | [0.028, 0.093] |
| B7 vs B3x | 0.017 | [−0.062, 0.103] | [−0.072, 0.092] |
| B3l vs B3x | −0.361 | [−0.649, −0.101] | [−0.670, −0.083] |
| B4x vs B3x | −0.573 | [−0.850, −0.294] | [−0.850, −0.300] |

The same contrasts in the **confirmation half** (descriptive; 19 system clusters, 20 publication-group clusters; same
file, half `confirmation`). Several intervals include 0 here, so the V5 statement below is a selection-half statement:

| candidate vs comparator | Δ | system cluster | publication-group cluster |
|---|---|---|---|
| B3i vs B0 | 0.260 | [0.050, 0.495] | [−0.014, 0.582] |
| B3i vs B1 | 0.197 | [−0.016, 0.422] | [−0.133, 0.507] |
| B3i vs B2 | 0.121 | [−0.010, 0.245] | [−0.058, 0.324] |
| B3x vs B0 | 0.208 | [0.012, 0.444] | [−0.052, 0.522] |
| B3i vs B3x | 0.051 | [0.026, 0.075] | [0.024, 0.082] |
| B7 vs B3x | 0.085 | [−0.048, 0.203] | [−0.101, 0.220] |

BCa intervals, p, `mde_80` and leave-one-cluster-out minima are in the same rows.

**V5 registered sensitivities** (macro MAE; `tables/preseal_summary.csv`, metric `mae`, aggregation `unit_macro`,
half `selection`, stratum `all` unless stated). "Identical" = the row-level B3x and B3i predictions equal those of
V5-primary on every (fold, row) (`difficulty.json` → `V5_sensitivity_prediction_identity.jobs.<job>.arms_with_identical_predictions`):

| sensitivity (job / filter / stratum) | cells | B0 | B2 | B3x | B3i | B3x/B3i identical to primary |
|---|---|---|---|---|---|---|
| loose (`V5__loose`) | 165 | 1.414 | 1.130 | 0.882 | 0.841 | no |
| strict (`V5__strict`) | 41 | 1.391 | 1.161 | 0.819 | 0.729 | no |
| HNO3-only cells (`V5__hno3_only`) | 99 | 1.554 | 1.162 | 0.870 | 0.804 | no |
| cell-only hiding (`V5__cell_only`) | 105 | 1.526 | 1.130 | 0.827 | 0.767 | **yes** |
| parent-structure hiding (`V5__parent_structure`) | 105 | 1.526 | 1.130 | 0.827 | 0.767 | **yes** |
| Sr(III)-dropped training (`V5__sr_iii_dropped_training`) | 105 | 1.526 | 1.130 | 0.827 | 0.767 | **yes** |
| censoring candidates excluded (`V5__primary`, filter `censoring_candidates_excluded_scoring`) | 105 | 1.502 | 1.115 | 0.832 | 0.763 | — |
| acid-grid rows excluded (`V5__primary`, filter `acid_grid_rows_excluded_scoring`) | 105 | 1.528 | 1.132 | 0.823 | 0.764 | — |
| non-DGA stratum (`V5__primary`, stratum `dga_stratum=non_DGA`) | 46 | 1.670 | 1.137 | 0.908 | 0.849 | — |
| V5-P (`V5P__base`; its comparator B4x = 1.502) | 56 | 1.7045 | 1.209 | 1.502 | 1.461 | — |
| wildcard copies excluded (`V5__primary`, filter `wildcard_copies_excluded_scoring`; registered by the §2 resolution: `status` registered in the table and in `difficulty_resolved.json` → `sensitivity_status.V5`; `difficulty.json` files it under the older exploratory key) | 105 | 1.506 | 1.116 | 0.827 | 0.760 | — |
| *exploratory:* strict wildcard copies excluded (filter `wildcard_copies_strict_excluded_scoring`) | 105 | 1.523 | 1.127 | 0.825 | 0.764 | — |

- The ordering B3i < B3x < B0 holds in every row. But three rows are not independent evidence for B3x and B3i: under
  cell-only and parent-structure hiding the lookups read only the query's own system, whose rows are hidden identically
  in both designs, so identity is by construction (INFERRED from the §5 definitions, confirmed by the identity check);
  under Sr(III)-dropped training the predictions are identical empirically. The ordering therefore rests on seven
  non-identical registered rows.
- In V5-P the lookups do **not** beat the extractant mean: B2 1.209 < B3i 1.461 < B3x 1.502.
- The wildcard-copy filters drop 69 selection-half rows (any partner) or 10 (strict partners) (`difficulty.json` →
  `wildcard_copy_sensitivity_exploratory.jobs.V5__primary`); B3i moves by less than 0.01.

**V5-PAIR** (309 cell pairs; `difficulty.json` → `V5_PAIR`): SF5_FLAT = 0.7986 (`SF5_FLAT`); the B3i-derived logSF
MAE is 0.385 (`chosen_lookup_derived_logsf_mae`). **DIR5 = 0.9498** (`DIR5`), set by HEAVIER on 242 Ln(III)–Ln(III)
cell pairs (6,166 pairs with |observed logSF| ≥ 0.3; `DIR5_components.HEAVIER`); B3x-derived 0.854 and B3i-derived
0.888 on 280 cell pairs (`DIR5_components.B3x`, `.B3i`). DIR5 is descriptive: since the S1(c) redefinition
(pre-registration §9) the direction criterion is a paired contrast on identical pairs, and DIR5 enters no rule.
Alternatives printed beside, not chosen: counting undefined HEAVIER pairs as ½ gives 0.889
(`DIR5_alternative_undefined_counts_half`); restricting all three yardsticks to Ln–Ln pairs gives 0.950, unchanged
(`DIR5_alternative_ln_ln_pairs_only`). In `tables/preseal_pair_summary.csv` those alternative rows (`yardstick_reading`
HEAVIER_undefined_counts_half, and every `pair_subset` Ln-Ln_only row) are `status` exploratory and `role` side. Their
HEAVIER rows carry arm `HEAVIER_alt_half` (undefined counted ½) or `HEAVIER_alt_lnln` (Ln–Ln subset), never HEAVIER;
the arm-derived and FLAT rows keep their arm names. In the confirmation half HEAVIER direction
accuracy is 0.561 on 116 cell pairs (same table, half `confirmation`, arm HEAVIER, `pair_subset` all_pairs, stratum
`all`, unit macro).

**V2, element-level leave-metal-out** (`difficulty.json` → `V2`): comparator **B3i** (`comparator_choice`). Focus-7
lanthanides in the selection half (Ce, Pr, Nd, Gd): B3i 0.603 vs B0 1.101 (`L2_focus7_selection`,
`C2_B0_focus7_selection`); all 12 selection-half states: B3i 0.643, B3x 0.701, B2 0.976, B0 = B1 1.132
(`macro_mae_by_summary.all.S`). Contrasts over 12 metal-state clusters (job `V2__element`): B3i vs B0 Δ 0.489
[0.240, 0.729]; B3i vs B2 Δ 0.333 [0.203, 0.457]; B3i vs B3x Δ 0.057 [0.024, 0.094]; B7 vs B3x Δ −0.064
[−0.142, −0.007].

**V1, exact leave-publication-group-out.** The registered V1 scoring unit is the **outer fold**, with the pooled
remainder fold as **one** unit (pre-registration §3.2, "Resolved (orchestrator, 2026-09-15 …)"). Since task X the tables
score it (`difficulty_resolved.json` → `V1.selection`, 39 selection-half units): B3 (≡ B4, checked: `difficulty.json` →
`V1.B4_equals_B3` true) 1.033, B0 1.142, B2 1.101, B3x 1.028, B3i 1.024, B7 1.105; confirmation half (descriptive,
`V1.confirmation_descriptive`, 44 units): B3 1.090, B0 1.200. The compute run predates the resolution, so the as-run
record scored each publication group as the headline (`difficulty.json` → `V1.macro_mae_selection`, 65 units: B3 1.262,
B0 1.164, B2 1.096, B3x 1.102, B3i 1.098, B7 1.129) and printed the registered reading as its
`V1.unit_sensitivities_exploratory.remainder_fold_as_one_unit`; the registered values equal that sensitivity exactly
(`difficulty_resolved.json` → `V1.registered_unit_equals_asrun_remainder_fold_as_one_unit_selection` true). 21 of the
104 folds have no scorable row (X(?)-only publications) and were not fitted (`evaluation/preseal/guard_log.csv`, job
`V1__copy`, status `no_scored_rows_not_fitted`: 13 selection, 8 confirmation).

The V1 result depends on how the pooled remainder fold is counted. The halves were balanced with that fold as one unit
(`data_audit/feasibility_halves.csv`, unit `REMAINDER`, weight 309, half S), whereas the per-group reading splits it:
**27 of its 65 selection-half units are remainder-fold groups (each below 20 MODEL rows), holding 232 of the 5,246
selection-half scored rows**, and the confirmation half has none
(`difficulty.json` → `V1.unit_sensitivities_exploratory.n_selection_units_from_remainder_fold`,
`n_selection_rows_in_remainder_fold`, `n_selection_rows`, `n_confirmation_units_from_remainder_fold`). The three
readings (`difficulty_resolved.json` → `V1.selection`, `V1.selection_exploratory_publication_group`,
`V1.exploratory_remainder_fold_excluded_scoring_selection`; contrasts from `tables/preseal_contrasts.csv`, job
`V1__copy`, `unit_variant` registered_unit / publication_group_exploratory / remainder_fold_excluded_scoring, clustered
on the scored unit, column `cluster_unit` = publication_group, the remainder fold one cluster under the registered unit):

| V1 unit reading | units | B0 | B2 | B3 | B3x | B3i | B3 vs B0: Δ [95 %] | B3x vs B0: Δ [95 %] |
|---|---|---|---|---|---|---|---|---|
| **outer fold, remainder fold as one unit (registered, §3.2)** | 39 | 1.142 | 1.101 | 1.033 | 1.028 | 1.024 | 0.110 [−0.037, 0.260] | 0.114 [−0.034, 0.268] |
| *exploratory:* each publication group (the as-run headline) | 65 | 1.164 | 1.096 | 1.262 | 1.102 | 1.098 | −0.098 [−0.308, 0.100] | 0.062 [−0.098, 0.210] |
| *exploratory:* remainder fold not scored | 38 | 1.146 | 1.105 | 1.026 | 1.027 | 1.023 | 0.120 [−0.034, 0.272] | 0.119 [−0.036, 0.276] |

The sign of B3 − B0 flips between the per-group reading and the other two; every interval includes 0 under all three.
The wildcard-copy filter (registered for V1 by the §2 resolution) drops 11 selection-half V1 rows (0 strict) and moves
no macro value by more than 0.001 under either unit (`difficulty.json` → `wildcard_copy_sensitivity_exploratory.jobs.V1__copy`,
per group; `difficulty_resolved.json` → `V1.wildcard_copies_excluded_scoring_selection`, outer fold).

**V0 (diagnostic only)**, mean over 5 discovery seeds (`difficulty.json` → `V0_diagnostic.macro_mae_mean_of_seeds`):
B3 0.601, B7 0.530, B0 1.175. Not used for any statement below.

**Fallback coverage** (selection half; `difficulty.json` → `fallback_share_selection`): on V5, B3i returns the radius
interpolation for 0.733 of rows and the unbracketed B3x value for 0.267; B7 uses its mass-action unit for 0.998; B4x
falls back to B2 for 0.077. On V2, B3x uses a borrowed metal for 0.967 of rows (0.033 B2:family). On V1, B3 uses the
exact pair for 0.577 of rows, B3l for 0.356 and B3x for 0.067.

**Intervals** (split-conformal; inner units from the fold builder's functions, seed 104729; calibration rows = the
scored population, no X(?) row; `tables/preseal_summary.csv`, half `selection`, metrics `coverage_50/80/95`, unit
macro). They replace the first run's intervals, which used a second inner-design implementation and, for V1 and V0,
X(?) calibration rows. They rest on one inner-fold seed; pre-registration §15 now registers the mean over the 5
discovery seeds for the deterministic arms' intervals in discovery, so these single-seed values are descriptive. The V1
rows use the registered outer-fold unit (column `unit_reading` registered); the per-group values the first version of this
record printed are the exploratory rows of the same table:

| design | arm | 50 % | 80 % | 95 % |
|---|---|---|---|---|
| V5-primary | B3i | 0.544 | 0.835 | 0.953 |
| V5-primary | B3x | 0.542 | 0.830 | 0.952 |
| V5-primary | B0 | 0.420 | 0.728 | 0.886 |
| V1 copy (outer fold) | B3 | 0.616 | 0.840 | 0.966 |
| V1 copy (outer fold) | B0 | 0.551 | 0.847 | 0.971 |
| *exploratory:* V1 copy (per publication group) | B3 | 0.543 | 0.784 | 0.945 |
| *exploratory:* V1 copy (per publication group) | B0 | 0.553 | 0.843 | 0.978 |
| V2 element | B3i | 0.465 | 0.763 | 0.936 |
| V2 element | B0 | 0.585 | 0.844 | 0.965 |

**Support** (`figures/F08_preseal_error_vs_support.png`; `difficulty.json` → `F08_descriptive_spearman`): across the
105 cells, Spearman ρ of B3i cell MAE with the nearest-radius distance 0.365, with the number of neighbouring metals
0.050, with the mean condition distance 0.357 — descriptive only, the §8 reliability floor was not assessed. §13 domain
status on V5 selection rows: 4,084 CROSS_METAL_LIGAND_TRANSFER rows (105 cells) and 17 UNSUPPORTED rows (4 cells)
(`evaluation/preseal/domain_status_counts.csv`). The per-fold thresholds were recomputed on every fitted fold (V1: 83 of
104; V5-primary: 210 of 224; V2 23 of 23; V0 25 of 25; `guard_log.csv`, status `fitted`) and equal the `support_tau` the
fold builder wrote beside each fold hash (`readings.support_tau`; the run aborts otherwise). `support_score` is computed
for all four support jobs under the s4 reading §13 now registers (own system counted when it has training rows of the
metal state, undefined `d_desc` skipped; `evaluation/preseal/support_status.json` → `jobs.<job>.support_score`, written by
`scripts/g19_update_support_preseal.py`, `manifests/g19_update_support_preseal.json`). The recomputed s4 equals the
stored pre-seal s4 on every row and `support_score` equals the stored `support_score_candidate` for every job
(`jobs.<job>.s4_recomputed_equals_preseal_s4`, `support_score_equals_preseal_candidate`); on V1 and V0 the own system
counts on 5,813 of 10,575 and 48,448 of 52,875 row entries (`n_rows_query_system_counts_in_s4`, `n_scored_row_entries`).
Since task X the pre-seal script calls the same update (`apply_support_update`), so a `--skip-compute` or full rerun
reproduces these files instead of writing the as-run "not computed" labels (`manifests/g19_run_preseal.json` →
`support_update.as_run_status_equals_update_manifest_pre_update_sha256`).

**Guards**: every fitted outer split passed `fold_isolation_check` at its registered level (210 checks for V5-primary,
840 for V5-P, 472 for V5-PAIR, 83 for V1; `guard_log.csv`, column `n_outer_guard_checks`). V5 inner calibration splits
used the wrapper's `nested_certificate` mode (209 certificate checks; `inner_guard_verification.json` →
`inner_guard_calls_by_job.V5__primary`); on 6 outer folds it was cross-checked against `every_split` (90 checks each,
identical residuals and quantiles; `folds[*]`). Two task X corrections postdate those counts. (i) The V1-level check of
`guard_v5p` (and of `guard_v1`) compared `g19_publication_id`, so a training row of a scored row's copy group carrying
another id was invisible to it; it now reads `group_cross_publication_copy` (`gen19ct.folds.io.publication_group_frame`,
test `test_publication_group_guard_sees_a_copy_group_row_with_another_publication_id`). The fold files were built
correct by construction and are unchanged; the hardened guard passed on the rebuilt batched V5-P folds
(`folds/INDEX.json` → `designs.V5P__base__batched.guard`, 440 checks, `all_ok` true) and, in the task X session (not
stored as a file), on every unbatched V5-P fold and every registered V1 fold. (ii) The §2 safeguard sample now covers
every fold file fitted with the V5 inner design — the only inner design whose splits carry a certificate — and labels
the V1 and V2 draws vacuous (`folds/INDEX.json` → `nested_certificate_safeguard_sample.designs.<stem>`
`.certificate_in_inner_splits`, `.vacuous_under_current_code`).

## Verdict (null / supported / ambiguous)

- **V5 (missing cell): supported in the selection half** — borrowing a metal inside the same extractant system beats the
  global, metal and extractant means, with both registered cluster intervals excluding 0, and B3i < B3x < B0 holds in
  every registered V5 sensitivity (seven of them non-identical for the lookups). B3i beats B3x by 0.060, interval
  excluding 0. Limits: in the confirmation half (descriptive) B3i vs B0 excludes 0 only under the system cluster, and
  B3i vs B1 and vs B2 include 0 under both clusters; in V5-P the extractant mean B2 beats both lookups.
- **V2 (unseen metal): supported in the selection half** — the same pattern over 12 metal-state clusters.
- **V1 (unseen publication): ambiguous.** Under the registered V1 unit (outer fold, remainder fold as one unit;
  pre-registration §3.2) the comparator B3 scores 1.033 against B0 1.142: B3 vs B0 Δ 0.110 [−0.037, 0.260] over 39
  selection-half units, interval including 0. Its TOST verdict is PASS, i.e. non-inferior at ε = 0.05 (90 % interval
  [−0.015, 0.239]), not NO_DIFFERENCE (`tables/preseal_contrasts.csv`, job `V1__copy`, `unit_variant`
  registered_unit, `tost_verdict_eps0.05`, `tost_low_90`, `tost_high_90`). The other two unit readings stay
  exploratory. No lookup separates from B0 or B2 under any of the three (every interval includes 0), and the sign of
  B3 − B0 depends on the reading: negative per publication group, where TOST is UNDECIDED. This is not a null.
- **Headroom**: the V5 comparator still sits 0.528 log D above the replicate noise floor; the registered S1(a) margin
  for a factorised model is δ5 = 0.106.

Baseline evidence only: this says that cross-metal information inside a system is transferable, not that a learned or
factorised model will add to it.

## Decision

- Record the §9 difficulty numbers as written in `evaluation/preseal/difficulty.json` (now quoted in pre-registration
  §9 with the file's SHA-256): L5 = 0.7670, C5 = 1.5262, N0 = 0.2386, δ5 = 0.1057, SF5_FLAT = 0.7986. DIR5 = 0.9498 is
  kept in the §9 table as a descriptive difficulty number only: S1(c) is now the paired rule of §9 and DIR5 enters no
  rule.
- The V5 / V5-PAIR / V6 log D comparator is **B3i**; the V2 comparator is **B3i**. Both are fixed before any learned
  model is scored.
- The pre-seal run gives no reason to change a registered design. The open items it raised were resolved by the
  orchestrator in the draft on 2026-09-15 (list below). `g19_seal_prereg.py --check` still exits 1 on the draft banner,
  2 remaining `TO BE FINALISED` markers (the banner and the §15 seed digest), 4 unticked §20 boxes and the missing
  seed commitment.

**Resolved before sealing (orchestrator, 2026-09-15)** (each resolution is in `preregistration_draft.md` and states
whether outcomes had been seen; only baseline outcomes existed):

1. **S1(c) / DIR5** → §9 S1(c), "S1(c) redefined (orchestrator, 2026-09-15; results seen: yes …)". The direction
   criterion is a paired contrast on identical pairs of the half being judged: Δ_Y = M2 − yardstick Y for Y ∈
   {HEAVIER, B3x-derived, B3i-derived} on the pairs where Y is defined. At confirmation min_Y Δ_Y ≥ γ5 = 0.05 with every
   Δ_Y's system-cluster percentile 95 % interval excluding 0. As a counterweight, on the selection half (seed 104729)
   min_Y Δ_Y ≥ −0.02. The logSF MAE part is paired too. `DIR5` is descriptive and enters no rule. Recorded as a change
   made with results seen, not as a clarification. The DIR5 alternatives in `tables/preseal_pair_summary.csv` are now
   `status` exploratory, `role` side, with arm `HEAVIER_alt_half` (undefined pairs counted ½) or `HEAVIER_alt_lnln`
   (HEAVIER, Ln–Ln-only subset). "Same fitted folds" (resolved the same day, not score-driven): B3x and B3i are
   re-fitted on exactly the batched V5-PAIR folds the candidate is fitted on (seed 104729 in discovery, each withheld
   seed at confirmation); HEAVIER needs no fit. At confirmation each Δ_Y is the mean over the 5 withheld seeds, with a
   system-cluster bootstrap of that mean (the same resampled systems in every seed) and Δ_Y > 0 in 5 of 5 seeds.
2. **V1 scoring unit** → §3.2 Halves, "Resolved (orchestrator, 2026-09-15 …)": the outer fold, the pooled remainder
   fold as one unit; the per-publication-group reading is printed beside it as exploratory; §4 is read accordingly.
3. **X(?) rows** → §2 Rows, "Scored rows": unscored in every design, V0 and V1 included, and out of every calibration
   and inner validation set.
4. **Wildcard copies** → §2 Publication groups, "Copies the key cannot see": `wildcard_copies_excluded_scoring` (any
   partner) is a registered R19 item 6 sensitivity of V1 and V5, and of V5-P and V5-PAIR by the §8 R19 item 6
   resolution (12 crossings on 11 scored entries per V5-P design, 371 on 359 per V5-PAIR design); the strict filter
   stays exploratory; publication groups, V1 folds and halves are not rebuilt.
5. **V5 inner-guard mode** → §2 Leakage guard: `nested_certificate` counts as passing `fold_isolation_check` for inner
   splits of every design, with an `every_split` safeguard on 20 outer folds per design at the start of discovery.
   **Interval seeds** → §15 Discovery seeds: the deterministic arms' conformal inner folds are drawn with each discovery
   seed and averaged (each withheld seed at confirmation). **V0 averaging and cluster unit** → §3.6: the as-run reading
   (publication group, mean over the 5 discovery seeds; F1 interval from a publication-group cluster bootstrap; no
   secondary cluster).
6. **Unmarked rules found by the verifiers**: B3 one-row tie rule → §5 B3 Ties (confirmed as implemented; the
   replicate-mean reading is not run); R19 item 3 → §8 item 3 (p < 0.05 under every registered cluster unit); X(?) rows
   in the power check → §8 Signal-injection power check (dropped from the injected refits of every arm); s4 → §13 s4 (own
   system counts when it has training rows of the metal state; undefined `d_desc` skipped).
7. **Discovery machinery** → §3.1 V5-PAIR "Resolved" (closed-form arms unbatched; heavy arms on batched V5-P / V5-PAIR
   folds, seed 104729, V5 inner design, freezing candidates only; the S1(c) exceptions: B3x / B3i re-fitted on the
   batched V5-PAIR folds M2 is fitted on, and at confirmation batched V5-PAIR folds built with each withheld seed);
   §6 "M-model training settings"; §7 "Compute plan" (nested tuning, ladder decisions on seed 104729, order of
   discovery, stop rule, 60-hour budget and demotion order).
8. **The halves differ strongly** (V5 B0 1.526 vs 1.120; HEAVIER 0.950 vs 0.561). Not a registration item and not
   resolved as one; the §9 S1(c) selection-half counterweight is stated as a response to the low confirmation-half
   HEAVIER value.
9. **Phase H tests** (brief §27 mass-balance invariance, process Monte Carlo reproducibility) are still not built; §14
   carries the reproducibility check. Not a pre-seal registration item.

Still open in §20. The code consequences are implemented and were re-verified by task X (2026-09-15): batched V5-P /
V5-PAIR heavy-arm folds (`manifests/g19_build_folds_incremental.json`); the V1 outer-fold unit and the registered
wildcard-copy status in the scorer (`scripts/g19_run_preseal.py`, `difficulty_resolved.json`); the V0 F1 interval
(`transfer.seed_mean_cluster_bootstrap`); the s4 reading (`support.s4_registered`,
`manifests/g19_update_support_preseal.json`); multi-seed conformal intervals (`ConformalWrapper(seeds=…)`); the paired
S1(c) evaluator, which guards every pair (`transfer.guard_scored_pairs`); the safeguard sample (`folds/INDEX.json`);
X(?) rows dropped in injected refits, with training taken from the fold's hidden rows (`InjectedRun.fold_inputs`).
Task X put two readings back to the orchestrator; both were resolved the same day and are implemented: (a) S1(c)
"same fitted folds" — `transfer.S1C_REGISTERED_FOLD_READING` = `yardsticks_refit_on_candidate_batched_folds`;
`check_s1c_fold_designs` requires every fitted arm and the pair set to name the same `<stem>@<design_hash>` of a batched
V5-PAIR fold file (a bare stem, the same for every seed, is refused); `s1c_half` refuses pairs of another half;
`s1c_seed_combination` combines the 5 withheld seeds; `s1c_paired_verdict` refuses, rather than FAILs, a selection
result not scored on the selection half on seed 104729; the re-fit helper is `gen19ct/models/s1c_yardsticks.py`
(unit-tested on a synthetic mini-corpus only); (b) the wildcard-copy sensitivity registered for V5-P and V5-PAIR as
well (`transfer.REGISTERED_SENSITIVITIES`; 371 crossings on 359 scored row entries in each V5-PAIR design, 12 on 11 in
each V5-P design †; `folds/wildcard_copy_crossings.csv`, column `stem`). The verification of those fixes put §9 S2(a) to the
orchestrator (its pair-level direction margin maxed over yardsticks defined on different pair sets). Resolved the same
day: S2(a) now uses the paired rule of S1(c) — min over HEAVIER, B3x-, B3i- and B8-derived of Δ_Y on identical V6 pairs
where Y is defined, ≥ 0.05, pooled and HNO3 — with no V6 score of any arm in existence (pre-registration §9 S2(a)). The S2
evaluator is built for the confirmation run, not now.

## Next action

*Both steps below are DONE; this section was written before the seal and is kept as the record of what D01 asked for.
What follows them is in `decisions/D02_factorization.md` and the README status table.*

1. ~~`g19_seal_prereg.py --commit-seeds`, replace the banner, tick §20, `--check`, and seal (orchestrator).~~ **Done
   2026-09-15**: sealed, footer digest `135842499a86…5641`, `manifests/prereg_sha256.txt`; two POST-HOC addenda below
   the footer since (1: the compute-driven plan reduction, 2: the ladder budget and the digest registry).
2. ~~Then discovery starts, per the §7 compute plan: B6 / B6r0 (with the batched-vs-exact check), then B5 = M0, FLAT_CAT
   and B8, then the M1–M7 ladder, on the discovery seeds and the selection halves only, judged against B3i on V5 with
   δ5 = 0.106.~~ **Done 2026-09-22**, with two registered departures the run itself recorded: the discovery seeds became
   seed 104729 alone (addendum 1 item 3, so R19 item 4 is NOT_EVALUATED in discovery), and M3–M7 were never fitted —
   the 60 h budget was exhausted at 76.6 h of recorded work, so the §7 item 5 rule demoted them
   (`evaluation/discovery/decisions/decisions.json` → `budget.discovery`), and addendum 2 gives the ladder its own 40 h.
   B6 / B6r0, B5 = M0, FLAT_CAT, B8, M1 and M2 ran; the stop rule is **not** triggered (M2 vs B3i@V5 passes R19 items
   1–3 and 5, Δ = +0.263 ≥ δ5 = 0.10569; B6 vs B3i fails), and because item 4 cannot be evaluated on one seed the full
   R19 verdict of that contrast is UNDECIDED, so nothing is frozen as confirmed.

**What D01's own question needs next** (it asked whether there is transferable signal *before* advanced architecture,
and the discovery run answered the architecture half but not this one):

3. `scripts/g19_run_power.py` — the §8 signal-injection power check. `decisions.json` → `power_check.status` is **not
   implemented**, and it is required before B6 vs B3i, M2 vs B0, M2 vs B3i and M2 vs B6r0 are reported as nulls. Until
   it runs, every FAIL in D01 and D02 is "no detected difference", not "no difference".
4. `scripts/g19_run_ladder.py` (M3 → M7) then `g19_run_h3.py` — H3 is the actinide-transfer question D01 could not
   answer with closed-form baselines, and D03 is written from it.
5. Confirmation (orchestrator, §15) — the ≤ 5 frozen claims on the withheld seeds and the confirmation half, V6 once.
   No pre-seal or discovery number in this file is confirmed; the halves differ strongly (item 8 above), so the
   selection-half values here are the optimistic end of the range.
