# gen6 aggregate — 20260819T105104Z

1 run directory aggregated. An aggregate is only as meaningful as the claim that its inputs are comparable, so that claim is checked first and printed first.

## 1. Provenance compatibility (the gate)

| fingerprint | status | runs stating it | distinct values | absent in |
|---|---|---|---|---|
| dataset_file_sha256 | AGREES | 1 | 1 | — |
| source_table_sha256 | AGREES | 1 | 1 | — |
| feature_registry_sha256 | AGREES | 1 | 1 | — |
| chemistry_cluster_definition | AGREES | 1 | 1 | — |
| cohort_definition | AGREES | 1 | 1 | — |

Single input run, so there is nothing to disagree with. This section records **what** the run was built from, so a later aggregate can be checked against it.

## 2. Runs

| run_id | layer | path | created_utc | validated_ok | has_success_marker | dataset_file_sha256 | feature_registry_sha256 | n_folds | leaderboard_file | leaderboard_rows | bootstrap_file | bootstrap_rows | problems |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gen6_diversity_causal_20260819T102822Z | gen6_diversity | runs/gen6_expA_5seed | 2026-08-19T10:31:16.076625+00:00 | yes | yes | fefbefc6fe993aa9 | 0769624bbbc9345e | 50 | arm_metrics.csv | 40 | contrast_summary.csv | 90 |  |

`problems` lists tolerated absences. A run with no leaderboard contributes provenance only.

## 3. Combined leaderboard

40 rows from 1 run(s); full table in `combined_leaderboard.csv`.

Scored unit: `arm`; ranked by mean `macro_mae` (lower is better), grouped by every facet column present so that an 'overall' row is never averaged with a 'hard chemistry' one. `n_runs` and the sd are part of the result: a ranking whose gaps are smaller than its spread is not a ranking.

| rank | arm | feature_set | macro_mae_mean | macro_mae_sd | macro_mae_min | macro_mae_max | n_rows | n_runs |
|---|---|---|---|---|---|---|---|---|
| 1 | EXPANDED_ROWMATCHED | MC_donors | 1.012 | 0.03864 | 0.9568 | 1.053 | 5 | 1 |
| 2 | EXPANDED | MC_donors | 1.015 | 0.03894 | 0.9628 | 1.063 | 5 | 1 |
| 3 | EXPANDED | MC_lig2d_ext_massaction | 1.047 | 0.01463 | 1.027 | 1.068 | 5 | 1 |
| 4 | EXPANDED_ROWMATCHED | MC_lig2d_ext_massaction | 1.049 | 0.02031 | 1.029 | 1.082 | 5 | 1 |
| 5 | BASE | MC_donors | 1.187 | 0.02342 | 1.159 | 1.214 | 5 | 1 |
| 6 | BASE | MC_lig2d_ext_massaction | 1.21 | 0.01312 | 1.192 | 1.227 | 5 | 1 |
| 7 | EXPANDED_SHUFFLED | MC_lig2d_ext_massaction | 1.219 | 0.03524 | 1.174 | 1.27 | 5 | 1 |
| 8 | EXPANDED_SHUFFLED | MC_donors | 1.252 | 0.01632 | 1.229 | 1.269 | 5 | 1 |

## 4. Bootstrap contrasts

90 rows from 1 run(s); full table in `combined_bootstrap.csv`.

Per contrast: how many runs put the delta above zero, and how many have a CI95 lower bound above zero. Intervals are **not** re-pooled — that would need the replicate draws, which the CSVs do not carry — so this counts runs, it does not combine them.

| comparison | reference | candidate | statistic | endpoint | feature_set | n_runs | n_rows | mean_delta | min_delta | max_delta | n_delta_positive | n_ci95_low_above_zero | n_with_ci | direction_consistent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | offset_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.5359 | 0.5359 | 0.5359 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | offset_mae | nn<0.4 | MC_donors | 1 | 1 | 0.5165 | 0.5165 | 0.5165 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | mae | nn<0.4 | MC_donors | 1 | 1 | 0.5125 | 0.5125 | 0.5125 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.4944 | 0.4944 | 0.4944 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | offset_mae | nn<0.4 | MC_donors | 1 | 1 | 0.4848 | 0.4848 | 0.4848 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | offset_mae | nn<0.4 | MC_donors | 1 | 1 | 0.4721 | 0.4721 | 0.4721 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | mae | nn<0.4 | MC_donors | 1 | 1 | 0.472 | 0.472 | 0.472 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | offset_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.4706 | 0.4706 | 0.4706 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.4681 | 0.4681 | 0.4681 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | offset_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.4677 | 0.4677 | 0.4677 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.463 | 0.463 | 0.463 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | mae | nn<0.4 | MC_donors | 1 | 1 | 0.4611 | 0.4611 | 0.4611 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | offset_mae | nn<0.6 | MC_donors | 1 | 1 | 0.2824 | 0.2824 | 0.2824 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | mae | nn<0.6 | MC_donors | 1 | 1 | 0.2776 | 0.2776 | 0.2776 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | offset_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.2374 | 0.2374 | 0.2374 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | mae | all | MC_donors | 1 | 1 | 0.2368 | 0.2368 | 0.2368 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | offset_mae | nn<0.6 | MC_donors | 1 | 1 | 0.2351 | 0.2351 | 0.2351 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | offset_mae | all | MC_donors | 1 | 1 | 0.2308 | 0.2308 | 0.2308 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | offset_mae | nn<0.6 | MC_donors | 1 | 1 | 0.2305 | 0.2305 | 0.2305 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | mae | nn<0.6 | MC_donors | 1 | 1 | 0.2283 | 0.2283 | 0.2283 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | mae | nn<0.6 | MC_donors | 1 | 1 | 0.2214 | 0.2214 | 0.2214 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.2175 | 0.2175 | 0.2175 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | offset_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.2165 | 0.2165 | 0.2165 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | offset_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.2133 | 0.2133 | 0.2133 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.2097 | 0.2097 | 0.2097 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.2076 | 0.2076 | 0.2076 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | offset_mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.1885 | 0.1885 | 0.1885 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | offset_mae | all | MC_donors | 1 | 1 | 0.1846 | 0.1846 | 0.1846 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | offset_mae | all | MC_donors | 1 | 1 | 0.1795 | 0.1795 | 0.1795 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | offset_mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.1773 | 0.1773 | 0.1773 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | offset_mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.1756 | 0.1756 | 0.1756 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | mae | all | MC_donors | 1 | 1 | 0.1755 | 0.1755 | 0.1755 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.1726 | 0.1726 | 0.1726 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | mae | all | MC_donors | 1 | 1 | 0.1721 | 0.1721 | 0.1721 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.1633 | 0.1633 | 0.1633 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.1614 | 0.1614 | 0.1614 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | shape_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.07559 | 0.07559 | 0.07559 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | shape_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.07425 | 0.07425 | 0.07425 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | shape_mae | all | MC_donors | 1 | 1 | 0.07363 | 0.07363 | 0.07363 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | shape_mae | nn<0.6 | MC_donors | 1 | 1 | 0.06997 | 0.06997 | 0.06997 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | shape_mae | nn<0.4 | MC_donors | 1 | 1 | 0.06958 | 0.06958 | 0.06958 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | shape_mae | nn<0.4 | MC_donors | 1 | 1 | 0.05412 | 0.05412 | 0.05412 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | shape_mae | nn<0.4 | MC_donors | 1 | 1 | 0.05211 | 0.05211 | 0.05211 | 1 | 1 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | shape_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.03963 | 0.03963 | 0.03963 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | shape_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.03632 | 0.03632 | 0.03632 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | shape_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.03576 | 0.03576 | 0.03576 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | shape_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | 0.03462 | 0.03462 | 0.03462 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | shape_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.03413 | 0.03413 | 0.03413 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_SHUFFLED | EXPANDED_SHUFFLED | EXPANDED | shape_mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.03245 | 0.03245 | 0.03245 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | shape_mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.02429 | 0.02429 | 0.02429 | 1 | 1 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | shape_mae | nn<0.6 | MC_donors | 1 | 1 | 0.0227 | 0.0227 | 0.0227 | 1 | 1 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | shape_mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.02154 | 0.02154 | 0.02154 | 1 | 0 | 1 | yes |
| EXPANDED_vs_BASE | BASE | EXPANDED | shape_mae | all | MC_donors | 1 | 1 | 0.01788 | 0.01788 | 0.01788 | 1 | 0 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | shape_mae | nn<0.6 | MC_donors | 1 | 1 | 0.01685 | 0.01685 | 0.01685 | 1 | 0 | 1 | yes |
| EXPANDED_ROWMATCHED_vs_BASE | BASE | EXPANDED_ROWMATCHED | shape_mae | all | MC_donors | 1 | 1 | 0.01126 | 0.01126 | 0.01126 | 1 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | shape_mae | all | MC_donors | 1 | 1 | 0.006612 | 0.006612 | 0.006612 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | shape_mae | nn<0.6 | MC_donors | 1 | 1 | 0.005851 | 0.005851 | 0.005851 | 1 | 1 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | offset_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.003217 | 0.003217 | 0.003217 | 1 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | shape_mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.002757 | 0.002757 | 0.002757 | 1 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.002017 | 0.002017 | 0.002017 | 1 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | shape_mae | nn<0.4 | MC_donors | 1 | 1 | 0.002011 | 0.002011 | 0.002011 | 1 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.001867 | 0.001867 | 0.001867 | 1 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | offset_mae | all | MC_lig2d_ext_massaction | 1 | 1 | 0.001688 | 0.001688 | 0.001688 | 1 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | shape_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | 0.001631 | 0.001631 | 0.001631 | 1 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | shape_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | -0.0005602 | -0.0005602 | -0.0005602 | 0 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | shape_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | -0.001345 | -0.001345 | -0.001345 | 0 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | offset_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | -0.002891 | -0.002891 | -0.002891 | 0 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | mae | all | MC_donors | 1 | 1 | -0.003444 | -0.003444 | -0.003444 | 0 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | offset_mae | nn<0.6 | MC_donors | 1 | 1 | -0.004591 | -0.004591 | -0.004591 | 0 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | offset_mae | all | MC_donors | 1 | 1 | -0.005101 | -0.005101 | -0.005101 | 0 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | -0.005123 | -0.005123 | -0.005123 | 0 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | mae | nn<0.6 | MC_donors | 1 | 1 | -0.006867 | -0.006867 | -0.006867 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | -0.007792 | -0.007792 | -0.007792 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | shape_mae | all | MC_lig2d_ext_massaction | 1 | 1 | -0.008153 | -0.008153 | -0.008153 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | mae | all | MC_lig2d_ext_massaction | 1 | 1 | -0.009293 | -0.009293 | -0.009293 | 0 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | mae | nn<0.4 | MC_donors | 1 | 1 | -0.01086 | -0.01086 | -0.01086 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | offset_mae | all | MC_lig2d_ext_massaction | 1 | 1 | -0.01122 | -0.01122 | -0.01122 | 0 | 0 | 1 | yes |
| EXPANDED_vs_EXPANDED_ROWMATCHED | EXPANDED_ROWMATCHED | EXPANDED | offset_mae | nn<0.4 | MC_donors | 1 | 1 | -0.01271 | -0.01271 | -0.01271 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | shape_mae | nn<0.4 | MC_donors | 1 | 1 | -0.01546 | -0.01546 | -0.01546 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | offset_mae | nn<0.6 | MC_lig2d_ext_massaction | 1 | 1 | -0.02088 | -0.02088 | -0.02088 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | -0.03142 | -0.03142 | -0.03142 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | offset_mae | nn<0.4 | MC_donors | 1 | 1 | -0.04441 | -0.04441 | -0.04441 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | shape_mae | nn<0.6 | MC_donors | 1 | 1 | -0.04727 | -0.04727 | -0.04727 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | offset_mae | all | MC_donors | 1 | 1 | -0.05127 | -0.05127 | -0.05127 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | mae | nn<0.4 | MC_donors | 1 | 1 | -0.05134 | -0.05134 | -0.05134 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | offset_mae | nn<0.6 | MC_donors | 1 | 1 | -0.05193 | -0.05193 | -0.05193 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | shape_mae | all | MC_donors | 1 | 1 | -0.05576 | -0.05576 | -0.05576 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | mae | nn<0.6 | MC_donors | 1 | 1 | -0.05612 | -0.05612 | -0.05612 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | mae | all | MC_donors | 1 | 1 | -0.06469 | -0.06469 | -0.06469 | 0 | 0 | 1 | yes |
| EXPANDED_SHUFFLED_vs_BASE | BASE | EXPANDED_SHUFFLED | offset_mae | nn<0.4 | MC_lig2d_ext_massaction | 1 | 1 | -0.06813 | -0.06813 | -0.06813 | 0 | 0 | 1 | yes |

## 5. What would falsify the aggregate

* Any of the section 1 fingerprints differing between two inputs. That is checked, and fatal.
* An arm whose rank flips between runs while its across-run sd exceeds the gap to its neighbour: the ranking is then noise, whatever the mean says.
* A contrast whose CI95 low is above zero in some runs and below in others (`n_ci95_low_above_zero` strictly between 0 and `n_runs`): the effect is not established.
* An input run whose `validated_ok` is not `yes`: its numbers were produced by a run that could not prove its own artifacts. Use `--require-success` to make that fatal.

