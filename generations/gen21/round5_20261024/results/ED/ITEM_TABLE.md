# Item 1.6 (E-D): per-item table

**Label: EXPLORATORY (seen); reproduction run TECHNICAL SMOKE; BP and C5-BTBP out SENSITIVITY**

| field | value |
|---|---|
| Item id and estimand | 1.6 E-D: M18 read-budget ladder + M17 policy envelope + recipe-2 row |
| Population (cells, publications, units, rows per seed) | row: C3 min(k, k_max), k = reads - 1 (E-A's queries); cells: 229; publications: 39; unit: cluster_cpg; units: 13; rows_per_seed: 1802.0000; seeds: 5 |
| Design | BPG (primary) and BP (labelled sensitivity); identical test folds |
| Read budget | 2-7 reads (code k = 1-6); 4 reads = the E-A rung; every other rung (2, 3, 5-7 reads) = a different read budget.  The row 'C3 min(k, k_max), k = reads - 1' at 5 reads is a mixture: BPG 286 of 1,802 rows per seed (the k_max = 3 cells) keep 4 reads, B1 R2 0.6018 on them; BP 286 of 1,802 rows per seed (the k_max = 3 cells) keep 4 reads, B1 R2 0.6021 on them; C4 (k_max >= 4) is the like-for-like 5-metal number (B1 at 5 reads BPG 0.4798, BP 0.4821) |
| Arms on identical rows | B1 (beta refit per fold and budget), B0 (beta = 0), HIST_D7 (DIAGNOSTIC), B1_k3beta (SENSITIVITY); 5th-read policies: the written label-free family DOPTQ, MAXMIN_R, MAXMIN_Z, BISECT_R, BISECT_Z, BISECT_RMID, HULL_BISECT, RANDOM; ORACLE_B1 [O] (bound only, never a member); LOOSE sensitivity MAXMIN_R_LOOSE, RANDOM_LOOSE; recipe-2 LADDER, LADDER_GUARD, LOCO_STATIC, HULL, DOPTQ, RANDOM_SET |
| R², MSE, relative SSE vs B1 | see 'R², MSE, relative SSE vs B1' below |
| Per-seed values | BPG C3 min(k, k_max), k = reads - 1 (5 reads), B1, DOPT_Q-nested (different read budget): 0.5047;0.4786;0.5026;0.5012;0.4967; BPG C4 fixed cohort (k_max >= 4), 5 reads, B1, like-for-like 5-metal (different read budget): 0.4893;0.4598;0.4853;0.4844;0.4801; BP C3 min(k, k_max), k = reads - 1 (5 reads), B1, DOPT_Q-nested (different read budget): 0.5079;0.4792;0.5035;0.5007;0.5027; BP C4 fixed cohort (k_max >= 4), 5 reads, B1, like-for-like 5-metal (different read budget): 0.4930;0.4599;0.4865;0.4840;0.4869 |
| Unit W/L/T and the descriptive tests | see 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | see 'LOPO and top-3 share' below |
| C5-BTBP in / out | see 'C5-BTBP in / out' below |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | BPG: isolation 25/25 folds, intervention failed 0/22420, peek caught 25/25, both-ends-read 0, illegal 0, DOPT 5th read not legal 0, static-set intervention failed 0/82026; BP: isolation 25/25 folds, intervention failed 0/22458, peek caught 25/25, both-ends-read 0, illegal 0, DOPT 5th read not legal 0, static-set intervention failed 0/82026; fit poisoning of the k-specific beta refits: all_identical True, matches_main_run True, ks [1, 2, 3, 4, 5, 6], 50 fold-fits, seal b96c3bef9828 (gate: passed); LOCO static selection poisoning: True |
| Compute | from-rows pass 187 s (BPG 92 s, BP 94 s); earlier passes under seal b96c3bef9828 (RUN_RECORD.md): reproduction (TECHNICAL SMOKE) 260 s, arms (rows, candidates, fits, checks) 907 s, fit poisoning of the k-specific beta refits 1208 s (1 pass); total 2562 s |
| Label | EXPLORATORY (seen); reproduction run TECHNICAL SMOKE; BP and C5-BTBP out SENSITIVITY |
| What changed scientifically | Paired on E-A's identical queries of the row 'C3 min(k, k_max), k = reads - 1', the DOPT_Q-nested 5th read changes B1's pooled R2 by BPG +0.0447, BP +0.0469 (a different read budget, EXPLORATORY seen, never an E-A value; at 5 reads BPG 286 of 1,802, BP 286 of 1,802 rows per seed keep 4 reads), and the like-for-like C4 5-metal level is BPG 0.4798, BP 0.4821. The pre-declared Pareto question (1.6-R01; vs DOPTQ, C3 min(4, k_max), 5 reads): no member of the written label-free family Pareto-dominates DOPTQ (BPG: none; BP: none); the ORACLE_B1 [O] bound (never a member) meets the criterion in BPG and BP; LOOSE sensitivity: BPG: none dominates; BP: none dominates. The saturation density-invariance rule (1.6-R02; CONSISTENCY CHECK, not a kill) fires in BPG and BP (BPG: strata below 0.48 with >= 3 clusters k_max 4 at 5 reads 0.4440, spread 0.1578; BP: strata below 0.48 with >= 3 clusters k_max 4 at 5 reads 0.4466, spread 0.1555). |
| Next registered step | Registry item 1.7 (M13 wider-pair table; E-W (EXPLORATORY, seen); registry status PLANNED) |

## R², MSE, relative SSE vs B1

| design | frame | row | rung_estimand | level_note | R2 | MSE | MAE | relSSE_vs_B1_4reads | seeds_ge_050 | P_boot_ge_050 (descriptive; never a posterior) | loco_min | loco_max | lopo_min | lopo_max | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BPG | C3pop\|r2 | C3 min(k, k_max), k = reads - 1 (2 reads) | E-D (different read budget) | NaN | 0.2317 | 0.0379 | 0.1293 | -0.4026 | 0.0000 | 0.0010 | 0.2088 | 0.2779 | 0.2198 | 0.2562 | EXPLORATORY (seen); different read budget |
| BPG | C3pop\|r3 | C3 min(k, k_max), k = reads - 1 (3 reads) | E-D (different read budget) | NaN | 0.4033 | 0.0294 | 0.1076 | -0.0890 | 0.0000 | 0.0070 | 0.3939 | 0.4174 | 0.3913 | 0.4296 | EXPLORATORY (seen); different read budget |
| BPG | C3pop\|r4 | C3 min(k, k_max), k = reads - 1 (4 reads) | E-A rung (4 reads; B1 = the E-A reference) | (E-A rows) | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | EXPLORATORY (seen) |
| BPG | C3pop\|r5 | C3 min(k, k_max), k = reads - 1 (5 reads) | E-D (different read budget) | NaN | 0.4968 | 0.0248 | 0.0934 | 0.0815 | 3.0000 | 0.4315 | 0.4857 | 0.5194 | 0.4866 | 0.5311 | EXPLORATORY (seen); different read budget |
| BPG | C4\|r4 | C4 fixed cohort (k_max >= 4), 4 reads | E-D 4-read rung of a fixed cohort | NaN | 0.4290 | 0.0294 | 0.1075 | 0.0000 | 0.0000 | 0.0070 | 0.4122 | 0.4503 | 0.4204 | 0.4647 | EXPLORATORY (seen) |
| BPG | C4\|r5 | C4 fixed cohort (k_max >= 4), 5 reads | E-D (different read budget) | NaN | 0.4798 | 0.0268 | 0.0996 | 0.0889 | 0.0000 | 0.1560 | 0.4616 | 0.5030 | 0.4689 | 0.5171 | EXPLORATORY (seen); different read budget |
| BPG | C5\|r6 | C5 fixed cohort (k_max >= 5), 6 reads | E-D (different read budget) | NaN | 0.5251 | 0.0247 | 0.0926 | 0.1681 | 5.0000 | 0.8015 | 0.4983 | 0.5518 | 0.5098 | 0.5679 | EXPLORATORY (seen); different read budget |
| BPG | C6\|r7 | C6 fixed cohort (k_max >= 6), 7 reads | E-D (different read budget) | NaN | 0.5558 | 0.0223 | 0.0848 | 0.2186 | 5.0000 | 0.9630 | 0.5280 | 0.5905 | 0.5385 | 0.6070 | EXPLORATORY (seen); different read budget |
| BP | C3pop\|r2 | C3 min(k, k_max), k = reads - 1 (2 reads) | E-D (different read budget) | NaN | 0.2345 | 0.0377 | 0.1287 | -0.3969 | 0.0000 | 0.0010 | 0.2119 | 0.2775 | 0.2227 | 0.2605 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BP | C3pop\|r3 | C3 min(k, k_max), k = reads - 1 (3 reads) | E-D (different read budget) | NaN | 0.4020 | 0.0295 | 0.1076 | -0.0909 | 0.0000 | 0.0070 | 0.3924 | 0.4160 | 0.3898 | 0.4282 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BP | C3pop\|r4 | C3 min(k, k_max), k = reads - 1 (4 reads) | E-A rung (4 reads; B1 = the E-A reference) | (E-A rows) | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | EXPLORATORY (seen); SENSITIVITY (design BP) |
| BP | C3pop\|r5 | C3 min(k, k_max), k = reads - 1 (5 reads) | E-D (different read budget) | NaN | 0.4988 | 0.0247 | 0.0934 | 0.0856 | 4.0000 | 0.4670 | 0.4880 | 0.5222 | 0.4885 | 0.5337 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BP | C4\|r4 | C4 fixed cohort (k_max >= 4), 4 reads | E-D 4-read rung of a fixed cohort | NaN | 0.4288 | 0.0294 | 0.1073 | 0.0000 | 0.0000 | 0.0075 | 0.4117 | 0.4509 | 0.4203 | 0.4645 | EXPLORATORY (seen); SENSITIVITY (design BP) |
| BP | C4\|r5 | C4 fixed cohort (k_max >= 4), 5 reads | E-D (different read budget) | NaN | 0.4821 | 0.0267 | 0.0995 | 0.0933 | 0.0000 | 0.1810 | 0.4641 | 0.5062 | 0.4692 | 0.5201 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BP | C5\|r6 | C5 fixed cohort (k_max >= 5), 6 reads | E-D (different read budget) | NaN | 0.5289 | 0.0245 | 0.0924 | 0.1751 | 5.0000 | 0.8315 | 0.5021 | 0.5570 | 0.5128 | 0.5729 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BP | C6\|r7 | C6 fixed cohort (k_max >= 6), 7 reads | E-D (different read budget) | NaN | 0.5592 | 0.0221 | 0.0845 | 0.2243 | 5.0000 | 0.9680 | 0.5317 | 0.5953 | 0.5409 | 0.6115 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |

## Unit W/L/T and the descriptive tests

| design | contrast | variant | label | G | W/L/T | SF_POOL_p | SIGN_p | SF_LOG_p | IM_REL_p | top3_unit_share | top3_unit_share_abs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BPG | C3pop 5 vs 4 reads | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 9/2/2 | 0.0029 | 0.0327 | 0.0054 | 0.0074 | 0.8142 | 0.8123 |
| BPG | C4 5 vs 4 reads | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 11 | 9/2/0 | 0.0029 | 0.0327 | 0.0054 | 0.0060 | 0.8142 | 0.8123 |
| BPG | C6 7 vs 4 reads | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 10 | 10/0/0 | 0.0010 | 0.0010 | 0.0010 | 0.0003 | 0.8165 | 0.8165 |
| BPG | BISECT_R - DOPTQ (C3pop, 5 reads) | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 8/2/3 | 0.1270 | 0.0547 | 0.0371 | 0.0392 | 1.2449 | 0.8430 |
| BPG | RANDOM - DOPTQ (C3pop, 5 reads) | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 4/6/3 | 0.7510 | 0.8281 | 0.3447 | 0.3659 | NaN | 0.7068 |
| BPG | BISECT_R - RANDOM (C3pop, 5 reads) | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 8/2/3 | 0.0205 | 0.0547 | 0.0078 | 0.0126 | 0.8753 | 0.8119 |
| BPG | HULL_BISECT - DOPTQ (C3pop, 5 reads) | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 6/4/3 | 0.4502 | 0.3770 | 0.3115 | 0.3049 | 9.3575 | 0.7710 |
| BPG | LADDER - HULL (2 reads) | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 6/6/1 | 0.8313 | 0.6128 | 0.0349 | 0.0415 | NaN | 0.7785 |
| BPG | LADDER - HULL (3 reads) | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 7/4/2 | 0.0923 | 0.2744 | 0.1250 | 0.1567 | 1.0414 | 0.7936 |
| BPG | LADDER - HULL (4 reads) | all | EXPLORATORY (seen); DIAGNOSTIC (descriptive test; never called significant) | 13 | 9/2/2 | 0.0073 | 0.0327 | 0.0029 | 0.0221 | 0.8648 | 0.8209 |
| BPG | LADDER - HULL (5 reads) | all | EXPLORATORY (seen); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 3/8/2 | 0.7417 | 0.9673 | 0.6860 | 0.7516 | NaN | 0.8111 |
| BP | C3pop 5 vs 4 reads | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 10/1/2 | 0.0044 | 0.0059 | 0.0024 | 0.0039 | 0.8178 | 0.8150 |
| BP | C4 5 vs 4 reads | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 11 | 10/1/0 | 0.0044 | 0.0059 | 0.0024 | 0.0029 | 0.8178 | 0.8150 |
| BP | C6 7 vs 4 reads | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 10 | 10/0/0 | 0.0010 | 0.0010 | 0.0010 | 0.0002 | 0.8253 | 0.8253 |
| BP | BISECT_R - DOPTQ (C3pop, 5 reads) | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 7/3/3 | 0.1348 | 0.1719 | 0.1387 | 0.1589 | 1.3169 | 0.8098 |
| BP | RANDOM - DOPTQ (C3pop, 5 reads) | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 3/7/3 | 0.8555 | 0.9453 | 0.3721 | 0.3931 | NaN | 0.6793 |
| BP | BISECT_R - RANDOM (C3pop, 5 reads) | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 8/2/3 | 0.0283 | 0.0547 | 0.0488 | 0.0530 | 0.8586 | 0.7945 |
| BP | HULL_BISECT - DOPTQ (C3pop, 5 reads) | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 6/4/3 | 0.4375 | 0.3770 | 0.3936 | 0.4516 | 8.3709 | 0.7810 |
| BP | LADDER - HULL (2 reads) | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 6/6/1 | 0.8311 | 0.6128 | 0.0349 | 0.0411 | NaN | 0.7784 |
| BP | LADDER - HULL (3 reads) | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 7/4/2 | 0.0903 | 0.2744 | 0.1245 | 0.1561 | 1.0390 | 0.7925 |
| BP | LADDER - HULL (4 reads) | all | EXPLORATORY (seen); SENSITIVITY (design BP); DIAGNOSTIC (descriptive test; never called significant) | 13 | 9/2/2 | 0.0073 | 0.0327 | 0.0029 | 0.0213 | 0.8646 | 0.8206 |
| BP | LADDER - HULL (5 reads) | all | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; DIAGNOSTIC (descriptive test; never called significant) | 13 | 3/8/2 | 0.7417 | 0.9673 | 0.6860 | 0.7480 | NaN | 0.8129 |

## LOPO and top-3 share

| design | frame | row | delta_R2 | relSSE | seeds_pos | MDE80 | lopo_min | lopo_max | lopo_min_by | lopo_max_by | units_WLT | top3_unit_share | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BPG | C3pop: r5 - r4 | C3 min(k, k_max), k = reads - 1 (5 reads) - 4 reads (identical queries) | 0.0447 | 0.0815 | 5 | 0.0242 | 0.0413 | 0.0471 | pub_b8175dc3a1 | pub_d68ed8c8cf | 9/2/2 | 0.8142 | EXPLORATORY (seen); different read budget (paired, identical queries) |
| BPG | C4: r5 - r4 | C4 fixed cohort (k_max >= 4), 5 reads - 4 reads (identical queries) | 0.0507 | 0.0889 | 5 | 0.0223 | 0.0470 | 0.0534 | pub_b8175dc3a1 | pub_126bbe7976 | 9/2/0 | 0.8142 | EXPLORATORY (seen); different read budget (paired, identical queries) |
| BPG | C6: r7 - r4 | C6 fixed cohort (k_max >= 6), 7 reads - 4 reads (identical queries) | 0.1241 | 0.2186 | 5 | 0.0690 | 0.1158 | 0.1312 | pub_8415538291 | pub_6a0ef8528c | 10/0/0 | 0.8165 | EXPLORATORY (seen); different read budget (paired, identical queries) |
| BP | C3pop: r5 - r4 | C3 min(k, k_max), k = reads - 1 (5 reads) - 4 reads (identical queries) | 0.0469 | 0.0856 | 5 | 0.0228 | 0.0427 | 0.0495 | pub_b0b3296ec5 | pub_d68ed8c8cf | 10/1/2 | 0.8178 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget (paired, identical queries) |
| BP | C4: r5 - r4 | C4 fixed cohort (k_max >= 4), 5 reads - 4 reads (identical queries) | 0.0532 | 0.0933 | 5 | 0.0209 | 0.0489 | 0.0560 | pub_b0b3296ec5 | pub_a49146c8fd | 10/1/0 | 0.8178 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget (paired, identical queries) |
| BP | C6: r7 - r4 | C6 fixed cohort (k_max >= 6), 7 reads - 4 reads (identical queries) | 0.1272 | 0.2243 | 5 | 0.0643 | 0.1185 | 0.1353 | pub_b0b3296ec5 | pub_6a0ef8528c | 10/0/0 | 0.8253 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget (paired, identical queries) |
| BPG | C3pop: BISECT_R - DOPTQ | C3 min(k, k_max), k = reads - 1 (5 reads) | 0.0062 | 0.0120 | 4 | 0.0207 | 0.0026 | 0.0096 | pub_b0b3296ec5 | pub_b8175dc3a1 | 8/2/3 | 1.2449 | EXPLORATORY (seen); different read budget |
| BPG | C3pop: RANDOM - DOPTQ | C3 min(k, k_max), k = reads - 1 (5 reads) | -0.0023 | -0.0046 | 3 | 0.0124 | -0.0038 | -0.0001 | pub_8415538291 | pub_b8175dc3a1 | 4/6/3 | NaN | EXPLORATORY (seen); different read budget |
| BPG | C3pop: BISECT_R - RANDOM | C3 min(k, k_max), k = reads - 1 (5 reads) | 0.0085 | 0.0165 | 4 | 0.0092 | 0.0059 | 0.0103 | pub_b0b3296ec5 | pub_6a0ef8528c | 8/2/3 | 0.8753 | EXPLORATORY (seen); different read budget |
| BP | C3pop: BISECT_R - DOPTQ | C3 min(k, k_max), k = reads - 1 (5 reads) | 0.0056 | 0.0110 | 3 | 0.0208 | 0.0029 | 0.0090 | pub_8415538291 | pub_b8175dc3a1 | 7/3/3 | 1.3169 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BP | C3pop: RANDOM - DOPTQ | C3 min(k, k_max), k = reads - 1 (5 reads) | -0.0035 | -0.0071 | 3 | 0.0130 | -0.0050 | -0.0013 | pub_8415538291 | pub_b8175dc3a1 | 3/7/3 | NaN | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BP | C3pop: BISECT_R - RANDOM | C3 min(k, k_max), k = reads - 1 (5 reads) | 0.0091 | 0.0179 | 5 | 0.0095 | 0.0067 | 0.0111 | pub_b0b3296ec5 | pub_6a0ef8528c | 8/2/3 | 0.8586 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BPG | r2: LADDER - HULL | C3 min(k, k_max), k = reads - 1 (2 reads) | -0.0273 | -0.0386 | 0 | 0.0494 | -0.0408 | -0.0070 | pub_b0b3296ec5 | pub_6a0ef8528c | 6/6/1 | NaN | EXPLORATORY (seen); different read budget; primary recipe-2 contrast |
| BPG | r4: LADDER - HULL | C3 min(k, k_max), k = reads - 1 (4 reads) | 0.0420 | 0.0727 | 5 | 0.0348 | 0.0332 | 0.0570 | pub_b0b3296ec5 | pub_126bbe7976 | 9/2/2 | 0.8648 | EXPLORATORY (seen); primary recipe-2 contrast |
| BP | r2: LADDER - HULL | C3 min(k, k_max), k = reads - 1 (2 reads) | -0.0273 | -0.0386 | 0 | 0.0495 | -0.0408 | -0.0070 | pub_b0b3296ec5 | pub_6a0ef8528c | 6/6/1 | NaN | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; primary recipe-2 contrast |
| BP | r4: LADDER - HULL | C3 min(k, k_max), k = reads - 1 (4 reads) | 0.0419 | 0.0726 | 5 | 0.0345 | 0.0331 | 0.0569 | pub_b0b3296ec5 | pub_126bbe7976 | 9/2/2 | 0.8646 | EXPLORATORY (seen); SENSITIVITY (design BP); primary recipe-2 contrast |

## C5-BTBP in / out

| design | frame | row | R2_all | R2_noC5 | R2_noC5cell | level_note | label |
|---|---|---|---|---|---|---|---|
| BPG | C3pop\|r2 | C3 min(k, k_max), k = reads - 1 (2 reads) | 0.2317 | 0.2562 | 0.2472 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; different read budget |
| BPG | C3pop\|r3 | C3 min(k, k_max), k = reads - 1 (3 reads) | 0.4033 | 0.4296 | 0.4295 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; different read budget |
| BPG | C3pop\|r4 | C3 min(k, k_max), k = reads - 1 (4 reads) | NaN | NaN | NaN | (E-A rows) | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY |
| BPG | C3pop\|r5 | C3 min(k, k_max), k = reads - 1 (5 reads) | 0.4968 | 0.5311 | 0.5283 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; different read budget |
| BPG | C4\|r4 | C4 fixed cohort (k_max >= 4), 4 reads | 0.4290 | 0.4647 | 0.4603 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY |
| BPG | C4\|r5 | C4 fixed cohort (k_max >= 4), 5 reads | 0.4798 | 0.5171 | 0.5146 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; different read budget |
| BPG | C5\|r6 | C5 fixed cohort (k_max >= 5), 6 reads | 0.5251 | 0.5679 | 0.5656 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; different read budget |
| BPG | C6\|r7 | C6 fixed cohort (k_max >= 6), 7 reads | 0.5558 | 0.6070 | 0.6031 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; different read budget |
| BP | C3pop\|r2 | C3 min(k, k_max), k = reads - 1 (2 reads) | 0.2345 | 0.2605 | 0.2503 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; SENSITIVITY (design BP); different read budget |
| BP | C3pop\|r3 | C3 min(k, k_max), k = reads - 1 (3 reads) | 0.4020 | 0.4282 | 0.4281 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; SENSITIVITY (design BP); different read budget |
| BP | C3pop\|r4 | C3 min(k, k_max), k = reads - 1 (4 reads) | NaN | NaN | NaN | (E-A rows) | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; SENSITIVITY (design BP) |
| BP | C3pop\|r5 | C3 min(k, k_max), k = reads - 1 (5 reads) | 0.4988 | 0.5337 | 0.5306 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; SENSITIVITY (design BP); different read budget |
| BP | C4\|r4 | C4 fixed cohort (k_max >= 4), 4 reads | 0.4288 | 0.4645 | 0.4601 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; SENSITIVITY (design BP) |
| BP | C4\|r5 | C4 fixed cohort (k_max >= 4), 5 reads | 0.4821 | 0.5201 | 0.5171 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; SENSITIVITY (design BP); different read budget |
| BP | C5\|r6 | C5 fixed cohort (k_max >= 5), 6 reads | 0.5289 | 0.5729 | 0.5698 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; SENSITIVITY (design BP); different read budget |
| BP | C6\|r7 | C6 fixed cohort (k_max >= 6), 7 reads | 0.5592 | 0.6115 | 0.6069 |  | EXPLORATORY (seen); C5-BTBP out = SENSITIVITY; SENSITIVITY (design BP); different read budget |

## Duplicate sensitivities

| design | estimand | family | frame | arm | variant | row | label | dropped_cells | R2 | rows_per_seed | level_note | delta_R2 | seeds_pos | per_seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BPG | E-D | DUPLICATES | C3pop\|r4 | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (4 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2) | 20.0000 | NaN | 1636.0000 | (E-A rows) | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop\|r5 | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (5 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); different read budget | 20.0000 | 0.4982 | 1636.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C4\|r4 | B1 | noV2 | C4 fixed cohort (k_max >= 4), 4 reads | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2) | 13.0000 | 0.4315 | 1395.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C4\|r5 | B1 | noV2 | C4 fixed cohort (k_max >= 4), 5 reads | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); different read budget | 13.0000 | 0.4803 | 1395.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop\|r5\|BISECT_R | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (5 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); different read budget | 20.0000 | 0.5023 | 1636.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop\|r2\|LADDER | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (2 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); different read budget | 20.0000 | 0.2698 | 1636.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop\|r2\|HULL | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (2 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); different read budget | 20.0000 | 0.2944 | 1636.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop 5 vs 4 reads | B1 | noV2 | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); different read budget; paired, identical queries | NaN | NaN | 1636.0000 | NaN | 0.0449 | 5.0000 | 0.0540;0.0354;0.0439;0.0448;0.0463 |
| BPG | E-D | DUPLICATES | C4 5 vs 4 reads | B1 | noV2 | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); different read budget; paired, identical queries | NaN | NaN | 1395.0000 | NaN | 0.0488 | 5.0000 | 0.0588;0.0385;0.0478;0.0487;0.0504 |
| BPG | E-D | DUPLICATES | BISECT_R - DOPTQ (C3pop, 5 reads) | B1 | noV2 | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); different read budget; paired, identical queries | NaN | NaN | 1636.0000 | NaN | 0.0040 | 3.0000 | -0.0061;0.0212;-0.0005;0.0050;0.0005 |
| BPG | E-D | DUPLICATES | LADDER - HULL (2 reads) | B1 | noV2 | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); different read budget; paired, identical queries | NaN | NaN | 1636.0000 | NaN | -0.0246 | 0.0000 | -0.0249;-0.0236;-0.0245;-0.0251;-0.0250 |
| BPG | E-D | DUPLICATES | C3pop\|r4 | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (4 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP) | 6.0000 | NaN | 1780.0000 | (E-A rows) | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop\|r5 | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (5 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); different read budget | 6.0000 | 0.4966 | 1780.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C4\|r4 | B1 | noBITDUP | C4 fixed cohort (k_max >= 4), 4 reads | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP) | 6.0000 | 0.4287 | 1494.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C4\|r5 | B1 | noBITDUP | C4 fixed cohort (k_max >= 4), 5 reads | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); different read budget | 6.0000 | 0.4795 | 1494.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop\|r5\|BISECT_R | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (5 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); different read budget | 6.0000 | 0.5030 | 1780.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop\|r2\|LADDER | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (2 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); different read budget | 6.0000 | 0.2648 | 1780.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop\|r2\|HULL | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (2 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); different read budget | 6.0000 | 0.2919 | 1780.0000 | NaN | NaN | NaN | NaN |
| BPG | E-D | DUPLICATES | C3pop 5 vs 4 reads | B1 | noBITDUP | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); different read budget; paired, identical queries | NaN | NaN | 1780.0000 | NaN | 0.0447 | 5.0000 | 0.0553;0.0416;0.0408;0.0428;0.0429 |
| BPG | E-D | DUPLICATES | C4 5 vs 4 reads | B1 | noBITDUP | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); different read budget; paired, identical queries | NaN | NaN | 1494.0000 | NaN | 0.0508 | 5.0000 | 0.0630;0.0473;0.0464;0.0487;0.0488 |
| BPG | E-D | DUPLICATES | BISECT_R - DOPTQ (C3pop, 5 reads) | B1 | noBITDUP | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); different read budget; paired, identical queries | NaN | NaN | 1780.0000 | NaN | 0.0063 | 4.0000 | -0.0046;0.0193;0.0058;0.0096;0.0016 |
| BPG | E-D | DUPLICATES | LADDER - HULL (2 reads) | B1 | noBITDUP | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); different read budget; paired, identical queries | NaN | NaN | 1780.0000 | NaN | -0.0271 | 0.0000 | -0.0273;-0.0262;-0.0270;-0.0275;-0.0275 |
| BP | E-D | DUPLICATES | C3pop\|r4 | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (4 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP) | 20.0000 | NaN | 1636.0000 | (E-A rows) | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop\|r5 | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (5 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP); different read budget | 20.0000 | 0.4995 | 1636.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C4\|r4 | B1 | noV2 | C4 fixed cohort (k_max >= 4), 4 reads | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP) | 13.0000 | 0.4305 | 1395.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C4\|r5 | B1 | noV2 | C4 fixed cohort (k_max >= 4), 5 reads | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP); different read budget | 13.0000 | 0.4817 | 1395.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop\|r5\|BISECT_R | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (5 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP); different read budget | 20.0000 | 0.5032 | 1636.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop\|r2\|LADDER | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (2 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP); different read budget | 20.0000 | 0.2690 | 1636.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop\|r2\|HULL | B1 | noV2 | C3 min(k, k_max), k = reads - 1 (2 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP); different read budget | 20.0000 | 0.2936 | 1636.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop 5 vs 4 reads | B1 | noV2 | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP); different read budget; paired, identical queries | NaN | NaN | 1636.0000 | NaN | 0.0470 | 5.0000 | 0.0528;0.0336;0.0479;0.0452;0.0558 |
| BP | E-D | DUPLICATES | C4 5 vs 4 reads | B1 | noV2 | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP); different read budget; paired, identical queries | NaN | NaN | 1395.0000 | NaN | 0.0512 | 5.0000 | 0.0575;0.0365;0.0521;0.0492;0.0607 |
| BP | E-D | DUPLICATES | BISECT_R - DOPTQ (C3pop, 5 reads) | B1 | noV2 | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP); different read budget; paired, identical queries | NaN | NaN | 1636.0000 | NaN | 0.0037 | 3.0000 | -0.0074;0.0212;0.0013;0.0056;-0.0022 |
| BP | E-D | DUPLICATES | LADDER - HULL (2 reads) | B1 | noV2 | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noV2); SENSITIVITY (design BP); different read budget; paired, identical queries | NaN | NaN | 1636.0000 | NaN | -0.0246 | 0.0000 | -0.0249;-0.0236;-0.0246;-0.0251;-0.0250 |
| BP | E-D | DUPLICATES | C3pop\|r4 | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (4 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP) | 6.0000 | NaN | 1780.0000 | (E-A rows) | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop\|r5 | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (5 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP); different read budget | 6.0000 | 0.4990 | 1780.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C4\|r4 | B1 | noBITDUP | C4 fixed cohort (k_max >= 4), 4 reads | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP) | 6.0000 | 0.4285 | 1494.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C4\|r5 | B1 | noBITDUP | C4 fixed cohort (k_max >= 4), 5 reads | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP); different read budget | 6.0000 | 0.4821 | 1494.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop\|r5\|BISECT_R | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (5 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP); different read budget | 6.0000 | 0.5047 | 1780.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop\|r2\|LADDER | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (2 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP); different read budget | 6.0000 | 0.2641 | 1780.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop\|r2\|HULL | B1 | noBITDUP | C3 min(k, k_max), k = reads - 1 (2 reads) | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP); different read budget | 6.0000 | 0.2913 | 1780.0000 | NaN | NaN | NaN | NaN |
| BP | E-D | DUPLICATES | C3pop 5 vs 4 reads | B1 | noBITDUP | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP); different read budget; paired, identical queries | NaN | NaN | 1780.0000 | NaN | 0.0472 | 5.0000 | 0.0541;0.0399;0.0450;0.0432;0.0536 |
| BP | E-D | DUPLICATES | C4 5 vs 4 reads | B1 | noBITDUP | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP); different read budget; paired, identical queries | NaN | NaN | 1494.0000 | NaN | 0.0536 | 5.0000 | 0.0616;0.0454;0.0511;0.0491;0.0610 |
| BP | E-D | DUPLICATES | BISECT_R - DOPTQ (C3pop, 5 reads) | B1 | noBITDUP | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP); different read budget; paired, identical queries | NaN | NaN | 1780.0000 | NaN | 0.0057 | 3.0000 | -0.0058;0.0192;0.0079;0.0101;-0.0031 |
| BP | E-D | DUPLICATES | LADDER - HULL (2 reads) | B1 | noBITDUP | NaN | EXPLORATORY (seen); SENSITIVITY (duplicates: noBITDUP); SENSITIVITY (design BP); different read budget; paired, identical queries | NaN | NaN | 1780.0000 | NaN | -0.0271 | 0.0000 | -0.0274;-0.0262;-0.0271;-0.0276;-0.0275 |

## Report-rule outcomes

### 1.6-R01: the pre-declared Pareto question (C3 min(4, k_max), 5 reads, vs DOPTQ)

Evaluated over the written label-free family only (MAXMIN_R, MAXMIN_Z, BISECT_R, BISECT_Z, BISECT_RMID, HULL_BISECT, RANDOM; DOPTQ is the reference).  ORACLE_B1 [O] is a bound on its own row, never a member; LOOSE legality is a labelled sensitivity on its own row.  Per-policy rows: envelope_pareto_<design>.csv.

| design | variant | policy | role | pareto_dominates | dominating_policies | keeps_every_block_above_floor_policies | policies | label |
|---|---|---|---|---|---|---|---|---|
| BPG | all | ANSWER: written label-free family | answer (1.6-R01): family members | False |  | HULL_BISECT | MAXMIN_R,MAXMIN_Z,BISECT_R,BISECT_Z,BISECT_RMID,HULL_BISECT,RANDOM | EXPLORATORY (seen); different read budget; pre-declared Pareto question 1.6-R01 over the written label-free family (descriptive) |
| BPG | all | ANSWER: ORACLE_B1 [O] bound | answer: ORACLE [O] bound (never a member) | True | ORACLE_B1 | ORACLE_B1 | ORACLE_B1 | ORACLE [O]; different read budget; bound only; dominates by construction (the legal 5th metal minimising the cell's B1 SSE) |
| BPG | all | ANSWER: LOOSE legality | answer: SENSITIVITY (LOOSE legality; never a member) | False |  |  | MAXMIN_R_LOOSE,RANDOM_LOOSE | EXPLORATORY (seen); SENSITIVITY (LOOSE legality); different read budget; LOOSE-legality sensitivity of the Pareto question |
| BPG | noC5 | ANSWER: written label-free family | answer (1.6-R01): family members | False |  |  | MAXMIN_R,MAXMIN_Z,BISECT_R,BISECT_Z,BISECT_RMID,HULL_BISECT,RANDOM | EXPLORATORY (seen); SENSITIVITY (C5-BTBP out); different read budget; pre-declared Pareto question 1.6-R01 over the written label-free family (descriptive) |
| BPG | noC5 | ANSWER: ORACLE_B1 [O] bound | answer: ORACLE [O] bound (never a member) | True | ORACLE_B1 | ORACLE_B1 | ORACLE_B1 | ORACLE [O]; SENSITIVITY (C5-BTBP out); different read budget; bound only; dominates by construction (the legal 5th metal minimising the cell's B1 SSE) |
| BPG | noC5 | ANSWER: LOOSE legality | answer: SENSITIVITY (LOOSE legality; never a member) | False |  |  | MAXMIN_R_LOOSE,RANDOM_LOOSE | EXPLORATORY (seen); SENSITIVITY (LOOSE legality); SENSITIVITY (C5-BTBP out); different read budget; LOOSE-legality sensitivity of the Pareto question |
| BPG | noC5cell | ANSWER: written label-free family | answer (1.6-R01): family members | False |  |  | MAXMIN_R,MAXMIN_Z,BISECT_R,BISECT_Z,BISECT_RMID,HULL_BISECT,RANDOM | EXPLORATORY (seen); SENSITIVITY (cell c950dc2982e96b83 out); different read budget; pre-declared Pareto question 1.6-R01 over the written label-free family (descriptive) |
| BPG | noC5cell | ANSWER: ORACLE_B1 [O] bound | answer: ORACLE [O] bound (never a member) | True | ORACLE_B1 | ORACLE_B1 | ORACLE_B1 | ORACLE [O]; SENSITIVITY (cell c950dc2982e96b83 out); different read budget; bound only; dominates by construction (the legal 5th metal minimising the cell's B1 SSE) |
| BPG | noC5cell | ANSWER: LOOSE legality | answer: SENSITIVITY (LOOSE legality; never a member) | False |  |  | MAXMIN_R_LOOSE,RANDOM_LOOSE | EXPLORATORY (seen); SENSITIVITY (LOOSE legality); SENSITIVITY (cell c950dc2982e96b83 out); different read budget; LOOSE-legality sensitivity of the Pareto question |
| BP | all | ANSWER: written label-free family | answer (1.6-R01): family members | False |  |  | MAXMIN_R,MAXMIN_Z,BISECT_R,BISECT_Z,BISECT_RMID,HULL_BISECT,RANDOM | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget; pre-declared Pareto question 1.6-R01 over the written label-free family (descriptive) |
| BP | all | ANSWER: ORACLE_B1 [O] bound | answer: ORACLE [O] bound (never a member) | True | ORACLE_B1 | ORACLE_B1 | ORACLE_B1 | ORACLE [O]; SENSITIVITY (design BP); different read budget; bound only; dominates by construction (the legal 5th metal minimising the cell's B1 SSE) |
| BP | all | ANSWER: LOOSE legality | answer: SENSITIVITY (LOOSE legality; never a member) | False |  |  | MAXMIN_R_LOOSE,RANDOM_LOOSE | EXPLORATORY (seen); SENSITIVITY (LOOSE legality); SENSITIVITY (design BP); different read budget; LOOSE-legality sensitivity of the Pareto question |
| BP | noC5 | ANSWER: written label-free family | answer (1.6-R01): family members | False |  |  | MAXMIN_R,MAXMIN_Z,BISECT_R,BISECT_Z,BISECT_RMID,HULL_BISECT,RANDOM | EXPLORATORY (seen); SENSITIVITY (C5-BTBP out); SENSITIVITY (design BP); different read budget; pre-declared Pareto question 1.6-R01 over the written label-free family (descriptive) |
| BP | noC5 | ANSWER: ORACLE_B1 [O] bound | answer: ORACLE [O] bound (never a member) | True | ORACLE_B1 | ORACLE_B1 | ORACLE_B1 | ORACLE [O]; SENSITIVITY (C5-BTBP out); SENSITIVITY (design BP); different read budget; bound only; dominates by construction (the legal 5th metal minimising the cell's B1 SSE) |
| BP | noC5 | ANSWER: LOOSE legality | answer: SENSITIVITY (LOOSE legality; never a member) | False |  |  | MAXMIN_R_LOOSE,RANDOM_LOOSE | EXPLORATORY (seen); SENSITIVITY (LOOSE legality); SENSITIVITY (C5-BTBP out); SENSITIVITY (design BP); different read budget; LOOSE-legality sensitivity of the Pareto question |
| BP | noC5cell | ANSWER: written label-free family | answer (1.6-R01): family members | False |  |  | MAXMIN_R,MAXMIN_Z,BISECT_R,BISECT_Z,BISECT_RMID,HULL_BISECT,RANDOM | EXPLORATORY (seen); SENSITIVITY (cell c950dc2982e96b83 out); SENSITIVITY (design BP); different read budget; pre-declared Pareto question 1.6-R01 over the written label-free family (descriptive) |
| BP | noC5cell | ANSWER: ORACLE_B1 [O] bound | answer: ORACLE [O] bound (never a member) | True | ORACLE_B1 | ORACLE_B1 | ORACLE_B1 | ORACLE [O]; SENSITIVITY (cell c950dc2982e96b83 out); SENSITIVITY (design BP); different read budget; bound only; dominates by construction (the legal 5th metal minimising the cell's B1 SSE) |
| BP | noC5cell | ANSWER: LOOSE legality | answer: SENSITIVITY (LOOSE legality; never a member) | False |  |  | MAXMIN_R_LOOSE,RANDOM_LOOSE | EXPLORATORY (seen); SENSITIVITY (LOOSE legality); SENSITIVITY (cell c950dc2982e96b83 out); SENSITIVITY (design BP); different read budget; LOOSE-legality sensitivity of the Pareto question |

### 1.6-R02: density invariance at saturation (CONSISTENCY CHECK, not a kill)

| design | frame | R2 | R2_B0 | MSE | rows_per_seed | cells | clusters | level_note | label |
|---|---|---|---|---|---|---|---|---|---|
| BPG | k_max 3 at 4 reads | 0.6018 | 0.5640 | 0.0141 | 286.0000 | 51 | 4 | (E-A rows) stratum: a breakdown of E-A's 4-read rows, the reference of the other budgets; never an E-D level | EXPLORATORY (seen); CONSISTENCY CHECK |
| BPG | k_max 4 at 5 reads | 0.4440 | 0.4153 | 0.0251 | 111.0000 | 34 | 4 | NaN | EXPLORATORY (seen); CONSISTENCY CHECK; different read budget |
| BPG | k_max 5 at 6 reads | 0.4764 | 0.4883 | 0.0508 | 56.0000 | 11 | 2 | NaN | EXPLORATORY (seen); CONSISTENCY CHECK; different read budget |
| BPG | k_max 6 at 7 reads | 0.5558 | 0.5216 | 0.0223 | 1349.0000 | 133 | 10 | NaN | EXPLORATORY (seen); CONSISTENCY CHECK; different read budget |
| BP | k_max 3 at 4 reads | 0.6021 | 0.5637 | 0.0140 | 286.0000 | 51 | 4 | (E-A rows) stratum: a breakdown of E-A's 4-read rows, the reference of the other budgets; never an E-D level | EXPLORATORY (seen); SENSITIVITY (design BP); CONSISTENCY CHECK |
| BP | k_max 4 at 5 reads | 0.4466 | 0.4153 | 0.0250 | 111.0000 | 34 | 4 | NaN | EXPLORATORY (seen); SENSITIVITY (design BP); CONSISTENCY CHECK; different read budget |
| BP | k_max 5 at 6 reads | 0.4777 | 0.4881 | 0.0507 | 56.0000 | 11 | 2 | NaN | EXPLORATORY (seen); SENSITIVITY (design BP); CONSISTENCY CHECK; different read budget |
| BP | k_max 6 at 7 reads | 0.5592 | 0.5259 | 0.0221 | 1349.0000 | 133 | 10 | NaN | EXPLORATORY (seen); SENSITIVITY (design BP); CONSISTENCY CHECK; different read budget |

| design | variant | rule | strata | strata_low | spread | rule_fires | strata_values | label |
|---|---|---|---|---|---|---|---|---|
| BPG | all | fires if a stratum with >= 3 clusters is below 0.48 or the spread is >= 0.05 | 4 | k_max 4 at 5 reads | 0.1578 | True | k_max 3 at 4 reads: 0.6018 (4 clusters);k_max 4 at 5 reads: 0.4440 (4 clusters);k_max 5 at 6 reads: 0.4764 (2 clusters);k_max 6 at 7 reads: 0.5558 (10 clusters) | EXPLORATORY (seen); CONSISTENCY CHECK (the k_max-4 value was known before the rule was written; not a kill); strata at 5-7 reads = different read budget |
| BPG | noC5 | fires if a stratum with >= 3 clusters is below 0.48 or the spread is >= 0.05 | 4 | k_max 4 at 5 reads | 0.1629 | True | k_max 3 at 4 reads: 0.6018 (4 clusters);k_max 4 at 5 reads: 0.4440 (4 clusters);k_max 5 at 6 reads: 0.4764 (2 clusters);k_max 6 at 7 reads: 0.6070 (10 clusters) | EXPLORATORY (seen); SENSITIVITY (C5-BTBP out); CONSISTENCY CHECK (the k_max-4 value was known before the rule was written; not a kill); strata at 5-7 reads = different read budget |
| BP | all | fires if a stratum with >= 3 clusters is below 0.48 or the spread is >= 0.05 | 4 | k_max 4 at 5 reads | 0.1555 | True | k_max 3 at 4 reads: 0.6021 (4 clusters);k_max 4 at 5 reads: 0.4466 (4 clusters);k_max 5 at 6 reads: 0.4777 (2 clusters);k_max 6 at 7 reads: 0.5592 (10 clusters) | EXPLORATORY (seen); SENSITIVITY (design BP); CONSISTENCY CHECK (the k_max-4 value was known before the rule was written; not a kill); strata at 5-7 reads = different read budget |
| BP | noC5 | fires if a stratum with >= 3 clusters is below 0.48 or the spread is >= 0.05 | 4 | k_max 4 at 5 reads | 0.1649 | True | k_max 3 at 4 reads: 0.6021 (4 clusters);k_max 4 at 5 reads: 0.4466 (4 clusters);k_max 5 at 6 reads: 0.4777 (2 clusters);k_max 6 at 7 reads: 0.6115 (10 clusters) | EXPLORATORY (seen); SENSITIVITY (C5-BTBP out); SENSITIVITY (design BP); CONSISTENCY CHECK (the k_max-4 value was known before the rule was written; not a kill); strata at 5-7 reads = different read budget |

### M17 envelope width over the label-free family (5 reads)

| design | frame | variant | envelope | lowest | highest | envelope_with_RANDOM_MC_mean | seen_envelope | per_seed_span_min | per_seed_span_max | label |
|---|---|---|---|---|---|---|---|---|---|---|
| BPG | C3pop | all | 0.0091 | RANDOM | BISECT_RMID | 0.0091 | 0.0091 | 0.4786 | 0.5147 | EXPLORATORY (seen); different read budget |
| BPG | C4 | all | 0.0104 | RANDOM | BISECT_RMID | NaN | NaN | 0.4598 | 0.4998 | EXPLORATORY (seen); different read budget |
| BPG | C3pop | noC5 | 0.0103 | RANDOM | BISECT_R | 0.0104 | NaN | 0.5139 | 0.5478 | EXPLORATORY (seen); SENSITIVITY (C5-BTBP out); different read budget |
| BPG | C4 | noC5 | 0.0120 | RANDOM | BISECT_R | NaN | NaN | 0.4978 | 0.5352 | EXPLORATORY (seen); SENSITIVITY (C5-BTBP out); different read budget |
| BPG | C3pop | noC5cell | 0.0095 | RANDOM | BISECT_RMID | 0.0096 | NaN | 0.5100 | 0.5462 | EXPLORATORY (seen); SENSITIVITY (cell c950dc2982e96b83 out); different read budget |
| BPG | C4 | noC5cell | 0.0109 | RANDOM | BISECT_RMID | NaN | NaN | 0.4942 | 0.5347 | EXPLORATORY (seen); SENSITIVITY (cell c950dc2982e96b83 out); different read budget |
| BP | C3pop | all | 0.0095 | RANDOM | BISECT_RMID | 0.0096 | 0.0096 | 0.4792 | 0.5211 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BP | C4 | all | 0.0108 | RANDOM | BISECT_RMID | NaN | NaN | 0.4599 | 0.5065 | EXPLORATORY (seen); SENSITIVITY (design BP); different read budget |
| BP | C3pop | noC5 | 0.0111 | RANDOM | BISECT_R | 0.0113 | NaN | 0.5146 | 0.5573 | EXPLORATORY (seen); SENSITIVITY (C5-BTBP out); SENSITIVITY (design BP); different read budget |
| BP | C4 | noC5 | 0.0130 | RANDOM | BISECT_R | NaN | NaN | 0.4980 | 0.5465 | EXPLORATORY (seen); SENSITIVITY (C5-BTBP out); SENSITIVITY (design BP); different read budget |
| BP | C3pop | noC5cell | 0.0099 | RANDOM | BISECT_RMID | 0.0100 | NaN | 0.5106 | 0.5532 | EXPLORATORY (seen); SENSITIVITY (cell c950dc2982e96b83 out); SENSITIVITY (design BP); different read budget |
| BP | C4 | noC5cell | 0.0114 | RANDOM | BISECT_RMID | NaN | NaN | 0.4944 | 0.5422 | EXPLORATORY (seen); SENSITIVITY (cell c950dc2982e96b83 out); SENSITIVITY (design BP); different read budget |

Note: RANDOM enters as its exact expectation (uniform over the legal set, independent across cells; ITEM_SPEC); the seen value [P3] is quoted to four decimals; with the RANDOM Monte Carlo mean in place of the exact expectation the width is envelope_with_RANDOM_MC_mean, so a last-digit difference from the seen value is expected (not a reproduction target) (C3 min(4, k_max), all rows: BPG exact 0.009144, with the Monte Carlo mean 0.009076, seen 0.0091; BP exact 0.009544, with the Monte Carlo mean 0.009561, seen 0.0096).

Seal: b96c3bef9828
