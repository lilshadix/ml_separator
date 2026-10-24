# Item 1.0 EA_CHECK (E-A, DIAGNOSTIC, seen; BP): per-item table

**Label: EXPLORATORY (seen); DIAGNOSTIC (E-A contrasts after M34's kill); BP and C5-out rows SENSITIVITY; HIST_D7 not fold-independent; seal b96c3bef9828**

| field | value |
|---|---|
| Item id and estimand | 1.0 EA_CHECK (BP): M35 positive-control and six-design check; estimand E-A (DIAGNOSTIC, seen) |
| Population (cells, publications, units, rows per seed) | cells: 229; publications: 39; unit: cluster_cpg; units: 13; rows_per_seed: 1802.0000; seeds: 5 |
| Design | BP (SENSITIVITY; same test folds as BPG, which decides); six-design table A, B, BR, BQ, BP, BPG for the training-free arms B0 and HIST_D7 only (sixdesign.csv) |
| Read budget | four reads (code budget k = 3), DOPT_Q nested read sets of each design's fold covariance |
| Arms on identical rows | B0, B1, HIST_D7; contrasts B1 - B0 (positive control, outside any family) and HIST_D7 - B1 under the sealed M35 gate; HIST_D7 - B0 on the six designs |
| R², MSE, relative SSE vs B1 | see 'R², MSE, relative SSE vs B1' below |
| Per-seed values | see 'Per-seed values' below |
| Unit W/L/T and the descriptive tests | see 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | see 'LOPO and top-3 share' below |
| C5-BTBP in / out | see 'C5-BTBP in / out' below |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | cache poisoning 100 folds identical: True; isolation 100/100 (design-aware); intervention 100/100 folds, max change 0 over 72080 predictions; block['Y'] peek caught in 100/100; assert_test_isolation on the 50 BP / BPG folds (repro); fixed denominators across the six designs |
| Compute | repro + rows 10s for A/B/BR/BQ, tables 19s, total 32s; python 3.14.5, numpy 2.5.3, pandas 3.0.5; caches: manifests/ea_check_caches.json |
| Label | EXPLORATORY (seen); DIAGNOSTIC (E-A contrasts after M34's kill); BP and C5-out rows SENSITIVITY; HIST_D7 not fold-independent; seal b96c3bef9828 |
| What changed scientifically | Nothing on E-A (no E-A slot): the sealed M35 gate passes its own positive control B1 - B0 on BPG (relative SSE +0.0586, SF_POOL p 0.00317, guard +0.00501), so M35's kill does not fire and the convention is kept (DIAGNOSTIC); BP sensitivity (same test folds): B1 - B0 passes the gate (relative SSE +0.0585, SF_POOL p 0.00269, guard +0.00526), HIST_D7 - B1 fails the gate (relative SSE -0.0025, SF_POOL p 0.69604, guard -0.00026). B1's own level 0.452 stands as E-A's reference; every number is a seen diagnostic. |
| Next registered step | M35 stays the E-A decision convention (DIAGNOSTIC); the next registered step is item 1.1 (E-CT core, results/ECT/); no E-A slot is registered (section 4.5). |

## R², MSE, relative SSE vs B1

| arm | R2 | MSE | relSSE_vs_B1 | MAE_balanced | R2_per_seed | design | stratum | label |
|---|---|---|---|---|---|---|---|---|
| B0 | 0.4179 | 0.0287 | -0.0622 | 0.1059 | 0.4154;0.4117;0.4233;0.4232;0.4158 | BP | C5_in | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| B1 | 0.4519 | 0.0270 | 0.0000 | 0.1006 | 0.4533;0.4397;0.4589;0.4581;0.4497 | BP | C5_in | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| HIST_D7 | 0.4506 | 0.0271 | -0.0025 | 0.1009 | 0.4489;0.4441;0.4563;0.4560;0.4477 | BP | C5_in | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| B0 | 0.4589 | 0.0238 | -0.0531 | 0.1051 | 0.4583;0.4538;0.4614;0.4623;0.4587 | BP | C5_out | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| B1 | 0.4862 | 0.0226 | 0.0000 | 0.0999 | 0.4893;0.4752;0.4891;0.4906;0.4866 | BP | C5_out | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| HIST_D7 | 0.4869 | 0.0226 | 0.0013 | 0.1001 | 0.4869;0.4810;0.4902;0.4908;0.4854 | BP | C5_out | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |

## Per-seed values

| design | stratum | contrast | split_seed | relSSE | R2_cand | R2_ref | dR2 | dMAE_bal | label |
|---|---|---|---|---|---|---|---|---|---|
| BP | C5_in | B1 - B0 | 104729 | 0.0648 | 0.4533 | 0.4154 | 0.0379 | 0.0064 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | B1 - B0 | 130363 | 0.0475 | 0.4397 | 0.4117 | 0.0280 | 0.0045 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | B1 - B0 | 155921 | 0.0617 | 0.4589 | 0.4233 | 0.0356 | 0.0054 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | B1 - B0 | 196613 | 0.0605 | 0.4581 | 0.4232 | 0.0349 | 0.0052 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | B1 - B0 | 262147 | 0.0581 | 0.4497 | 0.4158 | 0.0339 | 0.0048 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | HIST_D7 - B1 | 104729 | -0.0081 | 0.4489 | 0.4533 | -0.0044 | -0.0009 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | HIST_D7 - B1 | 130363 | 0.0079 | 0.4441 | 0.4397 | 0.0044 | 0.0006 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | HIST_D7 - B1 | 155921 | -0.0048 | 0.4563 | 0.4589 | -0.0026 | -0.0007 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | HIST_D7 - B1 | 196613 | -0.0039 | 0.4560 | 0.4581 | -0.0021 | 0.0000 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | HIST_D7 - B1 | 262147 | -0.0035 | 0.4477 | 0.4497 | -0.0020 | -0.0003 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |

## Unit W/L/T and the descriptive tests

| design | contrast | G | W/L/T | SF_POOL_p | SIGN_p | SF_LOG_p | IM_REL_p | gate_pass | label |
|---|---|---|---|---|---|---|---|---|---|
| BP | B1 - B0 | 13 | 11/1/1 | 0.0027 | 0.0032 | 0.0071 | 0.0110 | True | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | HIST_D7 - B1 | 13 | 3/9/1 | 0.6960 | 0.9807 | 0.9404 | 0.9423 | False | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |

## LOPO and top-3 share

| design | stratum | contrast | point | lo | hi | min | min_by | max | max_by | P | top3_unit_share | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BP | C5_in | B1 - B0 | 0.0585 | 0.0228 | 0.0942 | 0.0504 | pub_6a0ef8528c | 0.0673 | pub_1605e436d7 | 39 | 0.7941 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_out | B1 - B0 | 0.0504 | 0.0091 | 0.0918 | 0.0398 | pub_8415538291 | 0.0616 | pub_1605e436d7 | 38 | 0.7354 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | HIST_D7 - B1 | -0.0025 | -0.0138 | 0.0088 | -0.0053 | pub_1605e436d7 | 0.0013 | pub_6a0ef8528c | 39 | NaN | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_out | HIST_D7 - B1 | 0.0013 | -0.0097 | 0.0123 | -0.0022 | pub_1605e436d7 | 0.0033 | pub_f8a621ede4 | 38 | 4.4691 | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |

## C5-BTBP in / out

| design | stratum | contrast | rows_per_seed | relSSE | dR2 | SF_POOL_p | dMAE_bal | gate_pass | label |
|---|---|---|---|---|---|---|---|---|---|
| BP | C5_in | B1 - B0 | 1802.0000 | 0.0585 | 0.0340 | 0.0027 | 0.0053 | True | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_out | B1 - B0 | 1654.0000 | 0.0504 | 0.0273 | 0.0032 | 0.0052 | True | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_cell_out | B1 - B0 | 1792.0000 | 0.0627 | 0.0347 | 0.0027 | 0.0053 | True | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_in | HIST_D7 - B1 | 1802.0000 | -0.0025 | -0.0013 | 0.6960 | -0.0003 | False | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_out | HIST_D7 - B1 | 1654.0000 | 0.0013 | 0.0007 | 0.4482 | -0.0003 | False | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |
| BP | C5_cell_out | HIST_D7 - B1 | 1792.0000 | -0.0022 | -0.0011 | 0.6611 | -0.0003 | False | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY |

## Duplicate sensitivities

| design | variant | contrast | cells_kept | rows_per_seed | G | relSSE | SF_POOL_p | dR2 | dMAE_bal | gate_pass | label | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BP | noV2 | B1 - B0 | 209 | 1636.0000 | 13 | 0.0643 | 0.0027 | 0.0376 | 0.0053 | True | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY; SENSITIVITY (duplicate sensitivity) |  |
| BP | noBITDUP | B1 - B0 | 223 | 1780.0000 | 13 | 0.0581 | 0.0027 | 0.0338 | 0.0052 | True | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY; SENSITIVITY (duplicate sensitivity) |  |
| BP | V2-3 | B1 - B0 | 11 | 88.0000 | 1 | -0.0860 | NaN | -0.0476 | -0.0058 | not computable (G = 1) | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY; SENSITIVITY (duplicate sensitivity, descriptive) | G = 1 cluster_cpg unit: SF_POOL, W/L/T and the gate not computable; relative SSE, R2 of both arms, dR2 and dMAE_bal descriptive |
| BP | without_V2-3 | B1 - B0 | 218 | 1714.0000 | 13 | 0.0669 | 0.0027 | 0.0395 | 0.0053 | True | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY; SENSITIVITY (duplicate sensitivity) |  |
| BP | noV2 | HIST_D7 - B1 | 209 | 1636.0000 | 13 | -0.0043 | 0.7607 | -0.0023 | -0.0003 | False | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY; SENSITIVITY (duplicate sensitivity) |  |
| BP | noBITDUP | HIST_D7 - B1 | 223 | 1780.0000 | 13 | -0.0024 | 0.6890 | -0.0013 | -0.0002 | False | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY; SENSITIVITY (duplicate sensitivity) |  |
| BP | V2-3 | HIST_D7 - B1 | 11 | 88.0000 | 1 | 0.0295 | NaN | 0.0180 | 0.0024 | not computable (G = 1) | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY; SENSITIVITY (duplicate sensitivity, descriptive) | G = 1 cluster_cpg unit: SF_POOL, W/L/T and the gate not computable; relative SSE, R2 of both arms, dR2 and dMAE_bal descriptive |
| BP | without_V2-3 | HIST_D7 - B1 | 218 | 1714.0000 | 13 | -0.0047 | 0.7979 | -0.0025 | -0.0003 | False | EXPLORATORY (seen); DIAGNOSTIC; SENSITIVITY; SENSITIVITY (duplicate sensitivity) |  |
