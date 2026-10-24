# 1.5 E-DEC (BPG): per-item table

**Label: EXPLORATORY (seen); E-DEC: a decision estimand; never an R2 gain, never next to E-A's 0.452; seal b96c3bef9828**

| field | value |
|---|---|
| Item id and estimand | 1.5 E-DEC selective direction calls (M14) with the E-REC own-prior row and the E-RANK table (BPG) |
| Population (cells, publications, units, rows per seed) | E-A rows: cells: 229; publications: 39; unit: cluster_cpg; units: 13; rows_per_seed: 1802.0000; seeds: 5; E-REC PRIMARY_OWN: stratum: PRIMARY_OWN; cells: 112; publications: 16; units_pubchem: 18; clusters: 5; rows_per_seed: 790.0000; rows_dropped_incomplete: 0; chpt_pairtype_fallback_share: 0.2443 |
| Design | BPG (decides) |
| Read budget | 4 reads (E-A rows, DOPT_Q k = 3); E-REC: COPY 0, SH_own@2 / D7@2 2, SH_own@4 / D7@4 / B1@4 4 reads |
| Arms on identical rows | selective: B1, B0, HBK@ML (primary set), RIDGE2, LINE, G14@ZERO, HIST_D7 (DIAGNOSTIC); net benefit: B1, B0, HBK@ML, RIDGE2, Platt-LINE; E-RANK: + RANDOM, AMP_ORACLE [O]; E-REC: COPY_own, SH_own@2, SH_own@4, CHPT_MAJ, G14, D7@2, D7@4, B1@4 |
| R², MSE, relative SSE vs B1 | not applicable: E-DEC is a decision estimand (coverage, per-call false-sign rate, AURC, net benefit); never an R2 gain (primary statistics below) |
| Per-seed values | see 'Per-seed values' below |
| Unit W/L/T and the descriptive tests | see 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | see 'LOPO and top-3 share' below |
| C5-BTBP in / out | see 'C5-BTBP in / out' below |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | fit poisoning passed: 145205 fitted values over 13 clusters x 5 seeds bit-identical when the held-out cluster's labels are replaced; leaky positive control caught 13/13; scope: calibration labels only (the prediction models are not refitted; B0 is training-free); E-REC identities {'rows': 9010, 'b1_r4_vs_ref_maxabs': 8.881784197001252e-16, 'd7_r4_vs_ref_hist_d7_maxabs': 1.3461454173580023e-15, 'y_vs_ref_maxabs': 0.0, 'beta_min_abs': 0.029905959094989075, 'source': 'D:\\ml_separator_gh\\generations\\gen21\\round5_20261024\\results\\ECT\\rows_BPG.parquet'} |
| Compute | 275 s |
| Label | EXPLORATORY (seen); E-DEC: a decision estimand; never an R2 gain, never next to E-A's 0.452; seal b96c3bef9828 |
| What changed scientifically | EXPLORATORY (seen): under the valid primary calibration (LTT, alpha 0.10, hull rule, global thresholds) B1 signs 0.0049 of the adjacent pairs from 4 reads with C5-BTBP in and 0.1299 with it out (B0 0.0044 / 0.0686), while CRC and the plug-in at alpha 0.10 (hull rule, global) call 0.8405 / 0.8405 with C5-BTBP in (CRC controls the per-pair rate, realised 0.0459; the plug-in has no guarantee); the realised LTT per-call false-sign rate of B1 is 0.1942 as the registered seed mean (4/44 = 0.0909 pooled, 25 distinct queries, 2 with a false sign; calls in 4 of 5 seeds) against alpha 0.10; the seed mean rests on few calls (per-seed calls 23;15;3;0;3, below n_min 22 in 4 of 5 seeds, seeds without calls dropped).  Nothing here is a test or a claim; the seen valid fixed-sequence coverage 0.149 [IR M14] came from another implementation and is not a target. |
| Next registered step | New-source direction-call evaluation is data-gated (>= 5 sources and >= 100 calls, IR M14); the registry continues with item 1.6 (E-D). |

## Per-seed values

| arm | coverage_per_seed | calls_per_seed | fs_rate_per_seed | fs_rate (seed mean; registered) | fs_rate pooled | false_signs_pooled | calls_pooled | calls_distinct | false_signs_distinct | seeds_with_calls | few_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 0.0128;0.0083;0.0017;0.0000;0.0017 | 23;15;3;0;3 | 0.0435;0.0667;0.3333;NaN;0.3333 | 0.1942 | 0.0909 | 4 | 44 | 25 | 2 | 4 | seed mean rests on few calls (per-seed calls 23;15;3;0;3, below n_min 22 in 4 of 5 seeds, seeds without calls dropped) |
| B0 | 0.0122;0.0083;0.0000;0.0000;0.0017 | 22;15;0;0;3 | 0.0455;0.0667;NaN;NaN;0.3333 | 0.1485 | 0.0750 | 3 | 40 | 24 | 2 | 3 | seed mean rests on few calls (per-seed calls 22;15;0;0;3, below n_min 22 in 4 of 5 seeds, seeds without calls dropped) |
| HBK@ML | 0.0910;0.0072;0.0017;0.0100;0.0000 | 164;13;3;18;0 | 0.1037;0.0769;0.0000;0.0556;NaN | 0.0590 | 0.0960 | 19 | 198 | 168 | 17 | 4 | seed mean rests on few calls (per-seed calls 164;13;3;18;0, below n_min 22 in 4 of 5 seeds, seeds without calls dropped) |

## Unit W/L/T and the descriptive tests

| contrast | G | W/L/T | pooled_dcoverage | SF_POOL_p | SF_exact | SIGN_p | top3_share |
|---|---|---|---|---|---|---|---|
| B1 - B0 (coverage contribution, primary configuration) | 13 | 2/1/10 | 0.0004 | 0.3750 | True | 0.5000 | 1.2500 |
| B1 - HBK@ML (coverage contribution, primary configuration) | 13 | 2/3/8 | -0.0171 | 0.7188 | True | 0.8125 | NaN |

## LOPO and top-3 share

| statistic | point | LOPO_min (spread) | LOPO_min_by | LOPO_max (spread) | LOPO_max_by | jackknife SE | undefined leave-outs | P | jackknife t(P-1) interval (descriptive, anti-conservative) lo | jackknife t(P-1) interval (descriptive, anti-conservative) hi |
|---|---|---|---|---|---|---|---|---|---|---|
| AURC_B1 | 0.0402 | 0.0326 | pub_126bbe7976 | 0.0447 | pub_19783c039b | 0.0105 | 0 | 39 | 0.0190 | 0.0614 |
| AURC_B0 | 0.0439 | 0.0363 | pub_126bbe7976 | 0.0490 | pub_19783c039b | 0.0116 | 0 | 39 | 0.0204 | 0.0674 |
| dAURC_B0_minus_B1 | 0.0037 | 0.0014 | pub_6a0ef8528c | 0.0049 | pub_1605e436d7 | 0.0033 | 0 | 39 | -0.0029 | 0.0103 |
| cov_B1 | 0.0049 | 0.0033 | pub_b0b3296ec5 | 0.1299 | pub_6a0ef8528c | 0.1266 | 0 | 39 | -0.2514 | 0.2612 |
| cov_B0 | 0.0044 | 0.0004 | pub_1845fd6492 | 0.0686 | pub_6a0ef8528c | 0.0734 | 0 | 39 | -0.1442 | 0.1530 |
| dcov_B1_minus_B0 | 0.0004 | 0.0002 | pub_af7a1cd919 | 0.0613 | pub_6a0ef8528c | 0.0592 | 0 | 39 | -0.1194 | 0.1203 |
| fs_B1 | 0.1942 | 0.0405 | pub_a49146c8fd | 0.2000 | pub_723ceb6c2c | 0.1749 | 0 | 39 | -0.1598 | 0.5482 |
| fs_B0 | 0.1485 | 0.0476 | pub_a49146c8fd | 0.1894 | pub_b8175dc3a1 | 0.1228 | 0 | 39 | -0.1001 | 0.3970 |
| NB30_B1 | 0.0566 | 0.0449 | pub_126bbe7976 | 0.0643 | pub_5a68dc5665 | 0.0159 | 0 | 39 | 0.0243 | 0.0888 |
| NB30_B0 | 0.0425 | 0.0322 | pub_126bbe7976 | 0.0483 | pub_5a68dc5665 | 0.0136 | 0 | 39 | 0.0149 | 0.0700 |
| dNB30_B1_minus_B0 | 0.0141 | 0.0122 | pub_b0b3296ec5 | 0.0161 | pub_5a68dc5665 | 0.0047 | 0 | 39 | 0.0045 | 0.0237 |

## C5-BTBP in / out

| statistic | point | C5_out | C5_cell_out | few_calls |
|---|---|---|---|---|
| AURC_B1 | 0.0402 | 0.0372 | 0.0399 |  |
| AURC_B0 | 0.0439 | 0.0386 | 0.0435 |  |
| dAURC_B0_minus_B1 | 0.0037 | 0.0014 | 0.0036 |  |
| cov_B1 | 0.0049 | 0.1299 | 0.0049 |  |
| cov_B0 | 0.0044 | 0.0686 | 0.0045 |  |
| dcov_B1_minus_B0 | 0.0004 | 0.0613 | 0.0004 |  |
| fs_B1 | 0.1942 | 0.1103 | 0.1942 | seed mean rests on few calls (per-seed calls below n_min 22) in: point [23;15;3;0;3], C5_out [442;484;3;6;139], C5_cell_out [23;15;3;0;3], noV2 [23;15;3;0;3], noBITDUP [23;15;3;0;3], V2-3_stratum [0;0;0;0;0], without_V2-3 [23;15;3;0;3] |
| fs_B0 | 0.1485 | 0.1099 | 0.1485 | seed mean rests on few calls (per-seed calls below n_min 22) in: point [22;15;0;0;3], C5_out [359;61;3;4;140], C5_cell_out [22;15;0;0;3], noV2 [4;0;0;0;3], noBITDUP [22;15;0;0;3], V2-3_stratum [0;0;0;0;0], without_V2-3 [4;0;0;0;3] |
| NB30_B1 | 0.0566 | 0.0541 | 0.0573 |  |
| NB30_B0 | 0.0425 | 0.0418 | 0.0432 |  |
| dNB30_B1_minus_B0 | 0.0141 | 0.0123 | 0.0141 |  |
| fs_pooled_B1 | 0.0909 | 0.0624 | 0.0909 |  |
| false_signs_pooled_B1 | 4 | 67 | 4 |  |
| calls_pooled_B1 | 44 | 1074 | 44 |  |
| calls_distinct_B1 | 25 | 518 | 25 |  |
| false_signs_distinct_B1 | 2 | 34 | 2 |  |
| seeds_with_calls_B1 | 4 | 5 | 4 |  |
| fs_pooled_B0 | 0.0750 | 0.0741 | 0.0750 |  |
| false_signs_pooled_B0 | 3 | 42 | 3 |  |
| calls_pooled_B0 | 40 | 567 | 40 |  |
| calls_distinct_B0 | 24 | 383 | 24 |  |
| false_signs_distinct_B0 | 2 | 26 | 2 |  |
| seeds_with_calls_B0 | 3 | 5 | 3 |  |

## Duplicate sensitivities

| statistic | point | noV2 | noBITDUP | V2-3_stratum | without_V2-3 |
|---|---|---|---|---|---|
| AURC_B1 | 0.0402 | 0.0396 | 0.0404 | 0.0778 | 0.0390 |
| AURC_B0 | 0.0439 | 0.0439 | 0.0441 | 0.0512 | 0.0437 |
| dAURC_B0_minus_B1 | 0.0037 | 0.0042 | 0.0036 | -0.0266 | 0.0047 |
| cov_B1 | 0.0049 | 0.0054 | 0.0049 | 0.0000 | 0.0051 |
| cov_B0 | 0.0044 | 0.0009 | 0.0045 | 0.0000 | 0.0008 |
| dcov_B1_minus_B0 | 0.0004 | 0.0045 | 0.0004 | 0.0000 | 0.0043 |
| fs_B1 | 0.1942 | 0.1942 | 0.1942 | NaN | 0.1942 |
| fs_B0 | 0.1485 | 0.1667 | 0.1485 | NaN | 0.1667 |
| NB30_B1 | 0.0566 | 0.0606 | 0.0566 | -0.1526 | 0.0585 |
| NB30_B0 | 0.0425 | 0.0452 | 0.0427 | -0.1526 | 0.0435 |
| dNB30_B1_minus_B0 | 0.0141 | 0.0154 | 0.0139 | 0.0000 | 0.0150 |
| fs_pooled_B1 | 0.0909 | 0.0909 | 0.0909 | NaN | 0.0909 |
| false_signs_pooled_B1 | 4 | 4 | 4 | 0 | 4 |
| calls_pooled_B1 | 44 | 44 | 44 | 0 | 44 |
| calls_distinct_B1 | 25 | 25 | 25 | 0 | 25 |
| false_signs_distinct_B1 | 2 | 2 | 2 | 0 | 2 |
| seeds_with_calls_B1 | 4 | 4 | 4 | 0 | 4 |
| fs_pooled_B0 | 0.0750 | 0.1429 | 0.0750 | NaN | 0.1429 |
| false_signs_pooled_B0 | 3 | 1 | 3 | 0 | 1 |
| calls_pooled_B0 | 40 | 7 | 40 | 0 | 7 |
| calls_distinct_B0 | 24 | 6 | 24 | 0 | 6 |
| false_signs_distinct_B0 | 2 | 1 | 2 | 0 | 1 |
| seeds_with_calls_B0 | 3 | 2 | 3 | 0 | 2 |
