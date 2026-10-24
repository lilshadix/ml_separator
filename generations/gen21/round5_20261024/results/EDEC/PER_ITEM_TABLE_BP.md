# 1.5 E-DEC (BP): per-item table

**Label: EXPLORATORY (seen); SENSITIVITY (BP; BPG decides); E-DEC: a decision estimand; never an R2 gain, never next to E-A's 0.452; seal b96c3bef9828**

| field | value |
|---|---|
| Item id and estimand | 1.5 E-DEC selective direction calls (M14) with the E-REC own-prior row and the E-RANK table (BP) |
| Population (cells, publications, units, rows per seed) | E-A rows: cells: 229; publications: 39; unit: cluster_cpg; units: 13; rows_per_seed: 1802.0000; seeds: 5; E-REC PRIMARY_OWN: stratum: PRIMARY_OWN; cells: 112; publications: 16; units_pubchem: 18; clusters: 5; rows_per_seed: 790.0000; rows_dropped_incomplete: 0; chpt_pairtype_fallback_share: 0.2443 |
| Design | BP (SENSITIVITY) |
| Read budget | 4 reads (E-A rows, DOPT_Q k = 3); E-REC: COPY 0, SH_own@2 / D7@2 2, SH_own@4 / D7@4 / B1@4 4 reads |
| Arms on identical rows | selective: B1, B0, HBK@ML (primary set), RIDGE2, LINE, G14@ZERO, HIST_D7 (DIAGNOSTIC); net benefit: B1, B0, HBK@ML, RIDGE2, Platt-LINE; E-RANK: + RANDOM, AMP_ORACLE [O]; E-REC: COPY_own, SH_own@2, SH_own@4, CHPT_MAJ, G14, D7@2, D7@4, B1@4 |
| R², MSE, relative SSE vs B1 | not applicable: E-DEC is a decision estimand (coverage, per-call false-sign rate, AURC, net benefit); never an R2 gain (primary statistics below) |
| Per-seed values | see 'Per-seed values' below |
| Unit W/L/T and the descriptive tests | see 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | see 'LOPO and top-3 share' below |
| C5-BTBP in / out | see 'C5-BTBP in / out' below |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | fit poisoning passed: 145205 fitted values over 13 clusters x 5 seeds bit-identical when the held-out cluster's labels are replaced; leaky positive control caught 13/13; scope: calibration labels only (the prediction models are not refitted; B0 is training-free); E-REC identities {'rows': 9010, 'b1_r4_vs_ref_maxabs': 8.881784197001252e-16, 'd7_r4_vs_ref_hist_d7_maxabs': 1.3461454173580023e-15, 'y_vs_ref_maxabs': 0.0, 'beta_min_abs': 0.07478326586966128, 'source': 'D:\\ml_separator_gh\\generations\\gen21\\round5_20261024\\results\\ECT\\rows_BP.parquet'} |
| Compute | 275 s |
| Label | EXPLORATORY (seen); SENSITIVITY (BP; BPG decides); E-DEC: a decision estimand; never an R2 gain, never next to E-A's 0.452; seal b96c3bef9828 |
| What changed scientifically | EXPLORATORY (seen): under the valid primary calibration (LTT, alpha 0.10, hull rule, global thresholds) B1 signs 0.0349 of the adjacent pairs from 4 reads with C5-BTBP in and 0.1694 with it out (B0 0.0178 / 0.1116), while CRC and the plug-in at alpha 0.10 (hull rule, global) call 0.8374 / 0.8374 with C5-BTBP in (CRC controls the per-pair rate, realised 0.0461; the plug-in has no guarantee); the realised LTT per-call false-sign rate of B1 is 0.1586 as the registered seed mean (36/314 = 0.1146 pooled, 176 distinct queries, 23 with a false sign; calls in 4 of 5 seeds) against alpha 0.10; the seed mean rests on few calls (per-seed calls 169;15;127;0;3, below n_min 22 in 3 of 5 seeds, seeds without calls dropped).  Nothing here is a test or a claim; the seen valid fixed-sequence coverage 0.149 [IR M14] came from another implementation and is not a target. |
| Next registered step | New-source direction-call evaluation is data-gated (>= 5 sources and >= 100 calls, IR M14); the registry continues with item 1.6 (E-D). |

## Per-seed values

| arm | coverage_per_seed | calls_per_seed | fs_rate_per_seed | fs_rate (seed mean; registered) | fs_rate pooled | false_signs_pooled | calls_pooled | calls_distinct | false_signs_distinct | seeds_with_calls | few_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | 0.0938;0.0083;0.0705;0.0000;0.0017 | 169;15;127;0;3 | 0.1006;0.0667;0.1339;NaN;0.3333 | 0.1586 | 0.1146 | 36 | 314 | 176 | 23 | 4 | seed mean rests on few calls (per-seed calls 169;15;127;0;3, below n_min 22 in 3 of 5 seeds, seeds without calls dropped) |
| B0 | 0.0122;0.0083;0.0683;0.0000;0.0000 | 22;15;123;0;0 | 0.0455;0.0667;0.1382;NaN;NaN | 0.0834 | 0.1187 | 19 | 160 | 123 | 17 | 3 | seed mean rests on few calls (per-seed calls 22;15;123;0;0, below n_min 22 in 3 of 5 seeds, seeds without calls dropped) |
| HBK@ML | 0.0910;0.0072;0.0694;0.0100;0.0000 | 164;13;125;18;0 | 0.1037;0.0769;0.1600;0.0556;NaN | 0.0990 | 0.1219 | 39 | 320 | 171 | 24 | 4 | seed mean rests on few calls (per-seed calls 164;13;125;18;0, below n_min 22 in 3 of 5 seeds, seeds without calls dropped) |

## Unit W/L/T and the descriptive tests

| contrast | G | W/L/T | pooled_dcoverage | SF_POOL_p | SF_exact | SIGN_p | top3_share |
|---|---|---|---|---|---|---|---|
| B1 - B0 (coverage contribution, primary configuration) | 13 | 3/0/10 | 0.0171 | 0.1250 | True | 0.1250 | 1.0000 |
| B1 - HBK@ML (coverage contribution, primary configuration) | 13 | 3/1/9 | -0.0007 | 0.5625 | True | 0.3125 | NaN |

## LOPO and top-3 share

| statistic | point | LOPO_min (spread) | LOPO_min_by | LOPO_max (spread) | LOPO_max_by | jackknife SE | undefined leave-outs | P | jackknife t(P-1) interval (descriptive, anti-conservative) lo | jackknife t(P-1) interval (descriptive, anti-conservative) hi |
|---|---|---|---|---|---|---|---|---|---|---|
| AURC_B1 | 0.0400 | 0.0322 | pub_126bbe7976 | 0.0445 | pub_19783c039b | 0.0105 | 0 | 39 | 0.0187 | 0.0613 |
| AURC_B0 | 0.0441 | 0.0365 | pub_126bbe7976 | 0.0492 | pub_19783c039b | 0.0116 | 0 | 39 | 0.0205 | 0.0676 |
| dAURC_B0_minus_B1 | 0.0041 | 0.0018 | pub_6a0ef8528c | 0.0050 | pub_b0b3296ec5 | 0.0032 | 0 | 39 | -0.0024 | 0.0105 |
| cov_B1 | 0.0349 | 0.0077 | pub_b0b3296ec5 | 0.1694 | pub_6a0ef8528c | 0.1464 | 0 | 39 | -0.2615 | 0.3312 |
| cov_B0 | 0.0178 | 0.0022 | pub_b0b3296ec5 | 0.1116 | pub_6a0ef8528c | 0.1004 | 0 | 39 | -0.1855 | 0.2210 |
| dcov_B1_minus_B0 | 0.0171 | 0.0006 | pub_126bbe7976 | 0.0578 | pub_6a0ef8528c | 0.0626 | 0 | 39 | -0.1095 | 0.1437 |
| fs_B1 | 0.1586 | 0.0500 | pub_b0b3296ec5 | 0.1586 | pub_0fd6ab4e40 | 0.1934 | 0 | 39 | -0.2329 | 0.5501 |
| fs_B0 | 0.0834 | 0.0294 | pub_b0b3296ec5 | 0.1076 | pub_126bbe7976 | 0.0668 | 0 | 39 | -0.0517 | 0.2186 |
| NB30_B1 | 0.0556 | 0.0440 | pub_126bbe7976 | 0.0634 | pub_5a68dc5665 | 0.0161 | 0 | 39 | 0.0231 | 0.0881 |
| NB30_B0 | 0.0428 | 0.0321 | pub_126bbe7976 | 0.0489 | pub_5a68dc5665 | 0.0141 | 0 | 39 | 0.0142 | 0.0713 |
| dNB30_B1_minus_B0 | 0.0128 | 0.0115 | pub_f8a621ede4 | 0.0151 | pub_4cba27f5f3 | 0.0041 | 0 | 39 | 0.0046 | 0.0211 |

## C5-BTBP in / out

| statistic | point | C5_out | C5_cell_out | few_calls |
|---|---|---|---|---|
| AURC_B1 | 0.0400 | 0.0369 | 0.0396 |  |
| AURC_B0 | 0.0441 | 0.0388 | 0.0436 |  |
| dAURC_B0_minus_B1 | 0.0041 | 0.0018 | 0.0040 |  |
| cov_B1 | 0.0349 | 0.1694 | 0.0335 |  |
| cov_B0 | 0.0178 | 0.1116 | 0.0174 |  |
| dcov_B1_minus_B0 | 0.0171 | 0.0578 | 0.0161 |  |
| fs_B1 | 0.1586 | 0.0610 | 0.1580 | seed mean rests on few calls (per-seed calls below n_min 22) in: point [169;15;127;0;3], C5_out [459;484;445;6;7], C5_cell_out [162;15;120;0;3], noV2 [23;15;127;0;0], noBITDUP [169;15;127;0;3], V2-3_stratum [0;0;0;0;0], without_V2-3 [23;15;127;0;0] |
| fs_B0 | 0.0834 | 0.0694 | 0.0822 | seed mean rests on few calls (per-seed calls below n_min 22) in: point [22;15;123;0;0], C5_out [433;61;419;4;6], C5_cell_out [22;15;119;0;0], noV2 [22;0;123;0;0], noBITDUP [22;15;123;0;0], V2-3_stratum [0;0;0;0;0], without_V2-3 [22;0;123;0;0] |
| NB30_B1 | 0.0556 | 0.0540 | 0.0563 |  |
| NB30_B0 | 0.0428 | 0.0418 | 0.0432 |  |
| dNB30_B1_minus_B0 | 0.0128 | 0.0122 | 0.0131 |  |
| fs_pooled_B1 | 0.1146 | 0.0542 | 0.1133 |  |
| false_signs_pooled_B1 | 36 | 76 | 34 |  |
| calls_pooled_B1 | 314 | 1401 | 300 |  |
| calls_distinct_B1 | 176 | 542 | 166 |  |
| false_signs_distinct_B1 | 23 | 31 | 21 |  |
| seeds_with_calls_B1 | 4 | 5 | 4 |  |
| fs_pooled_B0 | 0.1187 | 0.0650 | 0.1154 |  |
| false_signs_pooled_B0 | 19 | 60 | 18 |  |
| calls_pooled_B0 | 160 | 923 | 156 |  |
| calls_distinct_B0 | 123 | 457 | 119 |  |
| false_signs_distinct_B0 | 17 | 34 | 16 |  |
| seeds_with_calls_B0 | 3 | 5 | 3 |  |

## Duplicate sensitivities

| statistic | point | noV2 | noBITDUP | V2-3_stratum | without_V2-3 |
|---|---|---|---|---|---|
| AURC_B1 | 0.0400 | 0.0397 | 0.0402 | 0.0701 | 0.0391 |
| AURC_B0 | 0.0441 | 0.0442 | 0.0442 | 0.0482 | 0.0440 |
| dAURC_B0_minus_B1 | 0.0041 | 0.0045 | 0.0040 | -0.0219 | 0.0050 |
| cov_B1 | 0.0349 | 0.0202 | 0.0353 | 0.0000 | 0.0193 |
| cov_B0 | 0.0178 | 0.0177 | 0.0180 | 0.0000 | 0.0169 |
| dcov_B1_minus_B0 | 0.0171 | 0.0024 | 0.0173 | 0.0000 | 0.0023 |
| fs_B1 | 0.1586 | 0.0813 | 0.1586 | NaN | 0.0813 |
| fs_B0 | 0.0834 | 0.0918 | 0.0834 | NaN | 0.0918 |
| NB30_B1 | 0.0556 | 0.0594 | 0.0557 | -0.1526 | 0.0574 |
| NB30_B0 | 0.0428 | 0.0452 | 0.0430 | -0.1526 | 0.0437 |
| dNB30_B1_minus_B0 | 0.0128 | 0.0143 | 0.0127 | 0.0000 | 0.0137 |
| fs_pooled_B1 | 0.1146 | 0.1152 | 0.1146 | NaN | 0.1152 |
| false_signs_pooled_B1 | 36 | 19 | 36 | 0 | 19 |
| calls_pooled_B1 | 314 | 165 | 314 | 0 | 165 |
| calls_distinct_B1 | 176 | 128 | 176 | 0 | 128 |
| false_signs_distinct_B1 | 23 | 17 | 23 | 0 | 17 |
| seeds_with_calls_B1 | 4 | 3 | 4 | 0 | 3 |
| fs_pooled_B0 | 0.1187 | 0.1241 | 0.1187 | NaN | 0.1241 |
| false_signs_pooled_B0 | 19 | 18 | 19 | 0 | 18 |
| calls_pooled_B0 | 160 | 145 | 160 | 0 | 145 |
| calls_distinct_B0 | 123 | 123 | 123 | 0 | 123 |
| false_signs_distinct_B0 | 17 | 17 | 17 | 0 | 17 |
| seeds_with_calls_B0 | 3 | 2 | 3 | 0 | 2 |
