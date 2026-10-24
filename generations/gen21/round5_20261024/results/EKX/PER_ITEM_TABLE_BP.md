# 1.8 M07 E-KX (BP): per-item table

**Label: EXPLORATORY (seen); E-KX-LOPO (known-chemotype deployment; B1 base = round-4 OOF, donors = other publications and other copy groups): never called BP, never placed next to E-A's 0.452; B1 on the same rows is the comparator**

| field | value |
|---|---|
| Item id and estimand | 1.8 M07 sc009 (TODGA-class) Nd dip; estimand E-KX-LOPO (known-chemotype deployment) |
| Population (cells, publications, units, rows per seed) | see 'Population (cells, publications, units, rows per seed)' below |
| Design | E-KX-LOPO: B1 base = round-4 BP OOF (REF_BP.parquet), donors = other publications and other copy groups, never called BP |
| Read budget | 4 reads (code k = 3; E-A's nested DOPT_Q read sets, unchanged) |
| Arms on identical rows | B1, KX_CHEM, KX_SC009, NDU, PRND_OFFSET, PRND_COPY, KX_FROZEN; frozen external u = -0.0776 (n = 12; the 2-decimal consistency line of the frozen -0.11 FAILED on V1: publication-first mean -0.1219, cell-weighted -0.1105; see 'Consistency lines (do not stop the item)' below) |
| R², MSE, relative SSE vs B1 | see 'R², MSE, relative SSE vs B1' below |
| Per-seed values | see 'Per-seed values' below |
| Unit W/L/T and the descriptive tests | see 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | see 'LOPO and top-3 share' below |
| C5-BTBP in / out | see 'C5-BTBP in / out' below |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | intervention passed on 229 targets x 10 donor configurations + offsets; positive controls caught 229/229 (self), 221/221 (same publication), 186/186 (leaky pool); supports exact; B1 identity 0.0e+00 |
| Compute | 37s |
| Label | EXPLORATORY (seen); E-KX-LOPO (known-chemotype deployment; B1 base = round-4 OOF, donors = other publications and other copy groups): never called BP, never placed next to E-A's 0.452; B1 on the same rows is the comparator |
| What changed scientifically | On the seen rows the sc009 Pr/Nd dR2 of the chemotype Nd term over B1 is +0.2430; without the top-3 publications R2(KX_CHEM) - R2(NDU) = -0.0012 on the pooled rows, so the report rule reads "narrowed to 'universal Nd term'" (CONSISTENCY CHECK; random-chemotype p = 0.0015, descriptive; every rule row in 'Report rule: every rule row' below).  Nothing here is a test; the Pr/Nd warning stands as practical guidance labelled EXPLORATORY. |
| Next registered step | Stage 2 rider: the external sign test on the first 4 new sc009 publications with the frozen u_ext = -0.0776 (killed if the Nd chord is >= 0 in >= 2 of the first 4 new sc009 publications, or if the Pr/Nd squared error against B1 does not fall in most (>= 3) of them [IR M07; brief 6.8]); data-gated, not a package claim (CP-13).  Seal b96c3bef9828. |

## Population (cells, publications, units, rows per seed)

| design | c5 | endpoint | cells | publications | unit | units | rows_per_seed | seeds | clusters | sc009_publications | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BP | C5_out | SC009_PRND | 46 | 10 | publication | 10 | 46.0000 | 5 | 1 | 10 | METADATA (population construction; no score) |
| BP | C5_out | POOLED | 214 | 38 | publication | 38 | 1654.0000 | 5 | 13 | 15 | METADATA (population construction; no score) |

## R², MSE, relative SSE vs B1

| endpoint | arm | R2 | MSE | relSSE_vs_B1 | MAE | MAE_balanced | rows_per_seed | label |
|---|---|---|---|---|---|---|---|---|
| SC009_PRND | B1 | 0.2648 | 0.0323 | 0.0000 | 0.1430 | NaN | 46.0000 | EXPLORATORY (seen) |
| SC009_PRND | KX_CHEM | 0.5078 | 0.0216 | 0.3232 | 0.1130 | NaN | 46.0000 | EXPLORATORY (seen) |
| SC009_PRND | KX_SC009 | 0.5078 | 0.0216 | 0.3232 | 0.1130 | NaN | 46.0000 | EXPLORATORY (seen) |
| SC009_PRND | NDU | 0.4149 | 0.0257 | 0.1999 | 0.1251 | NaN | 46.0000 | EXPLORATORY (seen) |
| SC009_PRND | PRND_OFFSET | 0.4993 | 0.0220 | 0.3094 | 0.1132 | NaN | 46.0000 | EXPLORATORY (seen) |
| SC009_PRND | PRND_COPY | -0.1126 | 0.0488 | -0.5338 | 0.1685 | NaN | 46.0000 | EXPLORATORY (seen) |
| SC009_PRND | KX_FROZEN | 0.5130 | 0.0214 | 0.3303 | 0.1124 | NaN | 46.0000 | DIAGNOSTIC (not fold-independent); EXPLORATORY (seen) |
| POOLED | B1 | 0.4862 | 0.0226 | 0.0000 | 0.0940 | 0.0999 | 1654.0000 | EXPLORATORY (seen) |
| POOLED | KX_CHEM | 0.4932 | 0.0223 | 0.0136 | 0.0931 | 0.0999 | 1654.0000 | EXPLORATORY (seen) |
| POOLED | KX_SC009 | 0.4932 | 0.0223 | 0.0137 | 0.0931 | 0.0999 | 1654.0000 | EXPLORATORY (seen) |
| POOLED | NDU | 0.4914 | 0.0224 | 0.0101 | 0.0937 | 0.1004 | 1654.0000 | EXPLORATORY (seen) |
| POOLED | PRND_OFFSET | 0.4927 | 0.0223 | 0.0127 | 0.0932 | 0.0999 | 1654.0000 | EXPLORATORY (seen) |
| POOLED | PRND_COPY | 0.4753 | 0.0231 | -0.0211 | 0.0949 | 0.1002 | 1654.0000 | EXPLORATORY (seen) |
| POOLED | KX_FROZEN | 0.4934 | 0.0223 | 0.0140 | 0.0931 | 0.0999 | 1654.0000 | DIAGNOSTIC (not fold-independent); EXPLORATORY (seen) |

## Per-seed values

| endpoint | cand | ref | dR2 | dR2_per_seed | seeds_positive | relSSE | dMAE | label |
|---|---|---|---|---|---|---|---|---|
| SC009_PRND | KX_SC009 | B1 | 0.2430 | 0.3090;0.1054;0.2930;0.2480;0.2595 | 5 | 0.3232 | 0.0300 | EXPLORATORY (seen) |
| SC009_PRND | KX_CHEM | B1 | 0.2430 | 0.3090;0.1054;0.2930;0.2480;0.2595 | 5 | 0.3232 | 0.0300 | EXPLORATORY (seen) |
| SC009_PRND | NDU | B1 | 0.1501 | 0.1857;0.0681;0.1771;0.1567;0.1629 | 5 | 0.1999 | 0.0179 | EXPLORATORY (seen) |
| SC009_PRND | KX_CHEM | NDU | 0.0929 | 0.1232;0.0374;0.1159;0.0914;0.0967 | 5 | 0.1566 | 0.0121 | EXPLORATORY (seen) |
| SC009_PRND | KX_SC009 | NDU | 0.0929 | 0.1232;0.0374;0.1159;0.0914;0.0967 | 5 | 0.1566 | 0.0121 | EXPLORATORY (seen) |
| SC009_PRND | PRND_OFFSET | B1 | 0.2345 | 0.3243;0.0693;0.2719;0.2465;0.2605 | 5 | 0.3094 | 0.0299 | EXPLORATORY (seen) |
| SC009_PRND | PRND_COPY | B1 | -0.3774 | -0.3529;-0.5311;-0.3103;-0.3366;-0.3561 | 0 | -0.5338 | -0.0254 | EXPLORATORY (seen) |
| SC009_PRND | KX_SC009 | PRND_OFFSET | 0.0085 | -0.0154;0.0361;0.0211;0.0015;-0.0010 | 3 | 0.0151 | 0.0002 | EXPLORATORY (seen) |
| SC009_PRND | KX_SC009 | PRND_COPY | 0.6204 | 0.6619;0.6365;0.6034;0.5847;0.6156 | 5 | 0.5576 | 0.0555 | EXPLORATORY (seen) |
| SC009_PRND | KX_FROZEN | B1 | 0.2482 | 0.3100;0.1106;0.2967;0.2565;0.2671 | 5 | 0.3303 | 0.0306 | DIAGNOSTIC (not fold-independent); EXPLORATORY (seen) |
| POOLED | KX_SC009 | B1 | 0.0070 | 0.0085;0.0051;0.0075;0.0069;0.0071 | 5 | 0.0137 | 0.0009 | EXPLORATORY (seen) |
| POOLED | KX_CHEM | B1 | 0.0070 | 0.0084;0.0049;0.0075;0.0070;0.0070 | 5 | 0.0136 | 0.0009 | EXPLORATORY (seen) |
| POOLED | NDU | B1 | 0.0052 | 0.0059;0.0060;0.0051;0.0043;0.0049 | 5 | 0.0101 | 0.0003 | EXPLORATORY (seen) |
| POOLED | KX_CHEM | NDU | 0.0018 | 0.0026;-0.0010;0.0024;0.0027;0.0022 | 4 | 0.0035 | 0.0006 | EXPLORATORY (seen) |
| POOLED | KX_SC009 | NDU | 0.0018 | 0.0026;-0.0008;0.0024;0.0027;0.0022 | 4 | 0.0036 | 0.0006 | EXPLORATORY (seen) |
| POOLED | PRND_OFFSET | B1 | 0.0065 | 0.0088;0.0025;0.0074;0.0067;0.0071 | 5 | 0.0127 | 0.0008 | EXPLORATORY (seen) |
| POOLED | PRND_COPY | B1 | -0.0109 | -0.0115;-0.0140;-0.0087;-0.0088;-0.0113 | 0 | -0.0211 | -0.0009 | EXPLORATORY (seen) |
| POOLED | KX_SC009 | PRND_OFFSET | 0.0005 | -0.0003;0.0027;0.0002;0.0003;-0.0001 | 3 | 0.0010 | 0.0000 | EXPLORATORY (seen) |
| POOLED | KX_SC009 | PRND_COPY | 0.0179 | 0.0200;0.0191;0.0162;0.0158;0.0184 | 5 | 0.0341 | 0.0018 | EXPLORATORY (seen) |
| POOLED | KX_FROZEN | B1 | 0.0072 | 0.0085;0.0053;0.0077;0.0072;0.0073 | 5 | 0.0140 | 0.0009 | DIAGNOSTIC (not fold-independent); EXPLORATORY (seen) |

## Unit W/L/T and the descriptive tests

| endpoint | unit | cand | ref | G | W/L/T | SF_POOL_p | SF_POOL_exact | SIGN_p | IM_REL_p | top3_share |
|---|---|---|---|---|---|---|---|---|---|---|
| SC009_PRND | publication | KX_SC009 | B1 | 10 | 7/3/0 | 0.0342 | True | 0.1719 | 0.8100 | 0.8490 |
| SC009_PRND | publication | KX_SC009 | NDU | 10 | 7/3/0 | 0.0811 | True | 0.1719 | 0.8220 | 1.0164 |
| POOLED | cluster_cpg | KX_CHEM | B1 | 13 | 4/1/8 | 0.2812 | True | 0.1875 | 0.0803 | 1.0222 |
| POOLED | publication | KX_CHEM | B1 | 38 | 15/9/14 | 0.0266 | False | 0.1537 | 0.0568 | 0.8457 |
| POOLED | publication | KX_CHEM | NDU | 38 | 16/16/6 | 0.2402 | False | 0.5700 | 0.0761 | 1.6661 |

## LOPO and top-3 share

| endpoint | cand | ref | point | lo | hi | min | min_by | max | max_by | top3_share |
|---|---|---|---|---|---|---|---|---|---|---|
| SC009_PRND | KX_SC009 | B1 | 0.2430 | -0.4381 | 0.9241 | 0.1636 | pub_19783c039b | 0.5439 | pub_126bbe7976 | 0.8490 |
| SC009_PRND | KX_SC009 | NDU | 0.0929 | -0.2307 | 0.4165 | 0.0576 | pub_19783c039b | 0.2361 | pub_126bbe7976 | 1.0164 |
| POOLED | KX_CHEM | B1 | 0.0070 | -0.0009 | 0.0148 | 0.0047 | pub_4cba27f5f3 | 0.0090 | pub_126bbe7976 | 0.8457 |
| POOLED | KX_CHEM | NDU | 0.0018 | -0.0037 | 0.0073 | 0.0006 | pub_4cba27f5f3 | 0.0034 | pub_126bbe7976 | 1.6661 |
| POOLED | NDU | B1 | 0.0052 | 0.0010 | 0.0094 | 0.0041 | pub_4cba27f5f3 | 0.0061 | pub_1bd76c48d9 | 0.6348 |

## C5-BTBP in / out

| c5 | endpoint | cand | ref | dR2 | relSSE | label |
|---|---|---|---|---|---|---|
| C5_out | SC009_PRND | KX_SC009 | B1 | 0.2430 | 0.3232 | EXPLORATORY (seen) |
| C5_out | SC009_PRND | KX_CHEM | B1 | 0.2430 | 0.3232 | EXPLORATORY (seen) |
| C5_out | SC009_PRND | NDU | B1 | 0.1501 | 0.1999 | EXPLORATORY (seen) |
| C5_out | SC009_PRND | KX_CHEM | NDU | 0.0929 | 0.1566 | EXPLORATORY (seen) |
| C5_out | SC009_PRND | KX_SC009 | NDU | 0.0929 | 0.1566 | EXPLORATORY (seen) |
| C5_out | POOLED | KX_SC009 | B1 | 0.0070 | 0.0137 | EXPLORATORY (seen) |
| C5_out | POOLED | KX_CHEM | B1 | 0.0070 | 0.0136 | EXPLORATORY (seen) |
| C5_out | POOLED | NDU | B1 | 0.0052 | 0.0101 | EXPLORATORY (seen) |
| C5_out | POOLED | KX_CHEM | NDU | 0.0018 | 0.0035 | EXPLORATORY (seen) |
| C5_out | POOLED | KX_SC009 | NDU | 0.0018 | 0.0036 | EXPLORATORY (seen) |
| C5_in | SC009_PRND | KX_SC009 | B1 | 0.2430 | 0.3232 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | SC009_PRND | KX_CHEM | B1 | 0.2430 | 0.3232 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | SC009_PRND | NDU | B1 | 0.1380 | 0.1838 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | SC009_PRND | KX_CHEM | NDU | 0.1050 | 0.1732 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | SC009_PRND | KX_SC009 | NDU | 0.1050 | 0.1732 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | KX_SC009 | B1 | 0.0058 | 0.0105 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | KX_CHEM | B1 | 0.0077 | 0.0140 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | NDU | B1 | 0.0017 | 0.0031 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | KX_CHEM | NDU | 0.0060 | 0.0109 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | KX_SC009 | NDU | 0.0041 | 0.0075 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | KX_SC009 | B1 | 0.2430 | 0.3232 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | KX_CHEM | B1 | 0.2430 | 0.3232 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | NDU | B1 | 0.1403 | 0.1868 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | KX_CHEM | NDU | 0.1027 | 0.1702 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | KX_SC009 | NDU | 0.1027 | 0.1702 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | KX_SC009 | B1 | 0.0062 | 0.0119 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | KX_CHEM | B1 | 0.0077 | 0.0149 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | NDU | B1 | 0.0024 | 0.0046 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | KX_CHEM | NDU | 0.0053 | 0.0103 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | KX_SC009 | NDU | 0.0038 | 0.0074 | SENSITIVITY; EXPLORATORY (seen) |

## Duplicate sensitivities

| variant | endpoint | cand | ref | dR2 | relSSE | label |
|---|---|---|---|---|---|---|
| PRIMARY | SC009_PRND | KX_SC009 | B1 | 0.2430 | 0.3232 | EXPLORATORY (seen) |
| PRIMARY | SC009_PRND | KX_CHEM | B1 | 0.2430 | 0.3232 | EXPLORATORY (seen) |
| PRIMARY | SC009_PRND | NDU | B1 | 0.1501 | 0.1999 | EXPLORATORY (seen) |
| PRIMARY | SC009_PRND | KX_CHEM | NDU | 0.0929 | 0.1566 | EXPLORATORY (seen) |
| PRIMARY | POOLED | KX_SC009 | B1 | 0.0070 | 0.0137 | EXPLORATORY (seen) |
| PRIMARY | POOLED | KX_CHEM | B1 | 0.0070 | 0.0136 | EXPLORATORY (seen) |
| PRIMARY | POOLED | NDU | B1 | 0.0052 | 0.0101 | EXPLORATORY (seen) |
| PRIMARY | POOLED | KX_CHEM | NDU | 0.0018 | 0.0035 | EXPLORATORY (seen) |
| NB | SC009_PRND | KX_SC009 | B1 | 0.2430 | 0.3232 | SENSITIVITY; EXPLORATORY (seen) |
| NB | SC009_PRND | KX_CHEM | B1 | 0.2430 | 0.3232 | SENSITIVITY; EXPLORATORY (seen) |
| NB | SC009_PRND | NDU | B1 | 0.1501 | 0.1999 | SENSITIVITY; EXPLORATORY (seen) |
| NB | SC009_PRND | KX_CHEM | NDU | 0.0929 | 0.1566 | SENSITIVITY; EXPLORATORY (seen) |
| NB | POOLED | KX_SC009 | B1 | 0.0070 | 0.0137 | SENSITIVITY; EXPLORATORY (seen) |
| NB | POOLED | KX_CHEM | B1 | 0.0070 | 0.0136 | SENSITIVITY; EXPLORATORY (seen) |
| NB | POOLED | NDU | B1 | 0.0052 | 0.0101 | SENSITIVITY; EXPLORATORY (seen) |
| NB | POOLED | KX_CHEM | NDU | 0.0018 | 0.0035 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | SC009_PRND | KX_SC009 | B1 | 0.2452 | 0.3263 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | SC009_PRND | KX_CHEM | B1 | 0.2452 | 0.3263 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | SC009_PRND | NDU | B1 | 0.1552 | 0.2066 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | SC009_PRND | KX_CHEM | NDU | 0.0900 | 0.1534 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | POOLED | KX_SC009 | B1 | 0.0071 | 0.0139 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | POOLED | KX_CHEM | B1 | 0.0071 | 0.0138 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | POOLED | NDU | B1 | 0.0053 | 0.0104 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | POOLED | KX_CHEM | NDU | 0.0017 | 0.0034 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | SC009_PRND | KX_SC009 | B1 | 0.2446 | 0.3253 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | SC009_PRND | KX_CHEM | B1 | 0.2446 | 0.3253 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | SC009_PRND | NDU | B1 | 0.1986 | 0.2644 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | SC009_PRND | KX_CHEM | NDU | 0.0460 | 0.0851 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | POOLED | KX_SC009 | B1 | 0.0071 | 0.0138 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | POOLED | KX_CHEM | B1 | 0.0070 | 0.0136 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | POOLED | NDU | B1 | 0.0062 | 0.0121 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | POOLED | KX_CHEM | NDU | 0.0007 | 0.0015 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | SC009_PRND | KX_SC009 | B1 | 0.2155 | 0.2868 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | SC009_PRND | KX_CHEM | B1 | 0.2155 | 0.2868 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | SC009_PRND | NDU | B1 | 0.1144 | 0.1524 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | SC009_PRND | KX_CHEM | NDU | 0.1011 | 0.1602 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | POOLED | KX_SC009 | B1 | 0.0063 | 0.0122 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | POOLED | KX_CHEM | B1 | 0.0062 | 0.0121 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | POOLED | NDU | B1 | 0.0042 | 0.0082 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | POOLED | KX_CHEM | NDU | 0.0020 | 0.0039 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | SC009_PRND | KX_SC009 | B1 | 0.2450 | 0.3258 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | SC009_PRND | KX_CHEM | B1 | 0.2450 | 0.3258 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | SC009_PRND | NDU | B1 | 0.1517 | 0.2020 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | SC009_PRND | KX_CHEM | NDU | 0.0933 | 0.1577 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | POOLED | KX_SC009 | B1 | 0.0071 | 0.0138 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | POOLED | KX_CHEM | B1 | 0.0070 | 0.0137 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | POOLED | NDU | B1 | 0.0053 | 0.0102 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | POOLED | KX_CHEM | NDU | 0.0018 | 0.0035 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | SC009_PRND | KX_SC009 | B1 | 0.2538 | 0.3370 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | SC009_PRND | KX_CHEM | B1 | 0.2538 | 0.3370 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | SC009_PRND | NDU | B1 | 0.1674 | 0.2229 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | SC009_PRND | KX_CHEM | NDU | 0.0863 | 0.1502 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | POOLED | KX_SC009 | B1 | 0.0073 | 0.0142 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | POOLED | KX_CHEM | B1 | 0.0072 | 0.0141 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | POOLED | NDU | B1 | 0.0056 | 0.0109 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | POOLED | KX_CHEM | NDU | 0.0016 | 0.0032 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | SC009_PRND | KX_SC009 | B1 | 0.2304 | 0.3065 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | SC009_PRND | KX_CHEM | B1 | 0.2304 | 0.3065 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | SC009_PRND | NDU | B1 | 0.1828 | 0.2435 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | SC009_PRND | KX_CHEM | NDU | 0.0476 | 0.0853 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | POOLED | KX_SC009 | B1 | 0.0067 | 0.0130 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | POOLED | KX_CHEM | B1 | 0.0066 | 0.0129 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | POOLED | NDU | B1 | 0.0060 | 0.0116 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | POOLED | KX_CHEM | NDU | 0.0007 | 0.0013 | SENSITIVITY; EXPLORATORY (seen) |

## Report rule: every rule row

BP decides this item's report rule (brief 6.8 'B1 base = the BP OOF prediction'; ITEM_SPEC; REGISTRY 1.8-S03), a sealed exception to brief 9.2 'BPG decides labels'; this table holds the primary row (role 'REPORT RULE (primary)': C5_out, POOLED, rows removed; REGISTRY 1.8-R01) and its sensitivities (1.8-R02).  Narrowed in 4 of 5 rows; not narrowed: C5_out SC009_PRND rows and donors (+0.0094).

| role | c5 | endpoint | cand | removal | top3_publications | top3_share_of_gain | dR2_cand_full | dR2_NDU_full | dR2_cand_without_top3 | dR2_NDU_without_top3 | rule_stat | rows_per_seed_without | narrowed | outcome | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REPORT RULE (primary) | C5_out | POOLED | KX_CHEM | rows | pub_19783c039b,pub_4cba27f5f3,pub_b0b3296ec5 | 0.8457 | 0.0070 | 0.0052 | 0.0013 | 0.0025 | -0.0012 | 1273.0000 | True | narrowed to 'universal Nd term' | CONSISTENCY CHECK; EXPLORATORY (seen) |
| rule sensitivity | C5_out | POOLED | KX_CHEM | rows and donors | pub_19783c039b,pub_4cba27f5f3,pub_b0b3296ec5 | 0.8457 | 0.0064 | 0.0045 | 0.0014 | 0.0022 | -0.0008 | 1273.0000 | True | narrowed to 'universal Nd term' | CONSISTENCY CHECK; SENSITIVITY; EXPLORATORY (seen) |
| rule sensitivity | C5_in | POOLED | KX_CHEM | rows | pub_19783c039b,pub_4cba27f5f3,pub_6a0ef8528c | 0.7629 | 0.0077 | 0.0017 | 0.0024 | 0.0028 | -0.0004 | 1449.0000 | True | narrowed to 'universal Nd term' | CONSISTENCY CHECK; SENSITIVITY; EXPLORATORY (seen) |
| rule sensitivity | C5_out | SC009_PRND | KX_SC009 | rows | pub_19783c039b,pub_4cba27f5f3,pub_b0b3296ec5 | 0.8490 | 0.2430 | 0.1501 | 0.0412 | 0.0429 | -0.0017 | 26.0000 | True | narrowed to 'universal Nd term' | CONSISTENCY CHECK; SENSITIVITY; EXPLORATORY (seen) |
| rule sensitivity | C5_out | SC009_PRND | KX_SC009 | rows and donors | pub_19783c039b,pub_4cba27f5f3,pub_b0b3296ec5 | 0.8490 | 0.2219 | 0.1270 | 0.0448 | 0.0353 | 0.0094 | 26.0000 | False | chemotype Nd term not narrowed | CONSISTENCY CHECK; SENSITIVITY; EXPLORATORY (seen) |

## Consistency lines (do not stop the item)

Registered lines of ITEM_SPEC 'Reproduction targets' (CONSISTENCY CHECK); a failed line is reported and never stops the item.  The frozen -0.11: about -0.11 [IR M07], computation not specified.

| line | observed | reference | result | label |
|---|---|---|---|---|
| sc009 Pr/Nd rows per seed (BP) | 46 | 46 (seen) | match | CONSISTENCY CHECK; EXPLORATORY (seen) |
| V1 sc009 mean Nd chord, publication means first, vs the frozen -0.11 (2 decimals) | -0.1219 | -0.11 (frozen; about -0.11 [IR M07], computation not specified) | FAILED (consistent_2dp = false; -0.11 and u_ext stay as frozen) | CONSISTENCY CHECK; EXPLORATORY (seen) |
| sc009 publications with a negative mean Nd chord | 11 of 12 | 11 of 12 (seen) | match | CONSISTENCY CHECK; EXPLORATORY (seen) |
| V1 sc009 mean Nd chord, cell-weighted (descriptive; not a registered line) | -0.1105 | -0.11 (frozen) | descriptive | DIAGNOSTIC; EXPLORATORY (seen) |
