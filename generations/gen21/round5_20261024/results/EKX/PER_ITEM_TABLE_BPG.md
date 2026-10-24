# 1.8 M07 E-KX (BPG): per-item table

**Label: SENSITIVITY; EXPLORATORY (seen); E-KX-LOPO (known-chemotype deployment; B1 base = round-4 OOF, donors = other publications and other copy groups): never called BP, never placed next to E-A's 0.452; B1 on the same rows is the comparator**

| field | value |
|---|---|
| Item id and estimand | 1.8 M07 sc009 (TODGA-class) Nd dip; estimand E-KX-LOPO (known-chemotype deployment) |
| Population (cells, publications, units, rows per seed) | see 'Population (cells, publications, units, rows per seed)' below |
| Design | E-KX-LOPO: B1 base = round-4 BPG OOF (REF_BPG.parquet), donors = other publications and other copy groups, never called BP (SENSITIVITY: BPG base) |
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
| Label | SENSITIVITY; EXPLORATORY (seen); E-KX-LOPO (known-chemotype deployment; B1 base = round-4 OOF, donors = other publications and other copy groups): never called BP, never placed next to E-A's 0.452; B1 on the same rows is the comparator |
| What changed scientifically | On the seen rows the sc009 Pr/Nd dR2 of the chemotype Nd term over B1 is +0.2524; without the top-3 publications R2(KX_CHEM) - R2(NDU) = -0.0017 on the pooled rows, so the BPG sensitivity of the report rule (BP decides for this item) reads "narrowed to 'universal Nd term'" (CONSISTENCY CHECK; random-chemotype p = 0.0025, descriptive; every rule row in 'Report rule: every rule row' below).  Nothing here is a test; the Pr/Nd warning stands as practical guidance labelled EXPLORATORY. |
| Next registered step | Stage 2 rider: the external sign test on the first 4 new sc009 publications with the frozen u_ext = -0.0776 (killed if the Nd chord is >= 0 in >= 2 of the first 4 new sc009 publications, or if the Pr/Nd squared error against B1 does not fall in most (>= 3) of them [IR M07; brief 6.8]); data-gated, not a package claim (CP-13).  Seal b96c3bef9828. |

## Population (cells, publications, units, rows per seed)

| design | c5 | endpoint | cells | publications | unit | units | rows_per_seed | seeds | clusters | sc009_publications | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BPG | C5_out | SC009_PRND | 46 | 10 | publication | 10 | 46.0000 | 5 | 1 | 10 | METADATA (population construction; no score) |
| BPG | C5_out | POOLED | 214 | 38 | publication | 38 | 1654.0000 | 5 | 13 | 15 | METADATA (population construction; no score) |

## R², MSE, relative SSE vs B1

| endpoint | arm | R2 | MSE | relSSE_vs_B1 | MAE | MAE_balanced | rows_per_seed | label |
|---|---|---|---|---|---|---|---|---|
| SC009_PRND | B1 | 0.2597 | 0.0325 | 0.0000 | 0.1424 | NaN | 46.0000 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | KX_CHEM | 0.5121 | 0.0214 | 0.3328 | 0.1112 | NaN | 46.0000 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | KX_SC009 | 0.5121 | 0.0214 | 0.3328 | 0.1112 | NaN | 46.0000 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | NDU | 0.4136 | 0.0257 | 0.2033 | 0.1232 | NaN | 46.0000 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | PRND_OFFSET | 0.4979 | 0.0220 | 0.3117 | 0.1124 | NaN | 46.0000 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | PRND_COPY | -0.1126 | 0.0488 | -0.5256 | 0.1685 | NaN | 46.0000 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | KX_FROZEN | 0.5158 | 0.0213 | 0.3380 | 0.1107 | NaN | 46.0000 | DIAGNOSTIC (not fold-independent); EXPLORATORY (seen) |
| POOLED | B1 | 0.4863 | 0.0226 | 0.0000 | 0.0942 | 0.1003 | 1654.0000 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | KX_CHEM | 0.4934 | 0.0223 | 0.0139 | 0.0933 | 0.1003 | 1654.0000 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | KX_SC009 | 0.4935 | 0.0223 | 0.0140 | 0.0933 | 0.1003 | 1654.0000 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | NDU | 0.4921 | 0.0224 | 0.0113 | 0.0938 | 0.1007 | 1654.0000 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | PRND_OFFSET | 0.4930 | 0.0223 | 0.0132 | 0.0933 | 0.1003 | 1654.0000 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | PRND_COPY | 0.4760 | 0.0231 | -0.0200 | 0.0950 | 0.1006 | 1654.0000 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | KX_FROZEN | 0.4936 | 0.0223 | 0.0143 | 0.0933 | 0.1003 | 1654.0000 | DIAGNOSTIC (not fold-independent); EXPLORATORY (seen) |

## Per-seed values

| endpoint | cand | ref | dR2 | dR2_per_seed | seeds_positive | relSSE | dMAE | label |
|---|---|---|---|---|---|---|---|---|
| SC009_PRND | KX_SC009 | B1 | 0.2524 | 0.3104;0.1054;0.2656;0.2626;0.3181 | 5 | 0.3328 | 0.0312 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | KX_CHEM | B1 | 0.2524 | 0.3104;0.1054;0.2656;0.2626;0.3181 | 5 | 0.3328 | 0.0312 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | NDU | B1 | 0.1539 | 0.1864;0.0681;0.1634;0.1623;0.1893 | 5 | 0.2033 | 0.0193 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | KX_CHEM | NDU | 0.0985 | 0.1240;0.0374;0.1022;0.1003;0.1288 | 5 | 0.1654 | 0.0120 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | KX_SC009 | NDU | 0.0985 | 0.1240;0.0374;0.1022;0.1003;0.1288 | 5 | 0.1654 | 0.0120 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | PRND_OFFSET | B1 | 0.2382 | 0.3263;0.0693;0.2312;0.2571;0.3069 | 5 | 0.3117 | 0.0301 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | PRND_COPY | B1 | -0.3723 | -0.3509;-0.5311;-0.3190;-0.3710;-0.2896 | 0 | -0.5256 | -0.0261 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | KX_SC009 | PRND_OFFSET | 0.0142 | -0.0159;0.0361;0.0344;0.0055;0.0112 | 4 | 0.0256 | 0.0012 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | KX_SC009 | PRND_COPY | 0.6247 | 0.6613;0.6365;0.5846;0.6336;0.6077 | 5 | 0.5615 | 0.0573 | SENSITIVITY; EXPLORATORY (seen) |
| SC009_PRND | KX_FROZEN | B1 | 0.2561 | 0.3113;0.1106;0.2715;0.2691;0.3181 | 5 | 0.3380 | 0.0318 | DIAGNOSTIC (not fold-independent); EXPLORATORY (seen) |
| POOLED | KX_SC009 | B1 | 0.0072 | 0.0085;0.0051;0.0068;0.0073;0.0083 | 5 | 0.0140 | 0.0009 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | KX_CHEM | B1 | 0.0072 | 0.0084;0.0049;0.0068;0.0074;0.0082 | 5 | 0.0139 | 0.0009 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | NDU | B1 | 0.0058 | 0.0079;0.0059;0.0050;0.0044;0.0058 | 5 | 0.0113 | 0.0004 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | KX_CHEM | NDU | 0.0013 | 0.0006;-0.0010;0.0018;0.0030;0.0024 | 4 | 0.0027 | 0.0005 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | KX_SC009 | NDU | 0.0014 | 0.0006;-0.0008;0.0017;0.0029;0.0025 | 4 | 0.0028 | 0.0005 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | PRND_OFFSET | B1 | 0.0068 | 0.0097;0.0025;0.0063;0.0070;0.0084 | 5 | 0.0132 | 0.0009 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | PRND_COPY | B1 | -0.0103 | -0.0095;-0.0140;-0.0089;-0.0098;-0.0094 | 0 | -0.0200 | -0.0008 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | KX_SC009 | PRND_OFFSET | 0.0004 | -0.0012;0.0027;0.0005;0.0004;-0.0002 | 3 | 0.0008 | 0.0000 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | KX_SC009 | PRND_COPY | 0.0175 | 0.0180;0.0191;0.0156;0.0171;0.0177 | 5 | 0.0334 | 0.0017 | SENSITIVITY; EXPLORATORY (seen) |
| POOLED | KX_FROZEN | B1 | 0.0073 | 0.0085;0.0053;0.0070;0.0075;0.0083 | 5 | 0.0143 | 0.0009 | DIAGNOSTIC (not fold-independent); EXPLORATORY (seen) |

## Unit W/L/T and the descriptive tests

| endpoint | unit | cand | ref | G | W/L/T | SF_POOL_p | SF_POOL_exact | SIGN_p | IM_REL_p | top3_share |
|---|---|---|---|---|---|---|---|---|---|---|
| SC009_PRND | publication | KX_SC009 | B1 | 10 | 7/3/0 | 0.0264 | True | 0.1719 | 0.8181 | 0.8095 |
| SC009_PRND | publication | KX_SC009 | NDU | 10 | 7/3/0 | 0.0654 | True | 0.1719 | 0.8352 | 0.9485 |
| POOLED | cluster_cpg | KX_CHEM | B1 | 13 | 3/2/8 | 0.3125 | True | 0.5000 | 0.0647 | 1.0223 |
| POOLED | publication | KX_CHEM | B1 | 38 | 15/9/14 | 0.0226 | False | 0.1537 | 0.0477 | 0.8187 |
| POOLED | publication | KX_CHEM | NDU | 38 | 17/15/6 | 0.2874 | False | 0.4300 | 0.1244 | 2.0640 |

## LOPO and top-3 share

| endpoint | cand | ref | point | lo | hi | min | min_by | max | max_by | top3_share |
|---|---|---|---|---|---|---|---|---|---|---|
| SC009_PRND | KX_SC009 | B1 | 0.2524 | -0.4301 | 0.9350 | 0.1751 | pub_19783c039b | 0.5551 | pub_126bbe7976 | 0.8095 |
| SC009_PRND | KX_SC009 | NDU | 0.0985 | -0.2255 | 0.4225 | 0.0642 | pub_19783c039b | 0.2427 | pub_126bbe7976 | 0.9485 |
| POOLED | KX_CHEM | B1 | 0.0072 | -0.0007 | 0.0150 | 0.0049 | pub_4cba27f5f3 | 0.0092 | pub_126bbe7976 | 0.8187 |
| POOLED | KX_CHEM | NDU | 0.0013 | -0.0041 | 0.0067 | 0.0001 | pub_4cba27f5f3 | 0.0029 | pub_126bbe7976 | 2.0640 |
| POOLED | NDU | B1 | 0.0058 | 0.0018 | 0.0098 | 0.0047 | pub_4cba27f5f3 | 0.0066 | pub_1bd76c48d9 | 0.5654 |

## C5-BTBP in / out

| c5 | endpoint | cand | ref | dR2 | relSSE | label |
|---|---|---|---|---|---|---|
| C5_out | SC009_PRND | KX_SC009 | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| C5_out | SC009_PRND | KX_CHEM | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| C5_out | SC009_PRND | NDU | B1 | 0.1539 | 0.2033 | SENSITIVITY; EXPLORATORY (seen) |
| C5_out | SC009_PRND | KX_CHEM | NDU | 0.0985 | 0.1654 | SENSITIVITY; EXPLORATORY (seen) |
| C5_out | SC009_PRND | KX_SC009 | NDU | 0.0985 | 0.1654 | SENSITIVITY; EXPLORATORY (seen) |
| C5_out | POOLED | KX_SC009 | B1 | 0.0072 | 0.0140 | SENSITIVITY; EXPLORATORY (seen) |
| C5_out | POOLED | KX_CHEM | B1 | 0.0072 | 0.0139 | SENSITIVITY; EXPLORATORY (seen) |
| C5_out | POOLED | NDU | B1 | 0.0058 | 0.0113 | SENSITIVITY; EXPLORATORY (seen) |
| C5_out | POOLED | KX_CHEM | NDU | 0.0013 | 0.0027 | SENSITIVITY; EXPLORATORY (seen) |
| C5_out | POOLED | KX_SC009 | NDU | 0.0014 | 0.0028 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | SC009_PRND | KX_SC009 | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | SC009_PRND | KX_CHEM | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | SC009_PRND | NDU | B1 | 0.1414 | 0.1868 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | SC009_PRND | KX_CHEM | NDU | 0.1110 | 0.1822 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | SC009_PRND | KX_SC009 | NDU | 0.1110 | 0.1822 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | KX_SC009 | B1 | 0.0059 | 0.0108 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | KX_CHEM | B1 | 0.0078 | 0.0142 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | NDU | B1 | 0.0021 | 0.0039 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | KX_CHEM | NDU | 0.0057 | 0.0104 | SENSITIVITY; EXPLORATORY (seen) |
| C5_in | POOLED | KX_SC009 | NDU | 0.0038 | 0.0069 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | KX_SC009 | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | KX_CHEM | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | NDU | B1 | 0.1437 | 0.1899 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | KX_CHEM | NDU | 0.1087 | 0.1791 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | SC009_PRND | KX_SC009 | NDU | 0.1087 | 0.1791 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | KX_SC009 | B1 | 0.0063 | 0.0122 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | KX_CHEM | B1 | 0.0079 | 0.0152 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | NDU | B1 | 0.0029 | 0.0055 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | KX_CHEM | NDU | 0.0050 | 0.0097 | SENSITIVITY; EXPLORATORY (seen) |
| C5_cell_out | POOLED | KX_SC009 | NDU | 0.0035 | 0.0068 | SENSITIVITY; EXPLORATORY (seen) |

## Duplicate sensitivities

| variant | endpoint | cand | ref | dR2 | relSSE | label |
|---|---|---|---|---|---|---|
| PRIMARY | SC009_PRND | KX_SC009 | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| PRIMARY | SC009_PRND | KX_CHEM | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| PRIMARY | SC009_PRND | NDU | B1 | 0.1539 | 0.2033 | SENSITIVITY; EXPLORATORY (seen) |
| PRIMARY | SC009_PRND | KX_CHEM | NDU | 0.0985 | 0.1654 | SENSITIVITY; EXPLORATORY (seen) |
| PRIMARY | POOLED | KX_SC009 | B1 | 0.0072 | 0.0140 | SENSITIVITY; EXPLORATORY (seen) |
| PRIMARY | POOLED | KX_CHEM | B1 | 0.0072 | 0.0139 | SENSITIVITY; EXPLORATORY (seen) |
| PRIMARY | POOLED | NDU | B1 | 0.0058 | 0.0113 | SENSITIVITY; EXPLORATORY (seen) |
| PRIMARY | POOLED | KX_CHEM | NDU | 0.0013 | 0.0027 | SENSITIVITY; EXPLORATORY (seen) |
| NB | SC009_PRND | KX_SC009 | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| NB | SC009_PRND | KX_CHEM | B1 | 0.2524 | 0.3328 | SENSITIVITY; EXPLORATORY (seen) |
| NB | SC009_PRND | NDU | B1 | 0.1539 | 0.2033 | SENSITIVITY; EXPLORATORY (seen) |
| NB | SC009_PRND | KX_CHEM | NDU | 0.0985 | 0.1654 | SENSITIVITY; EXPLORATORY (seen) |
| NB | POOLED | KX_SC009 | B1 | 0.0072 | 0.0140 | SENSITIVITY; EXPLORATORY (seen) |
| NB | POOLED | KX_CHEM | B1 | 0.0072 | 0.0139 | SENSITIVITY; EXPLORATORY (seen) |
| NB | POOLED | NDU | B1 | 0.0058 | 0.0113 | SENSITIVITY; EXPLORATORY (seen) |
| NB | POOLED | KX_CHEM | NDU | 0.0013 | 0.0027 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | SC009_PRND | KX_SC009 | B1 | 0.2544 | 0.3357 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | SC009_PRND | KX_CHEM | B1 | 0.2544 | 0.3357 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | SC009_PRND | NDU | B1 | 0.1590 | 0.2100 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | SC009_PRND | KX_CHEM | NDU | 0.0954 | 0.1618 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | POOLED | KX_SC009 | B1 | 0.0073 | 0.0142 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | POOLED | KX_CHEM | B1 | 0.0072 | 0.0141 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | POOLED | NDU | B1 | 0.0059 | 0.0115 | SENSITIVITY; EXPLORATORY (seen) |
| OFS | POOLED | KX_CHEM | NDU | 0.0013 | 0.0026 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | SC009_PRND | KX_SC009 | B1 | 0.2540 | 0.3349 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | SC009_PRND | KX_CHEM | B1 | 0.2540 | 0.3349 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | SC009_PRND | NDU | B1 | 0.2042 | 0.2697 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | SC009_PRND | KX_CHEM | NDU | 0.0498 | 0.0919 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | POOLED | KX_SC009 | B1 | 0.0072 | 0.0141 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | POOLED | KX_CHEM | B1 | 0.0071 | 0.0139 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | POOLED | NDU | B1 | 0.0071 | 0.0138 | SENSITIVITY; EXPLORATORY (seen) |
| QNT | POOLED | KX_CHEM | NDU | 0.0000 | 0.0001 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | SC009_PRND | KX_SC009 | B1 | 0.2225 | 0.2936 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | SC009_PRND | KX_CHEM | B1 | 0.2225 | 0.2936 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | SC009_PRND | NDU | B1 | 0.1170 | 0.1547 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | SC009_PRND | KX_CHEM | NDU | 0.1054 | 0.1661 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | POOLED | KX_SC009 | B1 | 0.0064 | 0.0125 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | POOLED | KX_CHEM | B1 | 0.0063 | 0.0123 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | POOLED | NDU | B1 | 0.0046 | 0.0090 | SENSITIVITY; EXPLORATORY (seen) |
| V2 | POOLED | KX_CHEM | NDU | 0.0017 | 0.0033 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | SC009_PRND | KX_SC009 | B1 | 0.2544 | 0.3354 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | SC009_PRND | KX_CHEM | B1 | 0.2544 | 0.3354 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | SC009_PRND | NDU | B1 | 0.1555 | 0.2054 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | SC009_PRND | KX_CHEM | NDU | 0.0989 | 0.1665 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | POOLED | KX_SC009 | B1 | 0.0073 | 0.0142 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | POOLED | KX_CHEM | B1 | 0.0072 | 0.0141 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | POOLED | NDU | B1 | 0.0059 | 0.0114 | SENSITIVITY; EXPLORATORY (seen) |
| V25 | POOLED | KX_CHEM | NDU | 0.0013 | 0.0027 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | SC009_PRND | KX_SC009 | B1 | 0.2659 | 0.3500 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | SC009_PRND | KX_CHEM | B1 | 0.2659 | 0.3500 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | SC009_PRND | NDU | B1 | 0.1720 | 0.2270 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | SC009_PRND | KX_CHEM | NDU | 0.0940 | 0.1628 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | POOLED | KX_SC009 | B1 | 0.0075 | 0.0146 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | POOLED | KX_CHEM | B1 | 0.0074 | 0.0145 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | POOLED | NDU | B1 | 0.0063 | 0.0123 | SENSITIVITY; EXPLORATORY (seen) |
| C3DON | POOLED | KX_CHEM | NDU | 0.0011 | 0.0023 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | SC009_PRND | KX_SC009 | B1 | 0.2392 | 0.3156 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | SC009_PRND | KX_CHEM | B1 | 0.2392 | 0.3156 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | SC009_PRND | NDU | B1 | 0.1878 | 0.2482 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | SC009_PRND | KX_CHEM | NDU | 0.0514 | 0.0918 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | POOLED | KX_SC009 | B1 | 0.0068 | 0.0133 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | POOLED | KX_CHEM | B1 | 0.0068 | 0.0133 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | POOLED | NDU | B1 | 0.0067 | 0.0131 | SENSITIVITY; EXPLORATORY (seen) |
| CELLW | POOLED | KX_CHEM | NDU | 0.0000 | 0.0001 | SENSITIVITY; EXPLORATORY (seen) |

## Report rule: every rule row

BP decides this item's report rule (brief 6.8 'B1 base = the BP OOF prediction'; ITEM_SPEC; REGISTRY 1.8-S03), a sealed exception to brief 9.2 'BPG decides labels'; under the BPG base every row is a rule sensitivity (REGISTRY 1.8-R02, 1.8-S03).  Narrowed in 3 of 5 rows; not narrowed: C5_out SC009_PRND rows (+0.0057); C5_out SC009_PRND rows and donors (+0.0151).

| role | c5 | endpoint | cand | removal | top3_publications | top3_share_of_gain | dR2_cand_full | dR2_NDU_full | dR2_cand_without_top3 | dR2_NDU_without_top3 | rule_stat | rows_per_seed_without | narrowed | outcome | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rule sensitivity | C5_out | POOLED | KX_CHEM | rows | pub_4cba27f5f3,pub_19783c039b,pub_b0b3296ec5 | 0.8187 | 0.0072 | 0.0058 | 0.0016 | 0.0033 | -0.0017 | 1273.0000 | True | narrowed to 'universal Nd term' | CONSISTENCY CHECK; SENSITIVITY; EXPLORATORY (seen) |
| rule sensitivity | C5_out | POOLED | KX_CHEM | rows and donors | pub_4cba27f5f3,pub_19783c039b,pub_b0b3296ec5 | 0.8187 | 0.0065 | 0.0050 | 0.0016 | 0.0028 | -0.0012 | 1273.0000 | True | narrowed to 'universal Nd term' | CONSISTENCY CHECK; SENSITIVITY; EXPLORATORY (seen) |
| rule sensitivity | C5_in | POOLED | KX_CHEM | rows | pub_4cba27f5f3,pub_19783c039b,pub_6a0ef8528c | 0.7448 | 0.0078 | 0.0021 | 0.0027 | 0.0034 | -0.0008 | 1449.0000 | True | narrowed to 'universal Nd term' | CONSISTENCY CHECK; SENSITIVITY; EXPLORATORY (seen) |
| rule sensitivity | C5_out | SC009_PRND | KX_SC009 | rows | pub_19783c039b,pub_4cba27f5f3,pub_b0b3296ec5 | 0.8095 | 0.2524 | 0.1539 | 0.0540 | 0.0483 | 0.0057 | 26.0000 | False | chemotype Nd term not narrowed | CONSISTENCY CHECK; SENSITIVITY; EXPLORATORY (seen) |
| rule sensitivity | C5_out | SC009_PRND | KX_SC009 | rows and donors | pub_19783c039b,pub_4cba27f5f3,pub_b0b3296ec5 | 0.8095 | 0.2289 | 0.1298 | 0.0545 | 0.0394 | 0.0151 | 26.0000 | False | chemotype Nd term not narrowed | CONSISTENCY CHECK; SENSITIVITY; EXPLORATORY (seen) |

## Consistency lines (do not stop the item)

Registered lines of ITEM_SPEC 'Reproduction targets' (CONSISTENCY CHECK); a failed line is reported and never stops the item.  The frozen -0.11: about -0.11 [IR M07], computation not specified.

| line | observed | reference | result | label |
|---|---|---|---|---|
| sc009 Pr/Nd rows per seed (BPG) | 46 | 46 (seen) | match | CONSISTENCY CHECK; EXPLORATORY (seen) |
| V1 sc009 mean Nd chord, publication means first, vs the frozen -0.11 (2 decimals) | -0.1219 | -0.11 (frozen; about -0.11 [IR M07], computation not specified) | FAILED (consistent_2dp = false; -0.11 and u_ext stay as frozen) | CONSISTENCY CHECK; EXPLORATORY (seen) |
| sc009 publications with a negative mean Nd chord | 11 of 12 | 11 of 12 (seen) | match | CONSISTENCY CHECK; EXPLORATORY (seen) |
| V1 sc009 mean Nd chord, cell-weighted (descriptive; not a registered line) | -0.1105 | -0.11 (frozen) | descriptive | DIAGNOSTIC; EXPLORATORY (seen) |
