# 1.1 E-CT core (BP): per-item table

**Label: EXPLORATORY (seen); E-CT value: never placed next to E-A's 0.452; B1 on the same rows is the comparator**

| field | value |
|---|---|
| Item id and estimand | 1.1 E-CT core: condition transfer (M01; P1; PC P1) |
| Population (cells, publications, units, rows per seed) | cells: 99; publications: 12; units: 14; rows_per_seed: 690.0000; cpg_clusters: 3; kish_cpg: 1.2500 |
| Design | BP |
| Read budget | 4 reads (code k = 3) for every scored row; ladder rungs 0 / 2 / 3 reads on the same rows; NP 5 reads = a different read budget |
| Arms on identical rows | B1_r2-4, B0_r2-4, SIB_r2-4, SH_r2-4 (K = 1.78), SHK2, COPY, SH0, OFF, OFF_ALL, OFF_PUB, OFF_XEXT, NEAR, NEARSH, RAND1, RAND2, SUB m, PB_CHEM/XEXT/PUB (+ CM), PB_LABX, PB_PUBHOM/HET, ACOPY, AINT, NP_LAD, NP_DOPTQ_r5; variants NB_, V1_, I1_, SD_, OFS_, QNT_, V25_, INWIN_, OUTWIN_ |
| R², MSE, relative SSE vs B1 | see 'R², MSE, relative SSE vs B1' below |
| Per-seed values | SH_r4 - B1_r4 per seed (PRIMARY): +0.1698;+0.1406;+0.1888;+0.1697;+0.1600 |
| Unit W/L/T and the descriptive tests | SH_r4 - B1_r4 on PRIMARY, 14 publication x chemotype units: SIGN W/L/T 14/0/0, SIGN p 6.10e-05; IM-t on the per-unit relative SSE t 5.766 (p 3.27e-05), on the log SSE ratio t 4.242 (p 4.81e-04); equal-weight per-publication relative SSE 0.3830 (12 publications); per-unit contributions c in 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | dR2: 0.1658; lopo_lo: 0.0458; lopo_hi: 0.2858; lopo_min: 0.1214; lopo_min_without: pub_b0b3296ec5; top3_share: 0.7168; p_signflip_unit: 6.10e-05; p_signflip_chemotype_blocks: 0.0156 |
| C5-BTBP in / out | C5 out (PRIMARY) SH-B1: 0.1658; C5 in (WITH_C5) SH-B1: 0.1665 |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | intervention passed on 1145 targets x 2 passes; positive controls caught 650/650/650 of 650; supports exact, nested, legal (True); B1 max diff 8.9e-16 |
| Compute | 514 s (build + intervention + tables); seal b96c3bef9828 |
| Label | EXPLORATORY (seen); E-CT value: never placed next to E-A's 0.452; B1 on the same rows is the comparator |
| What changed scientifically | With the frozen section-5.1 bit-identity rule, PRIMARY SH_r4 - B1_r4 is +0.1658 (LOPO [+0.0458, +0.2858]) with 14 of 14 units up, against P1's seen BP +0.174 (reproduced here without the bit rule: PRIMARY_NOBIT +0.1739, 13 of 14 units up); SH and its comparators are now sealed code (r5/ect.py) that runs unchanged on new data, and every in-corpus number stays EXPLORATORY (seen), nothing confirmed in-corpus. |
| Next registered step | C-ECT on new laboratories (section 7.5; data-gated: U3 dry run on a corpus-linked laboratory, then the ledger). |

## R², MSE, relative SSE vs B1

| arm | R2 | MSE | relSSE_vs_B1_r4 | R2_per_seed | rows_per_seed | label |
|---|---|---|---|---|---|---|
| B1_r4 | 0.5316 | 0.0120 | 0.0000 | 0.5230;0.5548;0.5053;0.5394;0.5356 | 690.0000 | EXPLORATORY (seen) |
| B0_r4 | 0.4866 | 0.0132 | -0.0961 | 0.4757;0.5182;0.4576;0.4950;0.4864 | 690.0000 | EXPLORATORY (seen) |
| SH_r4 | 0.6974 | 0.0078 | 0.3533 | 0.6928;0.6954;0.6941;0.7091;0.6956 | 690.0000 | EXPLORATORY (seen) |
| SIB_r4 | 0.6832 | 0.0081 | 0.3229 | 0.6807;0.6796;0.6791;0.6935;0.6831 | 690.0000 | EXPLORATORY (seen) |
| SHK2_r4 | 0.6967 | 0.0078 | 0.3518 | 0.6919;0.6949;0.6933;0.7085;0.6948 | 690.0000 | EXPLORATORY (seen) |
| OFF | 0.6869 | 0.0081 | 0.3311 | 0.6882;0.6949;0.6758;0.6897;0.6859 | 690.0000 | EXPLORATORY (seen) |
| OFF_ALL | 0.6915 | 0.0079 | 0.3409 | 0.6934;0.6987;0.6790;0.6916;0.6947 | 690.0000 | EXPLORATORY (seen) |
| NEAR_r4 | 0.5463 | 0.0117 | 0.0303 | 0.5344;0.5321;0.5400;0.5834;0.5415 | 690.0000 | EXPLORATORY (seen) |
| NEARSH_r4 | 0.6485 | 0.0090 | 0.2491 | 0.6405;0.6522;0.6392;0.6639;0.6469 | 690.0000 | EXPLORATORY (seen) |
| COPY | 0.4831 | 0.0133 | -0.1049 | 0.4831;0.4831;0.4831;0.4831;0.4831 | 690.0000 | EXPLORATORY (seen) |
| SH0 | 0.4741 | 0.0135 | -0.1241 | 0.4687;0.4760;0.4748;0.4772;0.4738 | 690.0000 | EXPLORATORY (seen) |
| G14Z | -0.1025 | 0.0283 | -1.3547 | -0.1811;-0.0528;-0.1064;-0.0710;-0.1009 | 690.0000 | EXPLORATORY (seen) |
| E_RAND1_raw_r4 | 0.4597 | 0.0139 | -0.1549 | 0.4525;0.4515;0.4559;0.4815;0.4570 | 690.0000 | EXPLORATORY (seen) |
| E_RAND1_sh_r4 | 0.6181 | 0.0098 | 0.1842 | 0.6120;0.6251;0.6061;0.6290;0.6184 | 690.0000 | EXPLORATORY (seen) |
| E_RAND1_raw_r0 | 0.1248 | 0.0225 | -0.8709 | 0.1248;0.1248;0.1248;0.1248;0.1248 | 690.0000 | EXPLORATORY (seen) |

## Unit W/L/T and the descriptive tests

| publication | chemotype | c | rows | rho | logratio |
|---|---|---|---|---|---|
| pub_b0b3296ec5 | sc009 | 0.0727 | 152.0000 | 0.6268 | 0.9856 |
| pub_1605e436d7 | sc009 | 0.0286 | 84.0000 | 0.2846 | 0.3348 |
| pub_19783c039b | sc009 | 0.0175 | 109.0000 | 0.1381 | 0.1486 |
| pub_8415538291 | sc077 | 0.0120 | 60.0000 | 0.3548 | 0.4381 |
| pub_f8a621ede4 | sc009 | 0.0115 | 9.0000 | 0.8420 | 1.8449 |
| pub_1c1b558856 | sc027 | 0.0086 | 30.0000 | 0.6190 | 0.9649 |
| pub_4cba27f5f3 | sc009 | 0.0047 | 36.0000 | 0.2919 | 0.3452 |
| pub_d68ed8c8cf | sc009 | 0.0040 | 21.0000 | 0.2283 | 0.2591 |
| pub_cd72e4a7bb | sc020 | 0.0027 | 15.0000 | 0.5470 | 0.7920 |
| pub_5a68dc5665 | sc071 | 0.0018 | 42.0000 | 0.8213 | 1.7220 |
| pub_5a68dc5665 | sc076 | 0.0007 | 28.0000 | 0.5981 | 0.9115 |
| pub_5a68dc5665 | sc009 | 0.0006 | 63.0000 | 0.2619 | 0.3036 |
| pub_c8195b4ca7 | sc009 | 0.0002 | 38.0000 | 0.0086 | 0.0086 |
| pub_0a8da089ad | sc009 | 0.0002 | 3.0000 | 0.1053 | 0.1113 |

## Duplicate sensitivities

| stratum | cand | ref | dR2 | lopo_lo | lopo_hi | rows_per_seed | label |
|---|---|---|---|---|---|---|---|
| PRIMARY (CLEAN + bit rule; primary) | SH_r4 | B1_r4 | 0.1658 | 0.0458 | 0.2858 | 690.0000 | EXPLORATORY (seen) |
| PRIMARY_NOBIT (P1's CLEAN; items 1 and 3 kept) | NB_SH_r4 | B1_r4 | 0.1739 | 0.0672 | 0.2807 | 690.0000 | SENSITIVITY |
| PRIMARY_I1 (CLEAN minus the item-1 link only) | I1_SH_r4 | B1_r4 | 0.1647 | 0.0442 | 0.2852 | 690.0000 | SENSITIVITY |
| PRIMARY_SD (bit rule + shape-sd < 0.02 links dropped; label-based) | SD_SH_r4 | B1_r4 | 0.1650 | 0.0445 | 0.2855 | 690.0000 | SENSITIVITY |
| PRIMARY_OFS (bit rule + offset / near-identity links dropped) | OFS_SH_r4 | B1_r4 | 0.1575 | 0.0403 | 0.2747 | 652.0000 | SENSITIVITY |
| PRIMARY_QNT (bit rule + quantised cells dropped as targets and siblings) | QNT_SH_r4 | B1_r4 | 0.1621 | 0.0089 | 0.3152 | 636.0000 | SENSITIVITY |
| WITH_C5_NOBIT | NB_SH_r4 | B1_r4 | 0.1709 | 0.1211 | 0.2207 | 838.0000 | SENSITIVITY |
| WITH_C5 | SH_r4 | B1_r4 | 0.1665 | 0.1123 | 0.2208 | 838.0000 | SENSITIVITY |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_OFF_ALL | 0.0056 | -0.0094 | 0.0206 | 652.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_OFF_XEXT | 0.0653 | -0.1328 | 0.2633 | 152.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_PB_PUB | 0.0583 | -0.1672 | 0.2839 | 152.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | E_OFS_PB_PUB_CM | 0.0610 | -0.1538 | 0.2759 | 152.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_PB_XEXT | 0.2098 | -0.1853 | 0.6048 | 415.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_PB_CHEM | 0.1776 | -0.0062 | 0.3613 | 535.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_PB_LABX | 0.6008 | -0.1033 | 1.3048 | 285.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_NEAR_r4 | 0.1525 | -0.0770 | 0.3821 | 652.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_NEARSH_r4 | 0.0460 | -0.0246 | 0.1166 | 652.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
