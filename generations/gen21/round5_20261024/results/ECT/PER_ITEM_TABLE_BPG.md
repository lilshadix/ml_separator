# 1.1 E-CT core (BPG): per-item table

**Label: EXPLORATORY (seen); E-CT value: never placed next to E-A's 0.452; B1 on the same rows is the comparator**

| field | value |
|---|---|
| Item id and estimand | 1.1 E-CT core: condition transfer (M01; P1; PC P1) |
| Population (cells, publications, units, rows per seed) | cells: 99; publications: 12; units: 14; rows_per_seed: 690.0000; cpg_clusters: 3; kish_cpg: 1.2500 |
| Design | BPG |
| Read budget | 4 reads (code k = 3) for every scored row; ladder rungs 0 / 2 / 3 reads on the same rows; NP 5 reads = a different read budget |
| Arms on identical rows | B1_r2-4, B0_r2-4, SIB_r2-4, SH_r2-4 (K = 1.78), SHK2, COPY, SH0, OFF, OFF_ALL, OFF_PUB, OFF_XEXT, NEAR, NEARSH, RAND1, RAND2, SUB m, PB_CHEM/XEXT/PUB (+ CM), PB_LABX, PB_PUBHOM/HET, ACOPY, AINT, NP_LAD, NP_DOPTQ_r5; variants NB_, V1_, I1_, SD_, OFS_, QNT_, V25_, INWIN_, OUTWIN_ |
| R², MSE, relative SSE vs B1 | see 'R², MSE, relative SSE vs B1' below |
| Per-seed values | SH_r4 - B1_r4 per seed (PRIMARY): +0.1694;+0.1430;+0.1804;+0.1704;+0.1833 |
| Unit W/L/T and the descriptive tests | SH_r4 - B1_r4 on PRIMARY, 14 publication x chemotype units: SIGN W/L/T 14/0/0, SIGN p 6.10e-05; IM-t on the per-unit relative SSE t 5.842 (p 2.88e-05), on the log SSE ratio t 4.283 (p 4.45e-04); equal-weight per-publication relative SSE 0.3870 (12 publications); per-unit contributions c in 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | dR2: 0.1693; lopo_lo: 0.0553; lopo_hi: 0.2833; lopo_min: 0.1264; lopo_min_without: pub_b0b3296ec5; top3_share: 0.7222; p_signflip_unit: 6.10e-05; p_signflip_chemotype_blocks: 0.0156 |
| C5-BTBP in / out | C5 out (PRIMARY) SH-B1: 0.1693; C5 in (WITH_C5) SH-B1: 0.1682 |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | intervention passed on 1145 targets x 2 passes; positive controls caught 650/650/650 of 650; supports exact, nested, legal (True); B1 max diff 8.9e-16 |
| Compute | 522 s (build + intervention + tables); seal b96c3bef9828 |
| Label | EXPLORATORY (seen); E-CT value: never placed next to E-A's 0.452; B1 on the same rows is the comparator |
| What changed scientifically | With the frozen section-5.1 bit-identity rule, PRIMARY SH_r4 - B1_r4 is +0.1693 (LOPO [+0.0553, +0.2833]) with 14 of 14 units up, against P1's seen BPG +0.177 with 13 of 14 (reproduced here without the bit rule: PRIMARY_NOBIT +0.1774, 13 of 14 units up); SH and its comparators are now sealed code (r5/ect.py) that runs unchanged on new data, and every in-corpus number stays EXPLORATORY (seen), nothing confirmed in-corpus. |
| Next registered step | C-ECT on new laboratories (section 7.5; data-gated: U3 dry run on a corpus-linked laboratory, then the ledger). |

## R², MSE, relative SSE vs B1

| arm | R2 | MSE | relSSE_vs_B1_r4 | R2_per_seed | rows_per_seed | label |
|---|---|---|---|---|---|---|
| B1_r4 | 0.5309 | 0.0121 | 0.0000 | 0.5234;0.5519;0.5294;0.5387;0.5109 | 690.0000 | EXPLORATORY (seen) |
| B0_r4 | 0.4871 | 0.0132 | -0.0932 | 0.4757;0.5182;0.4819;0.4950;0.4645 | 690.0000 | EXPLORATORY (seen) |
| SH_r4 | 0.7002 | 0.0077 | 0.3604 | 0.6929;0.6949;0.7098;0.7090;0.6943 | 690.0000 | EXPLORATORY (seen) |
| SIB_r4 | 0.6862 | 0.0081 | 0.3306 | 0.6807;0.6796;0.6948;0.6935;0.6826 | 690.0000 | EXPLORATORY (seen) |
| SHK2_r4 | 0.6994 | 0.0077 | 0.3588 | 0.6920;0.6944;0.7091;0.7084;0.6932 | 690.0000 | EXPLORATORY (seen) |
| OFF | 0.6883 | 0.0080 | 0.3352 | 0.6882;0.6943;0.6854;0.6896;0.6838 | 690.0000 | EXPLORATORY (seen) |
| OFF_ALL | 0.6915 | 0.0079 | 0.3420 | 0.6934;0.6982;0.6900;0.6914;0.6843 | 690.0000 | EXPLORATORY (seen) |
| NEAR_r4 | 0.5539 | 0.0115 | 0.0485 | 0.5344;0.5321;0.5838;0.5834;0.5359 | 690.0000 | EXPLORATORY (seen) |
| NEARSH_r4 | 0.6501 | 0.0090 | 0.2538 | 0.6407;0.6510;0.6595;0.6635;0.6356 | 690.0000 | EXPLORATORY (seen) |
| COPY | 0.4831 | 0.0133 | -0.1027 | 0.4831;0.4831;0.4831;0.4831;0.4831 | 690.0000 | EXPLORATORY (seen) |
| SH0 | 0.4741 | 0.0135 | -0.1219 | 0.4686;0.4760;0.4748;0.4772;0.4738 | 690.0000 | EXPLORATORY (seen) |
| G14Z | -0.1027 | 0.0284 | -1.3508 | -0.1819;-0.0531;-0.1065;-0.0711;-0.1010 | 690.0000 | EXPLORATORY (seen) |
| E_RAND1_raw_r4 | 0.4648 | 0.0138 | -0.1417 | 0.4525;0.4515;0.4830;0.4815;0.4555 | 690.0000 | EXPLORATORY (seen) |
| E_RAND1_sh_r4 | 0.6196 | 0.0098 | 0.1889 | 0.6122;0.6239;0.6256;0.6288;0.6077 | 690.0000 | EXPLORATORY (seen) |
| E_RAND1_raw_r0 | 0.1248 | 0.0225 | -0.8672 | 0.1248;0.1248;0.1248;0.1248;0.1248 | 690.0000 | EXPLORATORY (seen) |

## Unit W/L/T and the descriptive tests

| publication | chemotype | c | rows | rho | logratio |
|---|---|---|---|---|---|
| pub_b0b3296ec5 | sc009 | 0.0724 | 152.0000 | 0.6274 | 0.9872 |
| pub_1605e436d7 | sc009 | 0.0289 | 84.0000 | 0.2924 | 0.3459 |
| pub_19783c039b | sc009 | 0.0210 | 109.0000 | 0.1615 | 0.1761 |
| pub_8415538291 | sc077 | 0.0120 | 60.0000 | 0.3548 | 0.4381 |
| pub_f8a621ede4 | sc009 | 0.0114 | 9.0000 | 0.8410 | 1.8390 |
| pub_1c1b558856 | sc027 | 0.0085 | 30.0000 | 0.6163 | 0.9578 |
| pub_4cba27f5f3 | sc009 | 0.0047 | 36.0000 | 0.2909 | 0.3437 |
| pub_d68ed8c8cf | sc009 | 0.0039 | 21.0000 | 0.2266 | 0.2570 |
| pub_cd72e4a7bb | sc020 | 0.0032 | 15.0000 | 0.5706 | 0.8453 |
| pub_5a68dc5665 | sc071 | 0.0018 | 42.0000 | 0.8225 | 1.7287 |
| pub_5a68dc5665 | sc076 | 0.0007 | 28.0000 | 0.5859 | 0.8816 |
| pub_5a68dc5665 | sc009 | 0.0006 | 63.0000 | 0.2627 | 0.3048 |
| pub_c8195b4ca7 | sc009 | 0.0002 | 38.0000 | 0.0086 | 0.0086 |
| pub_0a8da089ad | sc009 | 0.0002 | 3.0000 | 0.1053 | 0.1113 |

## Duplicate sensitivities

| stratum | cand | ref | dR2 | lopo_lo | lopo_hi | rows_per_seed | label |
|---|---|---|---|---|---|---|---|
| PRIMARY (CLEAN + bit rule; primary) | SH_r4 | B1_r4 | 0.1693 | 0.0553 | 0.2833 | 690.0000 | EXPLORATORY (seen) |
| PRIMARY_NOBIT (P1's CLEAN; items 1 and 3 kept) | NB_SH_r4 | B1_r4 | 0.1774 | 0.0757 | 0.2790 | 690.0000 | SENSITIVITY |
| PRIMARY_I1 (CLEAN minus the item-1 link only) | I1_SH_r4 | B1_r4 | 0.1682 | 0.0537 | 0.2828 | 690.0000 | SENSITIVITY |
| PRIMARY_SD (bit rule + shape-sd < 0.02 links dropped; label-based) | SD_SH_r4 | B1_r4 | 0.1685 | 0.0540 | 0.2831 | 690.0000 | SENSITIVITY |
| PRIMARY_OFS (bit rule + offset / near-identity links dropped) | OFS_SH_r4 | B1_r4 | 0.1612 | 0.0503 | 0.2720 | 652.0000 | SENSITIVITY |
| PRIMARY_QNT (bit rule + quantised cells dropped as targets and siblings) | QNT_SH_r4 | B1_r4 | 0.1654 | 0.0192 | 0.3116 | 636.0000 | SENSITIVITY |
| WITH_C5_NOBIT | NB_SH_r4 | B1_r4 | 0.1725 | 0.1238 | 0.2213 | 838.0000 | SENSITIVITY |
| WITH_C5 | SH_r4 | B1_r4 | 0.1682 | 0.1163 | 0.2201 | 838.0000 | SENSITIVITY |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_OFF_ALL | 0.0081 | -0.0114 | 0.0275 | 652.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_OFF_XEXT | 0.0652 | -0.1333 | 0.2636 | 152.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_PB_PUB | 0.0582 | -0.1676 | 0.2841 | 152.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | E_OFS_PB_PUB_CM | 0.0609 | -0.1542 | 0.2761 | 152.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_PB_XEXT | 0.2144 | -0.1747 | 0.6035 | 415.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_PB_CHEM | 0.1820 | 0.0030 | 0.3610 | 535.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_PB_LABX | 0.6102 | -0.1500 | 1.3705 | 285.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_NEAR_r4 | 0.1474 | -0.0704 | 0.3651 | 652.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
| PRIMARY_OFS (control pools) | OFS_SH_r4 | OFS_NEARSH_r4 | 0.0472 | -0.0199 | 0.1144 | 652.0000 | SENSITIVITY (duplicate sensitivity: offset / near-identity links removed from every pool, OFS_*) |
