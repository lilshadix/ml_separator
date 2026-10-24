# 1.2 M03 sibling-weighting factorial (BPG): per-item table

**Label: EXPLORATORY (seen); E-CT value: never placed next to E-A's 0.452; B1 on the same rows is the comparator**

| field | value |
|---|---|
| Item id and estimand | 1.2 M03 sibling-weighting factorial (E-CT annex) |
| Population (cells, publications, units, rows per seed) | cells: 99; publications: 12; units: 14; rows_per_seed: 690.0000; cpg_clusters: 3; kish_cpg: 1.2500 |
| Design | BPG |
| Read budget | 4 reads (code k = 3) for every kernel, 2-nearest and context arm; acid-trend arms 0 reads (labelled) |
| Arms on identical rows | KW grid (5 acid x 5 read lengths, raw and count-shrunk; uniform limit = SIB / SH), NEST_ACID / READ / BOTH (sh, raw; one-SE rules paired, classic, argmin), NEAR2, LEVEL2, READ2 (raw, sh), E_RAND2, B1, SH, SIB, COPY, SH0, NEAR, NEARSH; ACOPY (MEAN), AINT, PAIRMEAN, LIN, LINCLIP, LINHALF, LINHALFCLIP, POOLPAIR |
| R², MSE, relative SSE vs B1 | see 'R², MSE, relative SSE vs B1' below |
| Per-seed values | NEST_BOTH_sh - SH_r4 per seed (PRIMARY): +0.0249;+0.0279;+0.0212;+0.0213;+0.0251 |
| Unit W/L/T and the descriptive tests | see 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | dR2: 0.0241; lopo_renested_lo: -0.0223; lopo_renested_hi: 0.0705; lopo_rowdrop_lo: 0.0082; lopo_rowdrop_hi: 0.0400; top3_share: 0.8866; p_signflip_unit: 0.0003 |
| C5-BTBP in / out | PRIMARY NEST_BOTH_sh - SH_r4: 0.0241; WITH_C5 NEST_BOTH_sh - SH_r4: 0.0499; WITH_C5_NO_C5CELL NEST_BOTH_sh - SH_r4: 0.0452; PRIMARY_NO_ORNL NEST_BOTH_sh - SH_r4: 0.0203 |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | intervention passed (bit: 570 groups, positive controls 570/570/570/570 of 570, NB: 495 groups, positive controls 495/n.a./495/495 of 495, OFS: 475 groups, positive controls 475/475/475/475 of 475); fit poisoning 1512/1512 selections, 48/48 POOLPAIR slopes identical, leaky positive control caught 1512/1512; B1 max diff 8.9e-16; r5.ect identity on 570 groups |
| Compute | 339 s (build + nulls + tables); seal b96c3bef9828 |
| Label | EXPLORATORY (seen); E-CT value: never placed next to E-A's 0.452; B1 on the same rows is the comparator |
| What changed scientifically | On PRIMARY_NO_ORNL (the deploy-rule population, BPG) the registered deploy rule gives 'nested weighting passes the in-corpus consistency check (EXPLORATORY; no claim)' (dR2 NEST_BOTH_sh - SH_r4 +0.0203, 9 of 11 publications improve, uniform chosen in 0 of 11 folds), against the expected 'uniform shrunk mean'; the read-match kernel carries it (ACID family alone: deploy SH, dR2 -0.0028; READ family alone: passes, dR2 +0.0246; acid-permutation null p 0.964, read-permutation null p 0.002), under the classic one-SE rule (SENSITIVITY) uniform is chosen in 11 of 11 folds (dR2 +0.0000), and the re-nested leave-one-publication-out minimum is -0.0010 without pub_c8195b4ca7 (every rule outcome: rules_BPG.csv). |
| Next registered step | C-ECT stays sealed on SH@4 vs B1@4 (CONFIRMATION_PACKAGE; new laboratories, section 7.5, data-gated); nested read-match weighting passed in-corpus only (EXPLORATORY; no claim) and would need its own registration on new data. |

## R², MSE, relative SSE vs B1

| arm | R2 | MSE | relSSE_vs_B1_r4 | R2_per_seed | rows_per_seed | label |
|---|---|---|---|---|---|---|
| B1_r4 | 0.5309 | 0.0121 | 0.0000 | 0.5234;0.5519;0.5294;0.5387;0.5109 | 690.0000 | EXPLORATORY (seen) |
| SH_r4 | 0.7002 | 0.0077 | 0.3604 | 0.6929;0.6949;0.7098;0.7090;0.6943 | 690.0000 | EXPLORATORY (seen) |
| SIB_r4 | 0.6862 | 0.0081 | 0.3306 | 0.6807;0.6796;0.6948;0.6935;0.6826 | 690.0000 | EXPLORATORY (seen) |
| COPY | 0.4831 | 0.0133 | -0.1027 | 0.4831;0.4831;0.4831;0.4831;0.4831 | 690.0000 | EXPLORATORY (seen) |
| SH0 | 0.4741 | 0.0135 | -0.1219 | 0.4686;0.4760;0.4748;0.4772;0.4738 | 690.0000 | EXPLORATORY (seen) |
| NEAR_r4 | 0.5539 | 0.0115 | 0.0485 | 0.5344;0.5321;0.5838;0.5834;0.5359 | 690.0000 | EXPLORATORY (seen) |
| NEARSH_r4 | 0.6501 | 0.0090 | 0.2538 | 0.6407;0.6510;0.6595;0.6635;0.6356 | 690.0000 | EXPLORATORY (seen) |
| NEAR2_raw | 0.6662 | 0.0086 | 0.2879 | 0.6565;0.6512;0.6830;0.6823;0.6580 | 690.0000 | EXPLORATORY (seen) |
| NEAR2_sh | 0.6837 | 0.0081 | 0.3253 | 0.6761;0.6812;0.6927;0.6937;0.6747 | 690.0000 | EXPLORATORY (seen) |
| LEVEL2_raw | 0.7353 | 0.0068 | 0.4351 | 0.7371;0.7312;0.7353;0.7349;0.7378 | 690.0000 | EXPLORATORY (seen) |
| LEVEL2_sh | 0.7336 | 0.0068 | 0.4319 | 0.7317;0.7335;0.7376;0.7389;0.7264 | 690.0000 | EXPLORATORY (seen) |
| READ2_raw | 0.7190 | 0.0072 | 0.4007 | 0.7161;0.7398;0.7086;0.7079;0.7223 | 690.0000 | EXPLORATORY (seen) |
| READ2_sh | 0.7251 | 0.0071 | 0.4138 | 0.7228;0.7348;0.7226;0.7238;0.7215 | 690.0000 | EXPLORATORY (seen) |
| E_RAND2_raw_r4 | 0.6014 | 0.0102 | 0.1497 | 0.5928;0.5914;0.6145;0.6133;0.5950 | 690.0000 | EXPLORATORY (seen) |
| E_RAND2_sh_r4 | 0.6559 | 0.0088 | 0.2661 | 0.6488;0.6549;0.6638;0.6650;0.6469 | 690.0000 | EXPLORATORY (seen) |
| NEST_ACID_sh | 0.7002 | 0.0077 | 0.3604 | 0.6929;0.6949;0.7098;0.7090;0.6943 | 690.0000 | EXPLORATORY (seen) |
| NEST_ACID_sh_classic | 0.7002 | 0.0077 | 0.3604 | 0.6929;0.6949;0.7098;0.7090;0.6943 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_ACID_sh_argmin | 0.7004 | 0.0077 | 0.3609 | 0.6924;0.6935;0.7116;0.7110;0.6935 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_READ_sh | 0.7263 | 0.0070 | 0.4163 | 0.7206;0.7248;0.7322;0.7318;0.7222 | 690.0000 | EXPLORATORY (seen) |
| NEST_READ_sh_classic | 0.7002 | 0.0077 | 0.3604 | 0.6929;0.6949;0.7098;0.7090;0.6943 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_READ_sh_argmin | 0.7199 | 0.0072 | 0.4025 | 0.7166;0.7202;0.7226;0.7226;0.7175 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_BOTH_sh | 0.7242 | 0.0071 | 0.4118 | 0.7178;0.7227;0.7309;0.7303;0.7194 | 690.0000 | EXPLORATORY (seen) |
| NEST_BOTH_sh_classic | 0.7002 | 0.0077 | 0.3604 | 0.6929;0.6949;0.7098;0.7090;0.6943 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_BOTH_sh_argmin | 0.7235 | 0.0071 | 0.4101 | 0.7203;0.7174;0.7292;0.7291;0.7216 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_ACID_raw | 0.6862 | 0.0081 | 0.3306 | 0.6807;0.6796;0.6948;0.6935;0.6826 | 690.0000 | EXPLORATORY (seen) |
| NEST_ACID_raw_classic | 0.6862 | 0.0081 | 0.3306 | 0.6807;0.6796;0.6948;0.6935;0.6826 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_ACID_raw_argmin | 0.6624 | 0.0087 | 0.2799 | 0.6533;0.6504;0.6775;0.6763;0.6547 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_READ_raw | 0.7187 | 0.0072 | 0.4001 | 0.7130;0.7198;0.7232;0.7219;0.7158 | 690.0000 | EXPLORATORY (seen) |
| NEST_READ_raw_classic | 0.6862 | 0.0081 | 0.3306 | 0.6807;0.6796;0.6948;0.6935;0.6826 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_READ_raw_argmin | 0.7135 | 0.0074 | 0.3887 | 0.7128;0.7148;0.7113;0.7100;0.7186 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_BOTH_raw | 0.7218 | 0.0072 | 0.4066 | 0.7169;0.7239;0.7251;0.7238;0.7192 | 690.0000 | EXPLORATORY (seen) |
| NEST_BOTH_raw_classic | 0.6950 | 0.0078 | 0.3493 | 0.6891;0.6894;0.7035;0.7024;0.6904 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_BOTH_raw_argmin | 0.7593 | 0.0062 | 0.4862 | 0.7565;0.7448;0.7672;0.7658;0.7623 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |

## Unit W/L/T and the descriptive tests

| stratum | contrast | G_units | SIGN_W/L/T | SIGN_p | p_signflip_unit | signflip_unit_exact | p_signflip_chemotype_blocks | G_chemotypes | IMt_rho_p | IMt_log_p | eqw_pub_relSSE | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | NEST_BOTH_sh - SH_r4 | 14 | 13/1/0 | 0.0009 | 0.0003 | True | 0.0156 | 6 | 0.0008 | 0.0009 | 0.0493 | EXPLORATORY (seen) |
| PRIMARY_NO_ORNL | NEST_BOTH_sh - SH_r4 | 13 | 10/3/0 | 0.0461 | 0.0519 | True | 0.0312 | 6 | 0.0070 | 0.0070 | 0.0512 | SENSITIVITY |

c_u per publication x chemotype unit, NEST_BOTH_sh - SH_r4 on PRIMARY (units_BPG.csv):

| publication | chemotype | c | rows | rho | logratio |
|---|---|---|---|---|---|
| pub_19783c039b | sc009 | 0.0098 | 109.0000 | 0.0900 | 0.0943 |
| pub_1605e436d7 | sc009 | 0.0068 | 84.0000 | 0.0972 | 0.1022 |
| pub_b0b3296ec5 | sc009 | 0.0048 | 152.0000 | 0.1105 | 0.1171 |
| pub_4cba27f5f3 | sc009 | 0.0015 | 36.0000 | 0.1345 | 0.1444 |
| pub_d68ed8c8cf | sc009 | 0.0003 | 21.0000 | 0.0257 | 0.0260 |
| pub_c8195b4ca7 | sc009 | 0.0003 | 38.0000 | 0.0179 | 0.0180 |
| pub_8415538291 | sc077 | 0.0002 | 60.0000 | 0.0101 | 0.0102 |
| pub_f8a621ede4 | sc009 | 0.0001 | 9.0000 | 0.0618 | 0.0638 |
| pub_1c1b558856 | sc027 | 0.0001 | 30.0000 | 0.0240 | 0.0243 |
| pub_5a68dc5665 | sc071 | 0.0000 | 42.0000 | 0.0668 | 0.0691 |
| pub_5a68dc5665 | sc076 | 0.0000 | 28.0000 | 0.0479 | 0.0491 |
| pub_0a8da089ad | sc009 | 0.0000 | 3.0000 | 0.0062 | 0.0062 |
| pub_cd72e4a7bb | sc020 | 0.0000 | 15.0000 | 0.0029 | 0.0029 |
| pub_5a68dc5665 | sc009 | -0.0000 | 63.0000 | -0.0134 | -0.0133 |

## Duplicate sensitivities

| stratum | dR2 | lopo_renested_lo | lopo_renested_hi | lopo_renested_min | lopo_renested_min_by | lopo_renested_max | lopo_renested_max_by | rows_per_seed | label |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | 0.0241 | -0.0223 | 0.0705 | 0.0117 | pub_b0b3296ec5 | 0.0389 | pub_c8195b4ca7 | 690.0000 | EXPLORATORY (seen) |
| PRIMARY_NOBIT | 0.0237 | -0.0267 | 0.0740 | 0.0092 | pub_b0b3296ec5 | 0.0382 | pub_c8195b4ca7 | 690.0000 | SENSITIVITY |
| PRIMARY_OFS | 0.0267 | -0.0158 | 0.0693 | 0.0141 | pub_1605e436d7 | 0.0378 | pub_c8195b4ca7 | 652.0000 | SENSITIVITY |
| PRIMARY_NO_ORNL_OFS | 0.0275 | -0.0491 | 0.1041 | -0.0015 | pub_c8195b4ca7 | 0.0303 | pub_5a68dc5665 | 543.0000 | SENSITIVITY |
