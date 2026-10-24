# 1.2 M03 sibling-weighting factorial (BP): per-item table

**Label: EXPLORATORY (seen); E-CT value: never placed next to E-A's 0.452; B1 on the same rows is the comparator**

| field | value |
|---|---|
| Item id and estimand | 1.2 M03 sibling-weighting factorial (E-CT annex) |
| Population (cells, publications, units, rows per seed) | cells: 99; publications: 12; units: 14; rows_per_seed: 690.0000; cpg_clusters: 3; kish_cpg: 1.2500 |
| Design | BP |
| Read budget | 4 reads (code k = 3) for every kernel, 2-nearest and context arm; acid-trend arms 0 reads (labelled) |
| Arms on identical rows | KW grid (5 acid x 5 read lengths, raw and count-shrunk; uniform limit = SIB / SH), NEST_ACID / READ / BOTH (sh, raw; one-SE rules paired, classic, argmin), NEAR2, LEVEL2, READ2 (raw, sh), E_RAND2, B1, SH, SIB, COPY, SH0, NEAR, NEARSH; ACOPY (MEAN), AINT, PAIRMEAN, LIN, LINCLIP, LINHALF, LINHALFCLIP, POOLPAIR |
| R², MSE, relative SSE vs B1 | see 'R², MSE, relative SSE vs B1' below |
| Per-seed values | NEST_BOTH_sh - SH_r4 per seed (PRIMARY): +0.0249;+0.0279;+0.0216;+0.0213;+0.0210 |
| Unit W/L/T and the descriptive tests | see 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | dR2: 0.0234; lopo_renested_lo: -0.0187; lopo_renested_hi: 0.0654; lopo_rowdrop_lo: 0.0082; lopo_rowdrop_hi: 0.0385; top3_share: 0.8833; p_signflip_unit: 0.0003 |
| C5-BTBP in / out | PRIMARY NEST_BOTH_sh - SH_r4: 0.0234; WITH_C5 NEST_BOTH_sh - SH_r4: 0.0495; WITH_C5_NO_C5CELL NEST_BOTH_sh - SH_r4: 0.0447; PRIMARY_NO_ORNL NEST_BOTH_sh - SH_r4: 0.0197 |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | intervention passed (bit: 570 groups, positive controls 570/570/570/570 of 570, NB: 495 groups, positive controls 495/n.a./495/495 of 495, OFS: 475 groups, positive controls 475/475/475/475 of 475); fit poisoning 1512/1512 selections, 48/48 POOLPAIR slopes identical, leaky positive control caught 1512/1512; B1 max diff 8.9e-16; r5.ect identity on 570 groups |
| Compute | 337 s (build + nulls + tables); seal b96c3bef9828 |
| Label | EXPLORATORY (seen); E-CT value: never placed next to E-A's 0.452; B1 on the same rows is the comparator |
| What changed scientifically | On PRIMARY_NO_ORNL (the deploy-rule population, BP) the registered deploy rule gives 'nested weighting passes the in-corpus consistency check (EXPLORATORY; no claim)' (dR2 NEST_BOTH_sh - SH_r4 +0.0197, 9 of 11 publications improve, uniform chosen in 0 of 11 folds), against the expected 'uniform shrunk mean'; the read-match kernel carries it (ACID family alone: deploy SH, dR2 -0.0028; READ family alone: passes, dR2 +0.0241; acid-permutation null p 0.962, read-permutation null p 0.002), under the classic one-SE rule (SENSITIVITY) uniform is chosen in 11 of 11 folds (dR2 +0.0000), and the re-nested leave-one-publication-out minimum is -0.0012 without pub_c8195b4ca7 (every rule outcome: rules_BP.csv). |
| Next registered step | C-ECT stays sealed on SH@4 vs B1@4 (CONFIRMATION_PACKAGE; new laboratories, section 7.5, data-gated); nested read-match weighting passed in-corpus only (EXPLORATORY; no claim) and would need its own registration on new data. |

## R², MSE, relative SSE vs B1

| arm | R2 | MSE | relSSE_vs_B1_r4 | R2_per_seed | rows_per_seed | label |
|---|---|---|---|---|---|---|
| B1_r4 | 0.5316 | 0.0120 | 0.0000 | 0.5230;0.5548;0.5053;0.5394;0.5356 | 690.0000 | EXPLORATORY (seen) |
| SH_r4 | 0.6974 | 0.0078 | 0.3533 | 0.6928;0.6954;0.6941;0.7091;0.6956 | 690.0000 | EXPLORATORY (seen) |
| SIB_r4 | 0.6832 | 0.0081 | 0.3229 | 0.6807;0.6796;0.6791;0.6935;0.6831 | 690.0000 | EXPLORATORY (seen) |
| COPY | 0.4831 | 0.0133 | -0.1049 | 0.4831;0.4831;0.4831;0.4831;0.4831 | 690.0000 | EXPLORATORY (seen) |
| SH0 | 0.4741 | 0.0135 | -0.1241 | 0.4687;0.4760;0.4748;0.4772;0.4738 | 690.0000 | EXPLORATORY (seen) |
| NEAR_r4 | 0.5463 | 0.0117 | 0.0303 | 0.5344;0.5321;0.5400;0.5834;0.5415 | 690.0000 | EXPLORATORY (seen) |
| NEARSH_r4 | 0.6485 | 0.0090 | 0.2491 | 0.6405;0.6522;0.6392;0.6639;0.6469 | 690.0000 | EXPLORATORY (seen) |
| NEAR2_raw | 0.6605 | 0.0087 | 0.2743 | 0.6565;0.6512;0.6561;0.6823;0.6563 | 690.0000 | EXPLORATORY (seen) |
| NEAR2_sh | 0.6806 | 0.0082 | 0.3175 | 0.6760;0.6819;0.6740;0.6938;0.6771 | 690.0000 | EXPLORATORY (seen) |
| LEVEL2_raw | 0.7309 | 0.0069 | 0.4252 | 0.7371;0.7312;0.7176;0.7349;0.7338 | 690.0000 | EXPLORATORY (seen) |
| LEVEL2_sh | 0.7323 | 0.0069 | 0.4280 | 0.7317;0.7342;0.7227;0.7391;0.7338 | 690.0000 | EXPLORATORY (seen) |
| READ2_raw | 0.7151 | 0.0073 | 0.3918 | 0.7161;0.7398;0.6971;0.7079;0.7144 | 690.0000 | EXPLORATORY (seen) |
| READ2_sh | 0.7240 | 0.0071 | 0.4105 | 0.7228;0.7355;0.7108;0.7240;0.7268 | 690.0000 | EXPLORATORY (seen) |
| E_RAND2_raw_r4 | 0.5976 | 0.0103 | 0.1398 | 0.5928;0.5914;0.5943;0.6133;0.5961 | 690.0000 | EXPLORATORY (seen) |
| E_RAND2_sh_r4 | 0.6537 | 0.0089 | 0.2602 | 0.6487;0.6556;0.6461;0.6652;0.6532 | 690.0000 | EXPLORATORY (seen) |
| NEST_ACID_sh | 0.6974 | 0.0078 | 0.3533 | 0.6928;0.6954;0.6941;0.7091;0.6956 | 690.0000 | EXPLORATORY (seen) |
| NEST_ACID_sh_classic | 0.6974 | 0.0078 | 0.3533 | 0.6928;0.6954;0.6941;0.7091;0.6956 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_ACID_sh_argmin | 0.6974 | 0.0078 | 0.3533 | 0.6923;0.6941;0.6950;0.7111;0.6945 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_READ_sh | 0.7229 | 0.0071 | 0.4079 | 0.7206;0.7254;0.7173;0.7320;0.7191 | 690.0000 | EXPLORATORY (seen) |
| NEST_READ_sh_classic | 0.6974 | 0.0078 | 0.3533 | 0.6928;0.6954;0.6941;0.7091;0.6956 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_READ_sh_argmin | 0.7168 | 0.0073 | 0.3949 | 0.7166;0.7210;0.7077;0.7228;0.7157 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_BOTH_sh | 0.7208 | 0.0072 | 0.4033 | 0.7177;0.7233;0.7157;0.7305;0.7166 | 690.0000 | EXPLORATORY (seen) |
| NEST_BOTH_sh_classic | 0.6974 | 0.0078 | 0.3533 | 0.6928;0.6954;0.6941;0.7091;0.6956 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_BOTH_sh_argmin | 0.7190 | 0.0072 | 0.3995 | 0.7185;0.7173;0.7126;0.7290;0.7177 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_ACID_raw | 0.6832 | 0.0081 | 0.3229 | 0.6807;0.6796;0.6791;0.6935;0.6831 | 690.0000 | EXPLORATORY (seen) |
| NEST_ACID_raw_classic | 0.6832 | 0.0081 | 0.3229 | 0.6807;0.6796;0.6791;0.6935;0.6831 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_ACID_raw_argmin | 0.6583 | 0.0088 | 0.2696 | 0.6533;0.6504;0.6543;0.6763;0.6570 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_READ_raw | 0.7184 | 0.0072 | 0.3984 | 0.7180;0.7248;0.7106;0.7246;0.7139 | 690.0000 | EXPLORATORY (seen) |
| NEST_READ_raw_classic | 0.6832 | 0.0081 | 0.3229 | 0.6807;0.6796;0.6791;0.6935;0.6831 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_READ_raw_argmin | 0.7122 | 0.0074 | 0.3853 | 0.7145;0.7209;0.6998;0.7117;0.7143 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_BOTH_raw | 0.7177 | 0.0073 | 0.3968 | 0.7169;0.7239;0.7089;0.7238;0.7148 | 690.0000 | EXPLORATORY (seen) |
| NEST_BOTH_raw_classic | 0.6915 | 0.0079 | 0.3406 | 0.6889;0.6891;0.6855;0.7020;0.6918 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |
| NEST_BOTH_raw_argmin | 0.7537 | 0.0063 | 0.4737 | 0.7565;0.7448;0.7423;0.7658;0.7593 | 690.0000 | EXPLORATORY (seen); SENSITIVITY (one-SE rule) |

## Unit W/L/T and the descriptive tests

| stratum | contrast | G_units | SIGN_W/L/T | SIGN_p | p_signflip_unit | signflip_unit_exact | p_signflip_chemotype_blocks | G_chemotypes | IMt_rho_p | IMt_log_p | eqw_pub_relSSE | label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | NEST_BOTH_sh - SH_r4 | 14 | 13/1/0 | 0.0009 | 0.0003 | True | 0.0156 | 6 | 0.0007 | 0.0008 | 0.0484 | EXPLORATORY (seen) |
| PRIMARY_NO_ORNL | NEST_BOTH_sh - SH_r4 | 13 | 10/3/0 | 0.0461 | 0.0503 | True | 0.0312 | 6 | 0.0065 | 0.0066 | 0.0509 | SENSITIVITY |

c_u per publication x chemotype unit, NEST_BOTH_sh - SH_r4 on PRIMARY (units_BP.csv):

| publication | chemotype | c | rows | rho | logratio |
|---|---|---|---|---|---|
| pub_19783c039b | sc009 | 0.0096 | 109.0000 | 0.0877 | 0.0918 |
| pub_1605e436d7 | sc009 | 0.0063 | 84.0000 | 0.0872 | 0.0913 |
| pub_b0b3296ec5 | sc009 | 0.0048 | 152.0000 | 0.1101 | 0.1167 |
| pub_4cba27f5f3 | sc009 | 0.0016 | 36.0000 | 0.1356 | 0.1457 |
| pub_d68ed8c8cf | sc009 | 0.0003 | 21.0000 | 0.0255 | 0.0259 |
| pub_c8195b4ca7 | sc009 | 0.0003 | 38.0000 | 0.0167 | 0.0169 |
| pub_8415538291 | sc077 | 0.0002 | 60.0000 | 0.0101 | 0.0102 |
| pub_f8a621ede4 | sc009 | 0.0001 | 9.0000 | 0.0618 | 0.0637 |
| pub_1c1b558856 | sc027 | 0.0001 | 30.0000 | 0.0240 | 0.0242 |
| pub_5a68dc5665 | sc071 | 0.0000 | 42.0000 | 0.0672 | 0.0696 |
| pub_5a68dc5665 | sc076 | 0.0000 | 28.0000 | 0.0467 | 0.0479 |
| pub_cd72e4a7bb | sc020 | 0.0000 | 15.0000 | 0.0055 | 0.0055 |
| pub_0a8da089ad | sc009 | 0.0000 | 3.0000 | 0.0062 | 0.0062 |
| pub_5a68dc5665 | sc009 | -0.0000 | 63.0000 | -0.0134 | -0.0133 |

## Duplicate sensitivities

| stratum | dR2 | lopo_renested_lo | lopo_renested_hi | lopo_renested_min | lopo_renested_min_by | lopo_renested_max | lopo_renested_max_by | rows_per_seed | label |
|---|---|---|---|---|---|---|---|---|---|
| PRIMARY | 0.0234 | -0.0187 | 0.0654 | 0.0114 | pub_b0b3296ec5 | 0.0360 | pub_c8195b4ca7 | 690.0000 | EXPLORATORY (seen) |
| PRIMARY_NOBIT | 0.0231 | -0.0224 | 0.0687 | 0.0091 | pub_b0b3296ec5 | 0.0354 | pub_c8195b4ca7 | 690.0000 | SENSITIVITY |
| PRIMARY_OFS | 0.0259 | -0.0153 | 0.0671 | 0.0139 | pub_1605e436d7 | 0.0370 | pub_c8195b4ca7 | 652.0000 | SENSITIVITY |
| PRIMARY_NO_ORNL_OFS | 0.0268 | -0.0600 | 0.1136 | -0.0075 | pub_c8195b4ca7 | 0.0296 | pub_5a68dc5665 | 543.0000 | SENSITIVITY |
