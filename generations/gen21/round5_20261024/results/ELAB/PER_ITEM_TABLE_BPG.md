# 1.3 E-LAB (BPG): per-item table

**Label: EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression; E-LAB (campaign calibration of a new extractant; donors = the campaign's other extractants' full series): never placed next to E-A's 0.452; B1 on the same rows is the comparator; seal b96c3bef9828**

| field | value |
|---|---|
| Item id and estimand | 1.3 E-LAB (lab-1): campaign calibration of a new extractant (E-CT annex) |
| Population (cells, publications, units, rows per seed) | PRIMARY (POPX, near-copies excluded): 92 cells, 16 publications, 9 / 14 laboratories (conservative / liberal, PROVISIONAL key), 4 source clusters, 787 rows per seed |
| Design | BPG (primary) |
| Read budget | 4 reads (DOPT_Q, code budget k = 3; E-A's read sets); the campaign's other series in full |
| Arms on identical rows | B1, MX_LM10, MX_LM05, MX_LM20, MX_K6p1, MX_K12, MX_K20, MX_K35, MX_K60, MX_K100, MX_LM10_SS, MX_LM10_OS, MX_LM10_SM, MX_LM10_DM, MX_LM10_LO, MX_LM10_HI, MX_LM10_NB, MX_LM10_NC, MX_LM10_OFS, MX_LM10_NC05, MX_LM10_NC03, E_RM, E_PBC, E_PBA, E_PCM, E_PCMC, MX_GK_LOPO, MX_GK_O |
| R², MSE, relative SSE vs B1 | see 'R², MSE, relative SSE vs B1' below |
| Per-seed values | see 'Per-seed values' below |
| Unit W/L/T and the descriptive tests | see 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | see 'LOPO and top-3 share' below |
| C5-BTBP in / out | see 'C5-BTBP in / out' below |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | intervention passed on 460 target-seeds (92 targets); positive controls self 92/92, bit copy 2/2, leak 92/92; label-based sensitivities changed 170 times (declared); LOPO-K poisoning passed (leaky control caught 14/16); B1 identity 8.9e-16 |
| Compute | 229 s (one process) |
| Label | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression; E-LAB (campaign calibration of a new extractant; donors = the campaign's other extractants' full series): never placed next to E-A's 0.452; B1 on the same rows is the comparator; seal b96c3bef9828 |
| What changed scientifically | On the near-copy-cleaned POPX the level-matched campaign shape gives +0.0763 relative SSE vs B1 on identical rows, with laboratory (conservative, PROVISIONAL key) W/L/T 9/0/0 and SF_POOL p 0.0020 (publication units, the bridge to lab-1: seen before cleaning +0.090, after +0.076); it remains a seen, best-of-about-25 exploratory gain whose mechanism (campaign vs level-driven weighting) the controls only bound. |
| Next registered step | External leg data-gated (section 7): descriptive unless >= 8 independent laboratories with >= 2 extractants x >= 6 Ln in one campaign; no in-corpus kill fires. |

## R², MSE, relative SSE vs B1

| arm | R2 | MSE | relSSE_vs_B1 | MAE_balanced | label |
|---|---|---|---|---|---|
| B1 | 0.4437 | 0.0255 | 0.0000 | 0.1080 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 | 0.4862 | 0.0235 | 0.0763 | 0.1017 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM05 | 0.4839 | 0.0236 | 0.0721 | 0.1029 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM20 | 0.4776 | 0.0239 | 0.0608 | 0.1039 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_K6p1 | 0.4595 | 0.0247 | 0.0282 | 0.1054 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_K12 | 0.4635 | 0.0246 | 0.0354 | 0.1054 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_K20 | 0.4619 | 0.0246 | 0.0325 | 0.1057 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_K35 | 0.4580 | 0.0248 | 0.0256 | 0.1061 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_K60 | 0.4539 | 0.0250 | 0.0183 | 0.1066 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_K100 | 0.4507 | 0.0252 | 0.0124 | 0.1070 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10_LO | 0.4417 | 0.0256 | -0.0036 | 0.1082 | DIAGNOSTIC; EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10_HI | 0.4882 | 0.0234 | 0.0799 | 0.1015 | DIAGNOSTIC; EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10_NB | 0.4936 | 0.0232 | 0.0896 | 0.1009 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC | 0.4862 | 0.0235 | 0.0763 | 0.1017 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_OFS | 0.4830 | 0.0237 | 0.0706 | 0.1020 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC05 | 0.4913 | 0.0233 | 0.0854 | 0.1014 | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC03 | 0.4921 | 0.0233 | 0.0868 | 0.1013 | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_RM | 0.4769 | 0.0240 | 0.0595 | NaN | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PBC | 0.4349 | 0.0259 | -0.0159 | NaN | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PBA | 0.4362 | 0.0258 | -0.0136 | NaN | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PCM | 0.4290 | 0.0261 | -0.0266 | NaN | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PCMC | 0.4384 | 0.0257 | -0.0097 | NaN | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_GK_LOPO | 0.4509 | 0.0251 | 0.0128 | 0.1064 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_GK_O | 0.4635 | 0.0246 | 0.0354 | 0.1054 | ORACLE [O]; EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |

## Per-seed values

| arm | R2_per_seed | rel_per_seed |
|---|---|---|
| B1 | 0.4485;0.4307;0.4512;0.4348;0.4535 | 0.0000;0.0000;0.0000;0.0000;0.0000 |
| MX_LM10 | 0.4886;0.4761;0.4908;0.4808;0.4947 | 0.0727;0.0798;0.0722;0.0814;0.0755 |
| MX_LM05 | 0.4877;0.4781;0.4883;0.4755;0.4899 | 0.0710;0.0834;0.0677;0.0720;0.0666 |
| MX_LM20 | 0.4805;0.4712;0.4835;0.4708;0.4820 | 0.0580;0.0712;0.0589;0.0638;0.0522 |
| MX_K6p1 | 0.4618;0.4512;0.4665;0.4540;0.4640 | 0.0241;0.0360;0.0278;0.0340;0.0192 |
| MX_K12 | 0.4663;0.4538;0.4703;0.4571;0.4699 | 0.0322;0.0406;0.0348;0.0395;0.0300 |
| MX_K20 | 0.4651;0.4512;0.4688;0.4549;0.4693 | 0.0301;0.0360;0.0321;0.0356;0.0290 |
| MX_K35 | 0.4617;0.4464;0.4651;0.4504;0.4662 | 0.0239;0.0277;0.0253;0.0277;0.0234 |
| MX_K60 | 0.4580;0.4418;0.4612;0.4459;0.4627 | 0.0171;0.0195;0.0181;0.0197;0.0169 |
| MX_K100 | 0.4550;0.4382;0.4580;0.4423;0.4598 | 0.0117;0.0132;0.0124;0.0133;0.0116 |
| MX_LM10_LO | 0.4463;0.4289;0.4486;0.4334;0.4515 | -0.0040;-0.0031;-0.0048;-0.0024;-0.0037 |
| MX_LM10_HI | 0.4909;0.4779;0.4935;0.4822;0.4967 | 0.0768;0.0830;0.0770;0.0839;0.0792 |
| MX_LM10_NB | 0.4959;0.4835;0.4984;0.4893;0.5009 | 0.0859;0.0928;0.0859;0.0965;0.0867 |
| MX_LM10_NC | 0.4886;0.4761;0.4908;0.4808;0.4947 | 0.0727;0.0798;0.0722;0.0814;0.0755 |
| MX_LM10_OFS | 0.4851;0.4729;0.4879;0.4779;0.4912 | 0.0664;0.0742;0.0669;0.0762;0.0690 |
| MX_LM10_NC05 | 0.4937;0.4812;0.4961;0.4870;0.4986 | 0.0819;0.0887;0.0817;0.0923;0.0826 |
| MX_LM10_NC03 | 0.4944;0.4820;0.4968;0.4877;0.4994 | 0.0832;0.0902;0.0831;0.0937;0.0840 |
| E_RM | 0.4788;0.4667;0.4845;0.4714;0.4829 | 0.0548;0.0633;0.0607;0.0648;0.0538 |
| E_PBC | 0.4372;0.4247;0.4421;0.4261;0.4444 | -0.0205;-0.0104;-0.0167;-0.0153;-0.0166 |
| E_PBA | 0.4422;0.4247;0.4449;0.4242;0.4448 | -0.0114;-0.0105;-0.0114;-0.0188;-0.0158 |
| E_PCM | 0.4344;0.4189;0.4366;0.4200;0.4349 | -0.0256;-0.0206;-0.0267;-0.0262;-0.0340 |
| E_PCMC | 0.4424;0.4282;0.4450;0.4281;0.4480 | -0.0111;-0.0043;-0.0113;-0.0118;-0.0100 |
| MX_GK_LOPO | 0.4543;0.4408;0.4582;0.4438;0.4575 | 0.0104;0.0177;0.0127;0.0159;0.0073 |
| MX_GK_O | 0.4663;0.4538;0.4703;0.4571;0.4699 | 0.0322;0.0406;0.0348;0.0395;0.0300 |

## Unit W/L/T and the descriptive tests

| contrast | unit | G | W/L/T | SF_POOL_p | SIGN_p | IM_REL_p | unit_median_rel | label |
|---|---|---|---|---|---|---|---|---|
| MX_LM10 - B1 | publication | 16 | 10/4/2 | 0.0966 | 0.0898 | 0.0311 | 0.0493 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - B1 | lab_cons | 9 | 9/0/0 | 0.0020 | 0.0020 | 0.0098 | 0.0706 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - B1 | lab_lib | 14 | 10/2/2 | 0.0450 | 0.0193 | 0.0373 | 0.0493 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - B1 | cluster_cpg | 4 | 3/1/0 | 0.1250 | 0.3125 | 0.0695 | 0.1495 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PBC | publication | 16 | 13/3/0 | 0.0358 | 0.0106 | 0.0195 | 0.0546 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PBC | lab_cons | 9 | 8/1/0 | 0.0039 | 0.0195 | 0.0063 | 0.0862 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PBC | lab_lib | 14 | 12/2/0 | 0.0186 | 0.0065 | 0.0228 | 0.0654 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PBC | cluster_cpg | 4 | 4/0/0 | 0.0625 | 0.0625 | 0.0569 | 0.1548 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PCM | publication | 16 | 11/4/1 | 0.0194 | 0.0592 | 0.0134 | 0.0540 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PCM | lab_cons | 9 | 8/1/0 | 0.0039 | 0.0195 | 0.0043 | 0.0869 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PCM | lab_lib | 14 | 10/3/1 | 0.0062 | 0.0461 | 0.0150 | 0.0696 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PCM | cluster_cpg | 4 | 4/0/0 | 0.0625 | 0.0625 | 0.0477 | 0.1581 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_RM | publication | 16 | 10/2/4 | 0.0346 | 0.0193 | 0.0072 | 0.0017 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_RM | lab_cons | 9 | 8/0/1 | 0.0078 | 0.0039 | 0.0188 | 0.0070 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_RM | lab_lib | 14 | 9/1/4 | 0.0036 | 0.0107 | 0.0052 | 0.0056 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_RM | cluster_cpg | 4 | 2/1/1 | 0.3750 | 0.5000 | 0.2043 | 0.0012 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PBC - B1 | publication | 16 | 5/11/0 | 0.9908 | 0.9616 | 0.8659 | -0.0081 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PBC - B1 | lab_cons | 9 | 4/5/0 | 0.9316 | 0.7461 | 0.8577 | -0.0062 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PBC - B1 | lab_lib | 14 | 4/10/0 | 0.9774 | 0.9713 | 0.8665 | -0.0081 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PBC - B1 | cluster_cpg | 4 | 1/3/0 | 0.9375 | 0.9375 | 0.9360 | -0.0155 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PCM - B1 | publication | 16 | 4/11/1 | 0.9992 | 0.9824 | 0.9050 | -0.0153 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PCM - B1 | lab_cons | 9 | 2/7/0 | 0.9902 | 0.9805 | 0.9499 | -0.0311 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PCM - B1 | lab_lib | 14 | 3/10/1 | 0.9966 | 0.9888 | 0.9052 | -0.0213 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| E_PCM - B1 | cluster_cpg | 4 | 1/3/0 | 0.9375 | 0.9375 | 0.9302 | -0.0278 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |

## LOPO and top-3 share

| contrast | rel | LOPO_lo | LOPO_hi | LOPO_min | LOPO_min_without | LTPO_min | without_LEAD2 | top3_share |
|---|---|---|---|---|---|---|---|---|
| MX_LM10 - B1 | 0.0763 | -0.0110 | 0.1637 | 0.0473 | pub_1bd76c48d9 | 0.0061 | 0.0061 | 1.0351 |
| MX_LM10 - E_PBC | 0.0908 | 0.0035 | 0.1781 | 0.0624 | pub_1bd76c48d9 | 0.0216 | 0.0216 | 0.9322 |
| MX_LM10 - E_PCM | 0.1003 | 0.0102 | 0.1903 | 0.0717 | pub_1bd76c48d9 | 0.0337 | 0.0337 | 0.8888 |
| MX_LM10 - E_RM | 0.0179 | -0.0044 | 0.0402 | 0.0090 | pub_126bbe7976 | 0.0028 | 0.0096 | 0.9642 |
| E_PBC - B1 | -0.0159 | -0.0265 | -0.0053 | -0.0178 | pub_74428aeab2 | -0.0197 | -0.0158 | NaN |
| E_PCM - B1 | -0.0266 | -0.0467 | -0.0066 | -0.0299 | pub_74428aeab2 | -0.0346 | -0.0285 | NaN |

## C5-BTBP in / out

| c5 | unit | rel | W/L/T | SF_POOL_p | SF_POOL_p_enum | LOPO_lo | LOPO_hi | equals_headline | point_arms_changed_vs_C5_in | draw_arms_changed_vs_C5_in | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C5_in | publication | 0.0763 | 10/4/2 | 0.0966 | 0.0932 | -0.0110 | 0.1637 | True | 0 | 0 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| C5_in | lab_cons | 0.0763 | 9/0/0 | 0.0020 | NaN | -0.0110 | 0.1637 | True | 0 | 0 | EXPLORATORY (seen); best of about 25 arms on seen rows; expect regression |
| C5_out | publication | 0.0763 | 10/4/2 | 0.0930 | 0.0932 | -0.0110 | 0.1637 | True | 0 | 5 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| C5_out | lab_cons | 0.0763 | 9/0/0 | 0.0020 | NaN | -0.0110 | 0.1637 | True | 0 | 5 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| C5_cell_out | publication | 0.0763 | 10/4/2 | 0.0930 | 0.0932 | -0.0110 | 0.1637 | True | 0 | 5 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| C5_cell_out | lab_cons | 0.0763 | 9/0/0 | 0.0020 | NaN | -0.0110 | 0.1637 | True | 0 | 5 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |

## Duplicate sensitivities

| contrast | unit | rel | W/L/T | SF_POOL_p | SF_POOL_p_enum | LOPO_lo | LOPO_hi | equals_headline | label |
|---|---|---|---|---|---|---|---|---|---|
| MX_LM10_NB - B1 | publication | 0.0896 | 10/4/2 | 0.0404 | NaN | 0.0031 | 0.1761 | False | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NB - B1 | lab_cons | 0.0896 | 9/0/0 | 0.0020 | NaN | 0.0031 | 0.1761 | False | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC - B1 | publication | 0.0763 | 10/4/2 | 0.0980 | 0.0932 | -0.0110 | 0.1637 | True | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC - B1 | lab_cons | 0.0763 | 9/0/0 | 0.0020 | NaN | -0.0110 | 0.1637 | True | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_OFS - B1 | publication | 0.0706 | 10/4/2 | 0.1092 | NaN | -0.0140 | 0.1551 | False | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_OFS - B1 | lab_cons | 0.0706 | 9/0/0 | 0.0020 | NaN | -0.0140 | 0.1551 | False | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC05 - B1 | publication | 0.0854 | 10/4/2 | 0.0624 | NaN | -0.0032 | 0.1741 | False | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC05 - B1 | lab_cons | 0.0854 | 9/0/0 | 0.0020 | NaN | -0.0032 | 0.1741 | False | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC03 - B1 | publication | 0.0868 | 10/4/2 | 0.0540 | NaN | -0.0009 | 0.1745 | False | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC03 - B1 | lab_cons | 0.0868 | 9/0/0 | 0.0020 | NaN | -0.0009 | 0.1745 | False | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
