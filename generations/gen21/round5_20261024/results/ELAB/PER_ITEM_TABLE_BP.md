# 1.3 E-LAB (BP): per-item table

**Label: EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression; E-LAB (campaign calibration of a new extractant; donors = the campaign's other extractants' full series): never placed next to E-A's 0.452; B1 on the same rows is the comparator; seal b96c3bef9828**

| field | value |
|---|---|
| Item id and estimand | 1.3 E-LAB (lab-1): campaign calibration of a new extractant (E-CT annex) |
| Population (cells, publications, units, rows per seed) | PRIMARY (POPX, near-copies excluded): 92 cells, 16 publications, 9 / 14 laboratories (conservative / liberal, PROVISIONAL key), 4 source clusters, 787 rows per seed |
| Design | BP (sensitivity) |
| Read budget | 4 reads (DOPT_Q, code budget k = 3; E-A's read sets); the campaign's other series in full |
| Arms on identical rows | B1, MX_LM10, MX_LM05, MX_LM20, MX_K6p1, MX_K12, MX_K20, MX_K35, MX_K60, MX_K100, MX_LM10_SS, MX_LM10_OS, MX_LM10_SM, MX_LM10_DM, MX_LM10_LO, MX_LM10_HI, MX_LM10_NB, MX_LM10_NC, MX_LM10_OFS, MX_LM10_NC05, MX_LM10_NC03, E_RM, E_PBC, E_PBA, E_PCM, E_PCMC, MX_GK_LOPO, MX_GK_O |
| R², MSE, relative SSE vs B1 | see 'R², MSE, relative SSE vs B1' below |
| Per-seed values | see 'Per-seed values' below |
| Unit W/L/T and the descriptive tests | see 'Unit W/L/T and the descriptive tests' below |
| LOPO and top-3 share | see 'LOPO and top-3 share' below |
| C5-BTBP in / out | see 'C5-BTBP in / out' below |
| Duplicate sensitivities | see 'Duplicate sensitivities' below |
| Leakage status | intervention passed on 460 target-seeds (92 targets); positive controls self 92/92, bit copy 2/2, leak 92/92; label-based sensitivities changed 170 times (declared); LOPO-K poisoning passed (leaky control caught 14/16); B1 identity 8.9e-16 |
| Compute | 225 s (one process) |
| Label | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression; E-LAB (campaign calibration of a new extractant; donors = the campaign's other extractants' full series): never placed next to E-A's 0.452; B1 on the same rows is the comparator; seal b96c3bef9828 |
| What changed scientifically | On the near-copy-cleaned POPX the level-matched campaign shape gives +0.0780 relative SSE vs B1 on identical rows, with laboratory (conservative, PROVISIONAL key) W/L/T 8/1/0 and SF_POOL p 0.0039 (publication units, the bridge to lab-1: seen before cleaning +0.090, after +0.076); it remains a seen, best-of-about-25 exploratory gain whose mechanism (campaign vs level-driven weighting) the controls only bound. |
| Next registered step | External leg data-gated (section 7): descriptive unless >= 8 independent laboratories with >= 2 extractants x >= 6 Ln in one campaign; no in-corpus kill fires. |

## R², MSE, relative SSE vs B1

| arm | R2 | MSE | relSSE_vs_B1 | MAE_balanced | label |
|---|---|---|---|---|---|
| B1 | 0.4423 | 0.0255 | 0.0000 | 0.1079 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 | 0.4857 | 0.0235 | 0.0780 | 0.1014 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM05 | 0.4835 | 0.0236 | 0.0739 | 0.1026 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM20 | 0.4764 | 0.0240 | 0.0611 | 0.1037 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_K6p1 | 0.4583 | 0.0248 | 0.0287 | 0.1051 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_K12 | 0.4623 | 0.0246 | 0.0359 | 0.1053 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_K20 | 0.4606 | 0.0247 | 0.0329 | 0.1055 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_K35 | 0.4567 | 0.0249 | 0.0259 | 0.1060 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_K60 | 0.4526 | 0.0251 | 0.0185 | 0.1065 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_K100 | 0.4493 | 0.0252 | 0.0126 | 0.1068 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_LO | 0.4402 | 0.0256 | -0.0036 | 0.1081 | DIAGNOSTIC; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_HI | 0.4878 | 0.0235 | 0.0816 | 0.1012 | DIAGNOSTIC; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NB | 0.4936 | 0.0232 | 0.0920 | 0.1004 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC | 0.4857 | 0.0235 | 0.0780 | 0.1014 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_OFS | 0.4825 | 0.0237 | 0.0722 | 0.1017 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC05 | 0.4913 | 0.0233 | 0.0879 | 0.1009 | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC03 | 0.4920 | 0.0233 | 0.0893 | 0.1008 | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_RM | 0.4756 | 0.0240 | 0.0597 | NaN | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PBC | 0.4323 | 0.0260 | -0.0179 | NaN | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PBA | 0.4350 | 0.0259 | -0.0131 | NaN | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PCM | 0.4261 | 0.0263 | -0.0289 | NaN | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PCMC | 0.4359 | 0.0258 | -0.0114 | NaN | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_GK_LOPO | 0.4494 | 0.0252 | 0.0128 | 0.1062 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_GK_O | 0.4623 | 0.0246 | 0.0359 | 0.1053 | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |

## Per-seed values

| arm | R2_per_seed | rel_per_seed |
|---|---|---|
| B1 | 0.4544;0.4334;0.4525;0.4329;0.4380 | 0.0000;0.0000;0.0000;0.0000;0.0000 |
| MX_LM10 | 0.4951;0.4770;0.4975;0.4782;0.4810 | 0.0746;0.0769;0.0822;0.0798;0.0764 |
| MX_LM05 | 0.4941;0.4798;0.4929;0.4734;0.4772 | 0.0728;0.0819;0.0737;0.0714;0.0697 |
| MX_LM20 | 0.4853;0.4720;0.4857;0.4677;0.4712 | 0.0566;0.0682;0.0605;0.0613;0.0590 |
| MX_K6p1 | 0.4671;0.4521;0.4680;0.4511;0.4532 | 0.0233;0.0331;0.0282;0.0320;0.0269 |
| MX_K12 | 0.4720;0.4552;0.4721;0.4545;0.4576 | 0.0323;0.0386;0.0357;0.0380;0.0348 |
| MX_K20 | 0.4710;0.4530;0.4706;0.4525;0.4561 | 0.0304;0.0346;0.0330;0.0345;0.0321 |
| MX_K35 | 0.4676;0.4486;0.4668;0.4482;0.4522 | 0.0242;0.0268;0.0260;0.0270;0.0253 |
| MX_K60 | 0.4639;0.4441;0.4628;0.4438;0.4482 | 0.0174;0.0190;0.0187;0.0192;0.0181 |
| MX_K100 | 0.4609;0.4406;0.4595;0.4403;0.4449 | 0.0119;0.0128;0.0127;0.0130;0.0123 |
| MX_LM10_LO | 0.4522;0.4317;0.4496;0.4317;0.4361 | -0.0041;-0.0029;-0.0054;-0.0022;-0.0035 |
| MX_LM10_HI | 0.4973;0.4786;0.5005;0.4794;0.4829 | 0.0787;0.0799;0.0876;0.0820;0.0799 |
| MX_LM10_NB | 0.5024;0.4844;0.5050;0.4868;0.4893 | 0.0880;0.0901;0.0959;0.0949;0.0911 |
| MX_LM10_NC | 0.4951;0.4770;0.4975;0.4782;0.4810 | 0.0746;0.0769;0.0822;0.0798;0.0764 |
| MX_LM10_OFS | 0.4916;0.4738;0.4942;0.4752;0.4778 | 0.0681;0.0713;0.0761;0.0745;0.0707 |
| MX_LM10_NC05 | 0.5002;0.4821;0.5027;0.4844;0.4870 | 0.0839;0.0860;0.0916;0.0907;0.0872 |
| MX_LM10_NC03 | 0.5009;0.4829;0.5035;0.4851;0.4878 | 0.0853;0.0874;0.0931;0.0920;0.0885 |
| E_RM | 0.4842;0.4680;0.4857;0.4686;0.4715 | 0.0546;0.0610;0.0606;0.0629;0.0595 |
| E_PBC | 0.4441;0.4246;0.4412;0.4215;0.4300 | -0.0189;-0.0154;-0.0207;-0.0201;-0.0144 |
| E_PBA | 0.4455;0.4275;0.4464;0.4251;0.4305 | -0.0163;-0.0105;-0.0112;-0.0138;-0.0135 |
| E_PCM | 0.4376;0.4172;0.4371;0.4161;0.4228 | -0.0308;-0.0286;-0.0282;-0.0297;-0.0272 |
| E_PCMC | 0.4464;0.4287;0.4443;0.4262;0.4339 | -0.0147;-0.0082;-0.0150;-0.0119;-0.0073 |
| MX_GK_LOPO | 0.4597;0.4424;0.4591;0.4414;0.4444 | 0.0097;0.0159;0.0121;0.0150;0.0113 |
| MX_GK_O | 0.4720;0.4552;0.4721;0.4545;0.4576 | 0.0323;0.0386;0.0357;0.0380;0.0348 |

## Unit W/L/T and the descriptive tests

| contrast | unit | G | W/L/T | SF_POOL_p | SIGN_p | IM_REL_p | unit_median_rel | label |
|---|---|---|---|---|---|---|---|---|
| MX_LM10 - B1 | publication | 16 | 10/4/2 | 0.0954 | 0.0898 | 0.0340 | 0.0441 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - B1 | lab_cons | 9 | 8/1/0 | 0.0039 | 0.0195 | 0.0110 | 0.0728 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - B1 | lab_lib | 14 | 9/3/2 | 0.0428 | 0.0730 | 0.0411 | 0.0441 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - B1 | cluster_cpg | 4 | 3/1/0 | 0.1250 | 0.3125 | 0.0703 | 0.1502 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PBC | publication | 16 | 13/3/0 | 0.0302 | 0.0106 | 0.0193 | 0.0581 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PBC | lab_cons | 9 | 8/1/0 | 0.0059 | 0.0195 | 0.0068 | 0.0903 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PBC | lab_lib | 14 | 12/2/0 | 0.0170 | 0.0065 | 0.0229 | 0.0714 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PBC | cluster_cpg | 4 | 4/0/0 | 0.0625 | 0.0625 | 0.0558 | 0.1555 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PCM | publication | 16 | 11/4/1 | 0.0184 | 0.0592 | 0.0132 | 0.0508 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PCM | lab_cons | 9 | 8/1/0 | 0.0039 | 0.0195 | 0.0042 | 0.0890 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PCM | lab_lib | 14 | 10/3/1 | 0.0072 | 0.0461 | 0.0147 | 0.0661 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_PCM | cluster_cpg | 4 | 4/0/0 | 0.0625 | 0.0625 | 0.0488 | 0.1637 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_RM | publication | 16 | 10/3/3 | 0.0362 | 0.0461 | 0.0068 | 0.0014 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_RM | lab_cons | 9 | 8/1/0 | 0.0078 | 0.0195 | 0.0179 | 0.0066 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_RM | lab_lib | 14 | 9/2/3 | 0.0034 | 0.0327 | 0.0046 | 0.0054 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10 - E_RM | cluster_cpg | 4 | 2/2/0 | 0.3750 | 0.6875 | 0.2047 | 0.0011 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PBC - B1 | publication | 16 | 5/11/0 | 0.9920 | 0.9616 | 0.8775 | -0.0089 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PBC - B1 | lab_cons | 9 | 4/5/0 | 0.9492 | 0.7461 | 0.9126 | -0.0051 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PBC - B1 | lab_lib | 14 | 4/10/0 | 0.9850 | 0.9713 | 0.8780 | -0.0089 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PBC - B1 | cluster_cpg | 4 | 1/3/0 | 0.9375 | 0.9375 | 0.9370 | -0.0181 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PCM - B1 | publication | 16 | 4/11/1 | 0.9990 | 0.9824 | 0.9087 | -0.0201 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PCM - B1 | lab_cons | 9 | 1/8/0 | 0.9961 | 0.9980 | 0.9591 | -0.0396 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PCM - B1 | lab_lib | 14 | 3/10/1 | 0.9976 | 0.9888 | 0.9092 | -0.0230 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| E_PCM - B1 | cluster_cpg | 4 | 0/4/0 | 1.0000 | 1.0000 | 0.9735 | -0.0286 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |

## LOPO and top-3 share

| contrast | rel | LOPO_lo | LOPO_hi | LOPO_min | LOPO_min_without | LTPO_min | without_LEAD2 | top3_share |
|---|---|---|---|---|---|---|---|---|
| MX_LM10 - B1 | 0.0780 | -0.0119 | 0.1678 | 0.0480 | pub_1bd76c48d9 | 0.0050 | 0.0050 | 1.0427 |
| MX_LM10 - E_PBC | 0.0942 | 0.0043 | 0.1840 | 0.0641 | pub_1bd76c48d9 | 0.0218 | 0.0218 | 0.9344 |
| MX_LM10 - E_PCM | 0.1039 | 0.0103 | 0.1974 | 0.0737 | pub_1bd76c48d9 | 0.0342 | 0.0342 | 0.8935 |
| MX_LM10 - E_RM | 0.0194 | -0.0055 | 0.0443 | 0.0092 | pub_126bbe7976 | 0.0027 | 0.0099 | 0.9668 |
| E_PBC - B1 | -0.0179 | -0.0284 | -0.0074 | -0.0200 | pub_74428aeab2 | -0.0219 | -0.0172 | NaN |
| E_PCM - B1 | -0.0289 | -0.0507 | -0.0071 | -0.0329 | pub_a49146c8fd | -0.0390 | -0.0303 | NaN |

## C5-BTBP in / out

| c5 | unit | rel | W/L/T | SF_POOL_p | SF_POOL_p_enum | LOPO_lo | LOPO_hi | equals_headline | point_arms_changed_vs_C5_in | draw_arms_changed_vs_C5_in | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C5_in | publication | 0.0780 | 10/4/2 | 0.0954 | 0.0978 | -0.0119 | 0.1678 | True | 0 | 0 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| C5_in | lab_cons | 0.0780 | 8/1/0 | 0.0039 | NaN | -0.0119 | 0.1678 | True | 0 | 0 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| C5_out | publication | 0.0780 | 10/4/2 | 0.1046 | 0.0978 | -0.0119 | 0.1678 | True | 0 | 5 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| C5_out | lab_cons | 0.0780 | 8/1/0 | 0.0039 | NaN | -0.0119 | 0.1678 | True | 0 | 5 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| C5_cell_out | publication | 0.0780 | 10/4/2 | 0.0986 | 0.0978 | -0.0119 | 0.1678 | True | 0 | 5 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| C5_cell_out | lab_cons | 0.0780 | 8/1/0 | 0.0039 | NaN | -0.0119 | 0.1678 | True | 0 | 5 | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |

## Duplicate sensitivities

| contrast | unit | rel | W/L/T | SF_POOL_p | SF_POOL_p_enum | LOPO_lo | LOPO_hi | equals_headline | label |
|---|---|---|---|---|---|---|---|---|---|
| MX_LM10_NB - B1 | publication | 0.0920 | 10/4/2 | 0.0374 | NaN | 0.0021 | 0.1819 | False | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NB - B1 | lab_cons | 0.0920 | 8/1/0 | 0.0039 | NaN | 0.0021 | 0.1819 | False | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC - B1 | publication | 0.0780 | 10/4/2 | 0.0936 | 0.0978 | -0.0119 | 0.1678 | True | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC - B1 | lab_cons | 0.0780 | 8/1/0 | 0.0039 | NaN | -0.0119 | 0.1678 | True | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_OFS - B1 | publication | 0.0722 | 10/4/2 | 0.1236 | NaN | -0.0149 | 0.1592 | False | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_OFS - B1 | lab_cons | 0.0722 | 8/1/0 | 0.0039 | NaN | -0.0149 | 0.1592 | False | EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC05 - B1 | publication | 0.0879 | 10/4/2 | 0.0624 | NaN | -0.0042 | 0.1799 | False | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC05 - B1 | lab_cons | 0.0879 | 8/1/0 | 0.0039 | NaN | -0.0042 | 0.1799 | False | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC03 - B1 | publication | 0.0893 | 10/4/2 | 0.0538 | NaN | -0.0019 | 0.1804 | False | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
| MX_LM10_NC03 - B1 | lab_cons | 0.0893 | 8/1/0 | 0.0039 | NaN | -0.0019 | 0.1804 | False | ORACLE [O]; EXPLORATORY (seen); SENSITIVITY; best of about 25 arms on seen rows; expect regression |
