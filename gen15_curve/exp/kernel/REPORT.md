# gen15 / `kernel` — similarity-kernel models on molecular fingerprints

**Verdict: null.** Eighty-five kernel and linear-fingerprint arms were scored under all five
designs. Not one beats `G14` under BP by more than the bootstrap standard error. The best arm in
the whole study is `C_RBF_KRR_a1` at **0.4980** against `G14`'s **0.5001** — a gain of 0.0021
(p = 0.704, chemotype-blocked paired bootstrap, leave-one-chemotype-out sign *not* stable). For the
direction the kernels are not merely useless but actively **harmful**: the best Tanimoto direction
arm scores 0.6373 under BP, which is 0.049 *worse than predicting no separation at all*, and the
paired test against `FLAT` is significant in the wrong direction (p = 0.0006).

Reference points, extractant-macro MAE under BP, as this run measured them:

| arm | BP |
|---|---|
| `FLAT` (no separation) | 0.5885 |
| `G14` (deployed) | **0.5001** |
| best kernel arm, `C_RBF_KRR_a1` | 0.4980 |
| best magnitude kernel, `M_TAN_GP` | 0.4985 |
| best direction kernel, `D_TAN_KRR_a0.01` | 0.6373 |

`G14` reproduces at 0.5001 against the programme's locked 0.500, so the bench was driven correctly.

---

## 1. What was built

All files are under `D:\ml_separator_gh\gen15_curve\exp\kernel\`.

| file | what it does |
|---|---|
| `kernels.py` | the arm library: Tanimoto / RBF / Matérn / TOPO39-cosine / combined Grams, a weighted kernel ridge, a hand-rolled weighted kernel logistic (L-BFGS on the dual with an unpenalised intercept), a GP predictive mean and variance with the noise fixed at the corpus replicate sd, similarity-weighted kNN over *ligand* units, plain L2 logistic on the 2048 raw bits, plain ridge on the 206 RDKit descriptors, and the applicability-domain probe |
| `run.py` | runs one named batch through one design in its own process and saves the board plus the per-extractant frame |
| `tables.py` | merges the per-design boards of a batch and renders the markdown tables below |
| `contrasts.py` | gen13's chemotype-blocked paired bootstrap against `G14` and `FLAT`, off the saved per-extractant frames (no refitting) |
| `diracc.py` | direction accuracy of every candidate sign rule in gen14's own units (extractant unit, chemotype block) |
| `ad.py` | the applicability-domain probe: Tanimoto to the nearest training ligand per held-out cell, error by band, GP posterior sd as a confidence, the three confound guards, and leave-one-chemotype-out |
| `adcurve.py` | the same applicability-domain quantity in equal-count deciles, plus the distribution of the similarity itself |
| `smoke.py` | one-design sanity check, including the eigenvalues of the Tanimoto Gram |
| `results/` | 20 boards (4 batches x 5 designs), 20 per-extractant frames, 3 contrast tables, `direction_accuracy.csv`, `ad_*.csv`, `publeak_*.csv`. Note for whoever commits this: the 20 `perext_*.pkl` frames are about 40 MB in total and exist only so a paired contrast can be recomputed without refitting; every conclusion in this report is already in the CSVs, so the pickles can be dropped from a commit without losing anything. |
| `publeak.py` | the mechanism check: how much of the kernel's training weight sits in the test cell's own publication, and the error split by whether the publication is in training |

**Composition rules**, so a gain can be attributed to one thing. A `D_*` arm keeps gen14's
magnitude and curvature and replaces only the *sign*; an `M_*` arm keeps gen14's direction and
curvature and replaces only *|a|*; a `C_*` arm keeps gen14's `a` entirely and replaces only *b*; a
`J_*` arm predicts `(a, b)` outright. So `D_* − G14` is the value of the new direction call and
nothing else.

**Leakage discipline.** The Tanimoto Gram uses no fitted statistic, so it is computed once over all
521 cells. Every descriptor kernel *does* use fitted statistics (median imputation, standardisation,
the median-heuristic length scale) and is therefore refitted on the training fold of every fold and
cached one fold at a time. The kNN pool is collapsed to one row per *extractant*, because 61 of 82
scored extractants contribute a single cell and one contributes 63, so a cell-level pool would let
one much-measured ligand fill every neighbourhood.

**Penalty grids were fixed in advance and every value is reported.** Kernel ridge α ∈ {0.01, 0.1, 1,
10}; kernel logistic λ ∈ {0.1, 1, 10, 100}; SVC C ∈ {0.1, 1, 10}; logistic C ∈ {0.01, 0.1, 1};
descriptor ridge α ∈ {1, 10, 100, 1000}; k ∈ {1, 3, 5, 10}. Nothing was selected on a test score —
the tables below are the whole grid, sorted by BP only for readability.

---

## 2. The five-design tables

Sorted by BP. `G14` and `FLAT` carry every board.


#### `dir` -- extractant-macro MAE (lower is better), sorted by BP

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| G14 | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| FLAT | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| D_TAN_KRR_a0.01 | 0.5852 | 0.5929 | 0.5861 | 0.5300 | 0.6373 |
| D_KNN_TAN_k10 | 0.5887 | 0.5891 | 0.5897 | 0.5722 | 0.6402 |
| D_TAN_KRR_a0.1 | 0.5815 | 0.5866 | 0.5851 | 0.5315 | 0.6403 |
| D_TAN_KRR_a1 | 0.5892 | 0.5954 | 0.5854 | 0.5406 | 0.6483 |
| D_ECFPz_LOGIT_C1 | 0.6000 | 0.6277 | 0.6026 | 0.5316 | 0.6514 |
| D_KNN_TAN_k5 | 0.5945 | 0.6084 | 0.5973 | 0.5536 | 0.6543 |
| D_TAN_KRR_a10 | 0.6045 | 0.6059 | 0.6080 | 0.5845 | 0.6710 |
| D_ECFPraw_LOGIT_C1 | 0.5744 | 0.5885 | 0.5788 | 0.5219 | 0.6712 |
| D_ECFPz_LOGIT_C0.1 | 0.6248 | 0.6476 | 0.6204 | 0.5319 | 0.6770 |
| D_KNN_TAN_k3 | 0.6095 | 0.6077 | 0.6004 | 0.5427 | 0.6822 |
| D_KNN_TAN_k1 | 0.6406 | 0.6462 | 0.6176 | 0.5135 | 0.6860 |
| D_ECFPraw_LOGIT_C0.1 | 0.5736 | 0.5904 | 0.5771 | 0.5195 | 0.6866 |
| D_TAN_KLR_l0.1 | 0.5684 | 0.5875 | 0.5703 | 0.5163 | 0.6916 |
| D_TAN_SVC_C10 | 0.5782 | 0.5852 | 0.5806 | 0.5091 | 0.7006 |
| D_TAN_SVC_C1 | 0.5707 | 0.5945 | 0.5730 | 0.5146 | 0.7023 |
| D_ECFPz_LOGIT_C0.01 | 0.6267 | 0.6392 | 0.6362 | 0.5308 | 0.7049 |
| D_ECFPraw_LOGIT_C0.01 | 0.6547 | 0.6728 | 0.6833 | 0.5470 | 0.7099 |
| D_TAN_KLR_l1 | 0.5879 | 0.6279 | 0.5996 | 0.5258 | 0.7100 |
| D_TAN_SVC_C0.1 | 0.7398 | 0.7439 | 0.7344 | 0.5956 | 0.7434 |
| D_TAN_KLR_l10 | 0.7312 | 0.7369 | 0.7336 | 0.6031 | 0.7486 |
| D_TAN_KLR_l100 | 0.7529 | 0.7506 | 0.7504 | 0.7211 | 0.7564 |


#### `dir` -- macro sign accuracy on strong pairs (higher is better)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| G14 | 0.8081 | 0.8093 | 0.8102 | 0.8085 | 0.8122 |
| D_KNN_TAN_k10 | 0.6569 | 0.6619 | 0.6563 | 0.6844 | 0.5966 |
| D_TAN_KRR_a0.01 | 0.6288 | 0.6114 | 0.6250 | 0.7596 | 0.5751 |
| D_TAN_KRR_a0.1 | 0.6366 | 0.6267 | 0.6284 | 0.7468 | 0.5724 |
| D_KNN_TAN_k5 | 0.6356 | 0.6091 | 0.6335 | 0.7293 | 0.5719 |
| D_TAN_KRR_a1 | 0.6388 | 0.6194 | 0.6458 | 0.7317 | 0.5634 |
| D_ECFPz_LOGIT_C1 | 0.5886 | 0.5486 | 0.5902 | 0.7404 | 0.5389 |
| D_TAN_KRR_a10 | 0.6427 | 0.6385 | 0.6380 | 0.6779 | 0.5343 |
| D_KNN_TAN_k3 | 0.5797 | 0.5997 | 0.5968 | 0.7316 | 0.5125 |
| D_ECFPz_LOGIT_C0.1 | 0.5411 | 0.5151 | 0.5545 | 0.7376 | 0.5013 |
| D_KNN_TAN_k1 | 0.5049 | 0.5079 | 0.5456 | 0.7752 | 0.4862 |
| D_ECFPraw_LOGIT_C1 | 0.6396 | 0.6183 | 0.6321 | 0.7550 | 0.4819 |
| D_ECFPraw_LOGIT_C0.1 | 0.6412 | 0.6140 | 0.6340 | 0.7599 | 0.4561 |
| D_ECFPz_LOGIT_C0.01 | 0.5405 | 0.5269 | 0.5237 | 0.7396 | 0.4499 |
| D_TAN_KLR_l0.1 | 0.6492 | 0.6202 | 0.6479 | 0.7655 | 0.4493 |
| D_TAN_SVC_C10 | 0.6276 | 0.6195 | 0.6261 | 0.7816 | 0.4408 |
| D_TAN_SVC_C1 | 0.6461 | 0.6079 | 0.6422 | 0.7705 | 0.4389 |
| D_TAN_KLR_l1 | 0.6185 | 0.5530 | 0.5996 | 0.7464 | 0.4255 |
| D_ECFPraw_LOGIT_C0.01 | 0.4962 | 0.4686 | 0.4429 | 0.6874 | 0.4246 |
| D_TAN_SVC_C0.1 | 0.3441 | 0.3484 | 0.3555 | 0.6019 | 0.3619 |
| D_TAN_KLR_l10 | 0.3532 | 0.3589 | 0.3543 | 0.5920 | 0.3585 |
| D_TAN_KLR_l100 | 0.3167 | 0.3349 | 0.3267 | 0.3784 | 0.3434 |
| FLAT | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


#### `dir` -- macro pair Spearman (higher is better)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| G14 | 0.5062 | 0.5025 | 0.5078 | 0.5111 | 0.5038 |
| D_KNN_TAN_k10 | 0.3334 | 0.3340 | 0.3270 | 0.3580 | 0.2323 |
| D_TAN_KRR_a0.01 | 0.3051 | 0.2823 | 0.3043 | 0.4364 | 0.2236 |
| D_TAN_KRR_a0.1 | 0.3201 | 0.3070 | 0.3092 | 0.4262 | 0.2211 |
| D_TAN_KRR_a1 | 0.3177 | 0.2959 | 0.3290 | 0.4050 | 0.2088 |
| D_KNN_TAN_k5 | 0.2946 | 0.2548 | 0.2889 | 0.3976 | 0.1852 |
| D_TAN_KRR_a10 | 0.3162 | 0.3108 | 0.3071 | 0.3273 | 0.1601 |
| D_ECFPz_LOGIT_C1 | 0.2321 | 0.1628 | 0.2301 | 0.4277 | 0.1506 |
| D_KNN_TAN_k3 | 0.2459 | 0.2603 | 0.2606 | 0.4229 | 0.1264 |
| D_ECFPz_LOGIT_C0.1 | 0.1558 | 0.1080 | 0.1767 | 0.4219 | 0.0857 |
| D_KNN_TAN_k1 | 0.1345 | 0.1284 | 0.2029 | 0.4765 | 0.0812 |
| D_ECFPraw_LOGIT_C1 | 0.2836 | 0.2458 | 0.2680 | 0.4081 | 0.0323 |
| D_ECFPz_LOGIT_C0.01 | 0.1493 | 0.1184 | 0.1246 | 0.3995 | -0.0012 |
| D_ECFPraw_LOGIT_C0.1 | 0.2828 | 0.2376 | 0.2730 | 0.4147 | -0.0080 |
| D_TAN_KLR_l0.1 | 0.2941 | 0.2451 | 0.2912 | 0.4206 | -0.0106 |
| D_TAN_SVC_C10 | 0.2755 | 0.2557 | 0.2690 | 0.4402 | -0.0240 |
| D_TAN_SVC_C1 | 0.2898 | 0.2259 | 0.2832 | 0.4263 | -0.0269 |
| D_TAN_KLR_l1 | 0.2415 | 0.1321 | 0.2131 | 0.3943 | -0.0445 |
| D_ECFPraw_LOGIT_C0.01 | 0.0583 | 0.0111 | -0.0210 | 0.3243 | -0.0502 |
| D_TAN_SVC_C0.1 | -0.1551 | -0.1470 | -0.1390 | 0.1974 | -0.1154 |
| D_TAN_KLR_l10 | -0.1423 | -0.1454 | -0.1427 | 0.1820 | -0.1249 |
| D_TAN_KLR_l100 | -0.1919 | -0.1709 | -0.1750 | -0.1392 | -0.1413 |
| FLAT | nan | nan | nan | nan | nan |


#### `dir2` -- extractant-macro MAE (lower is better), sorted by BP

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| G14 | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| FLAT | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| D_LIG2D_RIDGE_a10 | 0.5547 | 0.5546 | 0.5609 | 0.5255 | 0.6172 |
| D_LIG2D_RIDGE_a1 | 0.5820 | 0.5878 | 0.5661 | 0.5530 | 0.6244 |
| D_RBF_KRR_a0.1 | 0.5729 | 0.5727 | 0.5676 | 0.5219 | 0.6297 |
| D_LIG2D_RIDGE_a100 | 0.5381 | 0.5274 | 0.5351 | 0.5182 | 0.6322 |
| D_LIG2D_RIDGE_a1000 | 0.5527 | 0.5488 | 0.5480 | 0.5263 | 0.6512 |
| D_RBF_KRR_a1 | 0.5806 | 0.5759 | 0.5750 | 0.5306 | 0.6585 |
| D_TOPOK_KLR_l1 | 0.7008 | 0.7065 | 0.7007 | 0.7169 | 0.6706 |
| D_RBF_KLR_l0.1 | 0.5995 | 0.6155 | 0.6103 | 0.5421 | 0.6709 |
| D_COMBO_KLR_l0.1 | 0.7111 | 0.7168 | 0.7110 | 0.7165 | 0.6720 |
| D_COMBO_KLR_l1 | 0.7029 | 0.7086 | 0.7028 | 0.7166 | 0.6937 |
| D_RBF_KRR_a10 | 0.6154 | 0.6123 | 0.6122 | 0.5823 | 0.6953 |
| D_RBF_KLR_l1 | 0.6431 | 0.6685 | 0.6542 | 0.5572 | 0.6968 |
| D_COMBO_KLR_l10 | 0.7057 | 0.7111 | 0.7056 | 0.7166 | 0.6976 |
| D_TOPOK_KLR_l10 | 0.7057 | 0.7116 | 0.7056 | 0.7166 | 0.6976 |
| D_COMBO_KLR_l100 | 0.7106 | 0.7211 | 0.7111 | 0.7166 | 0.7029 |
| D_MAT_KLR_l1 | 0.6472 | 0.6693 | 0.6526 | 0.5590 | 0.7044 |
| D_RBF_KLR_l10 | 0.7301 | 0.7237 | 0.7251 | 0.6146 | 0.7399 |
| D_MAT_KLR_l10 | 0.7290 | 0.7329 | 0.7297 | 0.6219 | 0.7487 |
| D_RBF_KLR_l100 | 0.7531 | 0.7502 | 0.7514 | 0.7194 | 0.7509 |


#### `dir2` -- macro sign accuracy on strong pairs (higher is better)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| G14 | 0.8081 | 0.8093 | 0.8102 | 0.8085 | 0.8122 |
| D_LIG2D_RIDGE_a10 | 0.6780 | 0.6768 | 0.6698 | 0.7458 | 0.6113 |
| D_RBF_KRR_a0.1 | 0.6481 | 0.6420 | 0.6579 | 0.7479 | 0.5982 |
| D_LIG2D_RIDGE_a1 | 0.6367 | 0.6178 | 0.6477 | 0.7048 | 0.5913 |
| D_LIG2D_RIDGE_a100 | 0.7120 | 0.7279 | 0.7155 | 0.7533 | 0.5895 |
| D_LIG2D_RIDGE_a1000 | 0.6861 | 0.6935 | 0.6947 | 0.7353 | 0.5535 |
| D_RBF_KRR_a1 | 0.6380 | 0.6447 | 0.6452 | 0.7361 | 0.5447 |
| D_RBF_KLR_l0.1 | 0.5919 | 0.5722 | 0.5766 | 0.7011 | 0.4994 |
| D_RBF_KRR_a10 | 0.6177 | 0.6219 | 0.6251 | 0.6788 | 0.4873 |
| D_TOPOK_KLR_l1 | 0.4117 | 0.4169 | 0.4177 | 0.3857 | 0.4776 |
| D_COMBO_KLR_l0.1 | 0.3877 | 0.3928 | 0.3937 | 0.3875 | 0.4753 |
| D_RBF_KLR_l1 | 0.5092 | 0.4784 | 0.4953 | 0.6680 | 0.4612 |
| D_MAT_KLR_l1 | 0.5006 | 0.4776 | 0.4977 | 0.6595 | 0.4435 |
| D_COMBO_KLR_l1 | 0.4086 | 0.4138 | 0.4146 | 0.3872 | 0.4311 |
| D_COMBO_KLR_l10 | 0.4047 | 0.4107 | 0.4107 | 0.3872 | 0.4251 |
| D_TOPOK_KLR_l10 | 0.4047 | 0.4099 | 0.4107 | 0.3872 | 0.4251 |
| D_COMBO_KLR_l100 | 0.3926 | 0.3866 | 0.3978 | 0.3872 | 0.4126 |
| D_RBF_KLR_l10 | 0.3570 | 0.3844 | 0.3708 | 0.5578 | 0.3790 |
| D_MAT_KLR_l10 | 0.3586 | 0.3688 | 0.3626 | 0.5466 | 0.3620 |
| D_RBF_KLR_l100 | 0.3158 | 0.3361 | 0.3248 | 0.3807 | 0.3580 |
| FLAT | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


#### `dir2` -- macro pair Spearman (higher is better)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| G14 | 0.5062 | 0.5025 | 0.5078 | 0.5111 | 0.5038 |
| D_LIG2D_RIDGE_a10 | 0.3724 | 0.3763 | 0.3582 | 0.4349 | 0.2640 |
| D_RBF_KRR_a0.1 | 0.3275 | 0.3181 | 0.3307 | 0.4295 | 0.2559 |
| D_LIG2D_RIDGE_a1 | 0.3174 | 0.2821 | 0.3290 | 0.3607 | 0.2204 |
| D_LIG2D_RIDGE_a100 | 0.3953 | 0.4214 | 0.4050 | 0.4360 | 0.2127 |
| D_LIG2D_RIDGE_a1000 | 0.3725 | 0.3799 | 0.3863 | 0.4071 | 0.1771 |
| D_RBF_KRR_a1 | 0.3107 | 0.3173 | 0.3180 | 0.4072 | 0.1710 |
| D_RBF_KRR_a10 | 0.2898 | 0.2967 | 0.2975 | 0.3290 | 0.0900 |
| D_RBF_KLR_l0.1 | 0.2329 | 0.1942 | 0.2085 | 0.3657 | 0.0636 |
| D_TOPOK_KLR_l1 | -0.0856 | -0.0826 | -0.0755 | -0.1266 | 0.0176 |
| D_COMBO_KLR_l0.1 | -0.1159 | -0.1128 | -0.1057 | -0.1253 | 0.0139 |
| D_RBF_KLR_l1 | 0.1111 | 0.0584 | 0.0869 | 0.3193 | 0.0030 |
| D_MAT_KLR_l1 | 0.0987 | 0.0514 | 0.0824 | 0.3102 | -0.0164 |
| D_COMBO_KLR_l1 | -0.0902 | -0.0871 | -0.0800 | -0.1252 | -0.0552 |
| D_TOPOK_KLR_l10 | -0.0971 | -0.0940 | -0.0869 | -0.1252 | -0.0650 |
| D_COMBO_KLR_l10 | -0.0971 | -0.0934 | -0.0869 | -0.1252 | -0.0650 |
| D_COMBO_KLR_l100 | -0.1124 | -0.1229 | -0.1029 | -0.1252 | -0.0799 |
| D_RBF_KLR_l10 | -0.1339 | -0.1020 | -0.1206 | 0.1541 | -0.0938 |
| D_MAT_KLR_l10 | -0.1373 | -0.1265 | -0.1349 | 0.1289 | -0.1116 |
| D_RBF_KLR_l100 | -0.1880 | -0.1697 | -0.1747 | -0.1347 | -0.1203 |
| FLAT | nan | nan | nan | nan | nan |


#### `mag` -- extractant-macro MAE (lower is better), sorted by BP

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| M_TAN_GP | 0.4758 | 0.4785 | 0.4740 | 0.4672 | 0.4985 |
| M_TANlin_KRR_a1 | 0.4775 | 0.4798 | 0.4757 | 0.4682 | 0.4993 |
| M_TANlin_KRR_a10 | 0.4866 | 0.4865 | 0.4850 | 0.4780 | 0.4996 |
| G14 | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| M_TANlin_KRR_a0.1 | 0.4740 | 0.4772 | 0.4716 | 0.4645 | 0.5004 |
| M_MAT_GP | 0.4768 | 0.4782 | 0.4755 | 0.4676 | 0.5027 |
| M_RBF_KRR_a0.1 | 0.4817 | 0.4823 | 0.4820 | 0.4786 | 0.5034 |
| M_TANlin_KRR_a0.01 | 0.4746 | 0.4787 | 0.4719 | 0.4588 | 0.5034 |
| M_KNN_TAN_k10 | 0.4909 | 0.4941 | 0.4886 | 0.4663 | 0.5038 |
| M_TAN_KRR_a0.1 | 0.4866 | 0.4892 | 0.4856 | 0.4783 | 0.5045 |
| M_TAN_KRR_a1 | 0.4898 | 0.4912 | 0.4901 | 0.4843 | 0.5046 |
| M_TAN_KRR_a10 | 0.4971 | 0.4947 | 0.4971 | 0.4930 | 0.5047 |
| M_TAN_KRR_a0.01 | 0.4865 | 0.4896 | 0.4851 | 0.4691 | 0.5053 |
| M_RBF_KRR_a10 | 0.4969 | 0.4952 | 0.4974 | 0.4889 | 0.5063 |
| M_RBF_KRR_a1 | 0.4884 | 0.4902 | 0.4902 | 0.4798 | 0.5065 |
| M_RBF_GP | 0.4813 | 0.4829 | 0.4800 | 0.4708 | 0.5066 |
| M_KNN_TAN_k5 | 0.4918 | 0.4933 | 0.4873 | 0.4647 | 0.5167 |
| M_KNN_TAN_k3 | 0.4851 | 0.4910 | 0.4804 | 0.4638 | 0.5312 |
| M_LIG2D_RIDGE_a1000 | 0.5040 | 0.5029 | 0.5025 | 0.5000 | 0.5318 |
| M_KNN_TAN_k1 | 0.5109 | 0.5099 | 0.5079 | 0.4939 | 0.5597 |
| M_LIG2D_RIDGE_a100 | 0.5134 | 0.5111 | 0.5055 | 0.5212 | 0.5764 |
| FLAT | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| M_LIG2D_RIDGE_a10 | 0.5903 | 0.5678 | 0.5680 | 0.5785 | 0.6199 |


#### `mag` -- macro sign accuracy on strong pairs (higher is better)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| M_MAT_GP | 0.8279 | 0.8269 | 0.8273 | 0.8246 | 0.8200 |
| M_RBF_GP | 0.8235 | 0.8247 | 0.8231 | 0.8245 | 0.8184 |
| M_TAN_GP | 0.8178 | 0.8142 | 0.8206 | 0.8137 | 0.8161 |
| M_TANlin_KRR_a1 | 0.8173 | 0.8122 | 0.8191 | 0.8139 | 0.8158 |
| M_TANlin_KRR_a10 | 0.8096 | 0.8090 | 0.8121 | 0.8118 | 0.8142 |
| M_TANlin_KRR_a0.1 | 0.8221 | 0.8134 | 0.8220 | 0.8123 | 0.8130 |
| G14 | 0.8081 | 0.8093 | 0.8102 | 0.8085 | 0.8122 |
| M_TANlin_KRR_a0.01 | 0.8185 | 0.8106 | 0.8209 | 0.8117 | 0.8117 |
| M_KNN_TAN_k10 | 0.8046 | 0.8027 | 0.8064 | 0.8024 | 0.8107 |
| M_RBF_KRR_a10 | 0.8129 | 0.8129 | 0.8121 | 0.8097 | 0.8027 |
| M_TAN_KRR_a10 | 0.8139 | 0.8130 | 0.8110 | 0.8102 | 0.8010 |
| M_RBF_KRR_a0.1 | 0.8094 | 0.8067 | 0.8069 | 0.8091 | 0.8004 |
| M_RBF_KRR_a1 | 0.8094 | 0.8075 | 0.8081 | 0.8110 | 0.7985 |
| M_TAN_KRR_a1 | 0.8134 | 0.8104 | 0.8120 | 0.8005 | 0.7940 |
| M_TAN_KRR_a0.1 | 0.8101 | 0.8064 | 0.8112 | 0.8010 | 0.7915 |
| M_TAN_KRR_a0.01 | 0.8114 | 0.8072 | 0.8134 | 0.8054 | 0.7913 |
| M_KNN_TAN_k5 | 0.7997 | 0.7977 | 0.8011 | 0.8016 | 0.7896 |
| M_KNN_TAN_k3 | 0.8023 | 0.7964 | 0.8025 | 0.8009 | 0.7742 |
| M_LIG2D_RIDGE_a1000 | 0.7803 | 0.7783 | 0.7843 | 0.7773 | 0.7579 |
| M_LIG2D_RIDGE_a100 | 0.7817 | 0.7753 | 0.7862 | 0.7826 | 0.7446 |
| M_LIG2D_RIDGE_a10 | 0.7894 | 0.7849 | 0.7920 | 0.7771 | 0.7441 |
| M_KNN_TAN_k1 | 0.7843 | 0.7807 | 0.7926 | 0.7762 | 0.7441 |
| FLAT | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


#### `mag` -- macro pair Spearman (higher is better)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| M_MAT_GP | 0.5261 | 0.5215 | 0.5257 | 0.5221 | 0.5114 |
| M_RBF_GP | 0.5206 | 0.5180 | 0.5220 | 0.5201 | 0.5085 |
| M_TANlin_KRR_a10 | 0.5050 | 0.5008 | 0.5073 | 0.5097 | 0.5049 |
| M_TANlin_KRR_a1 | 0.5119 | 0.5056 | 0.5164 | 0.5078 | 0.5049 |
| M_TAN_GP | 0.5143 | 0.5082 | 0.5178 | 0.5059 | 0.5042 |
| G14 | 0.5062 | 0.5025 | 0.5078 | 0.5111 | 0.5038 |
| M_KNN_TAN_k10 | 0.4990 | 0.4933 | 0.4995 | 0.4970 | 0.5034 |
| M_TANlin_KRR_a0.1 | 0.5157 | 0.5061 | 0.5188 | 0.5042 | 0.5005 |
| M_TANlin_KRR_a0.01 | 0.5113 | 0.5031 | 0.5159 | 0.4996 | 0.4996 |
| M_RBF_KRR_a0.1 | 0.5085 | 0.5076 | 0.5074 | 0.5053 | 0.4976 |
| M_RBF_KRR_a1 | 0.5084 | 0.5030 | 0.5068 | 0.5068 | 0.4914 |
| M_RBF_KRR_a10 | 0.5068 | 0.5028 | 0.5029 | 0.5070 | 0.4900 |
| M_TAN_KRR_a10 | 0.5028 | 0.4990 | 0.4968 | 0.4989 | 0.4874 |
| M_TAN_KRR_a1 | 0.5001 | 0.4942 | 0.4988 | 0.4900 | 0.4779 |
| M_TAN_KRR_a0.01 | 0.5003 | 0.4903 | 0.5002 | 0.4873 | 0.4775 |
| M_KNN_TAN_k5 | 0.4928 | 0.4849 | 0.4944 | 0.4914 | 0.4768 |
| M_TAN_KRR_a0.1 | 0.4992 | 0.4910 | 0.5007 | 0.4867 | 0.4745 |
| M_KNN_TAN_k3 | 0.4953 | 0.4817 | 0.4948 | 0.4860 | 0.4567 |
| M_LIG2D_RIDGE_a1000 | 0.4801 | 0.4784 | 0.4837 | 0.4758 | 0.4480 |
| M_LIG2D_RIDGE_a100 | 0.4745 | 0.4648 | 0.4847 | 0.4677 | 0.4242 |
| M_KNN_TAN_k1 | 0.4729 | 0.4694 | 0.4855 | 0.4554 | 0.4182 |
| M_LIG2D_RIDGE_a10 | 0.4661 | 0.4601 | 0.4772 | 0.4501 | 0.4052 |
| FLAT | nan | nan | nan | nan | nan |


#### `joint` -- extractant-macro MAE (lower is better), sorted by BP

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| C_RBF_KRR_a1 | 0.4889 | 0.4874 | 0.4858 | 0.4862 | 0.4980 |
| C_TAN_GP | 0.4878 | 0.4869 | 0.4854 | 0.4854 | 0.4988 |
| C_TAN_KRR_a10 | 0.4896 | 0.4884 | 0.4872 | 0.4863 | 0.4989 |
| C_TAN_KRR_a1 | 0.4875 | 0.4867 | 0.4848 | 0.4854 | 0.4990 |
| G14 | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| C_TAN_KRR_a0.1 | 0.4871 | 0.4862 | 0.4835 | 0.4846 | 0.5002 |
| C_KNN_TAN_k10 | 0.4944 | 0.4922 | 0.4908 | 0.4923 | 0.5089 |
| C_KNN_TAN_k5 | 0.4973 | 0.4959 | 0.4922 | 0.4948 | 0.5188 |
| C_KNN_TAN_k3 | 0.4976 | 0.4970 | 0.4935 | 0.4930 | 0.5220 |
| FLAT | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| J_RBF_GP | 0.5133 | 0.5195 | 0.5133 | 0.4753 | 0.6003 |
| J_RBF_KRR_a0.1 | 0.5107 | 0.5165 | 0.5076 | 0.4698 | 0.6009 |
| J_TAN_KRR_a0.1 | 0.5171 | 0.5245 | 0.5164 | 0.4699 | 0.6035 |
| J_TAN_GP | 0.5207 | 0.5278 | 0.5211 | 0.4755 | 0.6037 |
| J_RBF_KRR_a1 | 0.5191 | 0.5257 | 0.5200 | 0.4811 | 0.6039 |
| J_TAN_KRR_a1 | 0.5239 | 0.5316 | 0.5249 | 0.4824 | 0.6057 |
| J_TAN_KRR_a0.01 | 0.5181 | 0.5260 | 0.5167 | 0.4611 | 0.6070 |
| J_MAT_KRR_a1 | 0.5272 | 0.5333 | 0.5284 | 0.4831 | 0.6075 |
| J_RBF_KRR_a10 | 0.5533 | 0.5580 | 0.5556 | 0.5200 | 0.6130 |
| J_TAN_KRR_a10 | 0.5588 | 0.5643 | 0.5614 | 0.5239 | 0.6166 |
| J_KNN_TAN_k10 | 0.5640 | 0.5681 | 0.5581 | 0.4989 | 0.6362 |
| J_KNN_TAN_k5 | 0.5668 | 0.5721 | 0.5569 | 0.4903 | 0.6651 |
| J_KNN_TAN_k3 | 0.5619 | 0.5672 | 0.5540 | 0.4769 | 0.6917 |
| J_KNN_TAN_k1 | 0.6077 | 0.6165 | 0.5919 | 0.4914 | 0.7043 |


#### `joint` -- macro sign accuracy on strong pairs (higher is better)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| C_RBF_KRR_a1 | 0.8247 | 0.8263 | 0.8245 | 0.8340 | 0.8164 |
| C_TAN_KRR_a0.1 | 0.8225 | 0.8212 | 0.8198 | 0.8277 | 0.8140 |
| G14 | 0.8081 | 0.8093 | 0.8102 | 0.8085 | 0.8122 |
| C_TAN_KRR_a1 | 0.8195 | 0.8189 | 0.8191 | 0.8233 | 0.8119 |
| C_TAN_GP | 0.8153 | 0.8152 | 0.8157 | 0.8220 | 0.8118 |
| C_TAN_KRR_a10 | 0.8082 | 0.8081 | 0.8107 | 0.8122 | 0.8095 |
| C_KNN_TAN_k10 | 0.8158 | 0.8157 | 0.8172 | 0.8137 | 0.8029 |
| C_KNN_TAN_k5 | 0.8218 | 0.8208 | 0.8222 | 0.8229 | 0.8001 |
| C_KNN_TAN_k3 | 0.8217 | 0.8179 | 0.8196 | 0.8303 | 0.7977 |
| J_TAN_GP | 0.7048 | 0.6961 | 0.6971 | 0.7916 | 0.6125 |
| J_TAN_KRR_a0.1 | 0.7192 | 0.7102 | 0.7119 | 0.8003 | 0.6107 |
| J_TAN_KRR_a1 | 0.6890 | 0.6807 | 0.6857 | 0.7840 | 0.6084 |
| J_TAN_KRR_a0.01 | 0.7142 | 0.7000 | 0.7105 | 0.7988 | 0.6053 |
| J_RBF_KRR_a0.1 | 0.7017 | 0.6931 | 0.6999 | 0.7986 | 0.5868 |
| J_RBF_GP | 0.7005 | 0.6924 | 0.6935 | 0.7937 | 0.5807 |
| J_KNN_TAN_k10 | 0.6695 | 0.6623 | 0.6578 | 0.6904 | 0.5803 |
| J_RBF_KRR_a1 | 0.6955 | 0.6922 | 0.6914 | 0.7784 | 0.5632 |
| J_TAN_KRR_a10 | 0.6511 | 0.6449 | 0.6427 | 0.6787 | 0.5592 |
| J_MAT_KRR_a1 | 0.6681 | 0.6635 | 0.6567 | 0.7675 | 0.5501 |
| J_RBF_KRR_a10 | 0.6363 | 0.6311 | 0.6261 | 0.6853 | 0.5400 |
| J_KNN_TAN_k5 | 0.6587 | 0.6248 | 0.6526 | 0.7353 | 0.5109 |
| J_KNN_TAN_k3 | 0.6255 | 0.6187 | 0.6396 | 0.7525 | 0.4898 |
| J_KNN_TAN_k1 | 0.5779 | 0.5777 | 0.6216 | 0.8056 | 0.4829 |
| FLAT | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


#### `joint` -- macro pair Spearman (higher is better)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| C_RBF_KRR_a1 | 0.5391 | 0.5414 | 0.5387 | 0.5656 | 0.5193 |
| G14 | 0.5062 | 0.5025 | 0.5078 | 0.5111 | 0.5038 |
| C_TAN_GP | 0.5240 | 0.5223 | 0.5226 | 0.5536 | 0.5036 |
| C_TAN_KRR_a1 | 0.5289 | 0.5265 | 0.5272 | 0.5582 | 0.5026 |
| C_TAN_KRR_a0.1 | 0.5335 | 0.5287 | 0.5315 | 0.5651 | 0.5021 |
| C_TAN_KRR_a10 | 0.5127 | 0.5101 | 0.5135 | 0.5322 | 0.5012 |
| C_KNN_TAN_k10 | 0.5275 | 0.5228 | 0.5263 | 0.5409 | 0.4949 |
| C_KNN_TAN_k5 | 0.5413 | 0.5358 | 0.5436 | 0.5498 | 0.4730 |
| C_KNN_TAN_k3 | 0.5424 | 0.5316 | 0.5406 | 0.5655 | 0.4530 |
| J_TAN_GP | 0.4069 | 0.4026 | 0.3975 | 0.4857 | 0.2860 |
| J_TAN_KRR_a0.1 | 0.4229 | 0.4204 | 0.4139 | 0.4947 | 0.2829 |
| J_TAN_KRR_a1 | 0.3913 | 0.3848 | 0.3850 | 0.4760 | 0.2804 |
| J_TAN_KRR_a0.01 | 0.4231 | 0.4113 | 0.4160 | 0.4945 | 0.2762 |
| J_RBF_KRR_a0.1 | 0.3934 | 0.3944 | 0.3967 | 0.5148 | 0.2697 |
| J_RBF_GP | 0.3922 | 0.3906 | 0.3824 | 0.4936 | 0.2651 |
| J_RBF_KRR_a1 | 0.3819 | 0.3826 | 0.3755 | 0.4779 | 0.2445 |
| J_MAT_KRR_a1 | 0.3610 | 0.3582 | 0.3474 | 0.4667 | 0.2327 |
| J_KNN_TAN_k10 | 0.3680 | 0.3567 | 0.3584 | 0.4058 | 0.2321 |
| J_TAN_KRR_a10 | 0.3271 | 0.3228 | 0.3143 | 0.3492 | 0.2223 |
| J_RBF_KRR_a10 | 0.3219 | 0.3201 | 0.3120 | 0.3649 | 0.2170 |
| J_KNN_TAN_k5 | 0.3496 | 0.3045 | 0.3487 | 0.4411 | 0.0834 |
| J_KNN_TAN_k3 | 0.3171 | 0.2877 | 0.3348 | 0.4701 | 0.0438 |
| J_KNN_TAN_k1 | 0.2226 | 0.2041 | 0.3127 | 0.5312 | 0.0018 |
| FLAT | nan | nan | nan | nan | nan |

---

## 3. What the tables say, per question asked

### 3.1 Tanimoto kernel ridge / kernel logistic for the direction — harmful

Every direction arm is worse than `FLAT` under BP. Best is `D_TAN_KRR_a0.01` at 0.6373 against
`FLAT`'s 0.5885 and `G14`'s 0.5001. The paired bootstrap on the three most promising direction arms:

| design | arm | vs `G14` | vs `FLAT` (p) |
|---|---|---|---|
| BP | `D_TAN_SVC_C1` | −0.2022 (p = 0.0000) | −0.1138 (p = 0.0006) |
| BP | `D_TAN_KLR_l0.1` | −0.1915 (p = 0.0000) | −0.1031 (p = 0.0014) |
| BP | `D_ECFPraw_LOGIT_C0.1` | −0.1866 (p = 0.0000) | −0.0981 (p = 0.0076) |

Full table in `results/contrasts_dir.csv`.

Direction accuracy in gen14's own units (extractant unit, chemotype block,
`results/direction_accuracy.csv`) shows exactly where it breaks:

| rule | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| `G14_TOPO39_LOGIT` | 0.8400 | 0.8346 | 0.8400 | 0.8346 | **0.8205** |
| `ALWAYS_HEAVY` | 0.5586 | 0.5586 | 0.5586 | 0.5586 | 0.5586 |
| `TAN_KLR_l0.1` | 0.7115 | 0.6761 | 0.7091 | 0.8361 | **0.4762** |
| `ECFPraw_LOGIT_C0.1` | 0.6991 | 0.6674 | 0.6907 | 0.8271 | **0.4749** |
| `ECFPz_LOGIT_C1` | 0.6359 | 0.5989 | 0.6335 | 0.7756 | 0.5269 |
| `KNN_TAN_k10` | 0.6073 | 0.6195 | 0.6073 | 0.6273 | 0.5164 |
| `RBF_KLR_l0.1` | 0.6430 | 0.6251 | 0.6287 | 0.7537 | 0.5174 |

The fingerprint rules *do* learn something under B (0.70 against 0.56 for always-heavy) and a lot
under A (0.83). Under BP they fall to **below chance**. The 39 hand-built donor-topology columns
lose almost nothing (0.840 to 0.821). Whatever ECFP similarity is encoding about the direction is
not the donor set; it is something that travels with the laboratory, and BP removes it.

### 3.2 The linear L2 logistic on 2048 raw bits — asked, answered, no

Gen14's finding was that a linear model beat trees on 39 columns because only 22 distinct vectors
exist. The natural next question was a linear model on 2048 sparse bits. It behaves like the kernel:
`D_ECFPraw_LOGIT_C0.1` reaches 0.699 direction accuracy under B and 0.475 under BP; in MAE it is
0.5736 under B (marginally better than `FLAT`) and 0.6866 under BP. Standardising the bits
(`ECFPz`, which multiplies a bit present in 2 % of ligands by about 7) is *worse* under B and
*better* under BP, and both are far short of `G14` everywhere. Reported at all three C values.

### 3.3 The GP with the noise fixed at the replicate sd — the closest thing to a win, and it is not one

`M_TAN_GP` is the single best-motivated arm here: the Tanimoto kernel, the signal variance read off
the training fold, and the noise fixed at the corpus replicate sd of 0.237, so the shrinkage is
principled rather than tuned. Its five-design record against `G14` (`results/contrasts_mag.csv`):

| design | point | p | BCa 95 % | loco sign stable | passes P1 |
|---|---|---|---|---|---|
| B | +0.0174 | 0.136 | (−0.0035, 0.0363) | yes | no |
| BR | +0.0121 | 0.171 | (−0.0059, 0.0291) | yes | no |
| BQ | +0.0165 | 0.160 | (−0.0041, 0.0357) | yes | no |
| A | +0.0250 | **0.016** | (0.0070, 0.0477) | yes | **yes** |
| BP | **+0.0016** | 0.785 | (−0.0106, 0.0169) | **no** | no |

The sign is consistent across all five designs, which is the protocol's first requirement — but the
magnitude under BP is 0.0016 with a bootstrap standard error of 0.0074, i.e. indistinguishable from
zero, and the leave-one-chemotype-out range straddles zero (−0.0016 to +0.0073). The only design
where it passes is **A, the leaky one**. That is the signature of a leak, not of a discovery, and
section 4 shows what the leak is.

This is also a seventh independent prior on |a| that lands within 0.007 of a constant under BP,
alongside the six the programme already had. The magnitude remains unpredictable.

### 3.4 RDKit descriptor kernels and the ridge on LIG2D

RBF and Matérn on the standardised 206 columns, and a plain ridge on them, are all worse than the
Tanimoto arms and all worse than `G14`. `D_LIG2D_RIDGE_a10` is the best descriptor direction arm at
0.6172 under BP — still worse than `FLAT`. `M_RBF_KRR_a0.1` is the best descriptor magnitude arm at
0.5034, i.e. 0.003 worse than `G14`. `M_LIG2D_RIDGE_a10` at 0.6199 is worse than `FLAT`. The 206
RDKit descriptors are largely size and complexity counts, and under a hold-out that removes the
whole Tanimoto-0.7 cluster there is nothing left in them.

### 3.5 kNN-in-Tanimoto, and the applicability-domain curve

k = 1, 3, 5, 10, similarity-weighted, over ligand units. The direction kNN is the worst family in
the study; the magnitude kNN degrades monotonically as k falls (BP: k=10 0.5038, k=5 0.5167, k=3
0.5312, k=1 0.5597) — i.e. **the nearest neighbour is worse than the training mean**, and the arm
only approaches `G14` by averaging enough neighbours to become the training mean.

**The applicability-domain curve is flat, and the reason is structural.** Distribution of the
Tanimoto to the nearest training ligand over held-out cells (`results/ad_smax_quantiles.csv`):

| design | q05 | q25 | q50 | q75 | q95 | frac < 0.4 | frac = 1.0 |
|---|---|---|---|---|---|---|---|
| B | 0.3585 | 0.5600 | 0.6667 | 0.6667 | 0.6786 | 0.104 | 0.000 |
| BR | 0.3585 | 0.5366 | 0.6591 | 0.6667 | 0.6786 | 0.106 | 0.000 |
| BQ | 0.3585 | 0.5500 | 0.6452 | 0.6591 | 0.6786 | 0.104 | 0.000 |
| A | 0.3953 | 0.6786 | 0.8621 | 1.0000 | 1.0000 | 0.060 | **0.425** |
| BP | 0.3137 | 0.4571 | 0.6296 | 0.6667 | 0.6786 | 0.153 | 0.000 |

Design B is *defined* as single-linkage Tanimoto-0.7 chemotype hold-out on this very fingerprint
block, so under B/BR/BQ/BP the nearest training ligand is **capped below 0.7 by construction** —
q95 is 0.679 and not one held-out cell has a close analogue. The kernel is extrapolating for every
single prediction in every deployment-relevant design. Under A, 42.5 % of held-out cells have an
exact ligand match at Tanimoto 1.0, and that is the only design where the kernel passes anything.

Consequently the error does not track the similarity. Spearman(s_max, |a| error) over
well-determined cells: B +0.089, BR +0.106, BQ −0.012, A +0.126, **BP −0.004**. The decile table
under BP (`results/ad_deciles.csv`) is flat, and its *highest*-similarity decile has the *largest*
error, because those cells happen to have larger true |a|:

| decile | n | mean s_max | MAE GP | MAE `G14` | delta |
|---|---|---|---|---|---|
| 0 | 145 | 0.2853 | 0.3014 | 0.3338 | −0.0325 |
| 1 | 144 | 0.4086 | 0.3311 | 0.3723 | −0.0412 |
| 2 | 145 | 0.4623 | 0.2591 | 0.2621 | −0.0030 |
| 3 | 144 | 0.5228 | 0.2156 | 0.2251 | −0.0095 |
| 4 | 145 | 0.6074 | 0.2921 | 0.2906 | +0.0016 |
| 5 | 144 | 0.6551 | 0.2671 | 0.2762 | −0.0090 |
| 6 | 144 | 0.6667 | 0.2634 | 0.2799 | −0.0165 |
| 7 | 145 | 0.6667 | 0.2536 | 0.2726 | −0.0190 |
| 8 | 144 | 0.6674 | 0.2715 | 0.2797 | −0.0082 |
| 9 | 145 | 0.6787 | 0.4262 | 0.4277 | −0.0015 |

The GP posterior sd is no better as a confidence: its correlation with the error is **−0.032** under
BP (the wrong sign — high confidence goes with high error), and the low-sd half of the cells has a
*higher* MAE (0.3065) than the high-sd half (0.2695). `results/ad_confidence.csv`.

This is worth having whatever the headline says, and the finding is a clean negative: **this corpus
has no applicability domain in fingerprint space under any deployment-relevant design**, because the
split that defines the deployment is the same Tanimoto threshold the kernel would need to cross.

### 3.6 The curvature, and the joint arms

The `C_*` arms are the only family whose sign is consistent across all five designs, and they are
the best arms in the study under BP — but the size is 0.001 to 0.002:

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| `C_RBF_KRR_a1` | 0.4889 | 0.4874 | 0.4858 | 0.4862 | **0.4980** |
| `C_TAN_GP` | 0.4878 | 0.4869 | 0.4854 | 0.4854 | 0.4988 |
| `C_TAN_KRR_a1` | 0.4875 | 0.4867 | 0.4848 | 0.4854 | 0.4990 |
| `G14` | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |

Paired bootstrap under BP: `C_TAN_KRR_a1` +0.0011 (p = 0.856), `C_TAN_GP` +0.0012 (p = 0.808),
`C_RBF_KRR_a1` +0.0021 (p = 0.704) — and **loco sign stable = False for all three**. Only 41 to 43
of 90 extractants improve, i.e. fewer than half. This is noise around the constant b, and it joins
the programme's existing list of failed curvature priors rather than breaking it.

Predicting both coefficients outright (`J_*`) is much worse — `J_TAN_KRR_a0.1` is 0.6035 under BP,
significantly worse than `G14` (−0.1035, p = 0.012) and no better than `FLAT`. A kernel that
regresses the signed a shrinks it toward zero and destroys the direction bit that carries all of
`G14`'s value; that is the same failure gen13's `G13_FULL` tree regression had, in a new estimator.

---

## 4. Why the kernel gains under B/BR/BQ/A and not under BP

Design BP removes, from training, every cell whose *publication* also appears among the held-out
cells; B, BR and BQ do not. Splitting the GP magnitude arm's error by whether the test cell's
publication was in training (`results/publeak_strata.csv`, per-cell amplitude MAE):

| design | stratum | n | MAE GP | MAE `G14` | GP − `G14` |
|---|---|---|---|---|---|
| B | pub **in** train | 558 | 0.2594 | 0.2831 | **−0.0237** |
| B | pub NOT in train | 887 | 0.3127 | 0.3097 | **+0.0030** |
| BQ | pub **in** train | 558 | 0.2602 | 0.2832 | −0.0230 |
| BQ | pub NOT in train | 887 | 0.3100 | 0.3051 | +0.0049 |
| BR | pub **in** train | 517 | 0.2675 | 0.2921 | −0.0245 |
| BR | pub NOT in train | 928 | 0.2990 | 0.3032 | −0.0042 |
| A | pub **in** train | 688 | 0.2634 | 0.2863 | −0.0229 |
| BP | pub NOT in train | 1445 | 0.2882 | 0.3020 | −0.0139 |

The entire gain lives in the "publication in training" stratum and vanishes — or reverses — in the
complement. Under BP that stratum is empty by construction (`frac_pub_in_train` = 0.000,
`results/publeak_summary.csv`), and the gain goes with it. The magnitude of the radius coefficient
is strongly a property of the laboratory — the programme's own `O_PUBMAG` oracle scores 0.448
against `G14`'s 0.500 — and a similarity kernel is an efficient way to find the laboratory, because
a paper that reports one ligand usually reports several close relatives.

---

## 5. Confounds checked

The protocol requires guarding every correlation against publication identity, the number of metals
a cell measured, and chemotype. Over all 3321 pairs of distinct well-determined extractants
(`results/ad_confounds.csv`), relating the Tanimoto between two ligands to how similar their |a| and
their direction are:

| stratum | n pairs | Spearman(Tanimoto, delta abs a) | same-sign rate, top-quartile Tanimoto | same-sign rate, bottom quartile |
|---|---|---|---|---|
| all extractant pairs | 3321 | +0.012 | 0.616 | 0.504 |
| same chemotype | 293 | −0.114 | 0.824 | 0.767 |
| **different chemotype (the deployed regime)** | 3028 | −0.010 | **0.505** | **0.501** |
| share a publication | 225 | −0.050 | 0.842 | 0.839 |
| share no publication | 3096 | +0.033 | 0.543 | 0.508 |
| **diff chemotype AND no shared publication** | 2920 | +0.006 | **0.464** | 0.492 |
| matched n_metals (delta n at most 1), diff chemotype | 1107 | −0.107 | 0.556 | 0.588 |

The apparent whole-corpus signal — 0.616 same-sign at high Tanimoto against 0.504 at low — is
entirely carried by pairs that share a chemotype or a publication. Once both confounds are removed,
which is exactly the BP regime, high Tanimoto similarity gives a same-direction rate of **0.464**,
i.e. below a coin flip. Matching on the number of metals measured does not rescue it either. There
is no residual similarity-to-selectivity relation left to model.

Leave-one-chemotype-out under BP (`results/ad_loco.csv`): the GP magnitude arm helps in 24 of 40
chemotypes and hurts in 16, and the whole BP-level gain is dominated by two chemotypes (`sc061`,
n = 85, delta = −0.119) while `sc012` (n = 5) goes the other way by +0.332. Not stable.

---

## 6. Honest limits

* Only the ECFP block already in the cohort (2048 Morgan bits) and the LIG2D block (206 RDKit
  descriptors) were used. Other fingerprints — MACCS, Avalon, atom-pair, a different Morgan radius,
  or a count-based rather than binary encoding — were not tried, and a graph kernel (Weisfeiler-
  Lehman, shortest-path) was not tried. Section 5 argues these would not help, since the
  confound-free similarity-to-direction relation is already at chance, but that is an argument, not
  a measurement.
* The GP is a fixed-hyperparameter GP, not a marginal-likelihood-optimised one. That was the point —
  the noise is set from the corpus replicate sd so nothing is tuned — but a full GP with a learned
  length scale and an ARD kernel is a different (and, on roughly 80 to 300 ligands with a chemotype
  hold-out, probably an over-fitting) estimator.
* The kernel logistic is hand-rolled (L-BFGS on the dual with an unpenalised intercept). It was
  checked against `SVC(kernel='precomputed')` on the same Gram — the two agree in ranking and both
  fail the same way — but it has no independent reference implementation here.
* `C_RBF_KRR_a1`'s +0.0021 under BP is not proven to be zero; it is proven not to be distinguishable
  from zero at this corpus size (mde_80 = 0.0164 under BP, so the study could only have detected a
  gain of about 0.016 or more with 80 % power). A gain of 0.002 would be worthless anyway.
* The applicability-domain conclusion is conditional on the split definition. It says the corpus
  offers no near-analogue regime *under the hold-outs this programme judges by*; it does not say a
  deployment that happens to query a close analogue of a training ligand would be badly served — in
  fact design A shows such a deployment is served well (`M_TAN_GP` 0.4672, +0.025 over `G14`,
  p = 0.016). If a real user's queries look like design A rather than design B, the kernel is worth
  having. That is a statement about the user, not about the model.

---

## 7. What would have had to be true, and what is next

For a similarity kernel to work here, two things had to hold: (a) ligands close in fingerprint
space had to have similar separation curves once chemotype and publication were held fixed, and
(b) held-out ligands had to have close training neighbours. **Neither holds.** (a) fails at
0.464 same-direction for high-Tanimoto, different-chemotype, different-publication pairs — chance is
0.5. (b) fails by construction, because the hold-out threshold and the kernel's similarity are the
same number computed on the same bits.

Nothing here is worth deploying, and I would not recommend spending more of this programme's budget
on 2D-similarity methods of any flavour: the failure is not in the estimator — a kernel, a linear
model on raw bits, a kNN and a GP all fail identically — it is that the endpoint is not a smooth
function of 2D structural similarity at the resolution this hold-out demands.

The one thing this run adds that is worth carrying forward is negative infrastructure: `publeak.py`
and `ad.py` give any future candidate a cheap test for the same failure mode — split your gain by
whether the test cell's publication is in training, and if it lives in the "in training" stratum you
have found the laboratory, not the chemistry. Given how much of this corpus's magnitude signal is a
publication property (`O_PUBMAG` 0.448), that check is worth running on every future candidate.
