# gen15 / `embed` -- does a pretrained molecular representation carry the magnitude and curvature?

**Verdict: null, and on the direction actively harmful.** Three pretrained chemical language models
were rebuilt from scratch and scored as gen15 arms under all five designs. Not one arm of any kind
-- direction, magnitude, curvature, composite, or concatenated with gen14's own columns -- beats
`G14` under BP. The closest, MoLFormer's RBF magnitude head, is 0.5029 against `G14`'s
0.5001, a loss of 0.0028 with a paired *p* of 0.92. Used for the **direction**, every
embedding is a catastrophe: the best of them reaches 0.5687 macro accuracy under BP against
TOPO39's 0.8205, which is the always-heavy floor (0.5586) to within noise.

One thing did look alive and is reported in full below, because it is the most interesting number in
the experiment and because it does not survive: a ridge on MoLFormer's mean-pooled vector predicts
`log|a|` out of fold under BP at Spearman +0.23 (partial +0.25 with `n_metals` removed,
leave-one-chemotype-out stable), where TOPO39, LEAN209, Morgan counts, both ChemBERTa checkpoints and
a four-number size block all sit at or below zero. **A permutation null kills it.** Shuffling the
ligand->vector map -- every ligand keeps a real MoLFormer vector, just somebody else's -- gives a
null with sd 0.1815 and a 95th percentile of 0.2525, *above* the observed
0.2277 (one-sided *p* = 0.098). With 90 ligands in 45 chemotypes the out-of-fold rank
correlation in this pipeline is so noisy that +0.23 is an ordinary draw from a representation that
knows nothing about the ligands.

## Bench sanity checks (this run's own numbers, not quoted)

| quantity | expected from the programme | measured here |
|:---|---:|---:|
| `G14` extractant-macro MAE, BP | 0.500 | **0.5001** |
| `FLAT` extractant-macro MAE, BP | 0.589 | **0.5885** |
| TOPO39 direction macro accuracy, BP | 0.821 | **0.8205** |
| always-heavy direction macro accuracy | 0.559 | **0.5586** |

All four reproduce, in three independently launched scoring runs (`board_main...`, `board_final`,
`board_molformer` each carry `FLAT` and `G14` and agree to the fourth decimal).

## What was built

The embedding table the repo's manifest recorded but never shipped was rebuilt for the 90 distinct
extractant SMILES of the gen15 cohort, keeping both the mean-pooled last hidden state over
non-padding tokens and the CLS/first-token vector for each model:

| tag | model id | dim | pools | distinct rows |
|:---|:---|---:|:---|:---|
| `chemberta_mtr` | `DeepChem/ChemBERTa-77M-MTR` | 384 | mean, cls | 89/90 |
| `chemberta_mlm` | `DeepChem/ChemBERTa-77M-MLM` | 384 | mean, cls | 88/90 |
| `molformer` | `ibm-research/MoLFormer-XL-both-10pct` | 768 | mean, cls | 90/90 |

`transformers` 5.16.1, `torch` 2.14.0+cpu, 90 x 3072 float32 -> `ligand_embeddings.parquet`.

**MoLFormer did load, contrary to the first pass.** An earlier attempt recorded it as failed with an
`OSError` naming `pytorch_model.bin`; that diagnosis was wrong. The repo does ship
`model.safetensors`, and the real cause was that the 187 MB weight download kept stalling on this
machine at ~17 MB, after which `transformers` fell back to looking for a `.bin` and reported the
fallback's absence. Retrying `hf_hub_download` until the blob completed fixed it (680 s), and the
model then loads on transformers 5.16.1 with `trust_remote_code=True`; five `lm_head.*` keys are
reported unexpected, which is what loading an encoder out of an MLM checkpoint with `AutoModel`
should do. Encoding the 90 SMILES then took 11.8 s. Morgan **count** fingerprints at radius 2 and 3
(2048 bits, unset bits dropped) are carried throughout as the cheap control: a pretrained
representation has to beat the substructure count, not only the 39 hand-built topology columns.

## How to read the tables

`MAG_*` and `CUR_*` arms keep gen14's direction and replace only the magnitude or only the curvature,
so they measure what the representation adds on top of the deployed model. `DIR_*` replaces only the
direction. `COMP_*` uses the representation for both. `CONCAT_*` gives the logistic gen14's 39
topology columns *and* 16 principal components of the embedding, which asks whether the embedding
adds anything TOPO39 does not already have. Every fit -- imputation, standardisation, PCA, penalty --
happens strictly inside the training fold of every fold of every design. `C = 1` is gen14's rule; the
C sweep is shown so its whole range is visible, and nothing was selected on a held-out score.

## The five-design result

No arm beats `G14` under BP. Under B/BR/BQ/A a handful of MoLFormer magnitude and curvature arms are
nominally ahead of `G14` by 0.003-0.012, and **all of them turn negative under BP** -- the exact
sign-inconsistency the protocol exists to catch. Even where positive, every one of those contrasts
has a 95 % interval straddling zero (*p* between 0.18 and 0.99) and none passes the programme's
`passes_P1` gate. There is no arm here, under any design, that is distinguishable from `G14`.

## Confounds

* **Near-duplicate ligands.** At raw cosine >= 0.99, 28/90 ligands have a twin under
  `chemberta_mtr__mean` (8 of them cross-chemotype), 12/90 under `chemberta_mlm__mean` (0
  cross-chemotype), 20/90 under `morgan2` (11 cross-chemotype), and only 5/90 under
  `molformer__mean` (0 cross-chemotype). This is the confound that would *inflate* a gain, and there
  is no gain to inflate; it is reported because it would have mattered had the sign gone the other
  way. Note that the nearest neighbour shares the test ligand's chemotype only 34-54 % of the time,
  so the chemotype hold-out does **not** remove every near neighbour -- a real caveat for anyone who
  later does find a gain here.
* **Publication identity.** Designs B/BR/BQ/A leave publications free; BP masks them. Every arm's
  best showing is under A (exact extractant, leaky) and the collapse from A to BP is the whole story:
  `DIR_cbmtr_mean_C1` goes 0.5070 -> 0.6526, direction accuracy 0.7807 -> 0.5032. An embedding is a
  function of the ligand alone and cannot leak conditions, but a ligand studied by one laboratory
  only is still a publication label in disguise under any design that is not BP.
* **Number of measured metals.** Spearman +0.4357 with `|a|` in this corpus. The embedding norm's
  correlation with `|a|` is <= 0.18 in magnitude for every block, so the size axis is not smuggling
  the `n_metals` confound in, and the one live correlation is reported partialled on `n_metals`.
* **Leave-one-chemotype-out.** TOPO39 beats the embedding in 26 of 45 chemotypes under BP (7 losses,
  12 ties). Dropping any single chemotype, the best the embedding's macro accuracy reaches is 0.5504
  and the worst TOPO39 falls to is 0.7735 -- they do not overlap, so the direction verdict does not
  rest on any one chemotype. The gen13 contrast machinery's own `loco_sign_stable` flag agrees: every
  `DIR_*` contrast against `G14` is negative and sign-stable under BP.

## What would have had to be true

For the direction, the embedding would have had to encode which donor set a ligand presents to the
metal in a way a 90-example linear probe can read. It does not: mean-pooling a SMILES transformer
produces a vector dominated by size and gross composition, and the 39 topology columns are a
hand-built answer to exactly the question the endpoint asks. For the magnitude, a correlation would
have had to clear a null whose sd is 0.18 -- which at 90 ligands in 45 chemotypes means
nothing short of Spearman ~ 0.45 is interpretable from this probe at all. That number, not the
embedding, is the real finding: **the corpus cannot resolve a magnitude model, whatever the
representation.** The seven priors the programme already retired were not unlucky.

---

# Tables


### Extractant-macro MAE, every arm x every design

| arm                    |      B |     BR |     BQ |      A |     BP |
|:-----------------------|-------:|-------:|-------:|-------:|-------:|
| FLAT                   | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| G14                    | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| MAG_mf_mean_rbf        | 0.4851 | 0.4868 | 0.484  | 0.4797 | 0.5029 |
| MAG_morgan2_rbf        | 0.5052 | 0.5016 | 0.5067 | 0.5051 | 0.5033 |
| MAG_morgan3_rbf        | 0.504  | 0.5    | 0.5045 | 0.508  | 0.5039 |
| CUR_cbmtr_mean_cos     | 0.4924 | 0.4927 | 0.4896 | 0.4942 | 0.5068 |
| CUR_morgan2_ridge      | 0.4922 | 0.4914 | 0.487  | 0.4897 | 0.5068 |
| CUR_mf_mean_ridge      | 0.4917 | 0.4911 | 0.4841 | 0.4891 | 0.5076 |
| MAG_morgan2_cos        | 0.5036 | 0.5022 | 0.5022 | 0.4986 | 0.509  |
| MAG_cbmtr_mean_rbf     | 0.4963 | 0.4981 | 0.4954 | 0.4903 | 0.51   |
| MAG_morgan3_cos        | 0.4959 | 0.4954 | 0.4932 | 0.4973 | 0.5107 |
| MAG_mf_mean_cos        | 0.4861 | 0.4881 | 0.4806 | 0.4831 | 0.5111 |
| MAG_mf_mean_ridge      | 0.4933 | 0.4955 | 0.489  | 0.4886 | 0.5157 |
| CUR_cbmtr_mean_ridge   | 0.5023 | 0.5028 | 0.4981 | 0.5066 | 0.52   |
| MAG_cbmtr_mean_ridge1k | 0.5077 | 0.5086 | 0.5038 | 0.4964 | 0.5243 |
| CUR_cbmtr_mean_sign    | 0.5141 | 0.512  | 0.5112 | 0.5253 | 0.5278 |
| MAG_cbmtr_mean_cos     | 0.5103 | 0.5122 | 0.5038 | 0.5004 | 0.5286 |
| CUR_mf_mean_sign       | 0.5196 | 0.5102 | 0.5053 | 0.5045 | 0.53   |
| MAG_cbmtr_mean_ridge   | 0.5151 | 0.5233 | 0.5108 | 0.5137 | 0.5407 |
| CONCAT_mf_mean_P16     | 0.5265 | 0.5239 | 0.5237 | 0.5155 | 0.5471 |
| CONCAT_cbmtr_mean_P16  | 0.5623 | 0.5541 | 0.5632 | 0.5228 | 0.5532 |
| COMP_mf_mean           | 0.5647 | 0.5701 | 0.5641 | 0.5141 | 0.6104 |
| COMP_cbmtr_mean        | 0.5709 | 0.5798 | 0.5705 | 0.5167 | 0.6304 |
| DIR_cbmlm_mean_C1      | 0.5945 | 0.5991 | 0.6084 | 0.5305 | 0.6364 |
| DIR_morgan2_C1         | 0.6101 | 0.6301 | 0.6164 | 0.5293 | 0.6395 |
| DIR_mf_mean_C1         | 0.5885 | 0.5915 | 0.586  | 0.5297 | 0.6397 |
| DIR_mf_mean_C0.1       | 0.5841 | 0.5888 | 0.5816 | 0.5304 | 0.6441 |
| DIR_cbmtr_cls_C1       | 0.5588 | 0.5655 | 0.5626 | 0.5134 | 0.6467 |
| DIR_cbmtr_mean_P16     | 0.6157 | 0.6285 | 0.6389 | 0.5806 | 0.6496 |
| DIR_cbmtr_mean_C1      | 0.597  | 0.5995 | 0.5957 | 0.507  | 0.6526 |
| DIR_mf_mean_P16        | 0.5973 | 0.6009 | 0.5977 | 0.5626 | 0.653  |
| DIR_cbmtr_mean_C0.1    | 0.5977 | 0.6047 | 0.6011 | 0.5181 | 0.6534 |
| DIR_mf_mean_C0.01      | 0.5913 | 0.5892 | 0.589  | 0.5316 | 0.6565 |
| DIR_mf_cls_C1          | 0.6272 | 0.6275 | 0.6208 | 0.5509 | 0.6784 |
| DIR_cbmtr_mean_C0.01   | 0.6134 | 0.6303 | 0.6158 | 0.5447 | 0.6787 |
| DIR_cbmlm_cls_C1       | 0.6819 | 0.6835 | 0.6729 | 0.5673 | 0.6999 |


### Chemotype-blocked paired gain vs G14 (positive = better than G14)

| candidate              |       B |      BR |      BQ |       A |      BP |
|:-----------------------|--------:|--------:|--------:|--------:|--------:|
| MAG_mf_mean_rbf        |  0.0081 |  0.0038 |  0.0066 |  0.0124 | -0.0028 |
| MAG_morgan2_rbf        | -0.012  | -0.011  | -0.0161 | -0.013  | -0.0032 |
| MAG_morgan3_rbf        | -0.0108 | -0.0094 | -0.0139 | -0.0158 | -0.0038 |
| CUR_cbmtr_mean_cos     |  0.0008 | -0.0021 |  0.0009 | -0.0021 | -0.0067 |
| CUR_morgan2_ridge      |  0.001  | -0.0008 |  0.0035 |  0.0024 | -0.0067 |
| CUR_mf_mean_ridge      |  0.0015 | -0.0005 |  0.0065 |  0.003  | -0.0075 |
| MAG_morgan2_cos        | -0.0104 | -0.0116 | -0.0117 | -0.0064 | -0.009  |
| MAG_cbmtr_mean_rbf     | -0.0031 | -0.0075 | -0.0048 |  0.0018 | -0.0099 |
| MAG_morgan3_cos        | -0.0027 | -0.0048 | -0.0026 | -0.0052 | -0.0106 |
| MAG_mf_mean_cos        |  0.0071 |  0.0025 |  0.0099 |  0.009  | -0.011  |
| MAG_mf_mean_ridge      | -0.0001 | -0.0049 |  0.0015 |  0.0035 | -0.0156 |
| CUR_cbmtr_mean_ridge   | -0.0091 | -0.0123 | -0.0076 | -0.0144 | -0.0199 |
| MAG_cbmtr_mean_ridge1k | -0.0144 | -0.018  | -0.0132 | -0.0043 | -0.0242 |
| CUR_cbmtr_mean_sign    | -0.0209 | -0.0214 | -0.0207 | -0.0331 | -0.0277 |
| MAG_cbmtr_mean_cos     | -0.0171 | -0.0216 | -0.0133 | -0.0083 | -0.0286 |
| CUR_mf_mean_sign       | -0.0264 | -0.0196 | -0.0147 | -0.0124 | -0.0299 |
| MAG_cbmtr_mean_ridge   | -0.0219 | -0.0327 | -0.0202 | -0.0216 | -0.0406 |
| CONCAT_mf_mean_P16     | -0.0333 | -0.0333 | -0.0331 | -0.0234 | -0.047  |
| CONCAT_cbmtr_mean_P16  | -0.0691 | -0.0635 | -0.0727 | -0.0307 | -0.0531 |
| COMP_mf_mean           | -0.0714 | -0.0795 | -0.0736 | -0.0219 | -0.1103 |
| COMP_cbmtr_mean        | -0.0777 | -0.0892 | -0.08   | -0.0246 | -0.1303 |
| DIR_cbmlm_mean_C1      | -0.1013 | -0.1085 | -0.1179 | -0.0384 | -0.1363 |
| DIR_morgan2_C1         | -0.1169 | -0.1395 | -0.1258 | -0.0371 | -0.1394 |
| DIR_mf_mean_C1         | -0.0953 | -0.1009 | -0.0955 | -0.0376 | -0.1396 |
| DIR_mf_mean_C0.1       | -0.0909 | -0.0982 | -0.0911 | -0.0383 | -0.144  |
| DIR_cbmtr_cls_C1       | -0.0656 | -0.0749 | -0.072  | -0.0212 | -0.1466 |
| DIR_cbmtr_mean_P16     | -0.1225 | -0.1379 | -0.1484 | -0.0885 | -0.1495 |
| DIR_cbmtr_mean_C1      | -0.1038 | -0.1089 | -0.1052 | -0.0149 | -0.1526 |
| DIR_mf_mean_P16        | -0.1041 | -0.1103 | -0.1072 | -0.0705 | -0.1529 |
| DIR_cbmtr_mean_C0.1    | -0.1045 | -0.1141 | -0.1106 | -0.026  | -0.1533 |
| DIR_mf_mean_C0.01      | -0.0981 | -0.0986 | -0.0984 | -0.0395 | -0.1564 |
| DIR_mf_cls_C1          | -0.134  | -0.1369 | -0.1302 | -0.0588 | -0.1783 |
| DIR_cbmtr_mean_C0.01   | -0.1201 | -0.1397 | -0.1252 | -0.0526 | -0.1786 |
| DIR_cbmlm_cls_C1       | -0.1887 | -0.1929 | -0.1823 | -0.0752 | -0.1998 |


### Direction macro accuracy over extractants

| model               |      B |     BR |     BQ |      A |     BP |
|:--------------------|-------:|-------:|-------:|-------:|-------:|
| TOPO39              | 0.84   | 0.8346 | 0.84   | 0.8346 | 0.8205 |
| chemberta_mlm__mean | 0.6457 | 0.6473 | 0.6203 | 0.7954 | 0.5687 |
| molformer__mean     | 0.6568 | 0.6526 | 0.6519 | 0.7981 | 0.5499 |
| morgan2             | 0.6189 | 0.5887 | 0.6091 | 0.7848 | 0.5428 |
| chemberta_mtr__mean | 0.6262 | 0.62   | 0.6311 | 0.7807 | 0.5032 |
| chemberta_mtr__cls  | 0.6792 | 0.6684 | 0.6694 | 0.7861 | 0.5001 |
| morgan3             | 0.5652 | 0.5552 | 0.5896 | 0.7458 | 0.4771 |
| molformer__cls      | 0.5829 | 0.5876 | 0.5742 | 0.7171 | 0.4714 |
| chemberta_mlm__cls  | 0.476  | 0.4767 | 0.4883 | 0.7064 | 0.4457 |
| MAJORITY            | 0.3478 | 0.3478 | 0.3517 | 0.4414 | 0.3473 |


### Direction under BP, with the chemotype-blocked bootstrap

| model               |   macro_accuracy |   ci_low |   ci_high |   pooled_accuracy |     gain |   p_two_sided |
|:--------------------|-----------------:|---------:|----------:|------------------:|---------:|--------------:|
| TOPO39              |           0.8205 |   0.7037 |    0.9    |            0.782  | nan      |      nan      |
| chemberta_mlm__mean |           0.5687 |   0.4788 |    0.7038 |            0.4907 |  -0.2518 |        0.0058 |
| molformer__mean     |           0.5499 |   0.4582 |    0.6818 |            0.436  |  -0.2706 |        0.0014 |
| morgan2             |           0.5428 |   0.4665 |    0.6527 |            0.474  |  -0.2777 |        0.0008 |
| chemberta_mtr__mean |           0.5032 |   0.4078 |    0.6056 |            0.499  |  -0.3173 |        0.0006 |
| chemberta_mtr__cls  |           0.5001 |   0.4071 |    0.62   |            0.4657 |  -0.3204 |        0.0004 |
| morgan3             |           0.4771 |   0.4003 |    0.5926 |            0.4048 |  -0.3434 |        0      |
| molformer__cls      |           0.4714 |   0.4073 |    0.582  |            0.4062 |  -0.3492 |        0.0002 |
| chemberta_mlm__cls  |           0.4457 |   0.3494 |    0.5885 |            0.4436 |  -0.3748 |        0.0008 |
| MAJORITY            |           0.3473 |   0.248  |    0.5055 |            0.274  |  -0.4732 |        0      |


### Constant-direction floors

| model        |      B |     BR |     BQ |      A |     BP |
|:-------------|-------:|-------:|-------:|-------:|-------:|
| ALWAYS_HEAVY | 0.5586 | 0.5586 | 0.5586 | 0.5586 | 0.5586 |
| ALWAYS_LIGHT | 0.4414 | 0.4414 | 0.4414 | 0.4414 | 0.4414 |
| TOPO39       | 0.84   | 0.8346 | 0.84   | 0.8346 | 0.8205 |


### Out-of-fold prediction of log|a|: embedding vs the cheap alternatives

| block               | design   |   n_cells |   spearman |   partial_spearman_given_n_metals |   loco_spearman_min |   loco_spearman_max |   sd_pred |   sd_true |
|:--------------------|:---------|----------:|-----------:|----------------------------------:|--------------------:|--------------------:|----------:|----------:|
| size4               | B        |       289 |     0.1264 |                            0.096  |              0.0388 |              0.5113 |    0.3503 |     0.925 |
| size4               | BP       |       289 |     0.1107 |                           -0.0021 |              0.0209 |              0.4455 |    0.2558 |     0.925 |
| TOPO39              | B        |       289 |    -0.0448 |                           -0.1195 |             -0.1254 |              0.0505 |    0.2225 |     0.925 |
| TOPO39              | BP       |       289 |     0.0871 |                            0.0305 |             -0.1021 |              0.1912 |    0.2392 |     0.925 |
| LEAN209             | B        |       289 |     0.0164 |                            0.0136 |             -0.0643 |              0.1411 |    0.5717 |     0.925 |
| LEAN209             | BP       |       289 |    -0.1191 |                           -0.1683 |             -0.1772 |              0.1354 |    0.3743 |     0.925 |
| molformer__mean     | B        |       289 |     0.3732 |                            0.3993 |              0.3011 |              0.5616 |    0.3793 |     0.925 |
| molformer__mean     | BP       |       289 |     0.2277 |                            0.2539 |              0.1378 |              0.4366 |    0.3306 |     0.925 |
| chemberta_mtr__mean | B        |       289 |     0.1062 |                            0.0546 |              0.0164 |              0.4171 |    0.4981 |     0.925 |
| chemberta_mtr__mean | BP       |       289 |    -0.0334 |                           -0.1239 |             -0.1356 |              0.0149 |    0.3458 |     0.925 |
| morgan2             | B        |       289 |     0.128  |                            0.056  |              0.0539 |              0.2349 |    0.28   |     0.925 |
| morgan2             | BP       |       289 |    -0.0594 |                           -0.1283 |             -0.1776 |              0.0349 |    0.224  |     0.925 |


### Out-of-fold prediction of log|a| and of the curvature b

| block               | target    | design   | estimator   |   spearman |   partial_spearman_given_n_metals |   loco_spearman_min |   loco_spearman_max |
|:--------------------|:----------|:---------|:------------|-----------:|----------------------------------:|--------------------:|--------------------:|
| chemberta_mtr__mean | log_abs_a | B        | ridge       |     0.1062 |                            0.0546 |              0.0164 |              0.4171 |
| chemberta_mtr__mean | log_abs_a | BP       | ridge       |    -0.0334 |                           -0.1239 |             -0.1356 |              0.0149 |
| chemberta_mtr__mean | log_abs_a | B        | cosine      |     0.0922 |                            0.1045 |              0.0092 |              0.3455 |
| chemberta_mtr__mean | log_abs_a | BP       | cosine      |    -0.1045 |                           -0.0978 |             -0.2129 |              0.1217 |
| chemberta_mtr__mean | b         | B        | ridge       |     0.0631 |                            0.0627 |              0.0146 |              0.2158 |
| chemberta_mtr__mean | b         | BP       | ridge       |     0.0127 |                            0.013  |             -0.0262 |              0.0615 |
| chemberta_mtr__mean | b         | B        | cosine      |     0.0275 |                            0.0378 |             -0.0269 |              0.1373 |
| chemberta_mtr__mean | b         | BP       | cosine      |    -0.02   |                           -0.0138 |             -0.0779 |              0.0467 |
| molformer__mean     | log_abs_a | B        | ridge       |     0.3732 |                            0.3993 |              0.3011 |              0.5616 |
| molformer__mean     | log_abs_a | BP       | ridge       |     0.2277 |                            0.2539 |              0.1378 |              0.4366 |
| molformer__mean     | log_abs_a | B        | cosine      |     0.2436 |                            0.2314 |              0.1779 |              0.3472 |
| molformer__mean     | log_abs_a | BP       | cosine      |    -0.136  |                           -0.1979 |             -0.2456 |              0.0542 |
| molformer__mean     | b         | B        | ridge       |     0.1326 |                            0.1364 |              0.0895 |              0.432  |
| molformer__mean     | b         | BP       | ridge       |     0.0649 |                            0.0699 |              0.0185 |              0.2557 |
| molformer__mean     | b         | B        | cosine      |     0.0158 |                            0.0263 |             -0.0402 |              0.1713 |
| molformer__mean     | b         | BP       | cosine      |    -0.0135 |                           -0.0058 |             -0.0708 |              0.052  |
| morgan2             | log_abs_a | B        | ridge       |     0.128  |                            0.056  |              0.0539 |              0.2349 |
| morgan2             | log_abs_a | BP       | ridge       |    -0.0594 |                           -0.1283 |             -0.1776 |              0.0349 |
| morgan2             | log_abs_a | B        | cosine      |    -0.0045 |                           -0.0939 |             -0.0938 |              0.0515 |
| morgan2             | log_abs_a | BP       | cosine      |    -0.178  |                           -0.2719 |             -0.2895 |             -0.1314 |
| morgan2             | b         | B        | ridge       |     0.1567 |                            0.1596 |              0.1127 |              0.4177 |
| morgan2             | b         | BP       | ridge       |     0.0653 |                            0.0675 |              0.0279 |              0.2633 |
| morgan2             | b         | B        | cosine      |     0.0398 |                            0.0465 |             -0.013  |              0.173  |
| morgan2             | b         | BP       | cosine      |    -0.0297 |                           -0.0237 |             -0.0874 |              0.0273 |


### MoLFormer log|a| under BP: estimator sweep

| estimator   |   spearman |   partial_spearman_given_n_metals |   loco_spearman_min |   loco_spearman_max |   sd_pred |
|:------------|-----------:|----------------------------------:|--------------------:|--------------------:|----------:|
| ridge@10    |     0.2482 |                            0.2554 |              0.1603 |              0.5337 |    0.4509 |
| ridge@100   |     0.2277 |                            0.2539 |              0.1378 |              0.4366 |    0.3306 |
| ridge@1000  |     0.1721 |                            0.2298 |              0.0681 |              0.3399 |    0.261  |
| ridge@10000 |     0.2013 |                            0.2418 |              0.1037 |              0.4522 |    0.1927 |
| cosine@0.1  |     0.1793 |                            0.1753 |              0.085  |              0.4879 |    0.3631 |
| cosine@1    |     0.1653 |                            0.173  |              0.0629 |              0.4426 |    0.2808 |
| cosine@10   |     0.1698 |                            0.2015 |              0.078  |              0.374  |    0.1728 |
| rbf@0.1     |     0.1719 |                            0.1445 |              0.0973 |              0.3402 |    0.1987 |
| rbf@1       |     0.1303 |                            0.1602 |              0.0429 |              0.3059 |    0.1418 |
| rbf@10      |    -0.0906 |                           -0.0856 |             -0.2019 |              0.2279 |    0.1057 |


### Permutation null (ligand -> vector map shuffled, folds unchanged)

| design   | block           |   observed_spearman |   null_mean |   null_sd |   null_q95 |   null_max |   n_permutations |   p_one_sided |
|:---------|:----------------|--------------------:|------------:|----------:|-----------:|-----------:|-----------------:|--------------:|
| BP       | molformer__mean |              0.2277 |     -0.0184 |    0.1815 |     0.2525 |     0.3482 |               40 |        0.0976 |


### Near-duplicate ligands

| block               | cosine   |   threshold |   n_ligands |   n_with_twin |   n_with_cross_chemotype_twin |   n_pairs |   twin_direction_agreement |   nn_cosine_median |
|:--------------------|:---------|------------:|------------:|--------------:|------------------------------:|----------:|---------------------------:|-------------------:|
| chemberta_mtr__mean | raw      |       0.999 |          90 |             7 |                             0 |         5 |                     1      |             0.9834 |
| chemberta_mtr__mean | raw      |       0.99  |          90 |            28 |                             8 |        23 |                     0.6522 |             0.9834 |
| chemberta_mtr__mean | raw      |       0.95  |          90 |            68 |                            62 |       417 |                     0.6739 |             0.9834 |
| chemberta_mtr__cls  | raw      |       0.999 |          90 |             7 |                             0 |         5 |                     1      |             0.9576 |
| chemberta_mtr__cls  | raw      |       0.99  |          90 |            12 |                             0 |        10 |                     1      |             0.9576 |
| chemberta_mtr__cls  | raw      |       0.95  |          90 |            47 |                            18 |        79 |                     0.7342 |             0.9576 |
| chemberta_mlm__mean | raw      |       0.999 |          90 |             3 |                             0 |         3 |                     1      |             0.9703 |
| chemberta_mlm__mean | raw      |       0.99  |          90 |            12 |                             0 |        20 |                     1      |             0.9703 |
| chemberta_mlm__mean | raw      |       0.95  |          90 |            61 |                            49 |       213 |                     0.7136 |             0.9703 |
| chemberta_mlm__cls  | raw      |       0.999 |          90 |             3 |                             0 |         3 |                     1      |             0.9675 |
| chemberta_mlm__cls  | raw      |       0.99  |          90 |            21 |                             6 |        18 |                     0.8889 |             0.9675 |
| chemberta_mlm__cls  | raw      |       0.95  |          90 |            56 |                            51 |       304 |                     0.5724 |             0.9675 |
| molformer__mean     | raw      |       0.999 |          90 |             0 |                             0 |         0 |                   nan      |             0.9348 |
| molformer__mean     | raw      |       0.99  |          90 |             5 |                             0 |         4 |                     0.75   |             0.9348 |
| molformer__mean     | raw      |       0.95  |          90 |            38 |                            22 |        59 |                     0.678  |             0.9348 |
| molformer__cls      | raw      |       0.999 |          90 |             0 |                             0 |         0 |                   nan      |             0.9336 |
| molformer__cls      | raw      |       0.99  |          90 |             3 |                             0 |         2 |                     1      |             0.9336 |
| molformer__cls      | raw      |       0.95  |          90 |            30 |                            14 |        36 |                     0.6944 |             0.9336 |
| morgan2             | raw      |       0.999 |          90 |             7 |                             0 |         5 |                     1      |             0.9517 |
| morgan2             | raw      |       0.99  |          90 |            20 |                            11 |        42 |                     0.7857 |             0.9517 |
| morgan2             | raw      |       0.95  |          90 |            47 |                            38 |       335 |                     0.5731 |             0.9517 |
| morgan3             | raw      |       0.999 |          90 |             5 |                             0 |         4 |                     1      |             0.9292 |
| morgan3             | raw      |       0.99  |          90 |            15 |                             2 |        17 |                     0.7647 |             0.9292 |
| morgan3             | raw      |       0.95  |          90 |            33 |                            27 |       186 |                     0.6344 |             0.9292 |


### Nearest neighbour shares chemotype

| block               |   nn_same_chemotype |   n_chemotypes |
|:--------------------|--------------------:|---------------:|
| chemberta_mtr__mean |              0.4667 |             45 |
| chemberta_mtr__cls  |              0.5444 |             45 |
| chemberta_mlm__mean |              0.4667 |             45 |
| chemberta_mlm__cls  |              0.3444 |             45 |
| molformer__mean     |              0.4889 |             45 |
| molformer__cls      |              0.4778 |             45 |
| morgan2             |              0.4889 |             45 |
| morgan3             |              0.4889 |             45 |


### n_metals / embedding-norm confound

| block               | quantity       |   spearman_with_abs_a |
|:--------------------|:---------------|----------------------:|
| -                   | n_metals       |                0.4357 |
| chemberta_mtr__mean | embedding_norm |               -0.015  |
| chemberta_mtr__cls  | embedding_norm |               -0.0559 |
| chemberta_mlm__mean | embedding_norm |               -0.1573 |
| chemberta_mlm__cls  | embedding_norm |               -0.0761 |
| molformer__mean     | embedding_norm |               -0.1787 |
| molformer__cls      | embedding_norm |                0.0289 |
| morgan2             | embedding_norm |               -0.1285 |
| morgan3             | embedding_norm |               -0.1317 |


### Leave-one-chemotype-out summary

| design   |   n_chemotypes |   topo39_macro |   embed_macro |   embed_wins |   ties |   embed_loses |   worst_case_topo39_macro |   best_case_embed_macro | embed_ever_beats_topo39_after_dropping_one   |
|:---------|---------------:|---------------:|--------------:|-------------:|-------:|--------------:|--------------------------:|------------------------:|:---------------------------------------------|
| B        |             45 |         0.7987 |        0.6417 |            5 |     24 |            16 |                    0.7941 |                  0.6563 | False                                        |
| BP       |             45 |         0.7785 |        0.5381 |            7 |     12 |            26 |                    0.7735 |                  0.5504 | False                                        |

