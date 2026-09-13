# L3 -- two narrow decision questions: measurements saved, and ranking at k = 1

*Lead L3 of the gen16 fleet, `PRE_REGISTRATION.md` section 3 L3 (SHA-256
`d004c30388078ae97232af2537a1915360360a7cb9eda07ebe4076764c48f50e`).  L3b is **not run**: it is
registered as conditional on L1 stage 1 being positive and belongs to that lead's verdict.*

**Regime for every number below.**  Frozen gen13 cohort (fingerprint `4c3c6628ea0be949`, 521
cells, 90 extractants, 45 chemotypes), the five discovery seeds `(104729, 130363, 155921,
196613, 262147)` x 5 folds, all five hold-out designs (B, BR, BQ, A, BP), held-out predictions
from `gen15.valuebench.run_arms` with the deployed `gen15.arms.g14`.  `y = log SF = logD(A) -
logD(B)` with A the lighter lanthanide, base 10, so `y < 0` means the heavier metal is preferred.
Units are **measurements** (L3a) and **log10 units of regret** / rank correlation (L3c); the
aggregation unit is the (split seed, unordered metal pair) **task**, averaged within split seed
and then over the five seeds -- the frozen Q3 convention of
`gen15_curve/exp/decision/decmetrics.py`.  BP selects; design B never selects.


## 0. Summary

| question | registered endpoint | five designs (B / BR / BQ / A / BP) | rule met | verdict |
|---|---|---|---|---|
| **L3a** measurements saved by the direction call | `saved = E_random - E_model`, measurements | +2.207 / +2.149 / +2.179 / +2.492 / +1.988 | permutation p = 0.0000 and CI excluding 0 in 5/5 | **POSITIVE** |
| **L3c** ranking at k = 1 | `G14@k1 - NAIVE_LINE@k1` on regret (log10) | +0.0032 / +0.0018 / -0.0029 / -0.0168 / -0.0210 | sign not consistent, p 0.34-1.00 | **NULL** |
| L3c, second registered endpoint | the same on Spearman | -0.0229 / -0.0217 / -0.0230 / -0.0217 / -0.0268 | consistently **negative**, p 0.013-0.026 | **fails (reference arm wins)** |
| L3b ranking within a chemotype | -- | -- | conditional on L1 | **not run** |

**L3a, design BP (the design that selects):** the direction call cuts the expected number of measurements to the first useful candidate from 6.91 to 4.92, a saving of **1.99 measurements (28.8 % of the no-model cost)**, 95 % CI [+0.44, +3.78], permutation p = 0.0000, 5/5 seeds, LOCO-stable.  The cheapest competitor named in the pre-registration, the "always heavier" rule used the same way, saves **exactly 0.000000** in all five designs, and does so by construction.

**L3c, the same design:** one measurement per candidate is worth 1.09 log units of regret against the zero-shot model, but the corpus adds nothing to that measurement -- the registered contrast is a null on regret and a significant negative on rank correlation.  See section 3.

## 1. Tasks, and what was dropped

A **task** is one (split seed, unordered metal pair).  Its candidates are the held-out extractants of that seed that measured the pair; an extractant's several cells (condition sets) are collapsed by the **median** observed `log SF` and the median predicted value.  Tasks with fewer than 5 candidates are dropped and counted.

| design   |   l3a_n_tasks |   l3a_n_dropped_lt5 |   l3a_median_candidates |   l3a_n_candidate_slots |   l3c_n_tasks |   l3c_n_dropped_lt5 |   l3c_n_dropped_ptp0 |   l3c_median_candidates |   l3c_n_candidates_dropped_only_target |   n_cell_seeds_single_pair |   n_extractants |   n_chemotypes |
|:---------|--------------:|--------------------:|------------------------:|------------------------:|--------------:|--------------------:|---------------------:|------------------------:|---------------------------------------:|---------------------------:|----------------:|---------------:|
| B        |           455 |                   0 |                      58 |                   27365 |           455 |                   0 |                    0 |                      57 |                                     30 |                        690 |              90 |             45 |
| BR       |           455 |                   0 |                      58 |                   27365 |           455 |                   0 |                    0 |                      57 |                                     30 |                        690 |              90 |             45 |
| BQ       |           455 |                   0 |                      58 |                   27365 |           455 |                   0 |                    0 |                      57 |                                     30 |                        690 |              90 |             45 |
| A        |           455 |                   0 |                      58 |                   27365 |           455 |                   0 |                    0 |                      57 |                                     30 |                        690 |              90 |             45 |
| BP       |           455 |                   0 |                      58 |                   27365 |           455 |                   0 |                    0 |                      57 |                                     30 |                        690 |              90 |             45 |

`l3c_n_dropped_ptp0` are tasks whose candidates all observed the *same* `log SF`, which `decmetrics.cross_extractant` drops because there is nothing to rank; L3a keeps them because an expected number of draws is still defined.  `l3c_n_candidates_dropped_only_target` counts candidate slots lost because the extractant's only measured pair *is* the target pair, so it has nothing to measure at k = 1; `n_cell_seeds_single_pair` is the same drop counted per (seed, cell).

The G14 direction call is identical for every candidate in 0.0 % of the 455 tasks under BP (gen15 decision Q3 measured 0 % pooled across laboratories and 70-85 % within one laboratory).

## 2. L3a -- measurements saved by the direction call

A candidate **succeeds** for a requested direction if its observed `log SF` has that sign and `|log SF| >= 0.3` (gen13's strong-pair threshold).  Without the model the chemist measures in uniform random order: expected draws to the first success `E_random = (N + 1) / (K + 1)`.  With the model, candidates whose G14 sign call disagrees with the request are deferred (a prediction of exactly 0 is *no call* and is kept): `E_model = (N' + 1) / (K' + 1)` over the kept set, falling through to the deferred set when the kept set holds no success.  **saved = E_random - E_model**, averaged over the two requested directions and then over tasks.

**Registered decision rule (L3a).**  *Positive* iff `saved > 0` with permutation
p < 0.05 and the chemotype-blocked CI excluding zero **in all five designs**.  Permutation:
the model's sign calls permuted across the candidates of each task, 2000 replicates, seed
8675309.  Interval: chemotype-blocked bootstrap over extractants (chemotypes resampled with
replacement, every task rebuilt from the resampled extractant multiset), 2000 replicates,
seed 8675309, 2.5 / 97.5 percentiles.  Cheapest competitor: the "always heavier" rule used the
same way.

**Table 2a -- G14, all tasks, five designs** (measurements)

| design   |   n_tasks |   E_random |   E_model |   saved |   saved_frac |   ci95_low |   ci95_high |   p_perm |   sd_seed |
|:---------|----------:|-----------:|----------:|--------:|-------------:|-----------:|------------:|---------:|----------:|
| B        |       455 |     6.9056 |    4.6991 |  2.2065 |       0.3195 |     0.5207 |      4.1177 |        0 |    0.2904 |
| BR       |       455 |     6.9056 |    4.7564 |  2.1493 |       0.3112 |     0.455  |      3.9849 |        0 |    0.3013 |
| BQ       |       455 |     6.9056 |    4.7266 |  2.179  |       0.3155 |     0.4683 |      4.0696 |        0 |    0.2542 |
| A        |       455 |     6.9056 |    4.4141 |  2.4915 |       0.3608 |     0.7709 |      4.5442 |        0 |    0.0695 |
| BP       |       455 |     6.9056 |    4.9174 |  1.9882 |       0.2879 |     0.4425 |      3.7808 |        0 |    0.5004 |

`p_perm = 0.0000` is reported as computed -- the fraction of 2000 permuted `saved` values at or above the observed one, which is 0 in every design.  The resolution of that null is 1 / 2000 = 0.0005; it is not rounded up to look conservative and not restated as `< 1e-4`.  `saved_frac` is the ratio of the two seed-macro aggregates; the seed-macro of the per-task ratio (`saved_frac_task_mean` in `l3a_saved.csv`) is 0.389-0.433 and both are given.

Seeds positive / LOCO stability, registered rows: B 5/5 seeds, LOCO [+1.4906, +2.7286]; BR 5/5 seeds, LOCO [+1.5084, +2.6848]; BQ 5/5 seeds, LOCO [+1.4793, +2.6999]; A 5/5 seeds, LOCO [+1.7036, +3.0549]; BP 5/5 seeds, LOCO [+1.4591, +2.5406]

**Verdict L3a: POSITIVE** against the registered rule (saved > 0, permutation p < 0.05 and chemotype-blocked CI excluding 0 in all five designs): the rule is met in 5 of 5 designs.

**Table 2b -- the cheapest competitor: `HEAVIER_ALWAYS` used the same way** (exploratory)

| design   |   E_random |   E_model |   saved |   saved_heavy |   saved_light |
|:---------|-----------:|----------:|--------:|--------------:|--------------:|
| B        |     6.9056 |    6.9056 |       0 |             0 |             0 |
| BR       |     6.9056 |    6.9056 |       0 |             0 |             0 |
| BQ       |     6.9056 |    6.9056 |       0 |             0 |             0 |
| A        |     6.9056 |    6.9056 |       0 |             0 |             0 |
| BP       |     6.9056 |    6.9056 |       0 |             0 |             0 |

**`HEAVIER_ALWAYS` saves exactly nothing, and it does so by construction.**  Calling every candidate heavy keeps the whole set for a heavy request (`E_model = E_random`) and defers the whole set for a light one, which falls straight back to random order over the deferred set (`E_model = E_random` again).  A rule that is constant across the candidate set cannot partition it, so the entire saving in Table 2a is the *variation* of the direction call between candidates, not its accuracy.  That is the honest comparison the protocol asks for and it is the reason L3a is not a restatement of gen14's 0.812-against-0.630 sign accuracy.

**Table 2c -- split by requested direction and by dZ band** (exploratory)

| design   |   saved_heavy_request |   saved_light_request |   E_random_heavy |   E_random_light |
|:---------|----------------------:|----------------------:|-----------------:|-----------------:|
| B        |                0.7067 |                3.7064 |           5.0492 |            8.762 |
| BR       |                0.6839 |                3.6147 |           5.0492 |            8.762 |
| BQ       |                0.6678 |                3.6902 |           5.0492 |            8.762 |
| A        |                0.7008 |                4.2822 |           5.0492 |            8.762 |
| BP       |                0.3267 |                3.6497 |           5.0492 |            8.762 |

| split   |      B |     BR |     BQ |      A |     BP |
|:--------|-------:|-------:|-------:|-------:|-------:|
| dZ 2-4  | 2.2329 | 2.1448 | 2.2459 | 2.5533 | 1.8417 |
| dZ 5-8  | 1.888  | 1.8595 | 1.874  | 1.984  | 1.9304 |
| dZ 9-13 | 1.731  | 1.7106 | 1.7253 | 1.7315 | 1.7636 |
| dZ=1    | 3.6452 | 3.5361 | 3.4458 | 4.7645 | 2.8095 |

(`saved`, measurements, by dZ band; n tasks per band: dZ=1 60, dZ 2-4 150, dZ 5-8 150, dZ 9-13 90)

the same split as a **fraction** of `E_random`:

| split   |      B |     BR |     BQ |      A |     BP |
|:--------|-------:|-------:|-------:|-------:|-------:|
| dZ 2-4  | 0.2902 | 0.2788 | 0.2919 | 0.3319 | 0.2394 |
| dZ 5-8  | 0.4666 | 0.4596 | 0.4632 | 0.4904 | 0.4771 |
| dZ 9-13 | 0.5105 | 0.5045 | 0.5088 | 0.5106 | 0.5201 |
| dZ=1    | 0.207  | 0.2008 | 0.1957 | 0.2706 | 0.1595 |

**What the saving is made of.**  The saving is entirely the *variation* of the direction
call across the candidate set, not its accuracy: `HEAVIER_ALWAYS` is 81 %-vs-63 % worse as a
classifier and saves exactly the same amount as a coin that always says the same thing, namely
nothing.  Two structural facts set the size.  (i) `E_random` is much larger for a
**light-preferred** request (8.76 draws) than for a heavy-preferred one (5.05), because
light-selective systems with `|log SF| >= 0.3` are rarer in the cohort, so there is more to save
there and the model saves most of it.  (ii) As a *fraction* of `E_random` the saving grows
monotonically with how far apart the two metals are -- 0.16-0.27 for adjacent pairs, 0.50-0.52
for `dZ` 9-13 -- which is the same gradient gen14 measured in the direction call itself.  In
absolute measurements the ordering reverses (adjacent pairs start from `E_random` 17.6, so even a
20 % cut is 2.8-4.8 measurements), and both facts are reported because either one alone is
misleading.

## 3. L3c -- ranking candidates that each carry one measurement

Every candidate extractant receives **one** measured pair: its widest-dZ measured pair *excluding the target pair* (`gen15.fewshot.pick_support(..., 'widest')` on the remaining pairs, per cell; the extractant's prediction is then the median over its cells).  Candidates whose only measured pair is the target are dropped and counted.  Predictions for the target pair: **`G14@k1`** = the deployed measured-mode route (`fewshot.blup` with the G14 prior curve and the leave-chemotype-out, publication-masked residual covariance, shrink 0.25, `NOISE_VAR` 0.09), **`NAIVE_LINE@k1`** = a straight radius ramp through the one support, **`G14`** = the zero-shot curve, **`_RANDOM`** = the tie-expected pick.  Metrics are Q3's: cross-extractant Spearman, tie-expected top-1, and regret (observed `log SF` of the truly best candidate minus that of the pick), averaged over the two directions.

**Registered decision rule (L3c).**  *Positive* iff the regret gain
`G14@k1 - NAIVE_LINE@k1` has p < 0.05 with a consistent sign **in all five designs**.  The p
quoted for the rule is the most conservative of the task bootstrap, the chemotype-blocked
bootstrap over extractants and the permutation null (the permutation permutes the `G14@k1`
predictions across the candidates of each task); the interval quoted is the wider / less
significant of the two bootstraps.  All three use 2000 replicates and seed 8675309.

**Table 3a -- regret, log10 units (lower is better)**

| arm           |      B |     BR |     BQ |      A |     BP |
|:--------------|-------:|-------:|-------:|-------:|-------:|
| G14           | 1.5288 | 1.4429 | 1.533  | 1.4681 | 1.4541 |
| G14@k1        | 0.3363 | 0.3377 | 0.3424 | 0.3563 | 0.3605 |
| NAIVE_LINE@k1 | 0.3395 | 0.3395 | 0.3395 | 0.3395 | 0.3395 |
| _RANDOM       | 1.762  | 1.762  | 1.762  | 1.762  | 1.762  |

**cross-extractant Spearman (higher is better)**

| arm           |      B |     BR |     BQ |      A |     BP |
|:--------------|-------:|-------:|-------:|-------:|-------:|
| G14           | 0.3489 | 0.4014 | 0.3614 | 0.3816 | 0.3523 |
| G14@k1        | 0.7762 | 0.7775 | 0.7761 | 0.7774 | 0.7723 |
| NAIVE_LINE@k1 | 0.7991 | 0.7991 | 0.7991 | 0.7991 | 0.7991 |
| _RANDOM       | 0      | 0      | 0      | 0      | 0      |

**tie-expected top-1 (higher is better)**

| arm           |      B |     BR |     BQ |      A |     BP |
|:--------------|-------:|-------:|-------:|-------:|-------:|
| G14           | 0.008  | 0.0186 | 0.0062 | 0.0183 | 0.0223 |
| G14@k1        | 0.3429 | 0.3407 | 0.3352 | 0.311  | 0.3066 |
| NAIVE_LINE@k1 | 0.3297 | 0.3297 | 0.3297 | 0.3297 | 0.3297 |
| _RANDOM       | 0.0171 | 0.0171 | 0.0171 | 0.0171 | 0.0171 |

**Table 3b -- the registered contrast, `G14@k1 - NAIVE_LINE@k1`** (positive = the corpus beats the line)

*regret*

| design   |   point |   ci95_low |   ci95_high | interval          |   p_task |   p_block |   p_perm |   p_registered |   seeds_pos | loco_stable   | passes   |
|:---------|--------:|-----------:|------------:|:------------------|---------:|----------:|---------:|---------------:|------------:|:--------------|:---------|
| B        |  0.0032 |    -0.0252 |      0.022  | chemotype-blocked |    0.732 |     0.927 |        0 |          0.927 |           2 | False         | False    |
| BR       |  0.0018 |    -0.0248 |      0.0266 | chemotype-blocked |    0.838 |     0.997 |        0 |          0.997 |           3 | False         | False    |
| BQ       | -0.0029 |    -0.0253 |      0.0197 | chemotype-blocked |    0.708 |     0.744 |        0 |          0.744 |           2 | False         | False    |
| A        | -0.0168 |    -0.0329 |      0.0132 | chemotype-blocked |    0.064 |     0.335 |        0 |          0.335 |           2 | True          | False    |
| BP       | -0.021  |    -0.0418 |      0.0206 | chemotype-blocked |    0.03  |     0.461 |        0 |          0.461 |           1 | True          | False    |

*spearman*

| design   |   point |   ci95_low |   ci95_high | interval          |   p_task |   p_block |   p_perm |   p_registered |   seeds_pos | loco_stable   | passes   |
|:---------|--------:|-----------:|------------:|:------------------|---------:|----------:|---------:|---------------:|------------:|:--------------|:---------|
| B        | -0.0229 |    -0.0488 |     -0.0047 | chemotype-blocked |        0 |     0.015 |        0 |          0.015 |           0 | True          | False    |
| BR       | -0.0217 |    -0.0483 |     -0.0034 | chemotype-blocked |        0 |     0.026 |        0 |          0.026 |           0 | True          | False    |
| BQ       | -0.023  |    -0.0517 |     -0.0052 | chemotype-blocked |        0 |     0.013 |        0 |          0.013 |           0 | True          | False    |
| A        | -0.0217 |    -0.0455 |     -0.0034 | chemotype-blocked |        0 |     0.023 |        0 |          0.023 |           0 | True          | False    |
| BP       | -0.0268 |    -0.0505 |     -0.0065 | chemotype-blocked |        0 |     0.013 |        0 |          0.013 |           0 | True          | False    |

`p_perm` is 0 in every row above and carries no information about *this* contrast.  Permuting `G14@k1` across the candidates of a task destroys the measurement each candidate carries, which is worth 1.1-1.4 log units of regret; the difference being tested against `NAIVE_LINE@k1` is worth 0.02.  The permutation null therefore rejects trivially and the conservative p that the registered rule uses is the bootstrap one, which is what the `p_registered` column shows.  This was the pre-specified most-conservative-of-three rule, not a choice made after the numbers were seen.

**Verdict L3c: NOT POSITIVE** against the registered rule (regret gain p < 0.05 with a consistent sign in all five designs): the sign is NOT consistent across the five designs (2 of 5 positive, range -0.0210 to +0.0032) and the p rule is met in 0 of 5 designs (p 0.335-0.997).

The second registered endpoint, rank correlation, is negative in all five designs: -0.0268 to -0.0217, p 0.013-0.026, LOCO-stable True.  It **also fails the registered rule**, which asks for a positive gain; it is a significant result in the *reference* arm's favour and is reported as such, not converted into a claim by flipping the endpoint.

**Table 3c -- the exploratory comparisons** (written out, never promoted)

| comparison                  |      B |     BR |     BQ |       A |      BP |
|:----------------------------|-------:|-------:|-------:|--------:|--------:|
| G14k1_minus_G14zs_regret    | 1.1925 | 1.1053 | 1.1906 |  1.1118 |  1.0936 |
| G14k1_minus_G14zs_spearman  | 0.4273 | 0.3761 | 0.4147 |  0.3958 |  0.42   |
| G14k1_minus_NAIVE_top1      | 0.0132 | 0.011  | 0.0055 | -0.0187 | -0.0231 |
| G14k1_minus_RANDOM_regret   | 1.4257 | 1.4244 | 1.4196 |  1.4058 |  1.4015 |
| G14k1_minus_RANDOM_spearman | 0.7762 | 0.7775 | 0.7761 |  0.7774 |  0.7723 |
| NAIVE_minus_RANDOM_regret   | 1.4225 | 1.4225 | 1.4225 |  1.4225 |  1.4225 |
| NAIVE_minus_RANDOM_spearman | 0.7991 | 0.7991 | 0.7991 |  0.7991 |  0.7991 |

**Mechanism, and what the null is not.**  The registered contrast is a null on
regret and a *significant negative* on rank correlation: the straight line through the one
measurement orders the candidates better than the BLUP that adds the corpus to it, by 0.022-0.027
Spearman with p 0.013-0.026 and a LOCO-stable sign in all five designs.  The reason is visible in
the construction.  Ranking is a *between-candidate* comparison, and at k = 1 the two arms differ
only in what they add to the measurement: `NAIVE_LINE@k1` scales the measured `log SF` along the
radius axis and nothing else, so every log unit of its prediction is candidate-specific; the BLUP
mixes in the G14 prior curve, whose magnitude is **one number per training fold** and whose only
between-candidate content is the direction bit.  Shrinking a candidate-specific measurement
toward a nearly candidate-constant prior can only compress the spread the ranking depends on.
This is the same quantity gen15 section 5 measured on MAE (`G14 - NAIVE_LINE @ k = 1` = +0.011 to
+0.018, not significant); the sign flips when the endpoint changes from accuracy to ordering, and
both are small.

**The result that is not the registered one, and is exploratory.**  One measurement per candidate
is worth an enormous amount for this decision, and the corpus is not what delivers it.  Under BP,
regret falls from 1.454 (zero-shot `G14`) to 0.361 (`G14@k1`) against 1.762 for a random pick;
rank correlation rises from 0.352 to 0.772; and the tie-expected top-1 rises from 0.022 -- gen15's
"at chance" number, chance being 0.017 -- to 0.307.  `NAIVE_LINE@k1` reaches 0.340 / 0.799 /
0.330 on the same tasks with no corpus at all.  Gen15 section 8's verdict that the model "cannot
rank candidate ligands at all" is a statement about the **zero-shot** regime; the ranking problem
is solved by one measurement per candidate, and solved about equally well without the model.
Every number in this paragraph is `family=exploratory` and none of it is offered as a claim.

## 4. Loop-validity checks

| design   |   G14 curve route vs run_arms (max |d|) |   vs gen15 decision cached pairs (max |dG14|) |   weighted stats vs decmetrics (max |d|) |   matrix vs direct L3a (max |d|) |   covariance smooth fallbacks |
|:---------|----------------------------------------:|----------------------------------------------:|-----------------------------------------:|---------------------------------:|------------------------------:|
| B        |                                       0 |                                             0 |                              4.44089e-16 |                                0 |                             0 |
| BR       |                                       0 |                                             0 |                              4.44089e-16 |                                0 |                             0 |
| BQ       |                                       0 |                                             0 |                              4.44089e-16 |                                0 |                             0 |
| A        |                                       0 |                                             0 |                              8.88178e-16 |                                0 |                             0 |
| BP       |                                       0 |                                             0 |                              4.44089e-16 |                                0 |                             0 |

`gen15.fewshot.evaluate(ks=(0,1), how='widest', mask_publication=True)` under B: 67525 rows matched, max |dG14@k1| = 0.00e+00, max |dNAIVE_LINE@k1| = 0.00e+00, max |dG14 zero-shot| = 0.00e+00; every matched row's support is the cell's widest pair: True.

`gen15.fewshot.evaluate(ks=(0,1), how='widest', mask_publication=True)` under BR: 67525 rows matched, max |dG14@k1| = 0.00e+00, max |dNAIVE_LINE@k1| = 0.00e+00, max |dG14 zero-shot| = 0.00e+00; every matched row's support is the cell's widest pair: True.

`gen15.fewshot.evaluate(ks=(0,1), how='widest', mask_publication=True)` under BQ: 67525 rows matched, max |dG14@k1| = 0.00e+00, max |dNAIVE_LINE@k1| = 0.00e+00, max |dG14 zero-shot| = 0.00e+00; every matched row's support is the cell's widest pair: True.

`gen15.fewshot.evaluate(ks=(0,1), how='widest', mask_publication=True)` under A: 67525 rows matched, max |dG14@k1| = 0.00e+00, max |dNAIVE_LINE@k1| = 0.00e+00, max |dG14 zero-shot| = 0.00e+00; every matched row's support is the cell's widest pair: True.

`gen15.fewshot.evaluate(ks=(0,1), how='widest', mask_publication=True)` under BP: 67525 rows matched, max |dG14@k1| = 0.00e+00, max |dNAIVE_LINE@k1| = 0.00e+00, max |dG14 zero-shot| = 0.00e+00; every matched row's support is the cell's widest pair: True.

The frozen anchors reproduce inside this loop: G14 macro MAE under BP 0.5000794414203691 and FLAT 0.5885062528901843 come out of the same `run_arms` call that feeds every task above.

## 5. Limitations a refuter should attack first

1. **The candidate set is pooled across laboratories.**  Gen15 decision Q3 Table 3d showed the
   pooled cross-extractant signal is largely *between-laboratory* information: restricted to
   candidates measured in one publication under one protocol, gen14 ranks no better than random,
   because 70-85 % of those tasks give every candidate the same direction call.  Pooled, that
   figure is 0.0 % here (measured, `tasks_summary.csv`), which is exactly why L3a can save
   anything at all.  **A within-publication L3a is not a registered contrast and was not run.**
   It is the first thing to ask for, and its likely answer is that the saving collapses, because
   a call that is constant across the candidate set defers everyone or no one and saves nothing
   either way -- the `HEAVIER_ALWAYS` row is that limit measured.
2. **`E_random` and `E_model` are a stylised procurement model**, not an observed cost.  They
   assume the chemist measures in uniform random order, stops at the first success, and exhausts
   the kept set before touching the deferred one.  The fall-through is modelled honestly (a wrong
   deferral costs the whole kept set first), but the units are *expected draws under that policy*
   and not laboratory hours.
3. **The 455 tasks are 91 metal pairs x 5 split seeds**, so they are far from independent on the
   pair side.  The registered interval resamples chemotypes over the *candidate* side; the
   pair side is controlled instead by the permutation null, which reshuffles the calls inside each
   task and therefore holds the pair, its candidate set and its observed values completely fixed.
   No second interval was added on the pair side after the fact.
4. **L3a's success threshold (`|log SF| >= 0.3`) and the 5-candidate floor are gen13's and Q3's
   frozen conventions**, fixed before any number was seen.  Under BP the direction split is the
   one place where the five designs disagree materially (`saved_heavy` 0.33 under BP against
   0.67-0.70 elsewhere); the registered endpoint averages the two directions and is stable
   (1.99-2.49), and the disagreement is reported rather than smoothed.

## 6. Temptations resisted

*(Also filed in `REFUTATION_LOG.md`.  None of these was acted on.)*

1. **Reading "the CI" as the more favourable of the two bootstraps in L3c.**  The registered
   text says the conservative interval is quoted.  Every L3c row therefore carries
   `ci_task_low/high`, `ci_block_low/high` and a `conservative_interval` column naming which one
   the headline `ci95_*` came from, so the choice is auditable rather than asserted.
2. **Quoting the L3a permutation p as `(1 + #{perm >= obs}) / (R + 1)` when it came out 0.**
   The registered wording is "the fraction of permuted `saved` >= observed", so `p = 0.0000` is
   reported as it is computed, with the resolution `1/2000 = 0.0005` stated beside it.  It is
   not rounded up to look more conservative and it is not rounded down to look more significant.
3. **Dropping the `< 5 candidates` rule to keep more tasks, or lowering the |log SF| >= 0.3
   success threshold to raise the success rate.**  Both are Q3's frozen conventions
   (`decmetrics.MIN_EXT_RANK`, `decmetrics.STRONG`) and both were fixed before any number was
   seen.  The dropped-task and dropped-candidate counts are reported instead.
4. **Reporting L3a only for the direction where it wins.**  The split by requested direction is
   in `l3a_saved.csv` and in Table 2c, and the heavy-preferred column -- which is four to seven
   times smaller than the light-preferred one, and smaller again under BP -- is reported whatever
   it says.
5. **Promoting an exploratory row.**  The dZ bands, the by-direction split, the top-1 endpoint,
   `G14@k1` against zero-shot `G14` and against random, and `NAIVE_LINE@k1` against random are
   all written with `family=exploratory` and none of them is quoted as a claim.
6. **Presenting `HEAVIER_ALWAYS` saving exactly 0 as a win for the model without saying why it
   is 0.**  It is 0 by construction, not by measurement: a rule that gives every candidate the
   same call cannot partition the candidate set, so it keeps everyone for a heavy request and
   defers everyone (falling straight back to random order) for a light one.  The mechanism is
   stated wherever the number appears.
7. **Running the within-publication version of L3a once the pooled number came out large.**  It
   is not a registered contrast, gen15 decision Q3 Table 3d has already measured the mechanism
   that would drive it, and running it here would be re-scoping a question after seeing its
   answer.  It is written into the limitations as the first thing a refuter should demand, and
   the `HEAVIER_ALWAYS` row is left standing as the measured limit of a constant call.
8. **Adding a task-level bootstrap to L3a after seeing that the chemotype-blocked interval is
   wide (BP [+0.44, +3.78] on a point of +1.99).**  The registered interval for L3a is the
   chemotype-blocked one and it is quoted as registered.  A second interval chosen after the fact
   is a second chance at the same test.
9. **Quoting L3c's rank-correlation result as "the corpus is worse, significantly, in all five
   designs" as though that were a registered finding in the model's disfavour and therefore
   safe to state loosely.**  It is a registered contrast whose sign is negative; it is reported
   with its p and its interval and it fails the registered rule, which asked for a *positive*
   gain.  A significant negative is not a licence to reverse the endpoint and claim the line
   beats the model as a new result.
10. **Reporting only the exploratory k = 1 jump (regret 1.45 -> 0.36) and letting it stand in for
    the lead's outcome.**  It is the largest number in this report and it is not a registered
    contrast; it is labelled exploratory everywhere it appears, and the registered L3c verdict is
    stated as the null it is.

## 7. Files

| file | what it holds |
|---|---|
| `tasks_summary.csv` | task and drop counts per design |
| `l3a_saved.csv` | board: saved / E_random / E_model per design, arm and split |
| `l3a_per_task.csv` | one row per task: saved, the two directions, the call census |
| `l3a_contrasts.csv` | contrasts, `paired_contrasts` layout + `family` + `lead` |
| `l3c_metrics.csv` | board: Spearman, top-1, regret per design and arm |
| `l3c_per_task.csv` | `decmetrics.cross_extractant` per-task output |
| `l3c_contrasts.csv` | contrasts, `paired_contrasts` layout + `family` + `lead` |
| `checks.json` | every loop-validity check, with its residual |

Both contrast files carry the `paired_contrasts` column set plus `family` and `lead`.  `passes_registered` is the rule quoted in the `rule` column and is the one that decides each verdict above; `passes_P1` is gen13's generic P1 evaluated on the same row and is **not** the decision rule for this lead -- P1's 0.02 margin is a margin in extractant-macro MAE and means nothing applied to a count of measurements.  No BCa interval is computed here: the unit is the task, not the extractant, so `gen13sep.inference.paired_contrasts` (which is the only BCa implementation in the programme) does not apply, and its resampling construction and seed are reproduced instead.

Contrast rows written: 40 (L3a) + 45 (L3c) = 85; registered 15, exploratory 70.
