# Is the curve model useful for the decision a chemist makes?

*gen15_curve/exp/decision — every number below is computed by the scripts in this directory from the frozen gen13 cohort and written to `results/`; nothing is estimated.*

## 0. What was measured, and the conventions

Five arms, five designs, 5 split seeds x 5 folds each, 70 750 held-out (seed, cell, metal-pair) rows per design. `y = logD(A) - logD(B)` with A the **lighter** lanthanide, base-10, so `y < 0` means the system prefers the heavier metal.

| arm | what it is |
|---|---|
| `FLAT` | predict no separation at all — the honest floor |
| `MEAN_CURVE` | the training fold's mean curve |
| `HEAVIER_ALWAYS` | the rule 'always prefer the heavier lanthanide', ordered by the radius gap. Scored with gen13's own `heavier_always_sign_accuracy` convention; it is a sign/rank rule, so its MAE and calibration slope are not meaningful |
| `G14` | the deployed model: L2 logistic on 39 donor-topology columns for the direction, times the training fold's mean magnitude, curvature at the training mean |
| `G14_TIED` | **diagnostic, not deployable-different**: the same direction bit with ONE global magnitude and curvature instead of the fold's own. Isolates the cross-fold jitter that leave-one-group-out introduces when predictions from different folds are compared |
| `G13_FULL` | gen13's 209-column extra-trees regression on both coefficients |
| `O_BOTH` | **oracle**: the cell's own two coefficients — the representation ceiling |
| `_RANDOM` | picking uniformly at random from the same candidate set |

**Aggregation.** Mean within a (split seed, extractant), then over extractants, then over the five seeds — gen13's own unit, so the one chemotype that holds 375 of 521 cells cannot carry a number. Spread quoted is the sd over the five seeds.

**Ties.** A constant prediction has no ranking. Every rank statistic here is the *exact expectation under uniform random tie-breaking*, so an uninformative arm lands on chance rather than on NaN, and `FLAT` and `HEAVIER_ALWAYS` reproduce `_RANDOM` exactly wherever they carry no information. A prediction of exactly zero scores 0.5 on a sign call.

**Reference check** — the endpoint the programme is locked to, reproduced by this run (extractant-macro MAE of log SF):

| arm        |      B |     BR |     BQ |      A |     BP |
|:-----------|-------:|-------:|-------:|-------:|-------:|
| FLAT       | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| MEAN_CURVE | 0.5969 | 0.5969 | 0.5968 | 0.5832 | 0.6217 |
| G13_FULL   | 0.4823 | 0.4846 | 0.4803 | 0.4373 | 0.5523 |
| G14        | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| O_BOTH     | 0.1811 | 0.1811 | 0.1811 | 0.1811 | 0.1811 |

G14 = 0.5001 and G13_FULL = 0.5523 under BP reproduce the locked 0.500 / 0.552 to three decimals, so the bench is being driven correctly.

**Verification** (`checks.py`, all passing): the closed-form tie expectations agree with a Monte-Carlo draw of consistent orderings to 0.017; the `HEAVIER_ALWAYS` sign accuracy equals gen13's own `heavier_always_sign_accuracy` to 1e-9 (0.629795); `FLAT` and `HEAVIER_ALWAYS` reproduce `_RANDOM` exactly wherever they carry no ranking information; the `O_BOTH` column equals the cell's own coefficients times the basis difference; and a perfect predictor gets calibration slope exactly 1.0.

---

## Summary — the four decisions, design BP

Every row is the deployed model `G14` against the cheapest sensible alternative already in the repo and against the representation ceiling. `consistent` means the sign of the gain over the alternative is the same under all five hold-out designs.

| decision                                          | metric                                          |   G14 |   cheapest alternative | alternative       | ceiling (O_BOTH)   | consistent                    | verdict                  |
|:--------------------------------------------------|:------------------------------------------------|------:|-----------------------:|:------------------|:-------------------|:------------------------------|:-------------------------|
| Q1 call the direction of a target pair            | sign accuracy, abs(log SF) >= 0.3               | 0.812 |                  0.63  | heavier always    | 0.9721828318299754 | yes                           | clear win                |
| Q2 rank one system's own pairs                    | within-cell Spearman                            | 0.504 |                  0.261 | radius-gap order  | 0.766666955960425  | yes                           | clear win                |
| Q2 name the best-separated pair                   | top-1 hit rate                                  | 0.25  |                  0.412 | widest radius gap | 0.3570685101935102 | yes (always negative)         | LOSS to the rule         |
| Q3 order extractants for a target pair            | cross-extractant Spearman                       | 0.351 |                  0     | random pick       | 0.884080831647973  | yes                           | win, but see Q3d         |
| Q3 pick the single best extractant                | top-1 hit rate                                  | 0.023 |                  0.017 | random pick       | 0.5274725274725275 | no (sign flips)               | null — at chance         |
| Q3 pick the strongest separator, either direction | regret, log10 (lower better)                    | 1.65  |                  1.519 | random pick       | 0.2398335959604067 | yes (always worse)            | LOSS to random           |
| Q3 same, inside ONE laboratory                    | regret, log10 (lower better)                    | 0.643 |                  0.572 | random pick       | 0.0561214625119175 | no (worse in 4 of 5)          | null — worse than random |
| Q4 believe the predicted magnitude                | calibration slope (1 = perfect)                 | 1.124 |                 -0.249 | corpus mean curve | 1.0905575675573076 | yes                           | usable, biased -0.14     |
| Q4 know when it does not know                     | sign acc at 10 % coverage on its own confidence | 0.813 |                  0.812 | no abstention     | -                  | no (below baseline in 4 of 5) | null                     |

---

## Q1. Given a target metal pair, does the model call the direction right?

Sign accuracy on held-out pairs with |observed log SF| >= 0.3 (gen13's strong-pair threshold, about 0.9 pair-noise sd). 40 455 of 70 750 pairs qualify, over 405 cells and 86 extractants.

**Table 1a — pair-direction accuracy, all five designs**

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 0.5   | 0.5   | 0.5   | 0.5   | 0.5   |
| MEAN_CURVE     | 0.579 | 0.571 | 0.568 | 0.609 | 0.512 |
| HEAVIER_ALWAYS | 0.63  | 0.63  | 0.63  | 0.63  | 0.63  |
| G14            | 0.808 | 0.809 | 0.81  | 0.808 | 0.812 |
| G14_TIED       | 0.81  | 0.809 | 0.815 | 0.807 | 0.825 |
| G13_FULL       | 0.771 | 0.762 | 0.774 | 0.826 | 0.689 |
| O_BOTH         | 0.972 | 0.972 | 0.972 | 0.972 | 0.972 |

The cheapest sensible alternative is `HEAVIER_ALWAYS` at 0.630, not `FLAT`. G14 beats it by +0.18 under every design, and the gain is the most robust result in this study.

**Table 1b — by how far apart the metals are, and by how large the true separation is (design BP)**

| band    |   FLAT |   MEAN_CURVE |   HEAVIER_ALWAYS |   G14 |   G13_FULL |   O_BOTH |
|:--------|-------:|-------------:|-----------------:|------:|-----------:|---------:|
| 0.3-0.5 |    0.5 |        0.469 |            0.629 | 0.801 |      0.652 |    0.95  |
| 0.5-1.0 |    0.5 |        0.478 |            0.65  | 0.826 |      0.672 |    0.975 |
| 1.0-2.0 |    0.5 |        0.615 |            0.772 | 0.872 |      0.793 |    0.994 |
| >2.0    |    0.5 |        0.707 |            0.852 | 0.904 |      0.89  |    1     |
| all     |    0.5 |        0.512 |            0.63  | 0.812 |      0.689 |    0.972 |
| dZ 2-4  |    0.5 |        0.557 |            0.66  | 0.789 |      0.7   |    0.958 |
| dZ 5-8  |    0.5 |        0.493 |            0.636 | 0.829 |      0.69  |    0.99  |
| dZ 9-13 |    0.5 |        0.498 |            0.651 | 0.866 |      0.73  |    0.994 |
| dZ=1    |    0.5 |        0.641 |            0.725 | 0.769 |      0.723 |    0.881 |


Band sizes (pairs / extractants):

| band    |   n_pairs |   n_extractants |
|:--------|----------:|----------------:|
| all     |     40455 |              86 |
| dZ=1    |      1770 |              59 |
| dZ 2-4  |     13175 |              81 |
| dZ 5-8  |     16160 |              83 |
| dZ 9-13 |      8740 |              74 |
| 0.3-0.5 |     10695 |              82 |
| 0.5-1.0 |     13510 |              78 |
| 1.0-2.0 |     11480 |              50 |
| >2.0    |      4770 |              27 |

Direction is called best on wide, strongly separated pairs and worst on neighbours: 0.77 at dZ = 1 against 0.87 at dZ >= 9. The neighbour case is the industrially interesting one.

**Table 1c — gen13's chemotype-blocked paired bootstrap (10 000 replicates), reference minus candidate, positive = candidate better**

| comparison     |      B |     BR |     BQ |       A |     BP |
|:---------------|-------:|-------:|-------:|--------:|-------:|
| G13_vs_HEAVIER | 0.1408 | 0.132  | 0.144  |  0.1964 | 0.0594 |
| G14_vs_FLAT    | 0.3081 | 0.3093 | 0.3102 |  0.3085 | 0.3122 |
| G14_vs_G13     | 0.0375 | 0.0475 | 0.0364 | -0.0178 | 0.123  |
| G14_vs_HEAVIER | 0.1784 | 0.1795 | 0.1804 |  0.1787 | 0.1825 |


Two-sided bootstrap p:

| comparison     |      B |     BR |     BQ |      A |     BP |
|:---------------|-------:|-------:|-------:|-------:|-------:|
| G13_vs_HEAVIER | 0.0144 | 0.018  | 0.0122 | 0.0002 | 0.346  |
| G14_vs_FLAT    | 0      | 0      | 0      | 0      | 0      |
| G14_vs_G13     | 0.4026 | 0.2786 | 0.4192 | 0.5682 | 0.0018 |
| G14_vs_HEAVIER | 0.0176 | 0.0154 | 0.0164 | 0.0162 | 0.0116 |

G14 over the heavier-always rule passes gen13's full pre-registered rule P1 under all five designs. G13_FULL over the same rule does **not** under BP (p = 0.35).

---

## Q2. For one system, can the model rank its own pairs, and name its best one?

Held-out cells with >= 4 measured metals (321 cells). Spearman and Kendall tau-b between predicted and observed log SF over the cell's pairs; top-1 / top-3 for 'which pair does this system separate best', ranked by |predicted log SF| against |observed log SF|.

**Table 2a — within-cell rank correlation, all five designs**

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 0     | 0     | 0     | 0     | 0     |
| MEAN_CURVE     | 0.256 | 0.25  | 0.246 | 0.289 | 0.171 |
| HEAVIER_ALWAYS | 0.261 | 0.261 | 0.261 | 0.261 | 0.261 |
| G14            | 0.506 | 0.503 | 0.508 | 0.511 | 0.504 |
| G14_TIED       | 0.511 | 0.509 | 0.511 | 0.52  | 0.514 |
| G13_FULL       | 0.486 | 0.47  | 0.485 | 0.538 | 0.38  |
| O_BOTH         | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 |
| _RANDOM        | 0     | 0     | 0     | 0     | 0     |


Kendall tau-b:

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 0     | 0     | 0     | 0     | 0     |
| MEAN_CURVE     | 0.192 | 0.187 | 0.184 | 0.226 | 0.124 |
| HEAVIER_ALWAYS | 0.216 | 0.216 | 0.216 | 0.216 | 0.216 |
| G14            | 0.415 | 0.41  | 0.416 | 0.42  | 0.409 |
| G14_TIED       | 0.418 | 0.416 | 0.417 | 0.43  | 0.418 |
| G13_FULL       | 0.398 | 0.384 | 0.398 | 0.438 | 0.294 |
| O_BOTH         | 0.629 | 0.629 | 0.629 | 0.629 | 0.629 |
| _RANDOM        | 0     | 0     | 0     | 0     | 0     |

**Table 2b — naming the best-separated pair**

top-1:

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 |
| MEAN_CURVE     | 0.131 | 0.133 | 0.122 | 0.138 | 0.136 |
| HEAVIER_ALWAYS | 0.412 | 0.412 | 0.412 | 0.412 | 0.412 |
| G14            | 0.359 | 0.34  | 0.339 | 0.349 | 0.25  |
| G14_TIED       | 0.291 | 0.289 | 0.279 | 0.412 | 0.298 |
| G13_FULL       | 0.301 | 0.304 | 0.325 | 0.289 | 0.181 |
| O_BOTH         | 0.357 | 0.357 | 0.357 | 0.357 | 0.357 |
| _RANDOM        | 0.039 | 0.039 | 0.039 | 0.039 | 0.039 |


top-3:

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 0.112 | 0.112 | 0.112 | 0.112 | 0.112 |
| MEAN_CURVE     | 0.28  | 0.272 | 0.273 | 0.323 | 0.275 |
| HEAVIER_ALWAYS | 0.536 | 0.536 | 0.536 | 0.536 | 0.536 |
| G14            | 0.529 | 0.51  | 0.52  | 0.531 | 0.479 |
| G14_TIED       | 0.53  | 0.532 | 0.531 | 0.533 | 0.437 |
| G13_FULL       | 0.477 | 0.455 | 0.474 | 0.493 | 0.31  |
| O_BOTH         | 0.684 | 0.684 | 0.684 | 0.684 | 0.684 |
| _RANDOM        | 0.112 | 0.112 | 0.112 | 0.112 | 0.112 |

This is the study's first clean negative. Ranking a cell's pairs, G14 is far above the trivial rule (0.50 against 0.26 Spearman, every design). *Naming* the best-separated pair, it is **below** it under every design: 0.25 under BP and 0.34-0.36 under the others, against 0.41 for 'take the widest radius gap'. So is the oracle `O_BOTH`, at 0.357. The widest-gap pair *is* the truly best-separated one 41 % of the time, and adding a curvature term — even the cell's own — moves the predicted maximum off it more often than it helps.

**Table 2c — paired bootstrap on the two Q2 statistics**

|                                               |       B |      BR |      BQ |       A |      BP |
|:----------------------------------------------|--------:|--------:|--------:|--------:|--------:|
| ('Q2 best-pair top-1', 'G14_vs_HEAVIER')      | -0.0531 | -0.0721 | -0.0724 | -0.0629 | -0.1618 |
| ('Q2 best-pair top-1', 'OBOTH_vs_HEAVIER')    | -0.0546 | -0.0546 | -0.0546 | -0.0546 | -0.0546 |
| ('Q2 within-cell Spearman', 'G14_vs_G13')     |  0.0203 |  0.032  |  0.0226 | -0.0273 |  0.1234 |
| ('Q2 within-cell Spearman', 'G14_vs_HEAVIER') |  0.2448 |  0.2411 |  0.2464 |  0.2497 |  0.2423 |


p:

|                                               |      B |     BR |     BQ |      A |     BP |
|:----------------------------------------------|-------:|-------:|-------:|-------:|-------:|
| ('Q2 best-pair top-1', 'G14_vs_HEAVIER')      | 0      | 0.0004 | 0      | 0      | 0.0004 |
| ('Q2 best-pair top-1', 'OBOTH_vs_HEAVIER')    | 0.2204 | 0.2204 | 0.2204 | 0.2204 | 0.2204 |
| ('Q2 within-cell Spearman', 'G14_vs_G13')     | 0.6606 | 0.4844 | 0.6434 | 0.501  | 0.022  |
| ('Q2 within-cell Spearman', 'G14_vs_HEAVIER') | 0.0008 | 0.0006 | 0.0004 | 0.0002 | 0.0004 |

`G14_vs_HEAVIER` on top-1 is negative under all five designs with p <= 0.0004 and 0 of 5 seeds positive: a significant, reproducible **loss** to the trivial rule.

**Table 2d — confound: the number of metals the cell measured** (Spearman +0.49 with |a| in this corpus), design BP

| n_metals_band   |   FLAT |   MEAN_CURVE |   HEAVIER_ALWAYS |   G14 |   G13_FULL |   O_BOTH |   _RANDOM |
|:----------------|-------:|-------------:|-----------------:|------:|-----------:|---------:|----------:|
| 14              |      0 |        0.182 |            0.31  | 0.537 |      0.467 |    0.818 |         0 |
| 4-5             |      0 |       -0.297 |           -0.025 | 0.325 |     -0.145 |    0.723 |         0 |
| 6-8             |      0 |        0.357 |            0.586 | 0.509 |      0.508 |    0.761 |         0 |
| 9-13            |      0 |        0.044 |            0.095 | 0.508 |      0.346 |    0.725 |         0 |


top-1:

| n_metals_band   |   FLAT |   MEAN_CURVE |   HEAVIER_ALWAYS |   G14 |   G13_FULL |   O_BOTH |   _RANDOM |
|:----------------|-------:|-------------:|-----------------:|------:|-----------:|---------:|----------:|
| 14              |  0.013 |        0.045 |            0.385 | 0.181 |      0.104 |    0.287 |     0.013 |
| 4-5             |  0.158 |        0.439 |            0.717 | 0.48  |      0.526 |    0.636 |     0.158 |
| 6-8             |  0.058 |        0.26  |            0.59  | 0.463 |      0.27  |    0.592 |     0.058 |
| 9-13            |  0.028 |        0.047 |            0.223 | 0.156 |      0.107 |    0.279 |     0.028 |

G14's ranking advantage is not uniform: on cells that measured 6-8 metals the trivial rule ranks better than the model (0.59 vs 0.51). The top-1 loss holds in every band.

---

## Q3. The design question: for a target pair, which extractant separates it best?

For each split seed and metal pair, every held-out extractant that measured that pair is ranked by predicted log SF (an extractant's several condition sets collapsed by the median). 91 metal pairs x 5 seeds = 455 ranking tasks, median 58 extractants per task. `regret` = the observed log SF of the truly best extractant minus that of the model's pick, in log10 units, averaged over the two directions a chemist could ask for.

**Table 3a — cross-extractant selection, all five designs**

rank correlation with the observed ordering:

| arm            |      B |     BR |     BQ |      A |     BP |
|:---------------|-------:|-------:|-------:|-------:|-------:|
| FLAT           |  0     |  0     |  0     |  0     |  0     |
| MEAN_CURVE     | -0.249 | -0.272 | -0.267 | -0.056 | -0.374 |
| HEAVIER_ALWAYS |  0     |  0     |  0     |  0     |  0     |
| G14            |  0.348 |  0.4   |  0.36  |  0.38  |  0.351 |
| G14_TIED       |  0.51  |  0.5   |  0.505 |  0.516 |  0.515 |
| G13_FULL       |  0.503 |  0.47  |  0.5   |  0.583 |  0.304 |
| O_BOTH         |  0.884 |  0.884 |  0.884 |  0.884 |  0.884 |
| _RANDOM        |  0     |  0     |  0     |  0     |  0     |


top-1 hit rate (chance = 0.017):

| arm            |      B |     BR |     BQ |      A |     BP |
|:---------------|-------:|-------:|-------:|-------:|-------:|
| FLAT           | 0.0171 | 0.0171 | 0.0171 | 0.0171 | 0.0171 |
| MEAN_CURVE     | 0.0051 | 0.0044 | 0.0064 | 0.0121 | 0.0063 |
| HEAVIER_ALWAYS | 0.0171 | 0.0171 | 0.0171 | 0.0171 | 0.0171 |
| G14            | 0.0079 | 0.0187 | 0.0062 | 0.0181 | 0.0226 |
| G14_TIED       | 0.0304 | 0.0301 | 0.0305 | 0.0355 | 0.0352 |
| G13_FULL       | 0.0011 | 0.0132 | 0.0055 | 0.0319 | 0.0967 |
| O_BOTH         | 0.5275 | 0.5275 | 0.5275 | 0.5275 | 0.5275 |
| _RANDOM        | 0.0171 | 0.0171 | 0.0171 | 0.0171 | 0.0171 |


regret, log10 units (lower better; random = 1.783, oracle = 0.203):

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 1.783 | 1.783 | 1.783 | 1.783 | 1.783 |
| MEAN_CURVE     | 2.077 | 2.065 | 2.065 | 1.869 | 2.175 |
| HEAVIER_ALWAYS | 1.783 | 1.783 | 1.783 | 1.783 | 1.783 |
| G14            | 1.55  | 1.465 | 1.555 | 1.489 | 1.475 |
| G14_TIED       | 1.371 | 1.378 | 1.373 | 1.352 | 1.355 |
| G13_FULL       | 1.337 | 1.389 | 1.353 | 1.332 | 1.232 |
| O_BOTH         | 0.203 | 0.203 | 0.203 | 0.203 | 0.203 |
| _RANDOM        | 1.783 | 1.783 | 1.783 | 1.783 | 1.783 |

The model **orders** extractants better than chance (Spearman 0.35, consistent across designs) but its **top pick is at chance** (0.023 against 0.017). It removes 0.31 of the 1.58 log units of regret that separate a random pick from a perfect one — about 19 %.

**Table 3b — split by how far apart the target metals are (design BP)**

| band              |   _RANDOM |   G14 |   G14_TIED |   G13_FULL |   O_BOTH |
|:------------------|----------:|------:|-----------:|-----------:|---------:|
| adjacent (dZ=1)   |     0.863 | 0.806 |      0.797 |      0.841 |    0.584 |
| extreme (dZ 9-13) |     3.064 | 2.454 |      2.194 |      2.079 |    0.025 |
| far (dZ 5-8)      |     1.988 | 1.629 |      1.482 |      1.2   |    0.065 |
| near (dZ 2-4)     |     1.108 | 0.949 |      0.901 |      0.862 |    0.288 |


fraction of the achievable regret reduction captured:

| band              |   G14 |   G14_TIED |   G13_FULL |
|:------------------|------:|-----------:|-----------:|
| adjacent (dZ=1)   | 0.204 |      0.235 |      0.076 |
| extreme (dZ 9-13) | 0.201 |      0.286 |      0.324 |
| far (dZ 5-8)      | 0.187 |      0.263 |      0.41  |
| near (dZ 2-4)     | 0.193 |      0.253 |      0.3   |

The feasible arms capture a flat ~19-25 % of the achievable gain in every band. The oracle does not: it leaves 0.584 of 0.863 log units of regret on **adjacent** pairs (68 % of random) and 0.025 of 3.064 on extreme ones (0.8 %). The quadratic centred curve is a near-complete description of the design problem for wide pairs and close to useless for neighbours — a limit of the representation, not of any model fitted in it.

**Table 3c — the industrially relevant pairs, design BP** (Nd/Pr and Pr/Nd are the same unordered pair and appear once)


rank correlation:

| pair   |   _RANDOM |   G14 |   G14_TIED |   G13_FULL |   O_BOTH |
|:-------|----------:|------:|-----------:|-----------:|---------:|
| Dy/Ho  |         0 | 0.277 |      0.34  |      0.019 |    0.683 |
| Er/Yb  |         0 | 0.227 |      0.293 |     -0.122 |    0.687 |
| Eu/Gd  |         0 | 0.272 |      0.404 |      0.445 |    0.559 |
| La/Ce  |         0 | 0.321 |      0.377 |      0.187 |    0.813 |
| Nd/Dy  |         0 | 0.477 |      0.64  |      0.537 |    0.971 |
| Pr/Nd  |         0 | 0.234 |      0.312 |      0.36  |    0.698 |
| Sm/Eu  |         0 | 0.387 |      0.499 |      0.459 |    0.774 |


top-1 hit rate:

| pair   |   _RANDOM |   G14 |   G14_TIED |   G13_FULL |   O_BOTH |
|:-------|----------:|------:|-----------:|-----------:|---------:|
| Dy/Ho  |     0.018 | 0     |      0.012 |        0   |      0.5 |
| Er/Yb  |     0.017 | 0.017 |      0.031 |        0   |      0.5 |
| Eu/Gd  |     0.014 | 0     |      0.02  |        0.2 |      0   |
| La/Ce  |     0.015 | 0.059 |      0.027 |        0   |      0   |
| Nd/Dy  |     0.014 | 0     |      0.033 |        0   |      0.5 |
| Pr/Nd  |     0.014 | 0.024 |      0.031 |        0.4 |      1   |
| Sm/Eu  |     0.013 | 0.012 |      0.023 |        0   |      0   |


regret, log10 units:

| pair   |   _RANDOM |   G14 |   G14_TIED |   G13_FULL |   O_BOTH |
|:-------|----------:|------:|-----------:|-----------:|---------:|
| Dy/Ho  |     0.644 | 0.578 |      0.581 |      0.7   |    0.017 |
| Er/Yb  |     1.486 | 1.377 |      1.397 |      1.572 |    0.4   |
| Eu/Gd  |     0.824 | 0.763 |      0.755 |      0.546 |    0.719 |
| La/Ce  |     1.036 | 1.008 |      0.924 |      1.341 |    0.806 |
| Nd/Dy  |     2.045 | 1.65  |      1.502 |      1.143 |    0.037 |
| Pr/Nd  |     0.662 | 0.624 |      0.588 |      0.4   |    0     |
| Sm/Eu  |     1.055 | 0.94  |      0.935 |      1.184 |    0.759 |


candidates and spread per pair:

| pair   |   n_units |   obs_spread |
|:-------|----------:|-------------:|
| Nd/Dy  |        71 |         4.09 |
| Pr/Nd  |        71 |         1.32 |
| Sm/Eu  |        77 |         2.11 |
| Eu/Gd  |        70 |         1.65 |
| La/Ce  |        68 |         2.07 |
| Dy/Ho  |        57 |         1.29 |
| Er/Yb  |        58 |         2.97 |

On the pairs the field cares about, the model's top pick is at or near chance for every one of them (the one apparent exception, La/Ce at 0.059 against 0.015 under BP, runs 0.000-0.059 across the five designs and is noise), and regret falls by only 0.03-0.40 log units out of 0.64-2.05. The oracle cannot pick the best extractant for Eu/Gd, La/Ce or Sm/Eu either.

**Table 3d — the confound that matters most: hold the laboratory fixed.** Ranking is restricted to (metal pair, publication) blocks with >= 5 extractants measured by the same group under the same protocol — 1 230 tasks from 4 publications, median 6 candidates, mean observed spread 1.14 log units instead of the pooled 3.57.


rank correlation:

| arm            |      B |     BR |     BQ |      A |     BP |
|:---------------|-------:|-------:|-------:|-------:|-------:|
| FLAT           |  0     |  0     |  0     |  0     |  0     |
| MEAN_CURVE     |  0.002 | -0.135 | -0.167 |  0.034 | -0.281 |
| HEAVIER_ALWAYS |  0     |  0     |  0     |  0     |  0     |
| G14            | -0.092 | -0.007 | -0.075 | -0.118 | -0.128 |
| G14_TIED       |  0.036 |  0.047 |  0.036 |  0.036 |  0.058 |
| G13_FULL       | -0.015 | -0.025 | -0.062 |  0.027 | -0.182 |
| O_BOTH         |  0.739 |  0.739 |  0.739 |  0.739 |  0.739 |
| _RANDOM        |  0     |  0     |  0     |  0     |  0     |


top-1 (chance = 0.148):

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 0.148 | 0.148 | 0.148 | 0.148 | 0.148 |
| MEAN_CURVE     | 0.137 | 0.094 | 0.092 | 0.159 | 0.083 |
| HEAVIER_ALWAYS | 0.148 | 0.148 | 0.148 | 0.148 | 0.148 |
| G14            | 0.118 | 0.139 | 0.124 | 0.11  | 0.11  |
| G14_TIED       | 0.15  | 0.148 | 0.15  | 0.15  | 0.146 |
| G13_FULL       | 0.126 | 0.123 | 0.108 | 0.123 | 0.13  |
| O_BOTH         | 0.669 | 0.669 | 0.669 | 0.669 | 0.669 |
| _RANDOM        | 0.148 | 0.148 | 0.148 | 0.148 | 0.148 |


regret, log10 units (random = 0.572):

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 0.572 | 0.572 | 0.572 | 0.572 | 0.572 |
| MEAN_CURVE     | 0.585 | 0.652 | 0.678 | 0.542 | 0.753 |
| HEAVIER_ALWAYS | 0.572 | 0.572 | 0.572 | 0.572 | 0.572 |
| G14            | 0.618 | 0.566 | 0.608 | 0.654 | 0.643 |
| G14_TIED       | 0.54  | 0.532 | 0.54  | 0.54  | 0.523 |
| G13_FULL       | 0.518 | 0.521 | 0.541 | 0.469 | 0.696 |
| O_BOTH         | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 |
| _RANDOM        | 0.572 | 0.572 | 0.572 | 0.572 | 0.572 |


**Within one laboratory the model is not better than random.** G14's rank correlation is negative under four of five designs, its top-1 is below chance under four of five, and its regret is worse than a random pick under four of five. The oracle solves the same task (Spearman 0.739, top-1 0.669, regret 0.056), so the task is not intrinsically impossible — the model simply has nothing to say. The mechanism is direct: **70-85 % of these ranking tasks receive the same direction call for every candidate extractant**, so gen14's one bit is constant across the set being ranked. Pooled across laboratories that figure is 0 %. Nearly all of the pooled Q3 gain in Table 3a is therefore between-laboratory information, not the ligand comparison a chemist would actually make.

Paired bootstrap on the same tasks, blocked on the publication (only 4 blocks, so the interval is coarse — the point estimates carry the message); positive = candidate better than a random pick:

|                                              |       B |      BR |      BQ |       A |      BP |
|:---------------------------------------------|--------:|--------:|--------:|--------:|--------:|
| ('within-pub regret', 'G13_vs_RANDOM')       |  0.0535 |  0.0508 |  0.0306 |  0.1029 | -0.1248 |
| ('within-pub regret', 'G14TIED_vs_RANDOM')   |  0.0314 |  0.04   |  0.0314 |  0.0314 |  0.0486 |
| ('within-pub regret', 'G14_vs_RANDOM')       | -0.0465 |  0.0053 | -0.0365 | -0.0826 | -0.0715 |
| ('within-pub regret', 'OBOTH_vs_RANDOM')     |  0.5154 |  0.5154 |  0.5154 |  0.5154 |  0.5154 |
| ('within-pub spearman', 'G13_vs_RANDOM')     | -0.0154 | -0.0254 | -0.0625 |  0.0269 | -0.1819 |
| ('within-pub spearman', 'G14TIED_vs_RANDOM') |  0.0355 |  0.0468 |  0.0355 |  0.0355 |  0.0581 |
| ('within-pub spearman', 'G14_vs_RANDOM')     | -0.0921 | -0.0074 | -0.0751 | -0.1175 | -0.1276 |


p:

|                                              |      B |     BR |     BQ |      A |     BP |
|:---------------------------------------------|-------:|-------:|-------:|-------:|-------:|
| ('within-pub regret', 'G13_vs_RANDOM')       | 0.5348 | 0.5444 | 0.649  | 0.1246 | 0.082  |
| ('within-pub regret', 'G14TIED_vs_RANDOM')   | 0.1246 | 0.1246 | 0.1246 | 0.1246 | 0.1246 |
| ('within-pub regret', 'G14_vs_RANDOM')       | 0.1246 | 0.7512 | 0.1464 | 0.3714 | 0.1246 |
| ('within-pub regret', 'OBOTH_vs_RANDOM')     | 0      | 0      | 0      | 0      | 0      |
| ('within-pub spearman', 'G13_vs_RANDOM')     | 0.7876 | 0.3204 | 0      | 0.6218 | 0.082  |
| ('within-pub spearman', 'G14TIED_vs_RANDOM') | 0.1246 | 0.1246 | 0.1246 | 0.1246 | 0.1246 |
| ('within-pub spearman', 'G14_vs_RANDOM')     | 0.1464 | 0.8272 | 0.2374 | 0.2848 | 0.1246 |

The oracle beats a random pick by 0.515 log units with p < 1e-4. No feasible arm is significantly better than a random pick under any design; G14's point estimate is on the wrong side of zero under four of the five, and the one feasible arm that does reach significance — G13_FULL's rank correlation under BQ — is significantly *worse* than random.

**Table 3e — the cross-fold artifact.** `G14_TIED` differs from `G14` only in using one global magnitude instead of each fold's own. It is *better* on every Q3 statistic under every design (paired bootstrap on regret: +0.087 to +0.183 log units, p < 1e-3 everywhere). In a leave-one-group-out design the held-out group is exactly the group excluded from the fold's training mean, so out-of-fold predictions are anti-correlated with the truth across folds; `MEAN_CURVE`'s cross-extractant Spearman of -0.06 to -0.37 is the pure form of the same artifact. Any cross-system comparison must be made with one fixed model, never by pooling out-of-fold predictions.

**Table 3f — paired bootstrap on the Q3 statistics** (unit and block = the metal-pair task; pairs share cells and metals so this interval is the optimistic one)

|                                                   |      B |     BR |     BQ |      A |     BP |
|:--------------------------------------------------|-------:|-------:|-------:|-------:|-------:|
| ('Q3 cross-extractant Spearman', 'G13_vs_RANDOM') | 0.503  | 0.4695 | 0.5004 | 0.5835 | 0.3044 |
| ('Q3 cross-extractant Spearman', 'G14_vs_RANDOM') | 0.3476 | 0.4002 | 0.3601 | 0.3805 | 0.3513 |
| ('Q3 cross-extractant regret', 'G13_vs_RANDOM')   | 0.4458 | 0.394  | 0.43   | 0.4517 | 0.5517 |
| ('Q3 cross-extractant regret', 'G14TIED_vs_G14')  | 0.1799 | 0.0865 | 0.1825 | 0.1376 | 0.1202 |
| ('Q3 cross-extractant regret', 'G14_vs_RANDOM')   | 0.2328 | 0.3184 | 0.2281 | 0.2938 | 0.3078 |
| ('Q3 cross-extractant regret', 'OBOTH_vs_RANDOM') | 1.5804 | 1.5804 | 1.5804 | 1.5804 | 1.5804 |


G13_FULL against G14 on regret, and against the artifact-corrected G14_TIED:

| comparison     |      B |      BR |     BQ |      A |     BP |
|:---------------|-------:|--------:|-------:|-------:|-------:|
| G13_vs_G14     | 0.213  |  0.0756 | 0.2019 | 0.1579 | 0.2439 |
| G13_vs_G14TIED | 0.0331 | -0.0109 | 0.0193 | 0.0203 | 0.1237 |


p:

| comparison     |     B |     BR |     BQ |     A |     BP |
|:---------------|------:|-------:|-------:|------:|-------:|
| G13_vs_G14     | 0     | 0.008  | 0      | 0     | 0      |
| G13_vs_G14TIED | 0.066 | 0.6306 | 0.2526 | 0.407 | 0.0002 |

G13_FULL appears to beat G14 on the design question under all five designs — but against `G14_TIED` the advantage is -0.011 to +0.124 with p between 0.0002 and 0.63 and one negative sign, so it fails the consistency requirement. The apparent win is mostly G14's cross-fold jitter, not information in the 209 columns.

**Table 3g — the magnitude-only form of the same question.** The brief also names the weaker version: rank by |predicted log SF| and score against |observed log SF|, so a system that separates the pair strongly in the *wrong* direction still counts. Random = 0 Spearman, 0.017 top-1, 1.519 regret.


rank correlation:

| arm            |      B |     BR |     BQ |      A |     BP |
|:---------------|-------:|-------:|-------:|-------:|-------:|
| FLAT           |  0     |  0     |  0     |  0     |  0     |
| MEAN_CURVE     | -0.099 | -0.119 | -0.113 | -0.018 | -0.137 |
| HEAVIER_ALWAYS |  0     |  0     |  0     |  0     |  0     |
| G14            |  0.009 |  0.064 |  0.021 |  0.028 | -0.03  |
| G14_TIED       |  0.113 |  0.114 |  0.11  |  0.109 |  0.108 |
| G13_FULL       |  0.254 |  0.248 |  0.246 |  0.339 |  0.103 |
| O_BOTH         |  0.758 |  0.758 |  0.758 |  0.758 |  0.758 |
| _RANDOM        |  0     |  0     |  0     |  0     |  0     |


top-1:

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 0.017 | 0.017 | 0.017 | 0.017 | 0.017 |
| MEAN_CURVE     | 0.008 | 0.009 | 0.01  | 0.014 | 0.013 |
| HEAVIER_ALWAYS | 0.017 | 0.017 | 0.017 | 0.017 | 0.017 |
| G14            | 0.008 | 0.017 | 0.007 | 0.019 | 0.019 |
| G14_TIED       | 0.019 | 0.019 | 0.019 | 0.018 | 0.017 |
| G13_FULL       | 0     | 0.009 | 0     | 0.048 | 0.136 |
| O_BOTH         | 0.462 | 0.462 | 0.462 | 0.462 | 0.462 |
| _RANDOM        | 0.017 | 0.017 | 0.017 | 0.017 | 0.017 |


regret, log10 units:

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| FLAT           | 1.519 | 1.519 | 1.519 | 1.519 | 1.519 |
| MEAN_CURVE     | 1.724 | 1.723 | 1.742 | 1.574 | 1.716 |
| HEAVIER_ALWAYS | 1.519 | 1.519 | 1.519 | 1.519 | 1.519 |
| G14            | 1.661 | 1.676 | 1.683 | 1.586 | 1.65  |
| G14_TIED       | 1.474 | 1.474 | 1.477 | 1.48  | 1.477 |
| G13_FULL       | 1.25  | 1.354 | 1.309 | 1.308 | 1.048 |
| O_BOTH         | 0.24  | 0.24  | 0.24  | 0.24  | 0.24  |
| _RANDOM        | 1.519 | 1.519 | 1.519 | 1.519 | 1.519 |


Here G14 is **worse than a random pick under all five designs** (regret 1.59-1.68 against 1.519) with a rank correlation of -0.03 to +0.06. That is the direct consequence of its construction: its magnitude is one number per fold, so |prediction| is nearly constant across the extractants being ranked and the little variation it has is cross-fold jitter. G13_FULL, which does vary its magnitude per ligand, beats random consistently here (1.05-1.35) even though it loses to G14 on the programme's own MAE endpoint under BP. The endpoint and the decision disagree about which of the two models is more useful.

---

## Q4. Is the predicted size of the separation believable, and does the model know when it does not know?

Calibration slope is the weighted least-squares regression of observed on predicted log SF with each extractant weighted equally; 1.0 is perfect, > 1 means the model understates the separation.

**Table 4a — calibration slope and R^2**

| arm            |     B |    BR |    BQ |     A |     BP |
|:---------------|------:|------:|------:|------:|-------:|
| MEAN_CURVE     | 0.512 | 0.555 | 0.538 | 1.074 | -0.249 |
| HEAVIER_ALWAYS | 0.284 | 0.284 | 0.284 | 0.284 |  0.284 |
| G14            | 1.166 | 1.152 | 1.191 | 1.176 |  1.124 |
| G14_TIED       | 1.26  | 1.203 | 1.273 | 1.245 |  1.37  |
| G13_FULL       | 1.081 | 1.079 | 1.082 | 1.129 |  0.813 |
| O_BOTH         | 1.091 | 1.091 | 1.091 | 1.091 |  1.091 |


R^2:

| arm            |     B |    BR |    BQ |     A |    BP |
|:---------------|------:|------:|------:|------:|------:|
| MEAN_CURVE     | 0.009 | 0.01  | 0.01  | 0.034 | 0.003 |
| HEAVIER_ALWAYS | 0.058 | 0.058 | 0.058 | 0.058 | 0.058 |
| G14            | 0.278 | 0.285 | 0.291 | 0.285 | 0.261 |
| G14_TIED       | 0.299 | 0.297 | 0.31  | 0.306 | 0.304 |
| G13_FULL       | 0.289 | 0.278 | 0.293 | 0.374 | 0.124 |
| O_BOTH         | 0.905 | 0.905 | 0.905 | 0.905 | 0.905 |

G14's slope is 1.12-1.19 — the predicted magnitude is on the right scale, understating the true separation by 12-19 %. G13_FULL's is 0.81 under BP (over-spread) with R^2 = 0.12 against G14's 0.26. `HEAVIER_ALWAYS` is in radius units, so its slope is not a calibration number. Even the oracle sits at 1.09 / R^2 0.905, because the quadratic basis holds 95.6 % of the curve variance.

**Table 4b — observed against predicted in equal-count bins, design BP**


`G14`:

| bin                |    n |   pred_mean |   obs_mean |   obs_sd |
|:-------------------|-----:|------------:|-----------:|---------:|
| (-1.529, -0.671]   | 8869 |      -0.852 |     -1.288 |    1.087 |
| (-0.671, -0.449]   | 8820 |      -0.56  |     -1.065 |    0.922 |
| (-0.449, -0.271]   | 8912 |      -0.355 |     -0.709 |    0.697 |
| (-0.271, -0.162]   | 8779 |      -0.213 |     -0.522 |    0.504 |
| (-0.162, -0.0856]  | 8854 |      -0.125 |     -0.347 |    0.407 |
| (-0.0856, -0.0337] | 8893 |      -0.058 |     -0.194 |    0.299 |
| (-0.0337, 0.079]   | 8788 |       0.001 |     -0.068 |    0.294 |
| (0.079, 1.153]     | 8835 |       0.36  |      0.247 |    0.716 |


`G13_FULL`:

| bin                           |    n |   pred_mean |   obs_mean |   obs_sd |
|:------------------------------|-----:|------------:|-----------:|---------:|
| (-3.0469999999999997, -0.537] | 8844 |      -0.877 |     -1.284 |    1.042 |
| (-0.537, -0.286]              | 8844 |      -0.399 |     -0.98  |    0.945 |
| (-0.286, -0.146]              | 8843 |      -0.21  |     -0.622 |    0.789 |
| (-0.146, -0.061]              | 8844 |      -0.1   |     -0.386 |    0.617 |
| (-0.061, -0.00989]            | 8846 |      -0.034 |     -0.247 |    0.493 |
| (-0.00989, 0.034]             | 8841 |       0.012 |     -0.158 |    0.476 |
| (0.034, 0.102]                | 8844 |       0.063 |     -0.19  |    0.491 |
| (0.102, 1.834]                | 8844 |       0.254 |     -0.083 |    0.676 |


Every G14 bin is biased in the same direction: the observed mean is 0.2-0.4 log units more negative than predicted (fitted intercept -0.144). The model is scale-calibrated but systematically understates how heavy-selective this corpus is. G13_FULL's top two bins are non-monotone.

**Table 4c — risk-coverage: keep the most confident fraction of predictions**

|                                       |    mae |   mae_sd |   sign_acc |   sign_acc_sd |
|:--------------------------------------|-------:|---------:|-----------:|--------------:|
| ('G14', 'abs(prediction)', 0.05)      | 1.0019 |   0.096  |     0.8741 |        0.0285 |
| ('G14', 'abs(prediction)', 0.1)       | 0.9452 |   0.0345 |     0.8761 |        0.0212 |
| ('G14', 'abs(prediction)', 0.25)      | 0.7941 |   0.0258 |     0.8649 |        0.0159 |
| ('G14', 'abs(prediction)', 0.5)       | 0.6575 |   0.0185 |     0.8381 |        0.0138 |
| ('G14', 'abs(prediction)', 0.75)      | 0.5673 |   0.0141 |     0.8253 |        0.0176 |
| ('G14', 'abs(prediction)', 1.0)       | 0.5001 |   0.0158 |     0.8122 |        0.0169 |
| ('G13_FULL', 'abs(prediction)', 0.05) | 1.054  |   0.1086 |     0.7703 |        0.076  |
| ('G13_FULL', 'abs(prediction)', 0.1)  | 1.0074 |   0.0699 |     0.8156 |        0.0415 |
| ('G13_FULL', 'abs(prediction)', 0.25) | 0.8435 |   0.0347 |     0.802  |        0.0348 |
| ('G13_FULL', 'abs(prediction)', 0.5)  | 0.7025 |   0.0336 |     0.7358 |        0.0465 |
| ('G13_FULL', 'abs(prediction)', 0.75) | 0.5999 |   0.0232 |     0.7066 |        0.0518 |
| ('G13_FULL', 'abs(prediction)', 1.0)  | 0.5523 |   0.0231 |     0.6892 |        0.0402 |
| ('G14', 'gen14 dir conf', 0.05)       | 0.268  |   0.0283 |     0.7925 |        0.0407 |
| ('G14', 'gen14 dir conf', 0.1)        | 0.3825 |   0.0299 |     0.8132 |        0.0503 |
| ('G14', 'gen14 dir conf', 0.25)       | 0.4429 |   0.0358 |     0.7992 |        0.0286 |
| ('G14', 'gen14 dir conf', 0.5)        | 0.4836 |   0.0128 |     0.8267 |        0.0225 |
| ('G14', 'gen14 dir conf', 0.75)       | 0.4846 |   0.0151 |     0.8283 |        0.0253 |
| ('G14', 'gen14 dir conf', 1.0)        | 0.5001 |   0.0158 |     0.8122 |        0.0169 |


sign accuracy under gen14's own direction confidence (its |p - 0.5|), all designs:

| design   |   1.0 |   0.75 |   0.5 |   0.25 |   0.1 |   0.05 |
|:---------|------:|-------:|------:|-------:|------:|-------:|
| A        | 0.808 |  0.837 | 0.834 |  0.795 | 0.808 |  0.749 |
| B        | 0.808 |  0.809 | 0.808 |  0.733 | 0.781 |  0.775 |
| BP       | 0.812 |  0.828 | 0.827 |  0.799 | 0.813 |  0.792 |
| BQ       | 0.81  |  0.813 | 0.811 |  0.721 | 0.777 |  0.757 |
| BR       | 0.809 |  0.83  | 0.828 |  0.76  | 0.793 |  0.784 |


sd over seeds:

| design   |   1.0 |   0.75 |   0.5 |   0.25 |   0.1 |   0.05 |
|:---------|------:|-------:|------:|-------:|------:|-------:|
| A        | 0.024 |  0.012 | 0.02  |  0.028 | 0.019 |  0.012 |
| B        | 0.016 |  0.015 | 0.016 |  0.062 | 0.041 |  0.02  |
| BP       | 0.017 |  0.025 | 0.023 |  0.029 | 0.05  |  0.041 |
| BQ       | 0.017 |  0.015 | 0.016 |  0.024 | 0.044 |  0.019 |
| BR       | 0.023 |  0.017 | 0.016 |  0.049 | 0.044 |  0.017 |

**The model's own confidence is not a confidence.** Selecting the 5-25 % of cells the logistic is most certain about leaves sign accuracy at 0.71-0.84 against 0.81 on everything — at or below the unselected value under every design, with a seed sd of 0.02-0.06. The MAE does fall (0.50 to 0.27 at 5 % coverage) but only because those cells have smaller separations to begin with: mean |y| falls from 0.650 to 0.384, the strong-pair fraction from 0.572 to 0.431, and the retained set is 31 cells.

**Table 4d — what the confidence filter actually keeps, design BP**

|                           |   n_pairs |   n_cells |   mean_abs_y |   frac_strong |   mean_dZ |   pooled_mae |
|:--------------------------|----------:|----------:|-------------:|--------------:|----------:|-------------:|
| ('abs(prediction)', 1.0)  |     70750 |       521 |        0.65  |         0.572 |     4.94  |        0.511 |
| ('abs(prediction)', 0.5)  |     35375 |       408 |        0.972 |         0.74  |     7.075 |        0.735 |
| ('abs(prediction)', 0.25) |     17688 |       296 |        1.228 |         0.802 |     8.858 |        0.901 |
| ('abs(prediction)', 0.1)  |      7075 |       246 |        1.366 |         0.814 |    10.239 |        0.978 |
| ('abs(prediction)', 0.05) |      3538 |       189 |        1.357 |         0.803 |    10.717 |        0.966 |
| ('gen14 dir conf', 1.0)   |     70750 |       521 |        0.65  |         0.572 |     4.94  |        0.511 |
| ('gen14 dir conf', 0.5)   |     35375 |       513 |        0.638 |         0.573 |     5.005 |        0.515 |
| ('gen14 dir conf', 0.25)  |     17688 |       319 |        0.694 |         0.62  |     5.172 |        0.527 |
| ('gen14 dir conf', 0.1)   |      7075 |        58 |        0.639 |         0.576 |     5.098 |        0.497 |
| ('gen14 dir conf', 0.05)  |      3538 |        31 |        0.384 |         0.431 |     4.918 |        0.346 |

Selecting instead on |predicted log SF| does raise sign accuracy (0.812 to 0.876 at 10 % coverage, consistently across designs) — but that quantity is essentially the radius gap of the requested pair: mean dZ over the retained set rises from 4.94 to 10.72. It is known before any model is fitted, and the MAE gets *worse* (0.500 to 0.945) because the retained pairs are the large ones. There is no operating point at which abstention buys accuracy.

---

## Q5. Permutation null for every headline number

Inside the training fold only, the 137 ligand-derived columns are permuted between extractants — all of an extractant's cells receive the same substitute structure, while conditions, mass-action columns, weights, publication structure and targets stay attached to their own cell. Only the structure-to-curve map is destroyed. `FLAT`, `MEAN_CURVE`, `O_BOTH` and `HEAVIER_ALWAYS` are invariant under it by construction, which replicate 0 confirms numerically: permuted FLAT / MEAN_CURVE / O_BOTH return 0.588506 / 0.596858 / 0.181111 under B, identical to the real run in every one of the eight headline metrics to six decimals.

20 replicates per design; the smallest reportable one-sided p is 1/(R+1) = 0.048. (Two copies of the runner were briefly live at once and appended each replicate twice; the permutation is deterministic in (seed, fold, replicate) and the repeated rows agree to 1e-10, so one of each is dropped.)

**Table 5a — design BP, observed against the null**

|                                |   observed |   null_mean |   null_sd |   z_vs_null |   p_emp |
|:-------------------------------|-----------:|------------:|----------:|------------:|--------:|
| ('G14', 'macro_mae')           |     0.5001 |      0.7229 |    0.0242 |     -9.2149 |  0.0476 |
| ('G14', 'sign_acc')            |     0.8122 |      0.4006 |    0.0431 |      9.5431 |  0.0476 |
| ('G14', 'cell_spearman')       |     0.5038 |     -0.0621 |    0.0604 |      9.3647 |  0.0476 |
| ('G14', 'cell_top1')           |     0.2499 |      0.2343 |    0.0091 |      1.7203 |  0.0952 |
| ('G14', 'cross_spearman')      |     0.3513 |     -0.0642 |    0.0796 |      5.2198 |  0.0476 |
| ('G14', 'cross_top1')          |     0.0226 |      0.0151 |    0.0091 |      0.82   |  0.2381 |
| ('G14', 'cross_regret')        |     1.4755 |      1.7491 |    0.1239 |     -2.2079 |  0.0476 |
| ('G14', 'calib_slope')         |     1.1236 |     -0.226  |    0.1616 |      8.3523 |  0.0476 |
| ('G13_FULL', 'macro_mae')      |     0.5523 |      0.6666 |    0.0137 |     -8.3665 |  0.0476 |
| ('G13_FULL', 'sign_acc')       |     0.6892 |      0.4917 |    0.0313 |      6.3062 |  0.0476 |
| ('G13_FULL', 'cell_spearman')  |     0.3804 |      0.0693 |    0.0533 |      5.8313 |  0.0476 |
| ('G13_FULL', 'cell_top1')      |     0.1806 |      0.2014 |    0.0129 |     -1.6079 |  0.9524 |
| ('G13_FULL', 'cross_spearman') |     0.3044 |     -0.1319 |    0.0624 |      6.99   |  0.0476 |
| ('G13_FULL', 'cross_top1')     |     0.0967 |      0.0146 |    0.0132 |      6.218  |  0.0476 |
| ('G13_FULL', 'cross_regret')   |     1.2316 |      1.9499 |    0.1639 |     -4.3818 |  0.0476 |
| ('G13_FULL', 'calib_slope')    |     0.8129 |     -0.0579 |    0.1095 |      7.9512 |  0.0476 |

**Table 5b — z against the null, all designs**


`G14`:

| metric         |     B |    BR |    BQ |      A |    BP |
|:---------------|------:|------:|------:|-------:|------:|
| calib_slope    |  6.8  |  8.66 |  6.3  |  11.41 |  8.35 |
| cell_spearman  |  8.42 |  8.95 |  6.41 |  10.74 |  9.36 |
| cell_top1      |  2.75 |  2.49 |  3.08 |   2.98 |  1.72 |
| cross_regret   | -3.78 | -4.82 | -3.31 |  -4.26 | -2.21 |
| cross_spearman |  4.13 |  6.91 |  4.74 |  11.6  |  5.22 |
| cross_top1     | -0.47 |  1.62 | -0.17 |   0.3  |  0.82 |
| macro_mae      | -9.1  | -9.54 | -6.74 | -11.82 | -9.21 |
| sign_acc       |  8.86 |  9.62 |  6.8  |  10.63 |  9.54 |


`G13_FULL`:

| metric         |     B |    BR |    BQ |      A |    BP |
|:---------------|------:|------:|------:|-------:|------:|
| calib_slope    |  3.61 |  4.98 |  3.07 |   5.21 |  7.95 |
| cell_spearman  |  5.54 |  6.08 |  6.01 |   8.64 |  5.83 |
| cell_top1      |  2.45 |  2.78 |  3.22 |   0.13 | -1.61 |
| cross_regret   | -1.94 | -2.22 | -2.06 |  -0.97 | -4.38 |
| cross_spearman |  5.44 |  5.45 |  3.96 |  10.49 |  6.99 |
| cross_top1     | -0.98 |  0.62 | -0.11 |   1.69 |  6.22 |
| macro_mae      | -6.48 | -8.2  | -5.93 | -10.08 | -8.37 |
| sign_acc       |  6.37 |  8.01 |  6.58 |   8.01 |  6.31 |


Direction accuracy, within-cell Spearman, cross-extractant Spearman, regret and the calibration slope are all far outside their nulls under every design. The two *selection* statistics are not: G14's within-cell top-1 sits at z = +1.72 to +3.08 and its cross-extractant top-1 at z = -0.47 to +1.62, changing sign across designs (empirical p up to 0.71). G13_FULL's cross-extractant top-1 runs from z = -0.98 to +6.22 — the BP value that looked like the best selection number in the whole study is not reproducible under the other four designs.

The nulls also reproduce a known result from the outside. Under B / BR / BQ / A the *permuted* G13_FULL still reaches a within-cell Spearman of 0.20-0.32 and a cross-extractant Spearman of 0.09-0.31 with the ligand destroyed, because the 64 condition columns survive the permutation and act as a laboratory fingerprint. Under BP that channel is closed and the same nulls collapse to 0.07 and -0.13.

---

## Confounds checked

| confound | how it was handled | result |
|---|---|---|
| publication identity | all five designs reported; BP masks every training cell sharing a publication with the held-out set; BR and BQ are its size- and structure-matched controls | every sign above is consistent across the five designs, and the two that are not are reported as failures |
| laboratory as the source of the cross-extractant spread | Q3 rerun inside single publications (Table 3d) | the pooled Q3 gain does not survive it |
| number of metals measured (Spearman +0.49 with the amplitude) | Q2 stratified by band (Table 2d) | the model's ranking edge over the trivial rule disappears in the 6-8 metal band; the top-1 loss holds in every band |
| chemotype | leave-one-chemotype-out over all 45 chemotypes, every headline number | see below |
| cross-fold prediction jitter | `G14_TIED` control (Table 3e) | worth 0.09-0.18 log units of regret; it inflates every cross-system comparison made from out-of-fold predictions |

**Leave-one-chemotype-out, design BP — the range over all 45 chemotypes**

| arm            |   sign_acc min |   sign_acc max |   cell_spearman min |   cell_spearman max |   cell_top1 min |   cell_top1 max |   cross_spearman min |   cross_spearman max |   cross_regret min |   cross_regret max |
|:---------------|---------------:|---------------:|--------------------:|--------------------:|----------------:|----------------:|---------------------:|---------------------:|-------------------:|-------------------:|
| FLAT           |          0.5   |          0.5   |               0     |               0     |           0.036 |           0.041 |                0     |                0     |              1.642 |              1.783 |
| MEAN_CURVE     |          0.476 |          0.529 |               0.118 |               0.204 |           0.125 |           0.139 |               -0.404 |               -0.231 |              1.963 |              2.203 |
| HEAVIER_ALWAYS |          0.535 |          0.657 |               0.101 |               0.302 |           0.306 |           0.432 |                0     |                0     |              1.642 |              1.783 |
| G14            |          0.776 |          0.831 |               0.417 |               0.524 |           0.201 |           0.262 |                0.288 |                0.401 |              1.338 |              1.506 |
| G14_TIED       |          0.79  |          0.842 |               0.423 |               0.534 |           0.152 |           0.317 |                0.473 |                0.554 |              1.246 |              1.37  |
| G13_FULL       |          0.65  |          0.705 |               0.299 |               0.409 |           0.173 |           0.189 |                0.255 |                0.335 |              0.974 |              1.364 |
| O_BOTH         |          0.967 |          0.979 |               0.744 |               0.784 |           0.278 |           0.375 |                0.872 |                0.893 |              0.169 |              0.213 |

No single chemotype changes any conclusion: G14's direction accuracy stays in [0.776, 0.831] against the rule's [0.535, 0.657]; its within-cell top-1 stays in [0.201, 0.262] against the rule's [0.306, 0.432], i.e. the loss never flips.

---

## What the model can and cannot be trusted to do

Gen14 can be trusted for exactly one decision: **told a ligand and a metal pair, it calls which of the two metals will be extracted preferentially, and it is right about 81 % of the time on separations large enough to matter, against 63 % for the standing rule of thumb that the heavier lanthanide always wins.** That gain is +0.18, it is the same under all five hold-out designs including the publication-masked one, it survives dropping any of the 45 chemotypes, and it is seven to eleven standard deviations outside a permutation null that destroys the ligand-to-curve map. It also ranks a single system's own metal pairs sensibly (Spearman 0.50 against 0.26 for the radius-gap rule) and its predicted separations are on the right scale (calibration slope 1.12, with a systematic 0.14 log-unit understatement of heavy selectivity). Everything else fails. It cannot name which pair a system separates best — the trivial 'widest radius gap' rule beats it, 0.41 against 0.25, significantly and under every design, and it beats every other arm in the repertoire, including the oracle. It cannot pick an extractant: its top choice for a target pair is right at the chance rate (0.023 versus 0.017), for every industrially relevant pair, and although it does *order* extractants above chance and cuts the expected regret from 1.78 to 1.48 log units, that entire gain is between-laboratory information — restricted to candidates measured in the same publication, which is the comparison a chemist actually faces, it is no better than random and usually slightly worse, because in 70-85 % of those comparisons its single direction bit is identical for every candidate. Asked the magnitude-only form of the same question — which extractant separates this pair most strongly, in either direction — it is worse than a random pick under all five designs, since its magnitude is one number per training fold and carries no ligand information at all. And it has no usable notion of its own uncertainty: selecting the cells the classifier is most confident about does not raise its accuracy. A perfect fit of the same two-coefficient curve would fix the extractant-selection problem for widely separated pairs (regret 0.03 of an available 3.06) and would still fail for neighbours (0.58 of 0.86), so for adjacent-lanthanide selection — Eu/Gd, Sm/Eu, Pr/Nd, Dy/Ho — the binding constraint is the representation itself, not the model fitted in it.

---

## Files

| file | what it holds |
|---|---|
| `dec_arms.py` | the five arms with `n_jobs=2`, the gen14 probe that records the classifier's own probability, and the ligand-permutation wrapper |
| `decmetrics.py` | every decision metric, with the tie and macro conventions |
| `run_tables.py` | builds `tables/pairs_<design>.parquet` and `tables/g14_conf_<design>.csv` |
| `run_metrics.py` | Q1-Q4 plus the chemotype and n-metals confounds |
| `run_extra.py` | Q3 by dZ, the finer risk-coverage curve, and the paired bootstraps |
| `run_confound_pub.py` | the within-publication ranking control |
| `run_q3_abs.py` | the magnitude-only form of the cross-extractant question |
| `run_conf_diag.py` | what a confidence filter actually keeps |
| `run_perm.py`, `summarise_perm.py` | the permutation null and its summary |
| `checks.py` | self-checks on the metric implementations |
| `make_report.py` | renders this file from `results/` |
