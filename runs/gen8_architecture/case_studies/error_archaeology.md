# Error archaeology after calibration

*gen8 brief section 26. Which held-out-ligand failures disappear after one measurement, and which remain?*

Frozen global model `REC_ecfp_plus_recovered`, evaluated on held-out chemotypes over 5 split seeds x 5 folds. Source: `runs/gen8_architecture/active_acquisition/primary_detail.parquet`, written 2026-08-20T03:34:15. **Every number in this report is recomputed from that file; if the acquisition run is re-executed this report must be regenerated, because the RANDOM policy's draws move with it.**

## 1. What is compared, and why the comparison is exact

Five arms, all read from the same acquisition run:

| arm | adapter | policy | k |
|---|---|---|---|
| `k0` | `ZERO_SHOT_REF` | `NONE` | 0 |
| `RANDOM_k1` / `RANDOM_k2` | `OFFSET_K1` | `RANDOM` | 1 / 2 |
| `CENTRAL_k1` / `CENTRAL_k2` | `OFFSET_K1` | `CENTRAL` | 1 / 2 |

Under protocol P2 each ligand's rows are split once per repeat into a candidate pool and a disjoint evaluation set, and the zero-shot reference is emitted once per repeat on that same evaluation set. The five arms therefore share **8,580 identical (seed, fold, ligand, repeat) units** -- 143 ligands x 5 seeds x 12 repeats -- verified by set intersection before anything was averaged. Nothing below can be an artefact of one arm being scored on easier rows.

Per-ligand MAE is the mean over the 60 (seed, repeat) units of that ligand; macro is the unweighted mean over ligands.

| arm | macro MAE |
|---|---|
| k0 | 0.995 |
| RANDOM_k1 | 0.723 |
| RANDOM_k2 | 0.640 |
| CENTRAL_k1 | 0.651 |
| CENTRAL_k2 | 0.619 |

These are the macro numbers of the acquisition run named above and of no other. An earlier execution of the same protocol gave 0.998 / 0.702 / 0.646 / 0.627, and those four figures still circulate in the gen8 write-ups; they are superseded here by 0.995 / 0.723 / 0.651 / 0.640. The two executions differ in the P2 pool/evaluation split itself -- comparing the surviving smoke run against this one on the 286 (seed, fold, ligand, repeat) units they share, the evaluation sets are the same *size* in 286 of 286 cases and contain different *rows* in 280 of them -- so every arm moves, the zero-shot reference included. The size of that movement is the useful part: 0.003 on the zero-shot arm and 0.021 on the blind one-measurement arm. **A blind 1-shot policy is not determined to three decimals by its own protocol**, and the RANDOM numbers below should be read to two.

## 2. The level/shape budget the classification rests on

`_metrics` already splits every arm's error into the mean residual (`offset`) and the error left after that mean residual is removed (`shape_mae`). Because `OFFSET_K1` adds a **constant** to a ligand's predictions, `shape_mae` is identical in all five arms -- byte-identical, checked -- so it is the same quantity whatever the measurement budget and whatever the acquisition policy.

One correction to how that quantity is described elsewhere in gen8. `shape_mae` centres the residual on its **mean**; the constant that minimises MAE is the **median**. So `shape_mae` is an upper bound on the offset-only floor, not the floor. Recomputing both on each ligand's full out-of-fold row set: mean-centred 0.503, median-centred 0.480 -- a gap of 0.024 macro, and 0.052 over the ligands §3 calls RESIDUAL_SHAPE (median-centred is lower for 135 of 143 ligands). Every 'floor' in this report is therefore the mean-centred number, which overstates the true offset-only floor by roughly 5 % on the class the argument rests on. That does not change the direction of anything below, and it is stated so that no one reads `shape_mae` as an attained bound.

Macro over 143 ligands: zero-shot MAE 0.995, |offset| 0.833, `shape_mae` 0.472 (the two parts do not add: each is a mean of absolute values). The oracle 1-shot arm reaches 0.484, 0.012 from that number. **Everything a measurement can buy is the level; the shape is untouched by construction.** The question is therefore how the shape is distributed over ligands.

## 3. Failure classes

Classified from the ligand's own paired numbers, with `RANDOM` (a blindly chosen measurement) as the deployable default and `CENTRAL` as a sensitivity:

- **ALREADY_GOOD** -- zero-shot MAE <= 1.00
- **PURE_LEVEL** -- zero-shot MAE > 1.00 and 1-shot < 0.50 x zero-shot
- **PARTIAL_LEVEL** -- zero-shot MAE > 1.00 and 0.50 <= ratio <= 0.70
- **RESIDUAL_SHAPE** -- zero-shot MAE > 1.00 and 1-shot > 0.70 x zero-shot

`PARTIAL_LEVEL` is the fourth class the data forced: a ratio band between the two briefed cut-offs is not empty, and collapsing it into either neighbour would overstate that neighbour.

| failure_class | n | mae_k0 | mae_R1 | mae_R2 | mae_C1 | mae_C2 | oracle_k1 | offset_k0 | shape_k0 | shape_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| PURE_LEVEL | 25 | 1.906 | 0.494 | 0.440 | 0.451 | 0.425 | 0.339 | 1.904 | 0.337 | 0.194 |
| PARTIAL_LEVEL | 7 | 1.458 | 0.873 | 0.764 | 0.671 | 0.622 | 0.530 | 1.376 | 0.547 | 0.362 |
| RESIDUAL_SHAPE | 29 | 1.350 | 1.471 | 1.295 | 1.308 | 1.251 | 0.975 | 0.971 | 0.962 | 0.729 |
| ALREADY_GOOD | 82 | 0.552 | 0.516 | 0.458 | 0.477 | 0.454 | 0.350 | 0.410 | 0.334 | 0.591 |

(`mae_R1`/`R2` = RANDOM at k=1/2, `mae_C1`/`C2` = CENTRAL at k=1/2, `oracle_k1` = the non-deployable best single measurement, `shape_frac` = shape_k0 / mae_k0.)

Read across the `PURE_LEVEL` row and then the `RESIDUAL_SHAPE` row:

- **PURE_LEVEL (25 ligands): one measurement is a total repair.** 1.906 -> 0.494 on a random point, 0.451 on a central one. Their error was 19% shape and the rest a pure offset (|offset| 1.904 against a shape floor of 0.337). After one measurement they are **better than the ALREADY_GOOD ligands were before it**.
- **RESIDUAL_SHAPE (29 ligands): one measurement makes them worse.** 1.350 -> 1.471 random, and two measurements only reach 1.295 (1.251 with CENTRAL). Their shape floor alone is 0.962 -- 73% of their zero-shot error -- and the oracle 1-shot arm gets 0.975, within 0.013 of it while every deployable arm is 0.289 or more away. A better acquisition policy is worth the first gap and nothing beyond it, because an offset is the wrong object for these ligands.

Of the 29 RESIDUAL_SHAPE ligands, 2 drop below the 0.70 ratio by k=2 and 27 do not.

### One measurement is not free

57 of 143 ligands are *worse* after one random measurement than before it -- 39 of the 82 ALREADY_GOOD ones and 18 of the 29 RESIDUAL_SHAPE ones. A single point estimates the offset with the full shape noise of that point attached; when there is little offset to remove, that noise is all you buy. CENTRAL reduces the count to 42.

Paired chemotype bootstrap, positive = calibration helps (`paired_chemotype_bootstrap`, blocks = held-out Tanimoto chemotypes):

| class | comparison | point | ci95_low | ci95_high | bca_low | bca_high | n_units | units_improved | seeds_positive | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | k0 -> RANDOM k=1 | 0.272 | 0.168 | 0.412 | 0.173 | 0.423 | 143 | 86 | 5 | 5 |
| ALL | k0 -> RANDOM k=2 | 0.355 | 0.249 | 0.485 | 0.261 | 0.509 | 143 | 98 | 5 | 5 |
| ALL | k0 -> CENTRAL k=1 | 0.344 | 0.227 | 0.470 | 0.240 | 0.492 | 143 | 101 | 5 | 5 |
| ALL | k0 -> CENTRAL k=2 | 0.376 | 0.258 | 0.498 | 0.275 | 0.525 | 143 | 101 | 5 | 5 |
| PURE_LEVEL | k0 -> RANDOM k=1 | 1.412 | 1.166 | 1.686 | 1.200 | 1.731 | 25 | 25 | 5 | 5 |
| PURE_LEVEL | k0 -> RANDOM k=2 | 1.466 | 1.215 | 1.740 | 1.247 | 1.780 | 25 | 25 | 5 | 5 |
| PURE_LEVEL | k0 -> CENTRAL k=1 | 1.454 | 1.185 | 1.719 | 1.215 | 1.747 | 25 | 25 | 5 | 5 |
| PURE_LEVEL | k0 -> CENTRAL k=2 | 1.481 | 1.225 | 1.755 | 1.258 | 1.792 | 25 | 25 | 5 | 5 |
| PARTIAL_LEVEL | k0 -> RANDOM k=1 | 0.585 | 0.522 | 0.761 | 0.521 | 0.713 | 7 | 7 | 5 | 5 |
| PARTIAL_LEVEL | k0 -> RANDOM k=2 | 0.694 | 0.648 | 0.810 | 0.648 | 0.810 | 7 | 7 | 5 | 5 |
| PARTIAL_LEVEL | k0 -> CENTRAL k=1 | 0.787 | 0.705 | 0.984 | 0.691 | 0.944 | 7 | 7 | 5 | 5 |
| PARTIAL_LEVEL | k0 -> CENTRAL k=2 | 0.836 | 0.768 | 0.986 | 0.752 | 0.965 | 7 | 7 | 5 | 5 |
| RESIDUAL_SHAPE | k0 -> RANDOM k=1 | -0.122 | -0.282 | -0.010 | -0.308 | -0.023 | 29 | 11 | 0 | 5 |
| RESIDUAL_SHAPE | k0 -> RANDOM k=2 | 0.054 | -0.115 | 0.170 | -0.118 | 0.169 | 29 | 18 | 5 | 5 |
| RESIDUAL_SHAPE | k0 -> CENTRAL k=1 | 0.042 | -0.158 | 0.160 | -0.164 | 0.156 | 29 | 19 | 4 | 5 |
| RESIDUAL_SHAPE | k0 -> CENTRAL k=2 | 0.099 | -0.106 | 0.218 | -0.104 | 0.219 | 29 | 18 | 5 | 5 |
| ALREADY_GOOD | k0 -> RANDOM k=1 | 0.037 | -0.031 | 0.124 | -0.051 | 0.105 | 82 | 43 | 5 | 5 |
| ALREADY_GOOD | k0 -> RANDOM k=2 | 0.094 | 0.041 | 0.166 | 0.027 | 0.151 | 82 | 48 | 5 | 5 |
| ALREADY_GOOD | k0 -> CENTRAL k=1 | 0.075 | 0.029 | 0.138 | 0.027 | 0.135 | 82 | 50 | 5 | 5 |
| ALREADY_GOOD | k0 -> CENTRAL k=2 | 0.098 | 0.053 | 0.160 | 0.051 | 0.157 | 82 | 51 | 5 | 5 |

Two readings of that table are load-bearing and point in opposite directions, so neither should be quoted without the other. On RESIDUAL_SHAPE, one blind measurement is a **measured harm, not a null**: -0.122 with an interval of [-0.282, -0.010] that excludes zero and 0 of 5 seeds positive. But choosing the point centrally does **not** demonstrably repair them: +0.042 [-0.158, +0.160] straddles zero. CENTRAL turns a measured harm into an unmeasurable difference; on this evidence it does not turn it into a gain. The same caution applies to ALREADY_GOOD at k=1 (+0.037 [-0.031, +0.124]).

## 4. The classes are not an artefact of the thresholds

The two cut-offs do different jobs and are swept separately, because sweeping them together would hide which one matters. The **level cut** decides only which ligands are in scope. For a ligand in scope the level/shape verdict depends on the ratio alone, so two level cuts can never disagree about a ligand they both admit -- the RESIDUAL_SHAPE sets at different level cuts are exactly nested (verified below). The **ratio cuts** are therefore the only free parameters that can change a verdict.

**(a) Ratio cuts, at the primary level cut of 1.00.**

| pure_cut | persist_cut | ALREADY_GOOD | PURE_LEVEL | PARTIAL_LEVEL | RESIDUAL_SHAPE | residual_share_of_bad | residual_share_of_k2_macro | jaccard_vs_primary |
|---|---|---|---|---|---|---|---|---|
| 0.400 | 0.600 | 82 | 19 | 10 | 32 | 0.525 | 0.431 | 0.906 |
| 0.500 | 0.700 | 82 | 25 | 7 | 29 | 0.475 | 0.410 | 1.000 |
| 0.600 | 0.800 | 82 | 29 | 5 | 27 | 0.443 | 0.383 | 0.931 |

Moving both ratio cuts by +-0.1 moves the RESIDUAL_SHAPE count between 27 and 32, with Jaccard overlap against the primary set of at least 0.91. The verdict is not sitting on a cliff edge.

**(b) The level cut only resizes the population.**

| level_cut | ALREADY_GOOD | PURE_LEVEL | PARTIAL_LEVEL | RESIDUAL_SHAPE | residual_share_of_bad | residual_share_of_k2_macro | nested_with_primary |
|---|---|---|---|---|---|---|---|
| 0.750 | 56 | 31 | 9 | 47 | 0.540 | 0.601 | yes |
| 1.000 | 82 | 25 | 7 | 29 | 0.475 | 0.410 | yes |
| 1.250 | 99 | 22 | 4 | 18 | 0.409 | 0.279 | yes |
| 1.500 | 118 | 15 | 3 | 7 | 0.280 | 0.108 | yes |

Every set here is nested with the primary one (`nested_with_primary` is true in all 4 rows): lowering the cut to 0.75 adds 18 milder shape failures without reclassifying anyone, raising it to 1.5 keeps only the 7 most extreme. What the level cut buys is the headline share: at 0.75 the shape failures carry 0.60 of the k=2 macro error, at 1.5 only 0.11. That number is a function of where the line is drawn and is quoted below with the line stated.

**(c) A threshold-free version of the same split.** `shape_frac_k0` (= shape_mae / MAE at k=0) needs no cut-off at all, and because `OFFSET_K1` cannot change `shape_mae` it is the quantity the classification is really reading. Spearman against the 1-shot ratio is 0.930 over all 143 ligands and 0.952 over the 61 in scope. Class medians: PURE_LEVEL 0.201, PARTIAL_LEVEL 0.353, RESIDUAL_SHAPE 0.693, ALREADY_GOOD 0.627.

Note that ALREADY_GOOD sits high on this scale too (0.627): most of *their* small error is shape as well. That is the point of the two-dimensional definition -- the classification asks both how big the error is and what fraction of it a constant can remove, and only the second question is threshold-sensitive in any interesting way.

**(d) Re-derivation per split seed and per policy.** Re-classifying inside each of the 5 split seeds separately, 123 of 143 ligands keep their pooled class in >= 4 of 5 seeds (102 in all 5). Re-classifying with CENTRAL instead of RANDOM moves 10 of 143 ligands; 25/29 RESIDUAL_SHAPE and 24/25 PURE_LEVEL ligands are unchanged.

**(e) The classification is assigned on one set of seeds and scored on another.** (d) asks whether the label is stable; it does not answer the sharper objection, which is that a ligand enters scope because its zero-shot MAE was large on the very units its ratio is then computed from. Selection on a noisy quantity regresses. So: the class is fixed on the first 3 split seeds and every number in this block comes from the remaining 2.

| class_fit | ALREADY_GOOD | PARTIAL_LEVEL | PURE_LEVEL | RESIDUAL_SHAPE |
|---|---|---|---|---|
| ALREADY_GOOD | 81 | 1 | 2 | 1 |
| PARTIAL_LEVEL | 0 | 4 | 2 | 0 |
| PURE_LEVEL | 0 | 2 | 22 | 0 |
| RESIDUAL_SHAPE | 1 | 0 | 0 | 27 |

| class_fit | n | mae_k0 | mae_R1 | mae_R2 | mae_C2 | mean_ratio |
|---|---|---|---|---|---|---|
| ALREADY_GOOD | 85 | 0.577 | 0.534 | 0.471 | 0.467 | 0.931 |
| PARTIAL_LEVEL | 6 | 2.061 | 1.007 | 0.885 | 0.683 | 0.504 |
| PURE_LEVEL | 24 | 1.847 | 0.454 | 0.401 | 0.401 | 0.279 |
| RESIDUAL_SHAPE | 28 | 1.361 | 1.482 | 1.313 | 1.264 | 1.103 |

134 of 143 ligands land in the same class on both halves. Of the ligands called RESIDUAL_SHAPE on the fit seeds, 27/28 are RESIDUAL_SHAPE again on the held-out seeds, and their held-out mean ratio is 1.103 -- one measurement still makes them worse on seeds that had no say in labelling them. PURE_LEVEL keeps 22/24 at a held-out ratio of 0.279. And the regression-to-the-mean worry does not materialise: the 58 ligands selected into scope on the fit seeds have mean zero-shot MAE 1.609 there and 1.634 on the held-out seeds, with 57 of them still above the level cut.

## 5. The fifteen worst ligands at k = 0, 1 and 2

### k = 0 (zero-shot)

| rank | extractant | mae_k0 | failure_class | nn_train_tanimoto | n_rows | n_curves | true_span |
|---|---|---|---|---|---|---|---|
| 1 | CCCCOP(=S)(CP(=S)(OCCCC)OCCCC)OCCCC | 4.104 | PURE_LEVEL | 0.667 | 6 | 1 | 1.958 |
| 2 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)NCc1ccc(CNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)cc1 | 3.514 | PURE_LEVEL | 0.469 | 9 | 1 | 0.674 |
| 3 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)NCc1cccc(CNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)c1 | 2.745 | PURE_LEVEL | 0.446 | 37 | 6 | 4.559 |
| 4 | CCCCCCCCN(CCCCCCCC)C(=O)[C@H](CCC)O[C@H](CCC)C(=O)N(CCCCCCCC)CCCCCCCC | 2.705 | PURE_LEVEL | 0.656 | 14 | 1 | 1.080 |
| 5 | CCCCCCCCCCN(CCCCCCCCCC)C(=O)[C@H](CCC)O[C@H](CCC)C(=O)N(CCCCCCCCCC)CCCCCCCCCC | 2.671 | PURE_LEVEL | 0.656 | 13 | 1 | 0.995 |
| 6 | CCCCCCCCCCN(CCCCCCCCCC)C(=O)C(C)OC(C)C(=O)N(CCCCCCCCCC)CCCCCCCCCC | 2.519 | PURE_LEVEL | 0.676 | 45 | 14 | 2.733 |
| 7 | CC1(C)CCC(C)(C)c2nc(-c3ccc4ccc5ccc(-c6nnc7c(n6)C(C)(C)CCC7(C)C)nc5c4n3)nnc21 | 2.304 | PURE_LEVEL | 0.579 | 6 | 1 | 1.274 |
| 8 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)NCCCOc1cc(OCCCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)cc(OCCCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)c1 | 2.155 | RESIDUAL_SHAPE | 0.462 | 15 | 2 | 3.790 |
| 9 | CN(C(=O)CC(=O)N(C)c1ccccc1)c1ccccc1 | 2.141 | PURE_LEVEL | 0.679 | 14 | 1 | 1.512 |
| 10 | CCCCCCCCN(CCCCCCCC)C(=O)[C@H](C)O[C@@H](C)C(=O)N(CCCCCCCC)CCCCCCCC | 2.055 | PURE_LEVEL | 0.676 | 14 | 1 | 2.025 |
| 11 | CCCCOP(=S)(COCP(=S)(OCCCC)OCCCC)OCCCC | 2.010 | RESIDUAL_SHAPE | 0.667 | 6 | 1 | 3.133 |
| 12 | CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC | 1.978 | PARTIAL_LEVEL | 0.395 | 230 | 122 | 5.896 |
| 13 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)NCCCOCCCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC | 1.833 | PARTIAL_LEVEL | 0.522 | 23 | 4 | 4.346 |
| 14 | CCCCCCCCN(CCCCCCCC)C(=O)c1ccc2ccc3ccc(C(=O)N(CCCCCCCC)CCCCCCCC)nc3c2n1 | 1.773 | RESIDUAL_SHAPE | 0.684 | 35 | 5 | 2.322 |
| 15 | CCCCCCCCN(CCCCCCCC)C(=O)COC(C)C(=O)N(CCCCCCCC)CCCCCCCC | 1.690 | RESIDUAL_SHAPE | 0.676 | 12 | 2 | 5.672 |

Class mix of this top-15: PURE_LEVEL 9, RESIDUAL_SHAPE 4, PARTIAL_LEVEL 2.

### k = 1 (one random measurement, offset correction)

| rank | extractant | mae_RANDOM_k1 | failure_class | nn_train_tanimoto | n_rows | n_curves | true_span |
|---|---|---|---|---|---|---|---|
| 1 | CCCCCCCCN(CCCCCCCC)C(=O)C1CCC(C(=O)N(CCCCCCCC)CCCCCCCC)O1 | 2.638 | RESIDUAL_SHAPE | 0.500 | 5 | 1 | 3.901 |
| 2 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)NCCCOc1cc(OCCCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)cc(OCCCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)c1 | 2.405 | RESIDUAL_SHAPE | 0.462 | 15 | 2 | 3.790 |
| 3 | CCCCCCCCN(CCCCCCCC)C(=O)CO | 2.052 | RESIDUAL_SHAPE | 0.667 | 10 | 2 | 4.747 |
| 4 | CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC | 1.770 | RESIDUAL_SHAPE | 0.676 | 184 | 25 | 5.948 |
| 5 | CCCCCCCCN(CCCCCCCC)C(=O)COC(C)C(=O)N(CCCCCCCC)CCCCCCCC | 1.755 | RESIDUAL_SHAPE | 0.676 | 12 | 2 | 5.672 |
| 6 | CCCCCCn1c(-c2ccccc2)c(-c2ccccc2)c2cc3ccc4cc5c(-c6ccccc6)c(-c6ccccc6)n(CCCCCC)c(=O)c5nc4c3nc2c1=O | 1.748 | RESIDUAL_SHAPE | 0.288 | 28 | 2 | 5.070 |
| 7 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)NCCc1c(CC)c(CCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)c(CC)c(CCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)c1CC | 1.667 | RESIDUAL_SHAPE | 0.451 | 15 | 2 | 2.820 |
| 8 | CCCCOP(=S)(COCP(=S)(OCCCC)OCCCC)OCCCC | 1.595 | RESIDUAL_SHAPE | 0.667 | 6 | 1 | 3.133 |
| 9 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N1CCCCC1 | 1.584 | RESIDUAL_SHAPE | 0.556 | 12 | 2 | 4.322 |
| 10 | CN(C(=O)COCC(=O)N(C)c1ccccc1)c1ccccc1 | 1.558 | RESIDUAL_SHAPE | 0.679 | 78 | 11 | 5.590 |
| 11 | CCCc1nnc(-c2cccc(-c3nnc(CCC)c(CCC)n3)n2)nc1CCC | 1.539 | RESIDUAL_SHAPE | 0.400 | 14 | 1 | 4.699 |
| 12 | CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC | 1.447 | RESIDUAL_SHAPE | 0.679 | 425 | 122 | 5.838 |
| 13 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC | 1.423 | RESIDUAL_SHAPE | 0.657 | 1,488 | 339 | 7.471 |
| 14 | CCCCCCCCN(C)C(=O)C(CCOCCCCCC)C(=O)N(C)CCCCCCCC | 1.368 | RESIDUAL_SHAPE | 0.459 | 52 | 15 | 4.675 |
| 15 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)NCCNC(=O)c1cc(C(=O)NCCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)cc(C(=O)NCCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)c1 | 1.367 | RESIDUAL_SHAPE | 0.471 | 15 | 2 | 3.598 |

Class mix of this top-15: RESIDUAL_SHAPE 15.

### k = 2 (two random measurements, offset correction)

| rank | extractant | mae_RANDOM_k2 | failure_class | nn_train_tanimoto | n_rows | n_curves | true_span |
|---|---|---|---|---|---|---|---|
| 1 | CCCCCCCCN(CCCCCCCC)C(=O)C1CCC(C(=O)N(CCCCCCCC)CCCCCCCC)O1 | 2.234 | RESIDUAL_SHAPE | 0.500 | 5 | 1 | 3.901 |
| 2 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)NCCCOc1cc(OCCCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)cc(OCCCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)c1 | 2.122 | RESIDUAL_SHAPE | 0.462 | 15 | 2 | 3.790 |
| 3 | CCCCCCCCN(CCCCCCCC)C(=O)CO | 1.903 | RESIDUAL_SHAPE | 0.667 | 10 | 2 | 4.747 |
| 4 | CCCCCCn1c(-c2ccccc2)c(-c2ccccc2)c2cc3ccc4cc5c(-c6ccccc6)c(-c6ccccc6)n(CCCCCC)c(=O)c5nc4c3nc2c1=O | 1.705 | RESIDUAL_SHAPE | 0.288 | 28 | 2 | 5.070 |
| 5 | CCCCCCCCN(CCCCCCCC)C(=O)COC(C)C(=O)N(CCCCCCCC)CCCCCCCC | 1.624 | RESIDUAL_SHAPE | 0.676 | 12 | 2 | 5.672 |
| 6 | CCCCC(CC)CN(CC(CC)CCCC)C(=O)COCC(=O)N(CC(CC)CCCC)CC(CC)CCCC | 1.437 | RESIDUAL_SHAPE | 0.676 | 184 | 25 | 5.948 |
| 7 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)NCCc1c(CC)c(CCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)c(CC)c(CCNC(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC)c1CC | 1.413 | RESIDUAL_SHAPE | 0.451 | 15 | 2 | 2.820 |
| 8 | CCCc1nnc(-c2cccc(-c3nnc(CCC)c(CCC)n3)n2)nc1CCC | 1.376 | RESIDUAL_SHAPE | 0.400 | 14 | 1 | 4.699 |
| 9 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N1CCCCC1 | 1.349 | RESIDUAL_SHAPE | 0.556 | 12 | 2 | 4.322 |
| 10 | CC(C)(C)Cc1cc(-c2cccc(-c3cc(CC(C)(C)C)[nH]n3)n2)n[nH]1 | 1.326 | RESIDUAL_SHAPE | 0.280 | 7 | 1 | 2.854 |
| 11 | CN(C(=O)COCC(=O)N(C)c1ccccc1)c1ccccc1 | 1.318 | RESIDUAL_SHAPE | 0.679 | 78 | 11 | 5.590 |
| 12 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N1CCOCC1 | 1.303 | RESIDUAL_SHAPE | 0.469 | 12 | 2 | 4.492 |
| 13 | CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC | 1.263 | RESIDUAL_SHAPE | 0.657 | 1,488 | 339 | 7.471 |
| 14 | CCCCOP(=S)(COCP(=S)(OCCCC)OCCCC)OCCCC | 1.260 | RESIDUAL_SHAPE | 0.667 | 6 | 1 | 3.133 |
| 15 | CCCCCCCCN(C)C(=O)C(CCOCCCCCC)C(=O)N(C)CCCCCCCC | 1.233 | RESIDUAL_SHAPE | 0.459 | 52 | 15 | 4.675 |

Class mix of this top-15: RESIDUAL_SHAPE 15.

(`true_span` = mean per-seed range of the ligand's measured log D; `n_curves` from `series/curve_table.parquet`; SMILES are the canonical extractant strings, full precision in the CSV.)

The turnover is the finding. At k=0 the worst list is a **level** list: 9/15 are PURE_LEVEL, ligands the model places 2-4 log units off and then tracks correctly. One measurement deletes them from the list entirely. At k=1 the worst list is 15/15 RESIDUAL_SHAPE and at k=2 it is 15/15. **The population of hard ligands is completely replaced by one measurement.**

## 6. What separates RESIDUAL_SHAPE from PURE_LEVEL

29 vs 25 ligands, spread over 72 held-out Tanimoto chemotypes. Ligands inside one chemotype were held out together against the same reduced training set, so every interval below resamples **chemotypes**, not ligands. Two statistics are reported: the difference in means, and P(RESIDUAL_SHAPE > PURE_LEVEL) as a rank statistic robust to the two or three enormous ligands. p-values are two-sided bootstrap p, BH-adjusted within each block of covariates.

### Donor chemistry: no.

| column | mean_a | mean_b | auc | auc_lo | auc_hi | auc_p | auc_q_bh |
|---|---|---|---|---|---|---|---|
| mech__hbd | 0.621 | 0.160 | 0.582 | 0.520 | 0.658 | 0.028 | 0.994 |
| mech__n_ether_O | 1.241 | 0.960 | 0.563 | 0.400 | 0.691 | 0.393 | 1.000 |
| mech__n_chelate_pairs | 3.414 | 3.600 | 0.532 | 0.385 | 0.653 | 0.624 | 1.000 |
| mech__n_O_donor | 3.586 | 2.960 | 0.536 | 0.359 | 0.658 | 0.627 | 1.000 |
| mech__tpsa | 71.610 | 62.882 | 0.534 | 0.354 | 0.656 | 0.654 | 1.000 |
| mech__n_thiophosphoryl_S | 0.138 | 0.080 | 0.514 | 0.450 | 0.596 | 0.708 | 1.000 |
| mech__n_S_donor | 0.138 | 0.080 | 0.514 | 0.450 | 0.596 | 0.708 | 1.000 |
| mech__n_amine_N | 0.069 | 0.040 | 0.514 | 0.448 | 0.592 | 0.708 | 1.000 |

All 36 mechanism descriptors were tested; the top 8 by raw p are shown. **Not one survives correction** -- the smallest BH q over the whole mechanism block is 0.994, and the smallest raw p is 0.028. Softness, denticity, donor counts, N/O/S composition, chelate geometry, charge, lipophilicity: none of them tells a level failure from a shape failure. Both classes are dominated by neutral O-donor diglycolamides drawn from the same chemotypes.

### Measurement geometry: yes, decisively.

| column | mean_a | mean_b | median_a | median_b | auc | auc_lo | auc_hi | auc_p | auc_q_bh |
|---|---|---|---|---|---|---|---|---|---|
| mad_true | 1.153 | 0.388 | 1.134 | 0.305 | 0.961 | 0.905 | 1.000 | 0.000 | 0.001 |
| log_d_span_max | 4.039 | 1.531 | 3.901 | 1.176 | 0.927 | 0.859 | 1.000 | 0.000 | 0.001 |
| max_curve_yspan | 3.703 | 1.385 | 3.521 | 1.080 | 0.914 | 0.839 | 1.000 | 0.000 | 0.001 |
| true_span | 4.296 | 1.579 | 4.202 | 1.176 | 0.938 | 0.880 | 1.000 | 0.000 | 0.001 |
| true_sd | 1.384 | 0.484 | 1.348 | 0.360 | 0.961 | 0.910 | 1.000 | 0.000 | 0.001 |
| median_curve_linear_r2 | 0.859 | 0.536 | 0.910 | 0.568 | 0.797 | 0.693 | 0.930 | 0.000 | 0.001 |
| abs_median_curve_slope | 1.183 | 0.418 | 1.039 | 0.059 | 0.789 | 0.682 | 0.917 | 0.000 | 0.001 |
| n_series | 3.862 | 1.280 | 2.000 | 1.000 | 0.735 | 0.628 | 0.837 | 0.000 | 0.001 |
| n_doi_series_sum | 5.759 | 1.280 | 2.000 | 1.000 | 0.736 | 0.628 | 0.837 | 0.000 | 0.001 |
| span_compression | 0.341 | 1.329 | 0.292 | 0.521 | 0.284 | 0.114 | 0.407 | 0.001 | 0.003 |
| has_axis_acid | 0.793 | 0.320 | 1.000 | 0.000 | 0.737 | 0.620 | 0.841 | 0.002 | 0.004 |
| n_curves | 21.103 | 4.400 | 2.000 | 1.000 | 0.710 | 0.575 | 0.806 | 0.008 | 0.015 |
| n_curve_axes | 1.759 | 1.360 | 1.000 | 1.000 | 0.603 | 0.507 | 0.730 | 0.036 | 0.063 |
| pred_span | 1.396 | 0.753 | 1.228 | 0.563 | 0.706 | 0.510 | 0.824 | 0.042 | 0.069 |
| n_distinct_doi | 3.897 | 2.000 | 2.000 | 2.000 | 0.690 | 0.500 | 0.811 | 0.054 | 0.083 |
| has_axis_metal_series | 0.483 | 0.720 | 0.000 | 1.000 | 0.381 | 0.250 | 0.516 | 0.081 | 0.116 |
| has_axis_extractant | 0.310 | 0.160 | 0.000 | 0.000 | 0.575 | 0.470 | 0.684 | 0.181 | 0.245 |
| log10_n_rows | 1.374 | 1.162 | 1.176 | 1.146 | 0.581 | 0.390 | 0.716 | 0.347 | 0.440 |
| log10_n_eval | 1.061 | 0.846 | 0.845 | 0.845 | 0.568 | 0.401 | 0.701 | 0.363 | 0.440 |
| has_axis_contact_time | 0.069 | 0.120 | 0.000 | 0.000 | 0.474 | 0.418 | 0.509 | 0.448 | 0.516 |
| has_axis_metal_concentration | 0.069 | 0.000 | 0.000 | 0.000 | 0.534 | 0.500 | 0.571 | 0.700 | 0.767 |
| nn_train_tanimoto | 0.553 | 0.577 | 0.556 | 0.615 | 0.463 | 0.275 | 0.630 | 0.748 | 0.782 |
| has_axis_temperature | 0.034 | 0.040 | 0.000 | 0.000 | 0.497 | 0.479 | 0.506 | 1.000 | 1.000 |

Everything that separates the two classes is about **how far the ligand's log D actually travels and over how many separate experiments**; nothing that separates them is about what the molecule is made of. Listed with the BH q so the one that does not clear correction is visible as such:

- `log_d_span_max` (widest log D span of any one series): 4.039 vs 1.531, AUC 0.927 [0.859, 1.000], BH q = 0.0007
- `true_span` (range of the ligand's measured log D): 4.296 vs 1.579, AUC 0.938 [0.880, 1.000], BH q = 0.0007
- `max_curve_yspan` (widest single titration curve): 3.703 vs 1.385, AUC 0.914 [0.839, 1.000], BH q = 0.0007
- `median_curve_linear_r2` (how cleanly its curves are linear): 0.859 vs 0.536, AUC 0.797 [0.693, 0.930], BH q = 0.0007
- `abs_median_curve_slope` (|median curve slope|): 1.183 vs 0.418, AUC 0.789 [0.682, 0.917], BH q = 0.0007
- `n_series` (independent measurement series): 3.862 vs 1.280, AUC 0.735 [0.628, 0.837], BH q = 0.0010
- `n_doi_series_sum` (series-weighted publication count): 5.759 vs 1.280, AUC 0.736 [0.628, 0.837], BH q = 0.0010
- `n_distinct_doi` (distinct source publications): 3.897 vs 2.000, AUC 0.690 [0.500, 0.811], BH q = 0.0828
- `has_axis_acid` (has an acid titration): 0.793 vs 0.320, AUC 0.737 [0.620, 0.841], BH q = 0.0042

**The top of that table is close to a restatement of the definition, and saying so is the difference between a caveat and an honest reading.** A ligand is RESIDUAL_SHAPE when a constant cannot remove its error, and `shape_mae` is what the error would be if the model predicted a constant for it. The model is nearly flat within a ligand -- `shape_mae` is a median 0.84 of the ligand's own mean absolute deviation of log D -- so a wide-spanning ligand is close to *being* a shape failure rather than *explaining* one. The benchmark row is `mad_true`, the spread of the target with the model deleted entirely: AUC 0.961 [0.905, 1.000]. `true_sd` reaches 0.961 and `true_span` 0.938; none of them beats knowing nothing but the target's own spread. Read those rows as *where* the defect bites, not as an independent cause of it. The rows that are not definitional are the ones counting experiments -- `n_series`, `n_curves`, `has_axis_acid` -- and they are weaker, which is the honest shape of this result.

One label correction, since the two versions do not say the same thing. `series_table.n_doi` is a per-series count; summing it over a ligand counts a publication once per series it appears in, and for 124 of 143 ligands that sum is not the number of source publications. Counting distinct DOIs instead moves the contrast from AUC 0.736 (q = 0.0010) to 0.690 [0.500, 0.811] (q = 0.083). The weaker number is the one that answers "is this ligand studied in many papers". And the summed version is Spearman 0.96 with `n_series`, so the two are one variable and not two pieces of evidence.

Two negatives matter as much as the positives.

- **`nn_train_tanimoto` does not separate them**: 0.553 vs 0.577, AUC 0.463 [0.275, 0.630], p = 0.748. Distance to the nearest training ligand predicts *whether* a held-out ligand is hard, but not *which kind* of hard it is. This is consistent with the gen8 result that mechanism-aware similarity does not fix the zero-shot level.
- **Sheer row count does not separate them either**: log10 n_rows 1.374 vs 1.162, AUC 0.581 [0.390, 0.716], p = 0.347; the medians are 15 and 14 rows. The means differ only because a handful of RESIDUAL_SHAPE ligands are the most-studied extractants in the corpus. It is the **span**, not the count.

### The same contrast against ALREADY_GOOD

Repeating the whole contrast against the 82 ALREADY_GOOD ligands (`failure_class_contrasts_vs_already_good.csv`) gives the same ordering: `true_span` AUC 0.891 [0.837, 0.957] (BH q = 0.0008), `nn_train_tanimoto` AUC 0.458 [0.338, 0.600] (q = 0.576). One difference is worth recording: row count *does* separate RESIDUAL_SHAPE from ALREADY_GOOD (log10 n_rows AUC 0.662, q = 0.032) while it does not separate it from PURE_LEVEL. Being heavily measured makes a ligand more likely to be a failure at all; it does not decide which kind.

### Which curve axes

| failure_class | has_axis_acid | has_axis_extractant | has_axis_metal_series | has_axis_temperature | has_axis_contact_time | has_axis_metal_concentration |
|---|---|---|---|---|---|---|
| PURE_LEVEL | 0.320 | 0.160 | 0.720 | 0.040 | 0.120 | 0.000 |
| PARTIAL_LEVEL | 0.714 | 0.286 | 0.429 | 0.143 | 0.143 | 0.000 |
| RESIDUAL_SHAPE | 0.793 | 0.310 | 0.483 | 0.034 | 0.069 | 0.069 |
| ALREADY_GOOD | 0.439 | 0.134 | 0.622 | 0.037 | 0.024 | 0.024 |

Fraction of ligands with at least one curve on each axis. The acid axis is the discriminating one (0.79 of RESIDUAL_SHAPE vs 0.32 of PURE_LEVEL, AUC 0.737 [0.620, 0.841], BH q = 0.0042). The lanthanide-series axis runs the other way and does **not** clear correction (0.48 vs 0.72, AUC 0.381 [0.250, 0.516], BH q = 0.116) -- a ligand whose data is one lanthanide scan at fixed conditions is a level problem; a ligand with an acid titration is a shape problem.

### Can any of this be used in advance? Out-of-chemotype scoring

Every AUC above is an in-sample description of one 54-ligand split. Whether a covariate would let you *sort a new ligand* is a different question, and it has to be answered the way the model itself is scored: fit on all chemotypes but one, predict the held-out chemotype, repeat. Logistic regression, standardised inputs, RESIDUAL_SHAPE as the positive class.

| features | n_scored | loco_auc |
|---|---|---|
| target-free set (n_series, n_curves, n_curve_axes, has_axis_acid, log10_n_rows, nn_train_tanimoto) | 54 | 0.745 |
| has_axis_acid alone | 54 | 0.633 |
| n_series alone | 54 | 0.592 |
| n_curves alone | 54 | 0.369 |
| donor chemistry (36 mechanism descriptors) | 54 | 0.286 |
| true_span alone (target-derived, for scale) | 54 | 0.924 |

The target-free set holds up at 0.745 out of sample; no single target-free variable does, which is why the useful form of this is a rule over the campaign as a whole rather than one number. Donor chemistry is at 0.286 -- below chance, which for 36 descriptors on 54 ligands is what over-fitting nothing looks like. The target-derived `true_span` row (0.924) is there only as the scale: it is not available in advance and, per the definitional caveat above, it is close to reading the answer.

### Blocked check inside mixed chemotypes

The cluster bootstrap already respects the block structure, but only 4 chemotypes (tan049, tan054, tan055, tan078) contain both classes, so the strictly within-block contrast is reported separately as a weak but assumption-free check:

| column | n_blocks | mean_within_block_delta | blocks_positive |
|---|---|---|---|
| log_d_span_max | 4 | 1.707 | 4 |
| true_span | 4 | 2.150 | 4 |
| n_series | 4 | 2.818 | 4 |
| nn_train_tanimoto | 4 | 0.056 | 3 |
| abs_median_curve_slope | 4 | 0.399 | 3 |
| mech__softness_mean | 4 | -0.028 | 0 |
| mech__n_donor_total | 4 | 0.637 | 3 |

With 4 blocks this is not a test, and it is not offered as one. It agrees in direction with the pooled result on the span and series variables and is flat on similarity and donor chemistry.

## 7. Why the shape failures look the way they do

Joining the frozen model's own out-of-fold predictions gives the mechanism directly. Per ligand, averaged over seeds:

| failure_class | true_span | pred_span | true_sd | pred_sd | span_compression |
|---|---|---|---|---|---|
| PURE_LEVEL | 1.176 | 0.563 | 0.360 | 0.198 | 0.521 |
| PARTIAL_LEVEL | 3.086 | 0.664 | 0.935 | 0.198 | 0.236 |
| RESIDUAL_SHAPE | 4.202 | 1.228 | 1.348 | 0.389 | 0.292 |
| ALREADY_GOOD | 1.292 | 0.507 | 0.455 | 0.189 | 0.488 |

(Medians. `span_compression` = pred_span / true_span, undefined for the 6 ligands with a single distinct log D.)

RESIDUAL_SHAPE ligands' log D really moves 4.20 decades; the model moves its prediction 1.23. PURE_LEVEL ligands really move 1.18 and the model moves 0.56.

Two things are true at once here and the second one is easy to miss. The model does respond *more* for the RESIDUAL_SHAPE ligands in absolute terms (pred_span 1.23 vs 0.56), but it responds far less *proportionally*: it captures 29% of the true range against 52% for PURE_LEVEL, and that difference survives the chemotype bootstrap (AUC 0.284 [0.114, 0.407], BH q = 0.003). So the shape failures are not merely wider-ranging ligands hit by a uniform flattening -- they are *both* wider-ranging *and* flattened harder. Underlying both is the already-established gen8 slope result (median true d(logD)/d(log10[extractant]) 2.31 against 0.075 predicted).

The consequence for calibration is the same either way. A ligand measured at one condition has almost no shape to get wrong, so an offset repairs it completely; a ligand with a 4-decade acid titration has roughly 3 decades of unmodelled response that no constant can absorb, and adding a measurement only moves the constant.

So the two classes are not two chemistries. They are the same modelling defect measured where it costs nothing and where it costs everything.

## 8. What remains after two measurements

| | k = 0 | k = 1 (CENTRAL) | k = 2 (CENTRAL) |
|---|---|---|---|
| macro MAE, all 143 ligands | 0.995 | 0.651 | 0.619 |
| PURE_LEVEL (n=25) | 1.906 | 0.451 | 0.425 |
| PARTIAL_LEVEL (n=7) | 1.458 | 0.671 | 0.622 |
| RESIDUAL_SHAPE (n=29) | 1.350 | 1.308 | 1.251 |
| ALREADY_GOOD (n=82) | 0.552 | 0.477 | 0.454 |
| RESIDUAL_SHAPE share of macro | 0.28 | 0.41 | 0.41 |

**29 of 143 ligands (20%) carry 41% of the error that survives two measurements**, up from 28% at k=0. Their remaining 1.251 sits against a shape floor of 0.962 that offset calibration cannot cross at any k, with any policy, or with an oracle. That 41% is stated at the level cut used throughout; it is 60% at a 0.75 cut and 11% at a 1.5 cut, so it should be read with the line stated, not quoted bare.

The same conclusion has a form that needs no classification at all. Macro at k=2 (CENTRAL) is 0.619; the macro `shape_mae` -- the part of the error offset calibration does not touch at any k, for any ligand (and, per §2, an upper bound on the true offset-only floor) -- is 0.472. **76% of what survives two measurements is shape, not level**, and that number involves no threshold, no class and no cut-off.

Three consequences for gen9, stated as what the numbers do and do not license:

1. **More measurements per ligand is the wrong axis for these 29.** The gap between the deployable 2-shot arm (1.251) and `shape_mae` (0.962) is 0.289; the gap between that and zero is 0.962. Perfecting acquisition buys the small number. Only a model that bends the curve buys the large one.
2. **The target is the response function, not the level.** The three gen8 shape results now line up: the model predicts titration curves an order of magnitude too flat, cross-series transfer fails between curve types, and the ligands an offset cannot fix are exactly the wide-span, acid-titrated, multi-series ones. A slope-aware adapter -- or a model with the mass-action functional form built in rather than fitted -- is the intervention this analysis points at. Read off this same run, `OFFSET_K3` on two random measurements is 0.598 macro against `OFFSET_K1`'s 0.640: the first evidence in that direction, not tested per class here, and not claimed by this report.
3. **A triage exists before any measurement, but it is weaker than the in-sample AUCs suggest, and it is about the campaign rather than the molecule.** Scored the way a prediction has to be -- leave-one-chemotype-out, on the 54 ligands of the two classes -- the whole target-free set reaches AUC 0.745, while singly the same variables reach only 0.369-0.633; the marginal in-sample AUCs of §6 (0.46-0.74) are descriptions of this split, not out-of-sample performance. Donor chemistry scores 0.286, i.e. worse than a coin. A gen9 triage of the form "if this ligand's campaign is a multi-decade acid titration, do not expect a calibration point to help" is supported at that strength; a rule keyed on donor type is not supported at all.

## 9. Caveats

- `mad_true`, `log_d_span_max`, `true_span`, `true_sd`, `max_curve_yspan`, `median_curve_linear_r2` and the curve slopes are computed **from the targets**, and (per §6) the largest of them are near-restatements of the class definition rather than independent explanations of it. They are not features, were not used to build one, and must not be quoted as predictors. The target-free separators are `n_series`, `n_curves`, `n_curve_axes` and `has_axis_acid`.
- `n_distinct_doi` is target-free but not knowable for a molecule that has never been published, so it is excluded from the deployable triage set even though it is reported in §6.
- `shape_frac_k0` likewise uses targets, and its Spearman against the 1-shot ratio is high partly by construction: `OFFSET_K1` cannot change `shape_mae`, so the two quantities share a term. It is the descriptive decomposition, not an independent confirmation and not a predictor.
- The classes are read off the same units they then describe. §4(e) re-derives them on three split seeds and scores them on the other two; the seeds share the ligands' rows, so that tests the protocol's sampling noise and not the noise in the measurements themselves.
- The cut-offs (1.00 on the level, 0.50/0.70 on the ratio) were fixed before the contrasts were run but are not derived from anything; PARTIAL_LEVEL was added after seeing that the band between them was occupied. §4 is a sensitivity analysis, not a claim that the cut-offs are canonical.
- All 143 ligands have rows in `series_table.parquet` and `curve_table.parquet`, so no class contrast loses members to missing covariates. The one exception is `span_compression`, undefined for the 6 ligands with a single distinct log D value -- all 6 are ALREADY_GOOD, so the RESIDUAL_SHAPE / PURE_LEVEL contrast is unaffected.
- PARTIAL_LEVEL has 7 members and its per-class intervals are wide; no claim in this report rests on it.
- The within-chemotype blocked contrast has 4 blocks. It is a direction check, not a test.
- Everything is conditional on one frozen global model and the P2 protocol's half-pool/half-evaluation split. A ligand with 5 rows contributes an evaluation set of 2 or 3, so its per-ligand MAE is noisier than a ligand with 200; macro weights them equally by design.

