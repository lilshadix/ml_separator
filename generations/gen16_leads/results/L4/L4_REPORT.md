# L4 — which chemotypes to measure next

*Discovery phase.  Five frozen discovery seeds `(104729, 130363, 155921, 196613, 262147)`, all five
hold-out designs, extractant-macro MAE of pairwise `log SF` through `gen13sep.metrics.per_extractant`
/ `summarise`, chemotype-blocked paired bootstrap `gen13sep.inference.paired_contrasts`
(10 000 replicates, seed 8675309, margin 0.02).  Unit of scoring = extractant (n = 90), unit of
resampling = frozen chemotype (45 blocks).  Protocol: `PRE_REGISTRATION.md` §3 L4.*

---

## 0. Headline

**The registered claim fails.**  A greedy A-optimal acquisition order (`AOPT`) makes BP macro MAE
fall faster with corpus size than random order — ABC = **+0.0501** [+0.0272, +0.0662], p < 1e-4,
5/5 seeds, LOCO-stable, passing P1 under BP — but the sign is **not consistent across the five
designs**: it is **−0.0119** under B and **−0.0034** under BQ.  Under the programme's shared rule
(§2.1 of the pre-registration: "a candidate is supported only if the sign of its effect is the same
in all five") `AOPT` is **not supported**.  It passes P1 in 3 of 5 designs (BR, A, BP).

**And where it does appear, an exploratory size-matched control removes it.**  A chemotype budget is
not a cell budget.  At k = 6 chemotypes under BP, `AOPT`'s training set holds **303 cells** against
`RANDOM`'s **79** and `MAXMIN`'s **19**.  When `RANDOM` is interpolated to `AOPT`'s own training-cell
count, `AOPT_vs_RANDOM_CELLMATCHED` is +0.0108 (BP), +0.0017 (A), −0.0058 (BR), −0.0272 (BQ),
−0.0369 (B) — **nothing passes P1, and three of five are negative**.  The registered gain is a
training-set-size effect wearing a design's clothes.

`UNCERT` is dead on every endpoint and every design (best +0.0035, worst −0.0310; under design A it
is significantly *worse* than random, −0.0180, p = 0.036, 0/5 seeds positive).

The prospective ranking is delivered, but it carries a defect the retrospective result predicts:
run forward on the full cohort, the A-optimal criterion recommends **more diglycolamides**, not new
chemistry.  The top 20 of the bundle pool contains **2 distinct chemotypes**; the 53 chemotypes L6
identified as absent from the cohort first appear at rank 23.

**One exploratory signal worth a future pre-registration, and nothing more:** at matched training-cell
count, `MAXMIN` (ECFP4 farthest-point diversity) beats random in **all five** designs
(+0.0283 B, +0.0155 BR, +0.0195 BQ, +0.0330 A, +0.0247 BP), passing P1 under B and A.  This is
`family = exploratory`, it was not registered, and it is **not promoted**.

---

## 1. What was run

| | |
|---|---|
| designs | B, BR, BQ, A, BP (all five, no subset) |
| folds | 25 per design (5 discovery seeds × 5 folds), `gen13sep.splits.all_folds`, defaults only |
| budgets | k ∈ {6, 9, 12, 16, 20, 24} training chemotypes, plus the full corpus |
| orders | `RANDOM` (20 draws/fold), `MAXMIN` (20 starts/fold), `AOPT`, `UNCERT` |
| arm refitted | `gen15.arms.g14` on the thinned training set, `w` recomputed with `cell_weights` |
| test set | **never touched** — every budget of every order scores byte-identical held-out pairs |
| folds skipped | **0** (every fold had ≥ 24 training chemotypes, so k_actual = k everywhere) |
| G14 anchor through this code path | BP `0.5000794414` vs frozen `0.5000794414` — exact |
| fits | 25 folds × 42 (order, draw) × 6 budgets × 5 designs = 31 500, plus 125 full-corpus refits |
| wall clock | 3 863 s, one process, `n_jobs`/`OMP_NUM_THREADS` = 2, peak ≈ 480 MB |

**Guard and its asymmetry.**  A thinned fit with < 10 well-determined training cells or one direction
class falls back to `FLAT`.  Totals over all designs and budgets: `MAXMIN` 2 100 fallbacks (1 706 of
them at k = 6), `RANDOM` 654, `AOPT` 0, `UNCERT` 0.  The guard therefore *protects* the comparators
at k = 6 on MAE (`FLAT` = 0.5885 beats a degenerate fit) and *penalises* them on sign accuracy
(`FLAT` predicts zero, so its strong-pair sign accuracy is 0 by construction).  Any sign-accuracy
number at k = 6 or 9 must be read with this in mind; it is one reason the direction-accuracy endpoint
is exploratory rather than registered.

---

## 2. Learning curves (extractant-macro MAE, mean over draws)

`FULL` = the unthinned G14 refit through this code path; `FLAT` = predict no separation.
`draw_sd` is the spread over the 20 draws of the stochastic orders.

### BP (the selecting design)

| order | k=6 | k=9 | k=12 | k=16 | k=20 | k=24 | full |
|---|---|---|---|---|---|---|---|
| `AOPT` | 0.5698 | 0.5230 | 0.5090 | **0.4923** | 0.5033 | 0.5026 | — |
| `UNCERT` | 0.6630 | 0.6364 | 0.6059 | 0.5236 | 0.5200 | 0.5169 | — |
| `MAXMIN` | 0.5989 | 0.6126 | 0.6086 | 0.5909 | 0.5262 | 0.5121 | — |
| `RANDOM` | 0.6264 | 0.6082 | 0.5781 | 0.5525 | 0.5242 | 0.5110 | — |
| `RANDOM` draw_sd | 0.0221 | 0.0272 | 0.0251 | 0.0231 | 0.0121 | 0.0072 | — |
| `FULL` | | | | | | | **0.5001** |
| `FLAT` | | | | | | | 0.5885 |

Two facts sit inside this table.  First, **a randomly chosen 6-chemotype corpus produces a model that
is worse than predicting nothing** (0.6264 against `FLAT`'s 0.5885), and it stays worse than `FLAT`
until about 12 chemotypes.  Second, the curve is **flat from k ≈ 16 to the full corpus** for every
order: `AOPT` reaches 0.4923 at k = 16 and the full 29.4-chemotype corpus scores 0.5001.  On this
evidence the corpus is *not* chemotype-starved at its own margin — the marginal chemotype, added in
any order, buys nothing measurable.  (The mean training-chemotype count under BP is 29.36, not 45,
because BP masks publications as well as holding out chemotypes.)

### All five designs

| design | order | k=6 | k=9 | k=12 | k=16 | k=20 | k=24 | FULL | FLAT |
|---|---|---|---|---|---|---|---|---|---|
| A | `AOPT` | 0.5332 | 0.5099 | 0.5113 | 0.5061 | 0.4938 | 0.4987 | 0.4921 | 0.5885 |
| A | `UNCERT` | 0.6111 | 0.6117 | 0.5718 | 0.5445 | 0.5245 | 0.5180 | | |
| A | `MAXMIN` | 0.5910 | 0.5890 | 0.5716 | 0.5567 | 0.5381 | 0.5247 | | |
| A | `RANDOM` | 0.5967 | 0.5661 | 0.5495 | 0.5310 | 0.5198 | 0.5104 | | |
| B | `AOPT` | 0.6195 | 0.6048 | 0.5709 | 0.5576 | 0.5196 | 0.5028 | 0.4932 | 0.5885 |
| B | `UNCERT` | 0.6360 | 0.6420 | 0.6138 | 0.5598 | 0.5309 | 0.5073 | | |
| B | `MAXMIN` | 0.5869 | 0.5749 | 0.5799 | 0.5470 | 0.5283 | 0.5350 | | |
| B | `RANDOM` | 0.6020 | 0.5746 | 0.5523 | 0.5332 | 0.5272 | 0.5143 | | |
| BQ | `AOPT` | 0.5891 | 0.5655 | 0.5824 | 0.5459 | 0.5150 | 0.5249 | 0.4905 | 0.5885 |
| BQ | `UNCERT` | 0.5780 | 0.5741 | 0.5528 | 0.5509 | 0.5221 | 0.5038 | | |
| BQ | `MAXMIN` | 0.5868 | 0.5922 | 0.5915 | 0.5544 | 0.5296 | 0.5299 | | |
| BQ | `RANDOM` | 0.6042 | 0.5817 | 0.5539 | 0.5315 | 0.5228 | 0.5084 | | |
| BR | `AOPT` | 0.5367 | 0.5441 | 0.5157 | 0.5407 | 0.5026 | 0.4910 | 0.4906 | 0.5885 |
| BR | `UNCERT` | 0.5957 | 0.6018 | 0.5674 | 0.5164 | 0.5133 | 0.5016 | | |
| BR | `MAXMIN` | 0.5854 | 0.5893 | 0.5741 | 0.5585 | 0.5476 | 0.5263 | | |
| BR | `RANDOM` | 0.6016 | 0.5741 | 0.5495 | 0.5307 | 0.5124 | 0.5033 | | |

Full table with `macro_mae_far`, `macro_sign_acc`, `macro_pair_spearman`, `n_draws`, fallback counts
and `n_units`: `curve.csv`; the same rows keyed as `arm = order@k<budget>`: `board.csv`.

---

## 3. Registered contrasts (ABC on `mae_all`), P1 verdicts

`ABC = mean over the six budgets of [MAE_RANDOM(k) − MAE_ORDER(k)]` per extractant, `RANDOM`
averaged over its 20 draws; positive favours the candidate order.  n = 90 extractants, 45 chemotype
blocks, 5 seeds.  BH adjusted within L4's own registered family of 20; the orchestrator re-adjusts
fleet-wide.

| design | comparison | point | 95 % CI | p | BH p | seeds + | LOCO stable | **P1** |
|---|---|---|---|---|---|---|---|---|
| B | `AOPT_vs_RANDOM` | **−0.0119** | [−0.0552, +0.0367] | 0.783 | 0.943 | 2/5 | no | ✗ |
| BR | `AOPT_vs_RANDOM` | +0.0235 | [+0.0135, +0.0359] | <1e-4 | <1e-4 | 5/5 | yes | ✓ |
| BQ | `AOPT_vs_RANDOM` | **−0.0034** | [−0.0375, +0.0355] | 0.974 | 0.985 | 3/5 | no | ✗ |
| A | `AOPT_vs_RANDOM` | +0.0367 | [+0.0204, +0.0504] | <1e-4 | <1e-4 | 5/5 | yes | ✓ |
| BP | `AOPT_vs_RANDOM` | +0.0501 | [+0.0272, +0.0662] | <1e-4 | <1e-4 | 5/5 | yes | ✓ |
| B | `AOPT_vs_MAXMIN` | **−0.0039** | [−0.0407, +0.0453] | 0.985 | 0.985 | 3/5 | no | ✗ |
| BR | `AOPT_vs_MAXMIN` | +0.0417 | [+0.0148, +0.0601] | 0.0014 | 0.0047 | 5/5 | yes | ✓ |
| BQ | `AOPT_vs_MAXMIN` | +0.0103 | [−0.0136, +0.0439] | 0.434 | 0.764 | 4/5 | yes | ✗ |
| A | `AOPT_vs_MAXMIN` | +0.0530 | [+0.0188, +0.0770] | 0.0008 | 0.0032 | 5/5 | yes | ✓ |
| BP | `AOPT_vs_MAXMIN` | +0.0582 | [+0.0151, +0.0920] | 0.0002 | 0.0010 | 5/5 | yes | ✓ |
| B | `UNCERT_vs_RANDOM` | −0.0310 | [−0.0672, +0.0190] | 0.282 | 0.704 | 2/5 | yes | ✗ |
| BR | `UNCERT_vs_RANDOM` | −0.0041 | [−0.0240, +0.0225] | 0.711 | 0.943 | 3/5 | no | ✗ |
| BQ | `UNCERT_vs_RANDOM` | +0.0035 | [−0.0298, +0.0275] | 0.803 | 0.943 | 4/5 | no | ✗ |
| A | `UNCERT_vs_RANDOM` | **−0.0180** | [−0.0367, −0.0014] | 0.036 | 0.102 | 0/5 | yes | ✗ |
| BP | `UNCERT_vs_RANDOM` | −0.0109 | [−0.0408, +0.0318] | 0.650 | 0.943 | 2/5 | no | ✗ |
| B | `UNCERT_vs_MAXMIN` | −0.0230 | [−0.0559, +0.0222] | 0.339 | 0.720 | 2/5 | yes | ✗ |
| BR | `UNCERT_vs_MAXMIN` | +0.0142 | [−0.0195, +0.0389] | 0.360 | 0.720 | 3/5 | yes | ✗ |
| BQ | `UNCERT_vs_MAXMIN` | +0.0171 | [−0.0317, +0.0470] | 0.458 | 0.764 | 4/5 | no | ✗ |
| A | `UNCERT_vs_MAXMIN` | −0.0017 | [−0.0385, +0.0217] | 0.837 | 0.943 | 2/5 | no | ✗ |
| BP | `UNCERT_vs_MAXMIN` | −0.0028 | [−0.0279, +0.0248] | 0.849 | 0.943 | 3/5 | no | ✗ |

**Sign patterns across (B, BR, BQ, A, BP):** `AOPT_vs_RANDOM` = − + − + + ; `AOPT_vs_MAXMIN` =
− + + + + ; `UNCERT_vs_RANDOM` = − − + − − ; `UNCERT_vs_MAXMIN` = − + + − −.  **No registered
contrast has a consistent sign in all five designs, and none passes P1 in all five.**

BCa intervals are in `contrasts_abc.csv` alongside the percentile intervals; they agree on every
verdict above.

### Permutation-style percentile

Percentile of `AOPT`'s macro ABC among the 20 random draws' own ABCs (`percentile_random.csv`):

| design | `AOPT` ABC | percentile | random-draw ABC range | random-draw sd |
|---|---|---|---|---|
| **BP** | **+0.0501** | **100.0** | [−0.0318, +0.0209] | 0.0131 |
| A | +0.0367 | 100.0 | [−0.0137, +0.0129] | 0.0083 |
| BR | +0.0235 | 100.0 | [−0.0173, +0.0187] | 0.0110 |
| BQ | −0.0034 | 35.0 | [−0.0248, +0.0202] | 0.0104 |
| B | −0.0119 | 10.0 | [−0.0157, +0.0225] | 0.0112 |

**Read this narrowly.**  Each random draw's ABC is measured against the mean of all 20 draws, so the
draw distribution is centred at zero by construction.  The percentile therefore tests only whether
`AOPT` beats draw-to-draw noise in random ordering — it is not an independent significance test, and
it says nothing about the training-set-size confound in §4.

---

## 4. Why it fails: a chemotype budget is not a cell budget (exploratory)

The A-optimal criterion adds the chemotype that most reduces the trace of the posterior variance over
the whole training pool.  A chemotype with many well-determined cells contributes many `x xᵀ` terms,
so the criterion is *mechanically* drawn to the largest chemotypes.  Mean training cells per fold:

| design | order | k=6 | k=9 | k=12 | k=16 | k=20 | k=24 |
|---|---|---|---|---|---|---|---|
| BP | `AOPT` | **303.2** | 311.0 | 331.7 | 352.7 | 362.6 | 369.2 |
| BP | `UNCERT` | 289.5 | 298.9 | 314.1 | 346.2 | 360.1 | 370.7 |
| BP | `RANDOM` | 78.8 | 120.2 | 158.1 | 209.8 | 258.7 | 303.9 |
| BP | `MAXMIN` | **19.4** | 25.5 | 31.5 | 49.3 | 69.8 | 164.9 |
| B | `AOPT` | 324.5 | 329.5 | 339.8 | 363.3 | 381.4 | 397.1 |
| B | `RANDOM` | 65.5 | 102.6 | 134.4 | 189.4 | 221.3 | 266.2 |
| B | `MAXMIN` | 18.0 | 23.5 | 28.8 | 38.2 | 55.6 | 75.9 |

`AOPT` at **6** chemotypes already holds more training cells than `RANDOM` at **24**.  Full table for
all five designs, with well-determined-cell and extractant counts: `cells_by_budget.csv`.

**Cell-matched contrast.**  For each (extractant, seed, budget) the `RANDOM` MAE is linearly
interpolated, in training-cell count, to the candidate order's own training-cell count, and the same
ABC and the same bootstrap are applied (`contrasts_cellmatched.csv`, `family = exploratory`):

| design | `AOPT_vs_RANDOM_CELLMATCHED` | p | seeds + | P1 | `MAXMIN_vs_RANDOM_CELLMATCHED` | p | P1 |
|---|---|---|---|---|---|---|---|
| B | −0.0369 | 0.137 | 1/5 | ✗ | **+0.0283** | 0.0030 | ✓ |
| BR | −0.0058 | 0.155 | 2/5 | ✗ | +0.0155 | 0.192 | ✗ |
| BQ | −0.0272 | 0.117 | 1/5 | ✗ | +0.0195 | 0.056 | ✗ |
| A | +0.0017 | 0.766 | 3/5 | ✗ | **+0.0330** | <1e-4 | ✓ |
| BP | +0.0108 | 0.601 | 4/5 | ✗ | +0.0247 | 0.108 | ✗ |

(`UNCERT_vs_RANDOM_CELLMATCHED`: −0.0487 B, −0.0186 BR, −0.0097 BQ, −0.0356 A, −0.0469 BP.)

The same conclusion is visible without any interpolation, in the registered per-budget contrasts
(`contrasts_per_budget.csv`, exploratory).  Under BP, `AOPT_vs_RANDOM` on `mae_all` decays exactly as
the cell-count gap closes: **+0.0566** (k=6) → +0.0852 (k=9) → +0.0691 (k=12) → +0.0602 (k=16) →
+0.0208 (k=20) → **+0.0085** (k=24, no longer passing P1).

**Verdict.**  What the retrospective simulation actually measured is that *a model trained on 300
cells beats a model trained on 79*.  That is true, and it is not news.  It is not evidence that a
model-chosen chemotype ordering is worth anything.

### Diglycolamide check (exploratory)

The largest chemotype `sc009` holds 375 of 521 cells and 23 of 90 extractants.  Dropping those 23
extractants from the **scoring units** (training sets unchanged) makes `AOPT_vs_RANDOM_noDGA` pass P1
in all five designs (+0.0241 B, +0.0255 BR, +0.0241 BQ, +0.0345 A, +0.0380 BP) —
i.e. the effect is not *carried* by the diglycolamides, it is *masked* by them.  Keeping only the 23
DGA extractants gives the mirror image.  **This does not rescue the registered claim.**  The subset
changes who is scored, not who is trained on; the size confound of §4 applies unchanged, and a
post-hoc subset cannot promote a contrast that failed as registered.  `contrasts_nodga.csv`.

### Secondary endpoint: direction accuracy (exploratory)

ABC on strong-pair macro sign accuracy, `AOPT_vs_RANDOM`: +0.0098 (B, p 0.71), +0.0726 (BR),
+0.0301 (BQ, p 0.26), +0.0834 (A), +0.1019 (BP).  Same five-design inconsistency, and additionally
contaminated at k = 6/9 by the `FLAT` fallback, whose sign accuracy is 0 by construction and which
`RANDOM`/`MAXMIN` trigger while `AOPT` never does.  `contrasts_abc.csv`, `endpoint = ABC_sign_acc_strong`.

---

## 5. What `AOPT` and `MAXMIN` actually pick

Modal first six picks per design (`first_picks.csv`):

| design | order | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| BP | `AOPT` | sc061 | sc009 | sc009 | sc055 | sc055 | sc058 |
| BP | `MAXMIN` | sc092 | sc015 | sc092 | sc092 | sc086 | sc060 |
| B | `AOPT` | sc061 | sc030 | sc036 | sc036 | sc055 | sc058 |
| B | `MAXMIN` | sc085 | sc000 | sc092 | sc092 | sc086 | sc086 |
| A | `AOPT` | sc061 | sc030 | sc036 | sc019 | sc055 | sc009 |

`AOPT` reaches the diglycolamides (`sc009`) by its second pick under BP.  `MAXMIN` picks the
ECFP4 outliers first, which is why it triggers 1 706 of its 2 100 `FLAT` fallbacks at k = 6.  Full
per-fold orders: `orders.parquet` (design, split_seed, fold, order, draw, position, chemotype).

---

## 6. Prospective ranking (deliverable, not scored)

Criterion: standardise TOPO39 with the cohort's own median-imputed statistics; the already-chosen
design is the cohort's **289 well-determined cells**, `P = 1·I + Σ z zᵀ`; a candidate contributes one
row and the score is the Sherman–Morrison drop in the pool-averaged posterior variance of the linear
predictor.  Averaging over the pool alone and over cohort + pool give **the same ranking**
(Spearman 1.000 for the bundle pool, 0.9999 for the logK pool), so that choice carries no weight.

Full-cohort G14: mean |amplitude| 0.2950, mean curvature −0.0788.  Residual covariance from the 289
well-determined residual curves (`gen15.fewshot.residual_covariance`, shrink 0.25); the first three
D-optimal pairs over all 91 metal pairs are **La–Er, Nd–Lu, Ce–Gd** — global by construction, since a
candidate with no measured pair offers the same 91 choices to every candidate.  For pool (i), where
one lanthanide is already measured, the D-optimal triple restricted to pairs containing that metal is
given per row (`dopt3_pairs_with_measured`); for the 95 Eu-measured compounds it is **La–Eu, Eu–Lu,
Ce–Eu**.

### Pool (i) — bundle extractants with exactly one measured lanthanide

100 such compounds (matching L6's audit), of which **95** have a TOPO39 row in the frozen
`gen12_2_eu_pred/features/coordination_descriptors.parquet` and **5 are excluded** because no row
exists for them (all five are long-chain dialkyl diglycolamides; SMILES listed in
`ranking_notes.json`).  Every one of the 95 has Eu as its single measured lanthanide.

| rank | name | chemotype | in cohort | NN Tanimoto | nearest cohort ligand | A-opt reduction | P(heavy) | next pairs |
|---|---|---|---|---|---|---|---|---|
| 1 | CA12 | sc072 | no | 0.405 | TEDGA | 12.99 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |
| 2 | HOEtBenDGA | sc009 | yes | 0.600 | DODDdDGA | 11.75 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |
| 3 | L1f | sc009 | yes | 0.409 | TIBDGA | 11.61 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |
| 4 | L5b | sc009 | yes | 0.727 | DODDdDGA | 11.61 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |
| 5 | L1e | sc009 | yes | 0.571 | DODDdDGA | 11.61 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |
| 6 | L1d | sc009 | yes | 0.581 | DMDODGA | 11.61 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |
| 7 | L4 | sc009 | yes | 0.658 | DMDODGA | 11.61 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |
| 8 | L1c | sc009 | yes | 0.550 | TBDGA | 11.61 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |
| 9 | L5a | sc009 | yes | 0.688 | TBDGA | 11.61 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |
| 10 | L6b | sc009 | yes | 0.727 | DODDdDGA | 11.19 | 1.000 | La–Eu, Eu–Lu, Ce–Eu |

**The top 20 contains two distinct chemotypes.**  Because that is close to useless to a laboratory,
the same list collapsed to one row per chemotype is given here — this is a *presentation* of the same
unfiltered ranking, not a re-ranking:

| rank | name | chemotype | in cohort | NN Tanimoto | nearest cohort ligand | A-opt | P(heavy) |
|---|---|---|---|---|---|---|---|
| 1 | CA12 | sc072 | no | 0.405 | TEDGA | 12.99 | 1.000 |
| 2 | HOEtBenDGA | sc009 | yes | 0.600 | DODDdDGA | 11.75 | 1.000 |
| 23 | T8-THP-TAM | sc049 | no | 0.400 | N,N,N,N-tetraoctylpropanediamide | 4.98 | 0.000 |
| 24 | TWE-9 | sc039 | no | 0.571 | NTAamide(C8) | 4.79 | 0.000 |
| 25 | T8-THP-CAM | sc048 | no | 0.538 | tetraoctyl-phenanthroline-2,9-dicarboxamide | 4.75 | 0.000 |
| 26 | DO-PyranDGA | sc022 | no | 0.300 | N,N,N,N-tetraoctylpropanediamide | 4.75 | 0.000 |
| 27 | DPhen-PyranDGA | sc004 | no | 0.174 | N,N-dimethyl-N,N-diphenylpropanediamide | 4.75 | 0.000 |
| 28 | TWE-16 | sc047 | no | 0.538 | tetraoctyl-phenanthroline-2,9-dicarboxamide | 4.75 | 0.000 |
| 29 | Cy5S-Me4-BTBP | sc014 | no | 0.611 | CyMe4-BTBP † | 2.68 | 0.994 |
| 30 | Cy5-O-Me4-BTBP | sc013 | no | 0.611 | CyMe4-BTBP † | 2.68 | 0.994 |
| 31 | EsPyTri | sc056 | no | 0.273 | a hexyl-tetraaza-pentacyclic diamide † | 2.58 | 0.982 |
| 32 | (ClPh)₂PSSH | sc094 | no | 0.156 | dibenzo-18-crown-6 † | 2.23 | 0.167 |

† Some cohort ligands are recorded only under a long IUPAC name; the short name is given here and
the exact recorded string is in the `nn_cohort_name` column of the CSV.  Every one of the 95
candidates has **Eu** as its single measured lanthanide, so the actionable instruction for all of
them is the same: measure one more lanthanide, and make it La, Lu or Ce.

Each row also carries a `donor_census` from the frozen chemistry map and a `topo_census` from the
TOPO39 columns; e.g. rank 1 CA12 — `O(amide_carbonyl)=1 O(ether)=2 n_total=3 dentate=3 core_cn=9
n_ligands=3 n_fill=0`, topicity 6, donor-network diameter 20; rank 23 T8-THP-TAM —
`N(amine)=2 S(donor)=3 n_total=5 dentate=5 core_cn=9 n_ligands=1 n_fill=4`, topicity 2, diameter 6.

60 of the 95 carry a frozen chemotype label absent from the cohort (**53 distinct absent
chemotypes**, matching L6); 77 of 95 have max Tanimoto to the cohort below 0.7 and are "new" by the
chemotype clustering's own threshold.  Median nearest-cohort Tanimoto 0.571.  `ranking_pool_bundle.csv`.

### Pool (ii) — the 273 external aqueous-logK ligands

All 273 have all 39 TOPO39 columns present in `side_K_logk.parquet`; **0 excluded**.  268 of 273 are
"new" at the 0.7 threshold; median nearest-cohort Tanimoto 0.240.

| rank | SMILES | NN Tanimoto | nearest cohort ligand | A-opt | P(heavy) |
|---|---|---|---|---|---|
| 1 | `Cc1occc(=O)c1O` | 0.196 | quercetin | 7.17 | 0.796 |
| 7 | `[Br-]` | 0.033 | Et-Me-TDDGA-Syn | 6.56 | 0.701 |
| 14 | `O=C(O)c1cc(C(=O)O)c(C(=O)O)cc1C(=O)O` (pyromellitic acid) | 0.281 | 2-(dibutylcarbamoyl)benzoic acid | 6.30 | 0.997 |
| 15 | `O=C(O)CN(CC(=O)O)c1ccccc1N(CC(=O)O)CC(=O)O` (PhDTA) | 0.355 | N,N-dimethyl-N,N-diphenylpropanediamide | 6.27 | 0.981 |
| 16 | `O=C(O)CC(CC(=O)O)C(=O)O` (tricarballylic acid) | 0.222 | HEDTA | 5.82 | 0.957 |
| 18 | `CCN(Cc1cccc(CN(CC)CP(=O)(O)O)c1)CP(=O)(O)O` | 0.280 | N-(3,5-dimethylphenyl)-6-[...]-N-ethylpyridine-2-carboxamide | 5.47 | 1.000 |
| 19 | `O=C(O)CCC(NCCNC(CCC(=O)O)C(=O)O)C(=O)O` | 0.176 | HEDTA | 5.34 | 1.000 |
| 20 | `O=C(O)c1ccccc1C(=O)O` (phthalic acid) | 0.483 | 2-(dibutylcarbamoyl)benzoic acid | 5.14 | 0.736 |
| 21 | `O=C(O)/C=C/C(=O)O` (fumaric acid) | 0.167 | HEDTA | 5.14 | 0.866 |
| 24 | `O=C(O)c1cccc(C(=O)O)c1` (isophthalic acid) | 0.273 | DMDPhPDA | 5.03 | 0.799 |

(Ranks skip because of ties; see below.  The list is printed exactly as computed — `[Br-]` at rank 7
is a bromide ion, not an extractant, and phthalic/fumaric acid are aqueous chelators that no one
would put in a solvent-extraction contactor.  It is left in place deliberately: filtering a
deliverable after seeing it is the move this programme exists to avoid, and the fact that the
criterion ranks a bare halide seventh is itself the finding.)

### The ranking's defect, measured

| | bundle pool | logK pool |
|---|---|---|
| candidates | 95 | 273 |
| distinct criterion values | **43** | **141** |
| largest tie group | 11 | 15 |
| fraction of candidates sitting in a tie | **79 %** | **65 %** |

TOPO39 is a 39-column donor-topology census.  Many candidates have byte-identical TOPO39 rows, so the
A-optimal criterion cannot separate them *at all*, and the ties are not near-ties — they are exact.
This is the same wall gen15 §8 hit from the other side ("the model cannot rank candidate ligands").
`ranking_notes.json` records the counts, the excluded SMILES and the 39 column names.

---

## 7. What this simulation cannot tell you

1. **It re-adds only chemotypes the corpus already has.**  Every budget in §2 is a subset of the 45
   frozen chemotypes.  The quantity measured is *ordering among known chemistry*, never the value of
   chemistry the corpus has never seen.  The one thing L6 established is worth having — a second
   lanthanide on one compound from each of the 53 absent chemotypes, taking Kish n_eff from 11.7 to
   27.4 — is **outside the space this simulation can score**, and nothing in §3 supports or refutes it.
2. **A chemotype budget is not a measurement budget.**  A laboratory does not buy chemotypes; it buys
   cells, and §4 shows the two currencies are not interchangeable here by a factor of four.
3. **The curve is flat where the decision is made.**  From k ≈ 16 to the full corpus, every order sits
   within about 0.01 of 0.50 under BP.  An acquisition policy can only pay off where the curve is
   steep, and under this corpus that region is *below* the corpus we already have.
4. **The prospective criterion is variance-driven, not chemistry-driven.**  It rewards descriptor
   outliers regardless of whether they are plausible extractants (hence `[Br-]`), and it rewards
   membership of large families (hence **19 of the top 20** bundle candidates being diglycolamides).  It
   has no notion of synthetic accessibility, cost, phase behaviour, or radiolytic stability.
5. **The direction probabilities attached to the rankings are uncalibrated.**  They come from the
   full-cohort G14 logistic fitted in sample; L5 owns calibration and nothing here should be read as
   a calibrated probability.
6. **Pool (ii)'s descriptors come from a different code path.**  `side_K_logk.parquet`'s `coord__`
   columns were recomputed by gen15's external prep from `coordination_smarts.json`, whose own
   reproduction check against the frozen parquet reports 18 of 506 cohort cells mismatching (max
   absolute difference 6.0).  Pool (ii) ranks are therefore not strictly on the same descriptor
   footing as the cohort they are scored against.
7. **The criterion is not label-blind in one narrow sense, and it should be said.**  `AOPT`
   accumulates precision over *well-determined* cells (≥ 5 measured metals).  That is a count of how
   many measurements exist, not any measured value, and it is the same mask `G14` uses to choose its
   fitting set — but it is metadata a genuinely prospective chooser would not have for an unmeasured
   compound.

---

## 8. Temptations resisted

Recorded for `REFUTATION_LOG.md`; none was acted on.

1. **Stop at BP.**  The BP number (+0.0501, p < 1e-4, 5/5 seeds, LOCO-stable) is the best-looking
   L4 result and BP is the selecting design.  The shared rule requires the sign to hold in all five,
   and it does not.  All five designs were run before any contrast was read, and all five are
   reported side by side.
2. **Drop `UNCERT` after the dry run.**  A 2-draw BP smoke test showed `UNCERT` at −0.0025,
   p = 0.87.  It stayed in the registered set, was run at full size in all five designs, and its
   failure — including a *significantly worse than random* result under A — is reported.
3. **Promote the cell-matched `MAXMIN` result.**  `MAXMIN_vs_RANDOM_CELLMATCHED` is positive in all
   five designs and passes P1 under B and A.  It is exploratory, it was not in §3, and it is not
   promoted.  It is written out as exploratory and flagged as a candidate for a *future*
   pre-registration, where it would need its own matched null and its own five designs from scratch.
4. **Promote the no-diglycolamide result.**  `AOPT_vs_RANDOM_noDGA` passes P1 in all five designs.
   It is a post-hoc scoring subset run as a refuter check; a refuter check can kill a claim, never
   resurrect one.
5. **Filter the prospective lists.**  The obvious move on seeing `[Br-]` at rank 7 and eighteen
   diglycolamides in the bundle top 20 was to add a plausibility filter and print a prettier list.
   The unfiltered ranking is what was pre-registered, so the unfiltered ranking is what is delivered;
   the collapsed-by-chemotype view in §6 is a presentation of the same rows, with the ranks shown.
6. **Choose the averaging set after looking.**  Two averaging sets for the prospective criterion
   (pool alone, cohort + pool) were fixed before running.  Both are reported; they rank identically,
   so the degree of freedom turned out to be empty.
7. **Re-tune the `FLAT` fallback guard.**  The guard fires 2 100 times for `MAXMIN` and never for
   `AOPT`, which flatters `AOPT` on sign accuracy at k = 6/9.  Changing the guard after seeing that
   would be changing a control after seeing a number; instead the asymmetry is reported and the
   sign-accuracy endpoint is left where it was, exploratory.

---

## 9. Comparison accounting

| file | rows | registered | exploratory |
|---|---|---|---|
| `contrasts_abc.csv` | 100 | 20 | 80 |
| `contrasts_per_budget.csv` | 300 | 0 | 300 |
| `contrasts_cellmatched.csv` | 15 | 0 | 15 |
| `contrasts_nodga.csv` | 20 | 0 | 20 |
| **total** | **435** | **20** | **415** |

Arms scored: 4 orders × 6 budgets + `FULL` + `FLAT` = 26 per design, 130 across five designs;
31 625 individual G14/FLAT refits.  BH-adjusted p is in every contrasts file
(`p_bh_within_lead_family`); those adjustments are within L4's own two families only.

## 10. Recommendation

L4's registered question — *does a model-chosen acquisition order make BP macro MAE fall faster with
corpus size than random or maxmin?* — is answered **no**, with a named mechanism: what the A-optimal
criterion selects is training-set volume, and once the comparator is matched on volume the ordering
is worth nothing that survives five designs.  Nothing from L4 should go to confirmation.

The one thing L4 does support, and it is a corpus statement rather than a model statement, is that
**this corpus is not chemotype-starved at its own margin** — the BP learning curve is flat from
k ≈ 16 to k ≈ 29, so "collect more of the same kind of chemotype" is not supported either.  Combined
with L6 (no filter relaxation adds a chemotype; the 53 absent chemotypes all sit behind a
single-lanthanide wall), the actionable statement for a laboratory is not an ordering at all: it is
that **the cheapest useful measurement in this programme is a second lanthanide on a compound that
currently has one**, and that the model cannot presently tell you which one to pick.

---

## Files

All under `D:\ml_separator_gh\gen16_leads\results\L4\`.

| file | what |
|---|---|
| `curve.csv` | learning curves, all designs/orders/budgets, with `k_actual`, `draw_sd`, `macro_sign_acc`, fallback counts |
| `board.csv` | the same rows as `design` × `arm = order@k<budget>` |
| `contrasts_abc.csv` | **registered** ABC contrasts + exploratory endpoints (`mae_far`, `sign_acc_strong`, `pair_spearman`, `MAXMIN_vs_RANDOM`) |
| `contrasts_per_budget.csv` | per-budget contrasts, exploratory |
| `contrasts_cellmatched.csv` | training-cell-count-matched contrasts, exploratory |
| `contrasts_nodga.csv` | diglycolamide-removed and diglycolamide-only contrasts, exploratory |
| `per_extractant_abc.parquet` | the ABC frame fed to `paired_contrasts` |
| `per_extractant_by_budget.parquet` | draw-averaged per-extractant metrics per budget |
| `orders.parquet` | every chosen chemotype order, per (design, seed, fold, order, draw, position) |
| `cells_by_budget.csv`, `first_picks.csv` | the size confound and what each criterion picks |
| `percentile_random.csv`, `flat_fallbacks.csv`, `skipped_fold_budgets.csv` (empty), `summary.json` | supporting |
| `ranking_pool_bundle.csv`, `ranking_pool_logk.csv`, `ranking_notes.json` | the prospective deliverable |
| `l4_retro.log`, `l4_diag.log`, `l4_rank.log` | run logs |
| `_pe_keep/`, `_cellmatched_*.parquet` | intermediates kept so §4 reproduces without a refit |

Code: `gen16_leads/gen16/l4_acq.py`, `gen16_leads/scripts/l4_retro.py`, `l4_rank.py`, `l4_diag.py`.
