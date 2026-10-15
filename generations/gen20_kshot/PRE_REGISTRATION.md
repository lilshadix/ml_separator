# Gen20 — pre-registration

*Written 2026-10-14 after `AUDIT.md` and `IMPLEMENTATION_PLAN.md` (draft 2), before any
outcome-bearing run. Harness debugging used labelled smoke runs (`smoke*`, at most 40 cells of one
fold, tiny ensembles); one such table was inspected to fix the neural model's stopping rule (plan
§3.4). After the SHA-256 footer is appended, every outcome-bearing run checks it
(`gen20k.manifest.require_sealed_preregistration`) and this file is not edited; a change is a
dated addendum saying what changed, why, and whether any result had been seen.*

## 0. Seeds

Discovery split seeds are gen13's `(104729, 130363, 155921, 196613, 262147)`; the model seed is
gen13's `42 + fold·1009 + 9 999 991`; the bootstrap seed is 8675309 with 10 000 draws; random
support draws are seeded by SHA-256 of `gen20|cell_id|split_seed|random|draw`.

Confirmation split seeds follow gen16's public rule with the string
`gen20-confirmation-seed-{i}`: `int(sha256(string)[:8], 16) % 900000 + 100000` for i = 1, 2, …,
skipping any value equal to a discovery seed or to a gen16 confirmation seed, taking the first five
in sorted order. The rule is public and anyone can recompute the seeds; the guarantee is
procedural: no discovery number is computed on them, and the confirmation runner executes once.

## 1. Questions

**Primary.** For a chemically unseen extractant (chemotype held out; the held-out cells'
publications removed from training), do k measured separation factors make its adjacent-lanthanide
separation factors predictable, and does any model earn anything beyond a straight line through
the same measurements?

**Secondary.** Which support policy is worth the most; whether the reference-derived neural
contrast model beats the deployed baseline at any k; whether molecular structure still matters
after the supports (`FLAT@BLUP20` vs `G14@BLUP20` vs `NNC@BLUP20`); which adjacent pairs remain
unpredictable; design invariance; Setting B as an exploratory arm.

## 2. Cohort, target, hold-out

Frozen gen13 cohort (521 cells, 90 extractants, 45 chemotypes, 58 publications), loaded through
gen14's bench and asserted column-wise equal to `gen13_separation/manifests/cohort_exact.parquet`
(SHA-256 `3f11ac4c…`). Target `y(A/B) = log10 D(A) − log10 D(B)`, A the lighter metal, inside one
cell. Primary query population: strict adjacent pairs (|ΔZ| = 1, 12 pair types). Designs: BP
(selects), B, and BPG/BR/BQ/A as sensitivities, all from `gen20k/splits.py`; every fold set passes
`assert_design_integrity`. Inner selection only on each fold's inner chemotype split with the
design's mask re-applied inside.

## 3. Setting A design (plan §3.3)

The query set deviates from the recovery plan's hash-defined 20 % subset: `Q_adj` is *every*
strict-adjacent pair of the cell, which is fixed before any model runs, identical for every
budget, arm and policy (asserted on the run output by a test), and contains Pr/Nd whenever it is
measured. The gen13-definition set (adjacent plus Nd–Sm) is run as a separate labelled pass
(`discovery_nd_sm`, design BP, metal model).

Two measurement models are run for every held-out cell with ≥ 1 adjacent pair:

* **Metal model (registered).** A measurement is one metal's log D reading in the cell. After
  k + 1 metals are read every separation factor among them is known, so the support is the set of
  read metals (a star spanning tree is what adapters see; with the shared-reading noise model the
  tree choice is immaterial) and k is the number of independent separation factors. Because the
  query is fixed, the read set may never contain both metals of an adjacent pair (an independent set
  of the adjacency path; a metal is legal only if the maximum independent set stays reachable, so
  every policy reaches `k_max`, which is 6 for a 14-metal cell). k ∈ {0, 1, 2, 3, 5, 6}; policies
  RANDOM (5 draws), UNIFORM (lightest, heaviest, then farthest point), DOPT_CURVE, DOPT_Q and
  UNC_SEQ (neural ensemble only). Common-query cohorts C3 (k_max ≥ 3: 229 cells, 79 extractants,
  38 chemotypes), C5 (144 / 64 / 29) and C6 (133 / 58 / 27). The budget grid is
  {0, 1, 2, 3, 5, 6, 8, 10} for both models; the metal model never reaches 8 or 10.
* **Pair model (the brief's definition, exploratory).** A measurement is one separation factor;
  supports are non-adjacent pairs forming a forest under the per-pair no-closure rule; k ∈
  {0, 1, 2, 3, 5, 8, 10}; policies RANDOM, WIDEST, UNIFORM, DOPT_CURVE, DOPT_Q, UNC_SEQ; cohorts C3,
  C5, C10. Supports still fix linear combinations of the adjacent queries (every support is a sum
  of adjacent contrasts), so each row carries `lab_known` (both metals already read) and the
  arithmetic-only `PROJ` arm quantifies what the measurements alone determine; at k = 10 on a
  14-metal cell 10 of 13 curve degrees of freedom are measured and the model fills three.

All covariance-driven policies use the G14 inner-OOF covariance, so every arm sees identical
supports. The remaining-query regime (all non-support, non-closed, not-both-read pairs) is
secondary and labelled.

## 4. Arms (plan §3.4)

Model 1: `G14@ZERO`, `G14@BLUP20` (inner-OOF PSD covariance; shared-reading noise `R = s²DDᵀ`
with `s²` the squared median training-replicate sd over readings whose disagreement is below 0.5
log units, times a factor chosen on inner folds from {0.5, 1, 2, 4, 8, 16}; the sanity bound
`2s² < Var(y_adjacent)` is asserted per fold), `G14@BLUP_INDEP` (gen15's literal 0.09·I,
sensitivity). Model 2: `NNC@ZERO`, `NNC@BLUP20`, `NNC@HEAD_FT` (its own inner-chosen penalty),
`NNC@FULL_FT` (DOPT_Q at k ∈ {1, 3, 5, 10}, or everywhere if selected), `NNC@SELECTED`
(configuration, mechanism and penalty chosen on inner folds by the mean inner-validation adjacent
MAE at k = 0 and k = 3, metal model, DOPT_Q supports, nested covariance). Model 3: `FLAT@ZERO`,
`FLAT@BLUP20`, `LINE`, `PROJ`, `RIDGE2` (λ chosen on inner folds), `MEAN_CURVE@ZERO/@BLUP20`,
`NN_TANIMOTO@ZERO/@RESCALE/@BLUP20`.

The arm that carries the registered claims is decided from the inner selection records alone,
before any outer score is read, and the report script has no override: it is `NNC@SELECTED` if
the selected neural key's inner adjacent MAE at k = 3 is below `G14@BLUP20`'s inner adjacent MAE
at k = 3 (same inner cells, same supports, same nested covariance, recorded as `mstar_inner`) on a
majority of the 25 BP (seed, fold) selections, otherwise `G14@BLUP20`. Call it `M*`.

## 5. Metrics (plan §3.5)

On query rows: pooled R² (NA with reason when n < 2 or Var y = 0, never clipped), MAE, RMSE,
direction accuracy at |y| > 0, ≥ 0.1, ≥ 0.3 (a zero prediction scores 0.5), extractant-macro MAE,
per-pair-type MAE and R², ΔR²(k) = R²(k) − R²(0), ΔMAE(k). All per split seed; the headline is the
mean over seeds with per-seed values printed. MAE is the selection metric; R² is the complementary
metric.

## 6. Inference (plan §3.6)

Chemotype-blocked bootstrap over per-cluster sums (10 000 draws, one draw matrix shared by every
arm and budget, applied to every seed), percentile and BCa intervals, two-sided percentile p,
leave-one-chemotype-out, per-seed sign counts, and the minimum detectable effect `mde_80 =
2.80 × bootstrap sd` emitted with every contrast. Because the percentile chemotype bootstrap
rejects at 7–8 % at nominal 5 % on this cluster profile (gen13 inference audit), every registered
MAE claim also carries the **wild-cluster-t p (CR2, Rademacher)** on the per-extractant MAE deltas,
which holds its size, and that p must also be below 0.05.

Decision rule for a registered MAE claim: pooled point ≥ 0.02, percentile and BCa intervals both
exclude 0, percentile p < 0.05, wild-cluster-t p < 0.05, ≥ 4 of 5 seeds positive, every
leave-one-chemotype-out value positive. For the registered R² claim: the ΔR² percentile and BCa
intervals exclude 0 and R²(k) > 0. Registered claims use Bonferroni over four: the intervals that
decide are 98.75 %; 95 % intervals are descriptive. A claim whose `mde_80` exceeds its own margin
is reported as underpowered whatever its point estimate.

## 7. Registered claims (design BP, metal model, strict adjacent pairs, policy DOPT_Q)

| id | contrast | cohort | estimand |
|---|---|---|---|
| P1 | `M*` at k = 3 vs `LINE` at k = 3, identical supports | C3 | extractant-macro MAE |
| P2 | `M*` at k = 3 vs `M*` at k = 0 | C3 | extractant-macro MAE |
| P3 | `M*` at k = 3 vs `M*` at k = 0 | C3 | pooled ΔR² |
| P4 | `M*` at k = 6 vs `LINE` at k = 6, identical supports | C6 | extractant-macro MAE |

The MAE estimand is the extractant-macro MAE (one vote per extractant, its adjacent pairs pooled,
averaged over seeds; Kish effective extractants ≈ 12 on C3), inferred by the chemotype-blocked
bootstrap and the wild-cluster-t on the per-extractant deltas; the pooled MAE and the chemotype-
macro MAE are printed beside. P3 is the pooled ΔR² with the denominator recomputed in every draw.
The pair-weighted Kish effective number of chemotypes on these rows is about 4 (one chemotype
holds half of the adjacent pairs), which is printed with every cohort; the diglycolamide
hold-out fold is shown on its own and the remaining chemotypes without it.

Expected from the programme's history, stated now: P2 and P3 positive; P1 and P4 uncertain
(gen15 found +0.064 at D-optimal k = 3 on all pairs with arithmetic recovery included and
adjacent pairs at the noise floor). A null on P1/P4 is a result. C6 is a narrower population
(58 extractants, 27 chemotypes); k ≤ 3 is the headline, k = 5/6 and the pair model's k = 8/10
are sensitivities. "Chemically unseen" means the chemotype is held out: 63 diglycolamide cells
live in chemotypes other than the main one, so the main hold-out still trains on some
diglycolamide analogues; this is stated in the limitations.

Everything else is exploratory: the other policies, every per-pair table, Model 2 vs Model 1 at
each k, the remaining-query regime, the other designs, Setting B. The exploratory family is
counted and Benjamini–Hochberg-adjusted p-values are printed beside the raw ones.

## 8. Confirmation

The four registered contrasts are re-run once on the confirmation seeds with the same arms, the
same inner-selection procedure and the same cohorts; a claim is confirmed only if it passes its
rule on both the discovery and the confirmation seeds.

## 9. What will not be done

No arm, policy, penalty or cohort is chosen on an outer score; no query set is redefined after a
result is seen; no cell is dropped after a result is seen; no in-sample oracle is reported as a
ceiling; Setting B is never promoted to a claim; the neural model is never given 3D geometry or
the ECFP/LIG2D blocks.
SEALED-SHA256: 2915974a8a46a3ff465ae2a144f67e825edecc67e8d60265a542d4b0b78c5cc6
