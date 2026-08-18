# gen5 — log D level model: results analysis (run `gen5_levels_20260817T195411Z`)

Analysis date 2026-08-18. Run stamp `20260817T195642Z`, SLURM job 6113126, wall time 3,393 s.
Protocol: [`gen5_levels_protocol_20260817.md`](gen5_levels_protocol_20260817.md) (pre-registered,
committed 12 minutes before the run started).

All numbers below are in **log₁₀ D units** and were recomputed from
`runs/gen5_levels_20260817T195411Z/` — none are quoted from `decision_report.txt` without
independent recomputation. Primary metric is **macro MAE over the 74 ECFP clusters** (one ligand
= one vote), as frozen in protocol §5.

---

## 1. Bottom line

The level model is a **real but narrow** capability, and the run is **half a study**.

1. **It works where a close analogue is already in the training set, and nowhere else.** The
   champion's advantage over a no-model constant is +0.66 log units when the nearest training
   ligand is at Tanimoto ≥ 0.8, and **−0.04 (i.e. the model is worse than a constant)** below
   Tanimoto 0.4. The protocol explicitly refused to claim cross-scaffold generalisation without
   this stratification — the stratification is now done, and it does not support the claim.
2. **Two of the four pre-registered regimes never ran.** `unseen_chemotype` and `unseen_series`
   are absent from every artifact, which makes H1′, H2, H3 and H5 unevaluable. Nothing in
   `decision_report.txt` says so.
3. **Three hypotheses were evaluable and all three pass** — H1 (a 2D ligand family helps a new
   ligand), H4 (3D adds nothing over 2D — here, it actively hurts), H6 (k-shot beats the trivial
   null). H1 passes on a family the authors did *not* expect.
4. **The leaderboard ordering is noise.** The top nine `unseen_conditions` arms span 0.042 log
   units against a per-arm seed sd of 0.024–0.035. Reading `MC_everything` as "the winner" is not
   supported.
5. **Four reporting defects** materially mislead a reader of `decision_report.txt`, including one
   that inverts a headline conclusion and one that prints the study's single largest family effect
   as exactly `0.0000`.

---

## 2. What ran versus what was pre-registered

| Regime | Held-out unit | Pre-registered | Ran |
|---|---|---|---|
| `unseen_chemotype` | Tanimoto-0.7 super-cluster (40) | yes | **no** |
| `unseen_ligand` | ECFP cluster (74) | yes | yes |
| `unseen_series` | measurement series (230) | yes | **no** |
| `unseen_conditions` | condition vector (1,932) | yes | yes |

Cause: `scripts/run_gen5_levels.py:66` defines all four regimes, but both
`slurm/gen5_levels.slurm:26` and `slurm/submit_gen5_levels.sh:23` hard-code
`REGIMES=${REGIMES:-unseen_ligand unseen_conditions}`. The narrowing was **not** resource-driven —
all four fit inside the allocated 3 h / 8 G.

`decision_report.txt` contains zero occurrences of `unseen_chemotype` or `unseen_series`, has no
header line naming the regimes, and carries no caveat. The narrowing is recorded only in
`summary.json`'s `regimes` field. **A reader who has not opened the protocol sees a complete-looking
study.**

Everything else is complete and clean: 34 scored arms exactly matching protocol §3, 5 seeds × 5
folds × 2 regimes = 50 folds all executed, OOF coverage exact and gap-free (48,810 rows, zero NaN
across all 34 prediction columns), all nine promised artifacts present. Fold grouping held
perfectly: 0 of 9,660 (seed × condition) and 0 of 370 (seed × cluster) group-units span more than
one fold. **No leakage was found in the split machinery**, and all nine headline metrics reproduce
to machine epsilon (max |reported − recomputed| ≈ 4.4 × 10⁻¹⁶).

The protocol's 1,939-vs-1,932 condition discrepancy is benign: the protocol counted eligibility on
raw replicate rows, the shipped code counts unique (extractant, condition, metal) cells. One
extractant explains the entire 92/91, 75/74, 1939/1932 and 231/230 gap.

---

## 3. Hypothesis verdicts

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** | A 2D ligand family helps a new ligand: `MC_<fam>` < `MC` [unseen_ligand] | **PASS** | `MC_lig2d_ext` Δ = **+0.1229**, CI95 [+0.025, +0.234], **5/5 seeds**; `MC_donors` +0.0842, CI95 [+0.018, +0.156], 5/5 |
| H1′ | …survives new chemistry [unseen_chemotype] | **UNEVALUABLE** | regime never ran |
| H2 | Conditions carry real signal [unseen_series] | **UNEVALUABLE** | regime never ran (surrogates on both available regimes are large and unambiguous — it would almost certainly have passed) |
| H3 | Known ligand / new series beats the honest nulls [unseen_series] | **UNEVALUABLE** | regime never ran; the one informative surrogate suggests its harder clause was at real risk |
| **H4** | 3D adds nothing over 2D [unseen_ligand] | **PASS, strongly** | `MC_all3d` is **worse** than `MC_all2d` by 0.1017, CI95 [−0.177, −0.034], 0/5 seeds positive |
| H5 | Interpolation ≠ prediction | **UNEVALUABLE** | needs `unseen_series`; also stated as "reported, no gate" — not a test as written |
| **H6** | k-shot beats k=0 *and* beats `NULL_kmean` | **PASS** | both clauses positive on **5/5 seeds** (see §6) |

### H1 detail — and a correction the run could not make itself

H1's pass condition is defined on `MC_<fam>` vs **`MC`**. The shipped `paired_bootstrap.csv` uses
`MC_ecfp` as the reference for **all 66 rows** — the required CI exists nowhere in the run output
except for the ECFP family itself. Recomputing it with the harness's own
`paired_group_bootstrap` (5,000 replicates, cluster as unit):

| Family added to MC | Δ macro (unseen_ligand) | CI95 | seeds |
|---|---|---|---|
| **LIG2D_EXT** | **+0.1229** | **[+0.025, +0.234]** | **5/5** |
| **DONORS** | **+0.0842** | **[+0.018, +0.156]** | **5/5** |
| all2d | +0.0846 | [−0.012, +0.185] | 5/5 |
| physchem | +0.0592 | [−0.035, +0.154] | 5/5 |
| ECFP | +0.0482 | [−0.051, +0.148] | 4/5 |
| complex_phys | −0.0135 | [−0.109, +0.080] | 0/5 |
| all3d | −0.0171 | [−0.118, +0.076] | 0/5 |
| polyhedron | −0.0409 | [−0.141, +0.058] | 0/5 |

The authors expected H1 to be "marginal, ECFP Δ ≈ 0.0–0.08". **ECFP is the family that fails**; the
gen4 extended-2D descriptor table carries the signal. This is a notable reversal — on the *pair*
target those same extended 2D descriptors were rejected in gen4. It is chemically coherent: the
pair target cancels the absolute level by construction, which is exactly what bulk 2D descriptors
predict.

Caveat on H1's wording: 4 of the 13 DONORS columns (`DENTATE`, `coreCN`, `n_ligs`, `n_fill`) are
complex-construction scalars that vary with the metal, so counting DONORS as a clean "2D family" is
contestable. The verdict stands on `MC_lig2d_ext`, which is unambiguously 2D.

---

## 4. The critical finding: the model is a near-neighbour lookup

Protocol §5 states plainly: *"Not claimed: cross-scaffold generalisation without the
`nn_train_tanimoto` stratification."* Doing that stratification is the single most important
result of this run.

Macro MAE by nearest-training-ligand Tanimoto, `unseen_ligand`, mean over 5 seeds:

| nn Tanimoto | clusters | rows | `MC_lig2d_ext` | `NULL_global_mean` | **model edge** |
|---|---|---|---|---|---|
| < 0.4 | 10 | 192 | 1.195 | 1.155 | **−0.041** |
| 0.4 – 0.6 | 15 | 429 | 1.063 | 1.242 | +0.179 |
| 0.6 – 0.8 | 50 | 1,386 | 0.812 | 1.414 | +0.602 |
| ≥ 0.8 | 24 | 2,873 | 0.657 | 1.313 | **+0.656** |

**Below Tanimoto 0.4 the model is no better than predicting a single global constant** (and the
per-seed sign is positive in only 1 of 5 seeds). The same holds for `MC_all2d` (−0.052) and
`MC_ecfp` (+0.071, 1 of 5 seeds).

This matters because the test set is dominated by near-duplicates: **58.9 % of `unseen_ligand` test
rows have a training ligand at Tanimoto ≥ 0.8**, and only 3.9 % fall below 0.4. Median row-level
nearest-neighbour Tanimoto is 0.865. The headline 0.8358 is therefore a weighted statement about
homologues, not about new chemistry.

One lead worth pursuing: **`MC_donors` is the only arm that beats the constant on all 5 seeds below
Tanimoto 0.4** (mean per-seed edge +0.180), despite ranking only 5th overall. The 13-column
donor-atom census is the block least penalised by a near-duplicate-heavy test set.

### `unseen_conditions` is a blend, not pure interpolation

28 of 74 ECFP clusters were measured at exactly **one** condition vector, so holding out that
condition removes the ligand entirely. Measured directly: **12.4 % of (seed, fold, cluster) cells
and 8.95 % of rows have the held-out ligand absent from training.** Macro MAE splits sharply:

| | ligand absent | ligand present |
|---|---|---|
| `MC_everything` | 0.931 | 0.658 |
| `MC_ecfp` | 1.043 | 0.643 |
| `MC` | 1.403 | 0.781 |

The reported 0.7649 is the blend of a genuine interpolation task and a de-facto unseen-ligand task.

---

## 5. Feature families and the 3D question

### The `unseen_conditions` family gains are ligand-identity lookup

Every family gains 0.12–0.26 over `MC` under `unseen_conditions` (all CIs exclude zero), but only
0.05–0.12 under `unseen_ligand`. The mechanism is not chemistry: a block's gain there tracks how
many of the 74 clusters its feature vectors can tell apart (Pearson r = 0.930 across the six single
families), and that relation vanishes under `unseen_ligand` (r = −0.197). Adding ECFP to MC cuts
the mean |per-cluster residual mean| from 0.714 to 0.421 while the offset-removed residual MAE
improves only 0.621 → 0.545 — i.e. **75–85 % of the gain is per-ligand offset correction.**

### 3D: settled, and negative

| comparison | Δ macro | CI95 | seeds |
|---|---|---|---|
| `MC_all2d` vs `MC_all3d` [unseen_ligand] | −0.1017 | [−0.177, −0.034] | 0/5 |
| `MC` → `MC_polyhedron` | −0.0409 | [−0.141, +0.058] | 0/5 |
| `MC` → `MC_all3d` | −0.0171 | [−0.118, +0.076] | 0/5 |

3D is consistently worse on **every seed**, and the harm is concentrated where geometry actually
exists (on the 71 clusters with 3D data, adding POLYHEDRON costs 0.044; on the 3 without, it helps
by 0.036), so it is not a missing-data artifact. This is now the fifth consecutive study in which
3D geometry fails to beat 2D.

The `A_complex_phys` "best alone arm" result (0.9899 under `unseen_conditions`, beating
`A_ecfp` 1.1151) is an artifact: 24 of 26 COMPLEX_PHYS columns change value across metals within
the same extractant, so it is silently a ligand-**and-metal** descriptor while `A_ecfp` is
ligand-only. Its entire advantage disappears when the ligand is new (Δ vs `A_ecfp` = +0.0026,
CI95 [−0.080, +0.081]).

### The leaderboard ordering is not real

| regime | top-5 spread | median per-arm seed sd |
|---|---|---|
| `unseen_conditions` | 0.035 | 0.024 |
| `unseen_ligand` | 0.039 | 0.035 |

Nine `unseen_conditions` arms spanning 0.7649–0.8071 are mutually indistinguishable;
`MC_everything`'s advantage over `MC_ecfp` is +0.0347 with CI95 [−0.042, +0.117]. Under a cluster
bootstrap it is the true best with probability 0.475.

`MC_lig2d_ext` (unseen_ligand, 0.8358) is **the only arm in the whole study that beats the
pre-registered baseline `MC_ecfp` with a CI excluding zero** (+0.0746, CI95 [+0.001, +0.148], 5/5
seeds) — though the lower bound sits essentially on zero, and it is not separable from the
runner-up `MC_all2d@hgb`.

Macro and pooled disagree on the champion in **both** regimes (Spearman 0.91): pooled crowns
`MC_ecfp` under `unseen_conditions` and `MC_ecfp@hgb` under `unseen_ligand`. The gen2 full
rank-inversion does not recur globally, but it does recur pairwise at the top of the board. The
largest ECFP cluster is 1,992 of 4,881 rows (40.8 %); the effective number of clusters for the
pooled metric is 5.42 of 74.

---

## 6. k-shot: the one clean win

This is where gen5 beats its predecessor. The pair-model k-shot study collapsed into a trivial
baseline — from k ≈ 2 the model added nothing over a no-model ΔZ fit. On levels it does not.

| k | `NULL_kmean` | best model form | **model edge** | gain vs k=0 |
|---|---|---|---|---|
| 0 | — | 0.9420 | — | — |
| 1 | 1.2369 | 0.8927 | **+0.344** | +0.049 |
| 2 | 1.0913 | 0.8501 | +0.241 | +0.092 |
| 3 | 1.0404 | 0.8216 | +0.219 | +0.120 |
| 5 | 0.9992 | 0.7908 | **+0.208** | +0.151 |

The edge decays **39.5 %** from k=1 to k=5 — versus **75 %** for the pair model — and is still
+0.208 at k=5. H6 passes on 5/5 seeds for both clauses (vs k=0: +0.024 … +0.079; vs `NULL_kmean`:
+0.313 … +0.377). No calibration fit was rejected (0 of 68,000).

Two caveats. The k=0 clause is carried by a few ligands: `offset` at k=1 beats k=0 on only 16 of 40
extractants, the median extractant is made *worse* by 0.039, and the top 3 contribute 89 % of the
net gain — the cluster-level bootstrap for that clause spans zero. The `NULL_kmean` clause is
genuinely robust. Also, the k-shot study admits only 40 of 91 extractants (the n ≥ 15 gate), which
biases it toward heavily measured ligands.

---

## 7. Defects found — all fixed 2026-08-18

Ordered by how badly they mislead a reader. None of these change the fitted models — all are in
the reporting layer, and the underlying predictions are sound. **Every item below is now fixed in
the harness** (see §7a); the numbers quoted here are what the *published* run reports, and are
what a reader of `runs/gen5_levels_20260817T195411Z/decision_report.txt` still sees.

1. **`within_ligand_r2` is oracle-calibrated** (`levels.py`, metric block). It re-centres the
   prediction on the model's *own per-ligand mean computed from the held-out data*, granting the
   model a free per-ligand offset it does not possess under `unseen_ligand`. Verified:

   | arm (unseen_ligand) | reported | deployable (no free offset) |
   |---|---|---|
   | `MC_ecfp` | **+0.2564** | **−0.0938** |
   | `MC_lig2d_ext` | +0.2506 | −0.0772 |
   | `MC` | +0.1411 | −0.5105 |

   The sign flips. The model is actually *worse* within a new ligand than predicting that ligand's
   own mean. The bug's signature is that every constant-per-ligand arm scores exactly `0.0000`
   instead of a negative number.

2. **The property-family table prints the largest effect as zero.** The row labelled
   `cond (vs metal-only)` has `gain_over_MC = 0.0000` in both regimes because the code compares
   `MC` against `MC`. The number the label promises is `A_metal − MC` = **+0.2960**
   (unseen_conditions) and **+0.3342** (unseen_ligand) — the single largest family effect in the
   study.

3. **The bootstrap is anchored only on `MC_ecfp`**, so the CI that H1 is defined on does not exist
   in the artifacts for any family except ECFP. Add `MC` (and `MC_all2d`) as bootstrap references.

4. **k-shot `mae_free` and `mae_all` are averaged over different draw sets** — pandas silently
   skips the 8.9 % of draws with NaN `mae_free`, and those NaNs concentrate in four extractants
   (99 %, 97 %, 95 %, 65 % of their draws). At k=0 this **inverts the sign** of the contamination
   effect the report is built to show: printed gap −0.0096, same-draws gap +0.0236.

5. **`n_ext_with_free` is a boolean, not a count** (`run_gen5_levels.py:486`:
   `int(s.notna().sum() > 0)`). It prints `1` for every row; the true value is **40 of 40**.

6. **Three "different" nulls are one predictor printed three times.** Under `unseen_ligand`,
   `NULL_extractant_mean` and `NULL_nearest_condition` are bit-identical to `NULL_global_mean` on
   all 24,405 rows (max |diff| = 0 exactly) — an unseen ligand has no training mean to fall back
   on. Correct behaviour, misleading presentation: the leaderboard shows three rows tied at 1.3438.

7. **Stale prose in the report header.** `run_gen5_levels.py:445` hard-codes "92 extractants
   collapse to 75 clusters" two lines below a generated header saying 91 and 74, and the claim
   "folds are grouped on ECFP CLUSTER" is false for the `unseen_conditions` half, which groups on
   `condition_id`.

Minor: the ±50 % range clamp is nearly inert (it bounds 76 of 1,464,300 predictions, all in ridge
arms) because it clamps ±50 % of the training *span* ≈ ±4.6 log units; ridge still predicts +8.82
against an observed maximum of +4.21.

---

## 7a. What was changed in the harness (2026-08-18)

All fixes were verified three ways: unit tests (26 pass), a full four-regime run of the fixed
script, and an independent four-lens adversarial audit that recomputed every claim.

| # | Fix | Verification |
|---|---|---|
| **C1** | **`HGB` crashed on an all-NaN training column**, killing the `unseen_chemotype` run (job 6115512). `DropAllNaNColumns` now drops only columns with no observed value, inside the fold, leaving partial NaN intact. | The fatal fold (seed 130363 / fold 0, train 1,277) now fits. An all-NaN column occurs in exactly **6 of 400** (regime × seed × fold × HGB-arm) combinations, all under `unseen_chemotype`; refitting published folds gives **bit-identical** predictions (max diff 0.0). |
| **C2** | **`within_ligand_r2` was oracle-calibrated.** Now the deployable `1 − SSE/SST_within`; the old quantity is kept as `within_ligand_r2_shape`. | Old published value == new `_shape` to 0.0 on all 340 rows. `MC_ecfp`/unseen_ligand: **+0.2564 → −0.0938**. |
| **C3** | **Bootstrap anchored only on `MC_ecfp`.** References are now `MC_ecfp, MC, MC_all2d, A_metal`, so every pre-registered contrast has a CI. | 132 comparisons vs 33. All of H1, H4, the `cond` row and MC-vs-nulls now covered; published `MC_ecfp` rows reproduce to 0.0. |
| **C4** | **Bootstrap resampled ECFP clusters in every regime**, including `unseen_chemotype` where folds hold out Tanimoto super-clusters — CIs were ~1.35× too narrow. Scoring stays per cluster (so the point estimate is still the macro delta); resampling now follows the held-out block. | Point estimate unchanged (+0.122875 = macro delta exactly); CI width **0.211 → 0.256** on 40 blocks vs 74. H1 still passes: [+0.016, +0.272]. |
| **C5** | **CI depended on a comparison's position** in the dict (one shared RNG stream). One index matrix is now drawn and shared. | Re-ordering the comparisons changes the CI by exactly **0.0**. |
| **C6** | **The `cond` family row compared `MC` against `MC`**, printing the study's largest effect as `0.0000`. Each row now names an explicit reference and carries CI95 + per-seed count. | Now **+0.2960 / +0.3342**. Family table: 0 NaN gains, 0 missing CIs, reference match 24/24, `gain == point_delta_mae` to 0.0000. |
| **C7** | **k-shot `mae_free`/`mae_all` averaged over different draw sets**; the free mask used the `k_max` pool at every k. Added `mae_series_free` (different measurement series, the transfer number), a matched `none_at_k` baseline, and per-k common-draw aggregation. | k=0 sign inversion gone; within each k all forms share the same draws; masks shrink with k as they must. `series_id` is now **required** rather than silently degrading. |
| **C8** | Report now names **which regimes ran and which did not**, per-regime fold grouping, min-rows drop counts in cells, the largest *cluster* share (40.8 %, not the 30 % extractant share), that the three nulls collapse under a ligand-held-out regime, and that both within-ligand R² are row-weighted. | Generated report inspected end to end. |
| **C9** | SLURM ran **two of four regimes** by hard-coded default. All four are now the default in both files; `MIN_ROWS` is plumbed through; walltime 3 h → 8 h, memory 8 G → 16 G. | `bash -n` passes; defaults byte-identical across both files; `DRY_RUN=1 MIN_ROWS=3` prints the override. Measured cost: 4 regimes ≈ **1.9 h**, peak RSS ≈ 1.6 GB. |

Two changes deliberately **preserve** comparability rather than improve on it: the k-shot
admission gate stays at `n_query ≥ 10` (it is now the `--kshot-min-query` flag), and the primary
metric stays macro-MAE over ECFP clusters in every regime.

### Known limits that remain

* The k-shot table must be read **across** a k, never **down** it: each k keeps only the draws where
  a series-free row exists, so `n_draws`/`n_extractants` shrink with k and the k=1 and k=5 rows
  describe different ligand populations. **46 of 91** admitted extractants have a single measurement
  series and can never contribute a series-free row at any k. The report now says so.
* Under `min_rows = 3`, clusters below 8 rows would take **35 %** of the equal-vote macro metric for
  **7 %** of the rows, the guarded per-cluster R² would cover 41 % of clusters instead of 73 %, and
  `group_balanced_weights` would hand a 3-row cluster 664× the per-row weight of the 1,992-row one.
  Measured, the *ranking* is robust to this (Kendall τ 0.957 under subsampling; `MC_lig2d_ext` still
  wins 96 % of the time) — but the recovered chemistry's difficulty is unmeasured, which is the
  whole point of recovering it.

---

## 8. What is usable today

For a chemist choosing conditions for a **known** ligand (`unseen_conditions`, `MC_everything`):
MAE 0.765 macro / 0.595 pooled; **57.6 %** of predictions within 0.5 log, **81.5 %** within 1.0 log.
Error is worst at weak extraction (log D < −2: MAE 0.98) and best in the working range
(0 < log D < 1: MAE 0.46). Metal identity is a minor axis — macro MAE spans only 0.196 across the
14 lanthanides, with heavy slightly worse than light.

For a **new** ligand: MAE 0.836 — 4.4× the median replicate floor (0.19), 1.35× the pooled floor
(0.62), against a target sd of 1.643 — and only if that ligand has a close analogue in the
training set. Predictions are shrunk conditional estimates (dispersion 0.605) and **must not be
read as calibrated log D values**: rescaling them to match the truth variance makes MAE *worse* by
0.184, so the shrinkage is MAE-optimal and cannot be corrected away. The model reaches neither
tail — 0 of 159 rows per seed with true log D < −3 receive a prediction below −3.

A useful reframing: within-ligand pairwise **sign** concordance is 0.771 (unseen_ligand) vs 0.742
(unseen_conditions) — essentially regime-independent — while condition-only R² is 0.219 vs 0.564.
**A new ligand's response *shape* is predictable; its *level* is not.** That is precisely the gap
k-shot closes, and it explains why k-shot works so much better here than on the pair target.

---

## 9. Recommended next run

**Run all four regimes.** They are pre-registered, they cost ~1.9 h in total, and the two that
were skipped carry four of the seven hypotheses. All four are now the default, so:

```bash
DRY_RUN=0 slurm/submit_gen5_levels.sh
```

Then, as a second run, recover the chemistry the cohort filter discarded (see §10):

```bash
MIN_ROWS=3 RUN_TAG=gen5_levels_minrows3 DRY_RUN=0 slurm/submit_gen5_levels.sh
```

Two warnings for `unseen_chemotype`: the largest Tanimoto-0.7 super-cluster is 3,293 rows (67.5 %
of the cohort, 21 of 74 ECFP clusters), so the fold holding it trains on 32.5 % of the data —
projected test-fold max/min row ratio 13.5–20.1. Expect a near-degenerate split and interpret the
macro metric accordingly. Given §4, expect H1′ to **fail**: the effect H1 certifies is largely a
near-neighbour effect, and `unseen_chemotype` is designed to remove exactly that.

The higher-value scientific question is the one §4 raises: **the cohort has only 10 ECFP clusters
below Tanimoto 0.4 from their nearest neighbour.** No amount of re-splitting fixes that.

---

## 10. The cohort filter is where the chemistry was lost

`min_rows_per_extractant = 10` keeps 91 of the 190 extractants in the source table. The 99 dropped
carry only **7.9 %** of the cells — but they are not a random 7.9 %:

| | kept (91) | dropped (99) |
|---|---|---|
| nearest neighbour among the kept set (Tanimoto, median) | 0.773 | **0.561** |
| share with nearest neighbour < 0.4 | 7 % | **37 %** |
| diglycolamides | 41 of 91 — and **74 % of all cells** | 22 of 99 |

**37 of the dropped extractants resemble nothing that was kept** (Tanimoto < 0.4). The current
cohort has 10 such clusters in total. The filter systematically removed the crown ethers,
flavonoids, pyrazolyl-pyrazines, dithiophosphinates and furandicarboxamides — precisely the
chemistry §4 shows the model cannot handle.

Lowering the gate is cheap in rows and enormous in diversity:

| `min_rows` | rows | extractants | ECFP clusters | Tanimoto super-clusters |
|---|---|---|---|---|
| 10 (current) | 4,881 | 91 | 74 | 40 |
| 5 | 5,209 | 140 | 119 | 69 |
| **3** | **5,248** | **152** | **131** | **79** |
| 1 | 5,299 | 190 | 164 | 98 |

`min_rows = 3` buys **+77 % ECFP clusters and +97 % super-clusters for +7.5 % rows**, recovering 61
of the 99 dropped extractants including 25 at Tanimoto < 0.4.

The cost is real and quantified in §7a: small clusters take a large share of the equal-vote macro
metric, and half of them fall below the per-cluster R² gate. Measured, the arm *ranking* survives
this (Kendall τ 0.957 under subsampling), so the risk is interpretive rather than fatal — read the
`min_rows = 3` run as a test of *whether the model generalises*, not as a new leaderboard.

**This is the answer to "how do we improve the model".** Not new arms, not new descriptors, not 3D
(five studies, zero gain). More chemical classes. And §6 shows the complement: 2–3 measurements on
a genuinely new ligand already buy +0.2–0.3 log — so measuring a few points on many *different*
scaffolds beats measuring many points on more diglycolamides.
