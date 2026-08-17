# k-shot per-extractant calibration — study, 2026-08-17

Run: `runs/kshot_calibration_20260817T181016Z`
Code: `src/lanthanide_separation/kshot_calibration.py`, `scripts/run_kshot_calibration.py`,
`tests/test_kshot_calibration.py`
Input: `runs/gen4_candidates_20260817T003449Z/oof_predictions.csv` (5 split seeds × 6,699 pairs,
34 extractants, leave-extractant-out outer folds inherited from gen3).

Exploratory, post-hoc on frozen OOF predictions. **Not a protocol run.** No model was retrained.

## TL;DR

The gen4 study left a standing question: the per-ligand miscalibration is worth ~22% of macro MAE
(oracle offset 0.319 → 0.247), and no descriptor family predicts it. So instead of predicting the
per-ligand scale, *measure* it: give the model k pairs of the new extractant and recalibrate.

It works — and it makes the ML model almost irrelevant.

* **Same-experiment regime** (support drawn anywhere in the extractant): A2+TP macro MAE
  0.3327 → 0.1757 at k = 10. Passes the gate. But the trivial pair-mean table calibrated on the
  *same* 10 pairs reaches 0.1968, and a no-model `y ≈ b·ΔZ` fit reaches 0.2812. The champion's
  margin over the trivial baseline collapses from 0.107 (k = 0) to 0.021 (k = 10).
* **Ligand-transfer regime** (`cross_condition`: support from *other* conditions of the extractant,
  query = a held-out condition — the deployment case): A2+TP 0.3785 → 0.2413 at k = 10, but
  pair-mean reaches 0.2322 and the no-model ΔZ fit reaches 0.2395. **From k ≈ 2 onward the frozen
  champion is not better than a two-parameter fit on the same measurements, and is marginally worse
  than a pair-mean lookup table.**

So the answer to "how do we improve extraction predictions" is: with ≥ 2–3 measured pairs of the new
ligand, the current model contributes essentially nothing beyond what the measurements themselves
supply. The k-shot direction is a real improvement *over zero-shot prediction* and worth deploying,
but it is a measurement result, not a modelling result, and it removes most of the motivation for
further zero-shot descriptor work on this cohort.

## Protocol

For each (arm, split seed, extractant) a support pool of `k_max = 10` pairs is drawn; supports for
smaller k are nested prefixes, so every k — including k = 0 — is scored on **identical query rows**.
20 draws per cell. Calibrators, all ridge-shrunk toward the identity `a = 1, b = 0`:

| form | model | antisymmetric | transitivity-preserving |
|---|---|---|---|
| `scale` | `y ≈ a·p` | yes | yes |
| `offset` | `y ≈ p + b` | yes (canonical orientation) | **no** |
| `affine` | `y ≈ a·p + b` | yes (canonical orientation) | **no** |
| `trend` | `y ≈ a·p + b·ΔZ` | yes | yes |

Selection of `(form, λ)` on seed 104729; confirmation on the other four with a paired
extractant bootstrap, one-sided Holm across every gated (arm, policy, k) cell.

**Support policies.** `random` (anywhere in the extractant), `adjacent` (neighbouring lanthanide
pairs first, by rank in the observed metal series), `widest` (La–Lu-style extremes first),
`foreign` (**control** — support from a *different* extractant), `cross_condition` (support from
other conditions, query = one held-out condition; 12 eligible extractants).

**Nulls.** `PAIRMEAN_baseline` (per-metal-pair mean table) and `ZERO` (prediction ≡ 0, so `trend`
becomes `y ≈ b·ΔZ` — a two-parameter fit on the k measurements with no model at all) are run through
the identical draws and query rows.

### Two things that must be controlled, or the answer is ~2× too good

1. **Within-cell transitivity.** Inside one (extractant, condition) cell the target is *exactly*
   additive: `log_SF(A,C) = log_SF(A,B) + log_SF(B,C)`. Cells here are complete graphs, so a query
   pair whose metals are joined by support pairs is an exact arithmetic consequence of the support.
   At k = 10, ~32% of macro-weighted query rows are determined this way (91–97% for the smallest
   extractants). All headline numbers use the **free** stratum — rows *not* spanned by the support
   (union-find per cell over the k_max pool, so the stratum is fixed across k).
2. **Same experiment vs new experiment.** Uniform support sampling usually draws from the very
   condition being predicted (20 of 32 eligible extractants have a single condition). The
   `cross_condition` policy is the honest ligand-transfer measurement.

## Baselines

The harness's k = 0 is **not** the published leaderboard macro, and differs by policy (each policy
removes different pairs into its support). Always compare within a policy.

| arm | published (34 ext) | eligible (32 ext) | k=0 `random` | k=0 free `random` | k=0 free `cross_condition` |
|---|---|---|---|---|---|
| A2_refit_TP | 0.3175 | 0.3113 | 0.3116 | 0.3322 | 0.3737 |
| A2_refit | 0.3192 | 0.3131 | 0.3134 | 0.3340 | 0.3749 |
| PAIRMEAN_baseline | 0.4482 | 0.4436 | 0.4431 | 0.4421 | 0.3933 |
| ZERO (`y ≈ b·ΔZ`) | 0.6360 | 0.6440 | 0.6430 | 0.6487 | 0.4051 |

Two extractants (1 and 16 pairs) are below the 20-pair eligibility threshold; the threshold is
independent of `k_max` so runs with different `--ks` stay comparable.

## Results

### Ligand transfer — the deployment number (`cross_condition`, free rows, confirmation seeds)

12 extractants with ≥ 2 conditions; ~281 query rows per draw.

| k | A2_refit_TP | PAIRMEAN | ZERO (`y ≈ b·ΔZ`) | model − ZERO | model − PAIRMEAN |
|---|---|---|---|---|---|
| 0 | 0.3785 | 0.3934 | 0.4013 | +0.023 | +0.015 |
| 1 | 0.3275 | 0.3244 | 0.3541 | +0.027 | −0.003 |
| 2 | 0.2918 | 0.2928 | 0.2939 | +0.002 | +0.001 |
| 3 | 0.2773 | 0.2750 | 0.2719 | **−0.005** | −0.002 |
| 5 | 0.2589 | 0.2526 | 0.2493 | **−0.010** | −0.006 |
| 10 | 0.2413 | 0.2322 | 0.2395 | **−0.002** | −0.009 |

A2+TP passes the gate at k ≥ 2 (Holm p 0.016–0.038, CI95 low > 0, 4/4 confirmation seeds,
12/12 extractants improved at k = 10) — **but so do both nulls, by more.** Passing the gate here
measures that calibration works, not that the model does.

### Same-experiment regime (`random`, free rows, confirmation seeds)

32 extractants, ~5,851 free query rows.

| k | A2_refit_TP | PAIRMEAN | ZERO | model − PAIRMEAN | gate (A2+TP) |
|---|---|---|---|---|---|
| 0 | 0.3327 | 0.4400 | 0.6483 | 0.107 | — |
| 1 | 0.2809 | 0.3118 | 0.5325 | 0.031 | fails (CI spans 0) |
| 2 | 0.2517 | 0.2642 | 0.3693 | 0.013 | fails (Holm 0.128) |
| 3 | 0.2232 | 0.2384 | 0.3231 | 0.015 | fails (CI spans 0) |
| 5 | 0.1957 | 0.2149 | 0.2988 | 0.019 | fails (Holm 0.087) |
| 10 | 0.1757 | 0.1968 | 0.2812 | 0.021 | **passes** (Holm 0.016) |

The model keeps a real edge here (~0.10 over the ΔZ-only fit), because within one experiment its
per-pair ranking genuinely helps. That edge does not survive the move to a new condition.

### Controls

* **`foreign`** (support from another extractant) is negative at every k for both real arms
  (−0.002 to −0.018, 0/4 seeds positive, 66–92% of individual draws harmful). The gain is
  extractant-specific, as required.
* **In-sample oracle** (fit on all rows of the extractant): 0.1509 free-row macro with `trend`,
  pooled R² 0.709. k = 10 recovers most of that in the same-experiment regime.

### Practical notes

* **`trend` is the right form** at k ≥ 2 everywhere; `scale` wins at k = 1. Best λ ≈ 0.03–0.3 for
  `random`, higher under `adjacent`.
* **Measure wide pairs, not neighbours.** `widest` reaches its k = 1 gain (Δ 0.082) that `random`
  needs 2 pairs and `adjacent` needs 5+ pairs to match: adjacent pairs carry almost no scale
  information. Within `adjacent`, `trend` is not even identifiable (constant ΔZ) — the harness
  rejects those fits back to the identity (`reject_frac` up to 0.71 at k = 2).
* **Single-draw risk matters.** A chemist gets one set of k pairs, not the average of 20. At k = 1
  `cross_condition` the mean gain is +0.048 but 15% of individual draws are *harmful* and the 10th
  percentile is −0.013. By k ≥ 3 harm drops to ~3%.
* **Do not use `offset`/`affine` on a `_TP` arm**: they destroy the within-cell transitivity that
  gen4 confirmed. Verified on seed 104729 — `prediction_A2_refit_TP` has a triangle-violation rate
  of 0.0, which stays 0.0 under `scale` and `trend` and becomes 1.0 under `affine`. The report warns
  when a non-transitive form wins on a `_TP` arm.

## Threats to validity

* **Scope is diglycolamide analogues.** 26 of 32 eligible extractants share the `C(=O)COCC(=O)N`
  core; only one is a different scaffold. "New extractant" here means "new DGA analogue".
* **The 5 split seeds re-partition the same 6,699 pairs** (residual correlation r = 0.97–0.99), so
  "4/4 confirmation seeds" is close to one observation counted four times. `seed_delta_spread` is
  reported for this reason; the extractant bootstrap is the load-bearing statistic.
* **`cross_condition` rests on 12 extractants**, and its query rows are a different population from
  `random`'s — cross-policy level comparisons are invalid.
* The macro (equal-extractant) and pooled (row-weighted) views disagree at small k; the gate
  includes a do-no-harm clause on pooled MAE.

## What this changes

1. **Deploy k-shot calibration** — `trend` at λ ≈ 0.1, k ≥ 3, applied per extractant. It is the
   largest verified error reduction available and needs no new model.
2. **Stop treating zero-shot per-ligand prediction as the frontier on this cohort.** Three
   descriptor families and the simplicial network already failed; this study shows the remaining
   headroom is also reachable from 2–3 measurements without any model.
3. **The open question is now the one the model still owns**: within-experiment pair ranking (where
   it beats the ΔZ fit by ~0.10) and the unseen-*conditions* regime (R² 0.74). A model that helps
   *after* calibration must beat `PAIRMEAN + k-shot`, which is the null this study installs.

See [[gen4-candidate-study-outcome]], [[r2-regimes-and-calibration-bounds]].
