# Gen13 — pre-registration: zero-shot separation factors from the lanthanide-axis curve

*Frozen 2026-09-07 after `DATA_AUDIT.md` and before any locked Gen13 arm was fitted on the
frozen fold plan.  The exploratory prototypes listed in `DATA_AUDIT.md` §7 were run on the
same cells before this document; their numbers are disclosed there and are not results.
This file is not edited once an outcome-bearing run starts; a change is a dated addendum
stating what changed, why, and whether any result had been seen.*

## 1. Questions

**Primary (H1).**  Does modelling the lanthanide-axis curve of a cell directly — a low-rank
or physics basis over the 14 lanthanides whose coefficients are predicted from extractant
structure and conditions — predict zero-shot separation factors for chemically unseen
extractants better than the programme's incumbent approach, a row-wise `log D` model whose
predictions are differenced within the cell?

**Secondary.**
* **H2** — the curve model beats every no-chemistry baseline: the corpus mean curve, the
  per-pair-type mean and the nearest-Tanimoto-neighbour curve.
* **H3** — extractant structure adds to conditions: the same curve model with only the
  condition blocks (COND + MASSACT) is worse than with the ligand blocks.
* **H4** — the gain holds on far pairs (`dZ ≥ 5`) and on the sign of strong contrasts
  (|log SF| ≥ 0.3) against the "heavier always preferred" rule (0.681).
* **H5** — one measured separation factor per new cell improves the rest of its curve, and
  the model plus one pair beats a no-model line through the measured pair.
* **H6 (chemistry)** — Gen12.2's coordination block adds nothing measurable here; ECFP
  alone is not enough.  Reported as block ablations, exploratory.

## 2. Cohort, folds, metric — inherited conventions

* Cohort: `manifests/cohort_exact.parquet`, fingerprint `179c8de1fd4715af`, 521 cells,
  90 extractants, 45 frozen gen6 chemotypes (`DATA_AUDIT.md` §2).
* **Design B (primary):** chemotype hold-out with the repository's `seeded_group_kfold`,
  5 folds × 5 split seeds `{104729, 130363, 155921, 196613, 262147}`, model seed
  `42 + fold·1009 + 9,999,991`.  Inner validation: a 4-way chemotype split of the training
  fold (`gen13sep.splits`).  Design A (exact extractant) is a sensitivity, never a headline.
* **Unit of scoring: the extractant.**  Every metric is computed within an extractant over
  all its held-out pairs, then averaged over extractants within a seed, then over seeds.
  Chemotype-macro and pooled numbers are printed beside it, never selected on.
* **Unit of independence: the chemotype.**  Paired chemotype-blocked bootstrap
  (`gen8.inference.paired_chemotype_bootstrap`, 10,000 replicates, seed 8675309), both the
  percentile and the BCa interval, a two-sided bootstrap p, per-seed sign agreement and a
  leave-one-chemotype-out sweep.
* **Reproducibility floor:** 0.01 macro MAE (measured cross-machine tolerance of this
  pipeline).  Interpretability threshold for a claim: **0.02**.

## 3. Target and metrics

For every held-out cell every unordered metal pair `(A, B)` with `Z_A < Z_B` is scored:
`y = log D(A) − log D(B)` (replicate means).  Arms predict `y` either through a predicted
centred curve `c` (`ŷ = c_A − c_B`) or directly (pairwise arms).

| metric | definition | role |
|---|---|---|
| `mae_all` | MAE over all pairs of the extractant | **primary** |
| `mae_far` | pairs with `dZ ≥ 5` | secondary (noise-robust) |
| `mae_adjacent` | pairs with `dZ ≤ 2` (Nd–Sm included) | secondary; near the replicate noise floor |
| `sign_acc_strong` | sign accuracy on pairs with |y| ≥ 0.3 | secondary, vs 0.681 |
| `curve_spearman` | mean per-cell Spearman(observed, predicted) over cells with ≥ 4 metals | secondary |

## 4. Arms (all fitted on the same training cells, scored on byte-identical pairs)

Baselines (no ligand information): `B0_ZERO`, `B1_MEAN_CURVE` (chemotype-balanced training
mean centred curve), `B2_PAIRMEAN` (per pair-type mean), `B3_NN_TANIMOTO` (rank-2 curve of the
nearest training extractant; ties averaged).

Candidates (extremely randomised trees, 400 trees, `max_features` 0.5, leaf 2, median
imputation, chemotype-balanced weights; features = all seven blocks):
`M_LOWRANK_K1`, `M_LOWRANK_K2`, `M_LOWRANK_K3` (in-fold iterative-SVD basis),
`M_PHYSICS_radius+radius_sq`, `M_PHYSICS_radius+radius_sq+gd_break`,
`M_PHYSICS_radius+radius_sq+tetrad_e1+tetrad_e3`.

**Selection rule.**  `M_SELECTED` fits every candidate on the inner-training chemotypes,
scores each on the inner-validation chemotypes (macro MAE over chemotypes of all pairwise
`log SF`), keeps the best, and refits it on the full training fold.  The selection is made
per (seed, fold) and recorded in `predictions/<label>/selections.csv`.  **The primary
endpoint is `M_SELECTED`; no candidate is promoted on a test score.**  The best individual
candidate on test is reported as "best observed test arm" and labelled as such.

Reference: `C_DIRECT_ROW` — the same trees on `(cell features, Z, Shannon radius)` predicting
`log D` per row, differenced in-cell (the gen5–gen12 design applied to the pair quantity).
`C_PAIR_ANTISYM` (gen2–gen4's antisymmetric pair trees) is run at reduced cost as a second
reference where time allows.

## 5. Pre-specified contrasts and decision rules

Let `Δ = MAE(reference) − MAE(candidate)` per extractant; positive favours the candidate.

| id | candidate | reference | metric | rule |
|---|---|---|---|---|
| **P1** | `M_SELECTED` | `C_DIRECT_ROW` | `mae_all` | supported if Δ ≥ 0.02, percentile **and** BCa 95 % intervals exclude 0, p < 0.05, ≥ 4/5 seeds positive, no single chemotype flips the sign |
| S1 | `M_SELECTED` | `B1_MEAN_CURVE` | `mae_all` | same rule |
| S2 | `M_SELECTED` | `B3_NN_TANIMOTO` | `mae_all` | same rule |
| S3 | `M_SELECTED` | `B2_PAIRMEAN` | `mae_all` | same rule |
| S4 | `M_SELECTED` | `C_DIRECT_ROW` | `mae_far`, `sign_acc_strong`, `curve_spearman` | reported with intervals |
| S5 | `M_SELECTED` (all blocks) | `M_SELECTED` (COND + MASSACT only) | `mae_all` | H3 |
| S6 | one-pair adapted | zero-shot and no-model line | `mae_all` on query pairs | H5 |

Equivalence is never claimed from a non-significant contrast.  Power is reported after the
fact as the minimum detectable Δ at 80 % from the bootstrap sd.

## 6. Ablations and sensitivities (exploratory, each labelled)

Block ablations of `M_SELECTED`: all blocks; without COORD; ECFP + COND + MASSACT only;
COND + MASSACT only (S5); COORD + DONORS + COND + MASSACT (no fingerprint).
Design A folds.  `key_mode = relaxed`.  Post-hoc corrected coordination block.
Per-band (far / mid / near by max training Tanimoto) and per-family tables.

## 7. What will not be done

No arm is chosen per seed, per fold or per extractant on a test score.  No fold plan is
re-drawn.  No cell is dropped after a result is seen.  No pooled metric is used for
selection.  No 3D geometry or xTB energy column enters any arm (`DATA_AUDIT.md` §7.1).

---

## Addendum 1 — 2026-09-07, before the locked run; no locked result had been read

The first locked run was started and aborted about 40 minutes in, after an adversarial code
review (three lenses, two refuters per finding) returned defects in the model and analysis
code.  Only a one-seed smoke run of an earlier code state and the exploratory prototypes of
`DATA_AUDIT.md` §7 had been looked at.  Changes, all made before any locked prediction exists:

1. **Data basis**: the iterative-SVD completion never converged and let imputed entries drift;
   replaced by weighted alternating least squares on observed entries only, orthonormalised,
   every basis row scaled to norm √14 (physics rows likewise) so that the per-cell ridge
   (0.5) means the same 3.4 % shrinkage for both families on a full cell.  The old absolute
   ridge shrank every data-basis coefficient by 33 %.
2. **Basis sign rule**: component 1 is aligned with the radius curve; later components have
   their largest entry positive (was: La–Lu endpoint sign, unstable for concave components).
3. **`C_DIRECT_ROW` weights**: each cell now keeps its chemotype-balanced weight spread over
   its observed metals (was: cell weight per row, so chemotypes with many metals per cell
   were over-weighted relative to the curve arms).  Same normalisation for the pair arm.
4. **New baseline arm `B4_HEAVIER_ALWAYS`** (sign always heavy-preferred, magnitude = training
   mean |log SF| of the pair type) so the sign yardstick is scored with the same unit as every
   arm; contrast S4 now names it.  The correct yardstick numbers are 0.855 pooled / 0.63
   extractant-macro on |log SF| ≥ 0.3, not the 0.681 (dZ = 1, all pairs) quoted in §1/§3.
5. **Adjacent** is `dZ = 1` plus Nd–Sm (2,348 pairs per seed; was `dZ ≤ 2`, 4,135).
6. **`pair_spearman`** replaces the mislabelled `curve_spearman` (it is a Spearman over pairwise
   contrasts); a true curve-level Spearman over a cell's metals is computed from the saved
   curves and reported as `curve_spearman`.
7. **Inference**: one shared draw matrix per call gives percentile CI, BCa and the two-sided p;
   `passes_P1` implements the full §5 rule (margin 0.02, both intervals, p < 0.05, ≥ 4/5 seeds,
   LOCO sign-stable) and `mde_80 = 2.80·bootstrap sd` is reported; higher-is-better metrics are
   negated before the delta.  The registered contrast set is the analysis default.
8. **S6 (H5) specification**: primary few-shot contrast = `M_SELECTED` + `basis_shift`
   (λ = 0.7) versus (a) zero-shot `M_SELECTED` and (b) the no-model line through the measured
   pair; other arm × adapter combinations are exploratory.  `rescale` is gated at |contrast|
   ≥ 0.3 and clipped to a non-negative gain.  Support-pair index gap is recorded.
9. **Condition key**: floats rounded to 9 significant digits (no cell changed; cohort
   fingerprint `4c3c6628ea0be949`).  A `series` key mode (adds `experiment_series_id`) is a
   further sensitivity: 597 cells / 89 extractants.
10. **Exploratory arms added, all labelled `X_`**: `X_DIRECT_PROJ_K2`, two ensembles,
    `X_KRR_TANIMOTO_radius+radius_sq` (kernel ridge, Tanimoto × RBF(conditions)),
    `X_PHYSICS+RESID_K1`, and — at the user's request — a 3D *response* block `RESP3D`
    (`gen13sep.features3d`: slope/intercept/residual sd of 15 first-shell descriptors regressed
    on the standardised Shannon radius across a ligand's metal series) used only as a helper:
    `X_RESP3D_ADD`, `X_RESP3D_GATED` (inner validation decides), `X_RESP3D_RESIDUAL`
    (ridge on cross-fitted residuals) and `X_RESP3D_SHUFFLED` (width-matched null), each
    paired against the physics arm without the block.  None of these can become the primary.

---

## Addendum 2 — 2026-09-08, after the locked run and its analysis were read

Recorded because it changes the protocol for any successor generation, not the locked endpoints
of this one.  A second session measured that the 64 condition columns identify a held-out cell's
publication with 94 % 1-NN accuracy.  Design **BP** (chemotype hold-out plus removal of every
training cell from a held-out publication) and its two matched controls **BR** (same cells at
random) and **BQ** (whole unrelated publications, matched on cells and condition series) were
added to `gen13sep/splits.py` after the primary endpoints were scored.  They do not change P1 or
S1–S6, which stand as reported on design B.

They do change what may be selected and what may be quoted, and that rule is registered here for
successors: **the deployed configuration is chosen under BP, never under design B**, because on
this corpus an arm can win under design B through a conditions-only member that is reading the
laboratory rather than the chemistry (§5a of the decision report: conditions-only 0.471 → 0.625
under BP, while unrelated publication removal costs 0.000).  Design B remains the comparison to
gen2–gen12 and the near-analogue regime.

---

## Addendum 3 — 2026-09-08, protocol rules registered for successors

Two rules, both earned by a documented failure in this generation and registered here so that a
successor inherits them rather than rediscovering them.  They do not change anything reported for
gen13.

1. **All designs, always.**  A candidate is scored under every available hold-out design (A, B,
   BP, BR, BQ) and reported with the full table; it is supported only if the sign of its effect is
   consistent across them.  Choosing which second design to run is a researcher degree of freedom
   that a candidate can survive by accident: the amplitude-only arm scored +0.013 under BP and
   −0.007 to −0.024 under the other four.
2. **Two comparators.**  Every representation claim reports its increment over the constant
   baseline *and* over the cheapest sensible alternative already in the repository.  The direction
   result is +0.210 over "always heavy-selective" and +0.075 over the 13-column gen6 donor census;
   only the second establishes that the new block earned its place.
