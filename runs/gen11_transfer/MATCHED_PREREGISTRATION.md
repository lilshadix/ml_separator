# gen11 `matched` stage — pre-registration

**Written 2026-09-04, before the `matched` stage was run.** At the time of writing exactly
8 OOF parquets exist under `arms/` (the `primary` stage: 4 controls + B, C, D, E). No F or G
arm has been fitted. Nothing below was chosen after seeing a matched-stage number.

## 1. The question

gen11's headline is that auxiliary **actinide** rows improve zero-shot prediction for unseen
lanthanide chemistry (+0.0491 macro MAE vs the design-matched control, 5/5 seeds) while
auxiliary **non-actinide** rows harm it (−0.0534, 0/5 seeds). The stated reading is that the
sign of transfer is set by *chemical relevance*, not by the amount of added data.

That reading is not identified by the arms on disk, for two reasons.

## 2. Confound A — composition (why F vs G is the wrong contrast)

`G_RANDOM_AUX_MATCHED` draws from `(ACTINIDE,) + NON_ACTINIDE_OTHER`. That pool is
**5,050 cells of which 4,607 are actinide = 91.2 %**. A size-matched draw of ~357 cells is
therefore ~91 % actinide, against F's 100 %: the two arms differ by roughly **31 cells out of
357**. F vs G is close to a null contrast by construction and cannot answer the question.

## 3. Confound B — per-row amplification (the alternative hypothesis)

`arms.py::_scale_auxiliary` sets the auxiliary block's **total** mass to `aux_lambda x core
mass`. This was a deliberate choice — its docstring says normalising the total "is what makes
aux_lambda comparable across arms of very different size". The consequence is that per-row
weight is inversely proportional to block size. Measured from `arms/diagnostics_*.csv`
(mean over 25 folds, `aux_weight_share = 0.5000` in every fold of every arm):

| arm | n_core | n_aux | per-row aux:core amplification | measured effect |
|---|---|---|---|---|
| E_LN_PLUS_ALL | 4,198 | 4,247 | **0.99x** | **+0.0473** |
| C_LN_PLUS_ACTINIDES | 4,198 | 3,852 | **1.09x** | **+0.0491** |
| D_LN_PLUS_NON_ACTINIDE | 4,198 | 357 | **11.76x** | **-0.0534** |
| B_LN_EXPANDED | 4,198 | 38 | **109.91x** | -0.0037 |

**Both arms that help sit at ~1x amplification; both arms that hurt are heavily amplified.**
So the existing C-vs-D contrast varies chemistry and per-row weight *together*, by a factor of
~11. An equally consistent reading of the same numbers is:

> **H_AMP** — the sign of transfer is set by how far the auxiliary block is up-weighted, not by
> its chemistry. A small block forced to carry half the training mass distorts the fit
> regardless of which metals it contains.

## 4. The decisive contrast: F vs D

`F_ACTINIDES_ONLY_MATCHED` is size-matched to D. It therefore has **the same n, the same
`aux_lambda`, and the same ~11.8x amplification**, and differs from D only in chemistry.
F − D is the contrast that separates H_CHEM from H_AMP. It is the primary endpoint of this
stage. (F vs the design-matched control is reported but is not the discriminating quantity.)

## 5. Pre-registered predictions

Let `eff(X) = macro_MAE(design-matched control) − macro_MAE(X)`, positive = better.
Reference: control 0.97324; eff(C) = +0.0491; eff(D) = −0.0534. F has n_aux ≈ 357.

| hypothesis | prediction for `eff(F) − eff(D)` | prediction for `eff(F)` |
|---|---|---|
| **H_CHEM** — chemistry sets the sign | **≥ +0.04**, BCa excluding zero, ≥ 4/5 split seeds | ≥ −0.01 |
| **H_AMP** — amplification sets the sign | **\|·\| < 0.02**, BCa including zero | ≤ −0.03 |
| **H_MIX** — both contribute | between +0.02 and +0.04 | between −0.03 and −0.01 |

**Arithmetic check on §2** (falsifies my composition reasoning, not the science):
`|eff(G) − eff(F)| < 0.02`. G's pool is 91 % actinide, so G must track F. **If G instead tracks
D, my reading of the sampler is wrong** and §2 must be withdrawn before anything else is
interpreted.

## 6. Decision rule

* Primary endpoint: `eff(F) − eff(D)`, i.e. paired difference in macro MAE (one vote per ECFP
  cluster, 131 units) on identical held-out rows.
* Inference: paired block bootstrap resampling the Tanimoto-0.7 chemotype (79 blocks),
  5,000 replicates, percentile and BCa, RNG seed 8675309 — the repository's own
  `paired_chemotype_bootstrap`, imported, not reimplemented.
* F and G are each averaged over the 5 fixed draws (`MATCHED_SEEDS = 11, 22, 33, 44, 55`);
  per-draw values are reported separately so draw variance is visible.
* 5 split seeds (104729, 130363, 155921, 196613, 262147). Report `seeds_positive` and
  `draws_positive` separately; neither is evidence on its own.
* An effect below **0.01** macro MAE is not interpreted — that is the measured cross-machine
  tolerance of this pipeline.

## 7. Pre-declared diagnostics

1. **Am dose-response.** Am is 32.65 % of the pool and, at one seed, carries the whole LOMO
   contribution (+0.148 while every other metal is negative). For each F draw, record the Am
   fraction and correlate it with that draw's effect. **If Spearman ρ > 0.5, the mechanism is
   americium-specific, not actinide-general**, and must be reported that way.
2. **Level vs shape.** C's gain was entirely level (offset +0.0492, shape +0.0026). If F shows
   the same split it is the same mechanism at smaller n; if F's gain is in shape, it is not.
3. **Far band.** C's gain is absent on nn ≤ 0.4 (−0.0077) — the band where the archive was
   supposed to help most. Report F on the same band.

## 8. What will not be done

No draw will be re-rolled (`MATCHED_SEEDS` is fixed in source with the comment "cannot be
re-rolled after seeing a result"). No subset of draws or seeds will be selected. The reference
arm stays the design-matched control `A_GEN10_CONTROL|JOINT|GENERAL|ANNOTATION_SAFE|
HIERARCHICAL|lam1`. If H_AMP survives, that is the finding and it will be reported as such —
it would retract gen11's central claim, and the retraction is the result.

## 9. Follow-up already identified (exploratory, not pre-registered here)

If H_AMP survives, the direct test is a lambda ladder on D at per-row parity
(`aux_lambda = n_aux/n_core ≈ 0.085`) against D at lambda = 1. That arm does not exist and would
be labelled EXPLORATORY.
