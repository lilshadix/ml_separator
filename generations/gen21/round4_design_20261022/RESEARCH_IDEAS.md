# Gen21 round 4: evidence, hypotheses, and a defensible improvement target

Prepared 2026-10-22. This is an analysis and experiment design, not a trained model or a positive experimental result. The executable research brief is `CLAUDE_CODE_PROMPT.md` in this directory.

## Recommendation

Keep radius interpolation, add a strictly training-fitted Gd correction, and learn only the remaining, support-consistent curve shape. Prioritize source-correct labels and independent, matched-condition lanthanide series. Test electronic interactions and actinide-assisted learning as controlled hypotheses. Adapt the strain repository's separation of level and shape; do not import its trained weights, pooled OOF embeddings, or numerical gains.

The evidence does **not** establish data shortage as the sole or principal bottleneck. The present limitations include source defects, few effectively independent chemical groups, missing speciation/outer-sphere information, and small residual gains after interpolation. More unrelated observations need not resolve any of these.

“Continue until positive and significant” must mean a persistent, well-resourced search with new independent evidence when necessary. Repeatedly inspecting the same confirmation data until a threshold is crossed cannot deliver the requested scientific rigor.

## Evidence and versions

| Material | Version inspected | Limitation |
|---|---|---|
| User's pasted round-3 audit | Attachment and matching `../AUDIT_ROUND3_2026-10-22.md` | Audit findings were read; this task did not independently rerun its numerical verifications. |
| Local `ml_separator` | `9be774c30649f672f365f5586bba10be59ea78c2`, branch `gen21-logsf-target` | Round-3 implementation is absent locally. Existing two untracked audit files were preserved. |
| Round-3 implementation and reports | GitHub files pinned at `7d02f8e` | Read through the GitHub connector; local checkout was not fetched, merged, or reset. |
| Local `lanthanidestrain` | `7041d51970970afab8653eba822207ce01c773f8` | Read-only; existing `.DS_Store` change and `docs/poster/` were preserved. |
| Remote strain repository | `025f8084b52986773da98fffde13ea267c0ad908` | 55 commits ahead of that local checkout; core anchored-model/result files unchanged, newer transformer report also inspected. |
| Strain reproducibility assets | Code, reports, frozen pair identities | Required OOF/model artifacts were missing locally; the +0.0156 contrast and its uncertainty were not independently recomputed. |

## What the Gen21 audit establishes

The primary task is **strict-adjacent** `logSF(A/B) = log10 D(A) - log10 D(B)`, lighter metal first. `k=3` means **four measured metal logD values**, not three. C3 contains 229 cells and 1,802 scored pairs per seed. BP excludes held-out chemical groups and their publications; BPG additionally excludes known cross-publication copies. No query may have both endpoints measured.

| Predictor, C3/four readings | Final BP / BPG R² | Interpretation |
|---|---|---|
| PWL_R | 0.422 / 0.423 | Radius interpolation is the substantive reference. |
| R02, PWL_R plus geometry ridge | 0.422 / 0.423 | Approximately +0.0002 over PWL_R; no demonstrated improvement. “Geometry” here concerns interpolation/read geometry, not evidence of a useful molecular 3D encoder. |
| R14, PWL_R plus one shrunk Gd coefficient | 0.455 / 0.454 | About +0.03; the relevant stronger reference for new work. |
| R23, Gd correction with POL_ERR reads | 0.471 / 0.463 | Different read policy; a separate regime, requiring method-by-policy comparisons. |

These are historical audit values, not results of this task. See the [round-3 report](https://github.com/lilshadix/ml_separator/blob/7d02f8e/generations/gen21/next_research_20261019/FINAL_REPORT_RU.md) and the supplied audit.

The Gd correction's gain is predominantly a pair/read-set prior: the audit finds 93% survives replacement by pair-type constants, while the cell-specific remainder is only about +0.0005 to +0.002 R². A stronger reference should therefore include a regularized, training-only series/read-set prior when assessing claims of chemistry-dependent learning.

### Two important qualifications to the audit's interpretation

1. **The fixed -0.072 coefficient is not external physics.** `generations/gen13_separation/analysis/stage2/D7/FINDINGS.md:3-15` derives it from full curves in 268 cells of this historical corpus. It is a useful disclosed historical comparator, but it cannot become a pristine fold-independent physical prior by being frozen now. Use an outer-training-fitted, inner-tuned Gd coefficient as the strict reference. Include zero correction in tuning. The approximate 0.45 reference level is historical, not a guaranteed score for the new implementation.
2. **A Gd-centered improvement does not establish a unique electronic mechanism.** In a primary TODGA study, outer-sphere nitrate/water interactions explain important selectivity trends. The half-filled 4f shell motivates a candidate feature; the fitted dip alone does not prove causation. [Baldwin et al., 2018](https://doi.org/10.1021/acscentsci.8b00223).

The original within-pair shuffle was incapable of removing a between-pair Gd prior. Use controls matched to each claim: universal series priors, cell-dependent chemistry, electronic mapping, and geometry need different interventions. A powerless or logically irrelevant control must not decide model promotion.

## What the strain architecture actually does

It predicts logD-derived separations and is evaluated using R²; it does not predict R² itself. Its successful architecture combines **scalar shape predictions**:

```text
prediction = anchor + 0.65 * tabular_shape + 0.35 * encoder_shape
```

One CatBoost model learns the level; the champion's second model learns targets centered by **training extractant**, with composition-block centering available as another variant. The later blend centers predictions by composition block. A SchNet-style distance encoder supplies another shape estimate from atom types, charges, donor/metal flags, and interatomic distances. The common anchor cancels in pair differences. This is different from feeding independently fitted OOF latent embeddings into CatBoost. See [anchored_champion.py](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/topo/anchored_champion.py), [dist_gnn.py](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/topo/dist_gnn.py), and [anchored_3d.py](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/topo/anchored_3d.py).

The exact fresh-subset comparison is 0.1206581465 versus 0.1050815759, giving **+0.0155765706 R²**. However:

- The declared PASS rule required only a positive contrast; it did not require a confidence interval excluding zero. No contrast CI/p-value is supplied in the result JSON.
- The 444 pairs were newly eligible records in the same database after geometry-QC relaxation, not a newly collected external cohort. Their frozen identities span 117 condition blocks and 59 extractants.
- Conditions use the repository's **binned** composition key, rather than Gen21's exact experimental-cell contract.
- Those pairs have subsequently been reused in another architectural comparison; they cannot now serve as untouched confirmation data.

Evidence: [result JSON](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/reports/anchored_3d_confirm.json), [confirmation rule/scorer](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/topo/anchored_3d_confirm.py), [fresh-subset construction](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/topo/fresh_eval.py), [condition construction](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/dataset.py).

The newer [transformer report](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/reports/TRANSFORMERS.md) does not establish an architecture upgrade: attention beats the distance encoder on the 444 pairs by +0.0102, loses on the legacy pairs by -0.0059, and is essentially tied on the union (-0.0003). Tabular and block transformers lose to CatBoost. There is little justification for beginning Gen21 with a large architecture sweep.

### Reuse the architecture, rebuild the information boundaries

- A frozen supervised encoder can still leak if its original training included held-out SAFE records, extractants, publications, or copies. Audit its full training lineage or retrain within each outer fold.
- Selecting blend weights from pooled OOF predictions requires a dependency audit. Removing the target extractant's score is insufficient if models generating other meta-training predictions were trained on its labels. Regenerate all meta-training predictions entirely inside the current outer training set.
- Do not combine embeddings from different fold-trained encoders as if their latent coordinates were aligned. Scalar predictions avoid that problem.
- Train/test geometry availability must share one evaluation denominator. Use interpolation fallback and report geometry coverage; do not improve a score by retaining only geometry-successful queries.
- All centering, imputation, normalization, early stopping, masks, covariance estimation, and blending must respect the inner folds, not only the final predictor fit.

## These proposals have relevant negative precedents

| Prior work | Verified outcome | Consequence for round 4 |
|---|---|---|
| Gen11 expanded An training | Corrected report: INCONCLUSIVE. Macro-logD MAE gain +0.0491, registered CI [-0.00049, +0.11865]. Matched An-versus-non-An gain +0.0142, CI [-0.034, +0.085]. | More An rows are not an established remedy. Match source diversity and weights, not only row counts. |
| Gen19 transfer | An benefit unestablished; Pr/Nd S2 failed; descriptor M0 outperformed factorized M2 on V5. | Promote on actual Ln logSF improvement, not transfer of individual-metal levels. |
| Strain Ln+Am/Cm | Ln gain +0.0021, seed SD 0.0204; zero-shot Am/Eu R² -0.1362; joint Am/Eu 0.4666 versus family mean 0.4716. | Shared f shells did not produce a demonstrated Ln benefit in that experiment. |
| Existing electronic features | Gen11/19 already include f count and metal-state descriptors. The isolated strain metal-physics arm changed held-out R² by -0.0044. A composite endpoint adding those features, cutoff and pair weighting changed it by -0.0127, 90% CI [-0.0485, +0.0196]. | “Add f count” is not a new standalone experiment. The composite interval cannot isolate electronic-feature effects. |
| Gen15 geometry/energy | Naive energy-slope correlation 0.644 fell to 0.195, p=0.11 after composition control; 85/219 series changed stoichiometry. | Cross-metal energy differences need identical composition and reference states. |

Local evidence: `runs/gen11_transfer/GEN11_DECISION_REPORT.md` (September 4 corrected report); `src/lanthanide_separation/gen11/metalrep.py`; `generations/gen19_chem_transfer/SUMMARY.md`; `generations/gen15_curve/GEN15_REPORT.md`. Strain evidence: [An/Ln summary](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/reports/an_ln/summary.json), [C6 report](https://github.com/lilshadix/lanthanidestrain/blob/025f8084b52986773da98fffde13ea267c0ad908/automl/reports/C6_RESULTS.md).

## Prioritized hypotheses

| Priority | Hypothesis and implementation | What would distinguish it from old work | Essential comparison |
|---|---|---|---|
| 1 | **Source quality and matched Ln coverage.** Resolve reported floors, transcription errors, mixed stereochemistry/conditions, and copies; acquire new full Ln series from independent chemistry. | Adds validated information about the actual target rather than more database rows. | All models on the same corrected version; unchanged V1 analysis reported separately. |
| 2 | **Support-consistent residual curve.** Small hierarchical spline/kernel or CatBoost predicts departures from the training-fitted Gd-PWL reference; shrink unsupported corrections to zero. | Conditions on the real four-reading task and scores only residual separation skill. | Gd-PWL, plus a training-only universal series/read-set prior of matched capacity. |
| 3 | **Electronic features as interactions.** Small donor/condition-dependent amplitudes for prespecified f-shell/tetrad bases. | Asks whether ligand context changes a series effect beyond radius/Gd and generic flexible Z features. | Same-capacity Z/radius basis; electronic-map permutation; donor ablation. |
| 4 | **Anchored scalar 3D shape correction.** Rebuild distance-encoder shape and blend with tabular residual predictions inside each outer fold. Start with compact geometry descriptors. | Uses the strain architecture for the remaining interpolation error, with exact support consistency. | 2D-only, random encoder, block-shuffled geometry, and radius-only geometry controls. |
| 5 | **Selective An(III) transfer.** Shared representation with private Ln/An heads and an auxiliary weight that may be zero. | Compatible domains, strict provenance exclusions, target-normalized losses, and promotion solely on Ln residuals. | Ln-only scaffold already containing both residual and raw-logD tasks; extra-Ln/source-diversity control; naive shared head; permuted An labels. |
| 6 | **Read allocation and noise-aware conditioning.** Label-free policy using train-estimated residual uncertainty; robust/censored likelihood only where source evidence supports it. | May improve deployment efficiency, but is a separate information/policy regime. | Cross method and policy on identical queries; compare four versus six reads on the same C6 cells. |

These are hypotheses, not expected positive outcomes. The strongest starting pair is priorities 1 and 2. Geometry and An transfer should advance only if cheap, honest controls leave measurable residual signal.

### A concrete interpolation-preserving architecture

Let `I_S` be the existing radius-PWL operator from a legal read set S, with its existing constant boundary continuation. Let `h` be the Gd indicator and `beta` a coefficient fitted using only the allowed training tasks. Define

```text
b_S(m) = I_S[y_S - beta*h_S](m) + beta*h(m)
r_theta(m) = g_theta(m, chemistry, conditions, allowed support summaries)
y_hat(m) = b_S(m) + r_theta(m) - I_S[r_theta(S)](m)
logSF_hat(A/B) = y_hat(A) - y_hat(B)
```

At every measured support, `y_hat(s)=y_s`. The Gd term also removes contamination of neighboring interpolation values when Gd itself is read. Pair predictions are differences of one curve, so antisymmetry and cycle consistency follow. All `r_theta(S)` values are **model predictions**, not hidden measurements. Train with the actual masked-task adjacent-logSF loss; do not substitute a high logD score for success.

Test this projection rather than assume it helps. If an alternative smooths measured supports to account for analytical noise, it is a separately registered observation contract and needs its own baselines.

### How to use f-shell information correctly

For the ordinary Ln(III) series, formal `n_f=Z-57`; it adds no independent information beyond element identity. A benefit from `|n_f-7|`, tetrad terms, or a half-shell flag can reflect a better basis/regularizer. It is not evidence that the learner discovered electrons. State-specific configurations must come from verified ionic data: Nd(III) is 4f3; neutral-atom configurations are inappropriate. [NIST Nd(III) analysis](https://www.nist.gov/publications/analysis-free-ion-nd3-spectrum-nd-iv), [NIST ASD](https://www.nist.gov/pml/atomic-spectra-database).

Use oxidation state, ion/species, donor identity, charge, and a prespecified coordination-number-dependent radius. Do not combine U(VI)O2, Th(IV), Am(III), and Ln(III) as interchangeable “f-block” observations. Initially restrict transfer to verified comparable An(III) extraction chemistry. More covalent/selective An interactions need separate domain parameters; equal bond lengths need not imply equal extraction behavior. [Jensen et al., 2002](https://doi.org/10.1021/ja0178620), [C2-POPhen extraction study](https://doi.org/10.1021/acs.inorgchem.8b03592).

### What useful 3D would have to add

Require consistent metal oxidation, ligand identity/count, nitrate/water content, charge, and reference state before interpreting cross-metal strain or energy differences. Composition-changing calculations do not measure one deformation response. Multiple chemically plausible configurations may matter more than tiny differences between lanthanides; computed selectivity remains an auxiliary signal, not an experimental label. [Summers et al., JACS Au](https://doi.org/10.1021/jacsau.4c00770).

Separate free-ligand flexibility, metal-centered structure, and outer-sphere composition. Residualize geometry against radius/metal identity **within training** as a diagnostic of added ligand information. A scalar rescaling of old features and a shuffled-geometry twin must compete with every apparent 3D gain. Previously unavailable binding/strain reference jobs are not equivalent to an experimentally refuted strain hypothesis.

## Data acquisition that could change the result

1. Recover the C5BTBP detection limits and DMDPhPDA 5 M Pr/Nd source value named in the [existing data request](https://github.com/lilshadix/ml_separator/blob/7d02f8e/generations/gen21/next_research_20261019/EXPERIMENTAL_DATA_REQUEST.md). Never infer a detection limit from a flat curve. The audit's +0.041 from dropping one publication is a sensitivity diagnosis, not an authorized performance improvement.
2. Inventory SAFE against the current corpus using DOI, source table/figure, extractant stereochemistry, conditions, and copy groups. [SAFE](https://safe.lanl.gov/) includes multiple process/measurement types and computational records; its headline total is not a count of new eligible experimental extraction labels. [Download documentation](https://safe.lanl.gov/doc).
3. Inspect genuinely new matched-series sources, for example the supporting extraction data of [Green Chemistry D4GC01146E](https://doi.org/10.1039/D4GC01146E). Eligibility, exact condition matches, and overlap remain to be checked; no new usable row count is claimed here.
4. Obtain independent full-series preparations across donor classes, multiple conditions and laboratories, with per-metal detection limits and analytical duplicates. Replicate whole curves across fresh preparations/days; duplicate transcriptions and two readings of one sample do not estimate laboratory repeatability.

Acquisition tables must count **independent sources, extractants, chemotypes, exact-condition series and usable adjacent pairs**, not only rows. Source-compatible An observations may help shared chemistry even if they cannot form an An analogue of the Ln interpolation task; keep their auxiliary labels in separate heads.

## Evaluation and stopping

- Retain historical V1 and its primary metric definition: mean of per-seed pooled strict-adjacent R². Report balanced MAE, RMSE, bias, direction, per-pair R², Pr/Nd, non-sc009, and donor-defined DGA/non-DGA separately. Non-sc009 is not synonymous with non-DGA.
- Use identical supports/query rows for every model contrast. All masks, conformers and metal substitutions from one parent cell/source remain grouped. Fold construction applies to the union of Ln and auxiliary data.
- Split the question into **universal series prior**, **cell-conditioned chemistry**, **3D information**, and **auxiliary transfer**. A gain at one level does not prove the next.
- The round-3 final seeds are consumed. New seeds over these cells assess partition stability. The strain fresh444 are also consumed. Neither creates independent chemical confirmation.
- C3 has 38 nominal chemotypes but about four effective groups under pooled weighting; the report describes 14 connected chemotype/publication clusters. Resample independent clusters, keep repeated seed appearances together, and report small-cluster sensitivity and power. A narrow pairwise or seedwise interval is not evidence of precision.
- Use a finite, sealed registry and nested selection, including zero correction/zero transfer/zero 3D endpoints. Default planning margins in the Claude prompt are +0.01 R² and +0.005 balanced-MAE reduction; these are proposed design choices, not experimentally established process thresholds.
- Call a development gain exploratory. Confirm one frozen candidate once on source-independent data, with a predeclared paired analysis, meaningful-effect criterion, and power assessment. Flag inadequate planned power; judge the observed evidence by the sealed intervals, reporting unresolved effects as INCONCLUSIVE. Do not weaken the endpoint or run new seed batches until significant.
- Keep the 0.50 level as an aspiration distinct from improvement over the stronger baseline. From an illustrative 0.45 reference, reaching 0.50 would require about 9% less squared error on the identical population; a changed cohort cannot supply that claim.

## Handoff

Use `CLAUDE_CODE_PROMPT.md` to implement and execute the staged research. This task created only these two design documents. It did not train models, alter datasets, reveal/reuse final seeds, submit jobs, switch branches, commit, or push.
