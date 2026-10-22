# Claude Code research brief: Gen21 round 4

You are working in `ml_separator` on strict-adjacent lanthanide separation. Analyze the evidence, implement an additive experiment layer, and run the justified research stages to completion within a declared resource budget. Use parallel specialist agents where available for code, provenance, chemistry, and adversarial verification. Your goal is a practically meaningful, statistically defensible improvement over a strong interpolation reference.

Read `generations/gen21/round4_design_20261022/RESEARCH_IDEAS.md` first. It contains verified repository findings, source links, and limitations. Treat proposals as hypotheses. Do not assume that more data, f-shell features, actinides, 3D, or a larger network must help.

## 1. The user's requirements and the stopping contract

1. Avoid data leakage throughout the entire pipeline.
2. Make the experiments scientifically rigorous and reproducible.
3. Work persistently toward a positive and significant result, using the available tools and evidence.

Requirement 3 is a research objective, not permission to select results, change targets, reuse confirmation, or continue testing until a p-value falls below a threshold. Exhaust the registered, evidence-supported search; pursue new independent data when the remaining limitation is information or power. Report negative or inconclusive outcomes honestly. Never promise a positive result.

Do all independent, authorized work before requesting missing inputs. If a source, compute resource, or independent confirmation dataset is unavailable, finish the implementation, valid development experiments, controls, documentation and reproducible resume path that do not depend on it. Do not fabricate data or claim confirmation that did not run.

## 2. Orientation, version reconciliation, and scope

Before edits, read applicable AGENTS instructions and record Git status, branch, HEAD and pre-existing changes. Preserve all frozen generations, datasets, results, existing audits, environments and other user work. Put the new implementation under `generations/gen21/round4_20261022/`; keep the two design files in `round4_design_20261022/` intact.

At brief creation:

- local `ml_separator` HEAD was `9be774c30649f672f365f5586bba10be59ea78c2` on `gen21-logsf-target`;
- round 3 was read remotely at `7d02f8e`, under `generations/gen21/next_research_20261019/`;
- the matching audit is `generations/gen21/AUDIT_ROUND3_2026-10-22.md`;
- local strain HEAD was `7041d51970970afab8653eba822207ce01c773f8`;
- strain remote main was `025f8084b52986773da98fffde13ea267c0ad908`.

Verify current state instead of assuming it is unchanged. If round-3 code is still absent locally, obtain exact pinned source into an isolated reference directory/check-out and record file hashes; never reset/overwrite the user's checkout to make it match. Prefer existing package functions over a rewritten evaluation pipeline. Git mutations, pushes, paid compute and cluster submissions follow actual session authorization; this brief does not grant blanket authorization for them.

Read, in order:

1. Round-3 audit; original Gen21 `PROTOCOL.md`, `ATTEMPTS.md`, and reports.
2. Round-3 protocol, registry, final report, data-quality report and experimental-data request at `7d02f8e`.
3. Gen13 `analysis/stage2/D7/FINDINGS.md`; corrected `runs/gen11_transfer/GEN11_DECISION_REPORT.md`; Gen19 `SUMMARY.md`; Gen15 geometry/energy findings.
4. Strain `automl/topo/anchored_champion.py`, `anchored_3d.py`, `anchored_3d_confirm.py`, `dist_gnn.py`, `fresh_eval.py`; dataset/evaluation condition logic; An/Ln and metal-physics reports; newer `TRANSFORMERS.md` at the pinned remote commit.

Deliver a short evidence inventory before model coding. Explicitly distinguish claims verified from code/results, claims only reported by the audit, and hypotheses.

## 3. Freeze the actual task

Primary target: `logSF(A/B)=log10 D(A)-log10 D(B)`, A lighter, true atomic-number adjacency `|Z_A-Z_B|=1`. Nd/Sm is not adjacent; absent Pm is not an excuse to add the pair. Do not improve R² by including distant pairs, absolute logD, or a different target variance.

Primary historical regime: V1, C3, `k=3`, **four measured logD values** per held-out cell, original DOPT_Q reference policy, original fixed query set. Expected C3 counts: 229 cells, 1,802 queries per seed. Recompute and hash counts/row identities; explain discrepancies before scientific runs. Each scored query must retain at least one hidden endpoint. Read-policy legality may not be achieved by silently dropping difficult queries.

Use BP and BPG as implemented, including exclusions inside every inner fold. Keep zero-shot, other read budgets, POL_ERR/new policies, geometry-only subsets, new data versions and external confirmation under distinct regime labels. A comparison of four versus six readings must use the same eligible cells and query denominator; C6's 133-cell score cannot directly replace C3's 229-cell score.

Preserve the historical metric definition: calculate pooled query R² within each seed, then average seeds. Label pooled-all-OOF or ensemble-prediction R² separately. Preserve negative R² and report undefined R² with a reason. Do not pool repeated predictions as independent observations.

## 4. Strong references before new models

Reproduce PWL_R, PWL_Z, R14 and R02 from exact round-3 sources on the intended development data, checking row identity, support identity, sign, numerical tolerance and metric aggregation. Historical final values may be compared from stored reports, but do not reopen/rerun consumed final seeds.

Build strict references:

- **B0:** unmodified PWL_R, with constant continuation outside the support range and exact measured supports.
- **B1:** PWL_R plus a single Gd amplitude estimated only on the outer training tasks, shrinkage selected only in inner folds. Include amplitude zero / infinite shrinkage explicitly.
- **B2:** a small regularized universal metal-series/read-set prior, independent of cell chemistry, fitted and selected inside the same nesting. Use coherent per-metal potentials rather than unrelated pair constants where feasible. This tests whether a later model adds chemistry beyond a better series prior.
- Retain PWL_Z, HBK@ML, RIDGE2 and FLAT as historical diagnostics when their required inputs are available.

The fixed `-0.072` Gd coefficient originated in a full-corpus Gen13 diagnostic. Report it only as a disclosed historical comparator; never label it external or fold-independent. The audit's approximately 0.45 score is a reference magnitude, not a promised reproduction of a newly defined fit.

Implement the Gd correction consistently when Gd is read: subtract its modeled dip before interpolation and add it back afterward. Copy the logic of round-3 `gd_dip`, not a rule that simply subtracts a constant at an unread Gd.

Before outer scoring, write and seal `PROTOCOL.md`, `EXPERIMENT_REGISTRY.csv`, `DATA_CONTRACT.md`, `METRICS.md`, `RESOURCE_PLAN.md`, environment/input/code hashes and split definitions. Use LF-normalized text hashes and a dated, append-only run record. Archive the seal with an independently verifiable timestamp when an authorized route exists; a local seal alone is not public preregistration. Do not silently amend a sealed artifact. Record every amendment with all results already seen and its effect on confirmatory status.

## 5. Data first: distinguish observations from independent information

Create a provenance inventory joining Ln and auxiliary records by original DOI, table/figure, curator/source record, extractant including stereochemistry, exact conditions and copy groups. Group all masks, conformers, generated metal variants and duplicated transcriptions with their parent records.

Preserve V1 for historical comparability. Any corrected benchmark is a separately frozen V3 with documented source evidence and all models/baselines refitted. Rebuild every upstream prior, covariance and cache on V3; no V1-fit caches inside V3. Do not inherit V2-5's removal of hidden labels based on their own values.

Prioritize source recovery for C5BTBP detection limits and DMDPhPDA 5 M Pr/Nd. Never invent detection limits, impute “correct” values from curve smoothness, or remove rows because of test error. If a source confirms censoring, preserve the observation as an interval and preregister the likelihood and scoring treatment. Do not treat a censored value as an exact label or calculate exact R² against an invented midpoint.

For data expansion, count surviving independent publications/labs, extractants, chemical families, matched-condition series and strict-adjacent pairs **after each fold's exclusions**. Do not count SAFE's total records as new eligible rows. Audit all overlap against both repositories and their training datasets. Computational energies, binding constants and chromatography labels remain separate tasks; do not relabel them experimental extraction logD.

Use the same original Ln test population to measure training-data expansion. A new expanded evaluation population is a separate benchmark with its own baselines. Validation already used to decide which records to recover is not untouched confirmation.

Build training-only learning curves versus independent source/chemistry count, plus replicate/condition diagnostics. Preserve correlation across metals of a preparation. Do not use the old 0.302 logD spread as a measured laboratory noise ceiling, and do not treat copied measurements as independent repeats.

## 6. Model design: learn only what interpolation misses

Use a common support-preserving interface. For a legal read set S, radius interpolation `I_S`, Gd basis h and train-fitted beta:

```text
base(m) = I_S[y_S - beta*h_S](m) + beta*h(m)
correction(m) = g_theta(m, allowed cell metadata, allowed support summaries)
prediction(m) = base(m) + correction(m) - I_S[correction(S)](m)
predicted_logSF(A/B) = prediction(A) - prediction(B)
```

The correction is zero at supports. `correction(S)` contains predictions, not additional labels. Train directly on hidden strict-adjacent differences under training masks that reproduce the evaluation task. Normalize weights so more masks/pairs/conformers do not multiply an original cell's effective weight. Keep every mask of a cell in its parent split.

This is a proposed model, not a mandatory assumption of superiority. Include exact zero correction, simple linear/hierarchical residual baselines, and B2. If measured supports are denoised rather than kept exact, register a separate observation contract and compare its own baselines.

### Electronic interactions

Use verified ionic configurations and oxidation states. For ordinary Ln(III), `n_f=Z-57`; it is redundant information with Z. Test a small prespecified basis and donor/condition interactions, not a claim that electron count increases sample size. Compare against an equally flexible Z/radius basis, the Gd reference, donor ablation and permutation of the electronic mapping. Generic f count/tetrad features have already been tried in both repositories.

Use charge/CN-consistent external radii. No coordination number, term symbol, effective charge or geometry may be chosen by fitting a held-out metal's extraction response. A Gd-centered effect does not by itself establish a half-filled-shell mechanism.

### Strain architecture adaptation

Port the **level/shape decomposition and scalar blending**, not its old point score or hyperparameters. Train a small CatBoost correction using 2D ligand, conditions, support values/slopes and interpolation coordinates. Then test compact geometry descriptors and a distance encoder only if their inputs exist with usable provenance.

The strain architecture is not “OOF embeddings into CatBoost.” Supervised weights from the overlapping SAFE corpus are forbidden unless their full lineage proves exclusion of the present held-out data. Freezing contaminated weights does not remove leakage. Use a genuinely independent target-free pretrained initialization or fold-local training.

All centering of training targets uses training labels only. Center predictions at inference; never center by a held-out full-curve mean. Train one consistent encoder basis within each outer fold if using embeddings. For blending, generate inner OOF scalar predictions with every constituent fitted solely on that inner training set; then fit weights including zero. No full-dataset OOF shortcut.

Use physically comparable composition, charge, ligand count, nitrate/water content and energy reference for cross-metal geometry/strain features. Separate single-conformer geometry from ensembles. Freeze generation/QC rules without extraction-label access; preserve failures and use the reference fallback on missing geometry. Evaluate the full cohort, reporting a secondary coverage-matched subset. Do not manufacture condition-dependent coordinates or mix computed energies with experimental labels.

### Actinide transfer

Start with verified An(III) records in compatible donor/solvent/speciation regimes. Keep actinyl species and other oxidation states separate. The previous Gen11/19 and strain An experiments did not establish a Ln benefit.

Use a common Ln-only multitask scaffold before adding An: a selectivity-residual head plus a raw-logD auxiliary head trained on training-Ln data. Instantiate the same shared/private-head architecture in all arms, including zero-An. Preserve the auxiliary target definition, loss normalization and tuning budget when extra Ln or An is added; give the An domain its private head. Permit the auxiliary weight to be zero. If an An source supplies no matched series, do not fabricate an An adjacent-curve target.

Compare selective transfer against that same scaffold, extra-Ln training, a deliberately naive shared raw-logD head, source/diversity/weight-matched auxiliary controls, and block-permuted An targets. Hold Ln rows, their weights, targets and optimization budget identical in each An-versus-no-An contrast; extra-Ln is a separate data-expansion arm. Equal row counts alone are inadequate. Apply ligand/chemotype/publication/copy exclusions across the union of both domains before pretraining, fitting or choosing weights. An gains never substitute for gains on held-out Ln logSF.

## 7. Finite staged registry

Maximum **24 new outer-scored configurations** across this campaign, including selectable variants, controls, combinations, policies and data versions. Baseline reproductions on the identical task do not consume slots; they are not hidden tuning. Fix exact narrow inner grids, representations, arms and stage gates before scoring. Count every outer-viewed variant, including failed or off-registry runs. No post-result budget extension to chase significance.

Use the following proposed allocation; refine exact technical choices before sealing, preserving all required contrasts:

| Slots | Stage | Registered contrasts |
|---|---|---|
| 01-04 | Small residual/electronic hypotheses | Hierarchical residual model; electronic-by-chemistry interactions; matched-capacity flexible Z/radius control; electronic-map permutation control. |
| 05-10 | Strain adaptation | CatBoost tabular correction; compact geometry correction; distance-encoder correction; nested scalar blend; block-shuffled-geometry control of the selected geometry route; random-encoder control with matched head capacity. |
| 11-16 | Data/transfer | Common Ln-only multitask scaffold; extra independent Ln; compatible An(III) shared/private model; naive shared-head control; source/diversity/weight-matched auxiliary control; block-permuted An-target control. |
| 17-18 | Source quality/noise | Preselected method on independently source-corrected V3; source-supported robust/censored variant if meaningful exact/interval evaluation is defined. |
| 19-20 | Conditional combinations/deployment | One preregistered combination of independently supported components; one new-policy crossing experiment with reference/predictor scored on both policies. |
| 21-24 | Explicit additional controls | Donor-interaction ablation of the electronic model; radius-only geometry control; training-fitted rescaling/shrinkage competitor using simpler tabular/radius predictions; metal-state-mapping shuffle for the selective-transfer model. |

If data or geometry do not support a stage, mark its slots NOT RUN with the reason; do not invent information. Controls chosen for a selected route are evaluated as its diagnostic controls, not treated as new winners. Register any predeclared conditional rule that chooses their route using inner-only evidence or already labeled development results. Repeated draws of one prespecified randomized-control distribution estimate that test, rather than become selectable new variants. Additional outer-scored controls require a slot substitution before scoring, inside the same cap. Subgroup summaries of the same predictions do not consume slots; fitting subgroup-specific models does. Restrict slot 19 to a combination whose constituent ablations already exist, or substitute slots for those ablations before running it. No unrecorded pilot on outer labels.

Before expensive encoder training, require useful input coverage and a cheap inner-fold signal over B1/B2; use a synthetic technical smoke for shape, gradients, masks and recovery. Do not call technical overfitting capacity a scientific result. Small matrices/controls can run in parallel within memory/CPU limits; independent model seeds measure optimizer stability, not independent chemistry.

Estimate actual costs, record a finite wall/CPU/GPU budget, checkpoint and resume. Do not submit cluster work or spend money absent session authorization. If the budget cannot cover every stage, prioritize data provenance, strict references, residual/electronic arms and their controls before geometry/transfer escalation. Save all NOT RUN statuses and ready resume commands.

## 8. Leakage and scientific controls

Every retained candidate must pass the actual end-to-end chain on every relevant outer fold/design:

1. **Isolation:** disjoint permitted cells/ligands/chemotypes/publications/copies, including auxiliary data, all masks, generation parents and every inner fit. Produce machine-checkable overlap tables.
2. **Fit poisoning:** randomize all nontraining labels, including frame labels, cached coefficients and derived target fields. Learned parameters, transforms, selected hyperparameters and permitted caches must be unchanged. Preserve legal observed support values only when testing query invariance.
3. **Cache poisoning:** rebuild from poisoned hidden labels with caches bypassed, then compare allowed fields, supports and predictions. Cache keys include data version/content, code, split/design, support contract, feature state and hyperparameters.
4. **Prediction intervention:** change every unread held-out label, rebuild the allowed path, and require unchanged predictions/read sequences. Use a positive control that deliberately exposes an unread value to prove the test detects access.
5. **Support consistency:** exact supports, unchanged fixed query denominator, no scored pair with two measured endpoints, correct sign/adjacency, and antisymmetry/cycle consistency.
6. **Claim-specific falsification:** retain a series/read-set prior while block-shuffling cell-to-chemistry assignments to test conditional learning; use electronic-map placebos for electronic claims, geometry shuffles/random controls for geometry, and An-label permutation/auxiliary controls for transfer. Keep repeated masks/curves together and preserve the relevant nuisance distributions. Rerun inner tuning under each randomized control.
7. **Calibration control:** compare same-data, training-only rescaling/shrinkage of simpler predictions against any apparent R² gain. Report raw R², RMSE, bias, prediction variance and Pearson² as a diagnostic. Test-fitted optimal rescaling is an explicitly labeled oracle, never a deployable baseline or promotion score.
8. **Discriminating power:** demonstrate on synthetic null and known-signal tasks that the control tests its intended claim, including a zero-gain model. Report false-positive/power estimates and Monte Carlo uncertainty. Do not reuse the round-3 shuffle as a universal gate or apply a powerless test to reject a genuine universal prior.

A placebo-position scan is diagnostic and does not authorize choosing the best metal position on test labels. Leave-pair-type-out tests a new transfer question and must be declared separately from ordinary cell/chemotype generalization.

## 9. Statistical analysis and promotion

Keep distinct labels for **technical smoke**, **exploratory development**, **partition stability**, and **independent confirmation**. Round-3 final seeds and strain fresh444 are consumed. New seeds over the same chemistry cannot restore a blind scientific test.

After each experiment produce a compact table containing: ID/data/design/policy/read budget; independent cluster and pair counts; R²; balanced MAE; RMSE; paired gains over B1 and B2; uncertainty; geometry/auxiliary coverage; leakage/control status; compute; and decision. Add a short explanation of what changed scientifically and the next registered step.

Primary metric remains historical per-seed pooled R². Define balanced MAE explicitly before runs, using a fixed cell/extractant/chemotype weighting compatible with existing practice; do not silently select a favorable balancing scheme. Report per-pair results, Pr/Nd, Eu/Gd and Gd/Tb, other pairs, donor-defined DGA/non-DGA, sc009/non-sc009, publications and one-end-read/both-hidden separately.

Cluster uncertainty by independent sources/chemistry with the declared cross-publication-copy connectivity. Show nominal and effective cluster counts, leave-one-major-group-out sensitivity, percentile/BCa and small-cluster-robust sensitivity where valid. C3's 38 nominal chemotypes have about four effective groups under pooled weighting; graph-connected chemistry/publication clusters were 14 in round 3. Do not claim precision from thousands of correlated pairs or from seed standard errors. State when too few effective clusters make asymptotic inference unreliable.

Use development for selection and uncertainty diagnostics; selection makes its best contrast optimistic. Select a single finalist by a sealed rule. If multiple confirmatory claims/finalists are genuinely tested, predeclare the family and use Holm or simultaneous intervals; all-conditions-required co-primary endpoints use an appropriate intersection-union rule. Do not treat BP and BPG as independent replications.

Default planning criteria, to be sealed before runs:

- meaningful improvement over B1: point ΔR² at least **+0.01** and balanced-MAE reduction at least **0.005 log10 units**;
- evidence for cell-conditioned learning additionally requires improvement over B2 and its relevant falsification controls;
- source-independent confirmation requires paired two-sided **95% confidence intervals** excluding zero in the favorable direction for both claimed improvements, with **alpha=0.05** for the intersection-union decision on the two required endpoints; apply the declared multiplicity adjustment if there are additional confirmatory claims, and require no material predeclared subgroup harm;
- predefine the subgroup safeguard numerically using training-only information; do not demand significance in tiny subgroups or conceal a failed Pr/Nd slice under pooled success;
- R² at least 0.50 is an additional aspirational level claim, evaluated on unrounded values and supported by its own uncertainty, not a substitute for the paired improvement test.

The +0.01/0.005 margins are explicit proposal defaults, not measured noise/process limits. If the scientific target requires other margins, justify and seal them before any new outer result. Never relax them after seeing a failure.

Perform power/MDE analysis using independent cluster structure and plausible effect injection into training/development tasks before confirmation. Distinguish power to reject nonpositive gain from the probability of passing the full promotion gate. At a true effect equal to a point-estimate threshold, crossing that threshold can remain near 50% even with high precision. Target at least 80% power for rejecting nonpositive gain at the planning margins, and separately simulate the complete gate at zero, at the margins, and at prespecified plausible effects above them; report Monte Carlo uncertainty. If planning power is inadequate, label the study underpowered and quantify additional independent sources/curves needed. Judge any observed confirmation by the sealed intervals and meaningful-effect criteria; classify unresolved intervals as INCONCLUSIVE. Low prospective power alone does not invalidate an actually precise favorable result. Do not turn a post-hoc bootstrap tail fraction into a posterior probability of a true R² threshold.

Freeze predictor, priors, representations, support policy, data inclusion, analysis and hashes before new external labels arrive. An evaluator should expose only the four legal support readings to prediction and reveal hidden labels only after predictions are sealed. Run the primary confirmation once. An error-triggered rerun requires a documented code defect, unchanged scientific choices and a full disclosure; it does not create a fresh holdout.

## 10. Outputs and completion

Produce under `generations/gen21/round4_20261022/`:

- evidence/inventory, frozen protocol/registry/contracts/resource plan, input and code manifests;
- additive implementation with focused tests for task construction, leakage, projection/support invariance, synthetic null/signal recovery and cache identity;
- row-level OOF predictions and supports, fitted parameters, inner-selection records, preprocessing/encoder provenance, full run/control logs and restartable commands;
- machine-generated result/contrast tables, development-versus-confirmation labels, all attempted/failed/NOT RUN arms and budget accounting;
- `REPORT.md` answering what improves interpolation, whether f-shell/3D/An adds incremental information, where it fails, practical separation implications and remaining uncertainty;
- `DATA_REQUEST.md` for source recovery, independent whole-curve replicates and fresh chemistry if needed; `RESUME.md` for any externally blocked work.

Report the strongest reproducible model even if it is B1 or B2. A legitimate positive result may be a better physical/series prior or corrected data, but label its contribution accurately; do not call it learned chemistry, 3D information, or actinide transfer without the corresponding contrast.

Keep moving through the justified stages. Stop the current inferential campaign when the registered budget is exhausted or its gates close. Further progress then requires a newly justified plan or new independent information, not another look at the same confirmation data. Finish with a result table and a concise evidence-based verdict: demonstrated improvement, exploratory gain, no detected gain, or insufficient power/data.
