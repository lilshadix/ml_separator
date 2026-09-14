# GEN19 — Chemistry Transfer Engine
## Technical implementation brief for Claude Code

### Mission

Build **Gen19** as a chemistry-transfer system for solvent extraction.

The purpose is **not** to obtain another high random-split R² score.

The purpose is to test the following scientific hypothesis:

> **Can unmeasured metal–extractant combinations be reconstructed from shared structure across metals, extractants, mechanisms, conditions, and publications, with uncertainty propagated into multistage process design?**

Gen19 must use the expanded extraction corpus, including lanthanides, actinides, transition metals, other metals, multiple extractant families, and all available experimental conditions.

The central applied target remains rare-earth separation, especially **Pr/Nd**, but Gen19 must learn from the entire corpus.

Do not treat each metal–extractant pair as an isolated prediction task.

The required conceptual model is:

```text
metal properties
      ×
extractant properties
      ×
chemical mechanism
      ×
experimental conditions
      ×
publication/source effects
      ↓
distribution coefficient logD
      +
prediction uncertainty
      ↓
pairwise selectivity / logSF
      ↓
Gen18 process simulator
      ↓
purity / recovery distributions
      ↓
robust process optimization
```

---

# 1. Non-negotiable principles

Gen19 must obey the following rules.

## 1.1 No random-split success claims

A random train/test split may be reported only as a diagnostic.

It must never be the main performance claim.

Primary validation must test transfer to chemistry that was deliberately hidden.

---

## 1.2 No silent extrapolation

Every prediction must receive a domain-status label:

- `IN_DOMAIN`
- `INTERPOLATION`
- `CROSS_METAL_TRANSFER`
- `CROSS_LIGAND_TRANSFER`
- `CROSS_METAL_LIGAND_TRANSFER`
- `FAMILY_EXTRAPOLATION`
- `CONDITION_EXTRAPOLATION`
- `UNSUPPORTED`

The process optimizer must not treat these categories equally.

---

## 1.3 No prediction without uncertainty

Gen19 must not return only:

```text
logD = 0.82
```

It must return something equivalent to:

```text
mean_logD
std_logD
lower_95
upper_95
domain_status
support_score
nearest_support
```

Uncertainty must increase for weakly supported transfer and extrapolation.

---

## 1.4 Preserve chemistry provenance

Every experimental record must retain:

- publication/source ID
- DOI if available
- table/figure/page if available
- metal
- oxidation state
- extractant
- extractant family
- extractant structure identifier
- diluent
- acid
- aqueous composition
- extractant concentration
- initial metal concentration
- equilibrium/final pH if available
- acidity if available
- O/A
- temperature
- contact time
- saponification
- modifiers
- complexants
- loading
- measured D
- whether D is directly reported or reconstructed
- uncertainty if reported
- data-status / quality flag

Never discard source identifiers during preprocessing.

---

## 1.5 Gen15 must not become hidden ground truth

Gen15 or any previous ML model may be used only for:

- representation initialization
- ranking hints
- feature generation
- qualitative priors

It must never supply synthetic D values that are treated as experimental truth.

---

## 1.6 Gen18 remains the process layer

Do not rewrite the verified Gen18 cascade solver unless a real numerical bug is found.

Gen19 should primarily replace/improve the **chemical prediction layer** feeding Gen18.

Gen18 validated process arithmetic should remain isolated and regression-tested.

---

# 2. Core scientific idea

The central assumption to test is that extraction chemistry has reusable latent structure.

Instead of:

```text
(metal_name, extractant_name, conditions) -> logD
```

model:

```text
metal_representation
× extractant_representation
× mechanism_representation
× conditions
× source hierarchy
-> logD distribution
```

The model must be able to learn facts such as:

- behavior shared across trivalent lanthanides
- smooth trends along the lanthanide series
- similarities and differences between lanthanides and actinides
- extractant-family behavior
- differences between acidic and neutral extractants
- systematic source/publication offsets
- condition-dependent slopes
- metal–ligand interaction residuals

---

# 3. Target mathematical structure

Start with the following conceptual decomposition:

\[
\log D_{m,l,c,s}
=
B_k(c)
+
I(m,l)
+
R(m,l,c)
+
S_s
+
\epsilon
\]

where:

- `m` = metal
- `l` = ligand/extractant
- `c` = conditions
- `s` = publication/source
- `k` = chemical mechanism/extractant family
- `B_k(c)` = mechanism-specific base behavior
- `I(m,l)` = reusable metal–ligand affinity
- `R(m,l,c)` = nonlinear residual response to conditions
- `S_s` = publication/source effect
- `epsilon` = residual noise

The implementation does not have to use this exact formula internally, but it must preserve these conceptual factors.

---

# 4. Required representations

## 4.1 Metal representation

Build a metal descriptor table.

At minimum investigate:

- atomic number
- group / period where meaningful
- oxidation state
- formal ionic charge
- ionic radius for the relevant coordination/oxidation state
- electronegativity
- polarizability
- hard/soft acid descriptors
- electron configuration descriptors
- f-electron count
- lanthanide / actinide / transition / other category
- redox-relevant information where available

Do not blindly use all descriptors.

Record missingness and provenance.

For lanthanides and actinides, explicitly support series-aware features.

---

## 4.2 Extractant representation

Each extractant should have:

- canonical name
- aliases
- family
- acidic / neutral / basic classification
- donor atoms
- donor atom counts
- denticity if meaningful
- molecular weight
- molecular descriptors
- fingerprint
- SMILES where known
- pKa or acidity information if known
- lipophilicity descriptors where available
- mechanism label if supported

Examples of families:

- phosphoric acids
- phosphonic acids
- phosphinic acids
- diglycolamides
- monoamides
- malonamides
- beta-diketones
- organophosphorus neutral solvating ligands
- ionic extractants
- synergistic systems
- other

Do not collapse PC88A, D2EHPA, Cyanex 272, TODGA, etc. into mere categorical IDs.

---

## 4.3 Condition representation

Normalize and preserve:

- acid identity
- acid molarity
- pH
- extractant molarity
- aqueous metal concentration
- organic loading
- O/A
- diluent
- modifier
- aqueous complexant
- temperature
- saponification
- contact time
- ionic strength where recoverable

Do not assume pH and acid molarity are interchangeable.

Do not assume records from different acid systems belong to the same local response surface.

---

# 5. Mechanistic experts

Implement a **Mixture of Chemical Experts** or equivalent mechanism-aware architecture.

At minimum support separate experts or separate parameterizations for:

1. acidic cation-exchange extractants
2. neutral solvating extractants
3. ion-pair / basic extractants if present
4. chelating extractants if present
5. synergistic/two-ligand systems if sufficient data exist

The router may use:

- known extractant-family labels
- mechanistic metadata
- learned gating

But known chemistry should take priority over purely learned routing where the family is already established.

---

# 6. Physics-informed constraints

Gen19 must not be a pure black box.

Where chemically justified and only within appropriate domains, test soft constraints such as:

### Acidic extractants

Approximate mass-action structure:

\[
\log D \sim \log K + n_L \log[L] + n_H pH
\]

or equivalent acid-concentration formulation.

Possible soft constraints:

```text
d(logD)/d(pH) >= 0
d(logD)/d(log[extractant]) >= 0
```

ONLY enforce where chemically justified.

Do not force monotonicity outside the experimentally supported regime.

Do not impose one global slope across all metals or extractants.

---

### Loading

Retain Gen18's loading correction as a candidate feature/model component.

But Gen19 must re-evaluate it under broader data.

Do not assume R2 remains valid globally.

---

### Smoothness along chemical series

Test a soft smoothness prior for lanthanides:

```text
neighboring Ln elements should generally have related latent representations
```

Do not force logD itself to be monotonic across the entire series.

Allow anomalies.

---

# 7. Metal–ligand interaction model

Implement an explicit reusable interaction term.

A recommended baseline is:

\[
e_m = g_m(z_m)
\]

\[
e_l = g_l(z_l)
\]

\[
I(m,l) = e_m^T W e_l
\]

or a small nonlinear interaction network.

The important requirement is that:

- metal information is reusable across extractants
- extractant information is reusable across metals
- the model can predict a missing metal–extractant cell

This is the central Gen19 hypothesis.

---

# 8. Multitask learning

Train on both absolute extraction and selectivity.

## Absolute task

\[
L_D =
\left|
\hat{\log D}
-
\log D
\right|
\]

Use MAE/Huber as primary candidates.

---

## Pairwise task

For two metals measured under compatible conditions:

\[
\log SF_{A/B}
=
\log D_A
-
\log D_B
\]

Train:

\[
L_{pair}
=
\left|
(\hat{\log D_A}-\hat{\log D_B})
-
(\log D_A-\log D_B)
\right|
\]

Generate pairs only when the experimental conditions are legitimately comparable.

Do not pair arbitrary rows.

---

## Composite objective

Start from:

\[
L =
\lambda_D L_D
+
\lambda_{pair}L_{pair}
+
\lambda_{physics}L_{physics}
+
\lambda_{source}L_{source}
+
\lambda_{reg}L_{reg}
\]

All lambda values must be tuned only inside the training-validation structure.

Do not tune on the final holdout.

---

# 9. Publication/source hierarchy

Implement source-aware modeling.

At minimum compare:

### Model A
No source term.

### Model B
Fixed publication/source embedding.

### Model C
Hierarchical/random source effect.

Preferred conceptual form:

\[
\log D = F(m,l,c) + b_{source}
\]

At prediction time for a new publication/system:

```text
source effect = population prior
```

not an arbitrary learned offset.

Measure whether source modeling reduces cross-publication discontinuities.

---

# 10. Uncertainty

Uncertainty is mandatory.

Compare practical approaches such as:

- deep ensembles
- bootstrap ensembles
- heteroscedastic regression
- quantile regression
- conformal prediction
- Bayesian linear/hierarchical heads
- Monte Carlo dropout only as a secondary baseline

Evaluate both:

1. sharpness
2. calibration

Required plots/metrics:

- empirical 50% interval coverage
- empirical 80% interval coverage
- empirical 95% interval coverage
- interval width vs transfer distance
- error vs predicted uncertainty
- calibration by domain-status category

A model with slightly worse MAE but well-calibrated uncertainty may be preferred.

---

# 11. Required validation suite

This is the most important section.

Implement all tests below before claiming success.

---

## V0 — Random split

Purpose:

```text
debugging only
```

Report metrics but do not use for scientific claims.

---

## V1 — Leave-publication-out

Entire publications must be excluded from training.

Question:

> Can the model transfer across sources rather than memorize publication-specific chemistry?

Report:

- MAE(logD)
- RMSE(logD)
- R²
- Spearman
- interval coverage
- pairwise rank accuracy

---

## V2 — Leave-metal-out

Hide one metal completely.

Examples:

```text
train: all metals except Nd
test: Nd
```

Repeat across sufficiently represented metals.

Primary rare-earth focus:

- Pr
- Nd
- Sm
- Eu
- La
- Ce
- Gd

Question:

> Can the system reconstruct an unseen metal from periodic/chemical structure?

---

## V3 — Leave-extractant-out

Hide one extractant entirely.

Examples:

```text
train: everything except PC88A
test: PC88A
```

Do this only where descriptors/family information allow meaningful transfer.

Question:

> Can molecular and family descriptors support transfer to a new extractant?

---

## V4 — Leave-family-out

Hide an entire extractant family.

This is intentionally difficult.

Example:

```text
train: no phosphinic-acid systems
test: phosphinic-acid systems
```

The expected result may be poor.

That is acceptable.

This test defines the boundary of the model.

---

## V5 — Missing metal × extractant cell

THIS IS THE CENTRAL TEST.

For a selected pair:

```text
PC88A × Nd
```

remove every direct measurement of that combination.

Training may still contain:

```text
PC88A × La
PC88A × Ce
PC88A × Pr
PC88A × Sm
PC88A × Eu
...
```

and:

```text
Nd × D2EHPA
Nd × Cyanex 272
Nd × TODGA
...
```

Test whether the model reconstructs the hidden:

```text
PC88A × Nd
```

Repeat over many sufficiently supported cells.

This evaluates matrix-completion-style chemistry transfer.

---

## V6 — Pr/Nd double-cell holdout

Harder test.

Remove direct data for:

```text
PC88A × Pr
PC88A × Nd
```

or equivalent candidate extractant.

Ask whether the model correctly recovers:

- D_Pr
- D_Nd
- direction of selectivity
- logSF_Nd/Pr
- uncertainty

This is the most relevant test for the applied process case.

---

## V7 — Condition extrapolation

Hide a contiguous condition region.

Examples:

- high acidity
- low acidity
- high loading
- high extractant concentration

Do not randomly remove individual points.

Question:

> Does the model interpolate/extrapolate smoothly across conditions?

---

# 12. Anti-leakage rules

This project is highly vulnerable to leakage.

Before training, create an automated leakage audit.

Check:

- duplicate rows
- near-duplicate rows
- same experiment copied across multiple papers
- same table digitized twice
- alias collisions
- same DOI with different source IDs
- identical conditions and values in train/test
- derived pair rows crossing train/test boundaries
- ligand aliases split across folds
- metal aliases split across folds
- source metadata leaking target values

Pair generation must occur **after** fold assignment unless proved safe.

All preprocessing fitted to data must be fitted on train only.

This includes:

- scaling
- imputation
- learned embeddings
- feature selection
- PCA
- uncertainty calibration
- source priors

---

# 13. Baselines

Every proposed architecture must beat meaningful baselines.

Required baselines:

### B0
Global mean.

### B1
Metal mean.

### B2
Extractant mean.

### B3
Nearest-condition lookup from Gen18.

### B4
Nearest-condition same-source-excluded lookup.

### B5
Tree model using engineered descriptors.

Examples:

- CatBoost
- LightGBM / XGBoost if already available

### B6
Simple metal × extractant factorization.

### B7
Mechanism-aware mass-action model where applicable.

### B8
Previous best Gen-series model, evaluated under the exact same strict folds.

Do not compare a new model under hard holdout against an old model's random-split score.

---

# 14. Model progression

Do not jump immediately to a large neural model.

Use staged ablations.

Recommended sequence:

### M0
Descriptor CatBoost baseline.

### M1
Metal embedding + ligand embedding + conditions.

### M2
Explicit bilinear metal–ligand interaction.

### M3
Mechanism-specific experts.

### M4
Source hierarchy.

### M5
Pairwise multitask loss.

### M6
Physics-informed penalties.

### M7
Uncertainty ensemble.

At every step produce an ablation table.

If added complexity does not improve strict validation, remove it.

---

# 15. Transfer from actinides

Actinides may provide valuable information, but transfer must be structured.

Do NOT assume:

```text
Am(III) behavior == Nd(III) behavior
```

Instead use shared + series-specific structure:

\[
e_m
=
e_{shared}
+
e_{series}
+
e_{oxidation}
+
e_{element}
\]

Possible categories:

```text
lanthanide
actinide
transition
post-transition
other
```

Run an explicit ablation:

```text
Model WITHOUT actinide data
vs
Model WITH actinide data
```

Evaluate on hidden lanthanide targets.

If actinides do not improve lanthanide transfer, do not claim they help.

If they hurt, investigate negative transfer.

---

# 16. Chemistry support graph

Build an explicit support graph.

Nodes:

```text
metals
extractants
extractant families
conditions/systems
publications
```

Edges:

```text
measured experiment
shared family
shared metal
shared publication
shared condition regime
```

For each requested prediction calculate support information such as:

- number of direct neighboring metals
- number of same-family observations
- nearest ionic-radius distance
- nearest ligand descriptor distance
- number of supporting publications
- condition-space distance
- whether the exact pair exists
- whether neighboring series elements exist

Use this to derive a `support_score`.

The support score must be visible to Gen18/optimizer.

---

# 17. Process integration

Once the chemistry predictor passes the transfer benchmarks, connect it to Gen18.

Do not feed only one deterministic D.

For each operating point:

```text
sample D distributions
    ↓
run Gen18 cascade
    ↓
obtain distribution of:
    purity
    recovery
    reagent use
    loading
    feasibility
```

Use Monte Carlo or an efficient surrogate if needed.

Primary process outputs must include:

```text
median purity
median recovery
5/50/95 percentile purity
5/50/95 percentile recovery
probability(purity >= target)
probability(recovery >= target)
probability(both targets satisfied)
probability(phase/loading constraints satisfied)
```

---

# 18. Robust optimization objective

Do not optimize nominal purity alone.

Recommended objective:

\[
\max_x
P(
Purity(x) \ge P_{min}
\land
Recovery(x) \ge R_{min}
\land
Feasible(x)
)
\]

Then optimize reagent use / stage count as secondary objectives.

Example lexicographic ranking:

1. probability of meeting purity + recovery
2. process feasibility probability
3. reagent consumption
4. number of stages
5. throughput

Do not allow an `UNSUPPORTED` chemistry prediction to become the winning process recipe.

---

# 19. Applied Pr/Nd benchmark

After the transfer engine is validated, rerun Pr/Nd.

Candidate extractants should include any sufficiently supported systems in the corpus, especially:

- PC88A / P507
- Cyanex 272
- D2EHPA / P204
- TODGA

Do not force all four into the final comparison if one is unsupported.

For each extractant report:

```text
direct data count
same-family data count
Pr direct support
Nd direct support
neighboring-Ln support
actinide support
condition coverage
domain status
uncertainty
best robust process result
```

The final applied claim should distinguish:

```text
directly supported
transfer-supported
speculative
unsupported
```

---

# 20. Required outputs

Create a new directory:

```text
generations/gen19_chem_transfer/
```

Recommended structure:

```text
generations/gen19_chem_transfer/
├── README.md
├── GEN19_REPORT.md
├── SUMMARY.md
├── preregistration.md
├── data_audit/
├── descriptors/
├── folds/
├── models/
├── evaluation/
├── process/
├── figures/
├── tables/
├── manifests/
└── src/
```

---

# 21. Preregistration

Before fitting the final models, write and seal:

```text
preregistration.md
```

It must specify:

- primary hypothesis
- primary metrics
- primary holdouts
- success thresholds
- baselines
- allowed model families
- tuning procedure
- uncertainty evaluation
- process evaluation
- failure conditions

Hash the preregistration.

After sealing, any change to the registered primary analysis must be explicitly documented as post-hoc.

---

# 22. Suggested success criteria

Do not hard-code these blindly.

First calculate baseline difficulty, then finalize thresholds before final training.

But success should require something like:

### Scientific success

On `V5 metal × extractant holdout`:

- meaningful improvement over Gen18 lookup
- correct selectivity direction substantially above chance
- calibrated uncertainty
- error increasing sensibly with support distance

### Strong success

On `V6 Pr/Nd double holdout`:

- correct sign of logSF
- useful magnitude estimation
- uncertainty interval covering truth at the registered rate
- process recommendation remains stable under uncertainty

### Failure

Gen19 should be considered unsuccessful if:

- gains exist only on random split
- strict transfer folds collapse
- uncertainty is badly miscalibrated
- actinide transfer creates negative transfer
- process optimum depends on unsupported predictions
- architecture complexity adds no strict-fold benefit

A clean null result is acceptable.

Do not hide it.

---

# 23. Experiment tracking

Every experiment must record:

```text
git commit
dataset hash
fold hash
feature version
model version
random seed
hyperparameters
training date
metrics
hardware
runtime
```

Write machine-readable JSON/CSV summaries.

Do not rely only on markdown reports.

---

# 24. Reproducibility

Fix the Gen18 manifest portability problem.

Hash normalized file content where line-ending differences should not matter, or explicitly record both binary and normalized hashes.

Do not allow CRLF/LF differences to appear as scientific reproducibility failures.

---

# 25. Diagnostics and figures

Required figures should include:

1. metal × extractant observation matrix
2. data density by metal
3. data density by extractant family
4. lanthanide coverage map
5. actinide coverage map
6. condition-space coverage
7. predicted vs measured under each strict holdout
8. error vs support score
9. uncertainty calibration
10. learned metal embedding visualization
11. learned extractant embedding visualization
12. Pr/Nd hidden-cell reconstruction
13. selectivity-direction confusion matrix
14. process purity–recovery Pareto front with uncertainty
15. probability-of-specification process map

Do not use decorative figures.

Every figure should answer a scientific question.

---

# 26. Coding requirements

Use modular code.

Suggested modules:

```text
src/
├── data/
│   ├── load.py
│   ├── normalize.py
│   ├── provenance.py
│   └── leakage.py
├── chemistry/
│   ├── metals.py
│   ├── ligands.py
│   ├── mechanisms.py
│   └── support_graph.py
├── folds/
│   ├── source_holdout.py
│   ├── metal_holdout.py
│   ├── ligand_holdout.py
│   ├── family_holdout.py
│   └── cell_holdout.py
├── models/
│   ├── baselines.py
│   ├── factorized.py
│   ├── experts.py
│   ├── source_effects.py
│   └── uncertainty.py
├── losses/
│   ├── absolute.py
│   ├── pairwise.py
│   └── physics.py
├── evaluation/
│   ├── metrics.py
│   ├── calibration.py
│   └── transfer.py
└── process/
    ├── gen18_adapter.py
    ├── monte_carlo.py
    └── robust_optimize.py
```

Keep dependencies controlled.

Prefer transparent models and ablations over unnecessary complexity.

---

# 27. Tests

Write unit and integration tests before trusting results.

Required tests:

- descriptor consistency
- metal alias normalization
- ligand alias normalization
- fold isolation
- no source leakage
- no duplicate crossing folds
- pair-generation isolation
- support-score correctness
- uncertainty output shape
- process Monte Carlo reproducibility
- Gen18 regression tests
- mass-balance invariance
- manifest verification

A test must explicitly confirm that a hidden metal × ligand pair is absent from the training set.

---

# 28. Immediate first-pass tasks

Do these in order.

## Phase A — Corpus audit

1. Locate the expanded dataset.
2. Compute dataset hash.
3. List all columns.
4. Count rows.
5. Count unique:
   - metals
   - oxidation states
   - extractants
   - extractant families
   - publications
   - acids
   - diluents
6. Build the metal × extractant matrix.
7. Quantify sparsity.
8. Quantify Pr/Nd coverage.
9. Quantify lanthanide coverage.
10. Quantify actinide coverage.
11. Find alias collisions.
12. Find duplicates.
13. Find near-duplicates.
14. Identify which metadata required above are actually available.

Do not train anything before this audit is complete.

---

## Phase B — Feasibility analysis

Before model development answer:

1. How many metal × extractant cells have enough data for V5 testing?
2. How many extractants span >= 5 metals?
3. How many metals span >= 5 extractants?
4. Which extractant families have enough coverage?
5. Is PC88A present?
6. Is Cyanex 272 present?
7. Is D2EHPA present?
8. Is TODGA present?
9. How much Pr data exists?
10. How much Nd data exists?
11. Which neighboring lanthanides exist under the same extractants?
12. Which actinide systems overlap with the same extractants?
13. Are conditions rich enough to disentangle metal effects from condition effects?

Write:

```text
FEASIBILITY.md
```

Do not proceed blindly if the data graph is disconnected.

---

## Phase C — Baseline transfer benchmarks

Implement B0–B8.

Run V1, V2, V5 first.

This establishes whether there is any transferable signal before advanced architecture.

---

## Phase D — Factorized model

Implement the simplest explicit:

```text
metal encoder
ligand encoder
condition encoder
metal × ligand interaction
prediction head
```

No MoE yet.

Beat baselines first.

---

## Phase E — Mechanism experts and source hierarchy

Only after Phase D shows signal.

---

## Phase F — Pairwise + physics losses

Add one component at a time.

Ablate every addition.

---

## Phase G — Uncertainty

Calibrate uncertainty on strict holdouts.

---

## Phase H — Process integration

Only after the predictor satisfies registered transfer criteria.

---

# 29. Decision logic

At the end of each phase write a decision file.

Example:

```text
decisions/D01_transfer_signal.md
decisions/D02_factorization.md
decisions/D03_actinide_transfer.md
decisions/D04_mechanism_experts.md
decisions/D05_uncertainty.md
decisions/D06_process_integration.md
```

Each decision must contain:

```text
question
evidence
metrics
null / supported / ambiguous
decision
next action
```

No narrative hand-waving.

---

# 30. Important research question about actinides

Explicitly test:

> Does including actinide extraction data improve prediction of hidden lanthanide metal–extractant combinations?

Run:

```text
same architecture
same folds
same seeds
same lanthanide test set
```

Compare:

```text
training without actinides
training with actinides
```

Then report:

```text
delta MAE
delta rank accuracy
delta logSF error
delta calibration
```

This is a real scientific result by itself.

---

# 31. Important research question about architecture

Explicitly test:

> Does factorizing metals and extractants outperform flat categorical modeling under missing-cell transfer?

Compare:

```text
flat model
descriptor model
factorized interaction model
factorized + mechanism model
```

under V5.

This is more important than random-split leaderboard performance.

---

# 32. Important research question about chemistry priors

Explicitly test:

> Do mechanism-aware / physics-informed priors improve out-of-distribution chemical transfer?

Compare with and without:

- mechanism routing
- monotonic penalties
- source hierarchy
- series smoothness

Run under V1/V2/V5/V6.

---

# 33. What NOT to do

Do not:

- optimize random-split R²
- create synthetic D values and call them data
- hide null results
- use Gen15 predictions as measured labels
- mix train/test publications
- generate pairwise rows before fold isolation
- treat atomic number alone as sufficient metal chemistry
- assume actinides and lanthanides are equivalent
- force monotonic trends where chemistry can be non-monotonic
- build a huge neural architecture before baselines
- claim process feasibility from unsupported chemistry
- report a single optimum without uncertainty
- let the optimizer exploit sparse isolated records
- tune against the final Pr/Nd holdout
- silently change preregistered metrics
- merge Gen19 into main before review

---

# 34. Final deliverable

The final report must answer, clearly and quantitatively:

1. Is there transferable metal–extractant structure in the expanded corpus?
2. Can a missing metal × extractant combination be reconstructed?
3. Can Pr/Nd selectivity be reconstructed when direct pair data are hidden?
4. Does actinide data improve lanthanide prediction?
5. Which architecture components actually help?
6. How does error grow with chemical distance?
7. Is uncertainty calibrated?
8. Can the model identify when it does not know?
9. Do Gen18 process recommendations remain good after uncertainty propagation?
10. Which Pr/Nd process recommendations are:
    - directly supported
    - transfer-supported
    - speculative
    - unsupported?

The most important conclusion should not be:

> "Model X achieved R² = Y."

It should be:

> "Under deliberately hidden chemistry, the system can / cannot reconstruct missing extraction behavior to a degree sufficient for process decision-making."

That is the Gen19 standard.

---

# 35. Initial instruction to Claude Code

Start by reading the repository and locating:

- the expanded extraction dataset
- Gen18
- Gen15
- previous dataset manifests
- existing descriptor code
- existing pairwise-learning code
- existing process integration tests

Do not modify Gen18 yet.

First create:

```text
generations/gen19_chem_transfer/
```

Then produce only:

```text
DATA_AUDIT.md
FEASIBILITY.md
preregistration_draft.md
```

plus machine-readable audit tables.

Before training a new model, verify whether the corpus actually supports the proposed transfer tests.

If it does, implement the baseline suite and V1/V2/V5 folds.

Only then begin model development.
