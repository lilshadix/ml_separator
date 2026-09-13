# `ml_separator` — full research-programme report

**Prepared 2026-09-03 as a self-contained handoff.** The reader is assumed to know machine
learning but nothing about this project, and to have **no access to the repository**. Every
number below carries the cohort and evaluation regime it was measured under; in this project a
number without its regime is not merely imprecise, it is routinely wrong by a factor that
reverses conclusions.

---

## 0. How to read this document

This is a report on roughly a month of intensive work (2026-08-07 → 2026-09-03) on a single
scientific question, organised as eleven pre-registered "generations" of experiments. The
programme's output is **mostly negative results**, and that is not a euphemism: the most valuable
findings here are demonstrations that specific popular approaches (3D geometry, bigger models,
explicit hierarchical decomposition, learned acquisition, extended descriptors) do not work on
this problem, each established against a matched control with a paired block bootstrap.

Three warnings that apply to everything that follows:

1. **Never quote a metric without its cohort.** The same frozen model scores macro MAE `0.9695`
   and `1.0358` depending only on which cohort and averaging unit you use (§3). Both are correct.
2. **Macro and pooled metrics rank arms in opposite orders** on this data, because one extractant
   is 43.6 % of the pair cohort (§3.4).
3. **Several documents inside the repository are stale.** Where a repository file and this report
   disagree, this report states the corrected value and names the stale source. The known-stale
   set is listed in §10.3.

Sections §1–§4 are the vocabulary; nothing later is interpretable without them. §5 is the
narrative history. §6 is the consolidated "what is actually known". §7 is the shipped model.
§8 is the trap catalogue (the most transferable content). §9–§11 are state, risk and open work.

---

## 1. The problem and the two prediction targets

### 1.1 Chemistry

The project predicts **liquid–liquid solvent extraction of trivalent lanthanides** — the
separation chemistry behind rare-earth processing. An aqueous phase containing dissolved metal
ions is contacted with an organic phase containing an *extractant* (a designed organic ligand).
Each metal partitions between the phases. Two quantities matter:

- **`log D`** — base-10 distribution ratio of one metal, for one extractant, under one fully
  specified set of conditions. It answers *does this ligand extract this metal at all, and how
  strongly*. This is the **level**.
- **`log SF(A/B) = log D(A) − log D(B)`** — the separation factor between two lanthanides. It
  answers *which of the two does the ligand prefer*. This is the **selectivity**.

Both are needed for real process design: a ligand that separates La/Lu by 10× while extracting
1 % of either is useless. The two targets behave completely differently under machine learning,
and the programme's single most consequential design decision was switching between them.

### 1.2 Target 1 — the pair target (generations 2–4)

`log_SF_A_over_B`, formed **only inside a group of (canonical SMILES, all 64 condition columns)** —
i.e. exact condition matching, never across conditions. `A` is always the lighter lanthanide
(Z_A < Z_B). `pair_scope=all` forms every unordered metal pair, so Nd–Sm is a legal pair even
though Pm is absent from the data; `adjacent` scope exists only to reproduce a historical
benchmark.

Antisymmetry is **structural, not learned**: the training set is augmented with a reversed copy
(features swapped, difference features negated, target negated) and the prediction is
`(f(A,B) − f(B,A)) / 2`. Verified `max |p(A,B) + p(B,A)| = 4.441e-16`.

The fatal property of this target, discovered in gen5: **`log SF` cancels the ligand's absolute
level by construction.** A model can be excellent at `log SF` and know nothing about whether the
ligand extracts anything.

### 1.3 Target 2 — the level target (generations 5–11)

`log D` per `(extractant, condition_id, metal)` row. Replicated cells are averaged
(`replicate_policy='mean'`, replicate count recorded); three sentinel rows at `log D ≤ −6`
(min −12.47) are dropped. Geometry is not required, which is what allows the cohort to grow
from 34 extractants to 152.

### 1.4 Target 3 — the deployment unit (generations 8–11)

From gen8 the object of study stopped being "a prediction" and became **"a prediction plus *k*
requested laboratory measurements"** — few-shot calibration of a frozen model on a new extractant.
gen9 added a third target family: **curve shape**, i.e. `d(log D)/d(log₁₀[extractant])` within a
titration series. This reframing is where the programme's largest practical gains came from (§6.3).

---

## 2. The data

### 2.1 The frozen bundle (everything gen2–gen10 is measured on)

`dataset with 3D structures/dataset.parquet` — **5,992 rows × 2,261 columns**, SHA-256
`fefbefc6…4faf5dd`, pinned in `gen3_protocol.json`; the gen3 runner refuses to execute against a
different hash.

| property | value |
|---|---|
| distinct extractants | 190 canonical SMILES |
| metals | **14 lanthanides** (Ce Dy Er Eu Gd Ho La Lu Nd Pr Sm Tb Tm Yb) — no Pm, no Y |
| target `log_D` | mean 0.304, sd 1.666, median 0.416, range [−12.47, 4.21] |
| feature columns | 2,048 pre-computed ECFP bits, 64 `cond__*` condition columns, 10 RDKit scalars |
| metal imbalance | Eu alone is 1,566 rows |
| geometry | `geometry_ok` on 5,479 rows; 1,155 dual-QC XYZ accepted from 1,256 specs |

> **Correction to a repository document:** `docs/metrics_reproduction_20260818.md` §1 says "15
> lanthanides". The parquet holds 14. Use 14.

Geometry rejections are recorded with reasons (40 ambiguous coordination shell, 26 long bond, 23
unavailable, 12 borderline-long), and QC happened upstream of this repository, so absence of
performance-driven tuning is inferred from the recorded reasons rather than proven.

Side tables: `ligand_2d_descriptors.parquet` (190 × 207), `ligand_pretrained_embeddings.parquet`
(190 × 3,073, ChemBERTa-family), `features/vietoris_rips_inputs.npz` (65 MB simplicial asset).

**What is structurally unavailable and was not faked:** the bundle holds 1,155 `.xyz` files and
**zero `.mol2`/`.sdf`** — 756 recorded `mol2_path` entries all point at a dead NFS path. There is
therefore no bond table, so chelate bite angles are declared UNAVAILABLE rather than inferred from
distances. The Vietoris–Rips `edge_index` is a distance filtration, not connectivity. True
continuous shape measures against ideal polyhedra are not implemented; the distortion features
are proxies and are labelled as such. This discipline — recording a gap instead of approximating
it — is characteristic of the whole codebase.

### 2.2 Provenance

Publication identity is **not in the bundle** but is fully recoverable: the key
`safe_exp_id = "{stem}_SAFE:{exp_id}"` joins **5,992 / 5,992** rows to the sibling repository
`lanthanide_dataset_builder`. The corrected, frozen count is **105 publications, 0 % ambiguous**.

> `README.md` and one gen6 gate table still say 109. 105 is correct; the inflation came from
> incomplete DOI normalisation (`DOI 10.x` vs `DOI:10.x`) plus one mangled citation blob that
> split a single reference into four.

How much modelling structure crosses a study boundary (all 5,992 rows):

| grouping | groups spanning >1 publication | rows affected |
|---|---|---|
| extractant × condition × metal cell | 3 of 5,302 | 6 (0.10 %) |
| pair key (extractant × condition) | 8 | 61 (1.02 %) |
| **gen5 `unseen_series` CV group** | **27 of 344** | **1,577 (26.32 %)** |
| extractant (`unseen_ligand` group) | 28 of 190 | 3,710 (61.92 %) |

The first two are clean. **The third is not**: gen5's `unseen_series` regime — the one its own
write-up called "the honest known-ligand regime" — holds out a group that spans more than one
publication in 26 % of rows, so those numbers are optimistic by an unmeasured amount. The last row
is a batch-effect question, not a fold-integrity one.

### 2.3 The second dataset (`dataset_all_metals`, built 2026-08-22)

A deliberately separate multi-metal build that touches nothing frozen, so the control stays the
control. 41 immutable per-metal CSVs, **48,471 raw rows collapsing to 16,770 archive records ×
134 columns** — 31,701 rows were pure export fan-out (the source *file name* is the **queried**,
not the measured, metal; all 37 raw columns verified constant within every `exp_id`).

40 metals, 262 extractant systems, 83 solvents, 583 series, 2,331 usable curves. Subsets:
lanthanide 8,053 / non-lanthanide 6,799 / single-component 14,112. Readiness tiers are *labels,
not filters* — no row is dropped: A 11,026 / B 1,385 / C 2,455 / D 1,904, with 1,040 records in
`E_VALUE_CONFLICT` that are never averaged. Pipeline determinism verified byte-identically over
33 artefacts.

Two identity hazards in this build, both documented: the file name is not the measured metal (one
record appears byte-identically in up to 12 metal files), and TODGA's SMILES is attached to 22
unrelated extractant names because one sub-source records a *masking agent's* name next to the
organic extractant's structure. Any pipeline that infers metal from filename or identity from name
will mislabel a large fraction of rows.

This dataset was unused by every model through gen10. **gen11 is the first study to use it** (§5.9).

---

## 3. The evaluation contract — read this before any number

### 3.1 Cohorts

| name | definition | size |
|---|---|---|
| **pair cohort** | requires all 64 conditions present + geometry + unique replicates | 6,699 pairs / **34 extractants** / 28 exact-ECFP clusters / 91 pair types / 274 condition cells; `cohort_sha256 af3d91b7…5f0e` |
| **BASE91** | level target, ≥10 cells per extractant | 4,881 rows / 91 extractants / 74 ECFP clusters / 40 chemotypes / 230 series; target sd 1.643 |
| **C-FULL** (= EXPANDED152) | level target, ≥3 cells | **5,248 rows / 152 extractants / 131 ECFP clusters / 79 chemotypes**; fingerprint `bed178ec1a7a82b0` |
| **C-COMMON** | ≥5 pool and ≥2 eval rows in every arm at every *k* | 99 extractants / 41 chemotypes |
| **C-KSHOT** | few-shot study cohort | 143 extractants |
| **C-CURVE** | extractant-concentration curves with ≥4 points | 155 curves / 25 extractants (775 curve×seed) |

The pair-cohort filter ladder is worth stating because it explains the whole gen2–gen4 era:

```
5,992 rows
  −129   TODGA quarantine
  −2,851 incomplete conditions   ← this step collapses 190 extractants to 34
= 3,012 rows → 2,520 cells (198 replicated) → 8,195 candidate pairs
  −688 geometry  −802 non-unique replicate  −6 incomplete 3D
= 6,699 pairs
```

The dominant filter is *metadata completeness*, not chemistry. The pair-era conclusions
generalise to well-annotated extraction systems, not to the literature.

The frozen, target-independent chemistry map over all 190 extractants is **164 bit-identical ECFP
clusters → 98 Tanimoto-0.7 chemotypes**; the largest chemotype holds 51 extractants, 80 of 98 are
singletons, and 12 extractants have no neighbour above Tanimoto 0.4.

### 3.2 The same model, two numbers

> The identical frozen gen9-recomposed pipeline scores **macro MAE 0.9695 on C-FULL** (one vote per
> ECFP cluster, 131 units) and **1.0358 on C-COMMON** (one vote per extractant, 99 units).
> Recomputed-vs-reported agreement 5.6e-07. No figure in the repository puts both on one axis.

This is the single most important thing to carry into any discussion of these results.

### 3.3 The four validation regimes (level target)

| regime | held-out unit | meaning |
|---|---|---|
| `unseen_chemotype` | Tanimoto-0.7 single-linkage super-cluster | genuinely new chemistry — **the honest regime** |
| `unseen_ligand` | bit-identical ECFP cluster | new ligand, close homologues may be in training |
| `unseen_series` | measurement series | known ligand, new acid/diluent (**but see §2.2 — 26 % publication mixing**) |
| `unseen_conditions` | full condition vector | known ligand, titration interpolation |

Cross-validation is a **randomised** grouped K-fold (shuffled groups dealt round-robin) because
sklearn's `GroupKFold` is greedy-by-size and left 82 % of rows in the same fold across nominally
different seeds. 5 folds × 5 seeds; split seeds `{104729, 130363, 155921, 196613, 262147}`; **model
seed fixed at 42, independent of the split seed** — frozen across gen5–gen11.

Every test row in the two ligand regimes carries `nn_train_tanimoto`, so any result can be
stratified by actual chemical novelty. This is how the "near-neighbour lookup" finding (§6.1) was
established.

### 3.4 Metrics, and why pooled metrics are banned from selection

- **Primary (pair studies):** `equal_extractant_macro_mae` — mean over extractants of
  within-extractant MAE, unweighted. Sample weights make the model optimise exactly this.
- **Primary (level studies):** macro MAE, one vote per ECFP cluster on C-FULL, one vote per
  extractant on C-COMMON.
- **Pooled MAE / pooled R²:** computed, never used for selection — frozen out in
  `gen3_protocol.json`.

The reason, measured:

| arm | macro MAE | pooled MAE | pooled R² | sign accuracy |
|---|---|---|---|---|
| A2_refit (the model) | **0.319** | 0.419 | 0.323 | 0.847 |
| PAIRMEAN (trivial baseline) | 0.448 | **0.383** | **0.444** | **0.850** |

The trivial baseline beats the model on *both* pooled metrics *and* on sign accuracy; the model's
only win is macro MAE. The cause is that one extractant (TODGA) is **43.6 % of pair rows** and
forms an entire outer fold by itself at every split seed tested. Sign accuracy 0.847 is also
below the 0.853 of an "always negative" predictor — it is not evidence of quality.

**Effective sample size.** Kish `n_eff` for the pair cohort is **4.8 groups by extractant, 3.8 by
ECFP cluster** — not 6,699 rows. A row-level confidence interval here overstates confidence by
orders of magnitude.

### 3.5 The offset/shape decomposition (gen6 onward)

For each held-out ligand `l` with residuals `e_i`: `b_l = mean(e_i)` is the **level error**,
`offset_mae = mean|b_l|`; centring predictions and truth within the ligand gives **`shape_mae`**.
The identity `SSE_total = SSE_centred + Σ n_l b_l²` is unit-tested exactly. Ligands with one row
are excluded from shape statistics but keep their offset. This decomposition is what turned a
plateaued MAE number into a diagnosable system, and it drives every result from gen7 onward.

> **Trap:** `within_ligand_r2` has two incompatible definitions. The deployable one is
> `1 − SSE/SST_within`. The original granted a free per-ligand offset **taken from held-out data**
> and survives as `within_ligand_r2_shape`. Every figure from the first gen5 run is the shape
> variant. `span_recovery` likewise has guarded (0.0464) and unguarded (0.0507) variants, and
> `README.md` quotes the unguarded ladder.

### 3.6 Inference

Always a **paired block bootstrap whose resampling unit is the chemistry unit** (extractant, ECFP
cluster, or Tanimoto chemotype) — never the row. 5,000–10,000 replicates, RNG seed 8675309.
The decision rule (`scripts/gen4_decision.py`) passes a candidate only if **mean delta > 0 AND
≥ 3/4 confirming seeds AND CI95 lower bound > 0 AND Holm-corrected p < 0.05** over a frozen
candidate set. "5/5 seeds" is explicitly never sufficient on its own, because the five split
seeds re-partition the same ligands.

**A hard reproducibility floor:** the gen6 reproduction gate measured cross-machine fit tolerance
at mean 0.003 / max 0.010 macro MAE (a forest grown under a different BLAS is a different forest).
Consequently **an effect smaller than ≈0.01 macro MAE is not distinguishable from a change of
machine**, and this binds retroactively on all of gen5.

---

## 4. Leakage protection, and the one hole it found

`LEAKAGE_AUDIT.md` and `FEATURE_AUDIT.md` are audits written **before** any modelling change and
before any performance number was computed. The audit is a 17-item checklist scored with
evidence, not assertions. Verdicts: **13 SAFE, 2 LOW, 1 MEDIUM, 1 CRITICAL**.

What is enforced:

- no `train_test_split` anywhere; all learned preprocessing inside a per-fit `Pipeline`;
- fold-local median imputation with **persisted per-fold imputer statistics + SHA-256** per arm
  per fold;
- pre-declared **fail-closed** feature families — an unrecognised column raises;
- a forbidden-fragment assertion over the entire model contract that raises if `log_D`, `log_SF`,
  `safe_exp_id`, `build_id`, `geometry_key`, `sample_weight`, `asset_index` or `xyz_path` reaches
  the model;
- **asserted-zero** train/test overlap on extractant, source_id, geometry_key,
  geometry_feature_build_id and vr_graph_index (measured `[0,0,0,0,0]`);
- the pair→asset chain `source_id → row_geometry_map → vr_graph_index → VR build_ids`
  independently re-verified;
- hyperparameters selected on inner-OOF macro MAE only; the outer test set touched exactly once;
- frozen `experiment_contract.tsv` and `cohort_sha256` with fail-closed resume/aggregation.

**Negative controls are training-time transformations, and this is checked rather than assumed.**
Shufflers receive only the current training row index set; `shuffle_audit.csv` records
`test_rows_touched` per application and the assertion is `== 0`. An arm that does not beat its own
permuted twin is unsupported *regardless* of how it compares to the champion. The gen6 protocol
makes this a table of required nulls: a descriptor family needs a permuted twin of the same width
and marginals; a diversity expansion needs a row-count-matched depth control **and** target-shuffled
sparse rows; a transfer arm needs source-label shuffle, permuted source predictions, and a random
embedding of equal width; an active-learning claim needs random acquisition and a "widest"
heuristic; any *k*-shot claim needs PAIRMEAN + the same *k* measurements **and** a ΔZ trend + the
same *k* measurements.

### 4.1 The CRITICAL finding — still live under the default protocol

With `--group-mode extractant` (the runner default), ECFP cluster `0982ebf465b7` holds **five
distinct extractants spread across four outer folds**. Because each `canonical_smiles` maps to
exactly one fingerprint, those held-out "unseen extractants" have their bit-identical 2,048-bit
fingerprint present in training — all 2,058 ECFP columns are literally copied.

- At split seed 42: **3,334 / 6,699 rows = 49.8 %** affected.
- At the five *production* split seeds: a second multi-extractant cluster also crosses folds in
  4 of 5 seeds, giving **3,656 rows = 54.6 %**; only seed 104729 gives 49.8 %.
- **Report this as "50–55 % of the pair cohort depending on split seed."**

This inflates the 2D (A2) baseline specifically. The stricter `--group-mode ecfp-exact-cluster`
already exists in the code; the defect is that the runner does not default to it. The audit's
mandated protocol change — primary grouping must be `ecfp_exact_cluster`, literal-extractant
grouping reported only as the weaker sensitivity — is on record.

### 4.2 The encoding asymmetry (rated CRITICAL for scope, not leakage)

In the pair model, **2D descriptors and conditions enter as absolute values** (`base__X = X_A`)
while **3D enters only as an A−B difference** (`delta3d__X`). Since A and B are the same ligand
with a different lanthanide, the difference cancels the coordination environment and retains only
its *response to the metal swap*. Measured on the cohort: `donor_count_P/S/other` have sd 0 and
are 100 % exactly zero; `coordination_number` is 54.7 % exactly zero.

**Consequence:** the pre-specified A5/A6 arms cannot test "metal-centred 3D coordination
information helps". They test only the far narrower "the metal-*induced change* in coordination
geometry helps". Absolute coordination chemistry (CN 8 vs 9, polyhedron size, donor composition)
is structurally invisible to those arms. Symmetric arms A5s/A6s (`sym3d__X = (X_A + X_B)/2`) were
added *alongside* — never replacing — the pre-specified arms to address this.

### 4.3 Feature-block facts that are commonly mis-stated

- **The 2D block is 92 % constant**: 1,902 of 2,058 columns are zero-variance across the cohort,
  because 34 ligands cannot switch on most of a 2,048-bit Morgan fingerprint. 40 of 64 condition
  columns are also constant. So the argument "3D adds only 38 columns to a 2,130-column baseline,
  the effect would be diluted" is **wrong** — informative counts are A2: 2,130 → **188**.
- The surviving pair cohort is chemically homogeneous: mean 8.27 O donors (range 6–9), 0.24 N
  donors, **0.000 P/S/other**. Any conclusion about donor-atom identity is untestable on this
  cohort *regardless of what a model shows*. (In the full 5,479-row geometry bundle S reaches 8
  donors and P reaches 1 — the emptiness is a cohort property, not a dataset property.)
- xTB/electronic quantities (dipole, partial charges — 12 columns) are excluded from every arm by
  design, because their *availability* would act as a provenance shortcut.

### 4.4 The experiment that was never run

The pre-specified **A0–A6 2D-vs-3D production ablation** — the experiment actually designed to
test the repository's founding hypothesis — **has never been executed**. `README.md` records only
a bounded local smoke test; `LEAKAGE_AUDIT.md` item 14 records that no ablation run directory
exists. Meanwhile ~20 completed runs of the legacy Delta3D/simplicial paths exist under many
configurations. That is not leakage in executed code, but it is precisely the condition under
which post-hoc protocol selection happens; it is rated MEDIUM.

**The repository's 3D verdict therefore rests on the gen5 *level* study (H4), not on the pair
study it was designed for.** See §5.3.

---

## 5. Generation-by-generation history

Eleven generations, each pre-registered before execution. The pattern throughout: a protocol
document is frozen with hypotheses, arms, controls and a pass/fail rule; the run executes; a
decision report declares pass or fail against the frozen rule. **Most declare failure**, and the
programme treats a formally declared negative result as a deliverable.

### 5.1 gen2 + gen3 — "does 3D coordination geometry help?" → No

**Question.** Does local metal-centred 3D geometry (Architector-built Ln complexes) add
transferable signal over a strong 2D baseline for `log SF` on unseen extractants?

**The A0–A6 ladder** (frozen feature families: CONDITIONS 64, LN 8, 2D 2,058, 3D_GLOBAL 12,
3D_LOCAL 38):

| arm | contents | columns |
|---|---|---|
| A0 | CONDITIONS | 64 |
| A1 | + LN | 72 |
| **A2** | **+ 2D (the champion)** | **2,130** |
| A3 | CONDITIONS + LN + 3D_LOCAL (no 2D) | 110 |
| A4 / A5 / A6 | A2 + global / + local / + both 3D | 2,142 / 2,168 / 2,180 |

Extension arms: A5s/A6s (swap-symmetric), G2 (3D pair response, 46 cols, including a physically
motivated "compliance" term measuring how far the cavity follows the lanthanide contraction),
G4 (metal-site descriptors), E2/E3/E4 (electronic), S1–S5 (secondary 2D representations), plus
**shuffled twins of every block** at three seeds.

**Result** (COHORT-6699, unseen extractant, cross-seed OOF ensemble of model seeds
7/42/137/2027/9001 on split seed 104729; equal-extractant macro MAE):

| arm | macro MAE | Δ vs A2 | 95 % CI |
|---|---|---|---|
| **A2** | **0.31609** | — | — |
| A5 (A2 + local 3D) | 0.33089 | **−0.014796** | [−0.02568, −0.00476] |
| A6 (A2 + all 3D) | 0.33291 | **−0.016822** | [−0.03015, −0.00402] |
| A3 (3D instead of 2D) | 0.36232 | **−0.046224** | [−0.01648, −0.07697]* |
| A0 = trivial | 0.63596 | — | — |

\*sign convention: A2 beats A3 by +0.046.

**No arm beats A2 on the primary metric.** Adding 3D is *significantly worse*, and 3D cannot
substitute for the fingerprint. And the killer: **the real geometry blocks lose to their own
training-only shuffled twins** on macro — A5 −0.0148 against shuffles −0.0135/−0.0111/−0.0133;
G2 and G4 likewise worse than all three of their shuffles. Structural-novelty stratification does
not rescue them (A5 is −0.010 below Tanimoto 0.7 and −0.020 above).

> **Correction to a claim that circulated internally:** it is *not* true that E3 (A2 + electronic
> pair features) is the only block beating its own shuffled control on macro. Three do — E3
> (+0.0063/+0.0067/+0.0058), E2 (+0.0023/+0.0005/+0.0020) and S5 (+0.0008/+0.0051/+0.0002). E3 has
> the **largest margin**, not the only positive one. E3 still loses to A2 on macro (−0.0020)
> while *beating* it on pooled MAE (0.4068 vs 0.4187) — a live example of the trap.

**The simplicial neural network (G3).** A metal-centred 0/1/2-simplicial Siamese network on a
different cohort (COHORT-1081: 1,081 adjacent-lanthanide pairs, 32 extractants, grouped by exact
ECFP cluster): Ln plus donors within 3.10 Å, all VR edges, only Ln–Dᵢ–Dⱼ triangles, RBF support
4.0 Å, exact structural antisymmetry, blended with tabular Delta3D through an **inner-CV gate**
(`w ∈ {0, 0.25, 0.5, 0.75, 1.0}`, nonzero only if inner mean group-MAE reduction − 0.5·SE > 0).

- Nonzero `w` proposed in 10/25 outer folds, guard rejected 3, **final nonzero in 7/25, every
  surviving value 0.25** — the "hybrid" is mostly an exact fallback to the tabular model.
- Guarded hybrid − Delta3D: macro-group MAE **−0.001381, 0/5 seeds positive**; every bootstrap CI
  contains zero.
- **Raw unshrunk SNN − Delta3D: R² −0.147, group-balanced R² −0.182, macro-group MAE −0.025, sign
  accuracy −0.116, 0/5 seeds positive on every endpoint.**

Worse for the 3D programme: **the mandatory prerequisite ablation fails.** The roadmap requires
tabular Delta3D to beat a *matched 2D+2D ensemble*, not merely a baseline. Recomputed:
−0.0056/−0.0021/−0.0060/−0.0024/−0.0041, mean **−0.004029, 0/5 positive**. The handoff's
recommendation to keep tabular Delta3D as the preferred 3D model is therefore **not supported by
its own required control**, and the handoff does not mention this contrast.

**A retraction worth studying (the descriptor-width artefact).** The metal-site descriptor block
looked good on the small cohort (macro-group MAE +0.001433, 5/5 seeds positive). Then a *free
permutation null* (seed 33) beat the real block 5/5 on **both** primary endpoints (+0.0433
group-balanced R², +0.0039 macro-group MAE). Adding continuous columns of any kind to a mostly
binary 2,130-column fingerprint matrix changes how ExtraTrees samples split candidates, and on
that cohort the artefact is worth ±0.02–0.04 group-balanced R² — an **order of magnitude larger
than the effect being weighed**. The decisive real-vs-null comparison was never produced (the null
arm's aggregation failed fail-closed on a protocol-fingerprint mismatch), so the claim was
explicitly retracted, and the same block later lost to all three of its shuffles on the all-pairs
cohort.

**gen3** (pre-registered `gen3_protocol.json`, `status: frozen_before_experiments`, 2026-08-13)
tested three architectural hypotheses on top of A2: H1 antisymmetric CatBoost (9 weighting × loss
arms), H2 residual correction on electronic-pair features (6 arms), H3 a shared-scalar
antisymmetric latent-difference MLP.

**Outcome: every one of the 19 challenger arms has `win_rule_passed = False`. A2 is retained.
`valid_negative_result = true`.** Best H1 arm Δ −0.0095 (CI spans zero); all H2 arms fail the
pre-declared λ>0 eligibility gate; H3 is catastrophically worse (0.397 and 0.424 vs 0.319).

Three defects in gen3's own design, found by self-audit and worth carrying forward:

1. **H2's Stage 1 was not the champion.** The residual was formed on cross-fitted *CatBoost*
   predictions, i.e. H2 corrected a stage-1 model that is itself worse than A2. "ELEC_PAIR adds
   nothing" is confounded with "the stage-1 model was wrong".
2. **H2's compact-physics view was structurally dead.** All 14 swap-*even* electronic features
   have stability-selection frequency **exactly 0.000**, while all 7 swap-*odd* features are
   selected 0.68–0.89 of the time. Even-parity features cannot contribute to an odd-parity target
   under an antisymmetrised head — half the pre-registered view could not carry signal by
   construction.
3. **H3's early stopping ran against the training loss**, not a held-out split; several folds ran
   all 80 epochs. The architecture arm was never given a fair regularisation budget.

**The single most instructive result of this era** is gen3's H3 label-shuffle control:
`H3_A2_LABEL_SHUFFLED` **beats** the real `H3_A2` on pooled MAE (0.399 vs 0.458) and on pooled R²
(0.421 vs 0.185) while losing on macro (0.443 vs 0.424). *Shuffling the targets raises pooled R²
by 0.236*, because regressing to the pair-label mean is exactly what pooled R² rewards on a
TODGA-dominated test set. This is why `pooled_guardrail: "never used for model or champion
selection"` is written into the protocol.

> **Averaging-axis trap:** gen2 numbers are 5 *model* seeds on 1 split seed (A2 = 0.316092);
> gen3 numbers are 5 *split* seeds at 1 model seed (A2 = 0.319221). Same arm, different axis,
> different value. Never cross-compare them.

Also note: held-out extractants are **not fully novel** — 5 of 34 have max Tanimoto = 1.0 to a
training extractant in their own fold (median 0.746, min 0.292).

### 5.2 gen4 — candidate families, and the k-shot study that redirected the programme

Both gen4 documents are **self-declared exploratory**, not frozen protocols; the only pre-declared
object is the decision rule in `scripts/gen4_decision.py`. Everything runs on the **frozen gen3
outer folds** (so it inherits the 50–55 % ECFP-homolog leakage of §4.1, which neither result
document restates).

**Four candidate families** built on top of A2:

1. **Transitive projection (TP)** — a label-free, exact post-processing step: inside each
   `(extractant, condition_id)` cell, least-squares-project pair predictions onto per-metal scores
   `s_A − s_B`. Uses only predictions, no labels.
2. **HIER** — level model on cell means + within-cell deviation model.
3. **Scale/prior family** — `y = s(ligand,cond)·t(pair) + r`, with `s` a cross-fitted forest
   (SCALE, shrinkage variants), and the stacked **PRIOR** (A2 + two cross-fitted even columns),
   plus **PRIOR_TP**.
4. **lig2d** — 206 extended 2D ligand descriptors (168 RDKit + 38 hand-crafted DGA/amide graph
   descriptors) over 190 ligands.

**The pre-registered bar** (three parts, all required): 5-seed mean macro MAE below A2_refit **AND**
positive paired delta in ≥ 3 of 4 confirmation seeds **AND** extractant-unit paired bootstrap 95 %
CI excluding zero, Holm-corrected across the frozen candidate set {PRIOR, PRIOR_lig2d, A2_lig2d,
SCALE_s50}.

**No candidate family passed.**

| arm | 5-seed macro MAE | Δ vs A2_refit | seeds | verdict |
|---|---|---|---|---|
| A2_refit (reference) | 0.3188 (local) / 0.3192 (cluster) | — | — | retained |
| PRIOR | 0.3168 | +0.0019 | 3/5 | macro-neutral, CI spans 0 |
| A2_lig2d | 0.3269 | −0.0081 | 2/5 | worse |
| PRIOR_lig2d | — | −0.0100 | — | worse |
| SCALE_s50 | 0.3433 | −0.0246 | 1/5 | **CI excludes zero on the harmful side** |
| PRIOR_TP | 0.3145 | +0.0043 | 4/5 | best profile, **explicitly not claimed** |
| A2_refit_TP | 0.3172 | +0.0016 | **5/5** | adopted as post-processing |

> **Honesty qualifier on TP:** it is often described internally as "the only confirmed gen4 gain".
> Precisely: `passes_rule = False` for **every** row in `decision_table.csv`, and TP was never
> entered into the Holm-corrected candidate set at all — it is reported as uniform post-processing
> applied to every arm. Its 4-confirmation-seed CI is [+0.00007, +0.00375] (excludes zero), but the
> **all-five-seed CI is [−0.00012, +0.00336] and includes zero.** It was adopted because it is
> mathematically exact, label-free, and improved 5/5 seeds — not because it cleared the bar.

**The central negative result: the selection-seed illusion.** All three apparent gains on the
selection seed (104729) evaporated on confirmation: PRIOR went from −0.0098 to +0.000024;
A2_lig2d from −0.0048 to +0.0113 (worse); SCALE_s50 from −0.0109 to +0.0329 (much worse).
Selection on one split seed of a 34-group cohort is not informative.

**Why descriptors were abandoned.** The per-ligand selectivity scale is **not predictable from 2D
descriptors on unseen ligands**: the cross-fitted scale forest correlates `r = −0.04` with the
true cell slope, while A2's *implied* scale reaches `r = 0.34`, and the scale forest correlates
`r = 0.57` with A2's implied slope — i.e. it re-learns what A2 already knows. Three descriptor
families (3D geometry, xTB electronic, extended 2D) had now failed to move the primary metric,
and this diagnostic explained why. Work on richer ligand descriptors for the unseen-extractant
regime was stopped.

> **Reproducibility defect:** those `r = −0.04 / 0.34 / 0.57` numbers are attributed to a
> `scale_diag.py` that **does not exist anywhere in the repository**. The most-quoted negative
> result of gen4 is not reproducible from committed artefacts. Likewise the oracle headroom figure
> "0.319 → 0.247" appears in both documents with **no run artefact behind it**.

#### The k-shot calibration study — the pivot point of the whole programme

Post-hoc on frozen OOF predictions (no model retrained). Question: instead of *predicting* a new
ligand's offset, **measure `k` pairs of it and recalibrate**. Calibrator forms, all ridge-shrunk
toward identity: `scale` (transitivity-preserving), `offset`, `affine` (destroys transitivity),
`trend` (`y ~ a·p + b·ΔZ`, transitivity-preserving). Support policies: random, adjacent, widest,
**foreign** (control), **cross_condition** (support from other conditions, query from a held-out
condition — the honest deployment regime).

**Two methodological traps this study had to solve, both instructive:**

- **Within-cell transitivity.** Inside a complete `(extractant, condition)` cell `log SF` is
  *exactly additive*, so query pairs spanned by the support are arithmetic consequences, not
  predictions. At k=10 under random support, **31.6 % of macro-weighted query rows are determined
  this way** (8.0 % row-weighted; 89–98 % for the five smallest extractants). Scoring on all rows
  makes the answer roughly twice as good as it is. All headline numbers use the **free stratum**,
  computed by union-find over the k_max pool so it is fixed across k.
- **Same experiment vs new experiment.** Uniform support sampling usually draws from the very
  condition being predicted (20 of 32 eligible extractants have a single condition). The `random`
  regime therefore measures *within-experiment recalibration*, not ligand transfer.

**The result** (cross_condition, free rows, 12 extractants, equal-extractant macro MAE):

| k | model (A2_refit_TP) | PAIRMEAN table | no-model ΔZ fit | model − ΔZ |
|---|---|---|---|---|
| 0 | 0.3785 | 0.3934 | 0.4013 | +0.023 |
| 1 | 0.3275 | 0.3244 | 0.3541 | +0.027 |
| 2 | 0.2918 | 0.2928 | 0.2939 | **+0.002** |
| 3 | 0.2773 | 0.2750 | 0.2719 | **−0.005** |
| 5 | 0.2589 | 0.2526 | 0.2493 | **−0.010** |
| 10 | 0.2413 | 0.2322 | 0.2395 | **−0.002** |

**From k = 2 onward the model adds nothing over a two-parameter no-model fit `y ≈ b·ΔZ`, and is
marginally behind a trivial pair-mean lookup table.** This installed the standing null:
**PAIRMEAN + k-shot**. Any model claiming to help after calibration must beat a lookup table
calibrated on the same *k* measurements.

Secondary findings that carried forward: **which pairs you measure matters more than how many**
(widest-first at k=1 reaches what random needs k=2 and adjacent needs k≥5 to reach); the deployment
recipe is `trend` at λ ≈ 0.1, k ≥ 3, measuring *wide* metal pairs; never use `offset`/`affine` on a
TP arm because they destroy transitivity (violation rate goes 0.0 → 1.0). And **single-draw risk is
real**: at k=1 the mean gain is +0.048 but 15 % of individual draws are *harmful* (10th percentile
−0.013). A chemist gets one draw, not the mean of twenty.

> **Scope limit that applies to all of gen2–gen4:** "new extractant" here means "new
> diglycolamide analogue". 26 of 32 eligible extractants literally contain the diglycolamide core
> `C(=O)COCC(=O)N`; of the remaining six, five are close glycolamide/diamide variants and exactly
> **one** is a genuinely different scaffold.

> **Document defects to be aware of:** the gen4 study doc flips its sign convention mid-document;
> it says the shuffled-descriptor control was not re-run (a valid 5-seed control *does* exist:
> SHUF 0.3465 vs real 0.3275 vs no-descriptors 0.3192 — the block carries *some* signal but is
> worse than not using it); it says HIER was not carried to confirmation (it was: 5-seed 0.3218,
> 1/5 seeds); and it calls PRIOR_TRENDONLY "= A2" when over five seeds it is significantly
> *worse* (−0.0024, CI excluding zero).

### 5.3 gen5 — the pivot to `log D`, and the MASSACTION block

**Why pivot.** A pair row needs two metals measured under identical recorded conditions, and
**100 of the 190 extractants have only one metal measured** — pairs are impossible for them by
construction. A level row needs one measurement. Plus the pair target cancels exactly the quantity
process design needs.

`docs/gen5_levels_protocol_20260817.md` is a **pre-registration** (frozen 12 minutes before the
first run, after an adversarial five-lens review that produced ~38 findings). The design is **one
arm per property family** — METAL(3), COND(64), PHYSCHEM(10), ECFP(2048), LIG2D_EXT(206),
DONORS(13), COMPLEX_PHYS(26), POLYHEDRON(58), GEOM_COND(46) — giving 34 scored arms including
`@hgb`/`@ridge` learner variants, shuffled twins and four nulls, all on **one fixed, untuned
estimator** so that arm differences are attributable to features rather than tuning.

Variance decomposition of `log D` on this cohort: ligand 44 %, metal 5 %, ligand+condition 89 %,
replicate ceiling 97 %.

**Verdicts:**

| hypothesis | verdict |
|---|---|
| H1 (a 2D ligand family helps a new ligand) | **PASS** on LIG2D_EXT (+0.123, CI [+0.022, +0.229], 5/5) and DONORS (+0.084, CI [+0.021, +0.156], 5/5); **FAILS on ECFP** (+0.048, CI spans 0) — reversing the authors' expectation |
| H1′ (survives genuinely new chemistry) | **PASS**, despite the 08-18 document explicitly predicting failure |
| H2, H3 (conditions carry signal; beats honest nulls) | **PASS** decisively |
| H4 (3D adds over 2D) | **PASS IN THE NEGATIVE DIRECTION**: 3D is *worse* by 0.102, CI [−0.177, −0.034], **0/5 seeds** |
| H5 (interpolation vs prediction) | reported, no gate — and damning (below) |
| H6 (k-shot helps) | PASS on one metric, **partially retracted** on the repaired one |

#### The headline negative result: the level model is a near-neighbour lookup

Stratifying `unseen_ligand` OOF by nearest-training-ligand Tanimoto, the champion's edge over a
**single global constant**:

| NN Tanimoto | clusters | rows | model | global constant | edge |
|---|---|---|---|---|---|
| < 0.4 | 10 | 192 | 1.196 | 1.155 | **−0.041** (worse than a constant) |
| 0.4–0.6 | 15 | 429 | 1.063 | 1.242 | +0.179 |
| 0.6–0.8 | 50 | 1,386 | 0.812 | 1.414 | +0.602 |
| ≥ 0.8 | 24 | 2,873 | 0.657 | 1.313 | **+0.656** |

**58.9 % of test rows sit at Tanimoto ≥ 0.8 and only 3.9 % below 0.4** (median row-level NN 0.865),
so the headline 0.836 is a weighted statement *about homologues*. The protocol had refused in
advance to claim cross-scaffold generalisation without this stratification; the stratification was
done and does not support the claim.

> **Terminology trap:** the two gen5 results documents use "edge" for two *different* quantities —
> versus the null (08-18) and versus the MC baseline (08-19) — and they read **oppositely** at low
> similarity (−0.041 vs +0.013). Neither document flags the change. Always state the reference.

#### H5 — `unseen_conditions` is not prediction skill

Under `unseen_conditions` the model (MC_ecfp 0.800) is **statistically tied with "copy the ligand's
nearest measured condition"** (NULL_nearest_condition 0.868, Δ CI [−0.06, +0.20]). Every feature
family gains 0.19–0.24 there — *including 3D blocks that add nothing anywhere else*. The mechanism
was pinned down: a block's gain in that regime tracks how many of the 74 clusters its feature
vectors can tell apart (Pearson **r = 0.930** across six families), and that relation **vanishes**
under `unseen_ligand` (r = −0.197). It is ligand-identity lookup, not chemistry. And 75–85 % of
the ECFP gain there is per-ligand offset correction.

`unseen_conditions` is also not pure interpolation: 28 of 74 clusters were measured at exactly one
condition vector, so 8.95 % of rows have the held-out ligand absent from training entirely
(macro MAE 0.931 vs 0.658 for those rows).

#### The cohort filter is the real ceiling

`min_rows_per_extractant = 10` drops **99 of 190 extractants for only 7.9 % of cells** — but 37 of
those 99 have no kept neighbour above Tanimoto 0.4 (versus 7 % of kept ligands). It removes the
crown ethers, flavonoids, pyrazolyl-pyrazines, dithiophosphinates and furandicarboxamides. The
cohort ladder:

| min_rows | rows | extractants | ECFP clusters | chemotypes |
|---|---|---|---|---|
| 10 | 4,881 | 91 | 74 | 40 |
| 5 | 5,209 | 140 | 119 | 69 |
| **3** | **5,248** | **152** | **131** | **79** |
| 1 | 5,299 | 190 | 164 | 98 |

`min_rows = 3` buys **+77 % ECFP clusters and +97 % chemotypes for +7.5 % rows**. The recommended
run was never executed under gen5; the question was handed to gen6.

#### MASSACTION — the first metric gain since gen2 that had a mechanism

Post-hoc and explicitly not pre-registered. Eight columns: log₁₀ of the five continuous conditions,
plus `log[L]·DENTATE`, `log[L]·coreCN`, `log[L]·log[H⁺]`. The motivation is the extraction
equilibrium `Ln³⁺ + 3NO₃⁻ + n L(org) ⇌ Ln(NO₃)₃·nL(org)`, i.e.
`log D = log K_ex + n·log[L] + 3·log[NO₃⁻]` — **linear in the logarithms**, whereas the raw
condition columns feed the tree a molarity spanning 4–9 orders of magnitude, so an axis-aligned
tree must approximate a logarithm with a staircase of splits.

**The law was validated on this cohort before being used as features**: extractant-titration slope
2.64, IQR [2.36, 2.88], 100 % inside the chemically admissible 1.5–4.5, median linear R² 0.985.

Result (additive block, so the ablation is clean): **11 of 12 contrasts in the three ligand-known
regimes have CI95 entirely above zero and all 12 are 5/5 seeds** — gains of +0.028 to +0.062 macro
MAE. Its shuffled twin is *worse than not having the block*, so the signal is in the logarithms,
not the column count. Under `unseen_chemotype` the effect is **nil** (all CIs span zero) — exactly
what the chemistry predicts, since the law is about conditions, not about a new scaffold.

**The transferable lesson, and the programme's stated principle from here on: give the tree the
physical functional form rather than tuning it.**

#### Process failures worth carrying forward

- **Half a study shipped looking complete.** The first run silently executed **2 of 4
  pre-registered regimes** because both SLURM files hard-coded
  `REGIMES=${REGIMES:-unseen_ligand unseen_conditions}`. It was not resource-driven. The
  `decision_report.txt` contained zero occurrences of `unseen_chemotype` or `unseen_series` and no
  caveat; the narrowing appeared only in `summary.json`. Four hypotheses were unevaluable and a
  reader saw a complete study.
- **Nine harness defects (C1–C9), found and fixed 2026-08-18.** None changed a fitted model; all
  were in the reporting layer — which is precisely why they were dangerous. The worst:
  - **C2:** `within_ligand_r2` was **oracle-calibrated** — it re-centred predictions on the model's
    own per-ligand mean computed from *held-out* data. Removing the free offset flips the sign
    (MC_ecfp +0.256 → **−0.094**).
  - **C6:** the property-family table compared MC against MC, printing the study's **largest**
    effect as exactly `0.0000` instead of +0.296 to +0.376.
  - **C7:** k-shot `mae_free` and `mae_all` were averaged over *different* draw sets (pandas
    silently skipped 8.9 % NaN rows concentrated in four extractants), which **inverted the sign**
    of the contamination effect the report exists to show.
  - **C4:** the bootstrap resampled ECFP clusters even under `unseen_chemotype`, making CIs ~1.35×
    too narrow.
  - **C1:** HGB crashed on an all-NaN training column, killing an entire run.
- **Three "different" nulls are one predictor.** Under a ligand hold-out, `NULL_global_mean`,
  `NULL_extractant_mean` and `NULL_nearest_condition` are **bit-identical on all 24,405 rows** — an
  unseen ligand has no training mean. Correct behaviour, misleading presentation.
- **An "alone arm" artefact:** `A_complex_phys` looked like the best single-family arm, but 24 of
  its 26 columns change value across metals within the same extractant — it is silently a
  ligand-*and-metal* descriptor while `A_ecfp` is ligand-only. Its advantage vanishes when the
  ligand is new.
- **Predictions are shrunk conditional estimates and must not be read as calibrated `log D`.**
  Zero of 159 rows per seed with true `log D < −3` receive a prediction below −3; dispersion 0.605.
  Rescaling to match the truth variance makes MAE **worse** by 0.184 — the shrinkage is MAE-optimal
  and cannot be corrected away.

> **Later reversal:** gen5's H1 certified LIG2D_EXT as the champion ligand block. **gen7 found that
> deleting that same 206-column block significantly improves the model** and that it is the only
> block whose removal is a significant gain (§5.5). Any citation of H1 must carry this.

### 5.4 gen6 — the interventional answer: coverage, not capacity

Every generation to gen5 had varied the **model** and the **descriptors** and converged on the same
failure: on a genuinely new ligand the model predicts the *shape* of the response but not its
*level*. gen6 varied the **data** instead, and it is the only generation with a genuinely
interventional design.

**Experiment A design.** One shared cohort built once at `min_cells = 3` (C-FULL: 5,248 rows / 152
extractants / 131 ECFP clusters / 79 chemotypes). Folds hold out whole chemotypes. **Every arm is
scored on byte-identical test rows** (hashed per fold); only the *training row mask* changes.

| arm | training set |
|---|---|
| BASE | the historical ≥10-cell cohort (4,881 rows / 91 extractants / 74 clusters / 40 chemotypes) |
| EXPANDED | + 61 sparsely-measured extractants (367 rows, **+7.0 %**) bringing 57 clusters and 39 chemotypes BASE never sees |
| EXPANDED_ROWMATCHED | all sparse rows + dense rows back to BASE's *exact* row count |
| EXPANDED_SHUFFLED | the same rows with added targets permuted among themselves |

**Result — all four hypotheses PASS on both feature sets, 5/5 seeds:**

| claim | effect | interval |
|---|---|---|
| A1: EXPANDED beats BASE | **+0.1633** macro MAE | BCa [+0.013, +0.295]; cluster-robust [+0.013, +0.314], p = 0.034; block-macro +0.290 |
| A2: larger on distant chemistry | **+0.4630** (2.8×) | [+0.2527, +0.6903] on rows below Tanimoto 0.4 |
| A3: it lands in the level | offset **+0.1773** vs shape **+0.0243** | both 5/5 seeds |
| A4: information, not row count | row-matched control **+0.0019** | CI [−0.0031, +0.0061] |

Absolute: BASE 1.2100 → EXPANDED 1.0468; at nn<0.4, 1.5368 → 1.0429. **7 % more rows, chosen for
chemical novelty, bought more than every architectural change in the programme's history.**

**Experiment B — "you just gave it more rows" is answered.** At equal row budget with label-free
acquisition policies (label-freeness verified at run time by permuting the target — none of 72
selections moved):

| budget | DEPTH | DIVERSITY | RANDOM | MAXMIN |
|---|---|---|---|---|
| 250 | 1.858 | 1.342 | 1.342 | 1.258 |
| 1,000 | 1.672 | 1.226 | 1.217 | 1.119 |
| 3,000 | 1.398 | 1.058 | 1.117 | 1.042 |

Diversity beats depth by **0.28–0.58 macro MAE, CI-clean at every budget**. The mechanism is
ligands bought: at 2,000 rows DEPTH has acquired 20.5 extractants, MAXMIN 96.2; at 250 rows DEPTH
is a **1.3-ligand model**. But two honest qualifications: **round-robin-over-chemotypes is not
better than random** (CI spans zero at 4 of 5 budgets — only *max-min distance* is), and **B3
FAILS**: the depth curve is still improving at the largest budget (+0.351, CI [+0.200, +0.567]),
so **nothing is bounded from above**.

**The document's own qualifications are the most valuable part** — and this is a model for how to
report a positive result:

- 57 of 131 scoring clusters consist *entirely* of the added ligands and carry the whole effect
  (+0.367). On the 72 clusters made only of BASE-eligible chemistry the gain is **+0.008**
  (champion features) and **−0.026** (donor census).
- **60 of 152 ligands get worse**; the median ligand gains 0.033; the top decile supplies **83 %**
  of the net gain.
- **The anti-circularity evidence is a dose-response**: the gain scales with how much closer the
  expansion actually brought training chemistry — Spearman **+0.290, p = 2.9e-4**, with bins
  +0.038 / +0.126 / +0.321 / +0.491.
- The effect is not one giant fold: positive in **20/20** folds holding ordinary chemistry
  (+0.232), weakest in the 5 giant diglycolamide folds (+0.054, 4/5).
- **A retraction inside the same document:** an earlier draft said the expansion buys no condition
  space. On the rows that improve, exact-condition coverage jumps **12.3 % → 50.7 %**. The
  chemistry-only residual is still +0.281 on rows where EXPANDED supplies no matching condition.
- **The pre-registered form of A3 is a weaker test than it looks.** "Offset gain exceeds shape
  gain" was written in *absolute* log units, and BASE's offset MAE (1.044) is twice its shape MAE
  (0.527) — so **any uniform proportional shrinkage passes A3** with no level-specific content. It
  was scored exactly as written and the flaw recorded; the relative-reduction test (which a uniform
  shrinkage would score 1.0×) gives **4.3×** and **6.3×**, so the conclusion survives a stronger
  test than the one pre-registered.
- **EXPANDED_SHUFFLED is a harder comparator than BASE, not an equal one** — feeding 367 rows of
  mislabelled chemistry lands *worse* than not feeding them (1.219 vs 1.210), inflating the margin
  by 6–38 %. Quote the BASE contrast.
- **The percentile bootstrap is miscalibrated at the "all" endpoint**: scoring per ECFP cluster
  (131) while resampling per chemotype (79), with one chemotype holding 28 of the 131 scoring
  units, gives a measured one-sided Type-I rate of **~12.7 % instead of 2.5 %**. Found by
  adversarial review and reproduced. BCa, cluster-robust and block-macro are now printed beside it.

**Experiment C — hierarchical decomposition.** `log D ≈ α_l + F_cond + F_metal`, eight fold-local
models. On multi-metal test cells of new chemotypes: handing the model the **true level** removes
**0.497** macro MAE; handing it the **true metal response** removes **0.105**; the difference
**+0.392** is BCa-clean [+0.273, +0.557], 5/5 seeds. **C1 PASSES** — the level is where the
removable error lives, and metal-response modelling for new chemistry was stopped (the metal
response is 8 % of `log D` variance).

But **C3 passes in the direction that hurts structure**: the pre-registered two-stage model is
**0.144 worse** than the monolith (BCa [−0.240, −0.069], 0/5 seeds).

> **The two-stage residual trap — the most transferable negative result in the programme.**
> Training Stage B on a cross-fitted residual `y − Â_oof` under a chemotype hold-out makes the
> model worse than the monolith it decomposes (1.254 vs 1.076 on the first seed) and worse than
> its own Stage A alone (1.067). **The cross-fitted residual is mostly Stage A's *level* error on
> chemistry the inner model had not seen**; Stage B learns that from in-sample ligand features and
> mis-applies it to new ligands (mean |B̂| 0.53 vs 0.15). Centring on the true within-cell
> departure fixes it (1.042) — and recovers **parity, nothing more** (+0.004).

Honest reporting notes: **C1's magnitude is construction-dependent** (+0.392 with the true cell
mean, which includes 1/k of the scored row's own label; **+0.324** with a leave-one-out cell mean;
+0.104 when paired with the defective Stage B — "honest range +0.10…+0.39"). And the ridge family's
~0.10 gap to the forests is **capacity, not structure** (it uses a compact design without the
206-column block), so it must not be read as a verdict on hierarchy.

**Experiment F — acquisition.** From a deliberately narrow ten-ligand diglycolamide start, revealing
three rows per acquired ligand:

- **F1 PASS** — max-min Tanimoto acquisition beats random on hard chemistry at every checkpoint from
  k=5 (fold-seed-paired +0.21…+0.28, 12–14 of 15), and is the only result clean under *both*
  weightings.
- **F3 PASS** — buying another analogue of what you already have is **far worse than random**
  (−0.46 to −0.84, 0/15 fold-seeds). (Its apparent *growth* is an equal-cluster-weighting artefact.)
- **F2 NOT SUPPORTED** — model uncertainty alone is not an acquisition signal; both pure-uncertainty
  policies are worse than random for the first ~12 acquisitions.
- **NOT PROMOTED:** novelty × uncertainty. The first draft called it "the best rule of all"; that
  sentence was **retracted** because it was a pre-registered *policy* without a pre-registered
  *hypothesis*, ties max-min under the primary weighting, and beats it only in the row-weighted
  rerun.
- Revealing *every* row of an acquired ligand instead of three buys nothing measurable
  (Δ 0.015 on +390 rows vs +90, inside the run-to-run spread).

**Provenance (Phase 0)** established the 105-publication reconstruction and the fold-contamination
table of §2.2, and produced the corrected noise-floor reading: of 313 repeated cells, 79 differ in
a recoverable physical variable (median range 0.342), 208 differ only in the paper's figure caption
(0.368), and the **26 with nothing recoverable differing scatter *most* (0.875)**. If hidden
variables drove the spread the ordering would be reversed. **The honest conclusion is that the
floor is contaminated and its true value is unknown** — not that it is an artefact. This explicitly
retracts an earlier "287 of 313 cells have a hidden axis" claim that `README.md` still carries.

> **A process-integrity note that belongs in any honest handoff:** both gen6 protocols were
> committed to git **in the same commit as their own results**. The Phase 2 results document
> discloses this. The Phase 1 document asserts the protocol was "written and committed before any
> gen6 model was fitted", **which the repository does not support.** The pre-registration
> discipline is real and unusually strong throughout this project, but this specific claim cannot
> be verified from git history.

Also worth knowing: **"one extractant" is coarser than one chemical system** — 17 of 190 canonical
SMILES, covering 50.3 % of rows, carry more than one extractant name (TODGA carries 21, because one
source block recorded the aqueous holdback agent in the extractant-name field). This creates no
leakage but means part of the apparently irreducible within-ligand shape error is a real unencoded
experimental variable.

### 5.5 gen7 — the ceiling, and what "all of chemistry" is actually worth

gen7 asked: with the rows already owned, how much held-out-chemotype error can a better
representation, objective, decomposition, learner or inductive bias remove? 116 arms across 15
suites, on the frozen gen6 contract (byte-identical test rows, reproduced to 5e-6 by a
`tests/test_gen7_reproduction.py` gate).

#### The ceiling answer

**The decisive control had never been run in six generations.** `NULL_metal_cond` — an ExtraTrees
on METAL + COND blocks only, **with no ligand information at all**:

| arm | macro MAE | offset | shape |
|---|---|---|---|
| NULL_global_mean | 1.5525 | — | — |
| **NULL_metal_cond (no ligand at all)** | **1.0882** (3 seeds) / **1.0995** (5 seeds) | 0.8905 | 0.5614 |
| gen6 champion | 1.0468 | 0.8744 | 0.5069 |
| best registry arm | **0.9976** | 0.8468 | 0.5045 |
| ORACLE_level (given each ligand's true mean) | **0.4974** | ~0 | 0.5086 |
| ORACLE_batch (given the true (DOI, table) batch mean) | **0.3502** | — | 0.3634 |
| ORACLE_cell | 0.0000 | — | — |

> **Everything that every fingerprint, descriptor, donor census, 3D block and pretrained
> transformer contributes, together, is ≈0.09 log units. The unlearned per-ligand level is worth
> 0.50 — about six times all of chemistry.**

> **Required qualifier (from the figure metric audit):** `NULL_metal_cond` is **1.0882 at 3 seeds
> and 1.0995 at 5 seeds**, so the "all chemistry is worth X" figure is **0.090 (3-seed, C-FULL)**,
> **0.13 (5-seed, C-FULL)** or **0.083 (C-COMMON)**. Never quote it without saying which.

#### Level vs shape: the 0.09 splits, and only half of it is real

A paired per-ligand test of the best chemistry-aware model against the no-ligand model:

| component | gain | t | p | note |
|---|---|---|---|---|
| **offset (level)** | +0.0437 | 0.81 | **0.418** | improves 51.1 % of ligands — a coin flip |
| **shape** | +0.0569 | 2.75 | **0.007** | improves 56.1 % |

And the pattern is diagnostic: as the independence unit gets **stricter**, the offset result
**weakens** (p = 0.197 → 0.346 → 0.705) while the shape result **strengthens** (p < 0.001 → 0.008
→ 0.015). The gen6 champion versus no-ligand does not clear the bar on **either** statistic.
Binned by nearest-training Tanimoto, the offset gain is +0.28 below 0.4 but **−0.08 at 0.5–0.6** —
*negative where chemistry is closest*, i.e. not a similarity-driven signal.

#### The central experiment: isolate the level and ask what predicts it

One row per ligand, same chemotype folds, 456 ligand-evaluations, 8 model families × 11
representations × raw/PCA-16:

| target | null (global mean) | 1-NN Tanimoto lookup | best model of any kind | captured |
|---|---|---|---|---|
| raw ligand mean, **chemotype** hold-out | 1.3126 | 1.0626 | 0.9269 (donors + ridge) | 29.4 % |
| raw level, **ligand** hold-out (near-twins allowed) | 1.2042 | 0.9447 | 0.7760 (lig2d + ExtraTrees) | 35.6 % |
| **condition-adjusted level**, chemotype hold-out | 1.0581 | **0.9635** | **0.9902** (ChemBERTa-PCA16 + SVR) | **6.4 %** |

> **On the condition-adjusted level — the part that is actually chemistry — *no model of any kind
> beats a 1-nearest-neighbour Tanimoto lookup*.** The 29 % apparent capture on the raw target was
> the model learning *which conditions each ligand happened to be measured under.*

A **representation flip** falls out of the same table: `lig2d + ExtraTrees` wins the *ligand* split
and is mediocre under the *chemotype* split; `donors + ridge` wins the chemotype split.
High-capacity representations interpolate, low-capacity ones extrapolate — **so tuning on a random
split picks the wrong model.**

#### What actually moved the number — all of it information handling, not capacity

- **Missing-value indicators: +0.0673 macro, CI [+0.0323, +0.0974], BCa [+0.0334, +0.0988], 5/5
  seeds** — the only effect in the whole generation with an interval excluding zero under both
  percentile and BCa, and **worth more than the entire 206-column descriptor block**. The mechanism
  is chemistry, not reporting: eight `lig2d__hc__*` columns are null *precisely when the ligand has
  no alkyl chain or no amide* — four distinct patterns, constant within a ligand, explaining
  **16.9 % of ligand-level variance** (the 36 all-missing ligands sit 1.4 log units below the rest).
  Crucially the flags do **nothing** without ligand descriptors (1.0903 vs 1.0876). **Without this
  artefact the gen6 champion (1.109) is worse than a model with no ligand information (1.088).**
- **Structured metal representation** (RBF over ionic radius + |n_f − 7| + n_f): 1.0198 (none) →
  1.0078 (one-hot) → 0.9976 (continuous) → **0.9932** (physical). Four lines of code; direction
  never reverses; interval straddles zero.
- **Recovered upstream variables:** +0.035 to +0.056 macro, but CI [−0.0505, +0.0976] — directional,
  not established.
- **Ligand–diluent coupling (LIGPHYS):** +0.021 on donor blocks; nothing else in the bundle is a
  function of both ligand and diluent.

#### The two significant negative results

**1. `lig2d_ext` is harmful.** **Deleting** the gen5/gen6 champion's own 206-column descriptor block
is worth **+0.0513 macro, CI [+0.0173, +0.0823], BCa [+0.0203, +0.0858], 78/131 units, p = 0.002.**
`ABL_minus_lig2d` at **0.9696** is the lowest macro MAE anywhere in the study. **It is the only
block whose removal is a significant gain**, and it directly reverses gen5's H1 verdict.

**2. Batch deconfounding is significantly wrong-signed.** Subtracting shrunken per-(ligand, batch)
intercepts from the training target: offset **−0.0754**, CI [−0.1658, −0.0040], **0 of 5 seeds
positive**. A permutation null independently predicted this.

**Why**: the batch hypothesis is **rejected**. Under a within-ligand batch-label permutation null
(200 shuffles per ligand, 40 ligands with ≥2 multi-row batches), the median observed within-ligand
batch spread is **0.1594 against a null of 0.1746** — a *negative* excess; only 20 % of ligands have
p < 0.05, and 42.5 % have positive excess against a chance rate of 50 %. The implied floor on
`offset_mae` from study calibration is **0.133** against an **observed 0.84**. And `ORACLE_batch`
turns out to measure the experimental **series**, not laboratory calibration: 82 % of the 541
batches contain exactly one series, and the two partitions have adjusted mutual information 0.754.

#### Everything else that was rejected

| approach | result |
|---|---|
| pretrained encoders (ChemBERTa ×2, MoLFormer; 16 arms) | 1.097–1.245 — lose to the descriptors they were to replace |
| kernel ridge / exact GPs | 1.198–1.372, behind the trees *and* behind the no-ligand floor |
| hierarchical offset/shape network (FiLM/MoE/physics head) | 1.1329, **does not clear the no-ligand floor**, at 38× the cost |
| alternative learners (13 families) | ExtraTrees wins; **six families do not clear the no-ligand floor**, including CatBoost, XGBoost and both HGB variants |
| 3D (invariant route) | **last of eleven representations** on the isolated level problem |
| 3D (equivariant route) | **skipped with an explicit technical reason**: the bundle has one relaxed pose per complex and no conformational ensemble |
| honest inner-fold model selection | `SELECT_inner` 1.0197 — **loses to four fixed arms**; automatic selection fails at this data size |
| PADRE (pairwise-difference level regression) | +0.0295, CI [−0.0449, +0.0953], 3/5 seeds — **not established** over its own control (the 3-seed reading had called it a genuine improvement) |
| ensembling | equal-weight mean adds **0.004** over its best member; residual correlations 0.872–0.998 — the members are one model |

> **An explicit level head is a worse level estimator than an implicit one.** Every decomposed model
> in the study has the **best shape** (0.482–0.486 vs the monolith's 0.505) and a **much worse
> level** (offset 0.974–1.004 vs 0.847). Since offset dominates macro MAE roughly 5:1, the trade
> nets negative.

#### One-shot calibration — the result that redirected gen8

Per-ligand MAE on held-out chemotypes, k rows drawn *stratified across the ligand's series*:

| k | zero-shot | offset-only | no-model null (mean of k) |
|---|---|---|---|
| 0 | 1.0070 | — | — |
| 1 | — | **0.7138 (−29 %)** | 0.8595 |
| 2 | — | 0.6246 | 0.7567 |
| 3 | — | 0.5906 | 0.7148 |
| 5 | — | 0.5880 | 0.7141 |

**One measurement of a new extractant beats every modelling gain since gen2**, and offset-only
beats the no-model null at every k — **reversing gen5's k-shot null**. (Columns are not comparable
*down* the table: the ligand set shrinks 143 → 132 → 99 as k grows.)

gen7 read affine recalibration as unusable at low k (3.45–5.39 at k=2). **gen8 overturned that** —
see §5.6.

#### Data-quality findings surfaced by gen7

- **The holdback/radiolysis columns the brief expected are empty upstream** — `Holdback_Agent_Name`,
  `Holdback_Agent_Concentration_M`, `Radiolytic_Dosage_kGy` and three more are **0 % populated in
  all 48,138 upstream rows**.
- **What *is* recoverable:** all 83 upstream solvent strings parse into components with volume
  fractions over a 37-component table (dielectric constant, dipole, logP, molar volume, Hansen
  parameters) — the bundle's 34 diluent one-hots had dumped 43 solvents into "other"; plus
  phase-modifier concentration (4.4 %), shaking time (17.8 %, disjoint from contact time, raising
  equilibration-time coverage 63.0 % → 80.8 %).
- **An unmodelled second species, found but not shipped:** the aqueous-phase complexant, parsed out
  of free-text comments — **557 rows (9.3 %)**: DTPA 165, TEDGA 88, DOODA(C2) 88, NaNO₃ 70, and
  more. Permutation-adjusted η² = 0.299, p = 0.000 (0 of 400 permutations); a DOI control gives
  −0.000, p = 1.00. TEDGA depresses cell-centred `log D` by 0.567; malonamide raises it by 0.325.
- **The TODGA wrong-structure batch:** one canonical SMILES carries **21 extractant names, 20 of
  them different chemicals**. 111 dataset rows over 19 chemically distinct ligands carry TODGA's
  SMILES (742 rows upstream, all from one curator and one ingest batch). `pairs.py` already has a
  quarantine for it — **but the level runners never call it.**
- **DMDPhPDA exponent corruption:** 70 matched cells / 140 rows are **one DOI entered twice by two
  curators**; the 70 paired `log_D` differences are **exactly the integers 0/1/2/3/4** (counts
  11/17/14/14/14), tracking acid concentration. One copy is flat in acid, the other rises
  monotonically −3.70 → −0.20. Handled as a quarantine arm, never as an edit to the frozen cohort.
- **The single worst held-out ligand** is `CCCCOP(=S)(CP(=S)(OCCCC)OCCCC)OCCCC` — MAE **4.196**, of
  which offset 4.196 and shape 0.440. (See §8.4: later evidence suggests this is an *identity*
  error, not an exponent error.)

#### Error archaeology, and an arm-specific claim to be careful with

Of the worst 20 held-out ligands for one arm: LEVEL_ONLY 43.8 % of error, SINGLE_BATCH 32.3 %,
SPARSE 14.8 %, **TRUE_EXTRAPOLATION 0 %**. The worst 20 hold 30.7 % of total error, and their
median nearest-training Tanimoto (0.534) is *close to* the overall median (0.611) — **the worst
ligands are not the most chemically distant.** But the "zero true extrapolations" claim is
**arm-specific**: the gen6 champion's worst-20 contains 5 (3.8 %).

#### Two environment/infrastructure traps established here

- **`pip install tabpfn` silently downgraded scikit-learn 1.9.0 → 1.6.1 and pandas 3.0.5 → 2.3.3**,
  shifting the reference arm by 0.0014 — against gen7 effects of 0.004–0.02 and a total chemistry
  contribution of 0.090. It broke cross-suite comparability mid-session.
- **The pooled metric on this cohort is effectively computed on six independent units**: Kish
  `n_eff = 6.256` over 5,248 rows; the largest chemotype holds 63.6 % of rows.

**Frozen as a standing reporting rule:** `NULL_metal_cond` goes in every leaderboard from now on.
Six generations optimised inside a 0.09-wide band without a control that said how wide the band was.

### 5.6 gen8 — the frozen model, few-shot calibration, and acquisition as a first-class lever

**The single most important structural fact about gen8: the global model is frozen throughout.**
`protocol.md` §0 — "no global model is retrained for the primary endpoints." Every number is a
*k*-shot calibration of the frozen gen7 arm `REC_ecfp_plus_recovered` applied to its **own
out-of-fold predictions**. That freezing is what makes the entire study paired down to the row, and
it is what keeps gen8 from being another learner sweep.

**Three populations that must never be conflated:**

| population | size | zero-shot | use |
|---|---|---|---|
| **Table B / common cohort** | identical **99** ligands at every k | 1.061 | the *only* valid longitudinal adaptation curve |
| **Table A / maximal coverage** | **143** at k≤3, 99 at k=5 | 0.995 | the population gen7's references are on — pre-registered thresholds read here |
| **Exhaustive candidate table (P1)** | 143 ligands, 715 cells, **26,105 candidate rows, closed form, no sampling** | RANDOM 0.709 (exact expectation) | "which candidate should I measure" answered exactly |

#### What a measurement buys, and what *choosing* it buys

| step | value |
|---|---|
| one **random** measurement | **+0.268** (CI [+0.140, +0.462], 52/99, 5/5) |
| **choosing** it well (MEDOID vs RANDOM) | **+0.101** (CI [+0.037, +0.146], 77/99, 5/5) |
| … for comparison, **all zero-shot ligand chemistry** (gen7) | **0.090** |
| oracle single point vs random | **+0.260** headroom (0.533 vs 0.793) |

**Which point you measure is worth more than every zero-shot chemistry feature in the programme
combined.** The full attribution from zero-shot 1.061 to the best 5-shot arm 0.474: 0.268 first
measurement + 0.193 more measurements + 0.096 choosing them + **0.029 architecture**.

Adaptation curve (common cohort, best deployable arm): k=0 **1.061** → k=1 **0.667** → k=2 **0.588**
→ k=3 **0.523** → k=5 **0.474**. Marginal value: 0.394, then 0.079, 0.065, 0.025 per measurement.
The whole panel moves together: within-ligand Spearman 0.394 → 0.609, sign accuracy 0.670 → 0.763,
fraction within 0.5 log 0.344 → 0.688, hard-chemistry MAE 1.090 → 0.491.

Pre-registered thresholds: 1-shot "useful" ≤0.65 **met** (0.626), "strong" ≤0.60 **not met**;
2-shot "useful" ≤0.55 **met** (0.530), "strong" ≤0.50 **not met**.

#### Why model uncertainty cannot close the acquisition gap — an algebraic result

Under offset calibration, measuring row *i* gives `MAE = mean_j |r_j − r_i|`, which is minimised at
the ligand's **median residual**. So:

- `argmin |r_i − median(r)|` **reproduces the exhaustive oracle to 6.7e−16 in all 715 cells** — the
  oracle's rule is known *exactly*.
- `argmin |r_i|` — what a perfectly calibrated uncertainty approximates — scores **0.780 against
  random's 0.709**, i.e. **worse than choosing blindly**.

**Acquisition is a centrality problem; uncertainty measures extremity.** The uncertainties are real
(tree-ensemble sd ranks *which extractant* will be hard at across-ligand ρ 0.475) but useless
*inside* a ligand (within-ligand ρ 0.139 against |r|, **0.016** against the deviation from the
median). Both tails lose to random. The only uncertainty-shaped winner, `MIN[u_gp_var_design]`
(0.639, +0.070), is a **labelless design-geometry centrality measure wearing an uncertainty
costume**.

> **A hard audit finding underneath this:** `MAX_ENSEMBLE_SD` and `MIN_ENSEMBLE_SD` in the primary
> harness read an `extra__prediction_sd` column that is **100 % null** for the frozen model, and
> `_argmax_finite` falls back to `rng.choice` — so both arms were **literally the RANDOM policy**
> with a different generator key (verified as functions: 0 finite values out of 26,240).
> `MAX_PREDICTIVE_VARIANCE` at k=1 returns `pool[0]` — an ordering artefact. **The main report still
> quotes "max ensemble sd vs RANDOM +0.025" as an uncertainty result. It is an estimate of pure
> between-draw noise** — and usefully, it calibrates that noise floor at ≈0.025 log units, which is
> **larger than several claimed effects elsewhere in gen8**.

#### Flattening — the largest previously unreported defect

Scoring each of 1,176 reconstructed curves twice (measured vs frozen-model OOF prediction):

| axis | true median slope | predicted | span ratio | model slope MAE | median-slope null |
|---|---|---|---|---|---|
| **extractant** | **2.574** | **0.116** (×0.05) | 0.051 | 2.433 | **0.743** |
| acid | 1.656 | 0.355 | 0.218 | 1.567 | 1.048 |
| lanthanide | 0.084 | 0.040 | — | 0.094 | 0.092 (at par) |

**On the extractant axis the model's slope is three times worse than a null that just predicts the
corpus median slope.** An ensemble asked about an unseen chemotype returns something near the
conditional mean over training ligands, and averaging ligands whose curves sit at different heights
flattens the slope they share.

**The repair uses no target of any kind**: fit the frozen model's own predictions along a curve (a
function of features alone), compare the fitted slope with the training prior for that axis, add the
difference back re-centred so the ligand's level cannot move. On the 26 ligands with an extractant
titration it is worth +0.086 at k=1 (CI [+0.019, +0.125], 24/26, 5/5) and cuts shape MAE by 0.071;
extractant-axis shape MAE goes 0.555 → 0.194. It is **neutral on acid and harmful on temperature —
exactly where the mass-action law is not identifiable.** It is the **only gen8 change that
measurably reduces shape error**.

Corroboration that the physics is really there: on the raw measured data (not predictions),
extractant titrations have **median log–log R² 0.996 and median slope 2.305** across 241 curves,
while contact-time curves are flat noise (R² 0.165).

#### "Shrinkage, not degrees of freedom" — a correction to gen7

gen7 concluded that k=2 supports only one free coefficient, because an affine update scored 3.32
against 0.64. gen8 reproduces the catastrophe (four unpenalised coefficients at k=2 give **1.7e12**)
but shows it is the **penalty, not the parameter count**: the same four coefficients at λ=1 score
**0.571 against 0.640** for the plain offset. At k=1 every mode and penalty collapses to the same
number exactly, because the unpenalised intercept absorbs everything. **"As many coefficients as you
like, held down hard" replaces "only the level can be adapted."** (λ=1 beating the pre-registered
λ=4 is labelled exploratory; headline numbers use λ=4.)

#### Calibration is series-local — the pre-registered falsification that failed

Cross-series transfer matrix (gain over zero-shot on the same target rows):

| support → target | acid | extractant | lanthanide |
|---|---|---|---|
| **acid** | **+0.190** | — | −0.161 |
| **extractant** | −0.066 | **+0.243** | −0.180 |
| **lanthanide** | −0.036 | — | **+0.332** |

**The diagonal is strongly positive and most of the off-diagonal is negative. A ligand does not have
one level; it has a level per series.** This was a pre-registered falsification condition, it
**failed**, and it is reported as a headline rather than buried. The chemist-facing consequence:
*measure in the system you intend to predict* — do not expect one measurement to calibrate a
different diluent. Where to measure: **low acidity is the single worst place** (+0.244 cost versus a
mid-range point).

#### Architecture verdict

Of three families built to the same adapter interface, **only the physics-latent law applied to the
frozen model's *residual* clears the bar** (+0.056 at k=1, +0.079 at k=2 vs offset). And:

- the same law on **absolute** `log D`: **−0.055, 0/5 seeds**;
- an **absolute** conditional neural process: **−0.145, 0/5**;
- **adding ligand features inside the CNP: −0.125, 0/5**.

**Molecular features do not add value after calibration, and inside the CNP they actively hurt.**
What carries the winner is the **functional form**, not the chemistry-to-parameter map (ridge →
gradient boosting is worth +0.003, 3/5). The advantage is concentrated exactly where the mass-action
law is identifiable (acid titration ≥3 levels −0.091; extractant titration −0.112; but *no* acid
titration −0.010 with a CI covering zero). And it **collapses when the acquisition rule already
picks centrally** (+0.024, interval touching zero) — the two mechanisms overlap.

#### Failure classes after calibration

| class | n | zero-shot | after one random point |
|---|---|---|---|
| PURE_LEVEL | 25 | 1.906 | **0.494** |
| RESIDUAL_SHAPE | 29 | 1.350 | **1.471 (worse)** |

**Blind measurement actively harms a large minority: 57 of 143 ligands are worse after one random
measurement.** Choosing centrally reduces that to 42.

And **what distinguishes fixable from unfixable ligands is not chemistry**: all 36 mechanistic donor
descriptors give smallest BH-corrected q = 0.994–1.000; nearest-training Tanimoto gives AUC 0.463
(p = 0.748). What separates them is the target's own spread (measured log D span 4.30 vs 1.58
decades, AUC 0.938) — which the report itself flags as **near-definitional**: a ligand is
RESIDUAL_SHAPE when a constant cannot remove its error, and a wide-spanning ligand is close to
*being* a shape failure rather than explaining one.

Independently, **mechanism-aware similarity does not fix the zero-shot level** either — 1-NN
*Tanimoto* (0.989) beats 1-NN *mechanistic* (1.077) by 0.085, and the best mechanistic model (1.009)
loses to the Tanimoto lookup. That is gen7's ceiling reached by a third independent route.

#### Traps established in gen8

- **`hash()` is salted per process.** The pool/evaluation split was seeded on Python's builtin
  `hash()` of the SMILES, so every number was reproducible *within* a run and different *between*
  runs — architectures evaluated in separate processes were **not paired** while every table claimed
  pairing. Fixed with BLAKE2b; enforced by a test that re-derives the split **in a subprocess under
  a different `PYTHONHASHSEED`**.
- **`shape_mae` is not the floor it is quoted as.** It centres a ligand's residual on its **mean**;
  the MAE-minimising constant is the **median**. Mean-centred 0.503 vs median-centred 0.480 — and
  0.052 apart on exactly the RESIDUAL_SHAPE class the argument rests on. Every "floor" in the error
  archaeology overstates the true offset-only floor by ~5 % on the class that matters.
- **Protocol-draw instability.** An earlier execution of the *identical* protocol gave zero-shot /
  RANDOM_k1 / CENTRAL_k1 = 0.998 / 0.702 / 0.646 against the pinned run's 0.995 / 0.723 / 0.651.
  The two differ in the pool/evaluation split itself. **"A blind 1-shot policy is not determined to
  three decimals by its own protocol — read the RANDOM numbers to two."** The earlier figures still
  circulate in the write-ups.
- **A curve-membership bug** emitted rows with a missing axis coordinate as curve members with a NaN
  abscissa; those records were silently dropped from comparisons, which would have made them
  **unpaired**.
- **An unreported verification result:** a plain fitted-λ shrunk offset (`SHRINK_FITTED`, 0.6958)
  beats both the plain offset (0.7232) and the residual CNP (0.7094), CI excluding zero — and does
  not appear in the main report.
- **24.3 % of rows labelled "no modifier" physically carry one** (1,215 of 5,006), 122 of them
  hidden inside the `diluent = other` bucket.

#### The deployment reframing

gen8's stated conclusion is to **ship the recommender, not the regressor**: the deliverable is *"give
me your candidate conditions and I will tell you which one to run first"*, followed by a calibrated
surface. `scripts/gen8_recommend_experiment.py` takes a SMILES plus a candidate-condition table and
returns the experiment to run, **carrying the measured value of following it rather than a
confidence score.**

### 5.7 gen9 — curve shape: the pre-registered arm fails, an exploratory one wins

**A documentation warning first:** there is **no standalone gen9 narrative report** of the kind gen7
and gen8 have at the repository root. Everything lives in `runs/gen9_shape/` (`protocol.md`,
`decision_report.md`, `failure_analysis.md`, README + ~150 artefacts), plus a Russian section of the
main README.

**Primary question (pre-registered):** can held-out-chemotype error be reduced by training the
predictor to **preserve measured curve shape** rather than repairing flattened curves afterwards?

#### The pre-registered arm failed — and the hypothesis is wrong, not badly implemented

`CurveBoost` boosts small ExtraTrees on `Huber(row) + λ_delta·Huber(within-curve difference) +
λ_span·Huber(span)`. One stage at lr = 1 with λ = 0 is **algebraically the frozen model** (asserted
to 1e-9), so the frozen model is an exact special case of the arm family and every shape claim is
quoted against `A0_ROW_ONLY` — same learner, curve terms off.

Result on the extractant axis: **every sampler significantly reduces dynamic-range compression**
(span-recovery gain +0.033 to +0.107, 25/25 ligands, 5/5 seeds) **and nothing else moves**. Slope
MAE is *worse* for all four (−0.091 to −0.344); the median predicted slope *falls*; shape MAE
worsens; **within-curve Spearman collapses from 0.692 to 0.247–0.444**. The pre-declared primary
candidate came **last**.

The dose–response is a clean monotone trade-off: λ_delta 0 → 0.5 → 2 → 8 gives macro MAE
0.982 → 1.011 → 1.210 → **2.155** while span recovery goes 0.053 → 0.756 and **the acid slope
inverts** from +0.392 to −0.298. **There is no setting where the range returns and the model still
works.** Direction declared closed: *"do not retry a within-curve loss."*

The obvious rescue was tested and failed: a model trained **only** on the curve-centred target — no
level to compete with — recovers 2.5 % of the extractant range, **worse than the monolith**. That
ruled out level-vs-shape competition and produced the actual diagnosis.

#### The diagnosis, and the exploratory arm that worked

**The model had no coordinate for *where on its own curve* a row sat**, because different titrations
span different windows. Two arms were built *after* the pre-registered sweep was read and are
labelled EXPLORATORY throughout:

- **`GEN9_REL_MONOLITH`** — the frozen forest plus five **target-free** columns computed on each
  row's longest curve: `rel__position`, `rel__offset_from_mean`, `rel__window_width`,
  `rel__n_points`, `rel__is_endpoint`.
- **`GEN9_SHAPE_RECOMPOSED`** — (1) fit the monolith → `base`; (2) fit a second forest on the
  *training* curve-centred target `y − mean_curve(y)` with the five columns appended; (3) emit
  `mean_over_curve(base) + shape − mean(shape)`. **Mean-preserving**: the curve's predicted *level*
  is the monolith's; only the *shape around it* is replaced. Structurally this is gen8's post-hoc
  slope repair with the median-slope prior replaced by a model.

| arm | macro MAE (C-FULL, 5 seeds) |
|---|---|
| frozen gen8 global model | 0.9807 |
| `GEN9_REL_MONOLITH` (features only) | **0.9858 — worse** |
| **`GEN9_SHAPE_RECOMPOSED`** | **0.9695** |
| `A0_ROW_ONLY` control | 0.9894 |

**The decomposition is decisive: `offset_mae` is bit-identical between the frozen model and the
recomposition (0.82117153046591), while `shape_mae` goes 0.5025 → 0.4819.** None of the gain can be
level — and that identity is a *checkable design property*, not a measurement.

Extractant axis, frozen → +rel → +recomposition:

| statistic | frozen | +rel | +recomposed | measured |
|---|---|---|---|---|
| median slope | 0.116 | 0.521 | **1.024** | 2.574 |
| slope MAE | 2.433 | 1.984 | **1.510** | — |
| median span recovery | 0.051 | 0.210 | **0.423** | 1.0 |
| shape MAE | 0.665 | 0.567 | **0.469** | — |
| within-curve Spearman | 0.582 | 0.840 | **0.886** | — |
| sign accuracy | 0.762 | 0.903 | **0.926** | — |

All paired CIs exclude zero, 5/5 seeds. On the **lanthanide** axis the recomposition lands the
median predicted slope on the measured value (0.08433 vs 0.08444). Over-steepening guard: only
0.26 % of curves exceed the corpus's own 95th-percentile measured slope.

> **But the pre-registered "strong" bar (median extractant span recovery ≥ 0.5) is MISSED** (0.423),
> met only on the lanthanide axis. And on `RESIDUAL_SHAPE` ligands the pre-registered 0.08 threshold
> is missed (+0.046 vs frozen) — though the direction is exactly what the protocol demanded. On
> `PURE_LEVEL` ligands the same arm is significantly *worse*.

#### Frontier and attribution

99-ligand common cohort: gen9 **1.0358 / 0.6539 / 0.5746 / 0.5109 / 0.4675** at k = 0/1/2/3/5,
against gen8's 1.0605 / 0.6674 / 0.5879 / 0.5230 / 0.4743 — better at every k, with the gain
shrinking 0.025 → 0.007. **Pre-registered frontier goals were NOT met.**

Attribution: **the global model is the only component that pays at every k** (+0.031/+0.032/+0.031,
5/5 seeds); **learned acquisition costs −0.075 at k=1**; the series adapter pays only at k≥2.

> **Envelope trap (flagged by the figure audit):** those frontier numbers are the **minimum over
> every deployable arm at each k**, not a fixed recipe — and the winning k=2 arm *differs between
> generations*. Restated with **one pre-specified rule per generation**: gen8 1.0605/0.6752/0.5909/
> 0.5230/0.4772 vs gen9 1.0358/0.6539/0.5761/0.5127/0.4714.

#### gen9-B: learned acquisition fails its pre-registered target

`LEARNED_BLEND` 0.6505 vs `MEDOID` 0.6453 vs `RANDOM` 0.7205 vs `ORACLE` 0.4812. Against MEDOID:
−0.005, CI [−0.018, +0.014], **0/5 seeds**. Pre-registered target was ≥30 % of the MEDOID→oracle
gap; **achieved 0 %.** After the issue-8 fix the blend chooses **the pure geometric anchor in 81 of
100 selections**, and the chosen anchor is plain centrality in 99/100. More information makes it
*worse*, and the shuffled control shows there was nothing there.

#### Eight defects found in gen9's own code — three that changed conclusions

- **Issue 1 (Severity A, invalidated a whole study):** the acquisition script read an OOF parquet
  that **carries no condition columns**, and the axis helper returns zeros for a missing column — so
  **every geometry-driven policy saw an all-zero condition space**, tied on every candidate and
  returned the first index. Seven policies that disagree by construction all scored *the same
  number*. Discarded and rerun.
- **Issue 6 (Severity B):** `CurveBoost` was **not reproducible**. `ExtraTreesRegressor(n_jobs=-1)`
  sums trees in thread-completion order (~1e-15); inside a boosting recursion that reaches the next
  stage's pseudo-residual, flips a near-tied split, and compounds. **Two runs of the identical
  configuration in the same process differed by up to 0.1588 log units on all 382 held-out rows.**
  Fixed by rounding the running prediction to 1e-9 after each stage. The whole sweep was rerun at
  five seeds. (gen10 hit the same mechanism twice in new code and now calls it "the gen9 issue-6
  mechanism in new clothes".)
- **Issue 8 (Severity B):** `LEARNED_BLEND` could never fall back to the baseline it was built to
  nest, because its blend weight was selected on the very blocks the ranker was fitted on.
- **Issues 3/4/5 (data audit), and the lesson that generalises:** the duplicate-cell scan keyed on
  the full condition set, so DMDPhPDA's two copies (differing only by an unreported temperature)
  landed in different cells and **the known corruption was invisible** — 0 exact-decade pairs found;
  61 after the fix. The condition-adjusted level fell back to a ligand's own value, reporting **every
  level as exactly 0.0** and finding zero outliers (5 after the fix). In all three cases **the
  absence of an expected finding was the only signal.** The stated rule: *a scan that finds nothing
  should be tested against a known positive before it is believed.*

**Single-seed probes were actively misleading twice.** One arm looked like the best trade-off in the
study at one seed and at five seeds is the **worst arm in it** (macro 1.273, Spearman 0.090).

#### Documentation defects inside gen9's own reports

`decision_report.md` variously says eight issues (§0.3 table) and seven (§8); says 125 tests in the
header and 112 in §0.4 (125 is correct); says the manifest verifies **163** artefacts when
`manifest.json` records **154** and `--verify` prints 154/154. Two claims in §6 are **pre-fix
statements that survived** into a report that promises no pre-fix number appears in it — one of them
reversed sign after the fix. And §4.2 labels a table "common cohort" when the values come from the
**maximal-coverage** table (only the k=5 column coincides) — a material labelling error in a project
whose standing rule is that a number without its regime is worse than no number.

**Manifest integrity, verified in this session:** `154 recorded, 154 on disk: 0 changed, 0 missing,
0 new`.

> **gen9's own defect, discovered later by gen10:** the relative-position representation makes the
> model's answer for a point **depend on the candidate design supplied with it**. Adding two decoy
> points 2–3 decades outside the window shifts `SHAPE_RECOMPOSED` predictions by a median of
> **0.117** log units (p95-of-p95 0.68); the frozen model moves by exactly zero. **A decoy point two
> decades away moves a prediction by more than gen9's whole 0.025 macro gain.** gen9 never measured
> this. gen10's verdict: freeze the model, **report the dependence**, and tell the user to hand over
> the plan they will actually run.

### 5.8 gen10 — finalisation: what got frozen, and why the programme stopped

gen10 asked the question gen9 could not: **gen9's shape gain comes from five *design-relative*
columns that position a requested point inside the titration the user proposed — so a prediction is
a function of the whole candidate list. How much does it move when the list changes?**

#### The frozen pipeline

| component | choice | note |
|---|---|---|
| global model | **`GEN9_SHAPE_RECOMPOSED`** | unchanged from gen9, bit-reproduced inside gen10's classes (|Δ| ≤ 1.3e-15). **gen10 changed no part of the zero-shot model.** |
| first point | **`CENTRAL_THEN_SPREAD`** | medoid of standardised condition axes, then spread |
| k = 1 adapter | gen8's **`SLOPE_L_s1_K1`** | at k=1 every ridge adapter is algebraically identical (verified 3e-14), so this is a choice about the *slope repair* |
| k ≥ 2 adapter | **`SERIES_ML`** | **the single change gen10 makes** |

`SERIES_ML` is a ridge on `[1 | centred series indicators | standardised acid, extractant,
lanthanide]` with a free intercept and per-family penalties set by the **marginal likelihood of the
fold's training ligands' out-of-fold residuals**.

**The locked frontier** (99-ligand common cohort, macro MAE):

| k | gen8 | gen9 | **gen10 (frozen)** |
|---|---|---|---|
| 0 | 1.0605 | 1.0358 | **1.0358** |
| 1 | 0.6674 | 0.6539 | **0.6539** |
| 2 | 0.5879 | 0.5746 | **0.5593** |
| 3 | 0.5230 | 0.5109 | **0.4928** |
| 5 | 0.4743 | 0.4675 | **0.4405** |

k = 0 and k = 1 are gen9's numbers reproduced exactly; **the whole gen10 gain is the adapter at
k ≥ 2** (−0.015 / −0.018 / −0.027, all with BCa intervals excluding zero and 5/5 seeds).

#### The headline negative result: the model's answer depends on the question

Holding the **absolute point fixed** and varying the candidate design around it, across seven
families (two of which are exact-invariance controls that hold at 8.9e-16, and where the design-blind
frozen model shifts by **exactly 0.0**):

| perturbation | median shift | p95-of-p95 |
|---|---|---|
| **add two decoy points 2–3 decades outside the window** | **0.117** | **0.681** |
| extend one boundary 0.5–2 decades | 0.073 | 0.590 |
| drop nested end points | 0.046 | 0.518 |
| thin to 2–8 points | 0.042 | 0.641 |
| interior density | 0.018 | 0.623 |

**A decoy point two decades away moves a prediction by more than gen9's entire 0.025 macro gain.**
The shift is almost entirely **shape** (0.121 vs 0.024 level) and is largest for the **narrowest**
windows. **No gen10 architecture removes it** — the rank/spacing alternatives halve the decoy tail
and *double* the interior-density sensitivity. The pre-registered robustness criterion was met by
nothing. The verdict: freeze the model, **report the dependence**, and require the user to hand over
the candidate list they will actually run.

#### Every architectural alternative failed — and all of them failed on the *level*

| arm | macro MAE | offset MAE |
|---|---|---|
| **`GEN9_SHAPE_RECOMPOSED`** (frozen) | **0.9695** | 0.8212 |
| explicit level + shape heads | 0.9818 | 0.8300 |
| cross-fitted residual shape | 0.9821 | — |
| DeepSets set encoder | 1.0059 | **0.8212** |
| attention set encoder | 1.0087 | **0.8212** |
| two-branch / interaction | 0.996–1.003 | — |

This is the **third independent reproduction** of gen7's "the level is not chemistry". The learned
set encoders are especially clean evidence: they **match the handcrafted extractant shape**
(0.452–0.459 vs 0.469) with the level held **bit-identical**, and still lose macro.

**Feature access is closed as an explanation** (Phase 2B): no `max_features` setting (sqrt / 0.25 /
0.5 / 1.0) and no context-column replication (×8, ×32) approaches the recomposition — best 0.9852
against 0.9695. **The recomposition is an architecture, not a split-proposal artefact.**

**Learned acquisition is closed** for the second time: trained on *realised* one-shot regret,
`LEARNED_BLEND` scores 0.6342 against MEDOID's 0.6286 (0/5 seeds), and **the honestly-chosen blend
weight is 1.0 — pure centrality — in 24 of 25 folds.** gen10 also showed the gen8 *surrogate*
objective `|r − median(r)|` is itself a poor proxy under the deployment protocol: even its oracle
recovers only a third of the true oracle gap.

#### The error ceiling — why the programme stopped

Decomposing the 0.970 zero-shot macro MAE (on stored OOF predictions, nothing refitted):

| component | size |
|---|---|
| per-ligand constant **level** | **0.520** (one central measurement realistically removes 0.349) |
| within-curve **shape** | 0.192 (only 0.008 removed at k=1) |
| series-local level | 0.057 |
| sparse chemotype support | 0.060 |
| data quality | 0.037 |
| system identity | −0.002 |
| **unexplained** | 0.103 |

Oracle chain: current macro **0.9695** → per-ligand oracle constant **0.4495** → per-series
**0.3928** → per-curve level **0.3860** → **per-curve level *and slope* 0.1941 (the floor)**.

> **A zero-shot architecture can only touch the 0.19 shape component and the 0.10 unexplained, and
> gen9/gen10 already took half the shape. Conclusion: no further model generation is justified on
> this cohort. The lever is measurement allocation.**

Programme-level stopping condition, stated explicitly: revisit only when (a) extractant titrations
exist on new chemotypes (there are **8 chemotypes** carrying them today), or (b) the data problems
are resolved at source.

#### Measurement economics (Phase 9)

Marginal value of successive measurements on an unseen ligand: **first point 0.382**, second 0.078,
third 0.063, fourth/fifth 0.021 each. (Medians are much smaller — 0.166 / 0.048 / 0.046 / 0.013 —
the mean is driven by a tail of badly-levelled ligands.) First-point value by distance to training
chemistry: **far tercile 0.495, mid 0.448, near 0.235.**

Budget allocation at B = 49 over 99 ligands: new-chemotype-first **0.782** < max-min diversity 0.836
< one point on random ligands 0.847 < two each 0.924 < three each 0.953 < five each 0.986 (oracle
ordering 0.680). **Breadth beats depth until every ligand has one point** — but note the ordering
**reverses at B = 24**, where diversity beats new-chemotype-first.

#### Data-quality strata (Phase 7)

| cohort | rows / ligands | macro MAE |
|---|---|---|
| A consistent | 4,691 / 145 | 0.947 |
| B name/structure mismatch | 271 / 15 | 1.000 |
| C exact-decade duplicates | 218 / 3 | 1.302 |
| D TWE-24 | 6 / 1 | **4.098** |
| E flagged level outliers | 62 / 3 | 1.405 |

**Mismatch rows carry a significantly larger *irreducible* residual** (median 0.934 vs 0.582 after a
per-ligand offset; ligand-level p = 0.0018) — so the missing second species changes **shape, not
only level**. Against an honest no-chemistry baseline the model **wins on clean rows** (0.536 vs
0.639, 5/5 seeds, p = 4.2e-10) and **loses badly on data-artefact rows** (duplicates 1.170 vs 0.334,
0/5). But in aggregate the damage is small: pooled MAE would fall by **0.0148 of 1.1218 (1.3 %)** if
every non-consistent row errored like a consistent one.

#### Robustness of the one gen9/gen10 zero-shot gain

The extractant-axis shape gain (+0.196 shape MAE, recomposed vs frozen) is positive under
chemotype-, ligand-, publication- and two-factor-blocked bootstraps. **Quote the publication
interval [0.140, 0.257] or the two-factor [0.140, 0.258], not the chemotype [0.118, 0.203]** — the
chemotype interval is the narrowest *because only 8 chemotypes carry extractant titrations*, and the
source explicitly labels it the optimistic one. Leave-one-publication-out over 32 publications: the
largest carries 16.9 % of the summed gain; removing any single one leaves ≥ 0.176.

#### Deployment recommendation, as written

> Measure the new extractant **once**, at the **medoid of your plan**, **inside the series you intend
> to predict**. Spend budget breadth-first, new-chemotype-first. **Hand over the complete list of
> conditions you will actually run and keep it with the predictions, because the answer depends on
> the question.** Do **not** trust zero-shot levels for decisions, mixture rows, extrapolation
> outside measured windows, or adjacent-lanthanide selectivities.

#### gen10's own defects (seven in new code, all caught pre-result)

- **Defect 1** — `ResidualShapeModel` irreproducible at **3.4e-2** on 372 of 382 held-out rows: the
  base forest's unrounded prediction fed the correction target and ~1e-15 thread-order noise grew a
  materially different second forest. **This is gen9's issue 6 recurring in new code.** Fix:
  `stabilise()` rounds every model output that becomes another model's target to 1e-9; final
  predictions are never rounded.
- **Defect 2** — the same arm's correction target was the base forest's **in-sample** residual,
  which an ExtraTrees with `min_samples_leaf=2` nearly interpolates (target sd 0.198 vs 0.735
  held-out). The second stage had almost nothing to learn. Fixed by inner-chemotype cross-fitting;
  **the in-sample version is retained as a labelled trap control**.
- **Defect 3 (a leak)** — Phase 7's no-chemistry baseline pooled **same-fold** ligands into the
  condition-cell lookup and "beat" the model 0/5 seeds. **Caught because the sign of the comparison
  was implausible.** Fixed to other-fold rows only; the comparison reversed.
- Plus a salted-`hash()` RNG seeding (removed; an **AST scan** now enforces this across gen10 and
  the shared gen6/gen8 modules), a benchmark that would not finish, a decomposition producing
  negative contributions, and two bugs inside the self-audit script itself.

#### Two gen9 claims retracted by gen10

1. **gen9's `SERIES_MAP` adapter is worse than gen8's hand-set `OFFSET_K3`** by 0.075 macro MAE at
   k=3 (CI [−0.103, −0.052]) — the gen9 hierarchical adapter was a regression.
2. **gen9 diagnosed its estimator as *over*-shrinking; gen10's rediagnosis is the opposite** — the
   prior put the series family at the penalty **floor** (near-free series deviations), which overfits
   with two measured points. gen9's diagnosis is called "half right at best".

#### Deliberate non-cherry-picking, worth noting as a discipline

Several arms **beat** the frozen pipeline and were **not frozen** because the pre-registered bar was
0.02 macro MAE:

- `GEN10_REC_GEN9_PLUS` is better at **every** k with **5/5 seeds and BCa intervals excluding zero** —
  by 0.0014–0.0049. Left on the table.
- `GEN10_REC_HYBRID` buys +0.075 extractant shape MAE at neutral macro. Reported, not frozen.

> **Verified reproducibility:** identical configuration twice in one process agrees at 1.11e-15;
> fresh subprocesses under `PYTHONHASHSEED` 0/1/4242 agree at 8.9e-16; 214 tests pass; gen9's
> manifest re-verifies 154/154 with zero drift; the final self-audit passes 17 checks.

> **But nothing was committed.** The entire 353 MB gen10 run sits **untracked** in an agent-managed
> git worktree — see §10.

### 5.9 gen11 — multi-metal auxiliary transfer (unfinished, worktree-only, and its own report is stale)

**Read this section carefully: gen11 is the least settled part of the programme, and the only
narrative document describing it is superseded by artefacts written after it.**

**The question.** Can an audited multi-metal archive — 40 metals, mostly actinides — improve
prediction for *unseen lanthanide* chemistry without touching the frozen gen10 benchmark? There is
**no brief markdown**; the pre-registration lives in code (arm list, weightings, negative controls,
matched seeds, and a five-part stopping rule A–E with explicit thresholds).

**How auxiliary data enters.** From the 16,770-record multi-metal archive, the 5,992 frozen-bundle
rows join 1:1 and are **excluded**, leaving 10,778 candidates of which 5,438 are model-ready
(5,096 cells after replicate collapse; ~4,247 admissible per fold after leakage filtering). Rows are
featurised into the frozen 2,146-column contract, namespaced, and **enter training only** — they are
never scored (`assert_evaluation_unchanged`: 0 extra rows, max target delta 0.0 in every arm).

#### The critical design confound — why the comparison is not against the frozen anchor

The frozen 3-column metal block **cannot describe the auxiliary rows**: **5,353 of 5,438 (98.44 %)
have all three columns NaN**, which the median imputer silently renames **europium**. Two MASSACTION
columns are null on 0.02 % of cohort rows against 98.44 % / 40.66 % of auxiliary rows — i.e. they
would become an **"is-auxiliary" provenance flag**. So a richer metal representation (`GENERAL`) and
dropping those two columns (`ANNOTATION_SAFE`) are **forced on every auxiliary arm**.

**Every auxiliary arm therefore sits at a changed design corner, and the honest reference is the
design-matched control, not the frozen anchor.** The design change alone is worth −0.0038 (not
separated from zero), and `ANNOTATION_SAFE` on the frozen metal block is **significantly worse**
(−0.0086, BCa [−0.0156, −0.0041], 0/5 seeds).

#### The leaderboard (5 seeds, macro MAE per ECFP cluster — independently reconfirmed)

| arm | macro MAE | transfer effect vs design-matched control | seeds |
|---|---|---|---|
| **C_LN_PLUS_ACTINIDES** | **0.92410** | **+0.0491** | 5/5 |
| **E_LN_PLUS_ALL** | **0.92598** | **+0.0473** | 5/5 |
| *design-matched control* | *0.97324* | — | — |
| *frozen gen10 anchor* | *0.96948* | — | — |
| B_LN_EXPANDED | 0.97695 | −0.0037 | 2/5 |
| **D_LN_PLUS_NON_ACTINIDE** | 1.02662 | **−0.0534** | **0/5** |

> **The single most important qualifier:** the compiling agent ran gen11's own chemotype-blocked
> paired bootstrap and **the C and E gains do not separate from zero** — C +0.0491, percentile
> [−0.0123, +0.0996], **BCa [−0.0005, +0.1187]**; E +0.0473, BCa [−0.0002, +0.1138]. **D's negative
> transfer *is* significant** (BCa [−0.1318, −0.0101]). So the headline reads: *a large,
> seed-consistent point estimate whose interval touches zero, alongside a statistically supported
> harm from the wrong auxiliary chemistry.*

#### Is it actinide chemistry or just more rows? Not yet separated

The sign is chemistry-dependent, not size-dependent — D adds 456 rows and is **worse** by 0.053 in
5/5 seeds while C adds 4,896 and is better by 0.049 in 5/5. **But D is a degenerate arm**: 7
structures, one extractant system holding 70.8 %, effective chemotypes 1.35. "Non-actinides hurt" is
a statement about a handful of systems. **The clean test — actinides size-matched to D versus a
random size-matched draw — was never run**, so chemical relevance is still not separated from row
count at equal n.

#### Where the effect sits — and it is not where the hypothesis predicted

- **The gain is entirely level, not shape** (C: offset +0.0492, 5/5; shape +0.0026, 3/5). This
  **inverts the module's own stated mechanism**, which reasoned that auxiliary metals plausibly teach
  *shape* and implausibly teach a lanthanide's *level*.
- **It is absent exactly where the pre-registered criterion says it should be largest.** On the far
  chemotype band C is **−0.0077**; on the gen6 hard endpoint (nn < 0.4) C is **−0.0220 (1/5 seeds)**
  and E −0.0081 (1/5). At nn < 0.6 both are +0.069, 5/5. **It is a mid-band, level-only effect.**
- **Criterion E (robustness) FAILS.** The auxiliary arms are markedly *more* decoy-sensitive than the
  anchor: DECOY median shift 0.1380 (C) and 0.1625 (E) against 0.0918 (control) and 0.0783 (anchor)
  — C is +23.5 % worse than the control (79.1 % of units worse, p = 7e-12) and E +43.6 % (91.0 %
  worse, p = 4e-24). The transfer gain comes with a **robustness cost**.

#### Leave-one-metal-out: americium alone carries it

Recomputed from raw OOF (the method reproduces the two on-disk values exactly): **Am +0.1481**,
all-actinide +0.1205, and then a random size-matched removal at **+0.0291** — after which
**every other individual metal has a negative contribution** (U −0.0166, Th −0.0170, Pu −0.0061,
Np −0.0040, Cm −0.0049 …). Am is 1,664 of 5,096 pool cells (32.7 %).

> **But this is one seed only**, and the two size-matched random draws span **−0.0153 to +0.0291** —
> so any per-metal contribution inside that band is refit noise, not a measurement. Only Am and the
> actinide group clear it.

#### What the archive actually adds

**Metals and curves, not ligands.** 3,278 of 5,438 auxiliary rows (60.3 %) have a fingerprint-
identical cohort ligand; only 932 (17.1 %), on 45 structures / 38 new chemotypes, fall outside every
cohort chemotype; the row-weighted median maximum Tanimoto to the cohort is **1.0**. Usable curves
go 1,229 → 2,206.

#### Engineering receipts (these are strong)

- **Leakage: 14/14 adversarial checks pass across 25 folds** — and the audit **found and fixed a real
  defect**: a dictionary-based chemotype rule let **8,615 (row, fold) instances survive inside a
  held-out chemotype** (255 at Tanimoto 1.0); the closure rule drops it to **0**, max survivor
  Tanimoto 0.697. As a control, gen10's own folds put 0 cohort train/test pairs above 0.7.
- **2,986 (auxiliary, cohort) pairs describe the same experimental cell** (55.7 % bit-identical
  `log_D`); the headline policy removes all of them on all 25 folds.
- **Featurizer bit-reproduction of the frozen bundle is exact**: **0 mismatching cells of
  12,673,080**. (62 of 41,984 MASSACTION cells differ, fully explained by modal-value drift in two
  derived columns.)
- gen11 nests gen10 at 1.33e-15 when the auxiliary block is empty; 27 invariant tests pass.

#### Status: unfinished

**8 of 34 pre-registered arms have run** — the `primary` stage only. The mechanisms, weighting,
controls, matched-size and policy stages (26 arms) are all unrun, **including the F/G size-matched
pair that would settle the chemistry-vs-quantity question**.

> **The stale-report problem, stated plainly.** `GEN11_DECISION_REPORT.md` and its tables were
> written 2026-08-23 **00:24**, when only 4 controls plus arm B existed; the leaderboard and
> per-seed scores were rewritten at **13:35** with all 8 arms (mtimes confirmed). **The report's
> central verdict — "No auxiliary arm passes any criterion" — is superseded and must not be
> quoted.** Its design-effect numbers, its B numbers and its composition table remain correct.
> Similarly, the negative-transfer aggregates were written at 00:27 when 2 of 17 LOMO arms existed;
> the other 15 were fitted later and **never aggregated**.

**Honest one-line summary of gen11:** *a large, seed-consistent, level-only improvement from
actinide data whose confidence interval touches zero, measured against a design-matched control that
the auxiliary data itself forced, carrying a measured robustness cost, attributable almost entirely
to americium at one seed, with the decisive size-matched control never run — and described by a
report that predates its own results.*

---

## 6. What is actually known — the consolidated picture

Eleven generations reduce to six load-bearing findings. Each is stated with the strongest evidence
and its main limitation.

### 6.1 The quantity being predicted is mostly not a property of the ligand

A model given metal and conditions and **no ligand information at all** scores macro MAE **1.0995**
(5 seeds, C-FULL); the best ligand-aware arm scores **0.9695**. **Everything six generations of
fingerprints, descriptors, donor censuses, 3D geometry and pretrained transformers contribute is
≈0.09–0.13 log units** — and a paired per-ligand test says the part landing on the *level* is
indistinguishable from zero (p = 0.42, and it **weakens** as the independence unit gets stricter),
while only the part landing on the *shape* is real (p = 0.007, and it **strengthens**).

**Reached independently three times:** (a) the isolated level regression, where **no model of any
kind beats a 1-NN Tanimoto lookup** on the condition-adjusted level (6.4 % captured); (b) gen8's
mechanism-aware similarity study, where Tanimoto beats a purpose-built mechanistic metric; (c)
gen10's architecture sweep, where explicit level heads, residual corrections, two-branch models and
learned set encoders **all** lose — and all lose on the *level*, never the shape.

*Limitation:* the cohort has ~152 training ligands and only ~10 ECFP clusters below Tanimoto 0.4
from their nearest neighbour. The claim is "not recoverable from ~120 training examples", not "not
recoverable in principle".

### 6.2 Chemical coverage, not model capacity, is the binding constraint

The only interventional experiment in the programme: identical learner, identical folds,
**byte-identical test rows**, only the training row mask changes. Adding 61 sparsely-measured
extractants (**+7.0 % rows**) is worth **+0.163 macro MAE** (BCa [+0.013, +0.295], 5/5 seeds), rising
to **+0.463** on chemistry with no close analogue in training. A row-budget-matched control is
indistinguishable from the full expansion (+0.002) and a target-shuffled control recovers nothing.
Anti-circularity comes from a **dose-response** (Spearman +0.290, p = 2.9e-4). At equal row budget,
breadth beats depth by 0.28–0.58 macro MAE at every budget, CI-clean.

*Limitations:* 57 of 131 scoring clusters consist entirely of added ligands and carry the whole
effect; on already-coverable chemistry the gain is +0.019. The depth curve **had not saturated**, so
nothing is bounded from above. Part of the effect is condition transfer, not only chemistry.

### 6.3 One measurement beats every modelling gain in the programme's history

| step | value (macro MAE) |
|---|---|
| all zero-shot ligand chemistry (gen7) | 0.090–0.13 |
| **one measurement of the new extractant** | **≈0.38–0.39** |
| **choosing *which* measurement** | **+0.063 to +0.101** |
| every architectural change gen7→gen10 combined | ≈0.03 (zero-shot) + 0.027 (adapter at k≥2) |

The full frozen frontier: **1.0358 → 0.6539 → 0.5593 → 0.4928 → 0.4405** at k = 0/1/2/3/5. This
reversed gen5's k-shot null (which had found the model adds nothing over a no-model ΔZ fit from
k ≈ 2 on the *pair* target) — the level target behaves differently.

*Limitations:* **one measurement is not unconditionally safe** — 33 of 99 extractants are made
*worse* by a random one (28 % even under the best policy), and there is a `RESIDUAL_SHAPE` class of
~29 ligands that a single point makes worse on average. Calibration is **series-local**: it transfers
within a curve type and largely not across them.

### 6.4 Acquisition is a centrality problem, and uncertainty is the wrong signal

Under offset calibration the oracle rule is **exactly** `argmin |r_i − median(r)|` (reproduced to
6.7e-16 in all 715 cells). What a perfectly calibrated uncertainty approximates, `argmin |r_i|`,
**loses to random** (0.780 vs 0.709). Empirically both uncertainty tails lose; every learned
acquisition rule, in two independent formulations (surrogate label and realised regret), fails its
pre-registered bar; and **the honestly-chosen blend weight is pure centrality in 24 of 25 folds**.

The deployable rule: **measure at the medoid of your candidate plan** — mid-range acidity, mid-series
lanthanide, a condition where the model predicts a middling value — **inside the series you intend to
predict**. Low acidity is the single worst place to measure (+0.244 cost).

### 6.5 The model draws every unseen titration far too flat, and that is repairable

Measured median extractant-titration slope **2.574**; predicted **0.116** — a factor of twenty; on
the acid axis 1.656 vs 0.355. On the extractant axis the model's slope error is **three times worse
than a null that just predicts the corpus median slope**.

Two repairs work, both **target-free**: gen8's per-curve prior injection (extractant shape MAE
0.555 → 0.194) and gen9's mean-preserving recomposition (slope 0.116 → 1.024, span recovery
0.051 → 0.423, within-curve Spearman 0.582 → 0.886, shape MAE 0.665 → 0.469). **Neither touches the
level** — the recomposition's offset MAE is bit-identical to the baseline's, by construction.

*Limitations:* the pre-registered "strong" span bar (0.5) was missed. Only **8 chemotypes** carry
extractant titrations, so the chemotype-blocked interval is the narrowest available — quote the
publication-blocked one (+0.196 [0.140, 0.257]). And the mechanism introduces a **new fragility**
(§6.6).

### 6.6 The winning mechanism makes the answer depend on the question

gen9's relative-position columns position a requested point inside the titration *the user proposed*,
so a prediction is a function of the whole candidate list. Adding two decoy points 2–3 decades
outside the window shifts predictions of the **unchanged** points by a median of **0.117** log units
(p95-of-p95 0.681) — **larger than the entire 0.025 macro gain the mechanism buys**. The design-blind
model shifts by exactly zero. No architecture removes it; the alternatives trade the decoy tail for
interior-density sensitivity. **This is reported, not hidden**, and the deployment instruction is to
hand over the candidate list you will actually run.

### 6.7 Where the remaining error is, and why the programme stopped

Of 0.970 zero-shot macro MAE: **level 0.520**, curve shape 0.192, series-local level 0.057, sparse
chemotype support 0.060, data quality 0.037, **unexplained 0.103** (components overlap; this is not
a partition). The oracle chain: 0.9695 → 0.4495 (per-ligand level) → 0.3928 (per-series) → 0.3860
(per-curve) → **0.1941 (per-curve level *and* slope — the floor)**.

**A zero-shot architecture can only touch the 0.19 shape and the 0.10 unexplained, and half the
shape is already taken. The stated conclusion is that no further model generation is justified on
this cohort; the lever is measurement allocation.**

---

## 7. The frozen deliverable

### 7.1 Model card, in brief

| item | value |
|---|---|
| **task** | predict `log₁₀ D` for a (extractant SMILES, lanthanide, fully specified conditions) row |
| **global model** | two ExtraTrees regressors (400 trees, `max_features` 0.30, `min_samples_leaf` 2), training rows weighted so each ECFP cluster contributes equally, predictions clipped to 1.5× the training range |
| — forest 1 | `log D` from ECFP + 2D descriptors + recovered solvent/system variables + lanthanide index and properties + raw conditions + a **mass-action log-concentration block** |
| — forest 2 | the **curve-centred** target from the same design **plus five relative-position columns** |
| — combination | `prediction = mean_over_curve(forest 1) + centred shape of forest 2` — **mean-preserving**, so the per-extractant level is bit-identical to forest 1 alone; rows on no curve keep forest 1 |
| **first measurement** | `CENTRAL_THEN_SPREAD` — medoid of the standardised condition axes, then farthest-from-selected |
| **k = 1 adapter** | target-free slope repair + offset (`SLOPE_L_s1_K1`) |
| **k ≥ 2 adapter** | `SERIES_ML` — ridge-shrunk update of intercept, response coefficients and series-local terms, with shrinkage set by **maximising the marginal likelihood of the *training* extractants' out-of-fold residuals**, so no held-out extractant informs its own penalty |
| **the global model is never refitted during calibration** | — |
| **performance** | macro MAE **1.0358 / 0.6539 / 0.5593 / 0.4928 / 0.4405** at k = 0/1/2/3/5 (C-COMMON, 99 extractants, chemotype hold-out, 5 seeds × 12 repeats) |
| **reproducibility** | identical config twice in one process: 1.11e-15; fresh subprocesses under three `PYTHONHASHSEED` values: 8.9e-16; 214 tests pass; 17-check self-audit passes |

### 7.2 The deployment protocol, as written

> Measure the new extractant **once**, at the **medoid of your candidate plan**, **inside the series
> you intend to predict**. Spend budget **breadth-first, new-chemotype-first**. **Hand over the
> complete list of conditions you will actually run and keep it with the predictions, because the
> answer depends on the question.**

### 7.3 Stated limits — do not use it for these

- **Zero-shot levels for decisions.** The per-ligand level is the dominant error term and is not
  predictable from structure at this data size.
- **Mixture rows.** 557 rows carry an unmodelled aqueous complexant; 271 rows have a recorded name
  that does not match the modelled structure, and those rows carry a **significantly larger
  irreducible residual** even after a per-ligand offset.
- **Extrapolation outside measured windows** — and note that the prediction for a point *inside* the
  window also moves when you add candidates outside it.
- **Adjacent-lanthanide selectivities.** The programme's own pair-target work (gen2–gen4) never beat
  a trivial baseline on pooled metrics, and the metal axis is 8 % of `log D` variance.
- **Any absolute reading of predicted `log D`.** Predictions are shrunk conditional estimates:
  `sd(predicted) 0.85` against `sd(measured) 1.66`, mean residual **−0.39** (systematic
  under-prediction on held-out chemistry). Rescaling to match the truth variance makes MAE **worse**
  — the shrinkage is MAE-optimal and cannot be corrected away.

---

## 8. Catalogue of negative results and methodological traps

This is the most transferable content in the programme. Most of these were found by the project's
own adversarial audits, and several **reversed a published conclusion**.

### 8.1 Evaluation traps

| trap | what happened |
|---|---|
| **Macro vs pooled inversion** | One extractant is 43.6 % of the pair cohort. A trivial baseline beats the champion on pooled MAE, pooled R² **and** sign accuracy while losing by 0.13 on macro. In the sharpest case, a **label-shuffled control beat the real model on pooled R² by 0.236** — shuffling the targets *raises* pooled R², because regressing to the label mean is what pooled R² rewards on a dominated test set. |
| **Effective sample size** | Kish `n_eff` is **4.8 groups** for the pair cohort and **6.3** for the level cohort's pooled metric. Row-level CIs overstate confidence by orders of magnitude. Always resample the chemistry unit. |
| **Two incompatible definitions of one metric** | `within_ligand_r2` had a deployable form and one that granted a **free per-ligand offset taken from held-out data**; removing it flips the sign (+0.256 → −0.094). Its tell was that every constant-per-ligand arm scored *exactly* 0.0000. `span_recovery` and `shape_mae` have similar guarded/unguarded and mean/median variants. |
| **`shape_mae` is not the floor it is quoted as** | It centres on the **mean**; the MAE-minimising constant is the **median**. Every "floor" derived from it overstates the true floor by ~5 % on exactly the class the argument depends on. |
| **Best-of-N envelopes vs fixed rules** | Published "frontier" columns are the **minimum over eleven deployable arms at each k**, with the winner differing between generations. Comparing an envelope with a pre-specified recipe flatters the baseline. |
| **Cohort switching** | The same frozen model scores 0.9695 and 1.0358; the "all chemistry is worth X" figure is 0.090, 0.13 or 0.083 depending on seeds and cohort. |
| **Averaging-axis switching** | 5 model seeds on 1 split seed vs 5 split seeds on 1 model seed give different values for the same arm (0.316092 vs 0.319221). |
| **Selection on the plotted metric** | Ranking recipes by `idxmin` over ~15 candidates, with **no selection penalty**, on per-fold standard deviations that *exceed the differences being ranked*. |
| **The selection-seed illusion** | All three apparent gen4 gains evaporated on confirmation seeds; two reversed sign. |
| **Single-seed probes** | Misleading twice in gen9: an arm that looked like the study's best trade-off at one seed was **the worst arm in the study** at five. |
| **A cross-machine floor** | An effect below **≈0.01 macro MAE is indistinguishable from a change of machine** (measured, not assumed). |
| **The random-draw noise floor** | A degenerate "uncertainty" policy that was secretly the random policy calibrated between-draw noise at **≈0.025** — larger than several claimed effects. |

### 8.2 Modelling traps

- **The two-stage residual trap.** Training Stage B on a cross-fitted residual under a chemotype
  hold-out makes the model **worse than the monolith it decomposes** and worse than its own Stage A
  alone, because the residual is mostly Stage A's *level* error on unseen chemistry, which Stage B
  learns from in-sample features and mis-applies. Centring on the true within-cell departure recovers
  parity and nothing more.
- **An explicit level head is a worse level estimator than an implicit one.** Reproduced
  independently in gen7 (PADRE), gen8 (absolute-value CNP and law), gen9 and gen10 (level+shape,
  residual-shape, two-branch, DeepSets, attention). **All win the shape and lose the level**, and
  offset dominates macro MAE ~5:1.
- **Descriptor-width artefact.** Adding continuous columns of any kind to a mostly binary
  fingerprint matrix changes how ExtraTrees samples split candidates — worth ±0.02–0.04 on the
  cohort where it was measured, an **order of magnitude larger than the effect being weighed**. A
  free permutation null beat the real descriptor block 5/5 on both endpoints.
- **Shrinkage, not degrees of freedom.** A catastrophic affine update (1.7e12) was blamed on
  parameter count; it was the **penalty**. Four ridge-shrunk coefficients beat one free one.
- **Uncertainty is the wrong signal for acquisition.** Not a calibration failure — an algebraic one.
  The useful quantity is `|r − median(r)|`; uncertainty approximates `|r|`; a *perfect* uncertainty
  oracle **loses to random**.
- **Encoding asymmetry.** Features entering as an A−B difference cannot test an absolute-property
  hypothesis. Half of a pre-registered feature view was **structurally incapable of carrying signal**
  (swap-even features under an antisymmetrised head: stability-selection frequency *exactly* 0.000).
- **Stage-1 substitution.** A residual-correction hypothesis was tested on a Stage 1 that was **not
  the champion**, conflating "the features add nothing" with "the base model was wrong".
- **Early stopping against the training loss**, not a held-out split — an architecture arm never got
  a fair regularisation budget.
- **Shrunk predictions are MAE-optimal and cannot be corrected.** `sd(predicted) 0.85` vs
  `sd(measured) 1.66`; rescaling to match makes MAE **worse** by 0.184.

### 8.3 Harness and reproducibility traps

- **`hash()` is salted per process.** It silently broke cross-run pairing while every table claimed
  pairing. Fix: BLAKE2b, plus a test that re-derives the split **in a subprocess under a different
  `PYTHONHASHSEED`**, plus an **AST scan** that fails the suite on any use.
- **Parallel tree summation is not associative.** ~1e-15 differences, fed into a boosting recursion
  or into a second model's target, flip near-tied splits and compound: **0.1588 log units on all 382
  held-out rows** in gen9; **3.4e-2 on 372 of 382** in gen10. Fix: round every model output that
  becomes another model's target (never the final prediction).
- **In-sample residual targets collapse.** A forest with `min_samples_leaf=2` nearly interpolates,
  so the second stage has nothing to learn (target sd 0.198 vs 0.735 cross-fitted).
- **A dependency install moved the science.** `pip install tabpfn` silently downgraded scikit-learn
  and pandas, shifting every result by ~0.0014 against effects of 0.004–0.02.
- **Defaults that silently narrow a study.** A hard-coded `REGIMES=` default ran **2 of 4
  pre-registered regimes**, disclosed only in a JSON field, leaving four hypotheses unevaluable while
  the decision report looked complete.
- **A scan that finds nothing must be tested against a known positive.** Three gen9 data-audit bugs
  had *the absence of an expected finding as their only symptom* — a duplicate scan keyed too finely
  found 0 known-corrupt pairs (61 after the fix); a level scan reported **every** level as exactly
  0.0.
- **Degenerate policies masquerading as arms.** Seven geometry policies that disagree by construction
  all returned the same number because the input table lacked condition columns and a helper returned
  zeros; two "uncertainty" policies read an all-null column and fell back to `rng.choice`; one
  variance policy returned `pool[0]` on a sorted pool.
- **A selector that cannot fall back to what it nests.** A learned blend chose its weight on the very
  blocks its ranker was fitted on, so the baseline it was built to nest never won.
- **A leak caught by implausibility, not by a test.** A "no-chemistry" baseline pooled same-fold
  ligands and beat the model 0/5 seeds; **the sign of the comparison was the only signal.**
- **Protocol-draw instability.** Two executions of the *identical* protocol differ in the third
  decimal — "read the random numbers to two."

### 8.4 Data-quality findings, including one still unresolved after three explanations

- **DMDPhPDA (resolved).** 70 matched cells / 140 rows are **one DOI entered twice by two curators**;
  the paired differences are **exactly the integers 0–4 decades**, tracking acid concentration; one
  copy is monotone in acid for 14/14 metals, the other for 0/14. Handled as sensitivity cohorts
  (FROZEN / QUARANTINED / CORRECTED), **never as an edit to the frozen cohort** — and the choice
  moves every result by < 0.002.
- **TWE-24 — three competing explanations, none confirmed.** It is the worst ligand in the cohort
  (macro MAE **4.098** against a cohort ~0.99, all of it offset).
  1. **Exponent corruption** (~3–4.8 decades) — the reading in the gen7 and gen8 *main reports*.
  2. **A recorded solvent** — gen8's own case study says *"this verdict was reversed on review"*: the
     upstream export records a **12.5 vol % 1-octanol phase modifier** against neat diluent for its
     siblings, the bundle has no one-hot for that mixture so it lands in "other" **and is marked
     `modifier_class = none`**, and within that campaign modified ligands sit **+2.96 decades** above
     unmodified ones — the same order as the error. *"No corruption need be postulated."*
  3. **An identity/pairing error** — the data-quality review queue (verified directly for this
     report) records four rows: **`extractant_name_structure_conflict`** entries stating that the
     archive's consensus structure for the name "TWE-24" is a **phosphorothioate**
     (`CCCCOP(=S)(CP(=S)(OCCCC)OCCCC)OCCCC` — which is exactly the SMILES gen7 identified as the
     single worst held-out ligand) while the records under that name carry **hydroxy-diglycolamide**
     structures; plus a `component_pairing_ambiguous` row where the name appears **paired with TODGA
     in a two-component record whose name/structure order does not match consensus**.
  > **The main reports still carry explanation 1 and the "verify against its primary document" action
  > item, with no mention of 2 or 3.** If 3 is right, the fix is a re-join, not a library visit.
- **The TWE-24 contagion figure is a surrogate's number.** Removing it moves a neighbour's level
  error from 4.416 to 0.526 **under a condition-blind 1-NN lookup**, which copies a neighbour's mean
  wholesale. The deployed model's actual error on that neighbour is **2.03, not 3.9**, and no model
  was ever retrained with TWE-24 removed — **so the contamination cost to the deployed model is not
  measured.**
- **A 610-row curated review queue nobody has processed** — 307 `same_conditions_different_logD`,
  192 `same_value_conflicting_condition`, 39 `extractant_name_structure_conflict`, 36
  `component_pairing_ambiguous`, 19 `identical_record_different_publication`, each with DOIs, figure
  labels and record ids. This is the material that would settle several open questions.
- **An unmodelled second species.** 557 rows (9.3 %) carry an aqueous complexant parsed from free
  text, with permutation-adjusted η² = 0.299 (p = 0.000) against a DOI control of −0.000 (p = 1.00).
  Not shipped.
- **One SMILES is not one chemical system.** 17 of 190 canonical SMILES (50.3 % of rows) carry more
  than one extractant name; one carries **21 names, 20 of them different chemicals**, because a
  sub-source recorded the aqueous holdback agent in the extractant-name field. A quarantine exists in
  the pair builder and **the level runners never call it**.
- **The replicate noise floor is contaminated and its true value is unknown.** Of 313 repeated cells,
  79 differ in a recoverable physical variable, 208 only in a figure caption, and the **26 with
  nothing recoverable scatter most (0.875)** — the opposite of what a hidden-variable explanation
  predicts.

### 8.5 Reporting practices this project got right (worth copying)

- **Pre-registration with a machine-readable stopping rule**, and formally declaring a **valid
  negative result** when nothing passes.
- **Required negative controls per claim type** — a descriptor family needs a permuted twin of equal
  width and marginals; a data expansion needs a row-count-matched control **and** a target-shuffled
  control; an acquisition claim needs random and heuristic baselines; a k-shot claim needs a no-model
  fit calibrated on the same k points.
- **Controls verified as functions, not assumed** — `test_rows_touched` asserted `== 0`; adapter
  leakage tested by corrupting every unselected target and demanding bit-identical output.
- **Scoring on byte-identical test rows**, hashed per fold, so only the intervention varies.
- **Not cherry-picking:** arms that beat the frozen pipeline with 5/5 seeds and intervals excluding
  zero were **left unfrozen** because they missed a pre-registered threshold.
- **Declaring an example-selection rule before looking at examples.**
- **Reporting the quantity that survives the bias** rather than the best number.
- **Naming the exploratory arms as exploratory**, including the ones that won.
- **Publishing the negative result about your own contribution** (the query-set fragility supplement
  was declared "not optional").

---

## 9. Publication assets, and the guardrails on what may be claimed

`figures/` is the paper-preparation layer: **6 main figures, 10 supplementary, 23 scripts, 33
derived artefacts and 9 markdown documents.** Three of the nine are protocols (a figure plan with a
**pre-declared example-selection rule**, a rendering style contract, an editorial priority
argument); three are review results; three are draft manuscript prose.

### 9.1 The figure set

| figure | subject |
|---|---|
| **Fig 2 — few-shot frontier** | the paper's headline; ranked "a reviewer would refuse to review the paper without it" |
| Fig 3 | amplitude compression and its repair |
| Fig 4 | oracle-anchored error decomposition |
| Fig 1 | methodology / leakage defence |
| Fig 5 | generalisation-vs-distance **plus** the gen6 controlled coverage experiment |
| Fig 6 | acquisition policies (conditional on a six-figure allowance) |

Supplementary S1–S10 cover corpus composition, **query-set fragility (S2 — the programme's headline
negative result about its own contribution, declared "not optional")**, adapters, a 24-arm zero-shot
architecture ablation, shape by axis, seed stability, raw example curves, diagnostics and
data-quality strata, coverage dose-response, and fixed-budget strategies.

### 9.2 The verification harness

`figures/scripts/verify_metrics.py` recomputes **every plotted quantity from the rawest artefact
available** (per-row or per-unit predictions) and compares it with the value the repository already
reports. **Result: 67 of 67 checks PASS, 0 FAIL, max absolute delta 4.54e-5.** Tolerances are
per-group: 5e-5 for report-rounded values, 5e-6 for shape metrics, 5e-9 for the oracle cascade,
1e-9 for acquisition, **1e-12 for the mean-preserving offset identity**.

**Three plotting scripts refuse to draw on a failed self-check** — the error-budget script must
reproduce the stored cascade summary to 5e-9; the generalisation script all fifteen tercile marginal
gains to 1e-9; the acquisition script all twenty policies. Statistics are **imported, not
reimplemented**: the figure code re-exports the repository's own paired chemotype bootstrap and adds
exactly one new routine sharing its block construction, replicate count and RNG stream.

> **Scope limit, stated:** the 67 checks cover Figs 2–6 plus one design identity. **No
> supplementary-figure-specific quantity is in the audit**; S-figures are asserted to render without
> transformation, not audited.

### 9.3 The guardrails — what a reader must not do with these numbers

These are the ten findings in `METRIC_AUDIT.md` that constrain claims. **They are the most
practically important part of the figure work.**

1. **Best-of-eleven envelope.** The published `gen8_frontier` / `gen9_frontier` columns are the
   **minimum over every deployable arm at each k**, and the k=2 winner differs between generations
   (a policy gen10 never reran). Plotting them against gen10's fixed rule compares an envelope with
   a pre-specified recipe. Fig 2 therefore uses **four pre-specified rules from one table**.
2. **Marginal-gain provenance.** The stored `marginal_gains.csv` sits on **gen9's** adapter chain
   (its `second_point` 0.0778 chains 0.6539 → 0.5761 = the gen9 adapter), not the frozen `SERIES_ML`.
   Fig 2B recomputes the frozen chain: **0.382 / 0.095 / 0.066 / 0.026**.
3. **Two span-recovery definitions.** Guarded (0.046445 baseline) and unguarded (0.050718). **The
   famous 0.051 / 0.210 / 0.423 ladder in the README and the gen9 report is the unguarded one.**
4. **MEDOID vs RANDOM is two different experiments.** gen8: **+0.101** (99 extractants, gen8's
   global model). gen10: **+0.0631, BCa [+0.0280, +0.1050]**, 98/143, 5/5 (143 extractants, frozen
   model, realised regret). Both positive; not the same experiment. The figures quote gen10.
5. **`NULL_metal_cond` is seed-dependent.** 1.0882 (3 seeds) vs 1.0995 (5 seeds) on C-FULL, and
   **1.1149 on C-COMMON**. So "all of ligand chemistry is worth X" is **0.13 on C-FULL, 0.083 on
   C-COMMON**, and a third document rounds it to 0.12. *Quote the cohort with the number.*
6. **105 publications, not 109.** The README's 109 is wrong; the frozen provenance artefact says
   105 (canonical DOIs excluding the archive's self-citation). The upstream join is 5,992/5,992
   either way.
7. **The bit-identical offset identity is a design guarantee, not a measurement** — `offset_mae` is
   byte-equal for the baseline and every recomposed arm (0.82117153046591 on C-FULL, 0.84676720 on
   C-COMMON, all five seeds). It is legitimately quotable as evidence that level and shape separate,
   but it should be described as **checkable by construction**.
8. **The distance effect is underpowered.** Far − near at k=0 is **+0.3548 [−0.0432, +0.6602]** with
   17 chemotypes per tercile, and the interval includes zero at every k. **The monotonicity claim was
   retracted**; the interventional gen6 coverage panel carries that claim instead.
9. **The oracle arms bound less than they appear to.** The k=1 oracle is an oracle for a
   **level-only** calibration, and **by k=5 the deployable rule overtakes it** (0.441 vs 0.453).
   All oracle arms are hatched, green, and labelled "not deployable" **in the artwork**, not just the
   caption.
10. **The +0.163 coverage effect is not "better everywhere"** — see §5.4.

### 9.4 Retracted captions, and a discipline worth copying

- A caption asserting zero-shot error "is monotone in distance" was **demoted to a point estimate**
  when the interval was computed.
- A panel annotation claiming "regression to the mean: over-predicts low D, under-predicts high D"
  was **wrong** — the panel plots residual against *prediction* and shows no such pattern. It now
  reports what it actually shows: a **mean residual of −0.39, systematic under-prediction on
  held-out chemistry** — with the compression statement moved to the panel where
  `sd(predicted) 0.85 vs sd(measured) 1.66` supports it.
- A supplementary metric was **removed** because `gain_per_measurement` divides by measurements
  actually *spent*, so a breadth strategy that could not spend the budget scored best while achieving
  worse macro MAE.
- Truncated bars exaggerating architecture differences were **redrawn as points with intervals**.
- **Figure 3's example-selection rule was declared before any curve was inspected** (rank on the
  frozen model, fixed axis, fixed seed; median case, top-decile case, bottom-decile case, ties by
  id) and the alternatives are written to a CSV.

Accepted, disclosed risks: Fig 4 panel B is on C-FULL while A and C are on C-COMMON (never on one
axis, but a hurried reader could compare 0.520 with 0.507); Fig 6 is on 143 extractants against 99
elsewhere; Fig 5 panels A and B/C use **different learners** deliberately.

### 9.5 The figure-refinement pass (a separate, later audit)

`figure_refinement/` is an additive publication-quality rebuild of the rendered figures from **two**
repositories — this one and the sibling `lanthanidestrain` — with a governing rule: **aesthetic
changes stay aesthetic; genuine implementation defects are documented, not silently corrected.**

It ships a **layout linter** that reads drawn artist geometry and reports three decidable defects
(text–text overlap > 1.5 pt², clipped text escaping its axes, layout overflow > 14 %), plus an
export verifier (PNG ≥ 300 dpi, PDF/PNG aspect agreement, vector-ness). All 15 rebuilt figures lint
clean — and the README is explicit that **the linter is not sufficient**: every figure went through
at least two render–inspect–fix cycles.

**Two genuine defects found in this repository's own diagnostics:**

1. **A directional claim from a symmetric estimator.** The `cross_series_transfer` heat map is
   titled as a directional statement, but its post-calibration MAE is built from
   `|residual_i − residual_j|` — **symmetric by construction** (`max|M − Mᵀ| = 4.44e-16`). Every
   apparent asymmetry is *a different zero-shot baseline subtracted from an identical
   post-calibration MAE*. **The "calibration is series-local" reading survives; the directional
   reading does not.** Secondary: 30 of 49 cells have n ≤ 5, and the diverging colour limit is set
   by a single n = 3 cell.
2. **Seed double-counting in the error-budget figure.** Two run tables are concatenated;
   `NULL_metal_cond` is the only model in both, giving **8 rows with 3 duplicated (model, seed)
   pairs** and a plotted mean of **1.0953 that matches neither documented cohort**. The headline
   arrow therefore compares a 3-seed mean with an 8-seed mean containing duplicates, and is neither
   paired nor single-cohort.

Also recorded but not acted on: a block-ranking diagnostic picks winners by `idxmin` over ~15 recipes
**with no selection penalty**, on per-fold standard deviations (0.13–0.23) that **exceed the
differences being ranked**; three other diagnostics carry no uncertainty at all.

**In the sibling repository the audit found much more** — 34 of 37 figures rated high severity, led
by misleading encoding (36 occurrences), missing uncertainty (26) and unreadable-when-scaled (24).
Six named defects, of which two are worth carrying as general lessons:

- **A committed PNG and PDF asserting a withdrawn result.** All six rows of the underlying CSV are
  `valid = False`; the plotted value is the row whose recorded `withdrawn_reason` says it "measures
  condition variation"; the repository README agrees the quantity is not identifiable; and the
  current code **already refuses to redraw it**. A stale raster asserting a withdrawn number sits
  beside siblings that regenerate cleanly, with nothing marking it superseded.
- **A headline comparison against a baseline the project itself superseded** — the reference bar is
  an untuned configuration at +0.1422 while the same repository records a *tuned* single model at
  **+0.2784 on the same population, above the whole stack the figure headlines**. So the displayed
  gap is mostly loss function and hyperparameters, not the information the figure credits. (The
  audit records the counter-caveat too: the +0.2784 was selected on a screening split and carries
  optimism.)

And a reproducibility measurement worth having: **19 of 37 sibling-repository figures regenerate
from what is committed; 18 do not** — one uncommitted results CSV blocks four, an absent artefacts
directory blocks five, a `dropna` placed *before* the caller's own emptiness guard turns three
intended named skips into tracebacks, and five figures **have no producer among the 163 committed
Python files** although their rendered outputs are committed and one is cited in a report.

> **The pass's own blind spot, which it did not flag:** **the same defect now exists in *this*
> repository.** The figure path resolver points at the agent worktree, where the gen10 and gen11 run
> directories are **untracked**; 13 figure scripts reference them; the main checkout has no gen10
> directory at all; `figures/` and `figure_refinement/` are themselves untracked; and one parquet a
> figure script reads is gitignored. **The entire publication figure set depends on artefacts that
> exist only on this machine** — exactly the class of defect the pass criticises in the sibling
> repository. See §10.

---

## 10. Engineering state, reproducibility machinery, and repository risk

### 10.1 Inventory

| area | size |
|---|---|
| `src/lanthanide_separation` | **63 modules, 29,948 lines** — 21 top-level plus `gen6/` (8), `gen7/` (11), `gen8/` (14), `gen9/` (9); `gen10/` and `gen11/` live only in the worktree |
| `scripts/` | **79 Python + 1 shell, 33,774 lines** |
| `tests/` | **40 modules, 705 test functions, 14,329 lines** — 19 modules contain leakage assertions, 12 assert SHA-256 identities, 65 functions are determinism/reproducibility tests |
| `slurm/` | 11 job files + 7 submit wrappers, **all `DRY_RUN=1` by default** |

Packaging pins matter here: `requirements.txt` is loose while `requirements-gen3.txt` **pins the
exact frozen scientific environment** (numpy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, torch 2.13.0,
catboost 1.2.10, Python 3.11.11 / Linux x86_64) — a direct response to the incident where a casual
`pip install` moved scikit-learn underneath a completed sweep (§5.5).

### 10.2 The reproducibility contract — the best-engineered part of this project

- **One fail-closed manifest per run**, with 12 required keys (dataset path and SHA-256, source-table
  SHA-256, feature-registry SHA-256, code SHA-256, split definition, chemistry-cluster definition,
  provenance state, model seed, split seeds, preprocessing, folds) and 9 required per-fold keys
  including **`test_row_ids_sha256`**. The design note is explicit that gen2–gen5 each recorded a
  *different* provenance subset, and that the gen5 run which silently dropped two of four regimes
  disclosed nothing — this manifest exists because of that.
- **Frames are hashed through a canonical form** (sorted columns, stable row order, floats at 12
  significant digits) so a parquet round trip does not change the hash; duplicate column labels are a
  hard failure; file bytes are hashed separately for transport integrity.
- **The realised partition is stored, not just the seed.**
- `validate_run` re-reads what was written; `write_success` emits `_SUCCESS.json` **only** on a
  passing validation, else `_FAILED.json`.
- **Run completion contract:** a directory is complete only when `summary.json` **and**
  `_SUCCESS.json` exist; `_INCOMPLETE` is a hard failure marker; `run_config.json.status` stays
  `incomplete` for the life of a run and **must never be used as a completion signal**. SLURM workers
  write to `attempts/` and atomically publish only hash-validated directories.
- **Aggregators refuse to pool runs whose cohort hash differs** (`cohort_sha256` appears in 19 places
  across source and scripts).
- **AST scans** enforce that no split uses Python's salted `hash()`, with subprocess tests under
  several `PYTHONHASHSEED` values.
- **SLURM wrappers default to a dry run** that prints the plan and creates no directories, seeds,
  locks or jobs; every scientific switch is an environment variable with a frozen default, so an
  unset environment submits the frozen protocol.

### 10.3 Known-stale statements inside the repository

| document | says | correct value |
|---|---|---|
| `README.md` | 109 publications | **105** |
| `README.md` | 0.57 % / 1.49 % provenance contamination | **0.10 % / 1.02 %** |
| `README.md` | "287 of 313 replicated cells have a hidden axis" | **retracted** — 79 physical / 208 caption-only / 26 nothing-recoverable, and the 26 scatter *most* |
| `README.md` | acquisition effect +0.101 | gen10's realised study gives **+0.0631**; both are real, different experiments |
| `docs/metrics_reproduction_20260818.md` | 15 lanthanides | **14** |
| gen9 `decision_report.md` | 163 artefacts; 112 tests; "seven issues" | **154**; **125**; the table lists **eight** |
| gen9 `decision_report.md` §4.2 | "common cohort" | values are from the **maximal-coverage** table |
| gen8 main report | TWE-24 is an exponent corruption | **demoted** in the case study (see §8.4) |
| gen8 main report | max-ensemble-sd is an uncertainty result | it was **literally the random policy** |
| gen11 `GEN11_DECISION_REPORT.md` | three arms "not on disk" | they were fitted afterwards — **the report is stale** |

There are also small internal inconsistencies that never got reconciled: 74 vs 75 ECFP clusters,
230 vs 231 series and 1,932 vs 1,939 conditions for BASE91 in different gen5 documents; 26 % vs 27 %
publication straddling in two figure documents; 270 / 269 / 233 gen10 artefacts in three places.

### 10.4 Repository risk — the most urgent practical issue in this handoff

**The two most recent generations, the deployed model, and the entire publication figure set exist
only as untracked files on one machine.**

- `gen10` and `gen11` source (`src/lanthanide_separation/gen10/`, `gen11/`), their scripts, their
  tests and their run outputs live **only** in an agent-managed git worktree
  (`.claude/worktrees/lanthanide-separation-finalize-81154b`). `git status` there shows **103
  untracked/modified paths**, and **nothing has been committed on that branch past the gen6-era
  commit**. If the worktree is pruned, both final generations vanish.
- The gen10 run's own **provenance stamp is unusable**: `manifest.json` records
  `git.commit = fc39fac…, dirty: true` — a gen6-era commit containing **no gen7/gen8/gen9/gen10
  source at all**. The reproducibility receipt points at a tree that cannot rebuild the run.
- **The frozen cohort that every number from gen6 onward is defined against is in no version control
  at all.** Fingerprint `bed178ec1a7a82b0` is asserted by gen7, gen8, gen9, gen10 and gen11, but
  `runs/gen7_architecture/cache/cohort.parquet` is **gitignored in both checkouts**. Nobody can
  verify the fingerprint without re-running the builder against gitignored source data.
- `figures/` (42 MB) and `figure_refinement/` (16 MB) are **untracked** in the main checkout, and
  `figures/scripts/_paths.py` has the worktree path **compiled into the resolver** — so the figure
  scripts fail on any clone. The two-branch split is not an inconvenience; it is encoded in the
  plotting code.
- The auxiliary dataset gen11 depends on resolves through a **symlink into a gitignored directory**,
  so the newest result cannot be reproduced from either checkout alone.
- The sibling `lanthanidestrain` repository **is not on this machine**; the refinement pass vendored
  115 result files with a pinned commit so its figures run, but every fix it requests needs a clone.
- **There is no manuscript.** No `.tex`, no file matching `*paper*` or `*manuscript*` exists
  anywhere. The only paper prose is two untracked markdown files in `figures/`. A handoff that says
  "the figure set is built" implies a manuscript that does not exist.
- There is no `CLAUDE.md` or `AGENTS.md`, and `docs/` stops at gen6 — gen7/gen8 reports sit at the
  repository root, and gen9/gen10/gen11 exist only under `runs/`.

**Recommended first actions for whoever inherits this**, in order:

1. Commit the worktree branch (gen7–gen11 source, tests, scripts, run reports) and merge or rebase
   it onto the main branch.
2. Decide a storage policy for the frozen cohort parquet and the run artefacts, then remove the
   worktree path from `figures/scripts/_paths.py`.
3. Reconcile the stale numbers in §10.3 — most cheaply by making `README.md` cite the frozen
   provenance artefact rather than restating it.
4. Re-render `GEN11_DECISION_REPORT.md` against the arms that now exist on disk (§5.9).

---

## 11. Open questions, ranked

**Highest value, and the programme's own top recommendation:**

1. **Measure extractant-concentration titrations on chemotypes that currently have none.** Only
   **8 Tanimoto-0.7 chemotypes** carry an extractant titration, which is simultaneously (a) the
   population every shape result rests on, (b) the reason the chemotype-blocked interval is the
   narrowest available, and (c) the stated condition for revisiting the model. Estimated cost:
   **four to six titrations, five points each, ~30 measurements**, outside the diglycolamide family.
2. **Finish gen11's size-matched controls (arms F and G).** Until actinides size-matched to the
   non-actinide arm are compared with a random size-matched draw, "actinide chemistry transfers" is
   not separated from "more rows", and the C/E intervals touch zero. Also: re-render the stale
   decision report, and aggregate the 15 LOMO arms that were fitted but never analysed — at more
   than one seed, since the random-removal band is ±0.03.
3. **Process the 610-row data-quality review queue.** It contains the material to settle TWE-24
   (which of three explanations), the exact-decade duplicates, the 271 name/structure mismatch rows
   (which carry a *significantly larger irreducible residual*, so the missing species changes shape,
   not only level), and the replicate noise floor.

**Structural questions the programme left open:**

4. **Can the acquisition oracle gap be closed?** The oracle's rule is known *exactly*
   (`argmin |r_i − median(r)|`), so predicting that deviation from pre-measurement features alone is
   a well-posed supervised problem with a known target. Two formulations have failed (surrogate label
   and realised regret), and both collapsed to plain centrality — but the gap is still ~0.17–0.26.
5. **Is the query-set fragility removable?** No gen10 architecture removed it; the alternatives trade
   one sensitivity for another. Either find a design-relative coordinate system that is stable, or
   make the design-dependence an explicit, documented part of the interface.
6. **The pre-specified A0–A6 3D-vs-2D ablation has never been run** under its own protocol, with
   `ecfp-exact-cluster` grouping, the symmetric arms and the required trivial baselines. The
   repository's 3D verdict currently rests on the gen5 *level* study, not the pair study designed for
   it. (Five studies now agree 3D does not help; this is about protocol completeness, not expected
   outcome.)
7. **The simplicial encoder ladder, the geometry-null control and the conditions-only arm are
   implemented but never executed** — so it is unknown whether that architecture's failure is a
   simplicial-order failure, a representation failure, or the descriptor-width artefact.
8. **Does anything change under the grouping the repository's own audit says is correct?** Every
   gen2–gen4 number sits on folds where 50–55 % of rows have a bit-identical fingerprint across the
   boundary. The stricter grouping exists in the code and was never made the default.
9. **What is the true replicate noise floor?** The published floor is contaminated in a direction
   that rules out the obvious explanation. Recovering solvent at full resolution, phase-modifier
   concentration, oxidation state and shaking time is described as cheap and was never reported.
10. **Does the depth curve ever saturate?** gen6's B3 failed, so "diversity beats depth" bounds
    nothing from above.
11. **Would the missing `scale_diag.py` reproduce gen4's most-quoted negative result?** The
    "per-ligand scale is unpredictable from 2D descriptors" diagnostic — which justified stopping
    descriptor work — is **not reproducible from committed artefacts**.
12. **Would a min_rows ≥ 1 cohort change anything?** gen5 identified the eligibility filter as the
    real ceiling and gen6 relaxed it from 10 to 3; nobody tested 1, or the 100 extractants with a
    single metal measured.

---

## Appendix A — numbers that must never travel without their qualifier

| number | what it actually is |
|---|---|
| **0.9695** | macro MAE, **C-FULL**, one vote per **ECFP cluster** (131 units) |
| **1.0358** | the *same frozen model*, **C-COMMON**, one vote per **extractant** (99 units) |
| **"chemistry is worth 0.09"** | 3-seed C-FULL. It is **0.13** at 5 seeds on C-FULL and **0.083** on C-COMMON |
| **1.0882 / 1.0995 / 1.1149** | the no-ligand null at 3 seeds / 5 seeds / on C-COMMON |
| **frontier 1.0605 / 1.0358** rows | published columns are **best-of-eleven-policy envelopes**; the fixed-rule restatement is 1.0605/0.6752/0.5909/0.5230/0.4772 vs 1.0358/0.6539/0.5761/0.5127/0.4714 |
| **0.051 / 0.210 / 0.423** | the **unguarded** span ladder (guarded baseline is 0.046445) |
| **+0.101** (MEDOID vs RANDOM) | gen8, 99 extractants, gen8's model. gen10's realised study gives **+0.0631 [+0.0280, +0.1050]** on 143 extractants. Different experiments |
| **0.382 / 0.095 / 0.066 / 0.026** | marginal gains on the **frozen** adapter chain; the stored CSV is on gen9's chain |
| **+0.163** (coverage) | 57 of 131 scoring clusters are made entirely of added ligands and carry it; on already-coverable chemistry it is **+0.019**; row-weighted **+0.068** |
| **+0.196** (extractant shape gain) | quote the **publication-blocked** [0.140, 0.257], not the chemotype-blocked [0.118, 0.203] — only 8 chemotypes carry extractant titrations |
| **+0.35** (far − near distance effect) | interval **[−0.043, +0.660]** — includes zero at every k; the monotonicity claim was retracted |
| **+0.0491 / +0.0473** (gen11 C/E) | vs the **design-matched control**, not the frozen anchor; BCa **touches zero**; level-only; absent on far chemotypes |
| **0.82117153046591** | a **design identity**, not a measurement |
| **105 publications** | not the README's 109 |
| **14 lanthanides** | not the 15 in one document (no Pm) |
| **any pooled R²** | name the regime: 0.31 unseen-extractant vs 0.74 unseen-conditions, *same model* |
| **any effect < 0.01 macro MAE** | indistinguishable from a change of machine |

---

## Appendix B — where things live

| path | contents |
|---|---|
| `dataset with 3D structures/` | the frozen 5,992-row bundle, side tables, 1,155 QC'd geometries, the VR asset |
| `dataset_all_metals/` | the separate 16,770-record multi-metal archive (gen11's source) |
| `README.md` (Russian) | project overview — **stale on four specific numbers (§10.3)** |
| `LEAKAGE_AUDIT.md`, `FEATURE_AUDIT.md` | the 17-item audit and the feature-contract audit, written before any result |
| `docs/` | protocols and results, gen2 → gen6 only |
| `gen7_architecture_results_*.md`, `gen8_architecture_results_*.md` | the two large narrative reports, at the repository root |
| `runs/gen9_shape/` | gen9 in full — **no root-level narrative report exists** |
| `runs/sae_dataset_audit/` | the multi-metal build audit **and the 610-row review queue** |
| `figures/` | 6 main + 10 supplementary figures, `verify_metrics.py` (67/67 PASS), **`METRIC_AUDIT.md` — read before quoting anything** |
| `figure_refinement/` | the later publication-quality rebuild of both repositories' figures, with two `SCIENTIFIC_DEFECTS.md` files |
| `.claude/worktrees/lanthanide-separation-finalize-81154b/` | **gen10 and gen11 in full — source, tests, runs, model card — all untracked** |
| sibling `lanthanide_dataset_builder` | the 1:1 provenance join source (DOIs) |
| sibling `lanthanidestrain` | the related pair/`log SF` project — **not on this machine** |

---

*Prepared 2026-09-03. Every number in this report was extracted from the repository's own artefacts
and, where the source was a narrative document, checked against the underlying run outputs.
Where a repository document and this report disagree, the disagreement is stated explicitly and the
stale source named.*
