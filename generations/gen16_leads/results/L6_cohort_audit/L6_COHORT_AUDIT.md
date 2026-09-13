# L6 — cohort audit: are chemotypes being thrown away?

**Regime for every number below: the frozen bundle `dataset with 3D structures/dataset.parquet`
(sha256 `fefbef…f5dd`, 5 992 rows, 190 distinct `canonical_smiles`), gen13's
`gen13sep.cohort.build_cohort`, and the frozen gen6 chemistry map.  No model is fitted here and
no hold-out design is involved, so no design/seed qualifier applies.**  `DISCOVERY` phase; the
frozen cohort was read but never written.

---

## Verdict

**No.  Gen13's cohort construction throws away nothing.**

Every one of the 90 bundle extractants that has ≥ 2 distinct lanthanides measured anywhere, under
any conditions, is in the frozen cohort.  The 100 excluded extractants are excluded by exactly one
fact, and it is not a filter: **only one lanthanide was ever measured on them**, so no separation
factor exists for them at any level of relaxation.

**No single relaxation adds any new chemotype — the maximum over all seven relaxations is +0.**
The brief's bar was "≥ 5 independent chemotypes".  The result is 0, and it is 0 for a structural
reason rather than a marginal one: a relaxation can only merge or split cells among the 90
extractants that already have a measurable pair.

The gen5 failure mode (`min_rows` discarding 37 of 47 chemically novel extractants) **does not
recur**.  Gen13's docstring claim "No minimum-cell filter" is accurate, and its guardrail "100 of
190 bundle extractants have a single metal and never entered" is exactly right.

**But the audit found the real bottleneck, one level up.**  Those 100 single-lanthanide compounds
carry **53 chemotypes that the cohort does not contain at all**, and 77 of them sit at ECFP4
Tanimoto < 0.7 from every kept extractant.  They are not lost to a filter; they are lost to the
laboratory, because each was measured against Eu (or Pr, or Nd) alone.  Measuring **one second
lanthanide on 53 of them** would take the Kish effective sample size from **11.67 to 27.38 (2.35×)** —
larger than any modelling change in this brief.  That is an L4 deliverable, not an L6 one, and it
is handed over in §6.

---

## 0. Reproduction before trust

`gen16/cohort_audit.py` re-implements `build_cohort` step for step with one switch per rule.  With
every switch at its frozen default it reproduces all three of gen13's documented key modes
**exactly, fingerprints included**:

| key mode | cells (mine / frozen) | extractants (mine / frozen) | fingerprint (mine) | frozen fingerprint | match |
|---|---|---|---|---|---|
| `exact` (primary) | 521 / 521 | 90 / 90 | `4c3c6628ea0be949` | `4c3c6628ea0be949` | ✅ |
| `relaxed` | 509 / 509 | 90 / 90 | `97fb945371e30ac5` | `97fb945371e30ac5` | ✅ |
| `series` | 597 / 597 | 89 / 89 | `f0270097d692aaf3` | `f0270097d692aaf3` | ✅ |

The script aborts if any of the three misses.

**Kish, reproduced first as instructed.**  Gen13's published **11.7** is
`gen13sep.metrics.effective_sample_size` over the number of **distinct extractants per chemotype**
(`gen13_separation/scripts/g13_analysis.py:88–93`), not over cell counts:

| statistic | value |
|---|---|
| Kish n_eff over chemotypes, **by distinct extractants** — *gen13's 11.7* | **11.6715** |
| Kish n_eff over chemotypes, by **cell** counts | 1.9134 |

The cell-count version is 1.91 because the diglycolamide chemotype holds 375 of 521 cells.  Both are
reported in every table below; the extractant version is the one that is comparable to gen13.

**Recipe check.**  My own single-linkage ECFP4 (Morgan r=2, 2048 bit) Tanimoto-0.7 clustering of all
190 bundle structures reproduces the frozen `chem__supercluster` partition **exactly** — 98 clusters,
identical membership.  The frozen chemotype map is what the brief says it is.

---

## 1. Exclusion trace: extractant × rule

All 190 bundle extractants traced through `build_cohort`'s own rule order.

| rule | extractants excluded | new chemotypes it withholds | relaxation arm | cells if relaxed | extractants if relaxed | chemotypes if relaxed | Kish if relaxed |
|---|---|---|---|---|---|---|---|
| `single_metal_in_bundle` | **100** | 53 | **not relaxable** | — | — | — | — |
| `todga_name_quarantine` | 0 | 0 | `relax_todga_quarantine` | 521 | 90 | 45 | 11.6715 |
| `sentinel_log_D_le_-6` | 0 | 0 | `relax_sentinel` | 521 | 90 | 45 | 11.6715 |
| `missing_publication_id` | 0 | 0 | not relaxable (0 rows lack one) | — | — | — | — |
| `publication_in_key_splits_metals` | 0 | 0 | `relax_publication_in_key` | 521 | 90 | 45 | 11.6715 |
| `exact_condition_key_splits_metals` | 0 | 0 | `key_mode_none` | 131 | 90 | 45 | 11.6715 |
| `missing_chemotype_in_gen6_map` | 0 | 0 | `relax_require_chemotype` | 521 | 90 | 45 | 11.6715 |
| *(kept)* | 90 | — | — | 521 | 90 | 45 | 11.6715 |

**Extractants with ≥ 2 lanthanides measured anywhere but NOT in the frozen cohort: 0.**
The extractant×rule table is therefore degenerate, and that is the finding.  Per-extractant detail
for all 190 (including, for each, how many cells it would have under each relaxed key) is in
`excluded_extractants.csv`.

Why each rule is empty, checked individually rather than assumed:

- **TODGA quarantine (129 rows).**  Those rows carry TODGA's `canonical_smiles`, so they belong to
  an extractant that is in the cohort anyway; quarantining them removes rows, never a structure.
  See the defect in §5 — they are 20 *other* compounds mis-assigned TODGA's structure, and 19 of the
  20 have only Eu measured, so even a perfect structure fix would give only 1 more candidate, and
  that one (`TODGA,DHOA`) is a TODGA + DHOA synergistic *mixture*, not a new chemotype.
- **`log D ≤ −6` sentinel (3 rows).**  All three are Pr rows on `DMDDdDGA`, `DEDDdDGA`, `DBDDdDGA`.
  Each of those three extractants has **Pr as its only measured lanthanide** with or without the
  sentinel, so the rule costs nothing.
- **`require_publication`.**  0 of 5 992 bundle rows lack a gen6 `publication_id`; the rule never
  fires.
- **`publication_id` in the cell key.**  Relaxing it merges the 7 publication-spanning cells gen13
  documented (rows in cells 3 871 → 3 878, all-14 cells 78 → 79, cell count unchanged at 521), and
  gains **no** extractant.
- **Exact 64-column condition key.**  Even the maximal relaxation — no condition columns in the key
  at all, chemically wrong but a hard bound — gains **no** extractant.  It *loses* cells (521 → 131)
  because it merges conditions.
- **Missing chemotype in the frozen gen6 map.**  The map covers all 190 bundle structures; the rule
  can never fire.
- **`MIN_METALS_PER_CELL = 2`** is not relaxable in principle: `log SF(A/B)` needs two metals in one
  cell.

---

## 2. What the excluded compounds are

100 compounds, 698 bundle rows, measured on **Eu (567 rows), Pr (96), Nd (35)** — one metal each.
91 of the 100 come from a single publication.

| | |
|---|---|
| excluded compounds | 100 |
| chemotypes they represent (frozen gen6 map) | 60 |
| **of those, chemotypes absent from the kept cohort** | **53** |
| compounds joining an existing cohort chemotype | 40 |
| max ECFP4 Tanimoto to the kept cohort < 0.7 (would seed a new chemotype) | **77** |
| max ECFP4 Tanimoto ≥ 0.7 (joins an existing one) | 23 |
| max-Tanimoto distribution | median 0.576, min 0.140, max 1.000 |
| unmapped chemotypes | 0 — all 100 are in the frozen map |

Chemical families of the 100: diglycolamide 40, N-heterocyclic polydentate 25, amide-other 18,
sulfur-donor 8, phosphoryl 6, other 3.

The frozen-map count (60 chemotypes, 53 absent) and the Tanimoto-0.7 count (77 below threshold)
disagree on 17 of 100 compounds.  That is expected and not an error: the frozen map is
**single-linkage**, so a compound can be pulled into a kept chemotype through an intermediate
excluded compound without itself being within 0.7 of any kept structure.  The frozen-map number is
the one to quote for "independent chemotypes"; 77 is the count of compounds individually distant
from everything kept.

The ten most structurally novel excluded compounds (lowest max Tanimoto to the cohort):

| compound | chemotype | family | bundle rows | max T to kept |
|---|---|---|---|---|
| 3-5-diMe-N-DPPz | sc080 | N-heterocyclic polydentate | 6 | 0.140 |
| 3-Me-N-DPPz | sc082 | N-heterocyclic polydentate | 1 | 0.146 |
| (ClPh)2PSSH | sc094 | sulfur donor | 1 | 0.156 |
| DPhen-PyranDGA | sc004 | diglycolamide | 3 | 0.174 |
| N-DPP | sc095 | N-heterocyclic polydentate | 2 | 0.175 |
| TBAPYR | sc005 | N-heterocyclic polydentate | 1 | 0.176 |
| 3-Me-N-DPP | sc081 | N-heterocyclic polydentate | 7 | 0.200 |
| TWE-28 | sc074 | sulfur donor | 6 | 0.207 |
| 3-5-diMe-N-DPP | sc079 | N-heterocyclic polydentate | 6 | 0.213 |
| TWE-23 | sc067 | sulfur donor | 6 | 0.233 |

Full table with nearest kept neighbour for all 100: `nn_tanimoto.csv`.

---

## 3. Relaxations, one rule at a time

Never all at once.  `FROZEN` is the reference row; `key_mode_relaxed` and `key_mode_series` are
gen13's own documented sensitivities and match its manifests exactly.

| relaxation | cells | extractants | chemotypes | ECFP clusters | publications | rows in cells | all-14 cells | cells mixing pubs | **Kish (by extractant)** | Kish (by cell) | Δ extractants | Δ chemotypes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `FROZEN` | 521 | 90 | 45 | 77 | 58 | 3 871 | 78 | 0 | **11.6715** | 1.9134 | — | — |
| `relax_todga_quarantine` | 521 | 90 | 45 | 77 | 58 | 3 871 | 78 | 0 | 11.6715 | 1.9134 | +0 | **+0** |
| `relax_sentinel` | 521 | 90 | 45 | 77 | 58 | 3 871 | 78 | 0 | 11.6715 | 1.9134 | +0 | **+0** |
| `relax_publication_in_key` | 521 | 90 | 45 | 77 | 59 | 3 878 | 79 | 7 | 11.6715 | 1.9134 | +0 | **+0** |
| `key_mode_relaxed` | 509 | 90 | 45 | 77 | 61 | 4 026 | 113 | 0 | 11.6715 | 1.8003 | +0 | **+0** |
| `key_mode_series` | 597 | **89** | 45 | 77 | 58 | 3 845 | 78 | 0 | 11.5974 | 1.7416 | **−1** | **+0** |
| `key_mode_none` (maximal) | **131** | 90 | 45 | 77 | 63 | 4 673 | 63 | 0 | 11.6715 | 4.6747 | +0 | **+0** |
| `relax_require_chemotype` | 521 | 90 | 45 | 77 | 58 | 3 871 | 78 | 0 | 11.6715 | 1.9134 | +0 | **+0** |

Reading:

- **Every relaxation gives 45 chemotypes.**  The maximum new independent chemotypes over all seven
  is **0**, against the brief's bar of 5.
- Only `key_mode_series` changes the extractant count, and it changes it the *wrong way*: it
  **removes** one extractant (the diglycolamide
  `CCCCCCCCCCN(CCCCCCCCCC)C(=O)C(C)OC(C)C(=O)N(CCCCCCCCCC)CCCCCCCCCC`), whose metals are split
  across experiment series so that no series-keyed cell retains two.
- `key_mode_none` is a bound, not a proposal: pooling all conditions merges 521 cells into 131 that
  average genuinely different acid and diluent regimes.  Its Kish-by-cell of 4.67 is an artefact of
  that merging, not extra information.  It is reported only to show that **even destroying the
  condition key entirely recovers no chemistry.**
- An all-at-once relaxation was deliberately **not** run, and is not needed: since no relaxation can
  create a `canonical_smiles` that is not in the bundle, and every bundle structure with ≥ 2
  lanthanides is already kept, the union of all relaxations is also bounded at 90 extractants and 45
  chemotypes.

---

## 4. Verdict against the brief's bar

> *"If relaxing a filter adds even five independent chemotypes, that is worth more than any model
> change in this brief."*

**Maximum new independent chemotypes from any single relaxation: 0.  The bar is not met, and it is
not narrowly missed.**  There is no expanded cohort to run alongside the frozen one, because no
relaxation produces one.  The frozen fingerprint `4c3c6628ea0be949` stands unchanged, and gen13–gen15
comparability is not at risk from this lead.

---

## 5. Defects and caveats found while auditing

Reported because they are real, not because they change the verdict.

1. **20 distinct compound names share TODGA's `canonical_smiles`** (129 rows): `SO3-Ph-BTP`,
   `SO3-Ph-BTBP`, `(PhSO3Na)2-BTBP`, `(PhSO3Na)2-BTPhen`, `PHEN-6OH`, `PHEN-dialcohol`,
   `PyTri-diol`, `PyTri-Tetraol`, `CITAM`, `PPA`, `TWE-26/31/32/37/38/39/40/41/42`, `TODGA,DHOA`.
   Their `LIGAND_SMILES` column is TODGA too, so **the true structures are not recoverable from the
   bundle**.  Gen13's quarantine is correct and necessary — without it these rows would contaminate
   TODGA's own cells.  Cost of the defect: at most one candidate extractant, and that one is a
   mixture.  Fixing it upstream would need the source structures, not a code change.
2. **5 excluded compounds are ECFP4-identical (T = 1.000) to a kept extractant** under Morgan r=2 /
   2048 bits despite different canonical SMILES: `DEDDdDGA`≡`DEDODGA`, `DMDDdDGA`≡`DMDODGA`,
   `DPDDdDGA`≡`DPDODGA`, `D3DODGA`≡`DODDdDGA`, and `CyMe4-BTTP` ≡ its own IUPAC-named duplicate.
   These are alkyl-chain-length homologues that ECFP4 at this radius cannot separate.  This is not a
   leak — the frozen map already places them in the same supercluster, so the chemotype hold-out
   treats them as one unit, which is the conservative choice.  It does mean **ECFP4 similarity
   cannot resolve chain-length homology in this corpus**, which is worth knowing for any
   representation work under L1.
3. **`build_cohort`'s `series` mode is not "exact + series id."**  The expression
   `[c for c in cond if key_mode == "exact" or c not in RELAXABLE_CONDITION_COLUMNS]`
   (`gen13_separation/gen13sep/cohort.py:140`) means `series` uses the **relaxed** column set —
   contact time and metal concentration are dropped — and then appends `experiment_series_id`.
   The docstring calls `series` a sensitivity without saying this.  My first re-implementation read
   it as exact+series and produced 557 cells against the frozen 597; the frozen manifest caught it.
   This is a documentation gap, not a bug: the frozen artefacts are internally consistent.
4. **The `n_metals` filter reported by the trace is on `canonical_smiles`, not on names.**  If two
   chemically different compounds share a SMILES (defect 1), the trace counts them as one
   extractant.  This is the same convention `build_cohort` uses, so the audit is like-for-like.

---

## 6. Handover to L4: the corpus bottleneck is real and it is measurable

L6's null localises the constraint precisely.  The cohort is not filtered down to 11.7 effective
chemotypes — it is *measured* down to 11.7.  Counterfactual Kish, computed with gen13's own
`effective_sample_size` over extractants-per-chemotype, if a second lanthanide were measured on
each compound in a batch:

| scenario | compounds added | extractants | chemotypes | **Kish n_eff** | vs frozen |
|---|---|---|---|---|---|
| **FROZEN** | — | 90 | 45 | **11.671** | 1.00× |
| all 100 single-lanthanide compounds | +100 | 190 | 98 | 12.271 | 1.05× |
| only those with max Tanimoto to kept < 0.7 | +77 | 167 | 98 | 15.607 | 1.34× |
| only those in chemotypes absent from the cohort | +60 | 150 | 98 | 28.553 | 2.45× |
| **one compound per absent chemotype (cheapest batch)** | **+53** | 143 | 98 | **27.375** | **2.35×** |

The ordering is the point.  Adding **all 100** barely moves the Kish (1.05×) because **28 of the 100
are themselves diglycolamides in `sc009`**, the chemotype that already holds 23 of the cohort's 90
extractants — measuring them makes the imbalance worse, and the Kish penalises exactly that.
Adding **53 compounds, one per absent chemotype**, more than doubles it.  The cheapest batch is
therefore *smaller* than the greedy one and 2.2× more effective.

Composition of the 53-compound batch: N-heterocyclic polydentate 18, amide-other 10, diglycolamide
8, sulfur-donor 8, phosphoryl 6, other 3.  The per-compound list with chemotype, family, existing
row count and Tanimoto is in `nn_tanimoto.csv` (filter `chemotype_new == True`).

**Three caveats, stated because L4 must not overclaim from this.**

1. Kish is a **design-capacity** statistic over chemotype balance.  It says nothing about whether BP
   macro MAE would fall.  No performance claim is made or implied here.
2. The counterfactual assumes each added compound yields at least one ≥ 2-metal cell.  Which second
   lanthanide to measure is exactly the question `gen15/fewshot.py`'s D-optimal machinery answers,
   and it is not answered here.
3. These are **new laboratory measurements**, unavailable to any retrospective simulation.  L4's
   retrospective corpus-growth curves can only re-add chemotypes the cohort already has; this table
   is the part of the bottleneck those curves structurally cannot reach.

---

## 7. Files

| file | content |
|---|---|
| `gen16_leads/gen16/cohort_audit.py` | switched re-implementation of `build_cohort`, `verify_frozen()`, exclusion tracer, ECFP4/Tanimoto helpers |
| `gen16_leads/scripts/g16_cohort_audit.py` | runs everything; aborts unless all three frozen fingerprints reproduce |
| `gen16_leads/tests/test_l6_cohort_audit.py` | 15 regression tests, all passing: the three frozen fingerprints, Kish 11.6715, the `series` column-set trap, "no relaxation adds a chemotype", and the single-linkage-0.7 recipe check |
| `results/L6_cohort_audit/excluded_extractants.csv` | all 190 bundle extractants × rule, with cell counts under each relaxed key |
| `results/L6_cohort_audit/exclusions_by_rule.csv` | the rule table of §1 |
| `results/L6_cohort_audit/relaxations.csv` | the relaxation table of §3, with per-arm fingerprints and gained/lost extractants |
| `results/L6_cohort_audit/nn_tanimoto.csv` | the 100 excluded compounds, chemotype, nearest kept neighbour, max Tanimoto |
| `results/L6_cohort_audit/counterfactual_kish.csv` | the §6 table |
| `results/L6_cohort_audit/summary.json`, `run.log` | machine-readable summary and full run transcript |

Nothing under `gen13_separation/`, `gen14_direction/`, `gen15_curve/`, `src/` or
`dataset with 3D structures/` was modified.  The frozen cohort was read, never written.
