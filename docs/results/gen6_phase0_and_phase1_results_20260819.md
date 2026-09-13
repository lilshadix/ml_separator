# gen6 — Phase 0 and Phase 1 results (2026-08-19)

Protocol: [`gen6_diversity_protocol_20260819.md`](../protocols/gen6_diversity_protocol_20260819.md), written and
committed before any gen6 model was fitted. Runs:

| what | run directory | command |
|---|---|---|
| provenance reconstruction | `runs/gen6_provenance/` | `scripts/reconstruct_provenance.py` |
| frozen chemistry map | `runs/gen6_chemistry/` | `scripts/build_chemistry_map.py` |
| Phase 0 package (7 gates, all PASS) | `runs/gen6_phase0/` | `scripts/gen6_phase0.py` |
| Experiment A, 5 seeds | `runs/gen6_expA_5seed/` | `scripts/run_diversity_causal.py` |
| Experiment A, row weighting (sensitivity) | `runs/gen6_expA_5seed_rowweight/` | `… --weighting row` |
| Experiment B, 5 seeds | `runs/gen6_expB_5seed/` | `scripts/run_diversity_learning_curve.py` |

Everything ran locally on the laptop. One 400-tree fit on the 5,248-row cohort is ~1.2 s, so the
whole of Phase 1 is minutes, not cluster hours; `slurm/submit_gen6_diversity.sh` exists for larger
sweeps and is not needed to reproduce anything below.

---

## 0. The headline

**Chemical coverage — not model capacity — is the binding constraint on predicting `log D` for a new
extractant, and the missing information is the ligand's absolute *level*, not the shape of its
response.** On byte-identical test rows, letting the model see 61 sparsely-measured, chemically
distant extractants (7 % more rows, 77 % more ECFP clusters, 98 % more chemotypes) lowers macro MAE
by **0.163 log units** (CI95 [+0.043, +0.330], 5/5 seeds), the gain is **2.8× larger on test ligands
with no close analogue in the old cohort** (+0.463), and it lands almost entirely in the per-ligand
offset (+0.177) rather than the response shape (+0.024). Giving the model the *same* added rows with
their targets permuted recovers none of it.

That is the brief's decision-tree **case A**. All four pre-registered hypotheses pass on both
feature sets.

**Read that with §5.5 attached.** The gain is not spread across the cohort. 57 of the 131 ECFP
clusters that make up the macro vote consist *entirely* of the added ligands, and they carry the
whole effect (+0.367); on the 72 clusters made only of BASE-eligible chemistry the gain is +0.008 for
the champion feature set and **−0.026** for the donor census. So the defensible statement is narrower
than "the model got better": *the expansion repairs the chemistry that was missing and does nothing
for the chemistry that was not.* What keeps that from being circular is a dose-response — the gain
scales with how much closer the added ligands actually brought the training set to each test ligand
(ρ = +0.29, p = 3e-4), so this is chemical transfer, not a class label.

**Experiment B answers the budget question directly.** At an *equal* number of training rows,
spreading the budget over many ligands instead of concentrating it on the best-measured ones is worth
**0.30–0.62 macro MAE at every budget** (CI-clean throughout), and a deliberate max-min diversity
policy beats even random breadth by a further +0.05…+0.13. Round-robin over chemotypes does not beat
random. The depth curve had still not saturated at the largest budget, so the experiment bounds
nothing from above.

Two secondary results are, in their own way, as consequential:

* **Publication identity is recoverable, and it changes how one gen5 regime must be read.** 26.3 %
  of the rows in gen5's `unseen_series` folds sit in a group that spans more than one study — while
  the cells gen5 *averages* are almost clean (0.10 %).
* **The published noise floor is contaminated, though not refuted.** Of the 313 "replicated" cells
  behind it, **79 differ in a physical variable the bundle drops**; 208 differ only in the paper's
  own figure caption, which a genuine replicate reported twice also does. See §2.2 — the honest
  conclusion is that the floor's true value is unknown, not that it is an artefact.

---

## 1. Phase 0 gates

| gate | verdict | evidence |
|---|---|---|
| provenance audited | PASS | `publication_id` **reconstructed** (109 publications, 0 % ambiguous) |
| chemistry partition stable | PASS | frozen all-190 partition = each cohort's own partition, 0 splits / 0 merges |
| identical test rows, no leakage | PASS | 5 seeds × 5 folds; no extractant, ECFP cluster or chemotype shared between a fold's test rows and any arm's training rows |
| cohort reproduces | PASS | 4,881 rows, identical to the published gen5 run |
| split reproduces | PASS | **0 of 97,620** row-fold assignments differ |
| metric layer reproduces | PASS | 80 (regime, seed, arm) cells, max abs diff **4.4e-16** |
| refit within measured tolerance | PASS | mean 0.003, max **0.010** macro MAE (cross-machine) |

### 1.1 The reproduction gate is four claims, not one

The protocol originally demanded bit-equality on a refit. That was amended *before* Phase 1 (§4 of
the protocol records the amendment and the reason): a forest grown on a different CPU/BLAS is a
different forest. Decomposed, three layers are exact and the fourth is a measured tolerance.

The residual has a known, bounded cause. This harness reproduces the local
`gen5_massaction_local_20260818` run **bit-for-bit** on every fold whose training set contains no
all-NaN column; the folds that differ are exactly those affected by the deliberate
`DropAllNaNColumns` insertion in commit `09e04e3`, which removes a constant missing-indicator column
and so shifts the `max_features` draw. Against the *cluster* run the residual survives on folds with
no all-NaN column too, which localises it to the environment.

Two limits on what the fit gate covers, both worth stating: it refits `MC` and `MC_lig2d_ext` only,
because the published four-regime run contains no `*_massaction` arm — so **the feature set the whole
Phase 1 headline uses has never been through the reproduction gate**. And a gate that is skipped is
recorded as `SKIPPED`, never as a pass; the run logs the skipped list separately for that reason.

**Consequence for reading gen5: an effect smaller than about 0.01 macro MAE is not distinguishable
from a change of machine.** The MASSACTION block's +0.028…+0.062 clears that bar; the ECFP family's
+0.048 with a CI spanning zero does not clear it by much.

---

## 2. Provenance — what was actually recoverable

The bundle carries no bibliographic field at all: every `reference` column in
`dataset with 3D structures/provenance/*.csv` is empty and no DOI appears anywhere in the repository.
The best proxy that can be built from bundle-only information reaches ARI ≈ 0.55 against the truth
and splits about half the studies, so nothing is approximated here.

The upstream SAFE exports, however, are reachable
(`lanthanide_dataset_builder/raw_data/*_SAFE.csv`): `safe_exp_id` decomposes as
`{file stem}_SAFE:{exp_id}` and joins **5,992/5,992**, cross-checked on `D` and `metal_symbol`.

| identifier | status |
|---|---|
| `experiment_id` | reconstructed (`safe_exp_id`, row-unique) |
| `publication_id` | **reconstructed** — 105 publications, 0 % ambiguous |
| `experiment_series_id` | reconstructed from (publication, `comments_description`) — but that is free text ("Fig 2 …"), so a series is only as well defined as the paper's own figure labelling |
| `replicate_id` | **surrogate only** — see §2.2 |

### 2.1 How much of the modelling structure crosses a study boundary

| grouping | groups | spanning >1 publication | rows affected |
|---|---|---|---|
| extractant × condition × metal (the cell gen5 averages) | 5,302 | 3 | 6 (**0.10 %**) |
| `levels.series_labels` — the CV group of gen5's `unseen_series` | 344 | 27 | **1,577 (26.32 %)** |
| extractant × condition — the pair-cohort key for `log_SF` | 2,459 | 8 | 61 (1.02 %) |
| extractant — the CV group of `unseen_ligand` | 190 | 28 | 3,710 (61.92 %) |

A first draft of this table reported 16 cells / 0.57 %, 29 series groups / 27.10 % and 21 pair
groups / 1.49 %, from 109 publications. Those were inflated by incomplete DOI normalisation:
`DOI 10.x` and `DOI:10.x` are the same paper, and one mangled citation blob split a single reference
into four. Thirteen of the sixteen "multi-publication" cells were one study written two ways. The
fix (`_DOI_PREFIXES` plus an embedded-DOI regex) gives 105 publications and the numbers above.

Read these three ways. The **averaging** concern that five generations of docs warned about turns out
to be negligible: 0.10 % of rows, 3 cells. The **pair cohort** is nearly clean at 1.02 %. But gen5's
`unseen_series` regime — the one the gen5 write-up called "the honest known-ligand regime" — holds
out a group that in **26 %** of rows spans two studies, so for those rows the same study's other
measurements remain in training. That regime's numbers are optimistic by an unmeasured amount. This
is the one contamination figure that survives the correction essentially intact.

The 61.9 % figure for `unseen_ligand` is not leakage: it says 28 extractants were studied by more
than one group, which is a batch-effect question, not a fold-integrity one.

### 2.2 Is the published noise floor a noise floor?

gen5 quoted an MAE floor of 0.19 (median within-cell sd) / 0.62 (pooled) from 313 replicated cells.
Those cells cannot be plain replicates: the upstream deduplication key **includes `D`**, so anything
with identical conditions *and* identical `D` was collapsed before the bundle existed. Something
distinguishes each surviving pair. The question is *what*, and the answer splits in two:

| upstream column varying inside the cell | cells | share |
|---|---|---|
| `comments_description` — the paper's own figure/table caption | 233 | 74.4 % |
| `ini_comp` | 58 | 18.5 % |
| `Solvent_Name` / `f_Solvent_Name` | 36 | 11.5 % |
| `Metal_Oxidation_state` | 14 | 4.5 % |
| `Phase_Modifier_Concentration_M` | 6 | 1.9 % |
| `Shaking_Time_min` | 2 | 0.6 % |

Counting **physical** variables only — dropping the free-text caption, because a genuine replicate
reported in two figures differs there too — the picture is much weaker than a first pass suggests:

| subset | cells | median log D range | max |
|---|---|---|---|
| a physical variable differs | **79** (25 %) | 0.342 | 5.591 |
| only the caption differs | 208 (66 %) | 0.368 | 4.111 |
| nothing recoverable differs | 26 (8 %) | **0.875** | 1.978 |

Two things follow, and the second is why the strong version of this claim does not survive:

1. For a quarter of the repeated cells there *is* a recoverable physical variable the bundle drops —
   solvent at full resolution, phase-modifier concentration, oxidation state, shaking time. For
   those, part of the "noise" is missing-feature error, and recovering the columns is a cheap
   Phase-2 experiment.
2. But the partition carries almost no information about the spread: cells with a physical axis
   scatter 0.342, caption-only cells 0.368, and the cells with **no** recoverable difference at all
   scatter **0.875** — more than twice as much. If hidden variables drove the spread, that ordering
   would be the other way round.

**The honest conclusion is that the floor is contaminated and its true value is unknown**, not that
it is an artefact. An earlier draft of this document, and of the generated provenance report, said
"287 of 313 cells have a hidden axis, only 26 are true repeats" and attached the whole-set spread
statistics to those 26. Both were wrong: the 287 counts caption differences as experimental
variables, and the 26 cells' own spread is 0.875, not 0.398. The code and the reports were corrected
(`provenance.py` now separates `UPSTREAM_PHYSICAL_CONDITION_COLUMNS` from
`UPSTREAM_DESCRIPTIVE_COLUMNS` and reports the spread per subset).

---

## 3. The chemistry map (frozen, all 190 extractants)

190 extractants → 164 bit-identical ECFP clusters → **98 Tanimoto-0.7 chemotypes**. The largest
chemotype holds 51 extractants; 80 of the 98 hold exactly one; 12 extractants have no neighbour above
Tanimoto 0.4 anywhere in the 190.

Families, from the rdkit-derived motif columns already frozen in `ligand_2d_descriptors.parquet`
(rdkit itself is not installed locally, and the per-row `chem__family_source` records which route was
used): diglycolamide 81, N-heterocyclic polydentate 41, monoamide 28, phosphoryl 11, malonamide 11,
other 11, podand/ether 4, hydroxyl/acid 3.

**The freeze is safe, but the obvious proof of it is vacuous and a first draft of this document
relied on it.** `build_level_dataset` computes `tanimoto_cluster` *before* applying the row filter,
so the labels a cohort carries were already derived from all 190 extractants. Comparing the frozen
map against those is a genuine check only at the bit-identical level, where a fingerprint's identity
cannot depend on which other molecules are present.

Re-running single linkage on the cohort **alone**, as a study that had only ever seen those ligands
would have:

| cohort | level | from-scratch groups | frozen groups | frozen *splits* a local group | frozen merges several |
|---|---|---|---|---|---|
| BASE (91) | bit-identical | 74 | 74 | **0** | 0 |
| BASE (91) | chemotype | 45 | 40 | **0** | 3 |
| EXPANDED (152) | bit-identical | 131 | 131 | **0** | 0 |
| EXPANDED (152) | chemotype | 80 | 79 | **0** | 1 |

The frozen map is **coarser, never finer**: three chemotypes a BASE-only clustering would separate
are bridged by a ligand outside the cohort. Merging makes a held-out chemotype *larger* and the
hold-out *stricter*, which is the safe direction; a frozen group that *split* a local one would put
related chemistry on both sides of a fold, and that count is zero everywhere. It also means gen5's
own `unseen_chemotype` folds were already built with all-190 linkage — conservatively, as it turns
out.

Label *strings* are canonicalised (`sc000…`, ordered by first member) because scipy's cluster
numbering depends on row order and a frozen artifact gets hashed.

### 3.1 What the eligibility rule was costing

| cohort | rows | extractants | ECFP clusters | chemotypes |
|---|---|---|---|---|
| BASE (≥ 10 cells) | 4,881 | 91 | 74 | 40 |
| added at ≥ 3 cells | 367 | 61 | 59 | 46 |
| EXPANDED (≥ 3 cells) | 5,248 | 152 | 131 | 79 |

The 61 entering extractants bring **57 ECFP clusters and 39 chemotypes that BASE never sees**, for
7.0 % more rows. Their nearest neighbour in BASE is at median Tanimoto 0.538 versus 0.773 for BASE
ligands among themselves; 25 of 61 sit below 0.4. Families entering: diglycolamide 13, N-heterocyclic
polydentate 12, malonamide 9, other 8, phosphoryl 8, podand/ether 4, monoamide 4, hydroxyl/acid 3.

---

## 4. Where the error actually is (published gen5 out-of-fold predictions)

Decomposing the *published* gen5 OOF with the new metric layer — no refitting, so this is a
re-reading of the existing result, not a new one:

| regime | arm | macro MAE | offset MAE | shape MAE | shape R² | offset share of SSE |
|---|---|---|---|---|---|---|
| unseen_chemotype | MC | 1.147 | 0.931 | 0.624 | 0.078 | 0.502 |
| unseen_chemotype | MC_lig2d_ext | 1.037 | 0.811 | 0.596 | 0.170 | 0.399 |
| unseen_chemotype | MC_donors | 1.030 | 0.837 | 0.575 | 0.123 | 0.391 |
| unseen_ligand | MC | 0.959 | 0.729 | 0.563 | 0.141 | 0.430 |
| unseen_ligand | MC_lig2d_ext | 0.836 | 0.585 | 0.537 | 0.251 | 0.302 |
| unseen_ligand | MC_donors | 0.875 | 0.664 | 0.519 | 0.249 | 0.340 |
| unseen_series | MC_lig2d_ext | 0.822 | 0.467 | 0.575 | 0.286 | 0.249 |
| unseen_conditions | MC_lig2d_ext | 0.791 | 0.456 | 0.506 | 0.641 | 0.283 |

Two readings, and the second is the useful one:

1. The offset is the larger term in every ligand regime, and it is what the descriptors improve.
   Going from `MC` to `MC_lig2d_ext` on `unseen_ligand` moves the offset by −0.144 and the shape by
   only −0.026. The extended 2D descriptors were already doing level work, not shape work.
2. **Shape error is nearly constant** — 0.46 to 0.62 log units across every regime, arm and novelty
   bin measured here (the 16 rows above plus the 24 novelty-bin rows below; full tables in
   `gen5_offset_shape_*.csv`). No arm in the gen5 run, and no arm in Experiment A, moves it by more
   than ~0.05. Whatever governs the within-ligand response to metal and condition is either already
   captured by metal + conditions, or not representable by any feature block tried so far. This is a
   statement about the arms that have been run, not a proof of impossibility.

By nearest-neighbour Tanimoto (`unseen_ligand`), the descriptors' help with the offset collapses
exactly where it is needed most, and the donor census is the exception:

| bin | ligands | MC offset | +lig2d_ext | +donors |
|---|---|---|---|---|
| < 0.4 | 9 | 1.157 | 1.131 | **0.897** |
| 0.4–0.6 | 16 | 0.988 | 0.771 | 0.776 |
| 0.6–0.8 | 58 | 0.718 | 0.598 | 0.664 |
| ≥ 0.8 | 39 | 0.572 | **0.399** | 0.574 |

This is the mechanism behind gen5's observation that the donor census is the only block whose edge
grows with chemical distance: it is doing *level* work on distant chemistry, where the fingerprint-
driven descriptors have nothing to interpolate from.

---

## 5. Experiment A — BASE91 vs EXPANDED152 on identical test rows

One shared cohort at `min_cells = 3` (5,248 rows / 152 extractants / 131 ECFP clusters / 79
chemotypes). Folds hold out whole chemotypes. Every arm is scored on byte-identical test rows —
checked per fold by hashing the test row-id set — and only the training row mask changes. Learner and
fold-seed formula are gen5's, unchanged. 5 split seeds × 5 folds, 400 trees, bootstrap over 79
chemotype blocks with 5,000 replicates and one shared index matrix.

### 5.1 Arm leaderboard (macro MAE, one ECFP cluster = one vote, mean of 5 seeds)

| feature set | arm | macro MAE | offset MAE | shape MAE | median ligand MAE | worst-quartile ligand MAE |
|---|---|---|---|---|---|---|
| MC_lig2d_ext_massaction | BASE | 1.2100 | 1.0444 | 0.5269 | 1.0564 | 2.2855 |
| MC_lig2d_ext_massaction | **EXPANDED** | **1.0468** | 0.8744 | 0.5069 | 0.9027 | 2.0934 |
| MC_lig2d_ext_massaction | EXPANDED_ROWMATCHED | 1.0486 | 0.8748 | 0.5094 | 0.9156 | 2.0921 |
| MC_lig2d_ext_massaction | EXPANDED_SHUFFLED | 1.2193 | 1.0633 | 0.5392 | 1.0734 | 2.2785 |
| MC_donors | BASE | 1.1870 | 1.0270 | 0.5122 | 0.9804 | 2.2939 |
| MC_donors | **EXPANDED** | **1.0150** | 0.8613 | 0.4991 | 0.9172 | 1.9782 |
| MC_donors | EXPANDED_ROWMATCHED | 1.0115 | 0.8560 | 0.5056 | 0.9144 | 1.9715 |
| MC_donors | EXPANDED_SHUFFLED | 1.2517 | 1.0717 | 0.5716 | 1.1240 | 2.2434 |

Note the shape column: it barely moves (0.527 → 0.507, 0.512 → 0.499). Everything happens in the
offset.

### 5.1a The percentile interval alone is not trustworthy here

An adversarial review found, and I reproduced, that the percentile bootstrap is miscalibrated at
the `all` endpoint. The cause is structural: scoring votes per ECFP cluster (131 of them) while
resampling per chemotype (79), and **one chemotype holds 28 of the 131 scoring units** — 21 % of the
vote inside 1.3 % of the evidence — with a delta of the opposite sign to the population. A
double-bootstrap coverage simulation puts the real one-sided Type-I rate near **12.7 %**, not 2.5 %.
(The hard-chemistry endpoint is fine at ~2.9 %: no block dominates there.)

So the run now reports four numbers side by side, and the report says not to quote the first alone:

| feature set | statistic | point | percentile CI95 | BCa CI95 | cluster-robust CI95 (p) | block-macro |
|---|---|---|---|---|---|---|
| `MC_lig2d_ext_massaction` | macro MAE | +0.1633 | [+0.043, +0.330] | [+0.013, +0.295] | [+0.013, +0.314] (p 0.034) | **+0.290** |
| `MC_lig2d_ext_massaction` | offset MAE | +0.1773 | [+0.058, +0.346] | [+0.029, +0.311] | [+0.028, +0.326] (p 0.020) | +0.296 |
| `MC_donors` | macro MAE | +0.1721 | [+0.058, +0.349] | [+0.053, +0.335] | [+0.028, +0.316] (p 0.020) | +0.273 |
| `MC_donors` | offset MAE | +0.1795 | [+0.060, +0.364] | [+0.053, +0.350] | [+0.030, +0.329] (p 0.019) | +0.281 |

**Every interval excludes zero, on both feature sets, for both statistics.** A1 and A3 survive the
correction. Two further points follow:

* BCa is the tightest lower bound (+0.013 for the champion features) and is the number a sceptic
  should quote.
* `block-macro` votes once per *chemotype* — the unit the folds actually held out, and the
  well-calibrated one. It is **+0.290**, nearly twice the headline. The reported estimate is
  therefore **conservative**: voting per ECFP cluster understates the effect because the giant
  diglycolamide chemotype gets 28 votes for one independent observation.

### 5.2 Pre-registered verdicts

All four hypotheses **PASS** on both feature sets, 5/5 seeds, CI95 excluding zero.

| H | statement | `MC_lig2d_ext_massaction` | `MC_donors` |
|---|---|---|---|
| A1 | EXPANDED beats BASE overall | **+0.1633** [+0.0430, +0.3300] | **+0.1721** [+0.0581, +0.3487] |
| A2 | the gain is largest on the hardest chemistry (nn < 0.4) | **+0.4630** [+0.2527, +0.6903] vs +0.1633 overall | **+0.4611** [+0.1620, +0.8260] vs +0.1721 |
| A3 | the gain is in the level, not the shape | offset **+0.1773** [+0.0580, +0.3455] vs shape +0.0243 [+0.0008, +0.0555] | offset **+0.1795** [+0.0595, +0.3636] vs shape +0.0179 [−0.0017, +0.0427] |
| A4 | it is information, not row count | vs shuffled **+0.1726** [+0.0553, +0.3419]; vs row-matched +0.0019 [−0.0031, +0.0061] | vs shuffled **+0.2368** [+0.1246, +0.3988]; vs row-matched −0.0034 [−0.0096, +0.0011] |

A4 is the one worth dwelling on. `EXPANDED_ROWMATCHED` — all the sparse rows plus dense rows back to
BASE's exact row count — is indistinguishable from EXPANDED (Δ ≈ 0.002), so the effect is not the
extra 7 % of rows.

`EXPANDED_SHUFFLED` — the same rows with the added targets permuted among themselves — does **not**
merely "return to BASE": it lands slightly *worse* than BASE (1.219 vs 1.210 for the champion
features; 1.252 vs 1.187 for the donor census). Feeding a model 367 rows of mislabelled chemistry is
worse than not feeding it those rows at all, which is the expected direction but means the
`EXPANDED_SHUFFLED` contrast is a **harder** comparator than BASE. It inflates A4's apparent margin
relative to the BASE contrast — by 6 % for lig2d (+0.173 vs +0.163) and by **38 %** for the donor
census (+0.237 vs +0.172). The null does its job — it rules out regularisation and the weighting
change as mechanisms — but the number to quote for the size of the effect is the BASE contrast, not
the shuffled one.

#### A3 as pre-registered is a weaker test than it looks

A3 was written as "the offset gain exceeds the shape gain", compared in **absolute** log units. That
comparison is biased before any physics enters: BASE's offset MAE is 1.044 and its shape MAE 0.527,
so any uniform improvement — shrinking every residual towards the truth by a constant fraction, which
has no level-specific content whatsoever — moves the offset roughly twice as far and passes A3. That
is a flaw in the pre-registration, not in the result, and it is recorded rather than quietly patched:
the verdict above is scored exactly as written.

The test A3 *should* have been is the **relative** reduction, which a uniform shrinkage leaves equal:

| feature set | offset MAE | relative reduction | shape MAE | relative reduction | ratio |
|---|---|---|---|---|---|
| `MC_lig2d_ext_massaction` | 1.044 → 0.874 | **16.3 %** | 0.527 → 0.507 | 3.8 % | **4.3×** |
| `MC_donors` | 1.027 → 0.861 | **16.1 %** | 0.512 → 0.499 | 2.6 % | **6.3×** |

A uniform shrinkage scores 1.0×. The measured 4.3× and 6.3× say the improvement really is
level-specific. A3's conclusion survives the stronger test; its pre-registered *form* should be
replaced in the next generation's protocol.

### 5.3 The weighting confound, checked rather than argued

Training weights each ECFP cluster equally, and EXPANDED has more clusters (≈ 110 vs 65 per fold), so
the arm change also reweights the dense chemistry. That is a genuine confound, and the honest answer
is to rerun the whole experiment with row-uniform weights rather than argue about it
(`runs/gen6_expA_5seed_rowweight/`, 5 seeds, same folds, `MC_lig2d_ext_massaction`):

| endpoint | statistic | cluster weighting (primary) | row weighting (sensitivity) |
|---|---|---|---|
| all | macro MAE | +0.1633 [+0.0430, +0.3300] | **+0.1635** [+0.0825, +0.2830] |
| all | offset MAE | +0.1773 [+0.0580, +0.3455] | +0.1688 [+0.0863, +0.2898] |
| all | shape MAE | +0.0243 [+0.0008, +0.0555] | +0.0240 [+0.0027, +0.0528] |
| nn < 0.4 | macro MAE | +0.4630 [+0.2527, +0.6903] | +0.3712 [+0.1826, +0.5754] |
| nn < 0.4 | offset MAE | +0.4677 [+0.2419, +0.7024] | +0.3783 [+0.1773, +0.5933] |

The overall effect is unchanged to three decimals; the hard-chemistry gain shrinks from **2.8×** to
2.3× the overall gain but stays large and CI-clean. **The weighting is not the driver.**

### 5.4 The other alternative explanation: new conditions, not new chemistry

A sparse extractant could help simply by bringing experimental conditions BASE has never seen, in
which case the `COND`/`MASSACTION` blocks would be doing the work and the result would say little
about chemistry. Averaged over the whole cohort this looks settled — the 367 added rows contribute
123 new condition vectors out of 2,055, every one of their five continuous conditions lies inside
BASE's observed range (**0 %** outside on all five), and EXPANDED raises exact-condition coverage of
the test rows by only 2.6 percentage points.

**That average is computed where the effect is not.** Splitting it by the population that carries the
gain (§5.5) tells a different story:

| test rows | exact condition in BASE training | in EXPANDED training | change |
|---|---|---|---|
| BASE-eligible ligands (24,405 row-instances) | 12.6 % | 12.7 % | **+0.1 pp** |
| sparse ligands (1,835) | 12.3 % | 50.7 % | **+38.4 pp** |

So on exactly the rows that improve, the expansion *does* supply the experimental condition as well as
the chemistry — a different ligand measured at the byte-identical `condition_id`. Splitting the
sparse test rows accordingly:

| sparse test rows | n | BASE MAE | EXPANDED MAE | gain |
|---|---|---|---|---|
| exact condition supplied only by EXPANDED | 704 | 1.108 | 0.727 | **+0.380** |
| no shared condition | 1,131 | 1.399 | 1.119 | **+0.281** |

The gain is 35 % larger where the condition is shared, so part of the effect *is* condition transfer.
But a substantial **+0.281 survives on rows where EXPANDED supplies no matching condition at all**,
and that residual cannot be condition coverage. Together with the dose-response in §5.5 (the gain
scales with chemical proximity to what was added), the honest statement is:

> the expansion buys both — the chemistry and, for the sparse ligands, some of their experimental
> context — and the chemistry component alone is worth about +0.28 on the rows that isolate it.

An earlier draft of this section said flatly "the expansion is not buying condition space", which the
38-point figure contradicts. That sentence was wrong.

### 5.5 Where the gain actually lands — the load-bearing qualification

The macro number is a mean over 131 ECFP clusters, and **57 of those clusters consist entirely of the
61 added ligands**. BASE trains on none of them in any fold. Decomposing the reported macro gain by
cluster type (this reconstructs the headline exactly: 57·0.367 + 2·(−0.054) + 72·0.008, over 131):

| cluster type | units | gain, `MC_lig2d_ext_massaction` | gain, `MC_donors` |
|---|---|---|---|
| made only of added (3–9 cell) ligands | 57 | **+0.367** | **+0.427** |
| mixed | 2 | −0.054 | +0.013 |
| made only of BASE-eligible ligands | 72 | **+0.008** | **−0.026** |

Per test ligand, averaged over the five seeds:

| subset | ligands | gain, lig2d | gain, donors | fraction improved (lig2d / donors) |
|---|---|---|---|---|
| BASE-eligible (≥ 10 cells) | 91 | +0.019 | **−0.018** | 52 % / 40 % |
| sparse (3–9 cells) | 61 | +0.356 | +0.422 | 74 % / 74 % |
| nn to BASE training < 0.4 | 33 | +0.511 | — | 82 % |

**So the honest claim is narrower than "the model got better".** On the 91 ligands BASE could
already cover, the expansion is neutral at best and mildly *harmful* for the donor census. The whole
effect is the repair of chemistry that was absent. 60 of 152 ligands still get worse; the median
ligand gains +0.033; the top decile of ligands supplies 80 % of the summed gain.

**With one real exception, which the first draft missed.** Restrict to the hard-chemistry endpoint
and the BASE-eligible clusters *do* gain for the champion feature set: at nn < 0.4, dense-only
clusters improve by **+0.274, with 74 % of them improved** (`MC_donors`: −0.002, 52 % — a coin
flip). So the expansion is not purely self-serving: for a well-covered ligand that nonetheless sits
far from training chemistry in a given fold, added diversity helps it too. It is at the `all`
endpoint, and for the donor census everywhere, that the dense gain vanishes. Every run prints this
split itself — `gain_decomposition.csv` and the "Where the gain sits" section of
`decision_report.md`:

| feature set | endpoint | cluster type | units | gain | fraction improved |
|---|---|---|---|---|---|
| `MC_lig2d_ext_massaction` | all | added chemistry only | 57 | +0.367 | 72 % |
| `MC_lig2d_ext_massaction` | all | BASE-eligible only | 72 | +0.008 | 48 % |
| `MC_lig2d_ext_massaction` | nn < 0.4 | added chemistry only | 27 | +0.570 | 78 % |
| `MC_lig2d_ext_massaction` | nn < 0.4 | BASE-eligible only | 9 | **+0.274** | 74 % |
| `MC_donors` | all | BASE-eligible only | 72 | **−0.026** | — |
| `MC_donors` | nn < 0.4 | BASE-eligible only | 9 | −0.002 | 52 % |

#### Is that circular?

The obvious objection is that "the arm allowed to see this kind of chemistry predicts this kind of
chemistry better" is close to a tautology. Three things separate it from one.

*First*, a test ligand's own chemotype — and therefore its own ECFP cluster — is held out of **every**
arm's training set in every fold. EXPANDED never sees the test ligand or anything within Tanimoto 0.7
of it. Whatever it learns comes from *other* added ligands in other chemotypes.

*Second*, and decisively, there is a **dose-response**. For each test ligand, measure how much closer
the expansion actually brought the training set: `nn_expanded_tanimoto − nn_reference_tanimoto`. The
gain tracks it:

| how much closer EXPANDED training is | ligands | gain, lig2d | gain, donors | mean nn to BASE |
|---|---|---|---|---|
| 0 (no closer at all) | 66 | +0.038 | +0.073 | 0.578 |
| 0–0.05 | 50 | +0.126 | +0.156 | 0.538 |
| 0.05–0.15 | 18 | +0.321 | +0.149 | 0.482 |
| > 0.15 | 18 | **+0.491** | **+0.491** | 0.231 |

Spearman(gain, closeness gained) = **+0.290, p = 2.9e-4** for `MC_lig2d_ext_massaction`
(+0.128, p = 0.11 for `MC_donors` — weaker, and its middle bins are not monotone). A class-level
prior — "ligands unlike diglycolamides extract differently" — would help every added-class ligand
equally regardless of proximity. It does not: ligands the expansion brought no closer gain +0.038,
and those it brought much closer gain +0.491. That is chemical transfer.

*Third*, `EXPANDED_SHUFFLED` shares the row count, the feature distribution and the cluster weighting
and recovers nothing, so none of the three can be the mechanism.

**What remains fair in the objection**: the primary metric gives 43.5 % of its vote to clusters that
BASE structurally cannot serve, so the *magnitude* +0.163 is a property of this cohort's composition
as much as of the effect. The row-weighted pooled contrast — dominated by the 28 %-of-rows
diglycolamide, and for that reason not the primary metric here — is +0.068 (lig2d) and +0.017
(donors). Both framings are in `arm_metrics.csv`; neither is hidden.

### 5.6 Is one fold carrying the result?

63.6 % of the cohort sits in a single Tanimoto chemotype (the diglycolamides), so exactly one fold per
seed is a 3,700–4,000-row giant and the other four are small. If the macro number came from that one
fold, it would be an artefact of the cohort's shape. Per-fold macro MAE gain (BASE − EXPANDED,
`MC_lig2d_ext_massaction`, 25 fold × seed cells):

| folds | n | mean gain | min | max | fraction positive |
|---|---|---|---|---|---|
| the four small folds per seed | 20 | **+0.232** | +0.031 | +0.458 | **20/20** |
| the giant DGA-chemotype fold | 5 | +0.054 | −0.036 | +0.146 | 4/5 |

The effect is positive in **every one of the twenty** folds that hold out ordinary chemistry, and it is
*weakest* exactly where it should be — when the held-out set is the huge diglycolamide chemotype that
BASE already covers well. The single negative cell in the table is one giant fold at −0.036.

---

### 5.7 Does the model know where it is wrong? (the OOD layer)

Every gen6 prediction carries nearest-training-neighbour Tanimoto, neighbour counts above 0.5/0.7/0.8,
chemotype support and the ensemble sd across the forest's trees. On the EXPANDED arm's 26,240
out-of-fold rows:

| signal | Spearman with \|error\| |
|---|---|
| ensemble sd across trees | **+0.183** |
| nearest-training Tanimoto | −0.048 |
| neighbours above 0.7 / chemotype support | constant (zero) — by construction under a chemotype hold-out |

The last row is worth stating rather than hiding: under a held-out-chemotype protocol *no* test
ligand has a training neighbour above 0.7 and *no* chemotype has support, so two of the six OOD
columns carry no information in this regime. They will in `unseen_ligand`.

Error against a combined signal (rank of ensemble sd × rank of distance), reported as a **descriptive
calibration curve, not a tuned policy** — the protocol forbids tuning an abstention threshold on the
final test set, and nothing here does:

| rows abstained | 0 % | 10 % | 20 % | 30 % | 50 % |
|---|---|---|---|---|---|
| macro MAE, ensemble sd alone | 1.047 | 0.961 | 0.910 | 0.865 | 0.855 |
| macro MAE, distance alone | 1.047 | 1.038 | 0.881 | 0.845 | 0.899 |
| macro MAE, sd × distance | 1.047 | **0.874** | 0.834 | 0.838 | 0.820 |

Declining to answer on the 10 % least-supported rows would cut macro MAE by 17 %. That is the shape
of a useful deployment rule, and Phase 3 should fit its threshold on training folds and then test it,
which this table does not do.

---

## 6. Experiment B — diversity versus depth at equal row budget

Experiment A can be answered with "you just gave it more rows". Experiment B removes that: every arm
gets the **same number of training rows** from the same fold's pool, and only the acquisition policy
differs. Policies are label-free (verified at run time by permuting `log_D`: none of the 72 selections
moved) and acquire whole ligands in order, with only the straddling ligand contributing a partial
row set. 4 policies × 6 budgets × 3 acquisition draws × 5 seeds × 5 folds.

### 6.1 What a budget buys, and what it is worth

Macro MAE (one ECFP cluster = one vote), mean over draws, seeds and folds:

| budget (rows) | DEPTH | DIVERSITY | RANDOM | MAXMIN |
|---|---|---|---|---|
| 250 | 1.858 | 1.342 | 1.342 | **1.258** |
| 500 | 1.718 | 1.265 | 1.306 | **1.140** |
| 1,000 | 1.672 | 1.226 | 1.217 | **1.119** |
| 2,000 | 1.744 | 1.158 | 1.159 | **1.054** |
| 3,000 | 1.398 | 1.058 | 1.117 | **1.042** |
| all (≈ 4,200) | 1.047 | 1.047 | 1.047 | 1.047 |

The mechanism, in ligands bought per fold: at a 2,000-row budget DEPTH has acquired 20 extractants,
DIVERSITY 48, MAXMIN more still. DEPTH spends its budget re-measuring the best-measured ligands.

### 6.2 The contrast that matters is against RANDOM, not against DEPTH

Beating DEPTH at a small budget is close to tautological: at 250 rows DEPTH is a **1.3-ligand model**.
The protocol's §6 requires an acquisition claim to beat *random acquisition* too. Paired bootstrap
over chemotype blocks, positive = the first policy better:

| budget | RANDOM − DEPTH | CI95 low | MAXMIN − RANDOM | CI95 low | DIVERSITY − RANDOM | CI95 low |
|---|---|---|---|---|---|---|
| 250 | +0.599 | +0.478 | +0.046 | −0.059 | −0.003 | −0.043 |
| 500 | +0.472 | +0.295 | **+0.133** | +0.058 | +0.038 | −0.002 |
| 1,000 | +0.500 | +0.260 | **+0.061** | +0.003 | −0.016 | −0.058 |
| 2,000 | +0.620 | +0.368 | **+0.072** | +0.033 | −0.004 | −0.037 |
| 3,000 | +0.304 | +0.167 | **+0.052** | +0.024 | +0.037 | +0.003 |

Three separate conclusions, and only the first is large:

1. **Breadth beats depth, enormously and at every budget.** Simply spreading a fixed row budget over
   many ligands instead of concentrating it is worth 0.30–0.62 macro MAE, CI-clean throughout. At an
   equal number of measurements, one measurement each from many ligands is worth far more than many
   measurements of a ligand you already have.
2. **A deliberate max-min diversity policy beats random breadth** by a further +0.05 to +0.13, CI
   above zero at four of five budgets. Smaller, but real, and it is the policy an experimentalist can
   actually follow.
3. **Round-robin-over-chemotypes ("DIVERSITY") is not better than random.** Its CI includes zero at
   four of five budgets. Random acquisition already spreads across ligands; the extra structure of
   dealing one ligand per chemotype adds nothing. Only max-min distance does.

Pre-registered verdicts: **B1 PASS** (DIVERSITY beats DEPTH, CI-clean at all five budgets) — but as
written, B1 was the wrong test, and the honest reading is row 1 and row 2 above rather than B1's
literal statement. **B2 PASS** (the advantage is larger on hard chemistry: +0.64 to +0.87 versus
+0.34 to +0.59 overall, at every budget). **B3 FAIL** — the DEPTH curve is *still improving* between
the two largest budgets (Δ 0.346, CI95 [+0.195, +0.542]), so the curve has not saturated and the
whole experiment is budget-limited. Nothing here says where depth would plateau.

### 6.3 Caveats the run records itself

* 120 point-folds could not spend their budget because the fold's whole training pool was smaller
  than the budget (the fold holding the 64 %-of-rows chemotype has by far the smallest pool). In
  those folds every policy trains on the identical pool, so the contrast is structurally zero and the
  reported gains at 2,000 and 3,000 are **diluted towards zero**.
* One feature family (`MC_lig2d_ext_massaction`) only. A policy that helped just one representation
  would be an artefact of it; that check is a separate invocation and was not run.
* Policies acquire whole ligands. A policy allowed to choose individual rows is a different
  experiment.

## 7. What this does and does not license

**It licenses**, per the protocol's decision tree, case A:

* buying chemical diversity rather than more conditions on ligands already covered;
* an active ligand-acquisition study (Experiment F) as the next modelling step, since the value of a
  new chemotype is now measured rather than assumed;
* a hierarchical level model (Experiment C) aimed specifically at the per-ligand offset, which is
  where the whole effect lives.

**It does not license**:

* a claim about absolute accuracy. EXPANDED's macro MAE is 1.05 log units on this
  chemotype-held-out cohort. That is better than BASE and still far from useful for a single new
  ligand; the co-primary hard-chemistry endpoint is worse still (BASE 1.54, and even after the
  improvement the nn < 0.4 subset stays above 1 log unit).
* extrapolating the slope. 61 extractants bought 0.163 log units; nothing here says the next 61 buy
  another 0.163. Experiment B is the evidence on that question and it **fails B3**: the depth curve
  had not saturated at the largest budget, so no plateau has been located in either direction.
* a claim that the model got better on chemistry it already had. On the 91 BASE-eligible ligands the
  expansion is +0.019 (champion features) and **−0.018** (donor census) — §5.5.
* any statement about the shape term. It did not move, in any arm, in any bin, in any regime, in
  either experiment.
* treating the five split seeds as five experiments. They re-partition the same 152 ligands; the
  chemotype bootstrap is the evidence and the seed count is only a consistency check.
* a strong claim that the published noise floor is an artefact. Only a quarter of the "replicate"
  cells differ in a recoverable physical variable, and the cells with no recoverable difference
  scatter *more* than those with one — §2.2.

**What would falsify the headline**: a run in which EXPANDED's advantage disappears under row
weighting; in which `EXPANDED_SHUFFLED` matches `EXPANDED` (the gain would then be regularisation,
not chemistry); in which the nn < 0.4 gain is no larger than the overall gain (a depth effect in
coverage costume); or in which the gain does not scale with how much closer the added chemistry
brought the training set (it would then be a class label rather than chemical transfer). All four
were tested; none happened.

**Corrections made to this document after an adversarial review, recorded rather than silently
edited**: §2.2's "the noise floor is not a noise floor" was too strong and mis-attributed the
whole-set spread to the 26-cell subset — both the text and the generating code are fixed; §5.5 now
leads with the dense/sparse decomposition and reports the donor census's negative dense-cluster gain,
which the first draft omitted. The Experiment A numbers themselves did not change.

---

## 8. Limitations found by adversarial review, and what is not built

An adversarial review of this layer (five independent lenses plus a verifier) produced the
corrections already folded into §2.2, §3, §5.3, §5.4 and §5.5. Four further limitations are recorded
here rather than fixed, because fixing them is a different piece of work.

**1. "One extractant" is coarser than one chemical system.** 17 of the 190 `canonical_smiles`, covering
**50.3 % of rows**, carry more than one `extractant_name`. The worst is TODGA, whose SMILES carries
**21** names: in the CORDIS block the *aqueous holdback agent* (PyTri-diol, CITAM, PHEN-dialcohol …)
was recorded in the extractant-name field while the SMILES stayed TODGA. So a "ligand" in every
generation of this project — the grouping unit of every fold, the unit of `offset_mae`, and the unit
whose repeated cells the noise floor is estimated from — is sometimes a family of different aqueous
systems sharing one organic extractant. This does not create leakage (all rows of one SMILES stay on
one side of every fold) and it does not change any contrast here, since it affects every arm
identically. It does mean part of the irreducible-looking within-ligand *shape* error is a real,
unencoded experimental variable. Recovering the holdback agent from the upstream tables is the
cheapest available test of that.

**2. Cohort membership is weakly target-dependent** — inherited, not introduced.
`levels.build_level_dataset` drops rows with `log_D ≤ −6` *before* counting cells for eligibility, so
which extractants qualify depends slightly on their own targets. Measured: permuting `log_D` in the
source and rebuilding gives 5,249 rows instead of 5,248, with 5 row ids differing. Everything gen6
adds on top — the chemistry map, the clusters, the splits, the acquisition order — is exactly
target-invariant, and is unit-tested to be. Three rows in 5,992 is far below any effect reported
here, but `gen6/chemistry.py`'s "nothing here reads `log_D`" is true of the map and not of the cohort
the map is applied to.

**3. A pre-registered feature set was substituted.** The protocol named `MC_donors_massaction`, which
does not exist in `levels.LEVEL_ARMS`; the runs use `MC_donors`. That is a protocol deviation, it was
found by the runner's own defaults crashing rather than by inspection, and it is now pinned by a test
(`test_default_feature_sets_resolve`). The donor census without the mass-action block is a *weaker*
arm, so the substitution does not flatter any result.

**4. What is deliberately not built.** The brief lists nine scripts; five exist. The four that do not
— hierarchical level decomposition (C), multi-fidelity level→pair transfer (D), active k-shot design
(E), retrospective ligand acquisition (F) — are Phase 2 and Phase 3 in the brief's own execution
order, conditional on Phase 1's outcome. Phase 1 has now returned case A, which licenses C and F
specifically; neither was started, because the point of the decision tree is that the results choose
the next branch and those results are what this document reports.
