# Gen12Eu_pred — data audit

*Written before any Gen12 model was fitted. Every number below is produced by
`scripts/gen12_data_audit.py` and stored in `manifests/data_audit.json`; nothing here is
typed by hand. Reproduce with:*

```bash
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_data_audit.py
```

**Verdict up front.** The Eu cohort is buildable, its target is clean, and a chemotype-blocked
zero-shot benchmark is well defined. Three facts constrain what may be claimed from it, and all
three are structural rather than fixable: the effective number of independent chemical units is
about 14, not 183; one chemotype holds half the rows; and under a chemotype hold-out no test
extractant can ever be more than Tanimoto 0.698 from training, so "near" in this study means
"just below the clustering threshold", not "a close homologue was in training". One leakage
hazard is fatal if ignored and is handled by construction: 4,148 non-Eu rows sit on Eu cohort
extractants, so the cross-lanthanide arm must delete held-out chemistry under *every* metal.

---

## 1. Where the data comes from

| | |
|---|---|
| canonical source | `dataset with 3D structures/dataset.parquet` |
| SHA-256 | `fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd` |
| shape | 5,992 rows x 2,261 columns |
| metals | 14 lanthanides, no Pm and no Y |
| extractants | 190 canonical SMILES |

This is the immutable bundle every generation from gen2 onward is defined against; its digest is
pinned in `gen3_protocol.json` and the Gen12 cohort builder refuses to run if it has drifted.

Two side tables are joined, both keyed on structure and neither carrying a target:
`ligand_2d_descriptors.parquet` (190 x 207 RDKit and hand-crafted 2D descriptors, 100 % coverage
of the Eu structures) and `runs/gen7_architecture/cache/chemistry_map.parquet`, the frozen gen6
chemistry map over all 190 structures.

A second, larger archive exists — `dataset_all_metals/clean/master_clean.parquet`, 16,770 records
across 40 metals. **Gen12 does not train on it.** It is joined read-only, on
`safe_exp_id -> exp_id`, to recover publication identity, duplicate class, multi-component system
class and the data-quality review queue. The join succeeds on 1,566 of 1,566 Eu bundle rows and
the two copies of `log_D` agree to 3.0e-14, which is what licenses using the archive's provenance
columns as audit metadata for bundle rows.

## 2. How Eu is encoded, and what the Eu sample really is

Metal identity lives in two identical string columns, `metal` and `metal_symbol`; the Eu rows are
selected as `metal == "Eu"`. Every Eu row carries `metal_ox = 3`, `lanthanide_index = 7`,
`Atomic Number_metal = 63` and `Ionic Radius_metal = 1.066`. **All four are constant and are
excluded from every feature block** rather than passed in as zero-variance columns. Removing metal
identity as a source of variation is the entire point of fixing the metal, so leaving those columns
in would be a silent contradiction of the design.

The filter ladder from bundle to cohort:

| step | rows | extractants |
|---|---|---|
| bundle | 5,992 | 190 |
| `metal == Eu` | **1,566** | 183 |
| minus the TODGA name/structure quarantine | 1,441 | 183 |
| minus replicate collapse to one row per (extractant, condition) cell | **1,329** | **183** |

No minimum-rows-per-extractant filter is applied. gen5 measured that its `min_rows = 10` filter
discarded 37 of its 47 chemically novel extractants and later identified that filter as the real
ceiling on its own study. On a study whose subject *is* unseen chemistry, that filter would delete
the phenomenon. Few-shot eligibility is therefore a declared sub-cohort (§9), never a cohort filter.

**Cohort fingerprint `2a364bb5264e9935`.** 1,329 rows, 183 extractants, 162 bit-identical ECFP
clusters, 97 Tanimoto-0.7 chemotypes, 1,001 distinct condition vectors, 305 series.

## 3. Target

| | |
|---|---|
| column | `log_D` |
| definition | base-10 logarithm of the distribution ratio `D = [Eu]_org / [Eu]_aq` |
| units | dimensionless, read as decades |
| valid rows | 1,329 of 1,329 — finite everywhere, no nulls, no infinities |

Distribution after replicate collapse:

| statistic | value |
|---|---|
| mean | -0.060 |
| sd | 1.899 |
| median | 0.143 |
| min / max | -4.738 / 3.585 |
| interquartile range | -1.585 to 1.501 |
| rows below -4 | 3 |
| rows above 3 | 41 |

**`log_D(Eu)` is the primary target and no transformation is applied.** The three plausible
alternatives were all considered and rejected, and the reasons are recorded here so that the
choice cannot be revisited after seeing results:

1. **Raw `D`.** Spans nine orders of magnitude and is the quantity `log_D` is defined from. An MAE
   on `D` would be dominated by a handful of high-extraction rows. Rejected.
2. **A condition-adjusted level** — the per-(extractant, series) intercept that gen5 through gen11
   modelled. It is a derived quantity, it requires a fitted condition model to define, and gen7
   established that on that target no model beats a 1-nearest-neighbour Tanimoto lookup. It is a
   legitimate *secondary* decomposition and appears as such (§ level/shape in the
   pre-registration), but it is not the endpoint.
3. **`log_SF`, the pair separation factor** of gen2 through gen4. Undefined for a single metal.

Two censoring rules used elsewhere in the repository are inspected and **not triggered**: no Eu row
sits at or below gen5's `-6` detection-limit floor, and none of the three known sentinel-like
`safe_exp_id` values is an Eu row. No row is dropped on target grounds.

## 4. Canonical SMILES, duplicated structures and chemical identity

Structures are RDKit-canonical: re-canonicalising all 190 bundle SMILES reproduces them exactly,
183 of which appear with Eu. Every structure maps to exactly one fingerprint (asserted, fatal on
failure), and no extractant *name* resolves to more than one structure.

The reverse is not true and matters:

| hazard | count |
|---|---|
| structures carrying more than one recorded name | 15 (after quarantine) |
| cohort rows on such structures | 307 |
| structure pairs at Tanimoto 1.0 | 35 |
| highest similarity between two non-identical fingerprints | 0.973 |
| distinct structures collapsing to one InChIKey skeleton | 3 (one achiral SMILES plus two stereoisomers) |

The 35 bit-identical pairs are homologue series — an N-octyl and an N-dodecyl diglycolamide set
the same radius-2 bits — plus the stereoisomer triple. Under an exact-extractant hold-out these are
"unseen" extractants whose fingerprint is copied verbatim into training. This is the hazard the
repository's own leakage audit rates CRITICAL, and it is why the exact-extractant design is a
sensitivity here and not the headline (§8).

**The TODGA quarantine.** In the Eu rows, 476 carry TODGA's structure but 21 different names. One
upstream sub-source records a water-soluble *masking agent's* name beside the organic extractant's
*structure*, so 125 rows attach TODGA's structure, fingerprint and 3D complex to sulfonated BTBPs,
PyTri polyols, phenanthroline alcohols and eight `TWE-` systems. Those 125 rows sit 0.68 decades
higher in mean `log_D` than the 351 genuine TODGA rows, so they are not a harmless labelling
nuisance. The repository's provenance-backed rule
(`lanthanide_separation.pairs.apply_default_quarantine`) is applied unchanged and removes them.
Their identity is recorded in `manifests/data_audit.json` under `quarantine_names`.

## 5. Structure of the chemical space

| | |
|---|---|
| extractants | 183 |
| bit-identical ECFP clusters | 162 |
| chemotypes, frozen gen6 map | **97** |
| chemotypes, recomputed on Eu structures alone | 97 |
| largest chemotype | 46 extractants, 663 rows (**49.9 % of the cohort**) |
| extractants with no cohort neighbour above Tanimoto 0.4 | 12 |
| median nearest-neighbour similarity within the cohort | 0.723 |

The two chemotype counts coincide at 97 but the partitions are *not* identical: the frozen map,
computed over all 190 bundle structures, merges Eu clusters that non-Eu structures bridge (its
largest Eu chemotype holds 46 extractants against 36 for the Eu-local version). Merging makes a
held-out chemotype larger and the hold-out stricter, which is the safe direction, and it keeps the
Gen12 chemotype the same object as gen6 through gen11's. **The frozen map is used.**

Scaffold families, from the frozen map's RDKit-motif assignment: diglycolamide 74,
N-heterocyclic polydentate 41, other amide 28, malonamide 11, phosphoryl 11, sulfur donor 8,
podand ether 4, hydroxy acid 3, other 3.

**Only 662 of the 2,048 fingerprint bits vary across the cohort**; the other 1,386 are constant.
Any statement of the form "the model has 2,048 structural features" is wrong by a factor of three.

## 6. ECFP settings, recovered rather than assumed

The bundle ships the fingerprint but not its recipe. It was recovered by exhaustive search over
RDKit generator settings against all 190 structures:

| setting | value | evidence |
|---|---|---|
| algorithm | Morgan / ECFP | — |
| radius | **2** | 190/190 structures reproduce bit-exactly at radius 2; 0/190 at radius 1 or 3 |
| size | **2,048 bits**, folded | column count and exact match |
| chirality | **excluded** | `includeChirality=True` matches only 184/190 |
| feature invariants | **off** (standard connectivity invariants) | feature invariants match 0/190 |

`AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)` reproduces all 190. Gen12 uses the
bundle's stored bits and never recomputes them, so this is a provenance record rather than a
dependency. The achiral setting is the reason the stereoisomer triple in §4 is one ECFP cluster.

Chemotypes are single-linkage clusters at Tanimoto >= 0.7 over these bits
(`lanthanide_separation.levels.tanimoto_cluster_labels`), unchanged from gen5 onward.

## 7. Experimental conditions and missingness

64 `cond__*` columns: 9 acid one-hots, 44 diluent one-hots, 9 additive one-hots, and 5 continuous
quantities. The one-hot blocks are clean — every row has exactly one acid and exactly one diluent,
never zero and never two. **12 of the 64 columns are constant** across the Eu cohort.

| continuous condition | non-null | median | range |
|---|---|---|---|
| acid concentration (M) | 1,328 | 1.00 | 1e-4 to 7.4 |
| extractant concentration (M) | 1,328 | 0.10 | 2.3e-4 to 1.0 |
| temperature (C) | 1,319 | 22 | 5 to 55 |
| contact time (min) | 829 | 30 | 0.16 to 1,800 |
| metal concentration (mM) | 587 | 0.18 | 1e-6 to 65.8 |

Missingness by column: metal concentration 55.8 %, contact time 37.6 %, temperature 0.8 %, acid
and extractant concentration 0.08 % each. Everything else is complete. Per feature block:

| block | columns | columns with any missing | mean missing fraction |
|---|---|---|---|
| COND | 64 | 5 | 1.5 % |
| MASSACT | 8 | 8 | 11.8 % |
| ECFP | 2,048 | 0 | 0 |
| PHYSCHEM | 10 | 0 | 0 |
| DONORS | 15 | 0 | 0 |
| LIG2D | 206 | 8 | 1.1 % |

MASSACT is the log-concentration block and inherits the missingness of the columns it transforms.
All imputation is fold-local (§ pre-registration) with a missingness indicator.

**Two condition axes that exist upstream and are absent from the bundle** are recorded rather than
silently ignored: nitrate concentration (present on 403 of the 1,566 raw Eu rows in the archive)
and the organic-phase acid concentration (36 rows). Recovering them is a declared non-goal of
Gen12; they are named here so that a later generation does not mistake the bundle's silence for
their absence.

## 8. Duplicates, replicates and the review queue

**Replicates.** Collapsing to one row per (extractant, condition) cell merges 1,441 rows into
1,329. Fifty cells hold more than one measurement. Their spread is usually small (median 0.065
decades) but **8 cells disagree by more than a full decade**, up to 3.75. Those eight are an
irreducible noise floor inside the target, not model error.

**Archive duplicate classes** on the surviving cells: 1,226 unique, 79 exact database duplicates,
21 value conflicts, 2 possible independent replicates, 1 insufficient information.

**Review queue.** The curated 610-record data-quality queue reaches **343 of 1,329 cohort rows
(25.8 %) across 67 extractants**: diluent name ambiguous 225, same value with conflicting condition
131, same conditions with different `log_D` 23, identical record under a different publication 2.
The repository's own triage found that identity-shaped flags carry about 44 % more irreducible
residual while value-conflict flags sit *below* the clean baseline, and that cleaning the whole
queue moves pooled MAE by 0.0013. Gen12 therefore **corrects nothing** and carries the flags as a
per-row column for stratified reporting.

**Multi-component systems.** 69 cohort rows are not single-extractant systems: 53 carry a phase
modifier and 16 an aqueous complexant. The bundle flattens these to the primary extractant's
structure, so their second component is invisible to the model. This is a known limitation of the
frozen bundle, is recorded per row, and is a declared sensitivity stratum.

## 9. Rows per unit, and what is left for few-shot

| unit | median | p95 | max |
|---|---|---|---|
| rows per extractant | 2 | 22.6 | 306 |
| rows per chemotype | 3 | 31 | 663 |
| extractants per chemotype | 1 | 5.2 | 46 |
| series per extractant | 1 | 4 | 33 |

**85 of 183 extractants have exactly one row and 132 have a single series.** Because the cohort is
one row per condition cell, "conditions per extractant" is identical to "rows per extractant" by
construction; it is not an independent statistic.

Extractants with at least *n* rows: n=1: 183, n=2: 98, n=3: 80, n=4: 73, n=5: 71, n=6: 64, n=7: 42,
n=10: 35, n=20: 10.

Few-shot eligibility, under the rule "k support points plus at least two queries":

| k | eligible extractants | rows | chemotypes |
|---|---|---|---|
| 1 | 80 | 1,208 | 49 |
| 2 | 73 | 1,187 | 43 |
| 3 | 71 | 1,179 | 41 |
| 5 | **42** | 1,012 | **21** |

The common few-shot cohort — extractants scorable at every k including 5 — is **42 extractants in
21 chemotypes**. A learning curve drawn on a cohort that shrinks with k is not a learning curve, so
the curve is drawn on this common cohort and the per-k cohorts are reported beside it.

## 10. Power, stated before any comparison

Kish effective sample sizes on the cohort:

| quantity | n_eff |
|---|---|
| rows, weighted by extractant | 15.6 |
| rows, weighted by chemotype | **3.9** |
| extractants, weighted by chemotype | **13.7** |

A row-level confidence interval on this cohort overstates confidence by roughly two orders of
magnitude. The bootstrap therefore resamples **chemotypes** (97 blocks, one dominant), and the
minimum detectable effect is computed from the realised standard error before any model comparison
is interpreted. On the far band the design has 25 chemotype blocks and about 17 held-out
extractants per seed; that is the binding constraint on the study's central question and it is
stated here rather than discovered afterwards.

## 11. Split designs and their measured properties

Both designs use the repository's `seeded_group_kfold` (shuffle unique group labels, deal round
robin), 5 folds x 5 split seeds `{104729, 130363, 155921, 196613, 262147}`, with a nested inner
validation block carved from the training groups only.

| | **B — chemotype hold-out (primary)** | **A — exact extractant (sensitivity)** |
|---|---|---|
| group | frozen Tanimoto-0.7 chemotype | canonical SMILES |
| groups | 97 | 183 |
| extractant overlap train/test | 0 | 0 |
| **ECFP-cluster overlap** | **0** | **139** across 25 folds |
| **chemotype overlap** | **0** | **234** across 25 folds |
| max train similarity | **0.698** | **1.000** |
| median train similarity | 0.611 | 0.714 |
| test rows per fold | 72 to 821 | 128 to 543 |

Two consequences, both load-bearing:

- **Design A is not a test of unseen chemistry.** It leaks bit-identical fingerprints in 139
  instances and reaches Tanimoto 1.0. It is reported as a sensitivity because it is the design a
  reader would naively construct, and its gap to design B is the measured cost of that naivety.
- **Under design B, similarity is capped at 0.698 by construction.** Holding out a single-linkage
  0.7 cluster guarantees no test structure exceeds the threshold to any training structure. The
  "near" band under design B is therefore the narrow slice 0.60 to 0.698, not "a close homologue
  was in training". Any cross-generation comparison must respect this; the same cap applies to
  gen10 and gen11, which used the same hold-out.

Band populations under design B, computed before any model ran:

| band | rule | extractants (any seed) | chemotypes | extractants per seed | rows per seed |
|---|---|---|---|---|---|
| far | max train Tanimoto <= 0.40 | 27 | 25 | 17.4 | 76 |
| mid | 0.40 to 0.60 | 103 | 59 | 71.2 | 395 |
| near | > 0.60 | 105 | 58 | 94.4 | 858 |

The bands are gen10/gen11's cut points, reused unchanged and frozen before any Gen12 model was
fitted. The far band is thin — 76 rows and about 17 extractants per seed — and that is reported as
a limit on power, not hidden.

## 12. Leakage hazards, enumerated and resolved

The audit script exits non-zero on any hazard marked fatal.

| hazard | status | evidence |
|---|---|---|
| random row splitting | **absent by construction** | no row-level splitter exists in `gen12eu`; every split is grouped |
| same extractant under two names | **handled** | identity is structural; names are audit metadata only; 15 multi-name structures recorded |
| different SMILES, same molecule | **measured, not eliminated** | 35 bit-identical pairs and one stereoisomer triple; blocked by design B, leaks under design A and is counted there |
| structure/name misattribution | **quarantined** | 125 TODGA-structure rows with a different recorded name removed |
| same publication across train and test | **measured, not blocked** | 24 of 80 DOIs span more than one chemotype, 311 rows; a DOI-blocked bootstrap is the declared robustness variant |
| features computed from the target | **asserted impossible** | a forbidden-token check over every feature block raises on `log_D`, `D_value`, `safe_exp_id`, `doi`, ids and paths |
| global preprocessing before splitting | **forbidden** | every imputer, scaler and encoder is fitted inside the fold; asserted by test |
| condition imputation using test rows | **forbidden** | same mechanism; fold-local medians with an indicator |
| model selection on the test set | **forbidden** | hyperparameters chosen on the nested inner validation block of each outer fold |
| **test extractants reachable through another metal** | **fatal if ignored; handled by design** | **4,148 non-Eu rows sit on Eu cohort extractants**; the strict cross-lanthanide arm deletes every row of a held-out chemotype under every metal |
| replicate cell split across folds | **impossible** | a cell is one row |
| series split across folds | **impossible under both designs** | a series belongs to one extractant |

The DOI hazard deserves its own sentence. A chemotype hold-out does **not** block publication batch
effects: 12 chemotypes draw on more than one publication and 24 publications touch more than one
chemotype. The repository measured that publication blocking widens intervals by about 40 %. Gen12
reports the chemotype-blocked interval as primary and the DOI-blocked interval alongside it, and
states which is quoted every time.

## 13. Cross-lanthanide inventory, for the Phase-10 arm only

| | |
|---|---|
| non-Eu rows in the bundle | 4,426 |
| non-Eu extractants | 95 |
| **non-Eu rows on Eu cohort extractants** | **4,148** |
| non-Eu extractants shared with the Eu cohort | 88 |
| non-Eu extractants absent from the Eu cohort | 7 |
| Eu extractants measured with no other metal | 95 |

The multi-lanthanide arm therefore adds **almost no new chemistry** — 7 extractants — and almost
entirely adds *other metals on the same 88 ligands*. That is exactly the configuration in which a
naive multi-metal model would look excellent while having already seen the test chemistry. The
strict arm's exclusion is at the chemotype level across all metals, and the auxiliary rows that
survive are counted and reported before the arm's result is interpreted.

## 14. Environment

Python 3.13.0, numpy 2.5.1, pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.0, RDKit 2026.03.5,
CatBoost 1.2.10, XGBoost 3.4.1, PyTorch 2.13.0. Recorded in full, with the git commit, in
`manifests/data_audit.json`.

A prior generation was moved by an unrelated `pip install` that silently downgraded scikit-learn
and pandas, shifting every result by about 0.0014 against effects of 0.004 to 0.02. Any package
added for Gen12 is dry-run first and rejected if it would move the scientific stack. Chemprop 2.3.1
was checked this way: it adds 28 packages and moves none of numpy, pandas, scipy, scikit-learn,
PyTorch, RDKit, CatBoost, XGBoost or pyarrow.
