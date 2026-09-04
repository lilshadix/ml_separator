# GEN11 §9 — what each arm is actually made of

**Headline: the archive nearly doubles the row count and adds almost no new chemistry.**
The full auxiliary pool is 5,438 model-ready rows against the cohort's 5,938 source
records (+91.6 %). Of those 5,438 rows, **3,278 (60.3 %) carry a ligand whose 2048-bit
Morgan fingerprint is *identical* to a cohort ligand's** — Tanimoto 1.0, 77 structures —
and **4,634 (85.2 %) sit in gen10's "near" distance tercile**. Only **932 rows (17.1 %),
on 45 structures forming 38 new chemotypes, fall outside every one of the cohort's 79
Tanimoto chemotypes**, and only 315 rows (5.8 %) are in gen10's "far" tercile. The
row-weighted median maximum Tanimoto from an auxiliary row's ligand to the cohort is
**1.0**.

What the archive genuinely adds is **metals** (14 lanthanides → 40 elements) and
**curves** (1,229 → 2,206 usable curves), not ligands. Whether transfer works therefore
has to be a claim about *metals informing lanthanide behaviour on chemistry the cohort
already contains*, not a claim about new chemical coverage. Anything gained by arms C or
E should not be attributed to breadth of chemistry, because there is barely any.

Two numbers make the repetition concrete:

| | control (cohort only) | E_LN_PLUS_ALL |
|---|---|---|
| rows | 5,938 | 11,376 |
| distinct chemotypes | 79 | 117 |
| **effective** chemotypes (`exp` entropy of row shares) | **6.46** | **9.15** |
| effective publications | 47.38 | **44.22** |
| largest single publication's row share | 8.2 % | 18.0 % |

Doubling the rows buys 2.7 effective chemotypes and *reduces* effective publication
diversity, because 1,880 of the 5,438 auxiliary rows (34.6 %) come from one campaign
(`10.1021/acssuschemeng.4c06166`) and 1,841 (33.9 %) are americium.

## Files

| file | contents |
|---|---|
| `arm_composition.csv` | one row per arm; `aux_*` = the auxiliary pool alone, `total_*` = cohort ∪ auxiliary (what the arm trains on) |
| `curve_inventory_by_arm.csv` | usable curves per arm, twice: `scope = cohort_plus_auxiliary` and `scope = auxiliary_only`, split by axis and by ligand novelty |
| `chemotype_distance.csv` | one row per auxiliary structure: max Tanimoto to the cohort, nearest cohort structure and chemotype, assigned chemotype or `NEW`, gen10 distance bucket |
| `concentration_diagnostics.csv` | largest-unit row shares and effective unit counts, per arm, for `scope ∈ {aux, total}` |
| `composition_audit.json` | every self-check, the bucket cut points and their source, the overlap audit |

Built by `src/lanthanide_separation/gen11/composition.py` (`python -m
lanthanide_separation.gen11.composition`).

## Arms

Auxiliary pools are drawn from `gen11.overlap.build_overlap_map(...).auxiliary_candidates`
— archive records that are **not** frozen-bundle rows — further restricted to
`model_readiness == "A_model_ready"`. That gate takes 10,778 candidates down to 5,438; it
drops 993 tier-B (usable with caveats), 2,443 tier-C (not modelable) and 1,904 tier-D
(redundant duplicate) records. Mixing tier B in would confound "more data" with "worse
data".

| arm | auxiliary rows | structures | structures in **no** cohort chemotype | new chemotypes | rows on new chemistry | rows with a cohort-identical fingerprint |
|---|---|---|---|---|---|---|
| A_GEN10_CONTROL | 0 | 0 | 0 | 0 | 0 | 0 |
| B_LN_EXPANDED | 86 | 6 | 1 | 1 | 11 | 75 |
| C_LN_PLUS_ACTINIDES | 4,896 | 153 | 44 | 37 | 908 | 2,779 |
| D_LN_PLUS_NON_ACTINIDE | 456 | 7 | 1 | 1 | 13 | 424 |
| E_LN_PLUS_ALL | 5,438 | 155 | 45 | 38 | 932 | 3,278 |
| F_ACTINIDES_ONLY_MATCHED | 456 | 81 | 23 | 19 | 82 | 255 |
| G_RANDOM_AUX_MATCHED | 456 | 81 | 25 | 20 | 91 | 266 |

`B` is 86 rows over 6 structures with one metal holding 90.7 % of them. `D` is 456 rows
but only 7 structures, and a single extractant system holds 70.8 % of them (effective
structures 2.58, effective chemotypes 1.35). Neither is a diversity arm; both are size
arms, and a positive result on either would be a result about a handful of systems.

`F` and `G` are size-matched to `D` at the **row** level with a fixed seed
(`MATCH_SEED = 11`, `numpy.random.default_rng`, records sorted by
`canonical_measurement_id` first). That is the brief's wording and it makes the row
counts literally equal, but it has a cost that is visible in the curve table: random rows
are not curves. From the same 456 rows, `D` yields **79** usable auxiliary curves while
`F` yields **12** and `G` **13**. If the mechanism under test is shape transfer, `F`/`G`
are matched on the wrong quantity, and a series-level matched sampler should be added
before those two arms carry a shape claim.

## Curves

**Definition (the archive's own, restated).** A *curve* is a maximal subset of one series
in which exactly one axis varies and every other axis is held fixed
(`dataset_all_metals/scripts/stage04_series.py`, `SERIES_CATEGORICAL` + `CURVE_AXES`;
concentration axes on the log10 scale; `log_D` is never read to define membership). A
curve is *usable* when at least `MIN_CURVE_POINTS = 3` of its member records carry a
finite axis coordinate and a finite `log_D`, and those coordinates have non-zero range
(`stage04_series.py:205-222`).

That rule, recomputed here over every canonical archive record, reproduces the archive's
usable set exactly — 2,331 curves, **identical `curve_id` sets**, not merely an equal
count (`composition_audit.json → curve_rule_check`). A stricter variant that also demands
three *distinct* axis coordinates is reported alongside as
`usable_curves_distinct_axis_rule`; over the whole archive it admits 2,326, the five
differences being four metal series and one metal-concentration curve where three records
sit on two coordinates.

Per-arm counts restrict the rule to that arm's record set, because an arm holding most of
a titration does not necessarily hold a usable curve.

| arm | usable curves (cohort ∪ aux) | extractant titration | acid titration | metal series | other axes | auxiliary-only curves | of those, on new chemistry |
|---|---|---|---|---|---|---|---|
| A_GEN10_CONTROL | 1,229 | 244 | 459 | 407 | 119 | 0 | 0 |
| B_LN_EXPANDED | 1,235 | 244 | 464 | 407 | 120 | 6 | 0 |
| C_LN_PLUS_ACTINIDES | 2,104 | 596 | 775 | 488 | 245 | 864 | 156 |
| D_LN_PLUS_NON_ACTINIDE | 1,325 | 291 | 482 | 430 | 122 | 79 | 2 |
| E_LN_PLUS_ALL | 2,206 | 643 | 803 | 511 | 249 | 954 | 158 |
| F_ACTINIDES_ONLY_MATCHED | 1,242 | 249 | 466 | 408 | 119 | 12 | 2 |
| G_RANDOM_AUX_MATCHED | 1,243 | 249 | 466 | 409 | 119 | 13 | 5 |

The curve count is the one place the archive is unambiguously generous: `E` takes usable
curves from 1,229 to 2,206. Note the total rises by 977 while the auxiliary rows form only
954 curves on their own — 23 curves become usable only once cohort and auxiliary points on
the same curve are combined. But of `E`'s 954 auxiliary curves, **553 (58.0 %) sit on a
structure the cohort already contains** and only 158 (16.6 %) on structures outside every
cohort chemotype (262 near, 81 mid, 58 far). gen10's shape evidence was thin because it
rested on few chemotypes; the archive thickens it mostly on chemistry that was already
there, and mostly with other metals.

## Chemotype distance — the measured version of "new chemistry"

`chemotype_distance.csv` gives, for each of the 155 auxiliary structures, the maximum
Tanimoto to any of the cohort's 152 structures, computed on the dataset's own fingerprint.
That fingerprint is not assumed: `assert_fingerprint_reproduces_cohort` rebuilds the
cohort's `ecfp_*` block from SMILES on every run and requires all 152 ligands to match
**bit-for-bit** (Morgan radius 2, 2048 bits; radius 1 and radius 3 each match 0 of 152, so
the convention cannot drift silently).

A structure is mapped onto a cohort chemotype when that maximum reaches
`levels.TANIMOTO_CLUSTER_THRESHOLD = 0.7` — the same single-linkage threshold that defined
the cohort's 79 clusters, so an arm cannot "add a chemotype" that the fold grouping would
still treat as seen. 110 of 155 structures map onto a cohort chemotype (43 distinct
chemotypes touched); the remaining 45 are in none and form 38 new chemotypes among
themselves at the same threshold.

Distribution of the maximum Tanimoto over the 155 structures: min 0.209, p25 0.678,
median 0.929, p75 1.0, max 1.0. Seventy-seven structures reach 1.0 — 72 of them are
literally cohort SMILES; the other 5 are different molecules with bit-identical
fingerprints, the homolog trap `levels.ecfp_cluster_labels` was written for.

### near / mid / far

Reused verbatim from gen10, not redefined. Source: **`scripts/gen10_budget_simulation.py:156`**

```python
gains["distance_tercile"] = pd.qcut(1.0 - curves["nn_train_tanimoto"], 3,
                                    labels=["near", "mid", "far"])
```

— equal-frequency terciles of Tanimoto *distance* to the nearest training-set structure,
over the 99 common-cohort ligands in
`runs/gen10_final/budget_simulation/adaptation_curves.csv`. The cut points are data-defined
rather than constants, so they are recomputed from that frozen file at run time; they come
out at

* **near**: distance ≤ 0.342857 (Tanimoto ≥ 0.657143)
* **mid**: 0.342857 < distance ≤ 0.411716 (0.588284 ≤ Tanimoto < 0.657143)
* **far**: distance > 0.411716 (Tanimoto < 0.588284)

Applied to the auxiliary structures — where the reference set is the whole cohort rather
than one fold's training ligands — 120 structures are near, 10 mid, 25 far; by row, 4,634
near, 489 mid, 315 far. The far rows are dominated by short-chain monoamides and
dibutyl phosphate measured on Th/U/Np/Pu.

## Concentration

`concentration_diagnostics.csv`, `scope = aux` (the auxiliary pool alone):

| arm | largest publication | largest extractant system | largest metal | effective publications | effective structures | effective chemotypes |
|---|---|---|---|---|---|---|
| B_LN_EXPANDED | 0.660 | 0.430 | 0.907 | 3.35 | 2.11 | 1.76 |
| C_LN_PLUS_ACTINIDES | 0.427 | 0.172 | 0.376 | 11.83 | 37.41 | 9.97 |
| D_LN_PLUS_NON_ACTINIDE | 0.207 | 0.708 | 0.395 | 10.88 | 2.58 | 1.35 |
| E_LN_PLUS_ALL | 0.383 | 0.219 | 0.339 | 15.24 | 32.63 | 8.92 |
| F_ACTINIDES_ONLY_MATCHED | 0.430 | 0.175 | 0.366 | 11.17 | 30.95 | 8.95 |
| G_RANDOM_AUX_MATCHED | 0.402 | 0.197 | 0.322 | 12.74 | 30.15 | 8.65 |

Effective counts are `exp` of the Shannon entropy of the row shares; missing labels are
dropped rather than pooled, so the 534 auxiliary rows without a DOI do not read as one
enormous publication. The `scope = total` rows in the same file give the same statistics
for the training pool each arm actually sees.

Read `B` and `D` off this table before reading any result from them: `B` is effectively
3.35 publications, 2.11 structures and 1.76 chemotypes; `D` is 2.58 structures and 1.35
chemotypes. gen2's macro/micro trap was one extractant holding 43.6 % of pairs — `D` is
worse than that.

## Checks

All passing on the run that produced these files (`composition_audit.json`):

* ECFP recomputation is bit-exact on all **152/152** cohort ligands.
* Recomputed curve usability over all canonical records reproduces the archive's usable
  **curve_id set** exactly (2,331 curves); no count-only agreement was accepted.
* `overlap` audit: 5,992 frozen bundle rows join 1:1 to archive records, metal agreement
  1.0, max |Δ log_D| 5.55e-14; 5,938 of them feed the 5,248 cohort rows (54 dropped by the
  cohort's own filters), replicate-averaged target reproduces to 0.0.
* Chemotype labelling leaves **0** rows unresolved in every arm
  (`total_chemotypes_unresolved_rows`), and rdkit rejected 0 of the 155 auxiliary SMILES.

No check failed. One design issue, not a check failure, is recorded above: row-level size
matching leaves `F` and `G` with 12 and 13 usable auxiliary curves against `D`'s 79.
