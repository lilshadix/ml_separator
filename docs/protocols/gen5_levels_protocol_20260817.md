# gen5 — predicting the extraction level `log D` (protocol, pre-registered 2026-08-17)

Status: **pre-registration**. Written before the full run; only `--quick` smoke runs (1 seed,
3 folds, 80 trees, an arm subset) have been seen. Decision rules below are frozen as of the
commit that adds this file. The harness was adversarially reviewed before freezing (five lenses,
~38 findings; the design changes they forced are marked ★ below).

> **Results and errata — read alongside this file.**
> [`gen5_levels_results_20260818.md`](../results/gen5_levels_results_20260818.md) analyses the first run
> (`runs/gen5_levels_20260817T195411Z`) and records nine harness defects found afterwards, all
> fixed on 2026-08-18. Three of them affect how this protocol must be read:
> * The first run executed **two of the four regimes below** — `unseen_chemotype` and
>   `unseen_series` were hard-coded out of the SLURM defaults — so **H1′, H2, H3 and H5 were
>   unevaluable**. All four regimes are now the default.
> * H1's and H4's pass conditions are defined against `MC` and `MC_all2d`, but the shipped
>   bootstrap only ever anchored on `MC_ecfp`, so **the required CIs did not exist**. The
>   bootstrap now anchors on `MC_ecfp`, `MC`, `MC_all2d` and `A_metal`, and it resamples the
>   regime's *held-out* unit rather than always the ECFP cluster.
> * `within_ligand_r2` as originally implemented granted a free per-ligand offset taken from
>   held-out data. It is now the deployable `1 − SSE/SST_within`; the old quantity survives as
>   `within_ligand_r2_shape`. **Any `within_lig_r2` figure from the first run is the shape one.**
>
> Adjudicated outcomes: **H1 passes** (on `LIG2D_EXT`, not the expected `ECFP`), **H4 passes**
> strongly (3D is *worse* than 2D), **H6 passes**. The headline caveat is that the model's edge
> over a constant is ~0 below Tanimoto 0.4.

Code: `src/lanthanide_separation/levels.py`, `scripts/run_gen5_levels.py`,
`tests/test_levels.py` (20 tests), `slurm/gen5_levels.slurm`, `slurm/submit_gen5_levels.sh`.

## 1. Why a level model at all

gen2–gen4 modelled the pair target `log_SF = log D_A − log D_B`: *which* of two lanthanides an
extractant prefers. That cancels the extractant's absolute level by construction, so it cannot
answer the prior question — *does the extractant pull the metal into the organic phase at all,
and how much, at these conditions?* A ligand can separate La/Lu by 10× and still extract 1 % of
either. Cascade design needs both numbers; this study supplies the first.

Facts measured on the raw table that shape the design:

* **Larger, more diverse cohort.** A pair needs two metals under identical recorded conditions
  (100 of the 190 extractants have only one metal measured — pairs are impossible for them by
  construction). A level row needs one measurement. With ≥ 10 unique cells per extractant: **4,881 rows,
  91 extractants, 74 ECFP clusters, 40 Tanimoto-0.7 super-clusters, 1,939 conditions, 230
  measurement series**; ~50 of the 91 extractants are *not* diglycolamides.
* **Signal lives in ligand + conditions.** Variance of `log D` explained: ligand 44 %, metal 5 %,
  ligand + condition 89 %, replicate ceiling 97 %. Replicate reproducibility over 313 replicated
  cells: median within-cell sd 0.237 → MAE floor ≈ 0.19 against a single new measurement; the
  pooled sd is 0.78 because 36 cells differ by > 1 log unit and are almost certainly *not* true
  repeats (an unrecorded parameter differs). Both are reported; the median is the working floor.
* ★ **"Conditions" are titrations.** 1,939 distinct condition vectors collapse to 231 *series*
  (one extractant × one categorical setting; only acid molarity / time / temperature /
  concentrations vary). Holding out a condition leaves its titration neighbours in training.
* ★ **"New ligand" has degrees.** Bit-identical ECFP grouping (75 clusters) still leaves close
  homologues across a fold: median nearest-neighbour Tanimoto 0.72; 24 of 75 clusters have a
  neighbour ≥ 0.8. Single-linkage at Tanimoto ≥ 0.7 gives 40 super-clusters.
* ★ **Some expected descriptors are empty.** xTB binding energy, HOMO/LUMO, strain energy are
  all-null in this table (dropped and recorded in the audit); the `geom_cond__*` block is 10
  *string* categories (one-hot → 46 indicators).

## 2. Cohort

`build_level_dataset` defaults: `min_rows_per_extractant = 10`, `replicate_policy = "mean"`
(replicated (extractant, condition, metal) cells averaged, `n_replicates` recorded), rows with
`log D ≤ −6` dropped (3 sentinel rows at −12.5), geometry not required. One row per
(extractant, condition, metal). Identity labels: `ecfp_cluster` (sha1 of the 2,048 bits),
`tanimoto_cluster` (single-linkage ≥ 0.7), `condition_id` (full 64-vector hash, missing = a
value), `series_id` (extractant × categorical condition part).

## 3. One model per property family

Same estimator family as the pair champion for every arm (ExtraTrees, median-impute + missing
indicators, cluster-balanced sample weights, 400 trees, `max_features 0.30`, `min_samples_leaf 2`
— fixed, not tuned). Property families present in the data:

| family | cols | what it is |
|---|---|---|
| METAL | 3 | Z, ionic radius, lanthanide index |
| COND | 64 | acid, concentrations, diluent, additives, T, time, phase ratio |
| PHYSCHEM | 10 | RDKit scalars: MolWt, TPSA, logP, rings… |
| ECFP | 2,048 | Morgan fingerprint bits |
| LIG2D_EXT | 206 | gen4 extended 2D descriptor table |
| DONORS | 13 | donor-atom census (amide-O, ether-O, aromatic-N, S…), denticity, CN, ligand count |
| COMPLEX_PHYS | 26 | xTB dipole, donor partial charges, coordination number (5 energy cols empty) |
| POLYHEDRON | 58 | coordination-polyhedron geometry |
| GEOM_COND | 46 | one-hot geometry-environment bins |

Arms: **A_\<family\>** (family alone), **MC** (metal + conditions), **MC_\<family\>** (family
added to MC), **MC_all2d**, **MC_all3d**, **MC_everything** — 18 arms. Baseline for deltas:
`MC_ecfp`. Alternative learners on `MC`, `MC_ecfp`, `MC_lig2d_ext`, `MC_all2d`: histogram
gradient boosting (`@hgb`, absolute-error loss) and standardised ridge (`@ridge`); every learner
is clamped to the training target range ± 50 % (a ridge fit once extrapolated to MAE 4×10⁶).

Controls through identical folds: `MC_ecfp_SHUF`, `MC_lig2d_ext_SHUF`, `MC_donors_SHUF`,
`MC_cond_SHUF` (block permuted in training); nulls `NULL_global_mean`, `NULL_metal_mean`, ★
`NULL_extractant_mean` (training mean of the same ligand), ★ `NULL_nearest_condition` (copy
log D from the nearest training condition of the same ligand + metal). ★ Shuffle twins are known
to be *worse* than a true ablation (they inject noise, not absence), so a block's contribution is
read from `MC` vs `MC_<family>`, and the twin is used only as a sanity floor.

## 4. Regimes ★

★ Randomised grouped K-fold (shuffled groups dealt round-robin) — `sklearn.GroupKFold` is
greedy-by-size and left 82 % of rows in the same fold across "seeds". 5 folds × 5 seeds.

| regime | held-out unit | question |
|---|---|---|
| `unseen_chemotype` | Tanimoto-0.7 super-cluster (40) | genuinely new chemistry |
| `unseen_ligand` | ECFP cluster (75) | new ligand, close homologues may be in training |
| `unseen_series` | measurement series (231) | known ligand, new acid/diluent setting |
| `unseen_conditions` | condition vector (1,939) | known ligand, titration interpolation |

For the two ligand regimes every test row carries `nn_train_tanimoto` (nearest training ligand),
so results can be stratified by how novel the held-out ligand really was.

`kshot` — post-hoc on the `unseen_ligand` OOF: `offset` / `scale` / `affine` (`trend` is a
pair-model notion), λ = 1, k ∈ {1,2,3,5}, 20 draws; primary column `mae_free` (query rows whose
condition is *not* in the support); ★ null `NULL_kmean` = mean of the k measured values with no
model.

## 5. Metrics and decision rule (frozen)

Primary: **macro MAE over ECFP clusters** (one ligand cluster = one vote). Also: pooled MAE / R²
(row-weighted; TODGA is 1,714 rows), ★ **within-ligand R²** (variance explained after removing
each ligand's mean — what a chemist choosing conditions needs; pooled R² mostly ranks ligands),
dispersion, fraction within 0.5 / 1.0 log, per-cluster MAE and guarded R² (sd ≥ 0.5 log).
Paired **cluster** bootstrap (5,000 replicates), one-sided add-one `p_worse`.

Pre-registered hypotheses (macro metric; regime in brackets):

| # | Hypothesis | Passes iff |
|---|---|---|
| H1 | A ligand family helps a **new** ligand: `MC_<fam>` < `MC` [unseen_ligand] | CI95 low > 0, ≥ 4/5 seeds, for at least one 2D family |
| H1' | …and survives **new chemistry** [unseen_chemotype] | same, for the family that passed H1 |
| H2 | Conditions carry real signal: `MC` < `NULL_metal_mean` and `< MC_cond_SHUF` [unseen_series] | CI95 low > 0 for both |
| H3 | Known ligand / new **series** beats the honest nulls: `MC_ecfp` < `NULL_nearest_condition` and `< NULL_extractant_mean` [unseen_series] | CI95 low > 0 for both |
| H4 | 3D families add nothing over 2D (as in every pair study): `MC_all3d` ≥ `MC_all2d`; `MC_everything` ≥ `MC_all2d` [unseen_ligand] | Δ CI95 spans 0 or is < 0 |
| H5 | Interpolation is not the same as prediction: `unseen_conditions` macro < `unseen_series` macro by > 0.05, and `NULL_nearest_condition` is within 0.10 of `MC_ecfp` there | reported, no gate |
| H6 | k-shot: `offset` k=1 beats k=0 on `mae_free`, **and** beats `NULL_kmean` | both Δ > 0 on all seeds |

Stated expectations (so a surprise is recognisable; quick-run values in brackets): H1 marginal
(ECFP Δ ≈ 0.0–0.08); H1' likely fails (Δ ≈ 0.02); H2, H3 pass (0.82 vs 1.17 / 1.18); H4 null;
H5 yes (0.75 vs 0.82, nearest-condition null 0.84 ≈ model 0.75); H6 passes vs k=0, uncertain vs
`NULL_kmean`.

Not claimed: cross-scaffold generalisation without the `nn_train_tanimoto` stratification; any
comparison to the pair model's SF numbers.

## 6. Running it

Local smoke: `.venv/bin/python scripts/run_gen5_levels.py --quick` (~3 min).
Full run ≈ **1.9 h** on 8 CPUs for all four regimes (18 arms + 8 learner variants + 4 twins + 4
nulls × 4 regimes × 25 folds; HGB is the slow part). Measured peak RSS ≈ 1.6 GB. SLURM defaults
are now **all four regimes**, 16 G / 8 h:

```bash
DRY_RUN=0 slurm/submit_gen5_levels.sh
```

Useful overrides: `REGIMES=...` to narrow the run (it will be recorded in the report header),
`MIN_ROWS=3` to widen the cohort from 91 to 152 extractants, `KSHOT_MIN_QUERY` via
`--kshot-min-query` (keep it fixed when comparing k-shot tables across runs).

Outputs under `runs/gen5_levels_<stamp>/`: `oof_predictions.csv` (with `nn_train_tanimoto`),
`leaderboard.csv`, `property_families.csv`, `per_seed_metrics.csv`, `per_cluster_metrics.csv`,
`paired_bootstrap.csv`, `kshot_levels.csv`, `decision_report.txt`, `summary.json`.

## 6a. Post-hoc addition (2026-08-18, NOT pre-registered): the mass-action block

Added after the first run, so this is **exploratory** and every number from it must be labelled
so. The idea is chemistry, not tuning. For a neutral extractant L pulling Ln(III) out of a
nitrate medium,

```
Ln³⁺ + 3 NO₃⁻ + n L(org)  ⇌  Ln(NO₃)₃·Lₙ(org)        log D = log K_ex + n·log[L] + 3·log[NO₃⁻]
```

— **linear in the logarithms** of the concentrations that are actually varied. The raw `cond__*`
columns carry molarity spanning 4–9 orders of magnitude, so a tree has to approximate a
logarithm with a staircase of axis splits, which is exactly what produces the shrunken predictions
and unreachable tails seen in the first run.

The law was **checked on this cohort before being used**: on the 15 metal-series with a clean
extractant titration, the slope of log D vs log[L] is **2.64, IQR [2.36, 2.88]** — 100 % inside
the chemically admissible 1.5–4.5 (it is the solvation number) — with median linear R² 0.985; the
acid slope is 1.93 over 146 series.

Block `MASSACTION` (8 columns, prefix `massact__`): `log10` of the five continuous conditions,
plus the products the law asks for — `log[L]·DENTATE`, `log[L]·coreCN` (a tree cannot form a
product from its factors), and `log[L]·log[H⁺]`. It is **additive**: raw `COND` stays, so the
contribution is a clean ablation. Arms `MC_massaction`, `MC_ecfp_massaction`,
`MC_lig2d_ext_massaction`, `MC_all2d_massaction`, `MC_everything_massaction`; shuffle twin
`MC_massaction_SHUF`; the family table carries one row per pairing with CI and per-seed count.

Pilot (3 seeds, 300 trees, before it was wired in): every one of the 18 (seed × arm × regime)
comparisons improved; macro MAE −0.031 to −0.069. See the results doc for the full-harness
numbers.

## 7. Follow-ups this unlocks (not part of this run)

* **Expanded pair cohort.** Dropping "all 64 conditions recorded" takes the pair cohort from
  8,195 pairs / 35 extractants / 28 clusters to **14,173 / 90 / 77** (measured 2026-08-17);
  within-cell replicate spread is not worse for rows with missing condition values (median sd
  0.149 vs 0.290). Priority #1 of `gen4_combined_protocol.md`; next pair-model run.
* **SF from levels vs the direct pair model** on the shared pairs.
* **Cross-repo transfer** to/from lanthanidestrain's 162-extractant `log D` table.
