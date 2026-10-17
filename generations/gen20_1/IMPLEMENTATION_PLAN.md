# gen20_1: absolute log D versus adjacent log SF on identical leakage-safe splits — implementation plan

**FROZEN WITH THE SEALED PRE-REGISTRATION (S4) — approved for launch (S2, 2026-10-16) with every open item at its
default.**

*Draft 1, 2026-10-15, written at HEAD 44df763 (main); frozen at S4 with `PRE_REGISTRATION.md` (its LF-normalised
digest is recorded in `manifests/seal_g201.json`; later changes only as dated entries in `ADDENDA.md`).*

**Status.** The user instructed on 2026-10-16 to run the comparison without answering O1–O27, so every open item
holds at its default and the values the draft had left open are resolved in place (O7 precedence POINT_ONLY; O9
grid K1–K4; exploratory family m = 83; O18 reduction details and O19 mechanism as in `PRE_REGISTRATION.md`). Result
entries reading "not computed" describe the state at the seal; results are produced only by the sealed scripts and
reported in `GEN20_1_REPORT.md`. **[audit]** marks a sourced audit fact; **[decision]** a design choice that
`PRE_REGISTRATION.md` freezes at the seal (S4). IDs
are shared with `README.md` and `PRE_REGISTRATION.md`. Brief: `generations/gen20.1/GEN20_1 INSTRUCTIONS.rtf` (commit
44df763), cited as [BRIEF]:RTF line.

**Source aliases.** Repo-relative paths; `gen13sep/` = `generations/gen13_separation/gen13sep/`, `gen20k/` =
`generations/gen20_kshot/gen20k/`, "levels csv" =
`generations/gen20_kshot/results/discovery/levels_BP_metal_C3_DOPT_Q.csv`, "recovery plan" =
`docs/protocols/lanthanide_separation_recovery_plan_20260927.md`. **[CNT]** = design-time label-free counts made
2026-10-15 from the frozen inputs (cell/metal/pair/fold membership and NaN masks only; no target value read); the
counting script is not part of this draft; every [CNT] value is recomputed at S3 by `scripts/g201_counts.py`, and a
mismatch is a STOP. **[DMD]** and **[INNER]** = design-time label-free counts of exactly the same kind and status
(2026-10-15; no target value read; script not part of this draft; recomputed at S3 by `scripts/g201_counts.py`, a
mismatch is a STOP): [DMD] counts the cells, entries and strict-adjacent pairs matched by the §1.1 DMDPhPDA rule on
the 521-cell cohort (all of them in C3); [INNER] counts the inner-fold sizes and checks the first chemotype inner fold
against the stored `Fold.inner_*` indices (§4). **[TIM]** gen13's tracked
`generations/gen13_separation/predictions/{BP_all,BP_v2,BP_lean}/timings.csv` (column `seconds`, timed on gen13's machine; only the prediction parquets there are gitignored, `.gitignore:86`): mean
C_DIRECT_ROW in BP_lean 3.58 s (used for A), S4_CENTRED_ROW@lean in BP_v2 1.67 s (C), V2_DIRECT@cond in BP_v2 0.76 s
(BD_COND); ALL-block C_DIRECT_ROW 15.0 s (BP_v2) and 26.8 s (BP_all). **[ENV]** this machine, recorded by gen201
`environment()` in the S7 manifest (reading of 2026-10-15: 12 logical CPUs, 8,464,097,280 bytes RAM, Python 3.14.5,
numpy 2.5.3).

## 0. Summary

On the frozen gen13 exact cohort (521 cells, 3,359 replicate-mean (cell, metal) observations; §1.1) and exactly gen20's
outer folds (design BP: chemotype 5-fold hold-out with the publication mask, 5 split seeds × 5 folds; §4), gen20.1 asks
how well and how reliably **absolute log D** per (cell, metal) and **strict-adjacent log SF** (|ΔZ| = 1, lighter metal
first) can be predicted for unseen chemotypes and publications ([BRIEF]:12-15). Only **mode Z** (k = 0) is registered:
no test-cell label enters anything. Three variants share test cells, folds, seeds and training observations
([BRIEF]:43-47): **A**, one ExtraTrees row learner predicting uncentred log D for every metal; **B**, the in-cell
difference of A's out-of-fold predictions (no re-selection, so A's cell-level error cancels, I5); **C**, the same
learner, features, weights and grid on within-cell-centred log D (direct contrast training, I6), reported separately and
never pooled with B. Primary rows are gen20's 229 C3 cells: 2,425 log D entries (COH-D-C3) and 1,802 strict-adjacent
pairs per seed (COH-SF-C3) ([CNT]; levels csv:73), sharing 38 chemotypes and 39 publications and hence bootstrap draws.
The primary metric for both targets is the extractant-macro MAE (MET-01, log10). R² is never ranked across targets. OUT1
(log D reaches τ = 0.4 / 0.5) = T1 (pooled R²) with T1w (within-cell R²); OUT2 (log SF reaches τ) = T2 (B) and T3 (C);
OUT3 (gain over baselines under BP) = intersection–union claims H1 (A vs every log D baseline), H2 (B) and H3 (C) vs
every log SF baseline, with H4 (C vs B) beside; K = 4 Bonferroni, deciding intervals 98.75 % ([BRIEF]:60-63). Expected:
no threshold is reached and H2/H3 do not PASS against G14_ZERO (INCONCLUSIVE, ESTIMAND_INCONSISTENT or BASELINE_BETTER);
a null is a result.

## 1. What exists and is reused

### 1.1 Data, cohort and the counts chain [audit]

| step | count | unit | key or filter | source |
|---|---|---|---|---|
| bundle | 5,992 | raw rows | `dataset with 3D structures/dataset.parquet`, sha256 `fefbefc6…` | `generations/gen13_separation/DATA_AUDIT.md:37` (rows); `gen13sep/paths.py:22` and `dataset with 3D structures/bundle_manifest.json:30` (sha256) |
| quarantine | 5,860 | raw rows | minus 129 TODGA-SMILES/name-mismatch rows and 3 sentinel rows (log D ≤ −6); no overlap | DATA_AUDIT.md:37; `generations/gen13_separation/manifests/cohort_exact.json:72-77` |
| all gen13 cells | 2,395 / 5,233 | cells (any metal count) / (cell, metal) obs | cell = canonical SMILES @@ `publication_id` @@ 64-column condition key | [CNT] |
| multi-metal cells | 521 | cells with ≥ 2 lanthanides | `MIN_METALS_PER_CELL = 2` (gen13sep/cohort.py) | DATA_AUDIT.md:39; [CNT] |
| rows in them | 3,871 | raw rows | rows of the 521 cells | DATA_AUDIT.md:42 |
| averaged | 3,359 | replicate-mean (cell, metal) observations | arithmetic mean per (cell, metal); `nrep__`, `repsd__` (ddof 1) kept | `generations/gen13_separation/analysis/stage2/D5/FINDINGS.md:44`; non-null `logD__*` of `manifests/cohort_exact.parquet`; [CNT] |
| identity | 90 / 45 / 58 | extractants / chemotypes / publications of the 521 cells | chemotype = frozen gen6/gen7 Tanimoto-0.7 single-linkage supercluster | DATA_AUDIT.md:40-41; [CNT] |
| replicates | 216 in 41 cells (728 raw rows) | replicated (cell, metal) entries (nrep ≥ 2) | — | D5/FINDINGS.md:44; `generations/gen13_separation/manifests/cohort_exact.json:101` (216); 728 raw rows [CNT] |
| largest chemotype | 375 / 23 | cells of 521 / extractants of 90 in sc009 (diglycolamides) | — | DATA_AUDIT.md:44 |
| excluded | 1,874 / 1,989 | single-metal gen13-key cells / their raw rows | not in the bench; never used | [CNT] |

**Corrections [audit].** (1) 3,359 is not in DATA_AUDIT.md (its §2 table gives 521 cells and 3,871 rows): cite
D5/FINDINGS.md:44, the parquet and [CNT]; 3,871 − 728 + 216 = 3,359 (728 [CNT]; cohort_exact.json:84-98 gives Σ k·n_k =
3,359). (2) The quarantine removes 0 of its 132 rows from cells with ≥ 2 metals [CNT]; only adding `publication_id` to
the key changes the cohort (3,878 → 3,871 raw rows, 3,364 → 3,359 entries; [CNT]). (3) The 129 rows carry TODGA's SMILES
under a name other than "TODGA" (gen13sep/cohort.py:121): 111 rows of other extractants (PHEN-6OH, PyTri-Tetraol, …)
carrying TODGA's structure, plus 18 legitimate "TODGA,DHOA" mixture rows caught only by the name test
(`runs/gen7_architecture/ambiguity_audit.md:311`). These are the brief's "129 TODGA records": a
TODGA-SMILES/name-mismatch quarantine, not a quarantine of TODGA itself. (4) DATA_AUDIT §3 statistics (89 %/11 %,
variance 2.776, adjacent sd 0.231 with n = 2,115, replicate median 0.237) are pre-quarantine and pre-publication-key
(`generations/gen13_separation/data_audit/gen13_sf_data_audit_part1.py:14,159-161`; for 0.237,
`gen13_sf_data_audit_part2.py:9,63` and `gen13_sf_data_audit_part2.out.txt:54`); never R² context. (5) The fingerprint
drifts on rebuild (`generations/gen20_kshot/AUDIT.md:75`). (6) The cohort still holds the DMDPhPDA copy gen8 quarantined
(`runs/gen8_architecture/case_studies/dmdphpda_source_audit.md:21-23,63-70`). It is the printed Table-1 mantissa without
its column exponent, shifted by a constant +4/+3/+2/+1 decades in the 1–4 M cells and by +1 for La, Ce and Pr in the 5 M
cell. So 59 of its 70 log D entries differ from the curated copy, while its strict-adjacent log SF values equal the
curated copy's except the 5 M Pr/Nd pair (+1.0). The corruption therefore falls on the log D side of the comparison (A,
the `BD_*` baselines, T1, T1w, H1), not on B, C or the log SF baselines; rule: SMILES
`CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1` AND `cond__diluent__ch3cl == 1`, never a name filter; 5 cells, 70
entries, 60 strict-adjacent pairs, all in C3 ([DMD]); kept in the primary only to preserve gen20's 1,802-pair population
and its k = 0 anchors; COH-NODMD refit printed beside every T and H result (O15). (7) `gen20k.data.load_bench` checks
the frame, not `bench.Y` (gen20k/data.py:41-62).

**Cohort assertions [decision].** `gen20k.data.load_bench(check=True)`; `assert_matches_manifest` (column-wise
equality with `manifests/cohort_exact.parquet`, atol 1e-9); NaN-aware `bench.Y` == `bench.frame[logD__La..logD__Lu]`;
bundle sha256 `fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd`; parquet sha256
`3f11ac4ccb7c9f6b1fa768f4406e1f249881cfd76689ba49a9115efc08bb7fd6`; the `bench.pkl` sha256 and fingerprint recorded
(drift from `4c3c6628ea0be949` expected). Failure: STOP.

> **The count of "normal values" mentioned in the brief.** The brief asks that a count of "normal" values not be called
> normal before its origin is checked. That count is not written anywhere in this repository except the brief itself
> (`generations/gen20.1/GEN20_1 INSTRUCTIONS.rtf`, line 30; searched in md/py/txt/json/log/rtf files with comma, space,
> no-break and thin-space separators). The nearest documented populations count different things: (i) **4,881** averaged
> (extractant, 64-column condition, metal) level observations of the gen5/gen6 BASE91 cohort, keyed without publication
> and without the TODGA-SMILES quarantine (`docs/HANDOFF_FOR_CHATGPT.md:178`; mean over replicates 5,564 → 4,881:
> `runs/gen5_levels_20260818T211105Z/summary.json:10,15`; key (extractant, condition_id from cond__* only, metal):
> `runs/gen7_architecture/ambiguity_audit.md:307`; TODGA quarantine never applied:
> `runs/gen7_architecture/ambiguity_audit.md:310`); (ii) **4,746** rows of the external
> lanthanidestrain table whose 3D-geometry QC class is "OK". This is a geometry flag, not a log D quality flag, and the
> count cannot be reproduced here because that repository is not on this machine
> (`generations/gen20_kshot/IMPLEMENTATION_PLAN.md:120`); (iii) **4,691** gen7 level rows in the gen10 "A consistent"
> data-quality stratum (`runs/gen10_final/GEN10_ERROR_CEILING.md:72`). None of these is the gen20.1 population. None is
> defined on the gen13 cell key with publication, and none is described here as "normal". gen20.1 uses the frozen gen13
> exact cohort: 5,992 raw bundle rows → 5,860 after the 129-row TODGA-SMILES/name-mismatch quarantine and 3 sentinel
> rows (log D ≤ −6) → 521 cells (extractant SMILES × publication × 64-column condition key) with at least two
> lanthanides, holding 3,871 raw rows averaged into 3,359 replicate-mean (cell, metal) observations
> (`generations/gen13_separation/DATA_AUDIT.md:37-42`; `generations/gen13_separation/analysis/stage2/D5/FINDINGS.md:44`;
> `generations/gen13_separation/manifests/cohort_exact.parquet`). The primary scored subset is the 229 C3 cells: 2,425
> (cell, metal) observations and 1,802 strict-adjacent pairs per seed. Open item O1: where did the brief's figure come
> from, and what does it count?

### 1.2 Folds and seeds [audit]

Designs A/B/BR/BQ/BP: `gen13sep/splits.py`; BPG (BP plus the gen19 copy-group mask): gen20k/splits.py:1-12; `test_index`
is never masked, so each cell is tested once per seed in every design ([CNT]). Split seeds
104729, 130363, 155921, 196613, 262147; `fold.model_seed` = 42 + fold·1009 + 9,999,991 (gen13sep/splits.py:24-29,
58-60); inner seed seed·31 + fold·7919 + 17 (gen20k/splits.py:31-33). Digests: test-index plan `7046c640…`
(GEN20_REPORT.md:102), train+test `9a24b6e9…`, BP = B = BR = BQ `b50c9fd6…`, A `afda7475…`
(`generations/gen16_leads/results/anchors/anchors.json:31-40`). All 25 BP (seed, fold, n_train, n_test) match
`generations/gen20_kshot/predictions/discovery/folds_BP.json`, and design B/BR/BQ/BPG test sets equal design BP's
(design A's differ) ([CNT]). gen20 stored no cell-level plan; its confirmation seeds were spent on 2026-10-14
(`generations/gen20_kshot/results/confirmation/RUN_ONCE.json:2`).

### 1.3 gen20 reference numbers (cited, never recomputed as gen20.1 results) [audit]

**The motivating 0.31** ([BRIEF]:17). `RIDGE2|k3`, levels csv:73: seed-mean pooled R² 0.30991136191292135 (per seed
0.30270 / 0.33488 / 0.29426 / 0.31309 / 0.30462), 95 % chemotype percentile CI [0.2221, 0.4006], MAE 0.1206, RMSE
0.1844, extractant-macro MAE 0.1346. Configuration: arm RIDGE2 (radius curve shrunk to the training-fold G14 prior, λ
from {0.3, 1, 3, 10} on inner folds; a Model-3 baseline, exploratory even in gen20,
`generations/gen20_kshot/PRE_REGISTRATION.md:87,146`); metal measurement model, k = 3 (4 read metals of the test cell, 3
revealed differences; GEN20_REPORT.md:51; gen20k/design.py:423-428); policy DOPT_Q; design BP; cohort C3; regime Q;
1,802 pairs per seed in 229 cells, 79 extractants, 38 chemotypes, 39 publications; Kish 3.851; y_sd0 0.22196 (levels
csv:73); confirmation 0.321 (`generations/gen20_kshot/results/confirmation/levels_BP_metal_C3_DOPT_Q.csv:73`;
GEN20_REPORT.md:335). In gen20.1 it is `CTX_RIDGE2_K3`: measured-mode context only, never a comparator or threshold
reference, not re-run unless O4.

**k = 0 anchors on the identical 1,802 C3 pairs per seed** (FLAT, G14, MEAN_CURVE and NN_TANIMOTO reproduced at S7;
NNC@ZERO cited only, since NNC is never instantiated, §8; gen20's "y sd 0.236", GEN20_REPORT.md:163,
is this FLAT RMSE, i.e. y_rms0; y_sd0 is 0.22196, levels csv:73; gen20 never scored absolute log D: it revealed and
scored within-cell differences only, `generations/gen20_kshot/gen20k/design.py:423-428`):

| arm | R² (seed mean) | MAE pooled | MAE extractant-macro | RMSE | source |
|---|---|---|---|---|---|
| FLAT@ZERO | −0.13471207313225375 | 0.16425914006377063 | 0.1730711262478135 | 0.23644 (= y_rms0) | levels csv:6 |
| G14@ZERO | 0.05287231149042522 | 0.1468610842819455 | 0.16124988000838458 | 0.21601 | levels csv:18 |
| MEAN_CURVE@ZERO | −0.11561984257982724 | 0.16597 | — | 0.23444 | levels csv:30 |
| NNC@ZERO | −0.3217936240551599 | 0.17662 | — | 0.25514 | levels csv:50 |
| NN_TANIMOTO@ZERO | −0.3860763898872736 | 0.18083 | — | 0.26128 | levels csv:62 |

**Representative gen20 values ≥ 0.35, grouped by the task change behind them [decision; TAB-14].** The list is not
exhaustive: 1,220 rows of `results/*/levels_*.csv` have R² ≥ 0.35, maximum 0.744 (design B, pair model C10,
DOPT_CURVE, k = 10). Every one is measured mode (k ≥ 1) and comes from one of these changes: Nd–Sm added, a single pair
type, the pair measurement model (C3, C5 or C10, any design), or the metal model at C5/C6 under a non-DOPT_Q policy or a
design other than BP; none is design BP / metal model / DOPT_Q / strict-adjacent. None is a strict-task zero-shot R²,
and none is a target for gen20.1. Paths are under `generations/gen20_kshot/`.

| value | arm and setting | modification relative to the strict C3 k = 3 task | source |
|---|---|---|---|
| 0.4293; 0.39 / 0.38 / 0.35; 0.66 | RIDGE2; G14@BLUP20 / LINE / FLAT@BLUP20; the Nd/Sm type alone; k = 3, C3 | Nd–Sm added (gen13 adjacency; 1,984 pairs, 236 cells; target sd 0.2523) | GEN20_REPORT.md:184-185; `results/discovery_nd_sm/levels_BP_metal_C3_DOPT_Q.csv:73` |
| 0.42 / 0.38 / 0.36; 0.404 / 0.358 / 0.403; 0.31–0.42 | LINE Pr/Nd / LINE Sm/Eu / G14@BLUP20 Gd/Tb; RIDGE2 Sm/Eu / Tb/Dy / Dy/Ho; Pr/Nd, several arms; k = 3 | single pair type (e.g. 127 / 158 / 165 rows) | GEN20_REPORT.md:238,239,241,377; `results/discovery/per_pair_BP_metal_C3_DOPT_Q_RIDGE2_k3.csv:5,8,9` |
| 0.43 / 0.39 (k = 5); 0.66 / 0.63 (k = 10) | G14@BLUP20 / FLAT@BLUP20, C10 | pair measurement model with lab-known interpolation (114 cells) | GEN20_REPORT.md:297-299 |
| 0.39–0.46; 0.41–0.42; 0.39–0.43 | G14@BLUP20; LINE; RIDGE2; k = 3–10 | pair model C10, not-lab-known rows (215.2 rows, 97.4 cells per seed; y_sd 0.1997) | GEN20_REPORT.md:303; `results/discovery/levels_BP_pair_C10_DOPT_Q_not_lab_known.csv:141` |
| 0.471 | G14@BLUP20, k = 5 | pair measurement model, C5 (rows include lab-known) | `results/discovery/levels_BP_pair_C5_DOPT_Q.csv:16` |
| 0.354 / 0.364; 0.360 / 0.370 | G14@BLUP20 / G14@BLUP_INDEP, k = 3 | pair measurement model on C3 under BP (WIDEST policy, 275 cells, 1,980 pairs; DOPT_Q without sc009, 113 cells, 972 pairs) | `results/discovery/levels_BP_pair_C3_WIDEST.csv:13,17`; `results/discovery/levels_BP_pair_C3_DOPT_Q_without_sc009.csv:13,17` |
| 0.405; 0.405; 0.410; up to 0.436 | G14@BLUP_INDEP / G14@BLUP20, k = 3 | pair measurement model on C3 under designs B, A, BQ (1,980 pairs; 0.436 is A without sc009, 972 pairs) | `results/discovery/levels_B_pair_C3_DOPT_Q.csv:17`; `results/discovery_light/levels_A_pair_C3_DOPT_Q.csv:13`; `results/discovery_light/levels_BQ_pair_C3_DOPT_Q.csv:17`; `results/discovery_light/levels_A_pair_C3_DOPT_Q_without_sc009.csv:13` |
| 0.370 | G14@BLUP20, k = 6 | cohort C6, policy UNIFORM | `results/discovery/levels_BP_metal_C6_UNIFORM.csv:19` |
| 0.416 | G14@BLUP20, k = 6, C6, UNIFORM | design B (no publication mask) | `results/discovery/levels_B_metal_C6_UNIFORM.csv:19` |
| 0.381; 0.391; 0.408 | G14@BLUP20, k = 6, C6 | design B (DOPT_Q); design A (UNIFORM); design BQ (RANDOM) | `results/discovery/levels_B_metal_C6_DOPT_Q.csv:19`; `results/discovery_light/levels_A_metal_C6_UNIFORM.csv:19`; `results/discovery_light/levels_BQ_metal_C6_RANDOM.csv:19` |
| +0.426 / +0.43 | RIDGE2 ΔR²(k = 3 − k = 0) | a change in R², not an R² | GEN20_REPORT.md:154,335 |
| 0.401 | RIDGE2\|k3 CI upper bound | an interval bound, not attainment | levels csv:73 |

The best strict registered-setting value is RIDGE2 C6 k = 5 at 0.326
(`results/discovery/levels_BP_metal_C6_DOPT_Q.csv:108`); across the six designs RIDGE2 at k = 3 (measured mode) spans
0.310–0.325 (GEN20_REPORT.md:272).

### 1.4 Code that exists, and what does not [audit]

No absolute out-of-fold log D prediction exists: gen13 `DirectRowArm` centres its output (gen13sep/models.py:269) and
every gen20 arm predicts a centred curve or contrasts (`generations/gen20_kshot/gen20k/arms.py:6`: arms return centred
curves), so arm A is new code. `CentredRowArm` (gen13sep/arms_stage2.py:99-155) has BP results
(`generations/gen13_separation/metrics/BP_v2/leaderboard.csv:7`); `AntisymmetricPairArm` (models.py:272-316) was planned
"at reduced cost ... where time allows" (`generations/gen13_separation/PRE_REGISTRATION.md:82-83`) but never run: no
gen13 run artifact or metrics table names it (computed: `grep -rl C_PAIR_ANTISYM generations/gen13_separation` lists
only code and that plan entry); the repository has no log D analogue of FLAT (repository search, 2026-10-15), but
training-only log D baselines exist on other row populations: gen19's B0 global mean, B1 metal-state mean and B2
extractant mean (`generations/gen19_chem_transfer/gen19ct/models/baselines.py:9-12`), and gen18's per-metal training
mean and same-metal 1-NN (`generations/gen18_process/DESIGN.md:1115-1116`); BD_MEAN and BD_METAL are the analogues of
gen19's B0 and B1; gen20's per-row predictions are absent (`generations/gen20_kshot/.gitignore:1`), so gen20 references
are refitted in fold.

## 2. Data and units of analysis

**Units.** A *log D entry* (variant A) is one observed (cell, metal) with y = the frozen replicate mean of log10 D
(La..Lu; Pm absent, never imputed). A *log SF pair* (B, C) is a `Q_adj` row, a strict-adjacent pair (|ΔZ| = 1, 12 types)
inside one cell, lighter first: y(a/b) = log10 D(a) − log10 D(b) (gen20k/data.py:88-92), so heavier-preferred ⇔ y < 0
(heavier-first sources are converted). A *training row* is any observed (cell, metal) of the masked outer-training cells
(COH-TRAIN, non-C3 cells included), identical for A, C and every RowForest baseline. Clusters: chemotype (deciding) and
publication (co-unit); the extractant is the macro unit; seeds are not resampled. **C3** (gen20's cohort,
gen20k/report.py:69): cells with a non-empty strict-adjacent query and metal-model k_max ≥ 3 under
`gen20k.design.make_design(cell_view(bench, ci), 'adjacent', 'metal')`. At k = 0, "k_max ≥ 3" has no measurement
meaning; it only restricts scoring to metal-rich cells (5–14 measured metals each, [CNT]). It is primary for continuity
with gen20's headline metal-model k = 0 anchors (levels csv:6,18,30,50,62). gen20 also has k = 0 anchors on the
pair-model C3 cohort (275 cells, 1,980 strict-adjacent pairs;
`generations/gen20_kshot/results/discovery/levels_BP_pair_C3_DOPT_Q.csv:6,18`) and on the metal-model C5/C6 subsets. It
has none on COH-SF-ANY or COH-SF-C1, and none on absolute log D (O13, O14).

**Populations** (label-free except COH-REPCAP, seed-invariant, fixed before any fit; re-verified at S3, mismatch =
STOP):

| ID | definition | unit | role | count (source) |
|---|---|---|---|---|
| COH-FROZEN | gen13 exact cohort | cells / obs / raw rows | universe for folds and training | 521 / 3,359 / 3,871; 90 extractants, 45 chemotypes, 58 publications; 216 replicated entries (§1.1) |
| COH-TRAIN | observed entries of outer-training cells after the mask | (cell, metal) training row | training | design BP 76–492 cells and 725–3,084 rows per fold; design B 103–507 cells; design BR 76–492; design BQ 76–492; design BPG 73–492; design A 241–494 ([CNT]) |
| COH-INNER | 4-fold inner chemotype cross-fit of each training fold, mask per inner fold | cell | selection | masked inner-train ≥ 18 cells under BP (1 of 100 inner folds < 20, skipped); design B minimum 59 ([INNER]) |
| COH-D-C3 | every observed entry of the 229 C3 cells | (cell, metal) obs | **primary log D** (T1, T1w, H1) | 2,425 per seed; 229 cells, 79 extractants, 38 chemotypes, 39 publications; Kish (rows) 3.430; largest chemotype share 0.524; 161 replicated entries; 5–14 metals per cell ([CNT]) |
| COH-SF-C3 | `Q_adj` pairs of the C3 cells | within-cell pair | **primary log SF** (T2, T3, H2–H4) | 1,802 per seed; Kish (pairs) 3.8511214634053776; largest share 0.489 ([CNT]; levels csv:73; GEN20_REPORT.md:116) |
| COH-D-C3M | COH-D-C3 entries whose metal is in ≥ 1 scored `Q_adj` pair of its cell; centring over these rows | obs | secondary, matched to the log SF rows | 2,208 ([CNT]) |
| COH-D-ALL | all entries of the 521 cells | obs | secondary; cannot fulfil OUT1 | 3,359; Kish rows 2.734 ([CNT]) |
| COH-SF-ANY; COH-SF-C1 | strict-adjacent pairs of cells with ≥ 1 such pair; of cells with k_max ≥ 1 | pair | secondary | 2,114 pairs, 378 cells, 82 extractants, 40 chemotypes, 45 publications, 2,980 entries, Kish pairs 3.296; 2,067 pairs, 331 cells, 82 extractants, 40 chemotypes, 44 publications, 2,886 entries ([CNT]; `generations/gen20_kshot/AUDIT.md:45`; GEN20_REPORT.md:114) |
| COH-SF-NDSM | gen13 adjacency (strict + Nd–Sm) on the cells whose `'adjacent_gen13'` query is non-empty with metal-model k_max ≥ 3 under it (not the strict C3 cells) | pair | labelled pass; never pooled with strict numbers | 1,984 pairs, 236 cells, 79 extractants, 39 chemotypes, 39 publications; 2,466 entries ([CNT]; GEN20_REPORT.md:182) |
| COH-SC009; COH-NOSC009 | C3 restricted to sc009; C3 minus sc009 | obs / pairs | breakdowns (sc009: chemotype interval NA, "one cluster"; no-sc009: own draw matrices) | 129 cells, 23 extractants, 15 publications, 1,270 entries, 882 pairs; 100 cells, 56 extractants, 37 chemotypes, 29 publications, 1,155 entries, 920 pairs, Kish rows 13.14, pairs 12.97 ([CNT]) |
| COH-NODMD | every population minus the 5 DMDPhPDA cells (refit). COH-NODMD reuses the primary Fold objects of every design: the 5 cells are removed from train_index, from every masked inner-train and inner-validation set, and from the scored-row manifests; folds_for is never called on a reduced frame and the BR/BQ masks are not recomputed | as parent | sensitivity | C3: 224 cells, 2,355 entries, 1,742 pairs; cohort 516 cells, 3,289 entries (arithmetic on [CNT], [DMD]) |
| COH-REPCAP | scoring without entries with nrep ≥ 2 and repsd > 0.5, and pairs containing one | as parent | sensitivity (scoring only) | definition sealed; row manifest and count produced at S5 (after the seal) from `repsd__*`, labelled label-derived |
| COH-REP; COH-REPPAIR | replicated entries of outer-training cells per (seed, fold); run-paired replicated adjacent pairs recovered read-only from the bundle via `safe_exp_ids` | obs; pair | noise (§7) | cohort 216 in 41 cells (D5/FINDINGS.md:44), C3 161 ([CNT]), per fold at S3; COH-REPPAIR feasibility at S3, possibly 0 (O16) |
| COH-AUG | single-metal cells as extra training rows | cell | NOT RUN by default (O23) | 1,874 cells / 1,989 raw rows before masking ([CNT]) |

**Row manifests.** Written at S3 (COH-REPCAP's at S5): log D rows keyed (design, split_seed, fold, cell_id, metal),
log SF rows keyed (design, split_seed, fold, cell_id, a, b). Every arm must cover exactly these keys (set equality and
`n_pred_finite == n_rows`), or the run fails.

**Features, leak guard, weights.** Cell features only from `bench.frames` via `bench.matrix(blocks)`; default
LEAN209 = COND64 + MASSACT8 + PHYSCHEM10 + DONORS13 + COORD114; metal descriptors METAL_BASIC2 = (Z, CN8 Shannon
radius), target-free (`gen13sep.metals`). Every column list passes `gen13sep.features._assert_allowed`
(features.py:68-71) and a deny list (`n_metals`, `n_rows`, `n_replicated_metals`, `replicate_sd_median`, `metals`,
`experiment_series_ids`, `safe_exp_ids`, `recipe__*`). The only test-cell input is its feature row; measured metals only
choose what is scored. Per fit, w_c = `chemotype_balanced_weights` of the fit's training cells, row weight w_c /
n_obs(c) renormalised to mean 1 (models.py:257-261), shared by A, C, BD_COND, BD_LEVEL, A_PRIME and the
BD_MEAN/BD_METAL/BD_NN means; C_PAIR uses w_c / n_pairs(c) (models.py:304-307).

**Variance structure.** Every source agrees the between-cell level dominates log D variance (gen13 audit ratio
89/11 on mixed populations; gen6 92/8 on another cohort); the share on the gen20.1 rows is unmeasured
(DATA_AUDIT.md:63-64; `docs/results/gen6_phase2_results_20260819.md:60`). π = s_w (MET-09) is measured at S5; with
R²_within ≤ 0 a pooled τ needs R²_level ≥ τ/(1 − π); no number is plugged in.

## 3. Variants, arms and baselines

**Modes.** Mode Z (k = 0) is the only registered mode: no label of any outer-test cell (no log D value, contrast or
replicate statistic) enters any arm, baseline, inner selection, preprocessing, noise estimate or covariance. Mode M (k
= 3, gen20 metal model, DOPT_Q supports) is OFF (O4): a read metal reveals the test cell's absolute level, which raises
A's log D R² mechanically and leaves B unchanged on pairs without a read endpoint.

**Learner and grid (O9 default, fixed at S4).** gen13 `tree_pipeline` (SimpleImputer(median,
keep_empty_features = True) → ExtraTreesRegressor(400 trees, random_state = fold.model_seed), imputer fitted on training
rows; models.py:36-42) on LEAN209 + METAL_BASIC2, four configurations in simplicity order (ties |Δ| < 1e-12 → lowest K):
K1 (min_samples_leaf 8, max_features 0.3), K2 (8, 0.5), K3 (2, 0.3), K4 (2, 0.5; gen13 default, raw K4 ≡ C_DIRECT_ROW on
LEAN before centring, gate LT-13); one grid for A, C and BD_COND. O9 alternative: a feature-set grid (LEAN +
METAL_RICH7; LEAN + basic; ALL7 + basic; LEAN + rich7 leaf 8), whose gen13 results on these folds are known (as are
those of the default LEAN209 itself; O26) and whose ALL-block fits cost 15–27 s each ([TIM]). Rejected: neural or
CatBoost learners, more than 4 configurations.

**Registered arms.** **A**: RowForest(target = raw) on COH-TRAIN rows; per outer fold, nested, no vote across folds:
K1–K4 fitted on each inner fold, criterion MET-01 of log D over the pooled inner OOF rows (every observed metal of every
inner-validation cell), argmin (ties → lowest K) refit on the whole outer training fold; all four configs are also
refit for the "not selectable" appendix; the per-fold record keeps the 4 inner scores, inner RMSE (unused), inner cells
and rows, skipped inner folds. **B**: for every `Q_adj` row (c, a, b) of fold f, Â_f[c,a] − Â_f[c,b] from A's selected
model of that fold; no re-selection, recalibration, shrinkage or offset ([BRIEF]:44). **C**: RowForest(target =
centred), the same learner, rows, X, weights, random_state and grid as A, ỹ = y − mean over the training cell's observed
metals; config chosen per outer fold by inner MET-20; ĉ for all 14 metals, pair = ĉ[c,a] − ĉ[c,b]; C differs from B in
target representation and in selection criterion (inner MET-20 vs A's inner log D MET-01), so its per-fold
configuration may differ; C_ACFG (EX-04) is the target-only contrast (O10). If the centred learner fails its smoke
test, C is "not run" and H3/H4 are `NA`; no substitute is chosen ([BRIEF]:45).

**Exploratory and diagnostic arms** (no extra fits unless stated): B_selSF / C_selSF, raw / centred RowForest with the
config chosen by inner strict-adjacent log SF MET-01 (EX-01..EX-03); C_ACFG, centred RowForest with A's config
(target-only difference from B; EX-04); C_PAIR, gen13 `AntisymmetricPairArm` (models.py:272-316) on LEAN209 (200 trees,
max_features 0.3, min_samples_leaf 2, all training within-cell pairs + reversals with no subsampling (max_pairs_per_cell
None), prediction (f(a,b) − f(b,a))/2), designs BP and B only, after a passing `smoke*` test (EX-05/06, O11); A_K1..K4,
B_K1..K4, C_K1..K4, each config fixed in every fold (TAB-13, never selectable); A_PRIME, A's config with random_state +
1, BP only (NC1, EX-13); BD_METAL_SF, BD_COND_SF, BD_NN_SF, BD_LEVEL_CURVE_SF, in-cell differences of those log D
baselines (BD_LEVEL_SF ≡ 0 ≡ FLAT; EX-12, EX-14); MODE_M_A / MODE_M_B OFF, switched on only by the user's S2 answer to
O4 (entered in the drafts before the seal) or, after the seal, by a dated addendum before any run (EX-15). EX-09: the
H1–H4 contrasts on COH-D-C3M, COH-D-ALL, COH-SF-ANY, COH-SF-C1, COH-SF-NDSM and COH-NOSC009; EX-10: the same under
COH-NODMD and COH-REPCAP (both TAB-09).

**Baselines** ([BRIEF]:53): fitted on outer-training rows only (and inner folds where selection needs them),
scored on the candidate's rows; fallback counts printed, > 50 % fallbacks labelled. Rejected: gen19 extractant lookups
(undefined for held-out chemotypes), the test mean (recovery plan :165), frozen gen20 predictions (absent).

| ID | target | definition | role |
|---|---|---|---|
| BD_MEAN | log D | weighted mean of training log D entries (A's row weights) | H1 IUT member |
| BD_METAL | log D | per-metal weighted training mean; unseen metal → BD_MEAN | H1 IUT member |
| BD_COND | log D | RowForest(raw) on COND64 + MASSACT8 + METAL_BASIC2, K1–K4 nested by inner MET-01; K4 ≡ gen13 `V2_DIRECT@cond` (all-pairs centred macro MAE 0.624578611404633, `generations/gen13_separation/metrics/BP_v2/leaderboard.csv:11`) | H1 IUT member |
| BD_NN | log D | nearest training extractant(s) by ECFP Tanimoto (`bench.frames["ECFP"] > 0.5`, gen20 `NNTanimoto`, ties averaged; gen20k/arms.py:45-76): its weighted mean log D at m, else its mean offset from BD_METAL + BD_METAL(m); conditions ignored | H1 IUT member |
| BD_LEVEL | log D | A's per-fold config refit without metal columns; constant within a cell | level-attribution diagnostic; EX-08 |
| BD_LEVEL_CURVE | log D | BD_LEVEL(c) + MC(m), MC = training chemotype-balanced mean centred curve (MeanCurveArm logic) | diagnostic; EX-07 |
| BL_D_INNER | log D | per fold, the H1 member with the lowest inner MET-01; BD_MEAN, BD_METAL and BD_NN are refit on each masked inner-train set; BD_COND's inner score is the pooled inner-OOF MET-01 of the configuration it selects in that outer fold (optimistic best-of-four, stated) | EX-11 |
| FLAT | log SF | ŷ = 0 | H2/H3 IUT member; anchor |
| PAIRMEAN | log SF | gen13 `PairMeanArm` (chemotype-balanced) at strict-adjacent types; absent type → chemotype-balanced mean of all strict-adjacent training pairs | H2/H3 IUT member |
| G14_ZERO | log SF | gen20 G14 refit per outer fold: `cov.make_ctx(bench, fold, train, test)`, `arms.standard_arms()['G14']().fit(ctx).curves(test)`, pair = c[a] − c[b] | H2/H3 IUT member (O8); anchor |
| MEAN_CURVE_REF, NN_TANIMOTO_REF; HEAVIER_ALWAYS; BL_SF_INNER; CTX_RIDGE2_K3 | log SF / sign | gen20 arms refit in fold; gen13 `HeavierAlwaysArm` (y < 0 always, PAIRMEAN sign beside); per fold FLAT or PAIRMEAN by inner log SF MET-01; not run (O4, §1.3) | anchors/references only, never gain references; MET-14 yardstick; EX-11; context only (TAB-14) |

## 4. Splits, modes and leakage protection

**Designs.** For each design, `gen20k.splits.folds_for(bench.frame, design, seeds=SPLIT_SEEDS)` is called
exactly once (it runs `assert_design_integrity`); one Fold object per (design, seed, fold) goes to every arm, baseline
and reference; no arm builds its own split ([BRIEF]:47).

| design | outer grouping | training mask | role |
|---|---|---|---|
| BP | chemotype, 5-fold | drop training cells whose publication appears among the held-out cells | selects; carries every claim; headlined |
| B | chemotype | none | sensitivity; "publication-leak price" R2_pooled(A) under B minus under BP printed |
| BR | chemotype | BP-sized random cell drop | sensitivity |
| BQ | chemotype | whole random publications dropped, matched | sensitivity |
| BPG | chemotype | BP plus the gen19 copy-group mask | sensitivity |
| A | extractant | none | sensitivity; paired per (seed, cell, metal/pair), never per fold; never headlined |

All six designs run the full arm set with their own nested selection on the same scored rows; a PASS needs the same
point sign under designs B, BR, BQ, BPG and A (RULE-C (h)); a design not run gives `INSUFFICIENT_COMPUTE` (a label new
in gen20.1) when RULE-C (a)–(g) hold on design BP, otherwise that (h) counts as failed (`PRE_REGISTRATION.md` §8.1)
(sign rule: `generations/gen13_separation/PRE_REGISTRATION.md:175,189-193`).

**Fold gates** (S3 and every outcome-bearing script; failure = STOP `INCOMPARABLE_FOLDS`; folds are never regenerated
differently): (a) digests `7046c640…`, `9a24b6e9…`; (b) the §1.2 per-design digests, re-implemented in gen201 (never
imported from gen16: two packages share that name) and computed in a subprocess under PYTHONHASHSEED 1 and 2; (c) 25
BP (seed, fold, n_train, n_test) equal gen20 `folds_BP.json`; (d) design B/BR/BQ/BPG test sets equal design BP's; (e) a
new digest over (design, seed, fold, sorted test, sorted train, and the masked (inner-train, inner-validation) index
pairs of all four chemotype inner folds, skipped folds included and flagged) for six designs in
`manifests/fold_plan_g201.json`. `splits.fold_record()` is persisted for every design. `g201_confirm.py` re-asserts
(a)–(d) on the discovery seeds (code-path check) and asserts a confirmation outer+inner fold digest (form of (e)),
computed label-free and committed with the confirmation seeds at S4.

**Inner selection (deviation from gen20, O12).** `seeded_group_kfold(chemotype[train_f], 4, inner_seed(seed, fold))`,
the split of `gen20k.cov.inner_oof_curves` (cov.py:75-90), whose first fold equals `fold.inner_train_index` /
`inner_validation_index` (before `inner_mask`) for designs BP, B, BR, BQ and BPG (25 of 25 folds each; [INNER], asserted
at S3) but not for design A (0 of 25), whose stored gen13 inner split groups by extractant
(gen13sep/splits.py:30,105-107) and is never used; `gen20k.cov.inner_mask` re-applies the publication mask (BPG: copy
groups) inside each inner fold ([BRIEF]:47); chemotype grouping for all designs, design A included (a deviation from
design A's stored inner split, §10 row 25). An inner fold with < 20 masked inner-training cells is skipped and logged
(cov.py:84): under BP 1 of 100 (seed 155921, fold 2, inner fold 3, 18 cells), under B none (minimum 59) ([INNER]); all
four skipped → K1, logged. Rejected: gen20's single split `select_on_inner` (runner.py:100-104; 10–11 validation cells
in 4 BP folds, [CNT]); row-level random inner CV.

**The giant BP fold** (one per seed; holds out sc009; each predicts 52–64 % of its seed's 1,802 C3 pairs; [CNT]):
(seed, fold, training cells, training log D rows, test cells, test C3 cells, test C3 pairs) = (104729, 0, 76, 764,
418, 148, 1,070), (130363, 3, 98, 931, 394, 142, 989), (155921, 3, 82, 725, 415, 158, 1,150), (196613, 1, 85, 803,
400, 151, 1,102), (262147, 2, 85, 919, 400, 136, 935).

**What may never see a test label.** Arms, baselines, references, inner selection, imputers, weights, noise estimates
and G14's fold context read training (or masked inner-training) cells only ([BRIEF]:47); row manifests (except
COH-REPCAP's label-derived S5 manifest) and draw matrices are label-free; per-config outer scores are never selectable;
smoke computes no outer metric; S5 statistics come after the seal; frozen trees are write-protected (§8). No learner
uses early stopping: the tree fits have fixed tree counts (400; C_PAIR 200), and gen20's NNC, the only gen20k learner
that early-stops (`generations/gen20_kshot/gen20k/nnc.py:119`; computed: `grep -rn -i "early.stop\|patience"` over
`gen20k/*.py`), is not used, so the brief's requirement that early stopping stay inside the training fold ([BRIEF]:47)
holds trivially.

**Tests** in `generations/gen20_1/tests/` (conftest inserts `generations/gen20_1` and `generations/gen20_kshot`; run
from the repo root with `-p no:cacheprovider`); a failing leakage test invalidates the affected experiment. Mirrored
from gen20's 22 + 8 test functions (35 collected tests, GEN20_REPORT.md:42;
`generations/gen20_kshot/tests/test_leakage.py:47-399`, `tests/test_leakage_injection.py:23-99`): **LT-01** no forbidden
overlap in six designs, each cell tested once per seed; **LT-02** inner mask removes held-out publications (BP) and copy
groups (BPG); **LT-03** selection touches inner cells only; **LT-04** preprocessing sees training rows only; **LT-05**
subprocess reproducibility under PYTHONHASHSEED 1 and 2; **LT-06** orientation, antisymmetry, unmeasured metals never
scored; **LT-07** strict adjacency excludes Nd–Sm; **LT-08** R² NA cases, direction tie rule; **LT-09** cohort equals
the manifest (521 cells); **LT-10** injections (test cell in training, publication crossing, inner-validation cell in
inner training, target-derived feature). New: **LT-11** masked-label invariance: for all outer-test cells of the fold,
set `bench.Y`, the `bench.coef` rows and the `bench.frame` columns `logD__*`, `nrep__*`, `repsd__*`,
`n_replicated_metals` and `replicate_sd_median` to NaN, keeping `cell_view`, `n_metals` and `bench.n_obs` from the
unmasked bench; every A/B/C/`BD_*`/PAIRMEAN/G14_ZERO/reference prediction and every selection record must be unchanged.
In addition, spy on `cov.make_ctx`, `FitContext` and every `RowForest.fit` and assert that each training index is a
subset of the masked train_index or of the masked inner-train set (as gen20 test_leakage.py:239); **LT-12** A/C training
rows are exactly the finite Y[c, m] of the masked train_index; **LT-13** continuity hard gate (RowForest raw K4,
centred, equals in-process gen13 DirectRowArm on LEAN; centred K4, its 14-metal output centred per cell (as
arms_stage2.py CentredRowArm.predict_curves does), equals CentredRowArm(metal_features='basic', aggregate='mean',
weights='balanced') on LEAN; 1e-12 on 2 BP folds); **LT-14** A and C get byte-identical X, weights and random_state;
**LT-15** B equals the A-table difference with A's config; **LT-16** coverage of the S3 row manifests,
`n_pred_finite == n_rows`; **LT-17** identical keys and y across arms, identical chemotype and publication lists for COH-D-C3 and
COH-SF-C3; **LT-18** inner fits see masked inner-training cells only; **LT-19** S5 noise reads training cells only;
**LT-20** Nd/Pr ↔ Pr/Nd flip invariance; **LT-21** I1–I6 on synthetic arrays (only in tests/); **LT-22** MET-04/07
weights sum to 1 per subset and draw; **LT-23** RULE-T/RULE-C boundaries (values equal to τ or δ, NA draws, verdict
order); **LT-24** |R2_within_cell| ≤ 1e-12 for BD_MEAN and BD_LEVEL, and max − min of BD_LEVEL's prediction within each
cell = 0 exactly; **LT-25** DMDPhPDA rule = 5 cells, 70 entries, 60 pairs, absent from COH-NODMD, and the COH-NODMD
train sets equal the primary train sets minus the 5 cells in all 150 (design, seed, fold) units; **LT-26** frozen trees
untouched (forbidden writers, including `gen13sep.cohort.write_cohort`, monkeypatched to raise; gen13/14/15/16/20 files
snapshotted); **LT-27** gen13 output directories exist before `gen13sep.paths` import, missing
`generations/gen14_direction/cache/bench.pkl` fails fast; **LT-28** fold digests (a)–(e); **LT-29** report refuses
`smoke*`; **LT-30** injections that must be caught: a test row in A's training rows, a feature named `n_metals`,
`replicate_sd_median` or `logD__*`, a dummy arm using the test-cell mean log D, a baseline fitted with test rows, a
reference reading `bench.coef` of a test cell, a dropped scored row.

## 5. Metrics and results tables

**Notation.** i indexes scored rows (log D: (cell, metal) entries; log SF: `Q_adj` pairs), n rows; y_i the
replicate-mean target, ŷ_{i,s} the prediction in split seed s, e = y − ŷ; l extractants (L; n_l rows each), c cells (n_c
scored rows), g chemotypes; P_l = number of publications of extractant l among the scored rows; C_lp = number of cells
of l in publication p (a gen20.1 cell is the recovery plan's 'series'); N_lpc = number of scored rows (log D entries or
log SF pairs) of cell c; all recomputed per subset and per bootstrap draw. Rows are seed-invariant, so y, n and SST do
not depend on s. Every statistic is computed per seed; the headline is the mean over 5 seeds with per-seed values
printed ([BRIEF]:50-58).

| ID | definition | role |
|---|---|---|
| MET-01 | MAE_macro_ext,s = (1/L)·Σ_l (1/n_l)·Σ_{i∈l} abs(e_{i,s}); inference unit MAE_l = mean_s MAE_{l,s}, Δ_l = MAE_l(Y) − MAE_l(X) | **PRIMARY, both targets**; A's selection; H1–H4 |
| MET-02, MET-03, MET-05 | MAE_pooled = (1/n)·Σ abs(e); MAE_macro_chem = (1/G)·Σ_g (1/L_g)·Σ_{l∈g} MAE_l (gen20k/metrics.py:72-73); RMSE_pooled = sqrt((1/n)·Σ e²) | secondary |
| MET-04 | MAE_bal = Σ w_i abs(e_i), RMSE_bal = sqrt(Σ w_i e_i²), Bias_bal = Σ w_i (ŷ_i − y_i); w_i = 1/(L·P_l·C_lp·N_lpc), Σw = 1 (recovery plan :123-141) | secondary; sign required in RULE-C (i) |
| MET-06 | R2_pooled,s = 1 − Σ e_{i,s}² / Σ (y_i − ȳ)², ȳ the unweighted mean of the scored rows; seed mean by I4 | decides T1, T2, T3 |
| MET-07 | R2_balanced = 1 − Σ w_i e_i² / Σ w_i (y_i − ȳ_w)² (MET-04 weights); within-cell variant 1 − Σ w_i (ỹ_i − ŷ̃_i)² / Σ w_i ỹ_i² | RULE-T (vi); RULE-C (j) |
| MET-08 | R2_within_cell = 1 − Σ_c Σ_{i∈c} (ỹ_i − ŷ̃_i)² / Σ_c Σ_{i∈c} ỹ_i², ỹ_i = y_i − ȳ_c, ŷ̃_i = ŷ_i − mean_c(ŷ) over the cell's scored rows (centring after prediction) | decides T1w |
| MET-09 | R2_level = 1 − Σ_c n_c (ȳ_c − ŷ̄_c)² / Σ_c n_c (ȳ_c − ȳ)²; s_b = SST_between/SST, s_w = π = 1 − s_b | in every OUT1 sentence |
| MET-10, MET-11 | R2_0 = 1 − Σ e² / Σ y² (log SF, skill vs FLAT); Skill_MSE(X\|Y) = 1 − SSE_s(X)/SSE_s(Y) on identical rows (a gain, not an R²) | beside T2/T3; every named baseline |
| MET-12 | EV_abs = var0(y) − MSE = R2_pooled·var0(y) (log10²); ΔMSE, ΔRMSE; relative MAE gain Δ/MAE(Y) | cross-target card |
| MET-13 | card: n_rows, n_pred_finite, n_cells/extractants/chemotypes/publications; Kish_rows = (Σ_g n_g)²/Σ_g n_g²; Kish_ext; largest chemotype share; y_mean; y_sd0 = sqrt((1/n)Σ(y − ȳ)²); y_var0; y_rms0 = sqrt((1/n)Σy²) (log SF only; zero log D is no no-skill reference); s_b, s_w (log D) | beside every R² ([BRIEF]:33) |
| MET-14 | dir_acc_t over rows with abs(y) > 0 (t = 0) or abs(y) ≥ t (t = 0.1, 0.3): 1 if signs agree, 0.5 if ŷ = 0 exactly, else 0; n_t and n_zero_pred printed | log SF only |
| MET-15, MET-16 | share with abs(e) ≤ log10 1.5 = 0.17609 and ≤ log10 2 = 0.30103 (constants), pooled and MET-04-weighted; bias = mean(ŷ − y), sd0(ŷ)/sd0(y), within-cell dispersion sqrt(Σŷ̃²/Σỹ²), OLS slope of y on ŷ | secondary; diagnostic |
| MET-17 | on A's residuals r: λ = Σ_c n_c r̄_c² / Σ r²; κ_adj = Σ_pairs (r_a − r_b)² / Σ_pairs (r_a² + r_b²); ρ_adj = Pearson(r_a, r_b); ρ_within_adj on cell-centred residuals | diagnostic with CI and per-seed values |
| MET-18 | per strict pair type (12) and per metal (14): n, y_sd0, FLAT MAE (log SF) or BD_METAL MAE, RMSE and R² (log D), MAE, RMSE, within-type R², R2_reason, n_seeds_defined; < 10 extractants → "preliminary" | TAB-10 |
| MET-19, MET-22 | replicate noise (§7); per config: selection frequency over 25 BP selections, inner best-vs-second gap, skipped inner folds | TAB-02; TAB-11 |
| MET-20, MET-21 | MAE_macro_ext_within = MET-01 on abs(ỹ_i − ŷ̃_i), centring over the cell's scored (or inner-validation) rows; R2_pair_macro (recovery plan :151) and R2_within_system (:154-156; ≥ 3 condition sets, else NA) | C's selection, EX-07/EX-08; exploratory |
| MET-23, MET-24 | reliability panel: seed sd of MET-01, 95 % interval width / comparator MET-01, range over six designs, LOCO sign count, sc009 vs no-sc009 sign agreement, per-pair-type sign consistency; Pr/Nd card: MAE, R², y_sd0, systems, frac_within_2x on the Pr/Nd rows of COH-SF-C3 | both targets side by side, descriptive only, never a verdict on "more reliable" (`PRE_REGISTRATION.md` §1); descriptive (O22) |

**Identities** (fixed text in the report; unit-tested in LT-21). **I1** R²_pooled = s_b·R²_level + s_w·R²_within_cell on
any log D row set (centred residuals sum to 0 in each cell). **I2** R²_within_cell(log D) equals the FLAT-referenced
skill 1 − Σw·e²/Σw·y² of the derived log SF over all unordered within-cell pairs, pair weight 1/n_c: a shape score over
all pairs, not adjacent pairs, and not a deployable log D accuracy. **I3** (log SF) 1 − R²_pooled = (1 − R²_0)(1 −
R²_FLAT), R²_FLAT = −ȳ²/var0(y), y_rms0² = y_sd0² + ȳ². **I4** on a seed-invariant row set, seed-mean R² = 1 −
mean_s(SSE_s)/SST. **I5** e_SF = r̃_a − r̃_b (r = y − ŷ_A, r̃ centred within the cell): A's cell-level residual cancels.
**I6** Σ_i (ỹ_i − ĉ_i)² = (1/n_c)·Σ_{i<j} [(y_i − y_j) − (ĉ_i − ĉ_j)]² + n_c·(mean_i ĉ_i)²: C's loss is the all-pairs
contrast loss over n_c plus n_c·(mean ĉ)², a penalty on the cell mean of the predictions that drops out of every
contrast but still constrains the fit (contrast training with a mean-zero regulariser; C_PAIR is the pure pairwise form).

**R² across targets.** Never ranked, differenced, variance-normalised or turned into a "difficulty score"
([BRIEF]:33): R² = 1 − RMSE²/var0(y), so equal RMSE gives different R² on targets of different variance; the same G14
predictions score R² 0.117 or 0.386 by reference (`generations/gen16_protocol/results/g16_oof_amplitude_r2_BP.csv:2`);
gen6's ORACLE_LEVEL, which reads the test cell's true level, had pooled log D R² 0.935 (split seed 104729;
`runs/gen6_expC_5seed/arm_metrics.csv:8`), yet its derived pair macro MAE (0.503469) is identical to that of the same
shape model with a predicted level, C2_TRUECENTRE (0.503469; `runs/gen6_expC_5seed/decision_report.md:532,534`): the
level cancels exactly. Separation claims rest on the log SF table only.

**NA and negative R²** ([BRIEF]:57). NA with a reason when n < 2 or SST ≤ 0; every table and JSON carries
`R2_reason` and `n_seeds_defined` (JSON: null plus reason). `n_pred_finite == n_rows` is asserted before any metric
(gen20k `metrics.r2` drops non-finite rows silently, metrics.py:19-21); a non-finite prediction is a run failure. The
seed mean is NA if any seed is NA. NA bootstrap draws are kept (−∞ for lower, +∞ for upper bounds, unfavourable in
contrasts, counted); > 1 % NA draws → interval NA ("degenerate"). R² is never clipped, averaged across folds,
extractants or pair types, or replaced by corr²; Skill_MSE is NA when the baseline SSE is 0.

**Results tables.** Every table carries design, cohort, mode, regime, averaging unit and per-seed values, from
unrounded float64 values printed to 4 decimals. Status of every entry today: **not computed**. Every row or sentence
that prints an R² (MET-06/07/08/10, a per-type or per-metal R²) or a RULE-T class also prints, for the same rows:
n_rows and n_cells, y_sd0 and y_var0 (MET-13), MET-01, MET-02 and MET-05, and the R², MAE and RMSE of the target's
simple baseline on identical rows (BD_METAL for log D, FLAT for log SF) ([BRIEF]:33).

| ID | rows | columns | status |
|---|---|---|---|
| TAB-01 | every COH-* population; per-fold sizes for six designs; fold digests | cells, entries, pairs, extractants, chemotypes, publications, Kish, largest share, replicated entries | not computed (S3); §2 [CNT] values are the expected entries |
| TAB-02 | target statistics and noise: every scored population; the 25 BP training folds | MET-13 card incl. y_mean, y_sd0, y_var0, y_rms0, s_b/π, per-pair-type sd; MET-19 σ̂²_pool, σ̂²_pool,cap, median s_e (median and min–max over folds, entry counts); log SF noise or "not estimable" | not computed (S5) |
| TAB-03 | log D table (COH-D-C3, BP, mode Z): A, BD_MEAN, BD_METAL, BD_COND, BD_NN, BD_LEVEL, BD_LEVEL_CURVE, BL_D_INNER, A_PRIME | MET-13; MET-01 (seed mean, per seed, 95 % and 98.75 % chemotype and publication intervals); MET-02..05; MET-06 + R2_reason + n_seeds_defined; MET-07 (+ within); MET-08; MET-09; Skill_MSE vs each H1 member; MET-12; MET-15; MET-16; T1/T1w classes; fallback counts | not computed (S9/S10) |
| TAB-04 | log SF table (COH-SF-C3): B, C, B_selSF, C_selSF, C_ACFG, C_PAIR, FLAT, PAIRMEAN, G14_ZERO, MEAN_CURVE_REF, NN_TANIMOTO_REF, BL_SF_INNER, BD_*_SF; HEAVIER_ALWAYS (MET-14 only) | MET-13 incl. y_rms0; MET-01..07; MET-10; Skill_MSE vs FLAT, PAIRMEAN, G14_ZERO; MET-12; MET-14 at t = 0 / 0.1 / 0.3 with n_t, n_zero_pred; MET-15; MET-16; T2/T3 classes | not computed (S9/S10) |
| TAB-05 | registered verdicts: H1 (4 components), H2 (3), H3 (3), H4 | Δ, δ_Y, chemotype 98.75 % percentile and BCa bounds, two-sided p, p_wcr, p_wcu, publication 98.75 % bound, seeds positive, LOCO minimum and chemotype, signs under B/BR/BQ/BPG/A, ΔMAE_bal, ΔR2_balanced and bounds, own R2_balanced, Skill_MSE vs FLAT, mde80_95/dec/cr, verdict, sub-label, annotations | not computed (S9/S10) |
| TAB-06 | threshold attainment: T1, T1w, T2, T3 × τ ∈ {0.4, 0.5}; the same class for every baseline, reference and exploratory arm on the same rows | point, per-seed values, RULE-T (ii)–(vi) inputs, 97.5th percentiles, 98.75 % bounds, LOCO minimum, COH-SC009 point, COH-NOSC009 point and interval, COH-NODMD point, τ + 1.96·se_boot, I1 decomposition (T1), R2_0 (T2/T3), class, annotations; MET-13 (n_rows, n_cells, y_sd0, y_var0), MET-01, MET-02, MET-05, and the simple baseline's R²/MAE/RMSE on the same rows | not computed (S9/S10) |
| TAB-07 | cross-target card: log D on COH-D-C3 and on COH-D-C3M (matched to the log SF rows) beside log SF (COH-SF-C3); no ranking column (O27) | MET-13; R² with R2_reason; MET-01/02/03/05; Skill_MSE vs every named baseline; MET-12; MET-23; MET-17 with CI; I1–I5 as fixed text | not computed (S9/S10) |
| TAB-08 | design sensitivity: registered arms and IUT baselines × BP, B, BR, BQ, BPG, A | MET-01, MET-06, MET-08 (log D), MET-10 (log SF), sign of each claim's point Δ, publication-leak price; MET-13 (n_rows, n_cells, y_sd0, y_var0), MET-02, MET-05, and the simple baseline's R²/MAE/RMSE on the same rows | not computed (S9/S10) |
| TAB-09 | secondary populations and sensitivities, one panel each: (a) matched log D table on COH-D-C3M; (b) COH-D-ALL; (c) COH-SF-ANY, COH-SF-C1; (d) COH-SF-NDSM (labelled pass); (e) COH-SC009 / COH-NOSC009; (f) COH-NODMD (refit); (g) COH-REPCAP | TAB-03 / TAB-04 columns, identical for all arms; contrasts as EX-09 / EX-10 | not computed (S9/S10) |
| TAB-10 | per pair type (12, log SF), per metal (14, log D), Pr/Nd card | MET-18; MET-24 | not computed (S9/S10) |
| TAB-11 | per (design, seed, fold) | training cells and rows, test cells, test C3 cells and pairs, inner sizes, skipped inner folds, selected K for A, C, BD_COND, B_selSF, C_selSF, inner scores; MET-22 | not computed (S9/S10) |
| TAB-12, TAB-13 | exploratory family EX-01..EX-15; "not selectable" appendix A_K1..K4, B_K1..K4, C_K1..K4 | Δ, intervals, raw p_wcr, BH-adjusted q; outer MET-01, MET-06, MET-08 / MET-10 with MET-13 (n_rows, n_cells, y_sd0, y_var0), MET-02, MET-05, and the simple baseline's R²/MAE/RMSE on the same rows | not computed (S9/S10) |
| TAB-14 | gen20 context numbers with task changes | value, arm, setting, modification, sd and n, source (§1.3) | cited, not recomputed |

## 6. Inference and decision rules

**Bootstrap** ([BRIEF]:55; full rule text in `PRE_REGISTRATION.md`). Units: chemotype (38 clusters on C3; deciding) and
publication (39; required co-condition); never extractant, cell or row; seeds not resampled. 10,000 draws per unit via
gen20k `_draw_counts` (inference.py:102-108) over the sorted cluster list, seeds 8675309 (inference.py:29-30) and
20260930 (recovery plan :101); one matrix per (unit, cluster list) shared by every arm, target and seed. Every statistic
is recomputed per draw from per-cluster sufficient statistics (denominators about the draw's mean, weights recomputed);
macro = Σ_l u_l·MAE_l* / Σ_l u_l, MAE_l* = Σ_k c_k S_lk / Σ_k c_k n_lk, u_l = Σ_k c_k n_lk / n_l (gen20
`blocked_unit_contrast`, inference.py:282-307). Intervals: chemotype percentile and BCa (mid-rank z0, LOCO-jackknife
acceleration; inference.py:65-86) and publication percentile, at 95 % and 98.75 %. COH-SC009: chemotype interval NA
("one cluster"); < 10 clusters → "preliminary" (recovery plan :304).

**WCR, LOCO, MDE.** `gen13sep.wildcluster.wild_cluster_t` (CR2, null imposed, Rademacher; Webb only if G is below 12;
9,999 reps, seed 8675309; wildcluster.py:207-209) on per-extractant seed-mean deltas via gen20k `unit_contrast`
(inference.py:229-249; `unit_contrast` passes no weights argument, inference.py:239, so a G < 12 case calls
`wild_cluster_t(..., weights="webb")` directly). In gen13's null simulation on its BP_all cluster profile (45 chemotype
clusters, Kish 11.67, ICC 0.72) WCR rejected at 0.044–0.048 and gen13's pairs cluster percentile bootstrap at
0.071–0.077 at nominal 0.05 (`generations/gen13_separation/metrics/BP_all/inference_calibration.csv:2,9,20,27`); neither
was simulated at α = 0.0125 or on the C3 profile (38 chemotypes, Kish 3.43 rows / 3.85 pairs), so the size of the
deciding test here is unverified. LOCO: every finite value > 0, NA fails. MDE: 2.80·se_boot; (z_0.99375 +
z_0.80)·se_boot = 3.34·se_boot; `detectable(se_cr2, dof_BM, alpha = 0.0125)` (wildcluster.py:396-403); UNDERPOWERED uses
the larger of the last two.

**RULE-T** (O7, O24; [BRIEF]:58), per τ ∈ {0.4, 0.5} on seed-mean MET-06 (T1, T2, T3) or MET-08 (T1w). **REACHED** iff
(i) point ≥ τ; (ii) chemotype 2.5th percentile ≥ τ; (iii) chemotype 95 % BCa lower bound ≥ τ; (iv) publication 2.5th
percentile ≥ τ; (v) ≥ 4 of 5 seeds ≥ τ; (vi) seed-mean R2_balanced (within-cell for T1w) ≥ τ. **POINT_ONLY**: (i) and a
failing or NA (ii)–(vi). **RULED_OUT**: chemotype and publication 97.5th percentiles < τ. **NOT_REACHED**: (i) fails and
not RULED_OUT. **NA**: seed mean NA. When (i) holds and both 97.5th percentiles are below τ, POINT_ONLY and RULED_OUT
both apply; the class printed then is POINT_ONLY (O7 default, fixed at S4), which never counts as attainment, with
both percentiles printed. Fixed annotations: 98.75 % bounds, LOCO minimum, COH-SC009 point,
COH-NOSC009 point and interval, COH-NODMD point, τ + 1.96·se_boot, the O26 feature-set label; for T1 the I1
decomposition, "reached also by <baseline>" and "**level-attributable**" (BD_LEVEL's class ≥ A's (class order REACHED >
POINT_ONLY > NOT_REACHED > RULED_OUT, a POINT_ONLY/RULED_OUT overlap ranked as the class printed under O7; if either
class is NA, this clause is printed "not evaluable (NA)" and the other two still apply), or R2_within_cell(A) ≤ 0 or ≤
that of BD_LEVEL_CURVE); for T2/T3 R2_0, where gen20's R²_FLAT = −0.1347 makes pooled 0.4 ⇔ R²_0 ≈ 0.471 and 0.5 ⇔ ≈
0.559 (arithmetic on levels csv:6).

**Registered claims** (seed mean, design BP, mode Z; K = 4, α_K = 0.0125, deciding intervals 98.75 %).
BASELINE_BETTER is a union over a claim's comparators and lies outside this error statement. Each named-Y test is
one-sided at 0.00625 (two-sided 98.75 % bound and p_wcr < 0.0125), so a false BASELINE_BETTER for one H has probability
at most |Y|·0.00625 (H1: 0.025; H2, H3: 0.01875).

| ID | statement | estimand | comparators (IUT) |
|---|---|---|---|
| H1 | A predicts absolute log D of held-out chemotypes and publications better than every log D baseline | Δ_Y = MET-01(Y) − MET-01(A) on COH-D-C3 | BD_MEAN, BD_METAL, BD_COND, BD_NN |
| H2 | B predicts strict-adjacent log SF better than every log SF baseline | Δ_Y = MET-01(Y) − MET-01(B) on COH-SF-C3 | FLAT, PAIRMEAN, G14_ZERO |
| H3 | C does the same, reported separately from B | Δ_Y = MET-01(Y) − MET-01(C) on COH-SF-C3 | FLAT, PAIRMEAN, G14_ZERO |
| H4 | C and B differ (two-sided) | Δ = MET-01(B) − MET-01(C), positive favours C | — |

**RULE-C, RULE-H4, outcomes.** Per component: (a) Δ ≥ δ_Y = max(0.02, 0.10·MET-01(Y)) (0.02 against FLAT and G14_ZERO:
0.10 × 0.1731 and 0.10 × 0.1612, levels csv:6,18); (b, c) chemotype 98.75 % percentile and BCa lower bounds > 0; (d)
p_wcr < 0.0125 with a positive mean (O21); (e) publication 98.75 % lower bound > 0; (f) ≥ 4 of 5 seeds positive; (g)
every LOCO Δ > 0; (h) point Δ > 0 under each of designs B, BR, BQ, BPG and A; (i) ΔMAE_bal > 0; (j) log SF: ΔR2_balanced
≥ 0.05 with both 98.75 % lower bounds > 0 and own R2_balanced > 0; (k) log SF: Skill_MSE(X | FLAT) > 0 (O20). Verdicts
in order: `NA` → `BASELINE_BETTER` (some Y significantly better) → `INSUFFICIENT_COMPUTE` (a design needed for (h) not
run while (a)–(g) hold for every Y on design BP; otherwise that (h) counts as failed; `PRE_REGISTRATION.md` §8.1) →
`PASS` ((a)–(k) for every Y) → `ESTIMAND_INCONSISTENT` ((a)–(h) only) → `INCONCLUSIVE` (`ENCOURAGING`, `UNDERPOWERED`,
or both, printed in that order), with fixed annotations (NC1 flag, COH-NODMD flip, low power, Pr/Nd status per recovery
plan :275, the O26 feature-set label). H4: `NA` (C not run) → `INSUFFICIENT_COMPUTE` (as in `PRE_REGISTRATION.md`
§8.1) → `C_BETTER` / `B_BETTER` by RULE-C in either direction, `EQUIVALENT` when both 98.75 % intervals lie inside
±0.02, else `NOT_ESTABLISHED`; B_selSF, C_selSF, C_ACFG printed beside. OUT1 = T1 with T1w and I1; OUT2 = T2 and T3,
never inferred from OUT1; OUT3 = H1–H3 with H4 beside.

**Exploratory family, NC1, confirmation.** EX-01..EX-15 (roles in §3; EX-12 is interpreted only at Spearman–Brown
split-half reliability ≥ 0.3; EX-13 = NC1, A vs A_PRIME in both directions, whose pass flags every PASS) get BH q = 0.05
on p_wcr and are never promoted; membership and number of tests fixed at S4 (m = 83, EX-15 excluded because O4 is OFF; enumeration in `PRE_REGISTRATION.md` §8.1), including every
EX whose arm depends on an S8 smoke test (EX-01, EX-03, EX-04 and the C contrasts within EX-09, EX-10 and EX-11 on
the centred learner; EX-05, EX-06 on C_PAIR); an EX whose arm is "not run" after S8 is reported NA and enters BH with
p = 1, m unchanged. Confirmation (O5): five seeds from
int(sha256("gen20_1-confirmation-seed-{i}")[:8], 16) % 900000 + 100000, excluding the discovery, gen16 (137171, 506411,
721632, 937310, 939478) and gen20 (148569, 228670, 253534, 561551, 978178; GEN20_REPORT.md:320) seeds; computed and
committed at S4; run once on design BP; an H is "confirmed" only if it PASSes on both seed sets (H4: the same verdict on
both), with (a)–(g) and (i)–(k) re-evaluated on the confirmation seeds and (h) taken from discovery; a threshold class
is final REACHED (RULED_OUT) only if so on both sets, otherwise both are shown; a "partition-robustness check on the
same corpus, not independent validation".

**Pre-stated expectations** (no threshold reached; H2/H3 do not PASS against G14_ZERO (INCONCLUSIVE,
ESTIMAND_INCONSISTENT or BASELINE_BETTER); H4 EQUIVALENT or NOT_ESTABLISHED) and their sources are fixed in
`PRE_REGISTRATION.md`; they change no rule.

## 7. Replicates and noise ([BRIEF]:56)

- **Target and replicates**: the frozen arithmetic replicate mean per (cell, metal), `nrep__`/`repsd__` (ddof 1) kept,
  no nrep weighting, readings never cross-multiplied into pairs (recovery plan :83; gen13sep/cohort.py:159-167);
  replicates are sparse (41 of 521 cells, 216 entries, D5/FINDINGS.md:44; mostly hidden-axis rows, 38 replicated entries
  under key_mode = series, `generations/gen13_separation/manifests/cohort_series.json:5,99`) and about 12 % differ by
  whole decades (GEN20_REPORT.md:402); COH-REPCAP (scoring only) drops entries with nrep ≥ 2 and repsd > 0.5 and pairs
  containing one.
- **Noise (S5, descriptive only, never in a rule)**: log D per outer (seed, fold) from masked training cells, σ̂²_pool =
  Σ_e (n_e − 1) s_e² / Σ_e (n_e − 1), σ̂²_pool,cap (s_e ≤ 0.5) and median s_e with s_e ≤ 0.5 (the capped median is gen20
  `training_noise_s2`, cov.py:101-114, which returns its square; a gen20.1 caveat, not gen20's: the median of s_e is
  biased low for σ when n_e is small), median and min–max over the 25 BP folds (TAB-02); log SF only from COH-REPPAIR
  (d_r = x_a,r − x_b,r, s_SF = sd_r(d_r)), ≥ 20 pairs from ≥ 5 chemotypes per fold, else "not estimable", < 10
  chemotypes "preliminary"; √2·σ̂ only as a labelled upper bound (recovery plan :87).
- **Cited sensitivities only** (median replicate sd of log D unless stated): 0.237 (DATA_AUDIT.md:76, pre-quarantine);
  0.302 (DATA_AUDIT.md:79, frozen cohort, 216 entries); 0.225 (282 groups; `generations/gen18_process/DESIGN.md:42`);
  0.333 (`generations/gen17_pairdiff/results/g17_summary.txt:18`); 0.299, gen19's noise-floor sd
  (`generations/gen19_chem_transfer/preregistration.md:1070`); "~0.21 on one log SF", underived
  (`generations/gen20_kshot/IMPLEMENTATION_PLAN.md:193`). No R² or MAE noise ceiling is computed or quoted (O17;
  `docs/figures/refinement/lanthanidestrain/SCIENTIFIC_DEFECTS.md:123`).

## 8. Code layout, reuse map and hazards

**Layout** (nothing is created now): package `gen201` in `generations/gen20_1/` (precedent gen122); `scripts/g201_*.py`
(`g201_counts.py`, `g201_describe.py`, `g201_discovery.py`, `g201_report.py`, `g201_confirm.py`, plus anchor and smoke
entry points); `tests/`, `manifests/`, `results/`, `predictions/`; a `.gitignore` for `predictions/**/*.parquet`,
`results/**/*.parquet`, `smoke*/`; both interpreter forms (`.venv/Scripts/python.exe`, `.venv/bin/python`). **Output**:
per design two long float64 tables in `predictions/<label>/` (digests in the manifest), log D keyed (design,
split_seed, fold, cell_id, extractant, chemotype, publication, n_metals, metal, y) and log SF keyed (…, a, b, ia, ib,
dZ, pair, y), one column per arm, baseline and reference of §3; plus selection records (JSON) and inner OOF tables.
Every metric is recomputed from these tables alone (gen20 cast to float32, runner.py:335-337).

| class | modules and functions |
|---|---|
| import read-only (side-effect-free) | `gen20k.data` (load_bench, assert_matches_manifest, cell_view, pair_value, is_strict_adjacent, is_gen13_adjacent, observed_metals, LANTHANIDES, Z_VEC, sha256_of; data.py:33-141); `gen20k.splits` (folds_for, assert_design_integrity, inner_seed, N_INNER, fold_record, copy_groups_of); `gen20k.design` (make_design, query helpers); `gen20k.cov` (make_ctx, inner_mask, training_noise_s2, REPLICATE_SD_CAP); `gen20k.arms` (standard_arms, as_fit_curves); `gen20k.runner` (RunConfig, run_fold, select_on_inner; in memory only) (S7 only: importing it also imports gen20k.nnc and torch, runner.py:30, nnc.py:32; NNC is never instantiated; discovery, report and confirm do not import gen20k.runner); `gen20k.metrics` (behind the coverage assertion, NA reason kept); `gen20k.inference` (pooled_levels, pooled_contrasts, macro_mae_contrast, unit_contrast, blocked_unit_contrast, benjamini_hochberg, _draw_counts, interval_columns, REPLICATES, SEED, LEVELS; wrapped so NA seeds propagate); `gen20k.report` (kish, per_pair, wide_common, levels; wrapped); `gen20k.manifest` (git_state, dataset_digests); `gen13sep.models` (chemotype_balanced_weights, tree_pipeline, FitContext, DirectRowArm for LT-13 only, AntisymmetricPairArm, PairMeanArm, MeanCurveArm, HeavierAlwaysArm); `gen13sep.arms_stage2` (CentredRowArm for LT-13 only, metal_feature_matrix); `gen13sep.amplitude_bench.LEAN_BLOCKS`; `gen13sep.features._assert_allowed`; `gen13sep.metals`; `gen13sep.wildcluster` (wild_cluster_t, cluster_robust, detectable); `lanthanide_separation.gen6.cohorts.seeded_group_kfold` |
| re-implement under gen201 (never import) | RowForest without centring (`gen201.learners`); BD_* baselines; R2_within_cell, R2_level, MET-04/07 weighted routines, MET-17, MET-20; the macro-in-draw formula; the publication bootstrap wrapper; the gen16 fold digest; `Run`, `preregistration_state`, `require_sealed`, `seal` with the file path as a parameter; `environment()` that records whether torch is loaded and its thread count, adding platform, cpu_count, RAM and LOKY/OMP settings |
| forbidden (writes into frozen trees or checks gen20's seal) | `gen20k.manifest` Run, preregistration_state, require_sealed_preregistration, seal_preregistration, environment (imports torch; manifest.py:20-21,34-38,48-79,81-114; `Run.__exit__` writes `bootstrap.MANIFESTS`, :111-113); `gen20k.runner.run` (runner.py:322-346; a relative out_dir (gen20's label form) is written under gen20_kshot/predictions (:324), a Git-Bash-style `/d/...` path lands at `\d\...` on the current drive, and float64 columns are cast to float32 (:335-337)); `gen20k.report` load_tables, build_design, figures, setting_b_tables (report.py:35-37,128-170,267,330); gen20k.bootstrap RESULTS / PREDICTIONS / MANIFESTS as write targets; `scripts/g20_*.py`; `gen13sep.runner.run_ladder` (gen13sep/runner.py:58-75); `gen13sep.cohort.write_cohort` (cohort.py:236-243; overwrites manifests/cohort_exact.parquet/.json); anything named gen16; `gen14.dirbench.load` when the cache is missing (rebuilds and writes, dirbench.py:44,56-66); instantiating gen20k.nnc.NNC |

**Seal mechanics** (gen20k/manifest.py:20-21,48-79,111-113). The digest is taken over LF-normalised UTF-8 (this clone
has core.autocrlf = true; gen20's PRE_REGISTRATION hashes `e1533939…` raw vs `2915974a…` LF-normalised); sealing is
refused while `DRAFT / UNSEALED`, `TO BE FINALISED` or `- [ ]` remains (mechanism O19); describe, anchors, discovery,
report and confirm call `require_sealed(generations/gen20_1/PRE_REGISTRATION.md)`; counts and smoke do not. **Bootstrap
hazards** (`gen201/bootstrap.py` before any frozen import): `sys.dont_write_bytecode = True`; `LOKY_MAX_CPU_COUNT` and
`OMP_NUM_THREADS` = "2" via setdefault; insert generations/gen20_kshot, then import gen20k.bootstrap
(gen20k/bootstrap.py:14-25); fail fast if the local, gitignored `generations/gen14_direction/cache/bench.pkl`
(`.gitignore:91`; 10,901,386 bytes, local file size 2026-10-15) is missing; its sha256 is recorded at S3; LT-27
(gen13sep/paths.py:41-43 calls mkdir at import).

## 9. Stages, gates, stop conditions and compute

| ID | name | content | gate / stop condition |
|---|---|---|---|
| S1 | DRAFT | the three draft documents only | user review |
| S2 | APPROVAL | the user answers O1–O27; drafts revised | explicit approval in chat |
| S3 | COUNTS (Stage 0a) | `scripts/g201_counts.py`: the §1.1 cohort assertions and the §4 fold gates (a)–(e), `bench.Y` == frame, bench.pkl sha256, per-fold sizes, fold records, row manifests and digests, every §2 count, DMDPhPDA rule, COH-REP counts (nrep only), COH-REPPAIR feasibility, the §4 first-inner-fold equality; no y value read beyond NaN masks and equality assertions; imports frozen packages only through gen201/bootstrap.py (§8) | all pass, and every value marked [CNT], [DMD] or [INNER] in the three documents (here §1.1, §1.2, §2, the §4 giant-fold tuples and inner-fold sizes, including 2,425 / 2,208 / 1,802 / 229 / 79 / 38 / 39 / 5 / 70 / 60) equals its recomputed value; else STOP `INCOMPARABLE_FOLDS` |
| S4 | SEAL | resolve every `TO BE FINALISED` (grid O9, RULE-T precedence O7, exploratory family, reduction details O18, mechanism and plan freeze O19); compute and commit the confirmation seeds; seal `PRE_REGISTRATION.md` over LF-normalised text (O19); commit | sealer refuses while any draft marker remains; digest in the manifest; afterwards only dated addenda stating what changed, why, and whether S5, smoke or results had been seen |
| S5 | DESCRIBE (Stage 0b) | `scripts/g201_describe.py` (sealed gate): MET-13 on every scored population, π, per-pair-type sd, MET-19 per training fold; COH-REPCAP row manifest and count (label-derived); hashed manifest; no model fit | any later protocol change is an addendum stating that S5 was seen |
| S6 | CODE+TESTS | the rest of the gen201 package (bootstrap, the counts script, the seal helper, `require_sealed` and the describe script are written first, for S3–S5); LT-01..LT-30 | all green, including LT-13 and LT-26; then g201_counts.py and g201_describe.py are re-run with the tested package, and their row manifests, fold_plan_g201.json and S5 manifest must be byte-identical to the S3/S5 digests; otherwise STOP and diagnose (any S5 change is a dated addendum stating that S5 was seen) |
| S7 | ANCHORS | `run_fold` in memory (no NNC) or gen201 refs: FLAT/G14/MEAN_CURVE/NN_TANIMOTO @ZERO on COH-SF-C3 (n_rows 1,802, n_cells 229, Kish 3.851121; levels csv:6,18,30,62); gen201 refs equal the run_fold columns; RIDGE2\|k3 only if O4 | gate on the full-precision seed-mean columns R2, MAE and MAE_macro_extractant of levels csv:6,18,30,62: within 1e-9 "reproduced"; within 1e-6 "reproduced (float32 storage or platform drift)", recorded (gen20 cast predictions to float32 before scoring, runner.py:335-337; gen20: macOS, Python 3.13.0, numpy 2.5.1, `generations/gen20_kshot/predictions/discovery/manifest.json:12-14`; this machine [ENV]); beyond 1e-6 STOP and diagnose. Per-seed values are compared with the csv's 5-decimal per-seed strings (gen20k/inference.py:225) at 6e-6 (half a unit of the 5th decimal plus the 1e-6 drift allowance); beyond that STOP and diagnose, as for the seed means. n_rows and n_cells must match exactly; Kish must match within 1e-12 |
| S8 | SMOKE+TIMING | `smoke*` runs after S5: one seed, the BP sc009 fold plus one fold of about 490 training cells, ≤ 40 test cells; coverage, finiteness, C_PAIR antisymmetry, frozen-tree invariance; seconds and peak RSS per fit; no metric against outer y | coverage 100 %; if the estimate is > 2× the two-pass figure below (≈ 12 h), apply the reduction rule |
| S9 | DISCOVERY | `scripts/g201_discovery.py` (sealed gate): designs in the order BP, B, BR, BQ, BPG, A; all arms; primary + COH-NODMD refit (all six designs, same Fold objects, §2); long tables, selection records, inner OOF tables, manifests | coverage = row manifests; fold digests re-asserted; no NaN prediction |
| S10 | VERDICTS+REPORT | `scripts/g201_report.py`: TAB-01..TAB-14 from saved tables only; RULE-T, RULE-C, RULE-H4 by templates; BH; NC1; `GEN20_1_REPORT.md` opens with the applicable pre-written sentences of `PRE_REGISTRATION.md` (§1 "easier and more reliable"; §7 PAT/H4), verbatim | script output only; no outer-score choice of anything |
| S11 | CONFIRMATION | `scripts/g201_confirm.py`, once (RUN_ONCE), design BP, primary + COH-NODMD refit | executed exactly once; confirmation fold digest equal to the S4 commitment |
| S12 | AUDIT | two adversarial auditors, then a verification pass | findings resolved or listed |

**Compute estimate (8 GB Windows; re-measured at S8).** Per (design, seed, fold): A ≤ 4 configs × (≤ 4 inner + 1
outer) = ≤ 20 ExtraTrees fits; C ≤ 20; BD_COND ≤ 20; BD_LEVEL and A_PRIME (BP) 1 each; gen20 references take seconds
(`generations/gen20_kshot/IMPLEMENTATION_PLAN.md:440`). On [TIM] mean single-fit times (gen13's machine) A ≈ 20 ×
3.58 s, C ≈ 20 × 1.67 s, BD_COND ≈ 20 × 0.76 s, plus ≈ 2 × 3.6 s and 10–20 s of references: ≈ 140–150 s per unit;
≈ 6 h per pass over 150 units (6 designs × 25), ≈ 1 h per pass for BP alone; the confirmation (design BP, identical
procedure including the COH-NODMD refit, `PRE_REGISTRATION.md` §11) ≈ 2 h (arithmetic).
The COH-NODMD refit of S9 repeats every fit under all six designs on the same Fold objects (the 5 DMDPhPDA cells
removed from every inner and outer training set and from scoring; folds never re-dealt), so the planned discovery total
is ≈ 280–300 s per (design, seed, fold), ≈ 12 h for 150 units and ≈ 2 h for design BP alone, plus C_PAIR (designs BP
and B) and the S7 anchors; the S8 "> 2×" gate applies to this two-pass total. C_PAIR's training pair rows (about 27k)
and peak memory (below 2 GB of 8,464,097,280 bytes, [ENV]) are planning estimates (local file sizes 2026-10-15),
re-measured at S8; C_PAIR's cost is unknown until S8. The [TIM] times were taken without a thread cap (gen13's
`tree_pipeline` uses n_jobs = -1, `generations/gen13_separation/gen13sep/models.py:41`, and gen13sep sets no
LOKY_MAX_CPU_COUNT), whereas gen20.1 caps loky at 2, so single fits here may be several times slower; the S8
measurement, not this arithmetic, decides the gate. One process at a time, fits discarded after use,
`LOKY_MAX_CPU_COUNT = 2` (≤ 4 after smoke; fixed-seed ExtraTrees does not depend on n_jobs), `OMP_NUM_THREADS = 2`,
resumable per design. **Reduction rule** (pre-registered; details fixed at S4 as in `PRE_REGISTRATION.md` §8.1, O18 default: resumable units, designs started in the order BP, B, BR, BQ, BPG, A, at most two processes on whole designs, nothing dropped):
over budget, i.e. when the S8-measured estimate exceeds 2× this two-pass estimate (the S8 gate), designs run in
separate later invocations with the procedure unchanged; design BP, the publication mask and every design are kept; a
design never run makes RULE-C (h) unevaluable (`INSUFFICIENT_COMPUTE` when the BP conditions (a)–(g) hold, as defined
in `PRE_REGISTRATION.md` §8.1). The two-pass estimate already reaches the recovery plan's 12 h cap and "max 2
processes" (:313); the confirmation (≈ 2 h), C_PAIR and the S7 anchors come on top; any excess is a recorded
deviation (O18).

## 10. Deviations from and conflicts with existing conventions

| # | convention (source) | gen20.1 choice | why |
|---|---|---|---|
| 1 | recovery plan: absolute log D a non-goal; R²(log D) appendix only (:10, :61, :192) | A has its own table and primary metric; never in the separation-success table | the later brief governs |
| 2–3 | recovery plan: MAE_bal primary (:123); rival = best baseline on inner MAE_bal (:270); the log-D-then-subtract rival is selected and tuned by separation error (:201) | MET-01 primary, MAE_bal sign required (RULE-C (i)); IUT against every named baseline, inner-chosen form as EX-11; B inherits A's inner log D selection, and the SF-tuned form is the exploratory B_selSF (EX-02) | gen20 continuity; stricter, no outer choice; the brief defines B from A's out-of-fold predictions ([BRIEF]:44) |
| 4–5 | recovery plan: 2,000 draws, seed 20260930 (:101, :302); grouping seeds 20260927-29 (:101) | 10,000 chemotype (8675309) + 10,000 publication (20260930) draws; gen20's split seeds | lower Monte Carlo error; identical folds required ([BRIEF]:47) |
| 6 | recovery plan: A = heavier (:43) | lighter first; explicit conversion; flip test LT-20 | gen20 continuity |
| 7 | recovery plan: PAIRMEAN extractant-equal (:199) | chemotype-balanced (gen13 PairMeanArm), strict-adjacent fallback | matches A's weighting |
| 8 | recovery plan: Pr/Nd separate success (§9.2), item 4 | Pr/Nd descriptive; item 4 as PASS annotation | not requested by the brief (O22) |
| 9 | recovery plan: folder gen20_separation/ (:335) | `generations/gen20_1/` | the brief's path (O3) |
| 10 | recovery plan: 12 h cap, max 2 processes (:313) | one process; all six designs required; excess recorded | gen13's all-designs rule (O18) |
| 11 | gen20: single inner split `select_on_inner` (runner.py:100-104) | 4-fold inner cross-fit of the same split (design A: regrouped by chemotype) | 10–11-cell validation sets (O12) |
| 12 | gen20: cross-fold majority over 25 selections (`generations/gen20_kshot/PRE_REGISTRATION.md:90-94`) | nested per-fold argmin, no vote | 13/25 near tie (GEN20_REPORT.md:131-132) |
| 13–15 | gen20: WCR p < 0.05 with 98.75 % intervals; mde_80 = 2.80·se; margin 0.02 (`generations/gen20_kshot/PRE_REGISTRATION.md:114-118`; inference.py:145) | p_wcr < 0.0125; three MDEs, decision on the larger of the 98.75 % and CR2-t ones; max(0.02, 0.10·MAE(Y)) | level consistency (O21); wildcluster.py:396-403; log D scale (O6) |
| 16–17 | gen20: nanmean over seeds and draws, R2_reason not written, float32; rule string "gen20-confirmation-seed-{i}" | NA propagation, R2_reason column, float64; "gen20_1-confirmation-seed-{i}" excluding gen20's seeds | §5 NA rules; float64 output tables; gen20 seeds spent |
| 18 | gen20: last-line seal (manifest.py:53) vs gen18/19 below-footer addenda | no footer in the draft; mechanism decided at S4 | incompatible conventions (O19) |
| 19–20 | gen13 Addendum 3: five designs; gen13/14 code: zero prediction counted wrong | six designs (+BPG), sign required for PASS; 0.5 (gen20; recovery plan :187) | gen20 added BPG; 0.5 tie rule (recovery plan :187) |
| 21–22 | gen13 DATA_AUDIT §3 statistics as context; √2·sd for log SF noise (DATA_AUDIT.md:77) | never R² context, π at S5; run-paired estimate or "not estimable", √2·σ̂ only as a labelled bound | different populations; recovery plan :87 |
| 23 | gen19 sealer: `preregistration_draft.md` | one `PRE_REGISTRATION.md` with a DRAFT banner | the brief names the file |
| 24 | pre-existing: `verify_relocation.py` fails on this Windows clone (CRLF) | reported as pre-existing, not fixed, not attributed to gen20_1 | observed 2026-10-15 by running `generations/verify_relocation.py` on this clone (core.autocrlf = true) |
| 25 | gen13/gen20 design-A inner split by extractant (gen13sep/splits.py:30,105-107; used by gen20 `select_on_inner`, runner.py:100-101) | chemotype inner cross-fit for design A as for every other design | one selection unit across designs; stricter than A's outer grouping |

## 11. Open items for the user (the conservative default is in force until answered)

| ID | question | conservative default |
|---|---|---|
| O1 | Where does the brief's "normal values" count come from, and what does it count? | never used; 3,359 obs / 521 cells, and 2,425 / 1,802 on C3, with units |
| O2, O3 | Documents in English or Russian? Keep `gen20_1/` beside `gen20.1/`, add a topic suffix or move the RTF; when are index rows added? | English with an English digest and a pointer to the RTF; `gen20_1/` exactly, `gen20.1/` untouched, index rows after approval |
| O4 | Include measured mode M (k = 3, DOPT_Q, C3) and the RIDGE2\|k3 anchor? | OFF. If switched on by the user's S2 answer to O4 (entered in the drafts before the seal) or, after the seal, by a dated addendum before any run: MODE_M_A predicts an unread metal m as Â_m + mean_{s∈S}(y_s − Â_s), S = gen20 DOPT_Q k = 3 read set, read metals not scored, labelled "level revealed"; MODE_M_B uses y_s − Â'_m, oriented lighter-first, for a pair with a read endpoint s, otherwise B; gated on RIDGE2\|k3 = 0.30991136191292135 on 1,802 rows; never pooled with Z; outside all claims |
| O5, O6 | Register a one-shot confirmation on new seeds (BP only or six designs)? Margin max(0.02, 0.10·MAE(Y)) instead of gen20's 0.02? | registered, BP only, (h) from discovery; max() (= 0.02 for log SF against FLAT and G14_ZERO) |
| O7 | Threshold bounds at 95 % or 98.75 %? R2_balanced and publication bounds co-deciding? Which class is printed when POINT_ONLY and RULED_OUT both apply (point ≥ τ, both 97.5th percentiles below τ)? | 95 % with all six RULE-T conditions (T classes are classifications outside K; this is less strict than the 98.75 % alternative, which would make both REACHED and RULED_OUT harder and match the deciding level); 98.75 % printed; the user may choose 98.75 %; precedence POINT_ONLY, never counted as attainment, both percentiles printed (resolved at the default, S2/S4) |
| O8 | G14_ZERO (and PAIRMEAN) inside the H2/H3 IUT, or reported only? | inside; component verdicts printed separately |
| O9 | Learner grid? | §3 hyperparameter grid K1–K4 on LEAN209 + METAL_BASIC2; alternative: the feature-set grid |
| O10, O11 | C selected on its own target (inner MET-20) or on inner strict-adjacent log SF MAE? Primary C learner centred-row (tested) or the antisymmetric pair arm (never run)? | own target, C_selSF printed beside; centred-row, C_PAIR exploratory after a smoke test |
| O12 | Inner selection on the 4-fold cross-fit or on gen20's single split? | 4-fold cross-fit; inner sizes printed |
| O13, O14 | Primary log SF population C3 (229 / 1,802) or all strict pairs (378 / 2,114)? Primary log D population C3 entries (2,425) or all 3,359? | C3 for both; the others co-reported with identical columns; COH-D-ALL cannot fulfil OUT1 |
| O15 | Exclude the 5 DMDPhPDA cells from the primary? | keep; COH-NODMD refit with flip reporting (keeps gen20 comparability; biases log D results, and the 5 M Pr/Nd pair, toward larger error) |
| O16 | Can replicate readings be paired by run to estimate log SF noise? | checked at S3; else "not estimable" plus the labelled bound |
| O17 | Compute descriptive R² noise ceilings? | not computed (formulas such as 1 − σ̂²·mean(1/n_i)/var0(y) on request) |
| O18 | Compute budget: 12 h cap vs the six-design requirement | per-design invocations; excess recorded; never dropped silently; a design never run gives INSUFFICIENT_COMPUTE when RULE-C (a)–(g) hold on design BP, otherwise that (h) counts as failed (`PRE_REGISTRATION.md` §8.1) |
| O19 | Seal footer and addenda mechanism; is this plan, which `PRE_REGISTRATION.md` cites for formulas (MET-02..MET-24), fold gates, the deny list, the MODE_M specification and the compute estimate, frozen with it at S4? | gen20-style last-line seal over LF-normalised text; addenda in a dated `ADDENDA.md` listed in the manifest; propose `generations/gen20_1/.gitattributes` `*.md text eol=lf` at seal time; the plan is frozen with it: its LF-normalised digest is recorded in `manifests/seal_g201.json`, so any later change is a dated addendum (resolved at the default, S2/S4) |
| O20, O21 | Recovery-plan estimand-consistency clauses required or annotations? WCR threshold 0.0125 or gen20's 0.05? | required (RULE-C (i)–(k)), failure → ESTIMAND_INCONSISTENT; 0.0125 |
| O22 | A separate registered Pr/Nd claim? | descriptive (MET-24) plus the mandatory PASS annotation |
| O23 | Single-metal training augmentation for A? | not run |
| O24 | Which estimand and mode were the 0.4 / 0.5 thresholds meant for? (the repository's only R² target is the recovery plan's R2_balanced(k = 3) ≥ 0.50, :287) | descriptive, at k = 0, on R2_pooled (both targets) and R2_within_cell (log D), via RULE-T |
| O25 | May local, gitignored gen13 prediction files be read for informative continuity checks? | read if present; the hard gate is the in-process refit LT-13 |
| O26 | LEAN209 was chosen in gen13 on outer BP scores over these same folds (`generations/gen13_separation/analysis/stage2/STAGE2_REPORT.md:95-110,296-297`). Keep it fixed, or select block sets inside each training fold? | keep LEAN209 fixed (no gen20.1 outer choice, no added degrees of freedom); every H and T verdict and the confirmation label carry "feature set fixed from gen13 outer results on these folds" |
| O27 | Log D rows for cross-target statements: all C3 entries (COH-D-C3, 2,425) or only those underlying the scored log SF pairs (COH-D-C3M, 2,208), per the brief's matched held-out data ([BRIEF]:42)? | both printed in TAB-07; H1, T1 and T1w stay on COH-D-C3 |

## 12. Risks

| ID | risk | mitigation |
|---|---|---|
| RK1, RK2 | pooled log D R² crosses τ on the extraction level alone and is read as selectivity skill ([BRIEF]:32); cross-target R² ranking | T1 with T1w and I1, level-attribution label, "reached also by baseline" sentence, I2/I5 printed, separation claims from the log SF table only; no cross-target statistic, fixed card |
| RK3 | about 4 effective chemotypes (Kish 3.85 pairs / 3.43 rows; sc009 49 % of pairs; giant folds of 76–98 training cells, each predicting 52–64 % of its seed's C3 pairs; [CNT]) | MET-01 primary; Kish printed; sc009 breakdowns; publication unit required; RULED_OUT vs NOT_REACHED; per-fold sizes |
| RK4 | percentile bootstrap over-rejects (0.071–0.077 at 0.05 in gen13's 45-cluster simulation; unmeasured at Kish 3.85); R² intervals uncalibrated | WCR at α_K; RULE-T needs percentile, BCa, publication, seed and balanced conditions; NC1 |
| RK5, RK6 | selection near-ties or an outer-score choice; estimand switching after results | nested per-fold argmin; IUT; per-config scores not selectable; MET-22; one primary estimand and population per target; dated addenda only |
| RK7 | DMDPhPDA copy and decade-level replicate disagreements inflate log D error | MET-01 primary; COH-NODMD refit; COH-REPCAP; flip reporting |
| RK8 | tree learners tie adjacent metals (exact-zero SF predictions) | 0.5 tie rule for all arms; n_zero_pred printed |
| RK9, RK10 | noise estimate mistaken for measurement error or √2·sd for SF noise; underpowered claims read as evidence | labelled training-fold estimators, run-paired SF noise only, no ceiling; level-consistent and CR2-t MDEs, UNDERPOWERED labels, EQUIVALENT only from intervals |
| RK11 | target statistics seen before the seal steer the protocol | S5 after S4; hashed manifest; addenda state what was seen |
| RK12 | discovery folds are not blind (gen20 used them; gen13 chose the LEAN block set on outer BP results on them, STAGE2_REPORT.md:95-110; O26) | stated; hyperparameter-only grid; confirmation labelled a partition-robustness check |
| RK13, RK14 | macOS→Windows numeric drift; writes into frozen generations (gen20k writers, runner.run (writes relative labels into gen20_kshot/predictions), gen14 cache rebuild, gen13 import-time mkdir, bytecode) | graded tolerances, LF-normalised hashes, platform recorded; §8 lists, LT-26, LT-27, dont_write_bytecode |
| RK15, RK16 | compute overrun on 8 GB; the k = 3 measured-mode 0.31 or a changed-task "0.4+" value used as the bar for a k = 0 result | one process, fits discarded, per-design resume, reduction rule, peak RSS at S8; TAB-14 with task changes, CTX_RIDGE2_K3 context only, anchors the only comparators |
| RK17, RK18 | H4 dismissed because B is tuned on log D and C on its own target; synthetic arrays or smoke output mistaken for evidence | B_selSF / C_selSF / C_ACFG decomposition beside H4 (EX-01..EX-04); synthetic data only in tests/, `smoke*` refused by report code |
