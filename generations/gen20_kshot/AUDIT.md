# Gen20 — Phase 1 audit

*2026-10-14. Part A was verified by direct execution on this machine (macOS, Python 3.13.0,
scikit-learn 1.9.0, pandas 3.0.5, numpy 2.5.1, torch 2.13.0). Part B condenses the eight auditor
reports and the completeness critique (full JSON in `manifests/audit_reports.json`). Nothing in this
file is a gen20 result.*

## Part A — verified by direct execution

### A.1 Anchors reproduce

`.venv/bin/python -m pytest generations/gen16_leads/tests/test_anchors.py -q -m "not slow"` →
3 passed in 5 s: fold-plan SHA-256 in a subprocess, `G14` under BP 0.5000794414203691, `FLAT`
under BP 0.5885062528901843, and the recorded `G14_vs_FLAT` contrast digits. The gen14 bench cache
(`generations/gen14_direction/cache/bench.pkl`, gitignored) was built in 1 s.

### A.2 Gen15 measured-mode board reproduces

`gen15.fewshot.evaluate(bench, {G14, FLAT}, "BP", ks=(0,1,2,3), hows=(widest, dopt, random),
cov_kind="empirical", mask_publication=True)` (18 s, 57 638 scored pairs over 321 cells, 84
extractants; scratch output only):

| mode | macro MAE all | macro MAE adjacent (gen13 def.) | macro MAE far |
|---|---|---|---|
| `G14@doptk3` | **0.170039** (gen15: 0.1700) | 0.139 | 0.192 |
| `FLAT@doptk3` | 0.175665 | 0.145 | 0.199 |
| `NAIVE_LINE@doptk3` | 0.234656 (gen15: 0.2347) | 0.159 | 0.272 |
| `G14@widestk1` | 0.226054 (gen15: 0.2261) | 0.154 | 0.267 |
| `G14@k0` | 0.424242 (gen15: 0.4242) | 0.184 | 0.619 |
| `FLAT@k0` | 0.502550 (gen15: 0.5025) | 0.199 | 0.762 |

Two readings that shape gen20: after three D-optimal pairs the ligand chemistry is worth 0.006
(`G14` vs `FLAT` with the same BLUP), and on gen13-adjacent pairs the ladder runs 0.199 → 0.139,
so measurements do reach the adjacent regime. Note gen15's protocol has no arithmetic-closure
rule and its scoring set is all non-support pairs.

### A.3 Cohort facts used by the design (counted on the bench)

* 521 cells, 90 extractants, 45 chemotypes, 58 publications; `Y` is log10 D, range −4.99…4.21;
  `groups == chemotype`; `n_obs == n_metals`; LEAN blocks = 209 columns, 2.4 % NaN.
* Cells by metals: 2:138, 3:62, 4:32, 5:21, 6:72, 7:8, 8:57, 9:11, 10:4, 11:2, 12:1, 13:35, 14:78.
* Extractants with ≥ 2 cells: 26 of 90; with ≥ 2 publications: 9.
* Chemotype `sc009` (diglycolamides) holds 375 cells; chem_family diglycolamide 426 cells. Under BP
  seed 104729 fold 0 the training set is **76 cells** (test 418).
* Strict adjacent pairs (|ΔZ| = 1): 378 cells carry ≥ 1, 2 114 pairs in all; `y` sd 0.231,
  mean −0.089, `FLAT` MAE 0.167, median |y| 0.116. Per pair type (cells / extractants /
  chemotypes / publications): La/Ce 187/68/37/39, Ce/Pr 169/68/36/36, Pr/Nd 172/71/35/38,
  Sm/Eu 203/77/36/39, Eu/Gd 177/70/33/36, Gd/Tb 188/65/30/32, Tb/Dy 181/66/31/33,
  Dy/Ho 185/57/28/30, Ho/Er 188/57/28/31, Er/Tm 153/55/28/28, Tm/Yb 147/55/28/28,
  Yb/Lu 164/54/26/28.
* No-closure feasibility (supports = non-adjacent pairs, forest, no component may contain both
  metals of an adjacent pair; bipartite-lookahead merge rule): realised `k_max` equals the
  theoretical `n_metals − 2` on every one of the 378 cells. Eligible at k = 0/1/2/3/5/8/10:
  cells 378/331/302/275/196/120/114; extractants 82/82/82/81/74/51/50; chemotypes
  40/40/40/39/34/26/25; publications 45/44/42/40/36/25/25; adjacent query pairs
  2114/2067/2032/1980/1730/1337/1297.

### A.4 Compute

A `TabularNet`-style head (LayerNorm → 256 → 128 → 1, SiLU, dropout 0.15) with a within-cell
pairwise Huber contrast loss on one BP fold (764 (cell, metal) rows, 4 097 pairs) trains 60 epochs
in 0.4 s on CPU. Compute is not a constraint; the 76-cell training fold is.

### A.5 Reference repository (read-only, nothing executed)

Commit `025f8084` (2026-09-12), 6 081 files, 404 MB. Per-metal `raw_data/*_SAFE.csv` (lanthanides
plus Y, Sc-free, actinides and others), `automl/` (CatBoost/LightGBM AutoML, anchored
level/shape regressors, `topo/` with eight torch encoders and one training loop), `scripts/`,
`slurm/`, `reports/`. Headline: pooled adjacent-pair log SF R² +0.326 under leave-extractants-out
5 × 3 CV (not chemotype, not publication-masked). See `IMPLEMENTATION_PLAN.md` §1.5 for the
components and the compatibility verdict.

### A.6 Cohort fingerprint on this machine

`build_cohort("exact").fingerprint()` gives `854171b03411d901` here against the frozen
`4c3c6628ea0be949` (the README's known gen16 L6 failure). Compared column by column with
`manifests/cohort_exact.parquet` (SHA-256 `3f11ac4c…`): identical shape, columns, dtypes, identity
columns and every `logD__` value; **eight numeric columns differ at the last bit**
(`replicate_sd_median`, `repsd__La/Pr/Sm/Tb/Dy/Yb/Lu`, at most 2.2e-16, 1–3 cells each; found by
the completeness critic, who also showed that the stored parquet re-hashes to the frozen value under
this pandas). The cause is floating-point drift in the replicate sd under numpy 2.5.1, not string
hashing as an earlier draft of this file said. None of these columns is a feature (they are in
`FORBIDDEN_FEATURE_TOKENS`); `repsd__` enters gen20 only through the training-replicate noise
variance, where a 1e-16 change is immaterial. Gen20 records the manifest parquet SHA-256 and the
fingerprint obtained, and `gen20k.data.assert_matches_manifest` asserts column-wise equality at
1e-9 at load time. The bundle SHA-256 `fefbefc6…` is pinned in `gen3_protocol.json` and re-checked
by `gen13sep.paths`.

### A.7 Corrections from the completeness critic (manifests/audit_reports.json, `critique`)

* The 7.5 % closure figure (963 of 12 863) is a structural per-cell recount under gen15's
  inclusion rule, not a fraction of gen15's scored rows (57 638 under BP, three-strategy union);
  gen20's replication reports the closed fraction on gen15's own scored rows.
* gen15's boards do carry `macro_mae_adjacent` (gen13 definition); what does not exist anywhere is
  an out-of-sample R² of adjacent log SF.
* The strict adjacent count is 2 114 (gen13 DATA_AUDIT.md's 2 115 is a pre-quarantine figure).
* Three adjacency contracts now exist (gen13 PRE_REGISTRATION text dZ ≤ 2; gen13 code dZ = 1 plus
  Nd–Sm; gen20 strict dZ = 1); gen20 names its choice in PRE_REGISTRATION.md §2.
* Pr/Nd is measured in 172 cells / 71 extractants / 35 chemotypes / 38 publications (sd 0.234), so a
  Pr/Nd row of every per-pair table exists; gen19's V5-PAIR halves contain no Pr/Nd pairs.

## Part B — auditor findings

*(filled in from the audit workflow; see below)*

Eight auditors (workflow `gen20-kshot-audit`, 2026-10-14) plus a completeness critic; the full
structured reports are in `manifests/audit_reports.json`. Condensed, with the consequence for gen20.

### B.1 Data and target
Bundle 5 992 rows × 2 261 columns, `log_D` exactly log10(D), 14 lanthanides, no Pm. Cell =
SMILES @@ publication @@ exact 64-column key (NaN is a value; floats at 9 s.f.); replicates averaged
with counts and sd. Pairs per seed: 14 150 all, 2 348 adjacent (gen13 definition), 2 114 strict,
6 698 far. Lighter-first orientation in gen13 (heavier preferred ⇔ y < 0); gen19 is heavier-first.
sc009 holds 375/521 cells, 23/90 extractants, 25/58 publications, 53.6 % of adjacent pairs; TODGA
alone 171 cells / 18 publications / 25.8 % of adjacent pairs. 81 of 90 extractants come from one
publication. Replicate floor: 216 replicated (cell, metal) combinations, median sd 0.302.
Consequence: strict adjacency primary; every pooled number printed beside extractant- and
chemotype-macro; the 0.1/0.3 direction thresholds declared.

### B.2 Splits and leakage
Designs share test sets per seed (B/BP/BR/BQ) and differ in training rows; the BP mask is per fold;
inner split re-dealt on the masked set (only its first inner fold is stored, so gen20 re-deals all
four with the same seed formula, verified equal on the stored one). `assert_fold_integrity` fires the
publication test only for pure-BP sets and the frozen benches never call it: gen20 re-asserts every
design. 7 ECFP-identical groups (20 extractants) and the stereo collapse (90 → 88) stay inside
chemotypes. 45 cells / 5 publications / 3 copy groups span publications → design BPG. Closure: 963
of 12 863 gen15 widest-3 scored pairs (7.5 %) are support-determined. gen15's covariance uses other
test folds' residuals; mask opt-in. `hash()` of int tuples is unsalted (verified across processes);
`arms_stage2.extractant_key` hashes bytes (salted) but is off the bench path. gen14's `C = 1` and
gen15's policy were outer-selected.

### B.3 Existing k-shot implementations
One pairwise lineage (gen13 stage 2 → gen15 → gen16 L5): BLUP on the centred curve, NOISE_VAR 0.09,
widest/dopt/random, fixed scoring set, mask_publication; numbers in three non-interchangeable pair
sets (§4 widest-only: G14 0.4387 → 0.2314 → 0.2235 → 0.2058; §5 common set: dopt k3 0.1633–0.1700,
NAIVE_LINE 0.2347; §7 training-fold covariance: POOLED k3 0.1922). One measurement is worth +0.207
(passes P1); chemistry after one measurement +0.008 (p 0.35); the learned covariance after one
measurement +0.006 (p 0.18); D-optimal − widest at k3 +0.033 (passes in all designs); G14 − NAIVE
at dopt k3 +0.064 (p < 1e-4). Gen16 L5: covariance indefinite in 100 % of 430 estimates, 90 %
intervals cover 98.8 %, Ledoit–Wolf catastrophic (34.4 at k3), low-rank PSD alternatives +0.015
at k2 only. The log D lineage (gen7–gen12) shares no code. gen4's `kshot_calibration.py` is the
only k-shot code computing R² (older 34-extractant cohort). Max k ever evaluated on pairs: 3.

### B.4 Zero-shot arms
G14 = L2 logistic (C 1, lbfgs) on 39 TOPO39 columns behind median imputer + StandardScaler →
sign × training-fold balanced mean |amplitude| over well-determined cells, curvature = balanced
mean; basis = standardised Shannon CN8 radius and its square at row norm √14; per-cell labels by
ridge 0.5 on observed metals. Block widths: COND 64, MASSACT 8, PHYSCHEM 10, DONORS 13, COORD 114
(LEAN 209), ECFP 2048, LIG2D 206 (both harmful). Anchors reproduce here (G14 0.5000794414203691,
FLAT 0.5885062528901843, fold-plan SHA `7046c640…`). Five-design G14 range 0.0095. Costs: logistic
0.03 s/fold, gen13 trees ~0.6 s/fold, V2_BAG4@lean ~4 s/fold. gen19 M0 is not a drop-in
log D-then-subtract baseline (different corpus, unit, hold-out, identity); `C_DIRECT_ROW` is that
baseline inside the gen13 stack (0.583 under BP) and is not re-run here.

### B.5 Metrics and statistics
Extractant-macro MAE pools an extractant's cells; chemotype-macro and pooled printed beside; sign
accuracy counts a zero prediction wrong (gen19 and the recovery plan: ½). `paired_contrasts`
averages per unit over seeds before resampling, tests the lower bounds only, LOCO same-sign; BCa
clamps z0. Calibration: percentile rejects 0.071–0.077 at nominal 0.05 (3 000 sims, ICC 0.72 and
0.3, normal and t3), wild-cluster-t CR2/CR3 0.044; G14 − FLAT under BP has percentile p 0.047 and
WCR p 0.059. Kish 11.7 chemotypes (10.1 on the 289 rich cells); Bell–McCaffrey dof 11.1 / 9.6.
No R² on log SF anywhere; `cluster_bootstrap_statistic` (gen19) is a per-draw Python loop.
Consequence: `gen20k/inference.py` (sums-based, per seed, BCa mid-rank, strict LOCO, WCR beside).

### B.6 Reproducibility
Generation layout and sys.path convention followed; gen19's `manifest.Run` pattern adopted in
`gen20k/manifest.py` (git, environment, dataset digests, seeds, outputs, pre-registration seal);
gen16's seal (SHA-256 footer) and public confirmation-seed rule adopted; parquet outputs over the
size budget are gitignored and digested; compute cap 12 h, ≤ 2 workers, ≤ 4 threads
(`LOKY_MAX_CPU_COUNT`, `OMP_NUM_THREADS`, torch threads set in `bootstrap.py`). The recovery plan
names `gen20_separation/`; this study is `gen20_kshot/` and says so.

### B.7 Setting B feasibility
25 extractants have ≥ 2 cells in one publication (17 ≥ 3, 15 ≥ 5, 11 ≥ 10); 41 (extractant,
publication, identical metal set) groups with ≥ 3 cells over 15 extractants / 8 chemotypes; most
multi-cell groups vary ≥ 2 condition families (grids, not titrations). Five identical-log-D cell
pairs split only by a NaN condition (THDGA, TODGA, T2EHDGA in pub_b9dfc468cd; DMDODGA; TBDGA). The
leave-one-cell-out COPY scores 0.098 pooled / 0.108 macro vs FLAT 0.160 / 0.150; the single
nearest-acid support-cell copy 0.105 / 0.116; median cross-condition change 0.043, 91 % below 0.3.
Cross-publication variant: 9 extractants / 4 chemotypes. gen8's series-local calibration was a
log D result; in log SF the level cancels. Verdict adopted: exploratory only, COPY null.

### B.8 Reference repository
See IMPLEMENTATION_PLAN.md §1.5: the upstream dataset builder plus the collaborator's AutoML;
bundle byte-identical; no new lanthanide data; six torch architectures and one tabular control;
contrast objective and anchored level/shape decomposition; leave-extractants-out only; no few-shot
mechanism; nothing runs as checked out. Adaptable: the tabular head + contrast objective (adopted),
`pair_model.py` (CatBoost on within-block pairs) and the anchored champion (not adopted: the
recovery plan's CatBoost candidate is a separate study).
