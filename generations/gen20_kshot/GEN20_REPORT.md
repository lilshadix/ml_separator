# Gen20 — leakage-safe k-shot learning for adjacent-lanthanide separation

*2026-10-15. Every number quoted here is traceable to a CSV under `results/discovery/`,
`results/discovery_light/`, `results/discovery_nd_sm/` or `results/confirmation/`, and is listed with its
source row in `tables/report_numbers.csv`. Registered claims and verdicts: §C.3 and §C.11.*

## A. Research question

Can k experimental measurements significantly improve the prediction of adjacent-lanthanide
separation factors (`log10 SF`) for extractants of a chemotype the model has never seen, and does
any model earn anything beyond a straight line through the same measurements?

## B. Methodology

### B.1 Data

The frozen gen13 cohort of the repository's experimental bundle (`dataset with 3D structures/
dataset.parquet`, SHA-256 `fefbefc6…`): 521 cells, 90 extractants, 45 chemotypes (frozen gen6
single-linkage Tanimoto-0.7 superclusters of ECFP bits), 58 publications. A cell is one extractant
measured under one exactly matched 64-column condition key inside one publication, with replicates
averaged per (cell, metal). The bench used here is asserted column-wise equal to
`gen13_separation/manifests/cohort_exact.parquet` (SHA-256 `3f11ac4c…`) at load time.

Target: `y(A/B) = log10 D(A) − log10 D(B)` with A the lighter metal, inside one cell. Primary query
population: strict adjacent pairs (|ΔZ| = 1; 12 pair types; promethium is absent and Nd/Sm is not
adjacent). gen13's definition (adjacent plus Nd–Sm) is run as a separate labelled pass. Adjacent `y`
has sd 0.231 and mean |y| 0.167 (FLAT MAE) over 2 114 pairs; the replicate floor is a median
per-(cell, metal) sd of 0.30 log D, so adjacent separation factors sit inside the measurement
noise of the levels they are computed from.

### B.2 Hold-out and leakage prevention

Chemotype-grouped 5-fold hold-out × 5 split seeds (gen13's), with the publications of the held-out
cells removed from training (design BP, the selecting design), plus designs B (no publication
mask), BPG (BP plus gen19's cross-publication copy groups), BR, BQ (size- and structure-matched
random controls) and A (exact extractant, a leak price). Every fold set is re-checked for
chemotype, extractant, publication and copy-group crossings and for inner folds staying inside the
training fold. All learned objects (imputation, scalers, the neural network, the residual
covariance, the ridge penalties, the noise factor, the neural configuration and its k-shot
mechanism) are fitted or chosen on the training fold; selection uses each fold's inner
chemotype split with the design's mask re-applied inside it and a covariance nested on the
inner-training cells. The validity suite (`tests/`, 35 tests) runs the fold with query labels
masked and checks predictions are unchanged, checks the output tables for support rows and for
query identity across budgets and policies, spies on the selector, asserts covariance provenance,
reproduces fold plans, support draws and predictions in a subprocess, and injects deliberate
violations that must raise.

### B.3 k-shot setting

Two measurement models (PRE_REGISTRATION.md §3). **Metal model (registered):** a measurement is
one metal's log D in the cell; after k + 1 read metals every separation factor among them is known;
the query is every strict-adjacent pair of the cell, fixed before any model runs and identical for
every budget, arm and policy; the read set therefore never contains both metals of an adjacent pair
(k_max = 6 on a 14-metal cell). **Pair model (the brief's definition, exploratory):** a measurement
is one separation factor; supports form a forest of non-adjacent pairs with per-pair closure
excluded; supports still fix linear combinations of the adjacent contrasts, so rows carry
`lab_known` and the arithmetic-only `PROJ` arm is scored beside every model.

Budgets k ∈ {0, 1, 2, 3, 5, 6, 8, 10} (the metal model reaches 6). Policies, all label-free:
RANDOM (5 draws), WIDEST (pair model), UNIFORM (farthest-point coverage of Z), DOPT_CURVE and
DOPT_Q (greedy information gain under the G14 inner-OOF covariance, for the whole curve or the
query), and UNC_SEQ (sequential, neural-ensemble disagreement; label-dependent through the
supports only). Common-query cohorts: metal C3 (229 cells, 79 extractants, 38 chemotypes), C5
(144/64/29), C6 (133/58/27); pair C3 (275/81/39), C5 (196/74/34), C10 (114/50/25).

### B.4 Models

Model 1 (existing baseline): gen14's deployed `G14` (L2 logistic direction on 39 donor-topology
counts × training-fold magnitude and curvature; reproduced to the anchor 0.5000794414203691 under
BP) with the residual BLUP measured mode (`G14@BLUP20`: covariance from inner cross-fits of the
masked training fold, PSD-projected; shared-reading noise `R = s² D Dᵀ`, `s²` the squared median
replicate sd over readings disagreeing by < 0.5 log units times an inner-chosen factor). Model 2:
`NNC`, the reference repository's tabular control architecture with its pairwise-contrast
objective on the gen13 feature blocks plus metal descriptors, masked-validation stopping then a
refit on every training cell, 8 seeds; mechanisms `BLUP20`, `HEAD_FT`, `FULL_FT`; configuration,
mechanism and penalty chosen on inner folds (`NNC@SELECTED`). Model 3: `FLAT`, `FLAT@BLUP20`
(the covariance with no chemistry), `LINE` (gen15's naive line), `PROJ` (arithmetic only),
`RIDGE2` (two-parameter curve shrunk to the training prior), `MEAN_CURVE`, `NN_TANIMOTO` (+ rescale,
+ BLUP). The claim-carrying arm `M*` is decided from inner selection records alone.

### B.5 Metrics and statistics

Pooled R² (NA with reason, never clipped), MAE, RMSE, direction accuracy at |y| > 0, ≥ 0.1, ≥ 0.3
(a zero prediction scores 0.5), extractant- and chemotype-macro MAE, per-pair-type MAE and R², all
per split seed and averaged; chemotype-blocked bootstrap over per-cluster sums (10 000 draws,
percentile and BCa at 95 and 98.75 %), leave-one-chemotype-out, per-seed signs, wild-cluster-t p on
per-extractant deltas, minimum detectable effect; Benjamini–Hochberg over the counted exploratory
family. Registered claims (P1–P4) on the metal model, design BP, policy DOPT_Q, extractant-macro
MAE (P3: pooled ΔR²), Bonferroni over four.

## C. Results

Every number below is design BP, strict adjacent pairs, policy DOPT_Q and the metal measurement
model unless stated; pooled statistics are over query rows, "macro" is one vote per extractant,
intervals are chemotype-blocked bootstrap percentiles (10 000 draws; BCa and the 98.75 % level in
the CSVs), and the source CSV under `results/discovery/` is named with each table. Five discovery
seeds; per-seed values are in every CSV.

### C.1 Anchors and the gen15 replication

The gen16 anchors reproduce to the digit on this machine (`G14` 0.5000794414203691, `FLAT`
0.5885062528901843, fold-plan SHA-256 `7046c640…`). gen15's measured-mode board reproduces with
gen15's own code (`results/gen15_replication/board.csv`): `G14@doptk3` 0.1700393901, `NAIVE_LINE@
doptk3` 0.2346556469, `G14@widestk1` 0.2260538009, `G14@k0` 0.4242420252, `FLAT@k0` 0.5025495472.
On gen15's own 57 638 scored BP rows, 6.7 % of all pairs and **24.8 % of the adjacent pairs** are
exactly determined by the three widest supports (random supports: 3.1 % and 5.0 %);
`closure_summary.csv`. Those rows were scored as predictions in gen15. Gen20's query set contains
no such row by construction.

### C.2 Eligibility at each budget (`cohort_counts_BP_*.csv`, per-seed means)

| model | k | cells | extractants | chemotypes | publications | query pairs | Kish chemotypes (pair-weighted) | largest chemotype share |
|---|---|---|---|---|---|---|---|---|
| metal | 0–1 | 331 | 82 | 40 | 44 | 2 067 | 3.4 | 0.53 |
| metal | 2 | 291 | 81 | 39 | 40 | 2 002 | 3.5 | 0.51 |
| metal | 3 | 229 | 79 | 38 | 39 | 1 802 | 3.9 | 0.49 |
| metal | 5 | 144 | 64 | 29 | 29 | 1 405 | 4.2 | 0.46 |
| metal | 6 | 133 | 58 | 27 | 26 | 1 349 | 4.2 | 0.46 |
| pair | 3 | 275 | 81 | 39 | 40 | 1 980 | 3.6 | 0.51 |
| pair | 5 | 196 | 74 | 34 | 36 | 1 730 | 4.1 | 0.47 |
| pair | 8 | 120 | 51 | 26 | 25 | 1 337 | 4.3 | 0.45 |
| pair | 10 | 114 | 50 | 25 | 25 | 1 297 | 4.5 | 0.44 |

The metal model never reaches k = 8 or 10 with the query fixed (a 14-metal cell supports six
independent separation factors without reading both metals of an adjacent pair). The effective
number of chemotypes behind every pooled number is about four; the diglycolamide chemotype
(`sc009`) holds about half of the query pairs at every budget.

### C.3 The registered claims (`registered_contrasts_BP.csv`; Bonferroni 98.75 % intervals decide)

The registered arm rule picked the neural arm: `NNC@SELECTED` beat `G14@BLUP20` on the inner k = 3
ladder in 13 of 25 BP selections (`folds_BP.json`, `mstar_inner`), a near tie that the rule
resolves by majority. The pre-registered procedure was followed; the same contrasts for the other
arms are reported beside it as exploratory (`registered_style_contrasts_other_arms_BP.csv`).

| id | contrast (M* = NNC@SELECTED) | cohort | point | 98.75 % percentile | 98.75 % BCa | p (percentile) | p (wild-cluster-t) | MDE₈₀ | seeds + | LOCO all + | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | M* k=3 vs LINE k=3, macro MAE | C3 | **−0.032** | [−0.055, −0.018] | [−0.057, −0.019] | < 1e-4 | 0.0003 | 0.019 | 0/5 | no | **fails: the line is better** |
| P2 | M* k=3 vs k=0, macro MAE | C3 | **+0.034** | [+0.016, +0.073] | [+0.016, +0.073] | < 1e-4 | 0.0001 | 0.031 | 5/5 | yes | passes (margin 0.02 met) |
| P3 | M* k=3 vs k=0, pooled ΔR² | C3 | **+0.194** | [+0.12, +0.50] | [+0.09, +0.41] | < 1e-4 | — | 0.21 | 4/5 | yes | **fails: R²(k=3) = −0.13 < 0** |
| P4 | M* k=6 vs LINE k=6, macro MAE | C6 | **−0.017** | [−0.033, −0.007] | [−0.035, −0.008] | 0.0002 | 0.0009 | 0.013 | 1/5 | no | **fails: the line is better** |

So, under the registered procedure: measurements improve the registered arm (P2), but the
registered arm does not reach a positive R² (P3) and is significantly *worse* than a straight line
through the same three or six measurements (P1, P4). The MDEs (0.013–0.031) are below the 0.02
margin for P1 and P4 and just above it for P2, so the negative verdicts are not power artefacts.
The confirmation run (C.11) repeats all four verdicts.

The same four contrasts for the arms the rule did not pick (exploratory, same rows, same supports):

| arm | P1-style (vs LINE, k=3) | P2-style (k=3 vs k=0) | P3-style ΔR² [R²(k=3)] | P4-style (vs LINE, k=6) |
|---|---|---|---|---|
| `G14@BLUP20` (gen15's deployed measured mode, new covariance) | +0.003 [−0.005, +0.015], p 0.41 | +0.022 [+0.014, +0.036], p < 1e-4 | +0.195 [+0.15, +0.30], R² 0.25 | +0.015 [+0.004, +0.026], p 0.001 |
| `RIDGE2` (two-parameter curve shrunk to the training prior) | **+0.008 [+0.0001, +0.017], p 0.012, WCR p 0.031** | +0.042 [+0.022, +0.059] | +0.426 [+0.23, +0.54], R² 0.31 | **+0.016 [+0.008, +0.029], p < 1e-4** |
| `FLAT@BLUP20` (corpus covariance, no chemistry) | −0.005 [−0.015, +0.009], p 0.59 | +0.026 [+0.017, +0.041] | +0.324 [+0.22, +0.42], R² 0.19 | +0.010 [+0.001, +0.023], p 0.003 |
| `NNC@BLUP20` | −0.030 [−0.053, −0.014] | +0.035 [+0.018, +0.075] | +0.282, R² −0.04 | −0.006 [−0.025, +0.006] |

Only `RIDGE2` beats the line with intervals excluding zero at both budgets, and its gain at k = 3
(+0.008) is below the programme's 0.02 margin; at k = 6 it is +0.016, also below the margin. The
deployed gen15 measured mode is indistinguishable from the line at k = 3 (+0.003) and below the
margin at k = 6 (+0.015).

### C.4 R², MAE, RMSE and direction versus k (`levels_BP_metal_C3_DOPT_Q.csv`, C3: 229 cells, 79 extractants, 38 chemotypes, 1 802 adjacent pairs, FLAT MAE 0.164, y sd 0.236)

| k | arm | R² [95 %] | MAE [95 %] | RMSE | macro MAE (extractant / chemotype) | direction acc. (all / |y| ≥ 0.1 / ≥ 0.3) |
|---|---|---|---|---|---|---|
| 0 | FLAT | −0.135 [−0.21, −0.01] | 0.164 [0.132, 0.198] | 0.236 | 0.173 / 0.164 | 0.50 / 0.50 / 0.50 |
| 0 | G14 (zero-shot) | 0.053 [0.00, 0.12] | 0.147 [0.125, 0.178] | 0.216 | 0.161 / 0.155 | 0.71 / 0.78 / 0.80 |
| 0 | NNC (zero-shot) | −0.322 [−0.61, −0.23] | 0.177 [0.153, 0.219] | 0.255 | 0.208 / 0.222 | 0.58 / 0.59 / 0.59 |
| 1 | LINE | 0.168 [0.09, 0.25] | 0.137 [0.115, 0.169] | 0.202 | 0.146 / 0.133 | 0.73 / 0.80 / 0.83 |
| 1 | G14@BLUP20 | 0.118 [0.07, 0.23] | 0.141 [0.116, 0.167] | 0.208 | 0.149 / 0.135 | 0.69 / 0.78 / 0.83 |
| 2 | LINE | 0.237 [0.17, 0.30] | 0.130 [0.112, 0.156] | 0.194 | 0.143 / 0.132 | 0.74 / 0.82 / 0.86 |
| 2 | RIDGE2 | 0.279 [0.19, 0.37] | 0.124 [0.106, 0.152] | 0.188 | 0.136 / 0.124 | 0.76 / 0.84 / 0.87 |
| 3 | LINE | 0.236 [0.17, 0.30] | 0.130 [0.111, 0.156] | 0.194 | 0.143 / 0.132 | 0.75 / 0.82 / 0.84 |
| 3 | RIDGE2 | **0.310 [0.22, 0.40]** | **0.121 [0.101, 0.149]** | 0.184 | 0.135 / 0.124 | 0.77 / 0.86 / 0.88 |
| 3 | G14@BLUP20 | 0.248 [0.20, 0.34] | 0.129 [0.108, 0.149] | 0.192 | 0.140 / 0.129 | 0.73 / 0.82 / 0.88 |
| 3 | FLAT@BLUP20 | 0.189 [0.12, 0.31] | 0.135 [0.110, 0.155] | 0.200 | 0.147 / 0.133 | 0.71 / 0.80 / 0.90 |
| 3 | NNC@SELECTED | −0.128 [−0.23, −0.03] | 0.162 [0.135, 0.192] | 0.235 | 0.174 / 0.172 | 0.67 / 0.74 / 0.78 |
| 3 | NN_TANIMOTO@RESCALE | 0.055 [−0.00, 0.15] | 0.148 [0.122, 0.174] | 0.216 | 0.157 / 0.144 | 0.64 / 0.71 / 0.78 |
| 3 | PROJ (arithmetic only) | −4.55 | 0.352 | 0.523 | 0.314 / 0.310 | 0.55 / 0.57 / 0.61 |

**gen13's adjacency (strict plus Nd–Sm; `results/discovery_nd_sm/`, a separate pass, 236 cells,
79 extractants):** adding the Nd/Sm pair (sd 0.38, FLAT MAE 0.39, the widest "adjacent" contrast)
raises every arm's pooled R² at k = 3: RIDGE2 0.43, G14@BLUP20 0.39, LINE 0.38, FLAT@BLUP20 0.35,
NNC −0.13; Nd/Sm itself reaches R² 0.66 (MAE 0.15 against FLAT 0.39) with three measurements. The
strict-adjacent numbers above are therefore the conservative ones. On that query set the arm rule
would have picked `G14@BLUP20` (12 of 25 inner wins), which shows how close the inner decision was.

Zero-shot, the deployed model explains 5 % of the adjacent variance (its gain over FLAT is +0.012
macro MAE, p 0.036, below the margin). One measured metal pair takes a model-free line to R² 0.17;
three take the ridge line to 0.31, the deployed measured mode to 0.25, the line to 0.24. Between
k = 2 and k = 3 nothing moves by more than 0.01 R² except the ridge line. The independent-noise
sensitivity (`G14@BLUP_INDEP`) is within 0.002 of `G14@BLUP20` everywhere. On C6 at k = 6 the
order is the same: RIDGE2 0.120 macro, G14@BLUP20 0.121, FLAT@BLUP20 0.127, LINE 0.136,
NNC@SELECTED 0.153 (`levels_BP_metal_C6_DOPT_Q.csv`).

### C.5 Model architectures on identical supports (`exploratory_contrasts_BP.csv`, 6 066 contrasts; 4 559 pass BH 0.05)

* **Neural (Model 2) versus the deployed baseline:** `NNC@SELECTED − G14@BLUP20` is −0.030 to
  −0.033 macro MAE at every k from 0 to 3 (95 % intervals entirely negative, 0/5 seeds, BH p < 1e-4).
  The neural arm is worse at k = 0 (−0.030) and stays worse after adaptation; `HEAD_FT` and
  `FULL_FT` do not change that (`NNC@HEAD_FT` R² −0.26 at k = 3). The inner selection preferred
  `BLUP20` as its mechanism in 17 of 25 folds.
* **Structure after supports:** `G14@BLUP20 − FLAT@BLUP20` is +0.017 at k = 0 (p 0.02) and shrinks
  to +0.011, +0.007, +0.006 at k = 1, 2, 3 (p 0.07–0.16, 5/5 seeds); molecular structure contributes
  within one measurement what the covariance already carries.
* **Model versus line:** `G14@BLUP20 − LINE` is −0.005, −0.004, +0.001 at k = 1, 2, 3 (p 0.4–0.7);
  `RIDGE2 − LINE` +0.008 [+0.002, +0.015] at k = 3 and +0.016 [+0.010, +0.027] at k = 6
  (`extra_macro_contrasts_BP_metal_DOPT_Q.csv`).
* **Transfer versus arithmetic:** every model beats `PROJ` by 0.09–0.22 macro MAE at k = 1–3; the
  measurements alone, without a curve prior, do not predict the unmeasured adjacent contrasts.

### C.6 Support-selection policies on identical cells (`levels_BP_metal_C*_{policy}.csv`; contrasts in `exploratory_contrasts_BP.csv`)

Macro MAE at the cohort's budget (C3 at k = 3, C6 at k = 6):

| cohort | arm | RANDOM | UNIFORM | DOPT_CURVE | DOPT_Q | UNC_SEQ |
|---|---|---|---|---|---|---|
| C3 | G14@BLUP20 | 0.140 | 0.136 | 0.138 | 0.140 | 0.135 |
| C3 | RIDGE2 | 0.140 | 0.132 | 0.132 | 0.135 | 0.133 |
| C3 | LINE | 0.144 | 0.141 | 0.142 | 0.143 | 0.142 |
| C6 | G14@BLUP20 | 0.122 | 0.117 | 0.124 | 0.121 | 0.120 |
| C6 | RIDGE2 | 0.121 | 0.120 | 0.120 | 0.120 | 0.120 |

Against RANDOM on identical cells at k = 3: UNIFORM +0.005 [+0.000, +0.008] (p 0.03) and UNC_SEQ
+0.004 [+0.003, +0.006] (p < 1e-3) for `G14@BLUP20`; DOPT_Q −0.001 (p 0.64) and DOPT_CURVE +0.001
(p 0.66). Under the metal model, which metals are read matters by at most 0.005 macro MAE: the
first two reads are always the lightest and heaviest legal metals, and that choice is what gen15
found to be worth +0.10 at k = 1 in the pair model. The pre-registered policy (DOPT_Q) was not the
best; the differences are below the margin.

### C.7 Individual adjacent pairs (`per_pair_BP_metal_C3_DOPT_Q_*.csv`; R² per pair type, pooled over its rows, mean over seeds)

| pair | n rows | extractants | y sd | FLAT MAE | G14 k=0 R² / MAE | G14@BLUP20 k=3 R² / MAE | LINE k=3 R² / MAE | RIDGE2 k=3 MAE |
|---|---|---|---|---|---|---|---|---|
| La/Ce | 137 | 60 | 0.30 | 0.27 | 0.02 / 0.22 | 0.24 / 0.18 | 0.30 / 0.18 | 0.17 |
| Ce/Pr | 125 | 61 | 0.35 | 0.27 | 0.01 / 0.24 | 0.26 / 0.19 | 0.19 / 0.20 | 0.18 |
| Pr/Nd | 127 | 63 | 0.24 | 0.20 | 0.05 / 0.17 | 0.31 / 0.14 | 0.42 / 0.13 | 0.14 |
| Sm/Eu | 158 | 71 | 0.24 | 0.21 | 0.09 / 0.18 | 0.26 / 0.14 | 0.38 / 0.12 | 0.12 |
| Eu/Gd | 158 | 70 | 0.23 | 0.17 | −0.01 / 0.16 | 0.25 / 0.15 | −0.04 / 0.17 | 0.15 |
| Gd/Tb | 165 | 65 | 0.21 | 0.26 | −0.75 / 0.23 | 0.36 / 0.13 | 0.08 / 0.15 | 0.17 |
| Tb/Dy | 168 | 65 | 0.16 | 0.14 | −0.06 / 0.13 | 0.32 / 0.09 | 0.21 / 0.10 | 0.09 |
| Dy/Ho | 176 | 57 | 0.17 | 0.15 | −0.18 / 0.13 | 0.20 / 0.10 | 0.28 / 0.09 | 0.08 |
| Ho/Er | 173 | 56 | 0.13 | 0.10 | −0.01 / 0.09 | −0.72 / 0.11 | −0.18 / 0.09 | 0.08 |
| Er/Tm | 132 | 54 | 0.16 | 0.09 | −0.03 / 0.09 | −0.35 / 0.12 | −0.40 / 0.13 | 0.10 |
| Tm/Yb | 133 | 54 | 0.12 | 0.07 | 0.04 / 0.07 | −1.01 / 0.11 | −0.86 / 0.12 | 0.08 |
| Yb/Lu | 150 | 54 | 0.12 | 0.08 | −0.02 / 0.08 | −0.46 / 0.10 | −0.41 / 0.11 | 0.09 |

The light and middle pairs (La/Ce … Dy/Ho) become predictable with three measurements (R² 0.2–0.4);
the four heaviest pairs (Ho/Er, Er/Tm, Tm/Yb, Yb/Lu) have the smallest spread (sd 0.12–0.16, FLAT
MAE 0.07–0.10) and every k-shot arm is at or *worse* than FLAT on them (the ridge line beats FLAT
only on Ho/Er, 0.078 vs 0.097): the measured curve's slope, extrapolated to the heavy end,
overshoots contrasts that are inside the noise. Gd/Tb (across the
half-shell) is the pair where three measurements help most (R² −0.75 → 0.36). (`RIDGE2` per-pair MAE from
`per_pair_BP_metal_C3_DOPT_Q_RIDGE2_k3.csv`.)

### C.8 Evaluation designs and the diglycolamide hold-out

Extractant-macro MAE at k = 3 (metal model, C3, DOPT_Q) and pooled R² under the six hold-out
designs (`results/discovery*/levels_<design>_metal_C3_DOPT_Q.csv`; A = exact extractant, a leak
price; B = chemotype; BR/BQ = size- and structure-matched random controls; BP = chemotype +
publication mask, the selecting design; BPG = BP + copy groups):

| arm | A | B | BR | BQ | BP | BPG | range |
|---|---|---|---|---|---|---|---|
| G14 zero-shot | 0.160 | 0.161 | 0.161 | 0.161 | 0.161 | 0.161 | 0.001 |
| LINE k=3 | 0.143 | 0.144 | 0.143 | 0.143 | 0.143 | 0.143 | 0.001 |
| FLAT@BLUP20 k=3 | 0.138 | 0.141 | 0.146 | 0.143 | 0.147 | 0.147 | 0.009 |
| G14@BLUP20 k=3 | 0.132 | 0.136 | 0.139 | 0.136 | 0.140 | 0.140 | 0.008 |
| RIDGE2 k=3 | 0.134 | 0.135 | 0.135 | 0.134 | 0.135 | 0.133 | 0.002 |
| NNC@SELECTED k=3 | 0.157 | 0.165 | 0.169 | 0.163 | 0.174 | 0.179 | 0.022 |
| pooled R², RIDGE2 k=3 | 0.325 | 0.312 | 0.316 | 0.318 | 0.310 | 0.323 | 0.015 |
| pooled R², G14@BLUP20 k=3 | 0.314 | 0.288 | 0.254 | 0.283 | 0.248 | 0.246 | 0.068 |
| pooled R², NNC@SELECTED k=3 | −0.015 | −0.075 | −0.154 | −0.127 | −0.128 | −0.215 | 0.200 |

The line and the ridge line are design-invariant (range ≤ 0.002), the covariance arms lose
0.005–0.008 when the publication mask is applied, and the neural arm is the only arm whose score
depends on the hold-out: it is best where the design leaks (A) and worst under the copy-group
mask (BPG), the signature gen14 identified for over-parameterised arms on this corpus. BPG moves
nothing else by more than 0.002.

Inside BP (`levels_BP_metal_C3_DOPT_Q_sc009_only.csv`, `…_without_sc009.csv`): on the fold that
holds out the diglycolamides (129 cells, 23 extractants, trained on the other chemotypes), the
ridge line reaches R² 0.31 / macro 0.150 at k = 3 and the line 0.23 / 0.156, while the deployed
measured mode gives 0.19 / 0.162 and the neural arm −0.18 / 0.183; on the other 38 chemotypes
(100 cells, 56 extractants) the order is the same (RIDGE2 0.30 / 0.128, G14@BLUP20 0.28 / 0.131,
LINE 0.23 / 0.137, NNC −0.12 / 0.171). The ranking does not depend on which half of the cohort is
scored.

### C.9 The pair model (the brief's definition): arithmetic versus transfer (`levels_BP_pair_C10_DOPT_Q*.csv`, C10: 114 cells, 50 extractants, 25 chemotypes)

| k | G14@BLUP20 R² / MAE | FLAT@BLUP20 R² / MAE | LINE R² / MAE | RIDGE2 R² / MAE | PROJ R² / MAE | NNC@SELECTED R² / MAE |
|---|---|---|---|---|---|---|
| 0 | 0.05 / 0.151 | −0.16 / 0.169 | −0.16 / 0.169 | −0.13 / 0.170 | −0.16 / 0.169 | −0.31 / 0.179 |
| 1 | 0.08 / 0.149 | −0.05 / 0.162 | 0.14 / 0.141 | −0.08 / 0.162 | −1.81 / 0.244 | −0.27 / 0.174 |
| 3 | 0.28 / 0.134 | 0.19 / 0.143 | 0.23 / 0.134 | 0.29 / 0.125 | −3.08 / 0.299 | −0.18 / 0.166 |
| 5 | 0.43 / 0.119 | 0.39 / 0.126 | 0.23 / 0.133 | 0.32 / 0.121 | −2.86 / 0.303 | −0.13 / 0.160 |
| 8 | 0.58 / 0.103 | 0.54 / 0.108 | 0.23 / 0.133 | 0.32 / 0.120 | −1.57 / 0.259 | −0.07 / 0.152 |
| 10 | **0.66 / 0.091** | 0.63 / 0.096 | 0.23 / 0.133 | 0.32 / 0.120 | −0.02 / 0.158 | −0.02 / 0.145 |

Here R² climbs to 0.66 at k = 10, but the rows that drive it are those whose two metals a
laboratory would already have read: restricting C10 to rows not `lab_known` (about 215 rows and 97
cells per seed) gives G14@BLUP20 R² 0.39 at k = 3 and 0.46 at k = 10, the line 0.41 → 0.42, the ridge
line 0.39 → 0.43 (`…_not_lab_known.csv`). The covariance arms' advantage over the two-parameter
lines at k ≥ 5 in the full C10 table is therefore mostly interpolation between measured
contrasts, not transfer, and it is why the metal model was registered.

### C.10 Setting B (exploratory; `settingB_levels_BP.csv`, `settingB_contrasts_macro_BP.csv`)

24 units (17 extractants, 11 chemotypes), 1 397 query pairs per seed. Copying the adjacent
separation factor measured at another condition of the same extractant and publication (`COPY`)
scores R² 0.32, MAE 0.091, macro 0.115, direction 0.88, against FLAT 0.161 / 0.160. Every adapted
arm (`G14@BLUP20`, `FLAT@BLUP20`, `MEAN_CURVE@BLUP20` given the support cell's pairs) lands within
0.001 of COPY (macro +0.0009, p 0.7); zero-shot arms are 0.035–0.047 worse than COPY and the line
through the support cell's pairs 0.012 worse (p 0.4). A model adds nothing to copying the other
condition's value, which is what the audit's replicate-floor argument predicted.

### C.11 Confirmation on withheld seeds (`results/confirmation/`, executed once)

Seeds 148569, 228670, 253534, 561551, 978178 (public rule, commitment `c8d5228a…` written before
discovery; `RUN_ONCE.json`). Same arms, same inner selection (the neural arm won 10 of 25 inner
selections on these seeds), same cohorts (C3: 229 cells, 79 extractants; C6: 133 cells, 58
extractants), M* fixed at `NNC@SELECTED` from the discovery decision.

| id | contrast | discovery | confirmation (98.75 % percentile / BCa; p; WCR p; seeds; LOCO) | verdict |
|---|---|---|---|---|
| P1 | M* vs LINE, k=3, macro MAE | −0.032 | **−0.029** [−0.043, −0.012] / [−0.044, −0.012]; p < 1e-4; 0.0004; 0/5; no | **fails twice** |
| P2 | M* k=3 vs k=0, macro MAE | +0.034 | **+0.023** [+0.005, +0.064] / [+0.003, +0.058]; p 0.0002; 0.007; 5/5; yes | **confirmed** (point ≥ 0.02 on both runs) |
| P3 | M* k=3 vs k=0, pooled ΔR² | +0.194 | **−0.100** [−0.26, +0.40]; p 0.59; 2/5 | **fails** (and R²(k=3) = −0.29) |
| P4 | M* vs LINE, k=6, macro MAE | −0.017 | **−0.022** [−0.036, −0.004] / [−0.039, −0.005]; p 0.001; 0.0009; 0/5; no | **fails twice** |

The one confirmed registered claim is that three measured metal pairs reduce the registered arm's
adjacent error by 0.02–0.03 log units; the arm itself is not the reason, since on the same withheld
seeds the ridge line again beats the line (+0.009 [+0.001, +0.017], p 0.006, WCR 0.02, 5/5, LOCO)
and reaches R² 0.32 at k = 3 (ΔR² +0.43 [+0.23, +0.54]), the deployed measured mode again ties the
line (+0.002, p 0.65) with R² 0.19, and the neural arm's pooled R² is −0.29
(`registered_style_contrasts_other_arms_BP.csv`, `levels_BP_metal_C3_DOPT_Q.csv`). Zero-shot
`G14` on the withheld seeds: R² 0.04, macro MAE 0.162; FLAT 0.173; LINE at k = 1: R² 0.18.

## D. Discussion

**1. How much does performance improve as k increases?** On the registered (metal) design the
adjacent-pair error falls from the zero-shot 0.161 macro MAE (R² 0.05) to 0.146 / 0.168 at one
measured metal pair (the line), 0.143 / 0.24 at two and 0.135 / 0.31 at three (the ridge line);
from three to six metals the pooled R² of the best arm moves from 0.31 to 0.32 and the macro MAE
from 0.135 to 0.120 on the narrower C6 cohort. Direction accuracy on strong pairs rises from
0.80 to 0.88. In absolute terms three measurements remove about a quarter of the zero-shot error
and explain about a third of the adjacent variance; the rest sits inside a replicate floor (sd
0.30 on the levels) that is larger than the target itself (sd 0.24).

**2. Which k is the best trade-off?** Two measured metals (k = 1, one separation factor: the
lightest and heaviest legal metals) buy most of the gain (R² −0.13 → 0.17 for the line); the third
metal is where a prior starts to matter (the ridge line's 0.24 → 0.31); beyond three, gains per
measurement are ≤ 0.01 R² for every arm on the common cohort. Under the pair model the apparent
climb to R² 0.66 at k = 10 is interpolation between measured contrasts (C.9), not prediction.

**3. Does the neural architecture outperform simpler statistical models?** No. The reference-
derived contrast network is the worst arm at every budget: zero-shot R² −0.32, k = 3 R² −0.13, and
0.030–0.033 macro MAE behind the deployed baseline on identical supports with intervals that
exclude zero (C.5). The pre-registered rule nevertheless nominated it (13/25 inner wins); the
registered P1 and P4 therefore fail with the line significantly ahead, and P3 fails on the
R²(k) > 0 clause. The best arm overall is a two-parameter curve shrunk toward the training-fold
prior (`RIDGE2`), which beats the line by +0.008 [+0.0001, +0.017] at k = 3 and +0.016 [+0.008,
+0.029] at k = 6, both below the programme's 0.02 margin. gen15's deployed measured mode is tied
with the line at k = 3 (+0.003, p 0.41). This repeats, on adjacent pairs and with the arithmetic
removed, what gen4, gen15 and gen16 found on all pairs: after one measurement, the corpus is worth
very little.

**4. Does molecular structure add information after supports?** Marginally and decreasingly:
`G14@BLUP20 − FLAT@BLUP20` is +0.017 at k = 0 (p 0.02), +0.011 at k = 1, +0.006 at k = 3 (p 0.08).
The direction bit the structure carries is what the first measurement supplies directly.

**5. Which adjacent pairs remain difficult?** The four heaviest (Ho/Er, Er/Tm, Tm/Yb, Yb/Lu): their
observed spread (sd 0.12–0.16) is below the replicate floor and every k-shot arm is at or below
FLAT on them, because the measured slope extrapolates past contrasts that are mostly noise. The
light and middle pairs and the half-shell pair Gd/Tb are the ones measurements make predictable
(R² 0.2–0.4 at k = 3). Pr/Nd, the application pair, reaches R² 0.31–0.42 and MAE 0.13–0.14 at
k = 3 against FLAT 0.20.

**6. Are improvements consistent across held-out chemotypes?** The k = 3 gains of the line, the
ridge line and the deployed measured mode are positive in 5 of 5 seeds and in every leave-one-
chemotype-out subset, and the ranking of arms is the same on the diglycolamide hold-out fold and
on the other 38 chemotypes (C.8). The effective number of chemotypes behind any pooled number is
about four, so "consistent across chemotypes" means consistent across a handful of families.

**7. Does the model generalise to genuinely unseen chemical systems?** Zero-shot, barely:
R² 0.05 and +0.012 macro MAE over FLAT (p 0.04, below the margin), consistent with gen15's +0.088
on all pairs shrinking to the noise floor on adjacent ones. What generalises is the curve prior
(the radius ramp and its curvature) once a measurement pins it, not a structure-specific
prediction; and "unseen" here is a chemotype hold-out inside one literature corpus (E).

## E. Limitations

* **Dataset size and chemical diversity.** 90 extractants in 45 chemotypes; one chemotype holds
  375 of 521 cells and about half of the adjacent pairs; the pair-weighted Kish effective number of
  chemotypes on the registered rows is about 4. 63 diglycolamide cells live in chemotypes other than
  the main one, so "chemically unseen" is a chemotype hold-out, not a family hold-out.
* **Publication bias.** 81 of 90 extractants come from a single publication; the 64 condition
  columns identify a publication with 94 % 1-NN accuracy, which is why BP removes the held-out
  cells' publications; what remains is the chemistry that transfers between laboratories.
* **Measurement uncertainty.** Adjacent `log SF` (sd 0.23) is below the replicate floor of the
  levels it is computed from (sd 0.30), and 12 % of replicated readings differ by whole decades.
* **Experimental-condition coverage.** Setting B is a replicate-floor question on this cohort
  (median cross-condition change of an adjacent separation factor 0.043) and is exploratory.
* **Statistical power.** The minimum detectable effect is printed with every registered contrast;
  claims whose MDE exceeds their margin are reported as underpowered (P2's MDE, 0.031–0.034, is
  above its 0.02 margin, so P2 was powered only for an effect of the size it found).
* **The registered arm rule.** The inner k = 3 comparison between the neural arm and the deployed
  measured mode was a near tie (13/25 on discovery, 10/25 on confirmation, 12/25 on the Nd–Sm
  pass); the pre-registered majority rule therefore nominated an arm that the outer scores show to
  be the worst. The rule was followed as written; the ridge line's and the deployed mode's
  results are reported as exploratory because they were not nominated.
* **Generalisation.** All held-out chemistry comes from the same literature corpus; five split
  seeds re-partition the same cells and are not new data; the confirmation seeds are withheld
  partitions of the same corpus, not new measurements. Nothing here is a statement about
  real-world separation performance.
