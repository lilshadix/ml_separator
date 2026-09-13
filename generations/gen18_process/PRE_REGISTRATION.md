# Gen18 — pre-registration of the corpus-fitted distribution model

*Draft written 2026-09-13 by the design panel, before any fit. It is sealed by
`scripts/g18_seal_prereg.py` only after `scripts/g18_audit.py` has written `DATA_AUDIT.md`; the
SHA-256 of everything above the footer line is written into the footer, into
`results/eval/prereg_sha256.txt` and into `GEN18_REPORT.md`. `g18_fit_dmodels.py`,
`g18_eval_dmodels.py` and `g18_loading_check.py` refuse to run unless `g18_seal_prereg.py --check`
passes. After sealing this file is not edited; a change is a dated addendum at the end stating what
changed, why, and whether any result had been seen. Everything not written here is exploratory and
is labelled so in every table.*

## 0. What is committed to

| | |
|---|---|
| repository facts | `BRIEF.md` §2; design contract `DESIGN.md` §0.2 (data facts verified 2026-09-13) |
| frozen inputs | bundle `dataset with 3D structures/dataset.parquet`, SHA-256 `fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e024faf5dd`; gen6 provenance table; both read-only |
| quarantine | gen13 rules (`gen13sep.cohort.apply_quarantine`: TODGA structure under a foreign name, 129 rows; log D <= -6 sentinels, 3 rows) plus the unit-slip duplicate rule of `DESIGN.md` §4.6 (7 rows of the D3DODGA/Nd 3 M HNO3 series, pub_0e7f3e0563) |
| cohort counts | fixed by `DATA_AUDIT.md`, written before this file is sealed; the panel's verified counts are in §2 and the audit must reproduce them or the discrepancy is recorded in a dated pre-fit addendum before sealing |
| seeds | 18 everywhere (`numpy.random.default_rng(18)`); no wall-clock values in results |
| regime of every number | cohort, hold-out design, averaging unit, parameter status — `report.write_table` refuses tables without them |
| what the model is for | the **known-extractant regime** (design A of gen12–gen16: the extractant, acid class, diluent family and additive set are in the database; the conditions and the publication are new). It is the process-design use case. **No zero-shot claim is made or tested here**: the model has no skill for an extractant absent from the database, and no number in this generation may be quoted as such. |

## 1. Questions

* **Q1 (conditions, design A).** For a known extraction system, does the pooled-slope mass-action
  model predict log D at conditions and in a publication it was not fitted on better than the
  cheapest sensible alternatives?
* **Q2 (loading).** Given only the tracer-limit D of a publication-resolved loading series, does the
  ideal ligand-depletion correction predict log D at higher metal concentration better than keeping
  D constant?
* **Q3 (reliability).** Are the fitted slopes (`n`, `p_eff`) reproducible across publications
  within a system, and does the TODGA slope agree with the corpus-validated range?

## 2. Cohorts (frozen by DATA_AUDIT; panel-verified counts on 2026-09-13)

**System key** (`DESIGN.md` §3.1): (sorted canonical SMILES of extractant-role ligands, acid class,
diluent family, sorted modifier names, sorted aqueous-complexant SMILES). Temperature band
`20-30C` only for the primary analysis (NaN temperature assigned to that band and counted).

**E1 cohort.** Fit-eligible records (acid and extractant concentration present and positive, not
`UNIT_SLIP_DUPLICATE`) aggregated per replicate group (exact 64-column condition key +
publication + SMILES + metal). A (system, metal) group is *fittable* if it has >= 6 aggregated
points and >= 3 distinct levels on log acid or log extractant; it enters E1 if it also spans >= 2
publications. Verified: 233 fittable groups; **59 groups in 14 systems** span >= 2 publications
(TODGA/nitrate/aliphatic `sys_5cb78e5000d40860` 42 groups over 18 publications; TBDGA 7 groups /
4 publications; TEHDGA 3 / 5; eight further systems with one multi-publication group each). The
unit of the decision is the **system** (14), so TODGA counts once.

**E2 cohort.** The publication-aware loading series of `DESIGN.md` §4.5: >= 3 distinct initial
metal concentrations at fixed (system, metal, publication, nominal acid, extractant concentration),
after the duplicate rule. Verified: **10 series** — TEHDGA/Nd pub_15b174237f (3 points), TEHDGA/Nd
pub_5a68dc5665 (6), D3DODGA/Nd pub_0e7f3e0563 at 1 M (18) and at 3 M (8 after the duplicate rule),
TDDGA/Nd pub_5a68dc5665 (6), TODGA/Nd pub_5a68dc5665 (6), TODGA/Nd pub_d3c970567f at 0.1, 0.2, 0.3 M
TODGA (4, 4, 8), TODGA/Ce pub_917a4583d4 (8). Of these, four are known before fitting not to fall
with loading (the three flat pub_d3c970567f series and the rising Ce series); §7 says how they
count. The publication-blind definition (11 series) is a sensitivity only.

**Loading-active subset (secondary, defined by a data property, not by any model):** series whose
range of log D across points is >= 0.3. Verified: 6 series (the two D3DODGA, TODGA/TEHDGA/TDDGA
pub_5a68dc5665, TEHDGA pub_15b174237f). The Ce series (range 0.65, rising) also meets the range
criterion and is **included** in this subset; it is not removed for rising.

**Declared assumptions on corpus rows** (status `assumed`, flagged on every derived parameter):
`cond__metal_concentration_mM` is the initial aqueous concentration of the row's single metal;
O/A = 1; the nominal acid concentration is the equilibrium acidity (`EQUILIBRIUM_ACID_ASSUMED_NOMINAL`);
in HNO3 media the aqueous nitrate equals the nominal acid. If the source of pub_0e7f3e0563 or
pub_5a68dc5665 is read before sealing, the reported O/A and semantics replace the assumption *for
that series* and are recorded in a dated pre-fit addendum.

## 3. Models and baselines (exact definitions; implementation in `gen18proc/evalproto.py`)

* **M1 — pooled-slope mass-action fit.** Per system (ligand, band `20-30C`):
  `log D_k = a_m(k) + n * log10[L]_k + p_eff * log10[acid]_k + delta_p(k) + e_k`, one intercept per
  metal, slopes `n` and `p_eff` shared across metals, sum-to-zero publication effects for
  publications with >= 2 aggregated points (single-point publications: effect 0), weighted least
  squares with weight = number of replicate rows, **unpenalised**. An axis with < 3 distinct levels
  in the training set has its slope fixed at the mechanism prior (`n0 = 3.0`; `p0 = 2.0`, the
  rounded gen5 corpus acid slope 1.93; both pre-specified constants, status `assumed`, flagged).
  Mechanism: solvating for every system in the E1 cohort (the corpus has no fittable
  cation-exchange system).
* **B0 — constant per-(system, metal) mean** of the training publications' aggregated log D.
* **B1 — same-metal nearest-condition 1-NN** in (log10 acid, log10 extractant), Euclidean, among
  the training publications' aggregated points of the same metal; ties by smaller |delta log acid|,
  then smaller `record_id`. B1 is the cheapest sensible competitor and the null-fallback D source.
* **C0 — constant D from the tracer point** (loading baseline): log D at every point of a series
  equals the tracer value.
* **C1 — ideal depletion** (loading model): `log K` anchored so that `equilibrium.solve_stage`
  reproduces the tracer log D exactly; every other point predicted by `solve_stage` at O/A = 1,
  initial aqueous concentration = the row's mM, `L_T` = extractant concentration, nominal acid,
  `n` = the system's M1 slope from the **full** in-sample fit (prior `n0 = 3.0` when the system
  has no fittable group), `K_H` = null (uptake unmodelled, flagged).

Prediction for a held-out publication uses **publication effect 0** (M1). The offset-calibrated
variant (one held-out point sets the publication effect; the rest are scored) is reported and
labelled `one_measurement_mode`; it is not part of the decision.

## 4. Fitting and hold-out procedure

Leave-one-publication-out (LOPO) within each system of the E1 cohort: for each held-out
publication, fit M1 (and compute B0, B1) on the other publications' fit-eligible points (all
metals of the system), predict the held-out points of metals that have a training intercept
(metals absent from training are skipped and counted). In-sample numbers (fit on all
publications, scored on the same) are reported in the same tables, labelled `in_sample`.

## 5. Primary endpoint E1 and decision rule R1

**E1** = mean absolute error of predicted log D on held-out points, averaged in this order: over
points within (system, held-out publication, metal) -> over (publication, metal) pairs within the
system with equal weight -> **macro over the 14 systems**. Reported for M1, B0, B1 (and the labelled
variants). Point-weighted and in-sample versions are reported beside it, labelled.

**R1.** M1 is *adopted as the default D source for corpus systems* if and only if all of:
(i) macro E1(M1) < E1(B1) − 0.05 and E1(M1) < E1(B0) (margin 0.05 log D: below that the process
consequence is inside the replicate floor 0.225 and the cheaper B1 is preferred);
(ii) the paired system-level bootstrap (2000 resamples of systems with replacement, seed 18,
percentile 95 % interval of mean(E1(B1) − E1(M1))) excludes zero;
(iii) M1 beats B1 in at least 9 of the 14 systems (ceil(0.6 x number of systems); the audit's count
sets the number).
If any of (i)–(iii) fails the result is a **null**: the process chain uses **B1
(`NearestConditionD`, with the ideal depletion correction and the prior `n0`) as its default D
source for corpus systems**, M1 becomes an exploratory option with its LOPO table shown, and the
report says so verbatim. A null is a result of this generation, not a failure of it.

## 6. Reliability before interpretation (E3; `feedback-reliability-before-correlation`)

Before any slope is interpreted or used, per system: jackknife-by-publication estimate and SE of
`n` and `p_eff`; split-half **by publication** (random halves of publications, 20 seeded repeats,
seed 18) sign agreement of `n` and `p_eff` and Pearson r of the metal-intercept vectors, for
systems with >= 4 publications; systems with 2–3 publications report the jackknife only and
`split_half = None` (stated as undefined, not as passed). A slope is `interpretable` iff jackknife
SE < 0.5 and, where defined, split-half sign agreement >= 18/20. Never a split within a
publication. **TODGA check:** the in-sample pooled `n` of `sys_5cb78e5000d40860` (band 20-30C)
is compared with the corpus-validated interval [2.36, 2.88] (gen5 MASSACTION block); the
`validation`-marked test asserts it lies inside; a failure is reported as a defect with the
jackknife SE, never suppressed and never fixed by changing the fit. The unpenalised value is the
one compared (no prior pulls it toward the interval).

GP-BO in `optimize.py` and the offset-calibrated mode are gated on `interpretable` for the system
used.

## 7. Loading endpoint E2 and decision rule R2

**E2** = per-series MAE of predicted log D over the non-tracer points, for C0 and C1, macro over the
10 series (each series weight 1). **R2.** C1 is *supported* if and only if (i) E2(C1) < E2(C0) on
more than half of the series (>= 6 of 10) and (ii) the paired series-level bootstrap (2000
resamples, seed 18) 95 % interval of mean(E2(C0) − E2(C1)) excludes zero. **Every series counts as
it falls**: the three flat pub_d3c970567f series and the rising TODGA/Ce series are expected to be
wins for C0 and are not excluded, down-weighted or explained away; if they make R2 fail, the
verdict is "the direction of the loading effect is not consistent across corpus publications and
the ideal correction is not supported corpus-wide", and the process chain keeps the depletion term
*as a mechanism with a flag* (`OA_ASSUMED`, `HIGH_LOADING`), not as a validated magnitude.
Secondary, labelled: (a) E2 on the loading-active subset (6 series) with the same rule; (b) E2 at
O/A in {0.5, 1, 2}; (c) descriptive Spearman(log mM, log D) per series (the `validation` test
`test_loading_direction.py` asserts it is negative on the loading-active subset; a failure is
reported). The single-metal stage solve is used because every corpus row is a single-metal
experiment (assumed).

## 8. Exploratory arms (declared now, rules fixed now, all labelled exploratory)

* **X1 effective capacity `phi`** (`ActivityModel EffectiveCapacity`): series with >= 4 non-tracer
  points, leave-one-point-out, `phi` in [0.01, 1] fitted on the rest with `log K` re-anchored at
  the tracer for every `phi`. Rule: supported only if LOO-MAE(X1) < E2(C1) − 0.10 on >= 4 of the 6
  loading-active series; otherwise `phi` is a reported fudge and is not used anywhere.
* **X2 metal-specific slopes** (`n`, `p_eff` per metal) under LOPO, same tables as M1.
* **X3 cross-metal 1-NN** (B1 over all metals of the system) under LOPO.
* **X4 tied-D sensitivity**: E1 recomputed without `TIED_D` rows.
* **X5 publication-blind loading series** (11) for E2.
* **X6 diluent-identity split** within `sys_5cb78e5000d40860` (n-dodecane versus other aliphatics).
* **X7 offset-calibrated one-measurement mode** (reported by §3; counted here).
* **X8 Huber-loss robustness arm** of M1.

## 9. Comparison count

Primary contrasts: **3** (M1 vs B0, M1 vs B1, C1 vs C0). Secondary, pre-specified: 4 (E2 on the
active subset; E2 at O/A 0.5 and 2; the TODGA slope check). Exploratory: X1–X8 as listed, each a
named contrast (8 at declaration; the exact number is written by `ComparisonCounter` into
`results/eval/comparisons.csv`), with Benjamini–Hochberg over the exploratory family and both raw
and adjusted values reported. Any contrast not listed here is counted as exploratory when it is
added, with a dated addendum.

## 10. What is not claimed

No zero-shot skill; no ranking of extractants; no statement about a system absent from the
database; no equilibrium acidity for corpus rows; no activity corrections; no O/A other than the
declared 1 for corpus loading rows; no validation of the Pr/Nd case (its parameters are
placeholders and its checks are consistency checks); no cost in currency without a sourced price
table. The gen15 direction model is not a D source and is not evaluated here.

## 11. Determinism and reproduction

`g18_build_db.py` -> `g18_audit.py` -> `g18_seal_prereg.py` -> `g18_fit_dmodels.py --all` ->
`g18_eval_dmodels.py` -> `g18_loading_check.py`, from the repository root with
`.venv/Scripts/python.exe`, seed 18, single process. Every script writes a manifest with input
SHA-256s; `results/eval/DECISION.md` and `results/loading/DECISION.md` print the R1 and R2 verdicts
verbatim from the rules above.

## 12. Pre-fit addendum 1 (2026-09-13, orchestrator; written after `DATA_AUDIT.md`, before sealing, before any fit)

Recorded so that the cohort the audit froze and the text above agree. No result had been seen.

1. **E1 unit clarification.** §2 quotes "42 groups over 18 publications" for TODGA. The audit
   (`DATA_AUDIT.md` §2, §7) shows that 42 is the sum over the four TODGA-*named* systems of the
   §3.1 key (aliphatic, aromatic, other, alcohol diluent families); the system
   `sys_5cb78e5000d40860` itself has **14 groups over 27 publications**, largest single group 18
   publications. The decision unit is the system, 14 systems, unchanged; R1 (iii) needs 9 wins.
2. **NaN metal-concentration rows**: 1532 is the pre-quarantine count; after the gen13 quarantine
   it is 1407. No rule depends on it.
3. **Unit-slip rule as implemented** (`addenda/WB1.md` A3): the factor test (molar mass, 1000,
   1/1000) is applied on the metal-concentration axis only; the 7 flagged rows are
   Ca_SAFE:2693–2699 as the design verified; the acid axis is not tested (it would flag one
   rounded plateau, Eu_SAFE:16079/16082, which stays `TIED_D`).
4. **Loading-active subset** = 7 series (§2 said "6 … the Ce series also … included", i.e. 7).
5. **U2 (O/A and metal semantics).** Abstracts of all five E2 publications were read
   (`addenda/ORCHESTRATOR_prefit_20260913.md`); none states O/A. The declared assumptions
   (initial aqueous concentration, O/A = 1, `OA_ASSUMED`) stand for every E2 series.
6. **U6 (third-phase limit).** Tachimori, Sasaki, Suzuki, SEIE 20, 687 (2002), doi
   10.1081/SEI-120016073: loading capacity of 0.1 M TODGA/n-dodecane at 3 M HNO3 = **0.008 M
   Nd(III)** (abstract). Entered into the corpus TODGA/nitrate/aliphatic entry as
   `phase.loc_metal_M`, status `literature`. The Sasaki 2015 TODGA/Nd series
   (`ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1`, 4.9–12 mM) crosses this limit; under R2 it
   still **counts as it falls** (no exclusion, no re-weighting). The loading report interprets a
   C1 miss on it with the LOC; the cascade raises `THIRD_PHASE_RISK` above the sourced value for
   that system.
7. Nothing else above this section is changed.

---
Sealed SHA-256 of everything above this line: `62f5f2f26604da0ed091919342000b87eea6a796dbb2dd779a757ecc60565668`
