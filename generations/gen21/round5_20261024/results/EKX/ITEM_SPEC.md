# Item 1.8 specification: M07, the sc009 (TODGA-class) Nd dip frozen for the Pr/Nd warning (E-KX)

Frozen at the Stage-0 seal, before the item's first post-seal run (brief 1.2: Stage-0 seal -> item code seal -> reproduction run labelled TECHNICAL SMOKE -> arms; the pre-seal --smoke runs are PRE-SEAL ENGINEERING CHECKS, PROTOCOL 4 and 10; every scoring run executes the reproduction target first and the intervention before any scored write).  Every in-corpus number is EXPLORATORY (seen); nothing here is a test, and no in-corpus p-value is called significant.  E-KX is a separate estimand: never called BP, never placed next to E-A's 0.452; B1 on the same rows is the comparator in every table.  Code: `r5/ekx.py`, `scripts/s1_ekx.py`, `tests/test_ekx.py`; shared statistics from `r5/common.py`, frozen duplicate rules from `r5/dupes.py`, metadata from `r5/population.py`.  Sources: brief 6.8, 6.0, 6.10, 8, 9; IR M07; `ideas/critiques.json` M07 (test design and both skeptic verdicts); E1 3.4 (second differences).  No design-folder script implements M07, so nothing is copied; every level vector goes through `r4.refs.predict_levels` / `potentials` unchanged.

## Frozen object (external leg)

- u_Nd(sc009) = **-0.11**, the raw training-publication mean Nd chord anomaly [IR M07], shrunk toward 0 with **w = n / (n + 5)**, n = the number of donor publications (sealed reading: the source writes n / (n + 5) without defining n).
- External value: n = the V1 sc009 publications reporting Pr, Nd and Sm, counted from the `metals` metadata (label-free): **n = 12, w = 12/17 = 0.705882, u_ext = -0.077647** (`results/EKX/frozen_external.json`, written by the main run; `r5.ekx.frozen_external`).
- Form: a point potential at Nd added to B1, reads exact: level(m) = PWL_R[y_S - u_S](m) + u(m), u = beta_frozen 1[Gd] + u_ext 1[Nd] (`r5.ekx.external_pairs`, `kx_fit`).  The extractant-level Gd half of the original idea is dropped [IR M07].
- **Nd chord** of a cell with Pr, Nd and Sm measured: a_Nd = y_Nd - [(1 - w) y_Pr + w y_Sm], w = (x_Nd - x_Pr) / (x_Sm - x_Pr) along -Shannon CN8 radius (`r3.interp` 'R'; Pm absent) [E1 3.4].

## Rows, base and donors (E-KX-LOPO)

- Rows: round-4 REF rows (`r5.common.load_rows`: C3, k = 3, i.e. 4 reads with E-A's nested DOPT_Q read sets; 1,802 queries per seed; 5 dev seeds; 229 cells).  Pooled R^2 inside each seed, then the seed mean; MSE; relative pair-SSE vs B1; MAE; balanced MAE on pooled endpoints.
- **B1 base = the round-4 OOF B1 of the design's parquet.  BP is the primary base (brief 6.8: "B1 base = the BP OOF prediction"); BPG is a SENSITIVITY** (an exception to "BPG decides labels", stated by the brief for this estimand).
- **Donors** of target t: bench cells (all 521, any k_max) in another publication AND another copy group than t; never t; minus every cell linked to t by the frozen section-5.1 bit-identity rule (`r5.dupes.bit_identity`, `flagged_links`).  Per donor set: publication means first, then their plain mean = u_raw; n = donor publications; u = n / (n + 5) u_raw; u = 0 without a donor (the arm then equals B1).  `Donors.guard` refuses any set holding t, a cell of its publication or copy group, or (except NB) a bit-identical copy.  The donor rule is identical under both designs (LOPO with copy-group masking).

## Arms (identical rows; dphi_j = phi_j(A) - phi_j(B), phi_j(m) = 1[m = j] - w_j(m), 0 at a read metal)

| arm | definition | label |
|---|---|---|
| B1 | the parquet's B1 (comparator) | EXPLORATORY (seen) |
| KX_CHEM | B1 + u_Nd(chem(t)) dphi_Nd; donors of t's own chemotype; every chemotype ("chemotype Nd term") | EXPLORATORY (seen) |
| KX_SC009 | KX_CHEM on sc009 targets, B1 elsewhere (the narrowed deployment arm; = KX_CHEM on sc009 rows) | EXPLORATORY (seen) |
| NDU | B1 + u_Nd(all) dphi_Nd; donors of any chemotype ("B1-NdU", universal Nd potential) | EXPLORATORY (seen) |
| PRND_OFFSET | Pr/Nd rows: B1 + n/(n+5) mean_pub(e), e = (y_Pr - y_Nd) - B0 - beta_t dphi_Gd of the same-chemotype C3 donors with a scored Pr/Nd row in the same seed and design (the donor's B1 recomputed with the target's fold beta, so no label of t enters through a donor's fold fit); other pairs B1 (Pr/Nd chemotype x pair offset) | EXPLORATORY (seen) |
| PRND_COPY | Pr/Nd rows: the same-chemotype donors' publication-mean Pr/Nd log SF, unshrunk (copy another publication's SF); B1 without a donor and on other pairs | EXPLORATORY (seen) |
| KX_FROZEN | B1 + u_ext dphi_Nd on sc009 targets; NOT fold-independent (-0.11 was computed with every sc009 publication) | DIAGNOSTIC |

Arm list frozen; no arm is added after a result is seen (an addition goes into AMENDMENTS.md labelled post hoc).

## Endpoints, strata, sensitivities

- Endpoints: **SC009_PRND** (primary: sc009 Pr/Nd rows; R^2 and MAE), POOLED (all rows), PRND_ALL, SC009_ALL.
- C5-BTBP (pub_6a0ef8528c): **C5_out primary** (rows AND donors without it, as M07's seen numbers and its original kill), C5_in and C5_cell_out (c950dc2982e96b83) SENSITIVITY.
- Donor-pool sensitivities (C5_out; arms KX_CHEM, KX_SC009, NDU): NB (no bit rule), OFS (also drop offset / near-identity links, `r5.dupes.offset_scan`), QNT (drop Q_SMALLINT / Q_COARSE cells), V2 (drop V2-1/2/3 cells), V25 (a V2-5 floor reading at Pr, Nd or Sm voids the donor's chord; at Pr or Nd its SF), C3DON (C3 donors only), CELLW (cell-weighted mean instead of publication-weighted).
- Acid stratification: dla = |log10 acid_t - mean_pub(log10 acid of t's chemotype donors with a chord)| (concentrations floored at 1e-4 M; metadata only); NEAR (<= 0.5), FAR (> 0.5), NA.  Reported for SC009_PRND (KX_SC009) and POOLED (KX_CHEM), each with NDU, vs B1 and the chemotype-minus-universal contrast.

## Controls

- **Random-chemotype permutation**: chemotype labels shuffled over the 86 publication x chemotype units (the number of units per chemotype kept; labels of targets and donors move together), u recomputed with the primary donor rule; B = **2,000** draws (>= 1,000; `--perm-draws`, the run refuses fewer than 1,000), shared by both designs; statistic = seed-mean dR^2 vs B1 for KX_SC009 on SC009_PRND and KX_CHEM on POOLED, C5_out and C5_in; p = (1 + #{null >= obs}) / (1 + B) (DIAGNOSTIC).  The observed value is recomputed by the permutation code and must equal the r5.common value (assert, 1e-10).
- **Publication sign-flip over sc009 publications**: `r5.common.unit_stats` / `unit_tests` on SC009_PRND with publication units (G = 10; exact enumeration), KX_SC009 vs B1 and vs NDU; also POOLED with cluster_cpg and publication units.  SF_POOL, SIGN, SF_LOG, IM_REL (DIAGNOSTIC).
- **Leave-top-3-out**: the report rule below.
- Dropped as the brief and IR M07 say: the "4 of 5 seeds" leg, the random-extractant control, the Gd half, nested metal selection (Nd is fixed by the brief; the reliability of the derived chord is reported instead: split-half reliability of chemotype-mean chords and a per-target |mean| > 2 SE gate flag, DIAGNOSTIC).

## Report rule (CONSISTENCY CHECK: its expected outcome was seen)

- **Primary**: design BP, C5_out, POOLED rows.  Top-3 publications = the 3 with the largest per-publication dSSE (seed means; SSE_B1 - SSE_KX_CHEM), ties by publication id.  Remove their rows; rule statistic = R^2(KX_CHEM) - R^2(NDU) (seed mean) on the remaining rows.  **Narrowed to "universal Nd term" if the statistic <= 0** (expected; seen: chemotype +0.0010 vs universal +0.0027 without the top 3).
- Sensitivities (rule rows, labelled): C5_in; SC009_PRND with KX_SC009 (top 3 by its own dSSE); "rows and donors" (u recomputed with the top-3 publications removed from every donor set); the same table under the BPG base.
- Also reported: the top-3 share of the gain (`r5.common.top3_share`; seen 0.75) and each arm's full-row dR^2.

## Every item reports (brief 6.0)

C5-BTBP in / out (and the single cell), BPG and BP, per-seed values (R^2 and MAE per seed, dR^2 per seed, seeds positive), the unit table (`units_<design>.csv`, `unit_tests_<design>.csv`), leave-one-publication-out (`r5.common.lopo`, jackknife SE, t(P - 1) interval, min / max with the publication left out) and the top-3 share; the section-9.3 per-item table (`PER_ITEM_TABLE_<design>.md`).  Seen values of IR M07 are carried in `summary_<design>.json` for comparison only (none is a reproduction target).

## Reproduction targets (brief 6.0; `--repro`, TECHNICAL SMOKE; a mismatch stops the item)

- B1 recomputed from the fold betas (`round4_20261022/results/REF/fitted_<design>.json`) through `r4.refs.predict_levels` equals `REF_<design>.parquet` within 8.9e-16 on every row (18,020 over both designs).
- Construction identities: y = V1 log D difference exactly; B0 = PWL_R within 8.9e-16; B0 + beta dphi_Gd = B1 and B1 + u dphi_Nd = predict_levels(kx_fit) within 1e-12; no scored pair with both ends read; strict adjacency, lighter first; 1,802 rows per seed, 229 cells.
- Consistency lines (do not stop the item): sc009 Pr/Nd rows per seed (seen 46); the V1 raw sc009 publication-mean chord vs the frozen -0.11 (2 decimals) and its negative count (seen 11 of 12).

## Leakage (brief 8)

- Isolation: every donor set passes `Donors.guard` (asserted on every call); the B1 base is a round-4 run whose intervention and identity checks passed (`r5.common.load_rows`).
- Prediction intervention (`r5.ekx.intervention`, every target, both designs): scramble every finite label of every cell in the target's publication or copy group (its unread and read labels included), rebuild its arm inputs for all 10 donor configurations and its Pr/Nd offsets in every seed: bit-identical required.  Positive controls that must be caught: the target itself in its own pool (guard), a same-publication cell in its pool (guard), and a leaky pool keeping the own publication / copy group (must change under the scramble).
- Fit / cache poisoning: not applicable (no fitted quantity: u is a donor mean with K = 5 fixed; beta is the stored round-4 fold fit; no new cache).  Support checks as in the reproduction.

## External rider (brief 6.8; not a package claim, excluded by CP-13)

`r5.ekx.external_rider`: new sc009 publications in arrival order, each eligible if >= 1 cell reports Pr, Nd and Sm; per publication the mean Nd chord and the mean Pr/Nd squared error of B1 (beta_frozen) and of the frozen arm (u_ext) at the frozen external 4-read set (`r5.frozen.external_sequence`, `read_set(seq, 4)`), over cells whose Pr/Nd pair is scored ('falls' needs >= 1 scored row).  **Killed** if, over the first 4 eligible publications, >= 2 have a chord >= 0 or fewer than 3 'fall'; PENDING with fewer than 4.

## RNG seeds (`r5.paths.seed_for(key)` = 20261023 + crc32(key))

| procedure | key | seed |
|---|---|---|
| random-chemotype permutation (shared by BP and BPG) | `EKX\|PERM_CHEM` | 498720859 |
| split-half reliability (200 splits) | `EKX\|SPLITHALF` | 1432889228 |
| intervention scrambles | `EKX\|intervention\|BP`, `EKX\|intervention\|BPG` | 257350515, 3158259040 |
| smoke surrogate (noise labels) | `EKX\|SMOKE_SURROGATE` | 2997494442 |
| sign flips | exact at G <= 13; otherwise `EKX\|<design>\|<c5>\|<endpoint>\|<unit>\|<cand>-<ref>\|SF_POOL` (and `\|SF_LOG`), B = 4,999 | r5.common |

## Outputs (results/EKX/)

`rows_<design>.parquet` (row level: every arm column `<arm>@<c5>[~<variant>]`, u and n per row, dphi, flags, label, seal); `arms_`, `contrasts_`, `population_`, `units_`, `unit_tests_`, `lopo_`, `top3_rule_`, `permutation_`, `permutation_draws_`, `acid_strata_`, `dup_sens_`, `implied_sf_<design>.csv`; `PER_ITEM_TABLE_<design>.md`; `summary_<design>.json`; design-independent `targets.csv`, `chord_units.csv`, `chord_chemotypes.csv`, `chord_sc009.json`, `reliability.json`, `frozen_external.json`; `repro_<design>.json`; `SMOKE_COUNTS_<design>.json`; `results/leakage/EKX_<design>.json`.  Every row or object carries its label (EXPLORATORY (seen), SENSITIVITY, DIAGNOSTIC, CONSISTENCY CHECK, TECHNICAL SMOKE); every output records the seal tag and the code LF digests.

## Command lines (run from D:\ml_separator_gh)

```
set PYTHONIOENCODING=utf-8
rem technical smoke on real data (every metric suppressed; allowed before the seal)
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ekx.py --smoke --design both
rem after the seal: reproduction run (TECHNICAL SMOKE), then the arms
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ekx.py --repro --design both
.venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/s1_ekx.py --design both
.venv/Scripts/python.exe -B -W ignore -m pytest generations/gen21/round5_20261024/tests/test_ekx.py -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
rem after every run: round4_20261022/scripts/r4_seal.py --verify must report 5bacfe4c and the round-3 LF digests must match (brief 8)
```

Measured smoke cost (BP, 20 draws): 39 s, of which the intervention 23 s; the full run with 2,000 draws and both designs is expected at about 5 minutes, one process.
