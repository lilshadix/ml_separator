# Data contract, round-5 addendum (Stage 0)

Status: **Stage-0 document, written before the Stage-0 seal and revised at the pre-seal review
(`results/stage0/PRESEAL_REVIEW.md`); not sealed.** No round-5 score had been computed when it
was written (only the TECHNICAL SMOKE runs of the item code, every metric suppressed). Items marked **PROVISIONAL**
depend on open requests (U6 laboratory-key review, U5 planning amplitudes, U2 SAFE export, S1 V2-3 sign-off, U1 source
recovery); they are fixed only by an append-only amendment made **before any new value is read** (brief §4.6).

Scope. This addendum makes brief §5 concrete with the Stage-0 outputs and adds the E-CT information contract of brief
§8 and the frozen keys and maps of brief §5.5. **The round-4 data contract (`round4_20261022/DATA_CONTRACT.md`) stays in
force unchanged for E-A** (V1, the dev seeds and fold plans, what a predictor may see, what a fit may use, orphans,
subgroups, `cluster_cpg`). Where the two differ, this addendum governs only the round-5 estimands and pools it names.

Sources (paths under `generations/gen21/`): "brief" = `round5_design_20261023/CLAUDE_CODE_PROMPT.md`; IR, I2, P1–P5,
PC, DR as in the brief's source table; "S0:" = a file in `round5_20261024/results/stage0/`; "spec <ITEM>" =
`round5_20261024/results/<ITEM>/ITEM_SPEC.md`. Decisions D1–D11 (D9–D11 added at the pre-seal review) are those recorded in `DECISION_LOG.md` (reason and
sensitivity there); this file applies them.

Labels. Every in-corpus value quoted here is either a **count** (metadata or a label-based Stage-0 data rule; no
prediction) or a **seen** value quoted from the brief or its sources, labelled EXPLORATORY (seen). None is a round-5
score, and none is placed next to E-A's 0.452.

---

## 1. Data versions (brief §5.3)

| version | definition | role in round 5 |
|---|---|---|
| **V1** | the frozen gen13 cohort, 521 bench cells × 14 Ln log D (`cohort_exact.parquet`, SHA-256 `3f11ac4c…08bb7fd6`, S0: `dupscan_summary.json` "inputs") | **the data.** No V1 value is edited in round 5. |
| **V2** | round-3 rules (`r3/datav2.py`; `next_research_20261019/DATA_QUALITY_REPORT.md`): V2-1 re-import quarantine, V2-2 duplicate entry, V2-3 hidden merge, V2-5 resolution floor ("left-censored at D < 1.5·10^e", a data-internal value-pattern rule, round-3 amendment D-7) | V2-1/2/3 are target and sibling exclusions inside E-CT's CLEAN rule (26 bench cells dropped, S0: `ect_population.json` `n_v2drop_cells`) and sensitivities elsewhere. V2-5 floor readings are **kept** in E-CT's primary and dropped in a sensitivity (the `V25_*` arms, spec ECT §5). |
| **V3** | exists only after U1 (brief §7.1): V3-R = corrections from the random census; V3-HS = value-informed corrections, an upper bound, never a model-selection benchmark | sensitivity only; scored on byte-identical queries with the V1 read sets frozen; censored targets enter only through the interval line; every V3 number states its row count and is never placed next to 0.452 (brief §3.3, §7.1). |

**Label changes allowed without a document (not edits; brief §5.3, P5 §2, §4):** the 10 comment-located cells (P5 F1)
move from "no location" to "located"; DMDPhPDA copy B is a quarantined duplicate (V2-1; 0.4561 as a sensitivity, seen);
pub_84113c17ad carries a COMPONENT error flag (P5 F6: CDTA/HEDTA recorded as the extractant; the title names TEHDGA +
HDEHP).

**S1 open (V2-3 cohort key, DR S1).** Until the user signs off, the 11 C3 hidden-merge cells (6.4 % of B1's SSE, gain
share −9.0 %, seen; DR S1) are a separate condition-aware stratum reported as a sensitivity; V1 is unchanged. Items
1.0 (EA_CHECK, `duplicates_<design>.csv`; REGISTRY 1.0-S09) and 1.5 (EDEC, primary statistics; 1.5-S13) report it, with
noV2 and noBITDUP beside it (1.0-S07 / S08, 1.5-S11 / S12).

## 2. Duplicate and near-copy scan (brief §5.1; I2 rule 15)

Run before any sibling, donor or placebo pool was built, on all 521 bench cells and every unordered pair (135,460
pairs), within and across publications and extractants (S0: `dupscan_summary.json`; code `r5/dupes.py`,
`scripts/s0_dupscan.py`; copied sources and their LF digests are recorded in the summary). Every detection uses label
values, so each finding is a **sensitivity and a provenance question, never an edit to V1** (I2 "Duplicate and
provenance candidates"; IR rule 7).

### 2.1 The frozen bit-identity rule (sealed here, before any E-CT score)

1. Compare V1 log D as **float64 equality** at every metal finite in both cells.
2. **Flag a link** when **≥ 3 shared metals are identical** and they make up **≥ 50 % of the shared finite metals**,
   within or across publications and within or across extractants.
3. **Remove flagged links only, never cells.** A flagged link is removed from every E-CT and E-LAB pool (siblings,
   OFF donors, placebo pools, campaign donors) and from any other donor or sibling pool built in round 5.
4. **Count n_sib and assign strata after the removal.**
5. On new data the same rule runs **inside the value-sealing step** (brief §1.2 "Values from new sources"): the agent
   that seals extracted values applies it to full series (new against new, and new against V1) and passes only the
   resulting link list, never values, to the pipeline. A duplicate of a target can then never enter its pool, while the
   modeller stays blind to the target's unread metals.

The rule is the code `r5.dupes.bit_identity` (`BIT_MIN_IDENT` = 3, `BIT_MIN_FRAC` = 0.5); `r5.ect.guard_pool` and the
E-LAB guard refuse a strict pool that holds the target or a bit-identical copy of it (spec ECT §11, spec ELAB "Leakage").

### 2.2 The full flagged-link list on V1 (6 links; S0: `dupscan_summary.json` "bit_identity", `ect_population.json` "flagged_links_all")

Twenty-one pairs share ≥ 2 identical metals; six pass the rule. None crosses publications (5 within publication, same
extractant; 1 within publication, cross-extractant). Every flagged cell is a C3 cell.

| # | cell A | cell B | publication | extractants | identical / shared | V2 status | known item | pools from which the link is removed (by target) |
|---|---|---|---|---|---|---|---|---|
| 1 | 44012938119033a5 | 5169c377777d282b | pub_19783c039b | same (N-methyl-…-N-octylacetamide), 4.0 M / 5.0 M | 13 / 14 | — | item 1 | both targets: CLEAN siblings, OFF_ALL, OFF_PUB |
| 2 | d4c1e23a091d0960 | e4ee526bda3920c2 | pub_c8195b4ca7 | same (DHD2DGA) | 3 / 6 | — | item 3 | both targets: CLEAN siblings, OFF_ALL, OFF_PUB |
| 3 | 26cbed1ed70886fe | 259e68bb986bb484 | pub_b9dfc468cd | cross (TDdDGA / TDDGA) | 8 / 8 | — | item 4 | both targets: OFF_PUB, OFF_XEXT, PB_PUB (never a sibling: different extractant) |
| 4 | 247f57741141188a | 62e7cdd6fa003391 | pub_b9dfc468cd | same (THDGA) | 8 / 8 | V2-2 pair (62e7cdd6fa003391 dropped) | b9 pair 1 | only the dropped member's OFF_PUB pool |
| 5 | 5a6a2f3bf13a7d55 | c61ecb085d8df371 | pub_b9dfc468cd | same (T2EHDGA / TEHDGA) | 8 / 8 | V2-2 pair (c61ecb085d8df371 dropped) | b9 pair 2 | only the dropped member's OFF_PUB pool |
| 6 | 5df28757679cae26 | f8ed87d3cf7f4281 | pub_b9dfc468cd | same (TODGA) | 8 / 8 | V2-2 pair (f8ed87d3cf7f4281 dropped) | b9 pair 3 | only the dropped member's OFF_PUB pool |

Every known item the brief expected the rule to flag is flagged (items 1, 3, 4 and the three pub_b9dfc468cd 8-of-8
pairs, all three of which are V2-2 pairs; S0: `known_items_all_pass` = true). The links the brief expected **not** to be
flagged are not flagged: af971a0b699f29ae against the item-1 pair (4 / 14 and 5 / 14); 3 / 14 links inside
pub_e9799b75f3 (two) and pub_3f3b6220e0 (one); two 2-of-2 links (pub_6c8ed9481c, pub_19783c039b). The scan across
publications adds six unflagged cross-publication, cross-extractant links with 2–3 identical metals (3 / 14:
pub_3e0f6bb1b5–pub_0fd6ab4e40, pub_e9799b75f3–pub_3f3b6220e0, pub_af7a1cd919–pub_3e0f6bb1b5; 2 / 9 twice involving
pub_1bd76c48d9; 2 / 14 pub_ca616a18dd–pub_af7a1cd919), and two more within-publication 2-of-14 links (15 unflagged
links in all).

### 2.3 Resulting E-CT PRIMARY counts (S0: `ect_population.json`; counts only)

The rule removes **2 sibling links inside the E-CT pools** (item 1 and item 3; 4 directed links, S0:
`ect_sibling_links.csv`), so 4 targets each lose one sibling. **No cell enters or leaves any stratum**
(`PRIMARY_cells_lost_to_bit_rule` and `…_gained` are empty), so the PRIMARY counts **equal P1's**:

| stratum (brief §6.1) | cells | publications | publication × chemotype units | rows per seed (BPG = BP, every dev seed) |
|---|---|---|---|---|
| **PRIMARY** = CLEAN + bit rule, n_sib ≥ 2, without C5-BTBP | **99** | **12** | **14** | **690** |
| PRIMARY without the bit rule (P1's CLEAN; reproduction target, sensitivity) | 99 | 12 | 14 | 690 |
| WITH_C5 (sensitivity) | 114 | 13 | 15 | 838 |
| GE1 = n_sib ≥ 1, without C5 (sensitivity) | 115 | 17 | 20 | 820 |
| GE1_WITH_C5 | 130 | 18 | 21 | 968 |
| EQ1 = n_sib = 1, without C5 (the harm rule) | 16 | 8 | 9 | 130 |
| V1_GE1 (every covering sibling; E6 population) | 149 | 19 | 22 | 1,111 |
| V1_GE2_NOC5 (P1's V1 stratum; the bridge to IR's numbers) | 112 | 12 | 14 | 821 |

- Every P1 reproduction count matches (`p1_reproduction_all_match` = true); scored rows are identical in BP and BPG.
- PRIMARY n_sib distribution: 2 (11 cells), 3 (11), 4 (17), 5 (10), 6 (14), 8 (1), 14 (13), 15 (3), 18 (2), 19 (17).
- PRIMARY condition-change classes: acid concentration only 23, acid type (± concentration) 13, diluent only 19,
  extractant concentration only 3, temperature only 1, mixed 40.
- PRIMARY geometry (P1, seen; brief §6.1): 3 `cluster_cpg` clusters, 89 % of rows in g:pub_74428aeab2, Kish 1.25; 94 of
  99 cells DGA. Cluster-level inference is "not estimable".
- Laboratory annotation of the PRIMARY publications under the PROVISIONAL key (§8.1): 12 publications → 7 conservative
  / 11 liberal laboratories; V1_GE1's 19 publications → 9 / 16 (S0: `supply_audit.json` "ect_laboratory_annotation").
  Until U6 is answered, "same publication" reads "same study or campaign" (IR rule 8).

### 2.4 E-LAB pools (spec ELAB "Pools"; counts from the TECHNICAL SMOKE, no score)

- CLEAN pool = the same-publication other-extractant donors (SRC) minus links flagged by the bit-identity rule **or** by
  the near-identity flag (max |Δ| ≤ 0.05 over ≥ 6 shared metals): the V2-missed duplicates of I2 items 4–7 are
  quarantined. Metal-concentration duplicate groups are reduced to one member (0 groups on V1: the 69
  metal-concentration-only pairs share no metal in 68 cases and 2 metals in 1, i.e. complementary halves of one series,
  S0: `dupscan_summary.json` "metal_conc_duplicates").
- **Label-free lab-1 near-copy rule (primary, brief §5.1):** a donor is a near-copy if it is in the same medium (acid
  type, diluent, |Δlog10 acid| ≤ 0.15) and its log D lies within 0.15 of the target's at all 4 read metals of the design
  and seed. On V1: 1,025 target–donor pairs, 34 flagged, 23 targets with a near-copy donor in some read set
  (pub_b9dfc468cd 11, pub_1bd76c48d9 4, pub_3f3b6220e0 2, pub_4cba27f5f3 2, pub_5a68dc5665 2, pub_dfa96a5118 2);
  17 targets on the CLEAN pool (6 targets' near-copies are already quarantined).
- POPX 92 cells / 16 publications / 787 rows per seed; POPX0 60 / 13 / 534 (spec ELAB).

### 2.5 Near-copy, offset and quantisation sensitivities (never a primary pool rule)

| scan (rule; `dupscan_summary.json`) | V1 result | use |
|---|---|---|
| **constant offset** (a window of ≥ 6 metals, shift sd ≤ 0.03, \|shift\| ≥ 0.05) | 418 links (cross-pub cross-ext 239, within-pub same-ext 96, cross-pub same-ext 56, within-pub cross-ext 27); 266 DGA–DGA; 7 integer shifts. Most are parallel heavy-end segments of homologous solvating extractants (chemistry, not copying) | sensitivity only (`OFS_*` arms of E-CT, sibling and control pools; `PRIMARY_OFS` of M03; `MX_LM10_OFS` of E-LAB; `OFS` donor variant of E-KX) |
| **near identity** (max \|Δ\| ≤ 0.05) | 9 links (within-pub cross-ext 3, cross-pub same-ext 2, within-pub same-ext 2, cross-pub cross-ext 2) | quarantined in E-LAB's CLEAN pool; sensitivity in E-CT (`OFS_*`) |

The **primary E-CT pools follow the frozen bit rule only** (§2.1). The registered duplicate sensitivity removes every
OFFSET / NEAR_ID link from every E-CT pool: the sibling arms (`OFS_COPY`, `OFS_SH0`, `OFS_SIB_r4`, `OFS_SH_r4`; REGISTRY
1.1-S05), the placebo and OFF pools (`OFS_PB_*`, `E_OFS_PB_*_CM`, `OFS_OFF_ALL / PUB / XEXT`; 1.1-S20) and the near and
acid pools (`OFS_NEAR_r4`, `OFS_NEARSH_r4`, `OFS_ACOPY_r0`, `OFS_AINT_r0`; 1.1-S21), and M03 re-runs its selection on the
OFS pool (`PRIMARY_OFS`, `PRIMARY_NO_ORNL_OFS`; 1.2-S06 / S07). Correction (pre-seal review, s0-items-1): through the
sibling arms alone, near identity reached only 1 of the 9 links (4759cae892250232 – 830861fddf3571ee), because items 5
and 7 are cross-extractant and cross-publication links; the placebo / OFF / near / acid OFS columns cover the rest.
| **quantisation** (ties ≥ 3; coarse step 0.1 with half-median 0.01; small-integer D ratios: share ≥ 0.75, p/q ≤ 10) | Q_TIES 26 cells (24 C3); Q_ADJEQ 54 (40); Q_COARSE 8 (5); Q_SMALLINT 17 (16); Q_UNRESOLVED 181 (91) | `QNT_*` sensitivity of E-CT drops Q_SMALLINT or Q_COARSE cells; EDEC's E-REC row excludes the quantised pub_a49146c8fd (spec EDEC) |
| **label-based shape sd < 0.02** (E6) | flags 102 of 855 CLEAN links into PRIMARY targets and misses the bit-identical pair (shape sd 0.0217) [PC P1] | sensitivity only (`SD_*`; IR rule 7) |

E-CT sensitivity populations (TECHNICAL SMOKE counts, `results/ECT/SMOKE_COUNTS_BPG.json`): PRIMARY_OFS 95 cells (652
rows per seed); PRIMARY_QNT 94 cells (636); PRIMARY_I1 (item-1 link only, P1's pool) and PRIMARY_SD 99 cells; the V2-5
variant keeps 99 cells and drops every pair that touches a floor metal (8,838 of the 9,010 REF rows per design pass
`v25_row_ok`).

**Known sensitivity items** (brief §5.1 table; detected by the scans above, S0: `known_sensitivity_items`): item 5
(near identity, 0.0097), item 6 (near identity, 0.040), item 7 (4 JAEA cross-publication near-identities, ≤ 0.0101),
item 8 (5 constant-offset links: four C5-BTBP cells against DMDPDGA 1b71b86063cd1faf with shifts −5.37, −3.98, −6.15
and −4.71, sd 0.012–0.026, and pub_723ceb6c2c c6571910bb94ca86 against TODGA 354fe634113a078e at +1.95, sd 0.021), item
9 (Q_COARSE, cf100c942f2bcf49), item 10 (Q_ADJEQ, 2 cells), item 11 (Q_SMALLINT, 3 cells). **Item 2** (cell
652810ec40425074 against 44012938119033a5, sd 0.042 over 14 metals) is **not detected** by either scan; it stays on the
provenance list by name. Item 12 (pub_84113c17ad, component identity) is metadata and is carried as the COMPONENT flag.

**Provenance list for U1 / M25:** 25 links (BIT 6, KNOWN_ITEM 11, INTEGER_SHIFT 7, NEAR_ID_CROSS 1) in 12
publications, plus 29 quantisation-flag publications and the item-12 component error. The tiering was set **post hoc**,
after the broad offset counts were seen, and selects provenance requests only, never an analysis pool. Two publications
need a provenance check before any E-CT number is relied on: pub_19783c039b (all of P1's SH − OFF) and pub_b0b3296ec5
(41 % of P1's SH − B1 gain) [I2].

## 3. C5-BTBP in and out, everywhere (brief §5.2)

- **C5-BTBP** = pub_6a0ef8528c: 15 cells, 148 rows per seed; 23.1 % of B1's SSE and 34.9 % of its gain; the single cell
  c950dc2982e96b83 holds 11.8 % of the SSE; all 44 C3 readings below log D −4 are its readings [P5 §3, F4] (seen).
- **Rule.** Every round-5 number is reported with and without pub_6a0ef8528c, with leave-one-publication-out and with
  the single-cell influence of c950dc2982e96b83. A gain that lives in C5-BTBP is a provenance sensitivity, not a model
  result; dropping it also removes about 18 % of SST and was chosen for its SSE share, so it is outcome-dependent
  [PC P3].
- Per item (frozen in the item specs, D8): **E-CT PRIMARY excludes C5-BTBP**, WITH_C5 is the sensitivity (brief §6.1);
  M03 excludes it from training and scoring (with-C5 and its single-cell variant are sensitivities); E-A check (1.0),
  WINDOW, ED, EW and EDEC carry C5_in as the main rows with C5_out and the single cell out beside them; E-LAB keeps C5_in
  primary (no POPX target is C5-BTBP; only placebo pools change); E-KX takes **C5_out as primary** (as M07's seen numbers
  and its original kill); EDEC's E-REC population is C5-out (in_GE1).
- New data: no source plays C5-BTBP's role by name. The floor rules of §4 carry the general lesson.

## 4. Floors and limits (brief §5.4; IR M24; P5 F4; PC P5)

1. **Never mix clamped values with raw sub-limit reads in one pair.**
2. A floor rule derived from values is a **sensitivity** until a source states a limit; floor-aware predictors are
   closed on V1 at every F (closure ledger, IR M24).
3. The ORNL channel is itself clipped at exactly log D −4.0 (two readings in ORNL-54 C3 cells) [PC P5].
4. V2-5 resolution-floor readings: kept in E-CT's primary, dropped in its `V25_*` sensitivity; on new data the V2-5
   pattern rule is a sensitivity, never a primary filter, except for Gd chords, where V2-5 censoring is primary
   (CONFIRMATION_PACKAGE §13; brief §4.3 item 11; D9), which overrides the general rule of item 2 for those claims.
5. **New sources:** per-metal detection limits / LOQ are recorded at extraction. A value reported as "< DL" (or below a
   source-stated LOQ) is **censored**: it is never a point read and never a point target; it enters only the interval /
   bound line (M23; spec EDEC "One-sided bounds", L = −4.0 log D for the corpus floor sensitivity). Values below DL or LOQ
   are excluded from every Gd claim (CONFIRMATION_PACKAGE §10).
6. M24 reopens only if U1 recovers per-source limits and the floor-consistent ΔSSE gives ≥ +0.005 R² in ≥ 4 of 5 seeds
   [IR M24] (a V3 sensitivity, never an E-A slot).

## 5. Aqueous-complexant rule (IR M29; adopted unscored)

- **Record** complexant identity, concentration and pH for every cell and every new source.
- **Every Gd claim is defined, and the Gd term interpreted, only for nitrate and inert-salt media** (M29, adopted
  unscored); the frozen predictors B1 (fold β in-corpus, β_frozen on new data) and SH carry the Gd term unchanged in
  every stratum (D10). Complexant cells form a separate stratum in every table that splits by medium. The variant "B1@4
  and SH with u = 0 for AQ_COMPLEXANT (M01 / M02) targets" is registered now as a PLANNED C-ECT sensitivity on new data
  only (REGISTRY `C-ECT-S01`; CONFIRMATION_PACKAGE §4.2, §14), with the complexant stratum reported separately.
- **Frozen medium map** (S0: `mechanism_class_map.csv`, rule type `medium`, first match wins): M01 lactate, citrate,
  DTPA, HEDTA, CDTA, EDTA, NTA, oxalate, tartrate, malonate, glycolate, push-pull, aqueous BLPhen / phen, buffer →
  AQ_COMPLEXANT (excluded from Gd claims); M02 **acetate decided in advance as a complexant** (excluded from the Gd
  claims; inclusion is a sensitivity) [I2 confirm-2]; M03 thiocyanate, sulfate / H2SO4, phosphoric, picrate,
  trichloroacetic → NON_INERT (outside the Gd claims); M04 HNO3 / nitrate / NH4NO3 / NaNO3 / EAN → NITRATE (eligible);
  M05 perchlorate / chloride / HCl / CaCl2 → INERT_SALT (eligible); M06–M99 not stated → UNKNOWN (not eligible for a
  Gd claim until the medium is resolved from the source; the Stage-0 supply counts treat UNKNOWN as possibly eligible,
  so they are upper bounds). An "aqueous <ligand>" token flags the unit out of the Gd claims (flag F02).
- **E-CT:** aqueous-complexant changes are allowed as **T1** (aqueous-medium change); the complexant exclusion applies
  to the Gd claims only (brief §7.5).
- In the corpus: the pub_1605e436d7 TODGA/DTPA system and the pub_84113c17ad component-error system are complexant
  systems (IR M29; I2 reconciliation check); the V2-3 cells await S1 (§1).
- The quantitative external paired test of M29 (complexed fraction × the ligand's Gd chord; ≥ 8 pairs from ≥ 3
  laboratories outside JAEA) is logged in the supply ledger only [IR M29].

## 6. Riders recorded on every new source (brief §5.4, §7.5; never package claims)

Each new source that arrives for any reason carries these fields; metadata fields are recorded at the label-blind
audit, value-based fields are computed by the sealed evaluator only after the predictions are sealed.

| rider | content | source |
|---|---|---|
| Y present | yttrium measured (yes / no); a separate Y/Ho estimand is only registered for the future | IR M30 |
| rigid / preorganised | the rigid-SMARTS tag | IR M32 |
| non-DGA Gd triad | the unit reports Eu, Gd and Tb with a non-DGA extractant (mechanism class from §8.2) | IR M31 |
| multi-ligand campaign count | the number of extractant series with ≥ 6 Ln in the same campaign (E-LAB eligibility: ≥ 8 independent laboratories with ≥ 2 extractants × ≥ 6 Ln) | I2 lab-1; spec ELAB |
| shift-3 Ce/Pr acid sign (DGA-class, mineral acid) | frozen Ce/Pr and La/Pr acid-slope signs; the level-conditioned residual (per-metal acid slope against per-metal mean log D on Nd–Lu; linear primary, quadratic sensitivity); the laboratory. A unit counts if the Ce/Pr slope SE ≤ 0.2 per decade with Ce and Pr in [−2, 2.5]; complexant-driven series excluded by metadata. Sign only, no size claim | I2 shift-3; brief §7.5 |
| M07 Nd sign (sc009 publications) | the Nd chord sign and Pr/Nd squared error against B1; frozen u_Nd(sc009) = −0.11, shrunk with w = n/(n + 5), n = donor publications | IR M07; brief §6.8; spec EKX |
| M29 fields | complexant identity, concentration, pH (§5) | IR M29 |
| mechanism and flags | mechanism class, C-GD stratum flag, IL-diluent flag, synergist flag, aqueous-complexant flag (§8.2) | I2 confirm-2, confirm-4 |

## 7. Information contracts (brief §8)

### 7.1 E-CT (in-corpus and external)

| | in-corpus (Stage 1, seen) | external (C-ECT, new data) |
|---|---|---|
| **allowed** | the target's legal reads (union of the cached fold sequence `cell['seqs']['DOPT_Q'][:r − 1]`, r = 2, 3, 4); the siblings' full measured V1 series (same publication, same extractant SMILES, finite at every target metal; CLEAN; bit links removed); the fold's B1 β (round-4 `fitted_<design>.json`, fitted without the target's publication); the siblings' OOF residuals for OFF (same seed and design; leave-target-out) | the target's legal reads from the frozen external DOPT_Q object (`r5.frozen.external_sequence`, 4 reads = `seq[:3]`, 3 reads = `seq[:2]`; CONFIRMATION_PACKAGE §14); the laboratory's other full-series conditions of the same extractant (leave-one-condition-out), restricted in the primary to siblings that differ from the target in a T1–T3 field (§8.4); **β_frozen** (S0: `beta_frozen.json`) |
| **forbidden** | the target's unread metals; the target or a bit-identical copy of it in any pool (§2.1; `guard_pool` raises); any quantity tuned on E-CT test rows | the same; plus any value not yet sealed by the separate extraction agent; labels are revealed to the evaluator only after the predictions are sealed |
| **disclosed** | K = 1.78 = (1 − r)/r with r = 0.36, a roughness correlation measured on these data (seen) [P1 §15] | the same K (frozen) |

- The frozen module is `r5/ect.py`: read sets, β and the sibling pool are explicit inputs (`make_target`,
  `external_target`, `external_read_sets`), so the same code runs on new data (spec ECT §12 item 1).
- Leakage checks (spec ECT §11): prediction intervention (every metal outside the read sets replaced by noise →
  bit-identical predictions) and three positive controls that must be caught (the target in its own pool; a
  bit-identical copy in its pool; the target in its pool with the guard off). TECHNICAL SMOKE, BPG seed 104729: 229
  targets, 0 differences, 130 / 130 / 130 positive controls caught (`results/ECT/SMOKE_COUNTS_BPG.json`).
- Placebos are blended at the arm's own weight and count-matched (I2 rules 13–14).
- **OFF_ALL** (brief §6.1) is OFF over every other non-V2-dropped C3 cell of the publication × extractant (no
  cond__-difference or coverage rule; P1_ct_build.py parity), bit links removed: same-condition and non-covering cells
  enter, and P1's word "CLEAN" for this pool means non-dropped, not the sibling rule (s0-items-6).
- External family flag (data integrity, not a void switch): a family is flagged if the across-condition RMS difference
  of in-window adjacent log SF exceeds 0.10 log [I2 confirm-4].

### 7.2 E-LAB (spec ELAB "Leakage")

Allowed: the target's 4 legal reads, the campaign's other extractant series in full, the fold's β. Forbidden: the
target's unread metals; the target or a bit-identical copy in a strict pool. Label-based sensitivities that read the
target's unread labels (`MX_LM10_NC05`, `NC03`) are labelled ORACLE [O].

### 7.3 Label-blind supply (brief §5.5, §8)

Supply audits read metadata only and assert that no D, log D or SF column is loaded (`r5.supply.read_meta`, forbidden-
column pattern in `r5/supply.py`; the loaded columns are listed in S0: `supply_audit.json` "columns_loaded"). Web
summarisers, if permitted, return metadata only; any qualitative direction that leaks back is disclosed (P4 received
four: P4-063, P4-018, P4-022, P4-051) [P4 §8].

## 8. Frozen keys and maps (brief §5.5)

### 8.1 Laboratory key — PROVISIONAL (U6 open; D6)

Files: S0: `lab_key.csv` (227 rows: 135 corpus references, 92 literature candidates; `status` = "PROVISIONAL (U6 review
pending)"), `lab_key_links.csv` (329 corpus links); code `r5/supply.py`.

- **Parsing.** Titles dropped; initials recognised; Family–Given order resolved by a Chinese-family-name list,
  cross-reference attestation and the representation majority (2 references carry Family–Given decisions).
- **Common-surname links dropped (D6).** Primary: the four surnames named in I2 lab-4 (liu, zhang, zhou, xu) **plus P4's
  own common-surname list** (`P4_check_candidates.py` COMMON) never count toward a ≥ 2-shared-surname link.
  **Sensitivity:** only the four named surnames dropped.
- **Links.** L2 = ≥ 2 shared distinctive family names; LPI = the same person (family + first initial) in a PI position
  (first or last author) of both references. Conservative laboratory = connected component of L2 or LPI; liberal
  laboratory = the senior (last-listed) person, no transitive merging.
- **Literature candidates.** Liberal = P4's hand `lab_id`; conservative = P4's merge rule (CIAC groups, UFV / Aveiro, KU
  Leuven, ISSP-Turanov / INEOS-Odinets). **Institution-link (P) rule applied consistently** [PC P4]: CSM-Shafer,
  Osaka-Shinohara and Kanazawa-Imura are P (not new); P4-025 is Y by the two-co-author rule; P4-063 moves E1 → E2.
  Second-order links (Binnemans via P4-026; Kostikova via P4-001) are a sensitivity.
- **Counts (S0: `supply_audit.json`).** Corpus laboratories, conservative / liberal: all 135 registry references 45 /
  85; the 58 V1-cohort DOIs 24 / 44; the 39 C3 DOIs 17 / 31. Four-named-only sensitivity: 33 / 20 / 16 conservative
  (liberal unchanged). Corpus links 329 (L2 188, LPI 220; 141 single-PI-only); 386 in the sensitivity.
- **U6 review list** (prepared, not yet reviewed): the 141 single-PI-only corpus links, the 2 Family–Given references,
  the 3 institution fixes, the 2 second-order links and 25 hand-versus-automatic disagreements (S0:
  `supply_audit.json` "U6_review_list").
- **Use.** The laboratory is the unit of every Gd-amplitude and C-ECT claim (brief §4.2); in-corpus tables carry
  conservative and liberal laboratory columns. Without the U6 review the institution-link rule is applied mechanically
  and both counts are reported (brief §5.5). **The conservative laboratory key decides every claim's unit count and the
  supply audit; liberal counts are reported beside it (sensitivity).** Reason: merging can only lower G and puts
  dependent groups into one unit, so the decision stays conservative for both the minimum-G gate and the independence
  the sign-flip assumes (D6).

### 8.2 Mechanism-class map — frozen, applied blind to values

File S0: `mechanism_class_map.csv` (`r5.supply` MECH_RULES / MEDIUM_RULES): first matching rule on the chemotype label,
then on the extractant text, after removing an "in <diluent>" clause (IL-diluent flag) and "aqueous <complexant>" tokens
(complexant flag).

| rule | matches | class | C-GD stratum | E-CT report class |
|---|---|---|---|---|
| X01 | aqueous-only chelator or review | NOT_LLE | N | excluded |
| X02 | eutectic solvents | OTHER_EUTECTIC | N | other |
| X03 | **bifunctional ionic liquids** ([A336][P507], DGA-anion ILs, organophosphorus-anion) | IL_BIFUNCTIONAL | N | ionic liquid |
| X04 | **quaternary-ammonium / phosphonium nitrate or thiocyanate** (Aliquat, TOMANO3, [A336], [P66614]); **[A336][NO3] and TOMANO3 are one compound and count once** | IL_QUATERNARY | N | ionic liquid |
| X05 | **chelating cation exchangers**: β-diketones (HTTA), acylpyrazolones (HPMBP), 8-hydroxyquinolines, tropolone; **HTTA + TOPO and other chelator + neutral-donor synergists stay here** (synergist flag = sensitivity) | CX_CHELATING | **Y** | cation exchanger |
| X06 | carboxylic acids (Versatic, naphthenic, diglycolamic acid, crown-ether and phosphoryl carboxylic acids) | CX_CARBOXYLIC | **Y** | cation exchanger |
| X07 | acidic organophosphorus (HDEHP / D2EHPA, HEH[EHP] / P507, PC88A, Cyanex 272 / 301, phosphinic / phosphonic acids); acidic + neutral OP synergists stay here | CX_ACIDIC_OP | **Y** | cation exchanger |
| X08 | **CMPO-type phosphoryl acetamides**, phosphorylureas | NEUTRAL_CMPO | N | neutral solvating |
| X09 | **neutral organophosphorus** (TBP, TOPO, TRPO, Cyanex 923 / 925, phosphonate / phosphate esters) | NEUTRAL_OP | N | neutral solvating |
| X10 | soft N-donors: BTP, BTBP, BTPhen, **phenanthroline amides, PDA / dipicolinamides**, bipyridyl, triazinyl | SOFT_N | N (separate descriptive family; selectivity direction varies) | soft N-donor |
| X11 | amide O-donors: DGA, **malonamides**, DOODA, diamides | NEUTRAL_AMIDE | N | neutral solvating |
| X12 | crown ethers, calixarenes, macrocycles | NEUTRAL_MACROCYCLE | N | neutral solvating |
| X13 | other ionic-liquid extractants | IL_OTHER | N | ionic liquid |
| X99 | fallback | OTHER | N | other |

Flags: F01 IL diluent (C-GD sensitivity), F02 aqueous complexant (out of the Gd claims), F03 synergist / mixture
(sensitivity). The C-GD stratum is defined by this metadata class, never by the sign of the series slope (I2 confirm-2);
the reads-only slope sign computed without Eu, Gd and Tb is a sensitivity.

### 8.3 Chemotypes of new sources

New sources receive chemotypes from the **frozen gen6 Tanimoto single-linkage map** (`src/lanthanide_separation/gen6`;
the map behind the bench's chemotype ids), computed from the rebuilt extractant SMILES at the label-blind audit, never
from hand labels [IR M22; PC P4]. Stage-0 supply counts use P4's hand chemotype labels with [A336][NO3] = TOMANO3
merged (`chemotype_label_merged`); those counts are **upper bounds** (the gen6 map can merge, for example, diglycolamic
acid or [A336][BDGA] into the DGA cluster [P4]) and PROVISIONAL. This is a Stage-0 decision with its reason and
sensitivity in DECISION_LOG D6 ("Decision, chemotypes of candidates"): the verified decision units each hold one
extractant, so the decision supply and the kill cannot change under the map.

### 8.4 T0–T3 classification at pair level (brief §5.5; I2 confirm-4; D5)

- **T0**: pH, [HA] or phase-ratio changes only, for a cation exchanger, with no pH-dependent aqueous complexant or
  buffer, trace loading, below the acidity where solvation extraction starts. T0 pairs are near-copies by construction
  for ideal cation exchange and serve only as a data check.
- **T1** aqueous-medium change (aqueous-complexant changes included); **T2** organic-phase change (diluent, modifier,
  synergist, extractant concentration for a non-cation-exchanger); **T3** mixed (T1 and T2).
- Classification is **per target–sibling pair** (not per family), from the frozen sibling-factor rules T01–T07 of
  `mechanism_class_map.csv` (first match; "a|b" = a for a cation exchanger, b otherwise). The C-ECT primary SH uses only
  T1–T3 siblings; a target is admitted only if at least one remains; the T0-inclusive version is a sensitivity.
- **Chiba-Katsuta (P4-007; D5): its pH-only pairs are T0 and its diluent-change pairs T2 — PROVISIONAL until values
  arrive** (metadata `t_classes` = "T0;T2"; medium UNKNOWN in the metadata; S0: `lab_key.csv`).

### 8.5 Units and copy groups for new data

New-source unit: laboratory × chemotype; publications aggregated within a unit; **copy groups merged at arrival from
metadata, never retroactively** [IR M34]. A new publication forms its own `cluster_cpg` cluster unless the frozen key,
DOI or chemotype links it to an existing one; such links are frozen before its labels arrive (round-4 data contract,
"Groups used for uncertainty"). Value-based duplicate links of new data come only from the value-sealing step (§2.1
item 5).

### 8.6 Frozen numeric inputs referenced here

β_frozen (S0: `beta_frozen.json`, −0.110809, D2) and the external DOPT_Q object (S0: `ext_dopt.json`, pickle
`cache/ext/ext_dopt.pkl`, D1) are specified in CONFIRMATION_PACKAGE §11 and §14.

## 9. Open and PROVISIONAL inputs (fixed by append-only amendment before any new value is read)

| input | status | what fixes it |
|---|---|---|
| laboratory key, conservative / liberal counts | PROVISIONAL | U6 review (DR U6) |
| chemotypes of literature candidates | PROVISIONAL (hand labels; upper bounds) | gen6 map at the audit (U5 Phase A or values) |
| Chiba T0 / T2 pair classification | PROVISIONAL | values (U5 / U1 access) |
| HDEHP / P507 planning amplitudes | PROVISIONAL (unverified) | U5 item 4 (CONFIRMATION_PACKAGE §12) |
| SAFE units | 0–3 expected | U2 |
| V2-3 cohort key | sensitivity stratum | S1 |
| source-stated limits and V3 | not available | U1 |
| external confirmation evaluator (sealed family predictions → family, laboratory and unit values; chords) | NOT YET WRITTEN; specified by CONFIRMATION_PACKAGE §4.2, §5, §13 (D11) | written, tested on synthetic families and sealed by amendment before any new value is read |
| HDEHP solvating-regime acid threshold (C-GD eligibility) | PENDING-U5 (CONFIRMATION_PACKAGE §10; D11) | primary SF tables, U5 item 4, by amendment before any new value is read |
| frozen in-corpus interval of Gd-dip transport's conformal-coverage reading | PENDING (CONFIRMATION_PACKAGE §7.2) | named with its nominal level by amendment before any new value is read |

## 10. Decisions applied in this file

D1 (external DOPT_Q object fitted on all 521 bench cells; §8.6); D3 (fallbacks; the unit rules of §8.5 feed them); D9
(V2-5 censoring of Gd chords primary; §4); D10 (the frozen predictors keep the Gd term; §5); D11 (deferred inputs; §9); D5
(pair-level T0–T3, Chiba PROVISIONAL; §8.4); D6 (PROVISIONAL laboratory key with the widened common-surname list;
§8.1); D7 (every annex row an item's code computes is PLANNED, so the data rules above apply to all of them); D8 (the
item specs' frozen pool and stratum choices, quoted in §2.3–§3). Reasons and sensitivities: `DECISION_LOG.md`.
