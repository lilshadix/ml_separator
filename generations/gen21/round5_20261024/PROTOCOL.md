# Gen21 round 5: protocol (Stage 0)

Brief: `../round5_design_20261023/CLAUDE_CODE_PROMPT.md`, cited as "brief §x". This document makes brief §3–§4 concrete for round 5. It also fixes the Stage-1 order (brief §1.2, §6.0), the amendment rules, the labels (brief §9.1) and the leakage chain (brief §8). Evidence: `../round5_design_20261023/RESEARCH_IDEAS.md` and the round-5 `EVIDENCE_INVENTORY.md`. Source tags (EB, IR, I2, P1–P5, PC, RI, DR, R4-x) are the brief's ("Source abbreviations").

Companion documents are sealed with this one by `scripts/r5_seal.py` (`manifests/stage0_manifest.json`, `manifests/stage0_seal.txt`):
- `DATA_CONTRACT.md`: the round-5 addendum (brief §5, the E-CT information contract, the frozen keys);
- `METRICS.md`: the round-5 addendum to R4-METRICS;
- `CONFIRMATION_PACKAGE.md`: CP-13 and the claims of brief §7.5, with power in worlds;
- `DECISION_LOG.md`: the M35, M34 and CP-13 outcomes, the Stage-0 decisions D1–D11 (D9–D11 added at the pre-seal review), the closure ledger and the pre-slot gate;
- `REGISTRY.csv`: every Stage-1 item and arm, each PLANNED or EXCLUDED, with its RNG seed;
- `DATA_REQUEST.md`, copied from DR;
- `EVIDENCE_INVENTORY.md`;
- the Stage-0 inputs: `results/stage0/`, `cache/ext/`, `cache/frozen/` and `SUPPLY_LEDGER.csv`;
- the code (`r5/`, `scripts/`, `tests/`) and the nine item specifications `results/<ITEM>/ITEM_SPEC.md`;
- the design-folder digests, the output of round 4's `r4_seal.py --verify`, the round-3 LF check, and the environment.

**Records.**
- `RUN_RECORD.md` logs every run start and end and every seal, in UTC. It is append-only.
- `AMENDMENTS.md` records every change after the seal, together with every result already seen. A reseal names the seal it supersedes, and the previous manifest and seal are archived (`stage0_manifest_<prefix>.json`, `stage0_seal_<prefix>.txt`).
- Status that changes after the seal (request answers, Stage-1 outcomes) goes below the line `<!-- APPEND-ONLY BELOW -->` of `DECISION_LOG.md` and into `RESUME.md`, the two updated together. It is never written into a sealed text. The seal digests only the text above the marker (a whole line; an inline mention such as this one never truncates), and it records the byte length and digest of every append-only text (AMENDMENTS.md, RUN_RECORD.md and every marker tail), so `--verify` fails when an earlier line is edited instead of appended.

**Decision ids.** D1–D11 are the Stage-0 decisions recorded in DECISION_LOG (D9–D11 added at the pre-seal review), each with its reason and its sensitivity. They are not DR's earlier "D1", which is request S1 (brief §7.0).

## 0. Status, and what was seen before this seal

Stage 0 freezes the task, the separate estimands, the decision and inference conventions, the confirmation-package frame, the closure ledger, the pre-slot gate and the Stage-1 registry. It does so before any round-5 score is computed.

**What round 5 can and cannot claim** (brief, opening box):
- 0.50 at four read metals (E-A) is not reached by any tested or proposed model on these 229 cells, and gains below about 0.01 R² are undetectable here [EB §3; IR "Bottom line" 1].
- Every in-corpus number has already been seen by rounds 1–4, the round-5 probes and their checks. In-corpus legs are consistency checks, not tests [IR rule 2].
- A positive and significant result in round 5 can come only from Stage-2 data that arrive after the freeze.

**Seen before this seal.** Everything below was computed on the same 229 C3 cells and 5 dev seeds, or on metadata, or on synthetic data:
1. Every round 1–4 result. That includes round 3's single final check, whose seeds are consumed and never reused, and round 4's five scored slots and their close-out [R4-AMENDMENTS A5].
2. The round-5 design: EB (E1–E6), IR (M01–M39 with `critiques.json`), I2, the probes P1–P5 and their skeptic checks (PC), RI, DR and the design REVIEW_LOG. All of it is label-based and EXPLORATORY.
3. The Stage-0 inputs, computed in workflow `wf_b941fb7d-d2c` (RUN_RECORD 04:42–07:33 UTC):
   - `results/stage0/beta_frozen.json`: a training-side fit on all bench cells. Its inner-CV scores were seen.
   - `ext_dopt.json` and `cache/ext/ext_dopt.pkl`: a label-free read policy whose covariance was fitted on labels of all cells (training-side).
   - `dupscan_*`: a label-based scan of V1 values for bit identity, constant offsets, quantisation and near copies.
   - `ect_population.json` with its cell and link CSVs: counts from metadata plus the bit-identity rule.
   - `lab_key.csv`, `lab_key_links.csv`, `mechanism_class_map.csv`, `supply_audit.json` and `SUPPLY_LEDGER.csv`: metadata only. No D, log D or SF column was loaded.
   - `size_gate.csv` / `.json`: synthetic nulls, plus seen corpus pools recentred to the null.
   - `power_worlds.csv` / `.json`: synthetic and seen-corpus worlds.
   - Later in the same session, before the seal (`wf_2ceec1d9-225`):
     - `size_gate_ect_chemotype.csv` / `.json`: the fresh size check of D3's C-ECT fallback;
     - `power_cgd_beta_sens.csv` / `.json`: chord power at the D2 β references;
     - `supply_reach.json`: metadata-only forecasts of reaching each minimum G.
4. The 100 A / B / BR / BQ fold records of item 1.0 (label-free read sets; no score). Cache poisoning gave identical records in 100 of 100 and isolation passed in 100 of 100 (`results/leakage/cache_poison_EA_CHECK.json`, `manifests/ea_check_caches.json`).
5. PRE-SEAL ENGINEERING CHECKS (`--smoke`; real inputs, surrogate or suppressed labels, every metric suppressed) of every item's code, 1.0–1.8, plus item 1.0's cache build. Items 1.1, 1.2, 1.3, 1.5 and 1.8 write `results/<ITEM>/SMOKE_COUNTS_<design>.json` (counts, shapes, construction identities and leakage positive-control counts only); items 1.0, 1.4, 1.6 and 1.7 write nothing. They ran first in workflow `wf_b941fb7d-d2c` and again after the pre-seal review's code changes (RUN_RECORD). Unit tests ran on synthetic or construction-only data.
6. The pre-seal review of workflow `wf_2ceec1d9-225`. Its confirmed findings are folded in before the seal (`results/stage0/PRESEAL_REVIEW.md`); the Stage-0 scripts `s0_sizegate.py`, `s0_sizegate_ect_chemotype.py`, `s0_power_worlds.py`, `s0_supply.py` and `s0_package_supplements.py` were rerun (synthetic or metadata only).

**No round-5 score was computed before this seal**: no Stage-1 arm was scored on real labels.

**Adaptivity (declared).** The Stage-1 item code, arm lists and report rules were written after every result above had been seen, including the probes' in-corpus values of the same statistics. Stage 1 is therefore a set of consistency checks and descriptive deliverables, not tests (brief §6.0). Every item's code, arm list and report rule is sealed with Stage 0, before the item's first scoring run (§10).

## 1. Hard rules (brief §1.2), made concrete

- **Git.** Only read-only commands that record state. Nothing is added, committed, pushed, checked out, branched, reset or stashed without the user's separate request. Round 4 stays uncommitted.
  - State at the start (RUN_RECORD 04:40:13 UTC): branch `gen21-logsf-target`, HEAD `b96b0fc`. `round4_20261022/` and `round5_design_20261023/` were untracked and pre-existing.
- **Earlier seals.**
  - Round 4 (`5bacfe4cdcfc8d63171c46497e06b80605cc7fb1e0e7531c5fd81daadb1bbaa5`) verified at the start of round 5, through round 4's own `r4_seal.py --verify`.
  - The round-3 seal string is `93d3839a256c…`. Its LF digests (`next_research_20261019/manifests/freeze_lf_digests.json`) matched at the start: the freeze manifest plus 78 of 78 code files.
  - Both checks are repeated by `r5_seal.py` on every seal and verify, and after every Stage-1 run (§12). If either fails, the round stops and reports the diff. Round 4 is never resealed.
- **Withheld seeds.** The withheld-seeds file is never searched for, listed or opened. `next_research_20261019/cache_final/` and `next_research_20261019/results/final/` are never used. New split seeds over the same cells would measure partition stability only, and none is registered.
- **Downloads, logins, web.** None without the user's permission for that specific item. The agent never logs in to SAFE or a publisher.
- **Files.**
  - Round 5 writes only inside `generations/gen21/round5_20261024/` and the session scratchpad.
  - It never modifies `round4_20261022/`, `round4_design_20261022/`, `round5_design_20261023/`, `dataset_all_metals/`, `generations/gen21/cache/` or any earlier generation.
  - Text is written with LF line endings (the Write tool or `write_bytes`), and digests are LF-normalised.
- **Earlier code.**
  - The only round-4 script round 5 executes is `round4_20261022/scripts/r4_seal.py --verify`.
  - `r3.configs.INNER_CACHE_DIR` is redirected to `round5_20261024/cache/_inner` before any r3 or r4 runner path, and again after any import of `r4.registry`.
  - `g21.cache.build_all` is never called. Fold records are built with `g21.cache.build_fold` and pickled under `cache/`.
  - Design-folder scripts that write files are never executed. Reuse is by copying into `scripts/` or `r5/` (the source's LF SHA-256 is recorded in the head of the copy and checked by the seal) or by import.
- **Environment.**
  - `D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore`, run from `D:\ml_separator_gh` with `PYTHONIOENCODING=utf-8`.
  - One scoring process at a time (never more than 3 parallel jobs of any kind on 8 GB RAM).
  - No `pip` or `conda` install. Package versions are recorded and verified by the seal.
- **Values from new sources** are extracted and sealed by a separate agent before modellers see them. Auditors of old sources see the DOI, the conditions and the metal list only [IR M22, M25; P5 §7].

## 2. Task E-A (frozen; unchanged from round 4; brief §3.1)

- **Target.** `logSF(A/B) = log10 D(A) − log10 D(B)`, where A is the lighter metal and `|Z_A − Z_B| = 1`. There are 12 pair types; Nd/Sm is not adjacent [R4-PROTOCOL §1].
- **Regime P0.**
  - V1 data and cohort C3 (`k_max >= 3`);
  - code budget k = 3, i.e. **four** measured log D values per held-out cell;
  - DOPT_Q read sets from the cached nested sequences, with read set = union of `cell['seqs']['DOPT_Q'][:3]`;
  - the fixed query set: 229 cells and 1,802 queries per seed;
  - dev seeds 104729, 130363, 155921, 196613 and 262147;
  - designs BP and BPG.
- **Rows** are round 4's out-of-fold parquets `round4_20261022/predictions/REF_{BPG,BP}.parquet`: 9,010 rows per design and 18,020 over both. They are loaded by `r5.common.load_rows`, which refuses a run whose round-4 intervention or identity check did not pass.
- **Metric.** Pooled strict-adjacent R² within each seed, then the seed mean. Balanced MAE is as in R4-METRICS: extractant, then chemotype, then equal chemotype weights inside each seed. R² is never clipped, and repeated predictions are never pooled. The decision currency is relative pair-SSE (§5; METRICS).
- **Reference** (brief §2.2): B1 = PWL_R plus one fold-fitted Gd amplitude.
  - R² 0.4521 (BPG) / 0.4519 (BP); B0 0.4180 / 0.4179; HIST_D7 0.4511 / 0.4506 (diagnostic) [R4-REPORT §1].
  - B1's balanced MAE / MSE / RMSE are 0.1011 / 0.0270 / 0.164. Its SSE/SST is 0.548, so B1 would have to remove 8.8 % of its SSE to reach 0.50 [EB §1.1].
- **Designs.**
  - **BPG decides**: it isolates gen19 copy groups. BP is a labelled sensitivity.
  - BP and BPG share 100 % of fold assignments (B1 row errors correlate 0.988), so they are never independent replications.
  - One fold holds 58.2 % of scored rows and is trained on 51–69 C3 cells [EB §2.14].
- **Inference geometry.** 13 `cluster_cpg` clusters, with Kish 2.24 and the largest holding 65 % of rows. Kish is 3.85 by chemotype and 18.8 by publication [EB §2.19].
- **E-A is closed to new model slots on this corpus** (§9): none is registered. Item 1.0 scores only the reference contrasts (§10).
- **Information contract:** unchanged from R4-DATA_CONTRACT. A predictor sees the four read metals' log D, the metal identities, the query set and quantities fitted on the outer training fold.

## 3. Separate estimands and substitution rules (brief §3.2–§3.3)

Each separate estimand answers a different question. None is ever substituted for E-A, and none for another.

| label | question | population and information | primary statistic, comparator on identical rows | round-5 status |
|---|---|---|---|---|
| **E-CT** condition transfer | a study has measured this extractant's series at other conditions; how well do 0–4 reads at a new condition predict the strict-adjacent log SF? | targets with ≥ 1 same-publication, same-extractant sibling that covers the target's metals; the siblings' full series are allowed inputs, the target's unread metals are not [P1 §2] | R² and relative pair-SSE of SH vs B1 | Stage 1 item 1.1 (seen); C-ECT data-gated |
| **E-LAB** campaign calibration | a new extractant is read at 4 metals in a campaign that reports other extractants' full series; do they help? | POPX / POPX0 of I2 lab-1 | relative SSE of MX_LM10 vs B1 | item 1.3 (seen) |
| **E-D** read-budget ladder | how does accuracy grow with the number of reads, and how much does placement matter? | C3 queries at 2–5 reads; fixed cohorts C4–C6 | pooled R² per rung on identical queries; B1 refit per k | item 1.6, descriptive |
| **E-DEC** direction calls | from the reads, which adjacent pairs can be signed, and how often wrongly? | E-A rows | risk–coverage, AURC, false-sign rate at \|y\| ≥ 0.10 | item 1.5, descriptive |
| **E-W** wider pairs | how well are cut-point SFs predicted from the same reads? | same reads and cells, pairs with ΔZ > 1 | per-cut skill vs LINE and B0 | item 1.7, descriptive |

Labelled descriptive axes are also never substituted:
- the **level window** (M09, item 1.4);
- **E-KX**, known-chemotype deployment of the sc009 Nd dip (M07, item 1.8);
- **E-RANK** (the M15 salvage) and the **E-REC** own-prior rows (items 1.1 and 1.5);
- the equal-weight breadth statistic **E-S**, a diagnostic only [P2 §6];
- **E-CENS**, closed except as one sensitivity line (DECISION_LOG ledger).

Data versions V2 and V3 are sensitivities only (DATA_CONTRACT). The new-laboratory claims are C-ECT, C-GD and Gd-dip transport (CONFIRMATION_PACKAGE).

**Substitution rules (sealed).**
1. No E-CT, E-LAB, E-W, E-DEC, E-KX, E-RANK, E-S, E-CENS or window value is placed next to E-A's 0.452 as if comparable.
   - Every such table carries its estimand label in its title and shows B1 **on the same rows** as the comparator.
   - For example, B1 is 0.531 on E-CT's PRIMARY rows, not 0.452 [P1 header; IR "Hard separations"].
2. **E-D.** The 4-read rung is E-A itself. The 5-, 6- and 7-read rungs are a different read budget: they are reported as levels under E-D and as paired contrasts on E-A's identical queries, labelled "different read budget", and never as E-A values. In any E-D table that shows C3 rungs, the C3 4-read cell reads "(E-A rows)" and no 4-read C3 level is printed (the writer blanks it and sets `level_note`); C3 levels at 5 or more reads appear only in such tables, labelled "different read budget", and as paired contrasts on E-A's identical queries. Fixed-cohort C4–C6 levels may be printed at every rung, including 4 reads (C4 0.4290 is not E-A).
   - The C3 5-read row is named "C3 min(k, k_max), k = reads − 1" in every headline. It is a mixture: 286 of 1,802 rows per seed (15.9 %) keep 4 reads. The like-for-like 5-metal number is the C4 cohort [PC P3].
3. **V2 and V3** values on other row sets are never placed next to 0.452, and every V3 number states its row count.
4. An equal-weight (per-source) statistic never replaces pooled R² [IR "Hard separations"].
5. Strict pairs are never pooled with wider pairs. "Strict + Nd/Sm" (0.551) appears only as a labelled SUBSTITUTION TRAP row [IR M13].
6. **Read counts.**
   - E-A's "k = 3" means four reads. E-CT and E-D rungs are stated in reads r, and the code budget is k = r − 1, counting DOPT_Q sequence steps; the first step already holds two metals.
   - Read sets are the union of `dseq[:k]`: 2 reads = `[:1]`, 3 = `[:2]`, 4 = `[:3]`, 5 = `[:4]`.
   - β fits use `fit_reference(bench, rec, Y, 'B1', k = r − 1)` (brief §3.3).
   - The capital K is reserved for the shrinkage constants: 1.78 in E-CT and 6.1 in E-LAB.

## 4. Evaluation labels (brief §9.1)

Every number carries exactly one **base label**:

| base label | meaning in round 5 |
|---|---|
| TECHNICAL SMOKE | a technical run: the post-seal `--repro` reproduction runs of §10 and the reproduction target that every scoring run executes first. A reproduction value is a technical check, not a result. The pre-seal `--smoke` runs carry the sub-label **PRE-SEAL ENGINEERING CHECK** (real inputs, surrogate or suppressed labels, every metric suppressed). |
| EXPLORATORY (seen) | every in-corpus number of round 5 (the 229 C3 cells, the 5 dev seeds, BP / BPG, and the 521 bench cells where an item uses them) |
| PARTITION STABILITY | new seeds over the same cells; never blind. None is registered in round 5. |
| INDEPENDENT CONFIRMATION | only Stage-2 data, scored once under the sealed CONFIRMATION_PACKAGE. None exists at this seal. |

A base label may carry qualifiers:
- **SENSITIVITY**: BP, V2, V3, C5-BTBP variants, duplicate variants, the data-version and β-recipe sensitivities;
- **ORACLE [O]**: a test-label quantity. It bounds headroom only, and is never a score, baseline or selection rule;
- **DIAGNOSTIC**: breadth tests; every E-A contrast after M34's kill (§6); HIST_D7; unit tests on in-corpus contrasts;
- **CONSISTENCY CHECK**: an in-corpus threshold that sits below, or was set after, a seen value.

The item specifications add row qualifiers, always beside one base label (adopted, D8):
- SECONDARY (E-DEC's Mondrian thresholds);
- NON-CLAIM (E-W's heavy cuts);
- SUBSTITUTION TRAP (E-W);
- "different read budget" (E-D);
- "best of about 25 arms on seen rows; expect regression" (E-LAB, on every number).

**Further rules.**
- The 5 dev seeds share cells, so "k of 5 seeds" is not a replication count [P3 §1].
- No in-corpus p-value is called significant.
- A bootstrap tail fraction, such as P(R² ≥ 0.50), is never read as a posterior probability (brief §9.2).

## 5. M35: the one-currency decision convention for E-A (brief §4.1)

Round 4's two margins were in different currencies. A balanced-MAE gain of 0.005 corresponds to about +0.034 R² under B1's gain structure (+0.053 under uniform shrinkage), and block A failed that leg even as a test-label oracle [EB §2.17; R4-POWER]. Round 5 seals one currency:

1. **E-A pooled R²** is still reported, for continuity.
2. **One gate: pooled relative pair-SSE reduction against B1 ≥ 2 %** (METRICS). For each seed, compute `1 − SSE_cand / SSE_B1` over all scored rows, then take the seed mean. At SSE/SST 0.548 this is about +0.011 R². It is the same margin restated in one unit, not new power [IR M35; EB §6.5]. A pass needs all three of:
   - (a) the seed-mean relative SSE against B1 ≥ 0.02 on **BPG** (C5-BTBP in);
   - (b) SF_POOL on the seed-mean per-unit ΔSSE of the 13 `cluster_cpg` units, one-sided p ≤ 0.025, H0: mean ΔSSE ≤ 0. Use exact enumeration over all 2^13 = 8,192 sign vectors. If G > 13, use B = 4,999 random flips with p = (1 + #{T* ≥ T}) / (1 + B) (METRICS; `r5.common.sf_pool`);
   - (c) the balanced-MAE guard of item 3, on the seed-mean point estimate.

   A pass is labelled **DIAGNOSTIC**, because M34's kill fired (§6). No E-A pass is a claim.
3. **Balanced-MAE non-inferiority guard:** the seed-mean point Δbalanced-MAE (B1 − candidate) ≥ **−0.001** [IR M35; E5 §5.2].
4. **BPG decides; BP is a labelled sensitivity** (same test folds) [EB §2.14].
5. **Mandatory tables for every E-A contrast** (item 1.0, `results/EA_CHECK/`):
   - the 13-row per-cluster table, with provenance columns: ORNL-only, and the `data_location` class (FIG / TAB / MIXED / NONE / OTHER) from `round4_20261022/results/provenance/cells.csv`;
   - the big fold against the other folds (per seed, the fold with the most scored rows);
   - C5-BTBP in and out, plus the single cell c950dc2982e96b83 out (DATA_CONTRACT);
   - the six-design table (A, B, BR, BQ, BP, BPG), restricted to the training-free arms B0 and HIST_D7. The A, B, BR and BQ read sets come from the 100 round-5 fold records `cache/<design>_<seed>_<fold>.pkl`. These were built with `g21.cache.build_fold` for 4 designs × 5 dev seeds × 5 folds (`gen20k.splits.folds_for`), passed cache poisoning (100 / 100) and isolation (100 / 100), and are keyed in `manifests/ea_check_caches.json`. Item 1.0 refuses a record whose key or bytes changed.
6. **Breadth diagnostic (sealed, never a gate):** the equal-weight per-cluster log SSE ratio (SF_LOG, with an IM t beside it).
   - SMALL_POOLED is the sealed diagnostic: clusters with < 3 cells or < 20 rows per seed are pooled into one "small sources" unit.
   - ORNL_COLLAPSED (sensitivity) collapses the ORNL-only clusters into one unit. RAW (reference) uses the 13 clusters unpooled.
   - Publication-level relative SSE inside the giant cluster is also reported [IR M35].
7. **Positive control, outside any family:** B1 − B0. Holm over the 14-contrast replay family would have given it Holm p 0.16 and killed it [IR M35].
8. **Appendix (M08 relation-class moment table, with an identifiability column):** **PLANNED** (D7; `REGISTRY.csv`). It is computed by item 1.0 (ITEM_SPEC §7) and reported as "M08 appendix" [IR M08 salvage].

**Why the equal-weight leg is not a gate.**
- It fires on HBK@ML − B0 (exact sign-flip p 0.0115, close to B1 − B0's 0.0137), although HBK@ML is +0.13 % pooled [IR M35]. P2 §2 reports the same contrast as +0.14 % BPG, −0.64 % BP.
- Under a skewed heterogeneous null, SF_LOG reaches 0.111 (C-LOG, chemotype units) and SF_POOL 0.103 (C-POOL, cpg), so skewed nulls are non-gating for both [P2 §3; PC P2].
- **The specific M35 worlds were not run in Stage 0**: size under a heterogeneous null, power for a homogeneous 2 % effect, a gain only in tiny clusters, a ≥ 3 % gain confined to the giant cluster [PC P2]. As brief §4.1 allows, the demotion of the equal-weight leg is recorded in DECISION_LOG **as a decision, not a test result**.

**Replay facts recorded as seen** [IR M35; PC P2]:
- B1 − B0 is +5.85 % pooled relative SSE (t 1.99), +12.3 % equal-weight (IM p 0.012) and +14.75 % log ratio (t 2.88).
- Every round-4 slot is ≤ −0.02 % pooled. Outside the big fold no slot reaches 2 % (the best is S02, +0.28 to +0.35 %).
- cpg SF_POOL p is 0.00317 (BPG) / 0.00269 (BP).

**M35's own kill** (abandon the convention only if the fixed-sequence replay fails the positive control [RI §4.2]):
- On the seen replay the convention **passes** its positive control.
- Item 1.0 recomputes this as `M35_passes_its_positive_control` (C1 = B1 − B0 passes on BPG, C5 in). The seen values are compared as a CONSISTENCY CHECK: +5.85 % to 2 decimals in %, and p 0.00317 / 0.00269 to 5 decimals. A mismatch is reported; it does not stop the item.
- If the recomputed positive control fails, M35 is abandoned by amendment. No E-A slot is registered, so no other decision depends on it.

## 6. M34: the inference convention (brief §4.2)

**Tests** (P2's ids; one-sided, H1: the candidate is better; nominal 0.025 primary, 0.05 sensitivity; definitions in METRICS) [P2 §1, §3; PC P2]:

| test | estimand | round-5 status | evidence |
|---|---|---|---|
| SF_POOL: exact sign-flip on pooled ΔSSE (exact at G ≤ 13, else 4,999 random flips) | pooled | **primary for pooled claims** | never flagged under the gating nulls |
| SIGN: exact binomial sign test on unit wins, ties dropped | π, the share of units that improve | **primary for win-share claims**; conservative (mean size 0.012) | never flagged |
| SF_LOG, T_LOG, IM_REL | equal-weight breadth (E-S) | **diagnostics only** | they answer "does the average source improve?" |
| WCR_POOL (wild cluster bootstrap, CR2, Webb) | pooled | comparator only | 1 gate flag |
| SEALED (round-4 R² leg: percentile, BCa, jk-t(G − 1), Kish ≥ 3) | pooled ΔR² | comparator only | 2 gate flags |
| WCR_LOG | breadth | dropped | 6 flags; size up to 0.113 |
| JK_KISH (cluster jackknife-t, Kish df) | pooled | not a gate | real-pool maximum size 0.0515; flags start at G = 8 |
| **JK_G1** (round 4's sealed jackknife-t, df G − 1) | pooled | **dropped** | anti-conservative: real-pool mean size 0.047, maximum 0.096 (0.1245 on synthetic pools) |

**What the exact tests guarantee, and what they do not.**
- SIGN, SF_LOG and SF_POOL hold their size under the arm-swap and symmetric nulls **by construction** (0.004–0.027 in every checked cell). That is exactness under a sharp, exchangeable null, not an empirical finding [PC P2].
- Under skewed heterogeneous nulls, no test holds 0.025 at cluster or chemotype granularity: SF_POOL reaches 0.103 and SF_LOG 0.111. Publication-level skewed nulls stay ≤ 0.041, except SIGN under C-MED (0.106 at τ 0.065, G 80) [P2 §3].
- **Laboratory clustering.** Family-level sign-flips have size 0.026–0.054 at ICC 0.3–0.5 [I2 confirm-4]. Two-stage (chemotype → laboratory) nulls give 0.041–0.049 for balanced compositions at ICC 0.3, and 0.06–0.14 otherwise [I2 confirm-1].
  - The laboratory unit removes within-laboratory clustering only. Chemotype clustering among laboratories is handled by the size gate (§7, item 2).
  - Round 5's Stage-0 size gate (20,000 replicates per cell; `results/stage0/size_gate.json`) flags every chord statistic and both laboratory-level C-ECT statistics. The fallbacks are sealed as D3 (§7). The seen-pool skewed nulls of C-ECT are reported in both orientations as descriptive rows, never gating (brief §4.2).

**Units.**
- **New-source unit: laboratory × chemotype.** Publications are aggregated within a unit. Copy groups are merged at arrival from metadata, never retroactively [IR M34].
- **Gd-amplitude contrasts: the laboratory,** aggregated across chemotypes. The Gd chord is shared across a laboratory's papers: +0.37 to +0.47 on publication means, and +0.73 at cell level for the same extractant [I2 rule 12, lab-4].
- **E-CT on new data: the laboratory,** with families nested and each laboratory weighted 1; the family-level p is reported beside it [I2 confirm-4]. Round 5 seals the chemotype aggregation of D3 on top of this (§7, item 2).
- **In-corpus diagnostics:** `cluster_cpg` for E-A. For E-CT the unit is publication × chemotype (P1's unit); cluster-level inference is "not estimable", because PRIMARY has 3 cpg clusters, one of them holding 89 % of rows (Kish 1.25) [P1 §3, §12].

**Multiplicity.**
- One-sided 0.025 per claim, and no global claim.
- No Holm inside a claim at small G: with exact tests, G = 6 cannot reach 0.0125, because the smallest p is 1/64 = 0.0156.
- Fixed-sequence gatekeeping inside a claim [I2 rules 9–10, confirm-1, confirm-4].

**G floors, declared now.**
- G ≤ 5 cannot reject at 0.025 with SIGN or SF: the minimum p is 1/32 = 0.031 [P2 §3].
- C-ECT needs ≥ 7 units: laboratories as written in brief §7.5, chemotype units under D3.
- Gd-dip transport runs with ≥ 10 laboratory × chemotype units on the merged SAFE + literature ledger, and decides with ≥ 10 laboratories.
- C-GD, its pre-specified stratum, needs ≥ 6 laboratories from ≥ 3 families.

**Power is stated in worlds, never as one number** [I2 rule 11]:
- a corpus-DGA world, a halved-effect world, and a cation-exchanger or null-like world, plus 25 % and 50 % null-like mixtures where relevant;
- `results/stage0/power_worlds.json`: 10,000 replicates per world, one-sided 0.025, tables in CONFIRMATION_PACKAGE.
- Corpus-resampled power assumes that new units are exchangeable with corpus units. The new supply is not: P4 finds that 15 of 16 probable new laboratories use chemotypes the corpus lacks.

**Planning numbers (B1 − B0 positive control, one-sided 0.025; corrected values)** [P2 §4; PC P2]:

| unit | pooled tests, τ = 0 | pooled tests, τ = 0.065 | equal-weight SF_LOG |
|---|---|---|---|
| publication-like | SF_POOL ≈ 43; SEALED about 38–41 (P2 grid 39 BPG / 38.3 BP; direct evaluation ≈ 41); JK_KISH ≈ 40 (not a gate) | > 80 | 19–20 (τ 0); ≈ 26 (τ 0.065) |
| chemotype-like | 30–58 for the pooled tests (SF_POOL ≈ 28–30; WCR_POOL 40; SEALED ≈ 41; JK_KISH 58 on the P2 grid, ≈ 50–55 by direct evaluation); SIGN (π, not pooled) ≈ 55–58 | > 80 | never within 80 |
| publication × chemotype | 59–72 | > 80 | never within 80 |
| non-DGA publications only | SF_POOL 16 | 80 | SF_LOG 12 / 17 (SIGN, a win-share test: 11 / 27) |
| DGA publications only | > 80 | > 80 | 62 / 80 |

- A 2 % effect (B1 scaled to the M35 gate) needs > 80 units at τ = 0.065 for every test and unit, except the publication-level log-ratio tests at 74–77 [P2 §4]. That pool inherits B1's consistency, so it is optimistic for real candidates [PC P2].
- B1's per-unit gains were measured on the cells on which B1's form was chosen. Every G80 above is therefore an upper-side (optimistic) estimate for genuinely new sources [PC P2].

**M34's kill outcome (recorded in DECISION_LOG).**
- At G = 8 publication units, the gate-valid power for B1 − B0 is 0.228 (BPG) / 0.236 (BP) at τ = 0.065.
- On the pooled decision estimand, SF_POOL is 0.285–0.296 at τ = 0.
- Both are below 0.30, so **the kill fires on the pooled estimand at both τ** [PC P2].
- Sealed consequence: the 40–80-plus-source planning number is kept for any pooled E-A claim, and **every round-5 model contrast is run as a DIAGNOSTIC** [IR M34].

## 7. CP-13: the frozen confirmation-package frame (brief §4.3; claims in CONFIRMATION_PACKAGE.md)

**Supply.** About 7.0–11.5 new laboratories are expected with PC P4's corrections (13.5 under P4's own tier rule; 5–6 verified), so only a few questions are decidable [I2 confirm-1; PC P4]. The frame below is sealed now. The claims, the power tables in worlds and the operating characteristics are in CONFIRMATION_PACKAGE.md.

1. **Per-claim minimum G and a supply-scenario power table,** recomputed laboratory by laboratory from P4's frozen `lab_id`. These are G_HS, G_Gd and G_ECT. The power table is in CONFIRMATION_PACKAGE.
   - **Laboratory-by-laboratory supply** (`supply_audit.json` → `claim_supply_lab_by_lab`):
     - **G_ECT:** verified 2 (Chiba-Katsuta, KULeuven-Binnemans); 9 if every E1 + E2 candidate converts.
     - **G_HS:** verified 1 (Chiba-Katsuta); 3 under E1 + E2; 9 under E1 + E2 + U.
     - **G_Gd:** verified 2 laboratory × chemotype units; 10 under E1 + E2; 20 under E1 + E2 + U.
   - **Probability of reaching each minimum** (`results/stage0/supply_reach.json`). These are forecasts under P4's rates (pE2 0.5, pU 0.25) and PC P4's rates (pE2 2/6, pU 0), reported and never counted (D4):
     - C-ECT, 7 laboratories: 0.2266 / 0.0453.
     - C-ECT under D3's chemotype units: ≥ 7 distinct chemotypes 0.50 / 0.2757. Counted as linked chemotype units (chemotypes joined through a shared laboratory), there are at most 6 even if every candidate converts, so the probability is 0.
     - C-GD, 6 laboratories: 0.0623 / 0.0. This is an upper bound, because the "≥ 3 families" condition is not modelled.
     - Gd-dip transport, 10 units with SAFE at its maximum of 3: 0.8459 / 0.0879. With SAFE at 0: 0.3005 / 0.0002.
     - SAFE route: 0 (at most 3 units against 10).
2. **Size gate.**
   - **Rule.** A test is dropped before sealing if its size exceeds 0.035 (the lower limit of the two-sided 95 % Clopper–Pearson interval, a one-sided 97.5 % bound; P2's convention) at nominal one-sided 0.025, under the i.i.d. nulls or under two-stage (chemotype → laboratory) nulls at ICC {0.23, 0.3, 0.5, 0.69}. Skewed seen-pool nulls are reported and never gate (brief §4.2).
   - **Run.** `scripts/s0_sizegate.py` → `results/stage0/size_gate.{csv,json}`, before the seal. It used 20,000 replicates per cell (≥ 2,000 required), the compositions of `ideas2/confirm1skepB_out/clustered_null_size.csv`, within-laboratory sd {0.05, 0.08, 0.10}, i.i.d. sd {0.10, 0.12, 0.14, 0.16}, and seeds `default_rng(20261023 + crc32(key))`.
   - **Outcome: every gated statistic is FLAGGED.**
     - C-GD winsorised chord interval (CGD_W, primary): 328 of 496 gating rows flagged. Maximum size 0.1471 (two-stage, ICC 0.69); 0.05495 at ICC 0.23 (confirm rule); i.i.d. maximum 0.02775 (confirm) and 0.05065 (kill / harm).
     - C-GD raw interval (CGD_R, sensitivity): 328 of 496 flagged; maximum 0.1434.
     - Gd-dip transport interval (GDT_R): 322 of 480 flagged, 48 of them at a decidable G; maximum 0.1434.
     - C-ECT laboratory-level exact sign-flip (ECT_SF): 21 of 43 gating rows flagged, 11 at a decidable G. Maximum 0.0979 (ICC 0.69); 0.04535 at ICC 0.23; i.i.d. 0.0265. Under the seen skewed pool (descriptive) it reaches 0.0083 right-skewed and 0.0758 mirrored.
     - C-ECT laboratory-level SIGN (ECT_SIGN): 11 of 43 flagged, 1 at a decidable G; maximum 0.0979; i.i.d. 0.0169.
     - Under a one-sided 95 % bound 332 / 332 / 326 rows flag for CGD_W / CGD_R / GDT_R (12 more cells: comp (2, 2, 2), ICC 0.23), with the same verdicts; the C-ECT rows are unchanged.
   - **Sealed fallbacks (D3).**
     - **C-GD (winsorised primary, raw sensitivity) and the Gd-dip transport interval → fallback (i).** They move to the item-4 exclusion list as **descriptive intervals**. Their readings against 0, β_frozen/2 and β_frozen are printed as descriptive readings, never as confirm, kill, helps or harm decisions. A descriptive interval is not a rejection, so the B1-harm switch of item 6 cannot fire from these intervals in round 5.
     - **C-ECT → fallback (ii).** C-ECT unit = a connected component of the laboratory–chemotype graph of eligible families (chemotypes from the frozen gen6 map, DATA_CONTRACT §8.3), so that no laboratory contributes to two units; unit value = unweighted mean of its laboratories' values (each the unweighted mean of its families' log ratios); exact one-sided sign-flip over units; minimum G = 7 units. With no spanning laboratory this is exactly one value per chemotype.
       - The declared test is the exact one-sided sign-flip over these units, with **minimum G = 7 units**. The kill rule and the fixed sequence SH@4 → SH@3 are read on the same unit values.
       - **The fresh size check passes on every cell.** It was run by `scripts/s0_sizegate_ect_chemotype.py` and written to `results/stage0/size_gate_ect_chemotype.{csv,json}`, before the seal. It used the two-stage nulls at the ICC grid plus i.i.d. normal, t3 and seen-pool nulls, 17 compositions, and 20,000 replicates per cell.
         - 0 of 80 gating rows are flagged (56 of them at a decidable G).
         - The maximum size is 0.0268 (two-stage, ICC 0.3, 13 chemotypes with one laboratory each). Per ICC the maxima are 0.02455 / 0.0268 / 0.02475 / 0.02635.
         - The chemotype-level SIGN, reported beside it, has maximum size 0.0121.
         - Chemotype values are independent and symmetric about 0 under these nulls, so the exact sign-flip holds its size by construction (§6).
         - Under the seen skewed pool (descriptive, D3) the chemotype-level size is at most 0.0083 right-skewed and 0.0573–0.0723 mirrored (left-skewed): like every test here it does not hold 0.025 under skewed nulls, which are non-gating (brief §4.2).
       - The laboratory-level sign-flip and SIGN are reported beside it as descriptive. The same run reproduces their failure on fresh draws: ECT_LAB_SF has 37 of 68 rows flagged, with maximum 0.0972.
3. **One value per laboratory** for every Gd and E-CT claim (chemotype aggregation on top of it for C-ECT, D3). Laboratory-bootstrap power at G = 6 is 0.742, against 0.736 for units, so this costs nothing [I2 confirm-1].
4. **Exclusion list, sealed in advance** (not decidable with the supply):
   - pooled or E-A claims;
   - a TOST on β (power ≤ 0.08);
   - the window as a row flag (≤ 0.15);
   - "two reads plus your own series", i.e. SH with 2 reads (≤ 0.06);
   - the sc009 Nd dip;
   - any DGA-only Gd test [I2 confirm-1];
   - by D3: the C-GD chord intervals (raw and winsorised), the Gd-dip transport interval, and the laboratory-level C-ECT sign-flip and SIGN, all as descriptive statistics.
5. **Three-way decisions with explicit inconclusive zones.** Every kill rule is stated with its operating characteristics, P(kill | the corpus effect is true) and P(pass | null), at the realistic G (CONFIRMATION_PACKAGE, from `power_worlds.json`) [I2 rule 9].
6. **Non-rejection never switches the deployed model.**
   - Failing to reject "a Gd dip exists" reads INCONCLUSIVE, never "withdraw B1's Gd term". That rule would withdraw a real effect about 17 % of the time at G = 8 [I2 confirm-1], and an exactly calibrated term in 56 % of G = 8 replicates [I2 confirm-3].
   - Only a rejection can change the default: a C-GD 97.5 % lower bound above β_frozen/2 would switch new chemotypes to B0 [I2 confirm-3].
   - Under D3 no C-GD decision is available in round 5, so B1 with β_frozen stays the default for new chemotypes.
7. **Supply-audit kill (metadata only; D4).** `scripts/s0_supply.py` → `results/stage0/supply_audit.json`, run at the end of Stage 0. It is rerun after each answered request (brief §4.6).
   - **Decision supply** = the units verified in hand from metadata, under the frozen key with the institution-link rule applied consistently, plus SAFE's maximum of 3 where SAFE units count (U2 unanswered); C-ECT counts linked chemotype units (D3 ii). The conservative laboratory key decides every claim's unit count and the audit; liberal counts are reported beside it (D6). The kill reads the test claim (C-ECT); C-GD and Gd-dip transport are descriptive intervals (D3 i).
   - Forecasts are reported, never counted: P4's rates (pE2 0.5, pU 0.25) give G_ECT 5.5 laboratories, G_HS 3.5, G_Gd 8.5 (literature only; 11.5 with SAFE at its maximum 3); PC P4's rates (pE2 2/6, pU 0) give 4.33, 1.67, 4.67 (7.67 with SAFE).
   - **Outcome:**

     | claim | decision supply | minimum G |
     |---|---|---|
     | C-ECT (linked chemotype units) | 2 | 7 |
     | C-GD | 1 | 6 |
     | Gd-dip transport | 5 (2 literature + 3 SAFE) | 10 |
     | SAFE route | 3 | 10 |

   - Every claim is below its minimum, so **the kill fires for every claim**. No confirmation runs, and round 5 is reported as descriptive (brief §7.6).
8. **Eligibility for the Gd claims** (C-GD, Gd-dip transport, A3) [I2 confirm-1; mechanism map `results/stage0/mechanism_class_map.csv`]:
   - the unit itself reports Eu, Gd and Tb;
   - values below DL or LOQ are excluded;
   - aqueous-complexant media are excluded (M29 rule): lactate, citrate, DTPA, HEDTA, CDTA, EDTA, NTA, oxalate, tartrate, malonate, glycolate and buffers. **Acetate is decided in advance as a complexant**, and its inclusion is a sensitivity;
   - the unit carries a frozen mechanism-class tag.

   C-ECT has its own eligibility: ≥ 3 full-series conditions of one extractant in the laboratory, T0 excluded at pair level, and aqueous-complexant changes allowed as T1.
   - **D5:** the T0 / T1–T3 classification is made at pair level. Chiba's pH-only pairs are T0 and its diluent-change pairs T2, both PROVISIONAL until its condition tables are read.
9. **β_frozen** (D2; `results/stage0/beta_frozen.json`; recipe `r5/frozen.py`, `scripts/s0_beta_frozen.py`).
   - **The fold record.** Outer `train_index` = all 521 bench cells, with an empty test index. Design BPG, split seed 104729, fold 0, inner seed `gen20k.splits.inner_seed(104729, 0)` = 3246616. Its 4 inner blocks are those of `g21.cache.build_fold`. All 229 C3 cells have inner records, with 0 orphans. The record is `cache/frozen/fold_all_BPG_104729_0.pkl` (SHA-256 `7c862d87…`).
   - **The fit.** `fit_reference(bench, rec, Y, 'B1', k=3)` with the default grid {0.003, 0.03, 0.3, 3, 30, ∞}, ties toward the stronger penalty, one BLAS thread. It used 1,374 situations and 10,812 rows.
   - **Sealed value: β_frozen = −0.110809** (−0.11080937578907438), λ = 0.003, the inner-CV value.
   - **Disclosed fragility.** λ 0.003 beat λ 0.03 by 7.6e-5 in inner score (0.0278030 against 0.0278790; λ = ∞ scores 0.0299620).
   - **Sensitivities, reported beside it:**
     - design BP: −0.1095 (λ 0.003);
     - λ fixed at 0.03: −0.0878 (round 4 chose λ 0.03 in 23 of 25 folds of each design);
     - the round-4 fold-mean β: −0.0923 BPG (sd 0.0195, range −0.140 to −0.030) / −0.0949 BP [PC P3; P5 §4b].
   - **C-GD power** is reported at β_frozen and at each of these sensitivities (CONFIRMATION_PACKAGE).
     - `results/stage0/power_cgd_beta_sens.{csv,json}` covers β_frozen, BP −0.1095, λ 0.03 −0.0878 and the round-4 fold means −0.0923 / −0.0949, with 10,000 replicates per world. Its β_frozen rows reproduce `power_worlds.csv` exactly (768 of 768 rows).
     - `power_worlds.json` also covers the brief's −0.118, −0.072 and −0.094. At β = −0.118 power falls to 0.48 at G = 6 and 0.71 at G = 8; at −0.072 it is 0.89 / 0.99 [I2 confirm-1].
   - **β_frozen's anchor depends on U1.** B1's Gd transfer to the located documents is learned mainly from the 69 unlocated cells. Refitted without them, the β mean is −0.0708: a seed-unstable tie on located rows, and +0.0119 with λ fixed at 0.03 [P5 §4b; PC P5].
10. **Planning amplitudes per extractant family, from primary SF tables** (D6). HDEHP −0.26 and P507 ≈ −0.105 are **unverified** (a planning value and a search-snippet reading) [I2 confirm-1, confirm-2]. U5 item 4 is open, so both are sealed **PROVISIONAL (PENDING-U5)**, with power reported for an HDEHP-like and a P507-like centre separately. They are fixed by an append-only amendment before any new value is read.
11. **Chord hygiene:** V2-5 floor censoring, and a resolution rule for sources that report D to one decimal (about ±0.05 log of quantisation noise, comparable to β_frozen/2). Both are written in CONFIRMATION_PACKAGE §13 [I2 confirm-1, confirm-3]: V2-5 censoring of Gd chords is primary (with source-stated DL / LOQ), the chord without it a sensitivity; the resolution rule flags cells (D9).
12. **Freeze before any new value is read:** the predictor, β_frozen, the read policy, the censoring rule, data inclusion, the analysis code and the hashes. Two Stage-0 inputs are deferred and sealed by amendment before any new value is read (D11; DATA_CONTRACT §9): the external confirmation evaluator and the HDEHP solvating-regime acid threshold.
    - The evaluator exposes only the legal reads to prediction and reveals hidden labels only after the predictions are sealed. The primary confirmation runs once [R4-BRIEF §9].
    - **External read policy (D1)**, `results/stage0/ext_dopt.json`, `cache/ext/ext_dopt.pkl` (SHA-256 `fc261e21…`; rebuilt bit-identical):
      - A new cell's reads come from `gen20k.design.support_sequence(dz, 'DOPT_Q', min(k_max, 6), cov=C_all, noise_var=2·s2_all)`; 4 reads = `seq[:3]`, 3 reads = `seq[:2]`.
      - `C_all` and `s2_all` are fitted once with the `g21.cache.build_fold` outer recipe on **all 521 bench cells**: the recipe trains on all outer-training bench cells, which is the same training set as the β_frozen record. The results are s2_all 0.178532, noise_var 0.357063, the inner-selected noise factor 16.0, and minimum eigenvalue of C_all 5.6e-6.
      - **Sensitivities.** The literal C3-only fit (229 cells: s2_all 0.0501, noise factor 4.0) keeps the 4-read set of only 89 of 229 C3 cells (3-read set 98, full sequence 55). The BP record gives identical read sets for 229 of 229 cells.
      - **Frozen Gd-read rate** (part of the C-GD claim [I2 confirm-3]): Gd is in the external 4-read set of 95 of the 218 C3 cells with Gd measured, and in the 3-read set of 60 of 218.
      - B1 on new data uses β_frozen. For context, the external read sets equal the BPG dev-fold read sets for 73–114 of 229 C3 cells per seed at 4 reads.

**Frozen keys (D6; DATA_CONTRACT).** The laboratory key `results/stage0/lab_key.csv` is **PROVISIONAL**, because U6 is open.
- Common-surname links are widened to P4's own list; the liu / zhang / zhou / xu-only version is a sensitivity.
- Family–Given order is repaired. Links are resolved through a single principal investigator. The institution-link rule is applied consistently: CSM-Shafer, Osaka-Shinohara and Kanazawa-Imura are P.
- Conservative and liberal counts: all registry keys 45 / 85; V1-cohort DOIs 24 / 44; C3 DOIs 17 / 31; E-CT PRIMARY publications 7 / 11 laboratories. The four-named-only sensitivity gives 33 / 85, 20 / 44 and 16 / 31.
- The mechanism-class map (`mechanism_class_map.csv`) is applied blind to values.

## 8. Closure ledger (brief §4.4)

The closure ledger lives in **DECISION_LOG.md**, one line per route and no slot. It has two parts:
- the E-A model routes closed on this corpus, each with its evidence, grade, power caveat and source;
- the other closure lines.

**Meaning of "closed" and of the grades.** "Closed" means "cannot beat B1 by ≥ 0.01 R² under BP / BPG on this corpus", not "chemistry absent" [EB §3]. The grades are:
- **FIRM**: null when cross-fitted, oracle ceiling below the margin, or replicated;
- **MODERATE**: null, but the test lacked power;
- **WEAK**: not detected, and control power not shown.

**Power caveats for every MODERATE and WEAK line** (METRICS):
- Kish 2.24;
- the percentile cluster bootstrap rejects at 0.07–0.08 for a nominal 0.05;
- contrast-specific dev MDEs;
- BP is not independent of BPG;
- the synthetic studies used 6–8 replicates (exact 0/n upper bounds 0.37–0.46) and were miscalibrated (S05 and S11 gained about +0.02 over B1 in NULL worlds) [EB §3].

**Open but unscheduled lines** are listed with the ledger: H-COND on E-A (it would need §9); chemistry priors at ≤ 2 reads; Gd-aware or learned adaptive read policies; leave-pair-type-out (E-E).

## 9. Pre-slot decidability gate (brief §4.5; sealed now; retire-only)

No new model slot on E-A may be scored unless **all** of the following hold before any outer score of that slot. If the gate fails, the slot is not run, and E-A stays closed on this corpus.

1. **Oracle ceiling ≥ margin.** The slot's own feature set, fitted with test labels and labelled ORACLE, reduces pooled pair-SSE against B1 by ≥ 2 % on BPG [IR M35 item 7].
2. **Power ≥ 50 % at its plausible effect,** from a synthetic study with ≥ 50 replicates per world, run through the sealed rule. An observed 0/50 then has an exact upper bound of 0.07.
   - The nuisance is calibrated to the real B1 residual: adjacent pointwise ρ 0.69, cell-offset share 36 %, per-cluster heterogeneity and light-end variance.
   - A NULL world must be included in which the candidate does not gain. Round 4's NULL worlds gave S05 and S11 about +0.02 [E5 §5.3; EB §3].
3. **Label-free support (retire-only).** The slot is demoted when the deciding fold has fewer than 10 training cells sharing the test rows' donor class or chemotype [IR M35 item 7]. The slot states its training-cell need. A model that needs more than about 60 cells cannot be judged on the deciding fold [E5 §5.5].
4. **Reopen bar for per-metal priors** (lab-3). A pre-declared fold-legal variant must beat B1 by about 4–6 % relative SSE (≈ 0.02–0.03 R², the size 13 clusters can detect). It also needs a positive cluster-jackknife lower bound with and without C5-BTBP [I2 lab-3].

**If a slot passes.**
- It is sealed by amendment and scored under M35 as a DIAGNOSTIC (M34's kill fired).
- Its comparators are B1 and the B1 + edge reference variant: a fold-fitted damping d with inner CV, plus a noise-matched linear-tail control [IR M28; EB §4 H-EDGE].

**No such slot is registered for round 5.** Every proposal the gate sees is recorded in DECISION_LOG; none is expected.

## 10. Stage 1: order, registry and reproduction (brief §1.2, §6.0, §6.10)

**Order per item (sealed).** Stage-0 seal → item code seal → reproduction run of brief §6.0, labelled **TECHNICAL SMOKE** → arms.
- **The item code seal coincides with the Stage-0 seal.** Every item's code, frozen arm list, report rules and specification (`results/<ITEM>/ITEM_SPEC.md`) were written before it, so one manifest seals Stage 0 and every item together.
- **Before the seal**, only `--smoke` (PRE-SEAL ENGINEERING CHECK; every metric suppressed; §0) may run on real data.
- **Departure from brief §1.2 (declared).** The `--smoke` runs of items 1.1, 1.2, 1.3, 1.5 and 1.8 (and the 1.0 cache build) ran before the item-code seal, and their code and specifications were edited afterwards; they exposed only the SMOKE_COUNTS fields (row shapes, group counts, `b1_maxabs`, `y_maxabs` with surrogate y, read-set mismatches, NaN shares, positive-control counts). Items 1.4, 1.6 and 1.7 have no smoke output. After the pre-seal review every item's `--smoke` ran again on the final code (RUN_RECORD).
- **After the seal**, each item runs `--repro` first, then the arms. Every scoring run also executes its reproduction target first and stops on a mismatch (SystemExit, a STOP line in RUN_RECORD, no scored file), and runs its leakage checks before any scored write; a failed check stops the item the same way.
  - A mismatch fixed by a code change is resealed by an append-only amendment that lists what was seen. A scoring run repeats its reproduction target first where its specification says so.
- **The seal gate.** Every scoring script refuses to run unless `scripts/r5_seal.py --verify` passes (`r5.sealcheck.require_seal`). It records the 12-character seal tag next to every number it writes, a START and an END line in RUN_RECORD.md, and ends with the post-run check of §12 (`r5.sealcheck.post_run`). Options that shrink a leakage check (`--intervention-seeds`, `--ect-check-seeds`, `--iv-seeds`, `--poison-seeds`, `--no-intervention`, `--no-lopo`) are allowed in `--smoke` only.
- **Commands.** Each `ITEM_SPEC.md` gives the exact command lines, run from `D:\ml_separator_gh`, one process at a time.

**Common conventions (brief §6.0).**
- Rows are the round-4 REF parquets (BPG primary, BP sensitivity) over the 5 dev seeds, with R² pooled within seed and then averaged.
- Every arm of an item is scored on identical rows.
- **Every item reports:** C5-BTBP in / out (pub_6a0ef8528c) and the single-cell influence of c950dc2982e96b83 for every number (brief §5.2), BPG and BP, per-seed values, the unit table, leave-one-publication-out and the top-3 unit share (METRICS).
- In-corpus results carry "EXPLORATORY (seen)". Thresholds that sit below already-seen values make consistency checks, never tests.

**Registry** (`REGISTRY.csv`; every arm and annex row PLANNED or EXCLUDED; nothing optional remains after the seal):

| id | item (`results/<ITEM>/`) | estimand | reproduction target (TECHNICAL SMOKE; a mismatch stops the item) | deciding design | report rule (outcome label) |
|---|---|---|---|---|---|
| 1.0 | EA_CHECK: M35 positive control and six-design check | E-A, DIAGNOSTIC | B1 rebuilt from round 4's fold β, and B0 and HIST_D7, equal REF within 8.9e-16 on all 18,020 rows, with identical keys, y and read sets | BPG | records the seen positive-control outcome; no E-A slot (§5) |
| 1.1 | ECT: E-CT core, frozen `r5/ect.py` | E-CT | B1 identity (≤ 8.9e-16); V1_GE1 (149 cells, 1,111 rows per seed): SIB 0.5734, SH (K = 2) 0.5946, B1 0.4489; PRIMARY_NOBIT (99 / 12 / 14 / 690): B1 0.5309, SH 0.7082, SIB 0.6987, OFF 0.6963 (tolerance 5e-5, BPG) [P1 §2; PC P1] | BPG | brief §6.1 consistency checks; mechanism contrast SH − OFF_XEXT and SH − PB_PUB |
| 1.2 | ECT_M03: sibling-weighting factorial | E-CT annex | B1 identity (≤ 8.9e-16); shared arms equal `r5.ect` exactly; the kernel's uniform limit equals SIB / SH; PRIMARY_NOBIT B1 0.5309, SH 0.7082, SIB 0.6987 | BPG | deploy the uniform shrunk mean unless the nested rule passes: +0.01 pooled R² (seed mean) on PRIMARY_NO_ORNL, ≥ 7 of 11 publications improve, uniform not chosen in most folds (CONSISTENCY CHECK) |
| 1.3 | ELAB: campaign calibration (lab-1) | E-LAB | B1 identity; POPX 92 / 16 / 787, POPX0 60 / 13 / 534; MX_LM10 on the SRC pool equals `ideas2/lab_out/lab_rows2.parquet` within 1e-12 on 7,870 rows; BPG B1 0.444, MX_LM10 0.494 (3 decimals) [I2 lab-1] | BPG | "best of about 25 arms on seen rows; expect regression"; external leg descriptive and NOT RUN in Stage 1 |
| 1.4 | WINDOW: M09 level window | window axis | B1 identity; EB Appendix 2: BPG coverage per seed in [0.743, 0.754], inside R² 0.540 (BP 0.538), tolerance 0.0005 | BPG | row flag vs cell flag; tail flag passes if the \|res\| > 0.5 ratio is ≥ 3 with a bootstrap lower bound > 1, on all rows and without C5-BTBP; bulk precision needs ≥ 15 % narrowing |
| 1.5 | EDEC: M14 direction calls, E-REC own-prior row, E-RANK | E-DEC; E-RANK; E-REC | B1 identity; D7 at 4 reads equals REF HIST_D7 within 1e-12; y identical between E-CT and E-A rows | BPG | descriptive; never an R² gain |
| 1.6 | ED: M18 + M17 ladder, policy envelope, recipe-2 | E-D | B1 identity at 4 reads; P3 5-read DOPTQ-nested B1: C3 min(4, k_max) 0.4968 / 0.4988, C4 0.4798 / 0.4821 (5e-5); 5-read B0 equals round 3's R3-B0 [PC P3] | BPG | descriptive; pre-declared Pareto question; saturation rule (CONSISTENCY CHECK) |
| 1.7 | EW: M13 wider pairs | E-W | B1 identity on the STRICT rows | BPG | Nd/Sm B1 − LINE retired if ≤ +0.05, if its lower bound is ≤ 0, or if fewer than 2 clusters qualify |
| 1.8 | EKX: M07 sc009 Nd dip | E-KX | B1 identity (≤ 8.9e-16); construction identities (y, B0, B0 + β Δφ_Gd = B1, B1 + u Δφ_Nd = `predict_levels`) | E-KX-LOPO (B1 base = the BP OOF prediction, brief §6.8; never called BP); the BPG-base variant is a sensitivity | narrowed to "universal Nd term" if chemotype-Nd − B1-NdU ≤ 0 with the top 3 publications removed |
| 1.9 | `PRACTICAL_GUIDANCE.md` | — | — | — | every line labelled ESTABLISHED, EXPLORATORY or HYPOTHESIS, with its cost scenario (brief §6.9) |

**Run order.**
- The items run in registry order, one process at a time.
- Item 1.1 precedes 1.2 and 1.5, which reuse `r5/ect.py` and, for E-REC, the E-CT rows under the current seal.
- After every Stage-1 run, `round4_20261022/scripts/r4_seal.py --verify` must still report 5bacfe4c, and the round-3 LF digests must still match (§12).

**Annex and appendix rows (D7).** Every annex or appendix row that an item's code computes is **PLANNED**; none is EXCLUDED. This covers:
- E-CT's shift-1, recipe-1 E-REC, M02, M04 and M06 rows, the V2-5 floor sensitivity and the laboratory annotation;
- E-DEC's E-REC row and E-RANK table;
- E-D's recipe-2 row and saturation strata;
- item 1.0's M08 appendix.

The arms an item specification names as EXCLUDED at its freeze are listed there, so their absence is not post hoc: for example, E-LAB's MG_LM10_all, and M03's 2- and 3-read factorials and IR M03's ρ level.

**Item design choices (D8).** The choices written in each `ITEM_SPEC.md` ("Decisions the brief leaves open, frozen here", "Interpretations fixed at this freeze") are adopted as frozen, unless the pre-seal review found one contrary to the brief. Examples:
- E-CT's and M03's exact sign-flip enumeration up to 22 units;
- M03's exponential kernels and grids, and its paired one-SE rule;
- E-W's 33 cut-pair types;
- E-KX's n = 12 donor publications for the external shrinkage (w = 12/17, u_ext = −0.077647).

**Seeds.** Every random procedure is seeded with `default_rng(20261023 + crc32(key))` (`r5.paths.seed_for`), using the keys recorded in `REGISTRY.csv` and in each specification. The exceptions keep round 4's seeds: the shared cluster-bootstrap matrix `gen20k.inference._draw_counts(13, 2000, 8675309)`, and fit-poisoning seeds `split_seed × 10 + fold`.

**Stopping (brief §1.3).**
- Stage 1 is finite. An item that cannot run (missing input, failed gate) is marked NOT RUN with the reason.
- No arm is added after results are seen. If one is added, it is labelled post hoc in an amendment.
- The round stops when four things hold: Stage 0 is sealed; every Stage-1 item is DONE or NOT RUN; every request is recorded as answered or open; and the outputs of brief §10 exist.

## 11. Amendments and reseals

- **Every change after the seal** goes into `AMENDMENTS.md` (append-only). An entry gives the date (UTC), the sealed artifact that changed and why, every result already seen at that time (with per-item and per-unit tables where they exist), and the new seal.
- **A reseal** runs `scripts/r5_seal.py --seal --reason "..."`. The script refuses unless an AMENDMENTS.md entry names the previous round-5 seal prefix, and it archives the old manifest and seal.
- **A reproduction mismatch fixed by a code change** is resealed in this way before the item's arms run. The amendment states what the mismatch run showed.
- **Provisional Stage-0 inputs** (the laboratory key, U6; the planning amplitudes, U5 item 4; SAFE, U2) are updated only by an append-only amendment made **before any new value is read**. The supply audit is then rerun.
- **An arm, stratum or rule added after any result is seen** is labelled post hoc. It never changes an item's sealed report rule or label.
- **An error-triggered rerun** needs a documented code defect, unchanged scientific choices and full disclosure. It does not create a fresh holdout [R4-BRIEF §9].
- Margins, thresholds and labels are never relaxed after a result.

## 12. Leakage and controls (brief §8)

Every quantity that is fitted or selected in round 5 passes the round-4 chain on every relevant fold and design [R4-PROTOCOL §8].
- The checks are re-implemented in `r5` and write only under `results/leakage/`. `r4_poison.py` and `r4_cache_poison.py` are never run.
- After every Stage-1 run, `r4_seal.py --verify` must report 5bacfe4c, and the round-3 LF digests must match. `r5.sealcheck.post_run` checks both (with the round-5 seal) after every non-smoke run, also after a stop, writes `results/leakage/postrun_<ITEM>.json` and the END line of RUN_RECORD.md, and makes the run's exit status non-zero when a check fails.

**The chain.**
1. **Isolation.** `g21.evaluate.assert_test_isolation` on every BP / BPG fold. Zero overlap of cells, chemotypes and publications, and of copy groups under BPG. Inner blocks lie inside the outer training fold. Item 1.0 checks A, B, BR and BQ design-aware: zero overlap of cells and extractants under A, and of chemotypes too under B / BR / BQ.
2. **Fit poisoning.** Every non-training label field is randomised: log D, the gen13 coefficients, `logD__*`, `repsd__*` and `replicate_sd_median`. Every fitted quantity must then be bit-identical. This applies to the k-specific β refits of E-D (all 50 fold-fits), M03's nested selection, M09's operating points and nested choices, and M14's calibration thresholds. Each carries a positive control.
3. **Cache poisoning.** Every new cache is rebuilt from poisoned non-training labels and compared field by field. Cache keys include the data version and content, the code, the split, the support contract, the feature state and the hyperparameters. The only new caches are item 1.0's 100 fold records (passed 100 / 100 before the seal) and the Stage-0 records under `cache/frozen/` and `cache/ext/`, which are training-side by construction: every bench cell trains.
4. **Prediction intervention.** Every unread label of the held-out or target cell is replaced and everything is rebuilt; predictions and read sequences must be bit-identical. Positive controls must be caught:
   - for E-CT and its annexes, the target itself, and a bit-identical copy, in its own sibling pool;
   - for the E-A rows, a `block["Y"]` peek.
5. **Support consistency.** Exact supports and fixed denominators. No scored pair with both ends read. Correct sign and adjacency. Nested read sets in every ladder, and `legal_metals` for every added read.

**Per item** (details and positive controls in each `ITEM_SPEC.md`; reports `results/leakage/<ITEM>_<design>.json`):

| item | isolation | fit poisoning | cache poisoning | intervention + positive controls | support |
|---|---|---|---|---|---|
| 1.0 EA_CHECK | all six designs, design-aware | n/a: training-free arms; B1 is the stored round-4 fit | 100 / 100 (pre-seal) | A / B / BR / BQ folds; `block["Y"]` peek | four reads; exact reads; \|ΔZ\| = 1 |
| 1.1 ECT | round 4's (REF valid) | n/a: no fit (stored fold β, K fixed) | n/a: no new cache | two scramble forms; target in own pool, bit-identical copy, guard off | nested, legal, no both-read pair |
| 1.2 ECT_M03 | round 4's | nested selection and POOLPAIR slopes, each with a leaky positive control | n/a | target in own pool, bit-identical copy, guard off, read distances from the full series | as 1.1 |
| 1.3 ELAB | n/a by design: the campaign's other series are allowed inputs | LOPO-chosen K (`gk_poison_check`: production `gk_lopo` rerun on poisoned rows, leaky positive control) | n/a | target in own pool, bit-identical copy, guard off | fixed denominators |
| 1.4 WINDOW | every fold | operating points and nested choices on every fold, with a test-cells-added positive control | n/a | `block["Y"]` peek | read levels equal reads |
| 1.5 EDEC | round 4's | every calibrated threshold, Platt fit, interval nominal and bound quantile, per held-out cluster, with a leaky-calibration positive control; scope: calibration labels only | n/a | E-REC arms through 1.1 (reused only when item 1.1's intervention and reproduction passed under the current seal) | hull flag from read sets only |
| 1.6 ED | every fold | k-specific β refits, k = 1–6, all 50 fold-fits (`scripts/s1_ed_poison.py`; the tables are written only after it passed under the current seal); the recipe-2 LOCO static selection (`r5.ed.poison_loco`, leaky positive control) | n/a | every read set; `block["Y"]` peek through the level function | nested, legal ladders; identical row keys |
| 1.7 EW | every fold | n/a (no fit) | n/a | `block["Y"]` peek through the level function | ΔZ = 1 STRICT, ≥ 2 WIDE |
| 1.8 EKX | donor guard (other publication and other copy group) | n/a (donor means, K = 5 fixed) | n/a | scramble of every label in the target's publication and copy group; own-pool and leaky-pool positive controls | construction identities |

**Disclosed limitations.**
- **E-DEC:** a calibration row's prediction comes from its own outer fold's model, whose training may include cells of the held-out cluster. The calibration labels are only other clusters' labels. The fit-poisoning check's scope is therefore calibration labels only; it is stated in `results/leakage/EDEC_<design>.json` and the per-item leakage line.
- **E-KX:** KX_FROZEN (−0.11, computed with every sc009 publication) is not fold-independent, and is labelled DIAGNOSTIC.

**Round-5-specific rules.**
- **E-CT information contract.**
  - Allowed: the target's legal reads; the siblings' full measured series; the fold's B1 β, fitted without the target's publication; siblings' OOF residuals for OFF (leave-target-out).
  - Forbidden: the target's unread metals; a duplicate of the target in any pool (the bit-identity rule, DATA_CONTRACT); any quantity tuned on E-CT test rows.
  - K = 1.78 comes from a roughness correlation measured on these data, which is disclosed as seen [P1 §15].
- **Placebos at matched weights and counts** [I2 rule 14]: a placebo with w ≈ 1 loses for reasons unrelated to the claim.
- **Label-blind supply.** Metadata-only scripts assert that no value column is loaded. Web summarisers return metadata only, and any qualitative direction that leaks back is disclosed (P4 received four: P4-063, P4-018, P4-022, P4-051) [P4 §8].
- **Oracles [O]** bound headroom only.
- **Claim-specific controls:** E-CT cross-publication placebos and OFF_XEXT / PB_PUB; M03's selection-aware nulls; E-LAB's E_RM and E_PCM; M09's cell-matched abstention; M07's random-chemotype permutation; C-GD and shift-3 sign-reversal and permutation controls.
- **Discriminating power of confirmation tests.**
  - Size under arm-swap, symmetric and two-stage (chemotype → laboratory) nulls, with ≥ 2,000 replicates per cell and Clopper–Pearson bounds (§7, item 2).
  - Power in worlds with ≥ 50 replicates (10,000 used), with the Monte Carlo uncertainty reported: two-sided 95 % Clopper–Pearson bounds for every reported rate in `power_worlds.csv` (SE ≤ 0.005) [I2 confirm-1; E5 §5.3].
- **Confirmation hygiene.** Values are sealed by a separate agent, the evaluator shows only legal reads, labels are revealed after the predictions are sealed, and the confirmation runs once.

## 13. Stage 2 and what happens if nothing arrives (brief §7)

**The requests.**
- `DATA_REQUEST.md` is DR, copied in Stage 0. Requests U1-a … U1-i, U2, U3, U4, U5, U6 and S1 were shown to the user once (RUN_RECORD 04:42:20 UTC). All are OPEN at this seal.
- Request status is kept in `RESUME.md` and below the marker of DECISION_LOG, never in the sealed `DATA_REQUEST.md`. `RESUME.md` (request table, PROVISIONAL inputs, resume commands, NOT RUN list) is updated together with DECISION_LOG's append-only section, below its own marker.
- **Stage-2 rules on yes (sealed by reference).** If a request is granted, the brief's on-yes rules apply as written, with `DATA_REQUEST.md` "How values will be handled" and DATA_CONTRACT §4 item 6:
  - U1 per brief §7.1 items 1–6: V3 rules sealed before any recovered value is read; the label-free census order (SHA-256 `0060ef804f78c13c…`); blind extraction by a separate agent; the error classes of brief §7.1 item 2 (gross > 0.3, SOURCE-RESOLUTION, MODEL-DERIVED / GRID, NON-EQUILIBRIUM, MISATTRIBUTED, COMPONENT, UNLOCATABLE); the M25 thresholds (anchored at ≥ 50 %, provenance lever retired below 2 %, unverifiable stratum below 40 %); the M24 reopen rule (+0.005 R² in ≥ 4 of 5 seeds); the G-V3 gate.
  - U2 per brief §7.2; U3 per brief §7.3.
  - U4 per brief §7.4: the M11 decision rules; M12 laboratory signature if V_x/V_w ≥ 5 with lower bound ≥ 2, none if the upper bound is ≤ 2.
  - U5 per brief §7.5: the search is frozen (databases, query strings, date window, inclusion rules) and sealed by amendment before it runs; DOI-level counts only; no values read; a metadata-only look at the 5 Eu- or Nd-only archive DOIs [P4 §6.1]; the supply audit reruns (`scripts/s0_supply.py --out results/supply_audit/<UTC> --request U5`, which never touches `results/stage0/`).
- No request is paraphrased as granted. The round never blocks on a reply, and an unverified value is never substituted for a missing document.

**Outcome at this seal.** The supply-audit kill fires (§7, item 7), so round 5 is descriptive. The report gives:
- E-A closed on this corpus, with B1 as the reference;
- E-CT exploratory (seen), with its frozen code (`r5/ect.py`, called with the external read object and β_frozen) ready for new data;
- the confirmation package sealed and waiting;
- the open requests.

## 14. Git and publication

Nothing is committed or pushed without the user's separate request. The seal is local; it is not public preregistration.
