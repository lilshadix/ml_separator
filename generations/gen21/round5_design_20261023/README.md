# Gen21 round 5: design folder

Created 2026-10-23, indexed 2026-10-24. This folder holds the evidence, idea rounds, probes, requests and the executable
brief behind the round-5 programme for strict-adjacent lanthanide log SF at k = 3.

> **Status: exploratory design. Nothing is sealed, nothing is committed.**
> - No seal or manifest exists for round 5. Nothing here has been committed or pushed.
> - Every in-corpus number in this folder was computed on the 229 C3 cells and the 5 dev seeds (104729, 130363, 155921,
>   196613, 262147). Rounds 1–4, the probes and both skeptic rounds have already seen them. They are leads ("seen"),
>   never tests.
> - **Estimands.** E-A is round 4's estimand at four read metals (k = 3). Separate estimands and data versions: E-D
>   (read-budget ladder), E-CT, E-LAB, E-W, E-DEC (with the E-REC own-prior row), E-KX, E-RANK, E-S (breadth
>   diagnostic), E-CENS (sensitivity line), the window axis, V3, and the new-laboratory claims C-ECT, C-GD and Gd-dip
>   transport. None substitutes for another.
> - E-CT, E-LAB, E-W, E-DEC, E-KX, E-RANK and V3 values are never placed next to the E-A reference, B1 = 0.452. E-D's
>   4-read rung is E-A itself; its 5- and 6-read rungs are a different read budget, compared with the k = 3 rung
>   (B1 0.452) only as paired contrasts on identical queries, labelled as a different read budget, never as E-A values.
>   V3 values on other row sets are never placed next to 0.452.
> - Where `probes/PROBE_CHECKS.md` corrects a probe, the correction takes precedence.

---

## The round-5 plan in ten lines

1. **Target unchanged.** E-A is BPG pooled strict-adjacent R² at k = 3 (C3, DOPT_Q, 5 dev seeds). The reference B1 (radius interpolation plus one fold-fitted Gd amplitude) scores 0.452, and E-A is closed to new model slots on this corpus.
2. **Stage 0, decision (hours, before any round-5 score).** Seal M35: pooled relative pair-SSE reduction vs B1 ≥ 2 % (about +0.011 R²), balanced-MAE non-inferiority ≥ −0.001, BPG decides, BP is a labelled sensitivity; per-cluster, big-fold vs rest and C5-BTBP in/out tables are mandatory.
3. **Stage 0, inference.** Seal M34: exact sign-flip on pooled ΔSSE and the exact sign test; drop the jackknife-t with df = G − 1 (anti-conservative); equal-weight log ratios are breadth diagnostics only; unit laboratory × chemotype; fixed sequence, no Holm at small G; G floors and power stated in worlds.
4. **Stage 0, package and ledger.** Seal confirm-1's frozen confirmation package (inconclusive zones; non-rejection never switches the deployed model) and the closure ledger of E-A model routes, lab-3 included.
5. **Stage 1, E-CT (software; every number "seen").** Frozen shrunk own-series template SH (K = 1.78) after a duplicate scan; C5-BTBP in/out; reads 0/2/3/4; comparators on identical rows incl. B1, COPY, OFF and OFF_XEXT / PB_PUB (SH ≡ OFF on 73 % of rows); placebos; annexes E-LAB and M03.
6. **Stage 1, descriptive deliverables, never decision slots.** M09 tail-risk flag, M14 direction calls, M18 metal ladder with M17's policy envelope (about 0.009), M13 wider pairs, M07 sc009 Nd dip frozen at −0.11, and practical guidance for chemists.
7. **The k = 4 rung (E-D, exploratory; 5 reads, a different read budget).** At 5 reads the like-for-like C4 cohort (178 cells) scores 0.480 BPG / 0.482 BP (0/5 seeds). The C3 min(4, k_max) population row, in which 15.9 % of rows keep 4 reads, scores 0.497 (3/5 seeds). Without C5-BTBP: C4 0.517, C3 0.531; that removal is outcome-dependent (the publication was chosen for its SSE share) and also removes about 18 % of SST, so it is a provenance sensitivity, not a budget effect (PC P3).
8. **Stage 2, data-gated.** U1 source recovery → V3 sensitivities; U2 SAFE gap-ID export → supply gate M33; U3 Molecules 29:3548 SI → a dry run of the sealed E-CT pipeline (its laboratory, the MSU Nenajdenko group, is already in the corpus, P4-001, so it does not count toward C-ECT's 7 new laboratories); U4 laboratory collaborator → M11 / M12; U5 literature web reading; U6 laboratory-key review; S1 V2-3 cohort-key sign-off. All of U1–U6 and S1 are in `DATA_REQUESTS.md`, ranked by value per minute.
9. **Confirmation is sized to the supply.** 5–6 verified and about 7.0–11.5 expected new laboratories after the check's corrections (13.5 under P4's own tier rule), nearly all on chemotypes the corpus lacks. C-ECT needs ≥ 7 laboratories. The Gd-dip transport test needs ≥ 10 laboratory × chemotype units on the merged SAFE + literature ledger; C-GD, its pre-specified stratum, needs ≥ 6 laboratories from ≥ 3 families. Power is stated in worlds.
10. **Stopping.** No new E-A slot unless it passes the retire-only pre-slot decidability gate (oracle ceiling ≥ 2 % pooled relative SSE vs B1; power ≥ 50 % at its plausible effect; demote if the deciding fold has < 10 training cells sharing the test rows' donor class or chemotype; CLAUDE_CODE_PROMPT §4.5). 0.50 at four read metals (E-A) is not reached by any tested or proposed model; gains below about 0.01 R² are undetectable here (EB §3). At 5 reads (E-D, a different budget): C4 0.480; C3 min(4, k_max) mixture 0.497 (PC P3). On E-A itself only new or cleaner data can move the level, reported as V3 or U4 sensitivities.

Sources for these lines: `RESEARCH_IDEAS.md` §§1, 4, 6, 7; `CLAUDE_CODE_PROMPT.md` §4.5, §7.5; `DATA_REQUESTS.md`;
`ideas/IDEAS_RANKED.md` (M01, M33, M34, M35, closure ledger); `ideas2/IDEAS2_RANKED.md` (confirm-1, confirm-2,
confirm-4, lab-3); `probes/PROBE_CHECKS.md` (P1–P5); `probes/P4_candidates.csv` (P4-001).

---

## Reading order

For the implementing agent: `CLAUDE_CODE_PROMPT.md` first; it cites everything else.

1. **This README.**
2. **`RESEARCH_IDEAS.md`**: the round-5 programme in one document (evidence, closures, ranked ideas, guidance for chemists,
   data acquisition, evaluation and stopping).
3. **`CLAUDE_CODE_PROMPT.md`**: the executable brief (Stage 0 seals §4, Stage-1 registry §6.10, Stage-2 requests §7,
   leakage §8, outputs §10).
4. **`DATA_REQUESTS.md`**: what the programme needs from the user, ranked by value per minute of effort.
5. **`evidence/EVIDENCE_BASE.md`**, then E1–E6 as needed.
6. **`ideas/IDEAS_RANKED.md`** (round 1, M01–M39); `ideas/critiques.json` for the full verdicts.
7. **`probes/P1`–`P5`**, each followed by its section in **`probes/PROBE_CHECKS.md`**.
8. **`ideas2/IDEAS2_RANKED.md`** (round 2), then the theme notes for detail.

---

## Index

Files present when this index was written. Paths are relative to this folder.

### Top level

| path | what it is |
|---|---|
| `README.md` | this index, status and the ten-line plan |
| `RESEARCH_IDEAS.md` | the round-5 programme: merges the evidence base, both idea rounds, the probes and their checks, and round 4's record |
| `CLAUDE_CODE_PROMPT.md` | the executable round-5 brief for the implementing agent (Stage 0 seals, Stage-1 registry §6.10, Stage-2 requests §7, leakage §8, outputs §10) |
| `DATA_REQUESTS.md` | user-facing requests: U1 source recovery (U1-a … U1-i), U2 SAFE gap-ID export, U3 Molecules SI, U4 laboratory collaborator, U5 literature web reading, U6 laboratory-key review, S1 V2-3 cohort-key sign-off, H1 housekeeping |
| `REVIEW_LOG.md` | the review findings applied to the four top-level documents, and what changed |

### `evidence/`: what is known after round 4

| path | what it is |
|---|---|
| `EVIDENCE_BASE.md` | merge of E1–E6 with four reconciliation checks; established results, closed routes with firmness grades, open hypotheses ranked, contradictions and gaps |
| `E1_history.md` | what has been tried from gen13 to gen21 round 4, and what is still open; new diagnostics on the round-4 OOF rows |
| `E2_error_anatomy.md` | where B1's error lives: pair types, tails, clusters, noise floor |
| `E3_data_inventory.md` | the data lever: what data exist, why relaxed cohorts add no clusters, pending data-quality items, noise-ceiling arithmetic, the value of a 5th read, the Gd dip in raw data |
| `E4_literature.md` | literature on the physics, chemistry and ML of adjacent-Ln selectivity, with small checks |
| `E5_round4_design_review.md` | the round-4 design reviewed as a design: gates, fold geometry, margin currencies, what round 5 should change |
| `E6_measurement_lever.md` | measurement and decision levers: k ladder, read placement, sibling template, decision endpoints |

### `ideas/`: round-1 ideas

| path | what it is |
|---|---|
| `IDEAS_RANKED.md` | M01–M39 after two skeptics and three judges: tiers, corrected designs, kill criteria, consolidated requests U1–U4, closure ledger, merge map |
| `critiques.json` | every idea in full, both verdicts condensed, judge scores, tier, rank and priors |
| `merged_ideas.json` | the 39 merged ideas as proposed |

### `probes/`: five exploratory probes and their skeptic checks

| path | what it is |
|---|---|
| `P1_condition_transfer.md` | E-CT on the round-4 rows (M01): SH vs B1 on identical rows, SH ≡ OFF on aligned reads, read ladder, placebos |
| `P2_power_replay.md` | source-level replay of few-cluster paired tests (M34): size, G for 80 % power by unit; tables in `P2_out/` |
| `P3_k4_rung.md` | the k = 4 rung on identical C3 queries (H-BUDGET, M18, M17) |
| `P4_literature_supply.md` | label-blind literature yield count; `P4_candidates.csv` (92 candidates), `P4_corpus_registry.csv`, `P4_corpus_surnames.csv` |
| `P5_source_recovery.md` | stakes-first source recovery (M25): shares of B1's SSE and gain, scenarios, training-side refit, ranked request list, seeded census order |
| `PROBE_CHECKS.md` | independent skeptic re-runs of P1–P5; **corrections here take precedence** |
| `scripts/` | probe scripts (`P1_ct_*`, `P2_*`, `P3_k4_*`, `P4_*`, `P5_*`) and skeptic re-runs (`*SK_*`, `P4_skeptic_recheck.py`), with `P3_out/` and `P3SK_out/` |

### `ideas2/`: round-2 ideas

| path | what it is |
|---|---|
| `IDEAS2_RANKED.md` | 15 round-2 ideas (confirm-1…4, lab-1…4, recipe-1…4, shift-1…3) as SEAL / ANNEX / LEDGER / RIDER actions; rules 9–16; closure-ledger additions; duplicate and provenance candidates; the Gd reconciliation check |
| `shift.md`, `lab.md`, `recipe.md`, `confirm.md` | the four theme notes behind round 2 |
| `scripts/` | theme and skeptic scripts, including `addendum_gd_reconcile.py` and its log, plus run outputs in `scripts/*_out/` (confirm_out, confirm2_skA_out, confirm4_skA_out, shift_out, shift1_skA_out, shift1_skB_out, shift3_skA_out, shift_skA_out) |
| top-level `*_out/` folders | outputs of the theme runs and skeptic checks (logs, CSV, parquet), named by idea and skeptic |

---

## Corrections to keep in mind (from `probes/PROBE_CHECKS.md`)

- **P1.** A suspected bit-identical cross-acid copy pair inflates SH − B1 by about 0.009 (to +0.168). SH − OFF is
  unchanged. Register SH − OFF_XEXT as the mechanism contrast before any look.
- **P2.** Chemotype-unit G80 for the pooled tests is 30–58, not 58–70. On the pooled decision estimand, M34's kill fires
  at both τ. The jackknife-t size maxima come from synthetic pools.
- **P3.** "Decided by C5-BTBP" holds for C4 only. The like-for-like five-metal number is C4, about 0.48. The detectable
  effect depends on the contrast (about 0.009–0.020).
- **P4.** Verified new laboratories: 6 by P4's strict rule, 5 with the institution rule applied consistently. Expected
  after verification: 7.0–11.5 with the corrections, 13.5 under P4's own tier rule.
- **P5.** The refit without the 69 unlocated cells is a seed-unstable tie (−0.0045, 3 of 5 seeds negative), +0.012 with a
  fixed λ; "no model change can anchor it" is dropped. The document package covers about 26 % of SSE and 43.7 % of the
  gain, not 28 % and 52 %. New route: the JACS Au 2022 SI. The ORNL channel is clipped at log D −4.0.

## Reproduction and process notes

- Probe scripts run from `D:\ml_separator_gh` with `.venv/Scripts/python.exe -B -W ignore …`; each probe report lists its
  commands and outputs (for example P5 §1, P4 §9).
- **Several scripts in this folder write to fixed paths inside it**: `P2_power_replay.py`, `P2_summarise.py`,
  `P2_report_tables.py` (into `probes/P2_out/`); `P3_k4_compute.py`, `P3_k4_blocks.py`, `P3_k4_summarise.py`,
  `P3_patch_density.py` (into `probes/scripts/P3_out/`); `P3SK_recompute.py`, `P3SK_analyse.py` (into
  `probes/scripts/P3SK_out/`); `P4_corpus_registry.py` (overwrites `probes/P4_corpus_registry.csv` and
  `P4_corpus_surnames.csv`); and most `ideas2/scripts/*` scripts (into `ideas2/*_out/`, for example
  `confirm1skepB_strata.py`, `lab3skepB_closure.py`, `lab3_recompute_skA.py`). Never run them in place: copy one into the
  round-5 implementation's `scripts/`, change only its output path, and record the source file's LF SHA-256
  (`CLAUDE_CODE_PROMPT.md` §1.2). `P1_ct_build.py`, `P1_ct_analyze.py` and `P1_ct_diag.py` take `--out`. Print-only
  scripts (`P1SK_*`, `P2SK_*`, `P5SK_*`, `P5_accounting.py`, `P5_refit.py`, `P4_check_candidates.py`,
  `P4_skeptic_recheck.py`, `ideas2/scripts/lab1_recompute_skA.py`, `addendum_gd_reconcile.py`) may run in place with
  `-B`.
- The scripts behind `EVIDENCE_BASE.md` and `E3_data_inventory.md` live in the session scratchpad, outside the repository
  (their appendices give the paths). They cannot be rerun from this folder alone.
- Recorded process slips, as their authors reported them: read-only `git status` calls (E4, P3); PDFs auto-cached by a
  web tool outside the repository and never opened (P4 §8, `ideas2/IDEAS2_RANKED.md` "Process note"; see
  `DATA_REQUESTS.md` H1); scratch files written by some skeptics to the session scratchpad. None of them modified existing
  repository files.
