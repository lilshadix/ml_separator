# gen20_1 — S12 audit record (2026-10-17)

Two independent adversarial auditors and one verifier, after S11. Auditor scripts were kept outside the repository
(session scratchpad); this file records what they checked and found. Verdict: **no blocker; no reported number, class or
verdict changes.** Documentation findings are resolved in `ADDENDA.md` (Addendum 1) and `README.md`.

## Auditor 1 — independent recomputation (own code; no `gen201` import)

- **Integrity.** 354 DONE markers (discovery 300 = 6 designs × 2 variants × 25 units; confirmation 50; smoke_S8 4), every
  one carrying fitting-code digest `7e869f0a3d95e82edd1479d4d43d2f973984e84947bb30ec8207b54b3197b17b`, which equals the code
  on disk; every recorded file sha256 matches its file.
- **Coverage.** Every row-manifest key of COH-D-C3 (2,425), COH-SF-C3 (1,802) and their NODMD variants appears exactly once
  per split seed in every design and variant; 1,786,314 prediction rows across 1,416 outer and inner parquets, 0 non-finite.
- **Point metrics.** 2,688 comparisons with tab03, tab04, tab05, tab06 and tab08 (discovery and confirmation) at
  tolerance 1e-9: 0 failures. Construction identities hold exactly (B = A[a] − A[b]; C pairs = centred differences;
  y_SF = y_a − y_b; lighter metal first; 12 strict pair types).
- **Inference.** Own chemotype and publication bootstraps (seeds 8675309 / 20260930, 10,000 draws), BCa with LOCO
  jackknife, CR2 with Bell–McCaffrey dof, Rademacher wild-cluster-t (9,999 reps) and the three MDEs: 952 comparisons, 0
  failures; NODMD verdicts, NC1, COH-NOSC009 intervals and MET-17 intervals also match.
- **Rules.** RULE-C (a)–(k), the §8.1 verdict order and RULE-H4 reproduce H1 INCONCLUSIVE (UNDERPOWERED), H2
  INCONCLUSIVE, H3 INCONCLUSIVE (confirmation: UNDERPOWERED), H4 EQUIVALENT in both seed sets; RULE-T reproduces all 76
  classes per seed set, the O7 precedence and the level-attributable clause; BH over m = 83 gives 3 rejections.
- **Anchors.** FLAT and G14_ZERO recomputed on COH-SF-C3 reproduce gen20's levels csv rows 6 and 18.

## Auditor 2 — protocol conformance and leakage

- **Seal.** `require_sealed(PRE_REGISTRATION.md)` passes; body digest `f34a1b97…` equals the footer and `seal_g201.json`;
  the plan's LF digest `453654a4…` matches; the 5 S3 reference files and 28 row manifests match their recorded digests;
  the confirmation seeds re-derive by the sealed rule (commitment `a2bb89f7…`).
- **Frozen trees.** `git status` shows only `?? generations/gen20_1/`; `git diff HEAD` is empty; no file outside gen20_1 is
  newer than 2026-10-16 00:00.
- **Masked-label invariance at run scale.** Discovery units BP s104729 f0 (76 training / 418 test cells) and BP s196613 f3
  (422 / 34) refitted with the real 400/200-tree configuration after setting every outer-test cell's `bench.Y` row,
  `bench.coef` row and the 44 `logD__`/`nrep__`/`repsd__`/replicate frame columns to NaN: all 44 arm columns bitwise equal
  to the saved predictions; selection records identical.
- **Thread and process independence.** Confirmation unit BP s915723 f3 refitted at `LOKY_MAX_CPU_COUNT = 2` (the run used
  6): bitwise equal. Smoke tables (one process, 2 threads) equal discovery tables (two processes, 4 threads) bitwise.
- **Selection.** All 350 units' selection records rebuild from inner-only rows within 1e-12 (scores, argmins,
  BL_D_INNER, BL_SF_INNER); the one skipped inner fold (BP s155921 f2 inner 3, 18 cells) matches the S3 count.
- **Gates.** S7 anchors PASS (seed-mean deltas ≤ 4.1e-9; per-seed ≤ 4.7e-6 < 6e-6; run_fold cross-check 0.0 over 9,010
  rows); S8 smoke flags pass (projection 7.74 h < 24 h); fold gates a–e pass in the discovery parent, S7 and every design
  child; all code files were last modified before discovery started and are unchanged since.
- **Report.** The §1 "easier" sentence and PAT-5 are verbatim; no class is REACHED, so no attainment is claimed; H4
  EQUIVALENT holds on both seed sets; the confirmation is labelled a partition-robustness check.

## Verifier

16 findings: 12 confirmed, 4 modified, 0 rejected; 1 major (README described the pre-seal draft) and 15 minor
(undocumented operational deviations, report wording, provenance of the scoring code, no S4 git commit). All are
recorded or resolved in `ADDENDA.md` Addendum 1 and `README.md`; the generated report and every result table are unchanged.
