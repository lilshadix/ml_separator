# Round-5 amendments (append-only)

Each entry: the date (UTC), the sealed artifact that changed and why, every result already seen at the time, and the new seal. A reseal names the seal it supersedes.

## A1 — 2026-10-24 — E-D poisoning compares fits through a round-trip CSV parse (supersedes seal 1f0cae0064a6)

**Artifact changed.** `scripts/s1_ed_poison.py`, one line: the main-run betas are read from `results/ED/fits_<design>.csv`
with `pd.read_csv(p, float_precision="round_trip")` instead of the default C parser.

**Why.** Item 1.6's frozen order is full run -> `s1_ed_poison.py` -> `s1_ed.py --from-rows`, and the from-rows step refuses
unless the poisoning record has `all_identical` and `matches_main_run` both true.  The poisoning run gave
`all_identical: true` on all 50 fold-fits x 6 budgets, but `matches_main_run: false` on 224 of the 250 compared
(fold, k) pairs.  Diagnosis: `fits_<design>.csv` holds the exact shortest-repr text of every beta (for example
BP / 104729 / fold 0 / k=1 is written `-0.09635133272505349`, and a fresh refit returns exactly that double), but
pandas' default float parser is not round-trip exact and returned `-0.0963513327250534` (1 ulp, -8.3e-17).  Across
both designs the default parser differs from the CSV text on 272 of 300 betas; the round-trip parser on 0.  The
defect is in the comparison, not the fit; no estimand, population, metric, test, threshold, label or decision rule
changes, and no number in any table changes.

**Results seen at the time of this amendment.** Every Stage-1 output then present, at least as log tails and the
headline lines the runners print:
results/EA_CHECK/, results/ECT/, results/ECT_M03/, results/ED/ (rows, cand, checks, fits, loco_selection and the
reproduction lines: C3 0.4988 on BP, C4 0.4821), results/EDEC/, results/EKX/, results/ELAB/, results/EW/,
results/WINDOW/, results/leakage/ (including poison_ED_beta.json and poison_ED_loco.json), and results/logs/.
E-D's tables (`--from-rows`) have not been produced; they are produced after this reseal.

**Next.** Reseal; rerun `s1_ed_poison.py` (it refits all 50 x 6 and recompares), then `s1_ed.py --from-rows --design both`.

## A2 — 2026-10-24 — Stage-1 results review: reporting fixes in nine items and one report-rule correction (supersedes seal 2f74292f5f44)

**Trigger.** After every Stage-1 item had run, an adversarial results review (one reviewer per item, two for E-D, each
finding checked by up to three independent skeptics) produced 65 verified findings (50 in eight items; 15 in E-D, several
found by both of its reviewers). Each was fixed in code by one fixer
per item, every diff was checked by an independent agent against a pre-fix snapshot, and the full test suite passes
(384 passed, 1 skipped: the sealed P1 test, which runs only under the seal). Every change is a documented code defect in
the reporting layer, a registered output that was missing, or a recipe deviation from the frozen ITEM_SPEC. No estimand,
population, arm, policy, metric, test, threshold, seed or report-rule logic changes. This is an error-triggered rerun
under brief §8 (documented defects, unchanged scientific choices, full disclosure); it is not a fresh holdout, and every
number stays EXPLORATORY (seen).

**The one report-rule outcome that changes.** E-D 1.6-R01 (Pareto question) was answered "a dominating 5th-read policy
exists" in BPG and BP only because the aggregate counted the ORACLE_B1 [O] bound (and the LOOSE sensitivities) as family
members. Over the written label-free family (DOPTQ-nested, MAXMIN_R/Z, BISECT_R, BISECT_RMID, HULL_BISECT, RANDOM) the
stored per-policy table already showed that no member dominates, in BPG and BP and in the all / noC5 / noC5cell variants.
The corrected answer is "no member Pareto-dominates" (the seen P3 answer); ORACLE_B1 [O] is reported on its own bound row.

**Numbers that change on rerun (all others reproduce bit for bit from the stored rows):**
- WINDOW: bootstrap and control lower percentiles now use the literal 0.025 / 0.975 with inverted_cdf, as ITEM_SPEC
  states; (1 - 0.95)/2 = 0.025000000000000022 had taken the 51st instead of the 50th of 2,000 draws. Tail lower bounds
  fall by at most 0.039 (BPG) / 0.024 (BP), e.g. BPG OP75 1.931 -> 1.929, noC5 1.198 -> 1.197; BP noC5 0.966 -> 0.965;
  ctrl q2.5 by at most 8e-4. No outcome changes (BPG tail flag PASS, BP FAIL).
- EA_CHECK: the registered V2-3 duplicate stratum (1.0-S09; 11 cells, one cluster_cpg unit) is now reported
  descriptively instead of as all-NaN (BPG B1 - B0 relative SSE -0.090, HIST_D7 - B1 +0.032; BP -0.086 / +0.030), with
  SF_POOL, W/L/T and the gate marked not computable at G = 1.
- E-W: the per-item table's per-seed row showed the noC5cell variant; it now shows the primary 'all' variant (per-seed
  values move by at most about 0.008; no mean, label or rule outcome depended on that row).
- E-D: the C3 4-read cell reads "(E-A rows)" in every E-D table (recipe_levels, density_population, duplicates, C5 table)
  instead of printing E-A's level.

**Per item (reporting layer; details in each fixer report and the per-item diffs):**
- EA_CHECK (r5/ea_check.py, scripts/s1_ea_check.py): §9.3 per-item tables per design and ITEM_TABLE.md when both
  designs run (the item is rerun with both designs in one process, so ITEM_TABLE.md and summary.json are written as
  ITEM_SPEC §5 lists); computed sentences; the RUN_RECORD detail no longer prints a missing key as FAIL (correction
  appended to RUN_RECORD); the V2-3 stratum; the 1.0-R03 record notes that the seen 5.85 % is pooled over seeds while the
  sealed currency is the seed mean (5.856 %); match stays false.
- ECT (r5/ect_report.py, scripts/s1_ect.py; r5/ect.py unchanged): cand / ref named in the duplicate-sensitivity table;
  unit tests (SIGN, IM-t, equal-weight relSSE) printed beside the unit table, small p in scientific notation; 1.1-R01
  labelled EXPLORATORY (seen) with its NOBIT / I1 rows SENSITIVITY as DECISION_LOG seals (this resolves a conflict with
  ITEM_SPEC §6's blanket "report rules CONSISTENCY CHECK"); GE1_NOBIT nsib rows SENSITIVITY; a computed 'What changed'
  sentence quoting P1's seen value of the same design (BPG +0.177, BP +0.174); a GE1_WITH_C5 stratum in nsib and
  annex_erec (brief §5.2; the >= 8 bin becomes 51 cells, relSSE SH vs B1 0.278 instead of 0.549 without C5) and a v25
  check row showing V2-5 leaves C5-BTBP unchanged; the mechanism row carries SH - PB_PUB, the per-publication recovered
  shares and "6 families (publication x extractant) in 3 publications".
- ECT_M03 (r5/m03.py, scripts/s1_m03.py): *_classic / *_argmin arms labelled SENSITIVITY (one-SE rule); the acid-trend
  claim statistic labelled EXPLORATORY (seen) as DECISION_LOG 1.2-R04 seals (REGISTRY 1.2-R04 and ITEM_SPEC §10's blanket
  line say CONSISTENCY CHECK; DECISION_LOG governs); computed §9.3 sentences (nested weighting passes on PRIMARY_NO_ORNL,
  carried by the read-match kernel; the classic one-SE rule picks uniform; re-nested LOPO minimum -0.0010; C-ECT stays
  sealed on SH@4 vs B1@4); unit tests and the OFS duplicate rows in the per-item table; the NB copy guard printed n.a.
  Not changed: ITEM_SPEC §12 still quotes the pre-fix smoke counts 1,098 / 33 (final 1,512 / 48, PRESEAL_REVIEW.md), and
  ITEM_SPEC §2 lists a standalone BP --repro that did not run in the first pass (the BP reproduction ran as the first step
  of the BP scoring run).
- ELAB (r5/elab.py, scripts/s1_elab.py): rule rows name their W/L/T unit (laboratories vs publications); every seen value
  names its pool (lab-1 SRC before near-copy cleaning) with the like-for-like NB-pool in-corpus value beside it; BP seen
  values where the source gives them; the same-series row names its contrasts and adds SS - B1 / OS - B1; labels routed
  through label(design, c5), so BP and the C5 variants carry SENSITIVITY; ORACLE [O] only on MX_GK_O; a note and a
  descriptive exact-enumeration SF_POOL column for the contrasts that equal the headline bitwise (the registered
  random-flip p is kept).
- WINDOW (r5/window.py, scripts/s1_window.py): the row flag's scale stated (inside relative SSE at NEST_FAM; its inside R2
  is below the cell-matched control mean); controls whose draws reproduce the rule flagged and their shares printed n/a;
  the count of NEST_FAM fold-fits with membership constant within every cell (13 / 25 BPG, 11 / 25 BP) beside the
  cell-level-rule count (9 / 8); literal percentile levels (above); the summary's Monte Carlo field renamed
  excess_relMSE_vs_CELL_MC; the relMSE_in note; a seal column on every CSV; results/leakage/WINDOW_<design>.json.
- EDEC (r5/edec.py, scripts/s1_edec.py): LOPO min / max as the spread, and the t(P-1) columns named as the descriptive
  jackknife interval; pooled counts beside every per-call false-sign rate (BPG B1 seed mean 0.194 vs pooled 4 / 44 = 0.091
  over 25 distinct queries in 4 of 5 seeds) with a few-calls flag; computed sentences; labels on platt / bounds_rows;
  ORACLE [O] on E_AURC; the arms-run repro record labelled as the in-run check; the S03 no-hull note; computed support
  counts in the leakage record.
- ED (r5/ed.py, scripts/s1_ed.py): the Pareto correction (above); the "(E-A rows)" writer rule everywhere; the row named
  "C3 min(k, k_max), k = reads - 1", with the 5-read mixture stated (286 of 1,802 rows per seed keep 4 reads, B1 0.60 on
  them; C4 is the like-for-like 5-metal number); the "different read budget" tag on every 5-7-read number; relative SSE vs
  B1 at 4 reads in the per-item table; the saturation rule outcome in the per-item table; computed next step and compute
  line; the reproduction gate also requires the 5-read B0 vs R3-B0 identity and the 4-read B0 / HIST_D7 identities (all
  0.0 on the stored run; the 4-read B0 / HIST_D7 checks are disclosed hardening beyond ITEM_SPEC's list); the
  envelope-width note (exact RANDOM expectation, so BP 0.0095 vs the seen 0.0096 is expected).
- EW (r5/ew.py, scripts/s1_ew.py): per-seed row from variant 'all'; the WIDE population per design (225 cells,
  38 publications, 12 units; 4,394.2 / 4,384.6 rows per seed) with the STRICT denominator on a labelled line;
  results/leakage/EW_<design>.json; the 1.7-R01 label with the E-W marker and CONSISTENCY CHECK; BP rows SENSITIVITY; the
  interval's unit count P; absolute MSE columns.
- EKX (r5/ekx.py, scripts/s1_ekx.py): the frozen value's provenance reworded to "about -0.11 [IR M07], computation not
  specified; V1 publication-first mean -0.122, cell-weighted -0.110" (the frozen -0.11 and u_ext -0.077647 unchanged);
  the failed 2-decimal consistency line surfaced in the log, the per-item table and the summary; every top3_rule row
  listed with the sealed BP-decides exception; BPG labels SENSITIVITY; the E-KX-LOPO note on arms / contrasts labels; seal
  columns and digests on every output. Disclosed: the permutation null draws reproduce across processes only to about
  2e-16 (observed statistics, counts and p identical).

**Report-level items (no code change; disclosed in REPORT.md):** E-CT's standalone BP reproduction ran after the BP
scoring run in the first pass (the scoring run's own reproduction gate ran first and matched); EDEC's arms run overwrote
the standalone reproduction record (identical values); the E-A replay 5.85 % vs 5.86 % is definitional; the source
computation of E-KX's -0.11 is not identifiable.

**Results seen at the time of this amendment.** Every Stage-1 output then present, in full, through the review:
results/EA_CHECK/, results/ECT/, results/ECT_M03/, results/ED/, results/EDEC/, results/EKX/, results/ELAB/, results/EW/,
results/WINDOW/, results/leakage/ and results/logs/ (archived unchanged in cache/_superseded_2f74292f5f44/; the A1-era E-D
outputs are in cache/_superseded_1f0cae0064a6/).

**Next.** Reseal; rerun every item in the frozen order (reproduction first, TECHNICAL SMOKE; then the scoring runs;
E-D: --repro, run, s1_ed_poison.py, --from-rows; EA_CHECK with both designs in one process; EDEC after ECT); compare
every rerun table with the archive (only the changes listed above may differ); then the post-run seal checks.
