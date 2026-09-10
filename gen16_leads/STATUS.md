# Gen16 — status checkpoint

*Updated at the end of each phase so the work survives a context reset.  Newest entry first.*

## Phases 4, 5, 6 — 2026-09-10 03:40

**Phase 4, refutation.**  Six refuters, two per claim, blind to each other.  **No claim survived
both refuters untouched.**  L3A dented materially (the saving is between-laboratory: +0.005 within
one publication, interval containing zero in all five designs); L4BP killed (a featureless
"biggest chemotype first" order reproduces it, positive in all five designs, while A-optimal adds
+0.005 at p 0.36); L1's null dented materially and its **verdict corrected** from "closed" to "not
closed, not positive" — the registered estimator has split-half reliability −0.27 and jackknife
reliability 0.000, so the |ρ| ≥ 0.40 bar was unreachable in principle, and attenuation-free
corrections give +0.29 to +0.45.  Full log in `REFUTATION_LOG.md`.

**Phase 5, confirmation.**  Executed once on the withheld seeds after `g16_confirm.py` verified
them against the published commitment and the registered rule.  One claim qualified.  It
replicated: BP +1.909 against a discovery +1.988, interval excluding zero, 5/5 seeds, LOCO-stable,
passing in all five designs.  `CONFIRMATION.md`.

**Phase 6, audit.**  Two adversarial auditors (numbers: 311 statements traced to artefacts, 13
wrong; claims/scope/protocol: 64 checked, 2 wrong).  Both returned "publishable after fixes" and
both found the report **under-claiming**, not over-claiming.  All findings were verified against
the artefacts and folded in:

- **fatal** — the +0.29/+0.45 attenuation-free estimators that carried my L1 correction fail the
  mandatory diglycolamide control (`CYCLE_ADD` +0.291 → −0.026) and score +0.049 on
  constant-composition series against +0.476 where composition varies.  They re-admit the
  artefact.  **L1's verdict is now *undecided*, not "not closed, not positive"**, and the Stage 2
  recommendation is downgraded to "cheap and decisive, expect a null".
- **major** — L1's registered family-wise permutation bar (0.609) was computed and not quoted:
  nothing in the family clears it, gen15's +0.644 included.  L1's registered cheapest competitor
  (`frac_donor_pairs_within_3`, −0.42 to −0.53) was missing and is larger than every xTB estimator.
- **major** — the fold-purity dent had no number: +0.906 of E_random 5.52 (16.4 %) against 28.8 %.
  Now carried in the headline, §2 and guardrail 1.
- **major** — `LOWRANK2 − POOLED` @k2 is +0.0151, twice the `MIX6meanPC` gain I called "the largest
  real effect".  Under-claim, corrected.
- **major** — 940 vs the 860 the itemisation summed to: 80 rows are a discarded dry run in
  `results/L4/_dry/`, now disclosed; γ_nitrate is 17.8 eV from the yardstick, not 0.34 (only water
  is); "91 of 100 from one publication" was a per-extractant column misread (they span 23).
- **major** — an undisclosed deviation on the confirmed claim: L3 used its own 2 000-replicate
  percentile bootstrap, not the frozen 10 000-replicate percentile+BCa `paired_contrasts`, so the
  BCa half of P1 was never computed for it.  Now §10 item 10.
- **minor** — L2 reported as "cannot distinguish from zero" while L1 was un-closed for lack of
  power: the same argument applies, L2's MDE (0.039) is twice its own margin.  Now "closed by
  rule, underpowered by design".  BCa interval now printed.  Calibration paragraph's non sequitur
  fixed.  Layout deviation from `START_HERE.md` §8 now stated with its mapping.  Stale
  cross-references fixed.
- the composition-position finding now carries the fact that kills the hype: it correlates +0.79
  to +0.86 with the naive xTB slope, i.e. it *is* the composition step.

Deliverables complete: sealed pre-registration, `DECISION_REPORT.md`, `CONFIRMATION.md`,
`REFUTATION_LOG.md`, `results/MANIFEST.sha256` for the 26 excluded large artefacts, and 35 passing
regression tests (2 slow) pinning the report to its artefacts.

Comparison accounting, recounted from the CSVs: **940 contrast rows, 126 registered, 814
exploratory**; the confirmed claim's BP row is at BH q = 0.039 within the registered family and
q = 0.025 over all 940.

## Phase 2 complete — 2026-09-10 02:10

Five leads ran on the five discovery seeds under all five designs.  **One claim passed its
registered decision rule in all five designs (L3a).**  Two leads closed with a mechanism, one is
design-inconsistent, one is a set of sub-margin effects and two defects.

| lead | verdict | headline (BP unless stated) |
|---|---|---|
| **L1** cycle-corrected xTB | **CLOSED** | ρ(SPECIES slope, `a`) = **−0.084** on S8 (n = 62), CI [−0.297, +0.240], LOCO sign unstable.  Monotone dose–response in bookkeeping exactness: NAIVE +0.392 → SPECIES_NFILLCOL +0.374 → ELEM +0.313 → SPECIES_CONST +0.167 → SPECIES −0.084.  Gen15's +0.644 reproduces exactly (n = 39) and is a composition artefact; the naive slope and its curvature are one step function read twice (ρ = −0.981). |
| **L2** curvature | **CLOSED** (gate) | honest leave-pair-out headroom +0.0287, CI [−0.0013, +0.0532], p 0.061; MDE 0.039.  `r0` not run. |
| **L3a** measurements saved | **POSITIVE, all five designs** | the direction call cuts expected measurements to the first useful candidate from **6.91 to 4.92, saving 1.99 (28.8 %)**; five designs +1.99 to +2.49; permutation p = 0.0000 (2000 reps) in all five; CI [+0.44, +3.78]; "always heavier" saves exactly 0. |
| **L3c** ranking at k = 1 | **NULL** | `G14@k1 − NAIVE_LINE@k1` on regret: +0.003 / +0.002 / −0.003 / −0.017 / −0.021, p 0.34–1.00; on Spearman consistently **negative** (−0.022 to −0.027, p 0.013–0.026).  Once a candidate has one measurement, the corpus adds nothing to ranking. |
| **L3b** within-chemotype ranking | **not run** | registered as conditional on L1 being positive; L1 closed. |
| **L4** acquisition order | **design-inconsistent, fails the rule** | A-optimal vs random ABC: B −0.012, BR +0.023, BQ −0.003, A +0.037, **BP +0.050** [+0.027, +0.066], 5/5 seeds, percentile 100/100 among random draws.  Passes P1 under BR, A, BP; fails under B and BQ, which differ from BR only in training-side masking.  Prospective ranking of 95 bundle + 273 logK candidates delivered. |
| **L5** covariance and calibration | **no P1 claim; two defects** | `MIX6meanPC` beats the *deployed* baseline (the comparison gen15 never ran) by **+0.0070** at k = 3, p < 1e-4, design-invariant, 5/5 seeds, LOCO-stable — but 3× under the 0.02 margin.  Ledoit–Wolf is catastrophic (BP MAE 34.4 at k = 3) because the deployed residual covariance is **indefinite on 100 % of its 430 leave-chemotype-out estimates** and greedy D-optimal seeks those directions.  Measured-mode intervals over-cover: nominal 90 % covers **98.8 %** at k = 3 (zero-shot 92.6 %), driven by the fixed `NOISE_VAR = 0.09`.  Platt recalibration of the direction probability makes it worse; the raw probability beats the base rate on Brier in all five designs but its ECE is 0.139, above the registered 0.10 bar. |

Comparison accounting so far: L1 90, L2 20, L3 85, L4 435, L5 230 contrast rows written.

**Phase 4 launched**: two blind refuters each on L3A, L4BP and the L1 null.

## Phase 2 partial — L2 closed 2026-09-10 00:52

**L2 gate: CLOSED, the lead ends here** (`results/L2/L2_GATE.md`).  The honest leave-pair-out
curvature headroom under BP is **+0.0287**, percentile 95 % CI [−0.0013, +0.0532], p = 0.061,
61/90 extractants improved, 5/5 seeds, LOCO [+0.022, +0.034] — the point clears the 0.02 margin
but the interval does not exclude zero, so P1 fails and the registered gate is closed.  Same in
B (p 0.065), BR (0.104), BQ (0.129); only design A excludes zero and A never selects.  Bootstrap
SE 0.014 → minimum detectable headroom 0.039, so this corpus cannot resolve a curvature prize of
the registered size.  About 0.044 of gen15 §2's in-sample +0.073 was the oracle fitting the noise
of the pair it is scored on (gen15 §1a measured +0.047 by a different route).  The `r0`
reparametrisation was **not run**, per the registered stopping rule.  Loop validity: G14, O_CURV
and FLAT reproduce the gen15 BP anchors to the last digit.
Temptation recorded (not acted on): the BCa lower bound is +5.7e-6, so reading "the CI" as BCa
alone would have opened the gate on a six-millionths margin; P1 requires both intervals and
p = 0.061 fails regardless.

**The other four Phase 2 agents (L1, L3, L4, L5) were killed by a model usage limit, not by an
error**, and were resumed on the same workflow run (`wf_857279f0-21e`) after switching to
Opus 5.  L2 replays from cache.

## Phase 0 complete — 2026-09-10 00:35

**Confirmed**

- All three anchors reproduce exactly (`results/anchors/ANCHORS.md`, `tests/test_anchors.py`):
  `G13_ET_TOPO39` BP macro accuracy `0.7683085207475452` (exact); `G14` BP macro MAE
  `0.5000794414203691`; `FLAT` `0.5885062528901843`; `G14 − FLAT` +0.0884 [+0.0010, +0.1473]
  p 0.047, passes P1; G14 five designs 0.4932 / 0.4906 / 0.4905 / 0.4921 / 0.5001 (B/BR/BQ/A/BP).
- Fold plan SHA-256 `7046c640…d440d16` identical across subprocesses and across
  `PYTHONHASHSEED` values; the only `hash()` on the bench path is an int-tuple (unsalted).
- Environment has not drifted since gen13 (`results/env/ENV.md`): sklearn 1.9.0, pandas 3.0.5,
  numpy 2.5.3; `tabpfn` 2.2.1 is installed but not importable and its pins were never applied;
  `xtb` is absent from the machine (cluster route for L1 Stage 2).  Do not upgrade sklearn to 1.10
  (the frozen G14 arm uses a deprecated `penalty=` argument).
- Machine budget: one bench process peaks at ~230 MB; six concurrent bench processes with
  `n_jobs=2` / `OMP_NUM_THREADS=2` are safe on 8 GB.
- **L6 cohort audit is a clean null** (`results/L6_cohort_audit/L6_COHORT_AUDIT.md`): every
  bundle extractant with ≥ 2 lanthanides measured anywhere is in the frozen cohort; no single
  relaxation adds a chemotype (maximum +0 against the brief's bar of 5); Kish 11.67 reproduced as
  gen13's extractants-per-chemotype definition.  The 100 excluded compounds are single-lanthanide
  (Eu 567 rows, Pr 96, Nd 35) and carry **53 chemotypes absent from the cohort**; measuring one
  second lanthanide on one compound per absent chemotype would take Kish n_eff 11.67 → 27.4.
  Handed to L4 as candidate pool (i).  One documentation defect noted (key mode `series` uses the
  relaxed column set); no code changed.

**Pending**

- Phase 1: `PRE_REGISTRATION.md` finalised with its hash footer and committed (this checkpoint's
  commit).
- Phase 2 (launching next): L1 Stage 1 bookkeeping refit + Stage 2 hand-over; L2 gate (+ r0 if
  open); L4 acquisition simulation + prospective ranking; L5 covariance + calibration; L3a and
  L3c.  L3b waits on L1's verdict.
- Later: L1 MAE arm if Stage 1 is positive; refuters; confirmation; report; audit.

**Abandoned / closed**

- L6 closed as a clean null (no cohort expansion is available from the filters).

**Withheld confirmation seeds**: rule and commitment in `PRE_REGISTRATION.md` §0; values held by
the orchestrator outside the repository; `scripts/g16_audit_seeds.py` is clean.

**Uncommitted prior-session work found in the tree and left untouched** (not part of this brief,
never committed with gen16 work): `gen16_anchor/`, `gen16_protocol/`, `gen17_pairdiff/`,
`gen14_direction/scripts/g14_{straw,perm2,capacity}.py` and their results,
`gen13_separation/gen13sep/wildcluster.py`, `gen13_separation/scripts/g13_inference_audit.py`,
`gen15_curve/scripts/g15_anchor.py`, and three modified tracked files under `gen14_direction/`.
