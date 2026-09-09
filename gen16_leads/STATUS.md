# Gen16 — status checkpoint

*Updated at the end of each phase so the work survives a context reset.  Newest entry first.*

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
