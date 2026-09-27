# CONFIRMATION -- the single registered confirmation run (pre-registration section 15)

*Written 2026-09-26T23:14:58+00:00 by `scripts/g19_run_confirmation.py`. POST-HOC addendum 5 item 1 makes this the CORE run: the frozen claims on the confirmation half with the 5 withheld seeds, the single V6 run with S2(a)-(c), S1(c) and S1(d). Whatever it returns is the result (section 15).*

## The withheld seeds

- commitment (section 15): `65e8ae8ceb8e92947a1da27f42e2689e74b04cc7b820d2b6df26889ceabf5f82`
- `scripts/g19_seal_prereg.py --verify-seeds`: **VERIFIED** (5 seeds) -- verified: 5 seeds hash to the committed digest (seeds not printed).

| opaque index | withheld seed |
|---|---|
| i1 | **277601** |
| i2 | **339062** |
| i3 | **755801** |
| i4 | **759937** |
| i5 | **784080** |

*Revealed here and nowhere else: every fold file, record, table and log line of the run carries the opaque index and the commitment digest alone. POST-HOC addendum 6 item 4: 'The five seeds and the --verify-seeds verdict are written into decisions/CONFIRMATION.md AFTER the single run has completed, never before; no artefact written before or during the run contains a seed value, and the run logs only the commitment digest and an opaque per-seed index.' So: every fold file, record, prediction frame, table and log line of the run carries i1..i5 and the commitment digest alone (confirmation.scrub, checked by scan_for_seed_leak over everything the run wrote); decisions/CONFIRMATION.md is written LAST, and it is the ONE artefact that names the values, beside the --verify-seeds verdict. The leak scan therefore runs twice: before CONFIRMATION.md exists it must find no seed anywhere, and after it is written that file must contain all five and every other file still none*

## The frozen claims on the confirmation half

| claim | design | Delta | margin | R19 | item 4 (seeds) | TOST | confirmed |
|---|---|---|---|---|---|---|---|
| C1 M2 vs B3i @ V5@V5 | V5 | +0.41152 | 0.10568948790529951 | **PASS** | 5 of 5 | PASS | **yes** |
| C2 M2 vs B0 @ V5@V5 | V5 | +0.671243 | 0.05 | **PASS** | 5 of 5 | PASS | **yes** |
| C3 M2 vs B6r0 @ V5@V5 | V5 | +0.42856 | 0.05 | **PASS** | 5 of 5 | PASS | **yes** |
| C4 B6:WITH vs B6:ACT_PERMUTED @ V2@V2 | V2 | +0.0282274 | 0.05 | **FAIL** | 5 of 5 | PASS | **no** |

## S1 -- the scientific claim (H1, V5)

```json
{
 "components": {
  "S1a": "PASS",
  "S1b": "PASS",
  "S1c": "UNDECIDED",
  "S1d": "PASS",
  "S1e": "NOT_COMPUTED"
 },
 "families": {
  "H3 (\u00a719 H3 row)": [
   "C4"
  ],
  "S1(b)": [
   "C2",
   "C3"
  ],
  "primary (H1, S1(a))": [
   "C1"
  ]
 },
 "rule": "section 9 S1: (a) the margin, (b) the constant baselines, (c) selectivity direction and logSF magnitude, (d) calibration, (e) error grows with support distance; S1 is UNDECIDED, never passed, while any component is missing or incomplete",
 "verdict": "UNDECIDED"
}
```

## S2 -- the applied claim (V6, once)

```json
{
 "completeness_unit": "contrast_record_set",
 "n_pairs_by_stratum_seed_mean": {
  "HNO3": 142.0,
  "acid=HCl": 64.0,
  "acid=mixed": 0.0,
  "acid=organic_acid": 3.0,
  "pooled": 209.0,
  "system=TODGA": 0.0
 },
 "n_pairs_registered": {
  "HNO3": 142,
  "pooled": 209
 },
 "n_systems_scored": 13,
 "per_seed": {
  "1": {
   "direction_delta_by_system": {
    "HNO3": {
     "B3i_derived": {
      "CCCCCCCCCCCCN(C)C(=O)C1CC(OC)CC(C(=O)N(C)CCCCCCCCCCCC)O1": -0.8181818181818181,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": 0.0,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": -0.5,
      "CCCCCCCCCCN(CCCCCCCCCC)C(=O)C(C)OC(C)C(=O)N(CCCCCCCCCC)CCCCCCCCCC": 0.0,
      "CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC": -0.46153846153846156,
      "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC": 0.35714285714285715,
      "CCCCCCCCN(CCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCC)CCCCCCCC": -0.5,
      "CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC": 0.1333333333333333,
      "CN(C(=O)COCC(=O)N(C)c1ccccc1)c1ccccc1": 1.0,
      "CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1": 0.0
     },
     "B3x_derived": {
      "CCCCCCCCCCCCN(C)C(=O)C1CC(OC)CC(C(=O)N(C)CCCCCCCCCCCC)O1": -0.8181818181818181,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": 0.0,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": -0.5,
      "CCCCCCCCCCN(CCCCCCCCCC)C(=O)C(C)OC(C)C(=O)N(CCCCCCCCCC)CCCCCCCCCC": 0.0,
      "CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC": -0.46153846153846156,
      "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC": 0.35714285714285715,
      "CCCCCCCCN(CCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCC)CCCCCCCC": -0.5,
      "CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC": 0.1333333333333333,
      "CN(C(=O)COCC(=O)N(C)c1ccccc1)c1ccccc1": 1.0,
      "CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1": 0.0
     },
     "B8": {
      "CCCCCCCCCCCCN(C)C(=O)C1CC(OC)CC(C(=O)N(C)CCCCCCCCCCCC)O1": -0.8181818181818181,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": 0.0,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": -1.0,
      "CCCCCCCCCCN(CCCCCCCCCC)C(=O)C(C)OC(C)C(=O)N(CCCCCCCCCC)CCCCCCCCCC": 0.0,
      "CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC": -0.46153846153846156,
      "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC": -0.2857142857142857,
      "CCCCCCCCN(CCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCC)CCCCCCCC": -0.5,
      "CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC": 0.06666666666666665,
      "CN(C(=O)COCC(=O)N(C)c1ccccc1)c1ccccc1": 0.0,
      "CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1": -1.0
     },
     "HEAVIER": {
      "CCCCCCCCCCCCN(C)C(=O)C1CC(OC)CC(C(=O)N(C)CCCCCCCCCCCC)O1": -0.8181818181818181,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": 0.0,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": 0.0,
      "CCCCCCCCCCN(CCCCCCCCCC)C(=O)C(C)OC(C)C(=O)N(CCCCCCCCCC)CCCCCCCCCC": 0.0,
      "CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC": -0.46153846153846156,
      "CCCCCCCCN(CCCCCCCC)C(=O)COCC(=O)N(CCCCCCCC)CCCCCCCC": 0.0714285714285714,
      "CCCCCCCCN(CCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCC)CCCCCCCC": 0.5,
      "CCCCCc1nnc(-c2cccc(-c3cccc(-c4nnc(CCCCC)c(CCCCC)n4)n3)n2)nc1CCCCC": -0.06666666666666665,
      "CN(C(=O)COCC(=O)N(C)c1ccccc1)c1ccccc1": 0.0,
      "CN(C(=O)c1cccc(C(=O)N(C)c2ccccc2)n1)c1ccccc1": 0.0
     }
    },
    "pooled": {
     "B3i_derived": {
      "CCCCCCCCCCCCN(C)C(=O)C1CC(OC)CC(C(=O)N(C)CCCCCCCCCCCC)O1": -0.8181818181818181,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": 0.0,
      "CCCCCCCCCCCCN(CCCCCCCCCCCC)C(=O)COCCOCC(=O)N(CCCCCCCCCCCC)CCCCCCCCCCCC": -0.5,
      "CCCCCCCCCCN(CCCCCCCCCC)C(=O)C(C)OC(C)C(=O)N(CCCCCCCCCC)CCCCCCCCCC": 0.0,
      "CCCCCCCCN(C)C(=O)COCC(=O)N(C)CCCCCCCC": -0.2857142857142857,
      "CCCCCCCCN(CCCCCCCC)C(=O)
```

## What this run did NOT do, with its cost and its consequence

- **power_check: NOT_RUN** -- the section 8 signal-injection power check over the H3 contrasts (y' = y + kappa*s, kappa in {0.1, 0.25, 0.5, 1.0}, every endpoint of the contrast refitted at its selected hyperparameters).
  - not run by: POST-HOC addendum 5 items 1 and 4: 'the power-check debt is inventoried, not discharged'
  - cost: 20 of 20 H3 contrasts owe the check and 0 have one: 6,272 injected refits, about 436 h of serial compute at the measured per-fold cost (evaluation/h3/h3_summary.json -> power_debt; the totals are an upper bound because each contrast is priced separately and a shared WITH leg is counted twice, h3.POWER_COST_BASIS). The de-duplicated figure for the deployed arm's two V5 contrasts is 336 refits, about 50 h serial
  - consequence: every one of those contrasts is reported UNDECIDED (no registered power check) in the words addendum 4 item 4 fixes, NEVER as a null and never as 'no effect'; the inventory and its cost are part of this report so the omission is visible rather than implicit
- **s2d_process_stability: NOT_RUN** -- None.
  - not run by: None
  - cost: None
  - consequence: None
- **v6_actinide_deltas: NOT_RUN** -- the section 11 V6 actinide deltas (H3 at confirmation): arms M0 (= B5, the deployed configuration of addendum 3 item 2) and B6, transforms WITH / WITHOUT / ACT_PERMUTED at the frozen per-fold hyperparameters, deltas WITH - WITHOUT and WITH - PERMUTED with R19 and TOST (epsilon = 0.05) on macro MAE log D, rank accuracy, logSF MAE and calibration.
  - not run by: POST-HOC addendum 5 item 1 (compute-driven, chosen by the user): 'It does NOT run the section 11 V6 actinide deltas nor any section 8 power check; both are reported NOT_RUN with their cost and their consequence'
  - cost: 30.3
  - consequence: H3 keeps the discovery-side verdict: UNDECIDED for both arms, with no power check, so it is never reported as a null (addendum 4 item 4). Delta logSF MAE for H3 stays NOT_DEFINED -- V6 is the only design that could have supplied it (CONFIRMATION_PLAN section 3.4 item 2), so the quantity now exists nowhere and no later run may create it: section 3.4 says V6 runs ONCE and addendum 5 spends that run on S2. The section 11 consequence is unchanged: actinide rows enter the deployed Ln configuration only on a *helps* verdict, and there is none

## Seed hygiene

- files scanned for a withheld seed's decimal form: 8000; verdict **clean**
