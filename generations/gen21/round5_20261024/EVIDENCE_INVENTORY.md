# Round-5 evidence inventory (one page; Stage 0, 2026-10-24, before the seal)

There are three kinds of claim here: what this round's code or results verified; what only the design sources report;
and what is a hypothesis. No round-5 score exists. "S0:" = `results/stage0/`. Source tags are those of the brief header.

## A. Verified in this round (Stage-0 runs, checks and technical smokes; no score)

| claim | evidence |
|---|---|
| Round-4 seal 5bacfe4c verifies; round-3 seal 93d3839a holds (freeze manifest raw digest = `freeze_seal.txt`, LF digest 96cd23a9… matches, 78/78 code files match their LF digests) | `r4_seal.py --verify` at 04:40 UTC (RUN_RECORD) and at 07:37:41 UTC; direct LF check at about 07:38 UTC |
| β_frozen = -0.110809 (BPG, λ 0.003, 229/229 C3 cells, inner seed 3246616). λ 0.003 beat 0.03 by 7.6e-5. Sensitivities: BP -0.1095, λ 0.03 -0.0878, round-4 fold means -0.0923 / -0.0949 (from `fitted_<d>.json`; λ 0.03 in 23 of 25 folds) | S0 `beta_frozen.json` |
| The external DOPT_Q object rebuilds bit-identically (521 training cells, s2_all 0.1785). The C3-only fit keeps only 89/229 four-read sets | S0 `ext_dopt.json`; `cache/ext/ext_dopt.pkl` |
| The 5.1 bit rule flags 6 links (items 1, 3, 4 and three V2-2 pairs in pub_b9dfc468cd), none across publications. Items 5-11 are detected by the sensitivity scans. **Item 2 (cell 652810 ≈ 440129, sd 0.04) is detected by no scan.** 23 targets have a lab-1 near-copy donor | S0 `dupscan_summary.json` |
| E-CT PRIMARY 99 / 12 / 14 / 690, with and without the bit rule; all 7 P1 count targets reproduce | S0 `ect_population.json` |
| Supply audit is metadata-only (value columns asserted absent). The kill fires for every claim (C-ECT 2 vs 7; C-GD 1 vs 6; Gd-dip 5 vs 10; SAFE 3 vs 10) | S0 `supply_audit.json`; `SUPPLY_LEDGER.csv` |
| Every chord statistic and both laboratory-level C-ECT statistics fail the size gate (max sizes 0.1471 / 0.1434 / 0.0979). The size gate reproduces `clustered_null_size.csv` (63 cells, max difference 0.0069). The chemotype-level C-ECT sign-flip passes (max 0.0268, 0 of 84 rows flagged) | S0 `size_gate.json`, `size_gate_ect_chemotype.json` |
| C-GD power in worlds reproduces I2 confirm-1 (EMP_HS12: 0.48 / 0.71 at -0.118, 0.89 / 0.99 at -0.072). The D2 sensitivity file reproduces the β_frozen rows exactly | S0 `power_worlds.json`, `power_cgd_beta_sens.json` |
| A/B/BR/BQ fold records: 100 built; cache poisoning identical 100/100; isolation passed | `results/leakage/cache_poison_EA_CHECK.json`; `manifests/ea_check_caches.json` |
| Technical smokes on real data (metrics suppressed) show these identities: <br>- B1 from fold betas = REF within 8.9e-16; y exact; 0 read-set mismatches; no scored pair with both ends read (ECT, M03, ELAB, EKX). <br>- M03 shares r5.ect arms exactly (max difference 0.0); its fit poisoning is identical 1,098/1,098; its positive controls are caught 114/114. <br>- sc009 Pr/Nd has 46 rows per seed. <br>- E-REC PRIMARY_OWN is 112 / 16 / 790 | `results/{ECT,ECT_M03,ELAB,EKX,EDEC}/SMOKE_COUNTS_*.json` |
| Spec-listed RNG seeds reproduce the registry scheme 20261023 + crc32(key) (WINDOW, EKX) | `REGISTRY.csv`; `r5.paths.seed_for` |
| **Gaps found:** <br>- the β_frozen digest recorded by `size_gate.json` / `power_worlds.json` (7f6025fe…) no longer matches the current file (2b5d0e2d…; values differ by 4e-17); <br>- WINDOW, ED and EW smokes write no output file, so their smoke claims are specs only | DECISION_LOG N1; this table |

## B. Reported by the design sources only (not yet recomputed in round 5; Stage 1 checks them)

- E-A reference B1 0.4521 / 0.4519, and B1 - B0 +0.034 R² that fails the jackknife-t [R4-REPORT 1; EB 1.5]. Replay
  B1 - B0 +5.85 %, SF_POOL p 0.00317 / 0.00269 (item 1.0 checks it) [IR M35; PC P2].
- M34's kill powers 0.228 / 0.236 and 0.285-0.296; the planning G table; JK_G1 anti-conservative [P2 3-4; PC P2].
- Reproduction targets of brief 6.0 (E6 V1 0.5734 / 0.5946 / 0.4489; P1 PRIMARY 0.5309 / 0.7082 / 0.6987 / 0.6963; P3
  0.4968 / 0.4988, C4 0.4798 / 0.4821; window 0.540 at 0.743-0.754; E-LAB 0.444 → 0.494).
- Seen in-corpus leads:
  - SH - B1 +0.177 [+0.076, +0.279] [PC P1];
  - MX_LM10 +7.6 % cleaned [I2 lab-1];
  - window P(|res| > 0.5) 0.009 vs 0.043 [IR M09];
  - sc009 Pr/Nd 0.265 → 0.522 [IR M07];
  - Nd/Sm B1 - LINE +0.084 [IR M13];
  - envelope 0.0091 / 0.0096 [P3].
- The closure ledger, every line [EB 3; IR; I2].
- Provenance: C5-BTBP holds 23.1 % of B1's SSE and 34.9 % of its gain; the 69 unlocated cells hold 56.4 % / 68.4 %
  [P5 3; PC P5].
- Supply counts (16 E1+E2 laboratories; 15 of 16 on absent chemotypes) [P4; PC P4].
- Planning amplitudes HDEHP -0.26 and P507 -0.105, both **unverified** [I2 confirm-1, confirm-2].

## C. Hypotheses (under test, data-gated, or sealed for new data only)

- **H-ECT:** own-series templates plus 3-4 reads transfer to new laboratories (C-ECT; T1-T3 changes). In-corpus support
  is EXPLORATORY and 94 of 99 PRIMARY cells are DGA.
- **H-GD:** cation exchangers carry a deeper Gd dip (about -0.2 to -0.26) than β_frozen (C-GD; descriptive under D3).
  The Gd dip transports to new sources (M33).
- **H-ND:** a negative Nd chord for new sc009 publications (M07 rider). **H-CE:** the Ce/Pr acid-slope sign is negative
  in DGA-class mineral acid (shift-3 rider; sign only).
- **H-PROV:** B1's gain rests on verifiable values (U1 / M25: the "anchored" rule).
- **H-NOISE:** label noise explains a bounded share of B1's MSE, within [0.08, 0.87] (U4 / M11). **H-LAB:** a
  laboratory signature V_x/V_w ≥ 5 (M12).
- **Not hypotheses this round:** any E-A gain. E-A is closed on this corpus, gains below about 0.01 R² are undetectable,
  and no slot is registered.
