# R1 decision (written by scripts/g18_eval_dmodels.py)

Regime: cohort = E1 cohort (14 systems); hold-out = leave-one-publication-out within system; averaging unit = system (points -> (publication, metal) -> system -> macro); parameters fitted_from_corpus, band 20-30C.

## Rule (PRE_REGISTRATION.md section 5, verbatim)

**R1.** M1 is *adopted as the default D source for corpus systems* if and only if all of:
(i) macro E1(M1) < E1(B1) - 0.05 and E1(M1) < E1(B0) (margin 0.05 log D: below that the process consequence is inside the replicate floor 0.225 and the cheaper B1 is preferred);
(ii) the paired system-level bootstrap (2000 resamples of systems with replacement, seed 18, percentile 95 % interval of mean(E1(B1) - E1(M1))) excludes zero;
(iii) M1 beats B1 in at least 9 of the 14 systems (ceil(0.6 x number of systems); the audit's count sets the number).
If any of (i)-(iii) fails the result is a **null**: the process chain uses **B1 (`NearestConditionD`, with the ideal depletion correction and the prior `n0`) as its default D source for corpus systems**, M1 becomes an exploratory option with its LOPO table shown, and the report says so verbatim. A null is a result of this generation, not a failure of it.

## Numbers

| quantity | value |
|---|---|
| E1(M1) macro | 1.0904 |
| E1(B0) macro | 1.1830 |
| E1(B1) macro | 1.0637 |
| E1(M1_offset) macro (one_measurement_mode, labelled) | 1.2551 |
| condition (i): E1(M1) < E1(B1) - 0.05 | False |
| condition (i): E1(M1) < E1(B0) | True |
| condition (ii): 95 % bootstrap interval of mean(E1(B1) - E1(M1)) | [-0.3170, 0.2411] (mean -0.0267); excludes zero: False |
| condition (iii): wins of M1 over B1 | 7 of 14 (needed 9): False |

## Verdict

**the result is a null: the process chain uses B1 (`NearestConditionD`, with the ideal depletion correction and the prior `n0`) as its default D source for corpus systems, M1 becomes an exploratory option with its LOPO table shown, and the report says so verbatim. A null is a result of this generation, not a failure of it.**

Consequence for the chain: the default D source for corpus systems is **nearest (B1)** (`results/eval/decision.json`, key `m1_adopted = False`, read by `build_system_model(params_source="auto")`).
