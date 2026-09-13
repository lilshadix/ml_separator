# Exploratory probe: does M1's advantage track slope reliability?

**Exploratory, not pre-registered, changes no decision.** Written by `scripts/g18_reliability_probe.py` after R1 returned a null.

*regime: cohort=E1 cohort (14 systems); holdout=LOPO by publication (E1), reliability from jackknife/split-half by publication; averaging_unit=system; status_of_parameters=fitted_from_corpus; EXPLORATORY, not pre-registered*

## Hypothesis

The pooled mass-action fit M1 should beat the nearest-condition baseline B1 on systems whose fitted exponent `n` is reliable (`interpretable_n`, i.e. jackknife SE < 0.5 and, where defined, split-half sign agreement >= 18/20) and should lose where it is not. If true, gating M1 on the reliability flag would have rescued the R1 null.

## Numbers

| group | systems | mean advantage (MAE(B1) - MAE(M1)) | M1 wins |
|---|---|---|---|
| interpretable `n` | 6 | +0.072 | 3/6 |
| not interpretable | 8 | -0.101 | 4/8 |

- Spearman(jackknife SE of `n`, advantage) = **-0.117** (p = 0.690, n = 14). A reliability story predicts a clearly negative correlation.
- Mann-Whitney, interpretable > not interpretable: U = 25.0, p = 0.475.
- The two largest M1 wins (+0.741, +0.627) are both on systems whose slope is **not** interpretable (jackknife SE 0.48 and 2.32).

## Verdict

**NOT SUPPORTED: the reliability of the fitted exponent does not explain where M1 helps.**

Consequence: the R1 null is not an artefact of unreliable slopes, and a reliability gate is not a route to rescuing M1. The chain keeps B1 as its D source, as the sealed rule says. The reliability flag keeps its pre-registered job (it still gates GP-BO and the interpretation of any individual slope); it is simply not a predictor of where the mass-action form beats a nearest-neighbour lookup.

## Per-system table

| ligand | n | jackknife_se_n | interpretable_n | mae_M1 | mae_B1 | advantage_b1_minus_m1 |
|---|---|---|---|---|---|---|
| TODGA | 3.759 | 0.4842 | False | 1.001 | 1.742 | 0.7413 |
| TODGA | 1.408 | 2.317 | False | 0.5057 | 1.132 | 0.6268 |
| DMDODGA | 2.597 | 0.1944 | True | 0.4411 | 0.9704 | 0.5293 |
| DMDODGA | 2.222 | 0.3897 | True | 0.4172 | 0.9276 | 0.5104 |
| 2-[1-(dioctylamino)-1-oxopropan-2-yl]oxy-N-N-dioctylpropanamide | 3.066 | 0.03855 | True | 0.7222 | 0.9268 | 0.2046 |
| NTAamide(C8) | 3 | 0 | False | 1.301 | 1.351 | 0.04943 |
| TEHDGA | 3 | 0 | False | 1.401 | 1.438 | 0.03628 |
| TODGA | 2.453 | 0.5431 | False | 0.8336 | 0.7462 | -0.08744 |
| TEHDGA | 3.024 | 0.2523 | True | 0.7027 | 0.5837 | -0.119 |
| D3DODGA | 3 | 0 | False | 1.38 | 1.096 | -0.284 |
| TBDGA | 2.415 | 0.4289 | True | 0.9435 | 0.6409 | -0.3026 |
| DMDOHEMA | 2.502 | 0.2761 | True | 1.795 | 1.404 | -0.3905 |
| TODGA | 2.182 | 0.8303 | False | 1.475 | 0.9641 | -0.5105 |
| C5BTBP | 1.284 | 0.8579 | False | 2.346 | 0.9687 | -1.377 |

