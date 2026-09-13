# R2 decision (written by scripts/g18_loading_check.py)

Regime: cohort = E2 cohort: publication-aware loading series; hold-out = the tracer point anchors log K, the other points of the same series are scored; averaging unit = series (weight 1); parameters: n from the in-sample M1 fit (results/dmodels) or the prior n0 = 3, log K anchored per series, O/A = 1 assumed (OA_ASSUMED), K_H unknown.

## Rule (PRE_REGISTRATION.md section 7, verbatim)

**E2** = per-series MAE of predicted log D over the non-tracer points, for C0 and C1, macro over the 10 series (each series weight 1). **R2.** C1 is *supported* if and only if (i) E2(C1) < E2(C0) on more than half of the series (>= 6 of 10) and (ii) the paired series-level bootstrap (2000 resamples, seed 18) 95 % interval of mean(E2(C0) - E2(C1)) excludes zero. **Every series counts as it falls**: the three flat pub_d3c970567f series and the rising TODGA/Ce series are expected to be wins for C0 and are not excluded, down-weighted or explained away; if they make R2 fail, the verdict is "the direction of the loading effect is not consistent across corpus publications and the ideal correction is not supported corpus-wide", and the process chain keeps the depletion term *as a mechanism with a flag* (`OA_ASSUMED`, `HIGH_LOADING`), not as a validated magnitude.

## Numbers (primary: all series, O/A 1)

| quantity | value |
|---|---|
| n_series | 10 |
| E2(C0) macro | 0.5137 |
| E2(C1) macro | 0.3047 |
| condition (i): C1 wins | 6 of 10 (needed 6): True |
| condition (ii): 95 % bootstrap interval of mean(E2(C0) - E2(C1)) | [0.0296, 0.4739] (mean 0.2090); excludes zero: True |

Per-series n source: sys_07ee9637c98c1e20: results/dmodels/sys_07ee9637c98c1e20.json; sys_5cb78e5000d40860: results/dmodels/sys_5cb78e5000d40860.json; sys_740f07a521006be1: in-sample fit computed here (no results/dmodels file); sys_81bcc3c06cbf0b4c: results/dmodels/sys_81bcc3c06cbf0b4c.json; sys_a7195d8a9d8696e0: results/dmodels/sys_a7195d8a9d8696e0.json

## Verdict

**C1 is supported** (both conditions hold).

## Secondary and exploratory (labelled)

| subset | O/A | n | E2(C0) | E2(C1) | wins | CI of C0 - C1 | supported |
|---|---|---|---|---|---|---|---|
| all | 0.5 | 10 | 0.5137 | 0.3115 | 6 | [0.0178, 0.4533] | True |
| loading_active | 0.5 | 7 | 0.7294 | 0.4314 | 6 | [0.0435, 0.5915] | True |
| all | 1.0 | 10 | 0.5137 | 0.3047 | 6 | [0.0296, 0.4739] | True |
| loading_active | 1.0 | 7 | 0.7294 | 0.4237 | 6 | [0.0677, 0.6170] | True |
| all | 2.0 | 10 | 0.5137 | 0.3246 | 6 | [0.0352, 0.4174] | True |
| loading_active | 2.0 | 7 | 0.7294 | 0.4545 | 6 | [0.0671, 0.5573] | True |
| publication_blind | 1.0 | 11 | 0.4502 | 0.3822 | 7 | [-0.0286, 0.1795] | False |

X1 (effective capacity phi, leave-one-point-out, exploratory): supported = False (2 loading-active series where LOO-MAE(X1) < E2(C1) - 0.1; rule needs >= 4). When not supported phi is a reported fudge and is used nowhere.

## Interpretation of the Sasaki 2015 TODGA/Nd series with the sourced LOC (addenda/ORCHESTRATOR_prefit_20260913.md, U6; interpretation only, no exclusion)

Series `ls_sys_5cb78e5000d40860_Nd_pub_5a68dc5665_3_0.1` (0.1 M TODGA, n-dodecane, 3 M HNO3, 4.9-12 mM Nd, D 23.5 -> 0.60) counted as it fell: mae_constant = 0.859, mae_ideal = 0.756, C1 wins = True. The sourced third-phase limit for this system is 0.008 M Nd in the organic (Tachimori, Sasaki, Suzuki 2002, doi 10.1081/SEI-120016073, abstract level). The C1 prediction puts the organic Nd above that limit at 2 of 5 non-tracer points (predicted organic Nd up to 0.0112 M); the collapse of D along this series is therefore read as a third-phase boundary, not as ligand depletion, and the cascade raises `THIRD_PHASE_RISK` above the sourced value for this system. This reading does not change R2: the series keeps its weight.
