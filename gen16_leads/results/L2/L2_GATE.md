# L2 gate — the honest (leave-pair-out) curvature headroom

*Discovery seeds (5), designs B / BR / BQ / A / BP, extractant-macro MAE of pairwise log SF, n = 90 extractants, chemotype-blocked paired bootstrap (10 000 replicates). Run 2026-09-10T07:05:31, 102 s.*

## Verdict: **CLOSED**

Registered rule: OPEN if H = MAE(G14) − MAE(O_CURV_LPO) ≥ 0.02 under BP and the 95 % CI excludes zero.

* H under BP = **+0.0287**, percentile 95 % CI [-0.0013, +0.0532], BCa [+0.0000, +0.0544], p = 0.0608, 61/90 extractants improved, 5/5 seeds positive, LOCO sign stable = True, passes P1 = False.
* H ≥ 0.02: True.  CI excludes zero: False (BCa: True).
* H in the five designs: B +0.0298, BR +0.0268, BQ +0.0264, A +0.0323, BP +0.0287; same sign in all five: True.

## Reproduction of the gen15 anchors through this loop (BP)

| arm | gen15 | this run | abs diff | ok (≤ 0.001) |
|---|---|---|---|---|
| G14 | 0.500079 | 0.500079 | 0.00e+00 | True |
| O_CURV | 0.427211 | 0.427211 | 0.00e+00 | True |
| FLAT | 0.588506 | 0.588506 | 0.00e+00 | True |

Gate valid (O_CURV reproduces 0.4272 ± 0.001 and G14 0.5001): **True**.

## Five-design table (extractant-macro MAE)

| arm | B | BR | BQ | A | BP |
|---|---|---|---|---|---|
| FLAT | 0.5885 | 0.5885 | 0.5885 | 0.5885 | 0.5885 |
| G14 | 0.4932 | 0.4906 | 0.4905 | 0.4921 | 0.5001 |
| O_CURV | 0.4216 | 0.4218 | 0.4233 | 0.4168 | 0.4272 |
| O_CURV_LPO | 0.4634 | 0.4638 | 0.4642 | 0.4599 | 0.4714 |

## Contrasts (positive favours the candidate)

| design | comparison | family | point | 95 % CI | BCa | p | improved | seeds | LOCO | P1 |
|---|---|---|---|---|---|---|---|---|---|---|
| BP | OCURVLPO_vs_G14 | registered | +0.0287 | [-0.0013, +0.0532] | [+0.0000, +0.0544] | 0.0608 | 61/90 | 5/5 | True | False |
| BP | OCURV_vs_G14 | exploratory | +0.0729 | [+0.0304, +0.1334] | [+0.0391, +0.1624] | 0.0006 | 65/90 | 5/5 | True | True |
| BP | OCURVLPO_vs_FLAT | exploratory | +0.1171 | [+0.0149, +0.1860] | [+0.0386, +0.2041] | 0.0190 | 57/90 | 5/5 | True | True |
| BP | OCURV_vs_OCURVLPO | exploratory | +0.0442 | [+0.0158, +0.1027] | [+0.0192, +0.1341] | 0.0000 | 70/90 | 5/5 | True | True |
| B | OCURVLPO_vs_G14 | registered | +0.0298 | [-0.0020, +0.0563] | [-0.0018, +0.0565] | 0.0650 | 63/90 | 5/5 | True | False |
| B | OCURV_vs_G14 | exploratory | +0.0716 | [+0.0369, +0.1167] | [+0.0416, +0.1266] | 0.0000 | 62/90 | 5/5 | True | True |
| B | OCURVLPO_vs_FLAT | exploratory | +0.1251 | [+0.0176, +0.1978] | [+0.0433, +0.2187] | 0.0148 | 57/90 | 5/5 | True | True |
| B | OCURV_vs_OCURVLPO | exploratory | +0.0418 | [+0.0158, +0.0938] | [+0.0190, +0.1201] | 0.0000 | 68/90 | 5/5 | True | True |
| BR | OCURVLPO_vs_G14 | registered | +0.0268 | [-0.0062, +0.0529] | [-0.0047, +0.0539] | 0.1040 | 61/90 | 5/5 | True | False |
| BR | OCURV_vs_G14 | exploratory | +0.0688 | [+0.0333, +0.1127] | [+0.0389, +0.1240] | 0.0002 | 63/90 | 5/5 | True | True |
| BR | OCURVLPO_vs_FLAT | exploratory | +0.1248 | [+0.0159, +0.1990] | [+0.0421, +0.2199] | 0.0168 | 57/90 | 5/5 | True | True |
| BR | OCURV_vs_OCURVLPO | exploratory | +0.0420 | [+0.0159, +0.0942] | [+0.0191, +0.1193] | 0.0000 | 68/90 | 5/5 | True | True |
| BQ | OCURVLPO_vs_G14 | registered | +0.0264 | [-0.0087, +0.0538] | [-0.0093, +0.0535] | 0.1290 | 62/90 | 5/5 | True | False |
| BQ | OCURV_vs_G14 | exploratory | +0.0672 | [+0.0342, +0.1056] | [+0.0384, +0.1118] | 0.0002 | 62/90 | 5/5 | True | True |
| BQ | OCURVLPO_vs_FLAT | exploratory | +0.1243 | [+0.0162, +0.1973] | [+0.0425, +0.2180] | 0.0160 | 57/90 | 5/5 | True | True |
| BQ | OCURV_vs_OCURVLPO | exploratory | +0.0408 | [+0.0157, +0.0901] | [+0.0190, +0.1133] | 0.0000 | 68/90 | 5/5 | True | True |
| A | OCURVLPO_vs_G14 | registered | +0.0323 | [+0.0032, +0.0581] | [+0.0045, +0.0596] | 0.0306 | 65/90 | 5/5 | True | True |
| A | OCURV_vs_G14 | exploratory | +0.0753 | [+0.0379, +0.1292] | [+0.0444, +0.1489] | 0.0000 | 67/90 | 5/5 | True | True |
| A | OCURVLPO_vs_FLAT | exploratory | +0.1286 | [+0.0216, +0.2027] | [+0.0460, +0.2236] | 0.0108 | 58/90 | 5/5 | True | True |
| A | OCURV_vs_OCURVLPO | exploratory | +0.0431 | [+0.0159, +0.0987] | [+0.0190, +0.1261] | 0.0000 | 68/90 | 5/5 | True | True |

## In-sample vs leave-pair-out gap of the own-curvature oracle

gen15 `s4b_oracle_honesty.csv` (one seed, all 521 cells in one frame, **true** sign, global constants): SIGN_OWNB 0.3771 → SIGN_OWNB_LOPO 0.4245, gap +0.0474.

This run (5 discovery seeds, fold-wise pair tables, **G14's predicted** sign and training-fold constants): B +0.0418, BR +0.0420, BQ +0.0408, A +0.0431, BP +0.0442.

