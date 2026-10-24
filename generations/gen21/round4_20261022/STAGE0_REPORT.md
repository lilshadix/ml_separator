# Stage 0 report: strict references B0 / B1 / B2

All numbers here are **exploratory development**: V1, C3, k = 3 (four read metals), DOPT_Q, 5 dev seeds over the same 229 cells and 1,802 strict-adjacent pairs per seed.

- **Seal:** stage-0 seal `1a09964d…`, sealed 2026-10-22T16:16Z, before B1 / B2 were first scored.
- **Deciding design:** BPG, with BP as the robustness check (PROTOCOL §6).
- **Full tables:** `results/REF/TABLE.md`; raw files in `results/REF/`.

## Results

| arm | R² BPG | R² BP | balanced MAE BPG | R² 95% (BPG, source-cluster bootstrap) |
|---|---|---|---|---|
| B0 = PWL_R | 0.4180 | 0.4179 | 0.1061 | [0.328, 0.466] |
| **B1** = PWL_R + one Gd amplitude | **0.4521** | **0.4519** | **0.1011** | [0.394, 0.497] |
| B2 = 14 coherent potentials + 6 read-geometry features | 0.4299 | 0.4246 | 0.1050 | [0.351, 0.476] |
| HIST_D7 (fixed full-corpus −0.072; diagnostic only) | 0.4511 | 0.4506 | 0.1011 | [0.386, 0.496] |
| R14 (round 3, same rows) | 0.4520 | 0.4518 | 0.1011 | [0.394, 0.497] |
| HBK@ML / PWL_Z | 0.4188 / 0.4107 | 0.4142 / 0.4118 | | |

### Paired contrasts on BPG

Source clusters `cluster_cpg`: 13 nominal clusters, Kish 2.24, the largest holding 65% of rows.

| contrast | ΔR² | percentile 95% | jackknife-t 95% | Δbalanced MAE | seeds + |
|---|---|---|---|---|---|
| B1 − B0 | +0.0341 | [+0.018, +0.076] | [−0.009, +0.077] | +0.0050 | 5 / 5 |
| B2 − B0 | +0.0119 | [−0.001, +0.028] | [−0.003, +0.027] | +0.0011 | 4 / 5 |
| B2 − B1 | **−0.0222** | [−0.060, −0.012] | [−0.054, +0.009] | −0.0039 | 0 / 5 |
| HIST_D7 − B1 | −0.0010 | [−0.010, +0.003] | [−0.012, +0.010] | −0.0000 | 1 / 5 |

BP gives the same picture: B1 − B0 +0.0340, B2 − B1 −0.0273, HIST_D7 − B1 −0.0013.

### Where the reference gains sit (BPG, R²)

| subset | B0 | B1 | B2 |
|---|---|---|---|
| Eu/Gd | 0.276 | 0.422 | 0.338 |
| Gd/Tb | 0.321 | 0.546 | 0.442 |
| Pr/Nd | 0.422 | 0.420 | 0.408 |
| non-DGA (donor rule) | 0.386 | 0.431 | 0.398 |
| both ends hidden | 0.378 | 0.416 | 0.386 |

## Checks (all passed)

- **Task identity:** 229 / 1,802 per seed; rows and targets bit-identical to round 3.
- **Reproductions:** round 3's B0, R14 and R02 are bit-identical on both designs.
- **Isolation tables:** zero required overlaps in outer folds and every inner block. BP crosses copy groups as expected.
- **Cache poisoning:** over every non-training cell, 50 / 50 folds identical.
- **Fit poisoning:** the whole REF chain, 50 / 50 folds bit-identical.
- **Prediction intervention:** every method rebuilt on a scrambled bench, 108,120 predictions per design, max change 0.0. Two positive controls are caught in the tests.
- **Per-fold identity:** B1 = R14 exactly wherever λ is finite and the fold has no orphans. No fold chose λ = ∞. The one orphan fold now trains on 183 / 183 C3 training cells instead of 35.

## What this means

1. **B1 is the reference to beat.**
   - Point gain over interpolation: +0.034 R² and 0.005 balanced MAE, in 5 / 5 seeds on both designs. The gain sits at Eu/Gd and Gd/Tb.
   - The bootstrap intervals exclude zero, but the small-cluster jackknife-t interval includes it. With about 2–4 effective clusters, even this replicated, round-3-confirmed gain does not pass the sealed confirmation rule on these data. That is a statement about statistical power here, not about the effect.
2. **The fold-fitted Gd amplitude adds nothing over the fixed historical constant** (HIST_D7 − B1 −0.001). The learned part of B1 is one number, and its value was already known from the full corpus.
3. **B2, the broader universal series / read-set prior, is worse than B1** (−0.022 BPG, −0.027 BP, 0 / 5 seeds).
   - **Cause:** shrinking 14 potentials and 6 geometry terms under one λ dilutes the single real effect, the Gd dip. B2 recovers only part of B1's Eu/Gd and Gd/Tb gain, and inner CV still chose finite λ in 19 / 25 (BPG) and 20 / 25 (BP) folds. So inner selection did not protect against this.
   - **Status:** B2 is frozen as defined. It is not tuned after this result, because any new universal-prior variant would be a counted stage-1 slot.
   - **Consequence:** for stage 1, B1 is the stronger of the two series-prior references. A cell-conditioned chemistry claim must beat B1 (and B2), and its falsification controls must keep B1's series prior.
4. **0.50 is not reached** by any reference. B1 is 0.452, with a source-cluster interval of [0.394, 0.497].

## Next (stage 1, not started)

The stage-1 amendment seals each stage block's slot designs before any slot is scored. It must list every result above as seen. Order (brief §7):
1. data provenance (V3 candidates, C5BTBP floor, DMDPhPDA source);
2. residual / electronic slots S01–S04 and their controls;
3. geometry (gate G-geo);
4. transfer (gate G-transfer).

Power simulation comes before any confirmation design. The section-1 finding above already indicates that confirmation on this corpus is underpowered.
