# Targeted search for the Pr/Nd spec cells (exploratory)

*regime: Pr/Nd case, PC88A and Cyanex 272 literature placeholder entries; system sys_29976921e156a0a0; PLACEHOLDER_PARAMETERS (median draw); EXPLORATORY, not pre-registered*

A structured grid of 903 converged regimes (of 1620 tried), against the random LHS of `g18_case_prnd.py` which reached no cell in 64 draws.

| purity_min | recovery_min | Fenske N_min (total reflux, SF 1.4) | reached here | best purity at that recovery | stages of the best |
|---|---|---|---|---|---|
| 0.95 | 0.8 | 9.9 | **no** (0 regimes) | 0.8142 | - |
| 0.97 | 0.85 | 12.5 | **no** (0 regimes) | 0.8142 | - |
| 0.99 | 0.9 | 17.2 | **no** (0 regimes) | 0.8142 | - |

## Verdict

No regime of the structured grid meets any cell either, but the grid locates *why*, and it is not what the random search suggested.

**Saponification is decisive, as the acid balance predicts.** A cation-exchange extractant releases 3 H+ per Ln3+, so an unsaponified 0.1 M feed self-acidifies and extraction stalls; the grid shows exactly that, and shows it lifting:

| saponification_degree | max_recovery | max_purity | n |
|---|---|---|---|
| 0 | 0.6672 | 0.999 | 312 |
| 0.2 | 1 | 0.9991 | 251 |
| 0.35 | 1 | 0.9989 | 201 |
| 0.5 | 1 | 0.9989 | 74 |
| 0.65 | 1 | 0.8177 | 65 |


So **recovery near 1 is reachable** (from 0.667 unsaponified to 1.000 saponified) and **purity near 0.999 is reachable** -- but not at the same time: the best purity anywhere on the grid at recovery >= 0.80 is **0.8142**, short of the 0.95 the loosest cell needs. Over-saponification closes it from the other side (at 0.65 the organic takes everything and purity falls to 0.818).

**What this does and does not establish.** It is *not* evidence that the `log_k` windows are broken -- both ends of the trade-off are reachable with them. It is evidence that the purity-recovery frontier of *this circuit shape* stops below the cell. The Fenske minimum (about 10 stages) is a **total-reflux** bound; at finite reflux a SF-1.4 separation needs both more stages and the right internal reflux, and the variables that set it -- feed stage, scrub-return stage, per-section O/A -- were held at their defaults here. Whether tuning those closes the gap, or whether the placeholder parameters are simply too far from real PC88A, is unresolved and needs the transcribed values (open item U4).
