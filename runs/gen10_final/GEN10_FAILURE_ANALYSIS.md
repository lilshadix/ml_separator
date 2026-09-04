# GEN10 failure analysis — every arm that did not work, and why

*Losing arms are reported with the same care as winners. Each is diagnosed against
the same questions: did it fail to move shape; did it move shape but hurt level;
did it help one axis and hurt another; was the hypothesis wrong or the
implementation. Every number is five seeds on the FROZEN cohort unless marked
**development**. Sources are named per section.*

---

## 1. Feature access — the recomposition is not a split-proposal artefact (Phase 2A/2B)

gen9 left open whether its post-hoc recomposition was an architectural need or an
artefact of five context columns being lost among 2,160 (a random split subset at
`max_features = 0.3` reaches one of them about 1 % of the time). Eight arms, same
forest, same folds, same design, only the access changed
(`feature_access/leaderboard.csv`, `feature_access/shape/shape_by_axis.csv`):

| arm | macro MAE | extractant slope (true 2.574) | extractant shape MAE | span recovery |
|---|---|---|---|---|
| `REC_ecfp_plus_recovered` (frozen) | 0.9807 | 0.116 | 0.665 | 0.051 |
| `GEN9_REL_MONOLITH` (max_features 0.30) | 0.9858 | 0.521 | 0.567 | 0.210 |
| `MF_025` | 0.9852 | 0.487 | 0.576 | 0.204 |
| `MF_050` | 0.9909 | 0.517 | 0.568 | 0.219 |
| `MF_100` | 1.0710 | 0.384 | 0.591 | 0.202 |
| `MF_SQRT` | 1.0566 | 0.292 | 0.631 | 0.119 |
| `REP_8` (context columns ×8) | 0.9904 | — | — | — |
| `REP_32` (context columns ×32) | 0.9943 | — | — | — |
| `GEN9_SHAPE_RECOMPOSED` | **0.9695** | **1.024** | **0.469** | **0.423** |

Giving the tree every chance to split on the five columns changes neither level
nor shape beyond seed noise; `max_features = 1.0` and `sqrt` are significantly
worse for the usual reasons (an under-randomised and an over-randomised forest).
The recomposition's gain — 0.016 macro and half the slope error — is therefore
something the monolith cannot reach by access alone. **gen9 §7.2 is closed: the
recomposition is an architecture.**

## 2. Explicit level + zero-mean shape — reproduces the shape, pays on the level (Phase 2D)

`GEN10_LEVEL_SHAPE` trains a level forest on curve-mean targets and a shape
forest constrained to sum to zero over each requested curve, with no post-hoc
step (`level_shape/leaderboard.csv`, `level_shape/shape/`):

| arm | macro | offset MAE | extractant slope | shape MAE | span | Spearman |
|---|---|---|---|---|---|---|
| `GEN9_SHAPE_RECOMPOSED` | **0.9695** | **0.8212** | 1.024 | **0.469** | 0.423 | 0.886 |
| `GEN10_LEVEL_SHAPE` | 0.9818 | 0.8300 | 1.036 | 0.477 | 0.430 | 0.857 |

The shape is reproduced to within noise on every endpoint. The 0.012 it loses is
**entirely level**: offset MAE 0.830 against the recomposition's 0.821, which is
bit-identical to the frozen monolith's. Predicting a curve's level *directly* is a
worse level estimate than averaging the monolith's row-wise predictions over the
curve. This is gen7's "the level is not chemistry" and the memory note "an
explicit level head is worse", reproduced a third time with a cleaner design. The
hypothesis — that the recomposition is a level model plus a shape model — is
right; the implementation that makes the level explicit is the part that loses.

## 3. Additive residual correction — works only cross-fitted, and then not enough (Phase 2C)

`GEN10_RESIDUAL_SHAPE` keeps the monolith's shape and adds a learned,
curve-centred correction. Two versions, because the first one built was a trap:

| arm | macro | extractant slope | shape MAE | correction-target sd |
|---|---|---|---|---|
| in-sample residual (trap control) | 0.9748 | 0.347 | 0.621 | 0.198 |
| cross-fitted residual (inner chemotype folds) | 0.9821 | 0.737 | 0.523 | 0.735 |

An ExtraTrees forest with `min_samples_leaf = 2` nearly interpolates its training
rows, so its in-sample residual has a quarter of the spread of its held-out
residual and the second stage has almost nothing to learn: the arm collapses
toward the frozen model (its slope 0.35 is between frozen 0.12 and REL_MONOLITH
0.52). The cross-fitted version recovers twice the slope but loses 0.013 macro to
the recomposition. Replacing the monolith's within-curve shape beats correcting
it.

## 4. Two-branch model — a context branch that cannot see the chemistry hurts (Phase 2C)

`GEN10_TWO_BRANCH` adds a second forest that reads only the condition blocks and
the design-relative columns (no fingerprint, no recovered variables), fitted on
the cross-fitted curve-centred residual and re-centred per curve.

| arm | macro | extractant slope | shape MAE |
|---|---|---|---|
| `TWO_BRANCH` | 1.0030 | 0.547 | 0.553 |
| `TWO_BRANCH_INTERACT` (+ base prediction as a feature) | 0.9964 | 0.535 | 0.561 |

Both are worse than the frozen model on macro (0.9807). The within-curve response
*depends on the chemistry* — which ligand, which diluent — and a branch denied the
chemistry fits an average response that is wrong for most ligands. The one
interaction form (handing the branch the static prediction) recovers a third of
the damage and no more. Hypothesis wrong, not implementation: the premise that
the context could be modelled additively and separately from the chemistry is
what failed.

## 5. Learned set representation — matches the handcrafted shape, loses macro (Phase 3)

Contingent stage, run because Phase 1's smoke showed strong query-set dependence
and Phase 2 produced no better architecture. A DeepSets head (mean pooling) and a
one-block attention head replace the shape forest; the level is the monolith's
(`set_context/leaderboard.csv`, `set_context/shape/`):

| arm | macro | offset | extractant shape MAE | slope | acid shape MAE | lanthanide shape MAE |
|---|---|---|---|---|---|---|
| `GEN9_SHAPE_RECOMPOSED` | **0.9695** | 0.8212 | 0.469 | 1.024 | **0.545** | **0.291** |
| `GEN10_SET_MEAN` | 1.0059 | 0.8212 | **0.459** | 1.011 | 0.568 | 0.306 |
| `GEN10_SET_ATTENTION` | 1.0087 | 0.8212 | **0.452** | 1.066 | 0.556 | 0.305 |

With the level held identical, the head costs 0.036–0.039 macro: the macro
metric weights small ECFP clusters equally, and the MLP's shape is wrong on
exactly the ligands a forest handles by falling back to the nearest leaf. It
matches the recomposition's extractant shape and is worse on the other two axes.
Two defects in the first version are recorded: the head overfit badly without
early stopping (training loss 0.66 → 0.05 on a target of sd 0.7; fold-0 pooled
1.169 against frozen 1.138), and an inner chemotype hold-out to choose the epoch
count (the one fitted knob) repaired that to 1.134 — still behind the
recomposition's 1.098. **Fails the brief's bar on macro; not included.**

## 6. Representations that lost (Phase 4)

| arm | macro | extractant slope | shape MAE | note |
|---|---|---|---|---|
| `REC_LOCAL` (spacing-only, no rank, no endpoint) | 0.9750 | 0.297 | 0.614 | local gaps alone carry no position; worse than gen9's columns |
| `REC_GEN9_CURVEWINDOW` (gen9 columns, whole-curve window) | 0.9700 | 1.004 | 0.472 | indistinguishable from gen9's primary-curve window |
| `MONO_HYBRID` / `MONO_RANK` (monolith, not recomposed) | 0.9905 / 0.9935 | 0.845 / 0.759 | 0.496 / 0.513 | better representations help the monolith too, but the recomposition still doubles the gain |

The winners of Phase 4 (`REC_HYBRID`, `REC_ALL`, `REC_RANK`) are in the decision
report. Note that `REC_ALL`, which includes the endpoint-class columns, has the
best shape of all; whether it keeps that under a changed query set is the Phase 1
question on the finalists.

## 7. Series-local priors that lost (Phase 5)

Five seeds × 12 repeats, central-then-spread, on `GEN9_SHAPE_RECOMPOSED`
(`adaptation/summary.csv`, `adaptation/bootstrap_vs_offset_k3.csv`; positive =
better than `OFFSET_K3`):

| adapter | k = 2 | k = 3 | k = 5 | vs `OFFSET_K3` at k = 3 (CI) |
|---|---|---|---|---|
| `OFFSET_K3` (gen8, ridge 4.0) | 0.5447 | 0.4915 | 0.4789 | — |
| `SERIES_MAP` (gen9, old estimator) | 0.5915 | 0.5660 | 0.5572 | −0.075 [−0.103, −0.052] |
| `SERIES_FIXED` (gen9 design, 4.0 everywhere) | 0.5425 | 0.4868 | 0.4643 | +0.005 [0.000, 0.009] |
| `SERIES_INNER` (grid selected per k on training ligands) | 0.5350 | 0.5231 | 0.4651 | −0.032 [−0.052, −0.018] |
| **`SERIES_ML`** (marginal likelihood) | **0.5193** | **0.4609** | **0.4405** | **+0.031 [0.020, 0.044]** |

*gen9's estimator, rediagnosed.* `SERIES_MAP` reproduces gen9's numbers to three
decimals. Its prior puts the **series** family at the penalty *floor* (0.25) —
near-free series deviations — while the response family sits near 0.8. gen9's
"over-shrinkage" diagnosis was therefore half right at best: on these folds the
series deviations were barely shrunk, and with two measured points and several
series per ligand, near-free series columns overfit. The marginal-likelihood
estimator puts series at ~0.3 and response at ~1.9 and wins at every k.

*Why `INNER` failed.* It simulates the k-shot protocol on training ligands with a
random draw from the pool rather than the deployed central-then-spread sequence,
so it selects penalties for a different design; at k = 3 it chooses a response
penalty of 0.5 and loses 0.03. A version that replays the deployed policy would be
a reasonable repair; it was not attempted, because `ML` already clears the bar and
the brief forbids continuing to tune this branch.

*`FIXED` is a wash* (+0.002 to +0.015): adding series columns at the same penalty
as the response columns neither helps nor hurts.

## 8. Learned first-point acquisition on the realised objective (Phase 6)

Trained on realised one-shot MAE rather than the surrogate `|r − median r|`, on
`GEN9_SHAPE_RECOMPOSED`, five seeds × 8 repeats, 143 ligands
(`acquisition/realised_summary.csv`, `acquisition/realised_bootstrap.csv`):

| policy | one-shot macro MAE | vs MEDOID (CI) | seeds |
|---|---|---|---|
| `ORACLE` | 0.4628 | | |
| `MEDOID` | **0.6286** | — | |
| `CENTRAL` | 0.6317 | | |
| `LEARNED_BLEND[geometry]` | 0.6342 | −0.006 [−0.016, +0.007] | 0/5 |
| `LEARNED_SCALAR[geometry]` | 0.6404 | −0.012 [−0.033, +0.020] | 1/5 |
| `LEARNED_RANK[geometry]` | 0.6478 | −0.019 [−0.038, +0.005] | 1/5 |
| `RANDOM` | 0.6917 | | |

The decisive diagnostic is the blend's own α, chosen on inner held-out ligands
the ranker never saw: **α = 1.0 — pure centrality — in 24 of 25 folds.** Given
the chance to depart from MEDOID by any amount, the honest selection rule
declines. Changing the label from surrogate to realised regret moved nothing; the
part of the optimal first point that is predictable before measuring is the part
centrality already captures. **The direction is closed; MEDOID / central-then-
spread is shipped.**

## 9. Cross-cutting

**Every architectural alternative to the recomposition lost, and all of them lost
on the level.** Level+shape, residual correction, two-branch, the set encoder —
each reproduces or approaches the recomposition's *shape* and pays in macro MAE,
and in every case the mechanism is the same: the monolith's row-wise prediction,
averaged over a curve, is the best level estimate this corpus supports, and
anything that replaces or perturbs it loses more than the shape buys. The
recomposition wins because it is the *only* design that touches the shape and
leaves the level bit-identical.

**The representation, not the architecture, was where the headroom was.**
Changing five endpoint-based columns to rank- and spacing-based ones (Phase 4)
bought more shape than any architecture in Phase 2 or 3, at zero macro cost.

**Two of gen10's own defects were the gen9 issue-6 mechanism in new clothes.** An
unrounded parallel forest feeding a second model's target made the residual arm
irreproducible at 3.4e-2; an in-sample residual made it vacuous. Both were caught
by the same checks the brief asks for — refit twice, compare — and both are now
tested.
