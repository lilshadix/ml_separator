# gen7 decision report

*Cohort fingerprint `bed178ec1a7a82b0`. Reference reproduction verified exact:
`MC_lig2d_ext_massaction` returns macro 1.067993 / offset 0.936996 / shape 0.502044 on
split seed 104729, matching `runs/gen6_expA_5seed/arm_metrics.csv`.*

## The question gen7 was asked

*How much of the remaining held-out-chemotype error can be removed by a better
representation, objective, decomposition, learner or inductive bias, without collecting
new measurements?*

## The answer

**A little — 0.069 macro MAE, all of it from information handling rather than model
capacity — and the reason the rest cannot be removed is now measured rather than
suspected.**

The decisive number is a control nobody had run: a model given metal and conditions but
**no ligand information at all** scores macro MAE 1.088, against 0.998 for the best
model in the repository. Everything chemistry contributes is 0.090, and a paired
per-ligand test splits that cleanly — the part landing on the *level* is
indistinguishable from zero (+0.044, t = 0.81, p = 0.42, n = 152; it improves 51 % of
ligands, a coin flip), while only the part landing on the *shape* is real (+0.057,
t = 2.75, p = 0.007).

An oracle handed each test ligand's true level scores 0.497, so the level is worth 0.59
macro MAE — six times what chemistry currently buys. Isolating that level as its own
~120-example regression and asking what predicts it gives the result the generation
turns on: against the **condition-adjusted** level, no model of any kind beats a
1-nearest-neighbour Tanimoto lookup (0.9902 best vs 0.9635 for the lookup, 6.4 % of the
spread captured). The 29 % apparent capture on the *raw* ligand mean is the model
learning which conditions each ligand happened to be measured under.

## Decisions taken

**One caveat governs the whole table.** Under the project's own paired bootstrap —
scoring per ECFP cluster, resampling the Tanimoto chemotype the folds held out —
**only the missing-value indicators have an interval excluding zero**
(+0.061, 95 % CI [+0.029, +0.091], BCa agrees, 82/131 units improved). Every other
"Adopt" below is directionally consistent on 3/3 seeds with an interval that straddles
zero. They are worth shipping because the direction never reverses and the cost is nil;
they are not established effects and must not be quoted as such. The same applies to the
whole contribution of ligand chemistry (+0.097, CI [−0.016, +0.207]) and to the
ensemble's edge over its best member (+0.024, CI [−0.013, +0.075]).

| decision | evidence |
|---|---|
| **Adopt** explicit structural-absence flags in the descriptor build — *the one established effect* | worth 0.061, CI [+0.029, +0.091] — more than the whole 206-column descriptor block — and they encode "has no amide / has no alkyl chain", a 4-level class explaining 16.9 % of ligand-level variance |
| **Adopt** the recovered experimental variables | +0.035 to +0.056 macro MAE; 83/83 solvent strings parse; 305 rows rescued from `diluent__other` |
| **Adopt** the structured metal representation | 0.9932 vs 0.9976 (monotone continuous) vs 1.0078 (one-hot) |
| **Adopt** the ligand–diluent coupling terms | +0.021 on the donor blocks; nothing else in the bundle is a function of both ligand and diluent |
| **Reject** pretrained molecular encoders | 16 arms; best 1.097; raw MoLFormer 1.245, worse than no ligand at all; independently confirmed on the level sub-problem |
| **Reject** kernel methods and GPs | 1.198–1.372; behind the trees and behind the no-ligand floor |
| **Reject** the hierarchical offset/shape network | 1.30 pooled vs the tree's 1.19, at 38× the cost |
| **Reject** alternative learners | 13 families; ExtraTrees wins; 3 modern boosters miss the no-ligand floor |
| **Adopt** a non-negative stack, with tempered expectations | 0.9786 vs 0.9909 for the best member — but the members' residuals correlate 0.965–1.000, so there is almost no diversity to exploit |
| **Reject** 3D | last of 11 representations; and the bundle has one pose per complex, so an equivariant model has nothing to average over |
| **Quarantine** 140 DMDPhPDA rows | paired differences are exactly the integers 0–4 decades, tracking acid concentration |
| **Flag** 414 unmodelled-second-species rows | 20 aqueous complexants recorded against TODGA's SMILES |

## Where the remaining error actually sits

The twenty worst held-out ligands hold 30.7 % of the total error, and their mechanism
table says two of the three routes there are measurement problems rather than modelling
problems:

* `LEVEL_ONLY` (43.8 % of their error) — close training relatives existed and the level
  was still wrong;
* `SINGLE_BATCH` (32.3 %) — the ligand's level and its one study's calibration are not
  separable even in principle;
* `SPARSE` (14.8 %) — fewer than five rows;
* `TRUE_EXTRAPOLATION` — **zero** of the worst twenty. The worst failures have median
  nearest-neighbour Tanimoto 0.534 against 0.611 for all held-out ligands: barely more
  distant than average.

The single worst is a bis(dithiophosphonate) — soft sulfur donors among hard-oxygen
diglycolamides — with MAE 4.20 of which 4.20 is offset and 0.44 is shape, at Tanimoto
0.667. The model gets its response to metal and acid nearly right and places it four
decades from where it sits.

## What must change in how this project reports results

1. **`NULL_metal_cond` goes in every leaderboard.** Six generations optimised inside a
   0.09-wide band without a control that says how wide the band is.
2. **`library_versions` is stamped into every run.** `pip install tabpfn` silently
   downgraded scikit-learn 1.9.0 → 1.6.1 and pandas 3.0.5 → 2.3.3 mid-session, moving
   the reference arm by 0.0014 — small absolutely, fatal against effects of 0.004–0.02.
   Two runs whose versions differ must not share a leaderboard.
3. **Model selection is part of the model.** The gen6 report's champion was not gen6's
   best arm; quoting a better one post hoc is selection on the test set. `SELECT_inner`
   chooses on inner chemotype splits instead.
4. **Report the level and the shape separately, always.** They have different sizes,
   different causes and different cures, and their sum hides both.

## The deployable recommendation

**For a new extractant, measure it once.** Per-ligand MAE falls from 1.020 zero-shot to
**0.730** with a single measurement and **0.636** with two — a larger gain than
everything gen2–gen7 achieved by modelling. And `offset_only` beats the no-model null at
every k (0.730 vs 0.860 at k=1), so the model does earn its place here: it supplies the
shape, and the measurement supplies the level. That is the level/shape decomposition
cashed out, and it reverses gen5's k-shot conclusion on the pair task for a reason the
decomposition predicts. Use the fixed-slope offset correction, not an affine fit —
affine is unusable below k≈5 (5.39 at k=2).

## What to measure next

Ranked, and the first is not "more data":

1. Ship the structural-absence flags and the recovered variables into the bundle build.
2. **Chase the within-series residual, not the study offset.** gen7 expected the level
   to be hidden per-study calibration and tested it properly: under a permutation null
   the within-ligand between-batch offset is **not detectable above chance** (median
   excess −0.011; 42.5 % of ligands positive against 50 % by chance), and the
   variance-components estimate σ_batch = 0.216 implies an `offset_mae` floor of only
   **0.133** against 0.84 observed. The batch hypothesis is rejected. What
   `ORACLE_batch` actually buys (0.147) is series structure — 82 % of batches contain
   exactly one experimental series — so a quarter of the "irreducible" response scatter
   is a titration curve the model is not tracing smoothly. That is a modelling target.
3. Then more chemotypes, chosen by max-min Tanimoto or level uncertainty (gen6 Exp. F).
