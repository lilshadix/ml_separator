# gen11 §14 — negative transfer and per-metal contribution

Generated 2026-08-22T19:26:48Z.  Every number here is computed by
`scripts/gen11_negative_transfer.py`; none is carried over from another run.

## Headline, reported before any ablation

Full auxiliary pool against the design-matched control, 1 seed(s):

| quantity | control | full pool | delta (control - full) |
| --- | --- | --- | --- |
| macro MAE | 1.0025 | 0.9351 | +0.0675 |
| level MAE | 0.8393 | 0.7586 | +0.0807 |
| shape MAE | 0.4793 | 0.4740 | +0.0053 |

Chemotypes improved 46 / worsened 33; ligands improved 90 / worsened 62.  A positive delta means the auxiliary pool helped.

## Sign convention

    contribution(g) = macro_MAE(pool without g) - macro_MAE(full pool)

    contribution > 0  ->  removing g made prediction worse   ->  g HELPED
    contribution < 0  ->  removing g made prediction better  ->  g HURT  (negative transfer)

For the composition arms in `metal_group_ablations.csv` the convention is the
usual one, `reference - candidate`, so a positive `macro_mae_delta` means the
candidate arm is better than its reference.

## What "removing a metal" does

`HIERARCHICAL` weighting at `aux_lambda = 1` sets the auxiliary block's *total*
sample-weight mass to the lanthanide core's, whatever the block contains.  A LOMO
arm therefore does not train on less auxiliary influence — it redistributes the
removed metal's mass over the metals that remain.  `contribution(g)` answers
**"is g worth its place in the pool"**, not "is more auxiliary data better"; the
latter is the headline above.  `weight_share_audit.csv` checks that the
renormalisation actually held in every fitted arm (3 of 3 arms at the expected 0.500); if it
did not, the contributions are not comparable across arms and nothing below
should be read.

## The baseline

Auxiliary arms cannot run at the frozen design: `FROZEN_3` leaves 98.4 % of
auxiliary rows without a metal coordinate and `FROZEN_8`'s annotation-coupled
MASSACTION terms become an "is this an auxiliary row" flag.  The comparison
baseline is therefore the control at the *same* design corner,
`A_GEN10_CONTROL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`.  The bit-for-bit gen10 anchor `A_GEN10_CONTROL|JOINT|FROZEN_3|FROZEN_8|HIERARCHICAL|lam1` appears in
`metal_group_ablations.csv` under `source=design_control`, which is the design
change measured on its own, with no auxiliary rows in either arm.  Read it first;
it must be subtracted before any auxiliary arm means anything.

## What was reduced

The leave-one-metal-out sweep refits the whole model once per group, and the cost
does not depend on the group's size — dropping 22 cells costs what dropping 1,841
costs.  It was run at **1 split seed(s)** rather than five, because
the machine was simultaneously running the primary arm sweep.  That buys the
per-metal ranking and the chemotype-blocked bootstrap inside those seeds.  It
**cannot** show split-seed direction consistency, which is the §17 requirement,
so `n_seeds_group_helps` / `n_seeds_group_hurts` in
`per_metal_contribution.csv` are bounded by that seed count and must not be read
as five-seed agreement.  Everything derived from the pre-registered composition
arms keeps all five seeds; each row carries its own `n_seeds`.

Pairing a one-seed LOMO arm against a five-seed reference is only legitimate if
one seed of a five-seed run is bit-identical to a one-seed run of the same arm.
That was measured, not assumed, by refitting the frozen anchor at a single seed
and differencing: status **OK**, max |delta| 1.33e-15 over 5248 rows — the thread-order floor, i.e. identical.

## Noise floor

Removing a group that is a fraction of a percent of the pool cannot change the
chemistry, so the spread of those deltas is the sweep's own refit noise.
_not yet computable_.  A per-metal contribution below that magnitude is not a
measurement.

## Size control

The pool is 90 % actinide and its largest metal is a third of it, so "removing
that metal hurts" and "removing a third of the training rows hurts" are the same
sentence until a random removal of the same n is run.  The random arms
(`kind=random_matched`) are that control; they are deliberately **not** folded
into the noise floor, because removing a third of the pool is a real
intervention.

_no size-matched random removal on disk yet_

## Per-metal contribution

| group | kind | n_cells | pool_share | contribution_macro_mae | ci95_low | ci95_high | contribution_level_mae | contribution_shape_mae | chemotypes_group_helps | chemotypes_group_hurts | direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Am | metal | 1664 | 0.3265 | 0.1481 | 0.0326 | 0.2480 | 0.1804 | 0.0013 | 45 | 34 | HELPS |

Full table: `per_metal_contribution.csv`.

## Metal-group ablations

| source | label | n_seeds | macro_mae_delta | ci95_low | ci95_high | level_mae_delta | shape_mae_delta | units_improved | units_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| preregistered_arm | B_LN_EXPANDED | 5 | -0.0037 | -0.0182 | 0.0158 | -0.0004 | 0.0043 | 57 | 131 |
| preregistered_arm | E_LN_PLUS_ALL | 1 | 0.0675 | 0.0084 | 0.1192 | 0.0807 | 0.0053 | 79 | 131 |
| design_control | metal_scheme_only | 5 | 0.0003 | -0.0037 | 0.0056 | -0.0020 | 0.0004 | 65 | 131 |
| design_control | massaction_only | 5 | -0.0086 | -0.0149 | -0.0038 | -0.0089 | -0.0010 | 52 | 131 |
| design_control | both | 5 | -0.0038 | -0.0104 | 0.0015 | -0.0027 | -0.0014 | 65 | 131 |
| lomo_group | drop_all_actinides | 1 | 0.1205 | 0.0431 | 0.2147 | 0.1551 | -0.0025 | 76 | 131 |
| lomo_within_actinide | drop_Am | 1 | 0.1481 | 0.0326 | 0.2480 | 0.1804 | 0.0013 | 77 | 131 |

Full table: `metal_group_ablations.csv`.  **The full-pool result and every
ablation are reported side by side.**  §14 permits excluding a harmful metal
group only as a pre-declared follow-up ablation; no row here may be substituted
for the headline.

## Level versus shape

| comparison | n_seeds | macro_mae_delta | level_mae_delta | shape_mae_delta | chemotypes_improved | chemotypes_worsened | ligands_improved | ligands_worsened |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HEADLINE_full_pool_vs_control | 1 | 0.0675 | 0.0807 | 0.0053 | 46 | 33 | 90 | 62 |
| ARM_B_LN_EXPANDED_vs_control | 5 | -0.0037 | -0.0004 | 0.0043 | 35 | 44 | 68 | 84 |
| LOMO_Am | 1 | 0.1481 | 0.1804 | 0.0013 | 45 | 34 | 89 | 63 |
| LOMO_GROUP_all_actinide | 1 | 0.1205 | 0.1551 | -0.0025 | 45 | 34 | 89 | 63 |

Full table: `level_vs_shape.csv`.  The level is the per-(seed, ligand) mean
residual and the shape the deviation about it; gen7-gen10 showed they move
independently, so a macro gain that is entirely level and a macro gain that is
entirely shape are different results.

## Group definition

A metal gets its own leave-one-out arm at >= 100 pool
cells; below that it joins its `metal_category` remainder.  The rule is declared
in `gen11/negative.py` before the sweep runs, and the groups partition the pool
exactly (checked in `metal_groups`).

`n_cells` counts **pool cells**, i.e. after `pools.prepare_auxiliary` averages
replicates onto one row per (extractant, condition, metal) — the same
granularity a cohort training row has.  It is therefore smaller than the archive
record counts the brief quotes (Am 1841, U 995, Th 761, Pu 709, Np 432, Cm 136):
5,438 records reduce to 5,096 cells.  `GROUP_all_actinide` is a union of
partition members, not a member; it appears in the ablation table and is
deliberately absent from `per_metal_contribution.csv`, whose rows must be
disjoint.

| group | kind | metal_category | metals | n_cells | pool_share |
| --- | --- | --- | --- | --- | --- |
| Am | metal | actinide | Am | 1664 | 0.3265 |
| U | metal | actinide | U | 995 | 0.1953 |
| Th | metal | actinide | Th | 724 | 0.1421 |
| Pu | metal | actinide | Pu | 679 | 0.1332 |
| Np | metal | actinide | Np | 403 | 0.0791 |
| Sr | metal | alkaline_earth | Sr | 167 | 0.0328 |
| Cm | metal | actinide | Cm | 120 | 0.0235 |
| other_transition_metal | category_remainder | transition_metal | Cr|Fe|Hf|Mo|Pd|Ru|Tc|Zr | 109 | 0.0214 |
| other_alkaline_earth | category_remainder | alkaline_earth | Ba|Ca | 66 | 0.0130 |
| other_rare_earth_non_lanthanide | category_remainder | rare_earth_non_lanthanide | Sc|Y | 57 | 0.0112 |
| other_lanthanide | category_remainder | lanthanide | Eu|Gd|Nd|Pm|Sm | 46 | 0.0090 |
| other_post_transition_metal | category_remainder | post_transition_metal | Bi|Cd|In|Pb | 44 | 0.0086 |
| other_actinide | category_remainder | actinide | Cf|Pa | 22 | 0.0043 |
| GROUP_all_actinide | category | actinide | Am|Cf|Cm|Np|Pa|Pu|Th|U | 4607 | 0.9040 |

## Files

| file | what |
| --- | --- |
| `per_metal_contribution.csv` | leave-one-metal-out contribution per group, with blocked bootstrap CI |
| `metal_group_ablations.csv` | pre-registered composition arms, the design-control grid, and the group-level LOMO |
| `chemotype_deltas.csv` | per-chemotype delta for every comparison, ranked worst first |
| `worst_regressions.csv` | per-ligand deltas; all ligands for the headline, worst 15 elsewhere |
| `level_vs_shape.csv` | the level/shape split of every comparison |
| `metal_groups.csv` | the partition |
| `weight_share_audit.csv` | auxiliary weight share per arm; it must be lam/(1+lam) everywhere or the arms are not comparable |
| `seed_slice_equivalence.json` | proof that a one-seed run is a slice of a five-seed run |
| `negative_summary.json` | machine-readable summary, including which arms were absent |

## Arms absent when this ran

- `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
- `D_LN_PLUS_NON_ACTINIDE|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_Cm`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_Np`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_Pu`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_Sr`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_Th`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_U`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_other_actinide`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_other_alkaline_earth`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_other_lanthanide`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_other_post_transition_metal`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_other_rare_earth_non_lanthanide`
- `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|LOMO_other_transition_metal`
