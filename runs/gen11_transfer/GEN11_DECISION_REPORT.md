# GEN11 decision report

_Rendered from `runs/gen11_transfer/headline_tables/` at 2026-08-22T19:24:09+00:00. Every number below is read back from one of those CSVs, which are themselves recomputed from the raw OOF predictions in `runs/gen11_transfer/arms`. Nothing here is transcribed._

## 0. What was available when this report was rendered

* arms with stored predictions: **5** of **34** pre-registered
* design-matched baseline: `A_GEN10_CONTROL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* frozen gen10 anchor: `A_GEN10_CONTROL|JOINT|FROZEN_3|FROZEN_8|HIERARCHICAL|lam1`
* seeds per arm: [104729, 130363, 155921, 196613, 262147]
* cohort fingerprint asserted by the runner: `bed178ec1a7a82b0`
* the frozen anchor recomputed here scores macro MAE **0.969484543735** against gen10's own re-derivation from gen10's raw predictions, **0.969484543735** (|Δ| = 0) — the anchor is the same model, not a look-alike

Pre-registered arms **not** on disk when this ran (the primary run was still in progress, or the stage has not been launched):

* `C_LN_PLUS_ACTINIDES|AUX_PRETRAIN|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|CHEMOTYPE|lam1`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam0.25`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam0.5`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|PERMUTED_AUX_TARGET`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|PUBLICATION_BLOCKED`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|SHUFFLED_METAL_LABELS`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam2`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|METAL_BALANCED|lam1`
* `C_LN_PLUS_ACTINIDES|JOINT|GENERAL|ANNOTATION_SAFE|ROW|lam1`
* `C_LN_PLUS_ACTINIDES|SHARED_LEVEL|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* `C_LN_PLUS_ACTINIDES|SHARED_SHAPE|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* `D_LN_PLUS_NON_ACTINIDE|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* `E_LN_PLUS_ALL|AUX_PRETRAIN|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* `E_LN_PLUS_ALL|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|PUBLICATION_BLOCKED`
* `E_LN_PLUS_ALL|SHARED_LEVEL|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* `E_LN_PLUS_ALL|SHARED_SHAPE|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`
* `F_ACTINIDES_ONLY_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw11`
* `F_ACTINIDES_ONLY_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw22`
* `F_ACTINIDES_ONLY_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw33`
* `F_ACTINIDES_ONLY_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw44`
* `F_ACTINIDES_ONLY_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw55`
* `G_RANDOM_AUX_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw11`
* `G_RANDOM_AUX_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw22`
* `G_RANDOM_AUX_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw33`
* `G_RANDOM_AUX_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw44`
* `G_RANDOM_AUX_MATCHED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1|draw55`

Tables skipped, with the reason (an absent table is recorded, never emitted empty):

* `t5_kshot.csv` — no k-shot artefact under runs/gen11_transfer/; produce it with scripts/gen10_final_locked.py pointed at the gen11 arms
* `t7_controls.csv` — no negative-control arm has been run (scripts/gen11_run.py --stage controls)

## 1. The one framing that must not be dropped

Every auxiliary arm runs at a **changed design corner** (`GENERAL` metal representation + `ANNOTATION_SAFE` MASSACTION), because the frozen design cannot represent auxiliary rows. So `auxiliary arm − frozen gen10` is not the transfer effect; it is the sum of two effects, and every table here splits them:

```
design effect   = macro(control @ FROZEN_3|FROZEN_8) − macro(control @ GENERAL|ANNOTATION_SAFE)
transfer effect = macro(control @ same corner as the arm) − macro(arm)
total           = design effect + transfer effect          (positive = better)
```

Measured design effects (`t2_leaderboard.csv`), each control corner against the frozen anchor:

| model | macro_mae | offset_mae | shape_mae | pooled_mae | n_seeds | design_effect_macro |
|---|---|---|---|---|---|---|
| A_GEN10_CONTROL\|JOINT\|GENERAL\|FROZEN_8\|HIERARCHICAL\|lam1 | 0.9692 | 0.8225 | 0.4815 | 1.1246 | 5 | 0.0003 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | 0.9695 | 0.8212 | 0.4819 | 1.1218 | 5 | 0.0000 |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | 0.9732 | 0.8229 | 0.4833 | 1.1190 | 5 | -0.0038 |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | 0.9781 | 0.8291 | 0.4826 | 1.1239 | 5 | -0.0086 |


The same comparison as a paired, chemotype-blocked bootstrap (`t3_paired_bootstrap.csv`, `comparison_kind = DESIGN`). A design corner whose interval excludes zero is a real cost or benefit that an auxiliary arm inherits before it transfers anything:

| candidate | point_delta | bca_low | bca_high | cluster_robust_low | cluster_robust_high | units_improved | units_total | bca_direction |
|---|---|---|---|---|---|---|---|---|
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | -0.0086 | -0.0156 | -0.0041 | -0.0139 | -0.0033 | 52 | 131 | candidate worse |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | -0.0038 | -0.0109 | 0.0012 | -0.0094 | 0.0019 | 65 | 131 | not separated |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|FROZEN_8\|HIERARCHICAL\|lam1 | 0.0003 | -0.0039 | 0.0053 | -0.0044 | 0.0050 | 65 | 131 | not separated |


## 2. Leaderboard (`t2_leaderboard.csv`; per-seed values in `t2_leaderboard_by_seed.csv`)

| model | arm_kind | design_corner | macro_mae | macro_mae_sd | offset_mae | shape_mae | pooled_mae | n_seeds | transfer_effect_macro | design_effect_macro | total_vs_frozen_macro | n_seeds_improved_vs_baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | AUXILIARY | GENERAL\|ANNOTATION_SAFE | 0.9770 | 0.0242 | 0.8238 | 0.4782 | 1.1166 | 5 | -0.0037 | -0.0038 | -0.0075 | 2 |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | GENERAL\|FROZEN_8 | 0.9692 | 0.0241 | 0.8225 | 0.4815 | 1.1246 | 5 | n/a | 0.0003 | 0.0003 | n/a |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | FROZEN_3\|FROZEN_8 | 0.9695 | 0.0254 | 0.8212 | 0.4819 | 1.1218 | 5 | n/a | 0.0000 | 0.0000 | n/a |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | GENERAL\|ANNOTATION_SAFE | 0.9732 | 0.0301 | 0.8229 | 0.4833 | 1.1190 | 5 | n/a | -0.0038 | -0.0038 | n/a |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | FROZEN_3\|ANNOTATION_SAFE | 0.9781 | 0.0279 | 0.8291 | 0.4826 | 1.1239 | 5 | n/a | -0.0086 | -0.0086 | n/a |


## 3. What each arm is made of (`t1_arm_composition.csv`)

The archive's contribution is metals and curves, not ligands — read any transfer result against these counts.

| arm | aux_rows | aux_structures | aux_metals | aux_new_chemotypes | aux_rows_on_new_chemistry | aux_rows_fingerprint_identical_to_cohort | aux_usable_curves | aux_effective_chemotypes | total_rows | n_oof_variants_run |
|---|---|---|---|---|---|---|---|---|---|---|
| A_GEN10_CONTROL | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | 5938 | 4 |
| B_LN_EXPANDED | 86 | 6 | 5 | 1 | 11 | 75 | 6 | 1.7576 | 6024 | 1 |
| C_LN_PLUS_ACTINIDES | 4896 | 153 | 8 | 37 | 908 | 2779 | 864 | 9.9678 | 10834 | 0 |
| D_LN_PLUS_NON_ACTINIDE | 456 | 7 | 17 | 1 | 13 | 424 | 79 | 1.3519 | 6394 | 0 |
| E_LN_PLUS_ALL | 5438 | 155 | 30 | 38 | 932 | 3278 | 954 | 8.9211 | 11376 | 0 |
| F_ACTINIDES_ONLY_MATCHED | 456 | 81 | 6 | 19 | 82 | 255 | 12 | 8.9514 | 6394 | 0 |
| G_RANDOM_AUX_MATCHED | 456 | 81 | 19 | 20 | 91 | 266 | 13 | 8.6511 | 6394 | 0 |


## 4. Where a difference sits (`t6_decomposition.csv`)

Positive = the candidate is better. `level` is the per-ligand mean residual and `shape` the deviation about it; they move independently.

| candidate | band_scheme | stratum_kind | stratum | n_rows | n_units | abs_improvement | level_improvement | shape_improvement |
|---|---|---|---|---|---|---|---|---|
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | GEN11_ANALYSIS | band | far | 2138 | 30 | -0.0186 | -0.0028 | 0.0087 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | GEN11_ANALYSIS | band | mid | 5367 | 71 | -0.0163 | -0.0004 | -0.0027 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | GEN11_ANALYSIS | band | near | 18735 | 71 | 0.0070 | 0.0015 | 0.0061 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | n/a | ALL | ALL | 26240 | 131 | -0.0037 | -0.0004 | 0.0043 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | GEN10_TERCILE | band | far | 6736 | 87 | -0.0154 | 0.0008 | -0.0023 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | GEN10_TERCILE | band | mid | 3951 | 51 | 0.0075 | 0.0051 | 0.0002 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | GEN10_TERCILE | band | near | 15553 | 38 | 0.0114 | 0.0045 | 0.0132 |


Criterion D needs the far band to be *statistically* better, not merely better on a point estimate, so the far stratum carries its own chemotype-blocked paired bootstrap (`t6_far_band_bootstrap.csv`):

| candidate | band_scheme | n_rows | n_units | point_delta | bca_low | bca_high | units_improved | units_total | status |
|---|---|---|---|---|---|---|---|---|---|
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | GEN11_ANALYSIS | 2138 | 30 | -0.0186 | -0.0607 | 0.0424 | 12 | 30 | OK |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | GEN10_TERCILE | 6736 | 87 | -0.0154 | -0.0410 | 0.0138 | 36 | 87 | OK |


## 5. Curve shape by axis (`t4_shape_by_axis.csv`)

| model | axis_label | n_curves | shape_mae | baseline_shape_mae | shape_improvement_vs_baseline | slope_mae | span_recovery_median |
|---|---|---|---|---|---|---|---|
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | acid | 421 | 0.5398 | 0.5396 | -0.0002 | 1.3360 | 0.4262 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | contact_time | 8 | 0.0774 | 0.0716 | -0.0058 | 0.0044 | n/a |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | extractant | 155 | 0.4675 | 0.4741 | 0.0066 | 1.4585 | 0.4682 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | metal_concentration | 9 | 0.2699 | 0.2691 | -0.0008 | 0.8502 | 0.0151 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | metal_series | 321 | 0.2959 | 0.2927 | -0.0032 | 0.0944 | 0.6349 |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | temperature | 32 | 0.4087 | 0.4086 | -0.0001 | 0.0391 | 0.1579 |


## 6. Few-shot frontier (`t5_kshot.csv`)

_Not produced: no k-shot artefact under runs/gen11_transfer/; produce it with scripts/gen10_final_locked.py pointed at the gen11 arms._

## 7. Negative controls (`t7_controls.csv`)

_Not produced: no negative-control arm has been run (scripts/gen11_run.py --stage controls)._

## 8. The thirteen questions (§21)

_Question wording is reconstructed from the gen11 module contracts; the brief document is not in this worktree. Each answer names the table it came from._

### 1. Does the audited multi-metal archive improve zero-shot prediction for unseen lanthanide chemistry on the frozen gen10 benchmark?

_(maps to §1/§18A)_

Best auxiliary arm is `B_LN_EXPANDED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1` with a **transfer effect of -0.0037 macro MAE** against its design-matched control (`t2_leaderboard.csv`), improving in 2/5 seeds. The §18A bar is 0.02 with consistent seed direction. Chemotype-blocked paired bootstrap (`t3_paired_bootstrap.csv`): point -0.0037, BCa 95 % [-0.0187, 0.0154], percentile [-0.0184, 0.0159], 57/131 units improved.

### 2. How much of any observed change is the design change (general metal representation and annotation-safe MASSACTION) rather than the auxiliary data?

_(maps to §3/§7)_

Measured directly by running the control at every design corner (`t2_leaderboard.csv`, column `design_effect_macro`, positive = the changed design is better than frozen; intervals from `t3_paired_bootstrap.csv`): `GENERAL|FROZEN_8` 0.0003 [BCa -0.0039, 0.0053; not separated]; `FROZEN_3|FROZEN_8` 0.0000; `GENERAL|ANNOTATION_SAFE` -0.0038 [BCa -0.0109, 0.0012; not separated]; `FROZEN_3|ANNOTATION_SAFE` -0.0086 [BCa -0.0156, -0.0041; candidate worse]. Any `TOTAL_CONFOUNDED` row in `t3_paired_bootstrap.csv` is the sum of this and the transfer effect and must not be quoted as a transfer result.

### 3. Which auxiliary composition helps, if any — extra lanthanides (B), actinides (C), other metals (D), or everything (E)?

_(maps to §5 B-E)_

Transfer effect by composition (`t2_leaderboard.csv`): `B_LN_EXPANDED` -0.0037. Read these against `t1_arm_composition.csv`: the auxiliary pool adds metals and curves, not ligands.

### 4. At equal row count, is any gain chemical relevance or merely row count (F actinides-matched vs G random-matched)?

_(maps to §5 F/G)_

**Not answerable**, because the size-matched arms F and G have not been run (`t2_leaderboard.csv`); the F/G contrast is the only thing that separates chemical relevance from row count.

### 5. Which transfer mechanism does the work — joint training, auxiliary pretraining, or a shared representation with a lanthanide-only head?

_(maps to §6)_

**Not answerable**, because only the JOINT mechanism has been run; §6 requires the mechanisms to be compared, not collapsed (`t2_leaderboard.csv`).

### 6. Does any effect survive the pre-registered weighting schemes and auxiliary mass settings, or is it a weighting artefact?

_(maps to §12)_

**Not answerable**, because only one weighting/auxiliary-mass setting has been run, so §12's robustness question has no contrast to measure (`t2_leaderboard.csv`).

### 7. Does the effect sit in the per-ligand level or in the within-curve shape?

_(maps to §11)_

Level/shape split of the transfer effect (`t6_decomposition.csv`, `stratum = ALL`): `B_LN_EXPANDED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1`: level -0.0004, shape 0.0043, total -0.0037

### 8. Does the archive help most where the cohort is thinnest — on distant held-out chemotypes?

_(maps to §18D)_

Improvement by chemotype-distance band (`t6_decomposition.csv`): `B_LN_EXPANDED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1` [GEN10_TERCILE]: far -0.0154 (87 units), mid 0.0075 (51 units), near 0.0114 (38 units); `B_LN_EXPANDED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1` [GEN11_ANALYSIS]: far -0.0186 (30 units), mid -0.0163 (71 units), near 0.0070 (71 units). Both band definitions present in the repository are reported because they disagree (see the caveats).

### 9. Does the archive improve curve shape on a scientifically important axis?

_(maps to §8/§18C)_

Best per-axis shape gain (`t4_shape_by_axis.csv`): `B_LN_EXPANDED|JOINT|GENERAL|ANNOTATION_SAFE|HIERARCHICAL|lam1` on axis `extractant`, shape MAE 0.4741 → 0.4675 (0.0066 over 155 curves). The §18C bar is 0.10 with macro non-worse.

### 10. Does the archive change the k = 0/1/2/3/5 few-shot frontier?

_(maps to §18B)_

**Not answerable**, because no k-shot frontier exists for gen11 arms (`t5_kshot.csv` skipped); the frontier is produced by pointing `scripts/gen10_final_locked.py` at `runs/gen11_transfer/arms`.

### 11. Do the negative controls (permuted auxiliary target, shuffled metal labels) stay flat, as they must for any gain to be real?

_(maps to §19)_

**Not answerable**, because the §19 negative-control arms (permuted auxiliary target, shuffled metal labels) have not been run (`t7_controls.csv` skipped); until they are, no positive result can be attributed to auxiliary information rather than to the extra rows themselves.

### 12. Does any result survive publication-blocked auxiliary admissibility and publication-blocked intervals?

_(maps to §13)_

**Not answerable**, because no `PUBLICATION_BLOCKED` arm has been run, so §13's publication-level robustness cannot be assessed from arm predictions (`t2_leaderboard.csv`) — the publication-blocked *interval* variant is available separately by rerunning this script with `--block doi`.

### 13. Is there enough evidence to replace frozen gen10 with an auxiliary arm?

_(maps to §18)_

**No.** No auxiliary arm passes any §18 criterion it could be scored on (`stopping_rule.csv`, status counts {'FAIL': 3, 'NOT_APPLICABLE': 20, 'NOT_EVALUABLE': 2}). Frozen gen10 remains the control. Read the `NOT_EVALUABLE` cells separately from the `FAIL` cells: the first is missing evidence, the second is a measured negative.

## 9. The stopping rule (§18), evaluated mechanically

Status counts across all (arm, criterion) cells: **FAIL** 3, **NOT_APPLICABLE** 20, **NOT_EVALUABLE** 2.

| arm | arm_kind | criterion | measured_quantity | measured_value | threshold | requirement_met | status | reason |
|---|---|---|---|---|---|---|---|---|
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | A_ZERO_SHOT | not applicable | n/a | 0.02 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | B_FEW_SHOT | not applicable | n/a | 0.01 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | C_SHAPE | not applicable | n/a | 0.1 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | D_FAR_CHEMOTYPE | not applicable | n/a | n/a | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | E_ROBUSTNESS | not applicable | n/a | n/a | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | A_ZERO_SHOT | not applicable | n/a | 0.02 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | B_FEW_SHOT | not applicable | n/a | 0.01 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | C_SHAPE | not applicable | n/a | 0.1 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | D_FAR_CHEMOTYPE | not applicable | n/a | n/a | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|FROZEN_3\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | E_ROBUSTNESS | not applicable | n/a | n/a | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | A_ZERO_SHOT | not applicable | n/a | 0.02 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | B_FEW_SHOT | not applicable | n/a | 0.01 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | C_SHAPE | not applicable | n/a | 0.1 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | D_FAR_CHEMOTYPE | not applicable | n/a | n/a | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | E_ROBUSTNESS | not applicable | n/a | n/a | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | A_ZERO_SHOT | not applicable | n/a | 0.02 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | B_FEW_SHOT | not applicable | n/a | 0.01 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | C_SHAPE | not applicable | n/a | 0.1 | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | D_FAR_CHEMOTYPE | not applicable | n/a | n/a | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| A_GEN10_CONTROL\|JOINT\|GENERAL\|FROZEN_8\|HIERARCHICAL\|lam1 | DESIGN_CONTROL | E_ROBUSTNESS | not applicable | n/a | n/a | n/a | NOT_APPLICABLE | this arm carries no auxiliary rows; it is a design control, and §18 governs auxiliary arms |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | AUXILIARY | A_ZERO_SHOT | macro MAE improvement vs design-matched control | -0.0037 | 0.02 | 2/5 | FAIL | n/a |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | AUXILIARY | B_FEW_SHOT | macro MAE improvement at k in {1,2,3,5} | n/a | 0.01 | n/a | NOT_EVALUABLE | no k-shot frontier has been produced for this arm |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | AUXILIARY | C_SHAPE | shape MAE improvement on axis 'extractant' (155 curves) | 0.0066 | 0.1 | False | FAIL | n/a |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | AUXILIARY | D_FAR_CHEMOTYPE | far-band macro MAE improvement (30 units, 2138 rows) | -0.0186 | BCa 95 % lower bound > 0 | False | FAIL | n/a |
| B_LN_EXPANDED\|JOINT\|GENERAL\|ANNOTATION_SAFE\|HIERARCHICAL\|lam1 | AUXILIARY | E_ROBUSTNESS | query-design sensitivity at matched accuracy | n/a | lower than the control | n/a | NOT_EVALUABLE | the query-consistency run has not been produced for gen11 arms |


## 10. Reproducing every number here (§22)

```sh
PYTHONPATH=src python scripts/gen11_report_tables.py
```

It reads only `runs/gen11_transfer/arms/oof_*.parquet` (raw predictions), `runs/gen11_transfer/composition/` (counts) and, where present, the gen11 k-shot detail; it recomputes macro/offset/shape with `gen6.metrics` and `gen9.metrics`, the bootstrap with `gen6.metrics.paired_unit_bootstrap`, and rewrites this file from the CSVs it produced. Deleting `runs/gen11_transfer/headline_tables/` and rerunning reproduces both the tables and this report.

The only cached step is gen9's curve reconstruction, in `headline_tables/.shape_cache/`. Its key is a content hash of the arm's predictions, so a refitted arm cannot be served a stale entry, and deleting the directory rebuilds it. `--skip-shape` omits it entirely, at the cost of making criterion C `NOT_EVALUABLE`.

## 11. Caveats this run detected

* `gen11.analysis.CHEMOTYPE_BANDS` puts the far/mid cut at Tanimoto 0.400 and calls it gen10's, but `composition.py` recomputes gen10's own tercile cut at 0.588284 (near at 0.657143). They disagree, so `t6_decomposition.csv` reports both and criterion D is evaluated on the gen11 module's definition.
* the design change alone — `A_GEN10_CONTROL|JOINT|FROZEN_3|ANNOTATION_SAFE|HIERARCHICAL|lam1` against the frozen anchor, no auxiliary rows — is significantly *worse*: -0.0086 macro MAE, BCa [-0.0156, -0.0041]. no auxiliary arm on disk uses this corner, so it is reported as an isolated measurement of that one design change rather than as a deficit any arm carries.
* one chemotype block holds 21.4% of the scoring units in the bootstrap; gen6 measured the percentile interval at ~12.7 % one-sided Type-I error under exactly this leverage, so the BCa and cluster-robust columns of `t3_paired_bootstrap.csv` are the ones to read.
