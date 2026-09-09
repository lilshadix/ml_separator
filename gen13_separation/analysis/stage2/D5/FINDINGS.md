# D5 - who carries the error, and is it noise or signal
Best = X_ENS_DIRECT+LOWRANK_K2, incumbent = C_DIRECT_ROW, floor = B1_MEAN_CURVE. B_primary: 90
extractants, 521 cells, 14150 held-out pairs/seed x 5 seeds = 70750. Reproduced exactly:
extractant-macro MAE 0.4810 / 0.4953 / 0.6033; pooled 0.4917; mean|y| 0.6500.
## 1. A minority of extractants carries the error
Per-extractant MAE (n=90): median 0.349, IQR 0.216-0.590, p90 1.048, max 2.545, min 0.087, Gini 0.406.
Worst 10 (11% of extractants) carry 32.5% of the summed per-extractant MAE and 59% of all excess above
the median. Removing them: macro 0.4810 -> 0.3650 (incumbent 0.4953 -> 0.3778); worst 20 -> 0.3075.
Worst 15 (d5_worst15_extractants.csv) as MAE / mean|y| / n_cells / pairs-per-seed: 2-(dibutylcarbamoyl)
benzoic acid sc063 2.545/2.667/1/3; ADAAM(EH) sc016 2.065/1.825/1/1; tetraphenyl-tetraaza macrocycle
sc060 1.321/1.261/2/182; dipropyl-BTP sc061 1.227/1.723/1/91; NTAamide(C8) sc032 1.224/1.203/5/10;
then sc015 1.215, sc021 1.181, DODDdDGA 1.150, DMDOaDGA 1.115, dihexyl hydrogen phosphate 1.041,
DMDODGA sc009 0.958 (74 cells, 1282 pairs), sc030 0.924, sc068 0.922, DPDODGA 0.769, MMTODGA 0.763.
## 2. "Worst" mostly means "largest curve", not "hardest"
Spearman(MAE, mean|y|) over 90 extractants = 0.781; log-log fit MAE ~ mean|y|^0.64, R2 = 0.577. The 10
worst have median mean|y| 1.507 vs corpus median 0.390. n_cells does NOT predict difficulty (rho
-0.026), nor n_publications (-0.057) nor max n_metals (0.090). Normalised MAE/mean|y|: median 0.883,
IQR 0.667-1.087, Gini 0.338 - but 30 of 90 extractants have ratio >= 1, no better than a flat curve.
Size-adjusted hardest: sc083, sc078 DGAs (ratio 10.5, 6.8 on near-flat curves, mean|y| ~0.04).
## 3. Cuts (MAE, mean|y|, ratio; full tables in d5_cut_*.csv)
dZ 1..14 pooled MAE 0.148 -> 1.128 while mean|y| 0.167 -> 1.638, so the ratio only falls 0.885 ->
0.689: adjacent pairs are relatively the HARDEST. Fraction beating the mean curve 0.578 (dZ=1) -> 0.73
(dZ>=8); overall 0.652. Adjacent pairs (2114/seed) macro: model 0.1703, mean curve 0.1913, incumbent
0.1699 - gain over the mean curve 0.0210 [95% CI 0.0136-0.0290, bootstrap over 82 extractants]: real
but exactly at the 0.02 threshold, and the ensemble adds nothing over C_DIRECT_ROW there (-0.0004).
Its gain lives at dZ 4-7 (+0.018) and dZ>=8 (+0.023). Gd-crossing 0.758/1.040 (ratio 0.729);
heavy-heavy 0.175/0.178 (0.986, no skill); light-heavy 0.933/1.304 (0.715). Per metal La worst 0.729
to Dy best 0.394, but ratio 0.69-0.80 for all 14 - no metal is intrinsically hard. Cells with 2-3
metals score macro 0.684 vs 0.492 for 12-14 metals. Chemotype sc009 (23 extractants, 7210 pairs/seed)
0.533 pooled / 0.510 macro. Tanimoto far (<0.40) 0.653 pooled vs near 0.461 (macro 0.563 vs 0.395).
Acid >6 M ratio 1.29. replicate_sd_median bins show no monotone trend (0.504 at <=0.05 vs 0.250 at
>0.60): noisier cells are not harder.
## 4. The error is NOT usefully predictable without the label
Out-of-fold tree (depth 3-4; n_metals, n_rows, replicate_sd_median, max_train_tanimoto, chemotype
size, dZ, |prediction|, acid M): Spearman(pred|err|, |err|) = 0.414; ridge 0.173. But dZ alone gives
0.538 and |prediction| alone 0.445 - the model adds nothing over trivial covariates. Within-dZ
n-weighted Spearman for the tree is -0.019 (|prediction| 0.165, itself a magnitude artefact).
Abstaining on the worst predicted decile: pooled 0.492 -> 0.452, macro 0.481 -> 0.436, but MAE/mean|y|
among kept pairs goes 0.7564 -> 0.7505 (0.7540 at 20%) versus 0.7566 for random abstention and 0.7718
for dZ-only. Oracle abstention reaches 0.3527 pooled / 0.7211 normalised. Dropping the 10
highest-predicted-error extractants gives macro 0.4505 vs 0.3650 oracle, with only 2/10 of the truly
worst identified. An abstention rule is worth nothing.
## 5. Replicate-noise floor: unmeasurable for 89% of extractants
Only 41 of 521 cells (7.9%) carry replicate information, 216 of 3359 logD values. Only 10 of 90
extractants have a replicate_sd_median; 3 of those 10 sit at or below the sqrt(2)*sd pairwise floor
(same 3 for the incumbent), covering 363 of 14150 pairs/seed. Pair-level: 836 pairs/seed (5.9%, 26
cells, 9 extractants) have a measured sd for both metals; median floor sqrt(sdA^2+sdB^2) = 0.521 vs
MAE 0.493 there, and 55.8% of those pairs are already inside their own floor; 5 of those 9 extractants
are at or below it. Against the cohort floor (median sd 0.113 -> 0.160 pairwise) the macro MAE is
3.0x noise. The replicated subset is biased (median sd 0.302 vs 0.113 overall), so "how many
extractants are unimprovable" cannot be answered beyond n=10 on this cohort.
## Verdict: signal, not noise
Per-extractant MAE reproduces across the 5 split seeds with ICC = 0.978 (between-extractant variance
0.172, within 0.0038): the same extractants are hard every time. The SIGNED residual reproduces across
independent cells of the same extractant - split-half Pearson r = 0.732 mean / 0.816 median over 20
extractants (>=2 cells, >=6 shared pairs, 20 random splits each), 90% with r > 0.5. The model misses a
reproducible extractant-specific curve shape, not measurement noise. Attack the 10-20 large-curve
extractants (sc061/sc060/sc015/sc021 N-heterocycles, long-chain sc009 DGAs); do not build a confidence
or abstention layer.
