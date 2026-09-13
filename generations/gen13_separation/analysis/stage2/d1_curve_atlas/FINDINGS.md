# D1 - per-extractant curve atlas and curve reproducibility

Unit = the cell (extractant x exactly matched condition vector); target = the centred lanthanide curve. 521 cells, 90 extractants, 3359 observed (cell, metal) entries.
Acid colouring uses `cond__acid_concentration_M` restricted to `cond__acid__hno3 == 1` (429/521 cells; the 92 non-HNO3 cells are drawn grey).

## 1. What the corpus contains
Top 12 extractants by cells hold 409/521 cells (78.5%); 64 of 90 extractants have exactly 1 cell. TODGA alone has 171 cells (32.8%), 902 entries, 18 publications,
171 distinct condition keys. Metals per cell: median 6, mean 6.45, but 138 cells (26.5%) hold exactly 2 metals and only 321/521 (61.6%) reach 4. The high-cell
extractants are the sparse ones (median metals/cell: TODGA 3, DMDODGA 3, TBDGA 2); the dense 13-14 metal series belong to C5BTBP (16 cells), DMDPhPDA (10) and
many single-cell extractants.

## 2. Reproducibility of one extractant's curve (pairs with >= 4 shared metals)
MAE is after re-centring both curves on the shared metals; SF MAE = mean |dlog SF| over every metal pair in the shared set, i.e. the programme's scale.
| level | pairs | mean r | median r | frac r<0 | curve MAE | SF MAE |
|---|---|---|---|---|---|---|
| within extractant, same publication | 1066 | 0.746 | 0.956 | 7.5% | 0.190 | 0.279 |
| within extractant, different publication | 2142 | 0.681 | 0.934 | 12.3% | 0.250 | 0.374 |
| within extractant (all) | 3208 | 0.703 | 0.941 | 10.7% | 0.230 | 0.342 |
| different extractant, same chemotype | 7628 | 0.506 | 0.902 | 21.5% | 0.343 | 0.519 |
| random cell pairs | 20000 | 0.335 | 0.698 | 31.0% | 0.380 | 0.566 |

Mean r sits far below median r because a minority of anti-correlated pairs (mostly 2-4 metal cells whose "curve" is one noisy step) drag it down. Repeating the SF
metric at >= 2 shared metals (whole cohort, 9237 within-extractant pairs) preserves the ordering: 0.303 within extractant (0.229 same pub, 0.331 diff pub), 0.479 same chemotype, 0.526 random.

## 3. Headline - extractant-macro transfer error
Scored the programme's way: predict a cell's log SFs by copying a real measured sibling cell of the same extractant.
0.286 log units over 24 extractants (450/521 cells), pooled 0.295; 0.390 when the sibling must come from a different
publication (9 extractants only). Control - copying a cell of a different extractant in the same chemotype - 0.441 over
56 extractants; paired on the 16 where both exist, 0.257 own vs 0.454 other, so extractant identity is worth 0.196 log
units. References: best ensemble arm 0.481, C_DIRECT_ROW 0.495, corpus mean curve 0.603. One real sibling measurement
therefore beats the best model by 0.195 (0.091 across publications), and 0.286 is about the floor for a model whose only
handle on a cell is "which extractant". Disagreement grows with the acid gap but weakly (1417 pairs, Spearman(|dlog10
[HNO3]|, SF MAE) = 0.165): 0.218 at identical [HNO3] (217 pairs), 0.285 <0.2 dex, 0.360 at 0.2-0.5, 0.341 at 0.5-1,
0.377 above 1 dex (346 pairs).

## 4. Shape coefficients (quadratic in standardised Shannon CN8 radius, 289 cells with >= 5 metals)
R2 mean 0.876, median 0.962; the linear term alone gives R2 mean 0.744, so the tilt carries most of the shape. Amplitude
(linear coef; negative = heavier Ln favoured): mean -0.414, sd 0.680, range -2.300 to +2.053, negative in 75.8% of
cells; curvature mean -0.052, sd 0.415. Per extractant (cells fitted): TODGA -0.844 +/- 0.535 (63), C5BTBP -0.722 +/-
0.151 (16), TDdDGA -0.825 +/- 0.922 (11), DHD2DGA -0.098 +/- 0.096 (20), DOODA (C12) +0.901 +/- 0.555 (10, light-
selective), TEDGA +0.083 +/- 0.699 with linear-only R2 0.455 (strongly curved, peaks Nd-Sm). Variance split of amplitude
(21 extractants, 228 cells): 44.0% of SS between extractants, 56.0% within one extractant; sd_between 0.467, sd_within
0.554, ICC 0.415. Curvature ICC 0.188; overall curve size (sd of the centred curve) ICC 0.496.

## Answer
Both, and the split is measurable. The sign and rank order of an extractant's curve are stable (median within-extractant
r 0.94; identity buys 0.196 log units over chemotype alone). The magnitude is not: 56% of amplitude variance sits
between cells of one extractant, and a sibling from another paper is 0.11 worse than one from the same paper. The
extractant fixes the shape, the cell fixes the scale - so predict a per-extractant curve shape and let condition
features set its amplitude, rather than one fixed curve per extractant.

## Caveats
Every level except "same publication" is confounded by chemotype imbalance (diglycolamides are 375/521 cells, so the chemotype control is largely DGA-vs-DGA). The
>= 4 shared-metal analysis uses 321/521 cells; 2-metal cells enter only the >= 2 numbers and cannot support a correlation. The different-publication macro number
rests on 9 extractants. Transfer MAE uses a real measured sibling, so it is a ceiling estimate, not a held-out model score.

## Outputs
12 CSVs here (ranking, metal coverage, similarity levels, SF transfer, within-extractant pairs and summary, acid gap, quadratic fits, amplitude summary, variance
decomposition, macro transfer and control), scripts d1_curve_atlas.py + d1_macro_transfer.py, logs d1_stdout.txt + d1_macro_stdout.txt. Figures in figures/stage2/:
d1_atlas_grid_top12.png, d1_curve_01..12_<name>.png, d1_similarity_levels.png, d1_amplitude_curvature.png.
