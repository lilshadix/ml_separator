# Stage 3 outputs — which files are authoritative

Written 2026-09-08.  The report is `STAGE3_REPORT.md`; the independent audit that re-derived the
headline from scratch is `verify_direction/VERDICT.md`.  Several files here were superseded during
the work and are kept for provenance rather than for quoting.  Use this list.

## Quote from these

| file | what it holds |
|---|---|
| `s3_direction_accuracy_FIVE_DESIGNS.csv` | direction accuracy per model and design, B / BR / BQ / A / BP, with the correct constant baseline |
| `s3_direction_gain_FIVE_DESIGNS.csv` | paired gains over "always predict heavy-selective", one shared chemotype resample per design |
| `s3_direction_value_BP.csv`, `s3_direction_value_contrasts_BP.csv` | what a direction call is worth on the extractant-macro MAE |
| `s3_geometry_statistics.csv` | the donor-compactness correlations with chemotype-blocked intervals and every control |
| `s3_permutation_importance_BP.csv` | per-descriptor permutation importance of the direction model |
| `s3_amponly_boards.csv`, `s3_amponly_contrasts.csv` | the amplitude-only candidate across five designs (falsified) |
| `s3_attenuation_*.csv`, `s3_smooth_*.csv`, `s3_sweep_BP.csv`, `s3_sweep2_BP.csv` | the closed branches of §2 |

## Do not quote from these

* **`s3_direction_accuracy.csv`** — superseded twice.  Its `always_heavy` rows come from a
  mis-specified baseline: the first version of `s3_direction.py` predicted the *chemotype-weighted
  training majority* rather than "always heavy-selective", and because that weighted majority is
  often the light class it scores 0.35 instead of 0.56, which flatters every comparison against it.
  The file was then overwritten by a later run, so it now holds only the BR and BQ designs.  The
  corrected baseline and the paired intervals are in the two `_FIVE_DESIGNS` files.
* `s3_direction_accuracy_fixed.csv`, `s3_direction_gain.csv` — correct, but cover only B, BP and A;
  the `_FIVE_DESIGNS` files supersede them.
* `s3_direction_accuracy_all_designs.csv`, `s3_direction_gain_all_designs.csv` — correct, but cover
  only BR and BQ, for the same reason.
* `s3_direction_predictions.parquet` — holds only the last designs run (BR, BQ); the earlier
  per-cell predictions for B, BP and A were overwritten by that run.  Re-create with
  `s3_direction.py B,BP,A` if per-cell predictions are needed again.

## Two labelling corrections the audit forced

`LEAN_BLOCKS` is **209** columns (conditions + mass action + physchem + donors + coordination), not
137; the 137-column set is physchem + donors + coordination alone and was never run in the headline
comparison.  And the donor-topology family is **no worse than** the full set, not better: the
difference is inside its own interval.  Figures regenerated after the audit carry the corrected
labels; any copy of `s3_direction_accuracy.png` or `s3_direction_value.png` older than that does not.
