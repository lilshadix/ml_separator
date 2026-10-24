
## 11. Stage 1, block B: strain-architecture adaptation (amendment A2; DRAFT, not sealed)

**Question.** Does a small learned correction of B1's level residual at unread metals add skill beyond B1 and B2?
- **S05:** tabular information only.
- **S06:** tabular information plus compact 3D geometry of the cell's complexes.

This ports the strain architecture's level / shape idea in support-preserving form (brief §6):
- B1 plays the level and anchor;
- a CatBoost model supplies the scalar "shape" correction;
- read metals keep their exact readings.

Code: `r4/slots_b.py`.

### Model

- **Target.** For each training situation (B1's masks and weights) and each unread metal m: `t = y(m) − B1_S(m)`, the level residual against B1's prediction from the four reads.
- **Prediction.** `level(m) = B1_S(m) + f(features(m))` at unread metals, and `level = y` at read metals.
- **Estimator.** CatBoostRegressor (depth 4, learning rate 0.05, l2 3, RMSE, seed 0, 4 threads).
  - The number of trees comes from {0 (= B1), 100, 300, 1000} by inner chemotype-block pair loss, using staged predictions of one 1,000-tree fit per inner block.
  - B1's β is refitted per inner block (as in block A).
  - Ties go to fewer trees.
- **Features** (label-free for the held-out cell):
  - **S05:**
    - metal x and 4f count;
    - distances to the bracketing reads, interpolation fraction and extrapolation side;
    - the bracketing segment's slope and the least-squares slope of the four reads;
    - B1's coherent Gd feature;
    - round 3's 6 read-geometry features;
    - donor class (3);
    - 8 RDKit descriptors of the extractant.
  - **S06:** S05's features plus geometry of the cell's complex at metal m (`dataset with 3D structures/features/features_by_safe_exp_id.parquet`, joined through the cell's safe_exp_ids and the archive's metal):
    - mean Ln–donor distance, physical coordination number and metal partial charge;
    - each as g(m) and as g(m) − I_S[g(S)](m);
    - a coverage flag. Missing geometry falls back to CatBoost's missing branch.
  - **Coverage:** 89.0% of C3 queries have geometry at both ends (label-free; `results/provenance/geometry_coverage.json`). The full 229-cell denominator is always scored.

### Slots and controls

| slot | content | role |
|---|---|---|
| S05 | tabular correction | does a flexible chemistry / support-summary correction add to B1? |
| S06 | S05 + compact geometry | does 3D geometry add to S05? |
| S09 | S06 with every chemotype's geometry taken from another chemotype (5 draws) | geometry block-shuffle control of S06 |
| S22 | S06 with g replaced by the training-fitted per-metal mean of g | radius-only (series-only) geometry control; moved into block B |
| S07, S10 | distance encoder; random-encoder control | gate G-geo below |
| S08 | nested scalar blend of B1 / S05 / S06 | conditional on S05 and S06 both scoring |

### Gates and claims

- **G-geo inner gate (encoder slots S07, S10).** S06's inner-CV loss must be below B2's in ≥ 30 of the 50 outer fold-fits.
  - Even if it passes, the encoder is run only if its estimated CPU cost fits RESOURCE_PLAN. Otherwise S07 and S10 are NOT RUN, with the reason.
- **Criterion 1 / harm / designs** as in §6 and §10.
- **Geometry claim (S06).** All of, on BPG and on BP:
  - S06 passes criterion 1;
  - S06 − S05 ≥ 0.005 ΔR²;
  - S06 > every S09 draw;
  - mean(S09) < 0.5 × S06's gain over S05;
  - S06 − S22 ≥ 0.005.
- **Controls** (S09, S22, S10) are never finalist-eligible.
