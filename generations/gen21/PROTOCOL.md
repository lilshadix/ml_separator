# gen21 — protocol (written before any new method was run)

Task: `CLAUDE_TASK.md`. This file fixes the cohort, the measurement modes, the metric, the splits
and the success criterion **before** the first gen21 method is scored. Changes after that point go
into `ATTEMPTS.md` as dated amendments that say which results had already been seen.

Written 2026-10-18, after only one computation: the reproduction of gen20's baselines (§2).

## 1. Data, target, orientation

* Frozen gen13 cohort as gen20 loads it (`gen20k.data.load_bench`, asserted column-wise equal to
  `gen13_separation/manifests/cohort_exact.parquet`, SHA-256 `3f11ac4c…`): 521 cells, 90
  extractants, 45 chemotypes, 58 publications. A cell is one extractant under one exactly matched
  condition key in one publication; log D is the replicate mean per (cell, metal).
* Target: `logSF(A/B) = log10 D(A) − log10 D(B)` with **A the lighter lanthanide**, inside one cell.
  Heavier-first is a sign flip and is never mixed in.
* Query pairs: **strict adjacent** pairs (|ΔZ| = 1; 12 pair types, La/Ce … Yb/Lu; Pm is absent, so
  Nd/Sm is *not* adjacent). Every strict-adjacent pair of a cell is a query; the set is fixed
  before any method runs and is identical across budgets and methods.

## 2. Reproduced reference (computed before this protocol)

gen20 metal model, policy DOPT_Q, design BP, discovery seeds, cohort C3 (229 cells, 1,802 pairs
per seed), rebuilt from gen20's own functions (`scripts/g21_reproduce.py`,
`results/reproduction_gen20_BP_C3_DOPT_Q.csv`): RIDGE2 at k = 3, R² **0.3099**, MAE 0.1206, macro
MAE 0.1346; LINE 0.2356; G14@BLUP20 0.2482; FLAT@BLUP20 0.1890; zero-shot G14 0.0529; FLAT
−0.1347. This matches gen20's `results/discovery/levels_BP_metal_C3_DOPT_Q.csv` to four
decimals for every arm. **It is a k = 3 result (four metals read per cell), not zero-shot and
not a log D result.**

## 3. Measurement modes (what a new cell exposes at prediction time)

* **Zero-shot (k = 0):** the extractant's structure features and the condition columns; no
  measurement of the held-out cell.
* **k-shot, metal model (k = 1 … 6):** `k + 1` metals of the held-out cell are read, and their
  log D values are known (absolute values; gen20 revealed only their differences, and every
  gen21 method states which of the two it uses). The read set never contains both metals of a
  query pair, so **no scored log SF is computable from two open readings**; the table carries
  `a_read`/`b_read` and `lab_known` for every row and `lab_known` must be false everywhere.
* The read set is chosen without labels. The gen20 policy DOPT_Q is the default; a method may
  bring its own label-free policy, in which case it is scored **and** its baseline is re-scored
  on the method's supports, and both are also reported on DOPT_Q supports.
* Each budget k is a separate regime and is reported separately. A zero-shot number is never
  quoted for a k-shot regime or the reverse.

## 4. Cohorts and rows

* Primary rows: common-query cohort **C3** — (seed, cell) with `k_max ≥ 3` under the metal model
  (229 cells, 79 extractants, 38 chemotypes; 1,802 strict-adjacent pairs per seed). Every budget
  k ≤ 3 is scored on byte-identical rows.
* Secondary: C6 (`k_max ≥ 6`, 133 cells) for k = 5, 6; the k = 0 regime on C3 rows.
* Per pair type tables on C3 at k = 3.

## 5. Splits and hold-out

* **Development (dev):** gen20's chemotype-grouped 5-fold hold-out with the held-out cells'
  publications removed from training (design **BP**), split seeds 104729, 130363, 155921, 196613,
  262147 — the gen20 discovery seeds. gen20's results on these seeds are published, so dev is
  not blind with respect to the baselines; it is only used to choose methods.
* **Strict copy check:** design **BPG** (BP plus removal of every training cell whose gen19
  cross-publication copy group appears among the held-out cells), same seeds.
* **Inner fitting:** every free parameter of a gen21 method is fitted on the outer training fold
  only, through the four inner chemotype folds of that training fold (gen13's inner seed), with
  the design mask re-applied inside and a nested covariance (`g21/cache.py`). gen20's noise factor
  and ridge λ (chosen on gen20's inner split) are reused for the baselines unchanged.
* **Final check:** five new split seeds generated at random and stored outside the repository;
  only the SHA-256 of that file is committed (`manifests/final_seeds_commitment.txt`,
  `30ec7c0d…`). It is run **once**, only after one method has been frozen on dev, and only with
  the user's go-ahead. New split seeds re-partition the same 521 cells: they are a fresh
  partition, **not** independent laboratory data. Results seen on the final run are never
  re-tuned and called confirmation afterwards.

## 6. Metrics and comparison

* **Primary metric:** pooled R² of strict-adjacent log SF on C3 at the declared budget, computed
  per split seed over that seed's rows and averaged over seeds (gen20's convention); never
  clipped. Also MAE, RMSE, extractant-macro MAE, per pair type R²/MAE, direction accuracy.
* **log D:** for a method that fixes absolute levels (k ≥ 1), R²/MAE/RMSE of log D of the
  **unread** metals of C3 cells; a method that only predicts differences gets "not predicted".
* **Baselines on identical cells, pairs and supports:** RIDGE2 (the reference), LINE,
  G14@BLUP20, FLAT@BLUP20, zero-shot G14, FLAT (= 0), and for every attempt the previous best
  gen21 method. Intervals: chemotype-blocked bootstrap (gen20's `inference.pooled_levels`), per
  seed values printed.
* **Dev advancement rule for an attempt:** R² and MAE both better than RIDGE2 on the same
  supports, in at least 4 of 5 seeds, and the sign holds on BPG.

## 7. Success criterion (CLAUDE_TASK.md)

The target is declared reached only if a method frozen on dev, on the final check: R² of strict-
adjacent log SF ≥ 0.40 in its declared regime; better MAE than the comparable baseline (RIDGE2 on
the same supports); the result holds on the copy-group-strict split; and the leakage checks pass
(§8). Each seed, the bootstrap interval and the transfer limits are reported. A failing final
check is reported as failing; it is not re-composed or re-run.

## 8. Leakage checks (every candidate)

1. Fold isolation: zero overlap of cell, extractant, chemotype, publication (and copy group under
   BPG) between training and held-out cells; inner blocks inside the outer training fold
   (`evaluate.assert_test_isolation`).
2. Provenance: every fitted quantity (prior, λ, covariance, noise, method parameters) comes from
   outer-training cells; recorded per fold.
3. Intervention: the hidden log D values of held-out cells are replaced by random numbers with the
   read set and mask unchanged; every prediction must be bit-identical.
4. No scored pair has both metals read (`lab_known` false on every row).

## 9. Hashes

`manifests/inputs_sha256.json`: cohort manifest `3f11ac4c…`, log D matrix `f0be7fb3…`, BP fold
plan `56e6fded…` (12,073 rows), BPG fold plan `13632a17…` (12,031 rows), every cache file, the
gen21 code at the time of writing.
