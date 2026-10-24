# Metrics (stage 0; code `r4/metrics.py` on top of `r3/metrics.py`)

The scored population is P0: V1, C3, k = 3, DOPT_Q. That is 229 cells and 1,802 strict-adjacent queries per seed, over 5 dev seeds and BP / BPG. Every arm is scored on identical rows and supports, and **BPG decides** (PROTOCOL §6).

## Level statistics (seed means)

- **R² (primary, historical).** Inside each split seed, `1 - sum (y - p)^2 / sum (y - mean y)^2` over all of that seed's pooled queries, then the mean over seeds.
  - It is never clipped. Undefined values are NaN, with `R2_undefined_reason`.
  - Pooled-all-OOF or ensemble R² appears only under a separate name.
  - Repeated predictions are never pooled as independent rows.
- **Balanced MAE** (round 3's `MAE_macro_chem`, renamed). Inside each seed:
  1. average |p − y| over each extractant's rows;
  2. average the extractants inside each chemotype;
  3. average the chemotypes equally.

  Then take the mean over seeds. This is fixed, and no other balancing may be substituted. It is **not** the references' training weighting: training is chemotype-balanced per cell, times the number of hidden queries (DATA_CONTRACT).
- **Also reported:**
  - row MAE, RMSE, bias (p − y), direction accuracy (a zero prediction scores 0.5) and extractant-macro MAE;
  - per-seed R², min-seed R², and rows and cells per seed.
- **Levels with intervals** (`levels_*.csv`): R², MAE and RMSE, each with a cluster-bootstrap percentile 95% interval under both clusterings. This is the interval for the 0.50 aspiration.
- **Subsets** (`summary_*.csv`):
  - sc009 / non-sc009;
  - donor-defined DGA / non-DGA (`r4/donor.py`);
  - one end read / both hidden;
  - Pr/Nd, Eu/Gd, Gd/Tb.
- **Per pair type** (`per_pair_*.csv`) and **per publication** (`per_publication_*.csv`).
- **Calibration diagnostics** (`calibration_*.csv`): raw R², RMSE, bias, prediction variance, target variance and Pearson². The test-fitted optimal scale `cov(y, p) / var(p)` is labelled **ORACLE**; it is never a score or a baseline.

## Paired contrasts (`contrasts_*.csv`)

- **Form.** Candidate minus reference on identical rows, signed so that positive favours the candidate:
  - ΔR² = R²(cand) − R²(ref);
  - ΔMAE, ΔRMSE and Δbalanced-MAE = ref − cand.
- **Cluster bootstrap.**
  - One shared draw matrix: gen20k.inference, seed 8675309, 2,000 replicates.
  - Draws are over clusters, the same for every seed. Per-seed statistics come from cluster sums (balanced MAE from chemotype means reweighted by cluster counts), then are averaged over seeds.
- **Columns, for every contrast:**
  - percentile 95% and BCa 95%;
  - two-sided bootstrap p and bootstrap SE;
  - **cluster-jackknife t(G−1) 95%**, the small-cluster sensitivity: leave-one-cluster-out seed-mean contrasts, `SE = sqrt((G−1)/G · Σ(θ_i − θ̄)²)`;
  - leave-one-cluster-out min / max and whether all of them are positive;
  - per-seed signs;
  - `excludes_zero_all_three`.
- **Clusterings:**
  - `chemotype`: 38 nominal.
  - `cluster_cpg`: connected components of chemotype ∪ publication ∪ gen19 copy group. It has 13 nominal clusters, Kish about 2.24, and the largest holds about 65% of rows.
    - It is the declared clustering for inference.
    - Every chemotype, publication and copy group lies in one component (tested).
    - Counts are in `clusters_*.json`.
- **Small samples.** With these effective counts, every interval is approximate. BP and BPG are not independent replications, and seed standard errors are not precision.

## Decision rules

See PROTOCOL §6:
- the margins: ΔR² ≥ +0.01 **and** Δbalanced-MAE ≥ 0.005 vs B1, on BPG and also on BP;
- the same vs B2, plus controls, for cell-conditioned learning;
- confirmation: percentile, BCa and jackknife-t all exclude zero for both endpoints, with the Kish floor of 3;
- the subgroup-harm rule;
- the finalist rule.
