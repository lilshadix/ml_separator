# Power and minimum detectable effect (stage 1; `scripts/r4_power.py`)

**Question.** If a candidate truly improves on the B1 reference, how likely is it to pass the sealed confirmation rule (PROTOCOL §6.3) on new, source-independent data? How many independent sources does that take?

**The rule:** ΔR² and Δbalanced-MAE vs B1 must both have their percentile, BCa and cluster-jackknife-t 95% intervals exclude zero, with Kish ≥ 3. The **full gate** also requires point estimates of at least +0.01 R² and 0.005 balanced MAE.

## Generative model

- **Sources.** A confirmation set is G new independent sources. Each one is a chemotype drawn with replacement from the 38 C3 chemotypes of REF's BPG out-of-fold rows (dev seed 104729). Each drawn source is its own cluster.
- **Effect.** A synthetic candidate improves on B1 by a true dev ΔR² of δ, with δ ∈ {0.001, 0.01, 0.02, 0.034}.
- **Replicates.** 400 Monte Carlo sets per cell, 999 cluster-bootstrap draws each.
- **Effect structures:**
  - **uniform:** every row moves toward the target by the same fraction;
  - **concentrated-sparse:** large errors improve, and only 30% of sources improve at all;
  - **empirical_B1, the planning scenario:** a replay of the real B0 → B1 gain (the Gd term), scaled to δ, so per-source gains vary and can be negative.

## Runs

Each run is kept in `results/power/`; the reasons for each rerun are in RUN_RECORD.

| run | change | status |
|---|---|---|
| 1 | three mechanical scenarios | superseded: every source improves, so power depends on the cluster count only |
| 2 | + empirical_B1 | the **optimistic bound**: a source's gain is a deterministic multiple of its rows' structure, so there is no per-source noise independent of the effect (review finding A-POW-2) |
| 3 | + effect-independent per-source noise κ_g ~ N(0, τ), τ = 0.065; + boundary null | boundary-null construction wrong (did not preserve SSE); size numbers invalid |
| 4 | boundary null fixed | the **heterogeneous bound** (current `power_grid.csv`) |

**How τ was set.** τ = 0.065 is half the between-chemotype SD of the relative SSE change between two near-equal development models (HIST_D7 vs B1). It was fixed before run 3. It may include small-sample noise of small chemotypes, which would make run 4 pessimistic. The truth probably lies between runs 2 and 4.

## Results: probability of passing the interval rule (empirical_B1)

| δ (ΔR² vs B1) | run 2, G = 20 / 40 / 80 | run 4, G = 20 / 40 / 80 |
|---|---|---|
| 0.001 | not run | 0.00 / 0.01 / 0.01 |
| 0.010 | 0.50 / 0.93 / 1.00 | 0.03 / 0.08 / 0.04 |
| 0.020 | 0.44 / 0.92 / 1.00 | 0.07 / 0.12 / 0.17 |
| 0.034 (B1's own size) | 0.33 / 0.84 / 1.00 | 0.14 / 0.27 / 0.34 |

**Full gate (intervals plus both margins), δ = 0.034:**
- run 2: 0.30 / 0.56 / 0.66;
- run 4: 0.13 / 0.23 / 0.25.

**Other run-4 numbers**
- **Kish floor.** About 30% of sets with G ≤ 20 fail the Kish ≥ 3 floor. Drawing the large sc009-like chemotype dominates a small set. This failure does not depend on the effect.
- **Joint null.** Both endpoints at zero pass in 1–3% of sets.
- **Boundary null:** an SSE-preserving reshaping of B1's errors (median ΔR² about 0, balanced-MAE gain about +0.004). This gives the size of the intersection–union rule.
  - It passes in 1.3 / 4.8 / 3.5 / 5.5 / 5.0% of sets at G = 5 / 10 / 20 / 40 / 80 (Monte Carlo SE about 1%). That is **about α = 0.05**, not conservative.
  - The R² endpoint's three-interval exclusion alone is 5–8% under this null, against a nominal one-sided 2.5%. So the "95%" cluster intervals undercover with few effective clusters.
  - Before any confirmation design, a Kish-based degrees-of-freedom correction of the jackknife-t should be considered (§6.8). It is not changed now.

**Balanced-MAE gain implied by each δ** (`margins_implied.csv`)

| δ | empirical_B1 | uniform | concentrated |
|---|---|---|---|
| 0.010 | 0.0015 | 0.0009 | 0.0004 |
| 0.020 | 0.0031 | 0.0019 | 0.0009 |
| 0.034 | **0.0055** | 0.0032 | 0.0015 |

## What this means

1. **The current corpus cannot confirm anything.** It has about 13 source clusters, Kish 2.2, one holding 65% of rows. B1's own development gain, the most replicated effect in the project, fails the small-cluster interval on it (STAGE0_REPORT).
2. **New sources needed.**
   - Under the optimistic bound (run 2), about 40 new independent chemotype-like sources give 80% power for gains of 0.01–0.034 R².
   - Under the heterogeneous bound (run 4), even 80 sources give only about 1 in 3 for a B1-sized gain. The sealed rule (percentile, BCa and jackknife-t all excluding zero, plus the Kish floor) is strict, and per-source heterogeneity like the development data's makes confirmation very expensive.
   - Planning number: **at least about 40 independent sources, and possibly well over 80**.
3. **The balanced-MAE margin (0.005) is the binding margin.** Under the realistic gain structure, it corresponds to about +0.034 R², which is B1's whole gain over interpolation. So:
   - a candidate with a true +0.01 R² gain essentially never meets both margins;
   - a B1-sized gain meets them in at most two thirds of runs, even with many sources.
4. **The margins stay as sealed.** They are not relaxed after this result (PROTOCOL §6.7). Any stage-1 gain below them is reported as exploratory, with its interval.
5. **For the data request:** one new laboratory campaign of a few systems cannot confirm a model gain. Its value is the noise ceiling and a first look at transfer.

**Limitations.**
- New sources are modelled as resampled chemotypes of the current corpus, so they inherit its error structure. Real new chemistry may differ.
- The template is one development seed of BPG.
- τ's calibration is a declared choice, not a measurement.
