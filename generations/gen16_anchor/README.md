# gen16_anchor: anchor regression with the publication as the anchor

A gen16 side study, written around 2026-09-09. It fits anchor regression (Rothenhäusler, Meinshausen,
Bühlmann & Peters, JRSS-B 2021) with publication identity as the anchor. Gamma runs from 0 (within
publication, fixed effects) through 1 (pooled) to large values (between publications, the IV limit).
The study asks whether this predicts the curve amplitude `a`, and so pairwise log SF, better zero-shot
than gen14's deployed direction x constant-magnitude model (`G14_DIR_HARD`). It also asks whether any
within-publication signal exists, and whether anchoring only the magnitude |a| helps. Unless stated
otherwise, every number is extractant-macro MAE of pairwise log SF under design BP, averaged over 5
split seeds, on 90 extractants. **No anchor arm beats `G14_DIR_HARD` (0.5001); no contrast passes P1.**

## Status

**Unreviewed, exploratory.** The directory was not written in the gen16 session.
[../gen16_leads/STATUS.md](../gen16_leads/STATUS.md) found it uncommitted. Commit fdb1e15 (2026-09-10)
committed it without review, stating it "has not been reviewed or re-run". There is no report,
pre-registration, confirmation seed or test, and gen16's decision report does not use it. Commit
155dc6c (2026-09-13) moved it from the repository root; the scripts carry the matching path edits.

## Read first

1. The `scripts/g16_anchor_fast.py` docstring: the estimator, the gamma limits, the fixed-alpha confound.
2. `results/g16f_anchor_BP.csv` and `results/g16f_contrasts_BP.csv`, then `g16_within.py`,
   `g16_maganchor.py` and their CSVs. Read the v1 `g16_anchor.py` last.

## Key files and numbers

- `scripts/g16_anchor_fast.py` writes `results/g16f_anchor_BP.csv`, `g16f_alpha_BP.csv` and
  `g16f_contrasts_BP.csv`. Alpha is re-chosen at every gamma by exact leave-one-publication-out
  (LOPO). Blocks: COND_MA (64 condition + 8 mass-action columns) and TOPO39. `G14_DIR_HARD` is
  0.5000794414203691, the gen16_leads Phase 0 anchor to the last digit. The best arm is
  `ASIGN_TOPO39_g1000` (sign only) at 0.5715; G14 minus it is -0.0714 [-0.1127, -0.0104], p = 0.018
  (negative: the anchor arm is worse). The worst is `ARIDGE_COND_MA_g1000` at 0.7546. All 41 contrasts are negative.
- `g16_maganchor.py` writes `g16_maganchor_BP.csv`: gen14's out-of-fold direction times an
  anchor-ridge |a| (alpha 10, clipped to [0.05, 3.0]). The best arm, `MAGANCH_TOPO_MASSACT_g0.05`,
  scores 0.5107, against 0.5001 for G14 and 0.4864 for `DIR_ORACLE`. No contrast file exists.
- `g16_anchor.py` (v1: 209 lean columns, alpha fixed at 30) writes `g16_anchor_BP.csv`, which has no
  G14 row. Best `ANCH_g4_SIGN` 0.5991; `ANCH_g0` blows up to 6.18 (seed sd 7.41).
- `g16_within.py` writes `g16_within_between_r2.csv`: in-sample ridge R2 of the signed `a` on cells
  with n_metals >= 5 (alpha 30, no folds). Only MASSACT8 has within R2 above between R2 (0.0835 vs
  0.0415); COND64 is pooled 0.5752, within 0.1392, between 0.5629. The script, and its vectorised
  twin `g16_wcr.py`, run a restricted wild cluster bootstrap (Webb, B = 9999) whose p-values are only printed.
- `scripts/g16_anchor.py` is the slow refit version with an ALOGIT arm and no committed output; it is
  not the root-level `g16_anchor.py`. `scripts/g16_gamma_probe.py` writes `results/g16_gamma_probe.parquet`.
  `scripts/g16_anchor_limits.py` (gamma 0/1/1000) would overwrite `results/g16f_*_BP.csv`.
- `g16_anchor2.py` runs the alpha x gamma grid and builds `pairs_BP.pkl`, which `g16_maganchor.py`
  reads. The cache is gitignored and absent (digest in `PAIRS_BP_PKL.md`), and its CSV is not committed.

## How to run

Run from the repository root. Each script overwrites the outputs listed above.

    .venv/bin/python generations/gen16_anchor/scripts/g16_anchor_fast.py BP --blocks=COND_MA,TOPO39
    .venv/bin/python generations/gen16_anchor/scripts/g16_anchor.py BP --quick
    .venv/bin/python generations/gen16_anchor/g16_within.py
    .venv/bin/python generations/gen16_anchor/g16_anchor2.py BP LEAN209   # before g16_maganchor.py
    .venv/bin/python generations/gen16_anchor/g16_maganchor.py

- Pass `--blocks=`, with `=`. The docstring's space form, like no flag, runs all three blocks incl. ALL111.
- Only `scripts/g16_anchor_fast.py` and `scripts/g16_anchor.py` locate the repository from `__file__`.
  The others hardcode the Windows clone `D:\ml_separator_gh` and will not run on macOS until it is changed.
- `generations/gen14_direction/cache/bench.pkl` is gitignored; `gen14.dirbench.load` builds it on first use if it is missing.

## Tests

None. No test covers this directory, and `generations/verify_relocation.py` does not check it. The
only internal check: `G14_DIR_HARD` in both boards equals the G14 BP value pinned at 0.5001 by
[../gen16_leads/tests/test_anchors.py](../gen16_leads/tests/test_anchors.py).

## Dependencies

- [../gen13_separation/gen13sep/](../gen13_separation/gen13sep/) provides `amplitude_bench`,
  `splits`, `metrics` and `inference`; its `paths.py` puts `src/` on `sys.path`.
- [../gen14_direction/gen14/](../gen14_direction/gen14/) provides `dirbench` and `models`.
- A bench rebuild reads `dataset with 3D structures/`, `runs/gen7_architecture/cache/chemistry_map.parquet`
  and `runs/gen6_provenance/provenance_table.parquet`.

## Caveats and later corrections

- Only design BP was run, and ALL111 was not committed. No within-publication p-value was saved.
  Contrast p-values come from the percentile chemotype bootstrap, uncorrected over 41 contrasts, and the
  unreviewed [../gen16_protocol/results/g16_size.log](../gen16_protocol/results/g16_size.log) (Gaussian
  simulations, 2000 sims, an incomplete run) puts that test's size at 0.0705-0.0815 against a nominal 0.05. No arm is rescued, since every contrast favours G14.
- Alpha is fixed in `g16_anchor.py` and `g16_maganchor.py`, and `g16_anchor2.py` does not re-choose it
  per gamma, so their gamma paths are partly ridge paths. In `results/g16f_alpha_BP.csv` the median
  LOPO alpha for COND_MA is 1000, the top of the grid, at every gamma > 0 (25 seed-fold rows).
- The fixed-alpha gamma = 0 blow-ups look like a bug (read from the code, untested): the within transform
  zeroes publication means, but prediction adds back the untransformed grand mean. The fast path is stable.
- Between R2 is fitted on 41 publication means with up to 209 columns: LEAN209's 0.9259 is overfitting.
- `MEAN_CURVE` (0.6217) is not a baseline, because FLAT scores 0.5885 under BP
  ([../gen16_leads/DECISION_REPORT.md](../gen16_leads/DECISION_REPORT.md) section 11, guardrail 7).
  `ANCH_g4_SIGN` beats MEAN_CURVE and is still worse than FLAT.
- The headroom of 0.322 that `g16_maganchor.py` cites is an in-sample oracle, and those are optimistic.
  [../gen15_curve/GEN15_REPORT.md](../gen15_curve/GEN15_REPORT.md) section 1a: 0.3191 becomes 0.3370 leave-pair-out.
- GEN15_REPORT.md section 8 had already closed within-publication conditions (`condfe`, best BP
  0.4946). Unreviewed [../gen17_pairdiff/results/g17_summary.txt](../gen17_pairdiff/results/g17_summary.txt)
  (LOPO, 289 cells, not BP) finds no within-publication signal that transfers.
- gen16 adds no zero-shot skill. The deployed predictor remains
  [../gen15_curve/scripts/g15_predict.py](../gen15_curve/scripts/g15_predict.py).
