# Round 3 (next_research_20261019) — protocol

Written 2026-10-19, before any round-3 configuration was scored on the outer dev folds. Task:
`../INSTRUCTIONS FOR NEW IDEAS ROUND 3.md`. The protocol and amendments A1–A4 of the earlier rounds
(`../PROTOCOL.md`, `../ATTEMPTS.md`) still apply; this file adds the round-3 rules. Changes after a
result has been seen go into `ATTEMPTS.md` as dated amendments that list what had been seen (at most
two registry amendments, inside the 24-configuration budget).

## 0. Sources and gaps

* `generations/gen21/AUDIT_2026-10-19.md` is **not in the repository**, and `/Users/lilshadix/Downloads/audit.pdf`
  does not exist on this machine (Windows, `D:\ml_separator_gh`). Every claim the brief attributes to
  the audit is re-derived from the original artefacts here: the PWL numbers, the DMDPhPDA defect, and
  the noise statements. Nothing is taken from it unverified.
* No `AGENTS.md` exists in the repository.
* Environment: `manifests/environment.json` (Python 3.14.5, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1,
  sklearn 1.9.0, torch 2.14.0+cpu; interpreter `.venv/Scripts/python.exe`). No dependency was changed.

## 1. Data, target, cohort (unchanged)

* V1 = the frozen gen13 cohort through gen20's loader: 521 cells. Manifest `3f11ac4c…`, log D matrix
  `f0be7fb3…`, and BP/BPG fold plans `56e6fded…` / `13632a17…` are identical to rounds 1–2
  (`manifests/inputs_sha256.json`, which also holds the content digests of all 50 cache files).
* Target `logSF(A/B) = log10 D(A) − log10 D(B)`, A lighter; strict adjacency |ΔZ| = 1 only (Nd/Sm is
  not adjacent). Every strict-adjacent pair of a cell is a query, fixed before any run.
* **Primary regime: V1, C3, k = 3 (four measured metals), gen20's DOPT_Q as the fixed reference
  policy.** C3 = cells with k_max ≥ 3: 229 cells, 1,802 pairs per seed. No scored pair has both ends
  read (asserted on every row).
* Separate regimes, never mixed with the primary: k = 0, 1, 2; C6 with k = 5, 6; any new policy; V2.

## 2. Hold-out and nesting (unchanged)

* Dev = gen20's five discovery seeds; designs BP (chemotype hold-out with test publications removed)
  and BPG (plus cross-publication copy groups). Inner fitting uses the four inner chemotype blocks of
  the outer training fold, with the design mask re-applied and a nested covariance (gen21 cache).
* Final check: the withheld seeds (`../manifests/final_seeds_commitment.txt`, `30ec7c0d…`). **Not
  opened in this round.** A freeze manifest and a guarded runner are prepared; the runner refuses to
  run without an explicit permission flag and refuses a second run.

## 3. Metrics

* Primary: pooled R² of log SF per split seed, then the mean over seeds (historical definition).
  Also MAE, RMSE, bias, direction accuracy (FLAT = 0 as the constant baseline), extractant- and
  chemotype-macro MAE; DGA (sc009) / non-DGA; one end read / both ends hidden; 12 pair types and
  Pr/Nd. Negative R² is kept; undefined R² is NaN with a reason. Thresholds are compared on unrounded
  values.
* Paired ΔR² / ΔMAE on identical rows and supports against **PWL(Z), PWL(R) and HBK@ML** (and RIDGE2).
  Contrasts are signed so that positive favours the candidate. Intervals: chemotype-blocked bootstrap,
  nominal 95 % (about 90 % real coverage at ~4 effective chemotypes), with BCa reported beside the
  percentile interval. Seed counts measure partition stability, not replication.
* log D of unread metals is a diagnostic only.

## 4. Fixed baselines (scored with every configuration, on the same rows and supports)

PWL(Z), PWL(R) (radius = Shannon CN8, the table already behind the project's physics basis; declared
before any run), LINE, RIDGE2, HBK@ML, zero-shot G14, FLAT. PWL interpolates the read log D values,
holds them constant beyond the extreme reads, and keeps read metals exact. These are reproductions,
not hypotheses, and do not count against the budget.

## 5. Budget and registry

`EXPERIMENT_REGISTRY.csv`: at most 24 new configurations scored on outer dev, counting combinations,
policies, ablations, data versions, the larger budget and the negative control. At least four are
combinations or their controls. Two slots are reserved for up to two dated amendments. Each
configuration's inner tuning space is listed there; optimiser steps are not hypotheses. The registry
is closed when it is exhausted, when results plateau, or when resources run out. There is no
automatic extension to chase 0.50.

## 6. Information contracts

* Predictors see only the read metals' log D (`Observed.y_masked`), the star support, the cell's metal
  identities and its query set, plus quantities fitted on the outer training fold. Chemistry features
  are cell metadata available before any measurement (structure, conditions), standardised on the
  outer training cells. Label-derived columns (replicate statistics, coefficients, n_metals/metals
  missingness, identifiers) are excluded; see the inventory in `ATTEMPTS.md`.
* Learned correctors are trained on **training cells masked with real read sets**: DOPT_Q (nested),
  UNIFORM and seeded RANDOM legal sets at the same budget. Every mask of a cell stays in that cell's
  split, and a cell's total weight is its chemotype-balanced weight, independent of its number of masks.
* Inner selection (λ, kernel, noise multiplier, feature set, stacking weights) uses inner-validation
  hidden queries predicted from reads chosen with that block's own nested covariance. No fitted
  quantity of an inner prediction sees that block's labels.
* A new policy is label-free: it uses metal identities, queries and train-fitted quantities. It keeps
  gen20's legality (an independent read set, k_max reachable) and is scored together with its
  crossings (new method × {DOPT_Q, new policy}, PWL(Z/R) × {DOPT_Q, new policy}, HBK × new policy).
  Adaptive (reading-dependent) policies would be a separate regime and are not in this registry.

## 7. Selection rule (declared before any outer run)

A configuration on V1 / C3 / k = 3 is a **candidate** only if, on identical rows and supports, it:
(a) beats **both** PWL(R) and PWL(Z) in R² **and** MAE in ≥ 4/5 seeds under BP, with the same signs
under BPG;
(b) is not worse than HBK@ML in R² by more than 0.005 (BP and BPG);
(c) passes every leakage check of §8.
Among candidates, the one with the highest BP R² is chosen. A simpler method is preferred when the R²
difference is < 0.005, and methods whose R² contrast interval includes 0 are called
indistinguishable. If no configuration satisfies (a), round 3 reports that **no learned skill beyond
interpolation** was shown. The frozen reference then becomes the better of {PWL(R), HBK@ML} under
the same rule with (a) dropped.

## 8. Leakage checks (every candidate, all folds of both designs)

1. Fold isolation: cells, extractants, chemotypes and publications (copy groups under BPG), also inside
   the nested fits.
2. Training-only fits: poisoning every non-training cell's log D leaves every fitted quantity unchanged
   (tests); this includes closures over `bench.Y` and label-derived cache fields.
3. Cache rebuild with the held-out labels replaced by random numbers leaves the allowed cache fields,
   the read sets and the predictions unchanged (script `scripts/r3_cache_poison.py`).
4. Intervention: every unread held-out log D replaced at every k → bit-identical predictions and read
   sequences (`results/<ID>/intervention_<design>.json`).
5. Every scored pair has at least one hidden end; no reading is supplied from other cells or copies.
6. Label-shuffle negative control for the best learned candidate. Training residual targets are
   permuted between training cells (within pair type), and the **added learned gain over PWL on the
   same supports** must vanish.

## 9. Data versions

V1 is the primary benchmark. V2 applies confirmed corrections and quarantines. Its rules are written
in `DATA_QUALITY_REPORT.md` **before any V2 run**, from source evidence only: never from curve
smoothness, model predictions or test residuals. V2 is a new benchmark: every baseline is rerun on it,
and gains relative to V1 are not attributed to a model.
