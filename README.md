# Lanthanide separation: prediction for chemically unseen extractants

This repository predicts how well a solvent-extraction ligand (an extractant) separates the
lanthanides, for extractants the model has never seen. Two targets are modelled: the distribution
ratio `log D` of each metal (the "level", gen5–gen12), and the pairwise separation factor
`log SF(A/B) = log D(A) − log D(B)` along the 14-lanthanide series (gen2–gen4, and gen13 onwards).
The evaluation is designed to be leakage-safe. Whole chemotypes are held out, and in the strictest
design (BP) the held-out cell's publication is also removed from training. Every claim is
pre-registered, compared with a no-model floor and tested with a chemotype-blocked bootstrap. The
programme reports well-supported nulls as results.

## Current state (2026-09)

Every number below carries its regime, and a number without one should not be quoted. **BP** means
the chemotype is held out *and* every training cell from the held-out cell's publication is dropped.
**Extractant-macro MAE** is the error averaged within each extractant, then across extractants.
Five split seeds unless stated.

**Zero-shot, separation factors.** Gen13 cohort (521 cells, 90 extractants, 45 chemotypes), design
BP, extractant-macro MAE of pairwise `log SF`:

- Predicting no separation (`FLAT`) scores **0.589**. The deployed gen14 model (an L2 logistic
  direction call on 39 donor-topology counts, times a constant magnitude) scores **0.500**. The
  honest zero-shot gain is **+0.088**, 95 % CI [+0.001, +0.147], p = 0.047 (gen15 §1, reproduced
  exactly by gen16). An unreviewed wild-cluster re-inference in `generations/gen16_protocol/`
  gives p = 0.059 for the same contrast.
- The corpus mean curve (0.622) is *worse* than `FLAT`, so gains quoted against it overstate the
  value. Gen14's +0.122 is one example.
- **What the model can answer: which way an extractant separates.** Macro direction accuracy is
  0.821 over 82 well-determined extractants (BP). On strong pairs (|log SF| >= 0.3) the pair-sign
  accuracy is 0.812, against 0.630 for "always the heavier lanthanide".
- **What it cannot answer: which ligand to choose.** Picking the best extractant for a target pair
  succeeds 0.023 of the time, against 0.017 for a random pick (BP). What remains is the magnitude,
  and no structure-based magnitude prior beats a constant.

**Zero-shot, `log D` levels.** Most of the error sits in each extractant's overall level.
- Gen7 (level cohort, chemotype hold-out, macro MAE per ECFP cluster, 3 seeds): a model given no ligand
  information scores 1.088, against 0.998 for the best ligand-aware model. Those folds had no
  publication mask, so even this ~0.09 may partly be a publication fingerprint.
- Gen12 (europium, design B, 183 extractants): the best model significantly beats only
  predict-the-mean. Its edge over the same learner given only conditions (+0.086) has an interval
  that includes zero.

**A few chosen measurements.**
- Separation factors, BP. Measuring the widest pair takes gen14's error from 0.439 to 0.231 on
  identical held-out pairs (gen15 §4). Three pairs chosen by D-optimal design reach **0.163–0.170**
  across all five designs (gen15 §5, a different scoring set). At k = 1, measuring the widest pair
  instead of a random pair is worth +0.103, more than the entire zero-shot gain.
- Once one pair is measured, ligand chemistry adds nothing significant: the gen14 prior beats a
  straight line through the measured pair by only +0.008, p = 0.35. The +0.065 gain over that line
  at k = 3 comes from the residual covariance learned from the corpus.
- `log D`. The frozen gen10 pipeline goes 1.036 → 0.654 → 0.441 macro MAE at k = 0 / 1 / 5
  (C-COMMON, 99 extractants, chemotype hold-out).
- Gen16 found that the deployed measured-mode covariance is indefinite, and that nominal 90 %
  intervals cover 98.8 % at k = 3 (BP). The intervals are too wide.

**What gen16 closed** ([decision report](generations/gen16_leads/DECISION_REPORT.md)).
- **One claim confirmed** on five withheld seeds. For each target pair, set aside candidates whose
  direction call disagrees with the requested direction. Under BP this cuts the expected number of
  measurements to the first useful candidate from 6.91 to 4.92, and saves +1.909 on the withheld
  seeds. It holds only *between* laboratories: within one publication the saving is +0.0046 with
  an interval containing zero. Rebuilt fold-pure, it is 16.4 % rather than 28.8 %.
- **Closed:**
  - The xTB thermodynamic cycle: ρ = −0.084 from an estimator with reliability 0.80, family-wise
    p = 0.94.
  - The curvature headroom: +0.029, failing its own gate.
  - Corpus acquisition order: a featureless "biggest family first" order reproduces it.
  - Covariance and calibration: nothing above the 0.02 margin, and two defects found.
- No zero-shot skill was added. The deployed predictor remains
  `generations/gen15_curve/scripts/g15_predict.py`.
- The binding constraint is chemical coverage: Kish effective sample size is 11.7 chemotypes.
  Measuring a second lanthanide on one compound from each of the 53 absent chemotypes would raise
  it to 27.4. This is a design-capacity figure, not a measured gain.

## Repository map

```
.
├── src/lanthanide_separation/   gen2–gen11 library code (pairs, ablation, gen3_*, levels, gen6/ … gen11/)
├── scripts/                     gen2–gen11 runners and analysis scripts
├── tests/                       gen2–gen11 pytest suite ("slow" tests need the built cohort or run artefacts)
├── slurm/                       SLURM job and submit scripts for the gen2–gen6 cluster runs
├── runs/                        gen2–gen11 run artefacts and reports, incl. the frozen gen6–gen11 cohort
├── generations/                 gen12 onwards, one self-contained directory each, plus the 2026-09 relocation record
├── docs/                        documentation index, gen2–gen11 briefing, protocols, results, audits, figures
├── dataset with 3D structures/  frozen experimental bundle (log D rows, 3D geometries, 2D descriptors)
├── dataset_all_metals/          all-metals archive; gitignored external build (a symlink in some checkouts)
├── gen3_protocol.json           frozen gen3 protocol; pins the bundle SHA-256; read at the repository root
├── pyproject.toml               package lanthanide-separation (src layout), extras [deep] and [gen3]
├── requirements.txt             core dependencies
├── requirements-deep.txt        core + torch
└── requirements-gen3.txt        exact pinned scientific environment (Python 3.11.11, Linux x86_64)
```

- **Two layouts.** gen2–gen11 are spread across `src/`, `scripts/`, `runs/`, `slurm/` and `tests/`,
  and were never moved. gen12 onwards are self-contained directories under `generations/`, each
  with its own library package, scripts, results and reports.
- **Fixed locations.** `dataset with 3D structures/` (spaces included) is named literally by code
  and manifests. `gen3_protocol.json` must stay at the root because
  `scripts/run_gen3_benchmark.py` and `scripts/gen11_self_audit.py` read it as
  `REPO_ROOT / "gen3_protocol.json"`, and the gen3 runner refuses a dataset whose SHA-256 differs
  from the one pinned there.

## Where to start reading

| if you want … | read |
|---|---|
| the current verdict and every standing claim | [generations/gen16_leads/DECISION_REPORT.md](generations/gen16_leads/DECISION_REPORT.md) |
| the validity contract every result must pass | [generations/gen16_leads/START_HERE.md](generations/gen16_leads/START_HERE.md) §1 |
| an index of gen12–gen17 | [generations/README.md](generations/README.md) |
| the documentation index and the gen2–gen11 map | [docs/README.md](docs/README.md) |
| gen2–gen11 in one self-contained briefing | [docs/HANDOFF_FOR_CHATGPT.md](docs/HANDOFF_FOR_CHATGPT.md) (written before gen11's matched stage; see `runs/gen11_transfer/GEN11_DECISION_REPORT.md`) |
| what may and may not be claimed from a number | [docs/figures/METRIC_AUDIT.md](docs/figures/METRIC_AUDIT.md) |
| to predict a new extractant's separation curve | [generations/gen15_curve/GEN15_REPORT.md](generations/gen15_curve/GEN15_REPORT.md) §5–§6 and `generations/gen15_curve/scripts/g15_predict.py` |
| the frozen `log D` pipeline and its limits | [runs/gen10_final/GEN10_FINAL_MODEL_CARD.md](runs/gen10_final/GEN10_FINAL_MODEL_CARD.md) |
| whether the evaluation leaks | [docs/audits/LEAKAGE_AUDIT.md](docs/audits/LEAKAGE_AUDIT.md) |

## Generation map

"Unreviewed" marks prior-session work that was committed without review. No reviewed report
depends on it.

| gen | question | verdict (one line) | where |
|---|---|---|---|
| 2 | Does 3D coordination geometry beat a 2D fingerprint for `log SF`? | No: adding 3D is significantly worse (−0.015 macro MAE, CI excludes 0), on extractant-grouped folds that carry an ECFP-homolog leak. | `docs/results/metal_site_descriptor_experiment.md`, `runs/ablation_all_*`, `runs/simplicial_*` |
| 3 | Do learner, residual or architecture changes help? | No: a valid negative result. | `gen3_protocol.json`, `runs/gen3_primary_20260815T181555Z/` |
| 4 | Candidate families, and the first k-shot study | From k = 2 the model adds nothing over a two-parameter no-model fit (`log SF`, 12 extractants). | `docs/results/gen4_candidate_study_20260816.md`, `docs/results/kshot_calibration_study_20260817.md` |
| 5 | Pivot to the `log D` level | The level model behaves like a near-neighbour lookup; 3D is worse by 0.10; mass-action features help only when the ligand is known. | `docs/results/gen5_levels_results_20260819_full4regimes.md` |
| 6 | Coverage or capacity? | Coverage: 61 added extractants give +0.163 macro MAE on identical test rows; a two-stage model is worse. | `docs/results/gen6_phase0_and_phase1_results_20260819.md`, `docs/results/gen6_phase2_results_20260819.md`, `runs/gen6_*` |
| 7 | Where is the ceiling? | Chemistry is worth ~0.09 (no-ligand 1.088 vs best 0.998); nothing beats a 1-NN level lookup; one measurement beats every modelling gain. | `docs/results/gen7_architecture_results_20260819.md`, `runs/gen7_architecture/` |
| 8 | Few-shot calibration and acquisition | Which point you measure is worth +0.101; the model flattens unseen titration slopes (2.57 true vs 0.12 predicted). | `docs/results/gen8_architecture_results_20260820.md`, `runs/gen8_architecture/` |
| 9 | Curve shape | Exploratory +0.011 gain (0.9695 vs 0.9807), later found to depend on the requested query set. | `runs/gen9_shape/decision_report.md` |
| 10 | Finalise and freeze the `log D` pipeline | Frozen; zero-shot error is dominated by the per-ligand level; k-shot frontier 1.036 → 0.441 (C-COMMON). | `runs/gen10_final/` |
| 11 | Does a multi-metal (mostly actinide) archive transfer? | Inconclusive by power (+0.049, interval includes zero); not shipped. | `runs/gen11_transfer/GEN11_DECISION_REPORT.md` |
| 12 | Europium `log D` zero-shot and few-shot | Zero-shot beats only predict-the-mean; few-shot passes; strict multi-lanthanide transfer −0.003. | `generations/gen12_eu_pred/` |
| 12.2 | Do coordination-topology descriptors predict the Eu level? | +0.060 over a bare fingerprint (p = 0.007); borderline over the full representation (+0.034, p = 0.092); one measurement erases it. `gen12_eu_pred_2/` is an unfinished early code draft with no results. | `generations/gen12_2_eu_pred/`, `generations/gen12_eu_pred_2/` |
| 13 | Separation factors from the lanthanide-axis curve | The curve model ties the row model (P1 −0.006); "conditions beat chemistry" was a publication fingerprint, which is why BP exists; lean bag 0.536 under BP. | `generations/gen13_separation/` |
| 14 | The curve as one bit (direction) and one scalar | Direction 0.821 vs 0.768 for gen13's trees (BP); MAE 0.500, stable across designs; the remaining error is magnitude. | `generations/gen14_direction/` |
| 15 | Honest floor, curvature, measured mode | +0.088 over `FLAT`; an in-sample oracle is not a ceiling; D-optimal k = 3 reaches 0.163–0.170; answers "which way", not "which ligand". | `generations/gen15_curve/` |
| 16 | Six pre-registered leads, refuted and confirmed | One claim confirmed (a between-laboratory direction filter); every other lead closed; no zero-shot skill added. | `generations/gen16_leads/` |
| 16 (anchor) | Anchor regression with the publication as anchor | Every arm is worse than gen14 under BP. *Unreviewed.* | `generations/gen16_anchor/` |
| 16 (protocol) | Few-cluster inference audit, conditional designs, learning curve | The percentile bootstrap rejects 7–8 % at nominal 5 %; some runs are incomplete; no report. *Unreviewed.* | `generations/gen16_protocol/` |
| 17 | Within-publication pairwise difference learning | Every arm is worse than predicting zero difference; leave-one-publication-out, not BP. *Unreviewed.* | `generations/gen17_pairdiff/` |

## Setup and tests

Run everything from the repository root.

```bash
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/pip install -e ".[deep]"    # + torch
.venv/bin/pip install -e ".[gen3]"    # + torch, catboost
```

- **Requirements files.** `requirements.txt` (core) and `requirements-deep.txt` (core + torch)
  mirror the extras. `requirements-gen3.txt` pins the exact frozen scientific environment.
- **Extra packages.** gen12 onwards also use RDKit, and some arms use XGBoost; neither is in
  `pyproject.toml`. CatBoost comes with the `[gen3]` extra.
- **Do not install TabPFN.** It downgrades scikit-learn and pandas and shifts results by about
  0.0014.
- **Do not upgrade scikit-learn to 1.10.** The frozen gen14 arm uses a deprecated argument.

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q          # gen2–gen11; add -m "not slow" for the fast set
.venv/bin/python -m pytest generations/gen12_eu_pred/tests -q
.venv/bin/python -m pytest generations/gen12_2_eu_pred/tests -q
.venv/bin/python -m pytest generations/gen13_separation/tests -q
.venv/bin/python -m pytest generations/gen16_leads/tests -q
.venv/bin/python -m pytest generations/tests -q             # relocation checks for the 2026-09 move
```

- **Known failures.** Two gen16 L6 cohort-fingerprint tests in
  `generations/gen16_leads/tests/test_l6_cohort_audit.py` (the `exact` and `relaxed` modes) fail on
  this macOS machine: the `exact` rebuild fingerprints as `854171b03411d901`, not the frozen
  `4c3c6628ea0be949`. Two gen9 tests in `tests/test_gen9_curve_objective.py` also fail in any
  checkout without `runs/gen9_shape/curves/curve_membership.parquet`, which is gitignored
  (`runs/**/*.parquet`) and written by `scripts/gen9_curves.py`.
- **Windows commands.** Many generation READMEs and docstrings show the Windows interpreter
  `.venv/Scripts/python.exe`. On macOS or Linux use `.venv/bin/python` with the `generations/...`
  path.
- **Scripts write tracked files.** Most experiment and analysis scripts write committed result
  files (boards, contrast CSVs, reports). Copy what you need to keep before re-running anything.

## Data and frozen artefacts

- **The bundle.** `dataset with 3D structures/` is the frozen experimental bundle, and its
  directory name is frozen too. `dataset.parquet` is SHA-256-pinned in `gen3_protocol.json`
  (`fefbefc6…`) and re-checked by the gen3 runner and by gen13's path module. The directory carries
  its own `checksums.sha256`.
- **Ignored and rebuilt, not committed:**
  - `dataset_all_metals/`: about 84 MB, built by the sibling dataset-builder repository. It is the gen11
    auxiliary archive and the gen12 multi-lanthanide input.
  - Out-of-fold prediction dumps, logs and figures under `runs/`. Their digests are in each run's
    manifest.
  - gen13 prediction dumps and fitted model blobs.
  - The gen14 bench cache (`generations/gen14_direction/cache/`), rebuilt on first load.
  - gen15 derived tables.
  - `dataset with 3D structures/ligand_pretrained_embeddings.parquet`.
  - `.gitignore` also keeps a legacy block for the pre-move paths.
- **Frozen cohorts.** Every number is defined against one of these:
  - `runs/gen7_architecture/cache/cohort.parquet`: fingerprint `bed178ec1a7a82b0`; 5,248 rows,
    152 extractants, 79 chemotypes; gen6–gen11. Committed despite the `runs/**/*.parquet` rule.
  - `generations/gen12_eu_pred/manifests/cohort.parquet`: fingerprint `2a364bb5264e9935`;
    1,329 cells, 183 extractants, 97 chemotypes; gen12 and gen12.2.
  - `generations/gen13_separation/manifests/cohort_{exact,relaxed,series}.parquet`: exact
    fingerprint `4c3c6628ea0be949`; 521 cells, 90 extractants, 45 chemotypes; gen13–gen17.
- **Hash manifests:**
  - `generations/gen12_eu_pred/manifests/manifest.json` and
    `generations/gen12_2_eu_pred/manifests/manifest.json` hash nearly every file in those
    directories, code included.
  - `generations/gen16_leads/PRE_REGISTRATION.md` carries a sealed body SHA-256, checked by
    `generations/gen16_leads/scripts/g16_prereg_hash.py`.
  - `generations/gen16_leads/results/MANIFEST.sha256` digests gen16's large excluded artefacts.
  - Result and manifest files (JSON, CSV, parquet) are historical records. Do not edit them, even
    where they contain old paths.
- **The 2026-09 move.** Commit `155dc6c` moved gen12 onwards from the root into `generations/`.
  Because the gen12 manifests hash code, the few path lines that had to change are recorded as
  exact text substitutions in `generations/RELOCATION.json`.
  `.venv/bin/python generations/verify_relocation.py` undoes those substitutions and checks the
  result against the hashes read live from the manifests.

## Conventions every number must follow

The full contract is [START_HERE.md §1](generations/gen16_leads/START_HERE.md#1-the-validity-contract-non-negotiable).

- **Name the regime.** State the target (`log D` level or pairwise `log SF`), the design
  (A, B, BR, BQ, BP), the cohort, the averaging unit, the seeds and k. For example, the same frozen
  gen10 model scores 0.9695 on C-FULL (one vote per ECFP cluster) and 1.0358 on C-COMMON (one vote
  per extractant). Pooled metrics do not select models. Effects below ~0.01 are within
  cross-machine reproducibility.
- **BP selects the deployed model; design B never selects.** Without a publication mask, the 64
  condition columns identify a held-out cell's publication with 94 % 1-NN accuracy. Quote BP for
  new chemistry from a new laboratory, and report all five designs.
- **`FLAT` (predict no separation) is the floor.** It scores 0.589 under BP. The corpus mean curve
  (0.622) is worse than doing nothing and is not a baseline. Also quote the increment over the
  cheapest sensible competitor, for example `NAIVE_LINE` in the measured mode.
- **Oracles must be leave-pair-out.** An oracle fitted to the rows it is scored on is not a ceiling:
  the two-coefficient oracle is 0.181 in sample but 0.274 leave-pair-out. Older in-sample
  "floors", such as gen10's oracle chain and gen12's perfect-level 0.334, are optimistic.

## Original README

The gen2-era README, in Russian, describes the three original 3D benchmarks: tabular Delta3D, the
siamese simplicial neural network and the A0–A6 ablation. It is preserved at
[docs/design/README_gen2_simplicial_ru.md](docs/design/README_gen2_simplicial_ru.md).
