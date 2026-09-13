# gen16_protocol: few-cluster inference, BP1/BP1X, identifiability, power

This directory audits the programme's statistics on the corpus's chemotype clusters (written around 2026-09-09; seed constants 20260909). It asks whether
the gen13/gen14 chemotype percentile bootstrap keeps its nominal size, what reporting only design BP leaves out, and whether the order-dependent nested ANOVA
and "Kish n_eff 11.7" hold up. Side questions: what designs that let the model see the laboratory (BP1/BP1X) show, whether a ligand's effect can be told apart
from its laboratory's, whether the limit is the model or the number of chemotypes, and what effect the corpus could have detected. There is no report: the
questions come from the script docstrings, and eight scripts, three modules and `results/` are the whole record. What it found, with each number's regime:
- **Size** (simulated Gaussian nulls on the real 82-extractant / 40-chemotype membership, 2000 sims, ICC 0-0.6): the percentile bootstrap
  rejects at 0.0705-0.0815 against a nominal 0.05. The restricted wild cluster bootstrap-t (WCR) rejects at 0.041-0.053, and CV1-t at 0.064-0.113.
- **gen14 claims** (BP): claim 2 (LOGIT_TOPO39 - G13_ET_TOPO39 macro direction accuracy, 82 extractants) goes from p 0.006 to WCR 0.0233. Its point estimate
  +0.0522 is below its own MDE80 of 0.0892. Claim 1 (G13_FULL_MODEL - G14_DIR_HARD MAE, 90 extractants) has WCR p 0.0721.
- **G14 - FLAT** (extractant-macro MAE of pairwise log SF, BP, 5 seeds, 90 extractants in 45 chemotypes): +0.0884 with WCR p 0.0593
  (percentile p 0.0468). MEAN_CURVE - G14 is +0.1217 with p 0.0071.
- **Seeing the laboratory** (same arm and cohort, positive favours BP1/BP1X): for G14, BP@S - BP1 is -0.0025 (p 0.887; 42 extractants, 25 chemotypes),
  and BP@SX - BP1X is +0.0205 (p 0.0198; 21 extractants, 12 chemotypes).
- **Identifiability** (289 well-determined cells): 75 of 82 extractants were measured in one publication only, and the aliased shared R2 is 0.581 of a
  joint 0.829. For the same ligand in another laboratory (27 units, 7 ligands, 3 chemotypes), the leave-one-publication-out R2 on mean amplitude is 0.0046.
- **Chemotype learning curve** (BP, G14 macro MAE): 0.613 at 6 chemotypes, 0.506 at 24 and 0.500 at full. The power-law asymptote sits on its bound of 0.
- **Detectability** (gen13 cohort, assumed ICC 0.72, n_eff 34.26): the minimum detectable between-chemotype R2 is 0.196 with 1 predictor and 0.266 with 3.

## Status

- **Unreviewed, exploratory.** It was committed without review in fdb1e15 (2026-09-10, "Commit prior-session work that was uncommitted in
  the tree"). [`../gen16_leads/STATUS.md`](../gen16_leads/STATUS.md) lists `gen16_protocol/` as uncommitted prior-session work outside the
  gen16 brief, and `../gen16_leads/results/env/ENV.md` §5 lists it as untracked at df63929. There is no pre-registration.
- **Incomplete runs.** `results/g16_size.log` stops after the ICC 0.6 row, so `g16_size_study.csv` was never written. `results/g16_variance.log`
  stops at the REML header, so no variance-components table, Nakagawa R2 or Shapley split exists. No current script writes
  `results/g16_variance_components_profile.csv`, so its provenance is unverified.
- **Moved.** Commit 155dc6c moved the directory from the repository root with no content change, so at 155dc6c the scripts still use the pre-move
  paths. The follow-up relocation commit re-points them (`parents[3]`, `ROOT / "generations" / ...`).

## Read first

1. This README. Then `scripts/g16_size.py` with `results/g16_size.log` (why every p-value here is WCR), and the docstrings of `gen16/clusterboot.py` and `gen16/designs.py`.
2. `scripts/g16_dir_signflip.py`, `g16_bp1.py`, `g16_identifiability.py`, `g16_curve.py` and `g16_detectability.py`, each next to its CSVs.
3. [`../gen15_curve/GEN15_REPORT.md`](../gen15_curve/GEN15_REPORT.md) §1a and
   [`../gen16_leads/DECISION_REPORT.md`](../gen16_leads/DECISION_REPORT.md) §1, §5 and §6, before quoting anything from here.

## Key files

- `gen16/clusterboot.py` contains WCR (a chemotype sign-flip test for an intercept-only paired contrast), CV1/CV3 SEs, the test-inversion CI,
  the old percentile block bootstrap, the CSS effective cluster count and `size_study`.
- `gen16/designs.py` contains `bp1_folds`, `bp1_coverage`, `thin_folds` and `run_folds` (gen15 valuebench scoring on a supplied fold list).
- `gen16/variance.py` contains crossed REML, profile and parametric-bootstrap CIs, Nakagawa-Schielzeth R2 and the exact LMG/Shapley split.
- `results/`: files are named `g16_bp1_*`, `g16_curve*`, `g16_identifiability_*`, `g16_oof_amplitude_*`, `g16_detectability.csv`,
  `g16_dir_signflip_BP.csv` (the script's docstring says `g16_dir_signflip.csv`), plus the two logs.

## How to run (from the repository root; not run for this README)

Every command writes into `results/` and overwrites the committed files there, so copy them first.

    .venv/bin/python generations/gen16_protocol/scripts/g16_size.py 2000 999     # size study
    .venv/bin/python generations/gen16_protocol/scripts/g16_dir_signflip.py BP   # gen14 claims 1 and 2
    .venv/bin/python generations/gen16_protocol/scripts/g16_bp1.py               # add --full for G13_FULL
    .venv/bin/python generations/gen16_protocol/scripts/g16_curve.py 5           # draws per budget
    .venv/bin/python generations/gen16_protocol/scripts/g16_identifiability.py
    .venv/bin/python generations/gen16_protocol/scripts/g16_oof_amplitude.py BP  # before g16_variance.py
    .venv/bin/python generations/gen16_protocol/scripts/g16_variance.py 300      # recorded run never finished
    .venv/bin/python generations/gen16_protocol/scripts/g16_detectability.py

`PYTHONPATH=src` is not needed, because `gen13sep/paths.py` adds `src/` itself. `generations/gen14_direction/cache/bench.pkl` is gitignored;
if it is missing, the first bench load rebuilds the frozen gen13 bench.

## Tests

None. `generations/tests/test_generations_relocation.py` covers only the gen12 family.

## Dependencies

- `generations/gen13_separation` provides the gen13sep modules amplitude_bench, splits, metrics and wildcluster, plus `metrics/BP_all/per_extractant.csv` and
  `analysis/stage2/d6_condition_law/d6_cell_amplitudes.csv`. `generations/gen14_direction` provides gen14.dirbench and gen14.models, plus `results/g14_value_per_extractant_BP.csv` (claim 1).
- `generations/gen15_curve` provides gen15.arms and gen15.valuebench (Ctx, MIN_METALS). From `src/`, the scripts use `lanthanide_separation.gen6.cohorts.seeded_group_kfold`.
- The bench rebuild alone also reads `dataset with 3D structures/`, `runs/gen7_architecture/cache/chemistry_map.parquet`,
  `runs/gen6_provenance/provenance_table.parquet` and `generations/gen12_2_eu_pred/features/`.

## Caveats and later corrections

- **Two cohorts.** Size, identifiability, claim 2, OOF R2 and variance use the 289 well-determined cells (n_metals >= 5; 82 extractants,
  40 chemotypes, 41 publications; Kish 10.07). The BP1 board, the learning curve, claim 1 and detectability use the frozen gen13 cohort (521 cells,
  90 extractants, 45 chemotypes; Kish 11.67).
- **Size study.** It uses Gaussian simulations, not real MAE differences; `gen13_separation/metrics/BP_all/inference_calibration.csv` (45 clusters, 3000 sims)
  gives 0.071 for the percentile bootstrap at ICC 0.72. `g16_size.py` and `g16_detectability.py` hard-code ICC 0.72, but this directory's REML profile puts the
  chemotype share at 0.313 [0.139, 0.711] and the publication share at 0.392. At ICC 0.5, n_eff is 25.39.
- **Claim 2 prediction failed.** The docstring predicted a p floor near 0.03 from a few active chemotypes. The result has 13 active chemotypes and a
  floor of 0.00024. The p-value still rises, to 0.0233 under WCR and 0.0989 under CR2-t on Bell-McCaffrey dof.
- **Learning curve.** Budgets 28 and 32 average 26.8 and 27.5 training chemotypes against 27.63 at full, so the flat slope test (G14 p_wcr
  0.265 and 0.409) says nothing about adding chemotypes. Cell counts are not held fixed: budget 9 averages 70 cells and budget 6 averages 104. Both power-law fits
  sit on parameter bounds, and the bands in `g16_curve.csv` are percentile cluster bootstraps, not WCR.
- **BP1/BP1X are conditional designs.** The extractant, or the (publication, extractant) pair, is on both sides of the split on purpose, so report them next to BP,
  never instead of it. The BP1X board scores 22 extractants, while coverage and BP@SX use 21. The CSS G_star values (about 1.3-4.2) are order-of-magnitude diagnostics.
- **Identifiability** is in-sample with saturated indicators. The LOPO check has 27 units, and 5 of its 7 ligands are sc009 diglycolamides.
- **Oracle gaps are in-sample.** GEN15_REPORT §1a (47499b7, earlier, not applied here) moves "own a, constant b" from 0.3191 to 0.3370 leave-pair-out, so the G14 - O_AMP
  gaps here (0.178 BP, 0.155 BP1, 0.129 BP1X) and the "oracle 0.322" text in `g16_curve.py` and `g16_detectability.py` overstate the headroom (DECISION_REPORT §5 concurs).
- **Chemotype curves measure volume.** In DECISION_REPORT §6, a featureless biggest-chemotype-first order reproduces the A-optimal gain (SIZE - RANDOM +0.045
  under BP; AOPT - SIZE +0.005, p = 0.36). `g16_curve.py` does not hold cell count fixed. Table 6a's RANDOM row (0.626 at 6) comes from different draws.
- **Inference not adopted.** `../gen16_leads/PRE_REGISTRATION.md` (sealed) §0 and DECISION_REPORT §1 keep `gen13sep.inference.paired_contrasts` and report
  G14 - FLAT = +0.0884 [+0.0010, +0.1473], p = 0.047 under BP, against p_wcr 0.0593 here. Neither document resolves this.
- **Tension, not a correction.** `gen17_pairdiff/results/g17_summary.txt` §5 is a LOPO analysis on the same 289 cells, not BP, and is unreviewed. It finds that copying the
  extractant's mean curve from another paper (0.316 macro MAE of log SF) is about as good as copying it from the same paper (0.307). The same-ligand, other-lab R2 here is
  0.0046 on mean amplitude. The metric, target and unit differ.
