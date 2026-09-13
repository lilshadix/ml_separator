# Documentation index

Everything written about this project, organised by kind. The gen2–gen11 code lives in `src/`,
`scripts/`, `tests/` and `slurm/`, and their experiment outputs live in `runs/`. From gen12
onwards each generation is one self-contained directory under `generations/` (code, tests,
reports and results together); start at [`../generations/README.md`](../generations/README.md).

```
docs/
├── HANDOFF_FOR_CHATGPT.md   the gen2–gen11 programme in one document
├── audits/      leakage and feature audits, written before any result
├── protocols/   pre-registrations — hypotheses and decision rules, frozen before the run
├── results/     what the runs found
├── design/      roadmaps and design notes, including the old gen2-era Russian README
└── figures/     the publication figure set's knowledge (renders deleted, see below)
```

## Start here

| if you want… | read |
|---|---|
| the repository overview | [`../README.md`](../README.md) |
| gen12 onwards, one entry per generation | [`../generations/README.md`](../generations/README.md) |
| the whole gen2–gen11 programme in one document | [`HANDOFF_FOR_CHATGPT.md`](HANDOFF_FOR_CHATGPT.md) — self-contained, gen2→gen11, written 2026-09-03 before gen12 |
| what may and may not be claimed from a number | [`figures/METRIC_AUDIT.md`](figures/METRIC_AUDIT.md) |
| whether the evaluation is sound | [`audits/LEAKAGE_AUDIT.md`](audits/LEAKAGE_AUDIT.md) |
| the frozen gen10 `log D` pipeline and its limits | [`../runs/gen10_final/GEN10_FINAL_MODEL_CARD.md`](../runs/gen10_final/GEN10_FINAL_MODEL_CARD.md) |
| the deployed separation-curve predictor (gen15, left unchanged by gen16) | [`../generations/gen15_curve/scripts/g15_predict.py`](../generations/gen15_curve/scripts/g15_predict.py) |

**Read `figures/METRIC_AUDIT.md` before quoting any headline number.** The same frozen model
scores macro MAE `0.9695` and `1.0358` depending only on which cohort and averaging unit is used.
Both are correct; a number without its cohort is not.

## Generation map

| gen | question | protocol | results | run artefacts |
|---|---|---|---|---|
| 2 | does 3D coordination geometry beat 2D? | `protocols/generation2_protocol.md` | `results/metal_site_descriptor_experiment.md` | `runs/ablation_all_*`, `runs/simplicial_*` |
| 3 | do learner / residual / architecture changes help? | `protocols/gen3_implementation_plan.md`, `gen3_runbook.md`, `../gen3_protocol.json` | — (valid negative result) | `runs/gen3_primary_20260815T181555Z` |
| 4 | candidate families + first k-shot study | `protocols/gen4_combined_protocol.md` | `results/gen4_candidate_study_20260816.md`, `results/kshot_calibration_study_20260817.md` | `runs/gen4_candidates_*`, `runs/kshot_calibration_*` |
| 5 | pivot to the `log D` level target | `protocols/gen5_levels_protocol_20260817.md` | `results/gen5_levels_results_20260818.md`, `..._20260819_full4regimes.md` | `runs/gen5_levels_*`, `runs/gen5_massaction_local_20260818` |
| 6 | coverage or capacity? | `protocols/gen6_diversity_protocol_20260819.md`, `gen6_phase2_protocol_20260819.md`, `gen6_phase2_runner_contract.md` | `results/gen6_phase0_and_phase1_results_20260819.md`, `results/gen6_phase2_results_20260819.md` | `runs/gen6_*` |
| 7 | where is the ceiling? | (in-run: `runs/gen7_architecture/block_a_diagnosis.md`) | `results/gen7_architecture_results_20260819.md` | `runs/gen7_architecture` |
| 8 | few-shot calibration + acquisition | `runs/gen8_architecture/protocol.md` | `results/gen8_architecture_results_20260820.md` | `runs/gen8_architecture` |
| 9 | curve shape | `runs/gen9_shape/protocol.md` | `runs/gen9_shape/decision_report.md`, `failure_analysis.md` | `runs/gen9_shape` |
| 10 | finalisation and freeze | (external brief) | [`GEN10_*.md`](../runs/gen10_final/) — reading order in [`runs/gen10_final/README.md`](../runs/gen10_final/README.md) | [`runs/gen10_final`](../runs/gen10_final/) |
| 11 | multi-metal actinide transfer | in code (`gen11/arms.py`, `analysis.py`); matched stage: [`MATCHED_PREREGISTRATION.md`](../runs/gen11_transfer/MATCHED_PREREGISTRATION.md) | [`GEN11_DECISION_REPORT.md`](../runs/gen11_transfer/GEN11_DECISION_REPORT.md) — rewritten 2026-09-04: inconclusive by power, gen11 does not ship; never quote `GEN11_DECISION_REPORT_SUPERSEDED_20260823.md` | [`runs/gen11_transfer`](../runs/gen11_transfer/) |
| 12 | Eu `log D` on chemically unseen extractants: zero-shot, few-shot, multi-lanthanide transfer | [`PRE_REGISTRATION.md`](../generations/gen12_eu_pred/PRE_REGISTRATION.md), [`DATA_AUDIT.md`](../generations/gen12_eu_pred/DATA_AUDIT.md) | [`DECISION_REPORT.md`](../generations/gen12_eu_pred/DECISION_REPORT.md) — zero-shot on new chemistry is not supported; only the few-shot hypothesis passes | [`generations/gen12_eu_pred`](../generations/gen12_eu_pred/) |
| 12.2 | do 2D coordination-topology descriptors predict an unseen extractant's intrinsic Eu level? | [`PRE_REGISTRATION.md`](../generations/gen12_2_eu_pred/PRE_REGISTRATION.md), [`COORDINATION_DESCRIPTOR_SPEC.md`](../generations/gen12_2_eu_pred/COORDINATION_DESCRIPTOR_SPEC.md) | [`DECISION_REPORT.md`](../generations/gen12_2_eu_pred/DECISION_REPORT.md) — the primary contrast is borderline and unresolved; one measurement erases the gain | [`generations/gen12_2_eu_pred`](../generations/gen12_2_eu_pred/) |
| 12.2 draft | unfinished early code draft of gen12.2 (bridge to gen12's folds, five level definitions) | — | none: no scripts, reports or results; cite gen12.2 instead | [`generations/gen12_eu_pred_2`](../generations/gen12_eu_pred_2/) |
| 13 | zero-shot separation factors from the 14-lanthanide curve | [`PRE_REGISTRATION.md`](../generations/gen13_separation/PRE_REGISTRATION.md), [`DATA_AUDIT.md`](../generations/gen13_separation/DATA_AUDIT.md) | [`DECISION_REPORT.md`](../generations/gen13_separation/DECISION_REPORT.md), [`STAGE2_REPORT.md`](../generations/gen13_separation/analysis/stage2/STAGE2_REPORT.md), [`STAGE3_REPORT.md`](../generations/gen13_separation/analysis/stage3/STAGE3_REPORT.md) — the curve model ties the row model; design-B gains were a laboratory fingerprint, so design BP selects | [`generations/gen13_separation`](../generations/gen13_separation/) |
| 14 | the separation curve as one bit (direction) and one scalar (magnitude) | — | [`GEN14_REPORT.md`](../generations/gen14_direction/GEN14_REPORT.md), [`results/TABLES.md`](../generations/gen14_direction/results/TABLES.md) — the direction call transfers; the remaining error is magnitude | [`generations/gen14_direction`](../generations/gen14_direction/) |
| 15 | the honest zero-shot floor, the curvature, and the measured mode | — | [`GEN15_REPORT.md`](../generations/gen15_curve/GEN15_REPORT.md) — small zero-shot gain over predicting no separation; one measured pair recovers most of the error | [`generations/gen15_curve`](../generations/gen15_curve/) |
| 16 | six pre-registered leads, blind refutation, one confirmation run | [`START_HERE.md`](../generations/gen16_leads/START_HERE.md) (brief), [`PRE_REGISTRATION.md`](../generations/gen16_leads/PRE_REGISTRATION.md) (sealed) | [`DECISION_REPORT.md`](../generations/gen16_leads/DECISION_REPORT.md), [`CONFIRMATION.md`](../generations/gen16_leads/CONFIRMATION.md), [`REFUTATION_LOG.md`](../generations/gen16_leads/REFUTATION_LOG.md) — one narrow claim confirmed: the gen14 direction call as a between-laboratory screening filter | [`generations/gen16_leads`](../generations/gen16_leads/) |
| 16 side, *unreviewed* | anchor regression with the publication as the anchor | — | none written; `results/g16f_*_BP.csv` — every anchor arm scores worse than gen14 under BP | [`generations/gen16_anchor`](../generations/gen16_anchor/) |
| 16 side, *unreviewed* | few-cluster inference, within-laboratory designs BP1/BP1X, ligand-vs-laboratory identifiability, power | — | none written; `results/` (two runs incomplete) | [`generations/gen16_protocol`](../generations/gen16_protocol/) |
| 17, *unreviewed* | within-publication pairwise difference learning | — | [`results/g17_summary.txt`](../generations/gen17_pairdiff/results/g17_summary.txt) — every arm worse than predicting zero difference; an exploratory probe, not the gen17 agenda gen16 set | [`generations/gen17_pairdiff`](../generations/gen17_pairdiff/) |

Cross-cutting: `results/metrics_reproduction_20260818.md` documents how the headline metrics are
recomputed from raw predictions.

gen12 onwards moved from the repository root into `generations/` in `155dc6c` (2026-09-13), with
their names unchanged; reports inside them cite paths relative to `generations/`. gen2–gen11 were
never moved. Rows marked *unreviewed* were committed from a prior session without review or re-run
(`fdb1e15`, 2026-09-10); nothing in gen16's decision report depends on them.

## gen10, gen11 and the frozen cohort are in version control

Earlier versions of this index, and `HANDOFF_FOR_CHATGPT.md` §5.8, §5.9, §9.5, §10.1, §10.4 and
Appendix B, describe the repository before 2026-09-04 and say otherwise.

- **gen10 and gen11** — source (`src/lanthanide_separation/gen10/`, `gen11/`), tests
  (`tests/test_gen10_*.py`, `tests/test_gen11_invariants.py`), reports and run outputs
  (`runs/gen10_final/`, `runs/gen11_transfer/`) were committed in `d6b8a38` (2026-09-04), and the
  gen11 matched stage in `fca7399` the same day. That includes the deployed model card and the
  frozen pipeline. Prediction dumps and other `*.parquet` / `*.log` artefacts under `runs/` stay
  gitignored; each run's manifest carries their SHA-256.
- **The frozen cohort** (`runs/gen7_architecture/cache/cohort.parquet`, fingerprint
  `bed178ec1a7a82b0`), which every number from gen6 through gen11 is defined against, is tracked:
  `.gitignore` excludes `runs/**/*.parquet` but re-includes it through the negation
  `!runs/gen7_architecture/cache/*.parquet`.

## figures/

The rendered figure set (6 main + 10 supplementary, 72 PNG/PDF, ~52 MB of renders and derived
caches) was **deleted on 2026-09-04 and will be rebuilt**. What survives in `docs/figures/`:

- the nine figure documents — plan, priority, captions, style guide, metric audit, figure audit,
  final report, and the results/methods manuscript prose (the only manuscript text that exists);
- all generator scripts, including `verify_metrics.py`, which recomputes every plotted quantity
  from raw predictions (last run: **67/67 PASS**);
- `derived_values/` — the small per-figure JSON and the metric-audit CSV the captions cite, so
  every plotted number stays checkable without a rebuild;
- `refinement/` — the later publication-quality rebuild: the layout linter, per-figure notes and
  scripts, and two `SCIENTIFIC_DEFECTS.md` audits covering this repository and the sibling
  `lanthanidestrain` project.

To rebuild, run the `prepare_*` then `verify_metrics.py` then `plot_*` scripts from
`docs/figures/scripts/`. `_paths.py` resolves `runs/` artefacts from the current checkout first
(gen10 and gen11 are committed), then `$MLSEP_RUNS_EXTRA`, then any git worktree, so a clean
clone needs no editing.

## Known-stale statements

| document | says | correct |
|---|---|---|
| [`design/README_gen2_simplicial_ru.md`](design/README_gen2_simplicial_ru.md) | 109 publications | **105** |
| [`design/README_gen2_simplicial_ru.md`](design/README_gen2_simplicial_ru.md) | 0.57 % / 1.49 % provenance contamination | **0.10 % / 1.02 %** |
| [`design/README_gen2_simplicial_ru.md`](design/README_gen2_simplicial_ru.md) | "287 of 313 replicated cells have a hidden axis" | retracted — 79 physical / 208 caption-only / 26 nothing-recoverable |
| `results/metrics_reproduction_20260818.md` | 15 lanthanides | **14** (no Pm) |
| `runs/gen9_shape/decision_report.md` | 163 artefacts, 112 tests, "seven issues" | **154**, **125**, the table lists **eight** |
| `runs/gen11_transfer/GEN11_DECISION_REPORT_SUPERSEDED_20260823.md` | "no auxiliary arm passes any criterion" | superseded — written before its own results; `GEN11_DECISION_REPORT.md` was rewritten 2026-09-04 |
| `HANDOFF_FOR_CHATGPT.md` §5.8, §9.5, §10.1, §10.4, Appendix B | gen10/gen11 source, tests and runs untracked in an agent worktree; frozen cohort gitignored; `figures/` untracked | all committed (`d6b8a38`); the figure material is in `docs/figures/` |
| `HANDOFF_FOR_CHATGPT.md` §5.9 | gen11's size-matched control was never run; its decision report is stale | the matched stage ran and the report was rewritten (`fca7399`) |
| `HANDOFF_FOR_CHATGPT.md` Appendix B | the audits and the gen7/gen8 reports sit at the repository root; `docs/` stops at gen6 | `audits/` and `results/` |

Full detail and the reasoning behind the first six corrections is in
[`HANDOFF_FOR_CHATGPT.md`](HANDOFF_FOR_CHATGPT.md) §10.3; the rows about the handoff itself are
checked against `git log` and `git ls-files`.
