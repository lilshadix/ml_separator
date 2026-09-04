# Documentation index

Everything written about this project, organised by kind. Code lives in `src/`, `scripts/`,
`tests/` and `slurm/`; experiment outputs live in `runs/`.

```
docs/
├── audits/      leakage and feature audits, written before any result
├── protocols/   pre-registrations — hypotheses and decision rules, frozen before the run
├── results/     what the runs found
├── design/      roadmaps and design notes
└── figures/     the publication figure set's knowledge (renders deleted, see below)
```

## Start here

| if you want… | read |
|---|---|
| the whole programme in one document | [`../HANDOFF_FOR_CHATGPT.md`](../HANDOFF_FOR_CHATGPT.md) — self-contained, gen2→gen11 |
| what may and may not be claimed from a number | [`figures/METRIC_AUDIT.md`](figures/METRIC_AUDIT.md) |
| whether the evaluation is sound | [`audits/LEAKAGE_AUDIT.md`](audits/LEAKAGE_AUDIT.md) |
| the shipped model and its limits | `runs/gen10_final/GEN10_FINAL_MODEL_CARD.md` (worktree — see below) |

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
| 10 | finalisation and freeze | (external brief) | `runs/gen10_final/GEN10_*.md` | worktree only |
| 11 | multi-metal actinide transfer | in code (`gen11/arms.py`, `analysis.py`) | `runs/gen11_transfer/` — **decision report is stale** | worktree only |

Cross-cutting: `results/metrics_reproduction_20260818.md` documents how the headline metrics are
recomputed from raw predictions.

## Two things that are not in this checkout

- **gen10 and gen11** — source, tests and run outputs live only in the git worktree
  `.claude/worktrees/lanthanide-separation-finalize-81154b`, uncommitted. That includes the
  deployed model card and the frozen pipeline.
- **The frozen cohort** (`runs/gen7_architecture/cache/cohort.parquet`, fingerprint
  `bed178ec1a7a82b0`) is gitignored, yet every number from gen6 onward is defined against it.

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
`docs/figures/scripts/`. Note `_paths.py` resolves gen10/gen11 artefacts through the worktree
path, so it needs editing on any other machine.

## Known-stale statements

| document | says | correct |
|---|---|---|
| `../README.md` | 109 publications | **105** |
| `../README.md` | 0.57 % / 1.49 % provenance contamination | **0.10 % / 1.02 %** |
| `../README.md` | "287 of 313 replicated cells have a hidden axis" | retracted — 79 physical / 208 caption-only / 26 nothing-recoverable |
| `results/metrics_reproduction_20260818.md` | 15 lanthanides | **14** (no Pm) |
| `runs/gen9_shape/decision_report.md` | 163 artefacts, 112 tests, "seven issues" | **154**, **125**, the table lists **eight** |
| `runs/gen11_transfer/GEN11_DECISION_REPORT.md` | "no auxiliary arm passes any criterion" | superseded — written before its own results |

Full detail and the reasoning behind each correction is in `../HANDOFF_FOR_CHATGPT.md` §10.3.
