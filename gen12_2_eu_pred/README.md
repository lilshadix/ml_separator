# Gen12.2 — coordination-aware prediction of the intrinsic Eu extraction level

Can chemically informed coordination-topology descriptors improve prediction of the
extractant-specific europium extraction level for chemically unseen extractants?

**Read `DATA_AUDIT.md`, then `COORDINATION_DESCRIPTOR_SPEC.md`, then `PRE_REGISTRATION.md`,
then `DECISION_REPORT.md`.** The audit and the descriptor specification were written before
the pre-registration, and the pre-registration before any outcome-bearing run.

The brief calls this generation "Gen12.2" and its directory `Gen1Eu_pred-2/`. The
repository's convention is `gen12_eu_pred/`, so the directory here is `gen12_2_eu_pred/`.

## What is new, and what is inherited

Gen12.2 adds exactly two things to Gen12: a **coordination-topology descriptor block**
computed from the molecular graph, and an explicit **level target** with a decomposed
`level + shape` predictor built on it.

Everything else is imported from `gen12_eu_pred/` rather than copied — the cohort builder,
the chemotype hold-out, the bands, the extractant-macro metric, the chemotype-blocked BCa
bootstrap and the few-shot draw function. The rebuilt design-B fold plan is compared
row-id by row-id against Gen12's frozen plan and matches in all 25 folds, so "the same
held-out chemistry" is a property of the code path.

**Nothing under `gen12_eu_pred/` is modified.** `scripts/g122_self_audit.py` re-hashes
every Gen12 artefact against Gen12's own manifest and fails if one has moved.

## Layout

```
gen122/          the library: paths, coordination descriptors, level targets, the level
                 ladder and runner, level metrics, the decomposed predictor
config/          the frozen SMARTS specification and the level-definition decision rule
scripts/         one script per phase, each runnable on its own
features/        the coordination feature matrix, its audit and its content hash
manifests/       the multi-arm subgroup, the level-definition choice, the self-audit,
                 the artefact manifest
predictions/     level/<definition>/<arm>.parquet and full/<arm>.parquet
metrics/         level and full leaderboards, bands, grouped importance, one-shot curves
bootstrap/       paired comparisons, power, influence, subgroup tests
analysis/        the level-definition study, the exploratory analysis, the level
                 bottleneck, and Gen12's failure family reproduced
headline_tables/ the tables the decision report quotes, as CSV and markdown
figures/         rendered figures plus the JSON of every plotted value
tests/           the fourteen pre-registered invariants, executable
```

## Reproducing

```bash
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_build_features.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_level_definition_study.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_exploratory_level.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_level_bottleneck.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_failure_family.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_run_level_ladder.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_run_level_ladder.py --definition LVL_COND_RESIDUAL
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_run_level_ladder.py --family xgboost
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_level_analysis.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_decomposed.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_full_analysis.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_one_shot.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_importance.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_headline_tables.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_figures.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_self_audit.py
PYTHONPATH=gen12_2_eu_pred .venv/bin/python -m pytest gen12_2_eu_pred/tests -q
PYTHONPATH=gen12_2_eu_pred .venv/bin/python gen12_2_eu_pred/scripts/g122_manifest.py
```

## Reading the numbers

Every quantity carries its regime, and mixing two of them makes a sentence false:

- **level** versus **full `log_D`** — a level MAE of 1.03 and a full-query macro MAE of
  1.03 are different quantities on different units;
- **full cohort** (183 extractants) versus **level-reliable cohort** (71 with at least 5
  cells);
- **`LVL_MEAN`** (primary) versus **`LVL_COND_RESIDUAL`** (secondary) level definition;
- **zero-shot** versus **one-shot**;
- **overall** versus **far / mid / near**;
- **selected model** (chosen on inner validation) versus **best observed test arm**.
