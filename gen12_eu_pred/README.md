# Gen12Eu_pred

Single-lanthanide prediction of europium extraction on chemically unseen extractants.

**Read `DATA_AUDIT.md`, then `PRE_REGISTRATION.md`, then `DECISION_REPORT.md`.** The audit was
written before the pre-registration and the pre-registration before any outcome-bearing run.

This directory is self-contained and touches nothing in gen1–gen11. It reads the frozen bundle,
the frozen gen6 chemistry map and the multi-metal archive **read-only**, and imports
`lanthanide_separation` only for the definitions that must stay identical to the earlier
generations: condition and series identifiers, bit-identical ECFP clusters, single-linkage
Tanimoto-0.7 chemotypes, the randomised grouped K-fold and the TODGA quarantine.

## Layout

```
gen12eu/            the library: cohort, chemistry, splits, preprocessing, models,
                    D-MPNN, few-shot, cross-lanthanide, metrics, inference
scripts/            one script per phase, each runnable on its own
config/             (reserved)
manifests/          cohort parquet, data_audit.json, manifest.json, self_audit.csv
splits/             per-fold similarity tables
predictions/        per-row out-of-fold predictions, one parquet per arm
metrics/            leaderboards, band tables, per-extractant and per-chemotype metrics
bootstrap/          paired comparisons, power, influence, hypothesis tests
figures/            rendered figures plus the JSON of every plotted value
headline_tables/    the tables the decision report quotes, as CSV and markdown
tests/              the twelve pre-registered invariants, executable
```

## Reproducing

```bash
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_data_audit.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_run_ladder.py --design B
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_run_ladder.py --design A
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_run_ladder.py --design B --with-graph --arms T3_DMPNN_COND
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_ablation.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_multiln.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_analysis.py --design B
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_hypotheses.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_frontier.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_fewshot.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_one_measurement.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_figures.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_headline_tables.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_manifest.py
PYTHONPATH=gen12_eu_pred .venv/bin/python gen12_eu_pred/scripts/gen12_self_audit.py
cd gen12_eu_pred && ../.venv/bin/python -m pytest tests/ -q
```

`gen12_manifest.py --verify` re-hashes every artefact and the frozen inputs, and exits non-zero on
drift. `gen12_self_audit.py` re-checks the pre-registered invariants against the files on disk
rather than against freshly built objects.

## Three things to know before quoting a number

1. **Every number needs its regime.** {zero-shot | k-shot} x {design B | design A} x
   {extractant-macro | chemotype-macro | row} x {overall | far | mid | near}. The same frozen model
   scores 1.09 and 0.77 depending only on the split design.
2. **Design B caps train similarity at 0.698 by construction.** Holding out a Tanimoto-0.7
   single-linkage cluster guarantees it. "near" means 0.60 to 0.698, not "a close homologue was in
   training".
3. **The study is underpowered for its own primary contrast.** The minimum detectable paired
   difference at 80 % power is about 0.19 macro MAE against an apparent effect of 0.09. A
   non-significant result here is not evidence of equivalence, and the equivalence margin is not
   met either.
