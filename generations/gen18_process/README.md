# gen18_process — the process-design chain

Generation 18 turns the frozen corpus plus literature entries into a process-selection tool:

    extraction-system composition -> D of each metal at the given loading -> countercurrent
    cascade (extraction + scrub + strip + organic recycle) -> purity, recovery, throughput,
    reagent consumption -> local regime optimisation

The product is a computed regime with its uncertainty and applicability flags, not a
regressor. Every D carries a provenance status; missing measurements stay missing. The first
applied case is PC88A versus Cyanex 272 for Pr/Nd in chloride (literature-parameterised, every
number a labelled placeholder with a range) and the only corpus-fitted model (a pooled-slope
mass-action fit) is pre-registered before it is fitted.

## Reading order

1. `BRIEF.md` — the task, repository facts, chemistry, validity contract.
2. `DESIGN.md` — the implementation contract (wins over the brief); `addenda/*.md` are dated
   interface addenda by the owners (`WB0.md` layout, `WB1.md` data, `ORCHESTRATOR_prefit_*.md`
   what was read about the loading-series sources before sealing).
3. `PRE_REGISTRATION.md` — frozen protocol of the corpus-fitted D model (sealed after the audit).
4. `LITERATURE_NOTES.md` — every literature number of the Pr/Nd case with its status.
5. `DATA_AUDIT.md` — cohort counts recomputed from the built database (the frozen source).
6. `GEN18_REPORT.md` — results (written last).

## Layout

```
gen18proc/   paths, types (every dataclass), systems (database + validator), literature,
             ingest (WB1); dmodel, domain, equilibrium, testsystems (WB2); cascade, metrics
             (WB3); evalproto, optimize, screen, report (WB4)
systems/     the database: sys_<id>.json (287 corpus + 2 literature), corpus_records.csv,
             series.csv, duplicates.csv, exclusions.csv, registry.json, INDEX.md
cases/       prnd_feed.json, prnd_spec.json, todga_prnd_feed.json (every number assumed)
config/      prices.json (WB3), design_spaces.json (WB4)
scripts/     g18_*.py, run in the order of DESIGN.md section 11
tests/       pytest invariants (markers: slow, validation)
results/     audit/ dmodels/ eval/ loading/ bench/ case_prnd/ recipes/ optimize/ screen/
```

## Running (from the repository root, one Python process at a time)

```
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_build_db.py      # 1 database
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_audit.py         # 2 DATA_AUDIT.md
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_seal_prereg.py   # 3 seal (orchestrator only)
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_seal_prereg.py --check
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_fit_dmodels.py --all      # 4
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_eval_dmodels.py           # 5
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_loading_check.py          # 6
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_bench.py                  # 7
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_case_prnd.py --feed generations/gen18_process/cases/prnd_feed.json --spec generations/gen18_process/cases/prnd_spec.json --allow-placeholders   # 8
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_optimize.py ...           # 9
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_recipes.py ...            # 10
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_screen.py ...             # 11
.venv/Scripts/python.exe generations/gen18_process/scripts/g18_report.py                 # 12
```

Set `PYTHONIOENCODING=utf-8`. Scripts 4-6 refuse to run until `g18_seal_prereg.py --check`
passes; the seal is applied by the orchestrator after reviewing `DATA_AUDIT.md`. Seeds default
to 18; no results file carries a wall-clock value except `results/bench/`.

## Tests

```
.venv/Scripts/python.exe -m pytest generations/gen18_process/tests -q -m "not slow and not validation"   # fast
.venv/Scripts/python.exe -m pytest generations/gen18_process/tests -q                                    # all
```

`test_ingest.py` reads the built database and skips (with a message) until scripts 1-2 have
run; its `slow` test re-runs the ingest. `validation`-marked tests report a result, they do not
block the suite (DESIGN.md section 12.3).

## Data facts in one line

5992 bundle rows -> 5860 after the gen13 quarantine; 105 publications; 287 corpus systems under
the key (ligand SMILES set, acid class, diluent family, modifiers, complexants); 10
publication-aware loading series (11 publication-blind); 7 unit-slip duplicate rows removed
from fitting; 130 tied-D groups kept and flagged; replicate floor 0.225 log D; E1 cohort 59
fittable multi-publication (system, metal) groups in 14 systems. `DATA_AUDIT.md` is the source.
