# gen8 — series-aware few-shot architecture for `log D`

**One sentence:** gen7 showed the model cannot tell where a new ligand's extraction
curve sits vertically; gen8 asks whether it knows the curve's *shape* well enough that
one or two real measurements anchor the rest — and finds that it does, that **which**
measurement you take is worth as much as a better model, and that the model's biggest
remaining defect is that it draws every unseen titration too flat.

*Cohort fingerprint `bed178ec1a7a82b0` — identical to gen6 and gen7. Frozen global
model: `REC_ecfp_plus_recovered`. Nothing here retrains it; every number is a
calibration of its out-of-fold predictions on held-out chemotypes, which is what
makes the whole study paired down to the row.*

## Read in this order

| file | what it answers |
|---|---|
| `../../gen8_architecture_results_20260820.md` | the report |
| `protocol.md` | what was pre-registered, and when |
| `series_audit.md` | §1 — what the model is supposed to represent |
| `finalists/primary_comparison.csv` | §21 — the primary table |
| `cross_series/calibration_geography.md` | §17/§18 — where to measure, and what it transfers to |
| `mechanism_similarity/mechanism_similarity.md` | §12 — does a better neighbourhood fix the level (no) |
| `functional/slopes.md` | §19 — the flattening result |
| `case_studies/` | §26/§27 — what remains after two measurements |
| `decision_report.md` | the gen9 decision |
| `leaderboard_all.csv` | every (adapter × policy × k) arm on the common cohort |
| `scores_by_seed.csv` | the same, split by seed, for the seed-agreement column |
| `finalists/primary_bootstrap.csv` | the paired chemotype-blocked intervals behind every claim |

## Directory map

```text
series/           curve reconstruction: 1,176 curves, membership, per-curve slopes
baselines/        spline, mass-action and local-GP curve baselines
recommender/      the experiment-recommendation prototype's demo output
functional/       slope diagnosis and the slope-restoration repair
cnp/              conditional neural process
physics_latent/   ligand -> mass-action parameters, MAP-updated by measurements
kshot/            the two protocols (P1 exhaustive, P2 acquisition-fair)
active_acquisition/  14 policies x 6 adapters x k, the primary detail table
cross_series/     calibration geography and the transfer matrix
mechanism_similarity/  36-column mechanistic distance vs Tanimoto on the level
uncertainty/      is uncertainty useful for choosing the next experiment
case_studies/     sulfur-donor failure, error archaeology, DMDPhPDA data audit
ablations/        adaptation-mode and axis-subset ablations
figures/          the six figures the report uses
finalists/        Table A / Table B, leaderboard, bootstrap, adaptation curve
```

## Reproducing

```bash
.venv/bin/python scripts/gen8_series_audit.py            # §1  series + curves
.venv/bin/python scripts/gen8_kshot.py --protocol both   # §7/§9 protocols P1, P2
.venv/bin/python scripts/gen8_calibration_geography.py   # §17/§18
.venv/bin/python scripts/gen8_run.py --seeds 5 --repeats 12 --with-extra
.venv/bin/python scripts/gen8_mechanism_similarity.py    # §12
.venv/bin/python scripts/gen8_slopes.py                  # §19
.venv/bin/python scripts/gen8_slope_restore.py           # §10
.venv/bin/python scripts/gen8_analysis.py                # tables + bootstrap
.venv/bin/python scripts/gen8_figures.py
.venv/bin/python scripts/gen8_manifest.py
```

Every script writes into `runs/gen8_architecture/` and re-running is idempotent.
`environment.json` records the library versions; a run under different versions must
not share a table with these (the project has lost a sweep to exactly that).

## The invariants gen8 enforces in code

* **Held-out chemotype, unchanged.** No ligand's Tanimoto-0.7 chemotype is ever
  split across the fold boundary.
* **An adapter only ever receives the targets of the rows it selected.** The
  interface passes `observed = truth[selected]` and nothing else, so leakage is
  structurally impossible rather than merely audited
  (`src/lanthanide_separation/gen8/adapters.py`).
* **Every policy sees the identical candidate pool and is scored on the identical
  evaluation rows.** Two policies that pick different points are never compared on
  different remainders (`src/lanthanide_separation/gen8/protocols.py`).
* **The pool/evaluation split is a pure function of (seed, repeat, ligand, n_rows)**,
  so an architecture evaluated in a later, separate run is still paired row for row
  with the primary run.
* **Independence is the chemotype, not the ligand.** Every interval resamples the
  chemotype the folds actually held out (`src/lanthanide_separation/gen8/inference.py`).
* **The split seed is a stable digest, not Python's salted `hash()`.** A first round of
  results had to be discarded because the pool/evaluation draw moved between interpreter
  processes, which quietly broke the pairing between separately-run architectures. A
  subprocess test under a different `PYTHONHASHSEED` now enforces it.
