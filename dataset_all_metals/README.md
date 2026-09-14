# `dataset_all_metals` — multi-metal Separation Archive for Elements build

A **separate, self-contained** dataset workspace. It does not read from, write to,
or otherwise touch the frozen gen7–gen10 lanthanide pipeline, its cohort, its
splits or its predictions. The control stays the control:
`dataset with 3D structures/dataset.parquet` still hashes to
`fefbefc6…4faf5dd`, the digest pinned in `gen3_protocol.json`.

## What this is for

The primary scientific objective is unchanged: **lanthanide separation**. The
expanded archive is preserved in full so that a later experiment can test
whether observations on other metals teach transferable relationships between
extractant structure, metal identity, solvent, conditions and `log D` — and
whether that improves prediction for *unseen lanthanide* systems. This build
makes no claim that it will. It only makes the question answerable without
rebuilding the data.

## Layout

```
raw/            41 immutable per-metal CSVs (read-only, checksummed)
intermediate/   normalized but not finally deduplicated frames
clean/          master_clean + the lanthanide / non-lanthanide / single-component subsets
audit/          duplicate groups, conflicts, curve inventory, manual-review queue
reports/        schema, quality report, coverage, field mapping, accounting, manifest
scripts/        the pipeline (stage01 … stage07) plus shared modules
tests/          the test suite covering the guarantees below
```

Deliverables are also published, with SHA-256 hashes, to
`../runs/sae_dataset_audit/`.

## Running it

```bash
../.venv/bin/python scripts/run_all.py --verify-determinism
```

The flag runs the whole pipeline twice and fails unless every artifact is
byte-identical.

```bash
../.venv/bin/python -m pytest tests/ -q
```

## The four things worth knowing before using this data

**1. The source file name is not the measured metal.** The archive was exported
one CSV per *queried* metal. A record whose measured metal is Nd appears
byte-identically in twelve different metal files. This is why 48,471 raw rows
carry only 16,770 distinct `exp_id` values. Before collapsing the fan-out the
pipeline verifies that all 37 raw columns are constant inside every `exp_id`
group — they are, in 0 of 16,770 groups is there any variation — so the fan-out
is pure export metadata.

**2. Structures are trustworthy; names are not.** TODGA's SMILES is attached to
22 unrelated extractant names, because one sub-source (`./ST*.json`) records a
water-soluble masking agent's *name* next to the organic extractant's
*structure* and hides the agent's real structure in `comments_description`.
Experimental identity is therefore keyed on canonical structure, never on name,
and every name/structure disagreement is queued for review rather than resolved.

**3. Multi-component systems keep every component.** `components` is a list of
role-tagged entries (`organic_extractant`, `phase_modifier`,
`aqueous_complexant`, `aqueous_holdback`), each with its own structure and
concentration. `extractant_system_key` sorts components before joining, because
the archive writes `"A, B"` and `"B, A"` interchangeably. The frozen gen10 table
flattens `TODGA,DHOA` to TODGA alone; this one does not.

**4. Nothing is averaged and nothing disappears.** Only classes A and B (same
conditions, same value, same or copied provenance) collapse. Value conflicts,
possible independent replicates, condition conflicts and under-specified records
all keep every row. `reports/row_accounting.json` is asserted to balance.

## Conventions the pipeline holds itself to

- A parser returns `None` when it cannot parse. It never guesses.
- Every normalisation that changes a value is counted in
  `reports/normalization_log.json`.
- Two chemical names are merged only when they denote the same substance beyond
  reasonable doubt. `CH3Cl` vs `CHCl3`, `octanol` vs `1-octanol` and
  `tetrachloroethane` vs `tetrachloroethylene` are kept apart and reported.
- `log_D` is never read to define an identity key, a series or a curve. The one
  exception is duplicate *classification*, where comparing the measured value is
  the whole point.
- Reference values that cannot be stated confidently are left null with a status
  column, not filled with a plausible-looking number. Nine of the 40 metals carry
  no ionic radius on any row, and `metal_coverage.csv` separates the two reasons
  (the ion is not tabulated at CN=8 vs. the archive never recorded an oxidation
  state, so the lookup has no key).

## Relationship to the frozen pipeline

`reports/field_mapping.csv` traces every raw archive column to what the existing
pipeline actually does with it, with file:line evidence. The short version:

- **Consumed directly**: `Extractant_SMILES`, `Extractant_Concentration_M`,
  `Acid_Name`, `Acid_Concentration_M`, `Solvent_Name`, `Metal_Name`,
  `obsTempsValue`, `obsDvaluesValue`.
- **Needs new model support for multi-metal**: the metal descriptor block is a
  hardcoded 14-entry lanthanide dict
  (`lanthanide-ml/scripts/dataset/build_dataset.py:127-142`); non-lanthanides
  are hard-deleted before it is merged. `pairs.py:647` filters to `LANTHANIDE_Z`.
  Both must be extended before any non-lanthanide row can reach a model.
- **Present here, ignored by gen10**: `Acid_Concentration_Organic_M`, phase
  modifier concentration, and everything the archive hides inside
  `comments_description` (aqueous complexant and holdback structures and
  concentrations, nitrate concentration, figure/table provenance).
