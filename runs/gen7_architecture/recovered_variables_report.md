# Recovered variables — what existed upstream and had been discarded

*Brief §2. Source: `lanthanide_dataset_builder/raw_data/*_SAFE.csv`, 31 files, 48,138
rows, joining **5,992/5,992** to the bundle on `safe_exp_id` (`{stem}_SAFE:{exp_id}`).
Built by `src/lanthanide_separation/gen7/recovered.py`; the machine-readable audit is
`data_audit_safe_columns.json` and the raw column-by-column table is
`recovered_variables_raw.md`.*

## Headline

Recovering these variables is worth **+0.035 to +0.056 macro MAE** on identical test
rows — the same order as everything else that worked in gen7, though on the
chemotype-blocked bootstrap the interval straddles zero (+0.035, CI [−0.051, +0.098]).

The brief expected the recovery to be dominated by holdback agents and radiolysis. It is
not, and the reason is worth stating first.

## What is NOT recoverable, and why

| upstream column | populated | verdict |
|---|---|---|
| `Holdback_Agent_Name` | **0 %** | empty in all 48,138 rows |
| `Holdback_Agent_Concentration_M` | **0 %** | empty |
| `Radiolytic_Dosage_kGy` | **0 %** | empty |
| `f_Metal_Concentration_mM` | **0 %** | empty |
| `thirdType` / `thirdValue` | **0 %** | empty |
| `ini_comp` | 100 % | 1,832 unique, but it is `acid, extractant, metal, solvent` re-concatenated — 900 rows carry a fifth token and in every case it is the second half of a solvent mixture `Solvent_Name` already names. No new variable. |
| `Metal_Oxidation_state` | 100 % | two values, `"III "` and `" "`. Constant. |
| `volValue` | 56.5 % | single value `1.0`. Constant. |

**The holdback hypothesis was right; the evidence is in the name, not the column.** One
canonical SMILES — TODGA, 1,714 rows — carries 21 `extractant_name` values, and 20 of
them are different chemicals: SO3-Ph-BTP, (PhSO3Na)2-BTBP, the TWE- series, PHEN-6OH,
PyTri-diol, CITAM. Those are water-soluble complexants added to the *aqueous* phase
alongside TODGA; the record kept the aqueous agent's name and left the organic
extractant's SMILES. 414 rows bundle-wide carry a name that is not the modal name of
their structure. Correcting an earlier audit of this: **no model-visible cell mixes
flagged and unflagged rows**, so this misattributes rows to a ligand rather than
contradicting it.

## What IS recovered

### `SOLVENT` — the substantial one

The bundle one-hots the diluent into 34 levels and dumps 43 of the 83 upstream solvents
into `cond__diluent__other` (305 rows). Worse, a one-hot cannot express that 1-octanol
and 1-decanol are nearly the same liquid, so 34 independent columns each learn their own
offset from their own rows.

`parse_solvent` resolves **all 83 names, 0 unparsed**, into base components with volume
fractions, handling the three notations the corpus uses:

```text
"n-Dodecane"                      -> {dodecane: 1.0}
"kerosene with 30 vol% 1-octanol" -> {kerosene: 0.7, octanol: 0.3}
"kerosene 0.7, 1-octanol 0.3"     -> {kerosene: 0.7, octanol: 0.3}
```

and emits volume-weighted physics from a table of 37 base
components: dielectric constant, dipole moment, logP, molar volume, Hansen dD/dP/dH,
aromatic/halogen/hydroxyl content, a technical-mixture flag, plus `log10(eps)` and the
polar (alcohol + TBP) volume fraction.

One judgement call is recorded: **`CH3Cl` is mapped to chloroform, not chloromethane.**
The 140-row DMDPhPDA duplicate is one paper entered twice, one copy writing `CH3Cl` and
the other `Chloroform` for the same experiments; treating them as different liquids
splits one experiment in two. The bundle keeps them as separate one-hots.

### Scalars absent downstream

| feature | coverage | note |
|---|---|---|
| `rec__phase_modifier_concentration_M` | 4.4 % | the bundle keeps `cond__additive__<name>` and drops the concentration |
| `rec__shaking_time_min` | 17.8 % | only `contact_time` survived downstream |
| `rec__acid_concentration_organic_M` | 0.6 % | kept for completeness; expected to do nothing |

### The unmodelled-second-species flags

| feature | non-zero cells | meaning |
|---|---|---|
| `rec__name_mismatch` | 353 | the recorded name is not the modal name of this structure |
| `rec__aqueous_complexant` | 86 | and the name matches a water-soluble complexant signature |
| `rec__n_names_for_structure` | all | how many names this structure carries (max 21) |

Safe at inference: an operator knows before measuring whether a second complexant was
added. The modal name is derived from *names only*, never from `log_D`, which puts it in
the same target-free class as the frozen chemistry map.

### Nuisance keys — never features

`nuisance__doi` (115 publications), `nuisance__data_location`, `nuisance__batch`
(694 (DOI, table) pairs bundle-wide, 541 inside the evaluation cohort),
`nuisance__entry_author`, `nuisance__addition_date`, `nuisance__extractant_name`.
`recovered_feature_columns()` excludes every one of them; they appear only in
`ORACLE_*` arms and in the variance-components analysis. A new ligand has no DOI.

## Per-column coverage


| column | kind | coverage | unique | dtype |
|---|---|---|---|---|
| `rec__acid_concentration_organic_M` | feature | 0.006 | 35 | float64 |
| `rec__has_phase_modifier` | feature | 1.000 | 2 | float64 |
| `rec__has_shaking_time` | feature | 1.000 | 2 | float64 |
| `rec__phase_modifier_concentration_M` | feature | 0.044 | 15 | float64 |
| `rec__shaking_time_min` | feature | 0.178 | 11 | float64 |
| `rec__solvent_arom` | feature | 1.000 | 3 | float64 |
| `rec__solvent_dD` | feature | 1.000 | 40 | float64 |
| `rec__solvent_dH` | feature | 1.000 | 39 | float64 |
| `rec__solvent_dP` | feature | 1.000 | 37 | float64 |
| `rec__solvent_eps` | feature | 1.000 | 50 | float64 |
| `rec__solvent_hal` | feature | 1.000 | 3 | float64 |
| `rec__solvent_log_eps` | feature | 1.000 | 50 | float64 |
| `rec__solvent_logp` | feature | 1.000 | 56 | float64 |
| `rec__solvent_mu` | feature | 1.000 | 35 | float64 |
| `rec__solvent_n_components` | feature | 1.000 | 2 | float64 |
| `rec__solvent_oh` | feature | 1.000 | 13 | float64 |
| `rec__solvent_parsed` | feature | 1.000 | 1 | float64 |
| `rec__solvent_polar_fraction` | feature | 1.000 | 13 | float64 |
| `rec__solvent_technical` | feature | 1.000 | 14 | float64 |
| `rec__solvent_vm` | feature | 1.000 | 55 | float64 |
| `nuisance__addition_date` | nuisance | 1.000 | 543 | str |
| `nuisance__batch` | nuisance | 1.000 | 694 | str |
| `nuisance__data_location` | nuisance | 1.000 | 634 | str |
| `nuisance__doi` | nuisance | 1.000 | 115 | str |
| `nuisance__entry_author` | nuisance | 1.000 | 3 | str |
| `nuisance__extractant_name` | nuisance | 1.000 | 228 | str |
