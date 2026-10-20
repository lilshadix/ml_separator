# Data quality and data version V2 (direction E)

Written 2026-10-19, **before any V2 run**. The evidence comes from a read-only inventory, saved in
full as `inventory_data_quality.md`, with its scripts in the session scratchpad. That inventory
rebuilt the cohort from `dataset with 3D structures/dataset.parquet` plus the gen6 provenance table,
and reproduced all of the following exactly: 521 cells, 3,871 rows, 3,359 entries, 216 replicated
entries, and C3 = 229 cells / 1,802 pairs. It also joined every cohort row to the gen19 archive
`dataset_all_metals/clean/master_clean.parquet` (`source_record_id`), which brings in `D_raw`,
comments, data location, curator, and the parsed complexant and solvent fields. The audit files named
in the brief (`AUDIT_2026-10-19.md`, `audit.pdf`) are not in this repository. The DMDPhPDA evidence is
instead the gen8 forensic audit (`runs/gen8_architecture/case_studies/dmdphpda_source_audit.md`) and
`gen20_1/IMPLEMENTATION_PLAN.md:82-90`, re-checked here.

## Defects found

| ID | defect | status | cells (C3) | C3 pairs touched |
|---|---|---|---|---|
| D1 | DMDPhPDA copy B (ORNL import): Table 1 mantissas without the column exponent, so log D is shifted by exactly +4/+3/+2/+1 decades at 1/2/3/4 M HNO₃ and +1 for La/Ce/Pr at 5 M (max deviation from an integer 4.9e-10, re-checked by `r3.datav2.verify`). Only one log SF is wrong: the 5 M Pr/Nd pair, by +1.000 | verified | 5 (5) | 60 |
| D4 | ORNL re-import of cis- and trans-Me2-TODGA under a stereo-free SMILES: two curves 1–2 decades apart averaged in one cell (`0ff595d576155f6c`) | verified provenance | 1 (1) | 12 |
| D2 | the same Table 1 entered by two curators, split only by a NaN contact time; bit-identical on 8 metals | verified | 3 (3) | 6 |
| D3 | hidden-condition merges: different aqueous complexant (name or concentration) or solvent mixture averaged into one cell | verified (archive fields) | 17 (11) | 88 |
| D5 | resolution floor: readings at one unit of the finest decimal in the cell, tied at the cell minimum | data-internal pattern | 6 (4) | 20 |
| D6 | replicate spread > 0.5 (gen20.1 COH-REPCAP); 57 of the 78 are D3 merges, 13 are D4, 7 are figure-vs-table, and 1 is unexplained (caae8f09ae6fe8a6 Gd) | label-derived; not in V2 | 15 (8) | 47 |
| D7 | gen20's "12 % integer-decade replicate errors" | **not supported**: none is an exact power of ten, and each is explained by D3, D4 or a different figure | — | — |
| D8 | isolated integer-decade jumps judged by curve shape | suspected only, never used | ≈6 | ≈10 |
| D10 | one-significant-figure readings | flag only | 24 C3 cells | 58 |
| D11 | cross-publication copy groups | handled by design BPG | 44 (32) | 299 |

There are no written detection limits: no `D_raw` contains "<" or ">", and no comment mentions a
limit. Floors can only be recognised by pattern. The suspected low-D floor of the C5BTBP light
lanthanides (log D ≈ −4.2, insensitive to conditions) has no source statement; it is a sensitivity
flag only.

**Replicates.** Once the hidden-condition merges and the D4 cell are excluded, the median replicate sd
of log D is **0.049** (n = 86), not the 0.302 used since gen13. For 53 adjacent pairs read twice under
the same hidden condition from different data locations, the errors of the two metals correlate at
0.983 and sd(Δ log SF) = 0.086. The errors are common-mode. These are probably double reports of one
measurement, so this bounds transcription noise, **not** laboratory repeatability. The laboratory
repeatability of log SF is still unmeasured (see `EXPERIMENTAL_DATA_REQUEST.md`).

## V2 rules (pinned before any V2 run; code `r3/datav2.py`)

* V2-1 quarantine (whole cell): D1's five copy-B cells and the D4 cell.
* V2-2 quarantine: the NaN member of each D2 duplicate.
* V2-3 quarantine (whole cell): the 17 D3 merges. Splitting them would create cells that do not exist
  in the fold plans.
* V2-5 censor (reading removed): the 24 D5 floor readings. **Deviation from PROTOCOL §9** (recorded
  2026-10-19 after the review): §9 asks for rules "from source evidence only", and D5 is a data-internal
  value pattern (no source states a detection limit). It is a value-pattern rule, fixed before any V2
  run and never chosen from model errors, but it removes readings because of their own values,
  including hidden metals of scored cells (4 C3 cells, 20 C3 pairs). R17/R18 therefore rest partly on a
  pattern rule; a V2 without V2-5 was not run (open sensitivity, `RESUME.md`).
* Not in V2: REPCAP (derived from replicate spread), the suspected floors, curve-shape jumps, the
  suspected ORNL cross-acid copy. Nothing is ever selected by model error or test residual.

Effect: 265 readings removed; **V2 C3 = 208 cells / 1,620 pairs** (V1: 229 / 1,802). V2 is a **new
benchmark**. Every baseline is rerun on it (R17), and no gain relative to V1 is attributed to a model.
Limitation: on V2 the DOPT_Q covariance, the G14 prior and RIDGE2's prior come from the cached V1
training fits (gen20's pipeline). Everything round 3 fits (HBK, corrector, error table) is refitted on
V2 training rows.

## Open (needs sources or people)

* The 5 M Pr/Nd value of DMDPhPDA copy A rests on a curator's reading of Figure 5. The paper
  (SX&IE 22(2) 147–161) is paywalled and was not read.
* Whether the C5BTBP light-lanthanide values sit at a real detection floor.
* Whether the ORNL DMDODGA cells re-transcribe curated cells.
