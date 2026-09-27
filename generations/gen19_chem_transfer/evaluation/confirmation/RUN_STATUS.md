# Confirmation run — status: COMPLETE

*Written 2026-09-25 17:15 local while the run executed; rewritten 2026-09-27 after it completed. Every number
below is the decision file's (`decisions/confirmation.json`) and is re-resolved from it by `GEN19_REPORT.md`.*

## What ran

The single once-only confirmation run of the four frozen claims C1–C4 of `decisions/CONFIRMATION_PLAN.md`
on the 5 withheld seeds, confirmation half, plus the single V6 Pr/Nd run (§3.4, POST-HOC addendum 6 item 2).

| | |
|---|---|
| once-only lock spent at | 2026-09-23T23:37:09Z (`decisions/run_started.json`) |
| assembly wrote `decisions/confirmation.json` | 2026-09-26T23:14:58Z |
| writer stage completed (`decisions/CONFIRMATION.md`, seeds revealed) | 2026-09-27 04:36 local, by `scripts/g19_finish_confirmation_writers.py` (addendum 9 item 2) |
| frozen plan | `1a6e1844ad37…` — claims `C1`, `C2`, `C3`, `C4` |
| code digest every record carries | `6329bc4da416…` (registry stage `confirmation`, now a superseded entry; all 3,965 records verify against it) |
| seed commitment verified at every start | `65e8ae8ceb8e…` |
| records | 3,965 (i0 586, i1 674, i2 674, i3 677, i4 677, i5 677); 0 errors; 87.6 h wall clock across the launches |

The five withheld seed values were never read, printed or written by any agent while the run executed; they
entered only through `--seed-store` and are revealed in `decisions/CONFIRMATION.md`, written last, as §15 and
addendum 6 item 4 register.

## The registered verdicts

| claim / criterion | verdict | what the file records |
|---|---|---|
| C1 `M2 vs B3i @ V5` | **PASS, confirmed** | Δ +0.4115 ≥ δ5 0.1057; R19 items 1–6 PASS; Δ > 0 in 5 of 5 withheld seeds; TOST PASS |
| C2 `M2 vs B0 @ V5` | **PASS, confirmed** | Δ +0.6712 |
| C3 `M2 vs B6r0 @ V5` | **PASS, confirmed** | Δ +0.4286 |
| C4 `B6:WITH vs B6:ACT_PERMUTED @ V2` | **FAIL** | Δ +0.0282 < 0.05; items 1–3 FAIL |
| S1 | **UNDECIDED** | S1(a) PASS, S1(b) PASS, S1(c) UNDECIDED (see below), S1(d) PASS, S1(e) NOT_COMPUTED (`support_s7` below the §8 reliability floor) |
| S2 | **FAIL** (record set COMPLETE, 13 systems) | S2(a) FAIL: sign agrees in 9 of 13 (needs 11), min_Y Δ_Y −0.210 pooled / −0.249 HNO₃; S2(b) FAIL: logSF MAE 0.159 vs FLAT 0.137 (gain −0.022, interval includes 0), log D MAE 0.357 ≤ L5 0.848; S2(c) FAIL: 80 % coverage 0.938 above [0.70, 0.90]; S2(d) NOT_RUN |

In words: on the confirmation half, chemistry transfer (M2) beats the within-system lookup and both constant
baselines on hidden metal × extractant cells, with every R19 item and all five withheld seeds agreeing — but
Pr/Nd selectivity with the pair hidden is **not** reconstructed to the registered standard: the sign of the
per-system median logSF is right in 9 of 13 systems, and the logSF error is worse than predicting no
selectivity at all (the FLAT floor). Actinide rows do not help B6 (C4). S1 stays UNDECIDED, so the process
layer remains exploratory and every process output transfer-unsupported (§14).

**S1(c): a runner defect, recorded, not corrected.** The decision file holds every confirmation-half check
of S1(c) as `true` (Δ > 0 in 5 of 5 seeds, every direction interval excludes 0, logSF MAE below B3i and below
FLAT by η5, min Δ ≥ γ5) and the selection-half counterweight the run itself read from the discovery scorer
(min Δ −0.0168 ≥ −0.02, `counterweight_pass: true`). Section 9 S1(c) as redefined is PASS when both hold.
The verdict field nevertheless says UNDECIDED with the reason `selection-half components missing`: the
combining step looked for an in-run `selection` block that the run does not compute, instead of the
counterweight it had already read. The registered file is not edited — re-scoring the single run is what
§15 forbids — so the field stands as written and the report states the discrepancy beside it. S1 overall is
UNDECIDED regardless, because S1(e) is NOT_COMPUTED.

## The refusal at the end, and how it was closed (POST-HOC addendum 9)

After the assembly, the writer stage refused before `decisions/CONFIRMATION.md` under the leak scan of
addendum 6 item 4: 24 of the 3,965 record JSONs contained the decimal form of a withheld seed as a substring
of their bytes. A value-blind diagnosis (key path, leaf type, digit-run length only; no value printed) found
every match inside the 14–16-digit mantissa of a float (`split_unit_errors[].sum_abs_error` ×21,
`fits[].valid_macro_mae` ×2, `selection.inner_fold_macro_mae` ×1); no leaf equals a seed and no string field
names one. Addendum 9 item 1 adopts the reading that a seed *appears* only as a whole number token
(`confirmation.seed_pattern`); item 2 completes the writer stage once with the run's own `write_outputs` on
the unchanged decision body (SHA-256 before `4f5363994dc5…`; the only addition is the `seed_leak_scan`
block, asserted); item 3 makes readers outside the run take the run's recorded code digest. The `confirmation`
stage was re-registered from the tree (`2c1da435a9a6…`) and the code change logged; all 3,965 records verify
against the superseded entry.

## Interruptions, and why none of them cost fitted work

| launch | ran | outcome |
|---|---|---|
| 1 | 2026-09-24 04:37 → 21:30 | killed with the Claude Code session; i0 complete, i1 partial |
| 2 | 2026-09-24 22:59 → 2026-09-25 06:20 | killed with the session; i1 complete, i2 partial |
| 3 | 2026-09-25 07:04 and 07:06 | stopped within minutes; no fold fitted |
| 4 | 2026-09-25 07:10 → 2026-09-27 04:14 | launched by the user from their own shell, outside Claude Code; i2–i5 and the assembly |

`--resume` re-fitted nothing already recorded (i0 skipped in 7 s, i1 in 11 s on launch 4) and refused any
record written under a different code digest. Measured cost per seed index: i2 7.55 h, i3 12.5 h, i4 12.4 h,
i5 12.4 h, at 2 workers × ≈2.2 cores; the run was priced at 67.5 h wall (band 47–88 h).

## What is in version control

Committed: the withheld-seed fold designs under `folds/`, the logs under `logs/`, `decisions/run_started.json`,
`decisions/confirmation.json`, the view files `v6_systems.csv` / `v6_rows.csv` / `v6_pairs.csv`
(`scripts/g19_export_confirmation_views.py`; descriptive, decide nothing), `MANIFEST.sha256` and this file;
at the generation root `decisions/CONFIRMATION.md` (the revelation) and `tables/confirmation_*.csv|md`.

Excluded by `.gitignore`, per the discovery run's convention: the raw per-fold records under `records/`.
Their SHA-256 and byte count are in `MANIFEST.sha256` (7,930 files, 366.1 MB, final):

    .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_manifest_discovery.py --check --root confirmation

## After the run — done on 2026-09-27

1. `scripts/g19_export_confirmation_views.py` — done (13 systems, 418 hidden Pr/Nd rows, 209 pairs; arm M2).
2. `scripts/g19_make_figures.py --only F12,F13` — done; F07–F11 carried over in the index.
3. `scripts/g19_run_process.py --exploratory --n-lhs 300` — re-run under the real confirmation verdicts so
   D06 and `evaluation/process/summary.json` carry them (still exploratory: S1 UNDECIDED).
4. `scripts/g19_build_report.py` — Q3 and S2 answered, R19 item 4 evaluated, S1(c) discrepancy stated.
5. Status lines of `decisions/CONFIRMATION_PLAN.md` and `decisions/D07_confirmation_not_run.md`; this file;
   `MANIFEST.sha256` rewritten.
6. Phase H as a registered run: not applicable (S1 not passed). F1: NOT_EVALUATED (no V0 design in the plan).
