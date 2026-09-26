# Confirmation run — status

*Written 2026-09-25 17:15 local, progress refreshed 2026-09-26 21:20, from the files in this directory and the run's own logs. This file is a
**status record of a run that is still executing**, not a result. **No claim has been scored.**
`decisions/confirmation.json` does not exist, so every registered verdict in `GEN19_REPORT.md`,
`SUMMARY.md` and `decisions/D07_confirmation_not_run.md` still stands as printed there: R19 item 4
NOT_EVALUATED, S1 UNDECIDED, S2 and brief §34 Q3 NOT_RUN.*

## What is running

The single once-only confirmation run of the four frozen claims C1–C4 of `decisions/CONFIRMATION_PLAN.md`
on the 5 withheld seeds, confirmation half, plus the single V6 Pr/Nd run (§3.4, POST-HOC addendum 6 item 2).

| | |
|---|---|
| once-only lock spent at | 2026-09-23T23:37:09Z (`decisions/run_started.json`) |
| frozen plan | `1a6e1844ad37…` — claims `C1`, `C2`, `C3`, `C4` |
| code digest every record carries | `6329bc4da416…` (registry stage `confirmation`) |
| seed commitment verified at every start | `65e8ae8ceb8e…` (5 seeds, values withheld) |
| workers | 2 (the measured limit of the machine; recorded in the lock) |

The five withheld seed **values** have never been read, printed or written by any agent. They enter only
through `--seed-store`, a file outside the repository that only the user holds; the logs and record paths
carry the seed **index** `i1`–`i5` and nothing else. They are revealed in `decisions/CONFIRMATION.md`
after the run (§15, addendum 6 item 4).

## Progress

| seed index | records | state |
|---|---|---|
| i0 (comparator legs, public seed) | 586 | complete |
| i1 | 674 | complete |
| i2 | 674 | complete |
| i3 | 677 | complete |
| i4 | 677 | complete |
| i5 | 510 (fitting since 15:48 on 2026-09-26) | fitting |

A complete seed index is 674 records over 17 arm × design groups: `V5__primary__exact` 105 each for B0,
B3i, B6, B6r0; `V5PAIR__primary__batched` 38 each for B3x, M1, M2; `V5__primary__batched_max4` 27 each for
M1, M2; `V6__prnd__exact` 13 each for B3i, B3x, B8, M1, M2; and the V2 actinide-ablation legs, 7 metal
states each for B6/WITH, B6/ACT_PERMUTED, B6r0/WITH. The strict and HNO₃-only colourings are recorded
under the `i0` directory by the runner's own naming.

**Cost measured, not estimated:** i2 took 7.55 h wall (07:21:03 → 14:54:11), i3 12.5 h (14:54 → 03:26)
and i4 12.4 h (03:26 → 15:48), at 2 workers × ≈2.2 cores; i5 started 15:48 on 2026-09-26 and is expected to
end in the early hours of 2026-09-27, then assembly. The run was priced at 67.5 h wall, band 47–88 h;
fitting so far is ≈62 h.

## Interruptions, and why none of them cost fitted work

The run has been interrupted three times. Every interruption was the same cause — a background process
started by Claude Code is killed when that process exits — and none of them lost a fitted fold, because
`--resume` re-fits nothing that is already recorded and refuses any record written under a different code
digest.

| launch | ran | outcome |
|---|---|---|
| 1 | 2026-09-24 04:37 → 21:30 | killed with the session; i0 complete, i1 partial |
| 2 | 2026-09-24 22:59 → 2026-09-25 06:20 | killed with the session; i1 complete, i2 partial |
| 3 | 2026-09-25 07:04 and 07:06 | stopped within minutes; no fold fitted |
| 4 | 2026-09-25 07:10 → running | launched by the user from their own shell, outside Claude Code |

On resume 4 the already-complete indices were skipped in seconds — i0 in 7 s (10 jobs), i1 in 11 s
(15 jobs) — which is the resume path working exactly as designed: `fold_resume_status` treats the fold's
whole write set, parquet and JSON, as the unit, under the live code digest.

## Resuming it again, if it stops

Safe at any time, as often as needed. From the repository root, with `<SEED_STORE>` the store outside the
repository that only the user holds:

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer .venv/Scripts/python.exe \
        generations/gen19_chem_transfer/scripts/g19_run_confirmation.py \
        --seed-store <SEED_STORE> --workers 2 --expect-addenda 8 --resume

It re-fits no completed fold, re-scores no claim, and never widens the claim list. Once
`decisions/confirmation.json` exists the run is spent and the runner refuses with or without `--resume`
(addendum 8 item 2) — that is the guard against a second scoring of the single run, not a fault.
**Never run two copies at once:** they would write the same records concurrently.

## What is in version control

Committed: the withheld-seed fold **designs** under `folds/` (4.7 MB — the evidence of which folds each
seed index produced; no seed is recoverable from a design), the run's stdout logs under `logs/`, the
once-only lock `decisions/run_started.json`, this file, and `MANIFEST.sha256`.

Excluded by `.gitignore`, per the same convention as the discovery run: the raw per-fold records under
`records/`. Their SHA-256 and byte count are recorded in `MANIFEST.sha256`, **a snapshot taken while the
run was still writing** — files listed there are final (a fold is written once and never rewritten), and
records written after the snapshot are reported by `--check` as `new (not listed, not a failure)`. Rewrite
it at completion:

    .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_manifest_discovery.py --root confirmation

## After the run completes

Rehearsed on 2026-09-26 against a synthetic FINISHED run (`tests/test_confirmation_rehearsal.py`'s root): the
readers of the report and of the process gate had been written to a decision-file schema the runner never
produces (`S1.passed` / `S2.a.pass` / `V6.run` / `seeds.verified`, and claim keys `'M2 vs B3i@V5'` where the
runner writes `'M2 vs B3i @ V5@V5'`), and F12 / the V6 panel of F13 / Q3's per-system line read three files
(`v6_systems.csv`, `v6_rows.csv`, `v6_pairs.csv`) that no stage wrote. Both were fixed the same day, in code
OUTSIDE the confirmation digest (the running process is unaffected): the readers now cite the runner's real
keys (`s1.verdict`, `s2.status` / `s2.verdict`, `s2.s2a..s2d.status`, `seed_store.verified_against_commitment`,
`claims[].claim` + `design`) so `verify_numbers` re-resolves every printed value from the file, and
`scripts/g19_export_confirmation_views.py` writes the three view files from the decision file and the verified
V6 records (seed mean over the seed indices; it decides nothing and refuses to run before
`decisions/confirmation.json` exists). The order of work is therefore:

1. `scripts/g19_export_confirmation_views.py` — the three `evaluation/confirmation/v6_*.csv` view files.
2. `scripts/g19_build_report.py` — Q3 and S2 resolve, R19 item 4 is evaluated, and every learned-arm
   verdict becomes decidable for the first time.
3. `scripts/g19_make_figures.py` — figure F12 and the V6 panel of F13 (from the view files of step 1).
4. `scripts/g19_run_process.py --decision-only` — D06's confirmation gate line (the gate now accepts the
   runner's `gen19.confirmation_decisions.v1` file).
5. Update the status lines of `decisions/CONFIRMATION_PLAN.md` and `decisions/D07_confirmation_not_run.md`,
   and rewrite `MANIFEST.sha256`.
6. Phase H (§14) as a registered run only if S1 passes; otherwise exploratory and labelled
   transfer-unsupported. F1 stays NOT_EVALUATED: it needs a V0 design, which the frozen plan does not contain.
