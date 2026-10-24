# Round 5: how to resume

Kept together with the append-only section of `DECISION_LOG.md` (PROTOCOL: request status lives here and below the
DECISION_LOG marker, never in the sealed `DATA_REQUEST.md`). The text above the marker is sealed with Stage 0; later
status goes below the marker, one dated (UTC) entry at a time.

## 1. Requests (shown to the user once, in DR rank order, on 2026-10-24; none answered)

| id | DR rank | request | used by (brief) | status |
|---|---|---|---|---|
| U1-a | 1 | say which publishers you can read through an institution | 7.1 | OPEN |
| U1-b | 2 | allow one download: Shimada 2004 thesis (K0379.pdf), into a quarantine folder | 7.1 | OPEN |
| U1-c | 3 | allow web reading (no downloads) of 6 open-access import papers | 7.1 | OPEN |
| U1-d | 5 | allow one download: the JACS Au 2022 SI spreadsheet | 7.1 | OPEN |
| U1-e | 9 | with Taylor & Francis access: DMDPhPDA Table 1 and Figure 5 | 7.1 | OPEN |
| U1-f | 11 | one e-mail (drafted by the agent, sent by the user) to the SAFE curator | 7.1 | OPEN |
| U1-g | 12 | with access: the two C5-BTBP papers (Foreman 2005, Nilsson 2006) | 7.1 | OPEN |
| U1-h | 13 | with access: the 15 other closed import papers, in the seeded census order | 7.1 | OPEN |
| U1-i | 14 | optional contacts: Chalmers, NEA EGFRC (IDEaL), JAEA | 7.1 | OPEN |
| U2 | 6 | SAFE gap-ID export, metadata only (M33 supply gate) | 7.2 | OPEN |
| U3 | 7 | allow fetching the SI of Molecules 29:3548 (E-CT dry run; corpus-linked laboratory) | 7.3 | OPEN |
| U4 | 15 | a laboratory collaborator (M11 re-measurement, M12 reproducibility) | 7.4 | OPEN |
| U5 | 4 | web reading for the literature supply, E2 verification, targeted E-CT search, planning amplitudes | 4.3 item 10, 7.5 | OPEN (amplitudes PENDING-U5) |
| U6 | 8 | review the ambiguous links of the draft laboratory key | 5.5 | OPEN (key PROVISIONAL) |
| S1 | 10 | sign off the cohort-key remedy for the 11 V2-3 cells | 5.3 | OPEN |
| H1 | not ranked | housekeeping: optionally delete PDFs that tools auto-cached outside the repository | none (hygiene) | OPEN |

No request is paraphrased as granted. An answer is recorded below the marker of DECISION_LOG and below the marker of
this file, with its date, before anything it allows is done.

## 2. PROVISIONAL and open inputs, and what fixes each (DATA_CONTRACT §9)

| input | status | what fixes it |
|---|---|---|
| laboratory key, conservative / liberal counts | PROVISIONAL | U6 review |
| chemotypes of literature candidates | PROVISIONAL (hand labels; upper bounds) | gen6 map at the audit (U5 Phase A or values) |
| Chiba T0 / T2 pair classification | PROVISIONAL | values (U5 / U1 access) |
| HDEHP / P507 planning amplitudes | PROVISIONAL (unverified) | U5 item 4 (CONFIRMATION_PACKAGE §12) |
| SAFE units | 0-3 expected | U2 |
| V2-3 cohort key | sensitivity stratum | S1 |
| source-stated limits and V3 | not available | U1 |
| external confirmation evaluator | NOT YET WRITTEN (CONFIRMATION_PACKAGE §4.2, §5, §13; D11) | written, tested on synthetic families and sealed by amendment before any new value is read |
| HDEHP solvating-regime acid threshold (C-GD eligibility) | PENDING-U5 (CONFIRMATION_PACKAGE §10; D11) | primary SF tables, U5 item 4, by amendment before any new value is read |
| frozen in-corpus interval of the Gd-dip transport conformal-coverage reading | PENDING (CONFIRMATION_PACKAGE §7.2) | named with its nominal level by amendment before any new value is read |

Every fix is an append-only amendment (`AMENDMENTS.md`) followed by a reseal, before any new value is read.

## 3. Resume commands

Run from `D:\ml_separator_gh`, one process at a time, `PYTHONIOENCODING=utf-8`,
`PY = D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore`. LF files only.

```
# 0. seals: every earlier seal, then this round's
$PY generations/gen21/round4_20261022/scripts/r4_seal.py --verify
$PY generations/gen21/round5_20261024/scripts/r5_seal.py --verify

# 1. Stage 1, per item in PROTOCOL order: item seal -> --repro (TECHNICAL SMOKE) -> scoring run.
#    A scoring run executes its reproduction target first, its leakage checks before any scored write, and the
#    post-run seal check last; a mismatch or failed check stops the item (STOP line in RUN_RECORD.md).
$PY generations/gen21/round5_20261024/scripts/s1_ea_check.py --repro ; then [--design BPG|BP]
$PY generations/gen21/round5_20261024/scripts/s1_ect.py      --repro ; then --design both
$PY generations/gen21/round5_20261024/scripts/s1_m03.py      --repro ; then --design both
$PY generations/gen21/round5_20261024/scripts/s1_elab.py     --repro ; then --design both
$PY generations/gen21/round5_20261024/scripts/s1_window.py   --repro ; then --design both
$PY generations/gen21/round5_20261024/scripts/s1_edec.py     --repro ; then --design both

# E-D in this order: rows (exits 3 until the poisoning record is valid), poisoning, then the tables from the rows
$PY generations/gen21/round5_20261024/scripts/s1_ed.py       --repro ; then --design both
$PY generations/gen21/round5_20261024/scripts/s1_ed_poison.py   (sharded: --shard 0/3 ... --shard 2/3, then --combine 3)
$PY generations/gen21/round5_20261024/scripts/s1_ed.py       --from-rows --design both

$PY generations/gen21/round5_20261024/scripts/s1_ew.py       --repro ; then --design both (--from-rows re-runs its analysis from its rows)
$PY generations/gen21/round5_20261024/scripts/s1_ekx.py      --repro ; then --design both

# 2. after an answered request: the supply audit is rerun into its own folder, never over results/stage0/
$PY generations/gen21/round5_20261024/scripts/s0_supply.py --out generations/gen21/round5_20261024/results/supply_audit/<UTC> --request <id>
```

## 4. NOT RUN (with the reason)

| what | reason |
|---|---|
| C-ECT (chemotype-unit exact sign-flip; fixed sequence SH@4 then SH@3) | supply-audit kill (D4): decision supply 2 chemotype units against a minimum of 7 |
| Gd-dip transport (M33), with C-GD as its cation-exchanger stratum | supply-audit kill (D4) |
| A3 positive control (B1 - B0 prospective; not a claim) | no new data |
| 1.3-R13 (E-LAB external leg) | data-gated (brief 7): runs only when its stated new-data supply exists |
| 1.8-R03 (E-KX external sign rider) | data-gated (brief 7): runs only when its stated new-data supply exists |
| C-ECT-S01 (sensitivity, no Gd term for AQ_COMPLEXANT targets) | data-gated with C-ECT (D10) |

The kill is re-read after each answered request with the command of section 3.2; a changed outcome is recorded below
the marker of DECISION_LOG and here.

## 5. Status entries (append-only below the marker; UTC)

<!-- APPEND-ONLY BELOW -->

- 2026-10-24: pre-seal review folded in (`results/stage0/PRESEAL_REVIEW.md`). Stage 0 unsealed; every request OPEN;
  no round-5 score computed.
- 2026-10-24 (UTC), Stage 1 complete:
  - **Seals.** Stage 0 was sealed (`1f0cae0064a6`), then resealed twice by amendment:
    - A1 (`2f74292f5f44`): the E-D poisoning check compared fits through a lossy CSV float parse.
    - A2 (`b96c3bef9828`): the results-review reporting fixes and the E-D Pareto-family correction.
  - **Items 1.0-1.8 are DONE** under `b96c3bef9828`, reproduction first, and the post-run seal checks verified round 4,
    round 3 and round 5. Outcomes are below the DECISION_LOG marker; `REPORT.md` and `PRACTICAL_GUIDANCE.md` are
    written.
  - **Still NOT RUN** (§4 above, unchanged): C-ECT, Gd-dip transport / C-GD, A3, C-ECT-S01, 1.3-R13 and 1.8-R03.
  - **Every request remains OPEN** (§1).
  - **Resume after an answered request:** record it below both markers, rerun the supply audit into its own folder
    (§3.2), and re-read the D4 kill.
  - **Rerun of an item:** the commands of §3 apply unchanged, with these notes:
    - EA_CHECK runs both designs in one process (no `--design`), so it writes `summary.json` and `ITEM_TABLE.md`.
    - E-D runs `--repro`, then the run (exit 3 by design), then `s1_ed_poison.py`, then `--from-rows --design both`.
    - EDEC runs after ECT.
  - Nothing is committed or pushed.
