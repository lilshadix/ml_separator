# Resume point (round 4)

**Current seal:** `5bacfe4c…` (amendment A5, close-out of block C; the stage-1 campaign is complete). The history is in `manifests/stage0_seal_*.txt` and the archived manifests.

**Nothing is committed or pushed.** `predictions/_parts/` (about 0.2 GB) and `predictions/_inner/` are regenerable caches; add them to `../.gitignore` before any commit.

## State (2026-10-23)

| stage | status |
|---|---|
| Stage 0: references B0 / B1 / B2 | done. B1 0.452 (BPG) is the reference; B2 lost to it (STAGE0_REPORT.md) |
| Provenance, source recovery, power | done (PROVENANCE.md, POWER.md, DATA_REQUEST.md) |
| Block A (S01, S02; A1) | done: **no gain over B1**. S03, S04 and S21 were NOT RUN (routes closed). Leakage chain complete |
| Block B (S05, S06; A2) | done: **worse than B1** (S05 −0.006 / −0.003, S06 −0.006 / −0.011). S07–S10 and S22 were NOT RUN (routes closed, encoder gate 9/50). Leakage chain complete |
| Block C (transfer S11, S13–S16, S24; A4) | done: **S11 no gain over B1** (−0.0005 BPG; −0.0017 BP, slightly worse in balanced MAE). G-transfer's inner part failed (3 of 50), so S13–S16 were NOT RUN. S24 was NOT RUN on its count gate |
| S12, S17, S18 | NOT RUN (gates failed) |
| S19, S20, S23 | NOT RUN (final: no candidate passed criterion 1; no gain to rescale) |

**Verdict (REPORT.md):** no detected gain over B1. The finalist is the reference B1. The study is underpowered for confirmation on the existing corpus.

## Commands (repository root, Windows venv)

```bash
.venv/Scripts/python.exe -m pytest generations/gen21/round4_20261022/tests -q -p no:cacheprovider -W ignore
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_seal.py --verify
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_run.py --config S05 --design BPG
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_run.py --config S06 --design BPG
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/report/r4_decide.py B
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_poison.py S06
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/report/r4_table.py S06
```

- **Other configurations:** run S05 / S06 on BP the same way. REF, S01 and S02 were scored under earlier seals: their results stand, and rerunning them needs the current seal to verify. The stage-0 commands are in README.md.
- **Designs:** BPG decides and BP must agree. Never more than 3 parallel jobs (RAM).

## Open items for the user

1. **The Shimada thesis PDF** (DOI 10.14945/00003509, about 24 MB) needs your permission to download. It settles the DMDPhPDA Pr/Nd value.
2. **The C5-BTBP source** (Nilsson 2006 vs Foreman 2005) needs journal access.
3. **SAFE** needs your login.
4. **`AUDIT_ROUND3_2026-10-22.md`** is not in the repository.
5. **Independent confirmation data** (DATA_REQUEST.md): about 40 or more new sources are needed for adequate power.
6. **Commit and push policy.**

## Block C (sealed A4 `b4c2eea0`, 2026-10-23)

**Outcome.**
- **S11**, the Ln-only multitask scaffold, gives no gain over B1:
  - ΔR² vs B1 −0.0005 (BPG) / −0.0017 (BP);
  - Δbalanced-MAE −0.0000 / −0.0020;
  - criterion 1 fails;
  - inner CV chose B1 in 24 / 23 of 25 folds.
- **Checks:** interventions passed (99,110 predictions, max change 0), and slot identity passed.
- **G-transfer** (`results/gate_c.json`):
  - the count part passes (min 557 rows, 12 copy groups, 77 extractants);
  - the inner part fails: S13 beat S11 and B2 in 3 of 50 fold-fits;
  - so S13, S14, S15 and S16 were NOT RUN, and their slots are not consumed;
  - `r4_run.py` refused S13, as designed (RUN_RECORD).
- **S24** was NOT RUN on its count gate.
- **Synthetic run 4:** the transfer claim fired 0/6 in every world. The gate failure is therefore read as "not detected; control power not shown". This is a reading, not a mechanical label: `r4_decide.py` issues that label only for a run S13.

**Closed.**
- **S11 fit poisoning:** 50/50 bit-identical (`results/leakage/poison_S11.json`).
- **A5:** sealed as `5bacfe4c`.
- **Campaign accounting:** 5 of 24 slots used (S01, S02, S05, S06, S11). No slot beats B1, so the finalist is the reference B1.
- **Nothing is committed.** `predictions/_parts/` and `predictions/_inner/` are regenerable caches; add them to `.gitignore` before any commit.
