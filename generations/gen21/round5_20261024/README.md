# Gen21 round 5: implementation folder (`generations/gen21/round5_20261024/`)

This folder implements the round-5 brief, `generations/gen21/round5_design_20261023/CLAUDE_CODE_PROMPT.md` (sections
3-10). Round 5 is not a model-search round. Stage 0 seals the conventions. Stage 1 builds and freezes E-CT and the
descriptive deliverables; every in-corpus number it produces is EXPLORATORY (seen). Stage 2 waits for new data.

## State (recorded 2026-10-24)

- **Repository.** `D:\ml_separator_gh`, branch `gen21-logsf-target`, HEAD `b96b0fc` ("gen21: add round-4 research
  analysis and Claude Code brief").
- **Untracked, uncommitted.**
  - `generations/gen21/round4_20261022/` and `generations/gen21/round5_design_20261023/`: they existed before this round
    and are left as they are.
  - `generations/gen21/round5_20261024/`: this folder, created 2026-10-24T04:40:13Z (RUN_RECORD).
  - This state comes from the session's git snapshot and RUN_RECORD; no git command was run for this file. Nothing is
    committed or pushed (brief 1.2).
- **Round-4 seal.** `5bacfe4cdcfc8d63171c46497e06b80605cc7fb1e0e7531c5fd81daadb1bbaa5` reports **VERIFIED** with
  `round4_20261022/scripts/r4_seal.py --verify`, at 04:40 UTC (RUN_RECORD) and again at 07:37:41 UTC.
- **Round-3 seal.** `93d3839a256c2185d3cb01f819881cf862d21192d968cc056b9c6c3213b39e26` (`freeze_seal.txt`) holds:
  - the raw SHA-256 of `next_research_20261019/manifests/freeze_manifest.json` equals the seal;
  - its LF digest `96cd23a9…090a075f` equals `freeze_lf_digests.json`;
  - 78 of 78 code files match their LF digests (checked directly, 2026-10-24).
- **Environment** (recorded by S0 `beta_frozen.json`): Python 3.14.5, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1,
  sklearn 1.9.0. Nothing was installed or upgraded.

## Status

- **Stage 0: UNSEALED.** There is no `manifests/stage0_seal.txt` yet. The Stage-0 inputs are computed (`results/stage0/`).
  The documents PROTOCOL, DATA_CONTRACT, METRICS, CONFIRMATION_PACKAGE, DECISION_LOG, REGISTRY.csv, DATA_REQUEST,
  EVIDENCE_INVENTORY, README and RESUME are written. The pre-seal review (workflow wf_2ceec1d9-225) is folded in:
  `results/stage0/PRESEAL_REVIEW.md` lists every finding, its verdict and what changed. The β_frozen digest drift
  (DECISION_LOG N1) is resolved by the pre-seal rerun of the Stage-0 scripts.
- **Decisions already fixed** (DECISION_LOG D1-D11; D9-D11 added at the pre-seal review):
  - β_frozen = -0.110809 (BPG, λ 0.003);
  - the external DOPT_Q is fitted on all 521 bench cells;
  - every chord statistic fails the size gate, so C-GD and Gd-dip transport become descriptive intervals, and C-ECT is
    aggregated to chemotype units (fresh size check passed);
  - the **supply-audit kill fires**, so round 5 is reported as descriptive (brief 7.6);
  - the laboratory key and the planning amplitudes are PROVISIONAL; the conservative key decides a claim's
    laboratory count (D6);
  - V2-5 censoring is primary for Gd chords and the resolution rule only flags (D9); the frozen B1 / SH keep the Gd
    term in every medium, with a registered no-Gd-term sensitivity on new data (D10); the external evaluator and the
    HDEHP acid threshold are deferred to an amendment before any new value is read (D11).
- **Stage 1.** Item code 1.0-1.8 is written (`r5/`, `scripts/s1_*.py`, `tests/`). Only two things have run: the
  pre-seal engineering checks (`--smoke`: real inputs, surrogate or suppressed labels, every metric suppressed; items
  1.1, 1.2, 1.3, 1.5 and 1.8 write `SMOKE_COUNTS_<design>.json`, items 1.0, 1.4, 1.6 and 1.7 write nothing) and the
  EA_CHECK cache build with its poisoning check (100/100 identical). **No round-5 score has been computed.**
  `REGISTRY.csv` marks every item, arm, annex row, sensitivity, control and report rule PLANNED or EXCLUDED (data-gated
  rows are PLANNED and run only when their new-data supply exists). No E-A model slot is registered.
- **Stage 2.** U1-a ... U1-i, U2, U3, U4, U5, U6, S1 and H1 were shown to the user on 2026-10-24. **All are OPEN**
  (DECISION_LOG, request-status section below the marker; `RESUME.md`).

## Layout

| path | content |
|---|---|
| `PROTOCOL.md`, `DATA_CONTRACT.md`, `METRICS.md`, `CONFIRMATION_PACKAGE.md` | Stage-0 sealed documents (brief 4.6) |
| `DECISION_LOG.md` | M35 / M34 / CP-13 summary, closure ledger, pre-slot gate, decisions D1-D11, Stage-1 report rules (pending); append-only request status and outcomes below `<!-- APPEND-ONLY BELOW -->` |
| `REGISTRY.csv` | one row per Stage-1 item and per arm, comparator, control, sensitivity, annex, appendix and report rule; status, RNG key and seed |
| `EVIDENCE_INVENTORY.md` | one page: verified, reported-only, hypotheses |
| `DATA_REQUEST.md` | the request list shown to the user (copy of the design folder's DATA_REQUESTS.md; never edited for status) |
| `SUPPLY_LEDGER.csv` | label-blind supply ledger (PROVISIONAL) |
| `RUN_RECORD.md`, `AMENDMENTS.md` | append-only run log (UTC) and amendments (results seen; reseals); the seal prefix-checks both |
| `RESUME.md` | request status, PROVISIONAL inputs and what fixes each, resume commands, NOT RUN list (append-only below its marker) |
| `r5/` | package: `paths` (seed scheme, inner-cache redirect), `common` (statistics), `frozen` (β_frozen and external DOPT_Q records), `dupes`, `population`, `supply`, `confirm`, `sealcheck`, and one module per item (`ea_check`, `ect`, `ect_report`, `m03`, `elab`, `window`, `edec`, `ed`, `ew`, `ekx`) |
| `scripts/` | `s0_*.py` (Stage-0 inputs), `s1_*.py` (Stage-1 drivers), `r5_seal.py` (seal / verify) |
| `tests/` | synthetic and construction-only tests, one file per module |
| `results/stage0/` | β_frozen, external DOPT_Q, duplicate scan, E-CT population, laboratory key, mechanism-class map, supply audit and reach, size gates, power in worlds, `PRESEAL_REVIEW.md` (the pre-seal review) |
| `results/<ITEM>/` | `ITEM_SPEC.md` (frozen item specification) and, so far, smoke counts only (EA_CHECK, ECT, ECT_M03, ELAB, WINDOW, EDEC, ED, EW, EKX) |
| `results/leakage/` | leakage-check reports (cache poisoning of item 1.0 so far; `postrun_<ITEM>.json` after sealed runs) |
| `cache/` | A/B/BR/BQ fold records (`<design>_<seed>_<fold>.pkl`), `frozen/` (all-cell β records), `ext/` (`ext_dopt.pkl`), `_inner/` (r3 inner-cache redirect), `_pytest/` |
| `manifests/` | `ea_check_caches.json` (cache keys); the Stage-0 manifest and seal once sealed |

## Commands

Run from `D:\ml_separator_gh`, one process at a time, with `PYTHONIOENCODING=utf-8` and
`PY = D:/ml_separator_gh/.venv/Scripts/python.exe -B -W ignore`. Write LF files only.

```
# earlier seals (read-only; run after every Stage-1 run)
$PY generations/gen21/round4_20261022/scripts/r4_seal.py --verify          # must print VERIFIED ok 5bacfe4c...

# Stage-0 inputs (already run; each refuses after the seal unless --amend; s0_supply.py takes --out <folder> for a
# post-seal rerun after an answered request, which never touches results/stage0/)
$PY generations/gen21/round5_20261024/scripts/s0_beta_frozen.py [--rebuild]
$PY generations/gen21/round5_20261024/scripts/s0_external_dopt.py
$PY generations/gen21/round5_20261024/scripts/s0_dupscan.py
$PY generations/gen21/round5_20261024/scripts/s0_ect_population.py
$PY generations/gen21/round5_20261024/scripts/s0_supply.py
$PY generations/gen21/round5_20261024/scripts/s0_sizegate.py [--reps 20000]
$PY generations/gen21/round5_20261024/scripts/s0_power_worlds.py [--reps 10000]
$PY generations/gen21/round5_20261024/scripts/s0_sizegate_ect_chemotype.py
$PY generations/gen21/round5_20261024/scripts/s0_package_supplements.py

# Stage-0 seal (after the pre-seal review), then verify
$PY generations/gen21/round5_20261024/scripts/r5_seal.py --seal --reason "..."
$PY generations/gen21/round5_20261024/scripts/r5_seal.py --verify

# Stage 1, per item, in this order: --smoke (PRE-SEAL ENGINEERING CHECK, allowed before the seal) -> item seal ->
# --repro (TECHNICAL SMOKE) -> run.  Every sealed run executes its reproduction target first, runs its leakage checks
# before any scored write (a failure stops it), writes START / END lines in RUN_RECORD.md and ends with the post-run
# seal check (round 4 and round 3 included); E-D writes its tables only after s1_ed_poison.py (then --from-rows).
$PY generations/gen21/round5_20261024/scripts/s1_ea_check.py --build-cache | --smoke | --repro | [--design BPG|BP]
$PY generations/gen21/round5_20261024/scripts/s1_ect.py    --smoke | --repro | --design BPG|BP|both
$PY generations/gen21/round5_20261024/scripts/s1_m03.py    --smoke | --repro | --design BPG|BP|both
$PY generations/gen21/round5_20261024/scripts/s1_elab.py   --smoke | --repro | --design BPG|BP|both
$PY generations/gen21/round5_20261024/scripts/s1_window.py --smoke | --repro | --design BPG|BP|both
$PY generations/gen21/round5_20261024/scripts/s1_edec.py   --smoke | --repro | --design BPG|BP|both
$PY generations/gen21/round5_20261024/scripts/s1_ed.py     --smoke | --repro | --design BPG|BP|both
$PY generations/gen21/round5_20261024/scripts/s1_ed_poison.py              # after the seal, after s1_ed.py (exit 3)
$PY generations/gen21/round5_20261024/scripts/s1_ed.py     --from-rows --design both   # the E-D tables
$PY generations/gen21/round5_20261024/scripts/s1_ew.py     --smoke | --repro | --design BPG|BP|both
$PY generations/gen21/round5_20261024/scripts/s1_ekx.py    --smoke | --repro | --design BPG|BP|both

# tests (synthetic / construction-only)
$PY -m pytest generations/gen21/round5_20261024/tests -q -p no:cacheprovider --basetemp generations/gen21/round5_20261024/cache/_pytest
```

The exact per-item command lines, runtimes and options are in each `results/<ITEM>/ITEM_SPEC.md`. Scoring scripts
refuse to run unless the Stage-0 seal verifies (`r5/sealcheck.py`).

## Rules carried from the brief (1.2)

- Write only inside this folder and the session scratchpad.
- Never modify `round4_20261022/`, `round4_design_20261022/`, `round5_design_20261023/`, `dataset_all_metals/`,
  `generations/gen21/cache/` or any earlier generation.
- Never search for or open the withheld-seeds file, `next_research_20261019/cache_final/` or
  `next_research_20261019/results/final/`.
- No downloads, logins or web access without the user's permission for the specific item.
- Never call `g21.cache.build_all`.
- Never run `r4_run.py`, `r4_poison.py` or `r4_cache_poison.py`.
- Every Stage-1 number is EXPLORATORY (seen). Never put one next to E-A's 0.452 unless it is on E-A's own rows.

## Status notes (append-only below the marker; UTC)

<!-- APPEND-ONLY BELOW -->

- 2026-10-24, pre-seal review folded in (`results/stage0/PRESEAL_REVIEW.md`); Stage 0 still unsealed.

- 2026-10-24 (UTC), after Stage 1:
  - **Stage 0 sealed.** First seal `1f0cae0064a6`, then A1 → `2f74292f5f44`, then A2 → `b96c3bef9828` (current;
    `AMENDMENTS.md`).
  - **Stage 1 DONE.** Every item 1.0-1.8 ran with its reproduction first; all reproduction targets matched.
  - **Review and rerun.** An adversarial results review found 65 findings, fixed under A2. Every item was rerun under
    `b96c3bef9828`. The rerun reproduced every stored number except the A2 changes (`results/a2_rerun_comparison.txt`;
    first-pass outputs in `cache/_superseded_*`).
  - **Deliverables.** Row-level outputs and per-item tables are in `results/<ITEM>/`; the leakage records are in
    `results/leakage/`. The report is `REPORT.md` (result table and verdict per estimand) and the guidance is
    `PRACTICAL_GUIDANCE.md`. The report-rule outcomes are below the DECISION_LOG marker.
  - **Not run.** C-ECT, Gd-dip transport / C-GD, A3, C-ECT-S01, 1.3-R13 and 1.8-R03 are NOT RUN (supply-audit kill or
    data-gated; `RESUME.md` §4).
  - **Open.** Every request is OPEN. Nothing is committed or pushed.
  - **Layout notes.** The text above the marker predates the seal: `results/<ITEM>/` now holds the full outputs, not
    just smoke counts. The EA_CHECK scoring run is the default both-design run, which writes `summary.json` and
    `ITEM_TABLE.md`.
