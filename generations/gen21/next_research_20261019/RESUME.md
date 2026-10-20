# Resume point (round 3)

**State (2026-10-19):** everything is done, including the **single final check** (run with the user's
explicit permission; `results/final/`, decision in `results/final/DECISION.json`). `results/final/RUN_ONCE.json`
exists, so the runner refuses any second run. `FINAL_REPORT_RU.md` is the report. Nothing has been committed
or pushed (brief §3). The last entries of `ATTEMPTS.md` and `AMENDMENTS.md` give the reasoning.

## Re-run anything (from the repository root, Windows venv)

```bash
.venv/Scripts/python.exe -W ignore generations/gen21/scripts/g21_build_cache.py --designs BP,BPG
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_manifest.py
.venv/Scripts/python.exe generations/gen21/next_research_20261019/scripts/r3_queue.py --jobs B0,R01,R02,R03,R04,R05,R06,R07,R08,R09,R13,R19 --workers 4
.venv/Scripts/python.exe generations/gen21/next_research_20261019/scripts/r3_queue.py --jobs R12,R15,R16,R17,R20,R21,R22 --workers 4
.venv/Scripts/python.exe generations/gen21/next_research_20261019/scripts/r3_queue.py --jobs R23,R24,R14,R18,CTRL23 --workers 4
.venv/Scripts/python.exe generations/gen21/next_research_20261019/scripts/r3_queue.py --jobs CTRL14W,CTRL23W,CTRL02W --workers 4
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_policy_cross.py --pol R09 --dopt B0
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_cache_poison.py
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_poison_chain.py R02
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_poison_chain.py R14
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_poison_chain.py R23
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_intervention.py --config R23 --design BP
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_ladders.py
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_cross2x2.py
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_diagnostics.py R23 R14 R18
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_selection_optimism.py
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_calibration.py
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_table.py
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_repro.py R02 BP generations/gen21/next_research_20261019/results/leakage/repro_R02_BP.json
.venv/Scripts/python.exe -m pytest generations/gen21/next_research_20261019/tests -q -p no:cacheprovider -W ignore
```

Notes:
* `r3_run.py` (called by the queue) checkpoints every outer fold under `predictions/_parts/<ID>_<design>/`. A
  rerun skips finished folds; delete a config's `_parts` folder to recompute it. Shared inner cross-fitted HBK
  predictions are cached in `predictions/_inner/`. The queue appends to `results/<ID>_<design>.log`.
* `r3_intervention.py` reruns the hidden-label intervention from the checkpoints with the current harness;
  repeat it for `R24` and `R18` (both designs) as well.
* The 2 × 2 policy × correction contrasts: `scripts/r3_cross2x2.py` (`results/policy_cross/contrasts_2x2_*.csv`).
* `r3_calibration.py`: see its docstring for the exact coverage.

## The freeze (done: seal `93d3839a256c2185…`; redo only if the code changes)

```bash
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_freeze.py --config R02 --column COR_G --secondary R14 --secondary R23 --reason "PROTOCOL section 7 applied as written (after the review): primary regime DOPT_Q; R14 and R08 fail section 8.6; simplicity clause among R22/R19/R02/R07; R02 passes section 8.6 (CTRL02W)"
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_final_runner.py --dry-check
```

Before hashing, the freeze runs the runner's compute() on one dev fold per design and requires bit-identical
rows (`results/leakage/runner_smoke.json`); it then hashes every repository module that run imported. Any
later code edit, data change or package-version change makes the runner refuse; then re-run the repro, the
tests and the freeze, and record it.

## The final check (RUN ONCE on 2026-10-19; it cannot be repeated)

Re-apply the frozen decision rule to the stored final outputs (read-only):

```bash
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_final_decision.py
```

The command that was run (kept for the record):

```bash
.venv/Scripts/python.exe -W ignore generations/gen21/next_research_20261019/scripts/r3_final_runner.py --seeds-file "<path of the withheld seeds file, supplied by the user>" --i-have-explicit-user-permission "<quote of the user's message>"
```

The runner checks the freeze seal, the code identity and the seeds-file hash against
`../manifests/final_seeds_commitment.txt`. It writes `results/final/RUN_ONCE.json` before computing and
refuses a second run. It computes R02 (candidate) and R14 / R23 (secondary, descriptive) on the same final
folds, asserts fold isolation on every fold and runs the intervention on every fold.

## Remaining actions

1. Done: the final check (once).
2. **User decision:** whether to commit / push round 3 (brief §3: only on a separate request). Note:
   Sizes: the directory is 731 MB, of which `predictions/_parts/` (per-fold checkpoints, 657 MB) and
   `predictions/_inner/` (15 MB) are regenerable caches, and `cache_final/` (8.7 MB) holds final-seed fold
   caches; none of them is covered by `../.gitignore` (which ignores only the dev `cache/`). The row-level
   OOF predictions (`predictions/<ID>_<design>.parquet`, about 40 MB) and `results/` (9 MB) are the deliverables.
3. Open sensitivity, not run: V2 without the value-pattern rule V2-5 (R17/R18), `AMENDMENTS.md` D-7.
4. Not done in round 3: the brief's solvent / extractant-concentration × series-shape interaction (the
   descriptors fingerprint publications; `ATTEMPTS.md`, chemistry pinning note).
5. For a next round: the registered §8.6 control cannot judge a pair-type-level parameter; a protocol for
   such a parameter needs a control that keeps support information but breaks the label-to-position link
   in a pre-declared way. Write it before any run.
6. Laboratory data: `EXPERIMENTAL_DATA_REQUEST.md`.
