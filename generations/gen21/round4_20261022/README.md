# Gen21 round 4 (implementation)

- **Brief and evidence** (kept unchanged): `../round4_design_20261022/CLAUDE_CODE_PROMPT.md` and `RESEARCH_IDEAS.md`.
- **Round 3** is imported read-only from `../next_research_20261019/` (`r3` package). Its seal `93d3839a…` must keep verifying; the stage-0 manifest records it.
- **The strain repository** is pinned at `025f8084b52986773da98fffde13ea267c0ad908`.
  - `git clone` over this connection stalled (about 10 kB/s), so instead **every one of the 6,081 blobs** of that commit was fetched over HTTPS. Each is verified against its git blob SHA-1 from the commit's recursive tree.
  - The files are in `D:/ml_separator_refs/lanthanidestrain_025f808/`, outside this checkout, with `MANIFEST.json` and `TREE_025f808.json`.
  - It is a byte-identical copy of the tree, not a git repository. Strain code is read, never imported or executed.

## Layout

| Path | What |
|---|---|
| `PROTOCOL.md`, `DATA_CONTRACT.md`, `METRICS.md`, `RESOURCE_PLAN.md`, `EXPERIMENT_REGISTRY.csv`, `README.md` | stage-0 documents, sealed by `scripts/r4_seal.py` |
| `EVIDENCE_INVENTORY.md`, `results/task_identity.json`, `results/reproduction.json`, `results/leakage/cache_poison_nontrain_*.json` | pre-seal evidence (sealed) |
| `RUN_RECORD.md` / `AMENDMENTS.md` | append-only run log (start and end of every run) / changes after a seal, with the results seen |
| `r4/` | `paths`, `refs` (B1/B2, orphan rule), `registry`, `metrics` (balanced MAE, source clusters, jackknife-t), `donor` (DGA rule), `leakage` (poisoned bench, isolation tables) |
| `scripts/` | `r4_task_identity`, `r4_run`, `r4_compare_repro`, `r4_cache_poison`, `r4_poison`, `r4_seal` |
| `tests/test_r4.py` | 13 tests (see the module docstring) |
| `predictions/`, `results/` | row-level OOF predictions; per-configuration summaries, contrasts, levels, calibration, identity, intervention, fitted choices |

## Commands (from the repository root, Windows venv)

```bash
.venv/Scripts/python.exe -m pytest generations/gen21/round4_20261022/tests -q -p no:cacheprovider -W ignore
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_task_identity.py
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_run.py --config R3:B0 --design BP
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_compare_repro.py B0 R14 R02
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_cache_poison.py --shard 0/3
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_seal.py --seal --reason "..."
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_seal.py --verify
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_run.py --config REF --design BPG
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_run.py --config REF --design BP
.venv/Scripts/python.exe -W ignore generations/gen21/round4_20261022/scripts/r4_poison.py REF
```

**Stage 1** (PROTOCOL §10–§12; the commands are in RESUME.md).
- **Code:** blocks A / B / C are in `r4/slots_a.py`, `slots_b.py` and `slots_c.py` (block C's An(III) data layer is `r4/an_data.py`).
- **Decisions:** `report/r4_decide.py` (A), `report/r4_decide.py B` and `report/r4_decide.py C`.
- **Block C** additionally uses:
  - `scripts/r4_gate_c.py`: `--counts` (label-free), `--shard i/3`, then `--combine 3`;
  - `scripts/r4_synthetic_c.py --rep i` with `report/r4_synthetic_c_summary.py`.
- **Route enforcement:** `r4_run.py` refuses off-route block-C runs.

**How the scripts behave.**
- `r4_run.py` refuses to score a round-4 configuration unless the seal verifies. Fold checkpoints are keyed by the seal; for `R3:` reproductions the key is a digest of round 3's seal plus the wrapper code.
- **Resealing:** `r4_seal.py --seal --reason` needs an AMENDMENTS.md entry naming the old seal. The old manifest is archived.

Nothing here is committed or pushed without the user's request.
