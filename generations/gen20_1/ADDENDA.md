# gen20_1 — dated addenda to the sealed protocol

`PRE_REGISTRATION.md` (SEALED-SHA256 `f34a1b97433b79b37024ecd8fd5349eeb0d4853b5d7cad60ed21470f1be2d232`) and the frozen
`IMPLEMENTATION_PLAN.md` (LF digest `453654a4bbc9b56ce7d716568b82c1ddb2e4058e5fb434b575a0af7de06541c1`) are not edited after
the seal (2026-10-16T15:03:59Z, `manifests/seal_g201.json`). Every operational deviation and every reading of the sealed
text is recorded here, with what changed, why, whether S5 statistics, smoke output or results had been seen, and whether
any number could change. The S12 audit (`results/AUDIT_S12.md`) found no reported number, class or verdict affected.

## Addendum 1 — 2026-10-17: operational record of S5–S12 (written after S11; results had been seen)

1. **S6 STOP and the counts-script fix (implementation, no numeric effect).** The first S6 re-run of
   `scripts/g201_counts.py --label S6` (manifest `manifests/g201_counts__S6.json`, status STOP) counted the 23
   `row_manifests/COH-REPCAP*.parquet` files as unexpected S3 reference files. Those files are label-derived S5 outputs
   written by `scripts/g201_describe.py` (digests in `manifests/describe_g201.json`), never by the counts script. All 32
   genuine S3 reference files were byte-identical. `reference_files()` was changed to skip `COH-REPCAP*`
   (new sha256 of `scripts/g201_counts.py`: `83a0fc3a406730f2bc795114694ae37df58e123392917cf4b5c260724899dd85`, edited
   2026-10-16T17:41:16Z, after S5 at 17:39:25Z and before discovery at 17:49:16Z), and `--label S6b` passed (32 reference
   files byte-identical; `manifests/g201_counts__S6b.json`). S5 output had been produced; no model output existed. The
   counts tests (`tests/test_s3_counts_script.py`) pass with the edited script.
2. **Stage order and pre-run fits.** The S6 code and tests were built before S5 (the plan lists S5 before S6); this is
   the safer direction (no target statistic could steer the code). During the build, real-data fits of BP split seed
   104729 folds 0 and 1 were run under smoke labels outside the repository to check integration and timing; every y
   column was overwritten with synthetic values before any scoring code ran, and no metric against outer-test y was
   computed. The fitting-code digest changed during the build (`acadf4af…` → `42e08d2b…` → `7e869f0a…`); every S8, S9
   and S11 unit was produced under `7e869f0a3d95e82edd1479d4d43d2f973984e84947bb30ec8207b54b3197b17b`, which still equals
   the code on disk.
3. **Processes and threads (no numeric effect).** S9 ran two concurrent processes on whole designs (`--workers 2`),
   as the S4 resolution of O18 in `PRE_REGISTRATION.md` §8.1 allows; the frozen plan's compute paragraph (§9) and §10
   row 10 still say one process, so the sealed documents disagree on this point and the pre-registration governs.
   S9 used `LOKY_MAX_CPU_COUNT = 4` (the plan's "≤ 4 after smoke"); **S11 used `LOKY_MAX_CPU_COUNT = 6`, above that
   limit** (`manifests/g201_confirm__confirmation.json`). Fixed-seed ExtraTrees fits do not depend on n_jobs: the S12
   audit refitted confirmation unit BP primary s915723 f3 at `LOKY_MAX_CPU_COUNT = 2` and found all 44 arm columns and
   the selection record bitwise equal, and the S8 smoke tables (one process, 2 threads) equal the discovery tables
   (two processes, 4 threads) bitwise on all 8 shared tables.
4. **No git commit at S4.** The plan's S4 row says "commit"; nothing was committed, because the user had asked that
   nothing be committed before their review. The seal and the confirmation-seed commitment are anchored only by the
   in-folder manifests, file digests and timestamps. Any later commit is made after S11 and is not a pre-outcome anchor.
5. **Scoring-side code digests.** The run manifests pin the fitting code (digest above) but not the code that turned
   saved predictions into tables, classes and verdicts. Its sha256 at S10/S11 (all files last modified before S9
   started and unchanged through S11, per the S12 audit): `gen201/metrics.py`
   `e7c5e63d826f10256509aaa4df8083e3b3641dd96eae2cdc8b3186b0bc7453ae`, `gen201/inference.py`
   `abd5bb890ea04cc2a22df5505e4ae04fbcd556a287fe26b281c6e6614dbe293f`, `gen201/rules.py`
   `46f0128bb4686053cf3ca906ea2e6b54d3f9c88ed558ca5ebe16f9901331b6b1`, `gen201/report.py`
   `e3b07650c7929dba87d4ef6e696dca7398098c3a8bf2a6ccec475697bcf873dd`, `scripts/g201_report.py`
   `ce722cbd372e0c9a97058e730df3bae1507d75e0f087d08535af7ae804a03c69`, `scripts/g201_confirm.py`
   `f77640e11dab1575552cf3abf97cd23cff75284acc755db17b1412858346fb29`.
6. **Readings of the generated report (no number changes; `GEN20_1_REPORT.md` is left as the scripts wrote it).**
   - The lines labelled "NO-REACHED" are the code's label for the §7 fallback ("other combinations: verdict by
     verdict"), not a pre-written sentence ID. The classes they list are the ones §9 expected, so they are not a
     "pattern not anticipated".
   - "3 of 83 tests have q ≤ 0.05": one is favourable (EX-11, B over BL_SF_INNER, Δ +0.0107) and two are unfavourable
     (EX-14, BD_METAL_SF and BD_NN_SF against FLAT, Δ −0.174 and −0.143: log SF differenced from those log D baselines
     is much worse than predicting zero). Exploratory results are never promoted.
   - The §1 cross-target sentence must cite both log D populations (O27): COH-D-C3 (2,425 entries; decides H1, T1, T1w)
     and COH-D-C3M (2,208 entries under the scored log SF pairs). Both cards, and the MET-23 reliability panels, are in
     `results/discovery/tab07_cross_target.csv` and `tab07_reliability_MET23.csv`.
   - The final `GEN20_1_REPORT.md` was re-rendered by `scripts/g201_confirm.py` (S11) through `report.render`, although
     its footer credits S10. `results/discovery/*.csv` are the S10 snapshot: their annotation "confirmation registered
     (S11), not yet run" predates S11; the current confirmation status is `results/confirmation/confirmation_status.json`
     and the report.
7. **README.** `README.md` (not sealed) was updated after S11 with the run status and a results summary.
8. **Two tests updated after S11 (test code only).** `tests/test_confirm_script.py::test_run_once_guard` and
   `tests/test_report_annotations_digest.py::test_confirm_resume_compares_the_code_digest` ended with
   `assert not RUN_ONCE.exists()`, a pre-run assumption that became false once S11 wrote the real
   `results/confirmation/RUN_ONCE.json`. Both now assert that the test leaves the real file byte-identical (sha256
   `0462f94274c31a08a34869318acd1793659835d6234638fb9988426332daa93c`); the guard logic they test is unchanged. The full suite
   then passes (288 tests).
9. **Line endings for the commit (2026-10-17).** On first committing the folder (branch `gen20-1-logd-vs-logsf`, after
   S11), `generations/gen20_1/.gitattributes` was added with `* text=auto eol=lf` and `*.parquet binary`, so that a
   checkout on a `core.autocrlf=true` clone keeps the LF bytes whose digests the manifests record. No file content
   changed (every file was already LF); the sealed digests are over LF-normalised text in any case.
