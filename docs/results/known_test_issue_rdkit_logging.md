# Known issue: two RDKit-logging guard tests fail only in the full suite

*Recorded 2026-09-04. Pre-existing — it reproduces before and after the gen7-gen11 merge, and no
source file was modified when it was observed.*

## Symptom

```
FAILED tests/test_gen8_mechanism.py::test_rdkit_log_suppression_is_scoped_not_global
FAILED tests/test_gen8_recommender.py::test_rdkit_logging_is_not_disabled_globally
```
953 passed, 2 failed, 15 deselected.

Both tests parse a deliberately broken SMILES and assert `"SMILES Parse Error"` reaches `capfd`.
In the failing runs the capture is empty.

## What is established

* **Both tests pass in isolation** (`2 passed in 6.22s`), so this is state left behind by an
  earlier test, not a defect in the guards or in the modules they guard.
* **The only unscoped `RDLogger.DisableLog` in the tree** is
  `src/lanthanide_separation/gen11/auxfeatures.py:453`, inside an `@lru_cache`'d
  `_morgan_generator()` — so it fires on first *call*, not on import. Every other call site
  correctly uses `rdBase.BlockLogs()`, which restores state on exit
  (`gen8/mechanism.py:419`, `gen11/composition.py:134`, `gen11/leakage_audit.py:97`).
* **It is nevertheless not the cause here.** A subprocess that calls `_morgan_generator()` and then
  parses a bad SMILES still gets the parse error on stderr, so in this RDKit build `DisableLog`
  does not silence the C++ stderr stream that `capfd` reads.
* Running the gen11 test module immediately before the guards does **not** reproduce it
  (28 passed).

## What is not established

Which test leaves the state behind. A bisect was started and abandoned: it competed for CPU with a
running experiment, and the issue does not affect any scientific result.

## Why it still matters

The guards exist because a process-global log disable would hide SMILES parse and sanitisation
failures across every module — exactly the class of silent data defect this project has been bitten
by elsewhere. A guard that only fires in isolation is not guarding.

## Suggested fix when someone picks this up

1. Bisect with `pytest -p no:randomly` over the files that run before `test_gen8_mechanism.py`.
2. Independently, make the guards robust to ordering by asserting on RDKit's log state directly
   (`rdBase.LogStatus()`) rather than on captured stderr, and add an autouse fixture that restores
   log state between tests.
3. Scope `auxfeatures._morgan_generator`'s `DisableLog` with `rdBase.BlockLogs()` regardless —
   it is process-global and permanent by contract even where this RDKit build tolerates it, and
   `scripts/gen8_recommend_experiment.py:319` documents that rule for the rest of the codebase.
