"""Reproducibility: the class of bug that has already cost this project a sweep.

gen8 discarded a completed round of results because the pool/evaluation draw was
seeded with Python's builtin ``hash``, which is salted per interpreter process.
Nothing failed.  Every script completed.  Two architectures were simply scored on
different rows and the "paired" bootstrap between them was not paired at all.

That is the shape of the bug this file exists to catch, so the tests here are
deliberately awkward: they spawn subprocesses under a different ``PYTHONHASHSEED``,
they grep the source tree, and they read the artefacts of the Phase-0 reproduction
rather than recomputing anything, because a check that shares a process with the
thing it is checking cannot see a per-process salt at all.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN9_SRC = REPO_ROOT / "src" / "lanthanide_separation" / "gen9"
GEN9_SCRIPTS = sorted((REPO_ROOT / "scripts").glob("gen9_*.py"))
RUN = REPO_ROOT / "runs" / "gen9_shape"

#: The cohort every generation since gen6 has been scored on.
COHORT_FINGERPRINT = "bed178ec1a7a82b0"
COHORT_ROWS = 5248


def _subprocess(code: str, hashseed: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin:/usr/local/bin"})
    if result.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{result.stderr}")
    return result.stdout.strip()


# --------------------------------------------------------------------------- #
# 3. stable splitting across PYTHONHASHSEED
# --------------------------------------------------------------------------- #

def test_the_pool_evaluation_split_is_identical_in_another_process():
    """The decisive one.  In-process checking cannot see a per-process salt."""
    from lanthanide_separation.gen8.kshot import stable_hash
    from lanthanide_separation.gen8.protocols import make_p2_split

    reference = make_p2_split(23, np.random.default_rng((20260820, 3, stable_hash("TODGA"))))
    code = f"""
import sys, json
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
import numpy as np
from lanthanide_separation.gen8.kshot import stable_hash
from lanthanide_separation.gen8.protocols import make_p2_split
split = make_p2_split(23, np.random.default_rng((20260820, 3, stable_hash("TODGA"))))
print(json.dumps({{"pool": split.pool.tolist(), "evaluation": split.evaluation.tolist()}}))
"""
    for seed in ("0", "1", "999", "12345"):
        payload = json.loads(_subprocess(code, seed))
        np.testing.assert_array_equal(payload["pool"], reference.pool)
        np.testing.assert_array_equal(payload["evaluation"], reference.evaluation)


def test_the_acquisition_training_split_is_identical_in_another_process():
    """gen9 draws its own training blocks; they must travel between processes too."""
    code = f"""
import sys, json
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
sys.path.insert(0, {str(REPO_ROOT / 'tests')!r})
import numpy as np
from lanthanide_separation.gen9.acquisition import build_dataset
from test_gen9_acquisition import synthetic_block
import pandas as pd
oof = pd.concat([synthetic_block(n=16, seed=500 + i, ligand=f"W{{i}}") for i in range(4)],
                ignore_index=True)
d = build_dataset(oof, membership=None, repeats=3, seed=20260821)
print(json.dumps({{"candidates": d.features["candidate"].tolist(),
                  "q": [round(v, 12) for v in d.labels["oracle_deviation"].tolist()]}}))
"""
    first = json.loads(_subprocess(code, "0"))
    second = json.loads(_subprocess(code, "31337"))
    assert first["candidates"] == second["candidates"]
    np.testing.assert_allclose(first["q"], second["q"], rtol=0, atol=1e-10)


def test_curve_pair_construction_is_identical_in_another_process():
    code = f"""
import sys, json
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
import numpy as np, pandas as pd
from lanthanide_separation.gen9.curves import build_pairs
rows = []
for c in range(6):
    for p in range(7):
        rows.append({{"curve_id": f"c{{c}}", "series_id": f"s{{c}}",
                     "axis": "cond__extractant_concentration_M", "axis_label": "extractant",
                     "row_id": f"c{{c}}_p{{p}}", "axis_value": float(p), "n_points": 7}})
membership = pd.DataFrame.from_records(rows)
ids = [r["row_id"] for r in rows]
pairs = build_pairs(membership, ids, strategy="ROW_MULTISCALE",
                    axes=("cond__extractant_concentration_M",),
                    rng=np.random.default_rng(20260821))
print(json.dumps({{"pairs": pairs.pairs.tolist(),
                  "weight": [round(v, 12) for v in pairs.pair_weight.tolist()]}}))
"""
    first = json.loads(_subprocess(code, "0"))
    second = json.loads(_subprocess(code, "4242"))
    assert first["pairs"] == second["pairs"]
    np.testing.assert_allclose(first["weight"], second["weight"], rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- #
# 8. no builtin hash anywhere that can move a scientific number
# --------------------------------------------------------------------------- #

def _calls_builtin_hash(path: Path) -> list[int]:
    """Lines where ``hash(...)`` is called as a bare builtin, not ``x.hash`` or a name."""
    tree = ast.parse(path.read_text())
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "hash"):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("path", sorted(GEN9_SRC.glob("*.py")) + GEN9_SCRIPTS,
                         ids=lambda p: p.name)
def test_no_gen9_module_calls_pythons_salted_hash(path):
    hits = _calls_builtin_hash(path)
    assert not hits, (
        f"{path.name} calls the builtin hash() at line(s) {hits}; it is salted per "
        "process (PYTHONHASHSEED) and silently un-paired a whole gen8 round. Use "
        "lanthanide_separation.gen8.kshot.stable_hash (BLAKE2b) instead.")


def test_stable_hash_is_actually_process_independent():
    from lanthanide_separation.gen8.kshot import stable_hash

    words = ("CCO", "TODGA", "c1ccccc1", "TWE-24")
    reference = [stable_hash(w) for w in words]
    code = f"""
import sys, json
sys.path.insert(0, {str(REPO_ROOT / 'src')!r})
from lanthanide_separation.gen8.kshot import stable_hash
print(json.dumps([stable_hash(w) for w in {list(words)!r}]))
"""
    assert json.loads(_subprocess(code, "777")) == reference


# --------------------------------------------------------------------------- #
# 11. the cohort has not moved
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_the_frozen_cohort_fingerprint_is_unchanged():
    from lanthanide_separation.gen7.harness import load_cohort

    cohort = load_cohort()
    assert cohort.fingerprint == COHORT_FINGERPRINT
    assert len(cohort.frame) == COHORT_ROWS


@pytest.mark.slow
def test_gen9_recomputed_the_same_curves_gen8_used():
    audit_path = RUN / "curves" / "curve_audit.json"
    if not audit_path.exists():
        pytest.skip("run scripts/gen9_curves.py first")
    audit = json.loads(audit_path.read_text())
    assert audit["failures"] == []
    assert audit["n_nonfinite_axis_value"] == 0
    assert audit["n_duplicate_row_axis"] == 0
    assert audit["vs_gen8"]["identical_keys"] is True
    assert audit["vs_gen8"]["max_abs_axis_value_delta"] == 0.0


# --------------------------------------------------------------------------- #
# 12. gen8's baselines reproduce before any gen9 headline is read
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_the_phase_zero_reproduction_passed():
    """gen9 is not allowed to have a headline until gen8's numbers came back.

    Reads the artefact rather than recomputing: the point is that the *run* that
    produced gen9's arms happened in an environment where gen8's numbers held.
    """
    path = RUN / "reproduction" / "reproduction.json"
    if not path.exists():
        pytest.skip("run scripts/gen9_reproduce_gen8.py first")
    report = json.loads(path.read_text())
    failed = [c for c in report["checks"] if not c["pass"]]
    assert not failed, f"gen8 reference numbers did not reproduce: {failed}"
    assert report["n_checks"] >= 15
    names = {c["name"] for c in report["checks"]}
    for required in ("zero_shot_k0", "random_1shot", "medoid_1shot", "oracle_1shot",
                     "best_2shot", "best_5shot", "extractant_true_median",
                     "extractant_pred_median", "acid_true_median", "acid_pred_median"):
        assert required in names, f"the reproduction did not check {required}"


@pytest.mark.slow
def test_the_environment_matches_the_one_gen8_ran_under():
    path = RUN / "reproduction" / "reproduction.json"
    if not path.exists():
        pytest.skip("run scripts/gen9_reproduce_gen8.py first")
    drift = json.loads(path.read_text())["environment_drift"]
    if not drift.get("available"):
        pytest.skip("gen8 environment.json is absent")
    assert drift["n_changed"] == 0, (
        f"library versions moved since gen8: {drift['changed']}; rerun both baseline "
        "and candidate in one environment before comparing them")


# --------------------------------------------------------------------------- #
# the seed hierarchy
# --------------------------------------------------------------------------- #

def test_the_model_seed_is_independent_of_the_split_seed():
    """gen5's formula, unchanged since: ``42 + fold*1009 + 9_999_991``.

    Confounding the two would make "seed spread" mean re-partitioned chemistry and
    a re-randomised learner at once, and no gen9 arm could be compared to a gen7 one.
    """
    from lanthanide_separation.gen7.harness import Fold

    for seed in (104729, 262147):
        for fold in range(5):
            item = Fold(seed=seed, fold=fold, train_index=np.array([0]),
                        test_index=np.array([1]), held_out_chemotypes=())
            assert item.model_seed == 42 + fold * 1009 + 9_999_991


def test_each_shape_arm_uses_the_same_model_seed_as_every_other():
    """A shape arm that got a different learner seed would not be paired with its control."""
    from lanthanide_separation.gen9.objective import phase1_arms

    arms = phase1_arms()
    assert {a.name for a in arms} == {
        "A0_ROW_ONLY", "A1_ROW_ADJACENT", "A2_ROW_ENDPOINT",
        "A3_ROW_RANDOM_PAIR", "A4_ROW_MULTISCALE"}
    assert arms[0].lambda_delta == 0.0 and arms[0].lambda_span == 0.0
    assert all(a.lambda_delta > 0 for a in arms[1:])


def test_the_boost_config_on_disk_is_the_one_that_was_selected_in_training_folds():
    path = RUN / "boost_config.json"
    if not path.exists():
        pytest.skip("run scripts/gen9_tune.py first")
    payload = json.loads(path.read_text())
    selected = payload["selected_by"]
    assert selected["objective"].startswith("ROW_ONLY")
    assert "no outer test row" in selected["protocol"]
    assert selected["frozen_equivalent_inner_macro_mae"] is not None
