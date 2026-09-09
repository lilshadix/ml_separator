"""Gen16 anchor regression tests: the frozen bench has not drifted.

Run (cheap, ~30 s):  .venv/Scripts/python.exe -m pytest gen16_leads/tests -q -m "not slow"
Run (all):           .venv/Scripts/python.exe -m pytest gen16_leads/tests -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

G16 = Path(__file__).resolve().parents[1]
if str(G16) not in sys.path:
    sys.path.insert(0, str(G16))
from gen16 import bootstrap  # noqa: E402,F401  (sys.path + thread cap)
from gen16 import anchors as AN  # noqa: E402
from gen15 import valuebench as V  # noqa: E402

RECORDED_PATH = G16 / "results" / "anchors" / "anchors.json"
RECORDED = json.loads(RECORDED_PATH.read_text(encoding="utf-8"))
HASH_SCRIPT = G16 / "scripts" / "g16_fold_hash.py"


@pytest.fixture(scope="module")
def bench():
    from gen14.dirbench import load
    return load()


@pytest.fixture(scope="module")
def bp_board(bench):
    B, C, _ = AN.value_board(bench, ["BP"])
    return V.wide(B), C


def test_fold_plan_hash_matches_recorded_in_subprocess():
    """Computed in a fresh process: per-process hash() salting cannot mask a drift."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, str(HASH_SCRIPT)], capture_output=True, text=True,
                       check=True, env=env)
    got = json.loads(r.stdout.strip().splitlines()[-1])
    rec = RECORDED["fold_plan"]
    assert got["n_folds"] == rec["n_folds"]
    assert got["sha256"] == rec["sha256"]
    assert got["sha256_train_test"] == rec["sha256_train_test"]


def test_g14_and_flat_under_bp_to_four_decimals(bp_board):
    """Anchors 2 and 3: extractant-macro MAE of log SF, design BP, 5 discovery seeds."""
    W, _ = bp_board
    g14, flat = float(W.loc["G14", "BP"]), float(W.loc["FLAT", "BP"])
    assert round(g14, 4) == 0.5001, f"G14 @ BP = {g14!r}"
    assert round(flat, 4) == 0.5885, f"FLAT @ BP = {flat!r}"


def test_g14_flat_and_contrast_under_bp_full_digits(bp_board):
    """Stricter drift guard against the digits recorded by g16_anchors.py (and by gen15)."""
    W, C = bp_board
    a = RECORDED["anchors"]
    assert float(W.loc["G14", "BP"]) == pytest.approx(
        a["G14_BP_macro_mae_extractant"]["obtained"], abs=1e-12)
    assert float(W.loc["FLAT", "BP"]) == pytest.approx(
        a["FLAT_BP_macro_mae_extractant"]["obtained"], abs=1e-12)
    row = C[(C.design == "BP") & (C.comparison == "G14_vs_FLAT")].iloc[0]
    for k, v in a["G14_vs_FLAT_BP"]["obtained"].items():
        assert float(row[k]) == pytest.approx(v, abs=1e-12), k


@pytest.mark.slow
def test_gen13_stage3_headline_exact(bench):
    """Anchor 1: G13_ET_TOPO39 under BP, macro direction accuracy, equal to the last digit."""
    acc, _, _ = AN.g13_direction_anchor(bench)
    assert acc == 0.7683085207475452, f"got {acc!r}"
