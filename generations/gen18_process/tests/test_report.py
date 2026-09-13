"""Tests for ``gen18proc.report`` (DESIGN.md section 12.2, row ``test_report``).

Run:  .venv/Scripts/python.exe -m pytest generations/gen18_process/tests/test_report.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

G18 = Path(__file__).resolve().parents[1]
if str(G18) not in sys.path:
    sys.path.insert(0, str(G18))

from gen18proc import report as R  # noqa: E402

REGIME = {"cohort": "E1", "holdout": "lopo", "averaging_unit": "system"}


def _df() -> pd.DataFrame:
    return pd.DataFrame({"system_id": ["a", "b"], "mae_M1": [0.123456789, np.nan],
                         "n": [3, 4], "note": ["x|y", ""]})


@pytest.mark.parametrize("missing", ["cohort", "holdout", "averaging_unit"])
def test_write_table_refuses_without_each_regime_key(tmp_path, missing):
    regime = {k: v for k, v in REGIME.items() if k != missing}
    with pytest.raises(ValueError):
        R.write_table(_df(), tmp_path / "t.csv", regime=regime)
    with pytest.raises(ValueError):
        R.write_table(_df(), tmp_path / "t.csv", regime={**REGIME, missing: ""})
    with pytest.raises(ValueError):
        R.write_table(_df(), tmp_path / "t.csv", regime=None)      # type: ignore[arg-type]
    assert not (tmp_path / "t.csv").exists()


def test_regime_table_needs_status_of_parameters(tmp_path):
    with pytest.raises(ValueError):
        R.write_table(_df(), tmp_path / "r.csv", regime=REGIME, regime_table=True)
    out = R.write_table(_df(), tmp_path / "r.csv",
                        regime={**REGIME, "status_of_parameters": "ASSUMED_PLACEHOLDER"},
                        regime_table=True)
    assert out.read_text(encoding="utf-8").startswith(
        "# regime: cohort=E1; holdout=lopo; averaging_unit=system; "
        "status_of_parameters=ASSUMED_PLACEHOLDER\n")


def test_csv_round_trip_with_regime_header(tmp_path):
    out = R.write_table(_df(), tmp_path / "t.csv", regime={**REGIME, "zeta": "1", "alpha": "2"})
    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == (
        "# regime: cohort=E1; holdout=lopo; averaging_unit=system; alpha=2; zeta=1")
    df, regime = R.read_table(out)
    assert regime == {**REGIME, "alpha": "2", "zeta": "1"}
    assert df.shape == (2, 4) and df["system_id"].tolist() == ["a", "b"]
    plain = pd.read_csv(out, comment="#")
    assert plain.shape == (2, 4)
    # byte-identical across two writes
    out2 = R.write_table(_df(), tmp_path / "t2.csv", regime={**REGIME, "zeta": "1", "alpha": "2"})
    assert out2.read_bytes() == out.read_bytes()


def test_markdown_table_and_caption(tmp_path):
    md = R.markdown_table(_df(), floatfmt=".3g")
    lines = md.splitlines()
    assert lines[0] == "| system_id | mae_M1 | n | note |"
    assert lines[1] == "|---|---|---|---|"
    assert lines[2] == "| a | 0.123 | 3 | x\\|y |"
    assert lines[3] == "| b | nan | 4 |  |"
    out = R.write_table(_df(), tmp_path / "t.md", regime=REGIME)
    text = out.read_text(encoding="utf-8")
    assert text.startswith(
        "*regime: cohort=E1; holdout=lopo; averaging_unit=system*\n\n| system_id")
    empty = R.markdown_table(pd.DataFrame(columns=["a", "b"])).splitlines()
    assert empty == ["| a | b |", "|---|---|"]


def test_manifest_hashes_inputs_and_outputs_without_wall_clock(tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text("hello", encoding="utf-8")
    out = tmp_path / "out.csv"
    R.write_table(_df(), out, regime=REGIME)
    m = R.manifest([out, tmp_path / "missing.csv"], [inp], seed=18, arguments={"n": 5})
    assert m["seed"] == 18 and m["arguments"] == {"n": 5}
    assert list(m["inputs"].values())[0] == \
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert m["outputs"][out.as_posix()] is not None
    assert m["outputs"][(tmp_path / "missing.csv").as_posix()] is None
    assert m["git_head"] is None or len(m["git_head"]) == 40
    assert not any(k in json.dumps(m).lower() for k in ("timestamp", "time", "date", "clock"))
    p = R.write_manifest(tmp_path / "manifest.json", [out], [inp], 18)
    assert json.loads(p.read_text(encoding="utf-8"))["schema"] == "gen18.manifest.1"
