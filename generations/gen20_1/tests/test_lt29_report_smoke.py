"""LT-29: the report refuses ``smoke*`` labels (module, scorer, build and the S10 script), before the seal check and
before reading or writing anything."""
from __future__ import annotations

import importlib.util

import pytest

from gen201 import bootstrap
from gen201 import guard
from gen201 import report as RP
from gen201 import seal


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("g201_report_script", bootstrap.G201 / "scripts" / "g201_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("label", ["smoke", "smoke_bp", "Smoke1", " smoke-s8", "SMOKE"])
def test_refuse_smoke(label):
    with pytest.raises(RP.ReportRefused, match="LT-29"):
        RP.refuse_smoke(label)


def test_other_labels_pass():
    for label in ("discovery", "confirmation", "nosmoke"):
        RP.refuse_smoke(label)


def test_scorer_and_build_refuse_before_anything(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be reached for a smoke label")
    monkeypatch.setattr(seal, "require_sealed", boom)
    with pytest.raises(RP.ReportRefused):
        RP.Scorer("smoke_s8", loader=boom, status=boom, manifests={}, selections=boom)
    with pytest.raises(RP.ReportRefused):
        RP.build("smoke_s8", out_dir=bootstrap.RESULTS / "smoke_s8")
    assert not (bootstrap.RESULTS / "smoke_s8").exists()


def test_script_refuses_smoke(script, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(script, "require_sealed", lambda *a, **k: called.append(1))
    assert script.main(["--label", "smoke_s8"]) == 2
    assert not called and "REFUSED" in capsys.readouterr().err
    assert not (bootstrap.RESULTS / "smoke_s8").exists()
    assert guard.installed()["gen20k.manifest"]          # the script installs the forbidden-writer guard
