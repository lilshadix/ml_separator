"""scripts/g201_confirm.py (S11) without fitting anything: the seed and digest assertions against the committed
manifests, the RUN_ONCE guard, the confirmation status, and the scoring with (h) taken from discovery (synthetic
tables)."""
from __future__ import annotations

import importlib.util
import json

import pytest

from gen201 import bootstrap
from gen201 import report as RP
from gen201 import rules as R

from test_report_synthetic import build_world, scorer


@pytest.fixture(scope="module")
def C():
    spec = importlib.util.spec_from_file_location("g201_confirm_script", bootstrap.G201 / "scripts" / "g201_confirm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_seeds_and_commitment(C):
    assert C.check_seeds() == sorted(C.CONF_SEEDS)
    assert C.commitment(C.CONF_SEEDS) == C.COMMITMENT_SHA256
    assert not set(C.CONF_SEEDS) & {104729, 130363, 155921, 196613, 262147}


def test_tampered_seeds_refused(C, tmp_path):
    rec = json.loads(C.SEEDS_JSON.read_text(encoding="utf-8"))
    rec["seeds"] = [215593, 276871, 348239, 385670, 915724]
    p = tmp_path / "seeds.json"
    p.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(C.ConfirmationRefused):
        C.check_seeds(p)
    rec = json.loads(C.SEEDS_JSON.read_text(encoding="utf-8"))
    rec["confirmation_fold_digest_BP"]["sha256"] = "0" * 64
    p.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(C.ConfirmationRefused):
        C.check_seeds(p)
    bad_txt = tmp_path / "sha.txt"
    bad_txt.write_text("0" * 64, encoding="utf-8")
    with pytest.raises(C.ConfirmationRefused):
        C.check_seeds(C.SEEDS_JSON, bad_txt)


def test_run_once_guard(C, tmp_path):
    real_before = C.RUN_ONCE.read_bytes() if C.RUN_ONCE.exists() else None
    p = tmp_path / "confirmation" / "RUN_ONCE.json"
    rec = {"seeds": [1, 2], "commitment_sha256": "x", "fold_digest_BP": "y"}
    C.run_once_open(p, record=rec)
    with pytest.raises(C.ConfirmationRefused, match="--resume"):
        C.run_once_open(p, record=rec)
    with pytest.raises(C.ConfirmationRefused, match="differ"):
        C.run_once_open(p, resume=True, record={**rec, "seeds": [1, 3]})
    r = C.run_once_open(p, resume=True, record=rec)
    assert len(r["resumes"]) == 1
    C.run_once_close(p)
    with pytest.raises(C.ConfirmationRefused, match="exactly once"):
        C.run_once_open(p, resume=True, record=rec)
    # the test works in tmp_path only: the real RUN_ONCE.json (present after S11) is untouched
    assert (C.RUN_ONCE.read_bytes() if C.RUN_ONCE.exists() else None) == real_before


def test_confirmation_scoring_reuses_discovery_h(C):
    world, man, sels = build_world(skill=1.0, designs=("BP", "B"), n_chem=12, seed=2)
    disc = RP._clean(scorer(world, man, sels, label="discovery").score(full=False))
    conf_world = {k: v for k, v in world.items() if k[0] == "BP"}
    st = C.score_confirmation(disc, scorer_kwargs={
        "loader": lambda lab, d, v: conf_world[(d, v)],
        "status": lambda lab, d, v: {"status": "complete" if (d, v) in conf_world else "not run"},
        "manifests": man, "selections": lambda lab, d, v: {}, "replicates": 200})
    # (h) comes from discovery: design B's points are copied, the never-run designs stay None
    for ev in st["registered"]["H1"]["evidence"]:
        dp = disc["registered"]["H1"]["evidence"][[e["Y"] for e in disc["registered"]["H1"]["evidence"]].index(ev["Y"])]
        assert ev["design_points"]["B"] == pytest.approx(dp["design_points"]["B"])
        assert ev["design_points"]["A"] is None
    status = C.confirmation_status(disc, RP._clean(st))
    assert set(status["H"]) == {"H1", "H2", "H3", "H4"} and status["label"] == R.CONFIRMATION_LABEL
    assert all(k.split("@")[0] in ("T1", "T1w", "T2", "T3") for k in status["T"])
