"""The shared-only metal embedding of section 11's last bullet (POST-HOC addendum 2 item 4): ``models.shared_only`` and its
H3 wiring (``evaluation.h3.FrozenSharedOnly``, ``shared_only_condition``, ``scripts/g19_run_h3.py``).

Synthetic rows only (the neural test corpus: real metal labels and system keys feed the static descriptor tables, every
target is generated).  No fold file, no discovery record and no measured log D is read; torch runs on 2 threads.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import h3 as H3
from gen19ct.models import interface as I
from gen19ct.models import neural as NN
from gen19ct.models import shared_only as SO

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TN = _module("test_neural", HERE / "test_neural.py")                 # the synthetic neural corpus
RH = _module("g19_run_h3", paths.G19_ROOT / "scripts" / "g19_run_h3.py")


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return TN.synthetic_frame(seed=7, rows_per_cell=4)


def _split(frame: pd.DataFrame, held: str = "Gd(III)"):
    tr = frame[frame[SG.METAL_COL] != held]
    q = frame[frame[SG.METAL_COL] == held].drop(columns=[I.TARGET_COL])
    return tr, q


# --------------------------------------------------------------------------------------------- #
# the network and the arm
# --------------------------------------------------------------------------------------------- #

def test_offsets_are_zero_after_a_synthetic_fit_and_predictions_differ_from_the_full_model(frame):
    """Addendum 2 item 4: 'a subclass of neural.FactorisedNet with the e_series and e_ox offsets held at zero' -- exactly
    zero and frozen AFTER training, while the full model's offsets move; same seed / config / epochs, different
    predictions; the module attribute neural.FactorisedNet is untouched after the fit."""
    tr, q = _split(frame)
    cfg = NN.NeuralConfig(4, 1e-3, 2)
    seed = NN.registered_model_seed(0)
    orig_cls = NN.FactorisedNet
    full = NN.FactorisedArm(cfg, n_epochs=4, model_seed=seed).fit(tr)
    shared = SO.SharedOnlyArm(cfg, n_epochs=4, model_seed=seed).fit(tr)
    assert NN.FactorisedNet is orig_cls                                    # restored: no lasting patch
    net_s, net_f = shared.result.net, full.result.net
    assert SO.is_shared_only(net_s) and not SO.is_shared_only(net_f)
    assert isinstance(net_s, SO.SharedOnlyFactorisedNet) and isinstance(net_s, NN.FactorisedNet)
    check = SO.assert_offsets_zero(net_s)
    assert all(v["zero"] and v["frozen"] and v["max_abs"] == 0.0 for v in check.values())
    assert set(check) == {"e_series", "e_ox"}
    with torch.no_grad():
        assert torch.count_nonzero(net_s.e_series.weight) == 0 and torch.count_nonzero(net_s.e_ox.weight) == 0
        assert not net_s.e_series.weight.requires_grad and not net_s.e_ox.weight.requires_grad
        # the full model trains its offsets (the same rows, seed and epochs): the difference is the offsets alone
        assert torch.count_nonzero(net_f.e_series.weight) > 0 or torch.count_nonzero(net_f.e_ox.weight) > 0
        # everything else identical in shape and count; only the frozen tables are not trainable
        assert net_s.n_parameters == net_f.n_parameters
        assert net_s.n_trainable_parameters == net_f.n_parameters - net_s.e_series.weight.numel() - net_s.e_ox.weight.numel()
        # the metal embedding is A z_m + e_element[z]: the offsets contribute nothing
        e = shared._encode(tr.head(20))
        t = e.tensors()
        assert torch.allclose(net_s.metal_embedding(t), net_s.e_shared(t["z_m"]) + net_s.e_element(t["element_id"]), atol=1e-7)
    ps = shared.predict(q)["mean_logD"].to_numpy(dtype=float)
    pf = full.predict(q)["mean_logD"].to_numpy(dtype=float)
    assert np.isfinite(ps).all() and np.isfinite(pf).all()
    assert not np.allclose(ps, pf, atol=1e-6), "the shared-only model must differ from the full model"
    # the fit record is labelled
    rec = shared.fit_record()
    assert rec["shared_only"] is True and rec["zero_offsets"] == ["e_series", "e_ox"] and "addendum 2" in rec["reading"]
    assert rec["n_epochs"] == 4 and rec["model_seed"] == seed and rec["config"] == full.fit_record()["config"]
    # deterministic: the same seed reproduces the shared-only predictions; a clone is a shared-only arm
    again = shared.clone().fit(tr).predict(q)["mean_logD"].to_numpy(dtype=float)
    assert np.allclose(ps, again)
    assert isinstance(SO.SharedOnlyArm.from_arm(full), SO.SharedOnlyArm)
    assert SO.SharedOnlyArm.from_arm(full).n_epochs == 4 and SO.SharedOnlyArm.from_arm(full).model_seed == seed


def test_patched_context_binds_by_name_restores_and_asserts_at_exit():
    dims = NN.InputDims(**TN.REAL_WIDTHS)
    cfg = NN.NeuralConfig(4, 1e-3, 0)
    orig = NN.FactorisedNet
    with SO.shared_only_training() as made:
        with NN.deterministic_torch(3):
            net = NN.FactorisedNet(dims, cfg)                                # constructed BY NAME -> the subclass
        assert SO.is_shared_only(net) and made == [net]
        assert NN.FactorisedNet is not orig and issubclass(NN.FactorisedNet, SO.SharedOnlyFactorisedNet)
        # re-entrant: an inner block leaves the binding as it is
        with SO.shared_only_training():
            assert NN.FactorisedNet is not orig
        assert NN.FactorisedNet is not orig
    assert NN.FactorisedNet is orig
    # an offset that moved inside the block is caught when the block ends
    with pytest.raises(AssertionError, match="not zero and frozen"):
        with SO.shared_only_training():
            with NN.deterministic_torch(3):
                bad = NN.FactorisedNet(dims, cfg)
            with torch.no_grad():
                bad.e_ox.weight[1, 0] = 0.5
    assert NN.FactorisedNet is orig                                          # restored even after the failure
    # not a shared-only network -> the assertion names it
    with NN.deterministic_torch(3):
        plain = NN.FactorisedNet(dims, cfg)
    with pytest.raises(AssertionError, match="not a shared-only network"):
        SO.assert_offsets_zero(plain)
    # the same seed gives the same non-offset parameters in both classes (identical RNG consumption)
    with NN.deterministic_torch(11):
        a = NN.FactorisedNet(dims, NN.NeuralConfig(4, 1e-3, 2))
    with NN.deterministic_torch(11):
        b = SO.SharedOnlyFactorisedNet(dims, NN.NeuralConfig(4, 1e-3, 2))
    assert torch.equal(a.e_shared.weight, b.e_shared.weight) and torch.equal(a.P, b.P) and torch.equal(a.Q, b.Q)
    assert torch.equal(a.head[-1].weight, b.head[-1].weight)


def test_ladder_net_subclass_has_zero_frozen_offsets_and_inherits_everything_else():
    """A deployed ladder step (LadderNet, a FactorisedNet subclass that inherits metal_embedding) is covered by the same
    mixin, bound by name in models.ladder for the fit."""
    from gen19ct.models import ladder as LAD

    dims = NN.InputDims(**TN.REAL_WIDTHS)
    cfg = LAD.LadderConfig(step="M3", emb_dim=4, weight_decay=1e-3, rank=2, experts=True)
    cls = SO.shared_only_subclass(LAD.LadderNet)
    assert cls is SO.shared_only_subclass(LAD.LadderNet) and issubclass(cls, LAD.LadderNet) and cls.__name__ == "SharedOnlyLadderNet"
    assert SO.shared_only_subclass(NN.FactorisedNet) is SO.SharedOnlyFactorisedNet
    with NN.deterministic_torch(5):
        net = cls(dims, cfg, n_experts=3, n_groups=0)
    assert SO.is_shared_only(net) and net.n_experts == 3
    SO.assert_offsets_zero(net)
    orig = LAD.LadderNet
    with SO.shared_only_ladder_training() as made:
        with NN.deterministic_torch(5):
            inner = LAD.LadderNet(dims, cfg, n_experts=3, n_groups=0)
        assert SO.is_shared_only(inner) and made == [inner]
    assert LAD.LadderNet is orig
    with pytest.raises(TypeError):
        SO.shared_only_subclass(torch.nn.Linear)


# --------------------------------------------------------------------------------------------- #
# the H3 arm: the WITH record's selections, refusal without a record
# --------------------------------------------------------------------------------------------- #

def _stub_fc(df: pd.DataFrame, hidden: pd.Index, ordinal: int = 0):
    mask = ~df.index.isin(hidden)
    corpus = SimpleNamespace(frame=df, table=SimpleNamespace(index=df.index), cv=None, systems=None)
    return SimpleNamespace(corpus=corpus, mask=mask, sc_labels=hidden, ordinal=ordinal, ctx=None,
                           job=SimpleNamespace(inner_mode="full"))


def test_frozen_shared_only_refits_at_the_with_records_selection_and_refuses_without_a_record(frame):
    hidden = frame.index[(frame[SG.METAL_COL] == "Eu(III)") & (frame[SG.SYSTEM_COL] == next(iter(TN.SYSTEMS)))]
    fc = _stub_fc(frame, hidden)
    rec = {"arm_record": {"selected": {"emb_dim": 4, "weight_decay": 1e-3, "rank": 2}, "n_epochs": 3, "model_seed": 11,
                          "inner_folds_used": [0, 1, 2]}, "selected_config": "M2_d4_wd1e-3_r2"}
    arm = H3.FrozenSharedOnly("M2")
    assert arm.transform == H3.SHARED_ONLY and not arm.is_ladder and arm.arms == ("M2",)
    with pytest.raises(ValueError, match="refused: the shared-only re-run of M2 needs the WITH record"):
        arm.point(fc, None)
    with pytest.raises(ValueError, match="refused"):
        arm.point(fc, {})
    with pytest.raises(ValueError, match="no metal embedding"):
        H3.FrozenSharedOnly("B6")
    with pytest.raises(ValueError, match="no metal embedding"):
        H3.shared_only_runner("B3i")
    out = arm.point(fc, rec)
    o = out["M2"]
    assert len(o.pred) == len(hidden) and o.model_seed == 11
    assert o.selected_config.endswith("; shared-only embedding") and o.selected_config.startswith("M2")
    fr = o.record["fit_record"]
    assert fr["shared_only"] is True and fr["n_epochs"] == 3 and fr["model_seed"] == 11 and fr["config"]["rank"] == 2
    assert o.record["shared_only"]["offset_check"]["e_series"]["zero"] and o.record["shared_only"]["offset_check"]["e_ox"]["frozen"]
    assert o.record["selected_hyperparameters"] == H3.selected_hyperparameters("M2", rec)
    assert "addendum 2" in o.record["shared_only"]["reading"]
    full = H3.FrozenNeural("M2").point(fc, rec)["M2"]
    ps, pf = o.pred["mean_logD"].to_numpy(dtype=float), full.pred["mean_logD"].to_numpy(dtype=float)
    assert np.isfinite(ps).all() and not np.allclose(ps, pf, atol=1e-6)
    # the calibration converts the cross-fit inner arms to shared-only twins (the WITH record's selections kept)
    inner = NN.FactorisedArm(NN.NeuralConfig(4, 1e-3, 2), n_epochs=2, model_seed=5)
    cal = SimpleNamespace(arms_by_fold={0: inner, 1: inner.clone()})
    arm.inner = SimpleNamespace(calibration=lambda fc, rec: (cal, {"calibration_folds": [0, 1]}))
    got, plan = arm.calibration(fc, rec["arm_record"])
    assert plan["shared_only"] is True and all(isinstance(a, SO.SharedOnlyArm) for a in got.arms_by_fold.values())
    assert got.arms_by_fold[0].n_epochs == 2 and got.arms_by_fold[0].model_seed == 5
    arm.inner = SimpleNamespace(calibration=lambda fc, rec: (None, {"status": H3.NOT_CALIBRATED_DIFFER}))
    assert arm.calibration(fc, rec["arm_record"]) == (None, {"status": H3.NOT_CALIBRATED_DIFFER})
    # a ladder step is a FrozenLadder inside the by-name patch; its training context is the ladder one
    lad = H3.FrozenSharedOnly("M5", min_expert_rows=50)
    assert lad.is_ladder and isinstance(lad.inner, H3.FrozenLadder)
    import contextlib
    assert isinstance(arm.training_context(), contextlib.nullcontext)


# --------------------------------------------------------------------------------------------- #
# the condition gate (section 11: "run if WITHOUT is better, point estimate or passed")
# --------------------------------------------------------------------------------------------- #

def _verdict(hurts: bool, point):
    return {"verdict": "hurts" if hurts else "UNDECIDED", "hurts": hurts, "v5_without_vs_with_point": point}


def test_condition_gate_reads_hurts_or_the_v5_point_estimate():
    met = H3.shared_only_condition(_verdict(True, -0.2), deployed_arm="M2")            # passed, point against: runs
    assert met["met"] and met["status"] == "run" and met["without_passed_r19_v5"] is True
    pt = H3.shared_only_condition(_verdict(False, 0.03), deployed_arm="M2")            # point estimate favours WITHOUT
    assert pt["met"] and pt["point_estimate_favours_without"] and pt["reason"].startswith("the V5 point estimate")
    no = H3.shared_only_condition(_verdict(False, -0.03), deployed_arm="M2")
    assert not no["met"] and no["status"] == H3.SHARED_ONLY_NOT_RUN == "not run (condition not met)"
    tie = H3.shared_only_condition(_verdict(False, 0.0), deployed_arm="M1")            # exactly equal is not "better"
    assert not tie["met"]
    nan = H3.shared_only_condition(_verdict(False, float("nan")), deployed_arm="M2")
    assert not nan["met"]
    none = H3.shared_only_condition(_verdict(False, None), deployed_arm="M2")
    assert not none["met"] and none["point_estimate_favours_without"] is None
    absent = H3.shared_only_condition(None, deployed_arm="M2")
    assert not absent["met"] and absent["status"].startswith("not run (condition undecided")
    for arm in ("B6", "B5", "M0", "B3i"):
        na = H3.shared_only_condition(_verdict(True, 0.5), deployed_arm=arm)
        assert not na["met"] and not na["applicable"] and na["status"].startswith("not run (not applicable")
    for arm in ("M1", "M2", "M3", "M4", "M5", "M6", "M7"):
        assert H3.shared_only_applicable(arm) and H3.shared_only_condition(_verdict(True, 0.1), deployed_arm=arm)["met"]
    assert "point estimate or passed" in met["condition"] and "v5_without_vs_with_point" in met["condition"]
    assert "hurts" in met["condition"] and "addendum 2" in H3.READINGS["shared_only_embedding"]
    assert "not run (condition not met)" in H3.READINGS["shared_only_embedding"]


def _res(point: float, *, scope: str = "PASS", equivalent: bool = False, non_inferior: bool = True):
    tost = {"verdict": "NO_DIFFERENCE" if equivalent else ("PASS" if non_inferior else "UNDECIDED"),
            "equivalent": equivalent, "non_inferior": non_inferior, "low_90": -0.01 if non_inferior else -0.4,
            "high_90": 0.02 if equivalent else 0.4}
    br = SimpleNamespace(percentile_interval=lambda level=0.95: ((0.1, 0.5) if point > 0 else (-0.5, -0.1)),
                         bca_interval=lambda level=0.95: ((0.1, 0.5) if point > 0 else (-0.5, -0.1)))
    return {"point": point, "scopes": {H3.VERDICT_SCOPE: {"verdict": scope}}, "tost": tost,
            "primary_cluster_unit": "system", "bootstraps": {"system": br}}


def test_h3_verdict_carries_the_reversed_v5_point_the_condition_reads():
    helps = {"V5": {"WITHOUT": _res(-0.3, scope="FAIL"), "ACT_PERMUTED": _res(-0.3, scope="FAIL")}}
    v = H3.h3_verdict(helps=helps, hurts={"WITHOUT": _res(0.3, scope="PASS")})
    assert v["verdict"] == "hurts" and v["hurts"] is True and v["v5_without_vs_with_point"] == 0.3
    assert H3.shared_only_condition(v, deployed_arm="M2")["met"]
    v2 = H3.h3_verdict(helps=helps, hurts={"WITHOUT": _res(0.02, scope="UNDECIDED")})
    assert v2["verdict"] != "hurts" and v2["v5_without_vs_with_point"] == 0.02
    assert H3.shared_only_condition(v2, deployed_arm="M2")["met"]                     # the point-estimate half
    v3 = H3.h3_verdict(helps=helps, hurts={"WITHOUT": _res(-0.02, scope="FAIL")})
    assert not H3.shared_only_condition(v3, deployed_arm="M2")["met"]
    assert H3.h3_verdict(helps={}, hurts={})["v5_without_vs_with_point"] is None


# --------------------------------------------------------------------------------------------- #
# jobs, plan, records and D03
# --------------------------------------------------------------------------------------------- #

def test_shared_only_job_plan_and_transform_labels(frame):
    base = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme="batched", seed=D.PRIMARY_SEED,
                     fold_seed=D.PRIMARY_SEED, writes=("M2",), stage=H3.STAGE)
    job = H3.shared_only_job(base, "M2")
    assert job.arm == "M2:SHARED_ONLY" and job.writes == ("M2:SHARED_ONLY",) and job.design_dir == base.design_dir
    assert "addendum 2 item 4" in job.purpose and job.seed == D.PRIMARY_SEED
    with pytest.raises(ValueError, match="applies to"):
        H3.shared_only_job(base, "B6")
    with pytest.raises(ValueError):                                          # SHARED_ONLY is not a discovery H3 transform
        H3.h3_job(base, "M2", H3.SHARED_ONLY)
    state = D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed")
    assert RH.shared_only_plan("B6", state) == [] and RH.shared_only_plan("B3i", state) == []
    plan = RH.shared_only_plan("M2", state)
    assert [e["design"] for e in plan] == list(H3.DESIGNS) and all(e["transform"] == "SHARED_ONLY" for e in plan)
    assert all(e["job"].arm == "M2:SHARED_ONLY" and e["with_job"].arm == "M2" for e in plan)
    assert [e["job"].design_dir for e in plan] == [H3.with_design_dir("M2", d, state) for d in H3.DESIGNS]
    m0 = RH.shared_only_plan("M5", state)
    assert all(e["model_arm"] == "M5" and e["with_job"].arm == "M5" for e in m0)
    # the re-run changes the network, never a training row
    train = frame.index[:50]
    assert H3.transformed_frame(frame, train, H3.SHARED_ONLY, seed=1) is frame


def test_run_fold_uses_the_shared_only_runner_and_labels_the_record(tmp_path, monkeypatch):
    TH = _module("test_h3", HERE / "test_h3.py")
    corpus = TH._corpus(tmp_path)
    TH._write([TH._cell_fold(corpus.frame, "Nd(III)", "S1"), TH._cell_fold(corpus.frame, "Eu(III)", "S2")], corpus.folds_dir)
    stub = TH.StubArm("M2")
    used = {"shared": 0}

    def shared_runner(arm):
        used["shared"] += 1
        return stub

    monkeypatch.setattr(H3, "shared_only_runner", shared_runner)
    monkeypatch.setattr(H3, "frozen_runner", lambda arm: (_ for _ in ()).throw(AssertionError("frozen_runner must not be used")))
    monkeypatch.setattr(RH, "with_record", lambda *a, **k: {"digest": "with-digest", "selected_config": "M2_d8", "arm_record": {}})
    base = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED,
                     writes=("M2",), stage=H3.STAGE)
    entry = {"model_arm": "M2", "transform": H3.SHARED_ONLY, "design": "V5", "with_job": base, "job": H3.shared_only_job(base, "M2")}
    out = tmp_path / "out"
    ledger = RH.run_plan([entry], corpus, out, D.PlanState(), code="c", steps=("point",), guard_fn=TH._ok_guard,
                         inner_check=TH._ok_inner)
    assert ledger["arms"]["M2:SHARED_ONLY@V5"]["fitted"] == 2 and used["shared"] == 2
    pq, js = H3.fold_paths(out, "M2", H3.SHARED_ONLY, entry["job"], "Nd(III)__S1")
    rec = json.loads(js.read_text(encoding="utf-8"))
    assert rec["transform"] == "SHARED_ONLY" and rec["shared_only"] is True and rec["arm"] == "M2:SHARED_ONLY"
    from gen19ct.evaluation import registry as REG
    assert rec["registry_stage"] == "h3" and rec["prereg_addenda_sha256"] == REG.below_footer_sha256("h3")   # registry, else the constant
    fr = pd.read_parquet(pq)
    assert (fr["arm"] == "M2:SHARED_ONLY").all() and (fr["half"] == "S").all()
    # the training mask is the WITH mask: actinide rows stay (the re-run changes the network, not the rows)
    an = D.actinide_rows(corpus.frame)
    assert all(int((mask & an).sum()) > 0 for _, mask, _ in stub.calls)
    assert str(js).replace("\\", "/").endswith("records/M2/SHARED_ONLY/V5__primary_exact/s104729/Nd(III)__S1.json")


def test_score_shared_only_runs_nothing_when_the_condition_is_not_met_and_names_absent_records():
    state = D.PlanState()
    no = RH.score_shared_only(None, "M2", state, delta5=0.1, verdict=_verdict(False, -0.1))
    assert no["status"] == "not run (condition not met)" and no["contrasts"] == [] and no["frame"].empty
    assert not no["condition"]["met"] and no["designs"] == {}
    na = RH.score_shared_only(None, "B6", state, delta5=0.1, verdict=_verdict(True, 0.5))
    assert na["status"].startswith("not run (not applicable")
    fake = SimpleNamespace(with_frame=lambda e: None, h3_frame=lambda e: None, kw=lambda a, d: {})
    met = RH.score_shared_only(fake, "M2", state, delta5=0.1, verdict=_verdict(True, 0.5))
    assert met["condition"]["met"] and met["status"] == "condition met; records incomplete" and not met["records_complete"]
    assert set(met["designs"]) == set(H3.DESIGNS) and all(v["status"].startswith("records absent") for v in met["designs"].values())


def test_d03_says_not_run_when_the_condition_is_not_met_and_prints_the_rows_when_it_ran():
    summary = {"deployed": {"arm": "M2", "basis": "b"}, "model_arms": ["M2", "B6", "B5"], "transforms": list(H3.ALL_ARMS),
               "verdicts": {}, "f4": {}, "git_head": "abc",
               "shared_only": {"condition": H3.shared_only_condition(_verdict(False, -0.05), deployed_arm="M2"),
                               "status": H3.SHARED_ONLY_NOT_RUN, "contrasts": []}}
    md = H3.d03_markdown(summary, None, None, None)
    assert "Shared-only-embedding re-run (section 11 last bullet; addendum 2 item 4" in md
    assert "**not run (condition not met)**" in md and "point estimate or passed" in md
    ran = dict(summary, shared_only={"condition": H3.shared_only_condition(_verdict(True, 0.2), deployed_arm="M2"),
                                      "status": "run",
                                      "contrasts": [{"model_arm": "M2", "contrast": "M2:SHARED_ONLY vs M2:WITH", "design": "V5",
                                                     "point": 0.04, "percentile_low": -0.01, "percentile_high": 0.09,
                                                     "bca_low": -0.01, "bca_high": 0.1, "p_two_sided": 0.2,
                                                     f"verdict_{H3.VERDICT_SCOPE}": "UNDECIDED", "tost_verdict_eps0.05": "PASS"}]})
    md2 = H3.d03_markdown(ran, None, None, None)
    assert "**run**" in md2 and "M2:SHARED_ONLY vs M2:WITH" in md2 and "positive favours the full embedding" in md2
    absent = H3.d03_markdown(dict(summary, shared_only=None), None, None, None)
    assert "no `shared_only` block" in absent
    # the runner's code digest covers the new module and the arm (a change there stales every H3 record)
    assert any(str(p).endswith("shared_only.py") for p in RH.CODE_FILES) and H3.FrozenSharedOnly in RH.CODE_OBJECTS


# --------------------------------------------------------------------------------------------- #
# task X verifier findings 1 and 5
# --------------------------------------------------------------------------------------------- #

def test_frames_do_not_cache_an_absent_record_set_and_reread_after_the_rerun(tmp_path, monkeypatch):
    """``main`` scores, fits the shared-only records when the condition holds, then scores the SAME ``Frames`` again: an
    absent record set must be re-read from disk, a present one is cached."""
    fr = object.__new__(RH.Frames)
    fr.out_root, fr.corpus, fr.state, fr.code = tmp_path, None, D.PlanState(), "c"
    fr._cache, fr.record_sets = {}, {}
    calls = {"n": 0}
    frame = pd.DataFrame({"row_id": ["r1"], "mean_logD": [0.1]})

    def read_set(root, exp, *, what, steps=H3.STEPS):
        # task X finding numbers VH-06: ``Frames.h3_frame`` asks for the steps whose columns it reads
        assert tuple(steps) == tuple(H3.STEPS)
        calls["n"] += 1
        return (None, {"status": "missing"}) if calls["n"] == 1 else (frame, {"status": "complete"})

    monkeypatch.setattr(RH, "expected_records", lambda entry, corpus, state, out_root, *, code: {})
    monkeypatch.setattr(H3, "read_record_set", read_set)
    monkeypatch.setattr(RH.Frames, "_scoring", lambda self, pred, design, what: pred)
    base = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED,
                     writes=("M2",), stage=H3.STAGE)
    job = H3.shared_only_job(base, "M2")
    entry = {"model_arm": "M2", "transform": H3.SHARED_ONLY, "design": "V5", "with_job": base, "job": job}
    key = f"{H3.SHARED_ONLY}/M2/{job.design_dir}/s{job.seed}"
    assert fr.h3_frame(entry) is None and calls["n"] == 1 and fr.record_sets[key]["status"] == "missing"
    assert ("h3", "M2", H3.SHARED_ONLY, "V5") not in fr._cache                  # absence is not cached
    assert fr.h3_frame(entry) is frame and calls["n"] == 2 and fr.record_sets[key]["status"] == "complete"
    assert fr.h3_frame(entry) is frame and calls["n"] == 2                     # a present set is cached


def test_frozen_shared_only_ladder_refuses_when_no_network_was_built_and_labels_when_one_was(monkeypatch):
    import contextlib

    lad = H3.FrozenSharedOnly("M5", min_expert_rows=50)
    RD = H3._runner_module()
    made: list = []

    @contextlib.contextmanager
    def ctx():
        yield made

    monkeypatch.setattr(lad, "training_context", ctx)
    lad.inner = SimpleNamespace(point=lambda fc, rec: {"M5": RD.ArmOutput(pred=pd.DataFrame(), selected_config="M5_cfg",
                                                                          model_seed=3, record={})})
    with pytest.raises(AssertionError, match="built no network"):
        lad.point(None, {"arm_record": {}})
    made.append(object())
    monkeypatch.setattr(SO, "assert_offsets_zero", lambda n: {"e_series": {"zero": True}, "e_ox": {"frozen": True}})
    o = lad.point(None, {"arm_record": {}})["M5"]
    assert o.record["shared_only"]["n_networks"] == 1 and o.record["shared_only"]["offset_checks"][0]["e_series"]["zero"]
    assert o.selected_config == "M5_cfg; shared-only embedding"
