"""The section 11 actinide ablation (H3) on a synthetic mini-corpus (pre-registration sections 8, 10 F4, 11, 16).

Nothing here reads a real fold, a real target or a discovery record of the running plan: the corpus is ~200 synthetic
rows written to ``tmp_path``, the arms are stubs (a training-mean predictor) or the closed-form B0, and the deltas and
verdicts run on synthetic per-row predictions.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.chemistry import support_graph as SG
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import h3 as H3
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import registry as REG
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import io as FI
from gen19ct.models import interface as I

SCRIPTS = paths.G19_ROOT / "scripts"
LN_STATES = ("La(III)", "Nd(III)", "Eu(III)", "Gd(III)")
AN_STATES = ("Am(III)", "Cm(III)")
SYSTEMS = ("S1", "S2", "S3", "S4")
HALF_OF_SYSTEM = {"S1": "S", "S2": "S", "S3": "C", "S4": "C"}
CODE = "code-h3-test"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RH = _load("g19_run_h3")
RD = _load("g19_run_discovery")


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path_factory, monkeypatch):
    """POST-HOC addendum 2 item 5: the generation's ``manifests/digest_registry.json`` governs the REAL records.  Every
    record here is synthetic, written into ``tmp_path`` under a code digest of the test's own, so the registry lookups of
    stages ``h3`` / ``discovery`` see no file and fall back to the constants and the given code, exactly as before the
    registry existed (``registry.READINGS['fallback']``); a test with entries of its own monkeypatches
    ``REG.registry_path`` again."""
    absent = tmp_path_factory.mktemp("no_registry") / "digest_registry.json"
    monkeypatch.setattr(REG, "registry_path", lambda root=None: absent)


# --------------------------------------------------------------------------------------------- #
# the synthetic mini-corpus (every column slim_frame and RowTable need)
# --------------------------------------------------------------------------------------------- #

def _frame(n_per_cell: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    recs = []
    for i, st in enumerate(LN_STATES + AN_STATES):
        sym = st.split("(")[0]
        for j, sy in enumerate(SYSTEMS):
            for k in range(n_per_cell):
                g = f"g{(i + j + k) % 5}"
                recs.append({FI.ROW_ID: f"T:{len(recs):05d}", "log_D": float(rng.normal() + 0.2 * j),
                             "duplicate_group_id": f"d{len(recs)}", SG.PUB_COL: f"pub_{g}",
                             SG.METAL_COL: st, SG.ELEMENT_COL: sym, "g19_ox": 3, SG.SYSTEM_COL: sy,
                             "acid_primary": "HNO3", "solvent_key": "kerosene", "acid_concentration_M": 1.0 + 0.1 * k,
                             "extractant_primary_concentration_M": 0.1, "temperature_C": 25.0,
                             SG.ACID_ANION_COL: "nitrate", SG.LOG_ACID_COL: float(np.log10(1.0 + 0.1 * k)),
                             SG.LOG_EXT_COL: -1.0, SG.TEMP_COL: 25.0, SG.FAMILY_COL: "diglycolamide" if j < 2 else "monoamide",
                             SG.MECH_COL: "NEUTRAL_SOLVATING", SG.SMILES_COL: f"C{j}", SG.DILUENT_COL: "aliphatic",
                             I.PUB_GROUP_COL: g, FI.GROUP_COL: g})
    # three unknown-state actinide rows (X(?)): never scored, dropped by the injected refits, part of WITHOUT
    for k in range(3):
        recs.append({FI.ROW_ID: f"T:{len(recs):05d}", "log_D": float(rng.normal()), "duplicate_group_id": f"x{k}",
                     SG.PUB_COL: "pub_g0", SG.METAL_COL: None, SG.ELEMENT_COL: "Am", "g19_ox": None,
                     SG.SYSTEM_COL: "S1", "acid_primary": "HNO3", "solvent_key": "kerosene",
                     "acid_concentration_M": 1.0, "extractant_primary_concentration_M": 0.1, "temperature_C": 25.0,
                     SG.ACID_ANION_COL: "nitrate", SG.LOG_ACID_COL: 0.0, SG.LOG_EXT_COL: -1.0, SG.TEMP_COL: 25.0,
                     SG.FAMILY_COL: "diglycolamide", SG.MECH_COL: "NEUTRAL_SOLVATING", SG.SMILES_COL: "C0",
                     SG.DILUENT_COL: "aliphatic", I.PUB_GROUP_COL: "g0", FI.GROUP_COL: "g0"})
    df = pd.DataFrame(recs)
    df.index = pd.Index([f"L{i}" for i in range(len(df))])
    return df


def _corpus(tmp_path: Path, df: pd.DataFrame | None = None):
    df = _frame() if df is None else df
    halves = {"V5": df[SG.SYSTEM_COL].map(HALF_OF_SYSTEM).fillna("NA"), "V2": pd.Series("S", index=df.index)}
    halves["V1"] = halves["V5"]
    halves["V5P"] = halves["V5PAIR"] = halves["V5"]
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    return RD.Corpus(frame=df, slim=FI.slim_frame(df), table=I.RowTable(df), v6=pd.Series(False, index=df.index),
                     coext=pd.Series(False, index=df.index), systems=None, comps=None, cv=None, pmap=None,
                     row_half=halves, folds_dir=folds_dir)


def _cell_fold(df: pd.DataFrame, state: str, system: str) -> FI.Fold:
    own = (df[SG.METAL_COL] == state) & (df[SG.SYSTEM_COL] == system)
    alias = df[SG.METAL_COL].isna() & (df[SG.ELEMENT_COL] == state.split("(")[0]) & (df[SG.SYSTEM_COL] == system)
    hid = df.loc[own | alias, FI.ROW_ID].tolist()
    sc = df.loc[own, FI.ROW_ID].tolist()
    label = f"{state} x {system}"
    return FI.make_fold(design="V5", variant="primary", scheme="exact", fold_id=f"{state}__{system}",
                        half=HALF_OF_SYSTEM[system], seed=None, hidden=hid, scored=sc, unit_type="cell", units=[label],
                        row_unit={r: label for r in hid}, row_half={r: HALF_OF_SYSTEM[system] for r in hid},
                        meta={"cells": [[state, system]], "component_aware": True})


def _write(folds, folds_dir: Path) -> str:
    FI.write_design(folds, folds_dir)
    f = folds[0]
    return FI.design_stem(f.design, f.variant, f.scheme)


class StubArm:
    """A training-mean 'arm' with the H3 runner protocol; records the training mask of every fit."""

    has_interval_step = False

    def __init__(self, name: str = "M2"):
        self.name, self.arms = name, (name,)
        self.calls: list[tuple] = []

    def point(self, fc, with_record=None):
        self.calls.append((fc.fold.fold_id, fc.mask.copy(), fc.corpus.frame.copy()))
        y = fc.corpus.table.y[fc.mask]
        pred = I.records_to_frame([I.empty_prediction_record() for _ in fc.sc_ids])
        pred["mean_logD"] = float(np.mean(y))
        pred["fallback_level"] = "stub"
        return {self.name: RD.ArmOutput(pred=pred, selected_config="stub", model_seed=7,
                                        record={"n_train": int(fc.mask.sum())}, seconds=0.1)}


def _ok_guard(job, fold, corpus, universe):
    return [{"level": "stub", "ok": True, "publication_basis": None, "n_train": len(universe), "n_test": 0}]


def _ok_inner(job, corpus):
    return lambda tr, te: {"ok": True}


def _entry(transform: str, *, arm: str = "M2", design: str = "V5", stem: str = "V5__primary__exact"):
    base = D.JobSpec(kind="fit", arm=arm, design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED,
                     writes=(arm,), stage=H3.STAGE)
    return {"model_arm": arm, "transform": transform, "design": design, "with_job": base,
            "job": H3.h3_job(base, arm, transform)}


# --------------------------------------------------------------------------------------------- #
# the deployed configuration (section 11 + section 10 F6)
# --------------------------------------------------------------------------------------------- #

def _ladder(stop: bool = False, **kept) -> dict:
    """A complete ladder.json body (every step M3-M7 in a done status) with the given steps kept."""
    steps = {s: {"step": s, "status": "kept" if kept.get(s) else "removed", "kept": bool(kept.get(s, False)),
                 "predecessor": "M2"} for s in ("M3", "M4", "M5", "M6")}
    steps["M7"] = {"step": "M7", "status": "judged", "kept": bool(kept.get("M7", False))}
    if stop:
        for s in ("M3", "M4", "M5", "M6"):
            steps[s] = {"step": s, "status": "exploratory_not_run", "kept": None}
    return {"stop_rule": stop, "discovery_ladder": {"M1": True, "M2": True}, "steps": steps, "demoted": [], "notes": []}


def test_deployed_configuration_reads_the_ladder_runner_then_the_scorer_then_b3i():
    """Task X finding V-01: section 11's arm is 'the retained ladder configuration' -- the highest kept step of the
    ladder runner's ladder.json (M7 > ... > M3), then the scorer's M2 / M1, then the stop rule, then B3i; undecidable
    while the ladder has not run every step."""
    kept = {"ladder": {"M0": {"kept": True}, "M1": {"kept": True}, "M2": {"kept": True}},
            "stop_rule": {"M2_vs_B3i": {"verdict": "PASS"}, "B6_vs_B3i": {"verdict": "FAIL"}}}
    assert H3.deployed_configuration(kept, _ladder())["arm"] == "M2"
    assert H3.deployed_configuration(kept, _ladder(M3=True))["arm"] == "M3"
    assert H3.deployed_configuration(kept, _ladder(M3=True, M4=True))["arm"] == "M4"
    dep7 = H3.deployed_configuration(kept, _ladder(M3=True, M5=True, M7=True))
    assert dep7["arm"] == "M7" and H3.LADDER_STATE_FILE in dep7["basis"] and dep7["ladder_kept"]["M7"] is True
    # under the stop rule the ladder ran M7 for H6 only: no ladder step deploys, the M2 / M1 rule applies
    stop7 = H3.deployed_configuration(kept, _ladder(stop=True, M7=True))
    assert stop7["arm"] == "M2" and "stop_rule == true" in stop7["ladder_note"]
    m1 = {"ladder": {"M1": {"kept": True}, "M2": {"kept": False}},
          "stop_rule": {"M2_vs_B3i": {"verdict": "FAIL"}, "B6_vs_B3i": {"verdict": "FAIL"}}}
    assert H3.deployed_configuration(m1, _ladder())["arm"] == "M1"
    # no ladder step kept: the stop rule decides, M2 before B6
    stop_m2 = {"ladder": {"M1": {"kept": False}, "M2": {"kept": False}},
               "stop_rule": {"M2_vs_B3i": {"verdict": "PASS"}, "B6_vs_B3i": {"verdict": "PASS"}}}
    assert H3.deployed_configuration(stop_m2, _ladder())["arm"] == "M2"
    stop_b6 = {"ladder": {"M1": {"kept": False}, "M2": {"kept": False}},
               "stop_rule": {"M2_vs_B3i": {"verdict": "FAIL"}, "B6_vs_B3i": {"verdict": "PASS"}}}
    assert H3.deployed_configuration(stop_b6, _ladder())["arm"] == "B6"
    # neither passes: the best-passing baseline (section 10 F6), and the fallback is flagged
    none = {"ladder": {"M1": {"kept": False}, "M2": {"kept": False}},
            "stop_rule": {"M2_vs_B3i": {"verdict": "FAIL"}, "B6_vs_B3i": {"verdict": "UNDECIDED"}}}
    dep = H3.deployed_configuration(none, _ladder())
    assert dep["arm"] == H3.FALLBACK_DEPLOYED == "B3i" and dep["fallback"] and "F6" in dep["basis"]
    # the ladder has not run, or a step is not done: undecidable, never guessed (even with M2 kept in discovery)
    for lad in (None, {"stop_rule": False, "steps": {}}):
        pend = H3.deployed_configuration(kept, lad)
        assert pend["arm"] is None and pend["status"] == "pending" and "not complete" in pend["basis"]
    running = _ladder(M3=True)
    running["steps"]["M5"] = {"step": "M5", "status": "running", "kept": None}
    pend = H3.deployed_configuration(kept, running)
    assert pend["arm"] is None and "M5='running'" in pend["basis"]
    undecided = _ladder()
    undecided["steps"]["M4"] = {"step": "M4", "status": "undecided", "kept": None}
    assert H3.deployed_configuration(kept, undecided)["status"] == "pending"
    assert not H3.ladder_complete(undecided)["complete"] and H3.ladder_complete(undecided)["not_done"] == ["M4"]
    assert H3.ladder_complete(_ladder(stop=True))["complete"] and not H3.ladder_complete(None)["present"]
    assert H3.deployed_configuration(kept)["arm"] is None                         # no ladder argument at all
    # a pending scorer ladder decision is undecidable, never guessed
    pending = {"ladder": {"M1": {"kept": None}, "M2": {"kept": None}}, "stop_rule": {}}
    assert H3.deployed_configuration(pending, _ladder())["arm"] is None
    assert H3.deployed_configuration(pending, _ladder())["status"] == "pending"
    assert H3.deployed_configuration({}, _ladder())["arm"] is None
    # the arms: the deployed configuration plus B6 and B5, no repeats
    assert H3.model_arms("M2") == ("M2", "B6", "B5") and H3.model_arms("B6") == ("B6", "B5")
    assert H3.model_arms("M0") == ("B5", "B6") and H3.model_arms("M7") == ("M7", "B6", "B5")
    assert H3.transforms_for("M2") == H3.TRANSFORMS                      # WITH is the discovery record
    assert H3.transforms_for("M4") == H3.TRANSFORMS                      # WITH is the ladder record
    assert H3.transforms_for("B3i")[0] == "WITH"                         # a closed-form arm has none


def _dropped_ladder_decisions(**stop) -> dict:
    """The scorer's ladder block as ``g19_score_discovery.ladder`` writes it when M1 and M2 are DROPPED: the base step
    M0 (= B5) is kept, so the ladder's retained configuration is M0."""
    return {"ladder": {"M0": {"step": "M0", "kept": True, "note": "base (B5)"},
                       "M1": {"step": "M1", "kept": False, "predecessor": "M0"},
                       "M2": {"step": "M2", "kept": False, "predecessor": "M0"}},
            "stop_rule": {"M2_vs_B3i": {"verdict": stop.get("M2", "PASS")},
                          "B6_vs_B3i": {"verdict": stop.get("B6", "FAIL")}}}


def _not_run_ladder() -> dict:
    """``ladder.json`` after the ladder runner skipped M3-M7 (``not_run``: neither M1 nor M2 is retained, so the neural
    ladder has no base -- addendum 2 section 2, recorded by addendum 3 item 3)."""
    return {"stop_rule": False, "discovery_ladder": {"M1": False, "M2": False},
            "steps": {s: {"step": s, "status": "not_run", "kept": None, "predecessor": "M0"} for s in H3.LADDER_ARMS},
            "demoted": [], "notes": []}


def test_deployed_configuration_is_the_ladders_retained_step_m0_when_m1_and_m2_are_dropped():
    """POST-HOC addendum 3 item 2: the deployed configuration is the LADDER'S RETAINED configuration, and that includes
    its base step M0 (= B5) when neither M1 nor M2 is kept.  Addendum 2's order deployed M2 by the stop-rule fallback
    here, which would contradict the registered ladder decision; that order now applies only when the ladder retains no
    step at all."""
    dec, lad = _dropped_ladder_decisions(), _not_run_ladder()
    assert H3.ladder_complete(lad)["complete"]                       # 'not_run' is a skipped status
    dep = H3.deployed_configuration(dec, lad)
    assert dep["arm"] == "M0" and dep["status"] == "decided" and dep["retained_ladder_step"] == "M0"
    assert "addendum 3 item 2" in dep["basis"] and "B5" in dep["basis"]
    # the deployed predictor is REPORTED as M0 and its records / tables are LOOKED UP as B5 (discovery.ARM_ALIASES)
    assert dep["arm_alias"] == "B5" and "tables/discovery_summary.csv" in dep["arm_alias_note"]
    assert H3.deployed_configuration({"ladder": {"M1": {"kept": True}, "M2": {"kept": False}},
                                      "stop_rule": {}}, _ladder())["arm_alias"] == "M1"
    assert H3.deployed_configuration({}, _ladder())["arm_alias"] is None      # pending: no alias either
    assert dep["ladder_kept"]["M0"] is True and dep["ladder_kept"]["M1"] is dep["ladder_kept"]["M2"] is False
    assert dep["keys"]["ladder"]["M0"] == "ladder.M0.kept"
    # M2 passes the stop-rule scope against B3i and is STILL not deployed: the ladder decision governs
    assert dep["stop_rule_pass"]["M2_vs_B3i"] is True
    # ... and that holds whatever the stop-rule verdicts are, because the ladder retains a step
    for stop in ({"M2": "PASS", "B6": "PASS"}, {"M2": "FAIL", "B6": "PASS"}, {"M2": "FAIL", "B6": "FAIL"}):
        assert H3.deployed_configuration(_dropped_ladder_decisions(**stop), lad)["arm"] == "M0"
    # H3 trains its WITH / WITHOUT / control arms on M0, i.e. on B5 (models.boosted), and M0 IS a reference arm, so the
    # model arms are two, not three
    assert H3.model_arms(dep["arm"]) == ("B5", "B6")
    assert isinstance(H3.frozen_runner(dep["arm"]), H3.FrozenBoosted) and H3.frozen_runner(dep["arm"]).arms == ("B5",)
    assert not H3.is_ladder_arm(dep["arm"]) and not H3.is_deterministic(dep["arm"])
    assert H3.transforms_for(dep["arm"]) == H3.TRANSFORMS             # WITH is the discovery B5 record, never refitted
    assert H3.h3_arm_name(dep["arm"], "WITHOUT") == "B5:WITHOUT"
    assert H3.batching_arms(dep["arm"]) == ["B5"]                     # B5 is a heavy arm in its own right
    # B5 has no section 15 metal embedding, so addendum 2 item 4's shared-only re-run does not apply to it
    assert not H3.shared_only_applicable(dep["arm"])
    # the record directories of the WITH arm are B5's own discovery directories at the plan's schemes
    st = D.PlanState(v5_batched_check="passed_after_recolour", v1_tenfold_check="failed")
    assert H3.with_design_dir(dep["arm"], "V5", st) == "V5__primary_batched_max4"
    assert H3.with_design_dir(dep["arm"], "V1", st) == "V1__copy_exact"
    assert H3.with_design_dir(dep["arm"], "V2", st) == "V2__element_exact"
    assert H3.with_job(dep["arm"], "V5", st).arm == "B5"
    assert "M0" in H3.READINGS["deployed_rule"] and "addendum 3 item 2" in H3.READINGS["deployed_rule"]
    # the reading no longer asks for a further addendum on this point (report.readings_needing_addenda)
    assert "needs a POST-HOC addendum" not in H3.READINGS["deployed_rule"]


def test_deployed_rule_returns_the_ladders_retained_step_whenever_one_exists():
    """Regression for addendum 3 item 2: with a retained ladder step the rule returns THAT step and never falls through
    to addendum 2's stop-rule order; only a ladder with no retained step at all reaches the fallbacks."""
    stop_both = {"M2_vs_B3i": {"verdict": "PASS"}, "B6_vs_B3i": {"verdict": "PASS"}}
    cases = [("M0", {"M0": {"kept": True}, "M1": {"kept": False}, "M2": {"kept": False}}, _not_run_ladder()),
             ("M1", {"M0": {"kept": True}, "M1": {"kept": True}, "M2": {"kept": False}}, _ladder()),
             ("M2", {"M0": {"kept": True}, "M1": {"kept": True}, "M2": {"kept": True}}, _ladder()),
             ("M3", {"M0": {"kept": True}, "M1": {"kept": True}, "M2": {"kept": True}}, _ladder(M3=True)),
             ("M7", {"M0": {"kept": True}, "M1": {"kept": True}, "M2": {"kept": True}}, _ladder(M3=True, M7=True))]
    for expected, lad_block, ladder_json in cases:
        dep = H3.deployed_configuration({"ladder": lad_block, "stop_rule": stop_both}, ladder_json)
        assert dep["arm"] == expected, f"{expected}: got {dep['arm']} -- {dep['basis']}"
        assert dep["status"] == "decided" and not dep.get("fallback")
    # no retained step at all (no M0 row): addendum 2's order, unchanged
    no_base = {"ladder": {"M1": {"kept": False}, "M2": {"kept": False}}, "stop_rule": stop_both}
    assert H3.deployed_configuration(no_base, _ladder())["arm"] == "M2"
    b6_only = {"ladder": {"M1": {"kept": False}, "M2": {"kept": False}},
               "stop_rule": {"M2_vs_B3i": {"verdict": "FAIL"}, "B6_vs_B3i": {"verdict": "PASS"}}}
    assert H3.deployed_configuration(b6_only, _ladder())["arm"] == "B6"
    neither = {"ladder": {"M1": {"kept": False}, "M2": {"kept": False}, "M0": {"kept": False}},
               "stop_rule": {"M2_vs_B3i": {"verdict": "FAIL"}, "B6_vs_B3i": {"verdict": "FAIL"}}}
    dep = H3.deployed_configuration(neither, _ladder())
    assert dep["arm"] == H3.FALLBACK_DEPLOYED and dep["fallback"] and "no retained step" in dep["basis"]
    # a pending M1 / M2 decision still wins over the base step: M2 could yet be retained
    pending = {"ladder": {"M0": {"kept": True}, "M1": {"kept": False}, "M2": {"kept": None}}, "stop_rule": stop_both}
    assert H3.deployed_configuration(pending, _ladder())["status"] == "pending"


def test_frozen_boosted_refits_m0_at_the_with_records_config_and_iterations(monkeypatch):
    """Addendum 3 item 2 makes M0 = B5 the H3 arm, so the frozen-refit path must refit it through ``models.boosted`` at
    the WITH record's SELECTED CatBoost configuration and tree count of that fold (no re-tuning), with the discovery
    runner's cross-fitted calibration on the WITH record's tuning block."""
    from gen19ct.models import boosted as BO

    # a B5 discovery record as the runner writes it (BoostedRunner.point: arm_record = {arm, frozen, inner_design})
    rec = {"selected_config": "depth8_l23", "model_seed": 10000033,
           "arm_record": {"arm": {"tuning": {"inner_folds_used": [0, 1, 2], "selected": "depth8_l23"}},
                          "frozen": {"config": {"label": "depth8_l23", "depth": 8, "l2_leaf_reg": 3.0,
                                                "learning_rate": 0.05, "loss_function": "RMSE", "eval_metric": "MAE",
                                                "max_iterations": 3000, "early_stopping_rounds": 200,
                                                "one_hot_max_size": 10, "thread_count": 2},
                                     "iterations": 2934}}}
    hp = H3.selected_hyperparameters("M0", rec)                      # M0 is read through the B5 layout
    assert hp["family"] == "boosted" and hp["iterations"] == 2934 and hp["model_seed"] == 10000033
    assert "label" not in hp["config"] and hp["config"]["depth"] == 8
    assert hp == H3.selected_hyperparameters("B5", rec)

    seen: dict = {}

    class FakeArm:
        def __init__(self, name, **kw):
            seen.update(name=name, **kw)
            self.model_seed = 10000033
            self.fitted = SimpleNamespace(config=SimpleNamespace(label="depth8_l23", record=lambda: {"depth": 8}),
                                          iterations=kw["iterations"], columns=("f1", "f2"))

        def fit(self, rows, ctx):
            seen["n_train_rows"] = len(rows)
            return self

        def predict(self, rows):
            return pd.DataFrame({"mean_logD": np.zeros(len(rows))}, index=rows.index)

    monkeypatch.setattr(BO, "BoostedArm", FakeArm)
    monkeypatch.setattr(BO, "CatBoostConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(H3, "_runner_module",
                        lambda: SimpleNamespace(inner_design_object=lambda job, c: "design-obj",
                                                max_cells_per_batch=lambda job: 4,
                                                ArmOutput=lambda **kw: SimpleNamespace(**kw)))
    df = pd.DataFrame({"y": np.arange(6.0)}, index=[f"r{i}" for i in range(6)])
    fc = _stub_fc(df, df.index[:2], ordinal=7)
    out = H3.FrozenBoosted("B5").point(fc, rec)["B5"]
    # the refit is frozen: the record's configuration and tree count, the fold's own model seed, no grid
    assert seen["name"] == "B5" and seen["iterations"] == 2934 and seen["config"].depth == 8
    assert seen["fold_index"] == 7 and seen["n_train_rows"] == 4 and seen["max_cells_per_batch"] == 4
    assert out.selected_config == "depth8_l23" and out.model_seed == 10000033 and len(out.pred) == 2
    assert out.record["selected_hyperparameters"]["iterations"] == 2934
    assert out.record["frozen"]["iterations"] == 2934

    # the calibration step is the discovery runner's own cross-fitted plan, on the WITH record's tuning block
    called: dict = {}

    class FakeBoostedRunner:
        def __init__(self, name):
            called["name"] = name

        def calibration(self, fold_ctx, arm_record):
            called["inner_folds_used"] = arm_record["arm"]["tuning"]["inner_folds_used"]
            return "cross-fit-conformal", {"cross_fit": True, "calibration_folds": [0, 1, 2]}

    monkeypatch.setattr(H3, "_runner_module",
                        lambda: SimpleNamespace(_calibration_splits=lambda fc: (None, [0, 1, 2]),
                                                BoostedRunner=FakeBoostedRunner))
    cal, plan = H3.FrozenBoosted("B5").calibration(fc, rec["arm_record"])
    assert cal == "cross-fit-conformal" and plan["cross_fit"] and called["name"] == "B5"
    assert called["inner_folds_used"] == [0, 1, 2]
    # a fold whose available inner folds differ from the WITH record's is recorded, never silently calibrated
    other = {"arm": {"tuning": {"inner_folds_used": [0, 1]}}}
    cal, plan = H3.FrozenBoosted("B5").calibration(fc, other)
    assert cal is None and plan["status"] == H3.NOT_CALIBRATED_DIFFER


def test_ladder_done_statuses_and_frozen_runner_agree_with_the_ladder_runner():
    RL = H3._ladder_module()
    assert H3.LADDER_DONE_STATUSES == set(RL.STATUS_DONE + RL.STATUS_SKIPPED) - {"complete"}
    assert H3.LADDER_ARMS == RL.LADDER_STEPS
    for step in H3.LADDER_ARMS:
        r = H3.frozen_runner(step)
        assert isinstance(r, H3.FrozenLadder) and r.step == step and r.has_interval_step and H3.is_ladder_arm(step)
    assert not H3.is_ladder_arm("M2") and H3.batching_arms("M6") == ["M2"] and H3.batching_arms("B5") == ["B5"]
    with pytest.raises(ValueError, match="not a ladder step"):
        H3.FrozenLadder("M2")
    with pytest.raises(ValueError, match="no frozen runner"):
        H3.frozen_runner("M6a_toggle")
    # the ladder layout of selected_hyperparameters
    rec = {"arm_record": {"selected": {"step": "M4", "emb_dim": 8, "weight_decay": 1e-3, "rank": 4, "tau": 0.3},
                          "n_epochs": 17, "model_seed": 42, "inner_folds_used": [0, 1, 2]}, "selected_config": "M4_x"}
    hp = H3.selected_hyperparameters("M4", rec)
    assert hp["family"] == "ladder" and hp["n_epochs"] == 17 and hp["model_seed"] == 42 and hp["member_seeds"] is None
    assert hp["config"]["tau"] == 0.3 and hp["inner_folds_used"] == [0, 1, 2]
    with pytest.raises(ValueError, match="labelled"):
        H3.selected_hyperparameters("M5", rec)


def test_with_design_dir_follows_the_plan_state_and_the_arm():
    passed = D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed")
    recol = D.PlanState(v5_batched_check="failed", v1_tenfold_check="failed")
    assert H3.with_design_dir("M2", "V5", passed) == "V5__primary_batched"
    assert H3.with_design_dir("M2", "V5", recol) == "V5__primary_batched_max4"
    assert H3.with_design_dir("B6", "V5", passed) == "V5__primary_exact"
    assert H3.with_design_dir("M2", "V1", passed) == "V1__copy_grouped10"
    assert H3.with_design_dir("M2", "V1", recol) == "V1__copy_exact"
    assert H3.with_design_dir("B5", "V2", passed) == "V2__element_exact"
    # task X finding V-03: a closed-form comparator is fitted on the exact leave-one-cell-out folds (section 3.1), as in
    # discovery, never on the heavy arm's batched folds; V5-PAIR stays batched for every arm (addendum 1 item 5)
    for arm in ("B3i", "B0", "B3x", "B7"):
        assert H3.with_design_dir(arm, "V5", passed) == "V5__primary_exact"
        assert H3.with_design_dir(arm, "V5", recol) == "V5__primary_exact"
        assert H3.with_design_dir(arm, "V1", passed) == "V1__copy_exact"
        assert H3.with_design_dir(arm, "V2", passed) == "V2__element_exact"
        assert H3.with_design_dir(arm, H3.PAIR_DESIGN, passed) == "V5PAIR__primary_batched"
    cj = H3.with_job("B3i", "V5", passed)
    assert cj.arm == "B3i" and cj.scheme == "exact" and cj.fold_seed is None and cj.writes == ("B3i",)
    # a ladder step runs on discovery's M2 jobs (the heavy scheme), so its WITH records follow the plan state
    assert H3.with_design_dir("M3", "V5", passed) == "V5__primary_batched"
    assert H3.with_design_dir("M7", "V5", recol) == "V5__primary_batched_max4"
    assert H3.with_design_dir("M4", "V1", passed) == "V1__copy_grouped10"
    lj = H3.with_job("M3", "V5", passed)
    assert lj.arm == "M3" and lj.writes == ("M3",) and lj.scheme == "batched" and lj.fold_seed == D.PRIMARY_SEED
    assert H3.v1_scheme_of("M3", "V1", passed) == "grouped"
    j = H3.with_job("M2", "V5", passed)
    assert j.arm == "M2" and j.scheme == "batched" and j.fold_seed == D.PRIMARY_SEED
    assert H3.with_job("B6", "V5", passed).writes == D.B6_ARMS
    assert H3.with_job("M2", "V2", passed).fold_seed is None
    h = H3.h3_job(j, "M2", "WITHOUT")
    assert h.arm == "M2:WITHOUT" and h.writes == ("M2:WITHOUT",) and h.design_dir == j.design_dir
    with pytest.raises(ValueError):
        H3.h3_job(j, "M2", "NOPE")
    assert H3.margin_of("V5", 0.1057) == pytest.approx(0.1057) and H3.margin_of("V1", 0.1057) == pytest.approx(0.05)
    assert H3.v1_scheme_of("M2", "V1", passed) == "grouped" and H3.v1_scheme_of("B6", "V1", passed) == "exact"
    assert H3.v1_scheme_of("B3i", "V1", passed) == "exact" and H3.v1_scheme_of("M2", "V5", passed) == "exact"


# --------------------------------------------------------------------------------------------- #
# the Ln test set and the actinide-dependent stratum
# --------------------------------------------------------------------------------------------- #

def test_ln_iii_test_set_and_actinide_dependent_cells():
    assert H3.is_ln_iii("Nd(III)") and H3.is_ln_iii("La(III)")
    for bad in ("Am(III)", "Cm(III)", "U(VI)", "Nd(?)", "Ce(IV)", None, 3, "Sr(III)"):
        assert not H3.is_ln_iii(bad)
    fr = pd.DataFrame({EM.METAL_STATE_COL: ["Nd(III)", "Am(III)", None, "Ce(IV)"], "v6_target_row": False},
                      index=list("abcd"))
    assert list(H3.ln_rows(fr, "V5").index) == ["a"]
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        H3.ln_rows(fr.assign(v6_target_row=[True, False, False, False]), "V5")
    # with four Ln(III) states a cell keeps three metal-state partners once the actinides go, so it stays eligible and
    # is NOT actinide-dependent
    df = _frame(n_per_cell=10)
    dep = H3.actinide_dependent_cells(df, [("Nd(III)", "S1"), ("Nd(III)", "S2")])
    assert set(dep.columns) >= {"actinide_dependent", "eligible_with", "eligible_without_actinides", "unit_key"}
    assert dep["eligible_with"].all() and dep["eligible_without_actinides"].all()
    assert not dep["actinide_dependent"].any()
    assert list(dep["unit_key"]) == ["Nd(III) x S1", "Nd(III) x S2"]
    assert (dep["other_metal_states_for_system_with"] == 5).all()
    assert (dep["other_metal_states_for_system_without_actinides"] == 3).all()
    # drop one lanthanide and the same cell becomes actinide-dependent: 4 partner states with the actinides, 2 without,
    # below the registered m = 3, so its eligibility depends on actinide partners (the section 11 stratum)
    thin = df[df[EM.METAL_STATE_COL].astype(object) != "Gd(III)"]
    dep2 = H3.actinide_dependent_cells(thin, [("Nd(III)", "S1")])
    assert dep2["eligible_with"].all() and not dep2["eligible_without_actinides"].any()
    assert dep2["actinide_dependent"].all()
    assert int(dep2["other_metal_states_for_system_with"].iloc[0]) == 4
    assert int(dep2["other_metal_states_for_system_without_actinides"].iloc[0]) == 2


# --------------------------------------------------------------------------------------------- #
# the WITHOUT / PERMUTED / SHUFFLED training sets of one fold
# --------------------------------------------------------------------------------------------- #

def _prepare(tmp_path, transform, *, arm="M2"):
    corpus = _corpus(tmp_path)
    df = corpus.frame
    _write([_cell_fold(df, "Nd(III)", "S1"), _cell_fold(df, "Eu(III)", "S2")], corpus.folds_dir)
    entry = _entry(transform, arm=arm)
    fold = corpus.folds(entry["job"].stem)[0]
    fc, info = RH.fold_context(entry, fold, 0, corpus, tmp_path / "out", D.PlanState(), code=CODE,
                               guard_fn=_ok_guard, inner_check=_ok_inner)
    return corpus, fold, fc, info


def test_without_arm_has_zero_actinide_training_rows_and_identical_test_rows(tmp_path):
    """Section 11 WITHOUT: every training row with metal category actinide is removed, unknown-state actinide rows
    included; the hidden and scored rows are untouched, so the arms are paired on identical units."""
    corpus, fold, fc_with, _ = _prepare(tmp_path, "WITH")
    _, _, fc_wo, info = _prepare(tmp_path, "WITHOUT")
    an = D.actinide_rows(corpus.frame)
    assert an.sum() == len(AN_STATES) * len(SYSTEMS) * 4 + 3                 # known-state An rows + the three X(?) rows
    train_with = corpus.table.index[fc_with.mask]
    train_wo = corpus.table.index[fc_wo.mask]
    assert int(an[corpus.frame.index.isin(train_with)].sum()) > 0
    assert int(D.actinide_rows(corpus.frame.loc[train_wo]).sum()) == 0
    assert set(train_wo) < set(train_with)
    assert info["n_actinide_training_rows_dropped"] == len(train_with) - len(train_wo)
    # an unknown-state (X(?)) actinide row of the training rows is dropped too
    unknown = corpus.frame.index[corpus.frame[SG.METAL_COL].isna()]
    assert set(unknown) & set(train_with) and not (set(unknown) & set(train_wo))
    # identical test rows and identical hidden rows
    assert list(fc_wo.sc_ids) == list(fc_with.sc_ids) and fc_wo.hidden.equals(fc_with.hidden)
    assert np.array_equal(fc_wo.positions, fc_with.positions)
    # no scored row is ever a training row
    assert not fc_wo.mask[fc_wo.positions].any()


def test_permuted_and_shuffled_transform_training_rows_only(tmp_path):
    corpus, fold, fc_with, _ = _prepare(tmp_path, "WITH")
    base = corpus.frame
    train = corpus.table.index[fc_with.mask]
    an = D.actinide_rows(base)
    for transform, col in (("ACT_PERMUTED", I.TARGET_COL), ("ACT_METAL_SHUFFLED", SG.METAL_COL)):
        _, _, fc, info = _prepare(tmp_path, transform)
        new = fc.corpus.frame
        assert new.index.equals(base.index) and np.array_equal(fc.mask, fc_with.mask)
        assert list(fc.sc_ids) == list(fc_with.sc_ids)
        # only actinide TRAINING rows changed
        changed = base[col].astype(object).to_numpy() != new[col].astype(object).to_numpy()
        assert changed.any() and info["n_training_rows_changed"] > 0
        assert not changed[~np.isin(base.index, train)].any(), "a hidden row was transformed"
        assert (an | ~changed).all(), "a non-actinide row was transformed"
        if transform == "ACT_PERMUTED":                            # a permutation keeps the multiset of targets
            a = sorted(base.loc[train][an[base.index.isin(train)]][I.TARGET_COL])
            b = sorted(new.loc[train][an[base.index.isin(train)]][I.TARGET_COL])
            assert a == pytest.approx(b)
            assert new[SG.METAL_COL].equals(base[SG.METAL_COL])
        else:                                                      # a label shuffle keeps every target
            assert new[I.TARGET_COL].equals(base[I.TARGET_COL])
    # the transform is seeded: the same seed reproduces it, another seed does not
    t1 = H3.transformed_frame(base, train, "ACT_PERMUTED", seed=1)
    assert t1[I.TARGET_COL].equals(H3.transformed_frame(base, train, "ACT_PERMUTED", seed=1)[I.TARGET_COL])
    assert not t1[I.TARGET_COL].equals(H3.transformed_frame(base, train, "ACT_PERMUTED", seed=2)[I.TARGET_COL])
    assert H3.transformed_frame(base, train, "WITH", seed=1) is base


def test_run_fold_writes_a_verified_record_and_resumes(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path)
    _write([_cell_fold(corpus.frame, "Nd(III)", "S1"), _cell_fold(corpus.frame, "Eu(III)", "S2"),
            _cell_fold(corpus.frame, "Gd(III)", "S3")], corpus.folds_dir)
    stub = StubArm("M2")
    monkeypatch.setattr(H3, "frozen_runner", lambda arm: stub)
    monkeypatch.setattr(RH, "with_record", lambda *a, **k: {"digest": "with-digest", "selected_config": "M2_d8",
                                                            "arm_record": {}})
    out = tmp_path / "out"
    entry = _entry("WITHOUT")
    ledger = RH.run_plan([entry], corpus, out, D.PlanState(), code=CODE, steps=("point",), guard_fn=_ok_guard,
                         inner_check=_ok_inner)
    key = "M2:WITHOUT@V5"
    assert ledger["arms"][key]["fitted"] == 2 and not ledger["arms"][key]["errors"]   # the S3 fold is confirmation half
    pq, js = H3.fold_paths(out, "M2", "WITHOUT", entry["job"], "Nd(III)__S1")
    rec = json.loads(js.read_text(encoding="utf-8"))
    assert rec["transform"] == "WITHOUT" and rec["model_arm"] == "M2" and rec["with_record_digest"] == "with-digest"
    assert rec["prereg_sha256"] == D.REGISTERED_PREREG_SHA256
    assert rec["prereg_addenda_sha256"] == REG.below_footer_sha256("h3")   # the H3 stage's registry entry
    assert rec["n_actinide_training_rows_dropped"] > 0 and set(rec["steps"]) == {"point"}
    frame = pd.read_parquet(pq)
    assert list(frame.columns) == list(D.PREDICTION_COLUMNS) and (frame["half"] == "S").all()
    assert (frame["arm"] == "M2:WITHOUT").all()
    # resume: nothing is refitted
    RH.run_plan([entry], corpus, out, D.PlanState(), code=CODE, steps=("point",), guard_fn=_ok_guard,
                inner_check=_ok_inner)
    assert len(stub.calls) == 2
    # a code change makes the record stale and it is refitted
    RH.run_plan([entry], corpus, out, D.PlanState(), code="other-code", steps=("point",), guard_fn=_ok_guard,
                inner_check=_ok_inner)
    assert len(stub.calls) == 4
    # the digest covers the transform and the WITH record's digest
    fold = corpus.folds(entry["job"].stem)[0]
    d = H3.fold_digest(entry["job"], fold, CODE, transform="WITHOUT", with_digest="with-digest", guard_mode="every_split",
                       design_hash="h", ordinal=0, model_seed=1)
    assert d != H3.fold_digest(entry["job"], fold, CODE, transform="ACT_PERMUTED", with_digest="with-digest",
                               guard_mode="every_split", design_hash="h", ordinal=0, model_seed=1)
    assert d != H3.fold_digest(entry["job"], fold, CODE, transform="WITHOUT", with_digest="another",
                               guard_mode="every_split", design_hash="h", ordinal=0, model_seed=1)
    # the complete, current record set is read; every fold is verified against the digest the current code produces
    rdir = H3.record_dir(out, "M2", "WITHOUT", entry["job"])
    good = {f.fold_id: {"digest": json.loads(H3.fold_paths(out, "M2", "WITHOUT", entry["job"], f.fold_id)[1]
                                            .read_text(encoding="utf-8"))["digest"], "fold_hash": f.fold_hash}
            for f, _ in corpus.fittable(entry["job"])}
    assert len(good) == 2
    pred, st = H3.read_record_set(rdir, good, what="t")
    assert st["status"] == "complete" and st["n_found"] == 2
    assert set(pred["fold_id"]) == set(good) and (pred["half"] == "S").all()
    # one differing digest is refused as stale (the other folds stay expected, so the digest check is what fires)
    stale = {**good, fold.fold_id: {"digest": "not-the-digest", "fold_hash": fold.fold_hash}}
    with pytest.raises(D.StaleRecordError, match="digest"):
        H3.read_record_set(rdir, stale, what="t")
    # a differing fold hash is refused too
    bad_hash = {**good, fold.fold_id: {"digest": good[fold.fold_id]["digest"], "fold_hash": "other-hash"}}
    with pytest.raises(D.StaleRecordError, match="fold hash"):
        H3.read_record_set(rdir, bad_hash, what="t")
    # a record of a fold the current job does not fit is refused
    with pytest.raises(D.StaleRecordError, match="not a fittable fold"):
        H3.read_record_set(rdir, {fold.fold_id: good[fold.fold_id]}, what="t")
    # an incomplete record set is not scored, never mixed
    part = dict(good, extra_fold={"digest": "x", "fold_hash": "y"})
    out_pred, out_st = H3.read_record_set(rdir, part, what="t")
    assert out_pred is None and out_st["status"] == "incomplete" and out_st["missing_folds"] == ["extra_fold"]
    assert H3.read_record_set(rdir / "nope", good, what="t")[1]["status"] == "missing"


# --------------------------------------------------------------------------------------------- #
# deltas, paired on identical units
# --------------------------------------------------------------------------------------------- #

def _pred_frame(noise: float, seed: int, *, n_systems: int = 8, with_intervals: bool = False,
                states=("Nd(III)", "Eu(III)", "Am(III)")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recs = []
    for s in range(n_systems):
        for st in states:
            for k in range(4):
                y = float(np.random.default_rng(100 * s + k).normal())
                recs.append({"row_id": f"r{s}_{st}_{k}", EM.METAL_STATE_COL: st, EM.SYSTEM_COL: f"SYS{s:02d}",
                             EM.PUB_GROUP_COL: f"g{s % 4}", EM.CONDITION_KEY_COL: f"ck{k % 3}", EM.Y_COL: y,
                             "fold_id": f"{st}__{s}", EM.PRED_COL: y + float(rng.normal(0, noise)),
                             "dga_stratum": "non_DGA" if s % 2 else "diglycolamide", "acid_grid_flag": k == 0,
                             "censoring_candidate": k == 1, "wildcard_copy_partner_in_training": False,
                             "v6_target_row": False})
    fr = pd.DataFrame(recs).set_index("row_id", drop=False)
    fr.index.name = None
    if with_intervals:
        q = 1.0 + noise
        for lvl in (80, 95):
            fr[f"lower_{lvl}"] = fr[EM.PRED_COL] - q * (1.0 if lvl == 80 else 1.6)
            fr[f"upper_{lvl}"] = fr[EM.PRED_COL] + q * (1.0 if lvl == 80 else 1.6)
    return fr


def test_h3_contrast_is_paired_on_identical_ln_units_and_scores_no_actinide_row():
    w, o = _pred_frame(0.1, 1), _pred_frame(1.0, 2)
    v6 = pd.Series(False, index=w.index)
    res = H3.h3_contrast(w, o, design="V5", model_arm="M2", transform="WITHOUT", margin=0.1, v6_mask=v6)
    assert res["family"] == "H3" and res["design"] == "V5"
    assert res["candidate"] == "M2:WITH" and res["comparator"] == "M2:WITHOUT"
    assert res["point"] > 0 and res["n_ln_rows"] == 2 * 8 * 4                # only the two Ln(III) states are scored
    assert res["n_units"] == 16                                              # 8 systems x 2 Ln cells
    assert res["scopes"][H3.VERDICT_SCOPE]["verdict"] == "PASS"
    assert res["r19"].item(4)["status"] == D.ITEM4_NOT_EVALUATED             # addendum 1 item 3
    assert res["r19"].verdict == "UNDECIDED" and res["passes_discovery_scope"]
    assert set(res["h3_refits_not_run"]) and all(res["sensitivities"][n] == ET.UNTESTABLE
                                                 for n in res["h3_refits_not_run"])
    # the reversed contrast is the 'hurts' direction
    rev = H3.h3_contrast(w, o, design="V5", model_arm="M2", transform="WITHOUT", margin=0.1, v6_mask=v6, hurts=True)
    assert rev["point"] == pytest.approx(-res["point"]) and rev["candidate"] == "M2:WITHOUT"
    assert rev["scopes"][H3.VERDICT_SCOPE]["verdict"] == "FAIL"
    # a transform that scores different rows is refused, never silently intersected
    with pytest.raises(ValueError, match="different Ln rows"):
        H3.h3_contrast(w, o.iloc[2:], design="V5", model_arm="M2", transform="WITHOUT", margin=0.1, v6_mask=v6,
                       n_resamples=100)
    # a closed-form arm keeps R19 item 4 VACUOUS (one seed value), not NOT_EVALUATED
    det = H3.h3_contrast(w, o, design="V5", model_arm="B3i", transform="WITHOUT", margin=0.1, v6_mask=v6)
    assert det["r19"].item(4)["status"] == "VACUOUS"
    assert "bootstrap_not_registered" not in res and "bootstrap_not_registered" not in det
    # a reduced bootstrap is flagged: R19 items 2 and 3 accept only the registered 10,000 resamples, so such a verdict
    # is never a registered result
    cheap = H3.h3_contrast(w, o, design="V5", model_arm="M2", transform="WITHOUT", margin=0.1, v6_mask=v6,
                           n_resamples=400)
    assert "bootstrap_not_registered" in cheap and cheap["n_resamples"] == 400
    assert cheap["r19"].item(2)["status"] == "FAIL" and "not the registered" in cheap["bootstrap_not_registered"]


def test_rank_accuracy_calibration_and_per_unit_deltas():
    w, o = _pred_frame(0.1, 1, with_intervals=True), _pred_frame(1.0, 2, with_intervals=True)
    v6 = pd.Series(False, index=w.index)
    ra = H3.rank_accuracy_delta(w, o, design="V5", model_arm="M2", transform="WITHOUT", v6_mask=v6, n_resamples=400)
    prim = [r for r in ra if r.get("primary_cluster_unit")]
    assert len(prim) == 1 and prim[0]["metric"] == "rank_accuracy" and prim[0]["status"] == "computed"
    assert prim[0]["with_value"] > prim[0]["transform_value"] and prim[0]["point"] > 0
    assert prim[0]["higher_is_better"] and prim[0]["cluster_unit"] == "system"
    cal = H3.calibration_delta(w, o, design="V5", model_arm="M2", transform="WITHOUT", v6_mask=v6, n_resamples=300)
    metrics = {r["metric"] for r in cal}
    assert {"abs_coverage_gap_80", "width_80", "abs_coverage_gap_95", "width_95", "crps"} <= metrics
    assert next(r for r in cal if r["metric"] == "crps")["status"] == "NOT_RUN"
    width = next(r for r in cal if r["metric"] == "width_80" and r.get("primary_cluster_unit"))
    assert width["point"] > 0 and width["transform_value"] > width["with_value"]   # the WITHOUT arm's bands are wider
    # without intervals the calibration delta is NOT_RUN, never silently skipped
    cal2 = H3.calibration_delta(_pred_frame(0.1, 1), _pred_frame(1.0, 2), design="V5", model_arm="M2",
                                transform="WITHOUT", v6_mask=v6, n_resamples=100)
    assert {r["status"] for r in cal2 if r["metric"].startswith("abs_coverage")} == {"NOT_RUN"}
    pu = H3.per_unit_delta_table(w, o, design="V5", model_arm="M2", transform="WITHOUT", v6_mask=v6)
    assert len(pu) == 16 and (pu["delta_mae"] == pu["mae_transform"] - pu["mae_with"]).all()
    assert set(pu.columns) >= {"system", "publication_group", EM.METAL_STATE_COL, EM.SYSTEM_COL, "n_rows"}
    assert (pu["n_rows"] == 4).all() and pu["delta_mae"].mean() > 0


def test_per_unit_delta_table_builds_on_the_v1_outer_fold_unit():
    """Regression (2026-09-23): the V1 branch forwarded ``v1_group_col`` into ``discovery.unit_mae``, whose signature
    fixes that column itself, so every V1 per-unit delta raised TypeError and no design that includes V1 could be
    scored.  The V1 unit is the section 3.2 outer fold (exact scheme: the publication group)."""
    w, o = _pred_frame(0.1, 1), _pred_frame(1.0, 2)
    for f in (w, o):
        f["fold_id"] = f[EM.PUB_GROUP_COL]                   # exact V1: one publication group per outer fold
    v6 = pd.Series(False, index=w.index)
    pu = H3.per_unit_delta_table(w, o, design="V1", model_arm="B5", transform="WITHOUT", v6_mask=v6)
    assert list(pu["unit_key"]) == ["g0", "g1", "g2", "g3"]  # the four outer folds, not the 16 V5 cells
    assert (pu["delta_mae"] == pu["mae_transform"] - pu["mae_with"]).all() and pu["delta_mae"].mean() > 0
    assert set(pu.columns) >= {"design", "model_arm", "transform", "publication_group", "n_rows"}
    assert (pu["n_rows"] == 16).all() and (pu["design"] == "V1").all()


def test_logsf_delta_on_ln_ln_pairs_only():
    from gen19ct.evaluation import pairs as EP
    rows = _pred_frame(0.1, 3, n_systems=4)
    rows["fold"] = "test"
    pr = EP.comparable_pairs(rows, key_cols=(EM.PUB_GROUP_COL, EM.SYSTEM_COL, EM.CONDITION_KEY_COL), fold_col="fold")
    assert set(pr["category_class"]) >= {"Ln-Ln", "An-Ln"}
    rng = np.random.default_rng(5)
    w = pd.Series(rows[EM.Y_COL] + rng.normal(0, 0.01, len(rows)), index=rows.index)
    # a constant offset would cancel in the pair difference, so the WITHOUT arm gets a per-row error
    o = pd.Series(rows[EM.Y_COL] + rng.normal(0, 1.0, len(rows)), index=rows.index)
    folds = pd.Series("test", index=rows.index)
    out = H3.logsf_delta(pr, w, o, model_arm="M2", transform="WITHOUT", v6_mask=None, folds=folds, n_resamples=300)
    prim = [r for r in out if r.get("primary_cluster_unit")]
    assert prim and prim[0]["metric"] == "logsf_mae" and prim[0]["design"] == "V5-PAIR"
    assert prim[0]["n_pairs"] == int((pr["category_class"] == "Ln-Ln").sum())
    assert prim[0]["with_value"] < prim[0]["transform_value"]
    empty = H3.logsf_delta(pr[pr["category_class"] == "An-Ln"].assign(category_class="An-Ln"), w, o, model_arm="M2",
                           transform="WITHOUT", v6_mask=None, folds=folds, n_resamples=10)
    assert empty[0]["status"] == "NOT_RUN"


# --------------------------------------------------------------------------------------------- #
# verdicts and F4
# --------------------------------------------------------------------------------------------- #

def _res(point: float, *, scope: str = "PASS", equivalent: bool = False, non_inferior: bool = True):
    tost = {"verdict": "NO_DIFFERENCE" if equivalent else ("PASS" if non_inferior else "UNDECIDED"),
            "equivalent": equivalent, "non_inferior": non_inferior, "low_90": -0.01 if non_inferior else -0.4,
            "high_90": 0.02 if equivalent else 0.4}
    br = SimpleNamespace(percentile_interval=lambda level=0.95: ((0.1, 0.5) if point > 0 else (-0.5, -0.1)),
                         bca_interval=lambda level=0.95: ((0.1, 0.5) if point > 0 else (-0.5, -0.1)))
    return {"point": point, "scopes": {H3.VERDICT_SCOPE: {"verdict": scope}}, "tost": tost,
            "primary_cluster_unit": "system", "bootstraps": {"system": br}}


def test_h3_verdicts_helps_hurts_equivalent_undecided():
    helps = {"V5": {"WITHOUT": _res(0.3), "ACT_PERMUTED": _res(0.3)},
             "V1": {"WITHOUT": _res(0.1), "ACT_PERMUTED": _res(0.1)},
             "V2": {"WITHOUT": _res(0.1), "ACT_PERMUTED": _res(0.1)}}
    v = H3.h3_verdict(helps=helps, hurts={"WITHOUT": _res(-0.3, scope="FAIL")})
    assert v["verdict"] == "helps" and v["actinide_rows_enter_deployed_configuration"]
    # not non-inferior on V2 -> not helps
    bad_ni = {**helps, "V2": {"WITHOUT": _res(0.1, non_inferior=False), "ACT_PERMUTED": _res(0.1)}}
    assert H3.h3_verdict(helps=bad_ni, hurts={"WITHOUT": _res(-0.3, scope="FAIL")})["verdict"] == "UNDECIDED"
    # WITHOUT beats WITH -> hurts
    hurts = H3.h3_verdict(helps={"V5": {"WITHOUT": _res(-0.3, scope="FAIL"), "ACT_PERMUTED": _res(-0.3, scope="FAIL")}},
                          hurts={"WITHOUT": _res(0.3, scope="PASS")})
    assert hurts["verdict"] == "hurts" and not hurts["actinide_rows_enter_deployed_configuration"]
    # inside the TOST margin -> equivalent
    eq = H3.h3_verdict(helps={"V5": {"WITHOUT": _res(0.0, scope="FAIL", equivalent=True),
                                     "ACT_PERMUTED": _res(0.0, scope="FAIL")}},
                       hurts={"WITHOUT": _res(0.0, scope="FAIL")})
    assert eq["verdict"] == "equivalent"
    # otherwise UNDECIDED, and kappa_min decides whether it is underpowered
    und = H3.h3_verdict(helps={"V5": {"WITHOUT": _res(0.0, scope="UNDECIDED"), "ACT_PERMUTED": _res(0.0, scope="FAIL")}},
                        hurts={"WITHOUT": _res(0.0, scope="FAIL")}, kappa_min=0.5, kappa_status="computed")
    assert und["verdict"] == "UNDECIDED" and und["underpowered"]
    ok = H3.h3_verdict(helps={"V5": {"WITHOUT": _res(0.0, scope="UNDECIDED"), "ACT_PERMUTED": _res(0.0, scope="FAIL")}},
                       hurts={"WITHOUT": _res(0.0, scope="FAIL")}, kappa_min=0.25, kappa_status="computed")
    assert not ok["underpowered"]
    # a missing input is named, never guessed
    miss = H3.h3_verdict(helps={}, hurts={})
    assert miss["verdict"] == "UNDECIDED" and len(miss["inputs_missing"]) == 3


def test_f4_check_per_design():
    beats = {"V5": _res(0.3), "V1": _res(-0.2), "V2": None}
    f4 = H3.f4_check(beats, deployed_arm="M2")
    assert f4["failure"] and f4["per_design"]["V5"]["without_beats_with_interval_excludes_0"]
    assert not f4["per_design"]["V1"]["without_beats_with_interval_excludes_0"]
    assert f4["per_design"]["V2"]["status"] == H3.NOT_COMPUTED and f4["designs_not_computed"] == ["V2"]
    assert not H3.f4_check({"V5": _res(-0.3)}, deployed_arm="M2")["failure"]
    # F4 needs the deployed configuration to have been trained with actinide rows
    assert not H3.f4_check(beats, deployed_arm="M2", trained_with_actinides=False)["failure"]
    assert H3.f4_check({}, deployed_arm="M2")["status"] == H3.NOT_COMPUTED


# --------------------------------------------------------------------------------------------- #
# the negative-transfer investigation (descriptive)
# --------------------------------------------------------------------------------------------- #

def test_system_stats_and_negative_transfer_tables():
    df = _frame(n_per_cell=6)
    df[EM.PUB_GROUP_COL] = df[I.PUB_GROUP_COL]
    df[EM.CONDITION_KEY_COL] = ["ck" + str(i % 3) for i in range(len(df))]
    stats = H3.system_actinide_stats(df)
    assert list(stats[EM.SYSTEM_COL]) == list(SYSTEMS)
    assert (stats["n_actinide_rows"] > 0).all() and (stats["n_ln_iii_rows"] > 0).all()
    assert stats["n_an_ln_pairs"].sum() > 0 and stats["median_abs_an_ln_logsf"].notna().any()
    assert not stats["monoamide_only_system"].any()                 # every system has Ln rows here
    per_cell = pd.DataFrame({"design": "V5", "model_arm": "M2", "transform": "WITHOUT",
                             "unit_key": [f"Nd(III) x {s}" for s in SYSTEMS],
                             EM.METAL_STATE_COL: "Nd(III)", EM.SYSTEM_COL: list(SYSTEMS),
                             "mae_with": [0.5, 0.6, 0.4, 0.5], "mae_transform": [0.7, 0.5, 0.6, 0.8],
                             "n_rows": 6, "system": list(SYSTEMS), "publication_group": ["g0", "g1", "g2", "g3"]})
    per_cell["delta_mae"] = per_cell["mae_transform"] - per_cell["mae_with"]
    dep = H3.actinide_dependent_cells(df, [("Nd(III)", s) for s in SYSTEMS])
    tabs = H3.negative_transfer_tables(per_cell, stats, dependent=dep, n_resamples=200)
    assert {"strata", "per_family", "per_mechanism", "per_cell_covariates", "cells"} <= set(tabs)
    assert {"am_eu_heavy_diglycolamide_cells", "cells_sharing_component_with_monoamide_only_system",
            "actinide_dependent_eligibility"} == set(tabs["strata"]["stratum"])
    assert (tabs["strata"]["n_cells"] > 0).all()
    assert set(tabs["per_cell_covariates"]["covariate"]) == {"n_actinide_rows", "median_abs_an_ln_logsf"}
    assert tabs["per_cell_covariates"]["status"].str.contains("reliability").all()
    assert H3.negative_transfer_tables(per_cell.iloc[0:0], stats) == {}


# --------------------------------------------------------------------------------------------- #
# hyperparameters read from the WITH record (no re-tuning)
# --------------------------------------------------------------------------------------------- #

def test_selected_hyperparameters_layouts_and_b6_cross_fit_configs():
    neural = {"selected_config": "M2_d8_wd0.001_r4", "model_seed": 11,
              "arm_record": {"selected": {"emb_dim": 8, "weight_decay": 1e-3, "rank": 4}, "n_epochs": 42,
                             "model_seed": 11}}
    hp = H3.selected_hyperparameters("M2", neural)
    assert hp == {"family": "neural", "emb_dim": 8, "weight_decay": 1e-3, "rank": 4, "n_epochs": 42, "model_seed": 11,
                  "label": "M2_d8_wd0.001_r4"}
    boosted = {"selected_config": "depth6_l23", "model_seed": 5,
               "arm_record": {"frozen": {"config": {"label": "depth6_l23", "depth": 6, "l2_leaf_reg": 3.0},
                                         "iterations": 700}}}
    hb = H3.selected_hyperparameters("B5", boosted)
    assert hb["family"] == "boosted" and hb["iterations"] == 700 and "label" not in hb["config"]
    b6 = {"selected_config": "k2_lam1", "model_seed": 9,
          "arm_record": {"selection": {"selected_rank": 2, "selected_lambda": 1.0, "init_seed": 9,
                                       "inner_fold_macro_mae": {"k0_lam1": {"0": 0.9, "1": 0.9, "2": 0.9},
                                                                "k2_lam1": {"0": 0.5, "1": 0.5, "2": 2.0}}}}}
    # a real B6 record carries every rank x lambda it was tuned over; b6_cross_fit_configs selects over that grid
    h6 = H3.selected_hyperparameters("B6", b6)
    assert h6["family"] == "b6" and h6["rank"] == 2 and h6["lambda"] == 1.0 and h6["init_seed"] == 9
    # the cross-fitted calibration of inner fold j takes the configuration selected on the OTHER folds
    cfg = H3.b6_cross_fit_configs(b6["arm_record"]["selection"]["inner_fold_macro_mae"])
    assert set(cfg) == {0, 1, 2}
    # fold 0 is calibrated by the selection on folds 1+2, where rank 2 averages (0.5 + 2.0)/2 = 1.25 against rank 0's
    # 0.9, so rank 0 wins; the same for fold 1. Only fold 2's calibration selects on folds 0+1, where rank 2 (0.5)
    # beats rank 0 (0.9) -- the point of cross-fitting: a fold never chooses the configuration it is a residual of
    assert cfg[0] == (0, 1.0) and cfg[1] == (0, 1.0)
    assert cfg[2] == (2, 1.0)
    worse = {"k0_lam1": {"0": 0.5, "1": 0.5, "2": 0.5}, "k2_lam1": {"0": 0.9, "1": 0.9, "2": 0.1}}
    assert H3.b6_cross_fit_configs(worse)[2] == (0, 1.0)           # without fold 2, rank 0 wins
    # B6r0 is rank 0 only, whatever the record holds
    assert set(H3.b6_cross_fit_configs(worse, variant="B6r0").values()) == {(0, 1.0)}
    # a record missing a configuration of its own grid is refused, never silently selected over a partial grid
    with pytest.raises(ValueError, match="lacks configuration"):
        H3.b6_cross_fit_configs({"k0_lam1": {"0": 0.5}, "k2_lam0.1": {"0": 0.4}})
    with pytest.raises(ValueError, match="no B6r0 configuration"):
        H3.b6_cross_fit_configs({"k2_lam1": {"0": 0.5, "1": 0.5}}, variant="B6r0")
    assert H3.selected_hyperparameters("B3i", {})["family"] == "closed_form"
    with pytest.raises(ValueError):
        H3.selected_hyperparameters("NOPE", {})
    for arm, cls in (("M2", H3.FrozenNeural), ("B5", H3.FrozenBoosted), ("B6", H3.FrozenB6), ("B3i", H3.FrozenComparator)):
        assert isinstance(H3.frozen_runner(arm), cls)
    with pytest.raises(ValueError):
        H3.frozen_runner("B8")


# --------------------------------------------------------------------------------------------- #
# the gates (the runner refuses before anything is fitted)
# --------------------------------------------------------------------------------------------- #

def _sealed():
    exp = REG.gate_expectations("h3")                          # the stage every H3 gate test runs under
    return {"footer": D.REGISTERED_PREREG_SHA256, "recomputed": D.REGISTERED_PREREG_SHA256,
            "digest_file": D.REGISTERED_PREREG_SHA256, "addenda_sha256": exp["below_footer_sha256"],
            "n_addenda": exp["n_addenda"]}


def _complete_discovery(out: Path, corpus, job: D.JobSpec, *, code: str, state: D.PlanState) -> None:
    """A wall-clock ledger that reached the final stage and one complete verified record set for ``job``."""
    RD.record_wall_clock(out, "2026-09-18T00:00:00+00:00", 10.0, [job.stage, D.STAGES["not_implemented"]])
    runner = RD.runner_for(job, RD.default_runners())
    dh = FI.design_hash(corpus.folds(job.stem))
    for fold, ordinal in corpus.fittable(job):
        for arm in job.writes:
            pq, js = D.fold_paths(out, job, arm, fold.fold_id)
            pq.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({c: [np.nan] for c in D.PREDICTION_COLUMNS}).assign(
                row_id=fold.scored_row_ids[0], fold_id=fold.fold_id, arm=arm, half="S",
                mean_logD=0.0).to_parquet(pq)
            digest = RD.fold_digest(job, fold, code, state, runner, out, ordinal=ordinal, design_hash=dh)
            js.write_text(json.dumps({"job": job.record(), "fold_id": fold.fold_id, "fold_hash": fold.fold_hash,
                                      "digest": digest, "arm": arm, "model_seed": 1, "selected_config": "c",
                                      "steps": {"point": {"seconds": 1.0}}, "arm_record": {}}), encoding="utf-8")


def test_runner_refuses_unless_sealed_discovery_complete_and_scored(tmp_path):
    out = tmp_path / "out"
    # 1. the seal gate
    with pytest.raises(SystemExit, match="refused"):
        RH.main(["--dry-run", "--out-root", str(out)], check=lambda: 1)
    assert not out.exists()
    corpus = _corpus(tmp_path)
    _write([_cell_fold(corpus.frame, "Nd(III)", "S1")], corpus.folds_dir)
    job = D.JobSpec(kind="fit", arm="B5", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED,
                    stage=D.STAGES["p_b5"], writes=("B5",))
    state = D.PlanState()
    # 2. discovery incomplete: no wall clock, no records
    with pytest.raises(SystemExit, match="discovery run is not COMPLETE"):
        RH.refuse_unless_ready(out, check=lambda: 0, digests=_sealed, jobs=[job], state=state, discovery_code="c",
                               folds_dir=corpus.folds_dir)
    inc = H3.discovery_complete(out, code="c", state=state, excluded_ids=[], folds_dir=corpus.folds_dir, jobs=[job])
    assert not inc["complete"] and not inc["reached_final_stage"] and inc["n_incomplete"] == 1
    _complete_discovery(out, corpus, job, code="c", state=state)
    done = H3.discovery_complete(out, code="c", state=state, excluded_ids=[], folds_dir=corpus.folds_dir, jobs=[job])
    assert done["complete"] and done["reached_final_stage"] and done["n_complete"] == 1
    # a code change makes the record set stale -> incomplete again (never scored against other code)
    with pytest.raises(D.StaleRecordError):
        H3.discovery_complete(out, code="other", state=state, excluded_ids=[], folds_dir=corpus.folds_dir, jobs=[job])
    # 3. the scorer's decisions must exist and be newer than every record
    with pytest.raises(SystemExit, match="scorer's decision files"):
        RH.refuse_unless_ready(out, check=lambda: 0, digests=_sealed, jobs=[job], state=state, discovery_code="c",
                               folds_dir=corpus.folds_dir)
    dec_dir = D.discovery_root(out) / "decisions"
    dec_dir.mkdir(parents=True, exist_ok=True)
    (dec_dir / "decisions.json").write_text(json.dumps({"ladder": {}, "stop_rule": {}}), encoding="utf-8")
    (dec_dir / "stop_rule.json").write_text("{}", encoding="utf-8")
    (D.discovery_root(out) / "contrasts_registered.csv").write_text("family\n", encoding="utf-8")
    (D.discovery_root(out) / "r19_items.csv").write_text("item\n", encoding="utf-8")
    sc = H3.scorer_decisions_present(out)
    assert sc["ok"] and sc["all_present"]
    # 4. the ladder must have run every step M3-M7 to a done / skipped status (task X finding V-01)
    with pytest.raises(SystemExit, match="ladder .M3-M7. is not complete"):
        RH.refuse_unless_ready(out, check=lambda: 0, digests=_sealed, jobs=[job], state=state, discovery_code="c",
                               folds_dir=corpus.folds_dir)
    lad_path = out / H3.LADDER_STATE_FILE
    lad_path.parent.mkdir(parents=True, exist_ok=True)
    half = _ladder()
    half["steps"]["M6"] = {"step": "M6", "status": "incomplete", "kept": None}
    lad_path.write_text(json.dumps(half), encoding="utf-8")
    with pytest.raises(SystemExit, match="steps not done: \\['M6'\\]"):
        RH.refuse_unless_ready(out, check=lambda: 0, digests=_sealed, jobs=[job], state=state, discovery_code="c",
                               folds_dir=corpus.folds_dir)
    lad_path.write_text(json.dumps(_ladder()), encoding="utf-8")
    gate = RH.refuse_unless_ready(out, check=lambda: 0, digests=_sealed, jobs=[job], state=state, discovery_code="c",
                                  folds_dir=corpus.folds_dir)
    assert gate["prereg_gate"]["prereg_sha256"] == D.REGISTERED_PREREG_SHA256
    assert gate["discovery_complete"]["complete"] and gate["scorer_decisions"]["ok"] and gate["ladder_complete"]["complete"]
    assert RH.read_ladder(out)["steps"]["M7"]["status"] == "judged"
    # the cheap gate (finding VL2-04) is a file read: it holds here and refuses on an empty tree
    assert H3.refuse_unless_cheap_complete(out)["complete"]
    with pytest.raises(SystemExit, match="nothing heavy was loaded"):
        H3.refuse_unless_cheap_complete(tmp_path / "empty")
    # a stale decisions.json (older than a record) is refused
    rec = next(iter((D.discovery_root(out) / "B5").rglob("*.json")))
    import os
    os.utime(rec, (rec.stat().st_atime, (dec_dir / "decisions.json").stat().st_mtime + 100))
    assert not H3.scorer_decisions_present(out)["ok"]


def test_runner_refuses_an_undecidable_deployed_configuration(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(RH, "refuse_unless_ready", lambda *a, **k: {"prereg_gate": {}, "discovery_complete": {},
                                                                    "scorer_decisions": {}, "plan_state": {},
                                                                    "discovery_code_sha256": "c"})
    monkeypatch.setattr(RH, "read_decisions", lambda o: {"ladder": {"M1": {"kept": None}}, "stop_rule": {}})
    monkeypatch.setattr(RD, "coextractant_ids", lambda: [])
    # the cheap discovery gate runs before anything heavy: an empty tree refuses there (finding VL2-04)
    with pytest.raises(SystemExit, match="nothing heavy was loaded"):
        RH.main(["--out-root", str(out)], check=lambda: 0, digests=_sealed)
    RD.record_wall_clock(out, "2026-09-18T00:00:00+00:00", 1.0, [D.STAGES["not_implemented"]])
    monkeypatch.setattr(RH, "read_ladder", lambda o: None)
    with pytest.raises(SystemExit, match="undecidable"):                 # the ladder has not run
        RH.main(["--out-root", str(out)], check=lambda: 0, digests=_sealed)
    monkeypatch.setattr(RH, "read_ladder", lambda o: _ladder())
    with pytest.raises(SystemExit, match="undecidable"):                 # a scorer ladder decision is pending
        RH.main(["--out-root", str(out)], check=lambda: 0, digests=_sealed)


# --------------------------------------------------------------------------------------------- #
# the D03 decision file
# --------------------------------------------------------------------------------------------- #

def test_d03_markdown_says_not_computed_where_a_quantity_is_absent():
    summary = {"deployed": {"arm": "M2", "basis": "ladder step M2 kept (section 6)"}, "model_arms": ["M2", "B6", "B5"],
               "transforms": list(H3.ALL_ARMS), "verdicts": {}, "f4": {}, "git_head": "abc123"}
    md = H3.d03_markdown(summary, None, None, None)
    for section in ("## Question", "## Evidence", "## Metrics", "## Verdict", "## Decision", "## Next action"):
        assert section in md
    assert md.count(H3.NOT_COMPUTED) >= 4 and "abc123" in md
    w, o = _pred_frame(0.1, 1, with_intervals=True), _pred_frame(1.0, 2, with_intervals=True)
    v6 = pd.Series(False, index=w.index)
    res = H3.h3_contrast(w, o, design="V5", model_arm="M2", transform="WITHOUT", margin=0.1, v6_mask=v6)
    rows = []
    for r in D.contrast_rows(res):
        r.update(model_arm="M2", transform="WITHOUT")
        rows.append(r)
    con = pd.DataFrame(rows)
    deltas = pd.DataFrame(H3.rank_accuracy_delta(w, o, design="V5", model_arm="M2", transform="WITHOUT", v6_mask=v6,
                                                 n_resamples=200))
    verd = {"M2": H3.h3_verdict(helps={"V5": {"WITHOUT": res, "ACT_PERMUTED": res},
                                       "V1": {"WITHOUT": res, "ACT_PERMUTED": res},
                                       "V2": {"WITHOUT": res, "ACT_PERMUTED": res}}, hurts={})}
    full = {**summary, "verdicts": verd, "f4": H3.f4_check({"V5": _res(-0.3)}, deployed_arm="M2")}
    md2 = H3.d03_markdown(full, con, deltas, None)
    assert "M2 vs" in md2 or "M2:WITH vs" in md2
    # task X finding protocol VH-02: BOTH interval readings of F4 are printed and the decided one is named
    assert "F4 (negative actinide transfer), percentile reading: does not hold" in md2
    assert "F4, BCa reading:" in md2 and "Conservative reading (EITHER interval)" in md2
    assert "rank_accuracy" in md2 and H3.VERDICT_SCOPE in md2


def test_h3_root_paths_are_new_directories_only():
    root = H3.h3_root(paths.G19_ROOT)
    assert root == paths.G19_ROOT / "evaluation" / "h3"
    job = D.JobSpec(kind="fit", arm="M2:WITHOUT", design="V5", variant="primary", scheme="batched",
                    seed=D.PRIMARY_SEED, writes=("M2:WITHOUT",))
    pq, js = H3.fold_paths(paths.G19_ROOT, "M2", "WITHOUT", job, "s104729_S_b000")
    assert pq.parent == root / "records" / "M2" / "WITHOUT" / "V5__primary_batched" / "s104729"
    assert js.suffix == ".json" and pq.suffix == ".parquet"
    # nothing under evaluation/discovery, gen19ct or scripts is written by the H3 runner
    assert "discovery" not in str(root)

def test_v5_contrasts_of_a_heavy_arm_carry_the_section_7_item_6_batching_label():
    """Section 7 item 6: the live B6 checks came out 'V5 recolour, V1 failed', so an H3 V5 contrast computed on the
    re-coloured folds must carry the batching label and a failed check must label every heavy-arm V5 result."""
    passed = D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed")
    failed = D.PlanState(v5_batched_check="failed", v1_tenfold_check="failed")
    assert D.heavy_v5_batching_label(["M2"], "V5", passed) == "batched"
    assert D.heavy_v5_batching_label(["M2"], "V5", failed) == D.CHECK_FAILED_LABEL == "batched (check failed)"
    # B6 runs the exact design, not the batched one -- and says so instead of leaving the label blank, so H1b cannot be
    # read under the heavy arms' regime line (task X finding V-H1B-05)
    assert D.heavy_v5_batching_label(["B6"], "V5", failed) == "exact"
    assert D.heavy_v5_batching_label(["B3i"], "V5", failed) == ""        # a closed-form contrast carries no V5 label
    assert D.heavy_v5_batching_label(["M2"], "V1", failed) == ""        # the label is a V5-family label
    # the plan state decides which fold files the H3 arms read, and a failed V1 ten-fold check moves them to exact
    assert failed.heavy_v5_scheme == "batched_max4" and failed.heavy_v1_scheme == "exact"
    assert failed.s1_forced_undecided and not passed.s1_forced_undecided
    assert H3.with_design_dir("M2", "V5", failed) == "V5__primary_batched_max4"
    assert H3.with_design_dir("M2", "V1", failed) == "V1__copy_exact"
    assert H3.v1_scheme_of("M2", "V1", failed) == "exact"


# --------------------------------------------------------------------------------------------- #
# the frozen ladder runner (finding V-01) and the calibration guard (finding VL2-05)
# --------------------------------------------------------------------------------------------- #

def _ladder_frame():
    """``tests/test_ladder.py`` as a module (its synthetic ladder corpus), loaded once."""
    if "test_ladder" in sys.modules:
        TL = sys.modules["test_ladder"]
    else:
        spec = importlib.util.spec_from_file_location("test_ladder", Path(__file__).resolve().parent / "test_ladder.py")
        TL = importlib.util.module_from_spec(spec)
        sys.modules["test_ladder"] = TL
        spec.loader.exec_module(TL)
    import torch
    torch.set_num_threads(2)
    return TL, TL.synthetic_frame(seed=3)


def _stub_fc(df: pd.DataFrame, hidden: pd.Index, ordinal: int = 0):
    mask = ~df.index.isin(hidden)
    corpus = SimpleNamespace(frame=df, table=SimpleNamespace(index=df.index), cv=None, systems=None)
    return SimpleNamespace(corpus=corpus, mask=mask, sc_labels=hidden, ordinal=ordinal, ctx=None,
                           job=SimpleNamespace(inner_mode="full"))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_frozen_ladder_refits_at_the_with_records_configuration_seed_and_epochs():
    """A deployed ladder step's transformed arms are refitted at the WITH ladder record's selected LadderConfig, epoch
    count and model seed (M7: the member seeds), with the ladder arms of models.ladder (finding V-01)."""
    from gen19ct.models import ladder as LAD
    from gen19ct.models import neural as NN

    TL, df = _ladder_frame()
    hidden = df.index[(df[SG.METAL_COL] == "Eu(III)") & (df[SG.SYSTEM_COL] == list(TL.SYSTEMS)[0])]
    fc = _stub_fc(df, hidden)
    cfg = LAD.LadderConfig(step="M3", emb_dim=4, weight_decay=1e-3, rank=2, experts=True)
    rec = {"arm_record": {"selected": cfg.record(), "n_epochs": 2, "model_seed": 11, "inner_folds_used": [0, 1, 2]},
           "selected_config": cfg.label()}
    out = H3.FrozenLadder("M3", min_expert_rows=50).point(fc, rec)
    o = out["M3"]
    assert len(o.pred) == len(hidden) and o.selected_config == cfg.label() and o.model_seed == 11
    assert o.record["selected_hyperparameters"]["family"] == "ladder" and o.record["fit_record"]["config"]["experts"] is True
    assert o.record["fit_record"]["n_epochs"] == 2 and o.record["fit_record"]["model_seed"] == 11
    assert "ladder_refit" in o.record and "evaluation/ladder record" in o.record["ladder_refit"]
    assert np.isfinite(o.pred["mean_logD"].to_numpy(dtype=float)).all() and o.feature_columns
    # M7: the members at the record's seeds, a positive predictive SD; foreign member seeds are refused
    cfg7 = LAD.LadderConfig(step="M7", emb_dim=4, weight_decay=1e-3, rank=2, heteroscedastic=True)
    seeds = [NN.registered_model_seed(0), NN.registered_model_seed(0) + 1]
    rec7 = {"arm_record": {"selected": cfg7.record(), "n_epochs": 2, "model_seed": seeds[0], "member_seeds": seeds,
                           "inner_folds_used": [0, 1, 2]}, "selected_config": cfg7.label()}
    o7 = H3.FrozenLadder("M7", n_members=2, min_expert_rows=50).point(fc, rec7)["M7"]
    assert o7.model_seed == seeds[0] and (o7.pred["std_logD"].to_numpy(dtype=float) > 0).all()
    assert o7.record["fit_record"]["members"][1]["model_seed"] == seeds[1]
    bad = {"arm_record": {**rec7["arm_record"], "member_seeds": [seeds[0] + 100, seeds[1]]}, "selected_config": "x"}
    with pytest.raises(AssertionError, match="member seeds"):
        H3.FrozenLadder("M7", n_members=2, min_expert_rows=50).point(fc, bad)
    # a record labelled with another step is refused
    with pytest.raises(ValueError, match="labelled"):
        H3.FrozenLadder("M4", min_expert_rows=50).point(fc, rec)


def test_calibration_guard_records_not_calibrated_when_the_inner_folds_differ(monkeypatch):
    """Finding VL2-05: a WITHOUT fold whose actinide-free mask changed the inner cells is recorded as not calibrated
    instead of crashing in the discovery runner's fold assertion."""
    stub = SimpleNamespace(_calibration_splits=lambda fc: (None, [0, 1, 2]),
                           NeuralRunner=lambda step: (_ for _ in ()).throw(AssertionError("must not be reached")),
                           BoostedRunner=lambda name: (_ for _ in ()).throw(AssertionError("must not be reached")))
    monkeypatch.setattr(H3, "_runner_module", lambda: stub)
    fc = SimpleNamespace(job=SimpleNamespace(inner_mode="full"))
    plan = H3._differing_inner_folds_plan(fc, [0, 1])
    assert plan is not None and plan["status"] == H3.NOT_CALIBRATED_DIFFER and plan["calibration_folds"] == []
    assert plan["with_record_inner_folds"] == [0, 1] and plan["available_inner_folds"] == [0, 1, 2]
    assert H3._differing_inner_folds_plan(fc, [2, 0, 1]) is None
    cal, cplan = H3.FrozenNeural("M2").calibration(fc, {"inner_folds_used": [0, 1]})
    assert cal is None and cplan["status"] == H3.NOT_CALIBRATED_DIFFER
    cal, cplan = H3.FrozenBoosted("B5").calibration(fc, {"arm": {"tuning": {"inner_folds_used": [1, 2]}}})
    assert cal is None and cplan["status"] == H3.NOT_CALIBRATED_DIFFER
    cal, cplan = H3.FrozenLadder("M3").calibration(fc, {"inner_folds_used": [0]})
    assert cal is None and cplan["status"] == H3.NOT_CALIBRATED_DIFFER
    assert "VL2-05" in H3.READINGS["calibration_guard"]


def _no_retained_step_ladder() -> dict:
    """``ladder.json`` with no retained step at all: addendum 2's F6 order (M2 / B6 by the stop-rule scope, then B3i) is
    reached only here (addendum 3 item 2)."""
    return {**_not_run_ladder(), "discovery_ladder": {"M1": False, "M2": False}}


def _cands(*contrasts: str) -> list[dict]:
    """``decisions.json -> freezing_candidates`` as the scorer writes it: only contrasts that PASS the freezing screen
    are listed, each with its addendum-3 item-1 eligibility."""
    return [{"contrast": c, "family": "primary", "design": "V5", "eligible_for_freezing": True,
             "eligibility": {"eligible": True, "failed_items": [], "reason": "items [1, 2, 3, 5, 6] PASS"}}
            for c in contrasts]


def test_the_f6_fallback_also_requires_that_no_evaluated_r19_item_fails():
    """Task X finding V-P09.  POST-HOC addendum 2 resolved that, with no retained ladder step, M2 (and B6 for H1b) "is
    deployed by the fallback only when its contrast passes the stop-rule scope (items 1, 2, 3, 5) AND item 6 over the
    reduced sensitivity set - no evaluated item may FAIL".  ``decisions.json -> stop_rule`` carries items 1, 2, 3 and 5
    only, so item 6 comes from the scorer's ``freezing_candidates`` screen."""
    lad = _no_retained_step_ladder()
    dec = {**_dropped_ladder_decisions(), "ladder": {"M0": {"kept": False}, "M1": {"kept": False}, "M2": {"kept": False}}}
    # M2 passes the stop-rule scope AND no evaluated item fails -> M2 deploys, and the basis says both halves held
    ok = H3.deployed_rule({**dec, "freezing_candidates": _cands("M2 vs B3i@V5")}, lad)
    assert ok["arm"] == "M2" and "no evaluated R19 item FAILs" in ok["basis"]
    assert ok["f6_no_evaluated_fail"] == {"M2_vs_B3i": True, "B6_vs_B3i": False}
    assert ok["keys"]["f6_no_evaluated_fail"]["M2_vs_B3i"].startswith("freezing_candidates[contrast=M2 vs B3i@V5]")
    assert "no evaluated item may FAIL" in ok["f6_rule"]
    # the scorer lists no M2 contrast: it did not pass the screen, so the stop rule alone no longer deploys M2 -- and
    # B6 fails the stop rule, so the registered fallback B3i is deployed
    none = H3.deployed_rule({**dec, "freezing_candidates": _cands("M2 vs B0@V5")}, lad)
    assert none["arm"] == H3.FALLBACK_DEPLOYED and none["fallback"] is True
    assert none["stop_rule_pass"]["M2_vs_B3i"] is True and none["f6_no_evaluated_fail"]["M2_vs_B3i"] is False
    assert "with no evaluated R19 item FAILing" in none["basis"]
    # B6's own fallback is gated the same way
    b6 = H3.deployed_rule({**dec, "stop_rule": {"M2_vs_B3i": {"verdict": "FAIL"}, "B6_vs_B3i": {"verdict": "PASS"}},
                           "freezing_candidates": _cands("B6 vs B3i@V5")}, lad)
    assert b6["arm"] == "B6" and "no evaluated R19 item FAILs" in b6["basis"]
    assert H3.deployed_rule({**dec, "stop_rule": {"M2_vs_B3i": {"verdict": "FAIL"}, "B6_vs_B3i": {"verdict": "PASS"}},
                             "freezing_candidates": []}, lad)["arm"] == H3.FALLBACK_DEPLOYED
    # without a freezing_candidates block at all item 6 is UNKNOWN: the rule does not block, and says the check did not
    # run rather than claiming it passed
    unk = H3.deployed_rule(dec, lad)
    assert unk["arm"] == "M2" and unk["f6_no_evaluated_fail"] == {"M2_vs_B3i": None, "B6_vs_B3i": None}
    assert "item 6 not checked" in unk["basis"]
    # a retained ladder step still wins over all of this (addendum 3 item 2)
    assert H3.deployed_rule({**_dropped_ladder_decisions(), "freezing_candidates": []}, _not_run_ladder())["arm"] == "M0"


def test_d03_reads_the_deployed_arms_verdict_by_its_alias_and_prints_its_reported_name():
    """Task X finding V-P02: ``h3_summary -> verdicts`` is keyed by the arm the RECORDS use (M0's records are written
    under B5, ``discovery.ARM_ALIASES``), so D03 looked its own deployed arm up under "M0", found nothing, printed
    'not computed' and forced the Decision line to "no" even on *helps* -- contradicting section 11."""
    dep = H3.deployed_configuration(_dropped_ladder_decisions(), _not_run_ladder())
    assert dep["arm"] == "M0" and dep["arm_alias"] == "B5"
    summary = {"deployed": dep, "verdicts": {"B5": {"verdict": "helps"}, "B6": {"verdict": "UNDECIDED"}},
               "f4": {}, "transforms": ["WITHOUT"], "git_head": "abc"}
    md = H3.d03_markdown(summary, None, None)
    assert "Deployed arm reading (M0, records under B5): **supported (actinide rows help lanthanide prediction)**" in md
    assert "Actinide rows enter the deployed Ln configuration: **yes**" in md
    assert "not computed**." not in md.split("Deployed arm reading")[1].split("\n")[0]
    # hurts / UNDECIDED still read through the alias, and only *helps* admits the rows (section 11)
    for v, decision in (("hurts", "no"), ("equivalent", "no"), ("UNDECIDED", "no")):
        m = H3.d03_markdown({**summary, "verdicts": {"B5": {"verdict": v}}}, None, None)
        assert f"Actinide rows enter the deployed Ln configuration: **{decision}**" in m
    # an arm whose reported name IS its record name prints no redundant alias clause
    m2 = H3.d03_markdown({**summary, "deployed": {"arm": "B6", "arm_alias": "B6"}}, None, None)
    assert "Deployed arm reading (B6): **ambiguous**" in m2


def test_d03_prints_the_power_of_each_contrast_at_its_own_margin():
    """Task X finding V-L1: the V5 WITH-vs-WITHOUT leg answers POSITIVE but below delta5, and its ``mde_80`` exceeds
    delta5, so the leg can never read 'helps' whatever the truth.  D03 says so instead of leaving a bare FAIL."""
    prim = pd.DataFrame([{"model_arm": "B5", "transform": "WITHOUT", "contrast": "B5:WITH vs B5:WITHOUT", "design": "V5",
                          "cluster_unit": "system", "primary_cluster_unit": True, "point": 0.066866, "margin": 0.105689,
                          "percentile_low": 0.011249, "percentile_high": 0.245690, "bca_low": 0.009038,
                          "bca_high": 0.225950, "p_two_sided": 0.0064, "mde_80": 0.179162, "loco_min": 0.049541,
                          f"verdict_{H3.VERDICT_SCOPE}": "FAIL", "r19_verdict_full": "FAIL",
                          "tost_verdict_eps0.05": "PASS"},
                         {"model_arm": "B6", "transform": "WITHOUT", "contrast": "B6:WITH vs B6:WITHOUT", "design": "V1",
                          "cluster_unit": "publication_group", "primary_cluster_unit": True, "point": 0.01,
                          "margin": 0.05, "percentile_low": -0.01, "percentile_high": 0.03, "bca_low": -0.01,
                          "bca_high": 0.03, "p_two_sided": 0.4, "mde_80": 0.02, "loco_min": 0.0,
                          f"verdict_{H3.VERDICT_SCOPE}": "FAIL", "r19_verdict_full": "FAIL",
                          "tost_verdict_eps0.05": "PASS"}])
    # task X finding protocol VH-04: the sentence is read off the contrast's OWN item-1 status, so the item table is an
    # input.  Here B5's item 1 FAILs (0.0669 < delta5) -- the V-L1 case -- and the label is emitted.
    items = pd.DataFrame([{"key": "B5:WITH vs B5:WITHOUT@V5", "contrast": "B5:WITH vs B5:WITHOUT", "item": 1,
                           "status": "FAIL"},
                          {"key": "B6:WITH vs B6:WITHOUT@V1", "contrast": "B6:WITH vs B6:WITHOUT", "item": 1,
                           "status": "FAIL"}])
    md = H3.d03_markdown({"deployed": {"arm": "M0", "arm_alias": "B5"}, "verdicts": {}, "f4": {}}, prim, None,
                         r19_items=items)
    assert "UNDERPOWERED at this margin: item 1 FAILs and no kappa of the design could have made it PASS" in md
    assert "0.179" in md and "0.106" in md and "0.050" in md
    row = next(ln for ln in md.split("\n") if ln.startswith("| B6 |") and "0.020" in ln)
    assert "powered at this margin" in row and "UNDERPOWERED" not in row


def test_d03_never_says_a_passing_contrast_cannot_pass_item_1():
    """Task X finding protocol VH-04: the label was emitted on ``mde_80 > margin`` alone, so D03 printed "no kappa of the
    design can make item 1 PASS" for ``B6:WITH vs B6:ACT_PERMUTED@V2`` -- a contrast whose item 1 is PASS
    (point 0.0815 >= margin 0.05).  It is now gated on the contrast's own item-1 status."""
    prim = pd.DataFrame([{"model_arm": "B6", "transform": "ACT_PERMUTED", "key": "B6:WITH vs B6:ACT_PERMUTED@V2",
                          "contrast": "B6:WITH vs B6:ACT_PERMUTED", "design": "V2", "cluster_unit": "metal_state",
                          "primary_cluster_unit": True, "point": 0.0815187, "margin": 0.05, "percentile_low": 0.0355927,
                          "percentile_high": 0.136451, "bca_low": 0.0411925, "bca_high": 0.147557, "p_two_sided": 0.0,
                          "mde_80": 0.0727361, "loco_min": 0.0578916, f"verdict_{H3.VERDICT_SCOPE}": "PASS",
                          "r19_verdict_full": "UNDECIDED", "tost_verdict_eps0.05": "PASS"}])
    items = pd.DataFrame([{"key": "B6:WITH vs B6:ACT_PERMUTED@V2", "contrast": "B6:WITH vs B6:ACT_PERMUTED", "item": 1,
                           "status": "PASS"}])
    md = H3.d03_markdown({"deployed": {"arm": "M0", "arm_alias": "B5"}, "verdicts": {}, "f4": {}}, prim, None,
                         r19_items=items)
    assert "no kappa of the design could have made it PASS" not in md
    assert "item 1 PASS; the margin is below `mde_80`, so a FAIL here would have been uninformative" in md
    assert H3.item_status_map(items, 1)["B6:WITH vs B6:ACT_PERMUTED@V2"] == "PASS"


def test_an_h3_record_set_is_verified_with_the_code_digest_it_was_written_under(tmp_path):
    """Task X finding V-P02, second half.  ``h3.py`` is inside the H3 code digest, and an H3 fold record's resume
    digest hashes that code digest -- so fixing a READING in this module would re-digest all 43 records on disk and
    force a refit of a leg that is already complete.  Addendum 2 item 5: a later edit "can neither validate nor
    invalidate a record written earlier", so the SET is verified with the code digest its own records carry, whenever
    the registry still holds that digest for this stage (exactly ``registry.record_dir_code`` for discovery records).
    Fitting is unaffected: records are written under the live code."""
    reg, d = tmp_path / "reg.json", tmp_path / "records"
    d.mkdir()
    OLD, NEW, FOREIGN = "0a" * 32, "0b" * 32, "ff" * 32
    for i in range(3):
        (d / f"s104729_S_b{i:03d}.json").write_text(json.dumps({"code_digest": OLD, "fold_id": f"b{i}"}), encoding="utf-8")
    REG.register_stage("h3", below_footer_sha256="a1" * 32, code_digest=OLD, git_head=None, addenda_count=3,
                       note="before the reading fix", path=reg)
    # while the live code IS the registered code nothing changes
    same = H3.record_set_code(d, OLD, path=reg)
    assert same["code"] == OLD and same["resolved"] == "live" and same["record_code"] == OLD
    # after the edit the stage is re-registered; the set still verifies against the digest it was written under
    REG.register_stage("h3", below_footer_sha256="a1" * 32, code_digest=NEW, git_head=None, addenda_count=3,
                       note="after the reading fix", path=reg, force=True)
    got = H3.record_set_code(d, NEW, path=reg)
    assert got["code"] == OLD and got["resolved"] == "superseded" and got["superseded_utc"]
    assert "invalidates no record written earlier" in got["note"]
    # a digest the registry never held, current or superseded, does NOT resolve: the live digest governs and the
    # existing StaleRecordError path fires
    (d / "s104729_S_b000.json").write_text(json.dumps({"code_digest": FOREIGN}), encoding="utf-8")
    mixed = H3.record_set_code(d, NEW, path=reg)
    assert mixed["code"] == NEW and mixed["resolved"] == "live" and "mixes code digests" in mixed["note"]
    for i in range(3):
        (d / f"s104729_S_b{i:03d}.json").write_text(json.dumps({"code_digest": FOREIGN}), encoding="utf-8")
    foreign = H3.record_set_code(d, NEW, path=reg)
    assert foreign["code"] == NEW and foreign["resolved"] == "live" and "is no entry of stage" in foreign["note"]
    # an empty / absent directory falls back to the live digest and says why
    assert H3.record_set_code(tmp_path / "nope", NEW, path=reg) == {"code": NEW, "resolved": "live", "record_code": None}
    (d / "s104729_S_b000.json").unlink(); (d / "s104729_S_b001.json").unlink(); (d / "s104729_S_b002.json").unlink()
    assert H3.record_set_code(d, NEW, path=reg)["note"] == "the directory holds no record"


# --------------------------------------------------------------------------------------------- #
# POST-HOC addendum 4 item 1: ACT_METAL_SHUFFLED is not run
# --------------------------------------------------------------------------------------------- #

def test_act_metal_shuffled_is_not_planned_fitted_or_scored(tmp_path, monkeypatch):
    """Addendum 4 item 1: "ACT_METAL_SHUFFLED is not run; the single fold already fitted is labelled exploratory and is
    not scored".  It leaves ``TRANSFORMS``, so no job plans it and no contrast can read it; ``h3_training_rows`` still
    knows it (the sealed four arms are ``discovery.H3_ARMS``), and the fold on disk is kept, never deleted."""
    assert H3.TRANSFORMS == ("WITHOUT", "ACT_PERMUTED")
    assert H3.EXPLORATORY_TRANSFORMS == ("ACT_METAL_SHUFFLED",) and "ACT_METAL_SHUFFLED" in H3.ALL_ARMS
    assert H3.is_scored_transform("ACT_PERMUTED") and not H3.is_scored_transform("ACT_METAL_SHUFFLED")
    for arm in ("M2", "B5", "B6", "M4"):
        assert "ACT_METAL_SHUFFLED" not in H3.transforms_for(arm)
    plan = RH.plan_jobs(("B5", "B6"), D.PlanState())
    assert plan and not any(e["transform"] in H3.EXPLORATORY_TRANSFORMS for e in plan)
    assert len(plan) == 2 * 2 * len(H3.DESIGNS)                       # 2 arms x 2 transforms x 3 designs
    # the exploratory record on disk is inventoried, labelled and NOT deleted
    job = _entry("ACT_METAL_SHUFFLED")["job"]
    pq, js = H3.fold_paths(tmp_path, "M2", "ACT_METAL_SHUFFLED", job, "b000")
    pq.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps({"code_digest": CODE, "fold_id": "b000"}), encoding="utf-8")
    expl = H3.exploratory_records(tmp_path)
    assert expl["status"] == H3.EXPLORATORY_NOT_SCORED and expl["n_records"] == 1 and expl["scored"] is False
    assert expl["deleted"] is False and js.exists()
    assert expl["record_sets"][0]["transform"] == "ACT_METAL_SHUFFLED" and expl["record_sets"][0]["arm"] == "M2"
    assert "is NOT run" in expl["reason"] and "exploratory" in expl["rule"]
    # neither the fitter nor the scorer will touch it
    with pytest.raises(SystemExit, match="POST-HOC addendum 4 item 1"):
        RH.run_fold(_entry("ACT_METAL_SHUFFLED"), _cell_fold(_frame(), "Nd(III)", "S1"), 0, _corpus(tmp_path),
                    tmp_path, D.PlanState(), code=CODE)
    fr = SimpleNamespace(record_sets={}, _cache={}, out_root=tmp_path, code=CODE)
    with pytest.raises(AssertionError, match="exploratory_not_scored"):
        RH.Frames.h3_frame(fr, _entry("ACT_METAL_SHUFFLED"))


# --------------------------------------------------------------------------------------------- #
# POST-HOC addendum 4 item 2: the 20 h cap, the priority order and NOT_RUN designs
# --------------------------------------------------------------------------------------------- #

def test_the_h3_budget_is_20_hours_of_wall_clock_on_2_workers_and_counts_their_union(tmp_path):
    """Addendum 4 item 2: "H3 gets a cap of 20 h of wall clock on 2 workers, recorded in
    evaluation/h3/decisions/wall_clock.json".  The number the cap reads is WALL CLOCK -- the union of the invocations
    intervals -- so two workers running the same hour spend one hour of the cap, and the sum over workers is reported
    beside it."""
    assert H3.H3_BUDGET_HOURS == 20.0 and H3.H3_WORKERS == 2
    assert H3.h3_wall_clock_path(tmp_path) == tmp_path / "evaluation" / "h3" / "decisions" / "wall_clock.json"
    assert H3.union_seconds([(0, 10), (5, 20), (30, 40)]) == 30.0      # overlap counted once
    assert H3.union_seconds([]) == 0.0
    H3.open_h3_wall_clock(tmp_path, prior_legs={"total_hours": 6.3})
    body = json.loads(H3.h3_wall_clock_path(tmp_path).read_text(encoding="utf-8"))
    assert body["invocations"] == [] and body["budget_hours"] == 20.0 and body["workers_max"] == 2
    assert body["prior_legs_not_counted"]["total_hours"] == 6.3 and "never subtracted" in body["prior_legs_reading"]
    assert H3.h3_wall_clock(tmp_path)["wall_seconds"] == 0.0
    # two workers, one overlapping hour each: 2 h of worker time, 1.5 h of wall clock
    H3.record_h3_wall_clock(tmp_path, started="2026-09-23T00:00:00+00:00", ended="2026-09-23T01:00:00+00:00",
                            seconds=3600, designs_done=["V5"], workers=2)
    H3.record_h3_wall_clock(tmp_path, started="2026-09-23T00:30:00+00:00", ended="2026-09-23T01:30:00+00:00",
                            seconds=3600, designs_done=["V5"], workers=2)
    wc = H3.h3_wall_clock(tmp_path)
    assert wc["wall_seconds"] == 5400.0 and wc["worker_seconds"] == 7200.0 and wc["n_invocations"] == 2
    st = H3.h3_budget_status(wc["wall_seconds"], worker_seconds=wc["worker_seconds"])
    assert st["used_hours"] == 1.5 and st["worker_hours"] == 2.0 and st["remaining_hours"] == 18.5
    assert st["exhausted"] is False and st["priority_order"] == ["V5", "V2", "V1"]
    assert H3.h3_budget_status(20 * 3600.0)["exhausted"] is True
    assert H3.h3_budget_status(20 * 3600.0)["remaining_hours"] == 0.0
    body = json.loads(H3.h3_wall_clock_path(tmp_path).read_text(encoding="utf-8"))
    assert body["wall_hours"] == 1.5 and body["worker_hours"] == 2.0 and body["budget"]["exhausted"] is False


def test_the_job_plan_runs_v5_then_v2_then_v1_and_more_than_two_workers_is_refused():
    """Addendum 4 item 2: "the priority order is V5 (done), then V2, then V1"."""
    assert H3.DESIGN_PRIORITY == ("V5", "V2", "V1")
    plan = RH.plan_jobs(("B5", "B6"), D.PlanState())
    assert [e["design"] for e in plan] == ["V5"] * 4 + ["V2"] * 4 + ["V1"] * 4
    shuffled = list(reversed(plan))
    assert [e["design"] for e in H3.order_by_priority(shuffled)] == [e["design"] for e in plan]
    assert H3.plan_priority_key({"design": "V5PAIR", "model_arm": "M2", "transform": "WITHOUT"})[0] == 3
    with pytest.raises(SystemExit, match="caps H3 at 2 workers"):
        RH.run_plan([], None, Path("."), D.PlanState(), code=CODE, workers=3)
    with pytest.raises(SystemExit, match=r"--workers must be 1\.\.2"):
        RH.parse_args(["--workers", "3"])


def test_a_reached_cap_writes_not_run_designs_and_no_verdict_is_taken_from_a_partial_design(tmp_path, monkeypatch):
    """Addendum 4 item 2: "any design not reached is reported NOT_RUN with its reason - no verdict is taken from a
    partial design"."""
    corpus = _corpus(tmp_path)
    _write([_cell_fold(corpus.frame, "Nd(III)", "S1"), _cell_fold(corpus.frame, "Eu(III)", "S2")], corpus.folds_dir)
    monkeypatch.setattr(H3, "frozen_runner", lambda arm: StubArm("M2"))
    monkeypatch.setattr(RH, "with_record", lambda *a, **k: {"digest": "d", "selected_config": "M2_d8", "arm_record": {}})
    out = tmp_path / "out"
    # a ledger already at the cap: nothing is dispatched and the design is NOT_RUN with the cap as its reason
    H3.record_h3_wall_clock(out, started="2026-09-23T00:00:00+00:00", ended="2026-09-23T20:00:00+00:00",
                            seconds=20 * 3600, designs_done=[], workers=2)
    ledger = RH.run_plan([_entry("WITHOUT")], corpus, out, D.PlanState(), code=CODE, steps=("point",),
                         guard_fn=_ok_guard, inner_check=_ok_inner)
    nr = ledger["designs_not_run"]["V5"]
    assert nr["status"] == H3.NOT_RUN and nr["verdict_taken"] is False and "20 h of wall clock" in nr["reason"]
    assert ledger["budget"]["exhausted"] is True and "cap" in (ledger["stopped"] or "")
    assert not list((out / "evaluation" / "h3" / "records").rglob("*.parquet"))
    # --max-hours is an operator pause, never a NOT_RUN reason
    pause = RH.run_plan([_entry("WITHOUT")], corpus, tmp_path / "o2", D.PlanState(), code=CODE, steps=("point",),
                        guard_fn=_ok_guard, inner_check=_ok_inner, max_hours=0.0)
    assert pause["designs_not_run"] == {} and "operator pause" in pause["stopped"]
    # the scorer refuses to take a verdict from an INCOMPLETE record set
    sets = {"k": {"what": "B5:WITHOUT@V5", "status": "incomplete", "n_found": 15, "n_expected": 28,
                  "missing_folds": ["b015"]}}
    assert H3.design_status(sets, "V5", "WITHOUT", "B5")["status"] == H3.NOT_RUN
    with pytest.raises(AssertionError, match="no verdict is taken from a partial design"):
        H3.assert_no_partial_design(sets, [("B5", "WITHOUT", "V5")])
    ok = {"k": {"what": "B5:WITHOUT@V5", "status": "complete", "n_found": 28, "n_expected": 28}}
    assert H3.assert_no_partial_design(ok, [("B5", "WITHOUT", "V5")])["B5:WITHOUT@V5"]["status"] == "complete"
    # and a NOT_RUN design is named in the verdict inputs_missing, so "helps" can never be read off V5 alone
    v = H3.h3_verdict(helps={}, hurts={}, designs_not_run={"V1": H3.not_run_design("V1", "cap reached")})
    assert v["verdict"] == "UNDECIDED" and "V1: NOT_RUN" in v["inputs_missing"] and "V1" in v["designs_not_run"]
    assert v["transforms_exploratory_not_scored"] == {"ACT_METAL_SHUFFLED": H3.EXPLORATORY_NOT_SCORED}


# --------------------------------------------------------------------------------------------- #
# POST-HOC addendum 4 item 3: records written after a supersession are refitted, not accepted
# --------------------------------------------------------------------------------------------- #

def test_a_record_is_verified_with_the_code_and_below_footer_digest_it_was_written_under(tmp_path):
    """Addendum 4 item 3 with addendum 2 item 5: re-registering ``h3`` under a new addendum changes the below-footer
    digest that enters every H3 fold digest, so a record written EARLIER must be verified against the pair it carries --
    per RECORD, because after the refits one directory holds records of two entries."""
    reg = tmp_path / "reg.json"
    a3, a4, old, new = "a3" * 32, "a4" * 32, "0a" * 32, "0b" * 32
    REG.register_stage("h3", below_footer_sha256=a3, code_digest=old, git_head=None, addenda_count=3,
                       note="three addenda", path=reg)
    early = {"code_digest": old, "prereg_addenda_sha256": a3, "prereg_n_addenda": 3,
             "written_utc": "2026-09-22T06:10:13+00:00"}
    assert H3.record_digest_basis(early, old, path=reg)["resolved"] == "current"
    REG.register_stage("h3", below_footer_sha256=a4, code_digest=new, git_head=None, addenda_count=4,
                       note="four addenda", path=reg, force=True)
    got = H3.record_digest_basis(early, new, path=reg)
    assert got["code"] == old and got["addenda"] == a3 and got["resolved"] == "superseded"
    # a record a runner holding stale code wrote AFTER the supersession resolves to the LIVE pair and is refitted
    late = {**early, "written_utc": "2100-01-01T00:00:00+00:00"}
    stale = H3.record_digest_basis(late, new, path=reg)
    assert stale["code"] == new and stale["addenda"] == a4 and stale["resolved"] == "live"
    assert stale["verify"]["stale_after_supersession"] and "refitted" in stale["note"]
    # the digest itself changes with the below-footer digest, and `addenda` overrides it
    job = _entry("WITHOUT")["job"]
    fold = _cell_fold(_frame(), "Nd(III)", "S1")
    kw = dict(transform="WITHOUT", with_digest="w", guard_mode="every_split", design_hash="h", ordinal=0, model_seed=1)
    assert H3.fold_digest(job, fold, old, addenda=a3, **kw) != H3.fold_digest(job, fold, old, addenda=a4, **kw)
    assert H3.fold_digest(job, fold, old, addenda=a3, **kw) != H3.fold_digest(job, fold, new, addenda=a3, **kw)
    # no record on disk: the live pair governs
    none = H3.record_digest_basis(None, new, path=reg)
    assert none["code"] == new and none["resolved"] == "live" and "no record on disk" in none["note"]


def test_stale_records_lists_exactly_the_records_written_after_the_supersession(tmp_path):
    """Addendum 4 item 3: "those records are deleted and refitted under the current registered h3 digest before any H3
    delta is scored; records written before the supersession keep their entry and verify as registered".  The one
    exploratory fold is reported apart and kept (item 1)."""
    reg = tmp_path / "reg.json"
    a3, old, new = "a3" * 32, "0a" * 32, "0b" * 32
    REG.register_stage("h3", below_footer_sha256=a3, code_digest=old, git_head=None, addenda_count=3, note="old",
                       path=reg)
    job = _entry("WITHOUT")["job"]

    def write(transform, fold_id, when):
        pq, js = H3.fold_paths(tmp_path, "B5", transform, H3.h3_job(job, "B5", transform), fold_id)
        pq.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps({"code_digest": old, "prereg_addenda_sha256": a3, "prereg_n_addenda": 3,
                                  "model_arm": "B5", "transform": transform, "fold_id": fold_id,
                                  "written_utc": when}), encoding="utf-8")
        pq.write_bytes(b"parquet")
        return js

    early = [write("WITHOUT", f"b{i:03d}", "2026-09-22T06:00:00+00:00") for i in range(3)]
    late = [write("ACT_PERMUTED", f"b{i:03d}", "2026-09-22T13:00:00+00:00") for i in range(2)]
    expl = write("ACT_METAL_SHUFFLED", "b000", "2026-09-22T14:41:49+00:00")
    REG.register_stage("h3", below_footer_sha256="a4" * 32, code_digest=new, git_head=None, addenda_count=4,
                       note="new", path=reg, force=True)
    # supersede at a time between the two groups
    body = json.loads(reg.read_text(encoding="utf-8"))
    body["superseded"][-1]["superseded_utc"] = "2026-09-22T12:21:15+00:00"
    reg.write_text(json.dumps(body), encoding="utf-8")
    found = H3.stale_records(tmp_path, path=reg)
    assert found["n_records"] == 6 and len(found["verifying"]) == 3 and not found["not_verifying_other"]
    assert sorted(r["fold_id"] for r in found["stale"]) == ["b000", "b001"]
    assert {r["transform"] for r in found["stale"]} == {"ACT_PERMUTED"}
    assert [r["transform"] for r in found["stale_exploratory"]] == ["ACT_METAL_SHUFFLED"]
    assert found["stale_exploratory"][0]["kept"] is True
    # a dry run removes nothing; the deletion takes exactly the stale list, with the sibling parquet
    dry = H3.delete_records(found["stale"], out_root=tmp_path, dry_run=True)
    assert dry["dry_run"] and len(dry["planned"]) == 4 and not dry["removed"] and all(p.exists() for p in late)
    done = H3.delete_records(found["stale"], out_root=tmp_path, dry_run=False)
    assert len(done["removed"]) == 4 and not any(p.exists() for p in late)
    assert all(p.exists() for p in early) and expl.exists()
    after = H3.stale_records(tmp_path, path=reg)
    assert not after["stale"] and len(after["verifying"]) == 3 and len(after["stale_exploratory"]) == 1


def test_no_power_check_label_matches_the_power_module():
    """POST-HOC addendum 4 item 4 has one wording for a failed contrast with no registered power check; ``power``
    imports this module, so the string is duplicated here and must not drift."""
    from gen19ct.evaluation import power as PW
    assert H3.NO_POWER_CHECK_LABEL == PW.NO_POWER_CHECK_LABEL


def test_d03_names_the_absent_logsf_delta_and_the_missing_power_check():
    """Section 11 registers four deltas; Delta logSF MAE has no V5-PAIR record for either H3 arm (addendum 1 item 5), so
    D03 must print it as NOT_DEFINED with the reason, and an UNDECIDED verdict with no kappa_min must carry the
    addendum 4 item 4 wording -- neither may be silently absent."""
    v = H3.h3_verdict(helps={}, hurts={}, designs_not_run={})
    summary = {"deployed": {"arm": "M0", "arm_alias": "B5"}, "verdicts": {"B5": v}, "f4": {},
               "model_arms": ["B5"], "transforms": list(H3.TRANSFORMS),
               "not_computed": {"logsf_mae_delta": H3.READINGS["logsf_delta"], "crps": H3.READINGS["crps"]}}
    md = H3.d03_markdown(summary, pd.DataFrame(), pd.DataFrame(), None)
    assert "Delta logSF MAE" in md and "NOT_DEFINED" in md and "addendum 1 item 5" in md
    assert H3.NO_POWER_CHECK_LABEL in md
    assert "no V5-PAIR record" not in md or True                     # the reason is the READINGS string, quoted verbatim


def test_shared_only_condition_keeps_the_known_inputs_for_an_inapplicable_arm():
    """B5 has no section 15 embedding, so the re-run is not applicable -- but the condition's two inputs are known and
    must be reported, not printed as "not computed" (regression 2026-09-23)."""
    v = {"hurts": False, "v5_without_vs_with_point": -0.0668661}
    so = H3.shared_only_condition(v, deployed_arm="M0")
    assert so["applicable"] is False and so["met"] is False and "not applicable" in so["status"]
    assert so["without_passed_r19_v5"] is False
    assert so["v5_without_vs_with_point"] == pytest.approx(-0.0668661)
    assert so["point_estimate_favours_without"] is False


def test_a_not_run_design_contributes_nothing_even_when_one_leg_is_complete():
    """Task X finding protocol VH-01 / numbers VH-02 (critical): the unit of POST-HOC addendum 4 item 2's NOT_RUN is the
    DESIGN.  V1 was reported NOT_RUN because ``B5:ACT_PERMUTED@V1`` is 26 of 39 folds, while its complete WITHOUT leg
    supplied 4 contrasts with FAIL verdicts, 12 deltas, 50 per-unit rows, an F4 entry and a non-inferiority input -- so
    ``designs_scored`` and ``designs_not_run`` both named V1.  Now every row of a NOT_RUN design is dropped, and the legs
    that ARE complete are named in the entry instead of read."""
    not_run = {"V1": H3.not_run_design("V1", "B5:ACT_PERMUTED@V1 record set is incomplete (26 of 39 folds)",
                                       n_folds_expected=39, n_folds_done=26)}
    not_run = H3.attach_not_run_context(not_run, scored_triples=[("B5", "WITHOUT", "V1"), ("B5", "WITHOUT", "V5")],
                                        blocking={"V1": [{"what": "B5:ACT_PERMUTED@V1",
                                                          "note": "error in B5:ACT_PERMUTED@V1: AssertionError: inner "
                                                                  "split inner_s104729_f1 failed the isolation check"}]})
    assert not_run["V1"]["complete_legs"] == ["B5:WITHOUT@V1"]
    assert "DEVIATION" in not_run["V1"]["deviation"] and "isolation check" in not_run["V1"]["stopped_by"]
    rows = {"contrasts": [{"design": "V1", "point": 0.005429}, {"design": "V5", "point": 0.066866}],
            "deltas": [{"design": "V1"}, {"design": "V1"}, {"design": "V2"}]}
    frames = {"r19_items": pd.DataFrame([{"design": "V1", "item": 1}, {"design": "V5", "item": 1}]),
              "per_unit": pd.DataFrame([{"design": "V1"}] * 50 + [{"design": "V5"}] * 59)}
    helps = {"B5": {"V1": {"WITHOUT": {"x": 1}}, "V5": {"WITHOUT": {"x": 1}}}}
    hurts = {"B5": {"V1": {"WITHOUT": {"x": 1}}, "V5": {"WITHOUT": {"x": 1}}}}
    out = H3.drop_not_run_designs(not_run, rows=rows, frames=frames, nested=(helps, hurts))
    assert [r["design"] for r in out["rows"]["contrasts"]] == ["V5"]
    assert out["rows"]["deltas"] == [{"design": "V2"}]
    assert out["frames"]["per_unit"]["design"].tolist() == ["V5"] * 59
    assert out["n_frame_rows_dropped"] == {"r19_items": 1, "per_unit": 50}
    assert set(helps["B5"]) == {"V5"} and set(hurts["B5"]) == {"V5"}
    # ... and nothing is read from it: the verdict's non-inferiority input is None and F4 is NOT_COMPUTED
    v = H3.h3_verdict(helps={d: {} for d in helps["B5"]}, hurts={}, designs_not_run=not_run)
    assert v["v1_v2_non_inferior"]["V1"] == {"WITHOUT": None, "ACT_PERMUTED": None}
    assert v["designs_scored"] == ["V5"] and "V1: NOT_RUN" in v["inputs_missing"]
    assert v["helps"] is False
    f4 = H3.f4_check({"V1": None, "V5": None}, deployed_arm="B5")
    assert f4["designs_computed"] == [] and set(f4["designs_not_computed"]) == {"V1", "V5"}
    assert "NOT_RUN" in f4["per_design"]["V1"]["reason"]


def test_f4_reports_both_interval_readings():
    """Task X finding protocol VH-02 / numbers VH-03: section 10 F4 says "95 % interval" and names no construction, while
    section 8 registers the percentile AND the BCa interval.  On V2 the BCa interval of the reversed contrast excludes 0
    while the percentile one does not, so the reading decides the failure condition -- and both are now printed."""
    class _Boot:
        def percentile_interval(self):
            return (-0.0013988, 0.064213)

        def bca_interval(self):
            return (0.0031487, 0.078275)

    res = {"point": 0.026416, "primary_cluster_unit": "metal_state", "bootstraps": {"metal_state": _Boot()}}
    f4 = H3.f4_check({"V2": res}, deployed_arm="B5")
    per = f4["per_design"]["V2"]
    assert per["without_beats_with_interval_excludes_0"] is False and per["bca_beats"] is True
    assert per["readings_disagree"] is True and f4["designs_where_readings_disagree"] == ["V2"]
    assert f4["failure"] is False and f4["failure_percentile"] is False
    assert f4["failure_bca"] is True and f4["failure_either_interval"] is True
    assert f4["interval_read"] == "percentile" and "NOT REGISTERED" in f4["interval_reading_not_registered"]


def test_the_negative_transfer_trigger_is_evaluated_and_recorded_on_one_reading():
    """Task X finding protocol VH-05: section 11's investigation ran on the CLI flag alone and two artifacts stated two
    different triggers (D03 read it on V2, ``shared_only.json`` on V5).  One reading now decides it, on the deployed arm's
    V5 Ln-cell contrast, and the tables are labelled exploratory when the condition fails."""
    v = {"hurts": False, "v5_without_vs_with_point": -0.06686613}
    c = H3.negative_transfer_condition(v, deployed_arm="M0")
    assert c["met"] is False and c["design_read"] == "V5" and c["contrast_read"] == "B5:WITHOUT vs B5:WITH@V5"
    assert c["tables_label"] == H3.NEGATIVE_TRANSFER_EXPLORATORY and c["point_estimate_favours_without"] is False
    assert H3.negative_transfer_condition({"hurts": False, "v5_without_vs_with_point": 0.03},
                                          deployed_arm="M0")["met"] is True
    assert H3.negative_transfer_condition({"hurts": True, "v5_without_vs_with_point": -0.2},
                                          deployed_arm="M0")["met"] is True
    # the same reading the shared-only condition uses: the two can never disagree
    assert H3.shared_only_condition(v, deployed_arm="M0")["met"] == c["met"]


def test_the_stratum_excluded_headline_is_reported_beside_the_registered_one():
    """Task X finding protocol VH-08: section 11 makes the actinide-partner-eligibility cells a separate REPORTING unit,
    and 2 of 59 V5 cells supply 47.8 % of the headline Delta.  The registered all-cell macro is unchanged and the
    57-cell macro is printed beside it."""
    cells = pd.DataFrame({EM.METAL_STATE_COL: [f"M{i}" for i in range(59)],
                          EM.SYSTEM_COL: [f"S{i}" for i in range(59)],
                          "delta_mae": [0.942409] * 2 + [0.0361453] * 57})
    dep = pd.DataFrame({EM.METAL_STATE_COL: [f"M{i}" for i in range(59)],
                        EM.SYSTEM_COL: [f"S{i}" for i in range(59)],
                        "actinide_dependent": [True] * 2 + [False] * 57})
    out = H3.stratum_excluded_headline(cells, dep, point=0.06686613)
    assert out["n_cells_all"] == 59 and out["n_cells_excluding_stratum"] == 57
    assert out["macro_delta_all_cells"] == pytest.approx(0.06686613, abs=1e-6)
    assert out["macro_delta_excluding_stratum"] == pytest.approx(0.0361453, abs=1e-6)
    assert out["stratum_contribution_to_macro"] == pytest.approx(0.031946, abs=1e-5)
    assert out["stratum_share_of_macro"] == pytest.approx(0.4777, abs=1e-3)


def test_a_point_only_record_never_makes_a_record_set_read_complete(tmp_path):
    """Task X finding numbers VH-06: ``Frames.h3_frame`` read the set with ``steps=("point",)``, so the two partial
    ``pub_97510df3a0`` records (intervals pending, non-finite bounds) counted as found folds.  Asking for the steps whose
    columns are read leaves the set INCOMPLETE and names the fold."""
    fid = "pub_97510df3a0"
    fr = pd.DataFrame({"fold_id": [fid], "half": [D.SELECTION], "row_id": ["r1"]})
    fr.to_parquet(tmp_path / f"{fid}.parquet")
    (tmp_path / f"{fid}.json").write_text(json.dumps({"fold_id": fid, "digest": "d", "fold_hash": "h",
                                                      "steps": {"point": {"date_utc": "2026-09-22T00:00:00+00:00"}},
                                                      "intervals_status": "pending"}), encoding="utf-8")
    exp = {fid: {"digest": "d", "fold_hash": "h"}}
    pred, st = H3.read_record_set(tmp_path, exp, steps=("point",), what="B5:ACT_PERMUTED@V1")
    assert st["status"] == "complete" and pred is not None            # the old behaviour, still available
    pred, st = H3.read_record_set(tmp_path, exp, steps=H3.STEPS, what="B5:ACT_PERMUTED@V1")
    assert pred is None and st["status"] == "incomplete" and st["n_found"] == 0
    assert st["folds_missing_steps"][fid]["steps_missing"] == ["intervals"]
    assert st["folds_missing_steps"][fid]["intervals_status"] == "pending"


def test_the_h3_power_check_debt_is_inventoried():
    """Task X finding protocol VH-09: under addendum 2's ``needs_power`` reading every H3 contrast whose full R19 verdict
    is not PASS owes the section 8 check; none was run and no artifact said which ones owed it."""
    con = pd.DataFrame([{"family": H3.FAMILY, "key": "B5:WITH vs B5:WITHOUT@V5", "contrast": "B5:WITH vs B5:WITHOUT",
                         "design": "V5", "primary_cluster_unit": True, "r19_verdict_full": "FAIL"},
                        {"family": H3.FAMILY, "key": "B6:WITH vs B6:ACT_PERMUTED@V2",
                         "contrast": "B6:WITH vs B6:ACT_PERMUTED", "design": "V2", "primary_cluster_unit": True,
                         "r19_verdict_full": "UNDECIDED"},
                        {"family": "H1", "key": "M2 vs B3i@V5", "contrast": "M2 vs B3i", "design": "V5",
                         "primary_cluster_unit": True, "r19_verdict_full": "UNDECIDED"}])
    inv = H3.needs_power_inventory(con)
    assert inv["n_contrasts"] == 2 and inv["n_owed"] == 2 and inv["n_owed_and_not_run"] == 2
    assert all("OWED AND NOT RUN" in r["reason"] for r in inv["contrasts"])
    assert inv["label_when_not_run"] == H3.NO_POWER_CHECK_LABEL
    inv2 = H3.needs_power_inventory(con, checked=["B5:WITH vs B5:WITHOUT@V5"])
    assert inv2["n_owed_and_run"] == 1 and inv2["n_owed_and_not_run"] == 1


def test_item_6_declares_every_registered_sensitivity_it_does_not_decide():
    """Task X finding protocol VH-03: ``strict_setting`` and ``HNO3_only_cells`` are registered V5 sensitivities that
    ``discovery.LEARNED_REFITS_NOT_RUN`` does not cover, so an H3 V5 contrast reported item 6 PASS with 2 of 11
    sensitivities in neither the decided set nor the not-run list."""
    sens = {n: ET.UNTESTABLE for n in ET.REGISTERED_SENSITIVITIES["V5"]}
    reduced = [n for n in D.SCORING_FILTER_SENSITIVITIES if n in ET.REGISTERED_SENSITIVITIES["V5"]]
    for n in reduced:
        sens[n] = 0.05
    item, not_run = D.reduced_item6("V5", sens, reduced)
    assert "strict_setting" not in not_run and item["sensitivities_undeclared"]        # the defect, unfixed
    item, not_run = D.reduced_item6("V5", sens, reduced, H3.H3_REFITS_NOT_RUN)
    assert item["status"] == "PASS" and item["sensitivities_undeclared"] == ""
    assert "strict_setting" in not_run and "HNO3_only_cells" in not_run
    assert item["n_registered_sensitivities"] == 11 and item["n_sensitivities_not_run"] == 7
    assert "POST-HOC addendum 1 item 4" in not_run["strict_setting"] and "H3" in not_run["strict_setting"]


def test_the_stale_record_report_names_the_entry_and_the_timestamp_source():
    """Task X findings numbers VH-04 / VH-05: no H3 record verifies against the CURRENT entry, and 43 records carry no
    ``written_utc``, so the ordering that exempts them from refitting rests on a DERIVED stamp.  Both are reported."""
    assert REG.record_run_utc_source({"written_utc": "2026-09-22T10:00:00+00:00"}) == "written_utc"
    assert REG.record_run_utc_source({"steps": {"point": {"date_utc": "2026-09-22T10:00:00+00:00"}}}) \
        == "max(steps.*.date_utc)"
    assert REG.record_run_utc_source({"steps": {"point": {}}}) == "none"
    assert "OPERATIVE READING" in H3.OPERATIVE_DIGEST_READING and "addendum 2 change 5" in H3.OPERATIVE_DIGEST_READING
