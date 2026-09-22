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
    assert "F4 (negative actinide transfer): does not hold" in md2
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
