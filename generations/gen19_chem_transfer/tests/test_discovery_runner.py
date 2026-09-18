"""The discovery runner and scorer core on a synthetic mini-corpus (pre-registration sections 0, 2, 3, 7, 8, 9, 11, 16).

Nothing here reads a real fold's target or fits a learned arm on a registered fold: the fold files are tiny synthetic
files written to ``tmp_path``, the arm is a stub (a training-mean predictor, and the closed-form B0 for the calibration
refits), and the R19 / stop-rule wiring runs on synthetic per-row predictions.
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
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import transfer as ET
from gen19ct.folds import cell_holdout as CH
from gen19ct.folds import io as FI
from gen19ct.models import baselines as B
from gen19ct.models import interface as I

SCRIPTS = paths.G19_ROOT / "scripts"
STATES = ("La(III)", "Nd(III)", "Eu(III)", "Gd(III)", "Am(III)")
SYSTEMS = ("S1", "S2", "S3", "S4")
HALF_OF_SYSTEM = {"S1": "S", "S2": "S", "S3": "C", "S4": "C"}


def _load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RD = _load_script("g19_run_discovery")


# --------------------------------------------------------------------------------------------- #
# the synthetic mini-corpus
# --------------------------------------------------------------------------------------------- #

def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    recs = []

    def add(state, system, n, group, element=None, acid="HNO3"):
        for _ in range(n):
            k = len(recs)
            el = element if state is None else SG.metal_properties(state)["symbol"]
            recs.append({FI.ROW_ID: f"T:{k:05d}", SG.METAL_COL: state, SG.ELEMENT_COL: el, SG.SYSTEM_COL: system,
                         SG.PUB_COL: f"pub_{group}", FI.GROUP_COL: group, I.PUB_GROUP_COL: group, "acid_primary": acid,
                         SG.ACID_ANION_COL: "nitrate", SG.LOG_ACID_COL: float(rng.normal()),
                         SG.LOG_EXT_COL: float(rng.normal() - 1), SG.TEMP_COL: 25.0,
                         I.TARGET_COL: float(rng.normal() + 0.3 * SYSTEMS.index(system))})
    for i, st in enumerate(STATES):
        for j, sy in enumerate(SYSTEMS):
            add(st, sy, 3, f"g{(i + j) % 6}")
            add(st, sy, 3, f"g{(i + 2 * j + 1) % 6}")
    add(None, "S1", 3, "g2", element="Nd")
    df = pd.DataFrame(recs)
    df.index = pd.Index([f"L{i}" for i in range(len(df))])
    return df


def _corpus(tmp_path: Path, df: pd.DataFrame | None = None) -> RD.Corpus:
    df = _frame() if df is None else df
    v6 = pd.Series(False, index=df.index)
    la_s2 = df.index[(df[SG.METAL_COL] == "La(III)") & (df[SG.SYSTEM_COL] == "S2")]
    v6.loc[la_s2[:1]] = True                                     # one V6_TARGET_ROWS row (La(III) x S2)
    coext = pd.Series(False, index=df.index)
    halves = {"V5": df[SG.SYSTEM_COL].map(HALF_OF_SYSTEM).fillna("NA"),
              "V1": df[SG.SYSTEM_COL].map(HALF_OF_SYSTEM).fillna("NA"),
              "V2": pd.Series("S", index=df.index)}
    halves["V5P"] = halves["V5PAIR"] = halves["V5"]
    folds_dir = tmp_path / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    return RD.Corpus(frame=df, slim=df, table=I.RowTable(df), v6=v6, coext=coext, systems=None, comps=None, cv=None,
                     pmap=None, row_half=halves, folds_dir=folds_dir)


def _cell_fold(df: pd.DataFrame, state: str, system: str, *, variant: str = "primary", half: str | None = None,
               row_half: str | None = None) -> FI.Fold:
    own = (df[SG.METAL_COL] == state) & (df[SG.SYSTEM_COL] == system)
    alias = df[SG.METAL_COL].isna() & (df[SG.ELEMENT_COL] == SG.metal_properties(state)["symbol"]) & \
        (df[SG.SYSTEM_COL] == system)
    hid = df.loc[own | alias, FI.ROW_ID].tolist()
    sc = df.loc[own, FI.ROW_ID].tolist()
    h = half or HALF_OF_SYSTEM[system]
    rh = row_half or h
    label = f"{state} x {system}"
    return FI.make_fold(design="V5", variant=variant, scheme="exact", fold_id=f"{state}__{system}", half=h, seed=None,
                        hidden=hid, scored=sc, unit_type="cell", units=[label], row_unit={r: label for r in hid},
                        row_half={r: rh for r in hid}, meta={"cells": [[state, system]], "component_aware": True})


def _write(folds, folds_dir: Path) -> str:
    FI.write_design(folds, folds_dir)
    f = folds[0]
    return FI.design_stem(f.design, f.variant, f.scheme)


class StubRunner:
    """A training-mean 'arm' with the runner protocol; the calibration refits use the closed-form B0 on a grouped
    3-fold inner design (the hook production runners do not define)."""

    has_interval_step = True

    def __init__(self, name: str = "B5"):
        self.arms = (name,)
        self.name = name
        self.point_calls = 0
        self.cal_calls = 0
        self.train_masks: list[np.ndarray] = []

    def point(self, fc):
        self.point_calls += 1
        self.train_masks.append((fc.fold, fc.mask.copy()))
        y = fc.corpus.table.y[fc.mask]
        pred = I.records_to_frame([I.empty_prediction_record() for _ in fc.sc_ids])
        pred["mean_logD"] = float(np.mean(y))
        pred["fallback_level"] = "stub"
        return {self.name: RD.ArmOutput(pred=pred, selected_config="stub_mean", model_seed=7,
                                        record={"n_train": int(fc.mask.sum())}, seconds=0.5)}

    def calibration(self, fc, rec):
        """The runner protocol: ``(CrossFitResidualConformal | None, plan)``; closed-form B0 per calibration fold."""
        self.cal_calls += 1
        spl = I.GroupKFoldCalibration(3)
        folds = sorted({sp.fold for sp in spl.splits(fc.corpus.table, fc.mask, fc.ctx)})
        plan = D.calibration_folds(folds, fc.job.inner_mode, tuned=False)
        return D.CrossFitResidualConformal({j: B.BaselineArm("B0") for j in plan["calibration_folds"]}, spl), plan


def _ok_guard(job, fold, corpus, universe):
    return [{"level": "stub", "ok": True, "publication_basis": None, "n_train": len(universe), "n_test": 0}]


def _ok_inner(job, corpus):
    return lambda tr, te: {"ok": True}


def _job(variant: str = "primary", arm: str = "B5", seed: int = D.PRIMARY_SEED) -> D.JobSpec:
    return D.JobSpec(kind="fit", arm=arm, design="V5", variant=variant, scheme="exact", seed=seed, stage="t")


def _run(job, corpus, out, runner, *, code="code-A", steps=("point", "intervals")):
    return RD.run_jobs([job], corpus, out, steps=list(steps), code=code, state=D.PlanState(), runners={job.arm: runner},
                       guard_fn=_ok_guard, inner_check=_ok_inner, with_support=False)


# --------------------------------------------------------------------------------------------- #
# resume, steps, records
# --------------------------------------------------------------------------------------------- #

def test_resume_skips_done_folds_and_recomputes_on_a_code_change(tmp_path):
    corpus = _corpus(tmp_path)
    df = corpus.frame
    _write([_cell_fold(df, "Nd(III)", "S1"), _cell_fold(df, "Eu(III)", "S2"), _cell_fold(df, "Gd(III)", "S3")],
           corpus.folds_dir)
    out, stub = tmp_path / "out", StubRunner()
    job = _job()
    led = _run(job, corpus, out, stub)
    assert stub.point_calls == 2                                  # the confirmation-half fold (S3) is never fitted
    assert led["jobs"][job.key]["fitted"] == 2 and not led["jobs"][job.key]["errors"]
    pq, js = D.fold_paths(out, job, "B5", "Nd(III)__S1")
    rec = json.loads(js.read_text(encoding="utf-8"))
    assert set(rec["steps"]) == {"point", "intervals"} and rec["selected_config"] == "stub_mean"
    assert rec["fold_hash"] == corpus.folds(job.stem)[0].fold_hash and rec["design_hash"] == corpus.design_hash(job.stem)
    assert rec["prereg_sha256"] == D.REGISTERED_PREREG_SHA256 and rec["model_fold_number"] == rec["fold_ordinal"]
    assert rec["steps"]["intervals"]["status"] == "calibrated" and rec["steps"]["intervals"]["calibration_folds"] == [0, 1, 2]
    frame = pd.read_parquet(pq)
    assert list(frame.columns) == list(D.PREDICTION_COLUMNS) and (frame["half"] == "S").all()
    assert np.isfinite(frame["lower_80"]).all() and (frame["intervals_status"] == "split_conformal_inner").all()
    assert not (tmp_path / "out" / "evaluation" / "discovery" / "B5" / "V5__primary_exact" / "s104729"
                / "Gd(III)__S3.parquet").exists()
    # the second run skips every done fold
    led2 = _run(job, corpus, out, stub)
    assert stub.point_calls == 2 and stub.cal_calls == 2
    assert led2["jobs"][job.key]["skipped"] == 2 and led2["jobs"][job.key]["fitted"] == 0
    # a different code digest makes the records stale: recomputed
    _run(job, corpus, out, stub, code="code-B")
    assert stub.point_calls == 4
    # the hidden rows (scored rows and the X(?) alias rows) were never training rows
    for f, m in stub.train_masks:
        assert not m[corpus.table.positions(corpus.labels_of(f.hidden_row_ids))].any()
        assert m.sum() == corpus.table.n - len(f.hidden_row_ids)
    assert D.ledger_from_records(out)["seconds"].sum() > 0


def test_m2_refuses_a_stale_m1_record(tmp_path):
    """Task X finding V-LP-03: M2 builds on M1's per-fold values only when M1's record digest is the current one."""
    df = _frame()
    f = _cell_fold(df, "Nd(III)", "S1")
    m2 = _job(arm="M2")
    state, out = D.PlanState(), tmp_path / "out"
    nr = RD.NeuralRunner("M2")
    fc = SimpleNamespace(job=m2, fold=f, out_root=out, code="c", state=state, ordinal=3, design_hash="h")
    with pytest.raises(RuntimeError, match="M1's per-fold values"):
        nr.m1_record(fc)
    mjob = nr.m1_job(m2)
    pq, js = D.fold_paths(out, mjob, "M1", f.fold_id)
    pq.parent.mkdir(parents=True)
    pd.DataFrame({"row_id": ["x"]}).to_parquet(pq)
    good = RD.fold_digest(mjob, f, "c", state, RD.NeuralRunner("M1"), out, ordinal=3, design_hash="h")
    js.write_text(json.dumps({"digest": "stale", "fold_hash": f.fold_hash, "steps": {"point": {}}}))
    with pytest.raises(D.StaleRecordError):
        nr.m1_record(fc)
    js.write_text(json.dumps({"digest": good, "fold_hash": f.fold_hash, "steps": {"point": {}}}))
    assert nr.m1_record(fc)["digest"] == good
    # M2's own digest follows M1's expected digest (a change of M1's inputs re-runs M2)
    d_m2 = RD.fold_digest(m2, f, "c", state, nr, out, ordinal=3, design_hash="h")
    assert d_m2 != RD.fold_digest(m2, f, "c2", state, nr, out, ordinal=3, design_hash="h")
    assert nr.digest_extra(m2, f, "c", state, out, ordinal=3, design_hash="h") == {"m1_expected_digest": good}


def test_intervals_step_can_be_deferred_and_resumed(tmp_path):
    corpus = _corpus(tmp_path)
    _write([_cell_fold(corpus.frame, "Nd(III)", "S1"), _cell_fold(corpus.frame, "Eu(III)", "S2")], corpus.folds_dir)
    out, stub, job = tmp_path / "out", StubRunner(), _job()
    _run(job, corpus, out, stub, steps=("point",))
    assert stub.point_calls == 2 and stub.cal_calls == 0
    pq, js = D.fold_paths(out, job, "B5", "Eu(III)__S2")
    assert set(json.loads(js.read_text(encoding="utf-8"))["steps"]) == {"point"}
    assert pd.read_parquet(pq)["lower_95"].isna().all()
    _run(job, corpus, out, stub, steps=("point", "intervals"))
    assert stub.point_calls == 2 and stub.cal_calls == 2            # the point step is not repeated
    rec = json.loads(js.read_text(encoding="utf-8"))
    assert rec["steps"]["intervals"]["n_calibration"] > 0
    fr = pd.read_parquet(pq)
    q = rec["steps"]["intervals"]["quantiles"]["0.8"]
    assert np.allclose(fr["upper_80"] - fr["mean_logD"], q)


def test_max_hours_stops_dispatch_and_wall_clock_ledger(tmp_path):
    corpus = _corpus(tmp_path)
    _write([_cell_fold(corpus.frame, "Nd(III)", "S1")], corpus.folds_dir)
    out, stub, job = tmp_path / "out", StubRunner(), _job()
    led = RD.run_jobs([job], corpus, out, steps=["point"], code="c", state=D.PlanState(), runners={"B5": stub},
                      guard_fn=_ok_guard, inner_check=_ok_inner, with_support=False, max_hours=60.0,
                      budget_used_s=lambda: 60.0 * 3600)
    assert stub.point_calls == 0 and "max-hours" in led["stopped"] and "operator pause" in led["stopped"]
    RD.record_wall_clock(out, "2026-09-15T00:00:00+00:00", 7200.0, ["1_B6"])
    RD.record_wall_clock(out, "2026-09-15T05:00:00+00:00", 1800.0, ["1_B6", "2_comparator_intervals"])
    RD.record_wall_clock(out, "2026-09-15T05:00:00+00:00", 3600.0, ["1_B6", "2_comparator_intervals"])   # same run
    assert RD.wall_clock_total(out) == 10800.0
    body = json.loads(RD.wall_clock_path(out).read_text(encoding="utf-8"))
    assert body["total_hours"] == 3.0 and not body["budget"]["exhausted"]


def test_budget_exhaustion_demotes_only_m3_to_m7(tmp_path):
    """Task X finding V-01: past 60 h the H1 / H1b / H4 work (and every other job of the plan) keeps running; only
    M3-M7 jobs are demoted and the exhaustion is recorded."""
    corpus = _corpus(tmp_path)
    _write([_cell_fold(corpus.frame, "Nd(III)", "S1")], corpus.folds_dir)
    out, stub = tmp_path / "out", StubRunner("M2")
    m2 = _job(arm="M2")
    m3 = D.JobSpec(kind="fit", arm="M3", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED)
    led = RD.run_jobs([m3, m2], corpus, out, steps=["point"], code="c", state=D.PlanState(), runners={"M2": stub},
                      guard_fn=_ok_guard, inner_check=_ok_inner, with_support=False,
                      budget_used_s=lambda: 75.0 * 3600)
    assert stub.point_calls == 1 and led["jobs"][m2.key]["fitted"] == 1 and not led["stopped"]
    assert led["demoted"] == [m3.key] and led["budget"]["exhausted"] and led["budget"]["demoted_now"][0] == "M7"
    assert "keeps running" in D.budget_status(75 * 3600)["note"]
    assert RD.parse_args([]).max_hours is None


def test_cross_fit_residual_conformal_uses_each_folds_arm():
    df = _frame()
    t = I.RowTable(df)
    mask = np.ones(t.n, dtype=bool)
    ctx = I.FitContext(table=t, v6_mask=pd.Series(False, index=df.index), seed=D.PRIMARY_SEED,
                       isolation_check=lambda tr, te: {"ok": True})
    spl = I.GroupKFoldCalibration(3)
    arms = {0: B.BaselineArm("B0"), 1: B.BaselineArm("B1"), 2: B.BaselineArm("B2")}
    cf = D.CrossFitResidualConformal(arms, spl, guard="every_split").fit_table(t, mask, ctx)

    class Only:
        def __init__(self, f):
            self.f = f

        def splits(self, table, m, c):
            return [sp for sp in spl.splits(table, m, c) if sp.fold == self.f]
    want = np.concatenate([I.ConformalWrapper(arms[f], splitter=Only(f)).fit_table(t, mask, ctx).residuals
                           for f in (0, 1, 2)])
    assert np.allclose(cf.residuals, want) and cf.record()["calibration_folds"] == [0, 1, 2]
    with pytest.raises(AssertionError, match="not a calibration fold"):
        D.CrossFitResidualConformal({0: arms[0]}, spl).fit_table(t, mask, ctx)
    assert D.calibration_folds([0, 1, 2], "full") == {"tuning_folds": [0, 1, 2], "calibration_folds": [0, 1, 2],
                                                     "cross_fit": True, "status": "calibrated"}
    assert D.calibration_folds([1, 2], "first")["calibration_folds"] == [2]
    assert D.calibration_folds([0], "first")["status"].startswith("not_calibrated")
    assert D.calibration_folds([0], "full")["status"].startswith("not_calibrated")
    assert D.calibration_folds([0, 1], "first", tuned=False)["calibration_folds"] == [0]


def test_seal_check_refusal_stops_before_anything(tmp_path, monkeypatch):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="refused"):
        RD.main(["--dry-run", "--out-root", str(out)], check=lambda: 1)
    assert not out.exists()
    # the default gate is the seal script's --check (monkeypatched here, never the real sealing)
    monkeypatch.setattr(RD, "seal_check", lambda: 2)
    with pytest.raises(SystemExit, match="exited 2"):
        RD.refuse_unless_sealed()
    good = {"footer": D.REGISTERED_PREREG_SHA256, "recomputed": D.REGISTERED_PREREG_SHA256,
            "digest_file": D.REGISTERED_PREREG_SHA256, "addenda_sha256": D.REGISTERED_ADDENDA_SHA256, "n_addenda": 1}
    rec = RD.refuse_unless_sealed(lambda: 0, lambda: good)
    assert rec["prereg_sha256"] == D.REGISTERED_PREREG_SHA256
    assert rec["addenda_sha256"] == rec["addenda_sha256_registered"] == D.REGISTERED_ADDENDA_SHA256
    assert rec["n_addenda"] == rec["n_addenda_expected"] == D.N_ADDENDA_EXPECTED == 1


def test_prereg_gate_pins_the_registered_digest(tmp_path):
    """Task X finding V-LP-02: a self-consistent re-sealed text passes --check but not the pinned digest."""
    resealed = "b27d2347b790" + "0" * 52
    for bad in ({"footer": resealed, "recomputed": resealed, "digest_file": resealed},
                {"footer": D.REGISTERED_PREREG_SHA256, "recomputed": resealed, "digest_file": D.REGISTERED_PREREG_SHA256},
                {"footer": D.REGISTERED_PREREG_SHA256, "recomputed": D.REGISTERED_PREREG_SHA256, "digest_file": None}):
        with pytest.raises(SystemExit, match="not the registered text"):
            RD.refuse_unless_sealed(lambda: 0, lambda b=bad: b)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="not the registered text"):
        RD.main(["--dry-run", "--out-root", str(out)], check=lambda: 0,
                digests=lambda: {"footer": resealed, "recomputed": resealed, "digest_file": resealed})
    assert not out.exists()
    # the real sealed file carries the registered digest (read with the seal script's own functions)
    real = RD.prereg_digests()
    assert real["footer"] == real["recomputed"] == real["digest_file"] == D.REGISTERED_PREREG_SHA256
    # every record carries the digest; the scorer calls the same gate
    SD = _load_script("g19_score_discovery")
    with pytest.raises(SystemExit, match="not the registered text"):
        SD.main(["--out-root", str(out), "--no-manifest"], check=lambda: 0,
                digests=lambda: {"footer": resealed, "recomputed": resealed, "digest_file": resealed})


def test_seal_gate_pins_the_number_of_posthoc_addenda(tmp_path):
    """The footer digest does not cover the POST-HOC addenda below it, so the gate pins how many there are: this code
    implements addendum 1 and refuses any other count unless the operator passes --expect-addenda."""
    base = {"footer": D.REGISTERED_PREREG_SHA256, "recomputed": D.REGISTERED_PREREG_SHA256,
            "digest_file": D.REGISTERED_PREREG_SHA256, "addenda_sha256": D.REGISTERED_ADDENDA_SHA256}
    SD = _load_script("g19_score_discovery")
    for n in (0, 2):
        with pytest.raises(SystemExit, match=f"carries {n} POST-HOC"):
            RD.refuse_unless_sealed(lambda: 0, lambda k=n: {**base, "n_addenda": k})
        deliberate = RD.refuse_unless_sealed(lambda: 0, lambda k=n: {**base, "n_addenda": k}, expect_addenda=n)
        assert deliberate["n_addenda"] == n and deliberate["addendum_implemented"] == D.N_ADDENDA_EXPECTED
        # ... but --expect-addenda never passes an addendum text the code does not register (task X finding V-F01)
        with pytest.raises(SystemExit, match="addendum text was edited or extended"):
            RD.refuse_unless_sealed(lambda: 0, lambda k=n: {**base, "n_addenda": k, "addenda_sha256": "abc"},
                                    expect_addenda=n)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="carries 0 POST-HOC"):
        RD.main(["--dry-run", "--out-root", str(out)], check=lambda: 0, digests=lambda: {**base, "n_addenda": 0})
    assert not out.exists()
    with pytest.raises(SystemExit, match="carries 2 POST-HOC"):
        SD.main(["--out-root", str(out), "--no-manifest"], check=lambda: 0, digests=lambda: {**base, "n_addenda": 2})
    # the sealed file on disk carries exactly the addendum this code implements, and its text is digested for the record
    real = RD.prereg_digests()
    assert real["n_addenda"] == D.N_ADDENDA_EXPECTED == 1 and len(str(real["addenda_sha256"])) == 64
    assert real["addenda_sha256"] == D.REGISTERED_ADDENDA_SHA256
    assert RD.refuse_unless_sealed(lambda: 0)["addenda_sha256"] == real["addenda_sha256"]


def test_seal_gate_pins_the_addendum_text_not_only_its_count(tmp_path):
    """Task X finding V-F01: an edit INSIDE the POST-HOC addendum (here the seed set of item 3) passes the seal script's
    ``--check`` (the footer digest does not cover the addenda) and keeps the count at 1, so the gate must refuse on the
    below-footer digest.  Run on a scratch copy; the real file is never touched."""
    import shutil
    S = RD._seal_module()
    root = tmp_path / "tree"
    (root / "manifests").mkdir(parents=True)
    text = (paths.G19_ROOT / "preregistration.md").read_bytes().decode("utf-8")
    needle = "**104729 only**"
    assert text.count(needle) == 1
    (root / "preregistration.md").write_bytes(text.replace(needle, "**104729 and 130363**").encode("utf-8"))
    for f in ("prereg_sha256.txt", "confirmation_seeds_sha256.txt"):
        shutil.copy(paths.G19_ROOT / "manifests" / f, root / "manifests" / f)
    ok, msgs = S.check(S.PreregPaths(root=root, repo_root=paths.REPO_ROOT))
    assert ok and any("addenda below the footer: 1" in m for m in msgs)          # the seal script alone accepts it
    tam = RD.prereg_digests(root / "preregistration.md", root / "manifests" / "prereg_sha256.txt")
    real = RD.prereg_digests()
    assert tam["footer"] == tam["recomputed"] == tam["digest_file"] == D.REGISTERED_PREREG_SHA256
    assert tam["n_addenda"] == real["n_addenda"] == 1 and tam["addenda_sha256"] != real["addenda_sha256"]
    with pytest.raises(SystemExit, match="addendum text was edited or extended"):
        RD.refuse_unless_sealed(lambda: 0, lambda: tam)
    with pytest.raises(SystemExit, match="addendum text was edited or extended"):
        RD.refuse_unless_sealed(lambda: 0, lambda: tam, expect_addenda=1)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="addendum text was edited"):
        RD.main(["--dry-run", "--out-root", str(out)], check=lambda: 0, digests=lambda: tam)
    assert not out.exists()
    SD = _load_script("g19_score_discovery")
    with pytest.raises(SystemExit, match="addendum text was edited"):
        SD.main(["--out-root", str(out), "--no-manifest"], check=lambda: 0, digests=lambda: tam)
    # the real file is the registered addendum text, and the digest is fixed in the code
    assert real["addenda_sha256"] == D.REGISTERED_ADDENDA_SHA256
    assert RD.refuse_unless_sealed(lambda: 0, lambda: real)["addenda_sha256_registered"] == D.REGISTERED_ADDENDA_SHA256


# --------------------------------------------------------------------------------------------- #
# the confirmation half and V6 are never scored
# --------------------------------------------------------------------------------------------- #

def test_confirmation_half_is_never_scored(tmp_path):
    corpus = _corpus(tmp_path)
    df = corpus.frame
    # a fold whose file claims the selection half for rows of a confirmation-half system
    tampered = _cell_fold(df, "Am(III)", "S3", variant="strict", half="S", row_half="S")
    stem = _write([tampered], corpus.folds_dir)
    job = _job("strict")
    with pytest.raises(AssertionError, match="confirmation half"):
        RD.prepare_fold(job, corpus.folds(stem)[0], 0, corpus, tmp_path / "out", D.PlanState(), _ok_guard, _ok_inner)
    stub = StubRunner()
    led = _run(job, corpus, tmp_path / "out", stub)
    assert stub.point_calls == 0 and led["jobs"][job.key]["errors"]
    # a confirmation-half fold is never a job fold, and a mixed-half fold keeps its selection rows only
    conf = _cell_fold(df, "Gd(III)", "S4")
    assert D.job_folds(job, [conf]) == []
    mixed = FI.make_fold(design="V1", variant="copy", scheme="grouped10", fold_id="s1_f0", half="NA", seed=1,
                         hidden=["a", "b", "c"], scored=["a", "b", "c"], unit_type="publication_group", units=["u"],
                         row_half={"a": "S", "b": "C", "c": "S"})
    assert D.selection_scored_ids(mixed) == ("a", "c")
    with pytest.raises(AssertionError, match="confirmation half"):
        D.assert_selection_rows(["a", "b"], {"a": "S", "b": "C"}, "x")
    with pytest.raises(AssertionError, match="confirmation half"):
        D.assert_selection_rows(["z"], {}, "x")                  # a row without a registered half is refused too
    # the pre-seal comparator reader never materialises confirmation-half rows
    p = tmp_path / "pre.parquet"
    pd.DataFrame({"row_id": ["a", "b"], "arm": ["B3i", "B3i"], "half": ["S", "C"], "mean_logD": [1.0, 2.0]}).to_parquet(p)
    got = D.read_preseal_selection(p, arms=["B3i"])
    assert got["row_id"].tolist() == ["a"]
    # a scoring frame holding a confirmation-half row raises
    attrs = pd.DataFrame({"registered_half_V5": ["S", "C"]}, index=pd.Index(["a", "b"]))
    pred = pd.DataFrame({"row_id": ["a", "b"], "fold_id": ["f", "f"], "mean_logD": [0.0, 0.0]})
    with pytest.raises(AssertionError, match="confirmation half"):
        D.scoring_frame(pred, attrs, design="V5", v6_mask=pd.Series(False, index=attrs.index), what="t")
    with pytest.raises(AssertionError, match="non-selection-half"):
        D.scoring_frame(pred.assign(half=["S", "C"]), attrs, design="V5", v6_mask=pd.Series(False, index=attrs.index),
                        what="t")


def test_v6_target_rows_are_never_scored(tmp_path):
    corpus = _corpus(tmp_path)
    stem = _write([_cell_fold(corpus.frame, "La(III)", "S2", variant="loose")], corpus.folds_dir)
    job = _job("loose")
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        RD.prepare_fold(job, corpus.folds(stem)[0], 0, corpus, tmp_path / "out", D.PlanState(), _ok_guard, _ok_inner)
    stub = StubRunner()
    led = _run(job, corpus, tmp_path / "out", stub)
    assert stub.point_calls == 0 and "V6_TARGET_ROWS" in led["jobs"][job.key]["errors"][0]["error"]
    attrs = pd.DataFrame({"registered_half_V5": ["S"]}, index=pd.Index(["a"]))
    with pytest.raises(AssertionError, match="V6_TARGET_ROWS"):
        D.scoring_frame(pd.DataFrame({"row_id": ["a"], "fold_id": ["f"], "mean_logD": [0.0]}), attrs, design="V5",
                        v6_mask=pd.Series(True, index=attrs.index), what="t")


def test_provenance_columns_never_become_features():
    D.assert_no_provenance_features(["metal__Z", "condition__log10_acid_M"], "ok")
    for bad in (["canonical_measurement_id"], ["extractant__g19_publication_id"], ["pub_group"], ["doi_all"],
                ["metal__group_cross_publication_copy"]):
        with pytest.raises((AssertionError, ValueError)):
            D.assert_no_provenance_features(bad, "bad")


# --------------------------------------------------------------------------------------------- #
# the plan (section 7 order), filters, digests
# --------------------------------------------------------------------------------------------- #

def test_plan_order_and_conditional_jobs():
    """POST-HOC addendum 1: discovery seed 104729 only, three simultaneous inner folds everywhere (``inner_mode``
    ``full``), the V5 strict / HNO3-only refits only, V5-PAIR for M2 with its M1 prerequisite, and a marker naming
    everything that is not run."""
    jobs = D.enumerate_plan(D.PlanState())
    kinds = [j.kind for j in jobs]
    assert kinds[0] == "safeguard" and jobs[-1].kind == "marker" and jobs[-1].message == D.NOT_IMPLEMENTED
    first_heavy = next(i for i, j in enumerate(jobs) if j.kind == "fit" and j.arm in D.HEAVY_ARMS)
    assert all(j.arm == "B6" for j in jobs[:first_heavy] if j.kind == "fit")
    b6 = [j for j in jobs if j.arm == "B6"]
    assert all(j.writes == ("B6", "B6r0") for j in b6) and b6[0].seed == D.PRIMARY_SEED
    # heavy-arm V5 / V1 jobs wait for the B6 checks; V2 does not
    assert not any(j.arm in D.HEAVY_ARMS and j.design in ("V5", "V1") for j in jobs)
    assert any(j.kind == "marker" and "batched-vs-exact" in j.message for j in jobs)
    passed = D.enumerate_plan(D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed"))
    fits = [j for j in passed if j.kind == "fit"]
    # item 3: every fitted arm runs discovery seed 104729 only -- no learned-arm job on any other seed, anywhere
    assert {j.seed for j in fits} == set(D.PLAN_SEEDS) == {D.PRIMARY_SEED}
    assert all(j.fold_seed in (None, D.PRIMARY_SEED) for j in fits)
    assert not any(j.kind == "comparator_intervals" for j in passed)
    assert any(j.kind == "marker" and j.arm == "comparator_intervals" for j in passed)
    # items 1-2: three inner folds on every job, no "first inner fold" reading left
    assert {j.inner_mode for j in fits} == {"full"}
    # item 4: only the V5 strict and HNO3-only refits are scheduled; no dropped sensitivity job is present
    assert {j.variant for j in fits if j.variant in D.V5_REFIT_VARIANTS} == set(D.LEARNED_REFIT_VARIANTS)
    assert set(D.LEARNED_REFIT_VARIANTS) == {"strict", "hno3_only"}
    assert not any(j.drop_sr for j in fits)
    assert not any(j.design == "V5P" for j in passed)
    assert not any(j.design == "V2" and j.variant == "state" for j in fits)
    assert not any(j.variant in ("loose", "cell_only", "parent_structure") for j in fits)
    # ... and each of them is named by a marker, never silently absent
    msgs = " ".join(j.message for j in passed if j.kind == "marker")
    for token in ("loose", "cell-only", "parent-structure", "Sr(III)-dropped", "V5-P", "130363", "V1 / V2 refit"):
        assert token in msgs
    # item 5: V5-PAIR carries M2 (with its M1 prerequisite); B3x / B3i are the scorer's closed-form re-fit
    pair = [j for j in fits if j.design == "V5PAIR"]
    assert {j.arm for j in pair} == {"M1", "M2"} and {j.stage for j in pair} == {D.STAGES["s1c"]}
    assert all(j.scheme == "batched" for j in pair)
    assert any("B3x and B3i are re-fitted" in j.message for j in passed if j.kind == "marker")
    # reading 6(a): the arm order inside each pass, and the pass order
    order = [j.arm for j in fits]
    stage_of = {a: min(i for i, x in enumerate(order) if x == a) for a in ("B6", "B5", "FLAT_CAT", "B8", "M1", "M2")}
    assert stage_of["B6"] < stage_of["B5"] < stage_of["FLAT_CAT"] < stage_of["B8"] < stage_of["M1"] < stage_of["M2"]
    pos = {j.key: i for i, j in enumerate(fits)}
    m2_h1 = pos["fit:M2:V5__primary_batched:s104729"]
    assert all(pos[j.key] < m2_h1 for j in fits if j.arm == "B6" and j.variant not in D.LEARNED_REFIT_VARIANTS)
    assert all(pos[j.key] > m2_h1 for j in fits if j.design == "V5PAIR")
    # reading 6(a) third pass: every strict / HNO3-only refit, B6's included, comes after the main designs and sits in
    # the refit stage, B6 first within it (task X finding V-F04)
    refits = [j for j in fits if j.variant in D.LEARNED_REFIT_VARIANTS]
    assert all(pos[j.key] > m2_h1 for j in refits) and {j.stage for j in refits} == {D.STAGES["refit"]}
    assert {j.arm for j in refits} == {"B6", "B5", "FLAT_CAT", "M1", "M2"}
    assert min(pos[j.key] for j in refits if j.arm == "B6") < min(pos[j.key] for j in refits if j.arm != "B6")
    assert not any(j.variant in D.LEARNED_REFIT_VARIANTS for j in fits if j.stage == D.STAGES["b6"])
    assert {j.stage for j in passed if j.arm == "not_run:B6_refit_sensitivities"} == {D.STAGES["refit"]}
    stages = sorted({j.stage for j in passed})
    assert stages == sorted(stages) and stages[0] == D.STAGES["safeguard"] and stages[-1] == D.STAGES["not_implemented"]
    assert D.STAGES["s1c"] == "07_s1c_v5pair" and not any(s.endswith("other_seeds") for s in stages)
    m2 = [j for j in fits if j.arm == "M2"]
    m1_keys = {(j.design_dir, j.seed) for j in fits if j.arm == "M1"}
    assert all((j.design_dir, j.seed) in m1_keys for j in m2)     # M2 always has its per-fold M1 values
    assert len({j.key for j in passed}) == len(passed)
    # section 11 specs follow addendum 1 item 3 (task X finding V-F06): seed 104729 only; the sealed enumeration on request
    assert len(D.h3_job_specs()) == 3 * 4 * 3 * len(D.PLAN_SEEDS) == 36 and {j.kind for j in D.h3_job_specs()} == {"h3_spec"}
    assert {j.seed for j in D.h3_job_specs()} == {D.PRIMARY_SEED}
    assert len(D.h3_job_specs(D.DISCOVERY_SEEDS)) == 3 * 4 * 3 * 5
    assert {j.seed for j in D.enumerate_plan(passed and D.PlanState(), include_h3_specs=True)} <= {None, D.PRIMARY_SEED}
    assert not any(D.demotable(j) for j in passed) and D.demotable(D.JobSpec(kind="fit", arm="M3", design="V5", seed=1))
    # conditional: the re-coloured B6 batches (seed 104729 only now) and a V5 freezing candidate
    rec = D.enumerate_plan(D.PlanState(v5_batched_check="recolour"))
    assert sum(j.scheme == "batched_max4" for j in rec) == 1
    cand = D.enumerate_plan(D.PlanState(v5_batched_check="passed_after_recolour", v1_tenfold_check="failed",
                                        freezing_candidates=[{"contrast": "M2 vs B8@V5", "arms": ["M2", "B8"],
                                                              "passed_items_1_5_v5_primary": True}]))
    cjobs = [j for j in cand if j.stage == D.STAGES["candidates"]]
    # the candidate's own arm gets the two refits; M2's and M1's are already scheduled in the refit pass (deduped)
    assert {j.arm for j in cjobs if j.kind == "fit"} == {"B8"}
    assert all(j.variant in D.LEARNED_REFIT_VARIANTS for j in cjobs if j.kind == "fit")
    assert not any(j.design == "V5P" for j in cjobs) and any("V5-P" in j.message for j in cjobs if j.kind == "marker")
    assert all(j.scheme == "batched_max4" for j in cand if j.kind == "fit" and j.arm in D.HEAVY_ARMS and j.design == "V5")
    assert all(j.scheme == "exact" for j in cand if j.kind == "fit" and j.arm in D.HEAVY_ARMS and j.design == "V1")


def test_dry_run_writes_the_addendum_plan(tmp_path):
    """``--dry-run`` enumerates the addendum-1 plan, prices it with whichever benchmark exists and writes the plan to
    ``evaluation/discovery/benchmark/plan_addendum1.txt`` -- the record of what the run will and will not do.  It fits
    nothing (the registered fold files are only counted)."""
    out = tmp_path / "out"

    def digests():
        return {"footer": D.REGISTERED_PREREG_SHA256, "recomputed": D.REGISTERED_PREREG_SHA256,
                "digest_file": D.REGISTERED_PREREG_SHA256, "addenda_sha256": D.REGISTERED_ADDENDA_SHA256, "n_addenda": 1}
    assert RD.main(["--dry-run", "--out-root", str(out)], check=lambda: 0, digests=digests) == 0
    text = (D.discovery_root(out) / "benchmark" / "plan_addendum1.txt").read_text(encoding="utf-8")
    assert f"seeds [{D.PRIMARY_SEED}]" in text and "markers (what is NOT run, named)" in text
    assert "fit:B6:V5__primary_exact:s104729" in text and D.NOT_IMPLEMENTED in text
    # no learned-arm entry on another seed anywhere in the record, the H3 specs included (task X finding V-F06)
    assert "11_h3_specs" in text and "h3_spec:B6:WITH" in text
    assert not any(f"s{s}" in text for s in D.DISCOVERY_SEEDS if s != D.PRIMARY_SEED)
    jobs_part = text.split("11_h3_specs")[0]
    assert "addendum 1 item 3" in jobs_part and "strict_setting" in jobs_part
    assert not (D.discovery_root(out) / "B6").exists()   # nothing was fitted


def test_v1_and_v2_freezing_candidates_get_markers_not_refit_jobs():
    """Addendum 1 item 4: a learned arm runs no V1 / V2 refit sensitivity, not even as a freezing candidate; the plan
    names every one of them as a marker (task X finding V-06 kept: never a silent absence)."""
    st = D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed",
                     freezing_candidates=[{"contrast": "M2 vs B3@V1", "arms": ["M2", "B3"], "design": "V1"},
                                          {"contrast": "B5 vs B3i@V2", "arms": ["B5", "B3i"]}])
    jobs = D.enumerate_plan(st)
    cand = [j for j in jobs if j.stage == D.STAGES["candidates"]]
    assert not [j for j in cand if j.kind == "fit"]
    marks = [j.message for j in cand if j.kind == "marker"]
    for name in ("sr_iii_dropped_training", "near_duplicate_key_groups_value_blind", "compilation_doi_groups",
                 "state_level_hiding"):
        assert any(name in m for m in marks)
    assert all("NOT run" in m for m in marks) and any("M1" in m for m in marks)
    assert D._design_of_contrast_key("M1 vs M0@V2#ladder") == "V2"
    # a V5-PAIR endpoint is the one learned V5-PAIR run a freezing candidate still gets (item 5)
    pair_st = D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed",
                          freezing_candidates=[{"contrast": "M2 vs FLAT@V5-PAIR", "arms": ["M2", "FLAT"],
                                                "design": "V5-PAIR"}])
    pair = [j for j in D.enumerate_plan(pair_st) if j.kind == "fit" and j.design == "V5PAIR"]
    assert {j.arm for j in pair} == {"M1", "M2"} and all(j.seed == D.PRIMARY_SEED for j in pair)


def test_filter_jobs_and_labels():
    jobs = D.enumerate_plan(D.PlanState(v5_batched_check="passed", v1_tenfold_check="passed"))
    only_b6 = [j for j in D.filter_jobs(jobs, "B6r0") if j.kind != "marker"]
    assert only_b6 and {j.arm for j in only_b6} == {"B6"}
    m2v5 = [j for j in D.filter_jobs(jobs, "M2:V5") if j.kind != "marker"]
    assert m2v5 and {(j.arm, j.design) for j in m2v5} == {("M2", "V5")}
    v1 = [j for j in D.filter_jobs(jobs, "M0,V1") if j.kind != "marker"]
    assert {(j.arm, j.design) for j in v1} == {("B5", "V1")}
    assert {j.kind for j in D.filter_jobs(jobs, "safeguard") if j.kind != "marker"} == {"safeguard"}
    j = D.JobSpec(kind="fit", arm="B5", design="V5", variant="primary", scheme="batched", seed=130363, fold_seed=130363,
                  drop_sr=True)
    assert j.variant_label == "primary_batched_sr_iii_dropped" and j.key == "fit:B5:V5__primary_batched_sr_iii_dropped:s130363"
    assert D.safe_fold_name("Nd(III)__abc") == "Nd(III)__abc" and "/" not in D.safe_fold_name("a/b x c")
    assert D.guard_mode_source("V5__loose__exact") == "V5__primary__exact"
    assert D.guard_mode_source("V5__cell_only__exact") == "V5__cell_only__batched"
    assert D.guard_mode_source("V5__primary__batched_max4") == "V5__primary__batched"
    assert D.guard_mode_source("V1__copy__exact") == "V1__copy__grouped10"
    folds = [FI.make_fold(design="V5", variant="p", scheme="batched", fold_id=f"s{s}_{h}_b{k:03d}", half=h, seed=s,
                          hidden=[f"r{s}{h}{k}"], scored=[f"r{s}{h}{k}"], unit_type="cell", units=["u"])
             for s in (1, 2) for h in ("S", "C") for k in range(3)]
    ords = D.fold_ordinals(folds)
    assert ords["s1_S_b002"] == 2 and ords["s1_C_b000"] == 3 and ords["s2_S_b000"] == 6
    js = D.JobSpec(kind="fit", arm="B5", design="V5", variant="p", scheme="batched", seed=2, fold_seed=2)
    assert [(f.fold_id, k) for f, k in D.job_folds(js, folds)] == [("s2_S_b000", 6), ("s2_S_b001", 7), ("s2_S_b002", 8)]


def test_model_fold_numbers_make_the_discovery_seed_drive_initialisation():
    """Task X finding V-02: on a seed-free fold file (V2) every discovery seed gets distinct model seeds; seed 104729
    keeps the file position; a seeded file numbers each seed's folds after the earlier seeds'."""
    free = [FI.make_fold(design="V2", variant="element", scheme="exact", fold_id=st, half="S", seed=None, hidden=[st],
                         scored=[st], unit_type="metal_state", units=[st]) for st in ("Ce(III)", "Nd(III)", "Pr(III)")]
    per_seed = {s: D.fold_ordinals(free, s) for s in D.DISCOVERY_SEEDS}
    assert per_seed[D.PRIMARY_SEED] == {"Ce(III)": 0, "Nd(III)": 1, "Pr(III)": 2}
    assert per_seed[130363] == {"Ce(III)": 3, "Nd(III)": 4, "Pr(III)": 5}
    all_numbers = [k for d in per_seed.values() for k in d.values()]
    assert len(set(all_numbers)) == 15
    jobs = {s: D.JobSpec(kind="fit", arm="B8", design="V2", variant="element", scheme="exact", seed=s)
            for s in D.DISCOVERY_SEEDS}
    seeds_of = {s: {k for _, k in D.job_folds(j, free)} for s, j in jobs.items()}
    assert all(not (seeds_of[a] & seeds_of[b]) for a in D.DISCOVERY_SEEDS for b in D.DISCOVERY_SEEDS if a != b)
    seeded = [FI.make_fold(design="V5", variant="primary", scheme="batched", fold_id=f"s{s}_S_b{k:03d}", half="S", seed=s,
                           hidden=[f"r{s}{k}"], scored=[f"r{s}{k}"], unit_type="cell", units=["u"])
              for s in D.DISCOVERY_SEEDS for k in range(4)]
    assert D.fold_ordinals(seeded, D.PRIMARY_SEED) == {f"s104729_S_b{k:03d}": k for k in range(4)}
    assert D.fold_ordinals(seeded, 155921)["s155921_S_b000"] == 8
    with pytest.raises(ValueError):
        D.fold_ordinals(free + seeded[:1])


def test_fold_digest_changes_with_code_job_and_fold():
    df = _frame()
    f1, f2 = _cell_fold(df, "Nd(III)", "S1"), _cell_fold(df, "Eu(III)", "S2")
    j = _job()
    d = D.fold_digest(j, f1, "c")
    assert d == D.fold_digest(j, f1, "c")
    assert d != D.fold_digest(j, f1, "c2") and d != D.fold_digest(j, f2, "c")
    assert d != D.fold_digest(_job(seed=130363), f1, "c") and d != D.fold_digest(j, f1, "c", {"guard_mode": "x"})
    # labels never change the digest; the model fold number and design hash (runner extras) do
    from dataclasses import replace as _replace
    assert D.fold_digest(_replace(j, registered=False, condition="freezing_candidate", stage="z"), f1, "c") == d
    state, stub = D.PlanState(), StubRunner()
    base = RD.fold_digest(j, f1, "c", state, stub, Path("."), ordinal=0, design_hash="h")
    assert base != RD.fold_digest(j, f1, "c", state, stub, Path("."), ordinal=1, design_hash="h")
    assert base != RD.fold_digest(j, f1, "c", state, stub, Path("."), ordinal=0, design_hash="h2")
    assert D.job_from_record({"job": j.record()}) == j
    assert any(str(paths.rel(f)).endswith("gen19ct/evaluation/support.py") for f in RD.CODE_FILES)


def test_resume_digest_covers_the_addenda_digest_and_the_inner_design(monkeypatch):
    """Task X findings V-F01 / VL-A1-01: a record fitted under another addendum text, or under another inner design
    (``discovery.INNER_N_FOLDS`` / ``INNER_MAX_CELLS_PER_FOLD`` are module constants, not digested code), is stale."""
    from gen19ct.models import inner_design as ID
    df = _frame()
    f1 = _cell_fold(df, "Nd(III)", "S1")
    state, stub = D.PlanState(), StubRunner()
    v5 = _job()
    v1 = D.JobSpec(kind="fit", arm="B5", design="V1", variant="copy", scheme="exact", seed=D.PRIMARY_SEED, writes=("B5",))
    kw = dict(ordinal=0, design_hash="h")
    base5, base1 = (RD.fold_digest(v5, f1, "c", state, stub, Path("."), **kw),
                    RD.fold_digest(v1, f1, "c", state, stub, Path("."), **kw))
    monkeypatch.setattr(D, "INNER_MAX_CELLS_PER_FOLD", 5)
    assert RD.fold_digest(v5, f1, "c", state, stub, Path("."), **kw) != base5
    assert RD.fold_digest(v1, f1, "c", state, stub, Path("."), **kw) == base1        # V1 has no V5 inner design
    monkeypatch.setattr(D, "INNER_MAX_CELLS_PER_FOLD", 30)
    monkeypatch.setattr(D, "INNER_N_FOLDS", 2)
    assert RD.fold_digest(v5, f1, "c", state, stub, Path("."), **kw) != base5
    monkeypatch.setattr(D, "INNER_N_FOLDS", 3)
    assert RD.fold_digest(v5, f1, "c", state, stub, Path("."), **kw) == base5
    monkeypatch.setattr(D, "REGISTERED_ADDENDA_SHA256", "0" * 64)
    assert RD.fold_digest(v5, f1, "c", state, stub, Path("."), **kw) != base5
    assert RD.fold_digest(v1, f1, "c", state, stub, Path("."), **kw) != base1
    monkeypatch.undo()
    # the digested signature is the resolved design's own description, for every V5 variant and the V5-P / V5-PAIR jobs
    frame = pd.DataFrame({"acid_primary": ["HNO3"]}, index=["r0"])
    for variant in CH.VARIANTS:
        job = replace(v5, variant=variant)
        sig = RD.inner_design_signature(job)
        desc = ID.SimultaneousInnerCells.for_variant(variant, frame=frame).describe()
        assert sig["module"] == D.INNER_DESIGN_MODULE and sig["max_cells"] == D.INNER_MAX_CELLS_PER_FOLD == 30
        assert all(desc[k] == sig[k] for k in sig if k != "module"), (variant, sig, desc)
    for design, variant in (("V5P", "base"), ("V5PAIR", "primary")):
        sig = RD.inner_design_signature(replace(v5, design=design, variant=variant))
        assert sig["thresholds"] == "k10_p1_m3" and sig["medium"] == "all" and sig["component_aware"]
    assert RD.inner_design_signature(v1) is None
    assert RD.inner_design_signature(D.JobSpec(kind="safeguard", arm="B3x", stem_override="V5__primary__exact")) is None


# --------------------------------------------------------------------------------------------- #
# inner splits for calibration
# --------------------------------------------------------------------------------------------- #

def test_tuning_split_calibration_converts_and_checks_splits():
    df = _frame()
    t = I.RowTable(df)
    mask = np.ones(t.n, dtype=bool)
    mask[:6] = False
    tr_lab = t.index[mask]

    def split(name, fold, hide):
        hid = pd.Index(tr_lab[hide])
        return SimpleNamespace(split_id=name, inner_fold=fold, train_index=tr_lab.difference(hid, sort=False),
                               hidden_index=hid, val_index=hid[:3], val_units=np.array(["u"] * 3, dtype=object),
                               guard_test_index=hid)

    class Design:
        def __init__(self, sp):
            self.sp = sp

        def splits(self, rows, context):
            assert rows.index.equals(tr_lab) and np.array_equal(rows[I.TARGET_COL].to_numpy(), t.y[mask])
            return self.sp
    sps = [split("a", 1, slice(0, 6)), split("b", 0, slice(6, 12)), split("c", 0, slice(12, 18))]
    full = D.TuningSplitCalibration(Design(sps), df, "full").splits(t, mask, I.FitContext(seed=1))
    assert [s.unit for s in full] == ["a", "b", "c"]
    s0 = full[0]
    assert not (s0.train_mask & ~mask).any() and not s0.train_mask[s0.cal_positions].any()
    assert np.array_equal(np.sort(t.index[s0.hidden_positions]), np.sort(sps[0].hidden_index))
    assert s0.certificate is not None and np.isin(s0.cal_positions, s0.certificate[1]).all()
    first = D.TuningSplitCalibration(Design(sps), df, "first").splits(t, mask, I.FitContext(seed=1))
    assert [s.unit for s in first] == ["b", "c"]
    assert all(s.row_units is not None and list(s.row_units) == ["u"] * len(s.cal_positions) for s in full)
    only1 = D.TuningSplitCalibration(Design(sps), df, "full", inner_folds=[1]).splits(t, mask, I.FitContext(seed=1))
    assert [s.unit for s in only1] == ["a"]
    bad = split("bad", 0, slice(0, 6))
    bad.train_index = bad.train_index.append(bad.val_index[:1])
    with pytest.raises(AssertionError, match="partition"):
        D.TuningSplitCalibration(Design([bad]), df, "full").splits(t, mask, I.FitContext(seed=1))
    wrapped = D.FirstInnerFold(SimpleNamespace(splits=lambda *a: [SimpleNamespace(fold=2), SimpleNamespace(fold=1)]))
    assert [s.fold for s in wrapped.splits(None, None, None)] == [1]


def test_v5_exact_inner_cells_medium_and_all_media_equivalence():
    rng = np.random.default_rng(5)
    recs = []
    for i, st in enumerate(("La(III)", "Nd(III)", "Eu(III)", "Gd(III)", "Tb(III)", "Am(III)")):
        for j, sy in enumerate(("S1", "S2", "S3", "S4", "S5")):
            for k in range(12):
                recs.append({FI.ROW_ID: f"R:{len(recs):05d}", SG.METAL_COL: st, SG.ELEMENT_COL: st.split("(")[0],
                             SG.SYSTEM_COL: sy, SG.PUB_COL: f"pub_{(i + j + k) % 7}", FI.GROUP_COL: f"g{(i + j) % 7}",
                             I.PUB_GROUP_COL: f"g{(i + j) % 7}",
                             "acid_primary": "HCl" if (k < 5 and (i + j) % 2 == 0) else "HNO3",
                             SG.ACID_ANION_COL: "nitrate", SG.LOG_ACID_COL: 0.0, SG.LOG_EXT_COL: -1.0,
                             I.TARGET_COL: float(rng.normal())})
    df = pd.DataFrame(recs, index=[f"L{i}" for i in range(len(recs))])
    t = I.RowTable(df)
    mask = np.ones(t.n, dtype=bool)
    mask[:12] = False
    ctx = I.FitContext(table=t, v6_mask=pd.Series(False, index=df.index), seed=104729,
                       isolation_check=lambda tr, te: {"ok": True})
    kw = dict(k=10, p=1, m=3, n_folds=3, max_cells_per_fold=30)
    parent = I.InnerCellCalibration(**kw).splits(t, mask, ctx)
    same = D.V5ExactInnerCells(**kw).splits(t, mask, ctx)
    assert [(s.fold, s.unit) for s in parent] == [(s.fold, s.unit) for s in same] and parent
    hno3 = D.V5ExactInnerCells(**kw, medium="HNO3", frame=df).splits(t, mask, ctx)
    med = df["acid_primary"].to_numpy() == "HNO3"
    assert hno3 and all(med[s.cal_positions].all() for s in hno3)
    # cells with fewer than 10 HNO3 rows are no inner cell under the HNO3 medium, although the parent's cache holds them
    n_hno3 = df[med].groupby([SG.METAL_COL, SG.SYSTEM_COL]).size()
    assert all(n_hno3.loc[s.unit] >= 10 for s in hno3)
    assert {s.unit for s in hno3} < {s.unit for s in parent}


def test_v5_tuning_and_calibration_use_the_simultaneous_inner_design(tmp_path):
    """Addendum 1 items 1-2: every V5 design and variant tunes and calibrates on
    ``inner_design.SimultaneousInnerCells`` -- one split per inner fold, drawn ONCE and shared by the tuner and the
    conformal calibration -- while V1 and V2 keep the registered inner designs."""
    from gen19ct.models import inner_design as ID

    corpus = _corpus(tmp_path)
    df = corpus.frame
    _write([_cell_fold(df, "Nd(III)", "S1", variant="loose"), _cell_fold(df, "Eu(III)", "S2", variant="loose")],
           corpus.folds_dir)
    job = _job("loose")
    fold = corpus.folds(job.stem)[0]
    fc, _ = RD.prepare_fold(job, fold, 0, corpus, tmp_path / "out", D.PlanState(), _ok_guard, _ok_inner)
    design = RD.simultaneous_inner_design(job, corpus)
    assert isinstance(design, ID.SimultaneousInnerCells) and design.name == D.INNER_DESIGN_NAME == ID.NAME
    thr = CH.VARIANTS["loose"].thresholds                      # the job's V5 variant, not the primary thresholds
    assert (design.k, design.p, design.m) == (thr.k, thr.p, thr.m)
    assert design.n_folds == D.INNER_N_FOLDS == 3 and design.max_cells == D.INNER_MAX_CELLS_PER_FOLD == 30
    assert design.medium == CH.VARIANTS["loose"].medium
    used, splits = RD.inner_splits_v5(fc)
    assert splits and len(splits) <= D.INNER_N_FOLDS
    assert [int(s.fold) for s in splits] == sorted({int(s.fold) for s in splits})
    for sp in splits:
        assert not (sp.train_mask & ~fc.mask).any() and not sp.train_mask[sp.cal_positions].any()
        assert len(sp.row_units) == len(sp.cal_positions) and len(set(map(str, sp.row_units))) >= 1
        assert np.isin(sp.cal_positions, sp.hidden_positions).all() and sp.certificate is not None
    assert RD.inner_splits_v5(fc)[0] is used                   # one draw, shared by tuning and calibration
    # the conformal calibration takes those splits, one inner fold at a time, cross-fitted
    full, avail = RD._calibration_splits(fc)
    assert isinstance(full, D.FixedInnerSplits) and avail == [int(s.fold) for s in splits]
    one = RD._restricted(full, [avail[0]])
    assert isinstance(one, D.InnerFoldSubset)
    assert [int(s.fold) for s in one.splits(corpus.table, fc.mask, fc.ctx)] == [avail[0]]
    plan = D.calibration_folds(avail, fc.job.inner_mode, tuned=True)
    assert plan["cross_fit"] is True and plan["calibration_folds"] == plan["tuning_folds"] == avail
    assert set(ID.cross_fit_plan(avail)) == set(avail)
    # the tuner is handed the same design as TuningSplits, and the two draws agree cell for cell
    rows = corpus.frame.loc[corpus.table.index[fc.mask]]
    tuner_design = RD.inner_design_object(job, corpus)
    assert type(tuner_design).__name__ == "V5SimultaneousTuning"
    tsp = tuner_design.splits(rows, fc.ctx)
    assert [int(t.inner_fold) for t in tsp] == [int(s.fold) for s in splits]
    for t, s in zip(tsp, splits):
        assert set(t.val_index) == set(corpus.table.index[s.cal_positions])
        assert set(map(str, t.val_units)) == set(map(str, s.row_units))
    # the neural arm sees the same splits as label indices
    vs = RD.validation_splits(fc)
    assert [v.inner_fold for v in vs] == [int(s.fold) for s in splits]
    assert all(len(v.valid_units) == len(v.valid_index) for v in vs)
    assert all(v.valid_index.isin(v.hidden_index).all() and not v.train_index.isin(v.hidden_index).any() for v in vs)
    assert all("verify_inner_splits" in str(v.guard) for v in vs)
    # B6 tunes on the same design, and the record says what the design did
    assert isinstance(RD.B6Runner().splitter(fc), D.FixedInnerSplits)
    rec = RD.inner_design_record(fc)
    assert rec["design"] == D.INNER_DESIGN_NAME and rec["n_inner_folds"] == len(splits)
    assert len(rec["splits"]) == len(splits) and rec["n_dropped_cells"] >= 0
    assert set(rec["dropped_cells_per_fold"]) == {str(f) for f in avail}
    # V1 and V2 are unchanged: never the simultaneous design
    for name, variant in (("V1", "copy"), ("V2", "element")):
        other = D.JobSpec(kind="fit", arm="B5", design=name, variant=variant, scheme="exact", seed=D.PRIMARY_SEED)
        assert not isinstance(RD.inner_design_object(other, corpus), ID.SimultaneousInnerCells)
        assert not RD.v5_family(other)
    assert RD.v5_family(job) and RD.v5_family(D.JobSpec(kind="fit", arm="M2", design="V5PAIR", seed=1))
    # a learned arm can never fall back to the per-cell design
    with pytest.raises(ValueError, match="per-cell"):
        ID.assert_learned_arm_design(I.InnerCellCalibration(), "M2")


def test_verify_inner_splits_refuses_a_leaking_or_unscorable_split(tmp_path):
    """The addendum's splits pass the same guards as the registered inner designs before any arm is fitted."""
    corpus = _corpus(tmp_path)
    _write([_cell_fold(corpus.frame, "Nd(III)", "S1", variant="loose")], corpus.folds_dir)
    job = _job("loose")
    fc, _ = RD.prepare_fold(job, corpus.folds(job.stem)[0], 0, corpus, tmp_path / "out", D.PlanState(), _ok_guard,
                            _ok_inner)
    _, splits = RD.inner_splits_v5(fc)
    good = splits[0]
    RD.verify_inner_splits(fc, [good])
    leak = replace(good, train_mask=good.train_mask | np.isin(np.arange(corpus.table.n), good.cal_positions))
    with pytest.raises(AssertionError, match="partition"):
        RD.verify_inner_splits(fc, [leak])
    with pytest.raises(AssertionError, match="two inner splits"):
        RD.verify_inner_splits(fc, [good, good])
    with pytest.raises(AssertionError, match="row_units"):
        RD.verify_inner_splits(fc, [replace(good, row_units=None)])
    with pytest.raises(ValueError, match="no validation split"):
        RD.verify_inner_splits(fc, [])
    # an inner split that fails the isolation check is refused
    fc.ctx.guard_cache.clear()
    fc.ctx.isolation_check = lambda tr, te: {"ok": False}
    with pytest.raises(AssertionError, match="isolation check"):
        RD.verify_inner_splits(fc, [good])


# --------------------------------------------------------------------------------------------- #
# R19, stop rule, ladder, BH on synthetic predictions
# --------------------------------------------------------------------------------------------- #

def _v5_rows(noise: float, seed: int, n_systems: int = 18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recs = []
    for s in range(n_systems):
        for st in ("Nd(III)", "Eu(III)"):
            for k in range(5):
                y = float(np.random.default_rng(1000 * s + 7 * k + len(st)).normal())
                recs.append({"row_id": f"r{s}_{st}_{k}", EM.METAL_STATE_COL: st, EM.SYSTEM_COL: f"SYS{s:02d}",
                             EM.PUB_GROUP_COL: f"g{s % 6}", EM.Y_COL: y, "fold_id": f"{st}__{s}",
                             EM.PRED_COL: y + float(rng.normal(0, noise)), "dga_stratum": "non_DGA" if s % 2 else "diglycolamide",
                             "acid_grid_flag": k == 0, "censoring_candidate": k == 1,
                             "wildcard_copy_partner_in_training": k == 2})
    fr = pd.DataFrame(recs).set_index("row_id", drop=False)
    fr.index.name = None
    return fr


def _contrast(cand_noise: float, comp_noise: float, *, seeds=D.DISCOVERY_SEEDS, sens_value: float | str = 0.2,
              margin: float = 0.1, name: str = "M2 vs B3i", learned: bool = False, reduced=None):
    v6 = pd.Series(False, index=_v5_rows(1, 0).index)
    per_seed = {}
    for s in seeds:
        cand, comp = _v5_rows(cand_noise, s), _v5_rows(comp_noise, 99)
        per_seed[s] = D.paired_units(cand, comp, "V5", candidate="M2", comparator="B3i", v6_mask=v6)
    primary = per_seed[D.PRIMARY_SEED]
    sens = {n: sens_value for n in ET.REGISTERED_SENSITIVITIES["V5"]}
    for n in D.SCORING_FILTER_SENSITIVITIES:
        c, k = D.filtered_pair(_v5_rows(cand_noise, D.PRIMARY_SEED), _v5_rows(comp_noise, 99), n)
        sens[n] = D.paired_units(c, k, "V5", candidate="M2", comparator="B3i", v6_mask=v6.loc[c.index]).delta
    return D.evaluate_contrast(name=name, family="primary", design="V5", primary=primary, margin=margin,
                               seed_deltas={s: per_seed[s].delta for s in seeds}, sensitivities=sens,
                               learned=learned, reduced_sensitivities=reduced)


def test_r19_and_stop_rule_wiring_on_synthetic_predictions():
    good = _contrast(0.1, 1.5)
    assert good["point"] > 0.5 and good["r19"].verdict == "PASS"
    assert {k: v["verdict"] for k, v in good["scopes"].items()} == dict.fromkeys(D.SCOPES, "PASS")
    assert set(good["bootstraps"]) == {"system", "publication_group"}
    assert all(b.n_resamples == ET.N_RESAMPLES and b.seed == ET.BOOTSTRAP_SEED for b in good["bootstraps"].values())
    bad = _contrast(1.5, 0.1)
    assert bad["point"] < 0 and bad["r19"].verdict == "FAIL" and bad["scopes"]["stop_rule"]["verdict"] == "FAIL"
    # stop rule: stop only when neither M2 nor B6 passes; pending while one is missing
    assert D.stop_rule(good, bad)["stop"] is False and D.stop_rule(bad, good)["stop"] is False
    s = D.stop_rule(bad, bad)
    assert s["stop"] is True and len(s["consequences"]) == 3 and s["evaluated_on"]["items"] == [1, 2, 3, 5]
    assert D.stop_rule(None, bad)["stop"] is None and D.stop_rule(None, bad)["pending"] == ["M2"]
    assert D.stop_rule(good, None)["stop"] is False
    # the margin binds item 1 only
    big = _contrast(0.1, 1.5, margin=5.0)
    assert big["r19"].item(1)["status"] == "FAIL" and big["scopes"]["stop_rule"]["verdict"] == "FAIL"
    # fewer seeds: item 4 NOT_RUN -> UNDECIDED (never PASS), the seed-104729 scopes still PASS
    few = _contrast(0.1, 1.5, seeds=(D.PRIMARY_SEED, 130363))
    assert few["r19"].item(4)["status"] == "NOT_RUN" and few["r19"].verdict == "UNDECIDED"
    assert few["scopes"]["stop_rule"]["verdict"] == "PASS" and few["scopes"]["items_1_5"]["verdict"] == "UNDECIDED"
    # a refit sensitivity not run: full R19 UNDECIDED, ladder scope (scoring filters only) PASS
    unt = _contrast(0.1, 1.5, sens_value=ET.UNTESTABLE)
    assert unt["r19"].verdict == "UNDECIDED" and unt["scopes"]["ladder"]["verdict"] == "PASS"
    assert unt["scopes"]["full"]["verdict"] == "UNDECIDED"
    # a negative scoring-filter sensitivity fails the ladder scope
    neg = dict(_contrast(0.1, 1.5))
    sens = dict(neg["sensitivities"], censoring_candidates_excluded_scoring=-0.1)
    assert D._scope_verdict({i["item"]: i["status"] for i in neg["r19"].items}, D.SCOPES["ladder"], sens,
                            "V5")["verdict"] == "FAIL"
    # ladder: kept only with the V5 ladder scope and V1 / V2 non-inferiority
    ni = {"verdict": "PASS", "non_inferior": True, "low_90": -0.01, "high_90": 0.2}
    inf = {"verdict": "UNDECIDED", "non_inferior": False, "low_90": -0.2, "high_90": 0.1}
    assert D.ladder_step("M1", "M0", good, ni, ni)["kept"] is True
    assert D.ladder_step("M1", "M0", good, ni, inf)["kept"] is False
    assert D.ladder_step("M1", "M0", bad, ni, ni)["kept"] is False
    assert D.ladder_step("M1", "M0", good, None, ni)["kept"] is None
    assert D.retained_predecessor({"M0": {"kept": True}, "M1": {"kept": False}}, "M2") == "M0"
    assert D.retained_predecessor({"M0": {"kept": True}, "M1": {"kept": True}}, "M2") == "M1"
    # contrast rows, BH per family, S1(a) / (b) components, freezing screen
    rows = pd.DataFrame(D.contrast_rows(good) + D.contrast_rows(dict(bad, family="exploratory", name="B5 vs B3i")))
    bh = D.apply_bh(rows)
    assert set(bh["bh_family"]) == {"registered", "exploratory"}
    prim = bh[bh["primary_cluster_unit"]]
    assert prim["p_bh"].notna().all() and bh.loc[~bh["primary_cluster_unit"], "p_bh"].isna().all()
    comp = D.s1ab_components({"M2 vs B3i@V5": good})
    assert comp["S1a_M2_vs_B3i"]["r19_full"] == "PASS" and comp["S1b_M2_vs_B0"] is None
    assert "optimistically biased" in comp["label"]
    cands = D.freezing_candidates({"M2 vs B3i@V5": good, "B5 vs B3i@V5": dict(bad, family="exploratory")})
    assert cands == [{"contrast": "M2 vs B3i@V5", "family": "primary", "arms": ["M2", "B3i"], "design": "V5",
                      "passed_items_1_5_v5_primary": True, "batching_label": "", "s1_forced_undecided": False}]
    # task X finding V-05: the failed re-coloured check forces S1 UNDECIDED and labels the heavy-arm V5 contrasts
    failed = D.PlanState(v5_batched_check="failed")
    comp_f = D.s1ab_components({"M2 vs B3i@V5": good}, failed)
    assert comp_f["S1_forced_undecided"] and comp_f["S1a_M2_vs_B3i"]["r19_full"] == "PASS"
    assert comp_f["S1a_M2_vs_B3i"]["reported_verdict"] == "UNDECIDED"
    assert comp_f["S1a_M2_vs_B3i"]["batching_label"] == D.CHECK_FAILED_LABEL
    cf = D.freezing_candidates({"M2 vs B3i@V5": good}, failed)[0]
    assert cf["batching_label"] == "batched (check failed)" and cf["s1_forced_undecided"]
    assert D.heavy_v5_batching_label(["B6", "B3i"], "V5", failed) == ""
    assert not D.s1ab_components({"M2 vs B3i@V5": good}, D.PlanState(v5_batched_check="passed"))["S1_forced_undecided"]
    # task X finding V-09: BH over the full registered family counts the contrasts not run as p = 1
    acc = D.registered_family_accounting(["M2 vs B3i@V5", "S1(c)", "M1 vs M0@V1#ladder"])
    # addendum 1 reading 6(f): m = 60 = the 57 discovery contrasts of section 19 + the 3 confirmation-only S2 contrasts
    # entered as p = 1 (task X finding V-F05); the test asserts the registered number, not the table length
    assert acc["m_full"] == D.REGISTERED_FULL_FAMILY_SIZE == 60 and acc["m_evaluated"] == 1 + 5 + 1
    assert acc["m_discovery"] == len(D.REGISTERED_FAMILY_TABLE) == 57 and len(acc["contrasts"]) == 60
    assert [c["contrast"] for c in acc["contrasts"] if c["family"] == "confirmation"] == list(D.CONFIRMATION_ONLY_CONTRASTS)
    assert all(c["status"] == D.CONFIRMATION_ONLY_STATUS for c in acc["contrasts"] if c["family"] == "confirmation")
    assert "60" in acc["full_family_reading"] and "addendum 2" in acc["full_family_reading"]
    assert {c["family"] for c in acc["contrasts"] if c["status"].startswith("not_run")} >= {"H3", "H5", "H4", "ladder",
                                                                                           "secondary designs", "H1b"}
    tab = D.apply_bh(pd.DataFrame(D.contrast_rows(good)), m_registered_full=acc["m_full"])
    pr = tab[tab["primary_cluster_unit"]]
    assert (pr["p_bh_full_family"] >= pr["p_bh"] - 1e-12).all() and (pr["bh_m"] == 1).all()
    assert set(D.UNCERTAINTY_NOT_RUN) >= {"gaussian_crps", "spearman_abs_error_vs_sd", "knows_when_it_does_not_know"}


def test_r19_item4_not_evaluated_and_the_reduced_sensitivity_set():
    """Addendum 1 items 3-4 inside R19: item 4 NOT_EVALUATED for a learned-arm contrast (so the full verdict is
    UNDECIDED, never PASS), item 6 decided on the reduced set and labelled, the dropped sensitivities named -- and the
    stop-rule, ladder and freezing scopes, which contain neither, unchanged."""
    reduced = [n for n in ET.REGISTERED_SENSITIVITIES["V5"] if n not in D.LEARNED_REFITS_NOT_RUN["V5"]]
    assert set(reduced) == {"strict_setting", "HNO3_only_cells", "non_DGA_stratum", "acid_grid_rows_excluded",
                            "censoring_candidates_excluded_scoring", ET.WILDCARD_COPY_SENSITIVITY}
    sens = {**{n: 0.2 for n in reduced}, **{n: ET.UNTESTABLE for n in D.LEARNED_REFITS_NOT_RUN["V5"]}}
    item, not_run = D.reduced_item6("V5", sens, reduced)
    assert item["status"] == "PASS" and item["sensitivity_set"] == D.ADDENDUM_LABEL
    assert set(not_run) == set(D.LEARNED_REFITS_NOT_RUN["V5"])
    assert "loose_setting" in item["sensitivities_not_run"] and "V5-P" in item["detail"]
    assert D.reduced_item6("V5", {**sens, "strict_setting": -0.1}, reduced)[0]["status"] == "FAIL"
    assert D.reduced_item6("V5", {**sens, "HNO3_only_cells": ET.UNTESTABLE}, reduced)[0]["status"] == "UNTESTABLE"
    # a learned-arm contrast on seed 104729 alone
    lea = _contrast(0.1, 1.5, seeds=D.PLAN_SEEDS, learned=True, reduced=reduced)
    assert lea["r19"].item(4)["status"] == D.ITEM4_NOT_EVALUATED and "104729 only" in lea["r19"].item(4)["detail"]
    assert lea["r19"].item(6)["status"] == "PASS" and lea["r19"].verdict == "UNDECIDED"
    assert lea["sensitivity_set"] == D.ADDENDUM_LABEL and lea["seeds_evaluated"] == list(D.PLAN_SEEDS)
    assert set(lea["sensitivities_not_run"]) == set(D.LEARNED_REFITS_NOT_RUN["V5"])
    for scope in ("stop_rule", "ladder", "freezing_screen"):
        assert lea["scopes"][scope]["verdict"] == "PASS"
    assert lea["scopes"]["items_1_5"]["verdict"] == "UNDECIDED" and lea["scopes"]["full"]["verdict"] == "UNDECIDED"
    assert D.freezing_candidates({"M2 vs B3i@V5": lea})[0]["passed_items_1_5_v5_primary"] is False
    assert D.stop_rule(lea, lea)["stop"] is False                  # the stop rule is unchanged by the addendum
    row = pd.DataFrame(D.contrast_rows(lea))
    assert (row["r19_item4"] == D.ITEM4_NOT_EVALUATED).all() and (row["sensitivity_set"] == D.ADDENDUM_LABEL).all()
    assert "loose_setting" in row["sensitivities_not_run"].iloc[0]
    # a failing contrast still FAILS, and a closed-form contrast keeps the full set and all five seeds
    bad = _contrast(1.5, 0.1, seeds=D.PLAN_SEEDS, learned=True, reduced=reduced)
    assert bad["r19"].verdict == "FAIL" and bad["scopes"]["stop_rule"]["verdict"] == "FAIL"
    det = _contrast(0.1, 1.5)
    assert det["r19"].item(4)["status"] == "PASS" and det["sensitivity_set"] == "registered (full)"
    assert det["r19"].verdict == "PASS" and not det["sensitivities_not_run"]
    assert D.r19_verdict([{"status": "PASS"}, {"status": D.ITEM4_NOT_EVALUATED}]) == "UNDECIDED"
    assert D.r19_verdict([{"status": "FAIL"}, {"status": D.ITEM4_NOT_EVALUATED}]) == "FAIL"
    assert D.r19_verdict([{"status": "PASS"}, {"status": "VACUOUS"}]) == "PASS"


def test_paired_units_refuses_different_rows_and_filters_pair_rows():
    a, b = _v5_rows(0.1, 1), _v5_rows(1.0, 2)
    v6 = pd.Series(False, index=a.index)
    with pytest.raises(ValueError, match="different rows"):
        D.paired_units(a.iloc[1:], b, "V5", candidate="x", comparator="y", v6_mask=v6)
    c, k = D.filtered_pair(a, b.assign(acid_grid_flag=~b["acid_grid_flag"]), "acid_grid_rows_excluded")
    assert c.empty and k.empty                                     # a row dropped for either arm is dropped for both
    c, k = D.filtered_pair(a, b, "non_DGA_stratum")
    assert (c["dga_stratum"] == "non_DGA").all() and c.index.equals(k.index)
    with pytest.raises(ValueError):
        D.apply_scoring_filter(a, "no_such_filter")


def test_b6_check_and_state_machine():
    units = pd.Index([f"u{i}" for i in range(10)])
    ex = {s: pd.Series(0.5, index=units) for s in D.DISCOVERY_SEEDS}
    ok = {s: pd.Series(0.505, index=units) for s in D.DISCOVERY_SEEDS}
    far = {**ok, 130363: pd.Series(0.53, index=units)}
    # addendum 1 item 3: the registered check is decided on the seeds discovery runs -- seed 104729
    first = D.b6_check(ex, ok)
    assert first["passed"] is True and set(first["per_seed"]) == set(D.PLAN_SEEDS) == {D.PRIMARY_SEED}
    assert D.b6_check(ex, far)["passed"] is True                    # a seed that is not run cannot decide it
    assert D.b6_check(ex, {**ok, D.PRIMARY_SEED: pd.Series(0.53, index=units)})["passed"] is False
    assert D.b6_check(ex, {})["passed"] is None and not D.b6_check(ex, {})["complete"]
    # the five-seed reading of the sealed plan is still available to a caller
    assert D.b6_check(ex, far, seeds=D.DISCOVERY_SEEDS)["passed"] is False
    assert D.b6_check(ex, {104729: ok[104729]}, seeds=D.DISCOVERY_SEEDS)["passed"] is None
    assert D.next_v5_check_state("pending", {"passed": True}, None) == "passed"
    assert D.next_v5_check_state("pending", {"passed": False}, None) == "recolour"
    assert D.next_v5_check_state("recolour", {"passed": False}, {"passed": True}) == "passed_after_recolour"
    assert D.next_v5_check_state("recolour", {"passed": False}, {"passed": False}) == "failed"
    assert D.PlanState(v5_batched_check="failed").heavy_v5_label == "batched (check failed)"


# --------------------------------------------------------------------------------------------- #
# cost estimator and budget
# --------------------------------------------------------------------------------------------- #

def test_cost_estimator_arithmetic():
    """Addendum 1: three simultaneous inner splits per outer fold, every job in ``full`` mode, and the per-stage
    cumulative wall clock the operator watches against the 60-hour budget."""
    uc = D.UnitCost("B5", "V5_batched", inner_fit_s=10, cal_fit_s=4, outer_refit_s=30, splits_full=3, splits_first=3,
                    guard_s=0.5)
    assert uc.fold_seconds("full") == {"tuning": 180, "calibration": 12, "outer_refit": 30, "guard": 1.5,
                                       "total": 223.5}
    sealed = D.UnitCost("B5", "V5_batched", inner_fit_s=10, cal_fit_s=4, outer_refit_s=30, splits_full=21,
                        splits_first=8, guard_s=0.5)
    assert sealed.fold_seconds("full")["total"] == 21 * 6 * 10 + 21 * 4 + 30 + 10.5      # the sealed plan's price
    b6 = D.UnitCost("B6", "V5_exact", inner_fit_s=5, cal_fit_s=99, outer_refit_s=2, splits_full=3, splits_first=3,
                    guard_s=0.35)
    assert b6.fold_seconds("full")["total"] == pytest.approx(3 * 5 + 0 + 2 + 3 * 0.35)
    m2 = D.UnitCost("M2", "V1", inner_fit_s=20, cal_fit_s=10, outer_refit_s=40, splits_full=3, splits_first=1, guard_s=1)
    assert m2.fold_seconds("full")["total"] == 3 * 3 * 20 + 3 * 10 + 40 + 3
    j1 = D.JobSpec(kind="fit", arm="B5", design="V5", variant="primary", scheme="batched", seed=D.PRIMARY_SEED,
                   fold_seed=D.PRIMARY_SEED, stage=D.STAGES["p_b5"])
    j2 = D.JobSpec(kind="fit", arm="B5", design="V5", variant="strict", scheme="batched", seed=D.PRIMARY_SEED,
                   fold_seed=D.PRIMARY_SEED, stage=D.STAGES["refit"])
    j3 = D.JobSpec(kind="fit", arm="B6", design="V5", variant="primary", scheme="exact", seed=D.PRIMARY_SEED,
                   writes=D.B6_ARMS, stage=D.STAGES["b6"])
    mk = D.JobSpec(kind="marker", arm="M3+", message=D.NOT_IMPLEMENTED)
    tab = D.estimate_cost([j3, j1, mk, j2], {j1.key: 3, j2.key: 2, j3.key: 1},
                          {("B5", "V5_batched"): uc, ("B6", "V5_exact"): b6}, workers=2)
    assert tab["compute_s"].tolist() == pytest.approx([18.05, 3 * 223.5, 2 * 223.5])
    assert tab["inner_mode"].tolist() == ["full", "full", "full"]
    assert tab["cumulative_compute_h"].iloc[-1] == pytest.approx((18.05 + 670.5 + 447.0) / 3600)
    assert tab["cumulative_wall_h"].iloc[-1] == pytest.approx(tab["cumulative_compute_h"].iloc[-1] / 2)
    assert tab.loc[1, "tuning_s"] == 3 * 180 and tab.loc[2, "calibration_s"] == 2 * 12
    summ = D.cost_summary(tab, workers=2)
    assert summ["fits_budget"] and summ["compute_h_by_arm"]["B5"] == pytest.approx(round(1117.5 / 3600, 3))
    cps = D.stage_checkpoints(tab, workers=2)
    assert [c["stage"] for c in cps] == [D.STAGES["b6"], D.STAGES["p_b5"], D.STAGES["refit"]]
    assert [c["n_folds"] for c in cps] == [1, 3, 2] and all(c["within_budget_wall"] for c in cps)
    assert cps[-1]["cumulative_wall_h"] == pytest.approx(tab["cumulative_wall_h"].iloc[-1], abs=5e-4)  # rounded
    assert cps[1]["stage_compute_h"] == pytest.approx(670.5 / 3600, abs=5e-4)
    tight = D.estimate_cost([j1, j2], {j1.key: 3, j2.key: 2}, {("B5", "V5_batched"): uc}, workers=1,
                            budget_hours=670.5 / 3600 + 1e-9)
    s2 = D.cost_summary(tight, workers=1, budget_hours=670.5 / 3600 + 1e-9)
    assert not s2["fits_budget"] and s2["first_job_beyond_budget"] == j2.key
    tight_cps = D.stage_checkpoints(tight, workers=1, budget_hours=670.5 / 3600 + 1e-9)
    assert tight_cps[0]["within_budget_wall"] and not tight_cps[-1]["within_budget_wall"]
    with pytest.raises(KeyError):
        D.estimate_cost([j3], {j3.key: 1}, {("B5", "V5_batched"): uc})
    bs = D.budget_status(61 * 3600)
    assert bs["exhausted"] and bs["demoted_now"] == ["M7", "M6", "M5", "M4", "M3"] and "H1" in bs["never_demoted"]
    assert not D.budget_status(10 * 3600)["exhausted"]


def test_addendum_cost_estimate_writes_checkpoints_and_a_verdict(tmp_path):
    """The ``--benchmark-addendum1`` writer on stand-in timings (the measurement itself is a fit-only run, not part of
    this test): the re-measured V5 unit costs, the sealed-plan V1 / V2 ones, the per-stage cumulative wall clock on 2
    workers and the fits-in-60-h verdict."""
    meas = [{"arm": a, "design_class": "V5_batched", "stem": "V5__primary__batched", "fold_id": "s104729_S_b000",
             "n_train": 11000, "splits_full": D.INNER_N_FOLDS, "splits_first": D.INNER_N_FOLDS, "inner_fit_s": 10.0,
             "cal_fit_s": 4.0, "outer_refit_s": 20.0, "inner_design": D.INNER_DESIGN_NAME,
             "n_cells_per_split": [30, 30, 29], "peak_rss_bytes": 800_000_000, "n_test_rows_predicted": 0}
            for a in RD.BENCH_ARMS]
    body = RD.write_cost_estimate_addendum1(meas, tmp_path, workers=2)
    ucs = RD.unit_costs_addendum1(meas, body["measurements_reused_from_sealed_plan"])
    assert ucs[("B5", "V5_batched")].splits_full == D.INNER_N_FOLDS
    assert ucs[("B6", "V5_exact")].inner_fit_s == ucs[("B6", "V5_batched")].inner_fit_s   # same inner design
    assert ucs[("B6r0", "V5_batched")] is ucs[("B6", "V5_batched")]
    assert all(m["design_class"] in ("V1", "V2") for m in body["measurements_reused_from_sealed_plan"])
    assert "sealed-plan measurement" in ucs[("B5", "V1")].source
    cps = body["stage_checkpoints"]
    assert [c["stage"] for c in cps] == sorted(c["stage"] for c in cps)
    assert cps[0]["stage"] == D.STAGES["safeguard"] and D.STAGES["s1c"] in [c["stage"] for c in cps]
    assert all(a["cumulative_wall_h"] <= b["cumulative_wall_h"] for a, b in zip(cps, cps[1:]))
    assert body["fits_in_60h"] == body["summary"]["fits_budget"] == cps[-1]["within_budget_wall"]
    assert body["addendum"] == D.N_ADDENDA_EXPECTED and body["plan_state_assumed"]["v5_batched_check"] == "passed"
    bdir = D.discovery_root(tmp_path) / "benchmark"
    md = (bdir / "cost_estimate_addendum1.md").read_text(encoding="utf-8")
    assert (bdir / "cost_estimate_addendum1.json").exists() and "60 h budget" in md
    assert "Not run at all under addendum 1" in md and "V5-P" in md
    assert "Cumulative wall clock by stage" in md and D.STAGES["s1c"] in md


# --------------------------------------------------------------------------------------------- #
# H3 job specs and training transforms (section 11)
# --------------------------------------------------------------------------------------------- #

def test_h3_training_rows_transforms():
    df = _frame()
    df = pd.concat([df, df.iloc[:3].assign(**{SG.METAL_COL: None, SG.ELEMENT_COL: "Am",
                                                FI.ROW_ID: ["X1", "X2", "X3"]}).set_axis(["Z1", "Z2", "Z3"])])
    an = D.actinide_rows(df)
    assert an.sum() == int((df[SG.ELEMENT_COL] == "Am").sum())
    assert D.h3_training_rows(df, "WITH", seed=1) is df
    wo = D.h3_training_rows(df, "WITHOUT", seed=1)
    assert not (wo[SG.ELEMENT_COL] == "Am").any() and len(wo) == len(df) - an.sum()
    perm = D.h3_training_rows(df, "ACT_PERMUTED", seed=1)
    assert perm.loc[~an, I.TARGET_COL].equals(df.loc[~an, I.TARGET_COL])
    for _, g in df[an].groupby([SG.SYSTEM_COL, I.PUB_GROUP_COL]):
        assert sorted(perm.loc[g.index, I.TARGET_COL]) == sorted(g[I.TARGET_COL])
    assert not perm.loc[an, I.TARGET_COL].equals(df.loc[an, I.TARGET_COL])
    sh = D.h3_training_rows(df, "ACT_METAL_SHUFFLED", seed=1)
    assert sh[I.TARGET_COL].equals(df[I.TARGET_COL]) and sh.loc[~an, SG.METAL_COL].equals(df.loc[~an, SG.METAL_COL])
    for _, g in df[an].groupby(SG.SYSTEM_COL):
        assert sorted(map(str, sh.loc[g.index, SG.METAL_COL])) == sorted(map(str, g[SG.METAL_COL]))
    with pytest.raises(ValueError):
        D.h3_training_rows(df, "NOPE", seed=1)


# --------------------------------------------------------------------------------------------- #
# the scorer end to end on synthetic records (selection half only)
# --------------------------------------------------------------------------------------------- #

SCORE_CODE = "code-scoring-test"


def _scoring_world(tmp_path: Path, *, v5_check: str = "passed"):
    """18 systems x 2 cells x 5 rows (selection half) plus confirmation-half rows the scorer must never read; pre-seal
    B3i / B0 predictions; synthetic fold files; discovery records of M2 (5 seeds, batched) and B6 / B6r0 (5 seeds,
    exact) whose digests are the ones ``g19_run_discovery.fold_digest`` gives them under ``SCORE_CODE``."""
    sel = _v5_rows(0.0, 0)
    conf = _v5_rows(0.0, 1, n_systems=4)
    conf.index = conf.index.map(lambda r: "C" + r)
    conf["row_id"] = conf.index
    conf[EM.SYSTEM_COL] = conf[EM.SYSTEM_COL].map(lambda s: "CSYS" + s[3:])
    rows = pd.concat([sel, conf])
    attrs = pd.DataFrame({FI.ROW_ID: rows.index, EM.METAL_STATE_COL: rows[EM.METAL_STATE_COL],
                          EM.SYSTEM_COL: rows[EM.SYSTEM_COL], EM.PUB_GROUP_COL: rows[EM.PUB_GROUP_COL],
                          EM.CONDITION_KEY_COL: [f"ck{i % 3}" for i in range(len(rows))], EM.Y_COL: rows[EM.Y_COL],
                          "dga_stratum": rows["dga_stratum"], "acid_grid_flag": rows["acid_grid_flag"],
                          "censoring_candidate": rows["censoring_candidate"], "acidic_coextractant_modifier": False,
                          "v6_target_row": False, "metal_class": "lanthanide", "acid_stratum": "HNO3"},
                         index=rows.index)
    for d in ("V5", "V5P", "V5PAIR", "V1", "V2", "V5-P"):
        attrs[f"registered_half_{d}"] = np.where(attrs.index.str.startswith("C"), "C", "S")
    out, pre, folds_dir = tmp_path / "out", tmp_path / "preseal", tmp_path / "folds"
    pre.mkdir(parents=True)
    recs = []
    for arm, noise in (("B3i", 1.5), ("B0", 2.0)):
        for half, fr in (("S", sel), ("C", conf)):
            rng = np.random.default_rng(len(arm) + len(half))
            recs.append(pd.DataFrame({"row_id": fr.index, "fold_id": fr["fold_id"], "arm": arm, "half": half,
                                      "unit": "", "seed": -1,
                                      "mean_logD": fr[EM.Y_COL] + rng.normal(0, noise, len(fr))}))
    pd.concat(recs, ignore_index=True).to_parquet(pre / "V5__primary.parquet")
    state = D.PlanState(v5_batched_check=v5_check, v1_tenfold_check="passed")
    (D.discovery_root(out) / "decisions").mkdir(parents=True)
    (D.discovery_root(out) / "decisions" / "plan_state.json").write_text(json.dumps(state.record()), encoding="utf-8")
    v5s = state.heavy_v5_scheme
    cells = list(sel.groupby("fold_id"))

    def fold(fid, scheme, seed, ids, label):
        return FI.make_fold(design="V5", variant="primary", scheme=scheme, fold_id=fid, half="S", seed=seed,
                            hidden=ids, scored=ids, unit_type="cell", units=[label], row_unit={r: label for r in ids},
                            row_half={r: "S" for r in ids})
    exact = [fold(fid, "exact", None, list(g.index), fid) for fid, g in cells]
    batched = [fold(f"s{s}_S_b{k:03d}", v5s, s, list(g.index), fid) for s in D.DISCOVERY_SEEDS
               for k, (fid, g) in enumerate(cells)]
    FI.write_design(exact, folds_dir)
    FI.write_design(batched, folds_dir)
    runners = RD.default_runners()

    def write(arm, job, folds, noise, *, seed):
        dh = FI.design_hash(folds)
        runner = RD.runner_for(job, runners)
        for f, k in D.fittable_folds(job, folds, []):
            d = D.discovery_root(out) / arm / job.design_dir / f"s{seed}"
            d.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(seed + len(arm) + k)
            g = sel.loc[list(f.scored_row_ids)]
            p = pd.DataFrame({c: np.nan for c in D.PREDICTION_COLUMNS}, index=range(len(g))).astype(object)
            p["row_id"], p["fold_id"], p["arm"] = g.index.to_numpy(), f.fold_id, arm
            p["design"], p["variant"], p["scheme"], p["seed"], p["half"] = "V5", "primary", job.scheme, seed, "S"
            p["mean_logD"] = (g[EM.Y_COL] + rng.normal(0, noise, len(g))).to_numpy(dtype=float)
            for c in ("lower_80", "upper_80", "lower_50", "upper_50", "lower_95", "upper_95"):
                p[c] = np.nan
            p.to_parquet(d / f"{D.safe_fold_name(f.fold_id)}.parquet")
            digest = RD.fold_digest(job, f, SCORE_CODE, state, runner, out, ordinal=k, design_hash=dh)
            (d / f"{D.safe_fold_name(f.fold_id)}.json").write_text(json.dumps(
                {"steps": {"point": {"seconds": 1.0}}, "job": job.record(), "fold_id": f.fold_id,
                 "fold_hash": f.fold_hash, "digest": digest}))
    for s in D.DISCOVERY_SEEDS:
        m2 = D.JobSpec(kind="fit", arm="M2", design="V5", variant="primary", scheme=v5s, seed=s, fold_seed=s)
        b6 = D.JobSpec(kind="fit", arm="B6", design="V5", variant="primary", scheme="exact", seed=s, writes=D.B6_ARMS)
        write("M2", m2, batched, 0.1, seed=s)
        write("B6", b6, exact, 1.45, seed=s)
        write("B6r0", b6, exact, 1.6, seed=s)
    crossings = pd.DataFrame({"stem": [f"V5__primary__{v5s}"], "fold_id": [f"s{D.PRIMARY_SEED}_S_b000"],
                              "row_id": [cells[0][1].index[0]], "partner_id": ["x"], "strict_copy": [True]})
    return attrs, out, pre, crossings, folds_dir


def test_scorer_end_to_end_on_synthetic_records(tmp_path):
    SD = _load_script("g19_score_discovery")
    attrs, out, pre, crossings, folds_dir = _scoring_world(tmp_path)
    kw = dict(crossings=crossings, delta5=0.1057, preseal_dir=pre, with_pairs=False, code=SCORE_CODE, folds_dir=folds_dir)
    res = SD.score(out, attrs, **kw)
    dec = res["decisions"]
    assert dec["stop_rule"]["stop"] is False and dec["stop_rule"]["M2_vs_B3i"]["verdict"] == "PASS"
    assert dec["stop_rule"]["B6_vs_B3i"]["verdict"] == "FAIL"
    assert {"M2 vs B3i@V5", "M2 vs B0@V5", "M2 vs B6r0@V5", "B6 vs B3i@V5", "B6 vs B6r0@V5"} <= set(dec["available_contrasts"])
    con = res["contrasts"]
    prim = con[(con["contrast"] == "M2 vs B3i") & con["primary_cluster_unit"].astype(bool)].iloc[0]
    assert prim["family"] == "primary" and prim["margin"] == pytest.approx(0.1057) and prim["point"] > 1.0
    # addendum 1 item 3: seed 104729 only, so R19 item 4 is NOT_EVALUATED and the full verdict is UNDECIDED; the
    # stop-rule, ladder and freezing scopes (which contain neither item 4 nor the refits) still decide
    assert prim["r19_item4"] == D.ITEM4_NOT_EVALUATED and prim["r19_verdict_full"] == "UNDECIDED"
    assert prim["verdict_items_1_5"] == "UNDECIDED" and prim["verdict_stop_rule"] == "PASS"
    assert prim["verdict_ladder"] == "PASS" and prim["verdict_freezing_screen"] == "PASS"
    assert prim["seeds_evaluated"] == str(D.PRIMARY_SEED)
    # addendum 1 item 4: item 6 is the reduced set, and the sensitivities that were not run are named
    assert prim["sensitivity_set"] == D.ADDENDUM_LABEL
    not_run = set(prim["sensitivities_not_run"].split(", "))
    assert not_run == set(D.LEARNED_REFITS_NOT_RUN["V5"]) >= {"loose_setting", "V5-P", "V5-cell-only",
                                                              "parent_structure_hiding", "sr_iii_dropped_training"}
    assert json.loads(prim["sensitivities"])["loose_setting"] == ET.UNTESTABLE
    item6 = res["r19_items"][(res["r19_items"]["key"] == "M2 vs B3i@V5") & (res["r19_items"]["item"] == 6)].iloc[0]
    # the two refits the addendum KEEPS are still required: without their predictions item 6 is UNTESTABLE, and the
    # detail separates them from the sensitivities the addendum does not run at all
    assert item6["status"] == "UNTESTABLE" and D.ADDENDUM_LABEL in item6["detail"]
    assert "strict_setting: UNTESTABLE" in item6["detail"] and "HNO3_only_cells: UNTESTABLE" in item6["detail"]
    assert "not run (addendum 1 item 4)" in item6["detail"] and "loose_setting" in item6["detail"]
    reason = json.loads(prim["sensitivities"])  # values only; the reasons live in decisions.json
    assert reason["strict_setting"] == reason["loose_setting"] == ET.UNTESTABLE
    item4 = res["r19_items"][(res["r19_items"]["key"] == "M2 vs B3i@V5") & (res["r19_items"]["item"] == 4)].iloc[0]
    assert item4["status"] == D.ITEM4_NOT_EVALUATED and "104729 only" in item4["detail"]
    s0 = json.loads(prim["seed_deltas"])
    assert list(s0) == [str(D.PRIMARY_SEED)] and all(v > 0 for v in s0.values())
    assert con.loc[con["contrast"] == "M2 vs B0", "margin"].iloc[0] == pytest.approx(0.05)
    assert set(con["bh_family"]) <= {"registered", "exploratory"} and con["p_bh"].notna().any()
    assert dec["S1_components"]["S1a_M2_vs_B3i"]["scopes"]["stop_rule"] == "PASS"
    assert dec["ladder"]["M1"]["kept"] is None and dec["ladder"]["M3+"]["note"] == D.NOT_IMPLEMENTED
    # the addendum block of decisions.json names what was reduced and what was not run
    add = dec["addendum_1"]
    assert add["addendum"] == 1 and add["seeds_run_in_discovery"] == [D.PRIMARY_SEED]
    assert add["r19_item_4"]["status_in_discovery"] == D.ITEM4_NOT_EVALUATED
    assert "M2 vs B3i@V5" in add["r19_item_4"]["contrasts"] and add["sensitivity_set"]["label"] == D.ADDENDUM_LABEL
    assert dec["not_run"]["addendum_1"]["discovery_seeds"] == [s for s in D.DISCOVERY_SEEDS if s != D.PRIMARY_SEED]
    assert "V5-P" in dec["not_run"]["addendum_1"]["sensitivities"]["V5"]
    cands = {c["contrast"]: c for c in res["freezing_candidates"]}
    # the contrast may still be frozen (items 1-3, 5 and the scoring filters), but the V5-P trigger cannot fire
    assert "M2 vs B3i@V5" in cands and "B6 vs B3i@V5" not in cands
    assert cands["M2 vs B3i@V5"]["passed_items_1_5_v5_primary"] is False
    summ = res["summary"]
    mae = summ[(summ["arm"] == "M2") & (summ["metric"] == "mae") & (summ["aggregation"] == "unit_macro")
               & (summ["stratum"] == "all") & (summ["scoring_filter"] == "none")]
    assert len(mae) == 1 and (mae["seed"] == D.PRIMARY_SEED).all() and (mae["half"] == "selection").all()
    assert (mae["n_units"] == 36).all()
    # the wildcard-copy filter removed exactly the flagged row of the M2 batched folds
    wc = summ[(summ["arm"] == "M2") & (summ["scoring_filter"] == ET.WILDCARD_COPY_SENSITIVITY)
              & (summ["metric"] == "mae") & (summ["aggregation"] == "row_pooled") & (summ["seed"] == D.PRIMARY_SEED)]
    assert int(wc["n_rows"].iloc[0]) == 179
    assert dec["registered_family_accounting"]["m_full"] == D.REGISTERED_FULL_FAMILY_SIZE == 60
    assert dec["registered_family_accounting"]["m_discovery"] == len(D.REGISTERED_FAMILY_TABLE) == 57
    assert con["p_bh_full_family"].notna().any() and not dec["S1_components"]["S1_forced_undecided"]
    assert dec["not_run"]["uncertainty_metrics"] == D.UNCERTAINTY_NOT_RUN
    assert all(v["status"] in ("complete", "missing") for v in dec["record_sets"].values())
    assert dec["record_sets"][f"M2/V5__primary_batched/s{D.PRIMARY_SEED}"]["status"] == "complete"
    # task X finding V-LP-03: records of other code are refused, not scored
    with pytest.raises(D.StaleRecordError, match="digest"):
        SD.score(out, attrs, **dict(kw, code="code-after-a-change"))
    # a record of a fold the current fold file does not fit is refused
    bdir = D.discovery_root(out) / "B6" / "V5__primary_exact" / f"s{D.PRIMARY_SEED}"
    victim = sorted(bdir.glob("*.json"))[0]
    body = json.loads(victim.read_text())
    foreign = bdir / "Zz(III)__SYS99.json"
    foreign.write_text(json.dumps(dict(body, fold_id="Zz(III)__SYS99")))
    victim.with_suffix(".parquet").replace(foreign.with_suffix(".parquet"))
    victim.rename(bdir / "moved.bak")
    with pytest.raises(D.StaleRecordError):
        SD.score(out, attrs, **kw)
    foreign.with_suffix(".parquet").replace(victim.with_suffix(".parquet"))
    foreign.unlink()
    # a prediction without its record is refused; an incomplete job (both files absent) is not scored, never mixed
    with pytest.raises(D.StaleRecordError, match="without a JSON record"):
        SD.score(out, attrs, **kw)
    victim.with_suffix(".parquet").rename(bdir / "moved.pq.hidden")
    part = SD.score(out, attrs, **kw)
    assert "B6 vs B3i@V5" not in part["decisions"]["available_contrasts"]
    assert part["decisions"]["record_sets"][f"B6/V5__primary_exact/s{D.PRIMARY_SEED}"]["status"] == "incomplete"
    (bdir / "moved.bak").rename(victim)
    (bdir / "moved.pq.hidden").rename(victim.with_suffix(".parquet"))
    # a confirmation-half row slipped into a stored record is refused
    bad = D.discovery_root(out) / "M2" / "V5__primary_batched" / f"s{D.PRIMARY_SEED}"
    first = sorted(bad.glob("*.parquet"))[0]
    p = pd.read_parquet(first)
    p.loc[0, "row_id"] = attrs.index[attrs.index.str.startswith("C")][0]
    p.to_parquet(first)
    with pytest.raises(AssertionError, match="selection half"):
        SD.score(out, attrs, **kw)


def test_scorer_reports_s1_undecided_when_the_recoloured_check_failed(tmp_path):
    """Task X finding V-05: with the re-coloured batched-vs-exact check failed, every S1 component is reported UNDECIDED
    and heavy-arm V5 contrasts, rows and freezing candidates carry 'batched (check failed)'."""
    SD = _load_script("g19_score_discovery")
    attrs, out, pre, crossings, folds_dir = _scoring_world(tmp_path, v5_check="failed")
    res = SD.score(out, attrs, crossings=crossings, delta5=0.1057, preseal_dir=pre, with_pairs=False, code=SCORE_CODE,
                   folds_dir=folds_dir)
    dec = res["decisions"]
    s1 = dec["S1_components"]
    assert s1["S1_forced_undecided"] and s1["S1a_M2_vs_B3i"]["reported_verdict"] == "UNDECIDED"
    assert s1["S1a_M2_vs_B3i"]["batching_label"] == D.CHECK_FAILED_LABEL
    con = res["contrasts"]
    prim = con[(con["contrast"] == "M2 vs B3i") & con["primary_cluster_unit"].astype(bool)].iloc[0]
    assert prim["reported_verdict"] == "UNDECIDED" and prim["batching_label"] == D.CHECK_FAILED_LABEL
    b6 = con[(con["contrast"] == "B6 vs B3i") & con["primary_cluster_unit"].astype(bool)].iloc[0]
    assert b6["batching_label"] == "" and dec["heavy_v5_batching_label"] == D.CHECK_FAILED_LABEL
    for c in res["freezing_candidates"]:
        if c["family"] in D.S1_FAMILIES:
            assert c["s1_forced_undecided"] and c["batching_label"] == D.CHECK_FAILED_LABEL
    assert "UNDECIDED" in SD.contrasts_markdown(con, dec)
