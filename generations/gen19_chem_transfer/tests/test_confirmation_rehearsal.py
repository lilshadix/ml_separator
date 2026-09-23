"""The single confirmation run REHEARSED WHOLE -- ``scripts/g19_run_confirmation.py::main()`` end to end -- on the
synthetic mini-corpus of ``confirmation_rehearsal_support.py`` with a fake seed store, plus one regression test per
runner blocker the fourth verification round left open (``decisions/D07_confirmation_not_run.md``):

1. a missing M1 record ends the M2 FOLD, not the run -- on the WORKER path too (addendum 8 item 1);
2. a spent run is refused with or without ``--resume``; a run that never wrote ``confirmation.json`` resumes and fits
   only what is missing (addendum 8 item 2);
3. claim C4's two legs: distinct record paths, the section 11 ACT_PERMUTED transform applied to the training rows, a
   FROZEN refit off the WITH record of the same fold, the addendum 5 item 2 guard rule, a non-zero delta;
4. the V6 configuration reading of section 3.4 / addenda 7-8: M1 tuned per section 7 on the V6 fold, M2 from M1's
   same-fold record (``m2_pairing``), the reading written into the record;
5. ``--max-folds`` is a PAUSE: nothing assembled, ``confirmation.json`` not written, the run resumable;
6. the S1(c) yardsticks re-fit on the SCRUBBED (seed-free) per-seed V5-PAIR file, re-seeded in memory;
7. a mixed-code record set is refused before anything is fitted or scored.

Nothing here touches the real corpus, a real fold file, the confirmation half of any real design, V6, or a withheld
seed: the corpus, the folds, the plan, the registry and the five seeds are the test's own, in ``tmp_path``.  The whole
rehearsal fits about 45 neural folds and is marked ``slow`` (roughly 20-30 minutes on the reference machine).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gen19ct import paths
from gen19ct.evaluation import confirmation as CF
from gen19ct.evaluation import discovery as D
from gen19ct.evaluation import h3 as H3
from gen19ct.evaluation import metrics as EM
from gen19ct.evaluation import registry as REG
from gen19ct.folds import io as FI
from gen19ct.models import s1c_yardsticks as SY

import confirmation_rehearsal_support as RS  # noqa: E402  -- tests/ is on sys.path (conftest), in the workers too

RC = RS.rc()
SEAL = RS.seal()
FAKE_SEEDS = RS.FAKE_SEEDS
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory) -> Path:
    return RS.make_corpus_dir(tmp_path_factory.mktemp("rehearsal_corpus"))


class Rehearsal:
    """One ``--out-root`` prepared for ``main()``: the skeleton, the fake store, the test's own registry, and every
    monkeypatch that points the runner at the synthetic corpus (in process AND, through the environment, in the
    spawned workers)."""

    def __init__(self, tmp_path: Path, monkeypatch, corpus_dir: Path, name: str):
        self.root = tmp_path / name
        self.store_path = tmp_path / "store" / "fake_confirmation_seeds.json"
        RS.write_run_skeleton(self.root, self.store_path)
        self.reg_path = self.root / "manifests" / "digest_registry.json"
        monkeypatch.setattr(REG, "registry_path", lambda r=None, _p=self.reg_path: _p)
        self.code = RC.code_digest()["combined"]
        REG.register_stage(CF.STAGE, below_footer_sha256="a" * 64, code_digest=self.code, git_head=None,
                           addenda_count=8, note="rehearsal", path=self.reg_path)
        self.corpus, self.attrs = RS.build_corpus(corpus_dir)
        self.manifests_out = tmp_path / "manifests_out"
        monkeypatch.setattr(paths, "MANIFESTS_DIR", self.manifests_out)
        monkeypatch.setattr(RC, "confirmation_corpus", lambda: self.corpus)
        monkeypatch.setattr(RC, "pair_attributes", lambda: self.attrs)
        monkeypatch.setattr(RC, "load_fold_corpus", lambda: RS.fold_corpus_stub(self.corpus.frame))
        monkeypatch.setattr(RC, "build_v5_batched", RS.build_v5_batched)
        monkeypatch.setattr(RC, "build_v5pair_batched", RS.build_v5pair_batched)
        monkeypatch.setattr(RC, "build_v6_folds", RS.build_v6_folds)
        monkeypatch.setattr(RC, "_init_conf_worker", RS.init)
        monkeypatch.setattr(RC, "INCOMPLETE_RECORD_SETS", set())
        # the strict variant at the primary thresholds, here and in every worker (RS.relax_strict_variant, documented)
        import dataclasses

        from gen19ct.folds import cell_holdout as CH
        monkeypatch.setitem(CH.VARIANTS, "strict", dataclasses.replace(CH.VARIANTS["strict"], thresholds=CH.PRIMARY))
        monkeypatch.setenv("G19_REHEARSAL_REGISTRY", str(self.reg_path))
        monkeypatch.setenv("G19_REHEARSAL_CORPUS", str(corpus_dir))
        monkeypatch.syspath_prepend(str(HERE))
        self.store = CF.load_seed_store(self.store_path, root=self.root, seal=SEAL,
                                        prereg_paths=SEAL.PreregPaths(root=self.root, repo_root=self.root.parent))

    def argv(self, *extra: str, root: Path | None = None, workers: int = 2) -> list[str]:
        return ["--out-root", str(root or self.root), "--seed-store", str(self.store_path), "--workers", str(workers),
                "--expect-addenda", "8", *extra]

    def main(self, *extra: str, root: Path | None = None, workers: int = 2) -> int:
        RC.INCOMPLETE_RECORD_SETS.clear()
        return RC.main(self.argv(*extra, root=root, workers=workers), check=lambda: 0, digests=RS.gate_digests,
                       seal=SEAL, registry_path=self.reg_path)

    # ---- what a run left on disk
    def decisions(self, root: Path | None = None) -> dict:
        return json.loads(CF.decisions_path(root or self.root).read_text(encoding="utf-8"))

    def records(self, root: Path | None = None) -> list[Path]:
        return sorted((CF.conf_root(root or self.root) / "records").rglob("*.json"))

    def copy_run(self, name: str, *, finished: bool) -> Path:
        """A copy of the finished run, with (``finished=False``) the assembly outputs removed so it is an UNFINISHED run
        whose lock is the start marker (addendum 8 item 2)."""
        dst = self.root.parent / name
        shutil.copytree(self.root, dst)
        if not finished:
            for p in (CF.decisions_path(dst), CF.report_path(dst)):
                p.unlink()
            shutil.rmtree(CF.tables_dir(dst), ignore_errors=True)
        return dst


def _no_seed_in(text: str) -> bool:
    return not any(re.search(rf"(?<!\d){s}(?!\d)", text) for s in FAKE_SEEDS)


def _seed_free_tree(root: Path, *allow: Path) -> list[str]:
    allowed = {str(a.resolve()) for a in allow}
    hits = []
    for f in sorted(q for q in root.rglob("*") if q.is_file()):
        if str(f.resolve()) in allowed:
            continue
        b = f.read_bytes()
        if any(str(s).encode() in b for s in FAKE_SEEDS):
            hits.append(str(f))
    return hits


# --------------------------------------------------------------------------------------------- #
# the per-blocker regression tests (fail without the fix)
# --------------------------------------------------------------------------------------------- #

def test_the_s1c_yardsticks_refit_on_a_scrubbed_seed_free_pair_file(tmp_path, corpus_dir):
    """Blocker 6 (found by the rehearsal): the confirmation run writes its per-seed V5-PAIR colouring SCRUBBED --
    ``seed=None`` on every fold (addendum 6 item 4) -- and ``models.s1c_yardsticks.read_batched_v5pair_design`` refused
    such a file ('drawn with one seed, got [None]'), so ``s1c_assembly`` would have died at its first seed after the
    56 h fit loop.  The caller now passes the withheld seed and the file is re-seeded in memory; the design hash is
    unchanged by it."""
    corpus, attrs = RS.build_corpus(corpus_dir)
    folds, ptab, _ = RS.build_v5pair_batched(RS.fold_corpus_stub(corpus.frame), FAKE_SEEDS[0])
    d = tmp_path / "i1"
    scrubbed = RC.scrub_folds(folds, RS.write_store(tmp_path / "s.json", tmp_path) and CF.load_seed_store(
        tmp_path / "s.json", root=tmp_path, seal=SEAL, prereg_paths=SEAL.PreregPaths(root=tmp_path, repo_root=tmp_path)))
    assert all(f.seed is None for f in scrubbed) and all("si1" in f.fold_id for f in scrubbed)
    _, js, _ = FI.write_design(scrubbed, d)
    with pytest.raises(ValueError, match="drawn with one seed"):
        SY.read_batched_v5pair_design(js)                          # the failure the real run would have hit
    got, stem, dhash = SY.read_batched_v5pair_design(js, seed=FAKE_SEEDS[0])
    assert stem == RS.V5PAIR_STEM and all(f.seed == FAKE_SEEDS[0] for f in got)
    assert dhash == FI.design_hash(scrubbed)                        # the seed is not part of the design's identity
    with pytest.raises(ValueError, match="not the seed given"):
        SY.read_batched_v5pair_design(FI.write_design(folds, tmp_path / "raw")[1], seed=FAKE_SEEDS[1])
    refit = SY.refit_lookup_yardsticks(js, corpus.frame, v6_mask=corpus.v6, guard=SY.registered_guard(corpus.frame),
                                       table=corpus.table, exclude_from_scoring=corpus.coext, halves=("C",),
                                       seed=FAKE_SEEDS[0])
    assert refit.seed == FAKE_SEEDS[0] and set(refit.arms) == {"B3x", "B3i"} and len(refit.predictions) > 0
    # ... and the builder's pair table is what the pair regeneration reproduces (the S1(c) assembly checks this)
    sp = ptab.assign(fold_id=[CF.scrub(v, refit and _store(tmp_path)) for v in ptab["fold_id"]])
    inp = SY.yardstick_pair_inputs(refit, attrs, v6_mask=RC.v6_by_id(corpus), builder_pairs=sp)
    assert len(inp["pairs"]) == len(sp) > 0


def test_the_wildcard_copy_scoring_filter_finds_its_column_on_this_runs_frames(tmp_path, monkeypatch, corpus_dir):
    """Blocker 7 (found by the whole rehearsal, in the assembly, AFTER the 22-minute fit loop): R19 item 6's registered
    V5 sensitivities include the wildcard-copy filter, and ``discovery.apply_scoring_filter`` reads a prediction-frame
    column only the discovery SCORER adds (``wildcard_copy_partner_in_training``, from the crossings of the REGISTERED
    designs).  The confirmation run's frames never had it, so ``score_claims`` raised ``KeyError`` -- the real run would
    have died there after its 56 h fit loop.  The runner now computes the flags inside the run with the fold builder's
    own ``folds.io.copy_crossings`` on the design each row was scored in, including the withheld-seed colourings."""
    from gen19ct.evaluation import transfer as ET

    rh = Rehearsal(tmp_path, monkeypatch, corpus_dir, "wc")
    corpus = rh.corpus
    v5f = RS.build_v5_batched(RS.fold_corpus_stub(corpus.frame), "primary", FAKE_SEEDS[0])[0][0]
    d = RC.seed_folds_dir(rh.root, 1)
    scrubbed = RC.scrub_folds([v5f], rh.store)
    FI.write_design(scrubbed, d)
    f = scrubbed[0]
    sc = sorted(f.scored_row_ids)
    frame = pd.DataFrame({"row_id": sc, "fold_id": f.fold_id, "design": "V5", "variant": "primary",
                          "scheme": "batched_max4", "mean_logD": 0.0})
    # a corpus without a pair table flags nothing, and the registered filter then keeps every row instead of raising
    out = RC.attach_wildcard_flags(frame, rh.root, 1, corpus)
    assert set(out[RC.WC_COL]) == {False} and set(out[RC.WC_COL + "_strict"]) == {False}
    assert len(D.apply_scoring_filter(out, ET.WILDCARD_COPY_SENSITIVITY)) == len(out)
    with pytest.raises(KeyError):
        D.apply_scoring_filter(frame, ET.WILDCARD_COPY_SENSITIVITY)          # the failure the real run would have hit
    # a pair table with one wildcard partner of a scored row that stays in the fold's TRAINING set: that row is flagged
    train = sorted(set(corpus.frame[FI.ROW_ID].astype(str)) - set(f.hidden_row_ids))
    pairs = pd.DataFrame({"kind": ["STATE_WILDCARD", "STATE_WILDCARD"], "id_a": [sc[0], sc[1]],
                          "id_b": [train[0], sc[2]], "same_publication": ["True", "True"],
                          "strict_copy": ["True", "False"]})
    monkeypatch.setattr(type(corpus), "wildcard_pairs", lambda self, _p=pairs: _p)
    out = RC.attach_wildcard_flags(frame, rh.root, 1, corpus)
    flagged = set(out.loc[out[RC.WC_COL], "row_id"])
    assert flagged == {sc[0]}                              # sc[1]'s partner is hidden with it: no crossing
    assert set(out.loc[out[RC.WC_COL + "_strict"], "row_id"]) == {sc[0]}
    assert len(D.apply_scoring_filter(out, ET.WILDCARD_COPY_SENSITIVITY)) == len(out) - 1
    # ... and the design is the SEED's colouring, read from the seed directory the run wrote, not a registered file
    assert (d / "V5__primary__batched_max4.json").exists()


def test_a_pandas_3_string_column_is_scrubbed_like_an_object_column(tmp_path):
    """Blocker 8 (found by the whole rehearsal): under pandas 3 a column of strings is dtype ``str``, not ``object``, so
    the ``dtype == object`` scrub of the fold-build phase left RAW withheld-seed fold ids in every seed directory's
    ``V5PAIR__primary__batched__pairs.parquet`` -- a seed leak the run's own pre-report scan would have refused after
    the whole fit loop, and a file S1(c) could not match against the scrubbed design.  ``CF.scrub_frame`` scrubs every
    text column whatever its dtype and leaves numeric columns alone."""
    store = _store(RS.write_store(tmp_path / "s.json", tmp_path).parent)
    seed = store.seed(1)
    fr = pd.DataFrame({"fold_id": [f"s{seed}_C_b000", f"s{seed}_C_b001"], "row_id": ["R:0", "R:1"],
                       "n": [seed, 2], "x": [0.5, 1.5]})
    fr["obj"] = pd.Series([f"batch s{seed}", None], dtype=object)
    assert str(fr["fold_id"].dtype) == "str"                               # pandas 3: not object
    naive = fr.copy()
    for col in naive.columns:
        if naive[col].dtype == object:
            naive[col] = [CF.scrub(v, store) for v in naive[col]]
    assert f"s{seed}_C_b000" in set(naive["fold_id"])                      # the old scrub missed it
    out = CF.scrub_frame(fr, store)
    assert list(out["fold_id"]) == ["si1_C_b000", "si1_C_b001"] and list(out["obj"])[0] == "batch si1"
    assert pd.isna(out["obj"].iloc[1]) and list(out["n"]) == [seed, 2] and list(out["x"]) == [0.5, 1.5]
    assert out["fold_id"].dtype == fr["fold_id"].dtype and out["obj"].dtype == fr["obj"].dtype   # dtypes kept
    p = tmp_path / "pairs.parquet"
    out.drop(columns=["n"]).to_parquet(p, index=False)
    assert str(seed).encode() not in p.read_bytes() and pd.read_parquet(p)["fold_id"].tolist() == ["si1_C_b000", "si1_C_b001"]


def test_the_refit_pass_never_inherits_another_seed_indexs_colouring(tmp_path, monkeypatch, corpus_dir):
    """Blocker 10 (found by replaying the resume scenarios in one process): ``attach_seed_folds`` replaced the corpus's
    cached design only when the seed directory had a file, so after index 5 had been attached, the index-0 refit pass --
    which has no seed directory (its designs are the registered seed-104729 files) -- fitted index 5's strict /
    HNO3-only colouring and wrote ``i0/si5_C_b000`` records.  Every index now resets the cache to the registered files
    for the stems it has no file for."""
    rh = Rehearsal(tmp_path, monkeypatch, corpus_dir, "cache")
    corpus = rh.corpus
    stem = "V5__strict__batched_max4"
    registered = corpus.folds(stem)
    assert {f.seed for f in registered} == {D.PRIMARY_SEED} and registered[0].fold_id == "s104729_C_b000"
    (f5,), _ = RS.build_v5_batched(RS.fold_corpus_stub(corpus.frame), "strict", FAKE_SEEDS[4])
    FI.write_design(RC.scrub_folds([f5], rh.store), RC.seed_folds_dir(rh.root, 5))
    got = RC.attach_seed_folds(corpus, rh.root, 5)
    assert stem in got and corpus.folds(stem)[0].fold_id == "si5_C_b000"      # index 5's colouring is attached
    got0 = RC.attach_seed_folds(corpus, rh.root, 0)                             # the refit pass: no seed directory
    assert got0 == {} and corpus.folds(stem)[0].fold_id == "s104729_C_b000"   # ... and the registered file is back
    assert corpus.design_hash(stem) == FI.design_hash(registered)
    # the job spec the refit pass builds from the corpus then pins the public seed of the multi-seed file again
    job = CF.ConfJob("R19 item 6 refits (discovery seed 104729 only)", "M1", "V5", stem, 0, 1)
    assert RC.fold_seeded_spec(job, CF.ITEM6_REFIT_SEED, corpus).fold_seed == D.PRIMARY_SEED


def test_the_decisions_body_survives_an_incomplete_claim(tmp_path):
    """Blocker 11 (found by the whole rehearsal's missing-M1 scenario): ``decisions_body`` indexed ``items[1]`` of every
    claim for the per-family BH, and an INCOMPLETE claim has no items -- ``IndexError`` at the END of the assembly
    whenever one claim was incomplete.  It now contributes NaN, which ``bh_adjust`` skips."""
    store = _store(RS.write_store(tmp_path / "s.json", tmp_path).parent)
    complete = {"claim_id": "C4", "claim": "B6:WITH vs B6:ACT_PERMUTED@V2", "family": "H3", "status": "COMPLETE",
                "items": [{"item": 1, "status": "PASS"}, {"item": 2, "status": "PASS", "p_two_sided": 0.02}]}
    incomplete = {"claim_id": "C1", "claim": "M2 vs B3i@V5", "family": "primary (H1, S1(a))", "status": "INCOMPLETE",
                  "r19_verdict": CF.NOT_EVALUATED, "point": float("nan")}
    plan = CF.read_plan(CF.plan_path(RS.write_run_skeleton(tmp_path / "root", tmp_path / "s2.json") or tmp_path / "root"))
    body = RC.decisions_body(plan=plan, gate={"prereg": RS.gate_digests()}, store=store, claims=[incomplete, complete],
                             s1c={"verdict": CF.UNDECIDED, "status": "INCOMPLETE"},
                             s1de={"s1d": {"status": "INCOMPLETE"}, "s1e": {"status": "INCOMPLETE"}},
                             s2={"status": "INCOMPLETE", "verdict": CF.UNDECIDED},
                             fold_plans=RC.build_confirmation_folds(None, tmp_path / "root", dry_run=True),
                             ledger=RC.FitLedger(), code="c" * 64)
    bh = body["bh_per_family"]["registered"]
    assert bh["B6:WITH vs B6:ACT_PERMUTED@V2"] == 0.02 and np.isnan(bh["M2 vs B3i@V5"])
    assert body["s1"]["verdict"] == CF.UNDECIDED and body["s1"]["components"]["S1a"] == CF.UNDECIDED


def _store(root: Path) -> CF.SeedStore:
    return CF.load_seed_store(root / "s.json", root=root, seal=SEAL,
                              prereg_paths=SEAL.PreregPaths(root=root, repo_root=root))


def test_c4s_control_leg_is_permuted_frozen_and_guarded_on_the_recorded_corpus(tmp_path, monkeypatch, corpus_dir):
    """Blocker 3: the ACT_PERMUTED leg of claim C4.  Before the fix the confirmation loop never called
    ``h3.transformed_frame``, ``discovery.h3_training_rows`` or ``h3.frozen_runner`` (measured: 0 calls each), fitted
    the control leg on the RECORDED log D with the tuning runner, and both legs shared one record path -- C4's delta
    was exactly 0.0.  Now, on one synthetic V2 fold with 108 actinide training rows:

    * the WITH leg is B6 tuned per section 7; the control leg is ``h3.FrozenB6`` at the WITH record's selection;
    * the permutation changes training log D (``n_training_rows_changed`` > 0) and touches no scored row;
    * the guard is bound to the RECORDED corpus (addendum 5 item 2);
    * the two records are distinct files, the control record names the WITH record's digest, and the predictions
      differ -- so the paired delta is not zero;
    * a control leg whose WITH record is absent is ONE incomplete fold (``INCOMPLETE_WITH_RECORD_MISSING``), not a run.
    """
    rh = Rehearsal(tmp_path, monkeypatch, corpus_dir, "c4")
    rd = RC.runner_module()
    corpus, store, code = rh.corpus, rh.store, rh.code
    fold = next(f for f in corpus.folds("V2__element__exact") if f.fold_id == "La(III)")
    plan = CF.read_plan(CF.plan_path(rh.root))
    legs = [j for j in CF.enumerate_jobs(plan) if j.purpose.startswith("claim C4") and j.seed_index == 1]
    with_spec, perm_spec = (RC.job_spec_of(j, store.seed(1)) for j in legs)
    assert (with_spec.condition, perm_spec.condition) == ("WITH", "ACT_PERMUTED")
    calls = {"transformed_frame": 0, "frozen_runner": 0, "guard_corpora": []}

    def counted(name, fn):
        def g(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return g

    real_guard = rd.outer_guard

    def spy_guard(job, f, c, universe):
        calls["guard_corpora"].append(id(c))
        return real_guard(job, f, c, universe)

    monkeypatch.setattr(H3, "transformed_frame", counted("transformed_frame", H3.transformed_frame))
    monkeypatch.setattr(H3, "frozen_runner", counted("frozen_runner", H3.frozen_runner))
    monkeypatch.setattr(rd, "outer_guard", spy_guard)
    kw = dict(store=store, seed_index=1, runners=rd.default_runners(), code=code, state=D.PlanState(guard_mode={}),
              prereg={"addenda_sha256": "a" * 64, "n_addenda": 8})
    out_root = rh.root

    # the control leg BEFORE its WITH record exists: one incomplete fold, the loop goes on (blocker 1's sibling case)
    with pytest.raises(CF.MissingSiblingRecordError) as exc:
        RC.run_fold(perm_spec, fold, 0, corpus, out_root, **kw)
    assert exc.value.status == CF.INCOMPLETE_WITH_RECORD_MISSING and "records/B6/WITH/" in str(exc.value)
    assert rh.records() == [] and calls["transformed_frame"] == 0        # found BEFORE any transform work is spent

    r_with = RC.run_fold(with_spec, fold, 0, corpus, out_root, **kw)
    assert r_with["status"] == "fitted" and calls["frozen_runner"] == 0 and calls["transformed_frame"] == 0
    r_perm = RC.run_fold(perm_spec, fold, 0, corpus, out_root, **kw)
    assert r_perm["status"] == "fitted"
    assert calls["frozen_runner"] == 1 and calls["transformed_frame"] == 1         # the transform and the frozen refit
    assert len(set(calls["guard_corpora"])) == 1 and calls["guard_corpora"][0] == id(corpus)   # recorded corpus only

    js_with, js_perm = Path(r_with["records"]["B6"]), Path(r_perm["records"]["B6"])
    assert js_with != js_perm and js_with.parent.parent.parent.name == "WITH" and \
        js_perm.parent.parent.parent.name == "ACT_PERMUTED"
    w, p = (json.loads(x.read_text(encoding="utf-8")) for x in (js_with, js_perm))
    assert w["transform"] == "WITH" and p["transform"] == "ACT_PERMUTED"
    assert p["n_actinide_training_rows"] == 108 and p["n_training_rows_changed"] > 0
    assert p["guard_value_source"] == "recorded_log_D" and p["with_record_digest"] == CF.confirmation_record_digest(w)
    assert p["with_pairing"]["with_record_digest"] == p["with_record_digest"]
    assert p["selected_config"] == w["selected_config"]                            # frozen at the WITH selection
    assert p["arm_record"]["selected_hyperparameters"]["rank"] == w["arm_record"]["selection"]["selected_rank"]
    assert "selection" in w["arm_record"] and "refit" in p["arm_record"]         # tuned vs frozen record layouts
    assert p["design_hash"] == w["design_hash"] == corpus.design_hash("V2__element__exact")
    fw, fp = (pd.read_parquet(x.with_suffix(".parquet")) for x in (js_with, js_perm))
    assert list(fw["row_id"]) == list(fp["row_id"]) == sorted(fold.scored_row_ids)   # the same scored rows
    assert not np.allclose(fw["mean_logD"], fp["mean_logD"])                          # ... and different predictions
    yv = rh.attrs[EM.Y_COL].reindex(fw["row_id"].to_numpy()).to_numpy(dtype=float)
    mae_w, mae_p = (np.abs(fw["mean_logD"].to_numpy() - yv).mean(), np.abs(fp["mean_logD"].to_numpy() - yv).mean())
    assert abs(mae_w - mae_p) > 1e-6, (mae_w, mae_p)      # the control MOVES the unit MAE, so C4's delta is not float noise
    for blob in (js_with.read_text(encoding="utf-8"), js_perm.read_text(encoding="utf-8"), str(js_perm)):
        assert _no_seed_in(blob)

    # the C4 claim reads the two legs from their two directories and restricts to Ln(III) rows: a non-zero delta
    RC.INCOMPLETE_RECORD_SETS.clear()
    pu = RC.claim_paired_units(out_root, CF.claim_of(plan, "C4"), rh.attrs, corpus, seed_index=1,
                               cand_dir=with_spec.design_dir, comp_dir=perm_spec.design_dir)
    assert pu is not None and pu.delta != 0.0 and set(pu.cand_mae.index) == {"La(III)"}


def test_a_missing_m1_record_is_one_incomplete_m2_fold_on_the_worker_path_too(tmp_path, monkeypatch, corpus_dir):
    """Blocker 1, the WORKER path: ``fit_loop`` at ``--workers 2`` dispatches ``_conf_worker_fold``; a
    ``MissingSiblingRecordError`` raised inside the REAL ``run_fold`` (M2 with no M1 record on disk) must come back as
    the status the ledger files as ``INCOMPLETE_M1_RECORD_MISSING`` -- and the loop must go on to the next job.  The pool
    is an in-process stand-in (the real spawn is exercised by the whole rehearsal below); the worker function and
    ``run_fold`` are the real ones."""
    rh = Rehearsal(tmp_path, monkeypatch, corpus_dir, "w")
    rd = RC.runner_module()
    corpus, store, code = rh.corpus, rh.store, rh.code
    RC.CW.corpus, RC.CW.runners, RC.CW.store, RC.CW.attached, RC.CW.pair_attrs = corpus, rd.default_runners(), store, 0, rh.attrs
    submitted = []

    class InlinePool:
        def __init__(self, **kw):
            pass

        def submit(self, fn, *a):
            f = Future()
            f.set_running_or_notify_cancel()
            try:
                f.set_result(fn(*a))
            except BaseException as exc:                  # noqa: BLE001 -- the pool's own contract
                f.set_exception(exc)
            submitted.append(fn.__name__)
            return f

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr(RC, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(RC, "attach_seed_folds", lambda *a, **k: {})
    pids = iter(range(1, 1000))
    monkeypatch.setattr(RC, "_conf_worker_state",                      # one distinct pid per probe (every worker served)
                        lambda *a: {"pid": next(pids), "n_rows": int(corpus.table.n), "n_runners": 1, "store_verified": True,
                                    "seed_commitment_sha256": store.digest, "n_seed_indices": CF.N_SEEDS})
    m2 = CF.ConfJob("claims C1-C3 (core)", "M2", "V5", "V5__strict__batched_max4", 0, 1)
    b0 = CF.ConfJob("claim comparators (closed form)", "B0", "V5", "V5__strict__exact", 0, 2)
    led = RC.fit_loop([m2, b0], store, rh.root, corpus=corpus, state=D.PlanState(guard_mode={}),
                      runners=rd.default_runners(), code=code, prereg={"addenda_sha256": "a" * 64, "n_addenda": 8},
                      workers=2, seed_store_path=str(rh.store_path))
    rec = led.record()
    assert "_conf_worker_fold" in submitted                                     # the REAL worker function ran
    assert [e["status"] for e in rec["errors"]] == [CF.INCOMPLETE_M1_RECORD_MISSING]
    assert "records/M1/" in rec["errors"][0]["detail"] and rec["errors"][0]["job"] == m2.key
    assert rec["n_folds_fitted"] == 2 and rec["by_job"][b0.key]["fitted"] == 2     # the next job still ran
    assert _no_seed_in(json.dumps(rec))
    marked = RC.mark_incomplete_record_sets(led, [m2, b0])
    assert marked and marked[0]["arm"] == "M2" and marked[0]["transform"] == ""


def test_resume_completes_a_fold_whose_write_set_is_partial_and_refuses_foreign_code(tmp_path, monkeypatch, corpus_dir):
    """Task X findings V-F1 and V-F2.  The resume unit is the FOLD -- every arm one fit writes (B6 -> B6 and B6r0), each
    with its parquet and its record JSON under the live code.  Measured before the fix: ``run_fold`` looked at job.arm's
    JSON alone, so a lost B6r0 record or a lost parquet was 'skipped_done' and surfaced only after the whole fit loop.
    Now, on one synthetic V2 fold: a complete fold is skipped; a lost sibling record, a lost parquet, an orphan parquet
    and a TRUNCATED record JSON (a worker killed mid-write) each make the fold MISSING and it is re-fitted whole, its
    partial artefacts removed first; a record under another code digest is still REFUSED, nothing deleted."""
    rh = Rehearsal(tmp_path, monkeypatch, corpus_dir, "resume")
    rd = RC.runner_module()
    corpus, store, code = rh.corpus, rh.store, rh.code
    fold = next(f for f in corpus.folds("V2__element__exact") if f.fold_id == "La(III)")
    spec = D.JobSpec(kind="fit", arm="B6", design="V2", variant="element", scheme="exact", seed=store.seed(1))
    runner = rd.runner_for(spec, rd.default_runners())
    assert RC.fold_writes(spec, runner) == ("B6", "B6r0")                         # the write set is the runner's
    kw = dict(store=store, seed_index=1, runners=rd.default_runners(), code=code, state=D.PlanState(guard_mode={}),
              prereg={"addenda_sha256": "a" * 64, "n_addenda": 8})
    r = RC.run_fold(spec, fold, 0, corpus, rh.root, **kw)
    assert r["status"] == "fitted" and set(r["records"]) == {"B6", "B6r0"} and "refit_partial_removed" not in r
    files = {a: CF.fold_paths(rh.root, a, spec.design_dir, 1, fold.fold_id) for a in ("B6", "B6r0")}
    assert all(p.exists() for pq_js in files.values() for p in pq_js)
    assert not list(files["B6"][1].parent.glob("*.tmp*"))                          # atomic writes leave no temp file
    assert RC.run_fold(spec, fold, 0, corpus, rh.root, **kw)["status"] == "skipped_done"

    def refit_after(break_it, expect_removed):
        break_it()
        r2 = RC.run_fold(spec, fold, 0, corpus, rh.root, **kw)
        assert r2["status"] == "fitted", r2
        assert set(r2["refit_partial_removed"]) == expect_removed, r2
        assert all(p.exists() for pq_js in files.values() for p in pq_js)
        assert all(D.read_record(js) is not None for _pq, js in files.values())
        return r2

    n_b6, n_r0 = files["B6"][1].name, files["B6r0"][1].name
    p_b6, p_r0 = files["B6"][0].name, files["B6r0"][0].name
    # 1. the sibling arm's record lost (B6r0 JSON + parquet): the fold is missing, B6's complete pair is rewritten with it
    r2 = refit_after(lambda: [files["B6r0"][1].unlink(), files["B6r0"][0].unlink()], {n_b6, p_b6})
    assert r2["refit_missing_arms"] == ["B6r0"]
    # 2. the primary arm's parquet lost: MISSING (the locator and the reader both need it), the orphan JSON removed first
    refit_after(lambda: files["B6"][0].unlink(), {n_b6, n_r0, p_r0})
    # 3. an orphan parquet (JSON never written): removed, the fold refit
    refit_after(lambda: files["B6r0"][1].unlink(), {n_r0, n_b6, p_b6})
    # 4. a TRUNCATED record JSON (V-F2): MISSING, not 'written under no code digest'
    refit_after(lambda: files["B6"][1].write_text(files["B6"][1].read_text(encoding="utf-8")[:200], encoding="utf-8"),
                {n_b6, p_b6, n_r0, p_r0})
    # 5. a record under ANOTHER code digest: refused, nothing deleted, nothing re-fitted (addendum 8 item 2)
    rec = json.loads(files["B6r0"][1].read_text(encoding="utf-8"))
    rec["code_digest"] = "f" * 64
    files["B6r0"][1].write_text(json.dumps(rec), encoding="utf-8")
    before = {p: p.stat().st_mtime_ns for pq_js in files.values() for p in pq_js}
    with pytest.raises(CF.RedactedRunError, match="different code digest"):
        RC.run_fold(spec, fold, 0, corpus, rh.root, **kw)
    assert {p: p.stat().st_mtime_ns for pq_js in files.values() for p in pq_js} == before
    for blob in (json.dumps(r2["refit_partial_removed"]), json.dumps(r2["refit_missing_arms"]), json.dumps(rec),
                 str(sorted(str(p) for pq_js in files.values() for p in pq_js))):
        assert _no_seed_in(blob)


def test_a_consumed_sibling_record_rewritten_after_the_fact_is_refused_at_dispatch_and_at_assembly(tmp_path, monkeypatch,
                                                                                                    corpus_dir):
    """Task X finding V-F4: ``m1_record_digest`` / ``with_record_digest`` were written 'so the pairing is auditable' but
    never audited.  Now :func:`RC.pairing_audit` recomputes the sibling's digest at dispatch (a skipped fold) and at
    assembly (``read_seed_predictions``): a WITH record rewritten consistently (label and nested block agree, so it still
    verifies) after C4's control leg consumed it is refused on both paths, and an M2 record is audited the same way
    against synthetic M1 records."""
    rh = Rehearsal(tmp_path, monkeypatch, corpus_dir, "audit")
    rd = RC.runner_module()
    corpus, store, code = rh.corpus, rh.store, rh.code
    fold = next(f for f in corpus.folds("V2__element__exact") if f.fold_id == "La(III)")
    plan = CF.read_plan(CF.plan_path(rh.root))
    legs = [j for j in CF.enumerate_jobs(plan) if j.purpose.startswith("claim C4") and j.seed_index == 1]
    with_spec, perm_spec = (RC.job_spec_of(j, store.seed(1)) for j in legs)
    kw = dict(store=store, seed_index=1, runners=rd.default_runners(), code=code, state=D.PlanState(guard_mode={}),
              prereg={"addenda_sha256": "a" * 64, "n_addenda": 8})
    RC.run_fold(with_spec, fold, 0, corpus, rh.root, **kw)
    r_perm = RC.run_fold(perm_spec, fold, 0, corpus, rh.root, **kw)
    js_with = CF.fold_paths(rh.root, "B6", with_spec.design_dir, 1, fold.fold_id, "WITH")[1]
    assert RC.run_fold(perm_spec, fold, 0, corpus, rh.root, **kw)["status"] == "skipped_done"     # audit holds
    RC.INCOMPLETE_RECORD_SETS.clear()
    assert RC.read_seed_predictions(rh.root, "B6", perm_spec.design_dir, 1, transform="ACT_PERMUTED") is not None
    # the WITH record rewritten CONSISTENTLY (a refit that selected another lambda): it still verifies on its own ...
    w = json.loads(js_with.read_text(encoding="utf-8"))
    sel = w["arm_record"]["selection"]
    sel["selected_lambda"] = float(sel["selected_lambda"]) * 10.0
    w["selected_config"] = f"k{int(sel['selected_rank'])}_lam{float(sel['selected_lambda']):g}"
    js_with.write_text(json.dumps(w), encoding="utf-8")
    RC.sibling_consistency("B6", w, what="t")                                       # ... consistent, so not stale by itself
    # ... but the control leg consumed the OLD values: refused at dispatch, naming the record to remove
    with pytest.raises(CF.RedactedRunError, match="with_pairing.with_record_digest .* is not the digest .* of the WITH record"):
        RC.run_fold(perm_spec, fold, 0, corpus, rh.root, **kw)
    # ... and at assembly
    with pytest.raises(D.StaleRecordError, match="with_pairing.with_record_digest"):
        RC.read_seed_predictions(rh.root, "B6", perm_spec.design_dir, 1, transform="ACT_PERMUTED")
    # the audited record itself is untouched by both refusals
    assert Path(r_perm["records"]["B6"]).exists()
    # a MISSING sibling: also refused (the pairing cannot be audited)
    js_with.unlink()
    with pytest.raises(CF.RedactedRunError, match="missing or unreadable"):
        RC.run_fold(perm_spec, fold, 0, corpus, rh.root, **kw)

    # ---- the M2 <- M1 pairing, on synthetic records (the neural fits are the slow test's)
    d = tmp_path / "audit_m2"
    m1_pq, m1_js = CF.fold_paths(d, "M1", "V5__primary_batched_max4", 2, "si2_C_b000")
    m2_pq, m2_js = CF.fold_paths(d, "M2", "V5__primary_batched_max4", 2, "si2_C_b000")
    m1 = {"schema": CF.SCHEMA, "registry_stage": CF.STAGE, "arm": "M1", "stem": "V5__primary__batched_max4",
          "fold_id": "si2_C_b000", "fold_hash": "h", "seed_index": 2, "code_digest": code, "selected_config": "e16_wd1e-05",
          "model_seed": 42, "arm_record": {"selected": {"emb_dim": 16, "weight_decay": 1e-5}, "n_epochs": 40,
                                           "model_seed": 42, "inner_folds_used": [0, 1, 2]}}
    from gen19ct.manifest import write_json as _wj
    _wj(m1_js, m1)
    m2 = {"arm": "M2", "m2_pairing": {"m1_record_digest": CF.confirmation_record_digest(m1)}}
    assert RC.pairing_audit(m2, out_root=d, design_dir="V5__primary_batched_max4", seed_index=2, fold_id="si2_C_b000") is None
    m1b = json.loads(json.dumps(m1))
    m1b["arm_record"]["selected"]["emb_dim"] = 32                                    # the VALUES M2 consumes, relabelled
    m1b["selected_config"] = "e32_wd1e-05"
    _wj(m1_js, m1b)
    reason = RC.pairing_audit(m2, out_root=d, design_dir="V5__primary_batched_max4", seed_index=2, fold_id="si2_C_b000")
    assert reason and "m2_pairing.m1_record_digest" in reason and "records/M1/V5__primary_batched_max4/i2" in reason
    m1_js.unlink()
    reason = RC.pairing_audit(m2, out_root=d, design_dir="V5__primary_batched_max4", seed_index=2, fold_id="si2_C_b000")
    assert reason and "missing or unreadable" in reason
    assert RC.pairing_audit({"arm": "B0"}, out_root=d, design_dir="x", seed_index=1, fold_id="f") is None   # nothing to audit


def test_the_pool_preflight_requires_every_worker_to_serve_a_probe(tmp_path, monkeypatch, corpus_dir):
    """Task X finding V-F3: all 2 x workers probes were served by whichever worker initialised first, and the log line
    'pool pre-flight: 1 worker process(es)' under --workers 2 was what had happened.  Now each probe HOLDS its worker and
    the loop insists on ``workers`` distinct pids, refusing when they never appear -- so the second worker's store
    verification and corpus load are exercised before the first fold is dispatched, not on it."""
    rh = Rehearsal(tmp_path, monkeypatch, corpus_dir, "preflight")
    rd = RC.runner_module()
    corpus, store, code = rh.corpus, rh.store, rh.code
    RC.CW.corpus, RC.CW.runners, RC.CW.store, RC.CW.attached, RC.CW.pair_attrs = corpus, rd.default_runners(), store, 0, rh.attrs
    holds = []

    class InlinePool:
        def __init__(self, **kw):
            pass

        def submit(self, fn, *a):
            f = Future()
            f.set_running_or_notify_cancel()
            if fn is RC._conf_worker_state:
                holds.append(a[0])
            f.set_result(fn(*a))
            return f

        def shutdown(self, wait=True):
            pass

    real_state = RC._conf_worker_state
    monkeypatch.setattr(RC, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(RC, "PREFLIGHT_HOLDS", (0.0, 0.0))
    monkeypatch.setattr(RC, "_conf_worker_state",                                  # ONE worker serves every probe
                        lambda *a: {"pid": 7, "n_rows": int(corpus.table.n), "n_runners": 1, "store_verified": True,
                                    "seed_commitment_sha256": store.digest, "n_seed_indices": CF.N_SEEDS})
    b0 = CF.ConfJob("claim comparators (closed form)", "B0", "V5", "V5__strict__exact", 0, 2)
    kw = dict(corpus=corpus, state=D.PlanState(guard_mode={}), runners=rd.default_runners(), code=code,
              prereg={"addenda_sha256": "a" * 64, "n_addenda": 8}, workers=2, seed_store_path=str(rh.store_path))
    with pytest.raises(SystemExit, match="only 1 of 2 worker process"):
        RC.fit_loop([b0], store, rh.root, **kw)
    assert holds == [0.0, 0.0, 0.0, 0.0] and not (CF.conf_root(rh.root) / "records").exists()   # 2 rounds x 2 probes, no fold
    # the real function accepts the hold and sleeps for it
    RC.CW.store = store
    t0 = time.perf_counter()
    st = real_state(0.2)
    assert time.perf_counter() - t0 >= 0.2 and st["store_verified"] and st["pid"] == os.getpid()


@pytest.mark.slow
def test_on_a_v6_fold_m1_is_tuned_per_section_7_and_m2_takes_m1s_same_fold_record(tmp_path, monkeypatch, corpus_dir):
    """Blocker 4, the reading of section 3.4's 'frozen configurations' (``CF.READINGS['v6_frozen_configurations']``),
    measured on the synthetic V6 fold: M1 tunes with the V5 inner design (addendum 7 item 2), M2 takes M1's retained
    configuration from the M1 record of the SAME V6 fold and seed index (addendum 8 item 1; ``m2_pairing`` names the
    record's digest), and both records carry the reading."""
    from gen19ct.models import neural as NN

    rh = Rehearsal(tmp_path, monkeypatch, corpus_dir, "v6")
    rd = RC.runner_module()
    corpus, store, code = rh.corpus, rh.store, rh.code
    (v6_fold,), _ = RS.build_v6_folds(RS.fold_corpus_stub(corpus.frame))
    d = RC.seed_folds_dir(rh.root, 1)
    FI.write_design([v6_fold], d)
    RC.attach_seed_folds(corpus, rh.root, 1)
    calls = {"tune_m1": 0, "tune_m2": 0}

    def counted(name, fn):
        def g(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return g

    monkeypatch.setattr(NN, "tune_m1", counted("tune_m1", NN.tune_m1))
    monkeypatch.setattr(NN, "tune_m2", counted("tune_m2", NN.tune_m2))
    kw = dict(store=store, seed_index=1, runners=rd.default_runners(), code=code, state=D.PlanState(guard_mode={}),
              prereg={"addenda_sha256": "a" * 64, "n_addenda": 8}, pair_attrs=rh.attrs)
    specs = {a: D.JobSpec(kind="fit", arm=a, design=CF.V6_DESIGN, variant=CF.V6_VARIANT_NAME, scheme=CF.V6_SCHEME,
                          seed=store.seed(1), writes=(a,)) for a in ("M1", "M2")}
    r1 = RC.run_fold(specs["M1"], v6_fold, 0, corpus, rh.root, **kw)
    r2 = RC.run_fold(specs["M2"], v6_fold, 0, corpus, rh.root, **kw)
    m1 = json.loads(Path(r1["records"]["M1"]).read_text(encoding="utf-8"))
    m2 = json.loads(Path(r2["records"]["M2"]).read_text(encoding="utf-8"))
    assert calls == {"tune_m1": 1, "tune_m2": 1}                                   # section 7 tuning, both arms
    assert len({f.get("config") for f in m1["arm_record"]["fits"]}) > 1              # M1: a grid, on the V6 fold
    assert m1["arm_record"]["inner_design"]["design"] == D.INNER_DESIGN_NAME          # the V5 inner design
    assert m2["m2_pairing"]["m1_record_digest"] == CF.confirmation_record_digest(m1)  # M1's SAME-fold record
    assert m2["arm_record"]["m1_config_used"]["emb_dim"] == m1["arm_record"]["selected"]["emb_dim"]
    assert m2["m2_pairing"]["m1_model_seed"] == m2["m2_pairing"]["m2_model_seed"]
    for rec in (m1, m2):
        assert rec["v6_configuration_reading"] == CF.READINGS["v6_frozen_configurations"]
        assert rec["half"] == "NA" and rec["seed_index"] == 1 and rec["design_hash"] == corpus.design_hash(CF.V6_STEM)
        assert _no_seed_in(json.dumps(rec))


# --------------------------------------------------------------------------------------------- #
# the whole run
# --------------------------------------------------------------------------------------------- #

@pytest.mark.slow
def test_the_whole_confirmation_run_rehearsed_end_to_end(tmp_path, monkeypatch, corpus_dir, capsys):
    """``main()`` -- not the library pieces -- on the synthetic corpus with a fake five-seed store, ``--workers 2``:
    the gates, the lock, the withheld-seed fold files built inside the run, the fit loop over EVERY arm the real plan
    needs (M1, M2, B6 / B6r0, B0, B3i, B3x, B8, B6:WITH, B6:ACT_PERMUTED) in real spawned workers, the assembly (R19
    seed-mean per addendum 6, S1(a)-(d), S2(a)-(c)), the writers and the seed revelation; then every lock and resume
    case of addendum 8 item 2 on copies of the finished run.  Nothing under ``generations/gen19_chem_transfer`` is
    written: ``--out-root``, the registry and the manifests directory are all under ``tmp_path``."""
    rh = Rehearsal(tmp_path, monkeypatch, corpus_dir, "run")
    plan = CF.read_plan(CF.plan_path(rh.root))
    jobs = CF.enumerate_jobs(plan)
    assert plan["claim_ids"] == ["C1", "C2", "C3", "C4"]

    # ---- A. the run itself
    assert rh.main() == 0
    out = capsys.readouterr()
    log = out.out + out.err
    assert "the once-only lock is now SPENT" in log and "post-run scan" in log and "pool pre-flight" in log
    assert _no_seed_in(log)                                                      # no seed in any log line
    body = rh.decisions()
    report = CF.report_path(rh.root)
    assert report.exists() and all(str(s) in report.read_text(encoding="utf-8") for s in FAKE_SEEDS)
    assert _seed_free_tree(rh.root, report) == [] and _seed_free_tree(rh.manifests_out) == []
    fit = body["fit"]
    assert fit["workers_used"] == 2 and fit["n_errors"] == 0 and fit["halted_by_max_folds"] is False
    assert fit["worker_preflight"] and len({p["pid"] for p in fit["worker_preflight"]}) >= 1
    assert os.getpid() not in {p["pid"] for p in fit["worker_preflight"]}      # the folds ran in CHILD processes
    n_total = fit["n_folds_fitted"]
    assert n_total > 0 and fit["n_folds_skipped_already_done"] == 0
    assert set(fit["by_job"]) == {j.key for j in jobs}                          # every job of the plan ran
    # every arm the real plan needs left records, in distinct directories
    arm_dirs = {p.relative_to(CF.conf_root(rh.root) / "records").parts[:2] for p in rh.records()}
    assert {("M1",), ("M2",), ("B6",), ("B6r0",), ("B0",), ("B3i",), ("B3x",), ("B8",)} <= {(a[0],) for a in arm_dirs}
    assert ("B6", "WITH") in arm_dirs and ("B6", "ACT_PERMUTED") in arm_dirs
    # the fold files: one file per (stem, seed index), seed-dependent colourings, the V6 design once per index
    ff = body["fold_files"]
    assert len(ff) == 5 * (len(RC.BATCHED_STEMS) + 1)
    hashes = {(f["stem"], f["design_hash"]) for f in ff}
    assert len({h for s, h in hashes if s == RS.V5_STEM}) == 5                   # five colourings, all different
    # every claim scored, on 5 of 5 seeds
    claims = {c["claim_id"]: c for c in body["claims"]}
    assert set(claims) == {"C1", "C2", "C3", "C4"}
    for c in claims.values():
        assert c["status"] == "COMPLETE" and c["r19_verdict"] in ("PASS", "FAIL") and c["item4"]["n_seeds_scored"] == 5
        assert np.isfinite(c["point"]) and len(c["seed_deltas_by_index"]) == 5
    assert claims["C4"]["point"] != 0.0 and all(v != 0.0 for v in claims["C4"]["seed_deltas_by_index"].values())
    assert claims["C4"]["n_clusters"] == {"metal_state": 3}                      # the three Ln(III) V2 folds, not Am(III)
    assert body["s1c"]["status"] == "COMPLETE" and body["s1c"]["verdict"] in ("PASS", "FAIL")
    assert body["s1c"]["counterweight_selection_half"]["status"] == "READ"
    assert body["s1d"]["status"] in ("PASS", "FAIL")
    assert body["s2"]["status"] == "COMPLETE" and body["s2"]["verdict"] in ("PASS", "FAIL")
    for part in ("s2a", "s2b", "s2c"):
        assert body["s2"][part]["status"] in ("PASS", "FAIL"), part
    assert body["s1"]["verdict"] in ("PASS", "FAIL", CF.UNDECIDED)
    assert body["not_run"]["v6_actinide_deltas"]["status"] == CF.NOT_RUN
    # the C4 control records carry the transform, the guard rule and the WITH digest
    perm = [json.loads(p.read_text(encoding="utf-8")) for p in rh.records() if "ACT_PERMUTED" in p.parts]
    assert perm and all(r["n_training_rows_changed"] > 0 and r["guard_value_source"] == "recorded_log_D" for r in perm)
    assert all(r["with_record_digest"] for r in perm)
    # the manifest of the run, scrubbed and in the test's manifests directory
    mf = json.loads((rh.manifests_out / f"{RC.NAME}.json").read_text(encoding="utf-8"))
    assert mf["stage"] == CF.STAGE and mf["claims"] == ["C1", "C2", "C3", "C4"] and "seed_store" not in mf["arguments"]
    assert mf["incomplete_record_sets"] == [] and mf["fit"]["n_errors"] == 0    # nothing left incomplete

    # ---- B. the finished run is SPENT: refused with and without --resume (addendum 8 item 2)
    for extra in ((), ("--resume",)):
        with pytest.raises(SystemExit) as exc:
            rh.main(*extra)
        assert "spent" in str(exc.value) or "ONCE" in str(exc.value)
    assert rh.decisions()["fit"]["n_folds_fitted"] == n_total                  # nothing was rewritten

    # ---- C. an UNFINISHED run: --resume fits ONLY the missing folds, then finishes
    trunc = rh.copy_run("truncated", finished=False)
    victims = [p for p in sorted((CF.conf_root(trunc) / "records").rglob("*.json"))
               if p.parts[-4:-2] == ("M2", "V5__primary_batched_max4") and p.parent.name == "i2"][:1]
    victims += [p for p in sorted((CF.conf_root(trunc) / "records").rglob("*.json"))
                if p.parts[-4:-2] == ("B3x", RC.V6_DESIGN_DIR) and p.parent.name == "i3"][:1]
    assert len(victims) == 2
    for v in victims:
        v.unlink()
        v.with_suffix(".parquet").unlink()
    with pytest.raises(SystemExit, match="STARTED"):
        rh.main(root=trunc)                                                    # no --resume: refused
    assert rh.main("--resume", root=trunc) == 0
    fit2 = rh.decisions(trunc)["fit"]
    assert fit2["n_folds_fitted"] == 2 and fit2["n_folds_skipped_already_done"] == n_total - 2 and fit2["n_errors"] == 0
    assert {c["status"] for c in rh.decisions(trunc)["claims"]} == {"COMPLETE"}
    assert _seed_free_tree(trunc, CF.report_path(trunc)) == []

    # ---- D. a MIXED-CODE record set is refused before anything is fitted or scored
    mixed = rh.copy_run("mixed", finished=False)
    first = min(p for p in (CF.conf_root(mixed) / "records" / "M1" / "V5__primary_batched_max4" / "i1").glob("*.json"))
    rec = json.loads(first.read_text(encoding="utf-8"))
    rec["code_digest"] = "f" * 64
    first.write_text(json.dumps(rec), encoding="utf-8")
    n_before = len(list((CF.conf_root(mixed) / "records").rglob("*.json")))
    with pytest.raises(CF.RedactedRunError, match="different code digest"):
        rh.main("--resume", root=mixed, workers=1)
    assert len(list((CF.conf_root(mixed) / "records").rglob("*.json"))) == n_before
    assert not CF.decisions_path(mixed).exists()

    # ---- E. a missing M1 record: exactly ONE incomplete M2 fold, the run continues to the end
    miss = rh.copy_run("m1_missing", finished=False)
    m1s = sorted((CF.conf_root(miss) / "records" / "M1" / "V5__primary_batched_max4" / "i1").glob("*.json"))
    target = m1s[0].stem
    for arm in ("M1", "M2"):
        p = CF.conf_root(miss) / "records" / arm / "V5__primary_batched_max4" / "i1" / f"{target}.json"
        p.unlink()
        p.with_suffix(".parquet").unlink()
    real_run_fold = RC.run_fold

    def m1_guard_fails(job, fold, *a, **k):
        if (job.arm == "M1" and job.stem == "V5__primary__batched_max4" and k.get("seed_index") == 1
                and CF.scrub(fold.fold_id, rh.store) == target):     # this ONE fold (the V5-PAIR colouring reuses the id)
            raise AssertionError("the guard of this M1 fold failed")       # so M1's record never appears
        return real_run_fold(job, fold, *a, **k)

    monkeypatch.setattr(RC, "run_fold", m1_guard_fails)
    assert rh.main("--resume", root=miss, workers=1) == 0
    monkeypatch.setattr(RC, "run_fold", real_run_fold)
    b = rh.decisions(miss)
    statuses = sorted(e["status"] for e in b["fit"]["errors"])
    assert statuses == sorted([CF.INCOMPLETE_GUARD_FAILURE, CF.INCOMPLETE_M1_RECORD_MISSING])
    assert b["fit"]["n_errors"] == 2 and b["fit"]["n_folds_fitted"] == 0
    c = {x["claim_id"]: x for x in b["claims"]}
    assert all(c[k]["status"] == "INCOMPLETE" and c[k]["seed_indices_missing"] == [1] for k in ("C1", "C2", "C3"))
    assert c["C4"]["status"] == "COMPLETE"                                     # the completeness unit is the contrast set
    mf_e = json.loads((rh.manifests_out / f"{RC.NAME}.json").read_text(encoding="utf-8"))   # named in the run manifest
    assert {(m["arm"], m["seed_index"]) for m in mf_e["incomplete_record_sets"]} == {("M1", 1), ("M2", 1)}
    assert CF.report_path(miss).exists()

    # ---- F. --max-folds is a PAUSE on a fresh run: nothing assembled, the lock at the start marker, resumable
    fresh = tmp_path / "paused"
    RS.write_run_skeleton(fresh, rh.store_path)
    assert rh.main("--max-folds", "1", root=fresh, workers=1) == 0
    log = capsys.readouterr().out
    assert "PAUSED by --max-folds 1" in log and _no_seed_in(log)
    assert CF.started_path(fresh).exists() and not CF.decisions_path(fresh).exists()
    assert not CF.report_path(fresh).exists() and not CF.tables_dir(fresh).exists()
    assert 1 <= len(list((CF.conf_root(fresh) / "records").rglob("*.json"))) <= 2      # M1 writes one record per fold
    lock = CF.lock_verdict(fresh, resume=False, code_digest=rh.code, claim_ids=plan["claim_ids"])
    assert lock["ok"] is False and lock["mode"] == "already_started"
    lock = CF.lock_verdict(fresh, resume=True, code_digest=rh.code, claim_ids=plan["claim_ids"])
    assert lock["ok"] is True and lock["mode"] == "resume_unfinished"
    assert _seed_free_tree(fresh) == []
