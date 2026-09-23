"""``scripts/g19_run_h3.py`` -- the section 11 actinide ablation (H3; brief sections 15, 30) after discovery.

Refuses to start unless, in this order:

1. ``scripts/g19_seal_prereg.py --check`` exits 0 AND the sealed text is the registered digest with POST-HOC addendum 1
   (``g19_run_discovery.refuse_unless_sealed``);
2. the discovery run is **COMPLETE** (``gen19ct.evaluation.h3.discovery_complete``): ``evaluation/discovery/decisions/
   wall_clock.json`` records the final plan stage ``10_not_implemented`` (the ``M3+`` marker the runner stops at) and
   every stage of the current plan, and every ``fit`` job of the plan has a verified COMPLETE record set for every arm it
   writes (digest, fold hash and fold set exactly what the current code, fold files and plan state produce);
3. the scorer's decision files exist and ``decisions.json`` is newer than every fold record
   (``h3.scorer_decisions_present``), because the deployed configuration is read from its ladder and stop-rule verdicts;
4. the ladder runner has run every step M3-M7 to a done / skipped status (``h3.ladder_complete`` on
   ``evaluation/ladder/decisions/ladder.json``), because section 11's arm is "the retained ladder configuration", which
   only the ladder decides (task X finding V-01).  Gate 0, run before anything heavy is loaded: ``wall_clock.json``
   reached the final stage (``h3.refuse_unless_cheap_complete``; finding VL2-04).

Then, for the configuration deployed for lanthanide prediction -- the ladder's RETAINED configuration: the highest kept
ladder step M7 > ... > M3 > M2 > M1 > M0 (= B5; POST-HOC addendum 3 item 2), and only when the ladder retains no step at
all M2 / B6 by the stop rule, else B3i (``h3.deployed_configuration``) -- plus B6 and B5:

* WITH is the discovery record of the same arm, design, seed and fold -- nothing is refitted for it (a ladder step's
  WITH record is its ``evaluation/ladder`` record, verified against the digests the current code and ladder decisions
  produce; a closed-form deployed arm has no discovery record and is fitted here, on the exact leave-one-cell-out folds
  of section 3.1 as in discovery);
* WITHOUT and ACT_PERMUTED are refitted at the WITH run's SELECTED hyperparameters of the same fold
  (``h3.frozen_runner``; no re-tuning) on the same folds, batches and seeds, with the cross-fitted split-conformal
  calibration of the discovery runner.  **POST-HOC addendum 4 item 1**: ACT_METAL_SHUFFLED is NOT planned, fitted or
  scored (``h3.TRANSFORMS``); the one fold fitted before the addendum stays on disk as ``exploratory_not_scored``
  (``h3.exploratory_records``, written to ``evaluation/h3/decisions/h3_decisions.json``);
* **POST-HOC addendum 4 item 2** -- the job plan runs in the priority order V5 -> V2 -> V1 (``h3.order_by_priority``)
  under a cap of 20 h of wall clock on at most 2 workers, ledgered in ``evaluation/h3/decisions/wall_clock.json``
  (``h3.h3_wall_clock``, ``h3.record_h3_wall_clock``) and **checked before each fold**.  A design the cap does not reach
  is written ``NOT_RUN`` with its reason (``h3.not_run_design``) and no verdict is taken from a partial design
  (``h3.assert_no_partial_design``, asserted in :func:`score`);
* **POST-HOC addendum 4 item 3** -- a record written after the ``h3`` registry entry it carries was superseded does not
  verify (``registry.verify_record`` -> ``stale_after_supersession``) and is refitted under the current registered
  digest; a record written EARLIER keeps its entry and verifies as registered, so the digest an existing record is
  checked against is the pair it was written under (``h3.record_digest_basis``), never the live pair.
  ``scripts/g19_h3_refit_stale.py`` lists and deletes exactly the stale records before this runner refits them;
* the Ln(III) scored rows of the V5-primary, V2 and V1 selection folds are scored, and the deltas (macro MAE under R19,
  rank accuracy, calibration, logSF MAE on V5-PAIR Ln-Ln pairs) are written with the section 8 paired cluster bootstrap;
* the section 11 verdicts, the F4 failure condition and the descriptive negative-transfer tables follow;
* **POST-HOC addendum 2 item 4** -- section 11's last bullet, "the WITH arm re-run with a shared-only metal embedding (no
  ``e_series``, ``e_ox``) against the full section 15 embedding": the deployed arm is refitted with
  ``models.shared_only`` (``h3.FrozenSharedOnly``: the WITH record's SELECTED hyperparameters, stopping count and model
  seed, the WITH training rows, the cross-fitted calibration) ONLY when ``h3.shared_only_condition`` holds on the deployed
  arm's verdict -- section 11's "run if WITHOUT is better, point estimate or passed": ``hurts == true`` (WITHOUT passes
  R19 on the freezing-screen scope against WITH on the V5 Ln cells) OR ``v5_without_vs_with_point > 0`` (the reversed V5
  contrast's Delta macro MAE, MAE(WITH) - MAE(WITHOUT)).  Its records live under ``evaluation/h3/records/<arm>/SHARED_ONLY``,
  its contrasts (Delta = MAE(shared-only) - MAE(WITH), exploratory BH family) under ``evaluation/h3/shared_only/``, and D03
  says ``not run (condition not met)`` otherwise;
* ``decisions/D03_actinide_transfer.md`` is generated from those files (brief section 29 format, "not computed" where a
  quantity is absent).

Digest registry (addendum 2 item 5): the seal gate and the discovery code digest prefer ``manifests/digest_registry.json``
(``gen19ct.evaluation.registry``, stage ``h3``) when it exists and fall back to the constants of ``evaluation.discovery``
otherwise, so nothing changes until the registry is created.

Outputs (new directories only): ``evaluation/h3/`` (records, contrasts, deltas, per-unit deltas, summary, verdicts, F4,
negative_transfer/), ``tables/h3_*.csv`` and ``decisions/D03_actinide_transfer.md``.  Every run is wrapped in
``gen19ct.manifest.Run``: git head, the pre-registration and addenda digests, the code digest of THIS script and its
module, the seeds and the runtime.

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_h3.py [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import h3 as H3  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.evaluation import transfer as ET  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.manifest import Run, git_head, write_csv, write_json, write_text  # noqa: E402
from gen19ct.models import interface as I  # noqa: E402

NAME = "g19_run_h3"
#: the registry stage of this runner (addendum 2 item 5)
STAGE = "h3"
#: the files whose content changes an H3 prediction, beyond the discovery code digest every record already carries
#: (``models/shared_only.py``: the shared-only embedding of addendum 2 item 4)
CODE_FILES: tuple[Path, ...] = (paths.G19_ROOT / "gen19ct" / "evaluation" / "h3.py",
                                paths.G19_ROOT / "gen19ct" / "models" / "shared_only.py", Path(__file__).resolve())
CODE_OBJECTS: tuple[Any, ...] = (H3.FrozenNeural, H3.FrozenBoosted, H3.FrozenB6, H3.FrozenComparator,
                                 H3.transformed_frame, H3.without_mask, H3.selected_hyperparameters,
                                 H3.b6_cross_fit_configs, H3.fold_digest, H3.ln_rows, H3.FrozenSharedOnly,
                                 H3.shared_only_job, H3.shared_only_condition)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def runner_module():
    return _script("g19_run_discovery")


def scorer_module():
    return _script("g19_score_discovery")


# ============================================================================================= #
# the gate
# ============================================================================================= #

def code_digest() -> dict[str, Any]:
    """The H3 code digest: the discovery code digest (which every WITH record carries) plus this script and its module."""
    rd = runner_module()
    return D.code_digest(rd.CODE_FILES + (Path(rd.__file__).resolve(),) + CODE_FILES, rd.RUNNER_OBJECTS + CODE_OBJECTS)


def refuse_unless_ready(out_root: Path, *, check: Callable[[], int] | None = None,
                        digests: Callable[[], Mapping[str, Any]] | None = None, expect_addenda: int | None = None,
                        excluded_ids: Sequence[str] = (), folds_dir: Path | None = None,
                        discovery_code: str | None = None, runners: Mapping[str, Any] | None = None,
                        jobs: Sequence[D.JobSpec] | None = None, state: D.PlanState | None = None,
                        require_ladder: bool = True) -> dict[str, Any]:
    """The four gates of the module docstring; raises ``SystemExit`` with what is missing.  ``require_ladder=False`` is
    for tests of the earlier gates only: the runner always requires the ladder."""
    rd = runner_module()
    # addendum 2 item 5: the registry's expectations for stage 'h3' when manifests/digest_registry.json exists, the
    # constants (g19_run_discovery.refuse_unless_sealed) otherwise; likewise the code digest discovery records are
    # verified with (the registry's discovery entry, else the live current_code_digest())
    prereg = REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=expect_addenda)
    st = state if state is not None else D.PlanState.read(rd.plan_state_path(out_root))
    code = discovery_code if discovery_code is not None else REG.discovery_code_digest()
    done = H3.discovery_complete(out_root, code=code, state=st, excluded_ids=excluded_ids, folds_dir=folds_dir,
                                 runners=runners, jobs=jobs)
    if not done["complete"]:
        raise SystemExit("refused: the discovery run is not COMPLETE, and section 11 compares against its records "
                         f"(reached final stage {done['final_stage']}: {done['reached_final_stage']}; missing stages "
                         f"{done['missing_stages']}; {done['n_incomplete']} of {done['n_fit_record_sets']} fit record "
                         f"sets incomplete, first: {done['incomplete_record_sets'][:3]}). Let "
                         "scripts/g19_run_discovery.py finish, then rerun")
    sc = H3.scorer_decisions_present(out_root)
    if not sc["ok"]:
        raise SystemExit("refused: the scorer's decision files are missing or older than a fold record, and the deployed "
                         f"configuration is read from them (present: {sc['present']}; decisions.json newer than every "
                         f"record: {sc['decisions_newer_than_every_record']}). Run scripts/g19_score_discovery.py, then "
                         "rerun")
    lc = H3.ladder_complete(H3.read_ladder_state(out_root))
    if require_ladder and not lc["complete"]:
        raise SystemExit("refused: the ladder (M3-M7) is not complete -- section 11's arm is 'the retained ladder "
                         f"configuration', which only scripts/g19_run_ladder.py decides ({H3.LADDER_STATE_FILE} "
                         f"{'absent' if not lc['present'] else 'steps not done: ' + str(lc['not_done'])}; definition: "
                         f"{lc['definition']}). Run the ladder to completion, then rerun")
    return {"prereg_gate": prereg, "discovery_complete": done, "scorer_decisions": sc, "ladder_complete": lc,
            "plan_state": st.record(), "discovery_code_sha256": code}


def prior_legs(out_root: Path) -> dict[str, Any]:
    """What the H3 legs already on disk cost, measured from their own fold records (``h3.measure_leg_seconds``).

    POST-HOC addendum 4 item 2 grants the 20 h cap after the deployed arm's V5 leg was fitted ("V5 (done), then V2,
    then V1"), so these hours are REPORTED in the ledger and not counted against the cap."""
    root = H3.h3_root(out_root) / "records"
    out: dict[str, Any] = {}
    for d in sorted(root.glob("*/*/*/s*")) if root.exists() else []:
        if not d.is_dir():
            continue
        arm, transform, design_dir = d.parts[-4], d.parts[-3], d.parts[-2]
        out[f"{arm}:{transform}@{design_dir}"] = {
            **H3.measure_leg_seconds(out_root, model_arm=arm, transform=transform, design_dir=design_dir,
                                     seed=int(d.name[1:])),
            "scored": H3.is_scored_transform(transform)}
    out["total_hours"] = round(sum(float(v["total_hours"]) for v in out.values() if isinstance(v, Mapping)), 4)
    return out


def read_decisions(out_root: Path) -> dict[str, Any]:
    body = D.read_record(D.discovery_root(out_root) / "decisions" / "decisions.json")
    if body is None:
        raise SystemExit("refused: evaluation/discovery/decisions/decisions.json is unreadable")
    return body


def read_ladder(out_root: Path) -> dict[str, Any] | None:
    """``evaluation/ladder/decisions/ladder.json`` (the deployed configuration reads it; ``None`` when absent)."""
    return H3.read_ladder_state(out_root)


def ladder_context(out_root: Path) -> tuple[Any, Any, dict[str, str]]:
    """The ladder runner module, its resumable state and its code digests -- what a ladder step's WITH records are
    verified with (``g19_run_ladder.verified_ladder_predictions``)."""
    RL = H3._ladder_module()
    lstate = RL.LadderState.read(RL.ladder_state_path(out_root))
    if lstate is None:
        raise SystemExit(f"refused: {H3.LADDER_STATE_FILE} is unreadable, yet the ladder gate passed; rerun the gate")
    return RL, lstate, RL.ladder_code_digest()


def verify_ladder_with_records(out_root: Path, arm: str, state: D.PlanState, *, excluded_ids: Sequence[str],
                               designs: Sequence[str] = tuple(H3.DESIGNS), folds_dir: Path | None = None,
                               seed: int = D.PRIMARY_SEED) -> dict[str, Any]:
    """The deployed ladder step's WITH record sets (one per design) verified COMPLETE against the digests the current
    code, fold files, plan state and ladder decisions produce; ``SystemExit`` otherwise (a stale record raises)."""
    RL, lstate, codes = ladder_context(out_root)
    out: dict[str, Any] = {}
    for design in designs:
        job = H3.with_job(arm, design, state, seed=seed)
        _, st = RL.verified_ladder_predictions(out_root, arm, job.design_dir, int(job.seed), code=codes["discovery"],
                                               state=state, excluded_ids=list(excluded_ids),
                                               runner=RL.LadderRunner(arm, lstate, codes), folds_dir=folds_dir,
                                               steps=("point",))
        out[design] = {k: v for k, v in st.items() if k != "arm"}
        if st.get("status") != "complete":
            raise SystemExit(f"refused: the WITH records of the deployed ladder step {arm} on {design} "
                             f"({job.design_dir}/s{job.seed}) are {st.get('status')}: {st}; rerun scripts/g19_run_ladder.py")
    return out


# ============================================================================================= #
# the plan
# ============================================================================================= #

def plan_jobs(model_arms: Sequence[str], state: D.PlanState, *, designs: Sequence[str] = tuple(H3.DESIGNS),
              seed: int = D.PRIMARY_SEED, with_pairs: bool = False) -> list[dict[str, Any]]:
    """One entry per (model arm, transform, design): the WITH discovery job and the H3 job that refits the transform.

    Returned in the POST-HOC addendum 4 item 2 priority order V5 -> V2 -> V1 (``h3.order_by_priority``), and holding only
    the transforms that are scored -- ACT_METAL_SHUFFLED is not planned (item 1, ``h3.transforms_for``)."""
    out = []
    dsg = list(designs) + ([H3.PAIR_DESIGN] if with_pairs else [])
    for arm in model_arms:
        for design in dsg:
            if design == H3.PAIR_DESIGN and D.ARM_ALIASES.get(arm, arm) != "M2":
                continue                                   # addendum 1 item 5: only M2 runs the V5-PAIR folds
            base = H3.with_job(arm, design, state, seed=seed)
            for transform in H3.transforms_for(arm):
                out.append({"model_arm": D.ARM_ALIASES.get(arm, arm), "transform": transform, "design": design,
                            "with_job": base, "job": H3.h3_job(base, arm, transform)})
    return H3.order_by_priority(out)


def shared_only_plan(deployed_arm: str, state: D.PlanState, *, designs: Sequence[str] = tuple(H3.DESIGNS),
                     seed: int = D.PRIMARY_SEED) -> list[dict[str, Any]]:
    """Addendum 2 item 4: one entry per design of the deployed arm's shared-only re-run (``transform`` SHARED_ONLY); empty
    when the deployed arm has no metal embedding.  Never the pair design (section 11 names the Ln test set only)."""
    arm = D.ARM_ALIASES.get(deployed_arm, deployed_arm)
    if not H3.shared_only_applicable(arm):
        return []
    out = []
    for design in designs:
        base = H3.with_job(arm, design, state, seed=seed)
        out.append({"model_arm": arm, "transform": H3.SHARED_ONLY, "design": design, "with_job": base,
                    "job": H3.shared_only_job(base, arm)})
    return out


def filter_plan(plan: Sequence[Mapping[str, Any]], only: str | None) -> list[dict[str, Any]]:
    if not only:
        return list(plan)
    toks = [t.strip() for t in str(only).split(",") if t.strip()]
    keep = []
    for e in plan:
        labels = {e["model_arm"], e["transform"], e["design"], f"{e['model_arm']}:{e['transform']}"}
        if labels & set(toks):
            keep.append(e)
    return keep


# ============================================================================================= #
# fitting one fold of one transform
# ============================================================================================= #

def with_record(out_root: Path, entry: Mapping[str, Any], fold: FI.Fold) -> dict[str, Any] | None:
    """The WITH discovery record of this arm, design, seed and fold (``None`` for a closed-form arm, which has none)."""
    arm = entry["model_arm"]
    if H3.is_deterministic(arm):
        return None
    job = entry["with_job"]
    if H3.is_ladder_arm(arm):
        _, js = H3._ladder_module().ladder_fold_paths(out_root, job, arm, fold.fold_id)
        what = "ladder"
    else:
        _, js = D.fold_paths(out_root, job, arm, fold.fold_id)
        what = "discovery"
    rec = D.read_record(js)
    if rec is None:
        raise SystemExit(f"refused: the WITH {what} record of {arm}/{job.design_dir}/s{job.seed}/{fold.fold_id} is "
                         f"missing, yet the {what} gate passed; rerun the gate")
    return rec


def transformed_corpus(corpus: Any, frame: pd.DataFrame) -> Any:
    """A :class:`g19_run_discovery.Corpus` over a transformed frame (same index, so the V6 mask, the co-extractant flags,
    the registered halves, the condition vectors, the descriptor tables and the fold files are reused unchanged); only
    the frame, its slim view and the ``RowTable`` are rebuilt."""
    rd = runner_module()
    if not frame.index.equals(corpus.frame.index):
        raise AssertionError("a training transform changed the corpus index")
    table = I.RowTable(frame, systems=corpus.systems, components=corpus.comps)
    return rd.Corpus(frame=frame, slim=FI.slim_frame(frame), table=table, v6=corpus.v6, coext=corpus.coext,
                     systems=corpus.systems, comps=corpus.comps, cv=corpus.cv, pmap=corpus.pmap,
                     row_half=corpus.row_half, folds_dir=corpus.folds_dir, index_json=corpus.index_json)


def fold_context(entry: Mapping[str, Any], fold: FI.Fold, ordinal: int, corpus: Any, out_root: Path,
                 state: D.PlanState, *, code: str, guard_fn: Callable | None = None,
                 inner_check: Callable | None = None) -> tuple[Any, dict[str, Any]]:
    """The fold context of one H3 arm: the discovery runner's ``prepare_fold`` (its section 2 guards unchanged) on the
    corpus the transform needs, then the transform's training mask.

    WITHOUT removes every actinide row from the training mask (unknown-state actinide rows included); ACT_PERMUTED and
    ACT_METAL_SHUFFLED transform the fold's TRAINING rows through ``discovery.h3_training_rows`` and rebuild the corpus
    over the transformed frame, so no hidden row and no scored row is touched."""
    rd = runner_module()
    transform = entry["transform"]
    job = entry["job"]
    kw: dict[str, Any] = {}
    if guard_fn is not None:
        kw["guard_fn"] = guard_fn
    if inner_check is not None:
        kw["inner_check"] = inner_check
    fc, info = rd.prepare_fold(job, fold, ordinal, corpus, out_root, state, code=code, **kw)
    info = dict(info)
    info["transform"] = transform
    if transform in ("WITH", H3.SHARED_ONLY):                   # the shared-only re-run trains on the WITH rows unchanged
        return fc, info
    if transform == "WITHOUT":
        mask, n_dropped = H3.without_mask(corpus.frame, fc.mask)
        if not mask.any():
            raise ValueError(f"{job.key}/{fold.fold_id}: WITHOUT leaves no training row")
        info.update(n_actinide_training_rows_dropped=n_dropped, n_train=int(mask.sum()),
                    n_actinide_training_rows=n_dropped)
        return replace(fc, mask=mask), info
    train_index = corpus.table.index[fc.mask]
    frame = H3.transformed_frame(corpus.frame, train_index, transform, seed=int(job.seed))
    tcorpus = transformed_corpus(corpus, frame)
    fc2, info2 = rd.prepare_fold(job, fold, ordinal, tcorpus, out_root, state, code=code, **kw)
    if not np.array_equal(fc2.mask, fc.mask):
        raise AssertionError(f"{job.key}/{fold.fold_id}: the transform changed the fold's training mask")
    if list(fc2.sc_ids) != list(fc.sc_ids):
        raise AssertionError(f"{job.key}/{fold.fold_id}: the transform changed the fold's scored rows")
    an = D.actinide_rows(corpus.frame)
    info = {**dict(info2), "transform": transform,
            "n_actinide_training_rows": int((fc.mask & an).sum()),
            "n_training_rows_changed": int((corpus.frame.loc[train_index, I.TARGET_COL].to_numpy(dtype=float)
                                           != frame.loc[train_index, I.TARGET_COL].to_numpy(dtype=float)).sum())
            if transform == "ACT_PERMUTED" else
            int((corpus.frame.loc[train_index, SG.METAL_COL].astype(object).to_numpy()
                 != frame.loc[train_index, SG.METAL_COL].astype(object).to_numpy()).sum())}
    return fc2, info


def run_fold(entry: Mapping[str, Any], fold: FI.Fold, ordinal: int, corpus: Any, out_root: Path, state: D.PlanState, *,
             code: str, steps: Sequence[str] = H3.STEPS, guard_fn: Callable | None = None,
             inner_check: Callable | None = None, prereg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit (or skip) one fold of one H3 arm and write its prediction parquet and record under ``evaluation/h3``."""
    rd = runner_module()
    arm, transform, job = entry["model_arm"], entry["transform"], entry["job"]
    if not H3.is_scored_transform(transform):
        raise SystemExit(f"refused: {transform} is not run (POST-HOC addendum 4 item 1); the fold already fitted is "
                         f"{H3.EXPLORATORY_NOT_SCORED} and nothing refits it. {H3.ACT_METAL_SHUFFLED_NOT_RUN}")
    wrec = with_record(out_root, entry, fold)
    design_hash = corpus.design_hash(job.stem)
    dg = dict(transform=transform, with_digest=None if wrec is None else wrec.get("digest"),
              guard_mode=rd.guard_mode_for(job, state), design_hash=design_hash, ordinal=ordinal,
              model_seed=rd.model_seed_of(ordinal))
    pq, js = H3.fold_paths(out_root, arm, transform, job, fold.fold_id)
    # POST-HOC addendum 4 item 3 / addendum 2 item 5: an EXISTING record is complete when its digest is what the code and
    # below-footer text IT WAS WRITTEN UNDER produce and the registry still holds that pair for this stage at the
    # record's own timestamp (h3.record_digest_basis).  A stale record (stale_after_supersession) resolves to the LIVE
    # pair, does not match, and is refitted -- never accepted, never rewritten in place by a differing digest.
    basis = H3.record_digest_basis(D.read_record(js), code, stage=STAGE)
    status = H3.resume_status(pq, js, H3.fold_digest(job, fold, basis["code"], addenda=basis["addenda"], **dg), steps)
    if status["complete"]:
        return {"arm": arm, "transform": transform, "fold_id": fold.fold_id, "status": "skipped_done",
                "code_resolution": basis["resolved"]}
    digest = H3.fold_digest(job, fold, code, **dg)           # a record written NOW carries the live pair
    t0 = time.perf_counter()
    fc, info = fold_context(entry, fold, ordinal, corpus, out_root, state, code=code, guard_fn=guard_fn,
                            inner_check=inner_check)
    # addendum 2 item 4: the shared-only re-run is its own frozen arm (the WITH record's selections, zero offsets)
    runner = H3.shared_only_runner(arm) if transform == H3.SHARED_ONLY else H3.frozen_runner(arm)
    outputs = runner.point(fc, wrec)
    if len(outputs) != 1:
        raise AssertionError(f"{job.key}: an H3 runner writes exactly one arm, got {sorted(outputs)}")
    o = next(iter(outputs.values()))
    istat = "pending" if o.intervals is None else (
        "split_conformal_inner" if o.intervals.get("status", "calibrated") == "calibrated" else str(o.intervals["status"]))
    frame = D.prediction_frame(o.pred, job=job, arm=job.arm, fold=fold, ordinal=ordinal, row_ids=fc.sc_ids,
                               selected_config=o.selected_config, model_seed=o.model_seed, intervals_status=istat,
                               batching_label=fc.batching_label, fit_seconds=o.seconds)
    D.assert_v6_clean(corpus.labels_of(frame["row_id"]), corpus.v6, f"{job.key}/{fold.fold_id} written")
    D.assert_selection_rows(frame["row_id"], corpus.half_by_id[job.design], f"{job.key}/{fold.fold_id} written")
    rec = {"schema": H3.SCHEMA, "job": job.record(), "model_arm": arm, "transform": transform, "arm": job.arm,
           "fold_id": fold.fold_id, "fold_hash": fold.fold_hash, "stem": job.stem, "design_hash": design_hash,
           "digest": digest, "code_digest": code, "fold_ordinal": ordinal, "model_seed": o.model_seed,
           "selected_config": o.selected_config, "with_record_digest": None if wrec is None else wrec.get("digest"),
           "with_selected_config": None if wrec is None else wrec.get("selected_config"),
           "prereg_sha256": D.REGISTERED_PREREG_SHA256, "prereg_addenda_sha256": REG.below_footer_sha256(STAGE),
           "prereg_n_addenda": REG.addenda_count(STAGE), "registry_stage": STAGE,
           "prereg_gate": dict(prereg) if prereg else None, "arm_record": o.record, "readings": H3.READINGS["tuning"],
           "shared_only": transform == H3.SHARED_ONLY,
           # WHEN this record was written, so registry.verify_record can order it against a later re-registration of
           # this stage (task X finding V-P06: the superseded fallback protects records written EARLIER)
           "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           **info,
           "steps": {"point": {"seconds": round(o.seconds, 3), "peak_rss_bytes": D.peak_rss_bytes(),
                               "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}}}
    rd._atomic_parquet(frame, pq)
    rd._atomic_json(rec, js)
    if "intervals" in steps and getattr(runner, "has_interval_step", False) and wrec is not None:
        t1 = time.perf_counter()
        cal, cplan = runner.calibration(fc, wrec["arm_record"])
        frame = pd.read_parquet(pq)
        if cal is None:
            frame = frame.assign(intervals_status=cplan["status"])
            rec["steps"]["intervals"] = {"seconds": round(time.perf_counter() - t1, 3), "status": cplan["status"],
                                         "plan": cplan, "n_calibration": 0,
                                         "method": "not calibrated (discovery.calibration_folds)"}
        else:
            import contextlib

            from gen19ct.models import ladder as LAD

            # a runner may need its fits wrapped (the shared-only ladder re-run builds LadderNet by name)
            with getattr(runner, "training_context", contextlib.nullcontext)():
                cal.fit_table(fc.corpus.table, fc.mask, fc.ctx)
            if isinstance(cal, LAD.NormalisedCrossFitConformal):
                frame = LAD.attach_normalised_intervals(frame, cal.quantiles, len(cal.residuals))
                method = "models.ladder.NormalisedCrossFitConformal (section 12 normalised split conformal, inner folds)"
            else:
                frame = D.attach_intervals(frame, cal.quantiles, len(cal.residuals))
                method = "discovery.CrossFitResidualConformal (cross-fitted inner splits at the WITH record's selections)"
            rec["steps"]["intervals"] = {"seconds": round(time.perf_counter() - t1, 3), "status": "calibrated",
                                        "plan": cplan, "method": method, **cal.record()}
        rd._atomic_parquet(frame, pq)
        rd._atomic_json(rec, js)
    return {"arm": arm, "transform": transform, "fold_id": fold.fold_id, "status": "fitted",
            "seconds": round(time.perf_counter() - t0, 3)}


def run_plan(plan: Sequence[Mapping[str, Any]], corpus: Any, out_root: Path, state: D.PlanState, *, code: str,
             steps: Sequence[str] = H3.STEPS, guard_fn: Callable | None = None, inner_check: Callable | None = None,
             prereg: Mapping[str, Any] | None = None, stop_on_error: bool = True,
             max_hours: float | None = None, workers: int = 1,
             budget_hours: float | None = H3.H3_BUDGET_HOURS) -> dict[str, Any]:
    """Every fold of every planned H3 arm, in the POST-HOC addendum 4 item 2 priority order V5 -> V2 -> V1
    (``h3.order_by_priority``; one process per worker, at most :data:`h3.H3_WORKERS`).

    The 20 h cap is checked BEFORE EACH FOLD against the ledger's wall clock -- the union of every recorded invocation's
    interval plus this invocation's elapsed loop time, so a second worker running the same hour does not double-count.
    When it is reached, the fold is not dispatched, the design being fitted and every later design in the priority order
    are written ``NOT_RUN`` with the reason, and nothing partial is scored.  ``--max-hours`` stays an operator pause and
    is recorded as such, never as a ``NOT_RUN`` reason.  ``budget_hours=None`` disables the cap (tests only)."""
    if int(workers) > H3.H3_WORKERS:
        raise SystemExit(f"refused: POST-HOC addendum 4 item 2 caps H3 at {H3.H3_WORKERS} workers, {workers} requested")
    ordered = H3.order_by_priority(plan)
    ledger: dict[str, Any] = {"arms": {}, "stopped": None, "workers": int(workers),
                              "priority_order": list(H3.DESIGN_PRIORITY),
                              "plan_order": [f"{e['model_arm']}:{e['transform']}@{e['design']}" for e in ordered],
                              "designs_not_run": {}, "budget": None}
    t0 = time.perf_counter()
    prev = H3.h3_wall_clock(out_root)                        # every earlier invocation, including the other worker's
    designs = list(dict.fromkeys(str(e["design"]) for e in ordered))

    def budget() -> dict[str, Any]:
        return H3.h3_budget_status(float(prev["wall_seconds"]) + (time.perf_counter() - t0),
                                   worker_seconds=float(prev["worker_seconds"]) + (time.perf_counter() - t0))

    def stop_for_budget(design: str, bud: Mapping[str, Any]) -> None:
        """Mark ``design`` and every later priority design NOT_RUN for THIS invocation (POST-HOC addendum 4 item 2).

        Whether a design is finally NOT_RUN is decided by its RECORD SETS at scoring time (``h3.design_status``), so an
        earlier invocation that already completed one of these designs is not contradicted; this is the reason record
        of the invocation whose cap fired."""
        reason = (f"the POST-HOC addendum 4 item 2 cap of {bud['budget_hours']:g} h of wall clock on "
                  f"{bud['workers_max']} workers was reached ({bud['used_hours']:g} h used) before this design's folds "
                  f"were fitted; priority order {' -> '.join(bud['priority_order'])}")
        start = designs.index(design) if design in designs else 0
        for d in designs[start:]:
            ledger["designs_not_run"].setdefault(d, H3.not_run_design(d, reason, budget=dict(bud)))
        ledger["budget"] = dict(bud)
        ledger["stopped"] = reason

    for entry in ordered:
        job = entry["job"]
        key = f"{entry['model_arm']}:{entry['transform']}@{entry['design']}"
        if max_hours is not None and (time.perf_counter() - t0) / 3600.0 >= max_hours:
            ledger["stopped"] = f"--max-hours {max_hours} reached (operator pause, resumable; not a NOT_RUN reason)"
            log(ledger["stopped"])
            break
        folds = corpus.fittable(job)
        res = []
        for fold, ordinal in folds:
            bud = budget()
            if budget_hours is not None and bud["exhausted"]:
                stop_for_budget(str(entry["design"]), bud)
                log(ledger["stopped"])
                break
            try:
                res.append(run_fold(entry, fold, ordinal, corpus, out_root, state, code=code, steps=steps,
                                    guard_fn=guard_fn, inner_check=inner_check, prereg=prereg))
            except Exception as exc:                        # noqa: BLE001 -- recorded with the traceback
                res.append({"arm": entry["model_arm"], "transform": entry["transform"], "fold_id": fold.fold_id,
                            "status": "error", "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-4000:]})
                if stop_on_error:
                    break
        errs = [r for r in res if r["status"] == "error"]
        ledger["arms"][key] = {"n_folds": len(folds), "fitted": sum(r["status"] == "fitted" for r in res),
                               "skipped": sum(r["status"] == "skipped_done" for r in res), "errors": errs[:3],
                               "seconds": round(sum(float(r.get("seconds") or 0) for r in res), 1)}
        log(f"{key}: {ledger['arms'][key]['fitted']} fitted, {ledger['arms'][key]['skipped']} skipped, {len(errs)} errors")
        if errs and stop_on_error:
            ledger["stopped"] = f"error in {key}: {errs[0]['error']}"
            break
        if ledger["designs_not_run"]:
            break
    ledger["budget"] = ledger["budget"] or dict(budget())
    ledger["loop_seconds"] = round(time.perf_counter() - t0, 3)
    ledger["wall_clock_before"] = prev
    return ledger


# ============================================================================================= #
# scoring
# ============================================================================================= #

def expected_records(entry: Mapping[str, Any], corpus: Any, state: D.PlanState, out_root: Path, *, code: str,
                     per_record_basis: bool = True) -> dict[str, dict[str, str]]:
    """The digest and fold hash every fittable fold of ``entry`` must carry.

    ``per_record_basis`` (the default) resolves the (code digest, below-footer digest) of EACH existing record through
    ``h3.record_digest_basis``, so a record written before a supersession of the ``h3`` entry is verified against the pair
    it was written under and one written after it with stale code is not (POST-HOC addendum 4 item 3).  A fold with no
    record on disk gets the live pair.  ``per_record_basis=False`` computes every digest from ``code`` and the live
    below-footer digest (tests of the digest itself)."""
    rd = runner_module()
    job = entry["job"]
    folds = corpus.folds(job.stem)
    dh = FI.design_hash(list(folds))
    out = {}
    for fold, ordinal in corpus.fittable(job):
        wrec = with_record(out_root, entry, fold)
        _, js = H3.fold_paths(out_root, entry["model_arm"], entry["transform"], job, fold.fold_id)
        basis = (H3.record_digest_basis(D.read_record(js), code, stage=STAGE) if per_record_basis
                 else {"code": code, "addenda": None})
        out[fold.fold_id] = {
            "digest": H3.fold_digest(job, fold, basis["code"], addenda=basis["addenda"], transform=entry["transform"],
                                     with_digest=None if wrec is None else wrec.get("digest"),
                                     guard_mode=rd.guard_mode_for(job, state), design_hash=dh, ordinal=ordinal,
                                     model_seed=rd.model_seed_of(ordinal)),
            "fold_hash": fold.fold_hash}
    return out


class Frames:
    """Ln(III) scoring frames of the WITH discovery records and of the H3 arms (verified records only)."""

    def __init__(self, out_root: Path, attrs: pd.DataFrame, state: D.PlanState, corpus: Any, *, code: str,
                 discovery_code: str, folds_dir: Path | None = None, runners: Mapping[str, Any] | None = None):
        self.out_root, self.attrs, self.state, self.corpus = Path(out_root), attrs, state, corpus
        self.code, self.discovery_code = code, discovery_code
        self.folds_dir = paths.FOLDS_DIR if folds_dir is None else Path(folds_dir)
        self.runners = runners
        self.rd = runner_module()
        self.v6 = attrs["v6_target_row"].astype(bool)
        flag = attrs["acidic_coextractant_modifier"].astype(bool) if "acidic_coextractant_modifier" in attrs.columns \
            else pd.Series(False, index=attrs.index)
        self.excluded_ids = sorted(attrs.index[flag.to_numpy()].astype(str))
        self.remainder_groups = sorted(g for g, n in attrs[EM.PUB_GROUP_COL].astype(str).value_counts().items()
                                       if n < 20)
        self.record_sets: dict[str, Any] = {}
        self._cache: dict = {}

    def _scoring(self, pred: pd.DataFrame | None, design: str, what: str) -> pd.DataFrame | None:
        if pred is None or pred.empty:
            return None
        label = H3._label(design)
        hc = f"registered_half_{'V5' if label in ('V5', 'V5-P', 'V5-PAIR') else design}"
        # R19 item 6's scoring-filter sensitivities read wildcard_copy_partner_in_training, which
        # discovery.prediction_frame does not carry: the scorer adds it per design stem from the crossings table
        # (g19_score_discovery.Store.frame) and so does the power runner; without it D.filtered_pair raises KeyError
        # and NO H3 contrast can be scored at all (found while applying task X finding V-L1).  The crossings depend on
        # the FOLDS only -- which fold hid which row, and which rows are copies -- never on an arm or a target.
        sc = scorer_module()
        stem = FI.design_stem(str(pred["design"].iloc[0]), str(pred["variant"].iloc[0]), str(pred["scheme"].iloc[0]))
        pred = sc.wildcard_flags(pred, stem, self.crossings())
        return D.scoring_frame(pred, self.attrs, design=label, v6_mask=self.v6, what=what, half_col=hc)

    def crossings(self) -> pd.DataFrame:
        """``folds/wildcard_copy_crossings.csv`` as the scorer reads it (``g19_score_discovery.CROSSINGS_CSV``), cached."""
        if "crossings" not in self._cache:
            self._cache["crossings"] = pd.read_csv(scorer_module().CROSSINGS_CSV,
                                                   dtype={"row_id": str, "partner_id": str})
        return self._cache["crossings"]

    def with_frame(self, entry: Mapping[str, Any]) -> pd.DataFrame | None:
        arm, design = entry["model_arm"], entry["design"]
        key = ("with", arm, design)
        if key in self._cache:
            return self._cache[key]
        if H3.is_deterministic(arm):
            fr = self.h3_frame({**entry, "transform": "WITH"})
        elif H3.is_ladder_arm(arm):
            job = entry["with_job"]
            RL, lstate, codes = ladder_context(self.out_root)
            pred, st = RL.verified_ladder_predictions(self.out_root, arm, job.design_dir, int(job.seed),
                                                      code=codes["discovery"], state=self.state,
                                                      excluded_ids=self.excluded_ids,
                                                      runner=RL.LadderRunner(arm, lstate, codes), folds_dir=self.folds_dir)
            self.record_sets[f"WITH/{arm}/{job.design_dir}/s{job.seed}"] = {k: v for k, v in st.items() if k != "arm"}
            fr = self._scoring(pred, design, f"WITH/{arm}/{design}")
        else:
            job = entry["with_job"]
            # the code digest of the WITH record set's own registry stage (registry.record_dir_code; the discovery
            # entry for every deployed arm's main-design and V5-PAIR records, the gate's digest without a registry)
            wcode = REG.record_dir_code(self.out_root, arm, job.design_dir, int(job.seed), default=self.discovery_code)
            pred, st = self.rd.verified_predictions(self.out_root, arm, job.design_dir, int(job.seed),
                                                    code=wcode, state=self.state,
                                                    excluded_ids=self.excluded_ids, folds_dir=self.folds_dir,
                                                    runners=self.runners)
            self.record_sets[f"WITH/{arm}/{job.design_dir}/s{job.seed}"] = st
            fr = self._scoring(pred, design, f"WITH/{arm}/{design}")
        self._cache[key] = fr
        return fr

    def h3_frame(self, entry: Mapping[str, Any]) -> pd.DataFrame | None:
        arm, transform, design, job = entry["model_arm"], entry["transform"], entry["design"], entry["job"]
        key = ("h3", arm, transform, design)
        if key in self._cache:
            return self._cache[key]
        if not H3.is_scored_transform(transform):
            raise AssertionError(f"POST-HOC addendum 4 item 1: {transform} is {H3.EXPLORATORY_NOT_SCORED} and no frame of "
                                 "it may be read for scoring")
        rdir = H3.record_dir(self.out_root, arm, transform, job)
        # the code digest and below-footer digest EACH record was written under, when the registry still holds the pair
        # for this stage at that record's timestamp (addendum 2 item 5; POST-HOC addendum 4 item 3) -- never used to FIT,
        # only to verify.  record_set_code is kept beside it as the set-level summary the record_sets block reports.
        rcode = H3.record_set_code(rdir, self.code, stage=STAGE)
        exp = expected_records(entry, self.corpus, self.state, self.out_root, code=self.code)
        # BOTH steps: the scorer reads the interval columns (calibration deltas, coverage), so a point-only record whose
        # intervals step never ran must leave the set INCOMPLETE rather than count as a found fold and turn an
        # interval-based delta silently NOT_RUN through ``_has_intervals`` (task X finding numbers VH-06)
        pred, st = H3.read_record_set(rdir, exp, steps=H3.STEPS, what=f"{arm}:{transform}@{design}")
        st = {**st, "code_resolution": rcode}
        self.record_sets[f"{transform}/{arm}/{job.design_dir}/s{job.seed}"] = st
        fr = self._scoring(pred, design, f"{arm}:{transform}/{design}")
        # an ABSENT record set is not cached: the same Frames scores again after ``main`` fits the shared-only
        # records in the same invocation (task X verifier finding 1), so absence must be re-read from disk
        if fr is not None:
            self._cache[key] = fr
        return fr

    def kw(self, arm: str, design: str) -> dict[str, Any]:
        return {"v6_mask": self.v6, "v1_scheme": H3.v1_scheme_of(arm, design, self.state),
                "remainder_groups": self.remainder_groups}


def score(out_root: Path, attrs: pd.DataFrame, corpus: Any, *, deployed: Mapping[str, Any], state: D.PlanState,
          code: str, discovery_code: str, delta5: float, folds_dir: Path | None = None,
          runners: Mapping[str, Any] | None = None, designs: Sequence[str] = tuple(H3.DESIGNS),
          with_pairs: bool = False, n_resamples: int = ET.N_RESAMPLES, seed: int = D.PRIMARY_SEED,
          negative_transfer: bool = True) -> dict[str, Any]:
    """Every H3 output as in-memory objects (contrasts, deltas, per-unit deltas, verdicts, F4, negative-transfer
    tables, the summary); records are verified against the digests the current code produces."""
    fr = Frames(out_root, attrs, state, corpus, code=code, discovery_code=discovery_code, folds_dir=folds_dir,
                runners=runners)
    arms = H3.model_arms(deployed["arm"])
    plan = plan_jobs(arms, state, designs=designs, seed=seed, with_pairs=with_pairs)
    contrast_rows: list[dict[str, Any]] = []
    item_rows: list[pd.DataFrame] = []
    delta_rows: list[dict[str, Any]] = []
    per_unit: list[pd.DataFrame] = []
    results: dict[str, dict[str, dict[str, Any]]] = {}
    hurts: dict[str, dict[str, Any]] = {}
    scored_triples: list[tuple[str, str, str]] = []
    not_run: dict[str, dict[str, Any]] = {}
    for entry in plan:
        arm, transform, design = entry["model_arm"], entry["transform"], entry["design"]
        if transform == "WITH":
            continue
        w = fr.with_frame(entry)
        o = fr.h3_frame(entry)
        if w is None or o is None:
            # POST-HOC addendum 4 item 2: an absent or INCOMPLETE record set makes the design NOT_RUN, never partial --
            # on the H3 side (the refits the cap pays for) and on the WITH side (its discovery / ladder records)
            st = (H3.design_status(fr.record_sets, design, transform, arm) if o is None else
                  H3.not_run_design(design, f"the WITH record set of {arm}@{design} is not complete "
                                            f"({fr.record_sets.get(f'WITH/{arm}/{entry['with_job'].design_dir}/s{entry['with_job'].seed}')})"))
            if design != H3.PAIR_DESIGN and st["status"] != "complete":
                prev = not_run.get(design)
                not_run[design] = st if prev is None else {**prev, "also": [*(prev.get("also") or []), st["reason"]]}
            continue
        scored_triples.append((arm, transform, design))
        kw = fr.kw(arm, design)
        if design == H3.PAIR_DESIGN:
            continue                                        # the pair endpoint is scored by logsf_delta below
        margin = H3.margin_of(design, delta5)
        res = H3.h3_contrast(w, o, design=design, model_arm=arm, transform=transform, margin=margin,
                             n_resamples=n_resamples, seed=seed, **kw)
        # section 7 item 6: a V5 contrast of a heavy arm carries the batching label of the plan state, so a result
        # computed on the re-coloured folds (or after the check failed) is never read as the registered batched one
        res["batching_label"] = D.heavy_v5_batching_label(H3.batching_arms(arm), H3._label(design), state)
        results.setdefault(arm, {}).setdefault(design, {})[transform] = res
        for row in D.contrast_rows(res):
            row.update(model_arm=arm, transform=transform, key=res["key"], direction=res["direction"])
            contrast_rows.append(row)
        item_rows.append(D.r19_item_rows(res).assign(key=res["key"], model_arm=arm, transform=transform))
        rev = H3.h3_contrast(w, o, design=design, model_arm=arm, transform=transform, margin=margin, hurts=True,
                             n_resamples=n_resamples, seed=seed, **kw)
        rev["batching_label"] = res["batching_label"]
        hurts.setdefault(arm, {}).setdefault(design, {})[transform] = rev
        for row in D.contrast_rows(rev):
            row.update(model_arm=arm, transform=transform, key=rev["key"], direction=rev["direction"])
            contrast_rows.append(row)
        item_rows.append(D.r19_item_rows(rev).assign(key=rev["key"], model_arm=arm, transform=transform))
        delta_rows += H3.rank_accuracy_delta(w, o, design=design, model_arm=arm, transform=transform,
                                            n_resamples=n_resamples, **kw)
        delta_rows += H3.calibration_delta(w, o, design=design, model_arm=arm, transform=transform,
                                           n_resamples=n_resamples, **kw)
        per_unit.append(H3.per_unit_delta_table(w, o, design=design, model_arm=arm, transform=transform, **kw))
    # POST-HOC addendum 4 item 2: "no verdict is taken from a partial design" -- every design a contrast was scored on
    # must have a COMPLETE record set, or nothing below is a verdict
    design_statuses = H3.assert_no_partial_design(fr.record_sets, scored_triples)
    # ... and the UNIT of NOT_RUN is the DESIGN, not one record set: a design one of whose arm x transform legs is
    # incomplete contributes NOTHING, even from a leg that is complete, or designs_scored and designs_not_run would name
    # the same design and D03 would print both NOT_RUN and a FAIL verdict for it (task X finding protocol VH-01)
    not_run = H3.attach_not_run_context(not_run, scored_triples=scored_triples,
                                        blocking=H3.blocking_errors(out_root))
    item_frame = pd.concat(item_rows, ignore_index=True) if item_rows else pd.DataFrame()
    per_unit_frame = pd.concat(per_unit, ignore_index=True) if per_unit else pd.DataFrame()
    dropped = H3.drop_not_run_designs(not_run, rows={"contrasts": contrast_rows, "deltas": delta_rows},
                                      frames={"r19_items": item_frame, "per_unit": per_unit_frame},
                                      nested=(results, hurts))
    contrast_rows, delta_rows = dropped["rows"]["contrasts"], dropped["rows"]["deltas"]
    item_frame, per_unit_frame = dropped["frames"]["r19_items"], dropped["frames"]["per_unit"]
    scored_triples = [t for t in scored_triples if t[2] not in set(not_run)]
    designs_scored = sorted({d for _, _, d in scored_triples})
    assert not (set(designs_scored) & set(not_run)), \
        (f"POST-HOC addendum 4 item 2: designs_scored {designs_scored} and designs_not_run {sorted(not_run)} name the "
         "same design; no verdict may be taken from a partial design")
    verdicts = {arm: H3.h3_verdict(helps=results.get(arm) or {}, hurts=(hurts.get(arm) or {}).get("V5", {}),
                                   designs_not_run=not_run)
                for arm in arms}
    dep = D.ARM_ALIASES.get(deployed["arm"] or "", "")
    shared = score_shared_only(fr, dep, state, delta5=delta5, designs=designs, n_resamples=n_resamples, seed=seed,
                               verdict=verdicts.get(dep))
    f4 = H3.f4_check({d: (hurts.get(dep) or {}).get(d, {}).get("WITHOUT") for d in designs}, deployed_arm=dep,
                     trained_with_actinides=True)
    contrasts = pd.DataFrame(contrast_rows)
    if not contrasts.empty:
        contrasts = D.apply_bh(contrasts, registered_families=(H3.FAMILY,))
    cells = per_unit_frame
    # section 11 states ONE condition for the whole negative-transfer investigation and names no design; it is evaluated
    # HERE, on one reading, and recorded beside the tables (task X finding protocol VH-05)
    neg_cond = H3.negative_transfer_condition(verdicts.get(dep), deployed_arm=dep)
    neg: dict[str, pd.DataFrame] = {}
    dependent = pd.DataFrame()
    stratum: dict[str, Any] = {"status": H3.NOT_COMPUTED}
    if negative_transfer and not cells.empty:
        v5 = cells[(cells["design"] == "V5") & (cells["model_arm"] == dep) & (cells["transform"] == "WITHOUT")]
        if not v5.empty:
            systems = H3.system_actinide_stats(corpus.frame.assign(**{
                EM.PUB_GROUP_COL: corpus.frame[I.PUB_GROUP_COL].astype(str),
                EM.CONDITION_KEY_COL: attrs[EM.CONDITION_KEY_COL].reindex(
                    corpus.frame[FI.ROW_ID].astype(str)).to_numpy()}))
            pairs_cells = [(str(a), str(b)) for a, b in zip(v5[EM.METAL_STATE_COL], v5[EM.SYSTEM_COL])]
            dependent = H3.actinide_dependent_cells(corpus.frame, pairs_cells)
            neg = H3.negative_transfer_tables(v5, systems, dependent=dependent, n_resamples=n_resamples)
            head = (results.get(dep) or {}).get("V5", {}).get("WITHOUT")
            # section 11 makes the actinide-partner-eligibility cells a separate REPORTING unit: the registered headline
            # stays the all-cell macro and the stratum-excluded macro is printed beside it (task X finding protocol VH-08)
            stratum = H3.stratum_excluded_headline(v5, dependent, point=None if head is None else head.get("point"))
            neg = {k: v.assign(tables_label=neg_cond["tables_label"]) for k, v in neg.items()}
    wc = H3.h3_wall_clock(out_root)
    summary = {"schema": H3.SCHEMA, "deployed": dict(deployed), "model_arms": list(arms),
               "transforms": list(H3.TRANSFORMS), "transforms_sealed": list(H3.ALL_ARMS),
               "exploratory": H3.exploratory_records(out_root),
               "designs": list(designs) + ([H3.PAIR_DESIGN] if with_pairs else []),
               "design_priority": list(H3.DESIGN_PRIORITY), "design_statuses": design_statuses,
               "designs_not_run": not_run, "designs_scored": designs_scored,
               # the counts only: the filtered rows and frames are the outputs themselves
               "designs_suppressed": {k: v for k, v in dropped.items() if k not in ("rows", "frames")},
               "negative_transfer_condition": neg_cond, "actinide_dependent_stratum": stratum,
               # the section 8 checks these contrasts owe, and which of them have one: a RECORDED debt, not an absence
               # (task X finding protocol VH-09)
               "power_debt": H3.needs_power_inventory(contrasts, checked=H3.power_checked_keys(out_root)),
               "wall_clock": wc,
               "budget": H3.h3_budget_status(wc["wall_seconds"], worker_seconds=wc["worker_seconds"]),
               "refits": H3.stale_records(out_root),
               "delta5": float(delta5), "seed": int(seed), "verdicts": verdicts, "f4": f4,
               "record_sets": fr.record_sets, "n_contrast_rows": int(len(contrasts)),
               "plan_state": state.record(), "heavy_v5_batching_label": state.heavy_v5_label,
               "heavy_v5_scheme": state.heavy_v5_scheme, "heavy_v1_scheme": state.heavy_v1_scheme,
               "s1_forced_undecided": state.s1_forced_undecided,
               "readings": H3.READINGS, "shared_only": {k: v for k, v in shared.items() if k != "frame"},
               "not_computed": {
                   "act_metal_shuffled": H3.ACT_METAL_SHUFFLED_NOT_RUN,
                   "designs_not_run": {d: v.get("reason") for d, v in sorted(not_run.items())} or None,
                   "crps": D.UNCERTAINTY_NOT_RUN["gaussian_crps"],
                   # section 11 registers Delta logSF MAE on V5-PAIR Ln pairs; neither H3 arm has a V5-PAIR record
                   # (addendum 1 item 5), so the quantity is not defined here -- named, never silently absent
                   "logsf_mae_delta": (None if with_pairs and H3.PAIR_DESIGN in [e["design"] for e in plan]
                                       else H3.READINGS["logsf_delta"]),
                   "shared_only_embedding_rerun": (None if shared["status"] == "run" else
                                                   f"{shared['status']}: {H3.READINGS['shared_only_embedding']}"),
                   "v6_deltas": "section 11: the V6 deltas are computed at confirmation only",
                   "power_check": "kappa_min comes from scripts/g19_run_power.py (section 8)"},
               "label": "discovery, optimistically biased (selection half, seed 104729)"}
    return {"contrasts": contrasts, "r19_items": item_frame,
            "deltas": pd.DataFrame(delta_rows), "per_unit": cells, "negative_transfer": neg,
            "actinide_dependent": dependent, "verdicts": verdicts, "f4": f4, "summary": summary,
            "shared_only": shared, "frames": fr}


def score_shared_only(fr: "Frames", deployed_arm: str, state: D.PlanState, *, delta5: float, verdict: Mapping[str, Any] | None,
                      designs: Sequence[str] = tuple(H3.DESIGNS), n_resamples: int = ET.N_RESAMPLES,
                      seed: int = D.PRIMARY_SEED) -> dict[str, Any]:
    """Addendum 2 item 4: the shared-only re-run "executes only under section 11's condition that WITHOUT beats WITH;
    otherwise D03 reports it as not run".  Evaluates :func:`h3.shared_only_condition` on the deployed arm's verdict and,
    when met, scores whatever complete SHARED_ONLY record sets exist: Delta = MAE(shared-only) - MAE(WITH) on each
    design's Ln set (positive favours the full section 15 embedding), exploratory BH family.  Returns the condition, the
    status (``run`` / ``not run (condition not met)`` / ``records absent`` per design) and the labelled rows."""
    cond = H3.shared_only_condition(verdict, deployed_arm=deployed_arm)
    out: dict[str, Any] = {"condition": cond, "status": cond["status"], "designs": {}, "contrasts": [],
                           "records_complete": False, "label": "exploratory: section 11 negative-transfer investigation, "
                                                                "shared-only embedding vs the full section 15 embedding"}
    if not cond["met"]:
        return {**out, "frame": pd.DataFrame()}
    arm = D.ARM_ALIASES.get(deployed_arm, deployed_arm)
    rows: list[dict[str, Any]] = []
    complete = True
    for entry in shared_only_plan(arm, state, designs=designs, seed=seed):
        design = entry["design"]
        w = fr.with_frame(entry)
        o = fr.h3_frame(entry)
        if w is None or o is None:
            out["designs"][design] = {"status": "records absent (run scripts/g19_run_h3.py; the re-run fits only when the "
                                                "condition holds)"}
            complete = False
            continue
        margin = H3.margin_of(design, delta5)
        res = H3.h3_contrast(w, o, design=design, model_arm=arm, transform=H3.SHARED_ONLY, margin=margin,
                             n_resamples=n_resamples, seed=seed, family=H3.SHARED_ONLY_FAMILY, **fr.kw(arm, design))
        res["batching_label"] = D.heavy_v5_batching_label(H3.batching_arms(arm), H3._label(design), state)
        for row in D.contrast_rows(res):
            row.update(model_arm=arm, transform=H3.SHARED_ONLY, key=res["key"], negative_transfer_investigation=True,
                       direction="shared-only embedding candidate; Delta = MAE(shared-only) - MAE(WITH), positive favours "
                                 "the full section 15 embedding", condition_met=True, label=out["label"])
            rows.append(row)
        out["designs"][design] = {"status": "scored", "point": res["point"], "key": res["key"]}
    out["records_complete"] = complete
    out["status"] = "run" if complete else "condition met; records incomplete"
    out["contrasts"] = [{k: v for k, v in r.items() if isinstance(v, (str, int, float, bool)) or v is None} for r in rows]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = D.apply_bh(frame, registered_families=(H3.FAMILY,))
    return {**out, "frame": frame}


# ============================================================================================= #
# main
# ============================================================================================= #

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--steps", default="point,intervals", help="point,intervals (default) or point")
    ap.add_argument("--only", default=None, help="comma list: model arm, transform, design or arm:transform")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the gate verdict; fit nothing")
    ap.add_argument("--score-only", action="store_true", help="score the records already written")
    ap.add_argument("--with-pairs", action="store_true", help="also refit the V5-PAIR folds of M2 (Delta logSF MAE)")
    ap.add_argument("--no-negative-transfer", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None, help="operator pause: dispatch no new fold after this")
    ap.add_argument("--workers", type=int, default=1,
                    help=f"concurrent H3 processes this invocation is one of (POST-HOC addendum 4 item 2 caps it at "
                         f"{H3.H3_WORKERS}); recorded in the wall-clock ledger")
    ap.add_argument("--no-budget", action="store_true",
                    help=argparse.SUPPRESS)                 # tests only: disable the 20 h cap
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--no-manifest", action="store_true")
    ns = ap.parse_args(argv)
    ns.steps = [s for s in ns.steps.split(",") if s]
    if not set(ns.steps) <= set(H3.STEPS) or H3.STEPS[0] not in ns.steps:
        raise SystemExit(f"--steps must include 'point' and be among {H3.STEPS}")
    if int(ns.workers) < 1 or int(ns.workers) > H3.H3_WORKERS:
        raise SystemExit(f"--workers must be 1..{H3.H3_WORKERS} (POST-HOC addendum 4 item 2: a cap of 20 h of wall "
                         f"clock on {H3.H3_WORKERS} workers)")
    return ns


def main(argv=None, *, check: Callable[[], int] | None = None, digests: Callable[[], Mapping[str, Any]] | None = None
         ) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    rd = runner_module()
    REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=ns.expect_addenda)
    H3.refuse_unless_cheap_complete(out_root)                   # before any heavy load (task X finding VL2-04)
    log("coextractant ids")
    coext = rd.coextractant_ids()
    gate = refuse_unless_ready(out_root, check=check, digests=digests, expect_addenda=ns.expect_addenda,
                               excluded_ids=coext)
    state = D.PlanState.read(rd.plan_state_path(out_root))
    decisions = read_decisions(out_root)
    deployed = H3.deployed_configuration(decisions, read_ladder(out_root))
    if deployed.get("arm") is None:
        raise SystemExit(f"refused: the configuration deployed for lanthanide prediction is undecidable ({deployed['basis']}); "
                         "section 11 names it as the H3 arm, so nothing is fitted (see h3.READINGS['deployed_rule'])")
    arms = H3.model_arms(deployed["arm"])
    log(f"deployed configuration: {deployed['arm']} ({deployed['basis']}); model arms {list(arms)}")
    if H3.is_ladder_arm(deployed["arm"]):
        gate["ladder_with_records"] = verify_ladder_with_records(out_root, deployed["arm"], state, excluded_ids=coext)
        log(f"ladder WITH records of {deployed['arm']} verified complete on {sorted(gate['ladder_with_records'])}")
    code = code_digest()
    plan = filter_plan(plan_jobs(arms, state, seed=D.PRIMARY_SEED, with_pairs=ns.with_pairs), ns.only)
    if ns.dry_run:
        for e in plan:
            print(f"{e['model_arm']:>8} | {e['transform']:<18} | {e['design']:<6} | {e['job'].design_dir} | "
                  f"s{e['job'].seed}")
        print(json.dumps({"deployed": deployed, "n_jobs": len(plan), "steps": ns.steps,
                          "discovery_complete": gate["discovery_complete"]["complete"],
                          "scorer_decisions_ok": gate["scorer_decisions"]["ok"]}, indent=2, default=str))
        return 0
    # POST-HOC addendum 4 item 3: nothing is scored while a record of a SCORED transform carries a superseded h3 digest
    # written after the supersession -- it is deleted and refitted first (scripts/g19_h3_refit_stale.py)
    stale = H3.stale_records(out_root, stage=STAGE)
    if stale["stale"]:
        raise SystemExit(f"refused: {len(stale['stale'])} H3 fold record(s) were written after the h3 registry entry they "
                         "carry was superseded and do not verify (POST-HOC addendum 4 item 3): "
                         f"{[r['path'] for r in stale['stale'][:5]]}. Delete exactly those and refit them under the "
                         "current registered digest: python generations/gen19_chem_transfer/scripts/g19_h3_refit_stale.py "
                         "--delete")
    t_proc = time.perf_counter()
    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    H3.open_h3_wall_clock(out_root, prior_legs=prior_legs(out_root),
                          note="opened by scripts/g19_run_h3.py before the first fold under POST-HOC addendum 4 item 2")
    log("corpus")
    corpus = rd.load_corpus(coext)
    attrs = scorer_module().build_attrs()
    delta5 = D.registered_delta5()
    with (Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=D.PRIMARY_SEED,
              extra={"prereg_gate": gate["prereg_gate"], "gate": {k: v for k, v in gate.items() if k != "prereg_gate"},
                     "code_sha256": code["combined"], "code_parts": code["parts"], "deployed": deployed,
                     "model_arms": list(arms), "readings": H3.READINGS,
                     "discovery_seeds": list(D.DISCOVERY_SEEDS), "seed": D.PRIMARY_SEED})
          if not ns.no_manifest else _Null()) as run:
        ledger = {} if ns.score_only else run_plan(plan, corpus, out_root, state, code=code["combined"], steps=ns.steps,
                                                   prereg=gate["prereg_gate"], max_hours=ns.max_hours,
                                                   workers=int(ns.workers),
                                                   budget_hours=None if ns.no_budget else H3.H3_BUDGET_HOURS)
        if ledger:
            # POST-HOC addendum 4 item 2: the fitting loop's own wall clock goes into the ledger BEFORE the scorer runs,
            # so the summary's budget block and the next invocation's check read what this invocation spent
            H3.record_h3_wall_clock(out_root, started=started, ended=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                                    seconds=float(ledger.get("loop_seconds") or 0.0),
                                    designs_done=sorted({k.split("@")[-1] for k in (ledger.get("arms") or {})}),
                                    workers=int(ns.workers), process_seconds=time.perf_counter() - t_proc,
                                    note=ledger.get("stopped"))
        res = score(out_root, attrs, corpus, deployed=deployed, state=state, code=code["combined"],
                    discovery_code=gate["discovery_code_sha256"], delta5=delta5, with_pairs=ns.with_pairs,
                    negative_transfer=not ns.no_negative_transfer)
        # addendum 2 item 4: the shared-only re-run fits ONLY once the condition is read from the scored verdicts
        so = res["shared_only"]
        if so["condition"]["met"] and not so["records_complete"] and not ns.score_only:
            splan = filter_plan(shared_only_plan(deployed["arm"], state, seed=D.PRIMARY_SEED), ns.only)
            log(f"section 11 condition met ({so['condition']['reason']}): shared-only re-run of {deployed['arm']} on "
                f"{[e['design'] for e in splan]}")
            ledger = {**ledger, "shared_only": run_plan(splan, corpus, out_root, state, code=code["combined"], steps=ns.steps,
                                                        prereg=gate["prereg_gate"], max_hours=ns.max_hours)}
            so = score_shared_only(res["frames"], deployed["arm"], state, delta5=delta5,
                                   verdict=res["verdicts"].get(D.ARM_ALIASES.get(deployed["arm"], deployed["arm"])))
            res["shared_only"] = so
            res["summary"]["shared_only"] = {k: v for k, v in so.items() if k != "frame"}
            res["summary"]["not_computed"]["shared_only_embedding_rerun"] = None if so["status"] == "run" else so["status"]
        else:
            log(f"shared-only re-run (addendum 2 item 4): {so['status']}")
        outs = write_outputs(out_root, res, ledger=ledger, gate=gate)
        if run is not None:
            run.outputs(*outs)
            run.extra.update({"ledger": H3.json_safe(ledger), "verdicts": H3.json_safe(res["verdicts"]),
                              "f4": H3.json_safe(res["f4"]), "confirmation_half_read": False,
                              "v6_target_rows_scored": 0,
                              "shared_only": H3.json_safe({k: v for k, v in res["shared_only"].items() if k != "frame"})})
    log(f"H3 verdicts: { {a: v['verdict'] for a, v in res['verdicts'].items()} }; F4 failure: {res['f4'].get('failure')}; "
        f"shared-only re-run: {res['shared_only']['status']}")
    return 0


def write_outputs(out_root: Path, res: Mapping[str, Any], *, ledger: Mapping[str, Any] | None = None,
                  gate: Mapping[str, Any] | None = None) -> list[Path]:
    """Every H3 output file (``evaluation/h3``, ``tables/h3_*.csv``, ``decisions/D03_actinide_transfer.md``)."""
    root = H3.h3_root(out_root)
    tables = Path(out_root) / "tables"
    outs: list[Path] = []
    summary = dict(res["summary"])
    summary["git_head"] = git_head()
    if not res["contrasts"].empty:
        outs.append(write_csv(res["contrasts"], root / "h3_contrasts.csv"))
        outs.append(write_csv(res["r19_items"], root / "h3_r19_items.csv"))
    if not res["deltas"].empty:
        outs.append(write_csv(res["deltas"], root / "h3_deltas.csv"))
    if not res["per_unit"].empty:
        outs.append(write_csv(res["per_unit"], root / "h3_per_unit_deltas.csv"))
        outs.append(write_csv(res["per_unit"], tables / "h3_per_unit_deltas.csv"))
    if isinstance(res.get("actinide_dependent"), pd.DataFrame) and not res["actinide_dependent"].empty:
        outs.append(write_csv(res["actinide_dependent"], root / "h3_actinide_dependent_cells.csv"))
    for name, tab in (res.get("negative_transfer") or {}).items():
        if isinstance(tab, pd.DataFrame) and not tab.empty:
            outs.append(write_csv(tab, root / "negative_transfer" / f"{name}.csv"))
    # section 11's trigger, recorded beside the tables exactly as shared_only.json records its own condition: the tables
    # are never "the section 11 investigation" unless this file says the condition was met (task X finding protocol VH-05)
    outs.append(write_json(root / "negative_transfer" / "condition.json", H3.json_safe({
        "schema": H3.SCHEMA, "git_head": summary.get("git_head"),
        **(summary.get("negative_transfer_condition") or {}),
        "tables": sorted(f"evaluation/h3/negative_transfer/{n}.csv" for n, t in (res.get("negative_transfer") or {}).items()
                         if isinstance(t, pd.DataFrame) and not t.empty),
        "actinide_dependent_stratum": summary.get("actinide_dependent_stratum")})))
    outs.append(write_json(root / "h3_summary.json", H3.json_safe(summary)))
    outs.append(write_json(root / "h3_verdicts.json", H3.json_safe(res["verdicts"])))
    outs.append(write_json(root / "h3_f4.json", H3.json_safe(res["f4"])))
    so = res.get("shared_only")
    if so is not None:                                      # addendum 2 item 4: the condition and status are always written
        outs.append(write_json(root / "shared_only" / "shared_only.json", H3.json_safe({k: v for k, v in so.items() if k != "frame"})))
        sf = so.get("frame")
        if isinstance(sf, pd.DataFrame) and not sf.empty:
            outs.append(write_csv(sf, root / "shared_only" / "contrasts.csv"))
            outs.append(write_csv(sf, tables / "h3_shared_only_contrasts.csv"))
    if ledger is not None:
        outs.append(write_json(root / "h3_run_ledger.json", H3.json_safe({"ledger": ledger, "gate": gate or {}})))
    # POST-HOC addendum 4 items 1-3: the exploratory transform, the per-design statuses under the 20 h cap and the
    # refit / verification state of every H3 record, in one decisions file beside the wall-clock ledger
    outs.append(write_json(H3.h3_decisions_path(out_root), H3.json_safe({
        "schema": H3.SCHEMA, "git_head": summary.get("git_head"),
        "addendum4_item1_exploratory": summary.get("exploratory"),
        "addendum4_item2_budget": summary.get("budget"), "addendum4_item2_wall_clock": summary.get("wall_clock"),
        "addendum4_item2_priority_order": list(H3.DESIGN_PRIORITY),
        "addendum4_item2_design_statuses": summary.get("design_statuses"),
        "addendum4_item2_designs_not_run": summary.get("designs_not_run"),
        "addendum4_item2_designs_scored": summary.get("designs_scored"),
        "addendum4_item2_suppressed_rows": summary.get("designs_suppressed"),
        "addendum4_item3_records": summary.get("refits"),
        "addendum4_item3_operative_reading": (summary.get("refits") or {}).get("operative_reading"),
        # the three readings this run records as REQUESTED but not registered, and the two debts
        "section11_negative_transfer_condition": summary.get("negative_transfer_condition"),
        "section11_actinide_dependent_stratum": summary.get("actinide_dependent_stratum"),
        "section10_f4_interval_reading": (res["f4"] or {}).get("interval_reading_not_registered"),
        "section8_power_debt": summary.get("power_debt"),
        "transforms_scored": list(H3.TRANSFORMS),
        "transforms_exploratory_not_scored": {t: H3.EXPLORATORY_NOT_SCORED for t in H3.EXPLORATORY_TRANSFORMS},
        "readings": {k: H3.READINGS[k] for k in ("act_metal_shuffled", "h3_budget", "design_priority",
                                                 "record_digest_basis")}})))
    if not res["contrasts"].empty:
        outs.append(write_csv(res["contrasts"], tables / "h3_contrasts.csv"))
    # the deviations file addendum 4 item 2's own vocabulary cannot hold: a design left incomplete by anything other than
    # the 20 h cap, with the blocking error named (task X finding protocol VH-07)
    nrun = summary.get("designs_not_run") or {}
    dev = {d: v for d, v in nrun.items() if isinstance(v, dict) and v.get("deviation")}
    outs.append(write_json(H3.h3_decisions_dir(out_root) / "deviations.json", H3.json_safe({
        "schema": H3.SCHEMA, "git_head": summary.get("git_head"), "n_deviations": len(dev),
        "rule": "POST-HOC addendum 4 item 2 registers the H3 design scope and says 'That is registered and is run in "
                "full'; the 20 h cap is the only registered reason a design may stay incomplete. A design incomplete "
                "while the cap is unreached is a DEVIATION and is recorded here with its cause.",
        "budget": summary.get("budget"), "deviations": dev,
        "power_debt": summary.get("power_debt"),
        "readings_requested_not_registered": {
            "section10_f4_interval": (res["f4"] or {}).get("interval_reading_not_registered"),
            "section11_negative_transfer_trigger": (summary.get("negative_transfer_condition") or {}).get("condition"),
            "addendum4_item3_operative": (summary.get("refits") or {}).get("operative_reading")}})))
    outs.append(write_text(Path(out_root) / "decisions" / "D03_actinide_transfer.md",
                           H3.d03_markdown(summary, res["contrasts"], res["deltas"], res.get("negative_transfer"),
                                           res.get("r19_items"))))
    return outs


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
