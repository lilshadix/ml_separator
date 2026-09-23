"""``scripts/g19_run_confirmation.py`` -- THE single registered confirmation run (pre-registration section 15).

One run, once.  It scores the frozen claims of ``decisions/CONFIRMATION_PLAN.md`` on the **confirmation half** with the
**5 withheld seeds**, runs **V6 once** with S2(a)-(c), and evaluates S1(c) and S1(d).  POST-HOC addendum 5 item 1 makes
it the **core** run: no section 11 V6 actinide deltas and no section 8 power check, both reported ``NOT_RUN`` with their
cost and their consequence.

Gates (all five, in order; :func:`gen19ct.evaluation.confirmation.gate`)
-----------------------------------------------------------------------
(a) ``scripts/g19_seal_prereg.py --check`` and the below-footer digest registered for stage ``confirmation``;
(b) the registry holds that stage and this code is the code it was registered with;
(c) ``decisions/CONFIRMATION_PLAN.md`` present, at most 5 eligible claims;
(d) ``--seed-store PATH`` that verifies against ``manifests/confirmation_seeds_sha256.txt``;
(e) the idempotence lock: ``evaluation/confirmation/decisions/confirmation.json`` absent, or ``--resume``.

The withheld seeds
------------------
They enter **only** through ``--seed-store PATH``, at run time, and only as a ``confirmation.SeedStore`` whose ``repr``
is redacted.  Everything written passes through ``confirmation.scrub`` first, which replaces a seed -- as an int or
inside any string, including a fold id -- by an opaque index ``i1``..``i5``; at the end
``confirmation.scan_for_seed_leak`` re-reads every file the run wrote and **fails the run** if a seed's decimal form
appears anywhere.  ``decisions/CONFIRMATION.md`` reveals the seeds only as the ``--verify-seeds`` verdict.

Modes
-----
``--dry-run``      enumerate the jobs, the folds and the cost from the measured unit costs; fit nothing, write nothing.
``--check-only``   run the five gates and print the verdict; enumerate nothing, fit nothing, write nothing.
``--resume``       complete unfitted folds of the recorded run.  It may do nothing else: the recorded code digest must be
                   the live one and the claim list may not grow, so no claim is ever re-scored under different code.

Fitting
-------
The confirmation half needs its own ``prepare_fold``: discovery's refuses a confirmation-half fold by construction
(*"a confirmation-half fold is never fitted in discovery"*).  :func:`prepare_fold` here is discovery's, with the half
switched and both directions of the V6 carve-out asserted (``confirmation.assert_v6_scope``); everything else -- the
section 2 guards, the tuning of addendum 1, the runners, the record schema -- is discovery's, imported and unchanged.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:                                     # noqa: E402
    sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.chemistry import support_graph as SG  # noqa: E402
from gen19ct.evaluation import confirmation as CF  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.manifest import Run, write_json  # noqa: E402
from gen19ct.models import interface as I  # noqa: E402

NAME = "g19_run_confirmation"
STAGE = CF.STAGE
#: this script's own prediction-affecting code, beside the discovery runner's (``code_digest``)
CODE_FILES: tuple[Path, ...] = (paths.G19_ROOT / "gen19ct" / "evaluation" / "confirmation.py", Path(__file__).resolve())
CODE_OBJECTS: tuple[Any, ...] = (CF.SeedStore, CF.confirmation_scored_ids, CF.assert_confirmation_rows,
                                 CF.assert_v6_scope, CF.item4, CF.score_claim, CF.seed_mean_system_bootstrap,
                                 CF.s1c_confirmation, CF.s2a_sign_count, CF.s2b_magnitude, CF.coverage_bands,
                                 CF.lock_verdict, CF.enumerate_jobs, CF.scrub)
MAX_WORKERS = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def runner_module():
    return _script("g19_run_discovery")


def seal_module():
    return _script("g19_seal_prereg")


def code_digest() -> dict[str, Any]:
    """The confirmation code digest: the discovery code (the arms are fitted with it) plus this script and its module."""
    rd = runner_module()
    return D.code_digest(rd.CODE_FILES + (Path(rd.__file__).resolve(),) + CODE_FILES, rd.RUNNER_OBJECTS + CODE_OBJECTS)


# ============================================================================================= #
# the withheld-seed fold files
# ============================================================================================= #

#: the fold files this run needs that do not exist for a withheld seed (the batched colourings and V6)
BATCHED_STEMS: tuple[str, ...] = ("V5__primary__batched_max4", "V5__strict__batched_max4",
                                  "V5__hno3_only__batched_max4", "V5PAIR__primary__batched")
SEEDLESS_STEMS: tuple[str, ...] = ("V5__primary__exact", "V2__element__exact")
V6_STEM = "V6__prnd__exact"


@dataclass(frozen=True)
class FoldPlan:
    """What must be built before any fit, per seed index: the stems, the rule and the recorded design hash."""

    stem: str
    seed_index: int
    n_folds: int
    design_hash: str
    rule: str

    def record(self) -> dict[str, Any]:
        return {"stem": self.stem, "seed_index": self.seed_index, "n_folds": self.n_folds,
                "design_hash": self.design_hash, "rule": self.rule}


def build_confirmation_folds(store: CF.SeedStore | None, out_root: Path, *, stems: Sequence[str] = BATCHED_STEMS,
                             dry_run: bool = False) -> list[FoldPlan]:
    """Build the withheld-seed colourings under the SAME registered rules as the discovery seeds' (section 3.1 /
    section 7 item 6: ``cell_holdout.v5_batched_folds`` with ``max_cells_per_batch = 4``, the same vertex order, batches
    formed inside each half, carved-out cells not batched), one file per seed INDEX.

    The file, its folds and their ids are written scrubbed (``confirmation.scrub``): the colouring depends on the seed,
    the FILE NAME and every recorded field carry ``i1``..``i5``.  ``dry_run`` returns the plan without building.
    """
    plans: list[FoldPlan] = []
    outdir = CF.folds_dir(out_root)
    indices = tuple(range(1, CF.N_SEEDS + 1)) if store is None else store.indices()
    for i in indices:
        for stem in stems:
            n = CF.FOLD_COUNTS_CONFIRMATION.get(stem, 0)
            plans.append(FoldPlan(stem=stem, seed_index=i, n_folds=n, design_hash="" if dry_run else "pending",
                                  rule="cell_holdout.v5_batched_folds (max_cells_per_batch=4) with this seed; "
                                       "V5PAIR__primary__batched via cell_holdout.v5pair_batched_folds; the registered "
                                       "vertex order and half-local batching, unchanged"))
    if dry_run:
        return plans
    outdir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(refusal("withheld_seed_fold_files"))


#: the stages of the single run, and which of them this tree implements.  The runner refuses at the first stage whose
#: code is not written, NAMING it -- it never produces a number from a stage that does not exist.  Everything above the
#: first ``False`` is implemented, tested (``tests/test_confirmation.py``) and exercised by ``--dry-run`` /
#: ``--check-only``; everything below it is the remaining work of the single run and is listed in the refusal.
RUN_STAGES: tuple[tuple[str, bool, str], ...] = (
    ("gates", True, "the five gates of confirmation.gate: seal + registered below-footer digest, the registry's "
                    "'confirmation' stage and this code, the plan with <= 5 claims, a verified --seed-store, the "
                    "idempotence lock"),
    ("plan", True, "decisions/CONFIRMATION_PLAN.md parsed into frozen claims (arm, comparator, design, margin)"),
    ("inventory_and_cost", True, "the job and fold inventory of the core run and its cost from the measured unit costs "
                                 "(--dry-run)"),
    ("seed_handling", True, "the store loaded and verified, seeds held in a redacted SeedStore, every written field "
                            "scrubbed to an opaque index, and the leak scan"),
    ("fold_isolation_and_record_writing", True, "the confirmation-half prepare_fold (discovery's assertions, the half "
                                                "switched, the V6 carve-out asserted in both directions) and run_fold"),
    ("claim_scoring", True, "R19 at confirmation over the 5 seeds: the seed-mean cluster bootstrap for items 1, 2, 3 "
                            "and 5, item 4 at 5 of 5, TOST, BH"),
    ("withheld_seed_fold_files", False, "building the withheld-seed colourings on the corpus: "
                                        "cell_holdout.v5_batched_folds (max_cells_per_batch = 4) and "
                                        "v5pair_batched_folds per seed, and the V6 design of section 3.4 "
                                        "(support_graph.hide_cells(component_aware=True) on Pr(III) x S and Nd(III) x S "
                                        "together, 13 systems)"),
    ("fit_loop", False, "the corpus load and the job -> fold dispatch at <= 2 workers with the wall-clock ledger"),
    ("s1c_s1d_s1e_assembly", False, "assembling S1(c)'s per-seed per-system direction deltas from the V5-PAIR records "
                                    "(the statistic itself is implemented and tested: "
                                    "confirmation.seed_mean_system_bootstrap / s1c_confirmation), and S1(d) / S1(e) "
                                    "from the prediction frames"),
    ("s2_assembly", False, "assembling S2(a)-(c) from the V6 records (the statistics are implemented and tested: "
                           "confirmation.s2a_sign_count / s2b_magnitude / coverage_bands)"),
    ("writers", False, "wiring decisions/confirmation.json, tables/confirmation_*.csv|md and decisions/CONFIRMATION.md "
                       "to a completed run (the writers themselves are implemented and tested: "
                       "confirmation.write_decisions / confirmation_tables / confirmation_report)"),
)


def refusal(stage: str) -> str:
    """The refusal text of an unimplemented stage: what is missing, what is not, and what must not happen meanwhile."""
    todo = [(n, why) for n, ok, why in RUN_STAGES if not ok]
    done = [n for n, ok, _ in RUN_STAGES if ok]
    return (f"refused: the confirmation run stage {stage!r} is NOT IMPLEMENTED in this tree, so the run stops here "
            "rather than producing a number from code that does not exist. Implemented and tested: "
            + ", ".join(done) + ". Still to write: "
            + "; ".join(f"{n} -- {why}" for n, why in todo)
            + ". Nothing was fitted, no confirmation-half row was scored, no withheld seed was written anywhere, and "
              "the idempotence lock is untouched, so the single registered run (section 15) has NOT been spent.")


# ============================================================================================= #
# one confirmation fold
# ============================================================================================= #

def prepare_fold(job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Any, out_root: Path, state: D.PlanState, *,
                 code: str = "", guard_fn: Callable | None = None, inner_check: Callable | None = None
                 ) -> tuple[Any, dict[str, Any]]:
    """Discovery's ``prepare_fold`` with the CONFIRMATION half (:data:`confirmation.READINGS` ``half``).

    Discovery's own refuses a confirmation-half fold, so this is the mirror: the scored rows are the fold's
    confirmation-half rows (``confirmation.confirmation_scored_ids``), every one of them is asserted to be in that half,
    and the V6 carve-out is asserted in BOTH directions -- a V6 job scores only ``V6_TARGET_ROWS`` and no other job
    scores any (``confirmation.assert_v6_scope``).  Everything else is discovery's code, called unchanged.
    """
    rd = runner_module()
    what = f"{job.key}/{fold.fold_id}"
    if fold.half == D.SELECTION:
        raise AssertionError(f"{what}: a selection-half fold is never fitted in the confirmation run")
    sel = list(CF.confirmation_scored_ids(fold))
    sc_ids = [r for r in sel if not bool(corpus.coext_by_id.get(r, False))]
    if not sc_ids:
        raise ValueError(f"{what}: no confirmation-half scored row")
    CF.assert_confirmation_rows(sc_ids, corpus.half_by_id[job.design], what)
    sc_labels = corpus.labels_of(sc_ids)
    CF.assert_v6_scope(sc_labels, pd.Series(corpus.v6, index=corpus.frame.index) if not isinstance(corpus.v6, pd.Series)
                       else corpus.v6, job.design, what)
    # from here the fold context is discovery's, built on the same objects: the guards, the training mask, the inner
    # design and the support file are its code, and the only difference is the half the scored rows come from
    fc, info = _prepare_with_half(rd, job, fold, ordinal, corpus, out_root, state, code, sc_ids, sc_labels, guard_fn,
                                  inner_check)
    info = dict(info)
    info.update(half=D.CONFIRMATION, n_scored_confirmation=len(sc_ids), reading=CF.READINGS["half"])
    return fc, info


def _prepare_with_half(rd: Any, job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Any, out_root: Path,
                       state: D.PlanState, code: str, sc_ids: Sequence[str], sc_labels: pd.Index,
                       guard_fn: Callable | None, inner_check: Callable | None) -> tuple[Any, dict[str, Any]]:
    """The body of discovery's ``prepare_fold`` on confirmation-half scored rows (its assertions kept, in its order)."""
    what = f"{job.key}/{fold.fold_id}"
    st = corpus.frame.loc[sc_labels, SG.METAL_COL]
    if st.isna().any() or (st == I.SR_III).any():
        raise AssertionError(f"{what}: an X(?) or Sr(III) row would be scored")
    hidden = corpus.labels_of(fold.hidden_row_ids)
    if not sc_labels.isin(hidden).all():
        raise AssertionError(f"{what}: a scored row is not hidden")
    sr_extra = corpus.sr_index.difference(hidden) if job.drop_sr else pd.Index([], dtype=hidden.dtype)
    universe = corpus.frame.index.difference(sr_extra)
    guards = (guard_fn or rd.outer_guard)(job, fold, corpus, universe)
    t = corpus.table
    mask = np.ones(t.n, dtype=bool)
    mask[t.positions(hidden)] = False
    if len(sr_extra):
        mask[t.positions(sr_extra)] = False
    if mask[t.positions(sc_labels)].any():
        raise AssertionError(f"{what}: a scored row is a training row")
    gmode = rd.guard_mode_for(job, state)
    cache = corpus.guard_cache.setdefault((job.design, job.variant, job.drop_sr), {})
    ctx = I.FitContext(systems=corpus.systems, components=corpus.comps, table=t, hidden_index=hidden.union(sr_extra),
                       v6_mask=corpus.v6, exclude_from_scoring=corpus.coext,
                       isolation_check=(inner_check or rd.inner_isolation_check)(job, corpus), guard_cache=cache,
                       seed=int(job.seed))
    fc = rd.FoldContext(job=job, fold=fold, ordinal=ordinal, corpus=corpus, mask=mask, hidden=hidden,
                        sc_ids=list(sc_ids), sc_labels=sc_labels, positions=t.positions(sc_labels), ctx=ctx,
                        out_root=out_root, guard_mode=gmode, batching_label=rd.batching_label(job, state), code=code,
                        state=state, design_hash=corpus.design_hash(job.stem))
    info = {"n_hidden": len(hidden), "n_train": int(mask.sum()), "n_scored_confirmation": len(sc_ids),
            "n_sr_iii_dropped": int(len(sr_extra)), "outer_guard": guards, "inner_guard_mode": gmode}
    return fc, info


def run_fold(job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Any, out_root: Path, *, store: CF.SeedStore,
             seed_index: int, runners: Mapping[str, Any], code: str, state: D.PlanState,
             steps: Sequence[str] = ("point", "intervals"), prereg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fit (or skip) one confirmation fold and write its prediction parquet and record, SCRUBBED of the seed.

    The record is discovery's schema with ``registry_stage`` = ``confirmation``, the opaque ``seed_index``, and the
    withheld seed nowhere: :func:`confirmation.scrub` runs over the whole record and over the prediction frame's labels
    before either is written.
    """
    rd = runner_module()
    runner = rd.runner_for(job, runners)
    pq, js = CF.fold_paths(out_root, job.arm, job.design_dir, seed_index, CF.scrub(fold.fold_id, store))
    if js.exists():
        return {"job": job.key, "fold_id": fold.fold_id, "status": "skipped_done"}
    REG.refuse_unless_writable(STAGE, code)
    fc, info = prepare_fold(job, fold, ordinal, corpus, out_root, state, code=code)
    outputs = runner.point(fc)
    out = {}
    for arm, o in outputs.items():
        frame = D.prediction_frame(o.pred, job=job, arm=arm, fold=fold, ordinal=ordinal, row_ids=fc.sc_ids,
                                   selected_config=o.selected_config, model_seed=o.model_seed,
                                   intervals_status="split_conformal_inner" if o.intervals else "pending",
                                   batching_label=fc.batching_label, fit_seconds=o.seconds)
        CF.assert_confirmation_rows(frame["row_id"], corpus.half_by_id[job.design], f"{job.key}/{fold.fold_id} written")
        rec = {"schema": CF.SCHEMA, "registry_stage": STAGE, "job": job.record(), "arm": arm, "fold_id": fold.fold_id,
               "fold_hash": fold.fold_hash, "stem": job.stem, "half": D.CONFIRMATION, "seed_index": int(seed_index),
               "seed_commitment_sha256": store.digest, "code_digest": code,
               "prereg_sha256": D.REGISTERED_PREREG_SHA256,
               "prereg_addenda_sha256": (prereg or {}).get("addenda_sha256"),
               "prereg_n_addenda": (prereg or {}).get("n_addenda"), "addendum_implemented": REG.addenda_count(STAGE),
               "selected_config": o.selected_config, "model_seed": o.model_seed, "arm_record": o.record,
               "steps": {"prediction": {"seconds": round(o.seconds, 3),
                                        "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}},
               **info}
        pq2, js2 = CF.fold_paths(out_root, arm, job.design_dir, seed_index, CF.scrub(fold.fold_id, store))
        pq2.parent.mkdir(parents=True, exist_ok=True)
        # the frame's numeric columns hold no seed; its fold_id and any seed-labelled column are scrubbed
        scrubbed = frame.copy()
        for col in scrubbed.columns:
            if scrubbed[col].dtype == object:
                scrubbed[col] = [CF.scrub(v, store) for v in scrubbed[col]]
        scrubbed.to_parquet(pq2, index=False)
        write_json(js2, CF.scrub(rec, store))
        out[arm] = str(js2)
    return {"job": job.key, "fold_id": fold.fold_id, "status": "fitted", "records": out}


# ============================================================================================= #
# scoring one claim over the 5 withheld seeds
# ============================================================================================= #

#: the seed combination of a claim's items 1, 2, 3 and 5 -- the one reading this run takes, and the fact that it is a
#: reading.  Item 4 is per seed and registered (5 of 5); the POOLED point estimate is not registered anywhere, so the
#: closest registered analogue governs: section 9 S1(c)'s confirmation rule, "the MEAN over the 5 withheld seeds of the
#: per-seed macro statistic, its interval a cluster bootstrap of that seed mean with the same resampled clusters applied
#: to every seed".  Because the design's macro statistic is a mean over units, that is IDENTICAL to building the paired
#: units from each unit's seed-mean MAE and bootstrapping them once -- which is what this code does, so no separate
#: resampling loop can disagree with it.
SEED_POOLING = (
    "READING (a POST-HOC addendum is REQUESTED): section 8 registers the paired cluster bootstrap and section 15 "
    "registers 5 withheld seeds, but no registered sentence says how R19 items 1, 2, 3 and 5 combine the seeds -- only "
    "item 4 is per seed. This run takes section 9 S1(c)'s confirmation rule as the analogue: the point estimate is the "
    "MEAN over the 5 withheld seeds of the per-seed unit-macro Delta, and its interval is the registered cluster "
    "bootstrap (10,000 resamples, seed 19) applied to the seed-mean per-unit MAE, i.e. THE SAME resampled clusters for "
    "every seed. Since the macro statistic is a mean over units, mean-over-seeds-of-macro equals "
    "macro-of-mean-over-seeds exactly, so this is one statistic, not a choice between two. Every per-seed Delta is "
    "reported beside it (seed_deltas_by_index), so a reader can see the spread the pooling hides")


def seed_mean_paired_units(per_seed: Mapping[int, D.PairedUnits], *, design: str, candidate: str, comparator: str
                           ) -> D.PairedUnits:
    """The paired units of the seed MEAN: each unit's MAE averaged over the withheld seeds (:data:`SEED_POOLING`)."""
    idx = sorted(per_seed)
    if not idx:
        raise ValueError("no seed scored")
    first = per_seed[idx[0]]
    units = first.cand_mae.index
    for i in idx:
        if not per_seed[i].cand_mae.index.equals(units):
            raise AssertionError(f"seed index {i}: different averaging units from seed index {idx[0]}; a claim's units "
                                 "are the confirmation half's scored cells and do not depend on the colouring")
    cand = sum(per_seed[i].cand_mae for i in idx) / len(idx)
    comp = sum(per_seed[i].comp_mae for i in idx) / len(idx)
    return D.PairedUnits(design=design, candidate=candidate, comparator=comparator, cand_mae=cand, comp_mae=comp,
                         clusters=first.clusters, n_rows=first.n_rows)


def score_claim_over_seeds(claim: CF.Claim, per_seed: Mapping[int, D.PairedUnits], *,
                           sensitivity_deltas: Mapping[str, float | str], deterministic: bool = False) -> dict[str, Any]:
    """R19 at confirmation for one frozen claim: the seed-mean bootstrap for items 1, 2, 3 and 5, the per-seed Deltas
    for item 4 (5 of 5), TOST under the primary cluster."""
    pooled = seed_mean_paired_units(per_seed, design=claim.design, candidate=claim.candidate,
                                    comparator=claim.comparator)
    boots = D.bootstraps(pooled, name=claim.key)
    out = CF.score_claim(claim, point=pooled.delta, bootstraps=boots,
                         seed_deltas={int(i): float(per_seed[i].delta) for i in sorted(per_seed)},
                         sensitivity_deltas=sensitivity_deltas, deterministic=deterministic)
    out["seed_pooling_reading"] = SEED_POOLING
    out["n_units"] = int(len(pooled.cand_mae))
    return out


# ============================================================================================= #
# the run
# ============================================================================================= #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--seed-store", default=None,
                    help="path OUTSIDE the repository holding the 5 withheld seeds; the ONLY way they enter this code")
    ap.add_argument("--dry-run", action="store_true", help="enumerate jobs, folds and the cost; fit nothing")
    ap.add_argument("--check-only", action="store_true", help="run the five gates and print the verdict; do nothing else")
    ap.add_argument("--resume", action="store_true",
                    help="complete unfitted folds of the recorded run (the only thing a second invocation may do)")
    ap.add_argument("--workers", type=int, default=1, help=f"at most {MAX_WORKERS}")
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def gate_verdict(out_root: Path, *, seed_store_path: str | None, resume: bool, check: Callable[[], int] | None = None,
                 digests: Callable[[], Mapping[str, Any]] | None = None, expect_addenda: int | None = None,
                 seal: Any = None, registry_path: Path | None = None, plan: Mapping[str, Any] | None = None
                 ) -> tuple[dict[str, Any], CF.SeedStore | None]:
    """The five gates.  (d) needs the store, so it is loaded here -- verified, and never printed."""
    code = code_digest()["combined"]
    store = None
    if seed_store_path is not None:
        store = CF.load_seed_store(seed_store_path, root=out_root, seal=seal if seal is not None else seal_module(),
                                   prereg_paths=(seal or seal_module()).PreregPaths(root=Path(out_root),
                                                                                    repo_root=paths.REPO_ROOT))
    g = CF.gate(out_root, seed_store=store, code_digest=code, resume=resume, check=check, digests=digests,
                expect_addenda=expect_addenda, plan=plan, registry_path=registry_path)
    return g, store


def main(argv: Sequence[str] | None = None, *, check: Callable[[], int] | None = None,
         digests: Callable[[], Mapping[str, Any]] | None = None, seal: Any = None,
         registry_path: Path | None = None) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    if int(ns.workers) > MAX_WORKERS:
        raise SystemExit(f"refused: --workers {ns.workers} exceeds {MAX_WORKERS} (the measured limit of this machine)")
    plan = CF.read_plan(CF.plan_path(out_root))
    log(f"plan: {plan['n_claims']} frozen claim(s) {plan['claim_ids']} (sha256 {plan['sha256'][:12]}...)")

    if ns.dry_run:
        jobs = CF.enumerate_jobs(plan)
        cost = CF.cost_estimate(jobs)
        folds = [p.record() for p in build_confirmation_folds(None, out_root, dry_run=True)]
        body = {"mode": "dry_run", "claims": plan["claim_ids"], "n_jobs": cost["n_jobs"], "n_folds": cost["n_folds"],
                "serial_hours": cost["serial_hours"], "wall_hours_2_workers": cost["wall_hours_2_workers"],
                "by_purpose_serial_hours": cost["by_purpose_serial_hours"], "fold_files_to_build": folds,
                "not_run": {"v6_actinide_deltas": CF.V6_ACTINIDE_DELTAS_NOT_RUN["status"],
                            "power_check": CF.POWER_CHECK_NOT_RUN["status"]},
                "jobs": cost["jobs"], "basis": cost["basis"],
                "run_stages": [{"stage": n, "implemented": ok, "what": why} for n, ok, why in RUN_STAGES],
                "note": "no seed was read: --dry-run enumerates from the plan and the measured unit costs alone"}
        print(json.dumps(body, indent=2, default=str))
        return 0

    g, store = gate_verdict(out_root, seed_store_path=ns.seed_store, resume=bool(ns.resume), check=check,
                            digests=digests, expect_addenda=ns.expect_addenda, seal=seal, registry_path=registry_path,
                            plan=plan)
    if ns.check_only:
        print(json.dumps({"mode": "check_only", **{k: v for k, v in g.items() if k != "readings"}}, indent=2,
                         default=str))
        return 0
    assert store is not None      # the gate refuses without a verified store
    log(f"gates passed; seed store verified against {store.digest[:12]}... ({CF.N_SEEDS} seeds, values withheld)")
    todo = next((n for n, ok, _ in RUN_STAGES if not ok), None)
    if todo is not None:
        # refuse BEFORE a manifest, a fold file or the lock is written: an incomplete runner must leave the single
        # registered run unspent (section 15), not half-spent
        raise SystemExit(refusal(todo))
    with Run(NAME, args={k: v for k, v in vars(ns).items() if k != "seed_store"}, seed=None,
             extra={"stage": STAGE, "seed_commitment_sha256": store.digest, "plan_sha256": plan["sha256"],
                    "claims": plan["claim_ids"], "readings": CF.READINGS,
                    "not_run": {"v6_actinide_deltas": CF.V6_ACTINIDE_DELTAS_NOT_RUN,
                                "power_check": CF.POWER_CHECK_NOT_RUN}}) as run:
        run.extra["gate"] = {k: v for k, v in g.items() if k != "readings"}
        build_confirmation_folds(store, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
