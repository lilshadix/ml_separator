"""``scripts/g19_run_ladder.py`` -- the ablation ladder M3 -> M4 -> M5 -> M6 -> M7 after discovery (pre-registration
sections 6 ladder table and "M-model training settings", 7 compute plan items 3-6 (order, stop rule, budget and demotion
order), 8 R19, 12 uncertainty, 13 domain status; POST-HOC addendum 1 items 1-4 and readings 6(a)-(g); brief sections
14, 29, 31, 32).

Gate (refuses to start unless ALL hold; ``--check-only`` prints the verdict)
--------------------------------------------------------------------------
0. Order (task X finding VL2-04): the file checks -- 2(a), 2(b) and 3 -- run BEFORE the corpus is loaded
   (``discovery_complete_cheap``, ``decision_files_ready``); only when they hold is the corpus loaded for 2(c).
1. ``scripts/g19_seal_prereg.py --check`` exits 0 and the sealed digest / addendum text are the registered ones
   (``g19_run_discovery.refuse_unless_sealed``).
2. **Discovery is complete** (:func:`discovery_complete`): (a) the discovery runner's progress ledger
   (``manifests/run_info/g19_run_discovery_progress.json``, or ``evaluation/discovery/run_info_progress.json`` under
   another ``--out-root``) lists the final-stage marker ``discovery.NOT_IMPLEMENTED`` ("M3+ not implemented", written
   only when ``run_plan`` reached stage ``10_not_implemented``) and no job with errors; (b)
   ``evaluation/discovery/records_index.csv`` exists (written at the end of ``g19_run_discovery.main``); (c) every
   ``fit`` job of the current plan (``discovery.enumerate_plan(PlanState.read(plan_state.json))``) has, for every
   fittable fold, a record whose digest is the one the current code, fold file and plan state produce and that holds
   both steps ``point`` and ``intervals`` (``discovery.resume_status`` with ``g19_run_discovery.fold_digest``).
3. **The scorer has decided the stop rule**: ``evaluation/discovery/decisions/stop_rule.json`` exists with ``stop`` in
   {true, false} (``discovery.stop_rule``; ``null`` = pending is refused), and ``decisions/decisions.json`` ->
   ``ladder`` holds boolean ``kept`` for M1 and M2 (the ladder's retained base).

What runs
---------
* stop = false: M3 -> M4 -> M5 -> M6 -> M7 in section 7 item 3 order, each on the seed-104729 main designs the
  discovery plan gave M2 (V5-primary at the heavy scheme, V1 at the heavy scheme, V2 element exact; the SAME JobSpecs
  with the arm renamed), tuned per outer fold on the addendum-1 inner design (``g19_run_discovery.validation_splits``:
  ``inner_design.SimultaneousInnerCells`` for V5) with the predecessor's retained values fixed per outer fold
  (``models.ladder.tune_step``).  After each step the keep / remove rule of section 6 is evaluated with the scorer's
  R19 machinery on the selection half, seed 104729: R19 items 1-3, 5 and the scoring-filter sensitivities on
  V5-primary (``discovery.SCOPES['ladder']``) against the retained predecessor, plus TOST non-inferiority (epsilon 0.05)
  on V1 and V2 (``discovery.ladder_step``).  A removed step is skipped by the next step's chain; an UNDECIDED step (a
  design frame missing, ``kept`` null) STOPS the ladder with status ``undecided`` -- section 6 removes a step only on
  evidence, so nothing builds past it until the missing input is resolved (task X finding V-05).  M7 (5 members at the
  retained configuration) is judged on calibration (section 12 / 9 S1(d) bands) with non-inferior MAE.  After M7 the
  registered H5 component ablations (M6a / M6b toggled at the retained configuration) and, when M3 is kept, M3's V5
  strict / HNO3-only refits (addendum 1 item 4) run; the H4 / H5 / ladder rows of ``M3 vs M2`` are then RE-EVALUATED with
  the refit frames present (``reevaluate_step_after_refits``; ``contrasts_M3.csv`` is rewritten and
  ``contrasts_M3_before_refits.csv`` keeps the earlier rows), so R19 item 6 of H4 M3 - M2 uses the reduced set as the
  addendum intends (task X finding V-04).  The M6 physics terms record whether they had anything to act on
  (``component_activity``): an M6 decision or an H5 M6a / M6b toggle whose term is inactive on every fold is labelled
  ``vacuous`` instead of being read as a component on / off comparison (task X finding VL2-07).
* stop = true: M3-M6 are marked ``exploratory_not_run`` (section 7 item 4) and only M7 runs, on the retained
  configuration of discovery (M2 if kept, else M1); H7 is not run (the process chain is another runner).
* Budget -- **POST-HOC addendum 2, "2. Ladder M3-M7" > "Budget"**: "the ladder has its own budget of 40 h of wall clock
  (``evaluation/ladder/decisions/wall_clock.json``), checked before each step; the demotion order M7 -> M6 -> M5 -> M4 ->
  M3 and every other rule of section 7 items 3-6 are unchanged; the discovery ledger stays as registered for the discovery
  stages."  This runner's own ledger alone (:func:`budget_used_seconds`) is compared with :data:`LADDER_BUDGET_HOURS`
  before each step; the discovery wall clock (``evaluation/discovery/decisions/wall_clock.json``) is recorded beside it
  for information and is NOT added (the earlier one-ledger reading is replaced, :data:`READINGS` ``budget``).  On
  exhaustion the steps not yet run are demoted in the order M7 -> M6 -> M5 -> M4 -> M3 (everything after the current
  point) and reported so.

Digest registry (addendum 2 item 5): the seal gate and the discovery code digest prefer ``manifests/digest_registry.json``
(``gen19ct.evaluation.registry``, stage ``ladder``) when it exists and fall back to the constants of
``evaluation.discovery`` otherwise.

Records: ``evaluation/ladder/<arm>/<design>__<variant>_<scheme>/s<seed>/<fold>.parquet + .json`` in the discovery
record schema (``discovery.PREDICTION_COLUMNS``; JSON with ``job``, ``digest``, ``fold_hash``, ``arm_record``,
``steps``), digest = ``g19_run_discovery.fold_digest`` with the ladder extras (this module's code digest, the expected
digests of the predecessor chain, the keep flags, the stop rule), so a change of code, fold file, plan state,
predecessor record or ladder decision makes them stale.  Decisions: ``evaluation/ladder/decisions/ladder.json`` (the
resumable ladder state), ``contrasts_<step>.csv``, ``evaluation/ladder/M7/metrics.json``, ``tables/ladder_*.csv``,
``decisions/D04_mechanism_experts.md`` and ``decisions/D05_uncertainty.md`` (brief section 29 format, generated from
the files; "not computed" where an input is absent).  Nothing reads the confirmation half or V6; every scored index
passes the discovery guards (``g19_run_discovery.prepare_fold``).

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_ladder.py --check-only
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_ladder.py --workers 2
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import calibration as EC  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import metrics as EM  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.evaluation import transfer as ET  # noqa: E402
from gen19ct.folds import io as FI  # noqa: E402
from gen19ct.manifest import Run, write_csv, write_json, write_text  # noqa: E402
from gen19ct.models import ladder as LAD  # noqa: E402
from gen19ct.models import neural as NN  # noqa: E402

NAME = "g19_run_ladder"
SCRIPTS = Path(__file__).resolve().parent
PREDICTION_STEP, INTERVAL_STEP = "point", "intervals"
#: this runner's own prediction-affecting files (digested beside the discovery code digest)
LADDER_CODE_FILES: tuple[Path, ...] = (paths.G19_ROOT / "gen19ct" / "models" / "ladder.py", Path(__file__).resolve())
LADDER_STEPS: tuple[str, ...] = LAD.STEPS
STAGES: dict[str, str] = {"M3": "L03_M3", "M4": "L04_M4", "M5": "L05_M5", "M6": "L06_M6", "M7": "L07_M7",
                          "H5": "L08_H5_ablations", "M3_refits": "L09_M3_refit_sensitivities"}
LADDER_ORDER: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
STATUS_DONE: tuple[str, ...] = ("kept", "removed", "complete", "judged")
STATUS_SKIPPED: tuple[str, ...] = ("exploratory_not_run", "not_run", "demoted")
DEMOTION_ORDER: tuple[str, ...] = D.DEMOTION_ORDER          # M7 -> M6 -> M5 -> M4 -> M3 (addendum 2: unchanged)
#: POST-HOC addendum 2, "2. Ladder M3-M7" > "Budget": "the ladder has its own budget of 40 h of wall clock
#: (evaluation/ladder/decisions/wall_clock.json), checked before each step" -- this runner's ledger alone; the discovery
#: ledger (``discovery.BUDGET_HOURS`` = 60 h) "stays as registered for the discovery stages" and is no longer added
#: one source for the figure: ``discovery.LADDER_BUDGET_HOURS``, which the scorer's budget block also reports
LADDER_BUDGET_HOURS = D.LADDER_BUDGET_HOURS
BUDGET_HOURS = LADDER_BUDGET_HOURS
DISCOVERY_BUDGET_HOURS = D.BUDGET_HOURS
SCHEMA = "gen19.ladder.v1"
#: the registry stage of this runner (addendum 2 item 5)
REGISTRY_STAGE = "ladder"
#: readings of this runner where the sealed text is silent or was changed by an addendum (written with every output)
READINGS: dict[str, str] = {
    "budget": "POST-HOC addendum 2, '2. Ladder M3-M7' > 'Budget' (a compute-driven change to section 7 item 5, chosen by "
              "the user on 2026-09-19 before any registered contrast was scored): 'the ladder has its own budget of 40 h of "
              "wall clock (evaluation/ladder/decisions/wall_clock.json), checked before each step; the demotion order M7 -> "
              "M6 -> M5 -> M4 -> M3 and every other rule of section 7 items 3-6 are unchanged; the discovery ledger stays as "
              "registered for the discovery stages' -- budget_used_seconds is this runner's ledger alone; the discovery wall "
              "clock is recorded as discovery_hours_not_counted and never added; --max-hours stays an operator pause, "
              "never a demotion",
    "digest_registry": "addendum 2 item 5: the seal gate and the discovery code digest are read from "
                       "manifests/digest_registry.json (stage 'ladder' / 'discovery') when it exists, else from the "
                       "constants of evaluation.discovery (gen19ct.evaluation.registry)",
}


def _load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RD = _load_script("g19_run_discovery")
SC = _load_script("g19_score_discovery")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================================= #
# paths, digests, readers
# ============================================================================================= #

def ladder_root(out_root: Path) -> Path:
    return Path(out_root) / "evaluation" / "ladder"


def ladder_fold_paths(out_root: Path, job: D.JobSpec, arm: str, fold_id: str) -> tuple[Path, Path]:
    """``(parquet, json)`` of one ladder arm's record of one outer fold (the discovery layout under ``evaluation/ladder``)."""
    d = ladder_root(out_root) / arm / job.design_dir / f"s{job.seed}"
    name = D.safe_fold_name(fold_id)
    return d / f"{name}.parquet", d / f"{name}.json"


def ladder_state_path(out_root: Path) -> Path:
    return ladder_root(out_root) / "decisions" / "ladder.json"


def ladder_wall_clock_path(out_root: Path) -> Path:
    return ladder_root(out_root) / "decisions" / "wall_clock.json"


def stop_rule_path(out_root: Path) -> Path:
    return D.discovery_root(out_root) / "decisions" / "stop_rule.json"


def decisions_path(out_root: Path) -> Path:
    return D.discovery_root(out_root) / "decisions" / "decisions.json"


def progress_path(out_root: Path) -> Path:
    """Where ``g19_run_discovery._write_progress`` wrote the discovery ledger for this ``out_root``."""
    if Path(out_root).resolve() == paths.G19_ROOT.resolve():
        return paths.MANIFESTS_DIR / "run_info" / "g19_run_discovery_progress.json"
    return D.discovery_root(out_root) / "run_info_progress.json"


def ladder_code_digest() -> dict[str, str]:
    """``own`` = this script + ``models/ladder.py``; ``discovery`` = the discovery runner's code digest (every record
    carries both; ``combined`` is their SHA-256)."""
    own = D.code_digest(LADDER_CODE_FILES)["combined"]
    disc = REG.discovery_code_digest()          # the registry's discovery entry when it exists, else the live digest
    return {"own": own, "discovery": disc, "combined": D.sha256_text(f"{disc}|{own}")}


def read_stop_rule(out_root: Path) -> dict[str, Any] | None:
    return D.read_record(stop_rule_path(out_root))


def read_discovery_ladder(out_root: Path) -> dict[str, Any] | None:
    body = D.read_record(decisions_path(out_root))
    if body is None or not isinstance(body.get("ladder"), Mapping):
        return None
    return dict(body["ladder"])


def discovery_complete_cheap(out_root: Path) -> dict[str, Any]:
    """Gate items 2(a)-(b): the progress ledger's final-stage marker without job errors and ``records_index.csv`` --
    file reads only, run before the corpus is loaded (task X finding VL2-04)."""
    reasons: list[str] = []
    prog = D.read_record(progress_path(out_root))
    if prog is None:
        reasons.append(f"progress ledger missing: {progress_path(out_root)}")
    else:
        markers = [str(m) for m in (prog.get("markers") or [])]
        if D.NOT_IMPLEMENTED not in markers:
            reasons.append(f"the discovery run has not reached its final stage (marker {D.NOT_IMPLEMENTED!r} absent "
                           "from the progress ledger)")
        bad = sorted(k for k, v in (prog.get("jobs") or {}).items() if isinstance(v, Mapping) and v.get("errors"))
        if bad:
            reasons.append(f"discovery jobs with errors: {bad[:5]}")
    if not (D.discovery_root(out_root) / "records_index.csv").exists():
        reasons.append("evaluation/discovery/records_index.csv missing (written at the end of g19_run_discovery.main)")
    return {"complete": not reasons, "reasons": reasons, "progress_path": str(progress_path(out_root))}


def decision_files_ready(out_root: Path) -> dict[str, Any]:
    """Gate item 3 (file reads only): ``stop_rule.json`` decided and ``decisions.json`` -> ``ladder`` with boolean
    ``kept`` for M1 and M2; ``SystemExit`` otherwise."""
    stop = read_stop_rule(out_root)
    if stop is None or stop.get("stop") not in (True, False):
        raise SystemExit(f"refused: {stop_rule_path(out_root)} is missing or its 'stop' is not decided (true / false); "
                         "run scripts/g19_score_discovery.py after discovery (section 7 item 4)")
    lad = read_discovery_ladder(out_root)
    kept = {} if lad is None else {s: (lad.get(s) or {}).get("kept") for s in ("M1", "M2")}
    if lad is None or any(kept.get(s) not in (True, False) for s in ("M1", "M2")):
        raise SystemExit(f"refused: {decisions_path(out_root)} lacks boolean ladder.kept for M1 and M2 (the scorer's "
                         "section 6 decisions the ladder builds on)")
    return {"stop_rule": {"stop": bool(stop["stop"]), "path": str(stop_rule_path(out_root))},
            "discovery_ladder": {"M1": bool(kept["M1"]), "M2": bool(kept["M2"])}}


def discovery_complete(out_root: Path, corpus: Any, state: D.PlanState, code: str, *,
                       runners: Mapping[str, Any] | None = None, jobs: Sequence[D.JobSpec] | None = None,
                       steps: Sequence[str] = (PREDICTION_STEP, INTERVAL_STEP)) -> dict[str, Any]:
    """The discovery-completeness check of the module docstring (gate item 2).  ``jobs`` defaults to the ``fit`` jobs
    of the current plan; tests pass synthetic ones."""
    cheap = discovery_complete_cheap(out_root)
    reasons: list[str] = list(cheap["reasons"])
    fit_jobs = list(jobs) if jobs is not None else [j for j in D.enumerate_plan(state) if j.kind == "fit"]
    runners = runners or RD.default_runners()
    incomplete: list[dict[str, Any]] = []
    n_folds = 0
    for job in fit_jobs:
        try:
            tasks = corpus.fittable(job)
            dh = corpus.design_hash(job.stem)
            runner = RD.runner_for(job, runners)
        except Exception as exc:                                    # noqa: BLE001 -- reported, not raised
            incomplete.append({"job": job.key, "error": f"{type(exc).__name__}: {exc}"})
            continue
        missing = []
        for f, k in tasks:
            n_folds += 1
            digest = RD.fold_digest(job, f, code, state, runner, out_root, ordinal=k, design_hash=dh)
            st = D.resume_status(job, f, out_root, digest, steps)
            if not st["complete"]:
                missing.append({"fold_id": f.fold_id, "stale": st["stale"], "missing": st["missing"]})
        if missing:
            incomplete.append({"job": job.key, "n_folds": len(tasks), "n_incomplete": len(missing), "first": missing[:3]})
    if incomplete:
        reasons.append(f"{len(incomplete)} discovery fit job(s) without a complete, current record for every fittable "
                       f"fold: {[i['job'] for i in incomplete[:5]]}")
    return {"complete": not reasons, "reasons": reasons, "n_fit_jobs": len(fit_jobs), "n_folds_checked": n_folds,
            "incomplete_jobs": incomplete, "progress_path": str(progress_path(out_root)),
            "rule": "progress ledger reached the final stage without errors; records_index.csv written; every fit "
                    "job of the current plan has a current record with steps point + intervals for every fittable fold"}


def refuse_unless_ready(out_root: Path, corpus: Any, state: D.PlanState, disc_code: str, *,
                        prereg: Mapping[str, Any] | None = None, jobs: Sequence[D.JobSpec] | None = None,
                        runners: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Gate items 2-3 (item 1, the seal, is ``g19_run_discovery.refuse_unless_sealed`` and is passed in as ``prereg``)
    in the cheap-first order of the module docstring: the ledger / index files, the decision files, then the record
    check.  ``corpus`` may be a zero-argument factory, called only once the file checks hold (task X finding VL2-04)."""
    cheap = discovery_complete_cheap(out_root)
    if not cheap["complete"]:
        raise SystemExit("refused: the discovery run is not complete -- " + "; ".join(cheap["reasons"]) +
                         " (the ladder builds on discovery's M1 / M2 records and the scorer's stop-rule decision; "
                         "section 7 items 3-4; nothing heavy was loaded)")
    dec = decision_files_ready(out_root)
    if callable(corpus):
        corpus = corpus()
    comp = discovery_complete(out_root, corpus, state, disc_code, runners=runners, jobs=jobs)
    if not comp["complete"]:
        raise SystemExit("refused: the discovery run is not complete -- " + "; ".join(comp["reasons"]) +
                         " (the ladder builds on discovery's M1 / M2 records and the scorer's stop-rule decision; "
                         "section 7 items 3-4)")
    return {"prereg_gate": None if prereg is None else dict(prereg), "discovery_complete": comp, **dec}


# ============================================================================================= #
# the ladder state (resumable decisions)
# ============================================================================================= #

@dataclass
class LadderState:
    stop_rule: bool
    discovery_ladder: dict[str, bool]                      # M1 / M2 kept
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)
    demoted: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def kept(self, step: str) -> bool | None:
        if step == "M0":
            return True
        if step in ("M1", "M2"):
            return bool(self.discovery_ladder.get(step))
        rec = self.steps.get(step)
        return None if rec is None else rec.get("kept")

    def retained_before(self, step: str) -> str:
        """The last kept step before ``step`` in the ladder order (M0 when none)."""
        i = LADDER_ORDER.index(step) if step in LADDER_ORDER else len(LADDER_ORDER)
        for prev in reversed(LADDER_ORDER[:i]):
            if prev == "M0":
                return "M0"
            if self.kept(prev):
                return prev
        return "M0"

    def status(self, step: str) -> str | None:
        rec = self.steps.get(step)
        return None if rec is None else rec.get("status")

    def decision_digest(self, step: str) -> str:
        """Digest of the keep flags of every step before ``step`` (part of the ladder record digest)."""
        i = LADDER_ORDER.index(step) if step in LADDER_ORDER else len(LADDER_ORDER)
        flags = {s: self.kept(s) for s in LADDER_ORDER[1:i]}
        return D.sha256_text(json.dumps({"stop_rule": self.stop_rule, "kept": flags}, sort_keys=True))

    def record(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "stop_rule": self.stop_rule, "discovery_ladder": dict(self.discovery_ladder),
                "steps": self.steps, "demoted": list(self.demoted), "notes": list(self.notes),
                "retained_final": self.retained_before("M7") if self.kept("M7") is not True else "M7",
                "label": "selection-half ladder decisions (seed 104729), optimistically biased; never confirmed"}

    @classmethod
    def read(cls, path: Path) -> "LadderState | None":
        body = D.read_record(path)
        if body is None:
            return None
        return cls(stop_rule=bool(body["stop_rule"]), discovery_ladder={k: bool(v) for k, v in body["discovery_ladder"].items()},
                   steps=dict(body.get("steps") or {}), demoted=list(body.get("demoted") or []),
                   notes=list(body.get("notes") or []))

    def write(self, path: Path) -> Path:
        return write_json(path, _json(self.record()))


def _json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def ladder_plan(state: LadderState) -> list[tuple[str, str]]:
    """``(step, why)`` in section 7 item 3 order; under the stop rule M3-M6 are exploratory-not-run and M7 runs."""
    if state.stop_rule:
        return [(s, "exploratory_not_run: section 7 item 4 stop rule (neither M2 nor B6 passed R19 items 1-3, 5 "
                    "against B3i)") for s in ("M3", "M4", "M5", "M6")] + \
               [("M7", "section 7 item 4: M7 run once on the retained configuration for H6")]
    return [(s, "registered ladder step (section 6)") for s in LADDER_STEPS]


# ============================================================================================= #
# jobs
# ============================================================================================= #

def discovery_main_jobs(state: D.PlanState, arm: str = "M2") -> list[D.JobSpec]:
    """The seed-104729 main-design fit jobs discovery gave ``arm`` (V5-primary, V1, V2 at the plan's schemes)."""
    return [j for j in D.enumerate_plan(state) if j.kind == "fit" and j.arm == arm and j.variant in ("primary", "copy", "element")
            and j.stage in (D.STAGES["p_m1"], D.STAGES["p_m2"])]


def ladder_jobs(step: str, state: D.PlanState, *, refits: bool = False) -> list[D.JobSpec]:
    """The ladder step's jobs: discovery's M2 main-design JobSpecs with the arm renamed (identical folds, schemes,
    fold seeds and run seed); ``refits``: M3's V5 strict / HNO3-only refits (addendum 1 item 4)."""
    base = discovery_main_jobs(state, "M2")
    if not base:
        raise RuntimeError("the discovery plan has no M2 main-design job (plan state pending?)")
    if refits:
        v5 = next((j for j in base if j.design == "V5"), None)
        if v5 is None:
            return []
        return [replace(v5, arm=step, writes=(step,), variant=v, stage=STAGES["M3_refits"],
                        group=f"{step} refit sensitivities (seed 104729)",
                        purpose=f"H4 M3 - M2: R19 item 6 {D.V5_SETTING_SENSITIVITY[v]} (addendum 1 item 4)")
                for v in D.LEARNED_REFIT_VARIANTS]
    stage = STAGES.get(step, STAGES["H5"])
    return [replace(j, arm=step, writes=(step,), stage=stage, group=f"{step} main designs (seed 104729)",
                    purpose=f"ladder step {step} (section 6)" if step in LADDER_STEPS else
                    f"H5 component ablation {step} (section 6 chemistry-priors question)") for j in base]


# ============================================================================================= #
# the runner (one arm)
# ============================================================================================= #

class LadderRunner:
    """M3-M7 and the H5 ablation arms on the discovery runner's fold protocol (``point`` / ``calibration`` /
    ``digest_extra``)."""

    has_interval_step = True

    def __init__(self, step: str, lstate: LadderState, codes: Mapping[str, str], *, tune_kwargs: Mapping[str, Any] | None = None,
                 min_expert_rows: int = LAD.MIN_EXPERT_ROWS, n_members: int = LAD.N_MEMBERS):
        if step not in LAD.ALL_ARMS:
            raise ValueError(f"unknown ladder arm {step!r}")
        self.step, self.lstate, self.codes = step, lstate, dict(codes)
        self.arms = (step,)
        self.tune_kwargs = dict(tune_kwargs or {})
        self.min_expert_rows = int(min_expert_rows)
        self.n_members = int(n_members)              # 5 registered; tests shrink it

    # ---- the predecessor chain -------------------------------------------------------------- #
    def chain_steps(self) -> list[str]:
        """The neural base (M2 if kept else M1) and every kept ladder step before this arm."""
        base = "M2" if self.lstate.kept("M2") else ("M1" if self.lstate.kept("M1") else None)
        if base is None:
            return []
        upto = LADDER_ORDER.index(self.step) if self.step in LADDER_ORDER else LADDER_ORDER.index("M7")
        return [base] + [s for s in ("M3", "M4", "M5", "M6") if LADDER_ORDER.index(s) < upto and self.lstate.kept(s)]

    def _discovery_job(self, job: D.JobSpec, arm: str) -> D.JobSpec:
        return replace(job, arm=arm, writes=(arm,))

    def expected_chain_digests(self, job: D.JobSpec, fold: FI.Fold, code: str, state: D.PlanState, out_root: Path, *,
                               ordinal: int, design_hash: str) -> list[dict[str, str]]:
        out = []
        for s in self.chain_steps():
            j = self._discovery_job(job, s)
            if s in ("M1", "M2"):
                dg = RD.fold_digest(j, fold, code, state, RD.NeuralRunner(s), out_root, ordinal=ordinal, design_hash=design_hash)
            else:
                dg = RD.fold_digest(j, fold, code, state, LadderRunner(s, self.lstate, self.codes), out_root,
                                    ordinal=ordinal, design_hash=design_hash)
            out.append({"step": s, "expected_digest": dg})
        return out

    def digest_extra(self, job: D.JobSpec, fold: FI.Fold, code: str, state: D.PlanState, out_root: Path, *,
                     ordinal: int, design_hash: str) -> dict[str, Any]:
        return {"ladder_code": self.codes["own"], "ladder_decisions": self.lstate.decision_digest(self.step),
                "predecessor_chain": self.expected_chain_digests(job, fold, code, state, out_root, ordinal=ordinal,
                                                                 design_hash=design_hash),
                "ladder_arm": self.step}

    def predecessor_config(self, fc: Any) -> tuple[LAD.LadderConfig, int, list[dict[str, Any]]]:
        """The outer fold's retained configuration and epoch count from the verified predecessor records."""
        chain = self.chain_steps()
        if not chain:
            raise RuntimeError(f"{self.step}: the retained predecessor is M0 (B5); the neural ladder has no base "
                               "(neither M1 nor M2 was kept)")
        cfg, epochs, recs = None, None, []
        for s in chain:
            j = self._discovery_job(fc.job, s)
            if s in ("M1", "M2"):
                pq, js = D.fold_paths(fc.out_root, j, s, fc.fold.fold_id)
                want = RD.fold_digest(j, fc.fold, fc.code, fc.state or D.PlanState(), RD.NeuralRunner(s), fc.out_root,
                                      ordinal=fc.ordinal, design_hash=fc.design_hash)
            else:
                pq, js = ladder_fold_paths(fc.out_root, j, s, fc.fold.fold_id)
                want = RD.fold_digest(j, fc.fold, fc.code, fc.state or D.PlanState(), LadderRunner(s, self.lstate, self.codes),
                                      fc.out_root, ordinal=fc.ordinal, design_hash=fc.design_hash)
            rec = D.read_record(js)
            if rec is None or not pq.exists() or PREDICTION_STEP not in (rec.get("steps") or {}):
                raise RuntimeError(f"{fc.job.key}/{fc.fold.fold_id}: the predecessor {s} has no record of this fold")
            if rec.get("digest") != want or rec.get("fold_hash") != fc.fold.fold_hash:
                raise D.StaleRecordError(f"{fc.job.key}/{fc.fold.fold_id}: the predecessor {s} record is stale (digest "
                                         "or fold hash differs from the current code, fold file, plan state and ladder "
                                         "decisions)")
            sel = rec["arm_record"]["selected"]
            if s in ("M1", "M2"):
                cfg = LAD.LadderConfig.from_neural(NN.NeuralConfig(int(sel["emb_dim"]), float(sel["weight_decay"]),
                                                                   int(sel.get("rank", 0))))
            else:
                cfg = LAD.LadderConfig.from_record(sel)
            epochs = int(rec["arm_record"]["n_epochs"])
            recs.append({"step": s, "digest": rec.get("digest"), "config": cfg.record(), "n_epochs": epochs})
        return cfg, epochs, recs

    # ---- the point step ---------------------------------------------------------------------- #
    def point(self, fc: Any) -> dict[str, Any]:
        c = fc.corpus
        rows = c.frame.loc[c.table.index[fc.mask]]
        prev_cfg, prev_epochs, chain = self.predecessor_config(fc)
        vs = RD.validation_splits(fc)
        inner_used = sorted({int(sp.inner_fold) for sp in vs})
        t0 = time.perf_counter()
        pred_block = {"step": prev_cfg.step, "config": prev_cfg.record(), "n_epochs": prev_epochs, "chain": chain,
                      "retained_before": self.lstate.retained_before(self.step)}
        if self.step in ("M3", "M4", "M5", "M6"):
            res = LAD.tune_step(self.step, rows, vs, prev_cfg, fold_index=fc.ordinal, condition_vectors=c.cv,
                                systems=c.systems, min_expert_rows=self.min_expert_rows, **self.tune_kwargs)
            arm = res.arm(rows=c.frame, condition_vectors=c.cv, systems=c.systems).fit(rows)
            pred = arm.predict(c.frame.loc[fc.sc_labels])
            cols = tuple(col for cs in arm.encoder.neural.block_columns.values() for col in cs)
            fits = res.fits
            rec = {"selected": res.selected.record(), "selected_label": res.selected.label(), "n_epochs": int(res.n_epochs),
                   "model_seed": int(res.model_seed), "scores": res.scores.to_dict(orient="records"),
                   "fits": fits.to_dict(orient="records"),
                   "split_unit_errors": res.split_unit_errors.to_dict(orient="records"), "inner_folds_used": inner_used,
                   "fit_record": arm.fit_record(), "predecessor": pred_block, "search": LAD.SEARCHED_FIELDS[self.step],
                   "grid": [x.record() for x in LAD.step_grid(self.step, prev_cfg)],
                   "inner_design": RD.inner_design_record(fc), "registration_choices": LAD.REGISTRATION_CHOICES}
            label, seed = res.selected.label(), int(res.model_seed)
        elif self.step == "M7":
            cfg = LAD.step_grid("M7", prev_cfg)[0]
            ens = LAD.EnsembleArm(cfg, n_epochs=prev_epochs, fold_index=fc.ordinal, rows=c.frame, condition_vectors=c.cv,
                                  systems=c.systems, min_expert_rows=self.min_expert_rows, n_members=self.n_members).fit(rows)
            pred = ens.predict(c.frame.loc[fc.sc_labels])
            cols = tuple(col for cs in ens.members[0].encoder.neural.block_columns.values() for col in cs)
            rec = {"selected": cfg.record(), "selected_label": cfg.label(), "n_epochs": int(prev_epochs),
                   "model_seed": int(ens.seeds[0]), "member_seeds": ens.seeds, "inner_folds_used": inner_used,
                   "fit_record": ens.fit_record(), "predecessor": pred_block, "search": [],
                   "inner_design": RD.inner_design_record(fc), "registration_choices": LAD.REGISTRATION_CHOICES}
            label, seed = cfg.label(), int(ens.seeds[0])
        else:                                                   # H5 ablation arms at the retained configuration
            cfg = LAD.h5_toggle_config(self.step, prev_cfg)
            arm = LAD.LadderArm(cfg, n_epochs=prev_epochs, model_seed=NN.registered_model_seed(fc.ordinal), rows=c.frame,
                                condition_vectors=c.cv, systems=c.systems, min_expert_rows=self.min_expert_rows).fit(rows)
            pred = arm.predict(c.frame.loc[fc.sc_labels])
            cols = tuple(col for cs in arm.encoder.neural.block_columns.values() for col in cs)
            rec = {"selected": cfg.record(), "selected_label": cfg.label(), "n_epochs": int(prev_epochs),
                   "model_seed": int(arm.model_seed), "inner_folds_used": inner_used, "fit_record": arm.fit_record(),
                   "predecessor": pred_block, "search": [], "toggled": self.step,
                   "inner_design": RD.inner_design_record(fc), "registration_choices": LAD.REGISTRATION_CHOICES}
            label, seed = cfg.label(), int(arm.model_seed)
        secs = time.perf_counter() - t0
        D.assert_no_provenance_features(cols, f"{self.step} inputs")
        return {self.step: RD.ArmOutput(pred=pred, selected_config=label, model_seed=seed, record=rec,
                                        feature_columns=cols, seconds=secs)}

    # ---- the intervals step ------------------------------------------------------------------ #
    def calibration(self, fc: Any, rec: Mapping[str, Any]) -> tuple[Any | None, dict]:
        full, avail = RD._calibration_splits(fc)
        plan = D.calibration_folds(avail, fc.job.inner_mode, tuned=self.step in ("M3", "M4", "M5", "M6"))
        RD._check_tuned_folds(fc, plan, rec["inner_folds_used"]) if plan["tuning_folds"] else None
        if not plan["calibration_folds"]:
            return None, plan
        c = fc.corpus
        arms, sels = {}, {}
        if self.step == "M7":
            cfg = LAD.LadderConfig.from_record(rec["selected"])
            for j in plan["calibration_folds"]:
                arms[j] = LAD.EnsembleArm(cfg, n_epochs=int(rec["n_epochs"]), fold_index=fc.ordinal, rows=c.frame,
                                          condition_vectors=c.cv, systems=c.systems, min_expert_rows=self.min_expert_rows,
                                          n_members=self.n_members)
            return LAD.NormalisedCrossFitConformal(arms, RD._restricted(full, plan["calibration_folds"])), plan
        for j in plan["calibration_folds"]:
            if self.step in ("M3", "M4", "M5", "M6") and plan["cross_fit"]:
                cfg, epochs, sel = LAD.select_ladder_excluding_folds(rec["scores"], rec["fits"], rec["split_unit_errors"], (j,))
            else:
                cfg, epochs = LAD.LadderConfig.from_record(rec["selected"]), int(rec["n_epochs"])
                sel = {"config": cfg.label(), "n_epochs": epochs, "fixed": True}
            arms[j] = LAD.LadderArm(cfg, n_epochs=epochs, model_seed=int(rec["model_seed"]), rows=c.frame,
                                    condition_vectors=c.cv, systems=c.systems, min_expert_rows=self.min_expert_rows)
            sels[j] = sel
        return D.CrossFitResidualConformal(arms, RD._restricted(full, plan["calibration_folds"]), selections=sels), plan


def ladder_runners(lstate: LadderState, codes: Mapping[str, str], **kw: Any) -> dict[str, LadderRunner]:
    return {a: LadderRunner(a, lstate, codes, **kw) for a in LAD.ALL_ARMS}


# ============================================================================================= #
# one fold (g19_run_discovery.run_fold with the ladder paths)
# ============================================================================================= #

def ladder_resume_status(job: D.JobSpec, fold: FI.Fold, out_root: Path, digest: str, steps: Sequence[str]) -> dict[str, Any]:
    done, stale, missing = [], [], []
    for arm in job.writes:
        pq, js = ladder_fold_paths(out_root, job, arm, fold.fold_id)
        rec = D.read_record(js)
        if rec is None or not pq.exists():
            missing.append(arm)
        elif rec.get("digest") != digest:
            stale.append(arm)
        elif not all(s in (rec.get("steps") or {}) for s in steps):
            missing.append(arm)
        else:
            done.append(arm)
    return {"complete": len(done) == len(job.writes), "done": done, "stale": stale, "missing": missing,
            "steps_done": {a: sorted((D.read_record(ladder_fold_paths(out_root, job, a, fold.fold_id)[1]) or {}).get("steps", {}))
                           for a in job.writes}}


def run_ladder_fold(job: D.JobSpec, fold: FI.Fold, ordinal: int, corpus: Any, out_root: Path, *, steps: Sequence[str],
                    runner: LadderRunner, code: str, state: D.PlanState, prereg: Mapping[str, Any] | None = None,
                    guard_fn: Callable = None, inner_check: Callable = None) -> dict:
    """Run (or skip) one fold of one ladder job: the discovery guards (``g19_run_discovery.prepare_fold``), the point
    step, the intervals step; records under ``evaluation/ladder``."""
    design_hash = corpus.design_hash(job.stem)
    digest = RD.fold_digest(job, fold, code, state, runner, out_root, ordinal=ordinal, design_hash=design_hash)
    status = ladder_resume_status(job, fold, out_root, digest, steps)
    if status["complete"]:
        return {"job": job.key, "fold_id": fold.fold_id, "status": "skipped_done"}
    t_all = time.perf_counter()
    kw = {}
    if guard_fn is not None:
        kw["guard_fn"] = guard_fn
    if inner_check is not None:
        kw["inner_check"] = inner_check
    fc, info = RD.prepare_fold(job, fold, ordinal, corpus, out_root, state, code=code, **kw)
    base = {"schema": D.SCHEMA, "ladder_schema": SCHEMA, "job": job.record(), "fold_id": fold.fold_id, "fold_hash": fold.fold_hash,
            "stem": job.stem, "design_hash": design_hash, "digest": digest, "code_digest": code,
            "ladder_code_digest": runner.codes["own"], "fold_ordinal": ordinal, "model_fold_number": ordinal,
            "model_seed": RD.model_seed_of(ordinal), "batching_label": fc.batching_label,
            "prereg_sha256": D.REGISTERED_PREREG_SHA256, "prereg_addenda_sha256": (prereg or {}).get("addenda_sha256"),
            "prereg_n_addenda": (prereg or {}).get("n_addenda"), "addendum_implemented": REG.addenda_count(REGISTRY_STAGE),
            "registry_stage": REGISTRY_STAGE,
            "prereg_gate": dict(prereg) if prereg else None, "ladder_decisions_digest": runner.lstate.decision_digest(runner.step),
            "stop_rule": runner.lstate.stop_rule, **info}
    need_point = any(PREDICTION_STEP not in (status["steps_done"].get(a) or []) for a in job.writes) or status["stale"]
    records: dict[str, dict] = {}
    if need_point:
        outputs = runner.point(fc)
        if set(outputs) != set(job.writes):
            raise AssertionError(f"{job.key}: the runner wrote {sorted(outputs)}, the job declares {list(job.writes)}")
        rss = D.peak_rss_bytes()
        for arm, o in outputs.items():
            frame = D.prediction_frame(o.pred, job=job, arm=arm, fold=fold, ordinal=ordinal, row_ids=fc.sc_ids,
                                       selected_config=o.selected_config, model_seed=o.model_seed,
                                       intervals_status="pending", batching_label=fc.batching_label, fit_seconds=o.seconds)
            D.assert_v6_clean(corpus.labels_of(frame["row_id"]), corpus.v6, f"{job.key}/{fold.fold_id}/{arm} written")
            D.assert_selection_rows(frame["row_id"], corpus.half_by_id[job.design], f"{job.key}/{fold.fold_id}/{arm} written")
            steps_rec = {PREDICTION_STEP: {"seconds": round(o.seconds, 3), "peak_rss_bytes": rss,
                                           "date_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}}
            rec = {**base, "arm": arm, "selected_config": o.selected_config, "model_seed": o.model_seed,
                   "arm_record": o.record, "n_feature_columns": len(o.feature_columns), "steps": steps_rec}
            pq, js = ladder_fold_paths(out_root, job, arm, fold.fold_id)
            RD._atomic_parquet(frame, pq)
            RD._atomic_json(rec, js)
            records[arm] = rec
    else:
        for arm in job.writes:
            records[arm] = D.read_record(ladder_fold_paths(out_root, job, arm, fold.fold_id)[1])
    if INTERVAL_STEP in steps:
        for arm in job.writes:
            rec = records[arm]
            if INTERVAL_STEP in rec.get("steps", {}):
                continue
            t0 = time.perf_counter()
            cal, plan = runner.calibration(fc, rec["arm_record"])
            pq, js = ladder_fold_paths(out_root, job, arm, fold.fold_id)
            frame = pd.read_parquet(pq)
            if cal is None:
                frame = frame.assign(intervals_status=plan["status"])
                rec["steps"][INTERVAL_STEP] = {"seconds": round(time.perf_counter() - t0, 3), "status": plan["status"],
                                               "plan": plan, "n_calibration": 0, "method": "not calibrated"}
            else:
                cal.fit_table(corpus.table, fc.mask, fc.ctx)
                if isinstance(cal, LAD.NormalisedCrossFitConformal):
                    frame = LAD.attach_normalised_intervals(frame, cal.quantiles, len(cal.residuals))
                    method = "models.ladder.NormalisedCrossFitConformal (section 12 normalised split conformal, inner folds)"
                else:
                    frame = D.attach_intervals(frame, cal.quantiles, len(cal.residuals))
                    method = "discovery.CrossFitResidualConformal (cross-fitted inner splits)"
                rec["steps"][INTERVAL_STEP] = {"seconds": round(time.perf_counter() - t0, 3), "peak_rss_bytes": D.peak_rss_bytes(),
                                               "status": "calibrated", "plan": plan, "method": method, **cal.record()}
            D.assert_v6_clean(corpus.labels_of(frame["row_id"]), corpus.v6, f"{job.key}/{fold.fold_id}/{arm} intervals")
            RD._atomic_parquet(frame, pq)
            RD._atomic_json(rec, js)
    return {"job": job.key, "fold_id": fold.fold_id, "status": "fitted", "seconds": round(time.perf_counter() - t_all, 3)}


# ============================================================================================= #
# verified readers (the discovery reader rooted at evaluation/ladder)
# ============================================================================================= #

def expected_ladder_records(job: D.JobSpec, folds: Sequence[FI.Fold], *, code: str, state: D.PlanState, out_root: Path,
                            excluded_ids: Iterable[str], runner: LadderRunner) -> dict[str, dict]:
    dh = FI.design_hash(list(folds))
    return {f.fold_id: {"digest": RD.fold_digest(job, f, code, state, runner, out_root, ordinal=k, design_hash=dh),
                        "fold_hash": f.fold_hash} for f, k in D.fittable_folds(job, folds, excluded_ids)}


def read_ladder_record_set(out_root: Path, arm: str, design_dir: str, seed: int, *, expected: Mapping[str, Mapping[str, str]],
                           steps: Sequence[str] = (PREDICTION_STEP,)) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """``discovery.read_discovery_record_set`` on the ladder records (same verification, same refusals)."""
    d = ladder_root(out_root) / arm / design_dir / f"s{seed}"
    exp = {str(k): dict(v) for k, v in expected.items()}
    status: dict[str, Any] = {"arm": arm, "design_dir": design_dir, "seed": int(seed), "n_expected": len(exp)}
    if not d.exists():
        return None, {**status, "status": "missing", "n_found": 0}
    frames, found = [], set()
    by_name = {D.safe_fold_name(k): k for k in exp}
    for pq in sorted(d.glob("*.parquet")):
        rec = D.read_record(pq.with_suffix(".json"))
        if rec is None:
            raise D.StaleRecordError(f"{pq}: prediction without a JSON record")
        fid = str(rec.get("fold_id"))
        if fid not in exp or by_name.get(pq.stem) != fid:
            raise D.StaleRecordError(f"{pq}: fold {fid!r} is not a fittable fold of the current job ({arm}/{design_dir}/s{seed})")
        if rec.get("digest") != exp[fid]["digest"]:
            raise D.StaleRecordError(f"{pq}: record digest differs from the current code, fold file, plan state or "
                                     "ladder decisions (rerun the job)")
        if rec.get("fold_hash") != exp[fid]["fold_hash"]:
            raise D.StaleRecordError(f"{pq}: fold hash differs from the registered fold")
        missing_steps = [x for x in steps if x not in (rec.get("steps") or {})]
        if missing_steps:
            raise D.StaleRecordError(f"{pq}: record lacks step(s) {missing_steps}")
        fr = pd.read_parquet(pq)
        if len(fr) and (fr["fold_id"].astype(str) != fid).any():
            raise D.StaleRecordError(f"{pq}: parquet rows of another fold")
        frames.append(fr)
        found.add(fid)
    status.update(n_found=len(found))
    if found != set(exp):
        return None, {**status, "status": "incomplete", "missing_folds": sorted(set(exp) - found)[:10]}
    out = pd.concat(frames, ignore_index=True) if frames else None
    if out is not None and (out["half"].astype(str) != D.SELECTION).any():
        raise AssertionError(f"{d}: a stored prediction of a non-selection-half row")
    return out, {**status, "status": "complete"}


def verified_ladder_predictions(out_root: Path, arm: str, design_dir: str, seed: int, *, code: str, state: D.PlanState,
                                excluded_ids: Iterable[str], runner: LadderRunner, folds_dir: Path | None = None,
                                steps: Sequence[str] = (PREDICTION_STEP,), fold_cache: dict | None = None
                                ) -> tuple[pd.DataFrame | None, dict]:
    d = ladder_root(out_root) / arm / design_dir / f"s{seed}"
    jsons = sorted(d.glob("*.json")) if d.exists() else []
    if not jsons:
        return None, {"arm": arm, "design_dir": design_dir, "seed": int(seed), "status": "missing"}
    bodies = [D.read_record(j) for j in jsons]
    if any(b is None or "job" not in b for b in bodies):
        raise D.StaleRecordError(f"{d}: an unreadable record")
    keys = {b["job"].get("key") for b in bodies}
    if len(keys) != 1:
        raise D.StaleRecordError(f"{d}: records of {len(keys)} different jobs")
    job = D.job_from_record(bodies[0])
    if arm not in job.writes or job.design_dir != design_dir or int(job.seed) != int(seed):
        raise D.StaleRecordError(f"{d}: the records' job {job.key} does not write {arm}/{design_dir}/s{seed}")
    cache = fold_cache if fold_cache is not None else {}
    if job.stem not in cache:
        cache[job.stem] = FI.read_design(job.stem, folds_dir or paths.FOLDS_DIR)
    expected = expected_ladder_records(job, cache[job.stem], code=code, state=state, out_root=out_root,
                                       excluded_ids=excluded_ids, runner=runner)
    return read_ladder_record_set(out_root, arm, design_dir, seed, expected=expected, steps=steps)


# ============================================================================================= #
# scoring: the scorer's Store with the ladder arms, the ladder contrasts, the M7 uncertainty metrics
# ============================================================================================= #

class LadderStore(SC.Store):
    """``g19_score_discovery.Store`` whose ladder arms (M3-M7, the H5 ablations) come from ``evaluation/ladder``."""

    def __init__(self, *args: Any, lstate: LadderState, codes: Mapping[str, str], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.lstate, self.codes = lstate, dict(codes)
        self.ladder_runners = ladder_runners(lstate, codes)

    def verified(self, arm: str, design_dir: str, seed: int, steps: Sequence[str] = ("point",)) -> pd.DataFrame | None:
        if arm not in LAD.ALL_ARMS:
            return super().verified(arm, design_dir, seed, steps)
        pred, st = verified_ladder_predictions(self.out_root, arm, design_dir, int(seed), code=self.code, state=self.state,
                                               excluded_ids=self.excluded_ids, runner=self.ladder_runners[arm],
                                               folds_dir=self.folds_dir, steps=steps, fold_cache=self._folds)
        self.record_sets[f"ladder:{arm}/{design_dir}/s{seed}"] = {k: v for k, v in st.items() if k != "arm"}
        return pred


def evaluate_ladder_spec(store: SC.Store, spec: D.ContrastSpec, delta5: float, *, v2_summary: str = "focus7"
                         ) -> dict[str, Any] | None:
    """``g19_score_discovery.evaluate_spec`` for the ladder arms (the scorer skips every M3 contrast while M3 is
    'not implemented'; this copy evaluates it).  Learned-arm rules of addendum 1 apply (seed 104729 only, item 4
    NOT_EVALUATED, reduced sensitivity set)."""
    design = spec.design
    kw = {"v2_summary": v2_summary}
    primary = SC._pair(store, spec.candidate, spec.comparator, design, "primary", D.PRIMARY_SEED, **kw)
    if primary is None:
        return None
    learned = SC.is_learned(spec.candidate, spec.comparator)
    seeds = {}
    for s in (D.PLAN_SEEDS if learned else D.DISCOVERY_SEEDS):
        pu = primary if s == D.PRIMARY_SEED else SC._pair(store, spec.candidate, spec.comparator, design, "primary", s, **kw)
        seeds[s] = None if pu is None else pu.delta
    sens: dict[str, Any] = {}
    reg = ET.REGISTERED_SENSITIVITIES[design]
    reduced: list[str] = []
    for name in D.SCORING_FILTER_SENSITIVITIES:
        if name in reg:
            pu = SC._pair(store, spec.candidate, spec.comparator, design, "primary", D.PRIMARY_SEED, filt=name, **kw)
            sens[name] = ET.UNTESTABLE if pu is None else pu.delta
            reduced.append(name)
    reasons: dict[str, str] = {}
    not_run = D.LEARNED_REFITS_NOT_RUN.get(design, {}) if learned else {}
    if design == "V5":
        for setting, name in D.V5_SETTING_SENSITIVITY.items():
            if name in not_run:
                sens[name] = ET.UNTESTABLE
                reasons[name] = not_run[name]
                continue
            pu = SC._pair(store, spec.candidate, spec.comparator, design, setting, D.PRIMARY_SEED)
            sens[name] = ET.UNTESTABLE if pu is None else pu.delta
            reduced.append(name)
    elif design in ("V1", "V2"):
        table = D.V1_REFIT_SETTINGS if design == "V1" else D.V2_REFIT_SETTINGS
        for setting, (_, _, name) in table.items():
            if name in not_run:
                sens[name] = ET.UNTESTABLE
                reasons[name] = not_run[name]
                continue
            if design == "V1" and setting in D.V1_GROUPING_SETTINGS and learned:
                sens[name] = ET.UNTESTABLE
                reasons[name] = SC.GROUPING_UNTESTABLE
                continue
            pu = SC._pair(store, spec.candidate, spec.comparator, design, setting, D.PRIMARY_SEED, **kw)
            sens[name] = ET.UNTESTABLE if pu is None else pu.delta
            reduced.append(name)
    res = D.evaluate_contrast(name=spec.name, family=spec.family, design=design, primary=primary,
                              margin=D.margin_value(spec, delta5), seed_deltas=seeds, sensitivities=sens,
                              learned=learned, reduced_sensitivities=sorted(set(reduced)) if learned else None)
    res["note"] = spec.note
    res["untestable_reason"] = {k: reasons.get(k, D.REFIT_NOT_RUN) for k, v in res["sensitivities"].items() if v == ET.UNTESTABLE}
    res["batching_label"] = D.heavy_v5_batching_label((spec.candidate, spec.comparator), design, store.state)
    res["reported_verdict"] = res["r19"].verdict
    return res


def step_contrasts(step: str, pred: str) -> list[D.ContrastSpec]:
    """The ladder contrast of ``step`` vs its retained predecessor on V5 / V1 / V2 (family ``ladder``), plus the
    registered H4 / H5 rows the same numbers answer (section 19): H4 M3 - M2, H5 M3 vs M2 (V1, V2, V5), H5 M4 on/off."""
    out = [D.ContrastSpec("ladder", step, pred, d, "delta5" if d == "V5" else "0.05",
                          f"ladder step {step} vs retained predecessor {pred}") for d in ("V5", "V1", "V2")]
    if step == "M3" and pred == "M2":
        out.append(D.ContrastSpec("H4", "M3", "M2", "V5", "delta5", "architecture question (section 6): M3 - M2"))
        out += [D.ContrastSpec("H5", "M3", "M2", d, "delta5" if d == "V5" else "0.05", "chemistry priors: mechanism routing")
                for d in ("V5", "V1", "V2")]
    if step == "M4":
        out += [D.ContrastSpec("H5", "M4", pred, d, "delta5" if d == "V5" else "0.05", "chemistry priors: source hierarchy on/off")
                for d in ("V5", "V1", "V2")]
    return out


def decide_step(step: str, lstate: LadderState, store: SC.Store, delta5: float) -> tuple[dict[str, Any], list[dict]]:
    """Section 6 keep / remove of ``step`` against its retained predecessor (``discovery.ladder_step``), from the
    verified records; returns the decision and the contrast rows."""
    pred = lstate.retained_before(step)
    per: dict[str, Any] = {}
    rows: list[dict] = []
    for spec in step_contrasts(step, pred):
        r = evaluate_ladder_spec(store, spec, delta5)
        if r is None:
            continue
        if spec.family == "ladder":
            per[spec.design] = r
        for rec in D.contrast_rows(r):
            rec["key"] = f"{spec.name}@{spec.design}#{spec.family}"
            rows.append(rec)
    dec = D.ladder_step(step, pred, per.get("V5"), None if per.get("V1") is None else per["V1"]["tost"],
                        None if per.get("V2") is None else per["V2"]["tost"])
    dec["contrast_summary"] = {d: {"point": r["point"], "verdict_ladder": r["scopes"]["ladder"]["verdict"],
                                   "r19_full": r["r19"].verdict, "tost": r["tost"]["verdict"], "n_units": r["n_units"]}
                               for d, r in per.items()}
    dec["missing_designs"] = [d for d in ("V5", "V1", "V2") if d not in per]
    return dec, rows


def h5_contrasts(lstate: LadderState) -> list[D.ContrastSpec]:
    """H5 M6a / M6b on/off: the toggled arm vs the retained configuration (M6 when kept, else its predecessor)."""
    retained = "M6" if lstate.kept("M6") else lstate.retained_before("M6")
    if retained in ("M0",):
        return []
    return [D.ContrastSpec("H5", arm, retained, d, "delta5" if d == "V5" else "0.05",
                           f"chemistry priors: {'monotonicity hinge' if arm == 'M6a_toggle' else 'series smoothness'} "
                           f"toggled vs the retained configuration {retained}")
            for arm in LAD.H5_ABLATION_ARMS for d in ("V5", "V1", "V2")]


def component_activity(out_root: Path, arm: str, jobs: Sequence[D.JobSpec], corpus: Any) -> dict[str, Any]:
    """Whether the M6 terms of ``arm`` had anything to act on in its fold records (``fit_record.hinge_active`` /
    ``smoothness_active`` of ``models.ladder``; M7: any member): per term the number of records whose configuration
    carries it, the number where it was active, and ``vacuous`` (configured everywhere it appears, active nowhere).  A
    term absent from every record is neither configured nor vacuous (task X finding VL2-07)."""
    counts = {"hinge": {"n_configured": 0, "n_active": 0}, "smoothness": {"n_configured": 0, "n_active": 0}}
    n_records = n_missing = 0
    for job in jobs:
        for fold, _ in corpus.fittable(job):
            rec = D.read_record(ladder_fold_paths(out_root, job, arm, fold.fold_id)[1])
            if rec is None:
                n_missing += 1
                continue
            n_records += 1
            fr = (rec.get("arm_record") or {}).get("fit_record") or {}
            for term, key in (("hinge", "hinge_active"), ("smoothness", "smoothness_active")):
                v = fr.get(key)
                if v is None:
                    continue
                counts[term]["n_configured"] += 1
                counts[term]["n_active"] += int(bool(v))
    out: dict[str, Any] = {"arm": arm, "n_records": n_records, "n_records_missing": n_missing}
    for term, c in counts.items():
        out[term] = {**c, "vacuous": bool(c["n_configured"] > 0 and c["n_active"] == 0)}
    return out


def vacuity_of(*activities: Mapping[str, Any], terms: Sequence[str]) -> dict[str, Any]:
    """A contrast on ``terms`` is vacuous when, across the arms given, every term is configured somewhere and active
    nowhere -- the arms then differ by a term that never acted, so the comparison is not a component on / off test."""
    per: dict[str, Any] = {}
    for term in terms:
        conf = sum(int(a.get(term, {}).get("n_configured", 0)) for a in activities)
        act = sum(int(a.get(term, {}).get("n_active", 0)) for a in activities)
        per[term] = {"n_configured": conf, "n_active": act, "vacuous": bool(conf > 0 and act == 0)}
    vac = all(v["vacuous"] for v in per.values()) if per else False
    reason = ("vacuous (" + "; ".join(f"{t} inactive on every fold" for t, v in per.items() if v["vacuous"]) + "): the arms "
              "differ only by a term that never acted, so this is not a component on / off comparison") if vac else ""
    return {"vacuous": vac, "terms": per, "reason": reason}


def reevaluate_step_after_refits(step: str, lstate: LadderState, store: SC.Store, delta5: float, out_root: Path
                                 ) -> dict[str, Any]:
    """Re-evaluate every contrast of ``step`` vs its retained predecessor once the step's V5 strict / HNO3-only refit
    frames exist (addendum 1 item 4): R19 item 6 of the H4 / H5 rows then covers the reduced sensitivity set instead of
    UNTESTABLE.  The keep decision is untouched: the ladder scope never reads a refit setting, which is asserted
    (``ladder_scope_unchanged``).  ``contrasts_<step>.csv`` is rewritten, the earlier rows kept as
    ``contrasts_<step>_before_refits.csv`` (task X finding V-04)."""
    pred = lstate.retained_before(step)
    dec_dir = ladder_root(out_root) / "decisions"
    before = dec_dir / f"contrasts_{step}.csv"
    if before.exists() and not (dec_dir / f"contrasts_{step}_before_refits.csv").exists():
        write_text(dec_dir / f"contrasts_{step}_before_refits.csv", before.read_text(encoding="utf-8"))
    prior = ((lstate.steps.get(step) or {}).get("decision") or {}).get("contrast_summary") or {}
    rows: list[dict] = []
    per: dict[str, Any] = {}
    now_run: set[str] = set()
    for spec in step_contrasts(step, pred):
        r = evaluate_ladder_spec(store, spec, delta5)
        if r is None:
            continue
        if spec.family == "ladder":
            per[spec.design] = r
        for name, v in r["sensitivities"].items():
            if name in D.V5_SETTING_SENSITIVITY.values() and v != ET.UNTESTABLE:
                now_run.add(name)
        for rec in D.contrast_rows(r):
            rec["key"] = f"{spec.name}@{spec.design}#{spec.family}"
            rec["refit_sensitivities_evaluated"] = True
            rows.append(rec)
    unchanged = all(prior.get(d, {}).get("verdict_ladder") == r["scopes"]["ladder"]["verdict"] for d, r in per.items()
                    if d in prior)
    if not unchanged:
        raise AssertionError(f"{step}: the ladder-scope verdict changed after the refits, yet the ladder scope reads no "
                             "refit setting -- a code defect")
    if rows:
        write_csv(pd.DataFrame(rows), dec_dir / f"contrasts_{step}.csv")
    return {"evaluated": True, "n_rows": len(rows), "ladder_scope_unchanged": unchanged,
            "refit_sensitivities_now_run": sorted(now_run), "note": "H4 / H5 / ladder rows of "
            f"{step} vs {pred} re-evaluated with the strict / HNO3-only refit frames (addendum 1 item 4)"}


# ---- M7 ------------------------------------------------------------------------------------- #

def _spearman_stat(fr: pd.DataFrame) -> float:
    err = np.abs(fr[EM.PRED_COL].to_numpy(dtype=float) - fr[EM.Y_COL].to_numpy(dtype=float))
    return EM.spearman_rho(err, fr["std_logD"].to_numpy(dtype=float))


def coverage_by_domain_status_for(store: SC.Store, fr: pd.DataFrame, arm: str, design_dir: str, seed: int, *,
                                  out_root: Path) -> pd.DataFrame | None:
    """Sections 12 / 13 coverage per domain-status category of one arm's V5-primary frame, from the discovery
    ``_support`` files of the same fold file, seed and folds (identical folds; ``g19_run_discovery.support_digest``)."""
    sup_dir = D.discovery_root(out_root) / "_support" / design_dir / f"s{seed}"
    if not sup_dir.exists():
        return None
    rec = next((D.read_record(j) for j in sorted((ladder_root(out_root) / arm / design_dir / f"s{seed}").glob("*.json"))), None)
    if rec is None:
        return None
    job = D.job_from_record(rec)
    folds = store._folds.setdefault(job.stem, FI.read_design(job.stem, store.folds_dir))
    for f, _ in D.fittable_folds(job, folds, store.excluded_ids):
        js = sup_dir / f"{D.safe_fold_name(f.fold_id)}.json"
        body = D.read_record(js)
        if body is None or not js.with_suffix(".parquet").exists():
            return None
        if body.get("digest") != store.rd.support_digest(job, f, store.code) or body.get("fold_hash") != f.fold_hash:
            raise D.StaleRecordError(f"{js}: support file of another code or fold file")
    sup = pd.concat([pd.read_parquet(p, columns=["row_id", "fold_id", "domain_status", "domain_status_ambiguous",
                                                 "support_score"]) for p in sorted(sup_dir.glob("*.parquet"))],
                    ignore_index=True)
    key = sup.set_index(["fold_id", "row_id"])
    idx = pd.MultiIndex.from_arrays([fr["fold_id"].astype(str), fr["row_id"].astype(str)])
    if not idx.isin(key.index).all():
        return None
    lab = key.reindex(idx)
    fr = fr.assign(domain_status=lab["domain_status"].to_numpy(), support_score=lab["support_score"].to_numpy(),
                   domain_status_ambiguous=lab["domain_status_ambiguous"].astype(bool).to_numpy())
    n_amb = int(fr["domain_status_ambiguous"].sum())
    reg = EM.Regime(design="V5", arm=arm, variant="primary", half="selection", seed=int(seed), seed_set="discovery")
    tab = EC.coverage_by_category(fr[~fr["domain_status_ambiguous"]], reg, category_col="domain_status",
                                  unit_cols=EM.CELL_COLS, v6_mask=store.v6)
    tab["n_rows_domain_status_ambiguous_excluded"] = n_amb
    return tab


def uncertainty_metrics(store: SC.Store, lstate: LadderState, out_root: Path, delta5: float) -> dict[str, Any]:
    """Section 12 for M7 on V5 / V1 / V2 (selection half, seed 104729): coverage 50 / 80 / 95, width, CRPS,
    Spearman(|error|, SD) with the primary-cluster bootstrap, the SD reliability curve, coverage by domain-status
    category (V5), the S1(d) / F3 bands and the 'knows when it does not know' test; MAE non-inferiority (TOST) vs the
    retained predecessor.  Every absent input is recorded as ``not computed``."""
    pred = lstate.retained_before("M7")
    out: dict[str, Any] = {"arm": "M7", "predecessor": pred, "seed": D.PRIMARY_SEED, "half": "selection", "designs": {},
                           "label": "discovery, optimistically biased (selection half); addendum 1: seed 104729 only",
                           "readings": LAD.REGISTRATION_CHOICES}
    tables: dict[str, pd.DataFrame] = {}
    cluster_of = {"V5": EM.SYSTEM_COL, "V1": EM.PUB_GROUP_COL, "V2": EM.METAL_STATE_COL}
    for design in ("V5", "V1", "V2"):
        fr = store.frame("M7", design, "primary", D.PRIMARY_SEED)
        block: dict[str, Any] = {}
        if fr is None:
            block["status"] = "not computed: no complete verified M7 record set"
            out["designs"][design] = block
            continue
        has_iv = np.isfinite(fr["lower_80"].to_numpy(dtype=float)).all()
        has_sd = np.isfinite(fr["std_logD"].to_numpy(dtype=float)).all()
        reg = EM.Regime(design=design, arm="M7", variant="primary", half="selection", seed=D.PRIMARY_SEED, seed_set="discovery")
        kw: dict[str, Any] = dict(v6_mask=store.v6)
        if design == "V1":
            kw.update(v1_scheme=store.v1_scheme("M7", design), v1_group_col=EM.PUB_GROUP_COL, remainder_groups=store.remainder_groups)
        block["n_rows"] = int(len(fr))
        if has_iv:
            iv = EM.design_interval_summary(fr, reg, exploratory_unit_readings=False, levels=EC.LEVELS,
                                            sd_col="std_logD" if has_sd else None, **kw)
            tables[f"intervals_{design}"] = iv
            block["interval_metrics"] = EM.as_mapping(iv)
        else:
            block["interval_metrics"] = "not computed: intervals absent"
        if has_sd:
            rel = EC.sd_reliability_table(fr, reg, v6_mask=store.v6, sd_col="std_logD")
            tables[f"sd_reliability_{design}"] = rel
            bs = ET.cluster_bootstrap_statistic(fr, cluster_of[design], _spearman_stat, contrast="spearman_abs_error_sd",
                                                cluster_unit=cluster_of[design], higher_is_better=True)
            lo, hi = bs.percentile_interval(0.95)
            block["spearman_abs_error_sd"] = {"point": bs.point, "low_95": lo, "high_95": hi, "cluster": cluster_of[design],
                                              "n_clusters": bs.n_clusters}
        else:
            block["spearman_abs_error_sd"] = "not computed: std_logD absent"
        if design == "V5" and has_iv:
            dd = store.design_dir("M7", "V5", "primary")
            cov = coverage_by_domain_status_for(store, fr, "M7", str(dd), D.PRIMARY_SEED, out_root=out_root)
            if cov is None:
                block["coverage_by_domain_status"] = "not computed: discovery _support files absent or not matching"
            else:
                tables["coverage_by_domain_status"] = cov
                prim = cov[cov["aggregation"] == "unit_macro"]
                by_cat: dict[str, dict[float, float]] = {}
                cat80: dict[str, tuple[float, int]] = {}
                for cat, sub in prim.groupby("category"):
                    by_cat[str(cat)] = {float(l): float(sub.loc[sub["metric"] == f"coverage_{int(round(l * 100))}", "value"].iloc[0])
                                        for l in EC.LEVELS}
                    cat80[str(cat)] = (by_cat[str(cat)][0.80], int(sub["category_units"].iloc[0]))
                block["coverage_by_domain_status"] = {k: {str(l): v for l, v in d.items()} for k, d in by_cat.items()}
                macro = {float(l): block["interval_metrics"].get(f"coverage_{int(round(l * 100))}", float("nan")) for l in EC.LEVELS}
                block["s1d_check"] = EC.s1d_check(macro, cat80)
                block["f3_check"] = EC.f3_check(macro[0.80], macro[0.95])
                sp = block.get("spearman_abs_error_sd")
                if isinstance(sp, Mapping):
                    block["knows_when_it_does_not_know"] = EC.knows_when_it_does_not_know(sp["point"], sp["low_95"], by_cat)
        if pred != "M0":
            spec = D.ContrastSpec("ladder", "M7", pred, design, "delta5" if design == "V5" else "0.05", "M7 MAE non-inferiority")
            r = evaluate_ladder_spec(store, spec, delta5)
            block["mae_tost_vs_predecessor"] = "not computed: predecessor frame absent" if r is None else \
                {k: r["tost"][k] for k in ("verdict", "non_inferior", "low_90", "high_90")} | {"point": r["point"]}
        out["designs"][design] = block
    v5 = out["designs"].get("V5", {})
    ni = [b.get("mae_tost_vs_predecessor", {}).get("non_inferior") if isinstance(b.get("mae_tost_vs_predecessor"), Mapping)
          else None for b in out["designs"].values()]
    s1d = v5.get("s1d_check", {}).get("pass") if isinstance(v5.get("s1d_check"), Mapping) else None
    out["verdict"] = {"mae_non_inferior_all_designs": (all(ni) if ni and None not in ni else None),
                      "calibration_s1d_pass": s1d,
                      "kept": (bool(all(ni)) and bool(s1d)) if (ni and None not in ni and s1d is not None) else None,
                      "rule": "section 6: M7 is judged on calibration (section 12; S1(d) bands on V5-primary) with "
                              "non-inferior MAE (TOST epsilon 0.05) on V5, V1 and V2 vs the retained configuration"}
    return {"metrics": out, "tables": tables}


# ============================================================================================= #
# budget and wall clock (section 7 item 5 as changed by POST-HOC addendum 2, "2. Ladder M3-M7" > "Budget")
# ============================================================================================= #

def ladder_wall_clock_total(out_root: Path) -> float:
    """Seconds of every earlier invocation of THIS runner (``evaluation/ladder/decisions/wall_clock.json``)."""
    body = D.read_record(ladder_wall_clock_path(out_root)) or {}
    return float(sum(float(i.get("seconds", 0.0)) for i in body.get("invocations", [])))


def ladder_budget_status(ladder_seconds: float, discovery_seconds: float | None = None) -> dict[str, Any]:
    """Addendum 2: "the ladder has its own budget of 40 h of wall clock ... checked before each step; the demotion order
    M7 -> M6 -> M5 -> M4 -> M3 ... unchanged; the discovery ledger stays as registered for the discovery stages" -- the
    ladder seconds against :data:`LADDER_BUDGET_HOURS`; the discovery hours are reported beside, never added."""
    st = D.budget_status(float(ladder_seconds), budget_hours=LADDER_BUDGET_HOURS)
    st.update({"ledger": "evaluation/ladder/decisions/wall_clock.json (this runner alone)",
               "discovery_hours_not_counted": None if discovery_seconds is None else round(float(discovery_seconds) / 3600.0, 4),
               "discovery_budget_hours_unchanged": DISCOVERY_BUDGET_HOURS, "reading": READINGS["budget"],
               "note": ("addendum 2 ('2. Ladder M3-M7' > 'Budget'): the ladder's own 40 h ledger; on exhaustion the steps "
                        "not yet run are demoted in the order M7 -> M6 -> M5 -> M4 -> M3 (section 7 item 5, unchanged); "
                        "--max-hours is an operator pause, never a demotion")})
    return st


def record_ladder_wall_clock(out_root: Path, started: str, seconds: float, steps_done: Sequence[str]) -> None:
    body = D.read_record(ladder_wall_clock_path(out_root)) or {"schema": SCHEMA, "invocations": []}
    inv = [i for i in body.get("invocations", []) if i.get("started_utc") != started]
    inv.append({"started_utc": started, "seconds": round(float(seconds), 1), "steps_done": list(steps_done)})
    body["invocations"] = inv
    ladder_s = sum(float(i["seconds"]) for i in inv)
    disc_s = RD.wall_clock_total(out_root)
    body["ladder_hours"] = round(ladder_s / 3600.0, 4)
    body["discovery_hours_not_counted"] = round(disc_s / 3600.0, 4)
    body["budget_hours"] = LADDER_BUDGET_HOURS
    body["budget"] = ladder_budget_status(ladder_s, disc_s)
    body["reading"] = READINGS["budget"]
    paths.ensure_dir(ladder_wall_clock_path(out_root).parent)
    RD._atomic_json(body, ladder_wall_clock_path(out_root))


def budget_used_seconds(out_root: Path) -> float:
    """Addendum 2: the ladder's own ledger alone -- the discovery wall clock is no longer added."""
    return ladder_wall_clock_total(out_root)


def demote_from(step: str) -> list[str]:
    """Section 7 item 5: the steps demoted when the budget is exhausted before ``step`` -- ``step`` and everything
    after it in the ladder, which is the registered order M7 -> M6 -> M5 -> M4 -> M3 read from the end."""
    i = LADDER_STEPS.index(step)
    return [s for s in DEMOTION_ORDER if s in LADDER_STEPS[i:]]


# ============================================================================================= #
# running the jobs of one step (fold-level parallelism; the pool loads its own corpus)
# ============================================================================================= #

W = SimpleNamespace(corpus=None)
_LOADED: dict[str, Any] = {}          # main's corpus, loaded by the gate's factory only after the file checks hold


def _init_worker(coext_ids: list[str]) -> None:
    import torch

    torch.set_num_threads(2)
    W.corpus = RD.load_corpus(coext_ids)


def _worker_fold(job: D.JobSpec, fold_id: str, ordinal: int, out_root: str, steps: list[str], code: str, state_rec: dict,
                 lstate_rec: dict, codes: dict, prereg: dict | None, tune_kwargs: dict, min_expert_rows: int) -> dict:
    try:
        corpus = W.corpus
        fold = next(f for f in corpus.folds(job.stem) if f.fold_id == fold_id)
        state = D.PlanState(**{k: state_rec[k] for k in ("v5_batched_check", "v1_tenfold_check", "guard_mode",
                                                            "freezing_candidates", "notes")})
        lstate = LadderState(stop_rule=bool(lstate_rec["stop_rule"]), discovery_ladder=dict(lstate_rec["discovery_ladder"]),
                             steps=dict(lstate_rec.get("steps") or {}))
        runner = LadderRunner(job.arm, lstate, codes, tune_kwargs=tune_kwargs, min_expert_rows=min_expert_rows)
        return run_ladder_fold(job, fold, ordinal, corpus, Path(out_root), steps=steps, runner=runner, code=code, state=state,
                               prereg=prereg)
    except Exception as exc:                                        # noqa: BLE001
        return {"job": job.key, "fold_id": fold_id, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]}


def run_step_jobs(jobs: Sequence[D.JobSpec], corpus: Any, out_root: Path, *, steps: Sequence[str], code: str, state: D.PlanState,
                  lstate: LadderState, codes: Mapping[str, str], workers: int = 1, coext_ids: Sequence[str] = (),
                  prereg: Mapping[str, Any] | None = None, tune_kwargs: Mapping[str, Any] | None = None,
                  min_expert_rows: int = LAD.MIN_EXPERT_ROWS, paused: Callable[[], bool] | None = None,
                  guard_fn: Callable = None, inner_check: Callable = None, stop_on_error: bool = True) -> dict[str, Any]:
    """Every fold of every job (plan order), ``workers`` folds in flight; returns the per-job ledger."""
    ledger: dict[str, Any] = {"jobs": {}, "stopped": None}
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(list(coext_ids),)) if workers > 1 else None
    prereg_rec = None if prereg is None else dict(prereg)
    tk = dict(tune_kwargs or {})
    try:
        for job in jobs:
            t0 = time.perf_counter()
            tasks = corpus.fittable(job)
            results = []
            if pool is None:
                runner = LadderRunner(job.arm, lstate, codes, tune_kwargs=tk, min_expert_rows=min_expert_rows)
                for f, k in tasks:
                    if paused is not None and paused():
                        ledger["stopped"] = f"--max-hours reached inside {job.key} (operator pause, resumable)"
                        break
                    try:
                        results.append(run_ladder_fold(job, f, k, corpus, out_root, steps=steps, runner=runner, code=code,
                                                       state=state, prereg=prereg_rec, guard_fn=guard_fn, inner_check=inner_check))
                    except Exception as exc:                        # noqa: BLE001
                        results.append({"job": job.key, "fold_id": f.fold_id, "status": "error",
                                        "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
                        if stop_on_error:
                            break
            else:
                pending, inflight, halt = iter(tasks), set(), False
                while True:
                    while not halt and len(inflight) < workers:
                        if paused is not None and paused():
                            ledger["stopped"] = f"--max-hours reached inside {job.key} (operator pause, resumable)"
                            halt = True
                            break
                        nxt = next(pending, None)
                        if nxt is None:
                            break
                        f, k = nxt
                        inflight.add(pool.submit(_worker_fold, job, f.fold_id, k, str(out_root), list(steps), code, state.record(),
                                                 lstate.record(), dict(codes), prereg_rec, tk, int(min_expert_rows)))
                    if not inflight:
                        break
                    done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                    for fu in done:
                        r = fu.result()
                        results.append(r)
                        if r["status"] == "error" and stop_on_error:
                            halt = True
            errs = [r for r in results if r["status"] == "error"]
            ledger["jobs"][job.key] = {"n_folds": len(tasks), "fitted": sum(r["status"] == "fitted" for r in results),
                                       "skipped": sum(r["status"] == "skipped_done" for r in results), "errors": errs[:3],
                                       "seconds": round(time.perf_counter() - t0, 1), "stage": job.stage,
                                       "complete": len(errs) == 0 and len(results) == len(tasks)}
            log(f"{job.key}: {ledger['jobs'][job.key]['fitted']} fitted, {ledger['jobs'][job.key]['skipped']} skipped, "
                f"{len(errs)} errors, {ledger['jobs'][job.key]['seconds']} s")
            if errs and stop_on_error:
                ledger["stopped"] = f"error in {job.key}: {errs[0]['error']}"
                break
            if ledger["stopped"]:
                break
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    return ledger


# ============================================================================================= #
# the ladder
# ============================================================================================= #

def build_store(out_root: Path, state: D.PlanState, lstate: LadderState, codes: Mapping[str, str]) -> LadderStore:
    attrs = SC.build_attrs()
    v6 = attrs["v6_target_row"].astype(bool)
    crossings = pd.read_csv(SC.CROSSINGS_CSV, dtype={"row_id": str, "partner_id": str})
    return LadderStore(out_root, attrs, v6, crossings, state, code=codes["discovery"], lstate=lstate, codes=codes)


def run_ladder(corpus: Any, out_root: Path, state: D.PlanState, codes: Mapping[str, str], gate: Mapping[str, Any], *,
               steps: Sequence[str], workers: int = 1, max_hours: float | None = None, only: str | None = None,
               coext_ids: Sequence[str] = (), prereg: Mapping[str, Any] | None = None,
               store_factory: Callable[[LadderState], SC.Store] | None = None, delta5: float | None = None,
               with_h5: bool = True, guard_fn: Callable = None, inner_check: Callable = None,
               tune_kwargs: Mapping[str, Any] | None = None, min_expert_rows: int = LAD.MIN_EXPERT_ROWS) -> dict[str, Any]:
    """M3 -> M7 (or M7 alone under the stop rule) with the keep / remove decision after each step, then the H5
    ablations and M3's refits; resumable through ``ladder.json`` and the per-fold records."""
    t0 = time.perf_counter()
    started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    lpath = ladder_state_path(out_root)
    lstate = LadderState.read(lpath)
    if lstate is None:
        lstate = LadderState(stop_rule=bool(gate["stop_rule"]["stop"]), discovery_ladder=dict(gate["discovery_ladder"]))
    elif lstate.stop_rule != bool(gate["stop_rule"]["stop"]) or lstate.discovery_ladder != dict(gate["discovery_ladder"]):
        raise SystemExit("refused: ladder.json was written under another stop-rule / discovery-ladder decision; the "
                         "discovery decisions changed after the ladder started (re-score, then remove evaluation/ladder)")
    lstate.write(lpath)
    ledger: dict[str, Any] = {"jobs": {}, "steps": [], "stopped": None, "budget": None, "prereg_gates": 1}
    prev_used = budget_used_seconds(out_root)

    def used_s() -> float:
        return prev_used + time.perf_counter() - t0

    def paused() -> bool:
        return max_hours is not None and (time.perf_counter() - t0) / 3600.0 >= max_hours

    store: SC.Store | None = None
    d5 = delta5 if delta5 is not None else D.registered_delta5()

    def get_store() -> SC.Store:
        nonlocal store
        if store is None:
            store = store_factory(lstate) if store_factory is not None else build_store(out_root, state, lstate, codes)
        return store

    done_steps: list[str] = []
    plan = ladder_plan(lstate)
    for step, why in plan:
        if only and step not in {s.strip() for s in only.split(",")}:
            continue
        cur = lstate.steps.get(step, {})
        if cur.get("status") in STATUS_DONE + STATUS_SKIPPED:
            done_steps.append(step)
            continue
        if why.startswith("exploratory_not_run"):
            lstate.steps[step] = {"step": step, "status": "exploratory_not_run", "kept": None, "note": why}
            lstate.write(lpath)
            done_steps.append(step)
            continue
        if used_s() / 3600.0 >= LADDER_BUDGET_HOURS:      # addendum 2: the ladder's own 40 h ledger, checked before each step
            dem = demote_from(step)
            ledger["budget"] = {**ladder_budget_status(used_s(), RD.wall_clock_total(out_root)), "exhausted_before": step,
                                "demoted": dem}
            for s in dem:
                if lstate.steps.get(s, {}).get("status") not in STATUS_DONE:
                    lstate.steps[s] = {"step": s, "status": "demoted", "kept": None,
                                       "note": f"section 7 item 5 as changed by POST-HOC addendum 2 ('2. Ladder M3-M7' > "
                                               f"'Budget'): the ladder's own {LADDER_BUDGET_HOURS:g} h wall-clock budget was "
                                               f"exhausted before {step}; demoted to exploratory in the order "
                                               f"{list(D.DEMOTION_ORDER)} (unchanged)"}
                    if s not in lstate.demoted:
                        lstate.demoted.append(s)
            lstate.write(lpath)
            log(f"budget exhausted before {step}: demoted {dem}")
            break
        pred = lstate.retained_before(step)
        if pred == "M0":
            lstate.steps[step] = {"step": step, "status": "not_run", "kept": None, "predecessor": "M0",
                                  "note": "the retained configuration is M0 (B5): neither M1 nor M2 was kept, so the neural "
                                          "ladder has no base (section 6 M3-M7 are components of the factorised model)"}
            lstate.write(lpath)
            done_steps.append(step)
            continue
        jobs = ladder_jobs(step, state)
        lstate.steps[step] = {**cur, "step": step, "status": "running", "kept": None, "predecessor": pred,
                              "jobs": [j.key for j in jobs], "note": why}
        lstate.write(lpath)
        led = run_step_jobs(jobs, corpus, out_root, steps=steps, code=codes["discovery"], state=state, lstate=lstate, codes=codes,
                            workers=workers, coext_ids=coext_ids, prereg=prereg, paused=paused, guard_fn=guard_fn,
                            inner_check=inner_check, tune_kwargs=tune_kwargs, min_expert_rows=min_expert_rows)
        ledger["jobs"].update(led["jobs"])
        lstate.steps[step]["ledger"] = _json(led["jobs"])
        if led["stopped"] or not all(v.get("complete") for v in led["jobs"].values()):
            lstate.steps[step]["status"] = "incomplete"
            lstate.steps[step]["note"] = led["stopped"] or "not every fold recorded"
            lstate.write(lpath)
            ledger["stopped"] = led["stopped"] or f"{step}: not every fold recorded"
            break
        if step == "M7" and INTERVAL_STEP not in steps:
            lstate.steps[step]["status"] = "incomplete"
            lstate.steps[step]["note"] = "point step only (--steps point): M7 is judged on its intervals (section 12)"
            lstate.write(lpath)
            ledger["stopped"] = "M7 needs the intervals step"
            break
        st = get_store()
        if step == "M7":
            um = uncertainty_metrics(st, lstate, out_root, d5)
            write_json(ladder_root(out_root) / "M7" / "metrics.json", _json(um["metrics"]))
            for name, tab in um["tables"].items():
                write_csv(tab, ladder_root(out_root) / "M7" / f"{name}.csv")
            lstate.steps[step].update({"status": "judged", "kept": um["metrics"]["verdict"]["kept"],
                                       "decision": um["metrics"]["verdict"], "metrics_path": str(ladder_root(out_root) / "M7" / "metrics.json")})
        else:
            dec, rows = decide_step(step, lstate, st, d5)
            if step == "M6":                                    # VL2-07: did the physics terms act at all?
                act = component_activity(out_root, "M6", jobs, corpus)
                vac = vacuity_of(act, terms=("hinge", "smoothness"))
                dec["component_activity"] = act
                dec["vacuous"] = vac["vacuous"]
                dec["vacuity"] = vac
                for r in rows:
                    r["vacuous"] = vac["vacuous"]
                    r["vacuous_reason"] = vac["reason"]
                if act["hinge"]["vacuous"] and not vac["vacuous"]:
                    dec["note"] = "hinge (M6a) inactive on every fold: only the smoothness term (M6b) acted in M6"
            if rows:
                write_csv(pd.DataFrame(rows), ladder_root(out_root) / "decisions" / f"contrasts_{step}.csv")
            status = "kept" if dec["kept"] else ("removed" if dec["kept"] is False else "undecided")
            lstate.steps[step].update({"status": status, "kept": dec["kept"], "decision": _json(dec)})
            if dec.get("vacuous"):
                lstate.steps[step]["note"] = dec["vacuity"]["reason"]
        lstate.write(lpath)
        ledger["steps"].append({"step": step, "kept": lstate.steps[step]["kept"], "status": lstate.steps[step]["status"]})
        done_steps.append(step)
        log(f"{step}: {lstate.steps[step]['status']} (predecessor {pred})")
        if lstate.steps[step]["status"] == "undecided":
            # V-05: section 6 removes a step only when it 'adds no strict-fold benefit'; an undecidable step (a design
            # frame missing) must not be skipped by the chain as if removed
            missing = (lstate.steps[step].get("decision") or {}).get("missing_designs")
            lstate.steps[step]["note"] = (f"keep decision undecided (design frames missing: {missing}); the ladder stops "
                                          "here -- resolve the missing input and rerun (resumable)")
            lstate.write(lpath)
            ledger["stopped"] = f"{step}: keep decision undecided (missing designs {missing}); section 6 gives no rule to build past it"
            record_ladder_wall_clock(out_root, started, time.perf_counter() - t0, done_steps)
            log(ledger["stopped"])
            break
        record_ladder_wall_clock(out_root, started, time.perf_counter() - t0, done_steps)
        if paused():
            ledger["stopped"] = f"--max-hours {max_hours} reached after {step} (operator pause, resumable)"
            break
    # ---- after the ladder: H5 component ablations and M3's refit sensitivities (registered contrasts of section 19)
    if not ledger["stopped"] and with_h5 and not lstate.stop_rule and lstate.kept("M7") is not None and (only is None):
        extra: dict[str, Any] = lstate.steps.setdefault("H5", {"step": "H5", "status": "running", "jobs": []})
        if extra.get("status") not in STATUS_DONE:
            if used_s() / 3600.0 >= LADDER_BUDGET_HOURS:
                extra.update(status="not_run", note=f"the ladder's own {LADDER_BUDGET_HOURS:g} h budget (addendum 2, '2. Ladder "
                                                    "M3-M7' > 'Budget') is exhausted: H5 ablations not run")
            else:
                jobs = [j for arm in LAD.H5_ABLATION_ARMS for j in ladder_jobs(arm, state)]
                refit_jobs = ladder_jobs("M3", state, refits=True) if lstate.kept("M3") else []
                jobs += refit_jobs
                extra["jobs"] = [j.key for j in jobs]
                led = run_step_jobs(jobs, corpus, out_root, steps=steps, code=codes["discovery"], state=state, lstate=lstate,
                                    codes=codes, workers=workers, coext_ids=coext_ids, prereg=prereg, paused=paused,
                                    guard_fn=guard_fn, inner_check=inner_check, tune_kwargs=tune_kwargs,
                                    min_expert_rows=min_expert_rows)
                ledger["jobs"].update(led["jobs"])
                extra["ledger"] = _json(led["jobs"])
                if led["stopped"]:
                    extra.update(status="incomplete", note=led["stopped"])
                    ledger["stopped"] = led["stopped"]
                else:
                    rows = []
                    acts = {arm: component_activity(out_root, arm, ladder_jobs(arm, state), corpus) for arm in LAD.H5_ABLATION_ARMS}
                    retained = "M6" if lstate.kept("M6") else lstate.retained_before("M6")
                    if retained in LAD.ALL_ARMS:
                        acts[retained] = component_activity(out_root, retained, ladder_jobs(retained, state), corpus)
                    for spec in h5_contrasts(lstate):
                        r = evaluate_ladder_spec(get_store(), spec, d5)
                        if r is not None:
                            term = "hinge" if spec.candidate == "M6a_toggle" else "smoothness"
                            vac = vacuity_of(acts[spec.candidate], acts.get(retained, {}), terms=(term,))
                            for rec in D.contrast_rows(r):
                                rec["key"] = f"{spec.name}@{spec.design}#H5"
                                rec["vacuous"] = vac["vacuous"]
                                rec["vacuous_reason"] = vac["reason"]
                                rows.append(rec)
                    if rows:
                        write_csv(pd.DataFrame(rows), ladder_root(out_root) / "decisions" / "contrasts_H5.csv")
                    extra.update(status="complete", n_contrast_rows=len(rows), component_activity=_json(acts))
                    if refit_jobs:                              # V-04: consume M3's refit frames
                        m3 = reevaluate_step_after_refits("M3", lstate, get_store(), d5, out_root)
                        lstate.steps["M3"]["refits"] = _json(m3)
                        extra["m3_refits"] = _json(m3)
            lstate.write(lpath)
            record_ladder_wall_clock(out_root, started, time.perf_counter() - t0, done_steps + ["H5"])
    record_ladder_wall_clock(out_root, started, time.perf_counter() - t0, done_steps)
    ledger["ladder"] = lstate.record()
    return ledger


# ============================================================================================= #
# decision files (brief section 29 format, generated from the files)
# ============================================================================================= #

def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "not computed"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "not computed" if not np.isfinite(v) else f"{float(v):.{nd}f}"
    return str(v)


def _step_lines(step: str, rec: Mapping[str, Any] | None) -> list[str]:
    if rec is None:
        return [f"- {step}: not computed (no ladder record)"]
    st = rec.get("status", "not computed")
    line = f"- {step}: status {st}; kept {_fmt(rec.get('kept'))}; predecessor {rec.get('predecessor', 'not computed')}"
    dec = rec.get("decision") or {}
    summ = dec.get("contrast_summary") or {}
    out = [line]
    for d in ("V5", "V1", "V2"):
        s = summ.get(d)
        if s is None:
            out.append(f"  - {d}: not computed")
        else:
            out.append(f"  - {d}: Delta {_fmt(s.get('point'))} log D; ladder-scope R19 {s.get('verdict_ladder')}; full R19 "
                       f"{s.get('r19_full')}; TOST {s.get('tost')}; units {s.get('n_units')}")
    if rec.get("note"):
        out.append(f"  - note: {rec['note']}")
    return out


def d04_text(lstate: LadderState | None, metrics: Mapping[str, Any] | None) -> str:
    L = ["# D04 - mechanism experts and the chemistry priors (ladder M3-M6)", "",
         "Generated by `scripts/g19_run_ladder.py` from `evaluation/ladder/decisions/ladder.json` and "
         "`evaluation/ladder/decisions/contrasts_*.csv`. Selection half, seed 104729: every number is optimistically "
         "biased (pre-registration section 6) and none is a confirmed claim.", "", "## question", "",
         "Do mechanism routing (M3), the publication-group hierarchy (M4), the pairwise selectivity loss (M5) and the "
         "physics penalties (M6) improve strict-fold transfer over the retained factorised model (section 6 ladder; brief "
         "sections 5, 6, 8, 9, 14, 32)?", "", "## evidence", ""]
    if lstate is None:
        L += ["- not computed: the ladder has not run (no ladder.json)"]
    else:
        L += [f"- stop rule (section 7 item 4): stop = {_fmt(lstate.stop_rule)}",
              f"- discovery ladder: M1 kept {_fmt(lstate.discovery_ladder.get('M1'))}, M2 kept {_fmt(lstate.discovery_ladder.get('M2'))}"]
        for s in ("M3", "M4", "M5", "M6"):
            L += _step_lines(s, lstate.steps.get(s))
        h5 = lstate.steps.get("H5")
        L += [f"- H5 component ablations (M6a / M6b toggled): {h5.get('status') if h5 else 'not computed'}"
              + (f"; {h5.get('n_contrast_rows')} contrast rows in contrasts_H5.csv" if h5 and h5.get("n_contrast_rows") is not None else "")]
        if lstate.demoted:
            L += [f"- demoted (the ladder's own {LADDER_BUDGET_HOURS:g} h wall-clock budget of POST-HOC addendum 2, '2. Ladder "
                  f"M3-M7' > 'Budget'; demotion order {list(DEMOTION_ORDER)} unchanged): {lstate.demoted}"]
    L += ["", "## metrics", "",
          "Delta = macro MAE(predecessor) - macro MAE(step), log D, per design; R19 items 1-3, 5 and the scoring-filter "
          "sensitivities (ladder scope) on V5-primary; TOST epsilon 0.05 on V1 / V2 (section 8). R19 item 4 is "
          "NOT_EVALUATED in discovery (addendum 1 item 3); refit sensitivities of a learned arm are the reduced set "
          "(addendum 1 item 4).", "", "## null / supported / ambiguous", ""]
    if lstate is None:
        L += ["- not computed"]
    else:
        for s in ("M3", "M4", "M5", "M6"):
            k = lstate.kept(s)
            st = lstate.status(s)
            verdict = ("supported (kept on the selection half)" if k is True else "null (removed: no strict-fold benefit)"
                       if k is False else f"ambiguous / not run ({st})")
            L.append(f"- {s}: {verdict}")
    L += ["", "## decision", ""]
    if lstate is None:
        L += ["- not computed"]
    else:
        L += [f"- retained configuration after M6: {lstate.retained_before('M7')} (section 6: a component without a "
              "strict-fold benefit is removed and the next step builds on the predecessor)"]
    L += ["", "## next action", "",
          "- M7 on the retained configuration (D05); confirmation of any frozen claim on the withheld seeds and the "
          "confirmation half only (section 15) -- nothing here is confirmed."]
    return "\n".join(L) + "\n"


def d05_text(lstate: LadderState | None, metrics: Mapping[str, Any] | None) -> str:
    L = ["# D05 - uncertainty (M7: 5-member heteroscedastic ensemble, normalised split conformal)", "",
         "Generated by `scripts/g19_run_ladder.py` from `evaluation/ladder/M7/metrics.json`. Selection half, seed "
         "104729, optimistically biased; section 12 metrics; S1(d) bands of section 9; F3 of section 10.", "",
         "## question", "",
         "Does the M7 ensemble give calibrated intervals (50 / 80 / 95 % coverage in the S1(d) bands) with non-inferior "
         "MAE, and does its predictive SD track the error (\"knows when it does not know\", section 12)?", "",
         "## evidence", ""]
    if metrics is None:
        L += ["- not computed: `evaluation/ladder/M7/metrics.json` absent (M7 not run or not judged)"]
        if lstate is not None and lstate.steps.get("M7"):
            L += [f"- M7 status: {lstate.steps['M7'].get('status')}; note: {lstate.steps['M7'].get('note', '')}"]
    else:
        L += [f"- predecessor (retained configuration): {metrics.get('predecessor')}"]
        for d, b in (metrics.get("designs") or {}).items():
            if not isinstance(b, Mapping) or "interval_metrics" not in b:
                L.append(f"- {d}: {b.get('status', 'not computed') if isinstance(b, Mapping) else 'not computed'}")
                continue
            im = b["interval_metrics"]
            if isinstance(im, Mapping):
                L.append(f"- {d}: coverage 50 / 80 / 95 = {_fmt(im.get('coverage_50'), 3)} / {_fmt(im.get('coverage_80'), 3)} / "
                         f"{_fmt(im.get('coverage_95'), 3)}; width 80 {_fmt(im.get('width_80'), 3)}; CRPS {_fmt(im.get('crps'), 3)}; "
                         f"rows {b.get('n_rows')}")
            else:
                L.append(f"- {d}: intervals {im}")
            sp = b.get("spearman_abs_error_sd")
            if isinstance(sp, Mapping):
                L.append(f"  - Spearman(|error|, SD) {_fmt(sp.get('point'), 3)} [{_fmt(sp.get('low_95'), 3)}, {_fmt(sp.get('high_95'), 3)}] "
                         f"({sp.get('cluster')} cluster bootstrap, {sp.get('n_clusters')} clusters)")
            else:
                L.append(f"  - Spearman(|error|, SD): {sp}")
            t = b.get("mae_tost_vs_predecessor")
            if isinstance(t, Mapping):
                L.append(f"  - MAE vs predecessor: Delta {_fmt(t.get('point'))}, TOST {t.get('verdict')} (non-inferior {_fmt(t.get('non_inferior'))}, "
                         f"90 % interval [{_fmt(t.get('low_90'))}, {_fmt(t.get('high_90'))}])")
            else:
                L.append(f"  - MAE vs predecessor: {t if t is not None else 'not computed'}")
            if d == "V5":
                cov = b.get("coverage_by_domain_status")
                if isinstance(cov, Mapping):
                    for cat, vals in cov.items():
                        L.append(f"  - {cat}: coverage 80 = {_fmt(vals.get('0.8'), 3)}")
                else:
                    L.append(f"  - coverage by domain status: {cov}")
                s1d = b.get("s1d_check")
                L.append(f"  - S1(d) bands: {_fmt(s1d.get('pass')) if isinstance(s1d, Mapping) else 'not computed'}")
                f3 = b.get("f3_check")
                L.append(f"  - F3 failure: {_fmt(f3.get('failure')) if isinstance(f3, Mapping) else 'not computed'}")
                kn = b.get("knows_when_it_does_not_know")
                L.append(f"  - knows when it does not know: {_fmt(kn.get('established')) if isinstance(kn, Mapping) else 'not computed'}")
    L += ["", "## metrics", "",
          "coverage_50 / 80 / 95 and width (unit macro, section 4 averaging), Gaussian CRPS (closed form), "
          "Spearman(|error|, SD) with the primary-cluster percentile interval, coverage by domain-status category "
          "(section 13; discovery `_support` files), S1(d) bands, F3, TOST on MAE (epsilon 0.05).", "",
          "## null / supported / ambiguous", ""]
    v = (metrics or {}).get("verdict") or {}
    kept = v.get("kept")
    L += [f"- calibration (S1(d) on V5-primary): {_fmt(v.get('calibration_s1d_pass'))}",
          f"- MAE non-inferior on V5, V1 and V2: {_fmt(v.get('mae_non_inferior_all_designs'))}",
          f"- verdict: {'supported' if kept is True else 'null' if kept is False else 'ambiguous / not computed'}", "",
          "## decision", "",
          f"- M7 {'is' if kept is True else 'is not' if kept is False else 'is not yet'} the preferred uncertainty model "
          "(section 12 'preferred model': worse MAE is tolerated only when non-inferior); selection half only.", "",
          "## next action", "",
          "- The section 12 designs at confirmation (V5, V1, V2 on the withheld seeds; V6 once) decide; the CRPS, "
          "Spearman and 'knows when it does not know' readings need the POST-HOC addendum on the M7 member seeds "
          "(models.ladder.REGISTRATION_CHOICES['member_seed'])."]
    return "\n".join(L) + "\n"


def write_decision_files(out_root: Path) -> list[Path]:
    lstate = LadderState.read(ladder_state_path(out_root))
    metrics = D.read_record(ladder_root(out_root) / "M7" / "metrics.json")
    dec_dir = Path(out_root) / "decisions"
    outs = [write_text(dec_dir / "D04_mechanism_experts.md", d04_text(lstate, metrics)),
            write_text(dec_dir / "D05_uncertainty.md", d05_text(lstate, metrics))]
    if lstate is not None:
        rows = [{"step": s, **{k: v for k, v in r.items() if k in ("status", "kept", "predecessor", "note")}}
                for s, r in lstate.steps.items()]
        outs.append(write_csv(pd.DataFrame(rows), Path(out_root) / "tables" / "ladder_decisions.csv"))
    return outs


# ============================================================================================= #
# main
# ============================================================================================= #

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, choices=(1, 2), default=2)
    ap.add_argument("--only", default=None, help="comma list of ladder steps to run (M3,M4,...) -- resumable")
    ap.add_argument("--check-only", action="store_true", help="run the gate (seal, discovery completeness, stop rule) and exit")
    ap.add_argument("--max-hours", type=float, default=None, help="operator pause (resumable; not the section 7 item 5 budget)")
    ap.add_argument("--steps", default="point,intervals", help="point,intervals (default) or point")
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--no-h5", action="store_true", help="skip the H5 component ablations after M7")
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--no-manifest", action="store_true")
    ns = ap.parse_args(argv)
    ns.steps = [s for s in ns.steps.split(",") if s]
    if not set(ns.steps) <= set(D.STEPS) or D.STEPS[0] not in ns.steps:
        raise SystemExit(f"--steps must include 'point' and be among {D.STEPS}")
    return ns


def main(argv=None, *, check: Callable[[], int] | None = None, digests: Callable[[], Mapping[str, Any]] | None = None) -> int:
    ns = parse_args(argv)
    # addendum 2 item 5: the registry's expectations for stage 'ladder' when it exists, else the constants
    prereg = REG.refuse_unless_sealed(REGISTRY_STAGE, check, digests, expect_addenda=ns.expect_addenda)
    out_root = Path(ns.out_root)
    codes = ladder_code_digest()
    # gate items 2(a)-(b) and 3 are file reads: refuse before the corpus-sized allocation (task X finding VL2-04)
    cheap = discovery_complete_cheap(out_root)
    if not cheap["complete"]:
        raise SystemExit("refused: the discovery run is not complete -- " + "; ".join(cheap["reasons"]) +
                         " (nothing heavy was loaded; let scripts/g19_run_discovery.py finish, then rerun)")
    decision_files_ready(out_root)
    state = D.PlanState.read(RD.plan_state_path(out_root))
    coext: list[str] = []

    def load() -> Any:
        nonlocal coext
        log("coextractant ids and corpus")
        coext = RD.coextractant_ids()
        return RD.load_corpus(coext, with_cv=ns.workers == 1)

    def load_and_keep() -> Any:
        _LOADED["corpus"] = load()
        return _LOADED["corpus"]

    gate = refuse_unless_ready(out_root, load_and_keep, state, codes["discovery"], prereg=prereg)
    corpus = _LOADED["corpus"]
    if ns.check_only:
        print(json.dumps(_json(gate), indent=2))
        return 0
    ctx_run = Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=D.PRIMARY_SEED,
                  extra={"code_sha256": codes, "prereg_gate": prereg, "gate": _json(gate), "readings": LAD.REGISTRATION_CHOICES,
                         "runner_readings": READINGS, "ladder_budget_hours": LADDER_BUDGET_HOURS,
                         "model_seed_rule": "42 + fold * 1009 + 9,999,991; M7 member k adds k"}) if not ns.no_manifest else None
    with (ctx_run or RD._Null()) as run:
        ledger = run_ladder(corpus, out_root, state, codes, gate, steps=ns.steps, workers=ns.workers, max_hours=ns.max_hours,
                            only=ns.only, coext_ids=coext, prereg=prereg, with_h5=not ns.no_h5)
        outs = write_decision_files(out_root)
        if run is not None:
            run.outputs(ladder_state_path(out_root), ladder_wall_clock_path(out_root), *outs,
                        *sorted((ladder_root(out_root) / "decisions").glob("contrasts_*.csv")),
                        *[p for p in [ladder_root(out_root) / "M7" / "metrics.json"] if p.exists()])
            run.extra.update({"ledger": _json(ledger),
                              "budget": ladder_budget_status(budget_used_seconds(out_root), RD.wall_clock_total(out_root)),
                              "confirmation_half_read": False, "v6_target_rows_scored": 0})
    log(f"ladder: {json.dumps(_json(ledger.get('steps')))}; stopped: {ledger.get('stopped')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
