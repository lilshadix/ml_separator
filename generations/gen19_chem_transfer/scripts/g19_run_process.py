"""``scripts/g19_run_process.py`` -- Phase H, the Gen19 -> gen18 process integration for the TODGA Pr/Nd nitrate case
(pre-registration section 14 in full, section 13 domain status, section 9 S2(d), section 10 F5; brief sections 1.6,
17, 18, 19, 25 items 14-15, 27, 29).

Gate (refuses to start unless ALL hold; ``--check-only`` prints the verdict)
--------------------------------------------------------------------------
1. ``scripts/g19_seal_prereg.py --check`` exits 0 and the sealed digest / addendum text are the registered ones
   (``g19_run_discovery.refuse_unless_sealed``).
2. **Discovery is COMPLETE** (:func:`discovery_complete`), defined as the conjunction of the two sibling definitions:
   (a) the discovery runner's progress ledger (``manifests/run_info/g19_run_discovery_progress.json``) lists the
   final-stage marker ``discovery.NOT_IMPLEMENTED`` ("M3+ not implemented") and no job carries errors;
   (b) ``evaluation/discovery/records_index.csv`` exists (written at the end of ``g19_run_discovery.main``);
   (c) ``evaluation/discovery/decisions/wall_clock.json`` records an invocation whose ``stages_done`` reached
   ``discovery.STAGES['not_implemented']``;
   (d) ``gen19ct.evaluation.h3.discovery_complete`` (every ``fit`` job of the current plan has a COMPLETE verified record
   set under the digest the current code, fold files and plan state produce) -- run only when (a)-(c) hold.
3. The scorer has decided the stop rule: ``evaluation/discovery/decisions/stop_rule.json`` -> ``stop`` in {true, false}
   (pending is refused).  ``stop = true`` means "H7 (process) not run: S1 cannot pass" (section 7 item 4): refused
   unless ``--exploratory``.
4. **S1 passed for the deployed predictor** (section 14 "Phase H runs only if S1 passes"): the confirmation run's
   decision file ``evaluation/confirmation/decisions/confirmation.json`` (schema ``gen19.confirmation.v1``; the JSON
   beside section 15's ``decisions/CONFIRMATION.md``) with keys ``S1.passed`` (bool), ``S1.deployed_predictor`` (arm
   name), ``S2.passed`` (bool or null), ``V6.run`` (bool), ``seeds.verified`` (bool).  ``S1.passed = false`` -> refused
   always (no Gen19 process evaluation; gen18's chain keeps its B1 lookup default).  File missing or ``S1.passed``
   null -> refused unless ``--exploratory``.
5. Labelling (section 14): unless ``S1.passed``, ``S2.passed`` and ``V6.run`` are all true, EVERY output is labelled
   ``transfer-unsupported`` and none can be a headline recommendation; ``--exploratory`` runs always carry that label.
   No confirmation runner exists yet: this file's expected schema is the contract it must write (see the final report).

Inputs (new directory ``evaluation/process/inputs/``)
-----------------------------------------------------
* ``todga_prnd_predictions.csv`` (``--predictions``): the deployed predictor's Gen19 prediction records for Pr and Nd on
  a (log acid, log ligand) grid (``gen18_adapter.PREDICTION_COLUMNS``).  When absent the runner writes the request
  ``todga_prnd_prediction_request.csv`` (the exact conditions to predict, from the gen18 design space) and refuses.  In
  registered mode the table's ``arm`` must equal ``S1.deployed_predictor``.
* ``prnd_residual_correlation.json`` (``--rho-file``): ``{"rho": r, "n_pairs": n, "source": ...}`` -- the section 14
  Pr/Nd residual correlation "estimated from comparable-pair residuals in inner folds, no V6_TARGET_ROWS row".
  Required in registered mode; in ``--exploratory`` mode a missing file gives independent draws (``rho = 0``, flagged:
  section 14 says independent draws inflate logSF uncertainty).

Case (section 14 "Case")
------------------------
System ``sys_5cb78e5000d40860`` (TODGA / nitrate / aliphatic, gen18 ``systems/``; named by ``cases/todga_prnd_feed.json``
-> ``system_id``); feed ``gen18 cases/todga_prnd_feed.json``; spec grid ``gen18 cases/prnd_spec.json`` (purity in
{0.95, 0.97, 0.99} x recovery in {0.80, 0.85, 0.90}, target Nd, impurity Pr); every value of both files is an assumed
placeholder and stays flagged.  Design space: gen18 ``config/design_spaces.json`` (defaults + ``diglycolamide`` family,
the entry's ``TODGA|20-30C`` applicability intervals for the ``from_domain`` axes) with the case's fixed variables
(:data:`CASE_FIXED`: no saponification -- family bound [0, 0] --, no complexant in the entry, no salting anion, no
dilution, no bleed) and the acid / ligand axes intersected with the prediction table's supported box.  gen18's cascade,
metrics and optimiser are imported read-only (``paths.add_gen18_to_path()``); the D source is
``gen19ct.process.gen18_adapter.Gen19DModel``.

Monte Carlo, objective, checks: ``gen19ct.process.monte_carlo`` (64 seeded joint draws, paired LHS design, per
operating point median / 5-50-95 % purity and recovery, P(purity >= t), P(recovery >= t), P(both), P(phase / loading
constraints), reagent use, stages) and ``gen19ct.process.robust_optimize`` (support rank -> P(both and feasible) ->
P(feasible) -> consumption -> stages -> throughput; S2(d) 20 bootstrap re-rankings, seed 19, >= 0.80; F5(i) / (ii); the
gate-lifted re-ranking).  The F5(ii) "allowed" variant is a second Monte Carlo run with ``allow_unsupported=True`` on
the untrimmed table, run only when the table carries UNSUPPORTED cells (``--skip-f5-allowed`` skips it).

Outputs (new directories only): ``evaluation/process/`` (``process_table.csv`` -- byte-identical for the same seed --,
``candidates.csv``, ``draws.csv``, ``usage.csv``, ``operating_points.csv``, ``rankings.csv``, ``winners.csv``,
``stability.json``, ``f5.json``, ``summary.json``, ``inputs_used.json``, ``f5_allowed/``), ``tables/process_*.csv``,
``figures/F14_process_pareto_uncertainty.png`` and ``figures/F15_probability_of_specification_map.png`` (brief section
25 items 14 and 15), ``decisions/D06_process_integration.md`` (brief section 29 format, generated from the files;
``--decision-only`` regenerates it, "not computed" where a file is absent), ``manifests/g19_run_process.json`` (git head,
prereg + addendum digests, the code digests of this runner's files AND of the gen18 files it imports, seeds, runtime).

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_process.py --check-only
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_process.py [--draws 64] [--n-lhs 1000]
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_process.py --exploratory
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_process.py --decision-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.manifest import Run, git_head, write_csv, write_json, write_text  # noqa: E402
from gen19ct.process import gen18_adapter as GA  # noqa: E402
from gen19ct.process import monte_carlo as MC  # noqa: E402
from gen19ct.process import robust_optimize as RO  # noqa: E402

paths.add_gen18_to_path()

from gen18proc import optimize as OPT  # noqa: E402  (read-only import)
from gen18proc.domain import coerce_domain  # noqa: E402
from gen18proc.metrics import Prices  # noqa: E402
from gen18proc.systems import load_system  # noqa: E402
from gen18proc.types import AqStream, Sourced  # noqa: E402

NAME = "g19_run_process"
#: the registry stage of this runner (addendum 2 item 5: the seal gate and the discovery code digest prefer
#: manifests/digest_registry.json when it exists and fall back to the constants of evaluation.discovery otherwise)
REGISTRY_STAGE = "process"
SCRIPTS = Path(__file__).resolve().parent
SCHEMA = "gen19.process.v1"
NOT_COMPUTED = "not computed"
TRANSFER_UNSUPPORTED = "transfer-unsupported"

#: this runner's own prediction-affecting files (digested into the manifest), including the deployment prediction step
#: that fills the request (``scripts/g19_predict_process_inputs.py``: the D source of every process number)
CODE_FILES: tuple[Path, ...] = (paths.G19_ROOT / "gen19ct" / "process" / "gen18_adapter.py",
                                paths.G19_ROOT / "gen19ct" / "process" / "monte_carlo.py",
                                paths.G19_ROOT / "gen19ct" / "process" / "robust_optimize.py",
                                Path(__file__).resolve(),
                                Path(__file__).resolve().parent / "g19_predict_process_inputs.py")


def members_identical(table: GA.PredictionTable, tol: float = 1e-9) -> bool:
    """True when every ``member_logD_k`` of the table equals ``mean_logD`` (a single-model deployed arm wrote the mean
    into the member columns: the registered draw then degenerates to mean + conformal residual and must be flagged)."""
    if not table.has_members or table.members is None:
        return False
    return bool(np.all(np.abs(table.members - table.mean[None, ...]) <= tol))
#: the gen18 files imported read-only (digested to prove which cascade code ran; never edited)
GEN18_FILES: tuple[Path, ...] = tuple(paths.GEN18_ROOT / "gen18proc" / f for f in (
    "cascade.py", "dmodel.py", "domain.py", "equilibrium.py", "metrics.py", "optimize.py", "systems.py", "types.py"))

CASE_FEED = paths.GEN18_ROOT / "cases" / "todga_prnd_feed.json"
CASE_SPEC = paths.GEN18_ROOT / "cases" / "prnd_spec.json"
DESIGN_SPACES = paths.GEN18_ROOT / "config" / "design_spaces.json"
PRICES_FILE = paths.GEN18_ROOT / "config" / "prices.json"
CASE_BAND = "20-30C"
#: variables the TODGA nitrate case fixes (module docstring); everything else follows config/design_spaces.json
CASE_FIXED: dict[str, float] = {"saponification_degree": 0.0, "scrub_complexant_M": 0.0, "strip_anion_M": 0.0,
                                "feed_dilution": 0.0, "f_bleed": 0.0}
PREDICTION_GRID_SHAPE = (25, 9)          # (n_acid, n_ligand) of the prediction request

#: the confirmation decision file this runner reads (contract for the confirmation runner)
CONFIRMATION_SCHEMA = "gen19.confirmation.v1"
CONFIRMATION_KEYS: dict[str, str] = {
    "S1.passed": "bool: section 9 S1 passed on the confirmation half with 5 of 5 withheld seeds for the deployed predictor",
    "S1.deployed_predictor": "str: the arm deployed (M2, M1, or the best-passing baseline under F6)",
    "S2.passed": "bool or null: section 9 S2 (V6, single confirmation run); null when V6 has not run",
    "V6.run": "bool: V6 has been run once (section 3.4)",
    "seeds.verified": "bool: the --verify-seeds verdict of section 15",
}

REGISTRATION_READINGS: dict[str, str] = {
    "draw_distribution": ("section 14: one draw = one M7 ensemble member (member_logD_k of the prediction record, one "
                          "member per draw shared by Pr and Nd) plus a residual at the calibrated conformal scale "
                          "(Gaussian with scale std_logD * conformal_q95 / 1.96, one per metal and draw, common over the "
                          "grid, Pr / Nd correlated with rho through a Gaussian copula) -- monte_carlo.REGISTRATION_READINGS; "
                          "a table without member columns is refused in registered mode and drawn as a truncated Gaussian "
                          "on the record only with --exploratory (flagged) (task X finding V-07)"),
    **{f"monte_carlo.{k}": v for k, v in MC.REGISTRATION_READINGS.items()},
    "design_count": ("section 14 registers 64 draws but no LHS design count; the default 1000 is gen18's "
                     "g18_case_prnd.py default (--n-lhs) -- a reading"),
    "objective_key_1": RO.OBJECTIVE_READING,
    "f5_ii_per_cell": RO.F5_READING,
    "outside_table": ("a stage evaluating the table outside its grid has no prediction: recorded as OUTSIDE_TABLE and "
                      "treated as UNSUPPORTED (ineligible) -- a reading"),
    "constraint_flags": ("P(phase and loading constraints satisfied) uses gen18's THIRD_PHASE_RISK, LOADING_CAP_HIT and "
                         "HIGH_LOADING; PHASE_BEHAVIOUR_UNKNOWN is reported separately as unknown -- a reading"),
    "design_space_intersection": ("the scrub / strip acid and ligand axes of the design space are intersected with the "
                                  "prediction table's supported box (no recipe evaluates a condition that has no "
                                  "supported prediction) -- a reading"),
    "case_fixed_variables": ("saponification 0 (family bound [0, 0]), no complexant (none in the entry), no salting "
                             "anion, no dilution, no bleed: the TODGA-only reading of gen18's 13.5 exploration"),
}


def _load_script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================================= #
# paths
# ============================================================================================= #

def process_root(out_root: Path) -> Path:
    return Path(out_root) / "evaluation" / "process"


def inputs_dir(out_root: Path) -> Path:
    return process_root(out_root) / "inputs"


def predictions_path(out_root: Path) -> Path:
    return inputs_dir(out_root) / "todga_prnd_predictions.csv"


def prediction_request_path(out_root: Path) -> Path:
    return inputs_dir(out_root) / "todga_prnd_prediction_request.csv"


def rho_path(out_root: Path) -> Path:
    return inputs_dir(out_root) / "prnd_residual_correlation.json"


def confirmation_path(out_root: Path) -> Path:
    return Path(out_root) / "evaluation" / "confirmation" / "decisions" / "confirmation.json"


def confirmation_md_path(out_root: Path) -> Path:
    return Path(out_root) / "decisions" / "CONFIRMATION.md"


def stop_rule_path(out_root: Path) -> Path:
    return D.discovery_root(out_root) / "decisions" / "stop_rule.json"


def progress_path(out_root: Path) -> Path:
    if Path(out_root).resolve() == paths.G19_ROOT.resolve():
        return paths.MANIFESTS_DIR / "run_info" / "g19_run_discovery_progress.json"
    return D.discovery_root(out_root) / "run_info_progress.json"


def d06_path(out_root: Path) -> Path:
    return Path(out_root) / "decisions" / "D06_process_integration.md"


def figures_dir(out_root: Path) -> Path:
    return Path(out_root) / "figures"


def tables_dir(out_root: Path) -> Path:
    return Path(out_root) / "tables"


def code_digests() -> dict[str, Any]:
    own = D.code_digest(CODE_FILES)
    g18 = D.code_digest(GEN18_FILES)
    return {"own": own["combined"], "own_parts": own["parts"], "gen18": g18["combined"], "gen18_parts": g18["parts"]}


# ============================================================================================= #
# gate
# ============================================================================================= #

def discovery_complete_cheap(out_root: Path) -> dict[str, Any]:
    """Gate items 2(a)-(c): the two ledgers and the records index, without reading any fold record."""
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
    wc = D.read_record(D.discovery_root(out_root) / "decisions" / "wall_clock.json") or {}
    stages_done: set[str] = set()
    for inv in wc.get("invocations") or []:
        stages_done |= set(inv.get("stages_done") or [])
    final = D.STAGES["not_implemented"]
    if final not in stages_done:
        reasons.append(f"wall_clock.json records no invocation reaching stage {final!r} (stages done: {sorted(stages_done)})")
    return {"complete": not reasons, "reasons": reasons, "stages_done": sorted(stages_done)}


def default_heavy_discovery_check(out_root: Path) -> Callable[[], dict[str, Any]]:
    """Gate item 2(d): ``h3.discovery_complete`` on the current plan state, code digest and co-extractant ids."""
    def run() -> dict[str, Any]:
        from gen19ct.evaluation import h3 as H3

        RD = _load_script("g19_run_discovery")
        state = D.PlanState.read(RD.plan_state_path(out_root))
        return H3.discovery_complete(out_root, code=REG.discovery_code_digest(), state=state,
                                     excluded_ids=RD.coextractant_ids())
    return run


def discovery_complete(out_root: Path, *, heavy: Callable[[], dict[str, Any]] | None = None) -> dict[str, Any]:
    """The module docstring's completeness definition: the cheap checks, then ``heavy`` (item 2(d)) only when they hold."""
    cheap = discovery_complete_cheap(out_root)
    out = {"complete": cheap["complete"], "reasons": list(cheap["reasons"]), "cheap": cheap, "record_check": None,
           "rule": ("progress ledger reached the final stage without job errors; records_index.csv written; wall_clock.json "
                    "reached the final stage; every fit job of the current plan has a COMPLETE verified record set "
                    "(h3.discovery_complete)")}
    if cheap["complete"] and heavy is not None:
        rec = heavy()
        out["record_check"] = {k: v for k, v in rec.items() if k != "incomplete_record_sets"} | {
            "incomplete_record_sets": (rec.get("incomplete_record_sets") or [])[:5]}
        if not rec.get("complete"):
            out["complete"] = False
            out["reasons"].append(f"{rec.get('n_incomplete', '?')} discovery fit record set(s) are not complete and current")
    return out


def read_confirmation(out_root: Path) -> dict[str, Any] | None:
    return D.read_record(confirmation_path(out_root))


def s1_decision(conf: Mapping[str, Any] | None) -> dict[str, Any]:
    """The keys of :data:`CONFIRMATION_KEYS` read from the confirmation decision file (``None`` where absent)."""
    if conf is None:
        return {"present": False, "s1_passed": None, "deployed_predictor": None, "s2_passed": None, "v6_run": None,
                "seeds_verified": None, "schema": None}
    s1, s2, v6, seeds = (conf.get(k) or {} for k in ("S1", "S2", "V6", "seeds"))
    def b(x: Any) -> bool | None:
        return None if x is None else bool(x)
    return {"present": True, "schema": conf.get("schema"), "s1_passed": b(s1.get("passed")),
            "deployed_predictor": s1.get("deployed_predictor"), "s2_passed": b(s2.get("passed")), "v6_run": b(v6.get("run")),
            "seeds_verified": b(seeds.get("verified"))}


def transfer_label(s1: Mapping[str, Any], *, exploratory: bool) -> dict[str, Any]:
    """Section 14 labelling: ``transfer-unsupported`` unless S1, S2 and V6 all hold (and never a headline then)."""
    registered = bool(s1.get("s1_passed") is True and s1.get("s2_passed") is True and s1.get("v6_run") is True)
    unsupported = exploratory or not registered
    reasons = []
    if exploratory:
        reasons.append("--exploratory run")
    if s1.get("s1_passed") is not True:
        reasons.append("S1 not passed / not decided")
    if s1.get("s2_passed") is not True:
        reasons.append("S2 not passed / not decided")
    if s1.get("v6_run") is not True:
        reasons.append("V6 has not run")
    return {"label": TRANSFER_UNSUPPORTED if unsupported else "registered", "headline_allowed": not unsupported,
            "reasons": reasons}


def refuse_unless_ready(out_root: Path, *, exploratory: bool = False, check: Callable[[], int] | None = None,
                        digests: Callable[[], Mapping[str, Any]] | None = None, expect_addenda: int | None = None,
                        heavy_discovery_check: Callable[[], dict[str, Any]] | None = None,
                        seal: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The gate of the module docstring (items 1-5).  ``seal`` defaults to the registry-preferring seal gate of stage
    ``process`` (``registry.refuse_unless_sealed``, which is ``g19_run_discovery.refuse_unless_sealed`` until
    ``manifests/digest_registry.json`` exists); ``heavy_discovery_check`` to :func:`default_heavy_discovery_check` (tests
    pass stubs)."""
    seal_fn = seal or (lambda c, d, expect_addenda=None: REG.refuse_unless_sealed(REGISTRY_STAGE, c, d, expect_addenda=expect_addenda))
    prereg = dict(seal_fn(check, digests, expect_addenda=expect_addenda))
    heavy = heavy_discovery_check if heavy_discovery_check is not None else default_heavy_discovery_check(out_root)
    comp = discovery_complete(out_root, heavy=heavy)
    if not comp["complete"]:
        raise SystemExit("refused: the discovery run is not complete -- " + "; ".join(comp["reasons"]) +
                         " (Phase H runs only after discovery, the scorer and the confirmation run; section 14)")
    stop = D.read_record(stop_rule_path(out_root))
    if stop is None or stop.get("stop") not in (True, False):
        raise SystemExit(f"refused: {stop_rule_path(out_root)} is missing or its 'stop' is not decided (true / false); "
                         "run scripts/g19_score_discovery.py after discovery (section 7 item 4)")
    if stop["stop"] is True and not exploratory:
        raise SystemExit("refused: the stop rule fired (section 7 item 4: 'H7 (process) not run: S1 cannot pass'); pass "
                         "--exploratory for a labelled transfer-unsupported run")
    conf = read_confirmation(out_root)
    s1 = s1_decision(conf)
    if s1["present"] and s1.get("schema") not in (None, CONFIRMATION_SCHEMA):
        raise SystemExit(f"refused: {confirmation_path(out_root)} has schema {s1['schema']!r}, expected {CONFIRMATION_SCHEMA!r}")
    if s1["s1_passed"] is False:
        raise SystemExit("refused: the confirmation run decided S1 FAILED for the deployed predictor "
                         f"({s1['deployed_predictor']!r}); section 14: no Gen19 process evaluation is run and gen18's "
                         "chain keeps its B1 lookup default (this holds with --exploratory too)")
    if s1["s1_passed"] is None and not exploratory:
        raise SystemExit(f"refused: {confirmation_path(out_root)} is missing or S1.passed is undecided (keys: "
                         f"{sorted(CONFIRMATION_KEYS)}); Phase H runs only if S1 passes (section 14). Pass --exploratory "
                         "for a labelled transfer-unsupported run")
    label = transfer_label(s1, exploratory=exploratory)
    return {"prereg_gate": prereg, "discovery_complete": comp, "stop_rule": {"stop": bool(stop["stop"]),
                                                                            "path": str(stop_rule_path(out_root))},
            "confirmation": {**s1, "path": str(confirmation_path(out_root)),
                             "md_present": confirmation_md_path(out_root).exists(), "keys": CONFIRMATION_KEYS},
            "exploratory": bool(exploratory), "label": label}


# ============================================================================================= #
# inputs and the case
# ============================================================================================= #

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sourced_value(obj: Any) -> float:
    return float(Sourced.from_json(obj).value)


def feed_stream(feed: Mapping[str, Any]) -> AqStream:
    """The feed ``AqStream`` of a gen18 case file (``scripts/g18_case_prnd.py::feed_stream``; metals mM -> mol/L)."""
    return AqStream(flow_L_h=sourced_value(feed["flow_L_h"]),
                    metals={m: sourced_value(v) * 1e-3 for m, v in feed["metals_mM"].items()},
                    h=sourced_value(feed["h_M"]), anion=sourced_value(feed["anion_M"]),
                    complexant_total=sourced_value(feed["complexant_total_M"]), sodium=sourced_value(feed["sodium_M"]))


def read_rho(path: Path, *, exploratory: bool) -> dict[str, Any]:
    body = D.read_record(path)
    if body is None or "rho" not in body:
        if not exploratory:
            raise SystemExit(f"refused: {path} is missing (section 14: the Pr / Nd residual correlation estimated from "
                             "comparable-pair residuals in inner folds, no V6_TARGET_ROWS row; schema {'rho': r, "
                             "'n_pairs': n, 'source': ...}). Pass --exploratory to draw independently (flagged)")
        return {"rho": 0.0, "present": False, "path": str(path),
                "flag": "independent Pr / Nd draws (rho = 0): section 14 says these inflate logSF uncertainty"}
    rho = float(body["rho"])
    if not (-1.0 < rho < 1.0):
        raise SystemExit(f"refused: rho = {rho} in {path} is not a correlation")
    # the file's own flag and source travel with the value: the producer's Ln-Ln stand-in (0 Pr/Nd pairs among the
    # calibration rows) was read as a plain 'Pr/Nd residual correlation' with flag None (task X finding V-PROC-02)
    return {"rho": rho, "present": True, "path": str(path), "n_pairs": body.get("n_pairs"), "source": body.get("source"),
            "n_pairs_prnd": body.get("n_pairs_prnd"), "rho_prnd": body.get("rho_prnd"), "flag": body.get("flag")}


def case_design_space(entry: Any, ligand: str, *, max_stages: int | None = None) -> OPT.DesignSpace:
    """gen18 ``config/design_spaces.json`` for the entry's family and ``TODGA|20-30C`` domain with :data:`CASE_FIXED`."""
    dom = coerce_domain(entry.applicability.get(f"{ligand}|{CASE_BAND}"))
    widen = {}
    if max_stages is not None:
        widen = {"n_ext": (1, max_stages), "n_scr": (1, max_stages), "n_str": (1, max_stages),
                 "feed_stage_offset": (0, max_stages - 1), "scrub_return_offset": (0, max_stages - 1)}
    return OPT.DesignSpace.from_config(family=entry.family, domain=dom, ligand=ligand, config_path=DESIGN_SPACES,
                                       widen=widen, fixed=CASE_FIXED)


def intersect_space_with_table(space: OPT.DesignSpace, table: GA.PredictionTable) -> tuple[OPT.DesignSpace, dict[str, Any]]:
    """Restrict the scrub / strip acid and ligand axes to the table's box (REGISTRATION_READINGS['design_space_intersection'])."""
    box = table.box()
    acid = (10.0 ** box["log_acid"][0], 10.0 ** box["log_acid"][1])
    lig = (10.0 ** box["log_ligand"][0], 10.0 ** box["log_ligand"][1])
    new: dict[str, tuple[float, float]] = {}
    changes = {}
    for k, win in (("scrub_acid_M", acid), ("strip_acid_M", acid), ("ligand_total_M", lig)):
        if k in space.bounds:
            lo, hi = space.bounds[k]
            lo2, hi2 = max(lo, win[0]), min(hi, win[1])
            if lo2 > hi2:
                raise SystemExit(f"refused: the design-space window of {k} [{lo}, {hi}] does not intersect the prediction "
                                 f"table's box [{win[0]:.4g}, {win[1]:.4g}]")
            new[k] = (lo2, hi2)
            changes[k] = {"before": [lo, hi], "after": [lo2, hi2]}
    return space.with_bounds(**new), changes


def build_case(table: GA.PredictionTable, *, max_stages: int | None, solver_kwargs: Mapping[str, Any],
               prices: Prices | None) -> tuple[MC.CaseSetup, dict[str, Any]]:
    feed_json, spec_json = load_json(CASE_FEED), load_json(CASE_SPEC)
    sid = str(feed_json["system_id"])
    entry = load_system(paths.GEN18_ROOT / "systems" / f"{sid}.json")
    ligand = GA._extractant_ligand(entry).name
    feed = feed_stream(feed_json)
    target = str(spec_json["target"])
    impurities = tuple(str(m) for m in spec_json["impurities"])
    purity_grid = tuple(float(v) for v in spec_json["purity_min_grid"]["values"])
    recovery_grid = tuple(float(v) for v in spec_json["recovery_min_grid"]["values"])
    space0 = case_design_space(entry, ligand, max_stages=max_stages)
    space, changes = intersect_space_with_table(space0, table)
    lig_lo, lig_hi = space.bounds["ligand_total_M"]
    base = MC.base_spec_for(feed, target, ligand, math.sqrt(lig_lo * lig_hi),
                            scrub_acid_M=min(feed.h, space.bounds["scrub_acid_M"][1]),
                            strip_acid_M=space.bounds["strip_acid_M"][0])
    setup = MC.CaseSetup(entry=entry, feed=feed, feed_anion=str(feed_json["anion"]),
                         temperature_C=sourced_value(feed_json["temperature_C"]), target=target, impurities=impurities,
                         purity_grid=purity_grid, recovery_grid=recovery_grid, space=space, base_spec=base, prices=prices,
                         band=CASE_BAND, solver_kwargs=dict(solver_kwargs))
    info = {"system_id": sid, "system_name": entry.name, "family": entry.family, "ligand": ligand, "feed_file": str(CASE_FEED),
            "spec_file": str(CASE_SPEC), "feed_status": feed_json.get("feed_status"), "spec_status": spec_json.get("spec_status"),
            "feed": {"flow_L_h": feed.flow_L_h, "metals_M": feed.metals, "h_M": feed.h, "anion_M": feed.anion},
            "target": target, "impurities": list(impurities), "purity_grid": list(purity_grid),
            "recovery_grid": list(recovery_grid), "design_space_file": str(DESIGN_SPACES),
            "design_space": space.to_json(), "design_space_before_table_intersection": space0.to_json(),
            "table_intersection": changes, "case_fixed": dict(CASE_FIXED), "band": CASE_BAND,
            "placeholder_note": "every value of the feed and spec files is an assumed placeholder (gen18 open item U1)"}
    return setup, info


def write_prediction_request(out_root: Path, *, max_stages: int | None) -> Path:
    """The conditions the deployment step must predict (module docstring), from the case's design space."""
    feed_json, spec_json = load_json(CASE_FEED), load_json(CASE_SPEC)
    sid = str(feed_json["system_id"])
    entry = load_system(paths.GEN18_ROOT / "systems" / f"{sid}.json")
    ligand = GA._extractant_ligand(entry).name
    space = case_design_space(entry, ligand, max_stages=max_stages)
    feed = feed_stream(feed_json)
    acid = {"feed": (feed.h, feed.h), "scrub": space.bounds["scrub_acid_M"], "strip": space.bounds["strip_acid_M"]}
    grid = GA.prediction_grid(acid_M=acid, ligand_M=space.bounds["ligand_total_M"], n_acid=PREDICTION_GRID_SHAPE[0],
                              n_ligand=PREDICTION_GRID_SHAPE[1])
    metals = [str(spec_json["target"])] + [str(m) for m in spec_json["impurities"]]
    return GA.write_prediction_request(prediction_request_path(out_root), grid, metals=metals, system_id=sid,
                                       acid=str(feed_json["acid"]), anion=str(feed_json["anion"]),
                                       note="fill mean_logD, std_logD, lower_95, upper_95, domain_status, support_score, "
                                            "nearest_support, arm with the deployed predictor's records (section 13); "
                                            "log_ligand is the formal TODGA concentration")


# ============================================================================================= #
# outputs
# ============================================================================================= #

def _json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else (
        sorted(o) if isinstance(o, (set, frozenset)) else str(o))))


def write_run_outputs(root: Path, res: MC.MonteCarloResult, setup: MC.CaseSetup, *, label: Mapping[str, Any],
                      stability_check: bool = True, n_boot: int = RO.N_BOOTSTRAP_RERANKINGS,
                      boot_seed: int = RO.BOOTSTRAP_SEED, ops_allowed: pd.DataFrame | None = None) -> dict[str, Any]:
    """Every table of one Monte Carlo run under ``root``; returns the summary block (paths + verdicts)."""
    root = paths.ensure_dir(root)
    outs: list[Path] = []
    var = list(setup.space.variables)
    pt = res.process_table.copy()
    pt.insert(0, "label", label["label"])
    outs.append(write_csv(pt, root / "process_table.csv", float_format="%.10g"))
    # EVERY table of the run carries the label, not only the two a reader is most likely to open: candidates.csv,
    # draws.csv and usage.csv were the four unlabelled outputs of the exploratory run (task X finding V-PROC-01)
    for name, frame, ff in (("candidates.csv", res.candidates, "%.10g"), ("draws.csv", res.draws, "%.10g"),
                            ("usage.csv", res.usage, "%.6g")):
        t = frame.copy()
        t.insert(0, "label", label["label"])
        outs.append(write_csv(t, root / name, float_format=ff))
    ops = MC.aggregate_operating_points(res.process_table, setup.purity_grid, setup.recovery_grid, res.usage, variables=var)
    ops.insert(0, "label", label["label"])
    outs.append(write_csv(ops, root / "operating_points.csv"))
    rankings, winners, stab = [], [], {}
    for cell in setup.cells:
        ranked = RO.rank_recipes(ops, cell)
        ranked.insert(0, "cell", MC.cell_key(*cell))
        rankings.append(ranked)
        top = RO.winner(ranked)
        RO.assert_no_unsupported_winner(top)                 # F5(i) code check
        key = MC.cell_key(*cell)
        row: dict[str, Any] = {"cell": key, "purity_min": cell[0], "recovery_min": cell[1], "label": label["label"],
                               "headline_allowed": label["headline_allowed"], "n_eligible": int(len(ranked)),
                               "n_ineligible": ranked.attrs.get("n_ineligible")}
        if top is not None:
            row.update({"candidate": int(top["candidate"]), "support_rank": int(top["support_rank"]),
                        "p_both_feasible": float(top["obj_p_both_feasible"]), "p_both": float(top["obj_p_both"]),
                        "p_feasible": float(top["p_feasible"]), "p_constraints": float(top["p_constraints"]),
                        "purity_p50": float(top["purity_p50"]), "recovery_p50": float(top["recovery_p50"]),
                        "purity_p5": float(top["purity_p5"]), "purity_p95": float(top["purity_p95"]),
                        "recovery_p5": float(top["recovery_p5"]), "recovery_p95": float(top["recovery_p95"]),
                        "consumption_index_median": float(top["consumption_index_median"]),
                        "n_stages_total": int(top["n_stages_total"]), "n_ext": int(top["n_ext"]), "n_scr": int(top["n_scr"]),
                        "n_str": int(top["n_str"]), "oa_ext": float(top["oa_ext"]), "throughput_median": float(top["throughput_median"]),
                        "statuses_used": top["statuses_used"]})
            for v in var:
                if v in top.index and v not in row:
                    row[v] = float(top[v])
        if stability_check:
            s = RO.stability(res.process_table, res.usage, cell, setup.purity_grid, setup.recovery_grid, n_boot=n_boot,
                             seed=boot_seed, variables=var)
            stab[key] = s
            row["s2d_fraction_kept"] = s["fraction_kept"]
            row["s2d_passes"] = s["passes"]
        winners.append(row)
    rank_df = pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()
    outs.append(write_csv(rank_df, root / "rankings.csv"))
    win_df = pd.DataFrame(winners)
    outs.append(write_csv(win_df, root / "winners.csv"))
    if stability_check:
        outs.append(write_json(root / "stability.json", {"schema": SCHEMA, "label": label["label"], "cells": stab,
                                                         "threshold": RO.STABILITY_THRESHOLD, "n_boot": n_boot,
                                                         "seed": boot_seed}))
    f5 = RO.f5_audit(ops, setup.cells, ops_allowed=ops_allowed)
    outs.append(write_json(root / "f5.json", {"schema": SCHEMA, "label": label["label"], **_json(f5)}))
    return {"outputs": outs, "ops": ops, "winners": win_df, "stability": stab, "f5": f5, "rankings": rank_df}


# ---- figures (brief section 25 items 14 and 15) ---------------------------------------------- #

_INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "surface": "#fcfcfb", "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
_BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _style(ax) -> None:
    ax.set_facecolor(_INK["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_INK["axis"])
    ax.tick_params(colors=_INK["secondary"], labelsize=8)
    ax.grid(True, color=_INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def figure_pareto(ops: pd.DataFrame, winners: pd.DataFrame, setup: MC.CaseSetup, label: str, path: Path) -> Path:
    """Item 14: median purity vs median recovery per eligible operating point with 5-95 % bars on the non-dominated
    (median) front; the spec grid as hairlines; per-cell winners marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = ops.copy()
    df["support_rank"] = [RO.support_rank(RO._statuses(s)) for s in df["statuses_used"]]
    elig = df.loc[(df["support_rank"] > 0) & df["purity_p50"].notna() & df["recovery_p50"].notna()]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=130)
    fig.patch.set_facecolor(_INK["surface"])
    _style(ax)
    if len(elig):
        pts = elig[["recovery_p50", "purity_p50"]].to_numpy()
        front = np.zeros(len(pts), dtype=bool)
        for i, (r, p) in enumerate(pts):
            dominated = np.any((pts[:, 0] >= r) & (pts[:, 1] >= p) & ((pts[:, 0] > r) | (pts[:, 1] > p)))
            front[i] = not dominated
        ax.scatter(elig["recovery_p50"], elig["purity_p50"], s=10, color=_INK["blue"], alpha=0.35, linewidths=0,
                   label=f"eligible operating point, median over {int(elig['n_draws'].max())} draws")
        fr = elig.loc[front]
        ax.errorbar(fr["recovery_p50"], fr["purity_p50"],
                    xerr=[np.clip(fr["recovery_p50"] - fr["recovery_p5"], 0, None), np.clip(fr["recovery_p95"] - fr["recovery_p50"], 0, None)],
                    yerr=[np.clip(fr["purity_p50"] - fr["purity_p5"], 0, None), np.clip(fr["purity_p95"] - fr["purity_p50"], 0, None)],
                    fmt="o", ms=4, color=_INK["blue"], ecolor=_INK["muted"], elinewidth=0.8, capsize=2,
                    label="median-front point, 5-95 % over draws")
    for p in setup.purity_grid:
        ax.axhline(p, color=_INK["axis"], linewidth=0.6, linestyle=(0, (3, 3)))
    for r in setup.recovery_grid:
        ax.axvline(r, color=_INK["axis"], linewidth=0.6, linestyle=(0, (3, 3)))
    if len(winners) and "candidate" in winners.columns:
        w = winners.dropna(subset=["candidate"])
        ax.scatter(w["recovery_p50"], w["purity_p50"], s=60, facecolors="none", edgecolors=_INK["orange"], linewidths=1.4,
                   label="lexicographic winner of a spec cell (support -> P(both) -> P(feasible) -> consumption -> stages -> throughput)")
        # one label per winning candidate (a candidate often wins several cells): "c12: 3 cells"
        for cand, g in w.groupby("candidate", sort=True):
            r = g.iloc[0]
            cells = f"{len(g)} cell{'s' if len(g) > 1 else ''}" if len(g) > 1 else str(r["cell"])
            ax.annotate(f"c{int(cand)}: {cells}", (r["recovery_p50"], r["purity_p50"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color=_INK["secondary"])
    ax.set_xlabel("recovery of Nd from the feed (median over D draws)", color=_INK["secondary"], fontsize=9)
    ax.set_ylabel("Nd purity, mol basis (median over D draws)", color=_INK["secondary"], fontsize=9)
    ax.set_title("Which TODGA / HNO3 cascades reach the Pr/Nd purity-recovery specification under the predictor's "
                 f"uncertainty? [{label}]", fontsize=9.5, color=_INK["primary"], loc="left", wrap=True)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    paths.ensure_dir(path.parent)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def figure_probability_map(ops: pd.DataFrame, winners: pd.DataFrame, setup: MC.CaseSetup, label: str, path: Path) -> Path:
    """Item 15: (a) the best P(both targets and feasible) of an eligible recipe per spec cell; (b) P(both & feasible)
    against the stage count for the consistency cell (0.97, 0.85), eligible recipes, winner marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_seq", _BLUE_RAMP)
    df = ops.copy()
    df["support_rank"] = [RO.support_rank(RO._statuses(s)) for s in df["statuses_used"]]
    elig = df.loc[df["support_rank"] > 0]
    P, R = list(setup.purity_grid), list(setup.recovery_grid)
    grid = np.full((len(P), len(R)), np.nan)
    for i, p in enumerate(P):
        for j, r in enumerate(R):
            col = f"p_both_feasible_{MC.cell_key(p, r)}"
            if len(elig) and col in elig.columns:
                grid[i, j] = float(elig[col].max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=130, gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor(_INK["surface"])
    im = ax1.imshow(np.nan_to_num(grid, nan=0.0), cmap=cmap, vmin=0.0, vmax=1.0, origin="lower", aspect="auto")
    ax1.set_xticks(range(len(R)), [f"{r:.2f}" for r in R])
    ax1.set_yticks(range(len(P)), [f"{p:.2f}" for p in P])
    ax1.set_xlabel("recovery target R_min", color=_INK["secondary"], fontsize=9)
    ax1.set_ylabel("purity target P_min", color=_INK["secondary"], fontsize=9)
    for i in range(len(P)):
        for j in range(len(R)):
            v = grid[i, j]
            txt = NOT_COMPUTED if not np.isfinite(v) else f"{v:.2f}"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="#ffffff" if (np.isfinite(v) and v > 0.55) else _INK["primary"])
    ax1.set_title("(a) best P(both targets & feasible) of an eligible recipe", fontsize=9, color=_INK["primary"], loc="left")
    ax1.tick_params(colors=_INK["secondary"], labelsize=8)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7.5, colors=_INK["secondary"])
    cb.outline.set_edgecolor(_INK["axis"])
    _style(ax2)
    cell = (0.97, 0.85) if (0.97 in P and 0.85 in R) else (P[0], R[0])
    col = f"p_both_feasible_{MC.cell_key(*cell)}"
    if len(elig) and col in elig.columns:
        ax2.scatter(elig["n_stages_total"], elig[col], s=12, color=_INK["blue"], alpha=0.45, linewidths=0,
                    label="eligible operating point")
        ax2.axhline(RO.F5_P_BOTH_THRESHOLD, color=_INK["axis"], linewidth=0.8, linestyle=(0, (3, 3)))
        ax2.text(ax2.get_xlim()[0], RO.F5_P_BOTH_THRESHOLD, " F5(ii) threshold 0.5", fontsize=7.5, color=_INK["muted"], va="bottom")
        if len(winners) and "candidate" in winners.columns:
            w = winners.loc[winners["cell"] == MC.cell_key(*cell)].dropna(subset=["candidate"])
            if len(w):
                ax2.scatter(w["n_stages_total"], w["p_both_feasible"], s=70, facecolors="none", edgecolors=_INK["orange"],
                            linewidths=1.4, label="lexicographic winner")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("total stages (extraction + scrub + strip)", color=_INK["secondary"], fontsize=9)
    ax2.set_ylabel(f"P(purity >= {cell[0]:.2f} & recovery >= {cell[1]:.2f} & feasible)", color=_INK["secondary"], fontsize=9)
    ax2.set_title(f"(b) probability of specification vs stage count, cell ({cell[0]:.2f}, {cell[1]:.2f})", fontsize=9,
                  color=_INK["primary"], loc="left")
    ax2.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(f"How likely is each TODGA / HNO3 recipe to meet the Pr/Nd specification, given the predictor's uncertainty? "
                 f"[{label}]", fontsize=9.5, color=_INK["primary"], x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths.ensure_dir(path.parent)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ---- D06 (brief section 29) ------------------------------------------------------------------- #

def _fmt(v: Any, fmt: str = "{:.3f}") -> str:
    if v is None:
        return NOT_COMPUTED
    if isinstance(v, float) and not math.isfinite(v):
        return "NaN"
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def d06_markdown(out_root: Path) -> str:
    """``decisions/D06_process_integration.md`` generated from the files under ``evaluation/process/`` (``not computed``
    where a file is absent): question, evidence, metrics, verdict, decision, next action."""
    root = process_root(out_root)
    summary = D.read_record(root / "summary.json")
    win = pd.read_csv(root / "winners.csv") if (root / "winners.csv").exists() else None
    stab = D.read_record(root / "stability.json")
    f5 = D.read_record(root / "f5.json")
    head = git_head() or NOT_COMPUTED
    label = (summary or {}).get("label", {}).get("label", NOT_COMPUTED) if summary else NOT_COMPUTED
    lines = ["# D06 -- Does the Gen19 predictor change the TODGA Pr/Nd process recommendation, and does it hold under the "
             "predictor's uncertainty? (H7, sections 14, 9 S2(d), 10 F5)", "",
             f"*Generated by `scripts/g19_run_process.py` at git HEAD `{head}` from the files under `evaluation/process/`. "
             f"Label of every number below: **{label}**. Every value of the feed and specification files is an assumed "
             "placeholder (gen18 open item U1); nothing here is a headline recommendation unless the label is "
             "`registered`.*", "",
             "## Question", "",
             "Fed by the deployed Gen19 predictor's records for Pr and Nd in the TODGA / HNO3 / aliphatic system "
             "(`sys_5cb78e5000d40860`), does gen18's cascade find a recipe that meets the registered purity x recovery "
             "grid with P(both targets and feasible) that survives the D-draw uncertainty (S2(d) >= 0.80), without any "
             "UNSUPPORTED prediction (F5(i)) and without the optimum depending on barred predictions (F5(ii))?", "",
             "## Evidence", ""]
    if summary is None:
        lines += [f"- Phase H has not run: `evaluation/process/summary.json` is absent ({NOT_COMPUTED}).", ""]
    else:
        g = summary.get("gate") or {}
        conf = g.get("confirmation") or {}
        inp = summary.get("inputs") or {}
        case = summary.get("case") or {}
        mc = summary.get("monte_carlo") or {}
        pg = g.get("prereg_gate") or {}
        # the registry gate returns 'prereg_sha256' (registry.refuse_unless_sealed); the older seal-check gate returned
        # 'footer' -- read either, so the line no longer prints 'not computed' beside a passed seal check (task X V-PROC-03)
        seal_digest = pg.get("prereg_sha256") or pg.get("footer") or NOT_COMPUTED
        rho_src = inp.get("rho_source")
        rho_note = "; ".join(x for x in (str(inp.get("rho_flag")) if inp.get("rho_flag") else "",
                                        f"source {rho_src}" if rho_src else "",
                                        f"{inp.get('rho_n_pairs')} pairs" if inp.get("rho_n_pairs") is not None else "",
                                        f"from {inp.get('rho_path')}") if x)
        lines += [f"- Gate: seal digest `{str(seal_digest)[:12]}...`; discovery complete "
                  f"= {(g.get('discovery_complete') or {}).get('complete', NOT_COMPUTED)}; stop rule = "
                  f"{(g.get('stop_rule') or {}).get('stop', NOT_COMPUTED)}; S1 passed = {conf.get('s1_passed', NOT_COMPUTED)} "
                  f"(deployed predictor `{conf.get('deployed_predictor', NOT_COMPUTED)}`); S2 passed = {conf.get('s2_passed', NOT_COMPUTED)}; "
                  f"V6 run = {conf.get('v6_run', NOT_COMPUTED)}; exploratory = {g.get('exploratory', NOT_COMPUTED)}.",
                  f"- Predictions: `{inp.get('predictions_path', NOT_COMPUTED)}` (arm `{inp.get('arm', NOT_COMPUTED)}`, "
                  f"{inp.get('n_cells', NOT_COMPUTED)} grid cells, statuses {inp.get('table_statuses', NOT_COMPUTED)}; "
                  f"registered run on the supported box {inp.get('supported_box', NOT_COMPUTED)}); residual correlation used for "
                  f"the Pr/Nd draws rho = {_fmt(inp.get('rho'))} ({rho_note}).",
                  f"- Case: {case.get('system_name', NOT_COMPUTED)}; feed {case.get('feed', NOT_COMPUTED)}; target "
                  f"{case.get('target', NOT_COMPUTED)} vs {case.get('impurities', NOT_COMPUTED)}; grid purity {case.get('purity_grid')} x "
                  f"recovery {case.get('recovery_grid')}; design space `{case.get('design_space_file', NOT_COMPUTED)}` "
                  f"(intersection with the table: {case.get('table_intersection', NOT_COMPUTED)}).",
                  f"- Monte Carlo: {mc.get('n_draws', NOT_COMPUTED)} draws x {mc.get('n_designs', NOT_COMPUTED)} LHS operating "
                  f"points (seed {mc.get('seed', NOT_COMPUTED)}); {mc.get('n_failed', NOT_COMPUTED)} failed and "
                  f"{mc.get('n_invalid_spec', NOT_COMPUTED)} invalid cascades; gen18 cascade code digest "
                  f"`{str(summary.get('gen18_code_sha256', NOT_COMPUTED))[:12]}...` (read-only import); wall clock "
                  f"{_fmt(mc.get('seconds'), '{:.0f}')} s."
                  + (f" The LHS design count {mc.get('n_designs')} is BELOW gen18's default {MC.N_DESIGNS_DEFAULT} "
                     "(section 14 registers the 64 draws, not a design count; reduced for the wall-clock cap, recorded)."
                     if isinstance(mc.get("n_designs"), (int, float)) and mc.get("n_designs") < MC.N_DESIGNS_DEFAULT else ""),
                  f"- Draw: mode `{inp.get('draw_mode', NOT_COMPUTED)}`, {inp.get('n_members', NOT_COMPUTED)} member column(s)"
                  + (f"; FLAG: {inp['draw_flag'].get('flag')}" if isinstance(inp.get("draw_flag"), dict) else "") + ".", ""]
    lines += ["## Metrics", "",
              "| cell (P_min, R_min) | winner | support rank | P(both & feasible) | P(both) | P(feasible) | purity p50 [p5, p95] | "
              "recovery p50 [p5, p95] | stages (ext+scr+str) | O/A | consumption (mol/kg oxide) | S2(d) kept | F5(ii) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f5_cells = {c["key"]: c for c in ((f5 or {}).get("cells") or [])}
    if win is None or len(win) == 0:
        lines.append(f"| {NOT_COMPUTED} | | | | | | | | | | | | |")
    else:
        for r in win.itertuples(index=False):
            key = str(r.cell)
            has = hasattr(r, "candidate") and pd.notna(getattr(r, "candidate", np.nan))
            f5c = f5_cells.get(key, {})
            lines.append("| ({:.2f}, {:.2f}) | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                float(r.purity_min), float(r.recovery_min),
                f"c{int(r.candidate)}" if has else "none eligible",
                _fmt(int(r.support_rank), "{}") if has else NOT_COMPUTED,
                _fmt(float(r.p_both_feasible)) if has else NOT_COMPUTED, _fmt(float(r.p_both)) if has else NOT_COMPUTED,
                _fmt(float(r.p_feasible)) if has else NOT_COMPUTED,
                (f"{_fmt(float(r.purity_p50))} [{_fmt(float(r.purity_p5))}, {_fmt(float(r.purity_p95))}]" if has else NOT_COMPUTED),
                (f"{_fmt(float(r.recovery_p50))} [{_fmt(float(r.recovery_p5))}, {_fmt(float(r.recovery_p95))}]" if has else NOT_COMPUTED),
                f"{int(r.n_stages_total)} ({int(r.n_ext)}+{int(r.n_scr)}+{int(r.n_str)})" if has else NOT_COMPUTED,
                _fmt(float(r.oa_ext), "{:.2f}") if has else NOT_COMPUTED,
                _fmt(float(r.consumption_index_median), "{:.1f}") if has else NOT_COMPUTED,
                _fmt(getattr(r, "s2d_fraction_kept", None), "{:.2f}") if hasattr(r, "s2d_fraction_kept") and pd.notna(getattr(r, "s2d_fraction_kept")) else NOT_COMPUTED,
                str(f5c.get("F5_ii", NOT_COMPUTED))))
    lines += [""]
    # verdicts
    s2d_pass = None
    if stab and stab.get("cells"):
        vals = [c.get("passes") for c in stab["cells"].values()]
        s2d_pass = None if all(v is None for v in vals) else all(bool(v) for v in vals if v is not None)
    f5_any = None if f5 is None else bool(f5.get("F5"))
    lines += ["## Verdict", ""]
    if summary is None:
        verdict = "null"
        lines += [f"- **null** -- Phase H has not run ({NOT_COMPUTED})."]
    else:
        headline = bool((summary.get("label") or {}).get("headline_allowed"))
        any_reach = bool(win is not None and len(win) and "p_both_feasible" in win.columns and
                         (pd.to_numeric(win["p_both_feasible"], errors="coerce") >= RO.F5_P_BOTH_THRESHOLD).any())
        if f5_any:
            verdict = "null"
            lines += ["- **null** -- F5 holds (section 10): the process optimum depends on unsupported predictions "
                      f"(F5(i) {(f5 or {}).get('F5_i_any_cell')}, F5(ii) {(f5 or {}).get('F5_ii_any_cell')})."]
        elif any_reach and s2d_pass is True and headline:
            verdict = "supported"
            lines += ["- **supported** -- at least one cell has an eligible recipe with P(both & feasible) >= 0.5, every cell's "
                      "top recipe is stable under the D-draw bootstrap (S2(d) >= 0.80), no F5 failure, and the run is "
                      "registered (S1, S2 and V6 hold)."]
        elif any_reach or s2d_pass is True:
            verdict = "ambiguous"
            lines += [f"- **ambiguous** -- reaches the 0.5 threshold in some cell: {any_reach}; S2(d) passes in every cell: "
                      f"{s2d_pass}; headline allowed: {headline} (label {label})."]
        else:
            verdict = "null"
            lines += ["- **null** -- no eligible recipe reaches P(both & feasible) >= 0.5 in any cell, or S2(d) fails."]
    lines += ["", "## Decision", ""]
    if summary is None:
        lines += ["- No process recommendation exists. gen18's chain keeps its B1 nearest-condition lookup as the default D "
                  "source (section 10)."]
    elif verdict == "supported":
        lines += ["- The per-cell winners in `evaluation/process/winners.csv` are the Gen19-fed robust recipes for the TODGA "
                  "Pr/Nd nitrate case; PC88A, Cyanex 272 and D2EHPA stay literature-only (section 14)."]
    else:
        lines += [f"- No headline recommendation: label {label}, verdict {verdict}. The tables stay as labelled evidence; gen18's "
                  "chain keeps its B1 lookup default."]
    lines += ["", "## Next action", ""]
    if summary is None:
        lines += ["- Run the confirmation run (writes `evaluation/confirmation/decisions/confirmation.json`), the deployment "
                  "prediction step (fills `evaluation/process/inputs/todga_prnd_predictions.csv` from the request file) and "
                  "the residual-correlation estimate; then `scripts/g19_run_process.py`."]
    else:
        lines += ["- Confirm the feed and specification placeholders with the user (gen18 U1); if the label is "
                  "transfer-unsupported, the S2 / V6 outcome of the confirmation run decides whether a registered run follows."]
    lines += ["", f"Files: `evaluation/process/summary.json`, `winners.csv`, `stability.json`, `f5.json`, `operating_points.csv`, "
              "`process_table.csv`; figures `figures/F14_process_pareto_uncertainty.png`, "
              "`figures/F15_probability_of_specification_map.png`."]
    return "\n".join(lines) + "\n"


def write_d06(out_root: Path) -> Path:
    return write_text(d06_path(out_root), d06_markdown(out_root))


# ============================================================================================= #
# main
# ============================================================================================= #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true", help="run the gate and exit")
    ap.add_argument("--exploratory", action="store_true",
                    help="run without the confirmation decision (every output labelled transfer-unsupported)")
    ap.add_argument("--decision-only", action="store_true", help="regenerate decisions/D06 from the existing files")
    ap.add_argument("--predictions", default=None, help="prediction table CSV (default evaluation/process/inputs/...)")
    ap.add_argument("--rho-file", default=None, help="Pr/Nd residual correlation JSON (default evaluation/process/inputs/...)")
    ap.add_argument("--draws", type=int, default=MC.K_DRAWS_REGISTERED, help="joint D draws (section 14: 64)")
    ap.add_argument("--n-lhs", type=int, default=MC.N_DESIGNS_DEFAULT, help="LHS operating points (gen18 default 1000)")
    ap.add_argument("--seed", type=int, default=MC.LHS_SEED_DEFAULT, help="LHS and draw seed (gen18 default 18)")
    ap.add_argument("--bootstrap-seed", type=int, default=RO.BOOTSTRAP_SEED)
    ap.add_argument("--n-bootstrap", type=int, default=RO.N_BOOTSTRAP_RERANKINGS)
    ap.add_argument("--max-stages", type=int, default=None, help="widen the stage bounds (default: the config's)")
    ap.add_argument("--solver-max-newton", type=int, default=40)
    ap.add_argument("--solver-max-sweeps", type=int, default=10)
    ap.add_argument("--skip-f5-allowed", action="store_true", help="skip the second (allow_unsupported) run of F5(ii)")
    ap.add_argument("--no-prices", action="store_true", help="no cost proxy (config/prices.json not read)")
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def run_case(out_root: Path, ns: argparse.Namespace, gate: Mapping[str, Any]) -> dict[str, Any]:
    label = dict(gate["label"])
    pred_path = Path(ns.predictions) if ns.predictions else predictions_path(out_root)
    if not pred_path.exists():
        req = write_prediction_request(out_root, max_stages=ns.max_stages)
        raise SystemExit(f"refused: the Gen19 prediction table {pred_path} is missing. The conditions to predict were written "
                         f"to {req} (fill it with the deployed predictor's records and save it as {pred_path})")
    table = GA.read_prediction_table(pred_path)
    conf = gate.get("confirmation") or {}
    if not gate.get("exploratory") and conf.get("deployed_predictor") and table.arm != conf["deployed_predictor"]:
        raise SystemExit(f"refused: the prediction table's arm {table.arm!r} is not the deployed predictor "
                         f"{conf['deployed_predictor']!r} of the confirmation decision")
    rho = read_rho(Path(ns.rho_file) if ns.rho_file else rho_path(out_root), exploratory=bool(gate.get("exploratory")))
    if not table.has_members and not gate.get("exploratory"):
        raise SystemExit(f"refused: the prediction table {pred_path} carries no M7 member columns ({GA.MEMBER_COLUMN_PREFIX}k) / "
                         f"{GA.CONFORMAL_Q95_COLUMN}; section 14 draws one ensemble member plus a conformal residual, which "
                         "needs them (the deployment step must fill the request file's member columns). Pass --exploratory "
                         "for the flagged truncated-Gaussian fallback")
    draw_flag = None if table.has_members else {"draw_mode": MC.DRAW_MODE_FALLBACK,
                                                "flag": "exploratory: truncated-Gaussian fallback on the record (no M7 "
                                                        "member columns); not the section 14 draw"}
    if table.has_members and members_identical(table):
        # a single-model deployed arm (M0 = B5, CatBoost): the deployment step writes the mean into every member column
        # (scripts/g19_predict_process_inputs.py), so the registered draw degenerates to mean + conformal residual --
        # the fallback reading of POST-HOC addendum 2 without the truncation; flagged, never silent
        draw_flag = {"draw_mode": MC.DRAW_MODE_REGISTERED,
                     "flag": (f"the {table.n_members} member_logD_k columns are identical to mean_logD (single-model arm, "
                              "no ensemble spread): the draw is mean + conformal-scaled Gaussian residual, i.e. the "
                              "addendum 2 fallback reading without truncation; not an M7 ensemble draw")}
        if not gate.get("exploratory"):
            raise SystemExit(f"refused: the prediction table {pred_path} carries member columns identical to mean_logD "
                             "(no ensemble); a registered run needs the M7 members (section 14). Pass --exploratory")
    prices = None if ns.no_prices else Prices.load(PRICES_FILE)
    solver_kwargs = {"max_newton": ns.solver_max_newton, "max_sweeps": ns.solver_max_sweeps}
    reg_table = table.supported() if table.has_unsupported else table
    setup, case = build_case(reg_table, max_stages=ns.max_stages, solver_kwargs=solver_kwargs, prices=prices)
    log(f"case {case['system_id']} ({case['ligand']}); table {table.shape} statuses {table.statuses()}; registered box "
        f"{reg_table.box()}; rho {rho['rho']}; {ns.draws} draws x {ns.n_lhs} designs")
    t0 = time.perf_counter()

    def prog(k: int, n: int, s: float) -> None:
        log(f"draw {k}/{n}: {s:.1f} s")

    res = MC.run_monte_carlo(reg_table, setup, n_draws=ns.draws, n_designs=ns.n_lhs, seed=ns.seed, rho=rho["rho"],
                             allow_unsupported=False, progress=prog, allow_fallback=bool(gate.get("exploratory")))
    mc_seconds = time.perf_counter() - t0
    root = process_root(out_root)
    ops_allowed = None
    allowed_info: dict[str, Any] = {"run": False}
    if table.has_unsupported and not ns.skip_f5_allowed:
        log("F5(ii) allowed variant: allow_unsupported=True on the untrimmed table")
        setup_all, _ = build_case(table, max_stages=ns.max_stages, solver_kwargs=solver_kwargs, prices=prices)
        res_all = MC.run_monte_carlo(table, setup_all, n_draws=ns.draws, n_designs=ns.n_lhs, seed=ns.seed, rho=rho["rho"],
                                     allow_unsupported=True, progress=prog, allow_fallback=bool(gate.get("exploratory")))
        lab_all = {**label, "label": f"{TRANSFER_UNSUPPORTED}; F5(ii) allowed variant (UNSUPPORTED predictions allowed)",
                   "headline_allowed": False}
        out_all = write_run_outputs(root / "f5_allowed", res_all, setup_all, label=lab_all, stability_check=False)
        ops_allowed = out_all["ops"]
        allowed_info = {"run": True, "n_failed": res_all.attrs["n_failed"], "outputs": [str(p) for p in out_all["outputs"]]}
    out = write_run_outputs(root, res, setup, label=label, n_boot=ns.n_bootstrap, boot_seed=ns.bootstrap_seed,
                            ops_allowed=ops_allowed)
    figs = [figure_pareto(out["ops"], out["winners"], setup, label["label"],
                          figures_dir(out_root) / "F14_process_pareto_uncertainty.png"),
            figure_probability_map(out["ops"], out["winners"], setup, label["label"],
                                   figures_dir(out_root) / "F15_probability_of_specification_map.png")]
    tabs = [write_csv(out["ops"], tables_dir(out_root) / "process_operating_points.csv"),
            write_csv(out["winners"], tables_dir(out_root) / "process_winners.csv")]
    inputs_used = {"label": label["label"], "predictions_path": str(pred_path), "predictions_digest": paths.digests(pred_path), "arm": table.arm,
                   "system_id": table.system_id, "n_cells": int(np.prod(table.shape)), "table_statuses": table.statuses(),
                   "table_box": table.box(), "supported_box": reg_table.box(), "n_intervals_repaired": table.meta.get("n_intervals_repaired"),
                   "rho": rho["rho"], "rho_path": rho.get("path"), "rho_present": rho.get("present"), "rho_flag": rho.get("flag"),
                   "rho_n_pairs": rho.get("n_pairs"), "rho_source": rho.get("source"), "rho_n_pairs_prnd": rho.get("n_pairs_prnd"),
                   "rho_prnd": rho.get("rho_prnd"), "prices": None if ns.no_prices else str(PRICES_FILE),
                   "provenance_convention": GA.PROVENANCE_CONVENTION, "draw_mode": res.attrs.get("draw_mode"),
                   "n_members": table.n_members, "draw_flag": draw_flag}
    write_json(root / "inputs_used.json", {"schema": SCHEMA, **_json(inputs_used)})
    summary = {"schema": SCHEMA, "label": label, "gate": _json(gate), "inputs": _json(inputs_used), "case": _json(case),
               "monte_carlo": {**_json(res.attrs), "seconds": round(mc_seconds, 1)}, "f5_allowed_variant": _json(allowed_info),
               "winners": _json(out["winners"].to_dict("records")), "f5": _json(out["f5"]),
               "s2d": {k: {"fraction_kept": v["fraction_kept"], "passes": v["passes"]} for k, v in out["stability"].items()},
               "readings": REGISTRATION_READINGS, "gen18_code_sha256": code_digests()["gen18"],
               "figures": [str(p) for p in figs], "tables": [str(p) for p in tabs]}
    write_json(root / "summary.json", summary)
    outputs = list(out["outputs"]) + figs + tabs + [root / "inputs_used.json", root / "summary.json"]
    outputs += [Path(p) for p in allowed_info.get("outputs", [])]
    return {"summary": summary, "outputs": outputs, "inputs": [pred_path, CASE_FEED, CASE_SPEC, DESIGN_SPACES] +
            ([Path(rho["path"])] if rho.get("present") else []) + list(GEN18_FILES)}


def main(argv: Sequence[str] | None = None, *, check: Callable[[], int] | None = None,
         digests: Callable[[], Mapping[str, Any]] | None = None,
         heavy_discovery_check: Callable[[], dict[str, Any]] | None = None) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    if ns.decision_only:
        REG.refuse_unless_sealed(REGISTRY_STAGE, check, digests, expect_addenda=ns.expect_addenda)
        p = write_d06(out_root)
        log(f"wrote {p}")
        return 0
    gate = refuse_unless_ready(out_root, exploratory=ns.exploratory, check=check, digests=digests,
                               expect_addenda=ns.expect_addenda, heavy_discovery_check=heavy_discovery_check)
    if ns.check_only:
        print(json.dumps(_json(gate), indent=2))
        return 0
    codes = code_digests()
    ctx = Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=int(ns.seed),
              extra={"code_sha256": {"own": codes["own"], "own_parts": codes["own_parts"]},
                     "gen18_code_sha256": {"combined": codes["gen18"], "parts": codes["gen18_parts"]},
                     "prereg_gate": gate["prereg_gate"], "gate": _json(gate), "readings": REGISTRATION_READINGS,
                     "seeds": {"lhs_and_draws": int(ns.seed), "bootstrap": int(ns.bootstrap_seed)},
                     "label": gate["label"]}) if not ns.no_manifest else None
    with (ctx or _Null()) as run:
        result = run_case(out_root, ns, gate)
        d06 = write_d06(out_root)
        if run is not None:
            run.inputs(*result["inputs"])
            run.outputs(*result["outputs"], d06)
            run.extra.update({"verdict_files": {"summary": str(process_root(out_root) / "summary.json"), "d06": str(d06)},
                              "confirmation_half_read": False, "v6_target_rows_scored": 0, "learned_arm_fitted": False})
    log(f"done: {len(result['outputs'])} files; label {gate['label']['label']}; D06 at {d06}")
    return 0


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
