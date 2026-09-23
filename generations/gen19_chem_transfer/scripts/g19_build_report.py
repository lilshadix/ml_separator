"""``scripts/g19_build_report.py`` -- ``GEN19_REPORT.md``, ``SUMMARY.md``, ``decisions/D02_factorization.md`` and
``tables/claims.json`` GENERATED FROM FILES (``gen19ct.evaluation.report``; brief sections 20, 22, 29, 34;
pre-registration sections 9, 10, 15, 16, 17, 19).

Gate (refuses to start unless BOTH hold; ``--check-only`` prints the verdict):

1. ``scripts/g19_seal_prereg.py --check`` exits 0 AND the below-footer text digests to the ``report`` stage's registry entry
   (``registry.refuse_unless_sealed``; addendum 2 item 5, re-registered after every addendum by addendum 7 item 3);
2. the discovery run is **COMPLETE** (``gen19ct.evaluation.h3.discovery_complete``): ``evaluation/discovery/decisions/
   wall_clock.json`` records an invocation whose ``stages_done`` reached the final plan stage ``10_not_implemented``, every
   stage of the current plan is done, and every ``fit`` job of ``discovery.enumerate_plan(plan_state)`` has a verified
   COMPLETE record set (digest, fold hash and fold set exactly what the current code, fold files and plan state produce).
3. the ladder has run every step M3-M7 to a done / skipped status (``h3.ladder_complete`` on
   ``evaluation/ladder/decisions/ladder.json``): the deployed predictor is 'the retained ladder configuration'
   (task X finding V-01).  Gate 0, before anything heavy: ``wall_clock.json`` reached the final stage
   (``h3.refuse_unless_cheap_complete``; finding VL2-04).

The report itself is conditioned by no sealed decision: every later step (scorer, ladder, H3, power, confirmation,
process) that has not run is printed as ``not computed (input missing: <path>)`` with its status ``not run``.

Every number written is recorded with its source path and key (``tables/report_numbers.csv``) and RE-RESOLVED from that
source after writing (``report.verify_numbers``; ``tables/report_numbers_verification.csv``); a number that does not
re-resolve makes the script exit 3.  Manifest ``manifests/g19_build_report.json``: git HEAD, the sealed digest and the
addenda digest, the code digest of this script and ``report.py``, the input list, runtime (``run_info``).

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_report.py --check-only
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_build_report.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen19ct import paths  # noqa: E402
from gen19ct.evaluation import discovery as D  # noqa: E402
from gen19ct.evaluation import h3 as H3  # noqa: E402
from gen19ct.evaluation import registry as REG  # noqa: E402
from gen19ct.evaluation import report as R  # noqa: E402
from gen19ct.manifest import Run, git_head, write_csv, write_json  # noqa: E402

NAME = "g19_build_report"
#: the registry stage of this runner (addendum 2 item 5: the seal gate and the discovery code digest prefer
#: manifests/digest_registry.json when it exists and fall back to the constants of evaluation.discovery otherwise; the
#: report prints every registry entry, evaluation.report.reproducibility_section)
STAGE = "report"
CODE_FILES: tuple[Path, ...] = (paths.G19_ROOT / "gen19ct" / "evaluation" / "report.py", Path(__file__).resolve())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def runner_module():
    return _load("g19_run_discovery")


def code_digest() -> dict[str, Any]:
    return D.code_digest(CODE_FILES)


def refuse_unless_ready(out_root: Path, *, check: Callable[[], int] | None = None,
                        digests: Callable[[], Mapping[str, Any]] | None = None, expect_addenda: int | None = None,
                        excluded_ids: Sequence[str] = (), folds_dir: Path | None = None,
                        discovery_code: str | None = None, runners: Mapping[str, Any] | None = None,
                        jobs: Sequence[D.JobSpec] | None = None, state: D.PlanState | None = None,
                        require_ladder: bool = True) -> dict[str, Any]:
    """The gates of the module docstring; raises ``SystemExit`` naming what is missing.  ``require_ladder``: the ladder
    has run every step M3-M7 to a done / skipped status (``h3.ladder_complete``) -- the report's deployed predictor is
    'the retained ladder configuration' and Q5 reads the same file (task X finding V-01)."""
    rd = runner_module()
    prereg = REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=expect_addenda)
    st = state if state is not None else D.PlanState.read(rd.plan_state_path(out_root))
    code = discovery_code if discovery_code is not None else REG.discovery_code_digest()
    done = H3.discovery_complete(out_root, code=code, state=st, excluded_ids=excluded_ids, folds_dir=folds_dir,
                                 runners=runners, jobs=jobs)
    if not done["complete"]:
        raise SystemExit("refused: the discovery run is not COMPLETE (reached final stage "
                         f"{done['final_stage']}: {done['reached_final_stage']}; missing stages {done['missing_stages']}; "
                         f"{done['n_incomplete']} of {done['n_fit_record_sets']} fit record sets incomplete). The report is "
                         "the final deliverable and is built only from a complete discovery run; let "
                         "scripts/g19_run_discovery.py finish, then rerun")
    lc = H3.ladder_complete(H3.read_ladder_state(out_root))
    if require_ladder and not lc["complete"]:
        raise SystemExit("refused: the ladder (M3-M7) is not complete -- the report's deployed predictor is 'the retained "
                         f"ladder configuration' ({H3.LADDER_STATE_FILE} {'absent' if not lc['present'] else 'steps not done: ' + str(lc['not_done'])}). "
                         "Run scripts/g19_run_ladder.py to completion, then rerun")
    return {"prereg_gate": prereg, "discovery_complete": done, "scorer_decisions": H3.scorer_decisions_present(out_root),
            "ladder_complete": lc, "plan_state": st.record(), "discovery_code_sha256": code}


def prereg_extra(out_root: Path) -> dict[str, Any]:
    """The digests the reproducibility section prints (sealed footer, recomputed, addenda) read with the seal script."""
    sealed = Path(out_root) / "preregistration.md"
    sha = Path(out_root) / "manifests" / "prereg_sha256.txt"
    if not sealed.exists():
        return {}
    try:
        dg = runner_module().prereg_digests(sealed, sha)
    except (OSError, ValueError):
        return {}
    return {"addenda_sha256": dg.get("addenda_sha256"), "n_addenda": dg.get("n_addenda"), "recomputed": dg.get("recomputed"),
            "footer": dg.get("footer")}


def build_and_write(out_root: Path, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build every artefact, write it, re-resolve every number; returns the result with ``verification``."""
    res = R.build_report(out_root, extra=extra)
    outs = R.write_outputs(out_root, res)
    ver = R.verify_numbers(out_root, res["numbers"])
    outs.append(write_csv(ver, Path(out_root) / "tables" / "report_numbers_verification.csv"))
    res["outputs"] = outs
    res["verification"] = {"n_numbers": int(len(ver)), "n_ok": int(ver["ok"].sum()) if len(ver) else 0,
                           "all_ok": bool(ver["ok"].all()) if len(ver) else True,
                           "failed": ver[~ver["ok"].astype(bool)].head(20).to_dict("records") if len(ver) else []}
    return res


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", default=str(paths.G19_ROOT))
    ap.add_argument("--check-only", action="store_true", help="run the gate and exit")
    ap.add_argument("--expect-addenda", type=int, default=None)
    ap.add_argument("--no-manifest", action="store_true")
    return ap.parse_args(argv)


def main(argv=None, *, check: Callable[[], int] | None = None, digests: Callable[[], Mapping[str, Any]] | None = None
         ) -> int:
    ns = parse_args(argv)
    out_root = Path(ns.out_root)
    rd = runner_module()
    REG.refuse_unless_sealed(STAGE, check, digests, expect_addenda=ns.expect_addenda)
    H3.refuse_unless_cheap_complete(out_root)                   # before any heavy load (task X finding VL2-04)
    log("coextractant ids")
    coext = rd.coextractant_ids()
    gate = refuse_unless_ready(out_root, check=check, digests=digests, expect_addenda=ns.expect_addenda, excluded_ids=coext)
    if ns.check_only:
        print(json.dumps({"discovery_complete": gate["discovery_complete"]["complete"],
                          "scorer_decisions_ok": gate["scorer_decisions"]["ok"], "ladder_complete": gate["ladder_complete"]},
                         indent=2, default=str))
        return 0
    if not gate["scorer_decisions"]["ok"]:
        log("warning: the scorer's decision files are missing or stale; the report will say so (not computed)")
    code = code_digest()
    extra = {"git_head": git_head(), **prereg_extra(out_root)}
    registry = REG.read_registry(Path(out_root) / REG.REGISTRY_REL)     # addendum 2 item 5: printed in full when present
    if registry is not None:
        extra["digest_registry"] = registry
    with (Run(NAME, args={k: v for k, v in vars(ns).items()}, seed=D.PRIMARY_SEED,
              extra={"prereg_gate": gate["prereg_gate"], "gate": {k: v for k, v in gate.items() if k != "prereg_gate"},
                     "code_sha256": code["combined"], "code_parts": code["parts"], "prereg_digests": extra,
                     "discovery_seeds": list(D.DISCOVERY_SEEDS), "seed": D.PRIMARY_SEED})
          if not ns.no_manifest else _Null()) as run:
        res = build_and_write(out_root, extra=extra)
        if run is not None:
            run.inputs(*[Path(out_root) / p for p, ok in res["ledger"].inputs.items() if ok])
            run.outputs(*res["outputs"])
            run.extra.update({"verification": res["verification"], "missing_inputs": sorted(res["ledger"].missing_inputs),
                              "missing_keys": sorted(res["ledger"].missing_keys),
                              "n_not_computed_in_report": int(res["report"].count(R.NOT_COMPUTED.split("{")[0])),
                              "n_not_computed_in_summary": int(res["summary"].count(R.NOT_COMPUTED.split("{")[0])),
                              "state": res["claims"].get("state"),
                              "deployed_predictor": res["context"]["deployed"], "confirmation_half_read": False,
                              "v6_target_rows_scored": 0})
    v = res["verification"]
    log(f"report written: {v['n_numbers']} numbers, {v['n_ok']} re-resolved from their sources; missing inputs: "
        f"{len(res['ledger'].missing_inputs)}; missing keys of existing files: {len(res['ledger'].missing_keys)}; "
        f"'not computed' sentences: {res['report'].count(R.NOT_COMPUTED.split('{')[0])} in the report, "
        f"{res['summary'].count(R.NOT_COMPUTED.split('{')[0])} in the summary")
    if not v["all_ok"]:
        log(f"ERROR: {v['n_numbers'] - v['n_ok']} number(s) do not re-resolve: {v['failed'][:5]}")
        return 3
    return 0


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
