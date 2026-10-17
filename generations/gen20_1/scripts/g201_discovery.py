"""gen20.1 stage S9 (DISCOVERY): every arm, baseline and reference on every (design, seed, fold) unit, primary and
COH-NODMD refit, written as long float64 tables (fitting.run_design).

IMPLEMENTATION_PLAN.md §9 S9; PRE_REGISTRATION.md §4, §4.4, §8.1 compute rule.  Sealed gate first
(``require_sealed(PRE_REGISTRATION.md)``), forbidden-writer guard installed, then, before anything is fitted:
  * the S8 smoke record (``smoke_flags``): the full-size ``g201_smoke`` run (not ``--tiny``) on the real prediction
    root, finished with every unit done; a learner that failed smoke is "not run" (its columns are absent);
  * the S7 anchor gate (``anchors_gate``, PRE_REGISTRATION.md §4.4 "before any gen20.1 outcome ... beyond that
    STOP"): results/anchors/anchors_g201.json with status PASS, every anchor "reproduced", the run_fold cross-check
    and the Kish check passed;
  * fold gates (a)-(e) re-asserted on the six designs (STOP INCOMPARABLE_FOLDS).
Designs run in the order BP, B, BR, BQ, BPG, A; every (design, seed, fold, variant) unit is resumable (DONE markers
carry the code digest of the imported code, the arm set, tree counts and library versions).  With ``--workers 1`` a
design that fails is recorded (``design_<d>``: status failed, error, traceback) and the next design starts; the
script exits 1 at the end (STOP conditions -- fold gates, leakage, changed code, a held design lock -- still stop at
once).  ``--workers 2`` runs at most two child processes, each on whole designs, started in that order (fixed
seeds, fits independent of n_jobs: numbers cannot change); the children get the parent's resolved paths and its
imported code digest (``--expect-code-digest``: a child whose imported code differs refuses before fitting), and
are terminated when the parent exits abnormally.  No metric is computed here; S10 (g201_report.py) scores the saved
tables.

Usage (repo root):
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B generations/gen20_1/scripts/g201_discovery.py \
      [--designs BP,B,BR,BQ,BPG,A] [--label discovery] [--variants primary,nodmd] [--workers 1|2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
G201 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G201))

from gen201 import bootstrap  # noqa: E402
from gen201 import guard  # noqa: E402
from gen201.seal import require_sealed  # noqa: E402

guard.install()

from gen201 import data as D  # noqa: E402
from gen201 import fitting as FI  # noqa: E402
from gen201 import folds as FO  # noqa: E402
from gen201 import learners as L  # noqa: E402
from gen201.run import Run, sha256_file  # noqa: E402

DEFAULT_SMOKE = bootstrap.MANIFESTS / "g201_smoke__smoke_S8.json"
DEFAULT_ANCHORS = bootstrap.RESULTS / "anchors" / "anchors_g201.json"
ANCHOR_NAMES: tuple[str, ...] = ("FLAT", "G14", "MEAN_CURVE", "NN_TANIMOTO")
#: exceptions that stop the whole run instead of failing one design
STOP_ERRORS = (FO.FoldGateError, L.LeakageError, FI.CodeChanged, FI.DesignLocked)


def smoke_flags(path: Path) -> dict:
    """The S8 pass flags; discovery (and the confirmation) refuse to start without a completed, full-size smoke record:
    written by ``g201_smoke`` with a ``smoke*`` label, not ``--tiny``, its units written under the real prediction
    root (generations/gen20_1/predictions) and every unit done."""
    if not Path(path).exists():
        raise RuntimeError(f"STOP: no S8 smoke record at {path}; run scripts/g201_smoke.py first")
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not str(rec.get("status", "")).startswith("ok"):
        raise RuntimeError(f"STOP: the S8 smoke run did not finish ({rec.get('status')})")
    if rec.get("name") != "g201_smoke":
        raise RuntimeError(f"STOP: {path} is not a g201_smoke record (name {rec.get('name')!r})")
    if not str(rec.get("label", "")).startswith("smoke"):
        raise RuntimeError(f"STOP: the S8 smoke record carries the label {rec.get('label')!r}, not 'smoke*'")
    if (rec.get("args") or {}).get("tiny"):
        raise RuntimeError("STOP: the S8 smoke record is a --tiny dry run (4/2 trees), not the registered smoke test")
    rd = (rec.get("records") or {}).get("run_design") or {}
    if not rd.get("root") or Path(rd["root"]).resolve() != bootstrap.PREDICTIONS.resolve():
        raise RuntimeError(f"STOP: the S8 smoke run wrote to {rd.get('root')!r}, not {bootstrap.PREDICTIONS}")
    units = rd.get("units") or []
    if not units or not all(u.get("status") == "done" for u in units):
        raise RuntimeError(f"STOP: the S8 smoke run did not fit every unit ({[u.get('status') for u in units]})")
    flags = rec.get("records", {}).get("pass")
    if not flags:
        raise RuntimeError("STOP: the S8 smoke record carries no pass flags")
    for k in ("coverage", "finite", "frozen_unchanged"):
        if not flags.get(k):
            raise RuntimeError(f"STOP: S8 smoke failed '{k}'")
    return {"C": bool(flags.get("C")), "C_PAIR": bool(flags.get("C_PAIR")), "label": rec.get("label"),
            "arm_failures": rec.get("records", {}).get("arm_failures") or {},
            "projection": rec.get("records", {}).get("projection"), "path": str(Path(path).resolve()),
            "sha256": sha256_file(path)}


def anchors_gate(path: Path) -> dict:
    """S7 anchor gate (PRE_REGISTRATION.md §4.4, IMPLEMENTATION_PLAN.md §9 S7): refuses (STOP) unless the anchor
    record has status PASS, all four anchors "reproduced" (either form), the run_fold cross-check passed and the Kish
    check passed."""
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"STOP: no S7 anchor record at {p}; run scripts/g201_anchors.py first (the anchor gate "
                           "precedes every gen20.1 outcome)")
    rec = json.loads(p.read_text(encoding="utf-8"))
    anchors = rec.get("anchors") or {}
    st = {k: str((v or {}).get("status", "")) for k, v in anchors.items()}
    problems = []
    if rec.get("status") != "PASS":
        problems.append(f"status {rec.get('status')!r}")
    missing = [a for a in ANCHOR_NAMES if a not in st]
    if missing:
        problems.append(f"anchors missing {missing}")
    bad = {k: v for k, v in st.items() if not v.startswith("reproduced")}
    if bad:
        problems.append(f"anchors not reproduced {bad}")
    cross = rec.get("run_fold_crosscheck") or {}
    if cross.get("pass") is not True:
        problems.append("the run_fold cross-check did not pass (or was skipped)")
    if rec.get("kish_pass") is not True:
        problems.append("the Kish check did not pass")
    if problems:
        raise RuntimeError(f"STOP: S7 anchor gate ({p}): " + "; ".join(problems))
    return {"path": str(p.resolve()), "sha256": sha256_file(p), "status": rec["status"], "anchors": st,
            "run_fold_crosscheck": {k: cross.get(k) for k in ("pass", "max_abs_delta", "rows_compared")},
            "kish_pass": True}


def ordered(designs) -> list[str]:
    bad = [d for d in designs if d not in FO.DESIGNS]
    if bad:
        raise ValueError(f"unknown designs {bad}")
    return [d for d in FO.DESIGNS if d in set(designs)]


def child_command(design: str, args, *, root: Path, code_sha: str) -> list[str]:
    """The child's command line: resolved paths (never the parent's raw arguments) and the parent's code digest."""
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--designs", design, "--label", args.label,
           "--variants", args.variants, "--workers", "1", "--child",
           "--smoke-record", str(Path(args.smoke_record).resolve()),
           "--anchors-record", str(Path(args.anchors_record).resolve()),
           "--root", str(Path(root).resolve()), "--expect-code-digest", code_sha]
    for flag in ("no_resume", "tiny", "no_subprocess"):
        if getattr(args, flag):
            cmd.append("--" + flag.replace("_", "-"))
    if args.manifest_dir:
        cmd += ["--manifest-dir", str(Path(args.manifest_dir).resolve())]
    return cmd


def run_children(designs, args, log, *, root: Path, code_sha: str, poll: float = 5.0) -> dict:
    """At most ``args.workers`` child processes, each on one whole design, started in the registered order; on an
    abnormal exit of the parent (exception, Ctrl+C) the running children are terminated."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    queue, running, results = list(designs), {}, {}
    try:
        while queue or running:
            while queue and len(running) < args.workers:
                d = queue.pop(0)
                running[d] = subprocess.Popen(child_command(d, args, root=root, code_sha=code_sha),
                                              cwd=str(bootstrap.ROOT), env=env)
                log(f"started child for design {d} (pid {running[d].pid})")
            time.sleep(poll)
            for d, p in list(running.items()):
                rc = p.poll()
                if rc is not None:
                    results[d] = rc
                    log(f"design {d} finished with exit code {rc}")
                    del running[d]
    finally:
        for d, p in running.items():
            if p.poll() is None:
                log(f"terminating child for design {d} (pid {p.pid})")
                p.terminate()
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
    return results


def run_designs(designs, run_one, run, log) -> list[str]:
    """``run_one(d)`` for every design in order; a design that raises is recorded as failed and the next design
    starts (STOP_ERRORS stop the run).  Returns the failed designs."""
    failed = []
    for d in designs:
        try:
            run.record(f"design_{d}", run_one(d))
        except STOP_ERRORS:
            raise
        except Exception as e:
            run.record(f"design_{d}", {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()})
            log(f"design {d} FAILED ({e!r}); continuing with the next design")
            failed.append(d)
    return failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--designs", default=",".join(FO.DESIGNS))
    ap.add_argument("--label", default="discovery")
    ap.add_argument("--variants", default="primary,nodmd")
    ap.add_argument("--workers", type=int, default=1, choices=(1, 2))
    ap.add_argument("--smoke-record", default=str(DEFAULT_SMOKE))
    ap.add_argument("--anchors-record", default=str(DEFAULT_ANCHORS))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--expect-code-digest", default=None, help=argparse.SUPPRESS)  # set by the parent for children
    ap.add_argument("--root", default=None, help="prediction root (default generations/gen20_1/predictions)")
    ap.add_argument("--manifest-dir", default=None, help="run-manifest directory (default generations/gen20_1/manifests)")
    ap.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)          # dry runs only: 4 / 2 trees
    ap.add_argument("--no-subprocess", action="store_true", help=argparse.SUPPRESS)  # dry runs only
    args = ap.parse_args(argv)
    seal_state = require_sealed(bootstrap.PREREG)
    if args.expect_code_digest is not None and args.expect_code_digest != FI.CODE_AT_IMPORT["sha256"]:
        print(f"STOP: this process imported code {FI.CODE_AT_IMPORT['sha256']}, the parent expects "
              f"{args.expect_code_digest}: the code changed between the parent's start and this child's",
              file=sys.stderr)
        return 1
    root = bootstrap.assert_under_g201(args.root if args.root else bootstrap.PREDICTIONS)
    if args.tiny:
        if root == bootstrap.PREDICTIONS.resolve():
            print("STOP: --tiny is for dry runs and never writes under predictions/", file=sys.stderr)
            return 1
        L.N_ESTIMATORS, L.C_PAIR_N_ESTIMATORS = 4, 2
    if args.label.startswith("smoke"):
        print("STOP: discovery refuses a 'smoke*' label", file=sys.stderr)
        return 1
    if not L.registered_tree_counts() and not args.tiny:
        print("STOP: tree counts differ from the registered 400/200", file=sys.stderr)
        return 1
    designs = ordered([d.strip() for d in args.designs.split(",") if d.strip()])
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    before = guard.snapshot()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_label = f"{args.label}_{'-'.join(designs)}_{stamp}"
    status = 1
    with Run("g201_discovery", label=run_label, args=vars(args), seeds=list(FO.SEEDS), manifest_dir=args.manifest_dir) as run:
        run.record("preregistration", seal_state)
        run.record("code_digest", FI.assert_code_unchanged())
        smoke = smoke_flags(Path(args.smoke_record))
        run.record("smoke", smoke)
        run.record("anchors_gate", anchors_gate(Path(args.anchors_record)))
        bench = D.load(check=True)
        run.record("cohort", D.LOAD_RECORD)
        F = FO.build(bench)                                   # one gen20k.splits.folds_for call per design
        run.record("fold_gates", FI.fold_gates(bench, F, subprocess_check=not (args.child or args.no_subprocess)))
        if args.workers > 1 and len(designs) > 1:
            rc = run_children(designs, args, run.log, root=root, code_sha=FI.CODE_AT_IMPORT["sha256"])
            run.record("children", rc)
            ok = all(v == 0 for v in rc.values()) and set(rc) == set(designs)
        else:
            ctx = FI.make_context(bench)

            def run_one(d):
                return FI.run_design(bench, d, seeds=FO.SEEDS, label=args.label, variants=variants,
                                     resume=not args.no_resume, include_c=smoke["C"], include_c_pair=smoke["C_PAIR"],
                                     folds_ctx=ctx, folds_list=F[d], log=run.log, out_root=root)

            failed = run_designs(designs, run_one, run, run.log)
            run.record("failed_designs", failed)
            ok = not failed
        guard.assert_unchanged(before)
        run.record("frozen_trees_unchanged", True)
        status = 0 if ok else 1
    return status


if __name__ == "__main__":
    sys.exit(main())
