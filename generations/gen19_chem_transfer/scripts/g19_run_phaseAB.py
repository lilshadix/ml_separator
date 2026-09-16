"""``scripts/g19_run_phaseAB.py`` -- run the brief section 28 Phase A (corpus audit) and Phase B
(feasibility) scripts in dependency order, and optionally prove the pipeline is byte-deterministic.

Nothing is trained or fitted.  Each step runs as its own Python process (the archive is loaded once
per step and freed when the step exits; the 8 GB machine never holds two copies):

    1  g19_build_extractants  descriptors/extractant_components.csv, extractant_systems.csv,
                              family_rules.json; data_audit/ligand_alias_collisions.csv,
                              family_coverage.csv, named_extractant_presence.csv
    2  g19_build_metals       descriptors/metals.csv, metals_sources.md;
                              data_audit/metal_alias_audit.csv, metal_descriptor_coverage.csv
    3  g19_audit_leakage      data_audit/leakage_*.csv, leakage_summary.json, metadata_availability.csv
    4  g19_audit_corpus       data_audit/counts.json, sparsity.json, dataset_hashes.csv, columns.csv,
                              matrix_*.csv, *_coverage.csv; figures/F01, F02, F04, F05, F06
    5  g19_feasibility        data_audit/feasibility.json, feasibility_*.csv; figures/F03
                              (reads the outputs of steps 1-3)

Every step writes its own ``manifests/<step>.json`` (deterministic) and ``manifests/run_info/<step>.json``
(volatile).  This runner writes no scientific output and therefore no ``Run`` manifest of its own; the
only file it writes is the determinism record below.

``--determinism`` runs the whole pipeline twice and compares, byte for byte, every file under
``data_audit/``, ``descriptors/``, ``figures/`` and every ``manifests/<step>.json`` between the two runs
(``manifests/run_info/`` is compared too but is volatile by design and never counts).  It also compares
run 1 with the tree as it was before the check (the outputs the documents cite), verifies each step
manifest's output digests against the files on disk, lists files no step manifest claims, and checks
that no code file and no git HEAD changed during the check.  ``PYTHONHASHSEED`` is left as the caller's
environment has it (normally unset, so string hashing is randomised per process): a result that depends
on set/dict iteration order of strings shows up as a difference.  Record:
``manifests/phaseAB_determinism.json`` (its ``run_info`` block is volatile).

Usage (repository root)::

    PYTHONIOENCODING=utf-8 PYTHONPATH=generations/gen19_chem_transfer \\
        .venv/Scripts/python.exe generations/gen19_chem_transfer/scripts/g19_run_phaseAB.py
    ... g19_run_phaseAB.py --steps g19_audit_leakage,g19_feasibility   # subset, kept in pipeline order
    ... g19_run_phaseAB.py --list
    ... g19_run_phaseAB.py --determinism                               # ~7 min: two full runs

Exit codes: 0 success (and, with ``--determinism``, byte-identical); 1 a determinism difference;
2 a step failed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen19ct import paths  # noqa: E402
from gen19ct.manifest import git_head, write_json  # noqa: E402

STEPS: tuple[str, ...] = (
    "g19_build_extractants",
    "g19_build_metals",
    "g19_audit_leakage",
    "g19_audit_corpus",
    "g19_feasibility",
)
SCRIPTS_DIR = paths.G19_ROOT / "scripts"
RECORD = paths.MANIFESTS_DIR / "phaseAB_determinism.json"
TEXT_DIFF_SUFFIXES = {".csv", ".json", ".md", ".txt"}
#: Code whose change during a determinism check would invalidate it.
CODE_GLOBS = ("gen19ct/**/*.py", "scripts/g19_*.py")


# --------------------------------------------------------------------------------------------- running
def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    g19 = str(paths.G19_ROOT)
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = g19 if not old else g19 + os.pathsep + old
    return env


def run_step(name: str, *, verbose: bool = False) -> dict[str, Any]:
    script = SCRIPTS_DIR / f"{name}.py"
    cmd = [sys.executable, str(script)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=paths.REPO_ROOT, env=_env(), capture_output=not verbose,
                          text=True, encoding="utf-8", errors="replace")
    dt = round(time.perf_counter() - t0, 1)
    tail = ""
    if not verbose:
        lines = ((proc.stdout or "") + (proc.stderr or "")).rstrip().splitlines()
        tail = "\n".join(lines[-25:] if proc.returncode else lines[-2:])
    return {"step": name, "returncode": proc.returncode, "runtime_s": dt, "tail": tail}


def run_pipeline(steps: tuple[str, ...], *, label: str = "", verbose: bool = False) -> list[dict[str, Any]]:
    results = []
    for i, name in enumerate(steps, 1):
        print(f"{label}[{i}/{len(steps)}] {name} ...", flush=True)
        r = run_step(name, verbose=verbose)
        results.append(r)
        status = "ok" if r["returncode"] == 0 else f"FAILED (exit {r['returncode']})"
        print(f"{label}[{i}/{len(steps)}] {name} {status} in {r['runtime_s']} s", flush=True)
        if r["tail"]:
            print("    " + r["tail"].replace("\n", "\n    "), flush=True)
        if r["returncode"] != 0:
            break
    return results


# --------------------------------------------------------------------------------------- snapshotting
def _category(p: Path) -> str | None:
    relp = p.relative_to(paths.G19_ROOT)
    top = relp.parts[0]
    if top == "manifests":
        if len(relp.parts) == 2 and p.suffix == ".json" and p != RECORD:
            return "manifest"
        if len(relp.parts) == 3 and relp.parts[1] == "run_info":
            return "run_info"
        return None
    return {"data_audit": "data_audit", "descriptors": "descriptors", "figures": "figure"}.get(top)


def tracked_files() -> list[Path]:
    out = []
    for d in (paths.DATA_AUDIT_DIR, paths.DESCRIPTORS_DIR, paths.FIGURES_DIR, paths.MANIFESTS_DIR):
        if d.exists():
            out.extend(p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                       and _category(p) is not None)
    return sorted(out)


def snapshot(keep_text: bool = False) -> dict[str, dict[str, Any]]:
    snap = {}
    for p in tracked_files():
        data = p.read_bytes()
        rec: dict[str, Any] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                               "category": _category(p)}
        if keep_text and p.suffix.lower() in TEXT_DIFF_SUFFIXES:
            rec["_data"] = data
        snap[paths.rel(p)] = rec
    return snap


def code_digests() -> dict[str, str]:
    out = {}
    for pattern in CODE_GLOBS:
        for p in sorted(paths.G19_ROOT.glob(pattern)):
            if "__pycache__" not in p.parts:
                out[paths.rel(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ------------------------------------------------------------------------------------------ diffing
def _short(v: Any, n: int = 80) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


def json_diff(a: Any, b: Any, where: str = "$", out: list[str] | None = None, limit: int = 15) -> list[str]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if len(out) >= limit:
                break
            if k not in a or k not in b:
                out.append(f"{where}.{k}: {'added in run 2' if k not in a else 'missing in run 2'}")
            else:
                json_diff(a[k], b[k], f"{where}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{where}: list length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            if len(out) >= limit:
                break
            tag = x.get("path", i) if isinstance(x, dict) else i
            json_diff(x, y, f"{where}[{tag}]", out, limit)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
        out.append(f"{where}: {_short(a)} -> {_short(b)}")
    return out


def text_diff(a: bytes, b: bytes, suffix: str) -> dict[str, Any]:
    la = a.decode("utf-8", "replace").splitlines()
    lb = b.decode("utf-8", "replace").splitlines()
    first = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
    info: dict[str, Any] = {
        "lines_run1": len(la), "lines_run2": len(lb),
        "n_differing_line_positions": sum(x != y for x, y in zip(la, lb)) + abs(len(la) - len(lb)),
        "same_lines_different_order": la != lb and sorted(la) == sorted(lb),
        "crlf_only": a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n"),
        "first_difference_line": first + 1,
        "first_difference_run1": la[first][:240] if first < len(la) else None,
        "first_difference_run2": lb[first][:240] if first < len(lb) else None,
    }
    if suffix == ".json":
        try:
            info["json_paths"] = json_diff(json.loads(a), json.loads(b))
        except ValueError as exc:  # pragma: no cover -- a broken JSON is itself the finding
            info["json_parse_error"] = str(exc)
    return info


def compare(s_a: dict[str, dict], s_b: dict[str, dict], *, with_detail: bool) -> dict[str, dict[str, Any]]:
    out = {}
    for k in sorted(set(s_a) | set(s_b)):
        ra, rb = s_a.get(k), s_b.get(k)
        if ra is None:
            out[k] = {"status": "missing_in_first"}
        elif rb is None:
            out[k] = {"status": "missing_in_second"}
        elif ra["sha256"] == rb["sha256"]:
            out[k] = {"status": "identical"}
        else:
            rec: dict[str, Any] = {"status": "different"}
            if with_detail and "_data" in ra and "_data" in rb:
                rec["detail"] = text_diff(ra["_data"], rb["_data"], Path(k).suffix.lower())
            out[k] = rec
    return out


# ------------------------------------------------------------------------------------ verification
def verify_manifests(steps: tuple[str, ...]) -> tuple[dict[str, Any], set[str]]:
    """Output digests of each step manifest vs the files on disk; returns (report, claimed outputs)."""
    report, claimed = {}, set()
    for name in steps:
        mp = paths.MANIFESTS_DIR / f"{name}.json"
        if not mp.exists():
            report[name] = {"manifest_missing": True}
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        bad, missing = [], []
        for o in m.get("outputs", []):
            claimed.add(o["path"])
            f = paths.REPO_ROOT / o["path"]
            if not f.exists():
                missing.append(o["path"])
            elif not paths.matches(f, o["sha256"]):
                bad.append(o["path"])
        report[name] = {"git_head": m.get("git_head"), "n_inputs": len(m.get("inputs", [])),
                        "n_outputs": len(m.get("outputs", [])), "outputs_digest_mismatch": bad,
                        "outputs_missing": missing}
    return report, claimed


# ------------------------------------------------------------------------------------- determinism
def determinism(steps: tuple[str, ...], *, verbose: bool) -> int:
    if steps != STEPS:
        print("--determinism always runs the full pipeline; --steps is ignored", flush=True)
    head0, code0 = git_head(), code_digests()
    t_start = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    pre = snapshot(keep_text=False)

    runs = []
    snaps = []
    for r in (1, 2):
        res = run_pipeline(STEPS, label=f"run {r} ", verbose=verbose)
        runs.append(res)
        if any(x["returncode"] != 0 for x in res):
            print(f"run {r}: a step failed; no determinism record written", flush=True)
            return 2
        snaps.append(snapshot(keep_text=True))
    head1, code1 = git_head(), code_digests()
    s1, s2 = snaps

    r12 = compare(s1, s2, with_detail=True)
    p01 = compare(pre, s1, with_detail=False)
    manifest_check, claimed = verify_manifests(STEPS)

    files = []
    for k in sorted(set(s1) | set(s2) | set(pre)):
        cat = (s2.get(k) or s1.get(k) or pre.get(k))["category"]
        rec: dict[str, Any] = {
            "path": k, "category": cat, "suffix": Path(k).suffix.lower(),
            "bytes": (s2.get(k) or s1.get(k) or {}).get("bytes"),
            "sha256_run1": s1.get(k, {}).get("sha256"), "sha256_run2": s2.get(k, {}).get("sha256"),
            "run1_vs_run2": r12.get(k, {"status": "absent_in_both_runs"})["status"],
            "before_vs_run1": p01.get(k, {"status": "absent_before_and_in_run1"})["status"]
                              .replace("_in_first", "_before").replace("_in_second", "_in_run1"),
            "claimed_by_a_step_manifest": k in claimed if cat != "run_info" else None,
        }
        if "detail" in r12.get(k, {}):
            rec["difference_detail"] = r12[k]["detail"]
        files.append(rec)

    def _count(pred) -> dict[str, int]:
        sel = [f for f in files if pred(f)]
        c: dict[str, int] = {}
        for f in sel:
            c[f["run1_vs_run2"]] = c.get(f["run1_vs_run2"], 0) + 1
        return {"n_files": len(sel), **dict(sorted(c.items()))}

    required = lambda f: f["category"] == "manifest" or (  # noqa: E731
        f["category"] in ("data_audit", "descriptors") and f["suffix"] in (".csv", ".json"))
    non_volatile = lambda f: f["category"] != "run_info"  # noqa: E731
    by_cat = {c: _count(lambda f, c=c: f["category"] == c)
              for c in ("manifest", "data_audit", "descriptors", "figure", "run_info")}
    req_ok = all(f["run1_vs_run2"] == "identical" for f in files if required(f))
    all_ok = all(f["run1_vs_run2"] == "identical" for f in files if non_volatile(f))
    cited_ok = all(f["before_vs_run1"] == "identical" for f in files if non_volatile(f))
    code_ok = code0 == code1 and head0 == head1
    manifests_ok = all(not v.get("manifest_missing") and not v["outputs_digest_mismatch"]
                       and not v["outputs_missing"] for v in manifest_check.values())
    unclaimed = [f["path"] for f in files if f["category"] not in ("manifest", "run_info")
                 and not f["claimed_by_a_step_manifest"]]

    record = {
        "schema": "gen19.phaseAB_determinism.v1",
        "script": "g19_run_phaseAB",
        "question": "Does rerunning the Phase A/B pipeline on the same commit and data reproduce every "
                    "output and manifest byte for byte?",
        "git_head": head1,
        "archive_master_sha256": paths.ARCHIVE_MASTER_SHA256,
        "steps_in_order": list(STEPS),
        "n_runs": 2,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset (randomised per process)"),
        "compared": "sha256 of every file under data_audit/, descriptors/, figures/ and every "
                    "manifests/<step>.json (manifests/run_info/ is volatile by design and never counts)",
        "verdict": {
            "manifests_and_data_audit_descriptors_csv_json_identical": req_ok,
            "all_non_volatile_files_identical": all_ok,
            "run1_reproduces_tree_before_check": cited_ok,
            "code_and_git_head_unchanged_during_check": code_ok,
            "step_manifest_output_digests_match_disk": manifests_ok,
            "DETERMINISTIC": req_ok and all_ok and code_ok and manifests_ok,
        },
        "counts_run1_vs_run2": {"required": _count(required), "all_non_volatile": _count(non_volatile),
                                 "by_category": by_cat},
        "counts_before_vs_run1": {
            "n_identical": sum(f["before_vs_run1"] == "identical" for f in files if non_volatile(f)),
            "n_other": sum(f["before_vs_run1"] != "identical" for f in files if non_volatile(f)),
            "other": [{"path": f["path"], "status": f["before_vs_run1"]} for f in files
                      if non_volatile(f) and f["before_vs_run1"] != "identical"],
        },
        "step_manifest_verification": manifest_check,
        "files_not_claimed_by_any_step_manifest": unclaimed,
        "code_files_checked": len(code1),
        "code_files_changed_during_check": sorted(k for k in set(code0) | set(code1) if code0.get(k) != code1.get(k)),
        "files": files,
        "run_info": {  # volatile: wall clock, never part of a byte-identity check
            "started_utc": t_start,
            "finished_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "runtime_s": [{r["step"]: r["runtime_s"] for r in res} for res in runs],
        },
    }
    write_json(RECORD, record)
    v = record["verdict"]
    c = record["counts_run1_vs_run2"]
    print(f"run1 vs run2: required {c['required']}; all non-volatile {c['all_non_volatile']}")
    print(f"before vs run1: {record['counts_before_vs_run1']['n_identical']} identical, "
          f"{record['counts_before_vs_run1']['n_other']} other")
    print(f"verdict: {v}")
    print(f"record: {paths.rel(RECORD)}")
    return 0 if v["DETERMINISTIC"] else 1


# ------------------------------------------------------------------------------------------------ main
def parse_steps(spec: str | None) -> tuple[str, ...]:
    if not spec:
        return STEPS
    wanted = [s.strip().removesuffix(".py") for s in spec.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; known: {', '.join(STEPS)}")
    return tuple(s for s in STEPS if s in wanted)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--steps", help="comma-separated subset of steps (always run in pipeline order)")
    ap.add_argument("--list", action="store_true", help="print the steps in order and exit")
    ap.add_argument("--determinism", action="store_true",
                    help=f"run the full pipeline twice and write {paths.rel(RECORD)}")
    ap.add_argument("--verbose", action="store_true", help="stream each step's output instead of a tail")
    ns = ap.parse_args(argv)
    steps = parse_steps(ns.steps)
    if ns.list:
        for i, s in enumerate(STEPS, 1):
            print(f"{i} {s}  ({paths.rel(SCRIPTS_DIR / (s + '.py'))})")
        return 0
    if ns.determinism:
        return determinism(steps, verbose=ns.verbose)
    res = run_pipeline(steps, verbose=ns.verbose)
    if any(r["returncode"] != 0 for r in res):
        return 2
    report, _ = verify_manifests(steps)
    bad = {k: v for k, v in report.items() if v.get("manifest_missing") or v["outputs_digest_mismatch"]
           or v["outputs_missing"]}
    print("step manifests verified against disk" if not bad else f"manifest verification FAILED: {bad}")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
