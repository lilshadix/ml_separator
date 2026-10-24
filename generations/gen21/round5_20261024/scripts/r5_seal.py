#!/usr/bin/env python
"""Stage-0 seal of round 5 (CLAUDE_CODE_PROMPT.md sections 1.2, 2.1, 4.6), modelled on round 4's scripts/r4_seal.py.

``manifests/stage0_manifest.json`` + ``manifests/stage0_seal.txt`` (= SHA-256 of the manifest bytes) hold:

* docs: LF-normalised digests of the round-5 Stage-0 documents (DOCS, README.md included).  An APPEND_DOCS document
  may contain the marker ``<!-- APPEND-ONLY BELOW -->`` as a whole line (MARKER_RE); then only the text before that
  line is digested (request status, Stage-1 outcomes, supply-audit reruns).  A marker mentioned inline never truncates.
  A marker line in any other document, or more than one marker line, refuses the seal.
* append_only: LF byte count + SHA-256 of AMENDMENTS.md, RUN_RECORD.md (taken before the SEAL line is appended) and the
  tail from the marker line on of every APPEND_DOCS file; verify checks each as a prefix of the live file, so appending
  verifies and an edit of an earlier line does not.
* code_local: every .py under r5/, scripts/, tests/ of the round-5 folder; code_closure: every .py in the directory of
  every repository module that a fresh interpreter imports when it loads every r5 module, a fixed list of reused
  modules (BASE_MODULES) and every repository module named in an import statement of r5/, scripts/, tests/ (child
  process ``--probe``, so the caller's own imports never change the set; module objects with a relative __file__ are
  skipped), plus round 4's r4_seal.py; copied_sources: every repository path + 64-hex LF SHA-256 recorded in the head
  of a round-5 code file (the copy rule of section 1.2) and whether the source still has that digest; a module
  that declares ``COPIED_FROM = {repository path: 64-hex}`` is read from that declaration (the head regex is the
  fallback for a module without one).
* inputs: every file under INPUT_DIRS (results/stage0/ in full, cache/ext/, cache/frozen/), every
  results/*/ITEM_SPEC.md, manifests/ea_check_caches.json, every cache/{A,B,BR,BQ}_<seed>_<fold>.pkl (item 1.0's fold
  caches) and SUPPLY_LEDGER.csv (LF digest for text, byte digest otherwise).  ``required_inputs`` (the nine item specs,
  the EA-CHECK cache manifest, its 4 families x 5 dev seeds x 5 folds = 100 caches, every Stage-0 deliverable of
  results/stage0/ named in STAGE0_FILES, cache/ext/ext_dopt.pkl and the two cache/frozen/ all-cell records) are
  recorded as MISSING when absent.  A full manifest adds the round-4 prediction parquets, the REF / slot fitted,
  intervention and identity records, the round-4 provenance tables round 5 reads (results/provenance/*.csv|json), the
  gen21 DOPT_Q fold caches (bytes); the bench log D matrix and the BP / BPG dev fold-record digests (computed as r4_seal
  does, and checked against the round-4 manifest).
* design: LF digests of every round5_design_20261023/*.md and of every file under its evidence/, ideas/, ideas2/ and
  probes/ folders (LF digest for text, byte digest for parquet / npz and other binaries); design_code: every *.py of
  the design folder (sources of copies).
* round4_seal: the output of ``round4_20261022/scripts/r4_seal.py --verify`` run as a subprocess (never resealed);
  round3: the round-3 seal string and a direct LF check of every file in next_research_20261019/manifests/
  freeze_lf_digests.json; environment: package versions.

A seal refuses when a Stage-0 document, the supply ledger or a required input is missing, a marker line sits where it
may not, a copied source changed, a scored output already exists before the first seal, the round-4 seal does not
verify to 5bacfe4c..., the round-3 check fails, or the bench / fold digests differ from the round-4 manifest.  A reseal
needs --reason, intact append-only prefixes of the previous seal, and an AMENDMENTS.md entry appended since that seal
that names the previous seal prefix and every scored output then present (its path or its directory followed by '/');
the old manifest and seal are archived as stage0_manifest_<prefix>.json / stage0_seal_<prefix>.txt.  A local seal is
not public preregistration.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/r5_seal.py --verify
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -B -W ignore generations/gen21/round5_20261024/scripts/r5_seal.py --seal --reason "..."
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

R5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R5))
from r5 import paths  # noqa: E402

DOCS = ("PROTOCOL.md", "DATA_CONTRACT.md", "METRICS.md", "CONFIRMATION_PACKAGE.md", "DECISION_LOG.md", "REGISTRY.csv",
        "DATA_REQUEST.md", "EVIDENCE_INVENTORY.md", "README.md", "RESUME.md")
APPEND_DOCS = ("DECISION_LOG.md", "CONFIRMATION_PACKAGE.md", "README.md", "RESUME.md")   # may hold one marker line
APPEND_FILES = ("AMENDMENTS.md", "RUN_RECORD.md")                                     # append-only in full
LEDGER = "SUPPLY_LEDGER.csv"
MARKER = b"<!-- APPEND-ONLY BELOW -->"
MARKER_RE = re.compile(rb"(?m)^<!-- APPEND-ONLY BELOW -->[ \t]*$")                    # a whole line only
CODE_DIRS = ("r5", "scripts", "tests")
INPUT_DIRS = ("results/stage0", "cache/ext", "cache/frozen")      # Stage-0 inputs (beta_frozen's all-cell records)
ITEM_SPEC_GLOB = "results/*/ITEM_SPEC.md"                         # every item's frozen design choices (DECISION_LOG D8)
ITEMS = ("EA_CHECK", "ECT", "ECT_M03", "ELAB", "WINDOW", "EDEC", "ED", "EW", "EKX")
EA_MANIFEST = "manifests/ea_check_caches.json"                   # item 1.0's cache records (the only sealed manifest)
EA_FAMILIES = ("A", "B", "BR", "BQ")                              # cache/<family>_<seed>_<fold>.pkl (g21.cache.build_fold)
EA_CACHE = re.compile(r"(?:A|B|BR|BQ)_\d+_\d+\.pkl")
N_FOLDS = 5
ECT_SIZE_CHECK = "results/stage0/size_gate_ect_chemotype.json"    # C-ECT fallback (ii) fresh size check (DECISION_LOG D3)
STAGE0_FILES = ("beta_frozen.json", "ext_dopt.json", "size_gate.json", "size_gate.csv", "power_worlds.json",
                "power_worlds.csv", "power_cgd_beta_sens.json", "power_cgd_beta_sens.csv", "lab_key.csv",
                "lab_key_links.csv", "mechanism_class_map.csv", "supply_audit.json", "supply_reach.json",
                "dupscan_summary.json", "dupscan_links.csv", "dupscan_nearcopy.csv", "dupscan_quant.csv",
                "ect_population.json", "ect_population_cells.csv", "ect_sibling_links.csv", "size_gate_ect_chemotype.csv")
FROZEN_OBJECTS = ("cache/ext/ext_dopt.pkl", "cache/frozen/fold_all_BPG_104729_0.pkl", "cache/frozen/fold_all_BP_104729_0.pkl")
SCORED_KEEP = re.compile(r"(ITEM_SPEC\.md|SMOKE_COUNTS_[A-Za-z0-9_]+\.json)")         # not scored outputs
LEAK_KEEP = re.compile(r"(cache_poison_EA_CHECK\.json)")
DESIGN_SUBDIRS = ("evidence", "ideas", "ideas2", "probes")
TEXT_EXT = {".md", ".csv", ".json", ".txt", ".py", ".tsv", ".yaml", ".yml", ".log", ".tex", ".out"}
BASE_MODULES = ("r5.paths", "r4.paths", "r4.refs", "r4.metrics", "r3.metrics", "r3.configs", "g21.cache", "g21.evaluate",
                "gen20k.data", "gen20k.design", "gen20k.splits", "gen20k.inference", "gen13sep.wildcluster")
R4_SEAL = "5bacfe4cdcfc8d63171c46497e06b80605cc7fb1e0e7531c5fd81daadb1bbaa5"
R3_SEAL = "93d3839a256c2185d3cb01f819881cf862d21192d968cc056b9c6c3213b39e26"
LIGHT_KEYS = ("docs", "code_local", "inputs_local", "copied_sources", "append_only")
FULL_KEYS = ("round4_seal", "round3", "code_closure", "environment", "inputs_bench", "inputs_round4", "design", "design_code")
STAGE = "stage 0 (round 5): M35 / M34 / CP-13 conventions, closure ledger, pre-slot gate, registry; before any round-5 score"
PROBE_TAG = "R5PROBE "
HEX64 = re.compile(r"(?i)sha-?256\w{0,12}[\s:=(\"']{0,6}\b([0-9a-f]{64})\b")   # 'SHA-256 <hex>', 'sha256_lf=<hex>'
PYPATH = re.compile(r"((?:[\w.\-]+/)+[\w.\-]+\.py)\b")


# ------------------------------------------------------------------------------------------------------------
# digests
# ------------------------------------------------------------------------------------------------------------
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _rel(p: Path, base: Path | None = None) -> str:
    p = Path(p).resolve()
    for b in ((base,) if base is not None else ()) + (paths.ROOT,):
        try:
            return str(p.relative_to(Path(b).resolve())).replace("\\", "/")
        except ValueError:
            continue
    return str(p).replace("\\", "/")


def _lf(p: Path) -> bytes:
    return Path(p).read_bytes().replace(b"\r\n", b"\n")


def marker_lines(b: bytes) -> list:
    """Start offsets of every whole-line append-only marker in LF bytes (an inline mention is not a marker)."""
    return [m.start() for m in MARKER_RE.finditer(b)]


def doc_digest(p: Path) -> tuple[str, bool]:
    """LF-normalised digest of a document; with a whole-line append-only marker, of the bytes before the first one."""
    b = _lf(p)
    ms = marker_lines(b)
    return (_sha(b[:ms[0]]), True) if ms else (_sha(b), False)


def file_digest(p: Path) -> str:
    """LF-normalised digest for text files, byte digest otherwise."""
    return paths.sha256_text_lf(p) if Path(p).suffix.lower() in TEXT_EXT else paths.sha256_file(p)


def _files(d: Path, pattern: str = "*") -> list[Path]:
    return sorted(p for p in d.rglob(pattern) if p.is_file() and "__pycache__" not in p.parts) if d.is_dir() else []


def docs_digests(base: Path) -> dict:
    out = {}
    for d in DOCS:
        p = base / d
        if p.exists():
            h, m = doc_digest(p)
            out[d] = {"sha256_lf": h, "append_only_marker": m, "marker_lines": len(marker_lines(_lf(p)))}
        else:
            out[d] = "MISSING"
    return out


def append_only_state(base: Path) -> dict:
    """LF byte count and SHA-256 of every append-only text: AMENDMENTS.md and RUN_RECORD.md in full, and the tail from
    the first marker line on of every APPEND_DOCS file (an empty tail when it has no marker)."""
    out = {}
    for name in APPEND_FILES:
        p = Path(base) / name
        if p.exists():
            b = _lf(p)
            out[name] = {"lf_bytes": len(b), "sha256_lf": _sha(b)}
        else:
            out[name] = "MISSING"
    for d in APPEND_DOCS:
        p = Path(base) / d
        if p.exists():
            b = _lf(p)
            ms = marker_lines(b)
            tail = b[ms[0]:] if ms else b""
            out[d + "#tail"] = {"lf_bytes": len(tail), "sha256_lf": _sha(tail)}
    return out


def append_only_problems(sealed: dict, base: Path) -> list:
    """Entries of a sealed append_only record whose live text no longer starts with the sealed bytes."""
    bad = []
    for k, rec in (sealed or {}).items():
        if rec == "MISSING":
            continue
        name, _, tail = k.partition("#")
        p = Path(base) / name
        if not p.exists():
            bad.append(k)
            continue
        b = _lf(p)
        if tail:
            ms = marker_lines(b)
            b = b[ms[0]:] if ms else b""
        n = int(rec["lf_bytes"])
        if len(b) < n or _sha(b[:n]) != rec["sha256_lf"]:
            bad.append(k)
    return bad


def scored_outputs(base: Path) -> list:
    """Every file under results/<ITEM>/ and results/leakage/ that a scoring run writes (ITEM_SPEC.md, SMOKE_COUNTS_*,
    the pre-seal cache-poisoning report and post-run checks excluded), relative to the round-5 folder."""
    base = Path(base)
    out = []
    for it in ITEMS:
        out += [_rel(q, base) for q in _files(base / "results" / it) if not SCORED_KEEP.fullmatch(q.name)]
    out += [_rel(q, base) for q in _files(base / "results" / "leakage")
            if not (LEAK_KEEP.fullmatch(q.name) or q.name.startswith("postrun_"))]
    return sorted(out)


def local_code(base: Path) -> dict:
    return {_rel(p, base): paths.sha256_text_lf(p) for sub in CODE_DIRS for p in _files(base / sub, "*.py")}


def required_inputs() -> list[str]:
    """Inputs a seal needs by name (relative paths): every item's ITEM_SPEC.md, the EA-CHECK cache manifest and its
    4 families x 5 dev seeds x 5 folds = 100 fold caches, the C-ECT chemotype-unit size check (D3), every other
    Stage-0 deliverable of results/stage0/ (STAGE0_FILES) and the frozen external objects (FROZEN_OBJECTS)."""
    req = [f"results/{it}/ITEM_SPEC.md" for it in ITEMS] + [EA_MANIFEST]
    req += [f"cache/{fam}_{s}_{k}.pkl" for fam in EA_FAMILIES for s in paths.DEV_SEEDS for k in range(N_FOLDS)]
    return req + [ECT_SIZE_CHECK] + [f"results/stage0/{f}" for f in STAGE0_FILES] + list(FROZEN_OBJECTS)


def local_inputs(base: Path) -> dict:
    """Digests of every round-5 input: INPUT_DIRS in full, every results/*/ITEM_SPEC.md, the EA-CHECK cache manifest,
    every cache/{A,B,BR,BQ}_<seed>_<fold>.pkl present (a cache added after the seal changes the set) and the supply
    ledger; a required input that is absent is recorded as MISSING."""
    base = Path(base)
    out = {_rel(p, base): file_digest(p) for sub in INPUT_DIRS for p in _files(base / sub)}
    out.update({_rel(p, base): file_digest(p) for p in sorted(base.glob(ITEM_SPEC_GLOB)) if p.is_file()})
    out.update({_rel(p, base): file_digest(p) for p in sorted((base / "cache").glob("*.pkl"))
                if p.is_file() and EA_CACHE.fullmatch(p.name)})
    for rel in required_inputs():
        if rel not in out:
            out[rel] = file_digest(base / rel) if (base / rel).is_file() else "MISSING"
    out[LEDGER] = paths.sha256_text_lf(base / LEDGER) if (base / LEDGER).exists() else "MISSING"
    return dict(sorted(out.items()))


def _resolve_source(s: str, base: Path) -> Path | None:
    roots = (paths.ROOT, paths.G21, paths.G21.parent, paths.DESIGN, paths.DESIGN / "probes" / "scripts",
             paths.DESIGN / "ideas2" / "scripts", paths.R4DIR, paths.R3DIR, base)
    for root in roots:
        p = root / s
        if p.is_file():
            return p
    return None


def declared_copies(f: Path) -> dict | None:
    """The module-level ``COPIED_FROM = {path: hex}`` literal of a code file (read with ast.literal_eval), or None."""
    try:
        tree = ast.parse(Path(f).read_bytes().decode("utf-8"), filename=str(f))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "COPIED_FROM" for t in node.targets):
            return {str(k): str(v) for k, v in ast.literal_eval(node.value).items()}
    return None


def copied_sources(base: Path, head_lines: int = 60) -> dict:
    """Copy records of every round-5 code file.  A module that declares ``COPIED_FROM`` (repository-relative path ->
    64-hex LF SHA-256) is read from that declaration; otherwise every (path to a .py file, 'SHA-256 <64 hex>') pair in
    its head: the first digest within 300 characters after a path (and before the next path) is the source's recorded
    LF SHA-256.  ``match`` says whether the source (resolved against the repository, gen21, the design folder and its
    script folders, round 4 / 3) still has it.  An unresolved source counts as a mismatch."""
    out = {}
    for sub in CODE_DIRS:
        for f in _files(base / sub, "*.py"):
            dec = declared_copies(f)
            if dec is not None:
                recs = []
                for src_rel, h in dec.items():
                    src = _resolve_source(src_rel, base)
                    live = paths.sha256_text_lf(src) if src is not None else None
                    recs.append({"source": src_rel, "resolved": _rel(src) if src is not None else None,
                                 "recorded": h.lower(), "live": live, "match": live == h.lower(), "declared": True})
                if recs:
                    out[_rel(f, base)] = recs
                continue
            head = "\n".join(f.read_bytes().decode("utf-8", "replace").replace("\r\n", "\n").split("\n")[:head_lines])
            ms = list(PYPATH.finditer(head))
            recs = []
            for i, m in enumerate(ms):
                stop = ms[i + 1].start() if i + 1 < len(ms) else len(head)
                h = HEX64.search(head, m.end(), min(stop, m.end() + 300))
                if not h:
                    continue
                src = _resolve_source(m.group(1), base)
                live = paths.sha256_text_lf(src) if src is not None else None
                recs.append({"source": m.group(1), "resolved": _rel(src) if src is not None else None,
                             "recorded": h.group(1).lower(), "live": live, "match": live == h.group(1).lower()})
            if recs:
                out[_rel(f, base)] = recs
    return out


def round4_inputs() -> dict:
    out = {"predictions": {p.name: paths.sha256_file(p) for p in sorted(paths.R4PRED.glob("*.parquet"))}, "records": {},
           "gen21_fold_caches": {p.name: paths.sha256_file(p) for p in sorted(paths.CACHE_R4.glob("*.pkl"))},
           "provenance": {_rel(p): paths.sha256_text_lf(p) for p in sorted((paths.R4RES / "provenance").glob("*"))
                          if p.is_file() and p.suffix in (".csv", ".json")}}
    cids = sorted({p.stem.rsplit("_", 1)[0] for p in paths.R4PRED.glob("*.parquet")})
    for cid in cids:
        for p in _files(paths.R4RES / cid, "*.json"):
            if p.name.split("_")[0] in ("fitted", "intervention", "identity"):
                out["records"][_rel(p)] = paths.sha256_text_lf(p)
    return out


def design_digests() -> tuple[dict, dict]:
    D = paths.DESIGN
    des = {_rel(p): paths.sha256_text_lf(p) for p in sorted(D.glob("*.md"))}
    for sub in DESIGN_SUBDIRS:                       # every file: LF digest for text, byte digest for binaries
        des.update({_rel(p): file_digest(p) for p in _files(D / sub)})
    return des, {_rel(p): paths.sha256_text_lf(p) for p in _files(D, "*.py")}


# ------------------------------------------------------------------------------------------------------------
# earlier seals
# ------------------------------------------------------------------------------------------------------------
def _env() -> dict:
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}


def round4_seal() -> dict:
    """Run round 4's own verifier (read-only) and record its output; round 4 is never resealed."""
    r = subprocess.run([sys.executable, "-B", "-W", "ignore", str(paths.R4DIR / "scripts" / "r4_seal.py"), "--verify"],
                       cwd=str(paths.ROOT), env=_env(), capture_output=True, text=True, encoding="utf-8", timeout=3600)
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    return {"returncode": r.returncode, "output": last,
            "verified": bool(r.returncode == 0 and last.startswith("VERIFIED") and R4_SEAL in last),
            "stored_seal": (paths.R4DIR / "manifests" / "stage0_seal.txt").read_text(encoding="utf-8").split()[0]}


def round3_check() -> dict:
    """Round-3 seal string plus a direct LF check of every file named in freeze_lf_digests.json."""
    man = paths.R3DIR / "manifests"
    lf = json.loads((man / "freeze_lf_digests.json").read_bytes().decode("utf-8"))
    seal = (man / "freeze_seal.txt").read_text(encoding="utf-8").split()[0]
    fm = lf["freeze_manifest.json"]
    bad = [f for f, h in sorted(lf["code_sha256_lf_normalised"].items())
           if not (paths.ROOT / f).is_file() or paths.sha256_text_lf(paths.ROOT / f) != h]
    rec = {"seal": seal, "seal_is_expected": seal == R3_SEAL == fm["sealed_sha256"],
           "freeze_lf_digests_sha256_lf": paths.sha256_text_lf(man / "freeze_lf_digests.json"),
           "freeze_manifest_lf_ok": paths.sha256_text_lf(man / "freeze_manifest.json") == fm["sha256_lf_normalised"],
           "code_files": len(lf["code_sha256_lf_normalised"]), "code_files_mismatched": bad}
    rec["ok"] = bool(rec["seal_is_expected"] and rec["freeze_manifest_lf_ok"] and not bad)
    return rec


# ------------------------------------------------------------------------------------------------------------
# the probe (child process): import closure, bench / fold digests, environment
# ------------------------------------------------------------------------------------------------------------
def _in_repo(p: Path) -> bool:
    try:
        p.resolve().relative_to(paths.ROOT)
    except ValueError:
        return False
    return ".venv" not in p.parts


def _ast_imports(files) -> set[str]:
    names = set()
    for f in files:
        tree = ast.parse(f.read_bytes().decode("utf-8"), filename=str(f))
        pkg = list(f.parent.relative_to(R5).parts)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level - 1 > len(pkg) or not pkg:
                        continue
                    mod = ".".join(pkg[:len(pkg) - (node.level - 1)] + ([node.module] if node.module else []))
                else:
                    mod = node.module
                names.add(mod)
                names.update(f"{mod}.{a.name}" for a in node.names if a.name != "*")
    return names


def _import_repo(name: str) -> bool:
    """Import ``name`` when its top-level package lives in the repository (outside .venv); returns True if imported."""
    try:
        spec = importlib.util.find_spec(name.split(".")[0])
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    origin = spec.origin or next(iter(spec.submodule_search_locations or []), None)
    if not origin or origin in ("built-in", "frozen") or not _in_repo(Path(origin)):
        return False
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name == name:                    # 'from pkg import attr' where attr is not a module
            return False
        raise
    return True


def probe() -> dict:
    import numpy as np
    for m in BASE_MODULES:
        importlib.import_module(m)
    for f in _files(R5 / "r5", "*.py"):
        if f.stem != "__init__":
            importlib.import_module("r5." + f.stem)
    files = [f for sub in CODE_DIRS for f in _files(R5 / sub, "*.py")]
    imported = sorted(n for n in _ast_imports(files) if _import_repo(n))
    paths.redirect_inner_cache()
    dirs = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not Path(f).is_absolute():          # e.g. torch.ops ('_ops.py'): never resolve against the cwd
            continue
        p = Path(f).resolve()
        if p.suffix == ".py" and _in_repo(p):
            dirs.add(p.parent)
    code = {p.resolve() for d in dirs for p in d.glob("*.py")}
    code = {p for p in code if not str(p).startswith(str(R5.resolve()))}      # round-5 files are in code_local
    code.add((paths.R4DIR / "scripts" / "r4_seal.py").resolve())                # executed with --verify
    from gen20k import data as D
    from gen20k.splits import fold_record, folds_for
    bench = D.load_bench()
    inputs = {"logD_matrix_sha256": _sha(np.ascontiguousarray(np.nan_to_num(bench.Y, nan=-999.0)).tobytes()),
              "n_cells": int(bench.Y.shape[0]), "folds": {}}
    for d in paths.DESIGNS:
        fr = fold_record(bench.frame, folds_for(bench.frame, d, seeds=list(paths.DEV_SEEDS)))
        inputs["folds"][d] = {"seeds": list(paths.DEV_SEEDS), "n_rows": int(len(fr)),
                              "sha256": _sha(fr.to_csv(index=False, lineterminator="\n").encode())}
    return {"code": {_rel(p): paths.sha256_text_lf(p) for p in sorted(code)}, "imported_from_ast": imported,
            "inputs": inputs, "environment": environment()}


def environment() -> dict:
    env = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "rdkit", "catboost", "pyarrow"):
        try:
            env[name] = importlib.import_module(name).__version__
        except Exception:                     # noqa: BLE001  (recorded, compared on verify)
            env[name] = "unavailable"
    return env


def run_probe() -> dict:
    r = subprocess.run([sys.executable, "-B", "-W", "ignore", str(Path(__file__).resolve()), "--probe"],
                       cwd=str(paths.ROOT), env=_env(), capture_output=True, text=True, encoding="utf-8", timeout=3600)
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith(PROBE_TAG)), None)
    if r.returncode != 0 or line is None:
        raise RuntimeError(f"probe failed (exit {r.returncode}): {r.stderr[-2000:]}")
    return json.loads(line[len(PROBE_TAG):])


# ------------------------------------------------------------------------------------------------------------
# build / verify / seal
# ------------------------------------------------------------------------------------------------------------
def build(base: Path = R5, *, full: bool = True) -> dict:
    """The manifest without its seal-time fields.  ``full=False`` (tests on a temporary copy) records only the
    round-5 folder's own documents, code, inputs and copy digests."""
    base = Path(base)
    man = {"stage": STAGE, "full": bool(full)}
    if full:
        man["round4_seal"] = round4_seal()              # first, before anything heavy
        man["round3"] = round3_check()
        pr = run_probe()
        man["code_closure"], man["environment"], man["inputs_bench"] = pr["code"], pr["environment"], pr["inputs"]
        man["imported_from_ast"] = pr["imported_from_ast"]                       # informational
        r4m = json.loads((paths.R4DIR / "manifests" / "stage0_manifest.json").read_bytes().decode("utf-8"))["inputs"]
        man["inputs_bench"]["matches_round4_manifest"] = bool(
            r4m["logD_matrix_sha256"] == pr["inputs"]["logD_matrix_sha256"]
            and all(r4m["folds"][d]["sha256"] == pr["inputs"]["folds"][d]["sha256"] for d in paths.DESIGNS))
        man["inputs_round4"] = round4_inputs()
        man["design"], man["design_code"] = design_digests()
    man["docs"] = docs_digests(base)
    man["append_only"] = append_only_state(base)
    man["code_local"] = local_code(base)
    man["inputs_local"] = local_inputs(base)
    man["copied_sources"] = copied_sources(base)
    return man


def _paths(base: Path) -> tuple[Path, Path]:
    return Path(base) / "manifests" / "stage0_manifest.json", Path(base) / "manifests" / "stage0_seal.txt"


def _diff(a, b) -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return []


def verify(base: Path = R5, *, full: bool = True) -> tuple[bool, str, str]:
    manifest, sealf = _paths(base)
    if not sealf.exists() or not manifest.exists():
        return False, "no round-5 stage-0 seal", ""
    seal = sealf.read_text(encoding="utf-8").split()[0]
    if _sha(manifest.read_bytes()) != seal:
        return False, "manifest digest != seal", seal
    sealed = json.loads(manifest.read_bytes().decode("utf-8"))
    if full and not sealed.get("full"):
        return False, "the sealed manifest is a light (test-only) manifest", seal
    live = build(base, full=full)
    for key in LIGHT_KEYS + (FULL_KEYS if full else ()):
        if key == "append_only":                 # prefix check: appending verifies, editing an earlier line does not
            if "append_only" not in sealed:
                return False, "append_only missing from the sealed manifest", seal
            bad = append_only_problems(sealed["append_only"], base)
            if bad:
                return False, f"append_only changed: {bad[:8]}", seal
            continue
        if live.get(key) != sealed.get(key):
            d = _diff(live.get(key), sealed.get(key))
            return False, f"{key} changed" + (f": {d[:8]}" if d else ""), seal
    return True, "ok", seal


def refusals(man: dict, base: Path | None = None, supersedes: str | None = None) -> list:
    out = [f"missing document {d}" for d, v in man["docs"].items() if v == "MISSING"]
    for d, v in man["docs"].items():
        if v == "MISSING":
            continue
        if d not in APPEND_DOCS and v.get("marker_lines", 0) > 0:
            out.append(f"marker line in a fully sealed document: {d}")
        if v.get("marker_lines", 0) > 1:
            out.append(f"more than one marker line in {d}")
    if base is not None and supersedes is None:
        scored = scored_outputs(base)
        if scored:
            out.append(f"scored outputs exist before the first seal: {scored[:8]}")
    if man["inputs_local"].get(LEDGER) == "MISSING":
        out.append(f"missing {LEDGER}")
    out += [f"missing input {k}" for k, v in man["inputs_local"].items() if v == "MISSING" and k != LEDGER]
    out += [f"copied source changed or unresolved: {f} <- {r['source']}" for f, rs in man["copied_sources"].items()
            for r in rs if not r["match"]]
    if man["full"]:
        if not man["round4_seal"]["verified"]:
            out.append(f"round-4 seal does not verify: {man['round4_seal']['output']!r}")
        if not man["round3"]["ok"]:
            out.append(f"round-3 check failed: {man['round3']}")
        if not man["inputs_bench"]["matches_round4_manifest"]:
            out.append("bench / fold digests differ from the round-4 manifest")
    return out


def seal(base: Path = R5, reason: str = "", *, full: bool = True, stage: str | None = None) -> str:
    base = Path(base)
    manifest, sealf = _paths(base)
    supersedes = None
    if sealf.exists():
        old = sealf.read_text(encoding="utf-8").split()[0]
        amp = base / "AMENDMENTS.md"
        amend = _lf(amp) if amp.exists() else b""
        prev = json.loads(manifest.read_bytes().decode("utf-8")).get("append_only", {}) if manifest.exists() else {}
        bad_prev = append_only_problems(prev, base)
        if bad_prev:
            raise SystemExit(f"a round-5 seal exists and its append-only texts were edited, not appended: {bad_prev}")
        n_old = int((prev.get("AMENDMENTS.md") or {}).get("lf_bytes", 0)) if prev.get("AMENDMENTS.md") != "MISSING" else 0
        new_part = amend[n_old:].decode("utf-8", "replace")
        if not reason or old[:12] not in new_part:
            raise SystemExit("a round-5 seal exists: resealing needs --reason and an AMENDMENTS.md entry, appended since "
                             f"that seal, that names the old seal ({old[:12]}) and lists every result seen")
        unlisted = [q for q in scored_outputs(base)
                    if q not in new_part and (q.rsplit("/", 1)[0] + "/") not in new_part]
        if unlisted:
            raise SystemExit(f"resealing needs every scored output named in the new AMENDMENTS.md entry: {unlisted[:8]}")
        for p in (manifest.parent / f"stage0_manifest_{old[:12]}.json", manifest.parent / f"stage0_seal_{old[:12]}.txt"):
            if p.exists():
                raise SystemExit(f"archive {p.name} already exists; refusing to overwrite it")
        supersedes = old
    man = build(base, full=full)                    # build first: a failing build leaves the current seal in place
    bad = refusals(man, base, supersedes)
    if bad:
        raise SystemExit("seal refused:\n  " + "\n  ".join(bad))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if supersedes:
        manifest.replace(manifest.parent / f"stage0_manifest_{supersedes[:12]}.json")     # archived, never overwritten
        sealf.replace(manifest.parent / f"stage0_seal_{supersedes[:12]}.txt")
    if stage:
        man["stage"] = stage
    man["sealed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    man["reason"] = reason
    man["supersedes"] = supersedes
    manifest.write_bytes(json.dumps(man, indent=1, sort_keys=True).encode("utf-8"))
    s = _sha(manifest.read_bytes())
    sealf.write_bytes(f"{s}  stage0_manifest.json\n".encode("utf-8"))
    n_design = len(man.get("design", {}))
    line = (f"- {man['sealed_utc']} SEAL {s} ({len(man['code_local'])} round-5 code files, "
            f"{len(man.get('code_closure', {}))} repository code files, {len(man['docs'])} documents, "
            f"{len(man['inputs_local'])} local inputs, {n_design} design files; supersedes {supersedes}; local only, not "
            f"public preregistration). {reason}\n")
    with open(base / "RUN_RECORD.md", "ab") as fh:
        fh.write(line.encode("utf-8"))
    return s


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seal", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--probe", action="store_true", help="internal: child process that reports the import closure")
    ap.add_argument("--reason", default="")
    ap.add_argument("--stage", default=None, help="stage text stored in the manifest (informational)")
    a = ap.parse_args()
    if a.probe:
        print(PROBE_TAG + json.dumps(probe(), sort_keys=True))
        return 0
    if a.verify:
        ok, msg, s = verify()
        print(("VERIFIED " if ok else "NOT VERIFIED ") + msg, s)
        return 0 if ok else 1
    if not a.reason:
        raise SystemExit("--seal needs --reason")
    s = seal(reason=a.reason, stage=a.stage)
    man = json.loads(_paths(R5)[0].read_bytes().decode("utf-8"))
    print("sealed", s)
    print("wrote", _rel(_paths(R5)[0]), "and", _rel(_paths(R5)[1]), "; appended a SEAL line to", _rel(R5 / "RUN_RECORD.md"))
    print(f"  {len(man['code_local'])} round-5 code files, {len(man['code_closure'])} repository code files, "
          f"{len(man['inputs_local'])} local inputs, {len(man['design'])} design files; round 4: "
          f"{man['round4_seal']['output']}; round 3 ok: {man['round3']['ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
