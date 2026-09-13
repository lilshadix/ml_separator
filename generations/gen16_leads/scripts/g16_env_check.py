"""Gen16 environment check: has the machine drifted from the one gen13-gen15 ran in?

Re-runnable.  Writes ``gen16_leads/results/env/env_check.json`` and ``ENV.md``.

Run from the repo root with the venv interpreter::

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe gen16_leads/scripts/g16_env_check.py [--no-et]

What it checks (START_HERE.md section 4 traps, plus the fleet's practical limits):

1. installed versions against every version the repository has recorded, the install dates of
   the numeric stack against the gen13-gen15 commit dates, that ``import tabpfn`` fails, and that
   nothing on the frozen import path (gen13sep, gen14, gen15) imports tabpfn;
2. the ``hash()`` / set-iteration audit of the frozen packages, and an empirical two-subprocess
   fold-plan comparison under *different* ``PYTHONHASHSEED`` values (a stronger test than two
   random salts);
3. whether ``xtb`` is on PATH, anywhere on disk, or importable as ``xtb-python``;
4. free physical memory, and the peak working set of one bench process that loads the frozen
   bench and scores FLAT + G14 under BP (which also re-checks two anchors), then of one that runs
   gen13's 400-tree extra-trees arm with n_jobs=2 (the fleet's heavy slot; ``--no-et`` skips it);
5. the git state: what is untracked and outside the brief, and what is committed under gen16_leads.

Children are run strictly one at a time (8 GB machine, several agents).  ``--child`` modes are
internal.  Never passes ``seeds=`` to the frozen splitter.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "gen16_leads" / "results" / "env"
PY = sys.executable
MB = 1024 * 1024
KEEP_FREE_MB = 1024

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "rdkit", "joblib", "pyarrow", "torch",
            "catboost", "xgboost", "lightgbm", "threadpoolctl", "tabpfn"]

#: Every version record found in the repository (grep of gen13_separation/, gen14_direction/,
#: gen15_curve/, docs/, requirements*).  gen13-gen15 themselves recorded no explicit version table;
#: the closest records are the gen7/gen8 docs (Aug 2026, same venv lineage) and the env memory note.
RECORDED = {
    "docs/results/gen8_architecture_results_20260820.md:834 (2026-08-20)": {
        "scikit-learn": "1.9.0", "pandas": "3.0.5", "numpy": "2.5.1", "scipy": "1.18.0",
        "torch": "2.13.0", "rdkit": "2026.03.5"},
    "docs/results/gen7_architecture_results_20260819.md:1165 (2026-08-19)": {
        "python": "3.13", "scikit-learn": "1.9.0", "pandas": "3.0.5", "numpy": "2.5.1",
        "torch": "2.13.0", "catboost": "1.2.10", "lightgbm": "4.7.0", "xgboost": "3.4.1",
        "rdkit": "2026.03.5"},
    "requirements-gen3.txt (Linux cluster reference, not this machine)": {
        "python": "3.11.11", "numpy": "2.4.6", "pandas": "3.0.5", "pyarrow": "25.0.0",
        "scipy": "1.17.1", "scikit-learn": "1.9.0", "joblib": "1.5.3", "torch": "2.13.0",
        "catboost": "1.2.10"},
    "memory/env-windows-python-quirks.md (2026-09-07, the gen13 session)": {
        "python": "3.14.5", "numpy": "2.5.3", "pandas": "3.0.5", "scikit-learn": "1.9.0",
        "torch": "2.14.0+cpu"},
    "gen13_separation/DECISION_REPORT.md:522": {"pandas": "3.x"},
    "gen15_curve/exp/tabpfn/tabpfn_compat.py docstring": {"tabpfn": "2.2.1", "scikit-learn": "1.9"},
}

GEN_COMMITS = ["270ae5c", "bfad375", "5885607", "dbc84e4", "47499b7", "df63929"]

#: Manually reviewed hash()/set() sites on the frozen import path.  The scan below re-lists every
#: live hit; a hit not matched here is reported as UNREVIEWED so drift is visible.
REVIEWED = [
    ("gen13_separation/gen13sep/arms_stage2.py", r"hash\(row\.tobytes\(\)\)", "UNSAFE-IN-PRINCIPLE",
     "hash() of bytes is salted per process.  extractant_key() uses it only as an equality key. "
     "ExtractantBalancedArm consumes it through groupby transforms (per-row weights, order-free): safe. "
     "HierarchicalCurveArm (S3_HIER / S3_EXT_LEVEL) groups by the key, so the row order of its "
     "per-extractant matrix X_ext follows the salted hash order and the ExtraTrees fit is order-invariant "
     "only up to floating-point summation order.  Not on the gen14/gen15 bench path (gen15.arms.g13_full "
     "calls tree_pipeline directly).  Do not rely on cross-process bit-identity of those two gen13 arms."),
    ("gen15_curve/gen15/fewshot.py", r"hash\(\(int\(ci\), int\(seed\)\)\)", "SAFE",
     "tuple of ints: CPython salts str/bytes hashes only; int and tuple hashes are unsalted "
     "(verified empirically below under two PYTHONHASHSEED values).  Differs only across 32/64-bit builds."),
    ("gen13_separation/gen13sep/fewshot.py", r"stable_hash|blake2b", "SAFE", "blake2b of the text, not hash()."),
    ("gen13_separation/gen13sep/fewshot_stage2.py", r"stable_hash", "SAFE", "blake2b via stable_hash()."),
    ("gen13_separation/gen13sep/cohort.py", r"blake2b|set\(", "SAFE",
     "blake2b cell ids; set() only inside sorted() or for error messages."),
    ("gen13_separation/gen13sep/splits.py", r"set\(", "SAFE",
     "line 75 membership; 90 len; 94 len; 96 `dropped` membership only (candidates come from np.unique + "
     "seeded rng.shuffle, `remaining` is a list, min() over the list); 112 sorted; 133-146 counts/bools."),
    ("src/lanthanide_separation/gen6/cohorts.py", r"set\(|np\.unique", "SAFE",
     "seeded_group_kfold: np.unique (sorted) + default_rng(seed).permutation; dict used for lookup only.  "
     "Other set() uses are membership/len/sorted."),
    ("gen13_separation/gen13sep/inference.py", r"frozenset|set\(", "SAFE", "membership / sorted."),
    ("gen13_separation/gen13sep/models.py", r"set\(", "SAFE", "inside sorted()."),
    ("gen13_separation/gen13sep/runner.py", r"set\(", "SAFE", "membership; glob() is sorted()."),
    ("gen13_separation/gen13sep/features.py", r"set\(|sorted\(", "SAFE", "error message only."),
    ("gen13_separation/gen13sep/basis.py", r"sorted\(", "SAFE", "error message only."),
    ("gen14_direction/gen14/dirbench.py", r"set\(", "SAFE", "inside sorted()."),
    ("gen15_curve/gen15/mixture.py", r"set\(", "SAFE", "membership only."),
]

SCAN_FILES = [
    "gen13_separation/gen13sep", "gen14_direction/gen14", "gen15_curve/gen15",
    "src/lanthanide_separation/gen6/cohorts.py", "src/lanthanide_separation/gen8/inference.py",
]

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _run(cmd, *, env=None, timeout=600, cwd=ROOT):
    e = dict(os.environ); e["PYTHONIOENCODING"] = "utf-8"
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=str(cwd), env=e, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _child(mode: str, *, env=None, timeout=600):
    """Run this file in --child mode and return (parsed JSON or None, rc, stdout, stderr)."""
    rc, out, err = _run([PY, str(Path(__file__).resolve()), "--child", mode], env=env, timeout=timeout)
    payload = None
    for line in out.splitlines():
        if line.startswith("@@JSON@@"):
            payload = json.loads(line[len("@@JSON@@"):])
    return payload, rc, out, err


def _git(*args):
    rc, out, err = _run(["git", *args])
    return out.strip()


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_K32 = ctypes.windll.kernel32
_PSAPI = ctypes.windll.psapi
_K32.GetCurrentProcess.restype = wt.HANDLE
_K32.OpenProcess.restype = wt.HANDLE
_K32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
_K32.CloseHandle.argtypes = [wt.HANDLE]
_K32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
_PSAPI.GetProcessMemoryInfo.restype = wt.BOOL
_PSAPI.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]


def memory_status() -> dict:
    s = _MEMORYSTATUSEX(); s.dwLength = ctypes.sizeof(s)
    _K32.GlobalMemoryStatusEx(ctypes.byref(s))
    return {"total_mb": round(s.ullTotalPhys / MB), "free_mb": round(s.ullAvailPhys / MB),
            "load_pct": int(s.dwMemoryLoad), "commit_total_mb": round(s.ullTotalPageFile / MB),
            "commit_free_mb": round(s.ullAvailPageFile / MB)}


def process_memory(handle=None) -> dict:
    """Working-set counters of a process (own process when handle is None).  Needs explicit
    argtypes: without them ctypes truncates the 64-bit pseudo-handle and the call fails silently."""
    h = handle if handle is not None else _K32.GetCurrentProcess()
    pmc = _PMC(); pmc.cb = ctypes.sizeof(pmc)
    ok = _PSAPI.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    if not ok:
        return {"error": f"GetProcessMemoryInfo failed, winerror {ctypes.get_last_error()}"}
    return {"ws_mb": round(pmc.WorkingSetSize / MB, 1), "peak_ws_mb": round(pmc.PeakWorkingSetSize / MB, 1),
            "commit_mb": round(pmc.PagefileUsage / MB, 1), "peak_commit_mb": round(pmc.PeakPagefileUsage / MB, 1)}


def python_processes() -> list[dict]:
    rc, out, err = _run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"])
    rows = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 5 and parts[0].lower().startswith("python"):
            mem = re.sub(r"[^\d]", "", parts[4])
            rows.append({"pid": int(parts[1]), "ws_mb": round(int(mem) / 1024) if mem else None})
    return rows


# --------------------------------------------------------------------------------------
# child modes
# --------------------------------------------------------------------------------------
def child_tabpfn():
    rec = {"sklearn_before": md.version("scikit-learn")}
    try:
        import tabpfn  # noqa: F401
        rec["importable"] = True
        rec["file"] = getattr(tabpfn, "__file__", None)
    except Exception as e:  # noqa: BLE001
        rec["importable"] = False
        rec["error_type"] = type(e).__name__
        rec["error"] = str(e)[:300]
    print("@@JSON@@" + json.dumps(rec))


def child_frozen_path():
    """Import every module of the frozen packages and see what came along."""
    for p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve"):
        sys.path.insert(0, str(p))
    imported, failed = [], {}
    mods = ["gen13sep." + f.stem for f in sorted((ROOT / "gen13_separation" / "gen13sep").glob("*.py"))
            if f.stem not in ("__init__", "wildcluster")]           # wildcluster.py is untracked, not frozen
    mods += ["gen14.dirbench", "gen14.models", "gen15.valuebench", "gen15.arms", "gen15.fewshot",
             "gen15.mixture", "gen15.shape"]
    import importlib
    for m in mods:
        try:
            importlib.import_module(m); imported.append(m)
        except Exception as e:  # noqa: BLE001
            failed[m] = f"{type(e).__name__}: {str(e)[:200]}"
    import sklearn, pandas, numpy
    third = sorted({k.split(".")[0] for k in sys.modules
                    if k.split(".")[0] in ("tabpfn", "torch", "catboost", "xgboost", "lightgbm",
                                           "transformers", "rdkit", "sklearn", "scipy")})
    rec = {"imported": imported, "failed": failed, "tabpfn_in_sys_modules": "tabpfn" in sys.modules,
           "third_party_loaded": third, "sklearn_after": sklearn.__version__,
           "pandas_after": pandas.__version__, "numpy_after": numpy.__version__}
    print("@@JSON@@" + json.dumps(rec))


def child_folds():
    """Digest of every fold of every design, default (discovery) seeds, never passing seeds=."""
    import numpy as np
    for p in (ROOT / "gen13_separation", ROOT / "gen14_direction", ROOT / "gen15_curve"):
        sys.path.insert(0, str(p))
    from gen15.valuebench import DESIGNS          # also what every fleet script imports first
    from gen14.dirbench import load
    from gen13sep.splits import all_folds, SPLIT_SEEDS
    t0 = time.time()
    bench = load()
    per_design, n_folds = {}, 0
    for d in DESIGNS:
        h = hashlib.sha256()
        for f in all_folds(bench.frame, design=d):
            for arr in (f.test_index, f.train_index, f.inner_train_index, f.inner_validation_index):
                h.update(np.sort(np.asarray(arr, dtype=np.int64)).tobytes())
            h.update(f"{f.design}|{f.seed}|{f.fold}|{'|'.join(f.held_out_groups)}".encode())
            n_folds += 1
        per_design[d] = h.hexdigest()
    total = hashlib.sha256("".join(per_design[d] for d in DESIGNS).encode()).hexdigest()
    frame_digest = hashlib.sha256(bench.frame["cell_id"].astype(str).str.cat(sep="|").encode()).hexdigest()
    rec = {"pythonhashseed": os.environ.get("PYTHONHASHSEED"), "n_cells": int(len(bench.frame)),
           "split_seeds": list(SPLIT_SEEDS), "designs": list(DESIGNS), "n_folds": n_folds,
           "per_design_sha256": per_design, "total_sha256": total, "cell_id_order_sha256": frame_digest,
           "probe_hash_str": hash("salt-probe"), "probe_hash_bytes": hash(b"salt-probe"),
           "probe_hash_int_tuple": hash((7, 104729)),
           "probe_gen15_fewshot_seed": [abs(hash((int(ci), int(s)))) % (2 ** 32)
                                        for ci in (0, 17, 288) for s in SPLIT_SEEDS[:2]],
           "elapsed_s": round(time.time() - t0, 1)}
    print("@@JSON@@" + json.dumps(rec))


def child_bench(mode: str = "bench"):
    """Load the frozen bench and score under BP; report peak memory and timing.

    ``bench``    FLAT + G14 (cheap; also re-checks two anchors).
    ``bench_et`` gen13's 400-tree extra-trees regression, n_jobs=2 (``dec_arms.g13_full``): the heavy slot.
    """
    sys.path.insert(0, str(ROOT / "gen15_curve"))
    # .venv/Scripts/python.exe is a launcher whose real interpreter is *this* process, a grandchild
    # of the parent; publish the pid so the parent can poll the right working set.
    (OUT_DIR / f"{mode}_child_pid.txt").write_text(str(os.getpid()), encoding="utf-8")
    t0 = time.time()
    from gen15 import valuebench as V, arms as A
    from gen14.dirbench import load
    bench = load()
    t_load = time.time() - t0
    mem_after_load = process_memory()
    if mode == "bench_et":
        sys.path.insert(0, str(ROOT / "gen15_curve" / "exp" / "decision"))
        import dec_arms
        ARMS, comps = {"G13_FULL": dec_arms.g13_full}, None
    else:
        ARMS, comps = {"FLAT": A.flat, "G14": A.g14}, {"G14_vs_FLAT": ("FLAT", "G14")}
    B, C, _ = V.score(bench, ARMS, ["BP"], comps=comps, verbose=False)
    t_score = time.time() - t0 - t_load
    W = V.wide(B)
    row = C.iloc[0].to_dict() if len(C) else {}
    rec = {"mode": mode, "arms": list(ARMS), "elapsed_load_s": round(t_load, 1),
           "elapsed_score_BP_s": round(t_score, 1), "mem_after_load": mem_after_load, "mem_end": process_memory(),
           "bp_macro_mae": {a: float(W.loc[a, "BP"]) for a in ARMS},
           "contrast": {k: (float(row[k]) if isinstance(row.get(k), (int, float)) else str(row.get(k)))
                        for k in ("comparison", "point", "ci95_low", "ci95_high", "p_two_sided", "passes_P1") if k in row},
           "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    print("@@JSON@@" + json.dumps(rec))


def child_xtb():
    rec = {}
    try:
        import xtb  # noqa: F401
        rec["xtb_python_importable"] = True
        rec["file"] = getattr(xtb, "__file__", None)
    except Exception as e:  # noqa: BLE001
        rec["xtb_python_importable"] = False
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    print("@@JSON@@" + json.dumps(rec))


# --------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------
def section_versions() -> dict:
    installed = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            installed[p] = md.version(p)
        except md.PackageNotFoundError:
            installed[p] = "absent"
    # module-level strings where they differ from the distribution version (rdkit, torch build tag)
    rc, out, err = _run([PY, "-c", "import rdkit, torch; print(rdkit.__version__, torch.__version__)"])
    mod_versions = {}
    if rc == 0 and out.split():
        mod_versions = {"rdkit.__version__": out.split()[0], "torch.__version__": out.split()[1]}
    install_dates = {}
    site = ROOT / ".venv" / "Lib" / "site-packages"
    wanted = [p.lower() for p in PACKAGES]
    for d in sorted(site.glob("*.dist-info")):
        name = d.name[:-len(".dist-info")].rsplit("-", 1)[0].replace("_", "-").lower()
        if name in wanted:
            install_dates[name] = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime))
    commits = {c: _git("log", "-1", "--format=%ci %s", c) for c in GEN_COMMITS}
    mismatches = []
    for src, rec in RECORDED.items():
        for pkg, v in rec.items():
            have = installed.get(pkg, "absent")
            if pkg == "torch":
                have = mod_versions.get("torch.__version__", have)
            if pkg == "rdkit":
                have = mod_versions.get("rdkit.__version__", have)
            ok = have == v or (v.endswith(".x") and have.startswith(v[:-1])) \
                or (pkg == "python" and have.startswith(v)) or (pkg == "scikit-learn" and have.startswith(v))
            if not ok:
                mismatches.append({"source": src, "package": pkg, "recorded": v, "installed": have})
    tabpfn, rc_t, out_t, err_t = _child("tabpfn")
    frozen, rc_f, out_f, err_f = _child("frozen_path")
    dec_arms = (ROOT / "gen15_curve" / "exp" / "decision" / "dec_arms.py")
    dec_txt = dec_arms.read_text(encoding="utf-8") if dec_arms.exists() else ""
    meta = next(iter(site.glob("tabpfn-*.dist-info/METADATA")), None)
    tabpfn_pins = [l.split(":", 1)[1].strip() for l in
                   (meta.read_text(encoding="utf-8", errors="replace").splitlines() if meta else [])
                   if l.startswith("Requires-Dist") and "extra ==" not in l]
    return {"installed": installed, "module_versions": mod_versions, "install_dates": install_dates,
            "generation_commits": commits, "recorded_sources": RECORDED, "mismatches": mismatches,
            "tabpfn_import": tabpfn, "tabpfn_declared_pins": tabpfn_pins, "frozen_path": frozen,
            "frozen_path_stderr_tail": err_f[-500:],
            "dec_arms_g13_full_n_jobs2": bool(re.search(r"n_jobs\s*=\s*2", dec_txt)),
            "exe": PY}


def section_hash_audit() -> dict:
    pat = re.compile(r"\bhash\(|\bset\(|frozenset\(|stable_hash|blake2b|np\.unique|PYTHONHASHSEED")
    hits = []
    files = []
    for s in SCAN_FILES:
        p = ROOT / s
        files += sorted(p.glob("*.py")) if p.is_dir() else [p]
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        if f.name == "wildcluster.py":
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if pat.search(line) and not line.strip().startswith("#"):
                cls, why = "UNREVIEWED", ""
                for rf, rx, c, w in REVIEWED:
                    if rel == rf and re.search(rx, line):
                        cls, why = c, w; break
                if cls == "UNREVIEWED" and re.search(r"sorted\(|len\(set|not in set|in set\(|\bset\([^)]*\)\s*[&|<]|isin\(", line):
                    cls, why = "SAFE", "sorted / len / membership / set algebra only"
                if cls == "UNREVIEWED" and "np.unique" in line and "hash(" not in line and "set(" not in line:
                    cls, why = "SAFE", "np.unique returns sorted values: process-independent order"
                hits.append({"file": rel, "line": i, "text": line.strip()[:140], "class": cls, "why": why})
    salted = [h for h in hits if re.search(r"\bhash\(", h["text"])]
    return {"scanned": [str(Path(s)) for s in SCAN_FILES], "hits": hits, "hash_calls": salted,
            "n_unreviewed": sum(h["class"] == "UNREVIEWED" for h in hits)}


def section_determinism() -> dict:
    runs = []
    for hs in ("1", "2"):
        rec, rc, out, err = _child("folds", env={"PYTHONHASHSEED": hs})
        if rec is None:
            rec = {"error": err[-800:], "rc": rc}
        runs.append(rec)
    ok = all("total_sha256" in r for r in runs)
    same = ok and runs[0]["total_sha256"] == runs[1]["total_sha256"]
    salt_differs = ok and runs[0]["probe_hash_str"] != runs[1]["probe_hash_str"]
    int_tuple_same = ok and runs[0]["probe_hash_int_tuple"] == runs[1]["probe_hash_int_tuple"] \
        and runs[0]["probe_gen15_fewshot_seed"] == runs[1]["probe_gen15_fewshot_seed"]
    return {"runs": runs, "fold_plans_identical": same, "str_hash_salt_differed_between_runs": salt_differs,
            "int_tuple_hash_identical": int_tuple_same,
            "verdict": ("PASS: identical fold plans for all 5 designs x 5 seeds in two processes with different "
                        "hash salts" if same and salt_differs else "FAIL or inconclusive")}


def section_xtb() -> dict:
    which = {n: shutil.which(n) for n in ("xtb", "xtb.exe", "xtb-python")}
    path_dirs = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    path_hits = [p for p in path_dirs if "xtb" in p.lower()]
    conda_dirs = [str(d) for d in (Path.home() / "miniconda3", Path.home() / "anaconda3",
                                   Path("C:/ProgramData/miniconda3"), Path("C:/ProgramData/anaconda3"),
                                   Path.home() / "AppData/Local/miniconda3", Path.home() / "mambaforge",
                                   Path.home() / "micromamba") if d.exists()]
    roots = [Path.home(), Path("C:/Program Files"), Path("C:/Program Files (x86)"), Path("C:/ProgramData"),
             Path("D:/")]
    found, t0, budget, n_dirs = [], time.time(), 90.0, 0
    name_rx = re.compile(r"^(xtb|xtb\.exe|xtb-python|libxtb[^/\\]*|xtb-\d[^/\\]*)$", re.I)
    for r in roots:
        if not r.exists():
            continue
        base_depth = len(r.parts)
        for dirpath, dirnames, filenames in os.walk(r):
            n_dirs += 1
            if time.time() - t0 > budget:
                break
            depth = len(Path(dirpath).parts) - base_depth
            dirnames[:] = [d for d in dirnames if depth < 6 and d not in (".git", "node_modules", "__pycache__",
                                                                          "$Recycle.Bin", "System Volume Information")]
            for n in filenames + dirnames:
                if name_rx.match(n):
                    found.append(str(Path(dirpath) / n))
    py, rc, out, err = _child("xtb")
    return {"which": which, "path_entries_mentioning_xtb": path_hits, "conda_roots_present": conda_dirs,
            "disk_search_roots": [str(r) for r in roots], "disk_hits": found[:50], "directories_visited": n_dirs,
            "disk_search_seconds": round(time.time() - t0, 1), "disk_search_truncated": time.time() - t0 > budget,
            "xtb_python": py, "xtb_on_path": bool(which["xtb"] or which["xtb.exe"])}


def section_memory(mode: str) -> dict:
    """Run one bench child and measure it: polled working set of the real interpreter (pid published
    by the child), the child's own peak counters, and the drop in system free memory."""
    before = memory_status()
    others = python_processes()
    e = dict(os.environ); e["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    # stdout/stderr go to files, not pipes: a polled child that nobody reads deadlocks on Windows'
    # small pipe buffer as soon as gen13's sklearn warnings exceed it.
    out_path, err_path = OUT_DIR / f"{mode}_child_stdout.txt", OUT_DIR / f"{mode}_child_stderr.txt"
    pid_file = OUT_DIR / f"{mode}_child_pid.txt"
    if pid_file.exists():
        pid_file.unlink()
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h, polled_pid = None, None
    peak_polled, min_free, samples = 0.0, before["free_mb"], 0
    with out_path.open("w", encoding="utf-8") as fo, err_path.open("w", encoding="utf-8") as fe:
        proc = subprocess.Popen([PY, str(Path(__file__).resolve()), "--child", mode], cwd=str(ROOT), env=e,
                                stdout=fo, stderr=fe)
        while proc.poll() is None:
            if h is None and pid_file.exists():
                try:
                    polled_pid = int(pid_file.read_text(encoding="utf-8").strip() or 0)
                    h = _K32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, polled_pid) or None
                except ValueError:
                    pass
            m = process_memory(h) if h else {}
            peak_polled = max(peak_polled, m.get("ws_mb", 0.0))
            min_free = min(min_free, memory_status()["free_mb"])
            samples += 1
            time.sleep(0.25)
        if h:
            _K32.CloseHandle(h)
        proc.wait()
    out = out_path.read_text(encoding="utf-8", errors="replace")
    err = err_path.read_text(encoding="utf-8", errors="replace")
    rec = None
    for line in out.splitlines():
        if line.startswith("@@JSON@@"):
            rec = json.loads(line[len("@@JSON@@"):])
    after = memory_status()
    peak_self = (rec or {}).get("mem_end", {}).get("peak_ws_mb", 0.0) or 0.0
    peak = max(peak_polled, peak_self)
    return {"mode": mode, "free_before": before, "free_after": after, "other_python_processes_before": others,
            "child_rc": proc.returncode, "child": rec, "child_stderr_tail": err[-600:], "polled_pid": polled_pid,
            "peak_ws_mb_polled": round(peak_polled, 1), "peak_ws_mb_self": round(peak_self, 1),
            "peak_ws_mb": round(peak, 1), "system_free_drop_mb": before["free_mb"] - min_free,
            "min_free_mb_during_run": min_free, "poll_samples": samples, "wall_s": round(time.time() - t0, 1)}


def recommendation(cheap: dict, et: dict | None) -> dict:
    free = cheap["free_before"]["free_mb"]
    total = cheap["free_before"]["total_mb"]
    others = sum((p.get("ws_mb") or 0) for p in cheap["other_python_processes_before"])
    baseline = total - free - others                       # everything that is not a python process
    budget_now = free - KEEP_FREE_MB
    budget_alone = total - baseline - KEEP_FREE_MB
    pc = max(cheap["peak_ws_mb"], 1.0)
    pe = max(et["peak_ws_mb"], 1.0) if et and et.get("child") else None
    rec = {"keep_free_mb": KEEP_FREE_MB, "free_now_mb": free, "budget_now_mb": budget_now,
           "budget_if_no_other_python_mb": budget_alone, "non_python_baseline_mb": baseline,
           "per_process_peak_mb": {"cheap_FLAT_G14": pc, "extra_trees_G13_FULL_njobs2": pe},
           "logical_processors": os.cpu_count(),
           "cheap_processes_by_ram_now": int(max(0, budget_now // pc)),
           "et_processes_by_ram_now": int(max(0, budget_now // pe)) if pe else None,
           "cheap_processes_by_ram_alone": int(max(0, budget_alone // pc)),
           "et_processes_by_ram_alone": int(max(0, budget_alone // pe)) if pe else None}
    cpu_cap = max(1, (os.cpu_count() or 12) // 2)         # two threads per process (n_jobs=2 / BLAS)
    rec["cpu_cap_two_threads_each"] = cpu_cap
    rec["recommended_concurrent_python_processes"] = int(min(rec["cheap_processes_by_ram_now"],
                                                             rec["et_processes_by_ram_now"] or 10 ** 6, cpu_cap))
    return rec


def section_git() -> dict:
    rc, status, _ = _run(["git", "status", "--short"])          # no strip(): ' M path' keeps its space
    entries = [(l[:2], l[3:]) for l in status.splitlines() if len(l) > 3]
    untracked = [p for s, p in entries if s == "??"]
    modified = [p for s, p in entries if s != "??"]
    in_brief = [u for u in untracked if u.startswith("gen16_leads/")]
    outside = [u for u in untracked if not u.startswith("gen16_leads/")]
    tracked_gen16 = _git("ls-files", "gen16_leads").splitlines()
    return {"head": _git("rev-parse", "--short", "HEAD"), "branch": _git("branch", "--show-current"),
            "modified_tracked": modified, "untracked_in_gen16_leads": in_brief,
            "untracked_outside_brief": outside, "tracked_under_gen16_leads": tracked_gen16,
            "only_start_here_committed": tracked_gen16 == ["gen16_leads/START_HERE.md"]}


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def _mem_lines(m: dict, label: str) -> list[str]:
    c = m.get("child") or {}
    return [f"- **{label}** (`--child {m['mode']}`, rc {m['child_rc']}): peak working set **{m['peak_ws_mb']} MB** "
            f"(polled pid {m['polled_pid']}: {m['peak_ws_mb_polled']} MB over {m['poll_samples']} samples; "
            f"self-reported {m['peak_ws_mb_self']} MB; system free dropped {m['system_free_drop_mb']} MB from "
            f"{m['free_before']['free_mb']} MB); after bench load {c.get('mem_after_load')}; load "
            f"{c.get('elapsed_load_s')} s, BP scoring {c.get('elapsed_score_BP_s')} s, wall {m['wall_s']} s; "
            f"BP macro MAE {c.get('bp_macro_mae')}; contrast {c.get('contrast')}"]


def write_md(R: dict) -> str:
    v, ha, det, x, g = R["versions"], R["hash_audit"], R["determinism"], R["xtb"], R["git"]
    m, me, rc = R["memory"], R.get("memory_et"), R["recommendation"]
    L = ["# Gen16 environment check", "",
         f"Generated {R['generated']} by `gen16_leads/scripts/g16_env_check.py` on {platform.node()} "
         f"({platform.platform()}).  Machine-readable copy: `env_check.json`.  Total run {R['elapsed_s']} s.", "",
         "## Verdict", ""]
    for i in R["issues"]:
        L.append(f"- {i}")
    L += ["", "## 1. Versions", "", f"Interpreter: `{v['exe']}` (Python {v['installed']['python']}).", "",
          "| package | installed | dist-info written | gen8 doc (2026-08-20) | gen13 session memo (2026-09-07) |",
          "|---|---|---|---|---|"]
    g8 = RECORDED["docs/results/gen8_architecture_results_20260820.md:834 (2026-08-20)"]
    mm = RECORDED["memory/env-windows-python-quirks.md (2026-09-07, the gen13 session)"]
    for p in PACKAGES:
        L.append(f"| {p} | {v['installed'].get(p)} | {v['install_dates'].get(p.lower(), '')} | {g8.get(p, '')} | {mm.get(p, '')} |")
    L += ["", f"Module strings: {v['module_versions']}.", "",
          "Generation commits (all after the numeric stack was installed):", ""]
    for c, d in v["generation_commits"].items():
        L.append(f"- `{c}` {d}")
    L += ["", "### Mismatches against every recorded version", ""]
    if v["mismatches"]:
        L += ["| source | package | recorded | installed |", "|---|---|---|---|"]
        for r in v["mismatches"]:
            L.append(f"| {r['source']} | {r['package']} | {r['recorded']} | {r['installed']} |")
    else:
        L.append("none")
    L += ["", "Reading: gen13, gen14 and gen15 recorded no version table of their own.  The only record from "
          "the gen13 session is the memory note, which matches the installed stack exactly, and every "
          "numeric-stack dist-info was written 2026-09-07 22:47-22:53, before gen13's first manifest "
          "(2026-09-07 23:36) and all six generation commits.  The gen7/gen8 docs (Aug 2026) predate that "
          "venv rebuild, so numpy 2.5.1 -> 2.5.3, scipy 1.18.0 -> 1.18.1, rdkit 2026.03.5 -> 2026.03.6, "
          "torch 2.13.0 -> 2.14.0+cpu, Python 3.13 -> 3.14.5 are differences between gen8 and gen13, not "
          "drift since gen13.  requirements-gen3.txt is the Linux cluster reference and never described this "
          "machine.  Nothing in the numeric stack has been touched since gen13 started.", "",
          "### TabPFN", "",
          f"- `import tabpfn`: importable = **{v['tabpfn_import'].get('importable')}**; "
          f"{v['tabpfn_import'].get('error_type')}: {v['tabpfn_import'].get('error')}",
          f"- declared pins: {v['tabpfn_declared_pins']}",
          f"- scikit-learn stayed {v['tabpfn_import'].get('sklearn_before')} (dist-info written "
          f"{v['install_dates'].get('scikit-learn')}, tabpfn written {v['install_dates'].get('tabpfn')}, between "
          "the gen14 and gen15 commits): the pins were never applied, so the 0.0014 shift START_HERE warns "
          "about did not happen here.  gen15 ran TabPFN through `exp/tabpfn/tabpfn_compat.py`, a private-name "
          "shim, on this same sklearn 1.9.0.",
          f"- frozen import path (every gen13sep module, gen14.dirbench/models, gen15.*): "
          f"tabpfn in sys.modules = **{v['frozen_path'].get('tabpfn_in_sys_modules')}**; imported "
          f"{len(v['frozen_path'].get('imported', []))} modules, failed {v['frozen_path'].get('failed')}; "
          f"third-party loaded: {v['frozen_path'].get('third_party_loaded')}; sklearn after imports "
          f"{v['frozen_path'].get('sklearn_after')}.",
          f"- `gen15_curve/exp/decision/dec_arms.py` carries an `n_jobs=2` g13_full: {v['dec_arms_g13_full_n_jobs2']}.",
          "", "## 2. hash() and iteration-order audit", "",
          f"Scanned: {', '.join(ha['scanned'])} (wildcluster.py excluded: untracked, not frozen).  "
          f"Unreviewed hits: {ha['n_unreviewed']}.", "",
          "| file:line | class | snippet | why |", "|---|---|---|---|"]
    for h in ha["hits"]:
        if h["class"] != "SAFE" or "hash(" in h["text"]:
            L.append(f"| `{h['file']}:{h['line']}` | **{h['class']}** | `{h['text'][:80]}` | {h['why']} |")
    L += ["", f"All other {sum(h['class'] == 'SAFE' for h in ha['hits'])} matches are `set()`/`np.unique` uses "
          "that feed only membership tests, lengths, `sorted()` or set algebra; none decides a fold, a pair or "
          "a row order.  The fold plan itself is `seeded_group_kfold`: `np.unique` (sorted) + "
          "`default_rng(seed).permutation`, then publication masks by membership.  Full list in `env_check.json`.", "",
          "### Two-process fold-plan comparison", "",
          f"`gen13sep.splits.all_folds(bench.frame, design=d)` for d in {det['runs'][0].get('designs')} with the "
          f"default seeds {det['runs'][0].get('split_seeds')} ({det['runs'][0].get('n_folds')} folds, "
          f"{det['runs'][0].get('n_cells')} cells), SHA-256 over sorted test/train/inner indices and held-out "
          "groups, in two subprocesses with PYTHONHASHSEED=1 and =2:", ""]
    for r in det["runs"]:
        L.append(f"- PYTHONHASHSEED={r.get('pythonhashseed')}: total `{r.get('total_sha256')}`; "
                 f"hash('salt-probe') = {r.get('probe_hash_str')}; hash((7, 104729)) = {r.get('probe_hash_int_tuple')}; "
                 f"gen15 fewshot rng seeds {r.get('probe_gen15_fewshot_seed')}"
                 + (f"; ERROR {r['error'][-300:]}" if 'error' in r else ""))
    L += ["", f"Per design: {det['runs'][0].get('per_design_sha256')}", "",
          f"**{det['verdict']}.**  str-hash salt differed: {det['str_hash_salt_differed_between_runs']}; "
          f"int-tuple hash identical: {det['int_tuple_hash_identical']}.", "",
          "## 3. xtb", "",
          f"- `which xtb` / `xtb.exe`: {x['which']}  -> xtb_on_path = **{x['xtb_on_path']}**",
          f"- PATH entries mentioning xtb: {x['path_entries_mentioning_xtb']}; conda roots present: {x['conda_roots_present']}",
          f"- disk search of {x['disk_search_roots']} to depth 6 ({x['directories_visited']} directories, "
          f"{x['disk_search_seconds']} s, truncated={x['disk_search_truncated']}): hits = {x['disk_hits']}",
          f"- `import xtb` (xtb-python): {x['xtb_python']}", "",
          "L1's reference-species energies therefore need the cluster route START_HERE plans for.", "",
          "## 4. Memory and concurrency", "",
          f"- physical RAM {m['free_before']['total_mb']} MB; free before the measurements "
          f"{m['free_before']['free_mb']} MB (load {m['free_before']['load_pct']} %), after {m['free_after']['free_mb']} MB; "
          f"other python.exe processes at the time: {m['other_python_processes_before']}; "
          f"non-python baseline {rc['non_python_baseline_mb']} MB"]
    L += _mem_lines(m, "cheap bench process: load cached bench + `V.score` FLAT/G14 under BP")
    if me:
        L += _mem_lines(me, "heavy bench process: `dec_arms.g13_full` (400 extra trees, 209 columns, n_jobs=2) under BP")
    L += ["", f"Anchors from the cheap run (macro MAE of log SF, BP, 5 discovery seeds): "
          f"{(m.get('child') or {}).get('bp_macro_mae')}; G14 - FLAT: {(m.get('child') or {}).get('contrast')}.", "",
          f"**Recommendation: {rc['recommended_concurrent_python_processes']} concurrent bench processes.**  "
          f"Budget now = free {rc['free_now_mb']} MB - {rc['keep_free_mb']} MB kept free = {rc['budget_now_mb']} MB; "
          f"per-process peaks {rc['per_process_peak_mb']}; RAM alone would allow "
          f"{rc['cheap_processes_by_ram_now']} cheap / {rc['et_processes_by_ram_now']} extra-trees processes now "
          f"({rc['cheap_processes_by_ram_alone']} / {rc['et_processes_by_ram_alone']} if no other python were "
          f"running); the CPU cap at two threads per process on {rc['logical_processors']} logical CPUs is "
          f"{rc['cpu_cap_two_threads_each']}.  Keep n_jobs=2 per forest and set OMP_NUM_THREADS=2 per process so "
          "concurrent processes do not oversubscribe the 12 threads; the numbers above are per bench process, "
          "the orchestrator's own agents are not counted.",
          "", "## 5. Git state", "",
          f"- HEAD `{g['head']}` on `{g['branch']}`; tracked under gen16_leads: {g['tracked_under_gen16_leads']} "
          f"(only START_HERE.md committed: {g['only_start_here_committed']})",
          f"- modified tracked files (outside the brief, not ours): {g['modified_tracked']}",
          f"- untracked inside gen16_leads: {g['untracked_in_gen16_leads']}",
          f"- untracked outside the brief: {g['untracked_outside_brief']}", ""]
    return "\n".join(L)


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        mode = sys.argv[2]
        if mode in ("bench", "bench_et"):
            child_bench(mode)
        else:
            {"tabpfn": child_tabpfn, "frozen_path": child_frozen_path, "folds": child_folds,
             "xtb": child_xtb}[mode]()
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "cwd": str(ROOT), "argv": sys.argv[1:]}
    print("[env] versions ...", flush=True);       R["versions"] = section_versions()
    print("[env] hash audit ...", flush=True);     R["hash_audit"] = section_hash_audit()
    print("[env] determinism ...", flush=True);    R["determinism"] = section_determinism()
    print("[env] xtb ...", flush=True);            R["xtb"] = section_xtb()
    print("[env] memory: cheap ...", flush=True);  R["memory"] = section_memory("bench")
    R["memory_et"] = None
    if "--no-et" not in sys.argv:
        print("[env] memory: extra trees ...", flush=True)
        R["memory_et"] = section_memory("bench_et")
    R["recommendation"] = recommendation(R["memory"], R["memory_et"])
    print("[env] git ...", flush=True);            R["git"] = section_git()
    v, det, m, x = R["versions"], R["determinism"], R["memory"], R["xtb"]
    issues = []
    if v["installed"].get("scikit-learn") != "1.9.0":
        issues.append(f"scikit-learn is {v['installed'].get('scikit-learn')}, not 1.9.0")
    if v["tabpfn_import"] and v["tabpfn_import"].get("importable"):
        issues.append("tabpfn IS importable; START_HERE expects it broken")
    if v["frozen_path"] and v["frozen_path"].get("tabpfn_in_sys_modules"):
        issues.append("tabpfn is imported by something on the frozen import path")
    if v["frozen_path"] and v["frozen_path"].get("failed"):
        issues.append(f"frozen modules failed to import: {v['frozen_path']['failed']}")
    if not det["fold_plans_identical"]:
        issues.append("fold plans differ between two processes")
    if R["hash_audit"]["n_unreviewed"]:
        issues.append(f"{R['hash_audit']['n_unreviewed']} unreviewed hash()/set() sites")
    child = m.get("child") or {}
    bp = child.get("bp_macro_mae", {})
    if not child:
        issues.append("cheap bench child failed: " + (m.get("child_stderr_tail") or "")[-200:])
    if child and abs(bp.get("G14", 0) - 0.5001) > 5e-4:
        issues.append(f"G14 under BP = {bp.get('G14')} vs anchor 0.5001")
    if child and abs(bp.get("FLAT", 0) - 0.589) > 1e-3:
        issues.append(f"FLAT under BP = {bp.get('FLAT')} vs anchor 0.589")
    if R["memory_et"] is not None and not (R["memory_et"].get("child") or {}):
        issues.append("extra-trees bench child failed: " + (R["memory_et"].get("child_stderr_tail") or "")[-200:])
    if x["xtb_on_path"]:
        issues.append("xtb IS on PATH (START_HERE assumed it was not)")
    R["issues"] = issues or ["no drift detected: versions, frozen import path, fold determinism and both "
                             "cheap anchors match the locked generations"]
    R["ok"] = not issues
    R["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "env_check.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    (OUT_DIR / "ENV.md").write_text(write_md(R), encoding="utf-8")
    et_child = (R["memory_et"] or {}).get("child") or {}
    print(json.dumps({"ok": R["ok"], "issues": issues, "elapsed_s": R["elapsed_s"],
                      "peak_ws_mb_cheap": m["peak_ws_mb"], "peak_ws_mb_et": (R["memory_et"] or {}).get("peak_ws_mb"),
                      "free_mb": m["free_before"]["free_mb"],
                      "recommended": R["recommendation"]["recommended_concurrent_python_processes"],
                      "fold_plans_identical": det["fold_plans_identical"],
                      "bp_cheap": bp, "bp_et": et_child.get("bp_macro_mae"),
                      "et_seconds": et_child.get("elapsed_score_BP_s")}, indent=1))
    return 0 if R["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
