#!/usr/bin/env python
"""Stamp the gen10 run: environment, git state, and a BLAKE2b digest of every artefact.

Same contract as ``gen9_manifest.py``: ``--verify`` recomputes every digest and
reports drift, exiting non-zero on any change or loss.  Records, in addition, the
gen9 manifest's own verification result at stamping time, so the chain of custody
from gen9's artefacts to gen10's tables is one document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RUN = REPO_ROOT / "runs" / "gen10_final"
HASHED = {".csv", ".parquet", ".json", ".md", ".py"}
LISTED = {".png", ".pdf", ".log", ".txt"}


def digest(path: Path, *, chunk: int = 1 << 20) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def environment() -> dict:
    import numpy, pandas, scipy, sklearn

    versions = {"python": sys.version.split()[0], "numpy": numpy.__version__,
                "pandas": pandas.__version__, "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__}
    for name in ("torch", "rdkit", "matplotlib"):
        try:
            versions[name] = __import__(name).__version__
        except Exception as error:                        # pragma: no cover
            versions[name] = f"unavailable ({error})"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                             text=True).strip())
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                         cwd=REPO_ROOT, text=True).strip()
    except Exception:                                     # pragma: no cover
        commit, dirty, branch = "unknown", None, "unknown"
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
            "machine": platform.machine(), "versions": versions,
            "git": {"commit": commit, "branch": branch, "dirty": dirty}}


def collect(run: Path, *, sources: bool = True) -> dict:
    artefacts: dict = {}
    paths = sorted(p for p in run.rglob("*") if p.is_file())
    if sources:
        paths += sorted((REPO_ROOT / "src" / "lanthanide_separation" / "gen10").glob("*.py"))
        paths += sorted((REPO_ROOT / "scripts").glob("gen10_*.py"))
        paths += sorted((REPO_ROOT / "tests").glob("test_gen10_*.py"))
    for path in paths:
        if path.suffix not in HASHED | LISTED or path.name == "manifest.json":
            continue
        relative = str(path.relative_to(REPO_ROOT))
        entry = {"bytes": path.stat().st_size,
                 "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
        if path.suffix in HASHED:
            entry["blake2b_128"] = digest(path)
        artefacts[relative] = entry
    return artefacts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.run / "manifest.json"
    artefacts = collect(args.run)

    if args.verify:
        if not manifest_path.exists():
            raise SystemExit(f"{manifest_path} does not exist; nothing to verify against")
        stored = json.loads(manifest_path.read_text())["artefacts"]
        changed, missing = [], []
        for name, entry in stored.items():
            if name not in artefacts:
                missing.append(name)
            elif "blake2b_128" in entry and entry["blake2b_128"] != artefacts[name].get("blake2b_128"):
                changed.append(name)
        added = [n for n in artefacts if n not in stored]
        print(f"{len(stored)} recorded, {len(artefacts)} on disk: "
              f"{len(changed)} changed, {len(missing)} missing, {len(added)} new")
        for name in changed:
            print(f"  changed: {name}")
        for name in missing:
            print(f"  missing: {name}")
        return 1 if (changed or missing) else 0

    gen9 = subprocess.run([sys.executable, "scripts/gen9_manifest.py", "--verify"], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    record = {
        "run": str(args.run.relative_to(REPO_ROOT)),
        "environment": environment(),
        "cohort_path": "runs/gen7_architecture/cache/cohort.parquet",
        "cohort_fingerprint": "bed178ec1a7a82b0",
        "split_seeds": [104729, 130363, 155921, 196613, 262147],
        "model_seed_formula": "42 + fold*1009 + 9999991",
        "gen9_manifest_verification": {"returncode": gen9.returncode,
                                       "output": gen9.stdout.strip().splitlines()[-1:]},
        "n_artefacts": len(artefacts),
        "artefacts": artefacts,
    }
    for name, path in (("phase0", args.run / "reproduction" / "phase0.json"),
                       ("pipeline", args.run / "final_locked" / "pipeline_frontier.json"),
                       ("self_audit", args.run / "self_audit" / "self_audit.json")):
        if path.exists():
            record[name] = json.loads(path.read_text())
    manifest_path.write_text(json.dumps(record, indent=2, default=str))
    print(f"wrote {manifest_path} ({len(artefacts)} artefacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
