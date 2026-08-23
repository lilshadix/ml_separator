#!/usr/bin/env python
"""Stamp the run: environment, git state, and a content hash of every artefact.

The point is falsifiability of the *run*, not of the science: a reader should be
able to tell, without trusting anyone, whether the tables in the report are the
tables the scripts produced, and whether the environment they were produced in is
the one gen8's numbers reproduced in.

Records, per artefact: path, size, modification time, and a BLAKE2b content digest.
Records, per run: interpreter and library versions, platform, git commit and
dirty/clean state, the cohort fingerprint, the frozen model, the split seeds, the
model-seed formula, and the selected booster configuration with the protocol that
selected it.
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

RUN = REPO_ROOT / "runs" / "gen9_shape"
#: Extensions worth hashing.  Figures and logs are listed but not hashed — a PNG's
#: bytes move with the matplotlib patch version and would produce spurious diffs.
HASHED = {".csv", ".parquet", ".json", ".md"}
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
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                         text=True)
    except Exception:                                     # pragma: no cover
        commit, status = "unknown", "?"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "library_versions": versions, "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "n_dirty_paths": len([line for line in status.splitlines() if line.strip()]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--verify", action="store_true",
                        help="recompute digests and report drift against the stored manifest")
    args = parser.parse_args(argv)

    manifest_path = args.run / "manifest.json"
    artefacts = {}
    for path in sorted(args.run.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        suffix = path.suffix.lower()
        if suffix not in HASHED and suffix not in LISTED:
            continue
        relative = str(path.relative_to(args.run))
        entry = {"bytes": path.stat().st_size,
                 "modified_utc": datetime.fromtimestamp(
                     path.stat().st_mtime, timezone.utc).isoformat()}
        if suffix in HASHED:
            entry["blake2b_128"] = digest(path)
        artefacts[relative] = entry

    if args.verify:
        if not manifest_path.exists():
            raise SystemExit(f"{manifest_path} does not exist; nothing to verify against")
        stored = json.loads(manifest_path.read_text())["artefacts"]
        changed, missing, added = [], [], []
        for name, entry in stored.items():
            if name not in artefacts:
                missing.append(name)
            elif ("blake2b_128" in entry
                  and entry["blake2b_128"] != artefacts[name].get("blake2b_128")):
                changed.append(name)
        added = [n for n in artefacts if n not in stored]
        print(f"{len(stored)} recorded, {len(artefacts)} on disk: "
              f"{len(changed)} changed, {len(missing)} missing, {len(added)} new")
        for name in changed:
            print(f"  changed: {name}")
        for name in missing:
            print(f"  missing: {name}")
        return 1 if (changed or missing) else 0

    record: dict = {
        "run": str(args.run.relative_to(REPO_ROOT)),
        "environment": environment(),
        "cohort_path": "runs/gen7_architecture/cache/cohort.parquet",
        "cohort_fingerprint": "bed178ec1a7a82b0",
        "frozen_global_model": "REC_ecfp_plus_recovered",
        "frozen_oof": "runs/gen7_architecture/finalists/oof_predictions.parquet",
        "split_seeds": [104729, 130363, 155921, 196613, 262147],
        "model_seed_formula": "42 + fold*1009 + 9999991",
        "n_artefacts": len(artefacts),
        "artefacts": artefacts,
    }
    for name, path in (("boost_config", args.run / "boost_config.json"),
                       ("reproduction", args.run / "reproduction" / "reproduction.json"),
                       ("curve_audit", args.run / "curves" / "curve_audit.json"),
                       ("data_audit", args.run / "data_audit" / "summary.json")):
        if path.exists():
            record[name] = json.loads(path.read_text())
    manifest_path.write_text(json.dumps(record, indent=2, default=str))
    print(f"wrote {manifest_path} ({len(artefacts)} artefacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
